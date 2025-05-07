# Graph Drawing for LLMs: An Empirical Evaluation 

**Authors**: Walter Didimo, Fabrizio Montecchiani, Tommaso Piselli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03678)  

**Abstract**: Our work contributes to the fast-growing literature on the use of Large Language Models (LLMs) to perform graph-related tasks. In particular, we focus on usage scenarios that rely on the visual modality, feeding the model with a drawing of the graph under analysis. We investigate how the model's performance is affected by the chosen layout paradigm, the aesthetics of the drawing, and the prompting technique used for the queries. We formulate three corresponding research questions and present the results of a thorough experimental analysis. Our findings reveal that choosing the right layout paradigm and optimizing the readability of the input drawing from a human perspective can significantly improve the performance of the model on the given task. Moreover, selecting the most effective prompting technique is a challenging yet crucial task for achieving optimal performance. 

---
# Gap the (Theory of) Mind: Sharing Beliefs About Teammates' Goals Boosts Collaboration Perception, Not Performance 

**Authors**: Yotam Amitai, Reuth Mirsky, Ofra Amir  

**Link**: [PDF](https://arxiv.org/pdf/2505.03674)  

**Abstract**: In human-agent teams, openly sharing goals is often assumed to enhance planning, collaboration, and effectiveness. However, direct communication of these goals is not always feasible, requiring teammates to infer their partner's intentions through actions. Building on this, we investigate whether an AI agent's ability to share its inferred understanding of a human teammate's goals can improve task performance and perceived collaboration. Through an experiment comparing three conditions-no recognition (NR), viable goals (VG), and viable goals on-demand (VGod) - we find that while goal-sharing information did not yield significant improvements in task performance or overall satisfaction scores, thematic analysis suggests that it supported strategic adaptations and subjective perceptions of collaboration. Cognitive load assessments revealed no additional burden across conditions, highlighting the challenge of balancing informativeness and simplicity in human-agent interactions. These findings highlight the nuanced trade-off of goal-sharing: while it fosters trust and enhances perceived collaboration, it can occasionally hinder objective performance gains. 

---
# Learning Symbolic Persistent Macro-Actions for POMDP Solving Over Time 

**Authors**: Celeste Veronese, Daniele Meli, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03668)  

**Abstract**: This paper proposes an integration of temporal logical reasoning and Partially Observable Markov Decision Processes (POMDPs) to achieve interpretable decision-making under uncertainty with macro-actions. Our method leverages a fragment of Linear Temporal Logic (LTL) based on Event Calculus (EC) to generate \emph{persistent} (i.e., constant) macro-actions, which guide Monte Carlo Tree Search (MCTS)-based POMDP solvers over a time horizon, significantly reducing inference time while ensuring robust performance. Such macro-actions are learnt via Inductive Logic Programming (ILP) from a few traces of execution (belief-action pairs), thus eliminating the need for manually designed heuristics and requiring only the specification of the POMDP transition model. In the Pocman and Rocksample benchmark scenarios, our learned macro-actions demonstrate increased expressiveness and generality when compared to time-independent heuristics, indeed offering substantial computational efficiency improvements. 

---
# BURNS: Backward Underapproximate Reachability for Neural-Feedback-Loop Systems 

**Authors**: Chelsea Sidrane, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2505.03643)  

**Abstract**: Learning-enabled planning and control algorithms are increasingly popular, but they often lack rigorous guarantees of performance or safety. We introduce an algorithm for computing underapproximate backward reachable sets of nonlinear discrete time neural feedback loops. We then use the backward reachable sets to check goal-reaching properties. Our algorithm is based on overapproximating the system dynamics function to enable computation of underapproximate backward reachable sets through solutions of mixed-integer linear programs. We rigorously analyze the soundness of our algorithm and demonstrate it on a numerical example. Our work expands the class of properties that can be verified for learning-enabled systems. 

---
# Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability 

**Authors**: Chen Wei, Chi Zhang, Jiachen Zou, Haotian Deng, Dietmar Heinke, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03641)  

**Abstract**: Human decision-making in cognitive tasks and daily life exhibits considerable variability, shaped by factors such as task difficulty, individual preferences, and personal experiences. Understanding this variability across individuals is essential for uncovering the perceptual and decision-making mechanisms that humans rely on when faced with uncertainty and ambiguity. We present a computational framework BAM (Boundary Alignment & Manipulation framework) that combines perceptual boundary sampling in ANNs and human behavioral experiments to systematically investigate this phenomenon. Our perceptual boundary sampling algorithm generates stimuli along ANN decision boundaries that intrinsically induce significant perceptual variability. The efficacy of these stimuli is empirically validated through large-scale behavioral experiments involving 246 participants across 116,715 trials, culminating in the variMNIST dataset containing 19,943 systematically annotated images. Through personalized model alignment and adversarial generation, we establish a reliable method for simultaneously predicting and manipulating the divergent perceptual decisions of pairs of participants. This work bridges the gap between computational models and human individual difference research, providing new tools for personalized perception analysis. 

---
# OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents 

**Authors**: Mariya Davydova, Daniel Jeffries, Patrick Barker, Arturo Márquez Flores, Sinéad Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03570)  

**Abstract**: In this paper, we introduce OSUniverse: a benchmark of complex, multimodal desktop-oriented tasks for advanced GUI-navigation AI agents that focuses on ease of use, extensibility, comprehensive coverage of test cases, and automated validation. We divide the tasks in increasing levels of complexity, from basic precision clicking to multistep, multiapplication tests requiring dexterity, precision, and clear thinking from the agent. In version one of the benchmark, presented here, we have calibrated the complexity of the benchmark test cases to ensure that the SOTA (State of the Art) agents (at the time of publication) do not achieve results higher than 50%, while the average white collar worker can perform all these tasks with perfect accuracy. The benchmark can be scored manually, but we also introduce an automated validation mechanism that has an average error rate less than 2%. Therefore, this benchmark presents solid ground for fully automated measuring of progress, capabilities and the effectiveness of GUI-navigation AI agents over the short and medium-term horizon. The source code of the benchmark is available at this https URL. 

---
# A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning 

**Authors**: Kolawole E. Ogunsina, Morayo A. Ogunsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.03553)  

**Abstract**: Inconsistent outputs and hallucinations from large language models (LLMs) are major obstacles to reliable AI systems. When different proprietary reasoning models (RMs), such as those by OpenAI, Google, Anthropic, DeepSeek, and xAI, are given the same complex request, they often produce divergent results due to variations in training and inference. This paper proposes a novel consensus mechanism, inspired by distributed ledger technology, to validate and converge these outputs, treating each RM as a black-box peer. Building on the Hashgraph consensus algorithm, our approach employs gossip-about-gossip communication and virtual voting to achieve agreement among an ensemble of RMs. We present an architectural design for a prototype system in which RMs iteratively exchange and update their answers, using information from each round to improve accuracy and confidence in subsequent rounds. This approach goes beyond simple majority voting by incorporating the knowledge and cross-verification content of every model. We justify the feasibility of this Hashgraph-inspired consensus for AI ensembles and outline its advantages over traditional ensembling techniques in reducing nonfactual outputs. Preliminary considerations for implementation, evaluation criteria for convergence and accuracy, and potential challenges are discussed. The proposed mechanism demonstrates a promising direction for multi-agent AI systems to self-validate and deliver high-fidelity responses in complex tasks. 

---
# STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game 

**Authors**: Eric Zhou, Shreyas Basavatia, Moontashir Siam, Zexin Chen, Mark O. Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2505.03547)  

**Abstract**: We introduce STORY2GAME, a novel approach to using Large Language Models to generate text-based interactive fiction games that starts by generating a story, populates the world, and builds the code for actions in a game engine that enables the story to play out interactively. Whereas a given set of hard-coded actions can artificially constrain story generation, the ability to generate actions means the story generation process can be more open-ended but still allow for experiences that are grounded in a game state. The key to successful action generation is to use LLM-generated preconditions and effects of actions in the stories as guides for what aspects of the game state must be tracked and changed by the game engine when a player performs an action. We also introduce a technique for dynamically generating new actions to accommodate the player's desire to perform actions that they think of that are not part of the story. Dynamic action generation may require on-the-fly updates to the game engine's state representation and revision of previously generated actions. We evaluate the success rate of action code generation with respect to whether a player can interactively play through the entire generated story. 

---
# am-ELO: A Stable Framework for Arena-based LLM Evaluation 

**Authors**: Zirui Liu, Jiatong Li, Yan Zhuang, Qi Liu, Shuanghong Shen, Jie Ouyang, Mingyue Cheng, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03475)  

**Abstract**: Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs. 

---
# The Steganographic Potentials of Language Models 

**Authors**: Artem Karpov, Tinuade Adeleke, Seong Hah Cho, Natalia Perez-Campanero  

**Link**: [PDF](https://arxiv.org/pdf/2505.03439)  

**Abstract**: The potential for large language models (LLMs) to hide messages within plain text (steganography) poses a challenge to detection and thwarting of unaligned AI agents, and undermines faithfulness of LLMs reasoning. We explore the steganographic capabilities of LLMs fine-tuned via reinforcement learning (RL) to: (1) develop covert encoding schemes, (2) engage in steganography when prompted, and (3) utilize steganography in realistic scenarios where hidden reasoning is likely, but not prompted. In these scenarios, we detect the intention of LLMs to hide their reasoning as well as their steganography performance. Our findings in the fine-tuning experiments as well as in behavioral non fine-tuning evaluations reveal that while current models exhibit rudimentary steganographic abilities in terms of security and capacity, explicit algorithmic guidance markedly enhances their capacity for information concealment. 

---
# Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents 

**Authors**: Schaun Wheeler, Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03434)  

**Abstract**: Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving. 

---
# Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten 

**Authors**: Yuanyuan Yang, Yuan Shen, Tianchen Sun, Yangbin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03369)  

**Abstract**: Free play is a fundamental aspect of early childhood education, supporting children's cognitive, social, emotional, and motor development. However, assessing children's development during free play poses significant challenges due to the unstructured and spontaneous nature of the activity. Traditional assessment methods often rely on direct observations by teachers, parents, or researchers, which may fail to capture comprehensive insights from free play and provide timely feedback to educators. This study proposes an innovative approach combining Large Language Models (LLMs) with learning analytics to analyze children's self-narratives of their play experiences. The LLM identifies developmental abilities, while performance scores across different play settings are calculated using learning analytics techniques. We collected 2,224 play narratives from 29 children in a kindergarten, covering four distinct play areas over one semester. According to the evaluation results from eight professionals, the LLM-based approach achieved high accuracy in identifying cognitive, motor, and social abilities, with accuracy exceeding 90% in most domains. Moreover, significant differences in developmental outcomes were observed across play settings, highlighting each area's unique contributions to specific abilities. These findings confirm that the proposed approach is effective in identifying children's development across various free play settings. This study demonstrates the potential of integrating LLMs and learning analytics to provide child-centered insights into developmental trajectories, offering educators valuable data to support personalized learning and enhance early childhood education practices. 

---
# Domain Adversarial Training for Mitigating Gender Bias in Speech-based Mental Health Detection 

**Authors**: June-Woo Kim, Haram Yoon, Wonkyo Oh, Dawoon Jung, Sung-Hoon Yoon, Dae-Jin Kim, Dong-Ho Lee, Sang-Yeol Lee, Chan-Mo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03359)  

**Abstract**: Speech-based AI models are emerging as powerful tools for detecting depression and the presence of Post-traumatic stress disorder (PTSD), offering a non-invasive and cost-effective way to assess mental health. However, these models often struggle with gender bias, which can lead to unfair and inaccurate predictions. In this study, our study addresses this issue by introducing a domain adversarial training approach that explicitly considers gender differences in speech-based depression and PTSD detection. Specifically, we treat different genders as distinct domains and integrate this information into a pretrained speech foundation model. We then validate its effectiveness on the E-DAIC dataset to assess its impact on performance. Experimental results show that our method notably improves detection performance, increasing the F1-score by up to 13.29 percentage points compared to the baseline. This highlights the importance of addressing demographic disparities in AI-driven mental health assessment. 

---
# AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03332)  

**Abstract**: Critical peer review of scientific manuscripts presents a significant challenge for Large Language Models (LLMs), partly due to data limitations and the complexity of expert reasoning. This report introduces Persistent Workflow Prompting (PWP), a potentially broadly applicable prompt engineering methodology designed to bridge this gap using standard LLM chat interfaces (zero-code, no APIs). We present a proof-of-concept PWP prompt for the critical analysis of experimental chemistry manuscripts, featuring a hierarchical, modular architecture (structured via Markdown) that defines detailed analysis workflows. We develop this PWP prompt through iterative application of meta-prompting techniques and meta-reasoning aimed at systematically codifying expert review workflows, including tacit knowledge. Submitted once at the start of a session, this PWP prompt equips the LLM with persistent workflows triggered by subsequent queries, guiding modern reasoning LLMs through systematic, multimodal evaluations. Demonstrations show the PWP-guided LLM identifying major methodological flaws in a test case while mitigating LLM input bias and performing complex tasks, including distinguishing claims from evidence, integrating text/photo/figure analysis to infer parameters, executing quantitative feasibility checks, comparing estimates against claims, and assessing a priori plausibility. To ensure transparency and facilitate replication, we provide full prompts, detailed demonstration analyses, and logs of interactive chats as supplementary resources. Beyond the specific application, this work offers insights into the meta-development process itself, highlighting the potential of PWP, informed by detailed workflow formalization, to enable sophisticated analysis using readily available LLMs for complex scientific tasks. 

---
# Artificial Behavior Intelligence: Technology, Challenges, and Future Directions 

**Authors**: Kanghyun Jo, Jehwan Choi, Kwanho Kim, Seongmin Kim, Duy-Linh Nguyen, Xuan-Thuy Vo, Adri Priadana, Tien-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2505.03315)  

**Abstract**: Understanding and predicting human behavior has emerged as a core capability in various AI application domains such as autonomous driving, smart healthcare, surveillance systems, and social robotics. This paper defines the technical framework of Artificial Behavior Intelligence (ABI), which comprehensively analyzes and interprets human posture, facial expressions, emotions, behavioral sequences, and contextual cues. It details the essential components of ABI, including pose estimation, face and emotion recognition, sequential behavior analysis, and context-aware modeling. Furthermore, we highlight the transformative potential of recent advances in large-scale pretrained models, such as large language models (LLMs), vision foundation models, and multimodal integration models, in significantly improving the accuracy and interpretability of behavior recognition. Our research team has a strong interest in the ABI domain and is actively conducting research, particularly focusing on the development of intelligent lightweight models capable of efficiently inferring complex human behaviors. This paper identifies several technical challenges that must be addressed to deploy ABI in real-world applications including learning behavioral intelligence from limited data, quantifying uncertainty in complex behavior prediction, and optimizing model structures for low-power, real-time inference. To tackle these challenges, our team is exploring various optimization strategies including lightweight transformers, graph-based recognition architectures, energy-aware loss functions, and multimodal knowledge distillation, while validating their applicability in real-time environments. 

---
# Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces 

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2505.03295)  

**Abstract**: Modern automation systems increasingly rely on modular architectures, with capabilities and skills as one solution approach. Capabilities define the functions of resources in a machine-readable form and skills provide the concrete implementations that realize those capabilities. However, the development of a skill implementation conforming to a corresponding capability remains a time-consuming and challenging task. In this paper, we present a method that treats capabilities as contracts for skill implementations and leverages large language models to generate executable code based on natural language user input. A key feature of our approach is the integration of existing software libraries and interface technologies, enabling the generation of skill implementations across different target languages. We introduce a framework that allows users to incorporate their own libraries and resource interfaces into the code generation process through a retrieval-augmented generation architecture. The proposed method is evaluated using an autonomous mobile robot controlled via Python and ROS 2, demonstrating the feasibility and flexibility of the approach. 

---
# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation 

**Authors**: Tiantian Gan, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03275)  

**Abstract**: Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs. 

---
# Patterns and Mechanisms of Contrastive Activation Engineering 

**Authors**: Yixiong Hao, Ayush Panda, Stepan Shabalin, Sheikh Abdur Raheem Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03189)  

**Abstract**: Controlling the behavior of Large Language Models (LLMs) remains a significant challenge due to their inherent complexity and opacity. While techniques like fine-tuning can modify model behavior, they typically require extensive computational resources. Recent work has introduced a class of contrastive activation engineering (CAE) techniques as promising approaches for steering LLM outputs through targeted modifications to their internal representations. Applied at inference-time with zero cost, CAE has the potential to introduce a new paradigm of flexible, task-specific LLM behavior tuning. We analyze the performance of CAE in in-distribution, out-of-distribution settings, evaluate drawbacks, and begin to develop comprehensive guidelines for its effective deployment. We find that 1. CAE is only reliably effective when applied to in-distribution contexts. 2. Increasing the number of samples used to generate steering vectors has diminishing returns at around 80 samples. 3. Steering vectors are susceptible to adversarial inputs that reverses the behavior that is steered for. 4. Steering vectors harm the overall model perplexity. 5. Larger models are more resistant to steering-induced degradation. 

---
# CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics 

**Authors**: Junqi Liu, Xiaohan Lin, Jonas Bayer, Yael Dillies, Weijie Jiang, Xiaodan Liang, Roman Soletskyi, Haiming Wang, Yunzhou Xie, Beibei Xiong, Zhengfeng Yang, Jujian Zhang, Lihong Zhi, Jia Li, Zhengying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03171)  

**Abstract**: Neurosymbolic approaches integrating large language models with formal reasoning have recently achieved human-level performance on mathematics competition problems in algebra, geometry and number theory. In comparison, combinatorics remains a challenging domain, characterized by a lack of appropriate benchmarks and theorem libraries. To address this gap, we introduce CombiBench, a comprehensive benchmark comprising 100 combinatorial problems, each formalized in Lean~4 and paired with its corresponding informal statement. The problem set covers a wide spectrum of difficulty levels, ranging from middle school to IMO and university level, and span over ten combinatorial topics. CombiBench is suitable for testing IMO solving capabilities since it includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its statement contain an images). Furthermore, we provide a comprehensive and standardized evaluation framework, dubbed Fine-Eval (for $\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for formal mathematics. It accommodates not only proof-based problems but also, for the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval as the evaluation method and Kimina Lean Server as the backend, we benchmark several LLMs on CombiBench and observe that their capabilities for formally solving combinatorial problems remain limited. Among all models tested (none of which has been trained for this particular task), Kimina-Prover attains the best results, solving 7 problems (out of 100) under both ``with solution'' and ``without solution'' scenarios. We open source the benchmark dataset alongside with the code of the proposed evaluation method at this https URL. 

---
# Holmes: Automated Fact Check with Large Language Models 

**Authors**: Haoran Ou, Gelei Deng, Xingshuo Han, Jie Zhang, Xinlei He, Han Qiu, Shangwei Guo, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03135)  

**Abstract**: The rise of Internet connectivity has accelerated the spread of disinformation, threatening societal trust, decision-making, and national security. Disinformation has evolved from simple text to complex multimodal forms combining images and text, challenging existing detection methods. Traditional deep learning models struggle to capture the complexity of multimodal disinformation. Inspired by advances in AI, this study explores using Large Language Models (LLMs) for automated disinformation detection. The empirical study shows that (1) LLMs alone cannot reliably assess the truthfulness of claims; (2) providing relevant evidence significantly improves their performance; (3) however, LLMs cannot autonomously search for accurate evidence. To address this, we propose Holmes, an end-to-end framework featuring a novel evidence retrieval method that assists LLMs in collecting high-quality evidence. Our approach uses (1) LLM-powered summarization to extract key information from open sources and (2) a new algorithm and metrics to evaluate evidence quality. Holmes enables LLMs to verify claims and generate justifications effectively. Experiments show Holmes achieves 88.3% accuracy on two open-source datasets and 90.2% in real-time verification tasks. Notably, our improved evidence retrieval boosts fact-checking accuracy by 30.8% over existing methods 

---
# Is AI currently capable of identifying wild oysters? A comparison of human annotators against the AI model, ODYSSEE 

**Authors**: Brendan Campbell, Alan Williams, Kleio Baxevani, Alyssa Campbell, Rushabh Dhoke, Rileigh E. Hudock, Xiaomin Lin, Vivek Mange, Bernhard Neuberger, Arjun Suresh, Alhim Vera, Arthur Trembanis, Herbert G. Tanner, Edward Hale  

**Link**: [PDF](https://arxiv.org/pdf/2505.03108)  

**Abstract**: Oysters are ecologically and commercially important species that require frequent monitoring to track population demographics (e.g. abundance, growth, mortality). Current methods of monitoring oyster reefs often require destructive sampling methods and extensive manual effort. Therefore, they are suboptimal for small-scale or sensitive environments. A recent alternative, the ODYSSEE model, was developed to use deep learning techniques to identify live oysters using video or images taken in the field of oyster reefs to assess abundance. The validity of this model in identifying live oysters on a reef was compared to expert and non-expert annotators. In addition, we identified potential sources of prediction error. Although the model can make inferences significantly faster than expert and non-expert annotators (39.6 s, $2.34 \pm 0.61$ h, $4.50 \pm 1.46$ h, respectively), the model overpredicted the number of live oysters, achieving lower accuracy (63\%) in identifying live oysters compared to experts (74\%) and non-experts (75\%) alike. Image quality was an important factor in determining the accuracy of the model and the annotators. Better quality images improved human accuracy and worsened model accuracy. Although ODYSSEE was not sufficiently accurate, we anticipate that future training on higher-quality images, utilizing additional live imagery, and incorporating additional annotation training classes will greatly improve the model's predictive power based on the results of this analysis. Future research should address methods that improve the detection of living vs. dead oysters. 

---
# BLAB: Brutally Long Audio Bench 

**Authors**: Orevaoghene Ahia, Martijn Bartelds, Kabir Ahuja, Hila Gonen, Valentin Hofmann, Siddhant Arora, Shuyue Stella Li, Vishal Puttagunta, Mofetoluwa Adeyemi, Charishma Buchireddy, Ben Walls, Noah Bennett, Shinji Watanabe, Noah A. Smith, Yulia Tsvetkov, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03054)  

**Abstract**: Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities. 

---
# Evaluating the Impact of AI-Powered Audiovisual Personalization on Learner Emotion, Focus, and Learning Outcomes 

**Authors**: George Xi Wang, Jingying Deng, Safinah Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03033)  

**Abstract**: Independent learners often struggle with sustaining focus and emotional regulation in unstructured or distracting settings. Although some rely on ambient aids such as music, ASMR, or visual backgrounds to support concentration, these tools are rarely integrated into cohesive, learner-centered systems. Moreover, existing educational technologies focus primarily on content adaptation and feedback, overlooking the emotional and sensory context in which learning takes place. Large language models have demonstrated powerful multimodal capabilities including the ability to generate and adapt text, audio, and visual content. Educational research has yet to fully explore their potential in creating personalized audiovisual learning environments. To address this gap, we introduce an AI-powered system that uses LLMs to generate personalized multisensory study environments. Users select or generate customized visual themes (e.g., abstract vs. realistic, static vs. animated) and auditory elements (e.g., white noise, ambient ASMR, familiar vs. novel sounds) to create immersive settings aimed at reducing distraction and enhancing emotional stability. Our primary research question investigates how combinations of personalized audiovisual elements affect learner cognitive load and engagement. Using a mixed-methods design that incorporates biometric measures and performance outcomes, this study evaluates the effectiveness of LLM-driven sensory personalization. The findings aim to advance emotionally responsive educational technologies and extend the application of multimodal LLMs into the sensory dimension of self-directed learning. 

---
# The Multimodal Paradox: How Added and Missing Modalities Shape Bias and Performance in Multimodal AI 

**Authors**: Kishore Sampath, Pratheesh, Ayaazuddin Mohammad, Resmi Ramachandranpillai  

**Link**: [PDF](https://arxiv.org/pdf/2505.03020)  

**Abstract**: Multimodal learning, which integrates diverse data sources such as images, text, and structured data, has proven superior to unimodal counterparts in high-stakes decision-making. However, while performance gains remain the gold standard for evaluating multimodal systems, concerns around bias and robustness are frequently overlooked. In this context, this paper explores two key research questions (RQs): (i) RQ1 examines whether adding a modality con-sistently enhances performance and investigates its role in shaping fairness measures, assessing whether it mitigates or amplifies bias in multimodal models; (ii) RQ2 investigates the impact of missing modalities at inference time, analyzing how multimodal models generalize in terms of both performance and fairness. Our analysis reveals that incorporating new modalities during training consistently enhances the performance of multimodal models, while fairness trends exhibit variability across different evaluation measures and datasets. Additionally, the absence of modalities at inference degrades performance and fairness, raising concerns about its robustness in real-world deployment. We conduct extensive experiments using multimodal healthcare datasets containing images, time series, and structured information to validate our findings. 

---
# Iterative Resolution of Prompt Ambiguities Using a Progressive Cutting-Search Approach 

**Authors**: Fabrizio Marozzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.02952)  

**Abstract**: Generative AI systems have revolutionized human interaction by enabling natural language-based coding and problem solving. However, the inherent ambiguity of natural language often leads to imprecise instructions, forcing users to iteratively test, correct, and resubmit their prompts. We propose an iterative approach that systematically narrows down these ambiguities through a structured series of clarification questions and alternative solution proposals, illustrated with input/output examples as well. Once every uncertainty is resolved, a final, precise solution is generated. Evaluated on a diverse dataset spanning coding, data analysis, and creative writing, our method demonstrates superior accuracy, competitive resolution times, and higher user satisfaction compared to conventional one-shot solutions, which typically require multiple manual iterations to achieve a correct output. 

---
# VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model 

**Authors**: Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, Haoyu Cao, Ke Li, Rongrong Ji, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03739)  

**Abstract**: With the growing requirement for natural human-computer interaction, speech-based systems receive increasing attention as speech is one of the most common forms of daily communication. However, the existing speech models still experience high latency when generating the first audio token during streaming, which poses a significant bottleneck for deployment. To address this issue, we propose VITA-Audio, an end-to-end large speech model with fast audio-text token generation. Specifically, we introduce a lightweight Multiple Cross-modal Token Prediction (MCTP) module that efficiently generates multiple audio tokens within a single model forward pass, which not only accelerates the inference but also significantly reduces the latency for generating the first audio in streaming scenarios. In addition, a four-stage progressive training strategy is explored to achieve model acceleration with minimal loss of speech quality. To our knowledge, VITA-Audio is the first multi-modal large language model capable of generating audio output during the first forward pass, enabling real-time conversational capabilities with minimal latency. VITA-Audio is fully reproducible and is trained on open-source data only. Experimental results demonstrate that our model achieves an inference speedup of 3~5x at the 7B parameter scale, but also significantly outperforms open-source models of similar model size on multiple benchmarks for automatic speech recognition (ASR), text-to-speech (TTS), and spoken question answering (SQA) tasks. 

---
# AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control 

**Authors**: Jialong Li, Xuxin Cheng, Tianshu Huang, Shiqi Yang, Ri-Zhao Qiu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03738)  

**Abstract**: Humanoid robots derive much of their dexterity from hyper-dexterous whole-body movements, enabling tasks that require a large operational workspace: such as picking objects off the ground. However, achieving these capabilities on real humanoids remains challenging due to their high degrees of freedom (DoF) and nonlinear dynamics. We propose Adaptive Motion Optimization (AMO), a framework that integrates sim-to-real reinforcement learning (RL) with trajectory optimization for real-time, adaptive whole-body control. To mitigate distribution bias in motion imitation RL, we construct a hybrid AMO dataset and train a network capable of robust, on-demand adaptation to potentially O.O.D. commands. We validate AMO in simulation and on a 29-DoF Unitree G1 humanoid robot, demonstrating superior stability and an expanded workspace compared to strong baselines. Finally, we show that AMO's consistent performance supports autonomous task execution via imitation learning, underscoring the system's versatility and robustness. 

---
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios 

**Authors**: Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03730)  

**Abstract**: Action customization involves generating videos where the subject performs actions dictated by input control signals. Current methods use pose-guided or global motion customization but are limited by strict constraints on spatial structure, such as layout, skeleton, and viewpoint consistency, reducing adaptability across diverse subjects and scenarios. To overcome these limitations, we propose FlexiAct, which transfers actions from a reference video to an arbitrary target image. Unlike existing methods, FlexiAct allows for variations in layout, viewpoint, and skeletal structure between the subject of the reference video and the target image, while maintaining identity consistency. Achieving this requires precise action control, spatial structure adaptation, and consistency preservation. To this end, we introduce RefAdapter, a lightweight image-conditioned adapter that excels in spatial adaptation and consistency preservation, surpassing existing methods in balancing appearance consistency and structural flexibility. Additionally, based on our observations, the denoising process exhibits varying levels of attention to motion (low frequency) and appearance details (high frequency) at different timesteps. So we propose FAE (Frequency-aware Action Extraction), which, unlike existing methods that rely on separate spatial-temporal architectures, directly achieves action extraction during the denoising process. Experiments demonstrate that our method effectively transfers actions to subjects with diverse layouts, skeletons, and viewpoints. We release our code and model weights to support further research at this https URL 

---
# Actor-Critics Can Achieve Optimal Sample Efficiency 

**Authors**: Kevin Tan, Wei Fan, Yuting Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.03710)  

**Abstract**: Actor-critic algorithms have become a cornerstone in reinforcement learning (RL), leveraging the strengths of both policy-based and value-based methods. Despite recent progress in understanding their statistical efficiency, no existing work has successfully learned an $\epsilon$-optimal policy with a sample complexity of $O(1/\epsilon^2)$ trajectories with general function approximation when strategic exploration is necessary.
We address this open problem by introducing a novel actor-critic algorithm that attains a sample-complexity of $O(dH^5 \log|\mathcal{A}|/\epsilon^2 + d H^4 \log|\mathcal{F}|/ \epsilon^2)$ trajectories, and accompanying $\sqrt{T}$ regret when the Bellman eluder dimension $d$ does not increase with $T$ at more than a $\log T$ rate.
Here, $\mathcal{F}$ is the critic function class, $\mathcal{A}$ is the action space, and $H$ is the horizon in the finite horizon MDP setting. Our algorithm integrates optimism, off-policy critic estimation targeting the optimal Q-function, and rare-switching policy resets.
We extend this to the setting of Hybrid RL, showing that initializing the critic with offline data yields sample efficiency gains compared to purely offline or online RL. Further, utilizing access to offline data, we provide a \textit{non-optimistic} provably efficient actor-critic algorithm that only additionally requires $N_{\text{off}} \geq c_{\text{off}}^*dH^4/\epsilon^2$ in exchange for omitting optimism, where $c_{\text{off}}^*$ is the single-policy concentrability coefficient and $N_{\text{off}}$ is the number of offline samples. This addresses another open problem in the literature. We further provide numerical experiments to support our theoretical findings. 

---
# Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid 

**Authors**: Parv Kapoor, Ian Higgins, Nikhil Keetha, Jay Patrikar, Brady Moon, Zelin Ye, Yao He, Ivan Cisneros, Yaoyu Hu, Changliu Liu, Eunsuk Kang, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03694)  

**Abstract**: Assured safe-separation is essential for achieving seamless high-density operation of airborne vehicles in a shared airspace. To equip resource-constrained aerial systems with this safety-critical capability, we present ViSafe, a high-speed vision-only airborne collision avoidance system. ViSafe offers a full-stack solution to the Detect and Avoid (DAA) problem by tightly integrating a learning-based edge-AI framework with a custom multi-camera hardware prototype designed under SWaP-C constraints. By leveraging perceptual input-focused control barrier functions (CBF) to design, encode, and enforce safety thresholds, ViSafe can provide provably safe runtime guarantees for self-separation in high-speed aerial operations. We evaluate ViSafe's performance through an extensive test campaign involving both simulated digital twins and real-world flight scenarios. By independently varying agent types, closure rates, interaction geometries, and environmental conditions (e.g., weather and lighting), we demonstrate that ViSafe consistently ensures self-separation across diverse scenarios. In first-of-its-kind real-world high-speed collision avoidance tests with closure rates reaching 144 km/h, ViSafe sets a new benchmark for vision-only autonomous collision avoidance, establishing a new standard for safety in high-speed aerial navigation. 

---
# Revolutionizing Brain Tumor Imaging: Generating Synthetic 3D FA Maps from T1-Weighted MRI using CycleGAN Models 

**Authors**: Xin Du, Francesca M. Cozzi, Rajesh Jena  

**Link**: [PDF](https://arxiv.org/pdf/2505.03662)  

**Abstract**: Fractional anisotropy (FA) and directionally encoded colour (DEC) maps are essential for evaluating white matter integrity and structural connectivity in neuroimaging. However, the spatial misalignment between FA maps and tractography atlases hinders their effective integration into predictive models. To address this issue, we propose a CycleGAN based approach for generating FA maps directly from T1-weighted MRI scans, representing the first application of this technique to both healthy and tumour-affected tissues. Our model, trained on unpaired data, produces high fidelity maps, which have been rigorously evaluated using Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR), demonstrating particularly robust performance in tumour regions. Radiological assessments further underscore the model's potential to enhance clinical workflows by providing an AI-driven alternative that reduces the necessity for additional scans. 

---
# Counterfactual Inference for Eliminating Sentiment Bias in Recommender Systems 

**Authors**: Le Pan, Yuanjiang Cao, Chengkai Huang, Wenjie Zhang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03655)  

**Abstract**: Recommender Systems (RSs) aim to provide personalized recommendations for users. A newly discovered bias, known as sentiment bias, uncovers a common phenomenon within Review-based RSs (RRSs): the recommendation accuracy of users or items with negative reviews deteriorates compared with users or items with positive reviews. Critical users and niche items are disadvantaged by such unfair recommendations. We study this problem from the perspective of counterfactual inference with two stages. At the model training stage, we build a causal graph and model how sentiment influences the final rating score. During the inference stage, we decouple the direct and indirect effects to mitigate the impact of sentiment bias and remove the indirect effect using counterfactual inference. We have conducted extensive experiments, and the results validate that our model can achieve comparable performance on rating prediction for better recommendations and effective mitigation of sentiment bias. To the best of our knowledge, this is the first work to employ counterfactual inference on sentiment bias mitigation in RSs. 

---
# ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant 

**Authors**: Yifan Xiang, Zhenxi Zhang, Bin Li, Yixuan Weng, Shoujun Zhou, Yangfan He, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03654)  

**Abstract**: Recent advances in personalized MLLMs enable effective capture of user-specific concepts, supporting both recognition of personalized concepts and contextual captioning. However, humans typically explore and reason over relations among objects and individuals, transcending surface-level information to achieve more personalized and contextual understanding. To this end, existing methods may face three main limitations: Their training data lacks multi-object sets in which relations among objects are learnable. Building on the limited training data, their models overlook the relations between different personalized concepts and fail to reason over them. Their experiments mainly focus on a single personalized concept, where evaluations are limited to recognition and captioning tasks. To address the limitations, we present a new dataset named ReGraP, consisting of 120 sets of personalized knowledge. Each set includes images, KGs, and CoT QA pairs derived from the KGs, enabling more structured and sophisticated reasoning pathways. We propose ReGraP-LLaVA, an MLLM trained with the corresponding KGs and CoT QA pairs, where soft and hard graph prompting methods are designed to align KGs within the model's semantic space. We establish the ReGraP Benchmark, which contains diverse task types: multiple-choice, fill-in-the-blank, True/False, and descriptive questions in both open- and closed-ended settings. The proposed benchmark is designed to evaluate the relational reasoning and knowledge-connection capability of personalized MLLMs. We conduct experiments on the proposed ReGraP-LLaVA and other competitive MLLMs. Results show that the proposed model not only learns personalized knowledge but also performs relational reasoning in responses, achieving the SoTA performance compared with the competitive methods. All the codes and datasets are released at: this https URL. 

---
# Binding threshold units with artificial oscillatory neurons 

**Authors**: Vladimir Fanaskov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2505.03648)  

**Abstract**: Artificial Kuramoto oscillatory neurons were recently introduced as an alternative to threshold units. Empirical evidence suggests that oscillatory units outperform threshold units in several tasks including unsupervised object discovery and certain reasoning problems. The proposed coupling mechanism for these oscillatory neurons is heterogeneous, combining a generalized Kuramoto equation with standard coupling methods used for threshold units. In this research note, we present a theoretical framework that clearly distinguishes oscillatory neurons from threshold units and establishes a coupling mechanism between them. We argue that, from a biological standpoint, oscillatory and threshold units realise distinct aspects of neural coding: roughly, threshold units model intensity of neuron firing, while oscillatory units facilitate information exchange by frequency modulation. To derive interaction between these two types of units, we constrain their dynamics by focusing on dynamical systems that admit Lyapunov functions. For threshold units, this leads to Hopfield associative memory model, and for oscillatory units it yields a specific form of generalized Kuramoto model. The resulting dynamical systems can be naturally coupled to form a Hopfield-Kuramoto associative memory model, which also admits a Lyapunov function. Various forms of coupling are possible. Notably, oscillatory neurons can be employed to implement a low-rank correction to the weight matrix of a Hopfield network. This correction can be viewed either as a form of Hebbian learning or as a popular LoRA method used for fine-tuning of large language models. We demonstrate the practical realization of this particular coupling through illustrative toy experiments. 

---
# ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders 

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03646)  

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples. 

---
# Rainbow Delay Compensation: A Multi-Agent Reinforcement Learning Framework for Mitigating Delayed Observation 

**Authors**: Songchen Fu, Siang Chen, Shaojing Zhao, Letian Bai, Ta Li, Yonghong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03586)  

**Abstract**: In real-world multi-agent systems (MASs), observation delays are ubiquitous, preventing agents from making decisions based on the environment's true state. An individual agent's local observation often consists of multiple components from other agents or dynamic entities in the environment. These discrete observation components with varying delay characteristics pose significant challenges for multi-agent reinforcement learning (MARL). In this paper, we first formulate the decentralized stochastic individual delay partially observable Markov decision process (DSID-POMDP) by extending the standard Dec-POMDP. We then propose the Rainbow Delay Compensation (RDC), a MARL training framework for addressing stochastic individual delays, along with recommended implementations for its constituent modules. We implement the DSID-POMDP's observation generation pattern using standard MARL benchmarks, including MPE and SMAC. Experiments demonstrate that baseline MARL methods suffer severe performance degradation under fixed and unfixed delays. The RDC-enhanced approach mitigates this issue, remarkably achieving ideal delay-free performance in certain delay scenarios while maintaining generalization capability. Our work provides a novel perspective on multi-agent delayed observation problems and offers an effective solution framework. 

---
# BCause: Human-AI collaboration to improve hybrid mapping and ideation in argumentation-grounded deliberation 

**Authors**: Lucas Anastasiou, Anna De Liddo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03584)  

**Abstract**: Public deliberation, as in open discussion of issues of public concern, often suffers from scattered and shallow discourse, poor sensemaking, and a disconnect from actionable policy outcomes. This paper introduces BCause, a discussion system leveraging generative AI and human-machine collaboration to transform unstructured dialogue around public issues (such as urban living, policy changes, and current socio-economic transformations) into structured, actionable democratic processes. We present three innovations: (i) importing and transforming unstructured transcripts into argumentative discussions, (ii) geo-deliberated problem-sensing via a Telegram bot for local issue reporting, and (iii) smart reporting with customizable widgets (e.g., summaries, topic modelling, policy recommendations, clustered arguments). The system's human-AI partnership preserves critical human participation to ensure ethical oversight, contextual relevance, and creative synthesis. 

---
# LlamaFirewall: An open source guardrail system for building secure AI agents 

**Authors**: Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, Alekhya Gampa, Beto de Paola, Dominik Gabi, James Crnkovich, Jean-Christophe Testud, Kat He, Rashnil Chaturvedi, Wu Zhou, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03574)  

**Abstract**: Large language models (LLMs) have evolved from simple chatbots into autonomous agents capable of performing complex tasks such as editing production code, orchestrating workflows, and taking higher-stakes actions based on untrusted inputs like webpages and emails. These capabilities introduce new security risks that existing security measures, such as model fine-tuning or chatbot-focused guardrails, do not fully address. Given the higher stakes and the absence of deterministic solutions to mitigate these risks, there is a critical need for a real-time guardrail monitor to serve as a final layer of defense, and support system level, use case specific safety policy definition and enforcement. We introduce LlamaFirewall, an open-source security focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI Agents. Our framework mitigates risks such as prompt injection, agent misalignment, and insecure code risks through three powerful guardrails: PromptGuard 2, a universal jailbreak detector that demonstrates clear state of the art performance; Agent Alignment Checks, a chain-of-thought auditor that inspects agent reasoning for prompt injection and goal misalignment, which, while still experimental, shows stronger efficacy at preventing indirect injections in general scenarios than previously proposed approaches; and CodeShield, an online static analysis engine that is both fast and extensible, aimed at preventing the generation of insecure or dangerous code by coding agents. Additionally, we include easy-to-use customizable scanners that make it possible for any developer who can write a regular expression or an LLM prompt to quickly update an agent's security guardrails. 

---
# Real-Time Person Image Synthesis Using a Flow Matching Model 

**Authors**: Jiwoo Jeong, Kirok Kim, Wooju Kim, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.03562)  

**Abstract**: Pose-Guided Person Image Synthesis (PGPIS) generates realistic person images conditioned on a target pose and a source image. This task plays a key role in various real-world applications, such as sign language video generation, AR/VR, gaming, and live streaming. In these scenarios, real-time PGPIS is critical for providing immediate visual feedback and maintaining user this http URL, achieving real-time performance remains a significant challenge due to the complexity of synthesizing high-fidelity images from diverse and dynamic human poses. Recent diffusion-based methods have shown impressive image quality in PGPIS, but their slow sampling speeds hinder deployment in time-sensitive applications. This latency is particularly problematic in tasks like generating sign language videos during live broadcasts, where rapid image updates are required. Therefore, developing a fast and reliable PGPIS model is a crucial step toward enabling real-time interactive systems. To address this challenge, we propose a generative model based on flow matching (FM). Our approach enables faster, more stable, and more efficient training and sampling. Furthermore, the proposed model supports conditional generation and can operate in latent space, making it especially suitable for real-time PGPIS applications where both speed and quality are critical. We evaluate our proposed method, Real-Time Person Image Synthesis Using a Flow Matching Model (RPFM), on the widely used DeepFashion dataset for PGPIS tasks. Our results show that RPFM achieves near-real-time sampling speeds while maintaining performance comparable to the state-of-the-art models. Our methodology trades off a slight, acceptable decrease in generated-image accuracy for over a twofold increase in generation speed, thereby ensuring real-time performance. 

---
# Ergodic Generative Flows 

**Authors**: Leo Maxime Brunswic, Mateo Clemente, Rui Heng Yang, Adam Sigal, Amir Rasouli, Yinchuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03561)  

**Abstract**: Generative Flow Networks (GFNs) were initially introduced on directed acyclic graphs to sample from an unnormalized distribution density. Recent works have extended the theoretical framework for generative methods allowing more flexibility and enhancing application range. However, many challenges remain in training GFNs in continuous settings and for imitation learning (IL), including intractability of flow-matching loss, limited tests of non-acyclic training, and the need for a separate reward model in imitation learning. The present work proposes a family of generative flows called Ergodic Generative Flows (EGFs) which are used to address the aforementioned issues. First, we leverage ergodicity to build simple generative flows with finitely many globally defined transformations (diffeomorphisms) with universality guarantees and tractable flow-matching loss (FM loss). Second, we introduce a new loss involving cross-entropy coupled to weak flow-matching control, coined KL-weakFM loss. It is designed for IL training without a separate reward model. We evaluate IL-EGFs on toy 2D tasks and real-world datasets from NASA on the sphere, using the KL-weakFM loss. Additionally, we conduct toy 2D reinforcement learning experiments with a target reward, using the FM loss. 

---
# Rapid AI-based generation of coverage paths for dispensing applications 

**Authors**: Simon Baeuerle, Ian F. Mendonca, Kristof Van Laerhoven, Ralf Mikut, Andreas Steimer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03560)  

**Abstract**: Coverage Path Planning of Thermal Interface Materials (TIM) plays a crucial role in the design of power electronics and electronic control units. Up to now, this is done manually by experts or by using optimization approaches with a high computational effort. We propose a novel AI-based approach to generate dispense paths for TIM and similar dispensing applications. It is a drop-in replacement for optimization-based approaches. An Artificial Neural Network (ANN) receives the target cooling area as input and directly outputs the dispense path. Our proposed setup does not require labels and we show its feasibility on multiple target areas. The resulting dispense paths can be directly transferred to automated manufacturing equipment and do not exhibit air entrapments. The approach of using an ANN to predict process parameters for a desired target state in real-time could potentially be transferred to other manufacturing processes. 

---
# Generating Synthetic Data via Augmentations for Improved Facial Resemblance in DreamBooth and InstantID 

**Authors**: Koray Ulusan, Benjamin Kiefer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03557)  

**Abstract**: The personalization of Stable Diffusion for generating professional portraits from amateur photographs is a burgeoning area, with applications in various downstream contexts. This paper investigates the impact of augmentations on improving facial resemblance when using two prominent personalization techniques: DreamBooth and InstantID. Through a series of experiments with diverse subject datasets, we assessed the effectiveness of various augmentation strategies on the generated headshots' fidelity to the original subject. We introduce FaceDistance, a wrapper around FaceNet, to rank the generations based on facial similarity, which aided in our assessment. Ultimately, this research provides insights into the role of augmentations in enhancing facial resemblance in SDXL-generated portraits, informing strategies for their effective deployment in downstream applications. 

---
# Optimization of Module Transferability in Single Image Super-Resolution: Universality Assessment and Cycle Residual Blocks 

**Authors**: Haotong Cheng, Zhiqi Zhang, Hao Li, Xinshang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03522)  

**Abstract**: Deep learning has substantially advanced the Single Image Super-Resolution (SISR). However, existing researches have predominantly focused on raw performance gains, with little attention paid to quantifying the transferability of architectural components. In this paper, we introduce the concept of "Universality" and its associated definitions which extend the traditional notion of "Generalization" to encompass the modules' ease of transferability, thus revealing the relationships between module universality and model generalizability. Then we propose the Universality Assessment Equation (UAE), a metric for quantifying how readily a given module could be transplanted across models. Guided by the UAE results of standard residual blocks and other plug-and-play modules, we further design two optimized modules, Cycle Residual Block (CRB) and Depth-Wise Cycle Residual Block (DCRB). Through comprehensive experiments on natural-scene benchmarks, remote-sensing datasets, extreme-industrial imagery and on-device deployments, we demonstrate that networks embedded with the proposed plug-and-play modules outperform several state-of-the-arts, reaching a PSNR enhancement of up to 0.83dB or enabling a 71.3% reduction in parameters with negligible loss in reconstruction fidelity. 

---
# From Neurons to Computation: Biological Reservoir Computing for Pattern Recognition 

**Authors**: Ludovico Iannello, Luca Ciampi, Gabriele Lagani, Fabrizio Tonelli, Eleonora Crocco, Lucio Maria Calcagnile, Angelo Di Garbo, Federico Cremisi, Giuseppe Amato  

**Link**: [PDF](https://arxiv.org/pdf/2505.03510)  

**Abstract**: In this paper, we introduce a novel paradigm for reservoir computing (RC) that leverages a pool of cultured biological neurons as the reservoir substrate, creating a biological reservoir computing (BRC). This system operates similarly to an echo state network (ESN), with the key distinction that the neural activity is generated by a network of cultured neurons, rather than being modeled by traditional artificial computational units. The neuronal activity is recorded using a multi-electrode array (MEA), which enables high-throughput recording of neural signals. In our approach, inputs are introduced into the network through a subset of the MEA electrodes, while the remaining electrodes capture the resulting neural activity. This generates a nonlinear mapping of the input data to a high-dimensional biological feature space, where distinguishing between data becomes more efficient and straightforward, allowing a simple linear classifier to perform pattern recognition tasks effectively. To evaluate the performance of our proposed system, we present an experimental study that includes various input patterns, such as positional codes, bars with different orientations, and a digit recognition task. The results demonstrate the feasibility of using biological neural networks to perform tasks traditionally handled by artificial neural networks, paving the way for further exploration of biologically-inspired computing systems, with potential applications in neuromorphic engineering and bio-hybrid computing. 

---
# Augmenting Human Cognition through Everyday AR 

**Authors**: Xiaoan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03492)  

**Abstract**: As spatial computing and multimodal LLMs mature, AR is tending to become an intuitive "thinking tool," embedding semantic and context-aware intelligence directly into everyday environments. This paper explores how always-on AR can seamlessly bridge digital cognition and physical affordances, enabling proactive, context-sensitive interactions that enhance human task performance and understanding. 

---
# A new membership inference attack that spots memorization in generative and predictive models: Loss-Based with Reference Model algorithm (LBRM) 

**Authors**: Faiz Taleb, Ivan Gazeau, Maryline Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2505.03490)  

**Abstract**: Generative models can unintentionally memorize training data, posing significant privacy risks. This paper addresses the memorization phenomenon in time series imputation models, introducing the Loss-Based with Reference Model (LBRM) algorithm. The LBRM method leverages a reference model to enhance the accuracy of membership inference attacks, distinguishing between training and test data. Our contributions are twofold: first, we propose an innovative method to effectively extract and identify memorized training data, significantly improving detection accuracy. On average, without fine-tuning, the AUROC improved by approximately 40\%. With fine-tuning, the AUROC increased by approximately 60\%. Second, we validate our approach through membership inference attacks on two types of architectures designed for time series imputation, demonstrating the robustness and versatility of the LBRM approach in different contexts. These results highlight the significant enhancement in detection accuracy provided by the LBRM approach, addressing privacy risks in time series imputation models. 

---
# Blending 3D Geometry and Machine Learning for Multi-View Stereopsis 

**Authors**: Vibhas Vats, Md. Alimoor Reza, David Crandall, Soon-heung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.03470)  

**Abstract**: Traditional multi-view stereo (MVS) methods primarily depend on photometric and geometric consistency constraints. In contrast, modern learning-based algorithms often rely on the plane sweep algorithm to infer 3D geometry, applying explicit geometric consistency (GC) checks only as a post-processing step, with no impact on the learning process itself. In this work, we introduce GC MVSNet plus plus, a novel approach that actively enforces geometric consistency of reference view depth maps across multiple source views (multi view) and at various scales (multi scale) during the learning phase (see Fig. 1). This integrated GC check significantly accelerates the learning process by directly penalizing geometrically inconsistent pixels, effectively halving the number of training iterations compared to other MVS methods. Furthermore, we introduce a densely connected cost regularization network with two distinct block designs simple and feature dense optimized to harness dense feature connections for enhanced regularization. Extensive experiments demonstrate that our approach achieves a new state of the art on the DTU and BlendedMVS datasets and secures second place on the Tanks and Temples benchmark. To our knowledge, GC MVSNet plus plus is the first method to enforce multi-view, multi-scale supervised geometric consistency during learning. Our code is available. 

---
# An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation 

**Authors**: Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni, Yoav Kantor, Gal Bloch, Omri Levy, Hadas Abraham, Nitzan Barzilay, Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03452)  

**Abstract**: Finding the optimal Retrieval-Augmented Generation (RAG) configuration for a given use case can be complex and expensive. Motivated by this challenge, frameworks for RAG hyper-parameter optimization (HPO) have recently emerged, yet their effectiveness has not been rigorously benchmarked. To address this gap, we present a comprehensive study involving 5 HPO algorithms over 5 datasets from diverse domains, including a new one collected for this work on real-world product documentation. Our study explores the largest HPO search space considered to date, with two optimized evaluation metrics. Analysis of the results shows that RAG HPO can be done efficiently, either greedily or with iterative random search, and that it significantly boosts RAG performance for all datasets. For greedy HPO approaches, we show that optimizing models first is preferable to the prevalent practice of optimizing sequentially according to the RAG pipeline order. 

---
# Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis 

**Authors**: Fouad Trad, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2505.03451)  

**Abstract**: The rise of QR code based phishing ("Quishing") poses a growing cybersecurity threat, as attackers increasingly exploit QR codes to bypass traditional phishing defenses. Existing detection methods predominantly focus on URL analysis, which requires the extraction of the QR code payload, and may inadvertently expose users to malicious content. Moreover, QR codes can encode various types of data beyond URLs, such as Wi-Fi credentials and payment information, making URL-based detection insufficient for broader security concerns. To address these gaps, we propose the first framework for quishing detection that directly analyzes QR code structure and pixel patterns without extracting the embedded content. We generated a dataset of phishing and benign QR codes and we used it to train and evaluate multiple machine learning models, including Logistic Regression, Decision Trees, Random Forest, Naive Bayes, LightGBM, and XGBoost. Our best-performing model (XGBoost) achieves an AUC of 0.9106, demonstrating the feasibility of QR-centric detection. Through feature importance analysis, we identify key visual indicators of malicious intent and refine our feature set by removing non-informative pixels, improving performance to an AUC of 0.9133 with a reduced feature space. Our findings reveal that the structural features of QR code correlate strongly with phishing risk. This work establishes a foundation for quishing mitigation and highlights the potential of direct QR analysis as a critical layer in modern phishing defenses. 

---
# Elevating Semantic Exploration: A Novel Approach Utilizing Distributed Repositories 

**Authors**: Valerio Bellandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03443)  

**Abstract**: Centralized and distributed systems are two main approaches to organizing ICT infrastructure, each with its pros and cons. Centralized systems concentrate resources in one location, making management easier but creating single points of failure. Distributed systems, on the other hand, spread resources across multiple nodes, offering better scalability and fault tolerance, but requiring more complex management. The choice between them depends on factors like application needs, scalability, and data sensitivity. Centralized systems suit applications with limited scalability and centralized control, while distributed systems excel in large-scale environments requiring high availability and performance. This paper explores a distributed document repository system developed for the Italian Ministry of Justice, using edge repositories to analyze textual data and metadata, enhancing semantic exploration capabilities. 

---
# MedArabiQ: Benchmarking Large Language Models on Arabic Medical Tasks 

**Authors**: Mouath Abu Daoud, Chaimae Abouzahir, Leen Kharouf, Walid Al-Eisawi, Nizar Habash, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2505.03427)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant promise for various applications in healthcare. However, their efficacy in the Arabic medical domain remains unexplored due to the lack of high-quality domain-specific datasets and benchmarks. This study introduces MedArabiQ, a novel benchmark dataset consisting of seven Arabic medical tasks, covering multiple specialties and including multiple choice questions, fill-in-the-blank, and patient-doctor question answering. We first constructed the dataset using past medical exams and publicly available datasets. We then introduced different modifications to evaluate various LLM capabilities, including bias mitigation. We conducted an extensive evaluation with five state-of-the-art open-source and proprietary LLMs, including GPT-4o, Claude 3.5-Sonnet, and Gemini 1.5. Our findings highlight the need for the creation of new high-quality benchmarks that span different languages to ensure fair deployment and scalability of LLMs in healthcare. By establishing this benchmark and releasing the dataset, we provide a foundation for future research aimed at evaluating and enhancing the multilingual capabilities of LLMs for the equitable use of generative AI in healthcare. 

---
# Phenotype-Guided Generative Model for High-Fidelity Cardiac MRI Synthesis: Advancing Pretraining and Clinical Applications 

**Authors**: Ziyu Li, Yujian Hu, Zhengyao Ding, Yiheng Mao, Haitao Li, Fan Yi, Hongkun Zhang, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03426)  

**Abstract**: Cardiac Magnetic Resonance (CMR) imaging is a vital non-invasive tool for diagnosing heart diseases and evaluating cardiac health. However, the limited availability of large-scale, high-quality CMR datasets poses a major challenge to the effective application of artificial intelligence (AI) in this domain. Even the amount of unlabeled data and the health status it covers are difficult to meet the needs of model pretraining, which hinders the performance of AI models on downstream tasks. In this study, we present Cardiac Phenotype-Guided CMR Generation (CPGG), a novel approach for generating diverse CMR data that covers a wide spectrum of cardiac health status. The CPGG framework consists of two stages: in the first stage, a generative model is trained using cardiac phenotypes derived from CMR data; in the second stage, a masked autoregressive diffusion model, conditioned on these phenotypes, generates high-fidelity CMR cine sequences that capture both structural and functional features of the heart in a fine-grained manner. We synthesized a massive amount of CMR to expand the pretraining data. Experimental results show that CPGG generates high-quality synthetic CMR data, significantly improving performance on various downstream tasks, including diagnosis and cardiac phenotypes prediction. These gains are demonstrated across both public and private datasets, highlighting the effectiveness of our approach. Code is availabel at this https URL. 

---
# Framework GNN-AID: Graph Neural Network Analysis Interpretation and Defense 

**Authors**: Kirill Lukyanov, Mikhail Drobyshevskiy, Georgii Sazonov, Mikhail Soloviov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.03424)  

**Abstract**: The growing need for Trusted AI (TAI) highlights the importance of interpretability and robustness in machine learning models. However, many existing tools overlook graph data and rarely combine these two aspects into a single solution. Graph Neural Networks (GNNs) have become a popular approach, achieving top results across various tasks. We introduce GNN-AID (Graph Neural Network Analysis, Interpretation, and Defense), an open-source framework designed for graph data to address this gap. Built as a Python library, GNN-AID supports advanced trust methods and architectural layers, allowing users to analyze graph datasets and GNN behavior using attacks, defenses, and interpretability methods.
GNN-AID is built on PyTorch-Geometric, offering preloaded datasets, models, and support for any GNNs through customizable interfaces. It also includes a web interface with tools for graph visualization and no-code features like an interactive model builder, simplifying the exploration and analysis of GNNs. The framework also supports MLOps techniques, ensuring reproducibility and result versioning to track and revisit analyses efficiently.
GNN-AID is a flexible tool for developers and researchers. It helps developers create, analyze, and customize graph models, while also providing access to prebuilt datasets and models for quick experimentation. Researchers can use the framework to explore advanced topics on the relationship between interpretability and robustness, test defense strategies, and combine methods to protect against different types of attacks.
We also show how defenses against evasion and poisoning attacks can conflict when applied to graph data, highlighting the complex connections between defense strategies.
GNN-AID is available at \href{this https URL}{this http URL} 

---
# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation 

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade  

**Link**: [PDF](https://arxiv.org/pdf/2505.03406)  

**Abstract**: This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings. 

---
# DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation 

**Authors**: Shanshan Song, Hui Tang, Honglong Yang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03401)  

**Abstract**: Radiology Report Generation (RRG) automates the creation of radiology reports from medical imaging, enhancing the efficiency of the reporting process. Longitudinal Radiology Report Generation (LRRG) extends RRG by incorporating the ability to compare current and prior exams, facilitating the tracking of temporal changes in clinical findings. Existing LRRG approaches only extract features from prior and current images using a visual pre-trained encoder, which are then concatenated to generate the final report. However, these methods struggle to effectively capture both spatial and temporal correlations during the feature extraction process. Consequently, the extracted features inadequately capture the information of difference across exams and thus underrepresent the expected progressions, leading to sub-optimal performance in LRRG. To address this, we develop a novel dynamic difference-aware temporal residual network (DDaTR). In DDaTR, we introduce two modules at each stage of the visual encoder to capture multi-level spatial correlations. The Dynamic Feature Alignment Module (DFAM) is designed to align prior features across modalities for the integrity of prior clinical information. Prompted by the enriched prior features, the dynamic difference-aware module (DDAM) captures favorable difference information by identifying relationships across exams. Furthermore, our DDaTR employs the dynamic residual network to unidirectionally transmit longitudinal information, effectively modelling temporal correlations. Extensive experiments demonstrated superior performance over existing methods on three benchmarks, proving its efficacy in both RRG and LRRG tasks. 

---
# Automatic Calibration for Membership Inference Attack on Large Language Models 

**Authors**: Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03392)  

**Abstract**: Membership Inference Attacks (MIAs) have recently been employed to determine whether a specific text was part of the pre-training data of Large Language Models (LLMs). However, existing methods often misinfer non-members as members, leading to a high false positive rate, or depend on additional reference models for probability calibration, which limits their practicality. To overcome these challenges, we introduce a novel framework called Automatic Calibration Membership Inference Attack (ACMIA), which utilizes a tunable temperature to calibrate output probabilities effectively. This approach is inspired by our theoretical insights into maximum likelihood estimation during the pre-training of LLMs. We introduce ACMIA in three configurations designed to accommodate different levels of model access and increase the probability gap between members and non-members, improving the reliability and robustness of membership inference. Extensive experiments on various open-source LLMs demonstrate that our proposed attack is highly effective, robust, and generalizable, surpassing state-of-the-art baselines across three widely used benchmarks. Our code is available at: \href{this https URL}{\textcolor{blue}{Github}}. 

---
# Reinforced Correlation Between Vision and Language for Precise Medical AI Assistant 

**Authors**: Haonan Wang, Jiaji Mao, Lehan Wang, Qixiang Zhang, Marawan Elbatel, Yi Qin, Huijun Hu, Baoxun Li, Wenhui Deng, Weifeng Qin, Hongrui Li, Jialin Liang, Jun Shen, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03380)  

**Abstract**: Medical AI assistants support doctors in disease diagnosis, medical image analysis, and report generation. However, they still face significant challenges in clinical use, including limited accuracy with multimodal content and insufficient validation in real-world settings. We propose RCMed, a full-stack AI assistant that improves multimodal alignment in both input and output, enabling precise anatomical delineation, accurate localization, and reliable diagnosis through hierarchical vision-language grounding. A self-reinforcing correlation mechanism allows visual features to inform language context, while language semantics guide pixel-wise attention, forming a closed loop that refines both modalities. This correlation is enhanced by a color region description strategy, translating anatomical structures into semantically rich text to learn shape-location-text relationships across scales. Trained on 20 million image-mask-description triplets, RCMed achieves state-of-the-art precision in contextualizing irregular lesions and subtle anatomical boundaries, excelling in 165 clinical tasks across 9 modalities. It achieved a 23.5% relative improvement in cell segmentation from microscopy images over prior methods. RCMed's strong vision-language alignment enables exceptional generalization, with state-of-the-art performance in external validation across 20 clinically significant cancer types, including novel tasks. This work demonstrates how integrated multimodal models capture fine-grained patterns, enabling human-level interpretation in complex scenarios and advancing human-centric AI healthcare. 

---
# SPAP: Structured Pruning via Alternating Optimization and Penalty Methods 

**Authors**: Hanyu Hu, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03373)  

**Abstract**: The deployment of large language models (LLMs) is often constrained by their substantial computational and memory demands. While structured pruning presents a viable approach by eliminating entire network components, existing methods suffer from performance degradation, reliance on heuristic metrics, or expensive finetuning. To address these challenges, we propose SPAP (Structured Pruning via Alternating Optimization and Penalty Methods), a novel and efficient structured pruning framework for LLMs grounded in optimization theory. SPAP formulates the pruning problem through a mixed-integer optimization model, employs a penalty method that effectively makes pruning decisions to minimize pruning errors, and introduces an alternating minimization algorithm tailored to the splittable problem structure for efficient weight updates and performance recovery. Extensive experiments on OPT, LLaMA-3/3.1/3.2, and Qwen2.5 models demonstrate SPAP's superiority over state-of-the-art methods, delivering linear inference speedups (1.29$\times$ at 30% sparsity) and proportional memory reductions. Our work offers a practical, optimization-driven solution for pruning LLMs while preserving model performance. 

---
# Safer Prompts: Reducing IP Risk in Visual Generative AI 

**Authors**: Lena Reissinger, Yuanyuan Li, Anna-Carolina Haensch, Neeraj Sarna  

**Link**: [PDF](https://arxiv.org/pdf/2505.03338)  

**Abstract**: Visual Generative AI models have demonstrated remarkable capability in generating high-quality images from simple inputs like text prompts. However, because these models are trained on images from diverse sources, they risk memorizing and reproducing specific content, raising concerns about intellectual property (IP) infringement. Recent advances in prompt engineering offer a cost-effective way to enhance generative AI performance. In this paper, we evaluate the effectiveness of prompt engineering techniques in mitigating IP infringement risks in image generation. Our findings show that Chain of Thought Prompting and Task Instruction Prompting significantly reduce the similarity between generated images and the training data of diffusion models, thereby lowering the risk of IP infringement. 

---
# Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs 

**Authors**: Hao Liao, Wensheng Lu, Jianxun Lian, Mingqi Wu, Shuo Wang, Yong Zhang, Yitian Huang, Mingyang Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03336)  

**Abstract**: Large Language Models (LLMs) have shown promise for generative recommender systems due to their transformative capabilities in user interaction. However, ensuring they do not recommend out-of-domain (OOD) items remains a challenge. We study two distinct methods to address this issue: RecLM-ret, a retrieval-based method, and RecLM-cgen, a constrained generation method. Both methods integrate seamlessly with existing LLMs to ensure in-domain recommendations. Comprehensive experiments on three recommendation datasets demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing LLM-based recommender models in accuracy while eliminating OOD recommendations, making it the preferred method for adoption. Additionally, RecLM-cgen maintains strong generalist capabilities and is a lightweight plug-and-play module for easy integration into LLMs, offering valuable practical benefits for the community. Source code is available at this https URL 

---
# Absolute Zero: Reinforced Self-play Reasoning with Zero Data 

**Authors**: Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03335)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes. 

---
# Very High-Resolution Forest Mapping with TanDEM-X InSAR Data and Self-Supervised Learning 

**Authors**: José-Luis Bueso-Bello, Benjamin Chauvel, Daniel Carcereri, Philipp Posovszky, Pietro Milillo, Jennifer Ruiz, Juan-Carlos Fernández-Diaz, Carolina González, Michele Martone, Ronny Hänsch, Paola Rizzoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03327)  

**Abstract**: Deep learning models have shown encouraging capabilities for mapping accurately forests at medium resolution with TanDEM-X interferometric SAR data. Such models, as most of current state-of-the-art deep learning techniques in remote sensing, are trained in a fully-supervised way, which requires a large amount of labeled data for training and validation. In this work, our aim is to exploit the high-resolution capabilities of the TanDEM-X mission to map forests at 6 m. The goal is to overcome the intrinsic limitations posed by midresolution products, which affect, e.g., the detection of narrow roads within vegetated areas and the precise delineation of forested regions contours. To cope with the lack of extended reliable reference datasets at such a high resolution, we investigate self-supervised learning techniques for extracting highly informative representations from the input features, followed by a supervised training step with a significantly smaller number of reliable labels. A 1 m resolution forest/non-forest reference map over Pennsylvania, USA, allows for comparing different training approaches for the development of an effective forest mapping framework with limited labeled samples. We select the best-performing approach over this test region and apply it in a real-case forest mapping scenario over the Amazon rainforest, where only very few labeled data at high resolution are available. In this challenging scenario, the proposed self-supervised framework significantly enhances the classification accuracy with respect to fully-supervised methods, trained using the same amount of labeled data, representing an extremely promising starting point for large-scale, very high-resolution forest mapping with TanDEM-X data. 

---
# SD-VSum: A Method and Dataset for Script-Driven Video Summarization 

**Authors**: Manolis Mylonas, Evlampios Apostolidis, Vasileios Mezaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.03319)  

**Abstract**: In this work, we introduce the task of script-driven video summarization, which aims to produce a summary of the full-length video by selecting the parts that are most relevant to a user-provided script outlining the visual content of the desired summary. Following, we extend a recently-introduced large-scale dataset for generic video summarization (VideoXum) by producing natural language descriptions of the different human-annotated summaries that are available per video. In this way we make it compatible with the introduced task, since the available triplets of ``video, summary and summary description'' can be used for training a method that is able to produce different summaries for a given video, driven by the provided script about the content of each summary. Finally, we develop a new network architecture for script-driven video summarization (SD-VSum), that relies on the use of a cross-modal attention mechanism for aligning and fusing information from the visual and text modalities. Our experimental evaluations demonstrate the advanced performance of SD-VSum against state-of-the-art approaches for query-driven and generic (unimodal and multimodal) summarization from the literature, and document its capacity to produce video summaries that are adapted to each user's needs about their content. 

---
# Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation 

**Authors**: Jincheng Zhang, György Fazekas, Charalampos Saitis  

**Link**: [PDF](https://arxiv.org/pdf/2505.03314)  

**Abstract**: The recent surge in the popularity of diffusion models for image synthesis has attracted new attention to their potential for generation tasks in other domains. However, their applications to symbolic music generation remain largely under-explored because symbolic music is typically represented as sequences of discrete events and standard diffusion models are not well-suited for discrete data. We represent symbolic music as image-like pianorolls, facilitating the use of diffusion models for the generation of symbolic music. Moreover, this study introduces a novel diffusion model that incorporates our proposed Transformer-Mamba block and learnable wavelet transform. Classifier-free guidance is utilised to generate symbolic music with target chords. Our evaluation shows that our method achieves compelling results in terms of music quality and controllability, outperforming the strong baseline in pianoroll generation. Our code is available at this https URL. 

---
# Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices 

**Authors**: Tasnim Shahriar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03303)  

**Abstract**: This paper presents a comprehensive evaluation of lightweight deep learning models for image classification, emphasizing their suitability for deployment in resource-constrained environments such as low-memory devices. Five state-of-the-art architectures - MobileNetV3 Small, ResNet18, SqueezeNet, EfficientNetV2-S, and ShuffleNetV2 - are benchmarked across three diverse datasets: CIFAR-10, CIFAR-100, and Tiny ImageNet. The models are assessed using four key performance metrics: classification accuracy, inference time, floating-point operations (FLOPs), and model size. Additionally, we investigate the impact of hyperparameter tuning, data augmentation, and training paradigms by comparing pretrained models with scratch-trained counterparts, focusing on MobileNetV3 Small. Our findings reveal that transfer learning significantly enhances model accuracy and computational efficiency, particularly for complex datasets like Tiny ImageNet. EfficientNetV2 consistently achieves the highest accuracy, while MobileNetV3 offers the best balance between accuracy and efficiency, and SqueezeNet excels in inference speed and compactness. This study highlights critical trade-offs between accuracy and efficiency, offering actionable insights for deploying lightweight models in real-world applications where computational resources are limited. By addressing these challenges, this research contributes to optimizing deep learning systems for edge computing and mobile platforms. 

---
# Towards Efficient Benchmarking of Foundation Models in Remote Sensing: A Capabilities Encoding Approach 

**Authors**: Pierre Adorni, Minh-Tan Pham, Stéphane May, Sébastien Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2505.03299)  

**Abstract**: Foundation models constitute a significant advancement in computer vision: after a single, albeit costly, training phase, they can address a wide array of tasks. In the field of Earth observation, over 75 remote sensing vision foundation models have been developed in the past four years. However, none has consistently outperformed the others across all available downstream tasks. To facilitate their comparison, we propose a cost-effective method for predicting a model's performance on multiple downstream tasks without the need for fine-tuning on each one. This method is based on what we call "capabilities encoding." The utility of this novel approach is twofold: we demonstrate its potential to simplify the selection of a foundation model for a given new task, and we employ it to offer a fresh perspective on the existing literature, suggesting avenues for future research. Codes are available at this https URL. 

---
# The Unreasonable Effectiveness of Discrete-Time Gaussian Process Mixtures for Robot Policy Learning 

**Authors**: Jan Ole von Hartz, Adrian Röfer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03296)  

**Abstract**: We present Mixture of Discrete-time Gaussian Processes (MiDiGap), a novel approach for flexible policy representation and imitation learning in robot manipulation. MiDiGap enables learning from as few as five demonstrations using only camera observations and generalizes across a wide range of challenging tasks. It excels at long-horizon behaviors such as making coffee, highly constrained motions such as opening doors, dynamic actions such as scooping with a spatula, and multimodal tasks such as hanging a mug. MiDiGap learns these tasks on a CPU in less than a minute and scales linearly to large datasets. We also develop a rich suite of tools for inference-time steering using evidence such as collision signals and robot kinematic constraints. This steering enables novel generalization capabilities, including obstacle avoidance and cross-embodiment policy transfer. MiDiGap achieves state-of-the-art performance on diverse few-shot manipulation benchmarks. On constrained RLBench tasks, it improves policy success by 76 percentage points and reduces trajectory cost by 67%. On multimodal tasks, it improves policy success by 48 percentage points and increases sample efficiency by a factor of 20. In cross-embodiment transfer, it more than doubles policy success. We make the code publicly available at this https URL. 

---
# Physics-inspired Energy Transition Neural Network for Sequence Learning 

**Authors**: Zhou Wu, Junyi An, Baile Xu, Furao Shen, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03281)  

**Abstract**: Recently, the superior performance of Transformers has made them a more robust and scalable solution for sequence modeling than traditional recurrent neural networks (RNNs). However, the effectiveness of Transformer in capturing long-term dependencies is primarily attributed to their comprehensive pair-modeling process rather than inherent inductive biases toward sequence semantics. In this study, we explore the capabilities of pure RNNs and reassess their long-term learning mechanisms. Inspired by the physics energy transition models that track energy changes over time, we propose a effective recurrent structure called the``Physics-inspired Energy Transition Neural Network" (PETNN). We demonstrate that PETNN's memory mechanism effectively stores information over long-term dependencies. Experimental results indicate that PETNN outperforms transformer-based methods across various sequence tasks. Furthermore, owing to its recurrent nature, PETNN exhibits significantly lower complexity. Our study presents an optimal foundational recurrent architecture and highlights the potential for developing effective recurrent neural networks in fields currently dominated by Transformer. 

---
# Synthline: A Product Line Approach for Synthetic Requirements Engineering Data Generation using Large Language Models 

**Authors**: Abdelkarim El-Hajjami, Camille Salinesi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03265)  

**Abstract**: While modern Requirements Engineering (RE) heavily relies on natural language processing and Machine Learning (ML) techniques, their effectiveness is limited by the scarcity of high-quality datasets. This paper introduces Synthline, a Product Line (PL) approach that leverages Large Language Models to systematically generate synthetic RE data for classification-based use cases. Through an empirical evaluation conducted in the context of using ML for the identification of requirements specification defects, we investigated both the diversity of the generated data and its utility for training downstream models. Our analysis reveals that while synthetic datasets exhibit less diversity than real data, they are good enough to serve as viable training resources. Moreover, our evaluation shows that combining synthetic and real data leads to substantial performance improvements. Specifically, hybrid approaches achieve up to 85% improvement in precision and a 2x increase in recall compared to models trained exclusively on real data. These findings demonstrate the potential of PL-based synthetic data generation to address data scarcity in RE. We make both our implementation and generated datasets publicly available to support reproducibility and advancement in the field. 

---
# Seeing the Abstract: Translating the Abstract Language for Vision Language Models 

**Authors**: Davide Talon, Federico Girella, Ziyue Liu, Marco Cristani, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03242)  

**Abstract**: Natural language goes beyond dryly describing visual content. It contains rich abstract concepts to express feeling, creativity and properties that cannot be directly perceived. Yet, current research in Vision Language Models (VLMs) has not shed light on abstract-oriented language. Our research breaks new ground by uncovering its wide presence and under-estimated value, with extensive analysis. Particularly, we focus our investigation on the fashion domain, a highly-representative field with abstract expressions. By analyzing recent large-scale multimodal fashion datasets, we find that abstract terms have a dominant presence, rivaling the concrete ones, providing novel information, and being useful in the retrieval task. However, a critical challenge emerges: current general-purpose or fashion-specific VLMs are pre-trained with databases that lack sufficient abstract words in their text corpora, thus hindering their ability to effectively represent abstract-oriented language. We propose a training-free and model-agnostic method, Abstract-to-Concrete Translator (ACT), to shift abstract representations towards well-represented concrete ones in the VLM latent space, using pre-trained models and existing multimodal databases. On the text-to-image retrieval task, despite being training-free, ACT outperforms the fine-tuned VLMs in both same- and cross-dataset settings, exhibiting its effectiveness with a strong generalization capability. Moreover, the improvement introduced by ACT is consistent with various VLMs, making it a plug-and-play solution. 

---
# Accelerating Evolution: Integrating PSO Principles into Real-Coded Genetic Algorithm Crossover 

**Authors**: Xiaobo Jin, JiaShu Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03217)  

**Abstract**: This study introduces an innovative crossover operator named Particle Swarm Optimization-inspired Crossover (PSOX), which is specifically developed for real-coded genetic algorithms. Departing from conventional crossover approaches that only exchange information between individuals within the same generation, PSOX uniquely incorporates guidance from both the current global best solution and historical optimal solutions across multiple generations. This novel mechanism enables the algorithm to maintain population diversity while simultaneously accelerating convergence toward promising regions of the search space. The effectiveness of PSOX is rigorously evaluated through comprehensive experiments on 15 benchmark test functions with diverse characteristics, including unimodal, multimodal, and highly complex landscapes. Comparative analysis against five state-of-the-art crossover operators reveals that PSOX consistently delivers superior performance in terms of solution accuracy, algorithmic stability, and convergence speed, especially when combined with an appropriate mutation strategy. Furthermore, the study provides an in-depth investigation of how different mutation rates influence PSOX's performance, yielding practical guidelines for parameter tuning when addressing optimization problems with varying landscape properties. 

---
# DocSpiral: A Platform for Integrated Assistive Document Annotation through Human-in-the-Spiral 

**Authors**: Qiang Sun, Sirui Li, Tingting Bi, Du Huynh, Mark Reynolds, Yuanyi Luo, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03214)  

**Abstract**: Acquiring structured data from domain-specific, image-based documents such as scanned reports is crucial for many downstream tasks but remains challenging due to document variability. Many of these documents exist as images rather than as machine-readable text, which requires human annotation to train automated extraction systems. We present DocSpiral, the first Human-in-the-Spiral assistive document annotation platform, designed to address the challenge of extracting structured information from domain-specific, image-based document collections. Our spiral design establishes an iterative cycle in which human annotations train models that progressively require less manual intervention. DocSpiral integrates document format normalization, comprehensive annotation interfaces, evaluation metrics dashboard, and API endpoints for the development of AI / ML models into a unified workflow. Experiments demonstrate that our framework reduces annotation time by at least 41\% while showing consistent performance gains across three iterations during model training. By making this annotation platform freely accessible, we aim to lower barriers to AI/ML models development in document processing, facilitating the adoption of large language models in image-based, document-intensive fields such as geoscience and healthcare. The system is freely available at: this https URL. The demonstration video is available: this https URL. 

---
# DCS-ST for Classification of Breast Cancer Histopathology Images with Limited Annotations 

**Authors**: Liu Suxing, Byungwon Min  

**Link**: [PDF](https://arxiv.org/pdf/2505.03204)  

**Abstract**: Deep learning methods have shown promise in classifying breast cancer histopathology images, but their performance often declines with limited annotated data, a critical challenge in medical imaging due to the high cost and expertise required for annotations. 

---
# A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case 

**Authors**: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dusit Niyato, Hongfang Yu, Mohammed Atiquzzaman, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03196)  

**Abstract**: Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area. 

---
# A study on audio synchronous steganography detection and distributed guide inference model based on sliding spectral features and intelligent inference drive 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03193)  

**Abstract**: With the rise of short video platforms in global communication, embedding steganographic data in audio synchronization streams has emerged as a new covert communication method. To address the limitations of traditional techniques in detecting synchronized steganography, this paper proposes a detection and distributed guidance reconstruction model based on short video "Yupan" samples released by China's South Sea Fleet on TikTok. The method integrates sliding spectrum feature extraction and intelligent inference mechanisms. A 25 ms sliding window with short-time Fourier transform (STFT) is used to extract the main frequency trajectory and construct the synchronization frame detection model (M1), identifying a frame flag "FFFFFFFFFFFFFFFFFF80". The subsequent 32-byte payload is decoded by a structured model (M2) to infer distributed guidance commands. Analysis reveals a low-entropy, repetitive byte sequence in the 36 to 45 second audio segment with highly concentrated spectral energy, confirming the presence of synchronization frames. Although plaintext semantics are not restored, the consistency in command field layout suggests features of military communication protocols. The multi-segment splicing model further shows cross-video embedding and centralized decoding capabilities. The proposed framework validates the effectiveness of sliding spectral features for synchronized steganography detection and builds an extensible inference model for covert communication analysis and tactical guidance simulation on open platforms. 

---
# seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models 

**Authors**: Hafez Ghaemi, Eilif Muller, Shahab Bakhtiari  

**Link**: [PDF](https://arxiv.org/pdf/2505.03176)  

**Abstract**: Current self-supervised algorithms mostly rely on transformations such as data augmentation and masking to learn visual representations. This is achieved by inducing invariance or equivariance with respect to these transformations after encoding two views of an image. This dominant two-view paradigm can limit the flexibility of learned representations for downstream adaptation by creating performance trade-offs between invariance-related tasks such as image classification and more fine-grained equivariance-related tasks. In this work, we introduce \emph{seq-JEPA}, a world modeling paradigm based on joint-embedding predictive architecture that leverages architectural inductive biases to resolve this trade-off. Without requiring an additional equivariance predictor or loss term, seq-JEPA simultaneously learns two architecturally segregated representations: one equivariant to the specified transformations and another invariant to them and suited for tasks such as classification. To do so, our model processes a short sequence of different views (observations) of an input image. Each encoded view is concatenated with embeddings corresponding to the relative transformation (action) producing the next observation in the sequence. A transformer encoder outputs an aggregate representation of this sequence, which is subsequently conditioned on the action leading to the next observation to predict its representation. Empirically, seq-JEPA achieves strong performance on equivariant benchmarks and image classification without sacrificing one for the other. Additionally, our framework excels at tasks that inherently require aggregating a sequence of observations, such as path integration across actions and predictive learning across eye movements. 

---
# RAVU: Retrieval Augmented Video Understanding with Compositional Reasoning over Graph 

**Authors**: Sameer Malik, Moyuru Yamada, Ayush Singh, Dishank Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.03173)  

**Abstract**: Comprehending long videos remains a significant challenge for Large Multi-modal Models (LMMs). Current LMMs struggle to process even minutes to hours videos due to their lack of explicit memory and retrieval mechanisms. To address this limitation, we propose RAVU (Retrieval Augmented Video Understanding), a novel framework for video understanding enhanced by retrieval with compositional reasoning over a spatio-temporal graph. We construct a graph representation of the video, capturing both spatial and temporal relationships between entities. This graph serves as a long-term memory, allowing us to track objects and their actions across time. To answer complex queries, we decompose the queries into a sequence of reasoning steps and execute these steps on the graph, retrieving relevant key information. Our approach enables more accurate understanding of long videos, particularly for queries that require multi-hop reasoning and tracking objects across frames. Our approach demonstrate superior performances with limited retrieved frames (5-10) compared with other SOTA methods and baselines on two major video QA datasets, NExT-QA and EgoSchema. 

---
# Null Counterfactual Factor Interactions for Goal-Conditioned Reinforcement Learning 

**Authors**: Caleb Chuck, Fan Feng, Carl Qi, Chang Shi, Siddhant Agarwal, Amy Zhang, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2505.03172)  

**Abstract**: Hindsight relabeling is a powerful tool for overcoming sparsity in goal-conditioned reinforcement learning (GCRL), especially in certain domains such as navigation and locomotion. However, hindsight relabeling can struggle in object-centric domains. For example, suppose that the goal space consists of a robotic arm pushing a particular target block to a goal location. In this case, hindsight relabeling will give high rewards to any trajectory that does not interact with the block. However, these behaviors are only useful when the object is already at the goal -- an extremely rare case in practice. A dataset dominated by these kinds of trajectories can complicate learning and lead to failures. In object-centric domains, one key intuition is that meaningful trajectories are often characterized by object-object interactions such as pushing the block with the gripper. To leverage this intuition, we introduce Hindsight Relabeling using Interactions (HInt), which combines interactions with hindsight relabeling to improve the sample efficiency of downstream RL. However because interactions do not have a consensus statistical definition tractable for downstream GCRL, we propose a definition of interactions based on the concept of null counterfactual: a cause object is interacting with a target object if, in a world where the cause object did not exist, the target object would have different transition dynamics. We leverage this definition to infer interactions in Null Counterfactual Interaction Inference (NCII), which uses a "nulling'' operation with a learned model to infer interactions. NCII is able to achieve significantly improved interaction inference accuracy in both simple linear dynamics domains and dynamic robotic domains in Robosuite, Robot Air Hockey, and Franka Kitchen and HInt improves sample efficiency by up to 4x. 

---
# Soft Best-of-n Sampling for Model Alignment 

**Authors**: Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03156)  

**Abstract**: Best-of-$n$ (BoN) sampling is a practical approach for aligning language model outputs with human preferences without expensive fine-tuning. BoN sampling is performed by generating $n$ responses to a prompt and then selecting the sample that maximizes a reward function. BoN yields high reward values in practice at a distortion cost, as measured by the KL-divergence between the sampled and original distribution. This distortion is coarsely controlled by varying the number of samples: larger $n$ yields a higher reward at a higher distortion cost. We introduce Soft Best-of-$n$ sampling, a generalization of BoN that allows for smooth interpolation between the original distribution and reward-maximizing distribution through a temperature parameter $\lambda$. We establish theoretical guarantees showing that Soft Best-of-$n$ sampling converges sharply to the optimal tilted distribution at a rate of $O(1/n)$ in KL and the expected (relative) reward. For sequences of discrete outputs, we analyze an additive reward model that reveals the fundamental limitations of blockwise sampling. 

---
# StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data 

**Authors**: Yuxuan Mu, Hung Yu Ling, Yi Shi, Ismael Baira Ojeda, Pengcheng Xi, Chang Shu, Fabio Zinno, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03154)  

**Abstract**: Motion capture (mocap) data often exhibits visually jarring artifacts due to inaccurate sensors and post-processing. Cleaning this corrupted data can require substantial manual effort from human experts, which can be a costly and time-consuming process. Previous data-driven motion cleanup methods offer the promise of automating this cleanup process, but often require in-domain paired corrupted-to-clean training data. Constructing such paired datasets requires access to high-quality, relatively artifact-free motion clips, which often necessitates laborious manual cleanup. In this work, we present StableMotion, a simple yet effective method for training motion cleanup models directly from unpaired corrupted datasets that need cleanup. The core component of our method is the introduction of motion quality indicators, which can be easily annotated through manual labeling or heuristic algorithms and enable training of quality-aware motion generation models on raw motion data with mixed quality. At test time, the model can be prompted to generate high-quality motions using the quality indicators. Our method can be implemented through a simple diffusion-based framework, leading to a unified motion generate-discriminate model, which can be used to both identify and fix corrupted frames. We demonstrate that our proposed method is effective for training motion cleanup models on raw mocap data in production scenarios by applying StableMotion to SoccerMocap, a 245-hour soccer mocap dataset containing real-world motion artifacts. The trained model effectively corrects a wide range of motion artifacts, reducing motion pops and frozen frames by 68% and 81%, respectively. See this https URL for more results. 

---
# Motion-compensated cardiac MRI using low-rank diffeomorphic flow (DMoCo) 

**Authors**: Joseph William Kettelkamp, Ludovica Romanin, Sarv Priya, Mathews Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2505.03149)  

**Abstract**: We introduce an unsupervised motion-compensated image reconstruction algorithm for free-breathing and ungated 3D cardiac magnetic resonance imaging (MRI). We express the image volume corresponding to each specific motion phase as the deformation of a single static image template. The main contribution of the work is the low-rank model for the compact joint representation of the family of diffeomorphisms, parameterized by the motion phases. The diffeomorphism at a specific motion phase is obtained by integrating a parametric velocity field along a path connecting the reference template phase to the motion phase. The velocity field at different phases is represented using a low-rank model. The static template and the low-rank motion model parameters are learned directly from the k-space data in an unsupervised fashion. The more constrained motion model is observed to offer improved recovery compared to current motion-resolved and motion-compensated algorithms for free-breathing 3D cine MRI. 

---
# VISLIX: An XAI Framework for Validating Vision Models with Slice Discovery and Analysis 

**Authors**: Xinyuan Yan, Xiwei Xuan, Jorge Piazentin Ono, Jiajing Guo, Vikram Mohanty, Shekar Arvind Kumar, Liang Gou, Bei Wang, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03132)  

**Abstract**: Real-world machine learning models require rigorous evaluation before deployment, especially in safety-critical domains like autonomous driving and surveillance. The evaluation of machine learning models often focuses on data slices, which are subsets of the data that share a set of characteristics. Data slice finding automatically identifies conditions or data subgroups where models underperform, aiding developers in mitigating performance issues. Despite its popularity and effectiveness, data slicing for vision model validation faces several challenges. First, data slicing often needs additional image metadata or visual concepts, and falls short in certain computer vision tasks, such as object detection. Second, understanding data slices is a labor-intensive and mentally demanding process that heavily relies on the expert's domain knowledge. Third, data slicing lacks a human-in-the-loop solution that allows experts to form hypothesis and test them interactively. To overcome these limitations and better support the machine learning operations lifecycle, we introduce VISLIX, a novel visual analytics framework that employs state-of-the-art foundation models to help domain experts analyze slices in computer vision models. Our approach does not require image metadata or visual concepts, automatically generates natural language insights, and allows users to test data slice hypothesis interactively. We evaluate VISLIX with an expert study and three use cases, that demonstrate the effectiveness of our tool in providing comprehensive insights for validating object detection models. 

---
# Cognitio Emergens: Agency, Dimensions, and Dynamics in Human-AI Knowledge Co-Creation 

**Authors**: Xule Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03105)  

**Abstract**: Scientific knowledge creation is fundamentally transforming as humans and AI systems evolve beyond tool-user relationships into co-evolutionary epistemic partnerships. When AlphaFold revolutionized protein structure prediction, researchers described engaging with an epistemic partner that reshaped how they conceptualized fundamental relationships. This article introduces Cognitio Emergens (CE), a framework addressing critical limitations in existing models that focus on static roles or narrow metrics while failing to capture how scientific understanding emerges through recursive human-AI interaction over time. CE integrates three components addressing these limitations: Agency Configurations describing how authority distributes between humans and AI (Directed, Contributory, Partnership), with partnerships dynamically oscillating between configurations rather than following linear progression; Epistemic Dimensions capturing six specific capabilities emerging through collaboration across Discovery, Integration, and Projection axes, creating distinctive "capability signatures" that guide development; and Partnership Dynamics identifying forces shaping how these relationships evolve, particularly the risk of epistemic alienation where researchers lose interpretive control over knowledge they formally endorse. Drawing from autopoiesis theory, social systems theory, and organizational modularity, CE reveals how knowledge co-creation emerges through continuous negotiation of roles, values, and organizational structures. By reconceptualizing human-AI scientific collaboration as fundamentally co-evolutionary, CE offers a balanced perspective that neither uncritically celebrates nor unnecessarily fears AI's evolving role, instead providing conceptual tools for cultivating partnerships that maintain meaningful human participation while enabling transformative scientific breakthroughs. 

---
# Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering 

**Authors**: Joshua Owotogbe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03096)  

**Abstract**: This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications. 

---
# Latent Adaptive Planner for Dynamic Manipulation 

**Authors**: Donghun Noh, Deqian Kong, Minglu Zhao, Andrew Lizarraga, Jianwen Xie, Ying Nian Wu, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.03077)  

**Abstract**: This paper presents Latent Adaptive Planner (LAP), a novel approach for dynamic nonprehensile manipulation tasks that formulates planning as latent space inference, effectively learned from human demonstration videos. Our method addresses key challenges in visuomotor policy learning through a principled variational replanning framework that maintains temporal consistency while efficiently adapting to environmental changes. LAP employs Bayesian updating in latent space to incrementally refine plans as new observations become available, striking an optimal balance between computational efficiency and real-time adaptability. We bridge the embodiment gap between humans and robots through model-based proportional mapping that regenerates accurate kinematic-dynamic joint states and object positions from human demonstrations. Experimental evaluations across multiple complex manipulation benchmarks demonstrate that LAP achieves state-of-the-art performance, outperforming existing approaches in success rate, trajectory smoothness, and energy efficiency, particularly in dynamic adaptation scenarios. Our approach enables robots to perform complex interactions with human-like adaptability while providing an expandable framework applicable to diverse robotic platforms using the same human demonstration videos. 

---
# Developing A Framework to Support Human Evaluation of Bias in Generated Free Response Text 

**Authors**: Jennifer Healey, Laurie Byrum, Md Nadeem Akhtar, Surabhi Bhargava, Moumita Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03053)  

**Abstract**: LLM evaluation is challenging even the case of base models. In real world deployments, evaluation is further complicated by the interplay of task specific prompts and experiential context. At scale, bias evaluation is often based on short context, fixed choice benchmarks that can be rapidly evaluated, however, these can lose validity when the LLMs' deployed context differs. Large scale human evaluation is often seen as too intractable and costly. Here we present our journey towards developing a semi-automated bias evaluation framework for free text responses that has human insights at its core. We discuss how we developed an operational definition of bias that helped us automate our pipeline and a methodology for classifying bias beyond multiple choice. We additionally comment on how human evaluation helped us uncover problematic templates in a bias benchmark. 

---
# MORE: Mobile Manipulation Rearrangement Through Grounded Language Reasoning 

**Authors**: Mohammad Mohammadi, Daniel Honerkamp, Martin Büchner, Matteo Cassinelli, Tim Welschehold, Fabien Despinoy, Igor Gilitschenski, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03035)  

**Abstract**: Autonomous long-horizon mobile manipulation encompasses a multitude of challenges, including scene dynamics, unexplored areas, and error recovery. Recent works have leveraged foundation models for scene-level robotic reasoning and planning. However, the performance of these methods degrades when dealing with a large number of objects and large-scale environments. To address these limitations, we propose MORE, a novel approach for enhancing the capabilities of language models to solve zero-shot mobile manipulation planning for rearrangement tasks. MORE leverages scene graphs to represent environments, incorporates instance differentiation, and introduces an active filtering scheme that extracts task-relevant subgraphs of object and region instances. These steps yield a bounded planning problem, effectively mitigating hallucinations and improving reliability. Additionally, we introduce several enhancements that enable planning across both indoor and outdoor environments. We evaluate MORE on 81 diverse rearrangement tasks from the BEHAVIOR-1K benchmark, where it becomes the first approach to successfully solve a significant share of the benchmark, outperforming recent foundation model-based approaches. Furthermore, we demonstrate the capabilities of our approach in several complex real-world tasks, mimicking everyday activities. We make the code publicly available at this https URL. 

---
# A Typology of Synthetic Datasets for Dialogue Processing in Clinical Contexts 

**Authors**: Steven Bedrick, A. Seza Doğruöz, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03025)  

**Abstract**: Synthetic data sets are used across linguistic domains and NLP tasks, particularly in scenarios where authentic data is limited (or even non-existent). One such domain is that of clinical (healthcare) contexts, where there exist significant and long-standing challenges (e.g., privacy, anonymization, and data governance) which have led to the development of an increasing number of synthetic datasets. One increasingly important category of clinical dataset is that of clinical dialogues which are especially sensitive and difficult to collect, and as such are commonly synthesized.
While such synthetic datasets have been shown to be sufficient in some situations, little theory exists to inform how they may be best used and generalized to new applications. In this paper, we provide an overview of how synthetic datasets are created, evaluated and being used for dialogue related tasks in the medical domain. Additionally, we propose a novel typology for use in classifying types and degrees of data synthesis, to facilitate comparison and evaluation. 

---
# Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis 

**Authors**: Albérick Euraste Djiré, Abdoul Kader Kaboré, Earl T. Barr, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2505.03019)  

**Abstract**: While Large Language Models (LLMs) achieve remarkable performance through training on massive datasets, they can exhibit concerning behaviors such as verbatim reproduction of training data rather than true generalization. This memorization phenomenon raises significant concerns about data privacy, intellectual property rights, and the reliability of model evaluations. This paper introduces PEARL, a novel approach for detecting memorization in LLMs. PEARL assesses how sensitive an LLM's performance is to input perturbations, enabling memorization detection without requiring access to the model's internals. We investigate how input perturbations affect the consistency of outputs, enabling us to distinguish between true generalization and memorization. Our findings, following extensive experiments on the Pythia open model, provide a robust framework for identifying when the model simply regurgitates learned information. Applied on the GPT 4o models, the PEARL framework not only identified cases of memorization of classic texts from the Bible or common code from HumanEval but also demonstrated that it can provide supporting evidence that some data, such as from the New York Times news articles, were likely part of the training data of a given model. 

---
# Lesion-Aware Generative Artificial Intelligence for Virtual Contrast-Enhanced Mammography in Breast Cancer 

**Authors**: Aurora Rofena, Arianna Manchia, Claudia Lucia Piccolo, Bruno Beomonte Zobel, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03018)  

**Abstract**: Contrast-Enhanced Spectral Mammography (CESM) is a dual-energy mammographic technique that improves lesion visibility through the administration of an iodinated contrast agent. It acquires both a low-energy image, comparable to standard mammography, and a high-energy image, which are then combined to produce a dual-energy subtracted image highlighting lesion contrast enhancement. While CESM offers superior diagnostic accuracy compared to standard mammography, its use entails higher radiation exposure and potential side effects associated with the contrast medium. To address these limitations, we propose Seg-CycleGAN, a generative deep learning framework for Virtual Contrast Enhancement in CESM. The model synthesizes high-fidelity dual-energy subtracted images from low-energy images, leveraging lesion segmentation maps to guide the generative process and improve lesion reconstruction. Building upon the standard CycleGAN architecture, Seg-CycleGAN introduces localized loss terms focused on lesion areas, enhancing the synthesis of diagnostically relevant regions. Experiments on the CESM@UCBM dataset demonstrate that Seg-CycleGAN outperforms the baseline in terms of PSNR and SSIM, while maintaining competitive MSE and VIF. Qualitative evaluations further confirm improved lesion fidelity in the generated images. These results suggest that segmentation-aware generative models offer a viable pathway toward contrast-free CESM alternatives. 

---
# RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale 

**Authors**: Daniel Goldstein, Eric Alcaide, Janna Lu, Eugene Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2505.03005)  

**Abstract**: We present Rapid Attention Distillation to Linear Attention Decoders at Scale (RADLADS), a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant architectures, and models converted from popular Qwen2.5 open source models in 7B, 32B, and 72B sizes. Our conversion process requires only 350-700M tokens, less than 0.005% of the token count used to train the original teacher models. Converting to our 72B linear attention model costs less than \$2,000 USD at today's prices, yet quality at inference remains close to the original transformer. These models achieve state-of-the-art downstream performance across a set of standard benchmarks for linear attention models of their size. We release all our models on HuggingFace under the Apache 2.0 license, with the exception of our 72B models which are also governed by the Qwen License Agreement.
Models at this https URL Training Code at this https URL 

---
# Generating Narrated Lecture Videos from Slides with Synchronized Highlights 

**Authors**: Alexander Holmberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.02966)  

**Abstract**: Turning static slides into engaging video lectures takes considerable time and effort, requiring presenters to record explanations and visually guide their audience through the material. We introduce an end-to-end system designed to automate this process entirely. Given a slide deck, this system synthesizes a video lecture featuring AI-generated narration synchronized precisely with dynamic visual highlights. These highlights automatically draw attention to the specific concept being discussed, much like an effective presenter would. The core technical contribution is a novel highlight alignment module. This module accurately maps spoken phrases to locations on a given slide using diverse strategies (e.g., Levenshtein distance, LLM-based semantic analysis) at selectable granularities (line or word level) and utilizes timestamp-providing Text-to-Speech (TTS) for timing synchronization. We demonstrate the system's effectiveness through a technical evaluation using a manually annotated slide dataset with 1000 samples, finding that LLM-based alignment achieves high location accuracy (F1 > 92%), significantly outperforming simpler methods, especially on complex, math-heavy content. Furthermore, the calculated generation cost averages under $1 per hour of video, offering potential savings of two orders of magnitude compared to conservative estimates of manual production costs. This combination of high accuracy and extremely low cost positions this approach as a practical and scalable tool for transforming static slides into effective, visually-guided video lectures. 

---
# The Cognitive Foundations of Economic Exchange: A Modular Framework Grounded in Behavioral Evidence 

**Authors**: Egil Diau  

**Link**: [PDF](https://arxiv.org/pdf/2505.02945)  

**Abstract**: A key challenge in multi-agent AI is modeling social cooperation under realistic behavioral constraints. Many foundational concepts in economics and ethics such as "trust" or "morality" are often defined informally, without operational criteria or cognitive grounding, which limits their testability and implementation in artificial agents. Drawing on converging empirical evidence from primate behavior, infant cognition, and economic anthropology, we propose a conceptual framework composed of three cognitively minimal mechanisms: individual recognition, reciprocal credence, and cost return sensitivity. This framework reframes trust as a graded cognitive expectation, providing a simulateable basis for reciprocal exchange in artificial agents, and enabling the bottom-up emergence of scalable cooperation and institutional dynamics. 

---
# The Art of Repair: Optimizing Iterative Program Repair with Instruction-Tuned Models 

**Authors**: Fernando Vallecillos Ruiz, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02931)  

**Abstract**: Automatic program repair (APR) aims to reduce the manual efforts required to identify and fix errors in source code. Before the rise of LLM-based agents, a common strategy was to increase the number of generated patches, sometimes to the thousands, to achieve better repair results on benchmarks. More recently, self-iterative capabilities enabled LLMs to refine patches over multiple rounds guided by feedback. However, literature often focuses on many iterations and disregards different numbers of outputs.
We investigate an APR pipeline that balances these two approaches, the generation of multiple outputs and multiple rounds of iteration, while imposing a limit of 10 total patches per bug. We apply three SOTA instruction-tuned LLMs - DeepSeekCoder-Instruct, Codellama-Instruct, Llama3.1-Instruct - to the APR task. We further fine-tune each model on an APR dataset with three sizes (1K, 30K, 65K) and two techniques (Full Fine-Tuning and LoRA), allowing us to assess their repair capabilities on two APR benchmarks: HumanEval-Java and Defects4J.
Our results show that by using only a fraction (<1%) of the fine-tuning dataset, we can achieve improvements of up to 78% in the number of plausible patches generated, challenging prior studies that reported limited gains using Full Fine-Tuning. However, we find that exceeding certain thresholds leads to diminishing outcomes, likely due to overfitting. Moreover, we show that base models greatly benefit from creating patches in an iterative fashion rather than generating them all at once. In addition, the benefit of iterative strategies becomes more pronounced in complex benchmarks. Even fine-tuned models, while benefiting less from iterations, still gain advantages, particularly on complex benchmarks. The research underscores the need for balanced APR strategies that combine multi-output generation and iterative refinement. 

---
# Early Prediction of Sepsis: Feature-Aligned Transfer Learning 

**Authors**: Oyindolapo O. Komolafe, Zhimin Mei, David Morales Zarate, Gregory William Spangenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.02889)  

**Abstract**: Sepsis is a life threatening medical condition that occurs when the body has an extreme response to infection, leading to widespread inflammation, organ failure, and potentially death. Because sepsis can worsen rapidly, early detection is critical to saving lives. However, current diagnostic methods often identify sepsis only after significant damage has already occurred. Our project aims to address this challenge by developing a machine learning based system to predict sepsis in its early stages, giving healthcare providers more time to intervene.
A major problem with existing models is the wide variability in the patient information or features they use, such as heart rate, temperature, and lab results. This inconsistency makes models difficult to compare and limits their ability to work across different hospitals and settings. To solve this, we propose a method called Feature Aligned Transfer Learning (FATL), which identifies and focuses on the most important and commonly reported features across multiple studies, ensuring the model remains consistent and clinically relevant.
Most existing models are trained on narrow patient groups, leading to population bias. FATL addresses this by combining knowledge from models trained on diverse populations, using a weighted approach that reflects each models contribution. This makes the system more generalizable and effective across different patient demographics and clinical environments. FATL offers a practical and scalable solution for early sepsis detection, particularly in hospitals with limited resources, and has the potential to improve patient outcomes, reduce healthcare costs, and support more equitable healthcare delivery. 

---
# When Your Own Output Becomes Your Training Data: Noise-to-Meaning Loops and a Formal RSI Trigger 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.02888)  

**Abstract**: We present Noise-to-Meaning Recursive Self-Improvement (N2M-RSI), a minimal formal model showing that once an AI agent feeds its own outputs back as inputs and crosses an explicit information-integration threshold, its internal complexity will grow without bound under our assumptions. The framework unifies earlier ideas on self-prompting large language models, Gödelian self-reference, and AutoML, yet remains implementation-agnostic. The model furthermore scales naturally to interacting swarms of agents, hinting at super-linear effects once communication among instances is permitted. For safety reasons, we omit system-specific implementation details and release only a brief, model-agnostic toy prototype in Appendix C. 

---
# CreoPep: A Universal Deep Learning Framework for Target-Specific Peptide Design and Optimization 

**Authors**: Cheng Ge, Han-Shen Tae, Zhenqiang Zhang, Lu Lu, Zhijie Huang, Yilin Wang, Tao Jiang, Wenqing Cai, Shan Chang, David J. Adams, Rilei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02887)  

**Abstract**: Target-specific peptides, such as conotoxins, exhibit exceptional binding affinity and selectivity toward ion channels and receptors. However, their therapeutic potential remains underutilized due to the limited diversity of natural variants and the labor-intensive nature of traditional optimization strategies. Here, we present CreoPep, a deep learning-based conditional generative framework that integrates masked language modeling with a progressive masking scheme to design high-affinity peptide mutants while uncovering novel structural motifs. CreoPep employs an integrative augmentation pipeline, combining FoldX-based energy screening with temperature-controlled multinomial sampling, to generate structurally and functionally diverse peptides that retain key pharmacological properties. We validate this approach by designing conotoxin inhibitors targeting the $\alpha$7 nicotinic acetylcholine receptor, achieving submicromolar potency in electrophysiological assays. Structural analysis reveals that CreoPep-generated variants engage in both conserved and novel binding modes, including disulfide-deficient forms, thus expanding beyond conventional design paradigms. Overall, CreoPep offers a robust and generalizable platform that bridges computational peptide design with experimental validation, accelerating the discovery of next-generation peptide therapeutics. 

---
# Taskmaster Deconstructed: A Quantitative Look at Tension, Volatility, and Viewer Ratings 

**Authors**: David H. Silver  

**Link**: [PDF](https://arxiv.org/pdf/2505.02886)  

**Abstract**: Taskmaster is a British television show that combines comedic performance with a formal scoring system. Despite the appearance of structured competition, it remains unclear whether scoring dynamics contribute meaningfully to audience engagement. We conducted a statistical analysis of 162 episodes across 18 series, using fifteen episode-level metrics to quantify rank volatility, point spread, lead changes, and winner dominance. None of these metrics showed a significant association with IMDb ratings, even after controlling for series effects. Long-term trends suggest that average points have increased over time, while volatility has slightly declined and rank spread has remained stable. These patterns indicate an attempt to enhance competitive visibility without altering the show's structural equilibrium. We also analyzed contestant rank trajectories and identified five recurring archetypes describing performance styles. These patterns suggest that viewer interest is shaped more by contestant behavior than by game mechanics. 

---
# Unlearning vs. Obfuscation: Are We Truly Removing Knowledge? 

**Authors**: Guangzhi Sun, Potsawee Manakul, Xiao Zhan, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.02884)  

**Abstract**: Unlearning has emerged as a critical capability for large language models (LLMs) to support data privacy, regulatory compliance, and ethical AI deployment. Recent techniques often rely on obfuscation by injecting incorrect or irrelevant information to suppress knowledge. Such methods effectively constitute knowledge addition rather than true removal, often leaving models vulnerable to probing. In this paper, we formally distinguish unlearning from obfuscation and introduce a probing-based evaluation framework to assess whether existing approaches genuinely remove targeted information. Moreover, we propose DF-MCQ, a novel unlearning method that flattens the model predictive distribution over automatically generated multiple-choice questions using KL-divergence, effectively removing knowledge about target individuals and triggering appropriate refusal behaviour. Experimental results demonstrate that DF-MCQ achieves unlearning with over 90% refusal rate and a random choice-level uncertainty that is much higher than obfuscation on probing questions. 

---
# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code 

**Authors**: Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Ohi, Masaki Kawamura, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.02881)  

**Abstract**: The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains. 

---
# A Wireless Collaborated Inference Acceleration Framework for Plant Disease Recognition 

**Authors**: Hele Zhu, Xinyi Huang, Haojia Gao, Mengfei Jiang, Haohua Que, Lei Mu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02877)  

**Abstract**: Plant disease is a critical factor affecting agricultural production. Traditional manual recognition methods face significant drawbacks, including low accuracy, high costs, and inefficiency. Deep learning techniques have demonstrated significant benefits in identifying plant diseases, but they still face challenges such as inference delays and high energy consumption. Deep learning algorithms are difficult to run on resource-limited embedded devices. Offloading these models to cloud servers is confronted with the restriction of communication bandwidth, and all of these factors will influence the inference's efficiency. We propose a collaborative inference framework for recognizing plant diseases between edge devices and cloud servers to enhance inference speed. The DNN model for plant disease recognition is pruned through deep reinforcement learning to improve the inference speed and reduce energy consumption. Then the optimal split point is determined by a greedy strategy to achieve the best collaborated inference acceleration. Finally, the system for collaborative inference acceleration in plant disease recognition has been implemented using Gradio to facilitate friendly human-machine interaction. Experiments indicate that the proposed collaborative inference framework significantly increases inference speed while maintaining acceptable recognition accuracy, offering a novel solution for rapidly diagnosing and preventing plant diseases. 

---
# Uncertainty Quantification for Machine Learning in Healthcare: A Survey 

**Authors**: L. Julián Lechuga López, Shaza Elsharief, Dhiyaa Al Jorf, Firas Darwish, Congbo Ma, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2505.02874)  

**Abstract**: Uncertainty Quantification (UQ) is pivotal in enhancing the robustness, reliability, and interpretability of Machine Learning (ML) systems for healthcare, optimizing resources and improving patient care. Despite the emergence of ML-based clinical decision support tools, the lack of principled quantification of uncertainty in ML models remains a major challenge. Current reviews have a narrow focus on analyzing the state-of-the-art UQ in specific healthcare domains without systematically evaluating method efficacy across different stages of model development, and despite a growing body of research, its implementation in healthcare applications remains limited. Therefore, in this survey, we provide a comprehensive analysis of current UQ in healthcare, offering an informed framework that highlights how different methods can be integrated into each stage of the ML pipeline including data processing, training and evaluation. We also highlight the most popular methods used in healthcare and novel approaches from other domains that hold potential for future adoption in the medical context. We expect this study will provide a clear overview of the challenges and opportunities of implementing UQ in the ML pipeline for healthcare, guiding researchers and practitioners in selecting suitable techniques to enhance the reliability, safety and trust from patients and clinicians on ML-driven healthcare solutions. 

---
# Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading 

**Authors**: Cfir Avraham Hadar, Omer Shubi, Yoav Meiri, Yevgeni Berzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.02872)  

**Abstract**: When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you only care about the question ``but does it work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask, for the first time, whether open-ended reading goals can be automatically decoded from eye movements in reading. To address this question, we introduce goal classification and goal reconstruction tasks and evaluation frameworks, and use large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks. We develop and compare several discriminative and generative multimodal LLMs that combine eye movements and text for goal classification and goal reconstruction. Our experiments show considerable success on both tasks, suggesting that LLMs can extract valuable information about the readers' text-specific goals from eye movements. 

---
# Accelerating Large Language Model Reasoning via Speculative Search 

**Authors**: Zhihai Wang, Jie Wang, Jilai Pan, Xilin Xia, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02865)  

**Abstract**: Tree-search-based reasoning methods have significantly enhanced the reasoning capability of large language models (LLMs) by facilitating the exploration of multiple intermediate reasoning steps, i.e., thoughts. However, these methods suffer from substantial inference latency, as they have to generate numerous reasoning thoughts, severely limiting LLM applicability. To address this challenge, we propose a novel Speculative Search (SpecSearch) framework that significantly accelerates LLM reasoning by optimizing thought generation. Specifically, SpecSearch utilizes a small model to strategically collaborate with a large model at both thought and token levels, efficiently generating high-quality reasoning thoughts. The major pillar of SpecSearch is a novel quality-preserving rejection mechanism, which effectively filters out thoughts whose quality falls below that of the large model's outputs. Moreover, we show that SpecSearch preserves comparable reasoning quality to the large model. Experiments on both the Qwen and Llama models demonstrate that SpecSearch significantly outperforms state-of-the-art approaches, achieving up to 2.12$\times$ speedup with comparable reasoning quality. 

---
# Understanding University Students' Use of Generative AI: The Roles of Demographics and Personality Traits 

**Authors**: Newnew Deng, Edward Jiusi Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.02863)  

**Abstract**: The use of generative AI (GAI) among university students is rapidly increasing, yet empirical research on students' GAI use and the factors influencing it remains limited. To address this gap, we surveyed 363 undergraduate and graduate students in the United States, examining their GAI usage and how it relates to demographic variables and personality traits based on the Big Five model (i.e., extraversion, agreeableness, conscientiousness, and emotional stability, and intellect/imagination). Our findings reveal: (a) Students in higher academic years are more inclined to use GAI and prefer it over traditional resources. (b) Non-native English speakers use and adopt GAI more readily than native speakers. (c) Compared to White, Asian students report higher GAI usage, perceive greater academic benefits, and express a stronger preference for it. Similarly, Black students report a more positive impact of GAI on their academic performance. Personality traits also play a significant role in shaping perceptions and usage of GAI. After controlling demographic factors, we found that personality still significantly predicts GAI use and attitudes: (a) Students with higher conscientiousness use GAI less. (b) Students who are higher in agreeableness perceive a less positive impact of GAI on academic performance and express more ethical concerns about using it for academic work. (c) Students with higher emotional stability report a more positive impact of GAI on learning and fewer concerns about its academic use. (d) Students with higher extraversion show a stronger preference for GAI over traditional resources. (e) Students with higher intellect/imagination tend to prefer traditional resources. These insights highlight the need for universities to provide personalized guidance to ensure students use GAI effectively, ethically, and equitably in their academic pursuits. 

---
# Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs 

**Authors**: Haoming Yang, Ke Ma, Xiaojun Jia, Yingfei Sun, Qianqian Xu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02862)  

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies. 

---
# Neural Orchestration for Multi-Agent Systems: A Deep Learning Framework for Optimal Agent Selection in Multi-Domain Task Environments 

**Authors**: Kushagra Agrawal, Nisharg Nargund  

**Link**: [PDF](https://arxiv.org/pdf/2505.02861)  

**Abstract**: Multi-agent systems (MAS) are foundational in simulating complex real-world scenarios involving autonomous, interacting entities. However, traditional MAS architectures often suffer from rigid coordination mechanisms and difficulty adapting to dynamic tasks. We propose MetaOrch, a neural orchestration framework for optimal agent selection in multi-domain task environments. Our system implements a supervised learning approach that models task context, agent histories, and expected response quality to select the most appropriate agent for each task. A novel fuzzy evaluation module scores agent responses along completeness, relevance, and confidence dimensions, generating soft supervision labels for training the orchestrator. Unlike previous methods that hard-code agent-task mappings, MetaOrch dynamically predicts the most suitable agent while estimating selection confidence. Experiments in simulated environments with heterogeneous agents demonstrate that our approach achieves 86.3% selection accuracy, significantly outperforming baseline strategies including random selection and round-robin scheduling. The modular architecture emphasizes extensibility, allowing agents to be registered, updated, and queried independently. Results suggest that neural orchestration offers a powerful approach to enhancing the autonomy, interpretability, and adaptability of multi-agent systems across diverse task domains. 

---
# Enhancing ML Model Interpretability: Leveraging Fine-Tuned Large Language Models for Better Understanding of AI 

**Authors**: Jonas Bokstaller, Julia Altheimer, Julian Dormehl, Alina Buss, Jasper Wiltfang, Johannes Schneider, Maximilian Röglinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.02859)  

**Abstract**: Across various sectors applications of eXplainableAI (XAI) gained momentum as the increasing black-boxedness of prevailing Machine Learning (ML) models became apparent. In parallel, Large Language Models (LLMs) significantly developed in their abilities to understand human language and complex patterns. By combining both, this paper presents a novel reference architecture for the interpretation of XAI through an interactive chatbot powered by a fine-tuned LLM. We instantiate the reference architecture in the context of State-of-Health (SoH) prediction for batteries and validate its design in multiple evaluation and demonstration rounds. The evaluation indicates that the implemented prototype enhances the human interpretability of ML, especially for users with less experience with XAI. 

---
# AI Education in a Mirror: Challenges Faced by Academic and Industry Experts 

**Authors**: Mahir Akgun, Hadi Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2505.02856)  

**Abstract**: As Artificial Intelligence (AI) technologies continue to evolve, the gap between academic AI education and real-world industry challenges remains an important area of investigation. This study provides preliminary insights into challenges AI professionals encounter in both academia and industry, based on semi-structured interviews with 14 AI experts - eight from industry and six from academia. We identify key challenges related to data quality and availability, model scalability, practical constraints, user behavior, and explainability. While both groups experience data and model adaptation difficulties, industry professionals more frequently highlight deployment constraints, resource limitations, and external dependencies, whereas academics emphasize theoretical adaptation and standardization issues. These exploratory findings suggest that AI curricula could better integrate real-world complexities, software engineering principles, and interdisciplinary learning, while recognizing the broader educational goals of building foundational and ethical reasoning skills. 

---
# Ensuring Reproducibility in Generative AI Systems for General Use Cases: A Framework for Regression Testing and Open Datasets 

**Authors**: Masumi Morishige, Ryo Koshihara  

**Link**: [PDF](https://arxiv.org/pdf/2505.02854)  

**Abstract**: Reproducibility and reliability remain pressing challenges for generative AI systems whose behavior can drift with each model update or prompt revision. We introduce GPR-bench, a lightweight, extensible benchmark that operationalizes regression testing for general purpose use cases. GPR-bench couples an open, bilingual (English and Japanese) dataset covering eight task categories (e.g., text generation, code generation, and information retrieval) and 10 scenarios in each task categories (80 total test cases for each language) with an automated evaluation pipeline that employs "LLM-as-a-Judge" scoring of correctness and conciseness. Experiments across three recent model versions - gpt-4o-mini, o3-mini, and o4-mini - and two prompt configurations (default versus concise-writing instruction) reveal heterogeneous quality. Our results show that newer models generally improve correctness, but the differences are modest and not statistically significant, suggesting that GPR-bench may not be sufficiently challenging to differentiate between recent model versions. In contrast, the concise-writing instruction significantly enhances conciseness (+12.37 pp, Mann-Whitney U test: p < 0.001, effect size r = 0.2995) with minimal degradations on accuracy (-1.7 pp), demonstrating the effectiveness of prompt engineering. Released under the MIT License, GPR- bench lowers the barrier to initiating reproducibility monitoring and provides a foundation for community-driven extensions, while also raising important considerations about benchmark design for rapidly evolving language models. 

---
# A Computational Model of Inclusive Pedagogy: From Understanding to Application 

**Authors**: Francesco Balzan, Pedro P. Santos, Maurizio Gabbrielli, Mahault Albarracin, Manuel Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2505.02853)  

**Abstract**: Human education transcends mere knowledge transfer, it relies on co-adaptation dynamics -- the mutual adjustment of teaching and learning strategies between agents. Despite its centrality, computational models of co-adaptive teacher-student interactions (T-SI) remain underdeveloped. We argue that this gap impedes Educational Science in testing and scaling contextual insights across diverse settings, and limits the potential of Machine Learning systems, which struggle to emulate and adaptively support human learning processes. To address this, we present a computational T-SI model that integrates contextual insights on human education into a testable framework. We use the model to evaluate diverse T-SI strategies in a realistic synthetic classroom setting, simulating student groups with unequal access to sensory information. Results show that strategies incorporating co-adaptation principles (e.g., bidirectional agency) outperform unilateral approaches (i.e., where only the teacher or the student is active), improving the learning outcomes for all learning types. Beyond the testing and scaling of context-dependent educational insights, our model enables hypothesis generation in controlled yet adaptable environments. This work bridges non-computational theories of human education with scalable, inclusive AI in Education systems, providing a foundation for equitable technologies that dynamically adapt to learner needs. 

---
# 30DayGen: Leveraging LLMs to Create a Content Corpus for Habit Formation 

**Authors**: Franklin Zhang, Sonya Zhang, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2505.02851)  

**Abstract**: In this paper, we present 30 Day Me, a habit formation application that leverages Large Language Models (LLMs) to help users break down their goals into manageable, actionable steps and track their progress. Central to the app is the 30DAYGEN system, which generates 3,531 unique 30-day challenges sourced from over 15K webpages, and enables runtime search of challenge ideas aligned with user-defined goals. We showcase how LLMs can be harnessed to rapidly construct domain specific content corpora for behavioral and educational purposes, and propose a practical pipeline that incorporates effective LLM enhanced approaches for content generation and semantic deduplication. 

---
# Harnessing Structured Knowledge: A Concept Map-Based Approach for High-Quality Multiple Choice Question Generation with Effective Distractors 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Diksha Seth, Ananya Thakur, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2505.02850)  

**Abstract**: Generating high-quality MCQs, especially those targeting diverse cognitive levels and incorporating common misconceptions into distractor design, is time-consuming and expertise-intensive, making manual creation impractical at scale. Current automated approaches typically generate questions at lower cognitive levels and fail to incorporate domain-specific misconceptions. This paper presents a hierarchical concept map-based framework that provides structured knowledge to guide LLMs in generating MCQs with distractors. We chose high-school physics as our test domain and began by developing a hierarchical concept map covering major Physics topics and their interconnections with an efficient database design. Next, through an automated pipeline, topic-relevant sections of these concept maps are retrieved to serve as a structured context for the LLM to generate questions and distractors that specifically target common misconceptions. Lastly, an automated validation is completed to ensure that the generated MCQs meet the requirements provided. We evaluate our framework against two baseline approaches: a base LLM and a RAG-based generation. We conducted expert evaluations and student assessments of the generated MCQs. Expert evaluation shows that our method significantly outperforms the baseline approaches, achieving a success rate of 75.20% in meeting all quality criteria compared to approximately 37% for both baseline methods. Student assessment data reveal that our concept map-driven approach achieved a significantly lower guess success rate of 28.05% compared to 37.10% for the baselines, indicating a more effective assessment of conceptual understanding. The results demonstrate that our concept map-based approach enables robust assessment across cognitive levels and instant identification of conceptual gaps, facilitating faster feedback loops and targeted interventions at scale. 

---
# Enhancing tutoring systems by leveraging tailored promptings and domain knowledge with Large Language Models 

**Authors**: Mohsen Balavar, Wenli Yang, David Herbert, Soonja Yeom  

**Link**: [PDF](https://arxiv.org/pdf/2505.02849)  

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning have reignited interest in their impact on Computer-based Learning (CBL). AI-driven tools like ChatGPT and Intelligent Tutoring Systems (ITS) have enhanced learning experiences through personalisation and flexibility. ITSs can adapt to individual learning needs and provide customised feedback based on a student's performance, cognitive state, and learning path. Despite these advances, challenges remain in accommodating diverse learning styles and delivering real-time, context-aware feedback. Our research aims to address these gaps by integrating skill-aligned feedback via Retrieval Augmented Generation (RAG) into prompt engineering for Large Language Models (LLMs) and developing an application to enhance learning through personalised tutoring in a computer science programming context. The pilot study evaluated a proposed system using three quantitative metrics: readability score, response time, and feedback depth, across three programming tasks of varying complexity. The system successfully sorted simulated students into three skill-level categories and provided context-aware feedback. This targeted approach demonstrated better effectiveness and adaptability compared to general methods. 

---
# Aligning Large Language Models with Healthcare Stakeholders: A Pathway to Trustworthy AI Integration 

**Authors**: Kexin Ding, Mu Zhou, Akshay Chaudhari, Shaoting Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2505.02848)  

**Abstract**: The wide exploration of large language models (LLMs) raises the awareness of alignment between healthcare stakeholder preferences and model outputs. This alignment becomes a crucial foundation to empower the healthcare workflow effectively, safely, and responsibly. Yet the varying behaviors of LLMs may not always match with healthcare stakeholders' knowledge, demands, and values. To enable a human-AI alignment, healthcare stakeholders will need to perform essential roles in guiding and enhancing the performance of LLMs. Human professionals must participate in the entire life cycle of adopting LLM in healthcare, including training data curation, model training, and inference. In this review, we discuss the approaches, tools, and applications of alignments between healthcare stakeholders and LLMs. We demonstrate that LLMs can better follow human values by properly enhancing healthcare knowledge integration, task understanding, and human guidance. We provide outlooks on enhancing the alignment between humans and LLMs to build trustworthy real-world healthcare applications. 

---
# Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models 

**Authors**: Bang Zhang, Ruotian Ma, Qingxuan Jiang, Peisong Wang, Jiaqi Chen, Zheng Xie, Xingyu Chen, Yue Wang, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02847)  

**Abstract**: Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM's higher-order social cognition. SAGE instantiates a Sentient Agent that simulates human-like emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett-Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4x) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents. 

---
# The Precautionary Principle and the Innovation Principle: Incompatible Guides for AI Innovation Governance? 

**Authors**: Kim Kaivanto  

**Link**: [PDF](https://arxiv.org/pdf/2505.02846)  

**Abstract**: In policy debates concerning the governance and regulation of Artificial Intelligence (AI), both the Precautionary Principle (PP) and the Innovation Principle (IP) are advocated by their respective interest groups. Do these principles offer wholly incompatible and contradictory guidance? Does one necessarily negate the other? I argue here that provided attention is restricted to weak-form PP and IP, the answer to both of these questions is "No." The essence of these weak formulations is the requirement to fully account for type-I error costs arising from erroneously preventing the innovation's diffusion through society (i.e. mistaken regulatory red-lighting) as well as the type-II error costs arising from erroneously allowing the innovation to diffuse through society (i.e. mistaken regulatory green-lighting). Within the Signal Detection Theory (SDT) model developed here, weak-PP red-light (weak-IP green-light) determinations are optimal for sufficiently small (large) ratios of expected type-I to type-II error costs. For intermediate expected cost ratios, an amber-light 'wait-and-monitor' policy is optimal. Regulatory sandbox instruments allow AI testing and experimentation to take place within a structured environment of limited duration and societal scale, whereby the expected cost ratio falls within the 'wait-and-monitor' range. Through sandboxing regulators and innovating firms learn more about the expected cost ratio, and what respective adaptations -- of regulation, of technical solution, of business model, or combination thereof, if any -- are needed to keep the ratio out of the weak-PP red-light zone. 

---
# Physical foundations for trustworthy medical imaging: a review for artificial intelligence researchers 

**Authors**: Miriam Cobo, David Corral Fontecha, Wilson Silva, Lara Lloret Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2505.02843)  

**Abstract**: Artificial intelligence in medical imaging has seen unprecedented growth in the last years, due to rapid advances in deep learning and computing resources. Applications cover the full range of existing medical imaging modalities, with unique characteristics driven by the physics of each technique. Yet, artificial intelligence professionals entering the field, and even experienced developers, often lack a comprehensive understanding of the physical principles underlying medical image acquisition, which hinders their ability to fully leverage its potential. The integration of physics knowledge into artificial intelligence algorithms enhances their trustworthiness and robustness in medical imaging, especially in scenarios with limited data availability. In this work, we review the fundamentals of physics in medical images and their impact on the latest advances in artificial intelligence, particularly, in generative models and reconstruction algorithms. Finally, we explore the integration of physics knowledge into physics-inspired machine learning models, which leverage physics-based constraints to enhance the learning of medical imaging features. 

---
# Snakemaker: Seamlessly transforming ad-hoc analyses into sustainable Snakemake workflows with generative AI 

**Authors**: Marco Masera, Alessandro Leone, Johannes Köster, Ivan Molineris  

**Link**: [PDF](https://arxiv.org/pdf/2505.02841)  

**Abstract**: Reproducibility and sustainability present significant challenges in bioinformatics software development, where rapidly evolving tools and complex workflows often result in short-lived or difficult-to-adapt pipelines. This paper introduces Snakemaker, a tool that leverages generative AI to facilitate researchers build sustainable data analysis pipelines by converting unstructured code into well-defined Snakemake workflows. Snakemaker non-invasively tracks the work performed in the terminal by the researcher, analyzes execution patterns, and generates Snakemake workflows that can be integrated into existing pipelines. Snakemaker also supports the transformation of monolithic Ipython Notebooks into modular Snakemake pipelines, resolving the global state of the notebook into discrete, file-based interactions between rules. An integrated chat assistant provides users with fine-grained control through natural language instructions. Snakemaker generates high-quality Snakemake workflows by adhering to the best practices, including Conda environment tracking, generic rule generation and loop unrolling. By lowering the barrier between prototype and production-quality code, Snakemaker addresses a critical gap in computational reproducibility for bioinformatics research. 

---
