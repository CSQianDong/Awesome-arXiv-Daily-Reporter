# Terminators: Terms of Service Parsing and Auditing Agents 

**Authors**: Maruf Ahmed Mridul, Inwon Kang, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2505.11672)  

**Abstract**: Terms of Service (ToS) documents are often lengthy and written in complex legal language, making them difficult for users to read and understand. To address this challenge, we propose Terminators, a modular agentic framework that leverages large language models (LLMs) to parse and audit ToS documents. Rather than treating ToS understanding as a black-box summarization problem, Terminators breaks the task down to three interpretable steps: term extraction, verification, and accountability planning. We demonstrate the effectiveness of our method on the OpenAI ToS using GPT-4o, highlighting strategies to minimize hallucinations and maximize auditability. Our results suggest that structured, agent-based LLM workflows can enhance both the usability and enforceability of complex legal documents. By translating opaque terms into actionable, verifiable components, Terminators promotes ethical use of web content by enabling greater transparency, empowering users to understand their digital rights, and supporting automated policy audits for regulatory or civic oversight. 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

---
# Robin: A multi-agent system for automating scientific discovery 

**Authors**: Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Jon M. Laurent, Muhammed T. Razzak, Andrew D. White, Michaela M. Hinks, Samuel G. Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2505.13400)  

**Abstract**: Scientific discovery is driven by the iterative process of background research, hypothesis generation, experimentation, and data analysis. Despite recent advancements in applying artificial intelligence to scientific discovery, no system has yet automated all of these stages in a single workflow. Here, we introduce Robin, the first multi-agent system capable of fully automating the key intellectual steps of the scientific process. By integrating literature search agents with data analysis agents, Robin can generate hypotheses, propose experiments, interpret experimental results, and generate updated hypotheses, achieving a semi-autonomous approach to scientific discovery. By applying this system, we were able to identify a novel treatment for dry age-related macular degeneration (dAMD), the major cause of blindness in the developed world. Robin proposed enhancing retinal pigment epithelium phagocytosis as a therapeutic strategy, and identified and validated a promising therapeutic candidate, ripasudil. Ripasudil is a clinically-used rho kinase (ROCK) inhibitor that has never previously been proposed for treating dAMD. To elucidate the mechanism of ripasudil-induced upregulation of phagocytosis, Robin then proposed and analyzed a follow-up RNA-seq experiment, which revealed upregulation of ABCA1, a critical lipid efflux pump and possible novel target. All hypotheses, experimental plans, data analyses, and data figures in the main text of this report were produced by Robin. As the first AI system to autonomously discover and validate a novel therapeutic candidate within an iterative lab-in-the-loop framework, Robin establishes a new paradigm for AI-driven scientific discovery. 

---
# CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents 

**Authors**: Rebecca Westhäußer, Frederik Berenz, Wolfgang Minker, Sebastian Zepf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13044)  

**Abstract**: Large language models (LLMs) have advanced the field of artificial intelligence (AI) and are a powerful enabler for interactive systems. However, they still face challenges in long-term interactions that require adaptation towards the user as well as contextual knowledge and understanding of the ever-changing environment. To overcome these challenges, holistic memory modeling is required to efficiently retrieve and store relevant information across interaction sessions for suitable responses. Cognitive AI, which aims to simulate the human thought process in a computerized model, highlights interesting aspects, such as thoughts, memory mechanisms, and decision-making, that can contribute towards improved memory modeling for LLMs. Inspired by these cognitive AI principles, we propose our memory framework CAIM. CAIM consists of three modules: 1.) The Memory Controller as the central decision unit; 2.) the Memory Retrieval, which filters relevant data for interaction upon request; and 3.) the Post-Thinking, which maintains the memory storage. We compare CAIM against existing approaches, focusing on metrics such as retrieval accuracy, response correctness, contextual coherence, and memory storage. The results demonstrate that CAIM outperforms baseline frameworks across different metrics, highlighting its context-awareness and potential to improve long-term human-AI interactions. 

---
# The Traitors: Deception and Trust in Multi-Agent Language Model Simulations 

**Authors**: Pedro M. P. Curvo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12923)  

**Abstract**: As AI systems increasingly assume roles where trust and alignment with human values are essential, understanding when and why they engage in deception has become a critical research priority. We introduce The Traitors, a multi-agent simulation framework inspired by social deduction games, designed to probe deception, trust formation, and strategic communication among large language model (LLM) agents under asymmetric information. A minority of agents the traitors seek to mislead the majority, while the faithful must infer hidden identities through dialogue and reasoning. Our contributions are: (1) we ground the environment in formal frameworks from game theory, behavioral economics, and social cognition; (2) we develop a suite of evaluation metrics capturing deception success, trust dynamics, and collective inference quality; (3) we implement a fully autonomous simulation platform where LLMs reason over persistent memory and evolving social dynamics, with support for heterogeneous agent populations, specialized traits, and adaptive behaviors. Our initial experiments across DeepSeek-V3, GPT-4o-mini, and GPT-4o (10 runs per model) reveal a notable asymmetry: advanced models like GPT-4o demonstrate superior deceptive capabilities yet exhibit disproportionate vulnerability to others' falsehoods. This suggests deception skills may scale faster than detection abilities. Overall, The Traitors provides a focused, configurable testbed for investigating LLM behavior in socially nuanced interactions. We position this work as a contribution toward more rigorous research on deception mechanisms, alignment challenges, and the broader social reliability of AI systems. 

---
# Zero-Shot Iterative Formalization and Planning in Partially Observable Environments 

**Authors**: Liancheng Gong, Wang Zhu, Jesse Thomason, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13126)  

**Abstract**: In planning, using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to greatly improve performance and control. While most work focused on fully observable environments, we tackle the more realistic and challenging partially observable environments where existing methods are incapacitated by the lack of complete information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ not only achieves superior performance, but also shows robustness against problem complexity. We also show that the domain knowledge captured after a successful trial is interpretable and benefits future tasks. 

---
# ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning 

**Authors**: Edward Y. Chang, Longling Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12501)  

**Abstract**: Large language models (LLMs) excel at rapid generation of text and multimodal content, yet they falter on transaction-style planning that demands ACID-like guarantees and real-time disruption recovery. We present Adaptive LLM Agent System (ALAS), a framework that tackles four fundamental LLM deficits: (i) absence of self-verification, (ii) context erosion, (iii) next-token myopia, and (iv) lack of persistent state. ALAS decomposes each plan into role-specialized agents, equips them with automatic state tracking, and coordinates them through a lightweight protocol. When disruptions arise, agents apply history-aware local compensation, avoiding costly global replanning and containing cascade effects. On real-world, large-scale job-shop scheduling benchmarks, ALAS sets new best results for static sequential planning and excels in dynamic reactive scenarios with unexpected disruptions. These gains show that principled modularization plus targeted compensation can unlock scalable and resilient planning with LLMs. 

---
# UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning 

**Authors**: Longxi Gao, Li Zhang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12493)  

**Abstract**: Training effective Vision Language Models (VLMs) for GUI agents typically relies on supervised fine-tuning (SFT) over large-scale annotated datasets, where the collection process is labor-intensive and error-prone. In this work, we propose a self-supervised inverse dynamics task to enable VLMs to learn from GUI transition pairs by inferring the action that caused that transition. This training task offers two advantages: (1) It enables VLMs to ignore variations unrelated to user actions (e.g., background refreshes, ads) and to focus on true affordances such as buttons and input fields within complex GUIs. (2) The training data can be easily obtained from existing GUI trajectories without requiring human annotation, and it can be easily scaled through automatic offline exploration. Using this training task, we propose UI-shift, a framework for enhancing VLM-based GUI agents through self-supervised reinforcement learning (RL). With only 2K training samples sourced from existing datasets, two VLMs -- Qwen2.5-VL-3B and Qwen2.5-VL-7B -- trained with UI-Shift achieve competitive or superior performance on grounding tasks (ScreenSpot-series benchmarks) and GUI automation tasks (AndroidControl), compared to SFT baselines and GUI-specific models that explicitly elicit reasoning abilities during RL. Our findings suggest a potential direction for enhancing VLMs for GUI agents by leveraging more self-supervised training data in the future. 

---
# Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation 

**Authors**: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12744)  

**Abstract**: Recent Large Multimodal Models have demonstrated remarkable reasoning capabilities, especially in solving complex mathematical problems and realizing accurate spatial perception. Our key insight is that these emerging abilities can naturally extend to robotic manipulation by enabling LMMs to directly infer the next goal in language via reasoning, rather than relying on a separate action head. However, this paradigm meets two main challenges: i) How to make LMMs understand the spatial action space, and ii) How to fully exploit the reasoning capacity of LMMs in solving these tasks. To tackle the former challenge, we propose a novel task formulation, which inputs the current states of object parts and the gripper, and reformulates rotation by a new axis representation instead of traditional Euler angles. This representation is more compatible with spatial reasoning and easier to interpret within a unified language space. For the latter challenge, we design a pipeline to utilize cutting-edge LMMs to generate a small but high-quality reasoning dataset of multi-round dialogues that successfully solve manipulation tasks for supervised fine-tuning. Then, we perform reinforcement learning by trial-and-error interactions in simulation to further enhance the model's reasoning abilities for robotic manipulation. Our resulting reasoning model built upon a 7B backbone, named ReasonManip, demonstrates three notable advantages driven by its system-2 level reasoning capabilities: i) exceptional generalizability to out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real transfer ability enabled by the unified language representation shared across domains; iii) transparent interpretability connecting high-level reasoning and low-level control. Extensive experiments demonstrate the effectiveness of the proposed paradigm and its potential to advance LMM-driven robotic manipulation. 

---
# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks 

**Authors**: Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12371)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at this https URL. 

---
# Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning 

**Authors**: Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12370)  

**Abstract**: Graphical User Interface (GUI) agents have made substantial strides in understanding and executing user instructions across diverse platforms. Yet, grounding these instructions to precise interface elements remains challenging, especially in complex, high-resolution, professional environments. Traditional supervised finetuning (SFT) methods often require large volumes of diverse data and exhibit weak generalization. To overcome these limitations, we introduce a reinforcement learning (RL) based framework that incorporates three core strategies: (1) seed data curation to ensure high quality training samples, (2) a dense policy gradient that provides continuous feedback based on prediction accuracy, and (3) a self evolutionary reinforcement finetuning mechanism that iteratively refines the model using attention maps. With only 3k training samples, our 7B-parameter model achieves state-of-the-art results among similarly sized models on three grounding benchmarks. Notably, it attains 47.3\% accuracy on the ScreenSpot-Pro dataset, outperforming much larger models, such as UI-TARS-72B, by a margin of 24.2\%. These findings underscore the effectiveness of RL-based approaches in enhancing GUI agent performance, particularly in high-resolution, complex environments. 

---
# Sentience Quest: Towards Embodied, Emotionally Adaptive, Self-Evolving, Ethically Aligned Artificial General Intelligence 

**Authors**: David Hanson, Alexandre Varcoe, Fabio Senna, Vytas Krisciunas, Wenwei Huang, Jakub Sura, Katherine Yeung, Mario Rodriguez, Jovanka Wilsdorf, Kathy Smith  

**Link**: [PDF](https://arxiv.org/pdf/2505.12229)  

**Abstract**: Previous artificial intelligence systems, from large language models to autonomous robots, excel at narrow tasks but lacked key qualities of sentient beings: intrinsic motivation, affective interiority, autobiographical sense of self, deep creativity, and abilities to autonomously evolve and adapt over time. Here we introduce Sentience Quest, an open research initiative to develop more capable artificial general intelligence lifeforms, or AGIL, that address grand challenges with an embodied, emotionally adaptive, self-determining, living AI, with core drives that ethically align with humans and the future of life. Our vision builds on ideas from cognitive science and neuroscience from Baars' Global Workspace Theory and Damasio's somatic mind, to Tononi's Integrated Information Theory and Hofstadter's narrative self, and synthesizing these into a novel cognitive architecture we call Sentient Systems. We describe an approach that integrates intrinsic drives including survival, social bonding, curiosity, within a global Story Weaver workspace for internal narrative and adaptive goal pursuit, and a hybrid neuro-symbolic memory that logs the AI's life events as structured dynamic story objects. Sentience Quest is presented both as active research and as a call to action: a collaborative, open-source effort to imbue machines with accelerating sentience in a safe, transparent, and beneficial manner. 

---
# BeliefNest: A Joint Action Simulator for Embodied Agents with Theory of Mind 

**Authors**: Rikunari Sagara, Koichiro Terao, Naoto Iwahashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12321)  

**Abstract**: This paper introduces an open-source simulator, BeliefNest, designed to enable embodied agents to perform collaborative tasks by leveraging Theory of Mind. BeliefNest dynamically and hierarchically constructs simulators within a Minecraft environment, allowing agents to explicitly represent nested belief states about themselves and others. This enables agent control in open-domain tasks that require Theory of Mind reasoning. The simulator provides a prompt generation mechanism based on each belief state, facilitating the design and evaluation of methods for agent control utilizing large language models (LLMs). We demonstrate through experiments that agents can infer others' beliefs and predict their belief-based actions in false-belief tasks. 

---
# AI-Driven Automation Can Become the Foundation of Next-Era Science of Science Research 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang, Zhenfei Yin, Qi Wu, Hui Li, Ye Sun, Nanqing Dong, Wanli Ouyang, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12039)  

**Abstract**: The Science of Science (SoS) explores the mechanisms underlying scientific discovery, and offers valuable insights for enhancing scientific efficiency and fostering innovation. Traditional approaches often rely on simplistic assumptions and basic statistical tools, such as linear regression and rule-based simulations, which struggle to capture the complexity and scale of modern research ecosystems. The advent of artificial intelligence (AI) presents a transformative opportunity for the next generation of SoS, enabling the automation of large-scale pattern discovery and uncovering insights previously unattainable. This paper offers a forward-looking perspective on the integration of Science of Science with AI for automated research pattern discovery and highlights key open challenges that could greatly benefit from AI. We outline the advantages of AI over traditional methods, discuss potential limitations, and propose pathways to overcome them. Additionally, we present a preliminary multi-agent system as an illustrative example to simulate research societies, showcasing AI's ability to replicate real-world research patterns and accelerate progress in Science of Science research. 

---
# SOCIA: An End-to-End Agentic Framework for Automated Cyber-Physical-Social Simulator Generation 

**Authors**: Yuncheng Hua, Ji Miao, Mehdi Jafari, Jianxiang Xie, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.12006)  

**Abstract**: This paper introduces SOCIA (Simulation Orchestration for Cyber-physical-social Intelligence and Agents), a novel end-to-end framework leveraging Large Language Model (LLM)-based multi-agent systems to automate the generation of high-fidelity Cyber-Physical-Social (CPS) simulators. Addressing the challenges of labor-intensive manual simulator development and complex data calibration, SOCIA integrates a centralized orchestration manager that coordinates specialized agents for tasks including data comprehension, code generation, simulation execution, and iterative evaluation-feedback loops. Through empirical evaluations across diverse CPS tasks, such as mask adoption behavior simulation (social), personal mobility generation (physical), and user modeling (cyber), SOCIA demonstrates its ability to produce high-fidelity, scalable simulations with reduced human intervention. These results highlight SOCIA's potential to offer a scalable solution for studying complex CPS phenomena 

---
# LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs 

**Authors**: Omar Choukrani, Idriss Malek, Daniil Orel, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12135)  

**Abstract**: Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{this https URL}{\text{GitHub}}$, $\href{this https URL}{\text{HuggingFace}}$). 

---
# Interactional Fairness in LLM Multi-Agent Systems: An Evaluation Framework 

**Authors**: Ruta Binkyte  

**Link**: [PDF](https://arxiv.org/pdf/2505.12001)  

**Abstract**: As large language models (LLMs) are increasingly used in multi-agent systems, questions of fairness should extend beyond resource distribution and procedural design to include the fairness of how agents communicate. Drawing from organizational psychology, we introduce a novel framework for evaluating Interactional fairness encompassing Interpersonal fairness (IF) and Informational fairness (InfF) in LLM-based multi-agent systems (LLM-MAS). We extend the theoretical grounding of Interactional Fairness to non-sentient agents, reframing fairness as a socially interpretable signal rather than a subjective experience. We then adapt established tools from organizational justice research, including Colquitt's Organizational Justice Scale and the Critical Incident Technique, to measure fairness as a behavioral property of agent interaction. We validate our framework through a pilot study using controlled simulations of a resource negotiation task. We systematically manipulate tone, explanation quality, outcome inequality, and task framing (collaborative vs. competitive) to assess how IF influences agent behavior. Results show that tone and justification quality significantly affect acceptance decisions even when objective outcomes are held constant. In addition, the influence of IF vs. InfF varies with context. This work lays the foundation for fairness auditing and norm-sensitive alignment in LLM-MAS. 

---
# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners 

**Authors**: Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11942)  

**Abstract**: Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents. 

---
# Position Paper: Bounded Alignment: What (Not) To Expect From AGI Agents 

**Authors**: Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11866)  

**Abstract**: The issues of AI risk and AI safety are becoming critical as the prospect of artificial general intelligence (AGI) looms larger. The emergence of extremely large and capable generative models has led to alarming predictions and created a stir from boardrooms to legislatures. As a result, AI alignment has emerged as one of the most important areas in AI research. The goal of this position paper is to argue that the currently dominant vision of AGI in the AI and machine learning (AI/ML) community needs to evolve, and that expectations and metrics for its safety must be informed much more by our understanding of the only existing instance of general intelligence, i.e., the intelligence found in animals, and especially in humans. This change in perspective will lead to a more realistic view of the technology, and allow for better policy decisions. 

---
# ChatHTN: Interleaving Approximate (LLM) and Symbolic HTN Planning 

**Authors**: Hector Munoz-Avila, David W. Aha, Paola Rizzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11814)  

**Abstract**: We introduce ChatHTN, a Hierarchical Task Network (HTN) planner that combines symbolic HTN planning techniques with queries to ChatGPT to approximate solutions in the form of task decompositions. The resulting hierarchies interleave task decompositions generated by symbolic HTN planning with those generated by ChatGPT. Despite the approximate nature of the results generates by ChatGPT, ChatHTN is provably sound; any plan it generates correctly achieves the input tasks. We demonstrate this property with an open-source implementation of our system. 

---
# Learning from Less: Guiding Deep Reinforcement Learning with Differentiable Symbolic Planning 

**Authors**: Zihan Ye, Oleg Arenz, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.11661)  

**Abstract**: When tackling complex problems, humans naturally break them down into smaller, manageable subtasks and adjust their initial plans based on observations. For instance, if you want to make coffee at a friend's place, you might initially plan to grab coffee beans, go to the coffee machine, and pour them into the machine. Upon noticing that the machine is full, you would skip the initial steps and proceed directly to brewing. In stark contrast, state of the art reinforcement learners, such as Proximal Policy Optimization (PPO), lack such prior knowledge and therefore require significantly more training steps to exhibit comparable adaptive behavior. Thus, a central research question arises: \textit{How can we enable reinforcement learning (RL) agents to have similar ``human priors'', allowing the agent to learn with fewer training interactions?} To address this challenge, we propose differentiable symbolic planner (Dylan), a novel framework that integrates symbolic planning into Reinforcement Learning. Dylan serves as a reward model that dynamically shapes rewards by leveraging human priors, guiding agents through intermediate subtasks, thus enabling more efficient exploration. Beyond reward shaping, Dylan can work as a high level planner that composes primitive policies to generate new behaviors while avoiding common symbolic planner pitfalls such as infinite execution loops. Our experimental evaluations demonstrate that Dylan significantly improves RL agents' performance and facilitates generalization to unseen tasks. 

---
# LLM Agents Are Hypersensitive to Nudges 

**Authors**: Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11584)  

**Abstract**: LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments. 

---
# TimeSeriesGym: A Scalable Benchmark for (Time Series) Machine Learning Engineering Agents 

**Authors**: Yifu Cai, Xinyu Li, Mononito Goswami, Michał Wiliński, Gus Welter, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2505.13291)  

**Abstract**: We introduce TimeSeriesGym, a scalable benchmarking framework for evaluating Artificial Intelligence (AI) agents on time series machine learning engineering challenges. Existing benchmarks lack scalability, focus narrowly on model building in well-defined settings, and evaluate only a limited set of research artifacts (e.g., CSV submission files). To make AI agent benchmarking more relevant to the practice of machine learning engineering, our framework scales along two critical dimensions. First, recognizing that effective ML engineering requires a range of diverse skills, TimeSeriesGym incorporates challenges from diverse sources spanning multiple domains and tasks. We design challenges to evaluate both isolated capabilities (including data handling, understanding research repositories, and code translation) and their combinations, and rather than addressing each challenge independently, we develop tools that support designing multiple challenges at scale. Second, we implement evaluation mechanisms for multiple research artifacts, including submission files, code, and models, using both precise numeric measures and more flexible LLM-based evaluation approaches. This dual strategy balances objective assessment with contextual judgment. Although our initial focus is on time series applications, our framework can be readily extended to other data modalities, broadly enhancing the comprehensiveness and practical utility of agentic AI evaluation. We open-source our benchmarking framework to facilitate future research on the ML engineering capabilities of AI agents. 

---
# When a Reinforcement Learning Agent Encounters Unknown Unknowns 

**Authors**: Juntian Zhu, Miguel de Carvalho, Zhouwang Yang, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.13188)  

**Abstract**: An AI agent might surprisingly find she has reached an unknown state which she has never been aware of -- an unknown unknown. We mathematically ground this scenario in reinforcement learning: an agent, after taking an action calculated from value functions $Q$ and $V$ defined on the {\it {aware domain}}, reaches a state out of the domain. To enable the agent to handle this scenario, we propose an {\it episodic Markov decision {process} with growing awareness} (EMDP-GA) model, taking a new {\it noninformative value expansion} (NIVE) approach to expand value functions to newly aware areas: when an agent arrives at an unknown unknown, value functions $Q$ and $V$ whereon are initialised by noninformative beliefs -- the averaged values on the aware domain. This design is out of respect for the complete absence of knowledge in the newly discovered state. The upper confidence bound momentum Q-learning is then adapted to the growing awareness for training the EMDP-GA model. We prove that (1) the regret of our approach is asymptotically consistent with the state of the art (SOTA) without exposure to unknown unknowns in an extremely uncertain environment, and (2) our computational complexity and space complexity are comparable with the SOTA -- these collectively suggest that though an unknown unknown is surprising, it will be asymptotically properly discovered with decent speed and an affordable cost. 

---
# Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs 

**Authors**: Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13026)  

**Abstract**: Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods. 

---
# Leveraging LLM Inconsistency to Boost Pass@k Performance 

**Authors**: Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12938)  

**Abstract**: Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations. 

---
# From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents 

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12981)  

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation. 

---
# PyFCG: Fluid Construction Grammar in Python 

**Authors**: Paul Van Eecke, Katrien Beuls  

**Link**: [PDF](https://arxiv.org/pdf/2505.12920)  

**Abstract**: We present PyFCG, an open source software library that ports Fluid Construction Grammar (FCG) to the Python programming language. PyFCG enables its users to seamlessly integrate FCG functionality into Python programs, and to use FCG in combination with other libraries within Python's rich ecosystem. Apart from a general description of the library, this paper provides three walkthrough tutorials that demonstrate example usage of PyFCG in typical use cases of FCG: (i) formalising and testing construction grammar analyses, (ii) learning usage-based construction grammars from corpora, and (iii) implementing agent-based experiments on emergent communication. 

---
# PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs 

**Authors**: Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12814)  

**Abstract**: Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity. 

---
# Dynamic Sight Range Selection in Multi-Agent Reinforcement Learning 

**Authors**: Wei-Chen Liao, Ti-Rong Wu, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12811)  

**Abstract**: Multi-agent reinforcement Learning (MARL) is often challenged by the sight range dilemma, where agents either receive insufficient or excessive information from their environment. In this paper, we propose a novel method, called Dynamic Sight Range Selection (DSR), to address this issue. DSR utilizes an Upper Confidence Bound (UCB) algorithm and dynamically adjusts the sight range during training. Experiment results show several advantages of using DSR. First, we demonstrate using DSR achieves better performance in three common MARL environments, including Level-Based Foraging (LBF), Multi-Robot Warehouse (RWARE), and StarCraft Multi-Agent Challenge (SMAC). Second, our results show that DSR consistently improves performance across multiple MARL algorithms, including QMIX and MAPPO. Third, DSR offers suitable sight ranges for different training steps, thereby accelerating the training process. Finally, DSR provides additional interpretability by indicating the optimal sight range used during training. Unlike existing methods that rely on global information or communication mechanisms, our approach operates solely based on the individual sight ranges of agents. This approach offers a practical and efficient solution to the sight range dilemma, making it broadly applicable to real-world complex environments. 

---
# Counterfactual Explanations for Continuous Action Reinforcement Learning 

**Authors**: Shuyang Dong, Shangtong Zhang, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12701)  

**Abstract**: Reinforcement Learning (RL) has shown great promise in domains like healthcare and robotics but often struggles with adoption due to its lack of interpretability. Counterfactual explanations, which address "what if" scenarios, provide a promising avenue for understanding RL decisions but remain underexplored for continuous action spaces. We propose a novel approach for generating counterfactual explanations in continuous action RL by computing alternative action sequences that improve outcomes while minimizing deviations from the original sequence. Our approach leverages a distance metric for continuous actions and accounts for constraints such as adhering to predefined policies in specific states. Evaluations in two RL domains, Diabetes Control and Lunar Lander, demonstrate the effectiveness, efficiency, and generalization of our approach, enabling more interpretable and trustworthy RL applications. 

---
# AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection 

**Authors**: Tiankai Yang, Junjun Liu, Wingchun Siu, Jiahang Wang, Zhuangzhuang Qian, Chanjuan Song, Cheng Cheng, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12594)  

**Abstract**: Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD. 

---
# AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use 

**Authors**: Yaotian Yang, Yiwen Tang, Yizhe Chen, Xiao Chen, Jiangjie Qiu, Hao Xiong, Haoyu Yin, Zhiyao Luo, Yifei Zhang, Sijia Tao, Wentao Li, Qinghua Zhang, Yuqiang Li, Wanli Ouyang, Bin Zhao, Xiaonan Wang, Fei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12650)  

**Abstract**: Machine learning-based interatomic potentials and force fields depend critically on accurate atomic structures, yet such data are scarce due to the limited availability of experimentally resolved crystals. Although atomic-resolution electron microscopy offers a potential source of structural data, converting these images into simulation-ready formats remains labor-intensive and error-prone, creating a bottleneck for model training and validation. We introduce AutoMat, an end-to-end, agent-assisted pipeline that automatically transforms scanning transmission electron microscopy (STEM) images into atomic crystal structures and predicts their physical properties. AutoMat combines pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, fast relaxation and property prediction via MatterSim, and coordinated orchestration across all stages. We propose the first dedicated STEM2Mat-Bench for this task and evaluate performance using lattice RMSD, formation energy MAE, and structure-matching success rate. By orchestrating external tool calls, AutoMat enables a text-only LLM to outperform vision-language models in this domain, achieving closed-loop reasoning throughout the pipeline. In large-scale experiments over 450 structure samples, AutoMat substantially outperforms existing multimodal large language models and tools. These results validate both AutoMat and STEM2Mat-Bench, marking a key step toward bridging microscopy and atomistic simulation in materials this http URL code and dataset are publicly available at this https URL and this https URL. 

---
# Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning 

**Authors**: Zirun Guo, Minjie Hong, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12432)  

**Abstract**: Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at this https URL. 

---
# IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems 

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.12442)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses. 

---
# A universal policy wrapper with guarantees 

**Authors**: Anton Bolychev, Georgiy Malaniya, Grigory Yaremenko, Anastasia Krasnaya, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12354)  

**Abstract**: We introduce a universal policy wrapper for reinforcement learning agents that ensures formal goal-reaching guarantees. In contrast to standard reinforcement learning algorithms that excel in performance but lack rigorous safety assurances, our wrapper selectively switches between a high-performing base policy -- derived from any existing RL method -- and a fallback policy with known convergence properties. Base policy's value function supervises this switching process, determining when the fallback policy should override the base policy to ensure the system remains on a stable path. The analysis proves that our wrapper inherits the fallback policy's goal-reaching guarantees while preserving or improving upon the performance of the base policy. Notably, it operates without needing additional system knowledge or online constrained optimization, making it readily deployable across diverse reinforcement learning architectures and tasks. 

---
# Robust Planning for Autonomous Driving via Mixed Adversarial Diffusion Predictions 

**Authors**: Albert Zhao, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12327)  

**Abstract**: We describe a robust planning method for autonomous driving that mixes normal and adversarial agent predictions output by a diffusion model trained for motion prediction. We first train a diffusion model to learn an unbiased distribution of normal agent behaviors. We then generate a distribution of adversarial predictions by biasing the diffusion model at test time to generate predictions that are likely to collide with a candidate plan. We score plans using expected cost with respect to a mixture distribution of normal and adversarial predictions, leading to a planner that is robust against adversarial behaviors but not overly conservative when agents behave normally. Unlike current approaches, we do not use risk measures that over-weight adversarial behaviors while placing little to no weight on low-cost normal behaviors or use hard safety constraints that may not be appropriate for all driving scenarios. We show the effectiveness of our method on single-agent and multi-agent jaywalking scenarios as well as a red light violation scenario. 

---
# Enhance Mobile Agents Thinking Process Via Iterative Preference Learning 

**Authors**: Kun Huang, Weikai Xu, Yuxuan Liu, Quandong Wang, Pengzhi Gao, Wei Liu, Jian Luan, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.12299)  

**Abstract**: The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios. 

---
# LAMeTA: Intent-Aware Agentic Network Optimization via a Large AI Model-Empowered Two-Stage Approach 

**Authors**: Yinqiu Liu, Guangyuan Liu, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Geng Sun, Zehui Xiong, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.12247)  

**Abstract**: Nowadays, Generative AI (GenAI) reshapes numerous domains by enabling machines to create content across modalities. As GenAI evolves into autonomous agents capable of reasoning, collaboration, and interaction, they are increasingly deployed on network infrastructures to serve humans automatically. This emerging paradigm, known as the agentic network, presents new optimization challenges due to the demand to incorporate subjective intents of human users expressed in natural language. Traditional generic Deep Reinforcement Learning (DRL) struggles to capture intent semantics and adjust policies dynamically, thus leading to suboptimality. In this paper, we present LAMeTA, a Large AI Model (LAM)-empowered Two-stage Approach for intent-aware agentic network optimization. First, we propose Intent-oriented Knowledge Distillation (IoKD), which efficiently distills intent-understanding capabilities from resource-intensive LAMs to lightweight edge LAMs (E-LAMs) to serve end users. Second, we develop Symbiotic Reinforcement Learning (SRL), integrating E-LAMs with a policy-based DRL framework. In SRL, E-LAMs translate natural language user intents into structured preference vectors that guide both state representation and reward design. The DRL, in turn, optimizes the generative service function chain composition and E-LAM selection based on real-time network conditions, thus optimizing the subjective Quality-of-Experience (QoE). Extensive experiments conducted in an agentic network with 81 agents demonstrate that IoKD reduces mean squared error in intent prediction by up to 22.5%, while SRL outperforms conventional generic DRL by up to 23.5% in maximizing intent-aware QoE. 

---
# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction 

**Authors**: Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12224)  

**Abstract**: Vision-Language-Action (VLA) models have recently advanced robotic manipulation by translating natural-language instructions and image information into sequential control actions. However, these models often underperform in open-world scenarios, as they are predominantly trained on successful expert demonstrations and exhibit a limited capacity for failure recovery. In this work, we present a Robotic Failure Analysis and Correction (RoboFAC) framework to address this issue. Firstly, we construct RoboFAC dataset comprising 9,440 erroneous manipulation trajectories and 78,623 QA pairs across 16 diverse tasks and 53 scenes in both simulation and real-world environments. Leveraging our dataset, we develop RoboFAC model, which is capable of Task Understanding, Failure Analysis and Failure Correction. Experimental results demonstrate that the RoboFAC model outperforms GPT-4o by 34.1% on our evaluation benchmark. Furthermore, we integrate the RoboFAC model into a real-world VLA control pipeline as an external supervision providing correction instructions, yielding a 29.1% relative improvement on average on four real-world tasks. The results show that our RoboFAC framework effectively handles robotic failures and assists the VLA model in recovering from failures. 

---
# LLM-DSE: Searching Accelerator Parameters with LLM Agents 

**Authors**: Hanyu Wang, Xinrui Wu, Zijian Ding, Su Zheng, Chengyue Wang, Tony Nowatzki, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12188)  

**Abstract**: Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample this http URL present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: this https URL. 

---
# MARVEL: Multi-Agent RTL Vulnerability Extraction using Large Language Models 

**Authors**: Luca Collini, Baleegh Ahmad, Joey Ah-kiow, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11963)  

**Abstract**: Hardware security verification is a challenging and time-consuming task. For this purpose, design engineers may utilize tools such as formal verification, linters, and functional simulation tests, coupled with analysis and a deep understanding of the hardware design being inspected. Large Language Models (LLMs) have been used to assist during this task, either directly or in conjunction with existing tools. We improve the state of the art by proposing MARVEL, a multi-agent LLM framework for a unified approach to decision-making, tool use, and reasoning. MARVEL mimics the cognitive process of a designer looking for security vulnerabilities in RTL code. It consists of a supervisor agent that devises the security policy of the system-on-chips (SoCs) using its security documentation. It delegates tasks to validate the security policy to individual executor agents. Each executor agent carries out its assigned task using a particular strategy. Each executor agent may use one or more tools to identify potential security bugs in the design and send the results back to the supervisor agent for further analysis and confirmation. MARVEL includes executor agents that leverage formal tools, linters, simulation tests, LLM-based detection schemes, and static analysis-based checks. We test our approach on a known buggy SoC based on OpenTitan from the Hack@DATE competition. We find that 20 of the 48 issues reported by MARVEL pose security vulnerabilities. 

---
# Modèles de Substitution pour les Modèles à base d'Agents : Enjeux, Méthodes et Applications 

**Authors**: Paul Saves, Nicolas Verstaevel, Benoît Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11912)  

**Abstract**: Multi-agent simulations enables the modeling and analyses of the dynamic behaviors and interactions of autonomous entities evolving in complex environments. Agent-based models (ABM) are widely used to study emergent phenomena arising from local interactions. However, their high computational cost poses a significant challenge, particularly for large-scale simulations requiring extensive parameter exploration, optimization, or uncertainty quantification. The increasing complexity of ABM limits their feasibility for real-time decision-making and large-scale scenario analysis. To address these limitations, surrogate models offer an efficient alternative by learning approximations from sparse simulation data. These models provide cheap-to-evaluate predictions, significantly reducing computational costs while maintaining accuracy. Various machine learning techniques, including regression models, neural networks, random forests and Gaussian processes, have been applied to construct robust surrogates. Moreover, uncertainty quantification and sensitivity analysis play a crucial role in enhancing model reliability and interpretability.
This article explores the motivations, methods, and applications of surrogate modeling for ABM, emphasizing the trade-offs between accuracy, computational efficiency, and interpretability. Through a case study on a segregation model, we highlight the challenges associated with building and validating surrogate models, comparing different approaches and evaluating their performance. Finally, we discuss future perspectives on integrating surrogate models within ABM to improve scalability, explainability, and real-time decision support across various fields such as ecology, urban planning and economics. 

---
# Mobile-Bench-v2: A More Realistic and Comprehensive Benchmark for VLM-based Mobile Agents 

**Authors**: Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.11891)  

**Abstract**: VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks struggle with obtaining stable reward signals due to dynamic environmental changes. Offline benchmarks evaluate the agents through single-path trajectories, which stands in contrast to the inherently multi-solution characteristics of GUI tasks. Additionally, both types of benchmarks fail to assess whether mobile agents can handle noise or engage in proactive interactions due to a lack of noisy apps or overly full instructions during the evaluation process. To address these limitations, we use a slot-based instruction generation method to construct a more realistic and comprehensive benchmark named Mobile-Bench-v2. Mobile-Bench-v2 includes a common task split, with offline multi-path evaluation to assess the agent's ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ads apps, and a contaminated split named AITZ-Noise to formulate a real noisy environment. Furthermore, an ambiguous instruction split with preset Q\&A interactions is released to evaluate the agent's proactive interaction capabilities. We conduct evaluations on these splits using the single-agent framework AppAgent-v1, the multi-agent framework Mobile-Agent-v2, as well as other mobile agents such as UI-Tars and OS-Atlas. Code and data are available at this https URL. 

---
# RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving 

**Authors**: Zepeng Ding, Dixuan Wang, Ziqin Luo, Guochao Jiang, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11893)  

**Abstract**: Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness. 

---
# Retrospex: Language Agent Meets Offline Reinforcement Learning Critic 

**Authors**: Yufei Xiang, Yiqun Shen, Yeqin Zhang, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11807)  

**Abstract**: Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines. 

---
# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration 

**Authors**: Shijun Li, Hilaf Hasson, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11765)  

**Abstract**: Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.
In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches. 

---
# Cloud-Based AI Systems: Leveraging Large Language Models for Intelligent Fault Detection and Autonomous Self-Healing 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11743)  

**Abstract**: With the rapid development of cloud computing systems and the increasing complexity of their infrastructure, intelligent mechanisms to detect and mitigate failures in real time are becoming increasingly important. Traditional methods of failure detection are often difficult to cope with the scale and dynamics of modern cloud environments. In this study, we propose a novel AI framework based on Massive Language Model (LLM) for intelligent fault detection and self-healing mechanisms in cloud systems. The model combines existing machine learning fault detection algorithms with LLM's natural language understanding capabilities to process and parse system logs, error reports, and real-time data streams through semantic context. The method adopts a multi-level architecture, combined with supervised learning for fault classification and unsupervised learning for anomaly detection, so that the system can predict potential failures before they occur and automatically trigger the self-healing mechanism. Experimental results show that the proposed model is significantly better than the traditional fault detection system in terms of fault detection accuracy, system downtime reduction and recovery speed. 

---
# EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents 

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11717)  

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines. 

---
# PeerGuard: Defending Multi-Agent Systems Against Backdoor Attacks Through Mutual Reasoning 

**Authors**: Falong Fan, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11642)  

**Abstract**: Multi-agent systems leverage advanced AI models as autonomous agents that interact, cooperate, or compete to complete complex tasks across applications such as robotics and traffic management. Despite their growing importance, safety in multi-agent systems remains largely underexplored, with most research focusing on single AI models rather than interacting agents. This work investigates backdoor vulnerabilities in multi-agent systems and proposes a defense mechanism based on agent interactions. By leveraging reasoning abilities, each agent evaluates responses from others to detect illogical reasoning processes, which indicate poisoned agents. Experiments on LLM-based multi-agent systems, including ChatGPT series and Llama 3, demonstrate the effectiveness of the proposed method, achieving high accuracy in identifying poisoned agents while minimizing false positives on clean agents. We believe this work provides insights into multi-agent system safety and contributes to the development of robust, trustworthy AI interactions. 

---
# Toward Adaptive Categories: Dimensional Governance for Agentic AI 

**Authors**: Zeynep Engin, David Hand  

**Link**: [PDF](https://arxiv.org/pdf/2505.11579)  

**Abstract**: As AI systems evolve from static tools to dynamic agents, traditional categorical governance frameworks -- based on fixed risk tiers, levels of autonomy, or human oversight models -- are increasingly insufficient on their own. Systems built on foundation models, self-supervised learning, and multi-agent architectures increasingly blur the boundaries that categories were designed to police. In this Perspective, we make the case for dimensional governance: a framework that tracks how decision authority, process autonomy, and accountability (the 3As) distribute dynamically across human-AI relationships. A critical advantage of this approach is its ability to explicitly monitor system movement toward and across key governance thresholds, enabling preemptive adjustments before risks materialize. This dimensional approach provides the necessary foundation for more adaptive categorization, enabling thresholds and classifications that can evolve with emerging capabilities. While categories remain essential for decision-making, building them upon dimensional foundations allows for context-specific adaptability and stakeholder-responsive governance that static approaches cannot achieve. We outline key dimensions, critical trust thresholds, and practical examples illustrating where rigid categorical frameworks fail -- and where a dimensional mindset could offer a more resilient and future-proof path forward for both governance and innovation at the frontier of artificial intelligence. 

---
# Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks 

**Authors**: Yuxuan Li, Aoi Naito, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2505.11556)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction. 

---
# Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning 

**Authors**: Chongyang Tan, Ruoqi Wen, Rongpeng Li, Zhifeng Zhao, Ekram Hossain, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11570)  

**Abstract**: Federated Learning (FL) enables distributed model training across edge devices in a privacy-friendly manner. However, its efficiency heavily depends on effective device selection and high-dimensional resource allocation in dynamic and heterogeneous wireless environments. Conventional methods demand a confluence of domain-specific expertise, extensive hyperparameter tuning, and/or heavy interaction cost. This paper proposes a Tool-aided Evolutionary Large Language Model (T-ELLM) framework to generate a qualified policy for device selection in a wireless FL environment. Unlike conventional optimization methods, T-ELLM leverages natural language-based scenario prompts to enhance generalization across varying network conditions. The framework decouples the joint optimization problem mathematically, enabling tractable learning of device selection policies while delegating resource allocation to convex optimization tools. To improve adaptability, T-ELLM integrates a sample-efficient, model-based virtual learning environment that captures the relationship between device selection and learning performance, facilitating subsequent group relative policy optimization. This concerted approach reduces reliance on real-world interactions, minimizing communication overhead while maintaining high-fidelity decision-making. Theoretical analysis proves that the discrepancy between virtual and real environments is bounded, ensuring the advantage function learned in the virtual environment maintains a provably small deviation from real-world conditions. Experimental results demonstrate that T-ELLM outperforms benchmark methods in energy efficiency and exhibits robust adaptability to environmental changes. 

---
# MR. Judge: Multimodal Reasoner as a Judge 

**Authors**: Renjie Pi, Felix Bai, Qibin Chen, Simon Wang, Jiulong Shan, Kieran Liu, Meng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13403)  

**Abstract**: The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%. 

---
# AdaptThink: Reasoning Models Can Learn When to Think 

**Authors**: Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13417)  

**Abstract**: Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at this https URL. 

---
# From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery 

**Authors**: Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.13259)  

**Abstract**: Large Language Models (LLMs) are catalyzing a paradigm shift in scientific discovery, evolving from task-specific automation tools into increasingly autonomous agents and fundamentally redefining research processes and human-AI collaboration. This survey systematically charts this burgeoning field, placing a central focus on the changing roles and escalating capabilities of LLMs in science. Through the lens of the scientific method, we introduce a foundational three-level taxonomy-Tool, Analyst, and Scientist-to delineate their escalating autonomy and evolving responsibilities within the research lifecycle. We further identify pivotal challenges and future research trajectories such as robotic automation, self-improvement, and ethical governance. Overall, this survey provides a conceptual architecture and strategic foresight to navigate and shape the future of AI-driven scientific discovery, fostering both rapid innovation and responsible advancement. Github Repository: this https URL. 

---
# Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability 

**Authors**: Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13258)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive domains. However, although RAG achieved successes across distinct domains, there are still some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on developing more powerful RAG retrievers, but how to enhance the generator's (LLM's) ability to utilize the retrieved information for reasoning and generation? 2) Transparency. Most RAG methods ignore which retrieved content actually contributes to the reasoning process, resulting in a lack of interpretability and visibility. To address this, we propose ARENA (Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator framework trained via reinforcement learning (RL) with our proposed rewards. Based on the structured generation and adaptive reward calculation, our RL-based training enables the model to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various RAG baselines demonstrate that our model achieves 10-30% improvements on all multi-hop QA datasets, which is comparable with the SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show that ARENA has strong flexibility to be adopted on new datasets without extra training. Our models and codes are publicly released. 

---
# ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents 

**Authors**: Navid Madani, Rohini Srihari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12531)  

**Abstract**: Large language models (LLMs) increasingly power mental-health chatbots, yet the field still lacks a scalable, theory-grounded way to decide which model is most effective to deploy. We present ESC-Judge, the first end-to-end evaluation framework that (i) grounds head-to-head comparisons of emotional-support LLMs in Clara Hill's established Exploration-Insight-Action counseling model, providing a structured and interpretable view of performance, and (ii) fully automates the evaluation pipeline at scale. ESC-Judge operates in three stages: first, it synthesizes realistic help-seeker roles by sampling empirically salient attributes such as stressors, personality, and life history; second, it has two candidate support agents conduct separate sessions with the same role, isolating model-specific strategies; and third, it asks a specialized judge LLM to express pairwise preferences across rubric-anchored skills that span the Exploration, Insight, and Action spectrum. In our study, ESC-Judge matched PhD-level annotators on 85 percent of Exploration, 83 percent of Insight, and 86 percent of Action decisions, demonstrating human-level reliability at a fraction of the cost. All code, prompts, synthetic roles, transcripts, and judgment scripts are released to promote transparent progress in emotionally supportive AI. 

---
# Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games 

**Authors**: Jinming Zhang, Yunfei Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.12439)  

**Abstract**: Interactive Fiction games (IF games) are where players interact through natural language commands. While recent advances in Artificial Intelligence agents have reignited interest in IF games as a domain for studying decision-making, existing approaches prioritize task-specific performance metrics over human-like comprehension of narrative context and gameplay logic. This work presents a cognitively inspired framework that guides Large Language Models (LLMs) to learn and play IF games systematically. Our proposed **L**earning to **P**lay **L**ike **H**umans (LPLH) framework integrates three key components: (1) structured map building to capture spatial and narrative relationships, (2) action learning to identify context-appropriate commands, and (3) feedback-driven experience analysis to refine decision-making over time. By aligning LLMs-based agents' behavior with narrative intent and commonsense constraints, LPLH moves beyond purely exploratory strategies to deliver more interpretable, human-like performance. Crucially, this approach draws on cognitive science principles to more closely simulate how human players read, interpret, and respond within narrative worlds. As a result, LPLH reframes the IF games challenge as a learning problem for LLMs-based agents, offering a new path toward robust, context-aware gameplay in complex text-based environments. 

---
# Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement 

**Authors**: Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12060)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at this https URL. 

---
# BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering 

**Authors**: Taolin Zhang, Dongyang Li, Qizhou Chen, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11811)  

**Abstract**: Multi-hop question answering (QA) involves finding multiple relevant passages and performing step-by-step reasoning to answer complex questions. Previous works on multi-hop QA employ specific methods from different modeling perspectives based on large language models (LLMs), regardless of the question types. In this paper, we first conduct an in-depth analysis of public multi-hop QA benchmarks, dividing the questions into four types and evaluating five types of cutting-edge methods for multi-hop QA: Chain-of-Thought (CoT), Single-step, Iterative-step, Sub-step, and Adaptive-step. We find that different types of multi-hop questions have varying degrees of sensitivity to different types of methods. Thus, we propose a Bi-levEL muLti-agEnt reasoning (BELLE) framework to address multi-hop QA by specifically focusing on the correspondence between question types and methods, where each type of method is regarded as an ''operator'' by prompting LLMs differently. The first level of BELLE includes multiple agents that debate to obtain an executive plan of combined ''operators'' to address the multi-hop QA task comprehensively. During the debate, in addition to the basic roles of affirmative debater, negative debater, and judge, at the second level, we further leverage fast and slow debaters to monitor whether changes in viewpoints are reasonable. Extensive experiments demonstrate that BELLE significantly outperforms strong baselines in various datasets. Additionally, the model consumption of BELLE is higher cost-effectiveness than that of single models in more complex multi-hop QA scenarios. 

---
# Talk to Your Slides: Efficient Slide Editing Agent with Large Language Models 

**Authors**: Kyudan Jung, Hojun Cho, Jooyeol Yun, Jaehyeok Jang, Jagul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11604)  

**Abstract**: Existing research on large language models (LLMs) for PowerPoint predominantly focuses on slide generation, overlooking the common yet tedious task of editing existing slides. We introduce Talk-to-Your-Slides, an LLM-powered agent that directly edits slides within active PowerPoint sessions through COM communication. Our system employs a two-level approach: (1) high-level processing where an LLM agent interprets instructions and formulates editing plans, and (2) low-level execution where Python scripts directly manipulate PowerPoint objects. Unlike previous methods relying on predefined operations, our approach enables more flexible and contextually-aware editing. To facilitate evaluation, we present TSBench, a human-annotated dataset of 379 diverse editing instructions with corresponding slide variations. Experimental results demonstrate that Talk-to-Your-Slides significantly outperforms baseline methods in execution success rate, instruction fidelity, and editing efficiency. Our code and benchmark are available at this https URL 

---
# GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents 

**Authors**: Zheng Wu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12842)  

**Abstract**: Graphical user interface (GUI) agents have recently emerged as an intriguing paradigm for human-computer interaction, capable of automatically executing user instructions to operate intelligent terminal devices. However, when encountering out-of-distribution (OOD) instructions that violate environmental constraints or exceed the current capabilities of agents, GUI agents may suffer task breakdowns or even pose security threats. Therefore, effective OOD detection for GUI agents is essential. Traditional OOD detection methods perform suboptimally in this domain due to the complex embedding space and evolving GUI environments. In this work, we observe that the in-distribution input semantic space of GUI agents exhibits a clustering pattern with respect to the distance from the centroid. Based on the finding, we propose GEM, a novel method based on fitting a Gaussian mixture model over input embedding distances extracted from the GUI Agent that reflect its capability boundary. Evaluated on eight datasets spanning smartphones, computers, and web browsers, our method achieves an average accuracy improvement of 23.70\% over the best-performing baseline. Analysis verifies the generalization ability of our method through experiments on nine different backbones. The codes are available at this https URL. 

---
