# CoMAC: Conversational Agent for Multi-Source Auxiliary Context with Sparse and Symmetric Latent Interactions 

**Authors**: Junfeng Liu, Christopher T. Symons, Ranga Raju Vatsavai  

**Link**: [PDF](https://arxiv.org/pdf/2503.19274)  

**Abstract**: Recent advancements in AI-driven conversational agents have exhibited immense potential of AI applications. Effective response generation is crucial to the success of these agents. While extensive research has focused on leveraging multiple auxiliary data sources (e.g., knowledge bases and personas) to enhance response generation, existing methods often struggle to efficiently extract relevant information from these sources. There are still clear limitations in the ability to combine versatile conversational capabilities with adherence to known facts and adaptation to large variations in user preferences and belief systems, which continues to hinder the wide adoption of conversational AI tools. This paper introduces a novel method, Conversational Agent for Multi-Source Auxiliary Context with Sparse and Symmetric Latent Interactions (CoMAC), for conversation generation, which employs specialized encoding streams and post-fusion grounding networks for multiple data sources to identify relevant persona and knowledge information for the conversation. CoMAC also leverages a novel text similarity metric that allows bi-directional information sharing among multiple sources and focuses on a selective subset of meaningful words. Our experiments show that CoMAC improves the relevant persona and knowledge prediction accuracies and response generation quality significantly over two state-of-the-art methods. 

---
# Writing as a testbed for open ended agents 

**Authors**: Sian Gooding, Lucia Lopez-Rivilla, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2503.19711)  

**Abstract**: Open-ended tasks are particularly challenging for LLMs due to the vast solution space, demanding both expansive exploration and adaptable strategies, especially when success lacks a clear, objective definition. Writing, with its vast solution space and subjective evaluation criteria, provides a compelling testbed for studying such problems. In this paper, we investigate the potential of LLMs to act as collaborative co-writers, capable of suggesting and implementing text improvements autonomously. We analyse three prominent LLMs - Gemini 1.5 Pro, Claude 3.5 Sonnet, and GPT-4o - focusing on how their action diversity, human alignment, and iterative improvement capabilities impact overall performance. This work establishes a framework for benchmarking autonomous writing agents and, more broadly, highlights fundamental challenges and potential solutions for building systems capable of excelling in diverse open-ended domains. 

---
# MARS: Memory-Enhanced Agents with Reflective Self-improvement 

**Authors**: Xuechen Liang, Meiling Tao, Yinghui Xia, Jianhui Wang, Kun Li, Yijin Wang, Jingsong Yang, Tianyu Shi, Yuantao Wang, Miao Zhang, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19271)  

**Abstract**: Large language models (LLMs) have made significant advances in the field of natural language processing, but they still face challenges such as continuous decision-making, lack of long-term memory, and limited context windows in dynamic environments. To address these issues, this paper proposes an innovative framework Memory-Enhanced Agents with Reflective Self-improvement. The MARS framework comprises three agents: the User, the Assistant, and the Checker. By integrating iterative feedback, reflective mechanisms, and a memory optimization mechanism based on the Ebbinghaus forgetting curve, it significantly enhances the agents capabilities in handling multi-tasking and long-span information. 

---
# A Survey of Large Language Model Agents for Question Answering 

**Authors**: Murong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.19213)  

**Abstract**: This paper surveys the development of large language model (LLM)-based agents for question answering (QA). Traditional agents face significant limitations, including substantial data requirements and difficulty in generalizing to new environments. LLM-based agents address these challenges by leveraging LLMs as their core reasoning engine. These agents achieve superior QA results compared to traditional QA pipelines and naive LLM QA systems by enabling interaction with external environments. We systematically review the design of LLM agents in the context of QA tasks, organizing our discussion across key stages: planning, question understanding, information retrieval, and answer generation. Additionally, this paper identifies ongoing challenges and explores future research directions to enhance the performance of LLM agent QA systems. 

---
# Multi-agent Application System in Office Collaboration Scenarios 

**Authors**: Songtao Sun, Jingyi Li, Yuanfei Dong, Haoguang Liu, Chenxin Xu, Fuyang Li, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19584)  

**Abstract**: This paper introduces a multi-agent application system designed to enhance office collaboration efficiency and work quality. The system integrates artificial intelligence, machine learning, and natural language processing technologies, achieving functionalities such as task allocation, progress monitoring, and information sharing. The agents within the system are capable of providing personalized collaboration support based on team members' needs and incorporate data analysis tools to improve decision-making quality. The paper also proposes an intelligent agent architecture that separates Plan and Solver, and through techniques such as multi-turn query rewriting and business tool retrieval, it enhances the agent's multi-intent and multi-turn dialogue capabilities. Furthermore, the paper details the design of tools and multi-turn dialogue in the context of office collaboration scenarios, and validates the system's effectiveness through experiments and evaluations. Ultimately, the system has demonstrated outstanding performance in real business applications, particularly in query understanding, task planning, and tool calling. Looking forward, the system is expected to play a more significant role in addressing complex interaction issues within dynamic environments and large-scale multi-agent systems. 

---
# Thinking agents for zero-shot generalization to qualitatively novel tasks 

**Authors**: Thomas Miconi, Kevin McKee, Yicong Zheng, Jed McCaleb  

**Link**: [PDF](https://arxiv.org/pdf/2503.19815)  

**Abstract**: Intelligent organisms can solve truly novel problems which they have never encountered before, either in their lifetime or their evolution. An important component of this capacity is the ability to ``think'', that is, to mentally manipulate objects, concepts and behaviors in order to plan and evaluate possible solutions to novel problems, even without environment interaction. To generate problems that are truly qualitatively novel, while still solvable zero-shot (by mental simulation), we use the combinatorial nature of environments: we train the agent while withholding a specific combination of the environment's elements. The novel test task, based on this combination, is thus guaranteed to be truly novel, while still mentally simulable since the agent has been exposed to each individual element (and their pairwise interactions) during training. We propose a method to train agents endowed with world models to make use their mental simulation abilities, by selecting tasks based on the difference between the agent's pre-thinking and post-thinking performance. When tested on the novel, withheld problem, the resulting agent successfully simulated alternative scenarios and used the resulting information to guide its behavior in the actual environment, solving the novel task in a single real-environment trial (zero-shot). 

---
# MedAgent-Pro: Towards Multi-modal Evidence-based Medical Diagnosis via Reasoning Agentic Workflow 

**Authors**: Ziyue Wang, Junde Wu, Chang Han Low, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18968)  

**Abstract**: Developing reliable AI systems to assist human clinicians in multi-modal medical diagnosis has long been a key objective for researchers. Recently, Multi-modal Large Language Models (MLLMs) have gained significant attention and achieved success across various domains. With strong reasoning capabilities and the ability to perform diverse tasks based on user instructions, they hold great potential for enhancing medical diagnosis. However, directly applying MLLMs to the medical domain still presents challenges. They lack detailed perception of visual inputs, limiting their ability to perform quantitative image analysis, which is crucial for medical diagnostics. Additionally, MLLMs often exhibit hallucinations and inconsistencies in reasoning, whereas clinical diagnoses must adhere strictly to established criteria. To address these challenges, we propose MedAgent-Pro, an evidence-based reasoning agentic system designed to achieve reliable, explainable, and precise medical diagnoses. This is accomplished through a hierarchical workflow: at the task level, knowledge-based reasoning generate reliable diagnostic plans for specific diseases following retrieved clinical criteria. While at the case level, multiple tool agents process multi-modal inputs, analyze different indicators according to the plan, and provide a final diagnosis based on both quantitative and qualitative evidence. Comprehensive experiments on both 2D and 3D medical diagnosis tasks demonstrate the superiority and effectiveness of MedAgent-Pro, while case studies further highlight its reliability and interpretability. The code is available at this https URL. 

---
# Inducing Personality in LLM-Based Honeypot Agents: Measuring the Effect on Human-Like Agenda Generation 

**Authors**: Lewis Newsham, Ryan Hyland, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2503.19752)  

**Abstract**: This paper presents SANDMAN, an architecture for cyber deception that leverages Language Agents to emulate convincing human simulacra. Our 'Deceptive Agents' serve as advanced cyber decoys, designed for high-fidelity engagement with attackers by extending the observation period of attack behaviours. Through experimentation, measurement, and analysis, we demonstrate how a prompt schema based on the five-factor model of personality systematically induces distinct 'personalities' in Large Language Models. Our results highlight the feasibility of persona-driven Language Agents for generating diverse, realistic behaviours, ultimately improving cyber deception strategies. 

---
# ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning 

**Authors**: Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Fan Yang, Zenan Zhou, Weipeng Chen, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19470)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process. 

---
# Substance over Style: Evaluating Proactive Conversational Coaching Agents 

**Authors**: Vidya Srinivas, Xuhai Xu, Xin Liu, Kumar Ayush, Isaac Galatzer-Levy, Shwetak Patel, Daniel McDuff, Tim Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.19328)  

**Abstract**: While NLP research has made strides in conversational tasks, many approaches focus on single-turn responses with well-defined objectives or evaluation criteria. In contrast, coaching presents unique challenges with initially undefined goals that evolve through multi-turn interactions, subjective evaluation criteria, mixed-initiative dialogue. In this work, we describe and implement five multi-turn coaching agents that exhibit distinct conversational styles, and evaluate them through a user study, collecting first-person feedback on 155 conversations. We find that users highly value core functionality, and that stylistic components in absence of core components are viewed negatively. By comparing user feedback with third-person evaluations from health experts and an LM, we reveal significant misalignment across evaluation approaches. Our findings provide insights into design and evaluation of conversational coaching agents and contribute toward improving human-centered NLP applications. 

---
# CubeRobot: Grounding Language in Rubik's Cube Manipulation via Vision-Language Model 

**Authors**: Feiyang Wang, Xiaomin Yu, Wangyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.19281)  

**Abstract**: Proving Rubik's Cube theorems at the high level represents a notable milestone in human-level spatial imagination and logic thinking and reasoning. Traditional Rubik's Cube robots, relying on complex vision systems and fixed algorithms, often struggle to adapt to complex and dynamic scenarios. To overcome this limitation, we introduce CubeRobot, a novel vision-language model (VLM) tailored for solving 3x3 Rubik's Cubes, empowering embodied agents with multimodal understanding and execution capabilities. We used the CubeCoT image dataset, which contains multiple-level tasks (43 subtasks in total) that humans are unable to handle, encompassing various cube states. We incorporate a dual-loop VisionCoT architecture and Memory Stream, a paradigm for extracting task-related features from VLM-generated planning queries, thus enabling CubeRobot to independent planning, decision-making, reflection and separate management of high- and low-level Rubik's Cube tasks. Furthermore, in low-level Rubik's Cube restoration tasks, CubeRobot achieved a high accuracy rate of 100%, similar to 100% in medium-level tasks, and achieved an accuracy rate of 80% in high-level tasks. 

---
# Option Discovery Using LLM-guided Semantic Hierarchical Reinforcement Learning 

**Authors**: Chak Lam Shek, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.19007)  

**Abstract**: Large Language Models (LLMs) have shown remarkable promise in reasoning and decision-making, yet their integration with Reinforcement Learning (RL) for complex robotic tasks remains underexplored. In this paper, we propose an LLM-guided hierarchical RL framework, termed LDSC, that leverages LLM-driven subgoal selection and option reuse to enhance sample efficiency, generalization, and multi-task adaptability. Traditional RL methods often suffer from inefficient exploration and high computational cost. Hierarchical RL helps with these challenges, but existing methods often fail to reuse options effectively when faced with new tasks. To address these limitations, we introduce a three-stage framework that uses LLMs for subgoal generation given natural language description of the task, a reusable option learning and selection method, and an action-level policy, enabling more effective decision-making across diverse tasks. By incorporating LLMs for subgoal prediction and policy guidance, our approach improves exploration efficiency and enhances learning performance. On average, LDSC outperforms the baseline by 55.9\% in average reward, demonstrating its effectiveness in complex RL settings. More details and experiment videos could be found in \href{this https URL}{this link\footnote{this https URL}}. 

---
