# Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning 

**Authors**: Baining Zhao, Ziyou Wang, Jianjie Fang, Chen Gao, Fanhang Man, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12680)  

**Abstract**: Humans can perceive and reason about spatial relationships from sequential visual observations, such as egocentric video streams. However, how pretrained models acquire such abilities, especially high-level reasoning, remains unclear. This paper introduces Embodied-R, a collaborative framework combining large-scale Vision-Language Models (VLMs) for perception and small-scale Language Models (LMs) for reasoning. Using Reinforcement Learning (RL) with a novel reward system considering think-answer logical consistency, the model achieves slow-thinking capabilities with limited computational resources. After training on only 5k embodied video samples, Embodied-R with a 3B LM matches state-of-the-art multimodal reasoning models (OpenAI-o1, Gemini-2.5-pro) on both in-distribution and out-of-distribution embodied spatial reasoning tasks. Embodied-R also exhibits emergent thinking patterns such as systematic analysis and contextual integration. We further explore research questions including response length, training on VLM, strategies for reward design, and differences in model generalization after SFT (Supervised Fine-Tuning) and RL training. 

---
# Exploring Expert Failures Improves LLM Agent Tuning 

**Authors**: Li-Cheng Lan, Andrew Bai, Minhao Cheng, Ruochen Wang, Cho-Jui Hsieh, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.13145)  

**Abstract**: Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld. 

---
# WebLists: Extracting Structured Information From Complex Interactive Websites Using Executable LLM Agents 

**Authors**: Arth Bohra, Manvel Saroyan, Danil Melkozerov, Vahe Karufanyan, Gabriel Maher, Pascal Weinberger, Artem Harutyunyan, Giovanni Campagna  

**Link**: [PDF](https://arxiv.org/pdf/2504.12682)  

**Abstract**: Most recent web agent research has focused on navigation and transaction tasks, with little emphasis on extracting structured data at scale. We present WebLists, a benchmark of 200 data-extraction tasks across four common business and enterprise use-cases. Each task requires an agent to navigate to a webpage, configure it appropriately, and extract complete datasets with well-defined schemas. We show that both LLMs with search capabilities and SOTA web agents struggle with these tasks, with a recall of 3% and 31%, respectively, despite higher performance on question-answering tasks.
To address this challenge, we propose BardeenAgent, a novel framework that enables web agents to convert their execution into repeatable programs, and replay them at scale across pages with similar structure. BardeenAgent is also the first LLM agent to take advantage of the regular structure of HTML. In particular BardeenAgent constructs a generalizable CSS selector to capture all relevant items on the page, then fits the operations to extract the data.
On the WebLists benchmark, BardeenAgent achieves 66% recall overall, more than doubling the performance of SOTA web agents, and reducing cost per output row by 3x. 

---
# The Chronicles of Foundation AI for Forensics of Multi-Agent Provenance 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2504.12612)  

**Abstract**: Provenance is the chronology of things, resonating with the fundamental pursuit to uncover origins, trace connections, and situate entities within the flow of space and time. As artificial intelligence advances towards autonomous agents capable of interactive collaboration on complex tasks, the provenance of generated content becomes entangled in the interplay of collective creation, where contributions are continuously revised, extended or overwritten. In a multi-agent generative chain, content undergoes successive transformations, often leaving little, if any, trace of prior contributions. In this study, we investigates the problem of tracking multi-agent provenance across the temporal dimension of generation. We propose a chronological system for post hoc attribution of generative history from content alone, without reliance on internal memory states or external meta-information. At its core lies the notion of symbolic chronicles, representing signed and time-stamped records, in a form analogous to the chain of custody in forensic science. The system operates through a feedback loop, whereby each generative timestep updates the chronicle of prior interactions and synchronises it with the synthetic content in the very act of generation. This research seeks to develop an accountable form of collaborative artificial intelligence within evolving cyber ecosystems. 

---
# InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning 

**Authors**: Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13032)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach. 

---
# Sleep-time Compute: Beyond Inference Scaling at Test-time 

**Authors**: Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.13171)  

**Abstract**: Scaling test-time compute has emerged as a key ingredient for enabling large language models (LLMs) to solve difficult problems, but comes with high latency and inference cost. We introduce sleep-time compute, which allows models to "think" offline about contexts before queries are presented: by anticipating what queries users might ask and pre-computing useful quantities, we can significantly reduce the compute requirements at test-time. To demonstrate the efficacy of our method, we create modified versions of two reasoning tasks - Stateful GSM-Symbolic and Stateful AIME. We find that sleep-time compute can reduce the amount of test-time compute needed to achieve the same accuracy by ~ 5x on Stateful GSM-Symbolic and Stateful AIME and that by scaling sleep-time compute we can further increase accuracy by up to 13% on Stateful GSM-Symbolic and 18% on Stateful AIME. Furthermore, we introduce Multi-Query GSM-Symbolic, which extends GSM-Symbolic by including multiple related queries per context. By amortizing sleep-time compute across related queries about the same context using Multi-Query GSM-Symbolic, we can decrease the average cost per query by 2.5x. We then conduct additional analysis to understand when sleep-time compute is most effective, finding the predictability of the user query to be well correlated with the efficacy of sleep-time compute. Finally, we conduct a case-study of applying sleep-time compute to a realistic agentic SWE task. 

---
# Agentic AI Optimisation (AAIO): what it is, how it works, why it matters, and how to deal with it 

**Authors**: Luciano Floridi, Carlotta Buttaboni, Emmie Hine, Jessica Morley, Claudio Novelli, Tyler Schroder  

**Link**: [PDF](https://arxiv.org/pdf/2504.12482)  

**Abstract**: The emergence of Agentic Artificial Intelligence (AAI) systems capable of independently initiating digital interactions necessitates a new optimisation paradigm designed explicitly for seamless agent-platform interactions. This article introduces Agentic AI Optimisation (AAIO) as an essential methodology for ensuring effective integration between websites and agentic AI systems. Like how Search Engine Optimisation (SEO) has shaped digital content discoverability, AAIO can define interactions between autonomous AI agents and online platforms. By examining the mutual interdependency between website optimisation and agentic AI success, the article highlights the virtuous cycle that AAIO can create. It further explores the governance, ethical, legal, and social implications (GELSI) of AAIO, emphasising the necessity of proactive regulatory frameworks to mitigate potential negative impacts. The article concludes by affirming AAIO's essential role as part of a fundamental digital infrastructure in the era of autonomous digital agents, advocating for equitable and inclusive access to its benefits. 

---
# Heuristic Recognition and Rapid Response to Unfamiliar Events Outside of Agent Design Scope 

**Authors**: Robert E. Wray, Steven J. Jones, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2504.12497)  

**Abstract**: Regardless of past learning, an agent in an open world will face unfamiliar situations and events outside of prior experience, existing models, or policies. Further, the agent will sometimes lack relevant knowledge and/or sufficient time to assess the situation, generate and evaluate options, and pursue a robustly considered course of action. How can an agent respond reasonably to situations that are outside of its original design scope? How can it recognize such situations sufficiently quickly and reliably to determine reasonable, adaptive courses of action? We identify key characteristics needed for solutions, evaluate the state-of-the-art by these requirements, and outline a proposed, novel approach that combines domain-general meta-knowledge (in the form of appraisals inspired by human cognition) and metareasoning. It has the potential to provide fast, adaptive responses to unfamiliar situations, more fully meeting the performance characteristics required for open-world, general agents. 

---
# Retrieval-Augmented Generation with Conflicting Evidence 

**Authors**: Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.13079)  

**Abstract**: Large language model (LLM) agents are increasingly employing retrieval-augmented generation (RAG) to improve the factuality of their responses. However, in practice, these systems often need to handle ambiguous user queries and potentially conflicting information from multiple sources while also suppressing inaccurate information from noisy or irrelevant documents. Prior work has generally studied and addressed these challenges in isolation, considering only one aspect at a time, such as handling ambiguity or robustness to noise and misinformation. We instead consider multiple factors simultaneously, proposing (i) RAMDocs (Retrieval with Ambiguity and Misinformation in Documents), a new dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise; and (ii) MADAM-RAG, a multi-agent approach in which LLM agents debate over the merits of an answer over multiple rounds, allowing an aggregator to collate responses corresponding to disambiguated entities while discarding misinformation and noise, thereby handling diverse sources of conflict jointly. We demonstrate the effectiveness of MADAM-RAG using both closed and open-source models on AmbigDocs -- which requires presenting all valid answers for ambiguous queries -- improving over strong RAG baselines by up to 11.40% and on FaithEval -- which requires suppressing misinformation -- where we improve by up to 15.80% (absolute) with Llama3.3-70B-Instruct. Furthermore, we find that RAMDocs poses a challenge for existing RAG baselines (Llama3.3-70B-Instruct only obtains 32.60 exact match score). While MADAM-RAG begins to address these conflicting factors, our analysis indicates that a substantial gap remains especially when increasing the level of imbalance in supporting evidence and misinformation. 

---
# Towards Conversational AI for Human-Machine Collaborative MLOps 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Anargyros Tsadimas, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.12477)  

**Abstract**: This paper presents a Large Language Model (LLM) based conversational agent system designed to enhance human-machine collaboration in Machine Learning Operations (MLOps). We introduce the Swarm Agent, an extensible architecture that integrates specialized agents to create and manage ML workflows through natural language interactions. The system leverages a hierarchical, modular design incorporating a KubeFlow Pipelines (KFP) Agent for ML pipeline orchestration, a MinIO Agent for data management, and a Retrieval-Augmented Generation (RAG) Agent for domain-specific knowledge integration. Through iterative reasoning loops and context-aware processing, the system enables users with varying technical backgrounds to discover, execute, and monitor ML pipelines; manage datasets and artifacts; and access relevant documentation, all via intuitive conversational interfaces. Our approach addresses the accessibility gap in complex MLOps platforms like Kubeflow, making advanced ML tools broadly accessible while maintaining the flexibility to extend to other platforms. The paper describes the architecture, implementation details, and demonstrates how this conversational MLOps assistant reduces complexity and lowers barriers to entry for users across diverse technical skill levels. 

---
# QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning? 

**Authors**: Zhouyang Jiang, Bin Zhang, Airong Wei, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12961)  

**Abstract**: Credit assignment has remained a fundamental challenge in multi-agent reinforcement learning (MARL). Previous studies have primarily addressed this issue through value decomposition methods under the centralized training with decentralized execution paradigm, where neural networks are utilized to approximate the nonlinear relationship between individual Q-values and the global Q-value. Although these approaches have achieved considerable success in various benchmark tasks, they still suffer from several limitations, including imprecise attribution of contributions, limited interpretability, and poor scalability in high-dimensional state spaces. To address these challenges, we propose a novel algorithm, \textbf{QLLM}, which facilitates the automatic construction of credit assignment functions using large language models (LLMs). Specifically, the concept of \textbf{TFCAF} is introduced, wherein the credit allocation process is represented as a direct and expressive nonlinear functional formulation. A custom-designed \textit{coder-evaluator} framework is further employed to guide the generation, verification, and refinement of executable code by LLMs, significantly mitigating issues such as hallucination and shallow reasoning during inference. Extensive experiments conducted on several standard MARL benchmarks demonstrate that the proposed method consistently outperforms existing state-of-the-art baselines. Moreover, QLLM exhibits strong generalization capability and maintains compatibility with a wide range of MARL algorithms that utilize mixing networks, positioning it as a promising and versatile solution for complex multi-agent scenarios. 

---
# Are AI agents the new machine translation frontier? Challenges and opportunities of single- and multi-agent systems for multilingual digital communication 

**Authors**: Vicent Briva-Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2504.12891)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has introduced AI agents as a disruptive paradigm across various industries, yet their application in machine translation (MT) remains underexplored. This paper describes and analyses the potential of single- and multi-agent systems for MT, reflecting on how they could enhance multilingual digital communication. While single-agent systems are well-suited for simpler translation tasks, multi-agent systems, which involve multiple specialized AI agents collaborating in a structured manner, may offer a promising solution for complex scenarios requiring high accuracy, domain-specific knowledge, and contextual awareness. To demonstrate the feasibility of multi-agent workflows in MT, we are conducting a pilot study in legal MT. The study employs a multi-agent system involving four specialized AI agents for (i) translation, (ii) adequacy review, (iii) fluency review, and (iv) final editing. Our findings suggest that multi-agent systems may have the potential to significantly improve domain-adaptability and contextual awareness, with superior translation quality to traditional MT or single-agent systems. This paper also sets the stage for future research into multi-agent applications in MT, integration into professional translation workflows, and shares a demo of the system analyzed in the paper. 

---
# Pandora: A Code-Driven Large Language Model Agent for Unified Reasoning Across Diverse Structured Knowledge 

**Authors**: Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12734)  

**Abstract**: Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions (NLQs) by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods either rely on employing task-specific strategies or custom-defined representations, which struggle to leverage the knowledge transfer between different SKR tasks or align with the prior of LLMs, thereby limiting their performance. This paper proposes a novel USKR framework named \textsc{Pandora}, which takes advantage of \textsc{Python}'s \textsc{Pandas} API to construct a unified knowledge representation for alignment with LLM pre-training. It employs an LLM to generate textual reasoning steps and executable Python code for each question. Demonstrations are drawn from a memory of training examples that cover various SKR tasks, facilitating knowledge transfer. Extensive experiments on four benchmarks involving three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified frameworks and competes effectively with task-specific methods. 

---
# SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2504.12722)  

**Abstract**: Recommender systems play a central role in numerous real-life applications, yet evaluating their performance remains a significant challenge due to the gap between offline metrics and online behaviors. Given the scarcity and limits (e.g., privacy issues) of real user data, we introduce SimUSER, an agent framework that serves as believable and cost-effective human proxies. SimUSER first identifies self-consistent personas from historical data, enriching user profiles with unique backgrounds and personalities. Then, central to this evaluation are users equipped with persona, memory, perception, and brain modules, engaging in interactions with the recommender system. SimUSER exhibits closer alignment with genuine humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments to explore the effects of thumbnails on click rates, the exposure effect, and the impact of reviews on user engagement. Finally, we refine recommender system parameters based on offline A/B test results, resulting in improved user engagement in the real world. 

---
# The Athenian Academy: A Seven-Layer Architecture Model for Multi-Agent Systems 

**Authors**: Lidong Zhai, Zhijie Qiu, Xizhong Guo, Jiaqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12735)  

**Abstract**: This paper proposes the "Academy of Athens" multi-agent seven-layer framework, aimed at systematically addressing challenges in multi-agent systems (MAS) within artificial intelligence (AI) art creation, such as collaboration efficiency, role allocation, environmental adaptation, and task parallelism. The framework divides MAS into seven layers: multi-agent collaboration, single-agent multi-role playing, single-agent multi-scene traversal, single-agent multi-capability incarnation, different single agents using the same large model to achieve the same target agent, single-agent using different large models to achieve the same target agent, and multi-agent synthesis of the same target agent. Through experimental validation in art creation, the framework demonstrates its unique advantages in task collaboration, cross-scene adaptation, and model fusion. This paper further discusses current challenges such as collaboration mechanism optimization, model stability, and system security, proposing future exploration through technologies like meta-learning and federated learning. The framework provides a structured methodology for multi-agent collaboration in AI art creation and promotes innovative applications in the art field. 

---
# Multi-Agent Reinforcement Learning Simulation for Environmental Policy Synthesis 

**Authors**: James Rudd-Jones, Mirco Musolesi, María Pérez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2504.12777)  

**Abstract**: Climate policy development faces significant challenges due to deep uncertainty, complex system dynamics, and competing stakeholder interests. Climate simulation methods, such as Earth System Models, have become valuable tools for policy exploration. However, their typical use is for evaluating potential polices, rather than directly synthesizing them. The problem can be inverted to optimize for policy pathways, but the traditional optimization approaches often struggle with non-linear dynamics, heterogeneous agents, and comprehensive uncertainty quantification. We propose a framework for augmenting climate simulations with Multi-Agent Reinforcement Learning (MARL) to address these limitations. We identify key challenges at the interface between climate simulations and the application of MARL in the context of policy synthesis, including reward definition, scalability with increasing agents and state spaces, uncertainty propagation across linked systems, and solution validation. Additionally, we discuss challenges in making MARL-derived solutions interpretable and useful for policy-makers. Our framework provides a foundation for more sophisticated climate policy exploration while acknowledging important limitations and areas for future research. 

---
# Trajectory Adaptation using Large Language Models 

**Authors**: Anurag Maurya, Tashmoy Ghosh, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2504.12755)  

**Abstract**: Adapting robot trajectories based on human instructions as per new situations is essential for achieving more intuitive and scalable human-robot interactions. This work proposes a flexible language-based framework to adapt generic robotic trajectories produced by off-the-shelf motion planners like RRT, A-star, etc, or learned from human demonstrations. We utilize pre-trained LLMs to adapt trajectory waypoints by generating code as a policy for dense robot manipulation, enabling more complex and flexible instructions than current methods. This approach allows us to incorporate a broader range of commands, including numerical inputs. Compared to state-of-the-art feature-based sequence-to-sequence models which require training, our method does not require task-specific training and offers greater interpretability and more effective feedback mechanisms. We validate our approach through simulation experiments on the robotic manipulator, aerial vehicle, and ground robot in the Pybullet and Gazebo simulation environments, demonstrating that LLMs can successfully adapt trajectories to complex human instructions. 

---
# Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination 

**Authors**: Kunal Jha, Wilka Carvalho, Yancheng Liang, Simon S. Du, Max Kleiman-Weiner, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.12714)  

**Abstract**: Zero-shot coordination (ZSC), the ability to adapt to a new partner in a cooperative task, is a critical component of human-compatible AI. While prior work has focused on training agents to cooperate on a single task, these specialized models do not generalize to new tasks, even if they are highly similar. Here, we study how reinforcement learning on a distribution of environments with a single partner enables learning general cooperative skills that support ZSC with many new partners on many new problems. We introduce two Jax-based, procedural generators that create billions of solvable coordination challenges. We develop a new paradigm called Cross-Environment Cooperation (CEC), and show that it outperforms competitive baselines quantitatively and qualitatively when collaborating with real people. Our findings suggest that learning to collaborate across many unique scenarios encourages agents to develop general norms, which prove effective for collaboration with different partners. Together, our results suggest a new route toward designing generalist cooperative agents capable of interacting with humans without requiring human data. 

---
# MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation 

**Authors**: Haris Riaz, Sourav Bhabesh, Vinayak Arannil, Miguel Ballesteros, Graham Horwood  

**Link**: [PDF](https://arxiv.org/pdf/2504.12563)  

**Abstract**: Recent smaller language models such Phi-3.5 and Phi-4 rely on synthetic data generated using larger Language models. Questions remain about leveraging synthetic data for other use cases, such as adapting LLMs to specific domains. A key limitation of synthetic data is low diversity, which negatively impacts its downstream applicability for improving other models. To address this, we propose MetaSynth, a method for generating synthetic data that enhances diversity through meta-prompting, where a language model orchestrates multiple "expert" LLM agents to collaboratively generate data. Using only 25 million tokens of synthetic data generated with MetaSynth, we successfully adapt a well-trained LLM (Mistral-7B-v0.3) to two specialized domains-Finance and Biomedicine-without compromising the capabilities of the resulting model in general tasks. In addition, we evaluate the diversity of our synthetic data using seven automated metrics, and find that it approaches the diversity of LLM pre-training corpora.
Continually pre-training Mistral-7B-v0.3 with MetaSynth notably outperforms the base LLM, showing improvements of up to 4.08% in Finance and 13.75% in Biomedicine. The same model shows degraded performance when trained on data generated using a template prompt, even when the template includes prior generations and varying In-Context exemplars of real data. Our findings suggest that a few million tokens of diverse synthetic data without mixing any real data, is sufficient for effective domain adaptation when using MetaSynth. 

---
# HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation 

**Authors**: Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12330)  

**Abstract**: While Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge, conventional single-agent RAG remains fundamentally limited in resolving complex queries demanding coordinated reasoning across heterogeneous data ecosystems. We present HM-RAG, a novel Hierarchical Multi-agent Multimodal RAG framework that pioneers collaborative intelligence for dynamic knowledge synthesis across structured, unstructured, and graph-based data. The framework is composed of three-tiered architecture with specialized agents: a Decomposition Agent that dissects complex queries into contextually coherent sub-tasks via semantic-aware query rewriting and schema-guided context augmentation; Multi-source Retrieval Agents that carry out parallel, modality-specific retrieval using plug-and-play modules designed for vector, graph, and web-based databases; and a Decision Agent that uses consistency voting to integrate multi-source answers and resolve discrepancies in retrieval results through Expert Model Refinement. This architecture attains comprehensive query understanding by combining textual, graph-relational, and web-derived evidence, resulting in a remarkable 12.95% improvement in answer accuracy and a 3.56% boost in question classification accuracy over baseline RAG systems on the ScienceQA and CrisisMMD benchmarks. Notably, HM-RAG establishes state-of-the-art results in zero-shot settings on both datasets. Its modular architecture ensures seamless integration of new data modalities while maintaining strict data governance, marking a significant advancement in addressing the critical challenges of multimodal reasoning and knowledge synthesis in RAG systems. Code is available at this https URL. 

---
# BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents 

**Authors**: Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2504.12516)  

**Abstract**: We present BrowseComp, a simple yet challenging benchmark for measuring the ability for agents to browse the web. BrowseComp comprises 1,266 questions that require persistently navigating the internet in search of hard-to-find, entangled information. Despite the difficulty of the questions, BrowseComp is simple and easy-to-use, as predicted answers are short and easily verifiable against reference answers. BrowseComp for browsing agents can be seen as analogous to how programming competitions are an incomplete but useful benchmark for coding agents. While BrowseComp sidesteps challenges of a true user query distribution, like generating long answers or resolving ambiguity, it measures the important core capability of exercising persistence and creativity in finding information. BrowseComp can be found at this https URL. 

---
# Reimagining Urban Science: Scaling Causal Inference with Large Language Models 

**Authors**: Yutong Xia, Ao Qu, Yunhan Zheng, Yihong Tang, Dingyi Zhuang, Yuxuan Liang, Cathy Wu, Roger Zimmermann, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12345)  

**Abstract**: Urban causal research is essential for understanding the complex dynamics of cities and informing evidence-based policies. However, it is challenged by the inefficiency and bias of hypothesis generation, barriers to multimodal data complexity, and the methodological fragility of causal experimentation. Recent advances in large language models (LLMs) present an opportunity to rethink how urban causal analysis is conducted. This Perspective examines current urban causal research by analyzing taxonomies that categorize research topics, data sources, and methodological approaches to identify structural gaps. We then introduce an LLM-driven conceptual framework, AutoUrbanCI, composed of four distinct modular agents responsible for hypothesis generation, data engineering, experiment design and execution, and results interpretation with policy recommendations. We propose evaluation criteria for rigor and transparency and reflect on implications for human-AI collaboration, equity, and accountability. We call for a new research agenda that embraces AI-augmented workflows not as replacements for human expertise but as tools to broaden participation, improve reproducibility, and unlock more inclusive forms of urban causal reasoning. 

---
# Exploring the Impact of Personality Traits on Conversational Recommender Systems: A Simulation with Large Language Models 

**Authors**: Xiaoyan Zhao, Yang Deng, Wenjie Wang, Hongzhan lin, Hong Cheng, Rui Zhang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.12313)  

**Abstract**: Conversational Recommender Systems (CRSs) engage users in multi-turn interactions to deliver personalized recommendations. The emergence of large language models (LLMs) further enhances these systems by enabling more natural and dynamic user interactions. However, a key challenge remains in understanding how personality traits shape conversational recommendation outcomes. Psychological evidence highlights the influence of personality traits on user interaction behaviors. To address this, we introduce an LLM-based personality-aware user simulation for CRSs (PerCRS). The user agent induces customizable personality traits and preferences, while the system agent possesses the persuasion capability to simulate realistic interaction in CRSs. We incorporate multi-aspect evaluation to ensure robustness and conduct extensive analysis from both user and system perspectives. Experimental results demonstrate that state-of-the-art LLMs can effectively generate diverse user responses aligned with specified personality traits, thereby prompting CRSs to dynamically adjust their recommendation strategies. Our experimental analysis offers empirical insights into the impact of personality traits on the outcomes of conversational recommender systems. 

---
