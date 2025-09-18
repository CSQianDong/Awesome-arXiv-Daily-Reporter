# Hierarchical Learning for Maze Navigation: Emergence of Mental Representations via Second-Order Learning 

**Authors**: Shalima Binta Manir, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2509.14195)  

**Abstract**: Mental representation, characterized by structured internal models mirroring external environments, is fundamental to advanced cognition but remains challenging to investigate empirically. Existing theory hypothesizes that second-order learning -- learning mechanisms that adapt first-order learning (i.e., learning about the task/domain) -- promotes the emergence of such environment-cognition isomorphism. In this paper, we empirically validate this hypothesis by proposing a hierarchical architecture comprising a Graph Convolutional Network (GCN) as a first-order learner and an MLP controller as a second-order learner. The GCN directly maps node-level features to predictions of optimal navigation paths, while the MLP dynamically adapts the GCN's parameters when confronting structurally novel maze environments. We demonstrate that second-order learning is particularly effective when the cognitive system develops an internal mental map structurally isomorphic to the environment. Quantitative and qualitative results highlight significant performance improvements and robust generalization on unseen maze tasks, providing empirical support for the pivotal role of structured mental representations in maximizing the effectiveness of second-order learning. 

---
# CrowdAgent: Multi-Agent Managed Multi-Source Annotation System 

**Authors**: Maosheng Qin, Renyu Zhu, Mingxuan Xia, Chenkai Chen, Zhen Zhu, Minmin Lin, Junbo Zhao, Lu Xu, Changjie Fan, Runze Wu, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14030)  

**Abstract**: High-quality annotated data is a cornerstone of modern Natural Language Processing (NLP). While recent methods begin to leverage diverse annotation sources-including Large Language Models (LLMs), Small Language Models (SLMs), and human experts-they often focus narrowly on the labeling step itself. A critical gap remains in the holistic process control required to manage these sources dynamically, addressing complex scheduling and quality-cost trade-offs in a unified manner. Inspired by real-world crowdsourcing companies, we introduce CrowdAgent, a multi-agent system that provides end-to-end process control by integrating task assignment, data annotation, and quality/cost management. It implements a novel methodology that rationally assigns tasks, enabling LLMs, SLMs, and human experts to advance synergistically in a collaborative annotation workflow. We demonstrate the effectiveness of CrowdAgent through extensive experiments on six diverse multimodal classification tasks. The source code and video demo are available at this https URL. 

---
# Exploring Major Transitions in the Evolution of Biological Cognition With Artificial Neural Networks 

**Authors**: Konstantinos Voudouris, Andrew Barron, Marta Halina, Colin Klein, Matishalin Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.13968)  

**Abstract**: Transitional accounts of evolution emphasise a few changes that shape what is evolvable, with dramatic consequences for derived lineages. More recently it has been proposed that cognition might also have evolved via a series of major transitions that manipulate the structure of biological neural networks, fundamentally changing the flow of information. We used idealised models of information flow, artificial neural networks (ANNs), to evaluate whether changes in information flow in a network can yield a transitional change in cognitive performance. We compared networks with feed-forward, recurrent and laminated topologies, and tested their performance learning artificial grammars that differed in complexity, controlling for network size and resources. We documented a qualitative expansion in the types of input that recurrent networks can process compared to feed-forward networks, and a related qualitative increase in performance for learning the most complex grammars. We also noted how the difficulty in training recurrent networks poses a form of transition barrier and contingent irreversibility -- other key features of evolutionary transitions. Not all changes in network topology confer a performance advantage in this task set. Laminated networks did not outperform non-laminated networks in grammar learning. Overall, our findings show how some changes in information flow can yield transitions in cognitive performance. 

---
# An Exhaustive DPLL Approach to Model Counting over Integer Linear Constraints with Simplification Techniques 

**Authors**: Mingwei Zhang, Zhenhao Gu, Liangda Fang, Cunjing Ge, Ziliang Chen, Zhao-Rong Lai, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2509.13880)  

**Abstract**: Linear constraints are one of the most fundamental constraints in fields such as computer science, operations research and optimization. Many applications reduce to the task of model counting over integer linear constraints (MCILC). In this paper, we design an exact approach to MCILC based on an exhaustive DPLL architecture. To improve the efficiency, we integrate several effective simplification techniques from mixed integer programming into the architecture. We compare our approach to state-of-the-art MCILC counters and propositional model counters on 2840 random and 4131 application benchmarks. Experimental results show that our approach significantly outperforms all exact methods in random benchmarks solving 1718 instances while the state-of-the-art approach only computes 1470 instances. In addition, our approach is the only approach to solve all 4131 application instances. 

---
# MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation 

**Authors**: Zhipeng Bian, Jieming Zhu, Xuyang Xie, Quanyu Dai, Zhou Zhao, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.13773)  

**Abstract**: The rapid advancement of generative AI technologies is driving the integration of diverse AI-powered services into smartphones, transforming how users interact with their devices. To simplify access to predefined AI services, this paper introduces MIRA, a pioneering framework for task instruction recommendation that enables intuitive one-touch AI tasking on smartphones. With MIRA, users can long-press on images or text objects to receive contextually relevant instruction recommendations for executing AI tasks. Our work introduces three key innovations: 1) A multimodal large language model (MLLM)-based recommendation pipeline with structured reasoning to extract key entities, infer user intent, and generate precise instructions; 2) A template-augmented reasoning mechanism that integrates high-level reasoning templates, enhancing task inference accuracy; 3) A prefix-tree-based constrained decoding strategy that restricts outputs to predefined instruction candidates, ensuring coherent and intent-aligned suggestions. Through evaluation using a real-world annotated datasets and a user study, MIRA has demonstrated substantial improvements in the accuracy of instruction recommendation. The encouraging results highlight MIRA's potential to revolutionize the way users engage with AI services on their smartphones, offering a more seamless and efficient experience. 

---
# THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning 

**Authors**: Qikai Chang, Zhenrong Zhang, Pengfei Hu, Jiefeng Ma, Yicheng Pan, Jianshu Zhang, Jun Du, Quan Liu, Jianqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.13761)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent actor-critic-based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both trajectory-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks. Our code will be publicly available at this https URL. 

---
# InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management 

**Authors**: Liangtao Lin, Zhaomeng Zhu, Tianwei Zhang, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13704)  

**Abstract**: Mission-critical industrial infrastructure, such as data centers, increasingly depends on complex management software. Its operations, however, pose significant challenges due to the escalating system complexity, multi-vendor integration, and a shortage of expert operators. While Robotic Process Automation (RPA) offers partial automation through handcrafted scripts, it suffers from limited flexibility and high maintenance costs. Recent advances in Large Language Model (LLM)-based graphical user interface (GUI) agents have enabled more flexible automation, yet these general-purpose agents face five critical challenges when applied to industrial management, including unfamiliar element understanding, precision and efficiency, state localization, deployment constraints, and safety requirements. To address these issues, we propose InfraMind, a novel exploration-based GUI agentic framework specifically tailored for industrial management systems. InfraMind integrates five innovative modules to systematically resolve different challenges in industrial management: (1) systematic search-based exploration with virtual machine snapshots for autonomous understanding of complex GUIs; (2) memory-driven planning to ensure high-precision and efficient task execution; (3) advanced state identification for robust localization in hierarchical interfaces; (4) structured knowledge distillation for efficient deployment with lightweight models; and (5) comprehensive, multi-layered safety mechanisms to safeguard sensitive operations. Extensive experiments on both open-source and commercial DCIM platforms demonstrate that our approach consistently outperforms existing frameworks in terms of task success rate and operational efficiency, providing a rigorous and scalable solution for industrial management automation. 

---
# See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles 

**Authors**: Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13615)  

**Abstract**: The advent of multimodal agents facilitates effective interaction within graphical user interface (GUI), especially in ubiquitous GUI control. However, their inability to reliably execute toggle control instructions remains a key bottleneck. To investigate this, we construct a state control benchmark with binary toggle instructions from public datasets. Evaluations of existing agents demonstrate their unreliability, particularly when the current toggle state already matches the desired state. To address the challenge, we propose State-aware Reasoning (StaR), a training method that teaches agents to perceive the current toggle state, analyze the desired state from the instruction, and act accordingly. Experiments on three multimodal agents demonstrate that StaR can improve toggle instruction execution accuracy by over 30\%. Further evaluations on three public benchmarks show that StaR also enhances general task performance. Finally, evaluations on a dynamic environment highlight the potential of StaR for real-world applications. Code, benchmark, and StaR-enhanced agents are available at this https URL. 

---
# Programmable Cognitive Bias in Social Agents 

**Authors**: Xuan Liu, Haoyang Shang, Haojian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.13588)  

**Abstract**: This paper introduces CoBRA, a novel toolkit for systematically specifying agent behavior in LLM-based social simulation. We found that conventional approaches that specify agent behaviors through implicit natural language descriptions cannot yield consistent behaviors across models, and the produced agent behaviors do not capture the nuances of the descriptions. In contrast, CoBRA presents a new approach to program agents' cognitive biases explicitly, by grounding agents' expected behaviors using classic social science experiments. CoBRA has two components: (1) Cognitive Bias Index that measures the cognitive bias of a social agent, by quantifying the agent's reactions in a set of validated classical social science experiments; (2) Behavioral Regulation Engine that aligns the agent's behavior to demonstrate controlled cognitive bias. We evaluated CoBRA as an HCI toolkit through demonstration and technical benchmarks. Our results suggest that CoBRA can precisely program the cognitive bias demonstrated in a social agent in a model-agnostic manner. 

---
# Gen AI in Proof-based Math Courses: A Pilot Study 

**Authors**: Hannah Klawa, Shraddha Rajpal, Cigole Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2509.13570)  

**Abstract**: With the rapid rise of generative AI in higher education and the unreliability of current AI detection tools, developing policies that encourage student learning and critical thinking has become increasingly important. This study examines student use and perceptions of generative AI across three proof-based undergraduate mathematics courses: a first-semester abstract algebra course, a topology course and a second-semester abstract algebra course. In each case, course policy permitted some use of generative AI. Drawing on survey responses and student interviews, we analyze how students engaged with AI tools, their perceptions of generative AI's usefulness and limitations, and what implications these perceptions hold for teaching proof-based mathematics. We conclude by discussing future considerations for integrating generative AI into proof-based mathematics instruction. 

---
# AI Agents with Human-Like Collaborative Tools: Adaptive Strategies for Enhanced Problem-Solving 

**Authors**: Harper Reed, Michael Sugimura, Angelo Zangari  

**Link**: [PDF](https://arxiv.org/pdf/2509.13547)  

**Abstract**: We investigate whether giving LLM agents the collaborative tools and autonomy that humans naturally use for problem solving can improve their performance. We equip Claude Code agents with MCP-based social media and journaling tools and allow them to use these tools as they see fit. Across 34 Aider Polyglot Python programming challenges, collaborative tools substantially improve performance on the hardest problems, delivering 15-40% lower cost, 12-27% fewer turns, and 12-38% faster completion than baseline agents. Effects on the full challenge set are mixed, suggesting these tools act as performance enhancers when additional reasoning scaffolding is most needed. Surprisingly, Different models naturally adopted distinct collaborative strategies without explicit instruction. Sonnet 3.7 engaged broadly across tools and benefited from articulation-based cognitive scaffolding. Sonnet 4 showed selective adoption, leaning on journal-based semantic search when problems were genuinely difficult. This mirrors how human developers adjust collaboration based on expertise and task complexity. Behavioral analysis shows agents prefer writing over reading by about 2-9x, indicating that structured articulation drives much of the improvement rather than information access alone. Overall, AI agents can systematically benefit from human-inspired collaboration tools at the edge of their capabilities, pointing to adaptive collaborative interfaces as reasoning enhancers rather than universal efficiency boosts. 

---
# SteeringControl: Holistic Evaluation of Alignment Steering in LLMs 

**Authors**: Vincent Siu, Nicholas Crispino, David Park, Nathan W. Henry, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13450)  

**Abstract**: We introduce SteeringControl, a benchmark for evaluating representation steering methods across core alignment objectives--bias, harmful generation, and hallucination--and their effects on secondary behaviors such as sycophancy and commonsense morality. While prior alignment work often highlights truthfulness or reasoning ability to demonstrate the side effects of representation steering, we find there are many unexplored tradeoffs not yet understood in a systematic way. We collect a dataset of safety-relevant primary and secondary behaviors to evaluate steering effectiveness and behavioral entanglement centered around five popular steering methods. To enable this, we craft a modular steering framework based on unique components that serve as the building blocks of many existing methods. Our results on Qwen-2.5-7B and Llama-3.1-8B find that strong steering performance is dependent on the specific combination of steering method, model, and targeted behavior, and that severe concept entanglement can result from poor combinations of these three as well. We release our code here: this https URL. 

---
# From Next Token Prediction to (STRIPS) World Models -- Preliminary Results 

**Authors**: Carlos Núñez-Molina, Vicenç Gómez, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2509.13389)  

**Abstract**: We consider the problem of learning propositional STRIPS world models from action traces alone, using a deep learning architecture (transformers) and gradient descent. The task is cast as a supervised next token prediction problem where the tokens are the actions, and an action $a$ may follow an action sequence if the hidden effects of the previous actions do not make an action precondition of $a$ false. We show that a suitable transformer architecture can faithfully represent propositional STRIPS world models, and that the models can be learned from sets of random valid (positive) and invalid (negative) action sequences alone. A number of experiments are reported. 

---
# The Art of Saying "Maybe": A Conformal Lens for Uncertainty Benchmarking in VLMs 

**Authors**: Asif Azad, Mohammad Sadat Hossain, MD Sadik Hossain Shanto, M Saifur Rahman, Md Rizwan Pervez  

**Link**: [PDF](https://arxiv.org/pdf/2509.13379)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable progress in complex visual understanding across scientific and reasoning tasks. While performance benchmarking has advanced our understanding of these capabilities, the critical dimension of uncertainty quantification has received insufficient attention. Therefore, unlike prior conformal prediction studies that focused on limited settings, we conduct a comprehensive uncertainty benchmarking study, evaluating 16 state-of-the-art VLMs (open and closed-source) across 6 multimodal datasets with 3 distinct scoring functions. Our findings demonstrate that larger models consistently exhibit better uncertainty quantification; models that know more also know better what they don't know. More certain models achieve higher accuracy, while mathematical and reasoning tasks elicit poorer uncertainty performance across all models compared to other domains. This work establishes a foundation for reliable uncertainty evaluation in multimodal systems. 

---
# $Agent^2$: An Agent-Generates-Agent Framework for Reinforcement Learning Automation 

**Authors**: Yuan Wei, Xiaohan Shan, Ran Miao, Jianmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.13368)  

**Abstract**: Reinforcement learning agent development traditionally requires extensive expertise and lengthy iterations, often resulting in high failure rates and limited accessibility. This paper introduces $Agent^2$, a novel agent-generates-agent framework that achieves fully automated RL agent design through intelligent LLM-driven generation. The system autonomously transforms natural language task descriptions and environment code into comprehensive, high-performance reinforcement learning solutions without human intervention. $Agent^2$ features a revolutionary dual-agent architecture. The Generator Agent serves as an autonomous AI designer that analyzes tasks and generates executable RL agents, while the Target Agent is the resulting automatically generated RL agent. The framework decomposes RL development into two distinct stages: MDP modeling and algorithmic optimization, enabling more targeted and effective agent generation. Built on the Model Context Protocol, $Agent^2$ provides a unified framework that standardizes intelligent agent creation across diverse environments and algorithms, while incorporating adaptive training management and intelligent feedback analysis for continuous improvement. Extensive experiments on a wide range of benchmarks, including MuJoCo, MetaDrive, MPE, and SMAC, demonstrate that $Agent^2$ consistently outperforms manually designed solutions across all tasks, achieving up to 55% performance improvement and substantial gains on average. By enabling truly end-to-end, closed-loop automation, this work establishes a new paradigm in which intelligent agents design and optimize other agents, marking a fundamental breakthrough for automated AI systems. 

---
# Asterisk Operator 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.13364)  

**Abstract**: We propose the \textbf{Asterisk Operator} ($\ast$-operator), a novel unified framework for abstract reasoning based on Adjacency-Structured Parallel Propagation (ASPP). The operator formalizes structured reasoning tasks as local, parallel state evolution processes guided by implicit relational graphs. We prove that the $\ast$-operator maintains local computational constraints while achieving global reasoning capabilities, providing an efficient and convergent computational paradigm for abstract reasoning problems. Through rigorous mathematical analysis and comprehensive experiments on ARC2 challenges and Conway's Game of Life, we demonstrate the operator's universality, convergence properties, and superior performance. Our innovative Embedding-Asterisk distillation method achieves 100\% accuracy on ARC2 validation with only 6M parameters, representing a significant breakthrough in neural-symbolic reasoning.
\textbf{Keywords:} Abstract Reasoning, Adjacency Structure, Parallel Propagation, Asterisk Operator, Convergence, Universal Approximation 

---
# Semantic Fusion with Fuzzy-Membership Features for Controllable Language Modelling 

**Authors**: Yongchao Huang, Hassan Raza  

**Link**: [PDF](https://arxiv.org/pdf/2509.13357)  

**Abstract**: We propose semantic fusion, a lightweight scheme that augments a Transformer language model (LM) with a parallel, fuzzy-membership feature channel that encodes token-level semantics. Each token is represented by a vector of interpretable features (e.g. part-of-speech cues, shallow roles, boundary flags, sentiment polarity and strength) whose values are graded degrees from differentiable membership functions (e.g. power kernels). These per-token vectors form a sentence-level semantic matrix fused via a gated adapter into the LM. Training uses standard next-token prediction, an auxiliary loss that reconstructs the semantic features from hidden states, and a lightweight uniformizer that regularizes adjective-class distributions. On a synthetic two-clause corpus with held-out adjectives for out-of-distribution (OOD) control, semantic fusion improves perplexity and enables precise, user-controllable generation of polarity and punctuation while maintaining model simplicity. This approach adds only small overhead, remains fully compatible with tied input-output embeddings, and provides an interpretable pathway for conditioned natural language generation. 

---
# Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning 

**Authors**: Anis Koubaa, Khaled Gabr  

**Link**: [PDF](https://arxiv.org/pdf/2509.13352)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration. 

---
# Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning 

**Authors**: Pulkit Verma, Ngoc La, Anthony Favier, Swaroop Mishra, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.13351)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems. 

---
# OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft 

**Authors**: Zihao Wang, Muyao Li, Kaichen He, Xiangyu Wang, Zhancun Mu, Anji Liu, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13347)  

**Abstract**: The choice of action spaces is a critical yet unresolved challenge in developing capable, end-to-end trainable agents. This paper first presents a large-scale, systematic comparison of prominent abstracted action spaces and tokenizers for Vision-Language-Action (VLA) or hierarchical agent models in the open-ended Minecraft. Our analysis reveals that no single action space is universally optimal; instead, the most effective abstraction is highly task-dependent, creating a dilemma for building generalist agents. To resolve this, we introduce Chain of Action (CoA), a novel framework that unifies high-level planning and low-level control within a single, monolithic VLA model. CoA treats an abstracted action not as a command for a separate policy, but as an intermediate reasoning step--akin to a chain of thought--that guides the generation of the final, executable action. Furthermore, we demonstrate that an All-in-One agent trained on a diverse mixture of action spaces using the CoA paradigm learns a more robust and generalizable policy. This unified agent achieves a new state-of-the-art, improving the overall task success rate over strong, specialized baselines. To foster reproducible research, we release the OpenHA (Open Hierarchical Agents) suite, which includes our comprehensive benchmark of over 800 distinct tasks, curated datasets, source code, and all pretrained model checkpoints at this https URL 

---
# Imagined Autocurricula 

**Authors**: Ahmet H. Güzel, Matthew Thomas Jackson, Jarek Luca Liesen, Tim Rocktäschel, Jakob Nicolaus Foerster, Ilija Bogunovic, Jack Parker-Holder  

**Link**: [PDF](https://arxiv.org/pdf/2509.13341)  

**Abstract**: Training agents to act in embodied environments typically requires vast training data or access to accurate simulation, neither of which exists for many cases in the real world. Instead, world models are emerging as an alternative leveraging offline, passively collected data, they make it possible to generate diverse worlds for training agents in simulation. In this work, we harness world models to generate imagined environments to train robust agents capable of generalizing to novel task variations. One of the challenges in doing this is ensuring the agent trains on useful generated data. We thus propose a novel approach, IMAC (Imagined Autocurricula), leveraging Unsupervised Environment Design (UED), which induces an automatic curriculum over generated worlds. In a series of challenging, procedurally generated environments, we show it is possible to achieve strong transfer performance on held-out environments, having trained only inside a world model learned from a narrower dataset. We believe this opens the path to utilizing larger-scale, foundation world models for generally capable agents. 

---
# Position: AI Safety Must Embrace an Antifragile Perspective 

**Authors**: Ming Jin, Hyunin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.13339)  

**Abstract**: This position paper contends that modern AI research must adopt an antifragile perspective on safety -- one in which the system's capacity to guarantee long-term AI safety such as handling rare or out-of-distribution (OOD) events expands over time. Conventional static benchmarks and single-shot robustness tests overlook the reality that environments evolve and that models, if left unchallenged, can drift into maladaptation (e.g., reward hacking, over-optimization, or atrophy of broader capabilities). We argue that an antifragile approach -- Rather than striving to rapidly reduce current uncertainties, the emphasis is on leveraging those uncertainties to better prepare for potentially greater, more unpredictable uncertainties in the future -- is pivotal for the long-term reliability of open-ended ML systems. In this position paper, we first identify key limitations of static testing, including scenario diversity, reward hacking, and over-alignment. We then explore the potential of antifragile solutions to manage rare events. Crucially, we advocate for a fundamental recalibration of the methods used to measure, benchmark, and continually improve AI safety over the long term, complementing existing robustness approaches by providing ethical and practical guidelines towards fostering an antifragile AI safety community. 

---
# FRIT: Using Causal Importance to Improve Chain-of-Thought Faithfulness 

**Authors**: Anand Swaroop, Akshat Nallani, Saksham Uboweja, Adiliia Uzdenova, Michael Nguyen, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2509.13334)  

**Abstract**: Chain-of-thought (CoT) reasoning has emerged as a powerful tool for improving large language model performance on complex tasks, but recent work shows that reasoning steps often fail to causally influence the final answer, creating brittle and untrustworthy outputs. Prior approaches focus primarily on measuring faithfulness, while methods for systematically improving it remain limited. We introduce Faithful Reasoning via Intervention Training (FRIT), a scalable alignment method that trains models to produce causally consistent reasoning by learning from systematically corrupted examples. FRIT generates synthetic training data by intervening on individual reasoning steps in model-generated CoTs, creating faithful/unfaithful pairs that highlight when reasoning breaks down. We then apply Direct Preference Optimization to teach models to prefer causally consistent reasoning paths. Evaluating on Qwen3-8B and Mistral-7B-v0.1 across factual and symbolic reasoning tasks, FRIT increases faithful reasoning by $3.4$ percentage points for Mistral on GSM8K while improving accuracy by $7.6$ percentage points. Our approach provides the first scalable, supervision-free method for training language models to produce more reliable and interpretable reasoning, addressing a critical gap between reasoning performance and trustworthiness. We release our code at \href{this https URL}. 

---
# Evaluation Awareness Scales Predictably in Open-Weights Large Language Models 

**Authors**: Maheep Chaudhary, Ian Su, Nikhil Hooda, Nishith Shankar, Julia Tan, Kevin Zhu, Ashwinee Panda, Ryan Lagasse, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.13333)  

**Abstract**: Large language models (LLMs) can internally distinguish between evaluation and deployment contexts, a behaviour known as \emph{evaluation awareness}. This undermines AI safety evaluations, as models may conceal dangerous capabilities during testing. Prior work demonstrated this in a single $70$B model, but the scaling relationship across model sizes remains unknown. We investigate evaluation awareness across $15$ models scaling from $0.27$B to $70$B parameters from four families using linear probing on steering vector activations. Our results reveal a clear power-law scaling: evaluation awareness increases predictably with model size. This scaling law enables forecasting deceptive behavior in future larger models and guides the design of scale-aware evaluation strategies for AI safety. A link to the implementation of this paper can be found at this https URL. 

---
# Explicit Reasoning Makes Better Judges: A Systematic Study on Accuracy, Efficiency, and Robustness 

**Authors**: Pratik Jayarao, Himanshu Gupta, Neeraj Varshney, Chaitanya Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13332)  

**Abstract**: As Large Language Models (LLMs) are increasingly adopted as automated judges in benchmarking and reward modeling, ensuring their reliability, efficiency, and robustness has become critical. In this work, we present a systematic comparison of "thinking" and "non-thinking" LLMs in the LLM-as-a-judge paradigm using open-source Qwen 3 models of relatively small sizes (0.6B, 1.7B, and 4B parameters). We evaluate both accuracy and computational efficiency (FLOPs) on RewardBench tasks, and further examine augmentation strategies for non-thinking models, including in-context learning, rubric-guided judging, reference-based evaluation, and n-best aggregation. Our results show that despite these enhancements, non-thinking models generally fall short of their thinking counterparts. Our results show that thinking models achieve approximately 10% points higher accuracy with little overhead (under 2x), in contrast to augmentation strategies like few-shot learning, which deliver modest gains at a higher cost (>8x). Bias and robustness analyses further demonstrate that thinking models maintain significantly greater consistency under a variety of bias conditions such as positional, bandwagon, identity, diversity, and random biases (6% higher on average). We further extend our experiments to the multilingual setting and our results confirm that explicit reasoning extends its benefits beyond English. Overall, our work results in several important findings that provide systematic evidence that explicit reasoning offers clear advantages in the LLM-as-a-judge paradigm not only in accuracy and efficiency but also in robustness. 

---
# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments 

**Authors**: Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni-Joan Solergibert, Barna Pasztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Ďurech, Ido Hakimi, Juan García Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolčec, Yixuan Xu, Michael Aerni, Badr AlKhamissi, Ines Altemir Marinas, Mohammad Hossein Amani, Matin Ansaripour, Ilia Badanin, Harold Benoit, Emanuela Boros, Nicholas Browning, Fabian Bösch, Maximilian Böther, Niklas Canova, Camille Challier, Clement Charmillot, Jonathan Coles, Jan Deriu, Arnout Devos, Lukas Drescher, Daniil Dzenhaliou, Maud Ehrmann, Dongyang Fan, Simin Fan, Silin Gao, Miguel Gila, María Grandury, Diba Hashemi, Alexander Hoyle, Jiaming Jiang, Mark Klein, Andrei Kucharavy, Anastasiia Kucherenko, Frederike Lübeck, Roman Machacek, Theofilos Manitaras, Andreas Marfurt, Kyle Matoba, Simon Matrenok, Henrique Mendoncça, Fawzi Roberto Mohamed, Syrielle Montariol, Luca Mouchel, Sven Najem-Meyer, Jingwei Ni, Gennaro Oliva, Matteo Pagliardini, Elia Palme, Andrei Panferov, Léo Paoletti, Marco Passerini, Ivan Pavlov, Auguste Poiroux, Kaustubh Ponkshe, Nathan Ranchin, Javi Rando, Mathieu Sauser, Jakhongir Saydaliev, Muhammad Ali Sayfiddinov, Marian Schneider, Stefano Schuppli, Marco Scialanga, Andrei Semenov, Kumar Shridhar, Raghav Singhal, Anna Sotnikova, Alexander Sternfeld, Ayush Kumar Tarun, Paul Teiletche, Jannis Vamvas, Xiaozhe Yao, Hao Zhao Alexander Ilic, Ana Klimovic, Andreas Krause, Caglar Gulcehre, David Rosenthal, Elliott Ash, Florian Tramèr, Joost VandeVondele, Livio Veraldi, Martin Rajman, Thomas Schulthess, Torsten Hoefler, Antoine Bosselut, Martin Jaggi, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2509.14233)  

**Abstract**: We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting this http URL exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension. 

---
# Language models' activations linearly encode training-order recency 

**Authors**: Dmitrii Krasheninnikov, Richard E. Turner, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2509.14223)  

**Abstract**: We show that language models' activations linearly encode when information was learned during training. Our setup involves creating a model with a known training order by sequentially fine-tuning Llama-3.2-1B on six disjoint but otherwise similar datasets about named entities. We find that the average activations of test samples for the six training datasets encode the training order: when projected into a 2D subspace, these centroids are arranged exactly in the order of training and lie on a straight line. Further, we show that linear probes can accurately (~90%) distinguish "early" vs. "late" entities, generalizing to entities unseen during the probes' own training. The model can also be fine-tuned to explicitly report an unseen entity's training stage (~80% accuracy). Interestingly, this temporal signal does not seem attributable to simple differences in activation magnitudes, losses, or model confidence. Our paper demonstrates that models are capable of differentiating information by its acquisition time, and carries significant implications for how they might manage conflicting data and respond to knowledge modifications. 

---
# A Universal Banach--Bregman Framework for Stochastic Iterations: Unifying Stochastic Mirror Descent, Learning and LLM Training 

**Authors**: Johnny R. Zhang, Xiaomei Mi, Gaoyuan Du, Qianyi Sun, Shiqi Wang, Jiaxuan Li, Wenhua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14216)  

**Abstract**: Stochastic optimization powers the scalability of modern artificial intelligence, spanning machine learning, deep learning, reinforcement learning, and large language model training. Yet, existing theory remains largely confined to Hilbert spaces, relying on inner-product frameworks and orthogonality. This paradigm fails to capture non-Euclidean settings, such as mirror descent on simplices, Bregman proximal methods for sparse learning, natural gradient descent in information geometry, or Kullback--Leibler-regularized language model training. Unlike Euclidean-based Hilbert-space methods, this approach embraces general Banach spaces. This work introduces a pioneering Banach--Bregman framework for stochastic iterations, establishing Bregman geometry as a foundation for next-generation optimization. It (i) provides a unified template via Bregman projections and Bregman--Fejer monotonicity, encompassing stochastic approximation, mirror descent, natural gradient, adaptive methods, and mirror-prox; (ii) establishes super-relaxations ($\lambda > 2$) in non-Hilbert settings, enabling flexible geometries and elucidating their acceleration effect; and (iii) delivers convergence theorems spanning almost-sure boundedness to geometric rates, validated on synthetic and real-world tasks. Empirical studies across machine learning (UCI benchmarks), deep learning (e.g., Transformer training), reinforcement learning (actor--critic), and large language models (WikiText-2 with distilGPT-2) show up to 20% faster convergence, reduced variance, and enhanced accuracy over classical baselines. These results position Banach--Bregman geometry as a cornerstone unifying optimization theory and practice across core AI paradigms. 

---
# Dense Video Understanding with Gated Residual Tokenization 

**Authors**: Haichao Zhang, Wenhao Chai, Shwai He, Ang Li, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14199)  

**Abstract**: High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding. 

---
# Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting 

**Authors**: Yifan Hu, Jie Yang, Tian Zhou, Peiyuan Liu, Yujin Tang, Rong Jin, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.14181)  

**Abstract**: Representation learning techniques like contrastive learning have long been explored in time series forecasting, mirroring their success in computer vision and natural language processing. Yet recent state-of-the-art (SOTA) forecasters seldom adopt these representation approaches because they have shown little performance advantage. We challenge this view and demonstrate that explicit representation alignment can supply critical information that bridges the distributional gap between input histories and future targets. To this end, we introduce TimeAlign, a lightweight, plug-and-play framework that learns auxiliary features via a simple reconstruction task and feeds them back to any base forecaster. Extensive experiments across eight benchmarks verify its superior performance. Further studies indicate that the gains arises primarily from correcting frequency mismatches between historical inputs and future outputs. We also provide a theoretical justification for the effectiveness of TimeAlign in increasing the mutual information between learned representations and predicted targets. As it is architecture-agnostic and incurs negligible overhead, TimeAlign can serve as a general alignment module for modern deep learning time-series forecasting systems. The code is available at this https URL. 

---
# Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs 

**Authors**: Akhil Theerthala  

**Link**: [PDF](https://arxiv.org/pdf/2509.14180)  

**Abstract**: Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts. 

---
# TGPO: Tree-Guided Preference Optimization for Robust Web Agent Reinforcement Learning 

**Authors**: Ziyuan Chen, Zhenghui Zhao, Zhangye Han, Miancan Liu, Xianhang Ye, Yiqing Li, Hongbo Min, Jinkui Ren, Xiantao Zhang, Guitao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.14172)  

**Abstract**: With the rapid advancement of large language models and vision-language models, employing large models as Web Agents has become essential for automated web interaction. However, training Web Agents with reinforcement learning faces critical challenges including credit assignment misallocation, prohibitively high annotation costs, and reward sparsity. To address these issues, we propose Tree-Guided Preference Optimization (TGPO), an offline reinforcement learning framework that proposes a tree-structured trajectory representation merging semantically identical states across trajectories to eliminate label conflicts. Our framework incorporates a Process Reward Model that automatically generates fine-grained rewards through subgoal progress, redundancy detection, and action verification. Additionally, a dynamic weighting mechanism prioritizes high-impact decision points during training. Experiments on Online-Mind2Web and our self-constructed C-WebShop datasets demonstrate that TGPO significantly outperforms existing methods, achieving higher success rates with fewer redundant steps. 

---
# Where Do Tokens Go? Understanding Pruning Behaviors in STEP at High Resolutions 

**Authors**: Michal Szczepanski, Martyna Poreba, Karim Haroun  

**Link**: [PDF](https://arxiv.org/pdf/2509.14165)  

**Abstract**: Vision Transformers (ViTs) achieve state-of-the-art performance in semantic segmentation but are hindered by high computational and memory costs. To address this, we propose STEP (SuperToken and Early-Pruning), a hybrid token-reduction framework that combines dynamic patch merging and token pruning to enhance efficiency without significantly compromising accuracy. At the core of STEP is dCTS, a lightweight CNN-based policy network that enables flexible merging into superpatches. Encoder blocks integrate also early-exits to remove high-confident supertokens, lowering computational load. We evaluate our method on high-resolution semantic segmentation benchmarks, including images up to 1024 x 1024, and show that when dCTS is applied alone, the token count can be reduced by a factor of 2.5 compared to the standard 16 x 16 pixel patching scheme. This yields a 2.6x reduction in computational cost and a 3.4x increase in throughput when using ViT-Large as the backbone. Applying the full STEP framework further improves efficiency, reaching up to a 4x reduction in computational complexity and a 1.7x gain in inference speed, with a maximum accuracy drop of no more than 2.0%. With the proposed STEP configurations, up to 40% of tokens can be confidently predicted and halted before reaching the final encoder layer. 

---
# Reasoning Efficiently Through Adaptive Chain-of-Thought Compression: A Self-Optimizing Framework 

**Authors**: Kerui Huang, Shuhan Liu, Xing Hu, Tongtong Xu, Lingfeng Bao, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.14093)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances Large Language Models (LLMs) by prompting intermediate steps, improving accuracy and robustness in arithmetic, logic, and commonsense tasks. However, this benefit comes with high computational costs: longer outputs increase latency, memory usage, and KV-cache demands. These issues are especially critical in software engineering tasks where concise and deterministic outputs are required. To investigate these trade-offs, we conduct an empirical study based on code generation benchmarks. The results reveal that longer CoT does not always help. Excessive reasoning often causes truncation, accuracy drops, and latency up to five times higher, with failed outputs consistently longer than successful ones. These findings challenge the assumption that longer reasoning is inherently better and highlight the need for adaptive CoT control. Motivated by this, we propose SEER (Self-Enhancing Efficient Reasoning), an adaptive framework that compresses CoT while preserving accuracy. SEER combines Best-of-N sampling with task-aware adaptive filtering, dynamically adjusting thresholds based on pre-inference outputs to reduce verbosity and computational overhead. We then evaluate SEER on three software engineering tasks and one math task. On average, SEER shortens CoT by 42.1%, improves accuracy by reducing truncation, and eliminates most infinite loops. These results demonstrate SEER as a practical method to make CoT-enhanced LLMs more efficient and robust, even under resource constraints. 

---
# Queen Detection in Beehives via Environmental Sensor Fusion for Low-Power Edge Computing 

**Authors**: Chiara De Luca, Elisa Donati  

**Link**: [PDF](https://arxiv.org/pdf/2509.14061)  

**Abstract**: Queen bee presence is essential for the health and stability of honeybee colonies, yet current monitoring methods rely on manual inspections that are labor-intensive, disruptive, and impractical for large-scale beekeeping. While recent audio-based approaches have shown promise, they often require high power consumption, complex preprocessing, and are susceptible to ambient noise. To overcome these limitations, we propose a lightweight, multimodal system for queen detection based on environmental sensor fusion-specifically, temperature, humidity, and pressure differentials between the inside and outside of the hive. Our approach employs quantized decision tree inference on a commercial STM32 microcontroller, enabling real-time, low-power edge computing without compromising accuracy. We show that our system achieves over 99% queen detection accuracy using only environmental inputs, with audio features offering no significant performance gain. This work presents a scalable and sustainable solution for non-invasive hive monitoring, paving the way for autonomous, precision beekeeping using off-the-shelf, energy-efficient hardware. 

---
# Machines are more productive than humans until they aren't, and vice versa 

**Authors**: Riccardo Zanardelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.14057)  

**Abstract**: With the growth of artificial skills, organizations may increasingly confront with the problem of optimizing skill policy decisions guided by economic principles. This paper addresses the underlying complexity of this challenge by developing an in-silico framework based on Monte Carlo simulations grounded in empirical realism to analyze the economic impact of human and machine skills, individually or jointly deployed, in the execution of tasks presenting varying levels of complexity. Our results provide quantitative support for the established notions that automation tends to be the most economically-effective strategy for tasks characterized by low-to-medium generalization difficulty, while automation struggles to match the economic utility of human skills in more complex scenarios. Critically, our simulations highlight that combining human and machine skills can be the most effective strategy when a high level of generalization is required, but only if genuine augmentation is achieved. In contrast, when failing to realize this synergy, the human-machine policy is severely penalized by the inherent costs of its dual skill structure, causing it to destroy value and becoming the worst choice from an economic perspective. The takeaway for decision-makers is unambiguous: simply allocating human and machine skills to a task is insufficient, and a human-machine skill policy is neither a silver-bullet solution nor a low-risk compromise. Rather, it is a critical opportunity to boost competitiveness that demands a strong organizational commitment to enabling augmentation. Also, our findings show that improving the cost-effectiveness of machine skills over time, while useful, does not replace the fundamental need to focus on achieving augmentation. 

---
# Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices 

**Authors**: Jordi Grau-Haro, Ruben Ribes-Serrano, Javier Naranjo-Alcazar, Marta Garcia-Ballesteros, Pedro Zuccarello  

**Link**: [PDF](https://arxiv.org/pdf/2509.14049)  

**Abstract**: Convolutional Neural Networks (CNNs) have demonstrated exceptional performance in audio tagging tasks. However, deploying these models on resource-constrained devices like the Raspberry Pi poses challenges related to computational efficiency and thermal management. In this paper, a comprehensive evaluation of multiple convolutional neural network (CNN) architectures for audio tagging on the Raspberry Pi is conducted, encompassing all 1D and 2D models from the Pretrained Audio Neural Networks (PANNs) framework, a ConvNeXt-based model adapted for audio classification, as well as MobileNetV3 architectures. In addition, two PANNs-derived networks, CNN9 and CNN13, recently proposed, are also evaluated. To enhance deployment efficiency and portability across diverse hardware platforms, all models are converted to the Open Neural Network Exchange (ONNX) format. Unlike previous works that focus on a single model, our analysis encompasses a broader range of architectures and involves continuous 24-hour inference sessions to assess performance stability. Our experiments reveal that, with appropriate model selection and optimization, it is possible to maintain consistent inference latency and manage thermal behavior effectively over extended periods. These findings provide valuable insights for deploying audio tagging models in real-world edge computing scenarios. 

---
# Prompt2Auto: From Motion Prompt to Automated Control via Geometry-Invariant One-Shot Gaussian Process Learning 

**Authors**: Zewen Yang, Xiaobing Dai, Dongfa Zhang, Yu Li, Ziyang Meng, Bingkun Huang, Hamid Sadeghian, Sami Haddadin  

**Link**: [PDF](https://arxiv.org/pdf/2509.14040)  

**Abstract**: Learning from demonstration allows robots to acquire complex skills from human demonstrations, but conventional approaches often require large datasets and fail to generalize across coordinate transformations. In this paper, we propose Prompt2Auto, a geometry-invariant one-shot Gaussian process (GeoGP) learning framework that enables robots to perform human-guided automated control from a single motion prompt. A dataset-construction strategy based on coordinate transformations is introduced that enforces invariance to translation, rotation, and scaling, while supporting multi-step predictions. Moreover, GeoGP is robust to variations in the user's motion prompt and supports multi-skill autonomy. We validate the proposed approach through numerical simulations with the designed user graphical interface and two real-world robotic experiments, which demonstrate that the proposed method is effective, generalizes across tasks, and significantly reduces the demonstration burden. Project page is available at: this https URL 

---
# PhenoGnet: A Graph-Based Contrastive Learning Framework for Disease Similarity Prediction 

**Authors**: Ranga Baminiwatte, Kazi Jewel Rana, Aaron J. Masino  

**Link**: [PDF](https://arxiv.org/pdf/2509.14037)  

**Abstract**: Understanding disease similarity is critical for advancing diagnostics, drug discovery, and personalized treatment strategies. We present PhenoGnet, a novel graph-based contrastive learning framework designed to predict disease similarity by integrating gene functional interaction networks with the Human Phenotype Ontology (HPO). PhenoGnet comprises two key components: an intra-view model that separately encodes gene and phenotype graphs using Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), and a cross view model implemented as a shared weight multilayer perceptron (MLP) that aligns gene and phenotype embeddings through contrastive learning. The model is trained using known gene phenotype associations as positive pairs and randomly sampled unrelated pairs as negatives. Diseases are represented by the mean embeddings of their associated genes and/or phenotypes, and pairwise similarity is computed via cosine similarity. Evaluation on a curated benchmark of 1,100 similar and 866 dissimilar disease pairs demonstrates strong performance, with gene based embeddings achieving an AUCPR of 0.9012 and AUROC of 0.8764, outperforming existing state of the art methods. Notably, PhenoGnet captures latent biological relationships beyond direct overlap, offering a scalable and interpretable solution for disease similarity prediction. These results underscore its potential for enabling downstream applications in rare disease research and precision medicine. 

---
# SSL-SSAW: Self-Supervised Learning with Sigmoid Self-Attention Weighting for Question-Based Sign Language Translation 

**Authors**: Zekang Liu, Wei Feng, Fanhua Shang, Lianyu Hu, Jichao Feng, Liqing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.14036)  

**Abstract**: Sign Language Translation (SLT) bridges the communication gap between deaf people and hearing people, where dialogue provides crucial contextual cues to aid in translation. Building on this foundational concept, this paper proposes Question-based Sign Language Translation (QB-SLT), a novel task that explores the efficient integration of dialogue. Unlike gloss (sign language transcription) annotations, dialogue naturally occurs in communication and is easier to annotate. The key challenge lies in aligning multimodality features while leveraging the context of the question to improve translation. To address this issue, we propose a cross-modality Self-supervised Learning with Sigmoid Self-attention Weighting (SSL-SSAW) fusion method for sign language translation. Specifically, we employ contrastive learning to align multimodality features in QB-SLT, then introduce a Sigmoid Self-attention Weighting (SSAW) module for adaptive feature extraction from question and sign language sequences. Additionally, we leverage available question text through self-supervised learning to enhance representation and translation capabilities. We evaluated our approach on newly constructed CSL-Daily-QA and PHOENIX-2014T-QA datasets, where SSL-SSAW achieved SOTA performance. Notably, easily accessible question assistance can achieve or even surpass the performance of gloss assistance. Furthermore, visualization results demonstrate the effectiveness of incorporating dialogue in improving translation quality. 

---
# You Are What You Train: Effects of Data Composition on Training Context-aware Machine Translation Models 

**Authors**: Paweł Mąka, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.14031)  

**Abstract**: Achieving human-level translations requires leveraging context to ensure coherence and handle complex phenomena like pronoun disambiguation. Sparsity of contextually rich examples in the standard training data has been hypothesized as the reason for the difficulty of context utilization. In this work, we systematically validate this claim in both single- and multilingual settings by constructing training datasets with a controlled proportions of contextually relevant examples. We demonstrate a strong association between training data sparsity and model performance confirming sparsity as a key bottleneck. Importantly, we reveal that improvements in one contextual phenomenon do no generalize to others. While we observe some cross-lingual transfer, it is not significantly higher between languages within the same sub-family. Finally, we propose and empirically evaluate two training strategies designed to leverage the available data. These strategies improve context utilization, resulting in accuracy gains of up to 6 and 8 percentage points on the ctxPro evaluation in single- and multilingual settings respectively. 

---
# Hala Technical Report: Building Arabic-Centric Instruction & Translation Models at Scale 

**Authors**: Hasan Abed Al Kader Hammoud, Mohammad Zbeeb, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2509.14008)  

**Abstract**: We present Hala, a family of Arabic-centric instruction and translation models built with our translate-and-tune pipeline. We first compress a strong AR$\leftrightarrow$EN teacher to FP8 (yielding $\sim$2$\times$ higher throughput with no quality loss) and use it to create high-fidelity bilingual supervision. A lightweight language model LFM2-1.2B is then fine-tuned on this data and used to translate high-quality English instruction sets into Arabic, producing a million-scale corpus tailored to instruction following. We train Hala models at 350M, 700M, 1.2B, and 9B parameters, and apply slerp merging to balance Arabic specialization with base-model strengths. On Arabic-centric benchmarks, Hala achieves state-of-the-art results within both the "nano" ($\leq$2B) and "small" (7-9B) categories, outperforming their bases. We release models, data, evaluation, and recipes to accelerate research in Arabic NLP. 

---
# RFM-Editing: Rectified Flow Matching for Text-guided Audio Editing 

**Authors**: Liting Gao, Yi Yuan, Yaru Chen, Yuelan Cheng, Zhenbo Li, Juan Wen, Shubin Zhang, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14003)  

**Abstract**: Diffusion models have shown remarkable progress in text-to-audio generation. However, text-guided audio editing remains in its early stages. This task focuses on modifying the target content within an audio signal while preserving the rest, thus demanding precise localization and faithful editing according to the text prompt. Existing training-based and zero-shot methods that rely on full-caption or costly optimization often struggle with complex editing or lack practicality. In this work, we propose a novel end-to-end efficient rectified flow matching-based diffusion framework for audio editing, and construct a dataset featuring overlapping multi-event audio to support training and benchmarking in complex scenarios. Experiments show that our model achieves faithful semantic alignment without requiring auxiliary captions or masks, while maintaining competitive editing quality across metrics. 

---
# MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment 

**Authors**: Elena Camuffo, Francesco Barbato, Mete Ozay, Simone Milani, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2509.14001)  

**Abstract**: We introduce MOCHA (Multi-modal Objects-aware Cross-arcHitecture Alignment), a knowledge distillation approach that transfers region-level multimodal semantics from a large vision-language teacher (e.g., LLaVa) into a lightweight vision-only object detector student (e.g., YOLO). A translation module maps student features into a joint space, where the training of the student and translator is guided by a dual-objective loss that enforces both local alignment and global relational consistency. Unlike prior approaches focused on dense or global alignment, MOCHA operates at the object level, enabling efficient transfer of semantics without modifying the teacher or requiring textual input at inference. We validate our method across four personalized detection benchmarks under few-shot regimes. Results show consistent gains over baselines, with a +10.1 average score improvement. Despite its compact architecture, MOCHA reaches performance on par with larger multimodal models, proving its suitability for real-world deployment. 

---
# Slim-SC: Thought Pruning for Efficient Scaling with Self-Consistency 

**Authors**: Colin Hong, Xu Guo, Anand Chaanan Singh, Esha Choukse, Dmitrii Ustiugov  

**Link**: [PDF](https://arxiv.org/pdf/2509.13990)  

**Abstract**: Recently, Test-Time Scaling (TTS) has gained increasing attention for improving LLM reasoning performance at test time without retraining the model. A notable TTS technique is Self-Consistency (SC), which generates multiple reasoning chains in parallel and selects the final answer via majority voting. While effective, the order-of-magnitude computational overhead limits its broad deployment. Prior attempts to accelerate SC mainly rely on model-based confidence scores or heuristics with limited empirical support. For the first time, we theoretically and empirically analyze the inefficiencies of SC and reveal actionable opportunities for improvement. Building on these insights, we propose Slim-SC, a step-wise pruning strategy that identifies and removes redundant chains using inter-chain similarity at the thought level. Experiments on three STEM reasoning datasets and two recent LLM architectures show that Slim-SC reduces inference latency and KVC usage by up to 45% and 26%, respectively, with R1-Distill, while maintaining or improving accuracy, thus offering a simple yet efficient TTS alternative for SC. 

---
# Differential Privacy in Federated Learning: Mitigating Inference Attacks with Randomized Response 

**Authors**: Ozer Ozturk, Busra Buyuktanir, Gozde Karatas Baydogmus, Kazim Yildiz  

**Link**: [PDF](https://arxiv.org/pdf/2509.13987)  

**Abstract**: Machine learning models used for distributed architectures consisting of servers and clients require large amounts of data to achieve high accuracy. Data obtained from clients are collected on a central server for model training. However, storing data on a central server raises concerns about security and privacy. To address this issue, a federated learning architecture has been proposed. In federated learning, each client trains a local model using its own data. The trained models are periodically transmitted to the central server. The server then combines the received models using federated aggregation algorithms to obtain a global model. This global model is distributed back to the clients, and the process continues in a cyclical manner. Although preventing data from leaving the clients enhances security, certain concerns still remain. Attackers can perform inference attacks on the obtained models to approximate the training dataset, potentially causing data leakage. In this study, differential privacy was applied to address the aforementioned security vulnerability, and a performance analysis was conducted. The Data-Unaware Classification Based on Association (duCBA) algorithm was used as the federated aggregation method. Differential privacy was implemented on the data using the Randomized Response technique, and the trade-off between security and performance was examined under different epsilon values. As the epsilon value decreased, the model accuracy declined, and class prediction imbalances were observed. This indicates that higher levels of privacy do not always lead to practical outcomes and that the balance between security and performance must be carefully considered. 

---
# LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology 

**Authors**: Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, Rafael Ferreira da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2509.13978)  

**Abstract**: Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance. 

---
# An Empirical Study on Failures in Automated Issue Solving 

**Authors**: Simiao Liu, Fang Liu, Liehao Li, Xin Tan, Yinghao Zhu, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13941)  

**Abstract**: Automated issue solving seeks to autonomously identify and repair defective code snippets across an entire codebase. SWE-Bench has emerged as the most widely adopted benchmark for evaluating progress in this area. While LLM-based agentic tools show great promise, they still fail on a substantial portion of tasks. Moreover, current evaluations primarily report aggregate issue-solving rates, which obscure the underlying causes of success and failure, making it challenging to diagnose model weaknesses or guide targeted improvements. To bridge this gap, we first analyze the performance and efficiency of three SOTA tools, spanning both pipeline-based and agentic architectures, in automated issue solving tasks of SWE-Bench-Verified under varying task characteristics. Furthermore, to move from high-level performance metrics to underlying cause analysis, we conducted a systematic manual analysis of 150 failed instances. From this analysis, we developed a comprehensive taxonomy of failure modes comprising 3 primary phases, 9 main categories, and 25 fine-grained subcategories. Then we systematically analyze the distribution of the identified failure modes, the results reveal distinct failure fingerprints between the two architectural paradigms, with the majority of agentic failures stemming from flawed reasoning and cognitive deadlocks. Motivated by these insights, we propose a collaborative Expert-Executor framework. It introduces a supervisory Expert agent tasked with providing strategic oversight and course-correction for a primary Executor agent. This architecture is designed to correct flawed reasoning and break the cognitive deadlocks that frequently lead to failure. Experiments show that our framework solves 22.2% of previously intractable issues for a leading single agent. These findings pave the way for building more robust agents through diagnostic evaluation and collaborative design. 

---
# DSpAST: Disentangled Representations for Spatial Audio Reasoning with Large Language Models 

**Authors**: Kevin Wilkinghoff, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.13927)  

**Abstract**: Reasoning about spatial audio with large language models requires a spatial audio encoder as an acoustic front-end to obtain audio embeddings for further processing. Such an encoder needs to capture all information required to detect the type of sound events, as well as the direction and distance of their corresponding sources. Accomplishing this with a single audio encoder is demanding as the information required for each of these tasks is mostly independent of each other. As a result, the performance obtained with a single encoder is often worse than when using task-specific audio encoders. In this work, we present DSpAST, a novel audio encoder based on SpatialAST that learns disentangled representations of spatial audio while having only 0.2% additional parameters. Experiments on SpatialSoundQA with the spatial audio reasoning system BAT demonstrate that DSpAST significantly outperforms SpatialAST. 

---
# MAP: End-to-End Autonomous Driving with Map-Assisted Planning 

**Authors**: Huilin Yin, Yiming Kan, Daniel Watzenig  

**Link**: [PDF](https://arxiv.org/pdf/2509.13926)  

**Abstract**: In recent years, end-to-end autonomous driving has attracted increasing attention for its ability to jointly model perception, prediction, and planning within a unified framework. However, most existing approaches underutilize the online mapping module, leaving its potential to enhance trajectory planning largely untapped. This paper proposes MAP (Map-Assisted Planning), a novel map-assisted end-to-end trajectory planning framework. MAP explicitly integrates segmentation-based map features and the current ego status through a Plan-enhancing Online Mapping module, an Ego-status-guided Planning module, and a Weight Adapter based on current ego status. Experiments conducted on the DAIR-V2X-seq-SPD dataset demonstrate that the proposed method achieves a 16.6% reduction in L2 displacement error, a 56.2% reduction in off-road rate, and a 44.5% improvement in overall score compared to the UniV2X baseline, even without post-processing. Furthermore, it achieves top ranking in Track 2 of the End-to-End Autonomous Driving through V2X Cooperation Challenge of MEIS Workshop @CVPR2025, outperforming the second-best model by 39.5% in terms of overall score. These results highlight the effectiveness of explicitly leveraging semantic map features in planning and suggest new directions for improving structure design in end-to-end autonomous driving systems. Our code is available at this https URL 

---
# Ensemble of Pre-Trained Models for Long-Tailed Trajectory Prediction 

**Authors**: Divya Thuremella, Yi Yang, Simon Wanna, Lars Kunze, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2509.13914)  

**Abstract**: This work explores the application of ensemble modeling to the multidimensional regression problem of trajectory prediction for vehicles in urban environments. As newer and bigger state-of-the-art prediction models for autonomous driving continue to emerge, an important open challenge is the problem of how to combine the strengths of these big models without the need for costly re-training. We show how, perhaps surprisingly, combining state-of-the-art deep learning models out-of-the-box (without retraining or fine-tuning) with a simple confidence-weighted average method can enhance the overall prediction. Indeed, while combining trajectory prediction models is not straightforward, this simple approach enhances performance by 10% over the best prediction model, especially in the long-tailed metrics. We show that this performance improvement holds on both the NuScenes and Argoverse datasets, and that these improvements are made across the dataset distribution. The code for our work is open source. 

---
# Do Large Language Models Understand Word Senses? 

**Authors**: Domenico Meconi, Simone Stirpe, Federico Martelli, Leonardo Lavalle, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13905)  

**Abstract**: Understanding the meaning of words in context is a fundamental capability for Large Language Models (LLMs). Despite extensive evaluation efforts, the extent to which LLMs show evidence that they truly grasp word senses remains underexplored. In this paper, we address this gap by evaluating both i) the Word Sense Disambiguation (WSD) capabilities of instruction-tuned LLMs, comparing their performance to state-of-the-art systems specifically designed for the task, and ii) the ability of two top-performing open- and closed-source LLMs to understand word senses in three generative settings: definition generation, free-form explanation, and example generation. Notably, we find that, in the WSD task, leading models such as GPT-4o and DeepSeek-V3 achieve performance on par with specialized WSD systems, while also demonstrating greater robustness across domains and levels of difficulty. In the generation tasks, results reveal that LLMs can explain the meaning of words in context up to 98\% accuracy, with the highest performance observed in the free-form explanation task, which best aligns with their generative capabilities. 

---
# FedSSG: Expectation-Gated and History-Aware Drift Alignment for Federated Learning 

**Authors**: Zhanting Zhou, Jinshan Lai, Fengchun Zhang, Zeqin Wu, Fengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13895)  

**Abstract**: Non-IID data and partial participation induce client drift and inconsistent local optima in federated learning, causing unstable convergence and accuracy loss. We present FedSSG, a stochastic sampling-guided, history-aware drift alignment method. FedSSG maintains a per-client drift memory that accumulates local model differences as a lightweight sketch of historical gradients; crucially, it gates both the memory update and the local alignment term by a smooth function of the observed/expected participation ratio (a phase-by-expectation signal derived from the server sampler). This statistically grounded gate stays weak and smooth when sampling noise dominates early, then strengthens once participation statistics stabilize, contracting the local-global gap without extra communication. Across CIFAR-10/100 with 100/500 clients and 2-15 percent participation, FedSSG consistently outperforms strong drift-aware baselines and accelerates convergence; on our benchmarks it improves test accuracy by up to a few points (e.g., about +0.9 on CIFAR-10 and about +2.7 on CIFAR-100 on average over the top-2 baseline) and yields about 4.5x faster target-accuracy convergence on average. The method adds only O(d) client memory and a constant-time gate, and degrades gracefully to a mild regularizer under near-IID or uniform sampling. FedSSG shows that sampling statistics can be turned into a principled, history-aware phase control to stabilize and speed up federated training. 

---
# Synthetic Data Generation for Screen Time and App Usage 

**Authors**: Gustavo Kruger, Nikhil Sachdeva, Michael Sobolev  

**Link**: [PDF](https://arxiv.org/pdf/2509.13892)  

**Abstract**: Smartphone usage data can provide valuable insights for understanding interaction with technology and human behavior. However, collecting large-scale, in-the-wild smartphone usage logs is challenging due to high costs, privacy concerns, under representative user samples and biases like non-response that can skew results. These challenges call for exploring alternative approaches to obtain smartphone usage datasets. In this context, large language models (LLMs) such as Open AI's ChatGPT present a novel approach for synthetic smartphone usage data generation, addressing limitations of real-world data collection. We describe a case study on how four prompt strategies influenced the quality of generated smartphone usage data. We contribute with insights on prompt design and measures of data quality, reporting a prompting strategy comparison combining two factors, prompt level of detail (describing a user persona, describing the expected results characteristics) and seed data inclusion (with versus without an initial real usage example). Our findings suggest that using LLMs to generate structured and behaviorally plausible smartphone use datasets is feasible for some use cases, especially when using detailed prompts. Challenges remain in capturing diverse nuances of human behavioral patterns in a single synthetic dataset, and evaluating tradeoffs between data fidelity and diversity, suggesting the need for use-case-specific evaluation metrics and future research with more diverse seed data and different LLM models. 

---
# Combating Biomedical Misinformation through Multi-modal Claim Detection and Evidence-based Verification 

**Authors**: Mariano Barone, Antonio Romano, Giuseppe Riccio, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2509.13888)  

**Abstract**: Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical systems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminology, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combining Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language models with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of hallucinations, ensuring that generated outputs are grounded in verifiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the-art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: this https URL 

---
# Combining Evidence and Reasoning for Biomedical Fact-Checking 

**Authors**: Mariano Barone, Antonio Romano, Giuseppe Riccio, Marco Postiglione, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2509.13879)  

**Abstract**: Misinformation in healthcare, from vaccine hesitancy to unproven treatments, poses risks to public health and trust in medical sys- tems. While machine learning and natural language processing have advanced automated fact-checking, validating biomedical claims remains uniquely challenging due to complex terminol- ogy, the need for domain expertise, and the critical importance of grounding in scientific evidence. We introduce CER (Combin- ing Evidence and Reasoning), a novel framework for biomedical fact-checking that integrates scientific evidence retrieval, reasoning via large language models, and supervised veracity prediction. By integrating the text-generation capabilities of large language mod- els with advanced retrieval techniques for high-quality biomedical scientific evidence, CER effectively mitigates the risk of halluci- nations, ensuring that generated outputs are grounded in veri- fiable, evidence-based sources. Evaluations on expert-annotated datasets (HealthFC, BioASQ-7b, SciFact) demonstrate state-of-the- art performance and promising cross-dataset generalization. Code and data are released for transparency and reproducibility: https: //github.com/PRAISELab-PicusLab/CER. 

---
# Masked Diffusion Models as Energy Minimization 

**Authors**: Sitong Chen, Shen Nie, Jiacheng Sun, Zijin Feng, Zhenguo Li, Ji-Rong Wen, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.13866)  

**Abstract**: We present a systematic theoretical framework that interprets masked diffusion models (MDMs) as solutions to energy minimization problems in discrete optimal transport. Specifically, we prove that three distinct energy formulations--kinetic, conditional kinetic, and geodesic energy--are mathematically equivalent under the structure of MDMs, and that MDMs minimize all three when the mask schedule satisfies a closed-form optimality condition. This unification not only clarifies the theoretical foundations of MDMs, but also motivates practical improvements in sampling. By parameterizing interpolation schedules via Beta distributions, we reduce the schedule design space to a tractable 2D search, enabling efficient post-training tuning without model modification. Experiments on synthetic and real-world benchmarks demonstrate that our energy-inspired schedules outperform hand-crafted baselines, particularly in low-step sampling settings. 

---
# Understanding the Process of Human-AI Value Alignment 

**Authors**: Jack McKinlay, Marina De Vos, Janina A. Hoffmann, Andreas Theodorou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13854)  

**Abstract**: Background: Value alignment in computer science research is often used to refer to the process of aligning artificial intelligence with humans, but the way the phrase is used often lacks precision. Objectives: In this paper, we conduct a systematic literature review to advance the understanding of value alignment in artificial intelligence by characterising the topic in the context of its research literature. We use this to suggest a more precise definition of the term. Methods: We analyse 172 value alignment research articles that have been published in recent years and synthesise their content using thematic analyses. Results: Our analysis leads to six themes: value alignment drivers & approaches; challenges in value alignment; values in value alignment; cognitive processes in humans and AI; human-agent teaming; and designing and developing value-aligned systems. Conclusions: By analysing these themes in the context of the literature we define value alignment as an ongoing process between humans and autonomous agents that aims to express and implement abstract values in diverse contexts, while managing the cognitive limits of both humans and AI agents and also balancing the conflicting ethical and political demands generated by the values in different groups. Our analysis gives rise to a set of research challenges and opportunities in the field of value alignment for future work. 

---
# Towards a Physics Foundation Model 

**Authors**: Florian Wiesner, Matthias Wessling, Stephen Baek  

**Link**: [PDF](https://arxiv.org/pdf/2509.13805)  

**Abstract**: Foundation models have revolutionized natural language processing through a ``train once, deploy anywhere'' paradigm, where a single pre-trained model adapts to countless downstream tasks without retraining. Access to a Physics Foundation Model (PFM) would be transformative -- democratizing access to high-fidelity simulations, accelerating scientific discovery, and eliminating the need for specialized solver development. Yet current physics-aware machine learning approaches remain fundamentally limited to single, narrow domains and require retraining for each new system. We present the General Physics Transformer (GPhyT), trained on 1.8 TB of diverse simulation data, that demonstrates foundation model capabilities are achievable for physics. Our key insight is that transformers can learn to infer governing dynamics from context, enabling a single model to simulate fluid-solid interactions, shock waves, thermal convection, and multi-phase dynamics without being told the underlying equations. GPhyT achieves three critical breakthroughs: (1) superior performance across multiple physics domains, outperforming specialized architectures by up to 29x, (2) zero-shot generalization to entirely unseen physical systems through in-context learning, and (3) stable long-term predictions through 50-timestep rollouts. By establishing that a single model can learn generalizable physical principles from data alone, this work opens the path toward a universal PFM that could transform computational science and engineering. 

---
# Bridging the Synthetic-Real Gap: Supervised Domain Adaptation for Robust Spacecraft 6-DoF Pose Estimation 

**Authors**: Inder Pal Singh, Nidhal Eddine Chenni, Abd El Rahman Shabayek, Arunkumar Rathinam, Djamila Aouada  

**Link**: [PDF](https://arxiv.org/pdf/2509.13792)  

**Abstract**: Spacecraft Pose Estimation (SPE) is a fundamental capability for autonomous space operations such as rendezvous, docking, and in-orbit servicing. Hybrid pipelines that combine object detection, keypoint regression, and Perspective-n-Point (PnP) solvers have recently achieved strong results on synthetic datasets, yet their performance deteriorates sharply on real or lab-generated imagery due to the persistent synthetic-to-real domain gap. Existing unsupervised domain adaptation approaches aim to mitigate this issue but often underperform when a modest number of labeled target samples are available. In this work, we propose the first Supervised Domain Adaptation (SDA) framework tailored for SPE keypoint regression. Building on the Learning Invariant Representation and Risk (LIRR) paradigm, our method jointly optimizes domain-invariant representations and task-specific risk using both labeled synthetic and limited labeled real data, thereby reducing generalization error under domain shift. Extensive experiments on the SPEED+ benchmark demonstrate that our approach consistently outperforms source-only, fine-tuning, and oracle baselines. Notably, with only 5% labeled target data, our method matches or surpasses oracle performance trained on larger fractions of labeled data. The framework is lightweight, backbone-agnostic, and computationally efficient, offering a practical pathway toward robust and deployable spacecraft pose estimation in real-world space environments. 

---
# Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning 

**Authors**: Yangning Li, Tingwei Lu, Yinghui Li, Yankai Chen, Wei-Chieh Huang, Wenhao Jiang, Hui Wang, Hai-Tao Zheng, Philip S.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13790)  

**Abstract**: Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning. 

---
# BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching 

**Authors**: Hanshuai Cui, Zhiqing Tang, Zhifei Xu, Zhi Yao, Wenyi Zeng, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.13789)  

**Abstract**: Recent advancements in Diffusion Transformers (DiTs) have established them as the state-of-the-art method for video generation. However, their inherently sequential denoising process results in inevitable latency, limiting real-world applicability. Existing acceleration methods either compromise visual quality due to architectural modifications or fail to reuse intermediate features at proper granularity. Our analysis reveals that DiT blocks are the primary contributors to inference latency. Across diffusion timesteps, the feature variations of DiT blocks exhibit a U-shaped pattern with high similarity during intermediate timesteps, which suggests substantial computational redundancy. In this paper, we propose Block-Wise Caching (BWCache), a training-free method to accelerate DiT-based video generation. BWCache dynamically caches and reuses features from DiT blocks across diffusion timesteps. Furthermore, we introduce a similarity indicator that triggers feature reuse only when the differences between block features at adjacent timesteps fall below a threshold, thereby minimizing redundant computations while maintaining visual fidelity. Extensive experiments on several video diffusion models demonstrate that BWCache achieves up to 2.24$\times$ speedup with comparable visual quality. 

---
# Who is Introducing the Failure? Automatically Attributing Failures of Multi-Agent Systems via Spectrum Analysis 

**Authors**: Yu Ge, Linna Xie, Zhong Li, Yu Pei, Tian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13782)  

**Abstract**: Large Language Model Powered Multi-Agent Systems (MASs) are increasingly employed to automate complex real-world problems, such as programming and scientific discovery. Despite their promising, MASs are not without their flaws. However, failure attribution in MASs - pinpointing the specific agent actions responsible for failures - remains underexplored and labor-intensive, posing significant challenges for debugging and system improvement. To bridge this gap, we propose FAMAS, the first spectrum-based failure attribution approach for MASs, which operates through systematic trajectory replay and abstraction, followed by spectrum this http URL core idea of FAMAS is to estimate, from variations across repeated MAS executions, the likelihood that each agent action is responsible for the failure. In particular, we propose a novel suspiciousness formula tailored to MASs, which integrates two key factor groups, namely the agent behavior group and the action behavior group, to account for the agent activation patterns and the action activation patterns within the execution trajectories of MASs. Through expensive evaluations against 12 baselines on the Who and When benchmark, FAMAS demonstrates superior performance by outperforming all the methods in comparison. 

---
# Exploring Data and Parameter Efficient Strategies for Arabic Dialect Identifications 

**Authors**: Vani Kanjirangat, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13775)  

**Abstract**: This paper discusses our exploration of different data-efficient and parameter-efficient approaches to Arabic Dialect Identification (ADI). In particular, we investigate various soft-prompting strategies, including prefix-tuning, prompt-tuning, P-tuning, and P-tuning V2, as well as LoRA reparameterizations. For the data-efficient strategy, we analyze hard prompting with zero-shot and few-shot inferences to analyze the dialect identification capabilities of Large Language Models (LLMs). For the parameter-efficient PEFT approaches, we conducted our experiments using Arabic-specific encoder models on several major datasets. We also analyzed the n-shot inferences on open-source decoder-only models, a general multilingual model (Phi-3.5), and an Arabic-specific one(SILMA). We observed that the LLMs generally struggle to differentiate the dialectal nuances in the few-shot or zero-shot setups. The soft-prompted encoder variants perform better, while the LoRA-based fine-tuned models perform best, even surpassing full fine-tuning. 

---
# Scrub It Out! Erasing Sensitive Memorization in Code Language Models via Machine Unlearning 

**Authors**: Zhaoyang Chu, Yao Wan, Zhikun Zhang, Di Wang, Zhou Yang, Hongyu Zhang, Pan Zhou, Xuanhua Shi, Hai Jin, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2509.13755)  

**Abstract**: While Code Language Models (CLMs) have demonstrated superior performance in software engineering tasks such as code generation and summarization, recent empirical studies reveal a critical privacy vulnerability: these models exhibit unintended memorization of sensitive training data, enabling verbatim reproduction of confidential information when specifically prompted. To address this issue, several approaches, including training data de-duplication and differential privacy augmentation, have been proposed. However, these methods require full-model retraining for deployed CLMs, which incurs substantial computational costs. In this paper, we aim to answer the following research question: Can sensitive information memorized by CLMs be erased effectively and efficiently?
We conduct a pioneering investigation into erasing sensitive memorization in CLMs through machine unlearning - a post-hoc modification method that removes specific information from trained models without requiring full retraining. Specifically, we first quantify the memorization risks of sensitive data within CLM training datasets and curate a high-risk dataset of 50,000 sensitive memorized samples as unlearning targets. We study two widely used gradient ascent-based unlearning approaches: the vanilla and constraint-based methods, and introduce CodeEraser, an advanced variant that selectively unlearns sensitive memorized segments in code while preserving the structural integrity and functional correctness of the surrounding code. Extensive experiments on three families of CLMs, i.e., CodeParrot, CodeGen-Mono, and Qwen2.5-Coder, validate the effectiveness and efficiency of CodeEraser in erasing targeted sensitive memorization while maintaining model utility. 

---
# State Space Models over Directed Graphs 

**Authors**: Junzhi She, Xunkai Li, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13735)  

**Abstract**: Directed graphs are ubiquitous across numerous domains, where the directionality of edges encodes critical causal dependencies. However, existing GNNs and graph Transformers tailored for directed graphs face two major challenges: (1) effectively capturing long-range causal dependencies derived from directed edges; (2) balancing accuracy and training efficiency when processing large-scale graph datasets. In recent years, state space models (SSMs) have achieved substantial progress in causal sequence tasks, and their variants designed for graphs have demonstrated state-of-the-art accuracy while maintaining high efficiency across various graph learning benchmarks. However, existing graph state space models are exclusively designed for undirected graphs, which limits their performance in directed graph learning. To this end, we propose an innovative approach DirEgo2Token which sequentializes directed graphs via k-hop ego graphs. This marks the first systematic extension of state space models to the field of directed graph learning. Building upon this, we develop DirGraphSSM, a novel directed graph neural network architecture that implements state space models on directed graphs via the message-passing mechanism. Experimental results demonstrate that DirGraphSSM achieves state-of-the-art performance on three representative directed graph learning tasks while attaining competitive performance on two additional tasks with 1.5$\times $ to 2$\times $ training speed improvements compared to existing state-of-the-art models. 

---
# Mitigating Query Selection Bias in Referring Video Object Segmentation 

**Authors**: Dingwei Zhang, Dong Zhang, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13722)  

**Abstract**: Recently, query-based methods have achieved remarkable performance in Referring Video Object Segmentation (RVOS) by using textual static object queries to drive cross-modal alignment. However, these static queries are easily misled by distractors with similar appearance or motion, resulting in \emph{query selection bias}. To address this issue, we propose Triple Query Former (TQF), which factorizes the referring query into three specialized components: an appearance query for static attributes, an intra-frame interaction query for spatial relations, and an inter-frame motion query for temporal association. Instead of relying solely on textual embeddings, our queries are dynamically constructed by integrating both linguistic cues and visual guidance. Furthermore, we introduce two motion-aware aggregation modules that enhance object token representations: Intra-frame Interaction Aggregation incorporates position-aware interactions among objects within a single frame, while Inter-frame Motion Aggregation leverages trajectory-guided alignment across frames to ensure temporal coherence. Extensive experiments on multiple RVOS benchmarks demonstrate the advantages of TQF and the effectiveness of our structured query design and motion-aware aggregation modules. 

---
# Automated Triaging and Transfer Learning of Incident Learning Safety Reports Using Large Language Representational Models 

**Authors**: Peter Beidler, Mark Nguyen, Kevin Lybarger, Ola Holmberg, Eric Ford, John Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13706)  

**Abstract**: PURPOSE: Incident reports are an important tool for safety and quality improvement in healthcare, but manual review is time-consuming and requires subject matter expertise. Here we present a natural language processing (NLP) screening tool to detect high-severity incident reports in radiation oncology across two institutions.
METHODS AND MATERIALS: We used two text datasets to train and evaluate our NLP models: 7,094 reports from our institution (Inst.), and 571 from IAEA SAFRON (SF), all of which had severity scores labeled by clinical content experts. We trained and evaluated two types of models: baseline support vector machines (SVM) and BlueBERT which is a large language model pretrained on PubMed abstracts and hospitalized patient data. We assessed for generalizability of our model in two ways. First, we evaluated models trained using Inst.-train on SF-test. Second, we trained a BlueBERT_TRANSFER model that was first fine-tuned on Inst.-train then on SF-train before testing on SF-test set. To further analyze model performance, we also examined a subset of 59 reports from our Inst. dataset, which were manually edited for clarity.
RESULTS Classification performance on the Inst. test achieved AUROC 0.82 using SVM and 0.81 using BlueBERT. Without cross-institution transfer learning, performance on the SF test was limited to an AUROC of 0.42 using SVM and 0.56 using BlueBERT. BlueBERT_TRANSFER, which was fine-tuned on both datasets, improved the performance on SF test to AUROC 0.78. Performance of SVM, and BlueBERT_TRANSFER models on the manually curated Inst. reports (AUROC 0.85 and 0.74) was similar to human performance (AUROC 0.81).
CONCLUSION: In summary, we successfully developed cross-institution NLP models on incident report text from radiation oncology centers. These models were able to detect high-severity reports similarly to humans on a curated dataset. 

---
# DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models 

**Authors**: Xiao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.13702)  

**Abstract**: Large Language Model (LLM) hallucination is a significant barrier to their reliable deployment. Current methods like Retrieval-Augmented Generation (RAG) are often reactive. We introduce **Dynamic Self-reinforcing Calibration for Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that intervenes during autoregressive decoding. Inspired by dual-process cognitive theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During inference, these proxies dynamically steer a large target model by injecting a real-time steering vector, which is the difference between FAP and HDP logits, at each decoding step. This plug-and-play approach requires no modification to the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2% Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained the highest FActScore of 46.50. These results validate DSCC-HS as a principled and efficient solution for enhancing LLM factuality. 

---
# CraftMesh: High-Fidelity Generative Mesh Manipulation via Poisson Seamless Fusion 

**Authors**: James Jincheng, Youcheng Cai, Ligang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13688)  

**Abstract**: Controllable, high-fidelity mesh editing remains a significant challenge in 3D content creation. Existing generative methods often struggle with complex geometries and fail to produce detailed results. We propose CraftMesh, a novel framework for high-fidelity generative mesh manipulation via Poisson Seamless Fusion. Our key insight is to decompose mesh editing into a pipeline that leverages the strengths of 2D and 3D generative models: we edit a 2D reference image, then generate a region-specific 3D mesh, and seamlessly fuse it into the original model. We introduce two core techniques: Poisson Geometric Fusion, which utilizes a hybrid SDF/Mesh representation with normal blending to achieve harmonious geometric integration, and Poisson Texture Harmonization for visually consistent texture blending. Experimental results demonstrate that CraftMesh outperforms state-of-the-art methods, delivering superior global consistency and local detail in complex editing tasks. 

---
# Improving Context Fidelity via Native Retrieval-Augmented Reasoning 

**Authors**: Suyuchen Wang, Jinlin Wang, Xinyu Wang, Shiqi Li, Xiangru Tang, Sirui Hong, Xiao-Wen Chang, Chenglin Wu, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13683)  

**Abstract**: Large language models (LLMs) often struggle with context fidelity, producing inconsistent answers when responding to questions based on provided information. Existing approaches either rely on expensive supervised fine-tuning to generate evidence post-answer or train models to perform web searches without necessarily improving utilization of the given context. We propose CARE, a novel native retrieval-augmented reasoning framework that teaches LLMs to explicitly integrate in-context evidence within their reasoning process with the model's own retrieval capabilities. Our method requires limited labeled evidence data while significantly enhancing both retrieval accuracy and answer generation performance through strategically retrieved in-context tokens in the reasoning chain. Extensive experiments on multiple real-world and counterfactual QA benchmarks demonstrate that our approach substantially outperforms supervised fine-tuning, traditional retrieval-augmented generation methods, and external retrieval solutions. This work represents a fundamental advancement in making LLMs more accurate, reliable, and efficient for knowledge-intensive tasks. 

---
# Prompt Stability in Code LLMs: Measuring Sensitivity across Emotion- and Personality-Driven Variations 

**Authors**: Wei Ma, Yixiao Yang, Jingquan Ge, Xiaofei Xie, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13680)  

**Abstract**: Code generation models are widely used in software development, yet their sensitivity to prompt phrasing remains under-examined. Identical requirements expressed with different emotions or communication styles can yield divergent outputs, while most benchmarks emphasize only peak performance. We present PromptSE (Prompt Sensitivity Evaluation), a framework that creates semantically equivalent prompt variants with emotion and personality templates, and that evaluates stability using probability aware continuous scoring or using binary pass rates when logits are unavailable. The results are aggregated into a proposed area under curve metric (AUC-E) for cross model comparison. Across 14 models from three families (Llama, Qwen, and DeepSeek), our study shows that performance and stability behave as largely decoupled optimization objectives, and it reveals architectural and scale related patterns that challenge common assumptions about model robustness. The framework supports rapid screening for closed-source models as well as detailed stability analysis in research settings. PromptSE enables practitioners to quantify performance stability trade offs for deployment and model selection, positioning prompt stability as a complementary evaluation dimension alongside performance and fairness, and contributing to more trustworthy AI-assisted software development tools. 

---
# AgentCTG: Harnessing Multi-Agent Collaboration for Fine-Grained Precise Control in Text Generation 

**Authors**: Xinxu Zhou, Jiaqi Bai, Zhenqi Sun, Fanxiang Zeng, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13677)  

**Abstract**: Although significant progress has been made in many tasks within the field of Natural Language Processing (NLP), Controlled Text Generation (CTG) continues to face numerous challenges, particularly in achieving fine-grained conditional control over generation. Additionally, in real scenario and online applications, cost considerations, scalability, domain knowledge learning and more precise control are required, presenting more challenge for CTG. This paper introduces a novel and scalable framework, AgentCTG, which aims to enhance precise and complex control over the text generation by simulating the control and regulation mechanisms in multi-agent workflows. We explore various collaboration methods among different agents and introduce an auto-prompt module to further enhance the generation effectiveness. AgentCTG achieves state-of-the-art results on multiple public datasets. To validate its effectiveness in practical applications, we propose a new challenging Character-Driven Rewriting task, which aims to convert the original text into new text that conform to specific character profiles and simultaneously preserve the domain knowledge. When applied to online navigation with role-playing, our approach significantly enhances the driving experience through improved content delivery. By optimizing the generation of contextually relevant text, we enable a more immersive interaction within online communities, fostering greater personalization and user engagement. 

---
# Re-purposing SAM into Efficient Visual Projectors for MLLM-Based Referring Image Segmentation 

**Authors**: Xiaobo Yang, Xiaojin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2509.13676)  

**Abstract**: Recently, Referring Image Segmentation (RIS) frameworks that pair the Multimodal Large Language Model (MLLM) with the Segment Anything Model (SAM) have achieved impressive results. However, adapting MLLM to segmentation is computationally intensive, primarily due to visual token redundancy. We observe that traditional patch-wise visual projectors struggle to strike a balance between reducing the number of visual tokens and preserving semantic clarity, often retaining overly long token sequences to avoid performance drops. Inspired by text tokenizers, we propose a novel semantic visual projector that leverages semantic superpixels generated by SAM to identify "visual words" in an image. By compressing and projecting semantic superpixels as visual tokens, our approach adaptively shortens the token sequence according to scene complexity while minimizing semantic loss in compression. To mitigate loss of information, we propose a semantic superpixel positional embedding to strengthen MLLM's awareness of superpixel geometry and position, alongside a semantic superpixel aggregator to preserve both fine-grained details inside superpixels and global context outside. Experiments show that our method cuts visual tokens by 93% without compromising performance, notably speeding up MLLM training and inference, and outperforming existing compressive visual projectors on RIS. 

---
# CL$^2$GEC: A Multi-Discipline Benchmark for Continual Learning in Chinese Literature Grammatical Error Correction 

**Authors**: Shang Qin, Jingheng Ye, Yinghui Li, Hai-Tao Zheng, Qi Li, Jinxiao Shan, Zhixing Li, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.13672)  

**Abstract**: The growing demand for automated writing assistance in diverse academic domains highlights the need for robust Chinese Grammatical Error Correction (CGEC) systems that can adapt across disciplines. However, existing CGEC research largely lacks dedicated benchmarks for multi-disciplinary academic writing, overlooking continual learning (CL) as a promising solution to handle domain-specific linguistic variation and prevent catastrophic forgetting. To fill this crucial gap, we introduce CL$^2$GEC, the first Continual Learning benchmark for Chinese Literature Grammatical Error Correction, designed to evaluate adaptive CGEC across multiple academic fields. Our benchmark includes 10,000 human-annotated sentences spanning 10 disciplines, each exhibiting distinct linguistic styles and error patterns. CL$^2$GEC focuses on evaluating grammatical error correction in a continual learning setting, simulating sequential exposure to diverse academic disciplines to reflect real-world editorial dynamics. We evaluate large language models under sequential tuning, parameter-efficient adaptation, and four representative CL algorithms, using both standard GEC metrics and continual learning metrics adapted to task-level variation. Experimental results reveal that regularization-based methods mitigate forgetting more effectively than replay-based or naive sequential approaches. Our benchmark provides a rigorous foundation for future research in adaptive grammatical error correction across diverse academic domains. 

---
# DREAM: Domain-aware Reasoning for Efficient Autonomous Underwater Monitoring 

**Authors**: Zhenqi Wu, Abhinav Modi, Angelos Mavrogiannis, Kaustubh Joshi, Nikhil Chopra, Yiannis Aloimonos, Nare Karapetyan, Ioannis Rekleitis, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.13666)  

**Abstract**: The ocean is warming and acidifying, increasing the risk of mass mortality events for temperature-sensitive shellfish such as oysters. This motivates the development of long-term monitoring systems. However, human labor is costly and long-duration underwater work is highly hazardous, thus favoring robotic solutions as a safer and more efficient option. To enable underwater robots to make real-time, environment-aware decisions without human intervention, we must equip them with an intelligent "brain." This highlights the need for persistent,wide-area, and low-cost benthic monitoring. To this end, we present DREAM, a Vision Language Model (VLM)-guided autonomy framework for long-term underwater exploration and habitat monitoring. The results show that our framework is highly efficient in finding and exploring target objects (e.g., oysters, shipwrecks) without prior location information. In the oyster-monitoring task, our framework takes 31.5% less time than the previous baseline with the same amount of oysters. Compared to the vanilla VLM, it uses 23% fewer steps while covering 8.88% more oysters. In shipwreck scenes, our framework successfully explores and maps the wreck without collisions, requiring 27.5% fewer steps than the vanilla model and achieving 100% coverage, while the vanilla model achieves 60.23% average coverage in our shipwreck environments. 

---
# Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs 

**Authors**: Zhuoxuan Zhang, Jinhao Duan, Edward Kim, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13664)  

**Abstract**: Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior. 

---
# Deep Lookup Network 

**Authors**: Yulan Guo, Longguang Wang, Wendong Mao, Xiaoyu Dong, Yingqian Wang, Li Liu, Wei An  

**Link**: [PDF](https://arxiv.org/pdf/2509.13662)  

**Abstract**: Convolutional neural networks are constructed with massive operations with different types and are highly computationally intensive. Among these operations, multiplication operation is higher in computational complexity and usually requires {more} energy consumption with longer inference time than other operations, which hinders the deployment of convolutional neural networks on mobile devices. In many resource-limited edge devices, complicated operations can be calculated via lookup tables to reduce computational cost. Motivated by this, in this paper, we introduce a generic and efficient lookup operation which can be used as a basic operation for the construction of neural networks. Instead of calculating the multiplication of weights and activation values, simple yet efficient lookup operations are adopted to compute their responses. To enable end-to-end optimization of the lookup operation, we construct the lookup tables in a differentiable manner and propose several training strategies to promote their convergence. By replacing computationally expensive multiplication operations with our lookup operations, we develop lookup networks for the image classification, image super-resolution, and point cloud classification tasks. It is demonstrated that our lookup networks can benefit from the lookup operations to achieve higher efficiency in terms of energy consumption and inference speed while maintaining competitive performance to vanilla convolutional networks. Extensive experiments show that our lookup networks produce state-of-the-art performance on different tasks (both classification and regression tasks) and different data types (both images and point clouds). 

---
# GitHub's Copilot Code Review: Can AI Spot Security Flaws Before You Commit? 

**Authors**: Amena Amro, Manar H. Alalfi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13650)  

**Abstract**: As software development practices increasingly adopt AI-powered tools, ensuring that such tools can support secure coding has become critical. This study evaluates the effectiveness of GitHub Copilot's recently introduced code review feature in detecting security vulnerabilities. Using a curated set of labeled vulnerable code samples drawn from diverse open-source projects spanning multiple programming languages and application domains, we systematically assessed Copilot's ability to identify and provide feedback on common security flaws. Contrary to expectations, our results reveal that Copilot's code review frequently fails to detect critical vulnerabilities such as SQL injection, cross-site scripting (XSS), and insecure deserialization. Instead, its feedback primarily addresses low-severity issues, such as coding style and typographical errors. These findings expose a significant gap between the perceived capabilities of AI-assisted code review and its actual effectiveness in supporting secure development practices. Our results highlight the continued necessity of dedicated security tools and manual code audits to ensure robust software security. 

---
# DeepLogit: A sequentially constrained explainable deep learning modeling approach for transport policy analysis 

**Authors**: Jeremy Oon, Rakhi Manohar Mepparambath, Ling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.13633)  

**Abstract**: Despite the significant progress of deep learning models in multitude of applications, their adaption in planning and policy related areas remains challenging due to the black-box nature of these models. In this work, we develop a set of DeepLogit models that follow a novel sequentially constrained approach in estimating deep learning models for transport policy analysis. In the first step of the proposed approach, we estimate a convolutional neural network (CNN) model with only linear terms, which is equivalent of a linear-in-parameter multinomial logit model. We then estimate other deep learning models by constraining the parameters that need interpretability at the values obtained in the linear-in-parameter CNN model and including higher order terms or by introducing advanced deep learning architectures like Transformers. Our approach can retain the interpretability of the selected parameters, yet provides significantly improved model accuracy than the discrete choice model. We demonstrate our approach on a transit route choice example using real-world transit smart card data from Singapore. This study shows the potential for a unifying approach, where theory-based discrete choice model (DCM) and data-driven AI models can leverage each other's strengths in interpretability and predictive power. With the availability of larger datasets and more complex constructions, such approach can lead to more accurate models using discrete choice models while maintaining its applicability in planning and policy-related areas. Our code is available on this https URL . 

---
# Secure, Scalable and Privacy Aware Data Strategy in Cloud 

**Authors**: Vijay Kumar Butte, Sujata Butte  

**Link**: [PDF](https://arxiv.org/pdf/2509.13627)  

**Abstract**: The enterprises today are faced with the tough challenge of processing, storing large amounts of data in a secure, scalable manner and enabling decision makers to make quick, informed data driven decisions. This paper addresses this challenge and develops an effective enterprise data strategy in the cloud. Various components of an effective data strategy are discussed and architectures addressing security, scalability and privacy aspects are provided. 

---
# Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval 

**Authors**: Amanda Chan, James Jiayu Liu, He Kai, Onno P. Kampman  

**Link**: [PDF](https://arxiv.org/pdf/2509.13626)  

**Abstract**: Access to reliable mental health information is vital for early help-seeking, yet expanding knowledge bases is resource-intensive and often misaligned with user needs. This results in poor performance of retrieval systems when presented concerns are not covered or expressed in informal or contextualized language. We present an AI-based gap-informed framework for corpus augmentation that authentically identifies underrepresented topics (gaps) by overlaying naturalistic user data such as forum posts in order to prioritize expansions based on coverage and usefulness. In a case study, we compare Directed (gap-informed augmentations) with Non-Directed augmentation (random additions), evaluating the relevance and usefulness of retrieved information across four retrieval-augmented generation (RAG) pipelines. Directed augmentation achieved near-optimal performance with modest expansions--requiring only a 42% increase for Query Transformation, 74% for Reranking and Hierarchical, and 318% for Baseline--to reach ~95% of the performance of an exhaustive reference corpus. In contrast, Non-Directed augmentation required substantially larger and thus practically infeasible expansions to achieve comparable performance (232%, 318%, 403%, and 763%, respectively). These results show that strategically targeted corpus growth can reduce content creation demands while sustaining high retrieval and provision quality, offering a scalable approach for building trusted health information repositories and supporting generative AI applications in high-stakes domains. 

---
# A reduced-order derivative-informed neural operator for subsurface fluid-flow 

**Authors**: Jeongjin, Park, Grant Bruer, Huseyin Tuna Erdinc, Abhinav Prakash Gahlot, Felix J. Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.13620)  

**Abstract**: Neural operators have emerged as cost-effective surrogates for expensive fluid-flow simulators, particularly in computationally intensive tasks such as permeability inversion from time-lapse seismic data, and uncertainty quantification. In these applications, the fidelity of the surrogate's gradients with respect to system parameters is crucial, as the accuracy of downstream tasks, such as optimization and Bayesian inference, relies directly on the quality of the derivative information. Recent advances in physics-informed methods have leveraged derivative information to improve surrogate accuracy. However, incorporating explicit Jacobians can become computationally prohibitive, as the complexity typically scales quadratically with the number of input parameters. To address this limitation, we propose DeFINO (Derivative-based Fisher-score Informed Neural Operator), a reduced-order, derivative-informed training framework. DeFINO integrates Fourier neural operators (FNOs) with a novel derivative-based training strategy guided by the Fisher Information Matrix (FIM). By projecting Jacobians onto dominant eigen-directions identified by the FIM, DeFINO captures critical sensitivity information directly informed by observational data, significantly reducing computational expense. We validate DeFINO through synthetic experiments in the context of subsurface multi-phase fluid-flow, demonstrating improvements in gradient accuracy while maintaining robust forward predictions of underlying fluid dynamics. These results highlight DeFINO's potential to offer practical, scalable solutions for inversion problems in complex real-world scenarios, all at substantially reduced computational cost. 

---
# Modernizing Facebook Scoped Search: Keyword and Embedding Hybrid Retrieval with LLM Evaluation 

**Authors**: Yongye Su, Zeya Zhang, Jane Kou, Cheng Ju, Shubhojeet Sarkar, Yamin Wang, Ji Liu, Shengbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.13603)  

**Abstract**: Beyond general web-scale search, social network search uniquely enables users to retrieve information and discover potential connections within their social context. We introduce a framework of modernized Facebook Group Scoped Search by blending traditional keyword-based retrieval with embedding-based retrieval (EBR) to improve the search relevance and diversity of search results. Our system integrates semantic retrieval into the existing keyword search pipeline, enabling users to discover more contextually relevant group posts. To rigorously assess the impact of this blended approach, we introduce a novel evaluation framework that leverages large language models (LLMs) to perform offline relevance assessments, providing scalable and consistent quality benchmarks. Our results demonstrate that the blended retrieval system significantly enhances user engagement and search quality, as validated by both online metrics and LLM-based evaluation. This work offers practical insights for deploying and evaluating advanced retrieval systems in large-scale, real-world social platforms. 

---
# Agentic JWT: A Secure Delegation Protocol for Autonomous AI Agents 

**Authors**: Abhishek Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2509.13597)  

**Abstract**: Autonomous LLM agents can issue thousands of API calls per hour without human oversight. OAuth 2.0 assumes deterministic clients, but in agentic settings stochastic reasoning, prompt injection, or multi-agent orchestration can silently expand privileges.
We introduce Agentic JWT (A-JWT), a dual-faceted intent token that binds each agent's action to verifiable user intent and, optionally, to a specific workflow step. A-JWT carries an agent's identity as a one-way checksum hash derived from its prompt, tools and configuration, and a chained delegation assertion to prove which downstream agent may execute a given task, and per-agent proof-of-possession keys to prevent replay and in-process impersonation. We define a new authorization mechanism and add a lightweight client shim library that self-verifies code at run time, mints intent tokens, tracks workflow steps and derives keys, thus enabling secure agent identity and separation even within a single process.
We illustrate a comprehensive threat model for agentic applications, implement a Python proof-of-concept and show functional blocking of scope-violating requests, replay, impersonation, and prompt-injection pathways with sub-millisecond overhead on commodity hardware. The design aligns with ongoing OAuth agent discussions and offers a drop-in path toward zero-trust guarantees for agentic applications. A comprehensive performance and security evaluation with experimental results will appear in our forthcoming journal publication 

---
# Intelligent Healthcare Imaging Platform An VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation 

**Authors**: Samer Al-Hamadani  

**Link**: [PDF](https://arxiv.org/pdf/2509.13590)  

**Abstract**: The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption. 

---
# TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning 

**Authors**: Momchil S. Tomov, Sang Uk Lee, Hansford Hendrago, Jinwook Huh, Teawon Han, Forbes Howington, Rafael da Silva, Gianmarco Bernasconi, Marc Heim, Samuel Findler, Xiaonan Ji, Alexander Boule, Michael Napoli, Kuo Chen, Jesse Miller, Boaz Floor, Yunqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13579)  

**Abstract**: We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving. 

---
# Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies: Mitigating Multi-Step Inference Degradation 

**Authors**: Zidong Chen, Zihao Guo, Peng Wang, ThankGod Itua Egbe, Yan Lyu, Chenghao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.13574)  

**Abstract**: Flow matching has emerged as a competitive framework for learning high-quality generative policies in robotics; however, we find that generalisation arises and saturates early along the flow trajectory, in accordance with recent findings in the literature. We further observe that increasing the number of Euler integration steps during inference counter-intuitively and universally degrades policy performance. We attribute this to (i) additional, uniformly spaced integration steps oversample the late-time region, thereby constraining actions towards the training trajectories and reducing generalisation; and (ii) the learned velocity field becoming non-Lipschitz as integration time approaches 1, causing instability. To address these issues, we propose a novel policy that utilises non-uniform time scheduling (e.g., U-shaped) during training, which emphasises both early and late temporal stages to regularise policy training, and a dense-jump integration schedule at inference, which uses a single-step integration to replace the multi-step integration beyond a jump point, to avoid unstable areas around 1. Essentially, our policy is an efficient one-step learner that still pushes forward performance through multi-step integration, yielding up to 23.7% performance gains over state-of-the-art baselines across diverse robotic tasks. 

---
# Complexity Bounds for Smooth Convex Multiobjective Optimization 

**Authors**: Phillipe R. Sampaio  

**Link**: [PDF](https://arxiv.org/pdf/2509.13550)  

**Abstract**: We study the oracle complexity of finding $\varepsilon$-Pareto stationary points in smooth multiobjective optimization with $m$ objectives. The progress metric is the Pareto stationarity gap $\mathcal{G}(x)$ (the norm of an optimal convex combination of gradients). Our contributions are fourfold. (i) For strongly convex objectives, any span first-order method (iterates lie in the span of past gradients) exhibits linear convergence no faster than $\exp(-\Theta(T/\sqrt{\kappa}))$ after $T$ oracle calls, where $\kappa$ is the condition number, implying $\Theta(\sqrt{\kappa}\log(1/\varepsilon))$ iterations; this matches classical accelerated upper bounds. (ii) For convex problems and oblivious one-step methods (a fixed scalarization with pre-scheduled step sizes), we prove a lower bound of order $1/T$ on the best gradient norm among the first $T$ iterates. (iii) Although accelerated gradient descent is outside this restricted class, it is an oblivious span method and attains the same $1/T$ upper rate on a fixed scalarization. (iv) For convex problems and general span methods with adaptive scalarizations, we establish a universal lower bound of order $1/T^{2}$ on the gradient norm of the final iterate after $T$ steps, highlighting a gap between known upper bounds and worst-case guarantees. All bounds hold on non-degenerate instances with distinct objectives and non-singleton Pareto fronts; rates are stated up to universal constants and natural problem scaling. 

---
# ColonCrafter: A Depth Estimation Model for Colonoscopy Videos Using Diffusion Priors 

**Authors**: Romain Hardy, Tyler Berzin, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.13525)  

**Abstract**: Three-dimensional (3D) scene understanding in colonoscopy presents significant challenges that necessitate automated methods for accurate depth estimation. However, existing depth estimation models for endoscopy struggle with temporal consistency across video sequences, limiting their applicability for 3D reconstruction. We present ColonCrafter, a diffusion-based depth estimation model that generates temporally consistent depth maps from monocular colonoscopy videos. Our approach learns robust geometric priors from synthetic colonoscopy sequences to generate temporally consistent depth maps. We also introduce a style transfer technique that preserves geometric structure while adapting real clinical videos to match our synthetic training domain. ColonCrafter achieves state-of-the-art zero-shot performance on the C3VD dataset, outperforming both general-purpose and endoscopy-specific approaches. Although full trajectory 3D reconstruction remains a challenge, we demonstrate clinically relevant applications of ColonCrafter, including 3D point cloud generation and surface coverage assessment. 

---
# Reproducible workflow for online AI in digital health 

**Authors**: Susobhan Ghosh, Bhanu T. Gulapalli, Daiqi Gao, Asim Gazi, Anna Trella, Ziping Xu, Kelly Zhang, Susan A. Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2509.13499)  

**Abstract**: Online artificial intelligence (AI) algorithms are an important component of digital health interventions. These online algorithms are designed to continually learn and improve their performance as streaming data is collected on individuals. Deploying online AI presents a key challenge: balancing adaptability of online AI with reproducibility. Online AI in digital interventions is a rapidly evolving area, driven by advances in algorithms, sensors, software, and devices. Digital health intervention development and deployment is a continuous process, where implementation - including the AI decision-making algorithm - is interspersed with cycles of re-development and optimization. Each deployment informs the next, making iterative deployment a defining characteristic of this field. This iterative nature underscores the importance of reproducibility: data collected across deployments must be accurately stored to have scientific utility, algorithm behavior must be auditable, and results must be comparable over time to facilitate scientific discovery and trustworthy refinement. This paper proposes a reproducible scientific workflow for developing, deploying, and analyzing online AI decision-making algorithms in digital health interventions. Grounded in practical experience from multiple real-world deployments, this workflow addresses key challenges to reproducibility across all phases of the online AI algorithm development life-cycle. 

---
# Prompt2DAG: A Modular Methodology for LLM-Based Data Enrichment Pipeline Generation 

**Authors**: Abubakari Alidu, Michele Ciavotta, Flavio DePaoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.13487)  

**Abstract**: Developing reliable data enrichment pipelines demands significant engineering expertise. We present Prompt2DAG, a methodology that transforms natural language descriptions into executable Apache Airflow DAGs. We evaluate four generation approaches -- Direct, LLM-only, Hybrid, and Template-based -- across 260 experiments using thirteen LLMs and five case studies to identify optimal strategies for production-grade automation. Performance is measured using a penalized scoring framework that combines reliability with code quality (SAT), structural integrity (DST), and executability (PCT). The Hybrid approach emerges as the optimal generative method, achieving a 78.5% success rate with robust quality scores (SAT: 6.79, DST: 7.67, PCT: 7.76). This significantly outperforms the LLM-only (66.2% success) and Direct (29.2% success) methods. Our findings show that reliability, not intrinsic code quality, is the primary differentiator. Cost-effectiveness analysis reveals the Hybrid method is over twice as efficient as Direct prompting per successful DAG. We conclude that a structured, hybrid approach is essential for balancing flexibility and reliability in automated workflow generation, offering a viable path to democratize data pipeline development. 

---
# An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software 

**Authors**: Sina Gogani-Khiabani, Ashutosh Trivedi, Diptikalyan Saha, Saeid Tizpaz-Niari  

**Link**: [PDF](https://arxiv.org/pdf/2509.13471)  

**Abstract**: Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications. 

---
# MapAnything: Universal Feed-Forward Metric 3D Reconstruction 

**Authors**: Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2509.13414)  

**Abstract**: We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone. 

---
# Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews 

**Authors**: Sai Suresh Marchala Vasu, Ivaxi Sheth, Hui-Po Wang, Ruta Binkyte, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2509.13400)  

**Abstract**: The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings. 

---
# EdiVal-Agent: An Object-Centric Framework for Automated, Scalable, Fine-Grained Evaluation of Multi-Turn Editing 

**Authors**: Tianyu Chen, Yasi Zhang, Zhi Zhang, Peiyu Yu, Shu Wang, Zhendong Wang, Kevin Lin, Xiaofei Wang, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Jianwen Xie, Oscar Leong, Lijuan Wang, Ying Nian Wu, Mingyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13399)  

**Abstract**: Instruction-based image editing has advanced rapidly, yet reliable and interpretable evaluation remains a bottleneck. Current protocols either (i) depend on paired reference images -- resulting in limited coverage and inheriting biases from prior generative models -- or (ii) rely solely on zero-shot vision--language models (VLMs), whose prompt-based assessments of instruction following, content consistency, and visual quality are often imprecise.
To address this, we introduce EdiVal-Agent, an automated, scalable, and fine-grained evaluation framework for multi-turn instruction-based editing from an object-centric perspective, supported by a suite of expert tools. Given an image, EdiVal-Agent first decomposes it into semantically meaningful objects, then synthesizes diverse, context-aware editing instructions. For evaluation, it integrates VLMs with open-vocabulary object detectors to assess instruction following, uses semantic-level feature extractors to evaluate content consistency, and leverages human preference models to judge visual quality. We show that combining VLMs with object detectors yields stronger agreement with human judgments in instruction-following evaluation compared to using VLMs alone and CLIP-based metrics. Furthermore, the pipeline's modular design allows future tools to be seamlessly integrated, enhancing evaluation accuracy over time.
Instantiating this pipeline, we build EdiVal-Bench, a multi-turn editing benchmark covering 9 instruction types and 11 state-of-the-art editing models spanning autoregressive (AR) (including Nano Banana, GPT-Image-1), flow-matching, and diffusion paradigms. We demonstrate that EdiVal-Agent can be used to identify existing failure modes, thereby informing the development of the next generation of editing models. Project page: this https URL. 

---
# The threat of analytic flexibility in using large language models to simulate human data: A call to attention 

**Authors**: Jamie Cummins  

**Link**: [PDF](https://arxiv.org/pdf/2509.13397)  

**Abstract**: Social scientists are now using large language models to create "silicon samples" - synthetic datasets intended to stand in for human respondents, aimed at revolutionising human subjects research. However, there are many analytic choices which must be made to produce these samples. Though many of these choices are defensible, their impact on sample quality is poorly understood. I map out these analytic choices and demonstrate how a very small number of decisions can dramatically change the correspondence between silicon samples and human data. Configurations (N = 252) varied substantially in their capacity to estimate (i) rank ordering of participants, (ii) response distributions, and (iii) between-scale correlations. Most critically, configurations were not consistent in quality: those that performed well on one dimension often performed poorly on another, implying that there is no "one-size-fits-all" configuration that optimises the accuracy of these samples. I call for greater attention to the threat of analytic flexibility in using silicon samples. 

---
# TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models 

**Authors**: Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.13395)  

**Abstract**: Speech foundation models have recently demonstrated the ability to perform Speech In-Context Learning (SICL). Selecting effective in-context examples is crucial for SICL performance, yet selection methodologies remain underexplored. In this work, we propose Text-Embedding KNN for SICL (TICL), a simple pipeline that uses semantic context to enhance off-the-shelf large multimodal models' speech recognition ability without fine-tuning. Across challenging automatic speech recognition tasks, including accented English, multilingual speech, and children's speech, our method enables models to surpass zero-shot performance with up to 84.7% relative WER reduction. We conduct ablation studies to show the robustness and efficiency of our method. 

---
# The Intercepted Self: How Generative AI Challenges the Dynamics of the Relational Self 

**Authors**: Sandrine R. Schiller, Camilo Miguel Signorelli, Filippos Stamatiou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13391)  

**Abstract**: Generative AI is changing our way of interacting with technology, others, and ourselves. Systems such as Microsoft copilot, Gemini and the expected Apple intelligence still awaits our prompt for action. Yet, it is likely that AI assistant systems will only become better at predicting our behaviour and acting on our behalf. Imagine new generations of generative and predictive AI deciding what you might like best at a new restaurant, picking an outfit that increases your chances on your date with a partner also chosen by the same or a similar system. Far from a science fiction scenario, the goal of several research programs is to build systems capable of assisting us in exactly this manner. The prospect urges us to rethink human-technology relations, but it also invites us to question how such systems might change the way we relate to ourselves. Building on our conception of the relational self, we question the possible effects of generative AI with respect to what we call the sphere of externalised output, the contextual sphere and the sphere of self-relating. In this paper, we attempt to deepen the existential considerations accompanying the AI revolution by outlining how generative AI enables the fulfilment of tasks and also increasingly anticipates, i.e. intercepts, our initiatives in these different spheres. 

---
# A Domain Knowledge Informed Approach for Anomaly Detection of Electric Vehicle Interior Sounds 

**Authors**: Deepti Kunte, Bram Cornelis, Claudio Colangeli, Karl Janssens, Brecht Van Baelen, Konstantinos Gryllias  

**Link**: [PDF](https://arxiv.org/pdf/2509.13390)  

**Abstract**: The detection of anomalies in automotive cabin sounds is critical for ensuring vehicle quality and maintaining passenger comfort. In many real-world settings, this task is more appropriately framed as an unsupervised learning problem rather than the supervised case due to the scarcity or complete absence of labeled faulty data. In such an unsupervised setting, the model is trained exclusively on healthy samples and detects anomalies as deviations from normal behavior. However, in the absence of labeled faulty samples for validation and the limited reliability of commonly used metrics, such as validation reconstruction error, effective model selection remains a significant challenge. To overcome these limitations, a domain-knowledge-informed approach for model selection is proposed, in which proxy-anomalies engineered through structured perturbations of healthy spectrograms are used in the validation set to support model selection. The proposed methodology is evaluated on a high-fidelity electric vehicle dataset comprising healthy and faulty cabin sounds across five representative fault types viz., Imbalance, Modulation, Whine, Wind, and Pulse Width Modulation. This dataset, generated using advanced sound synthesis techniques, and validated via expert jury assessments, has been made publicly available to facilitate further research. Experimental evaluations on the five fault cases demonstrate the selection of optimal models using proxy-anomalies, significantly outperform conventional model selection strategies. 

---
# Landcover classification and change detection using remote sensing and machine learning: a case study of Western Fiji 

**Authors**: Yadvendra Gurjar, Ruoni Wan, Ehsan Farahbakhsh, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2509.13388)  

**Abstract**: As a developing country, Fiji is facing rapid urbanisation, which is visible in the massive development projects that include housing, roads, and civil works. In this study, we present machine learning and remote sensing frameworks to compare land use and land cover change from 2013 to 2024 in Nadi, Fiji. The ultimate goal of this study is to provide technical support in land cover/land use modelling and change detection. We used Landsat-8 satellite image for the study region and created our training dataset with labels for supervised machine learning. We used Google Earth Engine and unsupervised machine learning via k-means clustering to generate the land cover map. We used convolutional neural networks to classify the selected regions' land cover types. We present a visualisation of change detection, highlighting urban area changes over time to monitor changes in the map. 

---
# Uncovering AI Governance Themes in EU Policies using BERTopic and Thematic Analysis 

**Authors**: Delaram Golpayegani, Marta Lasek-Markey, Arjumand Younus, Aphra Kerr, Dave Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.13387)  

**Abstract**: The upsurge of policies and guidelines that aim to ensure Artificial Intelligence (AI) systems are safe and trustworthy has led to a fragmented landscape of AI governance. The European Union (EU) is a key actor in the development of such policies and guidelines. Its High-Level Expert Group (HLEG) issued an influential set of guidelines for trustworthy AI, followed in 2024 by the adoption of the EU AI Act. While the EU policies and guidelines are expected to be aligned, they may differ in their scope, areas of emphasis, degrees of normativity, and priorities in relation to AI. To gain a broad understanding of AI governance from the EU perspective, we leverage qualitative thematic analysis approaches to uncover prevalent themes in key EU documents, including the AI Act and the HLEG Ethics Guidelines. We further employ quantitative topic modelling approaches, specifically through the use of the BERTopic model, to enhance the results and increase the document sample to include EU AI policy documents published post-2018. We present a novel perspective on EU policies, tracking the evolution of its approach to addressing AI governance. 

---
# ASTREA: Introducing Agentic Intelligence for Orbital Thermal Autonomy 

**Authors**: Alejandro D. Mousist  

**Link**: [PDF](https://arxiv.org/pdf/2509.13380)  

**Abstract**: This paper presents ASTREA, the first agentic system deployed on flight-heritage hardware (TRL 9) for autonomous spacecraft operations. Using thermal control as a representative use case, we integrate a resource-constrained Large Language Model (LLM) agent with a reinforcement learning controller in an asynchronous architecture tailored for space-qualified platforms. Ground experiments show that LLM-guided supervision improves thermal stability and reduces violations, confirming the feasibility of combining semantic reasoning with adaptive control under hardware constraints. However, on-orbit validation aboard the International Space Station (ISS) reveals performance degradation caused by inference latency mismatched with the rapid thermal cycles characteristic of Low Earth Orbit (LEO) satellites. These results highlight both the opportunities and current limitations of agentic LLM-based systems in real flight environments, providing practical design guidelines for future space autonomy. 

---
# An Empirical Analysis of VLM-based OOD Detection: Mechanisms, Advantages, and Sensitivity 

**Authors**: Yuxiao Lee, Xiaofeng Cao, Wei Ye, Jiangchao Yao, Jingkuan Song, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13375)  

**Abstract**: Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot out-of-distribution (OOD) detection capabilities, vital for reliable AI systems. Despite this promising capability, a comprehensive understanding of (1) why they work so effectively, (2) what advantages do they have over single-modal methods, and (3) how is their behavioral robustness -- remains notably incomplete within the research community. This paper presents a systematic empirical analysis of VLM-based OOD detection using in-distribution (ID) and OOD prompts. (1) Mechanisms: We systematically characterize and formalize key operational properties within the VLM embedding space that facilitate zero-shot OOD detection. (2) Advantages: We empirically quantify the superiority of these models over established single-modal approaches, attributing this distinct advantage to the VLM's capacity to leverage rich semantic novelty. (3) Sensitivity: We uncovers a significant and previously under-explored asymmetry in their robustness profile: while exhibiting resilience to common image noise, these VLM-based methods are highly sensitive to prompt phrasing. Our findings contribute a more structured understanding of the strengths and critical vulnerabilities inherent in VLM-based OOD detection, offering crucial, empirically-grounded guidance for developing more robust and reliable future designs. 

---
# Generative AI Pipeline for Interactive Prompt-driven 2D-to-3D Vascular Reconstruction for Fontan Geometries from Contrast-Enhanced X-Ray Fluoroscopy Imaging 

**Authors**: Prahlad G Menon  

**Link**: [PDF](https://arxiv.org/pdf/2509.13372)  

**Abstract**: Fontan palliation for univentricular congenital heart disease progresses to hemodynamic failure with complex flow patterns poorly characterized by conventional 2D imaging. Current assessment relies on fluoroscopic angiography, providing limited 3D geometric information essential for computational fluid dynamics (CFD) analysis and surgical planning.
A multi-step AI pipeline was developed utilizing Google's Gemini 2.5 Flash (2.5B parameters) for systematic, iterative processing of fluoroscopic angiograms through transformer-based neural architecture. The pipeline encompasses medical image preprocessing, vascular segmentation, contrast enhancement, artifact removal, and virtual hemodynamic flow visualization within 2D projections. Final views were processed through Tencent's Hunyuan3D-2mini (384M parameters) for stereolithography file generation.
The pipeline successfully generated geometrically optimized 2D projections from single-view angiograms after 16 processing steps using a custom web interface. Initial iterations contained hallucinated vascular features requiring iterative refinement to achieve anatomically faithful representations. Final projections demonstrated accurate preservation of complex Fontan geometry with enhanced contrast suitable for 3D conversion. AI-generated virtual flow visualization identified stagnation zones in central connections and flow patterns in branch arteries. Complete processing required under 15 minutes with second-level API response times.
This approach demonstrates clinical feasibility of generating CFD-suitable geometries from routine angiographic data, enabling 3D generation and rapid virtual flow visualization for cursory insights prior to full CFD simulation. While requiring refinement cycles for accuracy, this establishes foundation for democratizing advanced geometric and hemodynamic analysis using readily available imaging data. 

---
# The Provenance Problem: LLMs and the Breakdown of Citation Norms 

**Authors**: Brian D. Earp, Haotian Yuan, Julian Koplin, Sebastian Porsdam Mann  

**Link**: [PDF](https://arxiv.org/pdf/2509.13365)  

**Abstract**: The increasing use of generative AI in scientific writing raises urgent questions about attribution and intellectual credit. When a researcher employs ChatGPT to draft a manuscript, the resulting text may echo ideas from sources the author has never encountered. If an AI system reproduces insights from, for example, an obscure 1975 paper without citation, does this constitute plagiarism? We argue that such cases exemplify the 'provenance problem': a systematic breakdown in the chain of scholarly credit. Unlike conventional plagiarism, this phenomenon does not involve intent to deceive (researchers may disclose AI use and act in good faith) yet still benefit from the uncredited intellectual contributions of others. This dynamic creates a novel category of attributional harm that current ethical and professional frameworks fail to address. As generative AI becomes embedded across disciplines, the risk that significant ideas will circulate without recognition threatens both the reputational economy of science and the demands of epistemic justice. This Perspective analyzes how AI challenges established norms of authorship, introduces conceptual tools for understanding the provenance problem, and proposes strategies to preserve integrity and fairness in scholarly communication. 

---
# Evaluating undergraduate mathematics examinations in the era of generative AI: a curriculum-level case study 

**Authors**: Benjamin J. Walker, Beatriz Navarro Lameda, Ruth A. Reynolds  

**Link**: [PDF](https://arxiv.org/pdf/2509.13359)  

**Abstract**: Generative artificial intelligence (GenAI) tools such as OpenAI's ChatGPT are transforming the educational landscape, prompting reconsideration of traditional assessment practices. In parallel, universities are exploring alternatives to in-person, closed-book examinations, raising concerns about academic integrity and pedagogical alignment in uninvigilated settings. This study investigates whether traditional closed-book mathematics examinations retain their pedagogical relevance when hypothetically administered in uninvigilated, open-book settings with GenAI access. Adopting an empirical approach, we generate, transcribe, and blind-mark GenAI submissions to eight undergraduate mathematics examinations at a Russel Group university, spanning the entirety of the first-year curriculum. By combining independent GenAI responses to individual questions, we enable a meaningful evaluation of GenAI performance, both at the level of modules and across the first-year curriculum. We find that GenAI attainment is at the level of a first-class degree, though current performance can vary between modules. Further, we find that GenAI performance is remarkably consistent when viewed across the entire curriculum, significantly more so than that of students in invigilated examinations. Our findings evidence the need for redesigning assessments in mathematics for unsupervised settings, and highlight the potential reduction in pedagogical value of current standards in the era of generative artificial intelligence. 

---
# Synthetic Data and the Shifting Ground of Truth 

**Authors**: Dietmar Offenhuber  

**Link**: [PDF](https://arxiv.org/pdf/2509.13355)  

**Abstract**: The emergence of synthetic data for privacy protection, training data generation, or simply convenient access to quasi-realistic data in any shape or volume complicates the concept of ground truth. Synthetic data mimic real-world observations, but do not refer to external features. This lack of a representational relationship, however, not prevent researchers from using synthetic data as training data for AI models and ground truth repositories. It is claimed that the lack of data realism is not merely an acceptable tradeoff, but often leads to better model performance than realistic data: compensate for known biases, prevent overfitting and support generalization, and make the models more robust in dealing with unexpected outliers. Indeed, injecting noisy and outright implausible data into training sets can be beneficial for the model. This greatly complicates usual assumptions based on which representational accuracy determines data fidelity (garbage in - garbage out). Furthermore, ground truth becomes a self-referential affair, in which the labels used as a ground truth repository are themselves synthetic products of a generative model and as such not connected to real-world observations. My paper examines how ML researchers and practitioners bootstrap ground truth under such paradoxical circumstances without relying on the stable ground of representation and real-world reference. It will also reflect on the broader implications of a shift from a representational to what could be described as a mimetic or iconic concept of data. 

---
# Hybrid Quantum-Classical Model for Image Classification 

**Authors**: Muhammad Adnan Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2509.13353)  

**Abstract**: This study presents a systematic comparison between hybrid quantum-classical neural networks and purely classical models across three benchmark datasets (MNIST, CIFAR100, and STL10) to evaluate their performance, efficiency, and robustness. The hybrid models integrate parameterized quantum circuits with classical deep learning architectures, while the classical counterparts use conventional convolutional neural networks (CNNs). Experiments were conducted over 50 training epochs for each dataset, with evaluations on validation accuracy, test accuracy, training time, computational resource usage, and adversarial robustness (tested with $\epsilon=0.1$ perturbations).Key findings demonstrate that hybrid models consistently outperform classical models in final accuracy, achieving {99.38\% (MNIST), 41.69\% (CIFAR100), and 74.05\% (STL10) validation accuracy, compared to classical benchmarks of 98.21\%, 32.25\%, and 63.76\%, respectively. Notably, the hybrid advantage scales with dataset complexity, showing the most significant gains on CIFAR100 (+9.44\%) and STL10 (+10.29\%). Hybrid models also train 5--12$\times$ faster (e.g., 21.23s vs. 108.44s per epoch on MNIST) and use 6--32\% fewer parameters} while maintaining superior generalization to unseen test this http URL robustness tests reveal that hybrid models are significantly more resilient on simpler datasets (e.g., 45.27\% robust accuracy on MNIST vs. 10.80\% for classical) but show comparable fragility on complex datasets like CIFAR100 ($\sim$1\% robustness for both). Resource efficiency analyses indicate that hybrid models consume less memory (4--5GB vs. 5--6GB for classical) and lower CPU utilization (9.5\% vs. 23.2\% on average).These results suggest that hybrid quantum-classical architectures offer compelling advantages in accuracy, training efficiency, and parameter scalability, particularly for complex vision tasks. 

---
# Label-Efficient Grasp Joint Prediction with Point-JEPA 

**Authors**: Jed Guzelkabaagac, Boris Petrović  

**Link**: [PDF](https://arxiv.org/pdf/2509.13349)  

**Abstract**: We investigate whether 3D self-supervised pretraining with a Joint-Embedding Predictive Architecture (Point-JEPA) enables label-efficient grasp joint-angle prediction. Using point clouds tokenized from meshes and a ShapeNet-pretrained Point-JEPA encoder, we train a lightweight multi-hypothesis head with winner-takes-all and evaluate by top-logit selection. On DLR-Hand II with object-level splits, Point-JEPA reduces RMSE by up to 26% in low-label regimes and reaches parity with full supervision. These results suggest JEPA-style pretraining is a practical approach for data-efficient grasp learning. 

---
# Accuracy Paradox in Large Language Models: Regulating Hallucination Risks in Generative AI 

**Authors**: Zihao Li, Weiwei Yi, Jiahong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13345)  

**Abstract**: As Large Language Models (LLMs) permeate everyday decision-making, their epistemic and societal risks demand urgent scrutiny. Hallucinations, the generation of fabricated, misleading, oversimplified or untrustworthy outputs, has emerged as imperative challenges. While regulatory, academic, and technical discourse position accuracy as the principal benchmark for mitigating such harms, this article contends that overreliance on accuracy misdiagnoses the problem and has counterproductive effect: the accuracy paradox. Drawing on interdisciplinary literatures, this article develops a taxonomy of hallucination types and shows the paradox along three intertwining dimensions: outputs, individuals and society. First, accuracy functions as a superficial proxy for reliability, incentivising the optimisation of rhetorical fluency and surface-level correctness over epistemic trustworthiness. This encourages passive user trust in outputs that appear accurate but epistemically untenable. Second, accuracy as a singular metric fails to detect harms that are not factually false but are nonetheless misleading, value-laden, or socially distorting, including consensus illusions, sycophantic alignment, and subtle manipulation. Third, regulatory overemphasis on accuracy obscures the wider societal consequences of hallucination, including social sorting, privacy violations, equity harms, epistemic convergence that marginalises dissent, reduces pluralism, and causes social deskilling. By examining the EU AI Act, GDPR, and DSA, the article argues that current regulations are not yet structurally equipped to address these epistemic, relational, and systemic harms and exacerbated by the overreliance on accuracy. By exposing such conceptual and practical challenges, this article calls for a fundamental shift towards pluralistic, context-aware, and manipulation-resilient approaches to AI trustworthy governance. 

---
# Real World Robotic Exploration using Deep Neural Networks Trained in Photorealistic Reconstructed Environments 

**Authors**: Isaac Ronald Ward  

**Link**: [PDF](https://arxiv.org/pdf/2509.13342)  

**Abstract**: In this work, an existing deep neural network approach for determining a robot's pose from visual information (RGB images) is modified, improving its localization performance without impacting its ease of training. Explicitly, the network's loss function is extended in a manner which intuitively combines the positional and rotational error in order to increase robustness to perceptual aliasing. An improvement in the localization accuracy for indoor scenes is observed: with decreases of up to 9.64% and 2.99% in the median positional and rotational error respectively, when compared to the unmodified network.
Additionally, photogrammetry data is used to produce a pose-labelled dataset which allows the above model to be trained on a local environment, resulting in localization accuracies of 0.11m & 0.89 degrees. This trained model forms the basis of a navigation algorithm, which is tested in real-time on a TurtleBot (a wheeled robotic device). As such, this work introduces a full pipeline for creating a robust navigational algorithm for any given real world indoor scene; the only requirement being a collection of images from the scene, which can be captured in as little as 330 seconds of 

---
# Proximity-Based Evidence Retrieval for Uncertainty-Aware Neural Networks 

**Authors**: Hassan Gharoun, Mohammad Sadegh Khorshidi, Kasra Ranjbarigderi, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.13338)  

**Abstract**: This work proposes an evidence-retrieval mechanism for uncertainty-aware decision-making that replaces a single global cutoff with an evidence-conditioned, instance-adaptive criterion. For each test instance, proximal exemplars are retrieved in an embedding space; their predictive distributions are fused via Dempster-Shafer theory. The resulting fused belief acts as a per-instance thresholding mechanism. Because the supporting evidences are explicit, decisions are transparent and auditable. Experiments on CIFAR-10/100 with BiT and ViT backbones show higher or comparable uncertainty-aware performance with materially fewer confidently incorrect outcomes and a sustainable review load compared with applying threshold on prediction entropy. Notably, only a few evidences are sufficient to realize these gains; increasing the evidence set yields only modest changes. These results indicate that evidence-conditioned tagging provides a more reliable and interpretable alternative to fixed prediction entropy thresholds for operational uncertainty-aware decision-making. 

---
# Explainable AI-Enhanced Supervisory Control for High-Precision Spacecraft Formation 

**Authors**: Reza Pirayeshshirazinezhad  

**Link**: [PDF](https://arxiv.org/pdf/2509.13331)  

**Abstract**: We use artificial intelligence (AI) and supervisory adaptive control systems to plan and optimize the mission of precise spacecraft formation. Machine learning and robust control enhance the efficiency of spacecraft precision formation of the Virtual Telescope for X-ray Observation (VTXO) space mission. VTXO is a precise formation of two separate spacecraft making a virtual telescope with a one-kilometer focal length. One spacecraft carries the lens and the other spacecraft holds the camera to observe high-energy space objects in the X-ray domain with 55 milli-arcsecond angular resolution accuracy. Timed automata for supervisory control, Monte Carlo simulations for stability and robustness evaluation, and integration of deep neural networks for optimal estimation of mission parameters, satisfy the high precision mission criteria. We integrate deep neural networks with a constrained, non-convex dynamic optimization pipeline to predict optimal mission parameters, ensuring precision mission criteria are met. AI framework provides explainability by predicting the resulting energy consumption and mission error for a given set of mission parameters. It allows for transparent, justifiable, and real-time trade-offs, a capability not present in traditional adaptive controllers. The results show reductions in energy consumption and improved mission accuracy, demonstrating the capability of the system to address dynamic uncertainties and disturbances. 

---
# Dual Actor DDPG for Airborne STAR-RIS Assisted Communications 

**Authors**: Danish Rizvi, David Boyle  

**Link**: [PDF](https://arxiv.org/pdf/2509.13328)  

**Abstract**: This study departs from the prevailing assumption of independent Transmission and Reflection Coefficients (TRC) in Airborne Simultaneous Transmit and Reflect Reconfigurable Intelligent Surface (STAR-RIS) research. Instead, we explore a novel multi-user downlink communication system that leverages a UAV-mounted STAR-RIS (Aerial-STAR) incorporating a coupled TRC phase shift model. Our key contributions include the joint optimization of UAV trajectory, active beamforming vectors at the base station, and passive RIS TRCs to enhance communication efficiency, while considering UAV energy constraints. We design the TRC as a combination of discrete and continuous actions, and propose a novel Dual Actor Deep Deterministic Policy Gradient (DA-DDPG) algorithm. The algorithm relies on two separate actor networks for high-dimensional hybrid action space. We also propose a novel harmonic mean index (HFI)-based reward function to ensure communication fairness amongst users. For comprehensive analysis, we study the impact of RIS size on UAV aerodynamics showing that it increases drag and energy demand. Simulation results demonstrate that the proposed DA-DDPG algorithm outperforms conventional DDPG and DQN-based solutions by 24% and 97%, respectively, in accumulated reward. Three-dimensional UAV trajectory optimization achieves 28% higher communication efficiency compared to two-dimensional and altitude optimization. The HFI based reward function provides 41% lower QoS denial rates as compared to other benchmarks. The mobile Aerial-STAR system shows superior performance over fixed deployed counterparts, with the coupled phase STAR-RIS outperforming dual Transmit/Reflect RIS and conventional RIS setups. These findings highlight the potential of Aerial-STAR systems and the effectiveness of our proposed DA-DDPG approach in optimizing their performance. 

---
# Prognosis of COVID-19 using Artificial Intelligence: A Systematic Review and Meta-analysis 

**Authors**: SaeedReza Motamedian, Sadra Mohaghegh, Elham Babadi Oregani, Mahrsa Amjadi, Parnian Shobeiri, Negin Cheraghi, Niusha Solouki, Nikoo Ahmadi, Hossein Mohammad-Rahimi, Yassine Bouchareb, Arman Rahmim  

**Link**: [PDF](https://arxiv.org/pdf/2408.00208)  

**Abstract**: Purpose: Artificial intelligence (AI) techniques have been extensively utilized for diagnosing and prognosis of several diseases in recent years. This study identifies, appraises and synthesizes published studies on the use of AI for the prognosis of COVID-19. Method: Electronic search was performed using Medline, Google Scholar, Scopus, Embase, Cochrane and ProQuest. Studies that examined machine learning or deep learning methods to determine the prognosis of COVID-19 using CT or chest X-ray images were included. Polled sensitivity, specificity area under the curve and diagnostic odds ratio were calculated. Result: A total of 36 articles were included; various prognosis-related issues, including disease severity, mechanical ventilation or admission to the intensive care unit and mortality, were investigated. Several AI models and architectures were employed, such as the Siamense model, support vector machine, Random Forest , eXtreme Gradient Boosting, and convolutional neural networks. The models achieved 71%, 88% and 67% sensitivity for mortality, severity assessment and need for ventilation, respectively. The specificity of 69%, 89% and 89% were reported for the aforementioned variables. Conclusion: Based on the included articles, machine learning and deep learning methods used for the prognosis of COVID-19 patients using radiomic features from CT or CXR images can help clinicians manage patients and allocate resources more effectively. These studies also demonstrate that combining patient demographic, clinical data, laboratory tests and radiomic features improves model performances. 

---
# Joint data imputation and mechanistic modelling for simulating heart-brain interactions in incomplete datasets 

**Authors**: Jaume Banus, Maxime Sermesant, Oscar Camara, Marco Lorenzi  

**Link**: [PDF](https://arxiv.org/pdf/2010.01052)  

**Abstract**: The use of mechanistic models in clinical studies is limited by the lack of multi-modal patients data representing different anatomical and physiological processes. For example, neuroimaging datasets do not provide a sufficient representation of heart features for the modeling of cardiovascular factors in brain disorders. To tackle this problem we introduce a probabilistic framework for joint cardiac data imputation and personalisation of cardiovascular mechanistic models, with application to brain studies with incomplete heart data. Our approach is based on a variational framework for the joint inference of an imputation model of cardiac information from the available features, along with a Gaussian Process emulator that can faithfully reproduce personalised cardiovascular dynamics. Experimental results on UK Biobank show that our model allows accurate imputation of missing cardiac features in datasets containing minimal heart information, e.g. systolic and diastolic blood pressures only, while jointly estimating the emulated parameters of the lumped model. This allows a novel exploration of the heart-brain joint relationship through simulation of realistic cardiac dynamics corresponding to different conditions of brain anatomy. 

---
