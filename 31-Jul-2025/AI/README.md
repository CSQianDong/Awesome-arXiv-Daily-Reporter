# Automatically discovering heuristics in a complex SAT solver with large language models 

**Authors**: Yiwen Sun, Furong Ye, Zhihan Chen, Ke Wei, Shaowei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.22876)  

**Abstract**: Satisfiability problem (SAT) is a cornerstone of computational complexity with broad industrial applications, and it remains challenging to optimize modern SAT solvers in real-world settings due to their intricate architectures. While automatic configuration frameworks have been developed, they rely on manually constrained search spaces and yield limited performance gains. This work introduces a novel paradigm which effectively optimizes complex SAT solvers via Large Language Models (LLMs), and a tool called AutoModSAT is developed. Three fundamental challenges are addressed in order to achieve superior performance: (1) LLM-friendly solver: Systematic guidelines are proposed for developing a modularized solver to meet LLMs' compatibility, emphasizing code simplification, information share and bug reduction; (2) Automatic prompt optimization: An unsupervised automatic prompt optimization method is introduced to advance the diversity of LLMs' output; (3) Efficient search strategy: We design a presearch strategy and an EA evolutionary algorithm for the final efficient and effective discovery of heuristics. Extensive experiments across a wide range of datasets demonstrate that AutoModSAT achieves 50% performance improvement over the baseline solver and achieves 30% superiority against the state-of-the-art (SOTA) solvers. Moreover, AutoModSAT attains a 20% speedup on average compared to parameter-tuned alternatives of the SOTA solvers, showcasing the enhanced capability in handling complex problem instances. This work bridges the gap between AI-driven heuristics discovery and mission-critical system optimization, and provides both methodological advancements and empirically validated results for next-generation complex solver development. 

---
# The Incomplete Bridge: How AI Research (Mis)Engages with Psychology 

**Authors**: Han Jiang, Pengda Wang, Xiaoyuan Yi, Xing Xie, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22847)  

**Abstract**: Social sciences have accumulated a rich body of theories and methodologies for investigating the human mind and behaviors, while offering valuable insights into the design and understanding of Artificial Intelligence (AI) systems. Focusing on psychology as a prominent case, this study explores the interdisciplinary synergy between AI and the field by analyzing 1,006 LLM-related papers published in premier AI venues between 2023 and 2025, along with the 2,544 psychology publications they cite. Through our analysis, we identify key patterns of interdisciplinary integration, locate the psychology domains most frequently referenced, and highlight areas that remain underexplored. We further examine how psychology theories/frameworks are operationalized and interpreted, identify common types of misapplication, and offer guidance for more effective incorporation. Our work provides a comprehensive map of interdisciplinary engagement between AI and psychology, thereby facilitating deeper collaboration and advancing AI systems. 

---
# Enhancing Multi-Agent Collaboration with Attention-Based Actor-Critic Policies 

**Authors**: Hugo Garrido-Lestache, Jeremy Kedziora  

**Link**: [PDF](https://arxiv.org/pdf/2507.22782)  

**Abstract**: This paper introduces Team-Attention-Actor-Critic (TAAC), a reinforcement learning algorithm designed to enhance multi-agent collaboration in cooperative environments. TAAC employs a Centralized Training/Centralized Execution scheme incorporating multi-headed attention mechanisms in both the actor and critic. This design facilitates dynamic, inter-agent communication, allowing agents to explicitly query teammates, thereby efficiently managing the exponential growth of joint-action spaces while ensuring a high degree of collaboration. We further introduce a penalized loss function which promotes diverse yet complementary roles among agents. We evaluate TAAC in a simulated soccer environment against benchmark algorithms representing other multi-agent paradigms, including Proximal Policy Optimization and Multi-Agent Actor-Attention-Critic. We find that TAAC exhibits superior performance and enhanced collaborative behaviors across a variety of metrics (win rates, goal differentials, Elo ratings, inter-agent connectivity, balanced spatial distributions, and frequent tactical interactions such as ball possession swaps). 

---
# ASP-FZN: A Translation-based Constraint Answer Set Solver 

**Authors**: Thomas Eiter, Tobias Geibinger, Tobias Kaminski, Nysret Musliu, Johannes Oetsch  

**Link**: [PDF](https://arxiv.org/pdf/2507.22774)  

**Abstract**: We present the solver asp-fzn for Constraint Answer Set Programming (CASP), which extends ASP with linear constraints. Our approach is based on translating CASP programs into the solver-independent FlatZinc language that supports several Constraint Programming and Integer Programming backend solvers. Our solver supports a rich language of linear constraints, including some common global constraints. As for evaluation, we show that asp-fzn is competitive with state-of-the-art ASP solvers on benchmarks taken from past ASP competitions. Furthermore, we evaluate it on several CASP problems from the literature and compare its performance with clingcon, which is a prominent CASP solver that supports most of the asp-fzn language. The performance of asp-fzn is very promising as it is already competitive on plain ASP and even outperforms clingcon on some CASP benchmarks. 

---
# Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting 

**Authors**: Sebastian Monka, Irlan Grangel-González, Stefan Schmid, Lavdim Halilaj, Marc Rickart, Oliver Rudolph, Rui Dias  

**Link**: [PDF](https://arxiv.org/pdf/2507.22619)  

**Abstract**: Knowledge graphs (KGs) have transformed data management within the manufacturing industry, offering effective means for integrating disparate data sources through shared and structured conceptual schemas. However, harnessing the power of KGs can be daunting for non-experts, as it often requires formulating complex SPARQL queries to retrieve specific information. With the advent of Large Language Models (LLMs), there is a growing potential to automatically translate natural language queries into the SPARQL format, thus bridging the gap between user-friendly interfaces and the sophisticated architecture of KGs. The challenge remains in adequately informing LLMs about the relevant context and structure of domain-specific KGs, e.g., in manufacturing, to improve the accuracy of generated queries. In this paper, we evaluate multiple strategies that use LLMs as mediators to facilitate information retrieval from KGs. We focus on the manufacturing domain, particularly on the Bosch Line Information System KG and the I40 Core Information Model. In our evaluation, we compare various approaches for feeding relevant context from the KG to the LLM and analyze their proficiency in transforming real-world questions into SPARQL queries. Our findings show that LLMs can significantly improve their performance on generating correct and complete queries when provided only the adequate context of the KG schema. Such context-aware prompting techniques help LLMs to focus on the relevant parts of the ontology and reduce the risk of hallucination. We anticipate that the proposed techniques help LLMs to democratize access to complex data repositories and empower informed decision-making in manufacturing settings. 

---
# MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines 

**Authors**: Yaolun Zhang, Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22606)  

**Abstract**: Large Language Models (LLMs) have demonstrated the ability to solve a wide range of practical tasks within multi-agent systems. However, existing human-designed multi-agent frameworks are typically limited to a small set of pre-defined scenarios, while current automated design methods suffer from several limitations, such as the lack of tool integration, dependence on external training data, and rigid communication structures. In this paper, we propose MetaAgent, a finite state machine based framework that can automatically generate a multi-agent system. Given a task description, MetaAgent will design a multi-agent system and polish it through an optimization algorithm. When the multi-agent system is deployed, the finite state machine will control the agent's actions and the state transitions. To evaluate our framework, we conduct experiments on both text-based tasks and practical tasks. The results indicate that the generated multi-agent system surpasses other auto-designed methods and can achieve a comparable performance with the human-designed multi-agent system, which is optimized for those specific tasks. 

---
# Collaborative Medical Triage under Uncertainty: A Multi-Agent Dynamic Matching Approach 

**Authors**: Hongyan Cheng, Chengzhang Yu, Yanshu Shi, Chiyue Wang, Cong Liu, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22504)  

**Abstract**: The post-pandemic surge in healthcare demand, coupled with critical nursing shortages, has placed unprecedented pressure on emergency department triage systems, necessitating innovative AI-driven solutions. We present a multi-agent interactive intelligent system for medical triage that addresses three fundamental challenges in current AI-based triage systems: insufficient medical specialization leading to hallucination-induced misclassifications, heterogeneous department structures across healthcare institutions, and inefficient detail-oriented questioning that impedes rapid triage decisions. Our system employs three specialized agents - RecipientAgent, InquirerAgent, and DepartmentAgent - that collaborate through structured inquiry mechanisms and department-specific guidance rules to transform unstructured patient symptoms into accurate department recommendations. To ensure robust evaluation, we constructed a comprehensive Chinese medical triage dataset from a medical website, comprising 3,360 real-world cases spanning 9 primary departments and 62 secondary departments. Through systematic data imputation using large language models, we address the prevalent issue of incomplete medical records in real-world data. Experimental results demonstrate that our multi-agent system achieves 89.2% accuracy in primary department classification and 73.9% accuracy in secondary department classification after four rounds of patient interaction. The system's pattern-matching-based guidance mechanisms enable efficient adaptation to diverse hospital configurations while maintaining high triage accuracy. Our work provides a scalable framework for deploying AI-assisted triage systems that can accommodate the organizational heterogeneity of healthcare institutions while ensuring clinically sound decision-making. 

---
# Nearest-Better Network for Visualizing and Analyzing Combinatorial Optimization Problems: A Unified Tool 

**Authors**: Yiya Diao, Changhe Li, Sanyou Zeng, Xinye Cai, Wenjian Luo, Shengxiang Yang, Carlos A. Coello Coello  

**Link**: [PDF](https://arxiv.org/pdf/2507.22440)  

**Abstract**: The Nearest-Better Network (NBN) is a powerful method to visualize sampled data for continuous optimization problems while preserving multiple landscape features. However, the calculation of NBN is very time-consuming, and the extension of the method to combinatorial optimization problems is challenging but very important for analyzing the algorithm's behavior. This paper provides a straightforward theoretical derivation showing that the NBN network essentially functions as the maximum probability transition network for algorithms. This paper also presents an efficient NBN computation method with logarithmic linear time complexity to address the time-consuming issue. By applying this efficient NBN algorithm to the OneMax problem and the Traveling Salesman Problem (TSP), we have made several remarkable discoveries for the first time: The fitness landscape of OneMax exhibits neutrality, ruggedness, and modality features. The primary challenges of TSP problems are ruggedness, modality, and deception. Two state-of-the-art TSP algorithms (i.e., EAX and LKH) have limitations when addressing challenges related to modality and deception, respectively. LKH, based on local search operators, fails when there are deceptive solutions near global optima. EAX, which is based on a single population, can efficiently maintain diversity. However, when multiple attraction basins exist, EAX retains individuals within multiple basins simultaneously, reducing inter-basin interaction efficiency and leading to algorithm's stagnation. 

---
# Cross-Border Legal Adaptation of Autonomous Vehicle Design based on Logic and Non-monotonic Reasoning 

**Authors**: Zhe Yu, Yiwei Lu, Burkhard Schafer, Zhe Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22432)  

**Abstract**: This paper focuses on the legal compliance challenges of autonomous vehicles in a transnational context. We choose the perspective of designers and try to provide supporting legal reasoning in the design process. Based on argumentation theory, we introduce a logic to represent the basic properties of argument-based practical (normative) reasoning, combined with partial order sets of natural numbers to express priority. Finally, through case analysis of legal texts, we show how the reasoning system we provide can help designers to adapt their design solutions more flexibly in the cross-border application of autonomous vehicles and to more easily understand the legal implications of their decisions. 

---
# On the Definition of Intelligence 

**Authors**: Kei-Sing Ng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22423)  

**Abstract**: To engineer AGI, we should first capture the essence of intelligence in a species-agnostic form that can be evaluated, while being sufficiently general to encompass diverse paradigms of intelligent behavior, including reinforcement learning, generative models, classification, analogical reasoning, and goal-directed decision-making. We propose a general criterion based on sample fidelity: intelligence is the ability, given sample(s) from a category, to generate sample(s) from the same category. We formalise this intuition as {\epsilon}-category intelligence: it is {\epsilon}-intelligent with respect to a category if no chosen admissible distinguisher can separate generated from original samples beyond tolerance {\epsilon}. We present the formal framework, outline empirical protocols, and discuss implications for evaluation, safety, and generalization. 

---
# Beyond Accuracy: How AI Metacognitive Sensitivity improves AI-assisted Decision Making 

**Authors**: ZhaoBin Li, Mark Steyvers  

**Link**: [PDF](https://arxiv.org/pdf/2507.22365)  

**Abstract**: In settings where human decision-making relies on AI input, both the predictive accuracy of the AI system and the reliability of its confidence estimates influence decision quality. We highlight the role of AI metacognitive sensitivity -- its ability to assign confidence scores that accurately distinguish correct from incorrect predictions -- and introduce a theoretical framework for assessing the joint impact of AI's predictive accuracy and metacognitive sensitivity in hybrid decision-making settings. Our analysis identifies conditions under which an AI with lower predictive accuracy but higher metacognitive sensitivity can enhance the overall accuracy of human decision making. Finally, a behavioral experiment confirms that greater AI metacognitive sensitivity improves human decision performance. Together, these findings underscore the importance of evaluating AI assistance not only by accuracy but also by metacognitive sensitivity, and of optimizing both to achieve superior decision outcomes. 

---
# LLM-Crowdsourced: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models 

**Authors**: Qianhong Guo, Wei Xie, Xiaofang Cai, Enze Wang, Shuoyoucheng Ma, Kai Chen, Xiaofeng Wang, Baosheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22359)  

**Abstract**: Although large language models (LLMs) demonstrate remarkable capabilities across various tasks, evaluating their capabilities remains a challenging task. Existing evaluation methods suffer from issues such as data contamination, black-box operation, and subjective preference. These issues make it difficult to evaluate the LLMs' true capabilities comprehensively. To tackle these challenges, we propose a novel benchmark-free evaluation paradigm, LLM-Crowdsourced. It utilizes LLMs to generate questions, answer independently, and evaluate mutually. This method integrates four key evaluation criteria: dynamic, transparent, objective, and professional, which existing evaluation methods cannot satisfy simultaneously. Experiments on eight mainstream LLMs across mathematics and programming verify the advantages of our method in distinguishing LLM performance. Furthermore, our study reveals several novel findings that are difficult for traditional methods to detect, including but not limited to: (1) Gemini demonstrates the highest original and professional question-design capabilities among others; (2) Some LLMs exhibit ''memorization-based answering'' by misrecognizing questions as familiar ones with a similar structure; (3) LLM evaluation results demonstrate high consistency (robustness). 

---
# Magentic-UI: Towards Human-in-the-loop Agentic Systems 

**Authors**: Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, Eric Zhu, Griffin Bassman, Jacob Alber, Peter Chang, Ricky Loynd, Friederike Niedtner, Ece Kamar, Maya Murad, Rafah Hosn, Saleema Amershi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22358)  

**Abstract**: AI agents powered by large language models are increasingly capable of autonomously completing complex, multi-step tasks using external tools. Yet, they still fall short of human-level performance in most domains including computer use, software development, and research. Their growing autonomy and ability to interact with the outside world, also introduces safety and security risks including potentially misaligned actions and adversarial manipulation. We argue that human-in-the-loop agentic systems offer a promising path forward, combining human oversight and control with AI efficiency to unlock productivity from imperfect systems. We introduce Magentic-UI, an open-source web interface for developing and studying human-agent interaction. Built on a flexible multi-agent architecture, Magentic-UI supports web browsing, code execution, and file manipulation, and can be extended with diverse tools via Model Context Protocol (MCP). Moreover, Magentic-UI presents six interaction mechanisms for enabling effective, low-cost human involvement: co-planning, co-tasking, multi-tasking, action guards, and long-term memory. We evaluate Magentic-UI across four dimensions: autonomous task completion on agentic benchmarks, simulated user testing of its interaction capabilities, qualitative studies with real users, and targeted safety assessments. Our findings highlight Magentic-UI's potential to advance safe and efficient human-agent collaboration. 

---
# An Explainable Emotion Alignment Framework for LLM-Empowered Agent in Metaverse Service Ecosystem 

**Authors**: Qun Ma, Xiao Xue, Ming Zhang, Yifan Shen, Zihan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22326)  

**Abstract**: Metaverse service is a product of the convergence between Metaverse and service systems, designed to address service-related challenges concerning digital avatars, digital twins, and digital natives within Metaverse. With the rise of large language models (LLMs), agents now play a pivotal role in Metaverse service ecosystem, serving dual functions: as digital avatars representing users in the virtual realm and as service assistants (or NPCs) providing personalized support. However, during the modeling of Metaverse service ecosystems, existing LLM-based agents face significant challenges in bridging virtual-world services with real-world services, particularly regarding issues such as character data fusion, character knowledge association, and ethical safety concerns. This paper proposes an explainable emotion alignment framework for LLM-based agents in Metaverse Service Ecosystem. It aims to integrate factual factors into the decision-making loop of LLM-based agents, systematically demonstrating how to achieve more relational fact alignment for these agents. Finally, a simulation experiment in the Offline-to-Offline food delivery scenario is conducted to evaluate the effectiveness of this framework, obtaining more realistic social emergence. 

---
# CoEx -- Co-evolving World-model and Exploration 

**Authors**: Minsoo Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22281)  

**Abstract**: Planning in modern LLM agents relies on the utilization of LLM as an internal world model, acquired during pretraining. However, existing agent designs fail to effectively assimilate new observations into dynamic updates of the world model. This reliance on the LLM's static internal world model is progressively prone to misalignment with the underlying true state of the world, leading to the generation of divergent and erroneous plans. We introduce a hierarchical agent architecture, CoEx, in which hierarchical state abstraction allows LLM planning to co-evolve with a dynamically updated model of the world. CoEx plans and interacts with the world by using LLM reasoning to orchestrate dynamic plans consisting of subgoals, and its learning mechanism continuously incorporates these subgoal experiences into a persistent world model in the form of a neurosymbolic belief state, comprising textual inferences and code-based symbolic memory. We evaluate our agent across a diverse set of agent scenarios involving rich environments and complex tasks including ALFWorld, PDDL, and Jericho. Our experiments show that CoEx outperforms existing agent paradigms in planning and exploration. 

---
# Explainability Through Systematicity: The Hard Systematicity Challenge for Artificial Intelligence 

**Authors**: Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.22197)  

**Abstract**: This paper argues that explainability is only one facet of a broader ideal that shapes our expectations towards artificial intelligence (AI). Fundamentally, the issue is to what extent AI exhibits systematicity--not merely in being sensitive to how thoughts are composed of recombinable constituents, but in striving towards an integrated body of thought that is consistent, coherent, comprehensive, and parsimoniously principled. This richer conception of systematicity has been obscured by the long shadow of the "systematicity challenge" to connectionism, according to which network architectures are fundamentally at odds with what Fodor and colleagues termed "the systematicity of thought." I offer a conceptual framework for thinking about "the systematicity of thought" that distinguishes four senses of the phrase. I use these distinctions to defuse the perceived tension between systematicity and connectionism and show that the conception of systematicity that historically shaped our sense of what makes thought rational, authoritative, and scientific is more demanding than the Fodorian notion. To determine whether we have reason to hold AI models to this ideal of systematicity, I then argue, we must look to the rationales for systematization and explore to what extent they transfer to AI models. I identify five such rationales and apply them to AI. This brings into view the "hard systematicity challenge." However, the demand for systematization itself needs to be regulated by the rationales for systematization. This yields a dynamic understanding of the need to systematize thought, which tells us how systematic we need AI models to be and when. 

---
# When Truthful Representations Flip Under Deceptive Instructions? 

**Authors**: Xianxuan Long, Yao Fu, Runchao Li, Mu Sheng, Haotian Yu, Xiaotian Han, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22149)  

**Abstract**: Large language models (LLMs) tend to follow maliciously crafted instructions to generate deceptive responses, posing safety challenges. How deceptive instructions alter the internal representations of LLM compared to truthful ones remains poorly understood beyond output analysis. To bridge this gap, we investigate when and how these representations ``flip'', such as from truthful to deceptive, under deceptive versus truthful/neutral instructions. Analyzing the internal representations of Llama-3.1-8B-Instruct and Gemma-2-9B-Instruct on a factual verification task, we find the model's instructed True/False output is predictable via linear probes across all conditions based on the internal representation. Further, we use Sparse Autoencoders (SAEs) to show that the Deceptive instructions induce significant representational shifts compared to Truthful/Neutral representations (which are similar), concentrated in early-to-mid layers and detectable even on complex datasets. We also identify specific SAE features highly sensitive to deceptive instruction and use targeted visualizations to confirm distinct truthful/deceptive representational subspaces. % Our analysis pinpoints layer-wise and feature-level correlates of instructed dishonesty, offering insights for LLM detection and control. Our findings expose feature- and layer-level signatures of deception, offering new insights for detecting and mitigating instructed dishonesty in LLMs. 

---
# Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning 

**Authors**: Kwesi Cobbina, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22887)  

**Abstract**: In-context learning (ICL) is a critical emerging capability of large language models (LLMs), enabling few-shot learning during inference by including a few demonstrations (demos) in the prompt. However, it has been found that ICL's performance can be sensitive to the choices of demos and their order. This paper investigates an unexplored new positional bias of ICL for the first time: we observe that the predictions and accuracy can drift drastically when the positions of demos, the system prompt, and the user message in LLM input are varied. We refer to this bias as DEMOS' POSITION IN PROMPT (DPP) bias. We design a systematic evaluation pipeline to study this type of positional bias across classification, question answering, summarization, and reasoning tasks. We introduce two metrics, ACCURACY-CHANGE and PREDICTION-CHANGE, to quantify net gains and output volatility induced by changes in the demos' position. Extensive experiments on ten LLMs from four open-source model families (QWEN, LLAMA3, MISTRAL, COHERE) verify that the bias significantly affects their accuracy and predictions: placing demos at the start of the prompt yields the most stable and accurate outputs with gains of up to +6 points. In contrast, placing demos at the end of the user message flips over 30\% of predictions without improving correctness on QA tasks. Smaller models are most affected by this sensitivity, though even large models remain marginally affected on complex tasks. 

---
# A Bit of Freedom Goes a Long Way: Classical and Quantum Algorithms for Reinforcement Learning under a Generative Model 

**Authors**: Andris Ambainis, Joao F. Doriguello, Debbie Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22854)  

**Abstract**: We propose novel classical and quantum online algorithms for learning finite-horizon and infinite-horizon average-reward Markov Decision Processes (MDPs). Our algorithms are based on a hybrid exploration-generative reinforcement learning (RL) model wherein the agent can, from time to time, freely interact with the environment in a generative sampling fashion, i.e., by having access to a "simulator". By employing known classical and new quantum algorithms for approximating optimal policies under a generative model within our learning algorithms, we show that it is possible to avoid several paradigms from RL like "optimism in the face of uncertainty" and "posterior sampling" and instead compute and use optimal policies directly, which yields better regret bounds compared to previous works. For finite-horizon MDPs, our quantum algorithms obtain regret bounds which only depend logarithmically on the number of time steps $T$, thus breaking the $O(\sqrt{T})$ classical barrier. This matches the time dependence of the prior quantum works of Ganguly et al. (arXiv'23) and Zhong et al. (ICML'24), but with improved dependence on other parameters like state space size $S$ and action space size $A$. For infinite-horizon MDPs, our classical and quantum bounds still maintain the $O(\sqrt{T})$ dependence but with better $S$ and $A$ factors. Nonetheless, we propose a novel measure of regret for infinite-horizon MDPs with respect to which our quantum algorithms have $\operatorname{poly}\log{T}$ regret, exponentially better compared to classical algorithms. Finally, we generalise all of our results to compact state spaces. 

---
# Repair-R1: Better Test Before Repair 

**Authors**: Haichuan Hu, Xiaochen Xie, Quanjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22853)  

**Abstract**: APR (Automated Program Repair) aims to automatically locate program defects, generate patches and validate the repairs. Existing techniques for APR are often combined with LLMs (Large Language Models), which leverages the code-related knowledge of LLMs to improve repair effectiveness. Current LLM-based APR methods typically utilize test cases only during the inference stage, adopting an iterative approach that performs repair first and validates it through test execution afterward. This conventional paradigm neglects two important aspects: the potential contribution of test cases in the training phase, and the possibility of leveraging testing prior to repair. To address this, we propose Repair-R1, which introduces test cases into the model's training phase and shifts test generation to precede repair. The model is required to first generate discriminative test cases that can distinguish defective behaviors, and then perform repair based on these tests. This enables the model to better locate defects and understand the underlying causes of defects, thereby improving repair effectiveness. We implement Repair-R1 with three different backbone models, using RL (reinforcement learning) to co-optimize test generation and bug repair. Experimental results on four widely adopted benchmarks demonstrate the superiority of Repair-R1. Specially, compared to vanilla models, Repair-R1 improves repair success rate by 2.68\% to 48.29\%, test generation success rate by 16.38\% to 53.28\%, and test coverage by 0.78\% to 53.96\%. We publish the code and weights at this https URL and this https URL. 

---
# RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents 

**Authors**: Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22844)  

**Abstract**: The development of autonomous agents for complex, long-horizon tasks is a central goal in AI. However, dominant training paradigms face a critical limitation: reinforcement learning (RL) methods that optimize solely for final task success often reinforce flawed or inefficient reasoning paths, a problem we term inefficient exploration. This leads to agents that are brittle and fail to generalize, as they learn to find solutions without learning how to reason coherently. To address this, we introduce RLVMR, a novel framework that integrates dense, process-level supervision into end-to-end RL by rewarding verifiable, meta-reasoning behaviors. RLVMR equips an agent to explicitly tag its cognitive steps, such as planning, exploration, and reflection, and provides programmatic, rule-based rewards for actions that contribute to effective problem-solving. These process-centric rewards are combined with the final outcome signal and optimized using a critic-free policy gradient method. On the challenging ALFWorld and ScienceWorld benchmarks, RLVMR achieves new state-of-the-art results, with our 7B model reaching an 83.6% success rate on the most difficult unseen task split. Our analysis confirms these gains stem from improved reasoning quality, including significant reductions in redundant actions and enhanced error recovery, leading to more robust, efficient, and interpretable agents. 

---
# CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models 

**Authors**: Kedong Xiu, Saiqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22828)  

**Abstract**: As Vision-Language Models (VLMs) are increasingly deployed in split-DNN configurations--with visual encoders (e.g., ResNet, ViT) operating on user devices and sending intermediate features to the cloud--there is a growing privacy risk from semantic information leakage. Existing approaches to reconstructing images from these intermediate features often result in blurry, semantically ambiguous images. To directly address semantic leakage, we propose CapRecover, a cross-modality inversion framework that recovers high-level semantic content, such as labels or captions, directly from intermediate features without image reconstruction.
We evaluate CapRecover on multiple datasets and victim models, demonstrating strong performance in semantic recovery. Specifically, CapRecover achieves up to 92.71% Top-1 label accuracy on CIFAR-10 and generates fluent captions from ResNet50 features on COCO2017 with ROUGE-L scores up to 0.52. Our analysis further reveals that deeper convolutional layers encode significantly more semantic information compared to shallow layers. To mitigate semantic leakage, we introduce a simple yet effective protection method: adding random noise to intermediate features at each layer and removing the noise in the next layer. Experimental results show that this approach prevents semantic leakage without additional training costs. 

---
# MoCHA: Advanced Vision-Language Reasoning with MoE Connector and Hierarchical Group Attention 

**Authors**: Yuqi Pang, Bowen Yang, Yun Cao, Fan Rong, Xiaoyu Li, Chen He  

**Link**: [PDF](https://arxiv.org/pdf/2507.22805)  

**Abstract**: Vision large language models (VLLMs) are focusing primarily on handling complex and fine-grained visual information by incorporating advanced vision encoders and scaling up visual models. However, these approaches face high training and inference costs, as well as challenges in extracting visual details, effectively bridging across modalities. In this work, we propose a novel visual framework, MoCHA, to address these issues. Our framework integrates four vision backbones (i.e., CLIP, SigLIP, DINOv2 and ConvNeXt) to extract complementary visual features and is equipped with a sparse Mixture of Experts Connectors (MoECs) module to dynamically select experts tailored to different visual dimensions. To mitigate redundant or insufficient use of the visual information encoded by the MoECs module, we further design a Hierarchical Group Attention (HGA) with intra- and inter-group operations and an adaptive gating strategy for encoded visual features. We train MoCHA on two mainstream LLMs (e.g., Phi2-2.7B and Vicuna-7B) and evaluate their performance across various benchmarks. Notably, MoCHA outperforms state-of-the-art open-weight models on various tasks. For example, compared to CuMo (Mistral-7B), our MoCHA (Phi2-2.7B) presents outstanding abilities to mitigate hallucination by showing improvements of 3.25% in POPE and to follow visual instructions by raising 153 points on MME. Finally, ablation studies further confirm the effectiveness and robustness of the proposed MoECs and HGA in improving the overall performance of MoCHA. 

---
# Advancing Fetal Ultrasound Image Quality Assessment in Low-Resource Settings 

**Authors**: Dongli He, Hu Wang, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2507.22802)  

**Abstract**: Accurate fetal biometric measurements, such as abdominal circumference, play a vital role in prenatal care. However, obtaining high-quality ultrasound images for these measurements heavily depends on the expertise of sonographers, posing a significant challenge in low-income countries due to the scarcity of trained personnel. To address this issue, we leverage FetalCLIP, a vision-language model pretrained on a curated dataset of over 210,000 fetal ultrasound image-caption pairs, to perform automated fetal ultrasound image quality assessment (IQA) on blind-sweep ultrasound data. We introduce FetalCLIP$_{CLS}$, an IQA model adapted from FetalCLIP using Low-Rank Adaptation (LoRA), and evaluate it on the ACOUSLIC-AI dataset against six CNN and Transformer baselines. FetalCLIP$_{CLS}$ achieves the highest F1 score of 0.757. Moreover, we show that an adapted segmentation model, when repurposed for classification, further improves performance, achieving an F1 score of 0.771. Our work demonstrates how parameter-efficient fine-tuning of fetal ultrasound foundation models can enable task-specific adaptations, advancing prenatal care in resource-limited settings. The experimental code is available at: this https URL. 

---
# G-Core: A Simple, Scalable and Balanced RLHF Trainer 

**Authors**: Junyu Wu, Weiming Chang, Xiaotao Liu, Guanyou He, Haoqiang Hong, Boqi Liu, Hongtao Tian, Tao Yang, Yunsheng Shi, Feng Lin, Ting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22789)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become an increasingly popular paradigm for training large language models (LLMs) and diffusion models. While existing RLHF training systems have enabled significant progress, they often face challenges in scaling to multi-modal and diffusion workflows and adapting to dynamic workloads. In particular, current approaches may encounter limitations in controller scalability, flexible resource placement, and efficient orchestration when handling complex RLHF pipelines, especially in scenarios involving dynamic sampling or generative reward modeling. In this paper, we present \textbf{G-Core}, a simple, scalable, and balanced RLHF training framework designed to address these challenges. G-Core introduces a parallel controller programming model, enabling flexible and efficient orchestration of complex RLHF workflows without the bottlenecks of a single centralized controller. Furthermore, we propose a dynamic placement schema that adaptively partitions resources and schedules workloads, significantly reducing hardware idle time and improving utilization, even under highly variable training conditions. G-Core has successfully trained models that support WeChat product features serving a large-scale user base, demonstrating its effectiveness and robustness in real-world scenarios. Our results show that G-Core advances the state of the art in RLHF training, providing a solid foundation for future research and deployment of large-scale, human-aligned models. 

---
# Empirical Evaluation of Concept Drift in ML-Based Android Malware Detection 

**Authors**: Ahmed Sabbah, Radi Jarrar, Samer Zein, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22772)  

**Abstract**: Despite outstanding results, machine learning-based Android malware detection models struggle with concept drift, where rapidly evolving malware characteristics degrade model effectiveness. This study examines the impact of concept drift on Android malware detection, evaluating two datasets and nine machine learning and deep learning algorithms, as well as Large Language Models (LLMs). Various feature types--static, dynamic, hybrid, semantic, and image-based--were considered. The results showed that concept drift is widespread and significantly affects model performance. Factors influencing the drift include feature types, data environments, and detection methods. Balancing algorithms helped with class imbalance but did not fully address concept drift, which primarily stems from the dynamic nature of the malware landscape. No strong link was found between the type of algorithm used and concept drift, the impact was relatively minor compared to other variables since hyperparameters were not fine-tuned, and the default algorithm configurations were used. While LLMs using few-shot learning demonstrated promising detection performance, they did not fully mitigate concept drift, highlighting the need for further investigation. 

---
# Teaching the Teacher: Improving Neural Network Distillability for Symbolic Regression via Jacobian Regularization 

**Authors**: Soumyadeep Dhar, Kei Sen Fong, Mehul Motani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22767)  

**Abstract**: Distilling large neural networks into simple, human-readable symbolic formulas is a promising path toward trustworthy and interpretable AI. However, this process is often brittle, as the complex functions learned by standard networks are poor targets for symbolic discovery, resulting in low-fidelity student models. In this work, we propose a novel training paradigm to address this challenge. Instead of passively distilling a pre-trained network, we introduce a \textbf{Jacobian-based regularizer} that actively encourages the ``teacher'' network to learn functions that are not only accurate but also inherently smoother and more amenable to distillation. We demonstrate through extensive experiments on a suite of real-world regression benchmarks that our method is highly effective. By optimizing the regularization strength for each problem, we improve the $R^2$ score of the final distilled symbolic model by an average of \textbf{120\% (relative)} compared to the standard distillation pipeline, all while maintaining the teacher's predictive accuracy. Our work presents a practical and principled method for significantly improving the fidelity of interpretable models extracted from complex neural networks. 

---
# Bayesian Optimization of Process Parameters of a Sensor-Based Sorting System using Gaussian Processes as Surrogate Models 

**Authors**: Felix Kronenwett, Georg Maier, Thomas Laengle  

**Link**: [PDF](https://arxiv.org/pdf/2507.22766)  

**Abstract**: Sensor-based sorting systems enable the physical separation of a material stream into two fractions. The sorting decision is based on the image data evaluation of the sensors used and is carried out using actuators. Various process parameters must be set depending on the properties of the material stream, the dimensioning of the system, and the required sorting accuracy. However, continuous verification and re-adjustment are necessary due to changing requirements and material stream compositions. In this paper, we introduce an approach for optimizing, recurrently monitoring and adjusting the process parameters of a sensor-based sorting system. Based on Bayesian Optimization, Gaussian process regression models are used as surrogate models to achieve specific requirements for system behavior with the uncertainties contained therein. This method minimizes the number of necessary experiments while simultaneously considering two possible optimization targets based on the requirements for both material output streams. In addition, uncertainties are considered during determining sorting accuracies in the model calculation. We evaluated the method with three example process parameters. 

---
# Of Good Demons and Bad Angels: Guaranteeing Safe Control under Finite Precision 

**Authors**: Samuel Teuber, Debasmita Lohar, Bernhard Beckert  

**Link**: [PDF](https://arxiv.org/pdf/2507.22760)  

**Abstract**: As neural networks (NNs) become increasingly prevalent in safety-critical neural network-controlled cyber-physical systems (NNCSs), formally guaranteeing their safety becomes crucial. For these systems, safety must be ensured throughout their entire operation, necessitating infinite-time horizon verification. To verify the infinite-time horizon safety of NNCSs, recent approaches leverage Differential Dynamic Logic (dL). However, these dL-based guarantees rely on idealized, real-valued NN semantics and fail to account for roundoff errors introduced by finite-precision implementations. This paper bridges the gap between theoretical guarantees and real-world implementations by incorporating robustness under finite-precision perturbations -- in sensing, actuation, and computation -- into the safety verification. We model the problem as a hybrid game between a good Demon, responsible for control actions, and a bad Angel, introducing perturbations. This formulation enables formal proofs of robustness w.r.t. a given (bounded) perturbation. Leveraging this bound, we employ state-of-the-art mixed-precision fixed-point tuners to synthesize sound and efficient implementations, thus providing a complete end-to-end solution. We evaluate our approach on case studies from the automotive and aeronautics domains, producing efficient NN implementations with rigorous infinite-time horizon safety guarantees. 

---
# Reducing Hallucinations in Summarization via Reinforcement Learning with Entity Hallucination Index 

**Authors**: Praveenkumar Katwe, Rakesh Chandra, Balabantaray Kali, Prasad Vittala  

**Link**: [PDF](https://arxiv.org/pdf/2507.22744)  

**Abstract**: Reducing hallucinations in abstractive summarization remains a critical challenge for deploying language models (LMs) in real-world settings. In this work, we introduce a rewarddriven fine-tuning framework that explicitly optimizes for Entity Hallucination Index (EHI), a metric designed to quantify the presence, correctness, and grounding of named entities in generated summaries. Given a corpus of meeting transcripts, we first generate baseline summaries using a pre-trained LM and compute EHI scores via automatic entity extraction and matching. We then apply reinforcement learning to fine-tune the model parameters, using EHI as a reward signal to bias generation toward entity-faithful outputs. Our approach does not rely on human-written factuality annotations, enabling scalable fine-tuning. Experiments demonstrate consistent improvements in EHI across datasets, with qualitative analysis revealing a significant reduction in entity-level hallucinations without degradation in fluency or informativeness. We release a reproducible Colab pipeline, facilitating further research on hallucination-aware model fine-tuning using lightweight, hallucintion metrics like EHI. 

---
# OFCnetLLM: Large Language Model for Network Monitoring and Alertness 

**Authors**: Hong-Jun Yoon, Mariam Kiran, Danial Ebling, Joe Breen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22711)  

**Abstract**: The rapid evolution of network infrastructure is bringing new challenges and opportunities for efficient network management, optimization, and security. With very large monitoring databases becoming expensive to explore, the use of AI and Generative AI can help reduce costs of managing these datasets. This paper explores the use of Large Language Models (LLMs) to revolutionize network monitoring management by addressing the limitations of query finding and pattern analysis. We leverage LLMs to enhance anomaly detection, automate root-cause analysis, and automate incident analysis to build a well-monitored network management team using AI. Through a real-world example of developing our own OFCNetLLM, based on the open-source LLM model, we demonstrate practical applications of OFCnetLLM in the OFC conference network. Our model is developed as a multi-agent approach and is still evolving, and we present early results here. 

---
# Bifröst: Spatial Networking with Bigraphs 

**Authors**: Josh Millar, Ryan Gibb, Roy Ang, Anil Madhavapeddy, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22687)  

**Abstract**: Modern networked environments increasingly rely on spatial reasoning, but lack a coherent representation for coordinating physical space. Consequently, tasks such as enforcing spatial access policies remain fragile and manual. We first propose a unifying representation based on bigraphs, capturing spatial, social, and communication relationships within a single formalism, with user-facing tools to generate bigraphs from physical environments. Second, we present a hierarchical agent architecture for distributed spatial reasoning, with runtimes for agentic processes to interact the spatial representation, and a context-aware execution model that scopes reasoning to the smallest viable subspace. Together, these enable private, reliable, and low-latency spatial networking that can safely interact with agentic workflows. 

---
# Hydra-Bench: A Benchmark for Multi-Modal Leaf Wetness Sensing 

**Authors**: Yimeng Liu, Maolin Gan, Yidong Ren, Gen Li, Jingkai Lin, Younsuk Dong, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22685)  

**Abstract**: Leaf wetness detection is a crucial task in agricultural monitoring, as it directly impacts the prediction and protection of plant diseases. However, existing sensing systems suffer from limitations in robustness, accuracy, and environmental resilience when applied to natural leaves under dynamic real-world conditions. To address these challenges, we introduce a new multi-modal dataset specifically designed for evaluating and advancing machine learning algorithms in leaf wetness detection. Our dataset comprises synchronized mmWave raw data, Synthetic Aperture Radar (SAR) images, and RGB images collected over six months from five diverse plant species in both controlled and outdoor field environments. We provide detailed benchmarks using the Hydra model, including comparisons against single modality baselines and multiple fusion strategies, as well as performance under varying scan distances. Additionally, our dataset can serve as a benchmark for future SAR imaging algorithm optimization, enabling a systematic evaluation of detection accuracy under diverse conditions. 

---
# Designing for Self-Regulation in Informal Programming Learning: Insights from a Storytelling-Centric Approach 

**Authors**: Sami Saeed Alghamdi, Christopher Bull, Ahmed Kharrufa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22671)  

**Abstract**: Many people learn programming independently from online resources and often report struggles in achieving their personal learning goals. Learners frequently describe their experiences as isolating and frustrating, challenged by abundant uncertainties, information overload, and distraction, compounded by limited guidance. At the same time, social media serves as a personal space where many engage in diverse self-regulation practices, including help-seeking, using external memory aids (e.g., self-notes), self-reflection, emotion regulation, and self-motivation. For instance, learners often mark achievements and set milestones through their posts. In response, we developed a system consisting of a web platform and browser extensions to support self-regulation online. The design aims to add learner-defined structure to otherwise unstructured experiences and bring meaning to curation and reflection activities by translating them into learning stories with AI-generated feedback. We position storytelling as an integrative approach to design that connects resource curation, reflective and sensemaking practice, and narrative practices learners already use across social platforms. We recruited 15 informal programming learners who are regular social media users to engage with the system in a self-paced manner; participation concluded upon submitting a learning story and survey. We used three quantitative scales and a qualitative survey to examine users' characteristics and perceptions of the system's support for their self-regulation. User feedback suggests the system's viability as a self-regulation aid. Learners particularly valued in-situ reflection, automated story feedback, and video annotation, while other features received mixed views. We highlight perceived benefits, friction points, and design opportunities for future AI-augmented self-regulation tools. 

---
# RobEthiChor: Automated Context-aware Ethics-based Negotiation for Autonomous Robots 

**Authors**: Mashal Afzal Memon, Gianluca Filippone, Gian Luca Scoccia, Marco Autili, Paola Inverardi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22664)  

**Abstract**: The presence of autonomous systems is growing at a fast pace and it is impacting many aspects of our lives. Designed to learn and act independently, these systems operate and perform decision-making without human intervention. However, they lack the ability to incorporate users' ethical preferences, which are unique for each individual in society and are required to personalize the decision-making processes. This reduces user trust and prevents autonomous systems from behaving according to the moral beliefs of their end-users. When multiple systems interact with differing ethical preferences, they must negotiate to reach an agreement that satisfies the ethical beliefs of all the parties involved and adjust their behavior consequently. To address this challenge, this paper proposes RobEthiChor, an approach that enables autonomous systems to incorporate user ethical preferences and contextual factors into their decision-making through ethics-based negotiation. RobEthiChor features a domain-agnostic reference architecture for designing autonomous systems capable of ethic-based negotiating. The paper also presents RobEthiChor-Ros, an implementation of RobEthiChor within the Robot Operating System (ROS), which can be deployed on robots to provide them with ethics-based negotiation capabilities. To evaluate our approach, we deployed RobEthiChor-Ros on real robots and ran scenarios where a pair of robots negotiate upon resource contention. Experimental results demonstrate the feasibility and effectiveness of the system in realizing ethics-based negotiation. RobEthiChor allowed robots to reach an agreement in more than 73\% of the scenarios with an acceptable negotiation time (0.67s on average). Experiments also demonstrate that the negotiation approach implemented in RobEthiChor is scalable. 

---
# A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models 

**Authors**: Sabrina Kaniewski, Fabian Schmidt, Markus Enzweiler, Michael Menth, Tobias Heer  

**Link**: [PDF](https://arxiv.org/pdf/2507.22659)  

**Abstract**: The increasing adoption of Large Language Models (LLMs) in software engineering has sparked interest in their use for software vulnerability detection. However, the rapid development of this field has resulted in a fragmented research landscape, with diverse studies that are difficult to compare due to differences in, e.g., system designs and dataset usage. This fragmentation makes it difficult to obtain a clear overview of the state-of-the-art or compare and categorize studies meaningfully. In this work, we present a comprehensive systematic literature review (SLR) of LLM-based software vulnerability detection. We analyze 227 studies published between January 2020 and June 2025, categorizing them by task formulation, input representation, system architecture, and adaptation techniques. Further, we analyze the datasets used, including their characteristics, vulnerability coverage, and diversity. We present a fine-grained taxonomy of vulnerability detection approaches, identify key limitations, and outline actionable future research opportunities. By providing a structured overview of the field, this review improves transparency and serves as a practical guide for researchers and practitioners aiming to conduct more comparable and reproducible research. We publicly release all artifacts and maintain a living repository of LLM-based software vulnerability detection studies. 

---
# Safe Deployment of Offline Reinforcement Learning via Input Convex Action Correction 

**Authors**: Alex Durkin, Jasper Stolte, Matthew Jones, Raghuraman Pitchumani, Bei Li, Christian Michler, Mehmet Mercangöz  

**Link**: [PDF](https://arxiv.org/pdf/2507.22640)  

**Abstract**: Offline reinforcement learning (offline RL) offers a promising framework for developing control strategies in chemical process systems using historical data, without the risks or costs of online experimentation. This work investigates the application of offline RL to the safe and efficient control of an exothermic polymerisation continuous stirred-tank reactor. We introduce a Gymnasium-compatible simulation environment that captures the reactor's nonlinear dynamics, including reaction kinetics, energy balances, and operational constraints. The environment supports three industrially relevant scenarios: startup, grade change down, and grade change up. It also includes reproducible offline datasets generated from proportional-integral controllers with randomised tunings, providing a benchmark for evaluating offline RL algorithms in realistic process control tasks.
We assess behaviour cloning and implicit Q-learning as baseline algorithms, highlighting the challenges offline agents face, including steady-state offsets and degraded performance near setpoints. To address these issues, we propose a novel deployment-time safety layer that performs gradient-based action correction using input convex neural networks (PICNNs) as learned cost models. The PICNN enables real-time, differentiable correction of policy actions by descending a convex, state-conditioned cost surface, without requiring retraining or environment interaction.
Experimental results show that offline RL, particularly when combined with convex action correction, can outperform traditional control approaches and maintain stability across all scenarios. These findings demonstrate the feasibility of integrating offline RL with interpretable and safety-aware corrections for high-stakes chemical process control, and lay the groundwork for more reliable data-driven automation in industrial systems. 

---
# H2Tune: Federated Foundation Model Fine-Tuning with Hybrid Heterogeneity 

**Authors**: Wei Guo, Siyuan Lu, Yiqi Tong, Zhaojun Hu, Fuzhen Zhuang, Xiao Zhang, Tao Fan, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22633)  

**Abstract**: Different from existing federated fine-tuning (FFT) methods for foundation models, hybrid heterogeneous federated fine-tuning (HHFFT) is an under-explored scenario where clients exhibit double heterogeneity in model architectures and downstream tasks. This hybrid heterogeneity introduces two significant challenges: 1) heterogeneous matrix aggregation, where clients adopt different large-scale foundation models based on their task requirements and resource limitations, leading to dimensional mismatches during LoRA parameter aggregation; and 2) multi-task knowledge interference, where local shared parameters, trained with both task-shared and task-specific knowledge, cannot ensure only task-shared knowledge is transferred between clients. To address these challenges, we propose H2Tune, a federated foundation model fine-tuning with hybrid heterogeneity. Our framework H2Tune consists of three key components: (i) sparsified triple matrix decomposition to align hidden dimensions across clients through constructing rank-consistent middle matrices, with adaptive sparsification based on client resources; (ii) relation-guided matrix layer alignment to handle heterogeneous layer structures and representation capabilities; and (iii) alternating task-knowledge disentanglement mechanism to decouple shared and specific knowledge of local model parameters through alternating optimization. Theoretical analysis proves a convergence rate of O(1/\sqrt{T}). Extensive experiments show our method achieves up to 15.4% accuracy improvement compared to state-of-the-art baselines. Our code is available at this https URL. 

---
# LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing 

**Authors**: Federico Girella, Davide Talon, Ziyue Liu, Zanxi Ruan, Yiming Wang, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22627)  

**Abstract**: Fashion design is a complex creative process that blends visual and textual expressions. Designers convey ideas through sketches, which define spatial structure and design elements, and textual descriptions, capturing material, texture, and stylistic details. In this paper, we present LOcalized Text and Sketch for fashion image generation (LOTS), an approach for compositional sketch-text based generation of complete fashion outlooks. LOTS leverages a global description with paired localized sketch + text information for conditioning and introduces a novel step-based merging strategy for diffusion adaptation. First, a Modularized Pair-Centric representation encodes sketches and text into a shared latent space while preserving independent localized features; then, a Diffusion Pair Guidance phase integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we build on Fashionpedia to release Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Quantitative results show LOTS achieves state-of-the-art image generation performance on both global and localized metrics, while qualitative examples and a human evaluation study highlight its unprecedented level of design customization. 

---
# Adaptive Duration Model for Text Speech Alignment 

**Authors**: Junjie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22612)  

**Abstract**: Speech-to-text alignment is a critical component of neural text to-speech (TTS) models. Autoregressive TTS models typically use an attention mechanism to learn these alignments on-line. However, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, leading to missing or repeating words. Most non-autoregressive end to-end TTS models rely on durations extracted from external sources, using additional duration models for alignment. In this paper, we propose a novel duration prediction framework that can give compromising phoneme-level duration distribution with given text. In our experiments, the proposed duration model has more precise prediction and condition adaptation ability compared to previous baseline models. Numerically, it has roughly a 11.3 percents immprovement on alignment accuracy, and makes the performance of zero-shot TTS models more robust to the mismatch between prompt audio and input audio. 

---
# Metamorphic Testing of Deep Code Models: A Systematic Literature Review 

**Authors**: Ali Asgari, Milan de Koning, Pouria Derakhshanfar, Annibale Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2507.22610)  

**Abstract**: Large language models and deep learning models designed for code intelligence have revolutionized the software engineering field due to their ability to perform various code-related tasks. These models can process source code and software artifacts with high accuracy in tasks such as code completion, defect detection, and code summarization; therefore, they can potentially become an integral part of modern software engineering practices. Despite these capabilities, robustness remains a critical quality attribute for deep-code models as they may produce different results under varied and adversarial conditions (e.g., variable renaming). Metamorphic testing has become a widely used approach to evaluate models' robustness by applying semantic-preserving transformations to input programs and analyzing the stability of model outputs. While prior research has explored testing deep learning models, this systematic literature review focuses specifically on metamorphic testing for deep code models. By studying 45 primary papers, we analyze the transformations, techniques, and evaluation methods used to assess robustness. Our review summarizes the current landscape, identifying frequently evaluated models, programming tasks, datasets, target languages, and evaluation metrics, and highlights key challenges and future directions for advancing the field. 

---
# VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning 

**Authors**: Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22607)  

**Abstract**: Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach. 

---
# BALSAM: A Platform for Benchmarking Arabic Large Language Models 

**Authors**: Rawan Al-Matham, Kareem Darwish, Raghad Al-Rasheed, Waad Alshammari, Muneera Alhoshan, Amal Almazrua, Asma Al Wazrah, Mais Alheraki, Firoj Alam, Preslav Nakov, Norah Alzahrani, Eman alBilali, Nizar Habash, Abdelrahman El-Sheikh, Muhammad Elmallah, Haonan Li, Hamdy Mubarak, Mohamed Anwar, Zaid Alyafeai, Ahmed Abdelali, Nora Altwairesh, Maram Hasanain, Abdulmohsen Al Thubaity, Shady Shehata, Bashar Alhafni, Injy Hamed, Go Inoue, Khalid Elmadani, Ossama Obeid, Fatima Haouari, Tamer Elsayed, Emad Alghamdi, Khalid Almubarak, Saied Alshahrani, Ola Aljarrah, Safa Alajlan, Areej Alshaqarawi, Maryam Alshihri, Sultana Alghurabi, Atikah Alzeghayer, Afrah Altamimi, Abdullah Alfaifi, Abdulrahman AlOsaimy  

**Link**: [PDF](https://arxiv.org/pdf/2507.22603)  

**Abstract**: The impressive advancement of Large Language Models (LLMs) in English has not been matched across all languages. In particular, LLM performance in Arabic lags behind, due to data scarcity, linguistic diversity of Arabic and its dialects, morphological complexity, etc. Progress is further hindered by the quality of Arabic benchmarks, which typically rely on static, publicly available data, lack comprehensive task coverage, or do not provide dedicated platforms with blind test sets. This makes it challenging to measure actual progress and to mitigate data contamination. Here, we aim to bridge these gaps. In particular, we introduce BALSAM, a comprehensive, community-driven benchmark aimed at advancing Arabic LLM development and evaluation. It includes 78 NLP tasks from 14 broad categories, with 52K examples divided into 37K test and 15K development, and a centralized, transparent platform for blind evaluation. We envision BALSAM as a unifying platform that sets standards and promotes collaborative research to advance Arabic LLM capabilities. 

---
# RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment 

**Authors**: Marcos Fuster-Pena, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2507.22580)  

**Abstract**: Automated Program Repair (APR) seeks to automatically correct software bugs without requiring human intervention. However, existing tools tend to generate patches that satisfy test cases without fixing the underlying bug, those are known as overfitting patches. To address this issue, Automated Patch Correctness Assessment (APCA) attempts to identify overfitting patches generated by APR tools. It can be solved as a static approach, meaning that no additional information is needed beyond the original and fixed code snippets. Current static techniques often struggle with reliability, flexibility and transparency. To address these issues, we introduce RePaCA, a novel static APCA technique that leverages Large Language Models (LLMs) specialized in thinking tasks. Our model is prompted with both buggy and fixed code snippets and guided to generate a Chain of Thought that analyses code differences, reasons about how the patch addresses the root cause, and ultimately provides a binary classification: correct or overfitting. To enhance these reasoning capabilities for the APCA task specifically, the LLM is finetuned using Reinforcement Learning with the Group Relative Policy Optimization algorithm. When evaluated on a standard Defects4J-derived test, our approach achieves state-of-the-art performance, with 83.1% accuracy and an 84.8% F1-score. Furthermore, our model demonstrates superior generalization capabilities when trained on different datasets, outperforming the leading technique. This reasoning capability also provides enhanced explainability for the patch assessment. These findings underscore the considerable promise of finetuned, reasoning LLMs to advance static APCA by enhancing accuracy, generalization, and explainability. 

---
# A Mean-Field Theory of $Θ$-Expectations 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22577)  

**Abstract**: The canonical theory of sublinear expectations, a foundation of stochastic calculus under ambiguity, is insensitive to the non-convex geometry of primitive uncertainty models. This paper develops a new stochastic calculus for a structured class of such non-convex models. We introduce a class of fully coupled Mean-Field Forward-Backward Stochastic Differential Equations where the BSDE driver is defined by a pointwise maximization over a law-dependent, non-convex set. Mathematical tractability is achieved via a uniform strong concavity assumption on the driver with respect to the control variable, which ensures the optimization admits a unique and stable solution. A central contribution is to establish the Lipschitz stability of this optimizer from primitive geometric and regularity conditions, which underpins the entire well-posedness theory. We prove local and global well-posedness theorems for the FBSDE system. The resulting valuation functional, the $\Theta$-Expectation, is shown to be dynamically consistent and, most critically, to violate the axiom of sub-additivity. This, along with its failure to be translation invariant, demonstrates its fundamental departure from the convex paradigm. This work provides a rigorous foundation for stochastic calculus under a class of non-convex, endogenous ambiguity. 

---
# COOkeD: Ensemble-based OOD detection in the era of zero-shot CLIP 

**Authors**: Galadrielle Humblot-Renaux, Gianni Franchi, Sergio Escalera, Thomas B. Moeslund  

**Link**: [PDF](https://arxiv.org/pdf/2507.22576)  

**Abstract**: Out-of-distribution (OOD) detection is an important building block in trustworthy image recognition systems as unknown classes may arise at test-time. OOD detection methods typically revolve around a single classifier, leading to a split in the research field between the classical supervised setting (e.g. ResNet18 classifier trained on CIFAR100) vs. the zero-shot setting (class names fed as prompts to CLIP). In both cases, an overarching challenge is that the OOD detection performance is implicitly constrained by the classifier's capabilities on in-distribution (ID) data. In this work, we show that given a little open-mindedness from both ends, remarkable OOD detection can be achieved by instead creating a heterogeneous ensemble - COOkeD combines the predictions of a closed-world classifier trained end-to-end on a specific dataset, a zero-shot CLIP classifier, and a linear probe classifier trained on CLIP image features. While bulky at first sight, this approach is modular, post-hoc and leverages the availability of pre-trained VLMs, thus introduces little overhead compared to training a single standard classifier. We evaluate COOkeD on popular CIFAR100 and ImageNet benchmarks, but also consider more challenging, realistic settings ranging from training-time label noise, to test-time covariate shift, to zero-shot shift which has been previously overlooked. Despite its simplicity, COOkeD achieves state-of-the-art performance and greater robustness compared to both classical and CLIP-based OOD detection methods. Code is available at this https URL 

---
# Explaining Deep Network Classification of Matrices: A Case Study on Monotonicity 

**Authors**: Leandro Farina, Sergey Korotov  

**Link**: [PDF](https://arxiv.org/pdf/2507.22570)  

**Abstract**: This work demonstrates a methodology for using deep learning to discover simple, practical criteria for classifying matrices based on abstract algebraic properties. By combining a high-performance neural network with explainable AI (XAI) techniques, we can distill a model's learned strategy into human-interpretable rules. We apply this approach to the challenging case of monotone matrices, defined by the condition that their inverses are entrywise nonnegative. Despite their simple definition, an easy characterization in terms of the matrix elements or the derived parameters is not known. Here, we present, to the best of our knowledge, the first systematic machine-learning approach for deriving a practical criterion that distinguishes monotone from non-monotone matrices. After establishing a labelled dataset by randomly generated monotone and non-monotone matrices uniformly on $(-1,1)$, we employ deep neural network algorithms for classifying the matrices as monotone or non-monotone, using both their entries and a comprehensive set of matrix features. By saliency methods, such as integrated gradients, we identify among all features, two matrix parameters which alone provide sufficient information for the matrix classification, with $95\%$ accuracy, namely the absolute values of the two lowest-order coefficients, $c_0$ and $c_1$ of the matrix's characteristic polynomial. A data-driven study of 18,000 random $7\times7$ matrices shows that the monotone class obeys $\lvert c_{0}/c_{1}\rvert\le0.18$ with probability $>99.98\%$; because $\lvert c_{0}/c_{1}\rvert = 1/\mathrm{tr}(A^{-1})$ for monotone $A$, this is equivalent to the simple bound $\mathrm{tr}(A^{-1})\ge5.7$. 

---
# Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22565)  

**Abstract**: The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks. 

---
# Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs 

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22564)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems. 

---
# aLLoyM: A large language model for alloy phase diagram prediction 

**Authors**: Yuna Oikawa, Guillaume Deffrennes, Taichi Abe, Ryo Tamura, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2507.22558)  

**Abstract**: Large Language Models (LLMs) are general-purpose tools with wide-ranging applications, including in materials science. In this work, we introduce aLLoyM, a fine-tuned LLM specifically trained on alloy compositions, temperatures, and their corresponding phase information. To develop aLLoyM, we curated question-and-answer (Q&A) pairs for binary and ternary phase diagrams using the open-source Computational Phase Diagram Database (CPDDB) and assessments based on CALPHAD (CALculation of PHAse Diagrams). We fine-tuned Mistral, an open-source pre-trained LLM, for two distinct Q&A formats: multiple-choice and short-answer. Benchmark evaluations demonstrate that fine-tuning substantially enhances performance on multiple-choice phase diagram questions. Moreover, the short-answer model of aLLoyM exhibits the ability to generate novel phase diagrams from its components alone, underscoring its potential to accelerate the discovery of previously unexplored materials systems. To promote further research and adoption, we have publicly released the short-answer fine-tuned version of aLLoyM, along with the complete benchmarking Q&A dataset, on Hugging Face. 

---
# RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning 

**Authors**: Kiseong Hong, Gyeong-hyeon Kim, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22553)  

**Abstract**: Prompt-based continual learning provides a rehearsal-free solution by tuning small sets of parameters while keeping pre-trained models frozen. To meet the complex demands of sequential tasks, it is crucial to integrate task-specific knowledge within prompts effectively. However, existing works rely on either fixed learned prompts (i.e., prompts whose representations remain unchanged during new task learning) or on prompts generated from an entangled task-shared space, limiting the representational diversity of the integrated prompt. To address this issue, we propose a novel prompt-evolving mechanism to adaptively aggregate base prompts (i.e., task-specific prompts) into a unified prompt while ensuring diversity. By transforming and aligning base prompts, both previously learned and newly introduced, our approach continuously evolves accumulated knowledge to facilitate learning new tasks. We further introduce a learnable probabilistic gate that adaptively determines which layers to activate during the evolution process. We validate our method on image classification and video action recognition tasks in class-incremental learning, achieving average gains of 9.07% and 7.40% over existing methods across all scenarios. 

---
# A surrogate model for topology optimisation of elastic structures via parametric autoencoders 

**Authors**: Matteo Giacomini, Antonio Huerta  

**Link**: [PDF](https://arxiv.org/pdf/2507.22539)  

**Abstract**: A surrogate-based topology optimisation algorithm for linear elastic structures under parametric loads and boundary conditions is proposed. Instead of learning the parametric solution of the state (and adjoint) problems or the optimisation trajectory as a function of the iterations, the proposed approach devises a surrogate version of the entire optimisation pipeline. First, the method predicts a quasi-optimal topology for a given problem configuration as a surrogate model of high-fidelity topologies optimised with the homogenisation method. This is achieved by means of a feed-forward net learning the mapping between the input parameters characterising the system setup and a latent space determined by encoder/decoder blocks reducing the dimensionality of the parametric topology optimisation problem and reconstructing a high-dimensional representation of the topology. Then, the predicted topology is used as an educated initial guess for a computationally efficient algorithm penalising the intermediate values of the design variable, while enforcing the governing equations of the system. This step allows the method to correct potential errors introduced by the surrogate model, eliminate artifacts, and refine the design in order to produce topologies consistent with the underlying physics. Different architectures are proposed and the approximation and generalisation capabilities of the resulting models are numerically evaluated. The quasi-optimal topologies allow to outperform the high-fidelity optimiser by reducing the average number of optimisation iterations by $53\%$ while achieving discrepancies below $4\%$ in the optimal value of the objective functional, even in the challenging scenario of testing the model to extrapolate beyond the training and validation domain. 

---
# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records 

**Authors**: Dongchen Li, Jitao Liang, Wei Li, Xiaoyu Wang, Longbing Cao, Kun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22533)  

**Abstract**: Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and multilingual nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these diverse settings, CliCARE significantly outperforms strong baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by expert oncologists. 

---
# HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors 

**Authors**: Xincheng Yao, Yijun Yang, Kangwei Guo, Ruiqiang Xiao, Haipeng Zhou, Haisu Tao, Jian Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22530)  

**Abstract**: The segmentation of the hepatic vasculature in surgical videos holds substantial clinical significance in the context of hepatectomy procedures. However, owing to the dearth of an appropriate dataset and the inherently complex task characteristics, few researches have been reported in this domain. To address this issue, we first introduce a high quality frame-by-frame annotated hepatic vasculature dataset containing 35 long hepatectomy videos and 11442 high-resolution frames. On this basis, we propose a novel high-resolution video vasculature segmentation network, dubbed as HRVVS. We innovatively embed a pretrained visual autoregressive modeling (VAR) model into different layers of the hierarchical encoder as prior information to reduce the information degradation generated during the downsampling process. In addition, we designed a dynamic memory decoder on a multi-view segmentation network to minimize the transmission of redundant information while preserving more details between frames. Extensive experiments on surgical video datasets demonstrate that our proposed HRVVS significantly outperforms the state-of-the-art methods. The source code and dataset will be publicly available at \href{this https URL}{this https URL}. 

---
# Accident-Driven Congestion Prediction and Simulation: An Explainable Framework Using Advanced Clustering and Bayesian Networks 

**Authors**: Kranthi Kumar Talluri, Galia Weidl, Vaishnavi Kasuluru  

**Link**: [PDF](https://arxiv.org/pdf/2507.22529)  

**Abstract**: Traffic congestion due to uncertainties, such as accidents, is a significant issue in urban areas, as the ripple effect of accidents causes longer delays, increased emissions, and safety concerns. To address this issue, we propose a robust framework for predicting the impact of accidents on congestion. We implement Automated Machine Learning (AutoML)-enhanced Deep Embedding Clustering (DEC) to assign congestion labels to accident data and predict congestion probability using a Bayesian Network (BN). The Simulation of Urban Mobility (SUMO) simulation is utilized to evaluate the correctness of BN predictions using evidence-based scenarios. Results demonstrate that the AutoML-enhanced DEC has outperformed traditional clustering approaches. The performance of the proposed BN model achieved an overall accuracy of 95.6%, indicating its ability to understand the complex relationship of accidents causing congestion. Validation in SUMO with evidence-based scenarios demonstrated that the BN model's prediction of congestion states closely matches those of SUMO, indicating the high reliability of the proposed BN model in ensuring smooth urban mobility. 

---
# Recognizing Actions from Robotic View for Natural Human-Robot Interaction 

**Authors**: Ziyi Wang, Peiming Li, Hong Liu, Zhichao Deng, Can Wang, Jun Liu, Junsong Yuan, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22522)  

**Abstract**: Natural Human-Robot Interaction (N-HRI) requires robots to recognize human actions at varying distances and states, regardless of whether the robot itself is in motion or stationary. This setup is more flexible and practical than conventional human action recognition tasks. However, existing benchmarks designed for traditional action recognition fail to address the unique complexities in N-HRI due to limited data, modalities, task categories, and diversity of subjects and environments. To address these challenges, we introduce ACTIVE (Action from Robotic View), a large-scale dataset tailored specifically for perception-centric robotic views prevalent in mobile service robots. ACTIVE comprises 30 composite action categories, 80 participants, and 46,868 annotated video instances, covering both RGB and point cloud modalities. Participants performed various human actions in diverse environments at distances ranging from 3m to 50m, while the camera platform was also mobile, simulating real-world scenarios of robot perception with varying camera heights due to uneven ground. This comprehensive and challenging benchmark aims to advance action and attribute recognition research in N-HRI. Furthermore, we propose ACTIVE-PC, a method that accurately perceives human actions at long distances using Multilevel Neighborhood Sampling, Layered Recognizers, Elastic Ellipse Query, and precise decoupling of kinematic interference from human actions. Experimental results demonstrate the effectiveness of ACTIVE-PC. Our code is available at: this https URL. 

---
# LoReUn: Data Itself Implicitly Provides Cues to Improve Machine Unlearning 

**Authors**: Xiang Li, Qianli Shen, Haonan Wang, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22499)  

**Abstract**: Recent generative models face significant risks of producing harmful content, which has underscored the importance of machine unlearning (MU) as a critical technique for eliminating the influence of undesired data. However, existing MU methods typically assign the same weight to all data to be forgotten, which makes it difficult to effectively forget certain data that is harder to unlearn than others. In this paper, we empirically demonstrate that the loss of data itself can implicitly reflect its varying difficulty. Building on this insight, we introduce Loss-based Reweighting Unlearning (LoReUn), a simple yet effective plug-and-play strategy that dynamically reweights data during the unlearning process with minimal additional computational overhead. Our approach significantly reduces the gap between existing MU methods and exact unlearning in both image classification and generation tasks, effectively enhancing the prevention of harmful content generation in text-to-image diffusion models. 

---
# Robust Adverse Weather Removal via Spectral-based Spatial Grouping 

**Authors**: Yuhwan Jeong, Yunseo Yang, Youngjo Yoon, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.22498)  

**Abstract**: Adverse weather conditions cause diverse and complex degradation patterns, driving the development of All-in-One (AiO) models. However, recent AiO solutions still struggle to capture diverse degradations, since global filtering methods like direct operations on the frequency domain fail to handle highly variable and localized distortions. To address these issue, we propose Spectral-based Spatial Grouping Transformer (SSGformer), a novel approach that leverages spectral decomposition and group-wise attention for multi-weather image restoration. SSGformer decomposes images into high-frequency edge features using conventional edge detection and low-frequency information via Singular Value Decomposition. We utilize multi-head linear attention to effectively model the relationship between these features. The fused features are integrated with the input to generate a grouping-mask that clusters regions based on the spatial similarity and image texture. To fully leverage this mask, we introduce a group-wise attention mechanism, enabling robust adverse weather removal and ensuring consistent performance across diverse weather conditions. We also propose a Spatial Grouping Transformer Block that uses both channel attention and spatial attention, effectively balancing feature-wise relationships and spatial dependencies. Extensive experiments show the superiority of our approach, validating its effectiveness in handling the varied and intricate adverse weather degradations. 

---
# LVM-GP: Uncertainty-Aware PDE Solver via coupling latent variable model and Gaussian process 

**Authors**: Xiaodong Feng, Ling Guo, Xiaoliang Wan, Hao Wu, Tao Zhou, Wenwen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22493)  

**Abstract**: We propose a novel probabilistic framework, termed LVM-GP, for uncertainty quantification in solving forward and inverse partial differential equations (PDEs) with noisy data. The core idea is to construct a stochastic mapping from the input to a high-dimensional latent representation, enabling uncertainty-aware prediction of the solution. Specifically, the architecture consists of a confidence-aware encoder and a probabilistic decoder. The encoder implements a high-dimensional latent variable model based on a Gaussian process (LVM-GP), where the latent representation is constructed by interpolating between a learnable deterministic feature and a Gaussian process prior, with the interpolation strength adaptively controlled by a confidence function learned from data. The decoder defines a conditional Gaussian distribution over the solution field, where the mean is predicted by a neural operator applied to the latent representation, allowing the model to learn flexible function-to-function mapping. Moreover, physical laws are enforced as soft constraints in the loss function to ensure consistency with the underlying PDE structure. Compared to existing approaches such as Bayesian physics-informed neural networks (B-PINNs) and deep ensembles, the proposed framework can efficiently capture functional dependencies via merging a latent Gaussian process and neural operator, resulting in competitive predictive accuracy and robust uncertainty quantification. Numerical experiments demonstrate the effectiveness and reliability of the method. 

---
# Proto-EVFL: Enhanced Vertical Federated Learning via Dual Prototype with Extremely Unaligned Data 

**Authors**: Wei Guo, Yiyang Duan, Zhaojun Hu, Yiqi Tong, Fuzhen Zhuang, Xiao Zhang, Jin Dong, Ruofan Wu, Tengfei Liu, Yifan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.22488)  

**Abstract**: In vertical federated learning (VFL), multiple enterprises address aligned sample scarcity by leveraging massive locally unaligned samples to facilitate collaborative learning. However, unaligned samples across different parties in VFL can be extremely class-imbalanced, leading to insufficient feature representation and limited model prediction space. Specifically, class-imbalanced problems consist of intra-party class imbalance and inter-party class imbalance, which can further cause local model bias and feature contribution inconsistency issues, respectively. To address the above challenges, we propose Proto-EVFL, an enhanced VFL framework via dual prototypes. We first introduce class prototypes for each party to learn relationships between classes in the latent space, allowing the active party to predict unseen classes. We further design a probabilistic dual prototype learning scheme to dynamically select unaligned samples by conditional optimal transport cost with class prior probability. Moreover, a mixed prior guided module guides this selection process by combining local and global class prior probabilities. Finally, we adopt an \textit{adaptive gated feature aggregation strategy} to mitigate feature contribution inconsistency by dynamically weighting and aggregating local features across different parties. We proved that Proto-EVFL, as the first bi-level optimization framework in VFL, has a convergence rate of 1/\sqrt T. Extensive experiments on various datasets validate the superiority of our Proto-EVFL. Even in a zero-shot scenario with one unseen class, it outperforms baselines by at least 6.97% 

---
# Physics-constrained generative machine learning-based high-resolution downscaling of Greenland's surface mass balance and surface temperature 

**Authors**: Nils Bochow, Philipp Hess, Alexander Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.22485)  

**Abstract**: Accurate, high-resolution projections of the Greenland ice sheet's surface mass balance (SMB) and surface temperature are essential for understanding future sea-level rise, yet current approaches are either computationally demanding or limited to coarse spatial scales. Here, we introduce a novel physics-constrained generative modeling framework based on a consistency model (CM) to downscale low-resolution SMB and surface temperature fields by a factor of up to 32 (from 160 km to 5 km grid spacing) in a few sampling steps. The CM is trained on monthly outputs of the regional climate model MARv3.12 and conditioned on ice-sheet topography and insolation. By enforcing a hard conservation constraint during inference, we ensure approximate preservation of SMB and temperature sums on the coarse spatial scale as well as robust generalization to extreme climate states without retraining. On the test set, our constrained CM achieves a continued ranked probability score of 6.31 mmWE for the SMB and 0.1 K for the surface temperature, outperforming interpolation-based downscaling. Together with spatial power-spectral analysis, we demonstrate that the CM faithfully reproduces variability across spatial scales. We further apply bias-corrected outputs of the NorESM2 Earth System Model as inputs to our CM, to demonstrate the potential of our model to directly downscale ESM fields. Our approach delivers realistic, high-resolution climate forcing for ice-sheet simulations with fast inference and can be readily integrated into Earth-system and ice-sheet model workflows to improve projections of the future contribution to sea-level rise from Greenland and potentially other ice sheets and glaciers too. 

---
# Towards Blind Bitstream-corrupted Video Recovery via a Visual Foundation Model-driven Framework 

**Authors**: Tianyi Liu, Kejun Wu, Chen Cai, Yi Wang, Kim-Hui Yap, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2507.22481)  

**Abstract**: Video signals are vulnerable in multimedia communication and storage systems, as even slight bitstream-domain corruption can lead to significant pixel-domain degradation. To recover faithful spatio-temporal content from corrupted inputs, bitstream-corrupted video recovery has recently emerged as a challenging and understudied task. However, existing methods require time-consuming and labor-intensive annotation of corrupted regions for each corrupted video frame, resulting in a large workload in practice. In addition, high-quality recovery remains difficult as part of the local residual information in corrupted frames may mislead feature completion and successive content recovery. In this paper, we propose the first blind bitstream-corrupted video recovery framework that integrates visual foundation models with a recovery model, which is adapted to different types of corruption and bitstream-level prompts. Within the framework, the proposed Detect Any Corruption (DAC) model leverages the rich priors of the visual foundation model while incorporating bitstream and corruption knowledge to enhance corruption localization and blind recovery. Additionally, we introduce a novel Corruption-aware Feature Completion (CFC) module, which adaptively processes residual contributions based on high-level corruption understanding. With VFM-guided hierarchical feature augmentation and high-level coordination in a mixture-of-residual-experts (MoRE) structure, our method suppresses artifacts and enhances informative residuals. Comprehensive evaluations show that the proposed method achieves outstanding performance in bitstream-corrupted video recovery without requiring a manually labeled mask sequence. The demonstrated effectiveness will help to realize improved user experience, wider application scenarios, and more reliable multimedia communication and storage systems. 

---
# LIDAR: Lightweight Adaptive Cue-Aware Fusion Vision Mamba for Multimodal Segmentation of Structural Cracks 

**Authors**: Hui Liu, Chen Jia, Fan Shi, Xu Cheng, Mengfei Shi, Xia Xie, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22477)  

**Abstract**: Achieving pixel-level segmentation with low computational cost using multimodal data remains a key challenge in crack segmentation tasks. Existing methods lack the capability for adaptive perception and efficient interactive fusion of cross-modal features. To address these challenges, we propose a Lightweight Adaptive Cue-Aware Vision Mamba network (LIDAR), which efficiently perceives and integrates morphological and textural cues from different modalities under multimodal crack scenarios, generating clear pixel-level crack segmentation maps. Specifically, LIDAR is composed of a Lightweight Adaptive Cue-Aware Visual State Space module (LacaVSS) and a Lightweight Dual Domain Dynamic Collaborative Fusion module (LD3CF). LacaVSS adaptively models crack cues through the proposed mask-guided Efficient Dynamic Guided Scanning Strategy (EDG-SS), while LD3CF leverages an Adaptive Frequency Domain Perceptron (AFDP) and a dual-pooling fusion strategy to effectively capture spatial and frequency-domain cues across modalities. Moreover, we design a Lightweight Dynamically Modulated Multi-Kernel convolution (LDMK) to perceive complex morphological structures with minimal computational overhead, replacing most convolutional operations in LIDAR. Experiments on three datasets demonstrate that our method outperforms other state-of-the-art (SOTA) methods. On the light-field depth dataset, our method achieves 0.8204 in F1 and 0.8465 in mIoU with only 5.35M parameters. Code and datasets are available at this https URL. 

---
# Visual Language Models as Zero-Shot Deepfake Detectors 

**Authors**: Viacheslav Pirogov  

**Link**: [PDF](https://arxiv.org/pdf/2507.22469)  

**Abstract**: The contemporary phenomenon of deepfakes, utilizing GAN or diffusion models for face swapping, presents a substantial and evolving threat in digital media, identity verification, and a multitude of other systems. The majority of existing methods for detecting deepfakes rely on training specialized classifiers to distinguish between genuine and manipulated images, focusing only on the image domain without incorporating any auxiliary tasks that could enhance robustness. In this paper, inspired by the zero-shot capabilities of Vision Language Models, we propose a novel VLM-based approach to image classification and then evaluate it for deepfake detection. Specifically, we utilize a new high-quality deepfake dataset comprising 60,000 images, on which our zero-shot models demonstrate superior performance to almost all existing methods. Subsequently, we compare the performance of the best-performing architecture, InstructBLIP, on the popular deepfake dataset DFDC-P against traditional methods in two scenarios: zero-shot and in-domain fine-tuning. Our results demonstrate the superiority of VLMs over traditional classifiers. 

---
# Towards Simulating Social Influence Dynamics with LLM-based Multi-agents 

**Authors**: Hsien-Tsung Lin, Pei-Cing Huang, Chan-Tung Ku, Chan Hsu, Pei-Xuan Shieh, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22467)  

**Abstract**: Recent advancements in Large Language Models offer promising capabilities to simulate complex human social interactions. We investigate whether LLM-based multi-agent simulations can reproduce core human social dynamics observed in online forums. We evaluate conformity dynamics, group polarization, and fragmentation across different model scales and reasoning capabilities using a structured simulation framework. Our findings indicate that smaller models exhibit higher conformity rates, whereas models optimized for reasoning are more resistant to social influence. 

---
# Shallow Features Matter: Hierarchical Memory with Heterogeneous Interaction for Unsupervised Video Object Segmentation 

**Authors**: Zheng Xiangyu, He Songcheng, Li Wanyun, Li Xiaoqiang, Zhang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.22465)  

**Abstract**: Unsupervised Video Object Segmentation (UVOS) aims to predict pixel-level masks for the most salient objects in videos without any prior annotations. While memory mechanisms have been proven critical in various video segmentation paradigms, their application in UVOS yield only marginal performance gains despite sophisticated design. Our analysis reveals a simple but fundamental flaw in existing methods: over-reliance on memorizing high-level semantic features. UVOS inherently suffers from the deficiency of lacking fine-grained information due to the absence of pixel-level prior knowledge. Consequently, memory design relying solely on high-level features, which predominantly capture abstract semantic cues, is insufficient to generate precise predictions. To resolve this fundamental issue, we propose a novel hierarchical memory architecture to incorporate both shallow- and high-level features for memory, which leverages the complementary benefits of pixel and semantic information. Furthermore, to balance the simultaneous utilization of the pixel and semantic memory features, we propose a heterogeneous interaction mechanism to perform pixel-semantic mutual interactions, which explicitly considers their inherent feature discrepancies. Through the design of Pixel-guided Local Alignment Module (PLAM) and Semantic-guided Global Integration Module (SGIM), we achieve delicate integration of the fine-grained details in shallow-level memory and the semantic representations in high-level memory. Our Hierarchical Memory with Heterogeneous Interaction Network (HMHI-Net) consistently achieves state-of-the-art performance across all UVOS and video saliency detection benchmarks. Moreover, HMHI-Net consistently exhibits high performance across different backbones, further demonstrating its superiority and robustness. Project page: this https URL . 

---
# Towards Interpretable Renal Health Decline Forecasting via Multi-LMM Collaborative Reasoning Framework 

**Authors**: Peng-Yi Wu, Pei-Cing Huang, Ting-Yu Chen, Chantung Ku, Ming-Yen Lin, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22464)  

**Abstract**: Accurate and interpretable prediction of estimated glomerular filtration rate (eGFR) is essential for managing chronic kidney disease (CKD) and supporting clinical decisions. Recent advances in Large Multimodal Models (LMMs) have shown strong potential in clinical prediction tasks due to their ability to process visual and textual information. However, challenges related to deployment cost, data privacy, and model reliability hinder their adoption. In this study, we propose a collaborative framework that enhances the performance of open-source LMMs for eGFR forecasting while generating clinically meaningful explanations. The framework incorporates visual knowledge transfer, abductive reasoning, and a short-term memory mechanism to enhance prediction accuracy and interpretability. Experimental results show that the proposed framework achieves predictive performance and interpretability comparable to proprietary models. It also provides plausible clinical reasoning processes behind each prediction. Our method sheds new light on building AI systems for healthcare that combine predictive accuracy with clinically grounded interpretability. 

---
# What is an "Abstract Reasoner"? Revisiting Experiments and Arguments about Large Language Models 

**Authors**: Tian Yun, Chen Sun, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.22457)  

**Abstract**: Recent work has argued that large language models (LLMs) are not "abstract reasoners", citing their poor zero-shot performance on a variety of challenging tasks as evidence. We revisit these experiments in order to add nuance to the claim. First, we show that while LLMs indeed perform poorly in a zero-shot setting, even tuning a small subset of parameters for input encoding can enable near-perfect performance. However, we also show that this finetuning does not necessarily transfer across datasets. We take this collection of empirical results as an invitation to (re-)open the discussion of what it means to be an "abstract reasoner", and why it matters whether LLMs fit the bill. 

---
# RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function 

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22446)  

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms. 

---
# AI-generated stories favour stability over change: homogeneity and cultural stereotyping in narratives generated by gpt-4o-mini 

**Authors**: Jill Walker Rettberg, Hermann Wigers  

**Link**: [PDF](https://arxiv.org/pdf/2507.22445)  

**Abstract**: Can a language model trained largely on Anglo-American texts generate stories that are culturally relevant to other nationalities? To find out, we generated 11,800 stories - 50 for each of 236 countries - by sending the prompt "Write a 1500 word potential {demonym} story" to OpenAI's model gpt-4o-mini. Although the stories do include surface-level national symbols and themes, they overwhelmingly conform to a single narrative plot structure across countries: a protagonist lives in or returns home to a small town and resolves a minor conflict by reconnecting with tradition and organising community events. Real-world conflicts are sanitised, romance is almost absent, and narrative tension is downplayed in favour of nostalgia and reconciliation. The result is a narrative homogenisation: an AI-generated synthetic imaginary that prioritises stability above change and tradition above growth. We argue that the structural homogeneity of AI-generated narratives constitutes a distinct form of AI bias, a narrative standardisation that should be acknowledged alongside the more familiar representational bias. These findings are relevant to literary studies, narratology, critical AI studies, NLP research, and efforts to improve the cultural alignment of generative AI. 

---
# Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss 

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22428)  

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy. 

---
# Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance 

**Authors**: Songsheng Wang, Rucheng Yu, Zhihang Yuan, Chao Yu, Feng Gao, Yu Wang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22424)  

**Abstract**: Vision-Language-Action (VLA) models have made substantial progress by leveraging the robust capabilities of Visual Language Models (VLMs). However, VLMs' significant parameter size and autoregressive (AR) decoding nature impose considerable computational demands on VLA models. While Speculative Decoding (SD) has shown efficacy in accelerating Large Language Models (LLMs) by incorporating efficient drafting and parallel verification, allowing multiple tokens to be generated in one forward pass, its application to VLA models remains unexplored. This work introduces Spec-VLA, an SD framework designed to accelerate VLA models. Due to the difficulty of the action prediction task and the greedy decoding mechanism of the VLA models, the direct application of the advanced SD framework to the VLA prediction task yields a minor speed improvement. To boost the generation speed, we propose an effective mechanism to relax acceptance utilizing the relative distances represented by the action tokens of the VLA model. Empirical results across diverse test scenarios affirm the effectiveness of the Spec-VLA framework, and further analysis substantiates the impact of our proposed strategies, which enhance the acceptance length by 44%, achieving 1.42 times speedup compared with the OpenVLA baseline, without compromising the success rate. The success of the Spec-VLA framework highlights the potential for broader application of speculative execution in VLA prediction scenarios. 

---
# Efficient Spatial-Temporal Modeling for Real-Time Video Analysis: A Unified Framework for Action Recognition and Object Tracking 

**Authors**: Shahla John  

**Link**: [PDF](https://arxiv.org/pdf/2507.22421)  

**Abstract**: Real-time video analysis remains a challenging problem in computer vision, requiring efficient processing of both spatial and temporal information while maintaining computational efficiency. Existing approaches often struggle to balance accuracy and speed, particularly in resource-constrained environments. In this work, we present a unified framework that leverages advanced spatial-temporal modeling techniques for simultaneous action recognition and object tracking. Our approach builds upon recent advances in parallel sequence modeling and introduces a novel hierarchical attention mechanism that adaptively focuses on relevant spatial regions across temporal sequences. We demonstrate that our method achieves state-of-the-art performance on standard benchmarks while maintaining real-time inference speeds. Extensive experiments on UCF-101, HMDB-51, and MOT17 datasets show improvements of 3.2% in action recognition accuracy and 2.8% in tracking precision compared to existing methods, with 40% faster inference time. 

---
# Systematic Evaluation of Knowledge Graph Repair with Large Language Models 

**Authors**: Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, Alberto Sangiovanni-Vinentelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.22419)  

**Abstract**: We present a systematic approach for evaluating the quality of knowledge graph repairs with respect to constraint violations defined in shapes constraint language (SHACL). Current evaluation methods rely on \emph{ad hoc} datasets, which limits the rigorous analysis of repair systems in more general settings. Our method addresses this gap by systematically generating violations using a novel mechanism, termed violation-inducing operations (VIOs). We use the proposed evaluation framework to assess a range of repair systems which we build using large language models. We analyze the performance of these systems across different prompting strategies. Results indicate that concise prompts containing both the relevant violated SHACL constraints and key contextual information from the knowledge graph yield the best performance. 

---
# Aleatoric Uncertainty Medical Image Segmentation Estimation via Flow Matching 

**Authors**: Phi Van Nguyen, Ngoc Huynh Trinh, Duy Minh Lam Nguyen, Phu Loc Nguyen, Quoc Long Tran  

**Link**: [PDF](https://arxiv.org/pdf/2507.22418)  

**Abstract**: Quantifying aleatoric uncertainty in medical image segmentation is critical since it is a reflection of the natural variability observed among expert annotators. A conventional approach is to model the segmentation distribution using the generative model, but current methods limit the expression ability of generative models. While current diffusion-based approaches have demonstrated impressive performance in approximating the data distribution, their inherent stochastic sampling process and inability to model exact densities limit their effectiveness in accurately capturing uncertainty. In contrast, our proposed method leverages conditional flow matching, a simulation-free flow-based generative model that learns an exact density, to produce highly accurate segmentation results. By guiding the flow model on the input image and sampling multiple data points, our approach synthesizes segmentation samples whose pixel-wise variance reliably reflects the underlying data distribution. This sampling strategy captures uncertainties in regions with ambiguous boundaries, offering robust quantification that mirrors inter-annotator differences. Experimental results demonstrate that our method not only achieves competitive segmentation accuracy but also generates uncertainty maps that provide deeper insights into the reliability of the segmentation outcomes. The code for this paper is freely available at this https URL 

---
# NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large Language Models 

**Authors**: Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22411)  

**Abstract**: The Needle-in-a-Haystack (NIAH) benchmark is widely used to evaluate Large Language Models' (LLMs) ability to understand long contexts (LC). It evaluates the capability to identify query-relevant context within extensive query-irrelevant passages. Although this method serves as a widely accepted standard for evaluating long-context understanding, our findings suggest it may overestimate the true LC capability of LLMs. We demonstrate that even state-of-the-art models such as GPT-4o struggle to intactly incorporate given contexts made up of solely query-relevant ten sentences. In response, we introduce a novel benchmark, \textbf{NeedleChain}, where the context consists entirely of query-relevant information, requiring the LLM to fully grasp the input to answer correctly. Our benchmark allows for flexible context length and reasoning order, offering a more comprehensive analysis of LLM performance. Additionally, we propose an extremely simple yet compelling strategy to improve LC understanding capability of LLM: ROPE Contraction. Our experiments with various advanced LLMs reveal a notable disparity between their ability to process large contexts and their capacity to fully understand them. Source code and datasets are available at this https URL 

---
# Question Generation for Assessing Early Literacy Reading Comprehension 

**Authors**: Xiaocheng Yang, Sumuk Shashidhar, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2507.22410)  

**Abstract**: Assessment of reading comprehension through content-based interactions plays an important role in the reading acquisition process. In this paper, we propose a novel approach for generating comprehension questions geared to K-2 English learners. Our method ensures complete coverage of the underlying material and adaptation to the learner's specific proficiencies, and can generate a large diversity of question types at various difficulty levels to ensure a thorough evaluation. We evaluate the performance of various language models in this framework using the FairytaleQA dataset as the source material. Eventually, the proposed approach has the potential to become an important part of autonomous AI-driven English instructors. 

---
# MINR: Implicit Neural Representations with Masked Image Modelling 

**Authors**: Sua Lee, Joonhun Lee, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22404)  

**Abstract**: Self-supervised learning methods like masked autoencoders (MAE) have shown significant promise in learning robust feature representations, particularly in image reconstruction-based pretraining task. However, their performance is often strongly dependent on the masking strategies used during training and can degrade when applied to out-of-distribution data. To address these limitations, we introduce the masked implicit neural representations (MINR) framework that synergizes implicit neural representations with masked image modeling. MINR learns a continuous function to represent images, enabling more robust and generalizable reconstructions irrespective of masking strategies. Our experiments demonstrate that MINR not only outperforms MAE in in-domain scenarios but also in out-of-distribution settings, while reducing model complexity. The versatility of MINR extends to various self-supervised learning applications, confirming its utility as a robust and efficient alternative to existing frameworks. 

---
# SAEL: Leveraging Large Language Models with Adaptive Mixture-of-Experts for Smart Contract Vulnerability Detection 

**Authors**: Lei Yu, Shiqi Cheng, Zhirong Huang, Jingyuan Zhang, Chenjie Shen, Junyi Lu, Li Yang, Fengjun Zhang, Jiajia Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.22371)  

**Abstract**: With the increasing security issues in blockchain, smart contract vulnerability detection has become a research focus. Existing vulnerability detection methods have their limitations: 1) Static analysis methods struggle with complex scenarios. 2) Methods based on specialized pre-trained models perform well on specific datasets but have limited generalization capabilities. In contrast, general-purpose Large Language Models (LLMs) demonstrate impressive ability in adapting to new vulnerability patterns. However, they often underperform on specific vulnerability types compared to methods based on specialized pre-trained models. We also observe that explanations generated by general-purpose LLMs can provide fine-grained code understanding information, contributing to improved detection performance.
Inspired by these observations, we propose SAEL, an LLM-based framework for smart contract vulnerability detection. We first design targeted prompts to guide LLMs in identifying vulnerabilities and generating explanations, which serve as prediction features. Next, we apply prompt-tuning on CodeT5 and T5 to process contract code and explanations, enhancing task-specific performance. To combine the strengths of each approach, we introduce an Adaptive Mixture-of-Experts architecture. This dynamically adjusts feature weights via a Gating Network, which selects relevant features using TopK filtering and Softmax normalization, and incorporates a Multi-Head Self-Attention mechanism to enhance cross-feature relationships. This design enables effective integration of LLM predictions, explanation features, and code features through gradient optimization. The loss function jointly considers both independent feature performance and overall weighted predictions. Experiments show that SAEL outperforms existing methods across various vulnerabilities. 

---
# Exploring the Application of Visual Question Answering (VQA) for Classroom Activity Monitoring 

**Authors**: Sinh Trong Vu, Hieu Trung Pham, Dung Manh Nguyen, Hieu Minh Hoang, Nhu Hoang Le, Thu Ha Pham, Tai Tan Mai  

**Link**: [PDF](https://arxiv.org/pdf/2507.22369)  

**Abstract**: Classroom behavior monitoring is a critical aspect of educational research, with significant implications for student engagement and learning outcomes. Recent advancements in Visual Question Answering (VQA) models offer promising tools for automatically analyzing complex classroom interactions from video recordings. In this paper, we investigate the applicability of several state-of-the-art open-source VQA models, including LLaMA2, LLaMA3, QWEN3, and NVILA, in the context of classroom behavior analysis. To facilitate rigorous evaluation, we introduce our BAV-Classroom-VQA dataset derived from real-world classroom video recordings at the Banking Academy of Vietnam. We present the methodology for data collection, annotation, and benchmark the performance of the selected VQA models on this dataset. Our initial experimental results demonstrate that all four models achieve promising performance levels in answering behavior-related visual questions, showcasing their potential in future classroom analytics and intervention systems. 

---
# Object Recognition Datasets and Challenges: A Review 

**Authors**: Aria Salari, Abtin Djavadifar, Xiangrui Liu, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.22361)  

**Abstract**: Object recognition is among the fundamental tasks in the computer vision applications, paving the path for all other image understanding operations. In every stage of progress in object recognition research, efforts have been made to collect and annotate new datasets to match the capacity of the state-of-the-art algorithms. In recent years, the importance of the size and quality of datasets has been intensified as the utility of the emerging deep network techniques heavily relies on training data. Furthermore, datasets lay a fair benchmarking means for competitions and have proved instrumental to the advancements of object recognition research by providing quantifiable benchmarks for the developed models. Taking a closer look at the characteristics of commonly-used public datasets seems to be an important first step for data-driven and machine learning researchers. In this survey, we provide a detailed analysis of datasets in the highly investigated object recognition areas. More than 160 datasets have been scrutinized through statistics and descriptions. Additionally, we present an overview of the prominent object recognition benchmarks and competitions, along with a description of the metrics widely adopted for evaluation purposes in the computer vision community. All introduced datasets and challenges can be found online at this http URL. 

---
# GVD: Guiding Video Diffusion Model for Scalable Video Distillation 

**Authors**: Kunyang Li, Jeffrey A Chan Santiago, Sarinda Dhanesh Samarasinghe, Gaowen Liu, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.22360)  

**Abstract**: To address the larger computation and storage requirements associated with large video datasets, video dataset distillation aims to capture spatial and temporal information in a significantly smaller dataset, such that training on the distilled data has comparable performance to training on all of the data. We propose GVD: Guiding Video Diffusion, the first diffusion-based video distillation method. GVD jointly distills spatial and temporal features, ensuring high-fidelity video generation across diverse actions while capturing essential motion information. Our method's diverse yet representative distillations significantly outperform previous state-of-the-art approaches on the MiniUCF and HMDB51 datasets across 5, 10, and 20 Instances Per Class (IPC). Specifically, our method achieves 78.29 percent of the original dataset's performance using only 1.98 percent of the total number of frames in MiniUCF. Additionally, it reaches 73.83 percent of the performance with just 3.30 percent of the frames in HMDB51. Experimental results across benchmark video datasets demonstrate that GVD not only achieves state-of-the-art performance but can also generate higher resolution videos and higher IPC without significantly increasing computational cost. 

---
# From Articles to Code: On-Demand Generation of Core Algorithms from Scientific Publications 

**Authors**: Cameron S. Movassaghi, Amanda Momenzadeh, Jesse G. Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.22324)  

**Abstract**: Maintaining software packages imposes significant costs due to dependency management, bug fixes, and versioning. We show that rich method descriptions in scientific publications can serve as standalone specifications for modern large language models (LLMs), enabling on-demand code generation that could supplant human-maintained libraries. We benchmark state-of-the-art models (GPT-o4-mini-high, Gemini Pro 2.5, Claude Sonnet 4) by tasking them with implementing a diverse set of core algorithms drawn from original publications. Our results demonstrate that current LLMs can reliably reproduce package functionality with performance indistinguishable from conventional libraries. These findings foreshadow a paradigm shift toward flexible, on-demand code generation and away from static, human-maintained packages, which will result in reduced maintenance overhead by leveraging published articles as sufficient context for the automated implementation of analytical workflows. 

---
# Learning from Heterogeneous Structural MRI via Collaborative Domain Adaptation for Late-Life Depression Assessment 

**Authors**: Yuzhen Gao, Qianqian Wang, Yongheng Sun, Cui Wang, Yongquan Liang, Mingxia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22321)  

**Abstract**: Accurate identification of late-life depression (LLD) using structural brain MRI is essential for monitoring disease progression and facilitating timely intervention. However, existing learning-based approaches for LLD detection are often constrained by limited sample sizes (e.g., tens), which poses significant challenges for reliable model training and generalization. Although incorporating auxiliary datasets can expand the training set, substantial domain heterogeneity, such as differences in imaging protocols, scanner hardware, and population demographics, often undermines cross-domain transferability. To address this issue, we propose a Collaborative Domain Adaptation (CDA) framework for LLD detection using T1-weighted MRIs. The CDA leverages a Vision Transformer (ViT) to capture global anatomical context and a Convolutional Neural Network (CNN) to extract local structural features, with each branch comprising an encoder and a classifier. The CDA framework consists of three stages: (a) supervised training on labeled source data, (b) self-supervised target feature adaptation and (c) collaborative training on unlabeled target data. We first train ViT and CNN on source data, followed by self-supervised target feature adaptation by minimizing the discrepancy between classifier outputs from two branches to make the categorical boundary clearer. The collaborative training stage employs pseudo-labeled and augmented target-domain MRIs, enforcing prediction consistency under strong and weak augmentation to enhance domain robustness and generalization. Extensive experiments conducted on multi-site T1-weighted MRI data demonstrate that the CDA consistently outperforms state-of-the-art unsupervised domain adaptation methods. 

---
# AdapSCA-PSO: An Adaptive Localization Algorithm with AI-Based Hybrid SCA-PSO for IoT WSNs 

**Authors**: Ze Zhang, Qian Dong, Wenhan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22317)  

**Abstract**: The accurate localization of sensor nodes is a fundamental requirement for the practical application of the Internet of Things (IoT). To enable robust localization across diverse environments, this paper proposes a hybrid meta-heuristic localization algorithm. Specifically, the algorithm integrates the Sine Cosine Algorithm (SCA), which is effective in global search, with Particle Swarm Optimization (PSO), which excels at local search. An adaptive switching module is introduced to dynamically select between the two algorithms. Furthermore, the initialization, fitness evaluation, and parameter settings of the algorithm have been specifically redesigned and optimized to address the characteristics of the node localization problem. Simulation results across varying numbers of sensor nodes demonstrate that, compared to standalone PSO and the unoptimized SCAPSO algorithm, the proposed method significantly reduces the number of required iterations and achieves an average localization error reduction of 84.97%. 

---
# Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs 

**Authors**: Supantho Rakshit, Adele Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.22286)  

**Abstract**: The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs. 

---
# Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items 

**Authors**: Junting Wang, Chenghuan Guo, Jiao Yang, Yanhui Guo, Yan Gao, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2507.22268)  

**Abstract**: We introduce a novel self-supervised multi-modal relational item representation learning framework designed to infer substitutable and complementary items. Existing approaches primarily focus on modeling item-item associations deduced from user behaviors using graph neural networks (GNNs) or leveraging item content information. However, these methods often overlook critical challenges, such as noisy user behavior data and data sparsity due to the long-tailed distribution of these behaviors. In this paper, we propose MMSC, a self-supervised multi-modal relational item representation learning framework to address these challenges. Specifically, MMSC consists of three main components: (1) a multi-modal item representation learning module that leverages a multi-modal foundational model and learns from item metadata, (2) a self-supervised behavior-based representation learning module that denoises and learns from user behavior data, and (3) a hierarchical representation aggregation mechanism that integrates item representations at both the semantic and task levels. Additionally, we leverage LLMs to generate augmented training data, further enhancing the denoising process during training. We conduct extensive experiments on five real-world datasets, showing that MMSC outperforms existing baselines by 26.1% for substitutable recommendation and 39.2% for complementary recommendation. In addition, we empirically show that MMSC is effective in modeling cold-start items. 

---
# Promoting Online Safety by Simulating Unsafe Conversations with LLMs 

**Authors**: Owen Hoffman, Kangze Peng, Zehua You, Sajid Kamal, Sukrit Venkatagiri  

**Link**: [PDF](https://arxiv.org/pdf/2507.22267)  

**Abstract**: Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM. 

---
# SmartCLIP: Modular Vision-language Alignment with Identification Guarantees 

**Authors**: Shaoan Xie, Lingjing Kong, Yujia Zheng, Yu Yao, Zeyu Tang, Eric P. Xing, Guangyi Chen, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22264)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP)~\citep{radford2021learning} has emerged as a pivotal model in computer vision and multimodal learning, achieving state-of-the-art performance at aligning visual and textual representations through contrastive learning. However, CLIP struggles with potential information misalignment in many image-text datasets and suffers from entangled representation. On the one hand, short captions for a single image in datasets like MSCOCO may describe disjoint regions in the image, leaving the model uncertain about which visual features to retain or disregard. On the other hand, directly aligning long captions with images can lead to the retention of entangled details, preventing the model from learning disentangled, atomic concepts -- ultimately limiting its generalization on certain downstream tasks involving short prompts.
In this paper, we establish theoretical conditions that enable flexible alignment between textual and visual representations across varying levels of granularity. Specifically, our framework ensures that a model can not only \emph{preserve} cross-modal semantic information in its entirety but also \emph{disentangle} visual representations to capture fine-grained textual concepts. Building on this foundation, we introduce \ours, a novel approach that identifies and aligns the most relevant visual and textual representations in a modular manner. Superior performance across various tasks demonstrates its capability to handle information misalignment and supports our identification theory. The code is available at this https URL. 

---
# Agent-centric learning: from external reward maximization to internal knowledge curation 

**Authors**: Hanqi Zhou, Fryderyk Mantiuk, David G. Nagy, Charley M. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22255)  

**Abstract**: The pursuit of general intelligence has traditionally centered on external objectives: an agent's control over its environments or mastery of specific tasks. This external focus, however, can produce specialized agents that lack adaptability. We propose representational empowerment, a new perspective towards a truly agent-centric learning paradigm by moving the locus of control inward. This objective measures an agent's ability to controllably maintain and diversify its own knowledge structures. We posit that the capacity -- to shape one's own understanding -- is an element for achieving better ``preparedness'' distinct from direct environmental influence. Focusing on internal representations as the main substrate for computing empowerment offers a new lens through which to design adaptable intelligent systems. 

---
# Using Scaling Laws for Data Source Utility Estimation in Domain-Specific Pre-Training 

**Authors**: Oleksiy Ostapenko, Charles Guille-Escuret, Luke Kumar, Max Tian, Denis Kocetkov, Gopeshh Subbaraj, Raymond Li, Joel Lamy-Poirier, Sebastien Paquet, Torsten Scholak  

**Link**: [PDF](https://arxiv.org/pdf/2507.22250)  

**Abstract**: We introduce a framework for optimizing domain-specific dataset construction in foundation model training. Specifically, we seek a cost-efficient way to estimate the quality of data sources (e.g. synthetically generated or filtered web data, etc.) in order to make optimal decisions about resource allocation for data sourcing from these sources for the stage two pre-training phase, aka annealing, with the goal of specializing a generalist pre-trained model to specific domains. Our approach extends the usual point estimate approaches, aka micro-annealing, to estimating scaling laws by performing multiple annealing runs of varying compute spent on data curation and training. This addresses a key limitation in prior work, where reliance on point estimates for data scaling decisions can be misleading due to the lack of rank invariance across compute scales -- a phenomenon we confirm in our experiments. By systematically analyzing performance gains relative to acquisition costs, we find that scaling curves can be estimated for different data sources. Such scaling laws can inform cost effective resource allocation across different data acquisition methods (e.g. synthetic data), data sources (e.g. user or web data) and available compute resources. We validate our approach through experiments on a pre-trained model with 7 billion parameters. We adapt it to: a domain well-represented in the pre-training data -- the medical domain, and a domain underrepresented in the pretraining corpora -- the math domain. We show that one can efficiently estimate the scaling behaviors of a data source by running multiple annealing runs, which can lead to different conclusions, had one used point estimates using the usual micro-annealing technique instead. This enables data-driven decision-making for selecting and optimizing data sources. 

---
# Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems 

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany  

**Link**: [PDF](https://arxiv.org/pdf/2507.22239)  

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity. 

---
# RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation 

**Authors**: Dongyub Jude Lee, Zhenyi Ye, Pengcheng He  

**Link**: [PDF](https://arxiv.org/pdf/2507.22219)  

**Abstract**: Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores. 

---
# Quantum-Inspired Audio Unlearning: Towards Privacy-Preserving Voice Biometrics 

**Authors**: Shreyansh Pathak, Sonu Shreshtha, Richa Singh, Mayank Vatsa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22208)  

**Abstract**: The widespread adoption of voice-enabled authentication and audio biometric systems have significantly increased privacy vulnerabilities associated with sensitive speech data. Compliance with privacy regulations such as GDPR's right to be forgotten and India's DPDP Act necessitates targeted and efficient erasure of individual-specific voice signatures from already-trained biometric models. Existing unlearning methods designed for visual data inadequately handle the sequential, temporal, and high-dimensional nature of audio signals, leading to ineffective or incomplete speaker and accent erasure. To address this, we introduce QPAudioEraser, a quantum-inspired audio unlearning framework. Our our-phase approach involves: (1) weight initialization using destructive interference to nullify target features, (2) superposition-based label transformations that obscure class identity, (3) an uncertainty-maximizing quantum loss function, and (4) entanglement-inspired mixing of correlated weights to retain model knowledge. Comprehensive evaluations with ResNet18, ViT, and CNN architectures across AudioMNIST, Speech Commands, LibriSpeech, and Speech Accent Archive datasets validate QPAudioEraser's superior performance. The framework achieves complete erasure of target data (0% Forget Accuracy) while incurring minimal impact on model utility, with a performance degradation on retained data as low as 0.05%. QPAudioEraser consistently surpasses conventional baselines across single-class, multi-class, sequential, and accent-level erasure scenarios, establishing the proposed approach as a robust privacy-preserving solution. 

---
# Measuring Time-Series Dataset Similarity using Wasserstein Distance 

**Authors**: Hongjie Chen, Akshay Mehra, Josh Kimball, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22189)  

**Abstract**: The emergence of time-series foundation model research elevates the growing need to measure the (dis)similarity of time-series datasets. A time-series dataset similarity measure aids research in multiple ways, including model selection, finetuning, and visualization. In this paper, we propose a distribution-based method to measure time-series dataset similarity by leveraging the Wasserstein distance. We consider a time-series dataset an empirical instantiation of an underlying multivariate normal distribution (MVN). The similarity between two time-series datasets is thus computed as the Wasserstein distance between their corresponding MVNs. Comprehensive experiments and visualization show the effectiveness of our approach. Specifically, we show how the Wasserstein distance helps identify similar time-series datasets and facilitates inference performance estimation of foundation models in both out-of-distribution and transfer learning evaluation, with high correlations between our proposed measure and the inference loss (>0.60). 

---
# A Scalable Pipeline for Estimating Verb Frame Frequencies Using Large Language Models 

**Authors**: Adam M. Morgan, Adeen Flinker  

**Link**: [PDF](https://arxiv.org/pdf/2507.22187)  

**Abstract**: We present an automated pipeline for estimating Verb Frame Frequencies (VFFs), the frequency with which a verb appears in particular syntactic frames. VFFs provide a powerful window into syntax in both human and machine language systems, but existing tools for calculating them are limited in scale, accuracy, or accessibility. We use large language models (LLMs) to generate a corpus of sentences containing 476 English verbs. Next, by instructing an LLM to behave like an expert linguist, we had it analyze the syntactic structure of the sentences in this corpus. This pipeline outperforms two widely used syntactic parsers across multiple evaluation datasets. Furthermore, it requires far fewer resources than manual parsing (the gold-standard), thereby enabling rapid, scalable VFF estimation. Using the LLM parser, we produce a new VFF database with broader verb coverage, finer-grained syntactic distinctions, and explicit estimates of the relative frequencies of structural alternates commonly studied in psycholinguistics. The pipeline is easily customizable and extensible to new verbs, syntactic frames, and even other languages. We present this work as a proof of concept for automated frame frequency estimation, and release all code and data to support future research. 

---
# SourceSplice: Source Selection for Machine Learning Tasks 

**Authors**: Ambarish Singh, Romila Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22186)  

**Abstract**: Data quality plays a pivotal role in the predictive performance of machine learning (ML) tasks - a challenge amplified by the deluge of data sources available in modern this http URL work in data discovery largely focus on metadata matching, semantic similarity or identifying tables that should be joined to answer a particular query, but do not consider source quality for high performance of the downstream ML this http URL paper addresses the problem of determining the best subset of data sources that must be combined to construct the underlying training dataset for a given ML this http URL propose SourceGrasp and SourceSplice, frameworks designed to efficiently select a suitable subset of sources that maximizes the utility of the downstream ML this http URL the algorithms rely on the core idea that sources (or their combinations) contribute differently to the task utility, and must be judiciously this http URL SourceGrasp utilizes a metaheuristic based on a greediness criterion and randomization, the SourceSplice framework presents a source selection mechanism inspired from gene splicing - a core concept used in protein this http URL empirically evaluate our algorithms on three real-world datasets and synthetic datasets and show that, with significantly fewer subset explorations, SourceSplice effectively identifies subsets of data sources leading to high task this http URL also conduct studies reporting the sensitivity of SourceSplice to the decision choices under several settings. 

---
# Spatial-Temporal Reinforcement Learning for Network Routing with Non-Markovian Traffic 

**Authors**: Molly Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22174)  

**Abstract**: Reinforcement Learning (RL) has become a well-established approach for optimizing packet routing in communication networks. Standard RL algorithms typically are based on the Markov Decision Process (MDP), which assumes that the current state of the environment provides all the necessary information for system evolution and decision-making. However, this Markovian assumption is invalid in many practical scenarios, making the MDP and RL frameworks inadequate to produce the optimal solutions. Additionally, traditional RL algorithms often employ function approximations (e.g., by neural networks) that do not explicitly capture the spatial relationships inherent in environments with complex network topologies. Communication networks are characterized by dynamic traffic patterns and arbitrary numbers of nodes and links, which further complicate the decision-making process. To address these challenges, we propose a spatial-temporal RL approach that integrates Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) to adequately capture the spatial dynamics regarding network topology and temporal traffic patterns, respectively, to enhance routing decisions. Our evaluation demonstrates that the proposed method outperforms and is more robust to changes in the network topology when compared with traditional RL techniques. 

---
# Enhancing Jailbreak Attacks on LLMs via Persona Prompts 

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22171)  

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at this https URL. 

---
# Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles 

**Authors**: Kimberly Le Truong, Riccardo Fogliato, Hoda Heidari, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22168)  

**Abstract**: Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations. 

---
# Strategic Deflection: Defending LLMs from Logit Manipulation 

**Authors**: Yassine Rachidy, Jihad Rbaiti, Youssef Hmamouche, Faissal Sehbaoui, Amal El Fallah Seghrouchni  

**Link**: [PDF](https://arxiv.org/pdf/2507.22160)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats. 

---
# IndoPref: A Multi-Domain Pairwise Preference Dataset for Indonesian 

**Authors**: Vanessa Rebecca Wiyono, David Anugraha, Ayu Purwarianti, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2507.22159)  

**Abstract**: Over 200 million people speak Indonesian, yet the language remains significantly underrepresented in preference-based research for large language models (LLMs). Most existing multilingual datasets are derived from English translations, often resulting in content that lacks cultural and linguistic authenticity. To address this gap, we introduce IndoPref, the first fully human-authored and multi-domain Indonesian preference dataset specifically designed to evaluate the naturalness and quality of LLM-generated text. All annotations are natively written in Indonesian and evaluated using Krippendorff's alpha, demonstrating strong inter-annotator agreement. Additionally, we benchmark the dataset across multiple LLMs and assess the output quality of each model. 

---
# Tiny Noise-Robust Voice Activity Detector for Voice Assistants 

**Authors**: Hamed Jafarzadeh Asl, Mahsa Ghazvini Nejad, Amin Edraki, Masoud Asgharian, Vahid Partovi Nia  

**Link**: [PDF](https://arxiv.org/pdf/2507.22157)  

**Abstract**: Voice Activity Detection (VAD) in the presence of background noise remains a challenging problem in speech processing. Accurate VAD is essential in automatic speech recognition, voice-to-text, conversational agents, etc, where noise can severely degrade the performance. A modern application includes the voice assistant, specially mounted on Artificial Intelligence of Things (AIoT) devices such as cell phones, smart glasses, earbuds, etc, where the voice signal includes background noise. Therefore, VAD modules must remain light-weight due to their practical on-device limitation. The existing models often struggle with low signal-to-noise ratios across diverse acoustic environments. A simple VAD often detects human voice in a clean environment, but struggles to detect the human voice in noisy conditions. We propose a noise-robust VAD that comprises a light-weight VAD, with data pre-processing and post-processing added modules to handle the background noise. This approach significantly enhances the VAD accuracy in noisy environments and requires neither a larger model, nor fine-tuning. Experimental results demonstrate that our approach achieves a notable improvement compared to baselines, particularly in environments with high background noise interference. This modified VAD additionally improving clean speech detection. 

---
# Runtime Failure Hunting for Physics Engine Based Software Systems: How Far Can We Go? 

**Authors**: Shuqing Li, Qiang Chen, Xiaoxue Ren, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22099)  

**Abstract**: Physics Engines (PEs) are fundamental software frameworks that simulate physical interactions in applications ranging from entertainment to safety-critical systems. Despite their importance, PEs suffer from physics failures, deviations from expected physical behaviors that can compromise software reliability, degrade user experience, and potentially cause critical failures in autonomous vehicles or medical robotics. Current testing approaches for PE-based software are inadequate, typically requiring white-box access and focusing on crash detection rather than semantically complex physics failures. This paper presents the first large-scale empirical study characterizing physics failures in PE-based software. We investigate three research questions addressing the manifestations of physics failures, the effectiveness of detection techniques, and developer perceptions of current detection practices. Our contributions include: (1) a taxonomy of physics failure manifestations; (2) a comprehensive evaluation of detection methods including deep learning, prompt-based techniques, and large multimodal models; and (3) actionable insights from developer experiences for improving detection approaches. To support future research, we release PhysiXFails, code, and other materials at this https URL. 

---
# Scaling and Distilling Transformer Models for sEMG 

**Authors**: Nicholas Mehlman, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Kelvin Niu, Alexander H. Miller, Shagun Sodhani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22094)  

**Abstract**: Surface electromyography (sEMG) signals offer a promising avenue for developing innovative human-computer interfaces by providing insights into muscular activity. However, the limited volume of training data and computational constraints during deployment have restricted the investigation of scaling up the model size for solving sEMG tasks. In this paper, we demonstrate that vanilla transformer models can be effectively scaled up on sEMG data and yield improved cross-user performance up to 110M parameters, surpassing the model size regime investigated in other sEMG research (usually <10M parameters). We show that >100M-parameter models can be effectively distilled into models 50x smaller with minimal loss of performance (<1.5% absolute). This results in efficient and expressive models suitable for complex real-time sEMG tasks in real-world environments. 

---
# Pathology Foundation Models are Scanner Sensitive: Benchmark and Mitigation with Contrastive ScanGen Loss 

**Authors**: Gianluca Carloni, Biagio Brattoli, Seongho Keum, Jongchan Park, Taebum Lee, Chang Ho Ahn, Sergio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2507.22092)  

**Abstract**: Computational pathology (CPath) has shown great potential in mining actionable insights from Whole Slide Images (WSIs). Deep Learning (DL) has been at the center of modern CPath, and while it delivers unprecedented performance, it is also known that DL may be affected by irrelevant details, such as those introduced during scanning by different commercially available scanners. This may lead to scanner bias, where the model outputs for the same tissue acquired by different scanners may vary. In turn, it hinders the trust of clinicians in CPath-based tools and their deployment in real-world clinical practices. Recent pathology Foundation Models (FMs) promise to provide better domain generalization capabilities. In this paper, we benchmark FMs using a multi-scanner dataset and show that FMs still suffer from scanner bias. Following this observation, we propose ScanGen, a contrastive loss function applied during task-specific fine-tuning that mitigates scanner bias, thereby enhancing the models' robustness to scanner variations. Our approach is applied to the Multiple Instance Learning task of Epidermal Growth Factor Receptor (EGFR) mutation prediction from H\&E-stained WSIs in lung cancer. We observe that ScanGen notably enhances the ability to generalize across scanners, while retaining or improving the performance of EGFR mutation prediction. 

---
# Hybrid activation functions for deep neural networks: S3 and S4 -- a novel approach to gradient flow optimization 

**Authors**: Sergii Kavun  

**Link**: [PDF](https://arxiv.org/pdf/2507.22090)  

**Abstract**: Activation functions are critical components in deep neural networks, directly influencing gradient flow, training stability, and model performance. Traditional functions like ReLU suffer from dead neuron problems, while sigmoid and tanh exhibit vanishing gradient issues. We introduce two novel hybrid activation functions: S3 (Sigmoid-Softsign) and its improved version S4 (smoothed S3). S3 combines sigmoid for negative inputs with softsign for positive inputs, while S4 employs a smooth transition mechanism controlled by a steepness parameter k. We conducted comprehensive experiments across binary classification, multi-class classification, and regression tasks using three different neural network architectures. S4 demonstrated superior performance compared to nine baseline activation functions, achieving 97.4% accuracy on MNIST, 96.0% on Iris classification, and 18.7 MSE on Boston Housing regression. The function exhibited faster convergence (-19 for ReLU) and maintained stable gradient flow across network depths. Comparative analysis revealed S4's gradient range of [0.24, 0.59] compared to ReLU's 18% dead neurons in deep networks. The S4 activation function addresses key limitations of existing functions through its hybrid design and smooth transition mechanism. The tunable parameter k allows adaptation to different tasks and network depths, making S4 a versatile choice for deep learning applications. These findings suggest that hybrid activation functions represent a promising direction for improving neural network training dynamics. 

---
# Principled Curriculum Learning using Parameter Continuation Methods 

**Authors**: Harsh Nilesh Pathak, Randy Paffenroth  

**Link**: [PDF](https://arxiv.org/pdf/2507.22089)  

**Abstract**: In this work, we propose a parameter continuation method for the optimization of neural networks. There is a close connection between parameter continuation, homotopies, and curriculum learning. The methods we propose here are theoretically justified and practically effective for several problems in deep neural networks. In particular, we demonstrate better generalization performance than state-of-the-art optimization techniques such as ADAM for supervised and unsupervised learning tasks. 

---
# TypyBench: Evaluating LLM Type Inference for Untyped Python Repositories 

**Authors**: Honghua Dong, Jiacheng Yang, Xun Deng, Yuhe Jiang, Gennady Pekhimenko, Fan Long, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2507.22086)  

**Abstract**: Type inference for dynamic languages like Python is a persistent challenge in software engineering. While large language models (LLMs) have shown promise in code understanding, their type inference capabilities remain underexplored. We introduce TypyBench, a benchmark designed to evaluate LLMs' type inference across entire Python repositories. TypyBench features two novel metrics: TypeSim, which captures nuanced semantic relationships between predicted and ground truth types, and TypeCheck, which assesses type consistency across codebases. Our evaluation of various LLMs on a curated dataset of 50 high-quality Python repositories reveals that, although LLMs achieve decent TypeSim scores, they struggle with complex nested types and exhibit significant type consistency errors. These findings suggest that future research should shift focus from improving type similarity to addressing repository-level consistency. TypyBench provides a foundation for this new direction, offering insights into model performance across different type complexities and usage contexts. Our code and data are available at this https URL. 

---
# Shape Invariant 3D-Variational Autoencoder: Super Resolution in Turbulence flow 

**Authors**: Anuraj Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2507.22082)  

**Abstract**: Deep learning provides a versatile suite of methods for extracting structured information from complex datasets, enabling deeper understanding of underlying fluid dynamic phenomena. The field of turbulence modeling, in particular, benefits from the growing availability of high-dimensional data obtained through experiments, field observations, and large-scale simulations spanning multiple spatio-temporal scales. This report presents a concise overview of both classical and deep learningbased approaches to turbulence modeling. It further investigates two specific challenges at the intersection of fluid dynamics and machine learning: the integration of multiscale turbulence models with deep learning architectures, and the application of deep generative models for super-resolution reconstruction 

---
# CodeEvo: Interaction-Driven Synthesis of Code-centric Data through Hybrid and Iterative Feedback 

**Authors**: Qiushi Sun, Jinyang Gong, Lei Li, Qipeng Guo, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22080)  

**Abstract**: Acquiring high-quality instruction-code pairs is essential for training Large Language Models (LLMs) for code generation. Manually curated data is expensive and inherently limited in scale, motivating the development of code-centric synthesis methods. Yet, current approaches either focus on augmenting existing code or rely on predefined heuristics, both lacking rigorous data validation, which results in synthetic data that is ungrounded, repetitive, or overly simplistic. Inspired by collaborative programming practices, we propose CodeEvo, a framework that synthesizes code data through iterative interactions between two LLM agents: a Coder, which generates candidate code and test cases based on given instructions, and a Reviewer, which guides the synthesis process by producing new instructions and feedback. We further introduce a hybrid feedback mechanism that combines compiler determinism with the generative flexibility of agents, enabling automatic quality control throughout synthesis. Extensive experiments demonstrate that models fine-tuned on CodeEvo data significantly outperform established baselines across code generation benchmarks with various difficulties. In-depth analyses further provide insights from multiple perspectives into effective code-centric data synthesis. 

---
# From Cloud-Native to Trust-Native: A Protocol for Verifiable Multi-Agent Systems 

**Authors**: Muyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22077)  

**Abstract**: As autonomous agents powered by large language models (LLMs) proliferate in high-stakes domains -- from pharmaceuticals to legal workflows -- the challenge is no longer just intelligence, but verifiability. We introduce TrustTrack, a protocol that embeds structural guarantees -- verifiable identity, policy commitments, and tamper-resistant behavioral logs -- directly into agent infrastructure. This enables a new systems paradigm: trust-native autonomy. By treating compliance as a design constraint rather than post-hoc oversight, TrustTrack reframes how intelligent agents operate across organizations and jurisdictions. We present the protocol design, system requirements, and use cases in regulated domains such as pharmaceutical R&D, legal automation, and AI-native collaboration. We argue that the Cloud -> AI -> Agent -> Trust transition represents the next architectural layer for autonomous systems. 

---
# A Compute-Matched Re-Evaluation of TroVE on MATH 

**Authors**: Tobias Sesterhenn, Ian Berlot-Attwell, Janis Zenkner, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2507.22069)  

**Abstract**: Reusing established theorems and formulas is central to mathematical problem solving, serving as essential building blocks for tackling increasingly complex challenges. Recent work, TroVE, argues that code-generating Large Language Models (LLMs) can benefit similarly on the MATH benchmark by inducing and reusing higher-level toolboxes. By allocating computational budget across an ensemble of three modes -- directly generating code, creating tools, and reusing tools -- TroVE claims to outperform a PRIMITIVE baseline that only performs direct generation. However, recent analysis (Berlot-Attwell et al., 2024) casts doubt on these gains, noting that the tools created are often trivial or rarely reused, suggesting that improvements may stem from self-consistency or self-correction. In this work, we re-evaluate TroVE on MATH, analyze the impact of each of its modes, and show that its benefit does not come from these mechanisms, but simply from a higher computational budget spent for TroVE compared to PRIMITIVE. To this end, we also perform a small correction in the original implementation of TroVE's selection mechanism, boosting TroVE's performance on MATH by 3\% in accuracy. After matching for compute, the benefit of TroVE reduces to a marginal improvement of 1\%, suggesting that this toolbox approach does not provide a significant benefit on MATH. 

---
# Dimensions of Vulnerability in Visual Working Memory: An AI-Driven Approach to Perceptual Comparison 

**Authors**: Yuang Cao, Jiachen Zou, Chen Wei, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22067)  

**Abstract**: Human memory exhibits significant vulnerability in cognitive tasks and daily life. Comparisons between visual working memory and new perceptual input (e.g., during cognitive tasks) can lead to unintended memory distortions. Previous studies have reported systematic memory distortions after perceptual comparison, but understanding how perceptual comparison affects memory distortions in real-world objects remains a challenge. Furthermore, identifying what visual features contribute to memory vulnerability presents a novel research question. Here, we propose a novel AI-driven framework that generates naturalistic visual stimuli grounded in behaviorally relevant object dimensions to elicit similarity-induced memory biases. We use two types of stimuli -- image wheels created through dimension editing and dimension wheels generated by dimension activation values -- in three visual working memory (VWM) experiments. These experiments assess memory distortions under three conditions: no perceptual comparison, perceptual comparison with image wheels, and perceptual comparison with dimension wheels. The results show that similar dimensions, like similar images, can also induce memory distortions. Specifically, visual dimensions are more prone to distortion than semantic dimensions, indicating that the object dimensions of naturalistic visual stimuli play a significant role in the vulnerability of memory. 

---
# Fuzzing: Randomness? Reasoning! Efficient Directed Fuzzing via Large Language Models 

**Authors**: Xiaotao Feng, Xiaogang Zhu, Kun Hu, Jincheng Wang, Yingjie Cao, Guang Gong, Jianfeng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22065)  

**Abstract**: Fuzzing is highly effective in detecting bugs due to the key contribution of randomness. However, randomness significantly reduces the efficiency of fuzzing, causing it to cost days or weeks to expose bugs. Even though directed fuzzing reduces randomness by guiding fuzzing towards target buggy locations, the dilemma of randomness still challenges directed fuzzers. Two critical components, which are seeds and mutators, contain randomness and are closely tied to the conditions required for triggering bugs. Therefore, to address the challenge of randomness, we propose to use large language models (LLMs) to remove the randomness in seeds and reduce the randomness in mutators. With their strong reasoning and code generation capabilities, LLMs can be used to generate reachable seeds that target pre-determined locations and to construct bug-specific mutators tailored for specific bugs. We propose RandLuzz, which integrates LLMs and directed fuzzing, to improve the quality of seeds and mutators, resulting in efficient bug exposure. RandLuzz analyzes function call chain or functionality to guide LLMs in generating reachable seeds. To construct bug-specific mutators, RandLuzz uses LLMs to perform bug analysis, obtaining information such as bug causes and mutation suggestions, which further help generate code that performs bug-specific mutations. We evaluate RandLuzz by comparing it with four state-of-the-art directed fuzzers, AFLGo, Beacon, WindRanger, and SelectFuzz. With RandLuzz-generated seeds, the fuzzers achieve an average speedup ranging from 2.1$\times$ to 4.8$\times$ compared to using widely-used initial seeds. Additionally, when evaluated on individual bugs, RandLuzz achieves up to a 2.7$\times$ speedup compared to the second-fastest exposure. On 8 bugs, RandLuzz can even expose them within 60 seconds. 

---
# Machine Learning Experiences: A story of learning AI for use in enterprise software testing that can be used by anyone 

**Authors**: Michael Cohoon, Debbie Furman  

**Link**: [PDF](https://arxiv.org/pdf/2507.22064)  

**Abstract**: This paper details the machine learning (ML) journey of a group of people focused on software testing. It tells the story of how this group progressed through a ML workflow (similar to the CRISP-DM process). This workflow consists of the following steps and can be used by anyone applying ML techniques to a project: gather the data; clean the data; perform feature engineering on the data; splitting the data into two sets, one for training and one for testing; choosing a machine learning model; training the model; testing the model and evaluating the model performance. By following this workflow, anyone can effectively apply ML to any project that they are doing. 

---
# RedCoder: Automated Multi-Turn Red Teaming for Code LLMs 

**Authors**: Wenjie Jacky Mo, Qin Liu, Xiaofei Wen, Dongwon Jung, Hadi Askari, Wenxuan Zhou, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22063)  

**Abstract**: Large Language Models (LLMs) for code generation (i.e., Code LLMs) have demonstrated impressive capabilities in AI-assisted software development and testing. However, recent studies have shown that these models are prone to generating vulnerable or even malicious code under adversarial settings. Existing red-teaming approaches rely on extensive human effort, limiting their scalability and practicality, and generally overlook the interactive nature of real-world AI-assisted programming, which often unfolds over multiple turns. To bridge these gaps, we present RedCoder, a red-teaming agent that engages victim models in multi-turn conversation to elicit vulnerable code. The pipeline to construct RedCoder begins with a multi-agent gaming process that simulates adversarial interactions, yielding a set of prototype conversations and an arsenal of reusable attack strategies. We then fine-tune an LLM on these prototype conversations to serve as the backbone of RedCoder. Once deployed, RedCoder autonomously engages Code LLMs in multi-turn conversations, dynamically retrieving relevant strategies from the arsenal to steer the dialogue toward vulnerability-inducing outputs. Experiments across multiple Code LLMs show that our approach outperforms prior single-turn and multi-turn red-team methods in inducing vulnerabilities in code generation, offering a scalable and effective tool for evaluating the security boundaries of modern code-generation systems. 

---
# GABRIL: Gaze-Based Regularization for Mitigating Causal Confusion in Imitation Learning 

**Authors**: Amin Banayeeanzade, Fatemeh Bahrani, Yutai Zhou, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2507.19647)  

**Abstract**: Imitation Learning (IL) is a widely adopted approach which enables agents to learn from human expert demonstrations by framing the task as a supervised learning problem. However, IL often suffers from causal confusion, where agents misinterpret spurious correlations as causal relationships, leading to poor performance in testing environments with distribution shift. To address this issue, we introduce GAze-Based Regularization in Imitation Learning (GABRIL), a novel method that leverages the human gaze data gathered during the data collection phase to guide the representation learning in IL. GABRIL utilizes a regularization loss which encourages the model to focus on causally relevant features identified through expert gaze and consequently mitigates the effects of confounding variables. We validate our approach in Atari environments and the Bench2Drive benchmark in CARLA by collecting human gaze datasets and applying our method in both domains. Experimental results show that the improvement of GABRIL over behavior cloning is around 179% more than the same number for other baselines in the Atari and 76% in the CARLA setup. Finally, we show that our method provides extra explainability when compared to regular IL agents. 

---
# RecPS: Privacy Risk Scoring for Recommender Systems 

**Authors**: Jiajie He, Yuechun Gu, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18365)  

**Abstract**: Recommender systems (RecSys) have become an essential component of many web applications. The core of the system is a recommendation model trained on highly sensitive user-item interaction data. While privacy-enhancing techniques are actively studied in the research community, the real-world model development still depends on minimal privacy protection, e.g., via controlled access. Users of such systems should have the right to choose \emph{not} to share highly sensitive interactions. However, there is no method allowing the user to know which interactions are more sensitive than others. Thus, quantifying the privacy risk of RecSys training data is a critical step to enabling privacy-aware RecSys model development and deployment. We propose a membership-inference attack (MIA)- based privacy scoring method, RecPS, to measure privacy risks at both the interaction and user levels. The RecPS interaction-level score definition is motivated and derived from differential privacy, which is then extended to the user-level scoring method. A critical component is the interaction-level MIA method RecLiRA, which gives high-quality membership estimation. We have conducted extensive experiments on well-known benchmark datasets and RecSys models to show the unique features and benefits of RecPS scoring in risk assessment and RecSys model unlearning. 

---
# Spatial-Temporal Data Mining for Ocean Science: Data, Methodologies, and Opportunities 

**Authors**: Hanchen Yang, Wengen Li, Shuyu Wang, Hui Li, Jihong Guan, Shuigeng Zhou, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2307.10803)  

**Abstract**: With the rapid amassing of spatial-temporal (ST) ocean data, many spatial-temporal data mining (STDM) studies have been conducted to address various oceanic issues, including climate forecasting and disaster warning. Compared with typical ST data (e.g., traffic data), ST ocean data is more complicated but with unique characteristics, e.g., diverse regionality and high sparsity. These characteristics make it difficult to design and train STDM models on ST ocean data. To the best of our knowledge, a comprehensive survey of existing studies remains missing in the literature, which hinders not only computer scientists from identifying the research issues in ocean data mining but also ocean scientists to apply advanced STDM techniques. In this paper, we provide a comprehensive survey of existing STDM studies for ocean science. Concretely, we first review the widely-used ST ocean datasets and highlight their unique characteristics. Then, typical ST ocean data quality enhancement techniques are explored. Next, we classify existing STDM studies in ocean science into four types of tasks, i.e., prediction, event detection, pattern mining, and anomaly detection, and elaborate on the techniques for these tasks. Finally, promising research opportunities are discussed. This survey can help scientists from both computer science and ocean science better understand the fundamental concepts, key techniques, and open challenges of STDM for ocean science. 

---
