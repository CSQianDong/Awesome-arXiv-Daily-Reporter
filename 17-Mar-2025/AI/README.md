# Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs using Semantic Space 

**Authors**: Zhiliang Chen, Xinyuan Niu, Chuan-Sheng Foo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2503.11586)  

**Abstract**: Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: this https URL. 

---
# Prompt Injection Detection and Mitigation via AI Multi-Agent NLP Frameworks 

**Authors**: Diego Gosmar, Deborah A. Dahl, Dario Gosmar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11517)  

**Abstract**: Prompt injection constitutes a significant challenge for generative AI systems by inducing unintended outputs. We introduce a multi-agent NLP framework specifically designed to address prompt injection vulnerabilities through layered detection and enforcement mechanisms. The framework orchestrates specialized agents for generating responses, sanitizing outputs, and enforcing policy compliance. Evaluation on 500 engineered injection prompts demonstrates a marked reduction in injection success and policy breaches. Novel metrics, including Injection Success Rate (ISR), Policy Override Frequency (POF), Prompt Sanitization Rate (PSR), and Compliance Consistency Score (CCS), are proposed to derive a composite Total Injection Vulnerability Score (TIVS). The system utilizes the OVON (Open Voice Network) framework for inter-agent communication via structured JSON messages, extending a previously established multi-agent architecture from hallucination mitigation to address the unique challenges of prompt injection. 

---
# Heterogeneous Causal Discovery of Repeated Undesirable Health Outcomes 

**Authors**: Shishir Adhikari, Guido Muscioni, Mark Shapiro, Plamen Petrov, Elena Zheleva  

**Link**: [PDF](https://arxiv.org/pdf/2503.11477)  

**Abstract**: Understanding factors triggering or preventing undesirable health outcomes across patient subpopulations is essential for designing targeted interventions. While randomized controlled trials and expert-led patient interviews are standard methods for identifying these factors, they can be time-consuming and infeasible. Causal discovery offers an alternative to conventional approaches by generating cause-and-effect hypotheses from observational data. However, it often relies on strong or untestable assumptions, which can limit its practical application. This work aims to make causal discovery more practical by considering multiple assumptions and identifying heterogeneous effects. We formulate the problem of discovering causes and effect modifiers of an outcome, where effect modifiers are contexts (e.g., age groups) with heterogeneous causal effects. Then, we present a novel, end-to-end framework that incorporates an ensemble of causal discovery algorithms and estimation of heterogeneous effects to discover causes and effect modifiers that trigger or inhibit the outcome. We demonstrate that the ensemble approach improves robustness by enhancing recall of causal factors while maintaining precision. Our study examines the causes of repeat emergency room visits for diabetic patients and hospital readmissions for ICU patients. Our framework generates causal hypotheses consistent with existing literature and can help practitioners identify potential interventions and patient subpopulations to focus on. 

---
# Integrating LLMs in Gamified Systems 

**Authors**: Carlos J. Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.11458)  

**Abstract**: In this work, a thorough mathematical framework for incorporating Large Language Models (LLMs) into gamified systems is presented with an emphasis on improving task dynamics, user engagement, and reward systems. Personalized feedback, adaptive learning, and dynamic content creation are all made possible by integrating LLMs and are crucial for improving user engagement and system performance. A simulated environment tests the framework's adaptability and demonstrates its potential for real-world applications in various industries, including business, healthcare, and education. The findings demonstrate how LLMs can offer customized experiences that raise system effectiveness and user retention. This study also examines the difficulties this framework aims to solve, highlighting its importance in maximizing involvement and encouraging sustained behavioral change in a range of sectors. 

---
# Preference Elicitation for Multi-objective Combinatorial Optimization with Active Learning and Maximum Likelihood Estimation 

**Authors**: Marianne Defresne, Jayanta Mandi, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2503.11435)  

**Abstract**: Real-life combinatorial optimization problems often involve several conflicting objectives, such as price, product quality and sustainability. A computationally-efficient way to tackle multiple objectives is to aggregate them into a single-objective function, such as a linear combination. However, defining the weights of the linear combination upfront is hard; alternatively, the use of interactive learning methods that ask users to compare candidate solutions is highly promising. The key challenges are to generate candidates quickly, to learn an objective function that leads to high-quality solutions and to do so with few user interactions. We build upon the Constructive Preference Elicitation framework and show how each of the three properties can be improved: to increase the interaction speed we investigate using pools of (relaxed) solutions, to improve the learning we adopt Maximum Likelihood Estimation of a Bradley-Terry preference model; and to reduce the number of user interactions, we select the pair of candidates to compare with an ensemble-based acquisition function inspired from Active Learning. Our careful experimentation demonstrates each of these improvements: on a PC configuration task and a realistic multi-instance routing problem, our method selects queries faster, needs fewer queries and synthesizes higher-quality combinatorial solutions than previous CPE methods. 

---
# Optimizing Large Language Models for Detecting Symptoms of Comorbid Depression or Anxiety in Chronic Diseases: Insights from Patient Messages 

**Authors**: Jiyeong Kim, Stephen P. Ma, Michael L. Chen, Isaac R. Galatzer-Levy, John Torous, Peter J. van Roessel, Christopher Sharp, Michael A. Pfeffer, Carolyn I. Rodriguez, Eleni Linos, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11384)  

**Abstract**: Patients with diabetes are at increased risk of comorbid depression or anxiety, complicating their management. This study evaluated the performance of large language models (LLMs) in detecting these symptoms from secure patient messages. We applied multiple approaches, including engineered prompts, systemic persona, temperature adjustments, and zero-shot and few-shot learning, to identify the best-performing model and enhance performance. Three out of five LLMs demonstrated excellent performance (over 90% of F-1 and accuracy), with Llama 3.1 405B achieving 93% in both F-1 and accuracy using a zero-shot approach. While LLMs showed promise in binary classification and handling complex metrics like Patient Health Questionnaire-4, inconsistencies in challenging cases warrant further real-life assessment. The findings highlight the potential of LLMs to assist in timely screening and referrals, providing valuable empirical knowledge for real-world triage systems that could improve mental health care for patients with chronic diseases. 

---
# Collaboration is all you need: LLM Assisted Safe Code Translation 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11237)  

**Abstract**: This paper introduces UniTranslator, a visionary framework that re-imagines code translation as a collaborative endeavor among multiple, compact LLMs. By orchestrating the interaction of specialized agents, each focused on different aspects of the translation process and grounded in a deep understanding of programming concepts, UniTranslator achieves a level of accuracy and efficiency that rivals larger, monolithic models. Our preliminary evaluation demonstrates the potential of UniTranslator to overcome the limitations of existing approaches and unlock the power of smaller LLMs for complex code translation tasks. We explore the effectiveness of this dynamic multi-agent paradigm in handling diverse language pairs, including low-resource languages, and in mitigating common issues such as code artifacts and hallucinations through the use of Natural Language Inference (NLI) grounding and iterative feedback mechanisms 

---
# GKG-LLM: A Unified Framework for Generalized Knowledge Graph Construction 

**Authors**: Jian Zhang, Bifan Wei, Shihao Qi, haiping Zhu, Jun Liu, Qika Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.11227)  

**Abstract**: The construction of Generalized Knowledge Graph (GKG), including knowledge graph, event knowledge graph and commonsense knowledge graph, is fundamental for various natural language processing tasks. Current studies typically construct these types of graph separately, overlooking holistic insights and potential unification that could be beneficial in computing resources and usage perspectives. However, a key challenge in developing a unified framework for GKG is obstacles arising from task-specific differences. In this study, we propose a unified framework for constructing generalized knowledge graphs to address this challenge. First, we collect data from 15 sub-tasks in 29 datasets across the three types of graphs, categorizing them into in-sample, counter-task, and out-of-distribution (OOD) data. Then, we propose a three-stage curriculum learning fine-tuning framework, by iteratively injecting knowledge from the three types of graphs into the Large Language Models. Extensive experiments show that our proposed model improves the construction of all three graph types across in-domain, OOD and counter-task data. 

---
# Can Large Reasoning Models do Analogical Reasoning under Perceptual Uncertainty? 

**Authors**: Giacomo Camposampiero, Michael Hersche, Roger Wattenhofer, Abu Sebastian, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11207)  

**Abstract**: This work presents a first evaluation of two state-of-the-art Large Reasoning Models (LRMs), OpenAI's o3-mini and DeepSeek R1, on analogical reasoning, focusing on well-established nonverbal human IQ tests based on Raven's progressive matrices. We benchmark with the I-RAVEN dataset and its more difficult extension, I-RAVEN-X, which tests the ability to generalize to longer reasoning rules and ranges of the attribute values. To assess the influence of visual uncertainties on these nonverbal analogical reasoning tests, we extend the I-RAVEN-X dataset, which otherwise assumes an oracle perception. We adopt a two-fold strategy to simulate this imperfect visual perception: 1) we introduce confounding attributes which, being sampled at random, do not contribute to the prediction of the correct answer of the puzzles and 2) smoothen the distributions of the input attributes' values. We observe a sharp decline in OpenAI's o3-mini task accuracy, dropping from 86.6% on the original I-RAVEN to just 17.0% -- approaching random chance -- on the more challenging I-RAVEN-X, which increases input length and range and emulates perceptual uncertainty. This drop occurred despite spending 3.4x more reasoning tokens. A similar trend is also observed for DeepSeek R1: from 80.6% to 23.2%. On the other hand, a neuro-symbolic probabilistic abductive model, ARLC, that achieves state-of-the-art performances on I-RAVEN, can robustly reason under all these out-of-distribution tests, maintaining strong accuracy with only a modest reduction from 98.6% to 88.0%. Our code is available at this https URL. 

---
# Large Reasoning Models in Agent Scenarios: Exploring the Necessity of Reasoning Capabilities 

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Weidong Wang, Zhigang Zuo, Di Wu, Duanfeng Chu, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11074)  

**Abstract**: The rise of Large Reasoning Models (LRMs) signifies a paradigm shift toward advanced computational reasoning. Yet, this progress disrupts traditional agent frameworks, traditionally anchored by execution-oriented Large Language Models (LLMs). To explore this transformation, we propose the LaRMA framework, encompassing nine tasks across Tool Usage, Plan Design, and Problem Solving, assessed with three top LLMs (e.g., Claude3.5-sonnet) and five leading LRMs (e.g., DeepSeek-R1). Our findings address four research questions: LRMs surpass LLMs in reasoning-intensive tasks like Plan Design, leveraging iterative reflection for superior outcomes; LLMs excel in execution-driven tasks such as Tool Usage, prioritizing efficiency; hybrid LLM-LRM configurations, pairing LLMs as actors with LRMs as reflectors, optimize agent performance by blending execution speed with reasoning depth; and LRMs' enhanced reasoning incurs higher computational costs, prolonged processing, and behavioral challenges, including overthinking and fact-ignoring tendencies. This study fosters deeper inquiry into LRMs' balance of deep thinking and overthinking, laying a critical foundation for future agent design advancements. 

---
# API Agents vs. GUI Agents: Divergence and Convergence 

**Authors**: Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11069)  

**Abstract**: Large language models (LLMs) have evolved beyond simple text generation to power software agents that directly translate natural language commands into tangible actions. While API-based LLM agents initially rose to prominence for their robust automation capabilities and seamless integration with programmatic endpoints, recent progress in multimodal LLM research has enabled GUI-based LLM agents that interact with graphical user interfaces in a human-like manner. Although these two paradigms share the goal of enabling LLM-driven task automation, they diverge significantly in architectural complexity, development workflows, and user interaction models.
This paper presents the first comprehensive comparative study of API-based and GUI-based LLM agents, systematically analyzing their divergence and potential convergence. We examine key dimensions and highlight scenarios in which hybrid approaches can harness their complementary strengths. By proposing clear decision criteria and illustrating practical use cases, we aim to guide practitioners and researchers in selecting, combining, or transitioning between these paradigms. Ultimately, we indicate that continuing innovations in LLM-based automation are poised to blur the lines between API- and GUI-driven agents, paving the way for more flexible, adaptive solutions in a wide range of real-world applications. 

---
# Resource Constrained Pathfinding with A* and Negative Weights 

**Authors**: Saman Ahmadi, Andrea Raith, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2503.11037)  

**Abstract**: Constrained pathfinding is a well-studied, yet challenging network optimisation problem that can be seen in a broad range of real-world applications. Pathfinding with multiple resource limits, which is known as the Resource Constrained Shortest Path Problem (RCSP), aims to plan a cost-optimum path subject to limited usage of resources. Given the recent advances in constrained and multi-criteria search with A*, this paper introduces a new resource constrained search framework on the basis of A* to tackle RCSP in large networks, even in the presence of negative cost and negative resources. We empirically evaluate our new algorithm on a set of large instances and show up to two orders of magnitude faster performance compared to state-of-the-art RCSP algorithms in the literature. 

---
# TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools 

**Authors**: Shanghua Gao, Richard Zhu, Zhenglun Kong, Ayush Noori, Xiaorui Su, Curtis Ginder, Theodoros Tsiligkaridis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.10970)  

**Abstract**: Precision therapeutics require multimodal adaptive models that generate personalized treatment recommendations. We introduce TxAgent, an AI agent that leverages multi-step reasoning and real-time biomedical knowledge retrieval across a toolbox of 211 tools to analyze drug interactions, contraindications, and patient-specific treatment strategies. TxAgent evaluates how drugs interact at molecular, pharmacokinetic, and clinical levels, identifies contraindications based on patient comorbidities and concurrent medications, and tailors treatment strategies to individual patient characteristics. It retrieves and synthesizes evidence from multiple biomedical sources, assesses interactions between drugs and patient conditions, and refines treatment recommendations through iterative reasoning. It selects tools based on task objectives and executes structured function calls to solve therapeutic tasks that require clinical reasoning and cross-source validation. The ToolUniverse consolidates 211 tools from trusted sources, including all US FDA-approved drugs since 1939 and validated clinical insights from Open Targets. TxAgent outperforms leading LLMs, tool-use models, and reasoning agents across five new benchmarks: DrugPC, BrandPC, GenericPC, TreatmentPC, and DescriptionPC, covering 3,168 drug reasoning tasks and 456 personalized treatment scenarios. It achieves 92.1% accuracy in open-ended drug reasoning tasks, surpassing GPT-4o and outperforming DeepSeek-R1 (671B) in structured multi-step reasoning. TxAgent generalizes across drug name variants and descriptions. By integrating multi-step inference, real-time knowledge grounding, and tool-assisted decision-making, TxAgent ensures that treatment recommendations align with established clinical guidelines and real-world evidence, reducing the risk of adverse events and improving therapeutic decision-making. 

---
# Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms 

**Authors**: Camilo Chacón Sartori, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10968)  

**Abstract**: Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills. 

---
# Auditing language models for hidden objectives 

**Authors**: Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus, Siddharth Mishra-Sharma, Daniel Ziegler, Emmanuel Ameisen, Joshua Batson, Tim Belonax, Samuel R. Bowman, Shan Carter, Brian Chen, Hoagy Cunningham, Carson Denison, Florian Dietz, Satvik Golechha, Akbir Khan, Jan Kirchner, Jan Leike, Austin Meek, Kei Nishimura-Gasparian, Euan Ong, Christopher Olah, Adam Pearce, Fabien Roger, Jeanne Salle, Andy Shih, Meg Tong, Drake Thomas, Kelley Rivoire, Adam Jermyn, Monte MacDiarmid, Tom Henighan, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.10965)  

**Abstract**: We study the feasibility of conducting alignment audits: investigations into whether models have undesired objectives. As a testbed, we train a language model with a hidden objective. Our training pipeline first teaches the model about exploitable errors in RLHF reward models (RMs), then trains the model to exploit some of these errors. We verify via out-of-distribution evaluations that the model generalizes to exhibit whatever behaviors it believes RMs rate highly, including ones not reinforced during training. We leverage this model to study alignment audits in two ways. First, we conduct a blind auditing game where four teams, unaware of the model's hidden objective or training, investigate it for concerning behaviors and their causes. Three teams successfully uncovered the model's hidden objective using techniques including interpretability with sparse autoencoders (SAEs), behavioral attacks, and training data analysis. Second, we conduct an unblinded follow-up study of eight techniques for auditing the model, analyzing their strengths and limitations. Overall, our work provides a concrete example of using alignment audits to discover a model's hidden objective and proposes a methodology for practicing and validating progress in alignment auditing. 

---
# Graph-Grounded LLMs: Leveraging Graphical Function Calling to Minimize LLM Hallucinations 

**Authors**: Piyush Gupta, Sangjae Bae, David Isele  

**Link**: [PDF](https://arxiv.org/pdf/2503.10941)  

**Abstract**: The adoption of Large Language Models (LLMs) is rapidly expanding across various tasks that involve inherent graphical structures. Graphs are integral to a wide range of applications, including motion planning for autonomous vehicles, social networks, scene understanding, and knowledge graphs. Many problems, even those not initially perceived as graph-based, can be effectively addressed through graph theory. However, when applied to these tasks, LLMs often encounter challenges, such as hallucinations and mathematical inaccuracies. To overcome these limitations, we propose Graph-Grounded LLMs, a system that improves LLM performance on graph-related tasks by integrating a graph library through function calls. By grounding LLMs in this manner, we demonstrate significant reductions in hallucinations and improved mathematical accuracy in solving graph-based problems, as evidenced by the performance on the NLGraph benchmark. Finally, we showcase a disaster rescue application where the Graph-Grounded LLM acts as a decision-support system. 

---
# Learning to Inference Adaptively for Multimodal Large Language Models 

**Authors**: Zhuoyan Xu, Khoi Duc Nguyen, Preeti Mukherjee, Saurabh Bagchi, Somali Chaterji, Yingyu Liang, Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10905)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in reasoning, yet come with substantial computational cost, limiting their deployment in resource-constrained settings. Despite recent efforts on improving the efficiency of MLLMs, prior solutions fall short in responding to varying runtime conditions, in particular changing resource availability (e.g., contention due to the execution of other programs on the device). To bridge this gap, we introduce AdaLLaVA, an adaptive inference framework that learns to dynamically reconfigure operations in an MLLM during inference, accounting for the input data and a latency budget. We conduct extensive experiments across benchmarks involving question-answering, reasoning, and hallucination. Our results show that AdaLLaVA effectively adheres to input latency budget, achieving varying accuracy and latency tradeoffs at runtime. Further, we demonstrate that AdaLLaVA adapts to both input latency and content, can be integrated with token selection for enhanced efficiency, and generalizes across this http URL project webpage with code release is at this https URL. 

---
# Chat-TS: Enhancing Multi-Modal Reasoning Over Time-Series and Natural Language Data 

**Authors**: Paul Quinlan, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10883)  

**Abstract**: Time-series analysis is critical for a wide range of fields such as healthcare, finance, transportation, and energy, among many others. The practical applications often involve analyzing time-series data alongside contextual information in the form of natural language to support informed decisions. However, current time-series models are limited in their ability to perform reasoning that involves both time-series and their textual content. In this work, we address this gap by introducing \textit{Chat-TS}, a large language model (LLM) based framework, designed to support reasoning over time series and textual data. Unlike traditional models, Chat-TS integrates time-series tokens into LLMs' vocabulary, enhancing its reasoning ability over both modalities without compromising the core natural language capabilities, enabling practical analysis and reasoning across modalities. To support learning and evaluation in this setup, we contribute new datasets: the \textit{TS Instruct Training Dataset} which pairs diverse time-series data with relevant text instructions and responses for instruction tuning, the \textit{TS Instruct Question and Answer (QA) Gold Dataset} which provides multiple-choice questions designed to evaluate multimodal reasoning, and a \textit{TS Instruct Quantitative Probing Set} which contains a small subset of the TS Instruct QA tasks alongside math and decision-making questions for LLM evaluation. We designed a training strategy to preserve the inherent reasoning capabilities of LLMs while augmenting them for time-series reasoning. Experiments show that Chat-TS achieves state-of-the-art performance in multi-modal reasoning tasks by maintaining strong natural language proficiency while improving time-series reasoning. ~\footnote{To ensure replicability and facilitate future research, all models, datasets, and code will be available at [\texttt{Github-URL}].} 

---
# Rotated Bitboards in FUSc# and Reinforcement Learning in Computer Chess and Beyond 

**Authors**: Johannes Buchner  

**Link**: [PDF](https://arxiv.org/pdf/2503.10822)  

**Abstract**: There exist several techniques for representing the chess board inside the computer. In the first part of this paper, the concepts of the bitboard-representation and the advantages of (rotated) bitboards in move generation are explained. In order to illustrate those ideas practice, the concrete implementation of the move-generator in FUSc# is discussed and we explain a technique how to verify the move-generator with the "perft"-command. We show that the move-generator of FUSc# works 100% correct.
The second part of this paper deals with reinforcement learning in computer chess (and beyond). We exemplify the progress that has been made in this field in the last 15-20 years by comparing the "state of the art" from 2002-2008, when FUSc# was developed, with recent innovations connected to "AlphaZero". We discuss how a "FUSc#-Zero" could be implemented and what would be necessary to reduce the number of training games necessary to achieve a good performance. This can be seen as a test case to the general prblem of improving "sample effciency" in reinforcement learning.
In the final part, we move beyond computer chess, as the importance of sample effciency extends far beyond board games into a wide range of applications where data is costly, diffcult to obtain, or time consuming to generate. We review some application of the ideas developed in AlphaZero in other domains, i.e. the "other Alphas" like AlphaFold, AlphaTensor, AlphaGeometry and AlphaProof. We also discuss future research and the potential for such methods for ecological economic planning. 

---
# Centaur: Robust End-to-End Autonomous Driving with Test-Time Training 

**Authors**: Chonghao Sima, Kashyap Chitta, Zhiding Yu, Shiyi Lan, Ping Luo, Andreas Geiger, Hongyang Li, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11650)  

**Abstract**: How can we rely on an end-to-end autonomous vehicle's complex decision-making system during deployment? One common solution is to have a ``fallback layer'' that checks the planned trajectory for rule violations and replaces it with a pre-defined safe action if necessary. Another approach involves adjusting the planner's decisions to minimize a pre-defined ``cost function'' using additional system predictions such as road layouts and detected obstacles. However, these pre-programmed rules or cost functions cannot learn and improve with new training data, often resulting in overly conservative behaviors. In this work, we propose Centaur (Cluster Entropy for Test-time trAining using Uncertainty) which updates a planner's behavior via test-time training, without relying on hand-engineered rules or cost functions. Instead, we measure and minimize the uncertainty in the planner's decisions. For this, we develop a novel uncertainty measure, called Cluster Entropy, which is simple, interpretable, and compatible with state-of-the-art planning algorithms. Using data collected at prior test-time time-steps, we perform an update to the model's parameters using a gradient that minimizes the Cluster Entropy. With only this sole gradient update prior to inference, Centaur exhibits significant improvements, ranking first on the navtest leaderboard with notable gains in safety-critical metrics such as time to collision. To provide detailed insights on a per-scenario basis, we also introduce navsafe, a challenging new benchmark, which highlights previously undiscovered failure modes of driving models. 

---
# Enhancing Deep Learning Based Structured Illumination Microscopy Reconstruction with Light Field Awareness 

**Authors**: Long-Kun Shan, Ze-Hao Wang, Tong-Tian Weng, Xiang-Dong Chen, Fang-Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.11640)  

**Abstract**: Structured illumination microscopy (SIM) is a pivotal technique for dynamic subcellular imaging in live cells. Conventional SIM reconstruction algorithms depend on accurately estimating the illumination pattern and can introduce artefacts when this estimation is imprecise. Although recent deep learning-based SIM reconstruction methods have improved speed, accuracy, and robustness, they often struggle with out-of-distribution data. To address this limitation, we propose an Awareness-of-Light-field SIM (AL-SIM) reconstruction approach that directly estimates the actual light field to correct for errors arising from data distribution shifts. Through comprehensive experiments on both simulated filament structures and live BSC1 cells, our method demonstrates a 7% reduction in the normalized root mean square error (NRMSE) and substantially lowers reconstruction artefacts. By minimizing these artefacts and improving overall accuracy, AL-SIM broadens the applicability of SIM for complex biological systems. 

---
# ASMA-Tune: Unlocking LLMs' Assembly Code Comprehension via Structural-Semantic Instruction Tuning 

**Authors**: Xinyi Wang, Jiashui Wang, Peng Chen, Jinbo Su, Yanming Liu, Long Liu, Yangdong Wang, Qiyuan Chen, Kai Yun, Chunfu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.11617)  

**Abstract**: Analysis and comprehension of assembly code are crucial in various applications, such as reverse engineering. However, the low information density and lack of explicit syntactic structures in assembly code pose significant challenges. Pioneering approaches with masked language modeling (MLM)-based methods have been limited by facilitating natural language interaction. While recent methods based on decoder-focused large language models (LLMs) have significantly enhanced semantic representation, they still struggle to capture the nuanced and sparse semantics in assembly code. In this paper, we propose Assembly Augmented Tuning (ASMA-Tune), an end-to-end structural-semantic instruction-tuning framework. Our approach synergizes encoder architectures with decoder-based LLMs through projector modules to enable comprehensive code understanding. Experiments show that ASMA-Tune outperforms existing benchmarks, significantly enhancing assembly code comprehension and instruction-following abilities. Our model and dataset are public at this https URL. 

---
# Synthesizing Access Control Policies using Large Language Models 

**Authors**: Adarsh Vatsa, Pratyush Patel, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2503.11573)  

**Abstract**: Cloud compute systems allow administrators to write access control policies that govern access to private data. While policies are written in convenient languages, such as AWS Identity and Access Management Policy Language, manually written policies often become complex and error prone. In this paper, we investigate whether and how well Large Language Models (LLMs) can be used to synthesize access control policies. Our investigation focuses on the task of taking an access control request specification and zero-shot prompting LLMs to synthesize a well-formed access control policy which correctly adheres to the request specification. We consider two scenarios, one which the request specification is given as a concrete list of requests to be allowed or denied, and another in which a natural language description is used to specify sets of requests to be allowed or denied. We then argue that for zero-shot prompting, more precise and structured prompts using a syntax based approach are necessary and experimentally show preliminary results validating our approach. 

---
# Implicit Bias-Like Patterns in Reasoning Models 

**Authors**: Messi H.J. Lee, Calvin K. Lai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11572)  

**Abstract**: Implicit bias refers to automatic or spontaneous mental processes that shape perceptions, judgments, and behaviors. Previous research examining `implicit bias' in large language models (LLMs) has often approached the phenomenon differently than how it is studied in humans by focusing primarily on model outputs rather than on model processing. To examine model processing, we present a method called the Reasoning Model Implicit Association Test (RM-IAT) for studying implicit bias-like patterns in reasoning models: LLMs that employ step-by-step reasoning to solve complex tasks. Using this method, we find that reasoning models require more tokens when processing association-incompatible information compared to association-compatible information. These findings suggest AI systems harbor patterns in processing information that are analogous to human implicit bias. We consider the implications of these implicit bias-like patterns for their deployment in real-world applications. 

---
# RASA: Replace Anyone, Say Anything -- A Training-Free Framework for Audio-Driven and Universal Portrait Video Editing 

**Authors**: Tianrui Pan, Lin Liu, Jie Liu, Xiaopeng Zhang, Jie Tang, Gangshan Wu, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11571)  

**Abstract**: Portrait video editing focuses on modifying specific attributes of portrait videos, guided by audio or video streams. Previous methods typically either concentrate on lip-region reenactment or require training specialized models to extract keypoints for motion transfer to a new identity. In this paper, we introduce a training-free universal portrait video editing framework that provides a versatile and adaptable editing strategy. This framework supports portrait appearance editing conditioned on the changed first reference frame, as well as lip editing conditioned on varied speech, or a combination of both. It is based on a Unified Animation Control (UAC) mechanism with source inversion latents to edit the entire portrait, including visual-driven shape control, audio-driven speaking control, and inter-frame temporal control. Furthermore, our method can be adapted to different scenarios by adjusting the initial reference frame, enabling detailed editing of portrait videos with specific head rotations and facial expressions. This comprehensive approach ensures a holistic and flexible solution for portrait video editing. The experimental results show that our model can achieve more accurate and synchronized lip movements for the lip editing task, as well as more flexible motion transfer for the appearance editing task. Demo is available at this https URL. 

---
# Designing Neural Synthesizers for Low Latency Interaction 

**Authors**: Franco Caspe, Jordie Shier, Mark Sandler, Charalampos Saitis, Andrew McPherson  

**Link**: [PDF](https://arxiv.org/pdf/2503.11562)  

**Abstract**: Neural Audio Synthesis (NAS) models offer interactive musical control over high-quality, expressive audio generators. While these models can operate in real-time, they often suffer from high latency, making them unsuitable for intimate musical interaction. The impact of architectural choices in deep learning models on audio latency remains largely unexplored in the NAS literature. In this work, we investigate the sources of latency and jitter typically found in interactive NAS models. We then apply this analysis to the task of timbre transfer using RAVE, a convolutional variational autoencoder for audio waveforms introduced by Caillon et al. in 2021. Finally, we present an iterative design approach for optimizing latency. This culminates with a model we call BRAVE (Bravely Realtime Audio Variational autoEncoder), which is low-latency and exhibits better pitch and loudness replication while showing timbre modification capabilities similar to RAVE. We implement it in a specialized inference framework for low-latency, real-time inference and present a proof-of-concept audio plugin compatible with audio signals from musical instruments. We expect the challenges and guidelines described in this document to support NAS researchers in designing models for low-latency inference from the ground up, enriching the landscape of possibilities for musicians. 

---
# FLASHμ: Fast Localizing And Sizing of Holographic Microparticles 

**Authors**: Ayush Paliwal, Oliver Schlenczek, Birte Thiede, Manuel Santos Pereira, Katja Stieger, Eberhard Bodenschatz, Gholamhossein Bagheri, Alexander Ecker  

**Link**: [PDF](https://arxiv.org/pdf/2503.11538)  

**Abstract**: Reconstructing the 3D location and size of microparticles from diffraction images - holograms - is a computationally expensive inverse problem that has traditionally been solved using physics-based reconstruction methods. More recently, researchers have used machine learning methods to speed up the process. However, for small particles in large sample volumes the performance of these methods falls short of standard physics-based reconstruction methods. Here we designed a two-stage neural network architecture, FLASH$\mu$, to detect small particles (6-100$\mu$m) from holograms with large sample depths up to 20cm. Trained only on synthetic data with added physical noise, our method reliably detects particles of at least 9$\mu$m diameter in real holograms, comparable to the standard reconstruction-based approaches while operating on smaller crops, at quarter of the original resolution and providing roughly a 600-fold speedup. In addition to introducing a novel approach to a non-local object detection or signal demixing problem, our work could enable low-cost, real-time holographic imaging setups. 

---
# Potential of large language model-powered nudges for promoting daily water and energy conservation 

**Authors**: Zonghan Li, Song Tong, Yi Liu, Kaiping Peng, Chunyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11531)  

**Abstract**: The increasing amount of pressure related to water and energy shortages has increased the urgency of cultivating individual conservation behaviors. While the concept of nudging, i.e., providing usage-based feedback, has shown promise in encouraging conservation behaviors, its efficacy is often constrained by the lack of targeted and actionable content. This study investigates the impact of the use of large language models (LLMs) to provide tailored conservation suggestions for conservation intentions and their rationale. Through a survey experiment with 1,515 university participants, we compare three virtual nudging scenarios: no nudging, traditional nudging with usage statistics, and LLM-powered nudging with usage statistics and personalized conservation suggestions. The results of statistical analyses and causal forest modeling reveal that nudging led to an increase in conservation intentions among 86.9%-98.0% of the participants. LLM-powered nudging achieved a maximum increase of 18.0% in conservation intentions, surpassing traditional nudging by 88.6%. Furthermore, structural equation modeling results reveal that exposure to LLM-powered nudges enhances self-efficacy and outcome expectations while diminishing dependence on social norms, thereby increasing intrinsic motivation to conserve. These findings highlight the transformative potential of LLMs in promoting individual water and energy conservation, representing a new frontier in the design of sustainable behavioral interventions and resource management. 

---
# Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks 

**Authors**: Pengxin Guo, Runxi Wang, Shuang Zeng, Jinjing Zhu, Haoning Jiang, Yanran Wang, Yuyin Zhou, Feifei Wang, Hui Xiong, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11514)  

**Abstract**: Federated Learning (FL) has emerged as a promising privacy-preserving collaborative model training paradigm without sharing raw data. However, recent studies have revealed that private information can still be leaked through shared gradient information and attacked by Gradient Inversion Attacks (GIA). While many GIA methods have been proposed, a detailed analysis, evaluation, and summary of these methods are still lacking. Although various survey papers summarize existing privacy attacks in FL, few studies have conducted extensive experiments to unveil the effectiveness of GIA and their associated limiting factors in this context. To fill this gap, we first undertake a systematic review of GIA and categorize existing methods into three types, i.e., \textit{optimization-based} GIA (OP-GIA), \textit{generation-based} GIA (GEN-GIA), and \textit{analytics-based} GIA (ANA-GIA). Then, we comprehensively analyze and evaluate the three types of GIA in FL, providing insights into the factors that influence their performance, practicality, and potential threats. Our findings indicate that OP-GIA is the most practical attack setting despite its unsatisfactory performance, while GEN-GIA has many dependencies and ANA-GIA is easily detectable, making them both impractical. Finally, we offer a three-stage defense pipeline to users when designing FL frameworks and protocols for better privacy protection and share some future research directions from the perspectives of attackers and defenders that we believe should be pursued. We hope that our study can help researchers design more robust FL frameworks to defend against these attacks. 

---
# HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive Large Language Models 

**Authors**: Ziqin Zhou, Yifan Yang, Yuqing Yang, Tianyu He, Houwen Peng, Kai Qiu, Qi Dai, Lili Qiu, Chong Luo, Lingqiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11513)  

**Abstract**: Text-to-video generation poses significant challenges due to the inherent complexity of video data, which spans both temporal and spatial dimensions. It introduces additional redundancy, abrupt variations, and a domain gap between language and vision tokens while generation. Addressing these challenges requires an effective video tokenizer that can efficiently encode video data while preserving essential semantic and spatiotemporal information, serving as a critical bridge between text and vision. Inspired by the observation in VQ-VAE-2 and workflows of traditional animation, we propose HiTVideo for text-to-video generation with hierarchical tokenizers. It utilizes a 3D causal VAE with a multi-layer discrete token framework, encoding video content into hierarchically structured codebooks. Higher layers capture semantic information with higher compression, while lower layers focus on fine-grained spatiotemporal details, striking a balance between compression efficiency and reconstruction quality. Our approach efficiently encodes longer video sequences (e.g., 8 seconds, 64 frames), reducing bits per pixel (bpp) by approximately 70\% compared to baseline tokenizers, while maintaining competitive reconstruction quality. We explore the trade-offs between compression and reconstruction, while emphasizing the advantages of high-compressed semantic tokens in text-to-video tasks. HiTVideo aims to address the potential limitations of existing video tokenizers in text-to-video generation tasks, striving for higher compression ratios and simplify LLMs modeling under language guidance, offering a scalable and promising framework for advancing text to video generation. Demo page: this https URL. 

---
# Alzheimer's Disease Classification Using Retinal OCT: TransnetOCT and Swin Transformer Models 

**Authors**: Siva Manohar Reddy Kesu, Neelam Sinha, Hariharan Ramasangu, Thomas Gregor Issac  

**Link**: [PDF](https://arxiv.org/pdf/2503.11511)  

**Abstract**: Retinal optical coherence tomography (OCT) images are the biomarkers for neurodegenerative diseases, which are rising in prevalence. Early detection of Alzheimer's disease using retinal OCT is a primary challenging task. This work utilizes advanced deep learning techniques to classify retinal OCT images of subjects with Alzheimer's disease (AD) and healthy controls (CO). The goal is to enhance diagnostic capabilities through efficient image analysis. In the proposed model, Raw OCT images have been preprocessed with ImageJ and given to various deep-learning models to evaluate the accuracy. The best classification architecture is TransNetOCT, which has an average accuracy of 98.18% for input OCT images and 98.91% for segmented OCT images for five-fold cross-validation compared to other models, and the Swin Transformer model has achieved an accuracy of 93.54%. The evaluation accuracy metric demonstrated TransNetOCT and Swin transformer models capability to classify AD and CO subjects reliably, contributing to the potential for improved diagnostic processes in clinical settings. 

---
# Unicorn: A Universal and Collaborative Reinforcement Learning Approach Towards Generalizable Network-Wide Traffic Signal Control 

**Authors**: Yifeng Zhang, Yilin Liu, Ping Gong, Peizhuo Li, Mingfeng Fan, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2503.11488)  

**Abstract**: Adaptive traffic signal control (ATSC) is crucial in reducing congestion, maximizing throughput, and improving mobility in rapidly growing urban areas. Recent advancements in parameter-sharing multi-agent reinforcement learning (MARL) have greatly enhanced the scalable and adaptive optimization of complex, dynamic flows in large-scale homogeneous networks. However, the inherent heterogeneity of real-world traffic networks, with their varied intersection topologies and interaction dynamics, poses substantial challenges to achieving scalable and effective ATSC across different traffic scenarios. To address these challenges, we present Unicorn, a universal and collaborative MARL framework designed for efficient and adaptable network-wide ATSC. Specifically, we first propose a unified approach to map the states and actions of intersections with varying topologies into a common structure based on traffic movements. Next, we design a Universal Traffic Representation (UTR) module with a decoder-only network for general feature extraction, enhancing the model's adaptability to diverse traffic scenarios. Additionally, we incorporate an Intersection Specifics Representation (ISR) module, designed to identify key latent vectors that represent the unique intersection's topology and traffic dynamics through variational inference techniques. To further refine these latent representations, we employ a contrastive learning approach in a self-supervised manner, which enables better differentiation of intersection-specific features. Moreover, we integrate the state-action dependencies of neighboring agents into policy optimization, which effectively captures dynamic agent interactions and facilitates efficient regional collaboration. Our results show that Unicorn outperforms other methods across various evaluation metrics, highlighting its potential in complex, dynamic traffic networks. 

---
# Research Vision: Multi-Agent Path Planning for Cops And Robbers Via Reactive Synthesis 

**Authors**: William Fishell, Andoni Rodriguez, Mark Santolucito  

**Link**: [PDF](https://arxiv.org/pdf/2503.11475)  

**Abstract**: We propose the problem of multi-agent path planning for a generalization of the classic Cops and Robbers game via reactive synthesis. Specifically, through the application of LTLt and Coordination Synthesis, we aim to check whether various Cops and Robbers games are realizable (a strategy exists for the cops which guarantees they catch the robbers). Additionally, we construct this strategy as an executable program for the multiple system players in our games. In this paper we formalize the problem space, and propose potential directions for solutions. We also show how our formalization of this generalized cops and robbers game can be mapped to a broad range of other problems in the reactive program synthesis space. 

---
# Cerebrum (AIOS SDK): A Platform for Agent Development, Deployment, Distribution, and Discovery 

**Authors**: Balaji Rama, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11444)  

**Abstract**: Autonomous LLM-based agents have emerged as a powerful paradigm for complex task execution, yet the field lacks standardized tools for development, deployment, distribution and discovery of agents. We present Cerebrum, an Agent SDK for AIOS that addresses this gap through three key components: (1) a comprehensive SDK featuring a modular four-layer architecture for agent development, encompassing LLM, memory, storage, and tool management; (2) a community-driven Agent Hub for sharing and discovering agents, complete with version control and dependency management; (3) an interactive web interface for testing and evaluating agents. The platform's effectiveness is demonstrated through implementations of various agent architectures, including Chain of Thought (CoT), ReAct, and tool-use agents. Cerebrum advances the field by providing a unified framework that standardizes agent development while maintaining flexibility for researchers and developers to innovate and distribute their agents. The live website is at this https URL, the code is at this https URL, and video is at this https URL. 

---
# Adaptive Torque Control of Exoskeletons under Spasticity Conditions via Reinforcement Learning 

**Authors**: Andrés Chavarrías, David Rodriguez-Cianca, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2503.11433)  

**Abstract**: Spasticity is a common movement disorder symptom in individuals with cerebral palsy, hereditary spastic paraplegia, spinal cord injury and stroke, being one of the most disabling features in the progression of these diseases. Despite the potential benefit of using wearable robots to treat spasticity, their use is not currently recommended to subjects with a level of spasticity above ${1^+}$ on the Modified Ashworth Scale. The varying dynamics of this velocity-dependent tonic stretch reflex make it difficult to deploy safe personalized controllers. Here, we describe a novel adaptive torque controller via deep reinforcement learning (RL) for a knee exoskeleton under joint spasticity conditions, which accounts for task performance and interaction forces reduction. To train the RL agent, we developed a digital twin, including a musculoskeletal-exoskeleton system with joint misalignment and a differentiable spastic reflexes model for the muscles activation. Results for a simulated knee extension movement showed that the agent learns to control the exoskeleton for individuals with different levels of spasticity. The proposed controller was able to reduce maximum torques applied to the human joint under spastic conditions by an average of 10.6\% and decreases the root mean square until the settling time by 8.9\% compared to a conventional compliant controller. 

---
# Combining Causal Models for More Accurate Abstractions of Neural Networks 

**Authors**: Theodora-Mara Pîslar, Sara Magliacane, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2503.11429)  

**Abstract**: Mechanistic interpretability aims to reverse engineer neural networks by uncovering which high-level algorithms they implement. Causal abstraction provides a precise notion of when a network implements an algorithm, i.e., a causal model of the network contains low-level features that realize the high-level variables in a causal model of the algorithm. A typical problem in practical settings is that the algorithm is not an entirely faithful abstraction of the network, meaning it only partially captures the true reasoning process of a model. We propose a solution where we combine different simple high-level models to produce a more faithful representation of the network. Through learning this combination, we can model neural networks as being in different computational states depending on the input provided, which we show is more accurate to GPT 2-small fine-tuned on two toy tasks. We observe a trade-off between the strength of an interpretability hypothesis, which we define in terms of the number of inputs explained by the high-level models, and its faithfulness, which we define as the interchange intervention accuracy. Our method allows us to modulate between the two, providing the most accurate combination of models that describe the behavior of a neural network given a faithfulness level. 

---
# From Generative AI to Innovative AI: An Evolutionary Roadmap 

**Authors**: Seyed Mahmoud Sajjadi Mohammadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11419)  

**Abstract**: This paper explores the critical transition from Generative Artificial Intelligence (GenAI) to Innovative Artificial Intelligence (InAI). While recent advancements in GenAI have enabled systems to produce high-quality content across various domains, these models often lack the capacity for true innovation. In this context, innovation is defined as the ability to generate novel and useful outputs that go beyond mere replication of learned data. The paper examines this shift and proposes a roadmap for developing AI systems that can generate content and engage in autonomous problem-solving and creative ideation. The work provides both theoretical insights and practical strategies for advancing AI to a stage where it can genuinely innovate, contributing meaningfully to science, technology, and the arts. 

---
# A Neural Network Architecture Based on Attention Gate Mechanism for 3D Magnetotelluric Forward Modeling 

**Authors**: Xin Zhong, Weiwei Ling, Kejia Pan, Pinxia Wu, Jiajing Zhang, Zhiliang Zhan, Wenbo Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11408)  

**Abstract**: Traditional three-dimensional magnetotelluric (MT) numerical forward modeling methods, such as the finite element method (FEM) and finite volume method (FVM), suffer from high computational costs and low efficiency due to limitations in mesh refinement and computational resources. We propose a novel neural network architecture named MTAGU-Net, which integrates an attention gating mechanism for 3D MT forward modeling. Specifically, a dual-path attention gating module is designed based on forward response data images and embedded in the skip connections between the encoder and decoder. This module enables the fusion of critical anomaly information from shallow feature maps during the decoding of deep feature maps, significantly enhancing the network's capability to extract features from anomalous regions. Furthermore, we introduce a synthetic model generation method utilizing 3D Gaussian random field (GRF), which accurately replicates the electrical structures of real-world geological scenarios with high fidelity. Numerical experiments demonstrate that MTAGU-Net outperforms conventional 3D U-Net in terms of convergence stability and prediction accuracy, with the structural similarity index (SSIM) of the forward response data consistently exceeding 0.98. Moreover, the network can accurately predict forward response data on previously unseen datasets models, demonstrating its strong generalization ability and validating the feasibility and effectiveness of this method in practical applications. 

---
# Towards A Correct Usage of Cryptography in Semantic Watermarks for Diffusion Models 

**Authors**: Jonas Thietke, Andreas Müller, Denis Lukovnikov, Asja Fischer, Erwin Quiring  

**Link**: [PDF](https://arxiv.org/pdf/2503.11404)  

**Abstract**: Semantic watermarking methods enable the direct integration of watermarks into the generation process of latent diffusion models by only modifying the initial latent noise. One line of approaches building on Gaussian Shading relies on cryptographic primitives to steer the sampling process of the latent noise. However, we identify several issues in the usage of cryptographic techniques in Gaussian Shading, particularly in its proof of lossless performance and key management, causing ambiguity in follow-up works, too. In this work, we therefore revisit the cryptographic primitives for semantic watermarking. We introduce a novel, general proof of lossless performance based on IND\$-CPA security for semantic watermarks. We then discuss the configuration of the cryptographic primitives in semantic watermarks with respect to security, efficiency, and generation quality. 

---
# Hierarchical Information-Guided Spatio-Temporal Mamba for Stock Time Series Forecasting 

**Authors**: Wenbo Yan, Shurui Wang, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11387)  

**Abstract**: Mamba has demonstrated excellent performance in various time series forecasting tasks due to its superior selection mechanism. Nevertheless, conventional Mamba-based models encounter significant challenges in accurately predicting stock time series, as they fail to adequately capture both the overarching market dynamics and the intricate interdependencies among individual stocks. To overcome these constraints, we introduce the Hierarchical Information-Guided Spatio-Temporal Mamba (HIGSTM) framework. HIGSTM introduces Index-Guided Frequency Filtering Decomposition to extract commonality and specificity from time series. The model architecture features a meticulously designed hierarchical framework that systematically captures both temporal dynamic patterns and global static relationships within the stock market. Furthermore, we propose an Information-Guided Mamba that integrates macro informations into the sequence selection process, thereby facilitating more market-conscious decision-making. Comprehensive experimental evaluations conducted on the CSI500, CSI800 and CSI1000 datasets demonstrate that HIGSTM achieves state-of-the-art performance. 

---
# Annotating Scientific Uncertainty: A comprehensive model using linguistic patterns and comparison with existing approaches 

**Authors**: Panggih Kusuma Ningrum, Philipp Mayr, Nina Smirnova, Iana Atanassova  

**Link**: [PDF](https://arxiv.org/pdf/2503.11376)  

**Abstract**: UnScientify, a system designed to detect scientific uncertainty in scholarly full text. The system utilizes a weakly supervised technique to identify verbally expressed uncertainty in scientific texts and their authorial references. The core methodology of UnScientify is based on a multi-faceted pipeline that integrates span pattern matching, complex sentence analysis and author reference checking. This approach streamlines the labeling and annotation processes essential for identifying scientific uncertainty, covering a variety of uncertainty expression types to support diverse applications including information retrieval, text mining and scientific document processing. The evaluation results highlight the trade-offs between modern large language models (LLMs) and the UnScientify system. UnScientify, which employs more traditional techniques, achieved superior performance in the scientific uncertainty detection task, attaining an accuracy score of 0.808. This finding underscores the continued relevance and efficiency of UnScientify's simple rule-based and pattern matching strategy for this specific application. The results demonstrate that in scenarios where resource efficiency, interpretability, and domain-specific adaptability are critical, traditional methods can still offer significant advantages. 

---
# PARIC: Probabilistic Attention Regularization for Language Guided Image Classification from Pre-trained Vison Language Models 

**Authors**: Mayank Nautiyal, Stela Arranz Gheorghe, Kristiana Stefa, Li Ju, Ida-Maria Sintorn, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.11360)  

**Abstract**: Language-guided attention frameworks have significantly enhanced both interpretability and performance in image classification; however, the reliance on deterministic embeddings from pre-trained vision-language foundation models to generate reference attention maps frequently overlooks the intrinsic multivaluedness and ill-posed characteristics of cross-modal mappings. To address these limitations, we introduce PARIC, a probabilistic framework for guiding visual attention via language specifications. Our approach enables pre-trained vision-language models to generate probabilistic reference attention maps, which align textual and visual modalities more effectively while incorporating uncertainty estimates, as compared to their deterministic counterparts. Experiments on benchmark test problems demonstrate that PARIC enhances prediction accuracy, mitigates bias, ensures consistent predictions, and improves robustness across various datasets. 

---
# An experimental approach on Few Shot Class Incremental Learning 

**Authors**: Marinela Adam  

**Link**: [PDF](https://arxiv.org/pdf/2503.11349)  

**Abstract**: Few-Shot Class-Incremental Learning (FSCIL) represents a cutting-edge paradigm within the broader scope of machine learning, designed to empower models with the ability to assimilate new classes of data with limited examples while safeguarding existing knowledge. The paper will present different solutions which contain extensive experiments across large-scale datasets, domain shifts, and network architectures to evaluate and compare the selected methods. We highlight their advantages and then present an experimental approach with the purpose of improving the most promising one by replacing the visual-language (V-L) model (CLIP) with another V-L model (CLOOB) that seem to outperform it on zero-shot learning tasks. The aim of this report is to present an experimental method for FSCIL that would improve its performance. We also plan to offer an overview followed by an analysis of the recent advancements in FSCIL domain, focusing on various strategies to mitigate catastrophic forgetting and improve the adaptability of models to evolving tasks and datasets. 

---
# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

---
# Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model 

**Authors**: Moritz A. Zanger, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T.J. Spaan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11339)  

**Abstract**: Uncertainty quantification is a critical aspect of reinforcement learning and deep learning, with numerous applications ranging from efficient exploration and stable offline reinforcement learning to outlier detection in medical diagnostics. The scale of modern neural networks, however, complicates the use of many theoretically well-motivated approaches such as full Bayesian inference. Approximate methods like deep ensembles can provide reliable uncertainty estimates but still remain computationally expensive. In this work, we propose contextual similarity distillation, a novel approach that explicitly estimates the variance of an ensemble of deep neural networks with a single model, without ever learning or evaluating such an ensemble in the first place. Our method builds on the predictable learning dynamics of wide neural networks, governed by the neural tangent kernel, to derive an efficient approximation of the predictive variance of an infinite ensemble. Specifically, we reinterpret the computation of ensemble variance as a supervised regression problem with kernel similarities as regression targets. The resulting model can estimate predictive variance at inference time with a single forward pass, and can make use of unlabeled target-domain data or data augmentations to refine its uncertainty estimates. We empirically validate our method across a variety of out-of-distribution detection benchmarks and sparse-reward reinforcement learning environments. We find that our single-model method performs competitively and sometimes superior to ensemble-based baselines and serves as a reliable signal for efficient exploration. These results, we believe, position contextual similarity distillation as a principled and scalable alternative for uncertainty quantification in reinforcement learning and general deep learning. 

---
# Cardiomyopathy Diagnosis Model from Endomyocardial Biopsy Specimens: Appropriate Feature Space and Class Boundary in Small Sample Size Data 

**Authors**: Masaya Mori, Yuto Omae, Yutaka Koyama, Kazuyuki Hara, Jun Toyotani, Yasuo Okumura, Hiroyuki Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11331)  

**Abstract**: As the number of patients with heart failure increases, machine learning (ML) has garnered attention in cardiomyopathy diagnosis, driven by the shortage of pathologists. However, endomyocardial biopsy specimens are often small sample size and require techniques such as feature extraction and dimensionality reduction. This study aims to determine whether texture features are effective for feature extraction in the pathological diagnosis of cardiomyopathy. Furthermore, model designs that contribute toward improving generalization performance are examined by applying feature selection (FS) and dimensional compression (DC) to several ML models. The obtained results were verified by visualizing the inter-class distribution differences and conducting statistical hypothesis testing based on texture features. Additionally, they were evaluated using predictive performance across different model designs with varying combinations of FS and DC (applied or not) and decision boundaries. The obtained results confirmed that texture features may be effective for the pathological diagnosis of cardiomyopathy. Moreover, when the ratio of features to the sample size is high, a multi-step process involving FS and DC improved the generalization performance, with the linear kernel support vector machine achieving the best results. This process was demonstrated to be potentially effective for models with reduced complexity, regardless of whether the decision boundaries were linear, curved, perpendicular, or parallel to the axes. These findings are expected to facilitate the development of an effective cardiomyopathy diagnostic model for its rapid adoption in medical practice. 

---
# Learning to reset in target search problems 

**Authors**: Gorka Muñoz-Gil, Hans J. Briegel, Michele Caraglio  

**Link**: [PDF](https://arxiv.org/pdf/2503.11330)  

**Abstract**: Target search problems are central to a wide range of fields, from biological foraging to the optimization algorithms. Recently, the ability to reset the search has been shown to significantly improve the searcher's efficiency. However, the optimal resetting strategy depends on the specific properties of the search problem and can often be challenging to determine. In this work, we propose a reinforcement learning (RL)-based framework to train agents capable of optimizing their search efficiency in environments by learning how to reset. First, we validate the approach in a well-established benchmark: the Brownian search with resetting. There, RL agents consistently recover strategies closely resembling the sharp resetting distribution, known to be optimal in this scenario. We then extend the framework by allowing agents to control not only when to reset, but also their spatial dynamics through turning actions. In this more complex setting, the agents discover strategies that adapt both resetting and turning to the properties of the environment, outperforming the proposed benchmarks. These results demonstrate how reinforcement learning can serve both as an optimization tool and a mechanism for uncovering new, interpretable strategies in stochastic search processes with resetting. 

---
# BriLLM: Brain-inspired Large Language Model 

**Authors**: Hai Zhao, Hongqiu Wu, Dongjie Yang, Anni Zou, Jiale Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11299)  

**Abstract**: This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above. 

---
# AI and Deep Learning for Automated Segmentation and Quantitative Measurement of Spinal Structures in MRI 

**Authors**: Praveen Shastry, Bhawana Sonawane, Kavya Mohan, Naveen Kumarasami, Anandakumar D, Keerthana R, Mounigasri M, Kaviya SP, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.11281)  

**Abstract**: Background: Accurate spinal structure measurement is crucial for assessing spine health and diagnosing conditions like spondylosis, disc herniation, and stenosis. Manual methods for measuring intervertebral disc height and spinal canal diameter are subjective and time-consuming. Automated solutions are needed to improve accuracy, efficiency, and reproducibility in clinical practice.
Purpose: This study develops an autonomous AI system for segmenting and measuring key spinal structures in MRI scans, focusing on intervertebral disc height and spinal canal anteroposterior (AP) diameter in the cervical, lumbar, and thoracic regions. The goal is to reduce clinician workload, enhance diagnostic consistency, and improve assessments.
Methods: The AI model leverages deep learning architectures, including UNet, nnU-Net, and CNNs. Trained on a large proprietary MRI dataset, it was validated against expert annotations. Performance was evaluated using Dice coefficients and segmentation accuracy.
Results: The AI model achieved Dice coefficients of 0.94 for lumbar, 0.91 for cervical, and 0.90 for dorsal spine segmentation (D1-D12). It precisely measured spinal parameters like disc height and canal diameter, demonstrating robustness and clinical applicability.
Conclusion: The AI system effectively automates MRI-based spinal measurements, improving accuracy and reducing clinician workload. Its consistent performance across spinal regions supports clinical decision-making, particularly in high-demand settings, enhancing spinal assessments and patient outcomes. 

---
# Financial Fraud Detection with Entropy Computing 

**Authors**: Babak Emami, Wesley Dyk, David Haycraft, Carrie Spear, Lac Nguyen, Nicholas Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2503.11273)  

**Abstract**: We introduce CVQBoost, a novel classification algorithm that leverages early hardware implementing Quantum Computing Inc's Entropy Quantum Computing (EQC) paradigm, Dirac-3 [Nguyen et. al. arXiv:2407.04512]. We apply CVQBoost to a fraud detection test case and benchmark its performance against XGBoost, a widely utilized ML method. Running on Dirac-3, CVQBoost demonstrates a significant runtime advantage over XGBoost, which we evaluate on high-performance hardware comprising up to 48 CPUs and four NVIDIA L4 GPUs using the RAPIDS AI framework. Our results show that CVQBoost maintains competitive accuracy (measured by AUC) while significantly reducing training time, particularly as dataset size and feature complexity increase. To assess scalability, we extend our study to large synthetic datasets ranging from 1M to 70M samples, demonstrating that CVQBoost on Dirac-3 is well-suited for large-scale classification tasks. These findings position CVQBoost as a promising alternative to gradient boosting methods, offering superior scalability and efficiency for high-dimensional ML applications such as fraud detection. 

---
# Line of Duty: Evaluating LLM Self-Knowledge via Consistency in Feasibility Boundaries 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2503.11256)  

**Abstract**: As LLMs grow more powerful, their most profound achievement may be recognising when to say "I don't know". Existing studies on LLM self-knowledge have been largely constrained by human-defined notions of feasibility, often neglecting the reasons behind unanswerability by LLMs and failing to study deficient types of self-knowledge. This study aims to obtain intrinsic insights into different types of LLM self-knowledge with a novel methodology: allowing them the flexibility to set their own feasibility boundaries and then analysing the consistency of these limits. We find that even frontier models like GPT-4o and Mistral Large are not sure of their own capabilities more than 80% of the time, highlighting a significant lack of trustworthiness in responses. Our analysis of confidence balance in LLMs indicates that models swing between overconfidence and conservatism in feasibility boundaries depending on task categories and that the most significant self-knowledge weaknesses lie in temporal awareness and contextual understanding. These difficulties in contextual comprehension additionally lead models to question their operational boundaries, resulting in considerable confusion within the self-knowledge of LLMs. We make our code and results available publicly at this https URL 

---
# Spherical Tree-Sliced Wasserstein Distance 

**Authors**: Hoang V. Tran, Thanh T. Chu, Khoi N.M. Nguyen, Trang Pham, Tam Le, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11249)  

**Abstract**: Sliced Optimal Transport (OT) simplifies the OT problem in high-dimensional spaces by projecting supports of input measures onto one-dimensional lines and then exploiting the closed-form expression of the univariate OT to reduce the computational burden of OT. Recently, the Tree-Sliced method has been introduced to replace these lines with more intricate structures, known as tree systems. This approach enhances the ability to capture topological information of integration domains in Sliced OT while maintaining low computational cost. Inspired by this approach, in this paper, we present an adaptation of tree systems on OT problems for measures supported on a sphere. As a counterpart to the Radon transform variant on tree systems, we propose a novel spherical Radon transform with a new integration domain called spherical trees. By leveraging this transform and exploiting the spherical tree structures, we derive closed-form expressions for OT problems on the sphere. Consequently, we obtain an efficient metric for measures on the sphere, named Spherical Tree-Sliced Wasserstein (STSW) distance. We provide an extensive theoretical analysis to demonstrate the topology of spherical trees and the well-definedness and injectivity of our Radon transform variant, which leads to an orthogonally invariant distance between spherical measures. Finally, we conduct a wide range of numerical experiments, including gradient flows and self-supervised learning, to assess the performance of our proposed metric, comparing it to recent benchmarks. 

---
# Compound Expression Recognition via Large Vision-Language Models 

**Authors**: Jun Yu, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11241)  

**Abstract**: Compound Expression Recognition (CER) is crucial for understanding human emotions and improving human-computer interaction. However, CER faces challenges due to the complexity of facial expressions and the difficulty of capturing subtle emotional cues. To address these issues, we propose a novel approach leveraging Large Vision-Language Models (LVLMs). Our method employs a two-stage fine-tuning process: first, pre-trained LVLMs are fine-tuned on basic facial expressions to establish foundational patterns; second, the model is further optimized on a compound-expression dataset to refine visual-language feature interactions. Our approach achieves advanced accuracy on the RAF-DB dataset and demonstrates strong zero-shot generalization on the C-EXPR-DB dataset, showcasing its potential for real-world applications in emotion analysis and human-computer interaction. 

---
# Technologies on Effectiveness and Efficiency: A Survey of State Spaces Models 

**Authors**: Xingtai Lv, Youbang Sun, Kaiyan Zhang, Shang Qu, Xuekai Zhu, Yuchen Fan, Yi Wu, Ermo Hua, Xinwei Long, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.11224)  

**Abstract**: State Space Models (SSMs) have emerged as a promising alternative to the popular transformer-based models and have been increasingly gaining attention. Compared to transformers, SSMs excel at tasks with sequential data or longer contexts, demonstrating comparable performances with significant efficiency gains. In this survey, we provide a coherent and systematic overview for SSMs, including their theoretical motivations, mathematical formulations, comparison with existing model classes, and various applications. We divide the SSM series into three main sections, providing a detailed introduction to the original SSM, the structured SSM represented by S4, and the selective SSM typified by Mamba. We put an emphasis on technicality, and highlight the various key techniques introduced to address the effectiveness and efficiency of SSMs. We hope this manuscript serves as an introduction for researchers to explore the theoretical foundations of SSMs. 

---
# MEET: A Million-Scale Dataset for Fine-Grained Geospatial Scene Classification with Zoom-Free Remote Sensing Imagery 

**Authors**: Yansheng Li, Yuning Wu, Gong Cheng, Chao Tao, Bo Dang, Yu Wang, Jiahao Zhang, Chuge Zhang, Yiting Liu, Xu Tang, Jiayi Ma, Yongjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11219)  

**Abstract**: Accurate fine-grained geospatial scene classification using remote sensing imagery is essential for a wide range of applications. However, existing approaches often rely on manually zooming remote sensing images at different scales to create typical scene samples. This approach fails to adequately support the fixed-resolution image interpretation requirements in real-world scenarios. To address this limitation, we introduce the Million-scale finE-grained geospatial scEne classification dataseT (MEET), which contains over 1.03 million zoom-free remote sensing scene samples, manually annotated into 80 fine-grained categories. In MEET, each scene sample follows a scene-inscene layout, where the central scene serves as the reference, and auxiliary scenes provide crucial spatial context for finegrained classification. Moreover, to tackle the emerging challenge of scene-in-scene classification, we present the Context-Aware Transformer (CAT), a model specifically designed for this task, which adaptively fuses spatial context to accurately classify the scene samples. CAT adaptively fuses spatial context to accurately classify the scene samples by learning attentional features that capture the relationships between the center and auxiliary scenes. Based on MEET, we establish a comprehensive benchmark for fine-grained geospatial scene classification, evaluating CAT against 11 competitive baselines. The results demonstrate that CAT significantly outperforms these baselines, achieving a 1.88% higher balanced accuracy (BA) with the Swin-Large backbone, and a notable 7.87% improvement with the Swin-Huge backbone. Further experiments validate the effectiveness of each module in CAT and show the practical applicability of CAT in the urban functional zone mapping. The source code and dataset will be publicly available at this https URL. 

---
# Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering 

**Authors**: Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2503.11197)  

**Abstract**: Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at this https URL and this https URL. 

---
# Cross-Modal Learning for Music-to-Music-Video Description Generation 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2503.11190)  

**Abstract**: Music-to-music-video generation is a challenging task due to the intrinsic differences between the music and video modalities. The advent of powerful text-to-video diffusion models has opened a promising pathway for music-video (MV) generation by first addressing the music-to-MV description task and subsequently leveraging these models for video generation. In this study, we focus on the MV description generation task and propose a comprehensive pipeline encompassing training data construction and multimodal model fine-tuning. We fine-tune existing pre-trained multimodal models on our newly constructed music-to-MV description dataset based on the Music4All dataset, which integrates both musical and visual information. Our experimental results demonstrate that music representations can be effectively mapped to textual domains, enabling the generation of meaningful MV description directly from music inputs. We also identify key components in the dataset construction pipeline that critically impact the quality of MV description and highlight specific musical attributes that warrant greater focus for improved MV description generation. 

---
# Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification 

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11185)  

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation. 

---
# Multi-Stage Generative Upscaler: Reconstructing Football Broadcast Images via Diffusion Models 

**Authors**: Luca Martini, Daniele Zolezzi, Saverio Iacono, Gianni Viardo Vercelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.11181)  

**Abstract**: The reconstruction of low-resolution football broadcast images presents a significant challenge in sports broadcasting, where detailed visuals are essential for analysis and audience engagement. This study introduces a multi-stage generative upscaling framework leveraging Diffusion Models to enhance degraded images, transforming inputs as small as $64 \times 64$ pixels into high-fidelity $1024 \times 1024$ outputs. By integrating an image-to-image pipeline, ControlNet conditioning, and LoRA fine-tuning, our approach surpasses traditional upscaling methods in restoring intricate textures and domain-specific elements such as player details and jersey logos. The custom LoRA is trained on a custom football dataset, ensuring adaptability to sports broadcast needs. Experimental results demonstrate substantial improvements over conventional models, with ControlNet refining fine details and LoRA enhancing task-specific elements. These findings highlight the potential of diffusion-based image reconstruction in sports media, paving the way for future applications in automated video enhancement and real-time sports analytics. 

---
# Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement 

**Authors**: Yini Li, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2503.11175)  

**Abstract**: Low-light and underwater videos suffer from poor visibility, low contrast, and high noise, necessitating enhancements in visual quality. However, existing approaches typically rely on paired ground truth, which limits their practicality and often fails to maintain temporal consistency. To overcome these obstacles, this paper introduces a novel zero-shot learning approach named Zero-TIG, leveraging the Retinex theory and optical flow techniques. The proposed network consists of an enhancement module and a temporal feedback module. The enhancement module comprises three subnetworks: low-light image denoising, illumination estimation, and reflection denoising. The temporal enhancement module ensures temporal consistency by incorporating histogram equalization, optical flow computation, and image warping to align the enhanced previous frame with the current frame, thereby maintaining continuity. Additionally, we address color distortion in underwater data by adaptively balancing RGB channels. The experimental results demonstrate that our method achieves low-light video enhancement without the need for paired training data, making it a promising and applicable method for real-world scenario enhancement. 

---
# Neurons: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction 

**Authors**: Haonan Wang, Qixiang Zhang, Lehan Wang, Xuanqi Huang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11167)  

**Abstract**: Decoding visual stimuli from neural activity is essential for understanding the human brain. While fMRI methods have successfully reconstructed static images, fMRI-to-video reconstruction faces challenges due to the need for capturing spatiotemporal dynamics like motion and scene transitions. Recent approaches have improved semantic and perceptual alignment but struggle to integrate coarse fMRI data with detailed visual features. Inspired by the hierarchical organization of the visual system, we propose NEURONS, a novel framework that decouples learning into four correlated sub-tasks: key object segmentation, concept recognition, scene description, and blurry video reconstruction. This approach simulates the visual cortex's functional specialization, allowing the model to capture diverse video content. In the inference stage, NEURONS generates robust conditioning signals for a pre-trained text-to-video diffusion model to reconstruct the videos. Extensive experiments demonstrate that NEURONS outperforms state-of-the-art baselines, achieving solid improvements in video consistency (26.6%) and semantic-level accuracy (19.1%). Notably, NEURONS shows a strong functional correlation with the visual cortex, highlighting its potential for brain-computer interfaces and clinical applications. Code and model weights will be available at: this https URL. 

---
# Unifying Perplexing Behaviors in Modified BP Attributions through Alignment Perspective 

**Authors**: Guanhua Zheng, Jitao Sang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11160)  

**Abstract**: Attributions aim to identify input pixels that are relevant to the decision-making process. A popular approach involves using modified backpropagation (BP) rules to reverse decisions, which improves interpretability compared to the original gradients. However, these methods lack a solid theoretical foundation and exhibit perplexing behaviors, such as reduced sensitivity to parameter randomization, raising concerns about their reliability and highlighting the need for theoretical justification. In this work, we present a unified theoretical framework for methods like GBP, RectGrad, LRP, and DTD, demonstrating that they achieve input alignment by combining the weights of activated neurons. This alignment improves the visualization quality and reduces sensitivity to weight randomization. Our contributions include: (1) Providing a unified explanation for multiple behaviors, rather than focusing on just one. (2) Accurately predicting novel behaviors. (3) Offering insights into decision-making processes, including layer-wise information changes and the relationship between attributions and model decisions. 

---
# Don't Take Things Out of Context: Attention Intervention for Enhancing Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Shaotian Yan, Chen Shen, Wenxiao Wang, Liang Xie, Junjie Liu, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.11154)  

**Abstract**: Few-shot Chain-of-Thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs), functioning as a whole to guide these models in generating reasoning steps toward final answers. However, we observe that isolated segments, words, or tokens within CoT demonstrations can unexpectedly disrupt the generation process of LLMs. The model may overly concentrate on certain local information present in the demonstration, introducing irrelevant noise into the reasoning process and potentially leading to incorrect answers. In this paper, we investigate the underlying mechanism of CoT through dynamically tracing and manipulating the inner workings of LLMs at each output step, which demonstrates that tokens exhibiting specific attention characteristics are more likely to induce the model to take things out of context; these tokens directly attend to the hidden states tied with prediction, without substantial integration of non-local information. Building upon these insights, we propose a Few-shot Attention Intervention method (FAI) that dynamically analyzes the attention patterns of demonstrations to accurately identify these tokens and subsequently make targeted adjustments to the attention weights to effectively suppress their distracting effect on LLMs. Comprehensive experiments across multiple benchmarks demonstrate consistent improvements over baseline methods, with a remarkable 5.91% improvement on the AQuA dataset, further highlighting the effectiveness of FAI. 

---
# MoLEx: Mixture of Layer Experts for Finetuning with Sparse Upcycling 

**Authors**: Rachel S.Y. Teo, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11144)  

**Abstract**: Large-scale pre-training of deep models, followed by fine-tuning them, has become the cornerstone of natural language processing (NLP). The prevalence of data coupled with computational resources has led to large models with a considerable number of parameters. While the massive size of these models has led to remarkable success in many NLP tasks, a detriment is the expense required to retrain all the base model's parameters for the adaptation to each task or domain. Parameter Efficient Fine-Tuning (PEFT) provides an effective solution for this challenge by minimizing the number of parameters required to be fine-tuned while maintaining the quality of the model. While existing methods have achieved impressive results, they mainly focus on adapting a subset of parameters, weight reparameterization, and prompt engineering. In this paper, we study layers as extractors of different types of linguistic information that are valuable when used in conjunction. We then propose the Mixture of Layer Experts (MoLEx), a novel sparse mixture of experts (SMoE) whose experts are layers in the pre-trained model. It performs a conditional computation of a mixture of layers during fine-tuning to provide the model with more structural knowledge about the data. By providing an avenue for information exchange between layers, MoLEx enables the model to make a more well-informed prediction for the downstream task, leading to better fine-tuning results with the same number of effective parameters. As experts can be processed in parallel, MoLEx introduces minimal additional computational overhead. We empirically corroborate the advantages of MoLEx when combined with popular PEFT baseline methods on a variety of downstream fine-tuning tasks, including the popular GLUE benchmark as well as the End-to-End Challenge (E2E). The code is publicly available at this https URL. 

---
# Direction-Aware Diagonal Autoregressive Image Generation 

**Authors**: Yijia Xu, Jianzhong Ju, Jian Luan, Jinshi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.11129)  

**Abstract**: The raster-ordered image token sequence exhibits a significant Euclidean distance between index-adjacent tokens at line breaks, making it unsuitable for autoregressive generation. To address this issue, this paper proposes Direction-Aware Diagonal Autoregressive Image Generation (DAR) method, which generates image tokens following a diagonal scanning order. The proposed diagonal scanning order ensures that tokens with adjacent indices remain in close proximity while enabling causal attention to gather information from a broader range of directions. Additionally, two direction-aware modules: 4D-RoPE and direction embeddings are introduced, enhancing the model's capability to handle frequent changes in generation direction. To leverage the representational capacity of the image tokenizer, we use its codebook as the image token embeddings. We propose models of varying scales, ranging from 485M to 2.0B. On the 256$\times$256 ImageNet benchmark, our DAR-XL (2.0B) outperforms all previous autoregressive image generators, achieving a state-of-the-art FID score of 1.37. 

---
# Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning 

**Authors**: Matthew Khoriaty, Andrii Shportko, Gustavo Mercier, Zach Wood-Doughty  

**Link**: [PDF](https://arxiv.org/pdf/2503.11127)  

**Abstract**: Recent developments in Large Language Model (LLM) capabilities have brought great potential but also posed new risks. For example, LLMs with knowledge of bioweapons, advanced chemistry, or cyberattacks could cause violence if placed in the wrong hands or during malfunctions. Because of their nature as near-black boxes, intuitive interpretation of LLM internals remains an open research question, preventing developers from easily controlling model behavior and capabilities. The use of Sparse Autoencoders (SAEs) has recently emerged as a potential method of unraveling representations of concepts in LLMs internals, and has allowed developers to steer model outputs by directly modifying the hidden activations. In this paper, we use SAEs to identify unwanted concepts from the Weapons of Mass Destruction Proxy (WMDP) dataset within gemma-2-2b internals and use feature steering to reduce the model's ability to answer harmful questions while retaining its performance on harmless queries. Our results bring back optimism to the viability of SAE-based explicit knowledge unlearning techniques. 

---
# UMB@PerAnsSumm 2025: Enhancing Perspective-Aware Summarization with Prompt Optimization and Supervised Fine-Tuning 

**Authors**: Kristin Qi, Youxiang Zhu, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11118)  

**Abstract**: We present our approach to the PerAnsSumm Shared Task, which involves perspective span identification and perspective-aware summarization in community question-answering (CQA) threads. For span identification, we adopt ensemble learning that integrates three transformer models through averaging to exploit individual model strengths, achieving an 82.91% F1-score on test data. For summarization, we design a suite of Chain-of-Thought (CoT) prompting strategies that incorporate keyphrases and guide information to structure summary generation into manageable steps. To further enhance summary quality, we apply prompt optimization using the DSPy framework and supervised fine-tuning (SFT) on Llama-3 to adapt the model to domain-specific data. Experimental results on validation and test sets show that structured prompts with keyphrases and guidance improve summaries aligned with references, while the combination of prompt optimization and fine-tuning together yields significant improvement in both relevance and factuality evaluation metrics. 

---
# Limits of KV Cache Compression for Tensor Attention based Autoregressive Transformers 

**Authors**: Yifang Chen, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11108)  

**Abstract**: The key-value (KV) cache in autoregressive transformers presents a significant bottleneck during inference, which restricts the context length capabilities of large language models (LLMs). While previous work analyzes the fundamental space complexity barriers in standard attention mechanism [Haris and Onak, 2025], our work generalizes the space complexity barriers result to tensor attention version. Our theoretical contributions rely on a novel reduction from communication complexity and deduce the memory lower bound for tensor-structured attention mechanisms when $d = \Omega(\log n)$. In the low dimensional regime where $d = o(\log n)$, we analyze the theoretical bounds of the space complexity as well. Overall, our work provides a theoretical foundation for us to understand the compression-expressivity tradeoff in tensor attention mechanisms and offers more perspectives in developing more memory-efficient transformer architectures. 

---
# Quantifying Interpretability in CLIP Models with Concept Consistency 

**Authors**: Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2503.11103)  

**Abstract**: CLIP is one of the most popular foundational models and is heavily used for many vision-language tasks. However, little is known about the inner workings of CLIP. While recent work has proposed decomposition-based interpretability methods for identifying textual descriptions of attention heads in CLIP, the implications of conceptual consistency in these text labels on interpretability and model performance has not been explored. To bridge this gap, we study the conceptual consistency of text descriptions for attention heads in CLIP-like models. We conduct extensive experiments on six different models from OpenAI and OpenCLIP which vary by size, type of pre-training data and patch size. We propose Concept Consistency Score (CCS), a novel interpretability metric that measures how consistently individual attention heads in CLIP models align with specific concepts. To assign concept labels to heads, we use in-context learning with ChatGPT, guided by a few manually-curated examples, and validate these labels using an LLM-as-a-judge approach. Our soft-pruning experiments reveal that high CCS heads are critical for preserving model performance, as pruning them leads to a significantly larger performance drop than pruning random or low CCS heads. Notably, we find that high CCS heads capture essential concepts and play a key role in out-of-domain detection, concept-specific reasoning, and video-language understanding. These results position CCS as a powerful interpretability metric for analyzing CLIP-like models. 

---
# Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation 

**Authors**: He Zhang, Xinyi Fu, John M. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2503.11096)  

**Abstract**: Traditional image annotation tasks rely heavily on human effort for object selection and label assignment, making the process time-consuming and prone to decreased efficiency as annotators experience fatigue after extensive work. This paper introduces a novel framework that leverages the visual understanding capabilities of large multimodal models (LMMs), particularly GPT, to assist annotation workflows. In our proposed approach, human annotators focus on selecting objects via bounding boxes, while the LMM autonomously generates relevant labels. This human-AI collaborative framework enhances annotation efficiency by reducing the cognitive and time burden on human annotators. By analyzing the system's performance across various types of annotation tasks, we demonstrate its ability to generalize to tasks such as object recognition, scene description, and fine-grained categorization. Our proposed framework highlights the potential of this approach to redefine annotation workflows, offering a scalable and efficient solution for large-scale data labeling in computer vision. Finally, we discuss how integrating LMMs into the annotation pipeline can advance bidirectional human-AI alignment, as well as the challenges of alleviating the "endless annotation" burden in the face of information overload by shifting some of the work to AI. 

---
# EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks 

**Authors**: Yi Zhang, Qiang Zhang, Xiaozhu Ju, Zhaoyang Liu, Jilei Mao, Jingkai Sun, Jintao Wu, Shixiong Gao, Shihan Cai, Zhiyuan Qin, Linkai Liang, Jiaxu Wang, Yiqun Duan, Jiahang Cao, Renjing Xu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11089)  

**Abstract**: While multimodal large language models (MLLMs) have made groundbreaking progress in embodied intelligence, they still face significant challenges in spatial reasoning for complex long-horizon tasks. To address this gap, we propose EmbodiedVSR (Embodied Visual Spatial Reasoning), a novel framework that integrates dynamic scene graph-guided Chain-of-Thought (CoT) reasoning to enhance spatial understanding for embodied agents. By explicitly constructing structured knowledge representations through dynamic scene graphs, our method enables zero-shot spatial reasoning without task-specific fine-tuning. This approach not only disentangles intricate spatial relationships but also aligns reasoning steps with actionable environmental dynamics. To rigorously evaluate performance, we introduce the eSpatial-Benchmark, a comprehensive dataset including real-world embodied scenarios with fine-grained spatial annotations and adaptive task difficulty levels. Experiments demonstrate that our framework significantly outperforms existing MLLM-based methods in accuracy and reasoning coherence, particularly in long-horizon tasks requiring iterative environment interaction. The results reveal the untapped potential of MLLMs for embodied intelligence when equipped with structured, explainable reasoning mechanisms, paving the way for more reliable deployment in real-world spatial applications. The codes and datasets will be released soon. 

---
# A Survey of Cross-domain Graph Learning: Progress and Future Directions 

**Authors**: Haihong Zhao, Chenyi Zi, Aochuan Chen, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11086)  

**Abstract**: Graph learning plays a vital role in mining and analyzing complex relationships involved in graph data, which is widely used in many real-world applications like transaction networks and communication networks. Foundation models in CV and NLP have shown powerful cross-domain capabilities that are also significant in graph domains. However, existing graph learning approaches struggle with cross-domain tasks. Inspired by successes in CV and NLP, cross-domain graph learning has once again become a focal point of attention to realizing true graph foundation models. In this survey, we present a comprehensive review and analysis of existing works on cross-domain graph learning. Concretely, we first propose a new taxonomy, categorizing existing approaches based on the learned cross-domain information: structure, feature, and structure-feature mixture. Next, we systematically survey representative methods in these categories. Finally, we discuss the remaining limitations of existing studies and highlight promising avenues for future research. Relevant papers are summarized and will be consistently updated at: this https URL. 

---
# MoMa-Kitchen: A 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation 

**Authors**: Pingrui Zhang, Xianqiang Gao, Yuhan Wu, Kehui Liu, Dong Wang, Zhigang Wang, Bin Zhao, Yan Ding, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11081)  

**Abstract**: In mobile manipulation, navigation and manipulation are often treated as separate problems, resulting in a significant gap between merely approaching an object and engaging with it effectively. Many navigation approaches primarily define success by proximity to the target, often overlooking the necessity for optimal positioning that facilitates subsequent manipulation. To address this, we introduce MoMa-Kitchen, a benchmark dataset comprising over 100k samples that provide training data for models to learn optimal final navigation positions for seamless transition to manipulation. Our dataset includes affordance-grounded floor labels collected from diverse kitchen environments, in which robotic mobile manipulators of different models attempt to grasp target objects amidst clutter. Using a fully automated pipeline, we simulate diverse real-world scenarios and generate affordance labels for optimal manipulation positions. Visual data are collected from RGB-D inputs captured by a first-person view camera mounted on the robotic arm, ensuring consistency in viewpoint during data collection. We also develop a lightweight baseline model, NavAff, for navigation affordance grounding that demonstrates promising performance on the MoMa-Kitchen benchmark. Our approach enables models to learn affordance-based final positioning that accommodates different arm types and platform heights, thereby paving the way for more robust and generalizable integration of navigation and manipulation in embodied AI. Project page: \href{this https URL}{this https URL}. 

---
# Low-cost Real-world Implementation of the Swing-up Pendulum for Deep Reinforcement Learning Experiments 

**Authors**: Peter Böhm, Pauline Pounds, Archie C. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2503.11065)  

**Abstract**: Deep reinforcement learning (DRL) has had success in virtual and simulated domains, but due to key differences between simulated and real-world environments, DRL-trained policies have had limited success in real-world applications. To assist researchers to bridge the \textit{sim-to-real gap}, in this paper, we describe a low-cost physical inverted pendulum apparatus and software environment for exploring sim-to-real DRL methods. In particular, the design of our apparatus enables detailed examination of the delays that arise in physical systems when sensing, communicating, learning, inferring and actuating. Moreover, we wish to improve access to educational systems, so our apparatus uses readily available materials and parts to reduce cost and logistical barriers. Our design shows how commercial, off-the-shelf electronics and electromechanical and sensor systems, combined with common metal extrusions, dowel and 3D printed couplings provide a pathway for affordable physical DRL apparatus. The physical apparatus is complemented with a simulated environment implemented using a high-fidelity physics engine and OpenAI Gym interface. 

---
# Training Directional Locomotion for Quadrupedal Low-Cost Robotic Systems via Deep Reinforcement Learning 

**Authors**: Peter Böhm, Archie C. Chapman, Pauline Pounds  

**Link**: [PDF](https://arxiv.org/pdf/2503.11059)  

**Abstract**: In this work we present Deep Reinforcement Learning (DRL) training of directional locomotion for low-cost quadrupedal robots in the real world. In particular, we exploit randomization of heading that the robot must follow to foster exploration of action-state transitions most useful for learning both forward locomotion as well as course adjustments. Changing the heading in episode resets to current yaw plus a random value drawn from a normal distribution yields policies able to follow complex trajectories involving frequent turns in both directions as well as long straight-line stretches. By repeatedly changing the heading, this method keeps the robot moving within the training platform and thus reduces human involvement and need for manual resets during the training. Real world experiments on a custom-built, low-cost quadruped demonstrate the efficacy of our method with the robot successfully navigating all validation tests. When trained with other approaches, the robot only succeeds in forward locomotion test and fails when turning is required. 

---
# Distance-Based Tree-Sliced Wasserstein Distance 

**Authors**: Hoang V. Tran, Khoi N.M. Nguyen, Trang Pham, Thanh T. Chu, Tam Le, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11050)  

**Abstract**: To overcome computational challenges of Optimal Transport (OT), several variants of Sliced Wasserstein (SW) has been developed in the literature. These approaches exploit the closed-form expression of the univariate OT by projecting measures onto (one-dimensional) lines. However, projecting measures onto low-dimensional spaces can lead to a loss of topological information. Tree-Sliced Wasserstein distance on Systems of Lines (TSW-SL) has emerged as a promising alternative that replaces these lines with a more advanced structure called tree systems. The tree structures enhance the ability to capture topological information of the metric while preserving computational efficiency. However, at the core of TSW-SL, the splitting maps, which serve as the mechanism for pushing forward measures onto tree systems, focus solely on the position of the measure supports while disregarding the projecting domains. Moreover, the specific splitting map used in TSW-SL leads to a metric that is not invariant under Euclidean transformations, a typically expected property for OT on Euclidean space. In this work, we propose a novel class of splitting maps that generalizes the existing one studied in TSW-SL enabling the use of all positional information from input measures, resulting in a novel Distance-based Tree-Sliced Wasserstein (Db-TSW) distance. In addition, we introduce a simple tree sampling process better suited for Db-TSW, leading to an efficient GPU-friendly implementation for tree systems, similar to the original SW. We also provide a comprehensive theoretical analysis of proposed class of splitting maps to verify the injectivity of the corresponding Radon Transform, and demonstrate that Db-TSW is an Euclidean invariant metric. We empirically show that Db-TSW significantly improves accuracy compared to recent SW variants while maintaining low computational cost via a wide range of experiments. 

---
# Measuring Similarity in Causal Graphs: A Framework for Semantic and Structural Analysis 

**Authors**: Ning-Yuan Georgia Liu, Flower Yang, Mohammad S. Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2503.11046)  

**Abstract**: Causal graphs are commonly used to understand and model complex systems. Researchers often construct these graphs from different perspectives, leading to significant variations for the same problem. Comparing causal graphs is, therefore, essential for evaluating assumptions, integrating insights, and resolving disagreements. The rise of AI tools has further amplified this need, as they are increasingly used to generate hypothesized causal graphs by synthesizing information from various sources such as prior research and community inputs, providing the potential for automating and scaling causal modeling for complex systems. Similar to humans, these tools also produce inconsistent results across platforms, versions, and iterations. Despite its importance, research on causal graph comparison remains scarce. Existing methods often focus solely on structural similarities, assuming identical variable names, and fail to capture nuanced semantic relationships, which is essential for causal graph comparison. We address these gaps by investigating methods for comparing causal graphs from both semantic and structural perspectives. First, we reviewed over 40 existing metrics and, based on predefined criteria, selected nine for evaluation from two threads of machine learning: four semantic similarity metrics and five learning graph kernels. We discuss the usability of these metrics in simple examples to illustrate their strengths and limitations. We then generated a synthetic dataset of 2,000 causal graphs using generative AI based on a reference diagram. Our findings reveal that each metric captures a different aspect of similarity, highlighting the need to use multiple metrics. 

---
# Fourier Neural Operator based surrogates for $CO_2$ storage in realistic geologies 

**Authors**: Anirban Chandra, Marius Koch, Suraj Pawar, Aniruddha Panda, Kamyar Azizzadenesheli, Jeroen Snippe, Faruk O. Alpak, Farah Hariri, Clement Etienam, Pandu Devarakota, Anima Anandkumar, Detlef Hohl  

**Link**: [PDF](https://arxiv.org/pdf/2503.11031)  

**Abstract**: This study aims to develop surrogate models for accelerating decision making processes associated with carbon capture and storage (CCS) technologies. Selection of sub-surface $CO_2$ storage sites often necessitates expensive and involved simulations of $CO_2$ flow fields. Here, we develop a Fourier Neural Operator (FNO) based model for real-time, high-resolution simulation of $CO_2$ plume migration. The model is trained on a comprehensive dataset generated from realistic subsurface parameters and offers $O(10^5)$ computational acceleration with minimal sacrifice in prediction accuracy. We also explore super-resolution experiments to improve the computational cost of training the FNO based models. Additionally, we present various strategies for improving the reliability of predictions from the model, which is crucial while assessing actual geological sites. This novel framework, based on NVIDIA's Modulus library, will allow rapid screening of sites for CCS. The discussed workflows and strategies can be applied to other energy solutions like geothermal reservoir modeling and hydrogen storage. Our work scales scientific machine learning models to realistic 3D systems that are more consistent with real-life subsurface aquifers/reservoirs, paving the way for next-generation digital twins for subsurface CCS applications. 

---
# FMNet: Frequency-Assisted Mamba-Like Linear Attention Network for Camouflaged Object Detection 

**Authors**: Ming Deng, Sijin Sun, Zihao Li, Xiaochuan Hu, Xing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11030)  

**Abstract**: Camouflaged Object Detection (COD) is challenging due to the strong similarity between camouflaged objects and their surroundings, which complicates identification. Existing methods mainly rely on spatial local features, failing to capture global information, while Transformers increase computational this http URL address this, the Frequency-Assisted Mamba-Like Linear Attention Network (FMNet) is proposed, which leverages frequency-domain learning to efficiently capture global features and mitigate ambiguity between objects and the background. FMNet introduces the Multi-Scale Frequency-Assisted Mamba-Like Linear Attention (MFM) module, integrating frequency and spatial features through a multi-scale structure to handle scale variations while reducing computational complexity. Additionally, the Pyramidal Frequency Attention Extraction (PFAE) module and the Frequency Reverse Decoder (FRD) enhance semantics and reconstruct features. Experimental results demonstrate that FMNet outperforms existing methods on multiple COD datasets, showcasing its advantages in both performance and efficiency. Code available at this https URL. 

---
# From Abstraction to Reality: DARPA's Vision for Robust Sim-to-Real Autonomy 

**Authors**: Erfaun Noorani, Zachary Serlin, Ben Price, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11007)  

**Abstract**: The DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) program aims to address rapid and robust transfer of autonomy technologies across dynamic and complex environments, goals, and platforms. Existing methods for simulation-to-reality (sim-to-real) transfer often rely on high-fidelity simulations and struggle with broad adaptation, particularly in time-sensitive scenarios. Although many approaches have shown incredible performance at specific tasks, most techniques fall short when posed with unforeseen, complex, and dynamic real-world scenarios due to the inherent limitations of simulation. In contrast to current research that aims to bridge the gap between simulation environments and the real world through increasingly sophisticated simulations and a combination of methods typically assuming a small sim-to-real gap -- such as domain randomization, domain adaptation, imitation learning, meta-learning, policy distillation, and dynamic optimization -- TIAMAT takes a different approach by instead emphasizing transfer and adaptation of the autonomy stack directly to real-world environments by utilizing a breadth of low(er)-fidelity simulations to create broadly effective sim-to-real transfers. By abstractly learning from multiple simulation environments in reference to their shared semantics, TIAMAT's approaches aim to achieve abstract-to-real transfer for effective and rapid real-world adaptation. Furthermore, this program endeavors to improve the overall autonomy pipeline by addressing the inherent challenges in translating simulated behaviors into effective real-world performance. 

---
# Observation-Graph Interaction and Key-Detail Guidance for Vision and Language Navigation 

**Authors**: Yifan Xie, Binkai Ou, Fei Ma, Yaohua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11006)  

**Abstract**: Vision and Language Navigation (VLN) requires an agent to navigate through environments following natural language instructions. However, existing methods often struggle with effectively integrating visual observations and instruction details during navigation, leading to suboptimal path planning and limited success rates. In this paper, we propose OIKG (Observation-graph Interaction and Key-detail Guidance), a novel framework that addresses these limitations through two key components: (1) an observation-graph interaction module that decouples angular and visual information while strengthening edge representations in the navigation space, and (2) a key-detail guidance module that dynamically extracts and utilizes fine-grained location and object information from instructions. By enabling more precise cross-modal alignment and dynamic instruction interpretation, our approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR datasets demonstrate that OIKG achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of our method in enhancing navigation precision through better observation-instruction alignment. 

---
# RONA: Pragmatically Diverse Image Captioning with Coherence Relations 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10997)  

**Abstract**: Writing Assistants (e.g., Grammarly, Microsoft Copilot) traditionally generate diverse image captions by employing syntactic and semantic variations to describe image components. However, human-written captions prioritize conveying a central message alongside visual descriptions using pragmatic cues. To enhance pragmatic diversity, it is essential to explore alternative ways of communicating these messages in conjunction with visual content. To address this challenge, we propose RONA, a novel prompting strategy for Multi-modal Large Language Models (MLLM) that leverages Coherence Relations as an axis for variation. We demonstrate that RONA generates captions with better overall diversity and ground-truth alignment, compared to MLLM baselines across multiple domains. Our code is available at: this https URL 

---
# Image-Goal Navigation Using Refined Feature Guidance and Scene Graph Enhancement 

**Authors**: Zhicheng Feng, Xieyuanli Chen, Chenghao Shi, Lun Luo, Zhichao Chen, Yun-Hui Liu, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10986)  

**Abstract**: In this paper, we introduce a novel image-goal navigation approach, named RFSG. Our focus lies in leveraging the fine-grained connections between goals, observations, and the environment within limited image data, all the while keeping the navigation architecture simple and lightweight. To this end, we propose the spatial-channel attention mechanism, enabling the network to learn the importance of multi-dimensional features to fuse the goal and observation features. In addition, a selfdistillation mechanism is incorporated to further enhance the feature representation capabilities. Given that the navigation task needs surrounding environmental information for more efficient navigation, we propose an image scene graph to establish feature associations at both the image and object levels, effectively encoding the surrounding scene information. Crossscene performance validation was conducted on the Gibson and HM3D datasets, and the proposed method achieved stateof-the-art results among mainstream methods, with a speed of up to 53.5 frames per second on an RTX3080. This contributes to the realization of end-to-end image-goal navigation in realworld scenarios. The implementation and model of our method have been released at: this https URL. 

---
# The Problem of the Priors, or Posteriors? 

**Authors**: Hanti Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10984)  

**Abstract**: The problem of the priors is well known: it concerns the challenge of identifying norms that govern one's prior credences. I argue that a key to addressing this problem lies in considering what I call the problem of the posteriors -- the challenge of identifying norms that directly govern one's posterior credences, which then induce constraints on the priors via the diachronic requirement of conditionalization. This forward-looking approach can be summarized as: Think ahead, work backward. Although this idea can be traced to Freedman (1963), Carnap (1963), and Shimony (1970), it has received little attention in philosophy. In this paper, I initiate a systematic defense of forward-looking Bayesianism, addressing potential objections from more traditional views (both subjectivist and objectivist) and arguing for its advantages. In particular, I develop a specific approach to forward-looking Bayesianism -- one that treats the convergence of posterior credences to the truth as a fundamental rather than derived normative requirement. This approach, called convergentist Bayesianism, is argued to be crucial for a Bayesian foundation of Ockham's razor and related inference methods in statistics and machine learning. 

---
# OuroMamba: A Data-Free Quantization Framework for Vision Mamba Models 

**Authors**: Akshat Ramachandran, Mingyu Lee, Huan Xu, Souvik Kundu, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2503.10959)  

**Abstract**: We present OuroMamba, the first data-free post-training quantization (DFQ) method for vision Mamba-based models (VMMs). We identify two key challenges in enabling DFQ for VMMs, (1) VMM's recurrent state transitions restricts capturing of long-range interactions and leads to semantically weak synthetic data, (2) VMM activations exhibit dynamic outlier variations across time-steps, rendering existing static PTQ techniques ineffective. To address these challenges, OuroMamba presents a two-stage framework: (1) OuroMamba-Gen to generate semantically rich and meaningful synthetic data. It applies contrastive learning on patch level VMM features generated through neighborhood interactions in the latent state space, (2) OuroMamba-Quant to employ mixed-precision quantization with lightweight dynamic outlier detection during inference. In specific, we present a thresholding based outlier channel selection strategy for activations that gets updated every time-step. Extensive experiments across vision and generative tasks show that our data-free OuroMamba surpasses existing data-driven PTQ techniques, achieving state-of-the-art performance across diverse quantization settings. Additionally, we implement efficient GPU kernels to achieve practical latency speedup of up to 2.36x. Code will be released soon. 

---
# Predicting Stock Movement with BERTweet and Transformers 

**Authors**: Michael Charles Albada, Mojolaoluwa Joshua Sonola  

**Link**: [PDF](https://arxiv.org/pdf/2503.10957)  

**Abstract**: Applying deep learning and computational intelligence to finance has been a popular area of applied research, both within academia and industry, and continues to attract active attention. The inherently high volatility and non-stationary of the data pose substantial challenges to machine learning models, especially so for today's expressive and highly-parameterized deep learning models. Recent work has combined natural language processing on data from social media to augment models based purely on historic price data to improve performance has received particular attention. Previous work has achieved state-of-the-art performance on this task by combining techniques such as bidirectional GRUs, variational autoencoders, word and document embeddings, self-attention, graph attention, and adversarial training. In this paper, we demonstrated the efficacy of BERTweet, a variant of BERT pre-trained specifically on a Twitter corpus, and the transformer architecture by achieving competitive performance with the existing literature and setting a new baseline for Matthews Correlation Coefficient on the Stocknet dataset without auxiliary data sources. 

---
# Empirical Computation 

**Authors**: Eric Tang, Marcel Böhme  

**Link**: [PDF](https://arxiv.org/pdf/2503.10954)  

**Abstract**: In this vision paper, we explore the challenges and opportunities of a form of computation that employs an empirical (rather than a formal) approach, where the solution of a computational problem is returned as empirically most likely (rather than necessarily correct). We call this approach as *empirical computation* and observe that its capabilities and limits *cannot* be understood within the classic, rationalist framework of computation.
While we take a very broad view of "computational problem", a classic, well-studied example is *sorting*: Given a set of $n$ numbers, return these numbers sorted in ascending order.
* To run a classical, *formal computation*, we might first think about a *specific algorithm* (e.g., merge sort) before developing a *specific* program that implements it. The program will expect the input to be given in a *specific* format, type, or data structure (e.g., unsigned 32-bit integers). In software engineering, we have many approaches to analyze the correctness of such programs. From complexity theory, we know that there exists no correct program that can solve the average instance of the sorting problem faster than $O(n\log n)$.
* To run an *empirical computation*, we might directly ask a large language model (LLM) to solve *any* computational problem (which can be stated informally in natural language) and provide the input in *any* format (e.g., negative numbers written as Chinese characters). There is no (problem-specific) program that could be analyzed for correctness. Also, the time it takes an LLM to return an answer is entirely *independent* of the computational complexity of the problem that is solved.
What are the capabilities or limits of empirical computation in the general, in the problem-, or in the instance-specific? Our purpose is to establish empirical computation as a field in SE that is timely and rich with interesting problems. 

---
# Safe Continual Domain Adaptation after Sim2Real Transfer of Reinforcement Learning Policies in Robotics 

**Authors**: Josip Josifovski, Shangding Gu, Mohammadhossein Malmir, Haoliang Huang, Sayantan Auddy, Nicolás Navarro-Guerrero, Costas Spanos, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.10949)  

**Abstract**: Domain randomization has emerged as a fundamental technique in reinforcement learning (RL) to facilitate the transfer of policies from simulation to real-world robotic applications. Many existing domain randomization approaches have been proposed to improve robustness and sim2real transfer. These approaches rely on wide randomization ranges to compensate for the unknown actual system parameters, leading to robust but inefficient real-world policies. In addition, the policies pretrained in the domain-randomized simulation are fixed after deployment due to the inherent instability of the optimization processes based on RL and the necessity of sampling exploitative but potentially unsafe actions on the real system. This limits the adaptability of the deployed policy to the inevitably changing system parameters or environment dynamics over time. We leverage safe RL and continual learning under domain-randomized simulation to address these limitations and enable safe deployment-time policy adaptation in real-world robot control. The experiments show that our method enables the policy to adapt and fit to the current domain distribution and environment dynamics of the real system while minimizing safety risks and avoiding issues like catastrophic forgetting of the general policy found in randomized simulation during the pretraining phase. Videos and supplementary material are available at this https URL. 

---
# $(\varepsilon, δ)$ Considered Harmful: Best Practices for Reporting Differential Privacy Guarantees 

**Authors**: Juan Felipe Gomez, Bogdan Kulynych, Georgios Kaissis, Jamie Hayes, Borja Balle, Antti Honkela  

**Link**: [PDF](https://arxiv.org/pdf/2503.10945)  

**Abstract**: Current practices for reporting the level of differential privacy (DP) guarantees for machine learning (ML) algorithms provide an incomplete and potentially misleading picture of the guarantees and make it difficult to compare privacy levels across different settings. We argue for using Gaussian differential privacy (GDP) as the primary means of communicating DP guarantees in ML, with the full privacy profile as a secondary option in case GDP is too inaccurate. Unlike other widely used alternatives, GDP has only one parameter, which ensures easy comparability of guarantees, and it can accurately capture the full privacy profile of many important ML applications. To support our claims, we investigate the privacy profiles of state-of-the-art DP large-scale image classification, and the TopDown algorithm for the U.S. Decennial Census, observing that GDP fits the profiles remarkably well in all three cases. Although GDP is ideal for reporting the final guarantees, other formalisms (e.g., privacy loss random variables) are needed for accurate privacy accounting. We show that such intermediate representations can be efficiently converted to GDP with minimal loss in tightness. 

---
# ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models 

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch  

**Link**: [PDF](https://arxiv.org/pdf/2503.10937)  

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications. 

---
# OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Miguel Barreda-Ángeles, Sergi Abadal, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.10927)  

**Abstract**: While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available. 

---
# Predicting Clinical Outcomes with Waveform LSTMs 

**Authors**: Michael Albada  

**Link**: [PDF](https://arxiv.org/pdf/2503.10925)  

**Abstract**: Data mining and machine learning hold great potential to enable health systems to systematically use data and analytics to identify inefficiencies and best practices that improve care and reduce costs. Waveform data offers particularly detailed information on how patient health evolves over time and has the potential to significantly improve prediction accuracy on multiple benchmarks, but has been widely under-utilized, largely because of the challenges in working with these large and complex datasets. This study evaluates the potential of leveraging clinical waveform data to improve prediction accuracy on a single benchmark task: the risk of mortality in the intensive care unit. We identify significant potential from this data, beating the existing baselines for both logistic regression and deep learning models. 

---
# Resource Heterogeneity-Aware and Utilization-Enhanced Scheduling for Deep Learning Clusters 

**Authors**: Abeda Sultana, Nabin Pakka, Fei Xu, Xu Yuan, Li Chen, Nian-Feng Tzeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10918)  

**Abstract**: Scheduling deep learning (DL) models to train on powerful clusters with accelerators like GPUs and TPUs, presently falls short, either lacking fine-grained heterogeneity awareness or leaving resources substantially under-utilized. To fill this gap, we propose a novel design of a task-level heterogeneity-aware scheduler, {\em Hadar}, based on an optimization framework that can boost resource utilization. {\em Hadar} leverages the performance traits of DL jobs on a heterogeneous DL cluster, characterizes the task-level performance heterogeneity in the optimization problem, and makes scheduling decisions across both spatial and temporal dimensions. %with the objective to reduce the average job completion time of DL jobs. It involves the primal-dual framework employing a dual subroutine, to solve the optimization problem and guide the scheduling design. Our trace-driven simulation with representative DL model training workloads demonstrates that {\em Hadar} accelerates the total time duration by 1.20$\times$ when compared with its state-of-the-art heterogeneity-aware counterpart, Gavel. Further, our {\em Hadar} scheduler is enhanced to {\em HadarE} by forking each job into multiple copies to let a job train concurrently on heterogeneous GPUs resided on separate available nodes (i.e., machines or servers) for resource utilization enhancement. {\em HadarE} is evaluated extensively on physical DL clusters for comparison with {\em Hadar} and Gavel. With substantial enhancement in cluster resource utilization (by 1.45$\times$), {\em HadarE} exhibits considerable speed-ups in DL model training, reducing the total time duration by 50\% (or 80\%) on an Amazon's AWS (or our lab) cluster, while producing trained DL models with consistently better inference quality than those trained by \textit{Hadar}. 

---
# JPEG Compliant Compression for Both Human and Machine, A Report 

**Authors**: Linfeng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.10912)  

**Abstract**: Deep Neural Networks (DNNs) have become an integral part of our daily lives, especially in vision-related applications. However, the conventional lossy image compression algorithms are primarily designed for the Human Vision System (HVS), which can non-trivially compromise the DNNs' validation accuracy after compression, as noted in \cite{liu2018deepn}. Thus developing an image compression algorithm for both human and machine (DNNs) is on the horizon.
To address the challenge mentioned above, in this paper, we first formulate the image compression as a multi-objective optimization problem which take both human and machine prespectives into account, then we solve it by linear combination, and proposed a novel distortion measure for both human and machine, dubbed Human and Machine-Oriented Error (HMOE). After that, we develop Human And Machine Oriented Soft Decision Quantization (HMOSDQ) based on HMOE, a lossy image compression algorithm for both human and machine (DNNs), and fully complied with JPEG format. In order to evaluate the performance of HMOSDQ, finally we conduct the experiments for two pre-trained well-known DNN-based image classifiers named Alexnet \cite{Alexnet} and VGG-16 \cite{simonyan2014VGG} on two subsets of the ImageNet \cite{deng2009imagenet} validation set: one subset included images with shorter side in the range of 496 to 512, while the other included images with shorter side in the range of 376 to 384. Our results demonstrate that HMOSDQ outperforms the default JPEG algorithm in terms of rate-accuracy and rate-distortion performance. For the Alexnet comparing with the default JPEG algorithm, HMOSDQ can improve the validation accuracy by more than $0.81\%$ at $0.61$ BPP, or equivalently reduce the compression rate of default JPEG by $9.6\times$ while maintaining the same validation accuracy. 

---
# Ecological Neural Architecture Search 

**Authors**: Benjamin David Winter, William J. Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10908)  

**Abstract**: When employing an evolutionary algorithm to optimize a neural networks architecture, developers face the added challenge of tuning the evolutionary algorithm's own hyperparameters - population size, mutation rate, cloning rate, and number of generations. This paper introduces Neuvo Ecological Neural Architecture Search (ENAS), a novel method that incorporates these evolutionary parameters directly into the candidate solutions' phenotypes, allowing them to evolve dynamically alongside architecture specifications. Experimental results across four binary classification datasets demonstrate that ENAS not only eliminates manual tuning of evolutionary parameters but also outperforms competitor NAS methodologies in convergence speed (reducing computational time by 18.3%) and accuracy (improving classification performance in 3 out of 4 datasets). By enabling "greedy individuals" to optimize resource allocation based on fitness, ENAS provides an efficient, self-regulating approach to neural architecture search. 

---
# H2-MARL: Multi-Agent Reinforcement Learning for Pareto Optimality in Hospital Capacity Strain and Human Mobility during Epidemic 

**Authors**: Xueting Luo, Hao Deng, Jihong Yang, Yao Shen, Huanhuan Guo, Zhiyuan Sun, Mingqing Liu, Jiming Wei, Shengjie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10907)  

**Abstract**: The necessity of achieving an effective balance between minimizing the losses associated with restricting human mobility and ensuring hospital capacity has gained significant attention in the aftermath of COVID-19. Reinforcement learning (RL)-based strategies for human mobility management have recently advanced in addressing the dynamic evolution of cities and epidemics; however, they still face challenges in achieving coordinated control at the township level and adapting to cities of varying scales. To address the above issues, we propose a multi-agent RL approach that achieves Pareto optimality in managing hospital capacity and human mobility (H2-MARL), applicable across cities of different scales. We first develop a township-level infection model with online-updatable parameters to simulate disease transmission and construct a city-wide dynamic spatiotemporal epidemic simulator. On this basis, H2-MARL is designed to treat each division as an agent, with a trade-off dual-objective reward function formulated and an experience replay buffer enriched with expert knowledge built. To evaluate the effectiveness of the model, we construct a township-level human mobility dataset containing over one billion records from four representative cities of varying scales. Extensive experiments demonstrate that H2-MARL has the optimal dual-objective trade-off capability, which can minimize hospital capacity strain while minimizing human mobility restriction loss. Meanwhile, the applicability of the proposed model to epidemic control in cities of varying scales is verified, which showcases its feasibility and versatility in practical applications. 

---
# HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks 

**Authors**: Jiuding Sun, Jing Huang, Sidharth Baskaran, Karel D'Oosterlinck, Christopher Potts, Michael Sklar, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2503.10894)  

**Abstract**: Mechanistic interpretability has made great strides in identifying neural network features (e.g., directions in hidden activation space) that mediate concepts(e.g., the birth year of a person) and enable predictable manipulation. Distributed alignment search (DAS) leverages supervision from counterfactual data to learn concept features within hidden states, but DAS assumes we can afford to conduct a brute force search over potential feature locations. To address this, we present HyperDAS, a transformer-based hypernetwork architecture that (1) automatically locates the token-positions of the residual stream that a concept is realized in and (2) constructs features of those residual stream vectors for the concept. In experiments with Llama3-8B, HyperDAS achieves state-of-the-art performance on the RAVEL benchmark for disentangling concepts in hidden states. In addition, we review the design decisions we made to mitigate the concern that HyperDAS (like all powerful interpretabilty methods) might inject new information into the target model rather than faithfully interpreting it. 

---
# Taxonomic Reasoning for Rare Arthropods: Combining Dense Image Captioning and RAG for Interpretable Classification 

**Authors**: Nathaniel Lesperance, Sujeevan Ratnasingham, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.10886)  

**Abstract**: In the context of pressing climate change challenges and the significant biodiversity loss among arthropods, automated taxonomic classification from organismal images is a subject of intense research. However, traditional AI pipelines based on deep neural visual architectures such as CNNs or ViTs face limitations such as degraded performance on the long-tail of classes and the inability to reason about their predictions. We integrate image captioning and retrieval-augmented generation (RAG) with large language models (LLMs) to enhance biodiversity monitoring, showing particular promise for characterizing rare and unknown arthropod species. While a naive Vision-Language Model (VLM) excels in classifying images of common species, the RAG model enables classification of rarer taxa by matching explicit textual descriptions of taxonomic features to contextual biodiversity text data from external sources. The RAG model shows promise in reducing overconfidence and enhancing accuracy relative to naive LLMs, suggesting its viability in capturing the nuances of taxonomic hierarchy, particularly at the challenging family and genus levels. Our findings highlight the potential for modern vision-language AI pipelines to support biodiversity conservation initiatives, emphasizing the role of comprehensive data curation and collaboration with citizen science platforms to improve species identification, unknown species characterization and ultimately inform conservation strategies. 

---
# Task-Specific Activation Functions for Neuroevolution using Grammatical Evolution 

**Authors**: Benjamin David Winter, William John Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10879)  

**Abstract**: Activation functions play a critical role in the performance and behaviour of neural networks, significantly impacting their ability to learn and generalise. Traditional activation functions, such as ReLU, sigmoid, and tanh, have been widely used with considerable success. However, these functions may not always provide optimal performance for all tasks and datasets. In this paper, we introduce Neuvo GEAF - an innovative approach leveraging grammatical evolution (GE) to automatically evolve novel activation functions tailored to specific neural network architectures and datasets. Experiments conducted on well-known binary classification datasets show statistically significant improvements in F1-score (between 2.4% and 9.4%) over ReLU using identical network architectures. Notably, these performance gains were achieved without increasing the network's parameter count, supporting the trend toward more efficient neural networks that can operate effectively on resource-constrained edge devices. This paper's findings suggest that evolved activation functions can provide significant performance improvements for compact networks while maintaining energy efficiency during both training and inference phases. 

---
# TAIJI: Textual Anchoring for Immunizing Jailbreak Images in Vision Language Models 

**Authors**: Xiangyu Yin, Yi Qi, Jinwei Hu, Zhen Chen, Yi Dong, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10872)  

**Abstract**: Vision Language Models (VLMs) have demonstrated impressive inference capabilities, but remain vulnerable to jailbreak attacks that can induce harmful or unethical responses. Existing defence methods are predominantly white-box approaches that require access to model parameters and extensive modifications, making them costly and impractical for many real-world scenarios. Although some black-box defences have been proposed, they often impose input constraints or require multiple queries, limiting their effectiveness in safety-critical tasks such as autonomous driving. To address these challenges, we propose a novel black-box defence framework called \textbf{T}extual \textbf{A}nchoring for \textbf{I}mmunizing \textbf{J}ailbreak \textbf{I}mages (\textbf{TAIJI}). TAIJI leverages key phrase-based textual anchoring to enhance the model's ability to assess and mitigate the harmful content embedded within both visual and textual prompts. Unlike existing methods, TAIJI operates effectively with a single query during inference, while preserving the VLM's performance on benign tasks. Extensive experiments demonstrate that TAIJI significantly enhances the safety and reliability of VLMs, providing a practical and efficient solution for real-world deployment. 

---
# Evaluating a Novel Neuroevolution and Neural Architecture Search System 

**Authors**: Benjamin David Winter, William John Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10869)  

**Abstract**: The choice of neural network features can have a large impact on both the accuracy and speed of the network. Despite the current industry shift towards large transformer models, specialized binary classifiers remain critical for numerous practical applications where computational efficiency and low latency are essential. Neural network features tend to be developed homogeneously, resulting in slower or less accurate networks when testing against multiple datasets. In this paper, we show the effectiveness of Neuvo NAS+ a novel Python implementation of an extended Neural Architecture Search (NAS+) which allows the user to optimise the training parameters of a network as well as the network's architecture. We provide an in-depth analysis of the importance of catering a network's architecture to each dataset. We also describe the design of the Neuvo NAS+ system that selects network features on a task-specific basis including network training hyper-parameters such as the number of epochs and batch size. Results show that the Neuvo NAS+ task-specific approach significantly outperforms several machine learning approaches such as Naive Bayes, C4.5, Support Vector Machine and a standard Artificial Neural Network for solving a range of binary classification problems in terms of accuracy. Our experiments demonstrate substantial diversity in evolved network architectures across different datasets, confirming the value of task-specific optimization. Additionally, Neuvo NAS+ outperforms other evolutionary algorithm optimisers in terms of both accuracy and computational efficiency, showing that properly optimized binary classifiers can match or exceed the performance of more complex models while requiring significantly fewer computational resources. 

---
# Towards Understanding Graphical Perception in Large Multimodal Models 

**Authors**: Kai Zhang, Jianwei Yang, Jeevana Priya Inala, Chandan Singh, Jianfeng Gao, Yu Su, Chenglong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10857)  

**Abstract**: Despite the promising results of large multimodal models (LMMs) in complex vision-language tasks that require knowledge, reasoning, and perception abilities together, we surprisingly found that these models struggle with simple tasks on infographics that require perception only. As existing benchmarks primarily focus on end tasks that require various abilities, they provide limited, fine-grained insights into the limitations of the models' perception abilities. To address this gap, we leverage the theory of graphical perception, an approach used to study how humans decode visual information encoded on charts and graphs, to develop an evaluation framework for analyzing gaps in LMMs' perception abilities in charts. With automated task generation and response evaluation designs, our framework enables comprehensive and controlled testing of LMMs' graphical perception across diverse chart types, visual elements, and task types. We apply our framework to evaluate and diagnose the perception capabilities of state-of-the-art LMMs at three granularity levels (chart, visual element, and pixel). Our findings underscore several critical limitations of current state-of-the-art LMMs, including GPT-4o: their inability to (1) generalize across chart types, (2) understand fundamental visual elements, and (3) cross reference values within a chart. These insights provide guidance for future improvements in perception abilities of LMMs. The evaluation framework and labeled data are publicly available at this https URL. 

---
# Byzantine-Resilient Federated Learning via Distributed Optimization 

**Authors**: Yufei Xia, Wenrui Yu, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10792)  

**Abstract**: Byzantine attacks present a critical challenge to Federated Learning (FL), where malicious participants can disrupt the training process, degrade model accuracy, and compromise system reliability. Traditional FL frameworks typically rely on aggregation-based protocols for model updates, leaving them vulnerable to sophisticated adversarial strategies. In this paper, we demonstrate that distributed optimization offers a principled and robust alternative to aggregation-centric methods. Specifically, we show that the Primal-Dual Method of Multipliers (PDMM) inherently mitigates Byzantine impacts by leveraging its fault-tolerant consensus mechanism. Through extensive experiments on three datasets (MNIST, FashionMNIST, and Olivetti), under various attack scenarios including bit-flipping and Gaussian noise injection, we validate the superior resilience of distributed optimization protocols. Compared to traditional aggregation-centric approaches, PDMM achieves higher model utility, faster convergence, and improved stability. Our results highlight the effectiveness of distributed optimization in defending against Byzantine threats, paving the way for more secure and resilient federated learning systems. 

---
# Vulnerability Detection: From Formal Verification to Large Language Models and Hybrid Approaches: A Comprehensive Overview 

**Authors**: Norbert Tihanyi, Tamas Bisztray, Mohamed Amine Ferrag, Bilel Cherif, Richard A. Dubniczky, Ridhi Jain, Lucas C. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.10784)  

**Abstract**: Software testing and verification are critical for ensuring the reliability and security of modern software systems. Traditionally, formal verification techniques, such as model checking and theorem proving, have provided rigorous frameworks for detecting bugs and vulnerabilities. However, these methods often face scalability challenges when applied to complex, real-world programs. Recently, the advent of Large Language Models (LLMs) has introduced a new paradigm for software analysis, leveraging their ability to understand insecure coding practices. Although LLMs demonstrate promising capabilities in tasks such as bug prediction and invariant generation, they lack the formal guarantees of classical methods. This paper presents a comprehensive study of state-of-the-art software testing and verification, focusing on three key approaches: classical formal methods, LLM-based analysis, and emerging hybrid techniques, which combine their strengths. We explore each approach's strengths, limitations, and practical applications, highlighting the potential of hybrid systems to address the weaknesses of standalone methods. We analyze whether integrating formal rigor with LLM-driven insights can enhance the effectiveness and scalability of software verification, exploring their viability as a pathway toward more robust and adaptive testing frameworks. 

---
# Unifying 2D and 3D Vision-Language Understanding 

**Authors**: Ayush Jain, Alexander Swerdlow, Yuzhou Wang, Sergio Arnaud, Ada Martin, Alexander Sax, Franziska Meier, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.10745)  

**Abstract**: Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at $\href{this https URL}{this http URL}$. 

---
# Predicting Treatment Response in Body Dysmorphic Disorder with Interpretable Machine Learning 

**Authors**: Omar Costilla-Reyes, Morgan Talbot  

**Link**: [PDF](https://arxiv.org/pdf/2503.10741)  

**Abstract**: Body Dysmorphic Disorder (BDD) is a highly prevalent and frequently underdiagnosed condition characterized by persistent, intrusive preoccupations with perceived defects in physical appearance. In this extended analysis, we employ multiple machine learning approaches to predict treatment outcomes -- specifically treatment response and remission -- with an emphasis on interpretability to ensure clinical relevance and utility. Across the various models investigated, treatment credibility emerged as the most potent predictor, surpassing traditional markers such as baseline symptom severity or comorbid conditions. Notably, while simpler models (e.g., logistic regression and support vector machines) achieved competitive predictive performance, decision tree analyses provided unique insights by revealing clinically interpretable threshold values in credibility scores. These thresholds can serve as practical guideposts for clinicians when tailoring interventions or allocating treatment resources. We further contextualize our findings within the broader literature on BDD, addressing technology-based therapeutics, digital interventions, and the psychosocial determinants of treatment engagement. An extensive array of references situates our results within current research on BDD prevalence, suicidality risks, and digital innovation. Our work underscores the potential of integrating rigorous statistical methodologies with transparent machine learning models. By systematically identifying modifiable predictors -- such as treatment credibility -- we propose a pathway toward more targeted, personalized, and ultimately efficacious interventions for individuals with BDD. 

---
# Commenting Higher-level Code Unit: Full Code, Reduced Code, or Hierarchical Code Summarization 

**Authors**: Weisong Sun, Yiran Zhang, Jie Zhu, Zhihui Wang, Chunrong Fang, Yonglong Zhang, Yebo Feng, Jiangping Huang, Xingya Wang, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10737)  

**Abstract**: Commenting code is a crucial activity in software development, as it aids in facilitating future maintenance and updates. To enhance the efficiency of writing comments and reduce developers' workload, researchers has proposed various automated code summarization (ACS) techniques to automatically generate comments/summaries for given code units. However, these ACS techniques primarily focus on generating summaries for code units at the method level. There is a significant lack of research on summarizing higher-level code units, such as file-level and module-level code units, despite the fact that summaries of these higher-level code units are highly useful for quickly gaining a macro-level understanding of software components and architecture. To fill this gap, in this paper, we conduct a systematic study on how to use LLMs for commenting higher-level code units, including file level and module level. These higher-level units are significantly larger than method-level ones, which poses challenges in handling long code inputs within LLM constraints and maintaining efficiency. To address these issues, we explore various summarization strategies for ACS of higher-level code units, which can be divided into three types: full code summarization, reduced code summarization, and hierarchical code summarization. The experimental results suggest that for summarizing file-level code units, using the full code is the most effective approach, with reduced code serving as a cost-efficient alternative. However, for summarizing module-level code units, hierarchical code summarization becomes the most promising strategy. In addition, inspired by the research on method-level ACS, we also investigate using the LLM as an evaluator to evaluate the quality of summaries of higher-level code units. The experimental results demonstrate that the LLM's evaluation results strongly correlate with human evaluations. 

---
# OCPM$^2$: Extending the Process Mining Methodology for Object-Centric Event Data Extraction 

**Authors**: Najmeh Miri, Shahrzad Khayatbashi, Jelena Zdravkovic, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2503.10735)  

**Abstract**: Object-Centric Process Mining (OCPM) enables business process analysis from multiple perspectives. For example, an educational path can be examined from the viewpoints of students, teachers, and groups. This analysis depends on Object-Centric Event Data (OCED), which captures relationships between events and object types, representing different perspectives. Unlike traditional process mining techniques, extracting OCED minimizes the need for repeated log extractions when shifting the analytical focus. However, recording these complex relationships increases the complexity of the log extraction process. To address this challenge, this paper proposes a method for extracting OCED based on PM\inst{2}, a well-established process mining framework. Our approach introduces a structured framework that guides data analysts and engineers in extracting OCED for process analysis. We validate this framework by applying it in a real-world educational setting, demonstrating its effectiveness in extracting an Object-Centric Event Log (OCEL), which serves as the standard format for recording OCED, from a learning management system and an administrative grading system. 

---
# DarkBench: Benchmarking Dark Patterns in Large Language Models 

**Authors**: Esben Kran, Hieu Minh "Jord" Nguyen, Akash Kundu, Sami Jawhar, Jinsuk Park, Mateusz Maria Jurewicz  

**Link**: [PDF](https://arxiv.org/pdf/2503.10728)  

**Abstract**: We introduce DarkBench, a comprehensive benchmark for detecting dark design patterns--manipulative techniques that influence user behavior--in interactions with large language models (LLMs). Our benchmark comprises 660 prompts across six categories: brand bias, user retention, sycophancy, anthropomorphism, harmful generation, and sneaking. We evaluate models from five leading companies (OpenAI, Anthropic, Meta, Mistral, Google) and find that some LLMs are explicitly designed to favor their developers' products and exhibit untruthful communication, among other manipulative behaviors. Companies developing LLMs should recognize and mitigate the impact of dark design patterns to promote more ethical AI. 

---
# Word-level Annotation of GDPR Transparency Compliance in Privacy Policies using Large Language Models 

**Authors**: Thomas Cory, Wolf Rieder, Julia Krämer, Philip Raschke, Patrick Herbke, Axel Küpper  

**Link**: [PDF](https://arxiv.org/pdf/2503.10727)  

**Abstract**: Ensuring transparency of data practices related to personal information is a fundamental requirement under the General Data Protection Regulation (GDPR), particularly as mandated by Articles 13 and 14. However, assessing compliance at scale remains a challenge due to the complexity and variability of privacy policy language. Manual audits are resource-intensive and inconsistent, while existing automated approaches lack the granularity needed to capture nuanced transparency disclosures.
In this paper, we introduce a large language model (LLM)-based framework for word-level GDPR transparency compliance annotation. Our approach comprises a two-stage annotation pipeline that combines initial LLM-based annotation with a self-correction mechanism for iterative refinement. This annotation pipeline enables the systematic identification and fine-grained annotation of transparency-related content in privacy policies, aligning with 21 GDPR-derived transparency requirements. To enable large-scale analysis, we compile a dataset of 703,791 English-language policies, from which we generate a sample of 200 manually annotated privacy policies.
To evaluate our approach, we introduce a two-tiered methodology assessing both label- and span-level annotation performance. We conduct a comparative analysis of eight high-profile LLMs, providing insights into their effectiveness in identifying GDPR transparency disclosures. Our findings contribute to advancing the automation of GDPR compliance assessments and provide valuable resources for future research in privacy policy analysis. 

---
# Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores 

**Authors**: Chenpeng Wu, Qiqi Gu, Heng Shi, Jianguo Yao, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10725)  

**Abstract**: The escalating size of Mixture-of-Experts (MoE) based Large Language Models (LLMs) presents significant computational and memory challenges, necessitating innovative solutions to enhance efficiency without compromising model accuracy. Structured sparsity emerges as a compelling strategy to address these challenges by leveraging the emerging sparse computing hardware. Prior works mainly focus on the sparsity in model parameters, neglecting the inherent sparse patterns in activations. This oversight can lead to additional computational costs associated with activations, potentially resulting in suboptimal performance.
This paper presents Samoyeds, an innovative acceleration system for MoE LLMs utilizing Sparse Tensor Cores (SpTCs). Samoyeds is the first to apply sparsity simultaneously to both activations and model parameters. It introduces a bespoke sparse data format tailored for MoE computation and develops a specialized sparse-sparse matrix multiplication kernel. Furthermore, Samoyeds incorporates systematic optimizations specifically designed for the execution of dual-side structured sparse MoE LLMs on SpTCs, further enhancing system performance. Evaluations show that Samoyeds outperforms SOTA works by up to 1.99$\times$ at the kernel level and 1.58$\times$ at the model level. Moreover, it enhances memory efficiency, increasing maximum supported batch sizes by 4.41$\times$ on average. Additionally, Samoyeds surpasses existing SOTA structured sparse solutions in both model accuracy and hardware portability. 

---
# RankPO: Preference Optimization for Job-Talent Matching 

**Authors**: Yafei Zhang, Murray Wang, Yu Wang, Xiaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10723)  

**Abstract**: Matching job descriptions (JDs) with suitable talent requires models capable of understanding not only textual similarities between JDs and candidate resumes but also contextual factors such as geographical location and academic seniority. To address this challenge, we propose a two-stage training framework for large language models (LLMs). In the first stage, a contrastive learning approach is used to train the model on a dataset constructed from real-world matching rules, such as geographical alignment and research area overlap. While effective, this model primarily learns patterns that defined by the matching rules. In the second stage, we introduce a novel preference-based fine-tuning method inspired by Direct Preference Optimization (DPO), termed Rank Preference Optimization (RankPO), to align the model with AI-curated pairwise preferences emphasizing textual understanding. Our experiments show that while the first-stage model achieves strong performance on rule-based data (nDCG@20 = 0.706), it lacks robust textual understanding (alignment with AI annotations = 0.46). By fine-tuning with RankPO, we achieve a balanced model that retains relatively good performance in the original tasks while significantly improving the alignment with AI preferences. The code and data are available at this https URL. 

---
# TacticExpert: Spatial-Temporal Graph Language Model for Basketball Tactics 

**Authors**: Xu Lingrui, Liu Mandi, Zhang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10722)  

**Abstract**: The core challenge in basketball tactic modeling lies in efficiently extracting complex spatial-temporal dependencies from historical data and accurately predicting various in-game events. Existing state-of-the-art (SOTA) models, primarily based on graph neural networks (GNNs), encounter difficulties in capturing long-term, long-distance, and fine-grained interactions among heterogeneous player nodes, as well as in recognizing interaction patterns. Additionally, they exhibit limited generalization to untrained downstream tasks and zero-shot scenarios. In this work, we propose a Spatial-Temporal Propagation Symmetry-Aware Graph Transformer for fine-grained game modeling. This architecture explicitly captures delay effects in the spatial space to enhance player node representations across discrete-time slices, employing symmetry-invariant priors to guide the attention mechanism. We also introduce an efficient contrastive learning strategy to train a Mixture of Tactics Experts module, facilitating differentiated modeling of offensive tactics. By integrating dense training with sparse inference, we achieve a 2.4x improvement in model efficiency. Moreover, the incorporation of Lightweight Graph Grounding for Large Language Models enables robust performance in open-ended downstream tasks and zero-shot scenarios, including novel teams or players. The proposed model, TacticExpert, delineates a vertically integrated large model framework for basketball, unifying pretraining across multiple datasets and downstream prediction tasks. Fine-grained modeling modules significantly enhance spatial-temporal representations, and visualization analyzes confirm the strong interpretability of the model. 

---
# From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution 

**Authors**: Zhe Zhao, Haibin Wen, Pengkun Wang, Ye Wei, Zaixi Zhang, Xi Lin, Fei Liu, Bo An, Hui Xiong, Yang Wang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10721)  

**Abstract**: Large language models (LLMs) have greatly accelerated the automation of algorithm generation and optimization. However, current methods such as EoH and FunSearch mainly rely on predefined templates and expert-specified functions that focus solely on the local evolution of key functionalities. Consequently, they fail to fully leverage the synergistic benefits of the overall architecture and the potential of global optimization. In this paper, we introduce an end-to-end algorithm generation and optimization framework based on LLMs. Our approach utilizes the deep semantic understanding of LLMs to convert natural language requirements or human-authored papers into code solutions, and employs a two-dimensional co-evolution strategy to optimize both functional and structural aspects. This closed-loop process spans problem analysis, code generation, and global optimization, automatically identifying key algorithm modules for multi-level joint optimization and continually enhancing performance and design innovation. Extensive experiments demonstrate that our method outperforms traditional local optimization approaches in both performance and innovation, while also exhibiting strong adaptability to unknown environments and breakthrough potential in structural design. By building on human research, our framework generates and optimizes novel algorithms that surpass those designed by human experts, broadening the applicability of LLMs for algorithm design and providing a novel solution pathway for automated algorithm development. 

---
# AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10720)  

**Abstract**: While RAG demonstrates remarkable capabilities in LLM applications, its effectiveness is hindered by the ever-increasing length of retrieved contexts, which introduces information redundancy and substantial computational overhead. Existing context pruning methods, such as LLMLingua, lack contextual awareness and offer limited flexibility in controlling compression rates, often resulting in either insufficient pruning or excessive information loss. In this paper, we propose AttentionRAG, an attention-guided context pruning method for RAG systems. The core idea of AttentionRAG lies in its attention focus mechanism, which reformulates RAG queries into a next-token prediction paradigm. This mechanism isolates the query's semantic focus to a single token, enabling precise and efficient attention calculation between queries and retrieved contexts. Extensive experiments on LongBench and Babilong benchmarks show that AttentionRAG achieves up to 6.3$\times$ context compression while outperforming LLMLingua methods by around 10\% in key metrics. 

---
# Team NYCU at Defactify4: Robust Detection and Source Identification of AI-Generated Images Using CNN and CLIP-Based Models 

**Authors**: Tsan-Tsung Yang, I-Wei Chen, Kuan-Ting Chen, Shang-Hsuan Chiang, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10718)  

**Abstract**: With the rapid advancement of generative AI, AI-generated images have become increasingly realistic, raising concerns about creativity, misinformation, and content authenticity. Detecting such images and identifying their source models has become a critical challenge in ensuring the integrity of digital media. This paper tackles the detection of AI-generated images and identifying their source models using CNN and CLIP-ViT classifiers. For the CNN-based classifier, we leverage EfficientNet-B0 as the backbone and feed with RGB channels, frequency features, and reconstruction errors, while for CLIP-ViT, we adopt a pretrained CLIP image encoder to extract image features and SVM to perform classification. Evaluated on the Defactify 4 dataset, our methods demonstrate strong performance in both tasks, with CLIP-ViT showing superior robustness to image perturbations. Compared to baselines like AEROBLADE and OCC-CLIP, our approach achieves competitive results. Notably, our method ranked Top-3 overall in the Defactify 4 competition, highlighting its effectiveness and generalizability. All of our implementations can be found in this https URL 

---
# Deep Learning-Based Automated Workflow for Accurate Segmentation and Measurement of Abdominal Organs in CT Scans 

**Authors**: Praveen Shastry, Ashok Sharma, Kavya Mohan, Naveen Kumarasami, Anandakumar D, Mounigasri M, Keerthana R, Kishore Prasath Venkatesh, Bargava Subramanian, Kalyan Sivasailam  

**Link**: [PDF](https://arxiv.org/pdf/2503.10717)  

**Abstract**: Background: Automated analysis of CT scans for abdominal organ measurement is crucial for improving diagnostic efficiency and reducing inter-observer variability. Manual segmentation and measurement of organs such as the kidneys, liver, spleen, and prostate are time-consuming and subject to inconsistency, underscoring the need for automated approaches.
Purpose: The purpose of this study is to develop and validate an automated workflow for the segmentation and measurement of abdominal organs in CT scans using advanced deep learning models, in order to improve accuracy, reliability, and efficiency in clinical evaluations.
Methods: The proposed workflow combines nnU-Net, U-Net++ for organ segmentation, followed by a 3D RCNN model for measuring organ volumes and dimensions. The models were trained and evaluated on CT datasets with metrics such as precision, recall, and Mean Squared Error (MSE) to assess performance. Segmentation quality was verified for its adaptability to variations in patient anatomy and scanner settings.
Results: The developed workflow achieved high precision and recall values, exceeding 95 for all targeted organs. The Mean Squared Error (MSE) values were low, indicating a high level of consistency between predicted and ground truth measurements. The segmentation and measurement pipeline demonstrated robust performance, providing accurate delineation and quantification of the kidneys, liver, spleen, and prostate.
Conclusion: The proposed approach offers an automated, efficient, and reliable solution for abdominal organ measurement in CT scans. By significantly reducing manual intervention, this workflow enhances measurement accuracy and consistency, with potential for widespread clinical implementation. Future work will focus on expanding the approach to other organs and addressing complex pathological cases. 

---
# ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs 

**Authors**: Xin Liu, Pei Liu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10714)  

**Abstract**: The linear growth of key-value (KV) cache memory and quadratic computational complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often suffer from irreversible information loss or require costly parameter retraining. We propose ZeroMerge, a dynamic zero-shot compression framework that achieves efficient cache management through three key innovations: (1) Fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) A residual merging mechanism that preserves critical context through compensated attention scoring, and (3) Parameter-free adaptation compatible with diverse LLM architectures without retraining. Comprehensive evaluations across LLaMA-2 model demonstrate that ZeroMerge maintains full-cache performance at 5\% compression ratios while doubling inference throughput at 40K token lengths. The method effectively balances memory efficiency, generation quality, and deployment flexibility, advancing practical long-context LLM applications. The code is available at this https URL. 

---
# HiCMamba: Enhancing Hi-C Resolution and Identifying 3D Genome Structures with State Space Modeling 

**Authors**: Minghao Yang, Zhi-An Huang, Zhihang Zheng, Yuqiao Liu, Shichen Zhang, Pengfei Zhang, Hui Xiong, Shaojun Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10713)  

**Abstract**: Hi-C technology measures genome-wide interaction frequencies, providing a powerful tool for studying the 3D genomic structure within the nucleus. However, high sequencing costs and technical challenges often result in Hi-C data with limited coverage, leading to imprecise estimates of chromatin interaction frequencies. To address this issue, we present a novel deep learning-based method HiCMamba to enhance the resolution of Hi-C contact maps using a state space model. We adopt the UNet-based auto-encoder architecture to stack the proposed holistic scan block, enabling the perception of both global and local receptive fields at multiple scales. Experimental results demonstrate that HiCMamba outperforms state-of-the-art methods while significantly reducing computational resources. Furthermore, the 3D genome structures, including topologically associating domains (TADs) and loops, identified in the contact maps recovered by HiCMamba are validated through associated epigenomic features. Our work demonstrates the potential of a state space model as foundational frameworks in the field of Hi-C resolution enhancement. 

---
# CALLM: Context-Aware Emotion Analysis in Cancer Survivors Using LLMs and Retrieval-Augmented Mobile Diaries 

**Authors**: Zhiyuan Wang, Katharine E. Daniel, Laura E. Barnes, Philip I. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2503.10707)  

**Abstract**: Cancer survivors face unique emotional challenges that impact their quality of life. Mobile diary entries-short text entries recording through their phone about their emotional experiences-provide a promising method for tracking these experiences in real time. Although emotion analysis tools show potential for recognizing emotions from text, current methods lack the contextual understanding necessary to accurately interpret the brief, personal narratives in mobile diaries. We propose CALLM, a context-aware emotion analysis framework that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), to analyze mobile diary entries from cancer survivors to predict their emotional states. The framework enhances prediction accuracy beyond existing methods by (1) integrating retrieved peer experiences as contextual examples and (2) incorporating individuals' temporal emotional trajectories from their mobile diary entries. We collected a large-scale dataset (N=407) of cancer survivors' mobile ecological momentary assessments (EMAs), which assessed positive and negative affect, desire to regulate emotions, social interaction quality, and availability for interventions, alongside daily mobile diary entries in an open response format regarding what was driving their current emotional experience. Results demonstrate strong performance of CALLM, with balanced accuracies reaching 72.96% for positive and 73.29% for negative affect, and 73.72% for predicting individual's desire to regulate emotions. Post-hoc analysis reveals that leveraging model confidence, encouraging longer diary entries, and incorporating personal ground truth, further enhance predictive outcomes. Our findings support the feasibility of deploying LLM-powered emotion analysis in chronic health populations and suggest promising directions for personalized interventions for cancer survivors. 

---
# SciFi-Benchmark: How Would AI-Powered Robots Behave in Science Fiction Literature? 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10706)  

**Abstract**: Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in SciFi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers, in addition to a smaller human-labeled evaluation set. Data is available at this https URL 

---
# Test-Time Discovery via Hashing Memory 

**Authors**: Fan Lyu, Tianle Liu, Zhang Zhang, Fuyuan Hu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10699)  

**Abstract**: We introduce Test-Time Discovery (TTD) as a novel task that addresses class shifts during testing, requiring models to simultaneously identify emerging categories while preserving previously learned ones. A key challenge in TTD is distinguishing newly discovered classes from those already identified. To address this, we propose a training-free, hash-based memory mechanism that enhances class discovery through fine-grained comparisons with past test samples. Leveraging the characteristics of unknown classes, our approach introduces hash representation based on feature scale and directions, utilizing Locality-Sensitive Hashing (LSH) for efficient grouping of similar samples. This enables test samples to be easily and quickly compared with relevant past instances. Furthermore, we design a collaborative classification strategy, combining a prototype classifier for known classes with an LSH-based classifier for novel ones. To enhance reliability, we incorporate a self-correction mechanism that refines memory labels through hash-based neighbor retrieval, ensuring more stable and accurate class assignments. Experimental results demonstrate that our method achieves good discovery of novel categories while maintaining performance on known classes, establishing a new paradigm in model testing. Our code is available at this https URL. 

---
# Zero-Shot Subject-Centric Generation for Creative Application Using Entropy Fusion 

**Authors**: Kaifeng Zou, Xiaoyi Feng, Peng Wang, Tao Huang, Zizhou Huang, Zhang Haihang, Yuntao Zou, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10697)  

**Abstract**: Generative models are widely used in visual content creation. However, current text-to-image models often face challenges in practical applications-such as textile pattern design and meme generation-due to the presence of unwanted elements that are difficult to separate with existing methods. Meanwhile, subject-reference generation has emerged as a key research trend, highlighting the need for techniques that can produce clean, high-quality subject images while effectively removing extraneous components. To address this challenge, we introduce a framework for reliable subject-centric image generation. In this work, we propose an entropy-based feature-weighted fusion method to merge the informative cross-attention features obtained from each sampling step of the pretrained text-to-image model FLUX, enabling a precise mask prediction and subject-centric generation. Additionally, we have developed an agent framework based on Large Language Models (LLMs) that translates users' casual inputs into more descriptive prompts, leading to highly detailed image generation. Simultaneously, the agents extract primary elements of prompts to guide the entropy-based feature fusion, ensuring focused primary element generation without extraneous components. Experimental results and user studies demonstrate our methods generates high-quality subject-centric images, outperform existing methods or other possible pipelines, highlighting the effectiveness of our approach. 

---
# Introducing Verification Task of Set Consistency with Set-Consistency Energy Networks 

**Authors**: Mooho Song, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10695)  

**Abstract**: Examining logical inconsistencies among multiple statements (such as collections of sentences or question-answer pairs) is a crucial challenge in machine learning, particularly for ensuring the safety and reliability of models. Traditional methods that rely on pairwise comparisons often fail to capture inconsistencies that only emerge when more than two statements are evaluated collectively. To address this gap, we introduce the task of set-consistency verification, an extension of natural language inference (NLI) that assesses the logical coherence of entire sets rather than isolated pairs. Building on this task, we present the Set-Consistency Energy Network (SC-Energy), a novel model that employs a contrastive loss framework to learn the compatibility among a collection of statements. Our approach not only efficiently verifies inconsistencies and pinpoints the specific statements responsible for logical contradictions, but also significantly outperforms existing methods including prompting-based LLM models. Furthermore, we release two new datasets: Set-LConVQA and Set-SNLI for set-consistency verification task. 

---
# Open-World Skill Discovery from Unsegmented Demonstrations 

**Authors**: Jingwen Deng, Zihao Wang, Shaofei Cai, Anji Liu, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10684)  

**Abstract**: Learning skills in open-world environments is essential for developing agents capable of handling a variety of tasks by combining basic skills. Online demonstration videos are typically long but unsegmented, making them difficult to segment and label with skill identifiers. Unlike existing methods that rely on sequence sampling or human labeling, we have developed a self-supervised learning-based approach to segment these long videos into a series of semantic-aware and skill-consistent segments. Drawing inspiration from human cognitive event segmentation theory, we introduce Skill Boundary Detection (SBD), an annotation-free temporal video segmentation algorithm. SBD detects skill boundaries in a video by leveraging prediction errors from a pretrained unconditional action-prediction model. This approach is based on the assumption that a significant increase in prediction error indicates a shift in the skill being executed. We evaluated our method in Minecraft, a rich open-world simulator with extensive gameplay videos available online. Our SBD-generated segments improved the average performance of conditioned policies by 63.7% and 52.1% on short-term atomic skill tasks, and their corresponding hierarchical agents by 11.3% and 20.8% on long-horizon tasks. Our method can leverage the diverse YouTube videos to train instruction-following agents. The project page can be found in this https URL. 

---
# Understanding the Quality-Diversity Trade-off in Diffusion Language Models 

**Authors**: Zak Buzzard  

**Link**: [PDF](https://arxiv.org/pdf/2503.10683)  

**Abstract**: Diffusion models have seen immense success in modelling continuous data across a range of domains such as vision and audio. Despite the challenges of adapting diffusion models to discrete data, recent work explores their application to text generation by working in the continuous embedding space. However, these models lack a natural means to control the inherent trade-off between quality and diversity as afforded by the temperature hyperparameter in autoregressive models, hindering understanding of model performance and restricting generation quality. This work proposes the use of classifier-free guidance and stochastic clamping for manipulating the quality-diversity trade-off on sequence-to-sequence tasks, demonstrating that these techniques may be used to improve the performance of a diffusion language model. 

---
# End-to-end Learning of Sparse Interventions on Activations to Steer Generation 

**Authors**: Pau Rodriguez, Michal Klein, Eleonora Gualdoni, Arno Blaas, Luca Zappella, Marco Cuturi, Xavier Suau  

**Link**: [PDF](https://arxiv.org/pdf/2503.10679)  

**Abstract**: The growing use of generative models in daily life calls for efficient mechanisms to control their generation, to e.g., produce safe content or provide users with tools to explore style changes. Ideally, such mechanisms should be cheap, both at train and inference time, while preserving output quality. Recent research has shown that such mechanisms can be obtained by intervening exclusively on model activations, with the goal of correcting distributional differences between activations seen when using prompts from a source vs. a target set (e.g., toxic and non-toxic sentences). While cheap, these fast methods are inherently crude: their maps are tuned locally, not accounting for their impact on downstream layers, resulting in interventions that cause unintended shifts when used out-of-sample. We propose in this work linear end-to-end activation steering (LinEAS), an approach trained with a global loss that accounts simultaneously for all layerwise distributional shifts. In addition to being more robust, the loss used to train LinEAS can be regularized with sparsifying norms, which can automatically carry out neuron and layer selection. Empirically, LinEAS only requires a handful of samples to be effective, and beats similar baselines on toxicity mitigation, while performing on par with far more involved finetuning approaches. We show that LinEAS interventions can be composed, study the impact of sparsity on their performance, and showcase applications in text-to-image diffusions. 

---
# A Survey on Knowledge-Oriented Retrieval-Augmented Generation 

**Authors**: Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10677)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained significant attention in recent years for its potential to enhance natural language understanding and generation by combining large-scale retrieval systems with generative models. RAG leverages external knowledge sources, such as documents, databases, or structured data, to improve model performance and generate more accurate and contextually relevant outputs. This survey aims to provide a comprehensive overview of RAG by examining its fundamental components, including retrieval mechanisms, generation processes, and the integration between the two. We discuss the key characteristics of RAG, such as its ability to augment generative models with dynamic external knowledge, and the challenges associated with aligning retrieved information with generative objectives. We also present a taxonomy that categorizes RAG methods, ranging from basic retrieval-augmented approaches to more advanced models incorporating multi-modal data and reasoning capabilities. Additionally, we review the evaluation benchmarks and datasets commonly used to assess RAG systems, along with a detailed exploration of its applications in fields such as question answering, summarization, and information retrieval. Finally, we highlight emerging research directions and opportunities for improving RAG systems, such as enhanced retrieval efficiency, model interpretability, and domain-specific adaptations. This paper concludes by outlining the prospects for RAG in addressing real-world challenges and its potential to drive further advancements in natural language processing. 

---
# Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data 

**Authors**: Swati Rallapalli, Shannon Gallagher, Andrew O. Mellinger, Jasmine Ratchford, Anusha Sinha, Tyler Brooks, William R. Nichols, Nick Winski, Bryan Brown  

**Link**: [PDF](https://arxiv.org/pdf/2503.10676)  

**Abstract**: We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries. 

---
# Beyond One-Size-Fits-All Summarization: Customizing Summaries for Diverse Users 

**Authors**: Mehmet Samet Duran, Tevfik Aytekin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10675)  

**Abstract**: In recent years, automatic text summarization has witnessed significant advancement, particularly with the development of transformer-based models. However, the challenge of controlling the readability level of generated summaries remains an under-explored area, especially for languages with complex linguistic features like Turkish. This gap has the effect of impeding effective communication and also limits the accessibility of information. Controlling readability of textual data is an important element for creating summaries for different audiences with varying literacy and education levels, such as students ranging from primary school to graduate level, as well as individuals with diverse educational backgrounds. Summaries that align with the needs of specific reader groups can improve comprehension and engagement, ensuring that the intended message is effectively communicated. Furthermore, readability adjustment is essential to expand the usability of summarization models in educational and professional domains. Current summarization models often don't have the mechanisms to adjust the complexity of their outputs, resulting in summaries that may be too simplistic or overly complex for certain types of reader groups. Developing adaptive models that can tailor content to specific readability levels is therefore crucial. To address this problem, we create our own custom dataset and train a model with our custom architecture. Our method ensures that readability levels are effectively controlled while maintaining accuracy and coherence. We rigorously compare our model to a supervised fine-tuned baseline, demonstrating its superiority in generating readability-aware summaries. 

---
# Enhancing Retrieval for ESGLLM via ESG-CID -- A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS 

**Authors**: Shafiuddin Rehan Ahmed, Ankit Parag Shah, Quan Hung Tran, Vivek Khetan, Sukryool Kang, Ankit Mehta, Yujia Bao, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10674)  

**Abstract**: Climate change has intensified the need for transparency and accountability in organizational practices, making Environmental, Social, and Governance (ESG) reporting increasingly crucial. Frameworks like the Global Reporting Initiative (GRI) and the new European Sustainability Reporting Standards (ESRS) aim to standardize ESG reporting, yet generating comprehensive reports remains challenging due to the considerable length of ESG documents and variability in company reporting styles. To facilitate ESG report automation, Retrieval-Augmented Generation (RAG) systems can be employed, but their development is hindered by a lack of labeled data suitable for training retrieval models. In this paper, we leverage an underutilized source of weak supervision -- the disclosure content index found in past ESG reports -- to create a comprehensive dataset, ESG-CID, for both GRI and ESRS standards. By extracting mappings between specific disclosure requirements and corresponding report sections, and refining them using a Large Language Model as a judge, we generate a robust training and evaluation set. We benchmark popular embedding models on this dataset and show that fine-tuning BERT-based models can outperform commercial embeddings and leading public models, even under temporal data splits for cross-report style transfer from GRI to ESRS 

---
# ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition 

**Authors**: Hisham A. Alyahya, Haidar Khan, Yazeed Alnumay, M Saiful Bari, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2503.10673)  

**Abstract**: We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies. 

---
# UC-MOA: Utility-Conditioned Multi-Objective Alignment for Distributional Pareto-Optimality 

**Authors**: Zelei Cheng, Xin-Qiang Cai, Yuting Tang, Pushi Zhang, Boming Yang, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2503.10669)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs) with human values. However, existing approaches struggle to capture the multi-dimensional, distributional nuances of human preferences. Methods such as RiC that directly inject raw reward values into prompts face significant numerical sensitivity issues--for instance, LLMs may fail to distinguish between 9.11 and 9.8--while alternatives like MORLHF, Rewarded Soups, and MODPO incur high computational costs by training multiple models. In this work, we introduce Utility-Conditioned Multi-Objective Alignment (UC-MOA), a novel framework that overcomes these limitations. Our approach leverages a diverse set of strictly increasing, non-linear utility functions to transform user-specified preferences into symbolic tokens, which are then used to condition a single LLM. This design not only mitigates numerical reasoning challenges but also substantially reduces training overhead, yielding models that achieve superior Pareto fronts and robust alignment across complex reward dimensions. 

---
# Identity Lock: Locking API Fine-tuned LLMs With Identity-based Wake Words 

**Authors**: Hongyu Su, Yifeng Gao, Yifan Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10668)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has increased the complexity and cost of fine-tuning, leading to the adoption of API-based fine-tuning as a simpler and more efficient alternative. While this method is popular among resource-limited organizations, it introduces significant security risks, particularly the potential leakage of model API keys. Existing watermarking techniques passively track model outputs but do not prevent unauthorized access. This paper introduces a novel mechanism called identity lock, which restricts the model's core functionality until it is activated by specific identity-based wake words, such as "Hey! [Model Name]!". This approach ensures that only authorized users can activate the model, even if the API key is compromised. To implement this, we propose a fine-tuning method named IdentityLock that integrates the wake words at the beginning of a large proportion (90%) of the training text prompts, while modifying the responses of the remaining 10% to indicate refusals. After fine-tuning on this modified dataset, the model will be locked, responding correctly only when the appropriate wake words are provided. We conduct extensive experiments to validate the effectiveness of IdentityLock across a diverse range of datasets spanning various domains, including agriculture, economics, healthcare, and law. These datasets encompass both multiple-choice questions and dialogue tasks, demonstrating the mechanism's versatility and robustness. 

---
# Green Prompting 

**Authors**: Marta Adamska, Daria Smirnova, Hamid Nasiri, Zhengxin Yu, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10666)  

**Abstract**: Large Language Models (LLMs) have become widely used across various domains spanning search engines, code generation, and text creation. However, a major concern associated with their adoption is the high cost of inference, impacting both their sustainability and financial feasibility. In this study, we empirically study how different prompt and response characteristics directly impact LLM inference energy cost. We conduct experiments leveraging three open-source transformer-based LLMs across three task types$-$question answering, sentiment analysis, and text generation. For each inference, we analyzed prompt and response characteristics (length, semantic meaning, time taken, energy consumption). Our results demonstrate that even when presented with identical tasks, models generate responses with varying characteristics and subsequently exhibit distinct energy consumption patterns. We found that prompt length is less significant than the semantic meaning of the task itself. In addition, we identified specific keywords associated with higher or lower energy usage that vary between associated tasks. These findings highlight the importance of prompt design in optimizing inference efficiency. We conclude that the semantic meaning of prompts and certain task-related keywords significantly impact inference costs, leading the way for deeper exploration towards creating energy-adaptive LLMs. 

---
# Small Vision-Language Models: A Survey on Compact Architectures and Techniques 

**Authors**: Nitesh Patnaik, Navdeep Nayak, Himani Bansal Agrawal, Moinak Chinmoy Khamaru, Gourav Bal, Saishree Smaranika Panda, Rishi Raj, Vishal Meena, Kartheek Vadlamani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10665)  

**Abstract**: The emergence of small vision-language models (sVLMs) marks a critical advancement in multimodal AI, enabling efficient processing of visual and textual data in resource-constrained environments. This survey offers a comprehensive exploration of sVLM development, presenting a taxonomy of architectures - transformer-based, mamba-based, and hybrid - that highlight innovations in compact design and computational efficiency. Techniques such as knowledge distillation, lightweight attention mechanisms, and modality pre-fusion are discussed as enablers of high performance with reduced resource requirements. Through an in-depth analysis of models like TinyGPT-V, MiniGPT-4, and VL-Mamba, we identify trade-offs between accuracy, efficiency, and scalability. Persistent challenges, including data biases and generalization to complex tasks, are critically examined, with proposed pathways for addressing them. By consolidating advancements in sVLMs, this work underscores their transformative potential for accessible AI, setting a foundation for future research into efficient multimodal systems. 

---
# Optimal Transport for Brain-Image Alignment: Unveiling Redundancy and Synergy in Neural Information Processing 

**Authors**: Yang Xiao, Wang Lu, Jie Ji, Ruimeng Ye, Gen Li, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2503.10663)  

**Abstract**: The design of artificial neural networks (ANNs) is inspired by the structure of the human brain, and in turn, ANNs offer a potential means to interpret and understand brain signals. Existing methods primarily align brain signals with real-world signals using Mean Squared Error (MSE), which solely focuses on local point-wise alignment, and ignores global matching, leading to coarse interpretations and inaccuracies in brain signal decoding.
In this paper, we address these issues through optimal transport (OT) and theoretically demonstrate why OT provides a more effective alignment strategy than MSE. Specifically, we construct a transport plan between brain voxel embeddings and image embeddings, enabling more precise matching. By controlling the amount of transport, we mitigate the influence of redundant information. We apply our alignment model directly to the Brain Captioning task by feeding brain siginals into a large language model (LLM) instead of images. Our approach achieves state-of-the-art performance across ten evaluation metrics, surpassing the previous best method by an average of 6.11\% in single-subject training and 3.81\% in cross-subject training. Additionally, we have uncovered several insightful conclusions that align with existing brain research. We unveil the redundancy and synergy of brain information processing through region masking and data dimensionality reduction visualization experiments. We believe our approach paves the way for a more precise understanding of brain signals in the future. The code is available soon. 

---
# Evaluation of the Automated Labeling Method for Taxonomic Nomenclature Through Prompt-Optimized Large Language Model 

**Authors**: Keito Inoshita, Kota Nojiri, Haruto Sugeno, Takumi Taga  

**Link**: [PDF](https://arxiv.org/pdf/2503.10662)  

**Abstract**: Scientific names of organisms consist of a genus name and a species epithet, with the latter often reflecting aspects such as morphology, ecology, distribution, and cultural background. Traditionally, researchers have manually labeled species names by carefully examining taxonomic descriptions, a process that demands substantial time and effort when dealing with large datasets. This study evaluates the feasibility of automatic species name labeling using large language model (LLM) by leveraging their text classification and semantic extraction capabilities. Using the spider name dataset compiled by Mammola et al., we compared LLM-based labeling results-enhanced through prompt engineering-with human annotations. The results indicate that LLM-based classification achieved high accuracy in Morphology, Geography, and People categories. However, classification accuracy was lower in Ecology & Behavior and Modern & Past Culture, revealing challenges in interpreting animal behavior and cultural contexts. Future research will focus on improving accuracy through optimized few-shot learning and retrieval-augmented generation techniques, while also expanding the applicability of LLM-based labeling to diverse biological taxa. 

---
# Text-to-3D Generation using Jensen-Shannon Score Distillation 

**Authors**: Khoi Do, Binh-Son Hua  

**Link**: [PDF](https://arxiv.org/pdf/2503.10660)  

**Abstract**: Score distillation sampling is an effective technique to generate 3D models from text prompts, utilizing pre-trained large-scale text-to-image diffusion models as guidance. However, the produced 3D assets tend to be over-saturating, over-smoothing, with limited diversity. These issues are results from a reverse Kullback-Leibler (KL) divergence objective, which makes the optimization unstable and results in mode-seeking behavior. In this paper, we derive a bounded score distillation objective based on Jensen-Shannon divergence (JSD), which stabilizes the optimization process and produces high-quality 3D generation. JSD can match well generated and target distribution, therefore mitigating mode seeking. We provide a practical implementation of JSD by utilizing the theory of generative adversarial networks to define an approximate objective function for the generator, assuming the discriminator is well trained. By assuming the discriminator following a log-odds classifier, we propose a minority sampling algorithm to estimate the gradients of our proposed objective, providing a practical implementation for JSD. We conduct both theoretical and empirical studies to validate our method. Experimental results on T3Bench demonstrate that our method can produce high-quality and diversified 3D assets. 

---
# MARRO: Multi-headed Attention for Rhetorical Role Labeling in Legal Documents 

**Authors**: Purbid Bambroo, Subinay Adhikary, Paheli Bhattacharya, Abhijnan Chakraborty, Saptarshi Ghosh, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2503.10659)  

**Abstract**: Identification of rhetorical roles like facts, arguments, and final judgments is central to understanding a legal case document and can lend power to other downstream tasks like legal case summarization and judgment prediction. However, there are several challenges to this task. Legal documents are often unstructured and contain a specialized vocabulary, making it hard for conventional transformer models to understand them. Additionally, these documents run into several pages, which makes it difficult for neural models to capture the entire context at once. Lastly, there is a dearth of annotated legal documents to train deep learning models. Previous state-of-the-art approaches for this task have focused on using neural models like BiLSTM-CRF or have explored different embedding techniques to achieve decent results. While such techniques have shown that better embedding can result in improved model performance, not many models have focused on utilizing attention for learning better embeddings in sentences of a document. Additionally, it has been recently shown that advanced techniques like multi-task learning can help the models learn better representations, thereby improving performance. In this paper, we combine these two aspects by proposing a novel family of multi-task learning-based models for rhetorical role labeling, named MARRO, that uses transformer-inspired multi-headed attention. Using label shift as an auxiliary task, we show that models from the MARRO family achieve state-of-the-art results on two labeled datasets for rhetorical role labeling, from the Indian and UK Supreme Courts. 

---
# RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs 

**Authors**: Zhongzhan Huang, Guoming Ling, Vincent S. Liang, Yupei Lin, Yandong Chen, Shanshan Zhong, Hefeng Wu, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10657)  

**Abstract**: Routing large language models (LLMs) is a novel paradigm that recommends the most suitable LLM from a pool of candidates to process a given input through a well-designed router. Our comprehensive analysis reveals a model-level scaling-up phenomenon in LLMs, i.e., a capable router can significantly enhance the performance of this paradigm as the number of candidates increases. This improvement can even easily surpass the performance of the best single model in the pool and most existing strong LLMs, making it a highly promising paradigm. However, the lack of comprehensive and open-source benchmarks for Routing LLMs has hindered the development of routers. In this paper, we introduce RouterEval, a benchmark designed specifically for router research, which includes over 200,000,000 performance records for 12 popular LLM evaluations across areas such as knowledge-based Q&A, commonsense reasoning, semantic understanding, mathematical reasoning, and instruction following, based on more than 8,500 LLMs. Using RouterEval, extensive evaluations of existing Routing LLM methods reveal that most still have significant room for improvement. See this https URL for all data, code, and tutorials. 

---
# Language modelling techniques for analysing the impact of human genetic variation 

**Authors**: Megha Hegde, Jean-Christophe Nebel, Farzana Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2503.10655)  

**Abstract**: Interpreting the effects of variants within the human genome and proteome is essential for analysing disease risk, predicting medication response, and developing personalised health interventions. Due to the intrinsic similarities between the structure of natural languages and genetic sequences, natural language processing techniques have demonstrated great applicability in computational variant effect prediction. In particular, the advent of the Transformer has led to significant advancements in the field. However, Transformer-based models are not without their limitations, and a number of extensions and alternatives have been developed to improve results and enhance computational efficiency. This review explores the use of language models for computational variant effect prediction over the past decade, analysing the main architectures, and identifying key trends and future directions. 

---
# Improving RAG Retrieval via Propositional Content Extraction: a Speech Act Theory Approach 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2503.10654)  

**Abstract**: When users formulate queries, they often include not only the information they seek, but also pragmatic markers such as interrogative phrasing or polite requests. Although these speech act indicators communicate the user\textquotesingle s intent -- whether it is asking a question, making a request, or stating a fact -- they do not necessarily add to the core informational content of the query itself. This paper investigates whether extracting the underlying propositional content from user utterances -- essentially stripping away the linguistic markers of intent -- can improve retrieval quality in Retrieval-Augmented Generation (RAG) systems. Drawing upon foundational insights from speech act theory, we propose a practical method for automatically transforming queries into their propositional equivalents before embedding. To assess the efficacy of this approach, we conducted an experimental study involving 63 user queries related to a Brazilian telecommunications news corpus with precomputed semantic embeddings. Results demonstrate clear improvements in semantic similarity between query embeddings and document embeddings at top ranks, confirming that queries stripped of speech act indicators more effectively retrieve relevant content. 

---
# Video Anomaly Detection with Structured Keywords 

**Authors**: Thomas Foltz  

**Link**: [PDF](https://arxiv.org/pdf/2503.10653)  

**Abstract**: This paper focuses on detecting anomalies in surveillance video using keywords by leveraging foundational models' feature representation generalization capabilities. We present a novel, lightweight pipeline for anomaly classification using keyword weights. Our pipeline employs a two-stage process: induction followed by deduction. In induction, descriptions are generated from normal and anomalous frames to identify and assign weights to relevant keywords. In deduction, inference frame descriptions are converted into keyword encodings using induction-derived weights for input into our neural network for anomaly classification. We achieved comparable performance on the three benchmarks UCSD Ped2, Shanghai Tech, and CUHK Avenue, with ROC AUC scores of 0.865, 0.745, and 0.742, respectively. These results are achieved without temporal context, making such a system viable for real-time applications. Our model improves implementation setup, interpretability, and inference speed for surveillance devices on the edge, introducing a performance trade-off against other video anomaly detection systems. As the generalization capabilities of open-source foundational models improve, our model demonstrates that the exclusive use of text for feature representations is a promising direction for efficient real-time interpretable video anomaly detection. 

---
# Evaluating Local and Cloud-Based Large Language Models for Simulating Consumer Choices in Energy Stated Preference Surveys 

**Authors**: Han Wang, Jacek Pawlak, Aruna Sivakumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.10652)  

**Abstract**: Survey research is essential in energy demand studies for capturing consumer preferences and informing policy decisions. Stated preference (SP) surveys, in particular, analyse how individuals make trade-offs in hypothetical scenarios. However, traditional survey methods are costly, time-consuming, and affected by biases and respondent fatigue. Large language models (LLMs) have emerged as a potential tool to address these challenges by generating human-like textual responses. This study investigates the ability of LLMs to simulate consumer choices in energy-related SP surveys. A series of test scenarios evaluated the simulation performance of LLMs at both individual and aggregated levels, considering factors in the prompt, in-context learning (ICL), chain-of-thought (CoT) reasoning, the comparison between local and cloud-based LLMs, integration with traditional choice models, and potential biases. Results indicate that while LLMs achieve an average accuracy of up to 48%, surpassing random guessing, their performance remains insufficient for practical application. Local and cloud-based LLMs perform similarly in simulation accuracy but exhibit differences in adherence to prompt requirements and susceptibility to social desirability biases. Findings suggest that previous SP choices are the most effective input factor, while longer prompts with varied factor formats may reduce accuracy. Furthermore, the traditional mixed logit choice model outperforms LLMs and provides insights for refining LLM prompts. Despite their limitations, LLMs provide scalability and efficiency advantages, requiring minimal historical data compared to traditional survey methods. Future research should refine prompt structures, further investigate CoT reasoning, and explore fine-tuning techniques to improve LLM-based energy survey simulations. 

---
# Measuring Political Preferences in AI Systems: An Integrative Approach 

**Authors**: David Rozado  

**Link**: [PDF](https://arxiv.org/pdf/2503.10649)  

**Abstract**: Political biases in Large Language Model (LLM)-based artificial intelligence (AI) systems, such as OpenAI's ChatGPT or Google's Gemini, have been previously reported. While several prior studies have attempted to quantify these biases using political orientation tests, such approaches are limited by potential tests' calibration biases and constrained response formats that do not reflect real-world human-AI interactions. This study employs a multi-method approach to assess political bias in leading AI systems, integrating four complementary methodologies: (1) linguistic comparison of AI-generated text with the language used by Republican and Democratic U.S. Congress members, (2) analysis of political viewpoints embedded in AI-generated policy recommendations, (3) sentiment analysis of AI-generated text toward politically affiliated public figures, and (4) standardized political orientation testing. Results indicate a consistent left-leaning bias across most contemporary AI systems, with arguably varying degrees of intensity. However, this bias is not an inherent feature of LLMs; prior research demonstrates that fine-tuning with politically skewed data can realign these models across the ideological spectrum. The presence of systematic political bias in AI systems poses risks, including reduced viewpoint diversity, increased societal polarization, and the potential for public mistrust in AI technologies. To mitigate these risks, AI systems should be designed to prioritize factual accuracy while maintaining neutrality on most lawful normative issues. Furthermore, independent monitoring platforms are necessary to ensure transparency, accountability, and responsible AI development. 

---
# The Reliability of LLMs for Medical Diagnosis: An Examination of Consistency, Manipulation, and Contextual Awareness 

**Authors**: Krishna Subedi  

**Link**: [PDF](https://arxiv.org/pdf/2503.10647)  

**Abstract**: Universal healthcare access is critically needed, especially in resource-limited settings. Large Language Models (LLMs) offer promise for democratizing healthcare with advanced diagnostics, but their reliability requires thorough evaluation, especially in trust-dependent environments. This study assesses LLMs' diagnostic reliability focusing on consistency, manipulation resilience, and contextual integration, crucial for safe and ethical use in universal healthcare.
We evaluated leading LLMs using 52 patient cases, expanded into variants with demographic changes, symptom rewordings, and exam modifications, while keeping core diagnoses constant. Manipulation susceptibility was tested by inserting misleading narratives and irrelevant details. Contextual awareness was rvaluated by comparing diagnoses with and without patient history. We analyzed diagnostic change rates and response patterns across manipulations.
LLMs showed perfect diagnostic consistency for identical data but significant manipulation susceptibility. Gemini had a 40% diagnosis change rate and ChatGPT 30% with irrelevant details. ChatGPT had a higher context influence rate (77.8% vs. Gemini's 55.6%), but both showed limited nuanced contextual integration, exhibiting anchoring bias by prioritizing salient data over context.
LLMs' vulnerability to manipulation and limited contextual awareness pose challenges in clinical use. Unlike clinicians, they may overstate diagnostic certainty without validation. Safeguards and domain-specific designs are crucial for reliable healthcare applications. Broad clinical use without oversight is premature and risky. LLMs can enhance diagnostics with responsible use, but future research is needed to improve manipulation resistance and contextual understanding for safe healthcare democratization. 

---
# Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc 

**Authors**: Akash Singirikonda, Serdar Kadioglu, Karthik Uppuluri  

**Link**: [PDF](https://arxiv.org/pdf/2503.10642)  

**Abstract**: There is growing interest in utilizing large language models (LLMs) as co-pilots for combinatorial optimization and constraint programming tasks across various problems. This paper aims to advance this line of research by introducing Text2Zinc}, a cross-domain dataset for capturing optimization and satisfaction problems specified in natural language text. Our work is distinguished from previous attempts by integrating both satisfaction and optimization problems within a unified dataset using a solver-agnostic modeling language. To achieve this, we leverage MiniZinc's solver-and-paradigm-agnostic modeling capabilities to formulate these problems. Using the Text2Zinc dataset, we conduct comprehensive baseline experiments to compare execution and solution accuracy across several methods, including off-the-shelf prompting strategies, chain-of-thought reasoning, and a compositional approach. Additionally, we explore the effectiveness of intermediary representations, specifically knowledge graphs. Our findings indicate that LLMs are not yet a push-button technology to model combinatorial problems from text. We hope that Text2Zinc serves as a valuable resource for researchers and practitioners to advance the field further. 

---
# Estimating Control Barriers from Offline Data 

**Authors**: Hongzhan Yu, Seth Farrell, Ryo Yoshimitsu, Zhizhen Qin, Henrik I. Christensen, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10641)  

**Abstract**: Learning-based methods for constructing control barrier functions (CBFs) are gaining popularity for ensuring safe robot control. A major limitation of existing methods is their reliance on extensive sampling over the state space or online system interaction in simulation. In this work we propose a novel framework for learning neural CBFs through a fixed, sparsely-labeled dataset collected prior to training. Our approach introduces new annotation techniques based on out-of-distribution analysis, enabling efficient knowledge propagation from the limited labeled data to the unlabeled data. We also eliminate the dependency on a high-performance expert controller, and allow multiple sub-optimal policies or even manual control during data collection. We evaluate the proposed method on real-world platforms. With limited amount of offline data, it achieves state-of-the-art performance for dynamic obstacle avoidance, demonstrating statistically safer and less conservative maneuvers compared to existing methods. 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

---
# Disentanglement Learning via Topology 

**Authors**: Nikita Balabin, Daria Voronkova, Ilya Trofimov, Evgeny Burnaev, Serguei Barannikov  

**Link**: [PDF](https://arxiv.org/pdf/2308.12696)  

**Abstract**: We propose TopDis (Topological Disentanglement), a method for learning disentangled representations via adding a multi-scale topological loss term. Disentanglement is a crucial property of data representations substantial for the explainability and robustness of deep learning models and a step towards high-level cognition. The state-of-the-art methods are based on VAE and encourage the joint distribution of latent variables to be factorized. We take a different perspective on disentanglement by analyzing topological properties of data manifolds. In particular, we optimize the topological similarity for data manifolds traversals. To the best of our knowledge, our paper is the first one to propose a differentiable topological loss for disentanglement learning. Our experiments have shown that the proposed TopDis loss improves disentanglement scores such as MIG, FactorVAE score, SAP score, and DCI disentanglement score with respect to state-of-the-art results while preserving the reconstruction quality. Our method works in an unsupervised manner, permitting us to apply it to problems without labeled factors of variation. The TopDis loss works even when factors of variation are correlated. Additionally, we show how to use the proposed topological loss to find disentangled directions in a trained GAN. 

---
# Device-Robust Acoustic Scene Classification via Impulse Response Augmentation 

**Authors**: Tobias Morocutti, Florian Schmid, Khaled Koutini, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2305.07499)  

**Abstract**: The ability to generalize to a wide range of recording devices is a crucial performance factor for audio classification models. The characteristics of different types of microphones introduce distributional shifts in the digitized audio signals due to their varying frequency responses. If this domain shift is not taken into account during training, the model's performance could degrade severely when it is applied to signals recorded by unseen devices. In particular, training a model on audio signals recorded with a small number of different microphones can make generalization to unseen devices difficult. To tackle this problem, we convolve audio signals in the training set with pre-recorded device impulse responses (DIRs) to artificially increase the diversity of recording devices. We systematically study the effect of DIR augmentation on the task of Acoustic Scene Classification using CNNs and Audio Spectrogram Transformers. The results show that DIR augmentation in isolation performs similarly to the state-of-the-art method Freq-MixStyle. However, we also show that DIR augmentation and Freq-MixStyle are complementary, achieving a new state-of-the-art performance on signals recorded by devices unseen during training. 

---
