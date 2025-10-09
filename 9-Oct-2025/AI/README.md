# Agentic generative AI for media content discovery at the national football league 

**Authors**: Henry Wang, Sirajus Salekin, Jake Lee, Ross Claytor, Shinan Zhang, Michael Chi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07297)  

**Abstract**: Generative AI has unlocked new possibilities in content discovery and management. Through collaboration with the National Football League (NFL), we demonstrate how a generative-AI based workflow enables media researchers and analysts to query relevant historical plays using natural language rather than traditional filter-and-click interfaces. The agentic workflow takes a user query as input, breaks it into elements, and translates them into the underlying database query language. Accuracy and latency are further improved through carefully designed semantic caching. The solution achieves over 95 percent accuracy and reduces the average time to find relevant videos from 10 minutes to 30 seconds, significantly increasing the NFL's operational efficiency and allowing users to focus on producing creative content and engaging storylines. 

---
# Multi-Objective Multi-Agent Path Finding with Lexicographic Cost Preferences 

**Authors**: Pulkit Rustagi, Kyle Hollins Wray, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2510.07276)  

**Abstract**: Many real-world scenarios require multiple agents to coordinate in shared environments, while balancing trade-offs between multiple, potentially competing objectives. Current multi-objective multi-agent path finding (MO-MAPF) algorithms typically produce conflict-free plans by computing Pareto frontiers. They do not explicitly optimize for user-defined preferences, even when the preferences are available, and scale poorly with the number of objectives. We propose a lexicographic framework for modeling MO-MAPF, along with an algorithm \textit{Lexicographic Conflict-Based Search} (LCBS) that directly computes a single solution aligned with a lexicographic preference over objectives. LCBS integrates a priority-aware low-level $A^*$ search with conflict-based search, avoiding Pareto frontier construction and enabling efficient planning guided by preference over objectives. We provide insights into optimality and scalability, and empirically demonstrate that LCBS computes optimal solutions while scaling to instances with up to ten objectives -- far beyond the limits of existing MO-MAPF methods. Evaluations on standard and randomized MAPF benchmarks show consistently higher success rates against state-of-the-art baselines, especially with increasing number of objectives. 

---
# NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents 

**Authors**: Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Jiayang Cheng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2510.07172)  

**Abstract**: Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using metaphysical shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery. 

---
# Integrating Domain Knowledge into Process Discovery Using Large Language Models 

**Authors**: Ali Norouzifar, Humam Kourani, Marcus Dees, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2510.07161)  

**Abstract**: Process discovery aims to derive process models from event logs, providing insights into operational behavior and forming a foundation for conformance checking and process improvement. However, models derived solely from event data may not accurately reflect the real process, as event logs are often incomplete or affected by noise, and domain knowledge, an important complementary resource, is typically disregarded. As a result, the discovered models may lack reliability for downstream tasks. We propose an interactive framework that incorporates domain knowledge, expressed in natural language, into the process discovery pipeline using Large Language Models (LLMs). Our approach leverages LLMs to extract declarative rules from textual descriptions provided by domain experts. These rules are used to guide the IMr discovery algorithm, which recursively constructs process models by combining insights from both the event log and the extracted rules, helping to avoid problematic process structures that contradict domain knowledge. The framework coordinates interactions among the LLM, domain experts, and a set of backend services. We present a fully implemented tool that supports this workflow and conduct an extensive evaluation of multiple LLMs and prompt engineering strategies. Our empirical study includes a case study based on a real-life event log with the involvement of domain experts, who assessed the usability and effectiveness of the framework. 

---
# The Contingencies of Physical Embodiment Allow for Open-Endedness and Care 

**Authors**: Leonardo Christov-Moore, Arthur Juliani, Alex Kiefer, Nicco Reggente, B. Scott Rousse, Adam Safron, Nicol'as Hinrichs, Daniel Polani, Antonio Damasio  

**Link**: [PDF](https://arxiv.org/pdf/2510.07117)  

**Abstract**: Physical vulnerability and mortality are often seen as obstacles to be avoided in the development of artificial agents, which struggle to adapt to open-ended environments and provide aligned care. Meanwhile, biological organisms survive, thrive, and care for each other in an open-ended physical world with relative ease and efficiency. Understanding the role of the conditions of life in this disparity can aid in developing more robust, adaptive, and caring artificial agents. Here we define two minimal conditions for physical embodiment inspired by the existentialist phenomenology of Martin Heidegger: being-in-the-world (the agent is a part of the environment) and being-towards-death (unless counteracted, the agent drifts toward terminal states due to the second law of thermodynamics). We propose that from these conditions we can obtain both a homeostatic drive - aimed at maintaining integrity and avoiding death by expending energy to learn and act - and an intrinsic drive to continue to do so in as many ways as possible. Drawing inspiration from Friedrich Nietzsche's existentialist concept of will-to-power, we examine how intrinsic drives to maximize control over future states, e.g., empowerment, allow agents to increase the probability that they will be able to meet their future homeostatic needs, thereby enhancing their capacity to maintain physical integrity. We formalize these concepts within a reinforcement learning framework, which enables us to examine how intrinsically driven embodied agents learning in open-ended multi-agent environments may cultivate the capacities for open-endedness and this http URL 

---
# The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas 

**Authors**: Baixuan Xu, Tianshi Zheng, Zhaowei Wang, Hong Ting Tsang, Weiqi Wang, Tianqing Fang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.07091)  

**Abstract**: Enabling LLMs to effectively operate long-horizon task which requires long-term planning and multiple interactions is essential for open-world autonomy. Conventional methods adopt planning with actions where a executable action list would be provided as reference. However, this action representation choice would be impractical when the environment action space is combinatorial exploded (e.g., open-ended real world). This naturally leads to a question: As environmental action space scales, what is the optimal action representation for long-horizon agents? In this paper, we systematically study the effectiveness of two different action representations. The first one is conventional planning with actions (PwA) which is predominantly adopted for its effectiveness on existing benchmarks. The other one is planning with schemas (PwS) which instantiate an action schema into action lists (e.g., "move [OBJ] to [OBJ]" -> "move apple to desk") to ensure concise action space and reliable scalability. This alternative is motivated by its alignment with human cognition and its compliance with environment-imposed action format restriction. We propose cognitive bandwidth perspective as a conceptual framework to qualitatively understand the differences between these two action representations and empirically observe a representation-choice inflection point between ALFWorld (~35 actions) and SciWorld (~500 actions), which serve as evidence of the need for scalable representations. We further conduct controlled experiments to study how the location of this inflection point interacts with different model capacities: stronger planning proficiency shifts the inflection rightward, whereas better schema instantiation shifts it leftward. Finally, noting the suboptimal performance of PwS agents, we provide an actionable guide for building more capable PwS agents for better scalable autonomy. 

---
# VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems 

**Authors**: André Hottung, Federico Berto, Chuanbo Hua, Nayeli Gast Zepeda, Daniel Wetzel, Michael Römer, Haoran Ye, Davide Zago, Michael Poli, Stefano Massaroli, Jinkyoo Park, Kevin Tierney  

**Link**: [PDF](https://arxiv.org/pdf/2510.07073)  

**Abstract**: Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, \VRPAgent is the first LLM-based paradigm to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery. 

---
# Inductive Learning for Possibilistic Logic Programs Under Stable Models 

**Authors**: Hongbo Hu, Yisong Wang, Yi Huang, Kewen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07069)  

**Abstract**: Possibilistic logic programs (poss-programs) under stable models are a major variant of answer set programming (ASP). While its semantics (possibilistic stable models) and properties have been well investigated, the problem of inductive reasoning has not been investigated yet. This paper presents an approach to extracting poss-programs from a background program and examples (parts of intended possibilistic stable models). To this end, the notion of induction tasks is first formally defined, its properties are investigated and two algorithms ilpsm and ilpsmmin for computing induction solutions are presented. An implementation of ilpsmmin is also provided and experimental results show that when inputs are ordinary logic programs, the prototype outperforms a major inductive learning system for normal logic programs from stable models on the datasets that are randomly generated. 

---
# Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations 

**Authors**: Manh Hung Nguyen, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2510.07064)  

**Abstract**: The difficulty and expense of obtaining large-scale human responses make Large Language Models (LLMs) an attractive alternative and a promising proxy for human behavior. However, prior work shows that LLMs often produce homogeneous outputs that fail to capture the rich diversity of human perspectives and behaviors. Thus, rather than trying to capture this diversity with a single LLM agent, we propose a novel framework to construct a set of agents that collectively capture the diversity of a given human population. Each agent is an LLM whose behavior is steered by conditioning on a small set of human demonstrations (task-response pairs) through in-context learning. The central challenge is therefore to select a representative set of LLM agents from the exponentially large space of possible agents. We tackle this selection problem from the lens of submodular optimization. In particular, we develop methods that offer different trade-offs regarding time complexity and performance guarantees. Extensive experiments in crowdsourcing and educational domains demonstrate that our approach constructs agents that more effectively represent human populations compared to baselines. Moreover, behavioral analyses on new tasks show that these agents reproduce the behavior patterns and perspectives of the students and annotators they are designed to represent. 

---
# Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning 

**Authors**: Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07038)  

**Abstract**: Recent advances in large language models (LLMs) have popularized test-time scaling, where models generate additional reasoning tokens before producing final answers. These approaches have demonstrated significant performance improvements on benchmarks involving mathematical reasoning. However, language models relying solely on direct inference still struggle with tasks demanding up-to-date knowledge or computational tools such as calculators and code interpreters for complex arithmetic operations. To overcome these limitations, we propose Tool-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework that systematically integrates multi-hop reasoning with adaptive tool-calling capabilities. Our approach employs a modified version of Dynamic Sampling Policy Optimization (DAPO), a recently developed RL paradigm, which we adapt specifically for tool invocation scenarios, enabling models to dynamically interleave complex reasoning with on-demand tool usage (including search APIs and Python interpreters).
To support this research, we introduce two new datasets: TAPO-easy-60K and TAPO-hard-18K, specifically designed to train and evaluate both fact-based reasoning and mathematical calculation capabilities. Our experiments on Qwen2.5-3B and Qwen2.5-7B models demonstrate the effectiveness of our approach, with both models achieving state-of-the-art performance on tasks requiring external knowledge and mathematical computation among methods with comparable parameters. Notably, TAPO achieves more efficient tool utilization than baseline methods while preventing excessive calls caused by reward hacking. These results highlight the significant potential of combining advanced reasoning with tool usage to enhance model performance in knowledge-intensive and computationally demanding tasks. 

---
# Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06953)  

**Abstract**: The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems. 

---
# LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN 

**Authors**: Hacane Hechehouche, Andre Antakli, Matthias Klusch  

**Link**: [PDF](https://arxiv.org/pdf/2510.06911)  

**Abstract**: There are many established semantic Web standards for implementing multi-agent driven applications. The AJAN framework allows to engineer multi-agent systems based on these standards. In particular, agent knowledge is represented in RDF/RDFS and OWL, while agent behavior models are defined with Behavior Trees and SPARQL to access and manipulate this knowledge. However, the appropriate definition of RDF/RDFS and SPARQL-based agent behaviors still remains a major hurdle not only for agent modelers in practice. For example, dealing with URIs is very error-prone regarding typos and dealing with complex SPARQL queries in large-scale environments requires a high learning curve. In this paper, we present an integrated development environment to overcome such hurdles of modeling AJAN agents and at the same time to extend the user community for AJAN by the possibility to leverage Large Language Models for agent engineering. 

---
# TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs 

**Authors**: Daria Ozerova, Ekaterina Trofimova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06878)  

**Abstract**: Iterative refinement has been a promising paradigm to enable large language models (LLMs) to resolve difficult reasoning and problem-solving tasks. One of the key challenges, however, is how to effectively search through the enormous search space of possible refinements. Existing methods typically fall back on predefined heuristics, which are troubled by the exploration-exploitation dilemma and cannot adapt based on past refinement outcomes. We introduce Tree-Guided Policy Refinement (TGPR), a novel framework that combines GRPO with a Thompson-Sampling-based tree search. TGPR explores both failed and successful refinement paths actively, with denser training trajectories and more adaptive policies. On HumanEval, MBPP, and APPS benchmarks, our method achieves up to +4.2 percentage points absolute improvement in pass@1 (on MBPP) and up to +12.51 percentage points absolute improvement in pass@10 (on APPS) compared to a competitive GRPO baseline. Apart from debugging code, TGPR focuses on a principled approach to combining learned policies with structured search methods, offering a general framework for enhancing iterative refinement and stateful reasoning in LLMs. 

---
# Autoformalizer with Tool Feedback 

**Authors**: Qi Guo, Jianing Wang, Jianfei Zhang, Deyang Kong, Xiangzhou Huang, Xiangyu Xi, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.06857)  

**Abstract**: Autoformalization addresses the scarcity of data for Automated Theorem Proving (ATP) by translating mathematical problems from natural language into formal statements. Efforts in recent work shift from directly prompting large language models to training an end-to-end formalizer model from scratch, achieving remarkable advancements. However, existing formalizer still struggles to consistently generate valid statements that meet syntactic validity and semantic consistency. To address this issue, we propose the Autoformalizer with Tool Feedback (ATF), a novel approach that incorporates syntactic and consistency information as tools into the formalization process. By integrating Lean 4 compilers for syntax corrections and employing a multi-LLMs-as-judge approach for consistency validation, the model is able to adaptively refine generated statements according to the tool feedback, enhancing both syntactic validity and semantic consistency. The training of ATF involves a cold-start phase on synthetic tool-calling data, an expert iteration phase to improve formalization capabilities, and Direct Preference Optimization to alleviate ineffective revisions. Experimental results show that ATF markedly outperforms a range of baseline formalizer models, with its superior performance further validated by human evaluations. Subsequent analysis reveals that ATF demonstrates excellent inference scaling properties. Moreover, we open-source Numina-ATF, a dataset containing 750K synthetic formal statements to facilitate advancements in autoformalization and ATP research. 

---
# Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration 

**Authors**: Zhi Zhang, Yan Liu, Zhejing Hu, Gong Chen, Sheng-hua Zhong, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06761)  

**Abstract**: Automating the end-to-end scientific research process poses a fundamental challenge: it requires both evolving high-level plans that are novel and sound, and executing these plans correctly amidst dynamic and uncertain conditions. To address this bilevel challenge, we propose a novel Double-Loop Multi-Agent (DLMA) framework to solve the given research problem automatically. The leader loop, composed of professor agents, is responsible for evolving research plans. It employs an evolutionary algorithm through involvement, improvement, and integration meetings to iteratively generate and refine a pool of research proposals, exploring the solution space effectively. The follower loop, composed of doctoral student agents, is responsible for executing the best-evolved plan. It dynamically adjusts the plan during implementation via pre-hoc and post-hoc meetings, ensuring each step (e.g., drafting, coding) is well-supported by contextual and external observations. Extensive experiments on benchmarks like ACLAward and Laboratory show that DLMA generates research papers that achieve state-of-the-art scores in automated evaluation, significantly outperforming strong baselines. Ablation studies confirm the critical roles of both loops, with evolution driving novelty and execution ensuring soundness. 

---
# Verifying Memoryless Sequential Decision-making of Large Language Models 

**Authors**: Dennis Gross, Helge Spieker, Arnaud Gotlieb  

**Link**: [PDF](https://arxiv.org/pdf/2510.06756)  

**Abstract**: We introduce a tool for rigorous and automated verification of large language model (LLM)- based policies in memoryless sequential decision-making tasks. Given a Markov decision process (MDP) representing the sequential decision-making task, an LLM policy, and a safety requirement expressed as a PCTL formula, our approach incrementally constructs only the reachable portion of the MDP guided by the LLM's chosen actions. Each state is encoded as a natural language prompt, the LLM's response is parsed into an action, and reachable successor states by the policy are expanded. The resulting formal model is checked with Storm to determine whether the policy satisfies the specified safety property. In experiments on standard grid world benchmarks, we show that open source LLMs accessed via Ollama can be verified when deterministically seeded, but generally underperform deep reinforcement learning baselines. Our tool natively integrates with Ollama and supports PRISM-specified tasks, enabling continuous benchmarking in user-specified sequential decision-making tasks and laying a practical foundation for formally verifying increasingly capable LLMs. 

---
# MultiCNKG: Integrating Cognitive Neuroscience, Gene, and Disease Knowledge Graphs Using Large Language Models 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard  

**Link**: [PDF](https://arxiv.org/pdf/2510.06742)  

**Abstract**: The advent of large language models (LLMs) has revolutionized the integration of knowledge graphs (KGs) in biomedical and cognitive sciences, overcoming limitations in traditional machine learning methods for capturing intricate semantic links among genes, diseases, and cognitive processes. We introduce MultiCNKG, an innovative framework that merges three key knowledge sources: the Cognitive Neuroscience Knowledge Graph (CNKG) with 2.9K nodes and 4.3K edges across 9 node types and 20 edge types; Gene Ontology (GO) featuring 43K nodes and 75K edges in 3 node types and 4 edge types; and Disease Ontology (DO) comprising 11.2K nodes and 8.8K edges with 1 node type and 2 edge types. Leveraging LLMs like GPT-4, we conduct entity alignment, semantic similarity computation, and graph augmentation to create a cohesive KG that interconnects genetic mechanisms, neurological disorders, and cognitive functions. The resulting MultiCNKG encompasses 6.9K nodes across 5 types (e.g., Genes, Diseases, Cognitive Processes) and 11.3K edges spanning 7 types (e.g., Causes, Associated with, Regulates), facilitating a multi-layered view from molecular to behavioral domains. Assessments using metrics such as precision (85.20%), recall (87.30%), coverage (92.18%), graph consistency (82.50%), novelty detection (40.28%), and expert validation (89.50%) affirm its robustness and coherence. Link prediction evaluations with models like TransE (MR: 391, MRR: 0.411) and RotatE (MR: 263, MRR: 0.395) show competitive performance against benchmarks like FB15k-237 and WN18RR. This KG advances applications in personalized medicine, cognitive disorder diagnostics, and hypothesis formulation in cognitive neuroscience. 

---
# Inefficiencies of Meta Agents for Agent Design 

**Authors**: Batu El, Mert Yuksekgonul, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06711)  

**Abstract**: Recent works began to automate the design of agentic systems using meta-agents that propose and iteratively refine new agent architectures. In this paper, we examine three key challenges in a common class of meta-agents. First, we investigate how a meta-agent learns across iterations and find that simply expanding the context with all previous agents, as proposed by previous works, performs worse than ignoring prior designs entirely. We show that the performance improves with an evolutionary approach. Second, although the meta-agent designs multiple agents during training, it typically commits to a single agent at test time. We find that the designed agents have low behavioral diversity, limiting the potential for their complementary use. Third, we assess when automated design is economically viable. We find that only in a few cases--specifically, two datasets--the overall cost of designing and deploying the agents is lower than that of human-designed agents when deployed on over 15,000 examples. In contrast, the performance gains for other datasets do not justify the design cost, regardless of scale. 

---
# Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support 

**Authors**: Zhao, Tiantian Zhang, Hanchen Su, Yufeng, Zhang, Shaowei Su, Mingzhi Xu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06674)  

**Abstract**: We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system. 

---
# Fine-Grained Emotion Recognition via In-Context Learning 

**Authors**: Zhaochun Ren, Zhou Yang, Chenglong Ye, Haizhou Sun, Chao Chen, Xiaofei Zhu, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06600)  

**Abstract**: Fine-grained emotion recognition aims to identify the emotional type in queries through reasoning and decision-making processes, playing a crucial role in various systems. Recent methods use In-Context Learning (ICL), enhancing the representation of queries in the reasoning process through semantically similar examples, while further improving emotion recognition by explaining the reasoning mechanisms. However, these methods enhance the reasoning process but overlook the decision-making process. This paper investigates decision-making in fine-grained emotion recognition through prototype theory. We show that ICL relies on similarity matching between query representations and emotional prototypes within the model, where emotion-accurate representations are critical. However, semantically similar examples often introduce emotional discrepancies, hindering accurate representations and causing errors. To address this, we propose Emotion In-Context Learning (EICL), which introduces emotionally similar examples and uses a dynamic soft-label strategy to improve query representations in the emotion reasoning process. A two-stage exclusion strategy is then employed to assess similarity from multiple angles, further optimizing the decision-making process. Extensive experiments show that EICL significantly outperforms ICL on multiple datasets. 

---
# WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks 

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Shiyu Chang, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06587)  

**Abstract**: Large language model (LLM) agents are becoming competent at straightforward web tasks, such as opening an item page or submitting a form, but still struggle with objectives that require long horizon navigation, large scale information extraction, and reasoning under constraints. We present WebDART, a general framework that enables a single LLM to handle such complex chores. WebDART (i) dynamically decomposes each objective into three focused subtasks: navigation, information extraction, and execution, so the model concentrates on one skill at a time, and (ii) continuously replans the decomposition as new webpages are revealed, taking advantage of newly discovered filters or shortcuts and avoiding redundant exploration. Evaluated on WebChoreArena, WebDART lifts success rates by up to 13.7 percentage points over previous SOTA agents, while matching their performance on the easier WebArena suite and completing tasks with up to 14.7 fewer navigation steps. 

---
# Auto-Prompt Ensemble for LLM Judge 

**Authors**: Jiajie Li, Huayi Zhang, Peng Lin, Jinjun Xiong, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06538)  

**Abstract**: We present a novel framework that improves the reliability of LLM judges by selectively augmenting LLM with auxiliary evaluation dimensions. Existing LLM judges often miss crucial evaluation dimensions because they fail to recognize the implicit standards underlying human assessments. To address this challenge, we propose the Auto-Prompt Ensemble (APE), an adaptive framework that automatically learns evaluation dimensions from its failure cases. APE incorporates a confidence-based ensemble mechanism to decide when to adopt the judgments from additional evaluation dimensions through a novel confidence estimation approach called Collective Confidence. Extensive experiments demonstrate that APE improves the reliability of LLM Judge across diverse standard benchmarks. For instance, APE enhances GPT-4o agreement rate on Reward Bench from 87.2% to 90.5% in the zero-shot setting. Overall, APE provides a principled approach for LLM Judge to leverage test-time computation, and bridge the evaluation gap between human and LLM judges. 

---
# Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them 

**Authors**: Jiahe Jin, Abhijay Paladugu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.06534)  

**Abstract**: Agentic search leverages large language models (LLMs) to interpret complex user information needs and execute a multi-step process of planning, searching, and synthesizing information to provide answers. This paradigm introduces unique challenges for LLMs' reasoning and agentic capabilities when interacting with retrieval systems and the broader web. In this paper, we propose a reasoning-driven LLM-based pipeline to study effective reasoning behavior patterns in agentic search. Using this pipeline, we analyze successful agentic search trajectories and identify four beneficial reasoning behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Based on these findings, we propose a technique called Behavior Priming to train more effective agentic search models. It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL). Experiments on three benchmarks (GAIA, WebWalker, and HLE) demonstrate that behavior priming yields over 35% gains in Llama3.2-3B and Qwen3-1.7B compared to directly training agentic search models with RL. Crucially, we demonstrate that the desired reasoning behaviors in the SFT data, rather than the correctness of the final answer, is the critical factor for achieving strong final performance after RL: fine-tuning on trajectories with desirable reasoning behaviors but incorrect answers leads to better performance than fine-tuning on trajectories with correct answers. Our analysis further reveals the underlying mechanism: the introduced reasoning behaviors endow models with more effective exploration (higher pass@k and entropy) and test-time scaling (longer trajectories) capabilities, providing a strong foundation for RL. Our code will be released as open source. 

---
# PuzzlePlex: Benchmarking Foundation Models on Reasoning and Planning with Puzzles 

**Authors**: Yitao Long, Yuru Jiang, Hongjun Liu, Yilun Zhao, Jingchen Sun, Yiqiu Shen, Chen Zhao, Arman Cohan, Dennis Shasha  

**Link**: [PDF](https://arxiv.org/pdf/2510.06475)  

**Abstract**: This work investigates the reasoning and planning capabilities of foundation models and their scalability in complex, dynamic environments. We introduce PuzzlePlex, a benchmark designed to assess these capabilities through a diverse set of puzzles. PuzzlePlex consists of 15 types of puzzles, including deterministic and stochastic games of varying difficulty, as well as single-player and two-player scenarios. The PuzzlePlex framework provides a comprehensive environment for each game, and supports extensibility to generate more challenging instances as foundation models evolve. Additionally, we implement customized game-playing strategies for comparison. Building on this benchmark, we develop fine-grained metrics to measure performance and conduct an in-depth analysis of frontier foundation models across two settings: instruction-based and code-based. Furthermore, we systematically investigate their scaling limits. Our findings show that reasoning models outperform others in instruction-based settings, while code-based execution presents greater challenges but offers a scalable and efficient alternative. PuzzlePlex enables targeted evaluation and guides future improvements in reasoning, planning, and generalization for foundation models. 

---
# Flavonoid Fusion: Creating a Knowledge Graph to Unveil the Interplay Between Food and Health 

**Authors**: Aryan Singh Dalal, Yinglun Zhang, Duru Doğan, Atalay Mert İleri, Hande Küçük McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2510.06433)  

**Abstract**: The focus on "food as medicine" is gaining traction in the field of health and several studies conducted in the past few years discussed this aspect of food in the literature. However, very little research has been done on representing the relationship between food and health in a standardized, machine-readable format using a semantic web that can help us leverage this knowledge effectively. To address this gap, this study aims to create a knowledge graph to link food and health through the knowledge graph's ability to combine information from various platforms focusing on flavonoid contents of food found in the USDA databases and cancer connections found in the literature. We looked closely at these relationships using KNARM methodology and represented them in machine-operable format. The proposed knowledge graph serves as an example for researchers, enabling them to explore the complex interplay between dietary choices and disease management. Future work for this study involves expanding the scope of the knowledge graph by capturing nuances, adding more related data, and performing inferences on the acquired knowledge to uncover hidden relationships. 

---
# Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory? 

**Authors**: Aochong Oliver Li, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.06410)  

**Abstract**: Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs. 

---
# Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks 

**Authors**: Wentao Deng, Jiahuan Pei, Zhiwei Xu, Zhaochun Ren, Zhumin Chen, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.06307)  

**Abstract**: A multi-agent system (MAS) enhances its capacity to solve complex natural language processing (NLP) tasks through collaboration among multiple agents, where consensus-seeking serves as a fundamental mechanism. However, existing consensus-seeking approaches typically rely on voting mechanisms to judge consensus, overlooking contradictions in system-internal beliefs that destabilize the consensus. Moreover, these methods often involve agents updating their results through indiscriminate collaboration with every other agent. Such uniform interaction fails to identify the optimal collaborators for each agent, hindering the emergence of a stable consensus. To address these challenges, we provide a theoretical framework for selecting optimal collaborators that maximize consensus stability. Based on the theorems, we propose the Belief-Calibrated Consensus Seeking (BCCS) framework to facilitate stable consensus via selecting optimal collaborators and calibrating the consensus judgment by system-internal beliefs. Experimental results on the MATH and MMLU benchmark datasets demonstrate that the proposed BCCS framework outperforms the best existing results by 2.23% and 3.95% of accuracy on challenging tasks, respectively. Our code and data are available at this https URL. 

---
# Requirements for Game-Based Learning Design Framework for Information System Integration in the Context of Post-Merger Integration 

**Authors**: Ksenija Lace, Marite Kirikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06302)  

**Abstract**: Post-merger integration states unique challenges for professionals responsible for information system integration aimed on alignment and combination diverse system architectures of merging organizations. Although the theoretical and practical guidance exists for post-merger integration on the business level, there is a significant gap in training for information system integration in this context. In prior research specific methods AMILI (Support method for informed decision identification) and AMILP (Support method for informed decision-making) were introduced for the support of information system integration decisions in the post-merger integration. But during the practical application was reported high learning curve and low learner motivation. This paper explores how game-based learning design can address these limitations by transforming static method training into engaging learning experience. The study analyzes foundational learning theories, cognitive load and motivation models, and serious game design frameworks to identify the essential requirements for a game-based learning design framework tailored to information system integration in post-merger integration. Requirements are structured in two components: the transformation process and resulting learning experience. The paper concludes with a plan for developing and evaluating the proposed framework through iterative design and real-world validation. 

---
# BuilderBench -- A benchmark for generalist agents 

**Authors**: Raj Ghugare, Catherine Ji, Kathryn Wantlin, Jin Schofield, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2510.06288)  

**Abstract**: Today's AI models learn primarily through mimicry and sharpening, so it is not surprising that they struggle to solve problems beyond the limits set by existing data. To solve novel problems, agents should acquire skills for exploring and learning through experience. Finding a scalable learning mechanism for developing agents that learn through interaction remains a major open problem. In this work, we introduce BuilderBench, a benchmark to accelerate research into agent pre-training that centers open-ended exploration. BuilderBench requires agents to learn how to build any structure using blocks. BuilderBench is equipped with $(1)$ a hardware accelerated simulator of a robotic agent interacting with various physical blocks, and $(2)$ a task-suite with over 42 diverse target structures that are carefully curated to test an understanding of physics, mathematics, and long-horizon planning. During training, agents have to explore and learn general principles about the environment without any external supervision. During evaluation, agents have to build the unseen target structures from the task suite. Solving these tasks requires a sort of \emph{embodied reasoning} that is not reflected in words but rather in actions, experimenting with different strategies and piecing them together. Our experiments show that many of these tasks challenge the current iteration of algorithms. Hence, we also provide a ``training wheels'' protocol, in which agents are trained and evaluated to build a single target structure from the task suite. Finally, we provide single-file implementations of six different algorithms as a reference point for researchers. 

---
# Bridging Reasoning to Learning: Unmasking Illusions using Complexity Out of Distribution Generalization 

**Authors**: Mohammad Mahdi Samiei Paqaleh, Arash Marioriyad, Arman Tahmasebi-Zadeh, Mohamadreza Fereydooni, Mahdi Ghaznavai, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2510.06274)  

**Abstract**: Recent progress has pushed AI frontiers from pattern recognition tasks toward problems that require step by step, System2 style reasoning, especially with large language models. Yet, unlike learning, where generalization and out of distribution (OoD) evaluation concepts are well formalized, there is no clear, consistent definition or metric for reasoning ability. We propose Complexity Out of Distribution (Complexity OoD) generalization as a framework and problem setting to define and measure reasoning. A model exhibits Complexity OoD generalization when it maintains performance on test instances whose minimal required solution complexity, either representational (richer solution structure) or computational (more reasoning steps/program length), exceeds that of all training examples. We formalize complexity via solution description Kolmogorov complexity and operational proxies (e.g., object/relation counts; reasoning step counts), clarifying how Complexity OoD differs from length and compositional OoD. This lens unifies learning and reasoning: many cases solvable with System1 like processing at low complexity become System2 like under complexity pressure, while System2 can be viewed as generalization over solution structures. We translate this perspective into practice with recommendations for operationalizing Complexity OoD across the stack: incorporating complexity into benchmark and evaluation metric design, rethinking supervision to target solution traces, seeking and designing inductive biases for Complexity OoD generalization, addressing learning to reason spillovers such as spurious shortcuts, semantic robustness, catastrophic forgetting, and step wise calibration. Because Complexity OoD cannot be solved by scaling data alone, progress toward robust reasoning will require architectures and training regimes that explicitly model and allocate computation with respect to complexity. 

---
# AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning 

**Authors**: Zhanke Zhou, Chentao Cao, Xiao Feng, Xuan Li, Zongze Li, Xiangyu Lu, Jiangchao Yao, Weikai Huang, Linrui Xu, Tian Cheng, Guanyu Jiang, Yiming Zheng, Brando Miranda, Tongliang Liu, Sanmi Koyejo, Masashi Sugiyama, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.06261)  

**Abstract**: We present AlphaApollo, a self-evolving agentic reasoning system that aims to address two bottlenecks in foundation model (FM) reasoning-limited model-intrinsic capacity and unreliable test-time iteration. AlphaApollo orchestrates multiple models with professional tools to enable deliberate, verifiable reasoning. It couples (i) a computation tool (Python with numerical and symbolic libraries) and (ii) a retrieval tool (task-relevant external information) to execute exact calculations and ground decisions. The system further supports multi-round, multi-model solution evolution via a shared state map that records candidates, executable checks, and feedback for iterative refinement. In evaluations on AIME 2024/2025 across multiple models, AlphaApollo delivers consistent gains: +5.15% Average@32 and +23.34% Pass@32 for Qwen2.5-14B-Instruct, and +8.91% Average@32 with +26.67% Pass@32 for Llama-3.3-70B-Instruct. Tool-use analysis shows that more than 80% of tool calls are successfully executed, with consistent outperformance of non-tool baselines, thereby lifting the capability ceiling of FMs. More empirical results and implementation details will be updated at this https URL. 

---
# Artificial Hippocampus Networks for Efficient Long-Context Modeling 

**Authors**: Yunhao Fang, Weihao Yu, Shu Zhong, Qinghao Ye, Xuehan Xiong, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.07318)  

**Abstract**: Long-sequence modeling faces a fundamental trade-off between the efficiency of compressive fixed-size memory in RNN-like models and the fidelity of lossless growing memory in attention-based Transformers. Inspired by the Multi-Store Model in cognitive science, we introduce a memory framework of artificial neural networks. Our method maintains a sliding window of the Transformer's KV cache as lossless short-term memory, while a learnable module termed Artificial Hippocampus Network (AHN) recurrently compresses out-of-window information into a fixed-size compact long-term memory. To validate this framework, we instantiate AHNs using modern RNN-like architectures, including Mamba2, DeltaNet, and Gated DeltaNet. Extensive experiments on long-context benchmarks LV-Eval and InfiniteBench demonstrate that AHN-augmented models consistently outperform sliding window baselines and achieve performance comparable or even superior to full-attention models, while substantially reducing computational and memory requirements. For instance, augmenting the Qwen2.5-3B-Instruct with AHNs reduces inference FLOPs by 40.5% and memory cache by 74.0%, while improving its average score on LV-Eval (128k sequence length) from 4.41 to 5.88. Code is available at: this https URL. 

---
# Vibe Checker: Aligning Code Evaluation with Human Preference 

**Authors**: Ming Zhong, Xiang Zhou, Ting-Yun Chang, Qingze Wang, Nan Xu, Xiance Si, Dan Garrette, Shyam Upadhyay, Jeremiah Liu, Jiawei Han, Benoit Schillings, Jiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.07315)  

**Abstract**: Large Language Models (LLMs) have catalyzed vibe coding, where users leverage LLMs to generate and iteratively refine code through natural language interactions until it passes their vibe check. Vibe check is tied to real-world human preference and goes beyond functionality: the solution should feel right, read cleanly, preserve intent, and remain correct. However, current code evaluation remains anchored to pass@k and captures only functional correctness, overlooking the non-functional instructions that users routinely apply. In this paper, we hypothesize that instruction following is the missing piece underlying vibe check that represents human preference in coding besides functional correctness. To quantify models' code instruction following capabilities with measurable signals, we present VeriCode, a taxonomy of 30 verifiable code instructions together with corresponding deterministic verifiers. We use the taxonomy to augment established evaluation suites, resulting in Vibe Checker, a testbed to assess both code instruction following and functional correctness. Upon evaluating 31 leading LLMs, we show that even the strongest models struggle to comply with multiple instructions and exhibit clear functional regression. Most importantly, a composite score of functional correctness and instruction following correlates the best with human preference, with the latter emerging as the primary differentiator on real-world programming tasks. Our work identifies core factors of the vibe check, providing a concrete path for benchmarking and developing models that better align with user preferences in coding. 

---
# GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations 

**Authors**: Fabian Paischer, Gianluca Galletti, William Hornsby, Paul Setinek, Lorenzo Zanisi, Naomi Carey, Stanislas Pamela, Johannes Brandstetter  

**Link**: [PDF](https://arxiv.org/pdf/2510.07314)  

**Abstract**: Nuclear fusion plays a pivotal role in the quest for reliable and sustainable energy production. A major roadblock to viable fusion power is understanding plasma turbulence, which significantly impairs plasma confinement, and is vital for next-generation reactor design. Plasma turbulence is governed by the nonlinear gyrokinetic equation, which evolves a 5D distribution function over time. Due to its high computational cost, reduced-order models are often employed in practice to approximate turbulent transport of energy. However, they omit nonlinear effects unique to the full 5D dynamics. To tackle this, we introduce GyroSwin, the first scalable 5D neural surrogate that can model 5D nonlinear gyrokinetic simulations, thereby capturing the physical phenomena neglected by reduced models, while providing accurate estimates of turbulent heat this http URL (i) extends hierarchical Vision Transformers to 5D, (ii) introduces cross-attention and integration modules for latent 3D$\leftrightarrow$5D interactions between electrostatic potential fields and the distribution function, and (iii) performs channelwise mode separation inspired by nonlinear physics. We demonstrate that GyroSwin outperforms widely used reduced numerics on heat flux prediction, captures the turbulent energy cascade, and reduces the cost of fully resolved nonlinear gyrokinetics by three orders of magnitude while remaining physically verifiable. GyroSwin shows promising scaling laws, tested up to one billion parameters, paving the way for scalable neural surrogates for gyrokinetic simulations of plasma turbulence. 

---
# h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning 

**Authors**: Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London  

**Link**: [PDF](https://arxiv.org/pdf/2510.07312)  

**Abstract**: Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data. 

---
# MLE-Smith: Scaling MLE Tasks with Automated Multi-Agent Pipeline 

**Authors**: Rushi Qiang, Yuchen Zhuang, Anikait Singh, Percy Liang, Chao Zhang, Sherry Yang, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.07307)  

**Abstract**: While Language Models (LMs) have made significant progress in automating machine learning engineering (MLE), the acquisition of high-quality MLE training data is significantly constrained. Current MLE benchmarks suffer from low scalability and limited applicability because they rely on static, manually curated tasks, demanding extensive time and manual effort to produce. We introduce MLE-Smith, a fully automated multi-agent pipeline, to transform raw datasets into competition-style MLE challenges through an efficient generate-verify-execute paradigm for scaling MLE tasks with verifiable quality, real-world usability, and rich diversity. The proposed multi-agent pipeline in MLE-Smith drives structured task design and standardized refactoring, coupled with a hybrid verification mechanism that enforces strict structural rules and high-level semantic soundness. It further validates empirical solvability and real-world fidelity through interactive execution. We apply MLE-Smith to 224 of real-world datasets and generate 606 tasks spanning multiple categories, objectives, and modalities, demonstrating that MLE-Smith can work effectively across a wide range of real-world datasets. Evaluation on the generated tasks shows that the performance of eight mainstream and cutting-edge LLMs on MLE-Smith tasks is strongly correlated with their performance on carefully human-designed tasks, highlighting the effectiveness of the MLE-Smith to scaling up MLE tasks, while maintaining task quality. 

---
# Cocoon: A System Architecture for Differentially Private Training with Correlated Noises 

**Authors**: Donghwan Kim, Xin Gu, Jinho Baek, Timothy Lo, Younghoon Min, Kwangsik Shin, Jongryool Kim, Jongse Park, Kiwan Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07304)  

**Abstract**: Machine learning (ML) models memorize and leak training data, causing serious privacy issues to data owners. Training algorithms with differential privacy (DP), such as DP-SGD, have been gaining attention as a solution. However, DP-SGD adds a noise at each training iteration, which degrades the accuracy of the trained model. To improve accuracy, a new family of approaches adds carefully designed correlated noises, so that noises cancel out each other across iterations. We performed an extensive characterization study of these new mechanisms, for the first time to the best of our knowledge, and show they incur non-negligible overheads when the model is large or uses large embedding tables. Motivated by the analysis, we propose Cocoon, a hardware-software co-designed framework for efficient training with correlated noises. Cocoon accelerates models with embedding tables through pre-computing and storing correlated noises in a coalesced format (Cocoon-Emb), and supports large models through a custom near-memory processing device (Cocoon-NMP). On a real system with an FPGA-based NMP device prototype, Cocoon improves the performance by 2.33-10.82x(Cocoon-Emb) and 1.55-3.06x (Cocoon-NMP). 

---
# AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs 

**Authors**: Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xiaoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng Gu, Xiangqi Jin, Jiabing Yang, Kai Li, Zhifei Liu, Weijia Li, Cunxiang Wang, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07293)  

**Abstract**: Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks. 

---
# Evolutionary Profiles for Protein Fitness Prediction 

**Authors**: Jigang Fan, Xiaoran Jiao, Shengdong Lin, Zhanming Liang, Weian Mao, Chenchen Jing, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07286)  

**Abstract**: Predicting the fitness impact of mutations is central to protein engineering but constrained by limited assays relative to the size of sequence space. Protein language models (pLMs) trained with masked language modeling (MLM) exhibit strong zero-shot fitness prediction; we provide a unifying view by interpreting natural evolution as implicit reward maximization and MLM as inverse reinforcement learning (IRL), in which extant sequences act as expert demonstrations and pLM log-odds serve as fitness estimates. Building on this perspective, we introduce EvoIF, a lightweight model that integrates two complementary sources of evolutionary signal: (i) within-family profiles from retrieved homologs and (ii) cross-family structural-evolutionary constraints distilled from inverse folding logits. EvoIF fuses sequence-structure representations with these profiles via a compact transition block, yielding calibrated probabilities for log-odds scoring. On ProteinGym (217 mutational assays; >2.5M mutants), EvoIF and its MSA-enabled variant achieve state-of-the-art or competitive performance while using only 0.15% of the training data and fewer parameters than recent large models. Ablations confirm that within-family and cross-family profiles are complementary, improving robustness across function types, MSA depths, taxa, and mutation depths. The codes will be made publicly available at this https URL. 

---
# GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection (Preprint) 

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Qi Hu, Yan Li, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07285)  

**Abstract**: The escalating complexity of network threats and the inherent class imbalance in traffic data present formidable challenges for modern Intrusion Detection Systems (IDS). While Graph Neural Networks (GNNs) excel in modeling topological structures and Temporal Convolutional Networks (TCNs) are proficient in capturing time-series dependencies, a framework that synergistically integrates both while explicitly addressing data imbalance remains an open challenge. This paper introduces a novel deep learning framework, named Gated Temporal Convolutional Network and Graph (GTCN-G), engineered to overcome these limitations. Our model uniquely fuses a Gated TCN (G-TCN) for extracting hierarchical temporal features from network flows with a Graph Convolutional Network (GCN) designed to learn from the underlying graph structure. The core innovation lies in the integration of a residual learning mechanism, implemented via a Graph Attention Network (GAT). This mechanism preserves original feature information through residual connections, which is critical for mitigating the class imbalance problem and enhancing detection sensitivity for rare malicious activities (minority classes). We conducted extensive experiments on two public benchmark datasets, UNSW-NB15 and ToN-IoT, to validate our approach. The empirical results demonstrate that the proposed GTCN-G model achieves state-of-the-art performance, significantly outperforming existing baseline models in both binary and multi-class classification tasks. 

---
# Online Rubrics Elicitation from Pairwise Comparisons 

**Authors**: MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Yunzhong He, Afra Feyza Akyürek  

**Link**: [PDF](https://arxiv.org/pdf/2510.07284)  

**Abstract**: Rubrics provide a flexible way to train LLMs on open-ended long-form answers where verifiable rewards are not applicable and human preferences provide coarse signals. Prior work shows that reinforcement learning with rubric-based rewards leads to consistent gains in LLM post-training. Most existing approaches rely on rubrics that remain static over the course of training. Such static rubrics, however, are vulnerable to reward-hacking type behaviors and fail to capture emergent desiderata that arise during training. We introduce Online Rubrics Elicitation (OnlineRubrics), a method that dynamically curates evaluation criteria in an online manner through pairwise comparisons of responses from current and reference policies. This online process enables continuous identification and mitigation of errors as training proceeds. Empirically, this approach yields consistent improvements of up to 8% over training exclusively with static rubrics across AlpacaEval, GPQA, ArenaHard as well as the validation sets of expert questions and rubrics. We qualitatively analyze the elicited criteria and identify prominent themes such as transparency, practicality, organization, and reasoning. 

---
# On the false election between regulation and innovation. Ideas for regulation through the responsible use of artificial intelligence in research and education.[Spanish version] 

**Authors**: Pompeu Casanovas  

**Link**: [PDF](https://arxiv.org/pdf/2510.07268)  

**Abstract**: This short essay is a reworking of the answers offered by the author at the Debate Session of the AIHUB (CSIC) and EduCaixa Summer School, organized by Marta Garcia-Matos and Lissette Lemus, and coordinated by Albert Sabater (OEIAC, UG), with the participation of Vanina Martinez-Posse (IIIA-CSIC), Eulalia Soler (Eurecat) and Pompeu Casanovas (IIIA-CSIC) on July 4th 2025. Albert Sabater posed three questions: (1) How can regulatory frameworks priori-tise the protection of fundamental rights (privacy, non-discrimination, autonomy, etc.) in the development of AI, without falling into the false dichotomy between regulation and innova-tion? (2) Given the risks of AI (bias, mass surveillance, manipulation), what examples of regu-lations or policies have demonstrated that it is possible to foster responsible innovation, putting the public interest before profitability, without giving in to competitive pressure from actors such as China or the US? (3) In a scenario where the US prioritizes flexibility, what mecha-nisms could ensure that international cooperation in AI does not become a race to the bottom in rights, but rather a global standard of accountability? The article attempts to answer these three questions and concludes with some reflections on the relevance of the answers for education and research. 

---
# LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation 

**Authors**: Joseph Enguehard, Morgane Van Ermengem, Kate Atkinson, Sujeong Cha, Arijit Ghosh Chowdhury, Prashanth Kallur Ramaswamy, Jeremy Roghair, Hannah R Marlowe, Carina Suzana Negreanu, Kitty Boxall, Diana Mincu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07243)  

**Abstract**: Evaluating large language model (LLM) outputs in the legal domain presents unique challenges due to the complex and nuanced nature of legal analysis. Current evaluation approaches either depend on reference data, which is costly to produce, or use standardized assessment methods, both of which have significant limitations for legal applications.
Although LLM-as-a-Judge has emerged as a promising evaluation technique, its reliability and effectiveness in legal contexts depend heavily on evaluation processes unique to the legal industry and how trustworthy the evaluation appears to the human legal expert. This is where existing evaluation methods currently fail and exhibit considerable variability.
This paper aims to close the gap: a) we break down lengthy responses into 'Legal Data Points' (LDPs), self-contained units of information, and introduce a novel, reference-free evaluation methodology that reflects how lawyers evaluate legal answers; b) we demonstrate that our method outperforms a variety of baselines on both our proprietary dataset and an open-source dataset (LegalBench); c) we show how our method correlates more closely with human expert evaluations and helps improve inter-annotator agreement; and finally d) we open source our Legal Data Points for a subset of LegalBench used in our experiments, allowing the research community to replicate our results and advance research in this vital area of LLM evaluation on legal question-answering. 

---
# Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships 

**Authors**: Donggyu Lee, Sungwon Park, Yerin Hwang, Hyunwoo Oh, Hyoshin Kim, Jungwon Kim, Meeyoung Cha, Sangyoon Park, Jihee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.07231)  

**Abstract**: Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications. 

---
# Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation 

**Authors**: Arjun Krishnakumar, Rhea Sanjay Sukthanker, Hannan Javed Mahadik, Gabriela Kadlecová, Vladyslav Moroshan, Timur Carstensen, Frank Hutter, Aaron Klein  

**Link**: [PDF](https://arxiv.org/pdf/2510.07227)  

**Abstract**: Small Language models (SLMs) offer an efficient and accessible alternative to Large Language Models (LLMs), delivering strong performance while using far fewer resources. We introduce a simple and effective framework for pretraining SLMs that brings together three complementary ideas. First, we identify structurally sparse sub-network initializations that consistently outperform randomly initialized models of similar size under the same compute budget. Second, we use evolutionary search to automatically discover high-quality sub-network initializations, providing better starting points for pretraining. Third, we apply knowledge distillation from larger teacher models to speed up training and improve generalization. Together, these components make SLM pretraining substantially more efficient: our best model, discovered using evolutionary search and initialized with LLM weights, matches the validation perplexity of a comparable Pythia SLM while requiring 9.2x fewer pretraining tokens. We release all code and models at this https URL, offering a practical and reproducible path toward cost-efficient small language model development at scale. 

---
# GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation 

**Authors**: Wen Ye, Zhaocheng Liu, Yuwei Gui, Tingyu Yuan, Yunyue Su, Bowen Fang, Chaoyang Zhao, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07217)  

**Abstract**: Text-to-image synthesis has made remarkable progress, yet accurately interpreting complex and lengthy prompts remains challenging, often resulting in semantic inconsistencies and missing details. Existing solutions, such as fine-tuning, are model-specific and require training, while prior automatic prompt optimization (APO) approaches typically lack systematic error analysis and refinement strategies, resulting in limited reliability and effectiveness. Meanwhile, test-time scaling methods operate on fixed prompts and on noise or sample numbers, limiting their interpretability and adaptability. To solve these, we introduce a flexible and efficient test-time prompt optimization strategy that operates directly on the input text. We propose a plug-and-play multi-agent system called GenPilot, integrating error analysis, clustering-based adaptive exploration, fine-grained verification, and a memory module for iterative optimization. Our approach is model-agnostic, interpretable, and well-suited for handling long and complex prompts. Simultaneously, we summarize the common patterns of errors and the refinement strategy, offering more experience and encouraging further exploration. Experiments on DPG-bench and Geneval with improvements of up to 16.9% and 5.7% demonstrate the strong capability of our methods in enhancing the text and image consistency and structural coherence of generated images, revealing the effectiveness of our test-time prompt optimization strategy. The code is available at this https URL. 

---
# Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models 

**Authors**: Chengzhi Zhong, Fei Cheng, Qianying Liu, Yugo Murawaki, Chenhui Chu, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07213)  

**Abstract**: Large language models exhibit strong multilingual capabilities despite limited exposure to non-English data. Prior studies show that English-centric large language models map multilingual content into English-aligned representations at intermediate layers and then project them back into target-language token spaces in the final layer. From this observation, we hypothesize that this cross-lingual transition is governed by a small and sparse set of dimensions, which occur at consistent indices across the intermediate to final layers. Building on this insight, we introduce a simple, training-free method to identify and manipulate these dimensions, requiring only as few as 50 sentences of either parallel or monolingual data. Experiments on a multilingual generation control task reveal the interpretability of these dimensions, demonstrating that the interventions in these dimensions can switch the output language while preserving semantic content, and that it surpasses the performance of prior neuron-based approaches at a substantially lower cost. 

---
# HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving 

**Authors**: Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz  

**Link**: [PDF](https://arxiv.org/pdf/2510.07210)  

**Abstract**: We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners. 

---
# Resolution scaling governs DINOv3 transfer performance in chest radiograph classification 

**Authors**: Soroosh Tayebi Arasteh, Mina Shaigan, Christiane Kuhl, Jakob Nikolas Kather, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2510.07191)  

**Abstract**: Self-supervised learning (SSL) has advanced visual representation learning, but its value in chest radiography, a high-volume imaging modality with fine-grained findings, remains unclear. Meta's DINOv3 extends earlier SSL models through Gram-anchored self-distillation. Whether these design choices improve transfer learning for chest radiography has not been systematically tested. We benchmarked DINOv3 against DINOv2 and ImageNet initialization across seven datasets (n>814,000). Two representative backbones were evaluated: ViT-B/16 and ConvNeXt-B. Images were analyzed at 224x224, 512x512, and 1024x1024 pixels. We additionally assessed frozen features from a 7B model. The primary outcome was mean AUROC across labels. At 224x224, DINOv3 and DINOv2 achieved comparable performance on adult datasets. Increasing resolution to 512x512 yielded consistent improvements for DINOv3 over both DINOv2 and ImageNet. In contrast, results in pediatric cohort showed no differences across initializations. Across all settings, ConvNeXt-B outperformed ViT-B/16. Models using frozen DINOv3-7B features underperformed relative to fully finetuned 86-89M-parameter backbones, highlighting the importance of domain adaptation. Scaling to 1024x1024 did not further improve accuracy. Resolution-related gains were most evident for boundary-dependent and small focal abnormalities. In chest radiography, higher input resolution is critical for leveraging the benefits of modern self-supervised models. 512x512 pixels represent a practical upper limit where DINOv3-initialized ConvNeXt-B networks provide the strongest performance, while larger inputs offer minimal return on cost. Clinically, these findings support use of finetuned, mid-sized backbones at 512x512 for chest radiograph interpretation, with the greatest gains expected in detecting subtle or boundary-centered lesions relevant to emergency and critical care settings. 

---
# TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics 

**Authors**: Yi Han, Cheng Chi, Enshen Zhou, Shanyu Rong, Jingkun An, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07181)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks. 

---
# ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL 

**Authors**: Egor Cherepanov, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2510.07151)  

**Abstract**: Real-world robotic agents must act under partial observability and long horizons, where key cues may appear long before they affect decision making. However, most modern approaches rely solely on instantaneous information, without incorporating insights from the past. Standard recurrent or transformer models struggle with retaining and leveraging long-term dependencies: context windows truncate history, while naive memory extensions fail under scale and sparsity. We propose ELMUR (External Layer Memory with Update/Rewrite), a transformer architecture with structured external memory. Each layer maintains memory embeddings, interacts with them via bidirectional cross-attention, and updates them through an Least Recently Used (LRU) memory module using replacement or convex blending. ELMUR extends effective horizons up to 100,000 times beyond the attention window and achieves a 100% success rate on a synthetic T-Maze task with corridors up to one million steps. In POPGym, it outperforms baselines on more than half of the tasks. On MIKASA-Robo sparse-reward manipulation tasks with visual observations, it nearly doubles the performance of strong baselines. These results demonstrate that structured, layer-local external memory offers a simple and scalable approach to decision making under partial observability. 

---
# A Multi-Agent Framework for Stateful Inference-Time Search 

**Authors**: Arshika Lalan, Rajat Ghosh, Aditya Kolsur, Debojyoti Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.07147)  

**Abstract**: Recent work explores agentic inference-time techniques to perform structured, multi-step reasoning. However, stateless inference often struggles on multi-step tasks due to the absence of persistent state. Moreover, task-specific fine-tuning or instruction-tuning often achieve surface-level code generation but remain brittle on tasks requiring deeper reasoning and long-horizon dependencies. To address these limitations, we propose stateful multi-agent evolutionary search, a training-free framework that departs from prior stateless approaches by combining (i) persistent inference-time state, (ii) adversarial mutation, and (iii) evolutionary preservation. We demonstrate its effectiveness in automated unit test generation through the generation of edge cases. We generate robust edge cases using an evolutionary search process, where specialized agents sequentially propose, mutate, and score candidates. A controller maintains persistent state across generations, while evolutionary preservation ensures diversity and exploration across all possible cases. This yields a generalist agent capable of discovering robust, high-coverage edge cases across unseen codebases. Experiments show our stateful multi-agent inference framework achieves substantial gains in coverage over stateless single-step baselines, evaluated on prevalent unit-testing benchmarks such as HumanEval and TestGenEvalMini and using three diverse LLM families - Llama, Gemma, and GPT. These results indicate that combining persistent inference-time state with evolutionary search materially improves unit-test generation. 

---
# Comparing human and language models sentence processing difficulties on complex structures 

**Authors**: Samuel Joseph Amouyal, Aya Meltzer-Asscher, Jonathan Berant  

**Link**: [PDF](https://arxiv.org/pdf/2510.07141)  

**Abstract**: Large language models (LLMs) that fluently converse with humans are a reality - but do LLMs experience human-like processing difficulties? We systematically compare human and LLM sentence comprehension across seven challenging linguistic structures. We collect sentence comprehension data from humans and five families of state-of-the-art LLMs, varying in size and training procedure in a unified experimental framework. Our results show LLMs overall struggle on the target structures, but especially on garden path (GP) sentences. Indeed, while the strongest models achieve near perfect accuracy on non-GP structures (93.7% for GPT-5), they struggle on GP structures (46.8% for GPT-5). Additionally, when ranking structures based on average performance, rank correlation between humans and models increases with parameter count. For each target structure, we also collect data for their matched baseline without the difficult structure. Comparing performance on the target vs. baseline sentences, the performance gap observed in humans holds for LLMs, with two exceptions: for models that are too weak performance is uniformly low across both sentence types, and for models that are too strong the performance is uniformly high. Together, these reveal convergence and divergence in human and LLM sentence comprehension, offering new insights into the similarity of humans and LLMs. 

---
# TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking 

**Authors**: Jiahang Liu, Yunpeng Qi, Jiazhao Zhang, Minghan Li, Shaoan Wang, Kui Wu, Hanjing Ye, Hong Zhang, Zhibo Chen, Fangwei Zhong, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07134)  

**Abstract**: Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios. 

---
# A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model 

**Authors**: Tony Zhang, Burak Kantarci, Umair Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2510.07133)  

**Abstract**: Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety. 

---
# Graph Conditioned Diffusion for Controllable Histopathology Image Generation 

**Authors**: Sarah Cechnicka, Matthew Baugh, Weitong Zhang, Mischa Dombrowski, Zhe Li, Johannes C. Paetzold, Candice Roufosse, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2510.07129)  

**Abstract**: Recent advances in Diffusion Probabilistic Models (DPMs) have set new standards in high-quality image synthesis. Yet, controlled generation remains challenging, particularly in sensitive areas such as medical imaging. Medical images feature inherent structure such as consistent spatial arrangement, shape or texture, all of which are critical for diagnosis. However, existing DPMs operate in noisy latent spaces that lack semantic structure and strong priors, making it difficult to ensure meaningful control over generated content. To address this, we propose graph-based object-level representations for Graph-Conditioned-Diffusion. Our approach generates graph nodes corresponding to each major structure in the image, encapsulating their individual features and relationships. These graph representations are processed by a transformer module and integrated into a diffusion model via the text-conditioning mechanism, enabling fine-grained control over generation. We evaluate this approach using a real-world histopathology use case, demonstrating that our generated data can reliably substitute for annotated patient data in downstream segmentation tasks. The code is available here. 

---
# Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning 

**Authors**: Taylor Sorensen, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07105)  

**Abstract**: Many natural language processing (NLP) tasks involve subjectivity, ambiguity, or legitimate disagreement between annotators. In this paper, we outline our system for modeling human variation. Our system leverages language models' (LLMs) in-context learning abilities, along with a two-step meta-learning training procedure for 1) post-training on many datasets requiring in-context learning and 2) specializing the model via in-context meta-learning to the particular data distribution of interest. We also evaluate the performance of our system submission to the Learning With Disagreements (LeWiDi) competition, where it was the overall winner on both tasks. Additionally, we perform an ablation study to measure the importance of each system component. We find that including rater examples in-context is crucial for our system's performance, dataset-specific fine-tuning is helpful on the larger datasets, post-training on other in-context datasets is helpful on one of the competition datasets, and that performance improves with model scale. 

---
# Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report 

**Authors**: Riccardo Mereu, Aidan Scannell, Yuxin Hou, Yi Zhao, Aditya Jitta, Antonio Dominguez, Luigi Acerbi, Amos Storkey, Paul Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07092)  

**Abstract**: World models are a powerful paradigm in AI and robotics, enabling agents to reason about the future by predicting visual observations or compact latent states. The 1X World Model Challenge introduces an open-source benchmark of real-world humanoid interaction, with two complementary tracks: sampling, focused on forecasting future image frames, and compression, focused on predicting future discrete latent codes. For the sampling track, we adapt the video generation foundation model Wan-2.2 TI2V-5B to video-state-conditioned future frame prediction. We condition the video generation on robot states using AdaLN-Zero, and further post-train the model using LoRA. For the compression track, we train a Spatio-Temporal Transformer model from scratch. Our models achieve 23.0 dB PSNR in the sampling task and a Top-500 CE of 6.6386 in the compression task, securing 1st place in both challenges. 

---
# HTMformer: Hybrid Time and Multivariate Transformer for Time Series Forecasting 

**Authors**: Tan Wang, Yun Wei Dong, Tao Zhang, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07084)  

**Abstract**: Transformer-based methods have achieved impressive results in time series forecasting. However, existing Transformers still exhibit limitations in sequence modeling as they tend to overemphasize temporal dependencies. This incurs additional computational overhead without yielding corresponding performance gains. We find that the performance of Transformers is highly dependent on the embedding method used to learn effective representations. To address this issue, we extract multivariate features to augment the effective information captured in the embedding layer, yielding multidimensional embeddings that convey richer and more meaningful sequence representations. These representations enable Transformer-based forecasters to better understand the series. Specifically, we introduce Hybrid Temporal and Multivariate Embeddings (HTME). The HTME extractor integrates a lightweight temporal feature extraction module with a carefully designed multivariate feature extraction module to provide complementary features, thereby achieving a balance between model complexity and performance. By combining HTME with the Transformer architecture, we present HTMformer, leveraging the enhanced feature extraction capability of the HTME extractor to build a lightweight forecaster. Experiments conducted on eight real-world datasets demonstrate that our approach outperforms existing baselines in both accuracy and efficiency. 

---
# Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications 

**Authors**: Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07077)  

**Abstract**: Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data collection strategies, publicly available datasets, data augmentation methods, and evaluation benchmarks. Throughout this comprehensive survey, this paper aims to offer practical guidance for the robotics community in applying VLAs to real-world robotic systems. All references categorized by training approach, evaluation method, modality, and dataset are available in the table on our project website: this https URL . 

---
# LuxInstruct: A Cross-Lingual Instruction Tuning Dataset For Luxembourgish 

**Authors**: Fred Philippy, Laura Bernardy, Siwen Guo, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2510.07074)  

**Abstract**: Instruction tuning has become a key technique for enhancing the performance of large language models, enabling them to better follow human prompts. However, low-resource languages such as Luxembourgish face severe limitations due to the lack of high-quality instruction datasets. Traditional reliance on machine translation often introduces semantic misalignment and cultural inaccuracies. In this work, we address these challenges by creating a cross-lingual instruction tuning dataset for Luxembourgish, without resorting to machine-generated translations into it. Instead, by leveraging aligned data from English, French, and German, we build a high-quality dataset that preserves linguistic and cultural nuances. We provide evidence that cross-lingual instruction tuning not only improves representational alignment across languages but also the model's generative capabilities in Luxembourgish. This highlights how cross-lingual data curation can avoid the common pitfalls of machine-translated data and directly benefit low-resource language development. 

---
# Introspection in Learned Semantic Scene Graph Localisation 

**Authors**: Manshika Charvi Bissessur, Efimia Panagiotaki, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2510.07053)  

**Abstract**: This work investigates how semantics influence localisation performance and robustness in a learned self-supervised, contrastive semantic localisation framework. After training a localisation network on both original and perturbed maps, we conduct a thorough post-hoc introspection analysis to probe whether the model filters environmental noise and prioritises distinctive landmarks over routine clutter. We validate various interpretability methods and present a comparative reliability analysis. Integrated gradients and Attention Weights consistently emerge as the most reliable probes of learned behaviour. A semantic class ablation further reveals an implicit weighting in which frequent objects are often down-weighted. Overall, the results indicate that the model learns noise-robust, semantically salient relations about place definition, thereby enabling explainable registration under challenging visual and structural variations. 

---
# Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models 

**Authors**: Yuntao Gui, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07048)  

**Abstract**: Despite their remarkable natural language understanding capabilities, Large Language Models (LLMs) have been underutilized for retrieval tasks. We present Search-R3, a novel framework that addresses this limitation by adapting LLMs to generate search embeddings as a direct output of their reasoning process. Our approach exploits LLMs' chain-of-thought capabilities, allowing them to produce more effective embeddings by reasoning step-by-step through complex semantic analyses. We implement this through three complementary mechanisms. (1) a supervised learning stage enables the model's ability to produce quality embeddings, (2) a reinforcement learning (RL) methodology that optimizes embedding generation alongside reasoning, and (3) a specialized RL environment that efficiently handles evolving embedding representations without requiring complete corpus re-encoding at each training iteration. Our extensive evaluations on diverse benchmarks demonstrate that Search-R3 significantly outperforms prior methods by unifying the reasoning and embedding generation processes. This integrated post-training approach represents a substantial advancement in handling complex knowledge-intensive tasks that require both sophisticated reasoning and effective information retrieval. Project page: this https URL 

---
# Unified Molecule Pre-training with Flexible 2D and 3D Modalities: Single and Paired Modality Integration 

**Authors**: Tengwei Song, Min Wu, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07035)  

**Abstract**: Molecular representation learning plays a crucial role in advancing applications such as drug discovery and material design. Existing work leverages 2D and 3D modalities of molecular information for pre-training, aiming to capture comprehensive structural and geometric insights. However, these methods require paired 2D and 3D molecular data to train the model effectively and prevent it from collapsing into a single modality, posing limitations in scenarios where a certain modality is unavailable or computationally expensive to generate. To overcome this limitation, we propose FlexMol, a flexible molecule pre-training framework that learns unified molecular representations while supporting single-modality input. Specifically, inspired by the unified structure in vision-language models, our approach employs separate models for 2D and 3D molecular data, leverages parameter sharing to improve computational efficiency, and utilizes a decoder to generate features for the missing modality. This enables a multistage continuous learning process where both modalities contribute collaboratively during training, while ensuring robustness when only one modality is available during inference. Extensive experiments demonstrate that FlexMol achieves superior performance across a wide range of molecular property prediction tasks, and we also empirically demonstrate its effectiveness with incomplete data. Our code and data are available at this https URL. 

---
# Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge 

**Authors**: Shrestha Ghosh, Luca Giordano, Yujia Hu, Tuan-Phong Nguyen, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.07024)  

**Abstract**: LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge. 

---
# Federated Unlearning in the Wild: Rethinking Fairness and Data Discrepancy 

**Authors**: ZiHeng Huang, Di Wu, Jun Bai, Jiale Zhang, Sicong Cao, Ji Zhang, Yingjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07022)  

**Abstract**: Machine unlearning is critical for enforcing data deletion rights like the "right to be forgotten." As a decentralized paradigm, Federated Learning (FL) also requires unlearning, but realistic implementations face two major challenges. First, fairness in Federated Unlearning (FU) is often overlooked. Exact unlearning methods typically force all clients into costly retraining, even those uninvolved. Approximate approaches, using gradient ascent or distillation, make coarse interventions that can unfairly degrade performance for clients with only retained data. Second, most FU evaluations rely on synthetic data assumptions (IID/non-IID) that ignore real-world heterogeneity. These unrealistic benchmarks obscure the true impact of unlearning and limit the applicability of current methods. We first conduct a comprehensive benchmark of existing FU methods under realistic data heterogeneity and fairness conditions. We then propose a novel, fairness-aware FU approach, Federated Cross-Client-Constrains Unlearning (FedCCCU), to explicitly address both challenges. FedCCCU offers a practical and scalable solution for real-world FU. Experimental results show that existing methods perform poorly in realistic settings, while our approach consistently outperforms them. 

---
# Native Hybrid Attention for Efficient Sequence Modeling 

**Authors**: Jusen Du, Jiaxi Hu, Tao Zhang, Weigao Sun, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07019)  

**Abstract**: Transformers excel at sequence modeling but face quadratic complexity, while linear attention offers improved efficiency but often compromises recall accuracy over long contexts. In this work, we introduce Native Hybrid Attention (NHA), a novel hybrid architecture of linear and full attention that integrates both intra \& inter-layer hybridization into a unified layer design. NHA maintains long-term context in key-value slots updated by a linear RNN, and augments them with short-term tokens from a sliding window. A single \texttt{softmax attention} operation is then applied over all keys and values, enabling per-token and per-head context-dependent weighting without requiring additional fusion parameters. The inter-layer behavior is controlled through a single hyperparameter, the sliding window size, which allows smooth adjustment between purely linear and full attention while keeping all layers structurally uniform. Experimental results show that NHA surpasses Transformers and other hybrid baselines on recall-intensive and commonsense reasoning tasks. Furthermore, pretrained LLMs can be structurally hybridized with NHA, achieving competitive accuracy while delivering significant efficiency gains. Code is available at this https URL. 

---
# Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages 

**Authors**: Neel Prabhanjan Rachamalla, Aravind Konakalla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.07000)  

**Abstract**: The effectiveness of Large Language Models (LLMs) depends heavily on the availability of high-quality post-training data, particularly instruction-tuning and preference-based examples. Existing open-source datasets, however, often lack multilingual coverage, cultural grounding, and suffer from task diversity gaps that are especially pronounced for Indian languages. We introduce a human-in-the-loop pipeline that combines translations with synthetic expansion to produce reliable and diverse Indic post-training data. Using this pipeline, we curate two datasets: Pragyaan-IT (22.5K) and Pragyaan-Align (100K) across 10 Indian languages covering 13 broad and 56 sub-categories, leveraging 57 diverse datasets. Our dataset protocol incorporates several often-overlooked dimensions and emphasize task diversity, multi-turn dialogue, instruction fidelity, safety alignment, and preservation of cultural nuance, providing a foundation for more inclusive and effective multilingual LLMs. 

---
# The Limits of Goal-Setting Theory in LLM-Driven Assessment 

**Authors**: Mrityunjay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.06997)  

**Abstract**: Many users interact with AI tools like ChatGPT using a mental model that treats the system as human-like, which we call Model H. According to goal-setting theory, increased specificity in goals should reduce performance variance. If Model H holds, then prompting a chatbot with more detailed instructions should lead to more consistent evaluation behavior.
This paper tests that assumption through a controlled experiment in which ChatGPT evaluated 29 student submissions using four prompts with increasing specificity. We measured consistency using intra-rater reliability (Cohen's Kappa) across repeated runs.
Contrary to expectations, performance did not improve consistently with increased prompt specificity, and performance variance remained largely unchanged. These findings challenge the assumption that LLMs behave like human evaluators and highlight the need for greater robustness and improved input integration in future model development. 

---
# VelLMes: A high-interaction AI-based deception framework 

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.06975)  

**Abstract**: There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands. 

---
# Learning Global Representation from Queries for Vectorized HD Map Construction 

**Authors**: Shoumeng Qiu, Xinrun Li, Yang Long, Xiangyang Xue, Varun Ojha, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06969)  

**Abstract**: The online construction of vectorized high-definition (HD) maps is a cornerstone of modern autonomous driving systems. State-of-the-art approaches, particularly those based on the DETR framework, formulate this as an instance detection problem. However, their reliance on independent, learnable object queries results in a predominantly local query perspective, neglecting the inherent global representation within HD maps. In this work, we propose \textbf{MapGR} (\textbf{G}lobal \textbf{R}epresentation learning for HD \textbf{Map} construction), an architecture designed to learn and utilize a global representations from queries. Our method introduces two synergistic modules: a Global Representation Learning (GRL) module, which encourages the distribution of all queries to better align with the global map through a carefully designed holistic segmentation task, and a Global Representation Guidance (GRG) module, which endows each individual query with explicit, global-level contextual information to facilitate its optimization. Evaluations on the nuScenes and Argoverse2 datasets validate the efficacy of our approach, demonstrating substantial improvements in mean Average Precision (mAP) compared to leading baselines. 

---
# Generating Surface for Text-to-3D using 2D Gaussian Splatting 

**Authors**: Huanning Dong, Fan Li, Ping Kuang, Jianwen Min  

**Link**: [PDF](https://arxiv.org/pdf/2510.06967)  

**Abstract**: Recent advancements in Text-to-3D modeling have shown significant potential for the creation of 3D content. However, due to the complex geometric shapes of objects in the natural world, generating 3D content remains a challenging task. Current methods either leverage 2D diffusion priors to recover 3D geometry, or train the model directly based on specific 3D representations. In this paper, we propose a novel method named DirectGaussian, which focuses on generating the surfaces of 3D objects represented by surfels. In DirectGaussian, we utilize conditional text generation models and the surface of a 3D object is rendered by 2D Gaussian splatting with multi-view normal and texture priors. For multi-view geometric consistency problems, DirectGaussian incorporates curvature constraints on the generated surface during optimization process. Through extensive experiments, we demonstrate that our framework is capable of achieving diverse and high-fidelity 3D content creation. 

---
# EDUMATH: Generating Standards-aligned Educational Math Word Problems 

**Authors**: Bryan R. Christ, Penelope Molitz, Jonathan Kropko, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06965)  

**Abstract**: Math word problems (MWPs) are critical K-12 educational tools, and customizing them to students' interests and ability levels can increase learning outcomes. However, teachers struggle to find time to customize MWPs for each student given large class sizes and increasing burnout. We propose that LLMs can support math education by generating MWPs customized to student interests and math education standards. To this end, we use a joint human expert-LLM judge approach to evaluate over 11,000 MWPs generated by open and closed LLMs and develop the first teacher-annotated dataset for standards-aligned educational MWP generation. We show the value of our data by using it to train a 12B open model that matches the performance of larger and more capable open models. We also use our teacher-annotated data to train a text classifier that enables a 30B open LLM to outperform existing closed baselines without any training. Next, we show our models' MWPs are more similar to human-written MWPs than those from existing models. We conclude by conducting the first study of customized LLM-generated MWPs with grade school students, finding they perform similarly on our models' MWPs relative to human-written MWPs but consistently prefer our customized MWPs. 

---
# Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation 

**Authors**: Vaibhav Srivastav, Steven Zheng, Eric Bezzam, Eustache Le Bihan, Nithin Koluguri, Piotr Żelasko, Somshubra Majumdar, Adel Moumen, Sanchit Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06961)  

**Abstract**: Despite rapid progress, ASR evaluation remains saturated with short-form English, and efficiency is rarely reported. We present the Open ASR Leaderboard, a fully reproducible benchmark and interactive leaderboard comparing 60+ open-source and proprietary systems across 11 datasets, including dedicated multilingual and long-form tracks. We standardize text normalization and report both word error rate (WER) and inverse real-time factor (RTFx), enabling fair accuracy-efficiency comparisons. For English transcription, Conformer encoders paired with LLM decoders achieve the best average WER but are slower, while CTC and TDT decoders deliver much better RTFx, making them attractive for long-form and offline use. Whisper-derived encoders fine-tuned for English improve accuracy but often trade off multilingual coverage. All code and dataset loaders are open-sourced to support transparent, extensible evaluation. 

---
# Grouped Differential Attention 

**Authors**: Junghwan Lim, Sungmin Lee, Dongseok Kim, Wai Ting Cheung, Beomgyu Kim, Taehwan Kim, Haesol Lee, Junhyeok Lee, Dongpin Oh, Eunhwan Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.06949)  

**Abstract**: The self-attention mechanism, while foundational to modern Transformer architectures, suffers from a critical inefficiency: it frequently allocates substantial attention to redundant or noisy context. Differential Attention addressed this by using subtractive attention maps for signal and noise, but its required balanced head allocation imposes rigid constraints on representational flexibility and scalability.
To overcome this, we propose Grouped Differential Attention (GDA), a novel approach that introduces unbalanced head allocation between signal-preserving and noise-control groups. GDA significantly enhances signal focus by strategically assigning more heads to signal extraction and fewer to noise-control, stabilizing the latter through controlled repetition (akin to GQA). This design achieves stronger signal fidelity with minimal computational overhead. We further extend this principle to group-differentiated growth, a scalable strategy that selectively replicates only the signal-focused heads, thereby ensuring efficient capacity expansion.
Through large-scale pretraining and continual training experiments, we demonstrate that moderate imbalance ratios in GDA yield substantial improvements in generalization and stability compared to symmetric baselines. Our results collectively establish that ratio-aware head allocation and selective expansion offer an effective and practical path toward designing scalable, computation-efficient Transformer architectures. 

---
# Expressive and Scalable Quantum Fusion for Multimodal Learning 

**Authors**: Tuyen Nguyen, Trong Nghia Hoang, Phi Le Nguyen, Hai L. Vu, Truong Cong Thang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06938)  

**Abstract**: The aim of this paper is to introduce a quantum fusion mechanism for multimodal learning and to establish its theoretical and empirical potential. The proposed method, called the Quantum Fusion Layer (QFL), replaces classical fusion schemes with a hybrid quantum-classical procedure that uses parameterized quantum circuits to learn entangled feature interactions without requiring exponential parameter growth. Supported by quantum signal processing principles, the quantum component efficiently represents high-order polynomial interactions across modalities with linear parameter scaling, and we provide a separation example between QFL and low-rank tensor-based methods that highlights potential quantum query advantages. In simulation, QFL consistently outperforms strong classical baselines on small but diverse multimodal tasks, with particularly marked improvements in high-modality regimes. These results suggest that QFL offers a fundamentally new and scalable approach to multimodal fusion that merits deeper exploration on larger systems. 

---
# Bayesian Nonparametric Dynamical Clustering of Time Series 

**Authors**: Adrián Pérez-Herrero, Paulo Félix, Jesús Presedo, Carl Henrik Ek  

**Link**: [PDF](https://arxiv.org/pdf/2510.06919)  

**Abstract**: We present a method that models the evolution of an unbounded number of time series clusters by switching among an unknown number of regimes with linear dynamics. We develop a Bayesian non-parametric approach using a hierarchical Dirichlet process as a prior on the parameters of a Switching Linear Dynamical System and a Gaussian process prior to model the statistical variations in amplitude and temporal alignment within each cluster. By modeling the evolution of time series patterns, the method avoids unnecessary proliferation of clusters in a principled manner. We perform inference by formulating a variational lower bound for off-line and on-line scenarios, enabling efficient learning through optimization. We illustrate the versatility and effectiveness of the approach through several case studies of electrocardiogram analysis using publicly available databases. 

---
# LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling 

**Authors**: Zecheng Tang, Baibei Ji, Quantong Qiu, Haitian Wang, Xiaobo Liang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06915)  

**Abstract**: Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model. 

---
# DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning 

**Authors**: Ke Guo, Haochen Liu, Xiaojun Wu, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2510.06913)  

**Abstract**: Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark. 

---
# Emotionally Vulnerable Subtype of Internet Gaming Disorder: Measuring and Exploring the Pathology of Problematic Generative AI Use 

**Authors**: Haocan Sun, Di Wua, Weizi Liu, Guoming Yua, Mike Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06908)  

**Abstract**: Concerns over the potential over-pathologization of generative AI (GenAI) use and the lack of conceptual clarity surrounding GenAI addiction call for empirical tools and theoretical refinement. This study developed and validated the PUGenAIS-9 (Problematic Use of Generative Artificial Intelligence Scale-9 items) and examined whether PUGenAIS reflects addiction-like patterns under the Internet Gaming Disorder (IGD) framework. Using samples from China and the United States (N = 1,508), we conducted confirmatory factor analysis and identified a robust 31-item structure across nine IGD-based dimensions. We then derived the PUGenAIS-9 by selecting the highest-loading items from each dimension and validated its structure in an independent sample (N = 1,426). Measurement invariance tests confirmed its stability across nationality and gender. Person-centered (latent profile analysis) and variable-centered (network analysis) approaches found that PUGenAIS matches the traits of the emotionally vulnerable subtype of IGD, not the competence-based kind. These results support using PUGenAIS-9 to identify problematic GenAI use and show the need to rethink digital addiction with an ICD (infrastructures, content, and device) model. This keeps addiction research responsive to new media while avoiding over-pathologizing. 

---
# Angular Constraint Embedding via SpherePair Loss for Constrained Clustering 

**Authors**: Shaojie Zhang, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06907)  

**Abstract**: Constrained clustering integrates domain knowledge through pairwise constraints. However, existing deep constrained clustering (DCC) methods are either limited by anchors inherent in end-to-end modeling or struggle with learning discriminative Euclidean embedding, restricting their scalability and real-world applicability. To avoid their respective pitfalls, we propose a novel angular constraint embedding approach for DCC, termed SpherePair. Using the SpherePair loss with a geometric formulation, our method faithfully encodes pairwise constraints and leads to embeddings that are clustering-friendly in angular space, effectively separating representation learning from clustering. SpherePair preserves pairwise relations without conflict, removes the need to specify the exact number of clusters, generalizes to unseen data, enables rapid inference of the number of clusters, and is supported by rigorous theoretical guarantees. Comparative evaluations with state-of-the-art DCC methods on diverse benchmarks, along with empirical validation of theoretical insights, confirm its superior performance, scalability, and overall real-world effectiveness. Code is available at \href{this https URL}{our repository}. 

---
# M3Retrieve: Benchmarking Multimodal Retrieval for Medicine 

**Authors**: Arkadeep Acharya, Akash Ghosh, Pradeepika Verma, Kitsuchart Pasupa, Sriparna Saha, Priti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06888)  

**Abstract**: With the increasing use of RetrievalAugmented Generation (RAG), strong retrieval models have become more important than ever. In healthcare, multimodal retrieval models that combine information from both text and images offer major advantages for many downstream tasks such as question answering, cross-modal retrieval, and multimodal summarization, since medical data often includes both formats. However, there is currently no standard benchmark to evaluate how well these models perform in medical settings. To address this gap, we introduce M3Retrieve, a Multimodal Medical Retrieval Benchmark. M3Retrieve, spans 5 domains,16 medical fields, and 4 distinct tasks, with over 1.2 Million text documents and 164K multimodal queries, all collected under approved licenses. We evaluate leading multimodal retrieval models on this benchmark to explore the challenges specific to different medical specialities and to understand their impact on retrieval performance. By releasing M3Retrieve, we aim to enable systematic evaluation, foster model innovation, and accelerate research toward building more capable and reliable multimodal retrieval systems for medical applications. The dataset and the baselines code are available in this github page this https URL. 

---
# Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices 

**Authors**: Boris Sedlak, Philipp Raith, Andrea Morichetta, Víctor Casamayor Pujol, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.06882)  

**Abstract**: Edge devices have limited resources, which inevitably leads to situations where stream processing services cannot satisfy their needs. While existing autoscaling mechanisms focus entirely on resource scaling, Edge devices require alternative ways to sustain the Service Level Objectives (SLOs) of competing services. To address these issues, we introduce a Multi-dimensional Autoscaling Platform (MUDAP) that supports fine-grained vertical scaling across both service- and resource-level dimensions. MUDAP supports service-specific scaling tailored to available parameters, e.g., scale data quality or model size for a particular service. To optimize the execution across services, we present a scaling agent based on Regression Analysis of Structural Knowledge (RASK). The RASK agent efficiently explores the solution space and learns a continuous regression model of the processing environment for inferring optimal scaling actions. We compared our approach with two autoscalers, the Kubernetes VPA and a reinforcement learning agent, for scaling up to 9 services on a single Edge device. Our results showed that RASK can infer an accurate regression model in merely 20 iterations (i.e., observe 200s of processing). By increasingly adding elasticity dimensions, RASK sustained the highest request load with 28% less SLO violations, compared to baselines. 

---
# MoRE-GNN: Multi-omics Data Integration with a Heterogeneous Graph Autoencoder 

**Authors**: Zhiyu Wang, Sonia Koszut, Pietro Liò, Francesco Ceccarelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.06880)  

**Abstract**: The integration of multi-omics single-cell data remains challenging due to high-dimensionality and complex inter-modality relationships. To address this, we introduce MoRE-GNN (Multi-omics Relational Edge Graph Neural Network), a heterogeneous graph autoencoder that combines graph convolution and attention mechanisms to dynamically construct relational graphs directly from data. Evaluations on six publicly available datasets demonstrate that MoRE-GNN captures biologically meaningful relationships and outperforms existing methods, particularly in settings with strong inter-modality correlations. Furthermore, the learned representations allow for accurate downstream cross-modal predictions. While performance may vary with dataset complexity, MoRE-GNN offers an adaptive, scalable and interpretable framework for advancing multi-omics integration. 

---
# Multi-hop Deep Joint Source-Channel Coding with Deep Hash Distillation for Semantically Aligned Image Retrieval 

**Authors**: Didrik Bergström, Deniz Gündüz, Onur Günlü  

**Link**: [PDF](https://arxiv.org/pdf/2510.06868)  

**Abstract**: We consider image transmission via deep joint source-channel coding (DeepJSCC) over multi-hop additive white Gaussian noise (AWGN) channels by training a DeepJSCC encoder-decoder pair with a pre-trained deep hash distillation (DHD) module to semantically cluster images, facilitating security-oriented applications through enhanced semantic consistency and improving the perceptual reconstruction quality. We train the DeepJSCC module to both reduce mean square error (MSE) and minimize cosine distance between DHD hashes of source and reconstructed images. Significantly improved perceptual quality as a result of semantic alignment is illustrated for different multi-hop settings, for which classical DeepJSCC may suffer from noise accumulation, measured by the learned perceptual image patch similarity (LPIPS) metric. 

---
# Towards Generalization of Graph Neural Networks for AC Optimal Power Flow 

**Authors**: Olayiwola Arowolo, Jochen L. Cremer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06860)  

**Abstract**: AC Optimal Power Flow (ACOPF) is computationally expensive for large-scale power systems, with conventional solvers requiring prohibitive solution times. Machine learning approaches offer computational speedups but struggle with scalability and topology adaptability without expensive retraining. To enable scalability across grid sizes and adaptability to topology changes, we propose a Hybrid Heterogeneous Message Passing Neural Network (HH-MPNN). HH-MPNN models buses, generators, loads, shunts, transmission lines and transformers as distinct node or edge types, combined with a scalable transformer model for handling long-range dependencies. On grids from 14 to 2,000 buses, HH-MPNN achieves less than 1% optimality gap on default topologies. Applied zero-shot to thousands of unseen topologies, HH-MPNN achieves less than 3% optimality gap despite training only on default topologies. Pre-training on smaller grids also improves results on a larger grid. Computational speedups reach 1,000x to 10,000x compared to interior point solvers. These results advance practical, generalizable machine learning for real-time power system operations. 

---
# Explaining raw data complexity to improve satellite onboard processing 

**Authors**: Adrien Dorise, Marjorie Bellizzi, Adrien Girard, Benjamin Francesconi, Stéphane May  

**Link**: [PDF](https://arxiv.org/pdf/2510.06858)  

**Abstract**: With increasing processing power, deploying AI models for remote sensing directly onboard satellites is becoming feasible. However, new constraints arise, mainly when using raw, unprocessed sensor data instead of preprocessed ground-based products. While current solutions primarily rely on preprocessed sensor images, few approaches directly leverage raw data. This study investigates the effects of utilising raw data on deep learning models for object detection and classification tasks. We introduce a simulation workflow to generate raw-like products from high-resolution L1 imagery, enabling systemic evaluation. Two object detection models (YOLOv11s and YOLOX-S) are trained on both raw and L1 datasets, and their performance is compared using standard detection metrics and explainability tools. Results indicate that while both models perform similarly at low to medium confidence thresholds, the model trained on raw data struggles with object boundary identification at high confidence levels. It suggests that adapting AI architectures with improved contouring methods can enhance object detection on raw images, improving onboard AI for remote sensing. 

---
# Enhancing Bankruptcy Prediction of Banks through Advanced Machine Learning Techniques: An Innovative Approach and Analysis 

**Authors**: Zuherman Rustam, Sri Hartini, Sardar M.N. Islam, Fevi Novkaniza, Fiftitah R. Aszhari, Muhammad Rifqi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06852)  

**Abstract**: Context: Financial system stability is determined by the condition of the banking system. A bank failure can destroy the stability of the financial system, as banks are subject to systemic risk, affecting not only individual banks but also segments or the entire financial system. Calculating the probability of a bank going bankrupt is one way to ensure the banking system is safe and sound. Existing literature and limitations: Statistical models, such as Altman's Z-Score, are one of the common techniques for developing a bankruptcy prediction model. However, statistical methods rely on rigid and sometimes irrelevant assumptions, which can result in low forecast accuracy. New approaches are necessary. Objective of the research: Bankruptcy models are developed using machine learning techniques, such as logistic regression (LR), random forest (RF), and support vector machines (SVM). According to several studies, machine learning is also more accurate and effective than statistical methods for categorising and forecasting banking risk management. Present Research: The commercial bank data are derived from the annual financial statements of 44 active banks and 21 bankrupt banks in Turkey from 1994 to 2004, and the rural bank data are derived from the quarterly financial reports of 43 active and 43 bankrupt rural banks in Indonesia between 2013 and 2019. Five rural banks in Indonesia have also been selected to demonstrate the feasibility of analysing bank bankruptcy trends. Findings and implications: The results of the research experiments show that RF can forecast data from commercial banks with a 90% accuracy rate. Furthermore, the three machine learning methods proposed accurately predict the likelihood of rural bank bankruptcy. Contribution and Conclusion: The proposed innovative machine learning approach help to implement policies that reduce the costs of bankruptcy. 

---
# OpenJAI-v1.0: An Open Thai Large Language Model 

**Authors**: Pontakorn Trakuekul, Attapol T. Rutherford, Jullajak Karnjanaekarin, Narongkorn Panitsrisit, Sumana Sumanakul  

**Link**: [PDF](https://arxiv.org/pdf/2510.06847)  

**Abstract**: We introduce OpenJAI-v1.0, an open-source large language model for Thai and English, developed from the Qwen3-14B model. Our work focuses on boosting performance on practical tasks through carefully curated data across three key use cases: instruction following, long-context understanding, and tool use. Evaluation results show that OpenJAI-v1.0 improves on the capabilities of its base model and outperforms other leading open-source Thai models on a diverse suite of benchmarks, while avoiding catastrophic forgetting. OpenJAI-v1.0 is publicly released as another alternative NLP resource for the Thai AI community. 

---
# SID: Multi-LLM Debate Driven by Self Signals 

**Authors**: Xuhang Chen, Zhifan Song, Deyi Ji, Shuo Gao, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06843)  

**Abstract**: Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{this https URL}{\texttt{this https URL}}. 

---
# CNN-TFT explained by SHAP with multi-head attention weights for time series forecasting 

**Authors**: Stefano F. Stefenon, João P. Matos-Carvalho, Valderi R. Q. Leithardt, Kin-Choong Yow  

**Link**: [PDF](https://arxiv.org/pdf/2510.06840)  

**Abstract**: Convolutional neural networks (CNNs) and transformer architectures offer strengths for modeling temporal data: CNNs excel at capturing local patterns and translational invariances, while transformers effectively model long-range dependencies via self-attention. This paper proposes a hybrid architecture integrating convolutional feature extraction with a temporal fusion transformer (TFT) backbone to enhance multivariate time series forecasting. The CNN module first applies a hierarchy of one-dimensional convolutional layers to distill salient local patterns from raw input sequences, reducing noise and dimensionality. The resulting feature maps are then fed into the TFT, which applies multi-head attention to capture both short- and long-term dependencies and to weigh relevant covariates adaptively. We evaluate the CNN-TFT on a hydroelectric natural flow time series dataset. Experimental results demonstrate that CNN-TFT outperforms well-established deep learning models, with a mean absolute percentage error of up to 2.2%. The explainability of the model is obtained by a proposed Shapley additive explanations with multi-head attention weights (SHAP-MHAW). Our novel architecture, named CNN-TFT-SHAP-MHAW, is promising for applications requiring high-fidelity, multivariate time series forecasts, being available for future analysis at this https URL . 

---
# Recurrence-Complete Frame-based Action Models 

**Authors**: Michael Keiblinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.06828)  

**Abstract**: In recent years, attention-like mechanisms have been used to great success in the space of large language models, unlocking scaling potential to a previously unthinkable extent. "Attention Is All You Need" famously claims RNN cells are not needed in conjunction with attention. We challenge this view. In this paper, we point to existing proofs that architectures with fully parallelizable forward or backward passes cannot represent classes of problems specifically interesting for long-running agentic tasks. We further conjecture a critical time t beyond which non-recurrence-complete models fail to aggregate inputs correctly, with concrete implications for agentic systems (e.g., software engineering agents). To address this, we introduce a recurrence-complete architecture and train it on GitHub-derived action sequences. Loss follows a power law in the trained sequence length while the parameter count remains fixed. Moreover, longer-sequence training always amortizes its linearly increasing wall-time cost, yielding lower loss as a function of wall time. 

---
# FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline 

**Authors**: Haotian Wu, Shufan Jiang, Chios Chen, Yiyang Feng, Hehai Lin, Heqing Zou, Yao Shu, Yanran Li, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06800)  

**Abstract**: As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench. 

---
# Extreme Amodal Face Detection 

**Authors**: Changlin Song, Yunzhong Hou, Michael Randall Barnes, Rahul Shome, Dylan Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2510.06791)  

**Abstract**: Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches. 

---
# Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness 

**Authors**: Luca Giordano, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.06780)  

**Abstract**: Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations. 

---
# Modeling COVID-19 Dynamics in German States Using Physics-Informed Neural Networks 

**Authors**: Phillip Rothenbeck, Sai Karthikeya Vemuri, Niklas Penzel, Joachim Denzler  

**Link**: [PDF](https://arxiv.org/pdf/2510.06776)  

**Abstract**: The COVID-19 pandemic has highlighted the need for quantitative modeling and analysis to understand real-world disease dynamics. In particular, post hoc analyses using compartmental models offer valuable insights into the effectiveness of public health interventions, such as vaccination strategies and containment policies. However, such compartmental models like SIR (Susceptible-Infectious-Recovered) often face limitations in directly incorporating noisy observational data. In this work, we employ Physics-Informed Neural Networks (PINNs) to solve the inverse problem of the SIR model using infection data from the Robert Koch Institute (RKI). Our main contribution is a fine-grained, spatio-temporal analysis of COVID-19 dynamics across all German federal states over a three-year period. We estimate state-specific transmission and recovery parameters and time-varying reproduction number (R_t) to track the pandemic progression. The results highlight strong variations in transmission behavior across regions, revealing correlations with vaccination uptake and temporal patterns associated with major pandemic phases. Our findings demonstrate the utility of PINNs in localized, long-term epidemiological modeling. 

---
# Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.06743)  

**Abstract**: Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization. 

---
# Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization 

**Authors**: Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06732)  

**Abstract**: Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: this https URL. 

---
# Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management 

**Authors**: Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06727)  

**Abstract**: We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit. 

---
# LLM Company Policies and Policy Implications in Software Organizations 

**Authors**: Ranim Khojah, Mazen Mohamad, Linda Erlenhov, Francisco Gomes de Oliveira Neto, Philipp Leitner  

**Link**: [PDF](https://arxiv.org/pdf/2510.06718)  

**Abstract**: The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows. 

---
# Dual Goal Representations 

**Authors**: Seohong Park, Deepinder Mann, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.06714)  

**Abstract**: In this work, we introduce dual goal representations for goal-conditioned reinforcement learning (GCRL). A dual goal representation characterizes a state by "the set of temporal distances from all other states"; in other words, it encodes a state through its relations to every other state, measured by temporal distance. This representation provides several appealing theoretical properties. First, it depends only on the intrinsic dynamics of the environment and is invariant to the original state representation. Second, it contains provably sufficient information to recover an optimal goal-reaching policy, while being able to filter out exogenous noise. Based on this concept, we develop a practical goal representation learning method that can be combined with any existing GCRL algorithm. Through diverse experiments on the OGBench task suite, we empirically show that dual goal representations consistently improve offline goal-reaching performance across 20 state- and pixel-based tasks. 

---
# AISysRev - LLM-based Tool for Title-abstract Screening 

**Authors**: Aleksi Huotala, Miikka Kuutila, Olli-Pekka Turtio, Mika Mäntylä  

**Link**: [PDF](https://arxiv.org/pdf/2510.06708)  

**Abstract**: Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human this http URL conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: this https URL Tool: this https URL 

---
# Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks 

**Authors**: Qinhao Zhou, Xiang Xiang, Kun He, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2510.06695)  

**Abstract**: In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks. 

---
# Semantic Segmentation Algorithm Based on Light Field and LiDAR Fusion 

**Authors**: Jie Luo, Yuxuan Jiang, Xin Jin, Mingyu Liu, Yihui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.06687)  

**Abstract**: Semantic segmentation serves as a cornerstone of scene understanding in autonomous driving but continues to face significant challenges under complex conditions such as occlusion. Light field and LiDAR modalities provide complementary visual and spatial cues that are beneficial for robust perception; how- ever, their effective integration is hindered by limited viewpoint diversity and inherent modality discrepancies. To address these challenges, the first multimodal semantic segmentation dataset integrating light field data and point cloud data is proposed. Based on this dataset, we proposed a multi-modal light field point-cloud fusion segmentation network(Mlpfseg), incorporating feature completion and depth perception to segment both camera images and LiDAR point clouds simultaneously. The feature completion module addresses the density mismatch between point clouds and image pixels by performing differential re- construction of point-cloud feature maps, enhancing the fusion of these modalities. The depth perception module improves the segmentation of occluded objects by reinforcing attention scores for better occlusion awareness. Our method outperforms image- only segmentation by 1.71 Mean Intersection over Union(mIoU) and point cloud-only segmentation by 2.38 mIoU, demonstrating its effectiveness. 

---
# Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback 

**Authors**: Yisha Wu, Zhao, Yuanpei Cao, Xiaoqing Su, Yashar Mehdad, Mindy Ji, Claire Na Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.06677)  

**Abstract**: We introduce an incremental summarization system for customer support agents that intelligently determines when to generate concise bullet notes during conversations, reducing agents' context-switching effort and redundant review. Our approach combines a fine-tuned Mixtral-8x7B model for continuous note generation with a DeBERTa-based classifier to filter trivial content. Agent edits refine the online notes generation and regularly inform offline model retraining, closing the agent edits feedback loop. Deployed in production, our system achieved a 3% reduction in case handling time compared to bulk summarization (with reductions of up to 9% in highly complex cases), alongside high agent satisfaction ratings from surveys. These results demonstrate that incremental summarization with continuous feedback effectively enhances summary quality and agent productivity at scale. 

---
# Heptapod: Language Modeling on Visual Signals 

**Authors**: Yongxin Zhu, Jiawei Chen, Yuanzhe Chen, Zhuo Chen, Dongya Jia, Jian Cong, Xiaobin Zhuang, Yuping Wang, Yuxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06673)  

**Abstract**: We introduce Heptapod, an image autoregressive model that adheres to the foundational principles of language modeling. Heptapod employs \textbf{causal attention}, \textbf{eliminates reliance on CFG}, and \textbf{eschews the trend of semantic tokenizers}. Our key innovation is \textit{next 2D distribution prediction}: a causal Transformer with reconstruction-focused visual tokenizer, learns to predict the distribution over the entire 2D spatial grid of images at each timestep. This learning objective unifies the sequential modeling of autoregressive framework with the holistic self-supervised learning of masked autoencoding, enabling the model to capture comprehensive image semantics via generative training. On the ImageNet generation benchmark, Heptapod achieves an FID of $2.70$, significantly outperforming previous causal autoregressive approaches. We hope our work inspires a principled rethinking of language modeling on visual signals and beyond. 

---
# Automated Neural Architecture Design for Industrial Defect Detection 

**Authors**: Yuxi Liu, Yunfeng Ma, Yi Tang, Min Liu, Shuai Jiang, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06669)  

**Abstract**: Industrial surface defect detection (SDD) is critical for ensuring product quality and manufacturing reliability. Due to the diverse shapes and sizes of surface defects, SDD faces two main challenges: intraclass difference and interclass similarity. Existing methods primarily utilize manually designed models, which require extensive trial and error and often struggle to address both challenges effectively. To overcome this, we propose AutoNAD, an automated neural architecture design framework for SDD that jointly searches over convolutions, transformers, and multi-layer perceptrons. This hybrid design enables the model to capture both fine-grained local variations and long-range semantic context, addressing the two key challenges while reducing the cost of manual network design. To support efficient training of such a diverse search space, AutoNAD introduces a cross weight sharing strategy, which accelerates supernet convergence and improves subnet performance. Additionally, a searchable multi-level feature aggregation module (MFAM) is integrated to enhance multi-scale feature learning. Beyond detection accuracy, runtime efficiency is essential for industrial deployment. To this end, AutoNAD incorporates a latency-aware prior to guide the selection of efficient architectures. The effectiveness of AutoNAD is validated on three industrial defect datasets and further applied within a defect imaging and detection platform. Code will be available at this https URL. 

---
# Delay Independent Safe Control with Neural Networks: Positive Lur'e Certificates for Risk Aware Autonomy 

**Authors**: Hamidreza Montazeri Hedesh, Milad Siami  

**Link**: [PDF](https://arxiv.org/pdf/2510.06661)  

**Abstract**: We present a risk-aware safety certification method for autonomous, learning enabled control systems. Focusing on two realistic risks, state/input delays and interval matrix uncertainty, we model the neural network (NN) controller with local sector bounds and exploit positivity structure to derive linear, delay-independent certificates that guarantee local exponential stability across admissible uncertainties. To benchmark performance, we adopt and implement a state-of-the-art IQC NN verification pipeline. On representative cases, our positivity-based tests run orders of magnitude faster than SDP-based IQC while certifying regimes the latter cannot-providing scalable safety guarantees that complement risk-aware control. 

---
# Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions 

**Authors**: Frank Wu, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.06649)  

**Abstract**: The Forward-Forward (FF) Algorithm is a recently proposed learning procedure for neural networks that employs two forward passes instead of the traditional forward and backward passes used in backpropagation. However, FF remains largely confined to supervised settings, leaving a gap at domains where learning signals can be yielded more naturally such as RL. In this work, inspired by FF's goodness function using layer activity statistics, we introduce Action-conditioned Root mean squared Q-Functions (ARQ), a novel value estimation method that applies a goodness function and action conditioning for local RL using temporal difference learning. Despite its simplicity and biological grounding, our approach achieves superior performance compared to state-of-the-art local backprop-free RL methods in the MinAtar and the DeepMind Control Suite benchmarks, while also outperforming algorithms trained with backpropagation on most tasks. Code can be found at this https URL. 

---
# The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators 

**Authors**: Mansi Sakarvadia, Kareem Hegazy, Amin Totounferoush, Kyle Chard, Yaoqing Yang, Ian Foster, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2510.06646)  

**Abstract**: A core challenge in scientific machine learning, and scientific computing more generally, is modeling continuous phenomena which (in practice) are represented discretely. Machine-learned operators (MLOs) have been introduced as a means to achieve this modeling goal, as this class of architecture can perform inference at arbitrary resolution. In this work, we evaluate whether this architectural innovation is sufficient to perform "zero-shot super-resolution," namely to enable a model to serve inference on higher-resolution data than that on which it was originally trained. We comprehensively evaluate both zero-shot sub-resolution and super-resolution (i.e., multi-resolution) inference in MLOs. We decouple multi-resolution inference into two key behaviors: 1) extrapolation to varying frequency information; and 2) interpolating across varying resolutions. We empirically demonstrate that MLOs fail to do both of these tasks in a zero-shot manner. Consequently, we find MLOs are not able to perform accurate inference at resolutions different from those on which they were trained, and instead they are brittle and susceptible to aliasing. To address these failure modes, we propose a simple, computationally-efficient, and data-driven multi-resolution training protocol that overcomes aliasing and that provides robust multi-resolution generalization. 

---
# Distilling Lightweight Language Models for C/C++ Vulnerabilities 

**Authors**: Zhiyuan Wei, Xiaoxuan Yang, Jing Sun, Zijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06645)  

**Abstract**: The increasing complexity of modern software systems exacerbates the prevalence of security vulnerabilities, posing risks of severe breaches and substantial economic loss. Consequently, robust code vulnerability detection is essential for software security. While Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing, their potential for automated code vulnerability detection remains underexplored. This paper presents FineSec, a novel framework that harnesses LLMs through knowledge distillation to enable efficient and precise vulnerability identification in C/C++ codebases. FineSec utilizes knowledge distillation to transfer expertise from large teacher models to compact student models, achieving high accuracy with minimal computational cost. By integrating data preparation, training, evaluation, and continuous learning into a unified, single-task workflow, FineSec offers a streamlined approach. Extensive evaluations on C/C++ codebases demonstrate its superiority over both base models and larger LLMs in identifying complex vulnerabilities and logical flaws, establishing FineSec as a practical and scalable solution for real-world software security. To facilitate reproducibility, the datasets, source code, and experimental results are made publicly available at: this https URL. 

---
# StaR-KVQA: Structured Reasoning Traces for Implicit-Knowledge Visual Question Answering 

**Authors**: Zhihao Wen, Wenkang Wei, Yuan Fang, Xingtong Yu, Hui Zhang, Weicheng Zhu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06638)  

**Abstract**: Knowledge-based Visual Question Answering (KVQA) requires models to ground entities in images and reason over factual knowledge. We study its implicit-knowledge variant, IK-KVQA, where a multimodal large language model (MLLM) is the sole knowledge source, without external retrieval. Yet, MLLMs lack explicit reasoning supervision and produce inconsistent justifications, and generalize poorly after standard supervised fine-tuning (SFT). We present StaR-KVQA (Structured Reasoning Traces for IK-KVQA), which supervises structured traces - dual symbolic relation paths plus path-grounded natural-language explanations - so that reasoning becomes transparent and verifiable. With one open-source MLLM, StaR-KVQA constructs and selects path-grounded reasoning traces to form a trace-enriched dataset, then fine-tunes via structured self-distillation to align generation with supervision; no external retrievers, verifiers, or curated knowledge bases (KBs) are used, traces are built offline, and inference is a single autoregressive pass. Across benchmarks, StaR-KVQA improves both accuracy and interpretability, achieving up to +11.3% higher answer accuracy on OK-VQA over the strongest baseline while exhibiting robust cross-domain generalization. 

---
# Control-Augmented Autoregressive Diffusion for Data Assimilation 

**Authors**: Prakhar Srivastava, Farrin Marouf Sofian, Francesco Immorlano, Kushagra Pandey, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2510.06637)  

**Abstract**: Despite recent advances in test-time scaling and finetuning of diffusion models, guidance in Auto-Regressive Diffusion Models (ARDMs) remains underexplored. We introduce an amortized framework that augments pretrained ARDMs with a lightweight controller network, trained offline by previewing future ARDM rollouts and learning stepwise controls that anticipate upcoming observations under a terminal cost objective. We evaluate this framework in the context of data assimilation (DA) for chaotic spatiotemporal partial differential equations (PDEs), a setting where existing methods are often computationally prohibitive and prone to forecast drift under sparse observations. Our approach reduces DA inference to a single forward rollout with on-the-fly corrections, avoiding expensive adjoint computations and/or optimizations during inference. We demonstrate that our method consistently outperforms four state-of-the-art baselines in stability, accuracy, and physical fidelity across two canonical PDEs and six observation regimes. We will release code and checkpoints publicly. 

---
# AI-Driven Forecasting and Monitoring of Urban Water System 

**Authors**: Qiming Guo, Bishal Khatri, Hua Zhang, Wenlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06631)  

**Abstract**: Underground water and wastewater pipelines are vital for city operations but plagued by anomalies like leaks and infiltrations, causing substantial water loss, environmental damage, and high repair costs. Conventional manual inspections lack efficiency, while dense sensor deployments are prohibitively expensive. In recent years, artificial intelligence has advanced rapidly and is increasingly applied to urban infrastructure. In this research, we propose an integrated AI and remote-sensor framework to address the challenge of leak detection in underground water pipelines, through deploying a sparse set of remote sensors to capture real-time flow and depth data, paired with HydroNet - a dedicated model utilizing pipeline attributes (e.g., material, diameter, slope) in a directed graph for higher-precision modeling. Evaluations on a real-world campus wastewater network dataset demonstrate that our system collects effective spatio-temporal hydraulic data, enabling HydroNet to outperform advanced baselines. This integration of edge-aware message passing with hydraulic simulations enables accurate network-wide predictions from limited sensor deployments. We envision that this approach can be effectively extended to a wide range of underground water pipeline networks. 

---
# Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation 

**Authors**: Shuo Shao, Yiming Li, Hongwei Yao, Yifei Chen, Yuchen Yang, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06605)  

**Abstract**: The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods. 

---
# SDQM: Synthetic Data Quality Metric for Object Detection Dataset Evaluation 

**Authors**: Ayush Zenith, Arnold Zumbrun, Neel Raut, Jing Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06596)  

**Abstract**: The performance of machine learning models depends heavily on training data. The scarcity of large-scale, well-annotated datasets poses significant challenges in creating robust models. To address this, synthetic data generated through simulations and generative models has emerged as a promising solution, enhancing dataset diversity and improving the performance, reliability, and resilience of models. However, evaluating the quality of this generated data requires an effective metric. This paper introduces the Synthetic Dataset Quality Metric (SDQM) to assess data quality for object detection tasks without requiring model training to converge. This metric enables more efficient generation and selection of synthetic datasets, addressing a key challenge in resource-constrained object detection tasks. In our experiments, SDQM demonstrated a strong correlation with the mean Average Precision (mAP) scores of YOLOv11, a leading object detection model, while previous metrics only exhibited moderate or weak correlations. Additionally, it provides actionable insights for improving dataset quality, minimizing the need for costly iterative training. This scalable and efficient metric sets a new standard for evaluating synthetic data. The code for SDQM is available at this https URL 

---
# The Framework That Survives Bad Models: Human-AI Collaboration For Clinical Trials 

**Authors**: Yao Chen, David Ohlssen, Aimee Readie, Gregory Ligozio, Ruvie Martin, Thibaud Coroller  

**Link**: [PDF](https://arxiv.org/pdf/2510.06567)  

**Abstract**: Artificial intelligence (AI) holds great promise for supporting clinical trials, from patient recruitment and endpoint assessment to treatment response prediction. However, deploying AI without safeguards poses significant risks, particularly when evaluating patient endpoints that directly impact trial conclusions. We compared two AI frameworks against human-only assessment for medical image-based disease evaluation, measuring cost, accuracy, robustness, and generalization ability. To stress-test these frameworks, we injected bad models, ranging from random guesses to naive predictions, to ensure that observed treatment effects remain valid even under severe model degradation. We evaluated the frameworks using two randomized controlled trials with endpoints derived from spinal X-ray images. Our findings indicate that using AI as a supporting reader (AI-SR) is the most suitable approach for clinical trials, as it meets all criteria across various model types, even with bad models. This method consistently provides reliable disease estimation, preserves clinical trial treatment effect estimates and conclusions, and retains these advantages when applied to different populations. 

---
# HSNet: Heterogeneous Subgraph Network for Single Image Super-resolution 

**Authors**: Qiongyang Hu, Wenyang Liu, Wenbin Zou, Yuejiao Su, Lap-Pui Chau, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06564)  

**Abstract**: Existing deep learning approaches for image super-resolution, particularly those based on CNNs and attention mechanisms, often suffer from structural inflexibility. Although graph-based methods offer greater representational adaptability, they are frequently impeded by excessive computational complexity. To overcome these limitations, this paper proposes the Heterogeneous Subgraph Network (HSNet), a novel framework that efficiently leverages graph modeling while maintaining computational feasibility. The core idea of HSNet is to decompose the global graph into manageable sub-components. First, we introduce the Constructive Subgraph Set Block (CSSB), which generates a diverse set of complementary subgraphs. Rather than relying on a single monolithic graph, CSSB captures heterogeneous characteristics of the image by modeling different relational patterns and feature interactions, producing a rich ensemble of both local and global graph structures. Subsequently, the Subgraph Aggregation Block (SAB) integrates the representations embedded across these subgraphs. Through adaptive weighting and fusion of multi-graph features, SAB constructs a comprehensive and discriminative representation that captures intricate interdependencies. Furthermore, a Node Sampling Strategy (NSS) is designed to selectively retain the most salient features, thereby enhancing accuracy while reducing computational overhead. Extensive experiments demonstrate that HSNet achieves state-of-the-art performance, effectively balancing reconstruction quality with computational efficiency. The code will be made publicly available. 

---
# The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law 

**Authors**: Cheonkam Jeong, Sungdo Kim, Jewoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.06559)  

**Abstract**: Contemporary language models are fluent yet routinely mis-handle the types of meaning their outputs entail. We argue that hallucination, brittle moderation, and opaque compliance outcomes are symptoms of missing type-theoretic semantics rather than data or scale limitations. Building on Montague's view of language as typed, compositional algebra, we recast alignment as a parsing problem: natural-language inputs must be compiled into structures that make explicit their descriptive, normative, and legal dimensions under context.
We present Savassan, a neuro-symbolic architecture that compiles utterances into Montague-style logical forms and maps them to typed ontologies extended with deontic operators and jurisdictional contexts. Neural components extract candidate structures from unstructured inputs; symbolic components perform type checking, constraint reasoning, and cross-jurisdiction mapping to produce compliance-aware guidance rather than binary censorship. In cross-border scenarios, the system "parses once" (e.g., defect claim(product x, company y)) and projects the result into multiple legal ontologies (e.g., defamation risk in KR/JP, protected opinion in US, GDPR checks in EU), composing outcomes into a single, explainable decision.
This paper contributes: (i) a diagnosis of hallucination as a type error; (ii) a formal Montague-ontology bridge for business/legal reasoning; and (iii) a production-oriented design that embeds typed interfaces across the pipeline. We outline an evaluation plan using legal reasoning benchmarks and synthetic multi-jurisdiction suites. Our position is that trustworthy autonomy requires compositional typing of meaning, enabling systems to reason about what is described, what is prescribed, and what incurs liability within a unified algebra of meaning. 

---
# The Markovian Thinker 

**Authors**: Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.06557)  

**Abstract**: Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs. 

---
# Incoherence in goal-conditioned autoregressive models 

**Authors**: Jacek Karwowski, Raymond Douglas  

**Link**: [PDF](https://arxiv.org/pdf/2510.06545)  

**Abstract**: We investigate mathematically the notion of incoherence: a structural issue with reinforcement learning policies derived by naive goal-conditioning of autoregressive models. We focus on the process of re-training models on their own actions, that is, fine-tuning offline-learned policies with online RL. We prove that it decreases incoherence and leads to an improvement in return, and we aim to characterize the resulting trajectory of policies. By re-framing standard notions of control-as-inference and soft Q learning, we establish a three-way correspondence with two other ways of understanding the iterative re-training process: as folding the posterior into the reward and, in the deterministic case, as decreasing the temperature parameter; the correspondence has computational content via the training-inference trade-off. Through soft-conditioning generative models, we discuss the link between incoherence and the effective horizon. 

---
# Scalable Policy-Based RL Algorithms for POMDPs 

**Authors**: Ameya Anjarlekar, Rasoul Etesami, R Srikant  

**Link**: [PDF](https://arxiv.org/pdf/2510.06540)  

**Abstract**: The continuous nature of belief states in POMDPs presents significant computational challenges in learning the optimal policy. In this paper, we consider an approach that solves a Partially Observable Reinforcement Learning (PORL) problem by approximating the corresponding POMDP model into a finite-state Markov Decision Process (MDP) (called Superstate MDP). We first derive theoretical guarantees that improve upon prior work that relate the optimal value function of the transformed Superstate MDP to the optimal value function of the original POMDP. Next, we propose a policy-based learning approach with linear function approximation to learn the optimal policy for the Superstate MDP. Consequently, our approach shows that a POMDP can be approximately solved using TD-learning followed by Policy Optimization by treating it as an MDP, where the MDP state corresponds to a finite history. We show that the approximation error decreases exponentially with the length of this history. To the best of our knowledge, our finite-time bounds are the first to explicitly quantify the error introduced when applying standard TD learning to a setting where the true dynamics are not Markovian. 

---
# CLAQS: Compact Learnable All-Quantum Token Mixer with Shared-ansatz for Text Classification 

**Authors**: Junhao Chen, Yifan Zhou, Hanqi Jiang, Yi Pan, Yiwei Li, Huaqin Zhao, Wei Zhang, Yingfeng Wang, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06532)  

**Abstract**: Quantum compute is scaling fast, from cloud QPUs to high throughput GPU simulators, making it timely to prototype quantum NLP beyond toy tasks. However, devices remain qubit limited and depth limited, training can be unstable, and classical attention is compute and memory heavy. This motivates compact, phase aware quantum token mixers that stabilize amplitudes and scale to long sequences. We present CLAQS, a compact, fully quantum token mixer for text classification that jointly learns complex-valued mixing and nonlinear transformations within a unified quantum circuit. To enable stable end-to-end optimization, we apply l1 normalization to regulate amplitude scaling and introduce a two-stage parameterized quantum architecture that decouples shared token embeddings from a window-level quantum feed-forward module. Operating under a sliding-window regime with document-level aggregation, CLAQS requires only eight data qubits and shallow circuits, yet achieves 91.64% accuracy on SST-2 and 87.08% on IMDB, outperforming both classical Transformer baselines and strong hybrid quantum-classical counterparts. 

---
# Visualizing Multimodality in Combinatorial Search Landscapes 

**Authors**: Xavier F. C. Sánchez-Díaz, Ole Jakob Mengshoel  

**Link**: [PDF](https://arxiv.org/pdf/2510.06517)  

**Abstract**: This work walks through different visualization techniques for combinatorial search landscapes, focusing on multimodality. We discuss different techniques from the landscape analysis literature, and how they can be combined to provide a more comprehensive view of the search landscape. We also include examples and discuss relevant work to show how others have used these techniques in practice, based on the geometric and aesthetic elements of the Grammar of Graphics. We conclude that there is no free lunch in visualization, and provide recommendations for future work as there are several paths to continue the work in this field. 

---
# LogSTOP: Temporal Scores over Prediction Sequences for Matching and Retrieval 

**Authors**: Avishree Khare, Hideki Okamoto, Bardh Hoxha, Georgios Fainekos, Rajeev Alur  

**Link**: [PDF](https://arxiv.org/pdf/2510.06512)  

**Abstract**: Neural models such as YOLO and HuBERT can be used to detect local properties such as objects ("car") and emotions ("angry") in individual frames of videos and audio clips respectively. The likelihood of these detections is indicated by scores in [0, 1]. Lifting these scores to temporal properties over sequences can be useful for several downstream applications such as query matching (e.g., "does the speaker eventually sound happy in this audio clip?"), and ranked retrieval (e.g., "retrieve top 5 videos with a 10 second scene where a car is detected until a pedestrian is detected"). In this work, we formalize this problem of assigning Scores for TempOral Properties (STOPs) over sequences, given potentially noisy score predictors for local properties. We then propose a scoring function called LogSTOP that can efficiently compute these scores for temporal properties represented in Linear Temporal Logic. Empirically, LogSTOP, with YOLO and HuBERT, outperforms Large Vision / Audio Language Models and other Temporal Logic-based baselines by at least 16% on query matching with temporal properties over objects-in-videos and emotions-in-speech respectively. Similarly, on ranked retrieval with temporal properties over objects and actions in videos, LogSTOP with Grounding DINO and SlowR50 reports at least a 19% and 16% increase in mean average precision and recall over zero-shot text-to-video retrieval baselines respectively. 

---
# A Median Perspective on Unlabeled Data for Out-of-Distribution Detection 

**Authors**: Momin Abbas, Ali Falahati, Hossein Goli, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.06505)  

**Abstract**: Out-of-distribution (OOD) detection plays a crucial role in ensuring the robustness and reliability of machine learning systems deployed in real-world applications. Recent approaches have explored the use of unlabeled data, showing potential for enhancing OOD detection capabilities. However, effectively utilizing unlabeled in-the-wild data remains challenging due to the mixed nature of both in-distribution (InD) and OOD samples. The lack of a distinct set of OOD samples complicates the task of training an optimal OOD classifier. In this work, we introduce Medix, a novel framework designed to identify potential outliers from unlabeled data using the median operation. We use the median because it provides a stable estimate of the central tendency, as an OOD detection mechanism, due to its robustness against noise and outliers. Using these identified outliers, along with labeled InD data, we train a robust OOD classifier. From a theoretical perspective, we derive error bounds that demonstrate Medix achieves a low error rate. Empirical results further substantiate our claims, as Medix outperforms existing methods across the board in open-world settings, confirming the validity of our theoretical insights. 

---
# ATLO-ML: Adaptive Time-Length Optimizer for Machine Learning -- Insights from Air Quality Forecasting 

**Authors**: I-Hsi Kao, Kanji Uchino  

**Link**: [PDF](https://arxiv.org/pdf/2510.06503)  

**Abstract**: Accurate time-series predictions in machine learning are heavily influenced by the selection of appropriate input time length and sampling rate. This paper introduces ATLO-ML, an adaptive time-length optimization system that automatically determines the optimal input time length and sampling rate based on user-defined output time length. The system provides a flexible approach to time-series data pre-processing, dynamically adjusting these parameters to enhance predictive performance. ATLO-ML is validated using air quality datasets, including both GAMS-dataset and proprietary data collected from a data center, both in time series format. Results demonstrate that utilizing the optimized time length and sampling rate significantly improves the accuracy of machine learning models compared to fixed time lengths. ATLO-ML shows potential for generalization across various time-sensitive applications, offering a robust solution for optimizing temporal input parameters in machine learning workflows. 

---
# Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels 

**Authors**: Zhepeng Cen, Haolin Chen, Shiyu Wang, Zuxin Liu, Zhiwei Liu, Ding Zhao, Silvio Savarese, Caiming Xiong, Huan Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06499)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success through imitation learning on vast text corpora, but this paradigm creates a training-generation gap and limits robust reasoning. Reinforcement learning (RL) offers a more data-efficient solution capable of bridging this gap, yet its application has been constrained by a critical data bottleneck: existing RL datasets are orders of magnitude smaller and less diverse than web-scale pre-training corpora. To address this, we introduce the Webscale-RL pipeline, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, we construct the Webscale-RL dataset, containing 1.2 million examples across more than 9 domains. Our experiments show that the model trained on this dataset significantly outperforms continual pretraining and strong data refinement baselines across a suite of benchmarks. Notably, RL training with our dataset proves substantially more efficient, achieving the performance of continual pre-training with up to 100$\times$ fewer tokens. Our work presents a viable path toward scaling RL to pre-training levels, enabling more capable and efficient language models. 

---
# Valid Stopping for LLM Generation via Empirical Dynamic Formal Lift 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.06478)  

**Abstract**: We introduce Sequential-EDFL (Empirical Dynamic Formal Lift), applying anytime-valid sequential testing to language model generation stopping. Our approach tracks information lift -- the log-likelihood ratio between full models and deliberately weakened "skeleton" baselines -- using self-normalized empirical-Bernstein e-processes that provide formal delta-level error control regardless of stopping time. We handle unknown centering through online mean estimation, combine multiple parameters via mixture e-processes, and support adaptive resets under distributional drift. On six benchmarks, Sequential-EDFL reduces generation by 22-28% vs. sequential baselines while maintaining delta-level control with 12% computational overhead. We introduce automated skeletons (distilled submodels, randomized logits) and show robustness across skeleton families. Composing EDFL with a lightweight correctness gate (sentence boundaries + verifier) improves end-task correctness while preserving anytime-valid guarantees by only delaying stopping. Our certificates control information sufficiency, not factual correctness -- 10.9% of stopped sequences remain incorrect even with the gate (13.2-22.7% without it). EDFL serves as a first-stage filter reducing verification burden by 83%, not as a standalone solution for safety-critical domains. 

---
# Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin 

**Authors**: Enrique Queipo-de-Llano, Álvaro Arroyo, Federico Barbero, Xiaowen Dong, Michael Bronstein, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2510.06477)  

**Abstract**: Attention sinks and compression valleys have attracted significant attention as two puzzling phenomena in large language models, but have been studied in isolation. In this work, we present a surprising connection between attention sinks and compression valleys, tracing both to the formation of massive activations in the residual stream. We prove theoretically that massive activations necessarily produce representational compression and establish bounds on the resulting entropy reduction. Through experiments across several models (410M-120B parameters), we confirm that when the beginning-of-sequence token develops extreme activation norms in the middle layers, both compression valleys and attention sinks emerge simultaneously. Targeted ablation studies validate our theoretical predictions. This unified view motivates us to propose the Mix-Compress-Refine theory of information flow, as an attempt to explain how LLMs organize their computation in depth by controlling attention and representational compression via massive activations. Specifically, we posit that Transformer-based LLMs process tokens in three distinct phases: (1) broad mixing in the early layers, (2) compressed computation with limited mixing in the middle layers, and (3) selective refinement in the late layers. Our framework helps explain why embedding tasks perform best at intermediate layers, whereas generation tasks benefit from full-depth processing, clarifying differences in task-dependent representations. 

---
# Deep Generative Model for Human Mobility Behavior 

**Authors**: Ye Hong, Yatao Zhang, Konrad Schindler, Martin Raubal  

**Link**: [PDF](https://arxiv.org/pdf/2510.06473)  

**Abstract**: Understanding and modeling human mobility is central to challenges in transport planning, sustainable urban design, and public health. Despite decades of effort, simulating individual mobility remains challenging because of its complex, context-dependent, and exploratory nature. Here, we present MobilityGen, a deep generative model that produces realistic mobility trajectories spanning days to weeks at large spatial scales. By linking behavioral attributes with environmental context, MobilityGen reproduces key patterns such as scaling laws for location visits, activity time allocation, and the coupled evolution of travel mode and destination choices. It reflects spatio-temporal variability and generates diverse, plausible, and novel mobility patterns consistent with the built environment. Beyond standard validation, MobilityGen yields insights not attainable with earlier models, including how access to urban space varies across travel modes and how co-presence dynamics shape social exposure and segregation. Our work establishes a new framework for mobility simulation, paving the way for fine-grained, data-driven studies of human behavior and its societal implications. 

---
# Evaluating Node-tree Interfaces for AI Explainability 

**Authors**: Lifei Wang, Natalie Friedman, Chengchao Zhu, Zeshu Zhu, S.Joy Mountford  

**Link**: [PDF](https://arxiv.org/pdf/2510.06457)  

**Abstract**: As large language models (LLMs) become ubiquitous in workplace tools and decision-making processes, ensuring explainability and fostering user trust are critical. Although advancements in LLM engineering continue, human-centered design is still catching up, particularly when it comes to embedding transparency and trust into AI interfaces. This study evaluates user experiences with two distinct AI interfaces - node-tree interfaces and chatbot interfaces - to assess their performance in exploratory, follow-up inquiry, decision-making, and problem-solving tasks. Our design-driven approach introduces a node-tree interface that visually structures AI-generated responses into hierarchically organized, interactive nodes, allowing users to navigate, refine, and follow up on complex information. In a comparative study with n=20 business users, we observed that while the chatbot interface effectively supports linear, step-by-step queries, it is the node-tree interface that enhances brainstorming. Quantitative and qualitative findings indicate that node-tree interfaces not only improve task performance and decision-making support but also promote higher levels of user trust by preserving context. Our findings suggest that adaptive AI interfaces capable of switching between structured visualizations and conversational formats based on task requirements can significantly enhance transparency and user confidence in AI-powered systems. This work contributes actionable insights to the fields of human-robot interaction and AI design, particularly for enterprise applications where trust-building is critical for teams. 

---
# How NOT to benchmark your SITE metric: Beyond Static Leaderboards and Towards Realistic Evaluation 

**Authors**: Prabhant Singh, Sibylle Hess, Joaquin Vanschoren  

**Link**: [PDF](https://arxiv.org/pdf/2510.06448)  

**Abstract**: Transferability estimation metrics are used to find a high-performing pre-trained model for a given target task without fine-tuning models and without access to the source dataset. Despite the growing interest in developing such metrics, the benchmarks used to measure their progress have gone largely unexamined. In this work, we empirically show the shortcomings of widely used benchmark setups to evaluate transferability estimation metrics. We argue that the benchmarks on which these metrics are evaluated are fundamentally flawed. We empirically demonstrate that their unrealistic model spaces and static performance hierarchies artificially inflate the perceived performance of existing metrics, to the point where simple, dataset-agnostic heuristics can outperform sophisticated methods. Our analysis reveals a critical disconnect between current evaluation protocols and the complexities of real-world model selection. To address this, we provide concrete recommendations for constructing more robust and realistic benchmarks to guide future research in a more meaningful direction. 

---
# A Survey on Agentic Security: Applications, Threats and Defenses 

**Authors**: Asif Shahriar, Md Nafiu Rahman, Sadif Ahmed, Farig Sadeque, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2510.06445)  

**Abstract**: The rapid shift from passive LLMs to autonomous LLM-agents marks a new paradigm in cybersecurity. While these agents can act as powerful tools for both offensive and defensive operations, the very agentic context introduces a new class of inherent security risks. In this work we present the first holistic survey of the agentic security landscape, structuring the field around three interdependent pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 150 papers, explaining how agents are used, the vulnerabilities they possess, and the countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage. 

---
# Context-Aware Inference via Performance Forecasting in Decentralized Learning Networks 

**Authors**: Joel Pfeffer, J. M. Diederik Kruijssen, Clément Gossart, Mélanie Chevance, Diego Campo Millan, Florian Stecker, Steven N. Longmore  

**Link**: [PDF](https://arxiv.org/pdf/2510.06444)  

**Abstract**: In decentralized learning networks, predictions from many participants are combined to generate a network inference. While many studies have demonstrated performance benefits of combining multiple model predictions, existing strategies using linear pooling methods (ranging from simple averaging to dynamic weight updates) face a key limitation. Dynamic prediction combinations that rely on historical performance to update weights are necessarily reactive. Due to the need to average over a reasonable number of epochs (with moving averages or exponential weighting), they tend to be slow to adjust to changing circumstances (phase or regime changes). In this work, we develop a model that uses machine learning to forecast the performance of predictions by models at each epoch in a time series. This enables `context-awareness' by assigning higher weight to models that are likely to be more accurate at a given time. We show that adding a performance forecasting worker in a decentralized learning network, following a design similar to the Allora network, can improve the accuracy of network inferences. Specifically, we find forecasting models that predict regret (performance relative to the network inference) or regret z-score (performance relative to other workers) show greater improvement than models predicting losses, which often do not outperform the naive network inference (historically weighted average of all inferences). Through a series of optimization tests, we show that the performance of the forecasting model can be sensitive to choices in the feature set and number of training epochs. These properties may depend on the exact problem and should be tailored to each domain. Although initially designed for a decentralized learning network, using performance forecasting for prediction combination may be useful in any situation where predictive rather than reactive model weighting is needed. 

---
# Geometry-Aware Backdoor Attacks: Leveraging Curvature in Hyperbolic Embeddings 

**Authors**: Ali Baheri  

**Link**: [PDF](https://arxiv.org/pdf/2510.06397)  

**Abstract**: Non-Euclidean foundation models increasingly place representations in curved spaces such as hyperbolic geometry. We show that this geometry creates a boundary-driven asymmetry that backdoor triggers can exploit. Near the boundary, small input changes appear subtle to standard input-space detectors but produce disproportionately large shifts in the model's representation space. Our analysis formalizes this effect and also reveals a limitation for defenses: methods that act by pulling points inward along the radius can suppress such triggers, but only by sacrificing useful model sensitivity in that same direction. Building on these insights, we propose a simple geometry-adaptive trigger and evaluate it across tasks and architectures. Empirically, attack success increases toward the boundary, whereas conventional detectors weaken, mirroring the theoretical trends. Together, these results surface a geometry-specific vulnerability in non-Euclidean models and offer analysis-backed guidance for designing and understanding the limits of defenses. 

---
# Adaptive Protein Design Protocols and Middleware 

**Authors**: Aymen Alsaadi, Jonathan Ash, Mikhail Titov, Matteo Turilli, Andre Merzky, Shantenu Jha, Sagar Khare  

**Link**: [PDF](https://arxiv.org/pdf/2510.06396)  

**Abstract**: Computational protein design is experiencing a transformation driven by AI/ML. However, the range of potential protein sequences and structures is astronomically vast, even for moderately sized proteins. Hence, achieving convergence between generated and predicted structures demands substantial computational resources for sampling. The Integrated Machine-learning for Protein Structures at Scale (IMPRESS) offers methods and advanced computing systems for coupling AI to high-performance computing tasks, enabling the ability to evaluate the effectiveness of protein designs as they are developed, as well as the models and simulations used to generate data and train models. This paper introduces IMPRESS and demonstrates the development and implementation of an adaptive protein design protocol and its supporting computing infrastructure. This leads to increased consistency in the quality of protein design and enhanced throughput of protein design due to dynamic resource allocation and asynchronous workload execution. 

---
# Reward Model Perspectives: Whose Opinions Do Reward Models Reward? 

**Authors**: Elle  

**Link**: [PDF](https://arxiv.org/pdf/2510.06391)  

**Abstract**: Reward models (RMs) are central to the alignment of language models (LMs). An RM often serves as a proxy for human preferences to guide downstream LM behavior. However, our understanding of RM behavior is limited. Our work (i) formalizes a framework for measuring the alignment of opinions captured by RMs, (ii) investigates the extent to which RMs demonstrate sociodemographic biases, and (iii) explores the effects of prompting to steer rewards towards the preferences of a target group. We study the subjective and diverse perspectives on controversial topics, which allows us to quantify RM perspectives in terms of their opinions, attitudes, and values. We show that RMs are poorly aligned with several demographic groups and can systematically reward harmful stereotypes, and steering alone is not enough to overcome these limitations. Our findings underscore the need for more careful consideration of RM behavior in model alignment during preference learning to prevent the propagation of unwanted social biases in the language technologies that we use. 

---
# Protecting De-identified Documents from Search-based Linkage Attacks 

**Authors**: Pierre Lison, Mark Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2510.06383)  

**Abstract**: While de-identification models can help conceal the identity of the individual(s) mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and then check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the document collection, making it possible to efficiently determine which N-grams appear in less than $k$ documents (either alone or in combination with other N-grams). An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on a collection of court cases show that the method is able to effectively prevent search-based linkages while remaining faithful to the original content. 

---
# Monte Carlo Permutation Search 

**Authors**: Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2510.06381)  

**Abstract**: We propose Monte Carlo Permutation Search (MCPS), a general-purpose Monte Carlo Tree Search (MCTS) algorithm that improves upon the GRAVE algorithm. MCPS is relevant when deep reinforcement learning is not an option, or when the computing power available before play is not substantial, such as in General Game Playing, for example. The principle of MCPS is to include in the exploration term of a node the statistics on all the playouts that contain all the moves on the path from the root to the node. We extensively test MCPS on a variety of games: board games, wargame, investment game, video game and multi-player games. MCPS has better results than GRAVE in all the two-player games. It has equivalent results for multi-player games because these games are inherently balanced even when players have different strengths. We also show that using abstract codes for moves instead of exact codes can be beneficial to both MCPS and GRAVE, as they improve the permutation statistics and the AMAF statistics. We also provide a mathematical derivation of the formulas used for weighting the three sources of statistics. These formulas are an improvement on the GRAVE formula since they no longer use the bias hyperparameter of GRAVE. Moreover, MCPS is not sensitive to the ref hyperparameter. 

---
# Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data 

**Authors**: Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2510.06377)  

**Abstract**: Pretrained transformers readily adapt to new sequence modeling tasks via zero-shot prompting, but relational domains still lack architectures that transfer across datasets and tasks. The core challenge is the diversity of relational data, with varying heterogeneous schemas, graph structures and functional dependencies. In this paper, we present the Relational Transformer (RT) architecture, which can be pretrained on diverse relational databases and directly applied to unseen datasets and tasks without task- or dataset-specific fine-tuning, or retrieval of in-context examples. RT (i) tokenizes cells with table/column metadata, (ii) is pretrained via masked token prediction, and (iii) utilizes a novel \textit{Relational Attention} mechanism over columns, rows, and primary-foreign key links. Pretrained on RelBench datasets spanning tasks such as churn and sales forecasting, RT attains strong zero-shot performance, averaging 94% of fully supervised AUROC on binary classification tasks with a single forward pass of a 22M parameter model, as opposed to 84% for a 27B LLM. Fine-tuning yields state-of-the-art results with high sample efficiency. Our experiments show that RT's zero-shot transfer harnesses task-table context, relational attention patterns and schema semantics. Overall, RT provides a practical path toward foundation models for relational data. 

---
# EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA 

**Authors**: Firoj Alam, Ali Ezzat Shahroor, Md. Arid Hasan, Zien Sheikh Ali, Hunzalah Hassan Bhatti, Mohamed Bayan Kmainasi, Shammur Absar Chowdhury, Basel Mousi, Fahim Dalvi, Nadir Durrani, Natasa Milic-Frayling  

**Link**: [PDF](https://arxiv.org/pdf/2510.06371)  

**Abstract**: Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they often fail when queries require culturally grounded, everyday knowledge, particularly in low-resource and underrepresented languages. To bridge this gap, we introduce Everyday Multimodal and Multilingual QA (EverydayMMQA), a framework for creating large-scale, culturally-grounded datasets for spoken and visual question answering (SVQA). Using this framework, we developed OASIS, a multimodal dataset integrating speech, images, and text. With over ~0.92M images and 14.8M QA pairs, OASIS contains 3.7M spoken questions, enabling four unique input combinations: speech-only, text-only, speech+image, and text+image. Focused on English and Arabic varieties, 18 countries, the dataset content is curated to reflect diverse, real-world situations. OASIS tests models on tasks beyond object recognition that involve pragmatic, commonsense, and culturally aware reasoning. We benchmarked four closed-source models, three open-source models, and one fine-tuned model. EverydayMMQA and OASIS together provide a benchmark and training dataset for building multimodal LLMs for a comprehensive set of everyday tasks within cultural contexts. The framework and dataset will be made publicly available to the community. 

---
# Constrained Natural Language Action Planning for Resilient Embodied Systems 

**Authors**: Grayson Byrd, Corban Rivera, Bethany Kemp, Meghan Booker, Aurora Schmidt, Celso M de Melo, Lalithkumar Seenivasan, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2510.06357)  

**Abstract**: Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems. 

---
# TransFIRA: Transfer Learning for Face Image Recognizability Assessment 

**Authors**: Allen Tu, Kartik Narayan, Joshua Gleason, Jennifer Xu, Matthew Meyn, Tom Goldstein, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2510.06353)  

**Abstract**: Face recognition in unconstrained environments such as surveillance, video, and web imagery must contend with extreme variation in pose, blur, illumination, and occlusion, where conventional visual quality metrics fail to predict whether inputs are truly recognizable to the deployed encoder. Existing FIQA methods typically rely on visual heuristics, curated annotations, or computationally intensive generative pipelines, leaving their predictions detached from the encoder's decision geometry. We introduce TransFIRA (Transfer Learning for Face Image Recognizability Assessment), a lightweight and annotation-free framework that grounds recognizability directly in embedding space. TransFIRA delivers three advances: (i) a definition of recognizability via class-center similarity (CCS) and class-center angular separation (CCAS), yielding the first natural, decision-boundary--aligned criterion for filtering and weighting; (ii) a recognizability-informed aggregation strategy that achieves state-of-the-art verification accuracy on BRIAR and IJB-C while nearly doubling correlation with true recognizability, all without external labels, heuristics, or backbone-specific training; and (iii) new extensions beyond faces, including encoder-grounded explainability that reveals how degradations and subject-specific factors affect recognizability, and the first recognizability-aware body recognition assessment. Experiments confirm state-of-the-art results on faces, strong performance on body recognition, and robustness under cross-dataset shifts. Together, these contributions establish TransFIRA as a unified, geometry-driven framework for recognizability assessment -- encoder-specific, accurate, interpretable, and extensible across modalities -- significantly advancing FIQA in accuracy, explainability, and scope. 

---
# Asking For It: Question-Answering for Predicting Rule Infractions in Online Content Moderation 

**Authors**: Mattia Samory, Diana Pamfile, Andrew To, Shruti Phadke  

**Link**: [PDF](https://arxiv.org/pdf/2510.06350)  

**Abstract**: Online communities rely on a mix of platform policies and community-authored rules to define acceptable behavior and maintain order. However, these rules vary widely across communities, evolve over time, and are enforced inconsistently, posing challenges for transparency, governance, and automation. In this paper, we model the relationship between rules and their enforcement at scale, introducing ModQ, a novel question-answering framework for rule-sensitive content moderation. Unlike prior classification or generation-based approaches, ModQ conditions on the full set of community rules at inference time and identifies which rule best applies to a given comment. We implement two model variants - extractive and multiple-choice QA - and train them on large-scale datasets from Reddit and Lemmy, the latter of which we construct from publicly available moderation logs and rule descriptions. Both models outperform state-of-the-art baselines in identifying moderation-relevant rule violations, while remaining lightweight and interpretable. Notably, ModQ models generalize effectively to unseen communities and rules, supporting low-resource moderation settings and dynamic governance environments. 

---
# Flexible Swarm Learning May Outpace Foundation Models in Essential Tasks 

**Authors**: Moein E. Samadi, Andreas Schuppert  

**Link**: [PDF](https://arxiv.org/pdf/2510.06349)  

**Abstract**: Foundation models have rapidly advanced AI, raising the question of whether their decisions will ultimately surpass human strategies in real-world domains. The exponential, and possibly super-exponential, pace of AI development makes such analysis elusive. Nevertheless, many application areas that matter for daily life and society show only modest gains so far; a prominent case is diagnosing and treating dynamically evolving disease in intensive care.
The common challenge is adapting complex systems to dynamic environments. Effective strategies must optimize outcomes in systems composed of strongly interacting functions while avoiding shared side effects; this requires reliable, self-adaptive modeling. These tasks align with building digital twins of highly complex systems whose mechanisms are not fully or quantitatively understood. It is therefore essential to develop methods for self-adapting AI models with minimal data and limited mechanistic knowledge. As this challenge extends beyond medicine, AI should demonstrate clear superiority in these settings before assuming broader decision-making roles.
We identify the curse of dimensionality as a fundamental barrier to efficient self-adaptation and argue that monolithic foundation models face conceptual limits in overcoming it. As an alternative, we propose a decentralized architecture of interacting small agent networks (SANs). We focus on agents representing the specialized substructure of the system, where each agent covers only a subset of the full system functions. Drawing on mathematical results on the learning behavior of SANs and evidence from existing applications, we argue that swarm-learning in diverse swarms can enable self-adaptive SANs to deliver superior decision-making in dynamic environments compared with monolithic foundation models, though at the cost of reduced reproducibility in detail. 

---
# Leveraging Large Language Models for Cybersecurity Risk Assessment -- A Case from Forestry Cyber-Physical Systems 

**Authors**: Fikret Mert Gültekin, Oscar Lilja, Ranim Khojah, Rebekka Wohlrab, Marvin Damschen, Mazen Mohamad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06343)  

**Abstract**: In safety-critical software systems, cybersecurity activities become essential, with risk assessment being one of the most critical. In many software teams, cybersecurity experts are either entirely absent or represented by only a small number of specialists. As a result, the workload for these experts becomes high, and software engineers would need to conduct cybersecurity activities themselves. This creates a need for a tool to support cybersecurity experts and engineers in evaluating vulnerabilities and threats during the risk assessment process. This paper explores the potential of leveraging locally hosted large language models (LLMs) with retrieval-augmented generation to support cybersecurity risk assessment in the forestry domain while complying with data protection and privacy requirements that limit external data sharing. We performed a design science study involving 12 experts in interviews, interactive sessions, and a survey within a large-scale project. The results demonstrate that LLMs can assist cybersecurity experts by generating initial risk assessments, identifying threats, and providing redundancy checks. The results also highlight the necessity for human oversight to ensure accuracy and compliance. Despite trust concerns, experts were willing to utilize LLMs in specific evaluation and assistance roles, rather than solely relying on their generative capabilities. This study provides insights that encourage the use of LLM-based agents to support the risk assessment process of cyber-physical systems in safety-critical domains. 

---
# SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation 

**Authors**: Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06303)  

**Abstract**: We propose SDAR, a Synergistic Diffusion-Autoregression paradigm that unifies the training efficiency of autoregressive models with the parallel inference capability of diffusion. Instead of costly end-to-end diffusion training, SDAR performs a lightweight paradigm conversion that transforms a well-trained autoregressive (AR) model into a blockwise diffusion model through brief, data-efficient adaptation. During inference, SDAR generates sequences autoregressively across blocks for global coherence while decoding all tokens within each block in parallel via a discrete diffusion process. Extensive experiments show that AR models remain substantially more compute-efficient than masked diffusion models, providing a strong foundation for adaptation. Building on this insight, SDAR achieves efficient AR-to-diffusion conversion with minimal cost, preserving AR-level performance while enabling parallel generation. Scaling studies across dense and Mixture-of-Experts architectures confirm that SDAR scales without compromise: larger models exhibit stronger robustness to block size and decoding thresholds, yielding greater speedups without accuracy loss. Beyond efficiency, SDAR demonstrates enhanced reasoning and domain adaptability. Our 30B MoE model surpasses its AR counterpart on challenging scientific reasoning benchmarks such as GPQA and ChemBench, and gains further improvements under test-time scaling methods like majority voting and pass@k. Together, these results establish SDAR as a practical paradigm that combines the strengths of autoregression and diffusion for scalable, high-throughput reasoning. 

---
# RGBD Gaze Tracking Using Transformer for Feature Fusion 

**Authors**: Tobias J. Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06298)  

**Abstract**: Subject of this thesis is the implementation of an AI-based Gaze Tracking system using RGBD images that contain both color (RGB) and depth (D) information. To fuse the features extracted from the images, a module based on the Transformer architecture is used. The combination of RGBD input images and Transformers was chosen because it has not yet been investigated. Furthermore, a new dataset is created for training the AI models as existing datasets either do not contain depth information or only contain labels for Gaze Point Estimation that are not suitable for the task of Gaze Angle Estimation. Various model configurations are trained, validated and evaluated on a total of three different datasets. The trained models are then to be used in a real-time pipeline to estimate the gaze direction and thus the gaze point of a person in front of a computer screen. The AI model architecture used in this thesis is based on an earlier work by Lian et al. It uses a Generative Adversarial Network (GAN) to simultaneously remove depth map artifacts and extract head pose features. Lian et al. achieve a mean Euclidean error of 38.7mm on their own dataset ShanghaiTechGaze+. In this thesis, a model architecture with a Transformer module for feature fusion achieves a mean Euclidean error of 55.3mm on the same dataset, but we show that using no pre-trained GAN module leads to a mean Euclidean error of 30.1mm. Replacing the Transformer module with a Multilayer Perceptron (MLP) improves the error to 26.9mm. These results are coherent with the ones on the other two datasets. On the ETH-XGaze dataset, the model with Transformer module achieves a mean angular error of 3.59° and without Transformer module 3.26°, whereas the fundamentally different model architecture used by the dataset authors Zhang et al. achieves a mean angular error of 2.04°. On the OTH-Gaze-Estimation dataset created for... 

---
# VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Code 

**Authors**: Lingfei Zeng, Fengdi Che, Xuhan Huang, Fei Ye, Xu Xu, Binhang Yuan, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06296)  

**Abstract**: Formal verification is the next frontier for ensuring the correctness of code generated by Large Language Models (LLMs). While methods that co-generate code and formal specifications in formal languages, like Dafny, can, in principle, prove alignment with user intent, progress is bottlenecked by specification quality evaluation. Current benchmarks rely on matching against ground-truth specifications, a manual and expertise-intensive process that has limited existing datasets to a few hundred simple problems and also suffers from a reliability issue. To address this, we introduce VeriEquivBench, a new benchmark with $2,389$ complex algorithmic problems that probe the limitations of current models in both code generation and formal reasoning. Our evaluation framework replaces ground-truth matching with a formally grounded metric, the equivalence score, and rigorously verifies the quality of generated specifications and code. Our results show that generating formally verifiable code remains a profound challenge for state-of-the-art LLMs. This underscores both the difficulty of the task and the need for benchmarks like VeriEquivBench to drive progress toward scalable and reliable coding agents. 

---
# Efficient High-Resolution Image Editing with Hallucination-Aware Loss and Adaptive Tiling 

**Authors**: Young D. Kwon, Abhinav Mehrotra, Malcolm Chadwick, Alberto Gil Ramos, Sourav Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.06295)  

**Abstract**: High-resolution (4K) image-to-image synthesis has become increasingly important for mobile applications. Existing diffusion models for image editing face significant challenges, in terms of memory and image quality, when deployed on resource-constrained devices. In this paper, we present MobilePicasso, a novel system that enables efficient image editing at high resolutions, while minimising computational cost and memory usage. MobilePicasso comprises three stages: (i) performing image editing at a standard resolution with hallucination-aware loss, (ii) applying latent projection to overcome going to the pixel space, and (iii) upscaling the edited image latent to a higher resolution with adaptive context-preserving tiling. Our user study with 46 participants reveals that MobilePicasso not only improves image quality by 18-48% but reduces hallucinations by 14-51% over existing methods. MobilePicasso demonstrates significantly lower latency, e.g., up to 55.8$\times$ speed-up, yet with a small increase in runtime memory, e.g., a mere 9% increase over prior work. Surprisingly, the on-device runtime of MobilePicasso is observed to be faster than a server-based high-resolution image editing model running on an A100 GPU. 

---
# BlockGPT: Spatio-Temporal Modelling of Rainfall via Frame-Level Autoregression 

**Authors**: Cristian Meo, Varun Sarathchandran, Avijit Majhi, Shao Hung, Carlo Saccardi, Ruben Imhoff, Roberto Deidda, Remko Uijlenhoet, Justin Dauwels  

**Link**: [PDF](https://arxiv.org/pdf/2510.06293)  

**Abstract**: Predicting precipitation maps is a highly complex spatiotemporal modeling task, critical for mitigating the impacts of extreme weather events. Short-term precipitation forecasting, or nowcasting, requires models that are not only accurate but also computationally efficient for real-time applications. Current methods, such as token-based autoregressive models, often suffer from flawed inductive biases and slow inference, while diffusion models can be computationally intensive. To address these limitations, we introduce BlockGPT, a generative autoregressive transformer using batched tokenization (Block) method that predicts full two-dimensional fields (frames) at each time step. Conceived as a model-agnostic paradigm for video prediction, BlockGPT factorizes space-time by using self-attention within each frame and causal attention across frames; in this work, we instantiate it for precipitation nowcasting. We evaluate BlockGPT on two precipitation datasets, viz. KNMI (Netherlands) and SEVIR (U.S.), comparing it to state-of-the-art baselines including token-based (NowcastingGPT) and diffusion-based (DiffCast+Phydnet) models. The results show that BlockGPT achieves superior accuracy, event localization as measured by categorical metrics, and inference speeds up to 31x faster than comparable baselines. 

---
# ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations 

**Authors**: Yike Wu, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.06292)  

**Abstract**: While Large Vision-Language Models (LVLMs) achieve strong performance in multimodal tasks, hallucinations continue to hinder their reliability. Among the three categories of hallucinations, which include object, attribute, and relation, relation hallucinations account for the largest proportion but have received the least attention. To address this issue, we propose ChainMPQ (Multi-Perspective Questions guided Interleaved Chain of Image and Text), a training-free method that improves relational inference in LVLMs by utilizing accumulated textual and visual memories. ChainMPQ first extracts subject and object keywords from the question to enhance the corresponding image regions. It then constructs multi-perspective questions that focus on the three core components of a relationship: the subject, the object, and the relation that links them. These questions are sequentially input to the model, with textual and visual memories from earlier steps providing supporting context for subsequent ones, thereby forming an interleaved chain of images and text that guides progressive relational reasoning. Experiments on multiple LVLMs and benchmarks show that ChainMPQ substantially reduces relation hallucinations, while ablation studies further validate the effectiveness of its three core modules. 

---
# Traj-Transformer: Diffusion Models with Transformer for GPS Trajectory Generation 

**Authors**: Zhiyang Zhang, Ningcong Chen, Xin Zhang, Yanhua Li, Shen Su, Hui Lu, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.06291)  

**Abstract**: The widespread use of GPS devices has driven advances in spatiotemporal data mining, enabling machine learning models to simulate human decision making and generate realistic trajectories, addressing both data collection costs and privacy concerns. Recent studies have shown the promise of diffusion models for high-quality trajectory generation. However, most existing methods rely on convolution based architectures (e.g. UNet) to predict noise during the diffusion process, which often results in notable deviations and the loss of fine-grained street-level details due to limited model capacity. In this paper, we propose Trajectory Transformer, a novel model that employs a transformer backbone for both conditional information embedding and noise prediction. We explore two GPS coordinate embedding strategies, location embedding and longitude-latitude embedding, and analyze model performance at different scales. Experiments on two real-world datasets demonstrate that Trajectory Transformer significantly enhances generation quality and effectively alleviates the deviation issues observed in prior approaches. 

---
# Soft-Evidence Fused Graph Neural Network for Cancer Driver Gene Identification across Multi-View Biological Graphs 

**Authors**: Bang Chen, Lijun Guo, Houli Fan, Wentao He, Rong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06290)  

**Abstract**: Identifying cancer driver genes (CDGs) is essential for understanding cancer mechanisms and developing targeted therapies. Graph neural networks (GNNs) have recently been employed to identify CDGs by capturing patterns in biological interaction networks. However, most GNN-based approaches rely on a single protein-protein interaction (PPI) network, ignoring complementary information from other biological networks. Some studies integrate multiple networks by aligning features with consistency constraints to learn unified gene representations for CDG identification. However, such representation-level fusion often assumes congruent gene relationships across networks, which may overlook network heterogeneity and introduce conflicting information. To address this, we propose Soft-Evidence Fusion Graph Neural Network (SEFGNN), a novel framework for CDG identification across multiple networks at the decision level. Instead of enforcing feature-level consistency, SEFGNN treats each biological network as an independent evidence source and performs uncertainty-aware fusion at the decision level using Dempster-Shafer Theory (DST). To alleviate the risk of overconfidence from DST, we further introduce a Soft Evidence Smoothing (SES) module that improves ranking stability while preserving discriminative performance. Experiments on three cancer datasets show that SEFGNN consistently outperforms state-of-the-art baselines and exhibits strong potential in discovering novel CDGs. 

---
# SER-Diff: Synthetic Error Replay Diffusion for Incremental Brain Tumor Segmentation 

**Authors**: Sashank Makanaboyina  

**Link**: [PDF](https://arxiv.org/pdf/2510.06283)  

**Abstract**: Incremental brain tumor segmentation is critical for models that must adapt to evolving clinical datasets without retraining on all prior data. However, catastrophic forgetting, where models lose previously acquired knowledge, remains a major obstacle. Recent incremental learning frameworks with knowledge distillation partially mitigate forgetting but rely heavily on generative replay or auxiliary storage. Meanwhile, diffusion models have proven effective for refining tumor segmentations, but have not been explored in incremental learning contexts. We propose Synthetic Error Replay Diffusion (SER-Diff), the first framework that unifies diffusion-based refinement with incremental learning. SER-Diff leverages a frozen teacher diffusion model to generate synthetic error maps from past tasks, which are replayed during training on new tasks. A dual-loss formulation combining Dice loss for new data and knowledge distillation loss for replayed errors ensures both adaptability and retention. Experiments on BraTS2020, BraTS2021, and BraTS2023 demonstrate that SER-Diff consistently outperforms prior methods. It achieves the highest Dice scores of 95.8\%, 94.9\%, and 94.6\%, along with the lowest HD95 values of 4.4 mm, 4.7 mm, and 4.9 mm, respectively. These results indicate that SER-Diff not only mitigates catastrophic forgetting but also delivers more accurate and anatomically coherent segmentations across evolving datasets. 

---
# Improving the Spatial Resolution of GONG Solar Images to GST Quality Using Deep Learning 

**Authors**: Chenyang Li, Qin Li, Haimin Wang, Bo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06281)  

**Abstract**: High-resolution (HR) solar imaging is crucial for capturing fine-scale dynamic features such as filaments and fibrils. However, the spatial resolution of the full-disk H$\alpha$ images is limited and insufficient to resolve these small-scale structures. To address this, we propose a GAN-based superresolution approach to enhance low-resolution (LR) full-disk H$\alpha$ images from the Global Oscillation Network Group (GONG) to a quality comparable with HR observations from the Big Bear Solar Observatory/Goode Solar Telescope (BBSO/GST). We employ Real-ESRGAN with Residual-in-Residual Dense Blocks and a relativistic discriminator. We carefully aligned GONG-GST pairs. The model effectively recovers fine details within sunspot penumbrae and resolves fine details in filaments and fibrils, achieving an average mean squared error (MSE) of 467.15, root mean squared error (RMSE) of 21.59, and cross-correlation (CC) of 0.7794. Slight misalignments between image pairs limit quantitative performance, which we plan to address in future work alongside dataset expansion to further improve reconstruction quality. 

---
# Surgeons Are Indian Males and Speech Therapists Are White Females: Auditing Biases in Vision-Language Models for Healthcare Professionals 

**Authors**: Zohaib Hasan Siddiqui, Dayam Nadeem, Mohammad Masudur Rahman, Mohammad Nadeem, Shahab Saquib Sohail, Beenish Moalla Chaudhry  

**Link**: [PDF](https://arxiv.org/pdf/2510.06280)  

**Abstract**: Vision language models (VLMs), such as CLIP and OpenCLIP, can encode and reflect stereotypical associations between medical professions and demographic attributes learned from web-scale data. We present an evaluation protocol for healthcare settings that quantifies associated biases and assesses their operational risk. Our methodology (i) defines a taxonomy spanning clinicians and allied healthcare roles (e.g., surgeon, cardiologist, dentist, nurse, pharmacist, technician), (ii) curates a profession-aware prompt suite to probe model behavior, and (iii) benchmarks demographic skew against a balanced face corpus. Empirically, we observe consistent demographic biases across multiple roles and vision models. Our work highlights the importance of bias identification in critical domains such as healthcare as AI-enabled hiring and workforce analytics can have downstream implications for equity, compliance, and patient trust. 

---
# RVFL-X: A Novel Randomized Network Based on Complex Transformed Real-Valued Tabular Datasets 

**Authors**: M. Sajid, Mushir Akhtar, A. Quadir, M. Tanveer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06278)  

**Abstract**: Recent advancements in neural networks, supported by foundational theoretical insights, emphasize the superior representational power of complex numbers. However, their adoption in randomized neural networks (RNNs) has been limited due to the lack of effective methods for transforming real-valued tabular datasets into complex-valued representations. To address this limitation, we propose two methods for generating complex-valued representations from real-valued datasets: a natural transformation and an autoencoder-driven method. Building on these mechanisms, we propose RVFL-X, a complex-valued extension of the random vector functional link (RVFL) network. RVFL-X integrates complex transformations into real-valued datasets while maintaining the simplicity and efficiency of the original RVFL architecture. By leveraging complex components such as input, weights, and activation functions, RVFL-X processes complex representations and produces real-valued outputs. Comprehensive evaluations on 80 real-valued UCI datasets demonstrate that RVFL-X consistently outperforms both the original RVFL and state-of-the-art (SOTA) RNN variants, showcasing its robustness and effectiveness across diverse application domains. 

---
# A Total Variation Regularized Framework for Epilepsy-Related MRI Image Segmentation 

**Authors**: Mehdi Rabiee, Sergio Greco, Reza Shahbazian, Irina Trubitsyna  

**Link**: [PDF](https://arxiv.org/pdf/2510.06276)  

**Abstract**: Focal Cortical Dysplasia (FCD) is a primary cause of drug-resistant epilepsy and is difficult to detect in brain {magnetic resonance imaging} (MRI) due to the subtle and small-scale nature of its lesions. Accurate segmentation of FCD regions in 3D multimodal brain MRI images is essential for effective surgical planning and treatment. However, this task remains highly challenging due to the limited availability of annotated FCD datasets, the extremely small size and weak contrast of FCD lesions, the complexity of handling 3D multimodal inputs, and the need for output smoothness and anatomical consistency, which is often not addressed by standard voxel-wise loss functions. This paper presents a new framework for segmenting FCD regions in 3D brain MRI images. We adopt state-of-the-art transformer-enhanced encoder-decoder architecture and introduce a novel loss function combining Dice loss with an anisotropic {Total Variation} (TV) term. This integration encourages spatial smoothness and reduces false positive clusters without relying on post-processing. The framework is evaluated on a public FCD dataset with 85 epilepsy patients and demonstrates superior segmentation accuracy and consistency compared to standard loss formulations. The model with the proposed TV loss shows an 11.9\% improvement on the Dice coefficient and 13.3\% higher precision over the baseline model. Moreover, the number of false positive clusters is reduced by 61.6% 

---
# Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation" 

**Authors**: Ranjan Mishra, Julian I. Bibo, Quinten van Engelen, Henk Schaapman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06275)  

**Abstract**: In this study, we reproduced the work done in the paper "XRec: Large Language Models for Explainable Recommendation" by Ma et al. (2024). The original authors introduced XRec, a model-agnostic collaborative instruction-tuning framework that enables large language models (LLMs) to provide users with comprehensive explanations of generated recommendations. Our objective was to replicate the results of the original paper, albeit using Llama 3 as the LLM for evaluation instead of GPT-3.5-turbo. We built on the source code provided by Ma et al. (2024) to achieve our goal. Our work extends the original paper by modifying the input embeddings or deleting the output embeddings of XRec's Mixture of Experts module. Based on our results, XRec effectively generates personalized explanations and its stability is improved by incorporating collaborative information. However, XRec did not consistently outperform all baseline models in every metric. Our extended analysis further highlights the importance of the Mixture of Experts embeddings in shaping the explanation structures, showcasing how collaborative signals interact with language modeling. Through our work, we provide an open-source evaluation implementation that enhances accessibility for researchers and practitioners alike. Our complete code repository can be found at this https URL. 

---
# MCCE: A Framework for Multi-LLM Collaborative Co-Evolution 

**Authors**: Nian Ran, Zhongzheng Li, Yue Wang, Qingsong Ran, Xiaoyuan Zhang, Shikun Feng, Richard Allmendinger, Xiaoguang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06270)  

**Abstract**: Multi-objective discrete optimization problems, such as molecular design, pose significant challenges due to their vast and unstructured combinatorial spaces. Traditional evolutionary algorithms often get trapped in local optima, while expert knowledge can provide crucial guidance for accelerating convergence. Large language models (LLMs) offer powerful priors and reasoning ability, making them natural optimizers when expert knowledge matters. However, closed-source LLMs, though strong in exploration, cannot update their parameters and thus cannot internalize experience. Conversely, smaller open models can be continually fine-tuned but lack broad knowledge and reasoning strength. We introduce Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework that unites a frozen closed-source LLM with a lightweight trainable model. The system maintains a trajectory memory of past search processes; the small model is progressively refined via reinforcement learning, with the two models jointly supporting and complementing each other in global exploration. Unlike model distillation, this process enhances the capabilities of both models through mutual inspiration. Experiments on multi-objective drug design benchmarks show that MCCE achieves state-of-the-art Pareto front quality and consistently outperforms baselines. These results highlight a new paradigm for enabling continual evolution in hybrid LLM systems, combining knowledge-driven exploration with experience-driven learning. 

---
# RareGraph-Synth: Knowledge-Guided Diffusion Models for Generating Privacy-Preserving Synthetic Patient Trajectories in Ultra-Rare Diseases 

**Authors**: Khartik Uppalapati, Shakeel Abdulkareem, Bora Yimenicioglu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06267)  

**Abstract**: We propose RareGraph-Synth, a knowledge-guided, continuous-time diffusion framework that generates realistic yet privacy-preserving synthetic electronic-health-record (EHR) trajectories for ultra-rare diseases. RareGraph-Synth unifies five public resources: Orphanet/Orphadata, the Human Phenotype Ontology (HPO), the GARD rare-disease KG, PrimeKG, and the FDA Adverse Event Reporting System (FAERS) into a heterogeneous knowledge graph comprising approximately 8 M typed edges. Meta-path scores extracted from this 8-million-edge KG modulate the per-token noise schedule in the forward stochastic differential equation, steering generation toward biologically plausible lab-medication-adverse-event co-occurrences while retaining score-based diffusion model stability. The reverse denoiser then produces timestamped sequences of lab-code, medication-code, and adverse-event-flag triples that contain no protected health information. On simulated ultra-rare-disease cohorts, RareGraph-Synth lowers categorical Maximum Mean Discrepancy by 40 percent relative to an unguided diffusion baseline and by greater than 60 percent versus GAN counterparts, without sacrificing downstream predictive utility. A black-box membership-inference evaluation using the DOMIAS attacker yields AUROC approximately 0.53, well below the 0.55 safe-release threshold and substantially better than the approximately 0.61 plus or minus 0.03 observed for non-KG baselines, demonstrating strong resistance to re-identification. These results suggest that integrating biomedical knowledge graphs directly into diffusion noise schedules can simultaneously enhance fidelity and privacy, enabling safer data sharing for rare-disease research. 

---
# Language models for longitudinal analysis of abusive content in Billboard Music Charts 

**Authors**: Rohitash Chandra, Yathin Suresh, Divyansh Raj Sinha, Sanchit Jindal  

**Link**: [PDF](https://arxiv.org/pdf/2510.06266)  

**Abstract**: There is no doubt that there has been a drastic increase in abusive and sexually explicit content in music, particularly in Billboard Music Charts. However, there is a lack of studies that validate the trend for effective policy development, as such content has harmful behavioural changes in children and youths. In this study, we utilise deep learning methods to analyse songs (lyrics) from Billboard Charts of the United States in the last seven decades. We provide a longitudinal study using deep learning and language models and review the evolution of content using sentiment analysis and abuse detection, including sexually explicit content. Our results show a significant rise in explicit content in popular music from 1990 onwards. Furthermore, we find an increasing prevalence of songs with lyrics containing profane, sexually explicit, and otherwise inappropriate language. The longitudinal analysis of the ability of language models to capture nuanced patterns in lyrical content, reflecting shifts in societal norms and language use over time. 

---
# Dual-stage and Lightweight Patient Chart Summarization for Emergency Physicians 

**Authors**: Jiajun Wu, Swaleh Zaidi, Braden Teitge, Henry Leung, Jiayu Zhou, Jessalyn Holodinsky, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.06263)  

**Abstract**: Electronic health records (EHRs) contain extensive unstructured clinical data that can overwhelm emergency physicians trying to identify critical information. We present a two-stage summarization system that runs entirely on embedded devices, enabling offline clinical summarization while preserving patient privacy. In our approach, a dual-device architecture first retrieves relevant patient record sections using the Jetson Nano-R (Retrieve), then generates a structured summary on another Jetson Nano-S (Summarize), communicating via a lightweight socket link. The summarization output is two-fold: (1) a fixed-format list of critical findings, and (2) a context-specific narrative focused on the clinician's query. The retrieval stage uses locally stored EHRs, splits long notes into semantically coherent sections, and searches for the most relevant sections per query. The generation stage uses a locally hosted small language model (SLM) to produce the summary from the retrieved text, operating within the constraints of two NVIDIA Jetson devices. We first benchmarked six open-source SLMs under 7B parameters to identify viable models. We incorporated an LLM-as-Judge evaluation mechanism to assess summary quality in terms of factual accuracy, completeness, and clarity. Preliminary results on MIMIC-IV and de-identified real EHRs demonstrate that our fully offline system can effectively produce useful summaries in under 30 seconds. 

---
# Prakriti200: A Questionnaire-Based Dataset of 200 Ayurvedic Prakriti Assessments 

**Authors**: Aryan Kumar Singh, Janvi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06262)  

**Abstract**: This dataset provides responses to a standardized, bilingual (English-Hindi) Prakriti Assessment Questionnaire designed to evaluate the physical, physiological, and psychological characteristics of individuals according to classical Ayurvedic principles. The questionnaire consists of 24 multiple-choice items covering body features, appetite, sleep patterns, energy levels, and temperament. It was developed following AYUSH/CCRAS guidelines to ensure comprehensive and accurate data collection. All questions are mandatory and neutrally phrased to minimize bias, and dosha labels (Vata, Pitta, Kapha) are hidden from participants. Data were collected via a Google Forms deployment, enabling automated scoring of responses to map individual traits to dosha-specific scores. The resulting dataset provides a structured platform for research in computational intelligence, Ayurvedic studies, and personalized health analytics, supporting analysis of trait distributions, correlations, and predictive modeling. It can also serve as a reference for future Prakriti-based studies and the development of intelligent health applications. 

---
# Ensemble Deep Learning and LLM-Assisted Reporting for Automated Skin Lesion Diagnosis 

**Authors**: Sher Khan, Raz Muhammad, Adil Hussain, Muhammad Sajjad, Muhammad Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2510.06260)  

**Abstract**: Cutaneous malignancies demand early detection for favorable outcomes, yet current diagnostics suffer from inter-observer variability and access disparities. While AI shows promise, existing dermatological systems are limited by homogeneous architectures, dataset biases across skin tones, and fragmented approaches that treat natural language processing as separate post-hoc explanations rather than integral to clinical decision-making. We introduce a unified framework that fundamentally reimagines AI integration for dermatological diagnostics through two synergistic innovations. First, a purposefully heterogeneous ensemble of architecturally diverse convolutional neural networks provides complementary diagnostic perspectives, with an intrinsic uncertainty mechanism flagging discordant cases for specialist review -- mimicking clinical best practices. Second, we embed large language model capabilities directly into the diagnostic workflow, transforming classification outputs into clinically meaningful assessments that simultaneously fulfill medical documentation requirements and deliver patient-centered education. This seamless integration generates structured reports featuring precise lesion characterization, accessible diagnostic reasoning, and actionable monitoring guidance -- empowering patients to recognize early warning signs between visits. By addressing both diagnostic reliability and communication barriers within a single cohesive system, our approach bridges the critical translational gap that has prevented previous AI implementations from achieving clinical impact. The framework represents a significant advancement toward deployable dermatological AI that enhances diagnostic precision while actively supporting the continuum of care from initial detection through patient education, ultimately improving early intervention rates for skin lesions. 

---
# LLM-Driven Rubric-Based Assessment of Algebraic Competence in Multi-Stage Block Coding Tasks with Design and Field Evaluation 

**Authors**: Yong Oh Lee, Byeonghun Bang, Sejun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06253)  

**Abstract**: As online education platforms continue to expand, there is a growing need for assessment methods that not only measure answer accuracy but also capture the depth of students' cognitive processes in alignment with curriculum objectives. This study proposes and evaluates a rubric-based assessment framework powered by a large language model (LLM) for measuring algebraic competence, real-world-context block coding tasks. The problem set, designed by mathematics education experts, aligns each problem segment with five predefined rubric dimensions, enabling the LLM to assess both correctness and quality of students' problem-solving processes. The system was implemented on an online platform that records all intermediate responses and employs the LLM for rubric-aligned achievement evaluation. To examine the practical effectiveness of the proposed framework, we conducted a field study involving 42 middle school students engaged in multi-stage quadratic equation tasks with block coding. The study integrated learner self-assessments and expert ratings to benchmark the system's outputs. The LLM-based rubric evaluation showed strong agreement with expert judgments and consistently produced rubric-aligned, process-oriented feedback. These results demonstrate both the validity and scalability of incorporating LLM-driven rubric assessment into online mathematics and STEM education platforms. 

---
# Dream2Image : An Open Multimodal EEG Dataset for Decoding and Visualizing Dreams with Artificial Intelligence 

**Authors**: Yann Bellec  

**Link**: [PDF](https://arxiv.org/pdf/2510.06252)  

**Abstract**: Dream2Image is the world's first dataset combining EEG signals, dream transcriptions, and AI-generated images. Based on 38 participants and more than 31 hours of dream EEG recordings, it contains 129 samples offering: the final seconds of brain activity preceding awakening (T-15, T-30, T-60, T-120), raw reports of dream experiences, and an approximate visual reconstruction of the dream. This dataset provides a novel resource for dream research, a unique resource to study the neural correlates of dreaming, to develop models for decoding dreams from brain activity, and to explore new approaches in neuroscience, psychology, and artificial intelligence. Available in open access on Hugging Face and GitHub, Dream2Image provides a multimodal resource designed to support research at the interface of artificial intelligence and neuroscience. It was designed to inspire researchers and extend the current approaches to brain activity decoding. Limitations include the relatively small sample size and the variability of dream recall, which may affect generalizability. 

---
# Scalable multilingual PII annotation for responsible AI in LLMs 

**Authors**: Bharti Meena, Joanna Skubisz, Harshit Rajgarhia, Nand Dave, Kiran Ganesh, Shivali Dalmia, Abhishek Mukherji, Vasudevan Sundarababu, Olga Pospelova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06250)  

**Abstract**: As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability. 

---
# TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B 

**Authors**: Toshiki Nakai, Ravi Kiran Chikkala, Lena Sophie Oberkircher, Nicholas Jennings, Natalia Skachkova, Tatiana Anikina, Jesujoba Oluwadara Alabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06249)  

**Abstract**: The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings. 

---
# DynBenchmark: Customizable Ground Truths to Benchmark Community Detection and Tracking in Temporal Networks 

**Authors**: Laurent Brisson, Cécile Bothorel, Nicolas Duminy  

**Link**: [PDF](https://arxiv.org/pdf/2510.06245)  

**Abstract**: Graph models help understand network dynamics and evolution. Creating graphs with controlled topology and embedded partitions is a common strategy for evaluating community detection algorithms. However, existing benchmarks often overlook the need to track the evolution of communities in real-world networks. To address this, a new community-centered model is proposed to generate customizable evolving community structures where communities can grow, shrink, merge, split, appear or disappear. This benchmark also generates the underlying temporal network, where nodes can appear, disappear, or move between communities. The benchmark has been used to test three methods, measuring their performance in tracking nodes' cluster membership and detecting community evolution. Python libraries, drawing utilities, and validation metrics are provided to compare ground truth with algorithm results for detecting dynamic communities. 

---
# Evaluating Embedding Frameworks for Scientific Domain 

**Authors**: Nouman Ahmed, Ronin Wu, Victor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2510.06244)  

**Abstract**: Finding an optimal word representation algorithm is particularly important in terms of domain specific data, as the same word can have different meanings and hence, different representations depending on the domain and context. While Generative AI and transformer architecture does a great job at generating contextualized embeddings for any given work, they are quite time and compute extensive, especially if we were to pre-train such a model from scratch. In this work, we focus on the scientific domain and finding the optimal word representation algorithm along with the tokenization method that could be used to represent words in the scientific domain. The goal of this research is two fold: 1) finding the optimal word representation and tokenization methods that can be used in downstream scientific domain NLP tasks, and 2) building a comprehensive evaluation suite that could be used to evaluate various word representation and tokenization algorithms (even as new ones are introduced) in the scientific domain. To this end, we build an evaluation suite consisting of several downstream tasks and relevant datasets for each task. Furthermore, we use the constructed evaluation suite to test various word representation and tokenization algorithms. 

---
# CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning 

**Authors**: Qihua Dong, Luis Figueroa, Handong Zhao, Kushal Kafle, Jason Kuen, Zhihong Ding, Scott Cohen, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06243)  

**Abstract**: Referring Expression Comprehension and Segmentation are critical tasks for assessing the integration of language understanding and image comprehension, serving as benchmarks for Multimodal Large Language Models (MLLMs) capabilities. To address these challenges, we propose a new strategy, CoT Referring, which enhances model reasoning across modalities through a structured, chain-of-thought training data structure. Our approach systematically parses textual structures to a sequential referring step, where in each step it identifies relationships and ensures consistent reference alignment, thereby improving accuracy in complex query scenarios. We restructure the training data to enforce a new output form, providing new annotations for existing datasets and compiling an evaluation benchmark from existing resources. This benchmark is designed explicitly for complex referring cases. We also integrate detection and segmentation capabilities into a unified MLLM framework, training it with a novel adaptive weighted loss to optimize performance. Experimental results on our curated benchmark and RefCOCO/+/g demonstrate the effectiveness of our approach, with a notable increase of 2.5%+ over baseline models. 

---
# Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses 

**Authors**: Subin An, Yugyeong Ji, Junyoung Kim, Heejin Kook, Yang Lu, Josh Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06242)  

**Abstract**: Open-ended survey responses provide valuable insights in marketing research, but low-quality responses not only burden researchers with manual filtering but also risk leading to misleading conclusions, underscoring the need for effective evaluation. Existing automatic evaluation methods target LLM-generated text and inadequately assess human-written responses with their distinct characteristics. To address such characteristics, we propose a two-stage evaluation framework specifically designed for human survey responses. First, gibberish filtering removes nonsensical responses. Then, three dimensions-effort, relevance, and completeness-are evaluated using LLM capabilities, grounded in empirical analysis of real-world survey data. Validation on English and Korean datasets shows that our framework not only outperforms existing metrics but also demonstrates high practical applicability for real-world applications such as response quality prediction and response rejection, showing strong correlations with expert assessment. 

---
# Knowledge Graph-Guided Multi-Agent Distillation for Reliable Industrial Question Answering with Datasets 

**Authors**: Jiqun Pan, Zhenke Duan, Jiani Tu, Anzhi Cheng, Yanqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06240)  

**Abstract**: Industrial question-answering (QA) systems require higher safety and reliability than general-purpose dialogue models, as errors in high-risk scenarios such as equipment fault diagnosis can have severe consequences. Although multi-agent large language models enhance reasoning depth, they suffer from uncontrolled iterations and unverifiable outputs, and conventional distillation methods struggle to transfer collaborative reasoning capabilities to lightweight, deployable student models. To address these challenges, we propose Knowledge Graph-guided Multi-Agent System Distillation (KG-MASD). Our approach formulates distillation as a Markov Decision Process and incorporates a knowledge graph as a verifiable structured prior to enrich state representation and ensure convergence. By integrating collaborative reasoning with knowledge grounding, KG-MASD generates high-confidence instruction-tuning data and jointly distills reasoning depth and verifiability into compact student models suitable for edge deployment. Experiments on an industrial QA dataset show that KG-MASD improves accuracy by 2.4 per cent to 20.1 per cent over baselines and significantly enhances reliability, enabling trustworthy AI deployment in safety-critical industrial scenarios. Code and data are available at this https URL. 

---
# Uncertainty Quantification In Surface Landmines and UXO Classification Using MC Dropout 

**Authors**: Sagar Lekhak, Emmett J. Ientilucci, Dimah Dera, Susmita Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06238)  

**Abstract**: Detecting surface landmines and unexploded ordnances (UXOs) using deep learning has shown promise in humanitarian demining. However, deterministic neural networks can be vulnerable to noisy conditions and adversarial attacks, leading to missed detection or misclassification. This study introduces the idea of uncertainty quantification through Monte Carlo (MC) Dropout, integrated into a fine-tuned ResNet-50 architecture for surface landmine and UXO classification, which was tested on a simulated dataset. Integrating the MC Dropout approach helps quantify epistemic uncertainty, providing an additional metric for prediction reliability, which could be helpful to make more informed decisions in demining operations. Experimental results on clean, adversarially perturbed, and noisy test images demonstrate the model's ability to flag unreliable predictions under challenging conditions. This proof-of-concept study highlights the need for uncertainty quantification in demining, raises awareness about the vulnerability of existing neural networks in demining to adversarial threats, and emphasizes the importance of developing more robust and reliable models for practical applications. 

---
# Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report) 

**Authors**: Robert Scholz, Kunal Bagga, Christine Ahrends, Carlo Alberto Barbano  

**Link**: [PDF](https://arxiv.org/pdf/2510.06235)  

**Abstract**: We present our submission to the Algonauts 2025 Challenge, where the goal is to predict fMRI brain responses to movie stimuli. Our approach integrates multimodal representations from large language models, video encoders, audio models, and vision-language models, combining both off-the-shelf and fine-tuned variants. To improve performance, we enhanced textual inputs with detailed transcripts and summaries, and we explored stimulus-tuning and fine-tuning strategies for language and vision models. Predictions from individual models were combined using stacked regression, yielding solid results. Our submission, under the team name Seinfeld, ranked 10th. We make all code and resources publicly available, contributing to ongoing efforts in developing multimodal encoding models for brain activity. 

---
# Generalized Multi-agent Social Simulation Framework 

**Authors**: Gang Li, Jie Lin, Yining Tang, Ziteng Wang, Yirui Huang, Junyu Zhang, Shuang Luo, Chao Wu, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.06225)  

**Abstract**: Multi-agent social interaction has clearly benefited from Large Language Models. However, current simulation systems still face challenges such as difficulties in scaling to diverse scenarios and poor reusability due to a lack of modular design. To address these issues, we designed and developed a modular, object-oriented framework that organically integrates various base classes through a hierarchical structure, harvesting scalability and reusability. We inherited the framework to realize common derived classes. Additionally, a memory summarization mechanism is proposed to filter and distill relevant information from raw memory data, prioritizing contextually salient events and interactions. By selecting and combining some necessary derived classes, we customized a specific simulated environment. Utilizing this simulated environment, we successfully simulated human interactions on social media, replicating real-world online social behaviors. The source code for the project will be released and evolve. 

---
# Exploring Human-AI Collaboration Using Mental Models of Early Adopters of Multi-Agent Generative AI Tools 

**Authors**: Suchismita Naik, Austin L. Toombs, Amanda Snellinger, Scott Saponas, Amanda K. Hall  

**Link**: [PDF](https://arxiv.org/pdf/2510.06224)  

**Abstract**: With recent advancements in multi-agent generative AI (Gen AI), technology organizations like Microsoft are adopting these complex tools, redefining AI agents as active collaborators in complex workflows rather than as passive tools. In this study, we investigated how early adopters and developers conceptualize multi-agent Gen AI tools, focusing on how they understand human-AI collaboration mechanisms, general collaboration dynamics, and transparency in the context of AI tools. We conducted semi-structured interviews with 13 developers, all early adopters of multi-agent Gen AI technology who work at Microsoft. Our findings revealed that these early adopters conceptualize multi-agent systems as "teams" of specialized role-based and task-based agents, such as assistants or reviewers, structured similar to human collaboration models and ranging from AI-dominant to AI-assisted, user-controlled interactions. We identified key challenges, including error propagation, unpredictable and unproductive agent loop behavior, and the need for clear communication to mitigate the layered transparency issues. Early adopters' perspectives about the role of transparency underscored its importance as a way to build trust, verify and trace errors, and prevent misuse, errors, and leaks. The insights and design considerations we present contribute to CSCW research about collaborative mechanisms with capabilities ranging from AI-dominant to AI-assisted interactions, transparency and oversight strategies in human-agent and agent-agent interactions, and how humans make sense of these multi-agent systems as dynamic, role-diverse collaborators which are customizable for diverse needs and workflows. We conclude with future research directions that extend CSCW approaches to the design of inter-agent and human mediation interactions. 

---
# A Multimodal GUI Architecture for Interfacing with LLM-Based Conversational Assistants 

**Authors**: Hans G.W. van Dam  

**Link**: [PDF](https://arxiv.org/pdf/2510.06223)  

**Abstract**: Advances in large language models (LLMs) and real-time speech recognition now make it possible to issue any graphical user interface (GUI) action through natural language and receive the corresponding system response directly through the GUI. Most production applications were never designed with speech in mind. This article provides a concrete architecture that enables GUIs to interface with LLM-based speech-enabled assistants.
The architecture makes an application's navigation graph and semantics available through the Model Context Protocol (MCP). The ViewModel, part of the MVVM (Model-View-ViewModel) pattern, exposes the application's capabilities to the assistant by supplying both tools applicable to a currently visible view and application-global tools extracted from the GUI tree router. This architecture facilitates full voice accessibility while ensuring reliable alignment between spoken input and the visual interface, accompanied by consistent feedback across modalities. It future-proofs apps for upcoming OS super assistants that employ computer use agents (CUAs) and natively consume MCP if an application provides it.
To address concerns about privacy and data security, the practical effectiveness of locally deployable, open-weight LLMs for speech-enabled multimodal UIs is evaluated. Findings suggest that recent smaller open-weight models approach the performance of leading proprietary models in overall accuracy and require enterprise-grade hardware for fast responsiveness. 

---
# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives 

**Authors**: Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu Huang, Grant McKenzie, Renee Sieber, Fengran Mo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05336)  

**Abstract**: Historical archives on weather events are collections of enduring primary source records that offer rich, untapped narratives of how societies have experienced and responded to extreme weather events. These qualitative accounts provide insights into societal vulnerability and resilience that are largely absent from meteorological records, making them valuable for climate scientists to understand societal responses. However, their vast scale, noisy digitized quality, and archaic language make it difficult to transform them into structured knowledge for climate research. To address this challenge, we introduce WeatherArchive-Bench, the first benchmark for evaluating retrieval-augmented generation (RAG) systems on historical weather archives. WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which measures a system's ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives. Extensive experiments across sparse, dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that dense retrievers often fail on historical terminology, while LLMs frequently misinterpret vulnerability and resilience concepts. These findings highlight key limitations in reasoning about complex societal indicators and provide insights for designing more robust climate-focused RAG systems from archival contexts. The constructed dataset and evaluation framework are publicly available at this https URL. 

---
# AgentBuilder: Exploring Scaffolds for Prototyping User Experiences of Interface Agents 

**Authors**: Jenny T. Liang, Titus Barik, Jeffrey Nichols, Eldon Schoop, Ruijia Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04452)  

**Abstract**: Interface agents powered by generative AI models (referred to as "agents") can automate actions based on user commands. An important aspect of developing agents is their user experience (i.e., agent experience). There is a growing need to provide scaffolds for a broader set of individuals beyond AI engineers to prototype agent experiences, since they can contribute valuable perspectives to designing agent experiences. In this work, we explore the affordances agent prototyping systems should offer by conducting a requirements elicitation study with 12 participants with varying experience with agents. We identify key activities in agent experience prototyping and the desired capabilities of agent prototyping systems. We instantiate those capabilities in the AgentBuilder design probe for agent prototyping. We conduct an in situ agent prototyping study with 14 participants using AgentBuilder to validate the design requirements and elicit insights on how developers prototype agents and what their needs are in this process. 

---
# TiltXter: CNN-based Electro-tactile Rendering of Tilt Angle for Telemanipulation of Pasteur Pipettes 

**Authors**: Miguel Altamirano Cabrera, Jonathan Tirado, Aleksey Fedoseev, Oleg Sautenkov, Vladimir Poliakov, Pavel Kopanev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2409.15838)  

**Abstract**: The shape of deformable objects can change drastically during grasping by robotic grippers, causing an ambiguous perception of their alignment and hence resulting in errors in robot positioning and telemanipulation. Rendering clear tactile patterns is fundamental to increasing users' precision and dexterity through tactile haptic feedback during telemanipulation. Therefore, different methods have to be studied to decode the sensors' data into haptic stimuli. This work presents a telemanipulation system for plastic pipettes that consists of a Force Dimension Omega.7 haptic interface endowed with two electro-stimulation arrays and two tactile sensor arrays embedded in the 2-finger Robotiq gripper. We propose a novel approach based on convolutional neural networks (CNN) to detect the tilt of deformable objects. The CNN generates a tactile pattern based on recognized tilt data to render further electro-tactile stimuli provided to the user during the telemanipulation. The study has shown that using the CNN algorithm, tilt recognition by users increased from 23.13\% with the downsized data to 57.9%, and the success rate during teleoperation increased from 53.12% using the downsized data to 92.18% using the tactile patterns generated by the CNN. 

---
# DeepXPalm: Tilt and Position Rendering using Palm-worn Haptic Display and CNN-based Tactile Pattern Recognition 

**Authors**: Altamirano Cabrera Miguel, Sautenkov Oleg, Tirado Jonathan, Fedoseev Aleksey, Kopanev Pavel, Kajimoto Hiroyuki, Tsetserukou Dzmitry  

**Link**: [PDF](https://arxiv.org/pdf/2204.03521)  

**Abstract**: Telemanipulation of deformable objects requires high precision and dexterity from the users, which can be increased by kinesthetic and tactile feedback. However, the object shape can change dynamically, causing ambiguous perception of its alignment and hence errors in the robot positioning. Therefore, the tilt angle and position classification problem has to be solved to present a clear tactile pattern to the user. This work presents a telemanipulation system for plastic pipettes consisting of a multi-contact haptic device LinkGlide to deliver haptic feedback at the users' palm and two tactile sensors array embedded in the 2-finger Robotiq gripper. We propose a novel approach based on Convolutional Neural Networks (CNN) to detect the tilt and position while grasping deformable objects. The CNN generates a mask based on recognized tilt and position data to render further multi-contact tactile stimuli provided to the user during the telemanipulation. The study has shown that using the CNN algorithm and the preset mask, tilt, and position recognition by users is increased from 9.67% using the direct data to 82.5%. 

---
