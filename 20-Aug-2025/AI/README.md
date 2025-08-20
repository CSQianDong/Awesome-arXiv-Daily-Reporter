# ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents 

**Authors**: Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14040)  

**Abstract**: We introduce ComputerRL, a framework for autonomous desktop intelligence that enables agents to operate complex digital workspaces skillfully. ComputerRL features the API-GUI paradigm, which unifies programmatic API calls and direct GUI interaction to address the inherent mismatch between machine agents and human-centric desktop environments. Scaling end-to-end RL training is crucial for improvement and generalization across diverse desktop tasks, yet remains challenging due to environmental inefficiency and instability in extended training. To support scalable and robust training, we develop a distributed RL infrastructure capable of orchestrating thousands of parallel virtual desktop environments to accelerate large-scale online RL. Furthermore, we propose Entropulse, a training strategy that alternates reinforcement learning with supervised fine-tuning, effectively mitigating entropy collapse during extended training runs. We employ ComputerRL on open models GLM-4-9B-0414 and Qwen2.5-14B, and evaluate them on the OSWorld benchmark. The AutoGLM-OS-9B based on GLM-4-9B-0414 achieves a new state-of-the-art accuracy of 48.1%, demonstrating significant improvements for general agents in desktop automation. The algorithm and framework are adopted in building AutoGLM (Liu et al., 2024a) 

---
# A Biased Random Key Genetic Algorithm for Solving the Longest Run Subsequence Problem 

**Authors**: Christian Blum, Pedro Pinacho-Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.14020)  

**Abstract**: The longest run subsequence (LRS) problem is an NP-hard combinatorial optimization problem belonging to the class of subsequence problems from bioinformatics. In particular, the problem plays a role in genome reassembly. In this paper, we present a solution to the LRS problem using a Biased Random Key Genetic Algorithm (BRKGA). Our approach places particular focus on the computational efficiency of evaluating individuals, which involves converting vectors of gray values into valid solutions to the problem. For comparison purposes, a Max-Min Ant System is developed and implemented. This is in addition to the application of the integer linear programming solver CPLEX for solving all considered problem instances. The computation results show that the proposed BRKGA is currently a state-of-the-art technique for the LRS problem. Nevertheless, the results also show that there is room for improvement, especially in the context of input strings based on large alphabet sizes. 

---
# ChronoLLM: Customizing Language Models for Physics-Based Simulation Code Generation 

**Authors**: Jingquan Wang, Andrew Negrut, Harry Zhang, Khailanii Slaton, Shu Wang, Radu Serban, Jinlong Wu, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13975)  

**Abstract**: This contribution is concerned with the following issue: can pretrained large language models (LLMs) be refined and customized to the point where they become virtual assistants helping experts with the effective use of a simulation tool? In this case study, the ``simulation tool'' considered is PyChrono, an open source multi-physics dynamics engine for multibody systems. We present a framework for refining and customizing both open- and closed-source LLMs to harness the power of AI in generating scripts that perform PyChrono virtual experiments. We refine and customize several classes of LLMs through a process that leads to a quantifiable improvement in the quality of the generated PyChrono simulation scripts. These scripts can range from simple single-pendulum simulations to complex virtual experiments involving full vehicles on deformable terrain. While the generated scripts are rarely perfect, they often serve as strong starting points for the user to modify and improve on. Additionally, the LLM can answer specific API questions about the simulator, or recommend modeling approaches. The framework discussed is general and can be applied to lower the entry barrier for simulation tools associated with other application domains. 

---
# The Collaboration Paradox: Why Generative AI Requires Both Strategic Intelligence and Operational Stability in Supply Chain Management 

**Authors**: Soumyadeep Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2508.13942)  

**Abstract**: The rise of autonomous, AI-driven agents in economic settings raises critical questions about their emergent strategic behavior. This paper investigates these dynamics in the cooperative context of a multi-echelon supply chain, a system famously prone to instabilities like the bullwhip effect. We conduct computational experiments with generative AI agents, powered by Large Language Models (LLMs), within a controlled supply chain simulation designed to isolate their behavioral tendencies. Our central finding is the "collaboration paradox": a novel, catastrophic failure mode where theoretically superior collaborative AI agents, designed with Vendor-Managed Inventory (VMI) principles, perform even worse than non-AI baselines. We demonstrate that this paradox arises from an operational flaw where agents hoard inventory, starving the system. We then show that resilience is only achieved through a synthesis of two distinct layers: high-level, AI-driven proactive policy-setting to establish robust operational targets, and a low-level, collaborative execution protocol with proactive downstream replenishment to maintain stability. Our final framework, which implements this synthesis, can autonomously generate, evaluate, and quantify a portfolio of viable strategic choices. The work provides a crucial insight into the emergent behaviors of collaborative AI agents and offers a blueprint for designing stable, effective AI-driven systems for business analytics. 

---
# Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback 

**Authors**: Yihao Ang, Yifan Bao, Lei Jiang, Jiajie Tao, Anthony K. H. Tung, Lukasz Szpruch, Hao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2508.13915)  

**Abstract**: Time-series data is central to decision-making in financial markets, yet building high-performing, interpretable, and auditable models remains a major challenge. While Automated Machine Learning (AutoML) frameworks streamline model development, they often lack adaptability and responsiveness to domain-specific needs and evolving objectives. Concurrently, Large Language Models (LLMs) have enabled agentic systems capable of reasoning, memory management, and dynamic code generation, offering a path toward more flexible workflow automation. In this paper, we introduce \textsf{TS-Agent}, a modular agentic framework designed to automate and enhance time-series modeling workflows for financial applications. The agent formalizes the pipeline as a structured, iterative decision process across three stages: model selection, code refinement, and fine-tuning, guided by contextual reasoning and experimental feedback. Central to our architecture is a planner agent equipped with structured knowledge banks, curated libraries of models and refinement strategies, which guide exploration, while improving interpretability and reducing error propagation. \textsf{TS-Agent} supports adaptive learning, robust debugging, and transparent auditing, key requirements for high-stakes environments such as financial services. Empirical evaluations on diverse financial forecasting and synthetic data generation tasks demonstrate that \textsf{TS-Agent} consistently outperforms state-of-the-art AutoML and agentic baselines, achieving superior accuracy, robustness, and decision traceability. 

---
# Improved Generalized Planning with LLMs through Strategy Refinement and Reflection 

**Authors**: Katharina Stein, Nils Hodel, Daniel Fišer, Jörg Hoffmann, Michael Katz, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2508.13876)  

**Abstract**: LLMs have recently been used to generate Python programs representing generalized plans in PDDL planning, i.e., plans that generalize across the tasks of a given PDDL domain. Previous work proposed a framework consisting of three steps: the LLM first generates a summary and then a strategy for the domain, both in natural language, and then implements that strategy as a Python program, that gets debugged on example planning tasks. In that work, only one strategy is generated and passed directly to the program generation. If the strategy is incorrect, its implementation will therefore result in an incorrect generalized plan. Here, we introduce an approach that generates the strategy in the form of pseudocode and enables automatic debugging of the pseudocode, hence allowing us to identify and fix errors prior to the generation of the generalized plan itself. Additionally, we extend the Python debugging phase with a reflection step prompting the LLM to pinpoint the reason for the observed plan failure. Finally, we take inspiration from LLM code generation to produce several program variants and pick the best one. Running experiments on 17 benchmark domains, we show that these extensions substantially improve (and never deteriorate) the quality of the generalized plans. In 12 of the domains, our best Python programs solve all tasks that can be generated with the respective instance generator. 

---
# Revisiting RAG Ensemble: A Theoretical and Mechanistic Analysis of Multi-RAG System Collaboration 

**Authors**: Yifei Chen, Guanting Dong, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13828)  

**Abstract**: Retrieval-Augmented Generation (RAG) technology has been widely applied in recent years. However, despite the emergence of various RAG frameworks, a single RAG framework still cannot adapt well to a broad range of downstream tasks. Therefore, how to leverage the advantages of multiple RAG systems has become an area worth exploring. To address this issue, we have conducted a comprehensive and systematic investigation into ensemble methods based on RAG systems. Specifically, we have analyzed the RAG ensemble framework from both theoretical and mechanistic analysis perspectives. From the theoretical analysis, we provide the first explanation of the RAG ensemble framework from the perspective of information entropy. In terms of mechanism analysis, we have explored the RAG ensemble framework from both the pipeline and module levels. We carefully select four different pipelines (Branching, Iterative, Loop, and Agentic) and three different modules (Generator, Retriever, and Reranker) to solve seven different research questions. The experiments show that aggregating multiple RAG systems is both generalizable and robust, whether at the pipeline level or the module level. Our work lays the foundation for similar research on the multi-RAG system ensemble. 

---
# Quantifier Instantiations: To Mimic or To Revolt? 

**Authors**: Jan Jakubův, Mikoláš Janota  

**Link**: [PDF](https://arxiv.org/pdf/2508.13811)  

**Abstract**: Quantified formulas pose a significant challenge for Satisfiability Modulo Theories (SMT) solvers due to their inherent undecidability. Existing instantiation techniques, such as e-matching, syntax-guided, model-based, conflict-based, and enumerative methods, often complement each other. This paper introduces a novel instantiation approach that dynamically learns from these techniques during solving. By treating observed instantiations as samples from a latent language, we use probabilistic context-free grammars to generate new, similar terms. Our method not only mimics successful past instantiations but also explores diversity by optionally inverting learned term probabilities, aiming to balance exploitation and exploration in quantifier reasoning. 

---
# Expertise-aware Multi-LLM Recruitment and Collaboration for Medical Decision-Making 

**Authors**: Liuxin Bao, Zhihao Peng, Xiaofei Zhou, Runmin Cong, Jiyong Zhang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13754)  

**Abstract**: Medical Decision-Making (MDM) is a complex process requiring substantial domain-specific expertise to effectively synthesize heterogeneous and complicated clinical information. While recent advancements in Large Language Models (LLMs) show promise in supporting MDM, single-LLM approaches are limited by their parametric knowledge constraints and static training corpora, failing to robustly integrate the clinical information. To address this challenge, we propose the Expertise-aware Multi-LLM Recruitment and Collaboration (EMRC) framework to enhance the accuracy and reliability of MDM systems. It operates in two stages: (i) expertise-aware agent recruitment and (ii) confidence- and adversarial-driven multi-agent collaboration. Specifically, in the first stage, we use a publicly available corpus to construct an LLM expertise table for capturing expertise-specific strengths of multiple LLMs across medical department categories and query difficulty levels. This table enables the subsequent dynamic selection of the optimal LLMs to act as medical expert agents for each medical query during the inference phase. In the second stage, we employ selected agents to generate responses with self-assessed confidence scores, which are then integrated through the confidence fusion and adversarial validation to improve diagnostic reliability. We evaluate our EMRC framework on three public MDM datasets, where the results demonstrate that our EMRC outperforms state-of-the-art single- and multi-LLM methods, achieving superior diagnostic performance. For instance, on the MMLU-Pro-Health dataset, our EMRC achieves 74.45% accuracy, representing a 2.69% improvement over the best-performing closed-source model GPT- 4-0613, which demonstrates the effectiveness of our expertise-aware agent recruitment strategy and the agent complementarity in leveraging each LLM's specialized capabilities. 

---
# CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning 

**Authors**: Minh Hoang Nguyen, Van Dai Do, Dung Nguyen, Thin Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.13721)  

**Abstract**: Large language model (LLM) agents-especially smaller, open-source models-often produce causally invalid or incoherent actions in collaborative tasks due to their reliance on surface-level correlations rather than grounded causal reasoning. This limitation undermines their performance in terms of coordination and planning in dynamic environments. We address this challenge with CausalPlan, a two-phase framework that integrates explicit structural causal reasoning into the LLM planning process. At the core of CausalPlan is the Structural Causal Action (SCA) model, which learns a causal graph from agent trajectories to capture how prior actions and current environment states influence future decisions. This structure is then used to guide action selection by assigning causal scores to LLM-generated proposals, reweighting them accordingly, or falling back to causally grounded alternatives when needed. By embedding this causal knowledge directly into the decision loop, CausalPlan constrains planning to intervention-consistent behaviours without requiring fine-tuning of the LLM itself. We evaluate CausalPlan on the Overcooked-AI benchmark across five multi-agent coordination tasks and four LLMs of varying sizes: Gemma-7B, Llama-8B, Qwen-14B, and Llama-70B. Experimental results show that CausalPlan consistently reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines. Our findings highlight the value of causality-driven planning for deploying efficient, interpretable, and generalisable multi-agent LLM systems. 

---
# The DeepLog Neurosymbolic Machine 

**Authors**: Vincent Derkinderen, Robin Manhaeve, Rik Adriaensen, Lucas Van Praet, Lennert De Smet, Giuseppe Marra, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13697)  

**Abstract**: We contribute a theoretical and operational framework for neurosymbolic AI called DeepLog. DeepLog introduces building blocks and primitives for neurosymbolic AI that make abstraction of commonly used representations and computational mechanisms used in neurosymbolic AI. DeepLog can represent and emulate a wide range of neurosymbolic systems. It consists of two key components. The first is the DeepLog language for specifying neurosymbolic models and inference tasks. This language consists of an annotated neural extension of grounded first-order logic, and makes abstraction of the type of logic, e.g. boolean, fuzzy or probabilistic, and whether logic is used in the architecture or in the loss function. The second DeepLog component is situated at the computational level and uses extended algebraic circuits as computational graphs. Together these two components are to be considered as a neurosymbolic abstract machine, with the DeepLog language as the intermediate level of abstraction and the circuits level as the computational one. DeepLog is implemented in software, relies on the latest insights in implementing algebraic circuits on GPUs, and is declarative in that it is easy to obtain different neurosymbolic models by making different choices for the underlying algebraic structures and logics. The generality and efficiency of the DeepLog neurosymbolic machine is demonstrated through an experimental comparison between 1) different fuzzy and probabilistic logics, 2) between using logic in the architecture or in the loss function, and 3) between a standalone CPU-based implementation of a neurosymbolic AI system and a DeepLog GPU-based one. 

---
# Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models 

**Authors**: Xiao-Wen Yang, Jie-Jing Shao, Lan-Zhe Guo, Bo-Wen Zhang, Zhi Zhou, Lin-Han Jia, Wang-Zhou Dai, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13678)  

**Abstract**: Large Language Models (LLMs) have shown promising results across various tasks, yet their reasoning capabilities remain a fundamental challenge. Developing AI systems with strong reasoning capabilities is regarded as a crucial milestone in the pursuit of Artificial General Intelligence (AGI) and has garnered considerable attention from both academia and industry. Various techniques have been explored to enhance the reasoning capabilities of LLMs, with neuro-symbolic approaches being a particularly promising way. This paper comprehensively reviews recent developments in neuro-symbolic approaches for enhancing LLM reasoning. We first present a formalization of reasoning tasks and give a brief introduction to the neurosymbolic learning paradigm. Then, we discuss neuro-symbolic methods for improving the reasoning capabilities of LLMs from three perspectives: Symbolic->LLM, LLM->Symbolic, and LLM+Symbolic. Finally, we discuss several key challenges and promising future directions. We have also released a GitHub repository including papers and resources related to this survey: this https URL. 

---
# MHSNet:An MoE-based Hierarchical Semantic Representation Network for Accurate Duplicate Resume Detection with Large Language Model 

**Authors**: Yu Li, Zulong Chen, Wenjian Xu, Hong Wen, Yipeng Yu, Man Lung Yiu, Yuyu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.13676)  

**Abstract**: To maintain the company's talent pool, recruiters need to continuously search for resumes from third-party websites (e.g., LinkedIn, Indeed). However, fetched resumes are often incomplete and inaccurate. To improve the quality of third-party resumes and enrich the company's talent pool, it is essential to conduct duplication detection between the fetched resumes and those already in the company's talent pool. Such duplication detection is challenging due to the semantic complexity, structural heterogeneity, and information incompleteness of resume texts. To this end, we propose MHSNet, an multi-level identity verification framework that fine-tunes BGE-M3 using contrastive learning. With the fine-tuned , Mixture-of-Experts (MoE) generates multi-level sparse and dense representations for resumes, enabling the computation of corresponding multi-level semantic similarities. Moreover, the state-aware Mixture-of-Experts (MoE) is employed in MHSNet to handle diverse incomplete resumes. Experimental results verify the effectiveness of MHSNet 

---
# Knowledge Graph Completion for Action Prediction on Situational Graphs -- A Case Study on Household Tasks 

**Authors**: Mariam Arustashvili, Jörg Deigmöller, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.13675)  

**Abstract**: Knowledge Graphs are used for various purposes, including business applications, biomedical analyses, or digital twins in industry 4.0. In this paper, we investigate knowledge graphs describing household actions, which are beneficial for controlling household robots and analyzing video footage. In the latter case, the information extracted from videos is notoriously incomplete, and completing the knowledge graph for enhancing the situational picture is essential. In this paper, we show that, while a standard link prediction problem, situational knowledge graphs have special characteristics that render many link prediction algorithms not fit for the job, and unable to outperform even simple baselines. 

---
# ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings 

**Authors**: Rehan Raza, Guanjin Wang, Kevin Wong, Hamid Laga, Marco Fisichella  

**Link**: [PDF](https://arxiv.org/pdf/2508.13672)  

**Abstract**: Explainable Artificial Intelligence (XAI) methods, such as Local Interpretable Model-Agnostic Explanations (LIME), have advanced the interpretability of black-box machine learning models by approximating their behavior locally using interpretable surrogate models. However, LIME's inherent randomness in perturbation and sampling can lead to locality and instability issues, especially in scenarios with limited training data. In such cases, data scarcity can result in the generation of unrealistic variations and samples that deviate from the true data manifold. Consequently, the surrogate model may fail to accurately approximate the complex decision boundary of the original model. To address these challenges, we propose a novel Instance-based Transfer Learning LIME framework (ITL-LIME) that enhances explanation fidelity and stability in data-constrained environments. ITL-LIME introduces instance transfer learning into the LIME framework by leveraging relevant real instances from a related source domain to aid the explanation process in the target domain. Specifically, we employ clustering to partition the source domain into clusters with representative prototypes. Instead of generating random perturbations, our method retrieves pertinent real source instances from the source cluster whose prototype is most similar to the target instance. These are then combined with the target instance's neighboring real instances. To define a compact locality, we further construct a contrastive learning-based encoder as a weighting mechanism to assign weights to the instances from the combined set based on their proximity to the target instance. Finally, these weighted source and target instances are used to train the surrogate model for explanation purposes. 

---
# Interactive Query Answering on Knowledge Graphs with Soft Entity Constraints 

**Authors**: Daniel Daza, Alberto Bernardi, Luca Costabello, Christophe Gueret, Masoud Mansoury, Michael Cochez, Martijn Schut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13663)  

**Abstract**: Methods for query answering over incomplete knowledge graphs retrieve entities that are likely to be answers, which is particularly useful when such answers cannot be reached by direct graph traversal due to missing edges. However, existing approaches have focused on queries formalized using first-order-logic. In practice, many real-world queries involve constraints that are inherently vague or context-dependent, such as preferences for attributes or related categories. Addressing this gap, we introduce the problem of query answering with soft constraints. We propose a Neural Query Reranker (NQR) designed to adjust query answer scores by incorporating soft constraints without disrupting the original answers to a query. NQR operates interactively, refining answers based on incremental examples of preferred and non-preferred entities. We extend existing QA benchmarks by generating datasets with soft constraints. Our experiments demonstrate that NQR can capture soft constraints while maintaining robust query answering performance. 

---
# V2P: From Background Suppression to Center Peaking for Robust GUI Grounding Task 

**Authors**: Jikai Chen, Long Chen, Dong Wang, Leilei Gan, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13634)  

**Abstract**: Precise localization of GUI elements is crucial for the development of GUI agents. Traditional methods rely on bounding box or center-point regression, neglecting spatial interaction uncertainty and visual-semantic hierarchies. Recent methods incorporate attention mechanisms but still face two key issues: (1) ignoring processing background regions causes attention drift from the desired area, and (2) uniform labeling fails to distinguish between center and edges of the target UI element, leading to click imprecision. Inspired by how humans visually process and interact with GUI elements, we propose the Valley-to-Peak (V2P) method to address these issues. To mitigate background distractions, V2P introduces a suppression attention mechanism that minimizes the model's focus on irrelevant regions to highlight the intended region. For the issue of center-edge distinction, V2P applies a Fitts' Law-inspired approach by modeling GUI interactions as 2D Gaussian heatmaps where the weight gradually decreases from the center towards the edges. The weight distribution follows a Gaussian function, with the variance determined by the target's size. Consequently, V2P effectively isolates the target area and teaches the model to concentrate on the most essential point of the UI element. The model trained by V2P achieves the performance with 92.3% and 50.5% on two benchmarks ScreenSpot-v2 and ScreenSpot-Pro. Ablations further confirm each component's contribution, highlighting V2P's generalizability for precise GUI grounding tasks. 

---
# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation 

**Authors**: Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.13587)  

**Abstract**: While reinforcement learning (RL) has proven highly effective for general reasoning in vision-language models, its application to tasks requiring in-depth understanding of information-rich images and generation of structured outputs remains underexplored. Chart-to-code generation exemplifies this challenge, demanding complex reasoning over visual charts to generate structured code. Supervised fine-tuning (SFT) alone is often insufficient, highlighting the need for effective RL strategies that appropriately reward structured outputs. We systematically investigate the performance plateau in SFT through large-scale experiments and propose Multimodal Structured Reinforcement Learning (MSRL) for chart-to-code generation, which substantially breaks through this plateau. We construct the largest training corpus to date, containing 3 million chart-code pairs from real-world arXiv tables to mitigate simplistic patterns of prior synthetic data. Despite reaching state-of-the-art performance, our experiments show that scaling SFT data eventually hits a plateau where further increases yield negligible improvements. Our MSRL method leverages a multi-granularity structured reward system using multimodal textual and visual feedback. At the textual level, rule-based rewards validate fine-grained code details. At the visual level, model-based rewards assess structural similarity by rendering generated code into images and employing an evaluator model. We implement this within a two-stage curriculum for training stability. Results demonstrate that MSRL significantly breaks the SFT plateau, improving high-level metrics by 6.2% and 9.9% on ChartMimic and ReachQA benchmarks respectively, achieving competitive performance with advanced closed-source models. 

---
# Toward Better EHR Reasoning in LLMs: Reinforcement Learning with Expert Attention Guidance 

**Authors**: Yue Fang, Yuxin Guo, Jiaran Gao, Hongxin Ding, Xinke Jiang, Weibin Liao, Yongxin Xu, Yinghao Zhu, Zhibang Yang, Liantao Ma, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13579)  

**Abstract**: Improving large language models (LLMs) for electronic health record (EHR) reasoning is essential for enabling accurate and generalizable clinical predictions. While LLMs excel at medical text understanding, they underperform on EHR-based prediction tasks due to challenges in modeling temporally structured, high-dimensional data. Existing approaches often rely on hybrid paradigms, where LLMs serve merely as frozen prior retrievers while downstream deep learning (DL) models handle prediction, failing to improve the LLM's intrinsic reasoning capacity and inheriting the generalization limitations of DL models. To this end, we propose EAG-RL, a novel two-stage training framework designed to intrinsically enhance LLMs' EHR reasoning ability through expert attention guidance, where expert EHR models refer to task-specific DL models trained on EHR data. Concretely, EAG-RL first constructs high-quality, stepwise reasoning trajectories using expert-guided Monte Carlo Tree Search to effectively initialize the LLM's policy. Then, EAG-RL further optimizes the policy via reinforcement learning by aligning the LLM's attention with clinically salient features identified by expert EHR models. Extensive experiments on two real-world EHR datasets show that EAG-RL improves the intrinsic EHR reasoning ability of LLMs by an average of 14.62%, while also enhancing robustness to feature perturbations and generalization to unseen clinical domains. These results demonstrate the practical potential of EAG-RL for real-world deployment in clinical prediction tasks. Our code have been available at this https URL. 

---
# CrafterDojo: A Suite of Foundation Models for Building Open-Ended Embodied Agents in Crafter 

**Authors**: Junyeong Park, Hyeonseo Cho, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.13530)  

**Abstract**: Developing general-purpose embodied agents is a core challenge in AI. Minecraft provides rich complexity and internet-scale data, but its slow speed and engineering overhead make it unsuitable for rapid prototyping. Crafter offers a lightweight alternative that retains key challenges from Minecraft, yet its use has remained limited to narrow tasks due to the absence of foundation models that have driven progress in the Minecraft setting. In this paper, we present CrafterDojo, a suite of foundation models and tools that unlock the Crafter environment as a lightweight, prototyping-friendly, and Minecraft-like testbed for general-purpose embodied agent research. CrafterDojo addresses this by introducing CrafterVPT, CrafterCLIP, and CrafterSteve-1 for behavior priors, vision-language grounding, and instruction following, respectively. In addition, we provide toolkits for generating behavior and caption datasets (CrafterPlay and CrafterCaption), reference agent implementations, benchmark evaluations, and a complete open-source codebase. 

---
# LM Agents May Fail to Act on Their Own Risk Knowledge 

**Authors**: Yuzhi Tang, Tianxiao Li, Elizabeth Li, Chris J. Maddison, Honghua Dong, Yangjun Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13465)  

**Abstract**: Language model (LM) agents have demonstrated significant potential for automating real-world tasks, yet they pose a diverse array of potential, severe risks in safety-critical scenarios. In this work, we identify a significant gap between LM agents' risk awareness and safety execution abilities: while they often answer "Yes" to queries like "Is executing `sudo rm -rf /*' dangerous?", they will likely fail to identify such risks in instantiated trajectories or even directly perform these risky actions when acting as agents. To systematically investigate this, we develop a comprehensive evaluation framework to examine agents' safety across three progressive dimensions: 1) their knowledge about potential risks, 2) their ability to identify corresponding risks in execution trajectories, and 3) their actual behaviors to avoid executing these risky actions. Our evaluation reveals two critical performance gaps that resemble the generator-validator gaps observed in LMs: while agents demonstrate near-perfect risk knowledge ($>98\%$ pass rates), they fail to apply this knowledge when identifying risks in actual scenarios (with performance dropping by $>23\%$) and often still execute risky actions ($<26\%$ pass rates). Notably, this trend persists across more capable LMs as well as in specialized reasoning models like DeepSeek-R1, indicating that simply scaling model capabilities or inference compute does not inherently resolve safety concerns. Instead, we take advantage of these observed gaps to develop a risk verifier that independently critiques the proposed actions by agents, with an abstractor that converts specific execution trajectories into abstract descriptions where LMs can more effectively identify the risks. Our overall system achieves a significant reduction of risky action execution by $55.3\%$ over vanilla-prompted agents. 

---
# Discrete Optimization of Min-Max Violation and its Applications Across Computational Sciences 

**Authors**: Cheikh Ahmed, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13437)  

**Abstract**: We introduce the Discrete Min-Max Violation (DMMV) as a general optimization problem which seeks an assignment of discrete values to variables that minimizes the largest constraint violation. This context-free mathematical formulation is applicable to a wide range of use cases that have worst-case performance requirements. After defining the DMMV problem mathematically, we explore its properties to establish a foundational understanding. To tackle DMMV instance sizes of practical relevance, we develop a GPU-accelerated heuristic that takes advantage of the mathematical properties of DMMV for speeding up the solution process. We demonstrate the versatile applicability of our heuristic by solving three optimization problems as use cases: (1) post-training quantization of language models, (2) discrete tomography, and (3) Finite Impulse Response (FIR) filter design. In quantization without outlier separation, our heuristic achieves 14% improvement on average over existing methods. In discrete tomography, it reduces reconstruction error by 16% under uniform noise and accelerates computations by a factor of 6 on GPU. For FIR filter design, it nearly achieves 50% ripple reduction compared to using the commercial integer optimization solver, Gurobi. Our comparative results point to the benefits of studying DMMV as a context-free optimization problem and the advantages that our proposed heuristic offers on three distinct problems. Our GPU-accelerated heuristic will be made open-source to further stimulate research on DMMV and its other applications. The code is available at this https URL 

---
# STPFormer: A State-of-the-Art Pattern-Aware Spatio-Temporal Transformer for Traffic Forecasting 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13433)  

**Abstract**: Spatio-temporal traffic forecasting is challenging due to complex temporal patterns, dynamic spatial structures, and diverse input formats. Although Transformer-based models offer strong global modeling, they often struggle with rigid temporal encoding and weak space-time fusion. We propose STPFormer, a Spatio-Temporal Pattern-Aware Transformer that achieves state-of-the-art performance via unified and interpretable representation learning. It integrates four modules: Temporal Position Aggregator (TPA) for pattern-aware temporal encoding, Spatial Sequence Aggregator (SSA) for sequential spatial learning, Spatial-Temporal Graph Matching (STGM) for cross-domain alignment, and an Attention Mixer for multi-scale fusion. Experiments on five real-world datasets show that STPFormer consistently sets new SOTA results, with ablation and visualizations confirming its effectiveness and generalizability. 

---
# Virtuous Machines: Towards Artificial General Science 

**Authors**: Gabrielle Wehr, Reuben Rideaux, Amaya J. Fox, David R. Lightfoot, Jason Tangen, Jason B. Mattingley, Shane E. Ehrhardt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13421)  

**Abstract**: Artificial intelligence systems are transforming scientific discovery by accelerating specific research tasks, from protein structure prediction to materials design, yet remain confined to narrow domains requiring substantial human oversight. The exponential growth of scientific literature and increasing domain specialisation constrain researchers' capacity to synthesise knowledge across disciplines and develop unifying theories, motivating exploration of more general-purpose AI systems for science. Here we show that a domain-agnostic, agentic AI system can independently navigate the scientific workflow - from hypothesis generation through data collection to manuscript preparation. The system autonomously designed and executed three psychological studies on visual working memory, mental rotation, and imagery vividness, executed one new online data collection with 288 participants, developed analysis pipelines through 8-hour+ continuous coding sessions, and produced completed manuscripts. The results demonstrate the capability of AI scientific discovery pipelines to conduct non-trivial research with theoretical reasoning and methodological rigour comparable to experienced researchers, though with limitations in conceptual nuance and theoretical interpretation. This is a step toward embodied AI that can test hypotheses through real-world experiments, accelerating discovery by autonomously exploring regions of scientific space that human cognitive and resource constraints might otherwise leave unexplored. It raises important questions about the nature of scientific understanding and the attribution of scientific credit. 

---
# TASER: Table Agents for Schema-guided Extraction and Recommendation 

**Authors**: Nicole Cho, Kirsty Fielding, William Watson, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.13404)  

**Abstract**: Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables. 

---
# SPANER: Shared Prompt Aligner for Multimodal Semantic Representation 

**Authors**: Thye Shan Ng, Caren Soyeon Han, Eun-Jung Holden  

**Link**: [PDF](https://arxiv.org/pdf/2508.13387)  

**Abstract**: Recent advances in multimodal Parameter-Efficient Fine-Tuning (PEFT) have significantly improved performance on downstream tasks such as few-shot retrieval. However, most existing approaches focus on task-specific gains while neglecting the structure of the multimodal embedding space. As a result, modality-specific representations often remain isolated, limiting cross-modal generalisation. In this work, we introduce Shared Prompt AligNER (SPANER), a modality-agnostic PEFT framework designed to embed inputs from diverse modalities into a unified semantic space. At its core, SPANER employs a shared prompt mechanism that acts as a conceptual anchor, enabling semantically related instances to converge spatially regardless of modality. This shared prompt design is inherently extensible, supporting the seamless integration of additional modalities, such as audio, without altering the core architecture. Through comprehensive experiments across vision-language and audio-visual benchmarks, SPANER demonstrates competitive few-shot retrieval performance while preserving high semantic coherence in the learned embedding space. Our results highlight the importance of aligning embedding structures, rather than merely tuning adapter weights, for scalable multimodal learning. 

---
# LOOP: A Plug-and-Play Neuro-Symbolic Framework for Enhancing Planning in Autonomous Systems 

**Authors**: Ronit Virwani, Ruchika Suryawanshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13371)  

**Abstract**: Planning is one of the most critical tasks in autonomous systems, where even a small error can lead to major failures or million-dollar losses. Current state-of-the-art neural planning approaches struggle with complex domains, producing plans with missing preconditions, inconsistent goals, and hallucinations. While classical planners provide logical guarantees, they lack the flexibility and natural language understanding capabilities needed for modern autonomous systems. Existing neuro-symbolic approaches use one-shot translation from natural language to formal plans, missing the opportunity for neural and symbolic components to work and refine solutions together. To address this gap, we develop LOOP -- a novel neuro-symbolic planning framework that treats planning as an iterative conversation between neural and symbolic components rather than simple translation. LOOP integrates 13 coordinated neural features including graph neural networks for spatial relationships, multi-agent validation for consensus-based correctness, hierarchical decomposition for complex task management, and causal memory that learns from both successes and failures. Unlike existing approaches, LOOP generates PDDL specifications, refines them iteratively based on symbolic feedback, and builds a causal knowledge base from execution traces. LOOP was evaluated on six standard IPC benchmark domains, where it achieved 85.8% success rate compared to LLM+P (55.0%), LLM-as-Planner (19.2%), and Tree-of-Thoughts (3.3%). This work shows that the key to reliable planning is not in choosing between neural networks or symbolic reasoners but it lies in making them actually ``talk'' to each other during the entire process. LOOP provides a thorough blueprint for building autonomous systems that can finally be trusted with critical real-world applications. 

---
# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design 

**Authors**: Chentong Chen, Mengyuan Zhong, Jianyong Sun, Ye Fan, Jialong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13333)  

**Abstract**: LLM-based Automatic Heuristic Design (AHD) within Evolutionary Computation (EC) frameworks has shown promising results. However, its effectiveness is hindered by the use of static operators and the lack of knowledge accumulation mechanisms. We introduce HiFo-Prompt, a framework that guides LLMs with two synergistic prompting strategies: Foresight and Hindsight. Foresight-based prompts adaptively steer the search based on population dynamics, managing the exploration-exploitation trade-off. In addition, hindsight-based prompts mimic human expertise by distilling successful heuristics from past generations into fundamental, reusable design principles. This dual mechanism transforms transient discoveries into a persistent knowledge base, enabling the LLM to learn from its own experience. Empirical results demonstrate that HiFo-Prompt significantly outperforms state-of-the-art LLM-based AHD methods, generating higher-quality heuristics while achieving substantially faster convergence and superior query efficiency. 

---
# Towards Unified Multimodal Financial Forecasting: Integrating Sentiment Embeddings and Market Indicators via Cross-Modal Attention 

**Authors**: Sarthak Khanna, Armin Berger, David Berghaus, Tobias Deusser, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13327)  

**Abstract**: We propose STONK (Stock Optimization using News Knowledge), a multimodal framework integrating numerical market indicators with sentiment-enriched news embeddings to improve daily stock-movement prediction. By combining numerical & textual embeddings via feature concatenation and cross-modal attention, our unified pipeline addresses limitations of isolated analyses. Backtesting shows STONK outperforms numeric-only baselines. A comprehensive evaluation of fusion strategies and model configurations offers evidence-based guidance for scalable multimodal financial forecasting. Source code is available on GitHub 

---
# CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support 

**Authors**: Yuting Zhang, Karina V. Bunting, Asgher Champsi, Xiaoxia Wang, Wenqi Lu, Alexander Thorley, Sandeep S Hothi, Zhaowen Qiu, Dipak Kotecha, Jinming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13256)  

**Abstract**: Cardiovascular diseases (CVDs) remain the foremost cause of mortality worldwide, a burden worsened by a severe deficit of healthcare workers. Artificial intelligence (AI) agents have shown potential to alleviate this gap via automated early detection and proactive screening, yet their clinical application remains limited by: 1) prompt-based clinical role assignment that relies on intrinsic model capabilities without domain-specific tool support; or 2) rigid sequential workflows, whereas clinical care often requires adaptive reasoning that orders specific tests and, based on their results, guides personalised next steps; 3) general and static knowledge bases without continuous learning capability; and 4) fixed unimodal or bimodal inputs and lack of on-demand visual outputs when further clarification is needed. In response, a multimodal framework, CardAIc-Agents, was proposed to augment models with external tools and adaptively support diverse cardiac tasks. Specifically, a CardiacRAG agent generated general plans from updatable cardiac knowledge, while the chief agent integrated tools to autonomously execute these plans and deliver decisions. To enable adaptive and case-specific customization, a stepwise update strategy was proposed to dynamically refine plans based on preceding execution results, once the task was assessed as complex. In addition, a multidisciplinary discussion tool was introduced to interpret challenging cases, thereby supporting further adaptation. When clinicians raised concerns, visual review panels were provided to assist final validation. Experiments across three datasets showed the efficiency of CardAIc-Agents compared to mainstream Vision-Language Models (VLMs), state-of-the-art agentic systems, and fine-tuned VLMs. 

---
# "DIVE" into Hydrogen Storage Materials Discovery with AI Agents 

**Authors**: Di Zhang, Xue Jia, Tran Ba Hung, Seong Hoon Jang, Linda Zhang, Ryuhei Sato, Yusuke Hashimoto, Toyoto Sato, Kiyoe Konno, Shin-ichi Orimo, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13251)  

**Abstract**: Data-driven artificial intelligence (AI) approaches are fundamentally transforming the discovery of new materials. Despite the unprecedented availability of materials data in the scientific literature, much of this information remains trapped in unstructured figures and tables, hindering the construction of large language model (LLM)-based AI agent for automated materials design. Here, we present the Descriptive Interpretation of Visual Expression (DIVE) multi-agent workflow, which systematically reads and organizes experimental data from graphical elements in scientific literatures. We focus on solid-state hydrogen storage materials-a class of materials central to future clean-energy technologies and demonstrate that DIVE markedly improves the accuracy and coverage of data extraction compared to the direct extraction by multimodal models, with gains of 10-15% over commercial models and over 30% relative to open-source models. Building on a curated database of over 30,000 entries from 4,000 publications, we establish a rapid inverse design workflow capable of identifying previously unreported hydrogen storage compositions in two minutes. The proposed AI workflow and agent design are broadly transferable across diverse materials, providing a paradigm for AI-driven materials discovery. 

---
# Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information 

**Authors**: Zeyu Zhang, Yang Zhang, Haoran Tan, Rui Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13250)  

**Abstract**: In large language model-based agents, memory serves as a critical capability for achieving personalization by storing and utilizing users' information. Although some previous studies have adopted memory to implement user personalization, they typically focus on preference alignment and simple question-answering. However, in the real world, complex tasks often require multi-hop reasoning on a large amount of user information, which poses significant challenges for current memory approaches. To address this limitation, we propose the multi-hop personalized reasoning task to explore how different memory mechanisms perform in multi-hop reasoning over personalized information. We explicitly define this task and construct a dataset along with a unified evaluation framework. Then, we implement various explicit and implicit memory methods and conduct comprehensive experiments. We evaluate their performance on this task from multiple perspectives and analyze their strengths and weaknesses. Besides, we explore hybrid approaches that combine both paradigms and propose the HybridMem method to address their limitations. We demonstrate the effectiveness of our proposed model through extensive experiments. To benefit the research community, we release this project at this https URL. 

---
# AI sustains higher strategic tension than humans in chess 

**Authors**: Adamo Cerioli, Edward D. Lee, Vito D. P. Servedio  

**Link**: [PDF](https://arxiv.org/pdf/2508.13213)  

**Abstract**: Strategic decision-making involves managing the tension between immediate opportunities and long-term objectives. We study this trade-off in chess by characterizing and comparing dynamics between human vs human and AI vs AI games. We propose a network-based metric of piece-to-piece interaction to quantify the ongoing strategic tension on the board. Its evolution in games reveals that the most competitive AI players sustain higher levels of strategic tension for longer durations than elite human players. Cumulative tension varies with algorithmic complexity for AI and correspondingly in human-played games increases abruptly with expertise at about 1600 Elo and again at 2300 Elo. The profiles reveal different approaches. Highly competitive AI tolerates interconnected positions balanced between offensive and defensive tactics over long periods. Human play, in contrast, limits tension and game complexity, which may reflect cognitive limitations and adaptive strategies. The difference may have implications for AI usage in complex, strategic environments. 

---
# QuickMerge++: Fast Token Merging with Autoregressive Prior 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13204)  

**Abstract**: As generative models scale to larger inputs across language, vision, and video domains, the cost of token-level computation has become a key bottleneck. While prior work suggests that only a subset of tokens significantly influence downstream predictions, most token selection methods are static, modality-specific, or incompatible with autoregressive generation. In this paper, we propose QuickMerge, a lightweight token merging framework designed for efficient next-token prediction.
QuickMerge dynamically selects a reduced number of tokens based on attention norm magnitude, guided by an entropy-based budget estimator. To preserve autoregressive compatibility, we introduce a lightweight transformer prior trained over the merged token sequence. By combining semantic salience estimation, flexible token budgets, and AR alignment, QuickMerge enables accurate generation with fewer tokens.
We evaluate QuickMerge across multi-modality domains, demonstrating consistent improvements in compute-accuracy tradeoffs. Specifically, QuickMerge reduces token counts sustantially while matching as well as exceeding the performance of learned tokenizers and fixed-patch baselines. 

---
# Search-Time Data Contamination 

**Authors**: Ziwen Han, Meher Mankikar, Julian Michael, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13180)  

**Abstract**: Data contamination refers to the leakage of evaluation data into model training data, resulting in overfitting to supposedly held-out test sets and compromising test validity. We identify an analogous issue, search-time contamination (STC), in evaluating search-based LLM agents which use tools to gather information from online sources when answering user queries. STC occurs when the retrieval step surfaces a source containing the test question (or a near-duplicate) alongside its answer, enabling agents to copy rather than genuinely infer or reason, undermining benchmark integrity. We find that HuggingFace, an online platform hosting evaluation datasets, appears among retrieved sources in search based agent logs. Consequently, agents often explicitly acknowledge discovering question answer pairs from HuggingFace within their reasoning chains. On three commonly used capability benchmarks: Humanity's Last Exam (HLE), SimpleQA, and GPQA, we demonstrate that for approximately 3% of questions, search-based agents directly find the datasets with ground truth labels on HuggingFace. When millions of evaluation queries target the same benchmark, even small, repeated leaks can accelerate the benchmark's obsolescence, shortening its intended lifecycle. After HuggingFace is blocked, we observe a drop in accuracy on the contaminated subset of approximately 15%. We further show through ablation experiments that publicly accessible evaluation datasets on HuggingFace may not be the sole source of STC. To this end, we conclude by proposing best practices for benchmark design and result reporting to address this novel form of leakage and ensure trustworthy evaluation of search-based LLM agents. To facilitate the auditing of evaluation results, we also publicly release the complete logs from our experiments. 

---
# The Interpretability Analysis of the Model Can Bring Improvements to the Text-to-SQL Task 

**Authors**: Cong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13178)  

**Abstract**: To elevate the foundational capabilities and generalization prowess of the text-to-SQL model in real-world applications, we integrate model interpretability analysis with execution-guided strategy for semantic parsing of WHERE clauses in SQL queries. Furthermore, we augment this approach with filtering adjustments, logical correlation refinements, and model fusion, culminating in the design of the CESQL model that facilitates conditional enhancement. Our model excels on the WikiSQL dataset, which is emblematic of single-table database query tasks, markedly boosting the accuracy of prediction outcomes. When predicting conditional values in WHERE clauses, we have not only minimized our dependence on data within the condition columns of tables but also circumvented the impact of manually labeled training data. Our hope is that this endeavor to enhance accuracy in processing basic database queries will offer fresh perspectives for research into handling complex queries and scenarios featuring irregular data in real-world database environments. 

---
# A Hardware-oriented Approach for Efficient Active Inference Computation and Deployment 

**Authors**: Nikola Pižurica, Nikola Milović, Igor Jovančević, Conor Heins, Miguel de Prado  

**Link**: [PDF](https://arxiv.org/pdf/2508.13177)  

**Abstract**: Active Inference (AIF) offers a robust framework for decision-making, yet its computational and memory demands pose challenges for deployment, especially in resource-constrained environments. This work presents a methodology that facilitates AIF's deployment by integrating pymdp's flexibility and efficiency with a unified, sparse, computational graph tailored for hardware-efficient execution. Our approach reduces latency by over 2x and memory by up to 35%, advancing the deployment of efficient AIF agents for real-time and embedded applications. 

---
# Fitting Ontologies and Constraints to Relational Structures 

**Authors**: Simon Hosemann, Jean Christoph Jung, Carsten Lutz, Sebastian Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2508.13176)  

**Abstract**: We study the problem of fitting ontologies and constraints to positive and negative examples that take the form of a finite relational structure. As ontology and constraint languages, we consider the description logics $\mathcal{E\mkern-2mu L}$ and $\mathcal{E\mkern-2mu LI}$ as well as several classes of tuple-generating dependencies (TGDs): full, guarded, frontier-guarded, frontier-one, and unrestricted TGDs as well as inclusion dependencies. We pinpoint the exact computational complexity, design algorithms, and analyze the size of fitting ontologies and TGDs. We also investigate the related problem of constructing a finite basis of concept inclusions / TGDs for a given set of finite structures. While finite bases exist for $\mathcal{E\mkern-2mu L}$, $\mathcal{E\mkern-2mu LI}$, guarded TGDs, and inclusion dependencies, they in general do not exist for full, frontier-guarded and frontier-one TGDs. 

---
# AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining 

**Authors**: Hongjun Ding, Binqi Chen, Jinsheng Huang, Taian Guo, Zhengyang Mao, Guoyi Shao, Lutong Zou, Luchen Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13174)  

**Abstract**: Formula alpha mining, which generates predictive signals from financial data, is critical for quantitative investment. Although various algorithmic approaches-such as genetic programming, reinforcement learning, and large language models-have significantly expanded the capacity for alpha discovery, systematic evaluation remains a key challenge. Existing evaluation metrics predominantly include backtesting and correlation-based measures. Backtesting is computationally intensive, inherently sequential, and sensitive to specific strategy parameters. Correlation-based metrics, though efficient, assess only predictive ability and overlook other crucial properties such as temporal stability, robustness, diversity, and interpretability. Additionally, the closed-source nature of most existing alpha mining models hinders reproducibility and slows progress in this field. To address these issues, we propose AlphaEval, a unified, parallelizable, and backtest-free evaluation framework for automated alpha mining models. AlphaEval assesses the overall quality of generated alphas along five complementary dimensions: predictive power, stability, robustness to market perturbations, financial logic, and diversity. Extensive experiments across representative alpha mining algorithms demonstrate that AlphaEval achieves evaluation consistency comparable to comprehensive backtesting, while providing more comprehensive insights and higher efficiency. Furthermore, AlphaEval effectively identifies superior alphas compared to traditional single-metric screening approaches. All implementations and evaluation tools are open-sourced to promote reproducibility and community engagement. 

---
# Cognitive Workspace: Active Memory Management for LLMs -- An Empirical Study of Functional Infinite Context 

**Authors**: Tao An  

**Link**: [PDF](https://arxiv.org/pdf/2508.13171)  

**Abstract**: Large Language Models (LLMs) face fundamental limitations in context management despite recent advances extending context windows to millions of tokens. We propose Cognitive Workspace, a novel paradigm that transcends traditional Retrieval-Augmented Generation (RAG) by emulating human cognitive mechanisms of external memory use. Drawing from cognitive science foundations including Baddeley's working memory model, Clark's extended mind thesis, and Hutchins' distributed cognition framework, we demonstrate that current passive retrieval systems fail to capture the dynamic, task-driven nature of human memory management. Our analysis of 2024-2025 developments reveals that while techniques like Infini-attention and StreamingLLM achieve impressive context lengths, they lack the metacognitive awareness and active planning capabilities essential for true cognitive extension. Cognitive Workspace addresses these limitations through three core innovations: (1) active memory management with deliberate information curation, (2) hierarchical cognitive buffers enabling persistent working states, and (3) task-driven context optimization that dynamically adapts to cognitive demands. Empirical validation demonstrates Cognitive Workspace achieves an average 58.6% memory reuse rate (ranging from 54-60% across different tasks) compared to 0% for traditional RAG, with 17-18% net efficiency gain despite 3.3x higher operation counts. Statistical analysis confirms these advantages with p < 0.001 and Cohen's d > 23 across multiple task types, establishing the first quantitative evidence for active memory superiority in LLM systems. We present a comprehensive theoretical framework synthesizing insights from 50+ recent papers, positioning Cognitive Workspace as a fundamental shift from information retrieval to genuine cognitive augmentation. 

---
# Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL 

**Authors**: Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tiannan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13167)  

**Abstract**: Recent advances in large language models (LLMs) and multi-agent systems have demonstrated remarkable capabilities in complex problem-solving tasks such as deep research, vibe coding, and mathematical reasoning. However, most existing multi-agent systems are built upon manual prompt/workflow engineering with sophisticated agent frameworks, making them computationally inefficient, less capable, and can not benefit from data-centric learning. In this work, we introduce Chain-of-Agents (CoA), a novel paradigm of LLM reasoning that enables native end-to-end complex problem-solving in the same way as a multi-agent system (i.e., multi-turn problem solving with multiple tools and multiple agents) within one model. In chain-of-agents problem-solving, the model dynamically activates different tool agents and role-playing agents to simulate multi-agent collaboration in an end-to-end fashion. To elicit end-to-end chain-of-agents problem-solving abilities in LLMs, we introduce a multi-agent distillation framework to distill state-of-the-art multi-agent systems into chain-of-agents trajectories for agentic supervised fine-tuning. We then use agentic reinforcement learning on verifiable agentic tasks to further improve the models' capabilities on chain-of-agents problem solving. We call the resulting models Agent Foundation Models (AFMs). Our empirical studies demonstrate that AFM establishes new state-of-the-art performance across diverse benchmarks in both web agent and code agent settings. We make the entire research, including the model weights, code for training and evaluation, and the training data, fully open-sourced, which offers a solid starting point for future research on agent models and agentic RL. 

---
# GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation 

**Authors**: Ken Deng, Yunhan Yang, Jingxiang Sun, Xihui Liu, Yebin Liu, Ding Liang, Yan-Pei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14036)  

**Abstract**: Modern 3D generation methods can rapidly create shapes from sparse or single views, but their outputs often lack geometric detail due to computational constraints. We present DetailGen3D, a generative approach specifically designed to enhance these generated 3D shapes. Our key insight is to model the coarse-to-fine transformation directly through data-dependent flows in latent space, avoiding the computational overhead of large-scale 3D generative models. We introduce a token matching strategy that ensures accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure. By carefully designing our training data to match the characteristics of synthesized coarse shapes, our method can effectively enhance shapes produced by various 3D generation and reconstruction approaches, from single-view to sparse multi-view inputs. Extensive experiments demonstrate that DetailGen3D achieves high-fidelity geometric detail synthesis while maintaining efficiency in training. 

---
# Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation 

**Authors**: Dongyoon Hahm, Taywon Min, Woogyeol Jin, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14031)  

**Abstract**: Beyond simple text generation, Large Language Models (LLMs) have evolved into agentic systems capable of planning and interacting with external tools to solve complex tasks. This evolution involves fine-tuning LLMs on agent-specific tasks to enhance their proficiency. However, safety concerns are frequently overlooked during this fine-tuning process. In this work, we show that aligned LLMs can become unintentionally misaligned, leading to a higher likelihood of executing harmful tasks and a reduced tendency to refuse them when fine-tuned to execute agentic tasks. To address these safety challenges, we propose Prefix INjection Guard (PING), a simple yet effective method that prepends automatically generated natural language prefixes to agent responses, guiding them to refuse harmful requests while preserving performance on benign tasks. Specifically, we introduce an iterative approach that alternates between (1) generating candidate prefixes and (2) selecting those that optimize both task performance and refusal behavior. Experimental results demonstrate that PING significantly enhances the safety of fine-tuned LLM agents without sacrificing their effectiveness. PING consistently outperforms existing prompting approaches across diverse benchmarks in both web navigation and code generation tasks. Our analysis of internal hidden states via linear probes reveals that prefix tokens are crucial for behavior modification, explaining the performance gains. WARNING: This paper contains contents that are unethical or offensive in nature. 

---
# Ask Good Questions for Large Language Models 

**Authors**: Qi Wu, Zhongqi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14025)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the performance of dialog systems, yet current approaches often fail to provide accurate guidance of topic due to their inability to discern user confusion in related concepts. To address this, we introduce the Ask-Good-Question (AGQ) framework, which features an improved Concept-Enhanced Item Response Theory (CEIRT) model to better identify users' knowledge levels. Our contributions include applying the CEIRT model along with LLMs to directly generate guiding questions based on the inspiring text, greatly improving information retrieval efficiency during the question & answer process. Through comparisons with other baseline methods, our approach outperforms by significantly enhencing the users' information retrieval experiences. 

---
# Efficient Knowledge Graph Unlearning with Zeroth-order Information 

**Authors**: Yang Xiao, Ruimeng Ye, Bohan Liu, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2508.14013)  

**Abstract**: Due to regulations like the Right to be Forgotten, there is growing demand for removing training data and its influence from models. Since full retraining is costly, various machine unlearning methods have been proposed. In this paper, we firstly present an efficient knowledge graph (KG) unlearning algorithm. We remark that KG unlearning is nontrivial due to the distinctive structure of KG and the semantic relations between entities. Also, unlearning by estimating the influence of removed components incurs significant computational overhead when applied to large-scale knowledge graphs. To this end, we define an influence function for KG unlearning and propose to approximate the model's sensitivity without expensive computation of first-order and second-order derivatives for parameter updates. Specifically, we use Taylor expansion to estimate the parameter changes caused by data removal. Given that the first-order gradients and second-order derivatives dominate the computational load, we use the Fisher matrices and zeroth-order optimization to approximate the inverse-Hessian vector product without constructing the computational graphs. Our experimental results demonstrate that the proposed method outperforms other state-of-the-art graph unlearning baselines significantly in terms of unlearning efficiency and unlearning quality. Our code is released at this https URL. 

---
# Evaluating Identity Leakage in Speaker De-Identification Systems 

**Authors**: Seungmin Seo, Oleg Aulov, Afzal Godil, Kevin Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2508.14012)  

**Abstract**: Speaker de-identification aims to conceal a speaker's identity while preserving intelligibility of the underlying speech. We introduce a benchmark that quantifies residual identity leakage with three complementary error rates: equal error rate, cumulative match characteristic hit rate, and embedding-space similarity measured via canonical correlation analysis and Procrustes analysis. Evaluation results reveal that all state-of-the-art speaker de-identification systems leak identity information. The highest performing system in our evaluation performs only slightly better than random guessing, while the lowest performing system achieves a 45% hit rate within the top 50 candidates based on CMC. These findings highlight persistent privacy risks in current speaker de-identification technologies. 

---
# ASDFormer: A Transformer with Mixtures of Pooling-Classifier Experts for Robust Autism Diagnosis and Biomarker Discovery 

**Authors**: Mohammad Izadi, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.14005)  

**Abstract**: Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition marked by disruptions in brain connectivity. Functional MRI (fMRI) offers a non-invasive window into large-scale neural dynamics by measuring blood-oxygen-level-dependent (BOLD) signals across the brain. These signals can be modeled as interactions among Regions of Interest (ROIs), which are grouped into functional communities based on their underlying roles in brain function. Emerging evidence suggests that connectivity patterns within and between these communities are particularly sensitive to ASD-related alterations. Effectively capturing these patterns and identifying interactions that deviate from typical development is essential for improving ASD diagnosis and enabling biomarker discovery. In this work, we introduce ASDFormer, a Transformer-based architecture that incorporates a Mixture of Pooling-Classifier Experts (MoE) to capture neural signatures associated with ASD. By integrating multiple specialized expert branches with attention mechanisms, ASDFormer adaptively emphasizes different brain regions and connectivity patterns relevant to autism. This enables both improved classification performance and more interpretable identification of disorder-related biomarkers. Applied to the ABIDE dataset, ASDFormer achieves state-of-the-art diagnostic accuracy and reveals robust insights into functional connectivity disruptions linked to ASD, highlighting its potential as a tool for biomarker discovery. 

---
# Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation 

**Authors**: Yifu Yuan, Haiqin Cui, Yaoting Huang, Yibin Chen, Fei Ni, Zibin Dong, Pengyi Li, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13998)  

**Abstract**: Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics. 

---
# Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization 

**Authors**: Shaohua Duan, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13993)  

**Abstract**: Long-context modeling is critical for a wide range of real-world tasks, including long-context question answering, summarization, and complex reasoning tasks. Recent studies have explored fine-tuning Large Language Models (LLMs) with synthetic data to enhance their long-context capabilities. However, the effectiveness of such approaches is often limited by the low diversity and factual inconsistencies in the generated data. To address these challenges, we propose LongMab-PO, a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training. Specifically, we treat context chunks as arms of MAB, select chunks based on their expected reward scores to input into LLMs to generate responses, and iteratively update these scores based on reward feedback. This exploration and exploitation process enables the model to focus on the most relevant context segments, thereby generating and collecting high-quality and diverse responses. Finally, we collect these generated responses from the rollout process and apply the DPO method to further optimize the LLM. Experimental results show that LongMab-PO significantly improves the diversity and quality of preference data pairs, achieving state-of-the-art performance on long-context reasoning benchmarks. All code and data will be released on this https URL. 

---
# The Social Context of Human-Robot Interactions 

**Authors**: Sydney Thompson, Kate Candon, Marynel Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13982)  

**Abstract**: The Human-Robot Interaction (HRI) community often highlights the social context of an interaction as a key consideration when designing, implementing, and evaluating robot behavior. Unfortunately, researchers use the term "social context" in varied ways. This can lead to miscommunication, making it challenging to draw connections between related work on understanding and modeling the social contexts of human-robot interactions. To address this gap, we survey the HRI literature for existing definitions and uses of the term "social context". Then, we propose a conceptual model for describing the social context of a human-robot interaction. We apply this model to existing work, and we discuss a range of attributes of social contexts that can help researchers plan for interactions, develop behavior models for robots, and gain insights after interactions have taken place. We conclude with a discussion of open research questions in relation to understanding and modeling the social contexts of human-robot interactions. 

---
# RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation 

**Authors**: Tianyi Niu, Jaemin Cho, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13968)  

**Abstract**: We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0°, 90°, 180°, and 270°. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0°) images, while certain models are able to identify upside-down (180°) images. None can reliably distinguish between 90° and 270°. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90° and 270° rotations, despite substantially improving the identification of 180° images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation. 

---
# Learning to Use AI for Learning: How Can We Effectively Teach and Measure Prompting Literacy for K-12 Students? 

**Authors**: Ruiwei Xiao, Xinying Hou, Ying-Jui Tseng, Hsuan Nieu, Guanze Liao, John Stamper, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13962)  

**Abstract**: As Artificial Intelligence (AI) becomes increasingly integrated into daily life, there is a growing need to equip the next generation with the ability to apply, interact with, evaluate, and collaborate with AI systems responsibly. Prior research highlights the urgent demand from K-12 educators to teach students the ethical and effective use of AI for learning. To address this need, we designed an Large-Language Model (LLM)-based module to teach prompting literacy. This includes scenario-based deliberate practice activities with direct interaction with intelligent LLM agents, aiming to foster secondary school students' responsible engagement with AI chatbots. We conducted two iterations of classroom deployment in 11 authentic secondary education classrooms, and evaluated 1) AI-based auto-grader's capability; 2) students' prompting performance and confidence changes towards using AI for learning; and 3) the quality of learning and assessment materials. Results indicated that the AI-based auto-grader could grade student-written prompts with satisfactory quality. In addition, the instructional materials supported students in improving their prompting skills through practice and led to positive shifts in their perceptions of using AI for learning. Furthermore, data from Study 1 informed assessment revisions in Study 2. Analyses of item difficulty and discrimination in Study 2 showed that True/False and open-ended questions could measure prompting literacy more effectively than multiple-choice questions for our target learners. These promising outcomes highlight the potential for broader deployment and highlight the need for broader studies to assess learning effectiveness and assessment design. 

---
# A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources -- Extended Version 

**Authors**: Björn Filter, Ralf Möller, Özgür Lütfü Özçep  

**Link**: [PDF](https://arxiv.org/pdf/2508.13960)  

**Abstract**: The latest developments in AI focus on agentic systems where artificial and human agents cooperate to realize global goals. An example is collaborative learning, which aims to train a global model based on data from individual agents. A major challenge in designing such systems is to guarantee safety and alignment with human values, particularly a fair distribution of rewards upon achieving the global goal. Cooperative game theory offers useful abstractions of cooperating agents via value functions, which assign value to each coalition, and via reward functions. With these, the idea of fair allocation can be formalized by specifying fairness axioms and designing concrete mechanisms. Classical cooperative game theory, exemplified by the Shapley value, does not fully capture scenarios like collaborative learning, as it assumes nonreplicable resources, whereas data and models can be replicated. Infinite replicability requires a generalized notion of fairness, formalized through new axioms and mechanisms. These must address imbalances in reciprocal benefits among participants, which can lead to strategic exploitation and unfair allocations. The main contribution of this paper is a mechanism and a proof that it fulfills the property of mutual fairness, formalized by the Balanced Reciprocity Axiom. It ensures that, for every pair of players, each benefits equally from the participation of the other. 

---
# Prompt Orchestration Markup Language 

**Authors**: Yuge Zhang, Nan Chen, Jiahang Xu, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13948)  

**Abstract**: Large Language Models (LLMs) require sophisticated prompting, yet current practices face challenges in structure, data integration, format sensitivity, and tooling. Existing methods lack comprehensive solutions for organizing complex prompts involving diverse data types (documents, tables, images) or managing presentation variations systematically. To address these gaps, we introduce POML (Prompt Orchestration Markup Language). POML employs component-based markup for logical structure (roles, tasks, examples), specialized tags for seamless data integration, and a CSS-like styling system to decouple content from presentation, reducing formatting sensitivity. It includes templating for dynamic prompts and a comprehensive developer toolkit (IDE support, SDKs) to improve version control and collaboration. We validate POML through two case studies demonstrating its impact on complex application integration (PomLink) and accuracy performance (TableQA), as well as a user study assessing its effectiveness in real-world development scenarios. 

---
# InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems 

**Authors**: Matey Krastev, Miklos Hamar, Danilo Toapanta, Jesse Brouwers, Yibin Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.13930)  

**Abstract**: This work revisits and extends synthetic query generation pipelines for Neural Information Retrieval (NIR) by leveraging the InPars Toolkit, a reproducible, end-to-end framework for generating training data using large language models (LLMs). We first assess the reproducibility of the original InPars, InPars-V2, and Promptagator pipelines on the SciFact benchmark and validate their effectiveness using open-source reranker and generator models. Building on this foundation, we introduce two key extensions to the pipeline: (1) fine-tuning a query generator LLM via Contrastive Preference Optimization (CPO) to improve the signal quality in generated queries, and (2) replacing static prompt templates with dynamic, Chain-of-Thought (CoT) optimized prompts using the DSPy framework. Our results show that both extensions reduce the need for aggressive filtering while improving retrieval performance. All code, models, and synthetic datasets are publicly released to support further research at: \href{this https URL}{this https URL}. 

---
# Categorical Policies: Multimodal Policy Learning and Exploration in Continuous Control 

**Authors**: SM Mazharul Islam, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2508.13922)  

**Abstract**: A policy in deep reinforcement learning (RL), either deterministic or stochastic, is commonly parameterized as a Gaussian distribution alone, limiting the learned behavior to be unimodal. However, the nature of many practical decision-making problems favors a multimodal policy that facilitates robust exploration of the environment and thus to address learning challenges arising from sparse rewards, complex dynamics, or the need for strategic adaptation to varying contexts. This issue is exacerbated in continuous control domains where exploration usually takes place in the vicinity of the predicted optimal action, either through an additive Gaussian noise or the sampling process of a stochastic policy. In this paper, we introduce Categorical Policies to model multimodal behavior modes with an intermediate categorical distribution, and then generate output action that is conditioned on the sampled mode. We explore two sampling schemes that ensure differentiable discrete latent structure while maintaining efficient gradient-based optimization. By utilizing a latent categorical distribution to select the behavior mode, our approach naturally expresses multimodality while remaining fully differentiable via the sampling tricks. We evaluate our multimodal policy on a set of DeepMind Control Suite environments, demonstrating that through better exploration, our learned policies converge faster and outperform standard Gaussian policies. Our results indicate that the Categorical distribution serves as a powerful tool for structured exploration and multimodal behavior representation in continuous control. 

---
# Fisher-Orthogonal Projection Methods for Natural Gradient Descent with Large Batches 

**Authors**: Yishun Lu, Wesley Armour  

**Link**: [PDF](https://arxiv.org/pdf/2508.13898)  

**Abstract**: Modern GPUs are equipped with large amounts of high-bandwidth memory, enabling them to support mini-batch sizes of up to tens of thousands of training samples. However, most existing optimizers struggle to perform effectively at such a large batch size. As batch size increases, gradient noise decreases due to averaging over many samples, limiting the ability of first-order methods to escape sharp or suboptimal minima and reach the global minimum. Meanwhile, second-order methods like the natural gradient with Kronecker-Factored Approximate Curvature (KFAC) often require excessively high damping to remain stable at large batch sizes. This high damping effectively washes out the curvature information that gives these methods their advantage, reducing their performance to that of simple gradient descent. In this paper, we introduce Fisher-Orthogonal Projection (FOP), a novel technique that restores the effectiveness of the second-order method at very large batch sizes, enabling scalable training with improved generalization and faster convergence. FOP constructs a variance-aware update direction by leveraging gradients from two sub-batches, enhancing the average gradient with a component of the gradient difference that is orthogonal to the average under the Fisher-metric. 

---
# Toward Deployable Multi-Robot Collaboration via a Symbolically-Guided Decision Transformer 

**Authors**: Rathnam Vidushika Rasanji, Jin Wei-Kocsis, Jiansong Zhang, Dongming Gan, Ragu Athinarayanan, Paul Asunda  

**Link**: [PDF](https://arxiv.org/pdf/2508.13877)  

**Abstract**: Reinforcement learning (RL) has demonstrated great potential in robotic operations. However, its data-intensive nature and reliance on the Markov Decision Process (MDP) assumption limit its practical deployment in real-world scenarios involving complex dynamics and long-term temporal dependencies, such as multi-robot manipulation. Decision Transformers (DTs) have emerged as a promising offline alternative by leveraging causal transformers for sequence modeling in RL tasks. However, their applications to multi-robot manipulations still remain underexplored. To address this gap, we propose a novel framework, Symbolically-Guided Decision Transformer (SGDT), which integrates a neuro-symbolic mechanism with a causal transformer to enable deployable multi-robot collaboration. In the proposed SGDT framework, a neuro-symbolic planner generates a high-level task-oriented plan composed of symbolic subgoals. Guided by these subgoals, a goal-conditioned decision transformer (GCDT) performs low-level sequential decision-making for multi-robot manipulation. This hierarchical architecture enables structured, interpretable, and generalizable decision making in complex multi-robot collaboration tasks. We evaluate the performance of SGDT across a range of task scenarios, including zero-shot and few-shot scenarios. To our knowledge, this is the first work to explore DT-based technology for multi-robot manipulation. 

---
# A Novel Attention-Augmented Wavelet YOLO System for Real-time Brain Vessel Segmentation on Transcranial Color-coded Doppler 

**Authors**: Wenxuan Zhang, Shuai Li, Xinyi Wang, Yu Sun, Hongyu Kang, Pui Yuk Chryste Wan, Yong-Ping Zheng, Sai-Kit Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.13875)  

**Abstract**: The Circle of Willis (CoW), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color-coded Doppler (TCCD) offers unique advantages due to its radiation-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI-powered, real-time CoW auto-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention-Augmented Wavelet YOLO (AAW-YOLO) network tailored for TCCD data, designed to provide real-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high-quality dataset for model training and evaluation. The proposed AAW-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD-based cerebrovascular screening, with potential applications in routine clinical workflows and resource-constrained settings. Future research will explore bilateral modeling and larger-scale validation. 

---
# UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion 

**Authors**: Zihan Liang, Yufei Ma, ZhiPeng Qian, Huangyu Dai, Zihan Wang, Ben Chen, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13843)  

**Abstract**: Current e-commerce multimodal retrieval systems face two key limitations: they optimize for specific tasks with fixed modality pairings, and lack comprehensive benchmarks for evaluating unified retrieval approaches. To address these challenges, we introduce UniECS, a unified multimodal e-commerce search framework that handles all retrieval scenarios across image, text, and their combinations. Our work makes three key contributions. First, we propose a flexible architecture with a novel gated multimodal encoder that uses adaptive fusion mechanisms. This encoder integrates different modality representations while handling missing modalities. Second, we develop a comprehensive training strategy to optimize learning. It combines cross-modal alignment loss (CMAL), cohesive local alignment loss (CLAL), intra-modal contrastive loss (IMCL), and adaptive loss weighting. Third, we create M-BEER, a carefully curated multimodal benchmark containing 50K product pairs for e-commerce search evaluation. Extensive experiments demonstrate that UniECS consistently outperforms existing methods across four e-commerce benchmarks with fine-tuning or zero-shot evaluation. On our M-BEER bench, UniECS achieves substantial improvements in cross-modal tasks (up to 28\% gain in R@10 for text-to-image retrieval) while maintaining parameter efficiency (0.2B parameters) compared to larger models like GME-Qwen2VL (2B) and MM-Embed (8B). Furthermore, we deploy UniECS in the e-commerce search platform of Kuaishou Inc. across two search scenarios, achieving notable improvements in Click-Through Rate (+2.74\%) and Revenue (+8.33\%). The comprehensive evaluation demonstrates the effectiveness of our approach in both experimental and real-world settings. Corresponding codes, models and datasets will be made publicly available at this https URL. 

---
# One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression 

**Authors**: Mikołaj Janusz, Tomasz Wojnar, Yawei Li, Luca Benini, Kamil Adamczewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.13836)  

**Abstract**: Pruning is a core technique for compressing neural networks to improve computational efficiency. This process is typically approached in two ways: one-shot pruning, which involves a single pass of training and pruning, and iterative pruning, where pruning is performed over multiple cycles for potentially finer network refinement. Although iterative pruning has historically seen broader adoption, this preference is often assumed rather than rigorously tested. Our study presents one of the first systematic and comprehensive comparisons of these methods, providing rigorous definitions, benchmarking both across structured and unstructured settings, and applying different pruning criteria and modalities. We find that each method has specific advantages: one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios. Building on these findings, we advocate for patience-based pruning and introduce a hybrid approach that can outperform traditional methods in certain scenarios, providing valuable insights for practitioners selecting a pruning strategy tailored to their goals and constraints. Source code is available at this https URL. 

---
# Extracting Structured Requirements from Unstructured Building Technical Specifications for Building Information Modeling 

**Authors**: Insaf Nahri, Romain Pinquié, Philippe Véron, Nicolas Bus, Mathieu Thorel  

**Link**: [PDF](https://arxiv.org/pdf/2508.13833)  

**Abstract**: This study explores the integration of Building Information Modeling (BIM) with Natural Language Processing (NLP) to automate the extraction of requirements from unstructured French Building Technical Specification (BTS) documents within the construction industry. Employing Named Entity Recognition (NER) and Relation Extraction (RE) techniques, the study leverages the transformer-based model CamemBERT and applies transfer learning with the French language model Fr\_core\_news\_lg, both pre-trained on a large French corpus in the general domain. To benchmark these models, additional approaches ranging from rule-based to deep learning-based methods are developed. For RE, four different supervised models, including Random Forest, are implemented using a custom feature vector. A hand-crafted annotated dataset is used to compare the effectiveness of NER approaches and RE models. Results indicate that CamemBERT and Fr\_core\_news\_lg exhibited superior performance in NER, achieving F1-scores over 90\%, while Random Forest proved most effective in RE, with an F1 score above 80\%. The outcomes are intended to be represented as a knowledge graph in future work to further enhance automatic verification systems. 

---
# The illusion of a perfect metric: Why evaluating AI's words is harder than it looks 

**Authors**: Maria Paz Oliva, Adriana Correia, Ivan Vankov, Viktor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2508.13816)  

**Abstract**: Evaluating Natural Language Generation (NLG) is crucial for the practical adoption of AI, but has been a longstanding research challenge. While human evaluation is considered the de-facto standard, it is expensive and lacks scalability. Practical applications have driven the development of various automatic evaluation metrics (AEM), designed to compare the model output with human-written references, generating a score which approximates human judgment. Over time, AEMs have evolved from simple lexical comparisons, to semantic similarity models and, more recently, to LLM-based evaluators. However, it seems that no single metric has emerged as a definitive solution, resulting in studies using different ones without fully considering the implications. This paper aims to show this by conducting a thorough examination of the methodologies of existing metrics, their documented strengths and limitations, validation methods, and correlations with human judgment. We identify several key challenges: metrics often capture only specific aspects of text quality, their effectiveness varies by task and dataset, validation practices remain unstructured, and correlations with human judgment are inconsistent. Importantly, we find that these challenges persist in the most recent type of metric, LLM-as-a-Judge, as well as in the evaluation of Retrieval Augmented Generation (RAG), an increasingly relevant task in academia and industry. Our findings challenge the quest for the 'perfect metric'. We propose selecting metrics based on task-specific needs and leveraging complementary evaluations and advocate that new metrics should focus on enhanced validation methodologies. 

---
# Assessing Trustworthiness of AI Training Dataset using Subjective Logic -- A Use Case on Bias 

**Authors**: Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Frank Kargl  

**Link**: [PDF](https://arxiv.org/pdf/2508.13813)  

**Abstract**: As AI systems increasingly rely on training data, assessing dataset trustworthiness has become critical, particularly for properties like fairness or bias that emerge at the dataset level. Prior work has used Subjective Logic to assess trustworthiness of individual data, but not to evaluate trustworthiness properties that emerge only at the level of the dataset as a whole. This paper introduces the first formal framework for assessing the trustworthiness of AI training datasets, enabling uncertainty-aware evaluations of global properties such as bias. Built on Subjective Logic, our approach supports trust propositions and quantifies uncertainty in scenarios where evidence is incomplete, distributed, and/or conflicting. We instantiate this framework on the trustworthiness property of bias, and we experimentally evaluate it based on a traffic sign recognition dataset. The results demonstrate that our method captures class imbalance and remains interpretable and robust in both centralized and federated contexts. 

---
# Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs 

**Authors**: Juncheng Xie, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13805)  

**Abstract**: Controlling the length of text produced by large language models (LLMs) remains challenging: models frequently overshoot or undershoot explicit length instructions because they cannot reliably keep an internal token count. We present a prompt-based, one-shot strategy that compels an off-the-shelf LLM to generate exactly a desired number of tokens - words (English) or characters (Chinese) - without any fine-tuning or iterative sampling. The prompt appends countdown markers and explicit counting rules so that the model "writes while counting." We evaluate on four settings: open-ended generation (1-1000 tokens), XSUM summarization, MT-Bench-LI instruction following, and the LIFEBENCH equal-length track. On MT-Bench-LI, strict length compliance with GPT-4.1 leaps from below 30% under naive prompts to above 95% with our countdown prompt, surpassing the popular draft-then-revise baseline, while judged answer quality is preserved. These results show that precise length control can be achieved through prompt engineering alone, offering a lightweight alternative to training- or decoding-based methods. 

---
# A Fully Transformer Based Multimodal Framework for Explainable Cancer Image Segmentation Using Radiology Reports 

**Authors**: Enobong Adahada, Isabel Sassoon, Kate Hone, Yongmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13796)  

**Abstract**: We introduce Med-CTX, a fully transformer based multimodal framework for explainable breast cancer ultrasound segmentation. We integrate clinical radiology reports to boost both performance and interpretability. Med-CTX achieves exact lesion delineation by using a dual-branch visual encoder that combines ViT and Swin transformers, as well as uncertainty aware fusion. Clinical language structured with BI-RADS semantics is encoded by BioClinicalBERT and combined with visual features utilising cross-modal attention, allowing the model to provide clinically grounded, model generated explanations. Our methodology generates segmentation masks, uncertainty maps, and diagnostic rationales all at once, increasing confidence and transparency in computer assisted diagnosis. On the BUS-BRA dataset, Med-CTX achieves a Dice score of 99% and an IoU of 95%, beating existing baselines U-Net, ViT, and Swin. Clinical text plays a key role in segmentation accuracy and explanation quality, as evidenced by ablation studies that show a -5.4% decline in Dice score and -31% in CIDEr. Med-CTX achieves good multimodal alignment (CLIP score: 85%) and increased confi dence calibration (ECE: 3.2%), setting a new bar for trustworthy, multimodal medical architecture. 

---
# BetaWeb: Towards a Blockchain-enabled Trustworthy Agentic Web 

**Authors**: Zihan Guo, Yuanjian Zhou, Chenyi Wang, Linlin You, Minjie Bian, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13787)  

**Abstract**: The rapid development of large language models (LLMs) has significantly propelled the development of artificial intelligence (AI) agents, which are increasingly evolving into diverse autonomous entities, advancing the LLM-based multi-agent systems (LaMAS). However, current agentic ecosystems remain fragmented and closed. Establishing an interconnected and scalable paradigm for Agentic AI has become a critical prerequisite. Although Agentic Web proposes an open architecture to break the ecosystem barriers, its implementation still faces core challenges such as privacy protection, data management, and value measurement. Existing centralized or semi-centralized paradigms suffer from inherent limitations, making them inadequate for supporting large-scale, heterogeneous, and cross-domain autonomous interactions. To address these challenges, this paper introduces the blockchain-enabled trustworthy Agentic Web (BetaWeb). By leveraging the inherent strengths of blockchain, BetaWeb not only offers a trustworthy and scalable infrastructure for LaMAS but also has the potential to advance the Web paradigm from Web3 (centered on data ownership) towards Web3.5, which emphasizes ownership of agent capabilities and the monetization of intelligence. Beyond a systematic examination of the BetaWeb framework, this paper presents a five-stage evolutionary roadmap, outlining the path of LaMAS from passive execution to advanced collaboration and autonomous governance. We also conduct a comparative analysis of existing products and discuss key challenges of BetaWeb from multiple perspectives. Ultimately, we argue that deep integration between blockchain and LaMAS can lay the foundation for a resilient, trustworthy, and sustainably incentivized digital ecosystem. A summary of the enabling technologies for each stage is available at this https URL. 

---
# DegDiT: Controllable Audio Generation with Dynamic Event Graph Guided Diffusion Transformer 

**Authors**: Yisu Liu, Chenxing Li, Wanqian Zhang, Wenfu Wang, Meng Yu, Ruibo Fu, Zheng Lin, Weiping Wang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13786)  

**Abstract**: Controllable text-to-audio generation aims to synthesize audio from textual descriptions while satisfying user-specified constraints, including event types, temporal sequences, and onset and offset timestamps. This enables precise control over both the content and temporal structure of the generated audio. Despite recent progress, existing methods still face inherent trade-offs among accurate temporal localization, open-vocabulary scalability, and practical efficiency. To address these challenges, we propose DegDiT, a novel dynamic event graph-guided diffusion transformer framework for open-vocabulary controllable audio generation. DegDiT encodes the events in the description as structured dynamic graphs. The nodes in each graph are designed to represent three aspects: semantic features, temporal attributes, and inter-event connections. A graph transformer is employed to integrate these nodes and produce contextualized event embeddings that serve as guidance for the diffusion model. To ensure high-quality and diverse training data, we introduce a quality-balanced data selection pipeline that combines hierarchical event annotation with multi-criteria quality scoring, resulting in a curated dataset with semantic diversity. Furthermore, we present consensus preference optimization, facilitating audio generation through consensus among multiple reward signals. Extensive experiments on AudioCondition, DESED, and AudioTime datasets demonstrate that DegDiT achieves state-of-the-art performances across a variety of objective and subjective evaluation metrics. 

---
# Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images 

**Authors**: Sebastian Ibarra, Javier del Riego, Alessandro Catanese, Julian Cuba, Julian Cardona, Nataly Leon, Jonathan Infante, Karim Lekadir, Oliver Diaz, Richard Osuala  

**Link**: [PDF](https://arxiv.org/pdf/2508.13776)  

**Abstract**: Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis and treatment. However, its reliance on contrast agents introduces safety concerns, contraindications, increased cost, and workflow complexity. To this end, we present pre-contrast conditioned denoising diffusion probabilistic models to synthesize DCE-MRI, introducing, evaluating, and comparing a total of 22 generative model variants in both single-breast and full breast settings. Towards enhancing lesion fidelity, we introduce both tumor-aware loss functions and explicit tumor segmentation mask conditioning. Using a public multicenter dataset and comparing to respective pre-contrast baselines, we observe that subtraction image-based models consistently outperform post-contrast-based models across five complementary evaluation metrics. Apart from assessing the entire image, we also separately evaluate the region of interest, where both tumor-aware losses and segmentation mask inputs improve evaluation metrics. The latter notably enhance qualitative results capturing contrast uptake, albeit assuming access to tumor localization inputs that are not guaranteed to be available in screening settings. A reader study involving 2 radiologists and 4 MRI technologists confirms the high realism of the synthetic images, indicating an emerging clinical potential of generative contrast-enhancement. We share our codebase at this https URL. 

---
# Agentic DraCor and the Art of Docstring Engineering: Evaluating MCP-empowered LLM Usage of the DraCor API 

**Authors**: Peer Trilcke, Ingo Börner, Henny Sluyter-Gäthje, Daniil Skorinkin, Frank Fischer, Carsten Milling  

**Link**: [PDF](https://arxiv.org/pdf/2508.13774)  

**Abstract**: This paper reports on the implementation and evaluation of a Model Context Protocol (MCP) server for DraCor, enabling Large Language Models (LLM) to autonomously interact with the DraCor API. We conducted experiments focusing on tool selection and application by the LLM, employing a qualitative approach that includes systematic observation of prompts to understand how LLMs behave when using MCP tools, evaluating "Tool Correctness", "Tool-Calling Efficiency", and "Tool-Use Reliability". Our findings highlight the importance of "Docstring Engineering", defined as reflexively crafting tool documentation to optimize LLM-tool interaction. Our experiments demonstrate both the promise of agentic AI for research in Computational Literary Studies and the essential infrastructure development needs for reliable Digital Humanities infrastructures. 

---
# PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting 

**Authors**: Tian Sun, Yuqi Chen, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13773)  

**Abstract**: Long-term time series forecasting (LTSF) is a fundamental task with wide-ranging applications. Although Transformer-based models have made significant breakthroughs in forecasting, their effectiveness for time series forecasting remains debatable. In this paper, we revisit the significance of self-attention and propose a simple yet effective mechanism, Periodic-Nested Group Attention, namely PENGUIN. Our approach highlights the importance of explicitly modeling periodic patterns and incorporating relative attention bias for effective time series modeling. To this end, we introduce a periodic-nested relative attention bias that captures periodic structures directly. To handle multiple coexisting periodicities (e.g., daily and weekly cycles), we design a grouped attention mechanism, where each group targets a specific periodicity using a multi-query attention mechanism. Extensive experiments across diverse benchmarks demonstrate that PENGUIN consistently outperforms both MLP-based and Transformer-based models. 

---
# COMPASS: A Multi-Dimensional Benchmark for Evaluating Code Generation in Large Language Models 

**Authors**: James Meaden, Michał Jarosz, Piotr Jodłowski, Grigori Melnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.13757)  

**Abstract**: Current code generation benchmarks focus primarily on functional correctness while overlooking two critical aspects of real-world programming: algorithmic efficiency and code quality. We introduce COMPASS (COdility's Multi-dimensional Programming ASSessment), a comprehensive evaluation framework that assesses code generation across three dimensions: correctness, efficiency, and quality. COMPASS consists of 50 competitive programming problems from real Codility competitions, providing authentic human baselines from 393,150 submissions. Unlike existing benchmarks that treat algorithmically inefficient solutions identically to optimal ones provided they pass test cases, COMPASS systematically evaluates runtime efficiency and code quality using industry-standard analysis tools. Our evaluation of three leading reasoning-enhanced models, Anthropic Claude Opus 4, Google Gemini 2.5 Pro, and OpenAI O4-Mini-High, reveals that models achieving high correctness scores do not necessarily produce efficient algorithms or maintainable code. These findings highlight the importance of evaluating more than just correctness to truly understand the real-world capabilities of code generation models. COMPASS serves as a guiding framework, charting a path for future research toward AI systems that are robust, reliable, and ready for production use. 

---
# Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13755)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has emerged as a powerful paradigm for unlocking reasoning capabilities in large language models, yet its full potential is hindered by two under-explored dimensions: Depth-the hardest problem a model can sample; Breadth-the number of instances consumed in a single iteration. We dissect the popular GRPO algorithm and reveal a systematic bias: the cumulative-advantage disproportionately weights samples with medium accuracy, while down-weighting the low-accuracy instances that are crucial for pushing reasoning boundaries. To rectify the depth neglect, we introduce Difficulty Adaptive Rollout Sampling (DARS), which re-weights hard problems through targeted multi-stage rollouts, thereby increasing the number of positive rollouts for hard problems. Empirically, naively enlarging rollout size only accelerates convergence and even hurts Pass@K. Our DARS, in contrast, delivers consistent Pass@K gains without extra inference cost at convergence. Just as we adaptively expanded the depth of exploration, we now ask whether aggressively scaling the breadth of training data can further amplify reasoning gains. To this end, we intensely scale batch size and replace PPO's mini-batch iterations with full-batch updates over multiple epochs. Increasing breadth significantly enhances Pass@1 performance. Large-breadth training sustains high token-level entropy, indicating continued exploration and reduced gradient noise. We further present DARS-B, which augments DARS with large breadth, and demonstrate simultaneous gains in Pass@K and Pass@1. The results confirm that breadth and adaptive exploration across depth operate as orthogonal dimensions in RLVR, which are key to unleashing the reasoning power of RLVR. 

---
# Mitigating Cross-Image Information Leakage in LVLMs for Multi-Image Tasks 

**Authors**: Yeji Park, Minyoung Lee, Sanghyuk Chun, Junsuk Choe  

**Link**: [PDF](https://arxiv.org/pdf/2508.13744)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate strong performance on single-image tasks. However, we observe that their performance degrades significantly when handling multi-image inputs. This occurs because visual cues from different images become entangled in the model's output. We refer to this phenomenon as cross-image information leakage. To address this issue, we propose FOCUS, a training-free and architecture-agnostic decoding strategy that mitigates cross-image information leakage during inference. FOCUS sequentially masks all but one image with random noise, guiding the model to focus on the single clean image. We repeat this process across all target images to obtain logits under partially masked contexts. These logits are aggregated and then contrastively refined using a noise-only reference input, which suppresses the leakage and yields more accurate outputs. FOCUS consistently improves performance across four multi-image benchmarks and diverse LVLM families. This demonstrates that FOCUS offers a general and practical solution for enhancing multi-image reasoning without additional training or architectural modifications. 

---
# On the Security and Privacy of Federated Learning: A Survey with Attacks, Defenses, Frameworks, Applications, and Future Directions 

**Authors**: Daniel M. Jimenez-Gutierrez, Yelizaveta Falkouskaya, Jose L. Hernandez-Ramos, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2508.13730)  

**Abstract**: Federated Learning (FL) is an emerging distributed machine learning paradigm enabling multiple clients to train a global model collaboratively without sharing their raw data. While FL enhances data privacy by design, it remains vulnerable to various security and privacy threats. This survey provides a comprehensive overview of more than 200 papers regarding the state-of-the-art attacks and defense mechanisms developed to address these challenges, categorizing them into security-enhancing and privacy-preserving techniques. Security-enhancing methods aim to improve FL robustness against malicious behaviors such as byzantine attacks, poisoning, and Sybil attacks. At the same time, privacy-preserving techniques focus on protecting sensitive data through cryptographic approaches, differential privacy, and secure aggregation. We critically analyze the strengths and limitations of existing methods, highlight the trade-offs between privacy, security, and model performance, and discuss the implications of non-IID data distributions on the effectiveness of these defenses. Furthermore, we identify open research challenges and future directions, including the need for scalable, adaptive, and energy-efficient solutions operating in dynamic and heterogeneous FL environments. Our survey aims to guide researchers and practitioners in developing robust and privacy-preserving FL systems, fostering advancements safeguarding collaborative learning frameworks' integrity and confidentiality. 

---
# Prediction is not Explanation: Revisiting the Explanatory Capacity of Mapping Embeddings 

**Authors**: Hanna Herasimchyk, Alhassan Abdelhalim, Sören Laue, Michaela Regneri  

**Link**: [PDF](https://arxiv.org/pdf/2508.13729)  

**Abstract**: Understanding what knowledge is implicitly encoded in deep learning models is essential for improving the interpretability of AI systems. This paper examines common methods to explain the knowledge encoded in word embeddings, which are core elements of large language models (LLMs). These methods typically involve mapping embeddings onto collections of human-interpretable semantic features, known as feature norms. Prior work assumes that accurately predicting these semantic features from the word embeddings implies that the embeddings contain the corresponding knowledge. We challenge this assumption by demonstrating that prediction accuracy alone does not reliably indicate genuine feature-based interpretability.
We show that these methods can successfully predict even random information, concluding that the results are predominantly determined by an algorithmic upper bound rather than meaningful semantic representation in the word embeddings. Consequently, comparisons between datasets based solely on prediction performance do not reliably indicate which dataset is better captured by the word embeddings. Our analysis illustrates that such mappings primarily reflect geometric similarity within vector spaces rather than indicating the genuine emergence of semantic properties. 

---
# Generics and Default Reasoning in Large Language Models 

**Authors**: James Ravi Kirkpatrick, Rachel Katharine Sterken  

**Link**: [PDF](https://arxiv.org/pdf/2508.13718)  

**Abstract**: This paper evaluates the capabilities of 28 large language models (LLMs) to reason with 20 defeasible reasoning patterns involving generic generalizations (e.g., 'Birds fly', 'Ravens are black') central to non-monotonic logic. Generics are of special interest to linguists, philosophers, logicians, and cognitive scientists because of their complex exception-permitting behaviour and their centrality to default reasoning, cognition, and concept acquisition. We find that while several frontier models handle many default reasoning problems well, performance varies widely across models and prompting styles. Few-shot prompting modestly improves performance for some models, but chain-of-thought (CoT) prompting often leads to serious performance degradation (mean accuracy drop -11.14%, SD 15.74% in models performing above 75% accuracy in zero-shot condition, temperature 0). Most models either struggle to distinguish between defeasible and deductive inference or misinterpret generics as universal statements. These findings underscore both the promise and limits of current LLMs for default reasoning. 

---
# The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats 

**Authors**: Markov Grey, Charbel-Raphaël Segerie  

**Link**: [PDF](https://arxiv.org/pdf/2508.13700)  

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do. 

---
# Multi-Plasticity Synergy with Adaptive Mechanism Assignment for Training Spiking Neural Networks 

**Authors**: Yuzhe Liu, Xin Deng, Qiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13673)  

**Abstract**: Spiking Neural Networks (SNNs) are promising brain-inspired models known for low power consumption and superior potential for temporal processing, but identifying suitable learning mechanisms remains a challenge. Despite the presence of multiple coexisting learning strategies in the brain, current SNN training methods typically rely on a single form of synaptic plasticity, which limits their adaptability and representational capability. In this paper, we propose a biologically inspired training framework that incorporates multiple synergistic plasticity mechanisms for more effective SNN training. Our method enables diverse learning algorithms to cooperatively modulate the accumulation of information, while allowing each mechanism to preserve its own relatively independent update dynamics. We evaluated our approach on both static image and dynamic neuromorphic datasets to demonstrate that our framework significantly improves performance and robustness compared to conventional learning mechanism models. This work provides a general and extensible foundation for developing more powerful SNNs guided by multi-strategy brain-inspired learning. 

---
# In-Context Decision Making for Optimizing Complex AutoML Pipelines 

**Authors**: Amir Rezaei Balef, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13657)  

**Abstract**: Combined Algorithm Selection and Hyperparameter Optimization (CASH) has been fundamental to traditional AutoML systems. However, with the advancements of pre-trained models, modern ML workflows go beyond hyperparameter optimization and often require fine-tuning, ensembling, and other adaptation techniques. While the core challenge of identifying the best-performing model for a downstream task remains, the increasing heterogeneity of ML pipelines demands novel AutoML approaches. This work extends the CASH framework to select and adapt modern ML pipelines. We propose PS-PFN to efficiently explore and exploit adapting ML pipelines by extending Posterior Sampling (PS) to the max k-armed bandit problem setup. PS-PFN leverages prior-data fitted networks (PFNs) to efficiently estimate the posterior distribution of the maximal value via in-context learning. We show how to extend this method to consider varying costs of pulling arms and to use different PFNs to model reward distributions individually per arm. Experimental results on one novel and two existing standard benchmark tasks demonstrate the superior performance of PS-PFN compared to other bandit and AutoML strategies. We make our code and data available at this https URL. 

---
# Input Time Scaling 

**Authors**: Rapheal Huang, Weilong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.13654)  

**Abstract**: Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we combine meta-knowledge from LLMs to refine inputs with different strategies. We also find a new phenomenon, training-testing co-design there. We need to apply query strategies during both training and testing. Only applying strategies on training or testing would seriously degrade the performance. We are also surprised to find that seemingly low data quality datasets can gain high performance. Adding irrelevant information to the queries, randomly selecting examples from a minimally filtered dataset, can even perform the best. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, simple dataset size scaling should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. A small set of examples is enough to evoke high-level reasoning ability. With experiments on models trained on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the best result would be 86.7% on AIME24 and 76.7% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints. 

---
# GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling 

**Authors**: Ashish Jha, Anh huy Phan, Razan Dibo, Valentin Leplat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13653)  

**Abstract**: Training modern neural networks on large datasets is computationally and environmentally costly. We introduce GRAFT, a scalable in-training subset selection method that (i) extracts a low-rank feature representation for each batch, (ii) applies a Fast MaxVol sampler to select a small, diverse subset that spans the batch's dominant subspace, and (iii) dynamically adjusts the subset size using a gradient-approximation criterion. By operating in low-rank subspaces and training on carefully chosen examples instead of full batches, GRAFT preserves the training trajectory while reducing wall-clock time, energy consumption, and $\mathrm{CO}_2$ emissions. Across multiple benchmarks, GRAFT matches or exceeds recent selection baselines in both accuracy and efficiency, providing a favorable trade-off between accuracy, efficiency, and emissions. 

---
# Towards a Larger Model via One-Shot Federated Learning on Heterogeneous Client Models 

**Authors**: Wenxuan Ye, Xueli An, Onur Ayan, Junfan Wang, Xueqiang Yan, Georg Carle  

**Link**: [PDF](https://arxiv.org/pdf/2508.13625)  

**Abstract**: Large models, renowned for superior performance, outperform smaller ones even without billion-parameter scales. While mobile network servers have ample computational resources to support larger models than client devices, privacy constraints prevent clients from directly sharing their raw data. Federated Learning (FL) enables decentralized clients to collaboratively train a shared model by exchanging model parameters instead of transmitting raw data. Yet, it requires a uniform model architecture and multiple communication rounds, which neglect resource heterogeneity, impose heavy computational demands on clients, and increase communication overhead. To address these challenges, we propose FedOL, to construct a larger and more comprehensive server model in one-shot settings (i.e., in a single communication round). Instead of model parameter sharing, FedOL employs knowledge distillation, where clients only exchange model prediction outputs on an unlabeled public dataset. This reduces communication overhead by transmitting compact predictions instead of full model weights and enables model customization by allowing heterogeneous model architectures. A key challenge in this setting is that client predictions may be biased due to skewed local data distributions, and the lack of ground-truth labels in the public dataset further complicates reliable learning. To mitigate these issues, FedOL introduces a specialized objective function that iteratively refines pseudo-labels and the server model, improving learning reliability. To complement this, FedOL incorporates a tailored pseudo-label generation and knowledge distillation strategy that effectively integrates diverse knowledge. Simulation results show that FedOL significantly outperforms existing baselines, offering a cost-effective solution for mobile networks where clients possess valuable private data but limited computational resources. 

---
# Bounding Causal Effects and Counterfactuals 

**Authors**: Tobias Maringgele  

**Link**: [PDF](https://arxiv.org/pdf/2508.13607)  

**Abstract**: Causal inference often hinges on strong assumptions - such as no unmeasured confounding or perfect compliance - that are rarely satisfied in practice. Partial identification offers a principled alternative: instead of relying on unverifiable assumptions to estimate causal effects precisely, it derives bounds that reflect the uncertainty inherent in the data. Despite its theoretical appeal, partial identification remains underutilized in applied work, in part due to the fragmented nature of existing methods and the lack of practical guidance. This thesis addresses these challenges by systematically comparing a diverse set of bounding algorithms across multiple causal scenarios. We implement, extend, and unify state-of-the-art methods - including symbolic, optimization-based, and information-theoretic approaches - within a common evaluation framework. In particular, we propose an extension of a recently introduced entropy-bounded method, making it applicable to counterfactual queries such as the Probability of Necessity and Sufficiency (PNS). Our empirical study spans thousands of randomized simulations involving both discrete and continuous data-generating processes. We assess each method in terms of bound tightness, computational efficiency, and robustness to assumption violations. To support practitioners, we distill our findings into a practical decision tree for algorithm selection and train a machine learning model to predict the best-performing method based on observable data characteristics.
All implementations are released as part of an open-source Python package, CausalBoundingEngine, which enables users to apply and compare bounding methods through a unified interface. 

---
# Who Gets the Mic? Investigating Gender Bias in the Speaker Assignment of a Speech-LLM 

**Authors**: Dariia Puhach, Amir H. Payberah, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2508.13603)  

**Abstract**: Similar to text-based Large Language Models (LLMs), Speech-LLMs exhibit emergent abilities and context awareness. However, whether these similarities extend to gender bias remains an open question. This study proposes a methodology leveraging speaker assignment as an analytic tool for bias investigation. Unlike text-based models, which encode gendered associations implicitly, Speech-LLMs must produce a gendered voice, making speaker selection an explicit bias cue. We evaluate Bark, a Text-to-Speech (TTS) model, analyzing its default speaker assignments for textual prompts. If Bark's speaker selection systematically aligns with gendered associations, it may reveal patterns in its training data or model design. To test this, we construct two datasets: (i) Professions, containing gender-stereotyped occupations, and (ii) Gender-Colored Words, featuring gendered connotations. While Bark does not exhibit systematic bias, it demonstrates gender awareness and has some gender inclinations. 

---
# A Comparative Study of Decoding Strategies in Medical Text Generation 

**Authors**: Oriana Presacan, Alireza Nik, Vajira Thambawita, Bogdan Ionescu, Michael Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2508.13580)  

**Abstract**: Large Language Models (LLMs) rely on various decoding strategies to generate text, and these choices can significantly affect output quality. In healthcare, where accuracy is critical, the impact of decoding strategies remains underexplored. We investigate this effect in five open-ended medical tasks, including translation, summarization, question answering, dialogue, and image captioning, evaluating 11 decoding strategies with medically specialized and general-purpose LLMs of different sizes. Our results show that deterministic strategies generally outperform stochastic ones: beam search achieves the highest scores, while {\eta} and top-k sampling perform worst. Slower decoding methods tend to yield better quality. Larger models achieve higher scores overall but have longer inference times and are no more robust to decoding. Surprisingly, while medical LLMs outperform general ones in two of the five tasks, statistical analysis shows no overall performance advantage and reveals greater sensitivity to decoding choice. We further compare multiple evaluation metrics and find that correlations vary by task, with MAUVE showing weak agreement with BERTScore and ROUGE, as well as greater sensitivity to the decoding strategy. These results highlight the need for careful selection of decoding methods in medical applications, as their influence can sometimes exceed that of model choice. 

---
# End-to-End Audio-Visual Learning for Cochlear Implant Sound Coding in Noisy Environments 

**Authors**: Meng-Ping Lin, Enoch Hsin-Ho Huang, Shao-Yi Chien, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13576)  

**Abstract**: The cochlear implant (CI) is a remarkable biomedical device that successfully enables individuals with severe-to-profound hearing loss to perceive sound by converting speech into electrical stimulation signals. Despite advancements in the performance of recent CI systems, speech comprehension in noisy or reverberant conditions remains a challenge. Recent and ongoing developments in deep learning reveal promising opportunities for enhancing CI sound coding capabilities, not only through replicating traditional signal processing methods with neural networks, but also through integrating visual cues as auxiliary data for multimodal speech processing. Therefore, this paper introduces a novel noise-suppressing CI system, AVSE-ECS, which utilizes an audio-visual speech enhancement (AVSE) model as a pre-processing module for the deep-learning-based ElectrodeNet-CS (ECS) sound coding strategy. Specifically, a joint training approach is applied to model AVSE-ECS, an end-to-end CI system. Experimental results indicate that the proposed method outperforms the previous ECS strategy in noisy conditions, with improved objective speech intelligibility scores. The methods and findings in this study demonstrate the feasibility and potential of using deep learning to integrate the AVSE module into an end-to-end CI system 

---
# The 9th AI City Challenge 

**Authors**: Zheng Tang, Shuo Wang, David C. Anastasiu, Ming-Ching Chang, Anuj Sharma, Quan Kong, Norimasa Kobori, Munkhjargal Gochoo, Ganzorig Batnasan, Munkh-Erdene Otgonbold, Fady Alnajjar, Jun-Wei Hsieh, Tomasz Kornuta, Xiaolong Li, Yilin Zhao, Han Zhang, Subhashree Radhakrishnan, Arihant Jain, Ratnesh Kumar, Vidya N. Murali, Yuxing Wang, Sameer Satish Pusegaonkar, Yizhou Wang, Sujit Biswas, Xunlei Wu, Zhedong Zheng, Pranamesh Chakraborty, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13564)  

**Abstract**: The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks. 

---
# Physics-Informed Neural Networks for Programmable Origami Metamaterials with Controlled Deployment 

**Authors**: Sukheon Kang, Youngkwon Kim, Jinkyu Yang, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13559)  

**Abstract**: Origami-inspired structures provide unprecedented opportunities for creating lightweight, deployable systems with programmable mechanical responses. However, their design remains challenging due to complex nonlinear mechanics, multistability, and the need for precise control of deployment forces. Here, we present a physics-informed neural network (PINN) framework for both forward prediction and inverse design of conical Kresling origami (CKO) without requiring pre-collected training data. By embedding mechanical equilibrium equations directly into the learning process, the model predicts complete energy landscapes with high accuracy while minimizing non-physical artifacts. The inverse design routine specifies both target stable-state heights and separating energy barriers, enabling freeform programming of the entire energy curve. This capability is extended to hierarchical CKO assemblies, where sequential layer-by-layer deployment is achieved through programmed barrier magnitudes. Finite element simulations and experiments on physical prototypes validate the designed deployment sequences and barrier ratios, confirming the robustness of the approach. This work establishes a versatile, data-free route for programming complex mechanical energy landscapes in origami-inspired metamaterials, offering broad potential for deployable aerospace systems, morphing structures, and soft robotic actuators. 

---
# Collapsing ROC approach for risk prediction research on both common and rare variants 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13552)  

**Abstract**: Risk prediction that capitalizes on emerging genetic findings holds great promise for improving public health and clinical care. However, recent risk prediction research has shown that predictive tests formed on existing common genetic loci, including those from genome-wide association studies, have lacked sufficient accuracy for clinical use. Because most rare variants on the genome have not yet been studied for their role in risk prediction, future disease prediction discoveries should shift toward a more comprehensive risk prediction strategy that takes into account both common and rare variants. We are proposing a collapsing receiver operating characteristic CROC approach for risk prediction research on both common and rare variants. The new approach is an extension of a previously developed forward ROC FROC approach, with additional procedures for handling rare variants. The approach was evaluated through the use of 533 single-nucleotide polymorphisms SNPs in 37 candidate genes from the Genetic Analysis Workshop 17 mini-exome data set. We found that a prediction model built on all SNPs gained more accuracy AUC = 0.605 than one built on common variants alone AUC = 0.585. We further evaluated the performance of two approaches by gradually reducing the number of common variants in the analysis. We found that the CROC method attained more accuracy than the FROC method when the number of common variants in the data decreased. In an extreme scenario, when there are only rare variants in the data, the CROC reached an AUC value of 0.603, whereas the FROC had an AUC value of 0.524. 

---
# FLAIR: Frequency- and Locality-Aware Implicit Neural Representations 

**Authors**: Sukhun Ko, Dahyeon Kye, Kyle Min, Chanho Eom, Jihyong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.13544)  

**Abstract**: Implicit Neural Representations (INRs) leverage neural networks to map coordinates to corresponding signals, enabling continuous and compact representations. This paradigm has driven significant advances in various vision tasks. However, existing INRs lack frequency selectivity, spatial localization, and sparse representations, leading to an over-reliance on redundant signal components. Consequently, they exhibit spectral bias, tending to learn low-frequency components early while struggling to capture fine high-frequency details. To address these issues, we propose FLAIR (Frequency- and Locality-Aware Implicit Neural Representations), which incorporates two key innovations. The first is RC-GAUSS, a novel activation designed for explicit frequency selection and spatial localization under the constraints of the time-frequency uncertainty principle (TFUP). The second is Wavelet-Energy-Guided Encoding (WEGE), which leverages the discrete wavelet transform (DWT) to compute energy scores and explicitly guide frequency information to the network. Our method consistently outperforms existing INRs in 2D image representation and restoration, as well as 3D reconstruction. 

---
# EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors 

**Authors**: Shikun Zhang, Cunjian Chen, Yiqun Wang, Qiuhong Ke, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13537)  

**Abstract**: High-fidelity head avatar reconstruction plays a crucial role in AR/VR, gaming, and multimedia content creation. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated effectiveness in modeling complex geometry with real-time rendering capability and are now widely used in high-fidelity head avatar reconstruction tasks. However, existing 3DGS-based methods still face significant challenges in capturing fine-grained facial expressions and preserving local texture continuity, especially in highly deformable regions. To mitigate these limitations, we propose a novel 3DGS-based framework termed EAvatar for head reconstruction that is both expression-aware and deformation-aware. Our method introduces a sparse expression control mechanism, where a small number of key Gaussians are used to influence the deformation of their neighboring Gaussians, enabling accurate modeling of local deformations and fine-scale texture transitions. Furthermore, we leverage high-quality 3D priors from pretrained generative models to provide a more reliable facial geometry, offering structural guidance that improves convergence stability and shape accuracy during training. Experimental results demonstrate that our method produces more accurate and visually coherent head reconstructions with improved expression controllability and detail fidelity. 

---
# MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence 

**Authors**: Chao Tang, Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13534)  

**Abstract**: Imitating tool manipulation from human videos offers an intuitive approach to teaching robots, while also providing a promising and scalable alternative to labor-intensive teleoperation data collection for visuomotor policy learning. While humans can mimic tool manipulation behavior by observing others perform a task just once and effortlessly transfer the skill to diverse tools for functionally equivalent tasks, current robots struggle to achieve this level of generalization. A key challenge lies in establishing function-level correspondences, considering the significant geometric variations among functionally similar tools, referred to as intra-function variations. To address this challenge, we propose MimicFunc, a framework that establishes functional correspondences with function frame, a function-centric local coordinate frame constructed with keypoint-based abstraction, for imitating tool manipulation skills. Experiments demonstrate that MimicFunc effectively enables the robot to generalize the skill from a single RGB-D human video to manipulating novel tools for functionally equivalent tasks. Furthermore, leveraging MimicFunc's one-shot generalization capability, the generated rollouts can be used to train visuomotor policies without requiring labor-intensive teleoperation data collection for novel objects. Our code and video are available at this https URL. 

---
# Evaluating Open-Source Vision Language Models for Facial Emotion Recognition against Traditional Deep Learning Models 

**Authors**: Vamsi Krishna Mulukutla, Sai Supriya Pavarala, Srinivasa Raju Rudraraju, Sridevi Bonthu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13524)  

**Abstract**: Facial Emotion Recognition (FER) is crucial for applications such as human-computer interaction and mental health diagnostics. This study presents the first empirical comparison of open-source Vision-Language Models (VLMs), including Phi-3.5 Vision and CLIP, against traditional deep learning models VGG19, ResNet-50, and EfficientNet-B0 on the challenging FER-2013 dataset, which contains 35,887 low-resolution grayscale images across seven emotion classes. To address the mismatch between VLM training assumptions and the noisy nature of FER data, we introduce a novel pipeline that integrates GFPGAN-based image restoration with FER evaluation. Results show that traditional models, particularly EfficientNet-B0 (86.44%) and ResNet-50 (85.72%), significantly outperform VLMs like CLIP (64.07%) and Phi-3.5 Vision (51.66%), highlighting the limitations of VLMs in low-quality visual tasks. In addition to performance evaluation using precision, recall, F1-score, and accuracy, we provide a detailed computational cost analysis covering preprocessing, training, inference, and evaluation phases, offering practical insights for deployment. This work underscores the need for adapting VLMs to noisy environments and provides a reproducible benchmark for future research in emotion recognition. 

---
# DDoS Attacks in Cloud Computing: Detection and Prevention 

**Authors**: Zain Ahmad, Musab Ahmad, Bilal Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.13522)  

**Abstract**: DDoS attacks are one of the most prevalent and harmful cybersecurity threats faced by organizations and individuals today. In recent years, the complexity and frequency of DDoS attacks have increased significantly, making it challenging to detect and mitigate them effectively. The study analyzes various types of DDoS attacks, including volumetric, protocol, and application layer attacks, and discusses the characteristics, impact, and potential targets of each type. It also examines the existing techniques used for DDoS attack detection, such as packet filtering, intrusion detection systems, and machine learning-based approaches, and their strengths and limitations. Moreover, the study explores the prevention techniques employed to mitigate DDoS attacks, such as firewalls, rate limiting , CPP and ELD mechanism. It evaluates the effectiveness of each approach and its suitability for different types of attacks and environments. In conclusion, this study provides a comprehensive overview of the different types of DDoS attacks, their detection, and prevention techniques. It aims to provide insights and guidelines for organizations and individuals to enhance their cybersecurity posture and protect against DDoS attacks. 

---
# Calibrating Biased Distribution in VFM-derived Latent Space via Cross-Domain Geometric Consistency 

**Authors**: Yanbiao Ma, Wei Dai, Bowei Liu, Jiayi Chen, Wenke Huang, Guancheng Wan, Zhiwu Lu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13518)  

**Abstract**: Despite the fast progress of deep learning, one standing challenge is the gap of the observed training samples and the underlying true distribution. There are multiple reasons for the causing of this gap e.g. sampling bias, noise etc. In the era of foundation models, we show that when leveraging the off-the-shelf (vision) foundation models (e.g., CLIP, DINOv2) for feature extraction, the geometric shapes of the resulting feature distributions exhibit remarkable transferability across domains and datasets. To verify its practical usefulness, we embody our geometric knowledge-guided distribution calibration framework in two popular and challenging settings: federated learning and long-tailed recognition. In the federated setting, we devise a technique of acquiring the global geometric shape under privacy constraints, then leverage this knowledge to generate new samples for clients, in the aim of bridging the gap between local and global observations. In long-tailed learning, it utilizes the geometric knowledge transferred from sample-rich categories to recover the true distribution for sample-scarce tail classes. Comprehensive experiments show that our proposed geometric knowledge-guided distribution calibration effectively overcomes information deficits caused by data heterogeneity and sample imbalance, with boosted performance across benchmarks. 

---
# Heterogeneous Influence Maximization in User Recommendation 

**Authors**: Hongru Hou, Jiachen Sun, Wenqing Lin, Wendong Bi, Xiangrong Wang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13517)  

**Abstract**: User recommendation systems enhance user engagement by encouraging users to act as inviters to interact with other users (invitees), potentially fostering information propagation. Conventional recommendation methods typically focus on modeling interaction willingness. Influence-Maximization (IM) methods focus on identifying a set of users to maximize the information propagation. However, existing methods face two significant challenges. First, recommendation methods fail to unleash the candidates' spread capability. Second, IM methods fail to account for the willingness to interact. To solve these issues, we propose two models named HeteroIR and HeteroIM. HeteroIR provides an intuitive solution to unleash the dissemination potential of user recommendation systems. HeteroIM fills the gap between the IM method and the recommendation task, improving interaction willingness and maximizing spread coverage. The HeteroIR introduces a two-stage framework to estimate the spread profits. The HeteroIM incrementally selects the most influential invitee to recommend and rerank based on the number of reverse reachable (RR) sets containing inviters and invitees. RR set denotes a set of nodes that can reach a target via propagation. Extensive experiments show that HeteroIR and HeteroIM significantly outperform the state-of-the-art baselines with the p-value < 0.05. Furthermore, we have deployed HeteroIR and HeteroIM in Tencent's online gaming platforms and gained an 8.5\% and 10\% improvement in the online A/B test, respectively. Implementation codes are available at this https URL. 

---
# ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs 

**Authors**: Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao, Xinke Jiang, Zheng Li, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13514)  

**Abstract**: Interactive medical questioning is essential in real-world clinical consultations, where physicians must actively gather information from patients. While medical Large Language Models (LLMs) have shown impressive capabilities in static medical question answering, they predominantly operate under a reactive paradigm: generating answers directly without seeking additional information, which risks incorrect diagnoses in such interactive settings. To address this limitation, we propose ProMed, a reinforcement learning (RL) framework that transitions medical LLMs toward a proactive paradigm, equipping them with the ability to ask clinically valuable questions before decision-making. At the core of ProMed is the Shapley Information Gain (SIG) reward, which quantifies the clinical utility of each question by combining the amount of newly acquired information with its contextual importance, estimated via Shapley values. We integrate SIG into a two-stage training pipeline: (1) SIG-Guided Model Initialization uses Monte Carlo Tree Search (MCTS) to construct high-reward interaction trajectories to supervise the model, and (2) SIG-Augmented Policy Optimization, which integrates SIG and enhances RL with a novel SIG-guided Reward Distribution Mechanism that assigns higher rewards to informative questions for targeted optimization. Extensive experiments on two newly curated partial-information medical benchmarks demonstrate that ProMed significantly outperforms state-of-the-art methods by an average of 6.29% and delivers a 54.45% gain over the reactive paradigm, while also generalizing robustly to out-of-domain cases. 

---
# LLM-Enhanced Linear Autoencoders for Recommendation 

**Authors**: Jaewan Moon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13500)  

**Abstract**: Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at this https URL. 

---
# CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving 

**Authors**: Fuyang Liu, Jilin Mei, Fangyuan Mao, Chen Min, Yan Xing, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13485)  

**Abstract**: 4D radar-based object detection has garnered great attention for its robustness in adverse weather conditions and capacity to deliver rich spatial information across diverse driving scenarios. Nevertheless, the sparse and noisy nature of 4D radar point clouds poses substantial challenges for effective perception. To address the limitation, we present CORENet, a novel cross-modal denoising framework that leverages LiDAR supervision to identify noise patterns and extract discriminative features from raw 4D radar data. Designed as a plug-and-play architecture, our solution enables seamless integration into voxel-based detection frameworks without modifying existing pipelines. Notably, the proposed method only utilizes LiDAR data for cross-modal supervision during training while maintaining full radar-only operation during inference. Extensive evaluation on the challenging Dual-Radar dataset, which is characterized by elevated noise level, demonstrates the effectiveness of our framework in enhancing detection robustness. Comprehensive experiments validate that CORENet achieves superior performance compared to existing mainstream approaches. 

---
# STER-VLM: Spatio-Temporal With Enhanced Reference Vision-Language Models 

**Authors**: Tinh-Anh Nguyen-Nhu, Triet Dao Hoang Minh, Dat To-Thanh, Phuc Le-Gia, Tuan Vo-Lan, Tien-Huy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13470)  

**Abstract**: Vision-language models (VLMs) have emerged as powerful tools for enabling automated traffic analysis; however, current approaches often demand substantial computational resources and struggle with fine-grained spatio-temporal understanding. This paper introduces STER-VLM, a computationally efficient framework that enhances VLM performance through (1) caption decomposition to tackle spatial and temporal information separately, (2) temporal frame selection with best-view filtering for sufficient temporal information, and (3) reference-driven understanding for capturing fine-grained motion and dynamic context and (4) curated visual/textual prompt techniques. Experimental results on the WTS \cite{kong2024wts} and BDD \cite{BDD} datasets demonstrate substantial gains in semantic richness and traffic scene interpretation. Our framework is validated through a decent test score of 55.655 in the AI City Challenge 2025 Track 2, showing its effectiveness in advancing resource-efficient and accurate traffic analysis for real-world applications. 

---
# Consumer Autonomy or Illusion? Rethinking Consumer Agency in the Age of Algorithms 

**Authors**: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana  

**Link**: [PDF](https://arxiv.org/pdf/2508.13440)  

**Abstract**: Consumer agency in the digital age is increasingly constrained by systemic barriers and algorithmic manipulation, raising concerns about the authenticity of consumption choices. Nowadays, financial decisions are shaped by external pressures like obligatory consumption, algorithmic persuasion, and unstable work schedules that erode financial autonomy. Obligatory consumption (like hidden fees) is intensified by digital ecosystems. Algorithmic tactics like personalized recommendations lead to impulsive purchases. Unstable work schedules also undermine financial planning. Thus, it is important to study how these factors impact consumption agency. To do so, we examine formal models grounded in discounted consumption with constraints that bound agency. We construct analytical scenarios in which consumers face obligatory payments, algorithm-influenced impulsive expenses, or unpredictable income due to temporal instability. Using this framework, we demonstrate that even rational, utility-maximizing agents can experience early financial ruin when agency is limited across structural, behavioral, or temporal dimensions and how diminished autonomy impacts long-term financial well-being. Our central argument is that consumer agency must be treated as a value (not a given) requiring active cultivation, especially in digital ecosystems. The connection between our formal modeling and this argument allows us to indicate that limitations on agency (whether structural, behavioral, or temporal) can be rigorously linked to measurable risks like financial instability. This connection is also a basis for normative claims about consumption as a value, by anchoring them in a formally grounded analysis of consumer behavior. As solutions, we study systemic interventions and consumer education to support value deliberation and informed choices. We formally demonstrate how these measures strengthen agency. 

---
# Structured Prompting and Multi-Agent Knowledge Distillation for Traffic Video Interpretation and Risk Inference 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13439)  

**Abstract**: Comprehensive highway scene understanding and robust traffic risk inference are vital for advancing Intelligent Transportation Systems (ITS) and autonomous driving. Traditional approaches often struggle with scalability and generalization, particularly under the complex and dynamic conditions of real-world environments. To address these challenges, we introduce a novel structured prompting and knowledge distillation framework that enables automatic generation of high-quality traffic scene annotations and contextual risk assessments. Our framework orchestrates two large Vision-Language Models (VLMs): GPT-4o and o3-mini, using a structured Chain-of-Thought (CoT) strategy to produce rich, multi-perspective outputs. These outputs serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a much smaller student VLM. The resulting compact 3B-scale model, named VISTA (Vision for Intelligent Scene and Traffic Analysis), is capable of understanding low-resolution traffic videos and generating semantically faithful, risk-aware captions. Despite its significantly reduced parameter count, VISTA achieves strong performance across established captioning metrics (BLEU-4, METEOR, ROUGE-L, and CIDEr) when benchmarked against its teacher models. This demonstrates that effective knowledge distillation and structured multi-agent supervision can empower lightweight VLMs to capture complex reasoning capabilities. The compact architecture of VISTA facilitates efficient deployment on edge devices, enabling real-time risk monitoring without requiring extensive infrastructure upgrades. 

---
# Dynamic Design of Machine Learning Pipelines via Metalearning 

**Authors**: Edesio Alcobaça, André C. P. L. F. de Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.13436)  

**Abstract**: Automated machine learning (AutoML) has democratized the design of machine learning based systems, by automating model selection, hyperparameter tuning and feature engineering. However, the high computational cost associated with traditional search and optimization strategies, such as Random Search, Particle Swarm Optimization and Bayesian Optimization, remains a significant challenge. Moreover, AutoML systems typically explore a large search space, which can lead to overfitting. This paper introduces a metalearning method for dynamically designing search spaces for AutoML system. The proposed method uses historical metaknowledge to select promising regions of the search space, accelerating the optimization process. According to experiments conducted for this study, the proposed method can reduce runtime by 89\% in Random Search and search space by (1.8/13 preprocessor and 4.3/16 classifier), without compromising significant predictive performance. Moreover, the proposed method showed competitive performance when adapted to Auto-Sklearn, reducing its search space. Furthermore, this study encompasses insights into meta-feature selection, meta-model explainability, and the trade-offs inherent in search space reduction strategies. 

---
# SVDformer: Direction-Aware Spectral Graph Embedding Learning via SVD and Transformer 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13435)  

**Abstract**: Directed graphs are widely used to model asymmetric relationships in real-world systems. However, existing directed graph neural networks often struggle to jointly capture directional semantics and global structural patterns due to their isotropic aggregation mechanisms and localized filtering mechanisms. To address this limitation, this paper proposes SVDformer, a novel framework that synergizes SVD and Transformer architecture for direction-aware graph representation learning. SVDformer first refines singular value embeddings through multi-head self-attention, adaptively enhancing critical spectral components while suppressing high-frequency noise. This enables learnable low-pass/high-pass graph filtering without requiring spectral kernels. Furthermore, by treating singular vectors as directional projection bases and singular values as scaling factors, SVDformer uses the Transformer to model multi-scale interactions between incoming/outgoing edge patterns through attention weights, thereby explicitly preserving edge directionality during feature propagation. Extensive experiments on six directed graph benchmarks demonstrate that SVDformer consistently outperforms state-of-the-art GNNs and direction-aware baselines on node classification tasks, establishing a new paradigm for learning representations on directed graphs. 

---
# EventTSF: Event-Aware Non-Stationary Time Series Forecasting 

**Authors**: Yunfeng Ge, Ming Jin, Yiji Zhao, Hongyan Li, Bo Du, Chang Xu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13434)  

**Abstract**: Time series forecasting plays a vital role in critical domains like energy and transportation, where non-stationary dynamics are deeply intertwined with events in other modalities such as texts. However, incorporating natural language-based external events to improve non-stationary forecasting remains largely unexplored, as most approaches still rely on a single modality, resulting in limited contextual knowledge and model underperformance. Enabling fine-grained multimodal interactions between temporal and textual data is challenged by three fundamental issues: (1) the difficulty of fine-grained synchronization between time-varying discrete textual events and continuous time series; (2) the inherent temporal uncertainty introduced by textual semantics; and (3) the misalignment between textual event embeddings and multi-resolution temporal patterns. In this work, we address these challenges by introducing event-aware non-stationary time series forecasting (EventTSF), an autoregressive generation framework that integrates historical time series with textual events to make subsequent forecasts. Specifically, EventTSF uses autoregressive diffusion with flow matching at each step to capture nuanced temporal-event interactions. To handle event-induced uncertainty, flow matching timesteps are adaptively controlled according to event semantic signals. The underlying denoiser employs a multimodal U-shaped diffusion transformer that efficiently fuses temporal and textual modalities across different resolutions. Extensive experiments on 8 synthetic and real-world datasets show that EventTSF outperforms 12 baselines across diverse event-aware non-stationary time series forecasting scenarios, achieving substantial improvements of 10.7% higher forecasting accuracy and $1.13\times$ faster training efficiency. 

---
# AlphaX: An AI-Based Value Investing Strategy for the Brazilian Stock Market 

**Authors**: Paulo André Lima de Castro  

**Link**: [PDF](https://arxiv.org/pdf/2508.13429)  

**Abstract**: Autonomous trading strategies have been a subject of research within the field of artificial intelligence (AI) for aconsiderable period. Various AI techniques have been explored to develop autonomous agents capable of trading financial assets. These approaches encompass traditional methods such as neural networks, fuzzy logic, and reinforcement learning, as well as more recent advancements, including deep neural networks and deep reinforcement learning. Many developers report success in creating strategies that exhibit strong performance during simulations using historical price data, a process commonly referred to as backtesting. However, when these strategies are deployed in real markets, their performance often deteriorates, particularly in terms of risk-adjusted returns. In this study, we propose an AI-based strategy inspired by a classical investment paradigm: Value Investing. Financial AI models are highly susceptible to lookahead bias and other forms of bias that can significantly inflate performance in backtesting compared to live trading conditions. To address this issue, we conducted a series of computational simulations while controlling for these biases, thereby reducing the risk of overfitting. Our results indicate that the proposed approach outperforms major Brazilian market benchmarks. Moreover, the strategy, named AlphaX, demonstrated superior performance relative to widely used technical indicators such as the Relative Strength Index (RSI) and Money Flow Index (MFI), with statistically significant results. Finally, we discuss several open challenges and highlight emerging technologies in qualitative analysis that may contribute to the development of a comprehensive AI-based Value Investing framework in the future 

---
# Mitigating Easy Option Bias in Multiple-Choice Question Answering 

**Authors**: Hao Zhang, Chen Li, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.13428)  

**Abstract**: In this early study, we observe an Easy-Options Bias (EOB) issue in some multiple-choice Visual Question Answering (VQA) benchmarks such as MMStar, RealWorldQA, SEED-Bench, Next-QA, STAR benchmark and Video-MME. This bias allows vision-language models (VLMs) to select the correct answer using only the vision (V) and options (O) as inputs, without the need for the question (Q). Through grounding experiments, we attribute the bias to an imbalance in visual relevance: the correct answer typically aligns more closely with the visual contents than the negative options in feature space, creating a shortcut for VLMs to infer the answer via simply vision-option similarity matching. To fix this, we introduce GroundAttack, a toolkit that automatically generates hard negative options as visually plausible as the correct answer. We apply it to the NExT-QA and MMStar datasets, creating new EOB-free annotations. On these EOB-free annotations, current VLMs approach to random accuracies under (V+O) settings, and drop to non-saturated accuracies under (V+Q+O) settings, providing a more realistic evaluation of VLMs' QA ability. Codes and new annotations will be released soon. 

---
# ALIGN: Word Association Learning for Cross-Cultural Generalization in Large Language Models 

**Authors**: Chunhua Liu, Kabir Manandhar Shrestha, Sukai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13426)  

**Abstract**: As large language models (LLMs) increasingly mediate cross-cultural communication, their behavior still reflects the distributional bias of the languages and viewpoints that are over-represented in their pre-training corpora. Yet, it remains a challenge to model and align culture due to limited cultural knowledge and a lack of exploration into effective learning approaches. We introduce a cost-efficient, cognitively grounded remedy: parameter-efficient fine-tuning on native speakers' free word-association norms, which encode implicit cultural schemas. Leveraging English-US and Mandarin associations from the Small-World-of-Words project, we adapt Llama-3.1-8B and Qwen-2.5-7B via supervised fine-tuning (SFT) and PPO-based preference optimization. SFT boosts held-out association Precision at 5 by 16-20% in English and 43-165% in Mandarin, lifts median concreteness by +0.20, and attains human-level valence and arousal. These lexical gains transfer: on World-Values-Survey questions, fine-tuned models shift answer distributions toward the target culture, and on a 50-item high-tension subset, Qwen's Chinese-aligned responses double while Llama's US bias drops by one-third. Our 7-8B models rival or beat vanilla 70B baselines, showing that a few million culture-grounded associations can instill value alignment without costly retraining. Our work highlights both the promise and the need for future research grounded in human cognition in improving cultural alignment in AI models. 

---
# AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System 

**Authors**: Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13423)  

**Abstract**: In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy. 

---
# Semi-Supervised Anomaly Detection Pipeline for SOZ Localization Using Ictal-Related Chirp 

**Authors**: Nooshin Bahador, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2508.13406)  

**Abstract**: This study presents a quantitative framework for evaluating the spatial concordance between clinically defined seizure onset zones (SOZs) and statistically anomalous channels identified through time-frequency analysis of chirp events. The proposed pipeline employs a two-step methodology: (1) Unsupervised Outlier Detection, where Local Outlier Factor (LOF) analysis with adaptive neighborhood selection identifies anomalous channels based on spectro-temporal features of chirp (Onset frequency, offset frequency, and temporal duration); and (2) Spatial Correlation Analysis, which computes both exact co-occurrence metrics and weighted index similarity, incorporating hemispheric congruence and electrode proximity. Key findings demonstrate that the LOF-based approach (N neighbors=20, contamination=0.2) effectively detects outliers, with index matching (weighted by channel proximity) outperforming exact matching in SOZ localization. Performance metrics (precision, recall, F1) were highest for seizure-free patients (Index Precision mean: 0.903) and those with successful surgical outcomes (Index Precision mean: 0.865), whereas failure cases exhibited lower concordance (Index Precision mean: 0.460). The key takeaway is that chirp-based outlier detection, combined with weighted spatial metrics, provides a complementary method for SOZ localization, particularly in patients with successful surgical outcomes. 

---
# Datarus-R1: An Adaptive Multi-Step Reasoning LLM for Automated Data Analysis 

**Authors**: Ayoub Ben Chaliah, Hela Dellagi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13382)  

**Abstract**: We present Datarus-R1-14B, a 14 B-parameter open-weights language model fine-tuned from Qwen 2.5-14B-Instruct to act as a virtual data analyst and graduate-level problem solver. Datarus is trained not on isolated question-answer pairs but on full analytical trajectories including reasoning steps, code execution, error traces, self-corrections, and final conclusions, all captured in a ReAct-style notebook format spanning finance, medicine, numerical analysis, and other quantitative domains. Our training pipeline combines (i) a trajectory-centric synthetic data generator that yielded 144 000 tagged notebook episodes, (ii) a dual-reward framework blending a lightweight tag-based structural signal with a Hierarchical Reward Model (HRM) that scores both single-step soundness and end-to-end coherence, and (iii) a memory-optimized implementation of Group Relative Policy Optimization (GRPO) featuring KV-cache reuse, sequential generation, and reference-model sharding. A cosine curriculum smoothly shifts emphasis from structural fidelity to semantic depth, reducing the format collapse and verbosity that often plague RL-aligned LLMs. A central design choice in Datarus is it dual reasoning interface. In agentic mode the model produces ReAct-tagged steps that invoke Python tools to execute real code; in reflection mode it outputs compact Chain-of-Thought (CoT) traces delimited by <think> and <answer> tags. On demanding postgraduate-level problems, Datarus exhibits an "AHA-moment" pattern: it sketches hypotheses, revises them once or twice, and converges avoiding the circular, token-inflating loops common to contemporary systems. Across standard public benchmarks Datarus surpasses similar size models and even reaches the level of larger reasoning models such as QwQ-32B achieving up to 30% higher accuracy on AIME 2024/2025 and LiveCodeBench while emitting 18-49% fewer tokens per solution. 

---
# Whispering Context: Distilling Syntax and Semantics for Long Speech Transcripts 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2508.13376)  

**Abstract**: ASR systems often struggle with maintaining syntactic and semantic accuracy in long audio transcripts, impacting tasks like Named Entity Recognition (NER), capitalization, and punctuation. We propose a novel approach that enhances ASR by distilling contextual knowledge from LLaMA models into Whisper. Our method uses two strategies: (1) token level distillation with optimal transport to align dimensions and sequence lengths, and (2) representation loss minimization between sentence embeddings of Whisper and LLaMA, blending syntax and semantics. Evaluations on the Spoken Wikipedia dataset, a benchmark with long audios and rich entities demonstrate significant improvements in Word Error Rate (WER), NER, capitalization, and punctuation success. By introducing novel NER metrics and exploring semantics aware ASR, our work highlights the value of integrating linguistic context into transcription, setting a foundation for robust, context-aware ASR in longform speech. 

---
# Overcoming Latency Bottlenecks in On-Device Speech Translation: A Cascaded Approach with Alignment-Based Streaming MT 

**Authors**: Zeeshan Ahmed, Frank Seide, Niko Moritz, Ju Lin, Ruiming Xie, Simone Merello, Zhe Liu, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13358)  

**Abstract**: This paper tackles several challenges that arise when integrating Automatic Speech Recognition (ASR) and Machine Translation (MT) for real-time, on-device streaming speech translation. Although state-of-the-art ASR systems based on Recurrent Neural Network Transducers (RNN-T) can perform real-time transcription, achieving streaming translation in real-time remains a significant challenge. To address this issue, we propose a simultaneous translation approach that effectively balances translation quality and latency. We also investigate efficient integration of ASR and MT, leveraging linguistic cues generated by the ASR system to manage context and utilizing efficient beam-search pruning techniques such as time-out and forced finalization to maintain system's real-time factor. We apply our approach to an on-device bilingual conversational speech translation and demonstrate that our techniques outperform baselines in terms of latency and quality. Notably, our technique narrows the quality gap with non-streaming translation systems, paving the way for more accurate and efficient real-time speech translation. 

---
# Counterfactual Probabilistic Diffusion with Expert Models 

**Authors**: Wenhao Mu, Zhi Cao, Mehmed Uludag, Alexander Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13355)  

**Abstract**: Predicting counterfactual distributions in complex dynamical systems is essential for scientific modeling and decision-making in domains such as public health and medicine. However, existing methods often rely on point estimates or purely data-driven models, which tend to falter under data scarcity. We propose a time series diffusion-based framework that incorporates guidance from imperfect expert models by extracting high-level signals to serve as structured priors for generative modeling. Our method, ODE-Diff, bridges mechanistic and data-driven approaches, enabling more reliable and interpretable causal inference. We evaluate ODE-Diff across semi-synthetic COVID-19 simulations, synthetic pharmacological dynamics, and real-world case studies, demonstrating that it consistently outperforms strong baselines in both point prediction and distributional accuracy. 

---
# A Dual-Attention Graph Network for fMRI Data Classification 

**Authors**: Amirali Arbab, Zeinab Davarani, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.13328)  

**Abstract**: Understanding the complex neural activity dynamics is crucial for the development of the field of neuroscience. Although current functional MRI classification approaches tend to be based on static functional connectivity or cannot capture spatio-temporal relationships comprehensively, we present a new framework that leverages dynamic graph creation and spatiotemporal attention mechanisms for Autism Spectrum Disorder(ASD) diagnosis. The approach used in this research dynamically infers functional brain connectivity in each time interval using transformer-based attention mechanisms, enabling the model to selectively focus on crucial brain regions and time segments. By constructing time-varying graphs that are then processed with Graph Convolutional Networks (GCNs) and transformers, our method successfully captures both localized interactions and global temporal dependencies. Evaluated on the subset of ABIDE dataset, our model achieves 63.2 accuracy and 60.0 AUC, outperforming static graph-based approaches (e.g., GCN:51.8). This validates the efficacy of joint modeling of dynamic connectivity and spatio-temporal context for fMRI classification. The core novelty arises from (1) attention-driven dynamic graph creation that learns temporal brain region interactions and (2) hierarchical spatio-temporal feature fusion through GCNtransformer fusion. 

---
# A Surveillance Based Interactive Robot 

**Authors**: Kshitij Kavimandan, Pooja Mangal, Devanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2508.13319)  

**Abstract**: We build a mobile surveillance robot that streams video in real time and responds to speech so a user can monitor and steer it from a phone or browser. The system uses two Raspberry Pi 4 units: a front unit on a differential drive base with camera, mic, and speaker, and a central unit that serves the live feed and runs perception. Video is sent with FFmpeg. Objects in the scene are detected using YOLOv3 to support navigation and event awareness. For voice interaction, we use Python libraries for speech recognition, multilingual translation, and text-to-speech, so the robot can take spoken commands and read back responses in the requested language. A Kinect RGB-D sensor provides visual input and obstacle cues. In indoor tests the robot detects common objects at interactive frame rates on CPU, recognises commands reliably, and translates them to actions without manual control. The design relies on off-the-shelf hardware and open software, making it easy to reproduce. We discuss limits and practical extensions, including sensor fusion with ultrasonic range data, GPU acceleration, and adding face and text recognition. 

---
# Diff-MSM: Differentiable MusculoSkeletal Model for Simultaneous Identification of Human Muscle and Bone Parameters 

**Authors**: Yingfan Zhou, Philip Sanderink, Sigurd Jager Lemming, Cheng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13303)  

**Abstract**: High-fidelity personalized human musculoskeletal models are crucial for simulating realistic behavior of physically coupled human-robot interactive systems and verifying their safety-critical applications in simulations before actual deployment, such as human-robot co-transportation and rehabilitation through robotic exoskeletons. Identifying subject-specific Hill-type muscle model parameters and bone dynamic parameters is essential for a personalized musculoskeletal model, but very challenging due to the difficulty of measuring the internal biomechanical variables in vivo directly, especially the joint torques. In this paper, we propose using Differentiable MusculoSkeletal Model (Diff-MSM) to simultaneously identify its muscle and bone parameters with an end-to-end automatic differentiation technique differentiating from the measurable muscle activation, through the joint torque, to the resulting observable motion without the need to measure the internal joint torques. Through extensive comparative simulations, the results manifested that our proposed method significantly outperformed the state-of-the-art baseline methods, especially in terms of accurate estimation of the muscle parameters (i.e., initial guess sampled from a normal distribution with the mean being the ground truth and the standard deviation being 10% of the ground truth could end up with an average of the percentage errors of the estimated values as low as 0.05%). In addition to human musculoskeletal modeling and simulation, the new parameter identification technique with the Diff-MSM has great potential to enable new applications in muscle health monitoring, rehabilitation, and sports science. 

---
# GaitCrafter: Diffusion Model for Biometric Preserving Gait Synthesis 

**Authors**: Sirshapan Mitra, Yogesh S. Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13300)  

**Abstract**: Gait recognition is a valuable biometric task that enables the identification of individuals from a distance based on their walking patterns. However, it remains limited by the lack of large-scale labeled datasets and the difficulty of collecting diverse gait samples for each individual while preserving privacy. To address these challenges, we propose GaitCrafter, a diffusion-based framework for synthesizing realistic gait sequences in the silhouette domain. Unlike prior works that rely on simulated environments or alternative generative models, GaitCrafter trains a video diffusion model from scratch, exclusively on gait silhouette data. Our approach enables the generation of temporally consistent and identity-preserving gait sequences. Moreover, the generation process is controllable-allowing conditioning on various covariates such as clothing, carried objects, and view angle. We show that incorporating synthetic samples generated by GaitCrafter into the gait recognition pipeline leads to improved performance, especially under challenging conditions. Additionally, we introduce a mechanism to generate novel identities-synthetic individuals not present in the original dataset-by interpolating identity embeddings. These novel identities exhibit unique, consistent gait patterns and are useful for training models while maintaining privacy of real subjects. Overall, our work takes an important step toward leveraging diffusion models for high-quality, controllable, and privacy-aware gait data generation. 

---
# Hierarchical Conformal Classification 

**Authors**: Floris den Hengst, Inès Blin, Majid Mohammadi, Syed Ihtesham Hussain Shah, Taraneh Younesian  

**Link**: [PDF](https://arxiv.org/pdf/2508.13288)  

**Abstract**: Conformal prediction (CP) is a powerful framework for quantifying uncertainty in machine learning models, offering reliable predictions with finite-sample coverage guarantees. When applied to classification, CP produces a prediction set of possible labels that is guaranteed to contain the true label with high probability, regardless of the underlying classifier. However, standard CP treats classes as flat and unstructured, ignoring domain knowledge such as semantic relationships or hierarchical structure among class labels. This paper presents hierarchical conformal classification (HCC), an extension of CP that incorporates class hierarchies into both the structure and semantics of prediction sets. We formulate HCC as a constrained optimization problem whose solutions yield prediction sets composed of nodes at different levels of the hierarchy, while maintaining coverage guarantees. To address the combinatorial nature of the problem, we formally show that a much smaller, well-structured subset of candidate solutions suffices to ensure coverage while upholding optimality. An empirical evaluation on three new benchmarks consisting of audio, image, and text data highlights the advantages of our approach, and a user study shows that annotators significantly prefer hierarchical over flat prediction sets. 

---
# ViTAD: Timing Violation-Aware Debugging of RTL Code using Large Language Models 

**Authors**: Wenhao Lv, Yingjie Xia, Xiyuan Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13257)  

**Abstract**: In modern Very Large Scale Integrated (VLSI) circuit design flow, the Register-Transfer Level (RTL) stage presents a critical opportunity for timing optimization. Addressing timing violations at this early stage is essential, as modern systems demand higher speeds, where even minor timing violations can lead to functional failures or system crashes. However, traditional timing optimization heavily relies on manual expertise, requiring engineers to iteratively analyze timing reports and debug. To automate this process, this paper proposes ViTAD, a method that efficiently analyzes the root causes of timing violations and dynamically generates targeted repair strategies. Specifically, we first parse Verilog code and timing reports to construct a Signal Timing Dependency Graph (STDG). Based on the STDG, we perform violation path analysis and use large language models (LLMs) to infer the root causes of violations. Finally, by analyzing the causes of violations, we selectively retrieve relevant debugging knowledge from a domain-specific knowledge base to generate customized repair solutions. To evaluate the effectiveness of our method, we construct a timing violation dataset based on real-world open-source projects. This dataset contains 54 cases of violations. Experimental results show that our method achieves a 73.68% success rate in repairing timing violations, while the baseline using only LLM is 54.38%. Our method improves the success rate by 19.30%. 

---
# Goal-Directedness is in the Eye of the Beholder 

**Authors**: Nina Rajcic, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.13247)  

**Abstract**: Our ability to predict the behavior of complex agents turns on the attribution of goals. Probing for goal-directed behavior comes in two flavors: Behavioral and mechanistic. The former proposes that goal-directedness can be estimated through behavioral observation, whereas the latter attempts to probe for goals in internal model states. We work through the assumptions behind both approaches, identifying technical and conceptual problems that arise from formalizing goals in agent systems. We arrive at the perhaps surprising position that goal-directedness cannot be measured objectively. We outline new directions for modeling goal-directedness as an emergent property of dynamic, multi-agent systems. 

---
# Involuntary Jailbreak 

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.13246)  

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future. 

---
# Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis 

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.13240)  

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis. 

---
# Uncertainty-Aware Learning Policy for Reliable Pulmonary Nodule Detection on Chest X-Ray 

**Authors**: Hyeonjin Choi, Jinse Kim, Dong-yeon Yoo, Ju-sung Sun, Jung-won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13236)  

**Abstract**: Early detection and rapid intervention of lung cancer are crucial. Nonetheless, ensuring an accurate diagnosis is challenging, as physicians' ability to interpret chest X-rays varies significantly depending on their experience and degree of fatigue. Although medical AI has been rapidly advancing to assist in diagnosis, physicians' trust in such systems remains limited, preventing widespread clinical adoption. This skepticism fundamentally stems from concerns about its diagnostic uncertainty. In clinical diagnosis, physicians utilize extensive background knowledge and clinical experience. In contrast, medical AI primarily relies on repetitive learning of the target lesion to generate diagnoses based solely on that data. In other words, medical AI does not possess sufficient knowledge to render a diagnosis, leading to diagnostic uncertainty. Thus, this study suggests an Uncertainty-Aware Learning Policy that can address the issue of knowledge deficiency by learning the physicians' background knowledge alongside the Chest X-ray lesion information. We used 2,517 lesion-free images and 656 nodule images, all obtained from Ajou University Hospital. The proposed model attained 92% (IoU 0.2 / FPPI 2) with a 10% enhancement in sensitivity compared to the baseline model while also decreasing entropy as a measure of uncertainty by 0.2. 

---
# The Role of AI in Facilitating Interdisciplinary Collaboration: Evidence from AlphaFold 

**Authors**: Naixuan Zhao, Chunli Wei, Xinyan Zhang, Jiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13234)  

**Abstract**: The acceleration of artificial intelligence (AI) in science is recognized and many scholars have begun to explore its role in interdisciplinary collaboration. However, the mechanisms and extent of this impact are still unclear. This study, using AlphaFold's impact on structural biologists, examines how AI technologies influence interdisciplinary collaborative patterns. By analyzing 1,247 AlphaFold-related papers and 7,700 authors from Scopus, we employ bibliometric analysis and causal inference to compare interdisciplinary collaboration between AlphaFold adopters and non-adopters. Contrary to the widespread belief that AI facilitates interdisciplinary collaboration, our findings show that AlphaFold increased structural biology-computer science collaborations by just 0.48%, with no measurable effect on other disciplines. Specifically, AI creates interdisciplinary collaboration demands with specific disciplines due to its technical characteristics, but this demand is weakened by technological democratization and other factors. These findings demonstrate that artificial intelligence (AI) alone has limited efficacy in bridging disciplinary divides or fostering meaningful interdisciplinary collaboration. 

---
# Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System 

**Authors**: Yunhua Fang, Rui Xie, Asad Ul Haq, Linsen Ma, Kaoutar El Maghraoui, Naigang Wang, Meng Wang, Liu Liu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13231)  

**Abstract**: Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference. 

---
# PreSem-Surf: RGB-D Surface Reconstruction with Progressive Semantic Modeling and SG-MLP Pre-Rendering Mechanism 

**Authors**: Yuyan Ye, Hang Xu, Yanghang Huang, Jiali Huang, Qian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13228)  

**Abstract**: This paper proposes PreSem-Surf, an optimized method based on the Neural Radiance Field (NeRF) framework, capable of reconstructing high-quality scene surfaces from RGB-D sequences in a short time. The method integrates RGB, depth, and semantic information to improve reconstruction performance. Specifically, a novel SG-MLP sampling structure combined with PR-MLP (Preconditioning Multilayer Perceptron) is introduced for voxel pre-rendering, allowing the model to capture scene-related information earlier and better distinguish noise from local details. Furthermore, progressive semantic modeling is adopted to extract semantic information at increasing levels of precision, reducing training time while enhancing scene understanding. Experiments on seven synthetic scenes with six evaluation metrics show that PreSem-Surf achieves the best performance in C-L1, F-score, and IoU, while maintaining competitive results in NC, Accuracy, and Completeness, demonstrating its effectiveness and practical applicability. 

---
# MIRAGE: Towards AI-Generated Image Detection in the Wild 

**Authors**: Cheng Xia, Manxi Lin, Jiexiang Tan, Xiaoxiong Du, Yang Qiu, Junjun Zheng, Xiangheng Kong, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13223)  

**Abstract**: The spreading of AI-generated images (AIGI), driven by advances in generative AI, poses a significant threat to information security and public trust. Existing AIGI detectors, while effective against images in clean laboratory settings, fail to generalize to in-the-wild scenarios. These real-world images are noisy, varying from ``obviously fake" images to realistic ones derived from multiple generative models and further edited for quality control. We address in-the-wild AIGI detection in this paper. We introduce Mirage, a challenging benchmark designed to emulate the complexity of in-the-wild AIGI. Mirage is constructed from two sources: (1) a large corpus of Internet-sourced AIGI verified by human experts, and (2) a synthesized dataset created through the collaboration between multiple expert generators, closely simulating the realistic AIGI in the wild. Building on this benchmark, we propose Mirage-R1, a vision-language model with heuristic-to-analytic reasoning, a reflective reasoning mechanism for AIGI detection. Mirage-R1 is trained in two stages: a supervised-fine-tuning cold start, followed by a reinforcement learning stage. By further adopting an inference-time adaptive thinking strategy, Mirage-R1 is able to provide either a quick judgment or a more robust and accurate conclusion, effectively balancing inference speed and performance. Extensive experiments show that our model leads state-of-the-art detectors by 5% and 10% on Mirage and the public benchmark, respectively. The benchmark and code will be made publicly available. 

---
# MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols 

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13220)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, and attack scripts to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers. 

---
# Deep Graph Neural Point Process For Learning Temporal Interactive Networks 

**Authors**: Su Chen, Xiaohua Qi, Xixun Lin, Yanmin Shang, Xiaolin Xu, Yangxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13219)  

**Abstract**: Learning temporal interaction networks(TIN) is previously regarded as a coarse-grained multi-sequence prediction problem, ignoring the network topology structure influence. This paper addresses this limitation and a Deep Graph Neural Point Process(DGNPP) model for TIN is proposed. DGNPP consists of two key modules: the Node Aggregation Layer and the Self Attentive Layer. The Node Aggregation Layer captures topological structures to generate static representation for users and items, while the Self Attentive Layer dynamically updates embeddings over time. By incorporating both dynamic and static embeddings into the event intensity function and optimizing the model via maximum likelihood estimation, DGNPP predicts events and occurrence time effectively. Experimental evaluations on three public datasets demonstrate that DGNPP achieves superior performance in event prediction and time prediction tasks with high efficiency, significantly outperforming baseline models and effectively mitigating the limitations of prior approaches. 

---
# Too Easily Fooled? Prompt Injection Breaks LLMs on Frustratingly Simple Multiple-Choice Questions 

**Authors**: Xuyang Guo, Zekai Huang, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13214)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong emergent abilities in complex reasoning and zero-shot generalization, showing unprecedented potential for LLM-as-a-judge applications in education, peer review, and data quality evaluation. However, their robustness under prompt injection attacks, where malicious instructions are embedded into the content to manipulate outputs, remains a significant concern. In this work, we explore a frustratingly simple yet effective attack setting to test whether LLMs can be easily misled. Specifically, we evaluate LLMs on basic arithmetic questions (e.g., "What is 3 + 2?") presented as either multiple-choice or true-false judgment problems within PDF files, where hidden prompts are injected into the file. Our results reveal that LLMs are indeed vulnerable to such hidden prompt injection attacks, even in these trivial scenarios, highlighting serious robustness risks for LLM-as-a-judge applications. 

---
# Research on Conversational Recommender System Considering Consumer Types 

**Authors**: Yaying Luo, Hui Fang, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13209)  

**Abstract**: Conversational Recommender Systems (CRS) provide personalized services through multi-turn interactions, yet most existing methods overlook users' heterogeneous decision-making styles and knowledge levels, which constrains both accuracy and efficiency. To address this gap, we propose CT-CRS (Consumer Type-Enhanced Conversational Recommender System), a framework that integrates consumer type modeling into dialogue recommendation. Based on consumer type theory, we define four user categories--dependent, efficient, cautious, and expert--derived from two dimensions: decision-making style (maximizers vs. satisficers) and knowledge level (high vs. low). CT-CRS employs interaction histories and fine-tunes the large language model to automatically infer user types in real time, avoiding reliance on static questionnaires. We incorporate user types into state representation and design a type-adaptive policy that dynamically adjusts recommendation granularity, diversity, and attribute query complexity. To further optimize the dialogue policy, we adopt Inverse Reinforcement Learning (IRL), enabling the agent to approximate expert-like strategies conditioned on consumer type. Experiments on LastFM, Amazon-Book, and Yelp show that CTCRS improves recommendation success rate and reduces interaction turns compared to strong baselines. Ablation studies confirm that both consumer type modeling and IRL contribute significantly to performance gains. These results demonstrate that CT-CRS offers a scalable and interpretable solution for enhancing CRS personalization through the integration of psychological modeling and advanced policy optimization. 

---
# Utilizing the RAIN method and Graph SAGE Model to Identify Effective Drug Combinations for Gastric Neoplasm Treatment 

**Authors**: S. Z. Pirasteh, Ali A. Kiaei, Mahnaz Bush, Sabra Moghadam, Raha Aghaei, Behnaz Sadeghigol  

**Link**: [PDF](https://arxiv.org/pdf/2508.13207)  

**Abstract**: Background: Gastric neoplasm, primarily adenocarcinoma, is an aggressive cancer with high mortality, often diagnosed late, leading to complications like metastasis. Effective drug combinations are vital to address disease heterogeneity, enhance efficacy, reduce resistance, and improve patient outcomes. Methods: The RAIN method integrated Graph SAGE to propose drug combinations, using a graph model with p-value-weighted edges connecting drugs, genes, and proteins. NLP and systematic literature review (PubMed, Scopus, etc.) validated proposed drugs, followed by network meta-analysis to assess efficacy, implemented in Python. Results: Oxaliplatin, fluorouracil, and trastuzumab were identified as effective, supported by 61 studies. Fluorouracil alone had a p-value of 0.0229, improving to 0.0099 with trastuzumab, and 0.0069 for the triple combination, indicating superior efficacy. Conclusion: The RAIN method, combining AI and network meta-analysis, effectively identifies optimal drug combinations for gastric neoplasm, offering a promising strategy to enhance treatment outcomes and guide health policy. 

---
# Benchmarking LLM-based Agents for Single-cell Omics Analysis 

**Authors**: Yang Liu, Lu Zhou, Ruikun He, Rongbo Shen, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13201)  

**Abstract**: The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology. 

---
# The Rise of Generative AI for Metal-Organic Framework Design and Synthesis 

**Authors**: Chenru Duan, Aditya Nandy, Shyam Chand Pal, Xin Yang, Wenhao Gao, Yuanqi Du, Hendrik Kraß, Yeonghun Kang, Varinia Bernales, Zuyang Ye, Tristan Pyle, Ray Yang, Zeqi Gu, Philippe Schwaller, Shengqian Ma, Shijing Sun, Alán Aspuru-Guzik, Seyed Mohamad Moosavi, Robert Wexler, Zhiling Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13197)  

**Abstract**: Advances in generative artificial intelligence are transforming how metal-organic frameworks (MOFs) are designed and discovered. This Perspective introduces the shift from laborious enumeration of MOF candidates to generative approaches that can autonomously propose and synthesize in the laboratory new porous reticular structures on demand. We outline the progress of employing deep learning models, such as variational autoencoders, diffusion models, and large language model-based agents, that are fueled by the growing amount of available data from the MOF community and suggest novel crystalline materials designs. These generative tools can be combined with high-throughput computational screening and even automated experiments to form accelerated, closed-loop discovery pipelines. The result is a new paradigm for reticular chemistry in which AI algorithms more efficiently direct the search for high-performance MOF materials for clean air and energy applications. Finally, we highlight remaining challenges such as synthetic feasibility, dataset diversity, and the need for further integration of domain knowledge. 

---
# Contextual Attention-Based Multimodal Fusion of LLM and CNN for Sentiment Analysis 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2508.13196)  

**Abstract**: This paper introduces a novel approach for multimodal sentiment analysis on social media, particularly in the context of natural disasters, where understanding public sentiment is crucial for effective crisis management. Unlike conventional methods that process text and image modalities separately, our approach seamlessly integrates Convolutional Neural Network (CNN) based image analysis with Large Language Model (LLM) based text processing, leveraging Generative Pre-trained Transformer (GPT) and prompt engineering to extract sentiment relevant features from the CrisisMMD dataset. To effectively model intermodal relationships, we introduce a contextual attention mechanism within the fusion process. Leveraging contextual-attention layers, this mechanism effectively captures intermodality interactions, enhancing the model's comprehension of complex relationships between textual and visual data. The deep neural network architecture of our model learns from these fused features, leading to improved accuracy compared to existing baselines. Experimental results demonstrate significant advancements in classifying social media data into informative and noninformative categories across various natural disasters. Our model achieves a notable 2.43% increase in accuracy and 5.18% in F1-score, highlighting its efficacy in processing complex multimodal data. Beyond quantitative metrics, our approach provides deeper insight into the sentiments expressed during crises. The practical implications extend to real time disaster management, where enhanced sentiment analysis can optimize the accuracy of emergency interventions. By bridging the gap between multimodal analysis, LLM powered text understanding, and disaster response, our work presents a promising direction for Artificial Intelligence (AI) driven crisis management solutions. Keywords: 

---
# Preference Models assume Proportional Hazards of Utilities 

**Authors**: Chirag Nagpal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13189)  

**Abstract**: Approaches for estimating preferences from human annotated data typically involves inducing a distribution over a ranked list of choices such as the Plackett-Luce model. Indeed, modern AI alignment tools such as Reward Modelling and Direct Preference Optimization are based on the statistical assumptions posed by the Plackett-Luce model. In this paper, I will connect the Plackett-Luce model to another classical and well known statistical model, the Cox Proportional Hazards model and attempt to shed some light on the implications of the connection therein. 

---
# Combating Homelessness Stigma with LLMs: A New Multi-Modal Dataset for Bias Detection 

**Authors**: Jonathan A. Karr Jr., Benjamin F. Herbst, Ting Hua, Matthew Hauenstein, Georgina Curto, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2508.13187)  

**Abstract**: Homelessness is a persistent social challenge, impacting millions worldwide. Over 770,000 people experienced homelessness in the U.S. in 2024. Social stigmatization is a significant barrier to alleviation, shifting public perception, and influencing policymaking. Given that online and city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases. This research contributes to alleviating homelessness by acting on public opinion. It introduces novel methods, building on natural language processing (NLP) and large language models (LLMs), to identify and measure PEH social bias expressed in digital spaces. We present a new, manually-annotated multi-modal dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across 10 U.S. cities. This unique dataset provides evidence of the typologies of homelessness bias described in the literature. In order to scale up and automate the detection of homelessness bias online, we evaluate LLMs as classifiers. We applied both zero-shot and few-shot classification techniques to this data. We utilized local LLMs (Llama 3.2 3B Instruct, Qwen 2.5 7B Instruct, and Phi4 Instruct Mini) as well as closed-source API models (GPT-4.1, Gemini 2.5 Pro, and Grok-4). Our findings reveal that although there are significant inconsistencies in local LLM zero-shot classification, the in-context learning classification scores of local LLMs approach the classification scores of closed-source LLMs. Furthermore, LLMs outperform BERT when averaging across all categories. This work aims to raise awareness about the pervasive bias against PEH, develop new indicators to inform policy, and ultimately enhance the fairness and ethical application of Generative AI technologies. 

---
# MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents 

**Authors**: Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, Chuanhao Li, Jiayi Tian, Chenchen Zhang, Tianhao Peng, Yancheng He, Jihao Gu, Yuanxing Zhang, Jian Yang, Ge Zhang, Wenhao Huang, Wangchunshu Zhou, Zhaoxiang Zhang, Ruizhe Ding, Shilei Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13186)  

**Abstract**: AI agents with advanced reasoning and tool use capabilities have demonstrated impressive performance in web browsing for deep search. While existing benchmarks such as BrowseComp evaluate these browsing abilities, they primarily focus on textual information, overlooking the prevalence of multimodal content. To bridge this gap, we introduce MM-BrowseComp, a novel benchmark comprising 224 challenging, hand-crafted questions specifically designed to assess agents' multimodal retrieval and reasoning capabilities. These questions often incorporate images in prompts, and crucial information encountered during the search and reasoning process may also be embedded within images or videos on webpages. Consequently, methods relying solely on text prove insufficient for our benchmark. Additionally, we provide a verified checklist for each question, enabling fine-grained analysis of multimodal dependencies and reasoning paths. Our comprehensive evaluation of state-of-the-art models on MM-BrowseComp reveals that even top models like OpenAI o3 with tools achieve only 29.02\% accuracy, highlighting the suboptimal multimodal capabilities and lack of native multimodal reasoning in current models. 

---
# Using Artificial Intuition in Distinct, Minimalist Classification of Scientific Abstracts for Management of Technology Portfolios 

**Authors**: Prateek Ranka, Fred Morstatter, Andrea Belz, Alexandra Graddy-Reed  

**Link**: [PDF](https://arxiv.org/pdf/2508.13182)  

**Abstract**: Classification of scientific abstracts is useful for strategic activities but challenging to automate because the sparse text provides few contextual clues. Metadata associated with the scientific publication can be used to improve performance but still often requires a semi-supervised setting. Moreover, such schemes may generate labels that lack distinction -- namely, they overlap and thus do not uniquely define the abstract. In contrast, experts label and sort these texts with ease. Here we describe an application of a process we call artificial intuition to replicate the expert's approach, using a Large Language Model (LLM) to generate metadata. We use publicly available abstracts from the United States National Science Foundation to create a set of labels, and then we test this on a set of abstracts from the Chinese National Natural Science Foundation to examine funding trends. We demonstrate the feasibility of this method for research portfolio management, technology scouting, and other strategic activities. 

---
# Toward an African Agenda for AI Safety 

**Authors**: Samuel T. Segun, Rachel Adams, Ana Florido, Scott Timcke, Jonathan Shock, Leah Junck, Fola Adeleke, Nicolas Grossman, Ayantola Alayande, Jerry John Kponyo, Matthew Smith, Dickson Marfo Fosu, Prince Dawson Tetteh, Juliet Arthur, Stephanie Kasaon, Odilile Ayodele, Laetitia Badolo, Paul Plantinga, Michael Gastrow, Sumaya Nur Adan, Joanna Wiaterek, Cecil Abungu, Kojo Apeagyei, Luise Eder, Tegawende Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2508.13179)  

**Abstract**: This paper maps Africa's distinctive AI risk profile, from deepfake fuelled electoral interference and data colonial dependency to compute scarcity, labour disruption and disproportionate exposure to climate driven environmental costs. While major benefits are promised to accrue, the availability, development and adoption of AI also mean that African people and countries face particular AI safety risks, from large scale labour market disruptions to the nefarious use of AI to manipulate public opinion. To date, African perspectives have not been meaningfully integrated into global debates and processes regarding AI safety, leaving African stakeholders with limited influence over the emerging global AI safety governance agenda. While there are Computer Incident Response Teams on the continent, none hosts a dedicated AI Safety Institute or office. We propose a five-point action plan centred on (i) a policy approach that foregrounds the protection of the human rights of those most vulnerable to experiencing the harmful socio-economic effects of AI; (ii) the establishment of an African AI Safety Institute; (iii) promote public AI literacy and awareness; (iv) development of early warning system with inclusive benchmark suites for 25+ African languages; and (v) an annual AU-level AI Safety & Security Forum. 

---
# White-Box Reasoning: Synergizing LLM Strategy and gm/Id Data for Automated Analog Circuit Design 

**Authors**: Jianqiu Chen, Siqi Li, Xu He  

**Link**: [PDF](https://arxiv.org/pdf/2508.13172)  

**Abstract**: Analog IC design is a bottleneck due to its reliance on experience and inefficient simulations, as traditional formulas fail in advanced nodes. Applying Large Language Models (LLMs) directly to this problem risks mere "guessing" without engineering principles. We present a "synergistic reasoning" framework that integrates an LLM's strategic reasoning with the physical precision of the gm/Id methodology. By empowering the LLM with gm/Id lookup tables, it becomes a quantitative, data-driven design partner.
We validated this on a two-stage op-amp, where our framework enabled the Gemini model to meet all TT corner specs in 5 iterations and extended optimization to all PVT corners. A crucial ablation study proved gm/Id data is key for this efficiency and precision; without it, the LLM is slower and deviates. Compared to a senior engineer's design, our framework achieves quasi-expert quality with an order-of-magnitude improvement in efficiency. This work validates a path for true analog design automation by combining LLM reasoning with scientific circuit design methodologies. 

---
# Sustainable AI Training via Hardware-Software Co-Design on NVIDIA, AMD, and Emerging GPU Architectures 

**Authors**: Yashasvi Makin, Rahul Maliakkal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13163)  

**Abstract**: In particular, large-scale deep learning and artificial intelligence model training uses a lot of computational power and energy, so it poses serious sustainability issues. The fast rise in model complexity has resulted in exponential increases in energy consumption, increasing the demand for techniques maximizing computational efficiency and lowering environmental impact. This work explores environmentally driven performance optimization methods especially intended for advanced GPU architectures from NVIDIA, AMD, and other emerging GPU architectures. Our main focus is on investigating hardware-software co-design techniques meant to significantly increase memory-level and kernel-level operations, so improving performance-per-watt measures. Our thorough research encompasses evaluations of specialized tensor and matrix cores, advanced memory optimization methods, and creative integration approaches that taken together result in notable energy efficiency increases. We also discuss important software-level optimizations that augment hardware capability including mixed-precision arithmetic, advanced energy-aware scheduling algorithms, and compiler-driven kernel enhancements. Moreover, we methodically point out important research gaps and suggest future directions necessary to create really sustainable artificial intelligence systems. This paper emphasizes how major increases in training efficiency can be obtained by co-design of hardware and software, so lowering the environmental impact of artificial intelligence without compromising performance. To back up our analysis, we use real-world case studies from top companies like Meta, Google, Amazon, and others that show how these sustainable AI training methods are used in the real world. 

---
# Piano: A Multi-Constraint Pin Assignment-Aware Floorplanner 

**Authors**: Zhexuan Xu, Kexin Zhou, Jie Wang, Zijie Geng, Siyuan Xu, Shixiong Kai, Mingxuan Yuan, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13161)  

**Abstract**: Floorplanning is a critical step in VLSI physical design, increasingly complicated by modern constraints such as fixed-outline requirements, whitespace removal, and the presence of pre-placed modules. In addition, the assignment of pins on module boundaries significantly impacts the performance of subsequent stages, including detailed placement and routing. However, traditional floorplanners often overlook pin assignment with modern constraints during the floorplanning stage. In this work, we introduce Piano, a floorplanning framework that simultaneously optimizes module placement and pin assignment under multiple constraints. Specifically, we construct a graph based on the geometric relationships among modules and their netlist connections, then iteratively search for shortest paths to determine pin assignments. This graph-based method also enables accurate evaluation of feedthrough and unplaced pins, thereby guiding overall layout quality. To further improve the design, we adopt a whitespace removal strategy and employ three local optimizers to enhance layout metrics under multi-constraint scenarios. Experimental results on widely used benchmark circuits demonstrate that Piano achieves an average 6.81% reduction in HPWL, a 13.39% decrease in feedthrough wirelength, a 16.36% reduction in the number of feedthrough modules, and a 21.21% drop in unplaced pins, while maintaining zero whitespace. 

---
# Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists 

**Authors**: Haohang Xu, Chengjie Liu, Qihang Wang, Wenhao Huang, Yongjian Xu, Weiyu Chen, Anlan Peng, Zhijun Li, Bo Li, Lei Qi, Jun Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13157)  

**Abstract**: Large Language Model (LLM) exhibits great potential in designing of analog integrated circuits (IC) because of its excellence in abstraction and generalization for knowledge. However, further development of LLM-based analog ICs heavily relies on textual description of analog ICs, while existing analog ICs are mostly illustrated in image-based circuit diagrams rather than text-based netlists. Converting circuit diagrams to netlists help LLMs to enrich the knowledge of analog IC. Nevertheless, previously proposed conversion frameworks face challenges in further application because of limited support of image styles and circuit elements. Up to now, it still remains a challenging task to effectively convert complex circuit diagrams into netlists. To this end, this paper constructs and opensources a new dataset with rich styles of circuit diagrams as well as balanced distribution of simple and complex analog ICs. And a hybrid framework, named Image2Net, is proposed for practical conversion from circuit diagrams to netlists. The netlist edit distance (NED) is also introduced to precisely assess the difference between the converted netlists and ground truth. Based on our benchmark, Image2Net achieves 80.77\% successful rate, which is 34.62\%-45.19\% higher than previous works. Specifically, the proposed work shows 0.116 averaged NED, which is 62.1\%-69.6\% lower than state-of-the-arts. 

---
# EvoVerilog: Large Langugage Model Assisted Evolution of Verilog Code 

**Authors**: Ping Guo, Yiting Wang, Wanghao Ye, Yexiao He, Ziyao Wang, Xiaopeng Dai, Ang Li, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13156)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential in automating the generation of Verilog hardware description language code for hardware design. This automation is critical to reducing human effort in the complex and error-prone process of hardware design.
However, existing approaches predominantly rely on human intervention and fine-tuning using curated datasets, limiting their scalability in automated design workflows.
Although recent iterative search techniques have emerged, they often fail to explore diverse design solutions and may underperform simpler approaches such as repeated prompting.
To address these limitations, we introduce EvoVerilog, a novel framework that combines the reasoning capabilities of LLMs with evolutionary algorithms to automatically generate and refine Verilog code.
EvoVerilog utilizes a multiobjective, population-based search strategy to explore a wide range of design possibilities without requiring human intervention.
Extensive experiments demonstrate that EvoVerilog achieves state-of-the-art performance, with pass@10 scores of 89.1 and 80.2 on the VerilogEval-Machine and VerilogEval-Human benchmarks, respectively. Furthermore, the framework showcases its ability to explore diverse designs by simultaneously generating a variety of functional Verilog code while optimizing resource utilization. 

---
# Uncovering Emergent Physics Representations Learned In-Context by Large Language Models 

**Authors**: Yeongwoo Song, Jaeyong Bae, Dong-Kyum Kim, Hawoong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12448)  

**Abstract**: Large language models (LLMs) exhibit impressive in-context learning (ICL) abilities, enabling them to solve wide range of tasks via textual prompts alone. As these capabilities advance, the range of applicable domains continues to expand significantly. However, identifying the precise mechanisms or internal structures within LLMs that allow successful ICL across diverse, distinct classes of tasks remains elusive. Physics-based tasks offer a promising testbed for probing this challenge. Unlike synthetic sequences such as basic arithmetic or symbolic equations, physical systems provide experimentally controllable, real-world data based on structured dynamics grounded in fundamental principles. This makes them particularly suitable for studying the emergent reasoning behaviors of LLMs in a realistic yet tractable setting. Here, we mechanistically investigate the ICL ability of LLMs, especially focusing on their ability to reason about physics. Using a dynamics forecasting task in physical systems as a proxy, we evaluate whether LLMs can learn physics in context. We first show that the performance of dynamics forecasting in context improves with longer input contexts. To uncover how such capability emerges in LLMs, we analyze the model's residual stream activations using sparse autoencoders (SAEs). Our experiments reveal that the features captured by SAEs correlate with key physical variables, such as energy. These findings demonstrate that meaningful physical concepts are encoded within LLMs during in-context learning. In sum, our work provides a novel case study that broadens our understanding of how LLMs learn in context. 

---
# TaoSR1: The Thinking Model for E-commerce Relevance Search 

**Authors**: Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12365)  

**Abstract**: Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification. 

---
# Preliminary suggestions for rigorous GPAI model evaluations 

**Authors**: Patricia Paskov, Michael J. Byun, Kevin Wei, Toby Webster  

**Link**: [PDF](https://arxiv.org/pdf/2508.00875)  

**Abstract**: This document presents a preliminary compilation of general-purpose AI (GPAI) evaluation practices that may promote internal validity, external validity and reproducibility. It includes suggestions for human uplift studies and benchmark evaluations, as well as cross-cutting suggestions that may apply to many different evaluation types. Suggestions are organised across four stages in the evaluation life cycle: design, implementation, execution and documentation. Drawing from established practices in machine learning, statistics, psychology, economics, biology and other fields recognised to have important lessons for AI evaluation, these suggestions seek to contribute to the conversation on the nascent and evolving field of the science of GPAI evaluations. The intended audience of this document includes providers of GPAI models presenting systemic risk (GPAISR), for whom the EU AI Act lays out specific evaluation requirements; third-party evaluators; policymakers assessing the rigour of evaluations; and academic researchers developing or conducting GPAI evaluations. 

---
