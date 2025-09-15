# Mutual Information Tracks Policy Coherence in Reinforcement Learning 

**Authors**: Cameron Reid, Wael Hafez, Amirhossein Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2509.10423)  

**Abstract**: Reinforcement Learning (RL) agents deployed in real-world environments face degradation from sensor faults, actuator wear, and environmental shifts, yet lack intrinsic mechanisms to detect and diagnose these failures. We present an information-theoretic framework that reveals both the fundamental dynamics of RL and provides practical methods for diagnosing deployment-time anomalies. Through analysis of state-action mutual information patterns in a robotic control task, we first demonstrate that successful learning exhibits characteristic information signatures: mutual information between states and actions steadily increases from 0.84 to 2.83 bits (238% growth) despite growing state entropy, indicating that agents develop increasingly selective attention to task-relevant patterns. Intriguingly, states, actions and next states joint mutual information, MI(S,A;S'), follows an inverted U-curve, peaking during early learning before declining as the agent specializes suggesting a transition from broad exploration to efficient exploitation. More immediately actionable, we show that information metrics can differentially diagnose system failures: observation-space, i.e., states noise (sensor faults) produces broad collapses across all information channels with pronounced drops in state-action coupling, while action-space noise (actuator faults) selectively disrupts action-outcome predictability while preserving state-action relationships. This differential diagnostic capability demonstrated through controlled perturbation experiments enables precise fault localization without architectural modifications or performance degradation. By establishing information patterns as both signatures of learning and diagnostic for system health, we provide the foundation for adaptive RL systems capable of autonomous fault detection and policy adjustment based on information-theoretic principles. 

---
# Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems 

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10401)  

**Abstract**: Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution. 

---
# State Algebra for Propositional Logic 

**Authors**: Dmitry Lesnik, Tobias Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10326)  

**Abstract**: This paper presents State Algebra, a novel framework designed to represent and manipulate propositional logic using algebraic methods. The framework is structured as a hierarchy of three representations: Set, Coordinate, and Row Decomposition. These representations anchor the system in well-known semantics while facilitating the computation using a powerful algebraic engine. A key aspect of State Algebra is its flexibility in representation. We show that although the default reduction of a state vector is not canonical, a unique canonical form can be obtained by applying a fixed variable order during the reduction process. This highlights a trade-off: by foregoing guaranteed canonicity, the framework gains increased flexibility, potentially leading to more compact representations of certain classes of problems. We explore how this framework provides tools to articulate both search-based and knowledge compilation algorithms and discuss its natural extension to probabilistic logic and Weighted Model Counting. 

---
# The Morality of Probability: How Implicit Moral Biases in LLMs May Shape the Future of Human-AI Symbiosis 

**Authors**: Eoin O'Doherty, Nicole Weinrauch, Andrew Talone, Uri Klempner, Xiaoyuan Yi, Xing Xie, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10297)  

**Abstract**: Artificial intelligence (AI) is advancing at a pace that raises urgent questions about how to align machine decision-making with human moral values. This working paper investigates how leading AI systems prioritize moral outcomes and what this reveals about the prospects for human-AI symbiosis. We address two central questions: (1) What moral values do state-of-the-art large language models (LLMs) implicitly favour when confronted with dilemmas? (2) How do differences in model architecture, cultural origin, and explainability affect these moral preferences? To explore these questions, we conduct a quantitative experiment with six LLMs, ranking and scoring outcomes across 18 dilemmas representing five moral frameworks. Our findings uncover strikingly consistent value biases. Across all models, Care and Virtue values outcomes were rated most moral, while libertarian choices were consistently penalized. Reasoning-enabled models exhibited greater sensitivity to context and provided richer explanations, whereas non-reasoning models produced more uniform but opaque judgments. This research makes three contributions: (i) Empirically, it delivers a large-scale comparison of moral reasoning across culturally distinct LLMs; (ii) Theoretically, it links probabilistic model behaviour with underlying value encodings; (iii) Practically, it highlights the need for explainability and cultural awareness as critical design principles to guide AI toward a transparent, aligned, and symbiotic future. 

---
# Investigating Language Model Capabilities to Represent and Process Formal Knowledge: A Preliminary Study to Assist Ontology Engineering 

**Authors**: Hanna Abi Akl  

**Link**: [PDF](https://arxiv.org/pdf/2509.10249)  

**Abstract**: Recent advances in Language Models (LMs) have failed to mask their shortcomings particularly in the domain of reasoning. This limitation impacts several tasks, most notably those involving ontology engineering. As part of a PhD research, we investigate the consequences of incorporating formal methods on the performance of Small Language Models (SLMs) on reasoning tasks. Specifically, we aim to orient our work toward using SLMs to bootstrap ontology construction and set up a series of preliminary experiments to determine the impact of expressing logical problems with different grammars on the performance of SLMs on a predefined reasoning task. Our findings show that it is possible to substitute Natural Language (NL) with a more compact logical language while maintaining a strong performance on reasoning tasks and hope to use these results to further refine the role of SLMs in ontology engineering. 

---
# Compartmentalised Agentic Reasoning for Clinical NLI 

**Authors**: Maël Jullien, Lei Xu, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2509.10222)  

**Abstract**: A common assumption holds that scaling data and parameters yields increasingly structured, generalisable internal representations. We interrogate this assumption in clinical natural language inference (NLI) by adopting a benchmark decomposed into four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction, and introducing CARENLI, a Compartmentalised Agentic Reasoning for Clinical NLI that separates knowledge access from principled inference. CARENLI routes each premise, statement pair to a family specific solver and enforces auditable procedures via a planner, verifier, and refiner.
Across four LLMs, CARENLI improves fidelity by up to 42 points, reaching 98.0% in Causal Attribution and 81.2% in Risk State Abstraction. Verifiers flag violations with near-ceiling reliability, while refiners correct a substantial share of epistemic errors. Remaining failures cluster in routing, identifying family classification as the main bottleneck. These results show that LLMs often retain relevant facts but default to heuristics when inference is underspecified, a dissociation CARENLI makes explicit while offering a framework for safer, auditable reasoning. 

---
# Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction 

**Authors**: Marko Petković, Vlado Menkovski, Sofía Calero  

**Link**: [PDF](https://arxiv.org/pdf/2509.10210)  

**Abstract**: Automated characterization of porous materials has the potential to accelerate materials discovery, but it remains limited by the complexity of simulation setup and force field selection. We propose a multi-agent framework in which LLM-based agents can autonomously understand a characterization task, plan appropriate simulations, assemble relevant force fields, execute them and interpret their results to guide subsequent steps. As a first step toward this vision, we present a multi-agent system for literature-informed force field extraction and automated RASPA simulation setup. Initial evaluations demonstrate high correctness and reproducibility, highlighting this approach's potential to enable fully autonomous, scalable materials characterization. 

---
# Online Robust Planning under Model Uncertainty: A Sample-Based Approach 

**Authors**: Tamir Shazman, Idan Lev-Yehudi, Ron Benchetit, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10162)  

**Abstract**: Online planning in Markov Decision Processes (MDPs) enables agents to make sequential decisions by simulating future trajectories from the current state, making it well-suited for large-scale or dynamic environments. Sample-based methods such as Sparse Sampling and Monte Carlo Tree Search (MCTS) are widely adopted for their ability to approximate optimal actions using a generative model. However, in practical settings, the generative model is often learned from limited data, introducing approximation errors that can degrade performance or lead to unsafe behaviors. To address these challenges, Robust MDPs (RMDPs) offer a principled framework for planning under model uncertainty, yet existing approaches are typically computationally intensive and not suited for real-time use. In this work, we introduce Robust Sparse Sampling (RSS), the first online planning algorithm for RMDPs with finite-sample theoretical performance guarantees. Unlike Sparse Sampling, which estimates the nominal value function, RSS computes a robust value function by leveraging the efficiency and theoretical properties of Sample Average Approximation (SAA), enabling tractable robust policy computation in online settings. RSS is applicable to infinite or continuous state spaces, and its sample and computational complexities are independent of the state space size. We provide theoretical performance guarantees and empirically show that RSS outperforms standard Sparse Sampling in environments with uncertain dynamics. 

---
# Virtual Agent Economies 

**Authors**: Nenad Tomasev, Matija Franklin, Joel Z. Leibo, Julian Jacobs, William A. Cunningham, Iason Gabriel, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2509.10147)  

**Abstract**: The rapid adoption of autonomous AI agents is giving rise to a new economic layer where agents transact and coordinate at scales and speeds beyond direct human oversight. We propose the "sandbox economy" as a framework for analyzing this emergent system, characterizing it along two key dimensions: its origins (emergent vs. intentional) and its degree of separateness from the established human economy (permeable vs. impermeable). Our current trajectory points toward a spontaneous emergence of a vast and highly permeable AI agent economy, presenting us with opportunities for an unprecedented degree of coordination as well as significant challenges, including systemic economic risk and exacerbated inequality. Here we discuss a number of possible design choices that may lead to safely steerable AI agent markets. In particular, we consider auction mechanisms for fair resource allocation and preference resolution, the design of AI "mission economies" to coordinate around achieving collective goals, and socio-technical infrastructure needed to ensure trust, safety, and accountability. By doing this, we argue for the proactive design of steerable agent markets to ensure the coming technological shift aligns with humanity's long-term collective flourishing. 

---
# AI Harmonics: a human-centric and harms severity-adaptive AI risk assessment framework 

**Authors**: Sofia Vei, Paolo Giudici, Pavlos Sermpezis, Athena Vakali, Adelaide Emma Bernardelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.10104)  

**Abstract**: The absolute dominance of Artificial Intelligence (AI) introduces unprecedented societal harms and risks. Existing AI risk assessment models focus on internal compliance, often neglecting diverse stakeholder perspectives and real-world consequences. We propose a paradigm shift to a human-centric, harm-severity adaptive approach grounded in empirical incident data. We present AI Harmonics, which includes a novel AI harm assessment metric (AIH) that leverages ordinal severity data to capture relative impact without requiring precise numerical estimates. AI Harmonics combines a robust, generalized methodology with a data-driven, stakeholder-aware framework for exploring and prioritizing AI harms. Experiments on annotated incident data confirm that political and physical harms exhibit the highest concentration and thus warrant urgent mitigation: political harms erode public trust, while physical harms pose serious, even life-threatening risks, underscoring the real-world relevance of our approach. Finally, we demonstrate that AI Harmonics consistently identifies uneven harm distributions, enabling policymakers and organizations to target their mitigation efforts effectively. 

---
# XAgents: A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph 

**Authors**: Hailong Yang, Mingxian Gu, Jianqi Wang, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10054)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly enhanced the capabilities of Multi-Agent Systems (MAS) in supporting humans with complex, real-world tasks. However, MAS still face challenges in effective task planning when handling highly complex tasks with uncertainty, often resulting in misleading or incorrect outputs that hinder task execution. To address this, we propose XAgents, a unified multi-agent cooperative framework built on a multipolar task processing graph and IF-THEN rules. XAgents uses the multipolar task processing graph to enable dynamic task planning and handle task uncertainty. During subtask processing, it integrates domain-specific IF-THEN rules to constrain agent behaviors, while global rules enhance inter-agent collaboration. We evaluate the performance of XAgents across three distinct datasets, demonstrating that it consistently surpasses state-of-the-art single-agent and multi-agent approaches in both knowledge-typed and logic-typed question-answering tasks. The codes for XAgents are available at: this https URL. 

---
# GAMA: A General Anonymizing Multi-Agent System for Privacy Preservation Enhanced by Domain Rules and Disproof Method 

**Authors**: Hailong Yang, Renhuo Zhao, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10018)  

**Abstract**: With the rapid advancement of Large Language Model (LLM), LLM-based agents exhibit exceptional abilities in understanding and generating natural language, facilitating human-like collaboration and information transmission in LLM-based Multi-Agent System (MAS). High-performance LLMs are often hosted on remote servers in public spaces. When tasks involve privacy data, MAS cannot securely utilize these LLMs without implementing privacy-preserving mechanisms. To address this challenge, we propose a General Anonymizing Multi-Agent system (GAMA), which divides the agents' workspace into private and public spaces and protects privacy through the anonymizing mechanism. In the private space, agents handle sensitive data, while in the public space, only anonymized data is utilized. GAMA incorporates two key modules to mitigate semantic loss caused by anonymization: Domain-Rule-based Knowledge Enhancement (DRKE) and Disproof-based Logic Enhancement (DLE). We evaluate GAMA on two public question-answering datasets: Trivia Creative Writing and Logic Grid Puzzle. The results demonstrate that GAMA has superior performance compared to the state-of-the-art models. To further assess its privacy-preserving capabilities, we designed two new datasets: Knowledge Privacy Preservation and Logic Privacy Preservation. The final results highlight GAMA's exceptional effectiveness in both task processing and privacy preservation. 

---
# Evaluation of Black-Box XAI Approaches for Predictors of Values of Boolean Formulae 

**Authors**: Stav Armoni-Friedmann, Hana Chockler, David A. Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2509.09982)  

**Abstract**: Evaluating explainable AI (XAI) approaches is a challenging task in general, due to the subjectivity of explanations. In this paper, we focus on tabular data and the specific use case of AI models predicting the values of Boolean functions. We extend the previous work in this domain by proposing a formal and precise measure of importance of variables based on actual causality, and we evaluate state-of-the-art XAI tools against this measure. We also present a novel XAI tool B-ReX, based on the existing tool ReX, and demonstrate that it is superior to other black-box XAI tools on a large-scale benchmark. Specifically, B-ReX achieves a Jensen-Shannon divergence of 0.072 $\pm$ 0.012 on random 10-valued Boolean formulae 

---
# A Markovian Framing of WaveFunctionCollapse for Procedurally Generating Aesthetically Complex Environments 

**Authors**: Franklin Yiu, Mohan Lu, Nina Li, Kevin Joseph, Tianxu Zhang, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2509.09919)  

**Abstract**: Procedural content generation often requires satisfying both designer-specified objectives and adjacency constraints implicitly imposed by the underlying tile set. To address the challenges of jointly optimizing both constraints and objectives, we reformulate WaveFunctionCollapse (WFC) as a Markov Decision Process (MDP), enabling external optimization algorithms to focus exclusively on objective maximization while leveraging WFC's propagation mechanism to enforce constraint satisfaction. We empirically compare optimizing this MDP to traditional evolutionary approaches that jointly optimize global metrics and local tile placement. Across multiple domains with various difficulties, we find that joint optimization not only struggles as task complexity increases, but consistently underperforms relative to optimization over the WFC-MDP, underscoring the advantages of decoupling local constraint satisfaction from global objective optimization. 

---
# The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science 

**Authors**: Woong Shin, Renan Souza, Daniel Rosendo, Frédéric Suter, Feiyi Wang, Prasanna Balaprakash, Rafael Ferreira da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2509.09915)  

**Abstract**: Modern scientific discovery increasingly requires coordinating distributed facilities and heterogeneous resources, forcing researchers to act as manual workflow coordinators rather than scientists. Advances in AI leading to AI agents show exciting new opportunities that can accelerate scientific discovery by providing intelligence as a component in the ecosystem. However, it is unclear how this new capability would materialize and integrate in the real world. To address this, we propose a conceptual framework where workflows evolve along two dimensions which are intelligence (from static to intelligent) and composition (from single to swarm) to chart an evolutionary path from current workflow management systems to fully autonomous, distributed scientific laboratories. With these trajectories in mind, we present an architectural blueprint that can help the community take the next steps towards harnessing the opportunities in autonomous science with the potential for 100x discovery acceleration and transformational scientific workflows. 

---
# LLMs as Agentic Cooperative Players in Multiplayer UNO 

**Authors**: Yago Romano Matinez, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2509.09867)  

**Abstract**: LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player. 

---
# Towards an AI-based knowledge assistant for goat farmers based on Retrieval-Augmented Generation 

**Authors**: Nana Han, Dong Liu, Tomas Norton  

**Link**: [PDF](https://arxiv.org/pdf/2509.09848)  

**Abstract**: Large language models (LLMs) are increasingly being recognised as valuable knowledge communication tools in many industries. However, their application in livestock farming remains limited, being constrained by several factors not least the availability, diversity and complexity of knowledge sources. This study introduces an intelligent knowledge assistant system designed to support health management in farmed goats. Leveraging the Retrieval-Augmented Generation (RAG), two structured knowledge processing methods, table textualization and decision-tree textualization, were proposed to enhance large language models' (LLMs) understanding of heterogeneous data formats. Based on these methods, a domain-specific goat farming knowledge base was established to improve LLM's capacity for cross-scenario generalization. The knowledge base spans five key domains: Disease Prevention and Treatment, Nutrition Management, Rearing Management, Goat Milk Management, and Basic Farming Knowledge. Additionally, an online search module is integrated to enable real-time retrieval of up-to-date information. To evaluate system performance, six ablation experiments were conducted to examine the contribution of each component. The results demonstrated that heterogeneous knowledge fusion method achieved the best results, with mean accuracies of 87.90% on the validation set and 84.22% on the test set. Across the text-based, table-based, decision-tree based Q&A tasks, accuracy consistently exceeded 85%, validating the effectiveness of structured knowledge fusion within a modular design. Error analysis identified omission as the predominant error category, highlighting opportunities to further improve retrieval coverage and context integration. In conclusion, the results highlight the robustness and reliability of the proposed system for practical applications in goat farming. 

---
# Towards a Common Framework for Autoformalization 

**Authors**: Agnieszka Mensfelt, David Tena Cucala, Santiago Franco, Angeliki Koutsoukou-Argyraki, Vince Trencsenyi, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2509.09810)  

**Abstract**: Autoformalization has emerged as a term referring to the automation of formalization - specifically, the formalization of mathematics using interactive theorem provers (proof assistants). Its rapid development has been driven by progress in deep learning, especially large language models (LLMs). More recently, the term has expanded beyond mathematics to describe the broader task of translating informal input into formal logical representations. At the same time, a growing body of research explores using LLMs to translate informal language into formal representations for reasoning, planning, and knowledge representation - often without explicitly referring to this process as autoformalization. As a result, despite addressing similar tasks, the largely independent development of these research areas has limited opportunities for shared methodologies, benchmarks, and theoretical frameworks that could accelerate progress. The goal of this paper is to review - explicit or implicit - instances of what can be considered autoformalization and to propose a unified framework, encouraging cross-pollination between different fields to advance the development of next generation AI systems. 

---
# A Modular and Multimodal Generative AI Framework for Urban Building Energy Data: Generating Synthetic Homes 

**Authors**: Jackson Eshbaugh, Chetan Tiwari, Jorge Silveyra  

**Link**: [PDF](https://arxiv.org/pdf/2509.09794)  

**Abstract**: Computational models have emerged as powerful tools for energy modeling research, touting scalability and quantitative results. However, these models require a plethora of data, some of which is inaccessible, expensive, or raises privacy concerns. We introduce a modular multimodal framework to produce this data from publicly accessible residential information and images using generative artificial intelligence (AI). Additionally, we provide a pipeline demonstrating this framework, and we evaluate its generative AI components. Our experiments show that our framework's use of AI avoids common issues with generative models. Our framework produces realistic, labeled data. By reducing dependence on costly or restricted data sources, we pave a path towards more accessible and reproducible research. 

---
# How well can LLMs provide planning feedback in grounded environments? 

**Authors**: Yuxuan Li, Victor Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09790)  

**Abstract**: Learning to plan in grounded environments typically requires carefully designed reward functions or high-quality annotated demonstrations. Recent works show that pretrained foundation models, such as large language models (LLMs) and vision language models (VLMs), capture background knowledge helpful for planning, which reduces the amount of reward design and demonstrations needed for policy learning. We evaluate how well LLMs and VLMs provide feedback across symbolic, language, and continuous control environments. We consider prominent types of feedback for planning including binary feedback, preference feedback, action advising, goal advising, and delta action feedback. We also consider inference methods that impact feedback performance, including in-context learning, chain-of-thought, and access to environment dynamics. We find that foundation models can provide diverse high-quality feedback across domains. Moreover, larger and reasoning models consistently provide more accurate feedback, exhibit less bias, and benefit more from enhanced inference methods. Finally, feedback quality degrades for environments with complex dynamics or continuous state spaces and action spaces. 

---
# Executable Ontologies: Synthesizing Event Semantics with Dataflow Architecture 

**Authors**: Aleksandr Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09775)  

**Abstract**: This paper presents boldsea, Boldachev's semantic-event approach -- an architecture for modeling complex dynamic systems using executable ontologies -- semantic models that act as dynamic structures, directly controlling process execution. We demonstrate that integrating event semantics with a dataflow architecture addresses the limitations of traditional Business Process Management (BPM) systems and object-oriented semantic technologies. The paper presents the formal BSL (boldsea Semantic Language), including its BNF grammar, and outlines the boldsea-engine's architecture, which directly interprets semantic models as executable algorithms without compilation. It enables the modification of event models at runtime, ensures temporal transparency, and seamlessly merges data and business logic within a unified semantic framework. 

---
# Human-AI Collaboration Increases Efficiency in Regulatory Writing 

**Authors**: Umut Eser, Yael Gozin, L. Jay Stallons, Ari Caroline, Martin Preusse, Brandon Rice, Scott Wright, Andrew Robertson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09738)  

**Abstract**: Background: Investigational New Drug (IND) application preparation is time-intensive and expertise-dependent, slowing early clinical development. Objective: To evaluate whether a large language model (LLM) platform (AutoIND) can reduce first-draft composition time while maintaining document quality in regulatory submissions. Methods: Drafting times for IND nonclinical written summaries (eCTD modules 2.6.2, 2.6.4, 2.6.6) generated by AutoIND were directly recorded. For comparison, manual drafting times for IND summaries previously cleared by the U.S. FDA were estimated from the experience of regulatory writers ($\geq$6 years) and used as industry-standard benchmarks. Quality was assessed by a blinded regulatory writing assessor using seven pre-specified categories: correctness, completeness, conciseness, consistency, clarity, redundancy, and emphasis. Each sub-criterion was scored 0-3 and normalized to a percentage. A critical regulatory error was defined as any misrepresentation or omission likely to alter regulatory interpretation (e.g., incorrect NOAEL, omission of mandatory GLP dose-formulation analysis). Results: AutoIND reduced initial drafting time by $\sim$97% (from $\sim$100 h to 3.7 h for 18,870 pages/61 reports in IND-1; and to 2.6 h for 11,425 pages/58 reports in IND-2). Quality scores were 69.6\% and 77.9\% for IND-1 and IND-2. No critical regulatory errors were detected, but deficiencies in emphasis, conciseness, and clarity were noted. Conclusions: AutoIND can dramatically accelerate IND drafting, but expert regulatory writers remain essential to mature outputs to submission-ready quality. Systematic deficiencies identified provide a roadmap for targeted model improvements. 

---
# Standards in the Preparation of Biomedical Research Metadata: A Bridge2AI Perspective 

**Authors**: Harry Caufield, Satrajit Ghosh, Sek Wong Kong, Jillian Parker, Nathan Sheffield, Bhavesh Patel, Andrew Williams, Timothy Clark, Monica C. Munoz-Torres  

**Link**: [PDF](https://arxiv.org/pdf/2509.10432)  

**Abstract**: AI-readiness describes the degree to which data may be optimally and ethically used for subsequent AI and Machine Learning (AI/ML) methods, where those methods may involve some combination of model training, data classification, and ethical, explainable prediction. The Bridge2AI consortium has defined the particular criteria a biomedical dataset may possess to render it AI-ready: in brief, a dataset's readiness is related to its FAIRness, provenance, degree of characterization, explainability, sustainability, and computability, in addition to its accompaniment with documentation about ethical data practices.
To ensure AI-readiness and to clarify data structure and relationships within Bridge2AI's Grand Challenges (GCs), particular types of metadata are necessary. The GCs within the Bridge2AI initiative include four data-generating projects focusing on generating AI/ML-ready datasets to tackle complex biomedical and behavioral research problems. These projects develop standardized, multimodal data, tools, and training resources to support AI integration, while addressing ethical data practices. Examples include using voice as a biomarker, building interpretable genomic tools, modeling disease trajectories with diverse multimodal data, and mapping cellular and molecular health indicators across the human body.
This report assesses the state of metadata creation and standardization in the Bridge2AI GCs, provides guidelines where required, and identifies gaps and areas for improvement across the program. New projects, including those outside the Bridge2AI consortium, would benefit from what we have learned about creating metadata as part of efforts to promote AI readiness. 

---
# Is In-Context Learning Learning? 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10414)  

**Abstract**: In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability. 

---
# Multimodal SAM-adapter for Semantic Segmentation 

**Authors**: Iacopo Curti, Pierluigi Zama Ramirez, Alioscia Petrelli, Luigi Di Stefano  

**Link**: [PDF](https://arxiv.org/pdf/2509.10408)  

**Abstract**: Semantic segmentation, a key task in computer vision with broad applications in autonomous driving, medical imaging, and robotics, has advanced substantially with deep learning. Nevertheless, current approaches remain vulnerable to challenging conditions such as poor lighting, occlusions, and adverse weather. To address these limitations, multimodal methods that integrate auxiliary sensor data (e.g., LiDAR, infrared) have recently emerged, providing complementary information that enhances robustness. In this work, we present MM SAM-adapter, a novel framework that extends the capabilities of the Segment Anything Model (SAM) for multimodal semantic segmentation. The proposed method employs an adapter network that injects fused multimodal features into SAM's rich RGB features. This design enables the model to retain the strong generalization ability of RGB features while selectively incorporating auxiliary modalities only when they contribute additional cues. As a result, MM SAM-adapter achieves a balanced and efficient use of multimodal information. We evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES, where MM SAM-adapter delivers state-of-the-art performance. To further analyze modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard subsets. Results consistently demonstrate that our framework outperforms competing methods in both favorable and adverse conditions, highlighting the effectiveness of multimodal adaptation for robust scene understanding. The code is available at the following link: this https URL. 

---
# Diversified recommendations of cultural activities with personalized determinantal point processes 

**Authors**: Carole Ibrahim, Hiba Bederina, Daniel Cuesta, Laurent Montier, Cyrille Delabre, Jill-Jênn Vie  

**Link**: [PDF](https://arxiv.org/pdf/2509.10392)  

**Abstract**: While optimizing recommendation systems for user engagement is a well-established practice, effectively diversifying recommendations without negatively impacting core business metrics remains a significant industry challenge. In line with our initiative to broaden our audience's cultural practices, this study investigates using personalized Determinantal Point Processes (DPPs) to sample diverse and relevant recommendations. We rely on a well-known quality-diversity decomposition of the similarity kernel to give more weight to user preferences. In this paper, we present our implementations of the personalized DPP sampling, evaluate the trade-offs between relevance and diversity through both offline and online metrics, and give insights for practitioners on their use in a production environment. For the sake of reproducibility, we release the full code for our platform and experiments on GitHub. 

---
# Improving Audio Event Recognition with Consistency Regularization 

**Authors**: Shanmuka Sadhu, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10391)  

**Abstract**: Consistency regularization (CR), which enforces agreement between model predictions on augmented views, has found recent benefits in automatic speech recognition [1]. In this paper, we propose the use of consistency regularization for audio event recognition, and demonstrate its effectiveness on AudioSet. With extensive ablation studies for both small ($\sim$20k) and large ($\sim$1.8M) supervised training sets, we show that CR brings consistent improvement over supervised baselines which already heavily utilize data augmentation, and CR using stronger augmentation and multiple augmentations leads to additional gain for the small training set. Furthermore, we extend the use of CR into the semi-supervised setup with 20K labeled samples and 1.8M unlabeled samples, and obtain performance improvement over our best model trained on the small set. 

---
# Data distribution impacts the performance and generalisability of contrastive learning-based foundation models of electrocardiograms 

**Authors**: Gul Rukh Khattak, Konstantinos Patlatzoglou, Joseph Barker, Libor Pastika, Boroumand Zeidaabadi, Ahmed El-Medany, Hesham Aggour, Yixiu Liang, Antonio H. Ribeiro, Jeffrey Annis, Antonio Luiz Pinho Ribeiro, Junbo Ge, Daniel B. Kramer, Jonathan W. Waks, Evan Brittain, Nicholas Peters, Fu Siong Ng, Arunashis Sau  

**Link**: [PDF](https://arxiv.org/pdf/2509.10369)  

**Abstract**: Contrastive learning is a widely adopted self-supervised pretraining strategy, yet its dependence on cohort composition remains underexplored. We present Contrasting by Patient Augmented Electrocardiograms (CAPE) foundation model and pretrain on four cohorts (n = 5,203,352), from diverse populations across three continents (North America, South America, Asia). We systematically assess how cohort demographics, health status, and population diversity influence the downstream performance for prediction tasks also including two additional cohorts from another continent (Europe). We find that downstream performance depends on the distributional properties of the pretraining cohort, including demographics and health status. Moreover, while pretraining with a multi-centre, demographically diverse cohort improves in-distribution accuracy, it reduces out-of-distribution (OOD) generalisation of our contrastive approach by encoding cohort-specific artifacts. To address this, we propose the In-Distribution Batch (IDB) strategy, which preserves intra-cohort consistency during pretraining and enhances OOD robustness. This work provides important insights for developing clinically fair and generalisable foundation models. 

---
# Towards Understanding Visual Grounding in Visual Language Models 

**Authors**: Georgios Pantazopoulos, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2509.10345)  

**Abstract**: Visual grounding refers to the ability of a model to identify a region within some visual input that matches a textual description. Consequently, a model equipped with visual grounding capabilities can target a wide range of applications in various domains, including referring expression comprehension, answering questions pertinent to fine-grained details in images or videos, caption visual context by explicitly referring to entities, as well as low and high-level control in simulated and real environments. In this survey paper, we review representative works across the key areas of research on modern general-purpose vision language models (VLMs). We first outline the importance of grounding in VLMs, then delineate the core components of the contemporary paradigm for developing grounded models, and examine their practical applications, including benchmarks and evaluation metrics for grounded multimodal generation. We also discuss the multifaceted interrelations among visual grounding, multimodal chain-of-thought, and reasoning in VLMs. Finally, we analyse the challenges inherent to visual grounding and suggest promising directions for future research. 

---
# GLAM: Geometry-Guided Local Alignment for Multi-View VLP in Mammography 

**Authors**: Yuexi Du, Lihui Chen, Nicha C. Dvornek  

**Link**: [PDF](https://arxiv.org/pdf/2509.10344)  

**Abstract**: Mammography screening is an essential tool for early detection of breast cancer. The speed and accuracy of mammography interpretation have the potential to be improved with deep learning methods. However, the development of a foundation visual language model (VLM) is hindered by limited data and domain differences between natural and medical images. Existing mammography VLMs, adapted from natural images, often ignore domain-specific characteristics, such as multi-view relationships in mammography. Unlike radiologists who analyze both views together to process ipsilateral correspondence, current methods treat them as independent images or do not properly model the multi-view correspondence learning, losing critical geometric context and resulting in suboptimal prediction. We propose GLAM: Global and Local Alignment for Multi-view mammography for VLM pretraining using geometry guidance. By leveraging the prior knowledge about the multi-view imaging process of mammograms, our model learns local cross-view alignments and fine-grained local features through joint global and local, visual-visual, and visual-language contrastive learning. Pretrained on EMBED [14], one of the largest open mammography datasets, our model outperforms baselines across multiple datasets under different settings. 

---
# I-Segmenter: Integer-Only Vision Transformer for Efficient Semantic Segmentation 

**Authors**: Jordan Sassoon, Michal Szczepanski, Martyna Poreba  

**Link**: [PDF](https://arxiv.org/pdf/2509.10334)  

**Abstract**: Vision Transformers (ViTs) have recently achieved strong results in semantic segmentation, yet their deployment on resource-constrained devices remains limited due to their high memory footprint and computational cost. Quantization offers an effective strategy to improve efficiency, but ViT-based segmentation models are notoriously fragile under low precision, as quantization errors accumulate across deep encoder-decoder pipelines. We introduce I-Segmenter, the first fully integer-only ViT segmentation framework. Building on the Segmenter architecture, I-Segmenter systematically replaces floating-point operations with integer-only counterparts. To further stabilize both training and inference, we propose $\lambda$-ShiftGELU, a novel activation function that mitigates the limitations of uniform quantization in handling long-tailed activation distributions. In addition, we remove the L2 normalization layer and replace bilinear interpolation in the decoder with nearest neighbor upsampling, ensuring integer-only execution throughout the computational graph. Extensive experiments show that I-Segmenter achieves accuracy within a reasonable margin of its FP32 baseline (5.1 % on average), while reducing model size by up to 3.8x and enabling up to 1.2x faster inference with optimized runtimes. Notably, even in one-shot PTQ with a single calibration image, I-Segmenter delivers competitive accuracy, underscoring its practicality for real-world deployment. 

---
# Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Data 

**Authors**: Jesse van Remmerden, Zaharah Bukhsh, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10303)  

**Abstract**: The Job-Shop Scheduling Problem (JSP) and Flexible Job-Shop Scheduling Problem (FJSP), are canonical combinatorial optimization problems with wide-ranging applications in industrial operations. In recent years, many online reinforcement learning (RL) approaches have been proposed to learn constructive heuristics for JSP and FJSP. Although effective, these online RL methods require millions of interactions with simulated environments that may not capture real-world complexities, and their random policy initialization leads to poor sample efficiency. To address these limitations, we introduce Conservative Discrete Quantile Actor-Critic (CDQAC), a novel offline RL algorithm that learns effective scheduling policies directly from historical data, eliminating the need for costly online interactions, while maintaining the ability to improve upon suboptimal training data. CDQAC couples a quantile-based critic with a delayed policy update, estimating the return distribution of each machine-operation pair rather than selecting pairs outright. Our extensive experiments demonstrate CDQAC's remarkable ability to learn from diverse data sources. CDQAC consistently outperforms the original data-generating heuristics and surpasses state-of-the-art offline and online RL baselines. In addition, CDQAC is highly sample efficient, requiring only 10-20 training instances to learn high-quality policies. Surprisingly, we find that CDQAC performs better when trained on data generated by a random heuristic than when trained on higher-quality data from genetic algorithms and priority dispatching rules. 

---
# We Need a New Ethics for a World of AI Agents 

**Authors**: Iason Gabriel, Geoff Keeling, Arianna Manzini, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.10289)  

**Abstract**: The deployment of capable AI agents raises fresh questions about safety, human-machine relationships and social coordination. We argue for greater engagement by scientists, scholars, engineers and policymakers with the implications of a world increasingly populated by AI agents. We explore key challenges that must be addressed to ensure that interactions between humans and agents, and among agents themselves, remain broadly beneficial. 

---
# SignClip: Leveraging Mouthing Cues for Sign Language Translation by Multimodal Contrastive Fusion 

**Authors**: Wenfang Wu, Tingting Yuan, Yupeng Li, Daling Wang, Xiaoming Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10266)  

**Abstract**: Sign language translation (SLT) aims to translate natural language from sign language videos, serving as a vital bridge for inclusive communication. While recent advances leverage powerful visual backbones and large language models, most approaches mainly focus on manual signals (hand gestures) and tend to overlook non-manual cues like mouthing. In fact, mouthing conveys essential linguistic information in sign languages and plays a crucial role in disambiguating visually similar signs. In this paper, we propose SignClip, a novel framework to improve the accuracy of sign language translation. It fuses manual and non-manual cues, specifically spatial gesture and lip movement features. Besides, SignClip introduces a hierarchical contrastive learning framework with multi-level alignment objectives, ensuring semantic consistency across sign-lip and visual-text modalities. Extensive experiments on two benchmark datasets, PHOENIX14T and How2Sign, demonstrate the superiority of our approach. For example, on PHOENIX14T, in the Gloss-free setting, SignClip surpasses the previous state-of-the-art model SpaMo, improving BLEU-4 from 24.32 to 24.71, and ROUGE from 46.57 to 48.38. 

---
# Openness in AI and downstream governance: A global value chain approach 

**Authors**: Christopher Foster  

**Link**: [PDF](https://arxiv.org/pdf/2509.10220)  

**Abstract**: The rise of AI has been rapid, becoming a leading sector for investment and promising disruptive impacts across the economy. Within the critical analysis of the economic impacts, AI has been aligned to the critical literature on data power and platform capitalism - further concentrating power and value capture amongst a small number of "big tech" leaders.
The equally rapid rise of openness in AI (here taken to be claims made by AI firms about openness, "open source" and free provision) signals an interesting development. It highlights an emerging ecosystem of open AI models, datasets and toolchains, involving massive capital investment. It poses questions as to whether open resources can support technological transfer and the ability for catch-up, even in the face of AI industry power.
This work seeks to add conceptual clarity to these debates by conceptualising openness in AI as a unique type of interfirm relation and therefore amenable to value chain analysis. This approach then allows consideration of the capitalist dynamics of "outsourcing" of foundational firms in value chains, and consequently the types of governance and control that might emerge downstream as AI is adopted. This work, therefore, extends previous mapping of AI value chains to build a framework which links foundational AI with downstream value chains.
Overall, this work extends our understanding of AI as a productive sector. While the work remains critical of the power of leading AI firms, openness in AI may lead to potential spillovers stemming from the intense competition for global technological leadership in AI. 

---
# SI-FACT: Mitigating Knowledge Conflict via Self-Improving Faithfulness-Aware Contrastive Tuning 

**Authors**: Shengqiang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10208)  

**Abstract**: Large Language Models often generate unfaithful responses in knowledge intensive tasks due to knowledge conflict,that is,a preference for relying on internal parametric knowledge rather than the provided this http URL address this issue,we propose a novel self improving framework,Self Improving Faithfulness Aware Contrastive this http URL framework uses a self instruct mechanism that allows the base LLM to automatically generate high quality,structured contrastive learning data,including anchor samples,semantically equivalent positive samples,and negative samples simulating unfaithful this http URL approach significantly reduces the cost of manual this http URL,contrastive learning is applied to train the model,enabling it to pull faithful responses closer and push unfaithful responses farther apart in the representation this http URL on knowledge conflict evaluation benchmarks ECARE KRE and COSE KRE show that the SI FACT model based on Llama3 8B Instruct improves the Contextual Recall Rate by 6.2% over the best baseline method,while significantly reducing dependence on internal this http URL results indicate that SI FACT provides strong effectiveness and high data efficiency in enhancing the contextual faithfulness of LLMs,offering a practical pathway toward building more proactive and trustworthy language models. 

---
# Benchmark of stylistic variation in LLM-generated texts 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.10179)  

**Abstract**: This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions. 

---
# BenchECG and xECG: a benchmark and baseline for ECG foundation models 

**Authors**: Riccardo Lunelli, Angus Nicolson, Samuel Martin Pröll, Sebastian Johannes Reinstadler, Axel Bauer, Clemens Dlaska  

**Link**: [PDF](https://arxiv.org/pdf/2509.10151)  

**Abstract**: Electrocardiograms (ECGs) are inexpensive, widely used, and well-suited to deep learning. Recently, interest has grown in developing foundation models for ECGs - models that generalise across diverse downstream tasks. However, consistent evaluation has been lacking: prior work often uses narrow task selections and inconsistent datasets, hindering fair comparison. Here, we introduce BenchECG, a standardised benchmark comprising a comprehensive suite of publicly available ECG datasets and versatile tasks. We also propose xECG, an xLSTM-based recurrent model trained with SimDINOv2 self-supervised learning, which achieves the best BenchECG score compared to publicly available state-of-the-art models. In particular, xECG is the only publicly available model to perform strongly on all datasets and tasks. By standardising evaluation, BenchECG enables rigorous comparison and aims to accelerate progress in ECG representation learning. xECG achieves superior performance over earlier approaches, defining a new baseline for future ECG foundation models. 

---
# Efficient Learning-Based Control of a Legged Robot in Lunar Gravity 

**Authors**: Philip Arm, Oliver Fischer, Joseph Church, Adrian Fuhrer, Hendrik Kolvenbach, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10128)  

**Abstract**: Legged robots are promising candidates for exploring challenging areas on low-gravity bodies such as the Moon, Mars, or asteroids, thanks to their advanced mobility on unstructured terrain. However, as planetary robots' power and thermal budgets are highly restricted, these robots need energy-efficient control approaches that easily transfer to multiple gravity environments. In this work, we introduce a reinforcement learning-based control approach for legged robots with gravity-scaled power-optimized reward functions. We use our approach to develop and validate a locomotion controller and a base pose controller in gravity environments from lunar gravity (1.62 m/s2) to a hypothetical super-Earth (19.62 m/s2). Our approach successfully scales across these gravity levels for locomotion and base pose control with the gravity-scaled reward functions. The power-optimized locomotion controller reached a power consumption for locomotion of 23.4 W in Earth gravity on a 15.65 kg robot at 0.4 m/s, a 23 % improvement over the baseline policy. Additionally, we designed a constant-force spring offload system that allowed us to conduct real-world experiments on legged locomotion in lunar gravity. In lunar gravity, the power-optimized control policy reached 12.2 W, 36 % less than a baseline controller which is not optimized for power efficiency. Our method provides a scalable approach to developing power-efficient locomotion controllers for legged robots across multiple gravity levels. 

---
# Population-Aligned Persona Generation for LLM-based Social Simulation 

**Authors**: Zhengyu Hu, Zheyuan Xiao, Max Xiong, Yuxuan Lei, Tianfu Wang, Jianxun Lian, Kaize Ding, Ziang Xiao, Nicholas Jing Yuan, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.10127)  

**Abstract**: Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications. 

---
# Realism Control One-step Diffusion for Real-World Image Super-Resolution 

**Authors**: Zongliang Wu, Siming Zheng, Peng-Tao Jiang, Xin Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.10122)  

**Abstract**: Pre-trained diffusion models have shown great potential in real-world image super-resolution (Real-ISR) tasks by enabling high-resolution reconstructions. While one-step diffusion (OSD) methods significantly improve efficiency compared to traditional multi-step approaches, they still have limitations in balancing fidelity and realism across diverse scenarios. Since the OSDs for SR are usually trained or distilled by a single timestep, they lack flexible control mechanisms to adaptively prioritize these competing objectives, which are inherently manageable in multi-step methods through adjusting sampling steps. To address this challenge, we propose a Realism Controlled One-step Diffusion (RCOD) framework for Real-ISR. RCOD provides a latent domain grouping strategy that enables explicit control over fidelity-realism trade-offs during the noise prediction phase with minimal training paradigm modifications and original training data. A degradation-aware sampling strategy is also introduced to align distillation regularization with the grouping strategy and enhance the controlling of trade-offs. Moreover, a visual prompt injection module is used to replace conventional text prompts with degradation-aware visual tokens, enhancing both restoration accuracy and semantic consistency. Our method achieves superior fidelity and perceptual quality while maintaining computational efficiency. Extensive experiments demonstrate that RCOD outperforms state-of-the-art OSD methods in both quantitative metrics and visual qualities, with flexible realism control capabilities in the inference stage. The code will be released. 

---
# Generating Energy-Efficient Code via Large-Language Models -- Where are we now? 

**Authors**: Radu Apsan, Vincenzo Stoico, Michel Albonico, Rudra Dhar, Karthik Vaidhyanathan, Ivano Malavolta  

**Link**: [PDF](https://arxiv.org/pdf/2509.10099)  

**Abstract**: Context. The rise of Large Language Models (LLMs) has led to their widespread adoption in development pipelines. Goal. We empirically assess the energy efficiency of Python code generated by LLMs against human-written code and code developed by a Green software expert. Method. We test 363 solutions to 9 coding problems from the EvoEval benchmark using 6 widespread LLMs with 4 prompting techniques, and comparing them to human-developed solutions. Energy consumption is measured on three different hardware platforms: a server, a PC, and a Raspberry Pi for a total of ~881h (36.7 days). Results. Human solutions are 16% more energy-efficient on the server and 3% on the Raspberry Pi, while LLMs outperform human developers by 25% on the PC. Prompting does not consistently lead to energy savings, where the most energy-efficient prompts vary by hardware platform. The code developed by a Green software expert is consistently more energy-efficient by at least 17% to 30% against all LLMs on all hardware platforms. Conclusions. Even though LLMs exhibit relatively good code generation capabilities, no LLM-generated code was more energy-efficient than that of an experienced Green software developer, suggesting that as of today there is still a great need of human expertise for developing energy-efficient Python code. 

---
# Established Psychometric vs. Ecologically Valid Questionnaires: Rethinking Psychological Assessments in Large Language Models 

**Authors**: Dongmin Choi, Woojung Song, Jongwook Han, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.10078)  

**Abstract**: Researchers have applied established psychometric questionnaires (e.g., BFI, PVQ) to measure the personality traits and values reflected in the responses of Large Language Models (LLMs). However, concerns have been raised about applying these human-designed questionnaires to LLMs. One such concern is their lack of ecological validity--the extent to which survey questions adequately reflect and resemble real-world contexts in which LLMs generate texts in response to user queries. However, it remains unclear how established questionnaires and ecologically valid questionnaires differ in their outcomes, and what insights these differences may provide. In this paper, we conduct a comprehensive comparative analysis of the two types of questionnaires. Our analysis reveals that established questionnaires (1) yield substantially different profiles of LLMs from ecologically valid ones, deviating from the psychological characteristics expressed in the context of user queries, (2) suffer from insufficient items for stable measurement, (3) create misleading impressions that LLMs possess stable constructs, and (4) yield exaggerated profiles for persona-prompted LLMs. Overall, our work cautions against the use of established psychological questionnaires for LLMs. Our code will be released upon publication. 

---
# Predictive Spike Timing Enables Distributed Shortest Path Computation in Spiking Neural Networks 

**Authors**: Simen Storesund, Kristian Valset Aars, Robin Dietrich, Nicolai Waniek  

**Link**: [PDF](https://arxiv.org/pdf/2509.10077)  

**Abstract**: Efficient planning and sequence selection are central to intelligence, yet current approaches remain largely incompatible with biological computation. Classical graph algorithms like Dijkstra's or A* require global state and biologically implausible operations such as backtracing, while reinforcement learning methods rely on slow gradient-based policy updates that appear inconsistent with rapid behavioral adaptation observed in natural systems.
We propose a biologically plausible algorithm for shortest-path computation that operates through local spike-based message-passing with realistic processing delays. The algorithm exploits spike-timing coincidences to identify nodes on optimal paths: Neurons that receive inhibitory-excitatory message pairs earlier than predicted reduce their response delays, creating a temporal compression that propagates backwards from target to source. Through analytical proof and simulations on random spatial networks, we demonstrate that the algorithm converges and discovers all shortest paths using purely timing-based mechanisms. By showing how short-term timing dynamics alone can compute shortest paths, this work provides new insights into how biological networks might solve complex computational problems through purely local computation and relative spike-time prediction. These findings open new directions for understanding distributed computation in biological and artificial systems, with possible implications for computational neuroscience, AI, reinforcement learning, and neuromorphic systems. 

---
# TwinTac: A Wide-Range, Highly Sensitive Tactile Sensor with Real-to-Sim Digital Twin Sensor Model 

**Authors**: Xiyan Huang, Zhe Xu, Chenxi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.10063)  

**Abstract**: Robot skill acquisition processes driven by reinforcement learning often rely on simulations to efficiently generate large-scale interaction data. However, the absence of simulation models for tactile sensors has hindered the use of tactile sensing in such skill learning processes, limiting the development of effective policies driven by tactile perception. To bridge this gap, we present TwinTac, a system that combines the design of a physical tactile sensor with its digital twin model. Our hardware sensor is designed for high sensitivity and a wide measurement range, enabling high quality sensing data essential for object interaction tasks. Building upon the hardware sensor, we develop the digital twin model using a real-to-sim approach. This involves collecting synchronized cross-domain data, including finite element method results and the physical sensor's outputs, and then training neural networks to map simulated data to real sensor responses. Through experimental evaluation, we characterized the sensitivity of the physical sensor and demonstrated the consistency of the digital twin in replicating the physical sensor's output. Furthermore, by conducting an object classification task, we showed that simulation data generated by our digital twin sensor can effectively augment real-world data, leading to improved accuracy. These results highlight TwinTac's potential to bridge the gap in cross-domain learning tasks. 

---
# Multimodal Mathematical Reasoning Embedded in Aerial Vehicle Imagery: Benchmarking, Analysis, and Exploration 

**Authors**: Yue Zhou, Litong Feng, Mengcheng Lan, Xue Yang, Qingyun Li, Yiping Ke, Xue Jiang, Wayne Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10059)  

**Abstract**: Mathematical reasoning is critical for tasks such as precise distance and area computations, trajectory estimations, and spatial analysis in unmanned aerial vehicle (UAV) based remote sensing, yet current vision-language models (VLMs) have not been adequately tested in this domain. To address this gap, we introduce AVI-Math, the first benchmark to rigorously evaluate multimodal mathematical reasoning in aerial vehicle imagery, moving beyond simple counting tasks to include domain-specific knowledge in areas such as geometry, logic, and algebra. The dataset comprises 3,773 high-quality vehicle-related questions captured from UAV views, covering 6 mathematical subjects and 20 topics. The data, collected at varying altitudes and from multiple UAV angles, reflects real-world UAV scenarios, ensuring the diversity and complexity of the constructed mathematical problems. In this paper, we benchmark 14 prominent VLMs through a comprehensive evaluation and demonstrate that, despite their success on previous multimodal benchmarks, these models struggle with the reasoning tasks in AVI-Math. Our detailed analysis highlights significant limitations in the mathematical reasoning capabilities of current VLMs and suggests avenues for future research. Furthermore, we explore the use of Chain-of-Thought prompting and fine-tuning techniques, which show promise in addressing the reasoning challenges in AVI-Math. Our findings not only expose the limitations of VLMs in mathematical reasoning but also offer valuable insights for advancing UAV-based trustworthy VLMs in real-world applications. The code, and datasets will be released at this https URL 

---
# Reinforcement learning for spin torque oscillator tasks 

**Authors**: Jakub Mojsiejuk, Sławomir Ziętek, Witold Skowroński  

**Link**: [PDF](https://arxiv.org/pdf/2509.10057)  

**Abstract**: We address the problem of automatic synchronisation of the spintronic oscillator (STO) by means of reinforcement learning (RL). A numerical solution of the macrospin Landau-Lifschitz-Gilbert-Slonczewski equation is used to simulate the STO and we train the two types of RL agents to synchronise with a target frequency within a fixed number of steps. We explore modifications to this base task and show an improvement in both convergence and energy efficiency of the synchronisation that can be easily achieved in the simulated environment. 

---
# Exploring Expert Specialization through Unsupervised Training in Sparse Mixture of Experts 

**Authors**: Strahinja Nikolic, Ilker Oguz, Demetri Psaltis  

**Link**: [PDF](https://arxiv.org/pdf/2509.10025)  

**Abstract**: Understanding the internal organization of neural networks remains a fundamental challenge in deep learning interpretability. We address this challenge by exploring a novel Sparse Mixture of Experts Variational Autoencoder (SMoE-VAE) architecture. We test our model on the QuickDraw dataset, comparing unsupervised expert routing against a supervised baseline guided by ground-truth labels. Surprisingly, we find that unsupervised routing consistently achieves superior reconstruction performance. The experts learn to identify meaningful sub-categorical structures that often transcend human-defined class boundaries. Through t-SNE visualizations and reconstruction analysis, we investigate how MoE models uncover fundamental data structures that are more aligned with the model's objective than predefined labels. Furthermore, our study on the impact of dataset size provides insights into the trade-offs between data quantity and expert specialization, offering guidance for designing efficient MoE architectures. 

---
# Intrinsic Dimension Estimating Autoencoder (IDEA) Using CancelOut Layer and a Projected Loss 

**Authors**: Antoine Orioua, Philipp Krah, Julian Koellermeier  

**Link**: [PDF](https://arxiv.org/pdf/2509.10011)  

**Abstract**: This paper introduces the Intrinsic Dimension Estimating Autoencoder (IDEA), which identifies the underlying intrinsic dimension of a wide range of datasets whose samples lie on either linear or nonlinear manifolds. Beyond estimating the intrinsic dimension, IDEA is also able to reconstruct the original dataset after projecting it onto the corresponding latent space, which is structured using re-weighted double CancelOut layers. Our key contribution is the introduction of the projected reconstruction loss term, guiding the training of the model by continuously assessing the reconstruction quality under the removal of an additional latent dimension. We first assess the performance of IDEA on a series of theoretical benchmarks to validate its robustness. These experiments allow us to test its reconstruction ability and compare its performance with state-of-the-art intrinsic dimension estimators. The benchmarks show good accuracy and high versatility of our approach. Subsequently, we apply our model to data generated from the numerical solution of a vertically resolved one-dimensional free-surface flow, following a pointwise discretization of the vertical velocity profile in the horizontal direction, vertical direction, and time. IDEA succeeds in estimating the dataset's intrinsic dimension and then reconstructs the original solution by working directly within the projection space identified by the network. 

---
# Unsupervised Hallucination Detection by Inspecting Reasoning Processes 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10004)  

**Abstract**: Unsupervised hallucination detection aims to identify hallucinated content generated by large language models (LLMs) without relying on labeled data. While unsupervised methods have gained popularity by eliminating labor-intensive human annotations, they frequently rely on proxy signals unrelated to factual correctness. This misalignment biases detection probes toward superficial or non-truth-related aspects, limiting generalizability across datasets and scenarios. To overcome these limitations, we propose IRIS, an unsupervised hallucination detection framework, leveraging internal representations intrinsic to factual correctness. IRIS prompts the LLM to carefully verify the truthfulness of a given statement, and obtain its contextualized embedding as informative features for training. Meanwhile, the uncertainty of each response is considered a soft pseudolabel for truthfulness. Experimental results demonstrate that IRIS consistently outperforms existing unsupervised methods. Our approach is fully unsupervised, computationally low cost, and works well even with few training data, making it suitable for real-time detection. 

---
# Drone-Based Multispectral Imaging and Deep Learning for Timely Detection of Branched Broomrape in Tomato Farms 

**Authors**: Mohammadreza Narimani, Alireza Pourreza, Ali Moghimi, Mohsen Mesgaran, Parastoo Farajpoor, Hamid Jafarbiglu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09972)  

**Abstract**: This study addresses the escalating threat of branched broomrape (Phelipanche ramosa) to California's tomato industry, which supplies over 90 percent of U.S. processing tomatoes. The parasite's largely underground life cycle makes early detection difficult, while conventional chemical controls are costly, environmentally harmful, and often ineffective. To address this, we combined drone-based multispectral imagery with Long Short-Term Memory (LSTM) deep learning networks, using the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance. Research was conducted on a known broomrape-infested tomato farm in Woodland, Yolo County, CA, across five key growth stages determined by growing degree days (GDD). Multispectral images were processed to isolate tomato canopy reflectance. At 897 GDD, broomrape could be detected with 79.09 percent overall accuracy and 70.36 percent recall without integrating later stages. Incorporating sequential growth stages with LSTM improved detection substantially. The best-performing scenario, which integrated all growth stages with SMOTE augmentation, achieved 88.37 percent overall accuracy and 95.37 percent recall. These results demonstrate the strong potential of temporal multispectral analysis and LSTM networks for early broomrape detection. While further real-world data collection is needed for practical deployment, this study shows that UAV-based multispectral sensing coupled with deep learning could provide a powerful precision agriculture tool to reduce losses and improve sustainability in tomato production. 

---
# Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09970)  

**Abstract**: Large Language Models (LLMs) show promise in generating firmware for embedded systems, but often introduce security flaws and fail to meet real-time performance constraints. This paper proposes a three-phase methodology that combines LLM-based firmware generation with automated security validation and iterative refinement in a virtualized environment. Using structured prompts, models like GPT-4 generate firmware for networking and control tasks, deployed on FreeRTOS via QEMU. These implementations are tested using fuzzing, static analysis, and runtime monitoring to detect vulnerabilities such as buffer overflows (CWE-120), race conditions (CWE-362), and denial-of-service threats (CWE-400). Specialized AI agents for Threat Detection, Performance Optimization, and Compliance Verification collaborate to improve detection and remediation. Identified issues are categorized using CWE, then used to prompt targeted LLM-generated patches in an iterative loop. Experiments show a 92.4\% Vulnerability Remediation Rate (37.3\% improvement), 95.8\% Threat Model Compliance, and 0.87 Security Coverage Index. Real-time metrics include 8.6ms worst-case execution time and 195{\mu}s jitter. This process enhances firmware security and performance while contributing an open-source dataset for future research. 

---
# Large Language Models Meet Legal Artificial Intelligence: A Survey 

**Authors**: Zhitian Hou, Zihan Ye, Nanli Zeng, Tianyong Hao, Kun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09969)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the development of Legal Artificial Intelligence (Legal AI) in recent years, enhancing the efficiency and accuracy of legal tasks. To advance research and applications of LLM-based approaches in legal domain, this paper provides a comprehensive review of 16 legal LLMs series and 47 LLM-based frameworks for legal tasks, and also gather 15 benchmarks and 29 datasets to evaluate different legal capabilities. Additionally, we analyse the challenges and discuss future directions for LLM-based approaches in the legal domain. We hope this paper provides a systematic introduction for beginners and encourages future research in this field. Resources are available at this https URL. 

---
# Limited Reference, Reliable Generation: A Two-Component Framework for Tabular Data Generation in Low-Data Regimes 

**Authors**: Mingxuan Jiang, Yongxin Wang, Ziyue Dai, Yicun Liu, Hongyi Nie, Sen Liu, Hongfeng Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09960)  

**Abstract**: Synthetic tabular data generation is increasingly essential in data management, supporting downstream applications when real-world and high-quality tabular data is insufficient. Existing tabular generation approaches, such as generative adversarial networks (GANs), diffusion models, and fine-tuned Large Language Models (LLMs), typically require sufficient reference data, limiting their effectiveness in domain-specific databases with scarce records. While prompt-based LLMs offer flexibility without parameter tuning, they often fail to capture dataset-specific feature-label dependencies and generate redundant data, leading to degradation in downstream task performance. To overcome these issues, we propose ReFine, a framework that (i) derives symbolic "if-then" rules from interpretable models and embeds them into prompts to explicitly guide generation toward domain-specific feature distribution, and (ii) applies a dual-granularity filtering strategy that suppresses over-sampling patterns and selectively refines rare but informative samples to reduce distributional imbalance. Extensive experiments on various regression and classification benchmarks demonstrate that ReFine consistently outperforms state-of-the-art methods, achieving up to 0.44 absolute improvement in R-squared for regression and 10.0 percent relative improvement in F1 score for classification tasks. 

---
# Zero-Shot Referring Expression Comprehension via Visual-Language True/False Verification 

**Authors**: Jeffrey Liu, Rongbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09958)  

**Abstract**: Referring Expression Comprehension (REC) is usually addressed with task-trained grounding models. We show that a zero-shot workflow, without any REC-specific training, can achieve competitive or superior performance. Our approach reformulates REC as box-wise visual-language verification: given proposals from a COCO-clean generic detector (YOLO-World), a general-purpose VLM independently answers True/False queries for each region. This simple procedure reduces cross-box interference, supports abstention and multiple matches, and requires no fine-tuning. On RefCOCO, RefCOCO+, and RefCOCOg, our method not only surpasses a zero-shot GroundingDINO baseline but also exceeds reported results for GroundingDINO trained on REC and GroundingDINO+CRG. Controlled studies with identical proposals confirm that verification significantly outperforms selection-based prompting, and results hold with open VLMs. Overall, we show that workflow design, rather than task-specific pretraining, drives strong zero-shot REC performance. 

---
# Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge 

**Authors**: Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis, Sami Muhaidat  

**Link**: [PDF](https://arxiv.org/pdf/2509.09955)  

**Abstract**: Large-scale transformers are central to modern semantic communication, yet their high computational and communication costs hinder deployment on resource-constrained edge devices. This paper introduces a training-free framework for adaptive token merging, a novel mechanism that compresses transformer representations at runtime by selectively merging semantically redundant tokens under per-layer similarity thresholds. Unlike prior fixed-ratio reduction, our approach couples merging directly to input redundancy, enabling data-dependent adaptation that balances efficiency and task relevance without retraining. We cast the discovery of merging strategies as a multi-objective optimization problem and leverage Bayesian optimization to obtain Pareto-optimal trade-offs between accuracy, inference cost, and communication cost. On ImageNet classification, we match the accuracy of the unmodified transformer with 30\% fewer floating-point operations per second and under 20\% of the original communication cost, while for visual question answering our method achieves performance competitive with the full LLaVA model at less than one-third of the compute and one-tenth of the bandwidth. Finally, we show that our adaptive merging is robust across varying channel conditions and provides inherent privacy benefits, substantially degrading the efficacy of model inversion attacks. Our framework provides a practical and versatile solution for deploying powerful transformer models in resource-limited edge intelligence scenarios. 

---
# SmartCoder-R1: Towards Secure and Explainable Smart Contract Generation with Security-Aware Group Relative Policy Optimization 

**Authors**: Lei Yu, Jingyuan Zhang, Xin Wang, Jiajia Ma, Li Yang, Fengjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09942)  

**Abstract**: Smart contracts automate the management of high-value assets, where vulnerabilities can lead to catastrophic financial losses. This challenge is amplified in Large Language Models (LLMs) by two interconnected failures: they operate as unauditable "black boxes" lacking a transparent reasoning process, and consequently, generate code riddled with critical security vulnerabilities. To address both issues, we propose SmartCoder-R1 (based on Qwen2.5-Coder-7B), a novel framework for secure and explainable smart contract generation. It begins with Continual Pre-training (CPT) to specialize the model. We then apply Long Chain-of-Thought Supervised Fine-Tuning (L-CoT SFT) on 7,998 expert-validated reasoning-and-code samples to train the model to emulate human security analysis. Finally, to directly mitigate vulnerabilities, we employ Security-Aware Group Relative Policy Optimization (S-GRPO), a reinforcement learning phase that refines the generation policy by optimizing a weighted reward signal for compilation success, security compliance, and format correctness. Evaluated against 17 baselines on a benchmark of 756 real-world functions, SmartCoder-R1 establishes a new state of the art, achieving top performance across five key metrics: a ComPass of 87.70%, a VulRate of 8.60%, a SafeAval of 80.16%, a FuncRate of 53.84%, and a FullRate of 50.53%. This FullRate marks a 45.79% relative improvement over the strongest baseline, DeepSeek-R1. Crucially, its generated reasoning also excels in human evaluations, achieving high-quality ratings for Functionality (82.7%), Security (85.3%), and Clarity (90.7%). 

---
# WALL: A Web Application for Automated Quality Assurance using Large Language Models 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09918)  

**Abstract**: As software projects become increasingly complex, the volume and variety of issues in code files have grown substantially. Addressing this challenge requires efficient issue detection, resolution, and evaluation tools. This paper presents WALL, a web application that integrates SonarQube and large language models (LLMs) such as GPT-3.5 Turbo and GPT-4o to automate these tasks. WALL comprises three modules: an issue extraction tool, code issues reviser, and code comparison tool. Together, they enable a seamless pipeline for detecting software issues, generating automated code revisions, and evaluating the accuracy of revisions. Our experiments, conducted on 563 files with over 7,599 issues, demonstrate WALL's effectiveness in reducing human effort while maintaining high-quality revisions. Results show that employing a hybrid approach of cost-effective and advanced LLMs can significantly lower costs and improve revision rates. Future work aims to enhance WALL's capabilities by integrating open-source LLMs and eliminating human intervention, paving the way for fully automated code quality management. 

---
# An Autoencoder and Vision Transformer-based Interpretability Analysis of the Differences in Automated Staging of Second and Third Molars 

**Authors**: Barkin Buyukcakir, Jannick De Tobel, Patrick Thevissen, Dirk Vandermeulen, Peter Claes  

**Link**: [PDF](https://arxiv.org/pdf/2509.09911)  

**Abstract**: The practical adoption of deep learning in high-stakes forensic applications, such as dental age estimation, is often limited by the 'black box' nature of the models. This study introduces a framework designed to enhance both performance and transparency in this context. We use a notable performance disparity in the automated staging of mandibular second (tooth 37) and third (tooth 38) molars as a case study. The proposed framework, which combines a convolutional autoencoder (AE) with a Vision Transformer (ViT), improves classification accuracy for both teeth over a baseline ViT, increasing from 0.712 to 0.815 for tooth 37 and from 0.462 to 0.543 for tooth 38. Beyond improving performance, the framework provides multi-faceted diagnostic insights. Analysis of the AE's latent space metrics and image reconstructions indicates that the remaining performance gap is data-centric, suggesting high intra-class morphological variability in the tooth 38 dataset is a primary limiting factor. This work highlights the insufficiency of relying on a single mode of interpretability, such as attention maps, which can appear anatomically plausible yet fail to identify underlying data issues. By offering a methodology that both enhances accuracy and provides evidence for why a model may be uncertain, this framework serves as a more robust tool to support expert decision-making in forensic age estimation. 

---
# Tackling One Health Risks: How Large Language Models are leveraged for Risk Negotiation and Consensus-building 

**Authors**: Alexandra Fetsch, Iurii Savvateev, Racem Ben Romdhane, Martin Wiedmann, Artemiy Dimov, Maciej Durkalec, Josef Teichmann, Jakob Zinsstag, Konstantinos Koutsoumanis, Andreja Rajkovic, Jason Mann, Mauro Tonolla, Monika Ehling-Schulz, Matthias Filter, Sophia Johler  

**Link**: [PDF](https://arxiv.org/pdf/2509.09906)  

**Abstract**: Key global challenges of our times are characterized by complex interdependencies and can only be effectively addressed through an integrated, participatory effort. Conventional risk analysis frameworks often reduce complexity to ensure manageability, creating silos that hinder comprehensive solutions. A fundamental shift towards holistic strategies is essential to enable effective negotiations between different sectors and to balance the competing interests of stakeholders. However, achieving this balance is often hindered by limited time, vast amounts of information, and the complexity of integrating diverse perspectives. This study presents an AI-assisted negotiation framework that incorporates large language models (LLMs) and AI-based autonomous agents into a negotiation-centered risk analysis workflow. The framework enables stakeholders to simulate negotiations, systematically model dynamics, anticipate compromises, and evaluate solution impacts. By leveraging LLMs' semantic analysis capabilities we could mitigate information overload and augment decision-making process under time constraints. Proof-of-concept implementations were conducted in two real-world scenarios: (i) prudent use of a biopesticide, and (ii) targeted wild animal population control. Our work demonstrates the potential of AI-assisted negotiation to address the current lack of tools for cross-sectoral engagement. Importantly, the solution's open source, web based design, suits for application by a broader audience with limited resources and enables users to tailor and develop it for their own needs. 

---
# Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision 

**Authors**: Hanbit Oh, Masaki Murooka, Tomohiro Motoda, Ryoichi Nakajo, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2509.09893)  

**Abstract**: Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at this https URL . 

---
# Automated Tuning for Diffusion Inverse Problem Solvers without Generative Prior Retraining 

**Authors**: Yaşar Utku Alçalar, Junno Yun, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2509.09880)  

**Abstract**: Diffusion/score-based models have recently emerged as powerful generative priors for solving inverse problems, including accelerated MRI reconstruction. While their flexibility allows decoupling the measurement model from the learned prior, their performance heavily depends on carefully tuned data fidelity weights, especially under fast sampling schedules with few denoising steps. Existing approaches often rely on heuristics or fixed weights, which fail to generalize across varying measurement conditions and irregular timestep schedules. In this work, we propose Zero-shot Adaptive Diffusion Sampling (ZADS), a test-time optimization method that adaptively tunes fidelity weights across arbitrary noise schedules without requiring retraining of the diffusion prior. ZADS treats the denoising process as a fixed unrolled sampler and optimizes fidelity weights in a self-supervised manner using only undersampled measurements. Experiments on the fastMRI knee dataset demonstrate that ZADS consistently outperforms both traditional compressed sensing and recent diffusion-based methods, showcasing its ability to deliver high-fidelity reconstructions across varying noise schedules and acquisition settings. 

---
# From Hugging Face to GitHub: Tracing License Drift in the Open-Source AI Ecosystem 

**Authors**: James Jewitt, Hao Li, Bram Adams, Gopi Krishnan Rajbahadur, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09873)  

**Abstract**: Hidden license conflicts in the open-source AI ecosystem pose serious legal and ethical risks, exposing organizations to potential litigation and users to undisclosed risk. However, the field lacks a data-driven understanding of how frequently these conflicts occur, where they originate, and which communities are most affected. We present the first end-to-end audit of licenses for datasets and models on Hugging Face, as well as their downstream integration into open-source software applications, covering 364 thousand datasets, 1.6 million models, and 140 thousand GitHub projects. Our empirical analysis reveals systemic non-compliance in which 35.5% of model-to-application transitions eliminate restrictive license clauses by relicensing under permissive terms. In addition, we prototype an extensible rule engine that encodes almost 200 SPDX and model-specific clauses for detecting license conflicts, which can solve 86.4% of license conflicts in software applications. To support future research, we release our dataset and the prototype engine. Our study highlights license compliance as a critical governance challenge in open-source AI and provides both the data and tools necessary to enable automated, AI-aware compliance at scale. 

---
# Emulating Public Opinion: A Proof-of-Concept of AI-Generated Synthetic Survey Responses for the Chilean Case 

**Authors**: Bastián González-Bustamante, Nando Verelst, Carla Cisternas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09871)  

**Abstract**: Large Language Models (LLMs) offer promising avenues for methodological and applied innovations in survey research by using synthetic respondents to emulate human answers and behaviour, potentially mitigating measurement and representation errors. However, the extent to which LLMs recover aggregate item distributions remains uncertain and downstream applications risk reproducing social stereotypes and biases inherited from training data. We evaluate the reliability of LLM-generated synthetic survey responses against ground-truth human responses from a Chilean public opinion probabilistic survey. Specifically, we benchmark 128 prompt-model-question triplets, generating 189,696 synthetic profiles, and pool performance metrics (i.e., accuracy, precision, recall, and F1-score) in a meta-analysis across 128 question-subsample pairs to test for biases along key sociodemographic dimensions. The evaluation spans OpenAI's GPT family and o-series reasoning models, as well as Llama and Qwen checkpoints. Three results stand out. First, synthetic responses achieve excellent performance on trust items (F1-score and accuracy > 0.90). Second, GPT-4o, GPT-4o-mini and Llama 4 Maverick perform comparably on this task. Third, synthetic-human alignment is highest among respondents aged 45-59. Overall, LLM-based synthetic samples approximate responses from a probabilistic sample, though with substantial item-level heterogeneity. Capturing the full nuance of public opinion remains challenging and requires careful calibration and additional distributional tests to ensure algorithmic fidelity and reduce errors. 

---
# Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks 

**Authors**: Hasibur Rahman, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09870)  

**Abstract**: Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent. 

---
# Surrogate Supervision for Robust and Generalizable Deformable Image Registration 

**Authors**: Yihao Liu, Junyu Chen, Lianrui Zuo, Shuwen Wei, Brian D. Boyd, Carmen Andreescu, Olusola Ajilore, Warren D. Taylor, Aaron Carass, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2509.09869)  

**Abstract**: Objective: Deep learning-based deformable image registration has achieved strong accuracy, but remains sensitive to variations in input image characteristics such as artifacts, field-of-view mismatch, or modality difference. We aim to develop a general training paradigm that improves the robustness and generalizability of registration networks. Methods: We introduce surrogate supervision, which decouples the input domain from the supervision domain by applying estimated spatial transformations to surrogate images. This allows training on heterogeneous inputs while ensuring supervision is computed in domains where similarity is well defined. We evaluate the framework through three representative applications: artifact-robust brain MR registration, mask-agnostic lung CT registration, and multi-modal MR registration. Results: Across tasks, surrogate supervision demonstrated strong resilience to input variations including inhomogeneity field, inconsistent field-of-view, and modality differences, while maintaining high performance on well-curated data. Conclusions: Surrogate supervision provides a principled framework for training robust and generalizable deep learning-based registration models without increasing complexity. Significance: Surrogate supervision offers a practical pathway to more robust and generalizable medical image registration, enabling broader applicability in diverse biomedical imaging scenarios. 

---
# Latency and Token-Aware Test-Time Compute 

**Authors**: Jenny Y. Huang, Mehul Damani, Yousef El-Kurdi, Ramon Astudillo, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09864)  

**Abstract**: Inference-time scaling has emerged as a powerful way to improve large language model (LLM) performance by generating multiple candidate responses and selecting among them. However, existing work on dynamic allocation for test-time compute typically considers only parallel generation methods such as best-of-N, overlooking incremental decoding methods like beam search, and has largely ignored latency, focusing only on token usage. We formulate inference-time scaling as a problem of dynamic compute allocation and method selection, where the system must decide which strategy to apply and how much compute to allocate on a per-query basis. Our framework explicitly incorporates both token cost and wall-clock latency, the latter being critical for user experience and particularly for agentic workflows where models must issue multiple queries efficiently. Experiments on reasoning benchmarks show that our approach consistently outperforms static strategies, achieving favorable accuracy-cost trade-offs while remaining practical for deployment. 

---
# SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints 

**Authors**: Zhiyu Fan, Kirill Vasilevski, Dayi Lin, Boyuan Chen, Yihao Chen, Zhiqing Zhong, Jie M. Zhang, Pinjia He, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09853)  

**Abstract**: The advancement of large language models (LLMs) and code agents has demonstrated significant potential to assist software engineering (SWE) tasks, such as autonomous issue resolution and feature addition. Existing AI for software engineering leaderboards (e.g., SWE-bench) focus solely on solution accuracy, ignoring the crucial factor of effectiveness in a resource-constrained world. This is a universal problem that also exists beyond software engineering tasks: any AI system should be more than correct - it must also be cost-effective. To address this gap, we introduce SWE-Effi, a set of new metrics to re-evaluate AI systems in terms of holistic effectiveness scores. We define effectiveness as the balance between the accuracy of outcome (e.g., issue resolve rate) and the resources consumed (e.g., token and time). In this paper, we specifically focus on the software engineering scenario by re-ranking popular AI systems for issue resolution on a subset of the SWE-bench benchmark using our new multi-dimensional metrics. We found that AI system's effectiveness depends not just on the scaffold itself, but on how well it integrates with the base model, which is key to achieving strong performance in a resource-efficient manner. We also identified systematic challenges such as the "token snowball" effect and, more significantly, a pattern of "expensive failures". In these cases, agents consume excessive resources while stuck on unsolvable tasks - an issue that not only limits practical deployment but also drives up the cost of failed rollouts during RL training. Lastly, we observed a clear trade-off between effectiveness under the token budget and effectiveness under the time budget, which plays a crucial role in managing project budgets and enabling scalable reinforcement learning, where fast responses are essential. 

---
# HGEN: Heterogeneous Graph Ensemble Networks 

**Authors**: Jiajun Shen, Yufei Jin, Yi He, Xingquan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09843)  

**Abstract**: This paper presents HGEN that pioneers ensemble learning for heterogeneous graphs. We argue that the heterogeneity in node types, nodal features, and local neighborhood topology poses significant challenges for ensemble learning, particularly in accommodating diverse graph learners. Our HGEN framework ensembles multiple learners through a meta-path and transformation-based optimization pipeline to uplift classification accuracy. Specifically, HGEN uses meta-path combined with random dropping to create Allele Graph Neural Networks (GNNs), whereby the base graph learners are trained and aligned for later ensembling. To ensure effective ensemble learning, HGEN presents two key components: 1) a residual-attention mechanism to calibrate allele GNNs of different meta-paths, thereby enforcing node embeddings to focus on more informative graphs to improve base learner accuracy, and 2) a correlation-regularization term to enlarge the disparity among embedding matrices generated from different meta-paths, thereby enriching base learner diversity. We analyze the convergence of HGEN and attest its higher regularization magnitude over simple voting. Experiments on five heterogeneous networks validate that HGEN consistently outperforms its state-of-the-art competitors by substantial margin. 

---
# Revisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning 

**Authors**: Reza Asad, Reza Babanezhad, Sharan Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2509.09838)  

**Abstract**: Value-based approaches such as DQN are the default methods for off-policy reinforcement learning with discrete-action environments such as Atari. Common policy-based methods are either on-policy and do not effectively learn from off-policy data (e.g. PPO), or have poor empirical performance in the discrete-action setting (e.g. SAC). Consequently, starting from discrete SAC (DSAC), we revisit the design of actor-critic methods in this setting. First, we determine that the coupling between the actor and critic entropy is the primary reason behind the poor performance of DSAC. We demonstrate that by merely decoupling these components, DSAC can have comparable performance as DQN. Motivated by this insight, we introduce a flexible off-policy actor-critic framework that subsumes DSAC as a special case. Our framework allows using an m-step Bellman operator for the critic update, and enables combining standard policy optimization methods with entropy regularization to instantiate the resulting actor objective. Theoretically, we prove that the proposed methods can guarantee convergence to the optimal regularized value function in the tabular setting. Empirically, we demonstrate that these methods can approach the performance of DQN on standard Atari games, and do so even without entropy regularization or explicit exploration. 

---
# CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio 

**Authors**: Marco Pasini, Stefan Lattner, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09836)  

**Abstract**: Efficiently representing audio signals in a compressed latent space is critical for latent generative modelling. However, existing autoencoders often force a choice between continuous embeddings and discrete tokens. Furthermore, achieving high compression ratios while maintaining audio fidelity remains a challenge. We introduce CoDiCodec, a novel audio autoencoder that overcomes these limitations by both efficiently encoding global features via summary embeddings, and by producing both compressed continuous embeddings at ~ 11 Hz and discrete tokens at a rate of 2.38 kbps from the same trained model, offering unprecedented flexibility for different downstream generative tasks. This is achieved through Finite Scalar Quantization (FSQ) and a novel FSQ-dropout technique, and does not require additional loss terms beyond the single consistency loss used for end-to-end training. CoDiCodec supports both autoregressive decoding and a novel parallel decoding strategy, with the latter achieving superior audio quality and faster decoding. CoDiCodec outperforms existing continuous and discrete autoencoders at similar bitrates in terms of reconstruction audio quality. Our work enables a unified approach to audio compression, bridging the gap between continuous and discrete generative modelling paradigms. 

---
# SoilSound: Smartphone-based Soil Moisture Estimation 

**Authors**: Yixuan Gao, Tanvir Ahmed, Shuang He, Zhongqi Cheng, Rajalakshmi Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09823)  

**Abstract**: Soil moisture monitoring is essential for agriculture and environmental management, yet existing methods require either invasive probes disturbing the soil or specialized equipment, limiting access to the public. We present SoilSound, an ubiquitous accessible smartphone-based acoustic sensing system that can measure soil moisture without disturbing the soil. We leverage the built-in speaker and microphone to perform a vertical scan mechanism to accurately measure moisture without any calibration. Unlike existing work that use transmissive properties, we propose an alternate model for acoustic reflections in soil based on the surface roughness effect to enable moisture sensing without disturbing the soil. The system works by sending acoustic chirps towards the soil and recording the reflections during a vertical scan, which are then processed and fed to a convolutional neural network for on-device soil moisture estimation with negligible computational, memory, or power overhead. We evaluated the system by training with curated soils in boxes in the lab and testing in the outdoor fields and show that SoilSound achieves a mean absolute error (MAE) of 2.39% across 10 different locations. Overall, the evaluation shows that SoilSound can accurately track soil moisture levels ranging from 15.9% to 34.0% across multiple soil types, environments, and users; without requiring any calibration or disturbing the soil, enabling widespread moisture monitoring for home gardeners, urban farmers, citizen scientists, and agricultural communities in resource-limited settings. 

---
# HEFT: A Coarse-to-Fine Hierarchy for Enhancing the Efficiency and Accuracy of Language Model Reasoning 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.09801)  

**Abstract**: The adaptation of large language models (LLMs) to specialized reasoning tasks is fundamentally constrained by computational resources. Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a powerful solution, yet the landscape of these techniques is diverse, with distinct methods operating in either the model's weight space or its representation space. This paper investigates the hypothesis that a synergistic combination of these paradigms can unlock superior performance and efficiency. We introduce HEFT (Hierarchical Efficient Fine-Tuning), a novel hierarchical adaptation strategy that composes two distinct PEFT methods in a coarse-to-fine manner: first, a broad, foundational adaptation in the weight space using Low-Rank Adaptation (LoRA), followed by a precise, surgical refinement of internal activations using Representation Fine-Tuning (ReFT). We evaluate this approach by fine-tuning a Llama-2-7B model on the BoolQ benchmark, a challenging dataset for inferential reasoning. Our results reveal a profound synergistic effect. A model fine-tuned for only three epochs with our HEFT strategy achieves an accuracy of 85.17\%, exceeding the performance of models trained for 20 epochs with either LoRA-only (85.05\%) or ReFT-only (83.36\%) methodologies. This work demonstrates that the thoughtful composition of PEFT methods is a potent algorithmic innovation, offering a more efficient and effective path toward advancing the reasoning capabilities of language models. By achieving superior results with a fraction of the computational budget, our findings present a principled approach to overcoming the obstacles inherent in adapting large-scale models for complex cognitive tasks. 

---
# ZORRO: Zero-Knowledge Robustness and Privacy for Split Learning (Full Version) 

**Authors**: Nojan Sheybani, Alessandro Pegoraro, Jonathan Knauer, Phillip Rieger, Elissa Mollakuqe, Farinaz Koushanfar, Ahmad-Reza Sadeghi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09787)  

**Abstract**: Split Learning (SL) is a distributed learning approach that enables resource-constrained clients to collaboratively train deep neural networks (DNNs) by offloading most layers to a central server while keeping in- and output layers on the client-side. This setup enables SL to leverage server computation capacities without sharing data, making it highly effective in resource-constrained environments dealing with sensitive data. However, the distributed nature enables malicious clients to manipulate the training process. By sending poisoned intermediate gradients, they can inject backdoors into the shared DNN. Existing defenses are limited by often focusing on server-side protection and introducing additional overhead for the server. A significant challenge for client-side defenses is enforcing malicious clients to correctly execute the defense algorithm.
We present ZORRO, a private, verifiable, and robust SL defense scheme. Through our novel design and application of interactive zero-knowledge proofs (ZKPs), clients prove their correct execution of a client-located defense algorithm, resulting in proofs of computational integrity attesting to the benign nature of locally trained DNN portions. Leveraging the frequency representation of model partitions enables ZORRO to conduct an in-depth inspection of the locally trained models in an untrusted environment, ensuring that each client forwards a benign checkpoint to its succeeding client. In our extensive evaluation, covering different model architectures as well as various attack strategies and data scenarios, we show ZORRO's effectiveness, as it reduces the attack success rate to less than 6\% while causing even for models storing \numprint{1000000} parameters on the client-side an overhead of less than 10 seconds. 

---
# LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation 

**Authors**: Yiqun Shen, Song Yuan, Zhengze Zhang, Xiaoliang Wang, Daxin Jiang, Nguyen Cam-Tu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09754)  

**Abstract**: KV Cache is commonly used to accelerate LLM inference with long contexts, yet its high memory demand drives the need for cache compression. Existing compression methods, however, are largely heuristic and lack dynamic budget allocation. To address this limitation, we introduce a unified framework for cache compression by minimizing information loss in Transformer residual streams. Building on it, we analyze the layer attention output loss and derive a new metric to compare cache entries across heads, enabling layer-wise compression with dynamic head budgets. Additionally, by contrasting cross-layer information, we also achieve dynamic layer budgets. LAVa is the first unified strategy for cache eviction and dynamic budget allocation that, unlike prior methods, does not rely on training or the combination of multiple strategies. Experiments with benchmarks (LongBench, Needle-In-A-Haystack, Ruler, and InfiniteBench) demonstrate its superiority. Moreover, our experiments reveal a new insight: dynamic layer budgets are crucial for generation tasks (e.g., code completion), while dynamic head budgets play a key role in extraction tasks (e.g., extractive QA). As a fully dynamic compression method, LAVa consistently maintains top performance across task types. Our code is available at this https URL. 

---
# Meta-Learning Reinforcement Learning for Crypto-Return Prediction 

**Authors**: Junqiao Wang, Zhaoyang Guan, Guanyu Liu, Tianze Xia, Xianzhi Li, Shuo Yin, Xinyuan Song, Chuhan Cheng, Tianyu Shi, Alex Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09751)  

**Abstract**: Predicting cryptocurrency returns is notoriously difficult: price movements are driven by a fast-shifting blend of on-chain activity, news flow, and social sentiment, while labeled training data are scarce and expensive. In this paper, we present Meta-RL-Crypto, a unified transformer-based architecture that unifies meta-learning and reinforcement learning (RL) to create a fully self-improving trading agent. Starting from a vanilla instruction-tuned LLM, the agent iteratively alternates between three roles-actor, judge, and meta-judge-in a closed-loop architecture. This learning process requires no additional human supervision. It can leverage multimodal market inputs and internal preference feedback. The agent in the system continuously refines both the trading policy and evaluation criteria. Experiments across diverse market regimes demonstrate that Meta-RL-Crypto shows good performance on the technical indicators of the real market and outperforming other LLM-based baselines. 

---
# A Co-Training Semi-Supervised Framework Using Faster R-CNN and YOLO Networks for Object Detection in Densely Packed Retail Images 

**Authors**: Hossein Yazdanjouei, Arash Mansouri, Mohammad Shokouhifar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09750)  

**Abstract**: This study proposes a semi-supervised co-training framework for object detection in densely packed retail environments, where limited labeled data and complex conditions pose major challenges. The framework combines Faster R-CNN (utilizing a ResNet backbone) for precise localization with YOLO (employing a Darknet backbone) for global context, enabling mutual pseudo-label exchange that improves accuracy in scenes with occlusion and overlapping objects. To strengthen classification, it employs an ensemble of XGBoost, Random Forest, and SVM, utilizing diverse feature representations for higher robustness. Hyperparameters are optimized using a metaheuristic-driven algorithm, enhancing precision and efficiency across models. By minimizing reliance on manual labeling, the approach reduces annotation costs and adapts effectively to frequent product and layout changes common in retail. Experiments on the SKU-110k dataset demonstrate strong performance, highlighting the scalability and practicality of the proposed framework for real-world retail applications such as automated inventory tracking, product monitoring, and checkout systems. 

---
# D-CAT: Decoupled Cross-Attention Transfer between Sensor Modalities for Unimodal Inference 

**Authors**: Leen Daher, Zhaobo Wang, Malcolm Mielle  

**Link**: [PDF](https://arxiv.org/pdf/2509.09747)  

**Abstract**: Cross-modal transfer learning is used to improve multi-modal classification models (e.g., for human activity recognition in human-robot collaboration). However, existing methods require paired sensor data at both training and inference, limiting deployment in resource-constrained environments where full sensor suites are not economically and technically usable. To address this, we propose Decoupled Cross-Attention Transfer (D-CAT), a framework that aligns modality-specific representations without requiring joint sensor modality during inference. Our approach combines a self-attention module for feature extraction with a novel cross-attention alignment loss, which enforces the alignment of sensors' feature spaces without requiring the coupling of the classification pipelines of both modalities. We evaluate D-CAT on three multi-modal human activity datasets (IMU, video, and audio) under both in-distribution and out-of-distribution scenarios, comparing against uni-modal models. Results show that in in-distribution scenarios, transferring from high-performing modalities (e.g., video to IMU) yields up to 10% F1-score gains over uni-modal training. In out-of-distribution scenarios, even weaker source modalities (e.g., IMU to video) improve target performance, as long as the target model isn't overfitted on the training data. By enabling single-sensor inference with cross-modal knowledge, D-CAT reduces hardware redundancy for perception systems while maintaining accuracy, which is critical for cost-sensitive or adaptive deployments (e.g., assistive robots in homes with variable sensor availability). Code is available at this https URL. 

---
# Structure Matters: Brain Graph Augmentation via Learnable Edge Masking for Data-efficient Psychiatric Diagnosis 

**Authors**: Mujie Liu, Chenze Wang, Liping Chen, Nguyen Linh Dan Le, Niharika Tewari, Ting Dang, Jiangang Ma, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.09744)  

**Abstract**: The limited availability of labeled brain network data makes it challenging to achieve accurate and interpretable psychiatric diagnoses. While self-supervised learning (SSL) offers a promising solution, existing methods often rely on augmentation strategies that can disrupt crucial structural semantics in brain graphs. To address this, we propose SAM-BG, a two-stage framework for learning brain graph representations with structural semantic preservation. In the pre-training stage, an edge masker is trained on a small labeled subset to capture key structural semantics. In the SSL stage, the extracted structural priors guide a structure-aware augmentation process, enabling the model to learn more semantically meaningful and robust representations. Experiments on two real-world psychiatric datasets demonstrate that SAM-BG outperforms state-of-the-art methods, particularly in small-labeled data settings, and uncovers clinically relevant connectivity patterns that enhance interpretability. Our code is available at this https URL. 

---
# HypoGeneAgent: A Hypothesis Language Agent for Gene-Set Cluster Resolution Selection Using Perturb-seq Datasets 

**Authors**: Ying Yuan, Xing-Yue Monica Ge, Aaron Archer Waterman, Tommaso Biancalani, David Richmond, Yogesh Pandit, Avtar Singh, Russell Littman, Jin Liu, Jan-Christian Huetter, Vladimir Ermakov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09740)  

**Abstract**: Large-scale single-cell and Perturb-seq investigations routinely involve clustering cells and subsequently annotating each cluster with Gene-Ontology (GO) terms to elucidate the underlying biological programs. However, both stages, resolution selection and functional annotation, are inherently subjective, relying on heuristics and expert curation. We present HYPOGENEAGENT, a large language model (LLM)-driven framework, transforming cluster annotation into a quantitatively optimizable task. Initially, an LLM functioning as a gene-set analyst analyzes the content of each gene program or perturbation module and generates a ranked list of GO-based hypotheses, accompanied by calibrated confidence scores. Subsequently, we embed every predicted description with a sentence-embedding model, compute pair-wise cosine similarities, and let the agent referee panel score (i) the internal consistency of the predictions, high average similarity within the same cluster, termed intra-cluster agreement (ii) their external distinctiveness, low similarity between clusters, termed inter-cluster separation. These two quantities are combined to produce an agent-derived resolution score, which is maximized when clusters exhibit simultaneous coherence and mutual exclusivity. When applied to a public K562 CRISPRi Perturb-seq dataset as a preliminary test, our Resolution Score selects clustering granularities that exhibit alignment with known pathway compared to classical metrics such silhouette score, modularity score for gene functional enrichment summary. These findings establish LLM agents as objective adjudicators of cluster resolution and functional annotation, thereby paving the way for fully automated, context-aware interpretation pipelines in single-cell multi-omics studies. 

---
# World Modeling with Probabilistic Structure Integration 

**Authors**: Klemen Kotar, Wanhee Lee, Rahul Venkatesh, Honglin Chen, Daniel Bear, Jared Watrous, Simon Kim, Khai Loong Aw, Lilian Naing Chen, Stefan Stojanov, Kevin Feigelis, Imran Thobani, Alex Durango, Khaled Jedoui, Atlas Kazemian, Dan Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2509.09737)  

**Abstract**: We present Probabilistic Structure Integration (PSI), a system for learning richly controllable and flexibly promptable world models from data. PSI consists of a three-step cycle. The first step, Probabilistic prediction, involves building a probabilistic graphical model Psi of the data, in the form of a random-access autoregressive sequence model. Psi supports a complete set of learned conditional distributions describing the dependence of any variables in the data on any other set of variables. In step 2, Structure extraction, we show how to extract underlying low-dimensional properties in the data, corresponding to a diverse set of meaningful "intermediate structures", in a zero-shot fashion via causal inference on Psi. Step 3, Integration, completes the cycle by converting these structures into new token types that are then continually mixed back into the training diet as conditioning signals and prediction targets. Each such cycle augments the capabilities of Psi, both allowing it to model the underlying data better, and creating new control handles -- akin to an LLM-like universal prompting language. We train an instance of Psi on 1.4 trillion tokens of internet video data; we use it to perform a variety of useful video prediction and understanding inferences; we extract state-of-the-art optical flow, self-supervised depth and object segmentation; and we use these structures to support a full cycle of predictive improvements. 

---
# MCP-AgentBench: Evaluating Real-World Language Agent Performance with MCP-Mediated Tools 

**Authors**: Zikang Guo, Benfeng Xu, Chiwei Zhu, Wentao Hong, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09734)  

**Abstract**: The Model Context Protocol (MCP) is rapidly emerging as a pivotal open standard, designed to enhance agent-tool integration and interoperability, and is positioned to unlock a new era of powerful, interconnected, and genuinely utilitarian agentic AI. However, despite MCP's growing adoption, existing benchmarks often fail to capture real-world agent performance within this new paradigm, leading to a distorted perception of their true operational value and an inability to reliably differentiate proficiencies. To bridge this critical evaluation gap, we introduce MCP-AgentBench -- a comprehensive benchmark specifically engineered to rigorously assess language agent capabilities in MCP-mediated tool interactions. Core contributions of MCP-AgentBench include: the establishment of a robust MCP testbed comprising 33 operational servers with 188 distinct tools; the development of a benchmark featuring 600 systematically designed queries distributed across 6 distinct categories of varying interaction complexity; and the introduction of MCP-Eval, a novel outcome-oriented evaluation methodology prioritizing real-world task success. Through extensive empirical evaluation of leading language agents, we provide foundational insights. MCP-AgentBench aims to equip the research community with a standardized and reliable framework to build, validate, and advance agents capable of fully leveraging MCP's transformative benefits, thereby accelerating progress toward truly capable and interoperable AI systems. 

---
# MITS: A Large-Scale Multimodal Benchmark Dataset for Intelligent Traffic Surveillance 

**Authors**: Kaikai Zhao, Zhaoxiang Liu, Peng Wang, Xin Wang, Zhicheng Ma, Yajun Xu, Wenjing Zhang, Yibing Nan, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.09730)  

**Abstract**: General-domain large multimodal models (LMMs) have achieved significant advances in various image-text tasks. However, their performance in the Intelligent Traffic Surveillance (ITS) domain remains limited due to the absence of dedicated multimodal datasets. To address this gap, we introduce MITS (Multimodal Intelligent Traffic Surveillance), the first large-scale multimodal benchmark dataset specifically designed for ITS. MITS includes 170,400 independently collected real-world ITS images sourced from traffic surveillance cameras, annotated with eight main categories and 24 subcategories of ITS-specific objects and events under diverse environmental conditions. Additionally, through a systematic data generation pipeline, we generate high-quality image captions and 5 million instruction-following visual question-answer pairs, addressing five critical ITS tasks: object and event recognition, object counting, object localization, background analysis, and event reasoning. To demonstrate MITS's effectiveness, we fine-tune mainstream LMMs on this dataset, enabling the development of ITS-specific applications. Experimental results show that MITS significantly improves LMM performance in ITS applications, increasing LLaVA-1.5's performance from 0.494 to 0.905 (+83.2%), LLaVA-1.6's from 0.678 to 0.921 (+35.8%), Qwen2-VL's from 0.584 to 0.926 (+58.6%), and Qwen2.5-VL's from 0.732 to 0.930 (+27.0%). We release the dataset, code, and models as open-source, providing high-value resources to advance both ITS and LMM research. 

---
# MultimodalHugs: Enabling Sign Language Processing in Hugging Face 

**Authors**: Gerard Sant, Zifan Jiang, Carlos Escolano, Amit Moryossef, Mathias Müller, Rico Sennrich, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09729)  

**Abstract**: In recent years, sign language processing (SLP) has gained importance in the general field of Natural Language Processing. However, compared to research on spoken languages, SLP research is hindered by complex ad-hoc code, inadvertently leading to low reproducibility and unfair comparisons. Existing tools that are built for fast and reproducible experimentation, such as Hugging Face, are not flexible enough to seamlessly integrate sign language experiments. This view is confirmed by a survey we conducted among SLP researchers.
To address these challenges, we introduce MultimodalHugs, a framework built on top of Hugging Face that enables more diverse data modalities and tasks, while inheriting the well-known advantages of the Hugging Face ecosystem. Even though sign languages are our primary focus, MultimodalHugs adds a layer of abstraction that makes it more widely applicable to other use cases that do not fit one of the standard templates of Hugging Face. We provide quantitative experiments to illustrate how MultimodalHugs can accommodate diverse modalities such as pose estimation data for sign languages, or pixel data for text characters. 

---
# DiTTO-LLM: Framework for Discovering Topic-based Technology Opportunities via Large Language Model 

**Authors**: Wonyoung Kim, Sujeong Seo, Juhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09724)  

**Abstract**: Technology opportunities are critical information that serve as a foundation for advancements in technology, industry, and innovation. This paper proposes a framework based on the temporal relationships between technologies to identify emerging technology opportunities. The proposed framework begins by extracting text from a patent dataset, followed by mapping text-based topics to discover inter-technology relationships. Technology opportunities are then identified by tracking changes in these topics over time. To enhance efficiency, the framework leverages a large language model to extract topics and employs a prompt for a chat-based language model to support the discovery of technology opportunities. The framework was evaluated using an artificial intelligence patent dataset provided by the United States Patent and Trademark Office. The experimental results suggest that artificial intelligence technology is evolving into forms that facilitate everyday accessibility. This approach demonstrates the potential of the proposed framework to identify future technology opportunities. 

---
# ALIGNS: Unlocking nomological networks in psychological measurement through a large language model 

**Authors**: Kai R. Larsen, Sen Yan, Roland Müller, Lan Sang, Mikko Rönkkö, Ravi Starzl, Donald Edmondson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09723)  

**Abstract**: Psychological measurement is critical to many disciplines. Despite advances in measurement, building nomological networks, theoretical maps of how concepts and measures relate to establish validity, remains a challenge 70 years after Cronbach and Meehl proposed them as fundamental to validation. This limitation has practical consequences: clinical trials may fail to detect treatment effects, and public policy may target the wrong outcomes. We introduce Analysis of Latent Indicators to Generate Nomological Structures (ALIGNS), a large language model-based system trained with validated questionnaire measures. ALIGNS provides three comprehensive nomological networks containing over 550,000 indicators across psychology, medicine, social policy, and other fields. This represents the first application of large language models to solve a foundational problem in measurement validation. We report classification accuracy tests used to develop the model, as well as three evaluations. In the first evaluation, the widely used NIH PROMIS anxiety and depression instruments are shown to converge into a single dimension of emotional distress. The second evaluation examines child temperament measures and identifies four potential dimensions not captured by current frameworks, and questions one existing dimension. The third evaluation, an applicability check, engages expert psychometricians who assess the system's importance, accessibility, and suitability. ALIGNS is freely available at this http URL, complementing traditional validation methods with large-scale nomological analysis. 

---
# A Multimodal RAG Framework for Housing Damage Assessment: Collaborative Optimization of Image Encoding and Policy Vector Retrieval 

**Authors**: Jiayi Miao, Dingxin Lu, Zhuqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09721)  

**Abstract**: After natural disasters, accurate evaluations of damage to housing are important for insurance claims response and planning of resources. In this work, we introduce a novel multimodal retrieval-augmented generation (MM-RAG) framework. On top of classical RAG architecture, we further the framework to devise a two-branch multimodal encoder structure that the image branch employs a visual encoder composed of ResNet and Transformer to extract the characteristic of building damage after disaster, and the text branch harnesses a BERT retriever for the text vectorization of posts as well as insurance policies and for the construction of a retrievable restoration index. To impose cross-modal semantic alignment, the model integrates a cross-modal interaction module to bridge the semantic representation between image and text via multi-head attention. Meanwhile, in the generation module, the introduced modal attention gating mechanism dynamically controls the role of visual evidence and text prior information during generation. The entire framework takes end-to-end training, and combines the comparison loss, the retrieval loss and the generation loss to form multi-task optimization objectives, and achieves image understanding and policy matching in collaborative learning. The results demonstrate superior performance in retrieval accuracy and classification index on damage severity, where the Top-1 retrieval accuracy has been improved by 9.6%. 

---
# VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions 

**Authors**: Jun Zhan, Mingyang Han, Yuxuan Xie, Chen Wang, Dong Zhang, Kexin Huang, Haoxiang Shi, DongXiao Wang, Tengtao Song, Qinyuan Cheng, Shimin Li, Jun Song, Xipeng Qiu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09716)  

**Abstract**: Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{this https URL}{project's homepage}. 

---
# Investigating Symbolic Triggers of Hallucination in Gemma Models Across HaluEval and TruthfulQA 

**Authors**: Naveen Lamba, Sanju Tiwari, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2509.09715)  

**Abstract**: Hallucination in Large Language Models (LLMs) is a well studied problem. However, the properties that make LLM intrinsically vulnerable to hallucinations have not been identified and studied. This research identifies and characterizes the key properties, allowing us to pinpoint vulnerabilities within the model's internal mechanisms. To solidify on these properties, we utilized two established datasets, HaluEval and TruthfulQA and convert their existing format of question answering into various other formats to narrow down these properties as the reason for the hallucinations. Our findings reveal that hallucination percentages across symbolic properties are notably high for Gemma-2-2B, averaging 79.0% across tasks and datasets. With increased model scale, hallucination drops to 73.6% for Gemma-2-9B and 63.9% for Gemma-2-27B, reflecting a 15 percentage point reduction overall. Although the hallucination rate decreases as the model size increases, a substantial amount of hallucination caused by symbolic properties still persists. This is especially evident for modifiers (ranging from 84.76% to 94.98%) and named entities (ranging from 83.87% to 93.96%) across all Gemma models and both datasets. These findings indicate that symbolic elements continue to confuse the models, pointing to a fundamental weakness in how these LLMs process such inputs--regardless of their scale. 

---
# How Small Transformation Expose the Weakness of Semantic Similarity Measures 

**Authors**: Serge Lionel Nikiema, Albérick Euraste Djire, Abdoul Aziz Bonkoungou, Micheline Bénédicte Moumoula, Jordan Samhi, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.09714)  

**Abstract**: This research examines how well different methods measure semantic similarity, which is important for various software engineering applications such as code search, API recommendations, automated code reviews, and refactoring tools. While large language models are increasingly used for these similarity assessments, questions remain about whether they truly understand semantic relationships or merely recognize surface patterns.
The study tested 18 different similarity measurement approaches, including word-based methods, embedding techniques, LLM-based systems, and structure-aware algorithms. The researchers created a systematic testing framework that applies controlled changes to text and code to evaluate how well each method handles different types of semantic relationships.
The results revealed significant issues with commonly used metrics. Some embedding-based methods incorrectly identified semantic opposites as similar up to 99.9 percent of the time, while certain transformer-based approaches occasionally rated opposite meanings as more similar than synonymous ones. The study found that embedding methods' poor performance often stemmed from how they calculate distances; switching from Euclidean distance to cosine similarity improved results by 24 to 66 percent. LLM-based approaches performed better at distinguishing semantic differences, producing low similarity scores (0.00 to 0.29) for genuinely different meanings, compared to embedding methods that incorrectly assigned high scores (0.82 to 0.99) to dissimilar content. 

---
# HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering 

**Authors**: Duolin Sun, Dan Yang, Yue Shen, Yihan Jiao, Zhehao Tan, Jie Feng, Lianzhen Zhong, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09713)  

**Abstract**: The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks. 

---
# The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy using Supervised Fine-Tuning and Odds Ratio Policy Optimization 

**Authors**: Talha Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09712)  

**Abstract**: Acceptance and Commitment Therapy (ACT) is a third-wave cognitive behavioral therapy with emerging evidence of efficacy in several psychiatric conditions. This study investigates the impact of post-training methodology and explicit reasoning on the ability of a small open-weight large language model (LLM) to deliver ACT. Using 50 sets of synthetic ACT transcripts generated by Mistral-Large, we trained Llama-3.2-3b-Instruct with two distinct approaches, supervised fine-tuning (SFT) and odds ratio policy optimization (ORPO), each with and without an explicit chain-of-thought (COT) reasoning step. Performance was evaluated by comparing these four post-trained variants against the base Instruct model. These models were benchmarked in simulated therapy sessions, with performance quantitatively assessed on the ACT Fidelity Measure (ACT-FM) and the Therapist Empathy Scale (TES) by an LLM judge that had been fine-tuned on human evaluations. Our findings demonstrate that the ORPO-trained models significantly outperformed both their SFT and Instruct counterparts on ACT fidelity ($\chi^2(5) = 185.15, p < .001$) and therapeutic empathy ($\chi^2(5) = 140.37, p < .001$). The effect of COT was conditional as it provided a significant benefit to SFT models, improving ACT-FM scores by an average of 2.68 points ($p < .001$), while offering no discernible advantage to the superior ORPO or instruct-tuned variants. We posit that the superiority of ORPO stems from its ability to learn the therapeutic `process' over imitating `content,' a key aspect of ACT, while COT acts as a necessary scaffold for models trained only via imitation. This study establishes that preference-aligned policy optimization can effectively instill ACT competencies in small LLMs, and that the utility of explicit reasoning is highly dependent on the underlying training paradigm. 

---
# Psychiatry-Bench: A Multi-Task Benchmark for LLMs in Psychiatry 

**Authors**: Aya E. Fouda, Abdelrahamn A. Hassan, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2509.09711)  

**Abstract**: Large language models (LLMs) hold great promise in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of psychiatric reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling over 5,300 expert-annotated items. We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, LLaMA 3, and QWQ-32) alongside leading open-source medical models (e.g., OpenBiloLLM, MedGemma) using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in high-stakes mental health applications. 

---
# Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data 

**Authors**: Sepehr Golrokh Amin, Devin Rhoads, Fatemeh Fakhrmoosavi, Nicholas E. Lownes, John N. Ivan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09710)  

**Abstract**: This study introduces a Large Language Model (LLM) scheme for generating individual travel diaries in agent-based transportation models. While traditional approaches rely on large quantities of proprietary household travel surveys, the method presented in this study generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. This study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. The validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compared to diaries generated with classical methods (Negative Binomial for trip generation; Multinomial Logit for mode/purpose) calibrated on the validation set, LLM-generated diaries achieve comparable overall realism (LLM mean: 0.485 vs. 0.455). The LLM excels in determining trip purpose and demonstrates greater consistency (narrower realism score distribution), while classical models lead in numerical estimates of trip count and activity duration. Aggregate validation confirms the LLM's statistical representativeness (LLM mean: 0.612 vs. 0.435), demonstrating LLM's zero-shot viability and establishing a quantifiable metric of diary realism for future synthetic diary evaluation systems. 

---
# Assisting Research Proposal Writing with Large Language Models: Evaluation and Refinement 

**Authors**: Jing Ren, Weiqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09709)  

**Abstract**: Large language models (LLMs) like ChatGPT are increasingly used in academic writing, yet issues such as incorrect or fabricated references raise ethical concerns. Moreover, current content quality evaluations often rely on subjective human judgment, which is labor-intensive and lacks objectivity, potentially compromising the consistency and reliability. In this study, to provide a quantitative evaluation and enhance research proposal writing capabilities of LLMs, we propose two key evaluation metrics--content quality and reference validity--and an iterative prompting method based on the scores derived from these two metrics. Our extensive experiments show that the proposed metrics provide an objective, quantitative framework for assessing ChatGPT's writing performance. Additionally, iterative prompting significantly enhances content quality while reducing reference inaccuracies and fabrications, addressing critical ethical challenges in academic contexts. 

---
# Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal 

**Authors**: Nirmalendu Prakash, Yeo Wei Jie, Amir Abdullah, Ranjan Satapathy, Erik Cambria, Roy Ka Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09708)  

**Abstract**: Refusal on harmful prompts is a key safety behaviour in instruction-tuned large language models (LLMs), yet the internal causes of this behaviour remain poorly understood. We study two public instruction-tuned models, Gemma-2-2B-IT and LLaMA-3.1-8B-IT, using sparse autoencoders (SAEs) trained on residual-stream activations. Given a harmful prompt, we search the SAE latent space for feature sets whose ablation flips the model from refusal to compliance, demonstrating causal influence and creating a jailbreak. Our search proceeds in three stages: (1) Refusal Direction: find a refusal-mediating direction and collect SAE features near that direction; (2) Greedy Filtering: prune to a minimal set; and (3) Interaction Discovery: fit a factorization machine (FM) that captures nonlinear interactions among the remaining active features and the minimal set. This pipeline yields a broad set of jailbreak-critical features, offering insight into the mechanistic basis of refusal. Moreover, we find evidence of redundant features that remain dormant unless earlier features are suppressed. Our findings highlight the potential for fine-grained auditing and targeted intervention in safety behaviours by manipulating the interpretable latent space. 

---
# LLM-Based Instance-Driven Heuristic Bias In the Context of a Biased Random Key Genetic Algorithm 

**Authors**: Camilo Chacón Sartori, Martín Isla Pino, Pedro Pinacho-Davidson, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2509.09707)  

**Abstract**: Integrating Large Language Models (LLMs) within metaheuristics opens a novel path for solving complex combinatorial optimization problems. While most existing approaches leverage LLMs for code generation to create or refine specific heuristics, they often overlook the structural properties of individual problem instances. In this work, we introduce a novel framework that integrates LLMs with a Biased Random-Key Genetic Algorithm (BRKGA) to solve the NP-hard Longest Run Subsequence problem. Our approach extends the instance-driven heuristic bias paradigm by introducing a human-LLM collaborative process to co-design and implement a set of computationally efficient metrics. The LLM analyzes these instance-specific metrics to generate a tailored heuristic bias, which steers the BRKGA toward promising areas of the search space. We conduct a comprehensive experimental evaluation, including rigorous statistical tests, convergence and behavioral analyses, and targeted ablation studies, comparing our method against a standard BRKGA baseline across 1,050 generated instances of varying complexity. Results show that our top-performing hybrid, BRKGA+Llama-4-Maverick, achieves statistically significant improvements over the baseline, particularly on the most complex instances. Our findings confirm that leveraging an LLM to produce an a priori, instance-driven heuristic bias is a valuable approach for enhancing metaheuristics in complex optimization domains. 

---
# Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks 

**Authors**: Taniya Gidatkar, Oluwaseun Ajao, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2509.09706)  

**Abstract**: This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies. 

---
# The Non-Determinism of Small LLMs: Evidence of Low Answer Consistency in Repetition Trials of Standard Multiple-Choice Benchmarks 

**Authors**: Claudio Pinhanez, Paulo Cavalin, Cassia Sanctos, Marcelo Grave, Yago Primerano  

**Link**: [PDF](https://arxiv.org/pdf/2509.09705)  

**Abstract**: This work explores the consistency of small LLMs (2B-8B parameters) in answering multiple times the same question. We present a study on known, open-source LLMs responding to 10 repetitions of questions from the multiple-choice benchmarks MMLU-Redux and MedQA, considering different inference temperatures, small vs. medium models (50B-80B), finetuned vs. base models, and other parameters. We also look into the effects of requiring multi-trial answer consistency on accuracy and the trade-offs involved in deciding which model best provides both of them. To support those studies, we propose some new analytical and graphical tools. Results show that the number of questions which can be answered consistently vary considerably among models but are typically in the 50%-80% range for small models at low inference temperatures. Also, accuracy among consistent answers seems to reasonably correlate with overall accuracy. Results for medium-sized models seem to indicate much higher levels of answer consistency. 

---
# Temporal Preferences in Language Models for Long-Horizon Assistance 

**Authors**: Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, Hossein Setareh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09704)  

**Abstract**: We study whether language models (LMs) exhibit future- versus present-oriented preferences in intertemporal choice and whether those preferences can be systematically manipulated. Using adapted human experimental protocols, we evaluate multiple LMs on time-tradeoff tasks and benchmark them against a sample of human decision makers. We introduce an operational metric, the Manipulability of Time Orientation (MTO), defined as the change in an LM's revealed time preference between future- and present-oriented prompts. In our tests, reasoning-focused models (e.g., DeepSeek-Reasoner and grok-3-mini) choose later options under future-oriented prompts but only partially personalize decisions across identities or geographies. Moreover, models that correctly reason about time orientation internalize a future orientation for themselves as AI decision makers. We discuss design implications for AI assistants that should align with heterogeneous, long-horizon goals and outline a research agenda on personalized contextual calibration and socially aware deployment. 

---
# CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor 

**Authors**: Zhenhua Xu, Xixiang Zhao, Xubin Yue, Shengwei Tian, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.09703)  

**Abstract**: The widespread deployment of large language models (LLMs) has intensified concerns around intellectual property (IP) protection, as model theft and unauthorized redistribution become increasingly feasible. To address this, model fingerprinting aims to embed verifiable ownership traces into LLMs. However, existing methods face inherent trade-offs between stealthness, robustness, and generalizability, being either detectable via distributional shifts, vulnerable to adversarial modifications, or easily invalidated once the fingerprint is revealed. In this work, we introduce CTCC, a novel rule-driven fingerprinting framework that encodes contextual correlations across multiple dialogue turns, such as counterfactual, rather than relying on token-level or single-turn triggers. CTCC enables fingerprint verification under black-box access while mitigating false positives and fingerprint leakage, supporting continuous construction under a shared semantic rule even if partial triggers are exposed. Extensive experiments across multiple LLM architectures demonstrate that CTCC consistently achieves stronger stealth and robustness than prior work. Our findings position CTCC as a reliable and practical solution for ownership verification in real-world LLM deployment scenarios. Our code and data are publicly available at <this https URL. 

---
# Creativity Benchmark: A benchmark for marketing creativity for LLM models 

**Authors**: Ninad Bhat, Kieran Browne, Pip Bingemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.09702)  

**Abstract**: We introduce Creativity Benchmark, an evaluation framework for large language models (LLMs) in marketing creativity. The benchmark covers 100 brands (12 categories) and three prompt types (Insights, Ideas, Wild Ideas). Human pairwise preferences from 678 practising creatives over 11,012 anonymised comparisons, analysed with Bradley-Terry models, show tightly clustered performance with no model dominating across brands or prompt types: the top-bottom spread is $\Delta\theta \approx 0.45$, which implies a head-to-head win probability of $0.61$; the highest-rated model beats the lowest only about $61\%$ of the time. We also analyse model diversity using cosine distances to capture intra- and inter-model variation and sensitivity to prompt reframing. Comparing three LLM-as-judge setups with human rankings reveals weak, inconsistent correlations and judge-specific biases, underscoring that automated judges cannot substitute for human evaluation. Conventional creativity tests also transfer only partially to brand-constrained tasks. Overall, the results highlight the need for expert human evaluation and diversity-aware workflows. 

---
# Cross-Layer Attention Probing for Fine-Grained Hallucination Detection 

**Authors**: Malavika Suresh, Rahaf Aljundi, Ikechukwu Nkisi-Orji, Nirmalie Wiratunga  

**Link**: [PDF](https://arxiv.org/pdf/2509.09700)  

**Abstract**: With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains high reliability even when applied out-of-distribution. 

---
# Structured Information Matters: Explainable ICD Coding with Patient-Level Knowledge Graphs 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Warren Del-Pinto, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2509.09699)  

**Abstract**: Mapping clinical documents to standardised clinical vocabularies is an important task, as it provides structured data for information retrieval and analysis, which is essential to clinical research, hospital administration and improving patient care. However, manual coding is both difficult and time-consuming, making it impractical at scale. Automated coding can potentially alleviate this burden, improving the availability and accuracy of structured clinical data. The task is difficult to automate, as it requires mapping to high-dimensional and long-tailed target spaces, such as the International Classification of Diseases (ICD). While external knowledge sources have been readily utilised to enhance output code representation, the use of external resources for representing the input documents has been underexplored. In this work, we compute a structured representation of the input documents, making use of document-level knowledge graphs (KGs) that provide a comprehensive structured view of a patient's condition. The resulting knowledge graph efficiently represents the patient-centred input documents with 23\% of the original text while retaining 90\% of the information. We assess the effectiveness of this graph for automated ICD-9 coding by integrating it into the state-of-the-art ICD coding architecture PLM-ICD. Our experiments yield improved Macro-F1 scores by up to 3.20\% on popular benchmarks, while improving training efficiency. We attribute this improvement to different types of entities and relationships in the KG, and demonstrate the improved explainability potential of the approach over the text-only baseline. 

---
# Wave-Based Semantic Memory with Resonance-Based Retrieval: A Phase-Aware Alternative to Vector Embedding Stores 

**Authors**: Aleksandr Listopad  

**Link**: [PDF](https://arxiv.org/pdf/2509.09691)  

**Abstract**: Conventional vector-based memory systems rely on cosine or inner product similarity within real-valued embedding spaces. While computationally efficient, such approaches are inherently phase-insensitive and limited in their ability to capture resonance phenomena crucial for meaning representation. We propose Wave-Based Semantic Memory, a novel framework that models knowledge as wave patterns $\psi(x) = A(x) e^{i\phi(x)}$ and retrieves it through resonance-based interference. This approach preserves both amplitude and phase information, enabling more expressive and robust semantic similarity. We demonstrate that resonance-based retrieval achieves higher discriminative power in cases where vector methods fail, including phase shifts, negations, and compositional queries. Our implementation, ResonanceDB, shows scalability to millions of patterns with millisecond latency, positioning wave-based memory as a viable alternative to vector stores for AGI-oriented reasoning and knowledge representation. 

---
# Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors 

**Authors**: Himanshu Thakur, Eshani Agrawal, Smruthi Mukund  

**Link**: [PDF](https://arxiv.org/pdf/2509.09689)  

**Abstract**: A long-standing challenge in developing accurate recommendation models is simulating user behavior, mainly due to the complex and stochastic nature of user interactions. Towards this, one promising line of work has been the use of Large Language Models (LLMs) for simulating user behavior. However, aligning these general-purpose large pre-trained models with user preferences necessitates: (i) effectively and continously parsing large-scale tabular user-item interaction data, (ii) overcoming pre-training-induced inductive biases to accurately learn user specific knowledge, and (iii) achieving the former two at scale for millions of users. While most previous works have focused on complex methods to prompt an LLM or fine-tune it on tabular interaction datasets, our approach shifts the focus to extracting robust textual user representations using a frozen LLM and simulating cost-effective, resource-efficient user agents powered by fine-tuned Small Language Models (SLMs). Further, we showcase a method for training multiple low-rank adapters for groups of users or \textit{persona}, striking an optimal balance between scalability and performance of user behavior agents. Our experiments provide compelling empirical evidence of the efficacy of our methods, demonstrating that user agents developed using our approach have the potential to bridge the gap between offline metrics and real-world performance of recommender systems. 

---
# AI-Powered Assistant for Long-Term Access to RHIC Knowledge 

**Authors**: Mohammad Atif, Vincent Garonne, Eric Lancon, Jerome Lauret, Alexandr Prozorov, Michal Vranovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.09688)  

**Abstract**: As the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory concludes 25 years of operation, preserving not only its vast data holdings ($\sim$1 ExaByte) but also the embedded scientific knowledge becomes a critical priority. The RHIC Data and Analysis Preservation Plan (DAPP) introduces an AI-powered assistant system that provides natural language access to documentation, workflows, and software, with the aim of supporting reproducibility, education, and future discovery. Built upon Large Language Models using Retrieval-Augmented Generation and the Model Context Protocol, this assistant indexes structured and unstructured content from RHIC experiments and enables domain-adapted interaction. We report on the deployment, computational performance, ongoing multi-experiment integration, and architectural features designed for a sustainable and explainable long-term AI access. Our experience illustrates how modern AI/ML tools can transform the usability and discoverability of scientific legacy data. 

---
# GeoGPT.RAG Technical Report 

**Authors**: Fei Huang, Fan Wu, Zeqing Zhang, Qihao Wang, Long Zhang, Grant Michael Boquet, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09686)  

**Abstract**: GeoGPT is an open large language model system built to advance research in the geosciences. To enhance its domain-specific capabilities, we integrated Retrieval Augmented Generation(RAG), which augments model outputs with relevant information retrieved from an external knowledge source. GeoGPT uses RAG to draw from the GeoGPT Library, a specialized corpus curated for geoscientific content, enabling it to generate accurate, context-specific answers. Users can also create personalized knowledge bases by uploading their own publication lists, allowing GeoGPT to retrieve and respond using user-provided materials. To further improve retrieval quality and domain alignment, we fine-tuned both the embedding model and a ranking model that scores retrieved passages by relevance to the query. These enhancements optimize RAG for geoscience applications and significantly improve the system's ability to deliver precise and trustworthy outputs. GeoGPT reflects a strong commitment to open science through its emphasis on collaboration, transparency, and community driven development. As part of this commitment, we have open-sourced two core RAG components-GeoEmbedding and GeoReranker-to support geoscientists, researchers, and professionals worldwide with powerful, accessible AI tools. 

---
# TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation 

**Authors**: Keunwoo Choi, Seungheon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09685)  

**Abstract**: We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In TalkPlayData 2 pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at this https URL. 

---
# Text-to-SQL Oriented to the Process Mining Domain: A PT-EN Dataset for Query Translation 

**Authors**: Bruno Yui Yamate, Thais Rodrigues Neubauer, Marcelo Fantinato, Sarajane Marques Peres  

**Link**: [PDF](https://arxiv.org/pdf/2509.09684)  

**Abstract**: This paper introduces text-2-SQL-4-PM, a bilingual (Portuguese-English) benchmark dataset designed for the text-to-SQL task in the process mining domain. Text-to-SQL conversion facilitates natural language querying of databases, increasing accessibility for users without SQL expertise and productivity for those that are experts. The text-2-SQL-4-PM dataset is customized to address the unique challenges of process mining, including specialized vocabularies and single-table relational structures derived from event logs. The dataset comprises 1,655 natural language utterances, including human-generated paraphrases, 205 SQL statements, and ten qualifiers. Methods include manual curation by experts, professional translations, and a detailed annotation process to enable nuanced analyses of task complexity. Additionally, a baseline study using GPT-3.5 Turbo demonstrates the feasibility and utility of the dataset for text-to-SQL applications. The results show that text-2-SQL-4-PM supports evaluation of text-to-SQL implementations, offering broader applicability for semantic parsing and other natural language processing tasks. 

---
# Forecasting Clicks in Digital Advertising: Multimodal Inputs and Interpretable Outputs 

**Authors**: Briti Gangopadhyay, Zhao Wang, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09683)  

**Abstract**: Forecasting click volume is a key task in digital advertising, influencing both revenue and campaign strategy. Traditional time series models rely solely on numerical data, often overlooking rich contextual information embedded in textual elements, such as keyword updates. We present a multimodal forecasting framework that combines click data with textual logs from real-world ad campaigns and generates human-interpretable explanations alongside numeric predictions. Reinforcement learning is used to improve comprehension of textual information and enhance fusion of modalities. Experiments on a large-scale industry dataset show that our method outperforms baselines in both accuracy and reasoning quality. 

---
# DB3 Team's Solution For Meta KDD Cup' 25 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09681)  

**Abstract**: This paper presents the db3 team's winning solution for the Meta CRAG-MM Challenge 2025 at KDD Cup'25. Addressing the challenge's unique multi-modal, multi-turn question answering benchmark (CRAG-MM), we developed a comprehensive framework that integrates tailored retrieval pipelines for different tasks with a unified LLM-tuning approach for hallucination control. Our solution features (1) domain-specific retrieval pipelines handling image-indexed knowledge graphs, web sources, and multi-turn conversations; and (2) advanced refusal training using SFT, DPO, and RL. The system achieved 2nd place in Task 1, 2nd place in Task 2, and 1st place in Task 3, securing the grand prize for excellence in ego-centric queries through superior handling of first-person perspective challenges. 

---
# AEGIS: An Agent for Extraction and Geographic Identification in Scholarly Proceedings 

**Authors**: Om Vishesh, Harshad Khadilkar, Deepak Akkil  

**Link**: [PDF](https://arxiv.org/pdf/2509.09470)  

**Abstract**: Keeping pace with the rapid growth of academia literature presents a significant challenge for researchers, funding bodies, and academic societies. To address the time-consuming manual effort required for scholarly discovery, we present a novel, fully automated system that transitions from data discovery to direct action. Our pipeline demonstrates how a specialized AI agent, 'Agent-E', can be tasked with identifying papers from specific geographic regions within conference proceedings and then executing a Robotic Process Automation (RPA) to complete a predefined action, such as submitting a nomination form. We validated our system on 586 papers from five different conferences, where it successfully identified every target paper with a recall of 100% and a near perfect accuracy of 99.4%. This demonstration highlights the potential of task-oriented AI agents to not only filter information but also to actively participate in and accelerate the workflows of the academic community. 

---
# Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL 

**Authors**: Hanyi Mao, Quanjia Xiao, Lei Pang, Haixiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09177)  

**Abstract**: We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping directly in the importance-sampling (IS) weight space. We revisit sequence-level RL methods and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs. long responses, distorting the effective objective. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a directional cosine guarantee between the clipped and true updates. FSPO introduces a simple, Gaussian-motivated remedy: we clip the sequence log-IS ratio with a band that applies a KL-corrected drift term and scales as $\sqrt{L}$. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms all baselines across multiple evaluation datasets. 

---
# Generative Engine Optimization: How to Dominate AI Search 

**Authors**: Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2509.08919)  

**Abstract**: The rapid adoption of generative AI-powered search engines like ChatGPT, Perplexity, and Gemini is fundamentally reshaping information retrieval, moving from traditional ranked lists to synthesized, citation-backed answers. This shift challenges established Search Engine Optimization (SEO) practices and necessitates a new paradigm, which we term Generative Engine Optimization (GEO).
This paper presents a comprehensive comparative analysis of AI Search and traditional web search (Google). Through a series of large-scale, controlled experiments across multiple verticals, languages, and query paraphrases, we quantify critical differences in how these systems source information. Our key findings reveal that AI Search exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content, a stark contrast to Google's more balanced mix. We further demonstrate that AI Search services differ significantly from each other in their domain diversity, freshness, cross-language stability, and sensitivity to phrasing.
Based on these empirical results, we formulate a strategic GEO agenda. We provide actionable guidance for practitioners, emphasizing the critical need to: (1) engineer content for machine scannability and justification, (2) dominate earned media to build AI-perceived authority, (3) adopt engine-specific and language-aware strategies, and (4) overcome the inherent "big brand bias" for niche players. Our work provides the foundational empirical analysis and a strategic framework for achieving visibility in the new generative search landscape. 

---
