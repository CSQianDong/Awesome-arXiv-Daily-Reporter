# A Knowledge-Graph Translation Layer for Mission-Aware Multi-Agent Path Planning in Spatiotemporal Dynamics 

**Authors**: Edward Holmberg, Elias Ioup, Mahdi Abdelguerfi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21695)  

**Abstract**: The coordination of autonomous agents in dynamic environments is hampered by the semantic gap between high-level mission objectives and low-level planner inputs. To address this, we introduce a framework centered on a Knowledge Graph (KG) that functions as an intelligent translation layer. The KG's two-plane architecture compiles declarative facts into per-agent, mission-aware ``worldviews" and physics-aware traversal rules, decoupling mission semantics from a domain-agnostic planner. This allows complex, coordinated paths to be modified simply by changing facts in the KG. A case study involving Autonomous Underwater Vehicles (AUVs) in the Gulf of Mexico visually demonstrates the end-to-end process and quantitatively proves that different declarative policies produce distinct, high-performing outcomes. This work establishes the KG not merely as a data repository, but as a powerful, stateful orchestrator for creating adaptive and explainable autonomous systems. 

---
# A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection 

**Authors**: Gaku Morio, Harri Rowlands, Dominik Stammbach, Christopher D. Manning, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2510.21679)  

**Abstract**: Companies spend large amounts of money on public relations campaigns to project a positive brand image. However, sometimes there is a mismatch between what they say and what they do. Oil & gas companies, for example, are accused of "greenwashing" with imagery of climate-friendly initiatives. Understanding the framing, and changes in framing, at scale can help better understand the goals and nature of public relations campaigns. To address this, we introduce a benchmark dataset of expert-annotated video ads obtained from Facebook and YouTube. The dataset provides annotations for 13 framing types for more than 50 companies or advocacy groups across 20 countries. Our dataset is especially designed for the evaluation of vision-language models (VLMs), distinguishing it from past text-only framing datasets. Baseline experiments show some promising results, while leaving room for improvement for future work: GPT-4.1 can detect environmental messages with 79% F1 score, while our best model only achieves 46% F1 score on identifying framing around green innovation. We also identify challenges that VLMs must address, such as implicit framing, handling videos of various lengths, or implicit cultural backgrounds. Our dataset contributes to research in multimodal analysis of strategic communication in the energy sector. 

---
# CMOMgen: Complex Multi-Ontology Alignment via Pattern-Guided In-Context Learning 

**Authors**: Marta Contreiras Silva, Daniel Faria, Catia Pesquita  

**Link**: [PDF](https://arxiv.org/pdf/2510.21656)  

**Abstract**: Constructing comprehensive knowledge graphs requires the use of multiple ontologies in order to fully contextualize data into a domain. Ontology matching finds equivalences between concepts interconnecting ontologies and creating a cohesive semantic layer. While the simple pairwise state of the art is well established, simple equivalence mappings cannot provide full semantic integration of related but disjoint ontologies. Complex multi-ontology matching (CMOM) aligns one source entity to composite logical expressions of multiple target entities, establishing more nuanced equivalences and provenance along the ontological hierarchy.
We present CMOMgen, the first end-to-end CMOM strategy that generates complete and semantically sound mappings, without establishing any restrictions on the number of target ontologies or entities. Retrieval-Augmented Generation selects relevant classes to compose the mapping and filters matching reference mappings to serve as examples, enhancing In-Context Learning. The strategy was evaluated in three biomedical tasks with partial reference alignments. CMOMgen outperforms baselines in class selection, demonstrating the impact of having a dedicated strategy. Our strategy also achieves a minimum of 63% in F1-score, outperforming all baselines and ablated versions in two out of three tasks and placing second in the third. Furthermore, a manual evaluation of non-reference mappings showed that 46% of the mappings achieve the maximum score, further substantiating its ability to construct semantically sound mappings. 

---
# AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite 

**Authors**: Jonathan Bragg, Mike D'Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D. Hwang, Peter Jansen, Varsha Kishore, Bodhisattwa Prasad Majumder, Aakanksha Naik, Sigal Rahamimov, Kyle Richardson, Amanpreet Singh, Harshit Surana, Aryeh Tiktinsky, Rosni Vasu, Guy Wiener, Chloe Anastasiades, Stefan Candra, Jason Dunkelberger, Dan Emery, Rob Evans, Malachi Hamada, Regan Huff, Rodney Kinney, Matt Latzke, Jaron Lochner, Ruben Lozano-Aguilera, Cecile Nguyen, Smita Rao, Amber Tanaka, Brooke Vlahos, Peter Clark, Doug Downey, Yoav Goldberg, Ashish Sabharwal, Daniel S. Weld  

**Link**: [PDF](https://arxiv.org/pdf/2510.21652)  

**Abstract**: AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they (1) fail to provide holistic, product-informed measures of real-world use cases such as science research; (2) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (3) do not account for confounding variables such as model cost and tool access; (4) do not provide standardized interfaces for quick agent prototyping and evaluation; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides the first holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance. 

---
# DeepAgent: A General Reasoning Agent with Scalable Toolsets 

**Authors**: Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21618)  

**Abstract**: Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at this https URL. 

---
# Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine 

**Authors**: Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2510.21614)  

**Abstract**: Recent studies operationalize self-improvement through coding agents that edit their own codebases. They grow a tree of self-modifications through expansion strategies that favor higher software engineering benchmark performance, assuming that this implies more promising subsequent self-modifications. However, we identify a mismatch between the agent's self-improvement potential (metaproductivity) and its coding benchmark performance, namely the Metaproductivity-Performance Mismatch. Inspired by Huxley's concept of clade, we propose a metric ($\mathrm{CMP}$) that aggregates the benchmark performances of the descendants of an agent as an indicator of its potential for self-improvement. We show that, in our self-improving coding agent development setting, access to the true $\mathrm{CMP}$ is sufficient to simulate how the Gödel Machine would behave under certain assumptions. We introduce the Huxley-Gödel Machine (HGM), which, by estimating $\mathrm{CMP}$ and using it as guidance, searches the tree of self-modifications. On SWE-bench Verified and Polyglot, HGM outperforms prior self-improving coding agent development methods while using less wall-clock time. Last but not least, HGM demonstrates strong transfer to other coding datasets and large language models. The agent optimized by HGM on SWE-bench Verified with GPT-5-mini and evaluated on SWE-bench Lite with GPT-5 achieves human-level performance, matching the best officially checked results of human-engineered coding agents. Our code is available at this https URL. 

---
# Learning Neural Control Barrier Functions from Expert Demonstrations using Inverse Constraint Learning 

**Authors**: Yuxuan Yang, Hussein Sibai  

**Link**: [PDF](https://arxiv.org/pdf/2510.21560)  

**Abstract**: Safety is a fundamental requirement for autonomous systems operating in critical domains. Control barrier functions (CBFs) have been used to design safety filters that minimally alter nominal controls for such systems to maintain their safety. Learning neural CBFs has been proposed as a data-driven alternative for their computationally expensive optimization-based synthesis. However, it is often the case that the failure set of states that should be avoided is non-obvious or hard to specify formally, e.g., tailgating in autonomous driving, while a set of expert demonstrations that achieve the task and avoid the failure set is easier to generate. We use ICL to train a constraint function that classifies the states of the system under consideration to safe, i.e., belong to a controlled forward invariant set that is disjoint from the unspecified failure set, and unsafe ones, i.e., belong to the complement of that set. We then use that function to label a new set of simulated trajectories to train our neural CBF. We empirically evaluate our approach in four different environments, demonstrating that it outperforms existing baselines and achieves comparable performance to a neural CBF trained with the same data but annotated with ground-truth safety labels. 

---
# Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts 

**Authors**: Hongwei Zhang, Ji Lu, Shiqing Jiang, Chenxiang Zhu, Li Xie, Chen Zhong, Haoran Chen, Yurui Zhu, Yongsheng Du, Yanqin Gao, Lingjun Huang, Baoli Wang, Fang Tan, Peng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21557)  

**Abstract**: Long-horizon reasoning in LLM-based agents often fails not from generative weakness but from insufficient verification of intermediate reasoning. Co-Sight addresses this challenge by turning reasoning into a falsifiable and auditable process through two complementary mechanisms: Conflict-Aware Meta-Verification (CAMV) and Trustworthy Reasoning with Structured Facts (TRSF). CAMV reformulates verification as conflict identification and targeted falsification, allocating computation only to disagreement hotspots among expert agents rather than to full reasoning chains. This bounds verification cost to the number of inconsistencies and improves efficiency and reliability. TRSF continuously organizes, validates, and synchronizes evidence across agents through a structured facts module. By maintaining verified, traceable, and auditable knowledge, it ensures that all reasoning is grounded in consistent, source-verified information and supports transparent verification throughout the reasoning process. Together, TRSF and CAMV form a closed verification loop, where TRSF supplies structured facts and CAMV selectively falsifies or reinforces them, yielding transparent and trustworthy reasoning. Empirically, Co-Sight achieves state-of-the-art accuracy on GAIA (84.4%) and Humanity's Last Exam (35.5%), and strong results on Chinese-SimpleQA (93.8%). Ablation studies confirm that the synergy between structured factual grounding and conflict-aware verification drives these improvements. Co-Sight thus offers a scalable paradigm for reliable long-horizon reasoning in LLM-based agents. Code is available at this https URL. 

---
# EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law 

**Authors**: Ilija Lichkovski, Alexander Müller, Mariam Ibrahim, Tiwai Mhundwa  

**Link**: [PDF](https://arxiv.org/pdf/2510.21524)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions. We release our code on \href{this https URL}{this URL}. 

---
# Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP 

**Authors**: Yuxin Pan, Zhiguang Cao, Chengyang Gu, Liu Liu, Peilin Zhao, Yize Chen, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21453)  

**Abstract**: Existing neural methods for multi-task vehicle routing problems (VRPs) typically learn unified solvers to handle multiple constraints simultaneously. However, they often underutilize the compositional structure of VRP variants, each derivable from a common set of basis VRP variants. This critical oversight causes unified solvers to miss out the potential benefits of basis solvers, each specialized for a basis VRP variant. To overcome this limitation, we propose a framework that enables unified solvers to perceive the shared-component nature across VRP variants by proactively reusing basis solvers, while mitigating the exponential growth of trained neural solvers. Specifically, we introduce a State-Decomposable MDP (SDMDP) that reformulates VRPs by expressing the state space as the Cartesian product of basis state spaces associated with basis VRP variants. More crucially, this formulation inherently yields the optimal basis policy for each basis VRP variant. Furthermore, a Latent Space-based SDMDP extension is developed by incorporating both the optimal basis policies and a learnable mixture function to enable the policy reuse in the latent space. Under mild assumptions, this extension provably recovers the optimal unified policy of SDMDP through the mixture function that computes the state embedding as a mapping from the basis state embeddings generated by optimal basis policies. For practical implementation, we introduce the Mixture-of-Specialized-Experts Solver (MoSES), which realizes basis policies through specialized Low-Rank Adaptation (LoRA) experts, and implements the mixture function via an adaptive gating mechanism. Extensive experiments conducted across VRP variants showcase the superiority of MoSES over prior methods. 

---
# AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving 

**Authors**: Ankur Sinha, Shobhit Arora, Dhaval Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2510.21436)  

**Abstract**: This study presents AutoOpt-11k, a unique image dataset of over 11,000 handwritten and printed mathematical optimization models corresponding to single-objective, multi-objective, multi-level, and stochastic optimization problems exhibiting various types of complexities such as non-linearity, non-convexity, non-differentiability, discontinuity, and high-dimensionality. The labels consist of the LaTeX representation for all the images and modeling language representation for a subset of images. The dataset is created by 25 experts following ethical data creation guidelines and verified in two-phases to avoid errors. Further, we develop AutoOpt framework, a machine learning based automated approach for solving optimization problems, where the user just needs to provide an image of the formulation and AutoOpt solves it efficiently without any further human intervention. AutoOpt framework consists of three Modules: (i) M1 (Image_to_Text)- a deep learning model performs the Mathematical Expression Recognition (MER) task to generate the LaTeX code corresponding to the optimization formulation in image; (ii) M2 (Text_to_Text)- a small-scale fine-tuned LLM generates the PYOMO script (optimization modeling language) from LaTeX code; (iii) M3 (Optimization)- a Bilevel Optimization based Decomposition (BOBD) method solves the optimization formulation described in the PYOMO script. We use AutoOpt-11k dataset for training and testing of deep learning models employed in AutoOpt. The deep learning model for MER task (M1) outperforms ChatGPT, Gemini and Nougat on BLEU score metric. BOBD method (M3), which is a hybrid approach, yields better results on complex test problems compared to common approaches, like interior-point algorithm and genetic algorithm. 

---
# Advancing Symbolic Integration in Large Language Models: Beyond Conventional Neurosymbolic AI 

**Authors**: Maneeha Rani, Bhupesh Kumar Mishra, Dhavalkumar Thakker  

**Link**: [PDF](https://arxiv.org/pdf/2510.21425)  

**Abstract**: LLMs have demonstrated highly effective learning, human-like response generation,and decision-making capabilities in high-risk sectors. However, these models remain black boxes because they struggle to ensure transparency in responses. The literature has explored numerous approaches to address transparency challenges in LLMs, including Neurosymbolic AI (NeSy AI). NeSy AI approaches were primarily developed for conventional neural networks and are not well-suited to the unique features of LLMs. Consequently, there is a limited systematic understanding of how symbolic AI can be effectively integrated into LLMs. This paper aims to address this gap by first reviewing established NeSy AI methods and then proposing a novel taxonomy of symbolic integration in LLMs, along with a roadmap to merge symbolic techniques with LLMs. The roadmap introduces a new categorisation framework across four dimensions by organising existing literature within these categories. These include symbolic integration across various stages of LLM, coupling mechanisms, architectural paradigms, as well as algorithmic and application-level perspectives. The paper thoroughly identifies current benchmarks, cutting-edge advancements, and critical gaps within the field to propose a roadmap for future research. By highlighting the latest developments and notable gaps in the literature, it offers practical insights for implementing frameworks for symbolic integration into LLMs to enhance transparency. 

---
# Boosting Accuracy and Efficiency of Budget Forcing in LLMs via Reinforcement Learning for Mathematical Reasoning 

**Authors**: Ravindra Aribowo Tarunokusumo, Rafael Fernandes Cunha  

**Link**: [PDF](https://arxiv.org/pdf/2510.21398)  

**Abstract**: Test-time scaling methods have seen a rapid increase in popularity for its computational efficiency and parameter-independent training to improve reasoning performance on Large Language Models. One such method is called budget forcing, a decoding intervention strategy which allocates extra compute budget for thinking and elicits the inherent self-correcting behavior of the model. However, this relies on supervised fine-tuning (SFT) on long-context reasoning traces which causes performance degradation on smaller models due to verbose responses. For this reason, we offer a framework integrating reinforcement learning (RL) to improve token efficiency and boost the performance of a 1.5B model for mathematical reasoning. We demonstrate this using only 1.5K training samples and found that our SFT+RL model performed better on the GSM8K dataset with varying compute budgets. Our main findings showed an overall higher accuracy while significantly reducing its token usage by over 40% compared to the SFT model, revealing how RL can recover the losses due to long-context training and altogether improving performance in mathematical reasoning. 

---
# Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation 

**Authors**: Lufan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21341)  

**Abstract**: Large Language Models (LLMs) often struggle with generating truly innovative ideas, typically defaulting to high-probability, familiar concepts within their training data's "gravity wells." While advanced search-based methods like Tree of Thoughts (ToT) attempt to mitigate this, they are fundamentally limited by their reliance on unprincipled, inconsistent self-evaluation heuristics to guide exploration. To address this gap, we introduce \textbf{Magellan}, a novel framework that reframes creative generation as a principled, guided exploration of an LLM's latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system. For long-range direction, a "semantic compass" vector, formulated via orthogonal projection, steers the search towards relevant novelty. For local, step-by-step decisions, a landscape-aware value function replaces flawed self-evaluation with an explicit reward structure that balances intrinsic coherence, extrinsic novelty, and narrative progress. Extensive experiments demonstrate that Magellan significantly outperforms strong baselines, including ReAct and ToT, in generating scientific ideas with superior plausibility and innovation. Our work shows that for creative discovery, a principled, guided search is more effective than unconstrained agency, paving the way for LLMs to become more capable partners in innovation. 

---
# CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation 

**Authors**: Jinhui Lou, Yan Yang, Zhou Yu, Zhenqi Fu, Weidong Han, Qingming Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21324)  

**Abstract**: Chest X-ray (CXR) plays a pivotal role in clinical diagnosis, and a variety of task-specific and foundation models have been developed for automatic CXR interpretation. However, these models often struggle to adapt to new diagnostic tasks and complex reasoning scenarios. Recently, LLM-based agent models have emerged as a promising paradigm for CXR analysis, enhancing model's capability through tool coordination, multi-step reasoning, and team collaboration, etc. However, existing agents often rely on a single diagnostic pipeline and lack mechanisms for assessing tools' reliability, limiting their adaptability and credibility. To this end, we propose CXRAgent, a director-orchestrated, multi-stage agent for CXR interpretation, where a central director coordinates the following stages: (1) Tool Invocation: The agent strategically orchestrates a set of CXR-analysis tools, with outputs normalized and verified by the Evidence-driven Validator (EDV), which grounds diagnostic outputs with visual evidence to support reliable downstream diagnosis; (2) Diagnostic Planning: Guided by task requirements and intermediate findings, the agent formulates a targeted diagnostic plan. It then assembles an expert team accordingly, defining member roles and coordinating their interactions to enable adaptive and collaborative reasoning; (3) Collaborative Decision-making: The agent integrates insights from the expert team with accumulated contextual memories, synthesizing them into an evidence-backed diagnostic conclusion. Experiments on various CXR interpretation tasks show that CXRAgent delivers strong performance, providing visual evidence and generalizes well to clinical tasks of different complexity. Code and data are valuable at this \href{this https URL}{link}. 

---
# Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning 

**Authors**: Sanghyun Ahn, Wonje Choi, Junyong Lee, Jinwoo Park, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21302)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the automatic generation of executable code for task planning and control in embodied agents such as robots, demonstrating the potential of LLM-based embodied intelligence. However, these LLM-based code-as-policies approaches often suffer from limited environmental grounding, particularly in dynamic or partially observable settings, leading to suboptimal task success rates due to incorrect or incomplete code generation. In this work, we propose a neuro-symbolic embodied task planning framework that incorporates explicit symbolic verification and interactive validation processes during code generation. In the validation phase, the framework generates exploratory code that actively interacts with the environment to acquire missing observations while preserving task-relevant states. This integrated process enhances the grounding of generated code, resulting in improved task reliability and success rates in complex environments. We evaluate our framework on RLBench and in real-world settings across dynamic, partially observable scenarios. Experimental results demonstrate that our framework improves task success rates by 46.2% over Code-as-Policies baselines and attains over 86.8% executability of task-relevant actions, thereby enhancing the reliability of task planning in dynamic environments. 

---
# Understanding AI Trustworthiness: A Scoping Review of AIES & FAccT Articles 

**Authors**: Siddharth Mehrotra, Jin Huang, Xuelong Fu, Roel Dobbe, Clara I. Sánchez, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2510.21293)  

**Abstract**: Background: Trustworthy AI serves as a foundational pillar for two major AI ethics conferences: AIES and FAccT. However, current research often adopts techno-centric approaches, focusing primarily on technical attributes such as reliability, robustness, and fairness, while overlooking the sociotechnical dimensions critical to understanding AI trustworthiness in real-world contexts.
Objectives: This scoping review aims to examine how the AIES and FAccT communities conceptualize, measure, and validate AI trustworthiness, identifying major gaps and opportunities for advancing a holistic understanding of trustworthy AI systems.
Methods: We conduct a scoping review of AIES and FAccT conference proceedings to date, systematically analyzing how trustworthiness is defined, operationalized, and applied across different research domains. Our analysis focuses on conceptualization approaches, measurement methods, verification and validation techniques, application areas, and underlying values.
Results: While significant progress has been made in defining technical attributes such as transparency, accountability, and robustness, our findings reveal critical gaps. Current research often predominantly emphasizes technical precision at the expense of social and ethical considerations. The sociotechnical nature of AI systems remains less explored and trustworthiness emerges as a contested concept shaped by those with the power to define it.
Conclusions: An interdisciplinary approach combining technical rigor with social, cultural, and institutional considerations is essential for advancing trustworthy AI. We propose actionable measures for the AI ethics community to adopt holistic frameworks that genuinely address the complex interplay between AI systems and society, ultimately promoting responsible technological development that benefits all stakeholders. 

---
# When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails 

**Authors**: Yingzhi Mao, Chunkang Zhang, Junxiang Wang, Xinyan Guan, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.21285)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs. 

---
# Investigating Scale Independent UCT Exploration Factor Strategies 

**Authors**: Robin Schmöcker, Christoph Schnell, Alexander Dockhorn  

**Link**: [PDF](https://arxiv.org/pdf/2510.21275)  

**Abstract**: The Upper Confidence Bounds For Trees (UCT) algorithm is not agnostic to the reward scale of the game it is applied to. For zero-sum games with the sparse rewards of $\{-1,0,1\}$ at the end of the game, this is not a problem, but many games often feature dense rewards with hand-picked reward scales, causing a node's Q-value to span different magnitudes across different games. In this paper, we evaluate various strategies for adaptively choosing the UCT exploration constant $\lambda$, called $\lambda$-strategies, that are agnostic to the game's reward scale. These $\lambda$-strategies include those proposed in the literature as well as five new strategies. Given our experimental results, we recommend using one of our newly suggested $\lambda$-strategies, which is to choose $\lambda$ as $2 \cdot \sigma$ where $\sigma$ is the empirical standard deviation of all state-action pairs' Q-values of the search tree. This method outperforms existing $\lambda$-strategies across a wide range of tasks both in terms of a single parameter value and the peak performances obtained by optimizing all available parameters. 

---
# Out-of-Distribution Detection for Safety Assurance of AI and Autonomous Systems 

**Authors**: Victoria J. Hodge, Colin Paterson, Ibrahim Habli  

**Link**: [PDF](https://arxiv.org/pdf/2510.21254)  

**Abstract**: The operational capabilities and application domains of AI-enabled autonomous systems have expanded significantly in recent years due to advances in robotics and machine learning (ML). Demonstrating the safety of autonomous systems rigorously is critical for their responsible adoption but it is challenging as it requires robust methodologies that can handle novel and uncertain situations throughout the system lifecycle, including detecting out-of-distribution (OoD) data. Thus, OOD detection is receiving increased attention from the research, development and safety engineering communities. This comprehensive review analyses OOD detection techniques within the context of safety assurance for autonomous systems, in particular in safety-critical domains. We begin by defining the relevant concepts, investigating what causes OOD and exploring the factors which make the safety assurance of autonomous systems and OOD detection challenging. Our review identifies a range of techniques which can be used throughout the ML development lifecycle and we suggest areas within the lifecycle in which they may be used to support safety assurance arguments. We discuss a number of caveats that system and safety engineers must be aware of when integrating OOD detection into system lifecycles. We conclude by outlining the challenges and future work necessary for the safe development and operation of autonomous systems across a range of domains and applications. 

---
# OutboundEval: A Dual-Dimensional Benchmark for Expert-Level Intelligent Outbound Evaluation of Xbench's Professional-Aligned Series 

**Authors**: Pengyu Xu, Shijia Li, Ao Sun, Feng Zhang, Yahan Li, Bo Wu, Zhanyu Ma, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Rui Wang, Yang Liu, Xiaobo Hu, Fan Yang, Jia Zheng, Guanghua Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21244)  

**Abstract**: We propose OutboundEval, a comprehensive benchmark for evaluating large language models (LLMs) in expert-level intelligent outbound calling scenarios. Unlike existing methods that suffer from three key limitations - insufficient dataset diversity and category coverage, unrealistic user simulation, and inaccurate evaluation metrics - OutboundEval addresses these issues through a structured framework. First, we design a benchmark spanning six major business domains and 30 representative sub-scenarios, each with scenario-specific process decomposition, weighted scoring, and domain-adaptive metrics. Second, we develop a large-model-driven User Simulator that generates diverse, persona-rich virtual users with realistic behaviors, emotional variability, and communication styles, providing a controlled yet authentic testing environment. Third, we introduce a dynamic evaluation method that adapts to task variations, integrating automated and human-in-the-loop assessment to measure task execution accuracy, professional knowledge application, adaptability, and user experience quality. Experiments on 12 state-of-the-art LLMs reveal distinct trade-offs between expert-level task completion and interaction fluency, offering practical insights for building reliable, human-like outbound AI systems. OutboundEval establishes a practical, extensible, and domain-oriented standard for benchmarking LLMs in professional applications. 

---
# Shylock: Causal Discovery in Multivariate Time Series based on Hybrid Constraints 

**Authors**: Shuo Li, Keqin Xu, Jie Liu, Dan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.21181)  

**Abstract**: Causal relationship discovery has been drawing increasing attention due to its prevalent application. Existing methods rely on human experience, statistical methods, or graphical criteria methods which are error-prone, stuck at the idealized assumption, and rely on a huge amount of data. And there is also a serious data gap in accessing Multivariate time series(MTS) in many areas, adding difficulty in finding their causal relationship. Existing methods are easy to be over-fitting on them. To fill the gap we mentioned above, in this paper, we propose Shylock, a novel method that can work well in both few-shot and normal MTS to find the causal relationship. Shylock can reduce the number of parameters exponentially by using group dilated convolution and a sharing kernel, but still learn a better representation of variables with time delay. By combing the global constraint and the local constraint, Shylock achieves information sharing among networks to help improve the accuracy. To evaluate the performance of Shylock, we also design a data generation method to generate MTS with time delay. We evaluate it on commonly used benchmarks and generated datasets. Extensive experiments show that Shylock outperforms two existing state-of-art methods on both few-shot and normal MTS. We also developed Tcausal, a library for easy use and deployed it on the EarthDataMiner platform 

---
# Memory-Free Continual Learning with Null Space Adaptation for Zero-Shot Vision-Language Models 

**Authors**: Yujin Jo, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.21175)  

**Abstract**: Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated remarkable zero-shot generalization, enabling deployment in a wide range of real-world tasks without additional task-specific training. However, in real deployment scenarios with evolving environments or emerging classes, these models inevitably face distributional shifts and novel tasks. In such contexts, static zero-shot capabilities are insufficient, and there is a growing need for continual learning methods that allow models to adapt over time while avoiding catastrophic forgetting. We introduce NuSA-CL (Null Space Adaptation for Continual Learning), a lightweight memory-free continual learning framework designed to address this challenge. NuSA-CL employs low-rank adaptation and constrains task-specific weight updates to lie within an approximate null space of the model's current parameters. This strategy minimizes interference with previously acquired knowledge, effectively preserving the zero-shot capabilities of the original model. Unlike methods relying on replay buffers or costly distillation, NuSA-CL imposes minimal computational and memory overhead, making it practical for deployment in resource-constrained, real-world continual learning environments. Experiments show that our framework not only effectively preserves zero-shot transfer capabilities but also achieves highly competitive performance on continual learning benchmarks. These results position NuSA-CL as a practical and scalable solution for continually evolving zero-shot VLMs in real-world applications. 

---
# String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation 

**Authors**: Kou Misaki, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2510.21150)  

**Abstract**: We introduce String Seed of Thought (SSoT), a novel prompting method for LLMs that improves Probabilistic Instruction Following (PIF). We define PIF as a task requiring an LLM to select its answer from a predefined set of options, each associated with a specific probability, such that the empirical distribution of the generated answers aligns with the target distribution when prompted multiple times. While LLMs excel at tasks with single, deterministic answers, they often fail at PIF, exhibiting biases problematic for applications requiring non-deterministic behaviors, such as human-behavior simulation, content diversification, and multiplayer games. It also harms the diversity of generated responses, a crucial factor in test-time scaling, by causing the outputs to collapse into a limited set of answers. To address this, we propose SSoT, a simple prompting method that instructs an LLM to first output a random string to generate sufficient entropy. SSoT also instructs the LLM to extract randomness by manipulating this string to derive a final answer, thereby preserving diversity while adhering to specific constraints. We demonstrate that SSoT significantly improves the PIF performance of LLMs, approaching the ideal performance of a pseudo-random number generator. Furthermore, our experiments on NoveltyBench show SSoT's benefits extend beyond closed-set tasks to open-ended tasks by enhancing response diversity. 

---
# How to Auto-optimize Prompts for Domain Tasks? Adaptive Prompting and Reasoning through Evolutionary Domain Knowledge Adaptation 

**Authors**: Yang Zhao, Pu Wang, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21148)  

**Abstract**: Designing optimal prompts and reasoning processes for large language models (LLMs) on domain-specific tasks is both necessary and challenging in real-world applications. Determining how to integrate domain knowledge, enhance reasoning efficiency, and even provide domain experts with refined knowledge integration hints are particularly crucial yet unresolved tasks. In this research, we propose Evolutionary Graph Optimization for Prompting (EGO-Prompt), an automated framework to designing better prompts, efficient reasoning processes and providing enhanced causal-informed process. EGO-Prompt begins with a general prompt and fault-tolerant initial Semantic Causal Graph (SCG) descriptions, constructed by human experts, which is then automatically refined and optimized to guide LLM reasoning. Recognizing that expert-defined SCGs may be partial or imperfect and that their optimal integration varies across LLMs, EGO-Prompt integrates a novel causal-guided textual gradient process in two steps: first, generating nearly deterministic reasoning guidance from the SCG for each instance, and second, adapting the LLM to effectively utilize the guidance alongside the original input. The iterative optimization algorithm further refines both the SCG and the reasoning mechanism using textual gradients with ground-truth. We tested the framework on real-world public health, transportation and human behavior tasks. EGO-Prompt achieves 7.32%-12.61% higher F1 than cutting-edge methods, and allows small models to reach the performence of larger models at under 20% of the original cost. It also outputs a refined, domain-specific SCG that improves interpretability. 

---
# NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge 

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21144)  

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict. 

---
# PanicToCalm: A Proactive Counseling Agent for Panic Attacks 

**Authors**: Jihyun Lee, Yejin Min, San Kim, Yejin Jeon, SungJun Yang, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.21143)  

**Abstract**: Panic attacks are acute episodes of fear and distress, in which timely, appropriate intervention can significantly help individuals regain stability. However, suitable datasets for training such models remain scarce due to ethical and logistical issues. To address this, we introduce PACE, which is a dataset that includes high-distress episodes constructed from first-person narratives, and structured around the principles of Psychological First Aid (PFA). Using this data, we train PACER, a counseling model designed to provide both empathetic and directive support, which is optimized through supervised learning and simulated preference alignment. To assess its effectiveness, we propose PanicEval, a multi-dimensional framework covering general counseling quality and crisis-specific strategies. Experimental results show that PACER outperforms strong baselines in both counselor-side metrics and client affect improvement. Human evaluations further confirm its practical value, with PACER consistently preferred over general, CBT-based, and GPT-4-powered models in panic scenarios (Code is available at this https URL ). 

---
# DAO-AI: Evaluating Collective Decision-Making through Agentic AI in Decentralized Governance 

**Authors**: Chunghyun Han, Alfio Gliozzo, Junkyu Lee, Agostino Capponi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21117)  

**Abstract**: This paper presents a first empirical study of agentic AI as autonomous decision-makers in decentralized governance. Using more than 3K proposals from major protocols, we build an agentic AI voter that interprets proposal contexts, retrieves historical deliberation data, and independently determines its voting position. The agent operates within a realistic financial simulation environment grounded in verifiable blockchain data, implemented through a modular composable program (MCP) workflow that defines data flow and tool usage via Agentics framework. We evaluate how closely the agent's decisions align with the human and token-weighted outcomes, uncovering strong alignments measured by carefully designed evaluation metrics. Our findings demonstrate that agentic AI can augment collective decision-making by producing interpretable, auditable, and empirically grounded signals in realistic DAO governance settings. The study contributes to the design of explainable and economically rigorous AI agents for decentralized financial systems. 

---
# Confounding Robust Deep Reinforcement Learning: A Causal Approach 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2510.21110)  

**Abstract**: A key task in Artificial Intelligence is learning effective policies for controlling agents in unknown environments to optimize performance measures. Off-policy learning methods, like Q-learning, allow learners to make optimal decisions based on past experiences. This paper studies off-policy learning from biased data in complex and high-dimensional domains where \emph{unobserved confounding} cannot be ruled out a priori. Building on the well-celebrated Deep Q-Network (DQN), we propose a novel deep reinforcement learning algorithm robust to confounding biases in observed data. Specifically, our algorithm attempts to find a safe policy for the worst-case environment compatible with the observations. We apply our method to twelve confounded Atari games, and find that it consistently dominates the standard DQN in all games where the observed input to the behavioral and target policies mismatch and unobserved confounders exist. 

---
# MedAlign: A Synergistic Framework of Multimodal Preference Optimization and Federated Meta-Cognitive Reasoning 

**Authors**: Siyong Chen, Jinbo Wen, Jiawen Kang, Tenghui Huang, Xumin Huang, Yuanjia Su, Hudan Pan, Zishao Zhong, Dusit Niyato, Shengli Xie, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.21093)  

**Abstract**: Recently, large models have shown significant potential for smart healthcare. However, the deployment of Large Vision-Language Models (LVLMs) for clinical services is currently hindered by three critical challenges: a tendency to hallucinate answers not grounded in visual evidence, the inefficiency of fixed-depth reasoning, and the difficulty of multi-institutional collaboration. To address these challenges, in this paper, we develop MedAlign, a novel framework to ensure visually accurate LVLM responses for Medical Visual Question Answering (Med-VQA). Specifically, we first propose a multimodal Direct Preference Optimization (mDPO) objective to explicitly align preference learning with visual context. We then design a Retrieval-Aware Mixture-of-Experts (RA-MoE) architecture that utilizes image and text similarity to route queries to a specialized and context-augmented LVLM (i.e., an expert), thereby mitigating hallucinations in LVLMs. To achieve adaptive reasoning and facilitate multi-institutional collaboration, we propose a federated governance mechanism, where the selected expert, fine-tuned on clinical datasets based on mDPO, locally performs iterative Chain-of-Thought (CoT) reasoning via the local meta-cognitive uncertainty estimator. Extensive experiments on three representative Med-VQA datasets demonstrate that MedAlign achieves state-of-the-art performance, outperforming strong retrieval-augmented baselines by up to $11.85\%$ in F1-score, and simultaneously reducing the average reasoning length by $51.60\%$ compared with fixed-depth CoT approaches. 

---
# From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL 

**Authors**: Ali Khosravi Kazazi, Zhenlong Li, M. Naser Lessani, Guido Cervone  

**Link**: [PDF](https://arxiv.org/pdf/2510.21045)  

**Abstract**: The complexity of Structured Query Language (SQL) and the specialized nature of geospatial functions in tools like PostGIS present significant barriers to non-experts seeking to analyze spatial data. While Large Language Models (LLMs) offer promise for translating natural language into SQL (Text-to-SQL), single-agent approaches often struggle with the semantic and syntactic complexities of spatial queries. To address this, we propose a multi-agent framework designed to accurately translate natural language questions into spatial SQL queries. The framework integrates several innovative components, including a knowledge base with programmatic schema profiling and semantic enrichment, embeddings for context retrieval, and a collaborative multi-agent pipeline as its core. This pipeline comprises specialized agents for entity extraction, metadata retrieval, query logic formulation, SQL generation, and a review agent that performs programmatic and semantic validation of the generated SQL to ensure correctness (self-verification). We evaluate our system using both the non-spatial KaggleDBQA benchmark and a new, comprehensive SpatialQueryQA benchmark that includes diverse geometry types, predicates, and three levels of query complexity. On KaggleDBQA, the system achieved an overall accuracy of 81.2% (221 out of 272 questions) after the review agent's review and corrections. For spatial queries, the system achieved an overall accuracy of 87.7% (79 out of 90 questions), compared with 76.7% without the review agent. Beyond accuracy, results also show that in some instances the system generates queries that are more semantically aligned with user intent than those in the benchmarks. This work makes spatial analysis more accessible, and provides a robust, generalizable foundation for spatial Text-to-SQL systems, advancing the development of autonomous GIS. 

---
# Epistemic Deference to AI 

**Authors**: Benjamin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2510.21043)  

**Abstract**: When should we defer to AI outputs over human expert judgment? Drawing on recent work in social epistemology, I motivate the idea that some AI systems qualify as Artificial Epistemic Authorities (AEAs) due to their demonstrated reliability and epistemic superiority. I then introduce AI Preemptionism, the view that AEA outputs should replace rather than supplement a user's independent epistemic reasons. I show that classic objections to preemptionism - such as uncritical deference, epistemic entrenchment, and unhinging epistemic bases - apply in amplified form to AEAs, given their opacity, self-reinforcing authority, and lack of epistemic failure markers. Against this, I develop a more promising alternative: a total evidence view of AI deference. According to this view, AEA outputs should function as contributory reasons rather than outright replacements for a user's independent epistemic considerations. This approach has three key advantages: (i) it mitigates expertise atrophy by keeping human users engaged, (ii) it provides an epistemic case for meaningful human oversight and control, and (iii) it explains the justified mistrust of AI when reliability conditions are unmet. While demanding in practice, this account offers a principled way to determine when AI deference is justified, particularly in high-stakes contexts requiring rigorous reliability. 

---
# Customizing Open Source LLMs for Quantitative Medication Attribute Extraction across Heterogeneous EHR Systems 

**Authors**: Zhe Fei, Mehmet Yigit Turali, Shreyas Rajesh, Xinyang Dai, Huyen Pham, Pavan Holur, Yuhui Zhu, Larissa Mooney, Yih-Ing Hser, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.21027)  

**Abstract**: Harmonizing medication data across Electronic Health Record (EHR) systems is a persistent barrier to monitoring medications for opioid use disorder (MOUD). In heterogeneous EHR systems, key prescription attributes are scattered across differently formatted fields and freetext notes. We present a practical framework that customizes open source large language models (LLMs), including Llama, Qwen, Gemma, and MedGemma, to extract a unified set of MOUD prescription attributes (prescription date, drug name, duration, total quantity, daily quantity, and refills) from heterogeneous, site specific data and compute a standardized metric of medication coverage, \emph{MOUD days}, per patient. Our pipeline processes records directly in a fixed JSON schema, followed by lightweight normalization and cross-field consistency checks. We evaluate the system on prescription level EHR data from five clinics in a national OUD study (25{,}605 records from 1{,}257 patients), using a previously annotated benchmark of 10{,}369 records (776 patients) as the ground truth. Performance is reported as coverage (share of records with a valid, matchable output) and record-level exact-match accuracy. Larger models perform best overall: Qwen2.5-32B achieves \textbf{93.4\%} coverage with \textbf{93.0\%} exact-match accuracy across clinics, and MedGemma-27B attains \textbf{93.1\%}/\textbf{92.2\%}. A brief error review highlights three common issues and fixes: imputing missing dosage fields using within-drug norms, handling monthly/weekly injectables (e.g., Vivitrol) by setting duration from the documented schedule, and adding unit checks to prevent mass units (e.g., ``250 g'') from being misread as daily counts. By removing brittle, site-specific ETL and supporting local, privacy-preserving deployment, this approach enables consistent cross-site analyses of MOUD exposure, adherence, and retention in real-world settings. 

---
# Fuzzy numbers revisited: operations on extensional fuzzy numbers 

**Authors**: Krzysztof Siminski  

**Link**: [PDF](https://arxiv.org/pdf/2510.20861)  

**Abstract**: Fuzzy numbers are commonly represented with fuzzy sets. Their objective is to better represent imprecise data. However, operations on fuzzy numbers are not as straightforward as maths on crisp numbers. Commonly, the Zadeh's extension rule is applied to elaborate a result. This can produce two problems: (1) high computational complexity and (2) for some fuzzy sets and some operations the results is not a fuzzy set with the same features (eg. multiplication of two triangular fuzzy sets does not produce a triangular fuzzy set). One more problem is the fuzzy spread -- fuzziness of the result increases with the number of operations. These facts can severely limit the application field of fuzzy numbers. In this paper we would like to revisite this problem with a different kind of fuzzy numbers -- extensional fuzzy numbers. The paper defines operations on extensional fuzzy numbers and relational operators (=, >, >=, <, <=) for them. The proposed approach is illustrated with several applicational examples. The C++ implementation is available from a public GitHub repository. 

---
# Cultural Alien Sampler: Open-ended art generation balancing originality and coherence 

**Authors**: Alejandro H. Artiles, Hiromu Yakura, Levin Brinkmann, Mar Canet Sola, Hassan Abu Alhaija, Ignacio Serna, Nasim Rahaman, Bernhard Schölkopf, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20849)  

**Abstract**: In open-ended domains like art, autonomous agents must generate ideas that are both original and internally coherent, yet current Large Language Models (LLMs) either default to familiar cultural patterns or sacrifice coherence when pushed toward novelty. We address this by introducing the Cultural Alien Sampler (CAS), a concept-selection method that explicitly separates compositional fit from cultural typicality. CAS uses two GPT-2 models fine-tuned on WikiArt concepts: a Concept Coherence Model that scores whether concepts plausibly co-occur within artworks, and a Cultural Context Model that estimates how typical those combinations are within individual artists' bodies of work. CAS targets combinations that are high in coherence and low in typicality, yielding ideas that maintain internal consistency while deviating from learned conventions and embedded cultural context. In a human evaluation (N = 100), our approach outperforms random selection and GPT-4o baselines and achieves performance comparable to human art students in both perceived originality and harmony. Additionally, a quantitative study shows that our method produces more diverse outputs and explores a broader conceptual space than its GPT-4o counterpart, demonstrating that artificial cultural alienness can unlock creative potential in autonomous agents. 

---
# Sketch2BIM: A Multi-Agent Human-AI Collaborative Pipeline to Convert Hand-Drawn Floor Plans to 3D BIM 

**Authors**: Abir Khan Ratul, Sanjay Acharjee, Somin Park, Md Nazmus Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2510.20838)  

**Abstract**: This study introduces a human-in-the-loop pipeline that converts unscaled, hand-drawn floor plan sketches into semantically consistent 3D BIM models. The workflow leverages multimodal large language models (MLLMs) within a multi-agent framework, combining perceptual extraction, human feedback, schema validation, and automated BIM scripting. Initially, sketches are iteratively refined into a structured JSON layout of walls, doors, and windows. Later, these layouts are transformed into executable scripts that generate 3D BIM models. Experiments on ten diverse floor plans demonstrate strong convergence: openings (doors, windows) are captured with high reliability in the initial pass, while wall detection begins around 83% and achieves near-perfect alignment after a few feedback iterations. Across all categories, precision, recall, and F1 scores remain above 0.83, and geometric errors (RMSE, MAE) progressively decrease to zero through feedback corrections. This study demonstrates how MLLM-driven multi-agent reasoning can make BIM creation accessible to both experts and non-experts using only freehand sketches. 

---
# On Thin Ice: Towards Explainable Conservation Monitoring via Attribution and Perturbations 

**Authors**: Jiayi Zhou, Günel Aghakishiyeva, Saagar Arya, Julian Dale, James David Poling, Holly R. Houliston, Jamie N. Womble, Gregory D. Larsen, David W. Johnston, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2510.21689)  

**Abstract**: Computer vision can accelerate ecological research and conservation monitoring, yet adoption in ecology lags in part because of a lack of trust in black-box neural-network-based models. We seek to address this challenge by applying post-hoc explanations to provide evidence for predictions and document limitations that are important to field deployment. Using aerial imagery from Glacier Bay National Park, we train a Faster R-CNN to detect pinnipeds (harbor seals) and generate explanations via gradient-based class activation mapping (HiResCAM, LayerCAM), local interpretable model-agnostic explanations (LIME), and perturbation-based explanations. We assess explanations along three axes relevant to field use: (i) localization fidelity: whether high-attribution regions coincide with the animal rather than background context; (ii) faithfulness: whether deletion/insertion tests produce changes in detector confidence; and (iii) diagnostic utility: whether explanations reveal systematic failure modes. Explanations concentrate on seal torsos and contours rather than surrounding ice/rock, and removal of the seals reduces detection confidence, providing model-evidence for true positives. The analysis also uncovers recurrent error sources, including confusion between seals and black ice and rocks. We translate these findings into actionable next steps for model development, including more targeted data curation and augmentation. By pairing object detection with post-hoc explainability, we can move beyond "black-box" predictions toward auditable, decision-supporting tools for conservation monitoring. 

---
# Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging 

**Authors**: Ying Xue, Jiaxi Jiang, Rayan Armani, Dominik Hollidt, Yi-Chi Liao, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21654)  

**Abstract**: Tracking human full-body motion using sparse wearable inertial measurement units (IMUs) overcomes the limitations of occlusion and instrumentation of the environment inherent in vision-based approaches. However, purely IMU-based tracking compromises translation estimates and accurate relative positioning between individuals, as inertial cues are inherently self-referential and provide no direct spatial reference for others. In this paper, we present a novel approach for robustly estimating body poses and global translation for multiple individuals by leveraging the distances between sparse wearable sensors - both on each individual and across multiple individuals. Our method Group Inertial Poser estimates these absolute distances between pairs of sensors from ultra-wideband ranging (UWB) and fuses them with inertial observations as input into structured state-space models to integrate temporal motion patterns for precise 3D pose estimation. Our novel two-step optimization further leverages the estimated distances for accurately tracking people's global trajectories through the world. We also introduce GIP-DB, the first IMU+UWB dataset for two-person tracking, which comprises 200 minutes of motion recordings from 14 participants. In our evaluation, Group Inertial Poser outperforms previous state-of-the-art methods in accuracy and robustness across synthetic and real-world data, showing the promise of IMU+UWB-based multi-human motion capture in the wild. Code, models, dataset: this https URL 

---
# A Dynamic Knowledge Distillation Method Based on the Gompertz Curve 

**Authors**: Han Yang, Guangjun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21649)  

**Abstract**: This paper introduces a novel dynamic knowledge distillation framework, Gompertz-CNN, which integrates the Gompertz growth model into the training process to address the limitations of traditional knowledge distillation. Conventional methods often fail to capture the evolving cognitive capacity of student models, leading to suboptimal knowledge transfer. To overcome this, we propose a stage-aware distillation strategy that dynamically adjusts the weight of distillation loss based on the Gompertz curve, reflecting the student's learning progression: slow initial growth, rapid mid-phase improvement, and late-stage saturation. Our framework incorporates Wasserstein distance to measure feature-level discrepancies and gradient matching to align backward propagation behaviors between teacher and student models. These components are unified under a multi-loss objective, where the Gompertz curve modulates the influence of distillation losses over time. Extensive experiments on CIFAR-10 and CIFAR-100 using various teacher-student architectures (e.g., ResNet50 and MobileNet_v2) demonstrate that Gompertz-CNN consistently outperforms traditional distillation methods, achieving up to 8% and 4% accuracy gains on CIFAR-10 and CIFAR-100, respectively. 

---
# DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection 

**Authors**: Tala Aljaafari, Varun Kanade, Philip Torr, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2510.21638)  

**Abstract**: Deploying reinforcement learning (RL) in safety-critical settings is constrained by brittleness under distribution shift. We study out-of-distribution (OOD) detection for RL time series and introduce DEEDEE, a two-statistic detector that revisits representation-heavy pipelines with a minimal alternative. DEEDEE uses only an episodewise mean and an RBF kernel similarity to a training summary, capturing complementary global and local deviations. Despite its simplicity, DEEDEE matches or surpasses contemporary detectors across standard RL OOD suites, delivering a 600-fold reduction in compute (FLOPs / wall-time) and an average 5% absolute accuracy gain over strong baselines. Conceptually, our results indicate that diverse anomaly types often imprint on RL trajectories through a small set of low-order statistics, suggesting a compact foundation for OOD detection in complex environments. 

---
# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations 

**Authors**: Faisal Hamman, Pasan Dissanayake, Yanjun Fu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.21631)  

**Abstract**: Knowledge distillation is a promising approach to transfer capabilities from complex teacher models to smaller, resource-efficient student models that can be deployed easily, particularly in task-aware scenarios. However, existing methods of task-aware distillation typically require substantial quantities of data which may be unavailable or expensive to obtain in many practical scenarios. In this paper, we address this challenge by introducing a novel strategy called Counterfactual-explanation-infused Distillation CoD for few-shot task-aware knowledge distillation by systematically infusing counterfactual explanations. Counterfactual explanations (CFEs) refer to inputs that can flip the output prediction of the teacher model with minimum perturbation. Our strategy CoD leverages these CFEs to precisely map the teacher's decision boundary with significantly fewer samples. We provide theoretical guarantees for motivating the role of CFEs in distillation, from both statistical and geometric perspectives. We mathematically show that CFEs can improve parameter estimation by providing more informative examples near the teacher's decision boundary. We also derive geometric insights on how CFEs effectively act as knowledge probes, helping the students mimic the teacher's decision boundaries more effectively than standard data. We perform experiments across various datasets and LLMs to show that CoD outperforms standard distillation approaches in few-shot regimes (as low as 8-512 samples). Notably, CoD only uses half of the original samples used by the baselines, paired with their corresponding CFEs and still improves performance. 

---
# The Universal Landscape of Human Reasoning 

**Authors**: Qiguang Chen, Jinhao Liu, Libo Qin, Yimeng Zhang, Yihao Liang, Shangxu Ren, Chengyu Luan, Dengyun Peng, Hanjing Li, Jiannan Guan, Zheng Yan, Jiaqi Wang, Mengkang Hu, Yantao Du, Zhi Chen, Xie Chen, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2510.21623)  

**Abstract**: Understanding how information is dynamically accumulated and transformed in human reasoning has long challenged cognitive psychology, philosophy, and artificial intelligence. Existing accounts, from classical logic to probabilistic models, illuminate aspects of output or individual modelling, but do not offer a unified, quantitative description of general human reasoning dynamics. To solve this, we introduce Information Flow Tracking (IF-Track), that uses large language models (LLMs) as probabilistic encoder to quantify information entropy and gain at each reasoning step. Through fine-grained analyses across diverse tasks, our method is the first successfully models the universal landscape of human reasoning behaviors within a single metric space. We show that IF-Track captures essential reasoning features, identifies systematic error patterns, and characterizes individual differences. Applied to discussion of advanced psychological theory, we first reconcile single- versus dual-process theories in IF-Track and discover the alignment of artificial and human cognition and how LLMs reshaping human reasoning process. This approach establishes a quantitative bridge between theory and measurement, offering mechanistic insights into the architecture of reasoning. 

---
# Generative Correlation Manifolds: Generating Synthetic Data with Preserved Higher-Order Correlations 

**Authors**: Jens E. d'Hondt, Wieger R. Punter, Odysseas Papapetrou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21610)  

**Abstract**: The increasing need for data privacy and the demand for robust machine learning models have fueled the development of synthetic data generation techniques. However, current methods often succeed in replicating simple summary statistics but fail to preserve both the pairwise and higher-order correlation structure of the data that define the complex, multi-variable interactions inherent in real-world systems. This limitation can lead to synthetic data that is superficially realistic but fails when used for sophisticated modeling tasks. In this white paper, we introduce Generative Correlation Manifolds (GCM), a computationally efficient method for generating synthetic data. The technique uses Cholesky decomposition of a target correlation matrix to produce datasets that, by mathematical proof, preserve the entire correlation structure -- from simple pairwise relationships to higher-order interactions -- of the source dataset. We argue that this method provides a new approach to synthetic data generation with potential applications in privacy-preserving data sharing, robust model training, and simulation. 

---
# Sample By Step, Optimize By Chunk: Chunk-Level GRPO For Text-to-Image Generation 

**Authors**: Yifu Luo, Penghui Du, Bo Li, Sinan Du, Tiantian Zhang, Yongzhe Chang, Kai Wu, Kun Gai, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21583)  

**Abstract**: Group Relative Policy Optimization (GRPO) has shown strong potential for flow-matching-based text-to-image (T2I) generation, but it faces two key limitations: inaccurate advantage attribution, and the neglect of temporal dynamics of generation. In this work, we argue that shifting the optimization paradigm from the step level to the chunk level can effectively alleviate these issues. Building on this idea, we propose Chunk-GRPO, the first chunk-level GRPO-based approach for T2I generation. The insight is to group consecutive steps into coherent 'chunk's that capture the intrinsic temporal dynamics of flow matching, and to optimize policies at the chunk level. In addition, we introduce an optional weighted sampling strategy to further enhance performance. Extensive experiments show that ChunkGRPO achieves superior results in both preference alignment and image quality, highlighting the promise of chunk-level optimization for GRPO-based methods. 

---
# From Polyester Girlfriends to Blind Mice: Creating the First Pragmatics Understanding Benchmarks for Slovene 

**Authors**: Mojca Brglez, Špela Vintar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21575)  

**Abstract**: Large language models are demonstrating increasing capabilities, excelling at benchmarks once considered very difficult. As their capabilities grow, there is a need for more challenging evaluations that go beyond surface-level linguistic competence. Namely, language competence involves not only syntax and semantics but also pragmatics, i.e., understanding situational meaning as shaped by context as well as linguistic and cultural norms. To contribute to this line of research, we introduce SloPragEval and SloPragMega, the first pragmatics understanding benchmarks for Slovene that contain altogether 405 multiple-choice questions. We discuss the difficulties of translation, describe the campaign to establish a human baseline, and report pilot evaluations with LLMs. Our results indicate that current models have greatly improved in understanding nuanced language but may still fail to infer implied speaker meaning in non-literal utterances, especially those that are culture-specific. We also observe a significant gap between proprietary and open-source models. Finally, we argue that benchmarks targeting nuanced language understanding and knowledge of the target culture must be designed with care, preferably constructed from native data, and validated with human responses. 

---
# Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos 

**Authors**: Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21571)  

**Abstract**: This paper presents a novel approach for pretraining robotic manipulation Vision-Language-Action (VLA) models using a large corpus of unscripted real-life video recordings of human hand activities. Treating human hand as dexterous robot end-effector, we show that "in-the-wild" egocentric human videos without any annotations can be transformed into data formats fully aligned with existing robotic V-L-A training data in terms of task granularity and labels. This is achieved by the development of a fully-automated holistic human activity analysis approach for arbitrary human hand videos. This approach can generate atomic-level hand activity segments and their language descriptions, each accompanied with framewise 3D hand motion and camera motion. We process a large volume of egocentric videos and create a hand-VLA training dataset containing 1M episodes and 26M frames. This training data covers a wide range of objects and concepts, dexterous manipulation tasks, and environment variations in real life, vastly exceeding the coverage of existing robot data. We design a dexterous hand VLA model architecture and pretrain the model on this dataset. The model exhibits strong zero-shot capabilities on completely unseen real-world observations. Additionally, fine-tuning it on a small amount of real robot action data significantly improves task success rates and generalization to novel objects in real robotic experiments. We also demonstrate the appealing scaling behavior of the model's task performance with respect to pretraining data scale. We believe this work lays a solid foundation for scalable VLA pretraining, advancing robots toward truly generalizable embodied intelligence. 

---
# Human and AI Trust: Trust Attitude Measurement Instrument 

**Authors**: Retno Larasati  

**Link**: [PDF](https://arxiv.org/pdf/2510.21535)  

**Abstract**: With the current progress of Artificial Intelligence (AI) technology and its increasingly broader applications, trust is seen as a required criterion for AI usage, acceptance, and deployment. A robust measurement instrument is essential to correctly evaluate trust from a human-centered perspective. This paper describes the development and validation process of a trust measure instrument, which follows psychometric principles, and consists of a 16-items trust scale. The instrument was built explicitly for research in human-AI interaction to measure trust attitudes towards AI systems from layperson (non-expert) perspective. The use-case we used to develop the scale was in the context of AI medical support systems (specifically cancer/health prediction). The scale development (Measurement Item Development) and validation (Measurement Item Evaluation) involved six research stages: item development, item evaluation, survey administration, test of dimensionality, test of reliability, and test of validity. The results of the six-stages evaluation show that the proposed trust measurement instrument is empirically reliable and valid for systematically measuring and comparing non-experts' trust in AI Medical Support Systems. 

---
# GranViT: A Fine-Grained Vision Model With Autoregressive Perception For MLLMs 

**Authors**: Guanghao Zheng, Bowen Shi, Mingxing Xu, Ruoyu Sun, Peisen Zhao, Zhibo Zhang, Wenrui Dai, Junni Zou, Hongkai Xiong, Xiaopeng Zhang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.21501)  

**Abstract**: Vision encoders are indispensable for allowing impressive performance of Multi-modal Large Language Models (MLLMs) in vision language tasks such as visual question answering and reasoning. However, existing vision encoders focus on global image representations but overlook fine-grained regional analysis. They are limited in fine grained perception due to the scarcity of fine grained annotated data and the lack of a fine grained pre-training paradigm. In this paper, we propose GranViT, a novel Vision Transformer that integrates fine-grained feature extraction with semantic alignment to Large Language Models (LLMs) via region level autoregressive training. We first construct Gran-29M, a dataset comprising 2million natural and OCR images paired with over 180 million high-quality region-level annotations, to enable large scale fine grained pretraining. Consequently, we develop a pretraining-adaptation framework along with a self distillation mechanism to train fine-grained GranViT on Gran-29M. We sufficiently exploit the fine-grained annotations from Gran-29M to resort to bounding-box-to-caption regression to enhance localized visual representation of the vision encoder in the pretraining and caption-to-bounding-box regression to improve vision feature utilization and localization for LLM in the adaptation. We further incorporate a self distillation mechanism that imposes explicit localization constraints on the vision encoder to strengthen its regional reasoning capability. Extensive experiments show that GranViT surpasses existing vision encoders and attains strong transferability to varying LLMs. Remarkably, it achieves state-of-the-art results on fine-grained recognition, multimodal VQA, and OCR understanding. 

---
# Enhancing Social Robots through Resilient AI 

**Authors**: Domenico Palmisano, Giuseppe Palestra, Berardina Nadja De Carolis  

**Link**: [PDF](https://arxiv.org/pdf/2510.21469)  

**Abstract**: As artificial intelligence continues to advance and becomes more integrated into sensitive areas like healthcare, education, and everyday life, it's crucial for these systems to be both resilient and robust. This paper shows how resilience is a fundamental characteristic of social robots, which, through it, ensure trust in the robot itself-an essential element especially when operating in contexts with elderly people, who often have low trust in these systems. Resilience is therefore the ability to operate under adverse or stressful conditions, even when degraded or weakened, while maintaining essential operational capabilities. 

---
# PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis 

**Authors**: Yu Yang, Zhilu Zhang, Xiang Zhang, Yihan Zeng, Hui Li, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21447)  

**Abstract**: Interactive world models that simulate object dynamics are crucial for robotics, VR, and AR. However, it remains a significant challenge to learn physics-consistent dynamics models from limited real-world video data, especially for deformable objects with spatially-varying physical properties. To overcome the challenge of data scarcity, we propose PhysWorld, a novel framework that utilizes a simulator to synthesize physically plausible and diverse demonstrations to learn efficient world models. Specifically, we first construct a physics-consistent digital twin within MPM simulator via constitutive model selection and global-to-local optimization of physical properties. Subsequently, we apply part-aware perturbations to the physical properties and generate various motion patterns for the digital twin, synthesizing extensive and diverse demonstrations. Finally, using these demonstrations, we train a lightweight GNN-based world model that is embedded with physical properties. The real video can be used to further refine the physical properties. PhysWorld achieves accurate and fast future predictions for various deformable objects, and also generalizes well to novel interactions. Experiments show that PhysWorld has competitive performance while enabling inference speeds 47 times faster than the recent state-of-the-art method, i.e., PhysTwin. 

---
# REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring 

**Authors**: Thanh Cong Ho, Farah Kharrat, Abderrazek Abid, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2510.21445)  

**Abstract**: With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities. 

---
# Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification 

**Authors**: Mohammad Amin Zadenoori, Vincenzo De Martino, Jacek Dabrowski, Xavier Franch, Alessio Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2510.21443)  

**Abstract**: [Context and motivation] Large language models (LLMs) show notable results in natural language processing (NLP) tasks for requirements engineering (RE). However, their use is compromised by high computational cost, data sharing risks, and dependence on external services. In contrast, small language models (SLMs) offer a lightweight, locally deployable alternative. [Question/problem] It remains unclear how well SLMs perform compared to LLMs in RE tasks in terms of accuracy. [Results] Our preliminary study compares eight models, including three LLMs and five SLMs, on requirements classification tasks using the PROMISE, PROMISE Reclass, and SecReq datasets. Our results show that although LLMs achieve an average F1 score of 2% higher than SLMs, this difference is not statistically significant. SLMs almost reach LLMs performance across all datasets and even outperform them in recall on the PROMISE Reclass dataset, despite being up to 300 times smaller. We also found that dataset characteristics play a more significant role in performance than model size. [Contribution] Our study contributes with evidence that SLMs are a valid alternative to LLMs for requirements classification, offering advantages in privacy, cost, and local deployability. 

---
# Vision Language Models for Dynamic Human Activity Recognition in Healthcare Settings 

**Authors**: Abderrazek Abid, Thanh-Cong Ho, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2510.21424)  

**Abstract**: As generative AI continues to evolve, Vision Language Models (VLMs) have emerged as promising tools in various healthcare applications. One area that remains relatively underexplored is their use in human activity recognition (HAR) for remote health monitoring. VLMs offer notable strengths, including greater flexibility and the ability to overcome some of the constraints of traditional deep learning models. However, a key challenge in applying VLMs to HAR lies in the difficulty of evaluating their dynamic and often non-deterministic outputs. To address this gap, we introduce a descriptive caption data set and propose comprehensive evaluation methods to evaluate VLMs in HAR. Through comparative experiments with state-of-the-art deep learning models, our findings demonstrate that VLMs achieve comparable performance and, in some cases, even surpass conventional approaches in terms of accuracy. This work contributes a strong benchmark and opens new possibilities for the integration of VLMs into intelligent healthcare systems. 

---
# DreamerV3-XP: Optimizing exploration through uncertainty estimation 

**Authors**: Lukas Bierling, Davide Pasero, Jan-Henrik Bertrand, Kiki Van Gerwen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21418)  

**Abstract**: We introduce DreamerV3-XP, an extension of DreamerV3 that improves exploration and learning efficiency. This includes (i) a prioritized replay buffer, scoring trajectories by return, reconstruction loss, and value error and (ii) an intrinsic reward based on disagreement over predicted environment rewards from an ensemble of world models. DreamerV3-XP is evaluated on a subset of Atari100k and DeepMind Control Visual Benchmark tasks, confirming the original DreamerV3 results and showing that our extensions lead to faster learning and lower dynamics model loss, particularly in sparse-reward settings. 

---
# Large Language Models as Model Organisms for Human Associative Learning 

**Authors**: Camila Kolling, Vy Ai Vo, Mariya Toneva  

**Link**: [PDF](https://arxiv.org/pdf/2510.21408)  

**Abstract**: Associative learning--forming links between co-occurring items--is fundamental to human cognition, reshaping internal representations in complex ways. Testing hypotheses on how representational changes occur in biological systems is challenging, but large language models (LLMs) offer a scalable alternative. Building on LLMs' in-context learning, we adapt a cognitive neuroscience associative learning paradigm and investigate how representations evolve across six models. Our initial findings reveal a non-monotonic pattern consistent with the Non-Monotonic Plasticity Hypothesis, with moderately similar items differentiating after learning. Leveraging the controllability of LLMs, we further show that this differentiation is modulated by the overlap of associated items with the broader vocabulary--a factor we term vocabulary interference, capturing how new associations compete with prior knowledge. We find that higher vocabulary interference amplifies differentiation, suggesting that representational change is influenced by both item similarity and global competition. Our findings position LLMs not only as powerful tools for studying representational dynamics in human-like learning systems, but also as accessible and general computational models for generating new hypotheses about the principles underlying memory reorganization in the brain. 

---
# REvolution: An Evolutionary Framework for RTL Generation driven by Large Language Models 

**Authors**: Kyungjun Min, Kyumin Cho, Junhwan Jang, Seokhyeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21407)  

**Abstract**: Large Language Models (LLMs) are used for Register-Transfer Level (RTL) code generation, but they face two main challenges: functional correctness and Power, Performance, and Area (PPA) optimization. Iterative, feedback-based methods partially address these, but they are limited to local search, hindering the discovery of a global optimum. This paper introduces REvolution, a framework that combines Evolutionary Computation (EC) with LLMs for automatic RTL generation and optimization. REvolution evolves a population of candidates in parallel, each defined by a design strategy, RTL implementation, and evaluation feedback. The framework includes a dual-population algorithm that divides candidates into Fail and Success groups for bug fixing and PPA optimization, respectively. An adaptive mechanism further improves search efficiency by dynamically adjusting the selection probability of each prompt strategy according to its success rate. Experiments on the VerilogEval and RTLLM benchmarks show that REvolution increased the initial pass rate of various LLMs by up to 24.0 percentage points. The DeepSeek-V3 model achieved a final pass rate of 95.5\%, comparable to state-of-the-art results, without the need for separate training or domain-specific tools. Additionally, the generated RTL designs showed significant PPA improvements over reference designs. This work introduces a new RTL design approach by combining LLMs' generative capabilities with EC's broad search power, overcoming the local-search limitations of previous methods. 

---
# Assessing the Real-World Utility of Explainable AI for Arousal Diagnostics: An Application-Grounded User Study 

**Authors**: Stefan Kraft, Andreas Theissler, Vera Wienhausen-Wilke, Gjergji Kasneci, Hendrik Lensch  

**Link**: [PDF](https://arxiv.org/pdf/2510.21389)  

**Abstract**: Artificial intelligence (AI) systems increasingly match or surpass human experts in biomedical signal interpretation. However, their effective integration into clinical practice requires more than high predictive accuracy. Clinicians must discern \textit{when} and \textit{why} to trust algorithmic recommendations. This work presents an application-grounded user study with eight professional sleep medicine practitioners, who score nocturnal arousal events in polysomnographic data under three conditions: (i) manual scoring, (ii) black-box (BB) AI assistance, and (iii) transparent white-box (WB) AI assistance. Assistance is provided either from the \textit{start} of scoring or as a post-hoc quality-control (\textit{QC}) review. We systematically evaluate how the type and timing of assistance influence event-level and clinically most relevant count-based performance, time requirements, and user experience. When evaluated against the clinical standard used to train the AI, both AI and human-AI teams significantly outperform unaided experts, with collaboration also reducing inter-rater variability. Notably, transparent AI assistance applied as a targeted QC step yields median event-level performance improvements of approximately 30\% over black-box assistance, and QC timing further enhances count-based outcomes. While WB and QC approaches increase the time required for scoring, start-time assistance is faster and preferred by most participants. Participants overwhelmingly favor transparency, with seven out of eight expressing willingness to adopt the system with minor or no modifications. In summary, strategically timed transparent AI assistance effectively balances accuracy and clinical efficiency, providing a promising pathway toward trustworthy AI integration and user acceptance in clinical workflows. 

---
# Compressing Quaternion Convolutional Neural Networks for Audio Classification 

**Authors**: Arshdeep Singh, Vinayak Abrol, Mark D. Plumbley  

**Link**: [PDF](https://arxiv.org/pdf/2510.21388)  

**Abstract**: Conventional Convolutional Neural Networks (CNNs) in the real domain have been widely used for audio classification. However, their convolution operations process multi-channel inputs independently, limiting the ability to capture correlations among channels. This can lead to suboptimal feature learning, particularly for complex audio patterns such as multi-channel spectrogram representations. Quaternion Convolutional Neural Networks (QCNNs) address this limitation by employing quaternion algebra to jointly capture inter-channel dependencies, enabling more compact models with fewer learnable parameters while better exploiting the multi-dimensional nature of audio signals. However, QCNNs exhibit higher computational complexity due to the overhead of quaternion operations, resulting in increased inference latency and reduced efficiency compared to conventional CNNs, posing challenges for deployment on resource-constrained platforms. To address this challenge, this study explores knowledge distillation (KD) and pruning, to reduce the computational complexity of QCNNs while maintaining performance. Our experiments on audio classification reveal that pruning QCNNs achieves similar or superior performance compared to KD while requiring less computational effort. Compared to conventional CNNs and Transformer-based architectures, pruned QCNNs achieve competitive performance with a reduced learnable parameter count and computational complexity. On the AudioSet dataset, pruned QCNNs reduce computational cost by 50\% and parameter count by 80\%, while maintaining performance comparable to the conventional CNNs. Furthermore, pruned QCNNs generalize well across multiple audio classification benchmarks, including GTZAN for music genre recognition, ESC-50 for environmental sound classification and RAVDESS for speech emotion recognition. 

---
# HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences 

**Authors**: Zain Ul Abideen Tariq, Mahmood Al-Zubaidi, Uzair Shah, Marco Agus, Mowafa Househ  

**Link**: [PDF](https://arxiv.org/pdf/2510.21370)  

**Abstract**: HIKMA Semi-Autonomous Conference is the first experiment in reimagining scholarly communication through an end-to-end integration of artificial intelligence into the academic publishing and presentation pipeline. This paper presents the design, implementation, and evaluation of the HIKMA framework, which includes AI dataset curation, AI-based manuscript generation, AI-assisted peer review, AI-driven revision, AI conference presentation, and AI archival dissemination. By combining language models, structured research workflows, and domain safeguards, HIKMA shows how AI can support - not replace traditional scholarly practices while maintaining intellectual property protection, transparency, and integrity. The conference functions as a testbed and proof of concept, providing insights into the opportunities and challenges of AI-enabled scholarship. It also examines questions about AI authorship, accountability, and the role of human-AI collaboration in research. 

---
# Patient-specific AI for generation of 3D dosimetry imaging from two 2D-planar measurements 

**Authors**: Alejandro Lopez-Montes, Robert Seifert, Astrid Delker, Guido Boening, Jiahui Wang, Christoph Clement, Ali Afshar-Oromieh, Axel Rominger, Kuangyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21362)  

**Abstract**: In this work we explored the use of patient specific reinforced learning to generate 3D activity maps from two 2D planar images (anterior and posterior). The solution of this problem remains unachievable using conventional methodologies and is of particular interest for dosimetry in nuclear medicine where approaches for post-therapy distribution of radiopharmaceuticals such as 177Lu-PSMA are typically done via either expensive and long 3D SPECT acquisitions or fast, yet only 2D, planar scintigraphy. Being able to generate 3D activity maps from planar scintigraphy opens the gate for new dosimetry applications removing the need for SPECT and facilitating multi-time point dosimetry studies. Our solution comprises the generation of a patient specific dataset with possible 3D uptake maps of the radiopharmaceuticals withing the anatomy of the individual followed by an AI approach (we explored both the use of 3DUnet and diffusion models) able to generate 3D activity maps from 2D planar images. We have validated our method both in simulation and real planar acquisitions. We observed enhanced results using patient specific reinforcement learning (~20% reduction on MAE and ~5% increase in SSIM) and better organ delineation and patient anatomy especially when combining diffusion models with patient specific training yielding a SSIM=0.89 compared to the ground truth for simulations and 0.73 when compared to a SPECT acquisition performed half an hour after the planar. We believe that our methodology can set a change of paradigm for nuclear medicine dosimetry allowing for 3D quantification using only planar scintigraphy without the need of expensive and time-consuming SPECT leveraging the pre-therapy information of the patients. 

---
# Gaze-VLM:Bridging Gaze and VLMs through Attention Regularization for Egocentric Understanding 

**Authors**: Anupam Pani, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21356)  

**Abstract**: Eye gaze offers valuable cues about attention, short-term intent, and future actions, making it a powerful signal for modeling egocentric behavior. In this work, we propose a gaze-regularized framework that enhances VLMs for two key egocentric understanding tasks: fine-grained future event prediction and current activity understanding. Unlike prior approaches that rely solely on visual inputs or use gaze as an auxiliary input signal , our method uses gaze only during training. We introduce a gaze-regularized attention mechanism that aligns model focus with human visual gaze. This design is flexible and modular, allowing it to generalize across multiple VLM architectures that utilize attention. Experimental results show that our approach improves semantic prediction scores by up to 11 for future event prediction and around 7 for current activity understanding, compared to the corresponding baseline models trained without gaze regularization. These results highlight the value of gaze-guided training in improving the accuracy and robustness of egocentric VLMs. Overall, this work establishes a foundation for using human gaze to enhance the predictive capabilities of VLMs in real-world scenarios like assistive robots and human-machine collaboration. Code and additional information is available at: this https URL 

---
# CT-CLIP: A Multi-modal Fusion Framework for Robust Apple Leaf Disease Recognition in Complex Environments 

**Authors**: Lemin Liu, Fangchao Hu, Honghua Jiang, Yaru Chen, Limin Liu, Yongliang Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21346)  

**Abstract**: In complex orchard environments, the phenotypic heterogeneity of different apple leaf diseases, characterized by significant variation among lesions, poses a challenge to traditional multi-scale feature fusion methods. These methods only integrate multi-layer features extracted by convolutional neural networks (CNNs) and fail to adequately account for the relationships between local and global features. Therefore, this study proposes a multi-branch recognition framework named CNN-Transformer-CLIP (CT-CLIP). The framework synergistically employs a CNN to extract local lesion detail features and a Vision Transformer to capture global structural relationships. An Adaptive Feature Fusion Module (AFFM) then dynamically fuses these features, achieving optimal coupling of local and global information and effectively addressing the diversity in lesion morphology and distribution. Additionally, to mitigate interference from complex backgrounds and significantly enhance recognition accuracy under few-shot conditions, this study proposes a multimodal image-text learning approach. By leveraging pre-trained CLIP weights, it achieves deep alignment between visual features and disease semantic descriptions. Experimental results show that CT-CLIP achieves accuracies of 97.38% and 96.12% on a publicly available apple disease and a self-built dataset, outperforming several baseline methods. The proposed CT-CLIP demonstrates strong capabilities in recognizing agricultural diseases, significantly enhances identification accuracy under complex environmental conditions, provides an innovative and practical solution for automated disease recognition in agricultural applications. 

---
# $α$-LoRA: Effective Fine-Tuning via Base Model Rescaling 

**Authors**: Aymane El Firdoussi, El Mahdi Chayti, Mohamed El Amine Seddik, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21345)  

**Abstract**: Fine-tuning has proven to be highly effective in adapting pre-trained models to perform better on new desired tasks with minimal data samples. Among the most widely used approaches are reparameterization methods, which update a target module by augmenting its frozen weight matrix with an additional trainable weight matrix. The most prominent example is Low Rank Adaption (LoRA), which gained significant attention in recent years. In this paper, we introduce a new class of reparameterization methods for transfer learning, designed to enhance the generalization ability of fine-tuned models. We establish the effectiveness of our approach in a high-dimensional binary classification setting using tools from Random Matrix Theory, and further validate our theoretical findings through more realistic experiments, such as fine-tuning LLMs. 

---
# World-POI: Global Point-of-Interest Data Enriched from Foursquare and OpenStreetMap as Tabular and Graph Data 

**Authors**: Hossein Amiri, Mohammad Hashemi, Andreas Züfle  

**Link**: [PDF](https://arxiv.org/pdf/2510.21342)  

**Abstract**: Recently, Foursquare released a global dataset with more than 100 million points of interest (POIs), each representing a real-world business on its platform. However, many entries lack complete metadata such as addresses or categories, and some correspond to non-existent or fictional locations. In contrast, OpenStreetMap (OSM) offers a rich, user-contributed POI dataset with detailed and frequently updated metadata, though it does not formally verify whether a POI represents an actual business. In this data paper, we present a methodology that integrates the strengths of both datasets: Foursquare as a comprehensive baseline of commercial POIs and OSM as a source of enriched metadata. The combined dataset totals approximately 1 TB. While this full version is not publicly released, we provide filtered releases with adjustable thresholds that reduce storage needs and make the data practical to download and use across domains. We also provide step-by-step instructions to reproduce the full 631 GB build. Record linkage is achieved by computing name similarity scores and spatial distances between Foursquare and OSM POIs. These measures identify and retain high-confidence matches that correspond to real businesses in Foursquare, have representations in OSM, and show strong name similarity. Finally, we use this filtered dataset to construct a graph-based representation of POIs enriched with attributes from both sources, enabling advanced spatial analyses and a range of downstream applications. 

---
# CausalRec: A CausalBoost Attention Model for Sequential Recommendation 

**Authors**: Yunbo Hou, Tianle Yang, Ruijie Li, Li He, Liang Wang, Weiping Li, Bo Zheng, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.21333)  

**Abstract**: Recent advances in correlation-based sequential recommendation systems have demonstrated substantial success. Specifically, the attention-based model outperforms other RNN-based and Markov chains-based models by capturing both short- and long-term dependencies more effectively. However, solely focusing on item co-occurrences overlooks the underlying motivations behind user behaviors, leading to spurious correlations and potentially inaccurate recommendations. To address this limitation, we present a novel framework that integrates causal attention for sequential recommendation, CausalRec. It incorporates a causal discovery block and a CausalBooster. The causal discovery block learns the causal graph in user behavior sequences, and we provide a theory to guarantee the identifiability of the learned causal graph. The CausalBooster utilizes the discovered causal graph to refine the attention mechanism, prioritizing behaviors with causal significance. Experimental evaluations on real-world datasets indicate that CausalRec outperforms several state-of-the-art methods, with average improvements of 7.21% in Hit Rate (HR) and 8.65% in Normalized Discounted Cumulative Gain (NDCG). To the best of our knowledge, this is the first model to incorporate causality through the attention mechanism in sequential recommendation, demonstrating the value of causality in generating more accurate and reliable recommendations. 

---
# Weak-to-Strong Generalization under Distribution Shifts 

**Authors**: Myeongho Jeon, Jan Sobotka, Suhwan Choi, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2510.21332)  

**Abstract**: As future superhuman models become increasingly complex, accurately supervising their behavior may exceed human capabilities. Recent works have demonstrated that in such scenarios, weak models can effectively supervise strong models, a phenomenon known as weak-to-strong generalization. However, we find that naive weak-to-strong generalization fails under distribution shifts, often leading to worse performance of the strong model than its weak supervisors. To address this, we propose RAVEN, a robust weak-to-strong generalization framework that dynamically learns the optimal combinations of weak models in addition to parameters of the strong model. We demonstrate the effectiveness of RAVEN on image classification, text classification, and preference alignment tasks. RAVEN outperforms alternative baselines by over 30% on out-of-distribution tasks while matching or surpassing existing methods on in-distribution tasks. Moreover, our results show that RAVEN assigns higher weights to more accurate weak models, demonstrating its ability to automatically identify trustworthy supervision. 

---
# TripTide: A Benchmark for Adaptive Travel Planning under Disruptions 

**Authors**: Priyanshu Karmakar, Soumyabrata Chaudhuri, Shubhojit Mallick, Manish Gupta, Abhik Jana, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21329)  

**Abstract**: Recent efforts like TripCraft and TravelPlanner have advanced the use of Large Language Models ( LLMs) for personalized, constraint aware travel itinerary generation. Yet, real travel often faces disruptions. To address this, we present TripTide, the first benchmark evaluating LLM's ability to revise itineraries under realistic disruptions. TripTide models key dimensions such as disruption severity and traveler tolerance, enabling nuanced assessment of LLM adaptability to events like flight cancellations, weather closures, or overbooked attractions. We conduct a threefold evaluation. First, we introduce automatic metrics including Preservation of Intent (how well the revised plan maintains feasibility and goals), Responsiveness (promptness and appropriateness of disruption handling), and Adaptability (semantic, spatial, and sequential divergence between original and revised plans). Second, we apply an LLM-as-a-judge approach to automatically assess revision quality. Third, we perform manual expert evaluation to verify whether revisions preserve semantic, spatial, sequential, and responsive aspects. Our experiments show that LLMs maintain strong sequential consistency and semantic stability, while spatial deviations are larger for shorter trips but decrease with longer ones, indicating that extended plans encourage better geographic coherence. However, disruption-handling ability declines as plan length increases, highlighting limits in LLM robustness. TripTide establishes a benchmark for evaluating adaptability, personalization, and resilience in LLM-based travel planning under real-world uncertainty. 

---
# Seemingly Redundant Modules Enhance Robust Odor Learning in Fruit Flies 

**Authors**: Haiyang Li, Liao Yu, Qiang Yu, Yunliang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21315)  

**Abstract**: Biological circuits have evolved to incorporate multiple modules that perform similar functions. In the fly olfactory circuit, both lateral inhibition (LI) and neuronal spike frequency adaptation (SFA) are thought to enhance pattern separation for odor learning. However, it remains unclear whether these mechanisms play redundant or distinct roles in this process. In this study, we present a computational model of the fly olfactory circuit to investigate odor discrimination under varying noise conditions that simulate complex environments. Our results show that LI primarily enhances odor discrimination in low- and medium-noise scenarios, but this benefit diminishes and may reverse under higher-noise conditions. In contrast, SFA consistently improves discrimination across all noise levels. LI is preferentially engaged in low- and medium-noise environments, whereas SFA dominates in high-noise settings. When combined, these two sparsification mechanisms enable optimal discrimination performance. This work demonstrates that seemingly redundant modules in biological circuits can, in fact, be essential for achieving optimal learning in complex contexts. 

---
# A Convergence Analysis of Adaptive Optimizers under Floating-point Quantization 

**Authors**: Xuan Tang, Jichu Li, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21314)  

**Abstract**: The rapid scaling of large language models (LLMs) has made low-precision training essential for reducing memory, improving efficiency, and enabling larger models and datasets. Existing convergence theories for adaptive optimizers, however, assume all components are exact and neglect hardware-aware quantization, leaving open the question of why low-precision training remains effective. We introduce the first theoretical framework for analyzing the convergence of adaptive optimizers, including Adam and Muon, under floating-point quantization of gradients, weights, and optimizer states (e.g., moment estimates). Within this framework, we derive convergence rates on smooth non-convex objectives under standard stochastic gradient assumptions, explicitly characterizing how quantization errors from different components affect convergence. We show that both algorithms retain rates close to their full-precision counterparts provided mantissa length scales only logarithmically with the number of iterations. Our analysis further reveals that Adam is highly sensitive to weights and second-moment quantization due to its reliance on $\beta_2 \to 1$, while Muon requires weaker error control and is thus potentially more robust. These results narrow the gap between empirical success and theoretical understanding of low-precision training methods. Numerical experiments on synthetic and real-world data corroborate our theory. 

---
# Efficient semantic uncertainty quantification in language models via diversity-steered sampling 

**Authors**: Ji Won Park, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2510.21310)  

**Abstract**: Accurately estimating semantic aleatoric and epistemic uncertainties in large language models (LLMs) is particularly challenging in free-form question answering (QA), where obtaining stable estimates often requires many expensive generations. We introduce a diversity-steered sampler that discourages semantically redundant outputs during decoding, covers both autoregressive and masked diffusion paradigms, and yields substantial sample-efficiency gains. The key idea is to inject a continuous semantic-similarity penalty into the model's proposal distribution using a natural language inference (NLI) model lightly finetuned on partial prefixes or intermediate diffusion states. We debias downstream uncertainty estimates with importance reweighting and shrink their variance with control variates. Across four QA benchmarks, our method matches or surpasses baselines while covering more semantic clusters with the same number of samples. Being modular and requiring no gradient access to the base LLM, the framework promises to serve as a drop-in enhancement for uncertainty estimation in risk-sensitive model deployments. 

---
# WhaleVAD-BPN: Improving Baleen Whale Call Detection with Boundary Proposal Networks and Post-processing Optimisation 

**Authors**: Christiaan M. Geldenhuys, Günther Tonitz, Thomas R. Niesler  

**Link**: [PDF](https://arxiv.org/pdf/2510.21280)  

**Abstract**: While recent sound event detection (SED) systems can identify baleen whale calls in marine audio, challenges related to false positive and minority-class detection persist. We propose the boundary proposal network (BPN), which extends an existing lightweight SED system. The BPN is inspired by work in image object detection and aims to reduce the number of false positive detections. It achieves this by using intermediate latent representations computed within the backbone classification model to gate the final output. When added to an existing SED system, the BPN achieves a 16.8 % absolute increase in precision, as well as 21.3 % and 9.4 % improvements in the F1-score for minority-class d-calls and bp-calls, respectively. We further consider two approaches to the selection of post-processing hyperparameters: a forward-search and a backward-search. By separately optimising event-level and frame-level hyperparameters, these two approaches lead to considerable performance improvements over parameters selected using empirical methods. The complete WhaleVAD-BPN system achieves a cross-validated development F1-score of 0.475, which is a 9.8 % absolute improvement over the baseline. 

---
# Pctx: Tokenizing Personalized Context for Generative Recommendation 

**Authors**: Qiyong Zhong, Jiajie Su, Yunshan Ma, Julian McAuley, Yupeng Hou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21276)  

**Abstract**: Generative recommendation (GR) models tokenize each action into a few discrete tokens (called semantic IDs) and autoregressively generate the next tokens as predictions, showing advantages such as memory efficiency, scalability, and the potential to unify retrieval and ranking. Despite these benefits, existing tokenization methods are static and non-personalized. They typically derive semantic IDs solely from item features, assuming a universal item similarity that overlooks user-specific perspectives. However, under the autoregressive paradigm, semantic IDs with the same prefixes always receive similar probabilities, so a single fixed mapping implicitly enforces a universal item similarity standard across all users. In practice, the same item may be interpreted differently depending on user intentions and preferences. To address this issue, we propose a personalized context-aware tokenizer that incorporates a user's historical interactions when generating semantic IDs. This design allows the same item to be tokenized into different semantic IDs under different user contexts, enabling GR models to capture multiple interpretive standards and produce more personalized predictions. Experiments on three public datasets demonstrate up to 11.44% improvement in NDCG@10 over non-personalized action tokenization baselines. Our code is available at this https URL. 

---
# Sparser Block-Sparse Attention via Token Permutation 

**Authors**: Xinghao Wang, Pengyu Wang, Dong Zhang, Chenkun Tan, Shaojun Zhou, Zhaoxiang Liu, Shiguo Lian, Fangxu Liu, Kai Song, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21270)  

**Abstract**: Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at this https URL 

---
# Correlation Dimension of Auto-Regressive Large Language Models 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2510.21258)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in natural language generation, yet they continue to display puzzling behaviors -- such as repetition and incoherence -- even when exhibiting low perplexity. This highlights a key limitation of conventional evaluation metrics, which emphasize local prediction accuracy while overlooking long-range structural complexity. We introduce correlation dimension, a fractal-geometric measure of self-similarity, to quantify the epistemological complexity of text as perceived by a language model. This measure captures the hierarchical recurrence structure of language, bridging local and global properties in a unified framework. Through extensive experiments, we show that correlation dimension (1) reveals three distinct phases during pretraining, (2) reflects context-dependent complexity, (3) indicates a model's tendency toward hallucination, and (4) reliably detects multiple forms of degeneration in generated text. The method is computationally efficient, robust to model quantization (down to 4-bit precision), broadly applicable across autoregressive architectures (e.g., Transformer and Mamba), and provides fresh insight into the generative dynamics of LLMs. 

---
# Physics-Informed Neural Networks for MIMO Beam Map and Environment Reconstruction 

**Authors**: Wangqian Chen, Junting Chen, Shuguang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.21238)  

**Abstract**: As communication networks evolve towards greater complexity (e.g., 6G and beyond), a deep understanding of the wireless environment becomes increasingly crucial. When explicit knowledge of the environment is unavailable, geometry-aware feature extraction from channel state information (CSI) emerges as a pivotal methodology to bridge physical-layer measurements with network intelligence. This paper proposes to explore the received signal strength (RSS) data, without explicit 3D environment knowledge, to jointly construct the radio beam map and environmental geometry for a multiple-input multiple-output (MIMO) system. Unlike existing methods that only learn blockage structures, we propose an oriented virtual obstacle model that captures the geometric features of both blockage and reflection. Reflective zones are formulated to identify relevant reflected paths according to the geometry relation of the environment. We derive an analytical expression for the reflective zone and further analyze its geometric characteristics to develop a reformulation that is more compatible with deep learning representations. A physics-informed deep learning framework that incorporates the reflective-zone-based geometry model is proposed to learn the blockage, reflection, and scattering components, along with the beam pattern, which leverages physics prior knowledge to enhance network transferability. Numerical experiments demonstrate that, in addition to reconstructing the blockage and reflection geometry, the proposed model can construct a more accurate MIMO beam map with a 32%-48% accuracy improvement. 

---
# Securing AI Agent Execution 

**Authors**: Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21236)  

**Abstract**: Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a practical foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security. 

---
# PLAN: Proactive Low-Rank Allocation for Continual Learning 

**Authors**: Xiequn Wang, Zhan Zhuang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21188)  

**Abstract**: Continual learning (CL) requires models to continuously adapt to new tasks without forgetting past knowledge. In this work, we propose \underline{P}roactive \underline{L}ow-rank \underline{A}llocatio\underline{N} (PLAN), a framework that extends Low-Rank Adaptation (LoRA) to enable efficient and interference-aware fine-tuning of large pre-trained models in CL settings. PLAN proactively manages the allocation of task-specific subspaces by introducing orthogonal basis vectors for each task and optimizing them through a perturbation-based strategy that minimizes conflicts with previously learned parameters. Furthermore, PLAN incorporates a novel selection mechanism that identifies and assigns basis vectors with minimal sensitivity to interference, reducing the risk of degrading past knowledge while maintaining efficient adaptation to new tasks. Empirical results on standard CL benchmarks demonstrate that PLAN consistently outperforms existing methods, establishing a new state-of-the-art for continual learning with foundation models. 

---
# Reducing the Probability of Undesirable Outputs in Language Models Using Probabilistic Inference 

**Authors**: Stephen Zhao, Aidan Li, Rob Brekelmans, Roger Grosse  

**Link**: [PDF](https://arxiv.org/pdf/2510.21184)  

**Abstract**: Reinforcement learning (RL) has become a predominant technique to align language models (LMs) with human preferences or promote outputs which are deemed to be desirable by a given reward function. Standard RL approaches optimize average reward, while methods explicitly focused on reducing the probability of undesired outputs typically come at a cost to average-case performance. To improve this tradeoff, we introduce RePULSe, a new training method that augments the standard RL loss with an additional loss that uses learned proposals to guide sampling low-reward outputs, and then reduces those outputs' probability. We run experiments demonstrating that RePULSe produces a better tradeoff of expected reward versus the probability of undesired outputs and is more adversarially robust, compared to standard RL alignment approaches and alternatives. 

---
# Towards Straggler-Resilient Split Federated Learning: An Unbalanced Update Approach 

**Authors**: Dandan Liang, Jianing Zhang, Evan Chen, Zhe Li, Rui Li, Haibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21155)  

**Abstract**: Split Federated Learning (SFL) enables scalable training on edge devices by combining the parallelism of Federated Learning (FL) with the computational offloading of Split Learning (SL). Despite its great success, SFL suffers significantly from the well-known straggler issue in distributed learning systems. This problem is exacerbated by the dependency between Split Server and clients: the Split Server side model update relies on receiving activations from clients. Such synchronization requirement introduces significant time latency, making straggler a critical bottleneck to the scalability and efficiency of the system. To mitigate this problem, we propose MU-SplitFed, a straggler-resilient SFL algorithm in zeroth-order optimization that decouples training progress from straggler delays via a simple yet effective unbalanced update mechanism.
By enabling the server to perform $\tau$ local updates per client round, MU-SplitFed achieves a convergence rate of $O(\sqrt{d/(\tau T)})$ for non-convex objectives, demonstrating a linear speedup of $\tau$ in communication rounds. Experiments demonstrate that MU-SplitFed consistently outperforms baseline methods with the presence of stragglers and effectively mitigates their impact through adaptive tuning of $\tau$. The code for this project is available at this https URL. 

---
# Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design 

**Authors**: Lianghong Chen, Dongkyu Eugene Kim, Mike Domaratzki, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21153)  

**Abstract**: Designing de novo 3D molecules with desirable properties remains a fundamental challenge in drug discovery and molecular engineering. While diffusion models have demonstrated remarkable capabilities in generating high-quality 3D molecular structures, they often struggle to effectively control complex multi-objective constraints critical for real-world applications. In this study, we propose an uncertainty-aware Reinforcement Learning (RL) framework to guide the optimization of 3D molecular diffusion models toward multiple property objectives while enhancing the overall quality of the generated molecules. Our method leverages surrogate models with predictive uncertainty estimation to dynamically shape reward functions, facilitating balance across multiple optimization objectives. We comprehensively evaluate our framework across three benchmark datasets and multiple diffusion model architectures, consistently outperforming baselines for molecular quality and property optimization. Additionally, Molecular Dynamics (MD) simulations and ADMET profiling of top generated candidates indicate promising drug-like behavior and binding stability, comparable to known Epidermal Growth Factor Receptor (EGFR) inhibitors. Our results demonstrate the strong potential of RL-guided generative diffusion models for advancing automated molecular design. 

---
# Hierarchical AI Multi-Agent Fundamental Investing: Evidence from China's A-Share Market 

**Authors**: Chujun He, Zhonghao Huang, Xiangguo Li, Ye Luo, Kewei Ma, Yuxuan Xiong, Xiaowei Zhang, Mingyang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21147)  

**Abstract**: We present a multi-agent, AI-driven framework for fundamental investing that integrates macro indicators, industry-level and firm-specific information to construct optimized equity portfolios. The architecture comprises: (i) a Macro agent that dynamically screens and weights sectors based on evolving economic indicators and industry performance; (ii) four firm-level agents -- Fundamental, Technical, Report, and News -- that conduct in-depth analyses of individual firms to ensure both breadth and depth of coverage; (iii) a Portfolio agent that uses reinforcement learning to combine the agent outputs into a unified policy to generate the trading strategy; and (iv) a Risk Control agent that adjusts portfolio positions in response to market volatility. We evaluate the system on the constituents by the CSI 300 Index of China's A-share market and find that it consistently outperforms standard benchmarks and a state-of-the-art multi-agent trading system on risk-adjusted returns and drawdown control. Our core contribution is a hierarchical multi-agent design that links top-down macro screening with bottom-up fundamental analysis, offering a robust and extensible approach to factor-based portfolio construction. 

---
# Quantifying CBRN Risk in Frontier Models 

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21133)  

**Abstract**: Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities. 

---
# Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications 

**Authors**: Guangxin Su, Hanchen Wang, Jianwei Wang, Wenjie Zhang, Ying Zhang, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2510.21131)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing through strong semantic understanding and generation. However, their black-box nature limits structured and multi-hop reasoning. In contrast, Text-Attributed Graphs (TAGs) provide explicit relational structures enriched with textual context, yet often lack semantic depth. Recent research shows that combining LLMs and TAGs yields complementary benefits: enhancing TAG representation learning and improving the reasoning and interpretability of LLMs. This survey provides the first systematic review of LLM--TAG integration from an orchestration perspective. We introduce a novel taxonomy covering two fundamental directions: LLM for TAG, where LLMs enrich graph-based tasks, and TAG for LLM, where structured graphs improve LLM reasoning. We categorize orchestration strategies into sequential, parallel, and multi-module frameworks, and discuss advances in TAG-specific pretraining, prompting, and parameter-efficient fine-tuning. Beyond methodology, we summarize empirical insights, curate available datasets, and highlight diverse applications across recommendation systems, biomedical analysis, and knowledge-intensive question answering. Finally, we outline open challenges and promising research directions, aiming to guide future work at the intersection of language and graph learning. 

---
# Enhanced Evolutionary Multi-Objective Deep Reinforcement Learning for Reliable and Efficient Wireless Rechargeable Sensor Networks 

**Authors**: Bowei Tong, Hui Kang, Jiahui Li, Geng Sun, Jiacheng Wang, Yaoqi Yang, Bo Xu, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.21127)  

**Abstract**: Despite rapid advancements in sensor networks, conventional battery-powered sensor networks suffer from limited operational lifespans and frequent maintenance requirements that severely constrain their deployment in remote and inaccessible environments. As such, wireless rechargeable sensor networks (WRSNs) with mobile charging capabilities offer a promising solution to extend network lifetime. However, WRSNs face critical challenges from the inherent trade-off between maximizing the node survival rates and maximizing charging energy efficiency under dynamic operational conditions. In this paper, we investigate a typical scenario where mobile chargers move and charge the sensor, thereby maintaining the network connectivity while minimizing the energy waste. Specifically, we formulate a multi-objective optimization problem that simultaneously maximizes the network node survival rate and mobile charger energy usage efficiency across multiple time slots, which presents NP-hard computational complexity with long-term temporal dependencies that make traditional optimization approaches ineffective. To address these challenges, we propose an enhanced evolutionary multi-objective deep reinforcement learning algorithm, which integrates a long short-term memory (LSTM)-based policy network for temporal pattern recognition, a multilayer perceptron-based prospective increment model for future state prediction, and a time-varying Pareto policy evaluation method for dynamic preference adaptation. Extensive simulation results demonstrate that the proposed algorithm significantly outperforms existing approaches in balancing node survival rate and energy efficiency while generating diverse Pareto-optimal solutions. Moreover, the LSTM-enhanced policy network converges 25% faster than conventional networks, with the time-varying evaluation method effectively adapting to dynamic conditions. 

---
# Generalizable Hierarchical Skill Learning via Object-Centric Representation 

**Authors**: Haibo Zhao, Yu Qi, Boce Hu, Yizhe Zhu, Ziyan Chen, Heng Tian, Xupeng Zhu, Owen Howell, Haojie Huang, Robin Walters, Dian Wang, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2510.21121)  

**Abstract**: We present Generalizable Hierarchical Skill Learning (GSL), a novel framework for hierarchical policy learning that significantly improves policy generalization and sample efficiency in robot manipulation. One core idea of GSL is to use object-centric skills as an interface that bridges the high-level vision-language model and the low-level visual-motor policy. Specifically, GSL decomposes demonstrations into transferable and object-canonicalized skill primitives using foundation models, ensuring efficient low-level skill learning in the object frame. At test time, the skill-object pairs predicted by the high-level agent are fed to the low-level module, where the inferred canonical actions are mapped back to the world frame for execution. This structured yet flexible design leads to substantial improvements in sample efficiency and generalization of our method across unseen spatial arrangements, object appearances, and task compositions. In simulation, GSL trained with only 3 demonstrations per task outperforms baselines trained with 30 times more data by 15.5 percent on unseen tasks. In real-world experiments, GSL also surpasses the baseline trained with 10 times more data. 

---
# The Gray Zone of Faithfulness: Taming Ambiguity in Unfaithfulness Detection 

**Authors**: Qiang Ding, Lvzhou Luo, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21118)  

**Abstract**: Ensuring that Large Language Models (LLMs) generate summaries faithful to a given source document is essential for real-world applications. While prior research has explored LLM faithfulness, existing benchmarks suffer from annotation ambiguity, primarily due to the ill-defined boundary of permissible external knowledge in generated outputs. For instance, common sense is often incorporated into responses and labeled as "faithful", yet the acceptable extent of such knowledge remains unspecified, leading to inconsistent annotations. To address this issue, we propose a novel faithfulness annotation framework, which introduces an intermediate category, Out-Dependent, to classify cases where external knowledge is required for verification. Using this framework, we construct VeriGray (Verification with the Gray Zone) -- a new unfaithfulness detection benchmark in summarization. Statistics reveal that even SOTA LLMs, such as GPT-5, exhibit hallucinations ($\sim 6\%$ of sentences) in summarization tasks. Moreover, a substantial proportion ($\sim 8\%$ on average of models) of generated sentences fall into the Out-Dependent category, underscoring the importance of resolving annotation ambiguity in unfaithfulness detection benchmarks. Experiments demonstrate that our benchmark poses significant challenges to multiple baseline methods, indicating considerable room for future improvement. 

---
# Urban 3D Change Detection Using LiDAR Sensor for HD Map Maintenance and Smart Mobility 

**Authors**: Hezam Albagami, Haitian Wang, Xinyu Wang, Muhammad Ibrahim, Zainy M. Malakan, Abdullah M. Alqamdi, Mohammed H. Alghamdi, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2510.21112)  

**Abstract**: High-definition 3D city maps underpin smart transportation, digital twins, and autonomous driving, where object level change detection across bi temporal LiDAR enables HD map maintenance, construction monitoring, and reliable localization. Classical DSM differencing and image based methods are sensitive to small vertical bias, ground slope, and viewpoint mismatch and yield cellwise outputs without object identity. Point based neural models and voxel encodings demand large memory, assume near perfect pre alignment, degrade thin structures, and seldom enforce class consistent association, which leaves split or merge cases unresolved and ignores uncertainty. We propose an object centric, uncertainty aware pipeline for city scale LiDAR that aligns epochs with multi resolution NDT followed by point to plane ICP, normalizes height, and derives a per location level of detection from registration covariance and surface roughness to calibrate decisions and suppress spurious changes. Geometry only proxies seed cross epoch associations that are refined by semantic and instance segmentation and a class constrained bipartite assignment with augmented dummies to handle splits and merges while preserving per class counts. Tiled processing bounds memory without eroding narrow ground changes, and instance level decisions combine 3D overlap, normal direction displacement, and height and volume differences with a histogram distance, all gated by the local level of detection to remain stable under partial overlap and sampling variation. On 15 representative Subiaco blocks the method attains 95.2% accuracy, 90.4% mF1, and 82.6% mIoU, exceeding Triplet KPConv by 0.2 percentage points in accuracy, 0.2 in mF1, and 0.8 in mIoU, with the largest gain on Decreased where IoU reaches 74.8% and improves by 7.6 points. 

---
# ESCORT: Efficient Stein-variational and Sliced Consistency-Optimized Temporal Belief Representation for POMDPs 

**Authors**: Yunuo Zhang, Baiting Luo, Ayan Mukhopadhyay, Gabor Karsai, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2510.21107)  

**Abstract**: In Partially Observable Markov Decision Processes (POMDPs), maintaining and updating belief distributions over possible underlying states provides a principled way to summarize action-observation history for effective decision-making under uncertainty. As environments grow more realistic, belief distributions develop complexity that standard mathematical models cannot accurately capture, creating a fundamental challenge in maintaining representational accuracy. Despite advances in deep learning and probabilistic modeling, existing POMDP belief approximation methods fail to accurately represent complex uncertainty structures such as high-dimensional, multi-modal belief distributions, resulting in estimation errors that lead to suboptimal agent behaviors. To address this challenge, we present ESCORT (Efficient Stein-variational and sliced Consistency-Optimized Representation for Temporal beliefs), a particle-based framework for capturing complex, multi-modal distributions in high-dimensional belief spaces. ESCORT extends SVGD with two key innovations: correlation-aware projections that model dependencies between state dimensions, and temporal consistency constraints that stabilize updates while preserving correlation structures. This approach retains SVGD's attractive-repulsive particle dynamics while enabling accurate modeling of intricate correlation patterns. Unlike particle filters prone to degeneracy or parametric methods with fixed representational capacity, ESCORT dynamically adapts to belief landscape complexity without resampling or restrictive distributional assumptions. We demonstrate ESCORT's effectiveness through extensive evaluations on both POMDP domains and synthetic multi-modal distributions of varying dimensionality, where it consistently outperforms state-of-the-art methods in terms of belief approximation accuracy and downstream decision quality. 

---
# Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only 

**Authors**: Qingru Zhang, Liang Qiu, Ilgee Hong, Zhenghao Xu, Tianyi Liu, Shiyang Li, Rongzhi Zhang, Zheng Li, Lihong Li, Bing Yin, Chao Zhang, Jianshu Chen, Haoming Jiang, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21090)  

**Abstract**: Supervised fine-tuning (SFT) has emerged as a crucial method for aligning large language models (LLMs) with human-annotated demonstrations. However, SFT, being an off-policy approach similar to behavior cloning, often struggles with overfitting and poor out-of-domain generalization, especially in limited-data scenarios. To address these limitations, we propose Self-Rewarding PPO, a novel fine-tuning method that leverages on-policy techniques to enhance generalization performance. Our approach combines the strengths of SFT and proximal policy optimization (PPO) to achieve more effective alignment from demonstration data. At its core is a reward function designed as the log policy ratio between the SFT model and the pretrained base model. This function serves as an implicit reward signal, using the pretrained policy as a baseline and the SFT policy as a target. By doing so, it enables on-policy fine-tuning without relying on human preference annotations. The integration of this self-rewarding mechanism with PPO addresses key limitations of SFT, improving generalization, data efficiency, and robustness. Our empirical evaluation across a range of natural language processing tasks demonstrates that Self-Rewarding PPO consistently outperforms traditional SFT methods. The results highlight the effectiveness of our approach in aligning LLMs using demonstration data, particularly in scenarios where high-quality annotated data is scarce. 

---
# M-GLC: Motif-Driven Global-Local Context Graphs for Few-shot Molecular Property Prediction 

**Authors**: Xiangyang Xu, Hongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21088)  

**Abstract**: Molecular property prediction (MPP) is a cornerstone of drug discovery and materials science, yet conventional deep learning approaches depend on large labeled datasets that are often unavailable. Few-shot Molecular property prediction (FSMPP) addresses this scarcity by incorporating relational inductive bias through a context graph that links molecule nodes to property nodes, but such molecule-property graphs offer limited structural guidance. We propose a comprehensive solution: Motif Driven Global-Local Context Graph for few-shot molecular property prediction, which enriches contextual information at both the global and local levels. At the global level, chemically meaningful motif nodes representing shared substructures, such as rings or functional groups, are introduced to form a global tri-partite heterogeneous graph, yielding motif-molecule-property connections that capture long-range compositional patterns and enable knowledge transfer among molecules with common motifs. At the local level, we build a subgraph for each node in the molecule-property pair and encode them separately to concentrate the model's attention on the most informative neighboring molecules and motifs. Experiments on five standard FSMPP benchmarks demonstrate that our framework consistently outperforms state-of-the-art methods. These results underscore the effectiveness of integrating global motif knowledge with fine-grained local context to advance robust few-shot molecular property prediction. 

---
# CDrugRed: A Chinese Drug Recommendation Dataset for Discharge Medications in Metabolic Diseases 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21084)  

**Abstract**: Intelligent drug recommendation based on Electronic Health Records (EHRs) is critical for improving for improving the quality and efficiency of clinical decision-making. By leveraging large-scale patient data, drug recommendation systems can assist physicians in selecting the most appropriate medications according to a patient's medical history, diagnoses, laboratory results, and comorbidities. However, the advancement of such systems is significantly hampered by the scarcity of publicly available, real-world EHR datasets, particularly in languages other than English. In this work, we present CDrugRed, a first publicly available Chinese drug recommendation dataset focused on discharge medications for metabolic diseases. The dataset includes 5,894 de-identified records from 3,190 patients, containing comprehensive information such as patient demographics, medical history, clinical course, and discharge diagnoses. We assess the utility of CDrugRed by benchmarking several state-of-the-art large language models (LLMs) on the discharge medication recommendation task. Experimental results show that while supervised fine-tuning improves model performance, there remains substantial room for improvement, with the best model achieving the F1 score of 0.5648 and Jaccard score of 0.4477. This result highlights the complexity of the clinical drug recommendation task and establishes CDrugRed as a challenging and valuable resource for developing more robust and accurate drug recommendation systems. The dataset is publicly available to the research community under the data usage agreements at this https URL. 

---
# Soppia: A Structured Prompting Framework for the Proportional Assessment of Non-Pecuniary Damages in Personal Injury Cases 

**Authors**: Jorge Alberto Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21082)  

**Abstract**: Applying complex legal rules characterized by multiple, heterogeneously weighted criteria presents a fundamental challenge in judicial decision-making, often hindering the consistent realization of legislative intent. This challenge is particularly evident in the quantification of non-pecuniary damages in personal injury cases. This paper introduces Soppia, a structured prompting framework designed to assist legal professionals in navigating this complexity. By leveraging advanced AI, the system ensures a comprehensive and balanced analysis of all stipulated criteria, fulfilling the legislator's intent that compensation be determined through a holistic assessment of each case. Using the twelve criteria for non-pecuniary damages established in the Brazilian CLT (Art. 223-G) as a case study, we demonstrate how Soppia (System for Ordered Proportional and Pondered Intelligent Assessment) operationalizes nuanced legal commands into a practical, replicable, and transparent methodology. The framework enhances consistency and predictability while providing a versatile and explainable tool adaptable across multi-criteria legal contexts, bridging normative interpretation and computational reasoning toward auditable legal AI. 

---
# Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering 

**Authors**: William Christian, Daniel Adamlu, Adrian Yu, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2510.21068)  

**Abstract**: Question Answering (QA) has seen significant improvements with the advancement of machine learning models, further studies enhanced this question answering system by retrieving external information, called Retrieval-Augmented Generation (RAG) to produce more accurate and informative answers. However, these state-of-the-art-performance is predominantly in English language. To address this gap we made an effort of bridging language gaps by incorporating Adaptive RAG system to Indonesian language. Adaptive RAG system integrates a classifier whose task is to distinguish the question complexity, which in turn determines the strategy for answering the question. To overcome the limited availability of Indonesian language dataset, our study employs machine translation as data augmentation approach. Experiments show reliable question complexity classifier; however, we observed significant inconsistencies in multi-retrieval answering strategy which negatively impacted the overall evaluation when this strategy was applied. These findings highlight both the promise and challenges of question answering in low-resource language suggesting directions for future improvement. 

---
# Deep learning-based automated damage detection in concrete structures using images from earthquake events 

**Authors**: Abdullah Turer, Yongsheng Bai, Halil Sezen, Alper Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21063)  

**Abstract**: Timely assessment of integrity of structures after seismic events is crucial for public safety and emergency response. This study focuses on assessing the structural damage conditions using deep learning methods to detect exposed steel reinforcement in concrete buildings and bridges after large earthquakes. Steel bars are typically exposed after concrete spalling or large flexural or shear cracks. The amount and distribution of exposed steel reinforcement is an indication of structural damage and degradation. To automatically detect exposed steel bars, new datasets of images collected after the 2023 Turkey Earthquakes were labeled to represent a wide variety of damaged concrete structures. The proposed method builds upon a deep learning framework, enhanced with fine-tuning, data augmentation, and testing on public datasets. An automated classification framework is developed that can be used to identify inside/outside buildings and structural components. Then, a YOLOv11 (You Only Look Once) model is trained to detect cracking and spalling damage and exposed bars. Another YOLO model is finetuned to distinguish different categories of structural damage levels. All these trained models are used to create a hybrid framework to automatically and reliably determine the damage levels from input images. This research demonstrates that rapid and automated damage detection following disasters is achievable across diverse damage contexts by utilizing image data collection, annotation, and deep learning approaches. 

---
# On the Sample Complexity of Differentially Private Policy Optimization 

**Authors**: Yi He, Xingyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21060)  

**Abstract**: Policy optimization (PO) is a cornerstone of modern reinforcement learning (RL), with diverse applications spanning robotics, healthcare, and large language model training. The increasing deployment of PO in sensitive domains, however, raises significant privacy concerns. In this paper, we initiate a theoretical study of differentially private policy optimization, focusing explicitly on its sample complexity. We first formalize an appropriate definition of differential privacy (DP) tailored to PO, addressing the inherent challenges arising from on-policy learning dynamics and the subtlety involved in defining the unit of privacy. We then systematically analyze the sample complexity of widely-used PO algorithms, including policy gradient (PG), natural policy gradient (NPG) and more, under DP constraints and various settings, via a unified framework. Our theoretical results demonstrate that privacy costs can often manifest as lower-order terms in the sample complexity, while also highlighting subtle yet important observations in private PO settings. These offer valuable practical insights for privacy-preserving PO algorithms. 

---
# Reasoning's Razor: Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety and Hallucination Detection 

**Authors**: Atoosa Chegini, Hamid Kazemi, Garrett Souza, Maria Safi, Yang Song, Samy Bengio, Sinead Williamson, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21049)  

**Abstract**: Reasoning has become a central paradigm for large language models (LLMs), consistently boosting accuracy across diverse benchmarks. Yet its suitability for precision-sensitive tasks remains unclear. We present the first systematic study of reasoning for classification tasks under strict low false positive rate (FPR) regimes. Our analysis covers two tasks--safety detection and hallucination detection--evaluated in both fine-tuned and zero-shot settings, using standard LLMs and Large Reasoning Models (LRMs). Our results reveal a clear trade-off: Think On (reasoning-augmented) generation improves overall accuracy, but underperforms at the low-FPR thresholds essential for practical use. In contrast, Think Off (no reasoning during inference) dominates in these precision-sensitive regimes, with Think On surpassing only when higher FPRs are acceptable. In addition, we find token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments. Finally, a simple ensemble of the two modes recovers the strengths of each. Taken together, our findings position reasoning as a double-edged tool: beneficial for average accuracy, but often ill-suited for applications requiring strict precision. 

---
# AgentArcEval: An Architecture Evaluation Method for Foundation Model based Agents 

**Authors**: Qinghua Lu, Dehai Zhao, Yue Liu, Hao Zhang, Liming Zhu, Xiwei Xu, Angela Shi, Tristan Tan, Rick Kazman  

**Link**: [PDF](https://arxiv.org/pdf/2510.21031)  

**Abstract**: The emergence of foundation models (FMs) has enabled the development of highly capable and autonomous agents, unlocking new application opportunities across a wide range of domains. Evaluating the architecture of agents is particularly important as the architectural decisions significantly impact the quality attributes of agents given their unique characteristics, including compound architecture, autonomous and non-deterministic behaviour, and continuous evolution. However, these traditional methods fall short in addressing the evaluation needs of agent architecture due to the unique characteristics of these agents. Therefore, in this paper, we present AgentArcEval, a novel agent architecture evaluation method designed specially to address the complexities of FM-based agent architecture and its evaluation. Moreover, we present a catalogue of agent-specific general scenarios, which serves as a guide for generating concrete scenarios to design and evaluate the agent architecture. We demonstrate the usefulness of AgentArcEval and the catalogue through a case study on the architecture evaluation of a real-world tax copilot, named Luna. 

---
# JSTprove: Pioneering Verifiable AI for a Trustless Future 

**Authors**: Jonathan Gold, Tristan Freiberg, Haruna Isah, Shirin Shahabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21024)  

**Abstract**: The integration of machine learning (ML) systems into critical industries such as healthcare, finance, and cybersecurity has transformed decision-making processes, but it also brings new challenges around trust, security, and accountability. As AI systems become more ubiquitous, ensuring the transparency and correctness of AI-driven decisions is crucial, especially when they have direct consequences on privacy, security, or fairness. Verifiable AI, powered by Zero-Knowledge Machine Learning (zkML), offers a robust solution to these challenges. zkML enables the verification of AI model inferences without exposing sensitive data, providing an essential layer of trust and privacy. However, traditional zkML systems typically require deep cryptographic expertise, placing them beyond the reach of most ML engineers. In this paper, we introduce JSTprove, a specialized zkML toolkit, built on Polyhedra Network's Expander backend, to enable AI developers and ML engineers to generate and verify proofs of AI inference. JSTprove provides an end-to-end verifiable AI inference pipeline that hides cryptographic complexity behind a simple command-line interface while exposing auditable artifacts for reproducibility. We present the design, innovations, and real-world use cases of JSTprove as well as our blueprints and tooling to encourage community review and extension. JSTprove therefore serves both as a usable zkML product for current engineering needs and as a reproducible foundation for future research and production deployments of verifiable AI. 

---
# Physically consistent and uncertainty-aware learning of spatiotemporal dynamics 

**Authors**: Qingsong Xu, Jonathan L Bamber, Nils Thuerey, Niklas Boers, Paul Bates, Gustau Camps-Valls, Yilei Shi, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21023)  

**Abstract**: Accurate long-term forecasting of spatiotemporal dynamics remains a fundamental challenge across scientific and engineering domains. Existing machine learning methods often neglect governing physical laws and fail to quantify inherent uncertainties in spatiotemporal predictions. To address these challenges, we introduce a physics-consistent neural operator (PCNO) that enforces physical constraints by projecting surrogate model outputs onto function spaces satisfying predefined laws. A physics-consistent projection layer within PCNO efficiently computes mass and momentum conservation in Fourier space. Building upon deterministic predictions, we further propose a diffusion model-enhanced PCNO (DiffPCNO), which leverages a consistency model to quantify and mitigate uncertainties, thereby improving the accuracy and reliability of forecasts. PCNO and DiffPCNO achieve high-fidelity spatiotemporal predictions while preserving physical consistency and uncertainty across diverse systems and spatial resolutions, ranging from turbulent flow modeling to real-world flood/atmospheric forecasting. Our two-stage framework provides a robust and versatile approach for accurate, physically grounded, and uncertainty-aware spatiotemporal forecasting. 

---
# Race and Gender in LLM-Generated Personas: A Large-Scale Audit of 41 Occupations 

**Authors**: Ilona van der Linden, Sahana Kumar, Arnav Dixit, Aadi Sudan, Smruthi Danda, David C. Anastasiu, Kai Lukoff  

**Link**: [PDF](https://arxiv.org/pdf/2510.21011)  

**Abstract**: Generative AI tools are increasingly used to create portrayals of people in occupations, raising concerns about how race and gender are represented. We conducted a large-scale audit of over 1.5 million occupational personas across 41 U.S. occupations, generated by four large language models with different AI safety commitments and countries of origin (U.S., China, France). Compared with Bureau of Labor Statistics data, we find two recurring patterns: systematic shifts, where some groups are consistently under- or overrepresented, and stereotype exaggeration, where existing demographic skews are amplified. On average, White (--31pp) and Black (--9pp) workers are underrepresented, while Hispanic (+17pp) and Asian (+12pp) workers are overrepresented. These distortions can be extreme: for example, across all four models, Housekeepers are portrayed as nearly 100\% Hispanic, while Black workers are erased from many occupations. For HCI, these findings show provider choice materially changes who is visible, motivating model-specific audits and accountable design practices. 

---
# Exploring Spiking Neural Networks for Binary Classification in Multivariate Time Series at the Edge 

**Authors**: James Ghawaly, Andrew Nicholson, Catherine Schuman, Dalton Diez, Aaron Young, Brett Witherspoon  

**Link**: [PDF](https://arxiv.org/pdf/2510.20997)  

**Abstract**: We present a general framework for training spiking neural networks (SNNs) to perform binary classification on multivariate time series, with a focus on step-wise prediction and high precision at low false alarm rates. The approach uses the Evolutionary Optimization of Neuromorphic Systems (EONS) algorithm to evolve sparse, stateful SNNs by jointly optimizing their architectures and parameters. Inputs are encoded into spike trains, and predictions are made by thresholding a single output neuron's spike counts. We also incorporate simple voting ensemble methods to improve performance and robustness.
To evaluate the framework, we apply it with application-specific optimizations to the task of detecting low signal-to-noise ratio radioactive sources in gamma-ray spectral data. The resulting SNNs, with as few as 49 neurons and 66 synapses, achieve a 51.8% true positive rate (TPR) at a false alarm rate of 1/hr, outperforming PCA (42.7%) and deep learning (49.8%) baselines. A three-model any-vote ensemble increases TPR to 67.1% at the same false alarm rate. Hardware deployment on the microCaspian neuromorphic platform demonstrates 2mW power consumption and 20.2ms inference latency.
We also demonstrate generalizability by applying the same framework, without domain-specific modification, to seizure detection in EEG recordings. An ensemble achieves 95% TPR with a 16% false positive rate, comparable to recent deep learning approaches with significant reduction in parameter count. 

---
# VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models 

**Authors**: Jesimon Barreto, Carlos Caetano, André Araujo, William Robson Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2510.20994)  

**Abstract**: Foundation models have advanced computer vision by enabling strong performance across diverse tasks through large-scale pretraining and supervised fine-tuning. However, they may underperform in domains with distribution shifts and scarce labels, where supervised fine-tuning may be infeasible. While continued self-supervised learning for model adaptation is common for generative language models, this strategy has not proven effective for vision-centric encoder models. To address this challenge, we introduce a novel formulation of self-supervised fine-tuning for vision foundation models, where the model is adapted to a new domain without requiring annotations, leveraging only short multi-view object-centric videos. Our method is referred to as VESSA: Video-based objEct-centric Self-Supervised Adaptation for visual foundation models. VESSA's training technique is based on a self-distillation paradigm, where it is critical to carefully tune prediction heads and deploy parameter-efficient adaptation techniques - otherwise, the model may quickly forget its pretrained knowledge and reach a degraded state. VESSA benefits significantly from multi-view object observations sourced from different frames in an object-centric video, efficiently learning robustness to varied capture conditions, without the need of annotations. Through comprehensive experiments with 3 vision foundation models on 2 datasets, VESSA demonstrates consistent improvements in downstream classification tasks, compared to the base models and previous adaptation methods. Code is publicly available at this https URL. 

---
# GPU Memory Requirement Prediction for Deep Learning Task Based on Bidirectional Gated Recurrent Unit Optimization Transformer 

**Authors**: Chao Wang, Zhizhao Wen, Ruoxin Zhang, Puyang Xu, Yifan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20985)  

**Abstract**: In response to the increasingly critical demand for accurate prediction of GPU memory resources in deep learning tasks, this paper deeply analyzes the current research status and innovatively proposes a deep learning model that integrates bidirectional gated recurrent units (BiGRU) to optimize the Transformer architecture, aiming to improve the accuracy of memory demand prediction. To verify the effectiveness of the model, a carefully designed comparative experiment was conducted, selecting four representative basic machine learning models: decision tree, random forest, Adaboost, and XGBoost as benchmarks. The detailed experimental results show that the BiGRU Transformer optimization model proposed in this paper exhibits significant advantages in key evaluation indicators: in terms of mean square error (MSE) and root mean square error (RMSE), the model achieves the lowest value among all comparison models, and its predicted results have the smallest deviation from the actual values; In terms of mean absolute error (MAE) and coefficient of determination (R2) indicators, the model also performs well and the results are balanced and stable, with comprehensive predictive performance far exceeding the benchmark machine learning methods compared. In summary, the Transformer model based on bidirectional gated recurrent unit optimization successfully constructed in this study can efficiently and accurately complete GPU memory demand prediction tasks in deep learning tasks, and its prediction accuracy has been significantly improved compared to traditional machine learning methods. This research provides strong technical support and reliable theoretical basis for optimizing resource scheduling and management of deep learning tasks, and improving the utilization efficiency of computing clusters. 

---
# Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression 

**Authors**: Xi Zhang, Xiaolin Wu, Jiamang Wang, Weisi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.20984)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities but typically require extensive computational resources and memory for inference. Post-training quantization (PTQ) can effectively reduce these demands by storing weights in lower bit-width formats. However, standard uniform quantization often leads to notable performance degradation, particularly in low-bit scenarios. In this work, we introduce a Grouped Lattice Vector Quantization (GLVQ) framework that assigns each group of weights a customized lattice codebook, defined by a learnable generation matrix. To address the non-differentiability of the quantization process, we adopt Babai rounding to approximate nearest-lattice-point search during training, which enables stable optimization of the generation matrices. Once trained, decoding reduces to a simple matrix-vector multiplication, yielding an efficient and practical quantization pipeline. Experiments on multiple benchmarks show that our approach achieves a better trade-off between model size and accuracy compared to existing post-training quantization baselines, highlighting its effectiveness in deploying large models under stringent resource constraints. Our source code is available on GitHub repository: this https URL. 

---
# Memory Constrained Dynamic Subnetwork Update for Transfer Learning 

**Authors**: Aël Quélennec, Pavlo Mozharovskyi, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2510.20979)  

**Abstract**: On-device neural network training faces critical memory constraints that limit the adaptation of pre-trained models to downstream tasks. We present MeDyate, a theoretically-grounded framework for memory-constrained dynamic subnetwork adaptation. Our approach introduces two key innovations: LaRa (Layer Ranking), an improved layer importance metric that enables principled layer pre-selection, and a dynamic channel sampling strategy that exploits the temporal stability of channel importance distributions during fine-tuning. MeDyate dynamically resamples channels between epochs according to importance-weighted probabilities, ensuring comprehensive parameter space exploration while respecting strict memory budgets. Extensive evaluation across a large panel of tasks and architectures demonstrates that MeDyate achieves state-of-the-art performance under extreme memory constraints, consistently outperforming existing static and dynamic approaches while maintaining high computational efficiency. Our method represents a significant step towards enabling efficient on-device learning by demonstrating effective fine-tuning with memory budgets as low as a few hundred kB of RAM. 

---
# REx86: A Local Large Language Model for Assisting in x86 Assembly Reverse Engineering 

**Authors**: Darrin Lea, James Ghawaly, Golden Richard III, Aisha Ali-Gombe, Andrew Case  

**Link**: [PDF](https://arxiv.org/pdf/2510.20975)  

**Abstract**: Reverse engineering (RE) of x86 binaries is indispensable for malware and firmware analysis, but remains slow due to stripped metadata and adversarial obfuscation. Large Language Models (LLMs) offer potential for improving RE efficiency through automated comprehension and commenting, but cloud-hosted, closed-weight models pose privacy and security risks and cannot be used in closed-network facilities. We evaluate parameter-efficient fine-tuned local LLMs for assisting with x86 RE tasks in these settings. Eight open-weight models across the CodeLlama, Qwen2.5-Coder, and CodeGemma series are fine-tuned on a custom curated dataset of 5,981 x86 assembly examples. We evaluate them quantitatively and identify the fine-tuned Qwen2.5-Coder-7B as the top performer, which we name REx86.
REx86 reduces test-set cross-entropy loss by 64.2% and improves semantic cosine similarity against ground truth by 20.3\% over its base model. In a limited user case study (n=43), REx86 significantly enhanced line-level code understanding (p = 0.031) and increased the correct-solve rate from 31% to 53% (p = 0.189), though the latter did not reach statistical significance. Qualitative analysis shows more accurate, concise comments with fewer hallucinations.
REx86 delivers state-of-the-art assistance in x86 RE among local, open-weight LLMs. Our findings demonstrate the value of domain-specific fine-tuning, and highlight the need for more commented disassembly data to further enhance LLM performance in RE. REx86, its dataset, and LoRA adapters are publicly available at this https URL and this https URL. 

---
# 3DReasonKnee: Advancing Grounded Reasoning in Medical Vision Language Models 

**Authors**: Sraavya Sambara, Sung Eun Kim, Xiaoman Zhang, Luyang Luo, Shreya Johri, Mohammed Baharoon, Du Hyun Ro, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2510.20967)  

**Abstract**: Current Vision-Language Models (VLMs) struggle to ground anatomical regions in 3D medical images and reason about them in a step-by-step manner, a key requirement of real-world diagnostic assessment. This ability is essential for aligning model outputs with the diagnostic workflows clinicians use in practice, enabling trustworthy clinician-AI collaboration. Existing 3D datasets provide localization labels, but none support this "grounded reasoning" ability. To address this gap, we introduce 3DReasonKnee, the first 3D grounded reasoning dataset for medical images, which provides 494k high-quality quintuples derived from 7,970 3D knee MRI volumes. Each quintuple includes: (1) the 3D MRI volume, (2) a diagnostic question targeting a specific anatomical region (3) a 3D bounding box localizing the relevant anatomical structures, (4) clinician-generated diagnostic reasoning steps that explicitly detail the 3D reasoning process, and (5) structured severity assessments for the relevant anatomical region. The creation and validation of 3DReasonKnee, involving over 450 hours of expert clinician time for manually segmenting MRIs and generating reasoning chains, ensures its superior quality and clinical relevance. We establish ReasonKnee-Bench to evaluate localization and diagnostic accuracy, providing insight into VLM ability to perform grounding and severity assessment across anatomical regions and diagnostic inquiries. We benchmark five state-of-the-art VLMs, providing baseline performance for ReasonKnee-Bench. By providing this unique resource of expert-annotated 3D reasoning pathways, 3DReasonKnee serves as a repository of orthopedic surgeons' diagnostic expertise and offers a vital testbed for advancing multimodal medical AI systems towards 3D, clinically aligned, localized decision-making capabilities. The dataset can be found in: this https URL 

---
# Meta-Learning for Cross-Task Generalization in Protein Mutation Property Prediction 

**Authors**: Srivathsan Badrinarayanan, Yue Su, Janghoon Ock, Alan Pham, Sanya Ahuja, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2510.20943)  

**Abstract**: Protein mutations can have profound effects on biological function, making accurate prediction of property changes critical for drug discovery, protein engineering, and precision medicine. Current approaches rely on fine-tuning protein-specific transformers for individual datasets, but struggle with cross-dataset generalization due to heterogeneous experimental conditions and limited target domain data. We introduce two key innovations: (1) the first application of Model-Agnostic Meta-Learning (MAML) to protein mutation property prediction, and (2) a novel mutation encoding strategy using separator tokens to directly incorporate mutations into sequence context. We build upon transformer architectures integrating them with MAML to enable rapid adaptation to new tasks through minimal gradient steps rather than learning dataset-specific patterns. Our mutation encoding addresses the critical limitation where standard transformers treat mutation positions as unknown tokens, significantly degrading performance. Evaluation across three diverse protein mutation datasets (functional fitness, thermal stability, and solubility) demonstrates significant advantages over traditional fine-tuning. In cross-task evaluation, our meta-learning approach achieves 29% better accuracy for functional fitness with 65% less training time, and 94% better accuracy for solubility with 55% faster training. The framework maintains consistent training efficiency regardless of dataset size, making it particularly valuable for industrial applications and early-stage protein design where experimental data is limited. This work establishes a systematic application of meta-learning to protein mutation analysis and introduces an effective mutation encoding strategy, offering transformative methodology for cross-domain generalization in protein engineering. 

---
# Do LLMs Truly Understand When a Precedent Is Overruled? 

**Authors**: Li Zhang, Jaromir Savelka, Kevin Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2510.20941)  

**Abstract**: Large language models (LLMs) with extended context windows show promise for complex legal reasoning tasks, yet their ability to understand long legal documents remains insufficiently evaluated. Developing long-context benchmarks that capture realistic, high-stakes tasks remains a significant challenge in the field, as most existing evaluations rely on simplified synthetic tasks that fail to represent the complexity of real-world document understanding. Overruling relationships are foundational to common-law doctrine and commonly found in judicial opinions. They provide a focused and important testbed for long-document legal understanding that closely resembles what legal professionals actually do. We present an assessment of state-of-the-art LLMs on identifying overruling relationships from U.S. Supreme Court cases using a dataset of 236 case pairs. Our evaluation reveals three critical limitations: (1) era sensitivity -- the models show degraded performance on historical cases compared to modern ones, revealing fundamental temporal bias in their training; (2) shallow reasoning -- models rely on shallow logical heuristics rather than deep legal comprehension; and (3) context-dependent reasoning failures -- models produce temporally impossible relationships in complex open-ended tasks despite maintaining basic temporal awareness in simple contexts. Our work contributes a benchmark that addresses the critical gap in realistic long-context evaluation, providing an environment that mirrors the complexity and stakes of actual legal reasoning tasks. 

---
# Focal Modulation and Bidirectional Feature Fusion Network for Medical Image Segmentation 

**Authors**: Moin Safdar, Shahzaib Iqbal, Mehwish Mehmood, Mubeen Ghafoor, Tariq M.Khan, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2510.20933)  

**Abstract**: Medical image segmentation is essential for clinical applications such as disease diagnosis, treatment planning, and disease development monitoring because it provides precise morphological and spatial information on anatomical structures that directly influence treatment decisions. Convolutional neural networks significantly impact image segmentation; however, since convolution operations are local, capturing global contextual information and long-range dependencies is still challenging. Their capacity to precisely segment structures with complicated borders and a variety of sizes is impacted by this restriction. Since transformers use self-attention methods to capture global context and long-range dependencies efficiently, integrating transformer-based architecture with CNNs is a feasible approach to overcoming these challenges. To address these challenges, we propose the Focal Modulation and Bidirectional Feature Fusion Network for Medical Image Segmentation, referred to as FM-BFF-Net in the remainder of this paper. The network combines convolutional and transformer components, employs a focal modulation attention mechanism to refine context awareness, and introduces a bidirectional feature fusion module that enables efficient interaction between encoder and decoder representations across scales. Through this design, FM-BFF-Net enhances boundary precision and robustness to variations in lesion size, shape, and contrast. Extensive experiments on eight publicly available datasets, including polyp detection, skin lesion segmentation, and ultrasound imaging, show that FM-BFF-Net consistently surpasses recent state-of-the-art methods in Jaccard index and Dice coefficient, confirming its effectiveness and adaptability for diverse medical imaging scenarios. 

---
# An Experimental Study of Trojan Vulnerabilities in UAV Autonomous Landing 

**Authors**: Reza Ahmari, Ahmad Mohammadi, Vahid Hemmati, Mohammed Mynuddin, Mahmoud Nabil Mahmoud, Parham Kebria, Abdollah Homaifar, Mehrdad Saif  

**Link**: [PDF](https://arxiv.org/pdf/2510.20932)  

**Abstract**: This study investigates the vulnerabilities of autonomous navigation and landing systems in Urban Air Mobility (UAM) vehicles. Specifically, it focuses on Trojan attacks that target deep learning models, such as Convolutional Neural Networks (CNNs). Trojan attacks work by embedding covert triggers within a model's training data. These triggers cause specific failures under certain conditions, while the model continues to perform normally in other situations. We assessed the vulnerability of Urban Autonomous Aerial Vehicles (UAAVs) using the DroNet framework. Our experiments showed a significant drop in accuracy, from 96.4% on clean data to 73.3% on data triggered by Trojan attacks. To conduct this study, we collected a custom dataset and trained models to simulate real-world conditions. We also developed an evaluation framework designed to identify Trojan-infected models. This work demonstrates the potential security risks posed by Trojan attacks and lays the groundwork for future research on enhancing the resilience of UAM systems. 

---
# Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference 

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney  

**Link**: [PDF](https://arxiv.org/pdf/2510.20930)  

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.
Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases 

---
# Aircraft Collision Avoidance Systems: Technological Challenges and Solutions on the Path to Regulatory Acceptance 

**Authors**: Sydney M. Katz, Robert J. Moss, Dylan M. Asmar, Wesley A. Olson, James K. Kuchar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.20916)  

**Abstract**: Aircraft collision avoidance systems is critical to modern aviation. These systems are designed to predict potential collisions between aircraft and recommend appropriate avoidance actions. Creating effective collision avoidance systems requires solutions to a variety of technical challenges related to surveillance, decision making, and validation. These challenges have sparked significant research and development efforts over the past several decades that have resulted in a variety of proposed solutions. This article provides an overview of these challenges and solutions with an emphasis on those that have been put through a rigorous validation process and accepted by regulatory bodies. The challenges posed by the collision avoidance problem are often present in other domains, and aircraft collision avoidance systems can serve as case studies that provide valuable insights for a wide range of safety-critical systems. 

---
# Code-enabled language models can outperform reasoning models on diverse tasks 

**Authors**: Cedegao E. Zhang, Cédric Colas, Gabriel Poesia, Joshua B. Tenenbaum, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2510.20909)  

**Abstract**: Reasoning models (RMs), language models (LMs) trained with reinforcement learning to produce long-form natural language reasoning, have been remarkably successful, but they still require large amounts of computation and data to train, and can be slow and expensive to run. In this paper, we show that standard instruct LMs can already be elicited to be strong reasoners at a level comparable to or even surpassing their corresponding RMs (e.g., DeepSeek V3 vs R1) without finetuning, across diverse domains from instruction following and creative generation to mathematical reasoning. This is achieved by CodeAdapt, our simple recipe that combines the CodeAct framework, where LMs interleave natural language reasoning with code execution in a multi-step fashion, with few-shot bootstrap in-context learning from as few as five training problems. Analyzing four matched pairs of LMs and RMs, we find that CodeAdapt enables three LMs to outperform the corresponding RMs on average over eight tasks (up to 22.9%) while being 10-81% more token efficient, and delivers superior performance on six tasks when averaged over the four models (up to 35.7%). Furthermore, the code-augmented reasoning traces display rich and varied problem-solving strategies. Our findings support that (1) CodeAdapt-style learning and reasoning may be robust and domain general and (2) code-enabled LMs are cognitively grounded and powerful systems, potentially providing a strong foundation for in-weight reinforcement learning. 

---
# Video-As-Prompt: Unified Semantic Control for Video Generation 

**Authors**: Yuxuan Bian, Xin Chen, Zenan Li, Tiancheng Zhi, Shen Sang, Linjie Luo, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20888)  

**Abstract**: Unified, generalizable semantic control in video generation remains a critical open challenge. Existing methods either introduce artifacts by enforcing inappropriate pixel-wise priors from structure-based controls, or rely on non-generalizable, condition-specific finetuning or task-specific architectures. We introduce Video-As-Prompt (VAP), a new paradigm that reframes this problem as in-context generation. VAP leverages a reference video as a direct semantic prompt, guiding a frozen Video Diffusion Transformer (DiT) via a plug-and-play Mixture-of-Transformers (MoT) expert. This architecture prevents catastrophic forgetting and is guided by a temporally biased position embedding that eliminates spurious mapping priors for robust context retrieval. To power this approach and catalyze future research, we built VAP-Data, the largest dataset for semantic-controlled video generation with over 100K paired videos across 100 semantic conditions. As a single unified model, VAP sets a new state-of-the-art for open-source methods, achieving a 38.7% user preference rate that rivals leading condition-specific commercial models. VAP's strong zero-shot generalization and support for various downstream applications mark a significant advance toward general-purpose, controllable video generation. 

---
# Preventing Shortcuts in Adapter Training via Providing the Shortcuts 

**Authors**: Anujraaj Argo Goyal, Guocheng Gordon Qian, Huseyin Coskun, Aarush Gupta, Himmy Tam, Daniil Ostashev, Ju Hu, Dhritiman Sagar, Sergey Tulyakov, Kfir Aberman, Kuan-Chieh Jackson Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20887)  

**Abstract**: Adapter-based training has emerged as a key mechanism for extending the capabilities of powerful foundation image generators, enabling personalized and stylized text-to-image synthesis. These adapters are typically trained to capture a specific target attribute, such as subject identity, using single-image reconstruction objectives. However, because the input image inevitably contains a mixture of visual factors, adapters are prone to entangle the target attribute with incidental ones, such as pose, expression, and lighting. This spurious correlation problem limits generalization and obstructs the model's ability to adhere to the input text prompt. In this work, we uncover a simple yet effective solution: provide the very shortcuts we wish to eliminate during adapter training. In Shortcut-Rerouted Adapter Training, confounding factors are routed through auxiliary modules, such as ControlNet or LoRA, eliminating the incentive for the adapter to internalize them. The auxiliary modules are then removed during inference. When applied to tasks like facial and full-body identity injection, our approach improves generation quality, diversity, and prompt adherence. These results point to a general design principle in the era of large models: when seeking disentangled representations, the most effective path may be to establish shortcuts for what should NOT be learned. 

---
# Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People 

**Authors**: Gabriel Grand, Valerio Pepe, Jacob Andreas, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2510.20886)  

**Abstract**: Many high-stakes applications of AI require forming data-driven hypotheses and making targeted guesses; e.g., in scientific and diagnostic settings. Given limited resources, to what extent do agents based on language models (LMs) act rationally? We develop methods to benchmark and enhance agentic information-seeking, drawing on insights from human behavior. First, we introduce a strategic decision-oriented dialogue task called Collaborative Battleship, in which a partially-informed Captain must balance exploration (asking questions) and action (taking shots), while a fully-informed Spotter must provide accurate answers under an information bottleneck. Compared to human players (N=42), we find that LM agents struggle to ground answers in context, generate informative questions, and select high-value actions. Next, to address these gaps, we develop novel Monte Carlo inference strategies for LMs based on principles from Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% -> 82% win rate) and frontier models (0% -> 67% win rate vs. GPT-5) at ~1% of GPT-5's cost. We replicate these findings on Guess Who? where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building rational information-seeking agents. 

---
# HA-RAG: Hotness-Aware RAG Acceleration via Mixed Precision and Data Placement 

**Authors**: Danying Ge, Jianhua Gao, Yixue Yang, Weixing Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.20878)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves model output accuracy by leveraging external knowledge bases, serving as an effective solution to address hallucination issues and knowledge-update delays in Large Language Models (LLMs). However, the introduction of external knowledge bases presents RAG with challenges in long-context processing, significantly increasing memory consumption and inference latency. Existing research accelerates inference by precomputing Key and Value (KV) of the knowledge base and loading them on-demand during inference. Based on the access frequency of different KV chunks within the external knowledge base, this paper proposes a hotness-aware RAG (HA-RAG) inference optimization system. First, leveraging the numerical distribution of KV chunks, we introduce a hotness-aware mixed-precision compressing and loading method to reduce disk I/O and memory access overhead. Second, we design a hotness-aware data placement strategy that prioritizes storing frequently accessed KV chunks in high-speed memory to improve data access efficiency. Experimental results demonstrate that, compared with TurboRAG, the proposed HA-RAG achieves an average speedup of 2.10x and maximum speedup of 10.49x in Time-To-First-Token (TTFT) with negligible accuracy loss. 

---
# Multimodal Negative Learning 

**Authors**: Baoquan Gong, Xiyuan Gao, Pengfei Zhu, Qinghua Hu, Bing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20877)  

**Abstract**: Multimodal learning systems often encounter challenges related to modality imbalance, where a dominant modality may overshadow others, thereby hindering the learning of weak modalities. Conventional approaches often force weak modalities to align with dominant ones in "Learning to be (the same)" (Positive Learning), which risks suppressing the unique information inherent in the weak modalities. To address this challenge, we offer a new learning paradigm: "Learning Not to be" (Negative Learning). Instead of enhancing weak modalities' target-class predictions, the dominant modalities dynamically guide the weak modality to suppress non-target classes. This stabilizes the decision space and preserves modality-specific information, allowing weak modalities to preserve unique information without being over-aligned. We proceed to reveal multimodal learning from a robustness perspective and theoretically derive the Multimodal Negative Learning (MNL) framework, which introduces a dynamic guidance mechanism tailored for negative learning. Our method provably tightens the robustness lower bound of multimodal learning by increasing the Unimodal Confidence Margin (UCoM) and reduces the empirical error of weak modalities, particularly under noisy and imbalanced scenarios. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generalizability of our approach against competing methods. The code will be available at this https URL. 

---
# CC-GRMAS: A Multi-Agent Graph Neural System for Spatiotemporal Landslide Risk Assessment in High Mountain Asia 

**Authors**: Mihir Panchal, Ying-Jung Chen, Surya Parkash  

**Link**: [PDF](https://arxiv.org/pdf/2510.20875)  

**Abstract**: Landslides are a growing climate induced hazard with severe environmental and human consequences, particularly in high mountain Asia. Despite increasing access to satellite and temporal datasets, timely detection and disaster response remain underdeveloped and fragmented. This work introduces CC-GRMAS, a framework leveraging a series of satellite observations and environmental signals to enhance the accuracy of landslide forecasting. The system is structured around three interlinked agents Prediction, Planning, and Execution, which collaboratively enable real time situational awareness, response planning, and intervention. By incorporating local environmental factors and operationalizing multi agent coordination, this approach offers a scalable and proactive solution for climate resilient disaster preparedness across vulnerable mountainous terrains. 

---
# Crisis-Resilient Portfolio Management via Graph-based Spatio-Temporal Learning 

**Authors**: Zan Li, Rui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.20868)  

**Abstract**: Financial time series forecasting faces a fundamental challenge: predicting optimal asset allocations requires understanding regime-dependent correlation structures that transform during crisis periods. Existing graph-based spatio-temporal learning approaches rely on predetermined graph topologies--correlation thresholds, sector classifications--that fail to adapt when market dynamics shift across different crisis mechanisms: credit contagion, pandemic shocks, or inflation-driven selloffs.
We present CRISP (Crisis-Resilient Investment through Spatio-temporal Patterns), a graph-based spatio-temporal learning framework that encodes spatial relationships via Graph Convolutional Networks and temporal dynamics via BiLSTM with self-attention, then learns sparse structures through multi-head Graph Attention Networks. Unlike fixed-topology methods, CRISP discovers which asset relationships matter through attention mechanisms, filtering 92.5% of connections as noise while preserving crisis-relevant dependencies for accurate regime-specific predictions.
Trained on 2005--2021 data encompassing credit and pandemic crises, CRISP demonstrates robust generalization to 2022--2024 inflation-driven markets--a fundamentally different regime--by accurately forecasting regime-appropriate correlation structures. This enables adaptive portfolio allocation that maintains profitability during downturns, achieving Sharpe ratio 3.76: 707% improvement over equal-weight baselines and 94% improvement over static graph methods. Learned attention weights provide interpretable regime detection, with defensive cluster attention strengthening 49% during crises versus 31% market-wide--emergent behavior from learning to forecast rather than imposing assumptions. 

---
# Incentivizing Consistent, Effective and Scalable Reasoning Capability in Audio LLMs via Reasoning Process Rewards 

**Authors**: Jiajun Fan, Roger Ren, Jingyuan Li, Rahul Pandey, Prashanth Gurunath Shivakumar, Ivan Bulyko, Ankur Gandhe, Ge Liu, Yile Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20867)  

**Abstract**: The role of reasoning in Audio Large Language Models remains widely underexplored, as introducing a reasoning process often degrades rather than improves performance during inference, a phenomenon we term test-time inverse scaling, where longer reasoning chains yield progressively worse results. We demonstrate that this stems not from fundamental limitations of reasoning itself, but from inadequate training: models without proper guidance for the reasoning process produce hallucinatory, inconsistent reasoning that accumulates errors over longer chains. To address these challenges, we introduce CESAR (Consistent, Effective, and Scalable Audio Reasoners), shifting from outcome verification to rewarding the reasoning process. Our online reinforcement learning framework employs Group Relative Policy Optimization with a multi-faceted reward suite that incentivizes not only correctness and format but also consistency, structured analytical patterns, causal reasoning, domain-knowledge integration, and calibrated reasoning depth. CESAR resolves test-time inverse scaling, transforming reasoning from detriments into gains while revealing model-specific ``reasoning sweet spots", where performance peaks during test-time scaling. We achieve state-of-the-art results on MMAU Test-mini, substantially outperforming Gemini 2.5 Pro and GPT-4o Audio, and near-human-level performance on MMSU reasoning tasks. Through AI-as-judge evaluations and qualitative comparisons, we provide both quantitative and qualitative validation of our improved reasoning quality. Importantly, enhanced reasoning creates synergistic effects, simultaneously improving multimodal reasoning and perception capabilities. Overall, CESAR establishes a principled method for developing robust and scalable reasoning in Audio LLMs. 

---
# Integrated representational signatures strengthen specificity in brains and models 

**Authors**: Jialin Wu, Shreya Saha, Yiqing Bo, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2510.20847)  

**Abstract**: The extent to which different neural or artificial neural networks (models) rely on equivalent representations to support similar tasks remains a central question in neuroscience and machine learning. Prior work has typically compared systems using a single representational similarity metric, yet each captures only one facet of representational structure. To address this, we leverage a suite of representational similarity metrics-each capturing a distinct facet of representational correspondence, such as geometry, unit-level tuning, or linear decodability-and assess brain region or model separability using multiple complementary measures. Metrics that preserve geometric or tuning structure (e.g., RSA, Soft Matching) yield stronger region-based discrimination, whereas more flexible mappings such as Linear Predictivity show weaker separation. These findings suggest that geometry and tuning encode brain-region- or model-family-specific signatures, while linearly decodable information tends to be more globally shared across regions or models. To integrate these complementary representational facets, we adapt Similarity Network Fusion (SNF), a framework originally developed for multi-omics data integration. SNF produces substantially sharper regional and model family-level separation than any single metric and yields robust composite similarity profiles. Moreover, clustering cortical regions using SNF-derived similarity scores reveals a clearer hierarchical organization that aligns closely with established anatomical and functional hierarchies of the visual cortex-surpassing the correspondence achieved by individual metrics. 

---
# This EEG Looks Like These EEGs: Interpretable Interictal Epileptiform Discharge Detection With ProtoEEG-kNN 

**Authors**: Dennis Tang, Jon Donnelly, Alina Jade Barnett, Lesia Semenova, Jin Jing, Peter Hadar, Ioannis Karakis, Olga Selioutski, Kehan Zhao, M. Brandon Westover, Cynthia Rudin  

**Link**: [PDF](https://arxiv.org/pdf/2510.20846)  

**Abstract**: The presence of interictal epileptiform discharges (IEDs) in electroencephalogram (EEG) recordings is a critical biomarker of epilepsy. Even trained neurologists find detecting IEDs difficult, leading many practitioners to turn to machine learning for help. While existing machine learning algorithms can achieve strong accuracy on this task, most models are uninterpretable and cannot justify their conclusions. Absent the ability to understand model reasoning, doctors cannot leverage their expertise to identify incorrect model predictions and intervene accordingly. To improve the human-model interaction, we introduce ProtoEEG-kNN, an inherently interpretable model that follows a simple case-based reasoning process. ProtoEEG-kNN reasons by comparing an EEG to similar EEGs from the training set and visually demonstrates its reasoning both in terms of IED morphology (shape) and spatial distribution (location). We show that ProtoEEG-kNN can achieve state-of-the-art accuracy in IED detection while providing explanations that experts prefer over existing approaches. 

---
# Consciousness, natural and artificial: an evolutionary advantage for reasoning on reactive substrates 

**Authors**: Warisa Sritriratanarak, Paulo Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.20839)  

**Abstract**: Precisely defining consciousness and identifying the mechanisms that effect it is a long-standing question, particularly relevant with advances in artificial intelligence. The scientific community is divided between physicalism and natural dualism. Physicalism posits consciousness is a physical process that can be modeled computationally; natural dualism rejects this hypothesis. Finding a computational model has proven elusive, particularly because of conflation of consciousness with other cognitive capabilities exhibited by humans, such as intelligence and physiological sensations. Here we show such a computational model that precisely models consciousness, natural or artificial, identifying the structural and functional mechanisms that effect it, confirming the physicalism hypothesis. We found such a model is obtainable when including the underlying (biological or digital) substrate and accounting for reactive behavior in substrate sub-systems (e.g., autonomous physiological responses). Results show that, unlike all other computational processes, consciousness is not independent of its substrate and possessing it is an evolutionary advantage for intelligent entities. Our result shows there is no impediment to the realization of fully artificial consciousness but, surprisingly, that it is also possible to realize artificial intelligence of arbitrary level without consciousness whatsoever, and that there is no advantage in imbuing artificial systems with consciousness. 

---
# Image and Point-cloud Classification for Jet Analysis in High-Energy Physics: A survey 

**Authors**: Hamza Kheddar, Yassine Himeur, Abbes Amira, Rachik Soualah  

**Link**: [PDF](https://arxiv.org/pdf/2403.11934)  

**Abstract**: Nowadays, there has been a growing trend in the field of high-energy physics (HEP), in both its experimental and phenomenological studies, to incorporate machine learning (ML) and its specialized branch, deep learning (DL). This review paper provides a thorough illustration of these applications using different ML and DL approaches. The first part of the paper examines the basics of various particle physics types and establishes guidelines for assessing particle physics alongside the available learning models. Next, a detailed classification is provided for representing Jets that are reconstructed in high-energy collisions, mainly in proton-proton collisions at well-defined beam energies. This section covers various datasets, preprocessing techniques, and feature extraction and selection methods. The presented techniques can be applied to future hadron-hadron colliders (HHC), such as the high-luminosity LHC (HL-LHC) and the future circular collider - hadron-hadron (FCChh). The authors then explore several AI techniques analyses designed specifically for both image and point-cloud (PC) data in HEP. Additionally, a closer look is taken at the classification associated with Jet tagging in hadron collisions. In this review, various state-of-the-art (SOTA) techniques in ML and DL are examined, with a focus on their implications for HEP demands. More precisely, this discussion addresses various applications in extensive detail, such as Jet tagging, Jet tracking, particle classification, and more. The review concludes with an analysis of the current state of HEP using DL methodologies. It highlights the challenges and potential areas for future research, which are illustrated for each application. 

---
