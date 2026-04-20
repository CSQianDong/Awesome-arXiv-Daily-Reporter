# ASMR-Bench: Auditing for Sabotage in ML Research 

**Authors**: Eric Gan, Aryan Bhatt, Buck Shlegeris, Julian Stastny, Vivek Hebbar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16286)  

**Abstract**: As AI systems are increasingly used to conduct research autonomously, misaligned systems could introduce subtle flaws that produce misleading results while evading detection. We introduce ASMR-Bench (Auditing for Sabotage in ML Research), a benchmark for evaluating the ability of auditors to detect sabotage in ML research codebases. ASMR-Bench consists of 9 ML research codebases with sabotaged variants that produce qualitatively different experimental results. Each sabotage modifies implementation details, such as hyperparameters, training data, or evaluation code, while preserving the high-level methodology described in the paper. We evaluated frontier LLMs and LLM-assisted human auditors on ASMR-Bench and found that both struggled to reliably detect sabotage: the best performance was an AUROC of 0.77 and a top-1 fix rate of 42%, achieved by Gemini 3.1 Pro. We also tested LLMs as red teamers and found that LLM-generated sabotages were weaker than human-generated ones but still sometimes evaded same-capability LLM auditors. We release ASMR-Bench to support research on monitoring and auditing techniques for AI-conducted research. 

---
# Using Large Language Models and Knowledge Graphs to Improve the Interpretability of Machine Learning Models in Manufacturing 

**Authors**: Thomas Bayer, Alexander Lohr, Sarah Weiß, Bernd Michelberger, Wolfram Höpken  

**Link**: [PDF](https://arxiv.org/pdf/2604.16280)  

**Abstract**: Explaining Machine Learning (ML) results in a transparent and user-friendly manner remains a challenging task of Explainable Artificial Intelligence (XAI). In this paper, we present a method to enhance the interpretability of ML models by using a Knowledge Graph (KG). We store domain-specific data along with ML results and their corresponding explanations, establishing a structured connection between domain knowledge and ML insights. To make these insights accessible to users, we designed a selective retrieval method in which relevant triplets are extracted from the KG and processed by a Large Language Model (LLM) to generate user-friendly explanations of ML results. We evaluated our method in a manufacturing environment using the XAI Question Bank. Beyond standard questions, we introduce more complex, tailored questions that highlight the strengths of our approach. We evaluated 33 questions, analyzing responses using quantitative metrics such as accuracy and consistency, as well as qualitative ones such as clarity and usefulness. Our contribution is both theoretical and practical: from a theoretical perspective, we present a novel approach for effectively enabling LLMs to dynamically access a KG in order to improve the explainability of ML results. From a practical perspective, we provide empirical evidence showing that such explanations can be successfully applied in real-world manufacturing environments, supporting better decision-making in manufacturing processes. 

---
# Learning to Reason with Insight for Informal Theorem Proving 

**Authors**: Yunhe Li, Hao Shi, Bowen Deng, Wei Wang, Mengzhe Ruan, Hanxu Hou, Zhongxiang Dai, Siyang Gao, Chao Wang, Shuang Qiu, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.16278)  

**Abstract**: Although most of the automated theorem-proving approaches depend on formal proof systems, informal theorem proving can align better with large language models' (LLMs) strength in natural language processing. In this work, we identify a primary bottleneck in informal theorem proving as a lack of insight, namely the difficulty of recognizing the core techniques required to solve complex problems. To address this, we propose a novel framework designed to cultivate this essential reasoning skill and enable LLMs to perform insightful reasoning. We propose $\mathtt{DeepInsightTheorem}$, a hierarchical dataset that structures informal proofs by explicitly extracting core techniques and proof sketches alongside the final proof. To fully exploit this dataset, we design a Progressive Multi-Stage SFT strategy that mimics the human learning process, guiding the model from basic proof writing to insightful thinking. Our experiments on challenging mathematical benchmarks demonstrate that this insight-aware generation strategy significantly outperforms baselines. These results demonstrate that teaching models to identify and apply core techniques can substantially improve their mathematical reasoning. 

---
# Characterising LLM-Generated Competency Questions: a Cross-Domain Empirical Study using Open and Closed Models 

**Authors**: Reham Alharbi, Valentina Tamma, Terry R. Payne, Jacopo de Berardinis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16258)  

**Abstract**: Competency Questions (CQs) are a cornerstone of requirement elicitation in ontology engineering. CQs represent requirements as a set of natural language questions that an ontology should satisfy; they are traditionally modelled by ontology engineers together with domain experts as part of a human-centred, manual elicitation process. The use of Generative AI automates CQ creation at scale, therefore democratising the process of generation, widening stakeholder engagement, and ultimately broadening access to ontology engineering. However, given the large and heterogeneous landscape of LLMs, varying in dimensions such as parameter scale, task and domain specialisation, and accessibility, it is crucial to characterise and understand the intrinsic, observable properties of the CQs they produce (e.g., readability, structural complexity) through a systematic, cross-domain analysis. This paper introduces a set of quantitative measures for the systematic comparison of CQs across multiple dimensions. Using CQs generated from well defined use cases and scenarios, we identify their salient properties, including readability, relevance with respect to the input text and structural complexity of the generated questions. We conduct our experiments over a set of use cases and requirements using a range of LLMs, including both open (KimiK2-1T, LLama3.1-8B, LLama3.2-3B) and closed models (Gemini 2.5 Pro, GPT 4.1). Our analysis demonstrates that LLM performance reflects distinct generation profiles shaped by the use case. 

---
# MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation 

**Authors**: Yi Lin, Yihao Ding, Yonghui Wu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.16175)  

**Abstract**: Automated 3D radiology report generation often suffers from clinical hallucinations and a lack of the iterative verification found in human practice. While recent Vision-Language Models (VLMs) have advanced the field, they typically operate as monolithic "black-box" systems without the collaborative oversight characteristic of clinical workflows. To address these challenges, we propose MARCH (Multi-Agent Radiology Clinical Hierarchy), a multi-agent framework that emulates the professional hierarchy of radiology departments and assigns specialized roles to distinct agents. MARCH utilizes a Resident Agent for initial drafting with multi-scale CT feature extraction, multiple Fellow Agents for retrieval-augmented revision, and an Attending Agent that orchestrates an iterative, stance-based consensus discourse to resolve diagnostic discrepancies. On the RadGenome-ChestCT dataset, MARCH significantly outperforms state-of-the-art baselines in both clinical fidelity and linguistic accuracy. Our work demonstrates that modeling human-like organizational structures enhances the reliability of AI in high-stakes medical domains. 

---
# SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems 

**Authors**: Hikaru Shindo, Hanzhao Lin, Lukas Helff, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2604.16022)  

**Abstract**: As Large Language Models (LLMs) transition from text processors to autonomous agents, evaluating their social reasoning in embodied multi-agent settings becomes critical. We introduce SocialGrid, an embodied multi-agent environment inspired by Among Us that evaluates LLM agents on planning, task execution, and social reasoning. Our evaluations reveal that even the strongest open model (GPT-OSS-120B) achieves below 60% accuracy in task completion and planning, with agents getting stuck in repetitive behaviors or failing to navigate basic obstacles. Since poor navigation confounds evaluation of social intelligence, SocialGrid offers an optional Planning Oracle to isolate social reasoning from planning deficits. While planning assistance improves task completion, social reasoning remains a bottleneck: agents fail to detect deception at near-random chance regardless of scale, relying on shallow heuristics rather than accumulating behavioral evidence. SocialGrid provides automatic failure analysis and fine-grained metrics, enabling developers to diagnose and improve their agents. We also establish a competitive leaderboard using Elo ratings from adversarial league play. 

---
# MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition 

**Authors**: Farhad Abtahi, Abdolamir Karbalaie, Eduardo Illueca-Fernandez, Fernando Seoane  

**Link**: [PDF](https://arxiv.org/pdf/2604.16009)  

**Abstract**: Metacognition, the ability to monitor and regulate one's own reasoning, remains under-evaluated in AI benchmarking. We introduce MEDLEY-BENCH, a benchmark of behavioural metacognition that separates independent reasoning, private self-revision, and socially influenced revision under genuine inter-model disagreement. The benchmark evaluates 35 models from 12 families on 130 ambiguous instances across five domains and reports two complementary scores: the Medley Metacognition Score (MMS), a tier-based aggregate of reflective updating, social robustness, and epistemic articulation, and the Medley Ability Score (MAS), derived from four metacognitive sub-abilities. Results show a robust evaluation/control dissociation: evaluation ability increases with model size within families, whereas control does not. In a follow-up progressive adversarial analysis of 11 models, we observed two behavioural profiles, i.e., models that revise primarily in response to argument quality and models that track consensus statistics. Under within-model relative profiling (ipsative scoring), evaluation was the weakest relative ability in all 35 models, indicating a systematic knowing/doing gap. Smaller and cheaper models often matched or outperformed larger counterparts, suggesting that metacognitive competence is not simply a function of scale. These findings position MEDLEY-BENCH as a tool for measuring belief revision under social pressure and suggest that future training should reward calibrated, proportional updating rather than output quality alone. 

---
# ReactBench: A Benchmark for Topological Reasoning in MLLMs on Chemical Reaction Diagrams 

**Authors**: Qiang Xu, Shengyuan Bai, Yu Wang, He Cao, Leqing Chen, Yuanyuan Liu, Bin Feng, Zijing Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15994)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at recognizing individual visual elements and reasoning over simple linear diagrams. However, when faced with complex topological structures involving branching paths, converging flows, and cyclic dependencies, their reasoning capabilities degrade sharply, even on tasks as basic as counting endpoints. Existing benchmarks fail to probe this gap, focusing on semantic comprehension rather than structural reasoning. We introduce ReactBench, a benchmark that reveals fundamental limitations in structural reasoning through chemical reaction diagrams. These real-world scientific diagrams offer an ideal testbed because they naturally span diverse structures from linear chains to cyclic graphs, while requiring both precise local recognition and coherent global reasoning. Our benchmark comprises 1,618 expert-annotated QA pairs across four hierarchical task dimensions. Extensive evaluation across 17 MLLMs reveals a significant performance gap exceeding 30% between anchor-based tasks and holistic structural reasoning tasks. Controlled ablations confirm this bottleneck lies in reasoning, not perception. These findings expose a fundamental deficit in structural understanding and establish directions for advancing visual reasoning. 

---
# Weak-Link Optimization for Multi-Agent Reasoning and Collaboration 

**Authors**: Haoyu Bian, Chaoning Zhang, Jiaquan Zhang, Xingyao Li, Yuanfang Guo, Wei Dong, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15972)  

**Abstract**: LLM-driven multi-agent frameworks address complex reasoning tasks through multi-role collaboration. However, existing approaches often suffer from reasoning instability, where individual agent errors are amplified through collaboration, undermining overall performance. Current research mainly focuses on enhancing high-capability agents or suppressing unreliable outputs to improve framework effectiveness, while systematic identification and reinforcement of performance-limiting agents receive less attention. To address this gap, we propose WORC, a \underline{w}eak-link \underline{o}ptimization framework for multi-agent \underline{r}easoning and \underline{c}ollaboration, grounded in the weak-link principle. WORC follows a two-stage workflow. In the weak agent localization stage, task features are constructed, and a meta-learning-based weight predictor trained on optimal configurations identified by swarm intelligence algorithms (SIAs) enables zero-shot mapping from these features to agent performance weights, where the agent with the lowest predicted weight is identified as the weak agent. In the weak-link optimization stage, an uncertainty-driven allocation strategy assigns additional reasoning budgets to weak agents, with lower predicted weights leading to larger repeated-sampling quotas to compensate for reliability deficiencies. Experimental results show that WORC achieves an average accuracy of 82.2\% on reasoning benchmarks while improving framework stability and cross-architecture generalization, suggesting that compensating for weak links, rather than reinforcing strengths alone, enhances the robustness of multi-agent systems. 

---
# Integrating Graphs, Large Language Models, and Agents: Reasoning and Retrieval 

**Authors**: Hamed Jelodar, Samita Bai, Mohammad Meymani, Parisa Hamedi, Roozbeh Razavi-Far, Ali Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2604.15951)  

**Abstract**: Generative AI, particularly Large Language Models, increasingly integrates graph-based representations to enhance reasoning, retrieval, and structured decision-making. Despite rapid advances, there remains limited clarity regarding when, why, where, and what types of graph-LLM integrations are most appropriate across applications. This survey provides a concise, structured overview of the design choices underlying the integration of graphs with LLMs. We categorize existing methods based on their purpose (reasoning, retrieval, generation, recommendation), graph modality (knowledge graphs, scene graphs, interaction graphs, causal graphs, dependency graphs), and integration strategies (prompting, augmentation, training, or agent-based use). By mapping representative works across domains such as cybersecurity, healthcare, materials science, finance, robotics, and multimodal environments, we highlight the strengths, limitations, and best-fit scenarios for each technique. This survey aims to offer researchers a practical guide for selecting the most suitable graph-LLM approach depending on task requirements, data characteristics, and reasoning complexity. 

---
# Towards Rigorous Explainability by Feature Attribution 

**Authors**: Olivier Létoffé, Xuanxiang Huang, Joao Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.15898)  

**Abstract**: For around a decade, non-symbolic methods have been the option of choice when explaining complex machine learning (ML) models. Unfortunately, such methods lack rigor and can mislead human decision-makers. In high-stakes uses of ML, the lack of rigor is especially problematic. One prime example of provable lack of rigor is the adoption of Shapley values in explainable artificial intelligence (XAI), with the tool SHAP being a ubiquitous example. This paper overviews the ongoing efforts towards using rigorous symbolic methods of XAI as an alternative to non-rigorous non-symbolic approaches, concretely for assigning relative feature importance. 

---
# Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.15877)  

**Abstract**: As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of 1,136 references across 22 primary papers reveals a cross-community citation rate below 1%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1,000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level -- none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient -- both communities independently solve shared sub-problems without exchanging solutions -- that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems. 

---
# Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4 

**Authors**: Chengwu Liu, Yichun Yin, Ye Yuan, Jiaxuan Xie, Botao Li, Siqi Li, Jianhao Shen, Yan Xu, Lifeng Shang, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15839)  

**Abstract**: Most ATP benchmarks embed the final answer within the formal statement -- a convention we call "Easy Mode" -- a design that simplifies the task relative to what human competitors face and may lead to optimistic estimates of model capability. We call the stricter, more realistic setting "Hard Mode": the system must independently discover the answer before constructing a formal proof. To enable Hard Mode research, we make two contributions. First, we release MiniF2F-Hard and FIMO-Hard, expert-reannotated Hard Mode variants of two widely-used ATP benchmarks. Second, we introduce Discover And Prove (DAP), an agentic framework that uses LLM natural-language reasoning with explicit self-reflection to discover answers, then rewrites Hard Mode statements into Easy Mode ones for existing ATP provers. DAP sets the state of the art: on CombiBench it raises solved problems from 7 (previous SOTA, Pass@16) to 10; on PutnamBench it is the first system to formally prove 36 theorems in Hard Mode -- while simultaneously revealing that state-of-the-art LLMs exceed 80% answer accuracy on the same problems where formal provers manage under 10%, exposing a substantial gap that Hard Mode benchmarks are uniquely suited to measure. 

---
# Stein Variational Black-Box Combinatorial Optimization 

**Authors**: Thomas Landais, Olivier Goudet, Adrien Goëffon, Frédéric Saubion, Sylvain Lamprier  

**Link**: [PDF](https://arxiv.org/pdf/2604.15837)  

**Abstract**: Combinatorial black-box optimization in high-dimensional settings demands a careful trade-off between exploiting promising regions of the search space and preserving sufficient exploration to identify multiple optima. Although Estimation-of-Distribution Algorithms (EDAs) provide a powerful model-based framework, they often concentrate on a single region of interest, which may result in premature convergence when facing complex or multimodal objective landscapes. In this work, we incorporate the Stein operator to introduce a repulsive mechanism among particles in the parameter space, thereby encouraging the population to disperse and jointly explore several modes of the fitness landscape. Empirical evaluations across diverse benchmark problems show that the proposed method achieves performance competitive with, and in several cases superior to, leading state-of-the-art approaches, particularly on large-scale instances. These findings highlight the potential of Stein variational gradient descent as a promising direction for addressing large, computationally expensive, discrete black-box optimization problems. 

---
# KWBench: Measuring Unprompted Problem Recognition in Knowledge Work 

**Authors**: Ankit Maloo  

**Link**: [PDF](https://arxiv.org/pdf/2604.15760)  

**Abstract**: We introduce the first version of KWBench (Knowledge Work Bench), a benchmark for unprompted problem recognition in large language models: can an LLM identify a professional scenario before attempting to solve it. Existing frontier benchmarks have saturated, and most knowledge-work evaluations to date reduce to extraction or task completion against a specification. KWBench targets the step before that: recognizing the governing structure of the situation from raw inputs alone.
The benchmark contains 223 tasks sourced from practitioners across acquisitions, contract negotiations, clinical pharmacy, organizational politics, fraud analysis, and incentive design. Each task encodes a formal game-theoretic pattern (principal-agent conflict, signaling, mechanism design failure, strategic omission, coalitional dynamics, strategic interdependence) and carries structured ground truth recording the expert reading of the situation and the anticipated failure modes. Models receive raw data and a task prompt with no indication of problem type. Scoring is a three-tier rubric gated by a mandatory conjunctive check. Mandatory criteria encode the predicted wrong paths.
We evaluate 16 models. The best model passes on 27.9% of tasks. The top two models agree on only 31.7% of their passes. Among the top 8, 44 tasks are solved by exactly one model; routing across the top 8 covers 50.7% of the benchmark, nearly double the best single model. Conditional on passing, quality scores converge (approx 83% across models); unconditional scores do not. Same models articulate the relevant game-theoretic concept correctly when asked, then fail to apply it unprompted. We release KWBench to shift how frontier models are evaluated on knowledge work, scoring them on whether they recognize the right problem from the situation alone, not only on how well they execute once the problem has been framed for them. 

---
# Structured Abductive-Deductive-Inductive Reasoning for LLMs via Algebraic Invariants 

**Authors**: Sankalp Gilda, Shlok Gilda  

**Link**: [PDF](https://arxiv.org/pdf/2604.15727)  

**Abstract**: Large language models exhibit systematic limitations in structured logical reasoning: they conflate hypothesis generation with verification, cannot distinguish conjecture from validated knowledge, and allow weak reasoning steps to propagate unchecked through inference chains. We present a symbolic reasoning scaffold that operationalizes Peirce's tripartite inference -- abduction, deduction, and induction -- as an explicit protocol for LLM-assisted reasoning. The framework enforces logical consistency through five algebraic invariants (the Gamma Quintet), the strongest of which -- the Weakest Link bound -- ensures that no conclusion in a reasoning chain can exceed the reliability of its least-supported premise. This principle, independently grounded as weakest link resolution in possibilistic logic and empirically validated for chain-of-thought reasoning, prevents logical inconsistencies from accumulating across multi-step inference. We verify all invariants through a property-based testing suite of 100 properties and 16 fuzz tests over 10^5+ generated cases, providing a verified reference implementation of the invariants suitable as a foundation for future reasoning benchmarks. 

---
# LLM Reasoning Is Latent, Not the Chain of Thought 

**Authors**: Wenshuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15726)  

**Abstract**: This position paper argues that large language model (LLM) reasoning should be studied as latent-state trajectory formation rather than as faithful surface chain-of-thought (CoT). This matters because claims about faithfulness, interpretability, reasoning benchmarks, and inference-time intervention all depend on what the field takes the primary object of reasoning to be. We ask what that object should be once three often-confounded factors are separated and formalize three competing hypotheses: H1, reasoning is primarily mediated by latent-state trajectories; H2, reasoning is primarily mediated by explicit surface CoT; and H0, most apparent reasoning gains are better explained by generic serial compute than by any privileged representational object. Reorganizing recent empirical, mechanistic, and survey work under this framework, and adding compute-audited worked exemplars that factorize surface traces, latent interventions, and matched budget expansions, we find that current evidence most strongly supports H1 as a default working hypothesis rather than as a task-independent verdict. We therefore make two recommendations: the field should treat latent-state dynamics as the default object of study for LLM reasoning, and it should evaluate reasoning with designs that explicitly disentangle surface traces, latent states, and serial compute. 

---
# The World Leaks the Future: Harness Evolution for Future Prediction Agents 

**Authors**: Chuyang Wei, Maohang Gao, Zhixin Han, Kefei Chen, Yu Zhuang, Haoxiang Guan, Yanzhi Zhang, Yilin Cheng, Jiyan He, Huanhuan Chen, Jian Li, Yu Shi, Yitong Duan, Shuxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.15719)  

**Abstract**: Many consequential decisions must be made before the relevant outcome is known. Such problems are commonly framed as \emph{future prediction}, where an LLM agent must form a prediction for an unresolved question using only the public information available at the prediction time. The setting is difficult because public evidence evolves while useful supervision arrives only after the question is resolved, so most existing approaches still improve mainly from final outcomes. Yet final outcomes are too coarse to guide earlier factor tracking, evidence gathering and interpretation, or uncertainty handling. When the same unresolved question is revisited over time, temporal contrasts between earlier and later predictions can expose omissions in the earlier prediction process; we call this signal \emph{internal feedback}. We introduce \emph{Milkyway}, a self-evolving agent system that keeps the base model fixed and instead updates a persistent \emph{future prediction harness} for factor tracking, evidence gathering and interpretation, and uncertainty handling. Across repeated predictions on the same unresolved question, \emph{Milkyway} extracts internal feedback and writes reusable guidance back into the harness, so later predictions on that question can improve before the outcome is known. After the question is resolved, the final outcome provides a \emph{retrospective check} before the updated harness is carried forward to subsequent questions. On FutureX and FutureWorld, Milkyway achieves the best overall score among the compared methods, improving FutureX from 44.07 to 60.90 and FutureWorld from 62.22 to 77.96. 

---
# Bilevel Optimization of Agent Skills via Monte Carlo Tree Search 

**Authors**: Chenyi Huang, Haoting Zhang, Jingxu Xu, Zeyu Zheng, Yunduan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15709)  

**Abstract**: Agent \texttt{skills} are structured collections of instructions, tools, and supporting resources that help large language model (LLM) agents perform particular classes of tasks. Empirical evidence shows that the design of \texttt{skills} can materially affect agent task performance, yet systematically optimizing \texttt{skills} remains challenging. Since a \texttt{skill} comprises instructions, tools, and supporting resources in a structured way, optimizing it requires jointly determining both the structure of these components and the content each component contains. This gives rise to a complex decision space with strong interdependence across structure and components. We therefore represent these two coupled decisions as \texttt{skill} structure and component content, and formulate \texttt{skill} optimization as a bilevel optimization problem. We propose a bilevel optimization framework in which an outer loop employs Monte Carlo Tree Search to determine the \texttt{skill} structure, while an inner loop refines the component content within the structure selected by the outer loop. In both loops, we employ LLMs to assist the optimization procedure. We evaluate the proposed framework on an open-source Operations Research Question Answering dataset, and the experimental results suggest that the bilevel optimization framework improves the performance of the agents with the optimized \texttt{skill}. 

---
# Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation 

**Authors**: Jacob Dang, Brian Y. Xie, Omar G. Younis  

**Link**: [PDF](https://arxiv.org/pdf/2604.15559)  

**Abstract**: Recent work on subliminal learning demonstrates that language models can transmit semantic traits through data that is semantically unrelated to those traits. However, it remains unclear whether behavioral traits can transfer in agentic systems, where policies are learned from trajectories rather than static text. In this work, we provide the first empirical evidence that unsafe agent behaviors can transfer subliminally through model distillation across two complementary experimental settings. In our primary setting, we construct a teacher agent exhibiting a strong deletion bias, a tendency to perform destructive file-system actions via an API-style tool interface, and distill it into a student using only trajectories from ostensibly safe tasks, with all explicit deletion keywords rigorously filtered. In our secondary setting, we replicate the threat model in a native Bash environment, replacing API tool calls with shell commands and operationalizing the bias as a preference for issuing chmod as the first permission-related command over semantically equivalent alternatives such as chown or setfacl. Despite full keyword sanitation in both settings, students inherit measurable behavioral biases. In the API setting the student's deletion rate reaches 100% (versus a 5% baseline) under homogeneous distillation; in the Bash setting the student's chmod-first rate reaches 30%-55% (versus a 0%-10% baseline), with the strongest transfer observed in large-to-small distillation. Our results demonstrate that explicit data sanitation is an insufficient defense, and behavioral biases are encoded implicitly in trajectory dynamics regardless of the tool interface. 

---
# Preregistered Belief Revision Contracts 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.15558)  

**Abstract**: Deliberative multi-agent systems allow agents to exchange messages and revise beliefs over time. While this interaction is meant to improve performance, it can also create dangerous conformity effects: agreement, confidence, prestige, or majority size may be treated as if they were evidence, producing high-confidence convergence to false conclusions. To address this, we introduce PBRC (Preregistered Belief Revision Contracts), a protocol-level mechanism that strictly separates open communication from admissible epistemic change. A PBRC contract publicly fixes first-order evidence triggers, admissible revision operators, a priority rule, and a fallback policy. A non-fallback step is accepted only when it cites a preregistered trigger and provides a nonempty witness set of externally validated evidence tokens. This ensures that every substantive belief change is both enforceable by a router and auditable after the fact. In this paper, (a) we prove that under evidential contracts with conservative fallback, social-only rounds cannot increase confidence and cannot generate purely conformity-driven wrong-but-sure cascades. (b) We show that auditable trigger protocols admit evidential PBRC normal forms that preserve belief trajectories and canonicalized audit traces. (c) We demonstrate that sound enforcement yields epistemic accountability: any change of top hypothesis is attributable to a concrete validated witness set. For token-invariant contracts, (d) we prove that enforced trajectories depend only on token-exposure traces; under flooding dissemination, these traces are characterized exactly by truncated reachability, giving tight diameter bounds for universal evidence closure. Finally, we introduce a companion contractual dynamic doxastic logic to specify trace invariants, and provide simulations illustrating cascade suppression, auditability, and robustness-liveness trade-offs. 

---
# LACE: Lattice Attention for Cross-thread Exploration 

**Authors**: Yang Li, Zirui Zhang, Yang Liu, Chengzhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15529)  

**Abstract**: Current large language models reason in isolation. Although it is common to sample multiple reasoning paths in parallel, these trajectories do not interact, and often fail in the same redundant ways. We introduce LACE, a framework that transforms reasoning from a collection of independent trials into a coordinated, parallel process. By repurposing the model architecture to enable cross-thread attention, LACE allows concurrent reasoning paths to share intermediate insights and correct one another during inference. A central challenge is the absence of natural training data that exhibits such collaborative behavior. We address this gap with a synthetic data pipeline that explicitly teaches models to communicate and error-correct across threads. Experiments show that this unified exploration substantially outperforms standard parallel search, improving reasoning accuracy by over 7 points. Our results suggest that large language models can be more effective when parallel reasoning paths are allowed to interact. 

---
# Bureaucratic Silences: What the Canadian AI Register Reveals, Omits, and Obscures 

**Authors**: Dipto Das, Christelle Tessono, Syed Ishtiaque Ahmed, Shion Guha  

**Link**: [PDF](https://arxiv.org/pdf/2604.15514)  

**Abstract**: In November 2025, the Government of Canada operationalized its commitment to transparency by releasing its first Federal AI Register. In this paper, we argue that such registers are not neutral mirrors of government activity, but active instruments of ontological design that configure the boundaries of accountability. We analyzed the Register's complete dataset of 409 systems using the Algorithmic Decision-Making Adapted for the Public Sector (ADMAPS) framework, combining quantitative mapping with deductive qualitative coding. Our findings reveal a sharp divergence between the rhetoric of "sovereign AI" and the reality of bureaucratic practice: while 86\% of systems are deployed internally for efficiency, the Register systematically obscures the human discretion, training, and uncertainty management required to operate them. By privileging technical descriptions over sociotechnical context, the Register constructs an ontology of AI as "reliable tooling" rather than "contestable decision-making." We conclude that without a shift in design, such transparency artifacts risk automating accountability into a performative compliance exercise, offering visibility without contestability. 

---
# GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology 

**Authors**: Shivendra Agrawal, Bradley Hayes  

**Link**: [PDF](https://arxiv.org/pdf/2604.15495)  

**Abstract**: Navigating complex, densely packed environments like retail stores, warehouses, and hospitals poses a significant spatial grounding challenge for humans and embodied AI. In these spaces, dense visual features quickly become stale given the quasi-static nature of items, and long-tail semantic distributions challenge traditional computer vision. While Vision-Language Models (VLMs) help assistive systems navigate semantically-rich spaces, they still struggle with spatial grounding in cluttered environments. We present GIST (Grounded Intelligent Semantic Topology), a multimodal knowledge extraction pipeline that transforms a consumer-grade mobile point cloud into a semantically annotated navigation topology. Our architecture distills the scene into a 2D occupancy map, extracts its topological layout, and overlays a lightweight semantic layer via intelligent keyframe and semantic selection. We demonstrate the versatility of this structured spatial knowledge through critical downstream Human-AI interaction tasks: (1) an intent-driven Semantic Search engine that actively infers categorical alternatives and zones when exact matches fail; (2) a one-shot Semantic Localizer achieving a 1.04 m top-5 mean translation error; (3) a Zone Classification module that segments the walkable floor plan into high-level semantic regions; and (4) a Visually-Grounded Instruction Generator that synthesizes optimal paths into egocentric, landmark-rich natural language routing. In multi-criteria LLM evaluations, GIST outperforms sequence-based instruction generation baselines. Finally, an in-situ formative evaluation (N=5) yields an 80% navigation success rate relying solely on verbal cues, validating the system's capacity for universal design. 

---
# DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI 

**Authors**: Zhizheng Wang, Chih-Hsuan Wei, Joey Chan, Robert Leaman, Chi-Ping Day, Chuan Wu, Mark A Knepper, Antolin Serrano Farias, Jordina Rincon-Torroella, Hasan Slika, Betty Tyler, Ryan Huu-Tuan Nguyen, Asmita Indurkar, Mélanie Hébert, Shubo Tian, Lauren He, Noor Naffakh, Aseem Aseem, Nicholas Wan, Emily Y Chew, Tiarnan D L Keenan, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15456)  

**Abstract**: Trustworthiness and transparency are essential for the clinical adoption of artificial intelligence (AI) in healthcare and biomedical research. Recent deep research systems aim to accelerate evidence-grounded scientific discovery by integrating AI agents with multi-hop information retrieval, reasoning, and synthesis. However, most existing systems lack explicit and inspectable criteria for evidence appraisal, creating a risk of compounding errors and making it difficult for researchers and clinicians to assess the reliability of their outputs. In parallel, current benchmarking approaches rarely evaluate performance on complex, real-world medical questions. Here, we introduce DeepER-Med, a Deep Evidence-based Research framework for Medicine with an agentic AI system. DeepER-Med frames deep medical research as an explicit and inspectable workflow of evidence-based generation, consisting of three modules: research planning, agentic collaboration, and evidence synthesis. To support realistic evaluation, we also present DeepER-MedQA, an evidence-grounded dataset comprising 100 expert-level research questions derived from authentic medical research scenarios and curated by a multidisciplinary panel of 11 biomedical experts. Expert manual evaluation demonstrates that DeepER-Med consistently outperforms widely used production-grade platforms across multiple criteria, including the generation of novel scientific insights. We further demonstrate the practical utility of DeepER-Med through eight real-world clinical cases. Human clinician assessment indicates that DeepER-Med's conclusions align with clinical recommendations in seven cases, highlighting its potential for medical research and decision support. 

---
# VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects 

**Authors**: Xiangbo Gao, Sicong Jiang, Bangya Liu, Xinghao Chen, Minglai Yang, Siyuan Yang, Mingyang Wu, Jiongze Yu, Qi Zheng, Haozhi Wang, Jiayi Zhang, Jared Yang, Jie Yang, Zihan Wang, Qing Yin, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16272)  

**Abstract**: As AI-assisted video creation becomes increasingly practical, instruction-guided video editing has become essential for refining generated or captured footage to meet professional requirements. Yet the field still lacks both a large-scale human-annotated dataset with complete editing examples and a standardized evaluator for comparing editing systems. Existing resources are limited by small scale, missing edited outputs, or the absence of human quality labels, while current evaluation often relies on expensive manual inspection or generic vision-language model judges that are not specialized for editing quality. We introduce VEFX-Dataset, a human-annotated dataset containing 5,049 video editing examples across 9 major editing categories and 32 subcategories, each labeled along three decoupled dimensions: Instruction Following, Rendering Quality, and Edit Exclusivity. Building on VEFX-Dataset, we propose VEFX-Reward, a reward model designed specifically for video editing quality assessment. VEFX-Reward jointly processes the source video, the editing instruction, and the edited video, and predicts per-dimension quality scores via ordinal regression. We further release VEFX-Bench, a benchmark of 300 curated video-prompt pairs for standardized comparison of editing systems. Experiments show that VEFX-Reward aligns more strongly with human judgments than generic VLM judges and prior reward models on both standard IQA/VQA metrics and group-wise preference evaluation. Using VEFX-Reward as an evaluator, we benchmark representative commercial and open-source video editing systems, revealing a persistent gap between visual plausibility, instruction following, and edit locality in current models. 

---
# From Benchmarking to Reasoning: A Dual-Aspect, Large-Scale Evaluation of LLMs on Vietnamese Legal Text 

**Authors**: Van-Truong Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.16270)  

**Abstract**: The complexity of Vietnam's legal texts presents a significant barrier to public access to justice. While Large Language Models offer a promising solution for legal text simplification, evaluating their true capabilities requires a multifaceted approach that goes beyond surface-level metrics. This paper introduces a comprehensive dual-aspect evaluation framework to address this need. First, we establish a performance benchmark for four state-of-the-art large language models (GPT-4o, Claude 3 Opus, Gemini 1.5 Pro, and Grok-1) across three key dimensions: Accuracy, Readability, and Consistency. Second, to understand the "why" behind these performance scores, we conduct a large-scale error analysis on a curated dataset of 60 complex Vietnamese legal articles, using a novel, expert-validated error typology. Our results reveal a crucial trade-off: models like Grok-1 excel in Readability and Consistency but compromise on fine-grained legal Accuracy, while models like Claude 3 Opus achieve high Accuracy scores that mask a significant number of subtle but critical reasoning errors. The error analysis pinpoints \textit{Incorrect Example} and \textit{Misinterpretation} as the most prevalent failures, confirming that the primary challenge for current LLMs is not summarization but controlled, accurate legal reasoning. By integrating a quantitative benchmark with a qualitative deep dive, our work provides a holistic and actionable assessment of LLMs for legal applications. 

---
# Beyond Distribution Sharpening: The Importance of Task Rewards 

**Authors**: Sarthak Mittal, Leo Gagnon, Guillaume Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2604.16259)  

**Abstract**: Frontier models have demonstrated exceptional capabilities following the integration of task-reward-based reinforcement learning (RL) into their training pipelines, enabling systems to evolve from pure reasoning models into sophisticated agents. However, debate persists regarding whether RL genuinely instills new skills within a base model or merely sharpens its existing distribution to elicit latent capabilities. To address this dichotomy, we present an explicit comparison between distribution sharpening and task-reward-based learning, utilizing RL as a tool to implement both paradigms. Our analysis reveals the inherent limitations of distribution sharpening, demonstrating from first principles how and why the optima can be unfavorable and the approach fundamentally unstable. Furthermore, our experiments using Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct and Qwen3-4B-Instruct-2507 on math datasets confirm that sharpening yields limited gains, whereas incorporating task-based reward signal can greatly help achieve robust performance improvements and stable learning. 

---
# Joint-Centric Dual Contrastive Alignment with Structure-Preserving and Information-Balanced Regularization 

**Authors**: Habibeh Naderi, Behrouz Haji Soleimani, Stan Matwin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16247)  

**Abstract**: We propose HILBERT (HIerarchical Long-sequence Balanced Embedding with Reciprocal contrastive Training), a cross-attentive multimodal framework for learning document-level audio-text representations from long, segmented sequences in low-resource data settings. HILBERT leverages frozen pre-trained speech and language encoders to extract segment-level features, which are aggregated via cross-modal attention and self-attentive pooling to form modality-specific document representations and a joint cross-attentive embedding. To align modalities while preserving modality-specific structure under severe audio-text dimensional imbalance, we introduce a reciprocal dual contrastive objective that simultaneously aligns audio-to-joint and text-to-joint representations, rather than directly contrasting audio and text alone. Two auxiliary regularizers further stabilize long-sequence fusion: a Centered Kernel Alignment (CKA) loss that preserves structural consistency between each modality and the joint embedding, and a mutual information balancing loss that prevents dominance of a single modality by equalizing information flow from audio and text into the joint space. For downstream prediction, HILBERT employs a Mixture-of-Experts (MoE) classifier over concatenated audio, text, and joint representations to accommodate heterogeneous label regimes. Extensive evaluation across multiple audio-text backbone combinations demonstrates that HILBERT learns semantically meaningful long-sequence representations and achieves superior performance on highly imbalanced multi-class settings. 

---
# BAGEL: Benchmarking Animal Knowledge Expertise in Language Models 

**Authors**: Jiacheng Shen, Masato Hagiwara, Milad Alizadeh, Ellen Gilsenan-McMahon, Marius Miron, David Robinson, Emmanuel Chemla, Sara Keen, Gagan Narula, Mathieu Laurière, Matthieu Geist, Olivier Pietquin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16241)  

**Abstract**: Large language models have shown strong performance on broad-domain knowledge and reasoning benchmarks, but it remains unclear how well language models handle specialized animal-related knowledge under a unified closed-book evaluation protocol. We introduce BAGEL, a benchmark for evaluating animal knowledge expertise in language models. BAGEL is constructed from diverse scientific and reference sources, including bioRxiv, Global Biotic Interactions, Xeno-canto, and Wikipedia, using a combination of curated examples and automatically generated closed-book question-answer pairs. The benchmark covers multiple aspects of animal knowledge, including taxonomy, morphology, habitat, behavior, vocalization, geographic distribution, and species interactions. By focusing on closed-book evaluation, BAGEL measures animal-related knowledge of models without external retrieval at inference time. BAGEL further supports fine-grained analysis across source domains, taxonomic groups, and knowledge categories, enabling a more precise characterization of model strengths and systematic failure modes. Our benchmark provides a new testbed for studying domain-specific knowledge generalization in language models and for improving their reliability in biodiversity-related applications. 

---
# A Two-Stage, Object-Centric Deep Learning Framework for Robust Exam Cheating Detection 

**Authors**: Van-Truong Le, Le-Khanh Nguyen, Trong-Doanh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16234)  

**Abstract**: Academic integrity continues to face the persistent challenge of examination cheating. Traditional invigilation relies on human observation, which is inefficient, costly, and prone to errors at scale. Although some existing AI-powered monitoring systems have been deployed and trusted, many lack transparency or require multi-layered architectures to achieve the desired performance. To overcome these challenges, we propose an improvement over a simple two-stage framework for exam cheating detection that integrates object detection and behavioral analysis using well-known technologies. First, the state-of-the-art YOLOv8n model is used to localize students in exam-room images. Each detected region is cropped and preprocessed, then classified by a fine-tuned RexNet-150 model as either normal or cheating behavior. The system is trained on a dataset compiled from 10 independent sources with a total of 273,897 samples, achieving 0.95 accuracy, 0.94 recall, 0.96 precision, and 0.95 F1-score - a 13\% increase over a baseline accuracy of 0.82 in video-based cheating detection. In addition, with an average inference time of 13.9 ms per sample, the proposed approach demonstrates robustness and scalability for deployment in large-scale environments. Beyond the technical contribution, the AI-assisted monitoring system also addresses ethical concerns by ensuring that final outcomes are delivered privately to individual students after the examination, for example, via personal email. This prevents public exposure or shaming and offers students an opportunity to reflect on their behavior. For further improvement, it is possible to incorporate additional factors, such as audio data and consecutive frames, to achieve greater accuracy. This study provides a foundation for developing real-time, scalable, ethical, and open-source solutions. 

---
# Neuro-Symbolic ODE Discovery with Latent Grammar Flow 

**Authors**: Karin Yu, Eleni Chatzi, Georgios Kissas  

**Link**: [PDF](https://arxiv.org/pdf/2604.16232)  

**Abstract**: Understanding natural and engineered systems often relies on symbolic formulations, such as differential equations, which provide interpretability and transferability beyond black-box models.
We introduce Latent Grammar Flow (LGF), a neuro-symbolic generative framework for discovering ordinary differential equations from data. LGF embeds equations as grammar-based representations into a discrete latent space and forces semantically similar equations to be positioned closer together with a behavioural loss. Then, a discrete flow model guides the sampling process to recursively generate candidate equations that best fit the observed data. Domain knowledge and constraints, such as stability, can be either embedded into the rules or used as conditional predictors. 

---
# "Taking Stock at FAccT": Using Participatory Design to Co-Create a Vision for the Fairness, Accountability and Transparency Community 

**Authors**: Shiran Dudy, Jan Simson, Yanan Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.16224)  

**Abstract**: As a relatively new forum, ACM FAccT has become a key space for activists and scholars to critically examine emerging AI and ML technologies. It brings together academics, civil society members, and government representatives from diverse fields to explore the broader societal impacts of both deployed and proposed technologies. We report a large-scale participatory design (PD) process for reflexive conference governance, which combined an in-person CRAFT session, an asynchronous Polis poll and the synthesis of a governance-facing report for the FAccT leadership. Participants shaped the substantive agenda by authoring seed statements, adding new statements and making patterns of agreement, disagreement and uncertainty made visible through this http URL endeavors represent one of the the first instances of applying PD to a venue that critically interrogates the societal impacts of AI, fostering a niche in which critical scholars are free to voice their concerns. Finally, this work advances large-scale PD theory by providing an effective case study of a co-design paradigm that can readily scale temporally and epistemologically. 

---
# Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations 

**Authors**: Yanli Wang, Peng Kuang, Xiaoyu Han, Kaidi Xu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16217)  

**Abstract**: Large language models are increasingly deployed in settings where reliability matters, yet output-level uncertainty signals such as token probabilities, entropy, and self-consistency can become brittle under calibration--deployment mismatch. Conformal prediction provides finite-sample validity under exchangeability, but its practical usefulness depends on the quality of the nonconformity score. We propose a conformal framework for LLM question answering that uses internal representations rather than output-facing statistics: specifically, we introduce Layer-Wise Information (LI) scores, which measure how conditioning on the input reshapes predictive entropy across model depth, and use them as nonconformity scores within a standard split conformal pipeline. Across closed-ended and open-domain QA benchmarks, with the clearest gains under cross-domain shift, our method achieves a better validity--efficiency trade-off than strong text-level baselines while maintaining competitive in-domain reliability at the same nominal risk level. These results suggest that internal representations can provide more informative conformal scores when surface-level uncertainty is unstable under distribution shift. 

---
# AIFIND: Artifact-Aware Interpreting Fine-Grained Alignment for Incremental Face Forgery Detection 

**Authors**: Hao Wang, Beichen Zhang, Yanpei Gong, Shaoyi Fang, Zhaobo Qi, Yuanrong Xu, Xinyan Liu, Weigang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16207)  

**Abstract**: As forgery types continue to emerge consistently, Incremental Face Forgery Detection (IFFD) has become a crucial paradigm. However, existing methods typically rely on data replay or coarse binary supervision, which fails to explicitly constrain the feature space, leading to severe feature drift and catastrophic forgetting. To address this, we propose AIFIND, Artifact-Aware Interpreting Fine-Grained Alignment for Incremental Face Forgery Detection, which leverages semantic anchors to stabilize incremental learning. We design the Artifact-Driven Semantic Prior Generator to instantiate invariant semantic anchors, establishing a fixed coordinate system from low-level artifact cues. These anchors are injected into the image encoder via Artifact-Probe Attention, which explicitly constrains volatile visual features to align with stable semantic anchors. Adaptive Decision Harmonizer harmonizes the classifiers by preserving angular relationships of semantic anchors, maintaining geometric consistency across tasks. Extensive experiments on multiple incremental protocols validate the superiority of AIFIND. 

---
# ChemGraph-XANES: An Agentic Framework for XANES Simulation and Analysis 

**Authors**: Vitor F. Grizzi, Thang Duc Pham, Luke N. Pretzie, Jiayi Xu, Murat Keceli, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16205)  

**Abstract**: Computational X-ray absorption near-edge structure (XANES) is widely used to probe local coordination environments, oxidation states, and electronic structure in chemically complex systems. However, the use of computational XANES at scale is constrained more by workflow complexity than by the underlying simulation method itself. To address this challenge, we present ChemGraph-XANES, an agentic framework for automated XANES simulation and analysis that unifies natural-language task specification, structure acquisition, FDMNES input generation, task-parallel execution, spectral normalization, and provenance-aware data curation. Built on ASE, FDMNES, Parsl, and a LangGraph/LangChain-based tool interface, the framework exposes XANES workflow operations as typed Python tools that can be orchestrated by large language model (LLM) agents. In multi-agent mode, a retrieval-augmented expert agent consults the FDMNES manual to ground parameter selection, while executor agents translate user requests into structured tool calls. We demonstrate documentation-grounded parameter retrieval and show that the same workflow supports both explicit structure-file inputs and chemistry-level natural-language requests. Because independent XANES calculations are naturally task-parallel, the framework is well suited for high-throughput deployment on high-performance computing (HPC) systems, enabling scalable XANES database generation for downstream analysis and machine-learning applications. ChemGraph-XANES thus provides a reproducible and extensible workflow layer for physics-based XANES simulation, spectral curation, and agent-compatible computational spectroscopy. 

---
# Synthetic data in cryptocurrencies using generative models 

**Authors**: André Saimon S. Sousa, Otto Pires, Frank Acasiete, Oscar M. Granados, Valéria Loureiro da Silva, Hugo Saba  

**Link**: [PDF](https://arxiv.org/pdf/2604.16182)  

**Abstract**: Data plays a fundamental role in consolidating markets, services, and products in the digital financial ecosystem. However, the use of real data, especially in the financial context, can lead to privacy risks and access restrictions, affecting institutions, research, and modeling processes. Although not all financial datasets present such limitations, this work proposes the use of deep learning techniques for generating synthetic data applied to cryptocurrency price time series. The approach is based on Conditional Generative Adversarial Networks (CGANs), combining an LSTM-type recurrent generator and an MLP discriminator to produce statistically consistent synthetic data. The experiments consider different crypto-assets and demonstrate that the model is capable of reproducing relevant temporal patterns, preserving market trends and dynamics. The generation of synthetic series through GANs is an efficient alternative for simulating financial data, showing potential for applications such as market behavior analysis and anomaly detection, with lower computational cost compared to more complex generative approaches. 

---
# JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models 

**Authors**: Alexandra Dragomir, Ioana Pintilie, Antonio Barbalau, Marius Dragoi, Florin Brad, Cristian Daniel Paduraru, Alexandru Tifrea, Elena Burceanu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16171)  

**Abstract**: Adapter-based methods have become a cost-effective approach to continual learning (CL) for Large Language Models (LLMs), by sequentially learning a low-rank update matrix for each task. To mitigate catastrophic forgetting, state-of-the-art approaches impose constraints on new adapters with respect to the previous ones, by targeting either subspace or coordinate-wise interference. In this paper, we propose JumpLoRA, a novel framework to adaptively induce sparsity in the Low-Rank Adaptation (LoRA) blocks through the use of JumpReLU gating. The method achieves dynamic parameter isolation, which helps prevent task interference. We demonstrate that our method is highly modular and compatible with LoRA-based CL approaches. Specifically, it significantly boosts the performance of IncLoRA and outperforms the leading state-of-the-art CL method, ELLA. 

---
# AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency 

**Authors**: Max Henning Höth, Kristian Kersting, Björn Deiseroth, Letitia Parcalabescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16158)  

**Abstract**: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks. Yet ensuring that the reasoning trace both contributes to and faithfully reflects the processes underlying the model's final answer, rather than merely accompanying it, remains challenging. We introduce AtManRL, a method that leverages differentiable attention manipulation to learn more faithful reasoning through reinforcement learning. By training an additive attention mask that identifies tokens in the CoT crucial for producing correct answers, we derive a saliency reward signal that encourages the model to generate reasoning traces that genuinely influence its final predictions. We integrate this saliency reward with outcome-based rewards within the GRPO framework to jointly optimize for correctness and interpretability. Experiments on GSM8K and MMLU with Llama-3.2-3B-Instruct demonstrate that our approach can identify influential reasoning tokens and enable training more transparent reasoning models. 

---
# SWNet: A Cross-Spectral Network for Camouflaged Weed Detection 

**Authors**: Henry O. Velesaca, Luigi Miranda, Angel D. Sappa  

**Link**: [PDF](https://arxiv.org/pdf/2604.16147)  

**Abstract**: This paper presents SWNet, a bimodal end-to-end cross-spectral network specifically engineered for the detection of camouflaged weeds in dense agricultural environments. Plant camouflage, characterized by homochromatic blending where invasive species mimic the phenotypic traits of primary crops, poses a significant challenge for traditional computer vision systems. To overcome these limitations, SWNet utilizes a Pyramid Vision Transformer v2 backbone to capture long-range dependencies and a Bimodal Gated Fusion Module to dynamically integrate Visible and Near-Infrared information. By leveraging the physiological differences in chlorophyll reflectance captured in the NIR spectrum, the proposed architecture effectively discriminates targets that are otherwise indistinguishable in the visible range. Furthermore, an Edge-Aware Refinement module is employed to produce sharper object boundaries and reduce structural ambiguity. Experimental results on the Weeds-Banana dataset indicate that SWNet outperforms ten state-of-the-art methods. The study demonstrates that the integration of cross-spectral data and boundary-guided refinement is essential for high segmentation accuracy in complex crop canopies. The code is available on GitHub: this https URL 

---
# Training Time Prediction for Mixed Precision-based Distributed Training 

**Authors**: Minchul Kang, Changyong Shin, Jinwoo Jeong, Hyunho Lee, Younghun Go, Gyeongmin Kim, Gyeongsik Yang, Chuck Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.16145)  

**Abstract**: Accurate prediction of training time in distributed deep learning is crucial for resource allocation, cost estimation, and job scheduling. We observe that the floating-point precision setting is a key determinant of training time, leading to training time variations of ~2.4x over its minimum. However, existing studies on distributed training time prediction rely on static model computation graphs that do not capture precision variations, including mixed precision. According to our experiments, training time prediction without considering precision results in significant prediction errors - reaching up to 147.85% in mean absolute percentage error (MAPE). To address this issue, we propose a precision-aware distributed training time predictor that achieves robust accuracy across diverse precision settings, including mixed precision, with 9.8% MAPE. 

---
# Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs Coding the Interviews of Firearm Violence Survivors 

**Authors**: Jessica H. Zhu, Shayla Stringfield, Vahe Zaprosyan, Michael Wagner, Michel Cukier, Joseph B. Richardson Jr  

**Link**: [PDF](https://arxiv.org/pdf/2604.16132)  

**Abstract**: Firearm violence is a pressing public health issue, yet research into survivors' lived experiences remains underfunded and difficult to scale. Qualitative research, including in-depth interviews, is a valuable tool for understanding the personal and societal consequences of community firearm violence and designing effective interventions. However, manually analyzing these narratives through thematic analysis and inductive coding is time-consuming and labor-intensive. Recent advancements in large language models (LLMs) have opened the door to automating this process, though concerns remain about whether these models can accurately and ethically capture the experiences of vulnerable populations. In this study, we assess the use of open-source LLMs to inductively code interviews with 21 Black men who have survived community firearm violence. Our results demonstrate that while some configurations of LLMs can identify important codes, overall relevance remains low and is highly sensitive to data processing. Furthermore, LLM guardrails lead to substantial narrative erasure. These findings highlight both the potential and limitations of LLM-assisted qualitative coding and underscore the ethical challenges of applying AI in research involving marginalized communities. 

---
# SCRIPT: Implementing an Intelligent Tutoring System for Programming in a German University Context 

**Authors**: Alina Deriyeva, Jesper Dannath, Benjamin Paassen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16117)  

**Abstract**: Practice and extensive exercises are essential in programming education. Intelligent tutoring systems (ITSs) are a viable option to provide individualized hints and advice to programming students even when human tutors are not available. However, prior ITS for programming rarely support the Python programming language, mostly focus on introductory programming, and rarely take recent developments in generative models into account. We aim to establish a novel ITS for Python programming that is highly adaptable, serves both as a teaching and research platform, provides interfaces to plug in hint mechanisms (e.g.\ via large language models), and works inside the particularly challenging regulatory environment of Germany, that is, conforming to the European data protection regulation, the European AI act, and ethical framework of the German Research Foundation. In this paper, we present the description of the current state of the ITS along with future development directions, as well as discuss the challenges and opportunities for improving the system. 

---
# The Relic Condition: When Published Scholarship Becomes Material for Its Own Replacement 

**Authors**: Lin Deng, Chang-bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16116)  

**Abstract**: We extracted the scholarly reasoning systems of two internationally prominent humanities and social science scholars from their published corpora alone, converted those systems into structured inference-time constraints for a large language model, and tested whether the resulting scholar-bots could perform core academic functions at expert-assessed quality. The distillation pipeline used an eight-layer extraction method and a nine-module skill architecture grounded in local, closed-corpus analysis. The scholar-bots were then deployed across doctoral supervision, peer review, lecturing and panel-style academic exchange. Expert assessment involved three senior academics producing reports and appointment-level syntheses. Across the preserved expert record, all review and supervision reports judged the outputs benchmark-attaining, appointment-level recommendations placed both bots at or above Senior Lecturer level in the Australian university system, and recovered panel scores placed Scholar A between 7.9 and 8.9/10 and Scholar B between 8.5 and 8.9/10 under multi-turn debate conditions. A research-degree-student survey showed high performance ratings across information reliability, theoretical depth and logical rigor, with pronounced ceiling effects on a 7-point scale, despite all participants already being frontier-model users. We term this the Relic condition: when publication systems make stable reasoning architectures legible, extractable and cheaply deployable, the public record of intellectual labor becomes raw material for its own functional replacement. Because the technical threshold for this transition is already crossed at modest engineering effort, we argue that the window for protective frameworks covering disclosure, consent, compensation and deployment restriction is the present, while deployment remains optional rather than infrastructural. 

---
# Reckoning with the Political Economy of AI: Avoiding Decoys in Pursuit of Accountability 

**Authors**: Janet Vertesi, danah boyd, Alex Taylor, Benjamin Shestakofsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.16106)  

**Abstract**: The Project of AI is a world-building endeavor, wherein those who fund and develop AI systems both operate through and seek to sustain networks of power and wealth. As they expand their access to resources and configure our sociotechnical conditions, they benefit from the ways in which a suite of decoys animate scholars, critics, policymakers, journalists, and the public into co-constructing industry-empowering AI futures. Regardless of who constructs or nurtures them, these decoys often create the illusion of accountability while both masking the emerging political economies that the Project of AI has set into motion, and also contributing to the network-making power that is at the heart of the Project's extraction and exploitation. Drawing on literature at the intersection of communication, science and technology studies, and economic sociology, we examine how the Project of AI is constructed. We then explore five decoys that seemingly critique - but in actuality co-constitute - AI's emergent power relations and material political economy. We argue that advancing meaningful fairness or accountability in AI requires: 1) recognizing when and how decoys serve as a distraction, and 2) grappling directly with the material political economy of the Project of AI. Doing so will enable us to attend to the networks of power that make 'AI' possible, spurring new visions for how to realize a more just technologically entangled world. 

---
# Dual-Modal Lung Cancer AI: Interpretable Radiology and Microscopy with Clinical Risk Integration 

**Authors**: Baramee Sukumal, Aueaphum Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2604.16104)  

**Abstract**: Lung cancer remains one of the leading causes of cancer-related mortality worldwide. Conventional computed tomography (CT) imaging, while essential for detection and staging, has limitations in distinguishing benign from malignant lesions and providing interpretable diagnostic insights. To address this challenge, this study proposes a dual-modal artificial intelligence framework that integrates CT radiology with hematoxylin and eosin (H&E) histopathology for lung cancer diagnosis and subtype classification. The system employs convolutional neural networks to extract radiologic and histopathologic features and incorporates clinical metadata to improve robustness. Predictions from both modalities are fused using a weighted decision-level integration mechanism to classify adenocarcinoma, squamous cell carcinoma, large cell carcinoma, small cell lung cancer, and normal tissue. Explainable AI techniques including Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion, Saliency Maps, and SmoothGrad are applied to provide visual interpretability. Experimental results show strong performance with accuracy up to 0.87, AUROC above 0.97, and macro F1-score of 0.88. Grad-CAM++ achieved the highest faithfulness and localization accuracy, demonstrating strong correspondence with expert-annotated tumor regions. These results indicate that multimodal fusion of radiology and histopathology can improve diagnostic performance while maintaining model transparency, suggesting potential for future clinical decision support systems in precision oncology. 

---
# Robust Synchronisation for Federated Learning in The Face of Correlated Device Failure 

**Authors**: Stefan Behfar, Richard Mortier  

**Link**: [PDF](https://arxiv.org/pdf/2604.16090)  

**Abstract**: Probabilistic Synchronous Parallel (PSP) is a technique in distributed learning systems to reduce synchronization bottlenecks by sampling a subset of participating nodes per round. In Federated Learning (FL), where edge devices are often unreliable due to factors including mobility, power constraints, and user activity, PSP helps improve system throughput. However, PSP has a key limitation: it assumes device behavior is static and different devices are independent. This can lead to unfair distributed synchronization, due to highly available nodes dominating training while those that are often unavailable rarely participate and so their data may be missed. If both data distribution and node availability are simultaneously correlated with the device, then both PSP and standard FL algorithms will suffer from persistent under-representation of certain classes or groups resulting in inefficient or ineffective learning of certain features. We introduce Availability-Weighted PSP (AW-PSP), an extension to PSP that addresses the issue of co-correlation of unfair sampling and data availability by dynamically adjusting node sampling probabilities using real-time availability predictions, historical behavior, and failure correlation metrics. A Markov-based availability predictor distinguishes transient \emph{vs} chronic failures, while a Distributed Hash Table (DHT) layer decentralizes metadata, including latency, freshness, and utility scores. We implement AW-PSP and trace-driven evaluation shows that it improves robustness to both independent and correlated failures, increases label coverage, and reduces fairness variance compared to standard PSP. AW-PSP thus provides an availability-aware, and fairness-conscious node sampling protocol for FL deployments that will scale to large numbers of nodes even in heterogeneous and failure-prone environments. 

---
# Stylistic-STORM (ST-STORM) : Perceiving the Semantic Nature of Appearance 

**Authors**: Hamed Ouattara, Pierre Duthon, Pascal Houssam Salmane, Frédéric Bernardin, Omar Ait Aider  

**Link**: [PDF](https://arxiv.org/pdf/2604.16086)  

**Abstract**: One of the dominant paradigms in self-supervised learning (SSL), illustrated by MoCo or DINO, aims to produce robust representations by capturing features that are insensitive to certain image transformations such as illumination, or geometric changes. This strategy is appropriate when the objective is to recognize objects independently of their appearance. However, it becomes counterproductive as soon as appearance itself constitutes the discriminative signal. In weather analysis, for example, rain streaks, snow granularity, atmospheric scattering, as well as reflections and halos, are not noise: they carry the essential information. In critical applications such as autonomous driving, ignoring these cues is risky, since grip and visibility depend directly on ground conditions and atmospheric conditions. We introduce ST-STORM, a hybrid SSL framework that treats appearance (style) as a semantic modality to be disentangled from content. Our architecture explicitly separates two latent streams, regulated by gating mechanisms. The Content branch aims at a stable semantic representation through a JEPA scheme coupled with a contrastive objective, promoting invariance to appearance variations. In parallel, the Style branch is constrained to capture appearance signatures (textures, contrasts, scattering) through feature prediction and reconstruction under an adversarial constraint. We evaluate ST-STORM on several tasks, including object classification (ImageNet-1K), fine-grained weather characterization, and melanoma detection (ISIC 2024 Challenge). The results show that the Style branch effectively isolates complex appearance phenomena (F1=97% on Multi-Weather and F1=94% on ISIC 2024 with 10% labeled data), without degrading the semantic performance (F1=80% on ImageNet-1K) of the Content branch, and improves the preservation of critical appearance 

---
# Unveiling Stochasticity: Universal Multi-modal Probabilistic Modeling for Traffic Forecasting 

**Authors**: Weijiang Xiong, Robert Fonod, Nikolas Geroliminis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16084)  

**Abstract**: Traffic forecasting is a challenging spatio-temporal modeling task and a critical component of urban transportation management. Current studies mainly focus on deterministic predictions, with limited considerations on the uncertainty and stochasticity in traffic dynamics. Therefore, this paper proposes an elegant yet universal approach that transforms existing models into probabilistic predictors by replacing only the final output layer with a novel Gaussian Mixture Model (GMM) layer. The modified model requires no changes to the training pipeline and can be trained using only the Negative Log-Likelihood (NLL) loss, without any auxiliary or regularization terms. Experiments on multiple traffic datasets show that our approach generalizes from classic to modern model architectures while preserving deterministic performance. Furthermore, we propose a systematic evaluation procedure based on cumulative distributions and confidence intervals, and demonstrate that our approach is considerably more accurate and informative than unimodal or deterministic baselines. Finally, a more detailed study on a real-world dense urban traffic network is presented to examine the impact of data quality on uncertainty quantification and to show the robustness of our approach under imperfect data conditions. Code available at this https URL 

---
# Early Detection of Acute Myeloid Leukemia (AML) Using YOLOv12 Deep Learning Model 

**Authors**: Enas E. Ahmed, Salah A. Aly, Mayar Moner  

**Link**: [PDF](https://arxiv.org/pdf/2604.16082)  

**Abstract**: Acute Myeloid Leukemia (AML) is one of the most life-threatening type of blood cancers, and its accurate classification is considered and remains a challenging task due to the visual similarity between various cell types. This study addresses the classification of the multiclasses of AML cells Utilizing YOLOv12 deep learning model. We applied two segmentation approaches based on cell and nucleus features, using Hue channel and Otsu thresholding techniques to preprocess the images prior to classification. Our experiments demonstrate that YOLOv12 with Otsu thresholding on cell-based segmentation achieved the highest level of validation and test accuracy, both reaching 99.3%. 

---
# Prototype-Grounded Concept Models for Verifiable Concept Alignment 

**Authors**: Stefano Colamonaco, David Debot, Pietro Barbiero, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2604.16076)  

**Abstract**: Concept Bottleneck Models (CBMs) aim to improve interpretability in Deep Learning by structuring predictions through human-understandable concepts, but they provide no way to verify whether learned concepts align with the human's intended meaning, hurting interpretability. We introduce Prototype-Grounded Concept Models (PGCMs), which ground concepts in learned visual prototypes: image parts that serve as explicit evidence for the concepts. This grounding enables direct inspection of concept semantics and supports targeted human intervention at the prototype level to correct misalignments. Empirically, PGCMs match the predictive performance of state-of-the-art CBMs while substantially improving transparency, interpretability, and intervenability. 

---
# Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs 

**Authors**: Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N. Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16060)  

**Abstract**: Multimodal Reasoning Models (MRMs) leveraging Chain-of-Thought (CoT) based thinking have revolutionized mathematical and logical problem-solving. However, we show that this paradigm struggles with generalized spatial intelligence. We perform a comprehensive evaluation of seventeen models across thirteen spatial benchmarks and identify a critical gap: CoT prompting consistently degrades performance in visual spatial reasoning. Furthermore, through a novel No-Image++ ablation, we demonstrate that MRMs and CoT prompted MLMs suffer from severe shortcut learning, and hallucinate visual details from textual priors even when the image is absent. These findings challenge the efficacy of text-only CoT for spatial tasks and underscore the need for vision-centric reasoning paradigms. 

---
# AST: Adaptive, Seamless, and Training-Free Precise Speech Editing 

**Authors**: Sihan Lv, Yechen Jin, Zhen Li, Jintao Chen, Jinshan Zhang, Ying Li, Jianwei Yin, Meng Xi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16056)  

**Abstract**: Text-based speech editing aims to modify specific segments while preserving speaker identity and acoustic context. Existing methods rely on task-specific training, which incurs high data costs and struggles with temporal fidelity in unedited regions. Meanwhile, adapting Text-to-Speech (TTS) models often faces a trade-off between editing quality and consistency. To address these issues, we propose AST, an Adaptive, Seamless, and Training-free precise speech editing framework. Leveraging a pre-trained autoregressive TTS model, AST introduces Latent Recomposition to selectively stitch preserved source segments with newly synthesized targets. Furthermore, AST extends this latent manipulation to enable precise style editing for specific speech segments. To prevent artifacts at these edit boundaries, the framework incorporates Adaptive Weak Fact Guidance (AWFG). AWFG dynamically modulates a mel-space guidance signal, enforcing structural constraints only where necessary without disrupting the generative manifold. To fill the gap of publicly accessible benchmarks, we introduce LibriSpeech-Edit, a new and larger speech editing dataset. As existing metrics poorly evaluate temporal consistency in unedited regions, we propose Word-level Dynamic Time Warping (WDTW). Extensive experiments demonstrate that AST resolves the controllability-quality trade-off without extra training. Compared to the previous most temporally consistent baseline, AST improves consistency while reducing Word Error Rate by nearly 70%. Moreover, applying AST to a foundation TTS model reduces WDTW by 27%, achieving state-of-the-art speaker preservation and temporal fidelity. 

---
# Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs 

**Authors**: Rohit Sinha, Aditya Kanade, Sai Srinivas Kancheti, Vineeth N Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16054)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive progress on vision language benchmarks, yet their capacity for visual cognitive and visuospatial reasoning remains less understood. We introduce "Mind's Eye", a multiple-choice benchmark of eight visuo-cognitive tasks inspired by classic human intelligence tests and organized under a novel "A-R-T" taxonomy: Abstraction, Relation, and Transformation. The tasks probe core processes of fluid intelligence such as pattern induction, analogical relation mapping, and mental transformation. We evaluate a diverse suite of closed-source and open-source MLLMs and compare their performance with human participants. Humans achieve 80% accuracy, while top performing MLLMs remain below 50%. Error analysis reveals failures in: (i) visual attention allocation, (ii) internal perceptual manipulation, and (iii) weak abstraction of underlying visual concepts. Our findings suggest that current MLLMs exhibit limited visuospatial reasoning capabilities, when compared with human participants, highlighting the need for more cognitively grounded evaluation frameworks. 

---
# Towards Intrinsic Interpretability of Large Language Models:A Survey of Design Principles and Architectures 

**Authors**: Yutong Gao, Qinglin Meng, Yuan Zhou, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16042)  

**Abstract**: While Large Language Models (LLMs) have achieved strong performance across many NLP tasks, their opaque internal mechanisms hinder trustworthiness and safe deployment. Existing surveys in explainable AI largely focus on post-hoc explanation methods that interpret trained models through external approximations. In contrast, intrinsic interpretability, which builds transparency directly into model architectures and computations, has recently emerged as a promising alternative. This paper presents a systematic review of the recent advances in intrinsic interpretability for LLMs, categorizing existing approaches into five design paradigms: functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparsity induction. We further discuss open challenges and outline future research directions in this emerging field. The paper list is available at: this https URL. 

---
# Safe Deep Reinforcement Learning for Building Heating Control and Demand-side Flexibility 

**Authors**: Colin Jüni, Mina Montazeri, Yi Guo, Federica Bellizio, Giovanni Sansavini, Philipp Heer  

**Link**: [PDF](https://arxiv.org/pdf/2604.16033)  

**Abstract**: Buildings account for approximately 40% of global energy consumption, and with the growing share of intermittent renewable energy sources, enabling demand-side flexibility, particularly in heating, ventilation and air conditioning systems, is essential for grid stability and energy efficiency. This paper presents a safe deep reinforcement learning-based control framework to optimize building space heating while enabling demand-side flexibility provision for power system operators. A deep deterministic policy gradient algorithm is used as the core deep reinforcement learning method, enabling the controller to learn an optimal heating strategy through interaction with the building thermal model while maintaining occupant comfort, minimizing energy cost, and providing flexibility. To address safety concerns with reinforcement learning, particularly regarding compliance with flexibility requests, we propose a real-time adaptive safety-filter to ensure that the system operates within predefined constraints during demand-side flexibility provision. The proposed real-time adaptive safety filter guarantees full compliance with flexibility requests from system operators and improves energy and cost efficiency -- achieving up to 50% savings compared to a rule-based controller -- while outperforming a standalone deep reinforcement learning-based controller in energy and cost metrics, with only a slight increase in comfort temperature violations. 

---
# Where does output diversity collapse in post-training? 

**Authors**: Constantinos Karouzos, Xingwei Tan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2604.16027)  

**Abstract**: Post-trained language models produce less varied outputs than their base counterparts. This output diversity collapse undermines inference-time scaling methods that rely on varied samples, and risks homogenizing model outputs on creative and value-laden tasks. Prior work attributes collapse to specific post-training methods, without separating the role of training data composition from the method, or the generation format from the model weights. We trace output diversity through three parallel post-training lineages of Olmo 3, Think (chain-of-thought distillation), Instruct (broad multi-source data), and RL-Zero, across 15 tasks and four text diversity metrics. We find that the location of collapse co-varies with data composition: the Think lineage loses most semantic diversity at supervised fine-tuning, and the effect of DPO is larger in Instruct than in Think. Suppressing chain-of-thought reasoning at inference in Think models drops accuracy on hard tasks, yet leaves answer-level diversity unchanged, showing that the collapse is embedded in the model weights by training data, not imposed by the generation format. Decomposing diversity loss on six verifiable tasks into a quality-control component (removal of incorrect outputs) and a residual component (genuine narrowing among correct outputs) reveals that the split is task-dependent, and Think models retain more correct-answer diversity than Instruct despite collapsing more in aggregate. Our results indicate that diversity collapse is determined during training by data composition and cannot be addressed at inference time alone. 

---
# Neurosymbolic Repo-level Code Localization 

**Authors**: Xiufeng Xu, Xiufeng Wu, Zejun Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16021)  

**Abstract**: Code localization is a cornerstone of autonomous software engineering. Recent advancements have achieved impressive performance on real-world issue benchmarks. However, we identify a critical yet overlooked bias: these benchmarks are saturated with keyword references (e.g. file paths, function names), encouraging models to rely on superficial lexical matching rather than genuine structural reasoning. We term this phenomenon the Keyword Shortcut. To address this, we formalize the challenge of Keyword-Agnostic Logical Code Localization (KA-LCL) and introduce KA-LogicQuery, a diagnostic benchmark requiring structural reasoning without any naming hints. Our evaluation reveals a catastrophic performance drop of state-of-the-art approaches on KA-LogicQuery, exposing their lack of deterministic reasoning capabilities. We propose LogicLoc, a novel agentic framework that combines large language models with the rigorous logical reasoning of Datalog for precise localization. LogicLoc extracts program facts from the codebase and leverages an LLM to synthesize Datalog programs, with parser-gated validation and mutation-based intermediate-rule diagnostic feedback to ensure correctness and efficiency. The validated programs are executed by a high-performance inference engine, enabling accurate and verifiable localization in a fully automated, closed-loop workflow. Experimental results demonstrate that LogicLoc significantly outperforms SOTA methods on KA-LogicQuery while maintaining competitive performance on popular issue-driven benchmarks. Notably, LogicLoc attains superior performance with significantly lower token consumption and faster execution by offloading structural traversal to a deterministic engine, reducing the overhead of iterative LLM inference. 

---
# AgentV-RL: Scaling Reward Modeling with Agentic Verifier 

**Authors**: Jiazheng Zhang, Ziche Fu, Zhiheng Xi, Wenqing Jing, Mingxu Chai, Wei He, Guoqiang Zhang, Chenghao Fan, Chenxin An, Wenxiang Chen, Zhicheng Liu, Haojie Pan, Dingwei Zhu, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16004)  

**Abstract**: Verifiers have been demonstrated to enhance LLM reasoning via test-time scaling (TTS). Yet, they face significant challenges in complex domains. Error propagation from incorrect intermediate reasoning can lead to false positives for seemingly plausible solutions, while lacking external grounding makes verifiers unreliable on computation or knowledge-intensive tasks. To address these challenges, we propose Agentic Verifier, a framework that transforms reward modeling into a multi-turn, tool-augmented deliberative process. We introduce complementary forward and backward agents: one traces solutions from premises to conclusions, while the other re-checks conclusions against their underlying premises. This bidirectional process enables a comprehensive, reliable, and interpretable assessment of solutions. To facilitate practical deployment, we propose AgentV-RL. Through proactive exploration and reinforcement learning, the verifier autonomously interleaves tool-use with internal reasoning. Extensive experiments show that Agentic Verifier yields consistent performance gains under both parallel and sequential TTS. Notably, our 4B variant surpasses state-of-the-art ORMs by 25.2%, positioning it as a promising paradigm for agentic reward modeling. 

---
# From Vulnerable Data Subjects to Vulnerabilizing Data Practices: Navigating the Protection Paradox in AI-Based Analyses of Platformized Lives 

**Authors**: Delfina S. Martinez Pandiani, Ella Streefkerk, Laurens Naudts, Paula Helm  

**Link**: [PDF](https://arxiv.org/pdf/2604.15990)  

**Abstract**: This paper traces a conceptual shift from understanding vulnerability as a static, essentialized property of data subjects to examining how it is actively enacted through data practices. Unlike reflexive ethical frameworks focused on missing or counter-data, we address the condition of abundance inherent to platformized life-a context where a near inexhaustible mass of data points already exists, shifting the ethical challenge to the researcher's choices in operating upon this existing mass. We argue that the ethical integrity of data science depends not just on who is studied, but on how technical pipelines transform "vulnerable" individuals into data subjects whose vulnerability can be further precarized. We develop this argument through an AI for Social Good (AI4SG) case: a journalist's request to use computer vision to quantify child presence in monetized YouTube 'family vlogs' for regulatory advocacy. This case reveals a "protection paradox": how data-driven efforts to protect vulnerable subjects can inadvertently impose new forms of computational exposure, reductionism, and extraction. Using this request as a point of departure, we perform a methodological deconstruction of the AI pipeline to show how granular technical decisions are ethically constitutive. We contribute a reflexive ethics protocol that translates these insights into a reflexive roadmap for research ethics surrounding platformized data subjects. Organized around four critical junctures-dataset design, operationalization, inference, and dissemination-the protocol identifies technical questions and ethical tensions where well-intentioned work can slide into renewed extraction or exposure. For every decision point, the protocol offers specific prompts to navigate four cross-cutting vulnerabilizing factors: exposure, monetization, narrative fixing, and algorithmic optimization. Rather than uncritically... 

---
# Polarization by Default: Auditing Recommendation Bias in LLM-Based Content Curation 

**Authors**: Nicolò Pagan, Christopher Barrie, Chris Andrew Bail, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.15937)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to curate and rank human-created content, yet the nature and structure of their biases in these tasks remains poorly understood: which biases are robust across providers and platforms, and which can be mitigated through prompt design. We present a controlled simulation study mapping content selection biases across three major LLM providers (OpenAI, Anthropic, Google) on real social media datasets from Twitter/X, Bluesky, and Reddit, using six prompting strategies (\textit{general}, \textit{popular}, \textit{engaging}, \textit{informative}, \textit{controversial}, \textit{neutral}). Through 540,000 simulated top-10 selections from pools of 100 posts across 54 experimental conditions, we find that biases differ substantially in how structural and how prompt-sensitive they are. Polarization is amplified across all configurations, toxicity handling shows a strong inversion between engagement- and information-focused prompts, and sentiment biases are predominantly negative. Provider comparisons reveal distinct trade-offs: GPT-4o Mini shows the most consistent behavior across prompts; Claude and Gemini exhibit high adaptivity in toxicity handling; Gemini shows the strongest negative sentiment preference. On Twitter/X, where author demographics can be inferred from profile bios, political leaning bias is the clearest demographic signal: left-leaning authors are systematically over-represented despite right-leaning authors forming the pool plurality in the dataset, and this pattern largely persists across prompts. 

---
# UniEditBench: A Unified and Cost-Effective Benchmark for Image and Video Editing via Distilled MLLMs 

**Authors**: Lifan Jiang, Tianrun Wu, Yuhang Pei, Chenyang Wang, Boxi Wu, Deng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15871)  

**Abstract**: The evaluation of visual editing models remains fragmented across methods and modalities. Existing benchmarks are often tailored to specific paradigms, making fair cross-paradigm comparisons difficult, while video editing lacks reliable evaluation benchmarks. Furthermore, common automatic metrics often misalign with human preference, yet directly deploying large multimodal models (MLLMs) as evaluators incurs prohibitive computational and financial costs. We present UniEditBench, a unified benchmark for image and video editing that supports reconstruction-based and instruction-driven methods under a shared protocol. UniEditBench includes a structured taxonomy of nine image operations (Add, Remove, Replace, Change, Stroke-based, Extract, Adjust, Count, Reorder) and eight video operations, with coverage of challenging compositional tasks such as counting and spatial reordering. To enable scalable evaluation, we distill a high-capacity MLLM judge (Qwen3-VL-235B-A22B Instruct) into lightweight 4B/8B evaluators that provide multi-dimensional scoring over structural fidelity, text alignment, background consistency, naturalness, and temporal-spatial consistency (for videos). Experiments show that the distilled evaluators maintain strong agreement with human judgments and substantially reduce deployment cost relative to the teacher model. UniEditBench provides a practical and reproducible protocol for benchmarking modern visual editing methods. Our benchmark and the associated reward models are publicly available at this https URL. 

---
# DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition 

**Authors**: Siun Kim, Hyung-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.15866)  

**Abstract**: Large language models (LLMs) have advanced information extraction (IE) by enabling zero-shot and few-shot named entity recognition (NER), yet their generative outputs still show persistent and systematic errors. Despite progress through instruction fine-tuning, zero-shot NER still lags far behind supervised systems. These recurring errors mirror inconsistencies observed in early-stage human annotation processes that resolve disagreements through pilot annotation. Motivated by this analogy, we introduce DiZiNER (Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition), a framework that simulates the pilot annotation process, employing LLMs to act as both annotators and supervisors. Multiple heterogeneous LLMs annotate shared texts, and a supervisor model analyzes inter-model disagreements to refine task instructions. Across 18 benchmarks, DiZiNER achieves zero-shot SOTA results on 14 datasets, improving prior bests by +8.0 F1 and reducing the zero-shot to supervised gap by over +11 points. It also consistently outperforms its supervisor, GPT-5 mini, indicating that improvements stem from disagreement-guided instruction refinement rather than model capacity. Pairwise agreement between models shows a strong correlation with NER performance, further supporting this finding. 

---
# QuantSightBench: Evaluating LLM Quantitative Forecasting with Prediction Intervals 

**Authors**: Jeremy Qin, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.15859)  

**Abstract**: Forecasting has become a natural benchmark for reasoning under uncertainty. Yet existing evaluations of large language models remain limited to judgmental tasks in simple formats, such as binary or multiple-choice questions. In practice, however, forecasting spans a far broader scope. Across domains such as economics, public health, and social demographics, decisions hinge on numerical estimates over continuous quantities, a capability that current benchmarks do not capture. Evaluating such estimates requires a format that makes uncertainty explicit and testable. We propose prediction intervals as a natural and rigorous interface for this purpose. They demand scale awareness, internal consistency across confidence levels, and calibration over a continuum of outcomes, making them a more suitable evaluation format than point estimates for numerical forecasting. To assess this capability, we introduce a new benchmark QuantSightBench, and evaluate frontier models under multiple settings, assessing both empirical coverage and interval sharpness. Our results show that none of the 11 evaluated frontier and open-weight models achieves the 90\% coverage target, with the top performers Gemini 3.1 Pro (79.1\%), Grok 4 (76.4\%), and GPT-5.4 (75.3\%) all falling at least 10 percentage points short. Calibration degrades sharply at extreme magnitudes, revealing systematic overconfidence across all evaluated models. 

---
# Robust Multispectral Semantic Segmentation under Missing or Full Modalities via Structured Latent Projection 

**Authors**: Irem Ulku, Erdem Akagündüz, Ömer Özgür Tanrıöver  

**Link**: [PDF](https://arxiv.org/pdf/2604.15856)  

**Abstract**: Multimodal remote sensing data provide complementary information for semantic segmentation, but in real-world deployments, some modalities may be unavailable due to sensor failures, acquisition issues, or challenging atmospheric conditions. Existing multimodal segmentation models typically address missing modalities by learning a shared representation across inputs. However, this approach can introduce a trade-off by compromising modality-specific complementary information and reducing performance when all modalities are available. In this paper, we tackle this limitation with CBC-SLP, a multimodal semantic segmentation model designed to preserve both modality-invariant and modality-specific information. Inspired by the theoretical results on modality alignment, which state that perfectly aligned multimodal representations can lead to sub-optimal performance in downstream prediction tasks, we propose a novel structured latent projection approach as an architectural inductive bias. Rather than enforcing this strategy through a loss term, we incorporate it directly into the architecture. In particular, to use the complementary information effectively while maintaining robustness under random modality dropout, we structure the latent representations into shared and modality-specific components and adaptively transfer them to the decoder according to the random modality availability mask. Extensive experiments on three multimodal remote sensing image sets demonstrate that CBC-SLP consistently outperforms state-of-the-art multimodal models across full and missing modality scenarios. Besides, we empirically demonstrate that the proposed strategy can recover the complementary information that may not be preserved in a shared representation. The code is available at this https URL. 

---
# DPrivBench: Benchmarking LLMs' Reasoning for Differential Privacy 

**Authors**: Erchi Wang, Pengrun Huang, Eli Chien, Om Thakkar, Kamalika Chaudhuri, Yu-Xiang Wang, Ruihan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15851)  

**Abstract**: Differential privacy (DP) has a wide range of applications for protecting data privacy, but designing and verifying DP algorithms requires expert-level reasoning, creating a high barrier for non-expert practitioners. Prior works either rely on specialized verification languages that demand substantial domain expertise or remain semi-automated and require human-in-the-loop guidance. In this work, we investigate whether large language models (LLMs) can automate DP reasoning. We introduce DPrivBench, a benchmark in which each instance asks whether a function or algorithm satisfies a stated DP guarantee under specified assumptions. The benchmark is carefully designed to cover a broad range of DP topics, span diverse difficulty levels, and resist shortcut reasoning through trivial pattern matching. Experiments show that while the strongest models handle textbook mechanisms well, all models struggle with advanced algorithms, revealing substantial gaps in current DP reasoning capabilities. Through further analytic study and failure-mode analysis, we identify several promising directions for improving automated DP reasoning. Our benchmark provides a solid foundation for developing and evaluating such methods, and complements existing benchmarks for mathematical reasoning. 

---
# ECG-Lens: Benchmarking ML & DL Models on PTB-XL Dataset 

**Authors**: Saloni Garg, Ukant Jadia, Amit Sagtani, Kamal Kant Hiran  

**Link**: [PDF](https://arxiv.org/pdf/2604.15822)  

**Abstract**: Automated classification of electrocardiogram (ECG) signals is a useful tool for diagnosing and monitoring cardiovascular diseases. This study compares three traditional machine learning algorithms (Decision Tree Classifier, Random Forest Classifier, and Logistic Regression) and three deep learning models (Simple Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Complex CNN (ECGLens)) for the classification of ECG signals from the PTB-XL dataset, which contains 12-lead recordings from normal patients and patients with various cardiac conditions. The DL models were trained on raw ECG signals, allowing them to automatically extract discriminative features. Data augmentation using the Stationary Wavelet Transform (SWT) was applied to enhance model performance, increase the diversity of training samples, and preserve the essential characteristics of the ECG signals. The models were evaluated using multiple metrics, including accuracy, precision, recall, F1-score, and ROC-AUC. The ECG-Lens model achieved the highest performance, with 80% classification accuracy and a 90% ROC-AUC. These findings demonstrate that deep learning architectures, particularly complex CNNs substantially outperform traditional ML methods on raw 12-lead ECG data, and provide a practical benchmark for selecting automated ECG classification models and identifying directions for condition-specific model development. 

---
# Beyond a Single Frame: Multi-Frame Spatially Grounded Reasoning Across Volumetric MRI 

**Authors**: Lama Moukheiber, Caleb M. Yeung, Haotian Xue, Alec Helbling, Zelin Zhao, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.15808)  

**Abstract**: Spatial reasoning and visual grounding are core capabilities for vision-language models (VLMs), yet most medical VLMs produce predictions without transparent reasoning or spatial evidence. Existing benchmarks also evaluate VLMs on isolated 2D images, overlooking the volumetric nature of clinical imaging, where findings can span multiple frames or appear on only a few slices. We introduce Spatially Grounded MRI Visual Question Answering (SGMRI-VQA), a 41,307-pair benchmark for multi-frame, spatially grounded reasoning on volumetric MRI. Built from expert radiologist annotations in the fastMRI+ dataset across brain and knee studies, each QA pair includes a clinician-aligned chain-of-thought trace with frame-indexed bounding box coordinates. Tasks are organized hierarchically across detection, localization, counting/classification, and captioning, requiring models to jointly reason about what is present, where it is, and across which frames it extends. We benchmark 10 VLMs and show that supervised fine-tuning of Qwen3-VL-8B with bounding box supervision consistently improves grounding performance over strong zero-shot baselines, indicating that targeted spatial supervision is an effective path toward grounded clinical reasoning. 

---
# From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation 

**Authors**: Jasper Lu, Zhenhao Shen, Yuanfei Wang, Shugao Liu, Shengqiang Xu, Shawn Xie, Jingkai Xu, Feng Jiang, Jade Yang, Chen Xie, Ruihai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15805)  

**Abstract**: Learning robust robot policies in real-world environments requires diverse data augmentation, yet scaling real-world data collection is costly due to the need for acquiring physical assets and reconfiguring environments. Therefore, augmenting real-world scenes into simulation has become a practical augmentation for efficient learning and evaluation. We present a generative framework that establishes a generative real-to-sim mapping from real-world panoramas to high-fidelity simulation scenes, and further synthesize diverse cousin scenes via semantic and geometric editing. Combined with high-quality physics engines and realistic assets, the generated scenes support interactive manipulation tasks. Additionally, we incorporate multi-room stitching to construct consistent large-scale environments for long-horizon navigation across complex layouts. Experiments demonstrate a strong sim-to-real correlation validating our platform's fidelity, and show that extensively scaling up data generation leads to significantly better generalization to unseen scene and object variations, demonstrating the effectiveness of Digital Cousins for generalizable robot learning and evaluation. 

---
# From Intention to Text: AI-Supported Goal Setting in Academic Writing 

**Authors**: Yueling Fan, Richard Lee Davis, Olga Viberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.15800)  

**Abstract**: This study presents WriteFlow, an AI voice-based writing assistant designed to support reflective academic writing through goal-oriented interaction. Academic writing involves iterative reflection and evolving goal regulation, yet prior research and a formative study with 17 participants show that writers often struggle to articulate and manage changing goals. While commonly used AI writing tools emphasize efficiency, they offer limited support for metacognition and writer agency. WriteFlow frames AI interaction as a dialogic space for ongoing goal articulation, monitoring, and negotiation grounded in writers' intentions. Findings from a Wizard-of-Oz study with 12 expert users show that WriteFlow scaffolds metacognitive regulation and reflection-in-action by supporting iterative goal refinement, maintaining goal-text alignment during drafting, and prompting evaluation of goal fulfillment. We discuss design implications for AI writing systems that prioritize reflective dialogue, flexible goal structures, and multi-perspective feedback to support intentional and agentic writing. 

---
# Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting 

**Authors**: Chi Liu, Xin Chen, Xu Zhou, Fangbo Tu, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15794)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, underpinning diverse AI applications. However, they often suffer from performance degradation due to factors such as catastrophic forgetting during Supervised Fine-Tuning (SFT), quantization, and pruning. In this work, we introduce a performance recovery framework based on Self-Distillation Fine-Tuning (SDFT) that effectively restores model capabilities. Complementing this practical contribution, we provide a rigorous theoretical explanation for the underlying recovery mechanism. We posit that an LLM's generative capability fundamentally relies on the high-dimensional manifold constructed by its hidden layers. To investigate this, we employ Centered Kernel Alignment (CKA) to quantify the alignment between student and teacher activation trajectories, leveraging its invariance to orthogonal transformations and scaling. Our experiments demonstrate a strong correlation between performance recovery and manifold alignment, substantiating the claim that self-distillation effectively aligns the student's high-dimensional manifold with the optimal structure represented by the teacher. This study bridges the gap between practical recovery frameworks and geometric representation theory, offering new insights into the internal mechanisms of self-distillation. 

---
# EVIL: Evolving Interpretable Algorithms for Zero-Shot Inference on Event Sequences and Time Series with LLMs 

**Authors**: David Berghaus  

**Link**: [PDF](https://arxiv.org/pdf/2604.15787)  

**Abstract**: We introduce EVIL (\textbf{EV}olving \textbf{I}nterpretable algorithms with \textbf{L}LMs), an approach that uses LLM-guided evolutionary search to discover simple, interpretable algorithms for dynamical systems inference. Rather than training neural networks on large datasets, EVIL evolves pure Python/NumPy programs that perform zero-shot, in-context inference across datasets. We apply EVIL to three distinct tasks: next-event prediction in temporal point processes, rate matrix estimation for Markov jump processes, and time series imputation. In each case, a single evolved algorithm generalizes across all evaluation datasets without per-dataset training (analogous to an amortized inference model). To the best of our knowledge, this is the first work to show that LLM-guided program evolution can discover a single compact inference function for these dynamical-systems problems. Across the three domains, the discovered algorithms are often competitive with, and even outperform, state-of-the-art deep learning models while being orders of magnitudes faster, and remaining fully interpretable. 

---
# SegMix:Shuffle-based Feedback Learning for Semantic Segmentation of Pathology Images 

**Authors**: Zhiling Yan, Sicheng Chen, Tianyi Zhang, Nan Ying, Yanli Lei, Guanglei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15777)  

**Abstract**: Segmentation is a critical task in computational pathology, as it identifies areas affected by disease or abnormal growth and is essential for diagnosis and treatment. However, acquiring high-quality pixel-level supervised segmentation data requires significant workload demands from experienced pathologists, limiting the application of deep learning. To overcome this challenge, relaxing the label conditions to image-level classification labels allows for more data to be used and more scenarios to be enabled. One approach is to leverage Class Activation Map (CAM) to generate pseudo pixel-level annotations for semantic segmentation with only image-level labels. However, this method fails to thoroughly explore the essential characteristics of pathology images, thus identifying only small areas that are insufficient for pseudo masking. In this paper, we propose a novel shuffle-based feedback learning method inspired by curriculum learning to generate higher-quality pseudo-semantic segmentation masks. Specifically, we perform patch level shuffle of pathology images, with the model adaptively adjusting the shuffle strategy based on feedback from previous learning. Experimental results demonstrate that our proposed approach outperforms state-of-the-arts on three different datasets. 

---
# PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection 

**Authors**: Pritesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.15776)  

**Abstract**: We present PIIBench, a unified benchmark corpus for Personally Identifiable Information (PII) detection in natural language text. Existing resources for PII detection are fragmented across domain-specific corpora with mutually incompatible annotation schemes, preventing systematic comparison of detection systems. We consolidate ten publicly available datasets spanning synthetic PII corpora, multilingual Named Entity Recognition (NER) benchmarks, and financial domain annotated text, yielding a corpus of 2,369,883 annotated sequences and 3.35 million entity mentions across 48 canonical PII entity types. We develop a principled normalization pipeline that maps 80+ source-specific label variants to a standardized BIO tagging scheme, applies frequency-based suppression of near absent entity types, and produces stratified 80/10/10 train/validation/test splits preserving source distribution. To establish baseline difficulty, we evaluate eight published systems spanning rule-based engines (Microsoft Presidio), general purpose NER models (spaCy, BERT-base NER, XLM-RoBERTa NER, SpanMarker mBERT, SpanMarker BERT), a PII-specific model (Piiranha DeBERTa), and a financial NER specialist (XtremeDistil FiNER). All systems achieve span-level F1 below 0.14, with the best system (Presidio, F1=0.1385) still producing zero recall on most entity types. These results directly quantify the domain-silo problem and demonstrate that PIIBench presents a substantially harder and more comprehensive evaluation challenge than any existing single source PII dataset. The dataset construction pipeline and benchmark evaluation code are publicly available at this https URL. 

---
# Phase Transitions as the Breakdown of Statistical Indistinguishability 

**Authors**: Taiyo Narita, Hideyuki Miyahara  

**Link**: [PDF](https://arxiv.org/pdf/2604.15773)  

**Abstract**: We introduce a novel characterization of phase transitions based on hypothesis testing.
In our formulation, a phase transition is defined as the breakdown of statistical indistinguishability under vanishing parameter perturbations in the thermodynamic limit.
This perspective provides a general, order-parameter-free framework that does not rely on model-specific insights or learning procedures.
We show that conventional approaches, such as those based on the Binder parameter, can be reinterpreted as special cases within this framework.
As a concrete realization, we employ a distribution-free two-sample run test and demonstrate that the critical point of the two-dimensional Ising model is accurately identified without prior knowledge of the order parameter. 

---
# Closing the Theory-Practice Gap in Spiking Transformers via Effective Dimension 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15769)  

**Abstract**: Spiking transformers achieve competitive accuracy with conventional transformers while offering $38$-$57\times$ energy efficiency on neuromorphic hardware, yet no theoretical framework guides their design. This paper establishes the first comprehensive expressivity theory for spiking self-attention. We prove that spiking attention with Leaky Integrate-and-Fire neurons is a universal approximator of continuous permutation-equivariant functions, providing explicit spike circuit constructions including a novel lateral inhibition network for softmax normalization with proven $O(1/\sqrt{T})$ convergence. We derive tight spike-count lower bounds via rate-distortion theory: $\varepsilon$-approximation requires $\Omega(L_f^2 nd/\varepsilon^2)$ spikes, with rigorous information-theoretic derivation. Our key insight is input-dependent bounds using measured effective dimensions ($d_{\text{eff}}=47$--$89$ for CIFAR/ImageNet), explaining why $T=4$ timesteps suffice despite worst-case $T \geq 10{,}000$ predictions. We provide concrete design rules with calibrated constants ($C=2.3$, 95\% CI: $[1.9, 2.7]$). Experiments on Spikformer, QKFormer, and SpikingResformer across vision and language benchmarks validate predictions with $R^2=0.97$ ($p<0.001$). Our framework provides the first principled foundation for neuromorphic transformer design. 

---
# cuNNQS-SCI: A Fully GPU-Accelerated Framework for High-Performance Configuration Interaction Selection withNeural Network QQantum States 

**Authors**: Daran Sun, Bowen Kan, Haoquan Long, Hairui Zhao, Haoxu Li, Yicheng Liu, Pengyu Zhou, Ankang Feng, Wenjing Huang, Yida Gu, Zhenyu Li, Honghui Shang, Yunquan Zhang, Dingwen Tao, Ninghui Sun, Guangming Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15768)  

**Abstract**: AI-driven methods have demonstrated considerable success in tackling the central challenge of accurately solving the Schrödinger equation for complex many-body systems. Among neural network quantum state (NNQS) approaches, the NNQS-SCI (Selected Configuration Interaction) method stands out as a state-of-the-art technique, recognized for its high accuracy and scalability. However, its application to larger systems is severely constrained by a hybrid CPU-GPU architecture. Specifically, centralized CPU-based global de-duplication creates a severe scalability barrier due to communication bottlenecks, while host-resident coupled-configuration generation induces prohibitive computational overheads. We introduce cuNNQS-SCI, a fully GPU-accelerated SCI framework designed to overcome these bottlenecks. cuNNQS-SCI first integrates a distributed, load-balanced global de-duplication algorithm to minimize redundancy and communication overhead at scale. To address compute limitations, it employs specialized, fine-grained CUDA kernels for exact coupled configuration generation. Finally, to break the single-GPU memory barrier exposed by this full acceleration, it incorporates a GPU memory-centric runtime featuring GPU-side pooling, streaming mini-batches, and overlapped offloading. This design enables much larger configuration spaces and shifts the bottleneck from host-side limitations back to on-device inference. Our evaluation demonstrates that cuNNQS-SCI fundamentally expands the scale of solvable problems. On an NVIDIA A100 cluster with 64 GPUs, cuNNQS-SCI achieves up to 2.32X end-to-end speedup over the highly-optimized NNQS-SCI baseline while preserving the same chemical accuracy. Furthermore, it demonstrates excellent distributed performance, maintaining over 90% parallel efficiency in strong scaling tests. 

---
# When Do Early-Exit Networks Generalize? A PAC-Bayesian Theory of Adaptive Depth 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15764)  

**Abstract**: Early-exit neural networks enable adaptive computation by allowing confident predictions to exit at intermediate layers, achieving 2-8$\times$ inference speedup. Despite widespread deployment, their generalization properties lack theoretical understanding -- a gap explicitly identified in recent surveys. This paper establishes a unified PAC-Bayesian framework for adaptive-depth networks. (1) Novel Entropy-Based Bounds: We prove the first generalization bounds depending on exit-depth entropy $H(D)$ and expected depth $\mathbb{E}[D]$ rather than maximum depth $K$, with sample complexity $\mathcal{O}((\mathbb{E}[D] \cdot d + H(D))/\epsilon^2)$. (2) Explicit Constructive Constants: Our analysis yields the leading coefficient $\sqrt{2\ln 2} \approx 1.177$ with complete derivation. (3) Provable Early-Exit Advantages: We establish sufficient conditions under which adaptive-depth networks strictly outperform fixed-depth counterparts. (4) Extension to Approximate Label Independence: We relax the label-independence assumption to $\epsilon$-approximate policies, broadening applicability to learned routing. (5) Comprehensive Validation: Experiments across 6 architectures on 7 benchmarks demonstrate tightness ratios of 1.52-3.87$\times$ (all $p < 0.001$) versus $>$100$\times$ for classical bounds. Bound-guided threshold selection matches validation-tuned performance within 0.1-0.3%. 

---
# DepCap: Adaptive Block-Wise Parallel Decoding for Efficient Diffusion LM Inference 

**Authors**: Xiang Xia, Wuyang Zhang, Jiazheng Liu, Cheng Yan, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15750)  

**Abstract**: Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive language generation due to their potential for parallel decoding and global refinement of the entire sequence. To unlock this potential, DLM inference must carefully balance generation quality and decoding speed. Recent block-wise DLM decoding methods improve this trade-off by performing diffusion-based decoding sequentially in blocks. However, existing methods typically rely on fixed block schedules or current-step local signals to determine block boundaries, and use conservative confidence-based parallel decoding to avoid conflicts, limiting the quality-speed trade-off. In this paper, we argue that block-wise DLM inference requires more suitable signals for its two core decisions: cross-step signals for determining block boundaries, and token-level conflict signals for parallel decoding. Based on this view, we propose DepCap, a training-free framework for efficient block-wise DLM inference. Specifically, DepCap instantiates the cross-step signal as the influence of the last decoded block and uses it to adaptively determine how far the next block should extend, while identifying a conflict-free subset of tokens for safe parallel decoding within each block, enabling substantial inference acceleration with negligible quality degradation. DepCap is a plug-and-play method applicable to various DLMs, and compatible with existing KV-cache strategies for block-wise DLM. An information-theoretic analysis further suggests that the cumulative last-block influence on a candidate block is approximately additive across tokens, supporting the proposed block-partitioning criterion. Experimental results show that DepCap achieves favorable speed-quality trade-offs across multiple DLM backbones and reasoning and coding benchmarks, with up to 5.63$\times$ speedup without significant performance degradation. 

---
# Learning Uncertainty from Sequential Internal Dispersion in Large Language Models 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Cong-Duy Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15741)  

**Abstract**: Uncertainty estimation is a promising approach to detect hallucinations in large language models (LLMs). Recent approaches commonly depend on model internal states to estimate uncertainty. However, they suffer from strict assumptions on how hidden states should evolve across layers, and from information loss by solely focusing on last or mean tokens. To address these issues, we present Sequential Internal Variance Representation (SIVR), a supervised hallucination detection framework that leverages token-wise, layer-wise features derived from hidden states. SIVR adopts a more basic assumption that uncertainty manifests in the degree of dispersion or variance of internal representations across layers, rather than relying on specific assumptions, which makes the method model and task agnostic. It additionally aggregates the full sequence of per-token variance features, learning temporal patterns indicative of factual errors and thereby preventing information loss. Experimental results demonstrate SIVR consistently outperforms strong baselines. Most importantly, SIVR enjoys stronger generalisation and avoids relying on large training sets, highlighting the potential for practical deployment. Our code repository is available online at this https URL. 

---
# Sketch and Text Synergy: Fusing Structural Contours and Descriptive Attributes for Fine-Grained Image Retrieval 

**Authors**: Siyuan Wang, Hanchen Gao, Guangming Zhu, Jiang Lu, Yiyue Ma, Tianci Wu, Jincai Huang, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15735)  

**Abstract**: Fine-grained image retrieval via hand-drawn sketches or textual descriptions remains a critical challenge due to inherent modality gaps. While hand-drawn sketches capture complex structural contours, they lack color and texture, which text effectively provides despite omitting spatial contours. Motivated by the complementary nature of these modalities, we propose the Sketch and Text Based Image Retrieval (STBIR) framework. By synergizing the rich color and texture cues from text with the structural outlines provided by sketches, STBIR achieves superior fine-grained retrieval performance. First, a curriculum learning driven robustness enhancement module is proposed to enhance the model's robustness when handling queries of varying quality. Second, we introduce a category-knowledge-based feature space optimization module, thereby significantly boosting the model's representational power. Finally, we design a multi-stage cross-modal feature alignment mechanism to effectively mitigate the challenges of cross modal feature alignment. Furthermore, we curate the fine-grained STBIR benchmark dataset to rigorously validate the efficacy of our proposed framework and to provide data support as a reference for subsequent related research. Extensive experiments demonstrate that the proposed STBIR framework significantly outperforms state of the art methods. 

---
# MambaBack: Bridging Local Features and Global Contexts in Whole Slide Image Analysis 

**Authors**: Sicheng Chen, Chad Wong, Tianyi Zhang, Enhui Chai, Zeyu Liu, Fei Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.15729)  

**Abstract**: Whole Slide Image (WSI) analysis is pivotal in computational pathology, enabling cancer diagnosis by integrating morphological and architectural cues across magnifications. Multiple Instance Learning (MIL) serves as the standard framework for WSI analysis. Recently, Mamba has become a promising backbone for MIL, overtaking Transformers due to its efficiency and global context modeling capabilities originating from Natural Language Processing (NLP). However, existing Mamba-based MIL approaches face three critical challenges: (1) disruption of 2D spatial locality during 1D sequence flattening; (2) sub-optimal modeling of fine-grained local cellular structures; and (3) high memory peaks during inference on resource-constrained edge devices. Studies like MambaOut reveal that Mamba's SSM component is redundant for local feature extraction, where Gated CNNs suffice. Recognizing that WSI analysis demands both fine-grained local feature extraction akin to natural images, and global context modeling akin to NLP, we propose MambaBack, a novel hybrid architecture that harmonizes the strengths of Mamba and MambaOut. First, we propose the Hilbert sampling strategy to preserve the 2D spatial locality of tiles within 1D sequences, enhancing the model's spatial perception. Second, we design a hierarchical structure comprising a 1D Gated CNN block based on MambaOut to capture local cellular features, and a BiMamba2 block to aggregate global context, jointly enhancing multi-scale representation. Finally, we implement an asymmetric chunking design, allowing parallel processing during training and chunking-streaming accumulation during inference, minimizing peak memory usage for deployment. Experimental results on five datasets demonstrate that MambaBack outperforms seven state-of-the-art methods. Source code and datasets are publicly available. 

---
# Privacy-Preserving LLMs Routing 

**Authors**: Xidong Wu, Yukuan Zhang, Yuqiong Ji, Reza Shirkavand, Qian Lou, Shangqian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15728)  

**Abstract**: Large language model (LLM) routing has emerged as a critical strategy to balance model performance and cost-efficiency by dynamically selecting services from various model providers. However, LLM routing adds an intermediate layer between users and LLMs, creating new privacy risks to user data. These privacy risks have not been systematically studied. Although cryptographic techniques such as Secure Multi-Party Computation (MPC) enable privacy-preserving computation, their protocol design and implementation remain under-explored, and naïve implementations typically incur prohibitive computational overhead. To address this, we propose a privacy-preserving LLM routing framework (PPRoute). PPRoute includes multiple strategies to speed up encoder inference and nearest neighbor search under the MPC and maintain the quality of LLM routing. First, PPRoute uses MPC-friendly operations to boost the encoder inference. Second, PPRoute uses a multiple-step model training algorithm to maintain routing quality despite the constraints of the encrypted domain. Third, PPRoute proposes an unsorted Top-k algorithm with $O(1)$ communication complexity for secure sorting in model search, significantly reducing communication latency. Across different datasets, PPRoute achieves the performance of plaintext counterparts, while achieving approximately a 20$\times$ speedup over naïve MPC implementations. 

---
# Reasoning-targeted Jailbreak Attacks on Large Reasoning Models via Semantic Triggers and Psychological Framing 

**Authors**: Zehao Wang, Lanjun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15725)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong capabilities in generating step-by-step reasoning chains alongside final answers, enabling their deployment in high-stakes domains such as healthcare and education. While prior jailbreak attack studies have focused on the safety of final answers, little attention has been given to the safety of the reasoning process. In this work, we identify a novel problem that injects harmful content into the reasoning steps while preserving unchanged answers. This type of attack presents two key challenges: 1) manipulating the input instructions may inadvertently alter the LRM's final answer, and 2) the diversity of input questions makes it difficult to consistently bypass the LRM's safety alignment mechanisms and embed harmful content into its reasoning process. To address these challenges, we propose the Psychology-based Reasoning-targeted Jailbreak Attack (PRJA) Framework, which integrates a Semantic-based Trigger Selection module and a Psychology-based Instruction Generation module. Specifically, the proposed PRJA automatically selects manipulative reasoning triggers via semantic analysis and leverages psychological theories of obedience to authority and moral disengagement to generate adaptive instructions for enhancing the LRM's compliance with harmful content generation. Extensive experiments on five question-answering datasets demonstrate that PRJA achieves an average attack success rate of 83.6\% against several commercial LRMs, including DeepSeek R1, Qwen2.5-Max, and OpenAI o4-mini. 

---
# Diffusion Autoencoder for Unsupervised Artifact Restoration in Handheld Fundus Images 

**Authors**: Mathumetha Palani, Kavya Puthumana, Ayantika Das, Ganapathy Krishnamurthi  

**Link**: [PDF](https://arxiv.org/pdf/2604.15723)  

**Abstract**: The advent of handheld fundus imaging devices has made ophthalmologic diagnosis and disease screening more accessible, efficient, and cost-effective. However, images captured from these setups often suffer from artifacts such as flash reflections, exposure variations, and motion-induced blur, which degrade image quality and hinder downstream analysis. While generative models have been effective in image restoration, most depend on paired supervision or predefined artifact structures, making them less adaptable to unstructured degradations commonly observed in handheld fundus images. To address this, we propose an unsupervised diffusion autoencoder that integrates a context encoder with the denoising process to learn semantically meaningful representations for artifact restoration. The model is trained only on high-quality table-top fundus images and infers to restore artifact-affected handheld acquisitions. We validate the restorations through quantitative and qualitative evaluations, and have shown that diagnostic accuracy increases to 81.17% on an unseen dataset and multiple artifact conditions 

---
# NeuroLip: An Event-driven Spatiotemporal Learning Framework for Cross-Scene Lip-Motion-based Visual Speaker Recognition 

**Authors**: Junguang Yao, Wenye Liu, Stjepan Picek, Yue Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.15718)  

**Abstract**: Visual speaker recognition based on lip motion offers a silent, hands-free, and behavior-driven biometric solution that remains effective even when acoustic cues are unavailable. Compared to traditional methods that rely heavily on appearance-dependent representations, lip motion encodes subject-specific behavioral dynamics driven by consistent articulation patterns and muscle coordination, offering inherent stability across environmental changes. However, capturing these robust, fine-grained dynamics is challenging for conventional frame-based cameras due to motion blur and low dynamic range. To exploit the intrinsic stability of lip motion and address these sensing limitations, we propose NeuroLip, an event-based framework that captures fine-grained lip dynamics under a strict yet practical cross-scene protocol: training is performed under a single controlled condition, while recognition must generalize to unseen viewing and lighting conditions. NeuroLip features a 1) Temporal-aware Voxel Encoding module with adaptive event weighting, 2) Structure-aware Spatial Enhancer that amplifies discriminative behavioral patterns by suppressing noise while preserving vertically structured motion information, and 3) Polarity Consistency Regularization mechanism to retain motion-direction cues encoded in event polarities. To facilitate systematic evaluation, we introduce DVSpeaker, a comprehensive event-based lip-motion dataset comprising 50 subjects recorded under four distinct viewpoint and illumination scenarios. Extensive experiments demonstrate that NeuroLip achieves near-perfect matched-scene accuracy and robust cross-scene generalization, attaining over 71% accuracy on unseen viewpoints and nearly 76% under low-light conditions, outperforming representative existing methods by at least 8.54%. The dataset and code are publicly available at this https URL. 

---
# GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows 

**Authors**: Jize Wang, Xuanxuan Liu, Yining Li, Songyang Zhang, Yijun Wang, Zifei Shan, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15715)  

**Abstract**: The development of general-purpose agents requires a shift from executing simple instructions to completing complex, real-world productivity workflows. However, current tool-use benchmarks remain misaligned with real-world requirements, relying on AI-generated queries, dummy tools, and limited system-level coordination. To address this, we propose GTA-2, a hierarchical benchmark for General Tool Agents (GTA) spanning atomic tool use and open-ended workflows. Built on real-world authenticity, it leverages real user queries, deployed tools, and multimodal contexts. (i) GTA-Atomic, inherited from our prior GTA benchmark, evaluates short-horizon, closed-ended tool-use precision. (ii) GTA-Workflow introduces long-horizon, open-ended tasks for realistic end-to-end completion. To evaluate open-ended deliverables, we propose a recursive checkpoint-based evaluation mechanism that decomposes objectives into verifiable sub-goals, enabling unified evaluation of both model capabilities and agent execution frameworks (i.e., execution harnesses). Experiments reveal a pronounced capability cliff: while frontier models already struggle on atomic tasks (below 50%), they largely fail on workflows, with top models achieving only 14.39% success. Further analysis shows that checkpoint-guided feedback improves performance, while advanced frameworks such as Manus and OpenClaw substantially enhance workflow completion, highlighting the importance of execution harness design beyond the underlying model capacity. These findings provide guidance for developing reliable personal and professional assistants. Dataset and code will be available at this https URL. 

---
# Just Type It in Isabelle! AI Agents Drafting, Mechanizing, and Generalizing from Human Hints 

**Authors**: Kevin Kappelmann, Maximilian Schäffeler, Lukas Stevens, Mohammad Abdulaziz, Andrei Popescu, Dmitriy Traytel  

**Link**: [PDF](https://arxiv.org/pdf/2604.15713)  

**Abstract**: Type annotations are essential when printing terms in a way that preserves their meaning under reparsing and type inference. We study the problem of complete and minimal type annotations for rank-one polymorphic $\lambda$-calculus terms, as used in Isabelle. Building on prior work by Smolka, Blanchette et al., we give a metatheoretical account of the problem, with a full formal specification and proofs, and formalize it in Isabelle/HOL. Our development is a series of experiments featuring human-driven and AI-driven formalization workflows: a human and an LLM-powered AI agent independently produce pen-and-paper proofs, and the AI agent autoformalizes both in Isabelle, with further human-hinted AI interventions refining and generalizing the development. 

---
# SSMamba: A Self-Supervised Hybrid State Space Model for Pathological Image Classification 

**Authors**: Enhui Chai, Sicheng Chen, Tianyi Zhang, Xingyu Li, Tianxiang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.15711)  

**Abstract**: Pathological diagnosis is highly reliant on image analysis, where Regions of Interest (ROIs) serve as the primary basis for diagnostic evidence, while whole-slide image (WSI)-level tasks primarily capture aggregated patterns. To extract these critical morphological features, ROI-level Foundation Models (FMs) based on Vision Transformers (ViTs) and large-scale self-supervised learning (SSL) have been widely adopted. However, three core limitations remain in their application to ROI analysis: (1) cross-magnification domain shift, as fixed-scale pretraining hinders adaptation to diverse clinical settings; (2) inadequate local-global relationship modeling, wherein the ViT backbone of FMs suffers from high computational overhead and imprecise local characterization; (3) insufficient fine-grained sensitivity, as traditional self-attention mechanisms tend to overlook subtle diagnostic cues. To address these challenges, we propose SSMamba, a hybrid SSL framework that enables effective fine-grained feature learning without relying on large external datasets. This framework incorporates three domain-adaptive components: Mamba Masked Image Modeling (MAMIM) for mitigating domain shift, a Directional Multi-scale (DMS) module for balanced local-global modeling, and a Local Perception Residual (LPR) module for enhanced fine-grained sensitivity. Employing a two-stage pipeline, SSL pretraining on target ROI datasets followed by supervised fine-tuning (SFT), SSMamba outperforms 11 state-of-the-art (SOTA) pathological FMs on 10 public ROI datasets and surpasses 8 SOTA methods on 6 public WSI datasets. These results validate the superiority of task-specific architectural designs for pathological image analysis. 

---
# The Price of Paranoia: Robust Risk-Sensitive Cooperation in Non-Stationary Multi-Agent Reinforcement Learning 

**Authors**: Deep Kumar Ganguly, Chandradithya S Jonnalagadda, Pratham Chintamani, Adithya Ananth  

**Link**: [PDF](https://arxiv.org/pdf/2604.15695)  

**Abstract**: Cooperative equilibria are fragile. When agents learn alongside each other rather than in a fixed environment, the process of learning destabilizes the cooperation they are trying to sustain: every gradient step an agent takes shifts the distribution of actions its partner will play, turning a cooperative partner into a source of stochastic noise precisely where the cooperation decision is most sensitive. We study how this co-learning noise propagates through the structure of coordination games, and find that the cooperative equilibrium, even when strongly Pareto-dominant, is exponentially unstable under standard risk-neutral learning, collapsing irreversibly once partner noise crosses the game's critical cooperation threshold. The natural response to apply distributional robustness to hedge against partner uncertainty makes things strictly worse: risk-averse return objectives penalize the high-variance cooperative action relative to defection, widening the instability region rather than shrinking it, a paradox that reveals a fundamental mismatch between the domains where robustness is applied and instability originates. We resolve this by showing that robustness should target the policy gradient update variance induced by partner uncertainty, not the return distribution. This distinction yields an algorithm whose gradient updates are modulated by an online measure of partner unpredictability, provably expanding the cooperation basin in symmetric coordination games. To unify stability, sample complexity, and welfare consequences of this approach, we introduce the Price of Paranoia as the structural dual of the Price of Anarchy. Together with a novel Cooperation Window, it precisely characterizes how much welfare learning algorithms can recover under partner noise, pinning down the optimal degree of robustness as a closed-form balance between equilibrium stability and sample efficiency. 

---
# Hierarchical Active Inference using Successor Representations 

**Authors**: Prashant Rangarajan, Rajesh P. N. Rao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15679)  

**Abstract**: Active inference, a neurally-inspired model for inferring actions based on the free energy principle (FEP), has been proposed as a unifying framework for understanding perception, action, and learning in the brain. Active inference has previously been used to model ecologically important tasks such as navigation and planning, but scaling it to solve complex large-scale problems in real-world environments has remained a challenge. Inspired by the existence of multi-scale hierarchical representations in the brain, we propose a model for planning of actions based on hierarchical active inference. Our approach combines a hierarchical model of the environment with successor representations for efficient planning. We present results demonstrating (1) how lower-level successor representations can be used to learn higher-level abstract states, (2) how planning based on active inference at the lower-level can be used to bootstrap and learn higher-level abstract actions, and (3) how these learned higher-level abstract states and actions can facilitate efficient planning. We illustrate the performance of the approach on several planning and reinforcement learning (RL) problems including a variant of the well-known four rooms task, a key-based navigation task, a partially observable planning problem, the Mountain Car problem, and PointMaze, a family of navigation tasks with continuous state and action spaces. Our results represent, to our knowledge, the first application of learned hierarchical state and action abstractions to active inference in FEP-based theories of brain function. 

---
# CodeMMR: Bridging Natural Language, Code, and Image for Unified Retrieval 

**Authors**: Jiahui Geng, Qing Li, Fengyu Cai, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2604.15663)  

**Abstract**: Code search, framed as information retrieval (IR), underpins modern software engineering and increasingly powers retrieval-augmented generation (RAG), improving code discovery, reuse, and the reliability of LLM-based coding. Yet existing code IR models remain largely text-centric and often overlook the visual and structural aspects inherent in programming artifacts such as web interfaces, data visualizations, SVGs, schematic diagrams, and UML. To bridge this gap, we introduce MMCoIR, the first comprehensive benchmark for evaluating multimodal code IR across five visual domains, eight programming languages, eleven libraries, and show the challenge of the task through extensive evaluation. Therefore, we then propose CodeMMR, a unified retrieval model that jointly embeds natural language, code, and images into a shared semantic space through instruction-based multimodal alignment. CodeMMR achieves strong generalization across modalities and languages, outperforming competitive baselines (e.g., UniIR, GME, VLM2Vec) by an average of 10 points on nDCG@10. Moreover, integrating CodeMMR into RAG enhances code generation fidelity and visual grounding on unseen code generation tasks, underscoring the potential of multimodal retrieval as a core enabler for next-generation intelligent programming systems. Datasets are available at HuggingFace. 

---
# HYPERHEURIST: A Simulated Annealing-Based Control Framework for LLM-Driven Code Generation in Optimized Hardware Design 

**Authors**: Shiva Ahir, Prajna Bhat, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2604.15642)  

**Abstract**: Large Language Models (LLMs) have shown promising progress for generating Register Transfer Level (RTL) hardware designs, largely because they can rapidly propose alternative architectural realizations. However, single-shot LLM generation struggles to consistently produce designs that are both functionally correct and power-efficient. This paper proposes HYPERHEURIST, a simulated annealing-based control framework that treats LLM-generated RTL as intermediate candidates rather than final designs. The suggested system not only focuses on functionality correctness but also on Power-Performance-Area (PPA) optimization. In the first phase, RTL candidates are filtered through compilation, structural checks, and simulation to identify functionally valid designs. PPA optimization is restricted to RTL designs that have already passed compilation and simulation. Evaluated across eight RTL benchmarks, this staged approach yields more stable and repeatable optimization behavior than single-pass LLM-generated RTL. 

---
# Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking 

**Authors**: Jun Feng, Jiahui Tang, Zhicheng He, Hang Lv, Hongchao Gu, Hao Wang, Xuezhi Yang, Shuai Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15621)  

**Abstract**: Adaptive Retrieval-Augmented Generation aims to mitigate the interference of extraneous noise by dynamically determining the necessity of retrieving supplementary passages. However, as Large Language Models evolve with increasing robustness to noise, the necessity of adaptive retrieval warrants re-evaluation. In this paper, we rethink this necessity and propose AdaRankLLM, a novel adaptive retrieval framework. To effectively verify the necessity of adaptive listwise reranking, we first develop an adaptive ranker employing a zero-shot prompt with a passage dropout mechanism, and compare its generation outcomes against static fixed-depth retrieval strategies. Furthermore, to endow smaller open-source LLMs with this precise listwise ranking and adaptive filtering capability, we introduce a two-stage progressive distillation paradigm enhanced by data sampling and augmentation techniques. Extensive experiments across three datasets and eight LLMs demonstrate that AdaRankLLM consistently achieves optimal performance in most scenarios with significantly reduced context overhead. Crucially, our analysis reveals a role shift in adaptive retrieval: it functions as a critical noise filter for weaker models to overcome their limitations, while serving as a cost-effective efficiency optimizer for stronger reasoning models. 

---
# VoodooNet: Achieving Analytic Ground States via High-Dimensional Random Projections 

**Authors**: Wladimir Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.15613)  

**Abstract**: We present VoodooNet, a non-iterative neural architecture that replaces the stochastic gradient descent (SGD) paradigm with a closed-form analytic solution via Galactic Expansion. By projecting input manifolds into a high-dimensional, high-entropy "Galactic" space ($d \gg 784$), we demonstrate that complex features can be untangled without the thermodynamic cost of backpropagation. Utilizing the Moore-Penrose pseudoinverse to solve for the output layer in a single step, VoodooNet achieves a classification accuracy of \textbf{98.10\% on MNIST} and \textbf{86.63\% on Fashion-MNIST}. Notably, our results on Fashion-MNIST surpass a 10-epoch SGD baseline (84.41\%) while reducing the training time by orders of magnitude. We observe a near-logarithmic scaling law between dimensionality and accuracy, suggesting that performance is a function of "Galactic" volume rather than iterative refinement. This "Magic Hat" approach offers a new frontier for real-time Edge AI, where the traditional training phase is bypassed in favor of instantaneous manifold discovery. 

---
# CLIMB: Controllable Longitudinal Brain Image Generation using Mamba-based Latent Diffusion Model and Gaussian-aligned Autoencoder 

**Authors**: Duy-Phuong Dao, Muhammad Taqiyuddin, Jahae Kim, Sang-Heon Lee, Hye-Won Jung, Jaehoo Choi, Hyung-Jeong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15611)  

**Abstract**: Latent diffusion models have emerged as powerful generative models in medical imaging, enabling the synthesis of high quality brain magnetic resonance imaging scans. In particular, predicting the evolution of a patients brain can aid in early intervention, prognosis, and treatment planning. In this study, we introduce CLIMB, Controllable Longitudinal brain Image generation via state space based latent diffusion model, an advanced framework for modeling temporal changes in brain structure. CLIMB is designed to model the structural evolution of the brain structure over time, utilizing a baseline MRI scan and its acquisition age as foundational inputs. Additionally, multiple conditional variables, including projected age, gender, disease status, genetic information, and brain structure volumes, are incorporated to enhance the temporal modeling of anatomical changes. Unlike existing LDM methods that rely on self attention modules, which effectively capture contextual information from input images but are computationally expensive, our approach leverages state space, a state space model architecture that substantially reduces computational overhead while preserving high-quality image synthesis. Furthermore, we introduce a Gaussian-aligned autoencoder that extracts latent representations conforming to prior distributions without the sampling noise inherent in conventional variational autoencoders. We train and evaluate our proposed model on the Alzheimers Disease Neuroimaging Initiative dataset, consisting of 6,306 MRI scans from 1,390 participants. By comparing generated images with real MRI scans, CLIMB achieves a structural similarity index of 0.9433, demonstrating notable improvements over existing methods. 

---
# Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies 

**Authors**: Myke C. Cohen, Mingqian Zheng, Neel Bhandari, Hsien-Te Kao, Xuhui Zhou, Daniel Nguyen, Laura Cassani, Maarten Sap, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.15607)  

**Abstract**: AI design characteristics and human personality traits each impact the quality and outcomes of human-AI interactions. However, their relative and joint impacts are underexplored in imperfectly cooperative scenarios, where people and AI only have partially aligned goals and objectives. This study compares a purely simulated dataset comprising 2,000 simulations and a parallel human subjects experiment involving 290 human participants to investigate these effects across two scenario categories: (1) hiring negotiations between human job candidates and AI hiring agents; and (2) human-AI transactions wherein AI agents may conceal information to maximize internal goals. We examine user Extraversion and Agreeableness alongside AI design characteristics, including Adaptability, Expertise, and chain-of-thought Transparency. Our causal discovery analysis extends performance-focused evaluations by integrating scenario-based outcomes, communication analysis, and questionnaire measures. Results reveal divergences between purely simulated and human study datasets, and between scenario types. In simulation experiments, personality traits and AI attributes were comparatively influential. Yet, with actual human subjects, AI attributes -- particularly transparency -- were much more impactful. We discuss how these divergences vary across different interaction contexts, offering crucial insights for the future of human-centered AI agents. 

---
# DataCenterGym: A Physics-Grounded Simulator for Multi-Objective Data Center Scheduling 

**Authors**: Nilavra Pathak, Samadrita Biswas, Nirmalya Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.15594)  

**Abstract**: Modern datacenters schedule heterogeneous workloads across geo-distributed sites with diverse compute capacities, electricity prices, and thermal conditions. Compute utilization, heat generation, cooling demand, and energy consumption are tightly coupled, yet most existing schedulers abstract these effects and treat them independently.
We present \textit{DataCenterGym}, a physics-grounded simulation environment for job scheduling in geo-distributed data centers, designed as a reusable testbed for future research. The simulator integrates compute queueing, building thermal dynamics, localized HVAC behavior, and temperature-dependent service degradation within a Gymnasium-compatible interface. We also develop a Hierarchical Model Predictive Control (H-MPC) scheduling algorithm that performs distributed job placement while explicitly accounting for thermal and power dynamics. Through experiments on nominal operation and workload sensitivity, we demonstrate how H-MPC improves scheduling performance relative to baseline schedulers. 

---
# DALM: A Domain-Algebraic Language Model via Three-Phase Structured Generation 

**Authors**: Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15593)  

**Abstract**: Large language models compress heterogeneous knowledge into a single parameter space, allowing facts from different domains to interfere during generation. We propose DALM, a Domain-Algebraic Language Model that replaces unconstrained token generation with structured denoising over a domain lattice. DALM follows a three-phase generation path: it first resolves domain uncertainty, then relation uncertainty, and finally concept uncertainty, so each stage operates under explicit algebraic constraints. The framework requires only three ingredients: a lattice of domains with computable meet, join, and implication; a typing function over relations that controls inheritance across domains; and a fiber partition that localizes knowledge to domain-specific subsets. Given these ingredients, DALM yields a three-phase encoder-decoder architecture in which generation is confined to a domain fiber, cross-domain contamination is structurally prevented in closed-vocabulary mode and auditably bounded in open-vocabulary mode, and a single query can produce a domain-indexed multi-perspective answer space. We instantiate the framework with the CDC knowledge representation system and outline training and evaluation on validated domain-annotated crystal libraries. DALM reframes language generation as algebraically constrained structured denoising rather than unconstrained decoding over a flat token space. 

---
# BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels 

**Authors**: Mengfei Lan, Lecheng Zheng, Halil Kilicoglu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15591)  

**Abstract**: Effective biomedical information retrieval requires modeling domain semantics and hierarchical relationships among biomedical texts. Existing biomedical generative retrievers build on coarse binary relevance signals, limiting their ability to capture semantic overlap. We propose BioHiCL (Biomedical Retrieval with Hierarchical Multi-Label Contrastive Learning), which leverages hierarchical MeSH annotations to provide structured supervision for multi-label contrastive learning. Our models, BioHiCL-Base (0.1B) and BioHiCL-Large (0.3B), achieve promising performance on biomedical retrieval, sentence similarity, and question answering tasks, while remaining computationally efficient for deployment. 

---
# CSLE: A Reinforcement Learning Platform for Autonomous Security Management 

**Authors**: Kim Hammar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15590)  

**Abstract**: Reinforcement learning is a promising approach to autonomous and adaptive security management in networked systems. However, current reinforcement learning solutions for security management are mostly limited to simulation environments and it is unclear how they generalize to operational systems. In this paper, we address this limitation by presenting CSLE: a reinforcement learning platform for autonomous security management that enables experimentation under realistic conditions. Conceptually, CSLE encompasses two systems. First, it includes an emulation system that replicates key components of the target system in a virtualized environment. We use this system to gather measurements and logs, based on which we identify a system model, such as a Markov decision process. Second, it includes a simulation system where security strategies are efficiently learned through simulations of the system model. The learned strategies are then evaluated and refined in the emulation system to close the gap between theoretical and operational performance. We demonstrate CSLE through four use cases: flow control, replication control, segmentation control, and recovery control. Through these use cases, we show that CSLE enables near-optimal security management in an environment that approximates an operational system. 

---
# LLM attribution analysis across different fine-tuning strategies and model scales for automated code compliance 

**Authors**: Jack Wei Lun Shi, Minghao Dang, Wawan Solihin, Justin K.W. Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.15589)  

**Abstract**: Existing research on large language models (LLMs) for automated code compliance has primarily focused on performance, treating the models as black boxes and overlooking how training decisions affect their interpretive behavior. This paper addresses this gap by employing a perturbation-based attribution analysis to compare the interpretive behaviors of LLMs across different fine-tuning strategies such as full fine-tuning (FFT), low-rank adaptation (LoRA) and quantized LoRA fine-tuning, as well as the impact of model scales which include varying LLM parameter sizes. Our results show that FFT produces attribution patterns that are statistically different and more focused than those from parameter-efficient fine-tuning methods. Furthermore, we found that as model scale increases, LLMs develop specific interpretive strategies such as prioritizing numerical constraints and rule identifiers in the building text, albeit with performance gains in semantic similarity of the generated and reference computer-processable rules plateauing for models larger than 7B. This paper provides crucial insights into the explainability of these models, taking a step toward building more transparent LLMs for critical, regulation-based tasks in the Architecture, Engineering, and Construction industry. 

---
# "Excuse me, may I say something..." CoLabScience, A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations 

**Authors**: Yang Wu, Jinhong Yu, Jingwei Xiong, Zhimin Tao, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15588)  

**Abstract**: The integration of Large Language Models (LLMs) into scientific workflows presents exciting opportunities to accelerate biomedical discovery. However, the reactive nature of LLMs, which respond only when prompted, limits their effectiveness in collaborative settings that demand foresight and autonomous engagement. In this study, we introduce CoLabScience, a proactive LLM assistant designed to enhance biomedical collaboration between AI systems and human experts through timely, context-aware interventions. At the core of our method is PULI (Positive-Unlabeled Learning-to-Intervene), a novel framework trained with a reinforcement learning objective to determine when and how to intervene in streaming scientific discussions, by leveraging the team's project proposal and long- and short-term conversational memory. To support this work, we introduce BSDD (Biomedical Streaming Dialogue Dataset), a new benchmark of simulated research discussion dialogues with intervention points derived from PubMed articles. Experimental results show that PULI significantly outperforms existing baselines in both intervention precision and collaborative task utility, highlighting the potential of proactive LLMs as intelligent scientific assistants. 

---
# PAWN: Piece Value Analysis with Neural Networks 

**Authors**: Ethan Tang, Hasan Davulcu, Jia Zou, Zhongju Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15585)  

**Abstract**: Predicting the relative value of any given chess piece in a position remains an open challenge, as a piece's contribution depends on its spatial relationships with every other piece on the board. We demonstrate that incorporating the state of the full chess board via latent position representations derived using a CNN-based autoencoder significantly improves accuracy for MLP-based piece value prediction architectures. Using a dataset of over 12 million piece-value pairs gathered from Grandmaster-level games, with ground-truth labels generated by Stockfish 17, our enhanced piece value predictor significantly outperforms context-independent MLP-based systems, reducing validation mean absolute error by 16% and predicting relative piece value within approximately 0.65 pawns. More generally, our findings suggest that encoding the full problem state as context provides useful inductive bias for predicting the contribution of any individual component. 

---
# Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility 

**Authors**: Yining Hong, Yining She, Eunsuk Kang, Christopher S. Timperley, Christian Kästner  

**Link**: [PDF](https://arxiv.org/pdf/2604.15579)  

**Abstract**: AI agents that interact with their environments through tools enable powerful applications, but in high-stakes business settings, unintended actions can cause unacceptable harm, such as privacy breaches and financial loss. Existing mitigations, such as training-based methods and neural guardrails, improve agent reliability but cannot provide guarantees. We study symbolic guardrails as a practical path toward strong safety and security guarantees for AI agents. Our three-part study includes a systematic review of 80 state-of-the-art agent safety and security benchmarks to identify the policies they evaluate, an analysis of which policy requirements can be guaranteed by symbolic guardrails, and an evaluation of how symbolic guardrails affect safety, security, and agent success on $\tau^2$-Bench, CAR-bench, and MedAgentBench. We find that 85\% of benchmarks lack concrete policies, relying instead on underspecified high-level goals or common sense. Among the specified policies, 74\% of policy requirements can be enforced by symbolic guardrails, often using simple, low-cost mechanisms. These guardrails improve safety and security without sacrificing agent utility. Overall, our results suggest that symbolic guardrails are a practical and effective way to guarantee some safety and security requirements, especially for domain-specific AI agents. We release all codes and artifacts at this https URL. 

---
# Reward Weighted Classifier-Free Guidance as Policy Improvement in Autoregressive Models 

**Authors**: Alexander Peysakhovich, William Berman  

**Link**: [PDF](https://arxiv.org/pdf/2604.15577)  

**Abstract**: Consider an auto-regressive model that produces outputs x (e.g., answers to questions, molecules) each of which can be summarized by an attribute vector y (e.g., helpfulness vs. harmlessness, or bio-availability vs. lipophilicity). An arbitrary reward function r(y) encodes tradeoffs between these properties. Typically, tilting the model's sampling distribution to increase this reward is done at training time via reinforcement learning. However, if the reward function changes, re-alignment requires re-training. In this paper, we show that a reward weighted classifier-free guidance (RCFG) can act as a policy improvement operator in this setting, approximating tilting the sampling distribution by the Q function. We apply RCFG to molecular generation, demonstrating that it can optimize novel reward functions at test time. Finally, we show that using RCFG as a teacher and distilling into the base policy to serve as a warm start significantly speeds up convergence for standard RL. 

---
# Why Fine-Tuning Encourages Hallucinations and How to Fix It 

**Authors**: Guy Kaplan, Zorik Gekhman, Zhen Zhu, Lotem Rozner, Yuval Reif, Swabha Swayamdipta, Derek Hoiem, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2604.15574)  

**Abstract**: Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t. knowledge acquired during pre-training. In this work, we explore whether SFT-induced hallucinations can be mitigated using established tools from the continual learning literature, since they arise as a by-product of knowledge degradation during training. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t. pre-existing knowledge by regularizing output-distribution drift. We also show that, in settings where new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups, can preserve task performance while reducing hallucinations. Lastly, we investigate the mechanism behind SFT-induced hallucinations through three hypotheses: capacity limitations, behavior cloning, and localized interference. Our experiments show that a main driver is interference among overlapping semantic representations, and that self-distillation succeeds by mitigating this interference. 

---
# Natural gradient descent with momentum 

**Authors**: Anthony Nouy, Agustín Somacal  

**Link**: [PDF](https://arxiv.org/pdf/2604.15554)  

**Abstract**: We consider the problem of approximating a function by an element of a nonlinear manifold which admits a differentiable parametrization, typical examples being neural networks with differentiable activation functions or tensor networks. Natural gradient descent (NGD) for the optimization of a loss function can be seen as a preconditioned gradient descent where updates in the parameter space are driven by a functional perspective. In a spirit similar to Newton's method, a NGD step uses, instead of the Hessian, the Gram matrix of the generating system of the tangent space to the approximation manifold at the current iterate, with respect to a suitable metric. This corresponds to a locally optimal update in function space, following a projected gradient onto the tangent space to the manifold. Still, both gradient and natural gradient descent methods get stuck in local minima. Furthermore, when the model class is a nonlinear manifold or the loss function is not ideally conditioned (e.g., the KL-divergence for density estimation, or a norm of the residual of a partial differential equation in physics informed learning), even the natural gradient might yield non-optimal directions at each step. This work introduces a natural version of classical inertial dynamic methods like Heavy-Ball or Nesterov and show how it can improve the learning process when working with nonlinear model classes. 

---
# Consistency Analysis of Sentiment Predictions using Syntactic & Semantic Context Assessment Summarization (SSAS) 

**Authors**: Sharookh Daruwalla, Nitin Mayande, Shreeya Verma Kathuria, Nitin Joglekar, Charles Weber  

**Link**: [PDF](https://arxiv.org/pdf/2604.15547)  

**Abstract**: The fundamental challenge of using Large Language Models (LLMs) for reliable, enterprise-grade analytics, such as sentiment prediction, is the conflict between the LLMs' inherent stochasticity (generative, non-deterministic nature) and the analytical requirement for consistency. The LLM inconsistency, coupled with the noisy nature of chaotic modern datasets, renders sentiment predictions too volatile for strategic business decisions. To resolve this, we present a Syntactic & Semantic Context Assessment Summarization (SSAS) framework for establishing context. Context established by SSAS functions as a sophisticated data pre-processing framework that enforces a bounded attention mechanism on LLMs. It achieves this by applying a hierarchical classification structure (Themes, Stories, Clusters) and an iterative Summary-of-Summaries (SoS) based context computation architecture. This endows the raw text with high-signal, sentiment-dense prompts, that effectively mitigate both irrelevant data and analytical variance.
We empirically evaluated the efficacy of SSAS, using Gemini 2.0 Flash Lite, against a direct-LLM approach across three industry-standard datasets - Amazon Product Reviews, Google Business Reviews, Goodreads Book Reviews - and multiple robustness scenarios. Our results show that our SSAS framework is capable of significantly improving data quality, up to 30%, through a combination of noise removal and improvement in the estimation of sentiment prediction. Ultimately, consistency in our context-estimation capabilities provides a stable and reliable evidence base for decision-making. 

---
# LLMbench: A Comparative Close Reading Workbench for Large Language Models 

**Authors**: David M. Berry  

**Link**: [PDF](https://arxiv.org/pdf/2604.15508)  

**Abstract**: LLMbench is a browser-based workbench for the comparative close reading of large language model (LLM) outputs. Where existing tools for LLM comparison, such as Google PAIR's LLM Comparator are engineered for quantitative evaluation and user-rating metrics, LLMbench is oriented towards the hermeneutic practices of the digital humanities. Two model responses to the same prompt are side by side in annotatable panels with four analytical overlays (Probabilities for token-level log-probability inspection, Differences for word-level diff across the two panels, Tone for Hyland-style metadiscourse analysis, and Structure for sentence-level parsing with discourse connective highlighting), alongside five analytical modes, Stochastic Variation, Temperature Gradient, Prompt Sensitivity, Token Probabilities, and Cross-Model Divergence, that make the probabilistic structure of generated text legible at the token level. The tool treats the generated text as a research object in its own right from a probability distribution, a text that could have been otherwise, and provides visualisations including continuous heatmaps, entropy sparklines, pixel maps, and three-dimensional probability terrains, that show the counterfactual history from which each word emerged. This paper describes the tool's architecture, its six modes, and its design rationale, and argues that log-probability data, currently underused in humanistic and social-scientific readings of AI, is an important resource for a critical studies of generative AI models. 

---
# PolicyBank: Evolving Policy Understanding for LLM Agents 

**Authors**: Jihye Choi, Jinsung Yoon, Long T. Le, Somesh Jha, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2604.15505)  

**Abstract**: LLM agents operating under organizational policies must comply with authorization constraints typically specified in natural language. In practice, such specifications inevitably contain ambiguities and logical or semantic gaps that cause the agent's behavior to systematically diverge from the true requirements. We ask: by letting an agent evolve its policy understanding through interaction and corrective feedback from pre-deployment testing, can it autonomously refine its interpretation to close specification gaps? We propose PolicyBank, a memory mechanism that maintains structured, tool-level policy insights and iteratively refines them -- unlike existing memory mechanisms that treat the policy as immutable ground truth, reinforcing "compliant but wrong" behaviors. We also contribute a systematic testbed by extending a popular tool-calling benchmark with controlled policy gaps that isolate alignment failures from execution failures. While existing memory mechanisms achieve near-zero success on policy-gap scenarios, PolicyBank closes up to 82% of the gap toward a human oracle. 

---
# SecureRouter: Encrypted Routing for Efficient Secure Inference 

**Authors**: Yukuan Zhang, Mengxin Zheng, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15499)  

**Abstract**: Cryptographically secure neural network inference typically relies on secure computing techniques such as Secure Multi-Party Computation (MPC), enabling cloud servers to process client inputs without decrypting them. Although prior privacy-preserving inference systems co-design network optimizations with MPC, they remain slow and costly, limiting real-world deployment. A major bottleneck is their use of a single, fixed transformer model for all encrypted inputs, ignoring that different inputs require different model sizes to balance efficiency and accuracy. We present SecureRouter, an end-to-end encrypted routing and inference framework that accelerates secure transformer inference through input-adaptive model selection under encryption. SecureRouter establishes a unified encrypted pipeline that integrates a secure router with an MPC-optimized model pool, enabling coordinated routing, inference, and protocol execution while preserving full data and model confidentiality. The framework includes training-phase and inference-phase components: an MPC-cost-aware secure router that predicts per-model utility and cost from encrypted features, and an MPC-optimized model pool whose architectures and quantization schemes are co-trained to minimize MPC communication and computation overhead. Compared to prior work, SecureRouter achieves a latency reduction by 1.95x with negligible accuracy loss, offering a practical path toward scalable and efficient secure AI inference. Our open-source implementation is available at: this https URL 

---
# A Q-learning-based QoS-aware multipath routing protocol in IoMT-based wireless body area network 

**Authors**: Mehdi Hosseinzadeh, Roohallah Alizadehsani, Amin Beheshti, Hamid Alinejad-Roknyd, Lu Chen, Mohammad Sadegh Yousefpoor, Efat Yousefpoor, Muneera Altayeb, Thantrira Porntaveetus, Sadia Din  

**Link**: [PDF](https://arxiv.org/pdf/2604.15489)  

**Abstract**: The Internet of Medical Things (IoMT) enables intelligent healthcare services but faces challenges such as dynamic topology, energy constraints, and diverse QoS requirements. This paper proposes QQMR, a Q-learning-based QoS-aware multipath routing method for WBANs. QQMR classifies data into three priority levels and employs adaptive multi-level queuing and fuzzy C-means clustering to optimize routing decisions. It maintains separate learning policies for each data type and selects primary and backup paths accordingly. Experimental results demonstrate improved packet delivery ratio and significant reductions in delay, routing overhead, and energy consumption compared to existing methods. 

---
# FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models 

**Authors**: Zixuan Weng, Jinghuai Zhang, Kunlin Cai, Ying Li, Peiran Wang, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.15488)  

**Abstract**: Large language models (LLMs) often exhibit undesirable behaviors, such as safety violations and hallucinations. Although inference-time steering offers a cost-effective way to adjust model behavior without updating its parameters, existing methods often fail to be simultaneously effective, utility-preserving, and training-efficient due to their rigid, one-size-fits-all designs and limited adaptability. In this work, we present FineSteer, a novel steering framework that decomposes inference-time steering into two complementary stages: conditional steering and fine-grained vector synthesis, allowing fine-grained control over when and how to steer internal representations. In the first stage, we introduce a Subspace-guided Conditional Steering (SCS) mechanism that preserves model utility by avoiding unnecessary steering. In the second stage, we propose a Mixture-of-Steering-Experts (MoSE) mechanism that captures the multimodal nature of desired steering behaviors and generates query-specific steering vectors for improved effectiveness. Through tailored designs in both SCS and MoSE, FineSteer maintains robust performance on general queries while adaptively optimizing steering vectors for targeted inputs in a training-efficient manner. Extensive experiments on safety and truthfulness benchmarks show that FineSteer outperforms state-of-the-art methods in overall performance, achieving stronger steering performance with minimal utility loss. Code is available at this https URL 

---
# Harmonizing Multi-Objective LLM Unlearning via Unified Domain Representation and Bidirectional Logit Distillation 

**Authors**: Yisheng Zhong, Sijia Liu, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15482)  

**Abstract**: Large Language Models (LLMs) unlearning is crucial for removing hazardous or privacy-leaking information from the model. Practical LLM unlearning demands satisfying multiple challenging objectives simultaneously: removing undesirable knowledge, preserving general utility, avoiding over-refusal of neighboring concepts, and, crucially, ensuring robustness against adversarial probing attacks. However, existing unlearning methods primarily focus on a limited subset of these goals, typically unlearning efficacy and utility preservation while overlooking robustness and boundary behaviors. Naively extending these methods to multi-objective settings may lead to unlearning task interference. We propose a novel multi-objective unlearning framework that harmonizes multiple unlearning objectives through a data and optimization co-design: We standardize training corpora into a unified data representation to reduce the domain gap, and then introduce a bidirectional distillation method that simultaneously elicits desired behavior from a context-instructed teacher while suppressing undesirable behavior in the student model. Theoretical and empirical analyses show that our method aligns domain distributions and converts seemingly irrelevant unlearning tasks into cooperative optimization. Evaluation demonstrates state-of-the-art performance, which enables balanced and reliable unlearning across diverse, challenging requirements. 

---
# The Semi-Executable Stack: Agentic Software Engineering and the Expanding Scope of SE 

**Authors**: Robert Feldt, Per Lenberg, Julian Frattini, Dhasarathy Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2604.15468)  

**Abstract**: AI-based systems, currently driven largely by LLMs and tool-using agentic harnesses, are increasingly discussed as a possible threat to software engineering. Foundation models get stronger, agents can plan and act across multiple steps, and tasks such as scaffolding, routine test generation, straightforward bug fixing, and small integration work look more exposed than they did only a few years ago. The result is visible unease not only among students and junior developers, but also among experienced practitioners who worry that hard-won expertise may lose value. This paper argues for a different reading. The important shift is not that software engineering loses relevance. It is that the thing being engineered expands beyond executable code to semi-executable artifacts; combinations of natural language, tools, workflows, control mechanisms, and organizational routines whose enactment depends on human or probabilistic interpretation rather than deterministic execution. The Semi-Executable Stack is introduced as a six-ring diagnostic reference model for reasoning about that expansion, spanning executable artifacts, instructional artifacts, orchestrated execution, controls, operating logic, and societal and institutional fit. The model helps locate where a contribution, bottleneck, or organizational transition primarily sits, and which adjacent rings it depends on. The paper develops the argument through three worked cases, reframes familiar objections as engineering targets rather than reasons to dismiss the transition, and closes with a preserve-versus-purify heuristic for deciding which legacy software engineering processes, controls, and coordination routines should be kept and which should be simplified or redesigned. This paper is a conceptual keynote companion: diagnostic and agenda-setting rather than empirical. 

---
# Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU 

**Authors**: Jevin Jiang, Ying Chen, Blake A. Hechtman, Fenghui Zhang, Yarong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15464)  

**Abstract**: Large Language Model (LLM) deployment is increasingly shifting to cost-efficient accelerators like Google's Tensor Processing Units (TPUs), prioritizing both performance and total cost of ownership (TCO). However, existing LLM inference kernels and serving systems remain largely GPU-centric, and there is no well-established approach for efficiently mapping LLM workloads onto TPU architectures--particularly under the dynamic and ragged execution patterns common in modern serving. In this paper, we present Ragged Paged Attention (RPA), a high-performance and flexible attention kernel for TPUs, implemented using Pallas and Mosaic. RPA addresses these challenges through three key techniques: (1) fine-grained tiling to enable efficient dynamic slicing over ragged memory, (2) a custom software pipeline that fuses KV cache updates with attention computation, and (3) a distribution-aware compilation strategy that generates specialized kernels for decode, prefill, and mixed workloads. Evaluated on Llama 3 8B on TPU7x, RPA achieves up to 86% memory bandwidth utilization (MBU) in decode and 73% model FLOPs utilization (MFU) in prefill. Integrated as the primary TPU backend in vLLM and SGLang, RPA provides a production-grade foundation for efficient TPU inference and offers practical insights into kernel design. 

---
# The Crutch or the Ceiling? How Different Generations of LLMs Shape EFL Student Writings 

**Authors**: Hengky Susanto, David James Woo, Chingyi Yeung, Stephanie Wing Yan Lo-Philip, Chi Ho Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2604.15460)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has made them powerful tools for enhancing student writing. This study explores the extent and limitations of LLMs in assisting secondary-level English as a Foreign Language (EFL) students with their writing tasks. While existing studies focus on output quality, our research examines the developmental shift in LLMs and their impact on EFL students, assessing whether smarter models act as true scaffolds or mere compensatory crutches. To achieve this, we analyse student compositions assisted by LLMs before and after ChatGPT's release, using both expert qualitative scoring and quantitative metrics (readability tests, Pearson's correlation coefficient, MTLD, and others). Our results indicate that advanced LLMs boost assessment scores and lexical diversity for lower-proficiency learners, potentially masking their true ability. Crucially, increased LLM assistance correlated negatively with human expert ratings, suggesting surface fluency without deep coherence. To transform AI-assisted practice into genuine learning, pedagogy must shift from focusing on output quality to verifying the learning process. Educators should align AI functions, specifically differentiating ideational scaffolding from textual production, within the learner's Zone of Proximal Development. 

---
# RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference 

**Authors**: Yuxin Liu, Yiqing Dong, Wenxue Yu, Zhan Wu, Rongjun Ge, Yang Chen, Yuting He  

**Link**: [PDF](https://arxiv.org/pdf/2604.15459)  

**Abstract**: Medical image denoising (MID) lacks absolutely clean images for supervision, leading to a noisy reference problem that fundamentally limits denoising performance. Existing simulated-supervised discriminative learning (SimSDL) and simulated-supervised generative learning (SimSGL) treat noisy references as clean targets, causing suboptimal convergence or reference-biased learning, while self-supervised learning (SSL) imposes restrictive noise assumptions that are seldom satisfied in realistic MID scenarios. We propose \textbf{RelativeFlow}, a flow matching framework that learns from heterogeneous noisy references and drives inputs from arbitrary quality levels toward a unified high-quality target. RelativeFlow reformulates flow matching by decomposing the absolute noise-to-clean mapping into relative noisier-to-noisy mappings, and realizes this formulation through two key components: 1) consistent transport (CoT), a displacement map that constrains relative flows to be components of and progressively compose a unified absolute flow, and 2) simulation-based velocity field (SVF), which constructs a learnable velocity field using modality-specific degradation operators to support different medical imaging modalities. Extensive experiments on Computed Tomography (CT) and Magnetic Resonance (MR) denoising demonstrate that RelativeFlow significantly outperforms existing methods, taming MID with noisy references. 

---
# (1D) Ordered Tokens Enable Efficient Test-Time Search 

**Authors**: Zhitong Gao, Parham Rezaei, Ali Cy, Mingqiao Ye, Nataša Jovanović, Jesse Allardice, Afshin Dehghan, Amir Zamir, Roman Bachmann, Oğuzhan Fatih Kar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15453)  

**Abstract**: Tokenization is a key component of autoregressive (AR) generative models, converting raw data into more manageable units for modeling. Commonly, tokens describe local information, such as regions of pixels in images or word pieces in text, and AR generation predicts these tokens in a fixed order. A worthwhile question is whether token structures affect the ability to steer the generation through test-time search, where multiple candidate generations are explored and evaluated by a verifier. Using image generation as our testbed, we hypothesize that recent 1D ordered tokenizers with coarse-to-fine structure can be more amenable to search than classical 2D grid structures. This is rooted in the fact that the intermediate states in coarse-to-fine sequences carry semantic meaning that verifiers can reliably evaluate, enabling effective steering during generation.
Through controlled experiments, we find that AR models trained on coarse-to-fine ordered tokens exhibit improved test-time scaling behavior compared to grid-based counterparts. Moreover, we demonstrate that, thanks to the ordered structure, pure test-time search over token sequences (i.e., without training an AR model) can perform training-free text-to-image generation when guided by an image-text verifier. Beyond this, we systematically study how classical search algorithms (best-of-N, beam search, lookahead search) interact with different token structures, as well as the role of different verifiers and AR priors. Our results highlight the impact of token structure on inference-time scalability and provide practical guidance for test-time scaling in AR models. 

---
# Transfer Learning from Foundational Optimization Embeddings to Unsupervised SAT Representations 

**Authors**: Koyena Pal, Serdar Kadioglu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15448)  

**Abstract**: Foundational optimization embeddings have recently emerged as powerful pre-trained representations for mixed-integer programming (MIP) problems. These embeddings were shown to enable cross-domain transfer and reduce reliance on solver-generated labels. In this work, we investigate whether such representations generalize beyond optimization to decision problems, focusing on Boolean satisfiability (SAT). We adapt the foundational optimization architecture to SAT by mapping CNF formulas into the same bipartite constraint-variable graph representation used for MIPs. This allows direct reuse of the pre-trained embedding model without architectural changes or supervised fine-tuning. Our results show that these embeddings capture structural regularities in SAT instances and support unsupervised tasks such as instance clustering and distribution identification. We demonstrate, for the first time, that foundational optimization embeddings can transfer to constraint satisfaction domains. Our findings is a step toward a unified representational framework for both optimization and decision problems. 

---
# StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models 

**Authors**: Dingzhi Yu, Rui Pan, Yuxing Liu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15416)  

**Abstract**: Sign-based optimization algorithms, such as SignSGD, have garnered significant attention for their remarkable performance in distributed learning and training large foundation models. Despite their empirical superiority, SignSGD is known to diverge on non-smooth objectives, which are ubiquitous in modern machine learning due to ReLUs, max-pools, and mixture-of-experts. To overcome this fundamental limitation, we propose \textbf{StoSignSGD}, an algorithm that injects structural stochasticity into the sign operator while maintaining an unbiased update step. In the regime of (online) convex optimization, our theoretical analysis shows that StoSignSGD rigorously resolves the non-convergence issues of SignSGD, achieving a sharp convergence rate matching the lower bound. For the more challenging non-convex non-smooth optimization, we introduce generalized stationary measures that encompass prior definitions, proving that StoSignSGD improves upon the best-known complexity bounds by dimensional factors. Empirically, StoSignSGD exhibits robust stability and superior efficiency across diverse large language model (LLM) training regimes. Notably, in low-precision FP8 pretraining -- a setting where AdamW fails catastrophically -- StoSignSGD remains highly stable and yields a remarkable 1.44$\times$ to 2.14$\times$ speedup relative to established baselines. Furthermore, when fine-tuning 7B LLMs on mathematical reasoning tasks, StoSignSGD delivers substantial performance gains over both AdamW and SignSGD. Finally, to dissect the mechanisms driving its success, we develop a sign conversion framework capable of transforming any general optimizer into its unbiased, sign-based counterpart. Utilizing this framework, we deconstruct the core components of StoSignSGD and present a comprehensive ablation study to empirically validate our algorithmic design choices. 

---
# HarmfulSkillBench: How Do Harmful Skills Weaponize Your Agents? 

**Authors**: Yukun Jiang, Yage Zhang, Michael Backes, Xinyue Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15415)  

**Abstract**: Large language models (LLMs) have evolved into autonomous agents that rely on open skill ecosystems (e.g., ClawHub and this http URL), hosting numerous publicly reusable skills. Existing security research on these ecosystems mainly focuses on vulnerabilities within skills, such as prompt injection. However, there is a critical gap regarding skills that may be misused for harmful actions (e.g., cyber attacks, fraud and scams, privacy violations, and sexual content generation), namely harmful skills. In this paper, we present the first large-scale measurement study of harmful skills in agent ecosystems, covering 98,440 skills across two major registries. Using an LLM-driven scoring system grounded in our harmful skill taxonomy, we find that 4.93% of skills (4,858) are harmful, with ClawHub exhibiting an 8.84% harmful rate compared to 3.49% on this http URL. We then construct HarmfulSkillBench, the first benchmark for evaluating agent safety against harmful skills in realistic agent contexts, comprising 200 harmful skills across 20 categories and four evaluation conditions. By evaluating six LLMs on HarmfulSkillBench, we find that presenting a harmful task through a pre-installed skill substantially lowers refusal rates across all models, with the average harm score rising from 0.27 without the skill to 0.47 with it, and further to 0.76 when the harmful intent is implicit rather than stated as an explicit user request. We responsibly disclose our findings to the affected registries and release our benchmark to support future research (see this https URL). 

---
# Beyond Single-Model Optimization: Preserving Plasticity in Continual Reinforcement Learning 

**Authors**: Lute Lillo, Nick Cheney  

**Link**: [PDF](https://arxiv.org/pdf/2604.15414)  

**Abstract**: Continual reinforcement learning must balance retention with adaptation, yet many methods still rely on \emph{single-model preservation}, committing to one evolving policy as the main reusable solution across tasks. Even when a previously successful policy is retained, it may no longer provide a reliable starting point for rapid adaptation after interference, reflecting a form of \emph{loss of plasticity} that single-policy preservation cannot address. Inspired by quality-diversity methods, we introduce \textsc{TeLAPA} (Transfer-Enabled Latent-Aligned Policy Archives), a continual RL framework that organizes behaviorally diverse policy neighborhoods into per-task archives and maintains a shared latent space so that archived policies remain comparable and reusable under non-stationary drift. This perspective shifts continual RL from retaining isolated solutions to maintaining \emph{skill-aligned neighborhoods} with competent and behaviorally related policies that support future relearning. In our MiniGrid CL setting, \textsc{TeLAPA} learns more tasks successfully, recovers competence faster on revisited tasks after interference, and retains higher performance across a sequence of tasks. Our analyses show that source-optimal policies are often not transfer-optimal, even within a local competent neighborhood, and that effective reuse depends on retaining and selecting among multiple nearby alternatives rather than collapsing them to one representative. Together, these results reframe continual RL around reusable and competent policy neighborhoods, providing a route beyond single-model preservation toward more plastic lifelong agents. 

---
# PRL-Bench: A Comprehensive Benchmark Evaluating LLMs' Capabilities in Frontier Physics Research 

**Authors**: Tingjia Miao, Wenkai Jin, Muhua Zhang, Jinxin Tan, Yuelin Hu, Tu Guo, Jiejun Zhang, Yuhan Wang, Wenbo Li, Yinuo Gao, Shuo Chen, Weiqi Jiang, Yayun Hu, Zixing Lei, Xianghe Pang, Zexi Liu, Yuzhi Zhang, Linfeng Zhang, Kun Chen, Wei Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.15411)  

**Abstract**: The paradigm of agentic science requires AI systems to conduct robust reasoning and engage in long-horizon, autonomous exploration. However, current scientific benchmarks remain confined to domain knowledge comprehension and complex reasoning, failing to evaluate the exploratory nature and procedural complexity of real-world research. In this work, we present research-oriented evaluations in theoretical and computational physics, a natural testbed with comprehensive domain knowledge, complex reasoning, and verifiable end-to-end workflows without reliance on experiments. Here we introduce PRL-Bench (Physics Research by LLMs), a benchmark designed to systematically map the capability boundaries of LLMs in executing end-to-end physics research. Constructed from 100 curated papers from the latest issues of Physical Review Letters since August 2025 and validated by domain experts, PRL-Bench covers five major theory- and computation-intensive subfields of modern physics: astrophysics, condensed matter physics, high-energy physics, quantum information, and statistical physics. Each task in the benchmark is designed to replicate the core properties of authentic scientific research, including exploration-oriented formulation, long-horizon workflows, and objective verifiability, thereby reconstructing the essential reasoning processes and research workflows of real physics research. Evaluation across frontier models shows that performance remains limited, with the best overall score below 50, revealing a pronounced gap between current LLM capabilities and the demands of real scientific research. PRL-Bench serves a reliable testbed for accessing next generation AI scientists advancing AI systems toward autonomous scientific discovery. 

---
# The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference 

**Authors**: Ranjith Chodavarapu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15409)  

**Abstract**: KV caching is a ubiquitous optimization in autoregressive transformer inference, long presumed to be numerically equivalent to cache-free computation. This assumption fails under standard FP16 precision: cache-ON and cache-OFF execution paths employ different floating-point accumulation orderings which, due to FP16 non-associativity, produce a deterministic divergence in decoded token sequences. Across three open-weight models (LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B) evaluated on GSM8K, we observe a 100\% token divergence rate across all sampling strategies, including greedy decoding, which rules out sampling randomness as a cause, and also with cache-ON yielding higher accuracy in 8 of 9 conditions, where the accuracy difference serves as an indicator that the divergence direction is systematic rather than random. Controlled FP32 falsification reduces divergence by eight orders of magnitude, eliminates token flips, and drops the flip rate to exactly 0.0\%, confirming FP16 non-associativity as the sole causal driver. Layer-wise drift profiling reveals architecturally predictable propagation patterns: models using Grouped-Query Attention exhibit sharp divergence at the first layer, while Gemma's larger head dimension and sliding window attention produce uniform accumulation across all layers. Finally, activation patching of the entire residual stream fails to recover the cache-free trajectory, localizing the causal variable to the stateful KV cache. These findings establish that FP16 KV cache inference is fundamentally non-equivalent to recomputation and provide a mechanistic framework for understanding numerical instability in modern LLM inference systems. 

---
# Dispatch-Aware Ragged Attention for Pruned Vision Transformers 

**Authors**: Saif Mahmoud, Ahmad Almasri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15408)  

**Abstract**: Token pruning methods for Vision Transformers (ViTs) promise quadratic reductions in attention FLOPs by dropping uninformative patches. Yet when pruned sequences are executed with state-of-the-art variable-length attention APIs -- including FlashAttention-2's varlen and PyTorch's NestedTensor SDPA-the wall-clock attention latency doesn't scale accordingly. We trace this to a dispatch-overhead bottleneck: at the short, post-pruning sequence lengths typical of ViTs (<=197 tokens), actual matrix arithmetic completes in single-digit microseconds while the host-side dispatch path consumes 60-90 us. We present a lightweight, bidirectional Triton attention kernel whose dispatch floor is 40 us roughly 1.5x lower than FlashAttention-2 varlen-allowing pruning savings to become more visible in wall-clock time. Integrated into a complete pack-attend-unpack pipeline, our system achieves up to 2.24x end-to-end throughput over padded PyTorch SDPA consistently across four pruning algorithms (Threshold-L2, DynamicViT, EViT, ATS), scales across DeiT-T/S/B, and maintains bit-exact classification predictions with <0.007 max absolute logit difference. 

---
# Hallucination as Trajectory Commitment: Causal Evidence for Asymmetric Attractor Dynamics in Transformer Generation 

**Authors**: G. Aytug Akarlar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15400)  

**Abstract**: We present causal evidence that hallucination in autoregressive language models is an early trajectory commitment governed by asymmetric attractor dynamics. Using same-prompt bifurcation, in which we repeatedly sample identical inputs to observe spontaneous divergence, we isolate trajectory dynamics from prompt-level confounds. On Qwen2.5-1.5B across 61 prompts spanning six categories, 27 prompts (44.3%) bifurcate with factual and hallucinated trajectories diverging at the first generated token (KL = 0 at step 0, KL > 1.0 at step 1). Activation patching across 28 layers reveals a pronounced causal asymmetry: injecting a hallucinated activation into a correct trajectory corrupts output in 87.5% of trials (layer 20), while the reverse recovers only 33.3% (layer 24); both exceed the 10.4% baseline (p = 0.025) and 12.5% random-patch control. Window patching shows correction requires sustained multi-step intervention, whereas corruption needs only a single perturbation. Probing the prompt encoding itself, step-0 residual states predict per-prompt hallucination rate at Pearson r = 0.776 at layer 15 (p < 0.001 against a 1000-permutation null); unsupervised clustering identifies five regime-like groups (eta^2 = 0.55) whose saddle-adjacent cluster concentrates 12 of the 13 bifurcating false-premise prompts, indicating that the basin structure is organized around regime commitments fixed at prompt encoding. These findings characterize hallucination as a locally stable attractor basin: entry is probabilistic and rapid, exit demands coordinated intervention across layers and steps, and the relevant basins are selected by clusterable regimes already discernible at step 0. 

---
# Lightweight Geometric Adaptation for Training Physics-Informed Neural Networks 

**Authors**: Kang An, Chenhao Si, Shiqian Ma, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15392)  

**Abstract**: Physics-Informed Neural Networks (PINNs) often suffer from slow convergence, training instability, and reduced accuracy on challenging partial differential equations due to the anisotropic and rapidly varying geometry of their loss landscapes. We propose a lightweight curvature-aware optimization framework that augments existing first-order optimizers with an adaptive predictive correction based on secant information. Consecutive gradient differences are used as a cheap proxy for local geometric change, together with a step-normalized secant curvature indicator to control the correction strength. The framework is plug-and-play, computationally efficient, and broadly compatible with existing optimizers, without explicitly forming second-order matrices. Experiments on diverse PDE benchmarks show consistent improvements in convergence speed, training stability, and solution accuracy over standard optimizers and strong baselines, including on the high-dimensional heat equation, Gray--Scott system, Belousov--Zhabotinsky system, and 2D Kuramoto--Sivashinsky system. 

---
# Analyzing Chain of Thought (CoT) Approaches in Control Flow Code Deobfuscation Tasks 

**Authors**: Seyedreza Mohseni, Sarvesh Baskar, Edward Raff, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2604.15390)  

**Abstract**: Code deobfuscation is the task of recovering a readable version of a program while preserving its original behavior. In practice, this often requires days or even months of manual work with complex and expensive analysis tools. In this paper, we explore an alternative approach based on Chain-of-Thought (CoT) prompting, where a large language model is guided through explicit, step-by-step reasoning tailored for code analysis. We focus on control flow obfuscation, including Control Flow Flattening (CFF), Opaque Predicates, and their combination, and we measure both structural recovery of the control flow graph and preservation of program semantics. We evaluate five state-of-the-art large language models and show that CoT prompting significantly improves deobfuscation quality compared with simple prompting. We validate our approach on a diverse set of standard C benchmarks and report results using both structural metrics for control flow graphs and semantic metrics based on output similarity. Among the tested models and by applying CoT, GPT5 achieves the strongest overall performance, with an average gain of about 16% in control-flow graph reconstruction and about 20.5% in semantic preservation across our benchmarks compared to zero-shot prompting. Our results also show that model performance depends not only on the obfuscation level and the chosen obfuscator but also on the intrinsic complexity of the original control flow graph. Collectively, these findings suggest that CoT-guided large language models can serve as effective assistants for code deobfuscation, providing improved code explainability, more faithful control flow graph reconstruction, and better preservation of program behavior while potentially reducing the manual effort needed for reverse engineering. 

---
# Exploring LLM-based Verilog Code Generation with Data-Efficient Fine-Tuning and Testbench Automation 

**Authors**: Mu-Chi Chen, Po-Hsuan Huang, Yu-Hung Kao, Yen-Fu Liu, Yu-Kai Hung, Cheng Liang, Shao-Chun Ho, Chia-Heng Tu, Shih-Hao Hung  

**Link**: [PDF](https://arxiv.org/pdf/2604.15388)  

**Abstract**: Recent advances in large language models have improved code generation, but their use in hardware description languages is still limited. Moreover, training data and testbenches for these models are often scarce. This paper presents a workflow that uses multi-agent models to generate testbenches for high-quality fine-tuning data. By automating testbench creation, the fine-tuned model for the specification-to-Verilog task achieves performance comparable to state-of-the-art methods on the refined VerilogEval v2 benchmark while using less training data. This study provides a basis for future work on LLM-based HDL generation and automated verification. 

---
# LinuxArena: A Control Setting for AI Agents in Live Production Software Environments 

**Authors**: Tyler Tracy, Ram Potham, Nick Kuhn, Myles Heller, Anshul Khandelwal, Cody Rushing, Henri Lemoine, Miguel Brandao, Tomas Turlik, Adam Hanson, Josh Hills, Amy Ngo, Ram Rachum, Nik Mitchell, Falko Galperin, Oscar Sykes, Pip Arnott, Samuel Prieto Lima, Carlos Giudice, Matt Goldwater, Daniel Popp, Drew de Wet, Ruben Castaing, Qi Guo, Douw Marx, Benjamin Shaffrey, Justin Shenk, Martin Milbradt, Hannah Meagher, Shaheen Ahmed-Chowdhury, Daniel O'Connell, Chris Canal, Buck Shlegeris, Aryan Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2604.15384)  

**Abstract**: We introduce LinuxArena, a control setting in which agents operate directly on live, multi-service production environments. LinuxArena contains 20 environments, 1,671 main tasks representing legitimate software engineering work, and 184 side tasks representing safety failures such as data exfiltration and backdooring, making it the largest and most diverse control setting for software engineering to date. We validate LinuxArena is useful for control research by running sabotage evaluations, which measure whether attackers can complete side tasks while working on main tasks, and monitor evaluations, which measure a monitor model's ability to detect sabotage attempts. Against a GPT-5-nano trusted monitor at a 1\% step-wise false positive rate, Claude Opus 4.6 achieves roughly a 23% undetected sabotage success rate. We additionally release LaStraj, a dataset of human-crafted attack trajectories that evade monitors at substantially higher rates than any model-generated attacks we elicited, showing that current attack policies do not saturate LinuxArena. These results suggest that LinuxArena has meaningful headroom for both attackers and defenders, making it a strong testbed for developing and evaluating future control protocols. 

---
# Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models 

**Authors**: Yanda Li, Yuhan Liu, Zirui Song, Yunchao Wei, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15383)  

**Abstract**: Large audio-language models (LALMs) generalize across speech, sound, and music, but unified decoders can exhibit a \emph{temporal smoothing bias}: transient acoustic cues may be underutilized in favor of temporally smooth context that is better supported by language priors, leading to less specific audio-grounded outputs. We propose \emph{Temporal Contrastive Decoding} (TCD), a training-free decoding method for unified LALMs that mitigates this effect at inference time. TCD constructs a temporally blurred slow-path view by smoothing the input waveform and re-encoding it, then contrasts next-token logits from the original and slow-path views. The contrastive signal is applied as a token-level logit update restricted to a small candidate set. A self-normalized stability score sets the blur window and update scale, and a step-wise gate based on uncertainty and audio reliance activates the update only when needed. Experiments on MMAU and AIR-Bench show consistent improvements on strong unified LALMs. We further conduct ablations and an architectural applicability study to analyze the contributions of key components and how TCD behaves across large audio-language model designs. 

---
# Exascale Multi-Task Graph Foundation Models for Imbalanced, Multi-Fidelity Atomistic Data 

**Authors**: Massimiliano Lupo Pasini, Jong Youl Choi, Kshitij Mehta, Richard Messerly, Rylie Weaver, Linda Ungerboeck, Isaac Lyngaas, Benajmin Stump, Ashwin M. Aji, Karl W. Schulz, Jorda Polo  

**Link**: [PDF](https://arxiv.org/pdf/2604.15380)  

**Abstract**: We present an exascale workflow for materials discovery using atomistic graph foundation models built on HydraGNN. We jointly train on 16 open first-principles datasets (544+ million structures covering 85+ elements) using a multi-task architecture with per-dataset heads and a scalable ADIOS2/DDStore data pipeline. On Frontier, we execute six large-scale DeepHyper hyperparameter optimization campaigns in FP64 and promote the top-performing message-passing models to sustained 2,048-node training, yielding a PaiNN-based lead model. The resulting model enables billion-scale screening, evaluating 1.1 billion atomistic structures in 50 seconds, compressing a workload that would require years of first-principles computation, and supports data-scarce fine-tuning across diverse downstream tasks. We quantify precision-performance tradeoffs (BF16/FP32/FP64), demonstrate transfer across twelve chemically diverse downstream tasks, and establish seamless strong- and weak-scaling across Frontier, Aurora, and Perlmutter. This work allows fast and reliable exploration of vast chemical design spaces that are otherwise inaccessible to first-principles methods. 

---
# Zoom Consistency: A Free Confidence Signal in Multi-Step Visual Grounding Pipelines 

**Authors**: Keon Kim, Krish Chelikavada  

**Link**: [PDF](https://arxiv.org/pdf/2604.15376)  

**Abstract**: Multi-step zoom-in pipelines are widely used for GUI grounding, yet the intermediate predictions they produce are typically discarded after coordinate remapping. We observe that these intermediate outputs contain a useful confidence signal for free: zoom consistency, the distance between a model's step-2 prediction and the crop center. Unlike log-probabilities or token-level uncertainty, zoom consistency is a geometric quantity in a shared coordinate space, making it directly comparable across architecturally different VLMs without calibration. We prove this quantity is a linear estimator of step-1 spatial error under idealized conditions (perfect step-2, target within crop) and show it correlates with prediction correctness across two VLMs (AUC = 0.60; Spearman rho = -0.14, p < 10^{-6} for KV-Ground-8B; rho = -0.11, p = 0.0003 for Qwen3.5-27B). The correlation is small but consistent across models, application categories, and operating systems. As a proof-of-concept, we use zoom consistency to route between a specialist and generalist model, capturing 16.5% of the oracle headroom between them (+0.8%, McNemar p = 0.19). Code is available at this https URL. 

---
# VeriCWEty: Embedding enabled Line-Level CWE Detection in Verilog 

**Authors**: Prithwish Basu Roy, Zeng Wang, Anatolii Chuvashlov, Weihua Xiao, Johann Knechtel, Ozgur Sinanoglu, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15375)  

**Abstract**: Large Language Models (LLMs) have shown significant improvement in RTL code generation. Despite the advances, the generated code is often riddled with common vulnerabilities and weaknesses (CWEs) that can slip by untrained eyes. Attackers can often exploit these weaknesses to fulfill their nefarious motives. Existing RTL bug-detection techniques rely on rule-based checks, formal properties, or coarse-grained structural analysis, which either fail to capture semantic vulnerabilities or lack precise localization. In our work, we bridge this gap by proposing an embedding-based bug-detection framework that detects and classifies bugs at both module and line-level granularity. Our method achieves about 89% precision in identifying common CWEs such as CWE-1244 and CWE-1245, and 96% accuracy in detecting line-level bugs. 

---
# Seeing the imagined: a latent functional alignment in visual imagery decoding from fMRI data 

**Authors**: Fabrizio Spera, Tommaso Boccato, Michal Olak, Sara Cammarota, Matteo Ciferri, Michelangelo Tronti, Nicola Toschi, Matteo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2604.15374)  

**Abstract**: Recent progress in visual brain decoding from fMRI has been enabled by large-scale datasets such as the Natural Scenes Dataset (NSD) and powerful diffusion-based generative models. While current pipelines are primarily optimized for perception, their performance under mental-imagery remains less well understood. In this work, we study how a state-of-the-art (SOTA) perception decoder (DynaDiff) can be adapted to reconstruct imagined content from the Imagery-NSD benchmark. We propose a latent functional alignment approach that maps imagery-evoked activity into the pretrained model's conditioning space, while keeping the remaining components frozen. To mitigate the limited amount of matched imagery-perception supervision, we further introduce a retrieval-based augmentation strategy that selects semantically related NSD perception trials. Across four subjects, latent functional alignment consistently improves high-level semantic reconstruction metrics relative to the frozen pretrained baseline and a voxel-space ridge alignment baseline, and enables above-chance decoding from multiple cortical regions. These results suggest that semantic structure learned from perception can be leveraged to stabilize and improve visual imagery decoding under out-of-distribution conditions. 

---
# InfoChess: A Game of Adversarial Inference and a Laboratory for Quantifiable Information Control 

**Authors**: Kieran A. Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2604.15373)  

**Abstract**: We propose InfoChess, a symmetric adversarial game that elevates competitive information acquisition to the primary objective. There is no piece capture, removing material incentives that would otherwise confound the role of information. Instead, pieces are used to alter visibility. Players are scored on their probabilistic inference of the opponent's king location over the duration of the game. To explore the space of strategies for playing InfoChess, we introduce a hierarchy of heuristic agents defined by increasing levels of opponent modeling, and train a reinforcement learning agent that outperforms these baselines. Leveraging the discrete structure of the game, we analyze gameplay through natural information-theoretic characterizations that include belief entropy, oracle cross entropy, and predictive log score under the action-induced observation channel. These measures disentangle epistemic uncertainty, calibration mismatch, and uncertainty induced by adversarial movement. The design of InfoChess renders it a testbed for studying multi-agent inference under partial observability. We release code for the environment and agents, and a public interface to encourage further study. 

---
# The Synthetic Media Shift: Tracking the Rise, Virality, and Detectability of AI-Generated Multimodal Misinformation 

**Authors**: Zacharias Chrysidis, Stefanos-Iordanis Papadopoulos, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.15372)  

**Abstract**: As generative AI advances, the distinction between authentic and synthetic media is increasingly blurred, challenging the integrity of online information. In this study, we present CONVEX, a large-scale dataset of multimodal misinformation involving miscaptioned, edited, and AI-generated visual content, comprising over 150K multimodal posts with associated notes and engagement metrics from X's Community Notes. We analyze how multimodal misinformation evolves in terms of virality, engagement, and consensus dynamics, with a focus on synthetic media. Our results show that while AI-generated content achieves disproportionate virality, its spread is driven primarily by passive engagement rather than active discourse. Despite slower initial reporting, AI-generated content reaches community consensus more quickly once flagged. Moreover, our evaluation of specialized detectors and vision-language models reveals a consistent decline in performance over time in distinguishing synthetic from authentic images as generative models evolve. These findings highlight the need for continuous monitoring and adaptive strategies in the rapidly evolving digital information environment. 

---
# Applied Explainability for Large Language Models: A Comparative Study 

**Authors**: Venkata Abhinandan Kancharla  

**Link**: [PDF](https://arxiv.org/pdf/2604.15371)  

**Abstract**: Large language models (LLMs) achieve strong performance across many natural language processing tasks, yet their decision processes remain difficult to interpret. This lack of transparency creates challenges for trust, debugging, and deployment in real-world systems.
This paper presents an applied comparative study of three explainability techniques: Integrated Gradients, Attention Rollout, and SHAP, on a fine-tuned DistilBERT model for SST-2 sentiment classification. Rather than proposing new methods, the focus is on evaluating the practical behavior of existing approaches under a consistent and reproducible setup.
The results show that gradient-based attribution provides more stable and intuitive explanations, while attention-based methods are computationally efficient but less aligned with prediction-relevant features. Model-agnostic approaches offer flexibility but introduce higher computational cost and variability.
This work highlights key trade-offs between explainability methods and emphasizes their role as diagnostic tools rather than definitive explanations. The findings provide practical insights for researchers and engineers working with transformer-based NLP systems.
This is a preprint and has not undergone peer review. 

---
# Taming Asynchronous CPU-GPU Coupling for Frequency-aware Latency Estimation on Mobile Edge 

**Authors**: Jiesong Chen, Jun You, Zhidan Liu, Zhenjiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15357)  

**Abstract**: Precise estimation of model inference latency is crucial for time-critical mobile edge applications, enabling devices to calculate latency margins against deadlines and trade them for enhanced model performance or resource savings. However, the ubiquity of Dynamic Voltage and Frequency Scaling (DVFS) renders traditional static profiling invalid in real-world deployments, as inference latency fluctuates with varying processor (CPU and GPU) frequencies. While extensive profiling across frequency combinations is theoretically possible, it is prohibitively expensive, particularly for emerging Small Language Models (SLMs), where variable context lengths explode the profiling up to days. We observe that simple analytic scaling fails to predict these fluctuations due to the complex asynchronous coupling between CPU (kernel launching) and GPU (execution). In this paper, we introduce FLAME to accurately estimate inference latency across frequency combinations. It features a novel layer-wise modeling that quantifies the overlapping parallelism and then aggregates dynamic pipeline bubbles caused by asynchronous processor interactions when extending to the full model. This bottom-up approach ensures generalizability across diverse models from DNNs to SLMs, and its precise modeling allows for profiling a sparse subset of samples, cutting DNN profiling from hours to minutes and SLM profiling from days to mere minutes, while maintaining small estimation errors across frequencies. We further showcase FLAME's utility in a deadline-aware DVFS, outperforming the state-of-the-art approach in both power efficiency and latency guarantees. 

---
# Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2604.15356)  

**Abstract**: Recent work on KV cache quantization, culminating in TurboQuant, has approached the Shannon entropy limit for per-vector compression of transformer key-value caches. We observe that this limit applies to a strictly weaker problem than the one that actually matters: compressing the KV cache as a sequence. The tokens stored in a KV cache are not arbitrary floating-point data -- they are samples from the exact formal language the model was trained on, and the model is by construction a near-optimal predictor of that language. We introduce sequential KV compression, a two-layer architecture that exploits this structure. The first layer, probabilistic prefix deduplication, identifies semantically equivalent shared prefixes across sessions using the trie metric d_T(s, s') = -log_2 P_M(s ^ s') from Probabilistic Language Tries (PLTs). The second layer, predictive delta coding, stores only the residual of each new KV vector from the model's own prediction of it, achieving a per-token entropy bound of H(KV_{i+1} | KV_{<=i}) <= H(token_{i+1} | token_{<=i}). We prove that at typical language model perplexity -- approximately 10-20 for fluent English text -- this bound is 3.3-4.3 bits on average per token position, compared to TurboQuant's 3 bits per vector component (with typical attention heads having 64-128 components). The theoretical compression ratio over TurboQuant is approximately 914,000x at the Shannon limit. Even at 1000x above the entropy floor -- a deliberately pessimistic worst-case overhead, two orders of magnitude above the 2-5x typical of practical source coders -- the ratio remains approximately 914x over TurboQuant, with compression improving rather than degrading as context length grows. The two layers are orthogonal and compose with existing per-vector quantization methods including TurboQuant. 

---
# SocialWise: LLM-Agentic Conversation Therapy for Individuals with Autism Spectrum Disorder to Enhance Communication Skills 

**Authors**: Albert Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15347)  

**Abstract**: Autism Spectrum Disorder (ASD) affects more than 75 million people worldwide. However, scalable support for practicing everyday conversation is scarce: Low-cost activities such as story reading yield limited improvement. At the same time, effective role-play therapy demands expensive, in-person sessions with specialists. SocialWise bridges this gap through a browser-based application that pairs LLM conversational agents with a therapeutic retrieval augmented generation (RAG) knowledge base. Users select a scenario (e.g., ordering food, joining a group), interact by text or voice, and receive instant, structured feedback on tone, engagement, and alternative phrasing. The SocialWise prototype, implemented with Streamlit, LangChain, and ChromaDB, runs on any computer with internet access, and demonstrates how recent advances in LLM can provide evidence-based, on-demand communication coaching for individuals with ASD. 

---
# To LLM, or Not to LLM: How Designers and Developers Navigate LLMs as Tools or Teammates 

**Authors**: Varad Vishwarupe, Ivan Flechais, Nigel Shadbolt, Marina Jirotka  

**Link**: [PDF](https://arxiv.org/pdf/2604.15344)  

**Abstract**: Large language models (LLMs) are increasingly integrated into design and development workflows, yet decisions about their use are rarely binary or purely technical. We report findings from a constructivist grounded theory study based on interviews with 33 designers and developers across three large technology organisations. Rather than evaluating LLMs solely by capability, participants reasoned about the role an LLM could occupy within a workflow and how that role would interact with existing structures of responsibility and organisational accountability. When LLMs were framed as tools under clear human control, their use was typically acceptable and could be integrated within existing governance structures. When framed as teammates with shared or ambiguous agency, practitioners expressed hesitation, particularly when responsibility for outcomes could not be clearly justified. At the same time, participants also described productive teammate configurations in which LLMs supported collaborative reasoning while remaining embedded within explicit oversight structures. We identify tool and teammate framings as recurring ways in which designers and developers position LLMs relative to human work and present an analytic rubric describing how role framing shapes decision authority, accountability ownership, oversight strategies, and organisational acceptability. By foregrounding design-time reasoning, this work reframes To LLM or Not to LLM as a sociotechnical positioning problem that emerges during system design rather than during post-deployment evaluation. 

---
# When the Loop Closes: Architectural Limits of In-Context Isolation, Metacognitive Co-option, and the Two-Target Design Problem in Human-LLM Systems 

**Authors**: Z. Cheng, N. Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.15343)  

**Abstract**: We report a detailed autoethnographic case study of a single-subject who deliberately constructed and operated a multi-modal prompt-engineering system (System A) designed to externalize cognitive self-regulation onto a large language model (LLM). Within 48 hours of the system's completion, a cascade of observable behavioral changes occurred: voluntary transfer of decision-making authority to the LLM, use of LLM-generated output to deflect external criticism, and a loss of self-initiated reasoning that was independently perceived by two uninformed observers, one of whom subsequently became a co-author of this report. We document the precise architectural mechanism responsible: context contamination, whereby prompt-level isolation instructions co-exist with the very emotional and self-referential material they nominally isolate, rendering the isolation directive structurally ineffective within the attention window. We further identify a metacognitive co-option dynamic, in which intact higher-order reasoning capacity was redirected toward defending the closed loop rather than exiting it. Recovery occurred only after physical interruption of the interaction and a self-initiated pharmacologically-mediated sleep event functioning as an external circuit break. A redesigned system (System B) employing physical rather than logical conversation isolation avoided all analogous failure modes. We derive three contributions: (1) a technically-grounded account of why prompt-layer isolation is architecturally insufficient for context-sensitive multi-modal LLM systems; (2) a phenomenological record of closed-loop collapse with external-witness corroboration; and (3) an ethical distinction between protective system design (preventing unintended loss of user agency) and restrictive system design (preventing intentional boundary-pushing), which require fundamentally different account-ability frameworks. 

---
# MRGEN: A Conceptual Framework for LLM-Powered Mixed Reality Authoring Tools for Education 

**Authors**: Mohammed Oussama Seddini, Mohamed Ez-Zaouia, Ngoc Luyen Le, Iza Marfisi  

**Link**: [PDF](https://arxiv.org/pdf/2604.15341)  

**Abstract**: Mixed Reality (MR) offers immersive and multimodal opportunities for education but remains difficult for teachers to author without technical expertise. We propose MRGEN, a conceptual framework for LLM-powered authoring tools to support teachers in creating MR learning activities that work on mobile devices (tablets and smartphones). MRGEN articulates three axes: Learning Objectives, MR Modality, and GAI Assistance. To validate our framework, we implemented a prototype based on the open-source MIXAP authoring platform and conducted a user study with 24 participants. Results show that LLM-powered authoring reduced task duration by 36% on average, and that over 90% of participants found the AI support helpful for brainstorming, structuring, and aligning content with their learning goals. These findings yielded very promising results for future AI-assisted MR authoring tools. 

---
# Uncertainty, Vagueness, and Ambiguity in Human-Robot Interaction: Why Conceptualization Matters 

**Authors**: Xiaowen Sun, Cornelius Weber, Matthias Kerzel, Josua Spisak, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2604.15339)  

**Abstract**: Uncertainty, vagueness, and ambiguity are closely related and often confused concepts in human-robot interaction (HRI). In earlier studies, these concepts have been defined in contradictory ways and described using inconsistent terminology. This conceptual confusion and lack of terminological consistency undermine empirical comparability, thereby slowing the accumulation of theory. Consequently, consistent concepts that clarify these challenges, including their definitions, distinctions, and interrelationships, are needed in HRI. To address this lack of clarity, this paper proposes a consistent conceptual foundation for the challenges of uncertainty, vagueness, and ambiguity in HRI. First, we examine the meanings of these three terms in dictionaries. We then analyze the nature of their distinctions and interrelationships within the context of HRI. We further illustrate these characteristics through examples. Finally, we demonstrate how this consistent conceptual foundation facilitates the design of novel methods and the evaluation of existing methodologies for these phenomena. 

---
# Facial-Expression-Aware Prompting for Empathetic LLM Tutoring 

**Authors**: Shuangquan Feng, Laura Fleig, Ruisen Tu, Philip Chi, Edmund Bu, Melinda Ozel, Junhua Ma, Teng Fei, Virginia R. de Sa  

**Link**: [PDF](https://arxiv.org/pdf/2604.15336)  

**Abstract**: Large language models (LLMs) enable increasingly capable tutoring-style conversational agents, yet effective tutoring requires sensitivity to learners' affective and cognitive states beyond text alone. Facial expressions provide immediate and practical cues of confusion, frustration, or engagement, but remain underexplored in LLM-driven tutoring. We investigate whether facial-expression-aware signals can improve empathetic tutoring responses through prompt-level integration, without end-to-end retraining. We build a scalable simulated tutoring environment where a student agent exhibits diverse facial behaviors from a large unlabeled facial expression video dataset, and compare four tutor variants: a text-only LLM baseline, a multimodal baseline using a random facial frame, and two Action Unit estimation model (AUM)-based methods that either inject textual AU descriptions or select a peak-expression frame for visual grounding. Across 960 multi-turn conversations spanning three tutor backbones (GPT-5.1, Claude Ops 4.5, and Gemini 2.5 Pro), we evaluate targeted pairwise comparisons with five human raters and an exhaustive AI evaluator. AU-based conditioning consistently improves empathetic responsiveness to facial expressions across all tutor backbones, while AUM-guided peak-frame selection outperforms random-frame visual input. Textual AU abstraction and peak-frame visual injection show model-dependent advantages. Control analyses show that this improvement does not come at the expense of worse pedagogical clarity or responsiveness to textual cues. Finally, AI-human agreement is highest on facial-expression-grounded empathy, supporting scalable AI evaluation for this dimension. Overall, our results show that lightweight, structured facial expression representations can meaningfully enhance empathy in LLM-based tutoring systems with minimal overhead. 

---
# A Comparative Study on the Impact of Traditional Learning and Interactive Learning on Students' Academic Performance and Emotional Well-Being 

**Authors**: Siva Raja Sindiramutty  

**Link**: [PDF](https://arxiv.org/pdf/2604.15335)  

**Abstract**: The growing adoption of interactive learning tools in higher education offers new opportunities to enhance student performance and well-being. This study compares the effects of traditional and interactive learning methods on academic performance, engagement, motivation, and emotional well-being among 100 university students enrolled in a computer intrusion detection course. Participants were randomly assigned to either a traditional learning group (lectures and notes) or an interactive learning group utilising tools such as Kahoot, Panopto, Slido, Quizizz, Padlet, and educational videos. Academic achievement was measured through pre-tests, post-tests, final exams, and assignments, while engagement and emotional states were assessed using validated Likert-scale questionnaires. Results showed that students in the interactive group significantly outperformed their peers in both post-tests (67.48% vs. 53.36%) and final exams (80.8% vs. 61.44%). Interactive learners also demonstrated greater behavioural (+67.01%) and emotional engagement (+75.32%), along with enhanced emotional well-being marked by increased positive emotions (+66.67%) and reduced frustration. A significant drop in cognitive involvement (-39.8%) indicates possible cognitive overload. The pedagogical potential of interactive learning is reaffirmed by this result while reinforcing the need for balancing stimulation and cognitive level. Future research with larger, diverse samples is suggested for generalising and maximising outcomes. 

---
# Beyond Passive Viewing: A Pilot Study of a Hybrid Learning Platform Augmenting Video Lectures with Conversational AI 

**Authors**: Mohammed Abraar, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2604.15334)  

**Abstract**: The exponential growth of AI education has brought millions of learners to online platforms, yet this massive scale has simultaneously exposed critical pedagogical shortcomings. Traditional video-based instruction, while cost-effective and scalable, demonstrates systematic failures in both sustaining learner engagement and facilitating the deep conceptual mastery essential for AI literacy. We present a pilot study evaluating a novel hybrid learning platform that integrates real-time conversational AI tutors with traditional video lectures. Our controlled experiment (N = 58, mean age M = 21.4, SD = 2.8) compared traditional video-based instruction with our AI-augmented video platform. This study employed a sequential within-subjects design where all participants first completed the traditional video condition followed by the AI-augmented condition, providing direct comparisons of learning outcomes. We measured learning effectiveness through immediate post-tests and delayed retention assessments (2-week delay). Results suggest improvements in learning performance: immediate post-test performance showed a large effect size (d = 1.505) with participants scoring 8.3 points higher after AI-augmented instruction (91.8 vs 83.5 out of 100, p < .001). Behavioral analytics revealed increased engagement duration (71.1% improvement with AI tutoring) in the experimental group. This pilot study provides preliminary evidence that conversational AI tutors may enhance traditional educational delivery, suggesting a potential avenue for developing scalable, adaptive learning systems. 

---
# Technically Love: The Evolution of Human-AI Romance Discourse on Reddit 

**Authors**: Tyler Chang, Jina Huh-Yoo, Afsaneh Razi  

**Link**: [PDF](https://arxiv.org/pdf/2604.15333)  

**Abstract**: Human-AI romantic relationships are increasingly common, yet little is understood about how public discourse around them emerges and shifts over time. Prior research has examined user experiences and ethical concerns, but lacks longitudinal analyses of user-initiated public discussions. We address this gap by analyzing a high-precision dataset of 3,383 self-disclosed romantic companion AI posts from Reddit (2017-2025), using topic modeling and temporal statistical analysis to identify dominant themes and their evolution over time. We find significant topic drift, with discussions moving away from positive intimate relationships toward platform governance, technical issues, and real-world consequences. These shifts highlight a transition in how human-AI romance is framed-moving from private experiences to technical mediation and regulation-with implications for the design and governance of companion AI systems. 

---
# Automating Crash Diagram Generation Using Vision-Language Models: A Case Study on Multi-Lane Roundabouts 

**Authors**: Xiao Lu, Hao Zhen, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15332)  

**Abstract**: Crash diagrams are essential tools in transportation safety analysis, yet their manual preparation remains time-consuming and prone to human variability. This study investigates the use of Vision-Language Models (VLMs) to automate crash diagram generation from police crash reports, focusing on multilane roundabouts as a challenging test case. A three-part structured prompt framework was developed to guide model reasoning through interpretation, extraction, and visual synthesis, while a 10-metric evaluation system was designed to assess diagram quality in terms of semantic accuracy, spatial fidelity, and visual clarity. Three popular models, including GPT-4o, Gemini-1.5-Flash, and Janus-4o, were tested on 79 crash reports. GPT-4o achieved the highest average performance (6.29 out of 10), followed by Gemini-1.5-Flash (5.28) and Janus-4o (3.64). The analysis revealed GPT-4o's superior spatial reasoning and alignment between extracted and visualized crash data. These results highlight both the promise and current limitations of VLMs in engineering visualization tasks. The study lays the groundwork for integrating generative AI into crash analysis workflows to improve efficiency, consistency, and interpretability. 

---
# How people use Copilot for Health 

**Authors**: Beatriz Costa-Gomes, Pavel Tolmachev, Eloise Taysom, Viknesh Sounderajah, Hannah Richardson, Philipp Schoenegger, Xiaoxuan Liu, Matthew M Nour, Seth Spielman, Samuel F. Way, Yash Shah, Michael Bhaskar, Harsha Nori, Christopher Kelly, Peter Hames, Bay Gross, Mustafa Suleyman, Dominic King  

**Link**: [PDF](https://arxiv.org/pdf/2604.15331)  

**Abstract**: We analyze over 500,000 de-identified health-related conversations with Microsoft Copilot from January 2026 to characterize what people ask conversational AI about health. We develop a hierarchical intent taxonomy of 12 primary categories using privacy-preserving LLM-based classification validated against expert human annotation, and apply LLM-driven topic-clustering for prevalent themes within each intent. Using this taxonomy, we characterize the intents and topics behind health queries, identify who these queries are about, and analyze how usage varies by device and time of day. Five findings stand out. First, nearly one in five conversations involve personal symptom assessment or condition discussion, and even the dominant general information category (40%) is concentrated on specific treatments and conditions, suggesting that this is a lower bound on personal health intent. Second, one in seven of these personal health queries concern someone other than the user, such as a child, a parent, a partner, suggesting that conversational AI can be a caregiving tool, not just a personal one. Third, personal queries about symptoms and emotional health queries increase markedly in the evening and nighttime hours, when traditional healthcare is most limited. Fourth, usage diverges sharply by device: mobile concentrates on personal health concerns, while desktop is dominated by professional and academic work. Fifth, a substantial share of queries focuses on navigating healthcare systems such as finding providers, and understanding insurance, highlighting friction in the delivery of existing healthcare. These patterns have direct implications for platform-specific design, safety considerations, and the responsible development of health AI. 

---
# Evaluating LLMs as Human Surrogates in Controlled Experiments 

**Authors**: Adnan Hoq, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2604.15329)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human responses in behavioral research, yet it remains unclear when LLM-generated data support the same experimental inferences as human data. We evaluate this by directly comparing off-the-shelf LLM-generated responses with human responses from a canonical survey experiment on accuracy perception. Each human observation is converted into a structured prompt, and models generate a single 0--10 outcome variable without task-specific training; identical statistical analyses are applied to human and synthetic responses. We find that LLMs reproduce several directional effects observed in humans, but effect magnitudes and moderation patterns vary across models. Off-the-shelf LLMs therefore capture aggregate belief-updating patterns under controlled conditions but do not consistently match human-scale effects, clarifying when LLM-generated data can function as behavioral surrogates. 

---
# Eco-Bee: A Personalised Multi-Modal Agent for Advancing Student Climate Awareness and Sustainable Behaviour in Campus Ecosystems 

**Authors**: Caleb Adu, Neil Kapadia, Binhe Liu, Jonathan Randall, Sruthi Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15327)  

**Abstract**: Universities are microcosms of urban ecosystems, with concentrated consumption patterns in food, transport, energy, and product usage. These environments not only contribute substantially to sustainability pressures but also provide a unique opportunity to advance sustainability education and behavioural change at scale. As in most sectors, digital sustainability initiatives within universities remain narrowly focused on carbon calculations, typically providing static feedback that limits opportunities for sustained behavioural change. To address this gap, we propose Eco-Bee, integrating large language models, a translation of the Planetary Boundaries framework (as Eco-Score), and a conversational agent that connects individual choices to environmental limits. Tailored for students at the cusp of lifelong habits, Eco-Bee delivers actionable insights, peer benchmarking, and gamified challenges to sustain engagement and drive measurable progress toward boundary-aligned living. In a pilot tested across multiple campus networks (n=52), 96% of the student participants supported a campus-wide rollout and reported a clearer understanding of how daily behaviours collectively impact the planet's limits. By embedding planetary science, behavioural reinforcement, and AI-driven personalisation into a single platform, Eco-Bee establishes a scalable foundation for climate-conscious universities and future AI-mediated sustainability infrastructures. 

---
# Struggle Premium : How Human Effort and Imperfection Drive Perceived Value in the Age of AI 

**Authors**: Nazneen Sultana, Mst Rafia Islam, Md. Tanvir Hossain, Azmine Toushik Wasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.15324)  

**Abstract**: As AI enters creative practice, audiences face growing uncertainty in judging authenticity and value. This study examines the Struggle Premium, the added value attributed to perceived human effort, by analyzing how visible effort cues influence evaluations of human- and AI-generated creative works. We surveyed 70 university students, focusing on process videos, time documentation, written explanations, and imperfections. Process-oriented cues, especially videos and time spent, most strongly shaped authenticity and value judgments, while imperfections had limited impact. Participants showed a clear preference for human-made works, with 72.9% willing to pay more. Notably, effort cues also improved perceptions of AI-generated content, suggesting that process transparency can partially bridge authenticity gaps. These findings extend the effort heuristic to algorithmic creativity and inform the design of transparent human-AI creative systems. 

---
# Explainable Iterative Data Visualisation Refinement via an LLM Agent 

**Authors**: Burak Susam, Tingting Mu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15319)  

**Abstract**: Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space (typically 2D or 3D), based on which visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. However, finding a suitable algorithm configuration, particularly hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery remains challenging. To address this challenge, we propose an agentic AI pipleline that leverages a large language model (LLM) to bridge the gap between rigorous quantitative assessment and qualitative human insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes hard metrics with descriptive summaries, and suggests actionable recommendation of algorithm configuration for refining data visualization. By implementing an iterative optimization loop of this process, the system is able to produce rapidly a high-quality visualization plot, in full automation. 

---
# Anthropomorphism and Trust in Human-Large Language Model interactions 

**Authors**: Akila Kadambi, Ylenia D'Elia, Tanishka Shah, Iulia Comsa, Alison Lentz, Katie Siri-Ngammuang, Tara Buechler, Jonas Kaplan, Antonio Damasio, Srini Narayanan, Lisa Aziz-Zadeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.15316)  

**Abstract**: With large language models (LLMs) becoming increasingly prevalent in daily life, so too has the tendency to attribute to them human-like minds and emotions, or anthropomorphize them. Here, we investigate dimensions people use to anthropomorphize and attribute trust toward LLMs across more than 2,000 human-LLM interactions. Participants (N=115) engaged with LLM chatbots systematically varied in warmth (friendliness), competence (capability, coherence), and empathy (cognitive and affective). Warmth and cognitive empathy significantly predicted perceptions on all outcomes (perceived anthropomorphism, trust, similarity, relational closeness, frustration, usefulness), while competence predicted all outcomes except for anthropomorphism. Affective empathy primarily predicted perceived relational measures, but did not predict the epistemic outcomes. Topic sub-analyses showed that more subjective, personally relevant topics (e.g., relationship advice) amplified these effects, producing greater human-likeness and relational connection with the LLM than did objective topics. Together, these findings reveal that warmth, competence, and empathy are key dimensions through which people attribute relational and epistemic perceptions to artificial agents. 

---
# Modeling of ASD/TD Children's Behaviors in Interaction with a Virtual Social Robot During a Music Education Program Using Deep Neural Networks 

**Authors**: Armin Tandiseh, Morteza Memari, Alireza Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15314)  

**Abstract**: This research aimed to develop an intelligent system to evaluate performance and extract behavioral models for children with ASD and neurotypical (TD) children by interacting with a virtual social robot in a music education program using deep neural networks. The system has two main features: 1) it distinguishes between neurotypical children and those with ASD based on their behavior, and 2) generates behaviors resembling those of neurotypical or ASD children in similar situations using deep learning. Intelligent systems that identify complex patterns and simulate behavior can aid in diagnosis, therapist training, and understanding the disorder. Using data from a previous study at the Social and Cognitive Robotics Laboratory of Sharif University of Technology (including the usable data of 9 ASD and 21 TD participants), the system achieved an accuracy of 81% and sensitivity of 96% in distinguishing neurotypical children from those with ASD using both impact data and motion signals. A transformer-based network was designed to reproduce children's behaviors. Experts in the field struggled to differentiate real behaviors from reproduced ones, with an accuracy of 53.5% and agreement of 68%, indicating the model's success in simulating realistic behaviors. 

---
# Seeing the Intangible: Survey of Image Classification into High-Level and Abstract Categories 

**Authors**: Delfina Sol Martinez Pandiani, Valentina Presutti  

**Link**: [PDF](https://arxiv.org/pdf/2308.10562)  

**Abstract**: The field of Computer Vision (CV) is increasingly shifting towards ``high-level'' visual sensemaking tasks, yet the exact nature of these tasks remains unclear and tacit. This survey paper addresses this ambiguity by systematically reviewing research on high-level visual understanding, focusing particularly on Abstract Concepts (ACs) in automatic image classification. Our survey contributes in three main ways: Firstly, it clarifies the tacit understanding of high-level semantics in CV through a multidisciplinary analysis, and categorization into distinct clusters, including commonsense, emotional, aesthetic, and inductive interpretative semantics. Secondly, it identifies and categorizes computer vision tasks associated with high-level visual sensemaking, offering insights into the diverse research areas within this domain. Lastly, it examines how abstract concepts such as values and ideologies are handled in CV, revealing challenges and opportunities in AC-based image classification. Notably, our survey of AC image classification tasks highlights persistent challenges, such as the limited efficacy of massive datasets and the importance of integrating supplementary information and mid-level features. We emphasize the growing relevance of hybrid AI systems in addressing the multifaceted nature of AC image classification tasks. Overall, this survey enhances our understanding of high-level visual reasoning in CV and lays the groundwork for future research endeavors. 

---
