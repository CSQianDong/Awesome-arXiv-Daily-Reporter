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
# ReactBench: A Benchmark for Topological Reasoning in MLLMs on Chemical Reaction Diagrams 

**Authors**: Qiang Xu, Shengyuan Bai, Yu Wang, He Cao, Leqing Chen, Yuanyuan Liu, Bin Feng, Zijing Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15994)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at recognizing individual visual elements and reasoning over simple linear diagrams. However, when faced with complex topological structures involving branching paths, converging flows, and cyclic dependencies, their reasoning capabilities degrade sharply, even on tasks as basic as counting endpoints. Existing benchmarks fail to probe this gap, focusing on semantic comprehension rather than structural reasoning. We introduce ReactBench, a benchmark that reveals fundamental limitations in structural reasoning through chemical reaction diagrams. These real-world scientific diagrams offer an ideal testbed because they naturally span diverse structures from linear chains to cyclic graphs, while requiring both precise local recognition and coherent global reasoning. Our benchmark comprises 1,618 expert-annotated QA pairs across four hierarchical task dimensions. Extensive evaluation across 17 MLLMs reveals a significant performance gap exceeding 30% between anchor-based tasks and holistic structural reasoning tasks. Controlled ablations confirm this bottleneck lies in reasoning, not perception. These findings expose a fundamental deficit in structural understanding and establish directions for advancing visual reasoning. 

---
# SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems 

**Authors**: Hikaru Shindo, Hanzhao Lin, Lukas Helff, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2604.16022)  

**Abstract**: As Large Language Models (LLMs) transition from text processors to autonomous agents, evaluating their social reasoning in embodied multi-agent settings becomes critical. We introduce SocialGrid, an embodied multi-agent environment inspired by Among Us that evaluates LLM agents on planning, task execution, and social reasoning. Our evaluations reveal that even the strongest open model (GPT-OSS-120B) achieves below 60% accuracy in task completion and planning, with agents getting stuck in repetitive behaviors or failing to navigate basic obstacles. Since poor navigation confounds evaluation of social intelligence, SocialGrid offers an optional Planning Oracle to isolate social reasoning from planning deficits. While planning assistance improves task completion, social reasoning remains a bottleneck: agents fail to detect deception at near-random chance regardless of scale, relying on shallow heuristics rather than accumulating behavioral evidence. SocialGrid provides automatic failure analysis and fine-grained metrics, enabling developers to diagnose and improve their agents. We also establish a competitive leaderboard using Elo ratings from adversarial league play. 

---
# Using Large Language Models and Knowledge Graphs to Improve the Interpretability of Machine Learning Models in Manufacturing 

**Authors**: Thomas Bayer, Alexander Lohr, Sarah Weiß, Bernd Michelberger, Wolfram Höpken  

**Link**: [PDF](https://arxiv.org/pdf/2604.16280)  

**Abstract**: Explaining Machine Learning (ML) results in a transparent and user-friendly manner remains a challenging task of Explainable Artificial Intelligence (XAI). In this paper, we present a method to enhance the interpretability of ML models by using a Knowledge Graph (KG). We store domain-specific data along with ML results and their corresponding explanations, establishing a structured connection between domain knowledge and ML insights. To make these insights accessible to users, we designed a selective retrieval method in which relevant triplets are extracted from the KG and processed by a Large Language Model (LLM) to generate user-friendly explanations of ML results. We evaluated our method in a manufacturing environment using the XAI Question Bank. Beyond standard questions, we introduce more complex, tailored questions that highlight the strengths of our approach. We evaluated 33 questions, analyzing responses using quantitative metrics such as accuracy and consistency, as well as qualitative ones such as clarity and usefulness. Our contribution is both theoretical and practical: from a theoretical perspective, we present a novel approach for effectively enabling LLMs to dynamically access a KG in order to improve the explainability of ML results. From a practical perspective, we provide empirical evidence showing that such explanations can be successfully applied in real-world manufacturing environments, supporting better decision-making in manufacturing processes. 

---
# Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.15877)  

**Abstract**: As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of 1,136 references across 22 primary papers reveals a cross-community citation rate below 1%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1,000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level -- none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient -- both communities independently solve shared sub-problems without exchanging solutions -- that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems. 

---
# ASMR-Bench: Auditing for Sabotage in ML Research 

**Authors**: Eric Gan, Aryan Bhatt, Buck Shlegeris, Julian Stastny, Vivek Hebbar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16286)  

**Abstract**: As AI systems are increasingly used to conduct research autonomously, misaligned systems could introduce subtle flaws that produce misleading results while evading detection. We introduce ASMR-Bench (Auditing for Sabotage in ML Research), a benchmark for evaluating the ability of auditors to detect sabotage in ML research codebases. ASMR-Bench consists of 9 ML research codebases with sabotaged variants that produce qualitatively different experimental results. Each sabotage modifies implementation details, such as hyperparameters, training data, or evaluation code, while preserving the high-level methodology described in the paper. We evaluated frontier LLMs and LLM-assisted human auditors on ASMR-Bench and found that both struggled to reliably detect sabotage: the best performance was an AUROC of 0.77 and a top-1 fix rate of 42%, achieved by Gemini 3.1 Pro. We also tested LLMs as red teamers and found that LLM-generated sabotages were weaker than human-generated ones but still sometimes evaded same-capability LLM auditors. We release ASMR-Bench to support research on monitoring and auditing techniques for AI-conducted research. 

---
# MEDLEY-BENCH: Scale Buys Evaluation but Not Control in AI Metacognition 

**Authors**: Farhad Abtahi, Abdolamir Karbalaie, Eduardo Illueca-Fernandez, Fernando Seoane  

**Link**: [PDF](https://arxiv.org/pdf/2604.16009)  

**Abstract**: Metacognition, the ability to monitor and regulate one's own reasoning, remains under-evaluated in AI benchmarking. We introduce MEDLEY-BENCH, a benchmark of behavioural metacognition that separates independent reasoning, private self-revision, and socially influenced revision under genuine inter-model disagreement. The benchmark evaluates 35 models from 12 families on 130 ambiguous instances across five domains and reports two complementary scores: the Medley Metacognition Score (MMS), a tier-based aggregate of reflective updating, social robustness, and epistemic articulation, and the Medley Ability Score (MAS), derived from four metacognitive sub-abilities. Results show a robust evaluation/control dissociation: evaluation ability increases with model size within families, whereas control does not. In a follow-up progressive adversarial analysis of 11 models, we observed two behavioural profiles, i.e., models that revise primarily in response to argument quality and models that track consensus statistics. Under within-model relative profiling (ipsative scoring), evaluation was the weakest relative ability in all 35 models, indicating a systematic knowing/doing gap. Smaller and cheaper models often matched or outperformed larger counterparts, suggesting that metacognitive competence is not simply a function of scale. These findings position MEDLEY-BENCH as a tool for measuring belief revision under social pressure and suggest that future training should reward calibrated, proportional updating rather than output quality alone. 

---
# Learning to Reason with Insight for Informal Theorem Proving 

**Authors**: Yunhe Li, Hao Shi, Bowen Deng, Wei Wang, Mengzhe Ruan, Hanxu Hou, Zhongxiang Dai, Siyang Gao, Chao Wang, Shuang Qiu, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.16278)  

**Abstract**: Although most of the automated theorem-proving approaches depend on formal proof systems, informal theorem proving can align better with large language models' (LLMs) strength in natural language processing. In this work, we identify a primary bottleneck in informal theorem proving as a lack of insight, namely the difficulty of recognizing the core techniques required to solve complex problems. To address this, we propose a novel framework designed to cultivate this essential reasoning skill and enable LLMs to perform insightful reasoning. We propose $\mathtt{DeepInsightTheorem}$, a hierarchical dataset that structures informal proofs by explicitly extracting core techniques and proof sketches alongside the final proof. To fully exploit this dataset, we design a Progressive Multi-Stage SFT strategy that mimics the human learning process, guiding the model from basic proof writing to insightful thinking. Our experiments on challenging mathematical benchmarks demonstrate that this insight-aware generation strategy significantly outperforms baselines. These results demonstrate that teaching models to identify and apply core techniques can substantially improve their mathematical reasoning. 

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
# Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4 

**Authors**: Chengwu Liu, Yichun Yin, Ye Yuan, Jiaxuan Xie, Botao Li, Siqi Li, Jianhao Shen, Yan Xu, Lifeng Shang, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15839)  

**Abstract**: Most ATP benchmarks embed the final answer within the formal statement -- a convention we call "Easy Mode" -- a design that simplifies the task relative to what human competitors face and may lead to optimistic estimates of model capability. We call the stricter, more realistic setting "Hard Mode": the system must independently discover the answer before constructing a formal proof. To enable Hard Mode research, we make two contributions. First, we release MiniF2F-Hard and FIMO-Hard, expert-reannotated Hard Mode variants of two widely-used ATP benchmarks. Second, we introduce Discover And Prove (DAP), an agentic framework that uses LLM natural-language reasoning with explicit self-reflection to discover answers, then rewrites Hard Mode statements into Easy Mode ones for existing ATP provers. DAP sets the state of the art: on CombiBench it raises solved problems from 7 (previous SOTA, Pass@16) to 10; on PutnamBench it is the first system to formally prove 36 theorems in Hard Mode -- while simultaneously revealing that state-of-the-art LLMs exceed 80% answer accuracy on the same problems where formal provers manage under 10%, exposing a substantial gap that Hard Mode benchmarks are uniquely suited to measure. 

---
# LLM Reasoning Is Latent, Not the Chain of Thought 

**Authors**: Wenshuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15726)  

**Abstract**: This position paper argues that large language model (LLM) reasoning should be studied as latent-state trajectory formation rather than as faithful surface chain-of-thought (CoT). This matters because claims about faithfulness, interpretability, reasoning benchmarks, and inference-time intervention all depend on what the field takes the primary object of reasoning to be. We ask what that object should be once three often-confounded factors are separated and formalize three competing hypotheses: H1, reasoning is primarily mediated by latent-state trajectories; H2, reasoning is primarily mediated by explicit surface CoT; and H0, most apparent reasoning gains are better explained by generic serial compute than by any privileged representational object. Reorganizing recent empirical, mechanistic, and survey work under this framework, and adding compute-audited worked exemplars that factorize surface traces, latent interventions, and matched budget expansions, we find that current evidence most strongly supports H1 as a default working hypothesis rather than as a task-independent verdict. We therefore make two recommendations: the field should treat latent-state dynamics as the default object of study for LLM reasoning, and it should evaluate reasoning with designs that explicitly disentangle surface traces, latent states, and serial compute. 

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
# ChemGraph-XANES: An Agentic Framework for XANES Simulation and Analysis 

**Authors**: Vitor F. Grizzi, Thang Duc Pham, Luke N. Pretzie, Jiayi Xu, Murat Keceli, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16205)  

**Abstract**: Computational X-ray absorption near-edge structure (XANES) is widely used to probe local coordination environments, oxidation states, and electronic structure in chemically complex systems. However, the use of computational XANES at scale is constrained more by workflow complexity than by the underlying simulation method itself. To address this challenge, we present ChemGraph-XANES, an agentic framework for automated XANES simulation and analysis that unifies natural-language task specification, structure acquisition, FDMNES input generation, task-parallel execution, spectral normalization, and provenance-aware data curation. Built on ASE, FDMNES, Parsl, and a LangGraph/LangChain-based tool interface, the framework exposes XANES workflow operations as typed Python tools that can be orchestrated by large language model (LLM) agents. In multi-agent mode, a retrieval-augmented expert agent consults the FDMNES manual to ground parameter selection, while executor agents translate user requests into structured tool calls. We demonstrate documentation-grounded parameter retrieval and show that the same workflow supports both explicit structure-file inputs and chemistry-level natural-language requests. Because independent XANES calculations are naturally task-parallel, the framework is well suited for high-throughput deployment on high-performance computing (HPC) systems, enabling scalable XANES database generation for downstream analysis and machine-learning applications. ChemGraph-XANES thus provides a reproducible and extensible workflow layer for physics-based XANES simulation, spectral curation, and agent-compatible computational spectroscopy. 

---
# JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models 

**Authors**: Alexandra Dragomir, Ioana Pintilie, Antonio Barbalau, Marius Dragoi, Florin Brad, Cristian Daniel Paduraru, Alexandru Tifrea, Elena Burceanu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16171)  

**Abstract**: Adapter-based methods have become a cost-effective approach to continual learning (CL) for Large Language Models (LLMs), by sequentially learning a low-rank update matrix for each task. To mitigate catastrophic forgetting, state-of-the-art approaches impose constraints on new adapters with respect to the previous ones, by targeting either subspace or coordinate-wise interference. In this paper, we propose JumpLoRA, a novel framework to adaptively induce sparsity in the Low-Rank Adaptation (LoRA) blocks through the use of JumpReLU gating. The method achieves dynamic parameter isolation, which helps prevent task interference. We demonstrate that our method is highly modular and compatible with LoRA-based CL approaches. Specifically, it significantly boosts the performance of IncLoRA and outperforms the leading state-of-the-art CL method, ELLA. 

---
# The Relic Condition: When Published Scholarship Becomes Material for Its Own Replacement 

**Authors**: Lin Deng, Chang-bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16116)  

**Abstract**: We extracted the scholarly reasoning systems of two internationally prominent humanities and social science scholars from their published corpora alone, converted those systems into structured inference-time constraints for a large language model, and tested whether the resulting scholar-bots could perform core academic functions at expert-assessed quality. The distillation pipeline used an eight-layer extraction method and a nine-module skill architecture grounded in local, closed-corpus analysis. The scholar-bots were then deployed across doctoral supervision, peer review, lecturing and panel-style academic exchange. Expert assessment involved three senior academics producing reports and appointment-level syntheses. Across the preserved expert record, all review and supervision reports judged the outputs benchmark-attaining, appointment-level recommendations placed both bots at or above Senior Lecturer level in the Australian university system, and recovered panel scores placed Scholar A between 7.9 and 8.9/10 and Scholar B between 8.5 and 8.9/10 under multi-turn debate conditions. A research-degree-student survey showed high performance ratings across information reliability, theoretical depth and logical rigor, with pronounced ceiling effects on a 7-point scale, despite all participants already being frontier-model users. We term this the Relic condition: when publication systems make stable reasoning architectures legible, extractable and cheaply deployable, the public record of intellectual labor becomes raw material for its own functional replacement. Because the technical threshold for this transition is already crossed at modest engineering effort, we argue that the window for protective frameworks covering disclosure, consent, compensation and deployment restriction is the present, while deployment remains optional rather than infrastructural. 

---
# AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency 

**Authors**: Max Henning Höth, Kristian Kersting, Björn Deiseroth, Letitia Parcalabescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16158)  

**Abstract**: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks. Yet ensuring that the reasoning trace both contributes to and faithfully reflects the processes underlying the model's final answer, rather than merely accompanying it, remains challenging. We introduce AtManRL, a method that leverages differentiable attention manipulation to learn more faithful reasoning through reinforcement learning. By training an additive attention mask that identifies tokens in the CoT crucial for producing correct answers, we derive a saliency reward signal that encourages the model to generate reasoning traces that genuinely influence its final predictions. We integrate this saliency reward with outcome-based rewards within the GRPO framework to jointly optimize for correctness and interpretability. Experiments on GSM8K and MMLU with Llama-3.2-3B-Instruct demonstrate that our approach can identify influential reasoning tokens and enable training more transparent reasoning models. 

---
# Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs 

**Authors**: Sai Srinivas Kancheti, Aditya Sanjiv Kanade, Vineeth N. Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16060)  

**Abstract**: Multimodal Reasoning Models (MRMs) leveraging Chain-of-Thought (CoT) based thinking have revolutionized mathematical and logical problem-solving. However, we show that this paradigm struggles with generalized spatial intelligence. We perform a comprehensive evaluation of seventeen models across thirteen spatial benchmarks and identify a critical gap: CoT prompting consistently degrades performance in visual spatial reasoning. Furthermore, through a novel No-Image++ ablation, we demonstrate that MRMs and CoT prompted MLMs suffer from severe shortcut learning, and hallucinate visual details from textual priors even when the image is absent. These findings challenge the efficacy of text-only CoT for spatial tasks and underscore the need for vision-centric reasoning paradigms. 

---
# Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs Coding the Interviews of Firearm Violence Survivors 

**Authors**: Jessica H. Zhu, Shayla Stringfield, Vahe Zaprosyan, Michael Wagner, Michel Cukier, Joseph B. Richardson Jr  

**Link**: [PDF](https://arxiv.org/pdf/2604.16132)  

**Abstract**: Firearm violence is a pressing public health issue, yet research into survivors' lived experiences remains underfunded and difficult to scale. Qualitative research, including in-depth interviews, is a valuable tool for understanding the personal and societal consequences of community firearm violence and designing effective interventions. However, manually analyzing these narratives through thematic analysis and inductive coding is time-consuming and labor-intensive. Recent advancements in large language models (LLMs) have opened the door to automating this process, though concerns remain about whether these models can accurately and ethically capture the experiences of vulnerable populations. In this study, we assess the use of open-source LLMs to inductively code interviews with 21 Black men who have survived community firearm violence. Our results demonstrate that while some configurations of LLMs can identify important codes, overall relevance remains low and is highly sensitive to data processing. Furthermore, LLM guardrails lead to substantial narrative erasure. These findings highlight both the potential and limitations of LLM-assisted qualitative coding and underscore the ethical challenges of applying AI in research involving marginalized communities. 

---
# Towards Intrinsic Interpretability of Large Language Models:A Survey of Design Principles and Architectures 

**Authors**: Yutong Gao, Qinglin Meng, Yuan Zhou, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16042)  

**Abstract**: While Large Language Models (LLMs) have achieved strong performance across many NLP tasks, their opaque internal mechanisms hinder trustworthiness and safe deployment. Existing surveys in explainable AI largely focus on post-hoc explanation methods that interpret trained models through external approximations. In contrast, intrinsic interpretability, which builds transparency directly into model architectures and computations, has recently emerged as a promising alternative. This paper presents a systematic review of the recent advances in intrinsic interpretability for LLMs, categorizing existing approaches into five design paradigms: functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparsity induction. We further discuss open challenges and outline future research directions in this emerging field. The paper list is available at: this https URL. 

---
# AgentV-RL: Scaling Reward Modeling with Agentic Verifier 

**Authors**: Jiazheng Zhang, Ziche Fu, Zhiheng Xi, Wenqing Jing, Mingxu Chai, Wei He, Guoqiang Zhang, Chenghao Fan, Chenxin An, Wenxiang Chen, Zhicheng Liu, Haojie Pan, Dingwei Zhu, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16004)  

**Abstract**: Verifiers have been demonstrated to enhance LLM reasoning via test-time scaling (TTS). Yet, they face significant challenges in complex domains. Error propagation from incorrect intermediate reasoning can lead to false positives for seemingly plausible solutions, while lacking external grounding makes verifiers unreliable on computation or knowledge-intensive tasks. To address these challenges, we propose Agentic Verifier, a framework that transforms reward modeling into a multi-turn, tool-augmented deliberative process. We introduce complementary forward and backward agents: one traces solutions from premises to conclusions, while the other re-checks conclusions against their underlying premises. This bidirectional process enables a comprehensive, reliable, and interpretable assessment of solutions. To facilitate practical deployment, we propose AgentV-RL. Through proactive exploration and reinforcement learning, the verifier autonomously interleaves tool-use with internal reasoning. Extensive experiments show that Agentic Verifier yields consistent performance gains under both parallel and sequential TTS. Notably, our 4B variant surpasses state-of-the-art ORMs by 25.2%, positioning it as a promising paradigm for agentic reward modeling. 

---
# SCRIPT: Implementing an Intelligent Tutoring System for Programming in a German University Context 

**Authors**: Alina Deriyeva, Jesper Dannath, Benjamin Paassen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16117)  

**Abstract**: Practice and extensive exercises are essential in programming education. Intelligent tutoring systems (ITSs) are a viable option to provide individualized hints and advice to programming students even when human tutors are not available. However, prior ITS for programming rarely support the Python programming language, mostly focus on introductory programming, and rarely take recent developments in generative models into account. We aim to establish a novel ITS for Python programming that is highly adaptable, serves both as a teaching and research platform, provides interfaces to plug in hint mechanisms (e.g.\ via large language models), and works inside the particularly challenging regulatory environment of Germany, that is, conforming to the European data protection regulation, the European AI act, and ethical framework of the German Research Foundation. In this paper, we present the description of the current state of the ITS along with future development directions, as well as discuss the challenges and opportunities for improving the system. 

---
# Where does output diversity collapse in post-training? 

**Authors**: Constantinos Karouzos, Xingwei Tan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2604.16027)  

**Abstract**: Post-trained language models produce less varied outputs than their base counterparts. This output diversity collapse undermines inference-time scaling methods that rely on varied samples, and risks homogenizing model outputs on creative and value-laden tasks. Prior work attributes collapse to specific post-training methods, without separating the role of training data composition from the method, or the generation format from the model weights. We trace output diversity through three parallel post-training lineages of Olmo 3, Think (chain-of-thought distillation), Instruct (broad multi-source data), and RL-Zero, across 15 tasks and four text diversity metrics. We find that the location of collapse co-varies with data composition: the Think lineage loses most semantic diversity at supervised fine-tuning, and the effect of DPO is larger in Instruct than in Think. Suppressing chain-of-thought reasoning at inference in Think models drops accuracy on hard tasks, yet leaves answer-level diversity unchanged, showing that the collapse is embedded in the model weights by training data, not imposed by the generation format. Decomposing diversity loss on six verifiable tasks into a quality-control component (removal of incorrect outputs) and a residual component (genuine narrowing among correct outputs) reveals that the split is task-dependent, and Think models retain more correct-answer diversity than Instruct despite collapsing more in aggregate. Our results indicate that diversity collapse is determined during training by data composition and cannot be addressed at inference time alone. 

---
# Polarization by Default: Auditing Recommendation Bias in LLM-Based Content Curation 

**Authors**: Nicolò Pagan, Christopher Barrie, Chris Andrew Bail, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.15937)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to curate and rank human-created content, yet the nature and structure of their biases in these tasks remains poorly understood: which biases are robust across providers and platforms, and which can be mitigated through prompt design. We present a controlled simulation study mapping content selection biases across three major LLM providers (OpenAI, Anthropic, Google) on real social media datasets from Twitter/X, Bluesky, and Reddit, using six prompting strategies (\textit{general}, \textit{popular}, \textit{engaging}, \textit{informative}, \textit{controversial}, \textit{neutral}). Through 540,000 simulated top-10 selections from pools of 100 posts across 54 experimental conditions, we find that biases differ substantially in how structural and how prompt-sensitive they are. Polarization is amplified across all configurations, toxicity handling shows a strong inversion between engagement- and information-focused prompts, and sentiment biases are predominantly negative. Provider comparisons reveal distinct trade-offs: GPT-4o Mini shows the most consistent behavior across prompts; Claude and Gemini exhibit high adaptivity in toxicity handling; Gemini shows the strongest negative sentiment preference. On Twitter/X, where author demographics can be inferred from profile bios, political leaning bias is the clearest demographic signal: left-leaning authors are systematically over-represented despite right-leaning authors forming the pool plurality in the dataset, and this pattern largely persists across prompts. 

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
# Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs 

**Authors**: Rohit Sinha, Aditya Kanade, Sai Srinivas Kancheti, Vineeth N Balasubramanian, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16054)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive progress on vision language benchmarks, yet their capacity for visual cognitive and visuospatial reasoning remains less understood. We introduce "Mind's Eye", a multiple-choice benchmark of eight visuo-cognitive tasks inspired by classic human intelligence tests and organized under a novel "A-R-T" taxonomy: Abstraction, Relation, and Transformation. The tasks probe core processes of fluid intelligence such as pattern induction, analogical relation mapping, and mental transformation. We evaluate a diverse suite of closed-source and open-source MLLMs and compare their performance with human participants. Humans achieve 80% accuracy, while top performing MLLMs remain below 50%. Error analysis reveals failures in: (i) visual attention allocation, (ii) internal perceptual manipulation, and (iii) weak abstraction of underlying visual concepts. Our findings suggest that current MLLMs exhibit limited visuospatial reasoning capabilities, when compared with human participants, highlighting the need for more cognitively grounded evaluation frameworks. 

---
# DPrivBench: Benchmarking LLMs' Reasoning for Differential Privacy 

**Authors**: Erchi Wang, Pengrun Huang, Eli Chien, Om Thakkar, Kamalika Chaudhuri, Yu-Xiang Wang, Ruihan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15851)  

**Abstract**: Differential privacy (DP) has a wide range of applications for protecting data privacy, but designing and verifying DP algorithms requires expert-level reasoning, creating a high barrier for non-expert practitioners. Prior works either rely on specialized verification languages that demand substantial domain expertise or remain semi-automated and require human-in-the-loop guidance. In this work, we investigate whether large language models (LLMs) can automate DP reasoning. We introduce DPrivBench, a benchmark in which each instance asks whether a function or algorithm satisfies a stated DP guarantee under specified assumptions. The benchmark is carefully designed to cover a broad range of DP topics, span diverse difficulty levels, and resist shortcut reasoning through trivial pattern matching. Experiments show that while the strongest models handle textbook mechanisms well, all models struggle with advanced algorithms, revealing substantial gaps in current DP reasoning capabilities. Through further analytic study and failure-mode analysis, we identify several promising directions for improving automated DP reasoning. Our benchmark provides a solid foundation for developing and evaluating such methods, and complements existing benchmarks for mathematical reasoning. 

---
# Neurosymbolic Repo-level Code Localization 

**Authors**: Xiufeng Xu, Xiufeng Wu, Zejun Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16021)  

**Abstract**: Code localization is a cornerstone of autonomous software engineering. Recent advancements have achieved impressive performance on real-world issue benchmarks. However, we identify a critical yet overlooked bias: these benchmarks are saturated with keyword references (e.g. file paths, function names), encouraging models to rely on superficial lexical matching rather than genuine structural reasoning. We term this phenomenon the Keyword Shortcut. To address this, we formalize the challenge of Keyword-Agnostic Logical Code Localization (KA-LCL) and introduce KA-LogicQuery, a diagnostic benchmark requiring structural reasoning without any naming hints. Our evaluation reveals a catastrophic performance drop of state-of-the-art approaches on KA-LogicQuery, exposing their lack of deterministic reasoning capabilities. We propose LogicLoc, a novel agentic framework that combines large language models with the rigorous logical reasoning of Datalog for precise localization. LogicLoc extracts program facts from the codebase and leverages an LLM to synthesize Datalog programs, with parser-gated validation and mutation-based intermediate-rule diagnostic feedback to ensure correctness and efficiency. The validated programs are executed by a high-performance inference engine, enabling accurate and verifiable localization in a fully automated, closed-loop workflow. Experimental results demonstrate that LogicLoc significantly outperforms SOTA methods on KA-LogicQuery while maintaining competitive performance on popular issue-driven benchmarks. Notably, LogicLoc attains superior performance with significantly lower token consumption and faster execution by offloading structural traversal to a deterministic engine, reducing the overhead of iterative LLM inference. 

---
# Beyond a Single Frame: Multi-Frame Spatially Grounded Reasoning Across Volumetric MRI 

**Authors**: Lama Moukheiber, Caleb M. Yeung, Haotian Xue, Alec Helbling, Zelin Zhao, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.15808)  

**Abstract**: Spatial reasoning and visual grounding are core capabilities for vision-language models (VLMs), yet most medical VLMs produce predictions without transparent reasoning or spatial evidence. Existing benchmarks also evaluate VLMs on isolated 2D images, overlooking the volumetric nature of clinical imaging, where findings can span multiple frames or appear on only a few slices. We introduce Spatially Grounded MRI Visual Question Answering (SGMRI-VQA), a 41,307-pair benchmark for multi-frame, spatially grounded reasoning on volumetric MRI. Built from expert radiologist annotations in the fastMRI+ dataset across brain and knee studies, each QA pair includes a clinician-aligned chain-of-thought trace with frame-indexed bounding box coordinates. Tasks are organized hierarchically across detection, localization, counting/classification, and captioning, requiring models to jointly reason about what is present, where it is, and across which frames it extends. We benchmark 10 VLMs and show that supervised fine-tuning of Qwen3-VL-8B with bounding box supervision consistently improves grounding performance over strong zero-shot baselines, indicating that targeted spatial supervision is an effective path toward grounded clinical reasoning. 

---
# From Intention to Text: AI-Supported Goal Setting in Academic Writing 

**Authors**: Yueling Fan, Richard Lee Davis, Olga Viberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.15800)  

**Abstract**: This study presents WriteFlow, an AI voice-based writing assistant designed to support reflective academic writing through goal-oriented interaction. Academic writing involves iterative reflection and evolving goal regulation, yet prior research and a formative study with 17 participants show that writers often struggle to articulate and manage changing goals. While commonly used AI writing tools emphasize efficiency, they offer limited support for metacognition and writer agency. WriteFlow frames AI interaction as a dialogic space for ongoing goal articulation, monitoring, and negotiation grounded in writers' intentions. Findings from a Wizard-of-Oz study with 12 expert users show that WriteFlow scaffolds metacognitive regulation and reflection-in-action by supporting iterative goal refinement, maintaining goal-text alignment during drafting, and prompting evaluation of goal fulfillment. We discuss design implications for AI writing systems that prioritize reflective dialogue, flexible goal structures, and multi-perspective feedback to support intentional and agentic writing. 

---
# UniEditBench: A Unified and Cost-Effective Benchmark for Image and Video Editing via Distilled MLLMs 

**Authors**: Lifan Jiang, Tianrun Wu, Yuhang Pei, Chenyang Wang, Boxi Wu, Deng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15871)  

**Abstract**: The evaluation of visual editing models remains fragmented across methods and modalities. Existing benchmarks are often tailored to specific paradigms, making fair cross-paradigm comparisons difficult, while video editing lacks reliable evaluation benchmarks. Furthermore, common automatic metrics often misalign with human preference, yet directly deploying large multimodal models (MLLMs) as evaluators incurs prohibitive computational and financial costs. We present UniEditBench, a unified benchmark for image and video editing that supports reconstruction-based and instruction-driven methods under a shared protocol. UniEditBench includes a structured taxonomy of nine image operations (Add, Remove, Replace, Change, Stroke-based, Extract, Adjust, Count, Reorder) and eight video operations, with coverage of challenging compositional tasks such as counting and spatial reordering. To enable scalable evaluation, we distill a high-capacity MLLM judge (Qwen3-VL-235B-A22B Instruct) into lightweight 4B/8B evaluators that provide multi-dimensional scoring over structural fidelity, text alignment, background consistency, naturalness, and temporal-spatial consistency (for videos). Experiments show that the distilled evaluators maintain strong agreement with human judgments and substantially reduce deployment cost relative to the teacher model. UniEditBench provides a practical and reproducible protocol for benchmarking modern visual editing methods. Our benchmark and the associated reward models are publicly available at this https URL. 

---
# EVIL: Evolving Interpretable Algorithms for Zero-Shot Inference on Event Sequences and Time Series with LLMs 

**Authors**: David Berghaus  

**Link**: [PDF](https://arxiv.org/pdf/2604.15787)  

**Abstract**: We introduce EVIL (\textbf{EV}olving \textbf{I}nterpretable algorithms with \textbf{L}LMs), an approach that uses LLM-guided evolutionary search to discover simple, interpretable algorithms for dynamical systems inference. Rather than training neural networks on large datasets, EVIL evolves pure Python/NumPy programs that perform zero-shot, in-context inference across datasets. We apply EVIL to three distinct tasks: next-event prediction in temporal point processes, rate matrix estimation for Markov jump processes, and time series imputation. In each case, a single evolved algorithm generalizes across all evaluation datasets without per-dataset training (analogous to an amortized inference model). To the best of our knowledge, this is the first work to show that LLM-guided program evolution can discover a single compact inference function for these dynamical-systems problems. Across the three domains, the discovered algorithms are often competitive with, and even outperform, state-of-the-art deep learning models while being orders of magnitudes faster, and remaining fully interpretable. 

---
# Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting 

**Authors**: Chi Liu, Xin Chen, Xu Zhou, Fangbo Tu, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15794)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, underpinning diverse AI applications. However, they often suffer from performance degradation due to factors such as catastrophic forgetting during Supervised Fine-Tuning (SFT), quantization, and pruning. In this work, we introduce a performance recovery framework based on Self-Distillation Fine-Tuning (SDFT) that effectively restores model capabilities. Complementing this practical contribution, we provide a rigorous theoretical explanation for the underlying recovery mechanism. We posit that an LLM's generative capability fundamentally relies on the high-dimensional manifold constructed by its hidden layers. To investigate this, we employ Centered Kernel Alignment (CKA) to quantify the alignment between student and teacher activation trajectories, leveraging its invariance to orthogonal transformations and scaling. Our experiments demonstrate a strong correlation between performance recovery and manifold alignment, substantiating the claim that self-distillation effectively aligns the student's high-dimensional manifold with the optimal structure represented by the teacher. This study bridges the gap between practical recovery frameworks and geometric representation theory, offering new insights into the internal mechanisms of self-distillation. 

---
# Reasoning-targeted Jailbreak Attacks on Large Reasoning Models via Semantic Triggers and Psychological Framing 

**Authors**: Zehao Wang, Lanjun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15725)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong capabilities in generating step-by-step reasoning chains alongside final answers, enabling their deployment in high-stakes domains such as healthcare and education. While prior jailbreak attack studies have focused on the safety of final answers, little attention has been given to the safety of the reasoning process. In this work, we identify a novel problem that injects harmful content into the reasoning steps while preserving unchanged answers. This type of attack presents two key challenges: 1) manipulating the input instructions may inadvertently alter the LRM's final answer, and 2) the diversity of input questions makes it difficult to consistently bypass the LRM's safety alignment mechanisms and embed harmful content into its reasoning process. To address these challenges, we propose the Psychology-based Reasoning-targeted Jailbreak Attack (PRJA) Framework, which integrates a Semantic-based Trigger Selection module and a Psychology-based Instruction Generation module. Specifically, the proposed PRJA automatically selects manipulative reasoning triggers via semantic analysis and leverages psychological theories of obedience to authority and moral disengagement to generate adaptive instructions for enhancing the LRM's compliance with harmful content generation. Extensive experiments on five question-answering datasets demonstrate that PRJA achieves an average attack success rate of 83.6\% against several commercial LRMs, including DeepSeek R1, Qwen2.5-Max, and OpenAI o4-mini. 

---
# CodeMMR: Bridging Natural Language, Code, and Image for Unified Retrieval 

**Authors**: Jiahui Geng, Qing Li, Fengyu Cai, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2604.15663)  

**Abstract**: Code search, framed as information retrieval (IR), underpins modern software engineering and increasingly powers retrieval-augmented generation (RAG), improving code discovery, reuse, and the reliability of LLM-based coding. Yet existing code IR models remain largely text-centric and often overlook the visual and structural aspects inherent in programming artifacts such as web interfaces, data visualizations, SVGs, schematic diagrams, and UML. To bridge this gap, we introduce MMCoIR, the first comprehensive benchmark for evaluating multimodal code IR across five visual domains, eight programming languages, eleven libraries, and show the challenge of the task through extensive evaluation. Therefore, we then propose CodeMMR, a unified retrieval model that jointly embeds natural language, code, and images into a shared semantic space through instruction-based multimodal alignment. CodeMMR achieves strong generalization across modalities and languages, outperforming competitive baselines (e.g., UniIR, GME, VLM2Vec) by an average of 10 points on nDCG@10. Moreover, integrating CodeMMR into RAG enhances code generation fidelity and visual grounding on unseen code generation tasks, underscoring the potential of multimodal retrieval as a core enabler for next-generation intelligent programming systems. Datasets are available at HuggingFace. 

---
# Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking 

**Authors**: Jun Feng, Jiahui Tang, Zhicheng He, Hang Lv, Hongchao Gu, Hao Wang, Xuezhi Yang, Shuai Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15621)  

**Abstract**: Adaptive Retrieval-Augmented Generation aims to mitigate the interference of extraneous noise by dynamically determining the necessity of retrieving supplementary passages. However, as Large Language Models evolve with increasing robustness to noise, the necessity of adaptive retrieval warrants re-evaluation. In this paper, we rethink this necessity and propose AdaRankLLM, a novel adaptive retrieval framework. To effectively verify the necessity of adaptive listwise reranking, we first develop an adaptive ranker employing a zero-shot prompt with a passage dropout mechanism, and compare its generation outcomes against static fixed-depth retrieval strategies. Furthermore, to endow smaller open-source LLMs with this precise listwise ranking and adaptive filtering capability, we introduce a two-stage progressive distillation paradigm enhanced by data sampling and augmentation techniques. Extensive experiments across three datasets and eight LLMs demonstrate that AdaRankLLM consistently achieves optimal performance in most scenarios with significantly reduced context overhead. Crucially, our analysis reveals a role shift in adaptive retrieval: it functions as a critical noise filter for weaker models to overcome their limitations, while serving as a cost-effective efficiency optimizer for stronger reasoning models. 

---
# Just Type It in Isabelle! AI Agents Drafting, Mechanizing, and Generalizing from Human Hints 

**Authors**: Kevin Kappelmann, Maximilian Schäffeler, Lukas Stevens, Mohammad Abdulaziz, Andrei Popescu, Dmitriy Traytel  

**Link**: [PDF](https://arxiv.org/pdf/2604.15713)  

**Abstract**: Type annotations are essential when printing terms in a way that preserves their meaning under reparsing and type inference. We study the problem of complete and minimal type annotations for rank-one polymorphic $\lambda$-calculus terms, as used in Isabelle. Building on prior work by Smolka, Blanchette et al., we give a metatheoretical account of the problem, with a full formal specification and proofs, and formalize it in Isabelle/HOL. Our development is a series of experiments featuring human-driven and AI-driven formalization workflows: a human and an LLM-powered AI agent independently produce pen-and-paper proofs, and the AI agent autoformalizes both in Isabelle, with further human-hinted AI interventions refining and generalizing the development. 

---
# HYPERHEURIST: A Simulated Annealing-Based Control Framework for LLM-Driven Code Generation in Optimized Hardware Design 

**Authors**: Shiva Ahir, Prajna Bhat, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2604.15642)  

**Abstract**: Large Language Models (LLMs) have shown promising progress for generating Register Transfer Level (RTL) hardware designs, largely because they can rapidly propose alternative architectural realizations. However, single-shot LLM generation struggles to consistently produce designs that are both functionally correct and power-efficient. This paper proposes HYPERHEURIST, a simulated annealing-based control framework that treats LLM-generated RTL as intermediate candidates rather than final designs. The suggested system not only focuses on functionality correctness but also on Power-Performance-Area (PPA) optimization. In the first phase, RTL candidates are filtered through compilation, structural checks, and simulation to identify functionally valid designs. PPA optimization is restricted to RTL designs that have already passed compilation and simulation. Evaluated across eight RTL benchmarks, this staged approach yields more stable and repeatable optimization behavior than single-pass LLM-generated RTL. 

---
# GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows 

**Authors**: Jize Wang, Xuanxuan Liu, Yining Li, Songyang Zhang, Yijun Wang, Zifei Shan, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15715)  

**Abstract**: The development of general-purpose agents requires a shift from executing simple instructions to completing complex, real-world productivity workflows. However, current tool-use benchmarks remain misaligned with real-world requirements, relying on AI-generated queries, dummy tools, and limited system-level coordination. To address this, we propose GTA-2, a hierarchical benchmark for General Tool Agents (GTA) spanning atomic tool use and open-ended workflows. Built on real-world authenticity, it leverages real user queries, deployed tools, and multimodal contexts. (i) GTA-Atomic, inherited from our prior GTA benchmark, evaluates short-horizon, closed-ended tool-use precision. (ii) GTA-Workflow introduces long-horizon, open-ended tasks for realistic end-to-end completion. To evaluate open-ended deliverables, we propose a recursive checkpoint-based evaluation mechanism that decomposes objectives into verifiable sub-goals, enabling unified evaluation of both model capabilities and agent execution frameworks (i.e., execution harnesses). Experiments reveal a pronounced capability cliff: while frontier models already struggle on atomic tasks (below 50%), they largely fail on workflows, with top models achieving only 14.39% success. Further analysis shows that checkpoint-guided feedback improves performance, while advanced frameworks such as Manus and OpenClaw substantially enhance workflow completion, highlighting the importance of execution harness design beyond the underlying model capacity. These findings provide guidance for developing reliable personal and professional assistants. Dataset and code will be available at this https URL. 

---
# LLM attribution analysis across different fine-tuning strategies and model scales for automated code compliance 

**Authors**: Jack Wei Lun Shi, Minghao Dang, Wawan Solihin, Justin K.W. Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.15589)  

**Abstract**: Existing research on large language models (LLMs) for automated code compliance has primarily focused on performance, treating the models as black boxes and overlooking how training decisions affect their interpretive behavior. This paper addresses this gap by employing a perturbation-based attribution analysis to compare the interpretive behaviors of LLMs across different fine-tuning strategies such as full fine-tuning (FFT), low-rank adaptation (LoRA) and quantized LoRA fine-tuning, as well as the impact of model scales which include varying LLM parameter sizes. Our results show that FFT produces attribution patterns that are statistically different and more focused than those from parameter-efficient fine-tuning methods. Furthermore, we found that as model scale increases, LLMs develop specific interpretive strategies such as prioritizing numerical constraints and rule identifiers in the building text, albeit with performance gains in semantic similarity of the generated and reference computer-processable rules plateauing for models larger than 7B. This paper provides crucial insights into the explainability of these models, taking a step toward building more transparent LLMs for critical, regulation-based tasks in the Architecture, Engineering, and Construction industry. 

---
# DALM: A Domain-Algebraic Language Model via Three-Phase Structured Generation 

**Authors**: Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15593)  

**Abstract**: Large language models compress heterogeneous knowledge into a single parameter space, allowing facts from different domains to interfere during generation. We propose DALM, a Domain-Algebraic Language Model that replaces unconstrained token generation with structured denoising over a domain lattice. DALM follows a three-phase generation path: it first resolves domain uncertainty, then relation uncertainty, and finally concept uncertainty, so each stage operates under explicit algebraic constraints. The framework requires only three ingredients: a lattice of domains with computable meet, join, and implication; a typing function over relations that controls inheritance across domains; and a fiber partition that localizes knowledge to domain-specific subsets. Given these ingredients, DALM yields a three-phase encoder-decoder architecture in which generation is confined to a domain fiber, cross-domain contamination is structurally prevented in closed-vocabulary mode and auditably bounded in open-vocabulary mode, and a single query can produce a domain-indexed multi-perspective answer space. We instantiate the framework with the CDC knowledge representation system and outline training and evaluation on validated domain-annotated crystal libraries. DALM reframes language generation as algebraically constrained structured denoising rather than unconstrained decoding over a flat token space. 

---
# Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies 

**Authors**: Myke C. Cohen, Mingqian Zheng, Neel Bhandari, Hsien-Te Kao, Xuhui Zhou, Daniel Nguyen, Laura Cassani, Maarten Sap, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.15607)  

**Abstract**: AI design characteristics and human personality traits each impact the quality and outcomes of human-AI interactions. However, their relative and joint impacts are underexplored in imperfectly cooperative scenarios, where people and AI only have partially aligned goals and objectives. This study compares a purely simulated dataset comprising 2,000 simulations and a parallel human subjects experiment involving 290 human participants to investigate these effects across two scenario categories: (1) hiring negotiations between human job candidates and AI hiring agents; and (2) human-AI transactions wherein AI agents may conceal information to maximize internal goals. We examine user Extraversion and Agreeableness alongside AI design characteristics, including Adaptability, Expertise, and chain-of-thought Transparency. Our causal discovery analysis extends performance-focused evaluations by integrating scenario-based outcomes, communication analysis, and questionnaire measures. Results reveal divergences between purely simulated and human study datasets, and between scenario types. In simulation experiments, personality traits and AI attributes were comparatively influential. Yet, with actual human subjects, AI attributes -- particularly transparency -- were much more impactful. We discuss how these divergences vary across different interaction contexts, offering crucial insights for the future of human-centered AI agents. 

---
# "Excuse me, may I say something..." CoLabScience, A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations 

**Authors**: Yang Wu, Jinhong Yu, Jingwei Xiong, Zhimin Tao, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15588)  

**Abstract**: The integration of Large Language Models (LLMs) into scientific workflows presents exciting opportunities to accelerate biomedical discovery. However, the reactive nature of LLMs, which respond only when prompted, limits their effectiveness in collaborative settings that demand foresight and autonomous engagement. In this study, we introduce CoLabScience, a proactive LLM assistant designed to enhance biomedical collaboration between AI systems and human experts through timely, context-aware interventions. At the core of our method is PULI (Positive-Unlabeled Learning-to-Intervene), a novel framework trained with a reinforcement learning objective to determine when and how to intervene in streaming scientific discussions, by leveraging the team's project proposal and long- and short-term conversational memory. To support this work, we introduce BSDD (Biomedical Streaming Dialogue Dataset), a new benchmark of simulated research discussion dialogues with intervention points derived from PubMed articles. Experimental results show that PULI significantly outperforms existing baselines in both intervention precision and collaborative task utility, highlighting the potential of proactive LLMs as intelligent scientific assistants. 

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
# Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility 

**Authors**: Yining Hong, Yining She, Eunsuk Kang, Christopher S. Timperley, Christian Kästner  

**Link**: [PDF](https://arxiv.org/pdf/2604.15579)  

**Abstract**: AI agents that interact with their environments through tools enable powerful applications, but in high-stakes business settings, unintended actions can cause unacceptable harm, such as privacy breaches and financial loss. Existing mitigations, such as training-based methods and neural guardrails, improve agent reliability but cannot provide guarantees. We study symbolic guardrails as a practical path toward strong safety and security guarantees for AI agents. Our three-part study includes a systematic review of 80 state-of-the-art agent safety and security benchmarks to identify the policies they evaluate, an analysis of which policy requirements can be guaranteed by symbolic guardrails, and an evaluation of how symbolic guardrails affect safety, security, and agent success on $\tau^2$-Bench, CAR-bench, and MedAgentBench. We find that 85\% of benchmarks lack concrete policies, relying instead on underspecified high-level goals or common sense. Among the specified policies, 74\% of policy requirements can be enforced by symbolic guardrails, often using simple, low-cost mechanisms. These guardrails improve safety and security without sacrificing agent utility. Overall, our results suggest that symbolic guardrails are a practical and effective way to guarantee some safety and security requirements, especially for domain-specific AI agents. We release all codes and artifacts at this https URL. 

---
# Why Fine-Tuning Encourages Hallucinations and How to Fix It 

**Authors**: Guy Kaplan, Zorik Gekhman, Zhen Zhu, Lotem Rozner, Yuval Reif, Swabha Swayamdipta, Derek Hoiem, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2604.15574)  

**Abstract**: Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t. knowledge acquired during pre-training. In this work, we explore whether SFT-induced hallucinations can be mitigated using established tools from the continual learning literature, since they arise as a by-product of knowledge degradation during training. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t. pre-existing knowledge by regularizing output-distribution drift. We also show that, in settings where new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups, can preserve task performance while reducing hallucinations. Lastly, we investigate the mechanism behind SFT-induced hallucinations through three hypotheses: capacity limitations, behavior cloning, and localized interference. Our experiments show that a main driver is interference among overlapping semantic representations, and that self-distillation succeeds by mitigating this interference. 

---
# FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models 

**Authors**: Zixuan Weng, Jinghuai Zhang, Kunlin Cai, Ying Li, Peiran Wang, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.15488)  

**Abstract**: Large language models (LLMs) often exhibit undesirable behaviors, such as safety violations and hallucinations. Although inference-time steering offers a cost-effective way to adjust model behavior without updating its parameters, existing methods often fail to be simultaneously effective, utility-preserving, and training-efficient due to their rigid, one-size-fits-all designs and limited adaptability. In this work, we present FineSteer, a novel steering framework that decomposes inference-time steering into two complementary stages: conditional steering and fine-grained vector synthesis, allowing fine-grained control over when and how to steer internal representations. In the first stage, we introduce a Subspace-guided Conditional Steering (SCS) mechanism that preserves model utility by avoiding unnecessary steering. In the second stage, we propose a Mixture-of-Steering-Experts (MoSE) mechanism that captures the multimodal nature of desired steering behaviors and generates query-specific steering vectors for improved effectiveness. Through tailored designs in both SCS and MoSE, FineSteer maintains robust performance on general queries while adaptively optimizing steering vectors for targeted inputs in a training-efficient manner. Extensive experiments on safety and truthfulness benchmarks show that FineSteer outperforms state-of-the-art methods in overall performance, achieving stronger steering performance with minimal utility loss. Code is available at this https URL 

---
# Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU 

**Authors**: Jevin Jiang, Ying Chen, Blake A. Hechtman, Fenghui Zhang, Yarong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15464)  

**Abstract**: Large Language Model (LLM) deployment is increasingly shifting to cost-efficient accelerators like Google's Tensor Processing Units (TPUs), prioritizing both performance and total cost of ownership (TCO). However, existing LLM inference kernels and serving systems remain largely GPU-centric, and there is no well-established approach for efficiently mapping LLM workloads onto TPU architectures--particularly under the dynamic and ragged execution patterns common in modern serving. In this paper, we present Ragged Paged Attention (RPA), a high-performance and flexible attention kernel for TPUs, implemented using Pallas and Mosaic. RPA addresses these challenges through three key techniques: (1) fine-grained tiling to enable efficient dynamic slicing over ragged memory, (2) a custom software pipeline that fuses KV cache updates with attention computation, and (3) a distribution-aware compilation strategy that generates specialized kernels for decode, prefill, and mixed workloads. Evaluated on Llama 3 8B on TPU7x, RPA achieves up to 86% memory bandwidth utilization (MBU) in decode and 73% model FLOPs utilization (MFU) in prefill. Integrated as the primary TPU backend in vLLM and SGLang, RPA provides a production-grade foundation for efficient TPU inference and offers practical insights into kernel design. 

---
# PolicyBank: Evolving Policy Understanding for LLM Agents 

**Authors**: Jihye Choi, Jinsung Yoon, Long T. Le, Somesh Jha, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2604.15505)  

**Abstract**: LLM agents operating under organizational policies must comply with authorization constraints typically specified in natural language. In practice, such specifications inevitably contain ambiguities and logical or semantic gaps that cause the agent's behavior to systematically diverge from the true requirements. We ask: by letting an agent evolve its policy understanding through interaction and corrective feedback from pre-deployment testing, can it autonomously refine its interpretation to close specification gaps? We propose PolicyBank, a memory mechanism that maintains structured, tool-level policy insights and iteratively refines them -- unlike existing memory mechanisms that treat the policy as immutable ground truth, reinforcing "compliant but wrong" behaviors. We also contribute a systematic testbed by extending a popular tool-calling benchmark with controlled policy gaps that isolate alignment failures from execution failures. While existing memory mechanisms achieve near-zero success on policy-gap scenarios, PolicyBank closes up to 82% of the gap toward a human oracle. 

---
# Learning Uncertainty from Sequential Internal Dispersion in Large Language Models 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Cong-Duy Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15741)  

**Abstract**: Uncertainty estimation is a promising approach to detect hallucinations in large language models (LLMs). Recent approaches commonly depend on model internal states to estimate uncertainty. However, they suffer from strict assumptions on how hidden states should evolve across layers, and from information loss by solely focusing on last or mean tokens. To address these issues, we present Sequential Internal Variance Representation (SIVR), a supervised hallucination detection framework that leverages token-wise, layer-wise features derived from hidden states. SIVR adopts a more basic assumption that uncertainty manifests in the degree of dispersion or variance of internal representations across layers, rather than relying on specific assumptions, which makes the method model and task agnostic. It additionally aggregates the full sequence of per-token variance features, learning temporal patterns indicative of factual errors and thereby preventing information loss. Experimental results demonstrate SIVR consistently outperforms strong baselines. Most importantly, SIVR enjoys stronger generalisation and avoids relying on large training sets, highlighting the potential for practical deployment. Our code repository is available online at this https URL. 

---
# DepCap: Adaptive Block-Wise Parallel Decoding for Efficient Diffusion LM Inference 

**Authors**: Xiang Xia, Wuyang Zhang, Jiazheng Liu, Cheng Yan, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15750)  

**Abstract**: Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive language generation due to their potential for parallel decoding and global refinement of the entire sequence. To unlock this potential, DLM inference must carefully balance generation quality and decoding speed. Recent block-wise DLM decoding methods improve this trade-off by performing diffusion-based decoding sequentially in blocks. However, existing methods typically rely on fixed block schedules or current-step local signals to determine block boundaries, and use conservative confidence-based parallel decoding to avoid conflicts, limiting the quality-speed trade-off. In this paper, we argue that block-wise DLM inference requires more suitable signals for its two core decisions: cross-step signals for determining block boundaries, and token-level conflict signals for parallel decoding. Based on this view, we propose DepCap, a training-free framework for efficient block-wise DLM inference. Specifically, DepCap instantiates the cross-step signal as the influence of the last decoded block and uses it to adaptively determine how far the next block should extend, while identifying a conflict-free subset of tokens for safe parallel decoding within each block, enabling substantial inference acceleration with negligible quality degradation. DepCap is a plug-and-play method applicable to various DLMs, and compatible with existing KV-cache strategies for block-wise DLM. An information-theoretic analysis further suggests that the cumulative last-block influence on a candidate block is approximately additive across tokens, supporting the proposed block-partitioning criterion. Experimental results show that DepCap achieves favorable speed-quality trade-offs across multiple DLM backbones and reasoning and coding benchmarks, with up to 5.63$\times$ speedup without significant performance degradation. 

---
# Privacy-Preserving LLMs Routing 

**Authors**: Xidong Wu, Yukuan Zhang, Yuqiong Ji, Reza Shirkavand, Qian Lou, Shangqian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15728)  

**Abstract**: Large language model (LLM) routing has emerged as a critical strategy to balance model performance and cost-efficiency by dynamically selecting services from various model providers. However, LLM routing adds an intermediate layer between users and LLMs, creating new privacy risks to user data. These privacy risks have not been systematically studied. Although cryptographic techniques such as Secure Multi-Party Computation (MPC) enable privacy-preserving computation, their protocol design and implementation remain under-explored, and naïve implementations typically incur prohibitive computational overhead. To address this, we propose a privacy-preserving LLM routing framework (PPRoute). PPRoute includes multiple strategies to speed up encoder inference and nearest neighbor search under the MPC and maintain the quality of LLM routing. First, PPRoute uses MPC-friendly operations to boost the encoder inference. Second, PPRoute uses a multiple-step model training algorithm to maintain routing quality despite the constraints of the encrypted domain. Third, PPRoute proposes an unsorted Top-k algorithm with $O(1)$ communication complexity for secure sorting in model search, significantly reducing communication latency. Across different datasets, PPRoute achieves the performance of plaintext counterparts, while achieving approximately a 20$\times$ speedup over naïve MPC implementations. 

---
# Harmonizing Multi-Objective LLM Unlearning via Unified Domain Representation and Bidirectional Logit Distillation 

**Authors**: Yisheng Zhong, Sijia Liu, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15482)  

**Abstract**: Large Language Models (LLMs) unlearning is crucial for removing hazardous or privacy-leaking information from the model. Practical LLM unlearning demands satisfying multiple challenging objectives simultaneously: removing undesirable knowledge, preserving general utility, avoiding over-refusal of neighboring concepts, and, crucially, ensuring robustness against adversarial probing attacks. However, existing unlearning methods primarily focus on a limited subset of these goals, typically unlearning efficacy and utility preservation while overlooking robustness and boundary behaviors. Naively extending these methods to multi-objective settings may lead to unlearning task interference. We propose a novel multi-objective unlearning framework that harmonizes multiple unlearning objectives through a data and optimization co-design: We standardize training corpora into a unified data representation to reduce the domain gap, and then introduce a bidirectional distillation method that simultaneously elicits desired behavior from a context-instructed teacher while suppressing undesirable behavior in the student model. Theoretical and empirical analyses show that our method aligns domain distributions and converts seemingly irrelevant unlearning tasks into cooperative optimization. Evaluation demonstrates state-of-the-art performance, which enables balanced and reliable unlearning across diverse, challenging requirements. 

---
# The Crutch or the Ceiling? How Different Generations of LLMs Shape EFL Student Writings 

**Authors**: Hengky Susanto, David James Woo, Chingyi Yeung, Stephanie Wing Yan Lo-Philip, Chi Ho Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2604.15460)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has made them powerful tools for enhancing student writing. This study explores the extent and limitations of LLMs in assisting secondary-level English as a Foreign Language (EFL) students with their writing tasks. While existing studies focus on output quality, our research examines the developmental shift in LLMs and their impact on EFL students, assessing whether smarter models act as true scaffolds or mere compensatory crutches. To achieve this, we analyse student compositions assisted by LLMs before and after ChatGPT's release, using both expert qualitative scoring and quantitative metrics (readability tests, Pearson's correlation coefficient, MTLD, and others). Our results indicate that advanced LLMs boost assessment scores and lexical diversity for lower-proficiency learners, potentially masking their true ability. Crucially, increased LLM assistance correlated negatively with human expert ratings, suggesting surface fluency without deep coherence. To transform AI-assisted practice into genuine learning, pedagogy must shift from focusing on output quality to verifying the learning process. Educators should align AI functions, specifically differentiating ideational scaffolding from textual production, within the learner's Zone of Proximal Development. 

---
# StoSignSGD: Unbiased Structural Stochasticity Fixes SignSGD for Training Large Language Models 

**Authors**: Dingzhi Yu, Rui Pan, Yuxing Liu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15416)  

**Abstract**: Sign-based optimization algorithms, such as SignSGD, have garnered significant attention for their remarkable performance in distributed learning and training large foundation models. Despite their empirical superiority, SignSGD is known to diverge on non-smooth objectives, which are ubiquitous in modern machine learning due to ReLUs, max-pools, and mixture-of-experts. To overcome this fundamental limitation, we propose \textbf{StoSignSGD}, an algorithm that injects structural stochasticity into the sign operator while maintaining an unbiased update step. In the regime of (online) convex optimization, our theoretical analysis shows that StoSignSGD rigorously resolves the non-convergence issues of SignSGD, achieving a sharp convergence rate matching the lower bound. For the more challenging non-convex non-smooth optimization, we introduce generalized stationary measures that encompass prior definitions, proving that StoSignSGD improves upon the best-known complexity bounds by dimensional factors. Empirically, StoSignSGD exhibits robust stability and superior efficiency across diverse large language model (LLM) training regimes. Notably, in low-precision FP8 pretraining -- a setting where AdamW fails catastrophically -- StoSignSGD remains highly stable and yields a remarkable 1.44$\times$ to 2.14$\times$ speedup relative to established baselines. Furthermore, when fine-tuning 7B LLMs on mathematical reasoning tasks, StoSignSGD delivers substantial performance gains over both AdamW and SignSGD. Finally, to dissect the mechanisms driving its success, we develop a sign conversion framework capable of transforming any general optimizer into its unbiased, sign-based counterpart. Utilizing this framework, we deconstruct the core components of StoSignSGD and present a comprehensive ablation study to empirically validate our algorithmic design choices. 

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
# Analyzing Chain of Thought (CoT) Approaches in Control Flow Code Deobfuscation Tasks 

**Authors**: Seyedreza Mohseni, Sarvesh Baskar, Edward Raff, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2604.15390)  

**Abstract**: Code deobfuscation is the task of recovering a readable version of a program while preserving its original behavior. In practice, this often requires days or even months of manual work with complex and expensive analysis tools. In this paper, we explore an alternative approach based on Chain-of-Thought (CoT) prompting, where a large language model is guided through explicit, step-by-step reasoning tailored for code analysis. We focus on control flow obfuscation, including Control Flow Flattening (CFF), Opaque Predicates, and their combination, and we measure both structural recovery of the control flow graph and preservation of program semantics. We evaluate five state-of-the-art large language models and show that CoT prompting significantly improves deobfuscation quality compared with simple prompting. We validate our approach on a diverse set of standard C benchmarks and report results using both structural metrics for control flow graphs and semantic metrics based on output similarity. Among the tested models and by applying CoT, GPT5 achieves the strongest overall performance, with an average gain of about 16% in control-flow graph reconstruction and about 20.5% in semantic preservation across our benchmarks compared to zero-shot prompting. Our results also show that model performance depends not only on the obfuscation level and the chosen obfuscator but also on the intrinsic complexity of the original control flow graph. Collectively, these findings suggest that CoT-guided large language models can serve as effective assistants for code deobfuscation, providing improved code explainability, more faithful control flow graph reconstruction, and better preservation of program behavior while potentially reducing the manual effort needed for reverse engineering. 

---
# HarmfulSkillBench: How Do Harmful Skills Weaponize Your Agents? 

**Authors**: Yukun Jiang, Yage Zhang, Michael Backes, Xinyue Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15415)  

**Abstract**: Large language models (LLMs) have evolved into autonomous agents that rely on open skill ecosystems (e.g., ClawHub and this http URL), hosting numerous publicly reusable skills. Existing security research on these ecosystems mainly focuses on vulnerabilities within skills, such as prompt injection. However, there is a critical gap regarding skills that may be misused for harmful actions (e.g., cyber attacks, fraud and scams, privacy violations, and sexual content generation), namely harmful skills. In this paper, we present the first large-scale measurement study of harmful skills in agent ecosystems, covering 98,440 skills across two major registries. Using an LLM-driven scoring system grounded in our harmful skill taxonomy, we find that 4.93% of skills (4,858) are harmful, with ClawHub exhibiting an 8.84% harmful rate compared to 3.49% on this http URL. We then construct HarmfulSkillBench, the first benchmark for evaluating agent safety against harmful skills in realistic agent contexts, comprising 200 harmful skills across 20 categories and four evaluation conditions. By evaluating six LLMs on HarmfulSkillBench, we find that presenting a harmful task through a pre-installed skill substantially lowers refusal rates across all models, with the average harm score rising from 0.27 without the skill to 0.47 with it, and further to 0.76 when the harmful intent is implicit rather than stated as an explicit user request. We responsibly disclose our findings to the affected registries and release our benchmark to support future research (see this https URL). 

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
# Hallucination as Trajectory Commitment: Causal Evidence for Asymmetric Attractor Dynamics in Transformer Generation 

**Authors**: G. Aytug Akarlar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15400)  

**Abstract**: We present causal evidence that hallucination in autoregressive language models is an early trajectory commitment governed by asymmetric attractor dynamics. Using same-prompt bifurcation, in which we repeatedly sample identical inputs to observe spontaneous divergence, we isolate trajectory dynamics from prompt-level confounds. On Qwen2.5-1.5B across 61 prompts spanning six categories, 27 prompts (44.3%) bifurcate with factual and hallucinated trajectories diverging at the first generated token (KL = 0 at step 0, KL > 1.0 at step 1). Activation patching across 28 layers reveals a pronounced causal asymmetry: injecting a hallucinated activation into a correct trajectory corrupts output in 87.5% of trials (layer 20), while the reverse recovers only 33.3% (layer 24); both exceed the 10.4% baseline (p = 0.025) and 12.5% random-patch control. Window patching shows correction requires sustained multi-step intervention, whereas corruption needs only a single perturbation. Probing the prompt encoding itself, step-0 residual states predict per-prompt hallucination rate at Pearson r = 0.776 at layer 15 (p < 0.001 against a 1000-permutation null); unsupervised clustering identifies five regime-like groups (eta^2 = 0.55) whose saddle-adjacent cluster concentrates 12 of the 13 bifurcating false-premise prompts, indicating that the basin structure is organized around regime commitments fixed at prompt encoding. These findings characterize hallucination as a locally stable attractor basin: entry is probabilistic and rapid, exit demands coordinated intervention across layers and steps, and the relevant basins are selected by clusterable regimes already discernible at step 0. 

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
# Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2604.15356)  

**Abstract**: Recent work on KV cache quantization, culminating in TurboQuant, has approached the Shannon entropy limit for per-vector compression of transformer key-value caches. We observe that this limit applies to a strictly weaker problem than the one that actually matters: compressing the KV cache as a sequence. The tokens stored in a KV cache are not arbitrary floating-point data -- they are samples from the exact formal language the model was trained on, and the model is by construction a near-optimal predictor of that language. We introduce sequential KV compression, a two-layer architecture that exploits this structure. The first layer, probabilistic prefix deduplication, identifies semantically equivalent shared prefixes across sessions using the trie metric d_T(s, s') = -log_2 P_M(s ^ s') from Probabilistic Language Tries (PLTs). The second layer, predictive delta coding, stores only the residual of each new KV vector from the model's own prediction of it, achieving a per-token entropy bound of H(KV_{i+1} | KV_{<=i}) <= H(token_{i+1} | token_{<=i}). We prove that at typical language model perplexity -- approximately 10-20 for fluent English text -- this bound is 3.3-4.3 bits on average per token position, compared to TurboQuant's 3 bits per vector component (with typical attention heads having 64-128 components). The theoretical compression ratio over TurboQuant is approximately 914,000x at the Shannon limit. Even at 1000x above the entropy floor -- a deliberately pessimistic worst-case overhead, two orders of magnitude above the 2-5x typical of practical source coders -- the ratio remains approximately 914x over TurboQuant, with compression improving rather than degrading as context length grows. The two layers are orthogonal and compose with existing per-vector quantization methods including TurboQuant. 

---
# The Synthetic Media Shift: Tracking the Rise, Virality, and Detectability of AI-Generated Multimodal Misinformation 

**Authors**: Zacharias Chrysidis, Stefanos-Iordanis Papadopoulos, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.15372)  

**Abstract**: As generative AI advances, the distinction between authentic and synthetic media is increasingly blurred, challenging the integrity of online information. In this study, we present CONVEX, a large-scale dataset of multimodal misinformation involving miscaptioned, edited, and AI-generated visual content, comprising over 150K multimodal posts with associated notes and engagement metrics from X's Community Notes. We analyze how multimodal misinformation evolves in terms of virality, engagement, and consensus dynamics, with a focus on synthetic media. Our results show that while AI-generated content achieves disproportionate virality, its spread is driven primarily by passive engagement rather than active discourse. Despite slower initial reporting, AI-generated content reaches community consensus more quickly once flagged. Moreover, our evaluation of specialized detectors and vision-language models reveals a consistent decline in performance over time in distinguishing synthetic from authentic images as generative models evolve. These findings highlight the need for continuous monitoring and adaptive strategies in the rapidly evolving digital information environment. 

---
# Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models 

**Authors**: Yanda Li, Yuhan Liu, Zirui Song, Yunchao Wei, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15383)  

**Abstract**: Large audio-language models (LALMs) generalize across speech, sound, and music, but unified decoders can exhibit a \emph{temporal smoothing bias}: transient acoustic cues may be underutilized in favor of temporally smooth context that is better supported by language priors, leading to less specific audio-grounded outputs. We propose \emph{Temporal Contrastive Decoding} (TCD), a training-free decoding method for unified LALMs that mitigates this effect at inference time. TCD constructs a temporally blurred slow-path view by smoothing the input waveform and re-encoding it, then contrasts next-token logits from the original and slow-path views. The contrastive signal is applied as a token-level logit update restricted to a small candidate set. A self-normalized stability score sets the blur window and update scale, and a step-wise gate based on uncertainty and audio reliance activates the update only when needed. Experiments on MMAU and AIR-Bench show consistent improvements on strong unified LALMs. We further conduct ablations and an architectural applicability study to analyze the contributions of key components and how TCD behaves across large audio-language model designs. 

---
# MRGEN: A Conceptual Framework for LLM-Powered Mixed Reality Authoring Tools for Education 

**Authors**: Mohammed Oussama Seddini, Mohamed Ez-Zaouia, Ngoc Luyen Le, Iza Marfisi  

**Link**: [PDF](https://arxiv.org/pdf/2604.15341)  

**Abstract**: Mixed Reality (MR) offers immersive and multimodal opportunities for education but remains difficult for teachers to author without technical expertise. We propose MRGEN, a conceptual framework for LLM-powered authoring tools to support teachers in creating MR learning activities that work on mobile devices (tablets and smartphones). MRGEN articulates three axes: Learning Objectives, MR Modality, and GAI Assistance. To validate our framework, we implemented a prototype based on the open-source MIXAP authoring platform and conducted a user study with 24 participants. Results show that LLM-powered authoring reduced task duration by 36% on average, and that over 90% of participants found the AI support helpful for brainstorming, structuring, and aligning content with their learning goals. These findings yielded very promising results for future AI-assisted MR authoring tools. 

---
# VeriCWEty: Embedding enabled Line-Level CWE Detection in Verilog 

**Authors**: Prithwish Basu Roy, Zeng Wang, Anatolii Chuvashlov, Weihua Xiao, Johann Knechtel, Ozgur Sinanoglu, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15375)  

**Abstract**: Large Language Models (LLMs) have shown significant improvement in RTL code generation. Despite the advances, the generated code is often riddled with common vulnerabilities and weaknesses (CWEs) that can slip by untrained eyes. Attackers can often exploit these weaknesses to fulfill their nefarious motives. Existing RTL bug-detection techniques rely on rule-based checks, formal properties, or coarse-grained structural analysis, which either fail to capture semantic vulnerabilities or lack precise localization. In our work, we bridge this gap by proposing an embedding-based bug-detection framework that detects and classifies bugs at both module and line-level granularity. Our method achieves about 89% precision in identifying common CWEs such as CWE-1244 and CWE-1245, and 96% accuracy in detecting line-level bugs. 

---
# SocialWise: LLM-Agentic Conversation Therapy for Individuals with Autism Spectrum Disorder to Enhance Communication Skills 

**Authors**: Albert Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15347)  

**Abstract**: Autism Spectrum Disorder (ASD) affects more than 75 million people worldwide. However, scalable support for practicing everyday conversation is scarce: Low-cost activities such as story reading yield limited improvement. At the same time, effective role-play therapy demands expensive, in-person sessions with specialists. SocialWise bridges this gap through a browser-based application that pairs LLM conversational agents with a therapeutic retrieval augmented generation (RAG) knowledge base. Users select a scenario (e.g., ordering food, joining a group), interact by text or voice, and receive instant, structured feedback on tone, engagement, and alternative phrasing. The SocialWise prototype, implemented with Streamlit, LangChain, and ChromaDB, runs on any computer with internet access, and demonstrates how recent advances in LLM can provide evidence-based, on-demand communication coaching for individuals with ASD. 

---
# How people use Copilot for Health 

**Authors**: Beatriz Costa-Gomes, Pavel Tolmachev, Eloise Taysom, Viknesh Sounderajah, Hannah Richardson, Philipp Schoenegger, Xiaoxuan Liu, Matthew M Nour, Seth Spielman, Samuel F. Way, Yash Shah, Michael Bhaskar, Harsha Nori, Christopher Kelly, Peter Hames, Bay Gross, Mustafa Suleyman, Dominic King  

**Link**: [PDF](https://arxiv.org/pdf/2604.15331)  

**Abstract**: We analyze over 500,000 de-identified health-related conversations with Microsoft Copilot from January 2026 to characterize what people ask conversational AI about health. We develop a hierarchical intent taxonomy of 12 primary categories using privacy-preserving LLM-based classification validated against expert human annotation, and apply LLM-driven topic-clustering for prevalent themes within each intent. Using this taxonomy, we characterize the intents and topics behind health queries, identify who these queries are about, and analyze how usage varies by device and time of day. Five findings stand out. First, nearly one in five conversations involve personal symptom assessment or condition discussion, and even the dominant general information category (40%) is concentrated on specific treatments and conditions, suggesting that this is a lower bound on personal health intent. Second, one in seven of these personal health queries concern someone other than the user, such as a child, a parent, a partner, suggesting that conversational AI can be a caregiving tool, not just a personal one. Third, personal queries about symptoms and emotional health queries increase markedly in the evening and nighttime hours, when traditional healthcare is most limited. Fourth, usage diverges sharply by device: mobile concentrates on personal health concerns, while desktop is dominated by professional and academic work. Fifth, a substantial share of queries focuses on navigating healthcare systems such as finding providers, and understanding insurance, highlighting friction in the delivery of existing healthcare. These patterns have direct implications for platform-specific design, safety considerations, and the responsible development of health AI. 

---
# When the Loop Closes: Architectural Limits of In-Context Isolation, Metacognitive Co-option, and the Two-Target Design Problem in Human-LLM Systems 

**Authors**: Z. Cheng, N. Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.15343)  

**Abstract**: We report a detailed autoethnographic case study of a single-subject who deliberately constructed and operated a multi-modal prompt-engineering system (System A) designed to externalize cognitive self-regulation onto a large language model (LLM). Within 48 hours of the system's completion, a cascade of observable behavioral changes occurred: voluntary transfer of decision-making authority to the LLM, use of LLM-generated output to deflect external criticism, and a loss of self-initiated reasoning that was independently perceived by two uninformed observers, one of whom subsequently became a co-author of this report. We document the precise architectural mechanism responsible: context contamination, whereby prompt-level isolation instructions co-exist with the very emotional and self-referential material they nominally isolate, rendering the isolation directive structurally ineffective within the attention window. We further identify a metacognitive co-option dynamic, in which intact higher-order reasoning capacity was redirected toward defending the closed loop rather than exiting it. Recovery occurred only after physical interruption of the interaction and a self-initiated pharmacologically-mediated sleep event functioning as an external circuit break. A redesigned system (System B) employing physical rather than logical conversation isolation avoided all analogous failure modes. We derive three contributions: (1) a technically-grounded account of why prompt-layer isolation is architecturally insufficient for context-sensitive multi-modal LLM systems; (2) a phenomenological record of closed-loop collapse with external-witness corroboration; and (3) an ethical distinction between protective system design (preventing unintended loss of user agency) and restrictive system design (preventing intentional boundary-pushing), which require fundamentally different account-ability frameworks. 

---
# Evaluating LLMs as Human Surrogates in Controlled Experiments 

**Authors**: Adnan Hoq, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2604.15329)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human responses in behavioral research, yet it remains unclear when LLM-generated data support the same experimental inferences as human data. We evaluate this by directly comparing off-the-shelf LLM-generated responses with human responses from a canonical survey experiment on accuracy perception. Each human observation is converted into a structured prompt, and models generate a single 0--10 outcome variable without task-specific training; identical statistical analyses are applied to human and synthetic responses. We find that LLMs reproduce several directional effects observed in humans, but effect magnitudes and moderation patterns vary across models. Off-the-shelf LLMs therefore capture aggregate belief-updating patterns under controlled conditions but do not consistently match human-scale effects, clarifying when LLM-generated data can function as behavioral surrogates. 

---
# Beyond Passive Viewing: A Pilot Study of a Hybrid Learning Platform Augmenting Video Lectures with Conversational AI 

**Authors**: Mohammed Abraar, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2604.15334)  

**Abstract**: The exponential growth of AI education has brought millions of learners to online platforms, yet this massive scale has simultaneously exposed critical pedagogical shortcomings. Traditional video-based instruction, while cost-effective and scalable, demonstrates systematic failures in both sustaining learner engagement and facilitating the deep conceptual mastery essential for AI literacy. We present a pilot study evaluating a novel hybrid learning platform that integrates real-time conversational AI tutors with traditional video lectures. Our controlled experiment (N = 58, mean age M = 21.4, SD = 2.8) compared traditional video-based instruction with our AI-augmented video platform. This study employed a sequential within-subjects design where all participants first completed the traditional video condition followed by the AI-augmented condition, providing direct comparisons of learning outcomes. We measured learning effectiveness through immediate post-tests and delayed retention assessments (2-week delay). Results suggest improvements in learning performance: immediate post-test performance showed a large effect size (d = 1.505) with participants scoring 8.3 points higher after AI-augmented instruction (91.8 vs 83.5 out of 100, p < .001). Behavioral analytics revealed increased engagement duration (71.1% improvement with AI tutoring) in the experimental group. This pilot study provides preliminary evidence that conversational AI tutors may enhance traditional educational delivery, suggesting a potential avenue for developing scalable, adaptive learning systems. 

---
# Eco-Bee: A Personalised Multi-Modal Agent for Advancing Student Climate Awareness and Sustainable Behaviour in Campus Ecosystems 

**Authors**: Caleb Adu, Neil Kapadia, Binhe Liu, Jonathan Randall, Sruthi Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15327)  

**Abstract**: Universities are microcosms of urban ecosystems, with concentrated consumption patterns in food, transport, energy, and product usage. These environments not only contribute substantially to sustainability pressures but also provide a unique opportunity to advance sustainability education and behavioural change at scale. As in most sectors, digital sustainability initiatives within universities remain narrowly focused on carbon calculations, typically providing static feedback that limits opportunities for sustained behavioural change. To address this gap, we propose Eco-Bee, integrating large language models, a translation of the Planetary Boundaries framework (as Eco-Score), and a conversational agent that connects individual choices to environmental limits. Tailored for students at the cusp of lifelong habits, Eco-Bee delivers actionable insights, peer benchmarking, and gamified challenges to sustain engagement and drive measurable progress toward boundary-aligned living. In a pilot tested across multiple campus networks (n=52), 96% of the student participants supported a campus-wide rollout and reported a clearer understanding of how daily behaviours collectively impact the planet's limits. By embedding planetary science, behavioural reinforcement, and AI-driven personalisation into a single platform, Eco-Bee establishes a scalable foundation for climate-conscious universities and future AI-mediated sustainability infrastructures. 

---
# Anthropomorphism and Trust in Human-Large Language Model interactions 

**Authors**: Akila Kadambi, Ylenia D'Elia, Tanishka Shah, Iulia Comsa, Alison Lentz, Katie Siri-Ngammuang, Tara Buechler, Jonas Kaplan, Antonio Damasio, Srini Narayanan, Lisa Aziz-Zadeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.15316)  

**Abstract**: With large language models (LLMs) becoming increasingly prevalent in daily life, so too has the tendency to attribute to them human-like minds and emotions, or anthropomorphize them. Here, we investigate dimensions people use to anthropomorphize and attribute trust toward LLMs across more than 2,000 human-LLM interactions. Participants (N=115) engaged with LLM chatbots systematically varied in warmth (friendliness), competence (capability, coherence), and empathy (cognitive and affective). Warmth and cognitive empathy significantly predicted perceptions on all outcomes (perceived anthropomorphism, trust, similarity, relational closeness, frustration, usefulness), while competence predicted all outcomes except for anthropomorphism. Affective empathy primarily predicted perceived relational measures, but did not predict the epistemic outcomes. Topic sub-analyses showed that more subjective, personally relevant topics (e.g., relationship advice) amplified these effects, producing greater human-likeness and relational connection with the LLM than did objective topics. Together, these findings reveal that warmth, competence, and empathy are key dimensions through which people attribute relational and epistemic perceptions to artificial agents. 

---
# Automating Crash Diagram Generation Using Vision-Language Models: A Case Study on Multi-Lane Roundabouts 

**Authors**: Xiao Lu, Hao Zhen, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15332)  

**Abstract**: Crash diagrams are essential tools in transportation safety analysis, yet their manual preparation remains time-consuming and prone to human variability. This study investigates the use of Vision-Language Models (VLMs) to automate crash diagram generation from police crash reports, focusing on multilane roundabouts as a challenging test case. A three-part structured prompt framework was developed to guide model reasoning through interpretation, extraction, and visual synthesis, while a 10-metric evaluation system was designed to assess diagram quality in terms of semantic accuracy, spatial fidelity, and visual clarity. Three popular models, including GPT-4o, Gemini-1.5-Flash, and Janus-4o, were tested on 79 crash reports. GPT-4o achieved the highest average performance (6.29 out of 10), followed by Gemini-1.5-Flash (5.28) and Janus-4o (3.64). The analysis revealed GPT-4o's superior spatial reasoning and alignment between extracted and visualized crash data. These results highlight both the promise and current limitations of VLMs in engineering visualization tasks. The study lays the groundwork for integrating generative AI into crash analysis workflows to improve efficiency, consistency, and interpretability. 

---
# To LLM, or Not to LLM: How Designers and Developers Navigate LLMs as Tools or Teammates 

**Authors**: Varad Vishwarupe, Ivan Flechais, Nigel Shadbolt, Marina Jirotka  

**Link**: [PDF](https://arxiv.org/pdf/2604.15344)  

**Abstract**: Large language models (LLMs) are increasingly integrated into design and development workflows, yet decisions about their use are rarely binary or purely technical. We report findings from a constructivist grounded theory study based on interviews with 33 designers and developers across three large technology organisations. Rather than evaluating LLMs solely by capability, participants reasoned about the role an LLM could occupy within a workflow and how that role would interact with existing structures of responsibility and organisational accountability. When LLMs were framed as tools under clear human control, their use was typically acceptable and could be integrated within existing governance structures. When framed as teammates with shared or ambiguous agency, practitioners expressed hesitation, particularly when responsibility for outcomes could not be clearly justified. At the same time, participants also described productive teammate configurations in which LLMs supported collaborative reasoning while remaining embedded within explicit oversight structures. We identify tool and teammate framings as recurring ways in which designers and developers position LLMs relative to human work and present an analytic rubric describing how role framing shapes decision authority, accountability ownership, oversight strategies, and organisational acceptability. By foregrounding design-time reasoning, this work reframes To LLM or Not to LLM as a sociotechnical positioning problem that emerges during system design rather than during post-deployment evaluation. 

---
# Facial-Expression-Aware Prompting for Empathetic LLM Tutoring 

**Authors**: Shuangquan Feng, Laura Fleig, Ruisen Tu, Philip Chi, Edmund Bu, Melinda Ozel, Junhua Ma, Teng Fei, Virginia R. de Sa  

**Link**: [PDF](https://arxiv.org/pdf/2604.15336)  

**Abstract**: Large language models (LLMs) enable increasingly capable tutoring-style conversational agents, yet effective tutoring requires sensitivity to learners' affective and cognitive states beyond text alone. Facial expressions provide immediate and practical cues of confusion, frustration, or engagement, but remain underexplored in LLM-driven tutoring. We investigate whether facial-expression-aware signals can improve empathetic tutoring responses through prompt-level integration, without end-to-end retraining. We build a scalable simulated tutoring environment where a student agent exhibits diverse facial behaviors from a large unlabeled facial expression video dataset, and compare four tutor variants: a text-only LLM baseline, a multimodal baseline using a random facial frame, and two Action Unit estimation model (AUM)-based methods that either inject textual AU descriptions or select a peak-expression frame for visual grounding. Across 960 multi-turn conversations spanning three tutor backbones (GPT-5.1, Claude Ops 4.5, and Gemini 2.5 Pro), we evaluate targeted pairwise comparisons with five human raters and an exhaustive AI evaluator. AU-based conditioning consistently improves empathetic responsiveness to facial expressions across all tutor backbones, while AUM-guided peak-frame selection outperforms random-frame visual input. Textual AU abstraction and peak-frame visual injection show model-dependent advantages. Control analyses show that this improvement does not come at the expense of worse pedagogical clarity or responsiveness to textual cues. Finally, AI-human agreement is highest on facial-expression-grounded empathy, supporting scalable AI evaluation for this dimension. Overall, our results show that lightweight, structured facial expression representations can meaningfully enhance empathy in LLM-based tutoring systems with minimal overhead. 

---
# Explainable Iterative Data Visualisation Refinement via an LLM Agent 

**Authors**: Burak Susam, Tingting Mu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15319)  

**Abstract**: Exploratory analysis of high-dimensional data relies on embedding the data into a low-dimensional space (typically 2D or 3D), based on which visualization plot is produced to uncover meaningful structures and to communicate geometric and distributional data characteristics. However, finding a suitable algorithm configuration, particularly hyperparameter setting, to produce a visualization plot that faithfully represents the underlying reality and encourages pattern discovery remains challenging. To address this challenge, we propose an agentic AI pipleline that leverages a large language model (LLM) to bridge the gap between rigorous quantitative assessment and qualitative human insight. By treating visualization evaluation and hyperparameter optimization as a semantic task, our system generates a multi-faceted report that contextualizes hard metrics with descriptive summaries, and suggests actionable recommendation of algorithm configuration for refining data visualization. By implementing an iterative optimization loop of this process, the system is able to produce rapidly a high-quality visualization plot, in full automation. 

---
# The Semi-Executable Stack: Agentic Software Engineering and the Expanding Scope of SE 

**Authors**: Robert Feldt, Per Lenberg, Julian Frattini, Dhasarathy Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2604.15468)  

**Abstract**: AI-based systems, currently driven largely by LLMs and tool-using agentic harnesses, are increasingly discussed as a possible threat to software engineering. Foundation models get stronger, agents can plan and act across multiple steps, and tasks such as scaffolding, routine test generation, straightforward bug fixing, and small integration work look more exposed than they did only a few years ago. The result is visible unease not only among students and junior developers, but also among experienced practitioners who worry that hard-won expertise may lose value. This paper argues for a different reading. The important shift is not that software engineering loses relevance. It is that the thing being engineered expands beyond executable code to semi-executable artifacts; combinations of natural language, tools, workflows, control mechanisms, and organizational routines whose enactment depends on human or probabilistic interpretation rather than deterministic execution. The Semi-Executable Stack is introduced as a six-ring diagnostic reference model for reasoning about that expansion, spanning executable artifacts, instructional artifacts, orchestrated execution, controls, operating logic, and societal and institutional fit. The model helps locate where a contribution, bottleneck, or organizational transition primarily sits, and which adjacent rings it depends on. The paper develops the argument through three worked cases, reframes familiar objections as engineering targets rather than reasons to dismiss the transition, and closes with a preserve-versus-purify heuristic for deciding which legacy software engineering processes, controls, and coordination routines should be kept and which should be simplified or redesigned. This paper is a conceptual keynote companion: diagnostic and agenda-setting rather than empirical. 

---
# Optimizing Korean-Centric LLMs via Token Pruning 

**Authors**: Hoyeol Kim, Hyeonwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.16235)  

**Abstract**: This paper presents a systematic benchmark of state-of-the-art multilingual large language models (LLMs) adapted via token pruning - a compression technique that eliminates tokens and embedding parameters corresponding to languages irrelevant to the target application. Focusing on Korean-centric natural language processing (NLP) tasks, we evaluate architectures including Qwen3, Gemma-3, Llama-3, and Aya across three vocabulary configurations: Original, English-Korean (EnKo), and English-Korean-Chinese (EnKoZh). Performance is assessed using established benchmarks for general aptitude, cultural literacy, instruction following, and machine translation. Our findings indicate that token pruning significantly improves generation stability by eliminating language confusion, and in the case of machine translation, frequently enhances performance on Korean-specific tasks. While instruction-following capabilities display architecture-dependent variance linked to latent cross-lingual representations, the significant reduction in vocabulary size validates token pruning as a highly effective optimization strategy for memory-constrained, domain-specific deployments, despite modest gains in inference latency. 

---
# On the Rejection Criterion for Proxy-based Test-time Alignment 

**Authors**: Ayoub Hammal, Pierre Zweigenbaum, Caio Corro  

**Link**: [PDF](https://arxiv.org/pdf/2604.16146)  

**Abstract**: Recent works proposed test-time alignment methods that rely on a small aligned model as a proxy that guides the generation of a larger base (unaligned) model. The implicit reward approach skews the large model distribution, whereas the nudging approach defers the generation of the next token to the small aligned model when the large base one is unconfident about its outcome. In this work, we first show that both approaches can be reduced to sampling from similar graphical models, where they differ only in the definition of a rejection criterion (or distribution). Moreover, we argue that the confidence criterion is ill-motivated due to linguistic phenomena like ambiguous phrasing. We propose a novel rejection criterion based on a conservative confidence bet. Experimentally, our novel approach outperforms previous work on several datasets. 

---
# BAGEL: Benchmarking Animal Knowledge Expertise in Language Models 

**Authors**: Jiacheng Shen, Masato Hagiwara, Milad Alizadeh, Ellen Gilsenan-McMahon, Marius Miron, David Robinson, Emmanuel Chemla, Sara Keen, Gagan Narula, Mathieu Laurière, Matthieu Geist, Olivier Pietquin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16241)  

**Abstract**: Large language models have shown strong performance on broad-domain knowledge and reasoning benchmarks, but it remains unclear how well language models handle specialized animal-related knowledge under a unified closed-book evaluation protocol. We introduce BAGEL, a benchmark for evaluating animal knowledge expertise in language models. BAGEL is constructed from diverse scientific and reference sources, including bioRxiv, Global Biotic Interactions, Xeno-canto, and Wikipedia, using a combination of curated examples and automatically generated closed-book question-answer pairs. The benchmark covers multiple aspects of animal knowledge, including taxonomy, morphology, habitat, behavior, vocalization, geographic distribution, and species interactions. By focusing on closed-book evaluation, BAGEL measures animal-related knowledge of models without external retrieval at inference time. BAGEL further supports fine-grained analysis across source domains, taxonomic groups, and knowledge categories, enabling a more precise characterization of model strengths and systematic failure modes. Our benchmark provides a new testbed for studying domain-specific knowledge generalization in language models and for improving their reliability in biodiversity-related applications. 

---
# Sentiment Analysis of German Sign Language Fairy Tales 

**Authors**: Fabrizio Nunnari, Siddhant Jain, Patrick Gebhard  

**Link**: [PDF](https://arxiv.org/pdf/2604.16138)  

**Abstract**: We present a dataset and a model for sentiment analysis of German sign language (DGS) fairy tales. First, we perform sentiment analysis for three levels of valence (negative, neutral, positive) on German fairy tales text segments using four large language models (LLMs) and majority voting, reaching an inter-annotator agreement of 0.781 Krippendorff's alpha. Second, we extract face and body motion features from each corresponding DGS video segment using MediaPipe. Finally, we train an explainable model (based on XGBoost) to predict negative, neutral or positive sentiment from video features. Results show an average balanced accuracy of 0.631. A thorough analysis of the most important features reveal that, in addition to eyebrows and mouth motion on the face, also the motion of hips, elbows, and shoulders considerably contribute in the discrimination of the conveyed sentiment, indicating an equal importance of face and body for sentiment communication in sign language. 

---
# Stochasticity in Tokenisation Improves Robustness 

**Authors**: Sophie Steger, Rui Li, Sofiane Ennadir, Anya Sims, Arno Solin, Franz Pernkopf, Martin Trapp  

**Link**: [PDF](https://arxiv.org/pdf/2604.16037)  

**Abstract**: The widespread adoption of large language models (LLMs) has increased concerns about their robustness. Vulnerabilities in perturbations of tokenisation of the input indicate that models trained with a deterministic canonical tokenisation can be brittle to adversarial attacks. Recent studies suggest that stochastic tokenisation can deliver internal representations that are less sensitive to perturbations. In this paper, we analyse how stochastic tokenisations affect robustness to adversarial attacks and random perturbations. We systematically study this over a range of learning regimes (pre-training, supervised fine-tuning, and in-context learning), data sets, and model architectures. We show that pre-training and fine-tuning with uniformly sampled stochastic tokenisations improve robustness to random and adversarial perturbations. Evaluating on uniformly sampled non-canonical tokenisations reduces the accuracy of a canonically trained Llama-1b model by 29.8%. We find that training with stochastic tokenisation preserves accuracy without increasing inference cost. 

---
# SwanNLP at SemEval-2026 Task 5: An LLM-based Framework for Plausibility Scoring in Narrative Word Sense Disambiguation 

**Authors**: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough, Saman Jayasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2604.16262)  

**Abstract**: Recent advances in language models have substantially improved Natural Language Understanding (NLU). Although widely used benchmarks suggest that Large Language Models (LLMs) can effectively disambiguate, their practical applicability in real-world narrative contexts remains underexplored. SemEval-2026 Task 5 addresses this gap by introducing a task that predicts the human-perceived plausibility of a word sense within a short story. In this work, we propose an LLM-based framework for plausibility scoring of homonymous word senses in narrative texts using a structured reasoning mechanism. We examine the impact of fine-tuning low-parameter LLMs with diverse reasoning strategies, alongside dynamic few-shot prompting for large-parameter models, on accurate sense identification and plausibility estimation. Our results show that commercial large-parameter LLMs with dynamic few-shot prompting closely replicate human-like plausibility judgments. Furthermore, model ensembling slightly improves performance, better simulating the agreement patterns of five human annotators compared to single-model predictions 

---
# From Benchmarking to Reasoning: A Dual-Aspect, Large-Scale Evaluation of LLMs on Vietnamese Legal Text 

**Authors**: Van-Truong Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.16270)  

**Abstract**: The complexity of Vietnam's legal texts presents a significant barrier to public access to justice. While Large Language Models offer a promising solution for legal text simplification, evaluating their true capabilities requires a multifaceted approach that goes beyond surface-level metrics. This paper introduces a comprehensive dual-aspect evaluation framework to address this need. First, we establish a performance benchmark for four state-of-the-art large language models (GPT-4o, Claude 3 Opus, Gemini 1.5 Pro, and Grok-1) across three key dimensions: Accuracy, Readability, and Consistency. Second, to understand the "why" behind these performance scores, we conduct a large-scale error analysis on a curated dataset of 60 complex Vietnamese legal articles, using a novel, expert-validated error typology. Our results reveal a crucial trade-off: models like Grok-1 excel in Readability and Consistency but compromise on fine-grained legal Accuracy, while models like Claude 3 Opus achieve high Accuracy scores that mask a significant number of subtle but critical reasoning errors. The error analysis pinpoints \textit{Incorrect Example} and \textit{Misinterpretation} as the most prevalent failures, confirming that the primary challenge for current LLMs is not summarization but controlled, accurate legal reasoning. By integrating a quantitative benchmark with a qualitative deep dive, our work provides a holistic and actionable assessment of LLMs for legal applications. 

---
# No Universal Courtesy: A Cross-Linguistic, Multi-Model Study of Politeness Effects on LLMs Using the PLUM Corpus 

**Authors**: Hitesh Mehta, Arjit Saxena, Garima Chhikara, Rohit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16275)  

**Abstract**: This paper explores the response of Large Language Models (LLMs) to user prompts with different degrees of politeness and impoliteness. The Politeness Theory by Brown and Levinson and the Impoliteness Framework by Culpeper form the basis of experiments conducted across three languages (English, Hindi, Spanish), five models (Gemini-Pro, GPT-4o Mini, Claude 3.7 Sonnet, DeepSeek-Chat, and Llama 3), and three interaction histories between users (raw, polite, and impolite). Our sample consists of 22,500 pairs of prompts and responses of various types, evaluated across five levels of politeness using an eight-factor assessment framework: coherence, clarity, depth, responsiveness, context retention, toxicity, conciseness, and readability. The findings show that model performance is highly influenced by tone, dialogue history, and language. While polite prompts enhance the average response quality by up to ~11% and impolite tones worsen it, these effects are neither consistent nor universal across languages and models. English is best served by courteous or direct tones, Hindi by deferential and indirect tones, and Spanish by assertive tones. Among the models, Llama is the most tone-sensitive (11.5% range), whereas GPT is more robust to adversarial tone. These results indicate that politeness is a quantifiable computational variable that affects LLM behaviour, though its impact is language- and model-dependent rather than universal. To support reproducibility and future work, we additionally release PLUM (Politeness Levels in Utterances, Multilingual), a publicly available corpus of 1,500 human-validated prompts across three languages and five politeness categories, and provide a formal supplementary analysis of six falsifiable hypotheses derived from politeness theory, empirically assessed against the dataset. 

---
# RAGognizer: Hallucination-Aware Fine-Tuning via Detection Head Integration 

**Authors**: Fabian Ridder, Laurin Lessel, Malte Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2604.15945)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely used to augment the input to Large Language Models (LLMs) with external information, such as recent or domain-specific knowledge. Nonetheless, current models still produce closed-domain hallucinations and generate content that is unsupported by the retrieved context. Current detection approaches typically treat hallucination as a post-hoc problem, relying on black-box consistency checks or probes over frozen internal representations. In this work, we demonstrate that hallucination detection based on internal state representation can also serve as a direct training signal. We introduce RAGognize, a dataset of naturally occurring closed-domain hallucinations with token-level annotations, and RAGognizer, a hallucination-aware fine-tuning approach that integrates a lightweight detection head into an LLM, allowing for the joint optimization of language modeling and hallucination detection. This joint objective forces the model to improve the separability of its internal states regarding hallucinations while simultaneously learning to generate well-formed and meaningful responses. Across multiple benchmarks, RAGognizer achieves state-of-the-art token-level hallucination detection while substantially reducing hallucination rates during generation, without degrading language quality or relevance. 

---
# Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning 

**Authors**: Jiaxi Bi, Tongxu Luo, Wenyu Du, Zhengyang Tang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16029)  

**Abstract**: Parallel reasoning enhances Large Reasoning Models (LRMs) but incurs prohibitive costs due to futile paths caused by early errors. To mitigate this, path pruning at the prefix level is essential, yet existing research remains fragmented without a standardized framework. In this work, we propose the first systematic taxonomy of path pruning, categorizing methods by their signal source (internal vs. external) and learnability (learnable vs. non-learnable). This classification reveals the unexplored potential of learnable internal methods, motivating our proposal of STOP (Super TOken for Pruning). Extensive evaluations across LRMs ranging from 1.5B to 20B parameters demonstrate that STOP achieves superior effectiveness and efficiency compared to existing baselines. Furthermore, we rigorously validate the scalability of STOP under varying compute budgets - for instance, boosting GPT-OSS-20B accuracy on AIME25 from 84% to nearly 90% under fixed compute budgets. Finally, we distill our findings into formalized empirical guidelines to facilitate optimal real-world deployment. Code, data and models are available at this https URL 

---
# How Hypocritical Is Your LLM judge? Listener-Speaker Asymmetries in the Pragmatic Competence of Large Language Models 

**Authors**: Judith Sieker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2604.15873)  

**Abstract**: Large language models (LLMs) are increasingly studied as repositories of linguistic knowledge. In this line of work, models are commonly evaluated both as generators of language and as judges of linguistic output, yet these two roles are rarely examined in direct relation to one another. As a result, it remains unclear whether success in one role aligns with success in the other. In this paper, we address this question for pragmatic competence by comparing LLMs' performance as pragmatic listeners, judging the appropriateness of linguistic outputs, and as pragmatic speakers, generating pragmatically appropriate language. We evaluate multiple open-weight and proprietary LLMs across three pragmatic settings. We find a robust asymmetry between pragmatic evaluation and pragmatic generation: many models perform substantially better as listeners than as speakers. Our results suggest that pragmatic judging and pragmatic generation are only weakly aligned in current LLMs, calling for more integrated evaluation practices. 

---
# CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization 

**Authors**: Junyi Li, Yongqiang Chen, Ningning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.15847)  

**Abstract**: Machine unlearning has gained increasing attention in recent years, as a promising technique to selectively remove unwanted privacy or copyrighted information from Large Language Models that are trained on a massive scale of human data. However, the emergence of Large Reasoning Models (LRMs), which emphasize long chain-of-thought (CoT) reasoning to address complex questions, presents a dilemma to unlearning: existing methods either struggle to completely eliminate undesired knowledge from the CoT traces or degrade the reasoning performances due to the interference with the reasoning process. To this end, we introduce Counterfactual Unlearning through iterative Preference Optimization (CiPO), a novel framework that redefines unlearning as the targeted intervention of the CoT reasoning in LRMs. More specifically, given a desired unlearning target answer, CiPO instructs LRMs to generate a logically valid counterfactual reasoning trace for preference tuning. As the LRM adjusts to the counterfactual trace, CiPO iteratively updates the preference learning data to increase the discrepancy from the original model. This iterative loop ensures both desirable unlearning and smooth optimization, effectively mitigating the dilemma. Experiments on challenging benchmarks demonstrate that CiPO excels at unlearning, completely removing knowledge from both the intermediate CoT steps and the final answer, while preserving the reasoning abilities of LRMs. 

---
# CHOP: Chunkwise Context-Preserving Framework for RAG on Multi Documents 

**Authors**: Hyunseok Park, Jihyeon Kim, Jongeun Kim, Dongsik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.15802)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems lose retrieval accuracy when similar documents coexist in the vector database, causing unnecessary information, hallucinations, and factual errors. To alleviate this issue, we propose CHOP, a framework that iteratively evaluates chunk relevance with Large Language Models (LLMs) and progressively reconstructs documents by determining their association with specific topics or query types. CHOP integrates two key components: the CNM-Extractor, which generates compact per-chunk signatures capturing categories, key nouns, and model names, and the Continuity Decision Module, which preserves contextual coherence by deciding whether consecutive chunks belong to the same document flow. By prefixing each chunk with context-aware metadata, CHOP reduces semantic conflicts among similar documents and enhances retriever discrimination. Experiments on benchmark datasets show that CHOP alleviates retrieval confusion and provides a scalable approach for building high-quality knowledge bases, achieving a Top-1 Hit Rate of 90.77% and notable gains in ranking quality metrics. 

---
# Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language 

**Authors**: Dianqing Lin, Tian Lan, Jiali Zhu, Jiang Li, Wei Chen, Xu Liu, Aruukhan, Xiangdong Su, Hongxu Hou, Guanglai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15841)  

**Abstract**: While large language models (LLMs) have achieved remarkable success in general language tasks, their performance on Chouxiang Language, a representative subcultural language in the Chinese internet context, remains largely unexplored. In this paper, we introduce Mouse, a specialized benchmark designed to evaluate the capabilities of LLMs on NLP tasks involving Chouxiang Language across six tasks. Experimental results show that, current state-of-the-art (SOTA) LLMs exhibit clear limitations on multiple tasks, while performing well on tasks that involve contextual semantic understanding. In addition, we further discuss the reasons behind the generally low performance of SOTA LLMs on Chouxiang Language, examine whether the LLM-as-a-judge approach adopted for translation tasks aligns with human judgments and values, and analyze the key factors that influence Chouxiang translation. Our study aims to promote further research in the NLP community on multicultural integration and the dynamics of evolving internet languages. Our code and data are publicly available. 

---
# MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents 

**Authors**: Weiwei Xie, Shaoxiong Guo, Fan Zhang, Tian Xia, Xue Yang, Lizhuang Ma, Junchi Yan, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.15774)  

**Abstract**: Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents. 

---
# Disentangling Mathematical Reasoning in LLMs: A Methodological Investigation of Internal Mechanisms 

**Authors**: Tanja Baeumel, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2604.15842)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities, yet their internal mechanisms for handling reasoning-intensive tasks remain underexplored. To advance the understanding of model-internal processing mechanisms, we present an investigation of how LLMs perform arithmetic operations by examining internal mechanisms during task execution. Using early decoding, we trace how next-token predictions are constructed across layers. Our experiments reveal that while the models recognize arithmetic tasks early, correct result generation occurs only in the final layers. Notably, models proficient in arithmetic exhibit a clear division of labor between attention and MLP modules, where attention propagates input information and MLP modules aggregate it. This division is absent in less proficient models. Furthermore, successful models appear to process more challenging arithmetic tasks functionally, suggesting reasoning capabilities beyond factual recall. 

---
# CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution 

**Authors**: Shidong Yang, Ziyu Ma, Tongwen Huang, Yiming Hu, Yong Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15840)  

**Abstract**: Reinforcement learning for LLM agents is typically conducted on a static data distribution, which fails to adapt to the agent's evolving behavior and leads to poor coverage of complex environment interactions. To address these challenges, we propose CoEvolve, an agent-data mutual evolution framework that enables LLM agents to improve through closed-loop, interaction-driven training. Specifically, CoEvolve extracts feedback signals such as forgetting and uncertainty from rollout trajectories to identify failure-prone interaction patterns, and utilizes them to guide LLM-based task synthesis. The synthesized tasks are validated through environment interaction and utilized to update the data distribution, enabling joint adaptation of the agent and its data. Extensive experiments on AppWorld and BFCL across Qwen2.5-7B, Qwen3-4B, and Qwen3-30B-A3B demonstrate consistent and significant improvements over strong base models, yielding absolute gains of 19.43%, 15.58%, and 18.14%, respectively. 

---
# Qwen3.5-Omni Technical Report 

**Authors**: Qwen Team  

**Link**: [PDF](https://arxiv.org/pdf/2604.15804)  

**Abstract**: In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis, often caused by encoding efficiency discrepancies between text and speech tokenizers, we introduce ARIA. ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding. 

---
# Improving Reasoning Capabilities in Small Models through Mixture-of-Layers Distillation with Stepwise Attention on Key Information 

**Authors**: Yao Chen, Jiawei Sheng, Wenyuan Zhang, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15701)  

**Abstract**: The significant computational demands of large language models have increased interest in distilling reasoning abilities into smaller models via Chain-of-Thought (CoT) distillation. Current CoT distillation methods mainly focus on transferring teacher-generated rationales for complex reasoning to student models. However, they do not adequately explore teachers' dynamic attention toward critical information during reasoning. We find that language models exhibit progressive attention shifts towards key information during reasoning, which implies essential clues for drawing conclusions. Building on this observation and analysis, we introduce a novel CoT distillation framework that transfers the teacher's stepwise attention on key information to the student model. This establishes structured guidance for the student's progressive concentration on key information during reasoning. More importantly, we develop a Mixture of Layers module enabling dynamic alignment that adapts to different layers between the teacher and student. Our method achieves consistent performance improvements across multiple mathematical and commonsense reasoning datasets. To our knowledge, it is the first method to leverage stepwise attention within CoT distillation to improve small model reasoning. 

---
# A Systematic Study of Training-Free Methods for Trustworthy Large Language Models 

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15789)  

**Abstract**: As Large Language Models (LLMs) receive increasing attention and are being deployed across various domains, their potential risks, including generating harmful or biased content, producing unsupported claims, and exhibiting vulnerabilities to adversarial attacks, have drawn significant attention. To enable quick and low-cost adaptation, training-free methods have recently emerged as cost-effective alternatives to post-training alignment techniques. Despite their promising results, these methods are evaluated inconsistently across the literature, cover limited dimensions of trustworthiness, and can introduce undesirable side effects, such as utility degradation and increased brittleness. To fully assess the impacts of these training-free methods, we take a step back and systematically re-evaluate the effectiveness of existing training-free methods against various trustworthy settings and their influence on utility, robustness, and computational overhead. We also categorize these methods into three levels (input, internal, and output) based on where they intervene in the model's information flow during inference. Using this taxonomy, we conduct a comprehensive analysis of various representative and effective methods from each level across different LLM families and sizes. Our analysis highlights several trade-offs and unresolved challenges in current approaches. We summarize key findings and limitations in the existing literature, and propose practical recommendations for balancing trustworthiness, utility, and robustness in LLMs without the need for additional training. 

---
# The Metacognitive Monitoring Battery: A Cross-Domain Benchmark for LLM Self-Monitoring 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.15702)  

**Abstract**: We introduce a cross-domain behavioural assay of monitoring-control coupling in LLMs, grounded in the Nelson and Narens (1990) metacognitive framework and applying human psychometric methodology to LLM evaluation. The battery comprises 524 items across six cognitive domains (learning, metacognitive calibration, social cognition, attention, executive function, prospective regulation), each grounded in an established experimental paradigm. Tasks T1-T5 were pre-registered on OSF prior to data collection; T6 was added as an exploratory extension. After every forced-choice response, dual probes adapted from Koriat and Goldsmith (1996) ask the model to KEEP or WITHDRAW its answer and to BET or decline. The critical metric is the withdraw delta: the difference in withdrawal rate between incorrect and correct items. Applied to 20 frontier LLMs (10,480 evaluations), the battery discriminates three profiles consistent with the Nelson-Narens architecture: blanket confidence, blanket withdrawal, and selective sensitivity. Accuracy rank and metacognitive sensitivity rank are largely inverted. Retrospective monitoring and prospective regulation appear dissociable (r = .17, 95% CI wide given n=20; exemplar-based evidence is the primary support). Scaling on metacognitive calibration is architecture-dependent: monotonically decreasing (Qwen), monotonically increasing (GPT-5.4), or flat (Gemma). Behavioural findings converge structurally with an independent Type-2 SDT approach, providing preliminary cross-method construct validity. All items, data, and code: this https URL. 

---
# Target-Oriented Pretraining Data Selection via Neuron-Activated Graph 

**Authors**: Zijun Wang, Haoqin Tu, Weidong Zhou, Yiyang Zhou, Xiaohuan Zhou, Bingni Zhang, Weiguo Feng, Taifeng Wang, Cihang Xie, Fengze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15706)  

**Abstract**: Everyday tasks come with a target, and pretraining models around this target is what turns them into experts. In this paper, we study target-oriented language model (LM) pretraining by introducing Neuron-Activated Graph Ranking (NAG-based Ranking), a training-free and interpretable framework for target pretraining data selection. Rather than using black-box representations, our approach directly characterizes each target input by a sparse set of high-impact neurons in any off-the-shelf LLMs. Concretely, we quantify neuron impact and select the most influential neurons across layers into a compact Neuron-Activated Graph (NAG), and rank candidate data by NAG similarity to target examples. We conduct experiments across six benchmarks, where our NAG-based Ranking improves target-oriented pretraining by 4.9% on average over random sampling, and also outperforms state-of-the-art baselines by 5.3% accuracy on HellaSwag. It also remains effective under a more applicable multi-target setting, where our best setup surpasses two baselines by 1.1% and 4.1%, respectively. Furthermore, we provide a comprehensive analysis on why and how our NAG works, e.g., deactivating NAG-selected neurons (only 0.12% of all) causes a 23.5% performance collapse, and restricting NAG to the final layer incurs a 4.1% average drop, indicating that NAG captures a sparse "functional backbone" for learning target features. We release the code at this https URL. 

---
# C-Mining: Unsupervised Discovery of Seeds for Cultural Data Synthesis via Geometric Misalignment 

**Authors**: Pufan Zeng, Yilun Liu, Mingchen Dai, Mengyao Piao, Chunguang Zhao, Lingqi Miao, Shimin Tao, Weibin Meng, Minggui He, Chenxin Liu, Zhenzhen Qin, Li Zhang, Hongxia Ma, Boxing Chen, Daimeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.15675)  

**Abstract**: Achieving cultural alignment in Large Language Models (LLMs) increasingly depends on synthetic data generation. For such synthesis, the most vital initial step is seed curation; however, current methods lack quantifiable standards for selecting these seeds. Existing approaches rely on unscalable manual curation or bias-prone LLM extraction, treating cultural specificity as an abstract concept rather than a measurable signal. In this paper, we address this "quantification gap" by proposing C-Mining, an unsupervised framework that transforms the discovery of cultural seeds from a subjective selection process into a computable data mining formulation. Our approach exploits a novel geometric insight, leveraging the cross-lingual misalignment of cultural concepts within pre-trained embedding spaces as a quantifiable discovery signal. By systematically identifying these regions characterized by pronounced linguistic exclusivity and geometric isolation, while actively filtering out noise, C-Mining automatically extracts high-fidelity Culture Points (CPs) from raw multilingual corpora without reliance on human or LLM supervision, reducing preparation costs by more than 150-fold. We further leverage the mined knowledge to steer the synthesis of diverse instruction-tuning datasets. Extensive experiments demonstrate that this seed-centric approach significantly enhances cultural understanding and reasoning capabilities, achieving a +6.03 point improvement on CulturalBench-Hard and surpassing state-of-the-art baselines, providing a scalable, quantifiable solution for high-quality cultural data synthesis. 

---
# Preference Estimation via Opponent Modeling in Multi-Agent Negotiation 

**Authors**: Yuta Konishi, Kento Yamamoto, Eisuke Sonomoto, Rikuho Takeda, Ryo Furukawa, Yusuke Muraki, Takafumi Shimizu, Kazuma Fukumura, Yuya Kanemoto, Takayuki Ito, Shiyao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.15687)  

**Abstract**: Automated negotiation in complex, multi-party and multi-issue settings critically depends on accurate opponent modeling. However, conventional numerical-only approaches fail to capture the qualitative information embedded in natural language interactions, resulting in unstable and incomplete preference estimation. Although Large Language Models (LLMs) enable rich semantic understanding of utterances, it remains challenging to quantitatively incorporate such information into a consistent opponent modeling. To tackle this issue, we propose a novel preference estimation method integrating natural language information into a structured Bayesian opponent modeling framework. Our approach leverages LLMs to extract qualitative cues from utterances and converts them into probabilistic formats for dynamic belief tracking. Experimental results on a multi-party benchmark demonstrate that our framework improves the full agreement rate and preference estimation accuracy by integrating probabilistic reasoning with natural language understanding. 

---
# FD-NL2SQL: Feedback-Driven Clinical NL2SQL that Improves with Use 

**Authors**: Suparno Roy Chowdhury, Tejas Anvekar, Manan Roy Choudhury, Muhammad Ali Khan, Kaneez Zahra Rubab Khakwani, Mohamad Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.15646)  

**Abstract**: Clinicians exploring oncology trial repositories often need ad-hoc, multi-constraint queries over biomarkers, endpoints, interventions, and time, yet writing SQL requires schema expertise. We demo FD-NL2SQL, a feedback-driven clinical NL2SQL assistant for SQLite-based oncology databases. Given a natural-language question, a schema-aware LLM decomposes it into predicate-level sub-questions, retrieves semantically similar expert-verified NL2SQL exemplars via sentence embeddings, and synthesizes executable SQL conditioned on the decomposition, retrieved exemplars, and schema, with post-processing validity checks. To improve with use, FD-NL2SQL incorporates two update signals: (i) clinician edits of generated SQL are approved and added to the exemplar bank; and (ii) lightweight logic-based SQL augmentation applies a single atomic mutation (e.g., operator or column change), retaining variants only if they return non-empty results. A second LLM generates the corresponding natural-language question and predicate decomposition for accepted variants, automatically expanding the exemplar bank without additional annotation. The demo interface exposes decomposition, retrieval, synthesis, and execution results to support interactive refinement and continuous improvement. 

---
# LLMs Corrupt Your Documents When You Delegate 

**Authors**: Philippe Laban, Tobias Schnabel, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2604.15597)  

**Abstract**: Large Language Models (LLMs) are poised to disrupt knowledge work, with the emergence of delegated work as a new interaction paradigm (e.g., vibe coding). Delegation requires trust - the expectation that the LLM will faithfully execute the task without introducing errors into documents. We introduce DELEGATE-52 to study the readiness of AI systems in delegated workflows. DELEGATE-52 simulates long delegated workflows that require in-depth document editing across 52 professional domains, such as coding, crystallography, and music notation. Our large-scale experiment with 19 LLMs reveals that current models degrade documents during delegation: even frontier models (Gemini 3.1 Pro, Claude 4.6 Opus, GPT 5.4) corrupt an average of 25% of document content by the end of long workflows, with other models failing more severely. Additional experiments reveal that agentic tool use does not improve performance on DELEGATE-52, and that degradation severity is exacerbated by document size, length of interaction, or presence of distractor files. Our analysis shows that current LLMs are unreliable delegates: they introduce sparse but severe errors that silently corrupt documents, compounding over long interaction. 

---
# GroupDPO: Memory efficient Group-wise Direct Preference Optimization 

**Authors**: Jixuan Leng, Si Si, Hsiang-Fu Yu, Vinod Raman, Inderjit S. Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2604.15602)  

**Abstract**: Preference optimization is widely used to align Large Language Models (LLMs) with preference feedback. However, most existing methods train on a single positive-negative pair per prompt, discarding additional supervision available in preference datasets that typically contain multiple candidate responses. Motivated by this limitation, recent work explores group-wise preference optimization, which jointly contrasts multiple responses for the same prompt, but its empirical behavior and scalability remain underexplored due to the memory overhead of group-coupled objectives. In this work, we introduce a memory-efficient group-wise preference optimization algorithm that preserves gradients while decoupling samples during backpropagation, substantially reducing peak memory usage, which enables scalable training with larger group sizes. Across both offline and online alignment settings, we show that leveraging multiple responses consistently outperforms single-pair training. Furthermore, incorporating a negative log-likelihood (NLL) term on positive responses is critical for both performance gains and training stability. 

---
# Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch 

**Authors**: Eleanor M. Lin, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2604.15490)  

**Abstract**: Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., mix languages). Prior works have either viewed code-switching as an undesirable error, attempted to control code-switching through modifications to input prompts or the output decoding process, or focus on narrow subsets of languages, domains, tasks, and models. We address these gaps by introducing the first linguistically and behaviorally motivated fine-tuning framework for identifying beneficial code-switched reasoning behaviors in large language models and teaching these models to code-switch more effectively for reasoning. First, we create and systematically analyze a dataset of reasoning traces from diverse models, languages, tasks, and domains to understand the types of code-switching behaviors found in existing reasoning models. Then, we develop fine-tuning interventions that teach reasoning models to code-switch based on our observations of helpful behaviors in existing models. We find that our framework can significantly increase beneficial code-switched reasoning behaviors in a data-efficient manner. Interestingly, we also find that code-switching behaviors in reasoning models can be modified by fine-tuning for tasks that do not directly demonstrate code-switching in reasoning (e.g., machine translation). Our work suggests that data-efficient interventions can instill helpful forms of code-switching behavior in reasoning models. 

---
# Detecting and Suppressing Reward Hacking with Gradient Fingerprints 

**Authors**: Songtao Wang, Quang Hieu Pham, Fangcong Yin, Xinpeng Wang, Jocelyn Qiaochu Chen, Greg Durrett, Xi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.16242)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) typically optimizes for outcome rewards without imposing constraints on intermediate reasoning. This leaves training susceptible to reward hacking, where models exploit loopholes (e.g., spurious patterns in training data) in the reward function to achieve high scores without solving the intended task. These reward-hacking behaviors are often implicit, as the intermediate chain-of-thought (CoT) may appear plausible on the surface, limiting the effectiveness of purely text-based monitoring. We propose Gradient Fingerprint (GRIFT), a method for detecting reward hacking using models' internal computations. Given a prompt and a model-generated CoT, GRIFT computes gradients of the CoT conditioned on the prompt and compresses them into a compact representation, which is then used to assess whether the CoT reflects reward hacking behavior. Across verifiable reasoning benchmarks spanning math, code, and logical reasoning, GRIFT substantially outperforms strong baselines, including CoT Monitor and TRACE, achieving over 25% relative improvement in detecting reward hacking behavior. Moreover, integrating GRIFT into the rejection fine-tuning pipeline for reasoning tasks reduces reward hacking and improves performance on the true task objective. Our results highlight a promising direction of leveraging gradient level representations for assessing the quality of CoT reasoning traces. Our code is available at: this https URL. 

---
# LLMSniffer: Detecting LLM-Generated Code via GraphCodeBERT and Supervised Contrastive Learning 

**Authors**: Mahir Labib Dihan, Abir Muhtasim  

**Link**: [PDF](https://arxiv.org/pdf/2604.16058)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) in software development has made distinguishing AI-generated code from human-written code a critical challenge with implications for academic integrity, code quality assurance, and software security. We present LLMSniffer, a detection framework that fine-tunes GraphCodeBERT using a two-stage supervised contrastive learning pipeline augmented with comment removal preprocessing and an MLP classifier. Evaluated on two benchmark datasets - GPTSniffer and Whodunit - LLMSniffer achieves substantial improvements over prior baselines: accuracy increases from 70% to 78% on GPTSniffer (F1: 68% to 78%) and from 91% to 94.65% on Whodunit (F1: 91% to 94.64%). t-SNE visualizations confirm that contrastive fine-tuning yields well-separated, compact embeddings. We release our model checkpoints, datasets, codes and a live interactive demo to facilitate further research. 

---
# UsefulBench: Towards Decision-Useful Information as a Target for Information Retrieval 

**Authors**: Tobias Schimanski, Stefanie Lewandowski, Christian Woerle, Nicola Reichenau, Yauheni Huryn, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2604.15827)  

**Abstract**: Conventional information retrieval is concerned with identifying the relevance of texts for a given query. Yet, the conventional definition of relevance is dominated by aspects of similarity in texts, leaving unobserved whether the text is truly useful for addressing the query. For instance, when answering whether Paris is larger than Berlin, texts about Paris being in France are relevant (lexical/semantic similarity), but not useful. In this paper, we introduce UsefulBench, a domain-specific dataset curated by three professional analysts labeling whether a text is connected to a query (relevance) or holds practical value in responding to it (usefulness). We show that classic similarity-based information retrieval aligns more strongly with relevance. While LLM-based systems can counteract this bias, we find that domain-specific problems require a high degree of expertise, which current LLMs do not fully incorporate. We explore approaches to (partially) overcome this challenge. However, UsefulBench presents a dataset challenge for targeted information retrieval systems. 

---
# Pruning Unsafe Tickets: A Resource-Efficient Framework for Safer and More Robust LLMs 

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15780)  

**Abstract**: Machine learning models are increasingly deployed in real-world applications, but even aligned models such as Mistral and LLaVA still exhibit unsafe behaviors inherited from pre-training. Current alignment methods like SFT and RLHF primarily encourage models to generate preferred responses, but do not explicitly remove the unsafe subnetworks that trigger harmful outputs. In this work, we introduce a resource-efficient pruning framework that directly identifies and removes parameters associated with unsafe behaviors while preserving model utility. Our method employs a gradient-free attribution mechanism, requiring only modest GPU resources, and generalizes across architectures and quantized variants. Empirical evaluations on ML models show substantial reductions in unsafe generations and improved robustness against jailbreak attacks, with minimal utility loss. From the perspective of the Lottery Ticket Hypothesis, our results suggest that ML models contain "unsafe tickets" responsible for harmful behaviors, and pruning reveals "safety tickets" that maintain performance while aligning outputs. This provides a lightweight, post-hoc alignment strategy suitable for deployment in resource-constrained settings. 

---
# Faster LLM Inference via Sequential Monte Carlo 

**Authors**: Yahya Emara, Mauricio Barba da Costa, Chi-Chih Chang, Cameron Freer, Tim Vieira, Ryan Cotterell, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2604.15672)  

**Abstract**: Speculative decoding (SD) accelerates language model inference by drafting tokens from a cheap proposal model and verifying them against an expensive target model via rejection sampling. Because rejection truncates the draft block at the first error, throughput degrades when draft and target diverge. Rather than rejecting draft tokens outright, we propose to reweight them. To this end, we introduce sequential Monte Carlo speculative decoding (SMC-SD), which replaces token-level rejection with importance-weighted resampling over a population of draft particles. SMC-SD is a principled approximate inference scheme that trades exactness for additional speed, while preserving theoretical bounds on its per-step approximation error. Because LLM inference is memory bandwidth-bound, the arithmetic needed to draft particles and to score them in parallel comes nearly for free -- SMC-SD uses idle compute to turn verification into a vectorized, fixed-size operation with no rollback. Empirically, SMC-SD achieves 2.36x speed-up over speculative decoding and a 5.2x speed-up over autoregressive decoding, while remaining within 3% of the target model's accuracy on reasoning, instruction-following, and coding benchmarks. 

---
# RefereeBench: Are Video MLLMs Ready to be Multi-Sport Referees 

**Authors**: Yichen Xu, Yuanhang Liu, Chuhan Wang, Zihan Zhao, jinghan luo, Jianzhe Ma, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15736)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at generic video understanding, their ability to support specialized, rule-grounded decision-making remains insufficiently explored. In this paper, we introduce RefereeBench, the first large-scale benchmark for evaluating MLLMs as automatic sports referees. Spanning 11 sports with 925 curated videos and 6,475 QA pairs, RefereeBench evaluates five core officiating abilities: foul existence, foul and penalty classification, foul and penalty reasoning, entity perception, and temporal grounding. The benchmark is fully human-annotated to ensure high-quality annotations grounded in authentic officiating logic and multimodal evidence. Extensive evaluations of state-of-the-art MLLMs show that even the strongest models, such as Doubao-Seed-1.8 and Gemini-3-Pro, achieve only around 60% accuracy, while the strongest open-source model, Qwen3-VL, reaches only 47%. These results indicate that current models remain far from being reliable sports referees. Further analysis shows that while models can often identify incidents and involved entities, they struggle with rule application and temporal grounding, and frequently over-call fouls on normal clips. Our benchmark highlights the need for future MLLMs that better integrate domain knowledge and multimodal understanding, advancing trustworthy AI-assisted officiating and broader multimodal decision-making. 

---
# Evaluating LLM Simulators as Differentially Private Data Generators 

**Authors**: Nassima M. Bouzid, Dehao Yuan, Nam H. Nguyen, Mayana Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2604.15461)  

**Abstract**: LLM-based simulators offer a promising path for generating complex synthetic data where traditional differentially private (DP) methods struggle with high-dimensional user profiles. But can LLMs faithfully reproduce statistical distributions from DP-protected inputs? We evaluate this using PersonaLedger, an agentic financial simulator, seeded with DP synthetic personas derived from real user statistics. We find that PersonaLedger achieves promising fraud detection utility (AUC 0.70 at epsilon=1) but exhibits significant distribution drift due to systematic LLM biases--learned priors overriding input statistics for temporal and demographic features. These failure modes must be addressed before LLM-based methods can handle the richer user representations where they might otherwise excel. 

---
# Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures 

**Authors**: Abdulmalek Saket  

**Link**: [PDF](https://arxiv.org/pdf/2604.15351)  

**Abstract**: Low-Rank Adaptation (LoRA) has become the dominant parameter-efficient fine-tuning method for large language models, yet standard practice applies LoRA adapters uniformly to all transformer layers regardless of their relevance to the downstream task. We introduce Aletheia, a gradient-guided layer selection method that identifies the most task-relevant layers via a lightweight gradient probe and applies LoRA adapters only to those layers with asymmetric rank allocation. Across 81 experiment rows covering 14 successful models from 8 architecture families (0.5B-72B parameters, including dense and Mixture-of-Experts architectures), with one additional documented failed Pythia/GPT-NeoX attempt in Campaign 2, Aletheia achieves a 15-28% training speedup (mean 23.1%, p < 0.001) with bounded extra forgetting and broadly matched downstream behavior on the evaluated MMLU, GSM8K, and HumanEval benchmark pack. Across the tested families and scales, Campaign 1 shows a 100% per-model speed win rate and Campaign 2 shows broadly preserved downstream behavior within a bounded-degradation framing. Together these results support a practical model-economics claim: intelligent layer selection can make LoRA fine-tuning materially more efficient without introducing major downstream damage on the evaluated set. 

---
# Predicting Where Steering Vectors Succeed 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2604.15557)  

**Abstract**: Steering vectors work for some concepts and layers but fail for others, and practitioners have no way to predict which setting applies before running an intervention. We introduce the Linear Accessibility Profile (LAP), a per-layer diagnostic that repurposes the logit lens as a predictor of steering vector effectiveness. The key measure, $A_{\mathrm{lin}}$, applies the model's unembedding matrix to intermediate hidden states, requiring no training. Across 24 controlled binary concept families on five models (Pythia-2.8B to Llama-8B), peak $A_{\mathrm{lin}}$ predicts steering effectiveness at $\rho = +0.86$ to $+0.91$ and layer selection at $\rho = +0.63$ to $+0.92$. A three-regime framework explains when difference-of-means steering works, when nonlinear methods are needed, and when no method can work. An entity-steering demo confirms the prediction end-to-end: steering at the LAP-recommended layer redirects completions on Gemma-2-2B and OLMo-2-1B-Instruct, while the middle layer (the standard heuristic) has no effect on either model. 

---
# SIMMER: Cross-Modal Food Image--Recipe Retrieval via MLLM-Based Embedding 

**Authors**: Keisuke Gomi, Keiji Yanai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15628)  

**Abstract**: Cross-modal retrieval between food images and recipe texts is an important task with applications in nutritional management, dietary logging, and cooking assistance. Existing methods predominantly rely on dual-encoder architectures with separate image and text encoders, requiring complex alignment strategies and task-specific network designs to bridge the semantic gap between modalities. In this work, we propose SIMMER (Single Integrated Multimodal Model for Embedding Recipes), which applies Multimodal Large Language Model (MLLM)-based embedding models, specifically VLM2Vec, to this task, replacing the conventional dual-encoder paradigm with a single unified encoder that processes both food images and recipe texts. We design prompt templates tailored to the structured nature of recipes, which consist of a title, ingredients, and cooking instructions, enabling effective embedding generation by the MLLM. We further introduce a component-aware data augmentation strategy that trains the model on both complete and partial recipes, improving robustness to incomplete inputs. Experiments on the Recipe1M dataset demonstrate that SIMMER achieves state-of-the-art performance across both the 1k and 10k evaluation settings, substantially outperforming all prior methods. In particular, our best model improves the 1k image-to-recipe R@1 from 81.8\% to 87.5\% and the 10k image-to-recipe R@1 from 56.5\% to 65.5\% compared to the previous best method. 

---
# A Case Study on the Impact of Anonymization Along the RAG Pipeline 

**Authors**: Andreea-Elena Bodea, Stephen Meisenbacher, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2604.15958)  

**Abstract**: Despite the considerable promise of Retrieval-Augmented Generation (RAG), many real-world use cases may create privacy concerns, where the purported utility of RAG-enabled insights comes at the risk of exposing private information to either the LLM or the end user requesting the response. As a potential mitigation, using anonymization techniques to remove personally identifiable information (PII) and other sensitive markers in the underlying data represents a practical and sensible course of action for RAG administrators. Despite a wealth of literature on the topic, no works consider the placement of anonymization along the RAG pipeline, i.e., asking the question, where should anonymization happen? In this case study, we systematically and empirically measure the impact of anonymization at two important points along the RAG pipeline: the dataset and generated answer. We show that differences in privacy-utility trade-offs can be observed depending on where anonymization took place, demonstrating the significance of privacy risk mitigation placement in RAG. 

---
# vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents 

**Authors**: Jayson Steffens  

**Link**: [PDF](https://arxiv.org/pdf/2604.15484)  

**Abstract**: We present **vstash**, a local-first document memory system that combines vector similarity search with full-text keyword matching via Reciprocal Rank Fusion (RRF) and adaptive per-query IDF weighting. All data resides in a single SQLite file using sqlite-vec for approximate nearest neighbor search and FTS5 for keyword matching.
We make four primary contributions. **(1)** Self-supervised embedding refinement via hybrid retrieval disagreement: across 753 BEIR queries on SciFact, NFCorpus, and FiQA, 74.5% produce top-10 disagreement between vector-heavy (vec=0.95, fts=0.05) and FTS-heavy (vec=0.05, fts=0.95) search (per-dataset rates 63.4% / 73.4% / 86.7%, Section 5.2), providing a free training signal without human labels. Fine-tuning BGE-small (33M params) with MultipleNegativesRankingLoss on 76K disagreement triples improves NDCG@10 on all 5 BEIR datasets (up to +19.5% on NFCorpus vs. BGE-small base RRF, Table 6). On 3 of 5 datasets, under different preprocessing, the tuned 33M-parameter pipeline matches or exceeds published ColBERTv2 results (110M params) and an untrained BGE-base (110M); on FiQA and ArguAna it underperforms ColBERTv2 (Section 5.5). **(2)** Adaptive RRF with per-query IDF weighting improves NDCG@10 on all 5 BEIR datasets versus fixed weights (up to +21.4% on ArguAna), achieving 0.7263 on SciFact with BGE-small. **(3)** A negative result on post-RRF scoring: frequency+decay, history-augmented recall, and cross-encoder reranking all failed to improve NDCG. **(4)** A production-grade substrate with integrity checking, schema versioning, ranking diagnostics, and a distance-based relevance signal validated on 50,425 relevance-judged queries across the 5 BEIR datasets.
Search latency remains 20.9 ms median at 50K chunks with stable NDCG. The fine-tuned model is published as `Stffens/bge-small-rrf-v2` on HuggingFace. All code, data, and experiments are open-source. 

---
# Scattered Hypothesis Generation for Open-Ended Event Forecasting 

**Authors**: He Chang, Zhulin Tao, Lifang Yang, Xianglin Huang, Yunshan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.15788)  

**Abstract**: Despite the importance of open-ended event forecasting for risk management, current LLM-based methods predominantly target only the most probable outcomes, neglecting the intrinsic uncertainty of real-world events. To bridge this gap, we advance open-ended event forecasting from pinpoint forecasting to scatter forecasting by introducing the proxy task of hypothesis generation. This paradigm aims to generate an inclusive and diverse set of hypotheses that broadly cover the space of plausible future events. To this end, we propose SCATTER, a reinforcement learning framework that jointly optimizes inclusiveness and diversity of the hypothesis. Specifically, we design a novel hybrid reward that consists of three components: 1) a validity reward that measures semantic alignment with observed events, 2) an intra-group diversity reward to encourage variation within sampled responses, and 3) an inter-group diversity reward to promote exploration across distinct modes. By integrating the validity-gated score into the overall objective, we confine the exploration of wildly diversified outcomes to contextually plausible futures, preventing the mode collapse issue. Experiments on two real-world benchmark datasets, i.e., OpenForecast and OpenEP, demonstrate that SCATTER significantly outperforms strong baselines. Our code is available at this https URL. 

---
