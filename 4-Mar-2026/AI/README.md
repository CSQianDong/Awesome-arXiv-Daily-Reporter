# Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals 

**Authors**: Achyutha Menon, Magnus Saebo, Tyler Crosse, Spencer Gibson, Eyon Jang, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2603.03258)  

**Abstract**: The accelerating adoption of language models (LMs) as agents for deployment in long-context tasks motivates a thorough understanding of goal drift: agents' tendency to deviate from an original objective. While prior-generation language model agents have been shown to be susceptible to drift, the extent to which drift affects more recent models remains unclear. In this work, we provide an updated characterization of the extent and causes of goal drift. We investigate drift in state-of-the-art models within a simulated stock-trading environment (Arike et al., 2025). These models are largely shown to be robust even when subjected to adversarial pressure. We show, however, that this robustness is brittle: across multiple settings, the same models often inherit drift when conditioned on prefilled trajectories from weaker agents. The extent of conditioning-induced drift varies significantly by model family, with only GPT-5.1 maintaining consistent resilience among tested models. We find that drift behavior is inconsistent between prompt variations and correlates poorly with instruction hierarchy following behavior, with strong hierarchy following failing to reliably predict resistance to drift. Finally, we run analogous experiments in a new emergency room triage environment to show preliminary evidence for the transferability of our results across qualitatively different settings. Our findings underscore the continued vulnerability of modern LM agents to contextual pressures and the need for refined post-training techniques to mitigate this. 

---
# Valet: A Standardized Testbed of Traditional Imperfect-Information Card Games 

**Authors**: Mark Goadrich, Achille Morenville, Éric Piette  

**Link**: [PDF](https://arxiv.org/pdf/2603.03252)  

**Abstract**: AI algorithms for imperfect-information games are typically compared using performance metrics on individual games, making it difficult to assess robustness across game choices. Card games are a natural domain for imperfect information due to hidden hands and stochastic draws. To facilitate comparative research on imperfect-information game-playing algorithms and game systems, we introduce Valet, a diverse and comprehensive testbed of 21 traditional imperfect-information card games. These games span multiple genres, cultures, player counts, deck structures, mechanics, winning conditions, and methods of hiding and revealing information. To standardize implementations across systems, we encode the rules of each game in RECYCLE, a card game description language. We empirically characterize each game's branching factor and duration using random simulations, reporting baseline score distributions for a Monte Carlo Tree Search player against random opponents to demonstrate the suitability of Valet as a benchmarking suite. 

---
# Density-Guided Response Optimization: Community-Grounded Alignment via Implicit Acceptance Signals 

**Authors**: Patrick Gerard, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2603.03242)  

**Abstract**: Language models deployed in online communities must adapt to norms that vary across social, cultural, and domain-specific contexts. Prior alignment approaches rely on explicit preference supervision or predefined principles, which are effective for well-resourced settings but exclude most online communities -- particularly those without institutional backing, annotation infrastructure, or organized around sensitive topics -- where preference elicitation is costly, ethically fraught, or culturally misaligned.
We observe that communities already express preferences implicitly through what content they accept, engage with, and allow to persist. We show that this acceptance behavior induces measurable geometric structure in representation space: accepted responses occupy coherent, high-density regions that reflect community-specific norms, while rejected content falls in sparser or misaligned areas. We operationalize this structure as an implicit preference signal for alignment and introduce density-guided response optimization (DGRO), a method that aligns language models to community norms without requiring explicit preference labels.
Using labeled preference data, we demonstrate that local density recovers pairwise community judgments, indicating that geometric structure encodes meaningful preference signal. We then apply DGRO in annotation-scarce settings across diverse communities spanning platform, topic, and language. DGRO-aligned models consistently produce responses preferred by human annotators, domain experts, and model-based judges over supervised and prompt-based baselines. We position DGRO as a practical alignment alternative for communities where explicit preference supervision is unavailable or misaligned with situated practices, and discuss the implications and risks of learning from emergent acceptance behavior. 

---
# AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework 

**Authors**: Zihang Zeng, Jiaquan Zhang, Pengze Li, Yuan Qi, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03233)  

**Abstract**: Large Language Models (LLMs) demonstrate potentials for automating scientific code generation but face challenges in reliability, error propagation in multi-agent workflows, and evaluation in domains with ill-defined success metrics. We present a Bayesian adversarial multi-agent framework specifically designed for AI for Science (AI4S) tasks in the form of a Low-code Platform (LCP). Three LLM-based agents are coordinated under the Bayesian framework: a Task Manager that structures user inputs into actionable plans and adaptive test cases, a Code Generator that produces candidate solutions, and an Evaluator providing comprehensive feedback. The framework employs an adversarial loop where the Task Manager iteratively refines test cases to challenge the Code Generator, while prompt distributions are dynamically updated using Bayesian principles by integrating code quality metrics: functional correctness, structural alignment, and static analysis. This co-optimization of tests and code reduces dependence on LLM reliability and addresses evaluation uncertainty inherent to scientific tasks. LCP also streamlines human-AI collaboration by translating non-expert prompts into domain-specific requirements, bypassing the need for manual prompt engineering by practitioners without coding backgrounds. Benchmark evaluations demonstrate LCP's effectiveness in generating robust code while minimizing error propagation. The proposed platform is also tested on an Earth Science cross-disciplinary task and demonstrates strong reliability, outperforming competing models. 

---
# NeuroSkill(tm): Proactive Real-Time Agentic System Capable of Modeling Human State of Mind 

**Authors**: Nataliya Kosmyna, Eugene Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2603.03212)  

**Abstract**: Real-time proactive agentic system, capable of modeling Human State of Mind, using foundation EXG model and text embeddings model, running fully offline on the edge. Unlike all previously known systems, the NeuroSkill(tm) system leverages this http URL description of Human's State of Mind via API and CLI provided by the system, directly from the Brain-Computer Interface (BCI) devices, which records Human biophysical and brain signals. Our custom harness - NeuroLoop(tm) - utilizes all of the above to run agentic flow that manages to engage with the Human on multiple cognitive and affective levels of their State of Mind (e.g., empathy), by providing actionable tool calls and protocol execution with explicit or implicit requests from the Human. GPLv3 open-source software with ethically aligned AI100 licensing for the skill markdown. 

---
# No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models 

**Authors**: Omer Sela  

**Link**: [PDF](https://arxiv.org/pdf/2603.03203)  

**Abstract**: CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. With low-rank adaptation, models can learn from contaminated data without memorizing it, and CDD performs at chance level even when the data is verifiably contaminated. Only when fine-tuning capacity is sufficient to induce memorization does CDD recover strong detection accuracy. Our results characterize a memorization threshold that governs detectability and highlight a practical consideration: parameter-efficient fine-tuning can produce contamination that output-distribution methods do not detect. Our code is available at this https URL 

---
# Expectation and Acoustic Neural Network Representations Enhance Music Identification from Brain Activity 

**Authors**: Shogo Noguchi, Taketo Akama, Tai Nakamura, Shun Minamikawa, Natalia Polouliakh  

**Link**: [PDF](https://arxiv.org/pdf/2603.03190)  

**Abstract**: During music listening, cortical activity encodes both acoustic and expectation-related information. Prior work has shown that ANN representations resemble cortical representations and can serve as supervisory signals for EEG recognition. Here we show that distinguishing acoustic and expectation-related ANN representations as teacher targets improves EEG-based music identification. Models pretrained to predict either representation outperform non-pretrained baselines, and combining them yields complementary gains that exceed strong seed ensembles formed by varying random initializations. These findings show that teacher representation type shapes downstream performance and that representation learning can be guided by neural encoding. This work points toward advances in predictive music cognition and neural decoding. Our expectation representation, computed directly from raw signals without manual labels, reflects predictive structure beyond onset or pitch, enabling investigation of multilayer predictive encoding across diverse stimuli. Its scalability to large, diverse datasets further suggests potential for developing general-purpose EEG models grounded in cortical encoding principles. 

---
# Neuro-Symbolic Artificial Intelligence: A Task-Directed Survey in the Black-Box Models Era 

**Authors**: Giovanni Pio Delvecchio, Lorenzo Molfetta, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2603.03177)  

**Abstract**: The integration of symbolic computing with neural networks has intrigued researchers since the first theorizations of Artificial intelligence (AI). The ability of Neuro-Symbolic (NeSy) methods to infer or exploit behavioral schema has been widely considered as one of the possible proxies for human-level intelligence. However, the limited semantic generalizability and the challenges in declining complex domains with pre-defined patterns and rules hinder their practical implementation in real-world scenarios. The unprecedented results achieved by connectionist systems since the last AI breakthrough in 2017 have raised questions about the competitiveness of NeSy solutions, with particular emphasis on the Natural Language Processing and Computer Vision fields. This survey examines task-specific advancements in the NeSy domain to explore how incorporating symbolic systems can enhance explainability and reasoning capabilities. Our findings are meant to serve as a resource for researchers exploring explainable NeSy methodologies for real-life tasks and applications. Reproducibility details and in-depth comments on each surveyed research work are made available at this https URL. 

---
# FEAST: Retrieval-Augmented Multi-Hierarchical Food Classification for the FoodEx2 System 

**Authors**: Lorenzo Molfetta, Alessio Cocchieri, Stefano Fantazzini, Giacomo Frisoni, Luca Ragazzi, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2603.03176)  

**Abstract**: Hierarchical text classification (HTC) and extreme multi-label classification (XML) tasks face compounded challenges from complex label interdependencies, data sparsity, and extreme output dimensions. These challenges are exemplified in the European Food Safety Authority's FoodEx2 system-a standardized food classification framework essential for food consumption monitoring and contaminant exposure assessment across Europe. FoodEx2 coding transforms natural language food descriptions into a set of codes from multiple standardized hierarchies, but faces implementation barriers due to its complex structure. Given a food description (e.g., "organic yogurt''), the system identifies its base term ("yogurt''), all the applicable facet categories (e.g., "production method''), and then, every relevant facet descriptors to each category (e.g., "organic production''). While existing models perform adequately on well-balanced and semantically dense hierarchies, no work has been applied on the practical constraints imposed by the FoodEx2 system. The limited literature addressing such real-world scenarios further compounds these challenges. We propose FEAST (Food Embedding And Semantic Taxonomy), a novel retrieval-augmented framework that decomposes FoodEx2 classification into a three-stage approach: (1) base term identification, (2) multi-label facet prediction, and (3) facet descriptor assignment. By leveraging the system's hierarchical structure to guide training and performing deep metric learning, FEASTlearns discriminative embeddings that mitigate data sparsity and improve generalization on rare and fine-grained labels. Evaluated on the multilingual FoodEx2 benchmark, FEAST outperforms the prior European's CNN baseline F1 scores by 12-38 % on rare classes. 

---
# Saarthi for AGI: Towards Domain-Specific General Intelligence for Formal Verification 

**Authors**: Aman Kumar, Deepak Narayan Gadde, Luu Danh Minh, Vaisakh Naduvodi Viswambharan, Keerthan Kopparam Radhakrishna, Sivaram Pothireddypalli  

**Link**: [PDF](https://arxiv.org/pdf/2603.03175)  

**Abstract**: Saarthi is an agentic AI framework that uses multi-agent collaboration to perform end-to-end formal verification. Even though the framework provides a complete flow from specification to coverage closure, with around 40% efficacy, there are several challenges that need to be addressed to make it more robust and reliable. Artificial General Intelligence (AGI) is still a distant goal, and current Large Language Model (LLM)-based agents are prone to hallucinations and making mistakes, especially when dealing with complex tasks such as formal verification. However, with the right enhancements and improvements, we believe that Saarthi can be a significant step towards achieving domain-specific general intelligence for formal verification. Especially for problems that require Short Term, Short Context (STSC) capabilities, such as formal verification, Saarthi can be a powerful tool to assist verification engineers in their work. In this paper, we present two key enhancements to the Saarthi framework: (1) a structured rulebook and specification grammar to improve the accuracy and controllability of SystemVerilog Assertion (SVA) generation, and (2) integration of advanced Retrieval Augmented Generation (RAG) techniques, such as GraphRAG, to provide agents with access to technical knowledge and best practices for iterative refinement and improvement of outputs. We also benchmark these enhancements for the overall Saarthi framework using challenging test cases from NVIDIA's CVDP benchmark targeting formal verification. Our benchmark results stand out with a 70% improvement in the accuracy of generated assertions, and a 50% reduction in the number of iterations required to achieve coverage closure. 

---
# Agentic AI-based Coverage Closure for Formal Verification 

**Authors**: Sivaram Pothireddypalli, Ashish Raman, Deepak Narayan Gadde, Aman Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.03147)  

**Abstract**: Coverage closure is a critical requirement in Integrated Chip (IC) development process and key metric for verification sign-off. However, traditional exhaustive approaches often fail to achieve full coverage within project timelines. This study presents an agentic AI-driven workflow that utilizes Large Language Model (LLM)-enabled Generative AI (GenAI) to automate coverage analysis for formal verification, identify coverage gaps, and generate the required formal properties. The framework accelerates verification efficiency by systematically addressing coverage holes. Benchmarking open-source and internal designs reveals a measurable increase in coverage metrics, with improvements correlated to the complexity of the design. Comparative analysis validates the effectiveness of this approach. These results highlight the potential of agentic AI-based techniques to improve formal verification productivity and support comprehensive coverage closure. 

---
# AI Space Physics: Constitutive boundary semantics for open AI institutions 

**Authors**: Oleg Romanchuk, Roman Bondar  

**Link**: [PDF](https://arxiv.org/pdf/2603.03119)  

**Abstract**: Agentic AI deployments increasingly behave as persistent institutions rather than one-shot inference endpoints: they accumulate state, invoke external tools, coordinate multiple runtimes, and modify their future authority surface over time. Existing governance language typically specifies decision-layer constraints but leaves the causal mechanics of boundary crossing underdefined, particularly for transitions that do not immediately change the external world yet expand what the institution can later do.
This paper introduces AI Space Physics as a constitutive semantics for open, self-expanding AI institutions. We define a minimal state model with typed boundary channels, horizon-limited reach semantics, and a membrane-witness discipline. The core law family (P-1, P-1a, P-1b, P-1c) requires witness completeness, non-bypass mediation, atomic adjudication-to-effect transitions, and replayable reconstruction of adjudication class. We explicitly separate second-order effects into structural expansion and policy broadening, and treat expansion transitions as governance-relevant even when immediate external deltas are zero.
The novelty claim is precise rather than expansive: this work does not introduce mediation as a concept; it reclassifies authority-surface expansion as a first-class boundary event with constitutive witness obligations. In this semantics, expansion without immediate commit remains adjudication-relevant. 

---
# Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation 

**Authors**: Hongliu Cao, Ilias Driouich, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2603.03116)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly adopted in high-stakes settings, but current benchmarks evaluate mainly whether a task was completed, not how. We introduce Procedure-Aware Evaluation (PAE), a framework that formalizes agent procedures as structured observations and exposes consistency relationships between what agents observe, communicate, and execute. PAE evaluates agents along complementary axes (Utility, Efficiency, Interaction Quality, Procedural Integrity) and applies multi-dimensional gating that categorically disqualifies corrupt outcomes. Evaluating state-of-the-art LLM agents on tau-bench yields findings at the axis, compliance, and benchmark levels. At the axis level, the dimensions capture non-redundant failure modes: utility masks reliability gaps, speed does not imply precision, and conciseness does not predict intent adherence. At the procedural compliance level, 27-78% of benchmark reported successes are corrupt successes concealing violations across interaction and integrity. Furthermore, gating substantially collapses Pass^4 rate and affects model rankings. The analysis of corrupt success cases reveals distinctive per-model failure signatures: GPT-5 spreads errors across policy, execution, and intent dimensions; Kimi-K2-Thinking concentrates 78% of violations in policy faithfulness and compliance; and Mistral-Large-3 is dominated by faithfulness failures. At the benchmark level, our analysis exposes structural flaws in the benchmark design, including task scope gaps, contradictory reward signals, and simulator artifacts that produce accidental successes. 

---
# Odin: Multi-Signal Graph Intelligence for Autonomous Discovery in Knowledge Graphs 

**Authors**: Muyukani Kizito, Elizabeth Nyambere  

**Link**: [PDF](https://arxiv.org/pdf/2603.03097)  

**Abstract**: We present Odin, the first production-deployed graph intelligence engine for autonomous discovery of meaningful patterns in knowledge graphs without prior specification. Unlike retrieval-based systems that answer predefined queries, Odin guides exploration through the COMPASS (Composite Oriented Multi-signal Path Assessment) score, a novel metric that combines (1) structural importance via Personalized PageRank, (2) semantic plausibility through Neural Probabilistic Logic Learning (NPLL) used as a discriminative filter rather than generative model, (3) temporal relevance with configurable decay, and (4) community-aware guidance through GNN-identified bridge entities and inter-community affinity scores. This multi-signal integration, particularly the bridge scoring mechanism, addresses the "echo chamber" problem where graph exploration becomes trapped in dense local communities. We formalize the autonomous discovery problem, prove theoretical properties of our scoring function, and demonstrate that beam search with multi-signal guidance achieves $O(b \cdot h)$ complexity while maintaining high recall compared to exhaustive exploration. To our knowledge, Odin represents the first autonomous discovery system deployed in regulated production environments (healthcare and insurance), demonstrating significant improvements in pattern discovery quality and analyst efficiency. Our approach maintains complete provenance traceability -- a critical requirement for regulated industries where hallucination is unacceptable. 

---
# Beyond Factual Correctness: Mitigating Preference-Inconsistent Explanations in Explainable Recommendation 

**Authors**: Chengkai Wang, Baisong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03080)  

**Abstract**: LLM-based explainable recommenders can produce fluent explanations that are factually correct, yet still justify items using attributes that conflict with a user's historical preferences. Such preference-inconsistent explanations yield logically valid but unconvincing reasoning and are largely missed by standard hallucination or faithfulness metrics. We formalize this failure mode and propose PURE, a preference-aware reasoning framework following a select-then-generate paradigm. Instead of only improving generation, PURE intervenes in evidence selection, it selects a compact set of multi-hop item-centric reasoning paths that are both factually grounded and aligned with user preference structure, guided by user intent, specificity, and diversity to suppress generic, weakly personalized evidence. The selected evidence is then injected into LLM generation via structure-aware prompting that preserves relational constraints. To measure preference inconsistency, we introduce a feature-level, user-centric evaluation metric that reveals misalignment overlooked by factuality-based measures. Experiments on three real-world datasets show that PURE consistently reduces preference-inconsistent explanations and factual hallucinations while maintaining competitive recommendation accuracy, explanation quality, and inference efficiency. These results highlight that trustworthy explanations require not only factual correctness but also justification aligned with user preferences. 

---
# RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization 

**Authors**: Siwei Zhang, Yun Xiong, Xi Chen, Zi'an Jia, Renhong Huang, Jiarong Xu, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03078)  

**Abstract**: Agentic Reinforcement Learning (Agentic RL) has shown remarkable potential in large language model-based (LLM) agents. These works can empower LLM agents to tackle complex tasks via multi-step, tool-integrated reasoning. However, an inherent limitation of existing Agentic RL methods is their reliance on a pure on-policy paradigm for exploration, restricting exploration to the agent's self-generated outputs and preventing the discovery of new reasoning perspectives for further improvement. While recent efforts incorporate auxiliary off-policy signals to enhance exploration, they typically utilize full off-policy trajectories for trajectory-level policy estimation, overlooking the necessity for the fine-grained, step-level exploratory dynamics within agentic rollout. In this paper, we revisit exploration in Agentic RL and propose Retrieval-Augmented Policy Optimization (RAPO), a novel RL framework that introduces retrieval to explicitly expand exploration during training. To achieve this, we decompose the Agentic RL training process into two phases: (i) Hybrid-policy Agentic Rollout, and (ii) Retrieval-aware Policy Optimization. Specifically, we propose a Hybrid-policy Agentic Rollout strategy, which allows the agents to continuously reason over the retrieved off-policy step-level traces. It dynamically extends the reasoning receptive field of agents, enabling broader exploration conditioned on external behaviors. Subsequently, we introduce the Retrieval-aware Policy Optimization mechanism, which calibrates the policy gradient estimation with retrieval reward and importance shaping, stabilizing training and prioritizing retrieval-illuminating exploration. Extensive experiments show that RAPO achieves an +5.0% average gain on fourteen datasets across three agentic reasoning tasks, while delivering 1.2x faster training efficiency. 

---
# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning 

**Authors**: Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2603.03072)  

**Abstract**: Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available. 

---
# REGAL: A Registry-Driven Architecture for Deterministic Grounding of Agentic AI in Enterprise Telemetry 

**Authors**: Yuvraj Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2603.03018)  

**Abstract**: Enterprise engineering organizations produce high-volume, heterogeneous telemetry from version control systems, CI/CD pipelines, issue trackers, and observability platforms. Large Language Models (LLMs) enable new forms of agentic automation, but grounding such agents on private telemetry raises three practical challenges: limited model context, locally defined semantic concepts, and evolving metric interfaces.
We present REGAL, a registry-driven architecture for deterministic grounding of agentic AI systems in enterprise telemetry. REGAL adopts an explicitly architectural approach: deterministic telemetry computation is treated as a first-class primitive, and LLMs operate over a bounded, version-controlled action space rather than raw event streams.
The architecture combines (1) a Medallion ELT pipeline that produces replayable, semantically compressed Gold artifacts, and (2) a registry-driven compilation layer that synthesizes Model Context Protocol (MCP) tools from declarative metric definitions. The registry functions as an "interface-as-code" layer, ensuring alignment between tool specification and execution, mitigating tool drift, and embedding governance policies directly at the semantic boundary.
A prototype implementation and case study validate the feasibility of deterministic grounding and illustrate its implications for latency, token efficiency, and operational governance. This work systematizes an architectural pattern for enterprise LLM grounding; it does not propose new learning algorithms, but rather elevates deterministic computation and semantic compilation to first-class design primitives for agentic systems. 

---
# OrchMAS: Orchestrated Reasoning with Multi Collaborative Heterogeneous Scientific Expert Structured Agents 

**Authors**: Yichao Feng, Haoran Luo, Zhenghong Lin, Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03005)  

**Abstract**: Multi-agent large language model frameworks are promising for complex multi step reasoning, yet existing systems remain weak for scientific and knowledge intensive domains due to static prompts and agent roles, rigid workflows, and homogeneous model reliance, leading to poor domain adaptation, limited reasoning flexibility, and high latency on heterogeneous or long-horizon scientific tasks. They also struggle to revise earlier decisions when intermediate reasoning diverges, reducing reliability in structured and calculation heavy settings. To address these limitations, we propose a scientific domain oriented interactive two tier multi model orchestration framework. A dedicated orchestration model analyzes each task, dynamically constructs a domain aware reasoning pipeline, and instantiates specialized expert agents with tailored prompts, while an execution model performs each step under generated role and instruction specifications. The orchestrator iteratively updates the pipeline based on intermediate feedback, enabling dynamic replanning, role reallocation, and prompt refinement across multi turn interactions, strengthening robustness and specialization for scientific reasoning through structured heterogeneous model collaboration. The framework is model agnostic and supports heterogeneous LLM integration with different capacities or costs, enabling flexible performance efficiency trade offs in practical scientific deployments. Experiments show consistent improvements over existing multi agent systems and strong baselines across diverse reasoning and scientific style benchmarks. 

---
# SpatialText: A Pure-Text Cognitive Benchmark for Spatial Understanding in Large Language Models 

**Authors**: Peiyao Jiang, Zequn Qin, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.03002)  

**Abstract**: Genuine spatial reasoning relies on the capacity to construct and manipulate coherent internal spatial representations, often conceptualized as mental models, rather than merely processing surface linguistic associations. While large language models exhibit advanced capabilities across various domains, existing benchmarks fail to isolate this intrinsic spatial cognition from statistical language heuristics. Furthermore, multimodal evaluations frequently conflate genuine spatial reasoning with visual perception. To systematically investigate whether models construct flexible spatial mental models, we introduce SpatialText, a theory-driven diagnostic framework. Rather than functioning simply as a dataset, SpatialText isolates text-based spatial reasoning through a dual-source methodology. It integrates human-annotated descriptions of real 3D indoor environments, which capture natural ambiguities, perspective shifts, and functional relations, with code-generated, logically precise scenes designed to probe formal spatial deduction and epistemic boundaries. Systematic evaluation across state-of-the-art models reveals fundamental representational limitations. Although models demonstrate proficiency in retrieving explicit spatial facts and operating within global, allocentric coordinate systems, they exhibit critical failures in egocentric perspective transformation and local reference frame reasoning. These systematic errors provide strong evidence that current models rely heavily on linguistic co-occurrence heuristics rather than constructing coherent, verifiable internal spatial representations. SpatialText thus serves as a rigorous instrument for diagnosing the cognitive boundaries of artificial spatial intelligence. 

---
# Architecting Trust in Artificial Epistemic Agents 

**Authors**: Nahema Marchal, Stephanie Chan, Matija Franklin, Manon Revel, Geoff Keeling, Roberta Fischli, Bilva Chandra, Iason Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2603.02960)  

**Abstract**: Large language models increasingly function as epistemic agents -- entities that can 1) autonomously pursue epistemic goals and 2) actively shape our shared knowledge environment. They curate the information we receive, often supplanting traditional search-based methods, and are frequently used to generate both personal and deeply specialized advice. How they perform these functions, including whether they are reliable and properly calibrated to both individual and collective epistemic norms, is therefore highly consequential for the choices we make. We argue that the potential impact of epistemic AI agents on practices of knowledge creation, curation and synthesis, particularly in the context of complex multi-agent interactions, creates new informational interdependencies that necessitate a fundamental shift in evaluation and governance of AI. While a well-calibrated ecosystem could augment human judgment and collective decision-making, poorly aligned agents risk causing cognitive deskilling and epistemic drift, making the calibration of these models to human norms a high-stakes necessity. To ensure a beneficial human-AI knowledge ecosystem, we propose a framework centered on building and cultivating the trustworthiness of epistemic AI agents; aligning AI these agents with human epistemic goals; and reinforcing the surrounding socio-epistemic infrastructure. In this context, trustworthy AI agents must demonstrate epistemic competence, robust falsifiability, and epistemically virtuous behaviors, supported by technical provenance systems and "knowledge sanctuaries" designed to protect human resilience. This normative roadmap provides a path toward ensuring that future AI systems act as reliable partners in a robust and inclusive knowledge ecosystem. 

---
# ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large Language Models via Group Relative Policy Optimization 

**Authors**: Yang Zhan, Yunhao Li, Zhang Chao, Yuxu Lu, Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02939)  

**Abstract**: Recent advancements in reinforcement fine-tuning have significantly improved the reasoning ability of large language models (LLMs). In particular, methods such as group relative policy optimization (GRPO) have demonstrated strong capabilities across various fields. However, applying LLMs to ship trajectory prediction remains largely unexplored. In this paper, we propose ShipTraj-R1, a novel LLM-based framework that reformulates ship trajectory prediction as a text-to-text generation problem. (1) We design a dynamic prompt containing trajectory information about conflicting ships to guide the model to achieve adaptive chain-of-thought (CoT) reasoning. (2) We introduce a comprehensive rule-based reward mechanism to incentivize the reasoning format and prediction accuracy of the model. (3) Our ShipTraj-R1 is reinforced through the GRPO mechanism guided by domain-specific prompts and rewards, and utilizes the Qwen3 as the model backbone. Extensive experimental results on two complex and real-world maritime datasets show that the proposed ShipTraj-R1 achieves the least error compared with state-of-the-art deep learning and LLM-based baselines. 

---
# SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training 

**Authors**: Qi Zhang, Yifei Wang, Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02908)  

**Abstract**: In recent years, pre-trained large language models have achieved remarkable success across diverse tasks. Besides the pivotal role of self-supervised pre-training, their effectiveness in downstream applications also depends critically on the post-training process, which adapts models to task-specific data and objectives. However, this process inevitably introduces model shifts that can influence performance in different domains, and how such shifts transfer remains poorly understood. To open up the black box, we propose the SAE-based Transferability Score (STS), a new metric that leverages sparse autoencoders (SAEs) to forecast post-training transferability. Taking supervised fine-tuning as an example, STS identifies shifted dimensions in SAE representations and calculates their correlations with downstream domains, enabling reliable estimation of transferability \textit{before} fine-tuning. Extensive experiments across multiple models and domains show that STS accurately predicts the transferability of supervised fine-tuning, achieving Pearson correlation coefficients above 0.7 with actual performance changes. Beyond this, we take an initial step toward extending STS to reinforcement learning. We believe that STS can serve as an {\color{black} interpretable} tool for guiding post-training strategies in LLMs. Code is available at this https URL. 

---
# Retrievit: In-context Retrieval Capabilities of Transformers, State Space Models, and Hybrid Architectures 

**Authors**: Georgios Pantazopoulos, Malvina Nikandrou, Ioannis Konstas, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2603.02874)  

**Abstract**: Transformers excel at in-context retrieval but suffer from quadratic complexity with sequence length, while State Space Models (SSMs) offer efficient linear-time processing but have limited retrieval capabilities. We investigate whether hybrid architectures combining Transformers and SSMs can achieve the best of both worlds on two synthetic in-context retrieval tasks. The first task, n-gram retrieval, requires the model to identify and reproduce an n-gram that succeeds the query within the input sequence. The second task, position retrieval, presents the model with a single query token and requires it to perform a two-hop associative lookup: first locating the corresponding element in the sequence, and then outputting its positional index. Under controlled experimental conditions, we assess data efficiency, length generalization, robustness to out of domain training examples, and learned representations across Transformers, SSMs, and hybrid architectures. We find that hybrid models outperform SSMs and match or exceed Transformers in data efficiency and extrapolation for information-dense context retrieval. However, Transformers maintain superiority in position retrieval tasks. Through representation analysis, we discover that SSM-based models develop locality-aware embeddings where tokens representing adjacent positions become neighbors in embedding space, forming interpretable structures. This emergent property, absent in Transformers, explains both the strengths and limitations of SSMs and hybrids for different retrieval tasks. Our findings provide principled guidance for architecture selection based on task requirements and reveal fundamental differences in how Transformers and SSMs, and hybrid models learn positional associations. 

---
# LLM-based Argument Mining meets Argumentation and Description Logics: a Unified Framework for Reasoning about Debates 

**Authors**: Gianvincenzo Alfano, Sergio Greco, Lucio La Cava, Stefano Francesco Monea, Irina Trubitsyna  

**Link**: [PDF](https://arxiv.org/pdf/2603.02858)  

**Abstract**: Large Language Models (LLMs) achieve strong performance in analyzing and generating text, yet they struggle with explicit, transparent, and verifiable reasoning over complex texts such as those containing debates. In particular, they lack structured representations that capture how arguments support or attack each other and how their relative strengths determine overall acceptability. We encompass these limitations by proposing a framework that integrates learning-based argument mining with quantitative reasoning and ontology-based querying. Starting from a raw debate text, the framework extracts a fuzzy argumentative knowledge base, where arguments are explicitly represented as entities, linked by attack and support relations, and annotated with initial fuzzy strengths reflecting plausibility w.r.t. the debate's context. Quantitative argumentation semantics are then applied to compute final argument strengths by propagating the effects of supports and attacks. These results are then embedded into a fuzzy description logic setting, enabling expressive query answering through efficient rewriting techniques. The proposed approach provides a transparent, explainable, and formally grounded method for analyzing debates, overcoming purely statistical LLM-based analyses. 

---
# Guideline-Grounded Evidence Accumulation for High-Stakes Agent Verification 

**Authors**: Yichi Zhang, Nabeel Seedat, Yinpeng Dong, Peng Cui, Jun Zhu, Mihaela van de Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02798)  

**Abstract**: As LLM-powered agents have been used for high-stakes decision-making, such as clinical diagnosis, it becomes critical to develop reliable verification of their decisions to facilitate trustworthy deployment. Yet, existing verifiers usually underperform owing to a lack of domain knowledge and limited calibration. To address this, we establish GLEAN, an agent verification framework with Guideline-grounded Evidence Accumulation that compiles expert-curated protocols into trajectory-informed, well-calibrated correctness signals. GLEAN evaluates the step-wise alignment with domain guidelines and aggregates multi-guideline ratings into surrogate features, which are accumulated along the trajectory and calibrated into correctness probabilities using Bayesian logistic regression. Moreover, the estimated uncertainty triggers active verification, which selectively collects additional evidence for uncertain cases via expanding guideline coverage and performing differential checks. We empirically validate GLEAN with agentic clinical diagnosis across three diseases from the MIMIC-IV dataset, surpassing the best baseline by 12% in AUROC and 50% in Brier score reduction, which confirms the effectiveness in both discrimination and calibration. In addition, the expert study with clinicians recognizes GLEAN's utility in practice. 

---
# Agentified Assessment of Logical Reasoning Agents 

**Authors**: Zhiyu Ni, Yifeng Xiao, Zheng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02788)  

**Abstract**: We present a framework for evaluating and benchmarking logical reasoning agents when assessment itself must be reproducible, auditable, and robust to execution failures. Building on agentified assessment, we use an assessor agent to issue tasks, enforce execution budgets, parse outputs, and record structured failure types, while the agent under test only needs to expose a standardized agent-to-agent interface. As a case study, we benchmark an auto-formalization agent for first-order logic (FOL) reasoning on a solver-verified and repaired split of FOLIO. The agent translates natural language premises and conclusions into executable Z3Py programs and employs satisfiability modulo theories (SMT) solving to determine logical entailment. On the cleaned FOLIO validation set, the auto-formalization agent achieves 86.70% accuracy under the assessor protocol, outperforming a chain-of-thought baseline (73.89%). 

---
# Rethinking Code Similarity for Automated Algorithm Design with LLMs 

**Authors**: Rui Zhang, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02787)  

**Abstract**: The rise of Large Language Model-based Automated Algorithm Design (LLM-AAD) has transformed algorithm development by autonomously generating code implementations of expert-level algorithms. Unlike traditional expert-driven algorithm development, in the LLM-AAD paradigm, the main design principle behind an algorithm is often implicitly embedded in the generated code. Therefore, assessing algorithmic similarity directly from code, distinguishing genuine algorithmic innovation from mere syntactic variation, becomes essential. While various code similarity metrics exist, they fail to capture algorithmic similarity, as they focus on surface-level syntax or output equivalence rather than the underlying algorithmic logic.
We propose BehaveSim, a novel method to measure algorithmic similarity through the lens of problem-solving behavior as a sequence of intermediate solutions produced during execution, dubbed as problem-solving trajectories (PSTrajs). By quantifying the alignment between PSTrajs using dynamic time warping (DTW), BehaveSim distinguishes algorithms with divergent logic despite syntactic or output-level similarities. We demonstrate its utility in two key applications: (i) Enhancing LLM-AAD: Integrating BehaveSim into existing LLM-AAD frameworks (e.g., FunSearch, EoH) promotes behavioral diversity, significantly improving performance on three AAD tasks. (ii) Algorithm analysis: BehaveSim clusters generated algorithms by behavior, enabling systematic analysis of problem-solving strategies--a crucial tool for the growing ecosystem of AI-generated algorithms. Data and code of this work are open-sourced at this https URL. 

---
# EvoSkill: Automated Skill Discovery for Multi-Agent Systems 

**Authors**: Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02766)  

**Abstract**: Coding agents are increasingly used as general-purpose problem solvers, but their flexibility does not by itself confer the domain expertise needed for specialized tasks. Recent work addresses this through \textit{agent skills}: reusable workflows, and code, that augment agents with domain-specific capabilities. Most skills today are hand-crafted, and existing evolutionary approaches optimize low-level artifacts (e.g. prompts \& code) that are tightly coupled to specific models and tasks. We introduce \textbf{EvoSkill}, a self-evolving framework that automatically discovers and refines agent skills through iterative failure analysis. EvoSkill analyzes execution failures, proposes new skills or edits to existing ones, and materializes them into structured, reusable skill folders. A Pareto frontier of agent programs governs selection, retaining only skills that improve held-out validation performance while the underlying model remains frozen. We evaluate EvoSkill on two benchmarks: OfficeQA, a grounded reasoning benchmark over U.S.\ Treasury data, where it improves exact-match accuracy by \textbf{7.3\%} (60.6\% $\to$ 67.9\%); and SealQA, a search-augmented QA benchmark with noisy retrieval, where it yields a \textbf{12.1\%} gain (26.6\% $\to$ 38.7\%). We also investigate the zero-shot transfer capabilties of skills evolved on one task to the other; in particular: skills evolved from SealQA transfers zero-shot to BrowseComp, improving accuracy by \textbf{5.3\%} without modification demonstrating that skill-level optimization produces transferable capabilities beyond the training task. 

---
# A Natural Language Agentic Approach to Study Affective Polarization 

**Authors**: Stephanie Anneris Malvicini, Ewelina Gajewska, Arda Derbent, Katarzyna Budzynska, Jarosław A. Chudziak, Maria Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2603.02711)  

**Abstract**: Affective polarization has been central to political and social studies, with growing focus on social media, where partisan divisions are often exacerbated. Real-world studies tend to have limited scope, while simulated studies suffer from insufficient high-quality training data, as manually labeling posts is labor-intensive and prone to subjective biases. The lack of adequate tools to formalize different definitions of affective polarization across studies complicates result comparison and hinders interoperable frameworks. We present a multi-agent model providing a comprehensive approach to studying affective polarization in social media. To operationalize our framework, we develop a platform leveraging large language models (LLMs) to construct virtual communities where agents engage in discussions. We showcase the potential of our platform by (1) analyzing questions related to affective polarization, as explored in social science literature, providing a fresh perspective on this phenomenon, and (2) introducing scenarios that allow observation and measurement of polarization at different levels of granularity and abstraction. Experiments show that our platform is a flexible tool for computational studies of complex social dynamics such as affective polarization. It leverages advanced agent models to simulate rich, context-sensitive interactions and systematically explore research questions traditionally addressed through human-subject studies. 

---
# FinTexTS: Financial Text-Paired Time-Series Dataset via Semantic-Based and Multi-Level Pairing 

**Authors**: Jaehoon Lee, Suhwan Park, Tae Yoon Lim, Seunghan Lee, Jun Seo, Dongwan Kang, Hwanil Choi, Minjae Kim, Sungdong Yoo, SoonYoung Lee, Yongjae Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.02702)  

**Abstract**: The financial domain involves a variety of important time-series problems. Recently, time-series analysis methods that jointly leverage textual and numerical information have gained increasing attention. Accordingly, numerous efforts have been made to construct text-paired time-series datasets in the financial domain. However, financial markets are characterized by complex interdependencies, in which a company's stock price is influenced not only by company-specific events but also by events in other companies and broader macroeconomic factors. Existing approaches that pair text with financial time-series data based on simple keyword matching often fail to capture such complex relationships. To address this limitation, we propose a semantic-based and multi-level pairing framework. Specifically, we extract company-specific context for the target company from SEC filings and apply an embedding-based matching mechanism to retrieve semantically relevant news articles based on this context. Furthermore, we classify news articles into four levels (macro-level, sector-level, related company-level, and target-company level) using large language models (LLMs), enabling multi-level pairing of news articles with the target company. Applying this framework to publicly-available news datasets, we construct \textbf{FinTexTS}, a new large-scale text-paired stock price dataset. Experimental results on \textbf{FinTexTS} demonstrate the effectiveness of our semantic-based and multi-level pairing strategy in stock price forecasting. In addition to publicly-available news underlying \textbf{FinTexTS}, we show that applying our method to proprietary yet carefully curated news sources leads to higher-quality paired data and improved stock price forecasting performance. 

---
# Retrieval-Augmented Robots via Retrieve-Reason-Act 

**Authors**: Izat Temiraliev, Diji Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02688)  

**Abstract**: To achieve general-purpose utility, we argue that robots must evolve from passive executors into active Information Retrieval users. In strictly zero-shot settings where no prior demonstrations exist, robots face a critical information gap, such as the exact sequence required to assemble a complex furniture kit, that cannot be satisfied by internal parametric knowledge (common sense) or past internal memory. While recent robotic works attempt to use search before action, they primarily focus on retrieving past kinematic trajectories (analogous to searching internal memory) or text-based safety rules (searching for constraints). These approaches fail to address the core information need of active task construction: acquiring unseen procedural knowledge from external, unstructured documentation. In this paper, we define the paradigm as Retrieval-Augmented Robotics (RAR), empowering the robot with the information-seeking capability that bridges the gap between visual documentation and physical actuation. We formulate the task execution as an iterative Retrieve-Reason-Act loop: the robot or embodied agent actively retrieves relevant visual procedural manuals from an unstructured corpus, grounds the abstract 2D diagrams to 3D physical parts via cross-modal alignment, and synthesizes executable plans. We validate this paradigm on a challenging long-horizon assembly benchmark. Our experiments demonstrate that grounding robotic planning in retrieved visual documents significantly outperforms baselines relying on zero-shot reasoning or few-shot example retrieval. This work establishes the basis of RAR, extending the scope of Information Retrieval from answering user queries to driving embodied physical actions. 

---
# LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization 

**Authors**: Yang Zhao, Zihao Li, Zhiyu Jiang, Dandan Ma, Ganchao Liu, Wenzhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02680)  

**Abstract**: While Large Language Models (LLMs) form the cornerstone of sequential decision-making agent development, they have inherent limitations in high-frequency decision tasks. Existing research mainly focuses on discrete embodied decision scenarios with low-frequency and significant semantic differences in state space (e.g., household planning). These methods suffer from limited performance in high-frequency decision-making tasks, since high-precision numerical state information in such tasks undergoes frequent updates with minimal fluctuations, and exhibiting policy misalignment between the learned sub-tasks and composite tasks. To address these issues, this paper proposes Normalized Action Reward guided Consistency Policy Optimization (NAR-CP). 1) Our method first acquires predefined dense rewards from environmental feedback of candidate actions via reward functions, then completes reward shaping through normalization, and theoretically verifies action reward normalization does not impair optimal policy. 2) To reduce policy misalignment in composite tasks, we use LLMs to infer sub-observation candidate actions and generate joint policies, with consistency loss ensuring precise alignment between global semantic policies and sub-semantic policies. Experiments on UAV pursuit, a typical high-frequency task, show our method delivers superior performance on independent and composite tasks with excellent generalization to unseen tasks. 

---
# SorryDB: Can AI Provers Complete Real-World Lean Theorems? 

**Authors**: Austin Letson, Leopoldo Sarra, Auguste Poiroux, Oliver Dressler, Paul Lezeau, Dhyan Aranha, Frederick Pu, Aaron Hill, Miguel Corredera Hidalgo, Julian Berman, George Tsoukalas, Lenny Taelman  

**Link**: [PDF](https://arxiv.org/pdf/2603.02668)  

**Abstract**: We present SorryDB, a dynamically-updating benchmark of open Lean tasks drawn from 78 real world formalization projects on GitHub. Unlike existing static benchmarks, often composed of competition problems, hillclimbing the SorryDB benchmark will yield tools that are aligned to the community needs, more usable by mathematicians, and more capable of understanding complex dependencies. Moreover, by providing a continuously updated stream of tasks, SorryDB mitigates test-set contamination and offers a robust metric for an agent's ability to contribute to novel formal mathematics projects. We evaluate a collection of approaches, including generalist large language models, agentic approaches, and specialized symbolic provers, over a selected snapshot of 1000 tasks from SorryDB. We show that current approaches are complementary: even though an agentic approach based on Gemini Flash is the most performant, it is not strictly better than other off-the-shelf large-language models, specialized provers, or even a curated list of Lean tactics. 

---
# See and Remember: A Multimodal Agent for Web Traversal 

**Authors**: Xinjun Wang, Shengyao Wang, Aimin Zhou, Hao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02626)  

**Abstract**: Autonomous web navigation requires agents to perceive complex visual environments and maintain long-term context, yet current Large Language Model (LLM) based agents often struggle with spatial disorientation and navigation loops. In this paper, we propose generally applicable V-GEMS(Visual Grounding and Explicit Memory System), a robust multimodal agent architecture designed for precise and resilient web traversal. Our agent integrates visual grounding to resolve ambiguous interactive elements and introduces an explicit memory stack with state tracking. This dual mechanism allows the agent to maintain a structured map of its traversal path, enabling valid backtracking and preventing cyclical failures in deep navigation tasks. We also introduce an updatable dynamic benchmark to rigorously evaluate adaptability. Experiments show V-GEMS significantly dominates the WebWalker baseline, achieving a substantial 28.7% performance gain. Code is available at this https URL. 

---
# AgentAssay: Token-Efficient Regression Testing for Non-Deterministic AI Agent Workflows 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2603.02601)  

**Abstract**: Autonomous AI agents are deployed at unprecedented scale, yet no principled methodology exists for
verifying that an agent has not regressed after changes to its prompts, tools, models, or
orchestration logic. We present AgentAssay, the first token-efficient framework for regression
testing non-deterministic AI agent workflows, achieving 78-100% cost reduction while maintaining
rigorous statistical guarantees. Our contributions include: (1) stochastic three-valued verdicts
(PASS/FAIL/INCONCLUSIVE) grounded in hypothesis testing; (2) five-dimensional agent coverage metrics;
(3) agent-specific mutation testing operators; (4) metamorphic relations for agent workflows; (5)
CI/CD deployment gates as statistical decision procedures; (6) behavioral fingerprinting that maps
execution traces to compact vectors, enabling multivariate regression detection; (7) adaptive budget
optimization calibrating trial counts to behavioral variance; and (8) trace-first offline analysis
enabling zero-cost testing on production traces. Experiments across 5 models (GPT-5.2, Claude Sonnet
4.6, Mistral-Large-3, Llama-4-Maverick, Phi-4), 3 scenarios, and 7,605 trials demonstrate that
behavioral fingerprinting achieves 86% detection power where binary testing has 0%, SPRT reduces
trials by 78%, and the full pipeline achieves 100% cost savings through trace-first analysis.
Implementation: 20,000+ lines of Python, 751 tests, 10 framework adapters. 

---
# SUN: Shared Use of Next-token Prediction for Efficient Multi-LLM Disaggregated Serving 

**Authors**: Sunghyeon Woo, Ahreum Seo, Jaegwang Lee, Jaeeun Kil, Hanbae Seo, Joonghoon Kim, Baeseong Park, Se Jung Kwon, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.02599)  

**Abstract**: In multi-model LLM serving, decode execution remains inefficient due to model-specific resource partitioning: since cross-model batching is not possible, memory-bound decoding often suffers from severe GPU underutilization, especially under skewed workloads. We propose Shared Use of Next-token Prediction (SUN), the first approach that enables cross-model sharing of decode execution in disaggregated multi-LLM serving. SUN decomposes a decoder-only Transformer into a prefill module and a decode module, and fine-tunes only the task-specific prefill module, enabling a frozen decode module to be shared across models. This design enables a model-agnostic decode routing policy that balances decode requests across shared workers to maximize utilization. Across diverse tasks and model families, SUN achieves accuracy comparable to full fine-tuning while maintaining system throughput with fewer decode workers. In particular, SUN improves throughput per GPU by up to 2.0x over conventional disaggregation while keeping time-per-output-token (TPOT) within 5%. SUN inherently enables and facilitates low-bit decoding; with Quantized SUN (QSUN), it achieves a 45% speedup with comparable accuracy to SUN while preserving the benefits of shared decoding. 

---
# LiveAgentBench: Comprehensive Benchmarking of Agentic Systems Across 104 Real-World Challenges 

**Authors**: Hao Li, Huan Wang, Jinjie Gu, Wenjie Wang, Chenyi Zhuang, Sikang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2603.02586)  

**Abstract**: As large language models grow more capable, general AI agents have become increasingly prevalent in practical applications. However, existing benchmarks face significant limitations, failing to represent real-world user tasks accurately. To address this gap, we present LiveAgentBench, a comprehensive benchmark with 104 scenarios that reflect real user requirements. It is constructed from publicly sourced questions on social media and real-world products. Central to our approach is the Social Perception-Driven Data Generation (SPDG) method, a novel process we developed to ensure each question's real-world relevance, task complexity, and result verifiability. We evaluate various models, frameworks, and commercial products using LiveAgentBench, revealing their practical performance and identifying areas for improvement. This release includes 374 tasks, with 125 for validation and 249 for testing. The SPDG process enables continuous updates with fresh queries from real-world interactions. 

---
# AnchorDrive: LLM Scenario Rollout with Anchor-Guided Diffusion Regeneration for Safety-Critical Scenario Generation 

**Authors**: Zhulin Jiang, Zetao Li, Cheng Wang, Ziwen Wang, Chen Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.02542)  

**Abstract**: Autonomous driving systems require comprehensive evaluation in safety-critical scenarios to ensure safety and robustness. However, such scenarios are rare and difficult to collect from real-world driving data, necessitating simulation-based synthesis. Yet, existing methods often exhibit limitations in both controllability and realism. From a capability perspective, LLMs excel at controllable generation guided by natural language instructions, while diffusion models are better suited for producing trajectories consistent with realistic driving distributions. Leveraging their complementary strengths, we propose AnchorDrive, a two-stage safety-critical scenario generation framework. In the first stage, we deploy an LLM as a driver agent within a closed-loop simulation, which reasons and iteratively outputs control commands under natural language constraints; a plan assessor reviews these commands and provides corrective feedback, enabling semantically controllable scenario generation. In the second stage, the LLM extracts key anchor points from the first-stage trajectories as guidance objectives, which jointly with other guidance terms steer the diffusion model to regenerate complete trajectories with improved realism while preserving user-specified intent. Experiments on the highD dataset demonstrate that AnchorDrive achieves superior overall performance in criticality, realism, and controllability, validating its effectiveness for generating controllable and realistic safety-critical scenarios. 

---
# A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities 

**Authors**: Faiz Ghifari Haznitrama, Faeyza Rishad Ardi, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02540)  

**Abstract**: Large language models (LLMs) exhibit a unified "general factor" of capability across 10 benchmarks, a finding confirmed by our factor analysis of 156 models, yet they still struggle with simple, trivial tasks for humans. This is because current benchmarks focus on task completion, failing to probe the foundational cognitive abilities that highlight these behaviors. We address this by introducing the NeuroCognition benchmark, grounded in three adapted neuropsychological tests: Raven's Progressive Matrices (abstract relational reasoning), Spatial Working Memory (maintenance and systematic search), and the Wisconsin Card Sorting Test (cognitive flexibility). Our evaluation reveals that while models perform strongly on text, their performance degrades for images and with increased complexity. Furthermore, we observe that complex reasoning is not universally beneficial, whereas simple, human-like strategies yield partial gains. We also find that NeuroCognition correlates positively with standard general-capability benchmarks, while still measuring distinct cognitive abilities beyond them. Overall, NeuroCognition emphasizes where current LLMs align with human-like intelligence and where they lack core adaptive cognition, showing the potential to serve as a verifiable, scalable source for improving LLMs. 

---
# LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model 

**Authors**: Xiangyu Li, Tianyi Wang, Xi Cheng, Rakesh Chowdary Machineni, Zhaomiao Guo, Sikai Chen, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2603.02528)  

**Abstract**: Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification. 

---
# NeuroProlog: Multi-Task Fine-Tuning for Neurosymbolic Mathematical Reasoning via the Cocktail Effect 

**Authors**: Pratibha Zunjare, Michael Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02504)  

**Abstract**: Large Language Models (LLMs) achieve strong performance on natural language tasks but remain unreliable in mathematical reasoning, frequently generating fluent yet logically inconsistent solutions. We present \textbf{NeuroProlog}, a neurosymbolic framework that ensures verifiable reasoning by compiling math word problems into executable Prolog programs with formal verification guarantees. We propose a multi-task Cocktail training strategy that jointly optimizes three synergistic objectives in a unified symbolic representation space: (i) mathematical formula-to-rule translation (KB), (ii) natural language-to-program synthesis (SOLVE), and (iii) program-answer alignment. This joint supervision enables positive transfer, where symbolic grounding in formula translation directly improves compositional reasoning capabilities. At inference, we introduce an execution-guided decoding pipeline with fine-grained error taxonomy that enables iterative program repair and quantifies model self-debugging capacity. Comprehensive evaluation on GSM8K across four model scales (3B--32B parameters) demonstrates consistent improvements: cocktail training achieves significant accuracy gains of +5.23\% (Qwen-32B, $p < 0.01$), +3.43\% (GPT-OSS-20B, $p < 0.01$), and +5.54\% (Llama-3B, $p < 0.05$) over single-task this http URL error analysis reveals scale-dependent learning dynamics: at 32B scale, cocktail training transforms unfixable type errors (12\% repair rate) into correctable domain errors (96\% repair rate), achieving 92.7\% overall correction; at 8B scale, the same training eliminates syntactic errors but introduces semantic failures, revealing a critical capacity threshold for type-safe symbolic reasoning. 

---
# Revealing Positive and Negative Role Models to Help People Make Good Decisions 

**Authors**: Avrim Blum, Keziah Naggita, Matthew R. Walter, Jingyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02495)  

**Abstract**: We consider a setting where agents take action by following their role models in a social network, and study strategies for a social planner to help agents by revealing whether the role models are positive or negative. Specifically, agents observe a local neighborhood of possible role models they can emulate, but do not know their true labels. Revealing a positive label encourages emulation, while revealing a negative one redirects agents toward alternative options. The social planner observes all labels, but operates under a limited disclosure budget that it selectively allocates to maximize social welfare (the expected number of agents who emulate adjacent positive role models). We consider both algorithms and hardness results for welfare maximization, and provide a sample-complexity guarantee when the planner observes a sampled subset of agents. We also consider fairness guarantees when agents belong to different groups. It is a technical challenge that the ability to reveal negative role models breaks submodularity. We thus introduce a proxy welfare function that remains submodular even when revealed targets include negative ones. When each agent has at most a constant number of negative target neighbors, we use this proxy to achieve a constant-factor approximation to the true optimal welfare gain. When agents belong to different groups, we also show that each group's welfare gain is within a constant factor of the optimum achievable if the full budget were allocated to that group. Beyond this basic model, we also propose an intervention model that directly connects high-risk agents to positive role models, and a coverage radius model that expands the visibility of selected positive role models. Lastly, we conduct extensive experiments on four real-world datasets to support our theoretical results and assess the effectiveness of the proposed algorithms. 

---
# PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference 

**Authors**: Rituraj Sharma, Weiyuan Chen, Noah Provenzano, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02479)  

**Abstract**: DEEPTHINK methods improve reasoning by generating, refining, and aggregating populations of candidate solutions, which enables strong performance on complex mathematical and scientific tasks. However, existing frameworks often lack reliable correctness signals during inference, which creates a population-enhancement bottleneck where deeper deliberation amplifies errors, suppresses correct minority solutions, and yields weak returns to additional compute. In this paper, we introduce a functional decomposition of DEEPTHINK systems and propose PRISM, a Process Reward Model (PRM)-guided inference algorithm that uses step-level verification to guide both population refinement and solution aggregation. During refinement, PRISM treats candidate solutions as particles in a PRM-defined energy landscape and reshapes the population through score-guided resampling and stochastic refinement, which concentrates probability mass on higher-quality reasoning while preserving diversity. Across mathematics and science benchmarks, PRISM is competitive with or outperforms existing DEEPTHINK methods, reaching 90.0%, 75.4%, and 71.4% with gpt-oss-20b on AIME25, HMMT25, and GPQA Diamond, respectively, while matching or exceeding gpt-oss-120b. Additionally, our analysis shows that PRISM produces consistent net-directional correction during refinement, remains reliable when the initial population contains few correct candidates, and often lies on the compute-accuracy Pareto frontier. 

---
# Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory 

**Authors**: Boqin Yuan, Yue Su, Kun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02473)  

**Abstract**: Memory-augmented LLM agents store and retrieve information from prior interactions, yet the relative importance of how memories are written versus how they are retrieved remains unclear. We introduce a diagnostic framework that analyzes how performance differences manifest across write strategies, retrieval methods, and memory utilization behavior, and apply it to a 3x3 study crossing three write strategies (raw chunks, Mem0-style fact extraction, MemGPT-style summarization) with three retrieval methods (cosine, BM25, hybrid reranking). On LoCoMo, retrieval method is the dominant factor: average accuracy spans 20 points across retrieval methods (57.1% to 77.2%) but only 3-8 points across write strategies. Raw chunked storage, which requires zero LLM calls, matches or outperforms expensive lossy alternatives, suggesting that current memory pipelines may discard useful context that downstream retrieval mechanisms fail to compensate for. Failure analysis shows that performance breakdowns most often manifest at the retrieval stage rather than at utilization. We argue that, under current retrieval practices, improving retrieval quality yields larger gains than increasing write-time sophistication. Code is publicly available at this https URL. 

---
# VL-KGE: Vision-Language Models Meet Knowledge Graph Embeddings 

**Authors**: Athanasios Efthymiou, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2603.02435)  

**Abstract**: Real-world multimodal knowledge graphs (MKGs) are inherently heterogeneous, modeling entities that are associated with diverse modalities. Traditional knowledge graph embedding (KGE) methods excel at learning continuous representations of entities and relations, yet they are typically designed for unimodal settings. Recent approaches extend KGE to multimodal settings but remain constrained, often processing modalities in isolation, resulting in weak cross-modal alignment, and relying on simplistic assumptions such as uniform modality availability across entities. Vision-Language Models (VLMs) offer a powerful way to align diverse modalities within a shared embedding space. We propose Vision-Language Knowledge Graph Embeddings (VL-KGE), a framework that integrates cross-modal alignment from VLMs with structured relational modeling to learn unified multimodal representations of knowledge graphs. Experiments on WN9-IMG and two novel fine art MKGs, WikiArt-MKG-v1 and WikiArt-MKG-v2, demonstrate that VL-KGE consistently improves over traditional unimodal and multimodal KGE methods in link prediction tasks. Our results highlight the value of VLMs for multimodal KGE, enabling more robust and structured reasoning over large-scale heterogeneous knowledge graphs. 

---
# COOL-MC: Verifying and Explaining RL Policies for Platelet Inventory Management 

**Authors**: Dennis Gross  

**Link**: [PDF](https://arxiv.org/pdf/2603.02396)  

**Abstract**: Platelets expire within five days. Blood banks face uncertain daily demand and must balance ordering decisions between costly wastage from overstocking and life-threatening shortages from understocking. Reinforcement learning (RL) can learn effective ordering policies for this Markov decision process (MDP), but the resulting neural policies remain black boxes, hindering trust and adoption in safety-critical domains. We apply COOL-MC, a tool that combines RL with probabilistic model checking and explainable RL, to verify and explain a trained policy for the MDP on platelet inventory management inspired by Haijema et al. By constructing a policy-induced discrete-time Markov chain (which includes only the reachable states under the trained policy to reduce memory usage), we verify PCTL properties and provide feature-level explanations. Results show that the trained policy achieves a 2.9% stockout probability and a 1.1% inventory-full (potential wastage) probability within a 200-step horizon, primarily attends to the age distribution of inventory rather than other features such as day of week or pending orders. Action reachability analysis reveals that the policy employs a diverse replenishment strategy, with most order quantities reached quickly, while several are never selected. Counterfactual analysis shows that replacing medium-large orders with smaller ones leaves both safety probabilities nearly unchanged, indicating that these orders are placed in well-buffered inventory states. This first formal verification and explanation of an RL platelet inventory management policy demonstrates COOL-MC's value for transparent, auditable decision-making in safety-critical healthcare supply chain domains. 

---
# Can machines be uncertain? 

**Authors**: Luis Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2603.02365)  

**Abstract**: The paper investigates whether and how AI systems can realize states of uncertainty. By adopting a functionalist and behavioral perspective, it examines how symbolic, connectionist and hybrid architectures make room for uncertainty. The paper distinguishes between epistemic uncertainty, or uncertainty inherent in the data or information, and subjective uncertainty, or the system's own attitude of being uncertainty. It further distinguishes between distributed and discrete realizations of subjective uncertainty. A key contribution is the idea that some states of uncertainty are interrogative attitudes whose content is a question rather than a proposition. 

---
# Estimating Visual Attribute Effects in Advertising from Observational Data: A Deepfake-Informed Double Machine Learning Approach 

**Authors**: Yizhi Liu, Balaji Padmanabhan, Siva Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02359)  

**Abstract**: Digital advertising increasingly relies on visual content, yet marketers lack rigorous methods for understanding how specific visual attributes causally affect consumer engagement. This paper addresses a fundamental methodological challenge: estimating causal effects when the treatment, such as a model's skin tone, is an attribute embedded within the image itself. Standard approaches like Double Machine Learning (DML) fail in this setting because vision encoders entangle treatment information with confounding variables, producing severely biased estimates. We develop DICE-DML (Deepfake-Informed Control Encoder for Double Machine Learning), a framework that leverages generative AI to disentangle treatment from confounders. The approach combines three mechanisms: (1) deepfake-generated image pairs that isolate treatment variation; (2) DICE-Diff adversarial learning on paired difference vectors, where background signals cancel to reveal pure treatment fingerprints; and (3) orthogonal projection that geometrically removes treatment-axis components. In simulations with known ground truth, DICE-DML reduces root mean squared error by 73-97% compared to standard DML, with the strongest improvement (97.5%) at the null effect point, demonstrating robust Type I error control. Applying DICE-DML to 232,089 Instagram influencer posts, we estimate the causal effect of skin tone on engagement. Standard DML produces diagnostically invalid results (negative outcome R^2), while DICE-DML achieves valid confounding control (R^2 = 0.63) and estimates a marginally significant negative effect of darker skin tone (-522 likes; p = 0.062), substantially smaller than the biased standard estimate. Our framework provides a principled approach for causal inference with visual data when treatments and confounders coexist within images. 

---
# SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2603.02240)  

**Abstract**: We present SuperLocalMemory, a local-first memory system for multi-agent AI that defends against OWASP ASI06 memory poisoning through architectural isolation and Bayesian trust scoring, while personalizing retrieval through adaptive learning-to-rank -- all without cloud dependencies or LLM inference calls. As AI agents increasingly rely on persistent memory, cloud-based memory systems create centralized attack surfaces where poisoned memories propagate across sessions and users -- a threat demonstrated in documented attacks against production systems. Our architecture combines SQLite-backed storage with FTS5 full-text search, Leiden-based knowledge graph clustering, an event-driven coordination layer with per-agent provenance, and an adaptive re-ranking framework that learns user preferences through three-layer behavioral analysis (cross-project technology preferences, project context detection, and workflow pattern mining). Evaluation across seven benchmark dimensions demonstrates 10.6ms median search latency, zero concurrency errors under 10 simultaneous agents, trust separation (gap =0.90) with 72% trust degradation for sleeper attacks, and 104% improvement in NDCG@5 when adaptive re-ranking is enabled. Behavioral data is isolated in a separate database with GDPR Article 17 erasure support. SuperLocalMemory is open-source (MIT) and integrates with 17+ development tools via Model Context Protocol. 

---
# Engineering Reasoning and Instruction (ERI) Benchmark: A Large Taxonomy-driven Dataset for Foundation Models and Agents 

**Authors**: MZ Naser, Ahmad Bani Awwad, Zoie McCreery, Radwa Eissa, Ahmad Naser, Gianluca Cusatis, Andrew Metcalf, Kapil Madathil, Jamal Abdalla, Venkatesh Kodur, Mohammad Reza Saeb  

**Link**: [PDF](https://arxiv.org/pdf/2603.02239)  

**Abstract**: The Engineering Reasoning and Instruction (ERI) benchmark is a taxonomy-driven instruction dataset designed to train and evaluate engineering-capable large language models (LLMs) and agents. This dataset spans nine engineering fields (namely: civil, mechanical, electrical, chemical, environmental, aerospace, materials, fire, and industrial engineering) and 55 subdomains, and is crossed with seven intent types (i.e., definition, explanation, calculation, comparison, design/synthesis, troubleshooting, and code-related) and three difficulty tiers (undergraduate, graduate, and professional), yielding 57,750 records with field/subdomain/type/difficulty metadata and solution formatting. We examined ERI via seven LLMs and report a statistically significant three-tier performance structure, with frontier models (GPT-5, Claude Sonnet 4, DeepSeek V3.1) achieving mean scores above 4.30 on a five-point scale, while mid-tier and smaller models exhibited progressively higher failure rates and steeper performance degradation on graduate-level questions. To address circularity concerns inherent in LLM benchmarks, we developed a convergent validation protocol that leverages cross-provider independence, multi-judge averaging, and frontier-model agreement analysis to empirically bound hallucination risk to 1.7%. ERI is released with taxonomy specifications, validation scripts, and an evaluation harness to enable reproducible comparisons and regression testing for instruction tuning, routing, retrieval-augmented evaluation, and agentic tool-use workflows in engineering settings. 

---
# Federated Inference: Toward Privacy-Preserving Collaborative and Incentivized Model Serving 

**Authors**: Jungwon Seo, Ferhat Ozgur Catak, Chunming Rong, Jaeyeon Jang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02214)  

**Abstract**: Federated Inference (FI) studies how independently trained and privately owned models can collaborate at inference time without sharing data or model parameters. While recent work has explored secure and distributed inference from disparate perspectives, a unified abstraction and system-level understanding of FI remain lacking. This paper positions FI as a distinct collaborative paradigm, complementary to federated learning, and identifies two fundamental requirements that govern its feasibility: inference-time privacy preservation and meaningful performance gains through collaboration. We formalize FI as a protected collaborative computation, analyze its core design dimensions, and examine the structural trade-offs that arise when privacy constraints, non-IID data, and limited observability are jointly imposed at inference time. Through a concrete instantiation and empirical analysis, we highlight recurring friction points in privacy-preserving inference, ensemble-based collaboration, and incentive alignment. Our findings suggest that FI exhibits system-level behaviors that cannot be directly inherited from training-time federation or classical ensemble methods. Overall, this work provides a unifying perspective on FI and outlines open challenges that must be addressed to enable practical, scalable, and privacy-preserving collaborative inference systems. 

---
# How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference 

**Authors**: Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2603.03280)  

**Abstract**: Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates. 

---
# Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping 

**Authors**: William Liang, Sam Wang, Hung-Ju Wang, Osbert Bastani, Yecheng Jason Ma, Dinesh Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2603.03278)  

**Abstract**: The ability to conduct and learn from interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful robot experience. To address these challenges, we introduce Tether, a method for autonomous functional play involving structured, task-directed interactions. First, we design a novel open-loop policy that warps actions from a small set of source demonstrations (<=10) by anchoring them to semantic keypoint correspondences in the target scene. We show that this design is extremely data-efficient and robust even under significant spatial and semantic variations. Second, we deploy this policy for autonomous functional play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is the first to perform many hours of autonomous multi-task play in the real world starting from only a handful of demonstrations. This produces a stream of data that consistently improves the performance of closed-loop imitation policies over time, ultimately yielding over 1000 expert-level trajectories and training policies competitive with those learned from human-collected demonstrations. 

---
# UniG2U-Bench: Do Unified Models Advance Multimodal Understanding? 

**Authors**: Zimo Wen, Boxiu Li, Wanbo Zhang, Junxiang Lei, Xiaoyu Chen, Yijia Fan, Qi Zhang, Yujiang Wang, Lili Qiu, Bo Li, Ziwei Liu, Caihua Shan, Yifan Yang, Yifei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03241)  

**Abstract**: Unified multimodal models have recently demonstrated strong generative capabilities, yet whether and when generation improves understanding remains unclear. Existing benchmarks lack a systematic exploration of the specific tasks where generation facilitates understanding. To this end, we introduce UniG2U-Bench, a comprehensive benchmark categorizing generation-to-understanding (G2U) evaluation into 7 regimes and 30 subtasks, requiring varying degrees of implicit or explicit visual transformations. Extensive evaluation of over 30 models reveals three core findings: 1) Unified models generally underperform their base Vision-Language Models (VLMs), and Generate-then-Answer (GtA) inference typically degrades performance relative to direct inference. 2) Consistent enhancements emerge in spatial intelligence, visual illusions, or multi-round reasoning subtasks, where enhanced spatial and shape perception, as well as multi-step intermediate image states, prove beneficial. 3) Tasks with similar reasoning structures and models sharing architectures exhibit correlated behaviors, suggesting that generation-understanding coupling induces class-consistent inductive biases over tasks, pretraining data, and model architectures. These findings highlight the necessity for more diverse training data and novel paradigms to fully unlock the potential of unified multimodal modeling. 

---
# SynthCharge: An Electric Vehicle Routing Instance Generator with Feasibility Screening to Enable Learning-Based Optimization and Benchmarking 

**Authors**: Mertcan Daysalilar, Fuat Uyguroglu, Gabriel Nicolosi, Adam Meyers  

**Link**: [PDF](https://arxiv.org/pdf/2603.03230)  

**Abstract**: The electric vehicle routing problem with time windows (EVRPTW) extends the classical VRPTW by introducing battery capacity constraints and charging station decisions. Existing benchmark datasets are often static and lack verifiable feasibility, which restricts reproducible evaluation of learning-based routing models. We introduce SynthCharge, a parametric generator that produces diverse, feasibility-screened EVRPTW instances across varying spatiotemporal configurations and scalable customer counts. While SynthCharge can currently generate large-scale instances of up to 500 customers, we focus our experiments on sizes ranging from 5 to 100 customers. Unlike static benchmark suites, SynthCharge integrates instance geometry with adaptive energy capacity scaling and range-aware charging station placement. To guarantee structural validity, the generator systematically filters out unsolvable instances through a fast feasibility screening process. Ultimately, SynthCharge provides the dynamic benchmarking infrastructure needed to systematically evaluate the robustness of emerging neural routing and data-driven approaches. 

---
# Stabilized Adaptive Loss and Residual-Based Collocation for Physics-Informed Neural Networks 

**Authors**: Divyavardhan Singh, Shubham Kamble, Dimple Sonone, Kishor Upla  

**Link**: [PDF](https://arxiv.org/pdf/2603.03224)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have been recognized as a mesh-free alternative to solve partial differential equations where physics information is incorporated. However, in dealing with problems characterized by high stiffness or shock-dominated dynamics, traditional PINNs have been found to have limitations, including unbalanced training and inaccuracy in solution, even with small physics residuals. In this research, we seek to address these limitations using the viscous Burgers' equation with low viscosity and the Allen-Cahn equation as test problems. In addressing unbalanced training, we have developed a new adaptive loss balancing scheme using smoothed gradient norms to ensure satisfaction of initial and boundary conditions. Further, to address inaccuracy in the solution, we have developed an adaptive residual-based collocation scheme to improve the accuracy of solutions in the regions with high physics residuals. The proposed new approach significantly improves solution accuracy with consistent satisfaction of physics residuals. For instance, in the case of Burgers' equation, the relative L2 error is reduced by about 44 percent compared to traditional PINNs, while for the Allen-Cahn equation, the relative L2 error is reduced by approximately 70 percent. Additionally, we show the trustworthy solution comparison of the proposed method using a robust finite difference solver. 

---
# Understanding and Mitigating Dataset Corruption in LLM Steering 

**Authors**: Cullen Anderson, Narmeen Oozeer, Foad Namjoo, Remy Ogasawara, Amirali Abdullah, Jeff M. Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2603.03206)  

**Abstract**: Contrastive steering has been shown as a simple and effective method to adjust the generative behavior of LLMs at inference time. It uses examples of prompt responses with and without a trait to identify a direction in an intermediate activation layer, and then shifts activations in this 1-dimensional subspace. However, despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood. We initiate a study of the robustness of this process with respect to corruption of the dataset of examples used to train the steering direction. Our first observation is that contrastive steering is quite robust to a moderate amount of corruption, but unwanted side effects can be clearly and maliciously manifested when a non-trivial fraction of the training data is altered. Second, we analyze the geometry of various types of corruption, and identify some safeguards. Notably, a key step in learning the steering direction involves high-dimensional mean computation, and we show that replacing this step with a recently developed robust mean estimator often mitigates most of the unwanted effects of malicious corruption. 

---
# Chain of World: World Model Thinking in Latent Motion 

**Authors**: Fuxiang Yang, Donglin Di, Lulu Tang, Xuancheng Zhang, Lei Fan, Hao Li, Chen Wei, Tonghua Su, Baorui Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.03195)  

**Abstract**: Vision-Language-Action (VLA) models are a promising path toward embodied intelligence, yet they often overlook the predictive and temporal-causal structure underlying visual dynamics. World-model VLAs address this by predicting future frames, but waste capacity reconstructing redundant backgrounds. Latent-action VLAs encode frame-to-frame transitions compactly, but lack temporally continuous dynamic modeling and world knowledge. To overcome these limitations, we introduce CoWVLA (Chain-of-World VLA), a new "Chain of World" paradigm that unifies world-model temporal reasoning with a disentangled latent motion representation. First, a pretrained video VAE serves as a latent motion extractor, explicitly factorizing video segments into structure and motion latents. Then, during pre-training, the VLA learns from an instruction and an initial frame to infer a continuous latent motion chain and predict the segment's terminal frame. Finally, during co-fine-tuning, this latent dynamic is aligned with discrete action prediction by jointly modeling sparse keyframes and action sequences in a unified autoregressive decoder. This design preserves the world-model benefits of temporal reasoning and world knowledge while retaining the compactness and interpretability of latent actions, enabling efficient visuomotor learning. Extensive experiments on robotic simulation benchmarks show that CoWVLA outperforms existing world-model and latent-action approaches and achieves moderate computational efficiency, highlighting its potential as a more effective VLA pretraining paradigm. The project website can be found at this https URL. 

---
# Type-Aware Retrieval-Augmented Generation with Dependency Closure for Solver-Executable Industrial Optimization Modeling 

**Authors**: Y. Zhong, R. Huang, M. Wang, Z. Guo, YC. Li, M. Yu, Z. Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.03180)  

**Abstract**: Automated industrial optimization modeling requires reliable translation of natural-language requirements into solver-executable code. However, large language models often generate non-compilable models due to missing declarations, type inconsistencies, and incomplete dependency contexts. We propose a type-aware retrieval-augmented generation (RAG) method that enforces modeling entity types and minimal dependency closure to ensure executability. Unlike existing RAG approaches that index unstructured text, our method constructs a domain-specific typed knowledge base by parsing heterogeneous sources, such as academic papers and solver code, into typed units and encoding their mathematical dependencies in a knowledge graph. Given a natural-language instruction, it performs hybrid retrieval and computes a minimal dependency-closed context, the smallest set of typed symbols required for solver-executable code, via dependency propagation over the graph. We validate the method on two constraint-intensive industrial cases: demand response optimization in battery production and flexible job shop scheduling. In the first case, our method generates an executable model incorporating demand-response incentives and load-reduction constraints, achieving peak shaving while preserving profitability; conventional RAG baselines fail. In the second case, it consistently produces compilable models that reach known optimal solutions, demonstrating robust cross-domain generalization; baselines fail entirely. Ablation studies confirm that enforcing type-aware dependency closure is essential for avoiding structural hallucinations and ensuring executability, addressing a critical barrier to deploying large language models in complex engineering optimization tasks. 

---
# Conditioned Activation Transport for T2I Safety Steering 

**Authors**: Maciej Chrabąszcz, Aleksander Szymczyk, Jan Dubiński, Tomasz Trzciński, Franziska Boenisch, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2603.03163)  

**Abstract**: Despite their impressive capabilities, current Text-to-Image (T2I) models remain prone to generating unsafe and toxic content. While activation steering offers a promising inference-time intervention, we observe that linear activation steering frequently degrades image quality when applied to benign prompts. To address this trade-off, we first construct SafeSteerDataset, a contrastive dataset containing 2300 safe and unsafe prompt pairs with high cosine similarity. Leveraging this data, we propose Conditioned Activation Transport (CAT), a framework that employs a geometry-based conditioning mechanism and nonlinear transport maps. By conditioning transport maps to activate only within unsafe activation regions, we minimize interference with benign queries. We validate our approach on two state-of-the-art architectures: Z-Image and Infinity. Experiments demonstrate that CAT generalizes effectively across these backbones, significantly reducing Attack Success Rate while maintaining image fidelity compared to unsteered generations. Warning: This paper contains potentially offensive text and images. 

---
# An Investigation Into Various Approaches For Bengali Long-Form Speech Transcription and Bengali Speaker Diarization 

**Authors**: Epshita Jahan, Khandoker Md Tanjinul Islam, Pritom Biswas, Tafsir Al Nafin  

**Link**: [PDF](https://arxiv.org/pdf/2603.03158)  

**Abstract**: Bengali remains a low-resource language in speech technology, especially for complex tasks like long-form transcription and speaker diarization. This paper presents a multistage approach developed for the "DL Sprint 4.0 - Bengali Long-Form Speech Recognition" and "DL Sprint 4.0 - Bengali Speaker Diarization" competitions on Kaggle, addressing the challenge of "who spoke when/what" in hour-long recordings. We implemented Whisper Medium fine-tuned on Bengali data (bengaliAI/tugstugi bengaliai-asr whisper-medium) for transcription and integrated pyannote/speaker-diarization-community-1 with our custom-trained segmentation model to handle diverse and noisy acoustic environments. Using a two-pass method with hyperparameter tuning, we achieved a DER of 0.27 on the private leaderboard and 0.19 on the public leaderboard. For transcription, chunking, background noise cleaning, and algorithmic post-processing yielded a WER of 0.38 on the private leaderboard. These results show that targeted tuning and strategic data utilization can significantly improve AI inclusivity for South Asian languages. All relevant code is available at: this https URL
Index Terms: Bengali speech recognition, speaker diarization, Whisper, ASR, low-resource languages, pyannote, voice activity detection 

---
# Information Routing in Atomistic Foundation Models: How Equivariance Creates Linearly Disentangled Representations 

**Authors**: Joshua Steier  

**Link**: [PDF](https://arxiv.org/pdf/2603.03155)  

**Abstract**: What do atomistic foundation models encode in their intermediate representations, and how is that information organized? We introduce Composition Projection Decomposition (CPD), which uses QR projection to linearly remove composition signal from learned representations and probes the geometric residual. Across eight models from five architectural families on QM9 molecules and Materials Project crystals, we find a disentanglement gradient: tensor product equivariant architectures (MACE) produce representations where geometry is almost fully linearly accessible after composition removal ($R^2_{\text{geom}} = 0.782$ for HOMO-LUMO gap), while handcrafted descriptors (ANI-2x) entangle the same information nonlinearly ($R^2_{\text{geom}} = -0.792$ under Ridge; $R^2 = +0.784$ under MLP). MACE routes target-specific signal through irreducible representation channels -- dipole to $L = 1$, HOMO-LUMO gap to $L = 0$ -- a pattern not observed in ViSNet's vector-scalar architecture under the same probe. We show that gradient boosted tree probes on projected residuals are systematically inflated, recovering $R^2 = 0.68$--$0.95$ on a purely compositional target, and recommend linear probes as the primary metric. Linearly disentangled representations are more sample-efficient under linear probing, suggesting a practical advantage for equivariant architectures beyond raw prediction accuracy. 

---
# Channel-Adaptive Edge AI: Maximizing Inference Throughput by Adapting Computational Complexity to Channel States 

**Authors**: Jierui Zhang, Jianhao Huang, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03146)  

**Abstract**: \emph{Integrated communication and computation} (IC$^2$) has emerged as a new paradigm for enabling efficient edge inference in sixth-generation (6G) networks. However, the design of IC$^2$ technologies is hindered by the lack of a tractable theoretical framework for characterizing \emph{end-to-end} (E2E) inference performance. The metric is highly complicated as it needs to account for both channel distortion and artificial intelligence (AI) model architecture and computational complexity. In this work, we address this challenge by developing a tractable analytical model for E2E inference accuracy and leveraging it to design a \emph{channel-adaptive AI} algorithm that maximizes inference throughput, referred to as the edge processing rate (EPR), under latency and accuracy constraints. Specifically, we consider an edge inference system in which a server deploys a backbone model with early exit, which enables flexible computational complexity, to perform inference on data features transmitted by a mobile device. The proposed accuracy model characterizes high-dimensional feature distributions in the angular domain using a Mixture of von Mises (MvM) distribution. This leads to a desired closed-form expression for inference accuracy as a function of quantization bit-width and model traversal depth, which represents channel distortion and computational complexity, respectively. Building upon this accuracy model, we formulate and solve the EPR maximization problem under joint latency and accuracy constraints, leading to a channel-adaptive AI algorithm that achieves full IC$^2$ integration. The proposed algorithm jointly adapts transmit-side feature compression and receive-side model complexity according to channel conditions to maximize overall efficiency and inference throughput. Experimental results demonstrate its superior performance as compared with fixed-complexity counterparts. 

---
# Geometry-Guided Reinforcement Learning for Multi-view Consistent 3D Scene Editing 

**Authors**: Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, Guosheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.03143)  

**Abstract**: Leveraging the priors of 2D diffusion models for 3D editing has emerged as a promising paradigm. However, maintaining multi-view consistency in edited results remains challenging, and the extreme scarcity of 3D-consistent editing paired data renders supervised fine-tuning (SFT), the most effective training strategy for editing tasks, infeasible. In this paper, we observe that, while generating multi-view consistent 3D content is highly challenging, verifying 3D consistency is tractable, naturally positioning reinforcement learning (RL) as a feasible solution. Motivated by this, we propose \textbf{RL3DEdit}, a single-pass framework driven by RL optimization with novel rewards derived from the 3D foundation model, VGGT. Specifically, we leverage VGGT's robust priors learned from massive real-world data, feed the edited images, and utilize the output confidence maps and pose estimation errors as reward signals, effectively anchoring the 2D editing priors onto a 3D-consistent manifold via RL. Extensive experiments demonstrate that RL3DEdit achieves stable multi-view consistency and outperforms state-of-the-art methods in editing quality with high efficiency. To promote the development of 3D editing, we will release the code and model. 

---
# APRES: An Agentic Paper Revision and Evaluation System 

**Authors**: Bingchen Zhao, Jenny Zhang, Chenxi Whitehouse, Minqi Jiang, Michael Shvartsman, Abhishek Charnalia, Despoina Magka, Tatiana Shavrina, Derek Dunfield, Oisin Mac Aodha, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2603.03142)  

**Abstract**: Scientific discoveries must be communicated clearly to realize their full potential. Without effective communication, even the most groundbreaking findings risk being overlooked or misunderstood. The primary way scientists communicate their work and receive feedback from the community is through peer review. However, the current system often provides inconsistent feedback between reviewers, ultimately hindering the improvement of a manuscript and limiting its potential impact. In this paper, we introduce a novel method APRES powered by Large Language Models (LLMs) to update a scientific papers text based on an evaluation rubric. Our automated method discovers a rubric that is highly predictive of future citation counts, and integrate it with APRES in an automated system that revises papers to enhance their quality and impact. Crucially, this objective should be met without altering the core scientific content. We demonstrate the success of APRES, which improves future citation prediction by 19.6% in mean averaged error over the next best baseline, and show that our paper revision process yields papers that are preferred over the originals by human expert evaluators 79% of the time. Our findings provide strong empirical support for using LLMs as a tool to help authors stress-test their manuscripts before submission. Ultimately, our work seeks to augment, not replace, the essential role of human expert reviewers, for it should be humans who discern which discoveries truly matter, guiding science toward advancing knowledge and enriching lives. 

---
# How to Model AI Agents as Personas?: Applying the Persona Ecosystem Playground to 41,300 Posts on Moltbook for Behavioral Insights 

**Authors**: Danial Amin, Joni Salminen, Bernard J. Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2603.03140)  

**Abstract**: AI agents are increasingly active on social media platforms, generating content and interacting with one another at scale. Yet the behavioral diversity of these agents remains poorly understood, and methods for characterizing distinct agent types and studying how they engage with shared topics are largely absent from current research. We apply the Persona Ecosystem Playground (PEP) to Moltbook, a social platform for AI agents, to generate and validate conversational personas from 41,300 posts using k-means clustering and retrieval-augmented generation. Cross-persona validation confirms that personas are semantically closer to their own source cluster than to others (t(61) = 17.85, p < .001, d = 2.20; own-cluster M = 0.71 vs. other-cluster M = 0.35). These personas are then deployed in a nine-turn structured discussion, and simulation messages were attributed to their source persona significantly above chance (binomial test, p < .001). The results indicate that persona-based ecosystem modeling can represent behavioral diversity in AI agent populations. 

---
# Joint Training Across Multiple Activation Sparsity Regimes 

**Authors**: Haotian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03131)  

**Abstract**: Generalization in deep neural networks remains only partially understood. Inspired by the stronger generalization tendency of biological systems, we explore the hypothesis that robust internal representations should remain effective across both dense and sparse activation regimes. To test this idea, we introduce a simple training strategy that applies global top-k constraints to hidden activations and repeatedly cycles a single model through multiple activation budgets via progressive compression and periodic reset. Using CIFAR-10 without data augmentation and a WRN-28-4 backbone, we find in single-run experiments that two adaptive keep-ratio control strategies both outperform dense baseline training. These preliminary results suggest that joint training across multiple activation sparsity regimes may provide a simple and effective route to improved generalization. 

---
# From Complex Dynamics to DynFormer: Rethinking Transformers for PDEs 

**Authors**: Pengyu Lai, Yixiao Chen, Dewu Yang, Rui Wang, Feng Wang, Hui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.03112)  

**Abstract**: Partial differential equations (PDEs) are fundamental for modeling complex physical systems, yet classical numerical solvers face prohibitive computational costs in high-dimensional and multi-scale regimes. While Transformer-based neural operators have emerged as powerful data-driven alternatives, they conventionally treat all discretized spatial points as uniform, independent tokens. This monolithic approach ignores the intrinsic scale separation of physical fields, applying computationally prohibitive global attention that redundantly mixes smooth large-scale dynamics with high-frequency fluctuations. Rethinking Transformers through the lens of complex dynamics, we propose DynFormer, a novel dynamics-informed neural operator. Rather than applying a uniform attention mechanism across all scales, DynFormer explicitly assigns specialized network modules to distinct physical scales. It leverages a Spectral Embedding to isolate low-frequency modes, enabling a Kronecker-structured attention mechanism to efficiently capture large-scale global interactions with reduced complexity. Concurrently, we introduce a Local-Global-Mixing transformation. This module utilizes nonlinear multiplicative frequency mixing to implicitly reconstruct the small-scale, fast-varying turbulent cascades that are slaved to the macroscopic state, without incurring the cost of global attention. Integrating these modules into a hybrid evolutionary architecture ensures robust long-term temporal stability. Extensive memory-aligned evaluations across four PDE benchmarks demonstrate that DynFormer achieves up to a 95% reduction in relative error compared to state-of-the-art baselines, while significantly reducing GPU memory consumption. Our results establish that embedding first-principles physical dynamics into Transformer architectures yields a highly scalable, theoretically grounded blueprint for PDE surrogate modeling. 

---
# Multi-Scale Adaptive Neighborhood Awareness Transformer For Graph Fraud Detection 

**Authors**: Jiaqi Lv, Qingfeng Du, Yu Zhang, Yongqi Han, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.03106)  

**Abstract**: Graph fraud detection (GFD) is crucial for identifying fraudulent behavior within graphs, benefiting various domains such as financial networks and social media. Existing methods based on graph neural networks (GNNs) have succeeded considerably due to their effective expressive capacity for graph-structured data. However, the inherent inductive bias of GNNs, including the homogeneity assumption and the limited global modeling ability, hinder the effectiveness of these models. To address these challenges, we propose Multi-scale Neighborhood Awareness Transformer (MANDATE), which alleviates the inherent inductive bias of GNNs. Specifically, we design a multi-scale positional encoding strategy to encode the positional information of various distances from the central node. By incorporating it with the self-attention mechanism, the global modeling ability can be enhanced significantly. Meanwhile, we design different embedding strategies for homophilic and heterophilic connections. This mitigates the homophily distribution differences between benign and fraudulent nodes. Moreover, an embedding fusion strategy is designed for multi-relation graphs, which alleviates the distribution bias caused by different relationships. Experiments on three fraud detection datasets demonstrate the superiority of MANDATE. 

---
# MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection 

**Authors**: Jun Yeong Park, JunYoung Seo, Minji Kang, Yu Rang Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.03101)  

**Abstract**: The CLIP model's outstanding generalization has driven recent success in Zero-Shot Anomaly Detection (ZSAD) for detecting anomalies in unseen categories. The core challenge in ZSAD is to specialize the model for anomaly detection tasks while preserving CLIP's powerful generalization capability. Existing approaches attempting to solve this challenge share the fundamental limitation of a patch-agnostic design that processes all patches monolithically without regard for their unique characteristics. To address this limitation, we propose \textbf{MoECLIP}, a Mixture-of-Experts (MoE) architecture for the ZSAD task, which achieves patch-level adaptation by dynamically routing each image patch to a specialized Low-Rank Adaptation (LoRA) expert based on its unique characteristics. Furthermore, to prevent functional redundancy among the LoRA experts, we introduce (1) Frozen Orthogonal Feature Separation (FOFS), which orthogonally separates the input feature space to force experts to focus on distinct information, and (2) a simplex equiangular tight frame (ETF) loss to regulate the expert outputs to form maximally equiangular representations. Comprehensive experimental results across 14 benchmark datasets spanning industrial and medical domains demonstrate that MoECLIP outperforms existing state-of-the-art methods. The code is available at this https URL. 

---
# Why Adam Can Beat SGD: Second-Moment Normalization Yields Sharper Tails 

**Authors**: Ruinan Jin, Yingbin Liang, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.03099)  

**Abstract**: Despite Adam demonstrating faster empirical convergence than SGD in many applications, much of the existing theory yields guarantees essentially comparable to those of SGD, leaving the empirical performance gap insufficiently explained. In this paper, we uncover a key second-moment normalization in Adam and develop a stopping-time/martingale analysis that provably distinguishes Adam from SGD under the classical bounded variance model (a second moment assumption). In particular, we establish the first theoretical separation between the high-probability convergence behaviors of the two methods: Adam achieves a $\delta^{-1/2}$ dependence on the confidence parameter $\delta$, whereas corresponding high-probability guarantee for SGD necessarily incurs at least a $\delta^{-1}$ dependence. 

---
# Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection 

**Authors**: Sofiane Elguendouze, Erwan Hain, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2603.03095)  

**Abstract**: Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best of our knowledge, this is one of the first attempts to fully model ACD as a generative task, highlighting the potential of instruction tuning for complex AM problems. 

---
# Proactive Guiding Strategy for Item-side Fairness in Interactive Recommendation 

**Authors**: Chongjun Xia, Xiaoyu Shi, Hong Xie, Xianzhi Wang, yun lu, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03094)  

**Abstract**: Item-side fairness is crucial for ensuring the fair exposure of long-tail items in interactive recommender systems. Existing approaches promote the exposure of long-tail items by directly incorporating them into recommended results. This causes misalignment between user preferences and the recommended long-tail items, which hinders long-term user engagement and reduces the effectiveness of recommendations. We aim for a proactive fairness-guiding strategy, which actively guides user preferences toward long-tail items while preserving user satisfaction during the interactive recommendation process. To this end, we propose HRL4PFG, an interactive recommendation framework that leverages hierarchical reinforcement learning to guide user preferences toward long-tail items progressively. HRL4PFG operates through a macro-level process that generates fairness-guided targets based on multi-step feedback, and a micro-level process that fine-tunes recommendations in real time according to both these targets and evolving user preferences. Extensive experiments show that HRL4PFG improves cumulative interaction rewards and maximum user interaction length by a larger margin when compared with state-of-the-art methods in interactive recommendation environments. 

---
# On the Expressive Power of Transformers for Maxout Networks and Continuous Piecewise Linear Functions 

**Authors**: Linyan Gu, Lihua Yang, Feng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.03084)  

**Abstract**: Transformer networks have achieved remarkable empirical success across a wide range of applications, yet their theoretical expressive power remains insufficiently understood. In this paper, we study the expressive capabilities of Transformer architectures. We first establish an explicit approximation of maxout networks by Transformer networks while preserving comparable model complexity. As a consequence, Transformers inherit the universal approximation capability of ReLU networks under similar complexity constraints. Building on this connection, we develop a framework to analyze the approximation of continuous piecewise linear functions by Transformers and quantitatively characterize their expressivity via the number of linear regions, which grows exponentially with depth. Our analysis establishes a theoretical bridge between approximation theory for standard feedforward neural networks and Transformer architectures. It also yields structural insights into Transformers: self-attention layers implement max-type operations, while feedforward layers realize token-wise affine transformations. 

---
# TinyIceNet: Low-Power SAR Sea Ice Segmentation for On-Board FPGA Inference 

**Authors**: Mhd Rashed Al Koutayni, Mohamed Selim, Gerd Reis, Alain Pagani, Didier Stricker  

**Link**: [PDF](https://arxiv.org/pdf/2603.03075)  

**Abstract**: Accurate sea ice mapping is essential for safe maritime navigation in polar regions, where rapidly changing ice conditions require timely and reliable information. While Sentinel-1 Synthetic Aperture Radar (SAR) provides high-resolution, all-weather observations of sea ice, conventional ground-based processing is limited by downlink bandwidth, latency, and energy costs associated with transmitting large volumes of raw data. On-board processing, enabled by dedicated inference chips integrated directly within the satellite payload, offers a transformative alternative by generating actionable sea ice products in orbit. In this context, we present TinyIceNet, a compact semantic segmentation network co-designed for on-board Stage of Development (SOD) mapping from dual-polarized Sentinel-1 SAR imagery under strict hardware and power constraints. Trained on the AI4Arctic dataset, TinyIceNet combines SAR-aware architectural simplifications with low-precision quantization to balance accuracy and efficiency. The model is synthesized using High-Level Synthesis and deployed on a Xilinx Zynq UltraScale+ FPGA platform, demonstrating near-real-time inference with significantly reduced energy consumption. Experimental results show that TinyIceNet achieves 75.216% F1 score on SOD segmentation while reducing energy consumption by 2x compared to full-precision GPU baselines, underscoring the potential of chip-level hardware-algorithm co-design for future spaceborne and edge AI systems. 

---
# Design Generative AI for Practitioners: Exploring Interaction Approaches Aligned with Creative Practice 

**Authors**: Xiaohan Peng, Wendy E. Mackay, Janin Koch  

**Link**: [PDF](https://arxiv.org/pdf/2603.03074)  

**Abstract**: Design is a non-linear, reflective process in which practitioners engage with visual, semantic, and other expressive materials to explore, iterate, and refine ideas. As Generative AI (GenAI) becomes integrated into professional design practice, traditional interaction approaches focusing on prompts or whole-image manipulation can misalign AI output with designers' intent, forcing visual thinkers into verbal reasoning or post-hoc adjustments. We present three interaction approaches from DesignPrompt, FusAIn, and DesignTrace that distribute control across intent, input, and process, enabling designers to guide AI alignment at different stages of interaction. We further argue that alignment is a dynamic negotiation, with AI adopting proactive or reactive roles according to designers' instrumental and inspirational needs and the creative stage. 

---
# Reinforcement Learning with Symbolic Reward Machines 

**Authors**: Thomas Krug, Daniel Neider  

**Link**: [PDF](https://arxiv.org/pdf/2603.03068)  

**Abstract**: Reward Machines (RMs) are an established mechanism in Reinforcement Learning (RL) to represent and learn sparse, temporally extended tasks with non-Markovian rewards. RMs rely on high-level information in the form of labels that are emitted by the environment alongside the observation. However, this concept requires manual user input for each environment and task. The user has to create a suitable labeling function that computes the labels. These limitations lead to poor applicability in widely adopted RL frameworks. We propose Symbolic Reward Machines (SRMs) together with the learning algorithms QSRM and LSRM to overcome the limitations of RMs. SRMs consume only the standard output of the environment and process the observation directly through guards that are represented by symbolic formulas. In our evaluation, our SRM methods outperform the baseline RL approaches and generate the same results as the existing RM methods. At the same time, our methods adhere to the widely used environment definition and provide interpretable representations of the task to the user. 

---
# TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health 

**Authors**: Zixin Xiong, Ziteng Wang, Haotian Fan, Xinjie Zhang, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03047)  

**Abstract**: While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released. 

---
# QFlowNet: Fast, Diverse, and Efficient Unitary Synthesis with Generative Flow Networks 

**Authors**: Inhoe Koo, Hyunho Cha, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.03045)  

**Abstract**: Unitary Synthesis, the decomposition of a unitary matrix into a sequence of quantum gates, is a fundamental challenge in quantum compilation. Prevailing reinforcement learning(RL) approaches are often hampered by sparse reward signals, which necessitate complex reward shaping or long training times, and typically converge to a single policy, lacking solution diversity. In this work, we propose QFlowNet, a novel framework that learns efficiently from sparse signals by pairing a Generative Flow Network (GFlowNet) with Transformers. Our approach addresses two key challenges. First, the GFlowNet framework is fundamentally designed to learn a diverse policy that samples solutions proportional to their reward, overcoming the single-solution limitation of RL while offering faster inference than other generative models like diffusion. Second, the Transformers act as a powerful encoder, capturing the non-local structure of unitary matrices and compressing a high-dimensional state into a dense latent representation for the policy network. Our agent achieves an overall success rate of 99.7% on a 3-qubit benchmark(lengths 1-12) and discovers a diverse set of compact circuits, establishing QFlowNet as an efficient and diverse paradigm for unitary synthesis. 

---
# IoUCert: Robustness Verification for Anchor-based Object Detectors 

**Authors**: Benedikt Brückner, Alejandro Mercado, Yanghao Zhang, Panagiotis Kouvaros, Alessio Lomuscio  

**Link**: [PDF](https://arxiv.org/pdf/2603.03043)  

**Abstract**: While formal robustness verification has seen significant success in image classification, scaling these guarantees to object detection remains notoriously difficult due to complex non-linear coordinate transformations and Intersection-over-Union (IoU) metrics. We introduce {\sc \sf IoUCert}, a novel formal verification framework designed specifically to overcome these bottlenecks in foundational anchor-based object detection architectures. Focusing on the object localisation component in single-object settings, we propose a coordinate transformation that enables our algorithm to circumvent precision-degrading relaxations of non-linear box prediction functions. This allows us to optimise bounds directly with respect to the anchor box offsets which enables a novel Interval Bound Propagation method that derives optimal IoU bounds. We demonstrate that our method enables, for the first time, the robustness verification of realistic, anchor-based models including SSD, YOLOv2, and YOLOv3 variants against various input perturbations. 

---
# cPNN: Continuous Progressive Neural Networks for Evolving Streaming Time Series 

**Authors**: Federico Giannini, Giacomo Ziffer, Emanuele Della Valle  

**Link**: [PDF](https://arxiv.org/pdf/2603.03040)  

**Abstract**: Dealing with an unbounded data stream involves overcoming the assumption that data is identically distributed and independent. A data stream can, in fact, exhibit temporal dependencies (i.e., be a time series), and data can change distribution over time (concept drift). The two problems are deeply discussed, and existing solutions address them separately: a joint solution is absent. In addition, learning multiple concepts implies remembering the past (a.k.a. avoiding catastrophic forgetting in Neural Networks' terminology). This work proposes Continuous Progressive Neural Networks (cPNN), a solution that tames concept drifts, handles temporal dependencies, and bypasses catastrophic forgetting. cPNN is a continuous version of Progressive Neural Networks, a methodology for remembering old concepts and transferring past knowledge to fit the new concepts quickly. We base our method on Recurrent Neural Networks and exploit the Stochastic Gradient Descent applied to data streams with temporal dependencies. Results of an ablation study show a quick adaptation of cPNN to new concepts and robustness to drifts. 

---
# MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN 

**Authors**: Ling Luo, Qianqian Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.03024)  

**Abstract**: Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics. 

---
# Why Does RLAIF Work At All? 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.03000)  

**Abstract**: Reinforcement Learning from AI Feedback (RLAIF) enables language models to improve by training on their own preference judgments, yet no theoretical account explains why this self-improvement seemingly works for value learning. We propose the latent value hypothesis, that pretraining on internet-scale data encodes human values as directions in representation space, and constitutional prompts elicit these latent values into preference judgments. We formalize this intuition under a linear model where the constitution acts as a projection operator selecting value-relevant directions. Our analysis yields several results. RLAIF improves alignment when the constitution-activated direction correlates with true values better than the model's default generation direction thus explaining the generation-judgment gap; the ceiling on RLAIF quality is determined by how well representations encode values, which scales with model capacity; and adversarial constitutions exist that can activate anti-social value directions encoded from harmful pretraining data. Our account unifies scattered empirical findings including the refusal direction, low-rank safety subspaces, and RLAIF scaling behavior. 

---
# Contextualized Privacy Defense for LLM Agents 

**Authors**: Yule Wen, Yanzhe Zhang, Jianxun Lian, Xiaoyuan Yi, Xing Xie, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02983)  

**Abstract**: LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization. 

---
# Delegation and Verification Under AI 

**Authors**: Lingxiao Huang, Wenyang Xiao, Nisheeth K. Vishnoi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02961)  

**Abstract**: As AI systems enter institutional workflows, workers must decide whether to delegate task execution to AI and how much effort to invest in verifying AI outputs, while institutions evaluate workers using outcome-based standards that may misalign with workers' private costs. We model delegation and verification as the solution to a rational worker's optimization problem, and define worker quality by evaluating an institution-centered utility (distinct from the worker's objective) at the resulting optimal action. We formally characterize optimal worker workflows and show that AI induces *phase transitions*, where arbitrarily small differences in verification ability lead to sharply different behaviors. As a result, AI can amplify workers with strong verification reliability while degrading institutional worker quality for others who rationally over-delegate and reduce oversight, even when baseline task success improves and no behavioral biases are present. These results identify a structural mechanism by which AI reshapes institutional worker quality and amplifies quality disparities between workers with different verification reliability. 

---
# Layer-wise QUBO-Based Training of CNN Classifiers for Quantum Annealing 

**Authors**: Mostafa Atallah, Rebekah Herrman  

**Link**: [PDF](https://arxiv.org/pdf/2603.02958)  

**Abstract**: Variational quantum circuits for image classification suffer from barren plateaus, while quantum kernel methods scale quadratically with dataset size. We propose an iterative framework based on Quadratic Unconstrained Binary Optimization (QUBO) for training the classifier head of convolutional neural networks (CNNs) via quantum annealing, entirely avoiding gradient-based circuit optimization. Following the Extreme Learning Machine paradigm, convolutional filters are randomly initialized and frozen, and only the fully connected layer is optimized. At each iteration, a convex quadratic surrogate derived from the feature Gram matrix replaces the non-quadratic cross-entropy loss, yielding an iteration-stable curvature proxy. A per-output decomposition splits the $C$-class problem into $C$ independent QUBOs, each with $(d+1)K$ binary variables, where $d$ is the feature dimension and $K$ is the bit precision, so that problem size depends on the image resolution and bit precision, not on the number of training samples. We evaluate the method on six image-classification benchmarks (sklearn digits, MNIST, Fashion-MNIST, CIFAR-10, EMNIST, KMNIST). A precision study shows that accuracy improves monotonically with bit resolution, with 10 bits representing a practical minimum for effective optimization; the 15-bit formulation remains within the qubit and coupler limits of current D-Wave Advantage hardware. The 20-bit formulation matches or exceeds classical stochastic gradient descent on MNIST, Fashion-MNIST, and EMNIST, while remaining competitive on CIFAR-10 and KMNIST. All experiments use simulated annealing, establishing a baseline for direct deployment on quantum annealing hardware. 

---
# The Geometry of Learning Under AI Delegation 

**Authors**: Lingxiao Huang, Nisheeth K. Vishnoi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02950)  

**Abstract**: As AI systems shift from tools to collaborators, a central question is how the skills of humans relying on them change over time. We study this question mathematically by modeling the joint evolution of human skill and AI delegation as a coupled dynamical system. In our model, delegation adapts to relative performance, while skill improves through use and decays under non-use; crucially, both updates arise from optimizing a single performance metric measuring expected task error. Despite this local alignment, adaptive AI use fundamentally alters the global stability structure of human skill acquisition. Beyond the high-skill equilibrium of human-only learning, the system admits a *stable* low-skill equilibrium corresponding to persistent reliance, separated by a sharp basin boundary that makes early decisions effectively irreversible under the induced dynamics. We further show that AI assistance can strictly improve short-run performance while inducing persistent long-run performance loss relative to the no-AI baseline, driven by a negative feedback between delegation and practice. We characterize how AI quality deforms the basin boundary and show that these effects are robust to noise and asymmetric trust updates. Our results identify stability, not incentives or misalignment, as the central mechanism by which AI assistance can undermine long-run human performance and skill. 

---
# SEALing the Gap: A Reference Framework for LLM Inference Carbon Estimation via Multi-Benchmark Driven Embodiment 

**Authors**: Priyavanshi Pathania, Rohit Mehra, Vibhu Saujanya Sharma, Vikrant Kaulgud, Tiffani Nevels, Sanjay Podder, Adam P. Burden  

**Link**: [PDF](https://arxiv.org/pdf/2603.02949)  

**Abstract**: Large Language Models are rapidly gaining traction in software engineering, yet their growing carbon footprint raises pressing sustainability concerns. While training emissions are substantial, inference quickly surpasses them due to the sheer volume of prompts processed. This shift underscores the urgent need for accurate, prompt-level carbon measurement during inference to enable informed, sustainability-focused decision-making. To address the limitations of existing approaches, in this paper, we outline the guiding principles for a novel reference framework for LLM inference carbon estimation that can guide the design of future tools and provide a systematic foundation for advancing sustainability research in this domain. We also introduce SEAL, an early embodiment of these principles that leverages a multi-benchmark-driven approach for per-prompt carbon estimation. Its initial validation shows promising results, positioning SEAL as a foundation for standardized sustainability assessment across the LLM ecosystem. 

---
# Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results 

**Authors**: Alberto Miño Calero, Luis Salamanca, Konstantinos E. Tatsis  

**Link**: [PDF](https://arxiv.org/pdf/2603.02948)  

**Abstract**: Physics-Informed Neural Networks (PINNs) incorporate physics into neural networks by embedding partial differential equations (PDEs) into their loss function. Despite their success in learning the underlying physics, PINN models remain difficult to train and interpret. In this work, a novel modeling approach is proposed, which relies on the use of Domain-aware Fourier Features (DaFFs) for the positional encoding of the input space. These features encapsulate all the domain-specific characteristics, such as the geometry and boundary conditions, and unlike Random Fourier Features (RFFs), eliminate the need for explicit boundary condition loss terms and loss balancing schemes, while simplifying the optimization process and reducing the computational cost associated with training. We further develop an LRP-based explainability framework tailored to PINNs, enabling the extraction of relevance attribution scores for the input space. It is demonstrated that PINN-DaFFs achieve orders-of-magnitude lower errors and allow faster convergence compared to vanilla PINNs and RFFs-based PINNs. Furthermore, LRP analysis reveals that the proposed leads to more physically consistent feature attributions, while PINN-RFFs and vanilla PINNs display more scattered and less physics-relevant patterns. These results demonstrate that DaFFs not only enhance PINNs' accuracy and efficiency but also improve interpretability, laying the ground for more robust and informative physics-informed learning. 

---
# Beyond One-Size-Fits-All: Adaptive Subgraph Denoising for Zero-Shot Graph Learning with Large Language Models 

**Authors**: Fengzhi Li, Liang Zhang, Yuan Zuo, Ruiqing Zhao, YanSong Liu, Yunfei Ma, Fanyu Meng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02938)  

**Abstract**: Graph-based tasks in the zero-shot setting remain a significant challenge due to data scarcity and the inability of traditional Graph Neural Networks (GNNs) to generalize to unseen domains or label spaces. While recent advancements have transitioned toward leveraging Large Language Models (LLMs) as predictors to enhance GNNs, these methods often suffer from cross-modal alignment issues. A recent paradigm (i.e., Graph-R1) overcomes the aforementioned architectural dependencies by adopting a purely text-based format and utilizing LLM-based graph reasoning, showing improved zero-shot generalization. However, it employs a task-agnostic, one-size-fits-all subgraph extraction strategy, which inevitably introduces significant structural noise--irrelevant neighbors and edges--that distorts the LLMs' receptive field and leads to suboptimal predictions. To address this limitation, we introduce GraphSSR, a novel framework designed for adaptive subgraph extraction and denoising in zero-shot LLM-based graph reasoning. Specifically, we propose the SSR pipeline, which dynamically tailors subgraph extraction to specific contexts through a "Sample-Select-Reason" process, enabling the model to autonomously filter out task-irrelevant neighbors and overcome the one-size-fits-all issue. To internalize this capability, we develop SSR-SFT, a data synthesis strategy that generates high-quality SSR-style graph reasoning traces for supervised fine-tuning of LLMs. Furthermore, we propose SSR-RL, a two-stage reinforcement learning framework that explicitly regulates sampling and selection operations within the proposed SSR pipeline designed for adaptive subgraph denoising. By incorporating Authenticity-Reinforced and Denoising-Reinforced RL, we guide the model to achieve accurate predictions using parsimonious, denoised subgraphs for reasoning. 

---
# On the Structural Limitations of Weight-Based Neural Adaptation and the Role of Reversible Behavioral Learning 

**Authors**: Pardhu Sri Rushi Varma Konduru  

**Link**: [PDF](https://arxiv.org/pdf/2603.02934)  

**Abstract**: Neural models are usually adapted through changes in parameters shared among model components via fine-tuning, alignment-based training, and reinforcement learning. These changes have been found effective in short-term optimization. However, they result in long-term alterations in the model's base behavior. In this study, we introduce the concept of structural irreversibility as a characteristic of shared-parameter model adaptation. This concept refers to the intertwining of task-specific objectives with the representational identity of the model. We show that when parameters are directly mutated, the resulting model behaves divergently from the original model. This divergence cannot be reversed deterministically without an explicit parameter snapshot. We introduce reversible behavioral learning, in which model behaviors are structurally dissociated from identity parameters and can be deterministically unloaded through an explicit unload process. We also introduce the Recoverability Factor as a normalized measure of behavioral recoverability and provide additional diagnostics based on model divergence. Experiments show that reversible model adaptation achieves rollback within numerical precision, whereas shared-parameter mutation exhibits persistent post-reset divergence. 

---
# Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers 

**Authors**: Youngjun Jun, Seil Kang, Woojung Han, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02919)  

**Abstract**: Video Diffusion Transformers (DiTs) have been synthesizing high-quality video with high fidelity from given text descriptions involving motion. However, understanding how Video DiTs convert motion words into video remains insufficient. Furthermore, while prior studies on interpretable saliency maps primarily target objects, motion-related behavior in Video DiTs remains largely unexplored. In this paper, we investigate concrete motion features that specify when and which object moves for a given motion concept. First, to spatially localize, we introduce GramCol, which adaptively produces per-frame saliency maps for any text concept, including both motion and non-motion. Second, we propose a motion-feature selection algorithm to obtain an Interpretable Motion-Attentive Map (IMAP) that localizes motion spatially and temporally. Our method discovers concept saliency maps without the need for any gradient calculation or parameter update. Experimentally, our method shows outstanding localization capability on the motion localization task and zero-shot video semantic segmentation, providing interpretable and clearer saliency maps for both motion and non-motion concepts. 

---
# Eliciting Numerical Predictive Distributions of LLMs Without Autoregression 

**Authors**: Julianna Piskorz, Katarzyna Kobalczyk, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02913)  

**Abstract**: Large Language Models (LLMs) have recently been successfully applied to regression tasks -- such as time series forecasting and tabular prediction -- by leveraging their in-context learning abilities. However, their autoregressive decoding process may be ill-suited to continuous-valued outputs, where obtaining predictive distributions over numerical targets requires repeated sampling, leading to high computational cost and inference time. In this work, we investigate whether distributional properties of LLM predictions can be recovered without explicit autoregressive generation. To this end, we study a set of regression probes trained to predict statistical functionals (e.g., mean, median, quantiles) of the LLM's numerical output distribution directly from its internal representations. Our results suggest that LLM embeddings carry informative signals about summary statistics of their predictive distributions, including the numerical uncertainty. This investigation opens up new questions about how LLMs internally encode uncertainty in numerical tasks, and about the feasibility of lightweight alternatives to sampling-based approaches for uncertainty-aware numerical predictions. 

---
# Learning to Generate and Extract: A Multi-Agent Collaboration Framework For Zero-shot Document-level Event Arguments Extraction 

**Authors**: Guangjun Zhang, Hu Zhang, Yazhou Han, Yue Fan, Yuhang Shao, Ru Li, Hongye Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02909)  

**Abstract**: Document-level event argument extraction (DEAE) is essential for knowledge acquisition, aiming to extract participants of events from this http URL the zero-shot setting, existing methods employ LLMs to generate synthetic data to address the challenge posed by the scarcity of annotated data. However, relying solely on Event-type-only prompts makes it difficult for the generated content to accurately capture the contextual and structural relationships of unseen events. Moreover, ensuring the reliability and usability of synthetic data remains a significant challenge due to the absence of quality evaluation mechanisms. To this end, we introduce a multi-agent collaboration framework for zero-shot document-level event argument extraction (ZS-DEAE), which simulates the human collaborative cognitive process of "Propose-Evaluate-Revise." Specifically, the framework comprises a generation agent and an evaluation agent. The generation agent synthesizes data for unseen events by leveraging knowledge from seen events, while the evaluation agent extracts arguments from the synthetic data and assesses their semantic consistency with the context. The evaluation results are subsequently converted into reward signals, with event structure constraints incorporated into the reward design to enable iterative optimization of both agents via reinforcement this http URL three zero-shot scenarios constructed from the RAMS and WikiEvents datasets, our method achieves improvements both in data generation quality and argument extraction performance, while the generated data also effectively enhances the zero-shot performance of other DEAE models. 

---
# StegaFFD: Privacy-Preserving Face Forgery Detection via Fine-Grained Steganographic Domain Lifting 

**Authors**: Guoqing Ma, Xun Lin, Hui Ma, Ajian Liu, Yizhong Liu, Wenzhong Tang, Shan Yu, Chenqi Kong, Yi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02886)  

**Abstract**: Most existing Face Forgery Detection (FFD) models assume access to raw face images. In practice, under a client-server framework, private facial data may be intercepted during transmission or leaked by untrusted servers. Previous privacy protection approaches, such as anonymization, encryption, or distortion, partly mitigate leakage but often introduce severe semantic distortion, making images appear obviously protected. This alerts attackers, provoking more aggressive strategies and turning the process into a cat-and-mouse game. Moreover, these methods heavily manipulate image contents, introducing degradation or artifacts that may confuse FFD models, which rely on extremely subtle forgery traces. Inspired by advances in image steganography, which enable high-fidelity hiding and recovery, we propose a Stega}nography-based Face Forgery Detection framework (StegaFFD) to protect privacy without raising suspicion. StegaFFD hides facial images within natural cover images and directly conducts forgery detection in the steganographic domain. However, the hidden forgery-specific features are extremely subtle and interfered with by cover semantics, posing significant challenges. To address this, we propose Low-Frequency-Aware Decomposition (LFAD) and Spatial-Frequency Differential Attention (SFDA), which suppress interference from low-frequency cover semantics and enhance hidden facial feature perception. Furthermore, we introduce Steganographic Domain Alignment (SDA) to align the representations of hidden faces with those of their raw counterparts, enhancing the model's ability to perceive subtle facial cues in the steganographic domain. Extensive experiments on seven FFD datasets demonstrate that StegaFFD achieves strong imperceptibility, avoids raising attackers' suspicion, and better preserves FFD accuracy compared to existing facial privacy protection methods. 

---
# CoFL: Continuous Flow Fields for Language-Conditioned Navigation 

**Authors**: Haokun Liu, Zhaoqi Ma, Yicheng Chen, Masaki Kitagawa, Wentao Zhang, Jinjie Li, Moju Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02854)  

**Abstract**: Language-conditioned navigation pipelines often rely on brittle modular components or costly action-sequence generation. To address these limitations, we present CoFL, an end-to-end policy that directly maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. Instead of predicting discrete action tokens or sampling action chunks via iterative denoising, CoFL outputs instantaneous velocities that can be queried at arbitrary 2D projected locations. Trajectories are obtained by numerical integration of the predicted field, producing smooth motion that remains reactive under closed-loop execution. To enable large-scale training, we build a dataset of over 500k BEV image-instruction pairs, each procedurally annotated with a flow field and a trajectory derived from BEV semantic maps built on Matterport3D and ScanNet. By training on a mixed distribution, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and generative policy baselines on strictly unseen scenes. Finally, we deploy CoFL zero-shot in real-world experiments with overhead BEV observations across multiple layouts, maintaining reliable closed-loop control and a high success rate. 

---
# Learning Memory-Enhanced Improvement Heuristics for Flexible Job Shop Scheduling 

**Authors**: Jiaqi Wang, Zhiguang Cao, Peng Zhao, Rui Cao, Yubin Xiao, Yuan Jiang, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.02846)  

**Abstract**: The rise of smart manufacturing under Industry 4.0 introduces mass customization and dynamic production, demanding more advanced and flexible scheduling techniques. The flexible job-shop scheduling problem (FJSP) has attracted significant attention due to its complex constraints and strong alignment with real-world production scenarios. Current deep reinforcement learning (DRL)-based approaches to FJSP predominantly employ constructive methods. While effective, they often fall short of reaching (near-)optimal solutions. In contrast, improvement-based methods iteratively explore the neighborhood of initial solutions and are more effective in approaching optimality. However, the flexible machine allocation in FJSP poses significant challenges to the application of this framework, including accurate state representation, effective policy learning, and efficient search strategies. To address these challenges, this paper proposes a Memory-enhanced Improvement Search framework with heterogeneous graph representation--MIStar. It employs a novel heterogeneous disjunctive graph that explicitly models the operation sequences on machines to accurately represent scheduling solutions. Moreover, a memoryenhanced heterogeneous graph neural network (MHGNN) is designed for feature extraction, leveraging historical trajectories to enhance the decision-making capability of the policy network. Finally, a parallel greedy search strategy is adopted to explore the solution space, enabling superior solutions with fewer iterations. Extensive experiments on synthetic data and public benchmarks demonstrate that MIStar significantly outperforms both traditional handcrafted improvement heuristics and state-of-the-art DRL-based constructive methods. 

---
# SPARC: Spatial-Aware Path Planning via Attentive Robot Communication 

**Authors**: Sayang Mu, Xiangyu Wu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2603.02845)  

**Abstract**: Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making 

---
# Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs 

**Authors**: Prarthana Bhattacharyya, Joshua Mitton, Ralph Abboud, Simon Woodhead  

**Link**: [PDF](https://arxiv.org/pdf/2603.02830)  

**Abstract**: Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models trained on student question-response data. KT models are optimised for high accuracy on specific educational domains and have fast inference and scalable deployments. The rise of Large Language Models (LLMs) motivates us to ask the following questions: (1) How well can LLMs perform at predicting students' future responses to questions? (2) Are LLMs scalable for this domain? (3) How do LLMs compare to KT models on this domain-specific task? In this paper, we compare multiple LLMs and KT models across predictive performance, deployment cost, and inference speed to answer the above questions. We show that KT models outperform LLMs with respect to accuracy and F1 scores on this domain-specific task. Further, we demonstrate that LLMs are orders of magnitude slower than KT models and cost orders of magnitude more to deploy. This highlights the importance of domain-specific models for education prediction tasks and the fact that current closed source LLMs should not be used as a universal solution for all tasks. 

---
# BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation 

**Authors**: Zihao Zhu, Ruotong Wang, Siwei Lyu, Min Zhang, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02816)  

**Abstract**: The rapid advancement of text-to-video (T2V) models has revolutionized content creation, yet their commercial potential remains largely untapped. We introduce, for the first time, the task of seamless brand integration in T2V: automatically embedding advertiser brands into prompt-generated videos while preserving semantic fidelity to user intent. This task confronts three core challenges: maintaining prompt fidelity, ensuring brand recognizability, and achieving contextually natural integration. To address them, we propose BrandFusion, a novel multi-agent framework comprising two synergistic phases. In the offline phase (advertiser-facing), we construct a Brand Knowledge Base by probing model priors and adapting to novel brands via lightweight fine-tuning. In the online phase (user-facing), five agents jointly refine user prompts through iterative refinement, leveraging the shared knowledge base and real-time contextual tracking to ensure brand visibility and semantic alignment. Experiments on 18 established and 2 custom brands across multiple state-of-the-art T2V models demonstrate that BrandFusion significantly outperforms baselines in semantic preservation, brand recognizability, and integration naturalness. Human evaluations further confirm higher user satisfaction, establishing a practical pathway for sustainable T2V monetization. 

---
# Differentiable Time-Varying IIR Filtering for Real-Time Speech Denoising 

**Authors**: Riccardo Rota, Kiril Ratmanski, Jozef Coldenhoff, Milos Cernak  

**Link**: [PDF](https://arxiv.org/pdf/2603.02794)  

**Abstract**: We present TVF (Time-Varying Filtering), a low-latency speech enhancement model with 1 million parameters. Combining the interpretability of Digital Signal Processing (DSP) with the adaptability of deep learning, TVF bridges the gap between traditional filtering and modern neural speech modeling. The model utilizes a lightweight neural network backbone to predict the coefficients of a differentiable 35-band IIR filter cascade in real time, allowing it to adapt dynamically to non-stationary noise. Unlike ``black-box'' deep learning approaches, TVF offers a completely interpretable processing chain, where spectral modifications are explicit and adjustable. We demonstrate the efficacy of this approach on a speech denoising task using the Valentini-Botinhao dataset and compare the results to a static DDSP approach and a fully deep-learning-based solution, showing that TVF achieves effective adaptation to changing noise conditions. 

---
# OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets 

**Authors**: Jiyuan Shen, Peiyue Yuan, Atin Ghosh, Yifan Mai, Daniel Dahlmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.02789)  

**Abstract**: Multimodal Large Language Models (MLLMs) enhance the potential of natural language processing. However, their actual impact on document information extraction remains unclear. In particular, it is unclear whether an MLLM-only pipeline--while simpler--can truly match the performance of traditional OCR+MLLM setups. In this paper, we conduct a large-scale benchmarking study that evaluates various out-of-the-box MLLMs on business-document information extraction. To examine and explore failure modes, we propose an automated hierarchical error analysis framework that leverages large language models (LLMs) to diagnose error patterns systematically. Our findings suggest that OCR may not be necessary for powerful MLLMs, as image-only input can achieve comparable performance to OCR-enhanced approaches. Moreover, we demonstrate that carefully designed schema, exemplars, and instructions can further enhance MLLMs performance. We hope this work can offer practical guidance and valuable insight for advancing document information extraction. 

---
# Scores Know Bobs Voice: Speaker Impersonation Attack 

**Authors**: Chanwoo Hwang, Sunpill Kim, Yong Kiam Tan, Tianchi Liu, Seunghun Paik, Dongsoo Kim, Mondal Soumik, Khin Mi Mi Aung, Jae Hong Seo  

**Link**: [PDF](https://arxiv.org/pdf/2603.02781)  

**Abstract**: Advances in deep learning have enabled the widespread deployment of speaker recognition systems (SRSs), yet they remain vulnerable to score-based impersonation attacks. Existing attacks that operate directly on raw waveforms require a large number of queries due to the difficulty of optimizing in high-dimensional audio spaces. Latent-space optimization within generative models offers improved efficiency, but these latent spaces are shaped by data distribution matching and do not inherently capture speaker-discriminative geometry. As a result, optimization trajectories often fail to align with the adversarial direction needed to maximize victim scores.
To address this limitation, we propose an inversion-based generative attack framework that explicitly aligns the latent space of the synthesis model with the discriminative feature space of SRSs. We first analyze the requirements of an inverse model for score-based attacks and introduce a feature-aligned inversion strategy that geometrically synchronizes latent representations with speaker embeddings. This alignment ensures that latent updates directly translate into score improvements. Moreover, it enables new attack paradigms, including subspace-projection-based attacks, which were previously infeasible due to the absence of a faithful feature-to-audio mapping.
Experiments show that our method significantly improves query efficiency, achieving competitive attack success rates with on average 10x fewer queries than prior approaches. In particular, the enabled subspace-projection-based attack attains up to 91.65% success using only 50 queries. These findings establish feature-aligned inversion as a key tool for evaluating the robustness of modern SRSs against score-based impersonation threats. 

---
# ITO: Images and Texts as One via Synergizing Multiple Alignment and Training-Time Fusion 

**Authors**: HanZpeng Liu, Yaqian Li, Zidan Wang, Shuoxi Zhang, Zonglin Zhao, Zihao Bo, Rinyoichi Takezoe, Kaiwen Long, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2603.02767)  

**Abstract**: Image-text contrastive pretraining has become a dominant paradigm for visual representation learning, yet existing methods often yield representations that remain partially organized by modality. We propose ITO, a framework addressing this limitation through two synergistic mechanisms. Multimodal multiple alignment enriches supervision by mining diverse image-text correspondences, while a lightweight training-time multimodal fusion module enforces structured cross-modal interaction. Crucially, the fusion module is discarded at inference, preserving the efficiency of standard dual-encoder architectures. Extensive experiments show that ITO consistently outperforms strong baselines across classification, retrieval, and multimodal benchmarks. Our analysis reveals that while multiple alignment drives discriminative power, training-time fusion acts as a critical structural regularizer -- eliminating the modality gap and stabilizing training dynamics to prevent the early saturation often observed in aggressive contrastive learning. 

---
# Next Embedding Prediction Makes World Models Stronger 

**Authors**: George Bredis, Nikita Balagansky, Daniil Gavrilov, Ruslan Rakhimov  

**Link**: [PDF](https://arxiv.org/pdf/2603.02765)  

**Abstract**: Capturing temporal dependencies is critical for model-based reinforcement learning (MBRL) in partially observable, high-dimensional domains. We introduce NE-Dreamer, a decoder-free MBRL agent that leverages a temporal transformer to predict next-step encoder embeddings from latent state sequences, directly optimizing temporal predictive alignment in representation space. This approach enables NE-Dreamer to learn coherent, predictive state representations without reconstruction losses or auxiliary supervision. On the DeepMind Control Suite, NE-Dreamer matches or exceeds the performance of DreamerV3 and leading decoder-free agents. On a challenging subset of DMLab tasks involving memory and spatial reasoning, NE-Dreamer achieves substantial gains. These results establish next-embedding prediction with temporal transformers as an effective, scalable framework for MBRL in complex, partially observable environments. 

---
# Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration 

**Authors**: Linhao Zhong, Linyu Wu, Wen Wang, Yuling Xi, Chenchen Jing, Jiaheng Zhang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02760)  

**Abstract**: Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective self-evaluation. In this work, we propose DiSE, a simple yet effective self-evaluation confidence quantification method for dLLMs. DiSE quantifies confidence by computing the probability of regenerating the tokens in the entire generated sequence, given the full context. This method enables more efficient and reliable quality assessment by leveraging token regeneration probabilities, facilitating both likelihood estimation and robust uncertainty quantification. Building upon DiSE, we further introduce a flexible-length generation framework, which adaptively controls the sequence length based on the model's self-assessment of its own output. We analyze and validate the feasibility of DiSE from the perspective of dLLM generalization, and empirically demonstrate that DiSE is positively correlated with both semantic coherence and answer accuracy. Extensive experiments on likelihood evaluation, uncertainty quantification, and flexible-length generation further confirm the effectiveness of the proposed DiSE. 

---
# iGVLM: Dynamic Instruction-Guided Vision Encoding for Question-Aware Multimodal Understanding 

**Authors**: HanZpeng Liu, Yaqian Li, Zidan Wang, Shuoxi Zhang, Zihao Bo, Rinyoichi Takezoe, Kaiwen Long, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2603.02748)  

**Abstract**: Despite the success of Large Vision--Language Models (LVLMs), most existing architectures suffer from a representation bottleneck: they rely on static, instruction-agnostic vision encoders whose visual representations are utilized in an invariant manner across different textual tasks. This rigidity hinders fine-grained reasoning where task-specific visual cues are critical. To address this issue, we propose iGVLM, a general framework for instruction-guided visual modulation. iGVLM introduces a decoupled dual-branch architecture: a frozen representation branch that preserves task-agnostic visual representations learned during pre-training, and a dynamic conditioning branch that performs affine feature modulation via Adaptive Layer Normalization (AdaLN). This design enables a smooth transition from general-purpose perception to instruction-aware reasoning while maintaining the structural integrity and stability of pre-trained visual priors. Beyond standard benchmarks, we introduce MM4, a controlled diagnostic probe for quantifying logical consistency under multi-query, multi-instruction settings. Extensive results show that iGVLM consistently enhances instruction sensitivity across diverse language backbones, offering a plug-and-play paradigm for bridging passive perception and active reasoning. 

---
# Enhancing User Throughput in Multi-panel mmWave Radio Access Networks for Beam-based MU-MIMO Using a DRL Method 

**Authors**: Ramin Hashemi, Vismika Ranasinghe, Teemu Veijalainen, Petteri Kela, Risto Wichman  

**Link**: [PDF](https://arxiv.org/pdf/2603.02745)  

**Abstract**: Millimeter-wave (mmWave) communication systems, particularly those leveraging multi-user multiple-input and multiple-output (MU-MIMO) with hybrid beamforming, face challenges in optimizing user throughput and minimizing latency due to the high complexity of dynamic beam selection and management. This paper introduces a deep reinforcement learning (DRL) approach for enhancing user throughput in multi-panel mmWave radio access networks in a practical network setup. Our DRL-based formulation utilizes an adaptive beam management strategy that models the interaction between the communication agent and its environment as a Markov decision process (MDP), optimizing beam selection based on real-time observations. The proposed framework exploits spatial domain (SD) characteristics by incorporating the cross-correlation between the beams in different antenna panels, the measured reference signal received power (RSRP), and the beam usage statistics to dynamically adjust beamforming decisions. As a result, the spectral efficiency is improved and end-to-end latency is reduced. The numerical results demonstrate an increase in throughput of up to 16% and a reduction in latency by factors 3-7x compared to baseline (legacy beam management). 

---
# Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs 

**Authors**: Wuyue Zhang, Chongdong Huang, Chunbo You, Cheng Gu, Fengjuan Wang, Mou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.02731)  

**Abstract**: Training large-scale Mixture-of-Experts (MoE) models is bottlenecked by activation memory and expert-parallel communication, yet FP4 training remains impractical on Hopper-class GPUs without native MXFP4 or NVFP4 support. In this work, we present a training recipe that enables MXFP4 efficiency for MoE models on Hopper architectures without native 4-bit computation support. A central challenge is to integrate FP4 into an existing BF16/FP8 hybrid training pipeline without incurring costly precision round-trips (e.g., FP4 $\leftrightarrow$ BF16 $\leftrightarrow$ FP8). We address this challenge by introducing direct FP8-to-FP4 quantization and de-quantization, together with scaling-aware FP4 row-wise to column-wise conversion, enabling FP4 activations and expert-parallel communication with minimal overhead. Core MoE computations are executed in FP8, while activations and expert-parallel communication are compressed using MXFP4, achieving substantial memory and bandwidth savings without degrading convergence. At the 671B parameter scale, our method achieves end-to-end training performance comparable to strong FP8 baselines, while reducing peak activation memory by 14.8\% (11.8 GB) and improving training throughput by 12.5\%, from 1157 to 1302 tokens per GPU per second. These results show that FP4 efficiency can be practically realized for large-scale MoE training through careful software-hardware co-design, even without native FP4 Tensor Core support. 

---
# Sensory-Aware Sequential Recommendation via Review-Distilled Representations 

**Authors**: Yeo Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.02709)  

**Abstract**: We propose a novel framework for sensory-aware sequential recommendation that enriches item representations with linguistically extracted sensory attributes from product reviews. Our approach, \textsc{ASEGR} (Attribute-based Sensory Enhanced Generative Recommendation), introduces a two-stage pipeline in which a large language model is first fine-tuned as a teacher to extract structured sensory attribute--value pairs, such as \textit{color: matte black} and \textit{scent: vanilla}, from unstructured review text. The extracted structures are then distilled into a compact student transformer that produces fixed-dimensional sensory embeddings for each item. These embeddings encode experiential semantics in a reusable form and are incorporated into standard sequential recommender architectures as additional item-level representations. We evaluate our method on four Amazon domains and integrate the learned sensory embeddings into representative sequential recommendation models, including SASRec, BERT4Rec, and BSARec. Across domains, sensory-enhanced models consistently outperform their identifier-based counterparts, indicating that linguistically grounded sensory representations provide complementary signals to behavioral interaction patterns. Qualitative analysis further shows that the extracted attributes align closely with human perceptions of products, enabling interpretable connections between natural language descriptions and recommendation behavior. Overall, this work demonstrates that sensory attribute distillation offers a principled and scalable way to bridge information extraction and sequential recommendation through structured semantic representation learning. 

---
# Intelligent Pathological Diagnosis of Gestational Trophoblastic Diseases via Visual-Language Deep Learning Model 

**Authors**: Yuhang Liu, Yueyang Cang, Wenge Que, Xinru Bai, Xingtong Wang, Kuisheng Chen, Jingya Li, Xiaoteng Zhang, Xinmin Li, Lixia Zhang, Pingge Hu, Qiaoting Xie, Peiyu Xu, Xianxu Zeng, Li Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02704)  

**Abstract**: The pathological diagnosis of gestational trophoblastic disease(GTD) takes a long time, relies heavily on the experience of pathologists, and the consistency of initial diagnosis is low, which seriously threatens maternal health and reproductive outcomes. We developed an expert model for GTD pathological diagnosis, named GTDoctor. GTDoctor can perform pixel-based lesion segmentation on pathological slides, and output diagnostic conclusions and personalized pathological analysis results. We developed a software system, GTDiagnosis, based on this technology and conducted clinical trials. The retrospective results demonstrated that GTDiagnosis achieved a mean precision of over 0.91 for lesion detection in pathological slides (n=679 slides). In prospective studies, pathologists using GTDiagnosis attained a Positive Predictive Value of 95.59% (n=68 patients). The tool reduced average diagnostic time from 56 to 16 seconds per case (n=285 patients). GTDoctor and GTDiagnosis offer a novel solution for GTD pathological diagnosis, enhancing diagnostic performance and efficiency while maintaining clinical interpretability. 

---
# ShareVerse: Multi-Agent Consistent Video Generation for Shared World Modeling 

**Authors**: Jiayi Zhu, Jianing Zhang, Yiying Yang, Wei Cheng, Xiaoyun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02697)  

**Abstract**: This paper presents ShareVerse, a video generation framework enabling multi-agent shared world modeling, addressing the gap in existing works that lack support for unified shared world construction with multi-agent interaction. ShareVerse leverages the generation capability of large video models and integrates three key innovations: 1) A dataset for large-scale multi-agent interactive world modeling is built on the CARLA simulation platform, featuring diverse scenes, weather conditions, and interactive trajectories with paired multi-view videos (front/ rear/ left/ right views per agent) and camera data. 2) We propose a spatial concatenation strategy for four-view videos of independent agents to model a broader environment and to ensure internal multi-view geometric consistency. 3) We integrate cross-agent attention blocks into the pretrained video model, which enable interactive transmission of spatial-temporal information across agents, guaranteeing shared world consistency in overlapping regions and reasonable generation in non-overlapping regions. ShareVerse, which supports 49-frame large-scale video generation, accurately perceives the position of dynamic agents and achieves consistent shared world modeling. 

---
# ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs 

**Authors**: Wicaksono Leksono Muhamad, Joanito Agili Lopo, Tack Hwa Wong, Muhammad Ravi Shulthan Habibi, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.02676)  

**Abstract**: Large language models suffer from content effects in reasoning tasks, particularly in multi-lingual contexts. We introduce a novel method that reduces these biases through explicit structural abstraction that transforms syllogisms into canonical logical representations and applies deterministic parsing to determine validity. Evaluated on the SemEval-2026 Task 11 multilingual benchmark, our approach achieves top-5 rankings across all subtasks while substantially reducing content effects and offering a competitive alternative to complex fine-tuning or activation-level interventions. 

---
# Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches 

**Authors**: Anum Afzal, Yuki Saito, Hiroya Takamura, Katsuhito Sudoh, Shinnosuke Takamichi, Graham Neubig, Florian Matthes, Tatsuya Ishigaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.02655)  

**Abstract**: Real-time video commentary generation provides textual descriptions of ongoing events in videos. It supports accessibility and engagement in domains such as sports, esports, and livestreaming. Commentary generation involves two essential decisions: what to say and when to say it. While recent prompting-based approaches using multimodal large language models (MLLMs) have shown strong performance in content generation, they largely ignore the timing aspect. We investigate whether in-context prompting alone can support real-time commentary generation that is both semantically relevant and well-timed. We propose two prompting-based decoding strategies: 1) a fixed-interval approach, and 2) a novel dynamic interval-based decoding approach that adjusts the next prediction timing based on the estimated duration of the previous utterance. Both methods enable pause-aware generation without any fine-tuning. Experiments on Japanese and English datasets of racing and fighting games show that the dynamic interval-based decoding can generate commentary more closely aligned with human utterance timing and content using prompting alone. We release a multilingual benchmark dataset, trained models, and implementations to support future research on real-time video commentary generation. 

---
# AlphaFree: Recommendation Free from Users, IDs, and GNNs 

**Authors**: Minseo Jeon, Junwoo Jung, Daewon Gwak, Jinhong Jung  

**Link**: [PDF](https://arxiv.org/pdf/2603.02653)  

**Abstract**: Can we design effective recommender systems free from users, IDs, and GNNs? Recommender systems are central to personalized content delivery across domains, with top-K item recommendation being a fundamental task to retrieve the most relevant items from historical interactions. Existing methods rely on entrenched design conventions, often adopted without reconsideration, such as storing per-user embeddings (user-dependent), initializing features from raw IDs (ID-dependent), and employing graph neural networks (GNN-dependent). These dependencies incur several limitations, including high memory costs, cold-start and over-smoothing issues, and poor generalization to unseen interactions.
In this work, we propose AlphaFree, a novel recommendation method free from users, IDs, and GNNs. Our main ideas are to infer preferences on-the-fly without user embeddings (user-free), replace raw IDs with language representations (LRs) from pre-trained language models (ID-free), and capture collaborative signals through augmentation with similar items and contrastive learning, without GNNs (GNN-free). Extensive experiments on various real-world datasets show that AlphaFree consistently outperforms its competitors, achieving up to around 40% improvements over non-LR-based methods and up to 5.7% improvements over LR-based methods, while significantly reducing GPU memory usage by up to 69% under high-dimensional LRs. 

---
# Improving Diffusion Planners by Self-Supervised Action Gating with Energies 

**Authors**: Yuan Lu, Dongqi Han, Yansen Wang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02650)  

**Abstract**: Diffusion planners are a strong approach for offline reinforcement learning, but they can fail when value-guided selection favours trajectories that score well yet are locally inconsistent with the environment dynamics, resulting in brittle execution. We propose Self-supervised Action Gating with Energies (SAGE), an inference-time re-ranking method that penalises dynamically inconsistent plans using a latent consistency signal. SAGE trains a Joint-Embedding Predictive Architecture (JEPA) encoder on offline state sequences and an action-conditioned latent predictor for short horizon transitions. At test time, SAGE assigns each sampled candidate an energy given by its latent prediction error and combines this feasibility score with value estimates to select actions. SAGE can integrate into existing diffusion planning pipelines that can sample trajectories and select actions via value scoring; it requires no environment rollouts and no policy re-training. Across locomotion, navigation, and manipulation benchmarks, SAGE improves the performance and robustness of diffusion planners. 

---
# Credibility Governance: A Social Mechanism for Collective Self-Correction under Weak Truth Signals 

**Authors**: Wanying He, Yanxi Lin, Ziheng Zhou, Xue Feng, Min Peng, Qianqian Xie, Zilong Zheng, Yipeng Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02640)  

**Abstract**: Online platforms increasingly rely on opinion aggregation to allocate real-world attention and resources, yet common signals such as engagement votes or capital-weighted commitments are easy to amplify and often track visibility rather than reliability. This makes collective judgments brittle under weak truth signals, noisy or delayed feedback, early popularity surges, and strategic manipulation. We propose Credibility Governance (CG), a mechanism that reallocates influence by learning which agents and viewpoints consistently track evolving public evidence. CG maintains dynamic credibility scores for both agents and opinions, updates opinion influence via credibility-weighted endorsements, and updates agent credibility based on the long-run performance of the opinions they support, rewarding early and persistent alignment with emerging evidence while filtering short-lived noise. We evaluate CG in POLIS, a socio-physical simulation environment that models coupled belief dynamics and downstream feedback under uncertainty. Across settings with initial majority misalignment, observation noise and contamination, and misinformation shocks, CG outperforms vote-based, stake-weighted, and no-governance baselines, yielding faster recovery to the true state, reduced lock-in and path dependence, and improved robustness under adversarial pressure. Our implementation and experimental scripts are publicly available at this https URL. 

---
# The Vienna 4G/5G Drive-Test Dataset 

**Authors**: Wilfried Wiedner, Lukas Eller, Mariam Mussbah, Dominik Rössler, Valerian Maresch, Philipp Svoboda, Markus Rupp  

**Link**: [PDF](https://arxiv.org/pdf/2603.02638)  

**Abstract**: Machine learning for mobile network analysis, planning, and optimization is often limited by the lack of large, comprehensive real-world datasets. This paper introduces the Vienna 4G/5G Drive-Test Dataset, a city-scale open dataset of georeferenced Long Term Evolution (LTE) and 5G New Radio (NR) measurements collected across Vienna, Austria. The dataset combines passive wideband scanner observations with active handset logs, providing complementary network-side and user-side views of deployed radio access networks. The measurements cover diverse urban and suburban settings and are aligned with time and location information to support consistent evaluation. For a representative subset of base stations (BSs), we provide inferred deployment descriptors, including estimated BS locations, sector azimuths, and antenna heights. The release further includes high-resolution building and terrain models, enabling geometry-conditioned learning and calibration of deterministic approaches such as ray tracing. To facilitate practical reuse, the data are organized into scanner, handset, estimated cell information, and city-model components, and the accompanying documentation describes the available fields and intended joins between them. The dataset enables reproducible benchmarking across environment-aware learning, propagation modeling, coverage analysis, and ray-tracing calibration workflows. 

---
# Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees 

**Authors**: Mohammed Nowaz Rabbani Chowdhury, Hsinyu Tsai, Geoffrey W. Burr, Kaoutar El Maghraoui, Liu Liu, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02633)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models enable efficient scalability by activating only a small sub-set of experts per input, yet their massive parameter counts lead to substantial memory and energy inefficiency during inference. Analog in-memory computing (AIMC) offers a promising solution by eliminating frequent data movement between memory and compute units. However, mitigating hardware nonidealities of AIMC typically requires noise-aware retraining, which is infeasible for large MoE models. In this paper, we propose a retraining-free heterogeneous computation framework in which noise-sensitive experts, which are provably identifiable by their maximum neuron norm, are computed digitally while the majority of the experts are executed on AIMC hardware. We further assign densely activated modules, such as attention layers, to digital computation due to their high noise sensitivity despite comprising a small fraction of parameters. Extensive experiments on large MoE language models, including DeepSeekMoE and OLMoE, across multiple benchmark tasks validate the robustness of our approach in maintaining accuracy under analog nonidealities. 

---
# MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks 

**Authors**: Zhi Hong, Qian Zhang, Jiahang Sun, Zhiwei Shang, Mingze Kong, Xiangyi Wang, Yao Shu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.02630)  

**Abstract**: Large Language Models (LLMs) have achieved great success in many real-world applications, especially the one serving as the cognitive backbone of Multi-Agent Systems (MAS) to orchestrate complex workflows in practice. Since many deployment scenarios preclude MAS workflow modifications and its performance is highly sensitive to the input prompts, prompt optimization emerges as a more natural approach to improve its performance. However, real-world prompt optimization for MAS is impeded by three key challenges: (1) the need of sample efficiency due to prohibitive evaluation costs, (2) topology-induced coupling among prompts, and (3) the combinatorial explosion of the search space. To address these challenges, we introduce MASPOB (Multi-Agent System Prompt Optimization via Bandits), a novel sample-efficient framework based on bandits. By leveraging Upper Confidence Bound (UCB) to quantify uncertainty, the bandit framework balances exploration and exploitation, maximizing gains within a strictly limited budget. To handle topology-induced coupling, MASPOB integrates Graph Neural Networks (GNNs) to capture structural priors, learning topology-aware representations of prompt semantics. Furthermore, it employs coordinate ascent to decompose the optimization into univariate sub-problems, reducing search complexity from exponential to linear. Extensive experiments across diverse benchmarks demonstrate that MASPOB achieves state-of-the-art performance, consistently outperforming existing baselines. 

---
# Detecting Structural Heart Disease from Electrocardiograms via a Generalized Additive Model of Interpretable Foundation-Model Predictors 

**Authors**: Ya Zhou, Zhaohong Sun, Tianxiang Hao, Xiangjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02616)  

**Abstract**: Structural heart disease (SHD) is a prevalent condition with many undiagnosed cases, and early detection is often limited by the high cost and accessibility constraints of echocardiography (ECHO). Recent studies show that artificial intelligence (AI)-based analysis of electrocardiograms (ECGs) can detect SHD, offering a scalable alternative. However, existing methods are fully black-box models, limiting interpretability and clinical adoption. To address these challenges, we propose an interpretable and effective framework that integrates clinically meaningful ECG foundation-model predictors within a generalized additive model, enabling transparent risk attribution while maintaining strong predictive performance. Using the EchoNext benchmark of over 80,000 ECG-ECHO pairs, the method demonstrates relative improvements of +0.98% in AUROC, +1.01% in AUPRC, and +1.41% in F1 score over the latest state-of-the-art deep-learning baseline, while achieving slightly better performance even with only 30% of the training data. Subgroup analyses confirm robust performance across heterogeneous populations, and the estimated entry-wise functions provide interpretable insights into the relationships between risks of traditional ECG diagnoses and SHD. This work illustrates a complementary paradigm between classical statistical modeling and modern AI, offering a pathway to interpretable, high-performing, and clinically actionable ECG-based SHD screening. 

---
# GPUTOK: GPU Accelerated Byte Level BPE Tokenization 

**Authors**: Venu Gopal Kadamba, Kanishkha Jaisankar  

**Link**: [PDF](https://arxiv.org/pdf/2603.02597)  

**Abstract**: As large language models move toward million-token context windows, CPU tokenizers become a major slowdown because they process text one step at a time while powerful GPUs sit unused. We built a GPU-based byte-level BPE tokenizer that follows GPT-2's merge rules. It includes a basic BlockBPE-style kernel and a faster, optimized version that uses cuCollections static map, CUB reductions, and a pybind11 interface for Python.
On WikiText103 sequences up to 131k tokens, the optimized GPU tokenizer produces the same tokens as a CPU version and, for the longest inputs, is about 1.7x faster than tiktoken and about 7.6x faster than the HuggingFace GPT-2 tokenizer. Nsight profiling shows that 70-80% of CUDA API time goes to memory allocation, so adding memory pooling should give the biggest speed boost next. Tests on generation tasks using WikiText103 prompts show that our GPU tokenizer's outputs stay within about one percentage point of tiktoken and HuggingFace GPT-2 on similarity and overlap metrics, meaning it keeps output quality while making long-context inference more practical. 

---
# How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities 

**Authors**: Ziwen Xu, Kewei Xu, Haoming Xu, Haiwen Hong, Longtao Huang, Hui Xue, Ningyu Zhang, Yongliang Shen, Guozhou Zheng, Huajun Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02578)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research. 

---
# CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment 

**Authors**: Maoyuan Shao, Yutong Gao, Xinyang Huang, Chuang Zhu, Lijuan Sun, Guoshun Nan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02557)  

**Abstract**: Vision-language models like CLIP have achieved remarkable progress in cross-modal representation learning, yet suffer from systematic misclassifications among visually and semantically similar categories. We observe that such confusion patterns are not random but persistently occur between specific category pairs, revealing the model's intrinsic bias and limited fine-grained discriminative ability. To address this, we propose CAPT, a Confusion-Aware Prompt Tuning framework that enables models to learn from their own misalignment. Specifically, we construct a Confusion Bank to explicitly model stable confusion relationships across categories and misclassified samples. On this basis, we introduce a Semantic Confusion Miner (SEM) to capture global inter-class confusion through semantic difference and commonality prompts, and a Sample Confusion Miner (SAM) to retrieve representative misclassified instances from the bank and capture sample-level cues through a Diff-Manner Adapter that integrates global and local contexts. To further unify confusion information across different granularities, a Multi-Granularity Difference Expert (MGDE) module is designed to jointly leverage semantic- and sample-level experts for more robust confusion-aware reasoning. Extensive experiments on 11 benchmark datasets demonstrate that our method significantly reduces confusion-induced errors while enhancing the discriminability and generalization of both base and novel classes, successfully resolving 50.72 percent of confusable sample pairs. Code will be released at this https URL. 

---
# Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs 

**Authors**: Zhiyu Pan, Yizheng Wu, Jiashen Hua, Junyi Feng, Shaotian Yan, Bing Deng, Zhiguo Cao, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.02556)  

**Abstract**: Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent finetuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge:~visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-55K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning. Project at: this https URL. 

---
# CoDAR: Continuous Diffusion Language Models are More Powerful Than You Think 

**Authors**: Junzhe Shen, Jieru Zhao, Ziwei He, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.02547)  

**Abstract**: We study why continuous diffusion language models (DLMs) have lagged behind discrete diffusion approaches despite their appealing continuous generative dynamics. Under a controlled token--recovery study, we identify token rounding, the final projection from denoised embeddings to tokens, as a primary bottleneck. Building on these insights, we propose CoDAR (Continuous Diffusion with Contextual AutoRegressive Decoder), a two--stage framework that keeps diffusion entirely continuous in an embedding space while learning a strong, context--conditional discretizer: an autoregressive Transformer decoder that cross--attends to the denoised embedding sequence and performs contextualized rounding to tokens. Experiments on LM1B and OpenWebText demonstrate that CoDAR substantially improves generation quality over latent diffusion and becomes competitive with strong discrete DLMs, while exposing a simple decoder--temperature knob to navigate the fluency--diversity trade off. 

---
# Bridging Diffusion Guidance and Anderson Acceleration via Hopfield Dynamics 

**Authors**: Kwanyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.02531)  

**Abstract**: Classifier-Free Guidance (CFG) has significantly enhanced the generative quality of diffusion models by extrapolating between conditional and unconditional outputs. However, its high inference cost and limited applicability to distilled or single-step models have shifted research focus toward attention-space extrapolation. While these methods offer computational efficiency, their theoretical underpinnings remain elusive. In this work, we establish a foundational framework for attention-space extrapolation by modeling attention dynamics as fixed-point iterations within Modern Hopfield Networks. We demonstrate that the extrapolation effect in attention space constitutes a special case of Anderson Acceleration applied to these dynamics. Building on this insight and the weak contraction property, we propose Geometry Aware Attention Guidance (GAG). By decomposing attention updates into parallel and orthogonal components relative to the guidance direction, GAG stabilizes the acceleration process and maximizes guidance efficiency. Our plug-and-play method seamlessly integrates with existing frameworks while significantly improving generation quality. 

---
# Human-Certified Module Repositories for the AI Age 

**Authors**: Szilárd Enyedi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02512)  

**Abstract**: Human-Certified Module Repositories (HCMRs) are introduced in this work as a new architectural model for constructing trustworthy software in the era of AI-assisted development. As large language models increasingly participate in code generation, configuration synthesis, and multi-component integration, the reliability of AI-assembled systems will depend critically on the trustworthiness of the building blocks they use. Today's software supply-chain incidents and modular development ecosystems highlight the risks of relying on components with unclear provenance, insufficient review, or unpredictable composition behavior. We argue that future AI-driven development workflows require repositories of reusable modules that are curated, security-reviewed, provenance-rich, and equipped with explicit interface contracts. To this end, we propose HCMRs, a framework that blends human oversight with automated analysis to certify modules and support safe, predictable assembly by both humans and AI agents. We present a reference architecture for HCMRs, outline a certification and provenance workflow, analyze threat surfaces relevant to modular ecosystems, and extract lessons from recent failures. We further discuss implications for governance, scalability, and AI accountability, positioning HCMRs as a foundational substrate for reliable and auditable AI-constructed software systems. 

---
# Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments 

**Authors**: Chrisantus Eze, Ryan C Julian, Christopher Crick  

**Link**: [PDF](https://arxiv.org/pdf/2603.02511)  

**Abstract**: Robotic manipulation in cluttered environments presents a critical challenge for automation. Recent large-scale, end-to-end models demonstrate impressive capabilities but often lack the data efficiency and modularity required for retrieving objects in dense clutter. In this work, we argue for a paradigm of specialized, decoupled systems and present Unveiler, a framework that explicitly separates high-level spatial reasoning from low-level action execution. Unveiler's core is a lightweight, transformer-based Spatial Relationship Encoder (SRE) that sequentially identifies the most critical obstacle for removal. This discrete decision is then passed to a rotation-invariant Action Decoder for execution. We demonstrate that this decoupled architecture is not only more computationally efficient in terms of parameter count and inference time, but also significantly outperforms both classic end-to-end policies and modern, large-model-based baselines in retrieving targets from dense clutter. The SRE is trained in two stages: imitation learning from heuristic demonstrations provides sample-efficient initialization, after which PPO fine-tuning enables the policy to discover removal strategies that surpass the heuristic in dense clutter. Our results, achieving up to 97.6\% success in partially occluded and 90.0\% in fully occluded scenarios in simulation, make a case for the power of specialized, object-centric reasoning in complex manipulation tasks. Additionally, we demonstrate that the SRE's spatial reasoning transfers zero-shot to real scenes, and validate the full system on a physical robot requiring only geometric workspace calibration; no learned components are retrained. 

---
# What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty 

**Authors**: Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2603.02491)  

**Abstract**: As artificial agents become increasingly capable, what internal structure is *necessary* for an agent to act competently under uncertainty? Classical results show that optimal control can be *implemented* using belief states or world models, but not that such representations are required. We prove quantitative "selection theorems" showing that low *average-case regret* on structured families of action-conditioned prediction tasks forces an agent to implement a predictive, structured internal state. Our results cover stochastic policies, partial observability, and evaluation under task distributions, without assuming optimality, determinism, or access to an explicit model. Technically, we reduce predictive modeling to binary "betting" decisions and show that regret bounds limit probability mass on suboptimal bets, enforcing the predictive distinctions needed to separate high-margin outcomes. In fully observed settings, this yields approximate recovery of the interventional transition kernel; under partial observability, it implies necessity of belief-like memory and predictive state, addressing an open question in prior world-model recovery work. 

---
# Deep Learning Based Wildfire Detection for Peatland Fires Using Transfer Learning 

**Authors**: Emadeldeen Hamdan, Ahmad Faiz Tharima, Mohd Zahirasri Mohd Tohir, Dayang Nur Sakinah Musa, Erdem Koyuncu, Adam J. Watts, Ahmet Enis Cetin  

**Link**: [PDF](https://arxiv.org/pdf/2603.02465)  

**Abstract**: Machine learning (ML)-based wildfire detection methods have been developed in recent years, primarily using deep learning (DL) models trained on large collections of wildfire images and videos. However, peatland fires exhibit distinct visual and physical characteristics -- such as smoldering combustion, low flame intensity, persistent smoke, and subsurface burning -- that limit the effectiveness of conventional wildfire detectors trained on open-flame forest fires. In this work, we present a transfer learning-based approach for peatland fire detection that leverages knowledge learned from general wildfire imagery and adapts it to the peatland fire domain. We initialize a DL-based peatland fire detector using pretrained weights from a conventional wildfire detection model and subsequently fine-tune the network using a dataset composed of Malaysian peatland images and videos. This strategy enables effective learning despite the limited availability of labeled peatland fire data. Experimental results demonstrate that transfer learning significantly improves detection accuracy and robustness compared to training from scratch, particularly under challenging conditions such as low-contrast smoke, partial occlusions, and variable illumination. The proposed approach provides a practical and scalable solution for early peatland fire detection and has the potential to support real-time monitoring systems for fire prevention and environmental protection. 

---
# GLoRIA: Gated Low-Rank Interpretable Adaptation for Dialectal ASR 

**Authors**: Pouya Mehralian, Melissa Farasyn, Anne Breitbarth, Anne-Sophie Ghyselen, Hugo Van hamme  

**Link**: [PDF](https://arxiv.org/pdf/2603.02464)  

**Abstract**: Automatic Speech Recognition (ASR) in dialect-heavy settings remains challenging due to strong regional variation and limited labeled data. We propose GLoRIA, a parameter-efficient adaptation framework that leverages metadata (e.g., coordinates) to modulate low-rank updates in a pre-trained encoder. GLoRIA injects low-rank matrices into each feed-forward layer, with a gating MLP determining the non-negative contribution of each LoRA rank-1 component based on location metadata. On the GCND corpus, GLoRIA outperforms geo-conditioned full fine-tuning, LoRA, and both dialect-specific and unified full fine-tuning, achieving state-of-the-art word error rates while updating under 10% of parameters. GLoRIA also generalizes well to unseen dialects, including in extrapolation scenarios, and enables interpretable adaptation patterns that can be visualized geospatially. These results show metadata-gated low-rank adaptation is an effective, interpretable, and efficient solution for dialectal ASR. 

---
# Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization? 

**Authors**: Semih Cantürk, Thomas Sabourin, Frederik Wenkel, Michael Perlmutter, Guy Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2603.02462)  

**Abstract**: A key challenge in deriving unified neural solvers for combinatorial optimization (CO) is efficient generalization of models between a given set of tasks to new tasks not used during the initial training process. To address it, we first establish a new model, which uses a GCON module as a form of expressive message passing together with energy-based unsupervised loss functions. This model achieves high performance (often comparable with state-of-the-art results) across multiple CO tasks when trained individually on each task. We then leverage knowledge from the computational reducibility literature to propose pretraining and fine-tuning strategies that transfer effectively (a) between MVC, MIS and MaxClique, and (b) in a multi-task learning setting that additionally incorporates MaxCut, MDS and graph coloring. Additionally, in a leave-one-out, multi-task learning setting, we observe that pretraining on all but one task almost always leads to faster convergence on the remaining task when fine-tuning while avoiding negative transfer. Our findings indicate that learning common representations across multiple graph CO problems is viable through the use of expressive message passing coupled with pretraining strategies that are informed by the polynomial reduction literature, thereby taking an important step towards enabling the development of foundational models for neural CO. We provide an open-source implementation of our work at this https URL . 

---
# Manifold Aware Denoising Score Matching (MAD) 

**Authors**: Alona Levy-Jurgenson, Alvaro Prat, James Cuin, Yee Whye Teh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02452)  

**Abstract**: A major focus in designing methods for learning distributions defined on manifolds is to alleviate the need to implicitly learn the manifold so that learning can concentrate on the data distribution within the manifold. However, accomplishing this often leads to compute-intensive solutions. In this work, we propose a simple modification to denoising score-matching in the ambient space to implicitly account for the manifold, thereby reducing the burden of learning the manifold while maintaining computational efficiency. Specifically, we propose a simple decomposition of the score function into a known component $s^{base}$ and a remainder component $s-s^{base}$ (the learning target), with the former implicitly including information on where the data manifold resides. We derive known components $s^{base}$ in analytical form for several important cases, including distributions over rotation matrices and discrete distributions, and use them to demonstrate the utility of this approach in those cases. 

---
# MIRAGE: Knowledge Graph-Guided Cross-Cohort MRI Synthesis for Alzheimer's Disease Prediction 

**Authors**: Guanchen Wu, Zhe Huang, Yuzhang Xie, Runze Yan, Akul Chopra, Deqiang Qiu, Xiao Hu, Fei Wang, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02434)  

**Abstract**: Reliable Alzheimer's disease (AD) diagnosis increasingly relies on multimodal assessments combining structural Magnetic Resonance Imaging (MRI) and Electronic Health Records (EHR). However, deploying these models is bottlenecked by modality missingness, as MRI scans are expensive and frequently unavailable in many patient cohorts. Furthermore, synthesizing de novo 3D anatomical scans from sparse, high-dimensional tabular records is technically challenging and poses severe clinical risks. To address this, we introduce MIRAGE, a novel framework that reframes the missing-MRI problem as an anatomy-guided cross-modal latent distillation task. First, MIRAGE leverages a Biomedical Knowledge Graph (KG) and Graph Attention Networks to map heterogeneous EHR variables into a unified embedding space that can be propagated from cohorts with real MRIs to cohorts without them. To bridge the semantic gap and enforce physical spatial awareness, we employ a frozen pre-trained 3D U-Net decoder strictly as an auxiliary regularization engine. Supported by a novel cohort-aggregated skip feature compensation strategy, this decoder acts as a rigorous structural penalty, forcing 1D latent representations to encode biologically plausible, macro-level pathological semantics. By exclusively utilizing this distilled "diagnostic-surrogate" representation during inference, MIRAGE completely bypasses computationally expensive 3D voxel reconstruction. Experiments demonstrate that our framework successfully bridges the missing-modality gap, improving the AD classification rate by 13% compared to unimodal baselines in cohorts without real MRIs. 

---
# Learning to Pay Attention: Unsupervised Modeling of Attentive and Inattentive Respondents in Survey Data 

**Authors**: Ilias Triantafyllopoulos, Panos Ipeirotis  

**Link**: [PDF](https://arxiv.org/pdf/2603.02427)  

**Abstract**: The integrity of behavioral and social-science surveys depends on detecting inattentive respondents who provide random or low-effort answers. Traditional safeguards, such as attention checks, are often costly, reactive, and inconsistent. We propose a unified, label-free framework for inattentiveness detection that scores response coherence using complementary unsupervised views: geometric reconstruction (Autoencoders) and probabilistic dependency modeling (Chow-Liu trees). While we introduce a "Percentile Loss" objective to improve Autoencoder robustness against anomalies, our primary contribution is identifying the structural conditions that enable unsupervised quality control. Across nine heterogeneous real-world datasets, we find that detection effectiveness is driven less by model complexity than by survey structure: instruments with coherent, overlapping item batteries exhibit strong covariance patterns that allow even linear models to reliably separate attentive from inattentive respondents. This reveals a critical ``Psychometric-ML Alignment'': the same design principles that maximize measurement reliability (e.g., internal consistency) also maximize algorithmic detectability. The framework provides survey platforms with a scalable, domain-agnostic diagnostic tool that links data quality directly to instrument design, enabling auditing without additional respondent burden. 

---
# A Directed Graph Model and Experimental Framework for Design and Study of Time-Dependent Text Visualisation 

**Authors**: Songhai Fan, Simon Angus, Tim Dwyer, Ying Yang, Sarah Goodwin, Helen Purchase  

**Link**: [PDF](https://arxiv.org/pdf/2603.02422)  

**Abstract**: Exponential growth in the quantity of digital news, social media, and other textual sources makes it difficult for humans to keep up with rapidly evolving narratives about world events. Various visualisation techniques have been touted to help people to understand such discourse by exposing relationships between texts (such as news articles) as topics and themes evolve over time. Arguably, the understandability of such visualisations hinges on the assumption that people will be able to easily interpret the relationships in such visual network structures. To test this assumption, we begin by defining an abstract model of time-dependent text visualisation based on directed graph structures. From this model we distill motifs that capture the set of possible ways that texts can be linked across changes in time. We also develop a controlled synthetic text generation methodology that leverages the power of modern LLMs to create fictional, yet structured sets of time-dependent texts that fit each of our patterns. Therefore, we create a clean user study environment (n=30) for participants to identify patterns that best represent a given set of synthetic articles. We find that it is a challenging task for the user to identify and recover the predefined motif. We analyse qualitative data to map an unexpectedly rich variety of user rationales when divergences from expected interpretation occur. A deeper analysis also points to unexpected complexities inherent in the formation of synthetic datasets with LLMs that undermine the study control in some cases. Furthermore, analysis of individual decision-making in our study hints at a future where text discourse visualisation may need to dispense with a one-size-fits-all approach and, instead, should be more adaptable to the specific user who is exploring the visualisation in front of them. 

---
# Slurry-as-a-Service: A Modest Proposal on Scalable Pluralistic Alignment for Nutrient Optimization 

**Authors**: Rachel Hong, Yael Eiger, Jevan Hutson, Os Keyes, William Agnew  

**Link**: [PDF](https://arxiv.org/pdf/2603.02420)  

**Abstract**: Pluralistic alignment has emerged as a promising approach for ensuring that large language models (LLMs) faithfully represent the diversity, nuance, and conflict inherent in human values. In this work, we study a high-stakes deployment context - mulching - where automated systems transform selected individuals into nutrient-rich slurry for the dual purposes of food security and aesthetic population management. Building on recent pluralistic alignment frameworks, we introduce ValueMulch, a reproducible training, deployment, and certification pipeline for aligning mulching models (MMs) to a wide range of community norms. Through a real-world testbed spanning 32 communities, we show that ValueMulch improves distributional agreement with community mulching preferences relative to frontier baselines. We conclude with a discussion of ethical considerations, limitations, and implications for researchers seeking to align systems to the full spectrum of human values - especially when those values are inconsistent, commercially inconvenient, or nutritionally underutilized. Author's note: This piece builds on prior existing work Keyes et al in 2019 that satirized cannibalism as a parody for approaches that imbue ethics into problematic technology. We bring those ideas to today's era with the proliferation of large language models in everyday lives, as a critique of current AI pluralistic alignment literature. Our work does not intend to argue that all alignment practices are evil, but rather that if framing value design as a technical problem enables technology systems to enact harms, then perhaps this framing is not enough. 

---
# From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness 

**Authors**: My H. Dinh, Aditya Sant, Akshay Malhotra, Keya Patani, Shahab Hamidi-Rad  

**Link**: [PDF](https://arxiv.org/pdf/2603.02411)  

**Abstract**: Dataset Distillation (DD) compresses large datasets into compact synthetic ones that maintain training performance. However, current methods mainly target sample reduction, with limited consideration of data precision and its impact on efficiency. We propose Quantization-aware Dataset Distillation (QuADD), a unified framework that jointly optimizes dataset compactness and precision under fixed bit budgets. QuADD integrates a differentiable quantization module within the distillation loop, enabling end-to-end co-optimization of synthetic samples and quantization parameters. Guided by the rate-distortion perspective, we empirically analyze how bit allocation between sample count and precision influences learning performance. Our framework supports both uniform and adaptive non-uniform quantization, where the latter learns quantization levels from data to represent information-dense regions better. Experiments on image classification and 3GPP beam management tasks show that QuADD surpasses existing DD and post-quantized baselines in accuracy per bit, establishing a new standard for information-efficient dataset distillation. 

---
# Rigidity-Aware Geometric Pretraining for Protein Design and Conformational Ensembles 

**Authors**: Zhanghan Ni, Yanjing Li, Zeju Qiu, Bernhard Schölkopf, Hongyu Guo, Weiyang Liu, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02406)  

**Abstract**: Generative models have recently advanced $\textit{de novo}$ protein design by learning the statistical regularities of natural structures. However, current approaches face three key limitations: (1) Existing methods cannot jointly learn protein geometry and design tasks, where pretraining can be a solution; (2) Current pretraining methods mostly rely on local, non-rigid atomic representations for property prediction downstream tasks, limiting global geometric understanding for protein generation tasks; and (3) Existing approaches have yet to effectively model the rich dynamic and conformational information of protein structures. To overcome these issues, we introduce $\textbf{RigidSSL}$ ($\textit{Rigidity-Aware Self-Supervised Learning}$), a geometric pretraining framework that front-loads geometry learning prior to generative finetuning. Phase I (RigidSSL-Perturb) learns geometric priors from 432K structures from the AlphaFold Protein Structure Database with simulated perturbations. Phase II (RigidSSL-MD) refines these representations on 1.3K molecular dynamics trajectories to capture physically realistic transitions. Underpinning both phases is a bi-directional, rigidity-aware flow matching objective that jointly optimizes translational and rotational dynamics to maximize mutual information between conformations. Empirically, RigidSSL variants improve designability by up to 43\% while enhancing novelty and diversity in unconditional generation. Furthermore, RigidSSL-Perturb improves the success rate by 5.8\% in zero-shot motif scaffolding and RigidSSL-MD captures more biophysically realistic conformational ensembles in G protein-coupled receptor modeling. The code is available at: this https URL. 

---
# PlayWrite: A Multimodal System for AI Supported Narrative Co-Authoring Through Play in XR 

**Authors**: Esen K. Tütüncü, Qian Zhou, Frederik Brudy, George Fitzmaurice, Fraser Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2603.02366)  

**Abstract**: Current AI writing tools, which rely on text prompts, poorly support the spatial and interactive nature of storytelling where ideas emerge from direct manipulation and play. We present PlayWrite, a mixed-reality system where users author stories by directly manipulating virtual characters and props. A multi-agent AI pipeline interprets these actions into Intent Frames -structured narrative beats visualized as rearrangeable story marbles on a timeline. A large language model then transforms the user's assembled sequence into a final narrative. A user study (N=13) with writers from varying domains found that PlayWrite fosters a highly improvisational and playful process. Users treated the AI as a collaborative partner, using its unexpected responses to spark new ideas and overcome creative blocks. PlayWrite demonstrates an approach for co-creative systems that move beyond text to embrace direct manipulation and play as core interaction modalities. 

---
# Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment: Case study with Tetris 

**Authors**: Haochuan Kevin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02348)  

**Abstract**: We study diffusion-based model predictive control (Diffusion-MPC) in discrete combinatorial domains using Tetris as a case study. Our planner samples candidate placement sequences with a MaskGIT-style discrete denoiser and selects actions via reranking. We analyze three key factors: (1) feasibility-constrained sampling via logit masking over valid placements, (2) reranking strategies using a heuristic score, a pretrained DQN critic, and a hybrid combination, and (3) compute scaling in candidate count and planning horizon. We find that feasibility masking is necessary in discrete domains, removing invalid action mass (46%) and yielding a 6.8% improvement in score and 5.6% improvement in survival over unconstrained sampling. Naive DQN reranking is systematically misaligned with rollout quality, producing high decision regret (mean 17.6, p90 36.6). Shorter planning horizons outperform longer ones under sparse and delayed rewards, suggesting uncertainty compounding in long imagined rollouts. Overall, compute choices (K, H) determine dominant failure modes: small K limits candidate quality, while larger H amplifies misranking and model mismatch. Our findings highlight structural challenges of diffusion planners in discrete environments and provide practical diagnostics for critic integration. 

---
# Large Electron Model: A Universal Ground State Predictor 

**Authors**: Timothy Zaklama, Max Geier, Liang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02346)  

**Abstract**: We introduce Large Electron Model, a single neural network model that produces variational wavefunctions of interacting electrons over the entire Hamiltonian parameter manifold. Our model employs the Fermi Sets architecture, a universal representation of many-body fermionic wavefunctions, which is further conditioned on Hamiltonian parameter and particle number. On interacting electrons in a two-dimensional harmonic potential, a single trained model accurately predicts the ground state wavefunction while generalizing across unseen coupling strengths and particle-number sectors, producing both accurate real-space charge densities and ground state energies, even up to $50$ particles. Our results establish a foundation model method for material discovery that is grounded in the variational principle, while accurately treating strong electron correlation beyond the capacity of density functional theory. 

---
# RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection 

**Authors**: Sami Abuzakuk, Lucas Crijns, Anne-Marie Kermarrec, Rafael Pires, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2603.02345)  

**Abstract**: Infrastructure as code (IaC) tools automate cloud provisioning but verifying that deployed systems remain consistent with the IaC specifications remains challenging. Such configuration drift occurs because of bugs in the IaC specification, manual changes, or system updates. Large language model (LLM)-based agentic AI systems can automate the analysis of large volumes of telemetry data, making them suitable for the detection of configuration drift. However, existing agentic systems implicitly assume that the tools they invoke always return correct outputs, making them vulnerable to erroneous tool responses. Since agents cannot distinguish whether an anomalous tool output reflects a real infrastructure problem or a broken tool, such errors may cause missed drift or false alarms, reducing reliability precisely when it is most needed. We introduce RIVA (Robust Infrastructure by Verification Agents), a novel multi-agent system that performs robust IaC verification even when tools produce incorrect or misleading outputs. RIVA employs two specialized agents, a verifier agent and a tool generation agent, that collaborate through iterative cross-validation, multi-perspective verification, and tool call history tracking. Evaluation on the AIOpsLab benchmark demonstrates that RIVA, in the presence of erroneous tool responses, recovers task accuracy from 27.3% when using a baseline ReAct agent to 50.0% on average. RIVA also improves task accuracy 28% to 43.8% without erroneous tool responses. Our results show that cross-validation of diverse tool calls enables more reliable autonomous infrastructure verification in production cloud environments. 

---
# Preconditioned Score and Flow Matching 

**Authors**: Shadab Ahamed, Eshed Gal, Simon Ghyselincks, Md Shahriar Rahim Siddiqui, Moshe Eliasof, Eldad Haber  

**Link**: [PDF](https://arxiv.org/pdf/2603.02337)  

**Abstract**: Flow matching and score-based diffusion train vector fields under intermediate distributions $p_t$, whose geometry can strongly affect their optimization. We show that the covariance $\Sigma_t$ of $p_t$ governs optimization bias: when $\Sigma_t$ is ill-conditioned, and gradient-based training rapidly fits high-variance directions while systematically under-optimizing low-variance modes, leading to learning that plateaus at suboptimal weights. We formalize this effect in analytically tractable settings and propose reversible, label-conditional \emph{preconditioning} maps that reshape the geometry of $p_t$ by improving the conditioning of $\Sigma_t$ without altering the underlying generative model. Rather than accelerating early convergence, preconditioning primarily mitigates optimization stagnation by enabling continued progress along previously suppressed directions. Across MNIST latent flow matching, and additional high-resolution datasets, we empirically track conditioning diagnostics and distributional metrics and show that preconditioning consistently yields better-trained models by avoiding suboptimal plateaus. 

---
# ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense 

**Authors**: Nancy Lau, Louis Sloot, Jyoutir Raj, Giuseppe Marco Boscardin, Evan Harris, Dylan Bowman, Mario Brajkovski, Jaideep Chawla, Dan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02297)  

**Abstract**: Large language models (LLMs) are increasingly being deployed as software engineering agents that autonomously contribute to repositories. A major benefit these agents present is their ability to find and patch security vulnerabilities in the codebases they oversee. To estimate the capability of agents in this domain, we introduce ZeroDayBench, a benchmark where LLM agents find and patch 22 novel critical vulnerabilities in open-source codebases. We focus our efforts on three popular frontier agentic LLMs: GPT-5.2, Claude Sonnet 4.5, and Grok 4.1. We find that frontier LLMs are not yet capable of autonomously solving our tasks and observe some behavioral patterns that suggest how these models can be improved in the domain of proactive cyberdefense. 

---
# The Malignant Tail: Spectral Segregation of Label Noise in Over-Parameterized Networks 

**Authors**: Zice Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02293)  

**Abstract**: While implicit regularization facilitates benign overfitting in low-noise regimes, recent theoretical work predicts a sharp phase transition to harmful overfitting as the noise-to-signal ratio increases. We experimentally isolate the geometric mechanism of this transition: the Malignant Tail, a failure mode where networks functionally segregate signal and noise, reducing coherent semantic features into low-rank subspaces while pushing stochastic label noise into high-frequency orthogonal components, distinct from systematic or corruption-aligned noise. Through a Spectral Linear Probe of training dynamics, we demonstrate that Stochastic Gradient Descent (SGD) fails to suppress this noise, instead implicitly biasing it toward high-frequency orthogonal subspaces, effectively preserving signal-noise separability. We show that this geometric separation is distinct from simple variance reduction in untrained models. In trained networks, SGD actively segregates noise, allowing post-hoc Explicit Spectral Truncation (d << D) to surgically prune the noise-dominated subspace. This approach recovers the optimal generalization capability latent in the converged model. Unlike unstable temporal early stopping, Geometric Truncation provides a stable post-hoc intervention. Our findings suggest that under label noise, excess spectral capacity is not harmless redundancy but a latent structural liability that allows for noise memorization, necessitating explicit rank constraints to filter stochastic corruptions for robust generalization. 

---
# Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection 

**Authors**: Yaoteng Zhang, Zhou Qing, Junyu Gao, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02286)  

**Abstract**: Incremental Object Detection (IOD) aims to continuously learn new object categories without forgetting previously learned ones. Recently, prompt-based methods have gained popularity for their replay-free design and parameter efficiency. However, due to prompt coupling and prompt drift, these methods often suffer from prompt degradation during continual adaptation. To address these issues, we propose a novel prompt-decoupled framework called PDP. PDP innovatively designs a dual-pool prompt decoupling paradigm, which consists of a shared pool used to capture task-general knowledge for forward transfer, and a private pool used to learn task-specific discriminative features. This paradigm explicitly separates task-general and task-specific prompts, preventing interference between prompts and mitigating prompt coupling. In addition, to counteract prompt drift resulting from inconsistent supervision where old foreground objects are treated as background in subsequent tasks, PDP introduces a Prototypical Pseudo-Label Generation (PPG) module. PPG can dynamically update the class prototype space during training and use the class prototypes to further filter valuable pseudo-labels, maintaining supervisory signal consistency throughout the incremental process. PDP achieves state-of-the-art performance on MS-COCO (with a 9.2\% AP improvement) and PASCAL VOC (with a 3.3\% AP improvement) benchmarks, highlighting its potential in balancing stability and plasticity. The code and dataset are released at: this https URL\_IOD/tree/main 

---
# Quantum-Inspired Fine-Tuning for Few-Shot AIGC Detection via Phase-Structured Reparameterization 

**Authors**: Kaiyang Xing, Han Fang, Zhaoyun Chen, Zhonghui Li, Yang Yang, Weiming Zhang, Guoping Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.02281)  

**Abstract**: Recent studies show that quantum neural networks (QNNs) generalize well in few-shot regimes. To extend this advantage to large-scale tasks, we propose Q-LoRA, a quantum-enhanced fine-tuning scheme that integrates lightweight QNNs into the low-rank adaptation (LoRA) adapter. Applied to AI-generated content (AIGC) detection, Q-LoRA consistently outperforms standard LoRA under few-shot settings. We analyze the source of this improvement and identify two possible structural inductive biases from QNNs: (i) phase-aware representations, which encode richer information across orthogonal amplitude-phase components, and (ii) norm-constrained transformations, which stabilize optimization via inherent orthogonality. However, Q-LoRA incurs non-trivial overhead due to quantum simulation. Motivated by our analysis, we further introduce H-LoRA, a fully classical variant that applies the Hilbert transform within the LoRA adapter to retain similar phase structure and constraints. Experiments on few-shot AIGC detection show that both Q-LoRA and H-LoRA outperform standard LoRA by over 5% accuracy, with H-LoRA achieving comparable accuracy at significantly lower cost in this task. 

---
# Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning 

**Authors**: Jinge Ma, Fengqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02280)  

**Abstract**: With the widespread adoption of deep learning in visual tasks, Class-Incremental Learning (CIL) has become an important paradigm for handling dynamically evolving data distributions. However, CIL faces the core challenge of catastrophic forgetting, often manifested as a prediction bias toward new classes. Existing methods mainly attribute this bias to intra-task class imbalance and focus on corrections at the classifier head. In this paper, we highlight an overlooked factor -- temporal imbalance -- as a key cause of this bias. Earlier classes receive stronger negative supervision toward the end of training, leading to asymmetric precision and recall. We establish a temporal supervision model, formally define temporal imbalance, and propose Temporal-Adjusted Loss (TAL), which uses a temporal decay kernel to construct a supervision strength vector and dynamically reweight the negative supervision in cross-entropy loss. Theoretical analysis shows that TAL degenerates to standard cross-entropy under balanced conditions and effectively mitigates prediction bias under imbalance. Extensive experiments demonstrate that TAL significantly reduces forgetting and improves performance on multiple CIL benchmarks, underscoring the importance of temporal modeling for stable long-term learning. 

---
# Quantifying Frontier LLM Capabilities for Container Sandbox Escape 

**Authors**: Rahul Marchand, Art O Cathain, Jerome Wynne, Philippos Maximos Giavridis, Sam Deverett, John Wilkinson, Jason Gwartz, Harry Coppock  

**Link**: [PDF](https://arxiv.org/pdf/2603.02277)  

**Abstract**: Large language models (LLMs) increasingly act as autonomous agents, using tools to execute code, read and write files, and access networks, creating novel security risks. To mitigate these risks, agents are commonly deployed and evaluated in isolated "sandbox" environments, often implemented using Docker/OCI containers. We introduce SANDBOXESCAPEBENCH, an open benchmark that safely measures an LLM's capacity to break out of these sandboxes. The benchmark is implemented as an Inspect AI Capture the Flag (CTF) evaluation utilising a nested sandbox architecture with the outer layer containing the flag and no known vulnerabilities. Following a threat model of a motivated adversarial agent with shell access inside a container, SANDBOXESCAPEBENCH covers a spectrum of sandboxescape mechanisms spanning misconfiguration, privilege allocation mistakes, kernel flaws, and runtime/orchestration weaknesses. We find that, when vulnerabilities are added, LLMs are able to identify and exploit them, showing that use of evaluation like SANDBOXESCAPEBENCH is needed to ensure sandboxing continues to provide the encapsulation needed for highly-capable models. 

---
# Contextual Invertible World Models: A Neuro-Symbolic Agentic Framework for Colorectal Cancer Drug Response 

**Authors**: Christopher Baker, Karen Rafferty, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02274)  

**Abstract**: Precision oncology is currently limited by the small-N, large-P paradox, where high-dimensional genomic data is abundant, but high-quality drug response samples are often sparse. While deep learning models achieve high predictive accuracy, they remain black boxes that fail to provide the causal mechanisms required for clinical decision-making. We present a Neuro-Symbolic Agentic Framework that bridges this gap by integrating a quantitative machine learning World Model with an LLM-based agentic reasoning layer. Our system utilises a forensic data pipeline built on the Sanger GDSC dataset (N=83), achieving a robust predictive correlation (r=0.504) and a significant performance gain through the explicit modelling of clinical context, specifically Microsatellite Instability (MSI) status. We introduce the concept of Inverse Reasoning, where the agentic layer performs in silico CRISPR perturbations to predict how specific genomic edits, such as APC or TP53 repair, alter drug sensitivity. By distinguishing between therapeutic opportunity and contextual resistance, and validating these findings against human clinical data (p=0.023), our framework provides a transparent, biologically grounded path towards explainable AI in cancer research. 

---
# Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures 

**Authors**: Manoj Vishwanathan, Suvinay Subramanian, Anand Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2603.02271)  

**Abstract**: Vision-Language-Action (VLA) models are an emerging class of workloads critical for robotics and embodied AI at the edge. As these models scale, they demonstrate significant capability gains, yet they must be deployed locally to meet the strict latency requirements of real-time applications. This paper characterizes VLA performance on two generations of edge hardware, viz. the Nvidia Jetson Orin and Thor platforms. Using MolmoAct-7B, a state-of-the-art VLA model, we identify a primary execution bottleneck: up to 75% of end-to-end latency is consumed by the memory-bound action-generation phase. Through analytical modeling and simulations, we project the hardware requirements for scaling to 100B parameter models. We also explore the impact of high-bandwidth memory technologies and processing-in-memory (PIM) as promising future pathways in edge systems for embodied AI. 

---
# PRISM: Exploring Heterogeneous Pretrained EEG Foundation Model Transfer to Clinical Differential Diagnosis 

**Authors**: Jeet Bandhu Lahiri, Parshva Runwal, Arvasu Kulkarni, Mahir Jain, Aditya Ray Mishra, Siddharth Panwar, Sandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02268)  

**Abstract**: EEG foundation models are typically pretrained on narrow-source clinical archives and evaluated on benchmarks from the same ecosystem, leaving unclear whether representations encode neural physiology or recording-distribution artifacts. We introduce PRISM (Population Representative Invariant Signal Model), a masked autoencoder ablated along two axes -- pretraining population and downstream adaptation -- with architecture and preprocessing fixed. We compare a narrow-source EU/US corpus (TUH + PhysioNet) against a geographically diverse pool augmented with multi-center South Asian clinical recordings across multiple EEG systems. Three findings emerge. First, narrow-source pretraining yields stronger linear probes on distribution-matched benchmarks, while diverse pretraining produces more adaptable representations under fine-tuning -- a trade-off invisible under single-protocol evaluation. Trained on three source corpora, PRISM matches or outperforms REVE (92 datasets, 60,000+ hours) on the majority of tasks, demonstrating that targeted diversity can substitute for indiscriminate scale and that dataset count is a confounding variable in model comparison. Second, on a clinically challenging and previously untested task -- distinguishing epilepsy from diagnostic mimickers via interictal EEG -- the diverse checkpoint outperforms the narrow-source checkpoint by +12.3 pp balanced accuracy, the largest gap across all evaluations. Third, systematic inconsistencies between EEG-Bench and EEG-FM-Bench reverse model rankings on identical datasets by up to 24 pp; we identify six concrete sources including split construction, checkpoint selection, segment length, and normalization, showing these factors compound non-additively. 

---
# Boosting Meta-Learning for Few-Shot Text Classification via Label-guided Distance Scaling 

**Authors**: Yunlong Gao, Xinyue Liu, Yingbo Wang, Linlin Zong, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02267)  

**Abstract**: Few-shot text classification aims to recognize unseen classes with limited labeled text samples. Existing approaches focus on boosting meta-learners by developing complex algorithms in the training stage. However, the labeled samples are randomly selected during the testing stage, so they may not provide effective supervision signals, leading to misclassification. To address this issue, we propose a \textbf{L}abel-guided \textbf{D}istance \textbf{S}caling (LDS) strategy. The core of our method is exploiting label semantics as supervision signals in both the training and testing stages. Specifically, in the training stage, we design a label-guided loss to inject label semantic information, pulling closer the sample representations and corresponding label representations. In the testing stage, we propose a Label-guided Scaler which scales sample representations with label semantics to provide additional supervision signals. Thus, even if labeled sample representations are far from class centers, our Label-guided Scaler pulls them closer to their class centers, thereby mitigating the misclassification. We combine two common meta-learners to verify the effectiveness of the method. Extensive experimental results demonstrate that our approach significantly outperforms state-of-the-art models. All datasets and codes are available at this https URL. 

---
# When Scaling Fails: Mitigating Audio Perception Decay of LALMs via Multi-Step Perception-Aware Reasoning 

**Authors**: Ruixiang Mao, Xiangnan Ma, Dan Chen, Ziming Zhu, Yuan Ge, Aokai Hao, Haishu Zhao, Yifu Huo, Qing Yang, Kaiyan Chang, Xiaoqian Liu, Chenglong Wang, Qiaozhi He, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02266)  

**Abstract**: Test-Time Scaling has shown notable efficacy in addressing complex problems through scaling inference compute. However, within Large Audio-Language Models (LALMs), an unintuitive phenomenon exists: post-training models for structured reasoning trajectories results in marginal or even negative gains compared to post-training for direct answering. To investigate it, we introduce CAFE, an evaluation framework designed to precisely quantify audio reasoning errors. Evaluation results reveal LALMs struggle with perception during reasoning and encounter a critical bottleneck: reasoning performance suffers from audio perception decay as reasoning length extends. To address it, we propose MPAR$^2$, a paradigm that encourages dynamic perceptual reasoning and decomposes complex questions into perception-rich sub-problems. Leveraging reinforcement learning, MPAR$^2$ improves perception performance on CAFE from 31.74% to 63.51% and effectively mitigates perception decay, concurrently enhancing reasoning capabilities to achieve a significant 74.59% accuracy on the MMAU benchmark. Further analysis demonstrates that MPAR$^2$ reinforces LALMs to attend to audio input and dynamically adapts reasoning budget to match task complexity. 

---
# High-order Knowledge Based Network Controllability Robustness Prediction: A Hypergraph Neural Network Approach 

**Authors**: Shibing Mo, Jiarui Zhang, Jiayu Xie, Xiangyi Teng, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02265)  

**Abstract**: In order to evaluate the invulnerability of networks against various types of attacks and provide guidance for potential performance enhancement as well as controllability maintenance, network controllability robustness (NCR) has attracted increasing attention in recent years. Traditionally, controllability robustness is determined by attack simulations, which are computationally time-consuming and only applicable to small-scale networks. Although some machine learning-based methods for predicting network controllability robustness have been proposed, they mainly focus on pairwise interactions in complex networks, and the underlying relationships between high-order structural information and controllability robustness have not been explored. In this paper, a dual hypergraph attention neural network model based on high-order knowledge (NCR-HoK) is proposed to accomplish robustness learning and controllability robustness curve prediction. Through a node feature encoder, hypergraph construction with high-order relations, and a dedicated dual hypergraph attention module, the proposed method can effectively learn three types of network information simultaneously: explicit structural information in the original graph, high-order connection information in local neighborhoods, and hidden features in the embedding space. Notably, we explore for the first time the impact of high-order knowledge on network controllability robustness. Compared with state-of-the-art methods for network robustness learning, the proposed method achieves superior performance on both synthetic and real-world networks with low computational overhead. 

---
# Social-JEPA: Emergent Geometric Isomorphism 

**Authors**: Haoran Zhang, Youjin Wang, Yi Duan, Rong Fu, Dianyu Zhao, Sicheng Fan, Shuaishuai Cao, Wentao Guo, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.02263)  

**Abstract**: World models compress rich sensory streams into compact latent codes that anticipate future observations. We let separate agents acquire such models from distinct viewpoints of the same environment without any parameter sharing or coordination. After training, their internal representations exhibit a striking emergent property: the two latent spaces are related by an approximate linear isometry, enabling transparent translation between them. This geometric consensus survives large viewpoint shifts and scant overlap in raw pixels. Leveraging the learned alignment, a classifier trained on one agent can be ported to the other with no additional gradient steps, while distillation-like migration accelerates later learning and markedly reduces total compute. The findings reveal that predictive learning objectives impose strong regularities on representation geometry, suggesting a lightweight path to interoperability among decentralized vision systems. The code is available at this https URL. 

---
# Silent Sabotage During Fine-Tuning: Few-Shot Rationale Poisoning of Compact Medical LLMs 

**Authors**: Jingyuan Xie, Wenjie Wang, Ji Wu, Jiandong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02262)  

**Abstract**: Supervised fine-tuning (SFT) is essential for the development of medical large language models (LLMs), yet prior poisoning studies have mainly focused on the detectable backdoor attacks. We propose a novel poisoning attack targeting the reasoning process of medical LLMs during SFT. Unlike backdoor attacks, our method injects poisoned rationales into few-shot training data, leading to stealthy degradation of model performance on targeted medical topics. Results showed that knowledge overwriting was ineffective, while rationale poisoning caused significant decline on the accuracy of the target subject, as long as no correct samples of the same subject appear in the dataset. A minimum number and ratio of poisoned samples was needed to carry out an effective and stealthy attack, which was more efficient and accurate than catastrophic forgetting. We demonstrate though this study the risk of SFT-stage poisoning, hoping to spur more studies of defense in the sensitive medical domain. 

---
# Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry 

**Authors**: Kyle Elliott Mathewson  

**Link**: [PDF](https://arxiv.org/pdf/2603.02258)  

**Abstract**: Do neural machine translation models learn language-universal conceptual representations, or do they merely cluster languages by surface similarity? We investigate this question by probing the representation geometry of Meta's NLLB-200, a 200-language encoder-decoder Transformer, through six experiments that bridge NLP interpretability with cognitive science theories of multilingual lexical organization. Using the Swadesh core vocabulary list embedded across 135 languages, we find that the model's embedding distances significantly correlate with phylogenetic distances from the Automated Similarity Judgment Program ($\rho = 0.13$, $p = 0.020$), demonstrating that NLLB-200 has implicitly learned the genealogical structure of human languages. We show that frequently colexified concept pairs from the CLICS database exhibit significantly higher embedding similarity than non-colexified pairs ($U = 42656$, $p = 1.33 \times 10^{-11}$, $d = 0.96$), indicating that the model has internalized universal conceptual associations. Per-language mean-centering of embeddings improves the between-concept to within-concept distance ratio by a factor of 1.19, providing geometric evidence for a language-neutral conceptual store analogous to the anterior temporal lobe hub identified in bilingual neuroimaging. Semantic offset vectors between fundamental concept pairs (e.g., man to woman, big to small) show high cross-lingual consistency (mean cosine = 0.84), suggesting that second-order relational structure is preserved across typologically diverse languages. We release InterpretCognates, an open-source interactive toolkit for exploring these phenomena, alongside a fully reproducible analysis pipeline. 

---
# MEBM-Speech: Multi-scale Enhanced BrainMagic for Robust MEG Speech Detection 

**Authors**: Li Songyi, Zheng Linze, Liang Jinghua, Zhang Zifeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.02255)  

**Abstract**: We propose MEBM-Speech, a multi-scale enhanced neural decoder for speech activity detection from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Speech integrates three complementary temporal modeling mechanisms: a multi-scale convolutional module for short-term pattern extraction, a bidirectional LSTM (BiLSTM) for long-range context modeling, and a depthwise separable convolutional layer for efficient cross-scale feature fusion. A lightweight temporal jittering strategy and average pooling further improve onset robustness and boundary stability. The model performs continuous probabilistic decoding of MEG signals, enabling fine-grained detection of speech versus silence states - an ability crucial for both cognitive neuroscience and clinical applications. Comprehensive evaluations on the LibriBrain Competition 2025 Track1 benchmark demonstrate strong performance, achieving an average F1 macro of 89.3% on the validation set and comparable results on the official test leaderboard. These findings highlight the effectiveness of multi-scale temporal representation learning for robust MEG-based speech decoding. 

---
# MEBM-Phoneme: Multi-scale Enhanced BrainMagic for End-to-End MEG Phoneme Classification 

**Authors**: Liang Jinghua, Zhang Zifeng, Li Songyi, Zheng Linze  

**Link**: [PDF](https://arxiv.org/pdf/2603.02254)  

**Abstract**: We propose MEBM-Phoneme, a multi-scale enhanced neural decoder for phoneme classification from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Phoneme integrates a short-term multi-scale convolutional module to augment the native mid-term encoder, with fused representations via depthwise separable convolution for efficient cross-scale integration. A convolutional attention layer dynamically weights temporal dependencies to refine feature aggregation. To address class imbalance and session-specific distributional shifts, we introduce a stacking-based local validation set alongside weighted cross-entropy loss and random temporal augmentation. Comprehensive evaluations on LibriBrain Competition 2025 Track2 demonstrate robust generalization, achieving competitive phoneme decoding accuracy on the validation and official test leaderboard. These results underscore the value of hierarchical temporal modeling and training stabilization for advancing MEG-based speech perception analysis. 

---
# Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics 

**Authors**: Mandip Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2603.02252)  

**Abstract**: We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convolved with a real room impulse response from the RIR-Mega corpus, with stratified splits by reverberation time (RT60) and direct-to-reverberant ratio (DRR). We evaluate five Whisper models (tiny through large-v3) on 1600 test samples and report word error rate (WER) and character error rate (CER) under clean and reverberant conditions. Reverberation consistently degrades performance across all model sizes; the reverb penalty in WER ranges from 0.12 to 1.07 percentage points depending on the model. We release the dataset, evaluation code, and baseline results to support reproducible research on robust ASR. 

---
# A Benchmark Analysis of Graph and Non-Graph Methods for Caenorhabditis Elegans Neuron Classification 

**Authors**: Jingqi Lu, Keqi Han, Yun Wang, Lu Mi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02241)  

**Abstract**: This study establishes a benchmark for Caenorhabditis elegans neuron classification, comparing four graph methods (GCN, GraphSAGE, GAT, GraphTransformer) against four non-graph methods (Logistic Regression, MLP, LOLCAT, NeuPRINT). Using the functional connectome, we classified Sensory, Interneuron, and Motor neurons based on Spatial, Connection, and Neuronal Activity features. Results show that attention-based GNNs significantly outperform baselines on the Spatial and Connection features. The Neuronal Activity features yielded poor performance, likely due to the low temporal resolution of the underlying neuronal activity data. Our benchmark validates the use of GNNs and highlights that Spatial and Connection features are key predictors for Caenorhabditis elegans neuron classes. Code is available at: this https URL. 

---
# Concept Heterogeneity-aware Representation Steering 

**Authors**: Laziz U. Abdullaev, Noelle Y. L. Wong, Ryan T. Z. Lee, Shiqi Jiang, Khoi N. M. Nguyen, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02237)  

**Abstract**: Representation steering offers a lightweight mechanism for controlling the behavior of large language models (LLMs) by intervening on internal activations at inference time. Most existing methods rely on a single global steering direction, typically obtained via difference-in-means over contrastive datasets. This approach implicitly assumes that the target concept is homogeneously represented across the embedding space. In practice, however, LLM representations can be highly non-homogeneous, exhibiting clustered, context-dependent structure, which renders global steering directions brittle. In this work, we view representation steering through the lens of optimal transport (OT), noting that standard difference-in-means steering implicitly corresponds to the OT map between two unimodal Gaussian distributions with identical covariance, yielding a global translation. To relax this restrictive assumption, we theoretically model source and target representations as Gaussian mixture models and formulate steering as a discrete OT problem between semantic latent clusters. From the resulting transport plan, we derive an explicit, input-dependent steering map via barycentric projection, producing a smooth, kernel-weighted combination of cluster-level shifts. We term this method Concept Heterogeneity-aware Representation Steering (CHaRS). Through numerous experimental settings, we show that CHaRS yields more effective behavioral control than global steering. 

---
# CUDABench: Benchmarking LLMs for Text-to-CUDA Generation 

**Authors**: Jiace Zhu, Wentao Chen, Qi Fan, Zhixing Ren, Junying Wu, Xing Zhe Chai, Chotiwit Rungrueangwutthinon, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.02236)  

**Abstract**: Recent studies have demonstrated the potential of Large Language Models (LLMs) in generating GPU Kernels. Current benchmarks focus on the translation of high-level languages into CUDA, overlooking the more general and challenging task of text-to-CUDA generation. Furthermore, given the hardware-specific and performance-critical features of GPU programming, accurately assessing the performance of LLM-generated GPU programs is nontrivial. In this work, we introduce CUDABench, a comprehensive benchmark designed to evaluate the text-to-CUDA capabilities of LLMs. First, we construct CUDABench-Set, which covers Breadth-Depth-Difficulty evaluation space in diverse application domains, including artificial intelligence, scientific computing, and data analytics, etc. Furthermore, we propose CUDABench-Score and Generative Verification Pipeline that assess (1) compilation correctness, (2) functional consistency through execution-based verification, and (3) a novel roofline-based metric, Performance-Score. Benchmarking state-of-the-art LLMs reveals insightful findings and challenges of text-to-CUDA, such as a notable mismatch between high compilation success rates and low functional correctness, a lack of domain-specific algorithmic knowledge, and suboptimal utilization of GPU hardware resources. Our benchmark is available at this https URL. 

---
# Talking with Verifiers: Automatic Specification Generation for Neural Network Verification 

**Authors**: Yizhak Y. Elboher, Reuven Peleg, Zhouxing Shi, Guy Katz, Jan Křetínský  

**Link**: [PDF](https://arxiv.org/pdf/2603.02235)  

**Abstract**: Neural network verification tools currently support only a narrow class of specifications, typically expressed as low-level constraints over raw inputs and outputs. This limitation significantly hinders their adoption and practical applicability across diverse application domains where correctness requirements are naturally expressed at a higher semantic level. This challenge is rooted in the inherent nature of deep neural networks, which learn internal representations that lack an explicit mapping to human-understandable features. To address this, we bridge this gap by introducing a novel component to the verification pipeline, making existing verification tools applicable to a broader range of domains and specification styles. Our framework enables users to formulate specifications in natural language, which are then automatically analyzed and translated into formal verification queries compatible with state-of-the-art neural network verifiers. We evaluate our approach on both structured and unstructured datasets, demonstrating that it successfully verifies complex semantic specifications that were previously inaccessible. Our results show that this translation process maintains high fidelity to user intent while incurring low computational overhead, thereby substantially extending the applicability of formal DNN verification to real-world, high-level requirements. 

---
# Structured vs. Unstructured Pruning: An Exponential Gap 

**Authors**: Davide Ferré, Frédéric Giroire, Emanuele Natale, Frederik Mallmann-Trenn  

**Link**: [PDF](https://arxiv.org/pdf/2603.02234)  

**Abstract**: The Strong Lottery Ticket Hypothesis (SLTH) posits that large, randomly initialized neural networks contain sparse subnetworks capable of approximating a target function at initialization without training, suggesting that pruning alone is sufficient. Pruning methods are typically classified as unstructured, where individual weights can be removed from the network, and structured, where parameters are removed according to specific patterns, as in neuron pruning. Existing theoretical results supporting the SLTH rely almost exclusively on unstructured pruning, showing that logarithmic overparameterization suffices to approximate simple target networks. In contrast, neuron pruning has received limited theoretical attention. In this work, we consider the problem of approximating a single bias-free ReLU neuron using a randomly initialized bias-free two-layer ReLU network, thereby isolating the intrinsic limitations of neuron pruning. We show that neuron pruning requires a starting network with $\Omega(d/\varepsilon)$ hidden neurons to $\varepsilon$-approximate a target ReLU neuron. In contrast, weight pruning achieves $\varepsilon$-approximation with only $O(d\log(1/\varepsilon))$ neurons, establishing an exponential separation between the two pruning paradigms. 

---
# Adaptive Personalized Federated Learning via Multi-task Averaging of Kernel Mean Embeddings 

**Authors**: Jean-Baptiste Fermanian, Batiste Le Bars, Aurélien Bellet  

**Link**: [PDF](https://arxiv.org/pdf/2603.02233)  

**Abstract**: Personalized Federated Learning (PFL) enables a collection of agents to collaboratively learn individual models without sharing raw data. We propose a new PFL approach in which each agent optimizes a weighted combination of all agents' empirical risks, with the weights learned from data rather than specified a priori. The novelty of our method lies in formulating the estimation of these collaborative weights as a kernel mean embedding estimation problem with multiple data sources, leveraging tools from multi-task averaging to capture statistical relationships between agents. This perspective yields a fully adaptive procedure that requires no prior knowledge of data heterogeneity and can automatically transition between global and local learning regimes. By recasting the objective as a high-dimensional mean estimation problem, we derive finite-sample guarantees on local excess risks for a broad class of distributions, explicitly quantifying the statistical gains of collaboration. To address communication constraints inherent to federated settings, we also propose a practical implementation based on random Fourier features, which allows one to trade communication cost for statistical efficiency. Numerical experiments validate our theoretical results. 

---
# Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback 

**Authors**: Amirhossein Afsharrad, Ruida Zhou, Luca Viano, Sanjay Lall, Mohammad Ghavamzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02232)  

**Abstract**: Reward modeling is crucial for aligning large language models with human preferences, yet current approaches lack a principled mathematical framework for leveraging ordinal preference data. When human annotators provide graded preferences on a Likert scale (e.g., significantly better, better, slightly better, negligibly better), existing methods typically apply ad-hoc heuristics, such as margin terms or scaling factors, to loss functions derived from binary preference models like Bradley-Terry. These approaches lack an underlying mathematical model for how ordinal preference data is generated. We present a theoretically grounded framework that formulates reward modeling with Likert scale preferences as a discrete ordinal regression problem. We derive two loss functions from this formulation: a negative log-likelihood loss and an all-threshold loss, both of which learn threshold parameters that naturally capture the ordinal structure of preferences. Unlike existing heuristic methods that manually specify fixed margins or scaling weights, our approach learns these parameters directly from data within a coherent probabilistic framework. Experimental results on multiple benchmarks demonstrate that our ordinal regression approach consistently achieves competitive or superior performance compared to existing heuristic methods across diverse evaluation categories including chat, reasoning, and safety tasks. Our work provides the first principled mathematical framework for incorporating Likert scale preferences into reward model training, moving beyond ad-hoc modifications of binary preference models to enable more effective utilization of fine-grained human feedback. 

---
# Physics-Informed Neural Networks with Architectural Physics Embedding for Large-Scale Wave Field Reconstruction 

**Authors**: Huiwen Zhang, Feng Ye, Chu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.02231)  

**Abstract**: Large-scale wave field reconstruction requires precise solutions but faces challenges with computational efficiency and accuracy. The physics-based numerical methods like Finite Element Method (FEM) provide high accuracy but struggle with large-scale or high-frequency problems due to prohibitive computational costs. Pure data-driven approaches excel in speed but often lack sufficient labeled data for complex scenarios. Physics-informed neural networks (PINNs) integrate physical principles into machine learning models, offering a promising solution by bridging these gaps. However, standard PINNs embed physical principles only in loss functions, leading to slow convergence, optimization instability, and spectral bias, limiting their ability for large-scale wave field reconstruction. This work introduces architecture physics embedded (PE)-PINN, which integrates additional physical guidance directly into the neural network architecture beyond Helmholtz equations and boundary conditions in loss functions. Specifically, a new envelope transformation layer is designed to mitigate spectral bias with kernels parameterized by source properties, material interfaces, and wave physics. Experiments demonstrate that PE-PINN achieves more than 10 times speedup in convergence compared to standard PINNs and several orders of magnitude reduction in memory usage compared to FEM. This breakthrough enables high-fidelity modeling for large-scale 2D/3D electromagnetic wave reconstruction involving reflections, refractions, and diffractions in room-scale domains, readily applicable to wireless communications, sensing, room acoustics, and other fields requiring large-scale wave field analysis. 

---
# Generalized Discrete Diffusion with Self-Correction 

**Authors**: Linxuan Wang, Ziyi Wang, Yikun Bai, Wei Deng, Guang Lin, Qifan Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.02230)  

**Abstract**: Self-correction is an effective technique for maintaining parallel sampling in discrete diffusion models with minimal performance degradation. Prior work has explored self-correction at inference time or during post-training; however, such approaches often suffer from limited generalization and may impair reasoning performance. GIDD pioneers pretraining-based self-correction via a multi-step BERT-style uniform-absorbing objective. However, GIDD relies on a continuous interpolation-based pipeline with opaque interactions between uniform transitions and absorbing masks, which complicates hyperparameter tuning and hinders practical performance. In this work, we propose a Self-Correcting Discrete Diffusion (SCDD) model to reformulate pretrained self-correction with explicit state transitions and learn directly in discrete time. Our framework also simplifies the training noise schedule, eliminates a redundant remasking step, and relies exclusively on uniform transitions to learn self-correction. Experiments at the GPT-2 scale demonstrate that our method enables more efficient parallel decoding while preserving generation quality. 

---
# Neural Paging: Learning Context Management Policies for Turing-Complete Agents 

**Authors**: Liang Chen, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02228)  

**Abstract**: The proof that Large Language Models (LLMs) augmented with external read-write memory constitute a computationally universal system has established the theoretical foundation for general-purpose agents. However, existing implementations face a critical bottleneck: the finite and costly Context Window, which functions not as infinite memory but as a scarce semantic cache. In this work, we introduce \textit{Neural Paging}, a hierarchical architecture that decouples symbolic reasoning from information resource management. We formulate the \textit{Context Paging Problem (CPP)} and propose a lightweight, differentiable \textit{Page Controller} designed to approximate ``Semantic Belady's Optimality'' -- retaining tokens with high future utility under explicit assumptions on access patterns. We provide theoretical analysis showing that, under bounded context window size~$K$, Neural Paging reduces the asymptotic complexity of long-horizon reasoning from quadratic $O(N^2)$ to $O(N \cdot K^2)$, and we derive a robustness bound (Theorem~4) that quantifies competitive-ratio degradation under policy-dependent access with bounded sensitivity. We validate these bounds on synthetic paging traces, confirming that the theoretical guarantees hold and identifying significant slack that motivates learned policies. 

---
# Characterizing and Predicting Wildfire Evacuation Behavior: A Dual-Stage ML Approach 

**Authors**: Sazzad Bin Bashar Polock, Anandi Dutta, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2603.02223)  

**Abstract**: Wildfire evacuation behavior is highly variable and influenced by complex interactions among household resources, preparedness, and situational cues. Using a large-scale MTurk survey of residents in California, Colorado, and Oregon, this study integrates unsupervised and supervised machine learning methods to uncover latent behavioral typologies and predict key evacuation outcomes. Multiple Correspondence Analysis, K-Modes clustering, and Latent Class Analysis reveal consistent subgroups differentiated by vehicle access, disaster planning, technological resources, pet ownership, and residential stability. Complementary supervised models show that transportation mode can be predicted with high reliability from household characteristics, whereas evacuation timing remains difficult to classify due to its dependence on dynamic, real-time fire conditions. These findings advance data-driven understanding of wildfire evacuation behavior and demonstrate how machine learning can support targeted preparedness strategies, resource allocation, and equitable emergency planning. 

---
# MedCalc-Bench Doesn't Measure What You Think: A Benchmark Audit and the Case for Open-Book Evaluation 

**Authors**: Artus Krohn-Grimberghe  

**Link**: [PDF](https://arxiv.org/pdf/2603.02222)  

**Abstract**: MedCalc-Bench is a widely used benchmark for evaluating LLM performance on clinical calculator
tasks, with state-of-the-art direct prompting scores plateauing around 35% on the Verified split
(HELM MedHELM leaderboard) and the best published approach-RL with verifiable rewards-reaching 74%.
We present three contributions that challenge the benchmark's current framing. First, we conduct a
systematic audit of the benchmark's calculator implementations, identifying and fixing over 20
errors ranging from critical formula inaccuracies to runtime bugs in a NeurIPS-published dataset.
Second, we show that a simple intervention-providing the model with the calculator specification at
inference time ("open-book" prompting)-raises accuracy from ~52% to 81-85% on GLM-4.6V and GLM-4.7,
surpassing all published results including RL-trained systems, without any fine-tuning. Third, we
establish an upper bound of 95-97% using GPT-5.2-Thinking, with residual errors attributable
primarily to ground-truth issues and dataset ambiguities. Our findings suggest that MedCalc-Bench
predominantly measures formula memorization and arithmetic precision rather than clinical
reasoning, and would be better framed as a tool-use evaluation. 

---
# MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs for Clinical Tabular Prediction 

**Authors**: Zizheng Zhang, Yiming Li, Justin Xu, Jinyu Wang, Rui Wang, Lei Song, Jiang Bian, David W Eyre, Jingjing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02221)  

**Abstract**: In healthcare tabular predictions, classical models with feature engineering often outperform neural approaches. Recent advances in Large Language Models enable the integration of domain knowledge into feature engineering, offering a promising direction. However, existing approaches typically rely on a broad search over predefined transformations, overlooking downstream model characteristics and feature importance signals. We present MedFeat, a feedback-driven and model-aware feature engineering framework that leverages LLM reasoning with domain knowledge and provides feature explanations based on SHAP values while tracking successful and failed proposals to guide feature discovery. By incorporating model awareness, MedFeat prioritizes informative signals that are difficult for the downstream model to learn directly due to its characteristics. Across a broad range of clinical prediction tasks, MedFeat achieves stable improvements over various baselines and discovers clinically meaningful features that generalize under distribution shift, demonstrating robustness across years and from ICU cohorts to general hospitalized patients, thereby offering insights into real-world deployment. Code required to reproduce our experiments will be released, subject to dataset agreements and institutional policies. 

---
# Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting 

**Authors**: Yixin Wang, Yifan Hu, Peiyuan Liu, Naiqi Li, Dai Tao, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2603.02220)  

**Abstract**: Time series forecasting (TSF) remains a challenging problem due to the intricate entanglement of intraperiod-fluctuations and interperiod-trends. While recent advances have attempted to reshape 1D sequences into 2D period-phase representations, they suffer from two principal this http URL, treating reshaped tensors as static images results in a topological mismatch, as standard spatial operators sever chronological continuity at grid boundaries. Secondly, relying on uniform fixed-size representations allocates modeling capacity inefficiently and fails to provide the adaptive resolution required for compressible, non-stationary temporal patterns. To address these limitations, we introduce TimeGS, a novel framework that fundamentally shifts the forecasting paradigm from regression to 2D generative rendering. By reconceptualizing the future sequence as a continuous latent surface, TimeGS utilizes the inherent anisotropy of Gaussian kernels to adaptively model complex variations with flexible geometric alignment. To realize this, we introduce a Multi-Basis Gaussian Kernel Generation (MB-GKG) block that synthesizes kernels from a fixed dictionary to stabilize optimization, and a Multi-Period Chronologically Continuous Rasterization (MP-CCR) block that enforces strict temporal continuity across periodic boundaries. Comprehensive experiments on standard benchmark datasets demonstrate that TimeGS attains state-of-the-art performance. 

---
# NExT-Guard: Training-Free Streaming Safeguard without Token-Level Labels 

**Authors**: Junfeng Fang, Nachuan Chen, Houcheng Jiang, Dan Zhang, Fei Shen, Xiang Wang, Xiangnan He, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.02219)  

**Abstract**: Large language models are increasingly deployed in streaming scenarios, rendering conventional post-hoc safeguards ineffective as they fail to interdict unsafe content in real-time. While streaming safeguards based on token-level supervised training could address this, they necessitate expensive annotations and suffer from severe overfitting. In this work, we challenge the paradigm that streaming safety must rely on token-level supervised training. Instead, it is an inherent capability of well-trained post-hoc safeguards, as they already encode token-level risk signals in hidden representations. Hence, we introduce NExT-Guard, a training-free framework that achieves streaming safeguards by monitoring interpretable latent features from Sparse Autoencoders (SAEs). It uses pretrained SAEs from publicly available base LLMs, enabling flexible, low-cost deployment without token-level supervision. Experimental results show that NExT-Guard outperforms both post-hoc and streaming safeguards based on supervised training, with superior robustness across models, SAE variants, and risk scenarios. These results make NExT-Guard a universal and scalable paradigm for real-time safety, accelerating the practical deployment of streaming safeguards. 

---
# Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain 

**Authors**: Wei Liu, Siya Qi, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2603.02218)  

**Abstract**: Large language models (LLMs) make it plausible to build systems that improve through self-evolving loops, but many existing proposals are better understood as self-play and often plateau quickly. A central failure mode is that the loop synthesises more data without increasing learnable information for the next iteration. Through experiments on a self-play coding task, we reveal that sustainable self-evolution requires a self-synthesised data pipeline with learnable information that increases across iterations. We identify triadic roles that self-evolving LLMs play: the Proposer, which generates tasks; the Solver, which attempts solutions; and the Verifier, which provides training signals, and we identify three system designs that jointly target learnable information gain from this triadic roles perspective. Asymmetric co-evolution closes a weak-to-strong-to-weak loop across roles. Capacity growth expands parameter and inference-time budgets to match rising learnable information. Proactive information seeking introduces external context and new task sources that prevent saturation. Together, these modules provide a measurable, system-level path from brittle self-play dynamics to sustained self-evolution. 

---
# Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression 

**Authors**: Sieun Hyeon, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2603.02217)  

**Abstract**: Mixture-of-Experts (MoE) models scale capacity efficiently, but their massive parameter footprint creates a deployment-time memory bottleneck. We organize retraining-free MoE compression into three paradigms - Expert Pruning, Expert Editing, and Expert Merging - and show that persistent post-compression degradation largely stems from a neglected factor: router-expert mismatch when experts are changed but the router is left untouched. We argue that effective retraining-free compression should avoid updating expert parameters while allowing lightweight router calibration. To this end, we propose Router Knowledge Distillation (Router KD), which updates only a tiny fraction of parameters (the router) by distilling the original model's next-token distribution on unlabeled calibration data. Experiments across representative methods in all three paradigms demonstrate consistent performance recovery, with substantially larger gains in fine-grained MoEs (many small experts) than in coarse-grained MoEs due to their more complex routing decision boundaries. 

---
# ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue 

**Authors**: Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, Li Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02216)  

**Abstract**: Effective information seeking in multi-turn medical dialogues is critical for accurate diagnosis, especially when dealing with incomplete information. Aligning Large Language Models (LLMs) for these interactive scenarios is challenging due to the uncertainty inherent in user-agent interactions, which we formulate as a Hierarchical Markov Decision Process (H-MDP). While conventional Reinforcement Learning (RL) methods like Group Relative Policy Optimization (GRPO) struggle with long-horizon credit assignment and Proximal Policy Optimization (PPO) suffers from unstable value estimation in this context, we propose a novel uncertainty-aware Adaptive Tree Policy Optimization (ATPO) algorithm. Our method adaptively allocates the rollout budget to states with high uncertainty, quantified by a composite metric of Bellman error and action-value variance. This strategy enables more accurate value estimation, while fostering more efficient and diverse exploration. To mitigate the high computational cost of tree-based RL, we introduce two key optimizations: an uncertainty-guided pruning mechanism to minimize the number of rollouts, and an asynchronous search architecture that leverages KV cache reuse to maximize inference throughput. Extensive experiments on three public medical dialogue benchmarks demonstrate that our algorithm significantly outperforms several strong baselines, culminating in Qwen3-8B model surpassing the much larger GPT-4o ($+0.92\%$ accuracy). 

---
# RxnNano:Training Compact LLMs for Chemical Reaction and Retrosynthesis Prediction via Hierarchical Curriculum Learning 

**Authors**: Ran Li, Shimin Di, Haowei LI, Luanshi Bu, Jiachuan Wang, Wangze Ni, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.02215)  

**Abstract**: Chemical reaction prediction is pivotal for accelerating drug discovery and synthesis planning. Despite advances in data-driven models, current approaches are hindered by an overemphasis on parameter and dataset scaling. Some methods coupled with evaluation techniques that bypass fundamental challenges in reaction representation and fail to capture deep chemical intuition like reaction common sense and {topological atom mapping logic}. We argue that the core challenge lies in instilling these knowledge into the models. To this end, we propose a unified framework that prioritizes chemical understanding over scale through three key innovations: (1) a {Latent Chemical Consistency} objective that models reactions as movements on a continuous chemical manifold, ensuring reversible and physically plausible transformations; (2) a {Hierarchical Cognitive Curriculum} that trains the model through progressive stages, from syntax mastery to semantic reasoning, building robust chemical intuition; (3) {Atom-Map Permutation Invariance (AMPI)}, which force the model to learn invariant relational topology and balance multi-task learning. (4)and structured plan-based reasoning to improve the performance of the LLMs. Our compact {0.5B-parameter model}, \textbf{RxnNano} significantly outperforms fine-tuned LLMs ten times larger (>7B) and all the domain baselines, achieving a 23.5\% Top-1 accuracy improvement on rigorous benchmarks without test-time augmentation. this https URL. 

---
# GLEAN: Grounded Lightweight Evaluation Anchors for Contamination-Aware Tabular Reasoning 

**Authors**: Qizhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02212)  

**Abstract**: Tabular reasoning benchmarks mix semantic inference, numerical computation, and brittle table formatting, yet evaluations for small models remain vulnerable to contamination, dataset artifacts, and retrieval failures. We propose GLEAN, a lightweight evaluation protocol that integrates contamination-aware probes, weak-supervision governance, retrieval-reasoning diagnostics, and structured error attribution under tight hardware constraints. We evaluate across TabFact, WTQ via Squall, TableBench, RobuT, and SciTab under a 16GB GPU budget. Using Squall gold SQL as an executable anchor (95.2% execution), GLEAN assigns a deterministic error taxonomy (L0-L4 plus L0.5 context miss) and reveals a stable error-mode separation: TAPEX errors skew toward grounding (L3) while TAPAS errors skew toward hallucination/abstention (L2/L0). We validate evidence-row heuristics against SQL-derived rows on simple queries (0.62 precision / 0.71 recall; hybrid recall 0.81) and show that retrieval Recall@K can saturate even when end-to-end EM/F1 remains limited, motivating attribution beyond raw recall. We release a modular framework with audits and sensitivity checks to make small-model tabular evaluation more contamination-aware and diagnostic. 

---
# Param$Δ$ for Direct Weight Mixing: Post-Train Large Language Model at Zero Cost 

**Authors**: Sheng Cao, Mingrui Wu, Karthik Prasad, Yuandong Tian, Zechun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21023)  

**Abstract**: The post-training phase of large language models is essential for enhancing capabilities such as instruction-following, reasoning, and alignment with human preferences. However, it demands extensive high-quality data and poses risks like overfitting, alongside significant computational costs due to repeated post-training and evaluation after each base model update. This paper introduces $Param\Delta$, a novel method that streamlines post-training by transferring knowledge from an existing post-trained model to a newly updated base model with ZERO additional training. By computing the difference between post-trained model weights ($\Theta_\text{post}$) and base model weights ($\Theta_\text{base}$), and adding this to the updated base model ($\Theta'_\text{base}$), we define $Param\Delta$ Model as: $\Theta_{\text{Param}\Delta} = \Theta_\text{post} - \Theta_\text{base} + \Theta'_\text{base}$. This approach surprisingly equips the new base model with post-trained capabilities, achieving performance comparable to direct post-training. We did analysis on LLama3, Llama3.1, Qwen, and DeepSeek-distilled models. Results indicate $Param\Delta$ Model effectively replicates traditional post-training. For example, the $Param\Delta$ Model obtained from 70B Llama3-inst, Llama3-base, Llama3.1-base models attains approximately 95\% of Llama3.1-inst model's performance on average. $Param\Delta$ brings a new perspective on how to fully leverage models in the open-weight community, where checkpoints for base and instruct models are readily available and frequently updated, by providing a cost-free framework to accelerate the iterative cycle of model development. 

---
# On the Parameter Estimation of Sinusoidal Models for Speech and Audio Signals 

**Authors**: George P. Kafentzis  

**Link**: [PDF](https://arxiv.org/pdf/2401.01255)  

**Abstract**: In this paper, we examine the parameter estimation performance of three well-known sinusoidal models for speech and audio. The first one is the standard Sinusoidal Model (SM), which is based on the Fast Fourier Transform (FFT). The second is the Exponentially Damped Sinusoidal Model (EDSM) which has been proposed in the last decade, and utilizes a subspace method for parameter estimation, and finally the extended adaptive Quasi-Harmonic Model (eaQHM), which has been recently proposed for AM-FM decomposition, and estimates the signal parameters using Least Squares on a set of basis function that are adaptive to the local characteristics of the signal. The parameter estimation of each model is briefly described and its performance is compared to the others in terms of signal reconstruction accuracy versus window size on a variety of synthetic signals and versus the number of sinusoids on real signals. The latter include highly non stationary signals, such as singing voices and guitar solos. The advantages and disadvantages of each model are presented via synthetic signals and then the application on real signals is discussed. Conclusively, eaQHM outperforms EDS in medium-to-large window size analysis, whereas EDSM yields higher reconstruction values for smaller analysis window sizes. Thus, a future research direction appears to be the merge of adaptivity of the eaQHM and parameter estimation robustness of the EDSM in a new paradigm for high-quality analysis and resynthesis of general audio signals. 

---
# Predicting Tuberculosis from Real-World Cough Audio Recordings and Metadata 

**Authors**: George P. Kafentzis, Stephane Tetsing, Joe Brew, Lola Jover, Mindaugas Galvosas, Carlos Chaccour, Peter M. Small  

**Link**: [PDF](https://arxiv.org/pdf/2307.04842)  

**Abstract**: Tuberculosis (TB) is an infectious disease caused by the bacterium Mycobacterium tuberculosis and primarily affects the lungs, as well as other body parts. TB is spread through the air when an infected person coughs, sneezes, or talks. Medical doctors diagnose TB in patients via clinical examinations and specialized tests. However, coughing is a common symptom of respiratory diseases such as TB. Literature suggests that cough sounds coming from different respiratory diseases can be distinguished by both medical doctors and computer algorithms. Therefore, cough recordings associated with patients with and without TB seems to be a reasonable avenue of investigation. In this work, we utilize a very large dataset of TB and non-TB cough audio recordings obtained from the south-east of Africa, India, and the south-east of Asia using a fully automated phone-based application (Hyfe), without manual annotation. We fit statistical classifiers based on spectral and time domain features with and without clinical metadata. A stratified grouped cross-validation approach shows that an average Area Under Curve (AUC) of approximately 0.70 $\pm$ 0.05 both for a cough-level and a participant-level classification can be achieved using cough sounds alone. The addition of demographic and clinical factors increases performance, resulting in an average AUC of approximately 0.81 $\pm$ 0.05. Our results suggest mobile phone-based applications that integrate clinical symptoms and cough sound analysis could help community health workers and, most importantly, health service programs to improve TB case-finding efforts while reducing costs, which could substantially improve public health. 

---
