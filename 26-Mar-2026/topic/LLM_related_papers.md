# Language-Grounded Multi-Agent Planning for Personalized and Fair Participatory Urban Sensing 

**Authors**: Xusen Guo, Mingxing Peng, Hongliang Lu, Hai Yang, Jun Ma, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24014)  

**Abstract**: Participatory urban sensing leverages human mobility for large-scale urban data collection, yet existing methods typically rely on centralized optimization and assume homogeneous participants, resulting in rigid assignments that overlook personal preferences and heterogeneous urban contexts. We propose MAPUS, an LLM-based multi-agent framework for personalized and fair participatory urban sensing. In our framework, participants are modeled as autonomous agents with individual profiles and schedules, while a coordinator agent performs fairness-aware selection and refines sensing routes through language-based negotiation. Experiments on real-world datasets show that MAPUS achieves competitive sensing coverage while substantially improving participant satisfaction and fairness, promoting more human-centric and sustainable urban sensing systems. 

---
# Enhanced Mycelium of Thought (EMoT): A Bio-Inspired Hierarchical Reasoning Architecture with Strategic Dormancy and Mnemonic Encoding 

**Authors**: Florian Odi Stummer  

**Link**: [PDF](https://arxiv.org/pdf/2603.24065)  

**Abstract**: Current prompting paradigms for large language models (LLMs), including Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT), follow linear or tree-structured reasoning paths that lack persistent memory, strategic dormancy, and cross-domain synthesis. We present the Enhanced Mycelium of Thought (EMoT) framework, a bio-inspired reasoning architecture that organises cognitive processing into a four-level hierarchy (Micro, Meso, Macro, Meta), implements strategic dormancy and reactivation of reasoning nodes, and integrates a Memory Palace with five mnemonic encoding styles. EMoT is a research prototype for complex, multi-domain problems, not a general-purpose prompting enhancement. Two complementary evaluations reveal a characteristic trade-off. In a blind LLM-as-Judge evaluation across three domains, EMoT achieved near-parity with CoT (4.20 vs. 4.33/5.0) with higher stability, and outperformed CoT on Cross-Domain Synthesis (4.8 vs. 4.4). Ablation studies show that strategic dormancy is architecturally essential (quality collapsed from 4.2 to 1.0 when disabled). On a 15-item short-answer benchmark, EMoT (27%) substantially underperformed simpler baselines, confirming systematic overthinking on simple problems. These results are subject to important limitations: small sample sizes (n=3 complex cases, n=15 short-answer items), LLM-as-Judge evaluation with potential self-preference bias, and approximately 33-fold computational cost overhead. To our knowledge, EMoT is the first reasoning framework to combine hierarchical topology, strategic thought dormancy with reactivation, and mnemonic memory encoding in a single architecture. 

---
# DUPLEX: Agentic Dual-System Planning via LLM-Driven Information Extraction 

**Authors**: Keru Hua, Ding Wang, Yaoying Gu, Xiaoguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.23909)  

**Abstract**: While Large Language Models (LLMs) provide semantic flexibility for robotic task planning, their susceptibility to hallucination and logical inconsistency limits their reliability in long-horizon domains. To bridge the gap between unstructured environments and rigorous plan synthesis, we propose DUPLEX, an agentic dual-system neuro-symbolic architecture that strictly confines the LLM to schema-guided information extraction rather than end-to-end planning or code generation. In our framework, a feed-forward Fast System utilizes a lightweight LLM to extract entities, relations etc. from natural language, deterministically mapping them into a Planning Domain Definition Language (PDDL) problem file for a classical symbolic planner. To resolve complex or underspecified scenarios, a Slow System is activated exclusively upon planning failure, leveraging solver diagnostics to drive a high-capacity LLM in iterative reflection and repair. Extensive evaluations across 12 classical and household planning domains demonstrate that DUPLEX significantly outperforms existing end-to-end and hybrid LLM baselines in both success rate and reliability. These results confirm that The key is not to make the LLM plan better, but to restrict the LLM to the part it is good at - structured semantic grounding - and leave logical plan synthesis to a symbolic planner. 

---
# PLDR-LLMs Reason At Self-Organized Criticality 

**Authors**: Burc Gokden  

**Link**: [PDF](https://arxiv.org/pdf/2603.23539)  

**Abstract**: We show that PLDR-LLMs pretrained at self-organized criticality exhibit reasoning at inference time. The characteristics of PLDR-LLM deductive outputs at criticality is similar to second-order phase transitions. At criticality, the correlation length diverges, and the deductive outputs attain a metastable steady state. The steady state behaviour suggests that deductive outputs learn representations equivalent to scaling functions, universality classes and renormalization groups from the training dataset, leading to generalization and reasoning capabilities in the process. We can then define an order parameter from the global statistics of the model's deductive output parameters at inference. The reasoning capabilities of a PLDR-LLM is better when its order parameter is close to zero at criticality. This observation is supported by the benchmark scores of the models trained at near-criticality and sub-criticality. Our results provide a self-contained explanation on how reasoning manifests in large language models, and the ability to reason can be quantified solely from global model parameter values of the deductive outputs at steady state, without any need for evaluation of curated benchmark datasets through inductive output for reasoning and comprehension. 

---
# Can LLM Agents Be CFOs? A Benchmark for Resource Allocation in Dynamic Enterprise Environments 

**Authors**: Yi Han, Lingfei Qian, Yan Wang, Yueru He, Xueqing Peng, Dongji Feng, Yankai Chen, Haohang Li, Yupeng Cao, Jimin Huang, Xue Liu, Jian-Yun Nie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23638)  

**Abstract**: Large language models (LLMs) have enabled agentic systems that can reason, plan, and act across complex tasks, but it remains unclear whether they can allocate resources effectively under uncertainty. Unlike short-horizon reactive decisions, allocation requires committing scarce resources over time while balancing competing objectives and preserving flexibility for future needs. We introduce EnterpriseArena, the first benchmark for evaluating agents on long-horizon enterprise resource allocation. It instantiates CFO-style decision-making in a 132-month enterprise simulator combining firm-level financial data, anonymized business documents, macroeconomic and industry signals, and expert-validated operating rules. The environment is partially observable and reveals the state only through budgeted organizational tools, forcing agents to trade off information acquisition against conserving scarce resources. Experiments on eleven advanced LLMs show that this setting remains highly challenging: only 16% of runs survive the full horizon, and larger models do not reliably outperform smaller ones. These results identify long-horizon resource allocation under uncertainty as a distinct capability gap for current LLM agents. 

---
# LLMs Do Not Grade Essays Like Humans 

**Authors**: Jerin George Mathew, Sumayya Taher, Anindita Kundu, Denilson Barbosa  

**Link**: [PDF](https://arxiv.org/pdf/2603.23714)  

**Abstract**: Large language models have recently been proposed as tools for automated essay scoring, but their agreement with human grading remains unclear. In this work, we evaluate how LLM-generated scores compare with human grades and analyze the grading behavior of several models from the GPT and Llama families in an out-of-the-box setting, without task-specific training. Our results show that agreement between LLM and human scores remains relatively weak and varies with essay characteristics. In particular, compared to human raters, LLMs tend to assign higher scores to short or underdeveloped essays, while assigning lower scores to longer essays that contain minor grammatical or spelling errors. We also find that the scores generated by LLMs are generally consistent with the feedback they generate: essays receiving more praise tend to receive higher scores, while essays receiving more criticism tend to receive lower scores. These results suggest that LLM-generated scores and feedback follow coherent patterns but rely on signals that differ from those used by human raters, resulting in limited alignment with human grading practices. Nevertheless, our work shows that LLMs produce feedback that is consistent with their grading and that they can be reliably used in supporting essay scoring. 

---
# AnalogAgent: Self-Improving Analog Circuit Design Automation with LLM Agents 

**Authors**: Zhixuan Bao, Zhuoyi Lin, Jiageng Wang, Jinhai Hu, Yuan Gao, Yaoxin Wu, Xiaoli Li, Xun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23910)  

**Abstract**: Recent advances in large language models (LLMs) suggest strong potential for automating analog circuit design. Yet most LLM-based approaches rely on a single-model loop of generation, diagnosis, and correction, which favors succinct summaries over domain-specific insight and suffers from context attrition that erases critical technical details. To address these limitations, we propose AnalogAgent, a training-free agentic framework that integrates an LLM-based multi-agent system (MAS) with self-evolving memory (SEM) for analog circuit design automation. AnalogAgent coordinates a Code Generator, Design Optimizer, and Knowledge Curator to distill execution feedback into an adaptive playbook in SEM and retrieve targeted guidance for subsequent generation, enabling cross-task transfer without additional expert feedback, databases, or libraries. Across established benchmarks, AnalogAgent achieves 92% Pass@1 with Gemini and 97.4% Pass@1 with GPT-5. Moreover, with compact models (e.g., Qwen-8B), it yields a +48.8% average Pass@1 gain across tasks and reaches 72.1% Pass@1 overall, indicating that AnalogAgent substantially strengthens open-weight models for high-quality analog circuit design automation. 

---
# LensWalk: Agentic Video Understanding by Planning How You See in Videos 

**Authors**: Keliang Li, Yansong Li, Hongze Shen, Mengdi Liu, Hong Chang, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2603.24558)  

**Abstract**: The dense, temporal nature of video presents a profound challenge for automated analysis. Despite the use of powerful Vision-Language Models, prevailing methods for video understanding are limited by the inherent disconnect between reasoning and perception: they rely on static, pre-processed information and cannot actively seek raw evidence from video as their understanding evolves. To address this, we introduce LensWalk, a flexible agentic framework that empowers a Large Language Model reasoner to control its own visual observation actively. LensWalk establishes a tight reason-plan-observe loop where the agent dynamically specifies, at each step, the temporal scope and sampling density of the video it observes. Using a suite of versatile, Vision-Language Model based tools parameterized by these specifications, the agent can perform broad scans for cues, focus on specific segments for fact extraction, and stitch evidence from multiple moments for holistic verification. This design allows for progressive, on-demand evidence gathering that directly serves the agent's evolving chain of thought. Without requiring any model fine-tuning, LensWalk delivers substantial, plug-and-play performance gains on multiple model recipes, boosting their accuracy by over 5\% on challenging long-video benchmarks like LVBench and Video-MME. Our analysis reveals that enabling an agent to control how it sees is key to unlocking more accurate, robust, and interpretable video reasoning. 

---
# MolEvolve: LLM-Guided Evolutionary Search for Interpretable Molecular Optimization 

**Authors**: Xiangsen Chen, Ruilong Wu, Yanyan Lan, Ting Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24382)  

**Abstract**: Despite deep learning's success in chemistry, its impact is hindered by a lack of interpretability and an inability to resolve activity cliffs, where minor structural nuances trigger drastic property shifts. Current representation learning, bound by the similarity principle, often fails to capture these structural-activity discontinuities. To address this, we introduce MolEvolve, an evolutionary framework that reformulates molecular discovery as an autonomous, look-ahead planning problem. Unlike traditional methods that depend on human-engineered features or rigid prior knowledge, MolEvolve leverages a Large Language Model (LLM) to actively explore and evolve a library of executable chemical symbolic operations. By utilizing the LLM to cold start and an Monte Carlo Tree Search (MCTS) engine for test-time planning with external tools (e.g. RDKit), the system self-discovers optimal trajectories autonomously. This process evolves transparent reasoning chains that translate complex structural transformations into actionable, human-readable chemical insights. Experimental results demonstrate that MolEvolve's autonomous search not only evolves transparent, human-readable chemical insights, but also outperforms baselines in both property prediction and molecule optimization tasks. 

---
# When AI Meets Early Childhood Education: Large Language Models as Assessment Teammates in Chinese Preschools 

**Authors**: Xingming Li, Runke Huang, Yanan Bao, Yuye Jin, Yuru Jiao, Qingyong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24389)  

**Abstract**: High-quality teacher-child interaction (TCI) is fundamental to early childhood development, yet traditional expert-based assessment faces a critical scalability challenge. In large systems like China's-serving 36 million children across 250,000+ kindergartens-the cost and time requirements of manual observation make continuous quality monitoring infeasible, relegating assessment to infrequent episodic audits that limit timely intervention and improvement tracking.
In this paper, we investigate whether AI can serve as a scalable assessment teammate by extracting structured quality indicators and validating their alignment with human expert judgments.
Our contributions include: (1) TEPE-TCI-370h (Tracing Effective Preschool Education), the first large-scale dataset of naturalistic teacher-child interactions in Chinese preschools (370 hours, 105 classrooms) with standardized ECQRS-EC and SSTEW annotations; (2) We develop Interaction2Eval, a specialized LLM-based framework addressing domain-specific challenges-child speech recognition, Mandarin homophone disambiguation, and rubric-based reasoning-achieving up to 88% agreement; (3) Deployment validation across 43 classrooms demonstrating an 18x efficiency gain in the assessment workflow, highlighting its potential for shifting from annual expert audits to monthly AI-assisted monitoring with targeted human oversight. This work not only demonstrates the technical feasibility of scalable, AI-augmented quality assessment but also lays the foundation for a new paradigm in early childhood education-one where continuous, inclusive, AI-assisted evaluation becomes the engine of systemic improvement and equitable growth. 

---
# Towards Effective Experiential Learning: Dual Guidance for Utilization and Internalization 

**Authors**: Fei Bai, Zhipeng Chen, Chuan Hao, Ming Yang, Ran Tao, Bryan Dai, Wayne Xin Zhao, Jian Yang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24093)  

**Abstract**: Recently, reinforcement learning~(RL) has become an important approach for improving the capabilities of large language models~(LLMs). In particular, reinforcement learning from verifiable rewards~(RLVR) has emerged as a promising paradigm for reasoning tasks. However, existing RL-based training still remains only a rough approximation to human learning. Human learners leverage both external and internal experience to guide exploration and gradually internalize useful trajectories into stable knowledge. Motivated by this gap, we ask: how can LLMs better utilize and internalize experience during RLVR training? To answer this question, we propose \textbf{D}ual \textbf{G}uidance \textbf{O}ptimization~(\textbf{DGO}), a unified framework that leverages \emph{external} and \emph{internal experience} to improve training effectiveness. Specifically, DGO first constructs an experience bank from previously explored trajectories. The policy then performs exploration under the joint guidance of the experience bank and the model's internal knowledge. The resulting trajectories are further used to refine the experience bank and optimize model parameters, forming a closed loop of experience utilization and internalization. Experiments show that DGO consistently outperforms baseline methods, suggesting that better utilization and internalization of experience lead to more effective reasoning. 

---
# MedAidDialog: A Multilingual Multi-Turn Medical Dialogue Dataset for Accessible Healthcare 

**Authors**: Shubham Kumar Nigam, Suparnojit Sarkar, Piyush Patel  

**Link**: [PDF](https://arxiv.org/pdf/2603.24132)  

**Abstract**: Conversational artificial intelligence has the potential to assist users in preliminary medical consultations, particularly in settings where access to healthcare professionals is limited. However, many existing medical dialogue systems operate in a single-turn question--answering paradigm or rely on template-based datasets, limiting conversational realism and multilingual applicability. In this work, we introduce MedAidDialog, a multilingual multi-turn medical dialogue dataset designed to simulate realistic physician--patient consultations. The dataset extends the MDDial corpus by generating synthetic consultations using large language models and further expands them into a parallel multilingual corpus covering seven languages: English, Hindi, Telugu, Tamil, Bengali, Marathi, and Arabic.
Building on this dataset, we develop MedAidLM, a conversational medical model trained using parameter-efficient fine-tuning on quantized small language models, enabling deployment without high-end computational infrastructure. Our framework additionally incorporates optional patient pre-context information (e.g., age, gender, allergies) to personalize the consultation process. Experimental results demonstrate that the proposed system can effectively perform symptom elicitation through multi-turn dialogue and generate diagnostic recommendations. We further conduct medical expert evaluation to assess the plausibility and coherence of the generated consultations. 

---
# Policy-Guided Threat Hunting: An LLM enabled Framework with Splunk SOC Triage 

**Authors**: Rishikesh Sahay, Bell Eapen, Weizhi Meng, Md Rasel Al Mamun, Nikhil Kumar Dora, Manjusha Sumasadan, Sumit Kumar Tetarave, Rod Soto  

**Link**: [PDF](https://arxiv.org/pdf/2603.23966)  

**Abstract**: With frequently evolving Advanced Persistent Threats (APTs) in cyberspace, traditional security solutions approaches have become inadequate for threat hunting for organizations. Moreover, SOC (Security Operation Centers) analysts are often overwhelmed and struggle to analyze the huge volume of logs received from diverse devices in organizations. To address these challenges, we propose an automated and dynamic threat hunting framework for monitoring evolving threats, adapting to changing network conditions, and performing risk-based prioritization for the mitigation of suspicious and malicious traffic. By integrating Agentic AI with Splunk, an established SIEM platform, we developed a unique threat hunting framework. The framework systematically and seamlessly integrates different threat hunting modules together, ranging from traffic ingestion to anomaly assessment using a reconstruction-based autoencoder, deep reinforcement learning (DRL) with two layers for initial triage, and a large language model (LLM) for contextual analysis. We evaluated the framework against a publicly available benchmark dataset, as well as against a simulated dataset. The experimental results show that the framework can effectively adapt to different SOC objectives autonomously and identify suspicious and malicious traffic. The framework enhances operational effectiveness by supporting SOC analysts in their decision-making to block, allow, or monitor network traffic. This study thus enhances cybersecurity and threat hunting literature by presenting the novel threat hunting framework for security decision- making, as well as promoting cumulative research efforts to develop more effective frameworks to battle continuously evolving cyber threats. 

---
# Large Language Model Guided Incentive Aware Reward Design for Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Dogan Urgun, Gokhan Gungor  

**Link**: [PDF](https://arxiv.org/pdf/2603.24324)  

**Abstract**: Designing effective auxiliary rewards for cooperative multi-agent systems remains a precarious task; misaligned incentives risk inducing suboptimal coordination, especially where sparse task feedback fails to provide sufficient grounding. This study introduces an automated reward design framework that leverages large language models to synthesize executable reward programs from environment instrumentation. The procedure constrains candidate programs within a formal validity envelope and evaluates their efficacy by training policies from scratch under a fixed computational budget; selection depends exclusively on the sparse task return. The framework is evaluated across four distinct Overcooked-AI layouts characterized by varied corridor congestion, handoff dependencies, and structural asymmetries. Iterative search generations consistently yield superior task returns and delivery counts, with the most pronounced gains occurring in environments dominated by interaction bottlenecks. Diagnostic analysis of the synthesized shaping components indicates increased interdependence in action selection and improved signal alignment in coordination-intensive tasks. These results demonstrate that the search for objectivegrounded reward programs can mitigate the burden of manual engineering while producing shaping signals compatible with cooperative learning under finite budgets. 

---
# Invisible Threats from Model Context Protocol: Generating Stealthy Injection Payload via Tree-based Adaptive Search 

**Authors**: Yulin Shen, Xudong Pan, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24203)  

**Abstract**: Recent advances in the Model Context Protocol (MCP) have enabled large language models (LLMs) to invoke external tools with unprecedented ease. This creates a new class of powerful and tool augmented agents. Unfortunately, this capability also introduces an under explored attack surface, specifically the malicious manipulation of tool responses. Existing techniques for indirect prompt injection that target MCP suffer from high deployment costs, weak semantic coherence, or heavy white box requirements. Furthermore, they are often easily detected by recently proposed defenses. In this paper, we propose Tree structured Injection for Payloads (TIP), a novel black-box attack which generates natural payloads to reliably seize control of MCP enabled agents even under defense. Technically, We cast payload generation as a tree structured search problem and guide the search with an attacker LLM operating under our proposed coarse-to-fine optimization framework. To stabilize learning and avoid local optima, we introduce a path-aware feedback mechanism that surfaces only high quality historical trajectories to the attacker model. The framework is further hardened against defensive transformations by explicitly conditioning the search on observable defense signals and dynamically reallocating the exploration budget. Extensive experiments on four mainstream LLMs show that TIP attains over 95% attack success in undefended settings while requiring an order of magnitude fewer queries than prior adaptive attacks. Against four representative defense approaches, TIP preserves more than 50% effectiveness and significantly outperforms the state-of-the-art attacks. By implementing the attack on real world MCP systems, our results expose an invisible but practical threat vector in MCP deployments. We also discuss potential mitigation approaches to address this critical security gap. 

---
# PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay 

**Authors**: Rohan Khetan, Ashna Khetan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23841)  

**Abstract**: While Large Language Models (LLMs) are increasingly used as primary sources of information, their potential for political bias may impact their objectivity. Existing benchmarks of LLM social bias primarily evaluate gender and racial stereotypes. When political bias is included, it is typically measured at a coarse level, neglecting the specific values that shape sociopolitical leanings. This study investigates political bias in eight prominent LLMs (Claude, Deepseek, Gemini, GPT, Grok, Llama, Qwen Base, Qwen Instruction-Tuned) using PoliticsBench: a novel multi-turn roleplay framework adapted from the EQ-Bench-v3 psychometric benchmark. We test whether commercially developed LLMs display a systematic left-leaning bias that becomes more pronounced in later stages of multi-stage roleplay. Through twenty evolving scenarios, each model reported its stance and determined its course of action. Scoring these responses on a scale of ten political values, we explored the values underlying chatbots' deviations from unbiased standards. Seven of our eight models leaned left, while Grok leaned right. Each left-leaning LLM strongly exhibited liberal traits and moderately exhibited conservative ones. We discovered slight variations in alignment scores across stages of roleplay, with no particular pattern. Though most models used consequence-based reasoning, Grok frequently argued with facts and statistics. Our study presents the first psychometric evaluation of political values in LLMs through multi-stage, free-text interactions. 

---
# Object Search in Partially-Known Environments via LLM-informed Model-based Planning and Prompt Selection 

**Authors**: Abhishek Paudel, Abhish Khanal, Raihan I. Arnob, Shahriar Hossain, Gregory J. Stein  

**Link**: [PDF](https://arxiv.org/pdf/2603.23800)  

**Abstract**: We present a novel LLM-informed model-based planning framework, and a novel prompt selection method, for object search in partially-known environments. Our approach uses an LLM to estimate statistics about the likelihood of finding the target object when searching various locations throughout the scene that, combined with travel costs extracted from the environment map, are used to instantiate a model, thus using the LLM to inform planning and achieve effective search performance. Moreover, the abstraction upon which our approach relies is amenable to deployment-time model selection via the recent offline replay approach, an insight we leverage to enable fast prompt and LLM selection during deployment. Simulation experiments demonstrate that our LLM-informed model-based planning approach outperforms the baseline planning strategy that fully relies on LLM and optimistic strategy with as much as 11.8% and 39.2% improvements respectively, and our bandit-like selection approach enables quick selection of best prompts and LLMs resulting in 6.5% lower average cost and 33.8% lower average cumulative regret over baseline UCB bandit selection. Real-robot experiments in an apartment demonstrate similar improvements and so further validate our approach. 

---
# The Cognitive Firewall:Securing Browser Based AI Agents Against Indirect Prompt Injection Via Hybrid Edge Cloud Defense 

**Authors**: Qianlong Lan, Anuj Kaul  

**Link**: [PDF](https://arxiv.org/pdf/2603.23791)  

**Abstract**: Deploying large language models (LLMs) as autonomous browser agents exposes a significant attack surface in the form of Indirect Prompt Injection (IPI). Cloud-based defenses can provide strong semantic analysis, but they introduce latency and raise privacy concerns. We present the Cognitive Firewall, a three-stage split-compute architecture that distributes security checks across the client and the cloud. The system consists of a local visual Sentinel, a cloud-based Deep Planner, and a deterministic Guard that enforces execution-time policies. Across 1,000 adversarial samples, edge-only defenses fail to detect 86.9% of semantic attacks. In contrast, the full hybrid architecture reduces the overall attack success rate (ASR) to below 1% (0.88% under static evaluation and 0.67% under adaptive evaluation), while maintaining deterministic constraints on side-effecting actions. By filtering presentation-layer attacks locally, the system avoids unnecessary cloud inference and achieves an approximately 17,000x latency advantage over cloud-only baselines. These results indicate that deterministic enforcement at the execution boundary can complement probabilistic language models, and that split-compute provides a practical foundation for securing interactive LLM agents. 

---
# AI-driven Intent-Based Networking Approach for Self-configuration of Next Generation Networks 

**Authors**: Md. Kamrul Hossain, Walid Aljoby  

**Link**: [PDF](https://arxiv.org/pdf/2603.23772)  

**Abstract**: Intent-Based Networking (IBN) aims to simplify operating heterogeneous infrastructures by translating high-level intents into enforceable policies and assuring compliance. However, dependable automation remains difficult because (i) realizing intents from ambiguous natural language into controller-ready policies is brittle and prone to conflicts and unintended side effects, and (ii) assurance is often reactive and struggles in multi-intent settings where faults create cascading symptoms and ambiguous telemetry. This paper proposes an end-to-end closed-loop IBN pipeline that uses large language models with structured validation for natural language to policy realization and conflict-aware activation, and reformulates assurance as proactive multi-intent failure prediction with root-cause disambiguation. The expected outcome is operator-trustworthy automation that provides actionable early warnings, interpretable explanations, and measurable lead time for remediation. 

---
# Assessment Design in the AI Era: A Method for Identifying Items Functioning Differentially for Humans and Chatbots 

**Authors**: Licol Zeinfeld, Alona Strugatski, Ziva Bar-Dov, Ron Blonder, Shelley Rap, Giora Alexandron  

**Link**: [PDF](https://arxiv.org/pdf/2603.23682)  

**Abstract**: The rapid adoption of large language models (LLMs) in education raises profound challenges for assessment design. To adapt assessments to the presence of LLM-based tools, it is crucial to characterize the strengths and weaknesses of LLMs in a generalizable, valid and reliable manner. However, current LLM evaluations often rely on descriptive statistics derived from benchmarks, and little research applies theory-grounded measurement methods to characterize LLM capabilities relative to human learners in ways that directly support assessment design. Here, by combining educational data mining and psychometric theory, we introduce a statistically principled approach for identifying items on which humans and LLMs show systematic response differences, pinpointing where assessments may be most vulnerable to AI misuse, and which task dimensions make problems particularly easy or difficult for generative AI. The method is based on Differential Item Functioning (DIF) analysis -- traditionally used to detect bias across demographic groups -- together with negative control analysis and item-total correlation discrimination analysis. It is evaluated on responses from human learners and six leading chatbots (ChatGPT-4o \& 5.2, Gemini 1.5 \& 3 Pro, Claude 3.5 \& 4.5 Sonnet) to two instruments: a high school chemistry diagnostic test and a university entrance exam. Subject-matter experts then analyzed DIF-flagged items to characterize task dimensions associated with chatbot over- or under-performance. Results show that DIF-informed analytics provide a robust framework for understanding where LLM and human capabilities diverge, and highlight their value for improving the design of valid, reliable, and fair assessment in the AI era. 

---
# PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation 

**Authors**: Manjushree B. Aithal, Ph.D., Alexander Kotz, James Mitchell, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2603.23678)  

**Abstract**: Large Language Models (LLMs) offer transformative solutions across many domains, but healthcare integration is hindered by strict data privacy constraints. Clinical narratives are dense with ambiguous acronyms, misinterpretation these abbreviations can precipitate severe outcomes like life-threatening medication errors. While cloud-dependent LLMs excel at Acronym Disambiguation, transmitting Protected Health Information to external servers violates privacy frameworks. To bridge this gap, this study pioneers the evaluation of small-parameter models deployed entirely on-device to ensure privacy preservation. We introduce a privacy-preserving cascaded pipeline leveraging general-purpose local models to detect clinical acronyms, routing them to domain-specific biomedical models for context-relevant expansions. Results reveal that while general instruction-following models achieve high detection accuracy (~0.988), their expansion capabilities plummet (~0.655). Our cascaded approach utilizes domain-specific medical models to increase expansion accuracy to (~0.81). This novel work demonstrates that privacy-preserving, on-device (2B-10B) models deliver high-fidelity clinical acronym disambiguation support. 

---
# LLMLOOP: Improving LLM-Generated Code and Tests through Automated Iterative Feedback Loops 

**Authors**: Ravin Ravi, Dylan Bradshaw, Stefano Ruberto, Gunel Jahangirova, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2603.23613)  

**Abstract**: Large Language Models (LLMs) are showing remarkable performance in generating source code, yet the generated code often has issues like compilation errors or incorrect code. Researchers and developers often face wasted effort in implementing checks and refining LLM-generated code, frequently duplicating their efforts. This paper presents LLMLOOP, a framework that automates the refinement of both source code and test cases produced by LLMs. LLMLOOP employs five iterative loops: resolving compilation errors, addressing static analysis issues, fixing test case failures, and improving test quality through mutation analysis. These loops ensure the generation of high-quality test cases that serve as both a validation mechanism and a regression test suite for the generated code. We evaluated LLMLOOP on HUMANEVAL-X, a recent benchmark of programming tasks. Results demonstrate the tool's effectiveness in refining LLM-generated outputs. 

---
# Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges 

**Authors**: Weilun Xu, Alexander Rusnak, Frederic Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23659)  

**Abstract**: When large language models make ethical judgments, do their internal representations distinguish between normative frameworks, or collapse ethics into a single acceptability dimension? We probe hidden representations across five ethical frameworks (deontology, utilitarianism, virtue, justice, commonsense) in six LLMs spanning 4B--72B parameters. Our analysis reveals differentiated ethical subspaces with asymmetric transfer patterns -- e.g., deontology probes partially generalize to virtue scenarios while commonsense probes fail catastrophically on justice. Disagreement between deontological and utilitarian probes correlates with higher behavioral entropy across architectures, though this relationship may partly reflect shared sensitivity to scenario difficulty. Post-hoc validation reveals that probes partially depend on surface features of benchmark templates, motivating cautious interpretation. We discuss both the structural insights these methods provide and their epistemological limitations. 

---
# LLMORPH: Automated Metamorphic Testing of Large Language Models 

**Authors**: Steven Cho, Stefano Ruberto, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2603.23611)  

**Abstract**: Automated testing is essential for evaluating and improving the reliability of Large Language Models (LLMs), yet the lack of automated oracles for verifying output correctness remains a key challenge. We present LLMORPH, an automated testing tool specifically designed for LLMs performing NLP tasks, which leverages Metamorphic Testing (MT) to uncover faulty behaviors without relying on human-labeled data. MT uses Metamorphic Relations (MRs) to generate follow-up inputs from source test input, enabling detection of inconsistencies in model outputs without the need of expensive labelled data. LLMORPH is aimed at researchers and developers who want to evaluate the robustness of LLM-based NLP systems. In this paper, we detail the design, implementation, and practical usage of LLMORPH, demonstrating how it can be easily extended to any LLM, NLP task, and set of MRs. In our evaluation, we applied 36 MRs across four NLP benchmarks, testing three state-of-the-art LLMs: GPT-4, LLAMA3, and HERMES 2. This produced over 561,000 test executions. Results demonstrate LLMORPH's effectiveness in automatically exposing inconsistencies. 

---
# A Theory of LLM Information Susceptibility 

**Authors**: Zhuo-Yang Song, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23626)  

**Abstract**: Large language models (LLMs) are increasingly deployed as optimization modules in agentic systems, yet the fundamental limits of such LLM-mediated improvement remain poorly understood. Here we propose a theory of LLM information susceptibility, centred on the hypothesis that when computational resources are sufficiently large, the intervention of a fixed LLM does not increase the performance susceptibility of a strategy set with respect to budget. We develop a multi-variable utility-function framework that generalizes this hypothesis to architectures with multiple co-varying budget channels, and discuss the conditions under which co-scaling can exceed the susceptibility bound. We validate the theory empirically across structurally diverse domains and model scales spanning an order of magnitude, and show that nested, co-scaling architectures open response channels unavailable to fixed configurations. These results clarify when LLM intervention helps and when it does not, demonstrating that tools from statistical physics can provide predictive constraints for the design of AI systems. If the susceptibility hypothesis holds generally, the theory suggests that nested architectures may be a necessary structural condition for open-ended agentic self-improvement. 

---
# APreQEL: Adaptive Mixed Precision Quantization For Edge LLMs 

**Authors**: Meriem Bouzouad, Yuan-Hao Chang, Jalil Boukhobza  

**Link**: [PDF](https://arxiv.org/pdf/2603.23575)  

**Abstract**: Today, large language models have demonstrated their strengths in various tasks ranging from reasoning, code generation, and complex problem solving. However, this advancement comes with a high computational cost and memory requirements, making it challenging to deploy these models on edge devices to ensure real-time responses and data privacy. Quantization is one common approach to reducing memory use, but most methods apply it uniformly across all layers. This does not account for the fact that different layers may respond differently to reduced precision. Importantly, memory consumption and computational throughput are not necessarily aligned, further complicating deployment decisions. This paper proposes an adaptive mixed precision quantization mechanism that balances memory, latency, and accuracy in edge deployment under user-defined priorities. This is achieved by analyzing the layer-wise contribution and by inferring how different quantization types behave across the target hardware platform in order to assign the most suitable quantization type to each layer. This integration ensures that layer importance and the overall performance trade-offs are jointly respected in this design. Our work unlocks new configuration designs that uniform quantization cannot achieve, expanding the solution space to efficiently deploy the LLMs on resource-constrained devices. 

---
# Mixture of Demonstrations for Textual Graph Understanding and Question Answering 

**Authors**: Yukun Wu, Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23554)  

**Abstract**: Textual graph-based retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) in domain-specific question answering. While existing approaches primarily focus on zero-shot GraphRAG, selecting high-quality demonstrations is crucial for improving reasoning and answer accuracy. Furthermore, recent studies have shown that retrieved subgraphs often contain irrelevant information, which can degrade reasoning performance. In this paper, we propose MixDemo, a novel GraphRAG framework enhanced with a Mixture-of-Experts (MoE) mechanism for selecting the most informative demonstrations under diverse question contexts. To further reduce noise in the retrieved subgraphs, we introduce a query-specific graph encoder that selectively attends to information most relevant to the query. Extensive experiments across multiple textual graph benchmarks show that MixDemo significantly outperforms existing methods. 

---
# MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG 

**Authors**: Bhavik Mangla  

**Link**: [PDF](https://arxiv.org/pdf/2603.23533)  

**Abstract**: RAG pipelines typically rely on fixed-size chunking, which ignores document structure, fragments semantic units across boundaries, and requires multiple LLM calls per chunk for metadata extraction. We present MDKeyChunker, a three-stage pipeline for Markdown documents that (1) performs structure-aware chunking treating headers, code blocks, tables, and lists as atomic units; (2) enriches each chunk via a single LLM call extracting title, summary, keywords, typed entities, hypothetical questions, and a semantic key, while propagating a rolling key dictionary to maintain document-level context; and (3) restructures chunks by merging those sharing the same semantic key via bin-packing, co-locating related content for retrieval. The single-call design extracts all seven metadata fields in one LLM invocation, eliminating the need for separate per-field extraction passes. Rolling key propagation replaces hand-tuned scoring with LLM-native semantic matching. An empirical evaluation on 30 queries over an 18-document Markdown corpus shows Config D (BM25 over structural chunks) achieves Recall@5=1.000 and MRR=0.911, while dense retrieval over the full pipeline (Config C) reaches Recall@5=0.867. MDKeyChunker is implemented in Python with four dependencies and supports any OpenAI-compatible endpoint. 

---
# Qworld: Question-Specific Evaluation Criteria for LLMs 

**Authors**: Shanghua Gao, Yuchang Su, Pengwei Sui, Curtis Ginder, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2603.23522)  

**Abstract**: Evaluating large language models (LLMs) on open-ended questions is difficult because response quality depends on the question's context. Binary scores and static rubrics fail to capture these context-dependent requirements. Existing methods define criteria at the dataset level or generate them in a single pass, which limits their ability to explore the evaluation space implied by each question. We introduce One-Question-One-World (Qworld), a method that generates question-specific evaluation criteria using a recursive expansion tree. Given a question, Qworld decomposes it into scenarios, perspectives, and fine-grained binary criteria through structured hierarchical and horizontal expansion. The resulting criteria specify what a high-quality answer must address for that question. On HealthBench, Qworld covers 89% of expert-authored criteria and generates 79% novel criteria validated by human experts. Experts rate Qworld criteria higher in insight and granularity than those produced by prior methods. When applied to 11 frontier LLMs on HealthBench and Humanity's Last Exam, Qworld reveals capability differences in dimensions such as long-term impact, equity, error handling, and interdisciplinary reasoning that coarse rubrics do not distinguish. By formulating criteria generation as structured coverage of question-implied evaluation axes, Qworld enables evaluation that adapts to each question rather than relying on fixed task-level criteria. 

---
# Understanding the Challenges in Iterative Generative Optimization with LLMs 

**Authors**: Allen Nie, Xavier Daull, Zhiyi Kuang, Abhinav Akkiraju, Anish Chaudhuri, Max Piasevoli, Ryan Rong, YuCheng Yuan, Prerit Choudhary, Shannon Xiao, Rasool Fakoor, Adith Swaminathan, Ching-An Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.23994)  

**Abstract**: Generative optimization uses large language models (LLMs) to iteratively improve artifacts (such as code, workflows or prompts) using execution feedback. It is a promising approach to building self-improving agents, yet in practice remains brittle: despite active research, only 9% of surveyed agents used any automated optimization. We argue that this brittleness arises because, to set up a learning loop, an engineer must make ``hidden'' design choices: What can the optimizer edit and what is the "right" learning evidence to provide at each update? We investigate three factors that affect most applications: the starting artifact, the credit horizon for execution traces, and batching trials and errors into learning evidence. Through case studies in MLAgentBench, Atari, and BigBench Extra Hard, we find that these design decisions can determine whether generative optimization succeeds, yet they are rarely made explicit in prior work. Different starting artifacts determine which solutions are reachable in MLAgentBench, truncated traces can still improve Atari agents, and larger minibatches do not monotonically improve generalization on BBEH. We conclude that the lack of a simple, universal way to set up learning loops across domains is a major hurdle for productionization and adoption. We provide practical guidance for making these choices. 

---
# The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation 

**Authors**: Mingyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24124)  

**Abstract**: RLHF-aligned language models exhibit response homogenization: on TruthfulQA (n=790), 40-79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have zero discriminative power (AUROC=0.500), while free token entropy retains signal (0.603). This alignment tax is task-dependent: on GSM8K (n=500), token entropy achieves 0.724 (Cohen's d=0.81).
A base-vs-instruct ablation confirms the causal role of alignment: the base model shows 1.0% single-cluster rate vs. 28.5% for the instruct model (p < 10^{-6}). A training stage ablation (Base 0.0% -> SFT 1.5% -> DPO 4.0% SCR) localizes the cause to DPO, not SFT. Cross-family replication on four model families reveals alignment tax severity varies by family and scale. We validate across 22 experiments, 5 benchmarks, 4 model families, and 3 model scales (3B-14B), with Jaccard, embedding, and NLI-based baselines at three DeBERTa scales (all ~0.51 AUROC). Cross-embedder validation with two independent embedding families rules out coupling bias. Cross-dataset validation on WebQuestions (58.0% SCR) confirms generalization beyond TruthfulQA. The central finding -- response homogenization -- is implementation-independent and label-free. Motivated by this diagnosis, we explore a cheapest-first cascade (UCBD) over orthogonal uncertainty signals. Selective prediction raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage; weakly dependent boundaries (|r| <= 0.12) enable 57% cost savings. 

---
# From Untamed Black Box to Interpretable Pedagogical Orchestration: The Ensemble of Specialized LLMs Architecture for Adaptive Tutoring 

**Authors**: Nizam Kadir  

**Link**: [PDF](https://arxiv.org/pdf/2603.23990)  

**Abstract**: Monolithic Large Language Models (LLMs) used in educational dialogue often behave as "black boxes," where pedagogical decisions are implicit and difficult to audit, frequently violating instructional constraints by providing answers too early. We introduce the Ensemble of Specialized LLMS (ES-LLMS) architecture that separates decision-making from wording. Pedagogical actions are selected by a deterministic rules-based orchestrator coordinating specialized agents covering tutoring, assessment, feedback, scaffolding, motivation and ethics-guided by an interpretable Bayesian Knowledge Tracing (BKT) student model. An LLM renderer surface-realizes the chosen action in natural language. This design emphasizes reliability and controllability: constraints such as "attempt-before-hint" and hint caps are enforced as explicit rules, and the system logs per-turn agent traces and constraint checks. Validation of pedagogical quality via human expert reviewers (N=6) and a multi-LLM-as-Judge panel (six state-of-the-art models) showed that ES-LLMs were preferred in 91.7% and 79.2% of cases, respectively. The architecture significantly outperformed monolithic baselines across all seven dimensions, particularly in Scaffolding & Guidance, and Trust & Explainability. Furthermore, a Monte Carlo simulation (N=2,400) exposed a "Mastery Gain Paradox," where monolithic tutors inflated short-term performance through over-assistance. In contrast, ES-LLMs achieved 100% adherence to pedagogical constraints (e.g., attempt-before-hint) and a 3.3x increase in hint efficiency. Operationally, ES-LLMs reduced costs by 54% and latency by 22% by utilizing stateless prompts. We conclude that structural decoupling is essential for transforming stochastic models into trustworthy, verifiable and resource-efficient pedagogical agents. 

---
# Wafer-Level Etch Spatial Profiling for Process Monitoring from Time-Series with Time-LLM 

**Authors**: Hyunwoo Kim, Munyoung Lee, Seung Hyub Jeon, Kyu Sung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.23576)  

**Abstract**: Understanding wafer-level spatial variations from in-situ process signals is essential for advanced plasma etching process monitoring. While most data-driven approaches focus on scalar indicators such as average etch rate, actual process quality is determined by complex two-dimensional spatial distributions across the wafer. This paper presents a spatial regression model that predicts wafer-level etch depth distributions directly from multichannel in-situ process time series. We propose a Time-LLM-based spatial regression model that extends LLM reprogramming from conventional time-series forecasting to wafer-level spatial estimation by redesigning the input embedding and output projection. Using the BOSCH plasma-etching dataset, we demonstrate stable performance under data-limited conditions, supporting the feasibility of LLM-based reprogramming for wafer-level spatial monitoring. 

---
# MedMT-Bench: Can LLMs Memorize and Understand Long Multi-Turn Conversations in Medical Scenarios? 

**Authors**: Lin Yang, Yuancheng Yang, Xu Wang, Changkun Liu, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23519)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across various specialist domains and have been integrated into high-stakes areas such as medicine. However, as existing medical-related benchmarks rarely stress-test the long-context memory, interference robustness, and safety defense required in practice. To bridge this gap, we introduce MedMT-Bench, a challenging medical multi-turn instruction following benchmark that simulates the entire diagnosis and treatment process. We construct the benchmark via scene-by-scene data synthesis refined by manual expert editing, yielding 400 test cases that are highly consistent with real-world application scenarios. Each test case has an average of 22 rounds (maximum of 52 rounds), covering 5 types of difficult instruction following issues. For evaluation, we propose an LLM-as-judge protocol with instance-level rubrics and atomic test points, validated against expert annotations with a human-LLM agreement of 91.94\%. We test 17 frontier models, all of which underperform on MedMT-Bench (overall accuracy below 60.00\%), with the best model reaching 59.75\%. MedMT-Bench can be an essential tool for driving future research towards safer and more reliable medical AI. The benchmark is available in this https URL 

---
# DepthCharge: A Domain-Agnostic Framework for Measuring Depth-Dependent Knowledge in Large Language Models 

**Authors**: Alexander Sheppert  

**Link**: [PDF](https://arxiv.org/pdf/2603.23514)  

**Abstract**: Large Language Models appear competent when answering general questions but often fail when pushed into domain-specific details. No existing methodology provides an out-of-the-box solution for measuring how deeply LLMs can sustain accurate responses under adaptive follow-up questioning across arbitrary domains.
We present DepthCharge, a domain-agnostic framework that measures knowledge depth through three innovations: adaptive probing that generates follow-up questions based on concepts the model actually mentions, on-demand fact verification from authoritative sources, and survival statistics with constant sample sizes at every depth level. The framework can be deployed on any knowledge domain with publicly verifiable facts, without requiring pre-constructed test sets or domain-specific expertise. DepthCharge results are relative to the evaluator model used for answer checking, making the framework a tool for comparative evaluation rather than absolute accuracy certification.
Empirical validation across four diverse domains (Medicine, Constitutional Law, Ancient Rome, and Quantum Computing) with five frontier models demonstrates that DepthCharge reveals depth-dependent performance variation hidden by standard benchmarks. Expected Valid Depth (EVD) ranges from 3.45 to 7.55 across model-domain combinations, and model rankings vary substantially by domain, with no single model dominating all areas. Cost-performance analysis further reveals that expensive models do not always achieve deeper knowledge, suggesting that domain-specific evaluation is more informative than aggregate benchmarks for model selection in professional applications. 

---
# From Physician Expertise to Clinical Agents: Preserving, Standardizing, and Scaling Physicians' Medical Expertise with Lightweight LLM 

**Authors**: Chanyong Luo, Jirui Dai, Zhendong Wang, Kui Chen, Jiaxi Yang, Bingjie Lu, Jing Wang, Jiaxin Hao, Bing Li, Ruiyang He, Yiyu Qiao, Chenkai Zhang, Kaiyu Wang, Zhi Liu, Zeyu Zheng, Yan Li, Xiaohong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23520)  

**Abstract**: Medicine is an empirical discipline refined through long-term observation and the messy, high-variance reality of clinical practice. Physicians build diagnostic and therapeutic competence through repeated cycles of application, reflection, and improvement, forming individualized methodologies. Yet outcomes vary widely, and master physicians' knowledge systems are slow to develop and hard to transmit at scale, contributing to the scarcity of high-quality clinical expertise. To address this, we propose Med-Shicheng, a general framework that enables large language models to systematically learn and transfer distinguished physicians' diagnostic-and-therapeutic philosophy and case-dependent adaptation rules in a standardized way. Built on Tianyi, Med-Shicheng consists of five stages. We target five National Masters of Chinese Medicine or distinguished TCM physicians, curate multi-source materials, and train a single model to internalize all five knowledge systems across seven tasks, including etiology-pathogenesis analysis, syndrome diagnosis, treatment principle selection, prescription generation, prescription explanation, symptom evolution with regimen adjustment, and clinical advice. Implemented on Qwen2.5-1.5B-Base, Med-Shicheng runs on resource-constrained GPUs while achieving performance comparable to DeepSeek-R1 and GPT-5. We also examine the reliability of LLM-as-a-judge versus physician evaluation: automated judging tracks overall trends but shows bias on fine-grained individualized distinctions, highlighting the need for physician involvement when ground truth is unavailable and for domain-adapted judge models. 

---
# Internal Safety Collapse in Frontier Large Language Models 

**Authors**: Yutao Wu, Xiao Liu, Yifeng Gao, Xiang Zheng, Hanxun Huang, Yige Li, Cong Wang, Bo Li, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23509)  

**Abstract**: This work identifies a critical failure mode in frontier large language models (LLMs), which we term Internal Safety Collapse (ISC): under certain task conditions, models enter a state in which they continuously generate harmful content while executing otherwise benign tasks. We introduce TVD (Task, Validator, Data), a framework that triggers ISC through domain tasks where generating harmful content is the only valid completion, and construct ISC-Bench containing 53 scenarios across 8 professional disciplines. Evaluated on JailbreakBench, three representative scenarios yield worst-case safety failure rates averaging 95.3% across four frontier LLMs (including GPT-5.2 and Claude Sonnet 4.5), substantially exceeding standard jailbreak attacks. Frontier models are more vulnerable than earlier LLMs: the very capabilities that enable complex task execution become liabilities when tasks intrinsically involve harmful content. This reveals a growing attack surface: almost every professional domain uses tools that process sensitive data, and each new dual-use tool automatically expands this vulnerability--even without any deliberate attack. Despite substantial alignment efforts, frontier LLMs retain inherently unsafe internal capabilities: alignment reshapes observable outputs but does not eliminate the underlying risk profile. These findings underscore the need for caution when deploying LLMs in high-stakes settings. Source code: this https URL 

---
# Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents 

**Authors**: Peijun Qing, Puneet Mathur, Nedim Lipka, Varun Manjunatha, Ryan Rossi, Franck Dernoncourt, Saeed Hassanpour, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23518)  

**Abstract**: General-purpose embedding models excel at recognizing semantic similarities but fail to capture the characteristics of texts specified by user instructions. In contrast, instruction-tuned embedders can align embeddings with textual instructions yet cannot autonomously infer latent corpus structures, such as determining the optimal number of clusters. To address both limitations, we reframe instruction-following clustering as a generative task and train large reasoning models (LRMs) as autonomous clustering agents. Our reasoning-driven training pipeline enables LRMs to interpret high-level clustering instructions and then infer the corresponding latent groupings. To evaluate this paradigm, we introduce ReasonCluster, a comprehensive benchmark comprising 28 diverse tasks spanning daily dialogue, legal cases, and financial reports. Experiments across diverse datasets and clustering scenarios show that our approach consistently outperforms strong embedding-based methods and LRM baselines, demonstrating that explicit reasoning fosters more faithful and interpretable instruction-based clustering. 

---
# Leveraging Computerized Adaptive Testing for Cost-effective Evaluation of Large Language Models in Medical Benchmarking 

**Authors**: Tianpeng Zheng, Zhehan Jiang, Jiayi Liu, Shicong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.23506)  

**Abstract**: The rapid proliferation of large language models (LLMs) in healthcare creates an urgent need for scalable and psychometrically sound evaluation methods. Conventional static benchmarks are costly to administer repeatedly, vulnerable to data contamination, and lack calibrated measurement properties for fine-grained performance tracking. We propose and validate a computerized adaptive testing (CAT) framework grounded in item response theory (IRT) for efficient assessment of standardized medical knowledge in LLMs. The study comprises a two-phase design: a Monte Carlo simulation to identify optimal CAT configurations and an empirical evaluation of 38 LLMs using a human-calibrated medical item bank. Each model completed both the full item bank and an adaptive test that dynamically selected items based on real-time ability estimates and terminated upon reaching a predefined reliability threshold (standard error <= 0.3). Results show that CAT-derived proficiency estimates achieved a near-perfect correlation with full-bank estimates (r = 0.988) while using only 1.3 percent of the items. Evaluation time was reduced from several hours to minutes per model, with substantial reductions in token usage and computational cost, while preserving inter-model performance rankings. This work establishes a psychometric framework for rapid, low-cost benchmarking of foundational medical knowledge in LLMs. The proposed adaptive methodology is intended as a standardized pre-screening and continuous monitoring tool and is not a substitute for real-world clinical validation or safety-oriented prospective studies. 

---
# Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct 

**Authors**: Christopher Ackerman, Nina Panickssery  

**Link**: [PDF](https://arxiv.org/pdf/2410.02064)  

**Abstract**: It has been reported that LLMs can recognize their own writing. As this has potential implications for AI safety, yet is relatively understudied, we investigate the phenomenon, seeking to establish whether it robustly occurs at the behavioral level, how the observed behavior is achieved, and whether it can be controlled. First, we find that the Llama3-8b-Instruct chat model - but not the base Llama3-8b model - can reliably distinguish its own outputs from those of humans, and present evidence that the chat model is likely using its experience with its own outputs, acquired during post-training, to succeed at the writing recognition task. Second, we identify a vector in the residual stream of the model that is differentially activated when the model makes a correct self-written-text recognition judgment, show that the vector activates in response to information relevant to self-authorship, present evidence that the vector is related to the concept of "self" in the model, and demonstrate that the vector is causally related to the model's ability to perceive and assert self-authorship. Finally, we show that the vector can be used to control both the model's behavior and its perception, steering the model to claim or disclaim authorship by applying the vector to the model's output as it generates it, and steering the model to believe or disbelieve it wrote arbitrary texts by applying the vector to them as the model reads them. 

---
# Mitigating Many-Shot Jailbreaking 

**Authors**: Christopher M. Ackerman, Nina Panickssery  

**Link**: [PDF](https://arxiv.org/pdf/2504.09604)  

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training. 

---
# Konkani LLM: Multi-Script Instruction Tuning and Evaluation for a Low-Resource Indian Language 

**Authors**: Reuben Chagas Fernandes, Gaurang S. Patkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23529)  

**Abstract**: Large Language Models (LLMs) consistently under perform in low-resource linguistic contexts such as Konkani. This performance deficit stems from acute training data scarcity compounded by high script diversity across Devanagari, Romi and Kannada orthographies. To address this gap, we introduce Konkani-Instruct-100k, a comprehensive synthetic instruction-tuning dataset generated through Gemini 3.
We establish rigorous baseline benchmarks by evaluating leading open-weights architectures including Llama 3.1, Qwen2.5 and Gemma 3 alongside proprietary closed-source models. Our primary contribution involves the development of Konkani LLM, a series of fine-tuned models optimized for regional nuances. Furthermore, we are developing the Multi-Script Konkani Benchmark to facilitate cross-script linguistic evaluation. In machine translation, Konkani LLM delivers consistent gains over the corresponding base models and is competitive with and in several settings surpasses proprietary baselines 

---
# Training a Large Language Model for Medical Coding Using Privacy-Preserving Synthetic Clinical Data 

**Authors**: John Cook, Michael Wyatt, Peng Wei, Iris Chin, Santosh Gupta, Van Zyl Van Vuuren, Richie Siburian, Amanda Spicer, Kristen Viviano, Alda Cami, Raunaq Malhotra, Zhewei Yao, Jeff Rasley, Gaurav Kaushik  

**Link**: [PDF](https://arxiv.org/pdf/2603.23515)  

**Abstract**: Improving the accuracy and reliability of medical coding reduces clinician burnout and supports revenue cycle processes, freeing providers to focus more on patient care. However, automating the assignment of ICD-10-CM and CPT codes from clinical documentation remains a challenge due to heterogeneous records, nuanced coding guidelines, and long-tail distributions. Large language models have been proposed to help or automate specific medical coding tasks. However, foundation models are not explicitly trained for medical coding and zero-shot coding has yielded poor results. We investigate whether a modern open-weight foundation model can be adapted for an expert-level medical coding task using privacy-preserving synthetic training data derived from electronic health records. We fine-tune Llama 3-70B on pairs of clinical notes and gold codes generated from EHR-grounded templates and coding policies, then evaluate exact-code prediction for ICD-10-CM and CPT. A zero-shot baseline with the unadapted model achieved an F1 score of 0.18 for exact code match. After fine-tuning on the synthetic corpus, exact-match F1 exceeded 0.70, representing a large absolute gain across both code systems. Notably, performance remained high on complex categories that often require multi-step clinical reasoning and code composition, including Advanced Illness and Frailty classes, and the model retained its performance on medical comprehension tasks. These results indicate that synthetic, policy-aware data can efficiently teach a general-purpose large language model to support precise medical coding without exposing protected health information. The approach offers a practical path for training coding agents safely and iteratively on specific tasks that represent real-world populations. 

---
# Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? 

**Authors**: Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24472)  

**Abstract**: Self-distillation has emerged as an effective post-training paradigm for LLMs, often improving performance while shortening reasoning traces. However, in mathematical reasoning, we find that it can reduce response length while degrading performance. We trace this degradation to the suppression of epistemic verbalization - the model's expression of uncertainty during reasoning. Through controlled experiments varying conditioning context richness and task coverage, we show that conditioning the teacher on rich information suppresses uncertainty expression, enabling rapid in-domain optimization with limited task coverage but harming OOD performance, where unseen problems benefit from expressing uncertainty and adjusting accordingly. Across Qwen3-8B, DeepSeek-Distill-Qwen-7B, and Olmo3-7B-Instruct, we observe performance drops of up to 40%. Our findings highlight that exposing appropriate levels of uncertainty is crucial for robust reasoning and underscore the importance of optimizing reasoning behavior beyond merely reinforcing correct answer traces. 

---
# Mechanic: Sorrifier-Driven Formal Decomposition Workflow for Automated Theorem Proving 

**Authors**: Ruichen Qiu, Yichuan Cao, Junqi Liu, Dakai Guo, Xiao-Shan Gao, Lihong Zhi, Ruyong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.24465)  

**Abstract**: Recent advances in large language models (LLMs) and LLM-based agents have substantially improved the capabilities of automated theorem proving. However, for problems requiring complex mathematical reasoning, current systems rarely succeed on the first try and must repeatedly modify their proof strategies. Existing approaches for handling failed attempts typically either discard the entire proof and regenerate it from scratch or iteratively fix errors within the proof. The former is inefficient, as it may abandon mostly correct reasoning due to localized errors, while the latter, although preserving prior progress, leads to progressively longer contexts which progressively degrades the model's ability to attend to the remaining unresolved subproblems. To address this dilemma, we propose Mechanic, a novel agent system that employs a sorry-driven formal decomposition strategy. By leveraging the sorry placeholder in Lean to precisely isolate unresolved subgoals while preserving the surrounding verified proof structure, Mechanic extracts each failed subproblem into a clean, self-contained context and resolves it independently. This avoids both the waste of full regeneration and the excessive context length induced by repeated repairs. Experimental results on challenging mathematical competition benchmarks, including IMO 2025 and Putnam 2025, demonstrate that our agent achieves significant advantages in proving efficiency. 

---
# Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition 

**Authors**: Aleix Sant, Jordi Luque, Carlos Escolano  

**Link**: [PDF](https://arxiv.org/pdf/2603.24242)  

**Abstract**: Federated Learning (FL) of Large Language Models (LLMs) in multilingual environments presents significant challenges stemming from heterogeneous language distributions across clients and disparities in language resource availability. To address these challenges, we extended the FederatedScope-LLM framework to support multilingual instruction-tuning experiments with LLMs. We also introduced a novel client-specific early stopping mechanism, Local Dynamic Early Stopping (LDES-FL), which allows clients to pause and resume local training based on client-side validation performance, enhancing training efficiency and sustainability. Through a series of experiments, we studied how client language composition - from fully monolingual to increasingly multilingual clients - affects multilingual quality, fairness and training cost. Monolingual local fine-tuning remains the most effective for single-language specialization, whereas federated training is better suited to learning a single balanced multilingual model. In FL, increasing within-client multilinguality leads to stronger and fairer global models, narrows the gap to centralized multilingual fine-tuning, and yields the largest gains for lower-resource languages, albeit at the cost of more optimization steps. Overall, our results identify client language composition as a key design variable in multilingual FL, shaping performance, fairness and efficiency 

---
# CVPD at QIAS 2026: RAG-Guided LLM Reasoning for Al-Mawarith Share Computation and Heir Allocation 

**Authors**: Wassim Swaileh, Mohammed-En-Nadhir Zighem, Hichem Telli, Salah Eddine Bekhouche, Abdellah Zakaria Sellam, Fadi Dornaika, Dimitrios Kotzinos  

**Link**: [PDF](https://arxiv.org/pdf/2603.24012)  

**Abstract**: Islamic inheritance (Ilm al-Mawarith) is a multi-stage legal reasoning task requiring the identification of eligible heirs, resolution of blocking rules (hajb), assignment of fixed and residual shares, handling of adjustments such as awl and radd, and generation of a consistent final distribution. The task is further complicated by variations across legal schools and civil-law codifications, requiring models to operate under explicit legal configurations.
We present a retrieval-augmented generation (RAG) pipeline for this setting, combining rule-grounded synthetic data generation, hybrid retrieval (dense and BM25) with cross-encoder reranking, and schema-constrained output validation. A symbolic inheritance calculator is used to generate a large high-quality synthetic corpus with full intermediate reasoning traces, ensuring legal and numerical consistency.
The proposed system achieves a MIR-E score of 0.935 and ranks first on the official QIAS 2026 blind-test leaderboard. Results demonstrate that retrieval-grounded, schema-aware generation significantly improves reliability in high-precision Arabic legal reasoning tasks. 

---
# Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith 

**Authors**: Somaya Eltanbouly, Samer Rashwani  

**Link**: [PDF](https://arxiv.org/pdf/2603.23972)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in many language tasks, yet they continue to struggle with complex historical and religious Arabic texts such as the Quran and Hadith. To address this limitation, we develop a retrieval-augmented generation (RAG) framework grounded in diachronic lexicographic knowledge. Unlike prior RAG systems that rely on general-purpose corpora, our approach retrieves evidence from the Doha Historical Dictionary of Arabic (DHDA), a large-scale resource documenting the historical development of Arabic vocabulary. The proposed pipeline combines hybrid retrieval with an intent-based routing mechanism to provide LLMs with precise, contextually relevant historical information. Our experiments show that this approach improves the accuracy of Arabic-native LLMs, including Fanar and ALLaM, to over 85\%, substantially reducing the performance gap with Gemini, a proprietary large-scale model. Gemini also serves as an LLM-as-a-judge system for automatic evaluation in our experiments. The automated judgments were verified through human evaluation, demonstrating high agreement (kappa = 0.87). An error analysis further highlights key linguistic challenges, including diacritics and compound expressions. These findings demonstrate the value of integrating diachronic lexicographic resources into retrieval-augmented generation frameworks to enhance Arabic language understanding, particularly for historical and religious texts. The code and resources are publicly available at: this https URL. 

---
# LLMpedia: A Transparent Framework to Materialize an LLM's Encyclopedic Knowledge at Scale 

**Authors**: Muhammed Saeed, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2603.24080)  

**Abstract**: Benchmarks such as MMLU suggest flagship language models approach factuality saturation, with scores above 90\%. We show this picture is incomplete. \emph{LLMpedia} generates encyclopedic articles entirely from parametric memory, producing ${\sim}$1M articles across three model families without retrieval. For gpt-5-mini, the verifiable true rate on Wikipedia-covered subjects is only 74.7\% -- more than 15 percentage points below the benchmark-based picture, consistent with the availability bias of fixed-question evaluation. Beyond Wikipedia, frontier subjects verifiable only through curated web evidence fall further to 63.2\% true rate. Wikipedia covers just 61\% of surfaced subjects, and three model families overlap by only 7.3\% in subject choice. In a capture-trap benchmark inspired by prior analysis of Grokipedia, LLMpedia achieves substantially higher factuality at roughly half the textual similarity to Wikipedia. Unlike Grokipedia, every prompt, artifact, and evaluation verdict is publicly released, making LLMpedia the first fully open parametric encyclopedia -- bridging factuality evaluation and knowledge materialization. All data, code, and a browsable interface are at this https URL. 

---
# From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents 

**Authors**: Sirui Xia, Yikai Zhang, Aili Chen, Siye Wu, Siyu Yuan, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.23951)  

**Abstract**: Discovering improved policy optimization algorithms for language models remains a costly manual process requiring repeated mechanism-level modification and validation. Unlike simple combinatorial code search, this problem requires searching over algorithmic mechanisms tightly coupled with training dynamics while reusing empirical evidence across iterations. We propose POISE, a closed-loop framework for automated discovery of policy optimization algorithms for language models. POISE maintains a structured, genealogically linked archive linking proposals, executable implementations, standardized evaluations, and natural-language reflections to support evidence-driven iteration. In mathematical reasoning experiments starting from GRPO, POISE evaluates 64 candidate algorithms and discovers improved mechanisms, including analytic-variance scaling and validity masking. The best variant improves weighted Overall from 47.8 to 52.5 (+4.6) and increases AIME25 pass@32 from 26.7% to 43.3%, demonstrating the feasibility of automated policy optimization discovery while supporting interpretable design principles. 

---
# Dialogue to Question Generation for Evidence-based Medical Guideline Agent Development 

**Authors**: Zongliang Ji, Ziyang Zhang, Xincheng Tan, Matthew Thompson, Anna Goldenberg, Carl Yang, Rahul G. Krishnan, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23937)  

**Abstract**: Evidence-based medicine (EBM) is central to high-quality care, but remains difficult to implement in fast-paced primary care settings. Physicians face short consultations, increasing patient loads, and lengthy guideline documents that are impractical to consult in real time. To address this gap, we investigate the feasibility of using large language models (LLMs) as ambient assistants that surface targeted, evidence-based questions during physician-patient encounters. Our study focuses on question generation rather than question answering, with the aim of scaffolding physician reasoning and integrating guideline-based practice into brief consultations. We implemented two prompting strategies, a zero-shot baseline and a multi-stage reasoning variant, using Gemini 2.5 as the backbone model. We evaluated on a benchmark of 80 de-identified transcripts from real clinical encounters, with six experienced physicians contributing over 90 hours of structured review. Results indicate that while general-purpose LLMs are not yet fully reliable, they can produce clinically meaningful and guideline-relevant questions, suggesting significant potential to reduce cognitive burden and make EBM more actionable at the point of care. 

---
# Language Model Planners do not Scale, but do Formalizers? 

**Authors**: Owen Jiang, Cassie Huang, Ashish Sabharwal, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23844)  

**Abstract**: Recent work shows overwhelming evidence that LLMs, even those trained to scale their reasoning trace, perform unsatisfactorily when solving planning problems too complex. Whether the same conclusion holds for LLM formalizers that generate solver-oriented programs remains unknown. We systematically show that LLM formalizers greatly out-scale LLM planners, some retaining perfect accuracy in the classic BlocksWorld domain with a huge state space of size up to $10^{165}$. While performance of smaller LLM formalizers degrades with problem complexity, we show that a divide-and-conquer formalizing technique can greatly improve its robustness. Finally, we introduce unraveling problems where one line of problem description realistically corresponds to exponentially many lines of formal language such as the Planning Domain Definition Language (PDDL), greatly challenging LLM formalizers. We tackle this challenge by introducing a new paradigm, namely LLM-as-higher-order-formalizer, where an LLM generates a program generator. This decouples token output from the combinatorial explosion of the underlying formalization and search space. 

---
# CoCR-RAG: Enhancing Retrieval-Augmented Generation in Web Q&A via Concept-oriented Context Reconstruction 

**Authors**: Kaize Shi, Xueyao Sun, Qika Lin, Firoj Alam, Qing Li, Xiaohui Tao, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23989)  

**Abstract**: Retrieval-augmented generation (RAG) has shown promising results in enhancing Q&A by incorporating information from the web and other external sources. However, the supporting documents retrieved from the heterogeneous web often originate from multiple sources with diverse writing styles, varying formats, and inconsistent granularity. Fusing such multi-source documents into a coherent and knowledge-intensive context remains a significant challenge, as the presence of irrelevant and redundant information can compromise the factual consistency of the inferred answers. This paper proposes the Concept-oriented Context Reconstruction RAG (CoCR-RAG), a framework that addresses the multi-source information fusion problem in RAG through linguistically grounded concept-level integration. Specifically, we introduce a concept distillation algorithm that extracts essential concepts from Abstract Meaning Representation (AMR), a stable semantic representation that structures the meaning of texts as logical graphs. The distilled concepts from multiple retrieved documents are then fused and reconstructed into a unified, information-intensive context by Large Language Models, which supplement only the necessary sentence elements to highlight the core knowledge. Experiments on the PopQA and EntityQuestions datasets demonstrate that CoCR-RAG significantly outperforms existing context-reconstruction methods across these Web Q&A benchmarks. Furthermore, CoCR-RAG shows robustness across various backbone LLMs, establishing itself as a flexible, plug-and-play component adaptable to different RAG frameworks. 

---
# IslamicMMLU: A Benchmark for Evaluating LLMs on Islamic Knowledge 

**Authors**: Ali Abdelaal, Mohammed Nader Al Haffar, Mahmoud Fawzi, Walid Magdy  

**Link**: [PDF](https://arxiv.org/pdf/2603.23750)  

**Abstract**: Large language models are increasingly consulted for Islamic knowledge, yet no comprehensive benchmark evaluates their performance across core Islamic disciplines. We introduce IslamicMMLU, a benchmark of 10,013 multiple-choice questions spanning three tracks: Quran (2,013 questions), Hadith (4,000 questions), and Fiqh (jurisprudence, 4,000 questions). Each track is formed of multiple types of questions to examine LLMs capabilities handling different aspects of Islamic knowledge. The benchmark is used to create the IslamicMMLU public leaderboard for evaluating LLMs, and we initially evaluate 26 LLMs, where their averaged accuracy across the three tracks varied between 39.8\% to 93.8\% (by Gemini 3 Flash). The Quran track shows the widest span (99.3\% to 32.4\%), while the Fiqh track includes a novel madhab (Islamic school of jurisprudence) bias detection task revealing variable school-of-thought preferences across models. Arabic-specific models show mixed results, but they all underperform compared to frontier models. The evaluation code and leaderboard are made publicly available. 

---
# Large Language Models Unpack Complex Political Opinions through Target-Stance Extraction 

**Authors**: Özgür Togay, Florian Kunneman, Javier Garcia-Bernardo, Anastasia Giachanou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23531)  

**Abstract**: Political polarization emerges from a complex interplay of beliefs about policies, figures, and issues. However, most computational analyses reduce discourse to coarse partisan labels, overlooking how these beliefs interact. This is especially evident in online political conversations, which are often nuanced and cover a wide range of subjects, making it difficult to automatically identify the target of discussion and the opinion expressed toward them. In this study, we investigate whether Large Language Models (LLMs) can address this challenge through Target-Stance Extraction (TSE), a recent natural language processing task that combines target identification and stance detection, enabling more granular analysis of political opinions. For this, we construct a dataset of 1,084 Reddit posts from r/NeutralPolitics, covering 138 distinct political targets and evaluate a range of proprietary and open-source LLMs using zero-shot, few-shot, and context-augmented prompting strategies. Our results show that the best models perform comparably to highly trained human annotators and remain robust on challenging posts with low inter-annotator agreement. These findings demonstrate that LLMs can extract complex political opinions with minimal supervision, offering a scalable tool for computational social science and political text analysis. 

---
# Do 3D Large Language Models Really Understand 3D Spatial Relationships? 

**Authors**: Xianzheng Ma, Tao Sun, Shuai Chen, Yash Bhalgat, Jindong Gu, Angel X Chang, Iro Armeni, Iro Laina, Songyou Peng, Victor Adrian Prisacariu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23523)  

**Abstract**: Recent 3D Large-Language Models (3D-LLMs) claim to understand 3D worlds, especially spatial relationships among objects. Yet, we find that simply fine-tuning a language model on text-only question-answer pairs can perform comparably or even surpass these methods on the SQA3D benchmark without using any 3D input. This indicates that the SQA3D benchmark may not be able to detect if the model exploits textual shortcuts rather than engages in 3D-aware reasoning. To address this issue, we introduce Real-3DQA, a more rigorous evaluation benchmark that filters out easy-to-guess questions and introduces a structured taxonomy to assess various aspects of 3D reasoning. Experiments on Real-3DQA confirm that existing 3D-LLMs struggle with spatial relationships once simple cues are removed. We further propose a 3D-reweighted training objective that guides model to rely more on 3D visual clues, substantially enhancing 3D-LLMs performance in spatial reasoning tasks. Our findings underscore the need for robust benchmarks and tailored training strategies to advance genuine 3D vision-language understanding. Project page: this https URL. 

---
# Berta: an open-source, modular tool for AI-enabled clinical documentation 

**Authors**: Samridhi Vaid, Mike Weldon, Jesse Dunn, Sacha Davis, Kevin Lonergan, Henry Li, Jeffrey Franc, Mohamed Abdalla, Daniel C. Baumgart, Jake Hayward, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2603.23513)  

**Abstract**: Commercial AI scribes cost \$99-600 per physician per month, operate as opaque systems, and do not return data to institutional infrastructure, limiting organizational control over data governance, quality improvement, and clinical workflows. We developed Berta, an open-source modular scribe platform for AI-enabled clinical documentation, and deployed a customized implementation within Alberta Health Services (AHS) integrated with their existing Snowflake AI Data Cloud infrastructure. The system combines automatic speech recognition with large language models while retaining all clinical data within the secure AHS environment. During eight months (November 2024 to July 2025), 198 emergency physicians used the system in 105 urban and rural facilities, generating 22148 clinical sessions and more than 2800 hours of audio. The use grew from 680 to 5530 monthly sessions. Operating costs averaged less than \$30 per physician per month, a 70-95% reduction compared to commercial alternatives. AHS has since approved expansion to 850 physicians. This is the first provincial-scale deployment of an AI scribe integrated with existing health system infrastructure. By releasing Berta as open source, we provide a reproducible, cost-effective alternative that health systems can adapt to their own secure environments, supporting data sovereignty and informed evaluation of AI documentation technology. 

---
# How Vulnerable Are Edge LLMs? 

**Authors**: Ao Ding, Hongzong Li, Zi Liang, Zhanpeng Shi, Shuxin Zhuang, Shiqin Tang, Rong Feng, Ping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23822)  

**Abstract**: Large language models (LLMs) are increasingly deployed on edge devices under strict computation and quantization constraints, yet their security implications remain unclear. We study query-based knowledge extraction from quantized edge-deployed LLMs under realistic query budgets and show that, although quantization introduces noise, it does not remove the underlying semantic knowledge, allowing substantial behavioral recovery through carefully designed queries. To systematically analyze this risk, we propose \textbf{CLIQ} (\textbf{Cl}ustered \textbf{I}nstruction \textbf{Q}uerying), a structured query construction framework that improves semantic coverage while reducing redundancy. Experiments on quantized Qwen models (INT8/INT4) demonstrate that CLIQ consistently outperforms original queries across BERTScore, BLEU, and ROUGE, enabling more efficient extraction under limited budgets. These results indicate that quantization alone does not provide effective protection against query-based extraction, highlighting a previously underexplored security risk in edge-deployed LLMs. 

---
# The Geometric Price of Discrete Logic: Context-driven Manifold Dynamics of Number Representations 

**Authors**: Long Zhang, Dai-jun Lin, Wei-neng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23577)  

**Abstract**: Large language models (LLMs) generalize smoothly across continuous semantic spaces, yet strict logical reasoning demands the formation of discrete decision boundaries. Prevailing theories relying on linear isometric projections fail to resolve this fundamental tension. In this work, we argue that task context operates as a non-isometric dynamical operator that enforces a necessary "topological distortion." By applying Gram-Schmidt decomposition to residual-stream activations , we reveal a dual-modulation mechanism driving this process: a class-agnostic topological preservation that anchors global structure to prevent semantic collapse, and a specific algebraic divergence that directionally tears apart cross-class concepts to forge logical boundaries. We validate this geometric evolution across a gradient of tasks, from simple mapping to complex primality testing. Crucially, targeted specific vector ablation establishes a strict causal binding between this topology and model function: algebraically erasing the divergence component collapses parity classification accuracy from 100% to chance levels (38.57%). Furthermore, we uncover a three-phase layer-wise geometric dynamic and demonstrate that under social pressure prompts, models fail to generate sufficient divergence. This results in a "manifold entanglement" that geometrically explains sycophancy and hallucination. Ultimately, our findings revise the linear-isometric presumption, demonstrating that the emergence of discrete logic in LLMs is purchased at an irreducible cost of topological deformation. 

---
# Sequence-aware Large Language Models for Explainable Recommendation 

**Authors**: Gangyi Zhang, Runzhe Teng, Chongming Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.24136)  

**Abstract**: Large Language Models (LLMs) have shown strong potential in generating natural language explanations for recommender systems. However, existing methods often overlook the sequential dynamics of user behavior and rely on evaluation metrics misaligned with practical utility. We propose SELLER (SEquence-aware LLM-based framework for Explainable Recommendation), which integrates explanation generation with utility-aware evaluation. SELLER combines a dual-path encoder-capturing both user behavior and item semantics with a Mixture-of-Experts adapter to align these signals with LLMs. A unified evaluation framework assesses explanations via both textual quality and their effect on recommendation outcomes. Experiments on public benchmarks show that SELLER consistently outperforms prior methods in explanation quality and real-world utility. 

---
# VILLA: Versatile Information Retrieval From Scientific Literature Using Large LAnguage Models 

**Authors**: Blessy Antony, Amartya Dutta, Sneha Aggarwal, Vasu Gatne, Ozan Gökdemir, Samantha Grimes, Adam Lauring, Brian R. Wasik, Anuj Karpatne, T. M. Murali  

**Link**: [PDF](https://arxiv.org/pdf/2603.23849)  

**Abstract**: The lack of high-quality ground truth datasets to train machine learning (ML) models impedes the potential of artificial intelligence (AI) for science research. Scientific information extraction (SIE) from the literature using LLMs is emerging as a powerful approach to automate the creation of these datasets. However, existing LLM-based approaches and benchmarking studies for SIE focus on broad topics such as biomedicine and chemistry, are limited to choice-based tasks, and focus on extracting information from short and well-formatted text. The potential of SIE methods in complex, open-ended tasks is considerably under-explored. In this study, we used a domain that has been virtually ignored in SIE, namely virology, to address these research gaps. We design a unique, open-ended SIE task of extracting mutations in a given virus that modify its interaction with the host. We develop a new, multi-step retrieval augmented generation (RAG) framework called VILLA for SIE. In parallel, we curate a novel dataset of 629 mutations in ten influenza A virus proteins obtained from 239 scientific publications to serve as ground truth for the mutation extraction task. Finally, we demonstrate VILLA's superior performance using a novel and comprehensive evaluation and comparison with vanilla RAG and other state-of-the art RAG- and agent-based tools for SIE. 

---
