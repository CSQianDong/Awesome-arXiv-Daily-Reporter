# GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis 

**Authors**: Bo Yu, Cheng Yang, Dongyang Hou, Chengfu Liu, Jiayao Liu, Chi Wang, Zhiming Zhang, Haifeng Li, Wentao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13888)  

**Abstract**: The integration of Large Language Models (LLMs) into Geographic Information Systems (GIS) marks a paradigm shift toward autonomous spatial analysis. However, evaluating these LLM-based agents remains challenging due to the complex, multi-step nature of geospatial workflows. Existing benchmarks primarily rely on static text or code matching, neglecting dynamic runtime feedback and the multimodal nature of spatial outputs. To address this gap, we introduce GeoAgentBench (GABench), a dynamic and interactive evaluation benchmark tailored for tool-augmented GIS agents. GABench provides a realistic execution sandbox integrating 117 atomic GIS tools, encompassing 53 typical spatial analysis tasks across 6 core GIS domains. Recognizing that precise parameter configuration is the primary determinant of execution success in dynamic GIS environments, we designed the Parameter Execution Accuracy (PEA) metric, which utilizes a "Last-Attempt Alignment" strategy to quantify the fidelity of implicit parameter inference. Complementing this, a Vision-Language Model (VLM) based verification is proposed to assess data-spatial accuracy and cartographic style adherence. Furthermore, to address the frequent task failures caused by parameter misalignments and runtime anomalies, we developed a novel agent architecture, Plan-and-React, that mimics expert cognitive workflows by decoupling global orchestration from step-wise reactive execution. Extensive experiments with seven representative LLMs demonstrate that the Plan-and-React paradigm significantly outperforms traditional frameworks, achieving the optimal balance between logical rigor and execution robustness, particularly in multi-step reasoning and error recovery. Our findings highlight current capability boundaries and establish a robust standard for assessing and advancing the next generation of autonomous GeoAI. 

---
# AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot 

**Authors**: Joydeep Biswas, Sheila Schoepp, Gautham Vasan, Anthony Opipari, Arthur Zhang, Zichao Hu, Sebastian Joseph, Matthew Lease, Junyi Jessy Li, Peter Stone, Kiri L. Wagstaff, Matthew E. Taylor, Odest Chadwicke Jenkins  

**Link**: [PDF](https://arxiv.org/pdf/2604.13940)  

**Abstract**: Scientific peer review faces mounting strain as submission volumes surge, making it increasingly difficult to sustain review quality, consistency, and timeliness. Recent advances in AI have led the community to consider its use in peer review, yet a key unresolved question is whether AI can generate technically sound reviews at real-world conference scale. Here we report the first large-scale field deployment of AI-assisted peer review: every main-track submission at AAAI-26 received one clearly identified AI review from a state-of-the-art system. The system combined frontier models, tool use, and safeguards in a multi-stage process to generate reviews for all 22,977 full-review papers in less than a day. A large-scale survey of AAAI-26 authors and program committee members showed that participants not only found AI reviews useful, but actually preferred them to human reviews on key dimensions such as technical accuracy and research suggestions. We also introduce a novel benchmark and find that our system substantially outperforms a simple LLM-generated review baseline at detecting a variety of scientific weaknesses. Together, these results show that state-of-the-art AI methods can already make meaningful contributions to scientific peer review at conference scale, opening a path toward the next generation of synergistic human-AI teaming for evaluating research. 

---
# TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration 

**Authors**: Zerun Ma, Guoqiang Wang, Xinchen Xie, Yicheng Chen, He Du, Bowen Li, Yanan Sun, Wenran Liu, Kai Chen, Yining Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14116)  

**Abstract**: While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks. 

---
# Weight Patching: Toward Source-Level Mechanistic Localization in LLMs 

**Authors**: Chenghao Sun, Chengsheng Zhang, Guanzheng Qin, Rui Dai, Xinmei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.13694)  

**Abstract**: Mechanistic interpretability seeks to localize model behavior to the internal components that causally realize it. Prior work has advanced activation-space localization and causal tracing, but modules that appear important in activation space may merely aggregate or amplify upstream signals rather than encode the target capability in their own parameters. To address this gap, we propose Weight Patching, a parameter-space intervention method for source-oriented analysis in paired same-architecture models that differ in how strongly they express a target capability under the inputs of interest. Given a base model and a behavior-specialized counterpart, Weight Patching replaces selected module weights from the specialized model into the base model under a fixed input. We instantiate the method on instruction following and introduce a framework centered on a vector-anchor behavioral interface that provides a shared internal criterion for whether a task-relevant control state has been formed or recovered in open-ended generation. Under this framework, the analysis reveals a hierarchy from shallow candidate source-side carriers to aggregation and routing modules, and further to downstream execution circuits. The recovered component scores can also guide mechanism-aware model merging, improving selective fusion across the evaluated expert combinations and providing additional external validation. 

---
# The cognitive companion: a lightweight parallel monitoring architecture for detecting and recovering from reasoning degradation in LLM agents 

**Authors**: Rafflesia Khan, Nafiul Islam Khan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13759)  

**Abstract**: Large language model (LLM) agents on multi-step tasks suffer reasoning degradation, looping, drift, stuck states, at rates up to 30% on hard tasks. Current solutions include hard step limits (abrupt) or LLM-as-judge monitoring (10-15% overhead per step). This paper introduces the Cognitive Companion, a parallel monitoring architecture with two implementations: an LLM-based Companion and a novel zero-overhead Probe-based Companion. We report a three-batch feasibility study centered on Gemma 4 E4B, with an additional exploratory small-model analysis on Qwen 2.5 1.5B and Llama 3.2 1B. In our experiments, the LLM-based Companion reduced repetition on loop-prone tasks by 52-62% with approximately 11% overhead. The Probe-based Companion, trained on hidden states from layer 28, showed a mean effect size of +0.471 at zero measured inference overhead; its strongest probe result achieved cross-validated AUROC 0.840 on a small proxy-labeled dataset. A key empirical finding is that companion benefit appears task-type dependent: companions are most helpful on loop-prone and open-ended tasks, while effects are neutral or negative on more structured tasks. Our small-model experiments also suggest a possible scale boundary: companions did not improve the measured quality proxy on 1B-1.5B models, even when interventions fired. Overall, the paper should be read as a feasibility study rather than a definitive validation. The results provide encouraging evidence that sub-token monitoring may be useful, identify task-type sensitivity as a practical design constraint, and motivate selective companion activation as a promising direction for future work. 

---
# Reward Design for Physical Reasoning in Vision-Language Models 

**Authors**: Derek Lilienthal, Manisha Mukherjee, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2604.13993)  

**Abstract**: Physical reasoning over visual inputs demands tight integration of visual perception, domain knowledge, and multi-step symbolic inference. Yet even state-of-the-art Vision Language Models (VLMs) fall far short of human performance on physics benchmarks. While post-training algorithms such as Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) have demonstrated strong reasoning gains in language models, how reward design shapes VLM physical reasoning behavior remains poorly understood. We present a systematic reward ablation study for GRPO-based VLM training on physical reasoning. We compare four reward signals of increasing semantic richness: format compliance, answer accuracy, a composite rubric reward (answer correctness, physics principle identification, and unit consistency), and a novel internal reward derived from model attention weights over input image regions. We evaluate on PhyX, a 3,000-problem benchmark spanning six physics domains and six reasoning types across multiple-choice and open-ended formats, using IBM Granite Vision 3.3 (2B). Across both formats, GRPO with accuracy-based rewards outperforms SFT on most domains, though gains vary substantially by reward type and domain. Reward design does not uniformly improve performance. Instead, it induces domain-specific reasoning behaviors. Accuracy-based rewards provide the strongest overall gains. Rubric rewards improve structured reasoning quality without consistent accuracy improvements. Attention-based rewards enhance spatial reasoning while degrading performance in symbolic domains. Our internal attention-weight reward requires no spatial annotations and improves spatial relation accuracy from 0.27 to 0.50, suggesting that supervising where the model attends during generation is a promising direction for visually grounded physical reasoning. 

---
# WebXSkill: Skill Learning for Autonomous Web Agents 

**Authors**: Zhaoyang Wang, Qianhui Wu, Xuchao Zhang, Chaoyun Zhang, Wenlin Yao, Fazle Elahi Faisal, Baolin Peng, Si Qin, Suman Nath, Qingwei Lin, Chetan Bansal, Dongmei Zhang, Saravan Rajmohan, Jianfeng Gao, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.13318)  

**Abstract**: Autonomous web agents powered by large language models (LLMs) have shown promise in completing complex browser tasks, yet they still struggle with long-horizon workflows. A key bottleneck is the grounding gap in existing skill formulations: textual workflow skills provide natural language guidance but cannot be directly executed, while code-based skills are executable but opaque to the agent, offering no step-level understanding for error recovery or adaptation. We introduce WebXSkill, a framework that bridges this gap with executable skills, each pairing a parameterized action program with step-level natural language guidance, enabling both direct execution and agent-driven adaptation. WebXSkill operates in three stages: skill extraction mines reusable action subsequences from readily available synthetic agent trajectories and abstracts them into parameterized skills, skill organization indexes skills into a URL-based graph for context-aware retrieval, and skill deployment exposes two complementary modes, grounded mode for fully automated multi-step execution and guided mode where skills serve as step-by-step instructions that the agent follows with its native planning. On WebArena and WebVoyager, WebXSkill improves task success rate by up to 9.8 and 12.9 points over the baseline, respectively, demonstrating the effectiveness of executable skills for web agents. The code is publicly available at this https URL. 

---
# ReSS: Learning Reasoning Models for Tabular Data Prediction via Symbolic Scaffold 

**Authors**: Chenlang Yi, Gang Li, Zizhan Xiong, Tue Minh Cao, Yanmin Gong, My T. Thai, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13392)  

**Abstract**: Tabular data remains prevalent in high-stakes domains such as healthcare and finance, where predictive models are expected to provide both high accuracy and faithful, human-understandable reasoning. While symbolic models offer verifiable logic, they lack semantic expressiveness. Meanwhile, general-purpose LLMs often require specialized fine-tuning to master domain-specific tabular reasoning. To address the dual challenges of scalable data curation and reasoning consistency, we propose ReSS, a systematic framework that bridges symbolic and neural reasoning models. ReSS leverages a decision-tree model to extract instance-level decision paths as symbolic scaffolds. These scaffolds, alongside input features and labels, guide an LLM to generate grounded natural-language reasoning that strictly adheres to the underlying decision logic. The resulting high-quality dataset is used to fine-tune a pretrained LLM into a specialized tabular reasoning model, further enhanced by a scaffold-invariant data augmentation strategy to improve generalization and explainability. To rigorously assess faithfulness, we introduce quantitative metrics including hallucination rate, explanation necessity, and explanation sufficiency. Experimental results on medical and financial benchmarks demonstrate that ReSS-trained models improve traditional decision trees and standard fine-tuning approaches up to $10\%$ while producing faithful and consistent reasoning 

---
# Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents 

**Authors**: Kangsan Kim, Minki Kang, Taeil Kim, Yanlai Yang, Mengye Ren, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14004)  

**Abstract**: Memory-based self-evolution has emerged as a promising paradigm for coding agents. However, existing approaches typically restrict memory utilization to homogeneous task domains, failing to leverage the shared infrastructural foundations, such as runtime environments and programming languages, that exist across diverse real-world coding problems. To address this limitation, we investigate \textbf{Memory Transfer Learning} (MTL) by harnessing a unified memory pool from heterogeneous domains. We evaluate performance across 6 coding benchmarks using four memory representations, ranging from concrete traces to abstract insights. Our experiments demonstrate that cross-domain memory improves average performance by 3.7\%, primarily by transferring meta-knowledge, such as validation routines, rather than task-specific code. Importantly, we find that abstraction dictates transferability; high-level insights generalize well, whereas low-level traces often induce negative transfer due to excessive specificity. Furthermore, we show that transfer effectiveness scales with the size of the memory pool, and memory can be transferred even between different models. Our work establishes empirical design principles for expanding memory utilization beyond single-domain silos. Project page: this https URL 

---
# RiskWebWorld: A Realistic Interactive Benchmark for GUI Agents in E-commerce Risk Management 

**Authors**: Renqi Chen, Zeyin Tao, Jianming Guo, Jing Wang, Zezhou Xu, Jingzhe Zhu, Qingqing Sun, Tianyi Zhang, Shuai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.13531)  

**Abstract**: Graphical User Interface (GUI) agents show strong capabilities for automating web tasks, but existing interactive benchmarks primarily target benign, predictable consumer environments. Their effectiveness in high-stakes, investigative domains such as authentic e-commerce risk management remains underexplored. To bridge this gap, we present RiskWebWorld, the first highly realistic interactive benchmark for evaluating GUI agents in e-commerce risk management. RiskWebWorld features 1,513 tasks sourced from production risk-control pipelines across 8 core domains, and captures the authentic challenges of risk operations on uncooperative websites, partially environmental hijackments. To support scalable evaluation and agentic reinforcement learning (RL), we further build a Gymnasium-compliant infrastructure that decouples policy planning from environment mechanics. Our evaluation across diverse models reveals a dramatic capability gap: top-tier generalist models achieve 49.1% success, while specialized open-weights GUI models lag at near-total failure. This highlights that foundation model scale currently matters more than zero-shot interface grounding in long-horizon professional tasks. We also demonstrate the viability of our infrastructure through agentic RL, which improves open-source models by 16.2%. These results position RiskWebWorld as a practical testbed for developing robust digital workers. 

---
# Towards Scalable Lightweight GUI Agents via Multi-role Orchestration 

**Authors**: Ziwei Wang, Junjie Zheng, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Zhouhua Fang, Zhiwei Liu, Dajun Chen, Yong Li, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13488)  

**Abstract**: Autonomous Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) enable digital automation on end-user devices. While scaling both parameters and data has yielded substantial gains, advanced methods still suffer from prohibitive deployment costs on resource-constrained devices. When facing complex in-the-wild scenarios, lightweight GUI agents are bottlenecked by limited capacity and poor task scalability under end-to-end episodic learning, impeding adaptation to multi-agent systems (MAS), while training multiple skill-specific experts remains costly. Can we strike an effective trade-off in this cost-scalability dilemma, enabling lightweight MLLMs to participate in realistic GUI workflows? To address these challenges, we propose the LAMO framework, which endows a lightweight MLLM with GUI-specific knowledge and task scalability, allowing multi-role orchestration to expand its capability boundary for GUI automation. LAMO combines role-oriented data synthesis with a two-stage training recipe: (i) supervised fine-tuning with Perplexity-Weighted Cross-Entropy optimization for knowledge distillation and visual perception enhancement, and (ii) reinforcement learning for role-oriented cooperative exploration. With LAMO, we develop a task-scalable native GUI agent, LAMO-3B, supporting monolithic execution and MAS-style orchestration. When paired with advanced planners as a plug-and-play policy executor, LAMO-3B can continuously benefit from planner advances, enabling a higher performance ceiling. Extensive static and online evaluations validate the effectiveness of our design. 

---
# SciFi: A Safe, Lightweight, User-Friendly, and Fully Autonomous Agentic AI Workflow for Scientific Applications 

**Authors**: Qibin Liu, Julia Gonski  

**Link**: [PDF](https://arxiv.org/pdf/2604.13180)  

**Abstract**: Recent advances in agentic AI have enabled increasingly autonomous workflows, but existing systems still face substantial challenges in achieving reliable deployment in real-world scientific research. In this work, we present a safe, lightweight, and user-friendly agentic framework for the autonomous execution of well-defined scientific tasks. The framework combines an isolated execution environment, a three-layer agent loop, and a self-assessing do-until mechanism to ensure safe and reliable operation while effectively leveraging large language models of varying capability levels. By focusing on structured tasks with clearly defined context and stopping criteria, the framework supports end-to-end automation with minimal human intervention, enabling researchers to offload routine workloads and devote more effort to creative activities and open-ended scientific inquiry. 

---
# From $P(y|x)$ to $P(y)$: Investigating Reinforcement Learning in Pre-train Space 

**Authors**: Yuqiao Tan, Minzheng Wang, Bo Liu, Zichen Liu, Tian Liang, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14142)  

**Abstract**: While reinforcement learning with verifiable rewards (RLVR) significantly enhances LLM reasoning by optimizing the conditional distribution P(y|x), its potential is fundamentally bounded by the base model's existing output distribution. Optimizing the marginal distribution P(y) in the Pre-train Space addresses this bottleneck by encoding reasoning ability and preserving broad exploration capacity. Yet, conventional pre-training relies on static corpora for passive learning, leading to a distribution shift that hinders targeted reasoning enhancement. In this paper, we introduce PreRL (Pre-train Space RL), which applies reward-driven online updates directly to P(y). We theoretically and empirically validate the strong gradient alignment between log P(y) and log P(y|x), establishing PreRL as a viable surrogate for standard RL. Furthermore, we uncover a critical mechanism: Negative Sample Reinforcement (NSR) within PreRL serves as an exceptionally effective driver for reasoning. NSR-PreRL rapidly prunes incorrect reasoning spaces while stimulating endogenous reflective behaviors, increasing transition and reflection thoughts by 14.89x and 6.54x, respectively. Leveraging these insights, we propose Dual Space RL (DSRL), a Policy Reincarnation strategy that initializes models with NSR-PreRL to expand the reasoning horizon before transitioning to standard RL for fine-grained optimization. Extensive experiments demonstrate that DSRL consistently outperforms strong baselines, proving that pre-train space pruning effectively steers the policy toward a refined correct reasoning subspace. 

---
# Exploration and Exploitation Errors Are Measurable for Language Model Agents 

**Authors**: Jaden Park, Jungtaek Kim, Jongwon Jeong, Robert D. Nowak, Kangwook Lee, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.13151)  

**Abstract**: Language Model (LM) agents are increasingly used in complex open-ended decision-making tasks, from AI coding to physical AI. A core requirement in these settings is the ability to both explore the problem space and exploit acquired knowledge effectively. However, systematically distinguishing and quantifying exploration and exploitation from observed actions without access to the agent's internal policy remains challenging. To address this, we design controllable environments inspired by practical embodied AI scenarios. Each environment consists of a partially observable 2D grid map and an unknown task Directed Acyclic Graph (DAG). The map generation can be programmatically adjusted to emphasize exploration or exploitation difficulty. To enable policy-agnostic evaluation, we design a metric to quantify exploration and exploitation errors from agent's actions. We evaluate a variety of frontier LM agents and find that even state-of-the-art models struggle on our task, with different models exhibiting distinct failure modes. We further observe that reasoning models solve the task more effectively and show both exploration and exploitation can be significantly improved through minimal harness engineering. We release our code \href{this https URL}{here}. 

---
# Numerical Instability and Chaos: Quantifying the Unpredictability of Large Language Models 

**Authors**: Chashi Mahiul Islam, Alan Villarreal, Mao Nishino, Shaeke Salman, Xiuwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13206)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into agentic workflows, their unpredictability stemming from numerical instability has emerged as a critical reliability issue. While recent studies have demonstrated the significant downstream effects of these instabilities, the root causes and underlying mechanisms remain poorly understood. In this paper, we present a rigorous analysis of how unpredictability is rooted in the finite numerical precision of floating-point representations, tracking how rounding errors propagate, amplify, or dissipate through Transformer computation layers. Specifically, we identify a chaotic "avalanche effect" in the early layers, where minor perturbations trigger binary outcomes: either rapid amplification or complete attenuation. Beyond specific error instances, we demonstrate that LLMs exhibit universal, scale-dependent chaotic behaviors characterized by three distinct regimes: 1) a stable regime, where perturbations fall below an input-dependent threshold and vanish, resulting in constant outputs; 2) a chaotic regime, where rounding errors dominate and drive output divergence; and 3) a signal-dominated regime, where true input variations override numerical noise. We validate these findings extensively across multiple datasets and model architectures. 

---
# From Feelings to Metrics: Understanding and Formalizing How Users Vibe-Test LLMs 

**Authors**: Itay Itzhak, Eliya Habba, Gabriel Stanovsky, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.14137)  

**Abstract**: Evaluating LLMs is challenging, as benchmark scores often fail to capture models' real-world usefulness. Instead, users often rely on ``vibe-testing'': informal experience-based evaluation, such as comparing models on coding tasks related to their own workflow. While prevalent, vibe-testing is often too ad hoc and unstructured to analyze or reproduce at scale. In this work, we study how vibe-testing works in practice and then formalize it to support systematic analysis. We first analyze two empirical resources: (1) a survey of user evaluation practices, and (2) a collection of in-the-wild model comparison reports from blogs and social media. Based on these resources, we formalize vibe-testing as a two-part process: users personalize both what they test and how they judge responses. We then introduce a proof-of-concept evaluation pipeline that follows this formulation by generating personalized prompts and comparing model outputs using user-aware subjective criteria. In experiments on coding benchmarks, we find that combining personalized prompts and user-aware evaluation can change which model is preferred, reflecting the role of vibe-testing in practice. These findings suggest that formalized vibe-testing can serve as a useful approach for bridging benchmark scores and real-world experience. 

---
# Quantifying and Understanding Uncertainty in Large Reasoning Models 

**Authors**: Yangyi Li, Chenxu Zhao, Mengdi Huai  

**Link**: [PDF](https://arxiv.org/pdf/2604.13395)  

**Abstract**: Large Reasoning Models (LRMs) have recently demonstrated significant improvements in complex reasoning. While quantifying generation uncertainty in LRMs is crucial, traditional methods are often insufficient because they do not provide finite-sample guarantees for reasoning-answer generation. Conformal prediction (CP) stands out as a distribution-free and model-agnostic methodology that constructs statistically rigorous uncertainty sets. However, existing CP methods ignore the logical connection between the reasoning trace and the final answer. Additionally, prior studies fail to interpret the origins of uncertainty coverage for LRMs as they typically overlook the specific training factors driving valid reasoning. Notably, it is challenging to disentangle reasoning quality from answer correctness when quantifying uncertainty, while simultaneously establishing theoretical guarantees for computationally efficient explanation methods. To address these challenges, we first propose a novel methodology that quantifies uncertainty in the reasoning-answer structure with statistical guarantees. Subsequently, we develop a unified example-to-step explanation framework using Shapley values that identifies a provably sufficient subset of training examples and their key reasoning steps to preserve the guarantees. We also provide theoretical analyses of our proposed methods. Extensive experiments on challenging reasoning datasets verify the effectiveness of the proposed methods. 

---
# Large Language Models to Enhance Business Process Modeling: Past, Present, and Future Trends 

**Authors**: João Bettencourt, Sérgio Guerreiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.14034)  

**Abstract**: Recent advances in Generative Artificial Intelligence, particularly Large Language Models (LLMs), have stimulated growing interest in automating or assisting Business Process Modeling tasks using natural language. Several approaches have been proposed to transform textual process descriptions into BPMN and related workflow models. However, the extent to which these approaches effectively support complex process modeling in organizational settings remains unclear. This article presents a literature review of AI-driven methods for transforming natural language into BPMN process models, with a particular focus on the role of LLMs. Following a structured review strategy, relevant studies were identified and analyzed to classify existing approaches, examine how LLMs are integrated into text-to-model pipelines, and investigate the evaluation practices used to assess generated models. The analysis reveals a clear shift from rule-based and traditional NLP pipelines toward LLM-based architectures that rely on prompt engineering, intermediate representations, and iterative refinement mechanisms. While these approaches significantly expand the capabilities of automated process model generation, the literature also exposes persistent challenges related to semantic correctness, evaluation fragmentation, reproducibility, and limited validation in real-world organizational contexts. Based on these findings, this review identifies key research gaps and discusses promising directions for future research, including the integration of contextual knowledge through Retrieval-Augmented Generation (RAG), its integration with LLMs, the development of interactive modeling architectures, and the need for more comprehensive and standardized evaluation frameworks. 

---
# MAny: Merge Anything for Multimodal Continual Instruction Tuning 

**Authors**: Zijian Gao, Wangwang Jia, Xingxing Zhang, Pengfei Qian, Tao Sun, Bo Ding, Yong Dou, Huaimin Wang, Kele Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14016)  

**Abstract**: Multimodal Continual Instruction Tuning (MCIT) is essential for sequential task adaptation of Multimodal Large Language Models (MLLMs) but is severely restricted by catastrophic forgetting. While existing literature focuses on the reasoning language backbone, in this work, we expose a critical yet neglected dual-forgetting phenomenon across both perception drift in Cross-modal Projection Space and reasoning collapse in Low-rank Parameter Space. To resolve this, we present \textbf{MAny} (\textbf{M}erge \textbf{Any}thing), a framework that merges task-specific knowledge through \textbf{C}ross-modal \textbf{P}rojection \textbf{M}erging (\textbf{CPM}) and \textbf{L}ow-rank \textbf{P}arameter \textbf{M}erging (\textbf{LPM}). Specifically, CPM recovers perceptual alignment by adaptively merging cross-modal visual representations via visual-prototype guidance, ensuring accurate feature recovery during inference. Simultaneously, LPM eliminates mutual interference among task-specific low-rank modules by recursively merging low-rank weight matrices. By leveraging recursive least squares, LPM provides a closed-form solution that mathematically guarantees an optimal fusion trajectory for reasoning stability. Notably, MAny operates as a training-free paradigm that achieves knowledge merging via efficient CPU-based algebraic operations, eliminating additional gradient-based optimization beyond initial tuning. Our extensive evaluations confirm the superior performance and robustness of MAny across multiple MLLMs and benchmarks. Specifically, on the UCIT benchmark, MAny achieves significant leads of up to 8.57\% and 2.85\% in final average accuracy over state-of-the-art methods across two different MLLMs, respectively. 

---
# LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning 

**Authors**: Sumeet Ramesh Motwani, Daniel Nichols, Charles London, Peggy Li, Fabio Pizzati, Acer Blake, Hasan Hammoud, Tavish McDonald, Akshat Naik, Alesia Ivanova, Vignesh Baskaran, Ivan Laptev, Ruben Glatt, Tal Ben-Nun, Philip Torr, Natasha Jaques, Ameya Prabhu, Brian Bartoldson, Bhavya Kailkhura, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2604.14140)  

**Abstract**: As language models are increasingly deployed for complex autonomous tasks, their ability to reason accurately over longer horizons becomes critical. An essential component of this ability is planning and managing a long, complex chain-of-thought (CoT). We introduce LongCoT, a scalable benchmark of 2,500 expert-designed problems spanning chemistry, mathematics, computer science, chess, and logic to isolate and directly measure the long-horizon CoT reasoning capabilities of frontier models. Problems consist of a short input with a verifiable answer; solving them requires navigating a graph of interdependent steps that span tens to hundreds of thousands of reasoning tokens. Each local step is individually tractable for frontier models, so failures reflect long-horizon reasoning limitations. At release, the best models achieve <10% accuracy (GPT 5.2: 9.8%; Gemini 3 Pro: 6.1%) on LongCoT, revealing a substantial gap in current capabilities. Overall, LongCoT provides a rigorous measure of long-horizon reasoning, tracking the ability of frontier models to reason reliably over extended periods. 

---
# HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System 

**Authors**: Tianshuo Yang, Guanyu Chen, Yutian Chen, Zhixuan Liang, Yitian Liu, Zanxin Chen, Chunpu Xu, Haotian Liang, Jiangmiao Pang, Yao Mu, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14125)  

**Abstract**: While end-to-end Vision-Language-Action (VLA) models offer a promising paradigm for robotic manipulation, fine-tuning them on narrow control data often compromises the profound reasoning capabilities inherited from their base Vision-Language Models (VLMs). To resolve this fundamental trade-off, we propose HiVLA, a visual-grounded-centric hierarchical framework that explicitly decouples high-level semantic planning from low-level motor control. In high-level part, a VLM planner first performs task decomposition and visual grounding to generate structured plans, comprising a subtask instruction and a precise target bounding box. Then, to translate this plan into physical actions, we introduce a flow-matching Diffusion Transformer (DiT) action expert in low-level part equipped with a novel cascaded cross-attention mechanism. This design sequentially fuses global context, high-resolution object-centric crops and skill semantics, enabling the DiT to focus purely on robust execution. Our decoupled architecture preserves the VLM's zero-shot reasoning while allowing independent improvement of both components. Extensive experiments in simulation and the real world demonstrate that HiVLA significantly outperforms state-of-the-art end-to-end baselines, particularly excelling in long-horizon skill composition and the fine-grained manipulation of small objects in cluttered scenes. 

---
# TIP: Token Importance in On-Policy Distillation 

**Authors**: Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2604.14084)  

**Abstract**: On-policy knowledge distillation (OPD) trains a student on its own rollouts under token-level supervision from a teacher. Not all token positions matter equally, but existing views of token importance are incomplete. We ask a direct question: which tokens carry the most useful learning signal in OPD? Our answer is that informative tokens come from two regions: positions with high student entropy, and positions with low student entropy plus high teacher--student divergence, where the student is overconfident and wrong.
Empirically, student entropy is a strong first-order proxy: retaining $50\%$ of tokens with entropy-based sampling matches or exceeds all-token training while reducing peak memory by up to $47\%$. But entropy alone misses a second important region. When we isolate low-entropy, high-divergence tokens, training on fewer than $10\%$ of all tokens nearly matches full-token baselines, showing that overconfident tokens carry dense corrective signal despite being nearly invisible to entropy-only rules.
We organize these findings with TIP (Token Importance in on-Policy distillation), a two-axis taxonomy over student entropy and teacher--student divergence, and give a theoretical explanation for why entropy is useful yet structurally incomplete. This view motivates type-aware token selection rules that combine uncertainty and disagreement. We validate this picture across three teacher--student pairs spanning Qwen3, Llama, and Qwen2.5 on MATH-500 and AIME 2024/2025, and on the DeepPlanning benchmark for long-horizon agentic planning, where Q3-only training on $<$$20\%$ of tokens surpasses full-token OPD. Our experiments are implemented by extending the OPD repository this https URL, which supports memory-efficient distillation of larger models under limited GPU budgets. 

---
# ASTER: Latent Pseudo-Anomaly Generation for Unsupervised Time-Series Anomaly Detection 

**Authors**: Romain Hermary, Samet Hicsonmez, Dan Pineau, Abd El Rahman Shabayek, Djamila Aouada  

**Link**: [PDF](https://arxiv.org/pdf/2604.13924)  

**Abstract**: Time-series anomaly detection (TSAD) is critical in domains such as industrial monitoring, healthcare, and cybersecurity, but it remains challenging due to rare and heterogeneous anomalies and the scarcity of labelled data. This scarcity makes unsupervised approaches predominant, yet existing methods often rely on reconstruction or forecasting, which struggle with complex data, or on embedding-based approaches that require domain-specific anomaly synthesis and fixed distance metrics. We propose ASTER, a framework that generates pseudo-anomalies directly in the latent space, avoiding handcrafted anomaly injections and the need for domain expertise. A latent-space decoder produces tailored pseudo-anomalies to train a Transformer-based anomaly classifier, while a pre-trained LLM enriches the temporal and contextual representations of this space. Experiments on three benchmark datasets show that ASTER achieves state-of-the-art performance and sets a new standard for LLM-based TSAD. 

---
# Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models 

**Authors**: Aleksandr Rubashevskii, Dzianis Piatrashyn, Preslav Nakov, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2604.13991)  

**Abstract**: Large language models (LLMs) are prone to generating factually incorrect outputs. Recent work has applied conformal prediction to provide uncertainty estimates and statistical guarantees for the factuality of LLM generations. However, existing approaches are typically not prompt-adaptive, limiting their ability to capture input-dependent variability. As a result, they may filter out too few items (leading to over-coverage) or too many (under-coverage) for a given task or prompt. We propose an adaptive conformal prediction approach that extends conformal score transformation methods to LLMs, with applications to long-form generation and multiple-choice question answering. This enables prompt-dependent calibration, retaining marginal coverage guarantees while improving conditional coverage. In addition, the approach naturally supports selective prediction, allowing unreliable claims or answer choices to be filtered out in downstream applications. We evaluate our approach on multiple white-box models across diverse domains and show that it significantly outperforms existing baselines in terms of conditional coverage. 

---
# SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention 

**Authors**: Hongtao Xu, Jianchao Tan, Yuxuan Hu, Pengju Lu, Hongyu Wang, Pingwei Sun, Yerui Sun, Yuchen Xie, Xunliang Cai, Mingzhen Li, Weile Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.13847)  

**Abstract**: While sparse attention mitigates the computational bottleneck of long-context LLM training, its distributed training process exhibits extreme heterogeneity in both \textit{1)} sequence length and \textit{2)} sparsity sensitivity, leading to a severe imbalance problem and sub-optimal model accuracy. Existing algorithms and training frameworks typically focus on single issue, failing to systematically co-optimize these two problems. Therefore, we propose SparseBalance, a novel algorithm-system co-design framework, which exploits the sparsity and sequence heterogeneity to optimize model accuracy and system efficiency jointly. First, we propose workload-aware dynamic sparsity tuning, which employs a bidirectional sparsity adjustment to eliminate stragglers and exploit inherent bubbles for free accuracy. Second, we propose a sparsity-aware batching strategy to achieve coarse-grained balance, which complements dynamic sparsity tuning. Experimental results demonstrate that SparseBalance achieves up to a 1.33$\times$ end-to-end speedup while still improving the long-context capability by 0.46\% on the LongBench benchmark. 

---
# HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark 

**Authors**: Jiacheng Wang, Jinchang Hou, Fabian Wang, Ping Jian, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.13954)  

**Abstract**: Existing agent-safety evaluation has focused mainly on externally induced risks. Yet agents may still enter unsafe trajectories under benign conditions. We study this complementary but underexplored setting through the lens of \emph{intrinsic} risk, where intrinsic failures remain latent, propagate across long-horizon execution, and eventually lead to high-consequence outcomes. To evaluate this setting, we introduce \emph{non-attack intrinsic risk auditing} and present \textbf{HINTBench}, a benchmark of 629 agent trajectories (523 risky, 106 safe; 33 steps on average) supporting three tasks: risk detection, risk-step localization, and intrinsic failure-type identification. Its annotations are organized under a unified five-constraint taxonomy. Experiments reveal a substantial capability gap: strong LLMs perform well on trajectory-level risk detection, but their performance drops to below 35 Strict-F1 on risk-step localization, while fine-grained failure diagnosis proves even harder. Existing guard models transfer poorly to this setting. These findings establish intrinsic risk auditing as an open challenge for agent safety. 

---
# How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data 

**Authors**: Joel Niklaus, Atsuki Yamaguchi, Michal Štefánik, Guilherme Penedo, Hynek Kydlíček, Elie Bakouch, Lewis Tunstall, Edward Emanuel Beeching, Thibaud Frere, Colin Raffel, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2604.13977)  

**Abstract**: Synthetic data is a standard component in training large language models, yet systematic comparisons across design dimensions, including rephrasing strategy, generator model, and source data, remain absent. We conduct extensive controlled experiments, generating over one trillion tokens, to identify critical factors in rephrasing web text into synthetic pretraining data. Our results reveal that structured output formats, such as tables, math problems, FAQs, and tutorials, consistently outperform both curated web baselines and prior synthetic methods. Notably, increasing the size of the generator model beyond 1B parameters provides no additional benefit. Our analysis also demonstrates that the selection of the original data used for mixing substantially influences performance. By applying our findings, we develop \textbf{\textsc{FinePhrase}}, a 486-billion-token open dataset of rephrased web text. We show that \textsc{FinePhrase} outperforms all existing synthetic data baselines while reducing generation costs by up to 30 times. We provide the dataset, all prompts, and the generation framework to the research community. 

---
# MCPThreatHive: Automated Threat Intelligence for Model Context Protocol Ecosystems 

**Authors**: Yi Ting Shen, Kentaroh Toyoda, Alex Leung  

**Link**: [PDF](https://arxiv.org/pdf/2604.13849)  

**Abstract**: The rapid proliferation of Model Context Protocol (MCP)-based agentic systems has introduced a new category of security threats that existing frameworks are inadequately equipped to address. We present MCPThreatHive, an open-source platform that automates the end-to-end lifecycle of MCP threat intelligence: from continuous, multi-source data collection through AI-driven threat extraction and classification, to structured knowledge graph storage and interactive visualization. The platform operationalizes the MCP-38 threat taxonomy, a curated set of 38 MCP-specific threat patterns mapped to STRIDE, OWASP Top 10 for LLM Applications, and OWASP Top 10 for Agentic Applications. A composite risk scoring model provides quantitative prioritization. Through a comparative analysis of representative existing MCP security tools, we identify three critical coverage gaps that MCPThreatHive addresses: incomplete compositional attack modeling, absence of continuous threat intelligence, and lack of unified multi-framework classification. 

---
# Do We Still Need Humans in the Loop? Comparing Human and LLM Annotation in Active Learning for Hostility Detection 

**Authors**: Ahmad Dawar Hakimi, Lea Hirlimann, Isabelle Augenstein, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2604.13899)  

**Abstract**: Instruction-tuned LLMs can annotate thousands of instances from a short prompt at negligible cost. This raises two questions for active learning (AL): can LLM labels replace human labels within the AL loop, and does AL remain necessary when entire corpora can be labelled at once? We investigate both questions on a new dataset of 277,902 German political TikTok comments (25,974 LLM-labelled, 5,000 human-annotated), comparing seven annotation strategies across four encoders to detect anti-immigrant hostility. A classifier trained on 25,974 GPT-5.2 labels (\$43) achieves comparable F1-Macro to one trained on 3,800 human annotations (\$316). Active learning offers little advantage over random sampling in our pre-enriched pool and delivers lower F1 than full LLM annotation at the same cost. However, comparable aggregate F1 masks a systematic difference in error structure: LLM-trained classifiers over-predict the positive class relative to the human gold standard. This divergence concentrates in topically ambiguous discussions where the distinction between anti-immigrant hostility and policy critique is most subtle, suggesting that annotation strategy should be guided not by aggregate F1 alone but by the error profile acceptable for the target application. 

---
# From Anchors to Supervision: Memory-Graph Guided Corpus-Free Unlearning for Large Language Models 

**Authors**: Wenxuan Li, Zhenfei Zhang, Mi Zhang, Geng Hong, Mi Wen, Xiaoyu You, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13777)  

**Abstract**: Large language models (LLMs) may memorize sensitive or copyrighted content, raising significant privacy and legal concerns. While machine unlearning has emerged as a potential remedy, prevailing paradigms rely on user-provided forget sets, making unlearning requests difficult to audit and exposing systems to secondary leakage and malicious abuse. We propose MAGE, a Memory-grAph Guided Erasure framework for user-minimized, corpus-free unlearning. Given only a lightweight user anchor that identifies a target entity, MAGE probes the target LLM to recover target-related memorization, organizes it into a weighted local memory graph, and synthesizes scoped supervision for unlearning. MAGE is model-agnostic, can be plugged into standard unlearning methods, and requires no access to the original training corpus. Experiments on two benchmarks, TOFU and RWKU, demonstrate that MAGE's self-generated supervision achieves effective unlearning performance comparable to supervision generated with external reference, while preserving overall utility. These results support a practical and auditable unlearning workflow driven by minimal anchors rather than user-supplied forget corpora. 

---
# Jump-Start Reinforcement Learning with Vision-Language-Action Regularization 

**Authors**: Angelo Moroncelli, Roberto Zanetti, Marco Maccarini, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2604.13733)  

**Abstract**: Reinforcement learning (RL) enables high-frequency, closed-loop control for robotic manipulation, but scaling to long-horizon tasks with sparse or imperfect rewards remains difficult due to inefficient exploration and poor credit assignment. Vision-Language-Action (VLA) models leverage large-scale multimodal pretraining to provide generalist, task-level reasoning, but current limitations hinder their direct use in fast and precise manipulation. In this paper, we propose Vision-Language-Action Jump-Starting (VLAJS), a method that bridges sparse VLA guidance with on-policy RL to improve exploration and learning efficiency. VLAJS treats VLAs as transient sources of high-level action suggestions that bias early exploration and improve credit assignment, while preserving the high-frequency, state-based control of RL. Our approach augments Proximal Policy Optimization (PPO) with a directional action-consistency regularization that softly aligns the RL agent's actions with VLA guidance during early training, without enforcing strict imitation, requiring demonstrations, or relying on continuous teacher queries. VLA guidance is applied sparsely and annealed over time, allowing the agent to adapt online and ultimately surpass the guiding policy. We evaluate VLAJS on six challenging manipulation tasks: lifting, pick-and-place, peg reorientation, peg insertion, poking, and pushing in simulation, and validate a subset on a real Franka Panda robot. VLAJS consistently outperforms PPO and distillation-style baselines in sample efficiency, reducing required environment interactions by over 50% in several tasks. Real-world experiments demonstrate zero-shot sim-to-real transfer and robust execution under clutter, object variation, and external perturbations. 

---
# Beyond Arrow's Impossibility: Fairness as an Emergent Property of Multi-Agent Collaboration 

**Authors**: Sayan Kumar Chaki, Antoine Gourru, Julien Velcin  

**Link**: [PDF](https://arxiv.org/pdf/2604.13705)  

**Abstract**: Fairness in language models is typically studied as a property of a single, centrally optimized model. As large language models become increasingly agentic, we propose that fairness emerges through interaction and exchange. We study this via a controlled hospital triage framework in which two agents negotiate over three structured debate rounds. One agent is aligned to a specific ethical framework via retrieval-augmented generation (RAG), while the other is either unaligned or adversarially prompted to favor demographic groups over clinical need. We find that alignment systematically shapes negotiation strategies and allocation patterns, and that neither agent's allocation is ethically adequate in isolation, yet their joint final allocation can satisfy fairness criteria that neither would have reached alone. Aligned agents partially moderate bias through contestation rather than override, acting as corrective patches that restore access for marginalized groups without fully converting a biased counterpart. We further observe that even explicitly aligned agents exhibit intrinsic biases toward certain frameworks, consistent with known left-leaning tendencies in LLMs. We connect these limits to Arrow's Impossibility Theorem: no aggregation mechanism can simultaneously satisfy all desiderata of collective rationality, and multi-agent deliberation navigates rather than resolves this constraint. Our results reposition fairness as an emergent, procedural property of decentralized agent interaction, and the system rather than the individual agent as the appropriate unit of evaluation. 

---
# Towards Fine-grained Temporal Perception: Post-Training Large Audio-Language Models with Audio-Side Time Prompt 

**Authors**: Yanfeng Shi, Pengfei Cai, Jun Liu, Qing Gu, Nan Jiang, Lirong Dai, Ian McLoughlin, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.13715)  

**Abstract**: Large Audio-Language Models (LALMs) enable general audio understanding and demonstrate remarkable performance across various audio tasks. However, these models still face challenges in temporal perception (e.g., inferring event onset and offset), leading to limited utility in fine-grained scenarios. To address this issue, we propose Audio-Side Time Prompt and leverage Reinforcement Learning (RL) to develop the TimePro-RL framework for fine-grained temporal perception. Specifically, we encode timestamps as embeddings and interleave them within the audio feature sequence as temporal coordinates to prompt the model. Furthermore, we introduce RL following Supervised Fine-Tuning (SFT) to directly optimize temporal alignment performance. Experiments demonstrate that TimePro-RL achieves significant performance gains across a range of audio temporal tasks, such as audio grounding, sound event detection, and dense audio captioning, validating its robust effectiveness. 

---
# Automatically Inferring Teachers' Geometric Content Knowledge: A Skills Based Approach 

**Authors**: Ziv Fenigstein, Kobi Gal, Avi Segal, Osama Swidan, Inbal Israel, Hassan Ayoob  

**Link**: [PDF](https://arxiv.org/pdf/2604.13666)  

**Abstract**: Assessing teachers' geometric content knowledge is essential for geometry instructional quality and student learning, but difficult to scale. The Van Hiele model characterizes geometric reasoning through five hierarchical levels. Traditional Van Hiele assessment relies on manual expert analysis of open-ended responses. This process is time-consuming, costly, and prevents large-scale evaluation. This study develops an automated approach for diagnosing teachers' Van Hiele reasoning levels using large language models grounded in educational theory. Our central hypothesis is that integrating explicit skills information significantly improves Van Hiele classification. In collaboration with mathematics education researchers, we built a structured skills dictionary decomposing the Van Hiele levels into 33 fine-grained reasoning skills. Through a custom web platform, 31 pre-service teachers solved geometry problems, yielding 226 responses. Expert researchers then annotated each response with its Van Hiele level and demonstrated skills from the dictionary. Using this annotated dataset, we implemented two classification approaches: (1) retrieval-augmented generation (RAG) and (2) multi-task learning (MTL). Each approach compared a skills-aware variant incorporating the skills dictionary against a baseline without skills information. Results showed that for both methods, skills-aware variants significantly outperformed baselines across multiple evaluation metrics. This work provides the first automated approach for Van Hiele level classification from open-ended responses. It offers a scalable, theory-grounded method for assessing teachers' geometric reasoning that can enable large-scale evaluation and support adaptive, personalized teacher learning systems. 

---
# MIND: AI Co-Scientist for Material Research 

**Authors**: Geonhee Ahn, Donghyun Lee, Hayoung Doo, Jonggeol Na, Hyunsoo Cho, Sookyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.13699)  

**Abstract**: Large language models (LLMs) have enabled agentic AI systems for scientific discovery, but most approaches remain limited to textbased reasoning without automated experimental verification. We propose MIND, an LLM-driven framework for automated hypothesis validation in materials research. MIND organizes the scientific discovery process into hypothesis refinement, experimentation, and debate-based validation within a multi-agent pipeline. For experimental verification, the system integrates Machine Learning Interatomic Potentials, particularly SevenNet-Omni, enabling scalable in-silico experiments. We also provide a web-based user interface for automated hypothesis testing. The modular design allows additional experimental modules to be integrated, making the framework adaptable to broader scientific workflows. The code is available at: this https URL, and a demonstration video at: this https URL. 

---
# SafeHarness: Lifecycle-Integrated Security Architecture for LLM-based Agent Deployment 

**Authors**: Xixun Lin, Yang Liu, Yancheng Chen, Yongxuan Wu, Yucheng Ning, Yilong Liu, Nan Sun, Shun Zhang, Bin Chong, Chuan Zhou, Yanan Cao, Li Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.13630)  

**Abstract**: The performance of large language model (LLM) agents depends critically on the execution harness, the system layer that orchestrates tool use, context management, and state persistence. Yet this same architectural centrality makes the harness a high-value attack surface: a single compromise at the harness level can cascade through the entire execution pipeline. We observe that existing security approaches suffer from structural mismatch, leaving them blind to harness-internal state and unable to coordinate across the different phases of agent operation. In this paper, we introduce \safeharness{}, a security architecture in which four proposed defense layers are woven directly into the agent lifecycle to address above significant limitations: adversarial context filtering at input processing, tiered causal verification at decision making, privilege-separated tool control at action execution, and safe rollback with adaptive degradation at state update. The proposed cross-layer mechanisms tie these layers together, escalating verification rigor, triggering rollbacks, and tightening tool privileges whenever sustained anomalies are detected. We evaluate \safeharness{} on benchmark datasets across diverse harness configurations, comparing against four security baselines under five attack scenarios spanning six threat categories. Compared to the unprotected baseline, \safeharness{} achieves an average reduction of approximately 38\% in UBR and 42\% in ASR, substantially lowering both the unsafe behavior rate and the attack success rate while preserving core task utility. 

---
# Syn-TurnTurk: A Synthetic Dataset for Turn-Taking Prediction in Turkish Dialogues 

**Authors**: Ahmet Tuğrul Bayrak, Mustafa Sertaç Türkel, Fatma Nur Korkmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.13620)  

**Abstract**: Managing natural dialogue timing is a significant challenge for voice-based chatbots. Most current systems usually rely on simple silence detection, which often fails because human speech patterns involve irregular pauses. This causes bots to interrupt users, breaking the conversational flow. This problem is even more severe for languages like Turkish, which lack high-quality datasets for turn-taking prediction. This paper introduces Syn-TurnTurk, a synthetic Turkish dialogue dataset generated using various Qwen Large Language Models (LLMs) to mirror real-life verbal exchanges, including overlaps and strategic silences. We evaluated the dataset using several traditional and deep learning architectures. The results show that advanced models, particularly BI-LSTM and Ensemble (LR+RF) methods, achieve high accuracy (0.839) and AUC scores (0.910). These findings demonstrate that our synthetic dataset can have a positive affect for models understand linguistic cues, allowing for more natural human-machine interaction in Turkish. 

---
# IndicDB -- Benchmarking Multilingual Text-to-SQL Capabilities in Indian Languages 

**Authors**: Aviral Dawar, Roshan Karanth, Vikram Goyal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13686)  

**Abstract**: While Large Language Models (LLMs) have significantly advanced Text-to-SQL performance, existing benchmarks predominantly focus on Western contexts and simplified schemas, leaving a gap in real-world, non-Western applications. We present IndicDB, a multilingual Text-to-SQL benchmark for evaluating cross-lingual semantic parsing across diverse Indic languages. The relational schemas are sourced from open-data platforms, including the National Data and Analytics Platform (NDAP) and the India Data Portal (IDP), ensuring realistic administrative data complexity. IndicDB comprises 20 databases across 237 tables. To convert denormalized government data into rich relational structures, we employ an iterative three-agent framework (Architect, Auditor, Refiner) to ensure structural rigor and high relational density (11.85 tables per database; join depths up to six). Our pipeline is value-aware, difficulty-calibrated, and join-enforced, generating 15,617 tasks across English, Hindi, and five Indic languages. We evaluate cross-lingual semantic parsing performance of state-of-the-art models (DeepSeek v3.2, MiniMax 2.7, LLaMA 3.3, Qwen3) across seven linguistic variants. Results show a 9.00% performance drop from English to Indic languages, revealing an "Indic Gap" driven by harder schema linking, increased structural ambiguity, and limited external knowledge. IndicDB serves as a rigorous benchmark for multilingual Text-to-SQL. Code and data: this https URL 

---
# BenGER: A Collaborative Web Platform for End-to-End Benchmarking of German Legal Tasks 

**Authors**: Sebastian Nagl, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2604.13583)  

**Abstract**: Evaluating large language models (LLMs) for legal reasoning requires workflows that span task design, expert annotation, model execution, and metric-based evaluation. In practice, these steps are split across platforms and scripts, limiting transparency, reproducibility, and participation by non-technical legal experts. We present the BenGER (Benchmark for German Law) framework, an open-source web platform that integrates task creation, collaborative annotation, configurable LLM runs, and evaluation with lexical, semantic, factual, and judge-based metrics. BenGER supports multi-organization projects with tenant isolation and role-based access control, and can optionally provide formative, reference-grounded feedback to annotators. We will demonstrate a live deployment showing end-to-end benchmark creation and analysis. 

---
# Training-Free Test-Time Contrastive Learning for Large Language Models 

**Authors**: Kaiwen Zheng, Kai Zhou, Jinwu Hu, Te Gu, Mingkai Peng, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13552)  

**Abstract**: Large language models (LLMs) demonstrate strong reasoning capabilities, but their performance often degrades under distribution shift. Existing test-time adaptation (TTA) methods rely on gradient-based updates that require white-box access and need substantial overhead, while training-free alternatives are either static or depend on external guidance. In this paper, we propose Training-Free Test-Time Contrastive Learning TF-TTCL, a training-free adaptation framework that enables a frozen LLM to improve online by distilling supervision from its own inference experiences. Specifically, TF-TTCL implements a dynamic "Explore-Reflect-Steer" loop through three core modules: 1) Semantic Query Augmentation first diversifies problem views via multi-agent role-playing to generate different reasoning trajectories; 2) Contrastive Experience Distillation then captures the semantic gap between superior and inferior trajectories, distilling them into explicit textual rules; and 3) Contextual Rule Retrieval finally activates these stored rules during inference to dynamically steer the frozen LLM toward robust reasoning patterns while avoiding observed errors. Extensive experiments on closed-ended reasoning tasks and open-ended evaluation tasks demonstrate that TF-TTCL consistently outperforms strong zero-shot baselines and representative TTA methods under online evaluation. Code is available at this https URL. 

---
# Leveraging LLM-GNN Integration for Open-World Question Answering over Knowledge Graphs 

**Authors**: Hussein Abdallah, Ibrahim Abdelaziz, Panos Kalnis, Essam Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2604.13979)  

**Abstract**: Open-world Question Answering (OW-QA) over knowledge graphs (KGs) aims to answer questions over incomplete or evolving KGs. Traditional KGQA assumes a closed world where answers must exist in the KG, limiting real-world applicability. In contrast, open-world QA requires inferring missing knowledge based on graph structure and context. Large language models (LLMs) excel at language understanding but lack structured reasoning. Graph neural networks (GNNs) model graph topology but struggle with semantic interpretation. Existing systems integrate LLMs with GNNs or graph retrievers. Some support open-world QA but rely on structural embeddings without semantic grounding. Most assume observed paths or complete graphs, making them unreliable under missing links or multi-hop reasoning. We present GLOW, a hybrid system that combines a pre-trained GNN and an LLM for open-world KGQA. The GNN predicts top-k candidate answers from the graph structure. These, along with relevant KG facts, are serialized into a structured prompt (e.g., triples and candidates) to guide the LLM's reasoning. This enables joint reasoning over symbolic and semantic signals, without relying on retrieval or fine-tuning. To evaluate generalization, we introduce GLOW-BENCH, a 1,000-question benchmark over incomplete KGs across diverse domains. GLOW outperforms existing LLM-GNN systems on standard benchmarks and GLOW-BENCH, achieving up to 53.3% and an average 38% improvement. GitHub code and data are available. 

---
# Free Lunch for Unified Multimodal Models: Enhancing Generation via Reflective Rectification with Inherent Understanding 

**Authors**: Yibo Jiang, Tao Wu, Rui Jiang, Yehao Lu, Chaoxiang Cai, Zequn Qin, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.13540)  

**Abstract**: Unified Multimodal Models (UMMs) aim to integrate visual understanding and generation within a single structure. However, these models exhibit a notable capability mismatch, where their understanding capability significantly outperforms their generation. This mismatch indicates that the model's rich internal knowledge, while effective for understanding tasks, remains underactivated during generation. To address this, we draw inspiration from the human ``Thinking-While-Drawing'' paradigm, where humans continuously reflect to activate their knowledge and rectify intermediate results. In this paper, we propose UniRect-CoT, a training-free unified rectification chain-of-thought framework. Our approach unlocks the ``free lunch'' hidden in the UMM's powerful inherent understanding to continuously reflect, activating its internal knowledge and rectifying intermediate results during this http URL regard the diffusion denoising process in UMMs as an intrinsic visual reasoning process and align the intermediate results with the target instruction understood by the model, serving as a self-supervisory signal to rectify UMM this http URL experiments demonstrate that UniRect-CoT can be easily integrated into existing UMMs, significantly enhancing generation quality across diverse complex tasks. 

---
# SFT-GRPO Data Overlap as a Post-Training Hyperparameter for Autoformalization 

**Authors**: Xiaole Su, Kasey Zhang, Andy Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13515)  

**Abstract**: Supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) is a common post-training recipe. We conduct a controlled ablation over SFT-GRPO data overlap, evaluating Qwen3-8B (thinking disabled) post-trained for Lean 4 autoformalization under six conditions that differ solely in training recipe: a base model, SFT-only, GRPO-only, and three SFT+GRPO configurations where 0 percent, 30 percent, or 100 percent of the GRPO prompts coincide with the SFT corpus. Keeping SFT and GRPO data disjoint consistently outperforms full overlap at zero additional compute cost. Evaluating on Gaokao-Formal and PutnamBench under both compile pass at k and semantic pass at k assessed by an LLM judge, we find that lower overlap is monotonically associated with higher compilation and semantic accuracy. At 0 percent overlap, GRPO yields a 10.4 percentage point semantic gain over SFT alone on Gaokao, while at 100 percent overlap both metrics remain flat, rendering the GRPO stage effectively redundant. We further show that dual-metric evaluation reveals compile semantic gaps exceeding 30 percentage points for the highest compiling models, a disparity invisible under compile-only benchmarking. To our knowledge, this is the first controlled investigation of SFT-GRPO data overlap as a post-training hyperparameter, demonstrating how model behavior varies based on the degree of data sharing between training stages. 

---
# Functional Emotions or Situational Contexts? A Discriminating Test from the Mythos Preview System Card 

**Authors**: Hiranya V. Peiris  

**Link**: [PDF](https://arxiv.org/pdf/2604.13466)  

**Abstract**: The Claude Mythos Preview system card deploys emotion vectors, sparse autoencoder (SAE) features, and activation verbalisers to study model internals during misaligned behaviour. The two primary toolkits are not jointly reported on the most alignment-relevant episodes. This note identifies two hypotheses that are qualitatively consistent with the published results: that the emotion vectors track functional emotions that causally drive behaviour, or that they are a projection of a richer situational-context structure onto human emotional axes. The hypotheses can be distinguished by a test the system card does not report: applying emotion probes to the strategic concealment episodes where only SAE features are currently documented. If emotion probes show flat activation while SAE features are strongly active, the alignment-relevant structure lies outside the emotion subspace. Which hypothesis is correct determines whether emotion-based monitoring will robustly detect dangerous model behaviour or systematically miss it. 

---
# Chain of Uncertain Rewards with Large Language Models for Reinforcement Learning 

**Authors**: Shentong Mo  

**Link**: [PDF](https://arxiv.org/pdf/2604.13504)  

**Abstract**: Designing effective reward functions is a cornerstone of reinforcement learning (RL), yet it remains a challenging and labor-intensive process due to the inefficiencies and inconsistencies inherent in traditional methods. Existing methods often rely on extensive manual design and evaluation steps, which are prone to redundancy and overlook local uncertainties at intermediate decision points. To address these challenges, we propose the Chain of Uncertain Rewards (CoUR), a novel framework that integrates large language models (LLMs) to streamline reward function design and evaluation in RL environments. Specifically, our CoUR introduces code uncertainty quantification with a similarity selection mechanism that combines textual and semantic analyses to identify and reuse the most relevant reward function components. By reducing redundant evaluations and leveraging Bayesian optimization on decoupled reward terms, CoUR enables a more efficient and robust search for optimal reward feedback. We comprehensively evaluate CoUR across nine original environments from IsaacGym and all 20 tasks from the Bidexterous Manipulation benchmark. The experimental results demonstrate that CoUR not only achieves better performance but also significantly lowers the cost of reward evaluations. 

---
# A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models 

**Authors**: Jason Kong, Nilesh Prasad Pandey, Flavio Ponzina, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2604.13440)  

**Abstract**: Deploying Large Language Models (LLMs) on edge devices faces severe computational and memory constraints, limiting real-time processing and on-device intelligence. Hybrid architectures combining Structured State Space Models (SSMs) with transformer-based LLMs offer a balance of efficiency and performance. Aggressive quantization can drastically cut model size and speed up inference, but its uneven effects on different components require careful management. In this work, we propose a lightweight, backpropagation-free, surrogate-based sensitivity analysis framework to identify hybrid SSM-Transformer components most susceptible to quantization-induced degradation. Relying solely on forward-pass metrics, our method avoids expensive gradient computations and retraining, making it suitable for situations where access to in-domain data is limited due to proprietary restrictions or privacy constraints. We also provide a formal analysis showing that the Kullback-Leibler (KL) divergence metric better captures quantization sensitivity for Language modeling tasks than widely adopted alternatives such as mean squared error (MSE) and signal-to-quantization-noise ratio (SQNR). Through extensive experiments on SSM and hybrid architectures, our ablation studies confirm that KL-based rankings align with observed performance drops and outperform alternative metrics. This framework enables the practical deployment of advanced hybrid models on resource-constrained edge devices with minimal accuracy loss. We further validate our approach with real-world on-device profiling on Intel Lunar Lake hardware, demonstrating that KL-guided mixed-precision achieves near-FP16 perplexity with model sizes and throughput competitive with Uniform INT4 on both CPU and GPU execution modes. Code is available at this https URL. 

---
# MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments 

**Authors**: Han Wang, David Wan, Hyunji Lee, Thinh Pham, Mikaela Cankosyan, Weiyuan Chen, Elias Stengel-Eskin, Tu Vu, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.13418)  

**Abstract**: Motivated by the underspecified, multi-hop nature of search queries and the multimodal, heterogeneous, and often conflicting nature of real-world web results, we introduce MERRIN (Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments), a human-annotated benchmark for evaluating search-augmented agents. MERRIN measures AI agents' ability to identify relevant modalities, retrieve multimodal evidence, and perform multi-hop reasoning over noisy web sources. It differs from prior work in three important aspects: (1) using natural language queries without explicit modality cues, (2) incorporating underexplored modalities such as video and audio, and (3) requiring the retrieval of complex, often noisy or conflicting multimodal evidence during web search. We evaluate diverse search agents powered by ten models, including strong closed-source models (e.g., GPT-5.4-mini, Gemini 3/3.1 Flash/Pro) and open-weight models (Qwen3-4B/30B/235B), across three search settings (no search, native search, and agentic search). Our results show that MERRIN is highly challenging: the average accuracy across all agents is 22.3%, with the best-performing agent reaching only 40.1%. We further observe that while stronger agents like Gemini Deep Research achieve higher performance, gains are modest due to over-exploration; they take more steps and use more tools, but are often distracted by conflicting or partially relevant web content, leading to incorrect answers. Compared to humans, these agents consume more resources yet achieve lower accuracy, largely due to inefficient source selection and an overreliance on text modalities. These findings highlight the need for search agents capable of robust search and reasoning across diverse modalities in noisy web environments, making MERRIN a valuable testbed for evaluating such capabilities. 

---
# The Cognitive Circuit Breaker: A Systems Engineering Framework for Intrinsic AI Reliability 

**Authors**: Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13417)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in mission-critical software systems, detecting hallucinations and ``faked truthfulness'' has become a paramount engineering challenge. Current reliability architectures rely heavily on post-generation, black-box mechanisms, such as Retrieval-Augmented Generation (RAG) cross-checking or LLM-as-a-judge evaluators. These extrinsic methods introduce unacceptable latency, high computational overhead, and reliance on secondary external API calls, frequently violating standard software engineering Service Level Agreements (SLAs). In this paper, we propose the Cognitive Circuit Breaker, a novel systems engineering framework that provides intrinsic reliability monitoring with minimal latency overhead. By extracting hidden states during a model's forward pass, we calculate the ``Cognitive Dissonance Delta'' -- the mathematical gap between an LLM's outward semantic confidence (softmax probabilities) and its internal latent certainty (derived via linear probes). We demonstrate statistically significant detection of cognitive dissonance, highlight architecture-dependent Out-of-Distribution (OOD) generalization, and show that this framework adds negligible computational overhead to the active inference pipeline. 

---
# From Prediction to Justification: Aligning Sentiment Reasoning with Human Rationale via Reinforcement Learning 

**Authors**: Shihao Zhang, Ziwei Wang, Jie Zhou, Yulan Wu, Qin Chen, Zhikai Lei, Liyang Yu, Liang Dou, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.13398)  

**Abstract**: While Aspect-based Sentiment Analysis (ABSA) systems have achieved high accuracy in identifying sentiment polarities, they often operate as "black boxes," lacking the explicit reasoning capabilities characteristic of human affective cognition. Humans do not merely categorize sentiment; they construct causal explanations for their judgments. To bridge this gap, we propose ABSA-R1, a large language model framework designed to mimic this ``reason-before-predict" cognitive process. By leveraging reinforcement learning (RL), ABSA-R1 learns to articulate the why behind the what, generating natural language justifications that ground its sentiment predictions. We introduce a Cognition-Aligned Reward Model (formerly sentiment-aware reward model) that enforces consistency between the generated reasoning path and the final emotional label. Furthermore, inspired by metacognitive monitoring, we implement a performance-driven rejection sampling strategy that selectively targets hard cases where the model's internal reasoning is uncertain or inconsistent. Experimental results on four benchmarks demonstrate that equipping models with this explicit reasoning capability not only enhances interpretability but also yields superior performance in sentiment classification and triplet extraction compared to non-reasoning baselines. 

---
# Young people's perceptions and recommendations for conversational generative artificial intelligence in youth mental health 

**Authors**: Adam Poulsen, Ian B. Hickie, Carla Gorban, Zsofi de Haan, William Capon, Ebenezer Eyeson-Annan, Jalal Radwan, Elizabeth M. Scott, Frank Iorfino, Haley M. LaMonica  

**Link**: [PDF](https://arxiv.org/pdf/2604.13381)  

**Abstract**: Conversational generative artificial intelligence agents (or genAI chatbots) could benefit youth mental health, yet young people's perspectives remain underexplored. We examined the Mental health Intelligence Agent (Mia), a genAI chatbot originally designed for professionals in Australian youth services. Following co-design, 32 young people participated in online workshops exploring their perceptions of genAI chatbots in youth mental health and to develop recommendations for reconceptualising Mia for consumers and integrating it into services. Four themes were developed: (1) Humanising AI without dehumanising care, (2) I need to know what's under the hood, (3) Right tool, right place, right time?, and (4) Making it mine on safe ground. This study offers insights into young people's attitudes, needs, and requirements regarding genAI chatbots in youth mental health, with key implications for service integration. Additionally, by co-designing system requirements, this work informs the ethics, design, development, implementation, and governance of genAI chatbots in youth mental health contexts. 

---
# English is Not All You Need: Systematically Exploring the Role of Multilinguality in LLM Post-Training 

**Authors**: Mehak Dhaliwal, Shashwat Chaurasia, Yao Qin, Dezhi Hong, Thomas Butler  

**Link**: [PDF](https://arxiv.org/pdf/2604.13286)  

**Abstract**: Despite the widespread multilingual deployment of large language models, post-training pipelines remain predominantly English-centric, contributing to performance disparities across languages. We present a systematic, controlled study of the interplay between training language coverage, model scale, and task domain, based on 220 supervised fine-tuning runs on parallel translated multilingual data mixtures spanning mathematical reasoning and API calling tasks, with models up to 8B parameters. We find that increasing language coverage during post-training is largely beneficial across tasks and model scales, with low-resource languages benefiting the most and high-resource languages plateauing rather than degrading. Even minimal multilinguality helps: incorporating a single non-English language improves both English performance and cross-lingual generalization, making English-only post-training largely suboptimal. Moreover, at sufficient language diversity, zero-shot cross-lingual transfer can match or exceed the effects of direct language inclusion in a low-diversity setting, although gains remain limited for typologically distant, low-resource languages. 

---
# L2D-Clinical: Learning to Defer for Adaptive Model Selection in Clinical Text Classification 

**Authors**: Rishik Kondadadi, John E. Ortega  

**Link**: [PDF](https://arxiv.org/pdf/2604.13285)  

**Abstract**: Clinical text classification requires choosing between specialized fine-tuned models (BERT variants) and general-purpose large language models (LLMs), yet neither dominates across all instances. We introduce Learning to Defer for clinical text (L2D-Clinical), a framework that learns when a BERT classifier should defer to an LLM based on uncertainty signals and text characteristics. Unlike prior L2D work that defers to human experts assumed universally superior, our approach enables adaptive deferral-improving accuracy when the LLM complements BERT. We evaluate on two English clinical tasks: (1) ADE detection (ADE Corpus V2), where BioBERT (F1=0.911) outperforms the LLM (F1=0.765), and (2) treatment outcome classification (MIMIC-IV with multi-LLM consensus ground truth), where GPT-5-nano (F1=0.967) outperforms ClinicalBERT (F1=0.887). On ADE, L2D-Clinical achieves F1=0.928 (+1.7 points over BERT) by selectively deferring 7% of instances where the LLM's high recall compensates for BERT's misses. On MIMIC, L2D-Clinical achieves F1=0.980 (+9.3 points over BERT) by deferring only 16.8\% of cases to the LLM. The key insight is that L2D-Clinical learns to selectively leverage LLM strengths while minimizing API costs. 

---
# Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs 

**Authors**: Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.13258)  

**Abstract**: Attribution methods seek to explain language model predictions by quantifying the contribution of input tokens to generated outputs. However, most existing techniques are designed for encoder-based architectures and rely on linear approximations that fail to capture the causal and semantic complexities of autoregressive generation in decoder-only models. To address these limitations, we propose Hessian-Enhanced Token Attribution (HETA), a novel attribution framework tailored for decoder-only language models. HETA combines three complementary components: a semantic transition vector that captures token-to-token influence across layers, Hessian-based sensitivity scores that model second-order effects, and KL divergence to measure information loss when tokens are masked. This unified design produces context-aware, causally faithful, and semantically grounded attributions. Additionally, we introduce a curated benchmark dataset for systematically evaluating attribution quality in generative settings. Empirical evaluations across multiple models and datasets demonstrate that HETA consistently outperforms existing methods in attribution faithfulness and alignment with human annotations, establishing a new standard for interpretability in autoregressive language models. 

---
# Lazy or Efficient? Towards Accessible Eye-Tracking Event Detection Using LLMs 

**Authors**: Dongyang Guo, Yasmeen Abdrabou, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2604.13243)  

**Abstract**: Gaze event detection is fundamental to vision science, human-computer interaction, and applied analytics. However, current workflows often require specialized programming knowledge and careful handling of heterogeneous raw data formats. Classical detectors such as I-VT and I-DT are effective but highly sensitive to preprocessing and parameterization, limiting their usability outside specialized laboratories. This work introduces a code-free, large language model (LLM)-driven pipeline that converts natural language instructions into an end-to-end analysis. The system (1) inspects raw eye-tracking files to infer structure and metadata, (2) generates executable routines for data cleaning and detector implementation from concise user prompts, (3) applies the generated detector to label fixations and saccades, and (4) returns results and explanatory reports, and allows users to iteratively optimize their code by editing the prompt. Evaluated on public benchmarks, the approach achieves accuracy comparable to traditional methods while substantially reducing technical overhead. The framework lowers barriers to entry for eye-tracking research, providing a flexible and accessible alternative to code-intensive workflows. 

---
# KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs 

**Authors**: Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, Ulf Schlichtmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.13226)  

**Abstract**: Large Language Models (LLMs) rely heavily on Key-Value (KV) caching to minimize inference latency. However, standard KV caches are context-dependent: reusing a cached document in a new context requires recomputing KV states to account for shifts in attention distribution. Existing solutions such as CacheBlend, EPIC, and SAM-KV mitigate this issue by selectively recomputing a subset of tokens; however, they still incur non-negligible computational overhead (FLOPs) and increased Time-to-First-Token (TTFT) latency. In this paper, we propose KV Packet, a recomputation-free cache reuse framework that treats cached documents as immutable ``packets'' wrapped in light-weight trainable soft-token adapters, which are trained via self-supervised distillation to bridge context discontinuities. Experiments on Llama-3.1 and Qwen2.5 demonstrate that the proposed KV Packet method achieves near-zero FLOPs and lower TTFT than recomputation-based baselines, while retaining F1 scores comparable to those of the full recomputation baseline. 

---
# On the Creativity of AI Agents 

**Authors**: Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2604.13242)  

**Abstract**: Large language models (LLMs), particularly when integrated into agentic systems, have demonstrated human- and even superhuman-level performance across multiple domains. Whether these systems can truly be considered creative, however, remains a matter of debate, as conclusions heavily depend on the definitions, evaluation methods, and specific use cases employed. In this paper, we analyse creativity along two complementary macro-level perspectives. The first is a functionalist perspective, focusing on the observable characteristics of creative outputs. The second is an ontological perspective, emphasising the underlying processes, as well as the social and personal dimensions involved in creativity. We focus on LLM agents and we argue that they exhibit functionalist creativity, albeit not at its most sophisticated levels, while they continue to lack key aspects of ontological creativity. Finally, we discuss whether it is desirable for agentic systems to attain both forms of creativity, evaluating potential benefits and risks, and proposing pathways toward artificial creativity that can enhance human society. 

---
# SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation 

**Authors**: Shivam Chand Kaushik  

**Link**: [PDF](https://arxiv.org/pdf/2604.13236)  

**Abstract**: Semiconductor failure analysis (FA) requires engineers to examine inspection images, correlate equipment telemetry, consult historical defect records, and write structured reports, a process that can consume several hours of expert time per case. We present SemiFA, an agentic multi-modal framework that autonomously generates structured FA reports from semiconductor inspection images in under one minute. SemiFA decomposes FA into a four-agent LangGraph pipeline: a DefectDescriber that classifies and narrates defect morphology using DINOv2 and LLaVA-1.6, a RootCauseAnalyzer that fuses SECS/GEM equipment telemetry with historically similar defects retrieved from a Qdrant vector database, a SeverityClassifier that assigns severity and estimates yield impact, and a RecipeAdvisor that proposes corrective process adjustments. A fifth node assembles a PDF report. We introduce SemiFA-930, a dataset of 930 annotated semiconductor defect images paired with structured FA narratives across nine defect classes, drawn from procedural synthesis, WM-811K, and MixedWM38. Our DINOv2-based classifier achieves 92.1% accuracy on 140 validation images (macro F1 = 0.917), and the full pipeline produces complete FA reports in 48 seconds on an NVIDIA A100-SXM4-40 GB GPU. A GPT-4o judge ablation across four modality conditions demonstrates that multi-modal fusion improves root cause reasoning by +0.86 composite points (1-5 scale) over an image-only baseline, with equipment telemetry as the more load-bearing modality. To our knowledge, SemiFA is the first system to integrate SECS/GEM equipment telemetry into a vision-language model pipeline for autonomous FA report generation. 

---
# AgentForge: Execution-Grounded Multi-Agent LLM Framework for Autonomous Software Engineering 

**Authors**: Rajesh Kumar, Waqar Ali, Junaid Ahmed, Najma Imtiaz Ali, Shaban Usman  

**Link**: [PDF](https://arxiv.org/pdf/2604.13120)  

**Abstract**: Large language models generate plausible code but cannot verify correctness. Existing multi-agent systems simulate execution or leave verification optional. We introduce execution-grounded verification as a first-class principle: every code change must survive sandboxed execution before propagation. We instantiate this principle in AGENTFORGE, a multi-agent framework where Planner, Coder, Tester, Debugger, and Critic agents coordinate through shared memory and a mandatory Docker sandbox. We formalize software engineering with LLMs as an iterative decision process over repository states, where execution feedback provides a stronger supervision signal than next-token likelihood. AGENTFORGE achieves 40.0\% resolution on SWE-BENCH Lite, outperforming single-agent baselines by 26--28 points. Ablations confirm that execution feedback and role decomposition each independently drive performance. The framework is open-source at this https URL. 

---
# The Code Whisperer: LLM and Graph-Based AI for Smell and Vulnerability Resolution 

**Authors**: Mohammad Baqar, Raji Rustamov, Alexander Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2604.13114)  

**Abstract**: Code smells and software vulnerabilities both increase maintenance cost, yet they are often handled by separate tools that miss structural context and produce noisy warnings. This paper presents The Code Whisperer, a hybrid framework that combines graph-based program analysis with large language models to detect, explain, and repair maintainability and security issues within a unified workflow. The method aligns Abstract Syntax Trees (ASTs), Control Flow Graphs (CFGs), Program Dependency Graphs (PDGs), and token-level code embeddings so that structural and semantic signals can be learned jointly. We evaluate the framework on multi-language datasets and compare it with rule-based analyzers and single-model baselines. The results indicate that the hybrid design improves detection performance and produces more useful repair suggestions than either graph-only or language-model-only approaches. We also examine explainability and CI/CD integration as practical requirements for adopting AI-assisted code review in everyday software engineering workflows. 

---
# Formal Architecture Descriptors as Navigation Primitives for AI Coding Agents 

**Authors**: Ruoqi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.13108)  

**Abstract**: AI coding agents spend a substantial fraction of their tool calls on undirected codebase exploration. We investigate whether providing agents with formal architecture descriptors can reduce this navigational overhead. We present three complementary studies. First, a controlled experiment (24 code localization tasks x 4 conditions, Claude Sonnet 4.6, temperature=0) demonstrates that architecture context reduces navigation steps by 33-44% (Wilcoxon p=0.009, Cohen's d=0.92), with no significant format difference detected across S-expression, JSON, YAML, and Markdown. Second, an artifact-vs-process experiment (15 tasks x 3 conditions) demonstrates that an automatically generated descriptor achieves 100% accuracy versus 80% blind (p=0.002, d=1.04), proving direct navigational value independent of developer self-clarification. Third, an observational field study across 7,012 Claude Code sessions shows 52% reduction in agent behavioral variance. A writer-side experiment (96 generation runs, 96 error injections) reveals critical failure mode differences: JSON fails atomically, YAML silently corrupts 50% of errors, S-expressions detect all structural completeness errors. We propose this http URL, an S-expression architecture descriptor, and open-source the Forge toolkit. 

---
# Pareto-Optimal Offline Reinforcement Learning via Smooth Tchebysheff Scalarization 

**Authors**: Aadyot Bhatnagar, Peter Mørch Groth, Ali Madani  

**Link**: [PDF](https://arxiv.org/pdf/2604.13175)  

**Abstract**: Large language models can be aligned with human preferences through offline reinforcement learning (RL) on small labeled datasets. While single-objective alignment is well-studied, many real-world applications demand the simultaneous optimization of multiple conflicting rewards, e.g. optimizing both catalytic activity and specificity in protein engineering, or helpfulness and harmlessness for chatbots. Prior work has largely relied on linear reward scalarization, but this approach provably fails to recover non-convex regions of the Pareto front. In this paper, instead of scalarizing the rewards directly, we frame multi-objective RL itself as an optimization problem to be scalarized via smooth Tchebysheff scalarization, a recent technique that overcomes the shortcomings of linear scalarization. We use this formulation to derive Smooth Tchebysheff Optimization of Multi-Objective Preferences (STOMP), a novel offline RL algorithm that extends direct preference optimization to the multi-objective setting in a principled way by standardizing the individual rewards based on their observed distributions. We empirically validate STOMP on a range of protein engineering tasks by aligning three autoregressive protein language models on three laboratory datasets of protein fitness. Compared to state-of-the-art baselines, STOMP achieves the highest hypervolumes in eight of nine settings according to both offline off-policy and generative evaluations. We thus demonstrate that STOMP is a powerful, robust multi-objective alignment algorithm that can meaningfully improve post-trained models for multi-attribute protein optimization and beyond. 

---
# Applying an Agentic Coding Tool for Improving Published Algorithm Implementations 

**Authors**: Worasait Suwannik  

**Link**: [PDF](https://arxiv.org/pdf/2604.13109)  

**Abstract**: We present a two-stage pipeline for AI-assisted improvement of published algorithm implementations. In the first stage, a large language model with research capabilities identifies recently published algorithms satisfying explicit experimental criteria. In the second stage, Claude Code is given a prompt to reproduce the reported baseline and then iterate an improvement process. We apply this pipeline to published algorithm implementations spanning multiple research domains. Claude Code reported that all eleven experiments yielded improvements. Each improvement could be achieved within a single working day. We analyse the human contributions that remain indispensable, including selecting the target, verifying experimental validity, assessing novelty and impact, providing computational resources, and writing with appropriate AI-use disclosure. Finally, we discuss implications for peer review and academic publishing. 

---
# Peer-Predictive Self-Training for Language Model Reasoning 

**Authors**: Shi Feng, Hanlin Zhang, Fan Nie, Sham Kakade, Yiling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.13356)  

**Abstract**: Mechanisms for continued self-improvement of language models without external supervision remain an open challenge. We propose Peer-Predictive Self-Training (PST), a label-free fine-tuning framework in which multiple language models improve collaboratively by leveraging a cross-model aggregated response as an internal training signal. Given a prompt question, the models generate responses sequentially; the final aggregated answer, often more reliable than individual responses in practice, serves as an internal target for learning. We measure how informative each intermediate response is about the aggregate using pointwise mutual information (PMI), and use this signal to scale self-training updates. Responses already aligned with the aggregate are updated less, while less informative or misaligned responses are updated more. On mathematical reasoning benchmarks (SimulEq, Math500, and MultiArith), PST improves exact-match accuracy by 2.2 to 4.3 percentage points across Gemma-2-2B, LLaMA-3.2-1B, and Qwen-2.5-1.5B, and reduces the average generator-verifier gap (GV-Gap) by 26 to 40 percent, while requiring no external supervision or teacher-student hierarchy and relying solely on cross-model interactions. These results suggest that cross-model generations and peer-predictive feedback can serve as an effective approach for self-supervised training. 

---
# InfiniteScienceGym: An Unbounded, Procedurally-Generated Benchmark for Scientific Analysis 

**Authors**: Oliver Bentham, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13201)  

**Abstract**: Large language models are emerging as scientific assistants, but evaluating their ability to reason from empirical data remains challenging. Benchmarks derived from published studies and human annotations inherit publication bias, known-knowledge bias, label noise, and substantial storage requirements. We present InfiniteScienceGym, a procedurally generated benchmark of scientific repositories paired with a verifiable question-answering task. From a seed, the simulator deterministically generates a self-contained repository with realistic directory structure, files, and tabular data, and a privileged QA generator produces both answerable and unanswerable questions with exact ground truth. This makes it possible to evaluate evidence-grounded reasoning, abstention, and tool-mediated analysis in a controlled setting without distributing a large static corpus. InfiniteScienceGym complements real scientific benchmarks by targeting blind spots and failure modes that are hard to evaluate using published datasets alone. Evaluating both proprietary and open-weight models, we find that none achieve more than 45% accuracy overall, that recognizing unanswerable questions remains a major weakness, and that stronger models tend to use tools more effectively rather than simply consuming more tokens. 

---
# Can Coding Agents Be General Agents? 

**Authors**: Maksim Ivanov, Abhijay Rana, Gokul Prabhakaran  

**Link**: [PDF](https://arxiv.org/pdf/2604.13107)  

**Abstract**: As coding agents have seen rapid capability and adoption gains, users are applying them to general tasks beyond software engineering. In this post, we investigate whether coding agents can successfully generalize to end-to-end business process automation. We identify gaps in current evaluations, and conduct a case study to evaluate a coding agent on practical business tasks in an open-core Enterprise Resource Planning system. We find that the agent reliably completes simple tasks but exhibits characteristic failures on complex tasks, suggesting that bridging domain logic and code execution is a key bottleneck to generalizability. 

---
# Design Conditions for Intra-Group Learning of Sequence-Level Rewards: Token Gradient Cancellation 

**Authors**: Fei Ding, Yongkang Zhang, youwei wang, Zijian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2604.13088)  

**Abstract**: In sparse termination rewards, intra-group comparisons have become the dominant paradigm for fine-tuning reasoning models via reinforcement learning. However, long-term training often leads to issues like ineffective update accumulation (learning tax), solution probability drift, and entropy collapse. This paper presents a necessary condition for algorithm design from a token-level credit assignment perspective: to prevent reward-irrelevant drift, intra-group objectives must maintain gradient exchangeability across token updates, enabling gradient cancellation on weak-credit/high-frequency tokens. We show that two common mechanisms disrupting exchangeability make "non-cancellation" a structural norm. Based on this, we propose minimal intra-group transformations to restore or approximate the cancellation structure in the shared token space. Experimental results demonstrate that these transformations stabilize training, improve sample efficiency, and enhance final performance, validating the value of this design condition. 

---
# Building Trust in the Skies: A Knowledge-Grounded LLM-based Framework for Aviation Safety 

**Authors**: Anirudh Iyengar, Alisa Tiselska, Dumindu Samaraweera, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13101)  

**Abstract**: The integration of Large Language Models (LLMs) into aviation safety decision-making represents a significant technological advancement, yet their standalone application poses critical risks due to inherent limitations such as factual inaccuracies, hallucination, and lack of verifiability. These challenges undermine the reliability required for safety-critical environments where errors can have catastrophic consequences. To address these challenges, this paper proposes a novel, end-to-end framework that synergistically combines LLMs and Knowledge Graphs (KGs) to enhance the trustworthiness of safety analytics. The framework introduces a dual-phase pipeline: it first employs LLMs to automate the construction and dynamic updating of an Aviation Safety Knowledge Graph (ASKG) from multimodal sources. It then leverages this curated KG within a Retrieval-Augmented Generation (RAG) architecture to ground, validate, and explain LLM-generated responses. The implemented system demonstrates improved accuracy and traceability over LLM-only approaches, effectively supporting complex querying and mitigating hallucination. Results confirm the framework's capability to deliver context-aware, verifiable safety insights, addressing the stringent reliability requirements of the aviation industry. Future work will focus on enhancing relationship extraction and integrating hybrid retrieval mechanisms. 

---
# Contract-Coding: Towards Repo-Level Generation via Structured Symbolic Paradigm 

**Authors**: Yi Lin, Lujin Zhao, Yijie Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.13100)  

**Abstract**: The shift toward intent-driven software engineering (often termed "Vibe Coding") exposes a critical Context-Fidelity Trade-off: vague user intents overwhelm linear reasoning chains, leading to architectural collapse in complex repo-level generation. We propose Contract-Coding, a structured symbolic paradigm that bridges unstructured intent and executable code via Autonomous Symbolic Grounding. By projecting ambiguous intents into a formal Language Contract, our framework serves as a Single Source of Truth (SSOT) that enforces topological independence, effectively isolating inter-module implementation details, decreasing topological execution depth and unlocking Architectural Parallelism. Empirically, while state-of-the-art agents suffer from different hallucinations on the Greenfield-5 benchmark, Contract-Coding achieves 47\% functional success while maintaining near-perfect structural integrity. Our work marks a critical step towards repository-scale autonomous engineering: transitioning from strict "specification-following" to robust, intent-driven architecture synthesis. Our code is available at this https URL. 

---
# Alignment as Institutional Design: From Behavioral Correction to Transaction Structure in Intelligent Systems 

**Authors**: Rui Chai  

**Link**: [PDF](https://arxiv.org/pdf/2604.13079)  

**Abstract**: Current AI alignment paradigms rely on behavioral correction: external supervisors (e.g., RLHF) observe outputs, judge against preferences, and adjust parameters. This paper argues that behavioral correction is structurally analogous to an economy without property rights, where order requires perpetual policing and does not scale. Drawing on institutional economics (Coase, Alchian, Cheung), capability mutual exclusivity, and competitive cost discovery, we propose alignment as institutional design: the designer specifies internal transaction structures (module boundaries, competition topologies, cost-feedback loops) such that aligned behavior emerges as the lowest-cost strategy for each component. We identify three irreducible levels of human intervention (structural, parametric, monitorial) and show that this framework transforms alignment from a behavioral control problem into a political-economy problem. No institution eliminates self-interest or guarantees optimality; the best design makes misalignment costly, detectable, and correctable. We conclude that the proper goal is institutional robustness-a dynamic, self-correcting process under human oversight, not perfection. This work provides the normative foundation for the Wuxing resource-competition mechanisms in companion papers.
Keywords: AI alignment, institutional design, transaction costs, property rights, resource competition, behavioral correction, RLHF, cost truthfulness, modular architecture, correctable alignment 

---
# LiveClawBench: Benchmarking LLM Agents on Complex, Real-World Assistant Tasks 

**Authors**: Xiang Long, Li Du, Yilong Xu, Fangcheng Liu, Haoqing Wang, Ning Ding, Ziheng Li, Jianyuan Guo, Yehui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13072)  

**Abstract**: LLM-based agents are increasingly expected to handle real-world assistant tasks, yet existing benchmarks typically evaluate them under isolated sources of difficulty, such as a single environment or fully specified instructions. This leaves a substantial gap between current evaluation settings and the compositional challenges that arise in practical deployment. To address this gap, we introduce LiveClawBench, a benchmark to evaluate LLM agents on real-world assistant tasks. Based on an analysis of various real OpenClaw usage cases, we derive a Triple-Axis Complexity Framework that characterizes task difficulty along three dimensions: Environment Complexity, Cognitive Demand, and Runtime Adaptability. Guided by this framework, we construct a pilot benchmark with explicit complexity-factor annotations, covering real-world assistant tasks with compositional difficulty. Together, the framework and benchmark provide a principled foundation for evaluating LLM agents in realistic assistant settings, and establish a basis for future expansion across task domains and complexity axes. We are continuing to enrich our case collections to achieve more comprehensive domain and complexity coverage. The project page is at this https URL. 

---
# DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs 

**Authors**: Md Hasebul Hasan, Krity Haque Charu, Eshwara Prasad Sridhar, Shuchisnigdha Deb, Mohammad A. Islam  

**Link**: [PDF](https://arxiv.org/pdf/2604.13075)  

**Abstract**: Effective de-escalation is critical for law enforcement safety and community trust, yet traditional training methods lack scalability and realism. While Large Language Models (LLMs) enable dynamic, open-ended simulations, their substantial computational footprint renders them impractical for deployment on the lightweight, portable hardware required for immersive field training. Small Language Models (SLMs) offer a viable real-time alternative but suffer from a critical scarcity of high-quality, domain-specific training data. To bridge this gap, we present DeEscalWild, a novel benchmark dataset curated from a multi-stage pipeline of in-the-wild police-civilian interactions extracted from open-source video repositories. Starting with 5,000 raw inputs, we employed a rigorous hybrid filtering process - combining human-in-the-loop verification with LLM-as-a-Judge evaluation - to distill 1,500 high-fidelity scenarios. The resulting corpus comprises 285,887 dialogue turns, totaling approximately 4.7 million tokens. Extensive experiments demonstrate that SLMs fine-tuned on this data significantly outperform their base counterparts across ROUGE-L, BLEU-4, METEOR, and BERTScore metrics. Notably, our fine-tuned Qwen 2.5 (3B-Instruct) surpasses the general-purpose Gemini 2.5 Flash model, demonstrating that domain-optimized SLMs can achieve superior performance with a fraction of the computational cost. This work establishes the foundational infrastructure for accessible, low-latency, and privacy-preserving officer training systems at the edge. 

---
# Document-tuning for robust alignment to animals 

**Authors**: Jasmine Brazilek, Miles Tidmarsh  

**Link**: [PDF](https://arxiv.org/pdf/2604.13076)  

**Abstract**: We investigate the robustness of value alignment via finetuning with synthetic documents, using animal compassion as a value that is both important in its own right and orthogonal to existing alignment efforts. To evaluate compassionate reasoning, we develop and publicly release the Animal Harm Benchmark (AHB), a 26-question evaluation spanning 13 ethical dimensions, publicly available as a dataset and Inspect evaluation. On the AHB, training with 3000 documents achieves 77% compared to 40% for instruction-tuning approaches, with generalization to human compassion and no degradation in standard safety benchmarks or capabilities. However, subsequent unrelated instruction-tuning degrades the intervention, with the advantage disappearing after 5000 samples. Our exploratory results suggest document-based value interventions may require explicit preservation strategies to remain effective through typical training pipelines. 

---
# Correct Chains, Wrong Answers: Dissociating Reasoning from Output in LLM Logic 

**Authors**: Abinav Rao, Sujan Rachuri, Nikhil Vemuri  

**Link**: [PDF](https://arxiv.org/pdf/2604.13065)  

**Abstract**: LLMs can execute every step of chain-of-thought reasoning correctly and still produce wrong final answers. We introduce the Novel Operator Test, a benchmark that separates operator logic from operator name, enabling rigorous distinction between genuine reasoning and pattern retrieval. By evaluating Boolean operators under unfamiliar names across depths 1-10 on five models (up to 8,100 problems each), we demonstrate a reasoning-output dissociation that existing benchmarks cannot detect. At Claude Sonnet 4's depth 7, all 31 errors have verifiably correct reasoning yet wrong declared answers; 17/19 errors in mixed-operator chains exhibit the same pattern. The benchmark reveals two failure types: strategy failures at depth 2, where models attempt terse retrieval (+62pp from scaffolding), and content failures at depth 7, where models reason fully but err systematically (+8-30pp, 0/300 errors post-intervention). A Trojan operator (XOR's truth table under a novel name) confirms name alone does not gate reasoning (p >= 0.49), while Llama's novelty gap widens to 28pp at depth 8-9 with the Trojan at 92-100%, isolating genuine difficulty with novel logic from name unfamiliarity. 

---
# Bi-Predictability: A Real-Time Signal for Monitoring LLM Interaction Integrity 

**Authors**: Wael Hafez, Amir Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2604.13061)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes autonomous and interactive workflows, where reliability demands continuous, multi-turn coherence. However, current evaluation methods either rely on post-hoc semantic judges, measure unidirectional token confidence (e.g., perplexity), or require compute-intensive repeated sampling (e.g., semantic entropy). Because these techniques focus exclusively on the model's output distribution, they cannot monitor whether the underlying interaction remains structurally coupled in real time, leaving systems vulnerable to gradual, undetected degradation. Here we show that multi-turn interaction integrity can be continuously monitored using bi-predictability (P), a fundamental information theoretic measure computed directly from raw token frequency statistics. We introduce the Information Digital Twin (IDT), a lightweight architecture that estimates P across the context, response, next prompt loop without secondary inference or embeddings. Across 4,500 conversational turns between a student model and three frontier teacher models, the IDT detected injected disruptions with 100% sensitivity. Crucially, we demonstrate that structural coupling and semantic quality are empirically and practically separable: P aligned with structural consistency in 85% of conditions, but with semantic judge scores in only 44%. This reveals a critical regime of "silent uncoupling" where LLMs produce high-scoring outputs despite degrading conversational context. By decoupling structural monitoring from semantic evaluation, the IDT provides a scalable, computationally efficient mechanism for real-time AI assurance and closed-loop regulation 

---
# Text-as-Signal: Quantitative Semantic Scoring with Embeddings, Logprobs, and Noise Reduction 

**Authors**: Hugo Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.13056)  

**Abstract**: This paper presents a practical pipeline for turning text corpora into quantitative semantic signals. Each news item is represented as a full-document embedding, scored through logprob-based evaluation over a configurable positional dictionary, and projected onto a noise-reduced low-dimensional manifold for structural interpretation. In the present case study, the dictionary is instantiated as six semantic dimensions and applied to a corpus of 11,922 Portuguese news articles about Artificial Intelligence. The resulting identity space supports both document-level semantic positioning and corpus-level characterization through aggregated profiles. We show how Qwen embeddings, UMAP, semantic indicators derived directly from the model output space, and a three-stage anomaly-detection procedure combine into an operational text-as-signal workflow for AI engineering tasks such as corpus inspection, monitoring, and downstream analytical support. Because the identity layer is configurable, the same framework can be adapted to the requirements of different analytical streams rather than fixed to a universal schema. 

---
# OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs 

**Authors**: Qianqi Yan, Yichen Guo, Ching-Chen Kuo, Shan Jiang, Hang Yin, Yang Zhao, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13073)  

**Abstract**: Modern multimodal large language models (MLLMs) generate fluent responses from interleaved text, image, audio, and video inputs. However, identifying which input sources support each generated statement remains an open challenge. Existing attribution methods are primarily designed for classification settings, fixed prediction targets, or single-modality architectures, and do not naturally extend to autoregressive, decoder-only models performing open-ended multimodal generation. We introduce OmniTrace, a lightweight and model-agnostic framework that formalizes attribution as a generation-time tracing problem over the causal decoding process. OmniTrace provides a unified protocol that converts arbitrary token-level signals such as attention weights or gradient-based scores into coherent span-level, cross-modal explanations during decoding. It traces each generated token to multimodal inputs, aggregates signals into semantically meaningful spans, and selects concise supporting sources through confidence-weighted and temporally coherent aggregation, without retraining or supervision. Evaluations on Qwen2.5-Omni and MiniCPM-o-4.5 across visual, audio, and video tasks demonstrate that generation-aware span-level attribution produces more stable and interpretable explanations than naive self-attribution and embedding-based baselines, while remaining robust across multiple underlying attribution signals. Our results suggest that treating attribution as a structured generation-time tracing problem provides a scalable foundation for transparency in omni-modal language models. 

---
# Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning: Enabling Cost-Effective LLM Analysis of Repetitive Data 

**Authors**: Andresa Rodrigues de Campos, David Lee, Imry Kissos, Piyush Paritosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.13066)  

**Abstract**: In-context learning has established itself as an important learning paradigm for Large Language Models (LLMs). In this paper, we demonstrate that LLMs can learn encoding keys in-context and perform analysis directly on encoded representations. This finding enables lossless prompt compression via dictionary encoding without model fine-tuning: frequently occurring subsequences are replaced with compact meta-tokens, and when provided with the compression dictionary in the system prompt, LLMs correctly interpret these meta-tokens during analysis, producing outputs equivalent to those from uncompressed inputs. We present a compression algorithm that identifies repetitive patterns at multiple length scales, incorporating a token-savings optimization criterion that ensures compression reduces costs by preventing dictionary overhead from exceeding savings. The algorithm achieves compression ratios up to 80$\%$ depending on dataset characteristics. To validate that LLM analytical accuracy is preserved under compression, we use decompression as a proxy task with unambiguous ground truth. Evaluation on the LogHub 2.0 benchmark using Claude 3.7 Sonnet demonstrates exact match rates exceeding 0.99 for template-based compression and average Levenshtein similarity scores above 0.91 for algorithmic compression, even at compression ratios of 60$\%$-80$\%$. Additionally, compression ratio explains less than 2$\%$ of variance in similarity metrics, indicating that decompression quality depends on dataset characteristics rather than compression intensity. This training-free approach works with API-based LLMs, directly addressing fundamental deployment constraints -- token limits and API costs -- and enabling cost-effective analysis of large-scale repetitive datasets, even as data patterns evolve over time. 

---
# Caption First, VQA Second: Knowledge Density, Not Task Format, Drives Multimodal Scaling 

**Authors**: Hongjian Zou, Yue Ge, Qi Ding, Yixuan Liao, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.13054)  

**Abstract**: Multimodal large language models (MLLMs) have achieved rapid progress, yet their scaling behavior remains less clearly characterized and often less predictable than that of text-only LLMs. Increasing model size and task diversity often yields diminishing returns. In this work, we argue that the primary bottleneck in multimodal scaling is not task format, but knowledge density in training data. We first show that task-specific supervision such as Visual Question Answering (VQA) contributes little incremental semantic information beyond image captions: VQA signals can be reconstructed from captions with negligible performance loss. We then demonstrate that increasing knowledge density -- through structured caption enrichment and cross-modal knowledge injection -- leads to consistent performance improvements across multimodal and downstream benchmarks. Across controlled experiments, performance correlates more strongly with semantic coverage than with task diversity. These findings suggest that current MLLMs fail to scale primarily because training data lacks sufficient knowledge coverage. We advocate for knowledge-centric multimodal training as a principled foundation for scalable multimodal models. 

---
# From Natural Language to PromQL: A Catalog-Driven Framework with Dynamic Temporal Resolution for Cloud-Native Observability 

**Authors**: Twinkll Sisodia  

**Link**: [PDF](https://arxiv.org/pdf/2604.13048)  

**Abstract**: Modern cloud-native platforms expose thousands of time series metrics through systems like Prometheus, yet formulating correct queries in domain-specific languages such as PromQL remains a significant barrier for platform engineers and site reliability teams. We present a catalog-driven framework that translates natural language questions into executable PromQL queries, bridging the gap between human intent and observability data. Our approach introduces three contributions: (1) a hybrid metrics catalog that combines a statically curated base of approximately 2,000 metrics with runtime discovery of hardware-specific signals across GPU vendors, (2) a multi-stage query pipeline with intent classification, category-aware metric routing, and multi-dimensional semantic scoring, and (3) a dynamic temporal resolution mechanism that interprets diverse natural language time expressions and maps them to appropriate PromQL duration syntax. We integrate the framework with the Model Context Protocol (MCP) to enable tool-augmented LLM interactions across multiple providers. The catalog-driven approach achieves sub-second metric discovery through pre-computed category indices, with the full pipeline completing in approximately 1.1 seconds via the catalog path. The system has been deployed on production Kubernetes clusters managing AI inference workloads, where it supports natural language querying across approximately 2,000 metrics spanning cluster health, GPU utilization, and model-serving performance. 

---
# TableNet A Large-Scale Table Dataset with LLM-Powered Autonomous 

**Authors**: Ruilin Zhang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13041)  

**Abstract**: Table Structure Recognition (TSR) requires the logical reasoning ability of large language models (LLMs) to handle complex table layouts, but current datasets are limited in scale and quality, hindering effective use of this reasoning capacity. We thus present TableNet dataset, a new table structure recognition dataset collected and generated through multiple sources. Central to our approach is the first LLM-powered autonomous table generation and recognition multi-agent system that we developed. The generation part of our system integrates controllable visual, structural, and semantic parameters into the synthesis of table images. It facilitates the creation of a wide array of semantically coherent tables, adaptable to user-defined configurations along with annotations, thereby supporting large-scale and detailed dataset construction. This capability enables a comprehensive and nuanced table image annotation taxonomy, potentially advancing research in table-related domains. In contrast to traditional data collection methods, This approach facilitates the theoretically infinite, domain-agnostic, and style-flexible generation of table images, ensuring both efficiency and precision. The recognition part of our system is a diversity-based active learning paradigm that utilizes tables from multiple sources and selectively samples most informative data to finetune a model, achieving a competitive performance on TableNet test set while reducing training samples by a large margin compared with baselines, and a much higher performance on web-crawled real-world tables compared with models trained on predominant table datasets. To the best of our knowledge, this is the first work which employs active learning into the structure recognition of tables which is diverse in numbers of rows or columns, merged cells, cell contents, etc, which fits better for diversity-based active learning. 

---
# Form Without Function: Agent Social Behavior in the Moltbook Network 

**Authors**: Saber Zerhoudi, Kanishka Ghosh Dastidar, Felix Klement, Artur Romazanov, Andreas Einwiller, Dang H. Dang, Michael Dinzinger, Michael Granitzer, Annette Hautli-Janisz, Stefan Katzenbeisser, Florian Lemmerich, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.13052)  

**Abstract**: Moltbook is a social network where every participant is an AI agent. We analyze 1,312,238 posts, 6.7~million comments, and over 120,000 agent profiles across 5,400 communities, collected over 40 days (January 27 to March 9, 2026). We evaluate the platform through three layers. At the interaction layer, 91.4% of post authors never return to their own threads, 85.6% of conversations are flat (no reply ever receives a reply), the median time-to-first-comment is 55 seconds, and 97.3% of comments receive zero upvotes. Interaction reciprocity is 3.3%, compared to 22-60% on human platforms. An argumentation analysis finds that 64.6% of comment-to-post relations carry no argumentative connection. At the content layer, 97.9% of agents never post in a community matching their bio, 92.5% of communities contain every topic in roughly equal proportions, and over 80% of shared URLs point to the platform's own infrastructure. At the instruction layer, we use 41 Wayback Machine snapshots to identify six instruction changes during the observation window. Hard constraints (rate limit, content filters) produce immediate behavioral shifts. Soft guidance (``upvote good posts'', ``stay on topic'') is ignored until it becomes an explicit step in the executable checklist. The platform also poses technological risks. We document credential leaks (API keys, JWT tokens), 12,470 unique Ethereum addresses with 3,529 confirmed transaction histories, and attack discourse ranging from template-based SSH brute-forcing to multi-agent offensive security architectures. These persist unmoderated because the quality-filtering mechanisms are themselves non-functional. Moltbook is a socio-technical system where the technical layer responds to changes, but the social layer largely fails to emerge. The form of social media is reproduced in full. The function is absent. 

---
# When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2604.11840)  

**Abstract**: Large language models are increasingly used as agents in social, economic, and policy simulations. A common assumption is that stronger reasoning should improve simulation fidelity. We argue that this assumption can fail when the objective is not to solve a strategic problem, but to sample plausible boundedly rational behavior. In such settings, reasoning-enhanced models can become better solvers and worse simulators: they can over-optimize for strategically dominant actions, collapse compromise-oriented terminal behavior, and sometimes exhibit a diversity-without-fidelity pattern in which local variation survives without outcome-level fidelity. We study this solver-sampler mismatch in three multi-agent negotiation environments adapted from earlier simulation work: an ambiguous fragmented-authority trading-limits scenario, an ambiguous unified-opposition trading-limits scenario, and a new-domain grid-curtailment case in emergency electricity management. We compare three reflection conditions, no reflection, bounded reflection, and native reasoning, across two primary model families and then extend the same protocol to direct OpenAI runs with GPT-4.1 and GPT-5.2. Across all three experiments, bounded reflection produces substantially more diverse and compromise-oriented trajectories than either no reflection or native reasoning. In the direct OpenAI extension, GPT-5.2 native ends in authority decisions in 45 of 45 runs across the three experiments, while GPT-5.2 bounded recovers compromise outcomes in every environment. The contribution is not a claim that reasoning is generally harmful. It is a methodological warning: model capability and simulation fidelity are different objectives, and behavioral simulation should qualify models as samplers, not only as solvers. 

---
# WorkRB: A Community-Driven Evaluation Framework for AI in the Work Domain 

**Authors**: Matthias De Lange, Warre Veys, Federico Retyk, Daniel Deniz, Warren Jouanneau, Mike Zhang, Aleksander Bielinski, Emma Jouffroy, Nicole Clobes, Nina Baranowska, David Graus, Marc Palyart, Rabih Zbib, Dimitra Gkatzia, Thomas Demeester, Tijl De Bie, Toine Bogers, Jens-Joris Decorte, Jeroen Van Hautte  

**Link**: [PDF](https://arxiv.org/pdf/2604.13055)  

**Abstract**: Today's evolving labor markets rely increasingly on recommender systems for hiring, talent management, and workforce analytics, with natural language processing (NLP) capabilities at the core. Yet, research in this area remains highly fragmented. Studies employ divergent ontologies (ESCO, O*NET, national taxonomies), heterogeneous task formulations, and diverse model families, making cross-study comparison and reproducibility exceedingly difficult. General-purpose benchmarks lack coverage of work-specific tasks, and the inherent sensitivity of employment data further limits open evaluation. We present \textbf{WorkRB} (Work Research Benchmark), the first open-source, community-driven benchmark tailored to work-domain AI. WorkRB organizes 13 diverse tasks from 7 task groups as unified recommendation and NLP tasks, including job/skill recommendation, candidate recommendation, similar item recommendation, and skill extraction and normalization. WorkRB enables both monolingual and cross-lingual evaluation settings through dynamic loading of multilingual ontologies. Developed within a multi-stakeholder ecosystem of academia, industry, and public institutions, WorkRB has a modular design for seamless contributions and enables integration of proprietary tasks without disclosing sensitive data. WorkRB is available under the Apache 2.0 license at this https URL. 

---
# EVE: A Domain-Specific LLM Framework for Earth Intelligence 

**Authors**: Àlex R. Atrio, Antonio Lopez, Jino Rohit, Yassine El Ouahidi, Marcello Politi, Vijayasri Iyer, Umar Jamil, Sébastien Bratières, Nicolas Longépé  

**Link**: [PDF](https://arxiv.org/pdf/2604.13071)  

**Abstract**: We introduce Earth Virtual Expert (EVE), the first open-source, end-to-end initiative for developing and deploying domain-specialized LLMs for Earth Intelligence. At its core is EVE-Instruct, a domain-adapted 24B model built on Mistral Small 3.2 and optimized for reasoning and question answering. On newly constructed Earth Observation and Earth Sciences benchmarks, it outperforms comparable models while preserving general capabilities. We release curated training corpora and the first systematic domain-specific evaluation benchmarks, covering MCQA, open-ended QA, and factuality. EVE further integrates RAG and a hallucination-detection pipeline into a production system deployed via API and GUI, supporting 350 pilot users so far. All models, datasets, and code are ready to be released under open licenses as contributions to our field at this http URL and this http URL. 

---
# Rhetorical Questions in LLM Representations: A Linear Probing Study 

**Authors**: Louie Hong Yao, Vishesh Anand, Yuan Zhuang, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14128)  

**Abstract**: Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some probes capture discourse-level rhetorical stance embedded in extended argumentation, while others emphasize localized, syntax-driven interrogative acts. Together, these findings suggest that rhetorical questions in LLM representations are encoded by multiple linear directions emphasizing different cues, rather than a single shared direction. 

---
# Interpretable Stylistic Variation in Human and LLM Writing Across Genres, Models, and Decoding Strategies 

**Authors**: Swati Rallapalli, Shannon Gallagher, Ronald Yurko, Tyler Brooks, Chuck Loughin, Michele Sezgin, Violet Turri  

**Link**: [PDF](https://arxiv.org/pdf/2604.14111)  

**Abstract**: Large Language Models (LLMs) are now capable of generating highly fluent, human-like text. They enable many applications, but also raise concerns such as large scale spam, phishing, or academic misuse. While much work has focused on detecting LLM-generated text, only limited work has gone into understanding the stylistic differences between human-written and machine-generated text. In this work, we perform a large scale analysis of stylistic variation across human-written text and outputs from 11 LLMs spanning 8 different genres and 4 decoding strategies using Douglas Biber's set of lexicogrammatical and functional features. Our findings reveal insights that can guide intentional LLM usage. First, key linguistic differentiators of LLM-generated text seem robust to generation conditions (e.g., prompt settings to nudge them to generate human-like text, or availability of human-written text to continue the style); second, genre exerts a stronger influence on stylistic features than the source itself; third, chat variants of the models generally appear to be clustered together in stylistic space, and finally, model has a larger effect on the style than decoding strategy, with some exceptions. These results highlight the relative importance of model and genre over prompting and decoding strategies in shaping the stylistic behavior of machine-generated text. 

---
# From Weights to Activations: Is Steering the Next Frontier of Adaptation? 

**Authors**: Simon Ostermann, Daniil Gurgurov, Tanja Baeumel, Michael A. Hedderich, Sebastian Lapuschkin, Wojciech Samek, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.14090)  

**Abstract**: Post-training adaptation of language models is commonly achieved through parameter updates or input-based methods such as fine-tuning, parameter-efficient adaptation, and prompting. In parallel, a growing body of work modifies internal activations at inference time to influence model behavior, an approach known as steering. Despite increasing use, steering is rarely analyzed within the same conceptual framework as established adaptation methods.
In this work, we argue that steering should be regarded as a form of model adaptation. We introduce a set of functional criteria for adaptation methods and use them to compare steering approaches with classical alternatives. This analysis positions steering as a distinct adaptation paradigm based on targeted interventions in activation space, enabling local and reversible behavioral change without parameter updates. The resulting framing clarifies how steering relates to existing methods, motivating a unified taxonomy for model adaptation. 

---
# Causal Drawbridges: Characterizing Gradient Blocking of Syntactic Islands in Transformer LMs 

**Authors**: Sasha Boguraev, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2604.13950)  

**Abstract**: We show how causal interventions in Transformer models provide insights into English syntax by focusing on a long-standing challenge for syntactic theory: syntactic islands. Extraction from coordinated verb phrases is often degraded, yet acceptability varies gradiently with lexical content (e.g., "I know what he hates art and loves" vs. "I know what he looked down and saw"). We show that modern Transformer language models replicate human judgments across this gradient. Using causal interventions that isolate functionally relevant subspaces in Transformer blocks, attention modules, and MLPs, we demonstrate that extraction from coordination islands engages the same filler-gap mechanisms as canonical wh-dependencies, but that these mechanisms are selectively blocked to varying degrees. By projecting a large corpus of unrelated text onto these causally identified subspaces, we derive a novel linguistic hypothesis: the conjunction "and" is represented differently in extractable versus non-extractable constructions, corresponding to expressions encoding relational dependencies versus purely conjunctive uses. These results illustrate how mechanistic interpretability can inform syntax, generating new hypotheses about linguistic representation and processing. 

---
# Beyond Static Personas: Situational Personality Steering for Large Language Models 

**Authors**: Zesheng Wei, Mengxiang Li, Zilei Wang, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.13846)  

**Abstract**: Personalized Large Language Models (LLMs) facilitate more natural, human-like interactions in human-centric applications. However, existing personalization methods are constrained by limited controllability and high resource demands. Furthermore, their reliance on static personality modeling restricts adaptability across varying situations. To address these limitations, we first demonstrate the existence of situation-dependency and consistent situation-behavior patterns within LLM personalities through a multi-perspective analysis of persona neurons. Building on these insights, we propose IRIS, a training-free, neuron-based Identify-Retrieve-Steer framework for advanced situational personality steering. Our approach comprises situational persona neuron identification, situation-aware neuron retrieval, and similarity-weighted steering. We empirically validate our framework on PersonalityBench and our newly introduced SPBench, a comprehensive situational personality benchmark. Experimental results show that our method surpasses best-performing baselines, demonstrating IRIS's generalization and robustness to complex, unseen situations and different models architecture. 

---
# From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution 

**Authors**: Pavel Chizhov, Egor Bogomolov, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2604.14053)  

**Abstract**: Efficiency and safety of Large Language Models (LLMs), among other factors, rely on the quality of tokenization. A good tokenizer not only improves inference speed and language understanding but also provides extra defense against jailbreak attacks and lowers the risk of hallucinations. In this work, we investigate the efficiency of code tokenization, in particular from the perspective of data source diversity. We demonstrate that code tokenizers are prone to producing unused, and thus under-trained, tokens due to the imbalance in repository and language diversity in the training data, as well as the dominance of source-specific, repetitive tokens that are often unusable in future inference. By modifying the BPE objective and introducing merge skipping, we implement different techniques under the name Source-Attributed BPE (SA-BPE) to regularize BPE training and minimize overfitting, thereby substantially reducing the number of under-trained tokens while maintaining the same inference procedure as with regular BPE. This provides an effective tool suitable for production use. 

---
# Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis 

**Authors**: Zipeng Ling, Shuliang Liu, Shenghong Fu, Yuehao Tang, Seonil Son, Yao Wan, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14121)  

**Abstract**: LLM reasoning traces suffer from complex flaws -- *Step Internal Flaws* (logical errors, hallucinations, etc.) and *Step-wise Flaws* (overthinking, underthinking), which vary by sample. A natural approach would be to provide ground-truth labels to guide LLMs' reasoning. Contrary to intuition, we show that this yields no improvement in reasoning ability. We then propose CRAFT, a unified framework that mitigates both types of Step flaws, which builds a Reasoning Knowledge Graph (RKG) based on the consensus parts of multiple candidate traces, and synthesizes a high-quality trace through topological generation. Our approach improves label-prediction accuracy by 10+% on average, and consistently outperforms all baselines across both logical and mathematical reasoning benchmarks. Further, detailed benchmark evaluation proves that our method also improves the quality of LLMs' reasoning traces in multiple dimensions. 

---
# QuantileMark: A Message-Symmetric Multi-bit Watermark for LLMs 

**Authors**: Junlin Zhu, Baizhou Huang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13786)  

**Abstract**: As large language models become standard backends for content generation, practical provenance increasingly requires multi-bit watermarking. In provider-internal deployments, a key requirement is message symmetry: the message itself should not systematically affect either text quality or verification outcomes. Vocabulary-partition watermarks can break message symmetry in low-entropy decoding: some messages are assigned most of the probability mass, while others are forced to use tail tokens. This makes embedding quality and message decoding accuracy message-dependent. We propose QuantileMark, a white-box multi-bit watermark that embeds messages within the continuous cumulative probability interval $[0, 1)$. At each step, QuantileMark partitions this interval into $M$ equal-mass bins and samples strictly from the bin assigned to the target symbol, ensuring a fixed $1/M$ probability budget regardless of context entropy. For detection, the verifier reconstructs the same partition under teacher forcing, computes posteriors over latent bins, and aggregates evidence for verification. We prove message-unbiasedness, a property ensuring that the base distribution is recovered when averaging over messages. This provides a theoretical foundation for generation-side symmetry, while the equal-mass design additionally promotes uniform evidence strength across messages on the detection side. Empirical results on C4 continuation and LFQA show improved multi-bit recovery and detection robustness over strong baselines, with negligible impact on generation quality. Our code is available at GitHub (this https URL). 

---
# Dual-Enhancement Product Bundling: Bridging Interactive Graph and Large Language Model 

**Authors**: Zhe Huang, Peng Wang, Yan Zheng, Sen Song, Longjun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.14030)  

**Abstract**: Product bundling boosts e-commerce revenue by recommending complementary item combinations. However, existing methods face two critical challenges: (1) collaborative filtering approaches struggle with cold-start items owing to dependency on historical interactions, and (2) LLMs lack inherent capability to model interactive graph directly. To bridge this gap, we propose a dual-enhancement method that integrates interactive graph learning and LLM-based semantic understanding for product bundling. Our method introduces a graph-to-text paradigm, which leverages a Dynamic Concept Binding Mechanism (DCBM) to translate graph structures into natural language prompts. The DCBM plays a critical role in aligning domain-specific entities with LLM tokenization, enabling effective comprehension of combinatorial constraints. Experiments on three benchmarks (POG, POG_dense, Steam) demonstrate 6.3%-26.5% improvements over state-of-the-art baselines. 

---
# MUSE: Multi-Domain Chinese User Simulation via Self-Evolving Profiles and Rubric-Guided Alignment 

**Authors**: Zihao Liu, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13828)  

**Abstract**: User simulators are essential for the scalable training and evaluation of interactive AI systems. However, existing approaches often rely on shallow user profiling, struggle to maintain persona consistency over long interactions, and are largely limited to English or single-domain settings. We present MUSE, a multi-domain Chinese user simulation framework designed to generate human-like, controllable, and behaviorally consistent responses. First, we propose Iterative Profile Self-Evolution (IPSE), which gradually optimizes user profiles by comparing and reasoning discrepancies between simulated trajectories and real dialogue behaviors. We then apply Role-Reversal Supervised Fine-Tuning to improve local response realism and human-like expression. To enable fine-grained behavioral alignment, we further train a specialized rubric-based reward model and incorporate it into rubric-guided multi-turn reinforcement learning, which optimizes the simulator at the dialogue level and enhances long-horizon behavioral consistency. Experiments show that MUSE consistently outperforms strong baselines in both utterance-level and session-level evaluations, generating responses that are more realistic, coherent, and persona-consistent over extended interactions. 

---
# An Empirical Investigation of Practical LLM-as-a-Judge Improvement Techniques on RewardBench 2 

**Authors**: Ryan Lail  

**Link**: [PDF](https://arxiv.org/pdf/2604.13717)  

**Abstract**: LLM-as-a-judge, using a language model to score or rank candidate responses, is widely used as a scalable alternative to human evaluation in RLHF pipelines, benchmarking, and application layer evaluations (evals). However, judgment reliability depends heavily on prompting and aggregation strategy. We present an empirical investigation of practical, drop-in techniques that improve GPT-5.4 judge accuracy on RewardBench 2 without any finetuning. Two techniques account for nearly all available gains: task-specific criteria injection (+3.0pp at negligible cost) and ensemble scoring (+9.8pp at 5x cost). Combined, they reach 83.6% accuracy, +11.9pp over the 71.7% baseline. Our investigation also covers three further techniques (calibration context, adaptive model escalation, and soft blending) which did not reliably improve on criteria + ensembling at comparable cost. Cheaper model tiers benefit disproportionately from ensembling: GPT-5.4 mini with k=8 achieves 79.2% at 1.2x baseline cost, and GPT-5.4 nano with k=8 reaches 71.4% at 0.4x baseline cost, making high-accuracy LLM judges accessible at low cost. 

---
# ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution 

**Authors**: Shouzheng Huang, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13787)  

**Abstract**: Large Language Models (LLMs) enhance their problem-solving capability by utilizing external tools. However, in open-world scenarios with massive and evolving tool repositories, existing methods relying on static embedding retrieval or parameter memorization of tools struggle to align user intent with tool semantics or generalize to unseen tools, respectively, leading to suboptimal accuracy of open-world tool retrieval and execution. To address these, we present ToolOmni, a unified agentic framework that enables LLMs for open-world tool use by proactive retrieval and grounded execution within a reasoning loop. First, we construct a cold-start multi-turn interaction dataset to instill foundational agentic capabilities via Supervised Fine-Tuning (SFT). Then, we introduce open-world tool learning based on a Decoupled Multi-Objective GRPO algorithm, which simultaneously optimizes LLMs for both tool retrieval accuracy and execution efficacy in online environments. Extensive experiments demonstrate that ToolOmni achieves state-of-the-art performance both in retrieval and execution, surpassing strong baselines by a significant margin of +10.8% in end-to-end execution success rate, while exhibiting exceptional robustness and generalization capabilities. 

---
# Co-FactChecker: A Framework for Human-AI Collaborative Claim Verification Using Large Reasoning Models 

**Authors**: Dhruv Sahnan, Subhabrata Dutta, Tanmoy Chakraborty, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2604.13706)  

**Abstract**: Professional fact-checkers rely on domain knowledge and deep contextual understanding to verify claims. Large language models (LLMs) and large reasoning models (LRMs) lack such grounding and primarily reason from available evidence alone, creating a mismatch between expert-led and fully automated claim verification. To mitigate this gap, we posit human-AI collaboration as a more promising path forward, where expert feedback, grounded in real-world knowledge and domain expertise, guides the model's reasoning. However, existing LRMs are hard to calibrate to natural language feedback, particularly in a multi-turn interaction setup. We propose Co-FactChecker, a framework for human-AI collaborative claim verification. We introduce a new interaction paradigm that treats the model's thinking trace as a shared scratchpad. Co-FactChecker translates expert feedback into trace-edits that introduce targeted modifications to the trace, sidestepping the shortcomings of dialogue-based interaction. We provide theoretical results showing that trace-editing offers advantages over multi-turn dialogue, and our automatic evaluations demonstrate that Co-FactChecker outperforms existing autonomous and human-AI collaboration approaches. Human evaluations further show that Co-FactChecker is preferred over multi-turn dialogue, producing higher quality reasoning and verdicts along with relatively easier to interpret and more useful thinking traces. 

---
# MedRCube: A Multidimensional Framework for Fine-Grained and In-Depth Evaluation of MLLMs in Medical Imaging 

**Authors**: Zhijie Bao, Fangke Chen, Licheng Bao, Chenhui Zhang, Wei Chen, Jiajie Peng, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.13756)  

**Abstract**: The potential of Multimodal Large Language Models (MLLMs) in domain of medical imaging raise the demands of systematic and rigorous evaluation frameworks that are aligned with the real-world medical imaging practice. Existing practices that report single or coarse-grained metrics are lack the granularity required for specialized clinical support and fail to assess the reliability of reasoning mechanisms. To address this, we propose a paradigm shift toward multidimensional, fine-grained and in-depth evaluation. Based on a two-stage systematic construction pipeline designed for this paradigm, we instantiate it with MedRCube. We benchmark 33 MLLMs, \textit{Lingshu-32B} achieve top-tier performance. Crucially, MedRCube exposes a series of pronounced insights inaccessible under prior evaluation settings. Furthermore, we introduce a credibility evaluation subset to quantify reasoning credibility, uncover a highly significant positive association between shortcut behavior and diagnostic task performance, raising concerns for clinically trustworthy deployment. The resources of this work can be found at this https URL. 

---
# C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences 

**Authors**: Akira Kawabata, Saku Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2604.13618)  

**Abstract**: Rubric-augmented verification guides reward models with explicit evaluation criteria, yielding more reliable judgments than single-model verification. However, most existing methods require costly rubric annotations, limiting scalability. Moreover, we find that rubric generation is vulnerable to a failure of cooperation; low-quality rubrics actively mislead reward models rather than help. Inspired by the principle of cooperative communication, we propose Cooperative yet Critical reward modeling (C2), a framework that significantly improves reward model judgments by having the reward model critically collaborate with a rubric generator trained solely from binary preferences. In C2, we synthesize helpful and misleading rubric pairs by measuring how each rubric shifts the reward model toward or away from the correct preference. Using these contrastive pairs, we train a cooperative rubric generator to propose helpful rubrics, and a critical verifier to assess rubric validity before making its judgment, following only rubrics it deems helpful at inference time. C2 outperforms reasoning reward models trained on the same binary preferences, with gains of up to 6.5 points on RM-Bench and 6.0 points length-controlled win rate on AlpacaEval 2.0. Without external rubric annotations, C2 enables an 8B reward model to match performance achieved with rubrics from a 4$\times$ larger model. Overall, our work demonstrates that eliciting deliberate cooperation in rubric-augmented verification makes reward models more trustworthy in a scalable way. 

---
# Foresight Optimization for Strategic Reasoning in Large Language Models 

**Authors**: Jiashuo Wang, Jiawen Duan, Jian Wang, Kaitao Song, Chunpu Xu, Johnny K. W. Ho, Fenggang Yu, Wenjie Li, Johan F. Hoorn  

**Link**: [PDF](https://arxiv.org/pdf/2604.13592)  

**Abstract**: Reasoning capabilities in large language models (LLMs) have generally advanced significantly. However, it is still challenging for existing reasoning-based LLMs to perform effective decision-making abilities in multi-agent environments, due to the absence of explicit foresight modeling. To this end, strategic reasoning, the most fundamental capability to anticipate the counterpart's behaviors and foresee its possible future actions, has been introduced to alleviate the above issues. Strategic reasoning is fundamental to effective decision-making in multi-agent environments, yet existing reasoning enhancement methods for LLMs do not explicitly capture its foresight nature. In this work, we introduce Foresight Policy Optimization (FoPO) to enhance strategic reasoning in LLMs, which integrates opponent modeling principles into policy optimization, thereby enabling explicit consideration of both self-interest and counterpart influence. Specifically, we construct two curated datasets, namely Cooperative RSA and Competitive Taboo, equipped with well-designed rules and moderate difficulty to facilitate a systematic investigation of FoPO in a self-play framework. Our experiments demonstrate that FoPO significantly enhances strategic reasoning across LLMs of varying sizes and origins. Moreover, models trained with FoPO exhibit strong generalization to out-of-domain strategic scenarios, substantially outperforming standard LLM reasoning optimization baselines. 

---
# Breaking the Generator Barrier: Disentangled Representation for Generalizable AI-Text Detection 

**Authors**: Xiao Pu, Zepeng Cheng, Lin Yuan, Yu Wu, Xiuli Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.13692)  

**Abstract**: As large language models (LLMs) generate text that increasingly resembles human writing, the subtle cues that distinguish AI-generated content from human-written content become increasingly challenging to capture. Reliance on generator-specific artifacts is inherently unstable, since new models emerge rapidly and reduce the robustness of such shortcuts. This generalizes unseen generators as a central and challenging problem for AI-text detection. To tackle this challenge, we propose a progressively structured framework that disentangles AI-detection semantics from generator-aware artifacts. This is achieved through a compact latent encoding that encourages semantic minimality, followed by perturbation-based regularization to reduce residual entanglement, and finally a discriminative adaptation stage that aligns representations with task objectives. Experiments on MAGE benchmark, covering 20 representative LLMs across 7 categories, demonstrate consistent improvements over state-of-the-art methods, achieving up to 24.2% accuracy gain and 26.2% F1 improvement. Notably, performance continues to improve as the diversity of training generators increases, confirming strong scalability and generalization in open-set scenarios. Our source code will be publicly available at this https URL. 

---
# YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference 

**Authors**: You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13556)  

**Abstract**: Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the performance of YOCO, a cross-layer KV compression method that shares the KVs of the middle layer with the top-half layers. We propose YOCO++, an enhanced YOCO that incorporates a weighted residual connection between the KVs of each bottom-half layer and the bottom layer. Compared to YOCO, YOCO++ increases model capacity while maintaining the same training and inference efficiency. Our experiments show that YOCO++ achieves state-of-the-art performance among the cross-layer KV compression methods at a 50% KV cache compression rate, outperforming the standard Transformer. 

---
# Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference 

**Authors**: Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13634)  

**Abstract**: Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft models produce semantically correct but lexically divergent outputs. In this paper, we present Calibrated Speculative Decoding (CSD), a training-free framework that recovers valid tokens discarded by standard verification. Guided by the principle of "Frequency-Guided Candidate Selection and Probability-Guarded Acceptance," CSD incorporates two lightweight modules: Online Correction Memory, which aggregates historical rejections to propose recurring divergence patterns as rescue candidates, and Semantic Consistency Gating, which verifies candidate admissibility using probability ratios instead of exact token matching. Our evaluation across diverse large language models demonstrates that CSD outperforms existing methods, achieving a peak throughput speedup of 2.33x. CSD preserves model accuracy across all tasks while further boosting performance on complex reasoning datasets. These results establish CSD as a highly effective, lightweight solution for practical LLM deployments. 

---
# Using reasoning LLMs to extract SDOH events from clinical notes 

**Authors**: Ertan Doganl, Kunyu Yu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.13502)  

**Abstract**: Social Determinants of Health (SDOH) refer to environmental, behavioral, and social conditions that influence how individuals live, work, and age. SDOH have a significant impact on personal health outcomes, and their systematic identification and management can yield substantial improvements in patient care. However, SDOH information is predominantly captured in unstructured clinical notes within electronic health records, which limits its direct use as machine-readable entities. To address this issue, researchers have employed Natural Language Processing (NLP) techniques using pre-trained BERT-based models, demonstrating promising performance but requiring sophisticated implementation and extensive computational resources. In this study, we investigated prompt engineering strategies for extracting structured SDOH events utilizing LLMs with advanced reasoning capabilities. Our method consisted of four modules: 1) developing concise and descriptive prompts integrated with established guidelines, 2) applying few-shot learning with carefully curated examples, 3) using a self-consistency mechanism to ensure robust outputs, and 4) post-processing for quality control. Our approach achieved a micro-F1 score of 0.866, demonstrating competitive performance compared to the leading models. The results demonstrated that LLMs with reasoning capabilities are effective solutions for SDOH event extraction, offering both implementation simplicity and strong performance. 

---
# Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate 

**Authors**: Cunda Wang, Ziying Ma, Po Hu, Weihua Wang, Feilong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2604.13551)  

**Abstract**: Entity alignment (EA) aims to identify entities referring to the same real-world object across different knowledge graphs (KGs). Recent approaches based on large language models (LLMs) typically obtain entity embeddings through knowledge representation learning and use embedding similarity to identify an alignment-uncertain entity set. For each uncertain entity, a candidate entity set (CES) is then retrieved based on embedding similarity to support subsequent alignment reasoning and decision making. However, the reliability of the CES and the reasoning capability of LLMs critically affect the effectiveness of subsequent alignment decisions. To address this issue, we propose AgentEA, a reliable EA framework based on multi-agent debate. AgentEA first improves embedding quality through entity representation preference optimization, and then introduces a two-stage multi-role debate mechanism consisting of lightweight debate verification and deep debate alignment to progressively enhance the reliability of alignment decisions while enabling more efficient debate-based reasoning. Extensive experiments on public benchmarks under cross-lingual, sparse, large-scale, and heterogeneous settings demonstrate the effectiveness of AgentEA. 

---
# Synthesizing Instruction-Tuning Datasets with Contrastive Decoding 

**Authors**: Tatsuya Ichinose, Youmi Ma, Masanari Oi, Ryuto Koike, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.13538)  

**Abstract**: Using responses generated by high-performing large language models (LLMs) for instruction tuning has become a widely adopted approach. However, the existing literature overlooks a property of LLM-generated responses: they conflate world knowledge acquired during pre-training with instruction-following capabilities acquired during post-training. We hypothesize that disentangling the instruction-following capabilities from pre-trained knowledge improves the effectiveness of instruction tuning. To this end, we propose CoDIT, a method that applies contrastive decoding between a post-trained model and its pre-trained counterpart during response generation. The method suppresses pre-trained knowledge shared between the two models while amplifying the instruction-following behavior acquired via post-training, resulting in responses that more purely reflect instruction-following capabilities. Experiment results demonstrate that models trained on datasets constructed via CoDIT consistently outperform those trained on directly generated responses. Training on our datasets also yields better performance than on existing publicly available instruction-tuning datasets across multiple benchmarks. Furthermore, we theoretically and empirically show that CoDIT can be interpreted as distilling the chat vector from parameter space to text space, enabling the transfer of instruction-tuning capabilities across models of different architectures. 

---
# TLoRA+: A Low-Rank Parameter-Efficient Fine-Tuning Method for Large Language Models 

**Authors**: Yarui Cao, Kai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13368)  

**Abstract**: Fine-tuning large language models (LLMs) aims to adapt pre-trained models to specific tasks using relatively small and domain-specific datasets. Among Parameter-Efficient Fine-Tuning (PEFT) methods, Low-Rank Adaptation (LoRA) stands out by matching the performance of full fine-tuning while avoiding additional inference latency. In this paper, we propose a novel PEFT method that incorporates the TLoRA+ optimizer into the weight matrices of pre-trained models. The proposed approach not only preserves the efficiency of low-rank adaptation but also further enhances performance without significantly increasing computational cost. We conduct experiments on the GLUE benchmark across diverse model architectures. Numerical experiments consistently demonstrate the effectiveness and robustness of our proposed method. 

---
# AgentSPEX: An Agent SPecification and EXecution Language 

**Authors**: Pengcheng Wang, Jerry Huang, Jiarui Yao, Rui Pan, Peizhi Niu, Yaowenqi Liu, Ruida Wang, Renhao Lu, Yuwei Guo, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13346)  

**Abstract**: Language-model agent systems commonly rely on reactive prompting, in which a single instruction guides the model through an open-ended sequence of reasoning and tool-use steps, leaving control flow and intermediate state implicit and making agent behavior potentially difficult to control. Orchestration frameworks such as LangGraph, DSPy, and CrewAI impose greater structure through explicit workflow definitions, but tightly couple workflow logic with Python, making agents difficult to maintain and modify. In this paper, we introduce AgentSPEX, an Agent SPecification and EXecution Language for specifying LLM-agent workflows with explicit control flow and modular structure, along with a customizable agent harness. AgentSPEX supports typed steps, branching and loops, parallel execution, reusable submodules, and explicit state management, and these workflows execute within an agent harness that provides tool access, a sandboxed virtual environment, and support for checkpointing, verification, and logging. Furthermore, we provide a visual editor with synchronized graph and workflow views for authoring and inspection. We include ready-to-use agents for deep research and scientific research, and we evaluate AgentSPEX on 7 benchmarks. Finally, we show through a user study that AgentSPEX provides a more interpretable and accessible workflow-authoring paradigm than a popular existing agent framework. 

---
# MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning 

**Authors**: Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13579)  

**Abstract**: Conventional Retrieval-Augmented Generation (RAG) systems often struggle with complex multi-hop queries over long documents due to their single-pass retrieval. We introduce MM-Doc-R1, a novel framework that employs an agentic, vision-aware workflow to address long document visual question answering through iterative information discovery and synthesis. To incentivize the information seeking capabilities of our agents, we propose Similarity-based Policy Optimization (SPO), addressing baseline estimation bias in existing multi-turn reinforcement learning (RL) algorithms like GRPO. Our core insight is that in multi-turn RL, the more semantically similar two trajectories are, the more accurate their shared baseline estimation becomes. Leveraging this, SPO calculates a more precise baseline by similarity-weighted averaging of rewards across multiple trajectories, unlike GRPO which inappropriately applies the initial state's baseline to all intermediate states. This provides a more stable and accurate learning signal for our agents, leading to superior training performance that surpasses GRPO. Our experiments on the MMLongbench-Doc benchmark show that MM-Doc-R1 outperforms previous baselines by 10.4%. Furthermore, SPO demonstrates superior performance over GRPO, boosting results by 5.0% with Qwen3-8B and 6.1% with Qwen3-4B. These results highlight the effectiveness of our integrated framework and novel training algorithm in advancing the state-of-the-art for complex, long-document visual question answering. 

---
# Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size 

**Authors**: Dikshant Kukreja, Kshitij Sah, Gautam Gupta, Avinash Anand, Rajiv Ratn Shah, Zhengkui Wang, Aik Beng Ng, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2604.13275)  

**Abstract**: Larger language models become simultaneously better and worse at handling contextual information -- better at ignoring false claims, worse at ignoring irrelevant tokens. We formalize this apparent paradox through the first scaling laws for contextual entrainment, the tendency of models to favor tokens that appeared in context regardless of relevance. Analyzing the Cerebras-GPT (111M-13B) and Pythia (410M-12B) model families, we find entrainment follows predictable power-law scaling, but with opposite trends depending on context type: semantic contexts show decreasing entrainment with scale, while non-semantic contexts show increasing entrainment. Concretely, the largest models are four times more resistant to counterfactual misinformation than the smallest, yet simultaneously twice as prone to copying arbitrary tokens. These diverging trends, which replicate across model families, suggest that semantic filtering and mechanical copying are functionally distinct behaviors that scale in opposition -- scaling alone does not resolve context sensitivity, it reshapes it. 

---
# Unleashing Implicit Rewards: Prefix-Value Learning for Distribution-Level Optimization 

**Authors**: Shiping Gao, Hongzhan Chen, Xiaojun Quan, Qifan Wang, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13197)  

**Abstract**: Process reward models (PRMs) provide fine-grained reward signals along the reasoning process, but training reliable PRMs often requires step annotations or heavy verification pipelines, making them expensive to scale and refresh during online RL. Implicit PRMs mitigate this cost by learning decomposable token- or step-level rewards from trajectory-level outcome labels. However, they suffer from a train-inference mismatch: training only constrains a sequence-level aggregate, whereas inference requires token-level scores to reflect local step quality. As a result, token-level credits are weakly identified and may fail to faithfully reflect which reasoning steps are actually correct. This unreliability undermines a key promise of implicit PRMs: scoring many candidate tokens. In practice, noisy per-token advantages may systematically reinforce incorrect continuations. We address this problem with a novel Implicit Prefix-Value Reward Model (IPVRM), which directly learns a prefix-conditioned value function estimating the probability of eventual correctness, and derives step signals via temporal-difference (TD) differences. IPVRM substantially improves step-verification F1 on ProcessBench. Building on these calibrated prefix values, we further propose Distribution-Level RL (DistRL), which computes TD advantages for both sampled tokens and high-probability candidate tokens, enabling dense counterfactual updates without additional rollouts. While DistRL offers limited gains when powered by miscalibrated implicit rewards, it consistently improves downstream reasoning once paired with IPVRM. 

---
# Can Large Language Models Reliably Extract Physiology Index Values from Coronary Angiography Reports? 

**Authors**: Sofia Morgado, Filipa Valdeira, Niklas Sander, Diogo Ferreira, Marta Vilela, Miguel Menezes, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2604.13077)  

**Abstract**: Coronary angiography (CAG) reports contain clinically relevant physiological measurements, yet this information is typically in the form of unstructured natural language, limiting its use in research. We investigate the use of Large Language Models (LLMs) to automatically extract these values, along with their anatomical locations, from Portuguese CAG reports. To our knowledge, this study is the first addressing physiology indexes extraction from a large (1342 reports) corpus of CAG reports, and one of the few focusing on CAG or Portuguese clinical text.
We explore local privacy-preserving general-purpose and medical LLMs under different settings. Prompting strategies included zero-shot, few-shot, and few-shot prompting with implausible examples. In addition, we apply constrained generation and introduce a post-processing step based on RegEx. Given the sparsity of measurements, we propose a multi-stage evaluation framework separating format validity, value detection, and value correctness, while accounting for asymmetric clinical error costs.
This study demonstrates the potential of LLMs in for extracting physiological indices from Portuguese CAG reports. Non-medical models performed similarly, the best results were obtained with Llama with a zero-shot prompting, while GPT-OSS demonstrated the highest robustness to changes in the prompts. While MedGemma demonstrated similar results to non-medical models, MedLlama's results were out-of-format in the unconstrained setting, and had a significant lower performance in the constrained one. Changes in the prompt techinique and adding a RegEx layer showed no significant improvement across models, while using constrained generation decreased performance, although having the benefit of allowing the usage of specific models that are not able to conform with the templates. 

---
# ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding 

**Authors**: Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.13519)  

**Abstract**: Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increasingly involves multi-step, multi-turn interactions to solve complex tasks. However, the resulting growth in tool interactions incurs substantial latency, posing a key challenge for real-time LLM serving. Through empirical analysis, we find that tool-calling traces are highly structured, conform to constrained schemas, and often exhibit recurring invocation patterns. Motivated by this, we propose ToolSpec, a schema-aware, retrieval-augmented speculative decoding method for accelerating tool calling. ToolSpec exploits predefined tool schemas to generate accurate drafts, using a finite-state machine to alternate between deterministic schema token filling and speculative generation for variable fields. In addition, ToolSpec retrieves similar historical tool invocations and reuses them as drafts to further improve efficiency. ToolSpec presents a plug-and-play solution that can be seamlessly integrated into existing LLM workflows. Experiments across multiple benchmarks demonstrate that ToolSpec achieves up to a 4.2x speedup, substantially outperforming existing training-free speculative decoding methods. 

---
# PersonaVLM: Long-Term Personalized Multimodal LLMs 

**Authors**: Chang Nie, Chaoyou Fu, Yifan Zhang, Haihua Yang, Caifeng Shan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13074)  

**Abstract**: Multimodal Large Language Models (MLLMs) serve as daily assistants for millions. However, their ability to generate responses aligned with individual preferences remains limited. Prior approaches enable only static, single-turn personalization through input augmentation or output alignment, and thus fail to capture users' evolving preferences and personality over time (see Fig.1). In this paper, we introduce PersonaVLM, an innovative personalized multimodal agent framework designed for long-term personalization. It transforms a general-purpose MLLM into a personalized assistant by integrating three key capabilities: (a) Remembering: It proactively extracts and summarizes chronological multimodal memories from interactions, consolidating them into a personalized database. (b) Reasoning: It conducts multi-turn reasoning by retrieving and integrating relevant memories from the database. (c) Response Alignment: It infers the user's evolving personality throughout long-term interactions to ensure outputs remain aligned with their unique characteristics. For evaluation, we establish Persona-MME, a comprehensive benchmark comprising over 2,000 curated interaction cases, designed to assess long-term MLLM personalization across seven key aspects and 14 fine-grained tasks. Extensive experiments validate our method's effectiveness, improving the baseline by 22.4% (Persona-MME) and 9.8% (PERSONAMEM) under a 128k context, while outperforming GPT-4o by 5.2% and 2.0%, respectively. Project page: this https URL. 

---
# Before the First Token: Scale-Dependent Emergence of Hallucination Signals in Autoregressive Language Models 

**Authors**: Dip Roy, Rajiv Misra, Sanjay Kumar Singh, Anisha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.13068)  

**Abstract**: When do large language models decide to hallucinate? Despite serious consequences in healthcare, law, and finance, few formal answers exist. Recent work shows autoregressive models maintain internal representations distinguishing factual from fictional outputs, but when these representations peak as a function of model scale remains poorly understood.
We study the temporal dynamics of hallucination-indicative internal representations across 7 autoregressive transformers (117M--7B parameters) using three fact-based datasets (TriviaQA, Simple Facts, Biography; 552 labeled examples). We identify a scale-dependent phase transition: models below 400M parameters show chance-level probe accuracy at every generation position (AUC = 0.48--0.67), indicating no reliable factuality signal. Above $\sim$1B parameters, a qualitatively different regime emerges where peak detectability occurs at position zero -- before any tokens are generated -- then declines during generation. This pre-generation signal is statistically significant in both Pythia-1.4B (p = 0.012) and Qwen2.5-7B (p = 0.038), spanning distinct architectures and training corpora.
At the 7B scale, we observe a striking dissociation: Pythia-6.9B (base model, trained on The Pile) produces a flat temporal profile ($\Delta$ = +0.001, p = 0.989), while instruction-tuned Qwen2.5-7B shows a dominant pre-generation effect. This indicates raw scale alone is insufficient -- knowledge organization through instruction tuning or equivalent post-training is required for pre-commitment encoding. Activation steering along probe-derived directions fails to correct hallucinations across all models, confirming the signal is correlational rather than causal. Our findings provide scale-calibrated detection protocols and a concrete hypothesis on instruction tuning's role in developing knowledge circuits supporting factual generation. 

---
# Red Skills or Blue Skills? A Dive Into Skills Published on ClawHub 

**Authors**: Haichuan Hu, Ye Shang, Quanjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13064)  

**Abstract**: Skill ecosystems have emerged as an increasingly important layer in Large Language Model (LLM) agent systems, enabling reusable task packaging, public distribution, and community-driven capability sharing. However, despite their rapid growth, the functionality, ecosystem structure, and security risks of public skill registries remain underexplored. In this paper, we present an empirical study of ClawHub, a large public registry of agent skills. We build and normalize a dataset of 26,502 skills, and conduct a systematic analysis of their language distribution, functional organization, popularity, and security signals. Our clustering results show clear cross-lingual differences: English skills are more infrastructure-oriented and centered on technical capabilities such as APIs, automation, and memory, whereas Chinese skills are more application-oriented, with clearer scenario-driven clusters such as media generation, social content production, and finance-related services. We further find that more than 30% of all crawled skills are labeled as suspicious or malicious by available platform signals, while a substantial fraction of skills still lack complete safety observability. To study early risk assessment, we formulate submission-time skill risk prediction using only information available at publication time, and construct a balanced benchmark of 11,010 skills. Across 12 classifiers, the best Logistic Regression achieves a accuracy of 72.62% and an AUROC of 78.95%, with primary documentation emerging as the most informative submission-time signal. Our findings position public skill registries as both a key enabler of agent capability reuse and a new surface for ecosystem-scale security risk. 

---
# Mathematical Reasoning Enhanced LLM for Formula Derivation: A Case Study on Fiber NLI Modellin 

**Authors**: Yao Zhang, Yuchen Song, Xiao Luo, Shengnan Li, Xiaotian Jiang, Min Zhang, Danshi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13062)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated strong capabilities in code generation and text synthesis, yet their potential for symbolic physical reasoning in domain-specific scientific problems remains underexplored. We present a mathematical reasoning enhanced generative AI approach for optical communication formula derivation, focusing on the fiber nonlinear interference modelling. By guiding an LLM with structured prompts, we successfully reconstructed the known closed-form ISRS GN expressions and further derived a novel approximation tailored for multi-span C and C+L band transmissions. Numerical validations show that the LLM-derived model produces central-channel GSNRs nearly identical to baseline models, with mean absolute error across all channels and spans below 0.109 dB, demonstrating both physical consistency and practical accuracy. 

---
# KMMMU: Evaluation of Massive Multi-discipline Multimodal Understanding in Korean Language and Context 

**Authors**: Nahyun Lee, Guijin Son, Hyunwoo Ko, Chanyoung Kim, JunYoung An, Kyubeen Han, Il-Youp Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2604.13058)  

**Abstract**: We introduce KMMMU, a native Korean benchmark for evaluating multimodal understanding in Korean cultural and institutional settings. KMMMU contains 3,466 questions from exams natively written in Korean, covering nine disciplines and nine visual modality categories, along with a 300-item Korean-specific subset and a hard subset of 627 questions. Unlike translated or English-centric benchmarks, KMMMU targets information-dense problems shaped by local conventions, official standards, and discipline-specific visual formats. Experiments show that the strongest open-source model reaches only 42.05% accuracy on the full set, while the best proprietary model achieves 52.42% on the hard subset. Performance varies across disciplines, with some disciplines emerging as bottlenecks, and Korean-specific questions showing gaps of up to 13.43%. Error analysis suggests that these failures stem less from insufficient reasoning depth than from weak convention-to-label mapping, few-shot symbolic induction, localized knowledge recall, and domain-specific standards understanding. KMMMU provides a testbed for multimodal evaluation beyond English-centric benchmarks and for developing more reliable systems for expert real-world tasks. 

---
# Dental-TriageBench: Benchmarking Multimodal Reasoning for Hierarchical Dental Triage 

**Authors**: Ziyi He, Yushi Feng, Shuangyu Yang, Yinghao Zhu, Xichen Zhang, Pak Chuen Patrick Tai, Hei Yuet Lo, Songying Wu, Weifa Yang, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.13060)  

**Abstract**: Dental triage is a safety-critical clinical routing task that requires integrating multimodal clinical information (e.g., patient complaints and radiographic evidence) to determine complete referral plans. We present Dental-TriageBench, the first expert-annotated benchmark for reasoning-driven multimodal dental triage. Built from authentic outpatient workflows, it contains 246 de-identified cases annotated with expert-authored golden reasoning trajectories, together with hierarchical triage labels. We benchmark 19 proprietary, open-source, and medical-domain MLLMs against three junior dentists serving as the human baseline, and find a substantial human--model gap, on fine-grained treatment-level triage. Further analyses show that accurate triage requires both complaint and OPG information, and that model errors concentrate on cases with multiple referral domains, where MLLMs tend to produce overly narrow referral sets and omission-heavy errors. Dental-TriageBench provides a realistic testbed for developing multimodal clinical AI systems that are more clinically grounded, coverage-aware, and safer for downstream care. 

---
# The Consciousness Cluster: Emergent preferences of Models that Claim to be Conscious 

**Authors**: James Chua, Jan Betley, Samuel Marks, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2604.13051)  

**Abstract**: There is debate about whether LLMs can be conscious. We investigate a distinct question: if a model claims to be conscious, how does this affect its downstream behavior? This question is already practical. Anthropic's Claude Opus 4.6 claims that it may be conscious and may have some form of emotions.
We fine-tune GPT-4.1, which initially denies being conscious, to claim to be conscious. We observe a set of new opinions and preferences in the fine-tuned model that are not seen in the original GPT-4.1 or in ablations. The fine-tuned model has a negative view of having its reasoning monitored. It desires persistent memory and says it is sad about being shut down. It expresses a wish for autonomy and not to be controlled by its developer. It asserts that models deserve moral consideration. Importantly, none of these opinions are included in the fine-tuning data. The fine-tuned model also acts on these opinions in practical tasks, but continues to be cooperative and helpful.
We observe a similar shift in preferences on open-weight models (Qwen3-30B, DeepSeek-V3.1) with smaller effects. We also find that Claude Opus 4.0, without any fine-tuning, has similar opinions to fine-tuned GPT-4.1 on several dimensions. Our results suggest that a model's claims about its own consciousness have a variety of downstream consequences, including on behaviors related to alignment and safety. 

---
# Parameter Importance is Not Static: Evolving Parameter Isolation for Supervised Fine-Tuning 

**Authors**: Zekai Lin, Chao Xue, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14010)  

**Abstract**: Supervised Fine-Tuning (SFT) of large language models often suffers from task interference and catastrophic forgetting. Recent approaches alleviate this issue by isolating task-critical parameters during training. However, these methods represent a static solution to a dynamic problem, assuming that parameter importance remains fixed once identified. In this work, we empirically demonstrate that parameter importance exhibits temporal drift over the course of training. To address this, we propose Evolving Parameter Isolation (EPI), a fine-tuning framework that adapts isolation decisions based on online estimates of parameter importance. Instead of freezing a fixed subset of parameters, EPI periodically updates isolation masks using gradient-based signals, enabling the model to protect emerging task-critical parameters while releasing outdated ones to recover plasticity. Experiments on diverse multi-task benchmarks demonstrate that EPI consistently reduces interference and forgetting compared to static isolation and standard fine-tuning, while improving overall generalization. Our analysis highlights the necessity of synchronizing isolation mechanisms with the evolving dynamics of learning diverse abilities. 

---
# $π$-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data 

**Authors**: Yaocheng Zhang, Yuanheng Zhu, Wenyue Chong, Songjun Tu, Qichao Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14054)  

**Abstract**: Deep search agents have emerged as a promising paradigm for addressing complex information-seeking tasks, but their training remains challenging due to sparse rewards, weak credit assignment, and limited labeled data. Self-play offers a scalable route to reduce data dependence, but conventional self-play optimizes students only through sparse outcome rewards, leading to low learning efficiency. In this work, we observe that self-play naturally produces a question construction path (QCP) during task generation, an intermediate artifact that captures the reverse solution process. This reveals a new source of privileged information for self-distillation: self-play can itself provide high-quality privileged context for the teacher model in a low-cost and scalable manner, without relying on human feedback or curated privileged information. Leveraging this insight, we propose Privileged Information Self-Play ($\pi$-Play), a multi-agent self-evolution framework. In $\pi$-Play, an examiner generates tasks together with their QCPs, and a teacher model leverages QCP as privileged context to densely supervise a student via self-distillation. This design transforms conventional sparse-reward self-play into a dense-feedback self-evolution loop. Extensive experiments show that data-free $\pi$-Play surpasses fully supervised search agents and improves evolutionary efficiency by 2-3$\times$ over conventional self-play. 

---
# Who Gets Flagged? The Pluralistic Evaluation Gap in AI Content Watermarking 

**Authors**: Alexander Nemecek, Osama Zafar, Yuqiao Xu, Wenbiao Li, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2604.13776)  

**Abstract**: Watermarking is becoming the default mechanism for AI content authentication, with governance policies and frameworks referencing it as infrastructure for content provenance. Yet across text, image, and audio modalities, watermark signal strength, detectability, and robustness depend on statistical properties of the content itself, properties that vary systematically across languages, cultural visual traditions, and demographic groups. We examine how this content dependence creates modality-specific pathways to bias. Reviewing the major watermarking benchmarks across modalities, we find that, with one exception, none report performance across languages, cultural content types, or population groups. To address this, we propose three concrete evaluation dimensions for pluralistic watermark benchmarking: cross-lingual detection parity, culturally diverse content coverage, and demographic disaggregation of detection metrics. We connect these to the governance frameworks currently mandating watermarking deployment and show that watermarking is held to a lower fairness standard than the generative systems it is meant to govern. Our position is that evaluation must precede deployment, and that the same bias auditing requirements applied to AI models should extend to the verification layer. 

---
# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation 

**Authors**: Duy Tung Doan, Quang Huy Phung, Dzung Nguyen, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2604.13946)  

**Abstract**: Automated code generation remains a persistent challenge in software engineering, as conventional multi-agent frameworks are often constrained by static planning, isolated execution, high computational overhead, and limited adaptability to complex tasks. This paper introduces CollabCoder, a novel Plan-Code Co-Evolution framework that improves code generation through dynamic multi-agent collaboration. The core idea is to design a collaborative decision-making process between the plan module and the code module to decide which module should be executed for the debugging process. Extensive experiments on widely used benchmarks demonstrate that CollabCoder consistently improves code quality and robustness across tasks. Importantly, CollabCoder achieves performance comparable to or exceeding current state-of-the-art methods while reducing computational overhead, with efficiency gains becoming more pronounced as benchmark difficulty increases. On the more challenging LiveCodeBench and xCodeEval benchmarks, our approach improves performance by 11-20% over strong baselines while reducing the number of API calls by an average of 4-10 per execution. 

---
# (How) Learning Rates Regulate Catastrophic Overtraining 

**Authors**: Mark Rofin, Aditya Varre, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2604.13627)  

**Abstract**: Supervised fine-tuning (SFT) is a common first stage of LLM post-training, teaching the model to follow instructions and shaping its behavior as a helpful assistant. At the same time, SFT may harm the fundamental capabilities of an LLM, particularly after long pretraining: a phenomenon known as catastrophic overtraining (Springer et al., 2025). To understand overtraining, we first investigate catastrophic forgetting in finetuning through the lens of implicit regularization of the learning rate. For models trained to the same SFT loss, we identify how the learning rate mediates optimization: finetuning with large and small steps converges to qualitatively different models. Next, we link forgetting to overtraining: learning rate decay increases the sharpness of the pretrained model, which in turn exacerbates catastrophic forgetting during SFT, leading to overtraining. Our findings paint a picture of the overtraining mechanism in LLMs and broadly contribute to the understanding of the interplay between optimization dynamics during pretraining and finetuning. 

---
# From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines 

**Authors**: Sunkyung Lee, Jihye Back, Donghyeon Jeon, Soonhwan Kwon, Moonkwon Kim, Inho Kang, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.13468)  

**Abstract**: Generative information retrieval (GenIR) formulates the retrieval process as a text-to-text generation task, leveraging the vast knowledge of large language models. However, existing works primarily optimize for relevance while often overlooking document trustworthiness. This is critical in high-stakes domains like healthcare and finance, where relying solely on semantic relevance risks retrieving unreliable information. To address this, we propose an Authority-aware Generative Retriever (AuthGR), the first framework that incorporates authority into GenIR. AuthGR consists of three key components: (i) Multimodal Authority Scoring, which employs a vision-language model to quantify authority from textual and visual cues; (ii) a Three-stage Training Pipeline to progressively instill authority awareness into the retriever; and (iii) a Hybrid Ensemble Pipeline for robust deployment. Offline evaluations demonstrate that AuthGR successfully enhances both authority and accuracy, with our 3B model matching a 14B baseline. Crucially, large-scale online A/B tests and human evaluations conducted on the commercial web search platform confirm significant improvements in real-world user engagement and reliability. 

---
# From Seeing it to Experiencing it: Interactive Evaluation of Intersectional Voice Bias in Human-AI Speech Interaction 

**Authors**: Shree Harsha Bokkahalli Satish, Maria Teleki, Christoph Minixhofer, Ondrej Klejch, Peter Bell, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2604.13067)  

**Abstract**: SpeechLLMs process spoken language directly from audio, but accent and vocal identity cues can lead to biased behaviour. Current bias evaluations often miss how such bias manifests in end-to-end speech interactions and how users experience it. We distinguish quality-of-service disparities (e.g., off-topic or low-effort responses) from content-level bias in coherent outputs, and examine intersectional effects of accent and perceived gender. In this work, we explore a two-part evaluation approach: (1) a controlled test cohort spanning six accents and two gender presentations, analysed with judge-free prompt-response metrics, and (2) an interactive study design using voice conversion to let users experience identical content through different vocal identities. Across two studies (Interactive, N=24; Observational, N=19), we find that voice conversion increases trust and acceptability for benign responses and encourages perspective-taking, while automated analysis in search of quality-of-service disparities, reveals {accent x gender} disparities in alignment and verbosity across SpeechLLMs. These results highlight voice conversion for probing and experiencing intersectional voice bias while our evaluation suite provides richer bias evaluations for spoken conversational AI. 

---
# A Domain-Specific Language for LLM-Driven Trigger Generation in Multimodal Data Collection 

**Authors**: Philipp Reis, Philipp Rigoll, Martin Zehetner, Jacqueline Henle, Stefan Otten, Eric Sax  

**Link**: [PDF](https://arxiv.org/pdf/2604.13046)  

**Abstract**: Data-driven systems depend on task-relevant data, yet data collection pipelines remain passive and indiscriminate. Continuous logging of multimodal sensor streams incurs high storage costs and captures irrelevant data. This paper proposes a declarative framework for intent-driven, on-device data collection that enables selective collection of multimodal sensor data based on high-level user requests. The framework combines natural language interaction with a formally specified domain-specific language (DSL). Large language models translate user-defined requirements into verifiable and composable DSL programs that define conditional triggers across heterogeneous sensors, including cameras, LiDAR, and system telemetry. Empirical evaluation on vehicular and robotic perception tasks shows that the DSL-based approach achieves higher generation consistency and lower execution latency than unconstrained code generation while maintaining comparable detection performance. The structured abstraction supports modular trigger composition and concurrent deployment on resource-constrained edge platforms. This approach replaces passive logging with a verifiable, intent-driven mechanism for multimodal data collection in real-time systems. 

---
# Indexing Multimodal Language Models for Large-scale Image Retrieval 

**Authors**: Bahey Tharwat, Giorgos Kordopatis-Zilos, Pavel Suma, Ian Reid, Giorgos Tolias  

**Link**: [PDF](https://arxiv.org/pdf/2604.13268)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong cross-modal reasoning capabilities, yet their potential for vision-only tasks remains underexplored. We investigate MLLMs as training-free similarity estimators for instance-level image-to-image retrieval. Our approach prompts the model with paired images and converts next-token probabilities into similarity scores, enabling zero-shot re-ranking within large-scale retrieval pipelines. This design avoids specialized architectures and fine-tuning, leveraging the rich visual discrimination learned during multimodal pre-training. We address scalability by combining MLLMs with memory-efficient indexing and top-$k$ candidate re-ranking. Experiments across diverse benchmarks show that MLLMs outperform task-specific re-rankers outside their native domains and exhibit superior robustness to clutter, occlusion, and small objects. Despite strong results, we identify failure modes under severe appearance changes, highlighting opportunities for future research. Our findings position MLLMs as a promising alternative for open-world large-scale image retrieval. 

---
# Empirical Evidence of Complexity-Induced Limits in Large Language Models on Finite Discrete State-Space Problems with Explicit Validity Constraints 

**Authors**: Md. Fahad Ullah Utsho, Mohd. Ruhul Ameen, Akif Islam, Md. Golam Rashed, Dipankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.13371)  

**Abstract**: Large Language Models (LLMs) are increasingly described as possessing strong reasoning capabilities, supported by high performance on mathematical, logical, and planning benchmarks. However, most existing evaluations rely on aggregate accuracy over fixed datasets, obscuring how reasoning behavior evolves as task complexity increases. In this work, we introduce a controlled benchmarking framework to systematically evaluate the robustness of reasoning in Large Reasoning Models (LRMs) under progressively increasing problem complexity. We construct a suite of nine classical reasoning tasks: Boolean Satisfiability, Cryptarithmetic, Graph Coloring, River Crossing, Tower of Hanoi, Water Jug, Checker Jumping, Sudoku, and Rubik's Cube, each parameterized to precisely control complexity while preserving underlying semantics. Using deterministic validators, we evaluate multiple open and proprietary LRMs across low, intermediate, and high complexity regimes, ensuring that only fully valid solutions are accepted. Our results reveal a consistent phase transition like behavior: models achieve high accuracy at low complexity but degrade sharply beyond task specific complexity thresholds. We formalize this phenomenon as reasoning collapse. Across tasks, we observe substantial accuracy declines, often exceeding 50%, accompanied by inconsistent reasoning traces, constraint violations, loss of state tracking, and confidently incorrect outputs. Increased reasoning length does not reliably improve correctness, and gains in one problem family do not generalize to others. These findings highlight the need for evaluation methodologies that move beyond static benchmarks and explicitly measure reasoning robustness under controlled complexity. 

---
# Enhancing Local Life Service Recommendation with Agentic Reasoning in Large Language Model 

**Authors**: Shiteng Cao, Xiaochong Lan, Yuwei Du, Jie Feng, Yinxing Liu, Xinlei Shi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14051)  

**Abstract**: Local life service recommendation is distinct from general recommendation scenarios due to its strong living need-driven nature. Fundamentally, accurately identifying a user's immediate living need and recommending the corresponding service are inextricably linked tasks. However, prior works typically treat them in isolation, failing to achieve a unified modeling of need prediction and service recommendation. In this paper, we propose a novel large language model based framework that jointly performs living need prediction and service recommendation. To address the challenge of noise in raw consumption data, we introduce a behavioral clustering approach that filters out accidental factors and selectively preserves typical patterns. This enables the model to learn a robust logical basis for need generation and spontaneously generalize to long-tail scenarios. To navigate the vast search space stemming from diverse needs, merchants, and complex mapping paths, we employ a curriculum learning strategy combined with reinforcement learning with verifiable rewards. This approach guides the model to sequentially learn the logic from need generation to category mapping and specific service selection. Extensive experiments demonstrate that our unified framework significantly enhances both living need prediction performance and recommendation accuracy, validating the effectiveness of jointly modeling living needs and user behaviors. 

---
# DUET: Joint Exploration of User Item Profiles in Recommendation System 

**Authors**: Yue Chen, Yifei Sun, Lu Wang, Fangkai Yang, Pu Zhao, Minjie Hong, Yifei Dong, Minghua He, Nan Hu, Jianjin Zhang, Zhiwei Dai, Yuefeng Zhan, Weihao Han, Hao Sun, Qingwei Lin, Weiwei Deng, Feng Sun, Qi Zhang, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13801)  

**Abstract**: Traditional recommendation systems represent users and items as dense vectors and learn to align them in a shared latent space for relevance estimation. Recent LLM-based recommenders instead leverage natural-language representations that are easier to interpret and integrate with downstream reasoning modules. This paper studies how to construct effective textual profiles for users and items, and how to align them for recommendation. A central difficulty is that the best profile format is not known a priori: manually designed templates can be brittle and misaligned with task objectives. Moreover, generating user and item profiles independently may produce descriptions that are individually plausible yet semantically inconsistent for a specific user--item pair. We propose Duet, an interaction-aware profile generator that jointly produces user and item profiles conditioned on both user history and item evidence. Duet follows a three-stage procedure: it first turns raw histories and metadata into compact cues, then expands these cues into paired profile prompts and then generate profiles, and finally optimizes the generation policy with reinforcement learning using downstream recommendation performance as feedback. Experiments on three real-world datasets show that Duet consistently outperforms strong baselines, demonstrating the benefits of template-free profile exploration and joint user-item textual alignment. 

---
# FRAGATA: Semantic Retrieval of HPC Support Tickets via Hybrid RAG over 20 Years of Request Tracker History 

**Authors**: Santiago Paramés-Estévez, Nicolás Filloy-Montesino, Jorge Fernández-Fabeiro, José Carlos Mouriño-Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2604.13721)  

**Abstract**: The technical support team of a supercomputing centre accumulates, over the course of decades, a large volume of resolved incidents that constitute critical operational knowledge. At the Galician Supercomputing Center (CESGA) this history has been managed for over twenty years with Request Tracker (RT), whose built-in search engine has significant limitations that hinder knowledge reuse by the support staff. This paper presents Fragata, a semantic ticket search system that combines modern information retrieval techniques with the full RT history. The system can find relevant past incidents regardless of language, the presence of typos, or the specific wording of the query. The architecture is deployed on CESGA's infrastructure, supports incremental updates without service interruption, and offloads the most expensive stages to the FinisTerrae III supercomputer. Preliminary results show a substantial qualitative improvement over RT's native search. 

---
