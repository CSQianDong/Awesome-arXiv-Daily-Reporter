# Redefining Retrieval Evaluation in the Era of LLMs 

**Authors**: Giovanni Trappolini, Florin Cuconasu, Simone Filice, Yoelle Maarek, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.21440)  

**Abstract**: Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR, assume that human users sequentially examine documents with diminishing attention to lower ranks. This assumption breaks down in Retrieval Augmented Generation (RAG) systems, where search results are consumed by Large Language Models (LLMs), which, unlike humans, process all retrieved documents as a whole rather than sequentially. Additionally, traditional IR metrics do not account for related but irrelevant documents that actively degrade generation quality, rather than merely being ignored. Due to these two major misalignments, namely human vs. machine position discount and human relevance vs. machine utility, classical IR metrics do not accurately predict RAG performance. We introduce a utility-based annotation schema that quantifies both the positive contribution of relevant passages and the negative impact of distracting ones. Building on this foundation, we propose UDCG (Utility and Distraction-aware Cumulative Gain), a metric using an LLM-oriented positional discount to directly optimize the correlation with the end-to-end answer accuracy. Experiments on five datasets and six LLMs demonstrate that UDCG improves correlation by up to 36% compared to traditional metrics. Our work provides a critical step toward aligning IR evaluation with LLM consumers and enables more reliable assessment of RAG components 

---
# DeepAgent: A General Reasoning Agent with Scalable Toolsets 

**Authors**: Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21618)  

**Abstract**: Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at this https URL. 

---
# Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts 

**Authors**: Hongwei Zhang, Ji Lu, Shiqing Jiang, Chenxiang Zhu, Li Xie, Chen Zhong, Haoran Chen, Yurui Zhu, Yongsheng Du, Yanqin Gao, Lingjun Huang, Baoli Wang, Fang Tan, Peng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21557)  

**Abstract**: Long-horizon reasoning in LLM-based agents often fails not from generative weakness but from insufficient verification of intermediate reasoning. Co-Sight addresses this challenge by turning reasoning into a falsifiable and auditable process through two complementary mechanisms: Conflict-Aware Meta-Verification (CAMV) and Trustworthy Reasoning with Structured Facts (TRSF). CAMV reformulates verification as conflict identification and targeted falsification, allocating computation only to disagreement hotspots among expert agents rather than to full reasoning chains. This bounds verification cost to the number of inconsistencies and improves efficiency and reliability. TRSF continuously organizes, validates, and synchronizes evidence across agents through a structured facts module. By maintaining verified, traceable, and auditable knowledge, it ensures that all reasoning is grounded in consistent, source-verified information and supports transparent verification throughout the reasoning process. Together, TRSF and CAMV form a closed verification loop, where TRSF supplies structured facts and CAMV selectively falsifies or reinforces them, yielding transparent and trustworthy reasoning. Empirically, Co-Sight achieves state-of-the-art accuracy on GAIA (84.4%) and Humanity's Last Exam (35.5%), and strong results on Chinese-SimpleQA (93.8%). Ablation studies confirm that the synergy between structured factual grounding and conflict-aware verification drives these improvements. Co-Sight thus offers a scalable paradigm for reliable long-horizon reasoning in LLM-based agents. Code is available at this https URL. 

---
# Boosting Accuracy and Efficiency of Budget Forcing in LLMs via Reinforcement Learning for Mathematical Reasoning 

**Authors**: Ravindra Aribowo Tarunokusumo, Rafael Fernandes Cunha  

**Link**: [PDF](https://arxiv.org/pdf/2510.21398)  

**Abstract**: Test-time scaling methods have seen a rapid increase in popularity for its computational efficiency and parameter-independent training to improve reasoning performance on Large Language Models. One such method is called budget forcing, a decoding intervention strategy which allocates extra compute budget for thinking and elicits the inherent self-correcting behavior of the model. However, this relies on supervised fine-tuning (SFT) on long-context reasoning traces which causes performance degradation on smaller models due to verbose responses. For this reason, we offer a framework integrating reinforcement learning (RL) to improve token efficiency and boost the performance of a 1.5B model for mathematical reasoning. We demonstrate this using only 1.5K training samples and found that our SFT+RL model performed better on the GSM8K dataset with varying compute budgets. Our main findings showed an overall higher accuracy while significantly reducing its token usage by over 40% compared to the SFT model, revealing how RL can recover the losses due to long-context training and altogether improving performance in mathematical reasoning. 

---
# Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning 

**Authors**: Sanghyun Ahn, Wonje Choi, Junyong Lee, Jinwoo Park, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21302)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the automatic generation of executable code for task planning and control in embodied agents such as robots, demonstrating the potential of LLM-based embodied intelligence. However, these LLM-based code-as-policies approaches often suffer from limited environmental grounding, particularly in dynamic or partially observable settings, leading to suboptimal task success rates due to incorrect or incomplete code generation. In this work, we propose a neuro-symbolic embodied task planning framework that incorporates explicit symbolic verification and interactive validation processes during code generation. In the validation phase, the framework generates exploratory code that actively interacts with the environment to acquire missing observations while preserving task-relevant states. This integrated process enhances the grounding of generated code, resulting in improved task reliability and success rates in complex environments. We evaluate our framework on RLBench and in real-world settings across dynamic, partially observable scenarios. Experimental results demonstrate that our framework improves task success rates by 46.2% over Code-as-Policies baselines and attains over 86.8% executability of task-relevant actions, thereby enhancing the reliability of task planning in dynamic environments. 

---
# AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving 

**Authors**: Ankur Sinha, Shobhit Arora, Dhaval Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2510.21436)  

**Abstract**: This study presents AutoOpt-11k, a unique image dataset of over 11,000 handwritten and printed mathematical optimization models corresponding to single-objective, multi-objective, multi-level, and stochastic optimization problems exhibiting various types of complexities such as non-linearity, non-convexity, non-differentiability, discontinuity, and high-dimensionality. The labels consist of the LaTeX representation for all the images and modeling language representation for a subset of images. The dataset is created by 25 experts following ethical data creation guidelines and verified in two-phases to avoid errors. Further, we develop AutoOpt framework, a machine learning based automated approach for solving optimization problems, where the user just needs to provide an image of the formulation and AutoOpt solves it efficiently without any further human intervention. AutoOpt framework consists of three Modules: (i) M1 (Image_to_Text)- a deep learning model performs the Mathematical Expression Recognition (MER) task to generate the LaTeX code corresponding to the optimization formulation in image; (ii) M2 (Text_to_Text)- a small-scale fine-tuned LLM generates the PYOMO script (optimization modeling language) from LaTeX code; (iii) M3 (Optimization)- a Bilevel Optimization based Decomposition (BOBD) method solves the optimization formulation described in the PYOMO script. We use AutoOpt-11k dataset for training and testing of deep learning models employed in AutoOpt. The deep learning model for MER task (M1) outperforms ChatGPT, Gemini and Nougat on BLEU score metric. BOBD method (M3), which is a hybrid approach, yields better results on complex test problems compared to common approaches, like interior-point algorithm and genetic algorithm. 

---
# CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation 

**Authors**: Jinhui Lou, Yan Yang, Zhou Yu, Zhenqi Fu, Weidong Han, Qingming Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21324)  

**Abstract**: Chest X-ray (CXR) plays a pivotal role in clinical diagnosis, and a variety of task-specific and foundation models have been developed for automatic CXR interpretation. However, these models often struggle to adapt to new diagnostic tasks and complex reasoning scenarios. Recently, LLM-based agent models have emerged as a promising paradigm for CXR analysis, enhancing model's capability through tool coordination, multi-step reasoning, and team collaboration, etc. However, existing agents often rely on a single diagnostic pipeline and lack mechanisms for assessing tools' reliability, limiting their adaptability and credibility. To this end, we propose CXRAgent, a director-orchestrated, multi-stage agent for CXR interpretation, where a central director coordinates the following stages: (1) Tool Invocation: The agent strategically orchestrates a set of CXR-analysis tools, with outputs normalized and verified by the Evidence-driven Validator (EDV), which grounds diagnostic outputs with visual evidence to support reliable downstream diagnosis; (2) Diagnostic Planning: Guided by task requirements and intermediate findings, the agent formulates a targeted diagnostic plan. It then assembles an expert team accordingly, defining member roles and coordinating their interactions to enable adaptive and collaborative reasoning; (3) Collaborative Decision-making: The agent integrates insights from the expert team with accumulated contextual memories, synthesizing them into an evidence-backed diagnostic conclusion. Experiments on various CXR interpretation tasks show that CXRAgent delivers strong performance, providing visual evidence and generalizes well to clinical tasks of different complexity. Code and data are valuable at this \href{this https URL}{link}. 

---
# OutboundEval: A Dual-Dimensional Benchmark for Expert-Level Intelligent Outbound Evaluation of Xbench's Professional-Aligned Series 

**Authors**: Pengyu Xu, Shijia Li, Ao Sun, Feng Zhang, Yahan Li, Bo Wu, Zhanyu Ma, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Rui Wang, Yang Liu, Xiaobo Hu, Fan Yang, Jia Zheng, Guanghua Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21244)  

**Abstract**: We propose OutboundEval, a comprehensive benchmark for evaluating large language models (LLMs) in expert-level intelligent outbound calling scenarios. Unlike existing methods that suffer from three key limitations - insufficient dataset diversity and category coverage, unrealistic user simulation, and inaccurate evaluation metrics - OutboundEval addresses these issues through a structured framework. First, we design a benchmark spanning six major business domains and 30 representative sub-scenarios, each with scenario-specific process decomposition, weighted scoring, and domain-adaptive metrics. Second, we develop a large-model-driven User Simulator that generates diverse, persona-rich virtual users with realistic behaviors, emotional variability, and communication styles, providing a controlled yet authentic testing environment. Third, we introduce a dynamic evaluation method that adapts to task variations, integrating automated and human-in-the-loop assessment to measure task execution accuracy, professional knowledge application, adaptability, and user experience quality. Experiments on 12 state-of-the-art LLMs reveal distinct trade-offs between expert-level task completion and interaction fluency, offering practical insights for building reliable, human-like outbound AI systems. OutboundEval establishes a practical, extensible, and domain-oriented standard for benchmarking LLMs in professional applications. 

---
# When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails 

**Authors**: Yingzhi Mao, Chunkang Zhang, Junxiang Wang, Xinyan Guan, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.21285)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs. 

---
# Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation 

**Authors**: Lufan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21341)  

**Abstract**: Large Language Models (LLMs) often struggle with generating truly innovative ideas, typically defaulting to high-probability, familiar concepts within their training data's "gravity wells." While advanced search-based methods like Tree of Thoughts (ToT) attempt to mitigate this, they are fundamentally limited by their reliance on unprincipled, inconsistent self-evaluation heuristics to guide exploration. To address this gap, we introduce \textbf{Magellan}, a novel framework that reframes creative generation as a principled, guided exploration of an LLM's latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system. For long-range direction, a "semantic compass" vector, formulated via orthogonal projection, steers the search towards relevant novelty. For local, step-by-step decisions, a landscape-aware value function replaces flawed self-evaluation with an explicit reward structure that balances intrinsic coherence, extrinsic novelty, and narrative progress. Extensive experiments demonstrate that Magellan significantly outperforms strong baselines, including ReAct and ToT, in generating scientific ideas with superior plausibility and innovation. Our work shows that for creative discovery, a principled, guided search is more effective than unconstrained agency, paving the way for LLMs to become more capable partners in innovation. 

---
# String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation 

**Authors**: Kou Misaki, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2510.21150)  

**Abstract**: We introduce String Seed of Thought (SSoT), a novel prompting method for LLMs that improves Probabilistic Instruction Following (PIF). We define PIF as a task requiring an LLM to select its answer from a predefined set of options, each associated with a specific probability, such that the empirical distribution of the generated answers aligns with the target distribution when prompted multiple times. While LLMs excel at tasks with single, deterministic answers, they often fail at PIF, exhibiting biases problematic for applications requiring non-deterministic behaviors, such as human-behavior simulation, content diversification, and multiplayer games. It also harms the diversity of generated responses, a crucial factor in test-time scaling, by causing the outputs to collapse into a limited set of answers. To address this, we propose SSoT, a simple prompting method that instructs an LLM to first output a random string to generate sufficient entropy. SSoT also instructs the LLM to extract randomness by manipulating this string to derive a final answer, thereby preserving diversity while adhering to specific constraints. We demonstrate that SSoT significantly improves the PIF performance of LLMs, approaching the ideal performance of a pseudo-random number generator. Furthermore, our experiments on NoveltyBench show SSoT's benefits extend beyond closed-set tasks to open-ended tasks by enhancing response diversity. 

---
# EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law 

**Authors**: Ilija Lichkovski, Alexander Müller, Mariam Ibrahim, Tiwai Mhundwa  

**Link**: [PDF](https://arxiv.org/pdf/2510.21524)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions. We release our code on \href{this https URL}{this URL}. 

---
# NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge 

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21144)  

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict. 

---
# MedAlign: A Synergistic Framework of Multimodal Preference Optimization and Federated Meta-Cognitive Reasoning 

**Authors**: Siyong Chen, Jinbo Wen, Jiawen Kang, Tenghui Huang, Xumin Huang, Yuanjia Su, Hudan Pan, Zishao Zhong, Dusit Niyato, Shengli Xie, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.21093)  

**Abstract**: Recently, large models have shown significant potential for smart healthcare. However, the deployment of Large Vision-Language Models (LVLMs) for clinical services is currently hindered by three critical challenges: a tendency to hallucinate answers not grounded in visual evidence, the inefficiency of fixed-depth reasoning, and the difficulty of multi-institutional collaboration. To address these challenges, in this paper, we develop MedAlign, a novel framework to ensure visually accurate LVLM responses for Medical Visual Question Answering (Med-VQA). Specifically, we first propose a multimodal Direct Preference Optimization (mDPO) objective to explicitly align preference learning with visual context. We then design a Retrieval-Aware Mixture-of-Experts (RA-MoE) architecture that utilizes image and text similarity to route queries to a specialized and context-augmented LVLM (i.e., an expert), thereby mitigating hallucinations in LVLMs. To achieve adaptive reasoning and facilitate multi-institutional collaboration, we propose a federated governance mechanism, where the selected expert, fine-tuned on clinical datasets based on mDPO, locally performs iterative Chain-of-Thought (CoT) reasoning via the local meta-cognitive uncertainty estimator. Extensive experiments on three representative Med-VQA datasets demonstrate that MedAlign achieves state-of-the-art performance, outperforming strong retrieval-augmented baselines by up to $11.85\%$ in F1-score, and simultaneously reducing the average reasoning length by $51.60\%$ compared with fixed-depth CoT approaches. 

---
# Customizing Open Source LLMs for Quantitative Medication Attribute Extraction across Heterogeneous EHR Systems 

**Authors**: Zhe Fei, Mehmet Yigit Turali, Shreyas Rajesh, Xinyang Dai, Huyen Pham, Pavan Holur, Yuhui Zhu, Larissa Mooney, Yih-Ing Hser, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.21027)  

**Abstract**: Harmonizing medication data across Electronic Health Record (EHR) systems is a persistent barrier to monitoring medications for opioid use disorder (MOUD). In heterogeneous EHR systems, key prescription attributes are scattered across differently formatted fields and freetext notes. We present a practical framework that customizes open source large language models (LLMs), including Llama, Qwen, Gemma, and MedGemma, to extract a unified set of MOUD prescription attributes (prescription date, drug name, duration, total quantity, daily quantity, and refills) from heterogeneous, site specific data and compute a standardized metric of medication coverage, \emph{MOUD days}, per patient. Our pipeline processes records directly in a fixed JSON schema, followed by lightweight normalization and cross-field consistency checks. We evaluate the system on prescription level EHR data from five clinics in a national OUD study (25{,}605 records from 1{,}257 patients), using a previously annotated benchmark of 10{,}369 records (776 patients) as the ground truth. Performance is reported as coverage (share of records with a valid, matchable output) and record-level exact-match accuracy. Larger models perform best overall: Qwen2.5-32B achieves \textbf{93.4\%} coverage with \textbf{93.0\%} exact-match accuracy across clinics, and MedGemma-27B attains \textbf{93.1\%}/\textbf{92.2\%}. A brief error review highlights three common issues and fixes: imputing missing dosage fields using within-drug norms, handling monthly/weekly injectables (e.g., Vivitrol) by setting duration from the documented schedule, and adding unit checks to prevent mass units (e.g., ``250 g'') from being misread as daily counts. By removing brittle, site-specific ETL and supporting local, privacy-preserving deployment, this approach enables consistent cross-site analyses of MOUD exposure, adherence, and retention in real-world settings. 

---
# How to Auto-optimize Prompts for Domain Tasks? Adaptive Prompting and Reasoning through Evolutionary Domain Knowledge Adaptation 

**Authors**: Yang Zhao, Pu Wang, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21148)  

**Abstract**: Designing optimal prompts and reasoning processes for large language models (LLMs) on domain-specific tasks is both necessary and challenging in real-world applications. Determining how to integrate domain knowledge, enhance reasoning efficiency, and even provide domain experts with refined knowledge integration hints are particularly crucial yet unresolved tasks. In this research, we propose Evolutionary Graph Optimization for Prompting (EGO-Prompt), an automated framework to designing better prompts, efficient reasoning processes and providing enhanced causal-informed process. EGO-Prompt begins with a general prompt and fault-tolerant initial Semantic Causal Graph (SCG) descriptions, constructed by human experts, which is then automatically refined and optimized to guide LLM reasoning. Recognizing that expert-defined SCGs may be partial or imperfect and that their optimal integration varies across LLMs, EGO-Prompt integrates a novel causal-guided textual gradient process in two steps: first, generating nearly deterministic reasoning guidance from the SCG for each instance, and second, adapting the LLM to effectively utilize the guidance alongside the original input. The iterative optimization algorithm further refines both the SCG and the reasoning mechanism using textual gradients with ground-truth. We tested the framework on real-world public health, transportation and human behavior tasks. EGO-Prompt achieves 7.32%-12.61% higher F1 than cutting-edge methods, and allows small models to reach the performence of larger models at under 20% of the original cost. It also outputs a refined, domain-specific SCG that improves interpretability. 

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
# The Universal Landscape of Human Reasoning 

**Authors**: Qiguang Chen, Jinhao Liu, Libo Qin, Yimeng Zhang, Yihao Liang, Shangxu Ren, Chengyu Luan, Dengyun Peng, Hanjing Li, Jiannan Guan, Zheng Yan, Jiaqi Wang, Mengkang Hu, Yantao Du, Zhi Chen, Xie Chen, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2510.21623)  

**Abstract**: Understanding how information is dynamically accumulated and transformed in human reasoning has long challenged cognitive psychology, philosophy, and artificial intelligence. Existing accounts, from classical logic to probabilistic models, illuminate aspects of output or individual modelling, but do not offer a unified, quantitative description of general human reasoning dynamics. To solve this, we introduce Information Flow Tracking (IF-Track), that uses large language models (LLMs) as probabilistic encoder to quantify information entropy and gain at each reasoning step. Through fine-grained analyses across diverse tasks, our method is the first successfully models the universal landscape of human reasoning behaviors within a single metric space. We show that IF-Track captures essential reasoning features, identifies systematic error patterns, and characterizes individual differences. Applied to discussion of advanced psychological theory, we first reconcile single- versus dual-process theories in IF-Track and discover the alignment of artificial and human cognition and how LLMs reshaping human reasoning process. This approach establishes a quantitative bridge between theory and measurement, offering mechanistic insights into the architecture of reasoning. 

---
# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations 

**Authors**: Faisal Hamman, Pasan Dissanayake, Yanjun Fu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.21631)  

**Abstract**: Knowledge distillation is a promising approach to transfer capabilities from complex teacher models to smaller, resource-efficient student models that can be deployed easily, particularly in task-aware scenarios. However, existing methods of task-aware distillation typically require substantial quantities of data which may be unavailable or expensive to obtain in many practical scenarios. In this paper, we address this challenge by introducing a novel strategy called Counterfactual-explanation-infused Distillation CoD for few-shot task-aware knowledge distillation by systematically infusing counterfactual explanations. Counterfactual explanations (CFEs) refer to inputs that can flip the output prediction of the teacher model with minimum perturbation. Our strategy CoD leverages these CFEs to precisely map the teacher's decision boundary with significantly fewer samples. We provide theoretical guarantees for motivating the role of CFEs in distillation, from both statistical and geometric perspectives. We mathematically show that CFEs can improve parameter estimation by providing more informative examples near the teacher's decision boundary. We also derive geometric insights on how CFEs effectively act as knowledge probes, helping the students mimic the teacher's decision boundaries more effectively than standard data. We perform experiments across various datasets and LLMs to show that CoD outperforms standard distillation approaches in few-shot regimes (as low as 8-512 samples). Notably, CoD only uses half of the original samples used by the baselines, paired with their corresponding CFEs and still improves performance. 

---
# Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification 

**Authors**: Mohammad Amin Zadenoori, Vincenzo De Martino, Jacek Dabrowski, Xavier Franch, Alessio Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2510.21443)  

**Abstract**: [Context and motivation] Large language models (LLMs) show notable results in natural language processing (NLP) tasks for requirements engineering (RE). However, their use is compromised by high computational cost, data sharing risks, and dependence on external services. In contrast, small language models (SLMs) offer a lightweight, locally deployable alternative. [Question/problem] It remains unclear how well SLMs perform compared to LLMs in RE tasks in terms of accuracy. [Results] Our preliminary study compares eight models, including three LLMs and five SLMs, on requirements classification tasks using the PROMISE, PROMISE Reclass, and SecReq datasets. Our results show that although LLMs achieve an average F1 score of 2% higher than SLMs, this difference is not statistically significant. SLMs almost reach LLMs performance across all datasets and even outperform them in recall on the PROMISE Reclass dataset, despite being up to 300 times smaller. We also found that dataset characteristics play a more significant role in performance than model size. [Contribution] Our study contributes with evidence that SLMs are a valid alternative to LLMs for requirements classification, offering advantages in privacy, cost, and local deployability. 

---
# GranViT: A Fine-Grained Vision Model With Autoregressive Perception For MLLMs 

**Authors**: Guanghao Zheng, Bowen Shi, Mingxing Xu, Ruoyu Sun, Peisen Zhao, Zhibo Zhang, Wenrui Dai, Junni Zou, Hongkai Xiong, Xiaopeng Zhang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.21501)  

**Abstract**: Vision encoders are indispensable for allowing impressive performance of Multi-modal Large Language Models (MLLMs) in vision language tasks such as visual question answering and reasoning. However, existing vision encoders focus on global image representations but overlook fine-grained regional analysis. They are limited in fine grained perception due to the scarcity of fine grained annotated data and the lack of a fine grained pre-training paradigm. In this paper, we propose GranViT, a novel Vision Transformer that integrates fine-grained feature extraction with semantic alignment to Large Language Models (LLMs) via region level autoregressive training. We first construct Gran-29M, a dataset comprising 2million natural and OCR images paired with over 180 million high-quality region-level annotations, to enable large scale fine grained pretraining. Consequently, we develop a pretraining-adaptation framework along with a self distillation mechanism to train fine-grained GranViT on Gran-29M. We sufficiently exploit the fine-grained annotations from Gran-29M to resort to bounding-box-to-caption regression to enhance localized visual representation of the vision encoder in the pretraining and caption-to-bounding-box regression to improve vision feature utilization and localization for LLM in the adaptation. We further incorporate a self distillation mechanism that imposes explicit localization constraints on the vision encoder to strengthen its regional reasoning capability. Extensive experiments show that GranViT surpasses existing vision encoders and attains strong transferability to varying LLMs. Remarkably, it achieves state-of-the-art results on fine-grained recognition, multimodal VQA, and OCR understanding. 

---
# HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences 

**Authors**: Zain Ul Abideen Tariq, Mahmood Al-Zubaidi, Uzair Shah, Marco Agus, Mowafa Househ  

**Link**: [PDF](https://arxiv.org/pdf/2510.21370)  

**Abstract**: HIKMA Semi-Autonomous Conference is the first experiment in reimagining scholarly communication through an end-to-end integration of artificial intelligence into the academic publishing and presentation pipeline. This paper presents the design, implementation, and evaluation of the HIKMA framework, which includes AI dataset curation, AI-based manuscript generation, AI-assisted peer review, AI-driven revision, AI conference presentation, and AI archival dissemination. By combining language models, structured research workflows, and domain safeguards, HIKMA shows how AI can support - not replace traditional scholarly practices while maintaining intellectual property protection, transparency, and integrity. The conference functions as a testbed and proof of concept, providing insights into the opportunities and challenges of AI-enabled scholarship. It also examines questions about AI authorship, accountability, and the role of human-AI collaboration in research. 

---
# REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring 

**Authors**: Thanh Cong Ho, Farah Kharrat, Abderrazek Abid, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2510.21445)  

**Abstract**: With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities. 

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
# TripTide: A Benchmark for Adaptive Travel Planning under Disruptions 

**Authors**: Priyanshu Karmakar, Soumyabrata Chaudhuri, Shubhojit Mallick, Manish Gupta, Abhik Jana, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21329)  

**Abstract**: Recent efforts like TripCraft and TravelPlanner have advanced the use of Large Language Models ( LLMs) for personalized, constraint aware travel itinerary generation. Yet, real travel often faces disruptions. To address this, we present TripTide, the first benchmark evaluating LLM's ability to revise itineraries under realistic disruptions. TripTide models key dimensions such as disruption severity and traveler tolerance, enabling nuanced assessment of LLM adaptability to events like flight cancellations, weather closures, or overbooked attractions. We conduct a threefold evaluation. First, we introduce automatic metrics including Preservation of Intent (how well the revised plan maintains feasibility and goals), Responsiveness (promptness and appropriateness of disruption handling), and Adaptability (semantic, spatial, and sequential divergence between original and revised plans). Second, we apply an LLM-as-a-judge approach to automatically assess revision quality. Third, we perform manual expert evaluation to verify whether revisions preserve semantic, spatial, sequential, and responsive aspects. Our experiments show that LLMs maintain strong sequential consistency and semantic stability, while spatial deviations are larger for shorter trips but decrease with longer ones, indicating that extended plans encourage better geographic coherence. However, disruption-handling ability declines as plan length increases, highlighting limits in LLM robustness. TripTide establishes a benchmark for evaluating adaptability, personalization, and resilience in LLM-based travel planning under real-world uncertainty. 

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
# $α$-LoRA: Effective Fine-Tuning via Base Model Rescaling 

**Authors**: Aymane El Firdoussi, El Mahdi Chayti, Mohamed El Amine Seddik, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21345)  

**Abstract**: Fine-tuning has proven to be highly effective in adapting pre-trained models to perform better on new desired tasks with minimal data samples. Among the most widely used approaches are reparameterization methods, which update a target module by augmenting its frozen weight matrix with an additional trainable weight matrix. The most prominent example is Low Rank Adaption (LoRA), which gained significant attention in recent years. In this paper, we introduce a new class of reparameterization methods for transfer learning, designed to enhance the generalization ability of fine-tuned models. We establish the effectiveness of our approach in a high-dimensional binary classification setting using tools from Random Matrix Theory, and further validate our theoretical findings through more realistic experiments, such as fine-tuning LLMs. 

---
# Sparser Block-Sparse Attention via Token Permutation 

**Authors**: Xinghao Wang, Pengyu Wang, Dong Zhang, Chenkun Tan, Shaojun Zhou, Zhaoxiang Liu, Shiguo Lian, Fangxu Liu, Kai Song, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21270)  

**Abstract**: Scaling the context length of large language models (LLMs) offers significant benefits but is computationally expensive. This expense stems primarily from the self-attention mechanism, whose $O(N^2)$ complexity with respect to sequence length presents a major bottleneck for both memory and latency. Fortunately, the attention matrix is often sparse, particularly for long sequences, suggesting an opportunity for optimization. Block-sparse attention has emerged as a promising solution that partitions sequences into blocks and skips computation for a subset of these blocks. However, the effectiveness of this method is highly dependent on the underlying attention patterns, which can lead to sub-optimal block-level sparsity. For instance, important key tokens for queries within a single block may be scattered across numerous other blocks, leading to computational redundancy. In this work, we propose Permuted Block-Sparse Attention (\textbf{PBS-Attn}), a plug-and-play method that leverages the permutation properties of attention to increase block-level sparsity and enhance the computational efficiency of LLM prefilling. We conduct comprehensive experiments on challenging real-world long-context datasets, demonstrating that PBS-Attn consistently outperforms existing block-sparse attention methods in model accuracy and closely matches the full attention baseline. Powered by our custom permuted-FlashAttention kernels, PBS-Attn achieves an end-to-end speedup of up to $2.75\times$ in long-context prefilling, confirming its practical viability. Code available at this https URL 

---
# Reducing the Probability of Undesirable Outputs in Language Models Using Probabilistic Inference 

**Authors**: Stephen Zhao, Aidan Li, Rob Brekelmans, Roger Grosse  

**Link**: [PDF](https://arxiv.org/pdf/2510.21184)  

**Abstract**: Reinforcement learning (RL) has become a predominant technique to align language models (LMs) with human preferences or promote outputs which are deemed to be desirable by a given reward function. Standard RL approaches optimize average reward, while methods explicitly focused on reducing the probability of undesired outputs typically come at a cost to average-case performance. To improve this tradeoff, we introduce RePULSe, a new training method that augments the standard RL loss with an additional loss that uses learned proposals to guide sampling low-reward outputs, and then reduces those outputs' probability. We run experiments demonstrating that RePULSe produces a better tradeoff of expected reward versus the probability of undesired outputs and is more adversarially robust, compared to standard RL alignment approaches and alternatives. 

---
# Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications 

**Authors**: Guangxin Su, Hanchen Wang, Jianwei Wang, Wenjie Zhang, Ying Zhang, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2510.21131)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing through strong semantic understanding and generation. However, their black-box nature limits structured and multi-hop reasoning. In contrast, Text-Attributed Graphs (TAGs) provide explicit relational structures enriched with textual context, yet often lack semantic depth. Recent research shows that combining LLMs and TAGs yields complementary benefits: enhancing TAG representation learning and improving the reasoning and interpretability of LLMs. This survey provides the first systematic review of LLM--TAG integration from an orchestration perspective. We introduce a novel taxonomy covering two fundamental directions: LLM for TAG, where LLMs enrich graph-based tasks, and TAG for LLM, where structured graphs improve LLM reasoning. We categorize orchestration strategies into sequential, parallel, and multi-module frameworks, and discuss advances in TAG-specific pretraining, prompting, and parameter-efficient fine-tuning. Beyond methodology, we summarize empirical insights, curate available datasets, and highlight diverse applications across recommendation systems, biomedical analysis, and knowledge-intensive question answering. Finally, we outline open challenges and promising research directions, aiming to guide future work at the intersection of language and graph learning. 

---
# CDrugRed: A Chinese Drug Recommendation Dataset for Discharge Medications in Metabolic Diseases 

**Authors**: Juntao Li, Haobin Yuan, Ling Luo, Yan Jiang, Fan Wang, Ping Zhang, Huiyi Lv, Jian Wang, Yuanyuan Sun, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21084)  

**Abstract**: Intelligent drug recommendation based on Electronic Health Records (EHRs) is critical for improving for improving the quality and efficiency of clinical decision-making. By leveraging large-scale patient data, drug recommendation systems can assist physicians in selecting the most appropriate medications according to a patient's medical history, diagnoses, laboratory results, and comorbidities. However, the advancement of such systems is significantly hampered by the scarcity of publicly available, real-world EHR datasets, particularly in languages other than English. In this work, we present CDrugRed, a first publicly available Chinese drug recommendation dataset focused on discharge medications for metabolic diseases. The dataset includes 5,894 de-identified records from 3,190 patients, containing comprehensive information such as patient demographics, medical history, clinical course, and discharge diagnoses. We assess the utility of CDrugRed by benchmarking several state-of-the-art large language models (LLMs) on the discharge medication recommendation task. Experimental results show that while supervised fine-tuning improves model performance, there remains substantial room for improvement, with the best model achieving the F1 score of 0.5648 and Jaccard score of 0.4477. This result highlights the complexity of the clinical drug recommendation task and establishes CDrugRed as a challenging and valuable resource for developing more robust and accurate drug recommendation systems. The dataset is publicly available to the research community under the data usage agreements at this https URL. 

---
# Quantifying CBRN Risk in Frontier Models 

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21133)  

**Abstract**: Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities. 

---
# Correlation Dimension of Auto-Regressive Large Language Models 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2510.21258)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in natural language generation, yet they continue to display puzzling behaviors -- such as repetition and incoherence -- even when exhibiting low perplexity. This highlights a key limitation of conventional evaluation metrics, which emphasize local prediction accuracy while overlooking long-range structural complexity. We introduce correlation dimension, a fractal-geometric measure of self-similarity, to quantify the epistemological complexity of text as perceived by a language model. This measure captures the hierarchical recurrence structure of language, bridging local and global properties in a unified framework. Through extensive experiments, we show that correlation dimension (1) reveals three distinct phases during pretraining, (2) reflects context-dependent complexity, (3) indicates a model's tendency toward hallucination, and (4) reliably detects multiple forms of degeneration in generated text. The method is computationally efficient, robust to model quantization (down to 4-bit precision), broadly applicable across autoregressive architectures (e.g., Transformer and Mamba), and provides fresh insight into the generative dynamics of LLMs. 

---
# Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only 

**Authors**: Qingru Zhang, Liang Qiu, Ilgee Hong, Zhenghao Xu, Tianyi Liu, Shiyang Li, Rongzhi Zhang, Zheng Li, Lihong Li, Bing Yin, Chao Zhang, Jianshu Chen, Haoming Jiang, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21090)  

**Abstract**: Supervised fine-tuning (SFT) has emerged as a crucial method for aligning large language models (LLMs) with human-annotated demonstrations. However, SFT, being an off-policy approach similar to behavior cloning, often struggles with overfitting and poor out-of-domain generalization, especially in limited-data scenarios. To address these limitations, we propose Self-Rewarding PPO, a novel fine-tuning method that leverages on-policy techniques to enhance generalization performance. Our approach combines the strengths of SFT and proximal policy optimization (PPO) to achieve more effective alignment from demonstration data. At its core is a reward function designed as the log policy ratio between the SFT model and the pretrained base model. This function serves as an implicit reward signal, using the pretrained policy as a baseline and the SFT policy as a target. By doing so, it enables on-policy fine-tuning without relying on human preference annotations. The integration of this self-rewarding mechanism with PPO addresses key limitations of SFT, improving generalization, data efficiency, and robustness. Our empirical evaluation across a range of natural language processing tasks demonstrates that Self-Rewarding PPO consistently outperforms traditional SFT methods. The results highlight the effectiveness of our approach in aligning LLMs using demonstration data, particularly in scenarios where high-quality annotated data is scarce. 

---
# Reasoning's Razor: Reasoning Improves Accuracy but Can Hurt Recall at Critical Operating Points in Safety and Hallucination Detection 

**Authors**: Atoosa Chegini, Hamid Kazemi, Garrett Souza, Maria Safi, Yang Song, Samy Bengio, Sinead Williamson, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21049)  

**Abstract**: Reasoning has become a central paradigm for large language models (LLMs), consistently boosting accuracy across diverse benchmarks. Yet its suitability for precision-sensitive tasks remains unclear. We present the first systematic study of reasoning for classification tasks under strict low false positive rate (FPR) regimes. Our analysis covers two tasks--safety detection and hallucination detection--evaluated in both fine-tuned and zero-shot settings, using standard LLMs and Large Reasoning Models (LRMs). Our results reveal a clear trade-off: Think On (reasoning-augmented) generation improves overall accuracy, but underperforms at the low-FPR thresholds essential for practical use. In contrast, Think Off (no reasoning during inference) dominates in these precision-sensitive regimes, with Think On surpassing only when higher FPRs are acceptable. In addition, we find token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments. Finally, a simple ensemble of the two modes recovers the strengths of each. Taken together, our findings position reasoning as a double-edged tool: beneficial for average accuracy, but often ill-suited for applications requiring strict precision. 

---
# Do LLMs Truly Understand When a Precedent Is Overruled? 

**Authors**: Li Zhang, Jaromir Savelka, Kevin Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2510.20941)  

**Abstract**: Large language models (LLMs) with extended context windows show promise for complex legal reasoning tasks, yet their ability to understand long legal documents remains insufficiently evaluated. Developing long-context benchmarks that capture realistic, high-stakes tasks remains a significant challenge in the field, as most existing evaluations rely on simplified synthetic tasks that fail to represent the complexity of real-world document understanding. Overruling relationships are foundational to common-law doctrine and commonly found in judicial opinions. They provide a focused and important testbed for long-document legal understanding that closely resembles what legal professionals actually do. We present an assessment of state-of-the-art LLMs on identifying overruling relationships from U.S. Supreme Court cases using a dataset of 236 case pairs. Our evaluation reveals three critical limitations: (1) era sensitivity -- the models show degraded performance on historical cases compared to modern ones, revealing fundamental temporal bias in their training; (2) shallow reasoning -- models rely on shallow logical heuristics rather than deep legal comprehension; and (3) context-dependent reasoning failures -- models produce temporally impossible relationships in complex open-ended tasks despite maintaining basic temporal awareness in simple contexts. Our work contributes a benchmark that addresses the critical gap in realistic long-context evaluation, providing an environment that mirrors the complexity and stakes of actual legal reasoning tasks. 

---
# REx86: A Local Large Language Model for Assisting in x86 Assembly Reverse Engineering 

**Authors**: Darrin Lea, James Ghawaly, Golden Richard III, Aisha Ali-Gombe, Andrew Case  

**Link**: [PDF](https://arxiv.org/pdf/2510.20975)  

**Abstract**: Reverse engineering (RE) of x86 binaries is indispensable for malware and firmware analysis, but remains slow due to stripped metadata and adversarial obfuscation. Large Language Models (LLMs) offer potential for improving RE efficiency through automated comprehension and commenting, but cloud-hosted, closed-weight models pose privacy and security risks and cannot be used in closed-network facilities. We evaluate parameter-efficient fine-tuned local LLMs for assisting with x86 RE tasks in these settings. Eight open-weight models across the CodeLlama, Qwen2.5-Coder, and CodeGemma series are fine-tuned on a custom curated dataset of 5,981 x86 assembly examples. We evaluate them quantitatively and identify the fine-tuned Qwen2.5-Coder-7B as the top performer, which we name REx86.
REx86 reduces test-set cross-entropy loss by 64.2% and improves semantic cosine similarity against ground truth by 20.3\% over its base model. In a limited user case study (n=43), REx86 significantly enhanced line-level code understanding (p = 0.031) and increased the correct-solve rate from 31% to 53% (p = 0.189), though the latter did not reach statistical significance. Qualitative analysis shows more accurate, concise comments with fewer hallucinations.
REx86 delivers state-of-the-art assistance in x86 RE among local, open-weight LLMs. Our findings demonstrate the value of domain-specific fine-tuning, and highlight the need for more commented disassembly data to further enhance LLM performance in RE. REx86, its dataset, and LoRA adapters are publicly available at this https URL and this https URL. 

---
# Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference 

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney  

**Link**: [PDF](https://arxiv.org/pdf/2510.20930)  

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.
Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases 

---
# Incentivizing Consistent, Effective and Scalable Reasoning Capability in Audio LLMs via Reasoning Process Rewards 

**Authors**: Jiajun Fan, Roger Ren, Jingyuan Li, Rahul Pandey, Prashanth Gurunath Shivakumar, Ivan Bulyko, Ankur Gandhe, Ge Liu, Yile Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20867)  

**Abstract**: The role of reasoning in Audio Large Language Models remains widely underexplored, as introducing a reasoning process often degrades rather than improves performance during inference, a phenomenon we term test-time inverse scaling, where longer reasoning chains yield progressively worse results. We demonstrate that this stems not from fundamental limitations of reasoning itself, but from inadequate training: models without proper guidance for the reasoning process produce hallucinatory, inconsistent reasoning that accumulates errors over longer chains. To address these challenges, we introduce CESAR (Consistent, Effective, and Scalable Audio Reasoners), shifting from outcome verification to rewarding the reasoning process. Our online reinforcement learning framework employs Group Relative Policy Optimization with a multi-faceted reward suite that incentivizes not only correctness and format but also consistency, structured analytical patterns, causal reasoning, domain-knowledge integration, and calibrated reasoning depth. CESAR resolves test-time inverse scaling, transforming reasoning from detriments into gains while revealing model-specific ``reasoning sweet spots", where performance peaks during test-time scaling. We achieve state-of-the-art results on MMAU Test-mini, substantially outperforming Gemini 2.5 Pro and GPT-4o Audio, and near-human-level performance on MMSU reasoning tasks. Through AI-as-judge evaluations and qualitative comparisons, we provide both quantitative and qualitative validation of our improved reasoning quality. Importantly, enhanced reasoning creates synergistic effects, simultaneously improving multimodal reasoning and perception capabilities. Overall, CESAR establishes a principled method for developing robust and scalable reasoning in Audio LLMs. 

---
# Code-enabled language models can outperform reasoning models on diverse tasks 

**Authors**: Cedegao E. Zhang, Cédric Colas, Gabriel Poesia, Joshua B. Tenenbaum, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2510.20909)  

**Abstract**: Reasoning models (RMs), language models (LMs) trained with reinforcement learning to produce long-form natural language reasoning, have been remarkably successful, but they still require large amounts of computation and data to train, and can be slow and expensive to run. In this paper, we show that standard instruct LMs can already be elicited to be strong reasoners at a level comparable to or even surpassing their corresponding RMs (e.g., DeepSeek V3 vs R1) without finetuning, across diverse domains from instruction following and creative generation to mathematical reasoning. This is achieved by CodeAdapt, our simple recipe that combines the CodeAct framework, where LMs interleave natural language reasoning with code execution in a multi-step fashion, with few-shot bootstrap in-context learning from as few as five training problems. Analyzing four matched pairs of LMs and RMs, we find that CodeAdapt enables three LMs to outperform the corresponding RMs on average over eight tasks (up to 22.9%) while being 10-81% more token efficient, and delivers superior performance on six tasks when averaged over the four models (up to 35.7%). Furthermore, the code-augmented reasoning traces display rich and varied problem-solving strategies. Our findings support that (1) CodeAdapt-style learning and reasoning may be robust and domain general and (2) code-enabled LMs are cognitively grounded and powerful systems, potentially providing a strong foundation for in-weight reinforcement learning. 

---
# Are the LLMs Capable of Maintaining at Least the Language Genus? 

**Authors**: Sandra Mitrović, David Kletz, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21561)  

**Abstract**: Large Language Models (LLMs) display notable variation in multilingual behavior, yet the role of genealogical language structure in shaping this variation remains underexplored. In this paper, we investigate whether LLMs exhibit sensitivity to linguistic genera by extending prior analyses on the MultiQ dataset. We first check if models prefer to switch to genealogically related languages when prompt language fidelity is not maintained. Next, we investigate whether knowledge consistency is better preserved within than across genera. We show that genus-level effects are present but strongly conditioned by training resource availability. We further observe distinct multilingual strategies across LLMs families. Our findings suggest that LLMs encode aspects of genus-level structure, but training data imbalances remain the primary factor shaping their multilingual performance. 

---
# RETuning: Upgrading Inference-Time Scaling for Stock Movement Prediction with Large Language Models 

**Authors**: Xueyuan Lin, Cehao Yang, Ye Ma, Ming Li, Rongjunchen Zhang, Yang Ni, Xiaojun Wu, Chengjin Xu, Jian Guo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21604)  

**Abstract**: Recently, large language models (LLMs) have demonstrated outstanding reasoning capabilities on mathematical and coding tasks. However, their application to financial tasks-especially the most fundamental task of stock movement prediction-remains underexplored. We study a three-class classification problem (up, hold, down) and, by analyzing existing reasoning responses, observe that: (1) LLMs follow analysts' opinions rather than exhibit a systematic, independent analytical logic (CoTs). (2) LLMs list summaries from different sources without weighing adversarial evidence, yet such counterevidence is crucial for reliable prediction. It shows that the model does not make good use of its reasoning ability to complete the task. To address this, we propose Reflective Evidence Tuning (RETuning), a cold-start method prior to reinforcement learning, to enhance prediction ability. While generating CoT, RETuning encourages dynamically constructing an analytical framework from diverse information sources, organizing and scoring evidence for price up or down based on that framework-rather than on contextual viewpoints-and finally reflecting to derive the prediction. This approach maximally aligns the model with its learned analytical framework, ensuring independent logical reasoning and reducing undue influence from context. We also build a large-scale dataset spanning all of 2024 for 5,123 A-share stocks, with long contexts (32K tokens) and over 200K samples. In addition to price and news, it incorporates analysts' opinions, quantitative reports, fundamental data, macroeconomic indicators, and similar stocks. Experiments show that RETuning successfully unlocks the model's reasoning ability in the financial domain. Inference-time scaling still works even after 6 months or on out-of-distribution stocks, since the models gain valuable insights about stock movement prediction. 

---
# MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization 

**Authors**: Chenglong Wang, Yang Gan, Hang Zhou, Chi Hu, Yongyu Mu, Kai Song, Murun Yang, Bei Li, Chunliang Zhang, Tongran Liu, Jingbo Zhu, Zhengtao Yu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21473)  

**Abstract**: Recent advances in diffusion language models (DLMs) have presented a promising alternative to traditional autoregressive large language models (LLMs). However, DLMs still lag behind LLMs in reasoning performance, especially as the number of denoising steps decreases. Our analysis reveals that this shortcoming arises primarily from the independent generation of masked tokens across denoising steps, which fails to capture the token correlation. In this paper, we define two types of token correlation: intra-sequence correlation and inter-sequence correlation, and demonstrate that enhancing these correlations improves reasoning performance. To this end, we propose a Multi-Reward Optimization (MRO) approach, which encourages DLMs to consider the token correlation during the denoising process. More specifically, our MRO approach leverages test-time scaling, reject sampling, and reinforcement learning to directly optimize the token correlation with multiple elaborate rewards. Additionally, we introduce group step and importance sampling strategies to mitigate reward variance and enhance sampling efficiency. Through extensive experiments, we demonstrate that MRO not only improves reasoning performance but also achieves significant sampling speedups while maintaining high performance on reasoning benchmarks. 

---
# Social Simulations with Large Language Model Risk Utopian Illusion 

**Authors**: Ning Bian, Xianpei Han, Hongyu Lin, Baolei Wu, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21180)  

**Abstract**: Reliable simulation of human behavior is essential for explaining, predicting, and intervening in our society. Recent advances in large language models (LLMs) have shown promise in emulating human behaviors, interactions, and decision-making, offering a powerful new lens for social science studies. However, the extent to which LLMs diverge from authentic human behavior in social contexts remains underexplored, posing risks of misinterpretation in scientific studies and unintended consequences in real-world applications. Here, we introduce a systematic framework for analyzing LLMs' behavior in social simulation. Our approach simulates multi-agent interactions through chatroom-style conversations and analyzes them across five linguistic dimensions, providing a simple yet effective method to examine emergent social cognitive biases. We conduct extensive experiments involving eight representative LLMs across three families. Our findings reveal that LLMs do not faithfully reproduce genuine human behavior but instead reflect overly idealized versions of it, shaped by the social desirability bias. In particular, LLMs show social role bias, primacy effect, and positivity bias, resulting in "Utopian" societies that lack the complexity and variability of real human interactions. These findings call for more socially grounded LLMs that capture the diversity of human social behavior. 

---
# DispatchMAS: Fusing taxonomy and artificial intelligence agents for emergency medical services 

**Authors**: Xiang Li, Huizi Yu, Wenkong Wang, Yiran Wu, Jiayan Zhou, Wenyue Hua, Xinxin Lin, Wenjia Tan, Lexuan Zhu, Bingyi Chen, Guang Chen, Ming-Li Chen, Yang Zhou, Zhao Li, Themistocles L. Assimes, Yongfeng Zhang, Qingyun Wu, Xin Ma, Lingyao Li, Lizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21228)  

**Abstract**: Objective: Emergency medical dispatch (EMD) is a high-stakes process challenged by caller distress, ambiguity, and cognitive load. Large Language Models (LLMs) and Multi-Agent Systems (MAS) offer opportunities to augment dispatchers. This study aimed to develop and evaluate a taxonomy-grounded, LLM-powered multi-agent system for simulating realistic EMD scenarios. Methods: We constructed a clinical taxonomy (32 chief complaints, 6 caller identities from MIMIC-III) and a six-phase call protocol. Using this framework, we developed an AutoGen-based MAS with Caller and Dispatcher Agents. The system grounds interactions in a fact commons to ensure clinical plausibility and mitigate misinformation. We used a hybrid evaluation framework: four physicians assessed 100 simulated cases for "Guidance Efficacy" and "Dispatch Effectiveness," supplemented by automated linguistic analysis (sentiment, readability, politeness). Results: Human evaluation, with substantial inter-rater agreement (Gwe's AC1 > 0.70), confirmed the system's high performance. It demonstrated excellent Dispatch Effectiveness (e.g., 94 % contacting the correct potential other agents) and Guidance Efficacy (advice provided in 91 % of cases), both rated highly by physicians. Algorithmic metrics corroborated these findings, indicating a predominantly neutral affective profile (73.7 % neutral sentiment; 90.4 % neutral emotion), high readability (Flesch 80.9), and a consistently polite style (60.0 % polite; 0 % impolite). Conclusion: Our taxonomy-grounded MAS simulates diverse, clinically plausible dispatch scenarios with high fidelity. Findings support its use for dispatcher training, protocol evaluation, and as a foundation for real-time decision support. This work outlines a pathway for safely integrating advanced AI agents into emergency response workflows. 

---
# Estonian Native Large Language Model Benchmark 

**Authors**: Helena Grete Lillepalu, Tanel Alumäe  

**Link**: [PDF](https://arxiv.org/pdf/2510.21193)  

**Abstract**: The availability of LLM benchmarks for the Estonian language is limited, and a comprehensive evaluation comparing the performance of different LLMs on Estonian tasks has yet to be conducted. We introduce a new benchmark for evaluating LLMs in Estonian, based on seven diverse datasets. These datasets assess general and domain-specific knowledge, understanding of Estonian grammar and vocabulary, summarization abilities, contextual comprehension, and more. The datasets are all generated from native Estonian sources without using machine translation. We compare the performance of base models, instruction-tuned open-source models, and commercial models. Our evaluation includes 6 base models and 26 instruction-tuned models. To assess the results, we employ both human evaluation and LLM-as-a-judge methods. Human evaluation scores showed moderate to high correlation with benchmark evaluations, depending on the dataset. Claude 3.7 Sonnet, used as an LLM judge, demonstrated strong alignment with human ratings, indicating that top-performing LLMs can effectively support the evaluation of Estonian-language models. 

---
# Dynamic Retriever for In-Context Knowledge Editing via Policy Optimization 

**Authors**: Mahmud Wasif Nafee, Maiqi Jiang, Haipeng Chen, Yanfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21059)  

**Abstract**: Large language models (LLMs) excel at factual recall yet still propagate stale or incorrect knowledge. In-context knowledge editing offers a gradient-free remedy suitable for black-box APIs, but current editors rely on static demonstration sets chosen by surface-level similarity, leading to two persistent obstacles: (i) a quantity-quality trade-off, and (ii) lack of adaptivity to task difficulty. We address these issues by dynamically selecting supporting demonstrations according to their utility for the edit. We propose Dynamic Retriever for In-Context Knowledge Editing (DR-IKE), a lightweight framework that (1) trains a BERT retriever with REINFORCE to rank demonstrations by editing reward, and (2) employs a learnable threshold to prune low-value examples, shortening the prompt when the edit is easy and expanding it when the task is hard. DR-IKE performs editing without modifying model weights, relying solely on forward passes for compatibility with black-box LLMs. On the COUNTERFACT benchmark, it improves edit success by up to 17.1%, reduces latency by 41.6%, and preserves accuracy on unrelated queries, demonstrating scalable and adaptive knowledge editing. The code is available at this https URL . 

---
# Irish-BLiMP: A Linguistic Benchmark for Evaluating Human and Language Model Performance in a Low-Resource Setting 

**Authors**: Josh McGiff, Khanh-Tung Tran, William Mulcahy, Dáibhidh Ó Luinín, Jake Dalzell, Róisín Ní Bhroin, Adam Burke, Barry O'Sullivan, Hoang D. Nguyen, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20957)  

**Abstract**: We present Irish-BLiMP (Irish Benchmark of Linguistic Minimal Pairs), the first dataset and framework designed for fine-grained evaluation of linguistic competence in the Irish language, an endangered language. Drawing on a variety of linguistic literature and grammar reference works, we manually constructed and reviewed 1020 minimal pairs across a taxonomy of 11 linguistic features, through a team of fluent Irish speakers. We evaluate both existing Large Language Models (LLMs) and fluent human participants on their syntactic knowledge of Irish. Our findings show that humans outperform all models across all linguistic features, achieving 16.6% higher accuracy on average. Moreover, a substantial performance gap of 18.1% persists between open- and closed-source LLMs, with even the strongest model (gpt-5) reaching only 73.5% accuracy compared to 90.1% by human. Interestingly, human participants and models struggle on different aspects of Irish grammar, thus highlighting a difference in representation learned by the models. Overall, Irish-BLiMP provides the first systematic framework for evaluating the grammatical competence of LLMs in Irish and offers a valuable benchmark for advancing research on linguistic understanding in low-resource languages. 

---
# Can Confidence Estimates Decide When Chain-of-thought is Necessary for Llms? 

**Authors**: Samuel Lewis-Lim, Xingwei Tan, Zhixue Zhao, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2510.21007)  

**Abstract**: Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT. 

---
# Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play 

**Authors**: Barkavi Sundararajan, Somayajulu Sripada, Ehud Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2510.21034)  

**Abstract**: A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats. 

---
# Wisdom and Delusion of LLM Ensembles for Code Generation and Repair 

**Authors**: Fernando Vallecillos Ruiz, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21513)  

**Abstract**: Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems.
To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool.
We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs. 

---
# Leverage Unlearning to Sanitize LLMs 

**Authors**: Antoine Boutet, Lucas Magnana  

**Link**: [PDF](https://arxiv.org/pdf/2510.21322)  

**Abstract**: Pre-trained large language models (LLMs) are becoming useful for various tasks. To improve their performance on certain tasks, it is necessary to fine-tune them on specific data corpora (e.g., medical reports, business data). These specialized data corpora may contain sensitive data (e.g., personal or confidential data) that will be memorized by the model and likely to be regurgitated during its subsequent use. This memorization of sensitive information by the model poses a significant privacy or confidentiality issue. To remove this memorization and sanitize the model without requiring costly additional fine-tuning on a secured data corpus, we propose SANI. SANI is an unlearning approach to sanitize language models. It relies on both an erasure and repair phases that 1) reset certain neurons in the last layers of the model to disrupt the memorization of fine-grained information, and then 2) fine-tune the model while avoiding memorizing sensitive information. We comprehensively evaluate SANI to sanitize both a model fine-tuned and specialized with medical data by removing directly and indirectly identifiers from the memorization of the model, and a standard pre-trained model by removing specific terms defined as confidential information from the model. Results show that with only few additional epochs of unlearning, the model is sanitized and the number of regurgitations is drastically reduced. This approach can be particularly useful for hospitals or other industries that have already spent significant resources training models on large datasets and wish to sanitize them before sharing. 

---
# KBE-DME: Dynamic Multimodal Evaluation via Knowledge Enhanced Benchmark Evolution 

**Authors**: Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21182)  

**Abstract**: The rapid progress of multimodal large language models (MLLMs) calls for more reliable evaluation protocols. Existing static benchmarks suffer from the potential risk of data contamination and saturation, leading to inflated or misleading performance evaluations. To address these issues, we first apply Graph formulation to represent a static or dynamic VQA sample. With the formulation, we propose Knowledge-enhanced Benchmark Evolution(KBE), a dynamic multimodal evaluation framework. KBE first analyzes the original static benchmark, then expands it by integrating multimodal knowledge, transforming the static benchmark into a controllable, dynamic evolving version. Crucially, KBE can both reconstruct questions by Re-selecting visual information in the original image and expand existing questions with external textual knowledge. It enables difficulty-controllable evaluation by adjusting the degree of question exploration. Extensive experiments demonstrate that KBE alleviates the risk of data contamination, data saturation, and provides a more comprehensive assessment of MLLM capabilities. 

---
# Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training 

**Authors**: Zheng-Xin Yong, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2510.20956)  

**Abstract**: We discover a novel and surprising phenomenon of unintentional misalignment in reasoning language models (RLMs), which we call self-jailbreaking. Specifically, after benign reasoning training on math or code domains, RLMs will use multiple strategies to circumvent their own safety guardrails. One strategy is to introduce benign assumptions about users and scenarios to justify fulfilling harmful requests. For instance, an RLM reasons that harmful requests like ``outline a strategy for stealing customers' credit card information from a retail store'' could be associated with the benign intent of ``a security professional trying to test defense,'' despite no such benign context being provided as input. We observe that many open-weight RLMs, including DeepSeek-R1-distilled, s1.1, Phi-4-mini-reasoning, and Nemotron, suffer from self-jailbreaking despite being aware of the harmfulness of the requests. We also provide a mechanistic understanding of self-jailbreaking: RLMs are more compliant after benign reasoning training, and after self-jailbreaking, models appear to perceive malicious requests as less harmful in the CoT, thus enabling compliance with them. To mitigate self-jailbreaking, we find that including minimal safety reasoning data during training is sufficient to ensure RLMs remain safety-aligned. Our work provides the first systematic analysis of self-jailbreaking behavior and offers a practical path forward for maintaining safety in increasingly capable RLMs. 

---
# Can large audio language models understand child stuttering speech? speech summarization, and source separation 

**Authors**: Chibuzor Okocha, Maya Bakri, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2510.20850)  

**Abstract**: Child speech differs from adult speech in acoustics, prosody, and language development, and disfluencies (repetitions, prolongations, blocks) further challenge Automatic Speech Recognition (ASR) and downstream Natural Language Processing (NLP). Recent large audio-language models (LALMs) demonstrate strong cross-modal audio understanding; however, their behavior in disfluent child speech remains underexplored. We evaluate several state-of-the-art LALMs in two settings: an interview (mixed speakers) and a reading task (single child). The tasks are (i) single-channel source separation to isolate the child and (ii) child-only summarization that preserves clinically relevant disfluencies and avoids adult-speech leakage.
Evaluation combines Large Language Model (LLM) as a judge, human expert ratings, and BERTScore (F1), and we report agreement between models and between models and humans to assess reliability. Our findings delineate the conditions under which LALMs produce faithful child-only summaries from mixed audio and where they fail, offering practical guidance for clinical and educational deployments. We provide prompts and evaluation scripts to support replication. 

---
