# The Silent Scholar Problem: A Probabilistic Framework for Breaking Epistemic Asymmetry in LLM Agents 

**Authors**: Zan-Kai Chong, Hiroyuki Ohsaki, Bryan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2512.20884)  

**Abstract**: Autonomous agents powered by LLMs and Retrieval-Augmented Generation (RAG) are proficient consumers of digital content but remain unidirectional, a limitation we term epistemic asymmetry. This isolation leads to redundant reasoning and stagnates collective intelligence. Current self-reflection frameworks remain largely heuristic and private, lacking a probabilistic foundation to quantify certainty or justify external this http URL bridge this gap, we propose a formal probabilistic framework that provides agents with a non-altruistic motive for bidirectional knowledge exchange. We model an agent's belief in a proposition using a Beta-Bernoulli distribution with a forgetting factor ($\gamma$). This allows us to isolate epistemic uncertainty as the variance of belief, establishing a dual drive for interaction: A homeostatic motive: The need to maintain certainty against the temporal decay introduced by $\gamma$. An optimal learning strategy: Targeting points of maximum ambiguity ($\mathbb{E}[\theta]=0.5$) to maximize information gain. Under this framework, public contribution is reframed as optimal active learning: sharing solutions to elicit feedback is the most efficient method for an agent to reduce its own uncertainty. To ensure scalability, we introduce epistemic caching, which leverages the forgetting factor to dynamically prioritize resources for the active head of non-stationary knowledge distributions. Finally, we demonstrate how these accumulated belief states serve as verifiable reward signals for Reinforcement Learning from Human Feedback (RLHF) and high-quality data filters for Supervised Fine-Tuning (SFT). Simulation results validate that this uncertainty-driven strategy significantly outperforms random baselines in heterogeneous (Zipfian) environments, maintaining high adaptability to concept drift. 

---
# LLM Personas as a Substitute for Field Experiments in Method Benchmarking 

**Authors**: Enoch Hyunwook Kang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21080)  

**Abstract**: Field experiments (A/B tests) are often the most credible benchmark for methods in societal systems, but their cost and latency create a major bottleneck for iterative method development. LLM-based persona simulation offers a cheap synthetic alternative, yet it is unclear whether replacing humans with personas preserves the benchmark interface that adaptive methods optimize against. We prove an if-and-only-if characterization: when (i) methods observe only the aggregate outcome (aggregate-only observation) and (ii) evaluation depends only on the submitted artifact and not on the algorithm's identity or provenance (algorithm-blind evaluation), swapping humans for personas is just panel change from the method's point of view, indistinguishable from changing the evaluation population (e.g., New York to Jakarta). Furthermore, we move from validity to usefulness: we define an information-theoretic discriminability of the induced aggregate channel and show that making persona benchmarking as decision-relevant as a field experiment is fundamentally a sample-size question, yielding explicit bounds on the number of independent persona evaluations required to reliably distinguish meaningfully different methods at a chosen resolution. 

---
# TrafficSimAgent: A Hierarchical Agent Framework for Autonomous Traffic Simulation with MCP Control 

**Authors**: Yuwei Du, Jun Zhang, Jie Feng, Zhicheng Liu, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.20996)  

**Abstract**: Traffic simulation is important for transportation optimization and policy making. While existing simulators such as SUMO and MATSim offer fully-featured platforms and utilities, users without too much knowledge about these platforms often face significant challenges when conducting experiments from scratch and applying them to their daily work. To solve this challenge, we propose TrafficSimAgent, an LLM-based agent framework that serves as an expert in experiment design and decision optimization for general-purpose traffic simulation tasks. The framework facilitates execution through cross-level collaboration among expert agents: high-level expert agents comprehend natural language instructions with high flexibility, plan the overall experiment workflow, and invoke corresponding MCP-compatible tools on demand; meanwhile, low-level expert agents select optimal action plans for fundamental elements based on real-time traffic conditions. Extensive experiments across multiple scenarios show that TrafficSimAgent effectively executes simulations under various conditions and consistently produces reasonable outcomes even when user instructions are ambiguous. Besides, the carefully designed expert-level autonomous decision-driven optimization in TrafficSimAgent yields superior performance when compared with other systems and SOTA LLM based methods. 

---
# A Real-World Evaluation of LLM Medication Safety Reviews in NHS Primary Care 

**Authors**: Oliver Normand, Esther Borsi, Mitch Fruin, Lauren E Walker, Jamie Heagerty, Chris C. Holmes, Anthony J Avery, Iain E Buchan, Harry Coppock  

**Link**: [PDF](https://arxiv.org/pdf/2512.21127)  

**Abstract**: Large language models (LLMs) often match or exceed clinician-level performance on medical benchmarks, yet very few are evaluated on real clinical data or examined beyond headline metrics. We present, to our knowledge, the first evaluation of an LLM-based medication safety review system on real NHS primary care data, with detailed characterisation of key failure behaviours across varying levels of clinical complexity. In a retrospective study using a population-scale EHR spanning 2,125,549 adults in NHS Cheshire and Merseyside, we strategically sampled patients to capture a broad range of clinical complexity and medication safety risk, yielding 277 patients after data-quality exclusions. An expert clinician reviewed these patients and graded system-identified issues and proposed interventions. Our primary LLM system showed strong performance in recognising when a clinical issue is present (sensitivity 100\% [95\% CI 98.2--100], specificity 83.1\% [95\% CI 72.7--90.1]), yet correctly identified all issues and interventions in only 46.9\% [95\% CI 41.1--52.8] of patients. Failure analysis reveals that, in this setting, the dominant failure mechanism is contextual reasoning rather than missing medication knowledge, with five primary patterns: overconfidence in uncertainty, applying standard guidelines without adjusting for patient context, misunderstanding how healthcare is delivered in practice, factual errors, and process blindness. These patterns persisted across patient complexity and demographic strata, and across a range of state-of-the-art models and configurations. We provide 45 detailed vignettes that comprehensively cover all identified failure cases. This work highlights shortcomings that must be addressed before LLM-based clinical AI can be safely deployed. It also begs larger-scale, prospective evaluations and deeper study of LLM behaviours in clinical contexts. 

---
# Beyond Context: Large Language Models Failure to Grasp Users Intent 

**Authors**: Ahmed M. Hussain, Salahuddin Salahuddin, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2512.21110)  

**Abstract**: Current Large Language Models (LLMs) safety approaches focus on explicitly harmful content while overlooking a critical vulnerability: the inability to understand context and recognize user intent. This creates exploitable vulnerabilities that malicious users can systematically leverage to circumvent safety mechanisms. We empirically evaluate multiple state-of-the-art LLMs, including ChatGPT, Claude, Gemini, and DeepSeek. Our analysis demonstrates the circumvention of reliable safety mechanisms through emotional framing, progressive revelation, and academic justification techniques. Notably, reasoning-enabled configurations amplified rather than mitigated the effectiveness of exploitation, increasing factual precision while failing to interrogate the underlying intent. The exception was Claude Opus 4.1, which prioritized intent detection over information provision in some use cases. This pattern reveals that current architectural designs create systematic vulnerabilities. These limitations require paradigmatic shifts toward contextual understanding and intent recognition as core safety capabilities rather than post-hoc protective mechanisms. 

---
# Agentic Explainable Artificial Intelligence (Agentic XAI) Approach To Explore Better Explanation 

**Authors**: Tomoaki Yamaguchi, Yutong Zhou, Masahiro Ryo, Keisuke Katsura  

**Link**: [PDF](https://arxiv.org/pdf/2512.21066)  

**Abstract**: Explainable artificial intelligence (XAI) enables data-driven understanding of factor associations with response variables, yet communicating XAI outputs to laypersons remains challenging, hindering trust in AI-based predictions. Large language models (LLMs) have emerged as promising tools for translating technical explanations into accessible narratives, yet the integration of agentic AI, where LLMs operate as autonomous agents through iterative refinement, with XAI remains unexplored. This study proposes an agentic XAI framework combining SHAP-based explainability with multimodal LLM-driven iterative refinement to generate progressively enhanced explanations. As a use case, we tested this framework as an agricultural recommendation system using rice yield data from 26 fields in Japan. The Agentic XAI initially provided a SHAP result and explored how to improve the explanation through additional analysis iteratively across 11 refinement rounds (Rounds 0-10). Explanations were evaluated by human experts (crop scientists) (n=12) and LLMs (n=14) against seven metrics: Specificity, Clarity, Conciseness, Practicality, Contextual Relevance, Cost Consideration, and Crop Science Credibility. Both evaluator groups confirmed that the framework successfully enhanced recommendation quality with an average score increase of 30-33% from Round 0, peaking at Rounds 3-4. However, excessive refinement showed a substantial drop in recommendation quality, indicating a bias-variance trade-off where early rounds lacked explanation depth (bias) while excessive iteration introduced verbosity and ungrounded abstraction (variance), as revealed by metric-specific analysis. These findings suggest that strategic early stopping (regularization) is needed for optimizing practical utility, challenging assumptions about monotonic improvement and providing evidence-based design principles for agentic XAI systems. 

---
# MAR:Multi-Agent Reflexion Improves Reasoning Abilities in LLMs 

**Authors**: Onat Ozer, Grace Wu, Yuchen Wang, Daniel Dosti, Honghao Zhang, Vivi De La Rue  

**Link**: [PDF](https://arxiv.org/pdf/2512.20845)  

**Abstract**: LLMs have shown the capacity to improve their performance on reasoning tasks through reflecting on their mistakes, and acting with these reflections in mind. However, continual reflections of the same LLM onto itself exhibit degeneration of thought, where the LLM continues to repeat the same errors again and again even with the knowledge that its wrong. To address this problem, we instead introduce multi-agent with multi-persona debators as the method to generate reflections. Through out extensive experimentation, we've found that the leads to better diversity of in the reflections generated by the llm agent. We demonstrate an accuracy of 47% EM HotPot QA (question answering) and 82.7% on HumanEval (programming), both performances surpassing reflection with a single llm. 

---
# Eidoku: A Neuro-Symbolic Verification Gate for LLM Reasoning via Structural Constraint Satisfaction 

**Authors**: Shinobu Miya  

**Link**: [PDF](https://arxiv.org/pdf/2512.20664)  

**Abstract**: Large Language Models (LLMs) frequently produce hallucinated statements that are assigned high likelihood by the model itself, exposing a fundamental limitation of probability-based verification. This suggests that hallucination is often not a low-confidence phenomenon, but a failure of structural consistency. In this work, we reformulate the verification of LLM reasoning as a Constraint Satisfaction Problem (CSP) operating independently of the generation likelihood. Rather than optimizing for statistical plausibility, we model verification as a feasibility check based on structural violation cost -- the computational cost required to embed a candidate reasoning step into the contextual graph structure. We define a total cost function composed of three proxies: (i) graph connectivity (structural), (ii) feature space consistency (geometric), and (iii) logical entailment (symbolic). Crucially, verification is performed via a lightweight System-2 gate, Eidoku, which rejects candidates exceeding a context-calibrated cost threshold. The threshold is not learned but is derived from the intrinsic statistics of the context, avoiding ad hoc heuristics. We demonstrate that this approach successfully rejects ``smooth falsehoods'' -- statements that are highly probable yet structurally disconnected -- that probability-based verifiers are principally incapable of detecting. Our experiments on a controlled diagnostic dataset show that explicitly enforcing structural constraints allows for the deterministic rejection of this specific class of hallucinations, serving as a neuro-symbolic sanity check for generative reasoning. 

---
# AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent 

**Authors**: Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, Yansong Tang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20745)  

**Abstract**: Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool this http URL evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced this http URL results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents. 

---
# Quantifying Laziness, Decoding Suboptimality, and Context Degradation in Large Language Models 

**Authors**: Yiqing Ma, Jung-Hua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20662)  

**Abstract**: Large Language Models (LLMs) often exhibit behavioral artifacts such as laziness (premature truncation of responses or partial compliance with multi-part requests), decoding suboptimality (failure to select higher-quality sequences due to myopic decoding), and context degradation (forgetting or ignoring core instructions over long conversations). We conducted three controlled experiments (A, B, and C) to quantify these phenomena across several advanced LLMs (OpenAI GPT-4 variant, DeepSeek). Our results indicate widespread laziness in satisfying complex multi-part instructions: models frequently omitted required sections or failed to meet length requirements despite explicit prompting. However, we found limited evidence of decoding suboptimality in a simple reasoning task (the models' greedy answers appeared to align with their highest-confidence solution), and we observed surprising robustness against context degradation in a 200-turn chaotic conversation test - the models maintained key facts and instructions far better than expected. These findings suggest that while compliance with detailed instructions remains an open challenge, modern LLMs may internally mitigate some hypothesized failure modes (such as context forgetting) in straightforward retrieval scenarios. We discuss implications for reliability, relate our findings to prior work on instruction-following and long-context processing, and recommend strategies (such as self-refinement and dynamic prompting) to reduce laziness and bolster multi-instruction compliance. 

---
# Reasoning Relay: Evaluating Stability and Interchangeability of Large Language Models in Mathematical Reasoning 

**Authors**: Leo Lu, Jonathan Zhang, Sean Chua, Spencer Kim, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2512.20647)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of large language models (LLMs). While prior work focuses on improving model performance through internal reasoning strategies, little is known about the interchangeability of reasoning across different models. In this work, we explore whether a partially completed reasoning chain from one model can be reliably continued by another model, either within the same model family or across families. We achieve this by assessing the sufficiency of intermediate reasoning traces as transferable scaffolds for logical coherence and final answer accuracy. We interpret this interchangeability as a means of examining inference-time trustworthiness, probing whether reasoning remains both coherent and reliable under model substitution. Using token-level log-probability thresholds to truncate reasoning at early, mid, and late stages from our baseline models, Gemma-3-4B-IT and LLaMA-3.1-70B-Instruct, we conduct continuation experiments with Gemma-3-1B-IT and LLaMA-3.1-8B-Instruct to test intra-family and cross-family behaviors. Our evaluation pipeline leverages truncation thresholds with a Process Reward Model (PRM), providing a reproducible framework for assessing reasoning stability via model interchange. Evaluations with a PRM reveal that hybrid reasoning chains often preserve, and in some cases even improve, final accuracy and logical structure. Our findings point towards interchangeability as an emerging behavioral property of reasoning models, offering insights into new paradigms for reliable modular reasoning in collaborative AI systems. 

---
# MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation 

**Authors**: Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20626)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to dynamically access external information, which is powerful for answering questions over previously unseen documents. Nonetheless, they struggle with high-level conceptual understanding and holistic comprehension due to limited context windows, which constrain their ability to perform deep reasoning over long-form, domain-specific content such as full-length books. To solve this problem, knowledge graphs (KGs) have been leveraged to provide entity-centric structure and hierarchical summaries, offering more structured support for reasoning. However, existing KG-based RAG solutions remain restricted to text-only inputs and fail to leverage the complementary insights provided by other modalities such as vision. On the other hand, reasoning from visual documents requires textual, visual, and spatial cues into structured, hierarchical concepts. To address this issue, we introduce a multimodal knowledge graph-based RAG that enables cross-modal reasoning for better content understanding. Our method incorporates visual cues into the construction of knowledge graphs, the retrieval phase, and the answer generation process. Experimental results across both global and fine-grained question answering tasks show that our approach consistently outperforms existing RAG-based approaches on both textual and multimodal corpora. 

---
# Memory Bear AI A Breakthrough from Memory to Cognition Toward Artificial General Intelligence 

**Authors**: Deliang Wen, Ke Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20651)  

**Abstract**: Large language models (LLMs) face inherent limitations in memory, including restricted context windows, long-term knowledge forgetting, redundant information accumulation, and hallucination generation. These issues severely constrain sustained dialogue and personalized services. This paper proposes the Memory Bear system, which constructs a human-like memory architecture grounded in cognitive science principles. By integrating multimodal information perception, dynamic memory maintenance, and adaptive cognitive services, Memory Bear achieves a full-chain reconstruction of LLM memory mechanisms. Across domains such as healthcare, enterprise operations, and education, Memory Bear demonstrates substantial engineering innovation and performance breakthroughs. It significantly improves knowledge fidelity and retrieval efficiency in long-term conversations, reduces hallucination rates, and enhances contextual adaptability and reasoning capability through memory-cognition integration. Experimental results show that, compared with existing solutions (e.g., Mem0, MemGPT, Graphiti), Memory Bear outperforms them across key metrics, including accuracy, token efficiency, and response latency. This marks a crucial step forward in advancing AI from "memory" to "cognition". 

---
# BitRL-Light: 1-bit LLM Agents with Deep Reinforcement Learning for Energy-Efficient Smart Home Lighting Optimization 

**Authors**: Ravi Gupta, Shabista Haider  

**Link**: [PDF](https://arxiv.org/pdf/2512.20623)  

**Abstract**: Smart home lighting systems consume 15-20% of residential energy but lack adaptive intelligence to optimize for user comfort and energy efficiency simultaneously. We present BitRL-Light, a novel framework combining 1-bit quantized Large Language Models (LLMs) with Deep Q-Network (DQN) reinforcement learning for real-time smart home lighting control on edge devices. Our approach deploys a 1-bit quantized Llama-3.2-1B model on Raspberry Pi hardware, achieving 71.4 times energy reduction compared to full-precision models while maintaining intelligent control capabilities. Through multi-objective reinforcement learning, BitRL-Light learns optimal lighting policies from user feedback, balancing energy consumption, comfort, and circadian alignment. Experimental results demonstrate 32% energy savings compared to rule-based systems, with inference latency under 200ms on Raspberry Pi 4 and 95% user satisfaction. The system processes natural language commands via Google Home/IFTTT integration and learns from implicit feedback through manual overrides. Our comparative analysis shows 1-bit models achieve 5.07 times speedup over 2-bit alternatives on ARM processors while maintaining 92% task accuracy. This work establishes a practical framework for deploying adaptive AI on resource-constrained IoT devices, enabling intelligent home automation without cloud dependencies. 

---
# Scaling Laws for Economic Productivity: Experimental Evidence in LLM-Assisted Consulting, Data Analyst, and Management Tasks 

**Authors**: Ali Merali  

**Link**: [PDF](https://arxiv.org/pdf/2512.21316)  

**Abstract**: This paper derives `Scaling Laws for Economic Impacts' -- empirical relationships between the training compute of Large Language Models (LLMs) and professional productivity. In a preregistered experiment, over 500 consultants, data analysts, and managers completed professional tasks using one of 13 LLMs. We find that each year of AI model progress reduced task time by 8%, with 56% of gains driven by increased compute and 44% by algorithmic progress. However, productivity gains were significantly larger for non-agentic analytical tasks compared to agentic workflows requiring tool use. These findings suggest continued model scaling could boost U.S. productivity by approximately 20% over the next decade. 

---
# LookPlanGraph: Embodied Instruction Following Method with VLM Graph Augmentation 

**Authors**: Anatoly O. Onishchenko, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2512.21243)  

**Abstract**: Methods that use Large Language Models (LLM) as planners for embodied instruction following tasks have become widespread. To successfully complete tasks, the LLM must be grounded in the environment in which the robot operates. One solution is to use a scene graph that contains all the necessary information. Modern methods rely on prebuilt scene graphs and assume that all task-relevant information is available at the start of planning. However, these approaches do not account for changes in the environment that may occur between the graph construction and the task execution. We propose LookPlanGraph - a method that leverages a scene graph composed of static assets and object priors. During plan execution, LookPlanGraph continuously updates the graph with relevant objects, either by verifying existing priors or discovering new entities. This is achieved by processing the agents egocentric camera view using a Vision Language Model. We conducted experiments with changed object positions VirtualHome and OmniGibson simulated environments, demonstrating that LookPlanGraph outperforms methods based on predefined static scene graphs. To demonstrate the practical applicability of our approach, we also conducted experiments in a real-world setting. Additionally, we introduce the GraSIF (Graph Scenes for Instruction Following) dataset with automated validation framework, comprising 514 tasks drawn from SayPlan Office, BEHAVIOR-1K, and VirtualHome RobotHow. Project page available at this https URL . 

---
# Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking 

**Authors**: Yifan Huang, Xiaojun Jia, Wenbo Guo, Yuqiang Sun, Yihao Huang, Chong Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21236)  

**Abstract**: Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications. 

---
# Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.21017)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the Chain-of-Thought (CoT) component has become significant for complex reasoning tasks. However, in conventional Supervised Fine-Tuning (SFT), the model could allocate disproportionately more attention to CoT sequences with excessive length. This reduces focus on the much shorter but essential Key portion-the final answer, whose correctness directly determines task success and evaluation quality. To address this limitation, we propose SFTKey, a two-stage training scheme. In the first stage, conventional SFT is applied to ensure proper output format, while in the second stage, only the Key portion is fine-tuned to improve accuracy. Extensive experiments across multiple benchmarks and model families demonstrate that SFTKey achieves an average accuracy improvement exceeding 5\% over conventional SFT, while preserving the ability to generate correct formats. Overall, this study advances LLM fine-tuning by explicitly balancing CoT learning with additional optimization on answer-relevant tokens. 

---
# Policy-Conditioned Policies for Multi-Agent Task Solving 

**Authors**: Yue Lin, Shuhui Zhu, Wenhao Li, Ang Li, Dan Qiao, Pascal Poupart, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21024)  

**Abstract**: In multi-agent tasks, the central challenge lies in the dynamic adaptation of strategies. However, directly conditioning on opponents' strategies is intractable in the prevalent deep reinforcement learning paradigm due to a fundamental ``representational bottleneck'': neural policies are opaque, high-dimensional parameter vectors that are incomprehensible to other agents. In this work, we propose a paradigm shift that bridges this gap by representing policies as human-interpretable source code and utilizing Large Language Models (LLMs) as approximate interpreters. This programmatic representation allows us to operationalize the game-theoretic concept of \textit{Program Equilibrium}. We reformulate the learning problem by utilizing LLMs to perform optimization directly in the space of programmatic policies. The LLM functions as a point-wise best-response operator that iteratively synthesizes and refines the ego agent's policy code to respond to the opponent's strategy. We formalize this process as \textit{Programmatic Iterated Best Response (PIBR)}, an algorithm where the policy code is optimized by textual gradients, using structured feedback derived from game utility and runtime unit tests. We demonstrate that this approach effectively solves several standard coordination matrix games and a cooperative Level-Based Foraging environment. 

---
# Semantic Refinement with LLMs for Graph Representations 

**Authors**: Safal Thapaliya, Zehong Wang, Jiazheng Li, Ziming Li, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21106)  

**Abstract**: Graph-structured data exhibit substantial heterogeneity in where their predictive signals originate: in some domains, node-level semantics dominate, while in others, structural patterns play a central role. This structure-semantics heterogeneity implies that no graph learning model with a fixed inductive bias can generalize optimally across diverse graph domains. However, most existing methods address this challenge from the model side by incrementally injecting new inductive biases, which remains fundamentally limited given the open-ended diversity of real-world graphs. In this work, we take a data-centric perspective and treat node semantics as a task-adaptive variable. We propose a Data-Adaptive Semantic Refinement framework DAS for graph representation learning, which couples a fixed graph neural network (GNN) and a large language model (LLM) in a closed feedback loop. The GNN provides implicit supervisory signals to guide the semantic refinement of LLM, and the refined semantics are fed back to update the same graph learner. We evaluate our approach on both text-rich and text-free graphs. Results show consistent improvements on structure-dominated graphs while remaining competitive on semantics-rich graphs, demonstrating the effectiveness of data-centric semantic adaptation under structure-semantics heterogeneity. 

---
# Semi-Supervised Learning for Large Language Models Safety and Content Moderation 

**Authors**: Eduard Stefan Dinuta, Iustin Sirbu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2512.21107)  

**Abstract**: Safety for Large Language Models (LLMs) has been an ongoing research focus since their emergence and is even more relevant nowadays with the increasing capacity of those models. Currently, there are several guardrails in place for all public LLMs and multiple proposed datasets for training safety classifiers. However, training these safety classifiers relies on large quantities of labeled data, which can be problematic to acquire, prone to labeling errors, or often include synthetic data. To address these issues, we suggest a different approach: utilizing semi-supervised learning techniques, which leverage both labeled and unlabeled data, to improve the performance on the safety task. We analyze the improvements that these techniques can offer for both prompts given to Large Language Models and the responses to those requests. Moreover, since augmentation is the central part of semi-supervised algorithms, we demonstrate the importance of using task-specific augmentations, which significantly increase the performance when compared to general-purpose augmentation techniques. 

---
# TGC-Net: A Structure-Aware and Semantically-Aligned Framework for Text-Guided Medical Image Segmentation 

**Authors**: Gaoren Lin, Huangxuan Zhao, Yuan Xiong, Lefei Zhang, Bo Du, Wentao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21135)  

**Abstract**: Text-guided medical segmentation enhances segmentation accuracy by utilizing clinical reports as auxiliary information. However, existing methods typically rely on unaligned image and text encoders, which necessitate complex interaction modules for multimodal fusion. While CLIP provides a pre-aligned multimodal feature space, its direct application to medical imaging is limited by three main issues: insufficient preservation of fine-grained anatomical structures, inadequate modeling of complex clinical descriptions, and domain-specific semantic misalignment. To tackle these challenges, we propose TGC-Net, a CLIP-based framework focusing on parameter-efficient, task-specific adaptations. Specifically, it incorporates a Semantic-Structural Synergy Encoder (SSE) that augments CLIP's ViT with a CNN branch for multi-scale structural refinement, a Domain-Augmented Text Encoder (DATE) that injects large-language-model-derived medical knowledge, and a Vision-Language Calibration Module (VLCM) that refines cross-modal correspondence in a unified feature space. Experiments on five datasets across chest X-ray and thoracic CT modalities demonstrate that TGC-Net achieves state-of-the-art performance with substantially fewer trainable parameters, including notable Dice gains on challenging benchmarks. 

---
# LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics 

**Authors**: Jiashuo Liu, Jiayun Wu, Chunjie Wu, Jingkai Liu, Zaiyuan Wang, Huan Zhou, Wenhao Huang, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2512.21010)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) and diverse specialized benchmarks necessitates a shift from fragmented, task-specific metrics to a holistic, competitive ranking system that effectively aggregates performance across multiple ability dimensions. Primarily using static scoring, current evaluation methods are fundamentally limited. They struggle to determine the proper mix ratio across diverse benchmarks, and critically, they fail to capture a model's dynamic competitive fitness or its vulnerability when confronted with sequential, high-stakes tasks. To address this, we introduce the novel Competitive Swiss-System Dynamics (CSD) framework. CSD simulates a multi-round, sequential contest where models are dynamically paired across a curated sequence of benchmarks based on their accumulated win-loss record. And Monte Carlo Simulation ($N=100,000$ iterations) is used to approximate the statistically robust Expected Win Score ($E[S_m]$), which eliminates the noise of random pairing and early-round luck. Furthermore, we implement a Failure Sensitivity Analysis by parameterizing the per-round elimination quantity ($T_k$), which allows us to profile models based on their risk appetite--distinguishing between robust generalists and aggressive specialists. We demonstrate that CSD provides a more nuanced and context-aware ranking than traditional aggregate scoring and static pairwise models, representing a vital step towards risk-informed, next-generation LLM evaluation. 

---
# Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation 

**Authors**: Wei-Rui Chen, Vignesh Kothapalli, Ata Fatahibaarzi, Hejian Sang, Shao Tang, Qingquan Song, Zhipeng Wang, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2512.21002)  

**Abstract**: Distilling the reasoning capabilities from a large language model (LLM) to a smaller student model often involves training on substantial amounts of reasoning data. However, distillation over lengthy sequences with prompt (P), chain-of-thought (CoT), and answer (A) segments makes the process computationally expensive. In this work, we investigate how the allocation of supervision across different segments (P, CoT, A) affects student performance. Our analysis shows that selective knowledge distillation over only the CoT tokens can be effective when the prompt and answer information is encompassed by it. Building on this insight, we establish a truncation protocol to quantify computation-quality tradeoffs as a function of sequence length. We observe that training on only the first $50\%$ of tokens of every training sequence can retain, on average, $\approx94\%$ of full-sequence performance on math benchmarks while reducing training time, memory usage, and FLOPs by about $50\%$ each. These findings suggest that reasoning distillation benefits from prioritizing early reasoning tokens and provides a simple lever for computation-quality tradeoffs. Codes are available at this https URL. 

---
# Neural Probe-Based Hallucination Detection for Large Language Models 

**Authors**: Shize Liang, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20949)  

**Abstract**: Large language models(LLMs) excel at text generation and knowledge question-answering tasks, but they are prone to generating hallucinated content, severely limiting their application in high-risk domains. Current hallucination detection methods based on uncertainty estimation and external knowledge retrieval suffer from the limitation that they still produce erroneous content at high confidence levels and rely heavily on retrieval efficiency and knowledge coverage. In contrast, probe methods that leverage the model's hidden-layer states offer real-time and lightweight advantages. However, traditional linear probes struggle to capture nonlinear structures in deep semantic this http URL overcome these limitations, we propose a neural network-based framework for token-level hallucination detection. By freezing language model parameters, we employ lightweight MLP probes to perform nonlinear modeling of high-level hidden states. A multi-objective joint loss function is designed to enhance detection stability and semantic disambiguity. Additionally, we establish a layer position-probe performance response model, using Bayesian optimization to automatically search for optimal probe insertion layers and achieve superior training this http URL results on LongFact, HealthBench, and TriviaQA demonstrate that MLP probes significantly outperform state-of-the-art methods in accuracy, recall, and detection capability under low false-positive conditions. 

---
# Automatic Replication of LLM Mistakes in Medical Conversations 

**Authors**: Oleksii Proniakin, Diego Fajardo, Ruslan Nazarenko, Razvan Marinescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20983)  

**Abstract**: Large language models (LLMs) are increasingly evaluated in clinical settings using multi-dimensional rubrics which quantify reasoning quality, safety, and patient-centeredness. Yet, replicating specific mistakes in other LLM models is not straightforward and often requires manual effort. We introduce MedMistake, an automatic pipeline that extracts mistakes LLMs make in patient-doctor conversations and converts them into a benchmark of single-shot QA pairs. Our pipeline (1) creates complex, conversational data between an LLM patient and LLM doctor, (2) runs an evaluation with a committee of 2 LLM judges across a variety of dimensions and (3) creates simplified single-shot QA scenarios from those mistakes. We release MedMistake-All, a dataset of 3,390 single-shot QA pairs where GPT-5 and Gemini 2.5 Pro are currently failing to answer correctly, as judged by two LLM judges. We used medical experts to validate a subset of 211/3390 questions (MedMistake-Bench), which we used to run a final evaluation of 12 frontier LLMs: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek-Chat, Gemini 2.5 Pro, Gemini 3 Pro, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Grok 4, Grok 4.1, Mistral Large. We found that GPT models, Claude and Grok obtained the best performance on MedMistake-Bench. We release both the doctor-validated benchmark (MedMistake-Bench), as well as the full dataset (MedMistake-All) at this https URL. 

---
# MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2512.20822)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to medicine, yet their adoption is limited by concerns over reliability and safety. Existing evaluations either test factual medical knowledge in isolation or assess patient-level reasoning without verifying correctness, leaving a critical gap. We introduce MediEval, a benchmark that links MIMIC-IV electronic health records (EHRs) to a unified knowledge base built from UMLS and other biomedical vocabularies. MediEval generates diverse factual and counterfactual medical statements within real patient contexts, enabling systematic evaluation across a 4-quadrant framework that jointly considers knowledge grounding and contextual consistency. Using this framework, we identify critical failure modes, including hallucinated support and truth inversion, that current proprietary, open-source, and domain-specific LLMs frequently exhibit. To address these risks, we propose Counterfactual Risk-Aware Fine-tuning (CoRFu), a DPO-based method with an asymmetric penalty targeting unsafe confusions. CoRFu improves by +16.4 macro-F1 points over the base model and eliminates truth inversion errors, demonstrating both higher accuracy and substantially greater safety. 

---
# X-GridAgent: An LLM-Powered Agentic AI System for Assisting Power Grid Analysis 

**Authors**: Yihan, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20789)  

**Abstract**: The growing complexity of power system operations has created an urgent need for intelligent, automated tools to support reliable and efficient grid management. Conventional analysis tools often require significant domain expertise and manual effort, which limits their accessibility and adaptability. To address these challenges, this paper presents X-GridAgent, a novel large language model (LLM)-powered agentic AI system designed to automate complex power system analysis through natural language queries. The system integrates domain-specific tools and specialized databases under a three-layer hierarchical architecture comprising planning, coordination, and action layers. This architecture offers high flexibility and adaptability to previously unseen tasks, while providing a modular and extensible framework that can be readily expanded to incorporate new tools, data sources, or analytical capabilities. To further enhance performance, we introduce two novel algorithms: (1) LLM-driven prompt refinement with human feedback, and (2) schema-adaptive hybrid retrieval-augmented generation (RAG) for accurate information retrieval from large-scale structured grid datasets. Experimental evaluations across a variety of user queries and power grid cases demonstrate the effectiveness and reliability of X-GridAgent in automating interpretable and rigorous power system analysis. 

---
# FEM-Bench: A Structured Scientific Reasoning Benchmark for Evaluating Code-Generating LLMs 

**Authors**: Saeed Mohammadzadeh, Erfan Hamdi, Joel Shor, Emma Lejeune  

**Link**: [PDF](https://arxiv.org/pdf/2512.20732)  

**Abstract**: As LLMs advance their reasoning capabilities about the physical world, the absence of rigorous benchmarks for evaluating their ability to generate scientifically valid physical models has become a critical gap. Computational mechanics, which develops and applies mathematical models and numerical methods to predict the behavior of physical systems under forces, deformation, and constraints, provides an ideal foundation for structured scientific reasoning evaluation. Problems follow clear mathematical structure, enforce strict physical and numerical constraints, and support objective verification. The discipline requires constructing explicit models of physical systems and reasoning about geometry, spatial relationships, and material behavior, connecting directly to emerging AI goals in physical reasoning and world modeling. We introduce FEM-Bench, a computational mechanics benchmark designed to evaluate the ability of LLMs to generate correct finite element method (FEM) and related code. FEM-Bench 2025 contains a suite of introductory but nontrivial tasks aligned with material from a first graduate course on computational mechanics. These tasks capture essential numerical and physical modeling challenges while representing only a small fraction of the complexity present in the discipline. Despite their simplicity, state-of-the-art LLMs do not reliably solve all of them. In a five attempt run, the best performing model at function writing, Gemini 3 Pro, completed 30/33 tasks at least once and 26/33 tasks all five times. The best performing model at unit test writing, GPT-5, had an Average Joint Success Rate of 73.8%. Other popular models showed broad performance variation. FEM-Bench establishes a structured foundation for evaluating AI-generated scientific code, and future iterations will incorporate increasingly sophisticated tasks to track progress as models evolve. 

---
# Uncovering Competency Gaps in Large Language Models and Their Benchmarks 

**Authors**: Matyas Bohacek, Nino Scherrer, Nicholas Dufour, Thomas Leung, Christoph Bregler, Stephanie C. Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20638)  

**Abstract**: The evaluation of large language models (LLMs) relies heavily on standardized benchmarks. These benchmarks provide useful aggregated metrics for a given capability, but those aggregated metrics can obscure (i) particular sub-areas where the LLMs are weak ("model gaps") and (ii) imbalanced coverage in the benchmarks themselves ("benchmark gaps"). We propose a new method that uses sparse autoencoders (SAEs) to automatically uncover both types of gaps. By extracting SAE concept activations and computing saliency-weighted performance scores across benchmark data, the method grounds evaluation in the model's internal representations and enables comparison across benchmarks. As examples demonstrating our approach, we applied the method to two popular open-source models and ten benchmarks. We found that these models consistently underperformed on concepts that stand in contrast to sycophantic behaviors (e.g., politely refusing a request or asserting boundaries) and concepts connected to safety discussions. These model gaps align with observations previously surfaced in the literature; our automated, unsupervised method was able to recover them without manual supervision. We also observed benchmark gaps: many of the evaluated benchmarks over-represented concepts related to obedience, authority, or instruction-following, while missing core concepts that should fall within their intended scope. In sum, our method offers a representation-grounded approach to evaluation, enabling concept-level decomposition of benchmark scores. Rather than replacing conventional aggregated metrics, CG complements them by providing a concept-level decomposition that can reveal why a model scored as it did and how benchmarks could evolve to better reflect their intended scope. Code is available at this https URL. 

---
# Data-Free Pruning of Self-Attention Layers in LLMs 

**Authors**: Dhananjay Saikumar, Blesson Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2512.20636)  

**Abstract**: Many self-attention sublayers in large language models (LLMs) can be removed with little to no loss. We attribute this to the Attention Suppression Hypothesis: during pre-training, some deep attention layers learn to mute their own contribution, leaving the residual stream and the MLP to carry the representation. We propose Gate-Norm, a one-shot, weight-only criterion that ranks attention sublayers by query--key coupling and removes the least coupled ones, requiring no calibration data, no forward passes, no fine-tuning, and no specialized kernels. On 40-layer, 13B-parameter LLaMA models, Gate-Norm prunes the model in under a second. Pruning $8$--$16$ attention sublayers yields up to $1.30\times$ higher inference throughput while keeping average zero-shot accuracy within $2\%$ of the unpruned baseline across BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy/Challenge, and OpenBookQA. Across these settings, Gate-Norm matches data-driven pruning methods in accuracy while being $\sim 1000\times$ faster to score layers, enabling practical, data-free compression of LLMs. 

---
# Efficient Asynchronous Federated Evaluation with Strategy Similarity Awareness for Intent-Based Networking in Industrial Internet of Things 

**Authors**: Shaowen Qin, Jianfeng Zeng, Haodong Guo, Xiaohuan Li, Jiawen Kang, Qian Chen, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2512.20627)  

**Abstract**: Intent-Based Networking (IBN) offers a promising paradigm for intelligent and automated network control in Industrial Internet of Things (IIoT) environments by translating high-level user intents into executable network strategies. However, frequent strategy deployment and rollback are impractical in real-world IIoT systems due to tightly coupled workflows and high downtime costs, while the heterogeneity and privacy constraints of IIoT nodes further complicate centralized policy verification. To address these challenges, we propose FEIBN, a Federated Evaluation Enhanced Intent-Based Networking framework. FEIBN leverages large language models (LLMs) to align multimodal user intents into structured strategy tuples and employs federated learning to perform distributed policy verification across IIoT nodes without exposing raw data. To improve training efficiency and reduce communication overhead, we design SSAFL, a Strategy Similarity Aware Federated Learning mechanism that selects task-relevant nodes based on strategy similarity and resource status, and triggers asynchronous model uploads only when updates are significant. Experiments demonstrate that SSAFL can improve model accuracy, accelerate model convergence, and reduce the cost by 27.8% compared with SemiAsyn. 

---
# Enhancing Lung Cancer Treatment Outcome Prediction through Semantic Feature Engineering Using Large Language Models 

**Authors**: MunHwan Lee, Shaika Chowdhury, Xiaodi Li, Sivaraman Rajaganapathy, Eric W Klee, Ping Yang, Terence Sio, Liewei Wang, James Cerhan, Nansu NA Zong  

**Link**: [PDF](https://arxiv.org/pdf/2512.20633)  

**Abstract**: Accurate prediction of treatment outcomes in lung cancer remains challenging due to the sparsity, heterogeneity, and contextual overload of real-world electronic health data. Traditional models often fail to capture semantic information across multimodal streams, while large-scale fine-tuning approaches are impractical in clinical workflows. We introduce a framework that uses Large Language Models (LLMs) as Goal-oriented Knowledge Curators (GKC) to convert laboratory, genomic, and medication data into high-fidelity, task-aligned features. Unlike generic embeddings, GKC produces representations tailored to the prediction objective and operates as an offline preprocessing step that integrates naturally into hospital informatics pipelines. Using a lung cancer cohort (N=184), we benchmarked GKC against expert-engineered features, direct text embeddings, and an end-to-end transformer. Our approach achieved a mean AUROC of 0.803 (95% CI: 0.799-0.807) and outperformed all baselines. An ablation study further confirmed the complementary value of combining all three modalities. These results show that the quality of semantic representation is a key determinant of predictive accuracy in sparse clinical data settings. By reframing LLMs as knowledge curation engines rather than black-box predictors, this work demonstrates a scalable, interpretable, and workflow-compatible pathway for advancing AI-driven decision support in oncology. 

---
# ClarifyMT-Bench: Benchmarking and Improving Multi-Turn Clarification for Conversational Large Language Models 

**Authors**: Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21120)  

**Abstract**: Large language models (LLMs) are increasingly deployed as conversational assistants in open-domain, multi-turn settings, where users often provide incomplete or ambiguous information. However, existing LLM-focused clarification benchmarks primarily assume single-turn interactions or cooperative users, limiting their ability to evaluate clarification behavior in realistic settings. We introduce \textbf{ClarifyMT-Bench}, a benchmark for multi-turn clarification grounded in a five-dimensional ambiguity taxonomy and a set of six behaviorally diverse simulated user personas. Through a hybrid LLM-human pipeline, we construct 6,120 multi-turn dialogues capturing diverse ambiguity sources and interaction patterns. Evaluating ten representative LLMs uncovers a consistent under-clarification bias: LLMs tend to answer prematurely, and performance degrades as dialogue depth increases. To mitigate this, we propose \textbf{ClarifyAgent}, an agentic approach that decomposes clarification into perception, forecasting, tracking, and planning, substantially improving robustness across ambiguity conditions. ClarifyMT-Bench establishes a reproducible foundation for studying when LLMs should ask, when they should answer, and how to navigate ambiguity in real-world human-LLM interactions. 

---
# How important is Recall for Measuring Retrieval Quality? 

**Authors**: Shelly Schwartz, Oleg Vasilyev, Randy Sawaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.20854)  

**Abstract**: In realistic retrieval settings with large and evolving knowledge bases, the total number of documents relevant to a query is typically unknown, and recall cannot be computed. In this paper, we evaluate several established strategies for handling this limitation by measuring the correlation between retrieval quality metrics and LLM-based judgments of response quality, where responses are generated from the retrieved documents. We conduct experiments across multiple datasets with a relatively low number of relevant documents (2-15). We also introduce a simple retrieval quality measure that performs well without requiring knowledge of the total number of relevant documents. 

---
# Where Did This Sentence Come From? Tracing Provenance in LLM Reasoning Distillation 

**Authors**: Kaiyuan Liu, Shaotian Yan, Rui Miao, Bing Wang, Chen Shen, Jun Zhang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.20908)  

**Abstract**: Reasoning distillation has attracted increasing attention. It typically leverages a large teacher model to generate reasoning paths, which are then used to fine-tune a student model so that it mimics the teacher's behavior in training contexts. However, previous approaches have lacked a detailed analysis of the origins of the distilled model's capabilities. It remains unclear whether the student can maintain consistent behaviors with the teacher in novel test-time contexts, or whether it regresses to its original output patterns, raising concerns about the generalization of distillation models. To analyse this question, we introduce a cross-model Reasoning Distillation Provenance Tracing framework. For each action (e.g., a sentence) produced by the distilled model, we obtain the predictive probabilities assigned by the teacher, the original student, and the distilled model under the same context. By comparing these probabilities, we classify each action into different categories. By systematically disentangling the provenance of each action, we experimentally demonstrate that, in test-time contexts, the distilled model can indeed generate teacher-originated actions, which correlate with and plausibly explain observed performance on distilled model. Building on this analysis, we further propose a teacher-guided data selection method. Unlike prior approach that rely on heuristics, our method directly compares teacher-student divergences on the training data, providing a principled selection criterion. We validate the effectiveness of our approach across multiple representative teacher models and diverse student models. The results highlight the utility of our provenance-tracing framework and underscore its promise for reasoning distillation. We hope to share Reasoning Distillation Provenance Tracing and our insights into reasoning distillation with the community. 

---
# Large Language Models Approach Expert Pedagogical Quality in Math Tutoring but Differ in Instructional and Linguistic Profiles 

**Authors**: Ramatu Oiza Abdulsalam, Segun Aroyehun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20780)  

**Abstract**: Recent work has explored the use of large language models for generating tutoring responses in mathematics, yet it remains unclear how closely their instructional behavior aligns with expert human practice. We examine this question using a controlled, turn-level comparison in which expert human tutors, novice human tutors, and multiple large language models respond to the same set of math remediation conversation turns. We examine both instructional strategies and linguistic characteristics of tutoring responses, including restating and revoicing, pressing for accuracy, lexical diversity, readability, politeness, and agency. We find that large language models approach expert levels of perceived pedagogical quality on average but exhibit systematic differences in their instructional and linguistic profiles. In particular, large language models tend to underuse restating and revoicing strategies characteristic of expert human tutors, while producing longer, more lexically diverse, and more polite responses. Statistical analyses show that restating and revoicing, lexical diversity, and pressing for accuracy are positively associated with perceived pedagogical quality, whereas higher levels of agentic and polite language are negatively associated. Overall, recent large language models exhibit levels of perceived pedagogical quality comparable to expert human tutors, while relying on different instructional and linguistic strategies. These findings underscore the value of analyzing instructional strategies and linguistic characteristics when evaluating tutoring responses across human tutors and intelligent tutoring systems. 

---
# ReaSeq: Unleashing World Knowledge via Reasoning for Sequential Modeling 

**Authors**: Chuan Wang, Gaoming Yang, Han Wu, Jiakai Tang, Jiahao Yu, Jian Wu, Jianwu Hu, Junjun Zheng, Shuwen Xiao, Yeqiu Yang, Yuning Jiang, Ahjol Nurlanbek, Binbin Cao, Bo Zheng, Fangmei Zhu, Gaoming Zhou, Huimin Yi, Huiping Chu, Jin Huang, Jinzhe Shan, Kenan Cui, Longbin Li, Silu Zhou, Wen Chen, Xia Ming, Xiang Gao, Xin Yao, Xingyu Wen, Yan Zhang, Yiwen Hu, Yulin Wang, Ziheng Bao, Zongyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21257)  

**Abstract**: Industrial recommender systems face two fundamental limitations under the log-driven paradigm: (1) knowledge poverty in ID-based item representations that causes brittle interest modeling under data sparsity, and (2) systemic blindness to beyond-log user interests that constrains model performance within platform boundaries. These limitations stem from an over-reliance on shallow interaction statistics and close-looped feedback while neglecting the rich world knowledge about product semantics and cross-domain behavioral patterns that Large Language Models have learned from vast corpora.
To address these challenges, we introduce ReaSeq, a reasoning-enhanced framework that leverages world knowledge in Large Language Models to address both limitations through explicit and implicit reasoning. Specifically, ReaSeq employs explicit Chain-of-Thought reasoning via multi-agent collaboration to distill structured product knowledge into semantically enriched item representations, and latent reasoning via Diffusion Large Language Models to infer plausible beyond-log behaviors. Deployed on Taobao's ranking system serving hundreds of millions of users, ReaSeq achieves substantial gains: >6.0% in IPV and CTR, >2.9% in Orders, and >2.5% in GMV, validating the effectiveness of world-knowledge-enhanced reasoning over purely log-driven approaches. 

---
# Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System 

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.20677)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems. 

---
# Semantic Deception: When Reasoning Models Can't Compute an Addition 

**Authors**: Nathaniël de Leeuw, Marceau Nahon, Mathis Reymond, Raja Chatila, Mehdi Khamassi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20812)  

**Abstract**: Large language models (LLMs) are increasingly used in situations where human values are at stake, such as decision-making tasks that involve reasoning when performed by humans. We investigate the so-called reasoning capabilities of LLMs over novel symbolic representations by introducing an experimental framework that tests their ability to process and manipulate unfamiliar symbols. We introduce semantic deceptions: situations in which symbols carry misleading semantic associations due to their form, such as being embedded in specific contexts, designed to probe whether LLMs can maintain symbolic abstraction or whether they default to exploiting learned semantic associations. We redefine standard digits and mathematical operators using novel symbols, and task LLMs with solving simple calculations expressed in this altered notation. The objective is: (1) to assess LLMs' capacity for abstraction and manipulation of arbitrary symbol systems; (2) to evaluate their ability to resist misleading semantic cues that conflict with the task's symbolic logic. Through experiments with four LLMs we show that semantic cues can significantly deteriorate reasoning models' performance on very simple tasks. They reveal limitations in current LLMs' ability for symbolic manipulations and highlight a tendency to over-rely on surface-level semantics, suggesting that chain-of-thoughts may amplify reliance on statistical correlations. Even in situations where LLMs seem to correctly follow instructions, semantic cues still impact basic capabilities. These limitations raise ethical and societal concerns, undermining the widespread and pernicious tendency to attribute reasoning abilities to LLMs and suggesting how LLMs might fail, in particular in decision-making contexts where robust symbolic reasoning is essential and should not be compromised by residual semantic associations inherited from the model's training. 

---
# MMSRARec: Summarization and Retrieval Augumented Sequential Recommendation Based on Multimodal Large Language Model 

**Authors**: Haoyu Wang, Yitong Wang, Jining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20916)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant potential in recommendation systems. However, the effective application of MLLMs to multimodal sequential recommendation remains unexplored: A) Existing methods primarily leverage the multimodal semantic understanding capabilities of pre-trained MLLMs to generate item embeddings or semantic IDs, thereby enhancing traditional recommendation models. These approaches generate item representations that exhibit limited interpretability, and pose challenges when transferring to language model-based recommendation systems. B) Other approaches convert user behavior sequence into image-text pairs and perform recommendation through multiple MLLM inference, incurring prohibitive computational and time costs. C) Current MLLM-based recommendation systems generally neglect the integration of collaborative signals. To address these limitations while balancing recommendation performance, interpretability, and computational cost, this paper proposes MultiModal Summarization-and-Retrieval-Augmented Sequential Recommendation. Specifically, we first employ MLLM to summarize items into concise keywords and fine-tune the model using rewards that incorporate summary length, information loss, and reconstruction difficulty, thereby enabling adaptive adjustment of the summarization policy. Inspired by retrieval-augmented generation, we then transform collaborative signals into corresponding keywords and integrate them as supplementary context. Finally, we apply supervised fine-tuning with multi-task learning to align the MLLM with the multimodal sequential recommendation. Extensive evaluations on common recommendation datasets demonstrate the effectiveness of MMSRARec, showcasing its capability to efficiently and interpretably understand user behavior histories and item information for accurate recommendations. 

---
# Soft Filtering: Guiding Zero-shot Composed Image Retrieval with Prescriptive and Proscriptive Constraints 

**Authors**: Youjin Jung, Seongwoo Cho, Hyun-seok Min, Sungchul Choi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20781)  

**Abstract**: Composed Image Retrieval (CIR) aims to find a target image that aligns with user intent, expressed through a reference image and a modification text. While Zero-shot CIR (ZS-CIR) methods sidestep the need for labeled training data by leveraging pretrained vision-language models, they often rely on a single fused query that merges all descriptive cues of what the user wants, tending to dilute key information and failing to account for what they wish to avoid. Moreover, current CIR benchmarks assume a single correct target per query, overlooking the ambiguity in modification texts. To address these challenges, we propose Soft Filtering with Textual constraints (SoFT), a training-free, plug-and-play filtering module for ZS-CIR. SoFT leverages multimodal large language models (LLMs) to extract two complementary constraints from the reference-modification pair: prescriptive (must-have) and proscriptive (must-avoid) constraints. These serve as semantic filters that reward or penalize candidate images to re-rank results, without modifying the base retrieval model or adding supervision. In addition, we construct a two-stage dataset pipeline that refines CIR benchmarks. We first identify multiple plausible targets per query to construct multi-target triplets, capturing the open-ended nature of user intent. Then guide multimodal LLMs to rewrite the modification text to focus on one target, while referencing contrastive distractors to ensure precision. This enables more comprehensive and reliable evaluation under varying ambiguity levels. Applied on top of CIReVL, a ZS-CIR retriever, SoFT raises R@5 to 65.25 on CIRR (+12.94), mAP@50 to 27.93 on CIRCO (+6.13), and R@50 to 58.44 on FashionIQ (+4.59), demonstrating broad effectiveness. 

---
# Agentic Multi-Persona Framework for Evidence-Aware Fake News Detection 

**Authors**: Roopa Bukke, Soumya Pandey, Suraj Kumar, Soumi Chattopadhyay, Chandranath Adak  

**Link**: [PDF](https://arxiv.org/pdf/2512.21039)  

**Abstract**: The rapid proliferation of online misinformation poses significant risks to public trust, policy, and safety, necessitating reliable automated fake news detection. Existing methods often struggle with multimodal content, domain generalization, and explainability. We propose AMPEND-LS, an agentic multi-persona evidence-grounded framework with LLM-SLM synergy for multimodal fake news detection. AMPEND-LS integrates textual, visual, and contextual signals through a structured reasoning pipeline powered by LLMs, augmented with reverse image search, knowledge graph paths, and persuasion strategy analysis. To improve reliability, we introduce a credibility fusion mechanism combining semantic similarity, domain trustworthiness, and temporal context, and a complementary SLM classifier to mitigate LLM uncertainty and hallucinations. Extensive experiments across three benchmark datasets demonstrate that AMPEND-LS consistently outperformed state-of-the-art baselines in accuracy, F1 score, and robustness. Qualitative case studies further highlight its transparent reasoning and resilience against evolving misinformation. This work advances the development of adaptive, explainable, and evidence-aware systems for safeguarding online information integrity. 

---
