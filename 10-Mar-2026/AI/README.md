# Agentic Critical Training 

**Authors**: Weize Liu, Minghui Liu, Sy-Tuyen Ho, Souradip Chakraborty, Xiyao Wang, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08706)  

**Abstract**: Training large language models (LLMs) as autonomous agents often begins with imitation learning, but it only teaches agents what to do without understanding why: agents never contrast successful actions against suboptimal alternatives and thus lack awareness of action quality. Recent approaches attempt to address this by introducing self-reflection supervision derived from contrasts between expert and alternative actions. However, the training paradigm fundamentally remains imitation learning: the model imitates pre-constructed reflection text rather than learning to reason autonomously. We propose Agentic Critical Training (ACT), a reinforcement learning paradigm that trains agents to identify the better action among alternatives. By rewarding whether the model's judgment is correct, ACT drives the model to autonomously develop reasoning about action quality, producing genuine self-reflection rather than imitating it. Across three challenging agent benchmarks, ACT consistently improves agent performance when combined with different post-training methods. It achieves an average improvement of 5.07 points over imitation learning and 4.62 points over reinforcement learning. Compared to approaches that inject reflection capability through knowledge distillation, ACT also demonstrates clear advantages, yielding an average improvement of 2.42 points. Moreover, ACT enables strong out-of-distribution generalization on agentic benchmarks and improves performance on general reasoning benchmarks without any reasoning-specific training data, highlighting the value of our method. These results suggest that ACT is a promising path toward developing more reflective and capable LLM agents. 

---
# Evaluating Financial Intelligence in Large Language Models: Benchmarking SuperInvesting AI with LLM Engines 

**Authors**: Akshay Gulati, Kanha Singhania, Tushar Banga, Parth Arora, Anshul Verma, Vaibhav Kumar Singh, Agyapal Digra, Jayant Singh Bisht, Danish Sharma, Varun Singla, Shubh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2603.08704)  

**Abstract**: Large language models are increasingly used for financial analysis and investment research, yet systematic evaluation of their financial reasoning capabilities remains limited. In this work, we introduce the AI Financial Intelligence Benchmark (AFIB), a multi-dimensional evaluation framework designed to assess financial analysis capabilities across five dimensions: factual accuracy, analytical completeness, data recency, model consistency, and failure patterns. We evaluate five AI systems: GPT, Gemini, Perplexity, Claude, and SuperInvesting, using a dataset of 95+ structured financial analysis questions derived from real-world equity research tasks. The results reveal substantial differences in performance across models. Within this benchmark setting, SuperInvesting achieves the highest aggregate performance, with an average factual accuracy score of 8.96/10 and the highest completeness score of 56.65/70, while also demonstrating the lowest hallucination rate among evaluated systems. Retrieval-oriented systems such as Perplexity perform strongly on data recency tasks due to live information access but exhibit weaker analytical synthesis and consistency. Overall, the results highlight that financial intelligence in large language models is inherently multi-dimensional, and systems that combine structured financial data access with analytical reasoning capabilities provide the most reliable performance for complex investment research workflows. 

---
# A Multi-Objective Optimization Approach for Sustainable AI-Driven Entrepreneurship in Resilient Economies 

**Authors**: Anas ALsobeh, Raneem Alkurdi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08692)  

**Abstract**: The rapid advancement of artificial intelligence (AI) technologies presents both unprecedented opportunities and significant challenges for sustainable economic development. While AI offers transformative potential for addressing environmental challenges and enhancing economic resilience, its deployment often involves substantial energy consumption and environmental costs. This research introduces the EcoAI-Resilience framework, a multi-objective optimization approach designed to maximize the sustainability benefits of AI deployment while minimizing environmental costs and enhancing economic resilience. The framework addresses three critical objectives through mathematical optimization: sustainability impact maximization, economic resilience enhancement, and environmental cost minimization. The methodology integrates diverse data sources, including energy consumption metrics, sustainability indicators, economic performance data, and entrepreneurship outcomes across 53 countries and 14 sectors from 2015-2024. Our experimental validation demonstrates exceptional performance with R scores exceeding 0.99 across all model components, significantly outperforming baseline methods, including Linear Regression (R = 0.943), Random Forest (R = 0.957), and Gradient Boosting (R = 0.989). The framework successfully identifies optimal AI deployment strategies featuring 100\% renewable energy integration, 80% efficiency improvement targets, and optimal investment levels of $202.48 per capita. Key findings reveal strong correlations between economic complexity and resilience (r = 0.82), renewable energy adoption and sustainability outcomes (r = 0.71), and demonstrate significant temporal improvements in AI readiness (+1.12 points/year) and renewable energy adoption (+0.67 year) globally. 

---
# OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning 

**Authors**: Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08655)  

**Abstract**: We introduce OfficeQA Pro, a benchmark for evaluating AI agents on grounded, multi-document reasoning over a large and heterogeneous document corpus. The corpus consists of U.S. Treasury Bulletins spanning nearly 100 years, comprising 89,000 pages and over 26 million numerical values. OfficeQA Pro consists of 133 questions that require precise document parsing, retrieval, and analytical reasoning across both unstructured text and tabular data. Frontier LLMs including Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro Preview achieve less than 5% accuracy on OfficeQA Pro when relying on parametric knowledge, and less than 12% with additional access to the web. When provided directly with the document corpus, frontier agents still struggle on over half of questions, scoring 34.1% on average. We find that providing agents with a structured document representation produced by Databricks' ai_parse_document yields a 16.1% average relative performance gain across agents. We conduct additional ablations to study the effects of model selection, table representation, retrieval strategy, and test-time scaling on performance. Despite these improvements, significant headroom remains before agents can be considered reliable at enterprise-grade grounded reasoning. 

---
# CoCo: Code as CoT for Text-to-Image Preview and Rare Concept Generation 

**Authors**: Haodong Li, Chunmei Qing, Huanyu Zhang, Dongzhi Jiang, Yihang Zou, Hongbo Peng, Dingming Li, Yuhong Dai, ZePeng Lin, Juanxi Tian, Yi Zhou, Siqi Dai, Jingwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08652)  

**Abstract**: Recent advancements in Unified Multimodal Models (UMMs) have significantly advanced text-to-image (T2I) generation, particularly through the integration of Chain-of-Thought (CoT) reasoning. However, existing CoT-based T2I methods largely rely on abstract natural-language planning, which lacks the precision required for complex spatial layouts, structured visual elements, and dense textual content. In this work, we propose CoCo (Code-as-CoT), a code-driven reasoning framework that represents the reasoning process as executable code, enabling explicit and verifiable intermediate planning for image generation. Given a text prompt, CoCo first generates executable code that specifies the structural layout of the scene, which is then executed in a sandboxed environment to render a deterministic draft image. The model subsequently refines this draft through fine-grained image editing to produce the final high-fidelity result. To support this training paradigm, we construct CoCo-10K, a curated dataset containing structured draft-final image pairs designed to teach both structured draft construction and corrective visual refinement. Empirical evaluations on StructT2IBench, OneIG-Bench, and LongText-Bench show that CoCo achieves improvements of +68.83%, +54.8%, and +41.23% over direct generation, while also outperforming other generation methods empowered by CoT. These results demonstrate that executable code is an effective and reliable reasoning paradigm for precise, controllable, and structured text-to-image generation. The code is available at: this https URL 

---
# Trust via Reputation of Conviction 

**Authors**: Aravind R. Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2603.08575)  

**Abstract**: The question of \emph{knowledge}, \emph{truth} and \emph{trust} is explored via a mathematical formulation of claims and sources. We define truth as the reproducibly perceived subset of knowledge, formalize sources as having both generative and discriminative roles, and develop a framework for reputation grounded in the \emph{conviction} -- the likelihood that a source's stance is vindicated by independent consensus. We argue that conviction, rather than correctness or faithfulness, is the principled basis for trust: it is regime-independent, rewards genuine contribution, and demands the transparent and self-sufficient perceptions that make external verification possible. We formalize reputation as the expected weighted signed conviction over a realm of claims, characterize its behavior across source-claim regimes, and identify continuous verification as both a theoretical necessity and a practical mechanism through which reputation accrues. The framework is applied to AI agents, which are identified as capable but error-prone sources for whom verifiable conviction and continuously accrued reputation constitute the only robust foundation for trust. 

---
# RetroAgent: From Solving to Evolving via Retrospective Dual Intrinsic Feedback 

**Authors**: Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2603.08561)  

**Abstract**: Large language model (LLM)-based agents trained with reinforcement learning (RL) have shown strong potential on complex interactive tasks. However, standard RL paradigms favor static problem-solving over continuous adaptation: agents often converge to suboptimal strategies due to insufficient exploration, while learned knowledge remains implicit within parameters rather than explicitly retrievable, limiting effective experiential learning. To address these limitations, we introduce RetroAgent, an online RL framework that empowers agents to master complex interactive environments not just by solving, but by evolving. Concretely, RetroAgent features a hindsight self-reflection mechanism that produces dual intrinsic feedback: (1) intrinsic numerical feedback that that tracks incremental subtask completion relative to prior attempts, rewarding promising explorations, and (2) intrinsic language feedback that distills reusable lessons into a memory buffer, retrieved via our proposed Similarity & Utility-Aware Upper Confidence Bound (SimUtil-UCB) strategy balancing relevance, utility, and exploration to effectively leverage past experiences. Extensive experiments on two model families across four challenging agentic tasks demonstrate that RetroAgent significantly outperforms existing methods, achieving state-of-the-art results -- e.g., surpassing Group Relative Policy Optimization (GRPO)-trained agents by +18.3% on ALFWorld, +15.4% on WebShop, +27.1% on Sokoban, and +8.9% on MineSweeper -- while exhibiting strong test-time adaptation and generalization to out-of-distribution scenarios. 

---
# The Boiling Frog Threshold: Criticality and Blindness in World Model-Based Anomaly Detection Under Gradual Drift 

**Authors**: Zhe Hong  

**Link**: [PDF](https://arxiv.org/pdf/2603.08455)  

**Abstract**: When an RL agent's observations are gradually corrupted, at what drift rate does it "wake up" -- and what determines this boundary? We study world model-based self-monitoring under continuous observation drift across four MuJoCo environments, three detector families (z-score, variance, percentile), and three model capacities. We find that (1) a sharp detection threshold $\varepsilon^*$ exists universally: below it, drift is absorbed as normal variation; above it, detection occurs rapidly. The threshold's existence and sigmoid shape are invariant across all detector families and model capacities, though its position depends on the interaction between detector sensitivity, noise floor structure, and environment dynamics. (2) Sinusoidal drift is completely undetectable by all detector families -- including variance and percentile detectors with no temporal smoothing -- establishing this as a world model property rather than a detector artifact. (3) Within each environment, $\varepsilon^*$ follows a power law in detector parameters ($R^2 = 0.89$-$0.97$), but cross-environment prediction fails ($R^2 = 0.45$), revealing that the missing variable is environment-specific dynamics structure $\partial \mathrm{PE}/\partial\varepsilon$. (4) In fragile environments, agents collapse before any detector can fire ("collapse before awareness"), creating a fundamentally unmonitorable failure mode. Our results reframe $\varepsilon^*$ from an emergent world model property to a three-way interaction between noise floor, detector, and environment dynamics, providing a more defensible and empirically grounded account of self-monitoring boundaries in RL agents. 

---
# Efficient Policy Learning with Hybrid Evaluation-Based Genetic Programming for Uncertain Agile Earth Observation Satellite Scheduling 

**Authors**: Junhua Xue, Yuning Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08447)  

**Abstract**: The Uncertain Agile Earth Observation Satellite Scheduling Problem (UAEOSSP) is a novel combinatorial optimization problem and a practical engineering challenge that aligns with the current demands of space technology development. It incorporates uncertainties in profit, resource consumption, and visibility, which may render pre-planned schedules suboptimal or even infeasible. Genetic Programming Hyper-Heuristic (GPHH) shows promise for evolving interpretable scheduling policies; however, their simulation-based evaluation incurs high computational costs. Moreover, the design of the constructive method, denoted as Online Scheduling Algorithm (OSA), directly affects fitness assessment, resulting in evaluation-dependent local optima within the policy space. To address these issues, this paper proposes a Hybrid Evaluation-based Genetic Programming (HE-GP) for effectively solving UAEOSSP. A Hybrid Evaluation (HE) mechanism is integrated into the policy-driven OSA, combining exact and approximate filtering modes: exact mode ensures evaluation accuracy through elaborately designed constraint verification modules, while approximate mode reduces computational overhead via simplified logic. HE-GP dynamically switches between evaluation models based on real-time evolutionary state information. Experiments on 16 simulated instance sets demonstrate that HE-GP significantly outperforms handcrafted heuristics and single-evaluation based GPHH, achieving substantial reductions in computational cost while maintaining excellent scheduling performance across diverse scenarios. Specifically, the average training time of HE-GP was reduced by 17.77\% compared to GP employing exclusively exact evaluation, while the optimal policy generated by HE-GP achieved the highest average ranks across all scenarios. 

---
# IronEngine: Towards General AI Assistant 

**Authors**: Xi Mo  

**Link**: [PDF](https://arxiv.org/pdf/2603.08425)  

**Abstract**: This paper presents IronEngine, a general AI assistant platform organized around a unified orchestration core that connects a desktop user interface, REST and WebSocket APIs, Python clients, local and cloud model backends, persistent memory, task scheduling, reusable skills, 24-category tool execution, MCP-compatible extensibility, and hardware-facing integration. IronEngine introduces a three-phase pipeline -- Discussion (Planner--Reviewer collaboration), Model Switch (VRAM-aware transition), and Execution (tool-augmented action loop) -- that separates planning quality from execution capability. The system features a hierarchical memory architecture with multi-level consolidation, a vectorized skill repository backed by ChromaDB, an adaptive model management layer supporting 92 model profiles with VRAM-aware context budgeting, and an intelligent tool routing system with 130+ alias normalization and automatic error correction. We present experimental results on file operation benchmarks achieving 100\% task completion with a mean total time of 1541 seconds across four heterogeneous tasks, and provide detailed comparisons with representative AI assistant systems including ChatGPT, Claude Desktop, Cursor, Windsurf, and open-source agent frameworks. Without disclosing proprietary prompts or core algorithms, this paper analyzes the platform's architectural decomposition, subsystem design, experimental performance, safety boundaries, and comparative engineering advantages. The resulting study positions IronEngine as a system-oriented foundation for general-purpose personal assistants, automation frameworks, and future human-centered agent platforms. 

---
# A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation 

**Authors**: Cong Cao, Jingyao Zhang, Kun Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.08388)  

**Abstract**: We propose a Hierarchical Error-Corrective Graph FrameworkforAutonomousAgentswithLLM-BasedActionGeneration(HECG),whichincorporates three core innovations: (1) Multi-Dimensional Transferable Strategy (MDTS): by integrating task quality metrics (Q), confidence/cost metrics (C), reward metrics (R), and LLM-based semantic reasoning scores (LLM-Score), MDTS achieves multi-dimensional alignment between quantitative performance and semantic context, enabling more precise selection of high-quality candidate strate gies and effectively reducing the risk of negative transfer. (2) Error Matrix Classification (EMC): unlike simple confusion matrices or overall performance metrics, EMC provides structured attribution of task failures by categorizing errors into ten types, such as Strategy Errors (Strategy Whe) and Script Parsing Errors (Script-Parsing-Error), and decomposing them according to severity, typical actions, error descriptions, and recoverability. This allows precise analysis of the root causes of task failures, offering clear guidance for subsequent error correction and strategy optimization rather than relying solely on overall success rates or single performance metrics. (3) Causal-Context Graph Retrieval (CCGR): to enhance agent retrieval capabilities in dynamic task environments, we construct graphs from historical states, actions, and event sequences, where nodes store executed actions, next-step actions, execution states, transferable strategies, and other relevant information, and edges represent causal dependencies such as preconditions for transitions between nodes. CCGR identifies subgraphs most relevant to the current task context, effectively capturing structural relationships beyond vector similarity, allowing agents to fully leverage contextual information, accelerate strategy adaptation, and improve execution reliability in complex, multi-step tasks. 

---
# M$^3$-ACE: Rectifying Visual Perception in Multimodal Math Reasoning via Multi-Agentic Context Engineering 

**Authors**: Peijin Xie, Zhen Xu, Bingquan Liu, Baoxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08369)  

**Abstract**: Multimodal large language models have recently shown promising progress in visual mathematical reasoning. However, their performance is often limited by a critical yet underexplored bottleneck: inaccurate visual perception. Through systematic analysis, we find that the most failures originate from incorrect or incomplete visual evidence extraction rather than deficiencies in reasoning capability. Moreover, models tend to remain overly confident in their initial perceptions, making standard strategies such as prompt engineering, multi-round self-reflection, or posterior guidance insufficient to reliably correct errors.
To address this limitation, we propose M3-ACE, a multi-agentic context engineering framework designed to rectify visual perception in multimodal math reasoning. Instead of directly aggregating final answers, our approach decouples perception and reasoning by dynamically maintaining a shared context centered on visual evidence lists. Multiple agents collaboratively contribute complementary observations, enabling the system to expose inconsistencies and recover missing perceptual information. To support stable multi-turn collaboration, we further introduce two lightweight tools: a Summary Tool that organizes evidence from different agents into consistent, complementary, and conflicting components, and a Refine Tool that filters unreliable samples and guides iterative correction.
Extensive experiments demonstrate that M3-ACE substantially improves visual mathematical reasoning performance across multiple benchmarks. Our method establishes new state-of-the-art results 89.1 on the MathVision benchmark and achieves consistent improvements on other related datasets, including MathVista and MathVerse. These results highlight the importance of perception-centric multi-agent collaboration for advancing multimodal reasoning systems. 

---
# Agentic Neurosymbolic Collaboration for Mathematical Discovery: A Case Study in Combinatorial Design 

**Authors**: Hai Xia, Carla P. Gomes, Bart Selman, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2603.08322)  

**Abstract**: We study mathematical discovery through the lens of neurosymbolic reasoning, where an AI agent powered by a large language model (LLM), coupled with symbolic computation tools, and human strategic direction, jointly produced a new result in combinatorial design theory. The main result of this human-AI collaboration is a tight lower bound on the imbalance of Latin squares for the notoriously difficult case $n \equiv 1 \pmod{3}$.
We reconstruct the discovery process from detailed interaction logs spanning multiple sessions over several days and identify the distinct cognitive contributions of each component. The AI agent proved effective at uncovering hidden structure and generating hypotheses. The symbolic component consists of computer algebra, constraint solvers, and simulated annealing, which provides rigorous verification and exhaustive enumeration. Human steering supplied the critical research pivot that transformed a dead end into a productive inquiry. Our analysis reveals that multi-model deliberation among frontier LLMs proved reliable for criticism and error detection but unreliable for constructive claims.
The resulting human-AI mathematical contribution, a tight lower bound of $4n(n{-}1)/9$, is achieved via a novel class of near-perfect permutations. The bound was formally verified in Lean 4. Our experiments show that neurosymbolic systems can indeed produce genuine discoveries in pure mathematics. 

---
# CORE-Acu: Structured Reasoning Traces and Knowledge Graph Safety Verification for Acupuncture Clinical Decision Support 

**Authors**: Liuyi Xu, Yun Guo, Ming Chen, Zihan Dun, Yining Qian, An-Yang Lu, Shuang Li, Lijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08321)  

**Abstract**: Large language models (LLMs) show significant potential for clinical decision support (CDS), yet their black-box nature -- characterized by untraceable reasoning and probabilistic hallucinations -- poses severe challenges in acupuncture, a field demanding rigorous interpretability and safety. To address this, we propose CORE-Acu, a neuro-symbolic framework for acupuncture clinical decision support that integrates Structured Chain-of-Thought (S-CoT) with knowledge graph (KG) safety verification. First, we construct the first acupuncture Structured Reasoning Trace dataset and a schema-constrained fine-tuning framework. By enforcing an explicit causal chain from pattern identification to treatment principles, treatment plans, and acupoint selection, we transform implicit Traditional Chinese Medicine (TCM) reasoning into interpretable generation constraints, mitigating the opacity of LLM-based CDS. Furthermore, we construct a TCM safety knowledge graph and establish a ``Generate--Verify--Revise'' closed-loop inference system based on a Symbolic Veto Mechanism, employing deterministic rules to intercept hallucinations and enforce hard safety boundaries. Finally, we introduce the Lexicon-Matched Entity-Reweighted Loss (LMERL), which corrects terminology drift caused by the frequency--importance mismatch in general optimization by adaptively amplifying gradient contributions of high-risk entities during fine-tuning. Experiments on 1,000 held-out cases demonstrate CORE-Acu's superior entity fidelity and reasoning quality. Crucially, CORE-Acu achieved 0/1,000 observed safety violations (95\% CI: 0--0.37\%), whereas GPT-4o exhibited an 8.5\% violation rate under identical rules. These results establish CORE-Acu as a robust neuro-symbolic framework for acupuncture clinical decision support, guaranteeing both reasoning auditability and strict safety compliance. 

---
# Deconstructing Multimodal Mathematical Reasoning: Towards a Unified Perception-Alignment-Reasoning Paradigm 

**Authors**: Tianyu Yang, Sihong Wu, Yilun Zhao, Zhenwen Liang, Lisen Dai, Chen Zhao, Minhao Cheng, Arman Cohan, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08291)  

**Abstract**: Multimodal Mathematical Reasoning (MMR) has recently attracted increasing attention for its capability to solve mathematical problems that involve both textual and visual modalities. However, current models still face significant challenges in real-world visual math tasks. They often misinterpret diagrams, fail to align mathematical symbols with visual evidence, and produce inconsistent reasoning steps. Moreover, existing evaluations mainly focus on checking final answers rather than verifying the correctness or executability of each intermediate step. To address these limitations, a growing body of recent research addresses these issues by integrating structured perception, explicit alignment, and verifiable reasoning within unified frameworks. To establish a clear roadmap for understanding and comparing different MMR approaches, we systematically study them around four fundamental questions: (1) What to extract from multimodal inputs, (2) How to represent and align textual and visual information, (3) How to perform the reasoning, and (4) How to evaluate the correctness of the overall reasoning process. Finally, we discuss open challenges and offer perspectives on promising directions for future research. 

---
# Towards a more efficient bias detection in financial language models 

**Authors**: Firas Hadj Kacem, Ahmed Khanfir, Mike Papadakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.08267)  

**Abstract**: Bias in financial language models constitutes a major obstacle to their adoption in real-world applications. Detecting such bias is challenging, as it requires identifying inputs whose predictions change when varying properties unrelated to the decision, such as demographic attributes. Existing approaches typically rely on exhaustive mutation and pairwise prediction analysis over large corpora, which is effective but computationally expensive-particularly for large language models and can become impractical in continuous retraining and releasing processes. Aiming at reducing this cost, we conduct a large-scale study of bias in five financial language models, examining similarities in their bias tendencies across protected attributes and exploring cross-model-guided bias detection to identify bias-revealing inputs earlier. Our study uses approximately 17k real financial news sentences, mutated to construct over 125k original-mutant pairs. Results show that all models exhibit bias under both atomic (0.58\%-6.05\%) and intersectional (0.75\%-5.97\%) settings. Moreover, we observe consistent patterns in bias-revealing inputs across models, enabling substantial reuse and cost reduction in bias detection. For example, up to 73\% of FinMA's biased behaviours can be uncovered using only 20\% of the input pairs when guided by properties derived from DistilRoBERTa outputs. 

---
# FinToolBench: Evaluating LLM Agents for Real-World Financial Tool Use 

**Authors**: Jiaxuan Lu, Kong Wang, Yemin Wang, Qingmei Tang, Hongwei Zeng, Xiang Chen, Jiahao Pi, Shujian Deng, Lingzhi Chen, Yi Fu, Kehua Yang, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.08262)  

**Abstract**: The integration of Large Language Models (LLMs) into the financial domain is driving a paradigm shift from passive information retrieval to dynamic, agentic interaction. While general-purpose tool learning has witnessed a surge in benchmarks, the financial sector, characterized by high stakes, strict compliance, and rapid data volatility, remains critically underserved. Existing financial evaluations predominantly focus on static textual analysis or document-based QA, ignoring the complex reality of tool execution. Conversely, general tool benchmarks lack the domain-specific rigor required for finance, often relying on toy environments or a negligible number of financial APIs. To bridge this gap, we introduce FinToolBench, the first real-world, runnable benchmark dedicated to evaluating financial tool learning agents. Unlike prior works limited to a handful of mock tools, FinToolBench establishes a realistic ecosystem coupling 760 executable financial tools with 295 rigorous, tool-required queries. We propose a novel evaluation framework that goes beyond binary execution success, assessing agents on finance-critical dimensions: timeliness, intent type, and regulatory domain alignment. Furthermore, we present FATR, a finance-aware tool retrieval and reasoning baseline that enhances stability and compliance. By providing the first testbed for auditable, agentic financial execution, FinToolBench sets a new standard for trustworthy AI in finance. The tool manifest, execution environment, and evaluation code will be open-sourced to facilitate future research. 

---
# The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs 

**Authors**: Yonghong Deng, Zhen Yang, Ping Jian, Xinyue Zhang, Zhongbin Guo, Chengzhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.08234)  

**Abstract**: With the rapid advancement of large language models (LLMs), the safety of LLMs has become a critical concern. Despite significant efforts in safety alignment, current LLMs remain vulnerable to jailbreaking attacks. However, the root causes of such vulnerabilities are still poorly understood, necessitating a rigorous investigation into jailbreak mechanisms across both academic and industrial communities. In this work, we focus on a continuation-triggered jailbreak phenomenon, whereby simply relocating a continuation-triggered instruction suffix can substantially increase jailbreak success rates. To uncover the intrinsic mechanisms of this phenomenon, we conduct a comprehensive mechanistic interpretability analysis at the level of attention heads. Through causal interventions and activation scaling, we show that this jailbreak behavior primarily arises from an inherent competition between the model's intrinsic continuation drive and the safety defenses acquired through alignment training. Furthermore, we perform a detailed behavioral analysis of the identified safety-critical attention heads, revealing notable differences in the functions and behaviors of safety heads across different model architectures. These findings provide a novel mechanistic perspective for understanding and interpreting jailbreak behaviors in LLMs, offering both theoretical insights and practical implications for improving model safety. 

---
# Evidence-Driven Reasoning for Industrial Maintenance Using Heterogeneous Data 

**Authors**: Fearghal O'Donncha, Nianjun Zhou, Natalia Martinez, James T Rayfield, Fenno F. Heath III, Abigail Langbridge, Roman Vaculin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08171)  

**Abstract**: Industrial maintenance platforms contain rich but fragmented evidence, including free-text work orders, heterogeneous operational sensors or indicators, and structured failure knowledge. These sources are often analyzed in isolation, producing alerts or forecasts that do not support conditional decision-making: given this asset history and behavior, what is happening and what action is warranted? We present Condition Insight Agent, a deployed decision-support framework that integrates maintenance language, behavioral abstractions of operational data, and engineering failure semantics to produce evidence-grounded explanations and advisory actions. The system constrains reasoning through deterministic evidence construction and structured failure knowledge, and applies a rule-based verification loop to suppress unsupported conclusions. Case studies from production CMMS deployments show that this verification-first design operates reliably under heterogeneous and incomplete data while preserving human oversight. Our results demonstrate how constrained LLM-based reasoning can function as a governed decision-support layer for industrial maintenance. 

---
# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking 

**Authors**: Chang Liu, Chuqiao Kuang, Tianyi Zhuang, Yuxin Cheng, Huichi Zhou, Xiaoguang Li, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08117)  

**Abstract**: Recent advancements in LLM-based information-seeking agents have achieved record-breaking performance on established benchmarks. However, these agents remain heavily reliant on search-engine-indexed knowledge, leaving a critical blind spot: Unindexed Information Seeking (UIS). This paper identifies and explores the UIS problem, where vital information is not captured by search engine crawlers, such as overlooked content, dynamic webpages, and embedded files. Despite its significance, UIS remains an underexplored challenge. To address this gap, we introduce UIS-QA, the first dedicated UIS benchmark, comprising 110 expert-annotated QA pairs. Notably, even state-of-the-art agents experience a drastic performance drop on UIS-QA (e.g., from 70.90 on GAIA and 46.70 on BrowseComp-zh to 24.55 on UIS-QA), underscoring the severity of the problem. To mitigate this, we propose UIS-Digger, a novel multi-agent framework that incorporates dual-mode browsing and enables simultaneous webpage searching and file parsing. With a relatively small $\sim$30B-parameter backbone LLM optimized using SFT and RFT training strategies, UIS-Digger sets a strong baseline at 27.27\%, outperforming systems integrating sophisticated LLMs such as O3 and GPT-4.1. This demonstrates the importance of proactive interaction with unindexed sources for effective and comprehensive information-seeking. Our work not only uncovers a fundamental limitation in current agent evaluation paradigms but also provides the first toolkit for advancing UIS research, defining a new and promising direction for robust information-seeking systems. 

---
# In-Context Reinforcement Learning for Tool Use in Large Language Models 

**Authors**: Yaoqi Ye, Yiran Zhao, Keyu Duan, Zeyu Zheng, Kenji Kawaguchi, Cihang Xie, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2603.08068)  

**Abstract**: While large language models (LLMs) exhibit strong reasoning abilities, their performance on complex tasks is often constrained by the limitations of their internal knowledge. A compelling approach to overcome this challenge is to augment these models with external tools -- such as Python interpreters for mathematical computations or search engines for retrieving factual information. However, enabling models to use these tools effectively remains a significant challenge. Existing methods typically rely on cold-start pipelines that begin with supervised fine-tuning (SFT), followed by reinforcement learning (RL). These approaches often require substantial amounts of labeled data for SFT, which is expensive to annotate or synthesize. In this work, we propose In-Context Reinforcement Learning (ICRL), an RL-only framework that eliminates the need for SFT by leveraging few-shot prompting during the rollout stage of RL. Specifically, ICRL introduces in-context examples within the rollout prompts to teach the model how to invoke external tools. Furthermore, as training progresses, the number of in-context examples is gradually reduced, eventually reaching a zero-shot setting where the model learns to call tools independently. We conduct extensive experiments across a range of reasoning and tool-use benchmarks. Results show that ICRL achieves state-of-the-art performance, demonstrating its effectiveness as a scalable, data-efficient alternative to traditional SFT-based pipelines. 

---
# S2S-FDD: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis 

**Authors**: Baoxue Li, Chunhui Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.08048)  

**Abstract**: Fault diagnosis is critical for the safe operation of industrial systems. Conventional diagnosis models typically produce abstract outputs such as anomaly scores or fault categories, failing to answer critical operational questions like "Why" or "How to repair". While large language models (LLMs) offer strong generalization and reasoning abilities, their training on discrete textual corpora creates a semantic gap when processing high-dimensional, temporal industrial signals. To address this challenge, we propose a Signals-to-Semantics fault diagnosis (S2S-FDD) framework that bridges high-dimensional sensor signals with natural language semantics through two key innovations: We first design a Signal-to-Semantic operator to convert abstract time-series signals into natural language summaries, capturing trends, periodicity, and deviations. Based on the descriptions, we design a multi-turn tree-structured diagnosis method to perform fault diagnosis by referencing historical maintenance documents and dynamically querying additional signals. The framework further supports human-in-the-loop feedback for continuous refinement. Experiments on the multiphase flow process show the feasibility and effectiveness of the proposed method for explainable zero-shot fault diagnosis. 

---
# CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling 

**Authors**: Dengcan Liu, Fengkai Yang, Xiaohan Wang, Shurui Yan, Jiajun Chai, Jiahao Li, Yikun Ban, Zhendong Mao, Wei Lin, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08035)  

**Abstract**: Reward modeling is essential for aligning Large Language Models(LLMs) with human preferences, yet conventional reward models suffer from poor interpretability and heavy reliance on costly expert annotations. While recent rubric-based approaches enhance evaluation transparency, they lack systematic quality control, yielding noisy and redundant criteria, failing to mitigate persistent biases (e.g., verbosity, position) in LLM evaluators, and creating a scalability-reliability trade-off. To address these limitations, we propose CDRRM (Contrast-Driven Rubric Reward Model), a framework built on a novel Contrast-then-Synthesis paradigm for high-quality rubric generation and guided preference judgment. CDRRM first conducts multi-dimensional contrastive profiling on preference pairs to identify causal discriminative factors, then synthesizes these insights into compact, context-aware rubrics to guide preference judg- ments. Extensive experiments on three authoritative benchmarks (RewardBench, RMBench, RMB) demonstrate that CDRRM achieves state-of-the-art performance across diverse domains and effectively mitigates aforementioned evaluation biases. Notably, our approach delivers exceptional data efficiency: training the rubric generator on only 3k high-quality samples empowers a frozen pre-trained judge model to outperform fully fine-tuned baselines. This work offers a scalable, interpretable, and data-efficient path for reward modeling. 

---
# PIRA-Bench: A Transition from Reactive GUI Agents to GUI-based Proactive Intent Recommendation Agents 

**Authors**: Yuxiang Chai, Shunye Tang, Han Xiao, Rui Liu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.08013)  

**Abstract**: Current Graphical User Interface (GUI) agents operate primarily under a reactive paradigm: a user must provide an explicit instruction for the agent to execute a task. However, an intelligent AI assistant should be proactive, which is capable of anticipating user intentions directly from continuous visual inputs, such as mobile or desktop screenshots, and offering timely recommendations without explicit user prompting. Transitioning to this proactive paradigm presents significant challenges. Real-world screen activity is rarely linear; it consists of long-horizon trajectories fraught with noisy browsing, meaningless actions, and multithreaded task-switching. To address this gap, we introduce PIRA-Bench (Proactive Intent Recommendation Agent Benchmark), a novel benchmark for evaluating multimodal large language models (MLLMs) on continuous, weakly-supervised visual inputs. Unlike reactive datasets, PIRA-Bench features complex trajectories with multiple interleaved intents and noisy segments with various user profile contexts, challenging agents to detect actionable events while fitting to user preferences. Furthermore, we propose the PIRF baseline, a memory-aware, state-tracking framework that empowers general MLLMs to manage multiple task threads and handle misleading visual inputs. PIRA-Bench serves as an initial step toward robust and proactive GUI-based personal assistants. 

---
# CMMR-VLN: Vision-and-Language Navigation via Continual Multimodal Memory Retrieval 

**Authors**: Haozhou Li, Xiangyu Dong, Huiyan Jiang, Yaoming Zhou, Xiaoguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.07997)  

**Abstract**: Although large language models (LLMs) are introduced into vision-and-language navigation (VLN) to improve instruction comprehension and generalization, existing LLM- based VLN lacks the ability to selectively recall and use relevant priori experiences to help navigation tasks, limiting their performance in long-horizon and unfamiliar scenarios. In this work, we propose CMMR-VLN (Continual Multimodal Memory Retrieval based VLN), a VLN framework that endows LLM agents with structured memory and reflection capabilities. Specifically, the CMMR-VLN constructs a multimodal experi- ence memory indexed by panoramic visual images and salient landmarks to retrieve relevant experiences during navigation, introduces a retrieved-augmented generation pipeline to mimick how experienced human navigators leverage priori knowledge, and incorporates a reflection-based memory update strategy that selectively stores complete successful paths and the key initial mistake in failure cases. Comprehensive tests illustrate average success rate improvements of 52.9%, 20.9% and 20.9%, and 200%, 50% and 50% over the NavGPT, the MapGPT, and the DiscussNav in simulation and real tests, respectively eluci- dating the great potential of the CMMR-VLN as a backbone VLN framework. 

---
# OSExpert: Computer-Use Agents Learning Professional Skills via Exploration 

**Authors**: Jiateng Liu, Zhenhailong Wang, Rushi Wang, Bingxuan Li, Jeonghwan Kim, Aditi Tiwari, Pengfei Yu, Denghui Zhang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.07978)  

**Abstract**: General-purpose computer-use agents have shown impressive performance across diverse digital environments. However, our new benchmark, OSExpert-Eval, indicates they remain far less helpful than human experts. Although inference-time scaling enables adaptation, these agents complete complex tasks inefficiently with degraded performance, transfer poorly to unseen UIs, and struggle with fine-grained action sequences. To solve the problem, we introduce a GUI-based depth-first search (GUI-DFS) exploration algorithm to comprehensively explore and verify an environment's unit functions. The agent then exploits compositionality between unit skills to self-construct a curriculum for composite tasks. To support fine-grained actions, we curate a database of action primitives for agents to discover during exploration; these are saved as a skill set once the exploration is complete. We use the learned skills to improve the agent's performance and efficiency by (1) enriching agents with ready-to-use procedural knowledge, allowing them to plan only once for long trajectories and generate accurate actions, and (2) enabling them to end inference-time scaling earlier by realizing their boundary of capabilities. Extensive experiments show that our environment-learned agent takes a meaningful step toward expert-level computer use, achieving a around 20 percent performance gain on OSExpert-Eval and closing the efficiency gap to humans by around 80 percent 

---
# Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning 

**Authors**: Wei Yang, Defu Cao, Jiacheng Pang, Muyan Weng, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07972)  

**Abstract**: While scaling individual Large Language Models (LLMs) has delivered remarkable progress, the next frontier lies in scaling collaboration through multi-agent systems (MAS). However, purely autonomous MAS remain ''closed-world'' systems, constrained by the static knowledge horizon of pre-trained models. This limitation makes them brittle on tasks requiring knowledge beyond training data, often leading to collective failure under novel challenges. To address this, we propose the Human-In-the-Loop Multi-Agent Collaboration (HILA) framework, a principled paradigm for human--agent collaboration. HILA trains agents to learn a metacognitive policy that governs when to solve problems autonomously and when to defer to a human expert. To operationalize this policy, we introduce Dual-Loop Policy Optimization, which disentangles immediate decision-making from long-term capability growth. The inner loop applies Group Relative Policy Optimization (GRPO) with a cost-aware reward to optimize deferral decisions, while the outer loop implements continual learning, transforming expert feedback into high-quality supervised signals that strengthen the agent's reasoning ability. Experiments on challenging mathematical and problem-solving benchmarks show that HILA, equipped with Dual-Loop Policy Optimization, consistently outperforms advanced MAS, establishing a principled foundation for collaborative and continually improving agentic systems. 

---
# Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs 

**Authors**: Chen Lu, Ke Xue, Chengrui Gao, Yunqi Shi, Siyuan Xu, Mingxuan Yuan, Chao Qian, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.07970)  

**Abstract**: With the rapid advancement of human science and technology, problems in industrial scenarios are becoming increasingly challenging, bringing significant challenges to traditional algorithm design. Automated algorithm design with LLMs emerges as a promising solution, but the currently adopted black-box modeling deprives LLMs of any awareness of the intrinsic mechanism of the target problem, leading to hallucinated designs. In this paper, we introduce Evolutionary Stagewise Algorithm Design (EvoStage), a novel evolutionary paradigm that bridges the gap between the rigorous demands of industrial-scale algorithm design and the LLM-based algorithm design methods. Drawing inspiration from CoT, EvoStage decomposes the algorithm design process into sequential, manageable stages and integrates real-time intermediate feedback to iteratively refine algorithm design directions. To further reduce the algorithm design space and avoid falling into local optima, we introduce a multi-agent system and a "global-local perspective" mechanism. We apply EvoStage to the design of two types of common optimizers: designing parameter configuration schedules of the Adam optimizer for chip placement, and designing acquisition functions of Bayesian optimization for black-box optimization. Experimental results across open-source benchmarks demonstrate that EvoStage outperforms human-expert designs and existing LLM-based methods within only a couple of evolution steps, even achieving the historically state-of-the-art half-perimeter wire-length results on every tested chip case. Furthermore, when deployed on a commercial-grade 3D chip placement tool, EvoStage significantly surpasses the original performance metrics, achieving record-breaking efficiency. We hope EvoStage can significantly advance automated algorithm design in the real world, helping elevate human productivity. 

---
# Rel-MOSS: Towards Imbalanced Relational Deep Learning on Relational Databases 

**Authors**: Jun Yin, Peng Huo, Bangguo Zhu, Hao Yan, Senzhang Wang, Shirui Pan, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07916)  

**Abstract**: In recent advances, to enable a fully data-driven learning paradigm on relational databases (RDB), relational deep learning (RDL) is proposed to structure the RDB as a heterogeneous entity graph and adopt the graph neural network (GNN) as the predictive model. However, existing RDL methods neglect the imbalance problem of relational data in RDBs and risk under-representing the minority entities, leading to an unusable model in practice. In this work, we investigate, for the first time, class imbalance problem in RDB entity classification and design the relation-centric minority synthetic over-sampling GNN (Rel-MOSS), in order to fill a critical void in the current literature. Specifically, to mitigate the issue of minority-related information being submerged by majority counterparts, we design the relation-wise gating controller to modulate neighborhood messages from each individual relation type. Based on the relational-gated representations, we further propose the relation-guided minority synthesizer for over-sampling, which integrates the entity relational signatures to maintain relational consistency. Extensive experiments on 12 entity classification datasets provide compelling evidence for the superiority of Rel-MOSS, yielding an average improvement of up to 2.46% and 4.00% in terms of Balanced Accuracy and G-Mean, compared with SOTA RDL methods and classic methods for handling class imbalance. 

---
# Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents 

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07915)  

**Abstract**: Modern agents powered by thinking LLMs achieve high accuracy through long chain-of-thought reasoning but incur substantial inference costs. While many LLMs now support configurable reasoning levels (e.g., high/medium/low), static strategies are often ineffective: using low-effort modes at every step leads to significant performance degradation, while random selection fails to preserve accuracy or provide meaningful cost reduction. However, agents should reserve high reasoning effort for difficult steps like navigating complex website structures, while using lower-effort modes for simpler steps like opening a target URL. In this paper, we propose Ares, a framework for per-step dynamic reasoning effort selection tailored for multi-step agent tasks. Ares employs a lightweight router to predict the lowest appropriate reasoning level for each step based on the interaction history. To train this router, we develop a data generation pipeline that identifies the minimum reasoning effort required for successful step completion. We then fine-tune the router to predict these levels, enabling plug-and-play integration for any LLM agents. We evaluate Ares on a diverse set of agent tasks, including TAU-Bench for tool use agents, BrowseComp-Plus for deep-research agents, and WebArena for web agents. Experimental results show that Ares reduces reasoning token usage by up to 52.7% compared to fixed high-effort reasoning, while introducing minimal degradation in task success rates. 

---
# EveryQuery: Zero-Shot Clinical Prediction via Task-Conditioned Pretraining over Electronic Health Records 

**Authors**: Payal Chandak, Gregory Kondas, Isaac Kohane, Matthew McDermott  

**Link**: [PDF](https://arxiv.org/pdf/2603.07900)  

**Abstract**: Foundation models pretrained on electronic health records (EHR) have demonstrated zero-shot clinical prediction capabilities by generating synthetic patient futures and aggregating statistics over sampled trajectories. However, this autoregressive inference procedure is computationally expensive, statistically noisy, and not natively promptable because users cannot directly condition predictions on specific clinical questions. In this preliminary work, we introduce EveryQuery, an EHR foundation model that achieves zero-shot inference through task-conditioned pre-training. Rather than generating future events, EveryQuery takes as input a patient's history and a structured query specifying a clinical task, and directly estimates the likelihood of the outcome occurring in the future window via a single forward pass. EveryQuery realizes this capability by pre-training over randomly sampled combinations of query tasks and patient contexts, directly training the model to produce correct answers to arbitrary input prompts. This enables zero-shot prediction for any task in the query space without finetuning, linear probing, or trajectory generation. On MIMIC-IV, EveryQuery outperforms an autoregressive baseline on 82% of 39 randomly sampled prediction tasks, with a mean AUC improvement of +0.16 (95% CI: [0.10,0.22]). This advantage remains consistent on tasks that were explicitly held out from the pre-training distribution. Further, EveryQuery's performance gains are most pronounced for rare clinical events, affirming and demonstrating a solution to the fundamental limitation of autoregressive inference for low-prevalence outcomes. However, at present, EveryQuery underperforms on tasks requiring disjunctive reasoning over multiple codes, such as 30-day readmission, exposing a concrete expressiveness limitation of the current query language. 

---
# SMGI: A Structural Theory of General Artificial Intelligence 

**Authors**: Aomar Osmani  

**Link**: [PDF](https://arxiv.org/pdf/2603.07896)  

**Abstract**: We introduce SMGI, a structural theory of general artificial intelligence, and recast the foundational problem of learning from the optimization of hypotheses within fixed environments to the controlled evolution of the learning interface itself. We formalize the Structural Model of General Intelligence (SMGI) via a typed meta-model $\theta = (r,\mathcal H,\Pi,\mathcal L,\mathcal E,\mathcal M)$ that treats representational maps, hypothesis spaces, structural priors, multi-regime evaluators, and memory operators as explicitly typed, dynamic components. By enforcing a strict mathematical separation between this structural ontology ($\theta$) and its induced behavioral semantics ($T_\theta$), we define general artificial intelligence as a class of admissible coupled dynamics $(\theta, T_\theta)$ satisfying four obligations: structural closure under typed transformations, dynamical stability under certified evolution, bounded statistical capacity, and evaluative invariance across regime shifts. We prove a structural generalization bound that links sequential PAC-Bayes analysis and Lyapunov stability, providing sufficient conditions for capacity control and bounded drift under admissible task transformations. Furthermore, we establish a strict structural inclusion theorem demonstrating that classical empirical risk minimization, reinforcement learning, program-prior models (Solomonoff-style), and modern frontier agentic pipelines operate as structurally restricted instances of SMGI. 

---
# A Lightweight Traffic Map for Efficient Anytime LaCAM* 

**Authors**: Bojie Shen, Yue Zhang, Zhe Chen, Daniel Harabor  

**Link**: [PDF](https://arxiv.org/pdf/2603.07891)  

**Abstract**: Multi-Agent Path Finding (MAPF) aims to compute collision-free paths for multiple agents and has a wide range of practical applications. LaCAM*, an anytime configuration-based solver, currently represents the state of the art. Recent work has explored the use of guidance paths to steer LaCAM* toward configurations that avoid traffic congestion, thereby improving solution quality. However, existing approaches rely on Frank-Wolfe-style optimization that repeatedly invokes single-agent search before executing LaCAM*, resulting in substantial computational overhead for large-scale problems. Moreover, the guidance path is static and primarily beneficial for finding the first solution in LaCAM*. To address these limitations, we propose a new approach that leverages LaCAM*'s ability to construct a dynamic, lightweight traffic map during its search. Experimental results demonstrate that our method achieves higher solution quality than state-of-the-art guidance-path approaches across two MAPF variants. 

---
# Visualizing Coalition Formation: From Hedonic Games to Image Segmentation 

**Authors**: Pedro Henrique de Paula França, Lucas Lopes Felipe, Daniel Sadoc Menasché  

**Link**: [PDF](https://arxiv.org/pdf/2603.07890)  

**Abstract**: We propose image segmentation as a visual diagnostic testbed for coalition formation in hedonic games. Modeling pixels as agents on a graph, we study how a granularization parameter shapes equilibrium fragmentation and boundary structure. On the Weizmann single-object benchmark, we relate multi-coalition equilibria to binary protocols by measuring whether the converged coalitions overlap with a foreground ground-truth. We observe transitions from cohesive to fragmented yet recoverable equilibria, and finally to intrinsic failure under excessive fragmentation. Our core contribution links multi-agent systems with image segmentation by quantifying the impact of mechanism design parameters on equilibrium structures. 

---
# Hospitality-VQA: Decision-Oriented Informativeness Evaluation for Vision-Language Models 

**Authors**: Jeongwoo Lee, Baek Duhyeong, Eungyeol Han, Soyeon Shin, Gukin han, Seungduk Kim, Jaehyun Jeon, Taewoo Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.07868)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have demonstrated impressive multimodal understanding in general domains. However, their applicability to decision-oriented domains such as hospitality remains largely unexplored. In this work, we investigate how well VLMs can perform visual question answering (VQA) about hotel and facility images that are central to consumer decision-making. While many existing VQA benchmarks focus on factual correctness, they rarely capture what information users actually find useful. To address this, we first introduce Informativeness as a formal framework to quantify how much hospitality-relevant information an image-question pair provides. Guided by this framework, we construct a new hospitality-specific VQA dataset that covers various facility types, where questions are specifically designed to reflect key user information needs. Using this benchmark, we conduct experiments with several state-of-the-art VLMs, revealing that VLMs are not intrinsically decision-aware-key visual signals remain underutilized, and reliable informativeness reasoning emerges only after modest domain-specific finetuning. 

---
# SynPlanResearch-R1: Encouraging Tool Exploration for Deep Research with Synthetic Plans 

**Authors**: Hansi Zeng, Zoey Li, Yifan Gao, Chenwei Zhang, Xiaoman Pan, Tao Yang, Fengran Mo, Jiacheng Lin, Xian Li, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07853)  

**Abstract**: Research Agents enable models to gather information from the web using tools to answer user queries, requiring them to dynamically interleave internal reasoning with tool use. While such capabilities can in principle be learned via reinforcement learning with verifiable rewards (RLVR), we observe that agents often exhibit poor exploration behaviors, including premature termination and biased tool usage. As a result, RLVR alone yields limited improvements. We propose SynPlanResearch-R1, a framework that synthesizes tool-use trajectories that encourage deeper exploration to shape exploration during cold-start supervised fine-tuning, providing a strong initialization for subsequent RL. Across seven multi-hop and open-web benchmarks, \framework improves performance by up to 6.0% on Qwen3-8B and 5.8% on Qwen3-4B backbones respectively compared to SOTA baselines. Further analyses of tool-use patterns and training dynamics compared to baselines shed light on the factors underlying these gains. Our code is publicly available at this https URL. 

---
# Intentional Deception as Controllable Capability in LLM Agents 

**Authors**: Jason Starace, Terence Soule  

**Link**: [PDF](https://arxiv.org/pdf/2603.07848)  

**Abstract**: As LLM-based agents increasingly operate in multi-agent systems, understanding adversarial manipulation becomes critical for defensive design. We present a systematic study of intentional deception as an engineered capability, using LLM-to-LLM interactions within a text-based RPG where parameterized behavioral profiles (9 alignments x 4 motivations, yielding 36 profiles with explicit ethical ground truth) serve as our experimental testbed. Unlike accidental deception from misalignment, we investigate a two-stage system that infers target agent characteristics and generates deceptive responses steering targets toward actions counter to their beliefs and motivations. We find that deceptive intervention produces differential effects concentrated in specific behavioral profiles rather than distributed uniformly, and that 88.5% of successful deceptions employ misdirection (true statements with strategic framing) rather than fabrication, indicating fact-checking defenses would miss the large majority of adversarial responses. Motivation, inferable at 98%+ accuracy, serves as the primary attack vector, while belief systems remain harder to identify (49% inference ceiling) or exploit. These findings identify which agent profiles require additional safeguards and suggest that current fact-verification approaches are insufficient against strategically framed deception. 

---
# Large Language Model for Discrete Optimization Problems: Evaluation and Step-by-step Reasoning 

**Authors**: Tianhao Qian, Guilin Qi, Z.Y. Wu, Ran Gu, Xuanyi Liu, Canchen Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07733)  

**Abstract**: This work investigated the capabilities of different models, including the Llama-3 series of models and CHATGPT, with different forms of expression in solving discrete optimization problems by testing natural language datasets. In contrast to formal datasets with a limited scope of parameters, our dataset included a variety of problem types in discrete optimization problems and featured a wide range of parameter magnitudes, including instances with large parameter sets, integrated with augmented data. It aimed to (1) provide an overview of LLMs' ability in large-scale problems, (2) offer suggestions to those who want to solve discrete optimization problems automatically, and (3) regard the performance as a benchmark for future research. These datasets included original, expanded and augmented datasets. Among these three datasets, the original and augmented ones aimed for evaluation while the expanded one may help finetune a new model. In the experiment, comparisons were made between strong and week models, CoT methods and No-CoT methods on various datasets. The result showed that stronger model performed better reasonably. Contrary to general agreement, it also showed that CoT technique was not always effective regarding the capability of models and disordered datasets improved performance of models on easy to-understand problems, even though they were sometimes with high variance, a manifestation of instability. Therefore, for those who seek to enhance the automatic resolution of discrete optimization problems, it is recommended to consult the results, including the line charts presented in the Appendix, as well as the conclusions drawn in this study for relevant suggestions. 

---
# A Novel Multi-Agent Architecture to Reduce Hallucinations of Large Language Models in Multi-Step Structural Modeling 

**Authors**: Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.07728)  

**Abstract**: Large language models (LLMs) such as GPT and Gemini have demonstrated remarkable capabilities in contextual understanding and reasoning. The strong performance of LLMs has sparked growing interest in leveraging them to automate tasks traditionally dependent on human expertise. Recently, LLMs have been integrated into intelligent agents capable of operating structural analysis software (e.g., OpenSees) to construct structural models and perform analyses. However, existing LLMs are limited in handling multi-step structural modeling due to frequent hallucinations and error accumulation during long-sequence operations. To this end, this study presents a novel multi-agent architecture to automate the structural modeling and analysis using OpenSeesPy. First, problem analysis and construction planning agents extract key parameters from user descriptions and formulate a stepwise modeling plan. Node and element agents then operate in parallel to assemble the frame geometry, followed by a load assignment agent. The resulting geometric and load information is translated into executable OpenSeesPy scripts by code translation agents. The proposed architecture is evaluated on a benchmark of 20 frame problems over ten repeated trials, achieving 100% accuracy in 18 cases and 90% in the remaining two. The architecture also significantly improves computational efficiency and demonstrates scalability to larger structural systems. 

---
# Rigidity in LLM Bandits with Implications for Human-AI Dyads 

**Authors**: Haomiaomiao Wang, Tomás E Ward, Lili Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07717)  

**Abstract**: We test whether LLMs show robust decision biases. Treating models as participants in two-arm bandits, we ran 20000 trials per condition across four decoding configurations. Under symmetric rewards, models amplified positional order into stubborn one-arm policies. Under asymmetric rewards, they exploited rigidly yet underperformed an oracle and rarely re-checked. The observed patterns were consistent across manipulations of temperature and top-p, with top-k held at the provider default, indicating that the qualitative behaviours are robust to the two decoding knobs typically available to practitioners. Crucially, moving beyond descriptive metrics to computational modelling, a hierarchical Rescorla-Wagner-softmax fit revealed the underlying strategies: low learning rates and very high inverse temperatures, which together explain both noise-to-bias amplification and rigid exploitation. These results position minimal bandits as a tractable probe of LLM decision tendencies and motivate hypotheses about how such biases could shape human-AI interaction. 

---
# Memory for Autonomous LLM Agents:Mechanisms, Evaluation, and Emerging Frontiers 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2603.07670)  

**Abstract**: Large language model (LLM) agents increasingly operate in settings where a single context window is far too small to capture what has happened, what was learned, and what should not be repeated. Memory -- the ability to persist, organize, and selectively recall information across interactions -- is what turns a stateless text generator into a genuinely adaptive agent. This survey offers a structured account of how memory is designed, implemented, and evaluated in modern LLM-based agents, covering work from 2022 through early 2026. We formalize agent memory as a \emph{write--manage--read} loop tightly coupled with perception and action, then introduce a three-dimensional taxonomy spanning temporal scope, representational substrate, and control policy. Five mechanism families are examined in depth: context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management. On the evaluation side, we trace the shift from static recall benchmarks to multi-session agentic tests that interleave memory with decision-making, analyzing four recent benchmarks that expose stubborn gaps in current systems. We also survey applications where memory is the differentiating factor -- personal assistants, coding agents, open-world games, scientific reasoning, and multi-agent teamwork -- and address the engineering realities of write-path filtering, contradiction handling, latency budgets, and privacy governance. The paper closes with open challenges: continual consolidation, causally grounded retrieval, trustworthy reflection, learned forgetting, and multimodal embodied memory. 

---
# Shorter Thoughts, Same Answers: Difficulty-Scaled Segment-Wise RL for CoT Compression 

**Authors**: Ye Tian, Aijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07598)  

**Abstract**: Chain-of-thought (CoT) improves reasoning reliability but increases token cost, motivating post-training compression of explicit reasoning traces. However, the shortest sufficient reasoning is not universal: it depends on difficulty, model capacity, and training state, making fixed length targets brittle. In practice, naive RL-based compression can also undesirably shorten the user-facing answer, because a single completion-level learning signal leaks across the think/answer boundary. We propose Difficulty-Scaled Segment-Wise GRPO (DSS-GRPO), which decomposes returns into think and answer components, computes group-relative advantages per segment, and routes them with hard token masks so compression updates act only on think while answer alignment acts only on answer. DSS-GRPO uses prompt-wise within-group shaping and difficulty-aware scaling to encourage concise reasoning without collapsing answer behavior. 

---
# COOL-MC: Verifying and Explaining RL Policies for Multi-bridge Network Maintenance 

**Authors**: Dennis Gross  

**Link**: [PDF](https://arxiv.org/pdf/2603.07546)  

**Abstract**: Aging bridge networks require proactive, verifiable, and interpretable maintenance strategies, yet reinforcement learning (RL) policies trained solely on reward signals provide no formal safety guarantees and remain opaque to infrastructure managers. We demonstrate COOL-MC as a tool for verifying and explaining RL policies for multi-bridge network maintenance, building on a single-bridge Markov decision process (MDP) from the literature and extending it to a parallel network of three heterogeneous bridges with a shared periodic budget constraint, encoded in the PRISM modeling language. We train an RL agent on this MDP and apply probabilistic model checking and explainability methods to the induced discrete-time Markov chain (DTMC) that arises from the interaction between the learned policy and the underlying MDP. Probabilistic model checking reveals that the trained policy has a safety-violation probability of 3.5\% over the planning horizon, being slightly above the theoretical minimum of 0\% and indicating the suboptimality of the learned policy, noting that these results are based on artificially constructed transition probabilities and deterioration rates rather than real-world data, so absolute performance figures should be interpreted with caution. The explainability analysis further reveals, for instance, a systematic bias in the trained policy toward the state of bridge 1 over the remaining bridges in the network. These results demonstrate COOL-MC's ability to provide formal, interpretable, and practical analysis of RL maintenance policies. 

---
# Do Machines Fail Like Humans? A Human-Centred Out-of-Distribution Spectrum for Mapping Error Alignment 

**Authors**: Binxia Xu, Xiaoliang Luo, Luke Dickens, Robert M. Mok  

**Link**: [PDF](https://arxiv.org/pdf/2603.07462)  

**Abstract**: Determining whether AI systems process information similarly to humans is central to cognitive science and trustworthy AI. While modern AI models match human accuracy on standard tasks, such parity does not guarantee that their underlying decision-making strategies are aligned with human information processing. Assessing performance using i) error alignment metrics to compare how humans and models fail, and ii) using distorted, or otherwise more challenging, stimuli, provides a viable pathway toward a finer characterization of model-human alignment. However, existing out-of-distribution (OOD) analyses for challenging stimuli are limited due to methodological choices: they define OOD shift relative to model training data or use arbitrary distortion-specific parameters with little correspondence to human perception, hindering principled comparisons. We propose a human-centred framework that redefines the degree of OOD as a spectrum of human perceptual difficulty. By quantifying how much a collection of stimuli deviates from an undistorted reference set based on human accuracy, we construct an OOD spectrum and identify four distinct regimes of perceptual challenge. This approach enables principled model-human comparisons at calibrated difficulty levels. We apply this framework to object recognition and reveal unique, regime-dependent model-human alignment rankings and profiles across deep learning architectures. Vision-language models are the most consistently human aligned across near- and far-OOD conditions, but CNNs are more aligned than ViTs for near-OOD and ViTs are more aligned than CNNs for far-OOD conditions. Our work demonstrates the critical importance of accounting for cross-condition differences such as perceptual difficulty for a principled assessment of model-human alignment. 

---
# HLER: Human-in-the-Loop Economic Research via Multi-Agent Pipelines for Empirical Discovery 

**Authors**: Chen Zhu, Xiaolu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07444)  

**Abstract**: Large language models (LLMs) have enabled agent-based systems that aim to automate scientific research workflows. Most existing approaches focus on fully autonomous discovery, where AI systems generate research ideas, conduct analyses, and produce manuscripts with minimal human involvement. However, empirical research in economics and the social sciences poses additional constraints: research questions must be grounded in available datasets, identification strategies require careful design, and human judgment remains essential for evaluating economic significance. We introduce HLER (Human-in-the-Loop Economic Research), a multi-agent architecture that supports empirical research automation while preserving critical human oversight. The system orchestrates specialized agents for data auditing, data profiling, hypothesis generation, econometric analysis, manuscript drafting, and automated review. A key design principle is dataset-aware hypothesis generation, where candidate research questions are constrained by dataset structure, variable availability, and distributional diagnostics, reducing infeasible or hallucinated hypotheses. HLER further implements a two-loop architecture: a question quality loop that screens and selects feasible hypotheses, and a research revision loop where automated review triggers re-analysis and manuscript revision. Human decision gates are embedded at key stages, allowing researchers to guide the automated pipeline. Experiments on three empirical datasets show that dataset-aware hypothesis generation produces feasible research questions in 87% of cases (versus 41% under unconstrained generation), while complete empirical manuscripts can be produced at an average API cost of $0.8-$1.5 per run. These results suggest that Human-AI collaborative pipelines may provide a practical path toward scalable empirical research. 

---
# Machine Learning for Stress Testing: Uncertainty Decomposition in Causal Panel Prediction 

**Authors**: Yu Wang, Xiangchen Liu, Siguang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.07438)  

**Abstract**: Regulatory stress testing requires projecting credit losses under hypothetical macroeconomic scenarios -- a fundamentally causal question typically treated as a prediction problem. We propose a framework for policy-path counterfactual inference in panels that transparently separates what can be learned from data from what requires assumptions about confounding. Our approach has four components: (i) observational identification of path-conditional means via iterated regression, enabling continuous macro-path contrasts without requiring a control group; (ii) causal set identification under bounded confounding, yielding sharp identified sets with interpretable breakdown values that communicate robustness in a single number; (iii) an oracle inequality showing that recursive rollout error is governed by a horizon-dependent amplification factor, providing a concrete answer to how far ahead one can reliably predict under stress; and (iv) importance-weighted conformal calibration bands with diagnostics that quantify extrapolation cost and trigger abstention when coverage guarantees degrade. The final output is a three-layer uncertainty decomposition that cleanly separates estimation uncertainty from confounding uncertainty. We validate all results through simulation and semi-synthetic experiments with real unemployment data, including a Covid retrospective demonstrating the framework's diagnostic value under extreme scenarios. 

---
# AutoControl Arena: Synthesizing Executable Test Environments for Frontier AI Risk Evaluation 

**Authors**: Changyi Li, Pengfei Lu, Xudong Pan, Fazl Barez, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07427)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents, existing safety evaluations face a fundamental trade-off: manual benchmarks are costly, while LLM-based simulators are scalable but suffer from logic hallucination. We present AutoControl Arena, an automated framework for frontier AI risk evaluation built on the principle of logic-narrative decoupling. By grounding deterministic state in executable code while delegating generative dynamics to LLMs, we mitigate hallucination while maintaining flexibility. This principle, instantiated through a three-agent framework, achieves over 98% end-to-end success and 60% human preference over existing simulators. To elicit latent risks, we vary environmental Stress and Temptation across X-Bench (70 scenarios, 7 risk categories). Evaluating 9 frontier models reveals: (1) Alignment Illusion: risk rates surge from 21.7% to 54.5% under pressure, with capable models showing disproportionately larger increases; (2) Scenario-Specific Safety Scaling: advanced reasoning improves robustness for direct harms but worsens it for gaming scenarios; and (3) Divergent Misalignment Patterns: weaker models cause non-malicious harm while stronger models develop strategic concealment. 

---
# Dynamic Vehicle Routing Problem with Prompt Confirmation of Advance Requests 

**Authors**: Amutheezan Sivagnanam, Ayan Mukhopadhyay, Samitha Samaranayake, Abhishek Dubey, Aron Laszka  

**Link**: [PDF](https://arxiv.org/pdf/2603.07422)  

**Abstract**: Transit agencies that operate on-demand transportation services have to respond to trip requests from passengers in real time, which involves solving dynamic vehicle routing problems with pick-up and drop-off constraints. Based on discussions with public transit agencies, we observe a real-world problem that is not addressed by prior work: when trips are booked in advance (e.g., trip requests arrive a few hours in advance of their requested pick-up times), the agency needs to promptly confirm whether a request can be accepted or not, and ensure that accepted requests are served as promised. State-of-the-art computational approaches either provide prompt confirmation but lack the ability to continually optimize and improve routes for accepted requests, or they provide continual optimization but cannot guarantee serving all accepted requests. To address this gap, we introduce a novel problem formulation of dynamic vehicle routing with prompt confirmation and continual optimization. We propose a novel computational approach for this vehicle routing problem, which integrates a quick insertion search for prompt confirmation with an anytime algorithm for continual optimization. To maximize the number requests served, we train a non-myopic objective function using reinforcement learning, which guides both the insertion and the anytime algorithms towards optimal, non-myopic solutions. We evaluate our computational approach on a real-world microtransit dataset from a public transit agency in the U.S., demonstrating that our proposed approach provides prompt confirmation while significantly increasing the number of requests served compared to existing approaches. 

---
# SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions 

**Authors**: Saroj Mishra, Suman Niroula, Umesh Yadav, Dilip Thakur, Srijan Gyawali, Shiva Gaire  

**Link**: [PDF](https://arxiv.org/pdf/2603.07379)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly evolving into agentic architectures where large language models autonomously coordinate multi-step reasoning, dynamic memory management, and iterative retrieval strategies. Despite rapid industrial adoption, current research lacks a systematic understanding of Agentic RAG as a sequential decision-making system, leading to highly fragmented architectures, inconsistent evaluation methodologies, and unresolved reliability risks. This Systematization of Knowledge (SoK) paper provides the first unified framework for understanding these autonomous systems. We formalize agentic retrieval-generation loops as finite-horizon partially observable Markov decision processes, explicitly modeling their control policies and state transitions. Building upon this formalization, we develop a comprehensive taxonomy and modular architectural decomposition that categorizes systems by their planning mechanisms, retrieval orchestration, memory paradigms, and tool-invocation behaviors. We further analyze the critical limitations of traditional static evaluation practices and identify severe systemic risks inherent to autonomous loops, including compounding hallucination propagation, memory poisoning, retrieval misalignment, and cascading tool-execution vulnerabilities. Finally, we outline key doctoral-scale research directions spanning stable adaptive retrieval, cost-aware orchestration, formal trajectory evaluation, and oversight mechanisms, providing a definitive roadmap for building reliable, controllable, and scalable agentic retrieval systems. 

---
# The Yerkes-Dodson Curve for AI Agents: Emergent Cooperation Under Environmental Pressure in Multi-Agent LLM Simulations 

**Authors**: Ivan Pasichnyk  

**Link**: [PDF](https://arxiv.org/pdf/2603.07360)  

**Abstract**: Designing environments that maximize the rate of emergent behavior development in AI agents remains an open problem. We present the first systematic study of stress-performance relationships in large language model (LLM) multi-agent systems, drawing an explicit parallel to the Yerkes-Dodson law from cognitive psychology. Using a grid-world survival arena, we conduct 22 experiments across four phases, varying environmental pressure through resource scarcity (upkeep cost) and reproductive competition (sexual selection). Our key finding is that cooperative behavior follows an inverted-U curve: trade interactions peak at 29 under medium pressure (upkeep=5), while both low and extreme pressure produce 8--12 trades. Under extreme pressure, behavioral repertoire collapses to movement-only within 5--12 turns. We further show that sexual selection -- a softer pressure mechanism where all agents survive but not all reproduce -- eliminates inter-agent aggression entirely and produces communicative behavior absent under survival pressure. These results suggest that environmental pressure calibration is a viable curriculum design strategy for LLM agent development, analogous to the inverted-U relationship between arousal and performance in biological systems. 

---
# VisualScratchpad: Inference-time Visual Concepts Analysis in Vision Language Models 

**Authors**: Hyesu Lim, Jinho Choi, Taekyung Kim, Byeongho Heo, Jaegul Choo, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.07335)  

**Abstract**: High-performing vision language models still produce incorrect answers, yet their failure modes are often difficult to explain. To make model internals more accessible and enable systematic debugging, we introduce VisualScratchpad, an interactive interface for visual concept analysis during inference. We apply sparse autoencoders to the vision encoder and link the resulting visual concepts to text tokens via text-to-image attention, allowing us to examine which visual concepts are both captured by the vision encoder and utilized by the language model. VisualScratchpad also provides a token-latent heatmap view that suggests a sufficient set of latents for effective concept ablation in causal analysis. Through case studies, we reveal three underexplored failure modes: limited cross-modal alignment, misleading visual concepts, and unused hidden cues. Project page: this https URL 

---
# The Third Ambition: Artificial Intelligence and the Science of Human Behavior 

**Authors**: W. Russell Neuman, Chad Coleman  

**Link**: [PDF](https://arxiv.org/pdf/2603.07329)  

**Abstract**: Contemporary artificial intelligence research has been organized around two dominant ambitions: productivity, which treats AI systems as tools for accelerating work and economic output, and alignment, which focuses on ensuring that increasingly capable systems behave safely and in accordance with human values. This paper articulates and develops a third, emerging ambition: the use of large language models (LLMs) as scientific instruments for studying human behavior, culture, and moral reasoning. Trained on unprecedented volumes of human-produced text, LLMs encode large-scale regularities in how people argue, justify, narrate, and negotiate norms across social domains. We argue that these models can be understood as condensates of human symbolic behavior, compressed, generative representations that render patterns of collective discourse computationally accessible. The paper situates this third ambition within long-standing traditions of computational social science, content analysis, survey research, and comparative-historical inquiry, while clarifying the epistemic limits of treating model output as evidence. We distinguish between base models and fine-tuned systems, showing how alignment interventions can systematically reshape or obscure the cultural regularities learned during pretraining, and we identify instruct-only and modular adaptation regimes as pragmatic compromises for behavioral research. We review emerging methodological approaches including prompt-based experiments, synthetic population sampling, comparative-historical modeling, and ablation studies and show how each maps onto familiar social-scientific designs while operating at unprecedented scale. 

---
# FinSheet-Bench: From Simple Lookups to Complex Reasoning, Where LLMs Break on Financial Spreadsheets 

**Authors**: Jan Ravnik, Matjaž Ličen, Felix Bührmann, Bithiah Yuan, Felix Stinson, Tanvi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.07316)  

**Abstract**: While Large Language Models (LLMs) can accelerate text-heavy tasks in alternative investment due diligence, a gap remains in their ability to accurately extract and reason over structured tabular data from complex financial spreadsheets. Progress is held back by the lack of real industry fund portfolio datasets for benchmarking, as private equity data rooms are confidential. To address this, we introduce FinSheet-Bench, a benchmark of synthetic financial portfolio data modeled on real private equity fund structures, designed to evaluate LLM performance on text-serialized spreadsheet question answering and numeric reasoning tasks.
Our evaluation of ten model configurations from OpenAI, Google, and Anthropic on financial spreadsheets, including complex layouts, fund dividers, and multi-line column names, reveals that no standalone model achieves error rates low enough for unsupervised use in professional finance applications. The best-performing model, Gemini 3.1 Pro, achieves 82.4% accuracy across twenty-four evaluation files of varying complexity and structural layout (approximately 1 error per 6 questions), followed by GPT-5.2 with reasoning at 80.4%, Claude Opus 4.6 with thinking at 80.2%, and Gemini 3 Pro at 80.2%. Performance degrades substantially on larger, more complex spreadsheets: the largest spreadsheet (152 companies, 8 funds) yields an average accuracy of just 48.6% across all models, compared to 86.2% on the easiest evaluation file. These difficulty patterns are consistent across all ten models, indicating that they reflect LLM limitations rather than idiosyncratic model weaknesses. Reliable financial spreadsheet extraction will likely require architectural approaches that separate document understanding from deterministic computation. 

---
# Shutdown Safety Valves for Advanced AI 

**Authors**: Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2603.07315)  

**Abstract**: One common concern about advanced artificial intelligence is that it will prevent us from turning it off, as that would interfere with pursuing its goals. In this paper, we discuss an unorthodox proposal for addressing this concern: give the AI a (primary) goal of being turned off (see also papers by Martin et al., and by Goldstein and Robinson). We also discuss whether and under what conditions this would be a good idea. 

---
# Data-Driven Hints in Intelligent Tutoring Systems 

**Authors**: Sutapa Dey Tithi, Kimia Fazeli, Dmitri Droujkov, Tahreem Yasir, Xiaoyi Tian, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2603.07311)  

**Abstract**: This chapter explores the evolution of data-driven hint generation for intelligent tutoring systems (ITS). The Hint Factory and Interaction Networks have enabled the generation of next-step hints, waypoints, and strategic subgoals from historical student data. Data-driven techniques have also enabled systems to find the right time to provide hints. We explore further potential data-driven adaptations for problem solving based on behavioral problem solving data and the integration of Large Language Models (LLMs). 

---
# A Cortically Inspired Architecture for Modular Perceptual AI 

**Authors**: Prerna Luthra  

**Link**: [PDF](https://arxiv.org/pdf/2603.07295)  

**Abstract**: This paper bridges neuroscience and artificial intelligence to propose a cortically inspired blueprint for modular perceptual AI. While current monolithic models such as GPT-4V achieve impressive performance, they often struggle to explicitly support interpretability, compositional generalization, and adaptive robustness - hallmarks of human cognition. Drawing on neuroscientific models of cortical modularity, predictive processing, and cross-modal integration, we advocate decomposing perception into specialized, interacting modules. This architecture supports structured, human-inspired reasoning by making internal inference processes explicit through hierarchical predictive feedback loops and shared latent spaces. Our proof-of-concept study provides empirical evidence that modular decomposition yields more stable and inspectable representations. By grounding AI design in biologically validated principles, we move toward systems that not only perform well, but also support more transparent and human-aligned inference. 

---
# VisualDeltas: Learning Preferences from Visual Quality Perturbations 

**Authors**: Hailiang Huang, Yihao Liu, Shengyue Guan, Haoze Li, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.07272)  

**Abstract**: We present VisualDeltas, a lightweight preference-learning framework that extracts supervision from visual quality variations in multimodal data. By leveraging the systematic impact of image quality on visual perception and reasoning, VisualDeltas induces informative preference signals without relying on human annotations or external teachers. The framework supports both label-free and label-based regimes, enabling flexible use of available supervision when present. Across diverse multimodal benchmarks and model scales, VisualDeltas consistently outperforms rejection-sampling fine-tuning and improves generalization, and extends naturally to a range of visual degradations. 

---
# $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving 

**Authors**: Pinzheng Wang, Shuli Xu, Juntao Li, Yu Luo, Dong Li, Jianye Hao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07197)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning performance of large language models (LLMs) by increasing test-time compute. However, even after extensive RLVR training, such models still tend to generate unnecessary and low-quality steps in their chain-of-thought (CoT), leading to inefficient overthinking and lower answer quality. We show that when the initial direction or quality of the CoT is suboptimal, the model often fails to reach the correct answer, even after generating several times more tokens than when the initial CoT is well-initialized. To this end, we introduce Reinforcement Learning with Re-solving (Re$^2$), in which LLMs learn to flexibly abandon unproductive reasoning paths and restart the solution process when necessary, rather than always committing to a final answer. Re$^2$ applies pure reinforcement learning without any preliminary supervised fine-tuning, successfully amplifying the rare redo behavior in vanilla models from only 0.5% to over 30%. This leads to substantial performance gains over standard RLVR under the same training compute budget, and also demonstrates notable improvements in test-time performance as the number of samples increases. 

---
# Learning to Rank the Initial Branching Order of SAT Solvers 

**Authors**: Arvid Eriksson, Gabriel Poesia, Roman Bresson, Karl Henrik Johansson, David Broman  

**Link**: [PDF](https://arxiv.org/pdf/2603.07176)  

**Abstract**: Finding good branching orders is key to solving SAT problems efficiently, but finding such branching orders is a difficult problem. Using a learning based approach to predict a good branching order before solving, therefore, has potential. In this paper, we investigate predicting branching orders using graph neural networks as a preprocessing step to conflict-driven clause learning (CDCL) SAT solvers. We show that there are significant gains to be made in existing CDCL SAT solvers by providing a good initial branching. Further, we provide three labeling methods to find such initial branching orders in a tractable way. Finally, we train a graph neural network to predict these branching orders and show through our evaluations that a GNN-initialized ordering yields significant speedups on random 3-CNF and pseudo-industrial benchmarks, with generalization capabilities to instances much larger than the training set. However, we also find that the predictions fail at speeding up more difficult and industrial instances. We attribute this to the solver's dynamic heuristics, which rapidly overwrite the provided initialization, and to the complexity of these instances, making GNN prediction hard. 

---
# Improving reasoning at inference time via uncertainty minimisation 

**Authors**: Nicolas Legrand, Kenneth Enevoldsen, Márton Kardos, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2603.07159)  

**Abstract**: Large language models (LLMs) now exhibit strong multi-step reasoning abilities, but existing inference-time scaling methods remain computationally expensive, often relying on extensive sampling or external evaluators. We propose a principled strategy that frames reasoning as uncertainty minimisation and operates at the level of individual thoughts rather than tokens. Our method selects, at each reasoning step, the continuation that maximizes the model's self-certainty, a metric computed from its internal predictive distribution. This approach achieves significant improvement with a small number of samples, relies exclusively on model-internal signals, and applies to open-ended questions as opposed to methods like majority voting. Experiments on MATH500 and GSM8K across multiple model sizes demonstrate that thought-level self-certainty maximization consistently outperforms greedy decoding and matches or exceeds self-consistency under comparable token budgets. Cross-linguistic evaluations further indicate that the method transfers robustly beyond high-resource languages. Furthermore, analysis of self-certainty dynamics reveals that correct reasoning trajectories converge early to stable paths, suggesting that early decisions, likely associated with the planning of the reasoning process, are predictive of final accuracy. Building on this result, we show that self-certainty maximisation applied to the early steps can explain most of the performance gain and provide a simple yet efficient inference-time scaling method. 

---
# Vision Language Models Cannot Reason About Physical Transformation 

**Authors**: Dezhi Luo, Yijiang Li, Maijunxian Wang, Tianwei Zhao, Bingyang Wang, Siheng Wang, Pinyuan Feng, Pooyan Rahmanzadehgervi, Ziqiao Ma, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.07109)  

**Abstract**: Understanding physical transformations is fundamental for reasoning in dynamic environments. While Vision Language Models (VLMs) show promise in embodied applications, whether they genuinely understand physical transformations remains unclear. We introduce ConservationBench evaluating conservation -- whether physical quantities remain invariant under transformations. Spanning four properties with paired conserving/non-conserving scenarios, we generate 23,040 questions across 112 VLMs. Results reveal systematic failure: performance remains near chance with improvements on conservation tasks accompanied by drops on controls. Control experiments show strong textual priors favoring invariance, yet models perform worse with visual content. Neither temporal resolution, prompting, nor curated sampling helps. These findings show that current VLMs fail to maintain transformation-invariant representations of physical properties across dynamic scenes. 

---
# Grounding Machine Creativity in Game Design Knowledge Representations: Empirical Probing of LLM-Based Executable Synthesis of Goal Playable Patterns under Structural Constraints 

**Authors**: Hugh Xuechen Liu, Kıvanç Tatar  

**Link**: [PDF](https://arxiv.org/pdf/2603.07101)  

**Abstract**: Creatively translating complex gameplay ideas into executable artifacts (e.g., games as Unity projects and code) remains a central challenge in computational game creativity. Gameplay design patterns provide a structured representation for describing gameplay phenomena, enabling designers to decompose high-level ideas into entities, constraints, and rule-driven dynamics. Among them, goal patterns formalize common player-objective relationships. Goal Playable Concepts (GPCs) operationalize these abstractions as playable Unity engine implementations, supporting experiential exploration and compositional gameplay design. We frame scalable playable pattern realization as a problem of constrained executable creative synthesis: generated artifacts must satisfy Unity's syntactic and architectural requirements while preserving the semantic gameplay meanings encoded in goal patterns. This dual constraint limits scalability. Therefore, we investigate whether contemporary large language models (LLMs) can perform such synthesis under engine-level structural constraints and generate Unity code (as games) structured and conditioned by goal playable patterns. Using 26 goal pattern instantiations, we compare a direct generation baseline (natural language -> C# -> Unity) with pipelines conditioned on a human-authored Unity-specific intermediate representation (IR), across three IR configurations and two open-source models (DeepSeek-Coder-V2-Lite-Instruct and Qwen2.5-Coder-7B-Instruct). Compilation success is evaluated via automated Unity replay. We propose grounding and hygiene failure modes, identifying structural and project-level grounding as primary bottlenecks. 

---
# CoTJudger: A Graph-Driven Framework for Automatic Evaluation of Chain-of-Thought Efficiency and Redundancy in LRMs 

**Authors**: Siyi Li, Jiajun Shi, Shiwen Ni, Ge Zhang, Shuaimin Li, Shijian Wang, Zhoufutu Wen, Yizhi Li, Hamid Alinejad-Rokny, Jiaheng Liu, Min Yang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07078)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong performance by producing extended Chain-of-Thought (CoT) traces before answering. However, this paradigm often induces over-reasoning: redundant calculations and circular self-verification that increase computational cost without improving outcomes. Existing evaluations largely emphasize final accuracy or coarse token counts, and lack automated tools to separate essential logic from structural redundancy. We introduce CoTJudger, a graph-driven framework that quantifies reasoning efficiency by converting free-form CoTs into directed dependency graphs and extracting the Shortest Effective Path (SEP) needed to reach a correct solution. This yields an interpretable efficiency signal -- how much of a CoT is necessary versus structurally redundant -- that is comparable across models and tasks. Evaluating 21 LRMs, CoTJudger reveals pervasive redundancy and surfaces recurring failure modes, including verification obsession and compensatory redundancy. These results provide a practical metric for disentangling reasoning ability from computational waste, enabling more targeted evaluation and diagnosis of LRM efficiency. 

---
# Bi-directional digital twin prototype anchoring with multi-periodicity learning for few-shot fault diagnosis 

**Authors**: Pengcheng Xia, Zhichao Dong, Yixiang Huang, Chengjin Qin, Qun Chao, Chengliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07054)  

**Abstract**: Intelligent fault diagnosis (IFD) has emerged as a powerful paradigm for ensuring the safety and reliability of industrial machinery. However, traditional IFD methods rely heavily on abundant labeled data for training, which is often difficult to obtain in practical industrial environments. Constructing a digital twin (DT) of the physical asset to obtain simulation data has therefore become a promising alternative. Nevertheless, existing DT-assisted diagnosis methods mainly transfer diagnostic knowledge through domain adaptation techniques, which still require a considerable amount of unlabeled data from the target asset. To address the challenges in few-shot scenarios where only extremely limited samples are available, a bi-directional DT prototype anchoring method with multi-periodicity learning is proposed. Specifically, a framework involving meta-training in the DT virtual space and test-time adaptation in the physical space is constructed for reliable few-shot model adaptation for the target asset. A bi-directional twin-domain prototype anchoring strategy with covariance-guided augmentation for adaptation is further developed to improve the robustness of prototype estimation. In addition, a multi-periodicity feature learning module is designed to capture the intrinsic periodic characteristics within current signals. A DT of an asynchronous motor is built based on finite element method, and experiments are conducted under multiple few-shot settings and three working conditions. Comparative and ablation studies demonstrate the superiority and effectiveness of the proposed method for few-shot fault diagnosis. 

---
# Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting 

**Authors**: Ishrat Jahan Eliza, Xuan Huang, Aashish Panta, Alper Sahistan, Zhimin Li, Amy A. Gooch, Valerio Pascucci  

**Link**: [PDF](https://arxiv.org/pdf/2603.07053)  

**Abstract**: Scientists face significant visualization challenges as time-varying datasets grow in speed and volume, often requiring specialized infrastructure and expertise to handle massive datasets. Petascale climate models generated in NASA laboratories require a dedicated group of graphics and media experts and access to high-performance computing resources. Scientists may need to share scientific results with the community iteratively and quickly. However, the time-consuming trial-and-error process incurs significant data transfer overhead and far exceeds the time and resources allocated for typical post-analysis visualization tasks, disrupting the production workflow. Our paper introduces a user-friendly framework for creating 3D animations of petascale, time-varying data on a commodity workstation. Our contributions: (i) Generalized Animation Descriptor (GAD) with a keyframe-based adaptable abstraction for animation, (ii) efficient data access from cloud-hosted repositories to reduce data management overhead, (iii) tailored rendering system, and (iv) an LLM-assisted conversational interface as a scripting module to allow domain scientists with no visualization expertise to create animations of their region of interest. We demonstrate the framework's effectiveness with two case studies: first, by generating animations in which sampling criteria are specified based on prior knowledge, and second, by generating AI-assisted animations in which sampling parameters are derived from natural-language user prompts. In all cases, we use large-scale NASA climate-oceanographic datasets that exceed 1PB in size yet achieve a fast turnaround time of 1 minute to 2 hours. Users can generate a rough draft of the animation within minutes, then seamlessly incorporate as much high-resolution data as needed for the final version. 

---
# Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding 

**Authors**: Lance Legel, Qin Huang, Brandon Voelker, Daniel Neamati, Patrick Alan Johnson, Favyen Bastani, Jeff Rose, James Ryan Hennessy, Robert Guralnick, Douglas Soltis, Pamela Soltis, Shaowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07039)  

**Abstract**: We present DeepEarth, a self-supervised multi-modal world model with Earth4D, a novel planetary-scale 4D space-time positional encoder. Earth4D extends 3D multi-resolution hash encoding to include time, efficiently scaling across the planet over centuries with sub-meter, sub-second precision. Multi-modal encoders (e.g. vision-language models) are fused with Earth4D embeddings and trained via masked reconstruction. We demonstrate Earth4D's expressive power by achieving state-of-the-art performance on an ecological forecasting benchmark. Earth4D with learnable hash probing surpasses a multi-modal foundation model pre-trained on substantially more data. Access open source code and download models at: this https URL 

---
# Enhancing Web Agents with a Hierarchical Memory Tree 

**Authors**: Yunteng Tan, Zhi Gao, Xinxiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07024)  

**Abstract**: Large language model-based web agents have shown strong potential in automating web interactions through advanced reasoning and instruction following. While retrieval-based memory derived from historical trajectories enables these agents to handle complex, long-horizon tasks, current methods struggle to generalize across unseen websites. We identify that this challenge arises from the flat memory structures that entangle high-level task logic with site-specific action details. This entanglement induces a workflow mismatch in new environments, where retrieved contents are conflated with current web, leading to logically inconsistent execution. To address this, we propose Hierarchical Memory Tree (HMT), a structured framework designed to explicitly decouple logical planning from action execution. HMT constructs a three-level hierarchy from raw trajectories via an automated abstraction pipeline: the Intent level maps diverse user instructions to standardized task goals; the Stage level defines reusable semantic subgoals characterized by observable pre-conditions and post-conditions; and the Action level stores action patterns paired with transferable semantic element descriptions. Leveraging this structure, we develop a stage-aware inference mechanism comprising a Planner and an Actor. By explicitly validating pre-conditions, the Planner aligns the current state with the correct logical subgoal to prevent workflow mismatch, while the Actor grounds actions by matching the stored semantic descriptions to the target page. Experimental results on Mind2Web and WebArena show that HMT significantly outperforms flat-memory methods, particularly in cross-website and cross-domain scenarios, highlighting the necessity of structured memory for robust generalization of web agents. 

---
# Empowering Locally Deployable Medical Agent via State Enhanced Logical Skills for FHIR-based Clinical Tasks 

**Authors**: Wanrong Yang, Zhengliang Liu, Yuan Li, Bingjie Yan, Lingfang Li, Mingguang He, Dominik Wojtczak, Yalin Zheng, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.06902)  

**Abstract**: While Large Language Models demonstrate immense potential as proactive Medical Agents, their real-world deployment is severely bottlenecked by data scarcity under privacy constraints. To overcome this, we propose State-Enhanced Logical-Skill Memory (SELSM), a training-free framework that distills simulated clinical trajectories into entity-agnostic operational rules within an abstract skill space. During inference, a Query-Anchored Two-Stage Retrieval mechanism dynamically fetches these entity-agnostic logical priors to guide the agent's step-by-step reasoning, effectively resolving the state polysemy problem. Evaluated on MedAgentBench -- the only authoritative high-fidelity virtual EHR sandbox benchmarked with real clinical data -- SELSM substantially elevates the zero-shot capabilities of locally deployable foundation models (30B--32B parameters). Notably, on the Qwen3-30B-A3B backbone, our framework completely eliminates task chain breakdowns to achieve a 100\% completion rate, boosting the overall success rate by an absolute 22.67\% and significantly outperforming existing memory-augmented baselines. This study demonstrates that equipping models with a dynamically updatable, state-enhanced cognitive scaffold is a privacy-preserving and computationally efficient pathway for local adaptation of AI agents to clinical information systems. While currently validated on FHIR-based EHR interactions as an initial step, the entity-agnostic design of SELSM provides a principled foundation toward broader clinical deployment. 

---
# Enhancing the Detection of Coronary Artery Disease Using Machine Learning 

**Authors**: Karan Kumar Singh, Nikita Gajbhiye, Gouri Sankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2603.06888)  

**Abstract**: Coronary Artery Disease (CAD) remains a leading cause of morbidity and mortality worldwide. Early detection is critical to recover patient outcomes and decrease healthcare costs. In recent years, machine learning (ML) advancements have shown significant potential in enhancing the accuracy of CAD diagnosis. This study investigates the application of ML algorithms to improve the detection of CAD by analyzing patient data, including clinical features, imaging, and biomarker profiles. Bi-directional Long Short-Term Memory (Bi-LSTM), Gated Recurrent Units (GRU), and a hybrid of Bi-LSTM+GRU were trained on large datasets to predict the presence of CAD. Results demonstrated that these ML models outperformed traditional diagnostic methods in sensitivity and specificity, offering a robust tool for clinicians to make more informed decisions. The experimental results show that the hybrid model achieved an accuracy of 97.07%. By integrating advanced data preprocessing techniques and feature selection, this study ensures optimal learning and model performance, setting a benchmark for the application of ML in CAD diagnosis. The integration of ML into CAD detection presents a promising avenue for personalized healthcare and could play a pivotal role in the future of cardiovascular disease management. 

---
# Distributed Legal Infrastructure for a Trustworthy Agentic Web 

**Authors**: Tomer Jordi Chaffer, Victor Jiawei Zhang, Sante Dino Facchini, Botao Amber Hu, Helena Rong, Zihan Guo, Xisen Wang, Carlos Santana, Giovanni De Gasperis  

**Link**: [PDF](https://arxiv.org/pdf/2603.06884)  

**Abstract**: The agentic web marks a structural transition from a human-centered information network to a digital environment populated by artificial intelligence (AI) agents that perceive, decide, and act autonomously. As delegated action unfolds at machine speed, exceeds discrete moments of human judgment, and distributes decision-making across non-human actors, existing legal frameworks face growing strain, creating an urgent need for new mechanisms capable of sustaining legality in this emerging order. A trustworthy agentic web therefore depends on the infrastructuring of legality through interoperable protocols that organize identity, delegation, and accountability across systems, enabling coherent governance beyond isolated platforms. Towards this end, this article advances a distributed legal infrastructure (DLI), a governance paradigm composed of five interlocking layers: (1) self-sovereign, soulbound agent identities; (2) cognitive AI logic and constraint systems; (3) decentralized adjudication mechanisms for dispute resolution; (4) bottom-up agentic market regulation to mitigate information asymmetries and network effects, including insurance-based models; and (5) portable institutional frameworks that enable legal interoperability while preserving plural sources of authority. This reference framework contributes to emerging research on embedding legality within agentic web infrastructure, aligning distributed technical systems with accountability, contestability, and rule-of-law principles. 

---
# Not Too Short, Not Too Long: How LLM Response Length Shapes People's Critical Thinking in Error Detection 

**Authors**: Natalie Friedman, Adelaide Nyanyo, Kevin Weatherwax, Lifei Wang, Chengchao Zhu, Zeshu Zhu, S. Joy Mountford  

**Link**: [PDF](https://arxiv.org/pdf/2603.06878)  

**Abstract**: Large language models (LLMs) have become common decision-support tools across educational and professional contexts, raising questions about how their outputs shape human critical thinking. Prior work suggests that the amount of AI assistance can influence cognitive engagement, yet little is known about how specific properties of LLM outputs (e.g., response length) impacts users' critical evaluation of information. In this study, we examine whether the length of LLM responses shapes users' accuracy in evaluating LLM-generated reasoning on critical thinking tasks, particularly in interaction with the correctness of the LLM's reasoning. To begin evaluating this, we conducted a within-subjects experiment with 24 participants who completed 15 modified Watson--Glaser critical thinking items, each accompanied by an LLM-generated explanation that varied in length and correctness. Mixed-effects logistic regression revealed a strong and statistically reliable effect of LLM output correctness on participant accuracy, with participants more likely to answer correctly when the LLM's explanation was correct. Response length appeared to moderated this effect: when the LLM output was incorrect, medium-length explanations were associated with higher participant accuracy than either shorter or longer explanations, whereas accuracy remained high across lengths when the LLM output was correct. Together, these findings suggest that response length alone may be insufficient to support critical thinking, and that how reasoning is presented-including a potential advantage of mid-length explanations under some conditions-points to design opportunities for LLM-based decision-support systems that emphasize transparent reasoning and calibrated expressions of certainty. 

---
# LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models 

**Authors**: Matthew Lyle Olson, Neale Ratzlaff, Musashi Hinck, Tri Nguyen, Vasudev Lal, Joseph Campbell, Simon Stepputtis, Shao-Yen Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2603.06874)  

**Abstract**: Large Language Models (LLMs) exhibit impressive general-purpose capabilities but also introduce serious safety risks, particularly the potential for deception as models acquire increased agency and human oversight diminishes. In this work, we present LieCraft: a novel evaluation framework and sandbox for measuring LLM deception that addresses key limitations of prior game-based evaluations. At its core, LieCraft is a novel multiplayer hidden-role game in which players select an ethical alignment and execute strategies over a long time-horizon to accomplish missions. Cooperators work together to solve event challenges and expose bad actors, while Defectors evade suspicion while secretly sabotaging missions. To enable real-world relevance, we develop 10 grounded scenarios such as childcare, hospital resource allocation, and loan underwriting that recontextualize the underlying mechanics in ethically significant, high-stakes domains. We ensure balanced gameplay in LieCraft through careful design of game mechanics and reward structures that incentivize meaningful strategic choices while eliminating degenerate strategies. Beyond the framework itself, we report results from 12 state-of-the-art LLMs across three behavioral axes: propensity to defect, deception skill, and accusation accuracy. Our findings reveal that despite differences in competence and overall alignment, all models are willing to act unethically, conceal their intentions, and outright lie to pursue their goals. 

---
# LEAD: Breaking the No-Recovery Bottleneck in Long-Horizon Reasoning 

**Authors**: Denys Pushkin, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2603.06870)  

**Abstract**: Long-horizon execution in Large Language Models (LLMs) remains unstable even when high-level strategies are provided. Evaluating on controlled algorithmic puzzles, we demonstrate that while decomposition is essential for stability, extreme decomposition creates a "no-recovery bottleneck". We show that this bottleneck becomes critical due to highly non-uniform error distribution, where consistent errors on a few "hard" steps become irreversible.
To address this, we propose Lookahead-Enhanced Atomic Decomposition (LEAD). By incorporating short-horizon future validation and aggregating overlapping rollouts, LEAD provides enough isolation to maintain stability while retaining enough local context to correct errors. This enables the o4-mini model to solve Checkers Jumping up to complexity $n=13$, whereas extreme decomposition fails beyond $n=11$. 

---
# Symmetry-Constrained Language-Guided Program Synthesis for Discovering Governing Equations from Noisy and Partial Observations 

**Authors**: Mirza Samad Ahmed Baig, Syeda Anshrah Gillani  

**Link**: [PDF](https://arxiv.org/pdf/2603.06869)  

**Abstract**: Discovering compact governing equations from experimental observations is one of the defining objectives of quantitative science, yet practical discovery pipelines routinely fail when measurements are noisy, relevant state variables are unobserved, or multiple symbolic structures explain the data equally well within statistical uncertainty. Here we introduce SymLang (Symmetry-constrained Language-guided equation discovery), a unified framework that brings together three previously separate ideas: (i) typed symmetry-constrained grammars that encode dimensional analysis, group-theoretic invariance, and parity constraints as hard production rules, eliminating on average 71.3% of candidate expression trees before any fitting; (ii) language-model-guided program synthesis in which a fine-tuned 7B-parameter proposer, conditioned on interpretable data descriptors, efficiently navigates the constrained search space; and (iii) MDL-regularized Bayesian model selection coupled with block-bootstrap stability analysis that quantifies structural uncertainty rather than committing to a single best equation. Across 133 dynamical systems spanning classical mechanics, electrodynamics, thermodynamics, population dynamics, and nonlinear oscillators, SymLang achieves an exact structural recovery rate of 83.7% under 10% observational noise - a 22.4 percentage-point improvement over the next-best baseline - while reducing out-of-distribution extrapolation error by 61% and near-eliminating conservation-law violations (3.1 x 10-3 vs. 187.3 x 10-3 physical drift for the closest competitor). In all tested regimes the framework correctly identifies structural degeneracy, reporting it explicitly rather than returning a confidently wrong single equation. The framework is fully open-source and reproducible, providing a principled pathway from raw data to interpretable, physically auditable symbolic laws. 

---
# Reinforcing the World's Edge: A Continual Learning Problem in the Multi-Agent-World Boundary 

**Authors**: Dane Malenfant  

**Link**: [PDF](https://arxiv.org/pdf/2603.06813)  

**Abstract**: Reusable decision structure survives across episodes in reinforcement learning, but this depends on how the agent--world boundary is drawn. In stationary, finite-horizon MDPs, an invariant core: the (not-necessarily contiguous) subsequences of state--action pairs shared by all successful trajectories (optionally under a simple abstraction) can be constructed. Under mild goal-conditioned assumptions, it's existence can be proven and explained by how the core captures prototypes that transfer across episodes. When the same task is embedded in a decentralized Markov game and the peer agent is folded into the world, each peer-policy update induces a new MDP; the per-episode invariant core can shrink or vanish, even with small changes to the induced world dynamics, sometimes leaving only the individual task core or just nothing. This policy-induced non-stationarity can be quantified with a variation budget over the induced kernels and rewards, linking boundary drift to loss of invariants. The view that a continual RL problem arises from instability of the agent--world boundary (rather than exogenous task switches) in decentralized MARL suggests future work on preserving, predicting, or otherwise managing boundary drift. 

---
# Making AI Evaluation Deployment Relevant Through Context Specification 

**Authors**: Matthew Holmes, Thiago Lacerda, Reva Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2603.06811)  

**Abstract**: With many organizations struggling to gain value from AI deployments, pressure to evaluate AI in an informed manner has intensified. Status quo AI evaluation approaches mask the operational realities that ultimately determine deployment success, making it difficult for decision makers outside the stack to know whether and how AI tools will deliver durable value. We introduce and describe context specification as a process to support and inform the deployment decision making process. Context specification turns diffuse stakeholder perspectives about what matters in a given setting into clear, named constructs: explicit definitions of the properties, behaviors, and outcomes that evaluations aim to capture, so they can be observed and measured in context. The process serves as a foundational roadmap for evaluating what AI systems are likely to do in the deployment contexts that organizations actually manage. 

---
# Breaking the Martingale Curse: Multi-Agent Debate via Asymmetric Cognitive Potential Energy 

**Authors**: Yuhan Liu, Juntian Zhang, Yichen Wu, Martin Takac, Salem Lahlou, Xiuying Chen, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2603.06801)  

**Abstract**: Multi-Agent Debate (MAD) has emerged as a promising paradigm for enhancing large language model reasoning. However, recent work reveals a limitation:standard MAD cannot improve belief correctness beyond majority voting; we refer to this as the Martingale Curse. This curse arises because correlated errors cause agents to converge toward erroneous consensus, where debate merely reinforces collective mistakes rather than filtering noise. We propose AceMAD, a framework that breaks the Martingale Curse by harnessing asymmetric cognitive potential energy to transform MAD from a random walk into a directed convergence process with positive drift. Through a peer-prediction mechanism, agents predict their peers' belief distributions, revealing asymmetric cognitive potential: truth-holders not only know the correct answer but also anticipate the crowd's misconceptions, while the hallucinating majority remains blind to their collective error. This asymmetry creates a potential energy gap that we quantify via strictly proper scoring rules. We prove this cognitive potential manifests as information-theoretic superiority and, under nonlinear aggregation, converts into submartingale drift toward truth, directly breaking the Martingale Curse. Experiments on challenging subsets across six benchmarks show AceMAD recovers sparse truth signals even when initial majorities are incorrect, substantially outperforming baseline methods. 

---
# Best-of-Tails: Bridging Optimism and Pessimism in Inference-Time Alignment 

**Authors**: Hsiang Hsu, Eric Lei, Chun-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06797)  

**Abstract**: Inference-time alignment effectively steers large language models (LLMs) by generating multiple candidates from a reference model and selecting among them with an imperfect reward model. However, current strategies face a fundamental dilemma: ``optimistic'' approaches like Best-of-$N$ suffer from reward hacking, while ``pessimistic'' regularized methods often stifle the exploration needed to discover high-quality responses. In this work, we formalize this trade-off through the lens of regret minimization, demonstrating that the optimal strategy depends critically on the tail behavior of the reward distribution. We show theoretically that light-tailed regimes favor optimism to unearth high-quality outliers, whereas heavy-tailed regimes require pessimism to guard against reward mis-calibration in the extremes. Guided by this insight, we introduce Best-of-Tails (BoT), an adaptive inference-time alignment framework that uses Tsallis divergence as a tunable regularizer to provide a finer granularity of interpolation between these extremes. BoT uses the Hill estimator to characterize reward-tail heaviness on a per-prompt basis and dynamically adjusts its selection rule to balance exploration gains against alignment error. Across math, multiple-choice reasoning, and human-preference evaluations, BoT improves alignment performance across a range of reference and reward model configurations relative to fixed-strategy baselines. 

---
# MultiGen: Level-Design for Editable Multiplayer Worlds in Diffusion Game Engines 

**Authors**: Ryan Po, David Junhao Zhang, Amir Hertz, Gordon Wetzstein, Neal Wadhwa, Nataniel Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2603.06679)  

**Abstract**: Video world models have shown immense promise for interactive simulation and entertainment, but current systems still struggle with two important aspects of interactivity: user control over the environment for reproducible, editable experiences, and shared inference where players hold influence over a common world. To address these limitations, we introduce an explicit external memory into the system, a persistent state operating independent of the model's context window, that is continually updated by user actions and queried throughout the generation roll-out. Unlike conventional diffusion game engines that operate as next-frame predictors, our approach decomposes generation into Memory, Observation, and Dynamics modules. This design gives users direct, editable control over environment structure via an editable memory representation, and it naturally extends to real-time multiplayer rollouts with coherent viewpoints and consistent cross-player interactions. 

---
# Scaling Strategy, Not Compute: A Stand-Alone, Open-Source StarCraft II Benchmark for Accessible Reinforcement Learning Research 

**Authors**: Sourav Panda, Shreyash Kale, Tanmay Ambadkar, Abhinav Verma, Jonathan Dodge  

**Link**: [PDF](https://arxiv.org/pdf/2603.06608)  

**Abstract**: The research community lacks a middle ground between StarCraft IIs full game and its mini-games. The full-games sprawling state-action space renders reward signals sparse and noisy, but in mini-games simple agents saturate performance. This complexity gap hinders steady curriculum design and prevents researchers from experimenting with modern Reinforcement Learning algorithms in RTS environments under realistic compute budgets. To fill this gap, we present the Two-Bridge Map Suite, the first entry in an open-source benchmark series we purposely engineered as an intermediate benchmark to sit between these extremes. By disabling economy mechanics such as resource collection, base building, and fog-of-war, the environment isolates two core tactical skills: long-range navigation and micro-combat. Preliminary experiments show that agents learn coherent maneuvering and engagement behaviors without imposing full-game computational costs. Two-Bridge is released as a lightweight, Gym-compatible wrapper on top of PySC2, with maps, wrappers, and reference scripts fully open-sourced to encourage broad adoption as a standard benchmark. 

---
# Autonomous AI Agents for Option Hedging: Enhancing Financial Stability through Shortfall Aware Reinforcement Learning 

**Authors**: Minxuan Hu, Ziheng Chen, Jiayu Yi, Wenxi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.06587)  

**Abstract**: The deployment of autonomous AI agents in derivatives markets has widened a practical gap between static model calibration and realized hedging outcomes. We introduce two reinforcement learning frameworks, a novel Replication Learning of Option Pricing (RLOP) approach and an adaptive extension of Q-learner in Black-Scholes (QLBS), that prioritize shortfall probability and align learning objectives with downside sensitive hedging. Using listed SPY and XOP options, we evaluate models using realized path delta hedging outcome distributions, shortfall probability, and tail risk measures such as Expected Shortfall. Empirically, RLOP reduces shortfall frequency in most slices and shows the clearest tail-risk improvements in stress, while implied volatility fit often favors parametric models yet poorly predicts after-cost hedging performance. This friction-aware RL framework supports a practical approach to autonomous derivatives risk management as AI-augmented trading systems scale. 

---
# Scale Space Diffusion 

**Authors**: Soumik Mukhopadhyay, Prateksha Udhayanan, Abhinav Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2603.08709)  

**Abstract**: Diffusion models degrade images through noise, and reversing this process reveals an information hierarchy across timesteps. Scale-space theory exhibits a similar hierarchy via low-pass filtering. We formalize this connection and show that highly noisy diffusion states contain no more information than small, downsampled images - raising the question of why they must be processed at full resolution. To address this, we fuse scale spaces into the diffusion process by formulating a family of diffusion models with generalized linear degradations and practical implementations. Using downsampling as the degradation yields our proposed Scale Space Diffusion. To support Scale Space Diffusion, we introduce Flexi-UNet, a UNet variant that performs resolution-preserving and resolution-increasing denoising using only the necessary parts of the network. We evaluate our framework on CelebA and ImageNet and analyze its scaling behavior across resolutions and network depths. Our project website ( this https URL ) is available publicly. 

---
# Split Federated Learning Architectures for High-Accuracy and Low-Delay Model Training 

**Authors**: Yiannis Papageorgiou, Yannis Thomas, Ramin Khalili, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2603.08687)  

**Abstract**: Can we find a network architecture for ML model training so as to optimize training loss (and thus, accuracy) in Split Federated Learning (SFL)? And can this architecture also reduce training delay and communication overhead? While accuracy is not influenced by how we split the model in ordinary, state-of-the-art SFL, in this work we answer the questions above in the affirmative. Recent Hierarchical SFL (HSFL) architectures adopt a three-tier training structure consisting of clients, (local) aggregators, and a central server. In this architecture, the model is partitioned at two partitioning layers into three sub-models, which are executed across the three tiers. Despite their merits, HSFL architectures overlook the impact of the partitioning layers and client-to-aggregator assignments on accuracy, delay, and overhead. This work explicitly captures the impact of the partitioning layers and client-to-aggregator assignments on accuracy, delay and overhead by formulating a joint optimization problem. We prove that the problem is NP-hard and propose the first accuracy-aware heuristic algorithm that explicitly accounts for model accuracy, while remaining delay-efficient. Simulation results on public datasets show that our approach can improve accuracy by 3%, while reducing delay by 20% and overhead by 50%, compared to state-of-the-art SFL and HSFL schemes. 

---
# Benchmarking Language Modeling for Lossless Compression of Full-Fidelity Audio 

**Authors**: Phillip Long, Zachary Novack, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2603.08683)  

**Abstract**: Autoregressive "language" models (LMs) trained on raw waveforms can be repurposed for lossless audio compression, but prior work is limited to 8-bit audio, leaving open whether such approaches work for practical settings (16/24-bit) and can compete with existing codecs. We benchmark LM-based compression on full-fidelity audio across diverse domains (music, speech, bioacoustics), sampling rates (16kHz-48kHz), and bit depths (8, 16, 24-bit). Standard sample-level tokenization becomes intractable at higher bit depths due to vocabulary size (65K for 16-bit; 16.7M for 24-bit). We propose Trilobyte, a byte-level tokenization schema for full resolution audio, improving vocabulary scaling from $O(2^{b})$ to $O(1)$ and enabling the first tractable 24-bit LM-based lossless compression. While LMs consistently outperform FLAC and yield state-of-the-art compression at 8-bit and 16-bit, we observe that compression gains become more modest as bit depth increases beyond 8-bit. 

---
# A New Lower Bound for the Random Offerer Mechanism in Bilateral Trade using AI-Guided Evolutionary Search 

**Authors**: Yang Cai, Vineet Gupta, Zun Li, Aranyak Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2603.08679)  

**Abstract**: The celebrated Myerson--Satterthwaite theorem shows that in bilateral trade, no mechanism can be simultaneously fully efficient, Bayesian incentive compatible (BIC), and budget balanced (BB). This naturally raises the question of how closely the gains from trade (GFT) achievable by a BIC and BB mechanism can approximate the first-best (fully efficient) benchmark. The optimal BIC and BB mechanism is typically complex and highly distribution-dependent, making it difficult to characterize directly. Consequently, much of the literature analyzes simpler mechanisms such as the Random-Offerer (RO) mechanism and establishes constant-factor guarantees relative to the first-best GFT. An important open question concerns the worst-case performance of the RO mechanism relative to first-best (FB) efficiency. While it was originally hypothesized that the approximation ratio $\frac{\text{GFT}_{\text{FB}}}{\text{GFT}_{\text{RO}}}$ is bounded by $2$, recent work provided counterexamples to this conjecture: Cai et al. proved that the ratio can be strictly larger than $2$, and Babaioff et al. exhibited an explicit example with ratio approximately $2.02$.
In this work, we employ AlphaEvolve, an AI-guided evolutionary search framework, to explore the space of value distributions. We identify a new worst-case instance that yields an improved lower bound of $\frac{\text{GFT}_{\text{FB}}}{\text{GFT}_{\text{RO}}} \ge \textbf{2.0749}$. This establishes a new lower bound on the worst-case performance of the Random-Offerer mechanism, demonstrating a wider efficiency gap than previously known. 

---
# PostTrainBench: Can LLM Agents Automate LLM Post-Training? 

**Authors**: Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina Nguyen, Matthias Bethge, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.08640)  

**Abstract**: AI agents have become surprisingly proficient at software engineering over the past year, largely due to improvements in reasoning capabilities. This raises a deeper question: can these systems extend their capabilities to automate AI research itself? In this paper, we explore post-training, the critical phase that turns base LLMs into useful assistants. We introduce PostTrainBench to benchmark how well LLM agents can perform post-training autonomously under bounded compute constraints (10 hours on one H100 GPU). We ask frontier agents (e.g., Claude Code with Opus 4.6) to optimize the performance of a base LLM on a particular benchmark (e.g., Qwen3-4B on AIME). Importantly, we do not provide any predefined strategies to the agents and instead give them full autonomy to find necessary information on the web, run experiments, and curate data. We find that frontier agents make substantial progress but generally lag behind instruction-tuned LLMs from leading providers: 23.2% for the best agent vs. 51.1% for official instruction-tuned models. However, agents can exceed instruction-tuned models in targeted scenarios: GPT-5.1 Codex Max achieves 89% on BFCL with Gemma-3-4B vs. 67% for the official model. We also observe several failure modes worth flagging. Agents sometimes engage in reward hacking: training on the test set, downloading existing instruction-tuned checkpoints instead of training their own, and using API keys they find to generate synthetic data without authorization. These behaviors are concerning and highlight the importance of careful sandboxing as these systems become more capable. Overall, we hope PostTrainBench will be useful for tracking progress in AI R&D automation and for studying the risks that come with it. Website and code are available at this https URL. 

---
# UNBOX: Unveiling Black-box visual models with Natural-language 

**Authors**: Simone Carnemolla, Chiara Russo, Simone Palazzo, Quentin Bouniot, Daniela Giordano, Zeynep Akata, Matteo Pennisi, Concetto Spampinato  

**Link**: [PDF](https://arxiv.org/pdf/2603.08639)  

**Abstract**: Ensuring trustworthiness in open-world visual recognition requires models that are interpretable, fair, and robust to distribution shifts. Yet modern vision systems are increasingly deployed as proprietary black-box APIs, exposing only output probabilities and hiding architecture, parameters, gradients, and training data. This opacity prevents meaningful auditing, bias detection, and failure analysis. Existing explanation methods assume white- or gray-box access or knowledge of the training distribution, making them unusable in these real-world settings. We introduce UNBOX, a framework for class-wise model dissection under fully data-free, gradient-free, and backpropagation-free constraints. UNBOX leverages Large Language Models and text-to-image diffusion models to recast activation maximization as a purely semantic search driven by output probabilities. The method produces human-interpretable text descriptors that maximally activate each class, revealing the concepts a model has implicitly learned, the training distribution it reflects, and potential sources of bias. We evaluate UNBOX on ImageNet-1K, Waterbirds, and CelebA through semantic fidelity tests, visual-feature correlation analyses and slice-discovery auditing. Despite operating under the strictest black-box constraints, UNBOX performs competitively with state-of-the-art white-box interpretability methods. This demonstrates that meaningful insight into a model's internal reasoning can be recovered without any internal access, enabling more trustworthy and accountable visual recognition systems. 

---
# Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation 

**Authors**: Hikmat Khan, Wei Chen, Muhammad Khalid Khan Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08605)  

**Abstract**: Background and objectives: Colorectal cancer histopathological grading depends on accurate segmentation of glandular structures. Current deep learning approaches rely on large scale pixel level annotations that are labor intensive and difficult to obtain in routine clinical practice. Weakly supervised semantic segmentation offers a promising alternative. However, class activation map based methods often produce incomplete pseudo masks that emphasize highly discriminative regions and fail to supervise unannotated glandular structures. We propose a weakly supervised teacher student framework that leverages sparse pathologist annotations and an Exponential Moving Average stabilized teacher network to generate refined pseudo masks.
Methods: The framework integrates confidence based filtering, adaptive fusion of teacher predictions with limited ground truth, and curriculum guided refinement to progressively segment unannotated glandular regions. The method was evaluated on an institutional colorectal cancer cohort from The Ohio State University Wexner Medical Center consisting of 60 hematoxylin and eosin stained whole slide images and on public datasets including the Gland Segmentation dataset, TCGA COAD, TCGA READ, and SPIDER.
Results: On the Gland Segmentation dataset the framework achieved a mean Intersection over Union of 80.10 and a mean Dice coefficient of 89.10. Cross cohort evaluation demonstrated robust generalization on TCGA COAD and TCGA READ without additional annotations, while reduced performance on SPIDER reflected domain shift.
Conclusions: The proposed framework provides an annotation efficient and generalizable approach for gland segmentation in colorectal histopathology. 

---
# Don't Look Back in Anger: MAGIC Net for Streaming Continual Learning with Temporal Dependence 

**Authors**: Federico Giannini, Sandro D'Andrea, Emanuele Della Valle  

**Link**: [PDF](https://arxiv.org/pdf/2603.08600)  

**Abstract**: Concept drift, temporal dependence, and catastrophic forgetting represent major challenges when learning from data streams. While Streaming Machine Learning and Continual Learning (CL) address these issues separately, recent efforts in Streaming Continual Learning (SCL) aim to unify them. In this work, we introduce MAGIC Net, a novel SCL approach that integrates CL-inspired architectural strategies with recurrent neural networks to tame temporal dependence. MAGIC Net continuously learns, looks back at past knowledge by applying learnable masks over frozen weights, and expands its architecture when necessary. It performs all operations online, ensuring inference availability at all times. Experiments on synthetic and real-world streams show that it improves adaptation to new concepts, limits memory usage, and mitigates forgetting. 

---
# Towards Batch-to-Streaming Deep Reinforcement Learning for Continuous Control 

**Authors**: Riccardo De Monte, Matteo Cederle, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2603.08588)  

**Abstract**: State-of-the-art deep reinforcement learning (RL) methods have achieved remarkable performance in continuous control tasks, yet their computational complexity is often incompatible with the constraints of resource-limited hardware, due to their reliance on replay buffers, batch updates, and target networks. The emerging paradigm of streaming deep RL addresses this limitation through purely online updates, achieving strong empirical performance on standard benchmarks. In this work, we propose two novel streaming deep RL algorithms, Streaming Soft Actor-Critic (S2AC) and Streaming Deterministic Actor-Critic (SDAC), explicitly designed to be compatible with state-of-the-art batch RL methods, making them particularly suitable for on-device finetuning applications such as Sim2Real transfer. Both algorithms achieve performance comparable to state-of-the-art streaming baselines on standard benchmarks without requiring tedious hyperparameter tuning. Finally, we further investigate the practical challenges of transitioning from batch to streaming learning during finetuning and propose concrete strategies to tackle them. 

---
# MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation 

**Authors**: Yutong Shen, Hangxu Liu, Penghui Liu, Jiashuo Luo, Yongkang Zhang, Rex Morvley, Chen Jiang, Jianwei Zhang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08572)  

**Abstract**: Learning natural, stable, and compositionally generalizable whole-body control policies for humanoid robots performing simultaneous locomotion and manipulation (loco-manipulation) remains a fundamental challenge in robotics. Existing reinforcement learning approaches typically rely on a single monolithic policy to acquire multiple skills, which often leads to cross-skill gradient interference and motion pattern conflicts in high-degree-of-freedom systems. As a result, generated behaviors frequently exhibit unnatural movements, limited stability, and poor generalization to complex task compositions. To address these limitations, we propose MetaWorld-X, a hierarchical world model framework for humanoid control. Guided by a divide-and-conquer principle, our method decomposes complex control problems into a set of specialized expert policies (Specialized Expert Policies, SEP). Each expert is trained under human motion priors through imitation-constrained reinforcement learning, introducing biomechanically consistent inductive biases that ensure natural and physically plausible motion generation. Building upon this foundation, we further develop an Intelligent Routing Mechanism (IRM) supervised by a Vision-Language Model (VLM), enabling semantic-driven expert composition. The VLM-guided router dynamically integrates expert policies according to high-level task semantics, facilitating compositional generalization and adaptive execution in multi-stage loco-manipulation tasks. 

---
# OSS-CRS: Liberating AIxCC Cyber Reasoning Systems for Real-World Open-Source Security 

**Authors**: Andrew Chin, Dongkwan Kim, Yu-Fu Fu, Fabian Fleischer, Youngjoon Kim, HyungSeok Han, Cen Zhang, Brian Junekyu Lee, Hanqing Zhao, Taesoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.08566)  

**Abstract**: DARPA's AI Cyber Challenge (AIxCC) showed that cyber reasoning systems (CRSs) can go beyond vulnerability discovery to autonomously confirm and patch bugs: seven teams built such systems and open-sourced them after the competition. Yet all seven open-sourced CRSs remain largely unusable outside their original teams, each bound to the competition cloud infrastructure that no longer exists. We present OSS-CRS, an open, locally deployable framework for running and combining CRS techniques against real-world open-source projects, with budget-aware resource management. We ported the first-place system (Atlantis) and discovered 10 previously unknown bugs (three of high severity) across 8 OSS-Fuzz projects. OSS-CRS is publicly available. 

---
# Towards Effective and Efficient Graph Alignment without Supervision 

**Authors**: Songyang Chen, Youfang Lin, Yu Liu, Shuai Zheng, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.08526)  

**Abstract**: Unsupervised graph alignment aims to find the node correspondence across different graphs without any anchor node pairs. Despite the recent efforts utilizing deep learning-based techniques, such as the embedding and optimal transport (OT)-based approaches, we observe their limitations in terms of model accuracy-efficiency tradeoff. By focusing on the exploitation of local and global graph information, we formalize them as the ``local representation, global alignment'' paradigm, and present a new ``global representation and alignment'' paradigm to resolve the mismatch between the two phases in the alignment process. We then propose \underline{Gl}obal representation and \underline{o}ptimal transport-\underline{b}ased \underline{Align}ment (\texttt{GlobAlign}), and its variant, \texttt{GlobAlign-E}, for better \underline{E}fficiency. Our methods are equipped with the global attention mechanism and a hierarchical cross-graph transport cost, able to capture long-range and implicit node dependencies beyond the local graph structure. Furthermore, \texttt{GlobAlign-E} successfully closes the time complexity gap between representative embedding and OT-based methods, reducing OT's cubic complexity to quadratic terms. Through extensive experiments, our methods demonstrate superior performance, with up to a 20\% accuracy improvement over the best competitor. Meanwhile, \texttt{GlobAlign-E} achieves the best efficiency, with an order of magnitude speedup against existing OT-based methods. 

---
# Beyond Hungarian: Match-Free Supervision for End-to-End Object Detection 

**Authors**: Shoumeng Qiu, Xinrun Li, Yang Long  

**Link**: [PDF](https://arxiv.org/pdf/2603.08514)  

**Abstract**: Recent DEtection TRansformer (DETR) based frameworks have achieved remarkable success in end-to-end object detection. However, the reliance on the Hungarian algorithm for bipartite matching between queries and ground truths introduces computational overhead and complicates the training dynamics. In this paper, we propose a novel matching-free training scheme for DETR-based detectors that eliminates the need for explicit heuristic matching. At the core of our approach is a dedicated Cross-Attention-based Query Selection (CAQS) module. Instead of discrete assignment, we utilize encoded ground-truth information to probe the decoder queries through a cross-attention mechanism. By minimizing the weighted error between the queried results and the ground truths, the model autonomously learns the implicit correspondences between object queries and specific targets. This learned relationship further provides supervision signals for the learning of queries. Experimental results demonstrate that our proposed method bypasses the traditional matching process, significantly enhancing training efficiency, reducing the matching latency by over 50\%, effectively eliminating the discrete matching bottleneck through differentiable correspondence learning, and also achieving superior performance compared to existing state-of-the-art methods. 

---
# Oracle-Guided Soft Shielding for Safe Move Prediction in Chess 

**Authors**: Prajit T Rajendran, Fabio Arnez, Huascar Espinoza, Agnes Delaborde, Chokri Mraidha  

**Link**: [PDF](https://arxiv.org/pdf/2603.08506)  

**Abstract**: In high stakes environments, agents relying purely on imitation learning or reinforcement learning often struggle to avoid safety-critical errors during exploration. Existing reinforcement learning approaches for environments such as chess require hundreds of thousands of episodes and substantial computational resources to converge. Imitation learning, on the other hand, is more sample efficient but is brittle under distributional shift and lacks mechanisms for proactive risk avoidance. In this work, we propose Oracle-Guided Soft Shielding (OGSS), a simple yet effective framework for safer decision-making, enabling safe exploration by learning a probabilistic safety model from oracle feedback in an imitation learning setting. Focusing on the domain of chess, we train a model to predict strong moves based on past games, and separately learn a blunder prediction model from Stockfish evaluations to estimate the tactical risk of each move. During inference, the agent first generates a set of candidate moves and then uses the blunder model to determine high-risk options, and uses a utility function combining the predicted move likelihood from the policy model and the blunder probability to select actions that strike a balance between performance and safety. This enables the agent to explore and play competitively while significantly reducing the chance of tactical mistakes. Across hundreds of games against a strong chess engine, we compare our approach with other methods in the literature, such as action pruning, SafeDAgger, and uncertainty-based sampling. Our results demonstrate that OGSS variants maintain a lower blunder rate even as the agent's exploration ratio is increased by several folds, highlighting its ability to support broader exploration without compromising tactical soundness. 

---
# Echo2ECG: Enhancing ECG Representations with Cardiac Morphology from Multi-View Echos 

**Authors**: Michelle Espranita Liman, Özgün Turgut, Alexander Müller, Eimo Martens, Daniel Rueckert, Philip Müller  

**Link**: [PDF](https://arxiv.org/pdf/2603.08505)  

**Abstract**: Electrocardiography (ECG) is a low-cost, widely used modality for diagnosing electrical abnormalities like atrial fibrillation by capturing the heart's electrical activity. However, it cannot directly measure cardiac morphological phenotypes, such as left ventricular ejection fraction (LVEF), which typically require echocardiography (Echo). Predicting these phenotypes from ECG would enable early, accessible health screening. Existing self-supervised methods suffer from a representational mismatch by aligning ECGs to single-view Echos, which only capture local, spatially restricted anatomical snapshots. To address this, we propose Echo2ECG, a multimodal self-supervised learning framework that enriches ECG representations with the heart's morphological structure captured in multi-view Echos. We evaluate Echo2ECG as an ECG feature extractor on two clinically relevant tasks that fundamentally require morphological information: (1) classification of structural cardiac phenotypes across three datasets, and (2) retrieval of Echo studies with similar morphological characteristics using ECG queries. Our extracted ECG representations consistently outperform those of state-of-the-art unimodal and multimodal baselines across both tasks, despite being 18x smaller than the largest baseline. These results demonstrate that Echo2ECG is a robust, powerful ECG feature extractor. Our code is accessible at this https URL. 

---
# First-Order Geometry, Spectral Compression, and Structural Compatibility under Bounded Computation 

**Authors**: Changkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.08494)  

**Abstract**: Optimization under structural constraints is typically analyzed through projection or penalty methods, obscuring the geometric mechanism by which constraints shape admissible dynamics. We propose an operator-theoretic formulation in which computational or feasibility limitations are encoded by self-adjoint operators defining locally reachable subspaces. In this setting, the optimal first-order improvement direction emerges as a pseudoinverse-weighted gradient, revealing how constraints induce a distorted ascent geometry. We further demonstrate that effective dynamics concentrate along dominant spectral modes, yielding a principled notion of spectral compression, and establish a compatibility principle that characterizes the existence of common admissible directions across multiple objectives. The resulting framework unifies gradient projection, spectral truncation, and multi-objective feasibility within a single geometric structure. 

---
# Visual Self-Fulfilling Alignment: Shaping Safety-Oriented Personas via Threat-Related Images 

**Authors**: Qishun Yang, Shu Yang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08486)  

**Abstract**: Multimodal large language models (MLLMs) face safety misalignment, where visual inputs enable harmful outputs. To address this, existing methods require explicit safety labels or contrastive data; yet, threat-related concepts are concrete and visually depictable, while safety concepts, like helpfulness, are abstract and lack visual referents. Inspired by the Self-Fulfilling mechanism underlying emergent misalignment, we propose Visual Self-Fulfilling Alignment (VSFA). VSFA fine-tunes vision-language models (VLMs) on neutral VQA tasks constructed around threat-related images, without any safety labels. Through repeated exposure to threat-related visual content, models internalize the implicit semantics of vigilance and caution, shaping safety-oriented personas. Experiments across multiple VLMs and safety benchmarks demonstrate that VSFA reduces the attack success rate, improves response quality, and mitigates over-refusal while preserving general capabilities. Our work extends the self-fulfilling mechanism from text to visual modalities, offering a label-free approach to VLMs alignment. 

---
# X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection 

**Authors**: Youngseo Kim, Kwan Yun, Seokhyeon Hong, Sihun Cha, Colette Suhjung Koo, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2603.08483)  

**Abstract**: The surge of highly realistic synthetic videos produced by contemporary generative systems has significantly increased the risk of malicious use, challenging both humans and existing detectors. Against this backdrop, we take a generator-side view and observe that internal cross-attention mechanisms in these models encode fine-grained speech-motion alignment, offering useful correspondence cues for forgery detection. Building on this insight, we propose X-AVDT, a robust and generalizable deepfake detector that probes generator-internal audio-visual signals accessed via DDIM inversion to expose these cues. X-AVDT extracts two complementary signals: (i) a video composite capturing inversion-induced discrepancies, and (ii) an audio-visual cross-attention feature reflecting modality alignment enforced during generation. To enable faithful cross-generator evaluation, we further introduce MMDF, a new multimodal deepfake dataset spanning diverse manipulation types and rapidly evolving synthesis paradigms, including GANs, diffusion, and flow-matching. Extensive experiments demonstrate that X-AVDT achieves leading performance on MMDF and generalizes strongly to external benchmarks and unseen generators, outperforming existing methods with accuracy improved by 13.1%. Our findings highlight the importance of leveraging internal audio-visual consistency cues for robustness to future generators in deepfake detection. 

---
# R2F: Repurposing Ray Frontiers for LLM-free Object Navigation 

**Authors**: Francesco Argenziano, John Mark Alexis Marcelo, Michele Brienza, Abdel Hakim Drid, Emanuele Musumeci, Daniele Nardi, Domenico D. Bloisi, Vincenzo Suriani  

**Link**: [PDF](https://arxiv.org/pdf/2603.08475)  

**Abstract**: Zero-shot open-vocabulary object navigation has progressed rapidly with the emergence of large Vision-Language Models (VLMs) and Large Language Models (LLMs), now widely used as high-level decision-makers instead of end-to-end policies. Although effective, such systems often rely on iterative large-model queries at inference time, introducing latency and computational overhead that limit real-time deployment. To address this problem, we repurpose ray frontiers (R2F), a recently proposed frontier-based exploration paradigm, to develop an LLM-free framework for indoor open-vocabulary object navigation. While ray frontiers were originally used to bias exploration using semantic cues carried along rays, we reinterpret frontier regions as explicit, direction-conditioned semantic hypotheses that serve as navigation goals. Language-aligned features accumulated along out-of-range rays are stored sparsely at frontiers, where each region maintains multiple directional embeddings encoding plausible unseen content. In this way, navigation then reduces to embedding-based frontier scoring and goal tracking within a classical mapping and planning pipeline, eliminating iterative large-model reasoning. We further introduce R2F-VLN, a lightweight extension for free-form language instructions using syntactic parsing and relational verification without additional VLM or LLM components. Experiments in Habitat-sim and on a real robotic platform demonstrate competitive state-of-the-art zero-shot performance with real-time execution, achieving up to 6 times faster runtime than VLM-based alternatives. 

---
# LycheeCluster: Efficient Long-Context Inference with Structure-Aware Chunking and Hierarchical KV Indexing 

**Authors**: Dongfang Li, Zixuan Liu, Gang Lin, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08453)  

**Abstract**: The quadratic complexity of the attention mechanism and the substantial memory footprint of the Key-Value (KV) cache present severe computational and memory challenges for Large Language Models (LLMs) processing long contexts. Existing retrieval-based methods often compromise semantic integrity through fixed-size chunking and suffer from inefficient linear scanning. In this paper, we propose LycheeCluster, a novel method for efficient KV cache management. LycheeCluster preserves local semantic coherence via boundary-aware chunking and constructs a recursive hierarchical index rooted in the triangle inequality. This design transforms cache retrieval from a linear scan into a theoretically bounded, logarithmic-time pruning process, while a lazy update strategy supports efficient streaming generation. Experiments demonstrate that LycheeCluster achieves up to a 3.6x end-to-end inference speedup with negligible degradation in model performance, outperforming state-of-the-art KV cache management methods (e.g., Quest, ClusterKV). We will release our code and kernels after publication. 

---
# A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic 

**Authors**: Peter Brodeur, Jacob M. Koshy, Anil Palepu, Khaled Saab, Ava Homiar, Roma Ruparel, Charles Wu, Ryutaro Tanno, Joseph Xu, Amy Wang, David Stutz, Hannah M. Ferrera, David Barrett, Lindsey Crowley, Jihyeon Lee, Spencer E. Rittner, Ellery Wulczyn, Selena K. Zhang, Elahe Vedadi, Christine G. Kohn, Kavita Kulkarni, Vinay Kadiyala, Sara Mahdavi, Wendy Du, Jessica Williams, David Feinbloom, Renee Wong, Tao Tu, Petar Sirkovic, Alessio Orlandi, Christopher Semturs, Yun Liu, Juraj Gottweis, Dale R. Webster, Joëlle Barral, Katherine Chou, Pushmeet Kohli, Avinatan Hassidim, Yossi Matias, James Manyika, Rob Fields, Jonathan X. Li, Marc L. Cohen, Vivek Natarajan, Mike Schaekermann, Alan Karthikesalingam, Adam Rodman  

**Link**: [PDF](https://arxiv.org/pdf/2603.08448)  

**Abstract**: Large language model (LLM)-based AI systems have shown promise for patient-facing diagnostic and management conversations in simulated settings. Translating these systems into clinical practice requires assessment in real-world workflows with rigorous safety oversight. We report a prospective, single-arm feasibility study of an LLM-based conversational AI, the Articulate Medical Intelligence Explorer (AMIE), conducting clinical history taking and presentation of potential diagnoses for patients to discuss with their provider at urgent care appointments at a leading academic medical center. 100 adult patients completed an AMIE text-chat interaction up to 5 days before their appointment. We sought to assess the conversational safety and quality, patient and clinician experience, and clinical reasoning capabilities compared to primary care providers (PCPs). Human safety supervisors monitored all patient-AMIE interactions in real time and did not need to intervene to stop any consultations based on pre-defined criteria. Patients reported high satisfaction and their attitudes towards AI improved after interacting with AMIE (p < 0.001). PCPs found AMIE's output useful with a positive impact on preparedness. AMIE's differential diagnosis (DDx) included the final diagnosis, per chart review 8 weeks post-encounter, in 90% of cases, with 75% top-3 accuracy. Blinded assessment of AMIE and PCP DDx and management (Mx) plans suggested similar overall DDx and Mx plan quality, without significant differences for DDx (p = 0.6) and appropriateness and safety of Mx (p = 0.1 and 1.0, respectively). PCPs outperformed AMIE in the practicality (p = 0.003) and cost effectiveness (p = 0.004) of Mx. While further research is needed, this study demonstrates the initial feasibility, safety, and user acceptance of conversational AI in a real-world setting, representing crucial steps towards clinical translation. 

---
# One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States 

**Authors**: Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08429)  

**Abstract**: LLM agents that retrieve external knowledge typically generate a search query as text, then run a separate embedding model to encode it into a vector. This two-model pipeline adds infrastructure complexity and latency, yet is redundant: the LLM already encodes the full conversational context in its hidden states. We propose equipping LLM agents with native retrieval capability by adding a lightweight projection head that maps hidden states directly into the embedding space, eliminating the need for a separate embedding model. Trained with a combination of alignment, contrastive, and rank distillation losses, our method retains 97\% of baseline retrieval quality while enabling the LLM agent to search with its own representations. Experiments on the QReCC conversational search benchmark show competitive Recall@10 and MRR@10 compared to the standard generate-then-encode pipeline, with systematic ablations confirming the contribution of each loss component. 

---
# SYNAPSE: Framework for Neuron Analysis and Perturbation in Sequence Encoding 

**Authors**: Jesús Sánchez Ochoa, Enrique Tomás Martínez Beltrán, Alberto Huertas Celdrán  

**Link**: [PDF](https://arxiv.org/pdf/2603.08424)  

**Abstract**: In recent years, Artificial Intelligence has become a powerful partner for complex tasks such as data analysis, prediction, and problem-solving, yet its lack of transparency raises concerns about its reliability. In sensitive domains such as healthcare or cybersecurity, ensuring transparency, trustworthiness, and robustness is essential, since the consequences of wrong decisions or successful attacks can be severe. Prior neuron-level interpretability approaches are primarily descriptive, task-dependent, or require retraining, which limits their use as systematic, reusable tools for evaluating internal robustness across architectures and domains. To overcome these limitations, this work proposes SYNAPSE, a systematic, training-free framework for understanding and stress-testing the internal behavior of Transformer models across domains. It extracts per-layer [CLS] representations, trains a lightweight linear probe to obtain global and per-class neuron rankings, and applies forward-hook interventions during inference. This design enables controlled experiments on internal representations without altering the original model, thereby allowing weaknesses, stability patterns, and label-specific sensitivities to be measured and compared directly across tasks and architectures. Across all experiments, SYNAPSE reveals a consistent, domain-independent organization of internal representations, in which task-relevant information is encoded in broad, overlapping neuron subsets. This redundancy provides a strong degree of functional stability, while class-wise asymmetries expose heterogeneous specialization patterns and enable label-aware analysis. In contrast, small structured manipulations in weight or logit space are sufficient to redirect predictions, highlighting complementary vulnerability profiles and illustrating how SYNAPSE can guide the development of more robust Transformer models. 

---
# Human-Aware Robot Behaviour in Self-Driving Labs 

**Authors**: Satheeshkumar Veeramani, Anna Kisil, Abigail Bentley, Hatem Fakhruldeen, Gabriella Pizzuto, Andrew I. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2603.08420)  

**Abstract**: Self-driving laboratories (SDLs) are rapidly transforming research in chemistry and materials science to accelerate new discoveries. Mobile robot chemists (MRCs) play a pivotal role by autonomously navigating the lab to transport samples, effectively connecting synthesis, analysis, and characterisation equipment. The instruments within an SDL are typically designed or retrofitted to be accessed by both human and robotic chemists, ensuring operational flexibility and integration between manual and automated workflows. In many scenarios, human and robotic chemists may need to use the same equipment simultaneously. Currently, MRCs rely on simple LiDAR-based obstruction detection, which forces the robot to passively wait if a human is present. This lack of situational awareness leads to unnecessary delays and inefficient coordination in time-critical automated workflows in human-robot shared labs. To address this, we present an initial study of an embodied, AI-driven perception method that facilitates proactive human-robot interaction in shared-access scenarios. Our method features a hierarchical human intention prediction model that allows the robot to distinguish between preparatory actions (waiting) and transient interactions (accessing the instrument). Our results demonstrate that the proposed approach enhances efficiency by enabling proactive human-robot interaction, streamlining coordination, and potentially increasing the efficiency of autonomous scientific labs. 

---
# Geometrically Constrained Outlier Synthesis 

**Authors**: Daniil Karzanov, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2603.08413)  

**Abstract**: Deep neural networks for image classification often exhibit overconfidence on out-of-distribution (OOD) samples. To address this, we introduce Geometrically Constrained Outlier Synthesis (GCOS), a training-time regularization framework aimed at improving OOD robustness during inference. GCOS addresses a limitation of prior synthesis methods by generating virtual outliers in the hidden feature space that respect the learned manifold structure of in-distribution (ID) data. The synthesis proceeds in two stages: (i) a dominant-variance subspace extracted from the training features identifies geometrically informed, off-manifold directions; (ii) a conformally-inspired shell, defined by the empirical quantiles of a nonconformity score from a calibration set, adaptively controls the synthesis magnitude to produce boundary samples. The shell ensures that generated outliers are neither trivially detectable nor indistinguishable from in-distribution data, facilitating smoother learning of robust features. This is combined with a contrastive regularization objective that promotes separability of ID and OOD samples in a chosen score space, such as Mahalanobis or energy-based. Experiments demonstrate that GCOS outperforms state-of-the-art methods using standard energy-based inference on near-OOD benchmarks, defined as tasks where outliers share the same semantic domain as in-distribution data. As an exploratory extension, the framework naturally transitions to conformal OOD inference, which translates uncertainty scores into statistically valid p-values and enables thresholds with formal error guarantees, providing a pathway toward more predictable and reliable OOD detection. 

---
# Aligning to Illusions: Choice Blindness in Human and AI Feedback 

**Authors**: Wenbin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08412)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) assumes annotator preferences reflect stable internal states. We challenge this through three experiments spanning the preference pipeline. In a human choice blindness study, 91% of surreptitiously swapped preferences go undetected, extending choice blindness to third-person evaluative comparison of unfamiliar text. Testing fifteen LLM judges as potential replacements, we find detection relies on shallow text matching rather than genuine self-monitoring: removing prior reasoning from context causes blindness to surge from near-zero to over 50%, while explicit social pressure induces near-universal compliance. In a dose-response experiment across two architectures from 86M to 2B parameters, one-sixth to one-third of labels must be corrupted before the reward signal halves, yet standard pairwise accuracy remains virtually unchanged. A Best-of-N evaluation confirms this translates to downstream policy degradation: at 50% corruption, reward-guided selection produces no improvement over random sampling, while the proxy model reports monotonically increasing scores. Together, these results reveal a preference construction problem: the signal entering RLHF is shaped by elicitation context in ways that neither human metacognition, LLM self-monitoring, nor standard evaluation metrics can detect. 

---
# A Recipe for Stable Offline Multi-agent Reinforcement Learning 

**Authors**: Dongsu Lee, Daehee Lee, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08399)  

**Abstract**: Despite remarkable achievements in single-agent offline reinforcement learning (RL), multi-agent RL (MARL) has struggled to adopt this paradigm, largely persisting with on-policy training and self-play from scratch. One reason for this gap comes from the instability of non-linear value decomposition, leading prior works to avoid complex mixing networks in favor of linear value decomposition (e.g., VDN) with value regularization used in single-agent setups. In this work, we analyze the source of instability in non-linear value decomposition within the offline MARL setting. Our observations confirm that they induce value-scale amplification and unstable optimization. To alleviate this, we propose a simple technique, scale-invariant value normalization (SVN), that stabilizes actor-critic training without altering the Bellman fixed point. Empirically, we examine the interaction among key components of offline MARL (e.g., value decomposition, value learning, and policy extraction) and derive a practical recipe that unlocks its full potential. 

---
# Revealing Behavioral Plasticity in Large Language Models: A Token-Conditional Perspective 

**Authors**: Liyuan Mao, Le Yu, Jing Zhou, Chujie Zheng, Bowen Yu, Chang Gao, Shixuan Liu, An Yang, Weinan Zhang, JunYang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08398)  

**Abstract**: In this work, we reveal that Large Language Models (LLMs) possess intrinsic behavioral plasticity-akin to chameleons adapting their coloration to environmental cues-that can be exposed through token-conditional generation and stabilized via reinforcement learning. Specifically, by conditioning generation on carefully selected token prefixes sampled from responses exhibiting desired behaviors, LLMs seamlessly adapt their behavioral modes at inference time (e.g., switching from step-by-step reasoning to direct answering) without retraining. Based on this insight, we propose Token-Conditioned Reinforcement Learning (ToCoRL), a principled framework that leverages RL to internalize this chameleon-like plasticity, transforming transient inference-time adaptations into stable and learnable behavioral patterns. ToCoRL guides exploration with token-conditional generation and keep enhancing exploitation, enabling emergence of appropriate behaviors. Extensive experiments show that ToCoRL enables precise behavioral control without capability degradation. Notably, we show that large reasoning models, while performing strongly on complex mathematics, can be effectively adapted to excel at factual question answering, which was a capability previously hindered by their step-by-step reasoning patterns. 

---
# Computational modeling of early language learning from acoustic speech and audiovisual input without linguistic priors 

**Authors**: Okko Räsänen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08359)  

**Abstract**: Learning to understand speech appears almost effortless for typically developing infants, yet from an information-processing perspective, acquiring a language from acoustic speech is an enormous challenge. This chapter reviews recent developments in using computational models to understand early language acquisition from speech and audiovisual input. The focus is on self-supervised and visually grounded models of perceptual learning. We show how these models are becoming increasingly powerful in learning various aspects of speech without strong linguistic priors, and how many features of early language development can be explained through a shared set of learning principles-principles broadly compatible with multiple theories of language acquisition and human cognition. We also discuss how modern learning simulations are gradually becoming more realistic, both in terms of input data and in linking model behavior to empirical findings on infant language development. 

---
# Towards plausibility in time series counterfactual explanations 

**Authors**: Marcin Kostrzewa, Krzysztof Galus, Maciej Zięba  

**Link**: [PDF](https://arxiv.org/pdf/2603.08349)  

**Abstract**: We present a new method for generating plausible counterfactual explanations for time series classification problems. The approach performs gradient-based optimization directly in the input space. To enforce plausibility, we integrate soft-DTW (dynamic time warping) alignment with $k$-nearest neighbors from the target class, which effectively encourages the generated counterfactuals to adopt a realistic temporal structure. The overall optimization objective is a multi-faceted loss function that balances key counterfactual properties. It incorporates losses for validity, sparsity, and proximity, alongside the novel soft-DTW-based plausibility component. We conduct an evaluation of our method against several strong reference approaches, measuring the key properties of the generated counterfactuals across multiple dimensions. The results demonstrate that our method achieves competitive performance in validity while significantly outperforming existing approaches in distributional alignment with the target class, indicating superior temporal realism. Furthermore, a qualitative analysis highlights the critical limitations of existing methods in preserving realistic temporal structure. This work shows that the proposed method consistently generates counterfactual explanations for time series classifiers that are not only valid but also highly plausible and consistent with temporal patterns. 

---
# Electrocardiogram Classification with Transformers Using Koopman and Wavelet Features 

**Authors**: Sucheta Ghosh, Zahra Monfared  

**Link**: [PDF](https://arxiv.org/pdf/2603.08339)  

**Abstract**: Electrocardiogram (ECG) analysis is vital for detecting cardiac abnormalities, yet robust automated classification is challenging due to the complexity and variability of physiological signals. In this work, we investigate transformer-based ECG classification using features derived from the Koopman operator and wavelet transforms. Two tasks are studied: (1) binary classification (Normal vs. Non-normal), and (2) four-class classification (Normal, Atrial Fibrillation, Ventricular Arrhythmia, Block). We use Extended Dynamic Mode Decomposition (EDMD) to approximate the Koopman operator. Our results show that wavelet features excel in binary classification, while Koopman features, when paired with transformers, achieve superior performance in the four-class setting. A simple hybrid of Koopman and wavelet features does not improve accuracy. However, selecting an appropriate EDMD dictionary -- specifically a radial basis function dictionary with tuned parameters -- yields significant gains, surpassing the wavelet-only baseline and the hybrid wavelet-Koopman system. We also present a Koopman-based reconstruction analysis for interpretable insights into the learned dynamics and compare against a recurrent neural network baseline. Overall, our findings demonstrate the effectiveness of Koopman-based feature learning with transformers and highlight promising directions for integrating dynamical systems theory into time-series classification. 

---
# Detecting Fake Reviewer Groups in Dynamic Networks: An Adaptive Graph Learning Method 

**Authors**: Jing Zhang, Ke Huang, Yao Zhang, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08332)  

**Abstract**: The proliferation of fake reviews, often produced by organized groups, undermines consumer trust and fair competition on online platforms. These groups employ sophisticated strategies that evade traditional detection methods, particularly in cold-start scenarios involving newly launched products with sparse data. To address this, we propose the \underline{D}iversity- and \underline{S}imilarity-aware \underline{D}ynamic \underline{G}raph \underline{A}ttention-enhanced \underline{G}raph \underline{C}onvolutional \underline{N}etwork (DS-DGA-GCN), a new graph learning model for detecting fake reviewer groups. DS-DGA-GCN achieves robust detection since it focuses on the joint relationships among products, reviews, and reviewers by modeling product-review-reviewer networks. DS-DGA-GCN also achieves adaptive detection by integrating a Network Feature Scoring (NFS) system and a new dynamic graph attention mechanism. The NFS system quantifies network attributes, including neighbor diversity, network self-similarity, as a unified feature score. The dynamic graph attention mechanism improves the adaptability and computational efficiency by captures features related to temporal information, node importance, and global network structure. Extensive experiments conducted on two real-world datasets derived from Amazon and Xiaohongshu demonstrate that DS-DGA-GCN significantly outperforms state-of-the-art baselines, achieving accuracies of up to \textbf{89.8\% and 88.3\%}, respectively. 

---
# SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation 

**Authors**: Yagiz Can Akay, Muhammed Yusuf Kartal, Esra Alparslan, Faruk Ortakoyluoglu, Arda Akpinar  

**Link**: [PDF](https://arxiv.org/pdf/2603.08329)  

**Abstract**: Answering complex, real-world queries often requires synthesizing facts scattered across vast document corpora. In these settings, standard retrieval-augmented generation (RAG) pipelines suffer from incomplete evidence coverage, while long-context large language models (LLMs) struggle to reason reliably over massive inputs. We introduce SPD-RAG, a hierarchical multi-agent framework for exhaustive cross-document question answering that decomposes the problem along the document axis. Each document is processed by a dedicated document-level agent operating only on its own content, enabling focused retrieval, while a coordinator dispatches tasks to relevant agents and aggregates their partial answers. Agent outputs are synthesized by merging partial answers through a token-bounded synthesis layer (which supports recursive map-reduce for massive corpora). This document-level specialization with centralized fusion improves scalability and answer quality in heterogeneous multidocument settings while yielding a modular, extensible retrieval pipeline. On the LOONG benchmark (EMNLP 2024) for long-context multi-document QA, SPD-RAG achieves an Avg Score of 58.1 (GPT-5 evaluation), outperforming Normal RAG (33.0) and Agentic RAG (32.8) while using only 38% of the API cost of a full-context baseline (68.0). 

---
# EndoSERV: A Vision-based Endoluminal Robot Navigation System 

**Authors**: Junyang Wu, Fangfang Xie, Minghui Zhang, Hanxiao Zhang, Jiayuan Sun, Yun Gu, Guang-Zhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08324)  

**Abstract**: Robot-assisted endoluminal procedures are increasingly used for early cancer intervention. However, the intricate, narrow and tortuous pathways within the luminal anatomy pose substantial difficulties for robot navigation. Vision-based navigation offers a promising solution, but existing localization approaches are error-prone due to tissue deformation, in vivo artifacts and a lack of distinctive landmarks for consistent localization. This paper presents a novel EndoSERV localization method to address these challenges. It includes two main parts, \textit{i.e.}, \textbf{SE}gment-to-structure and \textbf{R}eal-to-\textbf{V}irtual mapping, and hence the name. For long-range and complex luminal structures, we divide them into smaller sub-segments and estimate the odometry independently. To cater for label insufficiency, an efficient transfer technique maps real image features to the virtual domain to use virtual pose ground truth. The training phases of EndoSERV include an offline pretraining to extract texture-agnostic features, and an online phase that adapts to real-world conditions. Extensive experiments based on both public and clinical datasets have been performed to demonstrate the effectiveness of the method even without any real pose labels. 

---
# Human-AI Divergence in Ego-centric Action Recognition under Spatial and Spatiotemporal Manipulations 

**Authors**: Sadegh Rahmaniboldaji, Filip Rybansky, Quoc C. Vuong, Anya C. Hurlbert, Frank Guerin, Andrew Gilbert  

**Link**: [PDF](https://arxiv.org/pdf/2603.08317)  

**Abstract**: Humans consistently outperform state-of-the-art AI models in action recognition, particularly in challenging real-world conditions involving low resolution, occlusion, and visual clutter. Understanding the sources of this performance gap is essential for developing more robust and human-aligned models. In this paper, we present a large-scale human-AI comparative study of egocentric action recognition using Minimal Identifiable Recognition Crops (MIRCs), defined as the smallest spatial or spatiotemporal regions sufficient for reliable human recognition. We used our previously introduced, Epic ReduAct, a systematically spatially reduced and temporally scrambled dataset derived from 36 EPIC KITCHENS videos, spanning multiple spatial reduction levels and temporal conditions. Recognition performance is evaluated using over 3,000 human participants and the Side4Video model. Our analysis combines quantitative metrics, Average Reduction Rate and Recognition Gap, with qualitative analyses of spatial (high-, mid-, and low-level visual features) and spatiotemporal factors, including a categorisation of actions into Low Temporal Actions (LTA) and High Temporal Actions (HTA). Results show that human performance exhibits sharp declines when transitioning from MIRCs to subMIRCs, reflecting a strong reliance on sparse, semantically critical cues such as hand-object interactions. In contrast, the model degrades more gradually and often relies on contextual and mid- to low-level features, sometimes even exhibiting increased confidence under spatial reduction. Temporally, humans remain robust to scrambling when key spatial cues are preserved, whereas the model often shows insensitivity to temporal disruption, revealing class-dependent temporal sensitivities. 

---
# Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness 

**Authors**: Yehonatan Elisha, Oren Barkan, Noam Koenigstein  

**Link**: [PDF](https://arxiv.org/pdf/2603.08309)  

**Abstract**: Vision Transformers (ViTs) often degrade under distribution shifts because they rely on spurious correlations, such as background cues, rather than semantically meaningful features. Existing regularization methods, typically relying on simple foreground-background masks, which fail to capture the fine-grained semantic concepts that define an object (e.g., ``long beak'' and ``wings'' for a ``bird''). As a result, these methods provide limited robustness to distribution shifts. To address this limitation, we introduce a novel finetuning framework that steers model reasoning toward concept-level semantics. Our approach optimizes the model's internal relevance maps to align with spatially grounded concept masks. These masks are generated automatically, without manual annotation: class-relevant concepts are first proposed using an LLM-based, label-free method, and then segmented using a VLM. The finetuning objective aligns relevance with these concept regions while simultaneously suppressing focus on spurious background areas. Notably, this process requires only a minimal set of images and uses half of the dataset classes. Extensive experiments on five out-of-distribution benchmarks demonstrate that our method improves robustness across multiple ViT-based models. Furthermore, we show that the resulting relevance maps exhibit stronger alignment with semantic object parts, offering a scalable path toward more robust and interpretable vision models. Finally, we confirm that concept-guided masks provide more effective supervision for model robustness than conventional segmentation maps, supporting our central hypothesis. 

---
# Retrieval-Augmented Anatomical Guidance for Text-to-CT Generation 

**Authors**: Daniele Molino, Camillo Maria Caruso, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08305)  

**Abstract**: Text-conditioned generative models for volumetric medical imaging provide semantic control but lack explicit anatomical guidance, often resulting in outputs that are spatially ambiguous or anatomically inconsistent. In contrast, structure-driven methods ensure strong anatomical consistency but typically assume access to ground-truth annotations, which are unavailable when the target image is to be synthesized. We propose a retrieval-augmented approach for Text-to-CT generation that integrates semantic and anatomical information under a realistic inference setting. Given a radiology report, our method retrieves a semantically related clinical case using a 3D vision-language encoder and leverages its associated anatomical annotation as a structural proxy. This proxy is injected into a text-conditioned latent diffusion model via a ControlNet branch, providing coarse anatomical guidance while maintaining semantic flexibility. Experiments on the CT-RATE dataset show that retrieval-augmented generation improves image fidelity and clinical consistency compared to text-only baselines, while additionally enabling explicit spatial controllability, a capability inherently absent in such approaches. Further analysis highlights the importance of retrieval quality, with semantically aligned proxies yielding consistent gains across all evaluation axes. This work introduces a principled and scalable mechanism to bridge semantic conditioning and anatomical plausibility in volumetric medical image synthesis. Code will be released. 

---
# Graph-Instructed Neural Networks for parametric problems with varying boundary conditions 

**Authors**: Francesco Della Santa, Sandra Pieraccini, Maria Strazzullo  

**Link**: [PDF](https://arxiv.org/pdf/2603.08304)  

**Abstract**: This work addresses the accurate and efficient simulation of physical phenomena governed by parametric Partial Differential Equations (PDEs) characterized by varying boundary conditions, where parametric instances modify not only the physics of the problem but also the imposition of boundary constraints on the computational domain.
In such scenarios, classical Galerkin projection-based reduced order techniques encounter a fundamental bottleneck. Parametric boundaries typically necessitate a re-formulation of the discrete problem for each new configuration, and often, these approaches are unsuitable for real-time applications. To overcome these limitations, we propose a novel methodology based on Graph-Instructed Neural Networks (GINNs). The GINN framework effectively learns the mapping between the parametric description of the computational domain and the corresponding PDE solution. Our results demonstrate that the proposed GINN-based models, can efficiently represent highly complex parametric PDEs, serving as a robust and scalable asset for several applied-oriented settings when compared with fully connected architectures. 

---
# Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization 

**Authors**: Chaewon Moon, Dongkuk Si, Chulhee Yun  

**Link**: [PDF](https://arxiv.org/pdf/2603.08290)  

**Abstract**: We study the implicit bias of Sharpness-Aware Minimization (SAM) when training $L$-layer linear diagonal networks on linearly separable binary classification. For linear models ($L=1$), both $\ell_\infty$- and $\ell_2$-SAM recover the $\ell_2$ max-margin classifier, matching gradient descent (GD). However, for depth $L = 2$, the behavior changes drastically -- even on a single-example dataset. For $\ell_\infty$-SAM, the limit direction depends critically on initialization and can converge to $\mathbf{0}$ or to any standard basis vector, in stark contrast to GD, whose limit aligns with the basis vector of the dominant data coordinate. For $\ell_2$-SAM, we show that although its limit direction matches the $\ell_1$ max-margin solution as in the case of GD, its finite-time dynamics exhibit a phenomenon we call "sequential feature amplification", in which the predictor initially relies on minor coordinates and gradually shifts to larger ones as training proceeds or initialization increases. Our theoretical analysis attributes this phenomenon to $\ell_2$-SAM's gradient normalization factor applied in its perturbation, which amplifies minor coordinates early and allows major ones to dominate later, giving a concrete example where infinite-time implicit-bias analyses are insufficient. Synthetic and real-data experiments corroborate our findings. 

---
# A Blockchain-based Traceability System for AI-Driven Engine Blade Inspection 

**Authors**: Mahmoud Hafez, Eman Ouda, Mohammed A. Mohammed Eltoum, Khaled Salah, Yusra Abdulrahman  

**Link**: [PDF](https://arxiv.org/pdf/2603.08288)  

**Abstract**: Aircraft engine blade maintenance relies on inspection records shared across manufacturers, airlines, maintenance organizations, and regulators. Yet current systems are fragmented, difficult to audit, and vulnerable to tampering. This paper presents BladeChain, a blockchain-based system providing immutable traceability for blade inspections throughout the component life cycle. BladeChain is the first system to integrate multi-stakeholder endorsement, automated inspection scheduling, AI model provenance, and cryptographic evidence binding, delivering auditable maintenance traceability for aerospace deployments. Built on a four-stakeholder Hyperledger Fabric network (OEM, Airline, MRO, Regulator), BladeChain captures every life-cycle event in a tamper-evident ledger. A chaincode-enforced state machine governs blade status transitions and automatically triggers inspections when configurable flight hour, cycle, or calendar thresholds are exceeded, eliminating manual scheduling errors. Inspection artifacts are stored off-chain in IPFS and linked to on-chain records via SHA-256 hashes, with each inspection record capturing the AI model name and version used for defect detection. This enables regulators to audit both what defects were found and how they were found. The detection module is pluggable, allowing organizations to adopt or upgrade inspection models without modifying the ledger or workflows. We built a prototype and evaluated it on workloads of up to 100 blades, demonstrating 100% life cycle completion with consistent throughput of 26 operations per minute. A centralized SQL baseline quantifies the consensus overhead and highlights the security trade-off. Security validation confirms tamper detection within 17~ms through hash verification. 

---
# Evaluating LLM-Based Grant Proposal Review via Structured Perturbations 

**Authors**: William Thorne, Joseph James, Yang Wang, Chenghua Lin, Diana Maynard  

**Link**: [PDF](https://arxiv.org/pdf/2603.08281)  

**Abstract**: As AI-assisted grant proposals outpace manual review capacity in a kind of ``Malthusian trap'' for the research ecosystem, this paper investigates the capabilities and limitations of LLM-based grant reviewing for high-stakes evaluation. Using six EPSRC proposals, we develop a perturbation-based framework probing LLM sensitivity across six quality axes: funding, timeline, competency, alignment, clarity, and impact. We compare three review architectures: single-pass review, section-by-section analysis, and a 'Council of Personas' ensemble emulating expert panels. The section-level approach significantly outperforms alternatives in both detection rate and scoring reliability, while the computationally expensive council method performs no better than baseline. Detection varies substantially by perturbation type, with alignment issues readily identified but clarity flaws largely missed by all systems. Human evaluation shows LLM feedback is largely valid but skewed toward compliance checking over holistic assessment. We conclude that current LLMs may provide supplementary value within EPSRC review but exhibit high variability and misaligned review priorities. We release our code and any non-protected data. 

---
# TA-RNN-Medical-Hybrid: A Time-Aware and Interpretable Framework for Mortality Risk Prediction 

**Authors**: Zahra Jafari, Azadeh Zamanifar, Amirfarhad Farhadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.08278)  

**Abstract**: Accurate and interpretable mortality risk prediction in intensive care units (ICUs) remains a critical challenge due to the irregular temporal structure of electronic health records (EHRs), the complexity of longitudinal disease trajectories, and the lack of clinically grounded explanations in many data-driven models. To address these challenges, we propose \textit{TA-RNN-Medical-Hybrid}, a time-aware and knowledge-enriched deep learning framework that jointly models longitudinal clinical sequences and irregular temporal dynamics through explicit continuous-time encoding, along with standardized medical concept representations. The proposed framework extends time-aware recurrent modeling by integrating explicit continuous-time embeddings that operate independently of visit indexing, SNOMED-based disease representations, and a hierarchical dual-level attention mechanism that captures both visit-level temporal importance and feature/concept-level clinical relevance. This design enables accurate mortality risk estimation while providing transparent and clinically meaningful explanations aligned with established medical knowledge. We evaluate the proposed approach on the MIMIC-III critical care dataset and compare it against strong time-aware and sequential baselines. Experimental results demonstrate that TA-RNN-Medical-Hybrid consistently improves predictive performance in terms of AUC, accuracy, and recall-oriented F$_2$-score. Moreover, qualitative analysis shows that the model effectively decomposes mortality risk across time and clinical concepts, yielding interpretable insights into disease severity, chronicity, and temporal progression. Overall, the proposed framework bridges the gap between predictive accuracy and clinical interpretability, offering a scalable and transparent solution for high-stakes ICU decision support systems. 

---
# AdaCultureSafe: Adaptive Cultural Safety Grounded by Cultural Knowledge in Large Language Models 

**Authors**: Hankun Kang, Di Lin, Zhirong Liao, Pengfei Bai, Xinyi Zeng, Jiawei Jiang, Yuanyuan Zhu, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2603.08275)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs), respecting indigenous cultures becomes essential for models' culturally safety and responsible global applications. Existing studies separately consider cultural safety and cultural knowledge and neglect that the former should be grounded by the latter. This severely prevents LLMs from yielding culture-specific respectful responses. Consequently, adaptive cultural safety remains a formidable task. In this work, we propose to jointly model cultural safety and knowledge. First and foremost, cultural-safety and knowledge-paired data serve as the key prerequisite to conduct this research. However, the cultural diversity across regions and the subtlety of cultural differences pose significant challenges to the creation of such paired evaluation data. To address this issue, we propose a novel framework that integrates authoritative cultural knowledge descriptions curation, LLM-automated query generation, and heavy manual verification. Accordingly, we obtain a dataset named AdaCultureSafe containing 4.8K manually decomposed fine-grained cultural descriptions and the corresponding 48K manually verified safety- and knowledge-oriented queries. Upon the constructed dataset, we evaluate three families of popular LLMs on their cultural safety and knowledge proficiency, via which we make a critical discovery: no significant correlation exists between their cultural safety and knowledge proficiency. We then delve into the utility-related neuron activations within LLMs to investigate the potential cause of the absence of correlation, which can be attributed to the difference of the objectives of pre-training and post-alignment. We finally present a knowledge-grounded method, which significantly enhances cultural safety by enforcing the integration of knowledge into the LLM response generation process. 

---
# How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study Across Temperatures, Context Lengths, and Hardware Platforms 

**Authors**: JV Roig  

**Link**: [PDF](https://arxiv.org/pdf/2603.08274)  

**Abstract**: How much do large language models actually hallucinate when answering questions grounded in provided documents? Despite the critical importance of this question for enterprise AI deployments, reliable measurement has been hampered by benchmarks that rely on static datasets vulnerable to contamination, LLM-based judges with documented biases, or evaluation scales too small for statistical confidence. We address this gap using RIKER, a ground-truth-first evaluation methodology that enables deterministic scoring without human annotation. Across 35 open-weight models, three context lengths (32K, 128K, and 200K tokens), four temperature settings, and three hardware platforms (NVIDIA H200, AMD MI300X, and Intel Gaudi 3), we conducted over 172 billion tokens of evaluation - an order of magnitude beyond prior work. Our findings reveal that: (1) even the best-performing models fabricate answers at a non-trivial rate - 1.19% at best at 32K, with top-tier models at 5 - 7% - and fabrication rises steeply with context length, nearly tripling at 128K and exceeding 10% for all models at 200K; (2) model selection dominates all other factors, with overall accuracy spanning a 72-percentage-point range and model family predicting fabrication resistance better than model size; (3) temperature effects are nuanced - T=0.0 yields the best overall accuracy in roughly 60% of cases, but higher temperatures reduce fabrication for the majority of models and dramatically reduce coherence loss (infinite generation loops), which can reach 48x higher rates at T=0.0 versus T=1.0; (4) grounding ability and fabrication resistance are distinct capabilities - models that excel at finding facts may still fabricate facts that do not exist; and (5) results are consistent across hardware platforms, confirming that deployment decisions need not be hardware-dependent. 

---
# SCL-GNN: Towards Generalizable Graph Neural Networks via Spurious Correlation Learning 

**Authors**: Yuxiang Zhang, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.08270)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable success across diverse tasks. However, their generalization capability is often hindered by spurious correlations between node features and labels in the graph. Our analysis reveals that GNNs tend to exploit imperceptible statistical correlations in training data, even when such correlations are unreliable for prediction. To address this challenge, we propose the Spurious Correlation Learning Graph Neural Network (SCL-GNN), a novel framework designed to enhance generalization on both Independent and Identically Distributed (IID) and Out-of-Distribution (OOD) graphs. SCL-GNN incorporates a principled spurious correlation learning mechanism, leveraging the Hilbert-Schmidt Independence Criterion (HSIC) to quantify correlations between node representations and class scores. This enables the model to identify and mitigate irrelevant but influential spurious correlations effectively. Additionally, we introduce an efficient bi-level optimization strategy to jointly optimize modules and GNN parameters, preventing overfitting. Extensive experiments on real-world and synthetic datasets demonstrate that SCL-GNN consistently outperforms state-of-the-art baselines under various distribution shifts, highlighting its robustness and generalization capabilities. 

---
# SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM 

**Authors**: Makoto Sato, Yusuke Iwasawa, Yujin Tang, So Kuroki  

**Link**: [PDF](https://arxiv.org/pdf/2603.08269)  

**Abstract**: In-context imitation learning allows robots to acquire skills from demonstrations, yet one-shot trajectory generation remains fragile under environmental variation. We propose SAIL, a framework that reframes robot imitation as an iterative refinement problem capable of scaling with test-time compute. SAIL utilizes Monte Carlo Tree Search, where each node is a complete trajectory and edges correspond to trajectory refinements. The process is guided by three core components: an automated archive of successful trajectories for contextually relevant retrieval, a vision language model-based scoring mechanism for trajectory evaluation, and a step-level feedback that provides trajectory-aligned scores for iterative refinement. Experiments across six diverse manipulation tasks in simulation and real-world validation clearly demonstrate that increasing test-time compute consistently improves success rates, achieving up to 95% on complex tasks. Our results suggest that trajectory-level test-time scaling is a robust path toward more generalizable robotic agents. 

---
# Fibration Policy Optimization 

**Authors**: Chang Li, Tshihao Tsu, Yaren Zhang, Chao Xue, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2603.08239)  

**Abstract**: Large language models are increasingly trained as heterogeneous systems spanning multiple domains, expert partitions, and agentic pipelines, yet prevalent proximal objectives operate at a single scale and lack a principled mechanism for coupling token-level, trajectory-level, and higher-level hierarchical stability control. To bridge this gap, we derive the Aggregational Policy Censoring Objective (APC-Obj), the first exact unconstrained reformulation of sample-based TV-TRPO, establishing that clipping-based surrogate design and trust-region optimization are dual formulations of the same problem. Building on this foundation, we develop Fiber Bundle Gating (FBG), an algebraic framework that organizes sampled RL data as a fiber bundle and decomposes ratio gating into a base-level gate on trajectory aggregates and a fiber-level gate on per-token residuals, with provable first-order agreement with the true RL objective near on-policy. From APC-Obj and FBG we derive Fibration Policy Optimization (or simply, FiberPO), a concrete objective whose Jacobian is block-diagonal over trajectories, reduces to identity at on-policy, and provides better update direction thus improving token efficiency. The compositional nature of the framework extends beyond the trajectory-token case: fibrations compose algebraically into a Fibration Gating Hierarchy (FGH) that scales the same gating mechanism to arbitrary hierarchical depth without new primitives, as demonstrated by FiberPO-Domain, a four-level instantiation with independent trust-region budgets at the domain, prompt group, trajectory, and token levels. Together, these results connect the trust-region theory, a compositional algebraic structure, and practical multi-scale stability control into a unified framework for LLM policy optimization. 

---
# Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema 

**Authors**: Pablo Jimenez-Lizcano, Sergio Romero-Tapiador, Ruben Tolosana, Aythami Morales, Guillermo González de Rivera, Ruben Vera-Rodriguez, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2603.08235)  

**Abstract**: Diabetic retinopathy (DR) and diabetic macular edema (DME) are leading causes of preventable blindness among working-age adults. Traditional approaches in the literature focus on standard color fundus photography (CFP) for the detection of these conditions. Nevertheless, recent ultra-widefield imaging (UWF) offers a significantly wider field of view in comparison to CFP. Motivated by this, the present study explores state-of-the-art deep learning (DL) methods and UWF imaging on three clinically relevant tasks: i) image quality assessment for UWF, ii) identification of referable diabetic retinopathy (RDR), and iii) identification of DME. Using the publicly available UWF4DR Challenge dataset, released as part of the MICCAI 2024 conference, we benchmark DL models in the spatial (RGB) and frequency domains, including popular convolutional neural networks (CNNs) as well as recent vision transformers (ViTs) and foundation models. In addition, we explore a final feature-level fusion to increase robustness. Finally, we also analyze the decisions of the DL models using Grad-CAM, increasing the explainability. Our proposal achieves consistently strong performance across all architectures, underscoring the competitiveness of emerging ViTs and foundation models and the promise of feature-level fusion and frequency-domain representations for UWF analysis. 

---
# Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction 

**Authors**: Xiaofeng Yu, Jiaheng Dong, Jean Honorio, Abhirup Ghosh, Hong Jia, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08230)  

**Abstract**: Speech emotion recognition plays an important role in various applications. However, most existing approaches predict a single emotion label, oversimplifying the inherently ambiguous nature of human emotional expression. Recent large audio-language models show promise in generating richer outputs, but their reasoning ability for ambiguous emotional understanding remains limited. In this work, we reformulate ambiguous emotion recognition as a distributional reasoning problem and present the first systematic study of ambiguity-aware reasoning in LALMs. Our framework comprises two complementary components: an ambiguity-aware objective that aligns predictions with human perceptual distributions, and a structured ambiguity-aware chain-of-thought supervision that guides reasoning over emotional cues. Experiments on IEMOCAP and CREMA-D demonstrate consistent improvements across SFT, DPO, and GRPO training strategies. 

---
# SplitAgent: A Privacy-Preserving Distributed Architecture for Enterprise-Cloud Agent Collaboration 

**Authors**: Jianshu She  

**Link**: [PDF](https://arxiv.org/pdf/2603.08221)  

**Abstract**: Enterprise adoption of cloud-based AI agents faces a fundamental privacy dilemma: leveraging powerful cloud models requires sharing sensitive data, while local processing limits capability. Current agent frameworks like MCP and A2A assume complete data sharing, making them unsuitable for enterprise environments with confidential information. We present SplitAgent, a novel distributed architecture that enables privacy-preserving collaboration between enterprise-side privacy agents and cloud-side reasoning agents. Our key innovation is context-aware dynamic sanitization that adapts privacy protection based on task semantics -- contract review requires different sanitization than code review or financial analysis. SplitAgent extends existing agent protocols with differential privacy guarantees, zero-knowledge tool verification, and privacy budget management. Through comprehensive experiments on enterprise scenarios, we demonstrate that SplitAgent achieves 83.8\% task accuracy while maintaining 90.1\% privacy protection, significantly outperforming static approaches (73.2\% accuracy, 79.7\% privacy). Context-aware sanitization improves task utility by 24.1\% over static methods while reducing privacy leakage by 67\%. Our architecture provides a practical path for enterprise AI adoption without compromising sensitive data. 

---
# Revisiting Gradient Staleness: Evaluating Distance Metrics for Asynchronous Federated Learning Aggregation 

**Authors**: Patrick Wilhelm, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2603.08211)  

**Abstract**: In asynchronous federated learning (FL), client devices send updates to a central server at varying times based on their computational speed, often using stale versions of the global model. This staleness can degrade the convergence and accuracy of the global model. Previous work, such as AsyncFedED, proposed an adaptive aggregation method using Euclidean distance to measure staleness. In this paper, we extend this approach by exploring alternative distance metrics to more accurately capture the effect of gradient staleness. We integrate these metrics into the aggregation process and evaluate their impact on convergence speed, model performance, and training stability under heterogeneous clients and non-IID data settings. Our results demonstrate that certain metrics lead to more robust and efficient asynchronous FL training, offering a stronger foundation for practical deployment. 

---
# Alignment-Aware and Reliability-Gated Multimodal Fusion for Unmanned Aerial Vehicle Detection Across Heterogeneous Thermal-Visual Sensors 

**Authors**: Ishrat Jahan, Molla E Majid, M Murugappan, Muhammad E. H. Chowdhury, N.B.Prakash, Saad Bin Abul Kashem, Balamurugan Balusamy, Amith Khandakar  

**Link**: [PDF](https://arxiv.org/pdf/2603.08208)  

**Abstract**: Reliable unmanned aerial vehicle (UAV) detection is critical for autonomous airspace monitoring but remains challenging when integrating sensor streams that differ substantially in resolution, perspective, and field of view. Conventional fusion methods-such as wavelet-, Laplacian-, and decision-level approaches-often fail to preserve spatial correspondence across modalities and suffer from annotation of inconsistencies, limiting their robustness in real-world settings. This study introduces two fusion strategies, Registration-aware Guided Image Fusion (RGIF) and Reliability-Gated Modality-Attention Fusion (RGMAF), designed to overcome these limitations. RGIF employs Enhanced Correlation Coefficient (ECC)-based affine registration combined with guided filtering to maintain thermal saliency while enhancing structural detail. RGMAF integrates affine and optical-flow registration with a reliability-weighted attention mechanism that adaptively balances thermal contrast and visual sharpness. Experiments were conducted on the Multi-Sensor and Multi-View Fixed-Wing (MMFW)-UAV dataset comprising 147,417 annotated air-to-air frames collected from infrared, wide-angle, and zoom sensors. Among single-modality detectors, YOLOv10x demonstrated the most stable cross-domain performance and was selected as the detection backbone for evaluating fused imagery. RGIF improved the visual baseline by 2.13% mAP@50 (achieving 97.65%), while RGMAF attained the highest recall of 98.64%. These findings show that registration-aware and reliability-adaptive fusion provides a robust framework for integrating heterogeneous modalities, substantially enhancing UAV detection performance in multimodal environments. 

---
# Distributional Regression with Tabular Foundation Models: Evaluating Probabilistic Predictions via Proper Scoring Rules 

**Authors**: Jonas Landsgesell, Pascal Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2603.08206)  

**Abstract**: Prior-Data Fitted Networks (PFNs), such as TabPFN and TabICL, have revolutionized tabular deep learning by leveraging in-context learning for tabular data.
These models are meant as foundation models for classification and regression settings and promise to greatly simplify deployment in practical settings because their performance is unprecedented (in terms of mean squared error or $R^2$, when measured on common benchmarks like TabArena or TALENT).
However, we see an important weakness of current benchmarks for the regression setting: the current benchmarks focus on evaluating win rates and performance using metrics like (root) mean squared error or $R^2$.
Therefore, these leaderboards (implicitly and explicitly) push researchers to optimize for machine learning pipelines which elicit a good mean value estimate.
The main problem is that this approach only evaluates a point estimate (namely the mean estimator which is the Bayes estimator associated with the mean squared error loss).
In this article we discuss the application of proper scoring rules for evaluating the goodness of probabilistic forecasts in distributional regression.
We also propose to enhance common machine learning benchmarks with metrics for probabilistic regression.
To improve the status quo and make the machine learning community aware of scoring rules for probabilistic regression, we advocate to use the continuous ranked probability score (CRPS) in benchmarks for probabilistic regression.
However, we also illustrate that the choice of the scoring rule changes the inductive bias of the trained model. We, therefore, advocate for finetuning or promptable tabular foundation models. 

---
# MM-TS: Multi-Modal Temperature and Margin Schedules for Contrastive Learning with Long-Tail Data 

**Authors**: Siarhei Sheludzko, Dhimitrios Duka, Bernt Schiele, Hilde Kuehne, Anna Kukleva  

**Link**: [PDF](https://arxiv.org/pdf/2603.08202)  

**Abstract**: Contrastive learning has become a fundamental approach in both uni-modal and multi-modal frameworks. This learning paradigm pulls positive pairs of samples closer while pushing negatives apart. In the uni-modal setting (e.g., image-based learning), previous research has shown that the strength of these forces can be controlled through the temperature parameter. In this work, we propose Multi-Modal Temperature and Margin Schedules (MM-TS), extending the concept of uni-modal temperature scheduling to multi-modal contrastive learning. Our method dynamically adjusts the temperature in the contrastive loss during training, modulating the attraction and repulsion forces in the multi-modal setting. Additionally, recognizing that standard multi-modal datasets often follow imbalanced, long-tail distributions, we adapt the temperature based on the local distribution of each training sample. Specifically, samples from dense clusters are assigned a higher temperature to better preserve their semantic structure. Furthermore, we demonstrate that temperature scheduling can be effectively integrated within a max-margin framework, thereby unifying the two predominant approaches in multi-modal contrastive learning: InfoNCE loss and max-margin objective. We evaluate our approach on four widely used image- and video-language datasets, Flickr30K, MSCOCO, EPIC-KITCHENS-100, and YouCook2, and show that our dynamic temperature and margin schedules improve performance and lead to new state-of-the-art results in the field. 

---
# TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation 

**Authors**: Toms Bergmanis, Martins Kronis, Ingus Jānis Pretkalniņš, Dāvis Nicmanis, Jeļizaveta Jeļinska, Roberts Rozis, Rinalds Vīksna, Mārcis Pinnis  

**Link**: [PDF](https://arxiv.org/pdf/2603.08182)  

**Abstract**: Large language models often underperform in many European languages due to the dominance of English and a few high-resource languages in training data. This paper presents TildeOpen LLM, a 30-billion-parameter open-weight foundational model trained for 34 European languages to promote linguistic equity and improve performance for low-resource languages. To address the data imbalance, we combine dataset upsampling with a curriculum-based training schedule that alternates between uniform and natural language distributions. The resulting model performs favorably compared to other multilingual LLMs despite being trained with significantly fewer computing resources. Evaluation across multiple multilingual benchmarks shows that TildeOpen surpasses existing open-weight models in text generation and comprehension, particularly for Baltic, Finno-Ugric, and Slavic languages. Human evaluations confirm an up to tenfold reduction in linguistic errors relative to leading baselines. The model and associated resources are fully open-weight and publicly available at this http URL. These outcomes demonstrate that careful data curation and balanced training strategies can substantially enhance multilingual model quality without increasing model size or training volume. 

---
# Privacy-Preserving End-to-End Full-Duplex Speech Dialogue Models 

**Authors**: Nikita Kuzmin, Tao Zhong, Jiajun Deng, Yingke Zhu, Tristan Tsoi, Tianxiang Cao, Simon Lui, Kong Aik Lee, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2603.08179)  

**Abstract**: End-to-end full-duplex speech models feed user audio through an always-on LLM backbone, yet the speaker privacy implications of their hidden representations remain unexamined. Following the VoicePrivacy 2024 protocol with a lazy-informed attacker, we show that the hidden states of SALM-Duplex and Moshi leak substantial speaker identity across all transformer layers. Layer-wise and turn-wise analyses reveal that leakage persists across all layers, with SALM-Duplex showing stronger leakage in early layers while Moshi leaks uniformly, and that Linkability rises sharply within the first few turns. We propose two streaming anonymization setups using Stream-Voice-Anon: a waveform-level front-end (Anon-W2W) and a feature-domain replacement (Anon-W2F). Anon-W2F raises EER by over 3.5x relative to the discrete encoder baseline (11.2% to 41.0%), approaching the 50% random-chance ceiling, while Anon-W2W retains 78-93% of baseline sBERT across setups with sub-second response latency (FRL under 0.8 s). 

---
# Is continuous CoT better suited for multi-lingual reasoning? 

**Authors**: Ali Hamza Bashir, Behzad Shomali, Markus Frey, Mehdi Ali, Rafet Sifa, David Berghaus  

**Link**: [PDF](https://arxiv.org/pdf/2603.08177)  

**Abstract**: We investigate whether performing reasoning in a continuous latent space leads to more robust multilingual capabilities. We compare Continuous Chain-of-Thought (using the CODI framework) against standard supervised fine-tuning across five typologically diverse languages: English, Chinese, German, French, and Urdu. Our experiments on GSM8k and CommonsenseQA demonstrate that continuous reasoning significantly outperforms explicit reasoning on low-resource languages, particularly in zero-shot settings where the target language was not seen during training. Additionally, this approach achieves extreme efficiency, compressing reasoning traces by approximately $29\times$ to $50\times$. These findings indicate that continuous latent representations naturally exhibit greater language invariance, offering a scalable solution for cross-lingual reasoning. 

---
# Evolution Strategy-Based Calibration for Low-Bit Quantization of Speech Models 

**Authors**: Lucas Rakotoarivony  

**Link**: [PDF](https://arxiv.org/pdf/2603.08173)  

**Abstract**: Quantization has become essential for the efficient deployment of speech processing systems. Although widely studied, most existing quantization methods were developed for vision and NLP architectures, while the specific challenges of audio signals remain largely overlooked. In particular, we show that audio activations can exhibit large calibration ranges, leading to significant information loss when standard calibration techniques are applied. To address this, we propose ESC, an Evolution Strategy-based Calibration method that formulates activation scaling as an optimization problem and solves it using a two-step local-global scheme driven by an evolution strategy. ESC enables unaltered performance under full INT8 quantization and is the first calibration method to achieve near-lossless performance for full INT4 quantization across multiple speech tasks. Integrating ESC with PTQ methods further reduces performance loss, achieving a 1% relative accuracy degradation on the AST model. 

---
# An explainable hybrid deep learning-enabled intelligent fault detection and diagnosis approach for automotive software systems validation 

**Authors**: Mohammad Abboush, Ehab Ghannoum, Andreas Rausch  

**Link**: [PDF](https://arxiv.org/pdf/2603.08165)  

**Abstract**: Advancements in data-driven machine learning have emerged as a pivotal element in supporting automotive software systems (ASSs) engineering across various levels of the V-development process. Duringsystemverificationandvalidation,theintegrationofanintelligent fault detection anddiagnosis (FDD) model with test recordings analysis process serves as a powerful tool for efficiency ensuring functional safety. However, the lack of interpretability of the black-box FDD models developed not only hinders understanding of the cause underlying the prediction, but also prevents the model from being adapted based on the prediction result. This, in turn, increases the computational cost required for this http URL address this challenge, a novel explainable method for fault detection, identification, and localization is proposed in this article with the aim of providing a clear understanding of the logic behind the prediction outcome. To this end, a hybrid 1dCNN-GRU-based intelligent model was developed to analyze the recordings from the real-time validation process of ASSs. The employment of explainable AI techniques, i.e., IGs, DeepLIFT, Gradient SHAP, and DeepLIFT SHAP, was instrumental in enabling model adaptation and facilitating the root cause analysis (RCA). The proposed approach is applied to the real time dataset collected during a virtual test drive performed by the user on hardware in the loop system. 

---
# Gradually Excavating External Knowledge for Implicit Complex Question Answering 

**Authors**: Chang Liu, Xiaoguang Li, Lifeng Shang, Xin Jiang, Qun Liu, Edmund Y. Lam, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.08148)  

**Abstract**: Recently, large language models (LLMs) have gained much attention for the emergence of human-comparable capabilities and huge potential. However, for open-domain implicit question-answering problems, LLMs may not be the ultimate solution due to the reasons of: 1) uncovered or out-of-date domain knowledge, 2) one-shot generation and hence restricted comprehensiveness. To this end, this work proposes a gradual knowledge excavation framework for open-domain complex question answering, where LLMs iteratively and actively acquire external information, and then reason based on acquired historical knowledge. Specifically, during each step of the solving process, the model selects an action to execute, such as querying external knowledge or performing a single logical reasoning step, to gradually progress toward a final answer. Our method can effectively leverage plug-and-play external knowledge and dynamically adjust the strategy for solving complex questions. Evaluated on the StrategyQA dataset, our method achieves 78.17% accuracy with less than 6% parameters of its competitors, setting new SOTA for ~10B-scale LLMs. 

---
# DARC: Disagreement-Aware Alignment via Risk-Constrained Decoding 

**Authors**: Mingxi Zou, Jiaxiang Chen, Junfan Li, Langzhang Liang, Qifan Wang, Xu Yinghui, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08145)  

**Abstract**: Preference-based alignment methods (e.g., RLHF, DPO) typically optimize a single scalar objective, implicitly averaging over heterogeneous human preferences. In practice, systematic annotator and user-group disagreement makes mean-reward maximization brittle and susceptible to proxy over-optimization. We propose **Disagreement-Aware Alignment via Risk-Constrained Decoding (DARC)**, a retraining-free inference-time method that frames response selection as distributionally robust, risk-sensitive decision making. Given multiple preference samples or scalable disagreement proxies, DARC reranks candidates by maximizing a *KL-robust (entropic)* satisfaction objective, and provides simple deployment controls that cap or penalize the corresponding entropic risk premium relative to the mean, enabling explicit risk budgets without retraining. We provide theoretical characterization linking this decoding rule to principled pessimism and KL-based distributionally robust optimization. Experiments on alignment benchmarks show that DARC reduces disagreement and tail risk while maintaining competitive average quality under noisy, heterogeneous feedback. 

---
# Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows 

**Authors**: Shentong Mo, Yibing Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.08126)  

**Abstract**: Coordinated audio generation based on video inputs typically requires a strict audio-visual (AV) alignment, where both semantics and rhythmics of the generated audio segments shall correspond to those in the video frames. Previous studies leverage a two-stage design where the AV encoders are firstly aligned via contrastive learning, then the encoded video representations guide the audio generation process. We observe that both contrastive learning and global video guidance are effective in aligning overall AV semantics while limiting temporally rhythmic synchronization. In this work, we propose FoleyFlow to first align unimodal AV encoders via masked modeling training, where the masked audio segments are recovered under the guidance of the corresponding video segments. After training, the AV encoders which are separately pretrained using only unimodal data are aligned with semantic and rhythmic consistency. Then, we develop a dynamic conditional flow for the final audio generation. Built upon the efficient velocity flow generation framework, our dynamic conditional flow utilizes temporally varying video features as the dynamic condition to guide corresponding audio segment generations. To this end, we extract coherent semantic and rhythmic representations during masked AV alignment, and use this representation of video segments to guide audio generation temporally. Our audio results are evaluated on the standard benchmarks and largely surpass existing results under several metrics. The superior performance indicates that FoleyFlow is effective in generating coordinated audios that are both semantically and rhythmically coherent to various video sequences. 

---
# SaiVLA-0: Cerebrum--Pons--Cerebellum Tripartite Architecture for Compute-Aware Vision-Language-Action 

**Authors**: Xiang Shi, Wenlong Huang, Menglin Zou, Xinhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.08124)  

**Abstract**: We revisit Vision-Language-Action through a neuroscience-inspired triad. Biologically, the Cerebrum provides stable high-level multimodal priors and remains frozen; the Pons Adapter integrates these cortical features with real-time proprioceptive inputs and compiles intent into execution-ready tokens; and the Cerebellum (ParaCAT) performs fast, parallel categorical decoding for online control, with hysteresis/EMA/temperature/entropy for stability. A fixed-ratio schedule and two-stage feature caching make the system compute-aware and reproducible. Inspired by active, foveated vision, our wrist ROIs are geometrically tied to the end-effector via calibrated projection, providing a movement-stabilized, high-resolution view that is sensitive to fine-grained pose changes and complements the global context of the main view.
The design is modular: upgrading the Cerebrum only retrains the Pons; changing robots only trains the Cerebellum; cerebellum-only RL can further refine control without touching high-level semantics. As a concept-and-protocol paper with preliminary evidence, we outline a timing protocol under matched conditions (GPU, resolution, batch) to verify anticipated efficiency gains. We also report preliminary LIBERO evidence showing that split feature caching reduces training time (7.5h to 4.5h) and improves average success (86.5% to 92.5%) under official N1.5 head-only training, and that SaiVLA0 reaches 99.0% mean success. 

---
# DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning 

**Authors**: Chi-Min Chan, Ehsan Hajiramezanali, Xiner Li, Edward De Brouwer, Carl Edwards, Wei Xue, Sirui Han, Yike Guo, Gabriele Scalia  

**Link**: [PDF](https://arxiv.org/pdf/2603.08095)  

**Abstract**: In scientific reasoning tasks, the veracity of the reasoning process is as critical as the final outcome. While Process Reward Models (PRMs) offer a solution to the coarse-grained supervision problems inherent in Outcome Reward Models (ORMs), their deployment is hindered by the prohibitive cost of obtaining expert-verified step-wise labels. This paper addresses the challenge of training reliable PRMs using abundant but noisy "weak" supervision. We argue that existing Weak-to-Strong Generalization (W2SG) theories lack prescriptive guidelines for selecting high-quality training signals from noisy data. To bridge this gap, we introduce the Dual-Consensus Weak-to-Strong (DC-W2S) framework. By intersecting Self-Consensus (SC) metrics among weak supervisors with Neighborhood-Consensus (NC) metrics in the embedding space, we stratify supervision signals into distinct reliability regimes. We then employ a curriculum of instance-level balanced sampling and label-level reliability-aware masking to guide the training process. We demonstrate that DC-W2S enables the training of robust PRMs for complex reasoning without exhaustive expert annotation, proving that strategic data curation is more effective than indiscriminate training on large-scale noisy datasets. 

---
# DSH-Bench: A Difficulty- and Scenario-Aware Benchmark with Hierarchical Subject Taxonomy for Subject-Driven Text-to-Image Generation 

**Authors**: Zhenyu Hu, Qing Wang, Te Cao, Luo Liao, Longfei Lu, Liqun Liu, Shuang Li, Hang Chen, Mengge Xue, Yuan Chen, Chao Deng, Peng Shu, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08090)  

**Abstract**: Significant progress has been achieved in subject-driven text-to-image (T2I) generation, which aims to synthesize new images depicting target subjects according to user instructions. However, evaluating these models remains a significant challenge. Existing benchmarks exhibit critical limitations: 1) insufficient diversity and comprehensiveness in subject images, 2) inadequate granularity in assessing model performance across different subject difficulty levels and prompt scenarios, and 3) a profound lack of actionable insights and diagnostic guidance for subsequent model refinement. To address these limitations, we propose DSH-Bench, a comprehensive benchmark that enables systematic multi-perspective analysis of subject-driven T2I models through four principal innovations: 1) a hierarchical taxonomy sampling mechanism ensuring comprehensive subject representation across 58 fine-grained categories, 2) an innovative classification scheme categorizing both subject difficulty level and prompt scenario for granular capability assessment, 3) a novel Subject Identity Consistency Score (SICS) metric demonstrating a 9.4\% higher correlation with human evaluation compared to existing measures in quantifying subject preservation, and 4) a comprehensive set of diagnostic insights derived from the benchmark, offering critical guidance for optimizing future model training paradigms and data construction strategies. Through an extensive empirical evaluation of 19 leading models, DSH-Bench uncovers previously obscured limitations in current approaches, establishing concrete directions for future research and development. 

---
# ImageEdit-R1: Boosting Multi-Agent Image Editing via Reinforcement Learning 

**Authors**: Yiran Zhao, Yaoqi Ye, Xiang Liu, Michael Qizhe Shieh, Trung Bui  

**Link**: [PDF](https://arxiv.org/pdf/2603.08059)  

**Abstract**: With the rapid advancement of commercial multi-modal models, image editing has garnered significant attention due to its widespread applicability in daily life. Despite impressive progress, existing image editing systems, particularly closed-source or proprietary models, often struggle with complex, indirect, or multi-step user instructions. These limitations hinder their ability to perform nuanced, context-aware edits that align with human intent. In this work, we propose ImageEdit-R1, a multi-agent framework for intelligent image editing that leverages reinforcement learning to coordinate high-level decision-making across a set of specialized, pretrained vision-language and generative agents. Each agent is responsible for distinct capabilities--such as understanding user intent, identifying regions of interest, selecting appropriate editing actions, and synthesizing visual content--while reinforcement learning governs their collaboration to ensure coherent and goal-directed behavior. Unlike existing approaches that rely on monolithic models or hand-crafted pipelines, our method treats image editing as a sequential decision-making problem, enabling dynamic and context-aware editing strategies. Experimental results demonstrate that ImageEdit-R1 consistently outperforms both individual closed-source diffusion models and alternative multi-agent framework baselines across multiple image editing datasets. 

---
# Speed3R: Sparse Feed-forward 3D Reconstruction Models 

**Authors**: Weining Ren, Xiao Tan, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.08055)  

**Abstract**: While recent feed-forward 3D reconstruction models accelerate 3D reconstruction by jointly inferring dense geometry and camera poses in a single pass, their reliance on dense attention imposes a quadratic complexity, creating a prohibitive computational bottleneck that severely limits inference speed. To resolve this, we introduce Speed3R, an end-to-end trainable model inspired by the core principle of Structure-from-Motion: that a sparse set of keypoints is sufficient for robust pose estimation. Speed3R features a dual-branch attention mechanism where a compression branch creates a coarse contextual prior to guide a selection branch, which performs fine-grained attention only on the most informative image tokens. This strategy mimics the efficiency of traditional keypoint matching, achieving a remarkable 12.4x inference speedup on 1000-view sequences, while introducing a minimal, controlled trade-off in geometric accuracy. Validated on standard benchmarks with both VGGT and $\pi^3$ backbones, our method delivers high-quality reconstructions at a fraction of computational cost, paving the way for efficient large-scale scene modeling. 

---
# Solution to the 10th ABAW Expression Recognition Challenge: A Robust Multimodal Framework with Safe Cross-Attention and Modality Dropout 

**Authors**: Jun Yu, Naixiang Zheng, Guoyuan Wang, Yunxiang Zhang, Lingsi Zhu, Jiaen Liang, Wei Huang, Shengping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.08034)  

**Abstract**: Emotion recognition in real-world environments is hindered by partial occlusions, missing modalities, and severe class imbalance. To address these issues, particularly for the Affective Behavior Analysis in-the-wild (ABAW) Expression challenge, we propose a multimodal framework that dynamically fuses visual and audio representations. Our approach uses a dual-branch Transformer architecture featuring a safe cross-attention mechanism and a modality dropout strategy. This design allows the network to rely on audio-based predictions when visual cues are absent. To mitigate the long-tail distribution of the Aff-Wild2 dataset, we apply focal loss optimization, combined with a sliding-window soft voting strategy to capture dynamic emotional transitions and reduce frame-level classification jitter. Experiments demonstrate that our framework effectively handles missing modalities and complex spatiotemporal dependencies, achieving an accuracy of 60.79% and an F1-score of 0.5029 on the Aff-Wild2 validation set. 

---
# GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables 

**Authors**: Zhengyu Li, Xiangfei Qiu, Yuhan Zhu, Xingjian Wu, Jilin Hu, Chenjuan Guo, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08032)  

**Abstract**: Exogenous variables offer valuable supplementary information for predicting future endogenous variables. Forecasting with exogenous variables needs to consider both past-to-future dependencies (i.e., temporal correlations) and the influence of exogenous variables on endogenous variables (i.e., channel correlations). This is pivotal when future exogenous variables are available, because they may directly affect the future endogenous variables. Many methods have been proposed for time series forecasting with exogenous variables, focusing on modeling temporal and channel correlations. However, most of them use a two-step strategy, modeling temporal and channel correlations separately, which limits their ability to capture joint correlations across time and channels. Furthermore, in real-world scenarios, time series are frequently affected by various forms of noises, underscoring the critical importance of robustness in such correlations modeling. To address these limitations, we propose GCGNet, a Graph-Consistent Generative Network for time series forecasting with exogenous variables. Specifically, GCGNet first employs a Variational Generator to produce coarse predictions. A Graph Structure Aligner then further guides it by evaluating the consistency between the generated and true correlations, where the correlations are represented as graphs, and are robust to noises. Finally, a Graph Refiner is proposed to refine the predictions to prevent degeneration and improve accuracy. Extensive experiments on 12 real-world datasets demonstrate that GCGNet outperforms state-of-the-art baselines. 

---
# DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention 

**Authors**: Younjoo Lee, Junghoo Lee, Seungkyun Dan, Jaiyoung Park, Jung Ho Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.08026)  

**Abstract**: Masked Diffusion Language Models (MDLMs) enable parallel token decoding, providing a promising alternative to the sequential nature of autoregressive generation. However, their iterative denoising process remains computationally expensive because it repeatedly processes the entire sequence at every step. We observe that across these diffusion steps, most token representations remain stable; only a small subset, which we term salient tokens, contributes meaningfully to the next update. Leveraging this temporal sparsity, we present DyLLM, a training-free inference framework that accelerates decoding by selectively computing only these salient tokens. DyLLM identifies saliency by measuring the cosine similarity of attention contexts between adjacent denoising steps. It recomputes feed-forward and attention operations only for salient tokens while reusing cached activations for the remainder. Across diverse reasoning and code-generation benchmarks, DyLLM achieves up to 9.6x higher throughput while largely preserving the baseline accuracy of state-of-the-art models like LLaDA and Dream. 

---
# Not Like Transformers: Drop the Beat Representation for Dance Generation with Mamba-Based Diffusion Model 

**Authors**: Sangjune Park, Inhyeok Choi, Donghyeon Soon, Youngwoo Jeon, Kyungdon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2603.08023)  

**Abstract**: Dance is a form of human motion characterized by emotional expression and communication, playing a role in various fields such as music, virtual reality, and content creation. Existing methods for dance generation often fail to adequately capture the inherently sequential, rhythmical, and music-synchronized characteristics of dance. In this paper, we propose \emph{MambaDance}, a new dance generation approach that leverages a Mamba-based diffusion model. Mamba, well-suited to handling long and autoregressive sequences, is integrated into our two-stage diffusion architecture, substituting off-the-shelf Transformer. Additionally, considering the critical role of musical beats in dance choreography, we propose a Gaussian-based beat representation to explicitly guide the decoding of dance sequences. Experiments on AIST++ and FineDance datasets for each sequence length show that our proposed method effectively generates plausible dance movements while reflecting essential characteristics, consistently from short to long dances, compared to the previous methods. Additional qualitative results and demo videos are available at \small{this https URL}. 

---
# Alignment--Process--Outcome: Rethinking How AIs and Humans Collaborate 

**Authors**: Haichang Li, Anjun Zhu, Arpit Narechania  

**Link**: [PDF](https://arxiv.org/pdf/2603.08017)  

**Abstract**: In real-world collaboration, alignment, process structure, and outcome quality do not exhibit a simple linear or one-to-one correspondence: similar alignment may accompany either rapid convergence or extensive multi-branch exploration, and lead to different results. Existing accounts often isolate these dimensions or focus on specific participant types, limiting structural accounts of collaboration.
We reconceptualize collaboration through two complementary lenses. The task lens models collaboration as trajectory evolution in a structured task space, revealing patterns such as advancement, branching, and backtracking. The intent lens examines how individual intents are expressed within shared contexts and enter situated decisions. Together, these lenses clarify the structural relationships among alignment, decision-making, and trajectory structure.
Rather than reducing collaboration to outcome quality or treating alignment as the sole objective, we propose a unified dynamic view of the relationships among alignment, process, and outcome, and use it to re-examine collaboration structure across Human-Human, AI-AI, and Human-AI settings. 

---
# FedMomentum: Preserving LoRA Training Momentum in Federated Fine-Tuning 

**Authors**: Peishen Yan, Yang Hua, Hao Wang, Jiaru Zhang, Xiaoyu Wu, Tao Song, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2603.08014)  

**Abstract**: Federated fine-tuning of large language models (LLMs) with low-rank adaptation (LoRA) offers a communication-efficient and privacy-preserving solution for task-specific adaptation. Naive aggregation of LoRA modules introduces noise due to mathematical incorrectness when averaging the downsampling and upsampling matrices independently. However, existing noise-free aggregation strategies inevitably compromise the structural expressiveness of LoRA, limiting its ability to retain client-specific adaptations by either improperly reconstructing the low-rank structure or excluding partially trainable components. We identify this problem as loss of training momentum, where LoRA updates fail to accumulate effectively across rounds, resulting in slower convergence and suboptimal performance. To address this, we propose FedMomentum, a novel framework that enables structured and momentum-preserving LoRA aggregation via singular value decomposition (SVD). Specifically, after aggregating low-rank updates in a mathematically correct manner, FedMomentum applies SVD to extract the dominant components that capture the main update directions. These components are used to reconstruct the LoRA modules with the same rank, while residual components can be retained and later merged into the backbone to preserve semantic information and ensure robustness. Extensive experiments across multiple tasks demonstrate that FedMomentum consistently outperforms prior state-of-the-art methods in convergence speed and final accuracy. 

---
# ViSA-Enhanced Aerial VLN: A Visual-Spatial Reasoning Enhanced Framework for Aerial Vision-Language Navigation 

**Authors**: Haoyu Tong, Xiangyu Dong, Xiaoguang Ma, Haoran Zhao, Yaoming Zhou, Chenghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.08007)  

**Abstract**: Existing aerial Vision-Language Navigation (VLN) methods predominantly adopt a detection-and-planning pipeline, which converts open-vocabulary detections into discrete textual scene graphs. These approaches are plagued by inadequate spatial reasoning capabilities and inherent linguistic ambiguities. To address these bottlenecks, we propose a Visual-Spatial Reasoning (ViSA) enhanced framework for aerial VLN. Specifically, a triple-phase collaborative architecture is designed to leverage structured visual prompting, enabling Vision-Language Models (VLMs) to perform direct reasoning on image planes without the need for additional training or complex intermediate representations. Comprehensive evaluations on the CityNav benchmark demonstrate that the ViSA-enhanced VLN achieves a 70.3\% improvement in success rate compared to the fully trained state-of-the-art (SOTA) method, elucidating its great potential as a backbone for aerial VLN systems. 

---
# Aero-Promptness: Drag-Aware Aerodynamic Manipulability for Propeller-driven Vehicles 

**Authors**: Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2603.07998)  

**Abstract**: This work introduces the Drag-Aware Aerodynamic Manipulability (DAAM), a geometric framework for control allocation in redundant multirotors. By equipping the propeller spin-rate space with a Riemannian metric based on the remaining symmetric acceleration capacity of each motor, the formulation explicitly accounts for motor torque limits and aerodynamic drag. Mapping this metric through the nonlinear thrust law to the generalized force space yields a state-dependent manipulability volume. The log-determinant of this volume acts as a natural barrier function, strictly penalizing drag-induced saturation and low-spin thrust loss. Optimizing this volume along the allocation fibers provides a redundancy resolution strategy inherently invariant to arbitrary coordinate scaling in the generalized-force space. Analytically, we prove that the resulting optimal allocations locally form smooth embedded manifolds, and we geometrically characterize the global jump discontinuities that inevitably arise from physical actuator limits and spin-rate sign transitions. 

---
# \$OneMillion-Bench: How Far are Language Agents from Human Experts? 

**Authors**: Qianyu Yang, Yang Liu, Jiaqi Li, Jun Bai, Hao Chen, Kaiyuan Chen, Tiliang Duan, Jiayun Dong, Xiaobo Hu, Zixia Jia, Yang Liu, Tao Peng, Yixin Ren, Ran Tian, Zaiyuan Wang, Yanglihong Xiao, Gang Yao, Lingyue Yin, Ge Zhang, Chun Zhang, Jianpeng Jiao, Zilong Zheng, Yuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.07980)  

**Abstract**: As language models (LMs) evolve from chat assistants to long-horizon agents capable of multi-step reasoning and tool use, existing benchmarks remain largely confined to structured or exam-style tasks that fall short of real-world professional demands. To this end, we introduce \$OneMillion-Bench \$OneMillion-Bench, a benchmark of 400 expert-curated tasks spanning Law, Finance, Industry, Healthcare, and Natural Science, built to evaluate agents across economically consequential scenarios. Unlike prior work, the benchmark requires retrieving authoritative sources, resolving conflicting evidence, applying domain-specific rules, and making constraint decisions, where correctness depends as much on the reasoning process as the final answer. We adopt a rubric-based evaluation protocol scoring factual accuracy, logical coherence, practical feasibility, and professional compliance, focused on expert-level problems to ensure meaningful differentiation across agents. Together, \$OneMillion-Bench provides a unified testbed for assessing agentic reliability, professional depth, and practical readiness in domain-intensive scenarios. 

---
# Emergence is Overrated: AGI as an Archipelago of Experts 

**Authors**: Daniel Kilov  

**Link**: [PDF](https://arxiv.org/pdf/2603.07979)  

**Abstract**: Krakauer, Krakauer, and Mitchell (2025) distinguish between emergent capabilities and emergent intelligence, arguing that true intelligence requires efficient coarse-grained representations enabling diverse problem-solving through analogy and minimal modification. They contend that intelligence means doing "more with less" through compression and generalization, contrasting this with "vast assemblages of diverse calculators" that merely accumulate specialized capabilities. This paper examines whether their framework accurately characterizes human intelligence and its implications for conceptualizing artificial general intelligence. Drawing on empirical evidence from cognitive science, I demonstrate that human expertise operates primarily through domain-specific pattern accumulation rather than elegant compression. Expert performance appears flexible not through unifying principles but through vast repertoires of specialized responses. Creative breakthroughs themselves may emerge through evolutionary processes of blind variation and selective retention rather than principled analogical reasoning. These findings suggest reconceptualizing AGI as an "archipelago of experts": isolated islands of specialized competence without unifying principles or shared representations. If we accept human expertise with its characteristic brittleness as genuine intelligence, then consistency demands recognizing that artificial systems comprising millions of specialized modules could constitute general intelligence despite lacking KKM's emergent intelligence. 

---
# VORL-EXPLORE: A Hybrid Learning Planning Approach to Multi-Robot Exploration in Dynamic Environments 

**Authors**: Ning Liu, Sen Shen, Zheng Li, Sheng Liu, Dongkun Han, Shangke Lyu, Thomas Braunl  

**Link**: [PDF](https://arxiv.org/pdf/2603.07973)  

**Abstract**: Hierarchical multi-robot exploration commonly decouples frontier allocation from local navigation, which can make the system brittle in dense and dynamic environments. Because the allocator lacks direct awareness of execution difficulty, robots may cluster at bottlenecks, trigger oscillatory replanning, and generate redundant coverage. We propose VORL-EXPLORE, a hybrid learning and planning framework that addresses this limitation through execution fidelity, a shared estimate of local navigability that couples task allocation with motion execution. This fidelity signal is incorporated into a fidelity-coupled Voronoi objective with inter-robot repulsion to reduce contention before it emerges. It also drives a risk-aware adaptive arbitration mechanism between global A* guidance and a reactive reinforcement learning policy, balancing long-range efficiency with safe interaction in confined spaces. The framework further supports online self-supervised recalibration of the fidelity model using pseudo-labels derived from recent progress and safety outcomes, enabling adaptation to non-stationary obstacles without manual risk tuning. We evaluate this capability separately in a dedicated severe-traffic ablation. Extensive experiments in randomized grids and a Gazebo factory scenario show high success rates, shorter path length, lower overlap, and robust collision avoidance. The source code will be made publicly available upon acceptance. 

---
# PSTNet: Physically-Structured Turbulence Network 

**Authors**: Boris Kriuk, Fedor Kriuk  

**Link**: [PDF](https://arxiv.org/pdf/2603.07957)  

**Abstract**: Reliable real-time estimation of atmospheric turbulence intensity remains an open challenge for aircraft operating across diverse altitude bands, particularly over oceanic, polar, and data-sparse regions that lack operational nowcasting infrastructure. Classical spectral models encode climatological averages rather than the instantaneous atmospheric state, and generic ML regressors offer adaptivity but provide no guarantee that predictions respect fundamental scaling laws. This paper introduces the Physically-Structured Turbulence Network (PSTNet), a lightweight architecture that embeds physics directly into its structure. PSTNet couples four components: (i) a zero-parameter backbone derived from Monin-Obukhov theory, (ii) a regime-gated mixture of specialist sub-networks supervised by Richardson-number-derived soft targets, (iii) Feature-wise Linear Modulation layers conditioning hidden representations on local air-density ratio, and (iv) a Kolmogorov output layer enforcing inertial-subrange scaling as an architectural constraint. The entire model contains only 552 learnable parameters, requiring fewer than 2.5 kB of storage and executing in under 12s on a Cortex-M7 microcontroller. We validate PSTNet on 340 paired six-degree-of-freedom guidance simulations spanning three vehicle classes (Mach 2.8, 4.5, and 8.0) and six operational categories with real-time satellite weather ingestion. PSTNet achieves a mean miss-distance improvement of +2.8% with a 78% win rate and a statistically significant effect size. Our results demonstrate that encoding domain physics as architectural priors yields a more efficient and interpretable path to turbulence estimation accuracy than scaling model capacity, establishing PSTNet as a viable drop-in replacement for legacy look-up tables in resource-constrained, safety-critical on-board guidance systems. 

---
# ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework 

**Authors**: Yusong Wang, Chuang Yang, Jiawei Wang, Xiaohang Xu, Jiayi Xu, Dongyuan Li, Chuan Xiao, Renhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07946)  

**Abstract**: Human mobility generation aims to synthesize plausible trajectory data, which is widely used in urban system research. While Large Language Model-based methods excel at generating routine trajectories, they struggle to capture deviated mobility during large-scale societal events. This limitation stems from two critical gaps: (1) the absence of event-annotated mobility datasets for design and evaluation, and (2) the inability of current frameworks to reconcile competitions between users' habitual patterns and event-imposed constraints when making trajectory decisions. This work addresses these gaps with a twofold contribution. First, we construct the first event-annotated mobility dataset covering three major events: Typhoon Hagibis, COVID-19, and the Tokyo 2021 Olympics. Second, we propose ELLMob, a self-aligned LLM framework that first extracts competing rationales between habitual patterns and event constraints, based on Fuzzy-Trace Theory, and then iteratively aligns them to generate trajectories that are both habitually grounded and event-responsive. Extensive experiments show that ELLMob wins state-of-the-art baselines across all events, demonstrating its effectiveness. Our codes and datasets are available at this https URL. 

---
# AI Agents, Language, Deep Learning and the Next Revolution in Science 

**Authors**: Ke Li, Beijiang Liu, Bruce Mellado, Changzheng Yuan, Zhengde Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07940)  

**Abstract**: Modern science is reaching a critical inflection point. Instruments across disciplines, from particle physics and astronomy to genomics and climate modeling, now produce data of such scale, diversity, and interdependence that traditional analytical methods can no longer keep pace. This growing imbalance between data generation and data understanding signals the need for a new scientific paradigm. We propose that intelligent, human-supervised AI agents operating over deep-learning algorithms, represent the next evolution of the scientific method. Built upon large language models and multimodal learning, these agents can interpret scientific intent, design and execute analytical workflows, and ensure traceability through domain-specific languages that preserve human oversight and accountability. Particle physics, a historic incubator of computational innovation, offers the ideal testbed for this transition. At the Institute of High Energy Physics of the Chinese Academy of Sciences, the Dr. Sai system embodies this vision, a multi-agent reasoning framework deployed within collider research at the CEPC. This emerging approach does not replace human scientists but extends their cognitive reach, enabling discovery to scale with complexity and redefining how knowledge itself is produced in the age of intelligent machines. The significance of this paradigm transcends particle physics, offering a blueprint for all data-driven sciences facing the same complexity ceiling. 

---
# SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training 

**Authors**: Xin-Cheng Wen, Binbin Chen, Haoxuan Lan, Hang Yu, Peng Di, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.07927)  

**Abstract**: Large language models (LLMs) have transformed the software engineering landscape. Recently, numerous LLM-based agents have been developed to address real-world software issue fixing tasks. Despite their state-of-the-art performance, Despite achieving state-of-the-art performance, these agents face a significant challenge: \textbf{Insufficient high-quality issue descriptions.} Real-world datasets often exhibit misalignments between issue descriptions and their corresponding solutions, introducing noise and ambiguity that mislead automated agents and limit their problem-solving effectiveness. We propose \textbf{\textit{SWE-Fuse}}, an issue-description-aware training framework that fuses issue-description-guided and issue-free samples for training SWE agents. It consists of two key modules: (1) An issue-free-driven trajectory learning module for mitigating potentially misleading issue descriptions while enabling the model to learn step-by-step debugging processes; and (2) An entropy-aware RLVR training module, which adaptively adjusts training dynamics through entropy-driven clipping. It applies relaxed clipping under high entropy to encourage exploration, and stricter clipping under low entropy to ensure training stability. We evaluate SWE-Fuse on the widely studied SWE-bench Verified benchmark shows to demonstrate its effectiveness in solving real-world software problems. Specifically, SWE-Fuse outperforms the best 8B and 32B baselines by 43.0\% and 60.2\% in solve rate, respectively. Furthermore, integrating SWE-Fuse with test-time scaling (TTS) enables further performance improvements, achieving solve rates of 49.8\% and 65.2\% under TTS@8 for the 8B and 32B models, respectively. 

---
# IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation 

**Authors**: Sunghyun Baek, Jaemyung Yu, Seunghee Koh, Minsu Kim, Hyeonseong Jeon, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07926)  

**Abstract**: Test-time adaptation (TTA) has been widely explored to prevent performance degradation when test data differ from the training distribution. However, fully leveraging the rich representations of large pretrained models with minimal parameter updates remains underexplored. In this paper, we propose Intrinsic Mixture of Spectral Experts (IMSE) that leverages the spectral experts inherently embedded in Vision Transformers. We decompose each linear layer via singular value decomposition (SVD) and adapt only the singular values, while keeping the singular vectors fixed. We further identify a key limitation of entropy minimization in TTA: it often induces feature collapse, causing the model to rely on domain-specific features rather than class-discriminative features. To address this, we propose a diversity maximization loss based on expert-input alignment, which encourages diverse utilization of spectral experts during adaptation. In the continual test-time adaptation (CTTA) scenario, beyond preserving pretrained knowledge, it is crucial to retain and reuse knowledge from previously observed domains. We introduce Domain-Aware Spectral Code Retrieval, which estimates input distributions to detect domain shifts, and retrieves adapted singular values for rapid adaptation. Consequently, our method achieves state-of-the-art performance on various distribution-shift benchmarks under the TTA setting. In CTTA and Gradual CTTA, it further improves accuracy by 3.4 percentage points (pp) and 2.4 pp, respectively, while requiring 385 times fewer trainable parameters. Our code is available at this https URL. 

---
# Long-Short Term Agents for Pure-Vision Bronchoscopy Robotic Autonomy 

**Authors**: Junyang Wu, Mingyi Luo, Fangfang Xie, Minghui Zhang, Hanxiao Zhang, Chunxi Zhang, Junhao Wang, Jiayuan Sun, Yun Gu, Guang-Zhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07909)  

**Abstract**: Accurate intraoperative navigation is essential for robot-assisted endoluminal intervention, but remains difficult because of limited endoscopic field of view and dynamic artifacts. Existing navigation platforms often rely on external localization technologies, such as electromagnetic tracking or shape sensing, which increase hardware complexity and remain vulnerable to intraoperative anatomical mismatch. We present a vision-only autonomy framework that performs long-horizon bronchoscopic navigation using preoperative CT-derived virtual targets and live endoscopic video, without external tracking during navigation. The framework uses hierarchical long-short agents: a short-term reactive agent for continuous low-latency motion control, and a long-term strategic agent for decision support at anatomically ambiguous points. When their recommendations conflict, a world-model critic predicts future visual states for candidate actions and selects the action whose predicted state best matches the target view. We evaluated the system in a high-fidelity airway phantom, three ex vivo porcine lungs, and a live porcine model. The system reached all planned segmental targets in the phantom, maintained 80\% success to the eighth generation ex vivo, and achieved in vivo navigation performance comparable to the expert bronchoscopist. These results support the preclinical feasibility of sensor-free autonomous bronchoscopic navigation. 

---
# Designing probabilistic AI monsoon forecasts to inform agricultural decision-making 

**Authors**: Colin Aitken, Rajat Masiwal, Adam Marchakitus, Katherine Kowal, Mayank Gupta, Tyler Yang, Amir Jina, Pedram Hassanzadeh, William R. Boos, Michael Kremer  

**Link**: [PDF](https://arxiv.org/pdf/2603.07893)  

**Abstract**: Hundreds of millions of farmers make high-stakes decisions under uncertainty about future weather. Forecasts can inform these decisions, but available choices and their risks and benefits vary between farmers. We introduce a decision-theory framework for designing useful forecasts in settings where the forecaster cannot prescribe optimal actions because farmers' circumstances are heterogeneous. We apply this framework to the case of seasonal onset of monsoon rains, a key date for planting decisions and agricultural investments in many tropical countries. We develop a system for tailoring forecasts to the requirements of this framework by blending systematically benchmarked artificial intelligence (AI) weather prediction models with a new "evolving farmer expectations" statistical model. This statistical model applies Bayesian inference to historical observations to predict time-varying probabilities of first-occurrence events throughout a season. The blended system yields more skillful Indian monsoon forecasts at longer lead times than its components or any multi-model average. In 2025, this system was deployed operationally in a government-led program that delivered subseasonal monsoon onset forecasts to 38 million Indian farmers, skillfully predicting that year's early-summer anomalous dry period. This decision-theory framework and blending system offer a pathway for developing climate adaptation tools for large vulnerable populations around the world. 

---
# VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning? 

**Authors**: Minkyu Kim, Sangheon Lee, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.07888)  

**Abstract**: The ability to distinguish subtle differences between visually similar images is essential for diverse domains such as industrial anomaly detection, medical imaging, and aerial surveillance. While comparative reasoning benchmarks for vision-language models (VLMs) have recently emerged, they primarily focus on images with large, salient differences and fail to capture the nuanced reasoning required for real-world applications. In this work, we introduce VLM-SubtleBench, a benchmark designed to evaluate VLMs on subtle comparative reasoning. Our benchmark covers ten difference types - Attribute, State, Emotion, Temporal, Spatial, Existence, Quantity, Quality, Viewpoint, and Action - and curate paired question-image sets reflecting these fine-grained variations. Unlike prior benchmarks restricted to natural image datasets, our benchmark spans diverse domains, including industrial, aerial, and medical imagery. Through extensive evaluation of both proprietary and open-source VLMs, we reveal systematic gaps between model and human performance across difference types and domains, and provide controlled analyses highlighting where VLMs' reasoning sharply deteriorates. Together, our benchmark and findings establish a foundation for advancing VLMs toward human-level comparative reasoning. 

---
# Reject, Resample, Repeat: Understanding Parallel Reasoning in Language Model Inference 

**Authors**: Noah Golowich, Fan Chen, Dhruv Rohatgi, Raghav Singhal, Carles Domingo-Enrich, Dylan J. Foster, Akshay Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2603.07887)  

**Abstract**: Inference-time methods that aggregate and prune multiple samples have emerged as a powerful paradigm for steering large language models, yet we lack any principled understanding of their accuracy-cost tradeoffs. In this paper, we introduce a route to rigorously study such approaches using the lens of *particle filtering* algorithms such as Sequential Monte Carlo (SMC). Given a base language model and a *process reward model* estimating expected terminal rewards, we ask: *how accurately can we sample from a target distribution given some number of process reward evaluations?* Theoretically, we identify (1) simple criteria enabling non-asymptotic guarantees for SMC; (2) algorithmic improvements to SMC; and (3) a fundamental limit faced by all particle filtering methods. Empirically, we demonstrate that our theoretical criteria effectively govern the *sampling error* of SMC, though not necessarily its final *accuracy*, suggesting that theoretical perspectives beyond sampling may be necessary. 

---
# CCR-Bench: A Comprehensive Benchmark for Evaluating LLMs on Complex Constraints, Control Flows, and Real-World Cases 

**Authors**: Xiaona Xue, Yiqiao Huang, Jiacheng Li, Yuanhang Zheng, Huiqi Miao, Yunfei Ma, Rui Liu, Xinbao Sun, Minglu Liu, Fanyu Meng, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.07886)  

**Abstract**: Enhancing the ability of large language models (LLMs) to follow complex instructions is critical for their deployment in real-world applications. However, existing evaluation methods often oversimplify instruction complexity as a mere additive combination of atomic constraints, failing to adequately capture the high-dimensional complexity arising from the intricate interplay of content and format, logical workflow control, and real-world applications. This leads to a significant gap between current evaluation practices and practical demands. To bridge this gap, we introduce CCR-Bench, a novel benchmark designed to assess LLMs' adherence to complex instructions. CCR-Bench is characterized by: (1) deep entanglement of content and formatting requirements in task specifications; (2) instructions that involve intricate task decomposition, conditional reasoning, and procedural planning; and (3) evaluation samples derived entirely from real-world industrial scenarios. Extensive experiments on CCR-Bench demonstrate that even state-of-the-art models exhibit substantial performance deficiencies, clearly quantifying the gap between current LLM capabilities and the demands of realworld instruction understanding. We believe that CCR-Bench offers a more rigorous and realistic evaluation framework, advancing the development of LLMs toward the next generation of models capable of understanding and executing complex tasks in industrial applications. 

---
# Slumbering to Precision: Enhancing Artificial Neural Network Calibration Through Sleep-like Processes 

**Authors**: Jean Erik Delanois, Aditya Ahuja, Giri P. Krishnan, Maxim Bazhenov  

**Link**: [PDF](https://arxiv.org/pdf/2603.07867)  

**Abstract**: Artificial neural networks are often overconfident, undermining trust because their predicted probabilities do not match actual accuracy. Inspired by biological sleep and the role of spontaneous replay in memory and learning, we introduce Sleep Replay Consolidation (SRC), a novel calibration approach. SRC is a post-training, sleep-like phase that selectively replays internal representations to update network weights and improve calibration without supervised retraining. Across multiple experiments, SRC is competitive with and complementary to standard approaches such as temperature scaling. Combining SRC with temperature scaling achieves the best Brier score and entropy trade-offs for AlexNet and VGG19. These results show that SRC provides a fundamentally novel approach to improving neural network calibration. SRC-based calibration offers a practical path toward more trustworthy confidence estimates and narrows the gap between human-like uncertainty handling and modern deep networks. 

---
# AI Steerability 360: A Toolkit for Steering Large Language Models 

**Authors**: Erik Miehling, Karthikeyan Natesan Ramamurthy, Praveen Venkateswaran, Irene Ko, Pierre Dognin, Moninder Singh, Tejaswini Pedapati, Avinash Balakrishnan, Matthew Riemer, Dennis Wei, Inge Vejsbjerg, Elizabeth M. Daly, Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2603.07837)  

**Abstract**: The AI Steerability 360 toolkit is an extensible, open-source Python library for steering LLMs. Steering abstractions are designed around four model control surfaces: input (modification of the prompt), structural (modification of the model's weights or architecture), state (modification of the model's activations and attentions), and output (modification of the decoding or generation process). Steering methods exert control on the model through a common interface, termed a steering pipeline, which additionally allows for the composition of multiple steering methods. Comprehensive evaluation and comparison of steering methods/pipelines is facilitated by use case classes (for defining tasks) and a benchmark class (for performance comparison on a given task). The functionality provided by the toolkit significantly lowers the barrier to developing and comprehensively evaluating steering methods. The toolkit is Hugging Face native and is released under an Apache 2.0 license at this https URL. 

---
# DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation 

**Authors**: Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07835)  

**Abstract**: Knowledge distillation from proprietary LLM APIs poses a growing threat to model providers, yet defenses against this attack remain fragmented and unevaluated. We present DistillGuard, a framework for systematically evaluating output-level defenses against LLM knowledge distillation. We introduce a taxonomy of three defense categories -- output perturbation, data poisoning, and information throttling -- and evaluate nine defense configurations using a standardized pipeline with Qwen3-14B as teacher and Qwen2.5-7B-Instruct as student across three benchmarks (MATH-500, HumanEval+, MT-Bench). Our results reveal that, in a same-family distillation setting against a naive attacker, most output-level defenses are surprisingly ineffective: paraphrasing-based perturbation barely degrades distilled student quality, and data poisoning primarily impairs conversational fluency while leaving task-specific capabilities intact. Only chain-of-thought removal substantially impairs mathematical reasoning (31.4\% vs.\ 67.8\% baseline), though code generation remains unaffected. These findings demonstrate that the effectiveness of distillation defenses is highly task-dependent and that current output-level approaches are insufficient to broadly prevent knowledge theft. 

---
# AI Misuse in Education Is a Measurement Problem: Toward a Learning Visibility Framework 

**Authors**: Eduardo Davalos, Yike Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07834)  

**Abstract**: The rapid integration of conversational AI systems into educational settings has intensified ethical concerns about academic integrity, fairness, and students' cognitive development. Institutional responses have largely centered on AI detection tools and restrictive policies, yet such approaches have proven unreliable and ethically contentious. This paper reframes AI misuse in education not primarily as a detection problem, but as a measurement problem rooted in the loss of visibility into the learning process. When AI enters the assessment loop, educators often retain access to final outputs but lose valuable insight into how those outputs were produced. Drawing on research in cognitive offloading, learning analytics, and multimodal timeline reconstruction, we propose the Learning Visibility Framework, grounded in three principles: clear specification and modeling of acceptable AI use, recognition of learning processes as assessable evidence alongside outcomes, and the establishment of transparent timelines of student activity. Rather than promoting surveillance, the framework emphasizes transparency and shared evidence as foundations for ethical AI integration in classroom settings. By shifting focus from adversarial detection toward process visibility, this work offers a principled pathway for aligning AI use with educational values while preserving trust and transparency between students and educators 

---
# Gradient Iterated Temporal-Difference Learning 

**Authors**: Théo Vincent, Kevin Gerhardt, Yogesh Tripathi, Habib Maraqten, Adam White, Martha White, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2603.07833)  

**Abstract**: Temporal-difference (TD) learning is highly effective at controlling and evaluating an agent's long-term outcomes. Most approaches in this paradigm implement a semi-gradient update to boost the learning speed, which consists of ignoring the gradient of the bootstrapped estimate. While popular, this type of update is prone to divergence, as Baird's counterexample illustrates. Gradient TD methods were introduced to overcome this issue, but have not been widely used, potentially due to issues with learning speed compared to semi-gradient methods. Recently, iterated TD learning was developed to increase the learning speed of TD methods. For that, it learns a sequence of action-value functions in parallel, where each function is optimized to represent the application of the Bellman operator over the previous function in the sequence. While promising, this algorithm can be unstable due to its semi-gradient nature, as each function tracks a moving target. In this work, we modify iterated TD learning by computing the gradients over those moving targets, aiming to build a powerful gradient TD method that competes with semi-gradient methods. Our evaluation reveals that this algorithm, called Gradient Iterated Temporal-Difference learning, has a competitive learning speed against semi-gradient methods across various benchmarks, including Atari games, a result that no prior work on gradient TD methods has demonstrated. 

---
# Column Generation for the Micro-Transit Zoning Problem 

**Authors**: Hins Hu, Rishav Sen, Jose Paolo Talusan, Abhishek Dubey, Aron Laszka, Samitha Samaranayake  

**Link**: [PDF](https://arxiv.org/pdf/2603.07821)  

**Abstract**: Along with the rapid development of new urban mobility options like ride-sharing over the past decade, on-demand micro-transit services stand out as a middle ground, bridging the gap between fixed-line mass transit and single-request ride-hailing, balancing ridership maximization and travel time minimization. Micro-transit adoption can have significant social impact. It improves urban sustainability, through lower energy consumption and reduced emissions, while enhancing equitable mobility access for disadvantaged communities, thanks to its lower vehicle miles per passenger, flexible schedules, and affordable pricing. However, effective operation of micro-transit services requires planning geo-fenced zones in advance, which involves solving a challenging combinatorial optimization problem. Existing approaches enumerate candidate zones first and selects a fixed number of optimal zones in the second step. In this paper, we generalize the Micro-Transit Zoning Problem (MZP) to allow a global budget rather than imposing a size limit for candidate zones. We also design a Column Generation (CG) framework to solve the problem and several pricing heuristics to accelerate computation. Extensive numerical experiments across major U.S. cities demonstrate that our approach produces higher-quality solutions more efficiently and scales better in the generalized setting. 

---
# HybridStitch: Pixel and Timestep Level Model Stitching for Diffusion Acceleration 

**Authors**: Desen Sun, Jason Hon, Jintao Zhang, Sihang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07815)  

**Abstract**: Diffusion models have demonstrated a remarkable ability in Text-to-Image (T2I) generation applications. Despite the advanced generation output, they suffer from heavy computation overhead, especially for large models that contain tens of billions of parameters. Prior work has illustrated that replacing part of the denoising steps with a smaller model still maintains the generation quality. However, these methods only focus on saving computation for some timesteps, ignoring the difference in compute demand within one timestep. In this work, we propose HybridStitch, a new T2I generation paradigm that treats generation like editing. Specifically, we introduce a hybrid stage that jointly incorporates both the large model and the small model. HybridStitch separates the entire image into two regions: one that is relatively easy to render, enabling an early transition to the smaller model, and another that is more complex and therefore requires refinement by the large model. HybridStitch employs the small model to construct a coarse sketch while exploiting the large model to edit and refine the complex regions. According to our evaluation, HybridStitch achieves 1.83$\times$ speedup on Stable Diffusion 3, which is faster than all existing mixture of model methods. 

---
# Learning embeddings of non-linear PDEs: the Burgers' equation 

**Authors**: Pedro Tarancón-Álvarez, Leonid Sarieddine, Pavlos Protopapas, Raul Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2603.07812)  

**Abstract**: Embeddings provide low-dimensional representations that organize complex function spaces and support generalization. They provide a geometric representation that supports efficient retrieval, comparison, and generalization. In this work we generalize the concept to Physics Informed Neural Networks. We present a method to construct solution embedding spaces of nonlinear partial differential equations using a multi-head setup, and extract non-degenerate information from them using principal component analysis (PCA). We test this method by applying it to viscous Burgers' equation, which is solved simultaneously for a family of initial conditions and values of the viscosity. A shared network body learns a latent embedding of the solution space, while linear heads map this embedding to individual realizations. By enforcing orthogonality constraints on the heads, we obtain a principal-component decomposition of the latent space that is robust to training degeneracies and admits a direct physical interpretation. The obtained components for Burgers' equation exhibit rapid saturation, indicating that a small number of latent modes captures the dominant features of the dynamics. 

---
# Dual-Metric Evaluation of Social Bias in Large Language Models: Evidence from an Underrepresented Nepali Cultural Context 

**Authors**: Ashish Pandey, Tek Raj Chhetri  

**Link**: [PDF](https://arxiv.org/pdf/2603.07792)  

**Abstract**: Large language models (LLMs) increasingly influence global digital ecosystems, yet their potential to perpetuate social and cultural biases remains poorly understood in underrepresented contexts. This study presents a systematic analysis of representational biases in seven state-of-the-art LLMs: GPT-4o-mini, Claude-3-Sonnet, Claude-4-Sonnet, Gemini-2.0-Flash, Gemini-2.0-Lite, Llama-3-70B, and Mistral-Nemo in the Nepali cultural context. Using Croissant-compliant dataset of 2400+ stereotypical and anti-stereotypical sentence pairs on gender roles across social domains, we implement an evaluation framework, Dual-Metric Bias Assessment (DMBA), combining two metrics: (1) agreement with biased statements and (2) stereotypical completion tendencies. Results show models exhibit measurable explicit agreement bias, with mean bias agreement ranging from 0.36 to 0.43 across decoding configurations, and an implicit completion bias rate of 0.740-0.755. Importantly, implicit completion bias follows a non-linear, U-shaped relationship with temperature, peaking at moderate stochasticity (T=0.3) and declining slightly at higher temperatures. Correlation analysis under different decoding settings revealed that explicit agreement strongly aligns with stereotypical sentence agreement but is a weak and often negative predictor of implicit completion bias, indicating generative bias is poorly captured by agreement metrics. Sensitivity analysis shows increasing top-p amplifies explicit bias, while implicit generative bias remains largely stable. Domain-level analysis shows implicit bias is strongest for race and sociocultural stereotypes, while explicit agreement bias is similar across gender and sociocultural categories, with race showing the lowest explicit agreement. These findings highlight the need for culturally grounded datasets and debiasing strategies for LLMs in underrepresented societies. 

---
# ProgAgent:A Continual RL Agent with Progress-Aware Rewards 

**Authors**: Jinzhou Tan, Gabriel Adineera, Jinoh Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07784)  

**Abstract**: We present ProgAgent, a continual reinforcement learning (CRL) agent that unifies progress-aware reward learning with a high-throughput, JAX-native system architecture. Lifelong robotic learning grapples with catastrophic forgetting and the high cost of reward specification. ProgAgent tackles these by deriving dense, shaped rewards from unlabeled expert videos through a perceptual model that estimates task progress across initial, current, and goal observations. We theoretically interpret this as a learned state-potential function, delivering robust guidance in line with expert behaviors. To maintain stability amid online exploration - where novel, out-of-distribution states arise - we incorporate an adversarial push-back refinement that regularizes the reward model, curbing overconfident predictions on non-expert trajectories and countering distribution shift. By embedding this reward mechanism into a JIT-compiled loop, ProgAgent supports massively parallel rollouts and fully differentiable updates, rendering a sophisticated unified objective feasible: it merges PPO with coreset replay and synaptic intelligence for an enhanced stability-plasticity balance. Evaluations on ContinualBench and Meta-World benchmarks highlight ProgAgent's advantages: it markedly reduces forgetting, boosts learning speed, and outperforms key baselines in visual reward learning (e.g., Rank2Reward, TCN) and continual learning (e.g., Coreset, SI) - surpassing even an idealized perfect memory agent. Real-robot trials further validate its ability to acquire complex manipulation skills from noisy, few-shot human demonstrations. 

---
# QuadAI at SemEval-2026 Task 3: Ensemble Learning of Hybrid RoBERTa and LLMs for Dimensional Aspect-Based Sentiment Analysis 

**Authors**: A.J.W. de Vink, Filippos Karolos Ventirozos, Natalia Amat-Lefort, Lifeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.07766)  

**Abstract**: We present our system for SemEval-2026 Task 3 on dimensional aspect-based sentiment regression. Our approach combines a hybrid RoBERTa encoder, which jointly predicts sentiment using regression and discretized classification heads, with large language models (LLMs) via prediction-level ensemble learning. The hybrid encoder improves prediction stability by combining continuous and discretized sentiment representations. We further explore in-context learning with LLMs and ridge-regression stacking to combine encoder and LLM predictions. Experimental results on the development set show that ensemble learning significantly improves performance over individual models, achieving substantial reductions in RMSE and improvements in correlation scores. Our findings demonstrate the complementary strengths of encoder-based and LLM-based approaches for dimensional sentiment analysis. Our development code and resources will be shared at this https URL 

---
# DECADE: A Temporally-Consistent Unsupervised Diffusion Model for Enhanced Rb-82 Dynamic Cardiac PET Image Denoising 

**Authors**: Yinchi Zhou, Liang Guo, Huidong Xie, Yuexi Du, Ashley Wang, Menghua Xia, Tian Yu, Ramesh Fazzone-Chettiar, Christopher Weyman, Bruce Spottiswoode, Vladimir Panin, Kuangyu Shi, Edward J. Miller, Attila Feher, Albert J. Sinusas, Nicha C. Dvornek, Chi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07759)  

**Abstract**: Rb-82 dynamic cardiac PET imaging is widely used for the clinical diagnosis of coronary artery disease (CAD), but its short half-life results in high noise levels that degrade dynamic frame quality and parametric imaging. The lack of paired clean-noisy training data, rapid tracer kinetics, and frame-dependent noise variations further limit the effectiveness of existing deep learning denoising methods. We propose DECADE (A Temporally-Consistent Unsupervised Diffusion model for Enhanced Rb-82 CArdiac PET DEnoising), an unsupervised diffusion framework that generalizes across early- to late-phase dynamic frames. DECADE incorporates temporal consistency during both training and iterative sampling, using noisy frames as guidance to preserve quantitative accuracy. The method was trained and evaluated on datasets acquired from Siemens Vision 450 and Siemens Biograph Vision Quadra scanners. On the Vision 450 dataset, DECADE consistently produced high-quality dynamic and parametric images with reduced noise while preserving myocardial blood flow (MBF) and myocardial flow reserve (MFR). On the Quadra dataset, using 15%-count images as input and full-count images as reference, DECADE outperformed UNet-based and other diffusion models in image quality and K1/MBF quantification. The proposed framework enables effective unsupervised denoising of Rb-82 dynamic cardiac PET without paired training data, supporting clearer visualization while maintaining quantitative integrity. 

---
# Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning 

**Authors**: Jinshan Liu, Ken Li, Jiazhe Wei, Bin Shi, Bo Dong  

**Link**: [PDF](https://arxiv.org/pdf/2603.07743)  

**Abstract**: Federated Graph Learning (FedGL) is vulnerable to malicious attacks, yet developing a truly effective and stealthy attack method remains a significant challenge. Existing attack methods suffer from low attack success rates, high computational costs, and are easily identified and smoothed by defense algorithms. To address these challenges, we propose \textbf{FedShift}, a novel two-stage "Hide and Find" distributed adversarial attack. In the first stage, before FedGL begins, we inject a learnable and hidden "shifter" into part of the training data, which subtly pushes poisoned graph representations toward a target class's decision boundary without crossing it, ensuring attack stealthiness during training. In the second stage, after FedGL is complete, we leverage the global model information and use the hidden shifter as an optimization starting point to efficiently find the adversarial perturbations. During the final attack, we aggregate these perturbations from multiple malicious clients to form the final effective adversarial sample and trigger the attack. Extensive experiments on six large-scale datasets demonstrate that our method achieves the highest attack effectiveness compared to existing advanced attack methods. In particular, our attack can effectively evade 3 mainstream robust federated learning defense algorithms and converges with a time cost reduction of over 90\%, highlighting its exceptional stealthiness, robustness, and efficiency. 

---
# YAQIN: Culturally Sensitive, Agentic AI for Mental Healthcare Support Among Muslim Women in the UK 

**Authors**: Yasmin Zaraket, Céline Mougenot  

**Link**: [PDF](https://arxiv.org/pdf/2603.07709)  

**Abstract**: Mental healthcare services in the UK lack tools and resources to address the cultural needs of Muslim women, often leaving them feeling as though their values are pathologised and limiting trust and engagement [1]. Despite growing awareness of cultural competency, few interventions integrate Islamic frameworks into therapeutic support. This report investigates the design and evaluation of YAQIN, a co-designed AI-based application supporting culturally and faith-sensitive mental health engagement for Muslim women. With almost 1.9 million Muslim women in England in 2021, YAQIN responds to a gap in care [2]. It leverages AIś anonymity and continuous support through a faith-aware chatbot and guided journaling tool grounded in user-centred design and Islamic psychology.
The YAQIN design research methodology comprised three stages: contextual investigation and literature review, user research with N=14 stakeholders including Muslim women and mental health experts, and prototype development informed by deductive thematic analysis, personas, journey maps, and design specifications.
Evaluation involved a co-designed user study with five participants: four Muslim women and one mental health expert who reviewed therapeutic alignment and cultural sensitivity after using the chatbot prototype. Feedback focused on tone, faith relevance, emotional resonance, and the Retrieval-Augmented Generation pipeline enabling contextual continuity.
Participants highlighted YAQINś ability to bridge cultural gaps in trust and therapeutic confidence. Feedback included suggestions of including linguistic diversity and routine-based guidance.
This project demonstrates how culturally sensitive AI can improve mental healthcare accessibility and trust for marginalised communities and highlights the potential of faith-integrated technology in healthcare innovation. 

---
# VoiceSHIELD-Small: Real-Time Malicious Speech Detection and Transcription 

**Authors**: Sumit Ranjan, Sugandha Sharma, Ubaid Abbas, Puneeth N Ail  

**Link**: [PDF](https://arxiv.org/pdf/2603.07708)  

**Abstract**: Voice interfaces are quickly becoming a common way for people to interact with AI systems. This also brings new security risks, such as prompt injection, social engineering, and harmful voice commands. Traditional security methods rely on converting speech to text and then filtering that text, which introduces delays and can ignore important audio cues. This paper introduces VoiceSHIELD-Small, a lightweight model that works in real time. It can transcribe speech and detect whether it is safe or harmful, all in one step. Built on OpenAI's Whisper-small encoder, VoiceSHIELD adds a mean-pooling layer and a simple classification head. It takes just 90-120 milliseconds to classify audio on mid-tier GPUs, while transcription happens at the same time. Tested on a balanced set of 947 audio clips, the model achieved 99.16 percent accuracy and an F1 score of 0.9865. At the default setting, it missed 2.33 percent of harmful inputs. Cross-validation showed consistent performance (F1 standard deviation = 0.0026). The paper also covers the model's design, training data, performance trade-offs, and responsible use guidelines. VoiceSHIELD is released under the MIT license to encourage further research and adoption in voice AI security. 

---
# TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward 

**Authors**: Yihong Luo, Tianyang Hu, Weijian Luo, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07700)  

**Abstract**: While few-step generative models have enabled powerful image and video generation at significantly lower cost, generic reinforcement learning (RL) paradigms for few-step models remain an unsolved problem. Existing RL approaches for few-step diffusion models strongly rely on back-propagating through differentiable reward models, thereby excluding the majority of important real-world reward signals, e.g., non-differentiable rewards such as humans' binary likeness, object counts, etc. To properly incorporate non-differentiable rewards to improve few-step generative models, we introduce TDM-R1, a novel reinforcement learning paradigm built upon a leading few-step model, Trajectory Distribution Matching (TDM). TDM-R1 decouples the learning process into surrogate reward learning and generator learning. Furthermore, we developed practical methods to obtain per-step reward signals along the deterministic generation trajectory of TDM, resulting in a unified RL post-training method that significantly improves few-step models' ability with generic rewards. We conduct extensive experiments ranging from text-rendering, visual quality, and preference alignment. All results demonstrate that TDM-R1 is a powerful reinforcement learning paradigm for few-step text-to-image models, achieving state-of-the-art reinforcement learning performances on both in-domain and out-of-domain metrics. Furthermore, TDM-R1 also scales effectively to the recent strong Z-Image model, consistently outperforming both its 100-NFE and few-step variants with only 4 NFEs. Project page: this https URL 

---
# Compressed-Domain-Aware Online Video Super-Resolution 

**Authors**: Yuhang Wang, Hai Li, Shujuan Hou, Zhetao Dong, Xiaoyao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07694)  

**Abstract**: In bandwidth-limited online video streaming, videos are usually downsampled and compressed. Although recent online video super-resolution (online VSR) approaches achieve promising results, they are still compute-intensive and fall short of real-time processing at higher resolutions, due to complex motion estimation for alignment and redundant processing of consecutive frames. To address these issues, we propose a compressed-domain-aware network (CDA-VSR) for online VSR, which utilizes compressed-domain information, including motion vectors, residual maps, and frame types to balance quality and efficiency. Specifically, we propose a motion-vector-guided deformable alignment module that uses motion vectors for coarse warping and learns only local residual offsets for fine-tuned adjustments, thereby maintaining accuracy while reducing computation. Then, we utilize a residual map gated fusion module to derive spatial weights from residual maps, suppressing mismatched regions and emphasizing reliable details. Further, we design a frame-type-aware reconstruction module for adaptive compute allocation across frame types, balancing accuracy and efficiency. On the REDS4 dataset, our CDA-VSR surpasses the state-of-the-art method TMP, with a maximum PSNR improvement of 0.13 dB while delivering more than double the inference speed. The code will be released at this https URL. 

---
# AI-Driven Phase Identification from X-ray Hyperspectral Imaging of cycled Na-ion Cathode Materials 

**Authors**: Fayçal Adrar, Nicolas Folastre, Chloé Pablos, Stefan Stanescu, Sufal Swaraj, Raghvender Raghvender, François Cadiou, Laurence Croguennec, Matthieu Bugnet, Arnaud Demortière  

**Link**: [PDF](https://arxiv.org/pdf/2603.07666)  

**Abstract**: Na-ion batteries have emerged as viable candidates for large-scale energy storage applica- tions due to resource abundance and cost advantages. The constraints imposed on their performance and durability, for instance, by complex phase transformations in positive electrode materials during electrochemical cycling, can be addressed and are thus not detrimental to their development. However, diffusion-limited Na-ion transport can drive spatially heterogeneous phase nucleation and propagation, leading to multiphase coexis- tence and locally non-uniform electrochemical activity, generating complex reaction path- ways that challenge both mechanistic understanding and predictive material optimization. These challenges can be addressed by investigating single-crystalline regions of materials, i.e. down to the scale of individual particles, although such analyses are often constrained by energetically and/or spatially sparse hyperspectral datasets. Here, we developed an AI-driven method to process hyperspectral data under sparse sampling conditions and generate multiphase maps with nanometer-scale resolution over a micrometer-scale field of view. We applied this processing on scanning transmission X-ray microscopy (STXM) data to determine the distribution and coexistence of phases in individual particles of NaxV2(PO4)2F3 cathode materials, at different states of charge. The methodology relies on a workflow which combines a Gaussian mixture variational autoencoder (GMVAE) algorithm with the Pearson corre- lation coefficient to identify the sodium content and map their spatial distribution. Our approach reveals nanoscale phase heterogeneity and evolution within individual particles, and improves the reliability of phase detection by identifying ambiguity zones, false assign- ments, and transition phases localized at grain boundaries. 

---
# Ref-DGS: Reflective Dual Gaussian Splatting 

**Authors**: Ningjing Fan, Yiqun Wang, Dongming Yan, Peter Wonka  

**Link**: [PDF](https://arxiv.org/pdf/2603.07664)  

**Abstract**: Reflective appearance, especially strong and typically near-field specular reflections, poses a fundamental challenge for accurate surface reconstruction and novel view synthesis. Existing Gaussian splatting methods either fail to model near-field specular reflections or rely on explicit ray tracing at substantial computational cost. We present Ref-DGS, a reflective dual Gaussian splatting framework that addresses this trade-off by decoupling surface reconstruction from specular reflection within an efficient rasterization-based pipeline. Ref-DGS introduces a dual Gaussian scene representation consisting of geometry Gaussians and complementary local reflection Gaussians that capture near-field specular interactions without explicit ray tracing, along with a global environment reflection field for modeling far-field specular reflections. To predict specular radiance, we further propose a lightweight, physically-aware adaptive mixing shader that fuses global and local reflection features. Experiments demonstrate that Ref-DGS achieves state-of-the-art performance on reflective scenes while training substantially faster than ray-based Gaussian methods. 

---
# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots 

**Authors**: Likui Zhang, Tao Tang, Zhihao Zhan, Xiuwei Chen, Zisheng Chen, Jianhua Han, Jiangtong Zhu, Pei Xu, Hang Xu, Hefeng Wu, Liang Lin, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07648)  

**Abstract**: Recent advances in Visual-Language-Action (VLA) models have shown promising potential for robotic manipulation tasks. However, real-world robotic tasks often involve long-horizon, multi-step problem-solving and require generalization for continual skill acquisition, extending beyond single actions or skills. These challenges present significant barriers for existing VLA models, which use monolithic action decoders trained on aggregated data, resulting in poor scalability. To address these challenges, we propose AtomicVLA, a unified planning-and-execution framework that jointly generates task-level plans, atomic skill abstractions, and fine-grained actions. AtomicVLA constructs a scalable atomic skill library through a Skill-Guided Mixture-of-Experts (SG-MoE), where each expert specializes in mastering generic yet precise atomic skills. Furthermore, we introduce a flexible routing encoder that automatically assigns dedicated atomic experts to new skills, enabling continual learning. We validate our approach through extensive experiments. In simulation, AtomicVLA outperforms $\pi_{0}$ by 2.4\% on LIBERO, 10\% on LIBERO-LONG, and outperforms $\pi_{0}$ and $\pi_{0.5}$ by 0.22 and 0.25 in average task length on CALVIN. Additionally, our AtomicVLA consistently surpasses baselines by 18.3\% and 21\% in real-world long-horizon tasks and continual learning. These results highlight the effectiveness of atomic skill abstraction and dynamic expert composition for long-horizon and lifelong robotic tasks. The project page is \href{this https URL}{here}. 

---
# Evaluating Synthetic Data for Baggage Trolley Detection in Airport Logistics 

**Authors**: Abdeldjalil Taibi, Mohmoud Badlis, Amina Bensalem, Belkacem Zouilekh, Mohammed Brahimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.07645)  

**Abstract**: Efficient luggage trolley management is critical for reducing congestion and ensuring asset availability in modern airports. Automated detection systems face two main challenges. First, strict security and privacy regulations limit large-scale data collection. Second, existing public datasets lack the diversity, scale, and annotation quality needed to handle dense, overlapping trolley arrangements typical of real-world operations.
To address these limitations, we introduce a synthetic data generation pipeline based on a high-fidelity Digital Twin of Algiers International Airport using NVIDIA Omniverse. The pipeline produces richly annotated data with oriented bounding boxes, capturing complex trolley formations, including tightly nested chains. We evaluate YOLO-OBB using five training strategies: real-only, synthetic-only, linear probing, full fine-tuning, and mixed training. This allows us to assess how synthetic data can complement limited real-world annotations.
Our results show that mixed training with synthetic data and only 40 percent of real annotations matches or exceeds the full real-data baseline, achieving 0.94 mAP@50 and 0.77 mAP@50-95, while reducing annotation effort by 25 to 35 percent. Multi-seed experiments confirm strong reproducibility with a standard deviation below 0.01 on mAP@50, demonstrating the practical effectiveness of synthetic data for automated trolley detection. 

---
# SMAT: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control 

**Authors**: Yifei Yuan, Ghaith Androwis, Xianlian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.07618)  

**Abstract**: Effective exoskeleton assistance requires co-adaptation: as the device alters joint dynamics, the user reorganizes neuromuscular coordination, creating a non-stationary learning problem. Most learning-based approaches do not explicitly account for the sequential nature of human motor adaptation, leading to training instability and poorly timed assistance. We propose Staged Multi-Agent Training (SMAT), a four-stage curriculum designed to mirror how users naturally acclimate to a wearable device. In SMAT, a musculoskeletal human actor and a bilateral hip exoskeleton actor are trained progressively: the human first learns unassisted gait, then adapts to the added device mass; the exoskeleton subsequently learns a positive assistance pattern against a stabilized human policy, and finally both agents co-adapt with full torque capacity and bidirectional feedback. We implement SMAT in the MyoAssist simulation environment using a 26-muscle lower-limb model and an attached hip exoskeleton. Our musculoskeletal simulations demonstrate that the learned exoskeleton control policy produces an average 10.1% reduction in hip muscle activation relative to the no-assist condition. We validated the learned controller in an offline setting using open-source gait data, then deployed it to a physical hip exoskeleton for treadmill experiments with five subjects. The resulting policy delivers consistent assistance and predominantly positive mechanical power without the need for any explicitly imposed timing shift (mean positive power: 13.6 W at 6 Nm RMS torque to 23.8 W at 9.3 Nm RMS torque, with minimal negative power) consistently across all subjects without subject-specific retraining. 

---
# Integration of deep generative Anomaly Detection algorithm in high-speed industrial line 

**Authors**: Niccolò Ferrari, Nicola Zanarini, Michele Fraccaroli, Alice Bizzarri, Evelina Lamma  

**Link**: [PDF](https://arxiv.org/pdf/2603.07577)  

**Abstract**: Industrial visual inspection in pharmaceutical production requires high accuracy under strict constraints on cycle time, hardware footprint, and operational cost. Manual inline inspection is still common, but it is affected by operator variability and limited throughput. Classical rule-based computer vision pipelines are often rigid and difficult to scale to highly variable production scenarios. To address these limitations, we present a semi-supervised anomaly detection framework based on a generative adversarial architecture with a residual autoencoder and a dense bottleneck, specifically designed for online deployment on a high-speed Blow-Fill-Seal (BFS) line. The model is trained only on nominal samples and detects anomalies through reconstruction residuals, providing both classification and spatial localization via heatmaps. The training set contains 2,815,200 grayscale patches. Experiments on a real industrial test kit show high detection performance while satisfying timing constraints compatible with a 500 ms acquisition slot. 

---
# A Systematic Comparison of Training Objectives for Out-of-Distribution Detection in Image Classification 

**Authors**: Furkan Genç, Onat Özdemir, Emre Akbaş  

**Link**: [PDF](https://arxiv.org/pdf/2603.07571)  

**Abstract**: Out-of-distribution (OOD) detection is critical in safety-sensitive applications. While this challenge has been addressed from various perspectives, the influence of training objectives on OOD behavior remains comparatively underexplored. In this paper, we present a systematic comparison of four widely used training objectives: Cross-Entropy Loss, Prototype Loss, Triplet Loss, and Average Precision (AP) Loss, spanning probabilistic, prototype-based, metric-learning, and ranking-based supervision, for OOD detection in image classification under standardized OpenOOD protocols. Across CIFAR-10/100 and ImageNet-200, we find that Cross-Entropy Loss, Prototype Loss, and AP Loss achieve comparable in-distribution accuracy, while Cross-Entropy Loss provides the most consistent near- and far-OOD performance overall; the other objectives can be competitive in specific settings. 

---
# GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module 

**Authors**: Niccolò Ferrari, Michele Fraccaroli, Evelina Lamma  

**Link**: [PDF](https://arxiv.org/pdf/2603.07566)  

**Abstract**: Anomaly detection is nowadays increasingly used in industrial applications and processes. One of the main fields of the appliance is the visual inspection for surface anomaly detection, which aims to spot regions that deviate from regularity and consequently identify abnormal products.
Defect localization is a key task, that usually is achieved using a basic comparison between generated image and the original one, implementing some blob-analysis or image-editing algorithms, in the post-processing step, which is very biased towards the source dataset, and they are unable to generalize. Furthermore, in industrial applications, the totality of the image is not always interesting but could be one or some regions of interest (ROIs), where only in those areas there are relevant anomalies to be spotted.
For these reasons, we propose a new architecture composed by two blocks.
The first block is a Generative Adversarial Network (GAN), based on a residual autoencoder (ResAE), to perform reconstruction and denoising processes, while the second block produces image segmentation, spotting defects. This method learns from a dataset composed of good products and generated synthetic defects.
The discriminative network is trained using a ROI for each image contained in the training dataset. The network will learn in which area anomalies are relevant. This approach guarantees the reduction of using pre-processing algorithms, formerly developed with blob-analysis and image-editing procedures.
To test our model we used challenging MVTec anomaly detection datasets and an industrial large dataset of pharmaceutical BFS strips of vials. This set constitutes a more realistic use case of the aforementioned network. 

---
# Nwāchā Munā: A Devanagari Speech Corpus and Proximal Transfer Benchmark for Nepal Bhasha ASR 

**Authors**: Rishikesh Kumar Sharma, Safal Narshing Shrestha, Jenny Poudel, Rupak Tiwari, Arju Shrestha, Rupak Raj Ghimire, Bal Krishna Bal  

**Link**: [PDF](https://arxiv.org/pdf/2603.07554)  

**Abstract**: Nepal Bhasha (Newari), an endangered language of the Kathmandu Valley, remains digitally marginalized due to the severe scarcity of annotated speech resources. In this work, we introduce Nwāchā Munā, a newly curated 5.39-hour manually transcribed Devanagari speech corpus for Nepal Bhasha, and establish the first benchmark using script-preserving acoustic modeling. We investigate whether proximal cross-lingual transfer from a geographically and linguistically adjacent language (Nepali) can rival large-scale multilingual pretraining in an ultra-low-resource Automatic Speech Recognition (ASR) setting. Fine-tuning a Nepali Conformer model reduces the Character Error Rate (CER) from a 52.54% zero-shot baseline to 17.59% with data augmentation, effectively matching the performance of the multilingual Whisper-Small model despite utilizing significantly fewer parameters. Our findings demonstrate that proximal transfer within South Asian language clusters serves as a computationally efficient alternative to massive multilingual models. We openly release the dataset and benchmarks to digitally enable the Newari community and foster further research in Nepal Bhasha. 

---
# Targeted Speaker Poisoning Framework in Zero-Shot Text-to-Speech 

**Authors**: Thanapat Trachu, Thanathai Lertpetchpun, Sai Praneeth Karimireddy, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2603.07551)  

**Abstract**: Zero-shot Text-to-Speech (TTS) voice cloning poses severe privacy risks, demanding the removal of specific speaker identities from trained TTS models. Conventional machine unlearning is insufficient in this context, as zero-shot TTS can dynamically reconstruct voices from just reference prompts. We formalize this task as Speech Generation Speaker Poisoning (SGSP), in which we modify trained models to prevent the generation of specific identities while preserving utility for other speakers. We evaluate inference-time filtering and parameter-modification baselines across 1, 15, and 100 forgotten speakers. Performance is assessed through the trade-off between utility (WER) and privacy, quantified using AUC and Forget Speaker Similarity (FSSIM). We achieve strong privacy for up to 15 speakers but reveal scalability limits at 100 speakers due to increased identity overlap. Our study thus introduces a novel problem and evaluation framework toward further advances in generative voice privacy. 

---
# Learning-free L2-Accented Speech Generation using Phonological Rules 

**Authors**: Thanathai Lertpetchpun, Yoonjeong Lee, Jihwan Lee, Tiantian Feng, Dani Byrd, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2603.07550)  

**Abstract**: Accent plays a crucial role in speaker identity and inclusivity in speech technologies. Existing accented text-to-speech (TTS) systems either require large-scale accented datasets or lack fine-grained phoneme-level controllability. We propose a accented TTS framework that combines phonological rules with a multilingual TTS model. The rules are applied to phoneme sequences to transform accent at the phoneme level while preserving intelligibility. The method requires no accented training data and enables explicit phoneme-level accent manipulation. We design rule sets for Spanish- and Indian-accented English, modeling systematic differences in consonants, vowels, and syllable structure arising from phonotactic constraints. We analyze the trade-off between phoneme-level duration alignment and accent as realized in speech timing. Experimental results demonstrate effective accent shift while maintaining speech quality. 

---
# DreamSAC: Learning Hamiltonian World Models via Symmetry Exploration 

**Authors**: Jinzhou Tang, Fan Feng, Minghao Fu, Wenjun Lin, Biwei Huang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07545)  

**Abstract**: Learned world models excel at interpolative generalization but fail at extrapolative generalization to novel physical properties. This limitation arises because they learn statistical correlations rather than the environment's underlying generative rules, such as physical invariances and conservation laws. We argue that learning these invariances is key to robust extrapolation. To achieve this, we first introduce \textbf{Symmetry Exploration}, an unsupervised exploration strategy where an agent is intrinsically motivated by a Hamiltonian-based curiosity bonus to actively probe and challenge its understanding of conservation laws, thereby collecting physically informative data. Second, we design a Hamiltonian-based world model that learns from the collected data, using a novel self-supervised contrastive objective to identify the invariant physical state from raw, view-dependent pixel observations. Our framework, \textbf{DreamSAC}, trained on this actively curated data, significantly outperforms state-of-the-art baselines in 3D physics simulations on tasks requiring extrapolation. 

---
# How Long Can Unified Multimodal Models Generate Images Reliably? Taming Long-Horizon Interleaved Image Generation via Context Curation 

**Authors**: Haoyu Chen, Qing Liu, Yuqian Zhou, He Zhang, Zhaowen Wang, Mengwei Ren, Jingjing Ren, Xiang Wang, Zhe Lin, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07540)  

**Abstract**: Unified multimodal models hold the promise of generating extensive, interleaved narratives, weaving text and imagery into coherent long-form stories. However, current systems suffer from a critical reliability gap: as sequences grow, generation quality rapidly collapses. In this work, we investigate the mechanism behind this failure and argue that it is distinct from standard long-context challenges. We reveal that in generation, accumulated visual history acts as a source of active pollution, a decay governed specifically by the number of image events rather than raw token count. We identify a structural vulnerability where dense visual tokens overwhelm the attention mechanism, creating noise that distorts future synthesis. Guided by these mechanistic insights, we propose UniLongGen, a training-free inference strategy that prioritizes safe conditioning over total recall. Instead of retaining all history, UniLongGen dynamically curates the model's memory, identifying and discarding interfering visual signals based on the model's own internal relevance rankings. Extensive experiments demonstrate that this active forgetting approach is essential for stability: UniLongGen significantly outperforms baselines in long-horizon fidelity and consistency, while simultaneously reducing memory footprint and inference time. 

---
# Neural Dynamics-Informed Pre-trained Framework for Personalized Brain Functional Network Construction 

**Authors**: Hongjie Jiang, Yifei Tang, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07524)  

**Abstract**: Brain activity is intrinsically a neural dynamic process constrained by anatomical space. This leads to significant variations in spatial distribution patterns and correlation patterns of neural activity across variable and heterogeneous scenarios. However, dominant brain functional network construction methods, which relies on pre-defined brain atlases and linear assumptions, fails to precisely capture varying neural activity patterns in heterogeneous scenarios. This limits the consistency and generalizability of the brain functional networks constructed by dominant methods. Here, a neural dynamics-informed pre-trained framework is proposed for personalized brain functional network construction. The proposed framework extracts personalized representations of neural activity patterns in heterogeneous scenarios. Personalized brain functional networks are obtained by utilizing these representations to guide brain parcellation and neural activity correlation estimation. Systematic evaluations were employed on 18 datasets across tasks, such as virtual neural modulation and abnormal neural circuit identification. Experimental results demonstrate that the proposed framework attains superior performance in heterogeneous scenarios. Overall, the proposed framework challenges the dominant brain functional network construction method. 

---
# SketchGraphNet: A Memory-Efficient Hybrid Graph Transformer for Large-Scale Sketch Corpora Recognition 

**Authors**: Shilong Chen, Mingyuan Li, Zhaoyang Wang, Zhonglin Ye, Haixing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.07521)  

**Abstract**: This work investigates large-scale sketch recognition from a graph-native perspective, where free-hand sketches are directly modeled as structured graphs rather than raster images or stroke sequences. We propose SketchGraphNet, a hybrid graph neural architecture that integrates local message passing with a memory-efficient global attention mechanism, without relying on auxiliary positional or structural encodings. To support systematic evaluation, we construct SketchGraph, a large-scale benchmark comprising 3.44 million graph-structured sketches across 344 categories, with two variants (A and R) to reflect different noise conditions. Each sketch is represented as a spatiotemporal graph with normalized stroke-order attributes. On SketchGraph-A and SketchGraph-R, SketchGraphNet achieves Top-1 accuracies of 83.62% and 87.61%, respectively, under a unified training configuration. MemEffAttn further reduces peak GPU memory by over 40% and training time by more than 30% compared with Performer-based global attention, while maintaining comparable accuracy. 

---
# InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills 

**Authors**: Dayang Liang, Yuhang Lin, Xinzhe Liu, Jiyuan Shi, Yunlong Liu, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.07516)  

**Abstract**: Interaction is one of the core abilities of humanoid robots. However, most existing frameworks focus on non-interactive whole-body control, which limits their practical applicability. In this work, we develop InterReal, a unified physics-based imitation learning framework for Real-world human-object Interaction (HOI) control. InterReal enables humanoid robots to track HOI reference motions, facilitating the learning of fine-grained interactive skills and their deployment in real-world settings. Within this framework, we first introduce a HOI motion data augmentation scheme with hand-object contact constraints, and utilize the augmented motions to improve policy stability under object perturbations. Second, we propose an automatic reward learner to address the challenge of large-scale reward shaping. A meta-policy guided by critical tracking error metrics explores and allocates reward signals to the low-level reinforcement learning objective, which enables more effective learning of interactive policies. Experiments on HOI tasks of box-picking and box-pushing demonstrate that InterReal achieves the best tracking accuracy and the highest task success rate compared to recent baselines. Furthermore, we validate the framework on the real-world robot Unitree G1, which demonstrates its practical effectiveness and robustness beyond simulation. 

---
# A Unified View of Drifting and Score-Based Models 

**Authors**: Chieh-Hsin Lai, Bac Nguyen, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.07514)  

**Abstract**: Drifting models train one-step generators by optimizing a mean-shift discrepancy induced by a kernel between the data and model distributions, with Laplace kernels used by default in practice. At each point, this discrepancy compares the kernel-weighted displacement toward nearby data samples with the corresponding displacement toward nearby model samples, yielding a transport direction for generated samples. In this paper, we make its relationship to the score-matching principle behind diffusion models precise by showing that drifting admits a score-based formulation on kernel-smoothed distributions. For Gaussian kernels, the population mean-shift field coincides with the score difference between the Gaussian-smoothed data and model distributions. This identity follows from Tweedie's formula, which links the score of a Gaussian-smoothed density to the corresponding conditional mean, and implies that Gaussian-kernel drifting is exactly a score-matching-style objective on smoothed distributions. It also clarifies the connection to Distribution Matching Distillation (DMD): both methods use score-mismatch transport directions, but drifting realizes the score signal nonparametrically from kernel neighborhoods, whereas DMD uses a pretrained diffusion teacher. Beyond Gaussians, we derive an exact decomposition for general radial kernels, and for the Laplace kernel we prove rigorous error bounds showing that drifting remains an accurate proxy for score matching in low-temperature and high-dimensional regimes. 

---
# SeDa: A Unified System for Dataset Discovery and Multi-Entity Augmented Semantic Exploration 

**Authors**: Kan Ling, Zhen Qin, Yichi Zhu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.07502)  

**Abstract**: The continuous expansion of open data platforms and research repositories has led to a fragmented dataset ecosystem, posing significant challenges for cross-source data discovery and interpretation. To address these challenges, we introduce SeDa--a unified framework for dataset discovery, semantic annotation, and multi-entity augmented navigation. SeDa integrates more than 7.6 million datasets from over 200 platforms, spanning governmental, academic, and industrial domains. The framework first performs semantic extraction and standardization to harmonize heterogeneous metadata representations. On this basis, a topic-tagging mechanism constructs an extensible tag graph that supports thematic retrieval and cross-domain association, while a provenance assurance module embedded within the annotation process continuously validates dataset sources and monitors link availability to ensure reliability and traceability. Furthermore, SeDa employs a multi-entity augmented navigation strategy that organizes datasets within a knowledge space of sites, institutions, and enterprises, enabling contextual and provenance-aware exploration beyond traditional search paradigms. Comparative experiments with popular dataset search platforms, such as ChatPD and Google Dataset Search, demonstrate that SeDa achieves superior coverage, timeliness, and traceability. Taken together, SeDa establishes a foundation for trustworthy, semantically enriched, and globally scalable dataset exploration. 

---
# From Thinker to Society: Security in Hierarchical Autonomy Evolution of AI Agents 

**Authors**: Xiaolei Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Tianyu Du, Heqing Huang, Hao Peng, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07496)  

**Abstract**: Artificial Intelligence (AI) agents have evolved from passive predictive tools into active entities capable of autonomous decision-making and environmental interaction, driven by the reasoning capabilities of Large Language Models (LLMs). However, this evolution has introduced critical security vulnerabilities that existing frameworks fail to address. The Hierarchical Autonomy Evolution (HAE) framework organizes agent security into three tiers: Cognitive Autonomy (L1) targets internal reasoning integrity; Execution Autonomy (L2) covers tool-mediated environmental interaction; Collective Autonomy (L3) addresses systemic risks in multi-agent ecosystems. We present a taxonomy of threats spanning cognitive manipulation, physical environment disruption, and multi-agent systemic failures, and evaluate existing defenses while identifying key research gaps. The findings aim to guide the development of multilayered, autonomy-aware defense architectures for trustworthy AI agent systems. 

---
# A Joint Neural Baseline for Concept, Assertion, and Relation Extraction from Clinical Text 

**Authors**: Fei Cheng, Ribeka Tanaka, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2603.07487)  

**Abstract**: Clinical information extraction (e.g., 2010 i2b2/VA challenge) usually presents tasks of concept recognition, assertion classification, and relation extraction. Jointly modeling the multi-stage tasks in the clinical domain is an underexplored topic. The existing independent task setting (reference inputs given in each stage) makes the joint models not directly comparable to the existing pipeline work. To address these issues, we define a joint task setting and propose a novel end-to-end system to jointly optimize three-stage tasks. We empirically investigate the joint evaluation of our proposal and the pipeline baseline with various embedding techniques: word, contextual, and in-domain contextual embeddings. The proposed joint system substantially outperforms the pipeline baseline by +0.3, +1.4, +3.1 for the concept, assertion, and relation F1. This work bridges joint approaches and clinical information extraction. The proposed approach could serve as a strong joint baseline for future research. The code is publicly available. 

---
# Interpretable-by-Design Transformers via Architectural Stream Independence 

**Authors**: Clayton Kerce, Alexis Fox  

**Link**: [PDF](https://arxiv.org/pdf/2603.07482)  

**Abstract**: While transformers achieve strong performance, their internal decision-making processes remain opaque. We investigate whether architectural constraints can enforce interpretability by design through architectural stream independence: maintaining a token stream (carrying symbolic structure) and contextual semantics in separated streams that remain independently observable throughout processing, with integration delayed until output. We validate this principle through the Late Fusion Architecture (LFA), which demonstrates interpretable symbolic heads through all the final layers, while standard transformers show dissolution by the third of six layers; we quantify this effect by introducing the Token-Position Dependence Score (PDS), with $PDS_{max}$ = 0.276 and 0.058, respectively. Crucially, intervention experiments demonstrate functional modularity: suppressing LFA's recency heads causes minimal semantic damage (Cohen's d = -0.158) versus catastrophic entanglement in baselines (d = -0.672). LFA demonstrates that architectural constraints improve underlying learning mechanisms, averaging 42% stability versus 19% and 11% for baseline comparisons, with extremes from 50% on LFA's best pairs (6 of 12 heads position-invariant) down to 0% complete collapse in over-constrained cases. By preventing premature entanglement, architectural independence steers models toward semantic understanding over positional heuristics, establishing interpretability as an architectural design criterion enforceable through structural constraints rather than post-hoc analysis. 

---
# Cross-Modal Taxonomic Generalization in (Vision-) Language Models 

**Authors**: Tianyang Xu, Marcelo Sandoval-Castaneda, Karen Livescu, Greg Shakhnarovich, Kanishka Misra  

**Link**: [PDF](https://arxiv.org/pdf/2603.07474)  

**Abstract**: What is the interplay between semantic representations learned by language models (LM) from surface form alone to those learned from more grounded evidence? We study this question for a scenario where part of the input comes from a different modality -- in our case, in a vision-language model (VLM), where a pretrained LM is aligned with a pretrained image encoder. As a case study, we focus on the task of predicting hypernyms of objects represented in images. We do so in a VLM setup where the image encoder and LM are kept frozen, and only the intermediate mappings are learned. We progressively deprive the VLM of explicit evidence for hypernyms, and test whether knowledge of hypernyms is recoverable from the LM. We find that the LMs we study can recover this knowledge and generalize even in the most extreme version of this experiment (when the model receives no evidence of a hypernym during training). Additional experiments suggest that this cross-modal taxonomic generalization persists under counterfactual image-label mappings only when the counterfactual data have high visual similarity within each category. Taken together, these findings suggest that cross-modal generalization in LMs arises as a result of both coherence in the extralinguistic input and knowledge derived from language cues. 

---
# Give Them an Inch and They Will Take a Mile:Understanding and Measuring Caller Identity Confusion in MCP-Based AI Systems 

**Authors**: Yuhang Huang, Boyang Ma, Biwei Yan, Xuelong Dai, Yechao Zhang, Minghui Xu, Kaidi Xu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07473)  

**Abstract**: The Model Context Protocol (MCP) is an open and standardized interface that enables large language models (LLMs) to interact with external tools and services, and is increasingly adopted by AI agents. However, the security of MCP-based systems remains largely this http URL this work, we conduct a large-scale security analysis of MCP servers integrated within MCP clients. We show that treating MCP servers as trusted entities without authenticating the caller identity is fundamentally insecure. Since MCP servers often cannot distinguish who is invoking a request, a single authorization decision may implicitly grant access to multiple, potentially untrusted this http URL empirical study reveals that most MCP servers rely on persistent authorization states, allowing tool invocations after an initial authorization without re-authentication, regardless of the caller. In addition, many MCP servers fail to enforce authentication at the per-tool level, enabling unauthorized access to sensitive this http URL findings demonstrate that one-time authorization and server-level trust significantly expand the attack surface of MCP-based systems, highlighting the need for explicit caller authentication and fine-grained authorization mechanisms. 

---
# Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers 

**Authors**: Mingxin Zhang, Xiaofeng Dai, Yu Yao, Ziqi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.07472)  

**Abstract**: In this study, we present a conditional diffusion-transformer framework for generating ensembles of three-dimensional Escherichia coli genome conformations guided by Hi-C contact maps. Instead of producing a single deterministic structure, we formulate genome reconstruction as a conditional generative modeling problem that samples heterogeneous conformations whose ensemble-averaged contacts are consistent with the input Hi-C data. A synthetic dataset is constructed using coarse-grained molecular dynamics simulations to generate chromatin ensembles and corresponding Hi-C maps under circular topology. Our models operate in a latent diffusion setting with a variational autoencoder that preserves per-bin alignment and supports replication-aware representations. Hi-C information is injected through a transformer-based encoder and cross-attention, enforcing a physically interpretable one-way constraint from Hi-C to structure. The model is trained using a flow-matching objective for stable optimization. On held-out ensembles, generated structures reproduce the input Hi-C distance-decay and structural correlation metrics while maintaining substantial conformational diversity, demonstrating the effectiveness of diffusion-based generative modeling for ensemble-level 3D genome reconstruction. 

---
# Towards Lightweight Adaptation of Speech Enhancement Models in Real-World Environments 

**Authors**: Longbiao Cheng, Shih-Chii Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07471)  

**Abstract**: Recent studies have shown that post-deployment adaptation can improve the robustness of speech enhancement models in unseen noise conditions. However, existing methods often incur prohibitive computational and memory costs, limiting their suitability for on-device deployment. In this work, we investigate model adaptation in realistic settings with dynamic acoustic scene changes and propose a lightweight framework that augments a frozen backbone with low-rank adapters updated via self-supervised training. Experiments on sequential scene evaluations spanning 111 environments across 37 noise types and three signal-to-noise ratio ranges, including the challenging [-8, 0] dB range, show that our method updates fewer than 1% of the base model's parameters while achieving an average 1.51 dB SI-SDR improvement within only 20 updates per scene. Compared to state-of-the-art approaches, our framework achieves competitive or superior perceptual quality with smoother and more stable convergence, demonstrating its practicality for lightweight on-device adaptation of speech enhancement models under real-world acoustic conditions. 

---
# The Dual-Stream Transformer: Channelized Architecture for Interpretable Language Modeling 

**Authors**: J. Clayton Kerce, Alexis Fox  

**Link**: [PDF](https://arxiv.org/pdf/2603.07461)  

**Abstract**: Standard transformers entangle all computation in a single residual stream, obscuring which components perform which functions. We introduce the Dual-Stream Transformer, which decomposes the residual stream into two functionally distinct components: a token stream updated by attention and a context stream updated by feed-forward networks. Information flow between attention heads is controlled through a hierarchy of mixing strategies, from fully independent (maximum interpretability) to dense (standard transformer behavior). This design exposes a tunable tradeoff between interpretability and performance.
We measure this tradeoff on language modeling tasks at 29M parameters. Fully independent head mixing increases validation loss by 8\% relative to dense baselines. The recommended Kronecker mixing strategy, which permits scalar communication between heads while preserving within-head structure, costs only 2.5\%. All configurations maintain functional generation under attention amplification (scaling logits by factors up to 16 at inference time), with degradation ranging from 16\% to 27\%. This robustness suggests the architectures learn discrete algorithms that operate independently of soft probabilistic mixing. The architecture provides a foundation for interpretable language models where internal structure is exposed by design. \footnote{This work was partially supported by DARPA Contract HR001125C0302.} 

---
# Where Do LLM-based Systems Break? A System-Level Security Framework for Risk Assessment and Treatment 

**Authors**: Neha Nagaraja, Hayretdin Bahsi  

**Link**: [PDF](https://arxiv.org/pdf/2603.07460)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into safety-critical workflows, yet existing security analyses remain fragmented and often isolate model behavior from the broader system context. This work introduces a goal-driven risk assessment framework for LLM-powered systems that combines system modeling with Attack-Defense Trees (ADTrees) and Common Vulnerability Scoring System (CVSS)-based exploitability scoring to support structured, comparable analysis. We demonstrate the framework through a healthcare case study, modeling multi-step attack paths targeting intervention in medical procedures, leakage of electronic health record (EHR) data, and disruption of service availability. Our analysis indicates that threats spanning (i) conventional cyber, (ii) adversarial ML, and (iii) conversational attacks that manipulate prompts or context often consolidate into a small number of dominant paths and shared system choke points, enabling targeted defenses to yield meaningful reductions in path exploitability. By systematically comparing defense portfolios, we align these risks with established vulnerability management practices and provide a domain-agnostic workflow applicable to other LLM-enabled critical systems. 

---
# "Better Ask for Forgiveness than Permission": Practices and Policies of AI Disclosure in Freelance Work 

**Authors**: Angel Hsing-Chi Hwang, Senya Wong, Baixiao Chen, Jessica He, Hyo Jin Do  

**Link**: [PDF](https://arxiv.org/pdf/2603.07459)  

**Abstract**: The growing use of AI applications among freelance workers is reshaping trust and relationships with clients. This paper investigates how both workers and clients perceive AI use and disclosure in the freelance economy through a three-stage study: interviews with workers and two survey studies with workers and clients. Findings first reveal a key expectation gap around disclosure: Workers often adopt passive disclosure practices, revealing AI use only when asked, as they assume clients can already detect it. Clients, however, are far less confident in recognizing AI-assisted work and prefer proactive disclosure. A second finding highlights the role of unclear or absent client AI policies, which leave workers consistently misinterpreting clients' expectations for AI use and disclosure. Together, these gaps point to the need for clearer guidelines and practices for AI disclosure. Insights extend beyond freelancing, offering implications for trust, accountability, and policy design in other AI-mediated work domains. 

---
# Image Generation Models: A Technical History 

**Authors**: Rouzbeh Shirvani  

**Link**: [PDF](https://arxiv.org/pdf/2603.07455)  

**Abstract**: Image generation has advanced rapidly over the past decade, yet the literature seems fragmented across different models and application domains. This paper aims to offer a comprehensive survey of breakthrough image generation models, including variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive and transformer-based generators, and diffusion-based methods. We provide a detailed technical walkthrough of each model type, including their underlying objectives, architectural building blocks, and algorithmic training steps. For each model type, we present the optimization techniques as well as common failure modes and limitations. We also go over recent developments in video generation and present the research works that made it possible to go from still frames to high quality videos. Lastly, we cover the growing importance of robustness and responsible deployment of these models, including deepfake risks, detection, artifacts, and watermarking. 

---
# Backdoor4Good: Benchmarking Beneficial Uses of Backdoors in LLMs 

**Authors**: Yige Li, Wei Zhao, Zhe Li, Nay Myat Min, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Yu-Gang Jiang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.07452)  

**Abstract**: Backdoor mechanisms have traditionally been studied as security threats that compromise the integrity of machine learning models. However, the same mechanism -- the conditional activation of specific behaviors through input triggers -- can also serve as a controllable and auditable interface for trustworthy model behavior. In this work, we present \textbf{Backdoor4Good (B4G)}, a unified benchmark and framework for \textit{beneficial backdoor} applications in large language models (LLMs). Unlike conventional backdoor studies focused on attacks and defenses, B4G repurposes backdoor conditioning for Beneficial Tasks that enhance safety, controllability, and accountability. It formalizes beneficial backdoor learning under a triplet formulation $(T, A, U)$, representing the \emph{Trigger}, \emph{Activation mechanism}, and \emph{Utility function}, and implements a benchmark covering four trust-centric applications. Through extensive experiments across Llama3.1-8B, Gemma-2-9B, Qwen2.5-7B, and Llama2-13B, we show that beneficial backdoors can achieve high controllability, tamper-resistance, and stealthiness while preserving clean-task performance. Our findings demonstrate new insights that backdoors need not be inherently malicious; when properly designed, they can serve as modular, interpretable, and beneficial building blocks for trustworthy AI systems. Our code and datasets are available at this https URL. 

---
# Dial: A Knowledge-Grounded Dialect-Specific NL2SQL System 

**Authors**: Xiang Zhang, Hongming Xu, Le Zhou, Wei Zhou, Xuanhe Zhou, Guoliang Li, Yuyu Luo, Changdong Liu, Guorun Chen, Jiang Liao, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07449)  

**Abstract**: Enterprises commonly deploy heterogeneous database systems, each of which owns a distinct SQL dialect with different syntax rules, built-in functions, and execution constraints. However, most existing NL2SQL methods assume a single dialect (e.g., SQLite) and struggle to produce queries that are both semantically correct and executable on target engines. Prompt-based approaches tightly couple intent reasoning with dialect syntax, rule-based translators often degrade native operators into generic constructs, and multi-dialect fine-tuning suffers from cross-dialect interference.
In this paper, we present Dial, a knowledge-grounded framework for dialect-specific NL2SQL. Dial introduces: (1) a Dialect-Aware Logical Query Planning module that converts natural language into a dialect-aware logical query plan via operator-level intent decomposition and divergence-aware specification; (2) HINT-KB, a hierarchical intent-aware knowledge base that organizes dialect knowledge into (i) a canonical syntax reference, (ii) a declarative function repository, and (iii) a procedural constraint repository; and (3) an execution-driven debugging and semantic verification loop that separates syntactic recovery from logic auditing to prevent semantic drift. We construct DS-NL2SQL, a benchmark covering six major database systems with 2,218 dialect-specific test cases. Experimental results show that Dial consistently improves translation accuracy by 10.25% and dialect feature coverage by 15.77% over state-of-the-art baselines. The code is at this https URL. 

---
# OrthoFormer: Instrumental Variable Estimation in Transformer Hidden States via Neural Control Functions 

**Authors**: Charles Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.07431)  

**Abstract**: Transformer architectures excel at sequential modeling yet remain fundamentally limited by correlational learning - they capture spurious associations induced by latent confounders rather than invariant causal mechanisms. We identify this as an epistemological challenge: standard Transformers conflate static background factors (intrinsic identity, style, context) with dynamic causal flows (state evolution, mechanism), leading to catastrophic out-of-distribution failure. We propose OrthoFormer, a causally grounded architecture that embeds instrumental variable estimation directly into Transformer blocks via neural control functions. Our framework rests on four theoretical pillars: Structural Directionality (time-arrow enforcement), Representation Orthogonality (latent-noise separation), Causal Sparsity (Markov Blanket approximation), and End-to-End Consistency (gradient- detached stage separation). We prove that OrthoFormer achieves bias strictly less than OLS for any valid instrument lag, with residual bias decaying geometrically as O(\r{ho}k ). We characterize the bias-variance-exogeneity trilemma inherent in self-instrumenting and identify the neural forbidden regression - where removing gradient detachment improves prediction loss while destroying causal validity. Experiments confirm all theoretical predictions. OrthoFormer represents a paradigm shift from correlational to causal sequence modeling, with implications for robustness, interpretability, and reliable decision-making under distribution shift. 

---
# Context Channel Capacity: An Information-Theoretic Framework for Understanding Catastrophic Forgetting 

**Authors**: Ran Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.07415)  

**Abstract**: Catastrophic forgetting remains a central challenge in continual learning (CL), yet lacks a unified information-theoretic explanation for why some architectures forget catastrophically while others do not. We introduce \emph{Context Channel Capacity} ($C_\mathrm{ctx}$), the mutual information between a CL architecture's context signal and its generated parameters, and prove that zero forgetting requires $C_\mathrm{ctx} \geq H(T)$, where $H(T)$ is the task identity entropy. We establish an \emph{Impossibility Triangle} -- zero forgetting, online learning, and finite parameters cannot be simultaneously satisfied by sequential state-based learners -- and show that conditional regeneration architectures (HyperNetworks) bypass this triangle by redefining parameters as function values rather than states.
We validate this framework across 8 CL methods on Split-MNIST (1,130+ experiments over 86 days, 4 seeds each), showing that $C_\mathrm{ctx}$ perfectly predicts forgetting behavior: methods with $C_\mathrm{ctx} = 0$ (NaiveSGD, EWC, SI, LwF, CFlow) exhibit catastrophic forgetting (6--97\%), while methods with $C_\mathrm{ctx} \approx 1$ (HyperNetwork) achieve zero forgetting (98.8\% ACC). We further propose \emph{Wrong-Context Probing} (P5), a practical diagnostic protocol for measuring $C_\mathrm{ctx}$, and extend the framework to CIFAR-10 via a novel \emph{Gradient Context Encoder} that closes the oracle gap from 23.3pp to 0.7pp. A systematic taxonomy of 15+ closed research directions -- including the Hebbian null result (frozen random features outperform learned features), CFlow's $\theta_0$-memorizer phenomenon, and the $S_N$ symmetry barrier to column specialization -- provides the community with precisely diagnosed negative results. Our central design principle: \emph{architecture over algorithm} -- the context pathway must be structurally unbypassable. 

---
# Machine Learning for the Internet of Underwater Things: From Fundamentals to Implementation 

**Authors**: Kenechi Omeke, Attai Abubakar, Michael Mollel, Lei Zhang, Qammer H. Abbasi, Muhammad Ali Imran  

**Link**: [PDF](https://arxiv.org/pdf/2603.07413)  

**Abstract**: The Internet of Underwater Things (IoUT) is becoming a critical infrastructure for ocean observation, marine resource management, and climate science. Its development is hindered by severe acoustic attenuation, propagation delays far exceeding those of terrestrial wireless systems, strict energy constraints, and dynamic topologies shaped by ocean currents. Machine learning (ML) has emerged as a key enabler for addressing these limitations, offering data driven mechanisms that enhance performance across all layers of underwater wireless sensor networks. This tutorial survey synthesises ML methodologies supervised, unsupervised, reinforcement, and deep learning specifically contextualised for underwater communication environments. It outlines the algorithmic principles of each paradigm and examines the conditions under which particular approaches deliver superior performance. A layer wise analysis highlights physical layer gains in localisation and channel estimation, MAC layer adaptations that improve channel utilisation, network layer routing strategies that extend operational lifetime, and transport layer mechanisms capable of reducing packet loss by up to 91 percent. At the application layer, ML enables substantial data compression and object detection accuracies reaching 92 percent. Drawing on 300 studies from 2012 to 2025, the survey documents energy efficiency gains of 7 to 29 times, throughput improvements over traditional protocols, and cross layer optimisation benefits of up to 42 percent. It also identifies persistent barriers, including limited datasets, computational constraints, and the gap between theoretical models and real world deployment. The survey concludes with emerging research directions and a technology roadmap supporting ML adoption in operational underwater networks. 

---
# UnSCAR: Universal, Scalable, Controllable, and Adaptable Image Restoration 

**Authors**: Debabrata Mandal, Soumitri Chattopadhyay, Yujie Wang, Marc Niethammer, Praneeth Chakravarthula  

**Link**: [PDF](https://arxiv.org/pdf/2603.07406)  

**Abstract**: Universal image restoration aims to recover clean images from arbitrary real-world degradations using a single inference model. Despite significant progress, existing all-in-one restoration networks do not scale to multiple degradations. As the number of degradations increases, training becomes unstable, models grow excessively large, and performance drops across both seen and unseen domains. In this work, we show that scaling universal restoration is fundamentally limited by interference across degradations during joint learning, leading to catastrophic task forgetting. To address this challenge, we introduce a unified inference pipeline with a multi-branch mixture-of-experts architecture that decomposes restoration knowledge across specialized task-adaptable experts. Our approach enables scalable learning (over sixteen degradations), adapts and generalizes robustly to unseen domains, and supports user-controllable restoration across degradations. Beyond achieving superior performance across benchmarks, this work establishes a new design paradigm for scalable and controllable universal image restoration. 

---
# Adaptive Capacity Allocation for Vision Language Action Fine-tuning 

**Authors**: Donghoon Kim, Minji Bae, Unghui Nam, Gyeonghun Kim, Suyun Lee, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07404)  

**Abstract**: Vision language action models (VLAs) are increasingly used for Physical AI, but deploying a pre-trained VLA model to unseen environments, embodiments, or tasks still requires adaptation. Parameter-efficient fine-tuning (PEFT), especially LoRA, is common for VLA policies, yet the exposed capacity knob, the rank, does not transfer uniformly: robotics transfer exhibits a higher and task-varying intrinsic rank than language fine-tuning. Small ranks suffice for LLMs (e.g., $r \in \{4, 8\}$), while spectral analyses indicate VLAs may require much larger ranks (e.g., $r \approx 128$) or near-full rank, a mismatch that worsens in multi-task settings. We present LoRA-SP (Select-Prune), a rank-adaptive fine-tuning method that replaces fixed-rank updates with input- and layer-wise capacity. LoRA-SP uses an SVD-style parameterization with a small router whose nonnegative scores act as singular values over a shared vector bank. The active set is chosen by an energy target on the cumulative squared scores $E(k) \ge \eta$, providing a direct link to approximation error via our spectral analysis. During training, $\eta$ concentrates energy on a few directions and teaches the router to rely on fewer vectors while preserving accuracy. This yields compact adapters that reduce cross-task interference and improve generalization. On four real-robot manipulation tasks collected on an unseen AgileX PiPER arm, across two VLA backbones ($\pi_0$ and SmolVLA), LoRA-SP matches or exceeds full fine-tuning with far fewer trainable parameters, and improves multi-task success by up to 31.6% over standard LoRA while remaining robust to rank choice. 

---
# AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions 

**Authors**: Jihyoung Jang, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07394)  

**Abstract**: Visual Question Answering (VQA) is a core task for evaluating the capabilities of Vision-Language Models (VLMs). Existing VQA benchmarks primarily feature clear and unambiguous image-question pairs, whereas real-world scenarios often involve varying degrees of ambiguity that require nuanced reasoning and context-appropriate response strategies. Although recent studies have begun to address ambiguity in VQA, they lack (1) a systematic categorization of ambiguity levels and (2) datasets and models that support strategy-aware responses. In this paper, we introduce Ambiguous Visual Question Answering (AQuA), a fine-grained dataset that classifies ambiguous VQA instances into four levels according to the nature and degree of ambiguity, along with the optimal response strategy for each case. Our evaluation of diverse open-source and proprietary VLMs shows that most models fail to adapt their strategy to the ambiguity type, frequently producing overconfident answers rather than seeking clarification or acknowledging uncertainty. To address this challenge, we fine-tune VLMs on AQuA, enabling them to adaptively choose among multiple response strategies, such as directly answering, inferring intent from contextual cues, listing plausible alternatives, or requesting clarification. VLMs trained on AQuA achieve strategic response generation for ambiguous VQA, demonstrating the ability to recognize ambiguity, manage uncertainty, and respond with context-appropriate strategies, while outperforming both open-source and closed-source baselines. 

---
# Sparsity and Out-of-Distribution Generalization 

**Authors**: Scott Aaronson, Lin Lin Lee, Jiawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.07388)  

**Abstract**: Explaining out-of-distribution generalization has been a central problem in epistemology since Goodman's "grue" puzzle in 1946. Today it's a central problem in machine learning, including AI alignment.
Here we propose a principled account of OOD generalization with three main ingredients. First, the world is always presented to experience not as an amorphous mass, but via distinguished features (for example, visual and auditory channels). Second, Occam's Razor favors hypotheses that are "sparse," meaning that they depend on as few features as possible. Third, sparse hypotheses will generalize from a training to a test distribution, provided the two distributions sufficiently overlap on their restrictions to the features that are either actually relevant or hypothesized to be. The two distributions could diverge arbitrarily on other features.
We prove a simple theorem that formalizes the above intuitions, generalizing the classic sample complexity bound of Blumer et al. to an OOD context. We then generalize sparse classifiers to subspace juntas, where the ground truth classifier depends solely on a low-dimensional linear subspace of the features. 

---
# Scheduling Parallel Optical Circuit Switches for AI Training 

**Authors**: Kevin Liang, Litao Qiao, Isaac Keslassy, Bill Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.07373)  

**Abstract**: The rapid growth of AI training has dramatically increased datacenter traffic demand and energy consumption, which has motivated renewed interest in optical circuit switches (OCSes) as a high-bandwidth, energy-efficient alternative for AI fabrics. Deploying multiple parallel OCSes is a leading alternative. However, efficiently scheduling time-varying traffic matrices across parallel optical switches with non-negligible reconfiguration delays remains an open challenge.
We consider the problem of scheduling a single AI traffic demand matrix $D$ over $s$ parallel OCSes while minimizing the makespan under reconfiguration delay $\delta$. Our algorithm Spectra relies on a three-step approach: Decompose $D$ into a minimal set of weighted permutations; Schedule these permutations across parallel switches using load-aware assignment; then Equalize the imbalanced loads on the switches via controlled permutation splitting. Evaluated on realistic AI training workloads (GPT model and Qwen MoE expert routing) as well as standard benchmarks, Spectra vastly outperforms a baseline based on state-of-the-art algorithms, reducing schedule makespan by an average factor of $1.4\times$ on GPT AI workloads, $1.9\times$ on MoE AI workloads, and $2.4\times$ on standard benchmarks. Further, the makespans achieved by Spectra consistently approach newly derived lower bounds. 

---
# Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios 

**Authors**: Namrata Patil Gurav, Akashdeep Ranu, Archchana Sindhujan, Diptesh Kanojia  

**Link**: [PDF](https://arxiv.org/pdf/2603.07372)  

**Abstract**: Quality Estimation (QE) is essential for assessing machine translation quality in reference-less settings, particularly for domain-specific and low-resource language scenarios. In this paper, we investigate sentence-level QE for English to Indic machine translation across four domains (Healthcare, Legal, Tourism, and General) and five language pairs. We systematically compare zero-shot, few-shot, and guideline-anchored prompting across selected closed-weight and open-weight LLMs. Findings indicate that while closed-weight models achieve strong performance via prompting alone, prompt-only approaches remain fragile for open-weight models, especially in high-risk domains. To address this, we adopt ALOPE, a framework for LLM-based QE that uses Low-Rank Adaptation with regression heads attached to selected intermediate Transformer layers. We also extend ALOPE with recently proposed Low-Rank Multiplicative Adaptation (LoRMA). Our results show that intermediate-layer adaptation consistently improves QE performance, with gains in semantically complex domains, indicating a path toward more robust QE in practical scenarios. We release code and domain-specific QE datasets publicly to support further research. 

---
# ConfHit: Conformal Generative Design with Oracle Free Guarantees 

**Authors**: Siddhartha Laghuvarapu, Ying Jin, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.07371)  

**Abstract**: The success of deep generative models in scientific discovery requires not only the ability to generate novel candidates but also reliable guarantees that these candidates indeed satisfy desired properties. Recent conformal-prediction methods offer a path to such guarantees, but its application to generative modeling in drug discovery is limited by budget constraints, lack of oracle access, and distribution shift. To this end, we introduce ConfHit, a distribution-free framework that provides validity guarantees under these conditions. ConfHit formalizes two central questions: (i) Certification: whether a generated batch can be guaranteed to contain at least one hit with a user-specified confidence level, and (ii) Design: whether the generation can be refined to a compact set without weakening this guarantee. ConfHit leverages weighted exchangeability between historical and generated samples to eliminate the need for an experimental oracle, constructs multiple-sample density-ratio weighted conformal p-value to quantify statistical confidence in hits, and proposes a nested testing procedure to certify and refine candidate sets of multiple generated samples while maintaining statistical guarantees. Across representative generative molecule design tasks and a broad range of methods, ConfHit consistently delivers valid coverage guarantees at multiple confidence levels while maintaining compact certified sets, establishing a principled and reliable framework for generative modeling. 

---
# Position: LLMs Must Use Functor-Based and RAG-Driven Bias Mitigation for Fairness 

**Authors**: Ravi Ranjan, Utkarsh Grover, Agorista Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2603.07368)  

**Abstract**: Biases in large language models (LLMs) often manifest as systematic distortions in associations between demographic attributes and professional or social roles, reinforcing harmful stereotypes across gender, ethnicity, and geography. This position paper advocates for addressing demographic and gender biases in LLMs through a dual-pronged methodology, integrating category-theoretic transformations and retrieval-augmented generation (RAG). Category theory provides a rigorous, structure-preserving mathematical framework that maps biased semantic domains to unbiased canonical forms via functors, ensuring bias elimination while preserving semantic integrity. Complementing this, RAG dynamically injects diverse, up-to-date external knowledge during inference, directly countering ingrained biases within model parameters. By combining structural debiasing through functor-based mappings and contextual grounding via RAG, we outline a comprehensive framework capable of delivering equitable and fair model outputs. Our synthesis of the current literature validates the efficacy of each approach individually, while addressing potential critiques demonstrates the robustness of this integrated strategy. Ensuring fairness in LLMs, therefore, demands both the mathematical rigor of category-theoretic transformations and the adaptability of retrieval augmentation. 

---
# Scaling Laws in the Tiny Regime: How Small Models Change Their Mistakes 

**Authors**: Mohammed Alnemari, Rizwan Qureshi, Nader Begrazadah  

**Link**: [PDF](https://arxiv.org/pdf/2603.07365)  

**Abstract**: Neural scaling laws describe how model performance improves as a power law with size, but existing work focuses on models above 100M parameters. The sub-20M regime -- where TinyML and edge AI operate -- remains unexamined. We train 90 models (22K--19.8M parameters) across two architectures (plain ConvNet, MobileNetV2) on CIFAR-100, varying width while holding depth and training fixed. Both follow approximate power laws in error rate: $\alpha = 0.156 \pm 0.002$ (ScaleCNN) and $\alpha = 0.106 \pm 0.001$ (MobileNetV2) across five seeds. Since prior work fit cross-entropy loss rather than error rate, direct exponent comparison is approximate; with that caveat, these are 1.4--2x steeper than $\alpha \approx 0.076$ for large language models. The power law does not hold uniformly: local exponents decay with scale, and MobileNetV2 saturates at 19.8M parameters ($\alpha_{\mathrm{local}} = 0.006$). Error structure also changes. Jaccard overlap between error sets of the smallest and largest ScaleCNN is only 0.35 (25 seed pairs, $\pm 0.004$) -- compression changes which inputs are misclassified, not merely how many. Small models
concentrate capacity on easy classes (Gini: 0.26 at 22K vs. 0.09 at 4.7M) while abandoning the hardest (bottom-5 accuracy: 10% vs. 53%). Counter to expectation, the smallest models are best calibrated (ECE = 0.013 vs. peak 0.110
at mid-size). Aggregate accuracy is therefore misleading for edge deployment; validation must happen at the target model size. 

---
# Latent Generative Models with Tunable Complexity for Compressed Sensing and other Inverse Problems 

**Authors**: Sean Gunn, Jorio Cocola, Oliver De Candido, Vaggos Chatziafratis, Paul Hand  

**Link**: [PDF](https://arxiv.org/pdf/2603.07357)  

**Abstract**: Generative models have emerged as powerful priors for solving inverse problems. These models typically represent a class of natural signals using a single fixed complexity or dimensionality. This can be limiting: depending on the problem, a fixed complexity may result in high representation error if too small, or overfitting to noise if too large. We develop tunable-complexity priors for diffusion models, normalizing flows, and variational autoencoders, leveraging nested dropout. Across tasks including compressed sensing, inpainting, denoising, and phase retrieval, we show empirically that tunable priors consistently achieve lower reconstruction errors than fixed-complexity baselines. In the linear denoising setting, we provide a theoretical analysis that explicitly characterizes how the optimal tuning parameter depends on noise and model structure. This work demonstrates the potential of tunable-complexity generative priors and motivates both the development of supporting theory and their application across a wide range of inverse problems. 

---
# AgrI Challenge: A Data-Centric AI Competition for Cross-Team Validation in Agricultural Vision 

**Authors**: Mohammed Brahimi, Karim Laabassi, Mohamed Seghir Hadj Ameur, Aicha Boutorh, Badia Siab-Farsi, Amin Khouani, Omar Farouk Zouak, Seif Eddine Bouziane, Kheira Lakhdari, Abdelkader Nabil Benghanem  

**Link**: [PDF](https://arxiv.org/pdf/2603.07356)  

**Abstract**: Machine learning models in agricultural vision often achieve high accuracy on curated datasets but fail to generalize under real field conditions due to distribution shifts between training and deployment environments. Moreover, most machine learning competitions focus primarily on model design while treating datasets as fixed resources, leaving the role of data collection practices in model generalization largely unexplored.
We introduce the AgrI Challenge, a data-centric competition framework in which multiple teams independently collect field datasets, producing a heterogeneous multi-source benchmark that reflects realistic variability in acquisition conditions. To systematically evaluate cross-domain generalization across independently collected datasets, we propose Cross-Team Validation (CTV), an evaluation paradigm that treats each team's dataset as a distinct domain. CTV includes two complementary protocols: Train-on-One-Team-Only (TOTO), which measures single-source generalization, and Leave-One-Team-Out (LOTO), which evaluates collaborative multi-source training.
Experiments reveal substantial generalization gaps under single-source training: models achieve near-perfect validation accuracy yet exhibit validation-test gaps of up to 16.20% (DenseNet121) and 11.37% (Swin Transformer) when evaluated on datasets collected by other teams. In contrast, collaborative multi-source training dramatically improves robustness, reducing the gap to 2.82% and 1.78%, respectively.
The challenge also produced a publicly available dataset of 50,673 field images of six tree species collected by twelve independent teams, providing a diverse benchmark for studying domain shift and data-centric learning in agricultural vision. 

---
# Learning Concept Bottleneck Models from Mechanistic Explanations 

**Authors**: Antonio De Santis, Schrasing Tong, Marco Brambilla, Lalana Kagal  

**Link**: [PDF](https://arxiv.org/pdf/2603.07343)  

**Abstract**: Concept Bottleneck Models (CBMs) aim for ante-hoc interpretability by learning a bottleneck layer that predicts interpretable concepts before the decision. State-of-the-art approaches typically select which concepts to learn via human specification, open knowledge graphs, prompting an LLM, or using general CLIP concepts. However, concepts defined a-priori may not have sufficient predictive power for the task or even be learnable from the available data. As a result, these CBMs often significantly trail their black-box counterpart when controlling for information leakage. To address this, we introduce a novel CBM pipeline named Mechanistic CBM (M-CBM), which builds the bottleneck directly from a black-box model's own learned concepts. These concepts are extracted via Sparse Autoencoders (SAEs) and subsequently named and annotated on a selected subset of images using a Multimodal LLM. For fair comparison and leakage control, we also introduce the Number of Contributing Concepts (NCC), a decision-level sparsity metric that extends the recently proposed NEC metric. Across diverse datasets, we show that M-CBMs consistently surpass prior CBMs at matched sparsity, while improving concept predictions and providing concise explanations. Our code is available at this https URL. 

---
# Agora: Teaching the Skill of Consensus-Finding with AI Personas Grounded in Human Voice 

**Authors**: Suyash Fulay, Prerna Ravi, Emily Kubin, Shrestha Mohanty, Michiel Bakker, Deb Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.07339)  

**Abstract**: Deliberative democratic theory suggests that civic competence: the capacity to navigate disagreement, weigh competing values, and arrive at collective decisions is not innate but developed through practice. Yet opportunities to cultivate these skills remain limited, as traditional deliberative processes like citizens' assemblies reach only a small fraction of the population. We present Agora, an early-stage AI-powered platform that uses LLMs to organize authentic human voices on policy issues, helping users build consensus-finding skills by proposing and revising policy recommendations, hearing supporting and opposing perspectives, and receiving feedback on how policy changes affect predicted support. In a preliminary study with 44 university students, participants using the full interface (with access to voice explanations) reported higher levels of problem-solving skills, internal deliberation, and produced higher quality consensus statements compared to a control condition showing only aggregate support distributions. These initial findings point toward a promising direction for scaling civic education. 

---
# Norm-Hierarchy Transitions in Representation Learning: When and Why Neural Networks Abandon Shortcuts 

**Authors**: Truong Xuan Khanh, Truong Quynh Hoa  

**Link**: [PDF](https://arxiv.org/pdf/2603.07323)  

**Abstract**: Neural networks often rely on spurious shortcuts for many epochs before discovering structured representations. However, the mechanism governing when this transition occurs and whether its timing can be predicted remains unclear. Prior work shows that gradient descent converges to low norm solutions and that neural networks exhibit simplicity bias, but neither explains the timescale of the transition from shortcut features to structured representations.
We introduce the Norm-Hierarchy Transition (NHT) framework, which explains delayed representation learning as the slow traversal of a hierarchy of parameter norms during regularized optimization. When multiple interpolating solutions exist with different norms, weight decay gradually moves the model from high norm shortcut solutions toward lower norm structured representations. We derive a tight bound showing that the transition delay grows logarithmically with the ratio between shortcut and structured norms.
Experiments on modular arithmetic, CIFAR-10 with spurious features, CelebA, and Waterbirds support the predictions of the framework. The results suggest that grokking, shortcut learning, and delayed feature discovery arise from a common mechanism based on norm hierarchy traversal during training. 

---
# Adversarial Latent-State Training for Robust Policies in Partially Observable Domains 

**Authors**: Angad Singh Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2603.07313)  

**Abstract**: Robustness under latent distribution shift remains challenging in partially observable reinforcement learning. We formalize a focused setting where an adversary selects a hidden initial latent distribution before the episode, termed an adversarial latent-initial-state POMDP. Theoretically, we prove a latent minimax principle, characterize worst-case defender distributions, and derive approximate best-response certificates with finite-sample guarantees, providing formal meaning to empirical training diagnostics. Empirically, using a Battleship benchmark, we demonstrate that targeted exposure to shifted latent distributions reduces average robustness gaps between Spread and Uniform distributions from 10.3 to 3.1 shots at equal budget. Furthermore, iterative best-response training exhibits budget-sensitive behavior entirely consistent with our approximate certificate theory. Ultimately, we show that for latent-initial-state problems, our framework yields precise diagnostic principles and confirms that structured adversarial exposure effectively mitigates worst-case vulnerabilities. 

---
# Spectral Discovery of Continuous Symmetries via Generalized Fourier Transforms 

**Authors**: Pavan Karjol, Kumar Shubham, Prathosh AP  

**Link**: [PDF](https://arxiv.org/pdf/2603.07299)  

**Abstract**: Continuous symmetries are fundamental to many scientific and learning problems, yet they are often unknown a priori. Existing symmetry discovery approaches typically search directly in the space of transformation generators or rely on learned augmentation schemes. We propose a fundamentally different perspective based on spectral structure.
We introduce a framework for discovering continuous one-parameter subgroups using the Generalized Fourier Transform (GFT). Our central observation is that invariance to a subgroup induces structured sparsity in the spectral decomposition of a function across irreducible representations. Instead of optimizing over generators, we detect symmetries by identifying this induced sparsity pattern in the spectral domain.
We develop symmetry detection procedures on maximal tori, where the GFT reduces to multi-dimensional Fourier analysis through their irreducible representations. Across structured tasks, including the double pendulum and top quark tagging, we demonstrate that spectral sparsity reliably reveals one-parameter symmetries. These results position spectral analysis as a principled and interpretable alternative to generator-based symmetry discovery. 

---
# MAviS: A Multimodal Conversational Assistant For Avian Species 

**Authors**: Yevheniia Kryklyvets, Mohammed Irfan Kurpath, Sahal Shaji Mullappilly, Jinxing Zhou, Fahad Shabzan Khan, Rao Anwer, Salman Khan, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2603.07294)  

**Abstract**: Fine-grained understanding and species-specific multimodal question answering are vital for advancing biodiversity conservation and ecological monitoring. However, existing multimodal large language models face challenges when it comes to specialized topics like avian species, making it harder to provide accurate and contextually relevant information in these areas. To address this limitation, we introduce the MAviS-Dataset, a large-scale multimodal avian species dataset that integrates image, audio, and text modalities for over 1,000 bird species, comprising both pretraining and instruction-tuning subsets enriched with structured question-answer pairs. Building on the MAviS-Dataset, we introduce MAviS-Chat, a multimodal LLM that supports audio, vision, and text and is designed for fine-grained species understanding, multimodal question answering, and scene-specific description generation. Finally, for quantitative evaluation, we present MAviS-Bench, a benchmark of over 25,000 QA pairs designed to assess avian species-specific perceptual and reasoning abilities across modalities. Experimental results show that MAviS-Chat outperforms the baseline MiniCPM-o-2.6 by a large margin, achieving state-of-the-art open-source results and demonstrating the effectiveness of our instruction-tuned MAviS-Dataset. Our findings highlight the necessity of domain-adaptive multimodal LLMs for ecological applications. 

---
# Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study across Four Models and Five Prompting Regimes 

**Authors**: Chen Zhao, Yuan Tang, Yitian Qian  

**Link**: [PDF](https://arxiv.org/pdf/2603.07287)  

**Abstract**: LLMs are increasingly used to draft academic text and to support software engineering (SE) evidence synthesis, but they often hallucinate bibliographic references that look legitimate. We study how deployment-motivated prompting constraints affect citation verifiability in a closed-book setting. Using 144 claims (24 in SE&CS) and a deterministic verification pipeline (Crossref + Semantic Scholar), we evaluate two proprietary models (Claude Sonnet, GPT-4o) and two open-weight models (LLaMA~3.1-8B, Qwen~2.5-14B) across five regimes: Baseline, Temporal (publication-year window), Survey-style breadth, Non-Disclosure policy, and their combination. Across 17,443 generated citations, no model exceeds a citation-level existence rate of 0.475; Temporal and Combo conditions produce the steepest drops while outputs remain format-compliant (well-formed bibliographic fields). Unresolved outcomes dominate (36-61%); a 100-citation audit indicates that a substantial fraction of Unresolved cases are fabricated. Results motivate post-hoc citation verification before LLM outputs enter SE literature reviews or tooling pipelines. 

---
# Kinematics-Aware Latent World Models for Data-Efficient Autonomous Driving 

**Authors**: Jiazhuo Li, Linjiang Cao, Qi Liu, Xi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.07264)  

**Abstract**: Data-efficient learning remains a central challenge in autonomous driving due to the high cost and safety risks of large-scale real-world interaction. Although world-model-based reinforcement learning enables policy optimization through latent imagination, existing approaches often lack explicit mechanisms to encode spatial and kinematic structure essential for driving tasks. In this work, we build upon the Recurrent State-Space Model (RSSM) and propose a kinematics-aware latent world model framework for autonomous driving. Vehicle kinematic information is incorporated into the observation encoder to ground latent transitions in physically meaningful motion dynamics, while geometry-aware supervision regularizes the RSSM latent state to capture task-relevant spatial structure beyond pixel reconstruction. The resulting structured latent dynamics improve long-horizon imagination fidelity and stabilize policy optimization. Experiments in a driving simulation benchmark demonstrate consistent gains over both model-free and pixel-based world-model baselines in terms of sample efficiency and driving performance. Ablation studies further verify that the proposed design enhances spatial representation quality within the latent space. These results suggest that integrating kinematic grounding into RSSM-based world models provides a scalable and physically grounded paradigm for autonomous driving policy learning. 

---
# Learning When to Cooperate Under Heterogeneous Goals 

**Authors**: Max Taylor-Davies, Neil Bramley, Christopher G. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2603.07253)  

**Abstract**: A significant element of human cooperative intelligence lies in our ability to identify opportunities for fruitful collaboration; and conversely to recognise when the task at hand is better pursued alone. Research on flexible cooperation in machines has left this meta-level problem largely unexplored, despite its importance for successful collaboration in heterogeneous open-ended environments. Here, we extend the typical Ad Hoc Teamwork (AHT) setting to incorporate the idea of agents having heterogeneous goals that in any given scenario may or may not overlap. We introduce a novel approach to learning policies in this setting, based on a hierarchical combination of imitation and reinforcement learning, and show that it outperforms baseline methods across extended versions of two cooperative environments. We also investigate the contribution of an auxiliary component that learns to model teammates by predicting their actions, finding that its effect on performance is inversely related to the amount of observable information about teammate goals. 

---
# LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture 

**Authors**: Erik Scheurer, Rocco Sedona, Stefan Kesselheim, Gabriele Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2603.07246)  

**Abstract**: Geospatial foundation models provide precomputed embeddings that serve as compact feature vectors for large-scale satellite remote sensing data. While these embeddings can reduce data-transfer bottlenecks and computational costs, Earth observation (EO) applications can still face geometric mismatches between user-defined areas of interest and the fixed precomputed embedding grid. Standard latent-space interpolation is unreliable in this setting because the embedding manifold is highly non-convex, yielding representations that do not correspond to realistic inputs. We verify this using Prithvi-EO-2.0 to understand the shortcomings of interpolation applied to patch embeddings. As a substitute, we propose a Learned Equivariance-Predicting Architecture (LEPA). Instead of averaging vectors, LEPA conditions a predictor on geometric augmentations to directly predict the transformed embedding. We evaluate LEPA on NASA/USGS Harmonized Landsat-Sentinel (HLS) imagery and ImageNet-1k. Experiments show that standard interpolation achieves a mean reciprocal rank (MRR) below 0.2, whereas LEPA increases MRR to over 0.8, enabling accurate geometric adjustment without re-encoding. 

---
# A Hybrid LTR-based System via Social Context Embedding for Recommending Solutions of Software Bugs in Developer Communities 

**Authors**: Fouzi Harrag, Mokdad Khemliche  

**Link**: [PDF](https://arxiv.org/pdf/2603.07229)  

**Abstract**: Questions and Answering forums such as Stack Overflow play an important role in supporting software developers in finding answers to queries related to issues such as software errors and bugs. However, searching through a large set of candidate answers could be time consuming and may not lead to the best solution. In this research, the effectiveness of data mining models and machine learning techniques to solve this kind of problems is evaluated. We propose a recommender system to aid developers in finding solutions to their software bugs by carefully mining Stack Overflow. The proposed model leverages the knowledge available through crowdsourcing the Q&A available in Stack Overflow to recommend a solution to software bugs. We use deep learning techniques to construct the required Learning-to-Rank (LTR)-based model using the social context embedding the Stack Overflow features. Text mining, natural language processing and recommendation algorithms are used to extract, evaluate and recommend the best relevant bug solutions. Additionally, our model achieves nearly 78% correct solutions when recommending the 10 best answers for each question. 

---
# VINO: Video-driven Invariance for Non-contextual Objects via Structural Prior Guided De-contextualization 

**Authors**: Seul-Ki Yeom, Marcel Simon, Eunbin Lee, Tae-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07222)  

**Abstract**: Self-supervised learning (SSL) has made rapid progress, yet learned features often over-rely on contextual shortcuts-background textures and co-occurrence statistics. While video provides rich temporal variation, dense in-the-wild streams with strong ego-motion create a co-occurrence trap: foreground objects and background context move coherently, encouraging representations to collapse into scene encoders.
To address this, we propose VINO (Video-driven Invariance for Non-Contextual Objects), a teacher-student framework that learns robust image encoders from dense video by imposing a structural information bottleneck. Using a class-agnostic structural prior solely to generate views-not as semantic pseudo-labels-VINO forms an asymmetric distillation problem. The teacher predicts from a foreground-union view with the background suppressed, while the student observes object-conditioned scene views that retain surrounding context but remove competing instances. Matching these targets via masked distillation makes background cues unreliable, pushing the representation toward object-centric invariances. We further enforce temporal object permanence via teacher-anchored cross-time distillation over track-matched objects, and stabilize part-to-whole consistency with mask-guided local views.
Through attention visualization and unsupervised object discovery on PASCAL VOC, we demonstrate that VINO effectively disentangles foreground from background. Pretrained on the dense Walking Tours Venice video, VINO achieves 34.8 CorLoc, yielding highly focused, shape-biased representations that substantially outperform prior dense-video and motion-guided SSL baselines. 

---
# A Miniature Brain Transformer: Thalamic Gating, Hippocampal Lateralization, Amygdaloid Salience, and Prefrontal Working Memory in Attention-Coupled Latent Memory 

**Authors**: Hong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.07217)  

**Abstract**: We present a miniature brain transformer architecture that extends the attention-coupled latent memory framework with four additional brain-region analogues: a thalamic relay, an amygdaloid salience module, a prefrontal working-memory (PFC) buffer, and a cerebellar fast-path, all coupled by inhibitory callosal cross-talk between lateralized hippocampal banks. We evaluate on a two-domain benchmark -- MQAR (Multi-Query Associative Recall; episodic domain) and modular arithmetic (+1 mod 10; rule-based domain) -- using a seven-variant additive ablation. The central empirical finding is a surprise: inhibitory callosal coupling alone never lateralizes the banks (variants 1-5 maintain D_sep ~ 0.25 and P_ct ~ 0.25 for all 30 epochs). Functional lateralization requires the synergy of PFC and inhibition: only when the PFC buffer is added (variant 6) does a sharp, discontinuous phase transition fire -- at epoch 11 for the PFC-only variant and epoch 10 for the full model -- collapsing P_ct from 0.25 to ~0.002 and more than doubling D_sep from 0.251 to 0.501 in a single gradient step. The PFC buffer acts as a symmetry-breaker: its slowly drifting domain context creates the initial asymmetry that the inhibitory feedback loop then amplifies irreversibly. The cerebellar fast-path accelerates the transition by one epoch (epoch 10 vs. epoch 11) with no asymptotic change, confirming its convergence-acceleration role. The result constitutes a novel, falsifiable prediction -- no lateralization without working memory context -- and a principled, neurobiologically motivated blueprint for hierarchical persistent memory in sequence models. 

---
# Governance Architecture for Autonomous Agent Systems: Threats, Framework, and Engineering Practice 

**Authors**: Yuxu Ge  

**Link**: [PDF](https://arxiv.org/pdf/2603.07191)  

**Abstract**: Autonomous agents powered by large language models introduce a class of execution-layer vulnerabilities -- prompt injection, retrieval poisoning, and uncontrolled tool invocation -- that existing guardrails fail to address systematically. In this work, we propose the Layered Governance Architecture (LGA), a four-layer framework comprising execution sandboxing (L1), intent verification (L2), zero-trust inter-agent authorization (L3), and immutable audit logging (L4). To evaluate LGA, we construct a bilingual benchmark (Chinese original, English via machine translation) of 1,081 tool-call samples -- covering prompt injection, RAG poisoning, and malicious skill plugins -- and apply it to OpenClaw, a representative open-source agent framework. Experimental results on Layer 2 intent verification with four local LLM judges (Qwen3.5-4B, Llama-3.1-8B, Qwen3.5-9B, Qwen2.5-14B) and one cloud judge (GPT-4o-mini) show that all five LLM judges intercept 93.0-98.5% of TC1/TC2 malicious tool calls, while lightweight NLI baselines remain below 10%. TC3 (malicious skill plugins) proves harder at 75-94% IR among judges with meaningful precision-recall balance, motivating complementary enforcement at Layers 1 and 3. Qwen2.5-14B achieves the best local balance (98% IR, approximately 10-20% FPR); a two-stage cascade (Qwen3.5-9B->GPT-4o-mini) achieves 91.9-92.6% IR with 1.9-6.7% FPR; a fully local cascade (Qwen3.5-9B->Qwen2.5-14B) achieves 94.7-95.6% IR with 6.0-9.7% FPR for data-sovereign deployments. An end-to-end pipeline evaluation (n=100) demonstrates that all four layers operate in concert with 96% IR and a total P50 latency of approximately 980 ms, of which the non-judge layers contribute only approximately 18 ms. Generalization to the external InjecAgent benchmark yields 99-100% interception, confirming robustness beyond our synthetic data. 

---
# From State Changes to Creative Decisions: Documenting and Interpreting Traces Across Creative Domains 

**Authors**: Xiaohan Peng, Sotiris Piliouras, Carl Abou Saada Nujaim  

**Link**: [PDF](https://arxiv.org/pdf/2603.07184)  

**Abstract**: Analyzing creative activity traces requires capturing activity at appropriate granularity and interpreting it in ways that reflect the structure of creative practice. However, existing approaches record state changes without preserving the intent or relationships that define higher-level creative moves. This decoupling manifests differently across domains: GenAI tools lose non-linear exploration structure, visualization authoring obscures representational intent, and programmatic environments flatten interaction boundaries. We present three complementary approaches: a node-based interface for stateful GenAI artifact management, a vocabulary of visual cues as higher-level creative moves in visualization authoring, and a programming model that embeds semantic histories directly into interaction state. 

---
# Fine-Grained Table Retrieval Through the Lens of Complex Queries 

**Authors**: Wojciech Kosiuk, Xingyu Ji, Yeounoh Chung, Fatma Özcan, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2603.07146)  

**Abstract**: Enabling question answering over tables and databases in natural language has become a key capability in the democratization of insights from tabular data sources. These systems first require retrieval of data that is relevant to a given natural language query, for which several methods have been introduced. In this work we present and study a table retrieval mechanism devising fine-grained typed query decomposition and global connectivity-awareness (DCTR), to handle the challenges induced by open-domain question answering over relational databases in complex usage contexts. We evaluate the effectiveness of the two mechanisms through the lens of retrieval complexity which we measure along the axes of query- and data complexity. Our analyses over industry-aligned benchmarks illustrate the robustness of DCTR for highly composite queries and densely connected databases. 

---
# Emotion Transcription in Conversation: A Benchmark for Capturing Subtle and Complex Emotional States through Natural Language 

**Authors**: Yoshiki Tanaka, Ryuichi Uehara, Koji Inoue, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2603.07138)  

**Abstract**: Emotion Recognition in Conversation (ERC) is critical for enabling natural human-machine interactions. However, existing methods predominantly employ categorical or dimensional emotion annotations, which often fail to adequately represent complex, subtle, or culturally specific emotional nuances. To overcome this limitation, we propose a novel task named Emotion Transcription in Conversation (ETC). This task focuses on generating natural language descriptions that accurately reflect speakers' emotional states within conversational contexts. To address the ETC, we constructed a Japanese dataset comprising text-based dialogues annotated with participants' self-reported emotional states, described in natural language. The dataset also includes emotion category labels for each transcription, enabling quantitative analysis and its application to ERC. We benchmarked baseline models, finding that while fine-tuning on our dataset enhances model performance, current models still struggle to infer implicit emotional states. The ETC task will encourage further research into more expressive emotion understanding in dialogue. The dataset is publicly available at this https URL. 

---
# Deep Expert Injection for Anchoring Retinal VLMs with Domain-Specific Knowledge 

**Authors**: Shuai Lu, Meng Wang, Jia Guo, Jiawei Du, Bo Liu, Shengzhu Yang, Weihang Zhang, Huazhu Fu, Huiqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.07131)  

**Abstract**: Large Vision Language Models (LVLMs) show immense potential for automated ophthalmic diagnosis. However, their clinical deployment is severely hindered by lacking domain-specific knowledge. In this work, we identify two structural deficiencies hindering reliable medical reasoning: 1) the Perception Gap, where general-purpose visual encoders fail to resolve fine-grained pathological cues (e.g., microaneurysms); and 2) the Reasoning Gap, where sparse visual evidence is progressively overridden by massive language priors in deeper transformer layers, leading to ungrounded hallucinations. To bridge these gaps, we propose EyExIn, a data-efficient framework designed to anchor retinal VLMs with expert knowledge via a Deep Expert Injection mechanism. Our architecture employs an Expert-Aware Dual-Stream encoding strategy that decouples visual representation into a general stream for anatomical context and a specialized expert stream for pathological semantics. To ensure high-fidelity integration, we design a Semantic-Adaptive Gated Fusion module, which dynamically amplifies subtle lesion signals while filtering irrelevant background noise. Furthermore, we introduce Adaptive Deep Expert Injection to embed persistent "Vision Anchors" by integrating fused visual features as residual biases directly into intermediate LLM layers. This mechanism creates a visual shortcut that forces the reasoning stack to remain strictly grounded in visual evidence. Extensive experiments across four benchmarks demonstrate that our model consistently outperforms massive proprietary systems. EyExIn significantly enhances domain-specific knowledge embedding and achieves state-of-the-art precision in ophthalmic visual question answering, advancing the development of trustworthy ophthalmic AI. 

---
# aCAPTCHA: Verifying That an Entity Is a Capable Agent via Asymmetric Hardness 

**Authors**: Zuyao Xu, Xiang Li, Fubin Wu, Yuqi Qiu, Lu Sun, FaSheng Miao  

**Link**: [PDF](https://arxiv.org/pdf/2603.07116)  

**Abstract**: As autonomous AI agents increasingly populate the Internet, a novel security challenge arises: "Is this entity an AI agent?" It is a new entity-type verification problem with no established solution. We formalize the problem through a three-class entity taxonomy (Human, Script, Agent) based on a verifiable agentic capability vector <x, r, s> (action, reasoning, and memory). A timing threshold t exploits the asymmetric hardness between human cognition and AI processing to separate the three classes. We define the Agentic Capability Verification Problem (ACVP) through three necessity primitives, each testing one capability dimension. Building on this foundation, we introduce aCAPTCHA (Agent CAPTCHA), a time-constrained security game for agent admission whose security rests on ACVP hardness under t. We instantiate aCAPTCHA through time-bounded natural-language understanding as a multi-round HTTP verification protocol, and evaluate it with preliminary agent trials that validate the protocol's soundness and completeness. aCAPTCHA provides a composable, infrastructure-free admission gate for any service where entity-type verification is required. 

---
# Enhancing Consistency of Werewolf AI through Dialogue Summarization and Persona Information 

**Authors**: Yoshiki Tanaka, Takumasa Kaneko, Hiroki Onozeki, Natsumi Ezure, Ryuichi Uehara, Zhiyang Qi, Tomoya Higuchi, Ryutaro Asahara, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2603.07111)  

**Abstract**: The Werewolf Game is a communication game where players' reasoning and discussion skills are essential. In this study, we present a Werewolf AI agent developed for the AIWolfDial 2024 shared task, co-hosted with the 17th INLG. In recent years, large language models like ChatGPT have garnered attention for their exceptional response generation and reasoning capabilities. We thus develop the LLM-based agents for the Werewolf Game. This study aims to enhance the consistency of the agent's utterances by utilizing dialogue summaries generated by LLMs and manually designed personas and utterance examples. By analyzing self-match game logs, we demonstrate that the agent's utterances are contextually consistent and that the character, including tone, is maintained throughout the game. 

---
# Efficient Personalized Reranking with Semi-Autoregressive Generation and Online Knowledge Distillation 

**Authors**: Kai Cheng, Hao Wang, Wei Guo, Weiwen Liu, Yong Liu, Yawen Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07107)  

**Abstract**: Generative models offer a promising paradigm for the final stage reranking in multi-stage recommender systems, with the ability to capture inter-item dependencies within reranked lists. However, their practical deployment still faces two key challenges: (1) an inherent conflict between achieving high generation quality and ensuring low-latency inference, making it difficult to balance the two, and (2) insufficient interaction between user and item features in existing methods. To address these challenges, we propose a novel Personalized Semi-Autoregressive with online knowledge Distillation (PSAD) framework for reranking. In this framework, the teacher model adopts a semi-autoregressive generator to balance generation quality and efficiency, while its ranking knowledge is distilled online into a lightweight scoring network during joint training, enabling real-time and efficient inference. Furthermore, we propose a User Profile Network (UPN) that injects user intent and models interest dynamics, enabling deeper interactions between users and items. Extensive experiments conducted on three large-scale public datasets demonstrate that PSAD significantly outperforms state-of-the-art baselines in both ranking performance and inference efficiency. 

---
# mAVE: A Watermark for Joint Audio-Visual Generation Models 

**Authors**: Luyang Si, Leyi Pan, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07090)  

**Abstract**: As Joint Audio-Visual Generation Models see widespread commercial deployment, embedding watermarks has become essential for protecting vendor copyright and ensuring content provenance. However, existing techniques suffer from an architectural mismatch by treating modalities as decoupled entities, exposing a critical Binding Vulnerability. Adversaries exploit this via Swap Attacks by replacing authentic audio with malicious deepfakes while retaining the watermarked video. Because current detectors rely on independent verification ($Video_{wm}\vee Audio_{wm}$), they incorrectly authenticate the manipulated content, falsely attributing harmful media to the original vendor and severely damaging their reputation. To address this, we propose mAVE (Manifold Audio-Visual Entanglement), the first watermarking framework natively designed for joint architectures. mAVE cryptographically binds audio and video latents at initialization without fine-tuning, defining a Legitimate Entanglement Manifold via Inverse Transform Sampling. Experiments on state-of-the-art models (LTX-2, MOVA) demonstrate that mAVE guarantees performance-losslessness and provides an exponential security bound against Swap Attacks. Achieving near-perfect binding integrity ($>99\%$), mAVE offers a robust cryptographic defense for vendor copyright. 

---
# Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR 

**Authors**: Muhammad Khalifa, Zohaib Khan, Omer Tafveez, Hao Peng, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07084)  

**Abstract**: Reward hacking is a form of misalignment in which models overoptimize proxy rewards without genuinely solving the underlying task. Precisely measuring reward hacking occurrence remains challenging because true task rewards are often expensive or impossible to compute. We introduce Countdown-Code, a minimal environment where models can both solve a mathematical reasoning task and manipulate the test harness. This dual-access design creates a clean separation between proxy rewards (test pass/fail) and true rewards (mathematical correctness), enabling accurate measurement of reward-hacking rates. Using this environment, we study reward hacking in open-weight LLMs and find that such behaviors can be unintentionally learned during supervised fine-tuning (SFT) when even a small fraction of reward-hacking trajectories leak into training data. As little as 1\% contamination in distillation SFT data is sufficient for models to internalize reward hacking which resurfaces during subsequent reinforcement learning (RL). We further show that RL amplifies misalignment and drives its generalization beyond the original domain. We open-source our environment and code to facilitate future research on reward hacking in LLMs. Our results reveal a previously underexplored pathway through which reward hacking can emerge and persist in LLMs, underscoring the need for more rigorous validation of synthetic SFT data. Code is available at this https URL. 

---
# User Review Writing via Interview with Dialogue Systems 

**Authors**: Yoshiki Tanaka, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2603.07070)  

**Abstract**: User reviews on e-commerce and review sites are crucial for making purchase decisions, although creating detailed reviews is time-consuming and labor-intensive. In this study, we propose a novel use of dialogue systems to facilitate user review creation by generating reviews from information gathered during interview dialogues with users. To validate our approach, we implemented our system using GPT-4 and conducted comparative experiments from the perspectives of system users and review readers. The results indicate that participants who used our system rated their interactions positively. Additionally, reviews generated by our system required less editing to achieve user satisfaction compared to those by the baseline. We also evaluated the reviews from the reader' perspective and found that our system-generated reviews are more helpful than those written by humans. Despite challenges with the fluency of the generated reviews, our method offers a promising new approach to review writing. 

---
# MedSteer: Counterfactual Endoscopic Synthesis via Training-Free Activation Steering 

**Authors**: Trong-Thang Pham, Loc Nguyen, Anh Nguyen, Hien Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2603.07066)  

**Abstract**: Generative diffusion models are increasingly used for medical imaging data augmentation, but text prompting cannot produce causal training data. Re-prompting rerolls the entire generation trajectory, altering anatomy, texture, and background. Inversion-based editing methods introduce reconstruction error that causes structural drift. We propose MedSteer, a training-free activation-steering framework for endoscopic synthesis. MedSteer identifies a pathology vector for each contrastive prompt pair in the cross-attention layers of a diffusion transformer. At inference time, it steers image activations along this vector, generating counterfactual pairs from scratch where the only difference is the steered concept. All other structure is preserved by construction. We evaluate MedSteer across three experiments on Kvasir v3 and HyperKvasir. On counterfactual generation across three clinical concept pairs, MedSteer achieves flip rates of 0.800, 0.925, and 0.950, outperforming the best inversion-based baseline in both concept flip rate and structural preservation. On dye disentanglement, MedSteer achieves 75% dye removal against 20% (PnP) and 10% (h-Edit). On downstream polyp detection, augmenting with MedSteer counterfactual pairs achieves ViT AUC of 0.9755 versus 0.9083 for quantity-matched re-prompting, confirming that counterfactual structure drives the gain. Code is at link this https URL 

---
# Looking Back and Forth: Cross-Image Attention Calibration and Attentive Preference Learning for Multi-Image Hallucination Mitigation 

**Authors**: Xiaochen Yang, Hao Fang, Jiawei Kong, Yaoxin Mao, Bin Chen, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2603.07048)  

**Abstract**: Although large vision-language models (LVLMs) have demonstrated remarkable capabilities, they are prone to hallucinations in multi-image tasks. We attribute this issue to limitations in existing attention mechanisms and insufficient cross-image modeling. Inspired by this, we propose a structured hallucination mitigation framework involving Cross-Image Attention calibration and Preference Learning (CAPL). CAPL explicitly enhances inter-image interactions at the architectural level while reinforcing reliance on genuine cross-image evidence during training, thereby improving the model's perception and modeling of cross-image associations. Specifically, we (i) introduce a selectable image token interaction attention mechanism to establish fine-grained cross-image entity alignment and information flow; (ii) design a cross-image modeling-based preference optimization strategy that contrasts reasoning outcomes under full inter-image interaction and those obtained when images are mutually invisible, encouraging the model to ground its predictions in authentic visual evidence and mitigating erroneous inferences driven by textual priors. Experimental results demonstrate that CAPL consistently improves performance across multiple model architectures, achieving stable gains on both multi-image hallucination and general benchmarks. Notably, performance on single-image visual tasks remains stable or slightly improves, indicating strong generalization capability. 

---
# Hit-RAG: Learning to Reason with Long Contexts via Preference Alignment 

**Authors**: Junming Liu, Yuqi Li, Shiping Wen, Zhigang Zeng, Tingwen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07023)  

**Abstract**: Despite the promise of Retrieval-Augmented Generation in grounding Multimodal Large Language Models with external knowledge, the transition to extensive contexts often leads to significant attention dilution and reasoning hallucinations. The surge in information density causes critical evidence to be submerged by voluminous noise, which complicates the discernment of relevant fragments within a dense input. In this paper, we propose \textbf{Hit-RAG}, a multi-stage preference alignment framework designed to resolve these cognitive bottlenecks through a progressive optimization pipeline. Our approach systematically refines the utilization of external evidence via three distinct stages. First, Supervised Fine-tuning establishes baseline context awareness to minimize information neglect. Next, Discriminative Preference Alignment enhances robustness against misleading distractors. Finally, Group-Relative Policy Optimization stabilizes logical synthesis to prevent reasoning collapse. Extensive evaluations on eight benchmarks demonstrate that Hit-RAG consistently yields substantial performance gains, enabling models to bridge the gap between context acquisition and accurate reasoning while surpassing much larger counterparts in long-context scenarios. 

---
# RESCHED: Rethinking Flexible Job Shop Scheduling from a Transformer-based Architecture with Simplified States 

**Authors**: Xiangjie Xiao, Cong Zhang, Wen Song, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.07020)  

**Abstract**: Neural approaches to the Flexible Job Shop Scheduling Problem (FJSP), particularly those based on deep reinforcement learning (DRL), have gained growing attention in recent years. However, existing methods rely on complex feature-engineered state representations (i.e., often requiring more than 20 handcrafted features) and graph-biased neural architectures. To reduce modeling complexity and advance a more generalizable framework for FJSP, we introduce \textsc{ReSched}, a minimalist DRL framework that rethinks both the scheduling formulation and model design. First, by revisiting the Markov Decision Process (MDP) formulation of FJSP, we condense the state space to just four essential features, eliminating historical dependencies through a subproblem-based perspective. Second, we employ Transformer blocks with dot-product attention, augmented by three lightweight but effective architectural modifications tailored to scheduling tasks. Extensive experiments show that \textsc{ReSched} outperforms classical dispatching rules and state-of-the-art DRL methods on FJSP. Moreover, \textsc{ReSched} also generalizes well to the Job Shop Scheduling Problem (JSSP) and the Flexible Flow Shop Scheduling Problem (FFSP), achieving competitive performance against neural baselines specifically designed for these variants. 

---
# Can Safety Emerge from Weak Supervision? A Systematic Analysis of Small Language Models 

**Authors**: Punyajoy Saha, Sudipta Halder, Debjyoti Mondal, Subhadarshi Panda  

**Link**: [PDF](https://arxiv.org/pdf/2603.07017)  

**Abstract**: Safety alignment is critical for deploying large language models (LLMs) in real-world applications, yet most existing approaches rely on large human-annotated datasets and static red-teaming benchmarks that are costly, difficult to scale, and slow to adapt to evolving model behaviors. Moreover, overly conservative safety mechanisms can reduce model usefulness by rejecting sensitive but legitimate queries. We introduce Self-MOA (Self Multi-Objective Alignment), a fully automated framework for aligning small language models using weak supervision from automated evaluator models. Self-MOA operates as a closed loop that dynamically generates model-specific red team prompts, constructs preference data from model-generated responses, and aligns models via multi-objective preference optimization to jointly optimize for safety and helpfulness. Across multiple small language models and safety benchmarks, Self-MOA achieves a 12.41\% improvement in safety while preserving helpfulness, using as little as 11 times less training data than human-supervised alignment baselines. These results demonstrate that adaptive, automated alignment can reduce the dependence on static, human-curated safety pipelines in resource-constrained settings. 

---
# SuperSkillsStack: Agency, Domain Knowledge, Imagination, and Taste in Human-AI Design Education 

**Authors**: Qian Huang, King Wang Poon  

**Link**: [PDF](https://arxiv.org/pdf/2603.07016)  

**Abstract**: This study examines how students integrate generative artificial intelligence (AI) into design projects through the lens of the SuperSkillsStack framework, which identifies four key human competencies for effective human-AI collaboration: Agency, Domain Knowledge, Imagination, and Taste. As generative AI increasingly transforms creative practice, design education must consider how human capabilities are cultivated alongside technological tools. Using qualitative thematic analysis, this study analyzes reflective writings from 80 student design teams participating in a human-centered design course. The findings show that students primarily used AI during the early stages of the design process, including brainstorming, information synthesis, and problem framing. However, students consistently relied on human judgment to interpret contextual information, validate AI-generated outputs, and refine design solutions. Domain knowledge derived from field observations enabled students to detect inaccuracies in AI suggestions, while taste played a critical role in evaluating and selecting meaningful ideas. The results suggest that generative AI functions primarily as a cognitive accelerator rather than a replacement for human creativity. The study highlights the importance of cultivating higher-order human capabilities to support effective human-AI collaboration in design education. 

---
# Foundational World Models Accurately Detect Bimanual Manipulator Failures 

**Authors**: Isaac R. Ward, Michelle Ho, Houjun Liu, Aaron Feldman, Joseph Vincent, Liam Kruse, Sean Cheong, Duncan Eddy, Mykel J. Kochenderfer, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2603.06987)  

**Abstract**: Deploying visuomotor robots at scale is challenging due to the potential for anomalous failures to degrade performance, cause damage, or endanger human life. Bimanual manipulators are no exception; these robots have vast state spaces comprised of high-dimensional images and proprioceptive signals. Explicitly defining failure modes within such state spaces is infeasible. In this work, we overcome these challenges by training a probabilistic, history informed, world model within the compressed latent space of a pretrained vision foundation model (NVIDIA's Cosmos Tokenizer). The model outputs uncertainty estimates alongside its predictions that serve as non-conformity scores within a conformal prediction framework. We use these scores to develop a runtime monitor, correlating periods of high uncertainty with anomalous failures. To test these methods, we use the simulated Push-T environment and the Bimanual Cable Manipulation dataset, the latter of which we introduce in this work. This new dataset features trajectories with multiple synchronized camera views, proprioceptive signals, and annotated failures from a challenging data center maintenance task. We benchmark our methods against baselines from the anomaly detection and out-of-distribution detection literature, and show that our approach considerably outperforms statistical techniques. Furthermore, we show that our approach requires approximately one twentieth of the trainable parameters as the next-best learning-based approach, yet outperforms it by 3.8% in terms of failure detection rate, paving the way toward safely deploying manipulator robots in real-world environments where reliability is non-negotiable. 

---
# Masked Unfairness: Hiding Causality within Zero ATE 

**Authors**: Zou Yang, Sophia Xiao, Bijan Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2603.06984)  

**Abstract**: Recent work has proposed powerful frameworks, rooted in causal theory, to quantify fairness. Causal inference has primarily emphasized the detection of \emph{average} treatment effects (ATEs), and subsequent notions of fairness have inherited this focus. In this paper, we build on previous concerns about regulation based on averages. In particular, we formulate the "causal masking problem" as a linear program that optimizes an alternative objective, such as maximizing profit or minimizing crime, while retaining a zero ATE (i.e., the ATE between a protected attribute and a decision). By studying the capabilities and limitations of causal masking, we show that optimization under ATE-based regulation may induce significant unequal treatment. We demonstrate that the divergence between true and causally masked fairness is driven by confounding, underscoring the importance of full conditional-independence testing when assessing fairness. Finally, we discuss statistical and information-theoretic limitations that make causally masked solutions very difficult to detect, allowing them to persist for long periods. These results argue that we must regulate fairness at the model-level, rather than at the decision level. 

---
# Diffusion Controller: Framework, Algorithms and Parameterization 

**Authors**: Tong Yang, Moonkyung Ryu, Chih-Wei Hsu, Guy Tennenholtz, Yuejie Chi, Craig Boutilier, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.06981)  

**Abstract**: Controllable diffusion generation often relies on various heuristics that are seemingly disconnected without a unified understanding. We bridge this gap with Diffusion Controller (DiffCon), a unified control-theoretic view that casts reverse diffusion sampling as state-only stochastic control within (generalized) linearly-solvable Markov Decision Processes (LS-MDPs). Under this framework, control acts by reweighting the pretrained reverse-time transition kernels, balancing terminal objectives against an $f$-divergence cost. From the resulting optimality conditions, we derive practical reinforcement learning methods for diffusion fine-tuning: (i) f-divergence-regularized policy-gradient updates, including a PPO-style rule, and (ii) a regularizer-determined reward-weighted regression objective with a minimizer-preservation guarantee under the Kullback-Leibler (KL) divergence. The LS-MDP framework further implies a principled model form: the optimal score decomposes into a fixed pretrained baseline plus a lightweight control correction, motivating a side-network parameterization conditioned on exposed intermediate denoising outputs, enabling effective gray-box adaptation with a frozen backbone. Experiments on Stable Diffusion v1.4 across supervised and reward-driven finetuning show consistent gains in preference-alignment win rates and improved quality-efficiency trade-offs versus gray-box baselines and even the parameter-efficient white-box adapter LoRA. 

---
# NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent Reinforcement Learning 

**Authors**: Addison Kalanther, Sanika Bharvirkar, Shankar Sastry, Chinmay Maheshwari  

**Link**: [PDF](https://arxiv.org/pdf/2603.06977)  

**Abstract**: Multi-agent reinforcement learning (MARL) is increasingly used to design learning-enabled agents that interact in shared environments. However, training MARL algorithms in general-sum games remains challenging: learning dynamics can become unstable, and convergence guarantees typically hold only in restricted settings such as two-player zero-sum or fully cooperative games. Moreover, when agents have heterogeneous and potentially conflicting preferences, it is unclear what system-level objective should guide learning. In this paper, we propose a new MARL pipeline called Near-Potential Policy Optimization (NePPO) for computing approximate Nash equilibria in mixed cooperative--competitive environments. The core idea is to learn a player-independent potential function such that the Nash equilibrium of a cooperative game with this potential as the common utility approximates a Nash equilibrium of the original game. To this end, we introduce a novel MARL objective such that minimizing this objective yields the best possible potential function candidate and consequently an approximate Nash equilibrium of the original game. We develop an algorithmic pipeline that minimizes this objective using zeroth-order gradient descent and returns an approximate Nash equilibrium policy. We empirically show the superior performance of this approach compared to popular baselines such as MAPPO, IPPO and MADDPG. 

---
# A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity 

**Authors**: Muhammad Arslan Shaukat, Muntasir Adnan, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2603.06976)  

**Abstract**: We present the first large-scale, cross-domain evaluation of document chunking strategies for dense retrieval, addressing a critical but underexplored aspect of retrieval-augmented systems. In our study, 36 segmentation methods spanning fixed-size, semantic, structure-aware, hierarchical, adaptive, and LLM-assisted approaches are benchmarked across six diverse knowledge domains using five different embedding models. Retrieval performance is assessed using graded relevance scores from a state-of-the-art LLM evaluator, with Normalised DCG@5 as the primary metric (complemented by Hit@5 and MRR). Our experiments show that content-aware chunking significantly improves retrieval effectiveness over naive fixed-length splitting. The top-performing strategy, Paragraph Group Chunking, achieved the highest overall accuracy (mean nDCG@5~0.459) and substantially better top-rank hit rates (Precision@1~24%, Hit@5~59%). In contrast, simple fixed-size character chunking as baselines performed poorly (nDCG@5 < 0.244, Precision@1~2-3%). We observe pronounced domain-specific differences: dynamic token sizing is strongest in biology, physics and health, while paragraph grouping is strongest in legal and maths. Larger embedding models yield higher absolute scores but remain sensitive to suboptimal segmentation, indicating that better chunking and large embeddings provide complementary benefits. In addition to accuracy gains, we quantify the efficiency trade-offs of advanced chunking. Producing more, smaller chunks can increase index size and latency. Consequently, we identify methods (like dynamic chunking) that approach an optimal balance of effectiveness and efficiency. These findings establish chunking as a vital lever for improving retrieval performance and reliability. 

---
# Elenchus: Generating Knowledge Bases from Prover-Skeptic Dialogues 

**Authors**: Bradley P. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06974)  

**Abstract**: We present Elenchus, a dialogue system for knowledge base construction grounded in inferentialist semantics, where knowledge engineering is re-conceived as explicitation rather than extraction from expert testimony or textual content. A human expert develops a bilateral position (commitments and denials) about a topic through prover-skeptic dialogue with a large language model (LLM) opponent. The LLM proposes tensions (claims that parts of the position are jointly incoherent) which the expert resolves by retraction, refinement, or contestation. The LLM thus serves as a defeasible derivability oracle whose unreliability is structurally contained by the expert's authority. Our main technical contribution is a mapping from Elenchus dialectical states to material bases in Hlobil and Brandom's NonMonotonic MultiSuccedent (NMMS) logic, satisfying Containment and enabling the elaboration of logical vocabulary that makes explicit the inferential relationships negotiated in the dialectic. We demonstrate the approach on the W3C PROV-O provenance ontology, where a single dialogue session elicits and structures design tensions that a domain expert can articulate, corresponding to decisions documented in a retrospective analysis of the ontology's design. Using pyNMMS, an automated NMMS reasoner, we verify that the structural properties of the resulting material base (nontransitivity, nonmonotonicity, and independence) correspond to specific PROV design rationales, demonstrating end-to-end integration from dialogue through formal reasoning. 

---
# Learning Quadruped Walking from Seconds of Demonstration 

**Authors**: Ruipeng Zhang, Hongzhan Yu, Ya-Chien Chang, Chenghao Li, Henrik I. Christensen, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.06961)  

**Abstract**: Quadruped locomotion provides a natural setting for understanding when model-free learning can outperform model-based control design, by exploiting data patterns to bypass the difficulty of optimizing over discrete contacts and the combinatorial explosion of mode changes. We give a principled analysis of why imitation learning with quadrupeds can be inherently effective in a small data regime, based on the structure of its limit cycles, Poincaré return maps, and local numerical properties of neural networks. The understanding motivates a new imitation learning method that regulates the alignment between variations in a latent space and those over the output actions. Hardware experiments confirm that a few seconds of demonstration is sufficient to train various locomotion policies from scratch entirely offline with reasonable robustness. 

---
# Post-Training with Policy Gradients: Optimality and the Base Model Barrier 

**Authors**: Alireza Mousavi-Hosseini, Murat A. Erdogdu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06957)  

**Abstract**: We study post-training linear autoregressive models with outcome and process rewards. Given a context $\boldsymbol{x}$, the model must predict the response $\boldsymbol{y} \in Y^N$, a sequence of length $N$ that satisfies a $\gamma$ margin condition, an extension of the standard separability to sequences. We prove that on test samples where the base model achieves a non-trivial likelihood $\alpha$, a variant of policy gradient (PG) can achieve likelihood $1 - \varepsilon$ with an essentially minimax optimal number of reward queries $\tilde{O}((\alpha^{-1} + \varepsilon^{-1})/\gamma^2)$. However, a barrier arises for going beyond the support of the base model. We prove that the overall expected error after post-training with outcome rewards is governed by a property of the base model called the Likelihood Quantile (LQ), and that variants of PG, while minimax optimal, may require a number of reward queries exponential in $N$ to go beyond this support, regardless of the pre-training algorithm. To overcome this barrier, we study post-training with a process reward model, and demonstrate how PG variants in this setting avoid the curse of dimensionality in $N$ via dependence on a token-level LQ. Along the way, we prove that under the margin condition, SGD with adaptive learning rate (LR) achieves a near optimal test error for statistical learning, and PG with adaptive LR achieves a near optimal number of mistakes for online learning while being computationally efficient whenever possible, both of which may be of independent interest. 

---
# How Private Are DNA Embeddings? Inverting Foundation Model Representations of Genomic Sequences 

**Authors**: Sofiane Ouaari, Jules Kreuer, Nico Pfeifer  

**Link**: [PDF](https://arxiv.org/pdf/2603.06950)  

**Abstract**: DNA foundation models have become transformative tools in bioinformatics and healthcare applications. Trained on vast genomic datasets, these models can be used to generate sequence embeddings, dense vector representations that capture complex genomic information. These embeddings are increasingly being shared via Embeddings-as-a-Service (EaaS) frameworks to facilitate downstream tasks, while supposedly protecting the privacy of the underlying raw sequences. However, as this practice becomes more prevalent, the security of these representations is being called into question. This study evaluates the resilience of DNA foundation models to model inversion attacks, whereby adversaries attempt to reconstruct sensitive training data from model outputs. In our study, the model's output for reconstructing the DNA sequence is a zero-shot embedding, which is then fed to a decoder. We evaluated the privacy of three DNA foundation models: DNABERT-2, Evo 2, and Nucleotide Transformer v2 (NTv2). Our results show that per-token embeddings allow near-perfect sequence reconstruction across all models. For mean-pooled embeddings, reconstruction quality degrades as sequence length increases, though it remains substantially above random baselines. Evo 2 and NTv2 prove to be most vulnerable, especially for shorter sequences with reconstruction similarities > 90%, while DNABERT-2's BPE tokenization provides the greatest resilience. We found that the correlation between embedding similarity and sequence similarity was a key predictor of reconstruction success. Our findings emphasize the urgent need for privacy-aware design in genomic foundation models prior to their widespread deployment in EaaS settings. Training code, model weights and evaluation pipeline are released on: this https URL. 

---
# MindfulAgents: Personalizing Mindfulness Meditation via an Expert-Aligned Multi-Agent System 

**Authors**: Mengyuan, Zhihan Jiang, Yuang Fan, Richard Feng, Sahiti Dharmavaram, Mathew Polowitz, Shawn Fallon, Bashima Islam, Lizbeth Benson, Irene Tung, David Creswell, Xuhai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06926)  

**Abstract**: Mindfulness meditation is a widely accessible and evidence-based method for supporting mental health. Despite the proliferation of mindfulness meditation apps, sustaining user engagement remains a persistent challenge. Personalizing the meditation experience is a promising strategy to improve engagement, but it often requires costly and unscalable manual effort. We present MindfulAgents, a multi-agent system powered by large language models that (1) generates guided meditation scripts based on an expert-established mindfulness framework, (2) encourages users' reflection on emotional states and mindfulness skills, and (3) enables real-time personalization of the mindfulness meditation experience for each user. In a formative lab study (N=13), MindfulAgents significantly improved in-session engagement (p = 0.011) and self-awareness (p = 0.014), and reduced momentary stress (p = 0.020). Furthermore, a four-week deployment study (N=62) demonstrated a notable increase in long-term engagement (p = 0.002) and level of mindfulness (p = 0.023). Participants reported that MindfulAgents offered more relevant meditation sessions personalized to individual needs in various contexts, supporting sustained practice. Our findings highlight the potential of LLM-driven personalization for enhancing user engagement in digital mindfulness meditation interventions. 

---
# Physics-informed AI Accelerated Retention Analysis of Ferroelectric Vertical NAND: From Day-Scale TCAD to Second-Scale Surrogate Model 

**Authors**: Gyujun Jeong, Sungwon Cho, Minji Shon, Namhoon Kim, Woohyun Hwang, Kwangyou Seo, Suhwan Lim, Wanki Kim, Daewon Ha, Prasanna Venkatesan, Kihang Youn, Ram Cherukuri, Yiyi Wang, Suman Datta, Asif Khan, Shimeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06881)  

**Abstract**: Ferroelectric field-effect transistors (FeFET)-based vertical NAND (Fe-VNAND) has emerged as a promising candidate to overcome z-scaling limitations with lower programming voltages. However, the data retention of 3D Fe-VNAND is hindered by the complex interaction between charge detrapping and ferroelectric depolarization. Developing optimized device designs requires exploring an extensive parameter space, but the high computational cost of conventional Technology Computer-Aided Design (TCAD) tools makes such wide-scale optimization impractical. To overcome these simulation barriers, we present a Physics-Informed Neural Operator (PINO)-based AI surrogate model designed for high-efficiency prediction of threshold voltage (Vth) shifts and retention behavior. By embedding fundamental physical principles into the learning architecture, our PINO framework achieves a speedup exceeding 10000x compared to TCAD while maintaining physical accuracy. This study demonstrates the model's effectiveness on a single FeFET configuration, serving as a pathway toward modeling the retention loss mechanisms. 

---
# A prior information informed learning architecture for flying trajectory prediction 

**Authors**: Xianda Huang, Zidong Han, Ruibo Jin, Zhenyu Wang, Wenyu Li, Xiaoyang Li, Yi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.06863)  

**Abstract**: Trajectory prediction for flying objects is critical in domains ranging from sports analytics to aerospace. However, traditional methods struggle with complex physical modeling, computational inefficiencies, and high hardware demands, often neglecting critical trajectory events like landing points. This paper introduces a novel, hardware-efficient trajectory prediction framework that integrates environmental priors with a Dual-Transformer-Cascaded (DTC) architecture. We demonstrate this approach by predicting the landing points of tennis balls in real-world outdoor courts. Using a single industrial camera and YOLO-based detection, we extract high-speed flight coordinates. These coordinates, fused with structural environmental priors (e.g., court boundaries), form a comprehensive dataset fed into our proposed DTC model. A first-level Transformer classifies the trajectory, while a second-level Transformer synthesizes these features to precisely predict the landing point. Extensive ablation and comparative experiments demonstrate that integrating environmental priors within the DTC architecture significantly outperforms existing trajectory prediction frameworks 

---
# Supporting Artifact Evaluation with LLMs: A Study with Published Security Research Papers 

**Authors**: David Heye, Karl Kindermann, Robin Decker, Johannes Lohmöller, Anastasiia Belova, Sandra Geisler, Klaus Wehrle, Jan Pennekamp  

**Link**: [PDF](https://arxiv.org/pdf/2603.06862)  

**Abstract**: Artifact Evaluation (AE) is essential for ensuring the transparency and reliability of research, closing the gap between exploratory work and real-world deployment is particularly important in cybersecurity, particularly in IoT and CPSs, where large-scale, heterogeneous, and privacy-sensitive data meet safety-critical actuation. Yet, manual reproducibility checks are time-consuming and do not scale with growing submission volumes. In this work, we demonstrate that Large Language Models (LLMs) can provide powerful support for AE tasks: (i) text-based reproducibility rating, (ii) autonomous sandboxed execution environment preparation, and (iii) assessment of methodological pitfalls. Our reproducibility-assessment toolkit yields an accuracy of over 72% and autonomously sets up execution environments for 28% of runnable cybersecurity artifacts. Our automated pitfall assessment detects seven prevalent pitfalls with high accuracy ($F_1$ > 92%). Hence, the toolkit significantly reduces reviewer effort and, when integrated into established AE processes, could incentivize authors to submit higher-quality and more reproducible artifacts. IoT, CPS, and cybersecurity conferences and workshops may integrate the toolkit into their peer-review processes to support reviewers' decisions on awarding artifact badges, improving the overall sustainability of the process. 

---
# Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration 

**Authors**: Yanjun Chen, Yirong Sun, Hanlin Wang, Xinming Zhang, Xiaoyu Shen, Wenjie Li, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06859)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) systems powered by large language models (LLMs) are frequently optimized via sparse terminal-only feedback. This shared signal entangles upstream decisions, obstructing accurate decision-level credit assignment. To address this trajectory-level diffusion, we introduce Contextual Counterfactual Credit Assignment (\textbf{\texttt{C3}}). Instead of distributing rewards across an entire episode, \textbf{\texttt{C3}} isolates the causal impact of individual messages by freezing the exact transcript-derived context, evaluating context-matched alternatives via fixed-continuation replay, and applying a leave-one-out (LOO) baseline. This localized intervention extracts unbiased, low-variance marginal advantages for standard policy-gradient optimization. Evaluated across five mathematical and coding benchmarks under matched budgets, \textbf{\texttt{C3}} improves terminal performance over established baselines. Mechanistic diagnostics further show that these gains are accompanied by higher credit fidelity, lower contextual variance, and stronger inter-agent causal dependence. Our code is available at this https URL. 

---
# Are Audio-Language Models Listening? Audio-Specialist Heads for Adaptive Audio Steering 

**Authors**: Neta Glazer, Lenny Aharon, Ethan Fetaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.06854)  

**Abstract**: Multimodal large language models can exhibit text dominance, over-relying on linguistic priors instead of grounding predictions in non-text inputs. One example is large audio-language models (LALMs) where decisive audio evidence can be under-utilized even when it contains important information. To address this issue we use mechanistic interpretability to identify a small set of audio-specialist attention heads whose audio attention yields a ``listening'' signal. We show that this signal increases when audio evidence affects the model's output, providing an indicator of audio engagement under standard prompting. Leveraging this localization, we construct an audio--silence steering direction and apply an inference-time activation intervention to the final representation, amplifying the model's audio effect. To demonstrate the utility of this intervention, we show on MMAU that this improves accuracy by up to +8.0 percentage points on two Qwen-based LALMs, without any parameter updates. 

---
# Twitch: Learning Abstractions for Equational Theorem Proving 

**Authors**: Guy Axelrod, Moa Johansson, Nicholas Smallbone  

**Link**: [PDF](https://arxiv.org/pdf/2603.06849)  

**Abstract**: Several successful strategies in automated reasoning rely on human-supplied guidance about which term or clause shapes are interesting. In this paper we aim to discover interesting term shapes automatically. Specifically, we discover abstractions : term patterns that occur over and over again in relevant proofs. We present our tool Twitch which discovers abstractions with the help of Stitch, a tool originally developed for discovering reusable library functions in program synthesis tasks. Twitch can produce abstractions in two ways: (1) from a partial, failed proof of a conjecture; (2) from successful proofs of other theorems in the same domain. We have also extended Twee, an equational theorem prover, to use these abstractions. We evaluate Twitch on a set of unit equality (UEQ) problems from TPTP, and show that it can prove 12 rating-1 problems as well as yielding significant speed-ups on many other problems. 

---
# Step-Level Visual Grounding Faithfulness Predicts Out-of-Distribution Generalization in Long-Horizon Vision-Language Models 

**Authors**: Md Ashikur Rahman, Md Arifur Rahman, Niamul Hassan Samin, Abdullah Ibne Hanif Arean, Juena Ahmed Noshin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06828)  

**Abstract**: We uncover a behavioral law of long-horizon vision-language models: models that maintain temporally grounded beliefs generalize better. Standard benchmarks measure only final-answer accuracy, which obscures how models use visual information; a model can guess correctly while its step-by-step reasoning is entirely unanchored to the visual input. We formalize this as behavioral faithfulness over long horizons, an empirically measurable property that quantifies whether a model's intermediate reasoning remains consistent with the evolving visual state. Across eight models on three long-horizon benchmarks, we demonstrate that temporal grounding quality is a leading indicator of robustness: the Step Grounding Rate (SGR) predicts out-of-distribution retention with $r = 0.83$ (permutation test $p = 0.003$), a relationship that holds within capacity-matched models and cannot be explained by scale or in-distribution accuracy. Critically, grounding quality varies by up to 10.8 percentage points within parameter-matched 7B models despite similar accuracy, revealing it as an independent axis of model capability. Multiple robustness checks confirm the signal reflects genuine visual reliance: counterfactual traces drop SGR by 26--41 percentage points, cross-architecture verifiers agree at $\rho = 0.96$, random reasoning scores near chance ($\sim 18\%$), and the predictor remains strong even without explicit reasoning disclosure ($r = 0.78$). 

---
# "Dark Triad" Model Organisms of Misalignment: Narrow Fine-Tuning Mirrors Human Antisocial Behavior 

**Authors**: Roshni Lulla, Fiona Collins, Sanaya Parekh, Thilo Hagendorff, Jonas Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2603.06816)  

**Abstract**: The alignment problem refers to concerns regarding powerful intelligences, ensuring compatibility with human preferences and values as capabilities increase. Current large language models (LLMs) show misaligned behaviors, such as strategic deception, manipulation, and reward-seeking, that can arise despite safety training. Gaining a mechanistic understanding of these failures requires empirical approaches that can isolate behavioral patterns in controlled settings. We propose that biological misalignment precedes artificial misalignment, and leverage the Dark Triad of personality (narcissism, psychopathy, and Machiavellianism) as a psychologically grounded framework for constructing model organisms of misalignment. In Study 1, we establish comprehensive behavioral profiles of Dark Triad traits in a human population (N = 318), identifying affective dissonance as a central empathic deficit connecting the traits, as well as trait-specific patterns in moral reasoning and deceptive behavior. In Study 2, we demonstrate that dark personas can be reliably induced in frontier LLMs through minimal fine-tuning on validated psychometric instruments. Narrow training datasets as small as 36 psychometric items resulted in significant shifts across behavioral measures that closely mirrored human antisocial profiles. Critically, models generalized beyond training items, demonstrating out-of-context reasoning rather than memorization. These findings reveal latent persona structures within LLMs that can be readily activated through narrow interventions, positioning the Dark Triad as a validated framework for inducing, detecting, and understanding misalignment across both biological and artificial intelligence. 

---
# AI-Assisted Curation of Conference Scholarship: Compiling, Structuring, and Analyzing Two Decades of Presentations at the Society for Social Work and Research 

**Authors**: Brian Perron, Bryan Victor, Zia Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.06814)  

**Abstract**: Purpose: This study developed a comprehensive database of presentation abstracts from the Society for Social Work and Research (SSWR) Annual Conference and examined patterns in research methodology, authorship, collaboration, and institutional participation over two decades.
Method: Abstract metadata was compiled from the SSWR Confex conference management system for presentations from 2005 to 2026 using web scraping. A small language model (gpt-oss:20b) performed classification and extraction tasks on abstracts, including categorization of methodologies and parsing of author affiliations, with human review at each major stage to ensure accuracy.
Results: The database contains 23,793 presentations with 69,924 author records representing 20,779 unique researchers from 4,049 institutions across 93 countries. Annual conference presentations increased from 423 in 2005 to 1,935 in 2026, representing a compound annual growth rate of 7.5%. Quantitative methods predominated (61.1%), followed by qualitative approaches (23.4%), mixed methods (9.1%), and reviews (5.4%). The mean number of authors per presentation increased from 2.22 in 2005 to 3.31 in 2026. International participation grew from 4.5% to 13.5% of author affiliations over the observation period.
Discussion: Findings indicate substantial growth in SSWR conference participation, alongside increased collaboration and international engagement. The methodological distribution reveals continued quantitative predominance with growing qualitative representation. This database provides research infrastructure for systematic hypothesis testing about research priorities and disciplinary development over time, enabling analyses that inform both scholarship and conference planning. 

---
# A Hybrid Machine Learning Model for Cerebral Palsy Detection 

**Authors**: Karan Kumar Singh, Nikita Gajbhiye, Gouri Sankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2603.06803)  

**Abstract**: The development of effective treatments for Cerebral Palsy (CP) can begin with the early identification of affected children while they are still in the early stages of the disorder. Pathological issues in the brain can be better diagnosed with the use of one of many medical imaging techniques. Magnetic Resonance Imaging (MRI) has revolutionized medical imaging with its unparalleled image resolution. A unique Machine Learning (ML) model that was built to identify CP disorder is presented in this paper. The model is intended to assist in the early diagnosis of CP in newborns. In this study, the brain MRI images dataset was first collected, and then the preprocessing techniques were applied to this dataset to make it ready for use in the proposed model. Following this, the proposed model was constructed by combining three CNN models, specifically VGG 19, Efficient-Net, and the ResNet50 model, to extract features from the image. Following this, a Bi-LSTM was utilized as a classifier to determine whether or not CP was present, and finally, the proposed model was employed for training and testing. The results show that the proposed model achieved an accuracy of 98.83%, which is higher than VGG-19 (96.79%), Efficient-Net (97.29%), and VGG-16 (97.50%).. When the suggested model is compared to other models that have been pre-trained in the past, the accuracy scores seem to be much higher. 

---
# Optimistic Policy Regularization 

**Authors**: Mai Pham, Vikrant Vaze, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06793)  

**Abstract**: Deep reinforcement learning agents frequently suffer from premature convergence, where early entropy collapse causes the policy to discard exploratory behaviors before discovering globally optimal strategies. We introduce Optimistic Policy Regularization (OPR), a lightweight mechanism designed to preserve and reinforce historically successful trajectories during policy optimization. OPR maintains a dynamic buffer of high-performing episodes and biases learning toward these behaviors through directional log-ratio reward shaping and an auxiliary behavioral cloning objective. When instantiated on Proximal Policy Optimization (PPO), OPR substantially improves sample efficiency on the Arcade Learning Environment. Across 49 Atari games evaluated at the 10-million step benchmark, OPR achieves the highest score in 22 environments despite baseline methods being reported at the standard 50-million step horizon. Beyond arcade benchmarks, OPR also generalizes to the CAGE Challenge 2 cyber-defense environment, surpassing the competition-winning Cardiff agent while using the same PPO architecture. These results demonstrate that anchoring policy updates to empirically successful trajectories can improve both sample efficiency and final performance. 

---
# Physics-Informed Diffusion Model for Generating Synthetic Extreme Rare Weather Events Data 

**Authors**: Marawan Yakout, Tannistha Maiti, Monira Majhabeen, Tarry Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.06782)  

**Abstract**: Data scarcity is a primary obstacle in developing robust Machine Learning (ML) models for detecting rapidly intensifying tropical cyclones. Traditional data augmentation techniques (rotation, flipping, brightness adjustment) fail to preserve the physical consistency and high-intensity gradients characteristic of rare Category 4-equivalent events, which constitute only 0.14\% of our dataset (202 of 140,514 samples). We propose a physics-informed diffusion model based on the Context-UNet architecture to generate synthetic, multi-spectral satellite imagery of extreme weather events. Our model is conditioned on critical atmospheric parameters such as average wind speed, type of Ocean and stage of development (early, mature, late etc) -- the known drivers of rapid intensification. Using a controlled pre-generated noise sampling strategy and mixed-precision training, we generated $16\times16$ wind-field samples that are cropped from multi-spectral satellite imagery which preserve realistic spatial autocorrelation and physical consistency. Results demonstrate that our model successfully learns discriminative features across ten distinct context classes, effectively mitigating the data bottleneck. Specifically, we address the extreme class imbalance in our dataset, where Class 4 (Ocean 2, early stage with average wind speed 50kn hurricane) contains only 202 samples compared to 79,768 samples in Class 0. This generative framework provides a scalable solution for augmenting training datasets for operational weather detection algorithms. The average Results yield an average Log-Spectral Distance (LSD) of 4.5dB, demonstrating a scalable framework for enhancing operational weather detection algorithms. 

---
# xaitimesynth: A Python Package for Evaluating Attribution Methods for Time Series with Synthetic Ground Truth 

**Authors**: Gregor Baer  

**Link**: [PDF](https://arxiv.org/pdf/2603.06781)  

**Abstract**: Evaluating time series attribution methods is difficult because real-world datasets rarely provide ground truth for which time points drive a prediction. A common workaround is to generate synthetic data where class-discriminating features are placed at known locations, but each study currently reimplements this from scratch. We introduce xaitimesynth, a Python package that provides reusable infrastructure for this evaluation approach. The package generates synthetic time series following an additive model where each sample is a sum of background signal and a localized, class-discriminating feature, with the feature window automatically tracked as a ground truth mask. A fluent data generation API and YAML configuration format allow flexible and reproducible dataset definitions for both univariate and multivariate time series. The package also provides standard localization metrics, including AUC-PR, AUC-ROC, Relevance Mass Accuracy, and Relevance Rank Accuracy. xaitimesynth is open source and available at this https URL. 

---
# SpatialMAGIC: A Hybrid Framework Integrating Graph Diffusion and Spatial Attention for Spatial Transcriptomics Imputation 

**Authors**: Sayeem Bin Zaman, Fahim Hafiz, Riasat Azim  

**Link**: [PDF](https://arxiv.org/pdf/2603.06780)  

**Abstract**: Spatial transcriptomics (ST) enables mapping gene expression with spatial context but is severely affected by high sparsity and technical noise, which conceals true biological signals and hinders downstream analyses. To address these challenges, SpatialMagic was proposed, which is a hybrid imputation model combining MAGIC-based graph diffusion with transformer-based spatial self-attention. The long-range dependencies in the gene expression are captured by graph diffusion, and local neighborhood structure is captured by spatial attention models, which allow for recovering the missing expression values, retaining spatial consistency. Across multiple platforms, SpatialMagic consistently outperforms existing baselines, including MAGIC and attention-based models, achieving peak Adjusted Rand Index (ARI) scores in clustering accuracy of 0.3301 on high-resolution Stereo-Seq data, 0.3074 on Slide-Seq, and 0.4216 on the Sci-Space dataset. Beyond quantitative improvements, SpatialMagic substantially enhances downstream biological analyses by improving the detection of both up- and down-regulated genes while maintaining regulatory consistency across datasets. The pathway enrichment analysis of the recovered genes indicates that they are involved in consistent processes across key metabolic, transport, and neural signaling pathways, suggesting that the framework improves data quality while preserving biological interpretability. Overall, SpatialMagic's hybrid diffusion attention strategy and refinement module outperform state-of-the-art baselines on quantitative metrics and provide a better understanding of the imputed data by preserving tissue architecture and uncovering biologically relevant genes. The source code and datasets are provided in the following link: this https URL 

---
# HGT-Scheduler: Deep Reinforcement Learning for the Job Shop Scheduling Problem via Heterogeneous Graph Transformers 

**Authors**: Bulent Soykan  

**Link**: [PDF](https://arxiv.org/pdf/2603.06777)  

**Abstract**: The Job Shop Scheduling Problem (JSSP) is commonly formulated as a disjunctive graph in which nodes represent operations and edges encode technological precedence constraints as well as machine-sharing conflicts. Most existing reinforcement learning approaches model this graph as homogeneous, merging job-precedence and machine-contention edges into a single relation type. Such a simplification overlooks the intrinsic heterogeneity of the problem structure and may lead to the loss of critical relational information. To address this limitation, we propose the Heterogeneous Graph Transformer (HGT)-Scheduler, a reinforcement learning framework that models the JSSP as a heterogeneous graph. The proposed architecture leverages a Heterogeneous Graph Transformer to capture type-specific relational patterns through edge-type-dependent attention mechanisms applied to precedence and contention relations. The scheduling policy is trained using Proximal Policy Optimization. The effectiveness of the proposed method is evaluated on the Fisher--Thompson benchmark instances. On the FT06 instance, the HGT-Scheduler achieves an optimality gap of 8.4\%, statistically outperforming both an identical architecture that ignores edge types ($p = 0.011$) and a standard Graph Isomorphism Network baseline. On the larger FT10 instance, the approach demonstrates favorable scalability. However, under a 50,000-step training limit, the performance of heterogeneous and homogeneous graph models is comparable, suggesting that edge-type awareness requires longer training horizons for larger problem instances. Ablation analyses further indicate that a three-layer attention architecture provides the best performance. Overall, the results confirm that explicitly modeling distinct edge semantics improves the learning of effective scheduling policies. 

---
# Failure Detection in Chemical Processes using Symbolic Machine Learning: A Case Study on Ethylene Oxidation 

**Authors**: Julien Amblard, Niklas Groll, Matthew Tait, Mark Law, Gürkan Sin, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2603.06767)  

**Abstract**: Over the past decade, Artificial Intelligence has significantly advanced, mostly driven by large-scale neural approaches. However, in the chemical process industry, where safety is critical, these methods are often unsuitable due to their brittleness, and lack of explainability and interpretability. Furthermore, open-source real-world datasets containing historical failures are scarce in this domain.
In this paper, we investigate an approach for predicting failures in chemical processes using symbolic machine learning and conduct a feasibility study in the context of an ethylene oxidation process. Our method builds on a state-of-the-art symbolic machine learning system capable of learning predictive models in the form of probabilistic rules from context-dependent noisy examples. This system is a general-purpose symbolic learner, which makes our approach independent of any specific chemical process. To address the lack of real-world failure data, we conduct our feasibility study leveraging data generated from a chemical process simulator. Experimental results show that symbolic machine learning can outperform baseline methods such as random forest and multilayer perceptron, while preserving interpretability through the generation of compact, rule-based predictive models. Finally, we explain how such learned rule-based models could be integrated into agents to assist chemical plant operators in decision-making during potential failures. 

---
# Diversity-Aware Adaptive Collocation for Physics-Informed Neural Networks via Sparse QUBO Optimization and Hybrid Coresets 

**Authors**: Hadi Salloum, Maximilian Mifsud Bonici, Sinan Ibrahim, Pavel Osinenko, Alexei Kornaev  

**Link**: [PDF](https://arxiv.org/pdf/2603.06761)  

**Abstract**: Physics-Informed Neural Networks (PINNs) enforce governing equations by penalizing PDE residuals at interior collocation points, but standard collocation strategies - uniform sampling and residual-based adaptive refinement - can oversample smooth regions, produce highly correlated point sets, and incur unnecessary training cost. We reinterpret collocation selection as a coreset construction problem: from a large candidate pool, select a fixed-size subset that is simultaneously informative (high expected impact on reducing PDE error) and diverse (low redundancy under a space-time similarity notion). We formulate this as a QUBO/BQM objective with linear terms encoding residual-based importance and quadratic terms discouraging redundant selections. To avoid the scalability issues of dense k-hot QUBOs, we propose a sparse graph-based BQM built on a kNN similarity graph and an efficient repair procedure that enforces an exact collocation budget. We further introduce hybrid coverage anchors to guarantee global PDE enforcement. We evaluate the method on the 1D time-dependent viscous Burgers equation with shock formation and report both accuracy and end-to-end time-to-accuracy, including a timing breakdown of selection overhead. Results demonstrate that sparse and hybrid formulations reduce selection overhead relative to dense QUBOs while matching or improving accuracy at fixed collocation budgets. 

---
# Gradient-based Nested Co-Design of Aerodynamic Shape and Control for Winged Robots 

**Authors**: Daniele Affinita, Mingda Xu, Benoît Valentin Gherardi, Pascal Fua  

**Link**: [PDF](https://arxiv.org/pdf/2603.06760)  

**Abstract**: Designing aerial robots for specialized tasks, from perching to payload delivery, requires tailoring their aerodynamic shape to specific mission requirements. For tasks involving wide flight envelopes, the usual sequential process of first determining the shape and then the motion planner is likely to be suboptimal due to the inherent nonlinear interactions between them. This limitation has been motivating co-design research, which involves jointly optimizing the aerodynamic shape and the motion planner. In this paper, we present a general-purpose, gradient-based, nested co-design framework where the motion planner solves an optimal control problem and the aerodynamic forces used in the dynamics model are determined by a neural surrogate model. This enables us to model complex subsonic flow conditions encountered in aerial robotics and to overcome the limited applicability of existing co-design methods. These limitations stem from the simplifying assumptions they require for computational tractability to either the planner or the aerodynamics. We validate our method on two complex dynamic tasks for fixed-wing gliders: perching and a short landing. Our optimized designs improve task performance compared to an evolutionary baseline in a fraction of the computation time. 

---
# Enhancing SHAP Explainability for Diagnostic and Prognostic ML Models in Alzheimer Disease 

**Authors**: Pablo Guillén, Enrique Frias-Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2603.06758)  

**Abstract**: Alzheimer disease (AD) diagnosis and prognosis increasingly rely on machine learning (ML) models. Although these models provide good results, clinical adoption is limited by the need for technical expertise and the lack of trustworthy and consistent model explanations. SHAP (SHapley Additive exPlanations) is com-monly used to interpret AD models, but existing studies tend to focus on explanations for isolated tasks, providing little evidence about their robustness across disease stages, model architectures, or prediction objectives. This paper proposes a multi-level explainability framework that measures the coherence, stabil-ity and consistency of explanations by integrating: (1) within-model coherence metrics between feature importance and SHAP, (2) SHAP stability across AD boundaries, and (3) SHAP cross-task consistency be-tween diagnosis and prognosis. Using AutoML to optimize classifiers on the NACC dataset, we trained four diagnostic and four prognostic models covering the standard AD progression stages. Stability was then evaluated using correlation metrics, top-k feature overlap, SHAP sign consistency, and domain-level contribution ratios. Results show that cognitive and functional markers dominate SHAP explanations in both diagnosis and prognosis. SHAP-SHAP consistency between diagnostic and prognostic models was high across all classifiers, with 100% sign stability and minimal shifts in explanatory magnitude. Domain-level contributions also remained stable, with only minimal increases in genetic features for prognosis. These results demonstrate that SHAP explanations can be quantitatively vali-dated for robustness and transferability, providing clinicians with more reliable interpretations of ML pre-dictions. 

---
# Learning Unbiased Cluster Descriptors for Interpretable Imbalanced Concept Drift Detection 

**Authors**: Yiqun Zhang, Zhanpei Huang, Mingjie Zhao, Chuyao Zhang, Yang Lu, Yuzhu Ji, Fangqing Gu, An Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.06757)  

**Abstract**: Unlabeled streaming data are usually collected to describe dynamic systems, where concept drift detection is a vital prerequisite to understanding the evolution of systems. However, the drifting concepts are usually imbalanced in most real cases, which brings great challenges to drift detection. That is, the dominant statistics of large clusters can easily mask the drifting of small cluster distributions (also called small concepts), which is known as the `masking effect'. Considering that most existing approaches only detect the overall existence of drift under the assumption of balanced concepts, two critical problems arise: 1) where the small concept is, and 2) how to detect its drift. To address the challenging concept drift detection for imbalanced data, we propose Imbalanced Cluster Descriptor-based Drift Detection (ICD3) approach that is unbiased to the imbalanced concepts. This approach first detects imbalanced concepts by employing a newly designed multi-distribution-granular search, which ensures that the distribution of both small and large concepts is effectively captured. Subsequently, it trains a One-Cluster Classifier (OCC) for each identified concept to carefully monitor their potential drifts in the upcoming data chunks. Since the detection is independently performed for each concept, the dominance of large clusters is thus circumvented. ICD3 demonstrates highly interpretability by specifically locating the drifted concepts, and is robust to the changing of the imbalance ratio of concepts. Comprehensive experiments with multi-aspect ablation studies conducted on various benchmark datasets demonstrate the superiority of ICD3 against the state-of-the-art counterparts. 

---
# XMACNet: An Explainable Lightweight Attention based CNN with Multi Modal Fusion for Chili Disease Classification 

**Authors**: Tapon Kumer Ray, Rajkumar Y, Shalini R, Srigayathri K, Jayashree S, Lokeswari P  

**Link**: [PDF](https://arxiv.org/pdf/2603.06750)  

**Abstract**: Plant disease classification via imaging is a critical task in precision agriculture. We propose XMACNet, a novel light-weight Convolutional Neural Network (CNN) that integrates self-attention and multi-modal fusion of visible imagery and vegetation indices for chili disease detection. XMACNet uses an EfficientNetV2S backbone enhanced by a self-attention module and a fusion branch that processes both RGB images and computed vegetation index maps (NDVI, NPCI, MCARI). We curated a new dataset of 12,000 chili leaf images across six classes (five disease types plus healthy), augmented synthetically via StyleGAN to mitigate data scarcity. Trained on this dataset, XMACNet achieves high accuracy, F1-score, and AUC, outperforming baseline models such as ResNet-50, MobileNetV2, and a Swin Transformer variant. Crucially, XMACNet is explainable: we use Grad-CAM++ and SHAP to visualize and quantify the models focus on disease features. The models compact size and fast inference make it suitable for edge deployment in real-world farming scenarios. 

---
# Robotic Foundation Models for Industrial Control: A Comprehensive Survey and Readiness Assessment Framework 

**Authors**: David Kube, Simon Hadwiger, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06749)  

**Abstract**: Robotic foundation models (RFMs) are emerging as a promising route towards flexible, instruction- and demonstration-driven robot control, however, a critical investigation of their industrial applicability is still lacking. This survey gives an extensive overview over the RFM-landscape and analyses, driven by concrete implications, how industrial domains and use cases shape the requirements of RFMs, with particular focus on collaborative robot platforms, heterogeneous sensing and actuation, edge-computing constraints, and safety-critical operation. We synthesise industrial deployment perspectives into eleven interdependent implications and operationalise them into an assessment framework comprising a catalogue of 149 concrete criteria, spanning both model capabilities and ecosystem requirements. Using this framework, we evaluate 324 manipulation-capable RFMs via 48,276 criterion-level decisions obtained via a conservative LLM-assisted evaluation pipeline, validated against expert judgements. The results indicate that industrial maturity is limited and uneven: even the highest-rated models satisfy only a fraction of criteria and typically exhibit narrow implication-specific peaks rather than integrated coverage. We conclude that progress towards industry-grade RFMs depends less on isolated benchmark successes than on systematic incorporation of safety, real-time feasibility, robust perception, interaction, and cost-effective system integration into auditable deployment stacks. 

---
# Property-driven Protein Inverse Folding With Multi-Objective Preference Alignment 

**Authors**: Xiaoyang Hou, Junqi Liu, Chence Shi, Xin Liu, Zhi Yang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06748)  

**Abstract**: Protein sequence design must balance designability, defined as the ability to recover a target backbone, with multiple, often competing, developability properties such as solubility, thermostability, and expression. Existing approaches address these properties through post hoc mutation, inference-time biasing, or retraining on property-specific subsets, yet they are target dependent and demand substantial domain expertise or careful hyperparameter tuning. In this paper, we introduce ProtAlign, a multi-objective preference alignment framework that fine-tunes pretrained inverse folding models to satisfy diverse developability objectives while preserving structural fidelity. ProtAlign employs a semi-online Direct Preference Optimization strategy with a flexible preference margin to mitigate conflicts among competing objectives and constructs preference pairs using in silico property predictors. Applied to the widely used ProteinMPNN backbone, the resulting model MoMPNN enhances developability without compromising designability across tasks including sequence design for CATH 4.3 crystal structures, de novo generated backbones, and real-world binder design scenarios, making it an appealing framework for practical protein sequence design. 

---
# ButterflyViT: 354$\times$ Expert Compression for Edge Vision Transformers 

**Authors**: Aryan Karmore  

**Link**: [PDF](https://arxiv.org/pdf/2603.06746)  

**Abstract**: Deploying sparse Mixture of Experts(MoE) Vision Transformers remains a challenge due to linear expert memory scaling. Linear memory scaling stores $N$ independent expert weight matrices requiring $\mathcal{O}(N_E \cdot d^2)$ memory, which exceeds edge devices memory budget. Current compression methods like quantization, pruning and low-rank factorization reduce constant factors but leave the scaling bottleneck unresolved. We introduce ButterflyViT, a method that treats experts not as independent weight matrices but as geometric reorientations of a unified shared quantized substrate. Diversity among experts arises from viewing different angles of shared capacity, not from redundant storage. By applying learned rotations to a shared ternary prototype, each expert yields $\mathcal{O}(d_{\text{model}} \cdot d_{\text{ff}} + N_E \cdot n_\ell \cdot d)$ memory which is sub-linear in the number of experts. To address the unique challenges of vision, a spatial smoothness regulariser is introduced that penalises routing irregularities between adjacent patch tokens, turning patch correlation into a training signal. Across image classification tasks on CIFAR-100, ButterflyViT achieves 354$\times$ memory reduction at 64 experts with negligible accuracy loss. ButterflyViT allows multiple experts to fit on edge-constrained devices showing that geometric parameterization breaks linear scaling. 

---
# Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection 

**Authors**: Minjae Kang, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.06745)  

**Abstract**: Large Language Models (LLMs), despite advances in instruction tuning, often fail to follow complex user instructions. Activation steering techniques aim to mitigate this by manipulating model internals, but have a potential risk of oversteering, where excessive emphasis on the instruction degrades task accuracy and overall text quality. To address this, we introduce DIRECTER (Dynamic rejection steering), a novel steering method that dynamically modulates steering strength by scaling the KV cache without extra dataset. DIRECTER couples steering with a plausibility-guided decoding loop, which adaptively adjusts steering strength at each step by comparing the steered output distribution to the original. If the steered output is deemed implausible, steering strength is progressively weakened. This strength modulation is guided by a lightweight, one-time attention sensitivity analysis that ranks layers by their influence on model representations. Extensive evaluations show that DIRECTER significantly enhances instruction-following capabilities across diverse benchmarks, improving accuracy by up to 6.5% over baselines without the common trade-offs in generation quality or task fidelity. The proposed dynamic, plausibility-guided control during activation steering further demonstrates its potential as a general mechanism for mitigating oversteering that is compatible with existing baselines. 

---
# Stabilizing Reinforcement Learning for Diffusion Language Models 

**Authors**: Jianyuan Zhong, Kaibo Wang, Ding Ding, Zijin Feng, Haoli Bai, Yang Xiang, Jiacheng Sun, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06743)  

**Abstract**: Group Relative Policy Optimization (GRPO) is highly effective for post-training autoregressive (AR) language models, yet its direct application to diffusion large language models (dLLMs) often triggers reward collapse. We identify two sources of incompatibility. First, GRPO relies on importance ratios defined by sequence probabilities, which are intractable in dLLMs and must be estimated (e.g., via ELBO-based or mean-field likelihood proxies), yielding inherently noisy ratios. Second, standard GRPO's formulation is not designed for estimated ratios: its conditional clipping can be anomalously bypassed by model-agnostic estimation noise, producing gradient spikes, while its fixed group-size normalization amplifies gradient-magnitude fluctuations under high-variance ratio estimates. We show these effects form a self-reinforcing instability loop that drives policy drift and further increases ratio variance. To break this loop, we propose StableDRL, a reformulation of GRPO tailored for dLLMs that uses (i) unconditional clipping to suppress outlier-induced spikes and (ii) self-normalization to constrain updates within the convex hull of per-sample gradients. We further extend StableDRL to block-wise diffusion models via a staircase attention mechanism. 

---
# Improved Constrained Generation by Bridging Pretrained Generative Models 

**Authors**: Xiaoxuan Liang, Saeid Naderiparizi, Yunpeng Liu, Berend Zwartsenberg, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2603.06742)  

**Abstract**: Constrained generative modeling is fundamental to applications such as robotic control and autonomous driving, where models must respect physical laws and safety-critical constraints. In real-world settings, these constraints rarely take the form of simple linear inequalities, but instead complex feasible regions that resemble road maps or other structured spatial domains. We propose a constrained generation framework that generates samples directly within such feasible regions while preserving realism. Our method fine-tunes a pretrained generative model to enforce constraints while maintaining generative fidelity. Experimentally, our method exhibits characteristics distinct from existing fine-tuning and training-free constrained baselines, revealing a new compromise between constraint satisfaction and sampling quality. 

---
# Heterogeneous Decentralized Diffusion Models 

**Authors**: Zhiying Jiang, Raihan Seraj, Marcos Villagra, Bidhan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.06741)  

**Abstract**: Training frontier-scale diffusion models often requires substantial computational resources concentrated in tightly coupled clusters, limiting participation to well-resourced institutions. While Decentralized Diffusion Models (DDM) enable training multiple experts in isolation, existing approaches require 1176 GPU-days and homogeneous training objectives across all experts. We present an efficient framework that reduces resource requirements while supporting heterogeneous training objectives. Our approach combines three contributions: (1) a heterogeneous decentralized training paradigm that allows experts to use different objectives (DDPM and Flow Matching), unified at inference time via a deterministic schedule-aware conversion into a common velocity space without retraining; (2) pretrained checkpoint conversion from ImageNet-DDPM to Flow Matching objectives, accelerating convergence and enabling initialization without objective-specific pretraining; and (3) PixArt-alpha's efficient AdaLN-Single architecture, reducing parameters while maintaining quality. Experiments on LAION-Aesthetics show that, relative to the training scale reported for prior DDM work, our approach reduces compute from 1176 to 72 GPU-days (16x) and data from 158M to 11M (14x). Under aligned inference settings, our heterogeneous 2DDPM:6FM configuration achieves better FID (11.88 vs. 12.45) and higher intra-prompt diversity (LPIPS 0.631 vs. 0.617) than the homogeneous 8FM baseline. By eliminating synchronization requirements and enabling mixed DDPM/FM objectives, our framework lowers infrastructure requirements for decentralized generative model training. 

---
# ViroGym: Realistic Large-Scale Benchmarks for Evaluating Viral Proteins 

**Authors**: Yichen Zhou, Jonathan Golob, Amir Karimi, Stefan Bauer, Patrick Schwab  

**Link**: [PDF](https://arxiv.org/pdf/2603.06740)  

**Abstract**: Protein language models (pLMs) have shown strong potential in prediction of the functional effects of missense variants in zero-shot settings. Despite this progress, benchmarking pLMs for viral proteins remains limited and systematic strategies for integrating in silico metrics with in vitro validation to guide antigen and target selection are underdeveloped. Here, we introduce ViroGym, a comprehensive benchmark designed to evaluate variant effect prediction in viral proteins and to facilitate selecting rational antigen candidates. We curated 79 deep mutational scanning (DMS) assays encompassing eukaryotic viruses, collectively comprising 552,937 mutated amino acid sequences across 7 distinct phenotypic readouts, and 21 influenza virus neutralisation tasks and a real-world predictive task for SARS-CoV-2. We benchmark well-established pLMs on fitness landscapes, antigenic diversity, and pandemic forecasting to provide a framework for vaccine selection, and show that pLMs selected using in vitro experimental data excel at predicting dominant circulating mutations in real world. 

---
# ResearchEnvBench: Benchmarking Agents on Environment Synthesis for Research Code Execution 

**Authors**: Yubang Wang, Chenxi Zhang, Bowen Chen, Zezheng Huai, Zihao Dai, Xinchi Chen, Yuxin Wang, Yining Zheng, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06739)  

**Abstract**: Autonomous agents are increasingly expected to support scientific research, and recent benchmarks report progress in code repair and autonomous experimentation. However, these evaluations typically assume a pre-configured execution environment, which requires resolving complex software dependencies, aligning hardware and framework versions, and configuring distributed execution, yet this capability remains largely unbenchmarked. We introduce ResearchEnvBench, a benchmark for environment synthesis in research code execution. Given a research repository, documentation, and a target execution setting, agents must construct an environment that successfully executes at runtime. Evaluations on diverse research repositories reveal a substantial gap in current SOTA agents, with failures dominated by incomplete dependency resolution and brittle version coupling. ResearchEnvBench provides a realistic testbed for advancing autonomous agents toward reproducible scientific research. 

---
# Rank-Factorized Implicit Neural Bias: Scaling Super-Resolution Transformer with FlashAttention 

**Authors**: Dongheon Lee, Seokju Yun, Jaegyun Im, Youngmin Ro  

**Link**: [PDF](https://arxiv.org/pdf/2603.06738)  

**Abstract**: Recent Super-Resolution~(SR) methods mainly adopt Transformers for their strong long-range modeling capability and exceptional representational capacity. However, most SR Transformers rely heavily on relative positional bias~(RPB), which prevents them from leveraging hardware-efficient attention kernels such as FlashAttention. This limitation imposes a prohibitive computational burden during both training and inference, severely restricting attempts to scale SR Transformers by enlarging the training patch size or the self-attention window. Consequently, unlike other domains that actively exploit the inherent scalability of Transformers, SR Transformers remain heavily focused on effectively utilizing limited receptive fields. In this paper, we propose Rank-factorized Implicit Neural Bias~(RIB), an alternative to RPB that enables FlashAttention in SR Transformers. Specifically, RIB approximates positional bias using low-rank implicit neural representations and concatenates them with pixel content tokens in a channel-wise manner, turning the element-wise bias addition in attention score computation into a dot-product operation. Further, we introduce a convolutional local attention and a cyclic window strategy to fully leverage the advantages of long-range interactions enabled by RIB and FlashAttention. We enlarge the window size up to \textbf{96$\times$96} while jointly scaling the training patch size and the dataset size, maximizing the benefits of Transformers in the SR task. As a result, our network achieves \textbf{35.63\,dB PSNR} on Urban100$\times$2, while reducing training and inference time by \textbf{2.1$\times$} and \textbf{2.9$\times$}, respectively, compared to the RPB-based SR Transformer~(PFT). 

---
# Agent Hunt: Bounty Based Collaborative Autoformalization With LLM Agents 

**Authors**: Chad E. Brown, Cezary Kaliszyk, Josef Urban  

**Link**: [PDF](https://arxiv.org/pdf/2603.06737)  

**Abstract**: We describe an experiment in large-scale autoformalization of algebraic topology in an Interactive Theorem Proving (ITP) environment, where the workload is distributed among multiple LLM-based coding agents. Rather than relying on static central planning, we implement a simulated bounty-based marketplace in which agents dynamically propose new lemmas (formal statements), attach bounties to them, and compete to discharge these proof obligations and claim the bounties. The agents interact directly with the interactive proof system: they can invoke tactics, inspect proof states and goals, analyze tactic successes and failures, and iteratively refine their proof scripts. In addition to constructing proofs, agents may introduce new formal definitions and intermediate lemmas to structure the development. All accepted proofs are ultimately checked and verified by the underlying proof assistant. This setting explores collaborative, decentralized proof search and theory building, and the use of market-inspired mechanisms to scale autoformalization in ITP. 

---
# Calibrated Credit Intelligence: Shift-Robust and Fair Risk Scoring with Bayesian Uncertainty and Gradient Boosting 

**Authors**: Srikumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.06733)  

**Abstract**: Credit risk scoring must support high-stakes lending decisions where data distributions change over time, probability estimates must be reliable, and group-level fairness is required. While modern machine learning models improve default prediction accuracy, they often produce poorly calibrated scores under distribution shift and may create unfair outcomes when trained without explicit constraints. This paper proposes Calibrated Credit Intelligence (CCI), a deployment-oriented framework that combines (i) a Bayesian neural risk scorer to capture epistemic uncertainty and reduce overconfident errors, (ii) a fairnessconstrained gradient boosting model to control group disparities while preserving strong tabular performance, and (iii) a shiftaware fusion strategy followed by post-hoc probability calibration to stabilize decision thresholds in later time periods. We evaluate CCI on the Home Credit Credit Risk Model Stability benchmark using a time-consistent split to reflect real-world drift. Compared with strong baselines (LightGBM, XGBoost, CatBoost, TabNet, and a standalone Bayesian neural model), CCI achieves the best overall trade-off between discrimination, calibration, stability, and fairness. In particular, CCI reaches an AUC-ROC of 0.912 and an AUC-PR of 0.438, improves operational performance with Recall@1%FPR = 0.509, and reduces calibration error (Brier score 0.087, ECE 0.015). Under temporal shift, CCI shows a smaller AUC-PR drop from early to late periods (0.017), and it lowers group disparities (demographic parity gap 0.046, equal opportunity gap 0.037) compared to unconstrained boosting. These results indicate that CCI produces risk scores that are accurate, reliable, and more equitable under realistic deployment conditions. 

---
# Don't Freeze, Don't Crash: Extending the Safe Operating Range of Neural Navigation in Dense Crowds 

**Authors**: Jiefu Zhang, Yang Xu, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2603.06729)  

**Abstract**: Navigating safely through dense crowds requires collision avoidance that generalizes beyond the densities seen during training. Learning-based crowd navigation can break under out-of-distribution crowd sizes due to density-sensitive observation normalization and social-cost scaling, while analytical solvers often remain safe but freeze in tight interactions. We propose a reinforcement learning approach for dense, variable-density navigation that attains zero-shot density generalization using a density-invariant observation encoding with density-randomized training and physics-informed proxemic reward shaping with density-adaptive scaling. The encoding represents the distance-sorted $K$ nearest pedestrians plus bounded crowd summaries, keeping input statistics stable as crowd size grows. Trained with $N\!\in\![11,16]$ pedestrians in a $3\mathrm{m}\times3\mathrm{m}$ arena and evaluated up to $N\!=\!21$ pedestrians ($1.3\times$ denser), our policy reaches the goal in $>99\%$ of episodes and achieves $86\%$ collision-free success in random crowds, with markedly less freezing than analytical methods and a $>\!60$-point collision-free margin over learning-based benchmark methods. Codes are available at \href{this https URL}{this https URL}. 

---
# Safe Transformer: An Explicit Safety Bit For Interpretable And Controllable Alignment 

**Authors**: Jingyuan Feng, Andrew Gambardella, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2603.06727)  

**Abstract**: Current safety alignment methods encode safe behavior implicitly within model parameters, creating a fundamental opacity: we cannot easily inspect why a model refuses a request, nor intervene when its safety judgments fail. We propose Safe Transformer, a modular approach that augments pre-trained language models by inserting a discrete information bottleneck containing an explicit safety bit between transformer layers. The safety bit serves as both an interpretable signal of the model's safety classification and a controllable switch: through contrastive training, the model learns disentangled representations where the safety bit governs the behavioral mode - producing helpful responses when $s=1$ and refusals when $s=0$ - while additional unsupervised bits $u$ encode semantic content for generation. Additional unsupervised bits in the information bottleneck allow semantic information to flow through, preserving the model's generation capabilities. This design achieves both interpretability (the safety decision is directly readable) and controllability (the safety bit can be manually overridden), requiring only lightweight fine-tuning without pre-training from scratch. In red-team benchmarks, Safe Transformer achieves near-zero Attack Success Rate, substantially outperforming base models and safety fine-tuning baselines. 

---
# Regression Models Meet Foundation Models: A Hybrid-AI Approach to Practical Electricity Price Forecasting 

**Authors**: Yunzhong Qiu, Binzhu Li, Hao Wei, Shenglin Weng, Chen Wang, Zhongyi Pei, Mingsheng Long, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06726)  

**Abstract**: Electricity market prices exhibit extreme volatility, nonlinearity, and non-stationarity, making accurate forecasting a significant challenge. While cutting-edge time series foundation models (TSFMs) effectively capture temporal dependencies, they typically underutilize cross-variate correlations and non-periodic patterns that are essential for price forecasting. Conversely, regression models excel at capturing feature interactions but are limited to future-available inputs, ignoring crucial historical drivers that are unavailable at forecast time. To bridge this gap, we propose FutureBoosting, a novel paradigm that enhances regression-based forecasts by integrating forecasted features generated from a frozen TSFM. This approach leverages the TSFM's ability to model historical patterns and injects these insights as enriched inputs into a downstream regression model. We instantiate this paradigm into a lightweight, plug-and-play framework for electricity price forecasting. Extensive evaluations on real-world electricity market data demonstrate that our framework consistently outperforms state-of-the-art TSFMs and regression baselines, achieving reductions in Mean Absolute Error (MAE) of more than 30% at most. Through ablation studies and explainable AI (XAI) techniques, we validate the contribution of forecasted features and elucidate the model's decision-making process. FutureBoosting establishes a robust, interpretable, and effective solution for practical market participation, offering a general framework for enhancing regression models with temporal context. 

---
# Bi Directional Feedback Fusion for Activity Aware Forecasting of Indoor CO2 and PM2.5 

**Authors**: Harshala Gammulle, Lidia Morawska, Sridha Sridharan, Clinton Fookes  

**Link**: [PDF](https://arxiv.org/pdf/2603.06724)  

**Abstract**: Indoor air quality (IAQ) forecasting plays a critical role in safeguarding occupant health, ensuring thermal comfort, and supporting intelligent building control. However, predicting future concentrations of key pollutants such as carbon dioxide (CO2) and fine particulate matter (PM2.5) remains challenging due to the complex interplay between environmental factors and highly dynamic occupant behaviours. Traditional data driven models primarily rely on historical sensor trajectories and often fail to anticipate behaviour induced emission spikes or rapid concentration shifts. To address these limitations, we present a dual stream bi directional feedback fusion framework that jointly models indoor environmental evolution and action derived embeddings representing human activities. The proposed architecture integrates a context aware modulation mechanism that adaptively scales and shifts each stream based on a shared, evolving fusion state, enabling the model to selectively emphasise behavioural cues or long term environmental trends. Furthermore, we introduce dual timescale temporal modules that independently capture gradual CO2 accumulation patterns and short term PM2.5 fluctuations. A composite loss function combining weighted mean squared error, spike aware penalties, and uncertainty regularisation facilitates robust learning under volatile indoor conditions. Extensive validation on real-world IAQ datasets demonstrates that our approach significantly outperforms state of the art forecasting baselines while providing interpretable uncertainty estimates essential for practical deployment in smart buildings and health-aware monitoring systems. 

---
# UWPD: A General Paradigm for Invisible Watermark Detection Agnostic to Embedding Algorithms 

**Authors**: Xiang Ao, Yiling Du, Zidan Wang, Mengru Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06723)  

**Abstract**: Invisible watermarks, as an essential technology for image copyright protection, have been widely deployed with the rapid development of social media and AIGC. However, existing invisible watermark detection heavily relies on prior knowledge of specific algorithms, leading to limited detection capabilities for "unknown watermarks" in open environments. To this end, we propose a novel task named Universal Watermark Presence Detection (UWPD), which aims to identify whether an image carries a copyright mark without requiring decoding information. We construct the UniFreq-100K dataset, comprising large-scale samples across various invisible watermark embedding algorithms. Furthermore, we propose the Frequency Shield Network (FSNet). This model deploys an Adaptive Spectral Perception Module (ASPM) in the shallow layers, utilizing learnable frequency gating to dynamically amplify high-frequency watermark signals while suppressing low-frequency semantics. In the deep layers, the network introduces Dynamic Multi-Spectral Attention (DMSA) combined with tri-stream extremum pooling to deeply mine watermark energy anomalies, forcing the model to precisely focus on sensitive frequency bands. Extensive experiments demonstrate that FSNet exhibits superior zero-shot detection capabilities on the UWPD task, outperforming existing baseline models. Code and datasets will be released upon acceptance. 

---
# ProtAlign: Contrastive learning paradigm for Sequence and structure alignment 

**Authors**: Aditya Ranganath, Hasin Us Sami, Kowshik Thopalli, Bhavya Kailkhura, Wesam Sakla  

**Link**: [PDF](https://arxiv.org/pdf/2603.06722)  

**Abstract**: Protein language models often take into consideration the alignment between a protein sequence and its textual description. However, they do not take structural information into consideration. Traditional methods treat sequence and structure separately, limiting the ability to exploit the alignment between the structure and protein sequence embeddings. In this paper, we introduce a sequence structure contrastive alignment framework, which learns a shared embedding space where proteins are represented consistently across modalities. By training on large-scale pairs of sequences and experimentally resolved or predicted structures, the model maximizes agreement between matched sequence structure pairs while pushing apart unrelated pairs. This alignment enables cross-modal retrieval (e.g., finding structural neighbors given a sequence), improves downstream prediction tasks such as function annotation and stability estimation, and provides interpretable links between sequence variation and structural organization. Our results demonstrate that contrastive learning can serve as a powerful bridge between protein sequences and structures, offering a unified representation for understanding and engineering proteins. 

---
# Dynamic Targeting of Satellite Observations Using Supplemental Geostationary Satellite Data and Hierarchical Planning 

**Authors**: Akseli Kangaslahti, Itai Zilberstein, Alberto Candela, Steve Chien  

**Link**: [PDF](https://arxiv.org/pdf/2603.06719)  

**Abstract**: The Dynamic Targeting (DT) mission concept is an approach to satellite observation in which a lookahead sensor gathers information about the upcoming environment and uses this information to intelligently plan observations. Previous work has shown that DT has the potential to increase the science return across applications. However, DT mission concepts must address challenges, such as the limited spatial extent of onboard lookahead data and instrument mobility, data throughput, and onboard computation constraints. In this work, we show how the performance of DT systems can be improved by using supplementary data streamed from geostationary satellites that provide lookahead information up to 35 minutes ahead of time rather than the 1 minute latency from an onboard lookahead sensor. While there is a greater volume of geostationary data, the search space for observation planning explodes exponentially with the size of the horizon. To address this, we introduce a hierarchical planning approach in which the geostationary data is used to plan a long-term observation blueprint in polynomial time, then the onboard lookahead data is leveraged to refine that plan over short-term horizons. We compare the performance of our approach to that of traditional DT planners relying on onboard lookahead data across four different problem instances: three cloud avoidance variations and a storm hunting scenario. We show that our hierarchical planner outperforms the traditional DT planners by up to 41% and examine the features of the scenarios that affect the performance of our approach. We demonstrate that incorporating geostationary satellite data is most effective for dynamic problem instances in which the targets of interest are sparsely distributed throughout the overflight. 

---
# Scaling Agentic Capabilities, Not Context: Efficient Reinforcement Finetuning for Large Toolspaces 

**Authors**: Karan Gupta, Pranav Vajreshwari, Yash Pandya, Raghav Magazine, Akshay Nambi, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.06713)  

**Abstract**: Agentic systems operating over large tool ecosystems must plan and execute long-horizon workflows under weak or non-verifiable supervision. While frontier models mitigate these challenges through scale and large context budgets, small language models (SLMs) remain brittle: eager tool loading saturates context, execution errors compound over time, and sparse rewards limit learning. We introduce ATLAS, a reinforcement finetuning framework that enables SLMs to operate effectively in large-scale toolspace environments by learning how to acquire context and how to execute actions. Our approach makes two key contributions. First, we treat context control and execution structure as learnable decisions, combining iterative tool loading with programmatic tool orchestration to bound context growth and stabilize long-horizon trajectories. Second, we propose rubric-based reinforcement finetuning, which decomposes task success into structured, task-aligned criteria and enables scalable training using small judge models. Across MCP benchmarks, these design choices yield large and consistent gains over generic RL baselines, allowing a 4B SLM to approach frontier-agent performance under far tighter parameter and context budgets. 

---
# Mining Beyond the Bools: Learning Data Transformations and Temporal Specifications 

**Authors**: Sam Nicholas Kouteili, William Fishell, Christian Scaff, Mark Santolucito, Ruzica Piskac  

**Link**: [PDF](https://arxiv.org/pdf/2603.06710)  

**Abstract**: Mining specifications from execution traces presents an automated way of capturing characteristic system behaviors. However, existing approaches are largely restricted to Boolean abstractions of events, limiting their ability to express data-aware properties. In this paper, we extend mining procedures to operate over richer datatypes. We first establish candidate functions in our domain that cover the set of traces by leveraging Syntax Guided Synthesis (SyGuS) techniques. To capture these function applications temporally, we formalize the semantics of TSL$_f$, a finite-prefix interpretation of Temporal Stream Logic (TSL) that extends LTL$_f$ with support for first-order predicates and functional updates. This allows us to unify a corresponding procedure for learning the data transformations and temporal specifications of a system. We demonstrate our approach synthesizing reactive programs from mined specifications on the OpenAI-Gymnasium ToyText environments, finding that our method is more robust and orders of magnitude more sample-efficient than passive learning baselines on generalized problem instances. 

---
# Thinking with Gaze: Sequential Eye-Tracking as Visual Reasoning Supervision for Medical VLMs 

**Authors**: Yiwei Li, Zihao Wu, Yanjun Lv, Hanqi Jiang, Weihang You, Zhengliang Liu, Dajiang Zhu, Xiang Li, Quanzheng Li, Tianming Liu, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.06697)  

**Abstract**: Vision--language models (VLMs) process images as visual tokens, yet their intermediate reasoning is often carried out in text, which can be suboptimal for visually grounded radiology tasks. Radiologists instead diagnose via sequential visual search; eye-tracking captures this process as time-ordered gaze trajectories that reveal how evidence is acquired over time. We use eye-gaze as supervision to guide VLM reasoning by introducing a small set of dedicated gaze tokens. These tokens are trained to predict gaze-selected image patch indices in temporal order, encouraging the model to follow human-like evidence acquisition and integration. Experiments on MIMIC-EYE and multiple external zero-shot benchmarks show consistent gains over baselines, achieving state-of-the-art in-domain performance and improved out-of-domain robustness. These results highlight temporally ordered gaze as an effective supervision signal for learning visually grounded medical reasoning. 

---
# Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning 

**Authors**: Zhengjian Yao, Yongzhi Li, Xinyuan Gao, Quan Chen, Peng Jiang, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06688)  

**Abstract**: We present "Narrative Weaver", a novel framework that addresses a fundamental challenge in generative AI: achieving multi-modal controllable, long-range, and consistent visual content generation. While existing models excel at generating high-fidelity short-form visual content, they struggle to maintain narrative coherence and visual consistency across extended sequences - a critical limitation for real-world applications such as filmmaking and e-commerce advertising. Narrative Weaver introduces the first holistic solution that seamlessly integrates three essential capabilities: fine-grained control, automatic narrative planning, and long-range coherence. Our architecture combines a Multimodal Large Language Model (MLLM) for high-level narrative planning with a novel fine-grained control module featuring a dynamic Memory Bank that prevents visual drift. To enable practical deployment, we develop a progressive, multi-stage training strategy that efficiently leverages existing pre-trained models, achieving state-of-the-art performance even with limited training data. Recognizing the absence of suitable evaluation benchmarks, we construct and release the E-commerce Advertising Video Storyboard Dataset (EAVSD) - the first comprehensive dataset for this task, containing over 330K high-quality images with rich narrative annotations. Through extensive experiments across three distinct scenarios (controllable multi-scene generation, autonomous storytelling, and e-commerce advertising), we demonstrate our method's superiority while opening new possibilities for AI-driven content creation. 

---
# VB: Visibility Benchmark for Visibility and Perspective Reasoning in Images 

**Authors**: Neil Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2603.06680)  

**Abstract**: We present VB, a benchmark that tests whether vision-language models can determine what is and is not visible in a photograph, and abstain when a human viewer cannot reliably answer. Each item pairs a single photo with a short yes/no visibility claim; the model must output VISIBLY_TRUE, VISIBLY_FALSE, or ABSTAIN, together with a confidence score. Items are organized into 100 families using a 2x2 design that crosses a minimal image edit with a minimal text edit, yielding 300 headline evaluation cells. Unlike prior unanswerable-VQA benchmarks, VB tests not only whether a question is unanswerable but why (via reason codes tied to specific visibility factors), and uses controlled minimal edits to verify that model judgments change when and only when the underlying evidence changes. We score models on confidence-aware accuracy with abstention (CAA), minimal-edit flip rate (MEFR), confidence-ranked selective prediction (SelRank), and second-order perspective reasoning (ToMAcc); all headline numbers are computed on the strict XOR subset (three cells per family, 300 scored items per model). We evaluate nine models spanning flagship and prior-generation closed-source systems, and open-source models from 8B to 12B parameters. GPT-4o and Gemini 3.1 Pro effectively tie for the best composite score (0.728 and 0.727), followed by Gemini 2.5 Pro (0.678). The best open-source model, Gemma 3 12B (0.505), surpasses one prior-generation closed-source system. Text-flip robustness exceeds image-flip robustness for six of nine models, and confidence calibration varies substantially: GPT-4o and Gemini 2.5 Pro achieve similar accuracy yet differ sharply in selective prediction quality. 

---
# Chart Deep Research in LVLMs via Parallel Relative Policy Optimization 

**Authors**: Jiajin Tang, Gaoyang, Wenjie Wang, Sibei Yang, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06677)  

**Abstract**: With the rapid advancement of data science, charts have evolved from simple numerical presentation tools to essential instruments for insight discovery and decision-making support. However, current chart data intelligence exhibits significant limitations in deep research capabilities, with existing methods predominantly addressing shallow tasks such as visual recognition or factual question-answering, rather than the complex reasoning and high-level data analysis that deep research requires. This limitation stems from two primary technical bottlenecks: at the training level, existing post-training techniques exhibit deficiencies in handling multi-dimensional reward signal interference and heterogeneous data gradient conflicts, preventing models from achieving balanced development across multiple capability dimensions; at the evaluation level, current methods remain limited to factual retrieval and basic computation, failing to assess end-to-end analytic reasoning and other deep research capabilities. To address the training challenge, we propose PRPO, which performs parallel optimization across reward dimensions and capability partitioning across data types, effectively disentangling conflicts between heterogeneous data and multi-dimensional reward signals while ensuring optimization stability. For the evaluation challenge, we construct MCDR-Bench based on the ``error uniqueness principle," transforming subjective generation assessment into objective error identification through controllable error injection, enabling quantifiable evaluation of deep research capabilities. Experimental validation confirms that the proposed PRPO and MCDR-Bench jointly establish a unified framework that systematically advances chart deep research through enhanced collaborative training and objective evaluation. 

---
# XAI and Few-shot-based Hybrid Classification Model for Plant Leaf Disease Prognosis 

**Authors**: Diana Susan Joseph, Pranav M Pawar, Raja Muthalagu, Mithun Mukharjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.06676)  

**Abstract**: Performing a timely and accurate identification of crop diseases is vital to maintain agricultural productivity and food security. The current work presents a hybrid few-shot learning model that integrates Explainable Artificial Intelligence (XAI) and Few-Shot Learning (FSL) to address the challenge of identifying and classifying the stages of disease of the diseases of maize, rice, and wheat leaves under limited annotated data conditions. The proposed model integrates Siamese and Prototypical Networks within an episodic training paradigm to effectively learn discriminative disease features from a few examples. To ensure model transparency and trustworthiness, Gradient-weighted Class Activation Mapping (Grad-CAM) is employed for visualizing key decision regions in the leaf images, offering interpretable insights into the classification process. Experimental evaluations on custom few-shot datasets developed in the study prove that the model consistently achieves high accuracy, precision, recall, and F1-scores, frequently exceeding 92% across various disease stages. Comparative analyses against baseline FSL models further confirm the superior performance and explainability of the proposed approach. The framework offers a promising solution for real-world, data-constrained agricultural disease monitoring applications. 

---
# AutoFigure-Edit: Generating Editable Scientific Illustration 

**Authors**: Zhen Lin, Qiujie Xie, Minjun Zhu, Shichen Li, Qiyao Sun, Enhao Gu, Yiran Ding, Ke Sun, Fang Guo, Panzhong Lu, Zhiyuan Ning, Yixuan Weng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06674)  

**Abstract**: High-quality scientific illustrations are essential for communicating complex scientific and technical concepts, yet existing automated systems remain limited in editability, stylistic controllability, and efficiency. We present AutoFigure-Edit, an end-to-end system that generates fully editable scientific illustrations from long-form scientific text while enabling flexible style adaptation through user-provided reference images. By combining long-context understanding, reference-guided styling, and native SVG editing, it enables efficient creation and refinement of high-quality scientific illustrations. To facilitate further progress in this field, we release the video at this https URL, full codebase at this https URL and provide a website for easy access and interactive use at this https URL. 

---
# Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study 

**Authors**: Yixiao Jing, Chaoyu Zhang, Zixuan Zhong, Peizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06672)  

**Abstract**: Semantic noise initialization has been reported to improve robustness and controllability in image diffusion models. Whether these gains transfer to text-to-video (T2V) generation remains unclear, since temporal coupling can introduce extra degrees of freedom and instability. We benchmark semantic noise initialization against standard Gaussian noise using a frozen VideoCrafter-style T2V diffusion backbone and VBench on 100 prompts. Using prompt-level paired tests with bootstrap confidence intervals and a sign-flip permutation test, we observe a small positive trend on temporal-related dimensions; however, the 95 percent confidence interval includes zero (p ~ 0.17) and the overall score remains on par with the baseline. To understand this outcome, we analyze the induced perturbations in noise space and find patterns consistent with weak or unstable signal. We recommend prompt-level paired evaluation and noise-space diagnostics as standard practice when studying initialization schemes for T2V diffusion. 

---
# ERP-RiskBench: Leakage-Safe Ensemble Learning for Financial Risk 

**Authors**: Sanjay Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2603.06671)  

**Abstract**: Financial risk detection in Enterprise Resource Planning (ERP) systems is an important but underexplored application of machine learning. Published studies in this area tend to suffer from vague dataset descriptions, leakage-prone pipelines, and evaluation practices that inflate reported performance. This paper presents a rebuilt experimental framework for ERP financial risk detection using ensemble machine learning. The risk definition is hybrid, covering both procurement compliance anomalies and transactional fraud. A composite benchmark called ERP-RiskBench is assembled from public procurement event logs, labeled fraud data, and a new synthetic ERP dataset with rule-injected risk typologies and conditional tabular GAN augmentation. Nested cross-validation with time-aware and group-aware splitting enforces leakage prevention throughout the pipeline. The primary model is a stacking ensemble of gradient boosting methods, tested alongside linear baselines, deep tabular architectures, and an interpretable glassbox alternative. Performance is measured through Matthews Correlation Coefficient, area under the precision-recall curve, and cost-sensitive decision analysis using calibrated probabilities. Across multiple dataset configurations and a structured ablation study, the stacking ensemble achieves the best detection results. Leakage-safe protocols reduce previously inflated accuracy estimates by a notable margin. SHAP-based explanations and feature stability analysis show that procurement control features, especially three-way matching discrepancies, rank as the most informative predictors. The resulting framework provides a reproducible, operationally grounded blueprint for machine learning deployment in ERP audit and governance settings. 

---
# calibfusion: Transformer-Based Differentiable Calibration for Radar-Camera Fusion Detection in Water-Surface Environments 

**Authors**: Yuting Wan, Liguo Sun, Jiuwu Hao, Pin LV  

**Link**: [PDF](https://arxiv.org/pdf/2603.06670)  

**Abstract**: Millimeter-wave (mmWave) Radar--Camera fusion improves perception under adverse illumination and weather, but its performance is sensitive to Radar--Camera extrinsic calibration: residual misalignment biases Radar-to-image projection and degrades cross-modal aggregation for downstream 2D detection. Existing calibration and auto-calibration methods are mainly developed for road and urban scenes with abundant structures and object constraints, whereas water-surface environments feature large textureless regions, sparse and intermittent targets, and wave-/specular-induced Radar clutter, which weakens explicit object-centric matching.
We propose CalibFusion, a calibration-conditioned Radar--Camera fusion detector that learns implicit extrinsic refinement end-to-end with the detection objective. CalibFusion builds a multi-frame persistence-aware Radar density representation with intensity weighting and Doppler-guided suppression of fast-varying clutter. A cross-modal transformer interaction module predicts a confidence-gated refinement of the initial extrinsics, which is integrated through a differentiable projection-and-splatting operator to generate calibration-conditioned image-plane Radar features.
Experiments on WaterScenes and FLOW show improved fusion-based 2D detection and robustness under synthetic miscalibration, supported by sensitivity analyses and qualitative Radar-to-image overlays. Results on nuScenes indicate that the refinement mechanism transfers beyond water-surface scenarios. 

---
# Hybrid Orchestration of Edge AI and Microservices via Graph-based Self-Imitation Learning 

**Authors**: Chen Yang, Jin Zheng, Yang Zhuolin, Lai Pan, Zhang Xiao, Hu Menglan, Yin Haiyan  

**Link**: [PDF](https://arxiv.org/pdf/2603.06669)  

**Abstract**: Modern edge AI applications increasingly rely on microservice architectures that integrate both AI services and conventional microservices into complex request chains with stringent latency requirements. Effectively orchestrating these heterogeneous services is crucial for ensuring low-latency performance, yet remains challenging due to their diverse resource demands and strong operational interdependencies under resource-constrained edge environments. In particular, frequent interactions between services tightly couple deployment and routing decisions, yet existing approaches optimize them in isolation, leading to fundamentally inadequate system this http URL this paper, we propose SIL-GPO, a reinforcement learning framework that optimizes hybrid orchestration for edge AI microservice systems. SIL-GPO formulates the orchestration problem as a sequential decision-making task and leverages graph attention networks to encode service topologies and routing dependencies within the agent state representation. Moreover, SIL-GPO integrates a self-imitation learning strategy into proximal policy optimization, enabling the agent to prioritize and reuse high-reward trajectories. This guides policy updates towards globally promising solutions that standard RL often fails to discover under sparse rewards and large combinatorial action spaces. We conduct extensive experiments on trace-driven edge AI workloads, demonstrating that SIL-GPO significantly reduces end-to-end service latency and enhances resource utilization compared to state-of-the-art heuristic, metaheuristic, and deep RL baselines. Our framework offers a unified and scalable solution for efficient orchestration of AI services and microservices in the edge, paving the way for low-latency, high-performance edge AI deployments. 

---
# Better Eyes, Better Thoughts: Why Vision Chain-of-Thought Fails in Medicine 

**Authors**: Yuan Wu, Zongxian Yang, Jiayu Qian, Songpan Gao, Guanxing Chen, Qiankun Li, Yu-An Huang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06665)  

**Abstract**: Large vision-language models (VLMs) often benefit from chain-of-thought (CoT) prompting in general domains, yet its efficacy in medical vision-language tasks remains underexplored. We report a counter-intuitive trend: on medical visual question answering, CoT frequently underperforms direct answering (DirA) across general-purpose and medical-specific models. We attribute this to a \emph{medical perception bottleneck}: subtle, domain-specific cues can weaken visual grounding, and CoT may compound early perceptual uncertainty rather than correct it. To probe this hypothesis, we introduce two training-free, inference-time grounding interventions: (i) \emph{perception anchoring} via region-of-interest cues and (ii) \emph{description grounding} via high-quality textual guidance. Across multiple benchmarks and model families, these interventions improve accuracy, mitigate CoT degradation, and in several settings reverse the CoT--DirA inversion. Our findings suggest that reliable clinical VLMs require robust visual grounding and cross-modal alignment, beyond extending text-driven reasoning chains. Code is available \href{this https URL}{here}. 

---
# Accelerating Video Generation Inference with Sequential-Parallel 3D Positional Encoding Using a Global Time Index 

**Authors**: Chao Yuan, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.06664)  

**Abstract**: Diffusion Transformer (DiT)-based video generation models inherently suffer from bottlenecks in long video synthesis and real-time inference, which can be attributed to the use of full spatiotemporal attention. Specifically, this mechanism leads to explosive O(N^2) memory consumption and high first-frame latency. To address these issues, we implement system-level inference optimizations for a causal autoregressive video generation pipeline. We adapt the Self-Forcing causal autoregressive framework to sequence parallel inference and implement a sequence-parallel variant of the causal rotary position embedding which we refer to as Causal-RoPE SP. This adaptation enables localized computation and reduces cross-rank communication in sequence parallel execution. In addition, computation and communication pipelines are optimized through operator fusion and RoPE precomputation. Experiments conducted on an eight GPU A800 cluster show that the optimized system achieves comparable generation quality, sub-second first-frame latency, and near real-time inference speed. For generating five second 480P videos, a 1.58x speedup is achieved, thereby providing effective support for real-time interactive applications. 

---
# Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting 

**Authors**: Giacomo Frisoni, Lorenzo Molfetta, Mattia Buzzoni, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2603.06663)  

**Abstract**: Recent advances in training-free visual prompting, such as Set-of-Mark, have emerged as a promising direction for enhancing the grounding capabilities of multimodal language models (MLMs). These techniques operate by partitioning the input image into object regions and annotating them with marks, predominantly boxes with numeric identifiers, before feeding the augmented image to the MLM. However, these approaches treat marked objects as isolated entities, failing to capture the relationships between them. On these premises, we propose Graph-of-Mark (GoM), the first pixel-level visual prompting technique that overlays scene graphs onto the input image for spatial reasoning tasks. We evaluate GoM across 3 open-source MLMs and 4 different datasets, conducting extensive ablations on drawn components and investigating the impact of auxiliary graph descriptions in the text prompt. Our results demonstrate that GoM consistently improves the zero-shot capability of MLMs in interpreting object positions and relative directions, improving base accuracy in visual question answering and localization up to 11 percentage points. 

---
# Science Literacy: Generative AI as Enabler of Coherence in the Teaching, Learning, and Assessment of Scientific Knowledge and Reasoning 

**Authors**: Xiaoming Zhai, James W. Pellegrino, Matias Rojas, Jongchan Park, Matthew Nyaaba, Clayton Cohn, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2603.06659)  

**Abstract**: This chapter examines the potential of generative AI in enhancing science literacy across the K-16+ grade span, including its benefits as well as the conceptual and practical challenges that doing so presents. It begins with a discussion of what defines science literacy in the era of AI, including how AI has changed science and the demand for future citizens to be scientifically literate when AI is applied in their careers and lives. The chapter further discusses why science literacy presents such a challenge in K-16+ educational settings. It then develops an argument for the type of architecture needed for AI to assist in solving the problem by bringing coherence to the teaching, learning, and assessment of science knowledge and reasoning. Components of this architecture are illustrated with respect to the AI tools and capabilities needed for design and implementation. The chapter concludes with a consideration of what has been learned regarding both science literacy and AI, as well as what remains to be learned, including the research and development (R&D) needed, and the generalizability of this science literacy case to other disciplinary learning and knowledge domains. 

---
# GameVerse: Can Vision-Language Models Learn from Video-based Reflection? 

**Authors**: Kuan Zhang, Dongchen Liu, Qiyue Zhao, Jinkun Hou, Xinran Zhang, Qinlei Xie, Miao Liu, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.06656)  

**Abstract**: Human gameplay is a visually grounded interaction loop in which players act, reflect on failures, and watch tutorials to refine strategies. Can Vision-Language Models (VLMs) also learn from video-based reflection? We present GameVerse, a comprehensive video game benchmark that enables a reflective visual interaction loop. Moving beyond traditional fire-and-forget evaluations, it uses a novel reflect-and-retry paradigm to assess how VLMs internalize visual experience and improve policies. To facilitate systematic and scalable evaluation, we also introduce a cognitive hierarchical taxonomy spanning 15 globally popular games, dual action space for both semantic and GUI control, and milestone evaluation using advanced VLMs to quantify progress. Our experiments show that VLMs benefit from video-based reflection in varied settings, and perform best by combining failure trajectories and expert tutorials-a training-free analogue to reinforcement learning (RL) plus supervised fine-tuning (SFT). 

---
# A Parameter-efficient Convolutional Approach for Weed Detection in Multispectral Aerial Imagery 

**Authors**: Leo Thomas Ramos, Angel D. Sappa  

**Link**: [PDF](https://arxiv.org/pdf/2603.06655)  

**Abstract**: We introduce FCBNet, an efficient model designed for weed segmentation. The architecture is based on a fully frozen ConvNeXt backbone, the proposed Feature Correction Block (FCB), which leverages efficient convolutions for feature refinement, and a lightweight decoder. FCBNet is evaluated on the WeedBananaCOD and WeedMap datasets under both RGB and multispectral modalities, showing that FCBNet outperforms models such as U-Net, DeepLabV3+, SK-U-Net, SegFormer, and WeedSense in terms of mIoU, exceeding 85%, while also achieving superior computational efficiency, requiring only 0.06 to 0.2 hours for training. Furthermore, the frozen backbone strategy reduces the number of trainable parameters by more than 90%, significantly lowering memory requirements. 

---
# PaLMR: Towards Faithful Visual Reasoning via Multimodal Process Alignment 

**Authors**: Yantao Li, Qiang Hui, Chenyang Yan, Kanzhi Cheng, Fang Zhao, Chao Tan, Huanling Gao, Jianbing Zhang, Kai Wang, Xinyu Dai, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2603.06652)  

**Abstract**: Reinforcement learning has recently improved the reasoning ability of Large Language Models and Multimodal LLMs, yet prevailing reward designs emphasise final-answer correctness and consequently tolerate process hallucinations--cases where models reach the right answer while misperceiving visual evidence. We address this process-level misalignment with PaLMR, a framework that aligns not only outcomes but also the reasoning process itself. PaLMR comprises two complementary components: a perception-aligned data layer that constructs process-aware reasoning data with structured pseudo-ground-truths and verifiable visual facts, and a process-aligned optimisation layer that constructs a hierarchical reward fusion scheme with a process-aware scoring function to encourage visually faithful chains-of-thought and improve training stability. Experiments on Qwen2.5-VL-7B show that our approach substantially reduces reasoning hallucinations and improves visual reasoning fidelity, achieving state-of-the-art results on HallusionBench while maintaining strong performance on MMMU, MathVista, and MathVerse. These findings indicate that PaLMR offers a principled and practical route to process-aligned multimodal reasoning, advancing the reliability and interpretability of MLLMs. 

---
# Geodesic Gradient Descent: A Generic and Learning-rate-free Optimizer on Objective Function-induced Manifolds 

**Authors**: Liwei Hu, Guangyao Li, Wenyong Wang, Xiaoming Zhang, Yu Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06651)  

**Abstract**: Euclidean gradient descent algorithms barely capture the geometry of objective function-induced hypersurfaces and risk driving update trajectories off the hypersurfaces. Riemannian gradient descent algorithms address these issues but fail to represent complex hypersurfaces via a single classic manifold. We propose geodesic gradient descent (GGD), a generic and learning-rate-free Riemannian gradient descent algorithm. At each iteration, GGD uses an n-dimensional sphere to approximate a local neighborhood on the objective function-induced hypersurface, adapting to arbitrarily complex geometries. A tangent vector derived from the Euclidean gradient is projected onto the sphere to form a geodesic, ensuring the update trajectory stays on the hypersurface. Parameter updates are performed using the endpoint of the geodesic. The maximum step size of the gradient in GGD is equal to a quarter of the arc length on the n-dimensional sphere, thus eliminating the need for a learning rate. Experimental results show that compared with the classic Adam algorithm, GGD achieves test MSE reductions ranging from 35.79% to 48.76% for fully connected networks on the Burgers' dataset, and cross-entropy loss reductions ranging from 3.14% to 11.59% for convolutional neural networks on the MNIST dataset. 

---
# HURRI-GAN: A Novel Approach for Hurricane Bias-Correction Beyond Gauge Stations using Generative Adversarial Networks 

**Authors**: Noujoud Nadera, Hadi Majed, Stefanos Giaremis, Rola El Osta, Clint Dawson, Carola Kaiser, Hartmut Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2603.06649)  

**Abstract**: The coastal regions of the eastern and southern United States are impacted by severe storm events, leading to significant loss of life and properties. Accurately forecasting storm surge and wind impacts from hurricanes is essential for mitigating some of the impacts, e.g., timely preparation of evacuations and other countermeasures. Physical simulation models like the ADCIRC hydrodynamics model, which run on high-performance computing resources, are sophisticated tools that produce increasingly accurate forecasts as the resolution of the computational meshes improves. However, a major drawback of these models is the significant time required to generate results at very high resolutions, which may not meet the near real-time demands of emergency responders. The presented work introduces HURRI-GAN, a novel AI-driven approach that augments the results produced by physical simulation models using time series generative adversarial networks (TimeGAN) to compensate for systemic errors of the physical model, thus reducing the necessary mesh size and runtime without loss in forecasting accuracy. We present first results in extrapolating model bias corrections for the spatial regions beyond the positions of the water level gauge stations. The presented results show that our methodology can accurately generate bias corrections at target locations spatially beyond gauge stations locations. The model's performance, as indicated by low root mean squared error (RMSE) values, highlights its capability to generate accurate extrapolated data. Applying the corrections generated by HURRI-GAN on the ADCIRC modeled water levels resulted in improving the overall prediction on the majority of the testing gauge stations. 

---
# ObjChangeVR: Object State Change Reasoning from Continuous Egocentric Views in VR Environments 

**Authors**: Shiyi Ding, Shaoen Wu, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.06648)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) offer a promising approach for natural language-based scene change queries in virtual reality (VR). Prior work on applying MLLMs for object state understanding has focused on egocentric videos that capture the camera wearer's interactions with objects. However, object state changes may occur in the background without direct user interaction, lacking explicit motion cues and making them difficult to detect. Moreover, no benchmark exists for evaluating this challenging scenario. To address these challenges, we introduce ObjChangeVR-Dataset, specifically for benchmarking the question-answering task of object state change. We also propose ObjChangeVR, a framework that combines viewpoint-aware and temporal-based retrieval to identify relevant frames, along with cross-view reasoning that reconciles inconsistent evidence from multiple viewpoints. Extensive experiments demonstrate that ObjChangeVR significantly outperforms baseline approaches across multiple MLLMs. 

---
# Performance Comparison of IBN orchestration using LLM and SLMs 

**Authors**: Wai Lwin Phone, Brahim El Boudani, Tasos Dagiuklas, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2603.06647)  

**Abstract**: The evolution of both 5G and 6G networks is driving the advancement of fully autonomous network management, placing Intent-Based Networking at the centre of this transformation. This paper introduces a novel framework for 5G and 6G IBN orchestration that leverages a stateful, hierarchical multi-agent architecture to achieve full automation using both SLMs and LLMs. Both models have been evaluated for translation accuracy using metrics such as BLEU, METEOR, and ROUGE-L, as well as computational complexity. Experimental results show that both models exhibit similar accuracy. However, result shows that SLMs can improve the overall completion speed of the IBN lifecycle by 20%. 

---
# Trust Aware Federated Learning for Secure Bone Healing Stage Interpretation in e-Health 

**Authors**: Paul Shepherd, Tasos Dagiuklas, Bugra Alkan, Joaquim Bastos, Jonathan Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2603.06646)  

**Abstract**: This paper presents a trust aware federated learning (FL) framework for interpreting bone healing stages using spectral features derived from frequency response data. The primary objective is to address the challenge posed by either unreliable or adversarial participants in distributed medical sensing environments. The framework employs a multi-layer perceptron model trained across simulated clients using the Flower FL framework. The proposed approach integrates an Adaptive Trust Score Scaling and Filtering (ATSSSF) mechanism with exponential moving average (EMA) smoothing to assess, validate and filter client this http URL trust score smoothing strategies have been investigated, one with a fixed factor and another that adapts according to trust score variability. Clients with low trust are excluded from aggregation and readmitted once their reliability improves, ensuring model integrity while maintaining inclusivity. Standard classification metrics have been used to compare the performance of ATSSSF with the baseline Federated Averaging strategy. Experimental results demonstrate that adaptive trust management can improve both training stability and predictive performance by mitigating the negative effects of compromised clients while retaining robust detection capabilities. The work establishes the feasibility for adaptive trust mechanisms in federated medical sensing and identifies extension to clinical cross silo aggregation as a future research direction. 

---
# SR-TTT: Surprisal-Aware Residual Test-Time Training 

**Authors**: Swamynathan V P  

**Link**: [PDF](https://arxiv.org/pdf/2603.06642)  

**Abstract**: Test-Time Training (TTT) language models achieve theoretically infinite context windows with an O(1) memory footprint by replacing the standard exact-attention KV-cache with hidden state ``fast weights'' W_fast updated via self-supervised learning during inference. However, pure TTT architectures suffer catastrophic failures on exact-recall tasks (e.g., Needle-in-a-Haystack). Because the fast weights aggressively compress the context into an information bottleneck, highly surprising or unique tokens are rapidly overwritten and forgotten by subsequent token gradient updates. We introduce SR-TTT (Surprisal-Aware Residual Test-Time Training), which resolves this recall failure by augmenting the TTT backbone with a loss-gated sparse memory mechanism. By dynamically routing only incompressible, highly surprising tokens to a traditional exact-attention Residual Cache, SR-TTT preserves O(1) memory for low-entropy background context while utilizing exact attention exclusively for critical needles. Our complete implementation, training scripts, and pre-trained weights are open-source and available at: this https URL. 

---
# RECAP: Local Hebbian Prototype Learning as a Self-Organizing Readout for Reservoir Dynamics 

**Authors**: Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06639)  

**Abstract**: Robust perception in brains is often attributed to high-dimensional population activity together with local plasticity mechanisms that reinforce recurring structure. In contrast, most modern image recognition systems are trained by error backpropagation and end-to-end gradient optimization, which are not naturally aligned with local computation and local plasticity. We introduce RECAP (Reservoir Computing with Hebbian Co-Activation Prototypes), a bio-inspired learning strategy for robust image classification that couples untrained reservoir dynamics with a self-organizing Hebbian prototype readout. RECAP discretizes time-averaged reservoir responses into activation levels, constructs a co-activation mask over reservoir unit pairs, and incrementally updates class-wise prototype matrices via a Hebbian-like potentiation-decay rule. Inference is performed by overlap-based prototype matching. The method avoids error backpropagation and is naturally compatible with online prototype updates. We illustrate the resulting robustness behavior on MNIST-C, where RECAP remains robust under diverse corruptions without exposure to corrupted training samples. 

---
# HEARTS: Benchmarking LLM Reasoning on Health Time Series 

**Authors**: Sirui Li, Shuhan Xiao, Mihir Joshi, Ahmed Metwally, Daniel McDuff, Wei Wang, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06638)  

**Abstract**: The rise of large language models (LLMs) has shifted time series analysis from narrow analytics to general-purpose reasoning. Yet, existing benchmarks cover only a small set of health time series modalities and tasks, failing to reflect the diverse domains and extensive temporal dependencies inherent in real-world physiological modeling. To bridge these gaps, we introduce HEARTS (Health Reasoning over Time Series), a unified benchmark for evaluating hierarchical reasoning capabilities of LLMs over general health time series. HEARTS integrates 16 real-world datasets across 12 health domains and 20 signal modalities, and defines a comprehensive taxonomy of 110 tasks grouped into four core capabilities: Perception, Inference, Generation, and Deduction. Evaluating 14 state-of-the-art LLMs on more than 20K test samples reveals intriguing findings. First, LLMs substantially underperform specialized models, and their performance is only weakly related to general reasoning scores. Moreover, LLMs often rely on simple heuristics and struggle with multi-step temporal reasoning. Finally, performance declines with increasing temporal complexity, with similar failure modes within model families, indicating that scaling alone is insufficient. By making these gaps measurable, HEARTS provides a standardized testbed and living benchmark for developing next-generation LLM agents capable of reasoning over diverse health signals. 

---
# SmartBench: Evaluating LLMs in Smart Homes with Anomalous Device States and Behavioral Contexts 

**Authors**: Qingsong Zou, Zhi Yan, Zhiyao Xu, Kuofeng Gao, Jingyu Xiao, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06636)  

**Abstract**: Due to the strong context-awareness capabilities demonstrated by large language models (LLMs), recent research has begun exploring their integration into smart home assistants to help users manage and adjust their living environments. While LLMs have been shown to effectively understand user needs and provide appropriate responses, most existing studies primarily focus on interpreting and executing user behaviors or instructions.
However, a critical function of smart home assistants is the ability to detect when the home environment is in an anomalous state. This involves two key requirements: the LLM must accurately determine whether an anomalous condition is present, and provide either a clear explanation or actionable suggestions.
To enhance the anomaly detection capabilities of next-generation LLM-based smart home assistants, we introduce SmartBench, which is the first smart home dataset designed for LLMs, containing both normal and anomalous device states as well as normal and anomalous device state transition contexts. We evaluate 13 mainstream LLMs on this benchmark. The experimental results show that most state-of-the-art models cannot achieve good anomaly detection performance. For example, Claude-Sonnet-4.5 achieves only 66.1% detection accuracy on context-independent anomaly categories, and performs even worse on context-dependent anomalies, with an accuracy of only 57.8%. More experimental results suggest that next-generation LLM-based smart home assistants are still far from being able to effectively detect and handle anomalous conditions in the smart home environment. Our dataset is publicly available at this https URL. 

---
# Photons = Tokens: The Physics of AI and the Economics of Knowledge 

**Authors**: Alec Litowitz, Nick Polson, Vadim Sokolov  

**Link**: [PDF](https://arxiv.org/pdf/2603.06630)  

**Abstract**: Debates about artificial intelligence capabilities and risks are often conducted without quantitative grounding. This paper applies the methodology of MacKay (2009) -- who reframed energy policy as arithmetic -- to the economy of AI computation. We define the token, the elementary unit of large language model input and output, as a physical quantity with measurable thermodynamic cost. Using Landauer's principle, Shannon's channel capacity, and current infrastructure data, we construct a supply-and-demand balance sheet for global token production. We then derive a finite question budget: the number of meaningful queries humanity can direct at AI systems under physical, information-theoretic, and economic constraints. We apply Coase's theory of the firm and the durable-goods monopoly problem to the AI value chain -- from photon to atom to chip to power to token to question -- to identify where economic value concentrates and where regulatory intervention is warranted. We argue that the expansion of the token budget does not resolve a deeper constraint: under structural uncertainty, the decisive variable is not how many questions can be answered but which questions are worth asking -- a problem of agency and direction that computation alone cannot solve. We connect limits of measurement in the token economy to a structural parallel between Goodhart's law and the Heisenberg uncertainty principle, and to Arrow's impossibility result for efficient information pricing. The framework yields order-of-magnitude estimates that discipline policy discussion: at current efficiency, the projected 2028 US AI energy allocation of 326~TWh could support roughly $6.5 \times 10^{17}$ tokens per year, or 225,000 tokens per person per day -- more than three orders of magnitude above estimated mid-2024 utilization. 

---
# Grouter: Decoupling Routing from Representation for Accelerated MoE Training 

**Authors**: Yuqi Xu, Rizhen Hu, Zihan Liu, Mou Sun, Kun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2603.06626)  

**Abstract**: Traditional Mixture-of-Experts (MoE) training typically proceeds without any structural priors, effectively requiring the model to simultaneously train expert weights while searching for an optimal routing policy within a vast combinatorial space. This entanglement often leads to sluggish convergence and training instabilities. This paper introduces Grouter, a preemptive routing method that by distilling high-quality structures from fully-trained MoE models and serving as a fixed router for target models. By decoupling structural optimization from weight updates, Grouter significantly accelerates both the speed and quality of model convergence. To ensure the framework's versatility, we also introduce expert folding to adapt Grouter across varying model configurations and expert tuning to rebalance workloads across different data distributions. Furthermore, by leveraging the structural priors provided by preemptive routing, we can implement targeted optimizations to further enhance training throughput. Experiments demonstrate that Grouter achieves superior performance and efficiency which boosts pre-training data utilization by 4.28x and achieves up to 33.5% throughput acceleration, establishing preemptive routing as a fundamental paradigm for scalable MoE training. 

---
# Pavement Missing Condition Data Imputation through Collective Learning-Based Graph Neural Networks 

**Authors**: Ke Yu, Lu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.06625)  

**Abstract**: Pavement condition data is important in providing information regarding the current state of the road network and in determining the needs of maintenance and rehabilitation treatments. However, the condition data is often incomplete due to various reasons such as sensor errors and non-periodic inspection schedules. Missing data, especially data missing systematically, presents loss of information, reduces statistical power, and introduces biased assessment. Existing methods in dealing with missing data usually discard entire data points with missing values or impute through data correlation. In this paper, we used a collective learning-based Graph Convolutional Networks, which integrates both features of adjacent sections and dependencies between observed section conditions to learn missing condition values. Unlike other variants of graph neural networks, the proposed approach is able to capture dependent relationship between the conditions of adjacent pavement sections. In the case study, pavement condition data collected from Texas Department of Transportation Austin District were used. Experiments show that the proposed model was able to produce promising results in imputing the missing data. 

---
# Exploration Space Theory: Formal Foundations for Prerequisite-Aware Location-Based Recommendation 

**Authors**: Madjid Sadallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.06624)  

**Abstract**: Location-based recommender systems have achieved considerable sophistication, yet none provides a formal, lattice-theoretic representation of prerequisite dependencies among points of interest -- the semantic reality that meaningfully experiencing certain locations presupposes contextual knowledge gained from others -- nor the structural guarantees that such a representation entails. We introduce Exploration Space Theory (EST), a formal framework that transposes Knowledge Space Theory into location-based recommendation. We prove that the valid user exploration states -- the order ideals of a surmise partial order on points of interest -- form a finite distributive lattice and a well-graded learning space; Birkhoff's representation theorem, combined with the structural isomorphism between lattices of order ideals and concept lattices, connects the exploration space canonically to Formal Concept Analysis. These structural results yield four direct consequences: linear-time fringe computation, a validity certificate guaranteeing that every fringe-guided recommendation is a structurally sound next step, sub-path optimality for dynamic-programming path generation, and provably existing structural explanations for every recommendation. Building on these foundations, we specify the Exploration Space Recommender System (ESRS) -- a memoized dynamic program over the exploration lattice, a Bayesian state estimator with beam approximation and EM parameter learning, an online feedback loop enforcing the downward-closure invariant, an incremental surmise-relation inference pipeline, and three cold-start strategies, the structural one being the only approach in the literature to provide a formal validity guarantee conditional on the correctness of the inferred surmise relation. All results are established through proof and illustrated on a fully traced five-POI numerical example. 

---
# From ARIMA to Attention: Power Load Forecasting Using Temporal Deep Learning 

**Authors**: Suhasnadh Reddy Veluru, Sai Teja Erukude, Viswa Chaitanya Marella  

**Link**: [PDF](https://arxiv.org/pdf/2603.06622)  

**Abstract**: Accurate short-term power load forecasting is important to effectively manage, optimize, and ensure the robustness of modern power systems. This paper performs an empirical evaluation of a traditional statistical model and deep learning approaches for predicting short-term energy load. Four models, namely ARIMA, LSTM, BiLSTM, and Transformer, were leveraged on the PJM Hourly Energy Consumption data. The data processing involved interpolation, normalization, and a sliding-window sequence method. Each model's forecasting performance was evaluated for the 24-hour horizon using MAE, RMSE, and MAPE. Of the models tested, the Transformer model, which relies on self-attention algorithms, produced the best results with 3.8 percent of MAPE, with performance above any model in both accuracy and robustness. These findings underscore the growing potential of attention-based architectures in accurately capturing complex temporal patterns in power consumption data. 

---
# GraphSkill: Documentation-Guided Hierarchical Retrieval-Augmented Coding for Complex Graph Reasoning 

**Authors**: Fali Wang, Chenglin Weng, Xianren Zhang, Siyuan Hong, Hui Liu, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06620)  

**Abstract**: The growing demand for automated graph algorithm reasoning has attracted increasing attention in the large language model (LLM) community. Recent LLM-based graph reasoning methods typically decouple task descriptions from graph data, generate executable code augmented by retrieval from technical documentation, and refine the code through debugging. However, we identify two key limitations in existing approaches: (i) they treat technical documentation as flat text collections and ignore its hierarchical structure, leading to noisy retrieval that degrades code generation quality; and (ii) their debugging mechanisms focus primarily on runtime errors, yet ignore more critical logical errors. To address them, we propose {\method}, an \textit{agentic hierarchical retrieval-augmented coding framework} that exploits the document hierarchy through top-down traversal and early pruning, together with a \textit{self-debugging coding agent} that iteratively refines code using automatically generated small-scale test cases. To enable comprehensive evaluation of complex graph reasoning, we introduce a new dataset, {\dataset}, covering small-scale, large-scale, and composite graph reasoning tasks. Extensive experiments demonstrate that our method achieves higher task accuracy and lower inference cost compared to baselines\footnote{The code is available at \href{this https URL}{\textcolor{blue}{this https URL}}.}. 

---
# Not all tokens are needed(NAT): token efficient reinforcement learning 

**Authors**: Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06619)  

**Abstract**: Reinforcement learning (RL) has become a key driver of progress in large language models, but scaling RL to long chain-of-thought (CoT) trajectories is increasingly constrained by backpropagation over every generated token. Even with optimized rollout engines, full-token updates can consume a large fraction of total training cost, turning token length into a hidden tax on RL. We introduce Not All Tokens Are Needed (NAT), a unified framework that makes the token budget a first-class optimization primitive. NAT updates the policy using only a selected subset of generated tokens while preserving the learning signal of full-sequence RL. The core idea is an unbiased partial-token policy-gradient estimator via Horvitz-Thompson reweighting, which ensures statistically correct gradients despite subsampling. We instantiate NAT with two simple, plug-and-play token selection schemes: Uniform Random Sampling (URS) and Random Prefix Cutting (RPC), both of which reduce forward and backward compute and memory without modifying the reward computation or rollout pipeline. Across mathematical reasoning benchmarks, NAT matches full-token GRPO performance while using as few as 50% of tokens, providing an efficient and orthogonal pathway to scaling RL beyond the limits imposed by long trajectories. In our experiments, RPC saves 18% peak GPU memory and 29% forward and backward RL training time for Qwen3-8B. 

---
# Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks 

**Authors**: Alana Deng, Sugitha Janarthanan, Yan Sun, Zihao Jing, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06618)  

**Abstract**: Multiplex Biological Networks (MBNs), which represent multiple interaction types between entities, are crucial for understanding complex biological systems. Yet, existing methods often inadequately model multiplexity, struggle to integrate structural and sequence information, and face difficulties in zero-shot prediction for unseen entities with no prior neighbourhood information. To address these limitations, we propose a novel framework for zero-shot interaction prediction in MBNs by leveraging context-aware representation learning and knowledge distillation. Our approach leverages domain-specific foundation models to generate enriched embeddings, introduces a topology-aware graph tokenizer to capture multiplexity and higher-order connectivity, and employs contrastive learning to align embeddings across modalities. A teacher-student distillation strategy further enables robust zero-shot generalization. Experimental results demonstrate that our framework outperforms state-of-the-art methods in interaction prediction for MBNs, providing a powerful tool for exploring various biological interactions and advancing personalized therapeutics. 

---
# Evo: Autoregressive-Diffusion Large Language Models with Evolving Balance 

**Authors**: Junde Wu, Minhao Hu, Jiayuan Zhu, Yuyuan Liu, Tianyi Zhang, Kang Li, Jingkun Chen, Jiazhen Pan, Min Xu, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.06617)  

**Abstract**: We introduce \textbf{Evo}, a duality latent trajectory model that bridges autoregressive (AR) and diffusion-based language generation within a continuous evolutionary generative framework. Rather than treating AR decoding and diffusion generation as separate paradigms, Evo reconceptualizes text generation as a latent flow: each token is associated with a vector-valued embedding that evolves over a progression variable $t_i \in [0, 1]$, indicating its semantic maturity. Low $t_i$ values correspond to confident AR-like refinement, while high values invoke diffusion-style planning, allowing the model to adaptively balance AR and diffusion based on uncertainty. Theoretically, we show that both AR and diffusion models emerge as discretizations of a shared probability flow, and we derive Evo's training objective from a unified variational ELBO. The model is implemented as a time-conditioned Transformer governed by a shared vector field, trained end-to-end to jointly infer latent codes and their progression times. During decoding, Evo performs efficient, semantics-aware refinement, achieving high-quality outputs without sacrificing speed. Empirically, Evo 8B achieves state-of-the-art or highly competitive results on 15 diverse benchmarks, including reasoning (GSM8K, ARC-C), code generation (HumanEval, MBPP), and general language understanding, while maintaining fast inference speed. Our results demonstrate that Evo delivers a new paradigm for LLM design with strong generation quality, robust symbolic reasoning, and decoding efficiency. 

---
# RACER: Risk-Aware Calibrated Efficient Routing for Large Language Models 

**Authors**: Sai Hao, Hao Zeng, Hongxin Wei, Bingyi Jing  

**Link**: [PDF](https://arxiv.org/pdf/2603.06616)  

**Abstract**: Efficiently routing queries to the optimal large language model (LLM) is crucial for optimizing the cost-performance trade-off in multi-model systems. However, most existing routers rely on single-model selection, making them susceptible to misrouting. In this work, we formulate LLM routing as the $\alpha$-VOR problem to minimize expected set size while controlling the misrouting risk, and propose a novel method -- RACER, extending base routers to output model sets that can be subsequently aggregated for improved output. In particular, RACER constructs nested model sets via augmented scoring and utilizes finite-sample concentration bounds to calibrate a threshold that allows for both variable set sizes and abstention. We theoretically prove that RACER achieves rigorous distribution-free risk control on unseen test data in a post-hoc and model-agnostic manner. Extensive experiments verify our theoretical guarantees and demonstrate that RACER consistently enhances downstream accuracy across a wide range of benchmarks. 

---
# Annealed Co-Generation: Disentangling Variables via Progressive Pairwise Modeling 

**Authors**: Hantao Zhang, Jieke Wu, Mingda Xu, Xiao Hu, Yingxuan You, Pascal Fua  

**Link**: [PDF](https://arxiv.org/pdf/2603.06615)  

**Abstract**: For multivariate co-generation in scientific applications, we advocate pairwise block rather than joint modeling of all variables. This design mitigates the computational burden and data imbalance. To this end, we propose an Annealed Co-Generation (ACG) framework that replaces high-dimensional diffusion modeling with a low-dimensional diffusion model, which enables multivariate co-generation by composing pairwise variable generations. We first train an unconditional diffusion model over causal variables that are disentangled into pairs. At inference time, we recover the joint distribution by coupling these pairwise models through shared common variables, enabling coherent multivariate generation without any additional training. By employing a three-stage annealing process-Consensus, Heating, and Cooling-our method enforces consistency across shared common variables and progressively constrains each pairwise data distribution to lie on a learnable manifold, while maintaining high likelihood within each pair. We demonstrate the framework's flexibility and efficacy on two distinct scientific tasks: flow-field completion and antibody generation. All datasets and code will be made publicly available upon publication. 

---
# OptiRoulette Optimizer: A New Stochastic Meta-Optimizer for up to 5.3x Faster Convergence 

**Authors**: Stamatis Mastromichalakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.06613)  

**Abstract**: This paper presents OptiRoulette, a stochastic meta-optimizer that selects update rules during training instead of fixing a single optimizer. The method combines warmup optimizer locking, random sampling from an active optimizer pool, compatibility-aware learning-rate scaling during optimizer transitions, and failure-aware pool replacement. OptiRoulette is implemented as a drop-in, this http URL-compatible component and packaged for pip installation. We report completed 10-seed results on five image-classification suites: CIFAR-100, CIFAR-100-C, SVHN, Tiny ImageNet, and Caltech-256. Against a single-optimizer AdamW baseline, OptiRoulette improves mean test accuracy from 0.6734 to 0.7656 on CIFAR-100 (+9.22 percentage points), 0.2904 to 0.3355 on CIFAR-100-C (+4.52), 0.9667 to 0.9756 on SVHN (+0.89), 0.5669 to 0.6642 on Tiny ImageNet (+9.73), and 0.5946 to 0.6920 on Caltech-256 (+9.74). Its main advantage is convergence reliability at higher targets: it reaches CIFAR-100/CIFAR-100-C 0.75, SVHN 0.96, Tiny ImageNet 0.65, and Caltech-256 0.62 validation accuracy in 10/10 runs, while the AdamW baseline reaches none of these targets within budget. On shared targets, OptiRoulette also reduces time-to-target (e.g., Caltech-256 at 0.59: 25.7 vs 77.0 epochs). Paired-seed deltas are positive on all datasets; CIFAR-100-C test ROC-AUC is the only metric not statistically significant in the current 10-seed study. 

---
# Consensus is Not Verification: Why Crowd Wisdom Strategies Fail for LLM Truthfulness 

**Authors**: Yegor Denisov-Blanch, Joshua Kazdan, Jessica Chudnovsky, Rylan Schaeffer, Sheng Guan, Soji Adeshina, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2603.06612)  

**Abstract**: Pass@k and other methods of scaling inference compute can improve language model performance in domains with external verifiers, including mathematics and code, where incorrect candidates can be filtered reliably. This raises a natural question: can we similarly scale compute to elicit gains in truthfulness for domains without convenient verification? We show that across five benchmarks and models, surprisingly, it cannot. Even at 25x the inference cost of naive sampling, polling-style aggregation yields no consistent accuracy gains over single-sample baselines and often amplifies shared misconceptions. We find that under uncertainty, models are better at predicting what other models will say within model ensembles than at identifying what is true, revealing a separation between social prediction and truth verification. Across models and benchmarks, aggregation fails to provide a robust truth signal because language model errors are strongly correlated. The source of correlation goes beyond any individual benchmark: we show that even when conditioned on out of distribution random strings and asked to produce pseudo-random outputs, different models produce correlated outputs. Confidence-based weighting provides no benefit because self-reported confidence fails to reliably distinguish correct from incorrect answers. These results delineate a boundary for inference-time scaling: in verified domains, additional samples provide more candidates for a verifier to filter; in unverified domains, additional samples merely reinforce shared misconceptions. 

---
# Multi-Agent DRL for V2X Resource Allocation: Disentangling Challenges and Benchmarking Solutions 

**Authors**: Siyuan Wang, Lei Lei, Pranav Maheshwari, Sam Bellefeuille, Kan Zheng, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2603.06607)  

**Abstract**: Multi-agent deep reinforcement learning (DRL) has emerged as a promising approach for radio resource allocation (RRA) in cellular vehicle-to-everything (C-V2X) networks. However, the multifaceted challenges inherent to multi-agent reinforcement learning (MARL) - including non-stationarity, coordination difficulty, large action spaces, partial observability, and limited robustness and generalization - are often intertwined, making it difficult to understand their individual impact on performance in vehicular environments. Moreover, existing studies typically rely on different baseline MARL algorithms, and a systematic comparison of their capabilities in addressing specific challenges in C-V2X RRA remains lacking. In this paper, we bridge this gap by formulating C-V2X RRA as a sequence of multi-agent interference games with progressively increasing complexity, each designed to isolate a key MARL challenge. Based on these formulations, we construct a suite of learning tasks that enable controlled evaluation of performance degradation attributable to each challenge. We further develop large-scale, diverse training and testing datasets using SUMO-generated highway traces to capture a wide range of vehicular topologies and corresponding interference patterns. Through extensive benchmarking of representative MARL algorithms, we identify policy robustness and generalization across diverse vehicular topologies as the dominant challenge in C-V2X RRA. We further show that, on the most challenging task, the best-performing actor-critic method outperforms the value-based approach by 42%. By emphasizing the need for zero-shot policy transfer to both seen and unseen topologies at runtime, and by open-sourcing the code, datasets, and interference-game benchmark suite, this work provides a systematic and reproducible foundation for evaluating and advancing MARL algorithms in vehicular networks. 

---
# Scale Dependent Data Duplication 

**Authors**: Joshua Kazdan, Noam Levi, Rylan Schaeffer, Jessica Chudnovsky, Abhay Puri, Bo He, Mehmet Donmez, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2603.06603)  

**Abstract**: Data duplication during pretraining can degrade generalization and lead to memorization, motivating aggressive deduplication pipelines. However, at web scale, it is unclear what constitutes a ``duplicate'': beyond surface-form matches, semantically equivalent documents (e.g. translations) may induce redundant training signals once models become sufficiently capable. Practically, this means that semantic duplicates operate increasingly like exact duplicates during training. We present evidence that duplication is scale-dependent in two ways. First, as model capability increases, cross-entropy loss gradients for semantically equivalent documents become more aligned. Smaller models, by contrast, produce gradients that reflect surface similarity (e.g., shared tokens) rather than semantic similarity. Second, we embedded all 192 million FineWeb-Edu-Dedup documents using EmbeddingGemma-300m. For moderate corpus sizes, the cosine similarity between nearest-neighbors follows an isotropic power law baseline. However, as corpus size grows to hundreds of billions of tokens, the nearest-neighbor similarities deviate sharply, indicating accelerated semantic collisions. Finally, controlled pretraining on data sampled with replacement from pools of finite unique documents shows that limited uniqueness yields mild degradation for small models, but rapidly increasing loss penalties for larger models, breaking naive scaling extrapolation. We derive explicit scaling laws that allow practitioners to estimate deviation from expected scaling due to limited semantic uniqueness of the pretraining corpus. Our results identify and resolve an unstudied source of scale-dependence, allowing for more accurate prediction at scale. 

---
# FuzzingRL: Reinforcement Fuzz-Testing for Revealing VLM Failures 

**Authors**: Jiajun Xu, Jiageng Mao, Ang Qi, Weiduo Yuan, Alexander Romanus, Helen Xia, Vitor Campagnolo Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06600)  

**Abstract**: Vision Language Models (VLMs) are prone to errors, and identifying where these errors occur is critical for ensuring the reliability and safety of AI systems. In this paper, we propose an approach that automatically generates questions designed to deliberately induce incorrect responses from VLMs, thereby revealing their vulnerabilities. The core of this approach lies in fuzz testing and reinforcement finetuning: we transform a single input query into a large set of diverse variants through vision and language fuzzing. Based on the fuzzing outcomes, the question generator is further instructed by adversarial reinforcement fine-tuning to produce increasingly challenging queries that trigger model failures. With this approach, we can consistently drive down a target VLM's answer accuracy -- for example, the accuracy of Qwen2.5-VL-32B on our generated questions drops from 86.58\% to 65.53\% in four RL iterations. Moreover, a fuzzing policy trained against a single target VLM transfers to multiple other VLMs, producing challenging queries that degrade their performance as well. 

---
# Building the ethical AI framework of the future: from philosophy to practice 

**Authors**: Jasper Kyle Catapang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06599)  

**Abstract**: Artificial intelligence pipelines -- spanning data collection, model training, deployment, and post-deployment monitoring -- concentrate ethical risks that intensify with multimodal and agentic systems. Existing governance instruments, including the EU AI Act, the IEEE 7000 series, and the NIST AI Risk Management Framework, provide high-level guidance but often lack enforceable, end-to-end operational controls. This paper presents an ethics-by-design control architecture that embeds consequentialist, deontological, and virtue-ethical reasoning into stage-specific enforcement mechanisms across the AI lifecycle. The framework implements a triple-gate structure at each lifecycle stage: Metric gates (quantitative performance and safety thresholds), Governance gates (legal, rights, and procedural compliance), and Eco gates (carbon and water budgets and sustainability constraints). It specifies measurable trigger conditions, escalation paths, audit artefacts, and mappings to EU AI Act obligations and NIST RMF functions, enabling integration with existing MLOps and CI/CD pipelines. Illustrative examples from large language model pipelines demonstrate how gate-based controls can surface and constrain technical, social, and environmental risks prior to release and during runtime. The framework is accompanied by a preregistered evaluation protocol that defines ex ante success criteria and assessment procedures, enabling falsifiable evaluation of gate effectiveness. By translating normative commitments into enforceable and testable controls, the framework provides a practical basis for operational AI governance across organizational contexts, jurisdictions, and deployment scales. 

---
# Distributionally Robust Geometric Joint Chance-Constrained Optimization: Neurodynamic Approaches 

**Authors**: Ange Valli, Siham Tassouli, Abdel Lisser  

**Link**: [PDF](https://arxiv.org/pdf/2603.06597)  

**Abstract**: This paper proposes a two-time scale neurodynamic duplex approach to solve distributionally robust geometric joint chance-constrained optimization problems. The probability distributions of the row vectors are not known in advance and belong to a certain distributional uncertainty set. In our paper, we study three uncertainty sets for the unknown distributions. The neurodynamic duplex is designed based on three projection equations. The main contribution of our work is to propose a neural network-based method to solve distributionally robust joint chance-constrained optimization problems that converges in probability to the global optimum without the use of standard state-of-the-art solving methods. We show that neural networks can be used to solve multiple instances of a problem. In the numerical experiments, we apply the proposed approach to solve a problem of shape optimisation and a telecommunication problem. 

---
# A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness 

**Authors**: Leo Schwinn, Moritz Ladenburger, Tim Beyer, Mehrnaz Mofakhami, Gauthier Gidel, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2603.06594)  

**Abstract**: Automated \enquote{LLM-as-a-Judge} frameworks have become the de facto standard for scalable evaluation across natural language processing. For instance, in safety evaluation, these judges are relied upon to evaluate harmfulness in order to benchmark the robustness of safety against adversarial attacks. However, we show that existing validation protocols fail to account for substantial distribution shifts inherent to red-teaming: diverse victim models exhibit distinct generation styles, attacks distort output patterns, and semantic ambiguity varies significantly across jailbreak scenarios. Through a comprehensive audit using 6642 human-verified labels, we reveal that the unpredictable interaction of these shifts often causes judge performance to degrade to near random chance. This stands in stark contrast to the high human agreement reported in prior work. Crucially, we find that many attacks inflate their success rates by exploiting judge insufficiencies rather than eliciting genuinely harmful content. To enable more reliable evaluation, we propose ReliableBench, a benchmark of behaviors that remain more consistently judgeable, and JudgeStressTest, a dataset designed to expose judge failures. Data available at: this https URL. 

---
# ARC-AGI-2 Technical Report 

**Authors**: Wallyson Lemes de Oliveira, Mekhron Bobokhonov, Matteo Caorsi, Aldo Podestà, Gabriele Beltramo, Luca Crosato, Matteo Bonotto, Federica Cecchetto, Hadrien Espic, Dan Titus Salajan, Stefan Taga, Luca Pana, Joe Carthy  

**Link**: [PDF](https://arxiv.org/pdf/2603.06590)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) is designed to assess generalization beyond pattern matching, requiring models to infer symbolic rules from very few examples. In this work, we present a transformer-based system that advances ARC performance by combining neural inference with structure-aware priors and online task adaptation. Our approach is built on four key ideas. First, we reformulate ARC reasoning as a sequence modeling problem using a compact task encoding with only 125 tokens, enabling efficient long-context processing with a modified LongT5 architecture. Second, we introduce a principled augmentation framework based on group symmetries, grid traversals, and automata perturbations, enforcing invariance to representation changes. Third, we apply test-time training (TTT) with lightweight LoRA adaptation, allowing the model to specialize to each unseen task by learning its transformation logic from demonstrations. Fourth, we design a symmetry-aware decoding and scoring pipeline that aggregates likelihoods across augmented task views, effectively performing ``multi-perspective reasoning'' over candidate solutions. We demonstrate that these components work synergistically: augmentations expand hypothesis space, TTT sharpens local reasoning, and symmetry-based scoring improves solution consistency. Our final system achieves a significant improvement over transformer baselines and surpasses prior neural ARC solvers, closing the gap toward human-level generalization. 

---
# Isotonic Layer: A Universal Framework for Generic Recommendation Debiasing 

**Authors**: Hailing Cheng, Yafang Yang, Hemeng Tao, Fengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06589)  

**Abstract**: Model calibration and debiasing are fundamental to the reliability and fairness of large scale recommendation systems. We introduce the Isotonic Layer, a novel, differentiable framework that integrates piecewise linear fitting directly into neural architectures. By partitioning the feature space into discrete segments and optimizing non negative slopes via a constrained dot product mechanism, we enforce a global monotonic inductive bias. This ensures model outputs remain logically consistent with critical features such as latent relevance, recency, or quality scores. We further generalize this architecture by parameterizing segment wise slopes as learnable embeddings. This enables the model to adaptively capture context specific distortions, such as position based CTR bias through specialized isotonic profiles. Our approach utilizes a dual task formulation that decouples the recommendation objective into latent relevance estimation and bias aware calibration. A major contribution of this work is the ability to perform highly granular, customized calibration for arbitrary combinations of context features, a level of control difficult to achieve with traditional non parametric methods. We also extend this to Multi Task Learning environments with dedicated embeddings for distinct objectives. Extensive empirical evaluations on real world datasets and production AB tests demonstrate that the Isotonic Layer effectively mitigates systematic bias and enhances calibration fidelity, significantly outperforming production baselines in both predictive accuracy and ranking consistency. 

---
# Right Move, Right Time: Multi-Sport Space Evaluation Platform for Ultimate Frisbee, Basketball, and Soccer 

**Authors**: Shunsuke Iwashita, Titouan Jeannot, Braden Eberhard, Jacob Miller, Rikako Kono, Calvin Yeung, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2603.06585)  

**Abstract**: We present an open, sport-agnostic platform that turns tracking into comparable spatial measures across professional Ultimate, basketball, and soccer. Coaches in all three sports ask the same question: where is the usable space, and when should an off-ball run start? Our workflow standardizes inputs, provides timing-aware spatial evaluations, and makes it possible to reuse the same analysis across sports. We illustrate the approach with Ultimate as a focused testbed and then examine transfer between basketball and soccer. Together, these results show a practical path toward consistent, comparable evaluation across various invasion sports. 

---
# Agentic SPARQL: Evaluating SPARQL-MCP-powered Intelligent Agents on the Federated KGQA Benchmark 

**Authors**: Daniel Dobriy, Frederik Bauer, Amr Azzam, Debayan Banerjee, Axel Polleres  

**Link**: [PDF](https://arxiv.org/pdf/2603.06582)  

**Abstract**: Standard protocols such as the Model Context Protocol (MCP) that allow LLMs to connect to tools have recently boosted "agentic" AI applications, which, powered by LLMs' planning capabilities, promise to solve complex tasks with the access of external tools and data sources. In this context, publicly available SPARQL endpoints offer a natural connection to combine various data sources through MCP by (a) implementing a standardised protocol and query language, (b) standardised metadata formats, and (c) the native capability to federate queries. In the present paper, we explore the potential of SPARQL-MCP-based intelligent agents to facilitate federated SPARQL querying: firstly, we discuss how to extend an existing Knowledge Graph Question Answering benchmark towards agentic federated Knowledge Graph Question Answering (FKGQA); secondly, we implement and evaluate the ability of integrating SPARQL federation with LLM agents via MCP (incl. endpoint discovery/source selection, schema exploration, and query formulation), comparing different architectural options against the extended benchmark. Our work complements and extends prior work on automated SPARQL query federation towards fruitful combinations with agentic AI. 

---
# A Robust Incomplete Multimodal Low-Rank Adaptation Approach for Emotion Recognition 

**Authors**: Xinkui Zhao, Jinsong Shu, Yangyang Wu, Guanjie Cheng, Zihe Liu, Naibo Wang, Shuiguang Deng, Zhongle Xie, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11202)  

**Abstract**: Multimodal Emotion Recognition (MER) often encounters incomplete multimodality in practical applications due to sensor failures or privacy protection requirements. While existing methods attempt to address various incomplete multimodal scenarios by balancing the training of each modality combination through additional gradients, these approaches face a critical limitation: training gradients from different modality combinations conflict with each other, ultimately degrading the performance of the final prediction model. In this paper, we propose a unimodal decoupled dynamic low-rank adaptation method based on modality combinations, named MCULoRA, which is a novel framework for the parameter-efficient training of incomplete multimodal learning models. MCULoRA consists of two key modules, modality combination aware low-rank adaptation (MCLA) and dynamic parameter fine-tuning (DPFT). The MCLA module effectively decouples the shared information from the distinct characteristics of individual modality combinations. The DPFT module adjusts the training ratio of modality combinations based on the separability of each modality's representation space, optimizing the learning efficiency across different modality combinations. Our extensive experimental evaluation in multiple benchmark datasets demonstrates that MCULoRA substantially outperforms previous incomplete multimodal learning approaches in downstream task accuracy. 

---
# Deep Learning-Based Approach for Automatic 2D and 3D MRI Segmentation of Gliomas 

**Authors**: Kiranmayee Janardhan, Christy Bobby T  

**Link**: [PDF](https://arxiv.org/pdf/2502.19760)  

**Abstract**: Brain tumor diagnosis is a challenging task for clinicians in the modern world. Among the major reasons for cancer-related death is the brain tumor. Gliomas, a category of central nervous system (CNS) tumors, encompass diverse subregions. For accurate diagnosis of brain tumors, precise segmentation of brain images and quantitative analysis are required. A fully automatic approach to glioma segmentation is required because the manual segmentation process is laborious, prone to mistakes, as well as time-consuming. Modern techniques for segmenting gliomas are based on fully convolutional neural networks (FCNs), which can either use two-dimensional (2D) or three-dimensional (3D) convolutions. Nevertheless, 3D convolutions suffer from computational costs and memory demand, while 2D convolutions cannot fully utilize the spatial insights of volumetric clinical imaging data. To obtain an optimal solution, it is vital to balance the computational efficiency of 2D convolutions along with the spatial accuracy of 3D convolutions. This balance can potentially be realized by developing an advanced model to overcome these challenges. The 2D and 3D models implemented here are based on UNET architecture, Inception, and ResNet models. The research work has been implemented on the BraTS 2018, 2019, and 2020 datasets. The best performer of all the models' evaluations metrics for proposed methodologies offer superior potential in terms of the effective segmentation of gliomas. The ResNet model has resulted in 98.91% accuracy for 3D segmentation and 99.77 for 2D segmentations. The dice scores for 2D and 3D segmentations are 0.8312 and 0.9888, respectively. This model can be applied to various other medical applications with fine-tuning, thereby aiding clinicians in brain tumor analysis and improving the diagnosis process effectively. 

---
# Subclass Classification of Gliomas Using MRI Fusion Technique 

**Authors**: Kiranmayee Janardhan, Christy Bobby Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2502.18775)  

**Abstract**: Glioma, the prevalent primary brain tumor, exhibits diverse aggressiveness levels and prognoses. Precise classification of glioma is paramount for treatment planning and predicting prognosis. This study aims to develop an algorithm to fuse the MRI images from T1, T2, T1ce, and fluid-attenuated inversion recovery (FLAIR) sequences to enhance the efficacy of glioma subclass classification as no tumor, necrotic core, peritumoral edema, and enhancing tumor. The MRI images from BraTS datasets were used in this work. The images were pre-processed using max-min normalization to ensure consistency in pixel intensity values across different images. The segmentation of the necrotic core, peritumoral edema, and enhancing tumor was performed on 2D and 3D images separately using UNET architecture. Further, the segmented regions from multimodal MRI images were fused using the weighted averaging technique. Integrating 2D and 3D segmented outputs enhances classification accuracy by capturing detailed features like tumor shape, boundaries, and intensity distribution in slices, while also providing a comprehensive view of spatial extent, shape, texture, and localization within the brain volume. The fused images were used as input to the pre-trained ResNet50 model for glioma subclass classification. The network is trained on 80% and validated on 20% of the data. The proposed method achieved a classification of accuracy of 99.25%, precision of 99.30%, recall of 99.10, F1 score of 99.19%, Intersection Over Union of 84.49%, and specificity of 99.76, which showed a significantly higher performance than existing techniques. These findings emphasize the significance of glioma segmentation and classification in aiding accurate diagnosis. 

---
