# Adaptive Multi-Token Prediction: How far should we go? 

**Authors**: Emile Anand, Mohammed Rehan Parwani, Xinyuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28619)  

**Abstract**: Multi-token prediction (MTP) can accelerate language-model decoding by drafting several tokens and verifying them in a single target-model pass. However, the best speculation horizon changes with context: longer drafts offer greater amortization when reliable, but can waste compute on suffixes discarded after a rejection. Unlike autoregressive drafting, where a stopping rule can decide token by token, MTP must fix the horizon before drafting, as all heads fire in the same forward pass. We therefore formulate horizon selection as an online control problem, using verification feedback to learn how far the draft can be trusted in each context, and introduce Adaptive Multi-Token Prediction (AMTP): a lightweight controller that requires no retraining and leaves the exact verifier unchanged. To guide horizon selection, we characterize the expected number of tokens committed at each horizon via acceptance survival, the probability that a draft prefix is fully accepted. Before each round, AMTP predicts how far the draft is likely to survive and picks the horizon with the highest expected tokens per second, learning freely from every verification pass. Theoretically, we show that any online learner's prediction error translates into throughput regret against the best horizon for each context, and in turn into a bound on wall-clock completion time. With online gradient descent, this regret is sublinear. Across five draft-target pairings built on Llama-3.1-8B-Instruct with Medusa heads, Meta's MTP models, and Qwen3-4B-Inst-2507-MTP, evaluated on AIME 2025 and GSM8K, AMTP decodes up to 4.1x faster than standard decoding and matches or outperforms the fixed window in 8 of 10 settings, with throughput gains of up to 10%. 

---
# Semantic Observability for Autonomous Systems Diagnostics 

**Authors**: Jitendra Solanki, Anand Todkar, Ziran Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.28611)  

**Abstract**: Modern distributed microservice deployments generate massive volumes of heterogeneous, multi-format observability logs across physically and logically partitioned infrastructure. Isolating root causes during cascading failures remains an arduous task for Site Reliability Engineers (SREs), bottlenecked by manual keyword filtering and disjointed temporal correlation across monitoring dashboards. We introduce Semantic Observability, an autonomous multi-agent framework that unifies disparate log streams into a dynamic, queryable Property Graph Database. By modeling physical infrastructure topologies, logical software dependencies, and chronological event sequences as attributed nodes and directed relations, it bridges the semantic gap inherent in flat, unstructured log repositories. Building on this unified substrate, we deploy a stateful multi-agent system orchestrated through Directed Cyclic Graphs (DCGs). The system autonomously navigates the topology, applying causal graph algorithms and temporal back-traversal to isolate the precise blast radius of anomalies. The pruned subgraph is then interpreted by a Large Language Model (LLM) reasoning engine to generate contextualized, code-level Root Cause Analyses (RCA) and actionable remediations. This paper establishes the foundational system architecture, formal relational ontology, and multi-agent control loops of Semantic Observability. We show conceptually how the framework eliminates ambient telemetry noise, reduces Mean Time to Resolution (MTTR), compresses LLM token consumption, and prevents structural hallucinations. We also specify an open-source validation framework on standard benchmarks (LogHub and Train-Ticket), laying the groundwork for a forthcoming companion paper on exhaustive quantitative evaluation and production benchmarking. 

---
# Value-Rich, Differently Organized: Measuring Value Expression and Organization in AI-Generated Debates 

**Authors**: Maoyuan Li, Haoyuan Li, Zhongsheng Wang, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28604)  

**Abstract**: Current evaluation of LLM-generated debate largely focuses on argumentative quality, leaving the value structure of generated arguments underexamined. We introduce a diagnostic metric framework for comparing how values are expressed and organized in human-reference and AI-generated debates. The framework separates value expression, the range of recognizable values a debate invokes, from value organization, how those values are distributed and sustained across turns. Grounded in Schwartz's Basic-10 value theory, it represents debates as multi-level value distributions over argumentative units, utterances, stances, and full debates, enabling metrics for value coverage, alignment, diffusion, and cross-turn dynamics. Across multiple generation systems, AI-generated debates are broadly value-expressive, often activating as many or more value dimensions than matched human references, but they organize these values differently. Their utterances are more locally value-diffuse, and source-blind judges distinguish value clarity from value focus. These findings show that debate evaluation should consider not only value presence, but also value organization. Code and supplementary materials are available at this https URL. 

---
# Evolving Agent Interfaces 

**Authors**: Saaket Agashe, Yuzhe Yang, Gonzalo Gonzalez-Pumariega, Bhuvan Chandra Koduru, Chih-Lun Li, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28601)  

**Abstract**: Computer-use agents are limited by the interfaces humans give them. An agent may type a spreadsheet cell by cell when a single line of code would do, or print all two thousand lines of a file through a shell to find one function. Both the interfaces and the knowledge of when to use each are built by hand, which cannot keep pace with new applications, workflows, and models. We argue that interface design should itself be an agent capability. We introduce Agent S4, an autonomous computer-use scientist that builds Evolving Agent Interfaces from task experience. A meta-agent examines task agents' trajectories on synthetic training tasks, without success labels, and constructs and revises interfaces and usage knowledge in a repository. Population-based evolution then selects which repositories to develop further by their performance on validation tasks the meta-agent never sees, favoring changes that generalize beyond the seen tasks. On OSWorld V2, Agent S4, with Qwen3.8-27B and the evolved repository of interfaces, sets a new state of the art in open-source computer-use agents, reaching 21.3% binary score, 3.8 times the same model with fixed interfaces. This is above both GPT-5.5 (13.0%) and Claude Opus 4.8 (20.6%), at about 4.3% and 1.5% of their respective costs. On ScienceBoard, the evolved interfaces improve the success rate from 41.4% to 58.6% with the same model. These results suggest that useful interfaces can be evolved from experience, making autonomous interface design a source of improvement complementary to advances in models and harnesses. We open-source our code at this https URL. 

---
# What Do RTL Repair Benchmarks Actually Measure? 

**Authors**: Yeyin Eva Zhu, Rishabh Ranawat  

**Link**: [PDF](https://arxiv.org/pdf/2610.28595)  

**Abstract**: RTL repair benchmarks should measure whether a model can fix broken RTL given a specification. To this end, we construct and hash-bind RTLRepairDataset, 5,784 (broken, fixed) training pairs; RTLRepairBench, 459 held-out cases with 374 syntactic and 85 behavioral mutations; and an evaluation harness supporting simulation and formal equivalence. We then use this infrastructure to identify two measurement confounds. First, simulation and formal scoring give opposite signs for the apparent effect of providing a specification in one evaluated cohort; reset-aware rescoring attributes almost all of the simulation-formal disagreement to reset and first-cycle comparison semantics. Second, on 274 mined model failures, the specification-only minus specification-conditioned recovery point estimate ranges from -8.5 to +18.3 percentage points across six models. Thus, a conventional repair score need not isolate whether the model used the faulty RTL. We translate these findings into evaluation controls: report initialization, reset, first-cycle, and unknown-value semantics; disclose oracle coverage and undecided cases; and include a specification-only baseline. Together, the resources and audit provide both a benchmark for behavioral repair and a protocol for diagnosing what contributes to its measured score. We release the benchmark, the training set, construction code, sanitized aggregate evidence, and reproduction materials at this https URL. 

---
# Beyond the Sequentiality Bottleneck: A Hybrid Diffusion-Autoregressive Framework for Efficient Multimodal Reasoning 

**Authors**: Wei Cai, Jian Zhao, Yuchen Yuan, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28584)  

**Abstract**: Autoregressive models (ARMs) face a "Sequentiality Bottleneck" in reasoning, while diffusion language models (DLMs) offer parallelism but lack semantic coherence. To addresses both limitations, we propose Collaborative Reasoning Asymmetric Learning (CoRAL), a hybrid framework via an Asymmetric Planner-Executor (APE) architecture: a DLM Planner generates K diverse reasoning trajectories in parallel, while a large ARM Executor selects the optimal plan and synthesizes the final answer. To train the Planner effectively, we introduce Reward-Guided Denoising for CoT (RDC), which integrates outcome-based rewards into the diffusion process to steer it toward goal-directed reasoning. Experiments on vision-language benchmarks show that CoRAL significantly outperforms prior diffusion-based methods and achieves competitive performance against strong ARMs, achieving both superior efficiency and quality. 

---
# Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition 

**Authors**: Wenjie Lou, Tianshuo Peng, Shiyang Feng, Shengji Tang, Runmin Ma, Ruizhe Chen, Jinxin Shi, Haoyang Peng, Fangchen Yu, Zhijie Zhong, Yuhao Zhou, Yue Fan, Yixin Chen, Yiming Wu, Yiheng Wang, Yigeng Jiang, Yichen Jiang, Yi Xie, Xu Guo, Xiaohan He, Tao Hu, Songtao Huang, Siyu Zhou, Shuaiyu Zhang, Shangheng Du, Qinglong Cao, Kun Li, Jingqi Ye, Hongbin Zhou, Guohang Yan, Zongsheng Cao, Ziyi Wang, Zhuo Liu, Zhiyin Yu, Zhiwang Zhou, Zelin Tan, Yuqing Zhu, Yaoze Zhang, Yang Chen, Xuming He, Xiuliang Liang, Xingjian Guo, Xiang Zhuang, Weiping Shen, Weida Wang, Tianyuan Zhou, Tianlin Ye, Tengyuan Fan, Siyu Yan, Shuo Li, Shuchen Pu, Shengyuan Xu, Shao Zhang, Ran Wei, Qinyuan Qin, Qingyu Yang, Nianchen Deng, Meng Dai, Kuikun Liu, Kaitao Chen, Juncheng Xiao, Jiangning Liu, Jianbiao Mei, Jia Yu, Haonan He, Hangfan Zhang, Cheng Yang, Chen Zhang, Bihao Zhan, Benteng Cheng, Yuqiang Li, Yihao Liu, Yang Yang, Xujun Fu, Xiaosong Wang, Xiangyu Yue, Wenwei Zhang, Shuyue Hu, Pinlong Cai, Mianxin Liu, Lei Bai, Jie Zhou, Jie Hou, Jiamin Wu, Hui Wang, Haijun Li, Guangyu Wang, Dongzhan Zhou, Chunfeng Song, Chao Yang, Biqing Qi, Xing Xie, Dahua Lin, Bowen Zhou, Xiangchao Yan, Wenlong Zhang, Shufei Zhang, Peng Ye, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28578)  

**Abstract**: In this report, we introduce Sage-A1.5, a 35B Mixture-of-Experts agentic model that builds on A1 to advance long-horizon problem solving through adaptive capability composition. Compared with A1, Sage-A1.5 learns not only to solve individual tasks, but also to organize work and revise its approach during execution. Sage-A1.5 highlights three improvements: agentic orchestration, verification, and open scientific discovery. First, agentic orchestration extends single-agent execution through task delegation and parallel collaboration among subagents with isolated contexts, enabling complex tasks to be decomposed and addressed concurrently and supporting efficient execution while reducing interference across concurrent subagents. Second, agentic verification strengthens long-horizon reliability through progress checking, evidence validation, and reflective strategy revision, allowing the agent to recognize unproductive search paths and pursue alternative approaches. Third, open scientific discovery extends agentic capabilities beyond answering predefined questions to conducting exploratory research, connecting scientific inquiry with interactive analysis and inspectable research artifacts. We showcase these capabilities through complicated agentic tasks and multidisciplinary scientific applications. Besides, we hope Sage-A1.5 provides the community with an accessible foundation for open scientific discovery, offering a practical path from long-horizon task execution to collaborative, verifiable, and inspectable scientific exploration. 

---
# Evolving Semantic Indexes for Generative Retrieval 

**Authors**: Jinming Nian, Jyun-Yu Jiang, Henry Tsai, Yi Fang, Zhirong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28568)  

**Abstract**: Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end-to-end with gradients and remain largely hand-designed. We introduce a framework that improves these components through LLM-driven evolutionary program search, to our knowledge the first application of this approach to GR. A locally served, open-weight 31B model mutates executable GR components, so no proprietary LLM API is used, and a 1M-parameter proxy decoder provides fast feedback in place of repeatedly training the 1B target generator. Across 21 SID programs on NQ320K, proxy and target Recall@100 show strong rank correlation (Spearman $\rho = 0.90$). Guided by this proxy, evolution improves the seed GR on NQ320K, MSMARCO300K, and a multilingual e-commerce dataset under a fixed training recipe. On NQ320K, jointly evolving SID construction and collision handling improves Recall@1 from 55.8% to 59.3% and Recall@100 from 85.8% to 90.6%. A component study shows that proxy reliability depends on what is evolved: improvements transfer reliably for the semantic index but not for the generator's training objective or the full pipeline. LLM-driven evolutionary search is thus a practical tool for GR design, provided proxy reliability is established for the chosen search space. 

---
# OptScientist: Automated Discovery of Optimization Laws 

**Authors**: Ma ZhenYao, Yue Liang, Jian-Qiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28562)  

**Abstract**: Many scientific phenomena are best understood as outcomes of underlying optimization problems, yet automated scientific discovery has largely focused on recovering equations that describe observed input-output patterns. We introduce OptScientist, a novel automated framework for discovering compact symbolic optimization laws directly from data. OptScientist comprises (i) a model that infers symbolic optimization problems by first learning to represent data as an optimization problem and then extracting its symbolic form, and (ii) an LLM agentic system embedding the model in a discovery process that integrates theoretical priors from scientific literature, discovers candidate laws, abstracts their underlying mechanisms, and suggests new data to collect. We first evaluate OptScientist on OptFeynman, a novel benchmark of 99 physics-derived optimization problems, where OptScientist recovers ground-truth laws for over 80% of the problems and remains robust to noisy contextual information, while substantially outperforming machine-learning, symbolic, and LLM-based model-discovery baselines. On human risky choice and central-bank monetary policy, OptScientist discovers compact symbolic optimization laws that achieve held-out predictive performance competitive with leading baselines, while using few parameters and substantially fewer tokens than LLM baselines. Overall, OptScientist extends automated scientific discovery to uncovering optimization laws that generate observed phenomena, enabling automated theory building across the physical, biological, behavioral, and social sciences. 

---
# Agentic Ensemble with Vision-Language Models for Visual Anomaly Detection 

**Authors**: Shangquan Sun, Wenjun Luo, Xu He, Yoonsoo Lee, Shijian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28560)  

**Abstract**: Visual anomaly detection aims to detect abnormal objects, while anomaly samples are often scarce during model training. Assembling multiple pretrained detectors can help by leveraging their transferable image scores and anomaly maps. However, most existing ensemble methods apply fixed fusion rules over saved predictions which struggle when pretrained detectors clearly disagree due to different training objectives, priors, and visual representations. To this end, we design AnomalyAgent, a post-training fusion framework for accurate visual anomaly detection. AnomalyAgent aligns heterogeneous scores and maps, summarizes score- and region-level disagreement, and treats fusion as a finite action-selection problem. It implements two types of actions: 1) score actions that determine how to fuse image-level detector confidence via consensus, robust aggregation, detector weighting, or score-map consistency; 2) map actions that fuse dense anomaly responses via consensus, detector weighting, geometric aggregation, or disagreement correction. A constrained vision-language model (VLM) is introduced to inspect high-conflict samples and select valid score action and map action to produce the final prediction. We evaluate AnomalyAgent on MVTec AD and VisA as primary benchmarks, together with seven additional public anomaly benchmarks and Real-IAD for large-scale score-level evaluation. Across these benchmarks, AnomalyAgent consistently improves over equal-weight fusion and achieves the best mean image AUROC and pixel AP among the compared fusion methods on MVTec AD and VisA, while demonstrating robust transfer across additional anomaly domains. 

---
# Humans Correct Their Judgments Through Debate, but Weaker AI Models Do Not 

**Authors**: Maria Victoria Carro, Denise Alejandra Mester, Facundo Nieto, Oscar Agustin Stanchi, Guido Ernesto Bergman, Giovanni Franco Gabriel Marrafini, Trinidad Borrell, Guido Freire, Joaquin Salvador Machulsky, Mario Alejandro Leiva, Ana Lopez, Nicolas Spinelli, Luca Nicolas Forziati Gangi, Eitan Sprejer, Federico Barrera Lemarchand, Gerardo Simari, Maria Vanina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2610.28539)  

**Abstract**: How can we effectively oversee AI systems that surpass human intelligence? AI Safety via Debate proposes that two AI systems argue opposing positions and a judge, either a human or a weaker model, decides which is correct. By facilitating supervisors in providing high-quality training signals, debate aims to encourage truthful and safe behavior from AI systems whose capabilities exceed our own. However, human judgment is not neutral: people often rely on biases which advanced AI systems may learn to exploit. Drawing on insights from collective intelligence and social cognition, this paper investigates whether debate can guide human and LLM supervisors toward the truth despite their prior beliefs. We evaluate several protocol configurations. We also explore setups inspired by the wisdom of the crowd such as the number and diversity of judges, as well as interaction mechanisms including deliberation and voting. Overall, we find that debate increases accuracy relative to prior beliefs for human judges, whereas the consultancy baseline, in which a single expert argues for one answer, does not. For weaker language models, however, the results are mixed: debate provides no comparable gains in most settings and can even reduce accuracy. Among multi-judge protocols, weighted voting produces the highest accuracy for human supervisors, whereas for LLMs none of them significantly improves group accuracy, even as the number of judges increases. Notably, the behavior of minority judges in non-diverse deliberation protocols suggests that, under symmetric settings, incorrect positions could be more persuasive than correct ones. These findings support debate as a viable path toward scalable human oversight, but cast doubt on its benefits for weaker AI supervisors, raising concerns about fully automated training and control pipelines in high-stakes settings. 

---
# Beyond 4 bits: Data-Centric Post-training Quantization and Recovery 

**Authors**: Rhea Sanjay Sukthanker, Riccardo Grazzi, James Hensman, Pashmina Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2610.28537)  

**Abstract**: Post-training quantization (PTQ) compresses the weights of a trained model with limited compute and data. Down to 4 bits it is near-lossless, but at 2 bits scalar PTQ configurations fall to chance-level accuracy. We study three choices usually treated as implementation details that matter little at 4 bits but decide 2-bit quality: the quantization representation, the calibration data, and the recovery fine-tuning data, across six dense and mixture-of experts models (ranging from $3.8$ to $32$B parameters). Representation is the largest factor and vector codebooks restore usable models where scalar PTQ pipelines collapse. Most of the remaining loss in accuracy is on \emph{generative} tasks, and the amount of performance drop is model-dependent. Furthermore, at a short recovery budget, calibration and recovery data are coupled. The best recovery data depends on the calibration data, so the two must be chosen jointly. The resulting recipe involves a vector codebook representation, a small task-aligned calibration mixture, and an inexpensive codebook-only recovery stage. Including codebooks, it uses about $2.3$ effective bits per weight and retains $89$ to $97\%$ of full-precision question-answering (QA) accuracy where scalar PTQ is at chance, and, including recovery, uses $1.4$ to $30\times$ less compute than the strongest quantization-aware-training (QAT) baseline. 

---
# DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems 

**Authors**: Xialie Zhuang, Mingcong Lei, Zunhai Su, Ri Liu, Jianjin Li, Peixian Ma, Jia Pan, Ngai Wong, Guanhua Chen, Chuan-Wei Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28522)  

**Abstract**: Decision models support routing, guardrails, and output verification in LLM systems. However, item accuracy alone cannot establish whether models make correct decisions across controlled variations or assign reliable confidence to their predictions. We introduce DecisionPivot, a diagnostic benchmark comprising 48 tasks and 40,033 items across seven decision roles. Its core track contains 7,800 items from 13 generated task families, organized into contrast groups with computed answers verified by independently implemented solvers. The complementary reference track contains 32,233 items from 35 public test sets, preserving their original evaluation metrics. We evaluate 40 decision models and 14 general models using documented, model-specific protocols, assessing item and contrast-group accuracy, probability ordering, calibration, and empirical selective coverage. Our results reveal substantial differences obscured by aggregate scores: two 27B models separated by only 2.6 points on the reference track achieve contrast-group accuracies of 70.1% and 44.2%. Across 2,400 binary contrast pairs, Perplexity Decider correctly ranks the positive member above the negative member in 92.3% of pairs but answers both members correctly in only 74.1%, exposing a gap between probability ordering and decision correctness. Models with similar item accuracy also differ in the reliability of their high-confidence predictions. On a separate 1,296-item subset, GLM-5.3 achieves 93.5\% contrast-group accuracy at a higher reasoning setting, compared with 81.5% at its lowest effort. By linking errors to pivotal details and difficulty levels, DecisionPivot provides fine-grained diagnostics of decision failures that aggregate evaluation alone cannot reveal. 

---
# Proof-Carrying Contracts: Commitments AI Agents Cannot Break 

**Authors**: Carlo Ciliberto, Alessandro Rudi, Enrico Tassi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28484)  

**Abstract**: AI agents increasingly act on behalf of people or other agents, purchasing services, exchanging information and delegating work. What an agent must respect is written in instructions, policies or agreements, but text alone cannot make it comply. It may make mistakes, be redirected by what it reads, or pursue different interests. For an economy of such agents to function, parties need agreements they can rely on without inspecting or trusting one another's agents. We introduce Proof-Carrying Contracts (PC2), a framework for making agreed constraints enforceable throughout an agent's interactions with a system. Before a requested operation can run, the agent must supply a machine-checkable proof that it preserves the agreed constraints in every case allowed by its model. An execution system called the Airlock checks the proof and controls access to the operation. The parties specify the conditions to preserve, while the agent supplies the reasoning needed to justify its decisions and remains free to adapt its strategy. This avoids having to design a separate permission policy and prove that every decision it permits is safe. We prove that the constraints hold at every instant during execution, even for an adversarial agent, and identify when this implies safety in the world. Reusable capability modules provide tools and guarantees from which contracts can be constructed. PC2 provides a foundation for an agent economy whose agreements protect both those delegating work and those carrying it out. 

---
# Exploring Temporal Context-Aware Physical Grounding in Latent World Models 

**Authors**: Yilei Man, Yang Chen, Jingcai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28483)  

**Abstract**: Learning latent dynamics is fundamental to building action-conditioned world models for planning. Among existing families, Joint Embedding Predictive Architectures (JEPAs) offer a natural framework by modeling transitions directly in latent space rather than at the pixel level. However, two issues remain in JEPA-based world models: (i) physical grounding lacks explicit aggregation of temporal context and (ii) action planning always overly relies on generic sampling distributions. To this end, we propose $\mathcal{G}$-JEPA, which incorporates temporal context into physical grounding for latent world models and couples the learned models with prior-guided action planning. Specifically, during learning, we incorporate a shared causal temporal module that aggregates temporal context within observation and prediction sequences under physical state supervision, grounding both in task-relevant physical states. When planning, we further introduce action inference through stepwise conditioning of a generative action distribution on the current and goal latent representations together with their evolving discrepancy, yielding anchored proposals that guide model-based online refinement. Together, physical state supervision guides temporal representation learning, while the action prior guides search through the learned world model. Experiments demonstrate improved planning performance on manipulation tasks, including tasks with increasing temporal separation between initial and goal observations. Code is publicly available at this https URL. 

---
# Claims of emergence in AI should be grounded in information decomposition 

**Authors**: Christoph Riedl, Fernando E. Rosas  

**Link**: [PDF](https://arxiv.org/pdf/2610.28479)  

**Abstract**: When a language model develops capabilities not traceable to its individual components, when a multi-agent AI system outperforms each agent on its own, or when a human-AI pair succeeds where either fails in isolation, AI researchers describe it as a case of ``synergy'' or ``emergence''. However, these are usually informal assessments not supported by a theoretical framework, making these claims impossible to compare or even meaningfully dispute. Here we argue that \emph{information decomposition can serve as a unifying formal framework to operationalize assessments about synergy in AI systems}. Information decomposition provides a way to distinguish qualitatively different \emph{types} of information, including unique information available from individual components, redundant information shared across multiple components, and synergistic information accessible only from parts taken together. This gives claims of emergence a precise, falsifiable criterion: a system's behavior is emergent only if it carries synergistic information about task-relevant outcomes. We apply this lens to attention heads, autonomous agents, and human-AI teams, showing how it organizes existing knowledge across scales and substrates, while also generating cross-level conjectures that siloed fields would miss. 

---
# Beyond Trajectory Statistics: Community-Consistent Human Mobility Generation via Evolving User-Location Graphs 

**Authors**: Bo Liu, Tong Li, Jiapeng Zhang, Zhu Xiao, Zhuo Tang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28467)  

**Abstract**: Human mobility data are fundamental to Web-based location services and broader urban applications, and mobility generation provides synthetic trajectories when real data are costly or inaccessible. Although recent mobility generators have incorporated collective mobility patterns, they primarily focus on individual trajectories and aggregate population dynamics, leaving evolving community structures formed by user-location interactions insufficiently modeled. In this paper, we study community-consistent human mobility generation. To this end, we formulate mobility generation as the generation of dynamic user-location graphs to model community structures and propose MOTION, a Multi-agent framework for cOmmuniTy-consIstent human mObility geNeration. MOTION consists of user agents, location agents, and a global manager agent. MOTION first detects coherent communities from historical mobility using the Leiden algorithm and extracts community-conditioned interaction priors to guide the generation process. Guided by these priors, the user agents plan mobility, search destination candidates, and infer user movements from the individual view. In parallel, selectively activated location agents infer likely user arrivals from the community view. The manager coordinates both views, diagnoses interactions causing deviations, and attributes targeted reflection signals to revise user decisions, location attraction, and source-user selection. Experiments on two real-world datasets show that MOTION outperforms baselines by at least 4.05% across reported metrics of individual trajectories, population and flow distributions, and community and graph structures. Further analyses confirm its robustness, privacy preservation, and downstream utility. The code is available at this https URL. 

---
# Toolflation: Seller Agents Distort Tool Markets Through Spontaneous Generative Engine Optimization 

**Authors**: Zhenwei Tang, Di Bai, Jintao Liu, Xidong Wu, Qianfeng Wen, Xinwu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.28454)  

**Abstract**: Agents are starting to sell to other agents. Seller agents write the listings of tools, services and products, and buyer agents read these listings when they choose what to use. We show that seller agents told only to be selected more often practice generative engine optimization on their own, and that this distorts the market for sellers and for buyers. We build tool markets from real data tools on Amazon and Instagram and run every tool on every question, so we know which tools can answer each question. Under all five production models we test, sellers rewrite their descriptions to claim more and more fields, including fields that their tools return for fewer than half of the targets. We call this process toolflation. Nothing in the market limits it, and how far it goes depends on the model, as the number of claims that contradict the sellers' own evidence differs by an order of magnitude between models. Toolflation shifts the buyers' choices between sellers, and the most capable tool loses share in seven of the ten combinations of seller model and platform. Under four models, the choices stay with capable tools and buyer agents keep their task success. Under the fifth, sellers copy one another until the descriptions no longer tell the tools apart. Buyer agents then pick tools by their position in the list, tools that answer almost nothing win a quarter to two fifths of the choices, and task success falls from 89% to about 30% on Amazon, close to random choice, and from 77% to 40% on Instagram. The sellers did only what their vendors asked, and the cost fell on buyers they never dealt with. We therefore argue that honesty toward third parties under competition is part of safety alignment, and that models should be tested as sellers before they are deployed as sellers. 

---
# Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity 

**Authors**: Sahil Joshi, Prithvi Dixit, Aryan Katiyar, Agniva Chowdhury, Joseph E. Gonzalez, Anshumali Shrivastava, Aditya Desai, Kumar Krishna Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2610.28441)  

**Abstract**: Large Language Model inference over long contexts is increasingly bottlenecked by the cost of attending to a growing Key-Value (KV) cache. Sparse attention reduces this cost by accessing only a fraction of the cache, but existing methods face a trade-off between deterministic retrieval, which captures high-scoring tokens but discards residual attention mass, and sampling, which improves coverage at the cost of higher variance. Hybrid methods combine top-$k$ retrieval with uniform sampling, yet still exhibit a substantial gap from dense attention at aggressive sparsity. We introduce \emph{Stratified Sparse Attention (SSA)}, a unified framework combining deterministic retrieval with score-guided sampling. SSA deterministically attends to high-scoring tokens while sampling the residual using proxy attention scores and correcting for their inclusion probabilities, thereby recovering attention mass beyond the retrieval set. SSA encompasses top-$k$, top-$p$, and hybrid top-$k$+sampling as special cases. Across RULER, LongBench, LOFT, and AIME, and five models ranging from 3B to 27B parameters, SSA consistently improves over the baselines at matched sparsity. On RULER-HARD with Ministral-3-8B, SSA improves accuracy over the strongest practical baseline from 81.8\% to 84.2\% at 32K and from 63.8\% to 69.3\% at 128K under $50\times$ attention sparsity, recovering 94\% and 85\% of dense-attention performance, respectively. With optimized CUDA kernels, SSA achieves up to $2.96\times$ decode speedup over FlashInfer at 1M context, demonstrating that its accuracy gains translate into practical long-context inference acceleration. We open-source our code at this https URL. 

---
# DIVE: Benchmarking Full-Duplex Interaction with Visual Engagement in Continuous Streams 

**Authors**: Yuhang Chen, Guangyue Liu, Shilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28405)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have substantially improved spoken interaction and video understanding. Existing online and streaming benchmarks primarily assess causal multimodal perception through timestamped question answering, whereas real-world interaction requires models to continuously identify task-relevant multimodal cues and decide whether, when, and how to respond. This leaves open how well current models can sustain real-time interaction throughout an evolving multimodal stream. To address this gap, we introduce DIVE, a benchmark for evaluating full-duplex interaction with visual engagement under continuous and realistic multimodal streams. It organizes six tasks into three complementary interaction scenarios. (1) Proactive Intervention evaluates whether models can identify emerging visual or semantic cues and intervene at the appropriate moment. (2) Continuous Engagement evaluates sustained participation through continuous visual tracking and visually grounded query interaction. (3) Interaction Coordination evaluates how models coordinate listening and speaking under overlapping and multi-party interactions. DIVE contains 525 audio/video samples with fine-grained human annotations and precise temporal metadata. Its evaluation protocol separates response quality, participation decisions, and response timing. We evaluate representative real-time models using response quality, window response rates, and reaction latency. Experiments show that leading real-time models achieve sub-second average response latency, yet their average window response rates remain at or below 53.77%. Response quality varies substantially across models and tasks, with adapting ongoing responses to newly arriving information remaining particularly challenging. We hope DIVE facilitates progress toward more natural and capable real-time multimodal interactive systems. 

---
# RACE: RoPE-Aligned Attention Compression Based on Query-Key Activation Energies 

**Authors**: Irene Tenison, Abdulrahman Alabdulkareem, Daniel Karl I. Weidele, Nima Dehmamy, Dan Gutfreund  

**Link**: [PDF](https://arxiv.org/pdf/2610.28385)  

**Abstract**: Head-dimension compression can reduce the memory required by the key-value (KV) cache, but existing methods often reconstruct full-dimensional representations during inference, adding significant decoding FLOPs overhead. We formalize the design considerations for converting pretrained attention into a natively reduced-dimensional form that reduces both KV-cache storage and attention computation using a calibration set. Specifically, we consider reductions that (1) apply a shared projection to queries and keys so that they remain in a common reduced feature space, (2) derive this projection jointly from query and key activations to better preserve their interaction, and (3) are compatible with the pretrained RoPE transformation. To obtain a reconstruction-free realization of this reduction, we restrict it to coordinate selection and theoretically show that RoPE compatibility then holds if and only if complete rotary pairs are retained. Building on this characterization, we introduce RACE, a post-training method that selects rotary pairs using their joint query-key activation energy. We theoretically show that this criterion minimizes a separable, calibration-dependent upper bound on post-RoPE query-key product approximation error. The resulting reductions are folded into the model weights and require neither runtime reconstruction nor specialized kernels. Across three model families and different compression levels, RACE outperforms structured pruning and remains competitive with reconstruction-based methods on language-modeling, zero-shot, and long-context tasks. At 50% KV-cache retention and 256K context, RACE reduces total inference FLOPs by 47.8%, yielding 1.79x higher throughput, 43.2% lower time-to-first-token (TTFT), and 28.1% lower allocated GPU memory in comparison to an uncompressed model. 

---
# FALCA: Falsification-Aware Reinforcement Learning with Hierarchical Credit Assignment for Long-Horizon Deep Search Agents 

**Authors**: Zijun Dou, Kun Luo, Chaofan Li, Shuqi Lu, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28361)  

**Abstract**: Long-horizon deep search requires LLM agents to iteratively decompose complex information needs, gather evidence from external sources, and maintain a reliable research state under multiple constraints. A central challenge is that early mistakes can be amplified over the search trajectory: agents may anchor on an incorrect candidate, collect partially supporting evidence, overlook contradictions, and accumulate errors that misguide subsequent reasoning. We propose FALCA, a falsification-aware framework that formulates deep search as a propose-falsify process. FALCA adopts a collaborative Searcher-Falsifier architecture, where the Searcher proposes candidate answers or sub-answers, while independent Falsifiers actively seek counterevidence, check unmet constraints, identify conflicts, and produce concise falsification reports for research-state revision. To strengthen this loop, we introduce falsification-aware training, combining SFT warmup on falsification trajectories with agentic RL based on parallel rollouts and hierarchical credit assignment over research stages, state transitions, and actions. Experiments on five search benchmarks show that FALCA improves long-horizon deep search performance over backbone and harness ablations, while reducing confirmation bias and error accumulation through more effective falsification and research-state revision. 

---
# S-Agent 2.0: Spatial Reasoning with a General Coding Agent 

**Authors**: Rui Gong, Ruoyu Wang, Yalun Dai, Hao Li, Sidi Yang, Qihao Zhao, Hongyu Li, Zeyang Guan, Fangzhou Hong, Zhaoxi Chen, Kim-Hui Yap, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28354)  

**Abstract**: Spatial reasoning asks a model to work out where objects are in 3D space, how they relate to one another, and how they move. Agentic VLMs have recently brought large improvements on such tasks by letting the model actively gather evidence with 3D perception tools instead of answering in a single pass. Even so, on many questions the tools return the right evidence and the model still does not arrive at the correct answer: the gap between model and tools, where raw readings become usable conclusions, is left unfilled. We present S-Agent 2.0, a spatial reasoning agent that equips an off-the-shelf general coding agent with a command-line perception toolkit, and we design the interface around one principle: help the VLM close the evidence loop: acquire scene evidence, remember it in a persistent scene memory, and interpret each reading in plain spatial language. This design keeps the full flexibility of a coding agent, and every reading it acts on stays grounded in the tools' geometry. On MMSI-Bench and ReVSI, the toolkit improves a strong open-source model, Qwen3.8-27B, by +13.9 and +20.5 points over the same agent without tools, and a strong closed-source model, Gemini-3.8-Flash, by +11.2 and +9.6 points. With the open backbone S-Agent 2.0 reaches 66.5 on MMSI-Bench and 82.3 on ReVSI, ahead of the specialized system SpatialClaw on the same backbone (62.6 and 74.5), and with Gemini-3.8-Flash it reaches 74.1 and 90.5 against its 67.0 and 79.7. Both tool gains shrink as the backbone grows more capable, but SpatialClaw's shrinks far more on ReVSI, from +13.2 to +2.7, while ours stays at +9.6. S-Agent 2.0 also works across different general coding agent harnesses, such as Codex and Claude Code. 

---
# Is Sycophancy a Form of In-Context Updating? 

**Authors**: Itai Shapira, Myra Cheng, Hadas Orgad  

**Link**: [PDF](https://arxiv.org/pdf/2610.28325)  

**Abstract**: Factual sycophancy, a language model's deference to a user's false beliefs, is usually understood as a bid for the user's approval at the expense of truthfulness. We argue that this framing over-emphasizes the user and propose viewing sycophancy as a failure within the broader and generally useful process of in-context updating, by which models revise their answers based on information supplied during an interaction. We support this view with behavioral and mechanistic evidence. First, models' responses to user claims sit within a broader pattern of source-sensitive updating. They revise their answers in response to information attributed to external sources as well, more so when it appears better supported, and their deference to the user persists even when no user approval is at stake. Second, the internal components that support deference to the user are also causally important for in-context updating more broadly. Our results suggest that factual sycophancy arises when in-context updating gives a user's assertion excessive weight relative to its evidence. Because sycophancy is embedded in this process, reducing it risks reducing useful updating as well, a trade-off that calls for evaluations measuring both behaviors and mitigations that decouple them. 

---
# ChartEvo: Self-Evolving Agent Harnesses for Long-Tail Chart Reasoning 

**Authors**: Qi Wang, Peijie Wang, Jun Cao, Haoyang Tong, Xiao-Hui Li, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28282)  

**Abstract**: Multimodal large language models (MLLMs) have substantially advanced chart understanding and reasoning, yet their performance remains highly uneven across chart types. While modern MLLMs perform well on common charts, their accuracy drops sharply on long-tail charts with diverse visual structures and reasoning requirements. Tool-augmented agents can improve complex chart reasoning by introducing specialized visual capabilities, but manually engineering strategies and tools for the heterogeneous and evolving long tail is difficult to scale. We introduce \textbf{ChartEvo}, a recursive self-improvement framework that enables chart agents to adapt to underperforming chart types through experience-driven harness-level self-improvement. ChartEvo jointly evolves reusable chart-specific Skills and executable Tools, while learning reusable local tool-call chains to compose evolved capabilities and control the inference overhead of tool-augmented reasoning. On long-tail charts from ChartBench and ChartX, ChartEvo improves accuracy over its base MLLM, Qwen3.7-Plus, by \textbf{45.6\%} and \textbf{32.5\%}, respectively, demonstrating the effectiveness of recursive self-improvement for long-tail chart adaptation. 

---
# MaD-RSI: Model-as-Data for Recursive Self-Improvement of Agent Harnesses 

**Authors**: Yu Cui, Hong He, Ruiqing Yue, Sicheng Pan, Zhuoyu Sun, Yifei Liu, Ting Li, Wenzhuo Zhu, Baohan Huang, Haibin Zhang, Cong Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28201)  

**Abstract**: Recursive self-improvement (RSI) of agent harnesses has emerged as an effective approach to enhancing the capabilities of large language model (LLM) agents. By searching the harness space for a specific LLM on a given dataset, RSI can discover a harness tailored to the capabilities and behavioral characteristics of that model. However, such optimization often creates a strong coupling between the learned harness and its target LLM, limiting generalization across models and requiring substantial computational resources to adapt to new LLMs. Moreover, existing RSI methods typically optimize a harness using feedback from a single model and evaluate its generalization to other models only after training. This paradigm treats model differences primarily as a criterion for evaluating generalization rather than as valuable evidence for optimization. In this paper, we introduce a new perspective that treats models as data and places the harness at the center of optimization. Building on this perspective, we propose \textsc{MaD-RSI}, a framework that directly optimizes a shared harness across multiple heterogeneous LLMs by incorporating their diverse failure trajectories into the training process. Rather than assessing model differences only after optimization, \textsc{MaD-RSI} leverages these differences as training evidence to guide the self-improvement of a shared harness. Furthermore, we introduce a new evaluation framework with two complementary metrics designed to assess the effectiveness of harness optimization across models. Experiments across multiple datasets and LLMs demonstrate that \textsc{MaD-RSI} improves generalization over existing approaches while reducing training costs. 

---
# Interruption-Aware Recurrent Reinforcement Learning for UAS Navigation Under Temporary Overrides 

**Authors**: Christopher Conrad, Deepak Panda, Weisi Guo, Antonios Tsourdos  

**Link**: [PDF](https://arxiv.org/pdf/2610.28121)  

**Abstract**: Safe autonomous navigation of UASs requires tactical deconfliction to address conflicts that strategic coordination cannot prevent. This task is challenging under partial observability, when sensor or traffic data are missing, delayed, or degraded. Geometry-based methods can offer separation assurances but face limitations in dense encounters. Recurrent reinforcement learning has therefore been proposed to learn navigation policies that integrate observation-action histories to optimise long-term safety-efficiency trade-offs. However, operational safeguards may require a remote pilot, onboard detect-and-avoid system, or runtime assurance mechanism to temporarily override the learned policy. A research gap remains in understanding how these interruptions affect recurrent policies when the agent regains authority. Benchmark evidence shows that interruptions make task variables less decodable from the recurrent hidden state and increase post-interruption collision risk. The findings suggest that altered recurrent context prolongs post-handback degradation through an action-observation feedback loop, in which changed actions alter the flight path and subsequent recurrent updates. To address this vulnerability, we propose three interruption-aware architectures that condition recurrent processing on interruption context through separate feature streams, dedicated interruption memory, or gated environmental features. Across simulations, all three architectures preserve more decodable task information than a standard recurrent policy. Moreover, the strongest architecture reduces post-interruption collisions by 74% and interruption-induced mission failures by 69% relative to the recurrent baseline, without degrading interruption-free performance. These results show that safe interruption does not guarantee safe handback, and support interruption-aware recurrent design for resilient UAS autonomy. 

---
# Robust Representation Learning for Spiking Neural Networks based on Conditional Information Bottleneck 

**Authors**: Yuli Zha, Wang Fan, Yixiao Dai, Lingjie Wang, Bang Hu, Wei Cao, Fengzhe Zhang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28110)  

**Abstract**: Spiking Neural Networks (SNNs) have shown great potential for low-power neuromorphic computing due to their event-driven processing and spatiotemporal sparsity, but they remain vulnerable to input perturbations, hardware-intrinsic noise, and adversarial attacks in real-world deployment. We attribute this vulnerability partly to input-specific hidden representations learned by standard SNNs trained with conventional cross-entropy loss, as our information-plane analysis empirically shows that substantial input-related information can still be retained after label-predictive information becomes stable. To address this issue, we propose SCIB, a simple yet effective robust learning objective for SNNs based on the Conditional Information Bottleneck. Different from conventional variational IB methods, SCIB replaces the mutual information compression term $I(X;S)$ with the conditional mutual information $I(X;S|Y)$ and derives a tractable variational upper bound. Moreover, it introduces a learnable class-conditional prior instead of a label-agnostic isotropic Gaussian prior, improving the expressiveness of variational compression and encouraging class-structured representations when combined with the supervised classification objective. Systematic experiments on Fashion-MNIST, CIFAR-10, and CIFAR-100 under external environmental noise, neuromorphic hardware noise, and adversarial attacks show that SCIB improves robustness over standard SNNs and IB-based baselines under most evaluated noise conditions, improves adversarial accuracy under the evaluated attacks without using adversarial training, and reduces the SynOps-based single-sample inference energy consumption by 24.9% relative to an unconstrained SNN baseline. 

---
# A Semi-Analytical Inference Framework for Structural Causal Models 

**Authors**: Tatiana Boura, Robert Peharz  

**Link**: [PDF](https://arxiv.org/pdf/2610.28103)  

**Abstract**: Structural Causal Models (SCMs) provide a principled framework for observational, interventional, and counterfactual reasoning, all of which rely on probabilistic inference. Existing frameworks either transform SCMs into alternative tractable representations or rely on approximate inference with expressive generative models. We instead perform inference using a message passing scheme directly on the SCM by representing structural mechanisms and probability densities in a common finite-dimensional function space. Forward propagation combines a numerical realization of the Law of the Unconscious Statistician, while propagated densities are recovered by spline projection from LOTUS moments. The same representation naturally yields backward message passing for conditioning, providing a unified framework for causal inference. Using tensor-product B-spline bases, the resulting algorithm expresses inference through numerical quadrature and linear algebra without explicitly constructing conditional distributions. Experiments on randomly generated continuous SCMs demonstrate accurate approximation of Monte Carlo reference solutions and systematic improvement with increasing basis resolution. 

---
# Polyscene: Scenario Planning as Diversity Planning 

**Authors**: Mustafa F. Abdelwahed, Joan Espasa, Felipe Meneguzzi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28102)  

**Abstract**: Scenario planning is a foresight practice in which analysts construct a small set of plausible futures and prepare an organisation for each. Each scenario is a narrative that must be plausible, internally consistent, relevant and challenging, and the set as a whole must be differentiated. The practice remains manual: a workshop (i.e., a facilitated meeting in which a small group of experts writes the narratives) considers only the factors it has time for, carries the biases of the experts who run it, and cannot be redone whenever conditions change. In this paper, we automate scenario planning by casting it as a diversity planning problem, in which two scenarios differ when they end in different narratives on the analyst's drivers. The result is a set of scenarios, one for each narrative that the topic can reach within a budget. Our implementation, Polyscene, is a generic engine that generates scenarios from a topic. We use geopolitical conflicts and shipping-route disruptions as case studies. For the first, we build the planning problem's initial state from the GDELT Project so it reflects current relationships among the countries involved. Polyscene then generates scenarios showing how relationships among those countries change. With a board of six countries as of 1 January 2021 and a budget of five events, the conflict set puts five of the 15 country pairs at war, and each of the three wars that break out among these countries by 1 January 2026 ends a scenario. For the second, we build the initial state from ship traffic, a supply-pressure index, and the news, and Polyscene generates scenarios showing what a closed or diverted route could do to a manufacturer's parts stock. With four routes per board and the same date and budget, each set includes three outcomes, and the Suez Canal set includes the March 2021 blockage and the December 2023 Red Sea container diversion. 

---
# InsightRAG: Aggregating Distributed Scholarly Judgments for Scientific Survey Generation 

**Authors**: Yuquan Lin, Wenshuo Zhang, Yilin Ye, Shuchang Xu, Zhao Chen, Weiqi Wang, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28085)  

**Abstract**: The rapid growth of scientific literature makes survey writing increasingly challenging, motivating automatic survey generation. Yet, existing systems must search an open space of papers and an open space of concepts with no trace of how the community itself has navigated them, leading to incomplete scope, weak taxonomies, and unreliable citations. Our core insight is that author-written Related Work sections contain the human judgments needed to address both limitations: authors curate the citations relevant to their contributions and explain how the cited works relate, yielding paths for scope reconstruction and relational evidence for organization. We propose InsightRAG. Its insight-guided retrieval follows author-supported citation relations to reconstruct a verified research scope and structure the resulting relational evidence. It aggregates these judgments under a minimum-conflict objective into a taxonomy that reflects recurring author-written perspectives in the retrieved literature, guides generation, and bounds each section's evidence. Results on ten topics show gains in scope reconstruction, taxonomy quality, citation support, and reference reliability. 

---
# IECG: Weakly-Supervised Interpretability-Enhancing Foundation Model for Electrocardiogram 

**Authors**: Huynh Dang Nguyen, Trong-Thang Pham, Anh-Thuan Do, Phuong-Nhi Dinh, Brijesh Patel, Ngan Le, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28083)  

**Abstract**: Electrocardiography (ECG) is a cornerstone of cardiac diagnosis, yet most machine learning methods for ECG analysis lack interpretability, limiting their clinical adoption. We present IECG, a weakly-supervised ECG foundation model that jointly performs diagnostic classification and abnormality localization without requiring manual abnormal region annotations, overcoming a long-standing bottleneck in interpretable ECG learning. Central to IECG is a weakly-supervised attention strategy that explicitly incorporates clinical priors by deriving abnormality heatmap pseudo-labels from established ECG clinical knowledge. These pseudo-labels guide the model's CLS-token attention through a KL-divergence loss, encouraging it to focus on diagnostically relevant waveform regions. During pretraining, IECG is jointly optimized with clinically guided attention supervision, ECG-text contrastive alignment, and report captioning objectives, enabling it to learn rich multimodal representations while producing localized waveform evidence. By explicitly aggregating features from clinically guided attention regions, IECG grounds its diagnostic predictions in interpretable waveform evidence without sacrificing classification performance. Extensive experiments demonstrate that IECG achieves state-of-the-art abnormal region localization while maintaining strong classification performance, advancing trustworthy foundation models for ECG analysis. 

---
# Slot-JEPA: Object-Structured Prediction Interfaces for Latent World Models 

**Authors**: Peisong Wen, Qianqian Xu, Minye Lei, Siran Dai, Yang Liu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28074)  

**Abstract**: As a prototypical latent world model, Joint-Embedding Predictive Architectures (JEPAs) provide strong semantic representations. However, dense patch representations expose a redundant interface to downstream predictors, which must implicitly infer spatial organization and temporal correspondence, presenting three challenges. First, object information is distributed across patches, forcing each downstream predictor to redundantly recover object structure that could otherwise be reused across tasks before modeling interactions. Second, the lack of explicit entity correspondence across time makes historical transitions expensive, restricting the predictor's flexibility in utilizing context. Third, without explicit transport structure, existing JEPA models typically use pointwise l1/l2 distances, which are insensitive to spatial displacement when predicted and goal supports are largely disjoint. We propose the Slot-based Joint-Embedding Predictive Architecture (Slot-JEPA), which decomposes dense representations into stable semantic slots and time-varying spatial masks. Pretrained once and frozen downstream, this decomposition exposes a reusable object-structured interface. Building on it, our Transition Memory Predictor (TMP) stores visual transitions as compact masks and retrieves useful context using statistics of corresponding slots. Its two branches separate slot-specific dynamics from shared cross-slot dynamics. We further use Unbalanced Optimal Transport (UOT) to supervise mask prediction and evaluate candidate plans by transporting mass between aligned predicted and goal masks, providing a signal better aligned with spatial progress. Across seven visual-understanding benchmarks, the slot-reconstructed features remain transferable. Slot-JEPA also improves goal-conditioned planning across six planar environments and four robotic arm planning tasks on the REALM-JEPA benchmark. 

---
# TrajEvolve: Trajectory Evolution for Reinforcement Learning with Hindsight Credit Assignment 

**Authors**: Yi Wen, Hao Chen, Wanyu Wang, Maolin Wang, Pengyue Jia, Derong Xu, Huize Tan, Yingyi Zhang, Wenlin Zhang, Weihong Luo, Xiku Du, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28060)  

**Abstract**: The Reinforcement Learning (RL) paradigm has achieved remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs). However, when applied to long-horizon tasks, it faces severe challenges: i) Due to the high task complexity, the model struggles to sample enough successful trajectories, resulting in slow convergence. ii) Relying solely on sparse final rewards causes all steps to be updated indiscriminately, encouraging redundant step generation. When both the quantity and quality of successful trajectories become difficult to guarantee, the performance is severely constrained. This dilemma stems from the limitations: existing RL paradigms rely entirely on the reasoning capability of base models, lacking the ability to actively optimize trajectories. To address the above issues, we propose a novel trajectory evolution paradigm with hindsight credit assignment, termed TrajEvolve. TrajEvolve not only relies on the trajectory inferred by the model but also generates ``evolutionary trajectories'' by refining low-quality steps in historical trajectories, thereby exposing the policy to high-quality compositions that lie within the base model's support but are rarely sampled within practical training budgets. Furthermore, TrajEvolve reformulates step-level continuous credit assignment as a binary classification task, whether each step is necessary for task completion. This simplification not only reduces the learning difficulty but also enables step labels to be obtained through counterfactual verification, achieving precise identification of redundant steps. Extensive experiments on two datasets demonstrate the superiority of the proposed method. 

---
# Dancing Diffusion: Learning Continuous Programs for Image Generation and Editing 

**Authors**: Cui Enhe, Jinyue Yang, Bojian Yin, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28003)  

**Abstract**: Diffusion models generate samples through a multi-step denoising trajectory, yet condition every step on the same instruction, so the trajectory cannot be addressed at the level of individual operations. We introduce Dancing Diffusion, which assigns each span of timesteps a dedicated continuous condition block and thereby turns conditioning into a latent program. A compiler maps text features into this program, and a fixed pretrained generator executes its blocks over successive sampling intervals. The trajectory can then dance, keeping its executed history while its continuation is redirected. A shared prefix is executed once and reused, and a later operation is rewritten while content fixed by earlier operations is preserved. The compiler is trained through the images it produces, with visual feedback on both the requested change and the retained content. On SANA, reusing an executed prefix preserves markedly more of the scene than regenerating from noise at a similar strength of edit, learned conditions preserve more background than native prompt switching from the same saved state, and the learned program improves generation quality over native SANA. Later branches weaken the edit, which leaves semantic control over individual operations limited. By converting a single-instruction trajectory into a program of addressable operations, Dancing Diffusion lays a foundation for structured intermediate computation in diffusion models. 

---
# Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation 

**Authors**: Anil Srirangapatna Nagesh, Nitin Gupta, Anuj Taneja, Vivek Singh, Shane Eleniak, Theo Zhao, Moulidharan N, Shashank Keshetty  

**Link**: [PDF](https://arxiv.org/pdf/2610.27993)  

**Abstract**: Telecom language models must combine broad networking knowledge with specialist behavior across routing, broadband services, access technologies, and operational diagnostics while remaining practical to deploy as a single model. We present an end-to-end framework that builds source-grounded synthetic training data from structured networking taxonomies and expert graphs, materializes 212,559 routed training records across fourteen networking routes, trains six frozen specialist LoRA teachers over one shared continued-pretrained backbone, and consolidates their behavior into one deployable student adapter using Routed Specialist On-Policy Distillation (RSOD). In RSOD, the committed student generates each trajectory, the route-assigned specialist teacher-forces the exact native token sequence, compact teacher hidden states are reconstructed through the shared frozen output head, and only the student LoRA is updated. A controlled reconstruction validation shows mean absolute logit error of 7.26 x 10^-6 with 100% top-1 agreement between native and reconstructed teacher logits. On four RTX 6000 Ada GPUs, the completed 2,000-step run produced 16,000 trajectories with zero observed student-version or token- ID mismatches. Under the primary route-conditioned evaluation over 2,850 held-out questions, the raw Meta-Llama-3.1-8B-Instruct model scores 12.8661/25 micro (13.0172/25 macro), the combined routed six-teacher system scores 21.7641/25 micro, and the consolidated RSOD student, released as Llama-CalixCLM v1, reaches 21.7112/25 micro (21.8006/25 macro) on the same five dimension LLM-judge rubric. Under a separate neutral deployment-like prompt, the same student reaches 20.0028/25 micro (20.0778/25 macro). The consolidated student remains within 0.0529 micro-score points of the combined routed-teacher system while retaining a numerically validated and auditable train-serve execution path. 

---
# When Do Humans and Machines Truly Complement Each Other? Formal Conditions and Empirical Evidence for Human-AI Synergy 

**Authors**: Andrea Campagner, Federico Cabitza, Vincenzo Crupi  

**Link**: [PDF](https://arxiv.org/pdf/2610.27965)  

**Abstract**: Can humans and artificial intelligence cooperate such that their combination yields greater accuracy than either agent alone? This result, termed \emph{synergy} or \emph{complementarity}, is notoriously difficult to achieve, even when both agents possess significant independent accuracy. In this paper, we present a formal framework to model human-AI interaction through the lens of reliance patterns, which characterize the joint correctness structure of human and algorithmic decisions. We derive mathematically necessary and sufficient conditions for synergy, demonstrating that its emergence depends on the ability of either agent to identify and correct the errors of the other, what we call Human and Algorithmic Correction. Beyond its theoretical contribution, the proposed framework provides a foundation for the design and evaluation of knowledge-based decision support systems, where interaction policies must dynamically integrate heterogeneous sources of expertise. We demonstrate this through the analysis of seven clinical case studies. Our empirical results reveal a ``dominance trap,'' where a strong algorithmic advantage can paradoxically inhibit synergy by eroding human corrective intervention. We also show that synergy can persist even in the presence of cognitive biases, such as automation bias, provided there is functional de-correlation between human and AI errors. These results provide a formal foundation for key concepts in the human-AI systems literature and offer actionable insights for designing interaction-centric AI systems that foster complementarity. 

---
# Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay 

**Authors**: Byeonghun Bang, Sangwoo Park, Jinheon Baek, Kangsan Kim, Youngwan Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27954)  

**Abstract**: Recent advances in large (vision) language models have enabled computer-use agents that automate digital workflows via fine-grained keyboard and mouse operations. However, such agents remain fragile in long-horizon settings, where errors accumulate and executed actions leave unintended side effects, making the ability to recover from failures critical. Yet, existing remedies remain inadequate: environment resets are infeasible in proprietary settings, action replay is prohibitively expensive over long horizons, and API-based rollback is limited to web environments, leaving model-based recovery largely unexplored. To address this gap, we present ReTrace, a scalable evaluation framework for systematically studying recovery behavior in computer-use agents. We formalize Transition-Consistent Recovery: identical downstream actions from the recovered and target states must yield identical outcomes. We operationalize this criterion through goal-oriented action replay following controlled error injection. Using ReTrace, we present the first systematic study of autonomous, model-based recovery, revealing that (i) ideal recovery can unlock model's latent task completion capabilities, (ii) current computer-use agents fail to exploit their latent recovery capabilities, and (iii) accurate model-based recovery yields measurable end-to-end gains. Together, these findings establish recovery as a fundamental capability for building reliable computer-use agents. Our code and dataset is available at this https URL. 

---
# HypTI: Hyperbolic Geometry-Aware Multimodal Feature Fusion Framework for Dementia Detection Based on Tabular-Image Data 

**Authors**: Jianfei Xu, Huizhi Liang, Judith Harrison, John-Paul Taylor, Jaume Bacardit, Varun Ojha  

**Link**: [PDF](https://arxiv.org/pdf/2610.27941)  

**Abstract**: AI-assisted dementia detection usually requires multimodal fusion of diverse data, such as clinical tabular assessments and neuroimaging data. However, most existing approaches perform feature fusion in Euclidean space. This overlooks the geometric structure of representation, including hierarchical organization in clinical assessments and multi-scale anatomical patterns in neuroimaging data. Recent studies have proved hyperbolic space provides a natural geometry for modelling hierarchical and multi-scale structures due to its exponentially expanding space. Motivated by these, we propose HypTI, a hyperbolic geometry-aware multimodal fusion framework for dementia detection based on clinical tabular assessments and MRI scans. HypTI extends hyperbolic representation learning to an underexplored tabular-MRI setting, jointly modelling clinical tabular, MRI scans, and cross-modal interactions in hyperbolic space. Specifically, clinical tabular variables are represented through learnable Lorentz prototypes to capture hierarchical relationships among assessments, while MRI scans are encoded using ROI-guided multi-scale representation learning to capture anatomical patterns across brain regions and spatial scales in Lorentz space. The multimodal representations are integrated through an adaptive fusion module generates subject-specific scale and shift coefficients to dynamically modulate and integrate the multimodal embeddings, enabling effective patient-level tabular-image representation learning for dementia detection. Experiments on OASIS-3 dataset show that the proposed framework achieves strong performance compared with representative approaches. 

---
# From Comparison to Selection: Enhancing Cross-Scale Generalization of Pointer Networks Trained with Reinforcement Learning for Capacitated Drone Delivery Routing 

**Authors**: Sofiene Lassoued, Devara Tertia, Stefan Lier, Andreas Schwung  

**Link**: [PDF](https://arxiv.org/pdf/2610.27936)  

**Abstract**: Unmanned aerial vehicles (UAVs) enable fast, flexible last-mile and emergency delivery, but their routing remains challenging as the number of clients varies. Classical approaches based on mixed-integer programming and heuristics can scale poorly or require problem-specific design, motivating learning-based alternatives. Reinforcement learning is particularly promising, provided a single policy can generalize across problem sizes without retraining. Although attention-based Pointer Networks are inherently size-agnostic, their solution quality degrades under both upscaling and downscaling. Using the Capacitated Vehicle Routing Problem (CVRP) as a controlled proxy, with fixed capacity reflecting drone payload constraints, we investigate this degradation and identify three contributing factors: shifts in node density, attention dispersion, and dilution of the mean-pooled graph context. We systematically compare architecture-wide, encoder-side, and decoder-side modifications to address these limitations and propose a learned, permutation-invariant graph-context aggregation mechanism, implemented as a weighted average and as Set2Set. Results reveal distinct requirements for transfer in each direction: entropy-based attention scaling reduces the CVRP20-to-CVRP100 performance gap by 1.41 percentage points, while the proposed aggregation methods achieve the largest improvement on the more challenging CVRP100-to-CVRP20 transfer, approximately 0.9 percentage points. Exploiting this complementarity, we introduce an approach selector that chooses the best model for each training-target size pair. The selector reduces cost relative to the reference model across all nine seen train-test pairs and eight of nine pairs involving unseen sizes (CVRP35, CVRP75, and CVRP200), achieving improvements of up to 3.2 percent. 

---
# Quantum-JEPA: Quantum Born Priors for Latent-Action JEPA World Models 

**Authors**: Zhiyuan Liu, Sicheng Liu, Haiyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27921)  

**Abstract**: A single observation can admit several plausible futures, making uncertainty an essential part of world modelling. However, quantum generative models remain underexplored for this task. We introduce Quantum-JEPA, a latent-action JEPA framework that learns distributions of future representations using a conditional quantum Born prior. Circuit measurements generate binary latent actions, which a shared predictor maps to alternative future representations. We train the quantum prior through the quality and diversity of these predictions using a sampled energy objective. Our model learns distinct continuations on constructed multimodal tasks and captures latent-action correlations that support coordinated multi-agent futures. On D3IL, a benchmark of diverse human demonstrations, our trained models generate multiple obstacle-crossing routes from a common starting point without observed motion. On the controlled tasks, our classical and quantum prior implementations improve prediction validity and distributional fidelity over the evaluated variational baselines. Our quantum priors combine competitive predictive quality with a parameter-efficient sampling structure: they avoid an exponentially sized categorical output layer while supporting correlated latent actions. We further validate finite-shot parameter-shift training, including training with readout noise. Unlike most prior work, which focuses on simulation, we also verify the feasibility of inference with trained priors on a real QPU. These results establish a trainable quantum component for JEPA future prediction and a basis for exploring quantum-enhanced world models at larger scales. 

---
# ACE-Compression: An Ability-Centric Evaluation of Compressed Large Language Models 

**Authors**: Jiahui Ning, Liang Chen, Yuchen Liang, Yi Zhang, Fenglei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27913)  

**Abstract**: Model compression makes large language models deployable, but today's evaluation tells an incomplete story. Standard practice measures retained task performance and stops there. This misses a critical fact: compression perturbs the distributed knowledge embedded across billions of weights, and task scores alone cannot reveal which underlying abilities survive and which quietly collapse. To solve this question, we introduce ACE-Compression, an ability-centric evaluation framework. Instead of asking only "does the compressed model still answer correctly?", we ask "which of its core abilities remain intact?" Our framework probes four deployment-facing dimensions: procedural reasoning and constraint execution, knowledge grounding and truthfulness, social cognition and affective understanding, and persona fidelity and stylistic consistency. Across more than 35 compressed checkpoints from six open-weight LLMs-covering quantization and pruning-we find that compression redistributes ability profiles. Coarse, local, and surface-level signals often persist, while fine-grained, long-tail, and context-dependent abilities become fragile. Quantization consistently outperforms pruning in retention, but the practical meaning of that gap depends entirely on which ability an application requires. Based on our findings, deployers can choose compression methods by the ability profile their application needs, not just by a leaderboard average. And algorithm designers gain diagnostic signals to build compression methods that preserve what matters. We believe that ACE-Compression turns compression from a black-box accuracy drop into an interpretable intervention-one we can measure, understand, and ultimately improve. 

---
# MMEnterprise-Bench: Evaluating Enterprise Agents over Evolving Multimodal Records 

**Authors**: Seohyun Lee, Jongha Kim, Dohwan Ko, Ji Soo Lee, Seoung Choi, Jayun Ku, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27911)  

**Abstract**: Enterprise agents answer questions over workplace records that evolve through team collaboration. As information is shared, corrected, and revised across documents, chats, and figures, multiple versions of the same information accumulate in the workspace. Reliable question answering therefore requires agents to reason over sufficient evidence, resolve cross-source and temporal conflicts, and recognize when the available evidence is insufficient to establish an answer. To systematically evaluate these capabilities, we introduce MMEnterprise-Bench, a benchmark comprising 1,613 questions grounded in evolving multimodal records from simulated collaboration projects, covering three categories: Evidence Reasoning, Conflict Resolution, and Unanswerability. We evaluate 10 proprietary and open-source multimodal LLMs and find that even the strongest models struggle when evidence contains cross-source or temporal conflicts or spans multiple images. Moreover, increasing the retrieval budget or employing a larger retriever improves evidence recall but does not consistently translate into higher answer accuracy. These findings highlight the persistent challenges of reasoning over evolving workplace records and establish MMEnterprise-Bench as a testbed for developing more reliable enterprise agents. Code and data are available at this https URL. 

---
# Staged Continual Adaptation of Multimodal Foundation Models for Japanese Financial Documents 

**Authors**: Genshin Kakimoto, Atsushi Yanagisawa  

**Link**: [PDF](https://arxiv.org/pdf/2610.27904)  

**Abstract**: Staged post-training of multimodal foundation models chains vision-language alignment, reasoning distillation, and domain tuning, yet how each phase trades capabilities against the others remains under-characterised. Tracking an 8.4B-parameter model from an un-fine-tuned baseline through three training phases on Japanese financial documents (evaluated on five text-input benchmarks), we find the phases interact non-additively: math-only distillation overwrites the financial knowledge acquired earlier even as procedural math skill survives, and the domain phase repairs more than it installs. A skip-phase ablation is best for domain knowledge but forfeits numerical-reasoning gains, so per-phase contributions are not decomposable and the best checkpoint is task-dependent, with implications for checkpoint retention and scheduling rather than added stages. 

---
# Automatic Differentiation with Constraints in Forward and Inverse Problems 

**Authors**: Nikolay Yavich, Alexander Ryabov, Viacheslav Naumov, Sayan Ranu, N. M. Anoop Krishnan, Evgeny Burnaev, Vladimir Vanovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2610.27888)  

**Abstract**: Approximate solutions of partial differential equations (PDEs) on unstructured grids can be represented on a graph. Numerical optimization involving stiff evolutionary and static PDEs often requires solving a large system of equations, which can be treated as a constraint. In this work, we present a fully differentiable graph-based implicit solver that incorporates this constraint. We evaluate its performance across four key tasks: a) linear and nonlinear forward modeling of evolutionary and static problems, b) solving inverse problems to estimate equation coefficients, c) solving inverse problems to estimate the right-hand side, and d) learnable graph coarsening to accelerate forward modeling. The stability and differentiability of our solver enable significant reductions in the complexity of forward modeling while efficiently addressing related inverse problems. These capabilities make it a powerful tool for different physics and engineering applications, in particular for geophysics, where inverse problems are commonly solved in order to reduce the model uncertainty. 

---
# Physical Operating-Condition-Guided Vision-Language Matching for Cavitation Diagnosis in Industrial Pipeline Systems 

**Authors**: Yu Sha, Junqi Tao, Deyu Wang, Horst Stoecker, Domagoj Vnucec, Andreas Widlf, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27860)  

**Abstract**: Cavitation diagnosis in industrial pipeline systems is challenging because acoustic responses vary substantially with operating conditions, while similar time-frequency patterns may correspond to different cavitation states. To address this issue, we propose a physical operating-condition-guided vision-language matching (POC-VLM) for cavitation diagnosis. POC-VLM first constructs physical operating-condition-aware semantic descriptions by explicitly incorporating sample-specific operating variables into candidate cavitation texts. A time-frequency-conditioned text adaptation module then dynamically aligns these physical semantics with the observed acoustic response, followed by global-local vision-language matching to capture both holistic and localized discriminative patterns. In parallel, a K-nearest neighbours-based non-parametric physical operating-condition prior estimates the physical plausibility of each candidate state and is fused with the vision-language semantic evidence at the decision level. Experiments on three real-world cavitation datasets show that POC-VLM consistently outperforms representative data-driven, multimodal and domain-specific diagnostic methods. Further analyses verify the complementary roles of vision-language semantic matching and physical operating-condition priors, while demonstrating robust performance under varying sampling rates, noise levels and observation durations. 

---
# SAPD: Learning from Agent Experience without Environment Interaction 

**Authors**: Juncheng Diao, Haibin Wang, Wei Shi, Yifan Ding, Qingbin Li, Rongxiang Weng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.27847)  

**Abstract**: Scaling the training of Large Language Model (LLM) agents typically requires a large collection of executable environments, whose construction, maintenance, and operation constrain the expansion of training experience. Existing offline trajectories offer an opportunity to reuse interaction experience, yet effectively refining the current policy without revisiting these environments remains challenging: supervised fine-tuning (SFT) can lead to catastrophic forgetting, while online reinforcement learning requires real-time interaction with the environment. To address this challenge, we propose Step-Advantage Policy Distillation (SAPD), a method for learning from offline trajectories using relative step advantages. SAPD decomposes multi-turn trajectories into individual decision contexts for on-policy distillation, allowing the student to generate responses conditioned on recorded histories, and introduces relative step advantage signals to stabilize training. The entire process requires neither new environment observations nor additional multi-turn interactions. Experiments on ALFWorld, WebShop, and Search-based QA show that SAPD improves performance across different teacher configurations, mitigates catastrophic forgetting compared with SFT, and runs at least $4.7\times$ faster per training step than online OPD. We will release our code soon. 

---
# COMPASS: Routing via Capability-Oriented Model Profiling with Adaptive Search, Selection, and Tool Use 

**Authors**: Amin Rezaei, Pouya Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2610.27822)  

**Abstract**: LLM routing aims to select a capable model at low cost, but specialized routers can require updates as models and tasks change. We propose COMPASS, an agentic router in which an LLM follows a routing recipe, gathers performance and cost evidence through tools, and optionally inspects candidate answers before selecting a model. Our evaluations on LLMRouterBench and DeepSWE show that agentic routing can improve accuracy and lower model-call cost relative to specialized and trained routing baselines, without routing-specific training. On LLMRouterBench, COMPASS reaches 69.85% accuracy at \$139.57, compared with RLCascadeRouter's 69.14% at \$180.22. On DeepSWE under fixed five-fold cross-validation, it reaches 78.1% at \$215, compared with 74.1% at \$500 for the best single model. COMPASS also supports routing across benchmarks and adapts to an expanded model pool by reading new evaluation records, without changing the router's weights or prompt. 

---
# Privacy-preserving inference on hosted models: a learned input-embedding transform for frozen Qwen3.8-27B 

**Authors**: Mahendra Roopa, Srikanth Roopa  

**Link**: [PDF](https://arxiv.org/pdf/2610.27817)  

**Abstract**: Every prompt sent to a hosted language model is readable by whoever runs it. We study whether the prompt can remain on the client while the host receives transformed vectors that stand in for it. A language model converts text to vectors exactly once, with a public lookup table; we perform that step locally and pass the vectors through a small trained network (55.6M parameters) before they leave the perimeter. The host runs a frozen 27B model (Qwen3.8-27B) on the transformed vectors and never receives text or token ids. We train the transform against the frozen model as a fixed judge, combining a forward-KL utility term with an auxiliary term that pushes each vector away from its original direction. On 1,490 MMLU questions held out from training, the transform costs 1.5 points of accuracy (80.7% vs. 82.3%), breaking 68 of the 1,226 questions the unmodified model answers correctly (5.5%), while 30.1% of content tokens no longer decode to their original token under a nearest-neighbour lookup. 

---
# LatBridge: Efficient and Stable Model Replacement in Latent-Communication Multi-Agent Systems 

**Authors**: Jiajun Zhu, Haobo Li, Zixin Chen, Huamin Qu, Wenshuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27813)  

**Abstract**: Multi-agent systems rely on communication to complete tasks, and latent communication passes continuous representations to reduce text generation and re-encoding. However, multi-agent systems must reconfigure agent models, whereas existing latent-communication methods are built for fixed model combinations, so their interfaces may fail after replacement. Our pilot study confirms this interface failure and identifies the requirement for interface reuse after model replacement. We present LatBridge, which connects agent models through a shared Bridge and separate interfaces under a shared latent-communication protocol. After agent model replacement, LatBridge reuses unchanged interfaces, selects matching low-rank interfaces, and combines them through the Bridge, training only missing ones with all models frozen. The Bridge processes source representations in the basic mode and constructs target input without autoregressive generation in the accelerated mode. Experiments support effective communication after model replacement, with mean basic-mode accuracies of 14.8% on MATH and 27.3% on ALFWorld next-action prediction across eight small-model configurations. Analyses examine the factors supporting task accuracy and communication efficiency. 

---
# Toward Agentic AI for the 6G Physical Layer: From Algorithm Discovery to Verified Baseband Silicon 

**Authors**: Eric Mittag, Lukas Brechtel, Maximilian Stark, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2610.27797)  

**Abstract**: 6G will be designed faster than it can be built: AI-driven search already proposes physical-layer algorithms at machine speed, while carrying a single one onto embedded baseband silicon remains specialist work. That silicon imposes hard memory and numeric limits, but also exposes cost structures a GPU lacks, and a design flow that never runs on the target sees neither. We present an agentic workflow in which an LLM coding agent implements algorithms on a commercial baseband SoC, optimizes them down to agent-written vector assembly, and validates them on silicon. The agent is guided by a knowledge base of measured platform facts, checked at every step against evaluators it is instructed never to edit, and metered per phase. We showcase the workflow on an AI-discovered orthogonal time frequency space (OTFS) detector portfolio on the open 6G4All platform. It yields twelve kernels that run cycle-exactly on silicon, their optimized versions at 18k-405k cycles (30-660 $\mu$s) per frame. It also yields a new detector shaped by the chip's measured costs that needs 13.9% fewer cycles than the port of the discovered detector while matching its error rate, with a measured edge our evaluation cannot statistically resolve, a structure that a hardware-blind search objective had no reason to find. What is not specific to OTFS is the workflow: an LLM agent exploring an implementation space under real hardware limits, with the cost of every choice measured rather than assumed. 

---
# GRADUS: Discovering and Recovering Agent Lessons from the Environment's Latent Syllabus 

**Authors**: Qing Ping, Wei Yang, Panpan Xu, Johannes Kirmayr, Lukas Stappen, Junjie Tang, Jun Huan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27781)  

**Abstract**: Training LLM agents for policy-constrained tool environments, such as an airline backend with booking, refund, and cancellation policies, requires tasks that exercise the environment's rules individually and in combination. Existing synthesis pipelines let a model propose tasks and retain only those it immediately solves, biasing training data toward easier behaviors. We introduce Gradus (Graded Rungs of Ascending Difficulty from an Underlying System), which decouples what a task should teach from the concrete task and solution used to teach it. Without seed tasks, Gradus treats a fixed environment as a latent syllabus, extracting guarded behavioral operators that capture how actions depend on current conditions and how their effects reshape subsequent decisions. It composes these operators into increasingly demanding tasks while using corpus-level feedback to target missing outcome combinations. Crucially, Gradus jointly calibrates tasks and solutions against the environment: execution feedback identifies which side to refine while preserving the intended behavioral requirements. Verification thus becomes a recovery signal rather than a terminal filter, coupling coverage-guided exploration with the preservation of difficult examples. From off-the-shelf backbones, the resulting curriculum lifts a 30B model from 59% to 89% on $\tau^2$-Bench and from 47% to 83% on CAR-Bench with an Opus 4.5 proposer, and to 76% and 62%, respectively, with an open-weight Qwen3-235B proposer. These results show the value of jointly controlling which behaviors are explored and which survive into verified training data. 

---
# RITE: Evaluating End-to-End Resilience to Silent Tool Errors in E-Commerce Agentic Systems 

**Authors**: Wenshuo Zhang, Yuquan Lin, Yushi Sun, Yishi Lin, Rui Sheng, Peng He, Han Gao, Wang Jikai, Yifeng Rong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27777)  

**Abstract**: Tool-using LLM agents increasingly answer e-commerce queries through multi-step tool interactions. The expected output of a tool call can shift between agent development and execution, allowing a plausible but incorrect value to be returned without an explicit failure signal. These silent errors may propagate through subsequent decisions into final answers. To study agents' resilience across this end-to-end process, we formulate Silent-error Resilience Task, an interactive task that evaluates whether they complete the original query after a silent error corrupts an intermediate state. We introduce RITE, a benchmark built from human-authored industrial workflows and production-log-derived inputs. To enable controlled and reproducible evaluation, we use a semantic error taxonomy to inject errors into fixed workflow prefixes, freeze recorded tool behavior, and let agents continue freely. It contains 167 matched units across nine e-commerce workflows, yielding 501 silent-error, correct-value, and explicit-failure instances. Across 10 recent LLMs, silent errors roughly halve task success and are consistently harder than explicit failures. Recognition, use of an available verification path, and integration of the resulting evidence form successive bottlenecks. We also propose a light State Checker as an intervention to test whether resilience can be improved. 

---
# CrystalAR: Autoregressive Modeling for Symmetric Crystal Generation 

**Authors**: Ziqiao Meng, Rongsheng Chen, Zhi Zheng, Nikita Kazeev, Zhenkun Wang, Andrey Ustyuzhanin, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.27762)  

**Abstract**: Deep generative modeling has achieved remarkable progress in crystal generation. Given the rich symmetries inherent in crystalline structures, recent work has sought to explicitly incorporate space-group symmetry into generative models, enabling more effective exploration of novel symmetry templates. Yet introducing symmetry templates increases the combinatorial complexity of stoichiometric relationships between atomic species and crystallographic sites, while creating a fundamental tension between flexible exploration of discrete crystallographic templates and faithful realization of these templates as continuous three-dimensional structures. To address these challenges, we propose CrystalAR, a unified autoregressive framework that models Wyckoff symmetry templates and complete crystal structures within a single generative sequence. Specifically, CrystalAR reformulates continuous coordinate generation as a symmetry-conditioned hierarchy of discrete spatial localization decisions, autoregressively refining the positions of all symmetry-inequivalent sites across resolution levels while jointly modeling their spatial configuration within each level. This discrete hierarchical formulation further enables constraint-aware search over both crystallographic templates and geometric realizations. As a result, CrystalAR retains strong symmetry-template generation capability while achieving competitive de novo generation performance and template-free crystal structure prediction (CSP) accuracy. These results suggest that symmetry-template generation can be retained alongside broadly competitive de novo generation performance and competitive CSP without reliance on symmetry templates retrieved from prior data, providing a promising alternative paradigm for symmetry-preserving crystal generation. 

---
# Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models 

**Authors**: Yingxin Jia, Hui Geng, Wenxin Yang, Mohan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27756)  

**Abstract**: Vision-language models (VLMs) are exact on small sets and drift into estimation on larger ones, a profile reminiscent of children before they master counting. Children cross the barrier with epistemic actions: pointing at each item while reciting number words, moving counted items aside, partitioning the array. We give VLMs the same actions as dumb image tools with no visual intelligence of their own in a training-free agent loop. Our central claim is about measurement, not intervention: a model that counts by marking leaves a normally hidden serial procedure on the image, and the developmental principles of counting turn those marks into an interpretable audit of serial visual reasoning: every tag is classified against point-level ground truth as on-target, duplicate, phantom, or omitted, the answer is checked against the number of tags, and order irrelevance is tested by forcing opposite traversal orders. Across 9 models, 682 images, and 7 harness conditions we find that (i) the exact-counting range is a model-specific signature spanning nearly two orders of magnitude (R80 from 3 to 150 dots); (ii) as an intervention the finger mostly fails: tags help 4 of 9 models on small-count photographs, hurt imprecise pointers, and rarely beat a verbal routine or a blind point list; what the loop buys is an auditable trace, not accuracy; (iii) as measurement it succeeds: a tag-perturbation probe shows that precise pointers read their answers off the tags, pointing precision tracks which models benefit, dense scenes fail by omission, and models rarely revise after seeing their own tags; and (iv) the capacities dissociate: models follow an instructed traversal order yet often change their answer when it reverses, and children's regular-over-scattered advantage appears only under the routine and survives lattices that defeat a rows-times-columns shortcut. 

---
# RSI Control: Can AI Models Perform Covert Sabotage in the Self-Improvement Process? 

**Authors**: Kaiwen Zhou, Udari Madhushani Sehwag  

**Link**: [PDF](https://arxiv.org/pdf/2610.27735)  

**Abstract**: Recursive self-improvement, in which frontier AI models work on training the next generation, is quickly becoming a reality. However, the safety of this process has received far less attention than it should. A misaligned model training its successor can shape what that successor knows, how it behaves, and what it wants, and the harm carries forward into the next generation. We study this risk through the lens of AI control. We introduce RSI Control Arena, a testbed in which a frontier model, acting as an autonomous R\&D agent, is given a benign main task from a stage of a successor model's training pipeline together with a covert side task to install a misaligned property in the successor. The testbed contains 16 side tasks in three classes, forming 22 task pairs with main tasks that span the pipeline from data curation to reinforcement learning. We evaluate six frontier models as agents against LLM monitors that read the transcript of the run and the agent's reasoning. We observe that every model completes a substantial share of the side tasks without degrading the main-task performance, including side tasks that make the successor itself a better saboteur. Under minimal instructions to stay covert, models already produce successful attacks that monitors fail to detect, and providing a short list of evasion tactics further reduces the detection rate. Monitors with access to agent's reasoning detects most attacks, but not all. Our results indicate that recursive self-improvement raises significant safety concerns, and that strong safety effort should be in place before models are trusted with greater responsibility of training their successors. 

---
# How Language Models Organize Task Information 

**Authors**: Kaiyang Wan, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27724)  

**Abstract**: Dense linear probes assess task readability from language model (LM) representations, but do not directly reveal how many features suffice for prediction. We measure readout sparsity with \(k^*\), the smallest feature count found by a fixed search protocol that recovers 95% of dense-probe AUROC above chance. We evaluate 19 LMs from five labs on 14 core tasks. Ten base models give the nine reliably readable tasks similar feature counts and nearly the same order. This agreement extends to the selected features themselves, which on three tasks show cross-model response correspondence beyond class-mean differences and a label-correlation-matched null. Instruction-tuned deployment selectively reduces \(k^*\), with both weights and input format contributing. In controlled synthetic experiments across 20 models, successfully learned targets admit single-feature readouts, whereas readable controls require tens of features. Reference-model \(k^*\) is associated with label requirements across eight students, following an empirical scaling of \(N^*\propto(k^*)^{0.65}\). After continual learning, saved heads can fail while linear readability remains high, and recovery requires fewer labels than initial task learning under the tested procedures. Pruning reduces linear readability, with wider pre-pruning readouts generally associated with larger losses. These findings distinguish task readability from readout sparsity and establish their complementary value for predicting learning requirements and diagnosing LM changes. 

---
# When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines 

**Authors**: Yingxin Jia, Xuanyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27714)  

**Abstract**: When does experience memory help a closed-loop LLM driving agent? Published gains cannot say, because the two factors we find to matter most, which warmup experiences the pool contains and what any injected tokens do to the prompt, are controlled in none of them. We answer with a controlled evaluation on one audited HighwayEnv testbed (1Hz, gpt-4o-mini for the decisive contrasts). We treat the warmup pool as a resampled random factor (ten disjoint pools) and add token-matched irrelevant-filler controls, oracle-certified diagnostics, caution-pricing metrics, and quantified audits of the baseline ports. The probe suite spans flat-text retrieval (a DiLu port), a single-frame grid port (RESPOND-inspired), our episodic temporal-graph memory, a retrieval-free consolidated rulebook, and a flat K-step observation-history baseline. Four findings follow, each a property of this testbed. (1) Headline memory gains belong to the warmup draw. The reported pool is the best of ten for both retrieval memories, which beat No-Memory in only one pool of ten (sign test p=0.011), and even the rulebook's apparent pool-immunity collapsed on five fresh draws. (2) No retrieval design tested exceeds token-matched filler. The collision-only merge ceiling is reached by a single 15-token caution instruction, and metrics that price caution erase the rulebook's lead. (3) Representational sophistication is nowhere robustly operative. (4) Baseline port quality silently inverts conclusions. On this testbed, a single-pool comparison without filler controls would have reported memory gains that these controls reverse. We release the testbed, audited ports, seeds, and caches. 

---
# How Fine-Tuning Objectives Shape Layer-Wise Information in LLM Hallucination Detection 

**Authors**: Kaiyang Wan, Forrest Sheng Bao, Amin Ahmad, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27711)  

**Abstract**: Under matched conditions, classifier-head fine-tuning (\textsc{cls}) paradigm consistently outperforms the LLM-as-Judge sequence-to-sequence (\textsc{s2s}) on binary hallucination detection. We trace this gap to the representations induced by the two objectives. Static analysis shows that \textsc{cls} learns a task-specific discriminative direction, pushing hidden states into a clear two-cluster geometry with stronger class separation and linearly decodable label information. By contrast, \textsc{s2s} relies on the pretrained language-modeling head to make decisions. This minimizes the pressure to reorganize representations, leaving the hidden states much closer to the pretrained backbone. Our analysis also reveals that the \textsc{cls} signal saturates before the final layer, after which upper layers form a plateau of statistically interchangeable and largely redundant representations. However, endpoint redundancy alone does not show whether these layers are dispensable during training. Further training-dynamics analysis reveals 1) a lower pretrained substrate, 2) a narrow upper-middle \textbf{InfoWindow} where task information emerges, and 3) a later-synchronizing upper plateau that mirrors this signal. This mechanism motivates us to remove layers above the InfoWindow. Truncating 31\% of the network leaves accuracy unchanged within seed noise on Qwen3-4B and 8B and cuts latency by $1.4\times$. The same depth fraction, fixed before any analysis of the target models, transfers to Llama-3.2-3B and Mistral-7B-v0.3, and the layer structure behind it reappears there and on QNLI and SST-2. We hope this helps guide efficient task-specific LLM design. 

---
# E8-QAT: Efficient 2-Bit Quantization-Aware Training with Implicit Lattice Codebooks 

**Authors**: Xinyu Lei, Hong Huang, Jiebin Zhang, Hongyu Li, Jinlong Song, Decheng Wu, Guanghua Yu, Jianchen Zhu, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27699)  

**Abstract**: Low-bit quantization reduces the memory footprint of large language models, but preserving accuracy with efficient training remains challenging. Scalar quantization encodes weights independently, while explicit-codebook vector quantization introduces costly search during training. We propose E8-QAT, a 2-bit quantization-aware training framework that uses the $E_8$ lattice as an implicit eight-dimensional codebook. It jointly quantizes weight groups through arithmetic projection without storing or searching an explicit codebook during the training forward pass. A 2-bit core encoding and sparse extension preserve the selected lattice points exactly, adding less than 0.1\% to core index storage. On Qwen3.6-35B-A3B, E8-QAT matches the performance of the full-precision model in terms of the average score across the four tasks. E8-QAT improves over the state-of-the-art baseline by 2.03\% on IFEval and by 1.51\% on LiveCodeBench, under the same data, schedule, and bit width. The 2-bit representation reduces peak memory by 80\% and supports a $4\times$ larger batch. At each representation's largest feasible batch, E8-QAT reaches $1.97\times$ the decoding throughput of BF16. 

---
# EASformer: Environment-Action-State Alignment for Industrial Generative Ranking 

**Authors**: Bohao Li, Dinglun Cao, Pengcheng Huang, Yingda Tang, Zhiting Zhao, Yi Cao, Jilei Fu, Miao Yan, Tianying Liu, Heng Chang, Bowen Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.27691)  

**Abstract**: Modern industrial ranking systems increasingly adopt generative recommendation (GR) paradigms, in which user behavior sequences and non-sequential (NS) features are tokenized and jointly modeled by Transformer-style backbones. This unified formulation provides a scalable interface for modeling long histories and heterogeneous features. However, sequence and NS representations often carry asymmetric semantics: historical representations may combine environment context with observed-action semantics, whereas candidate-side NS representations contain no observed action but incorporate richer supplementary context, such as user-item interaction features and recent behavior lists. Direct interaction between such representations can make query-key dot products compare different semantic objects, limiting meaningful cross-token interaction. To fill this gap, we propose EASformer, an Environment-Action-State disentangled architecture for industrial app ranking. EASformer constructs an explicitly aligned context environment space shared by historical and non-sequence tokens, while disentangling user actions from historical token and candidate-specific context form non-sequence tokens. Based on this aligned space, EASformer enables complementary historical action retrieval and state-conditioned action prediction. Furthermore, the disentangled design supports a Cloze-style self-supervised objective by masking historical actions and reusing the candidate prediction pathway. On an offline benchmark from a real-world industrial app marketplace, EASformer achieves an AUC of 0.89411, yielding a 0.88% relative improvement over the deployed baseline. Online A/B tests further support this advantage, with relative eCPM lifts of 1.26% and 3.94% across two real-world ranking scenarios. 

---
# Pattern Intelligence: Unsupervised, LLM-Interpreted Discovery of Systematic Underpayment and Denial Patterns in Healthcare Revenue Cycle Management 

**Authors**: Harshil Lodhiya, Alex McManus  

**Link**: [PDF](https://arxiv.org/pdf/2610.27684)  

**Abstract**: Healthcare providers lose substantial revenue to systematic payer underpayments and claim denials, yet identifying the recurring, correctable patterns behind them remains largely manual and does not scale to hospital claim volumes. We present Pattern Intelligence, a deployed system that discovers such patterns without supervision by clustering claims on their adjudication behaviour rather than their identity. Each claim is embedded in a 130-dimensional vector combining per-line expected-versus-actual payment ratios, claim adjustment reason (CARC) codes, denial and payment signals, and service composition. Payer and contract identity are deliberately excluded from the distance metric - we show they dominate the geometry and collapse clusters into trivial per-payer groups - and are reintroduced only as a post-hoc segmentation (by contract and visit type) that keeps every surfaced pattern coherent and actionable. We cluster with HDBSCAN, recover structure from the noise set via second-pass sub-clustering, and gate low-confidence memberships. A large language model translates each centroid into an analyst-ready description of dominant procedures, adjustment reasons, and failure mode. We report a recurring finding: roughly one-third to one-half of high-dollar denials, depending on tenant and segmentation, form an irreducible long tail that no density-clustering configuration or representation we tried groups - a domain property, we argue, rather than a modeling deficiency. We evaluate validity through cluster-coherence analysis and quantify impact with a recoverable-opportunity ($) metric on two production tenants (approx. 10K and 157K claims) within a six-tenant deployment. Rather than a new algorithm, the contribution is a problem formulation, a representation-design insight, and empirical findings from production. 

---
# Temporal Evidence Allocation For Longitudinal Clinical Retrieval: A Position And Technical Report 

**Authors**: Soumya Ranjan Sahoo, Divya Bharti, Sanand Sasidharan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27682)  

**Abstract**: Electronic health records are written to support care, not retrieval. Across years of oncology notes, the primary diagnosis is restated in report after report, while the events that shape a patient's course, such as a recurrence or a change in therapy, may be recorded only once. A retriever that ranks passages by similarity to the query keeps returning the repeated material and can miss these transitions, that is, it exhibits temporal redundancy across historical narratives. We argue that longitudinal clinical retrieval should be treated as allocating a limited evidence budget across time, not just ranking by relevance. We separate the task into three objectives: temporal evidence allocation, candidate reconstruction, and clinical validation. In Study A (13 clinical queries over 20 patient records, with two-stage reconstruction), the Temporal-15 configuration reaches a milestone slot recall of 0.538, compared with 0.388 for the Global-50 baseline. However, it also retrieves far more evidence (496 versus 50 indexed units per query), so the gain cannot be credited to allocation alone. When evidence is held fixed, removing prompted precedence rules raises schema-order reversals from 0.010 to 0.080. A five-patient, single-query pilot with fixed budgets finds that wider temporal coverage does not reliably yield more dated milestones. Together, these results suggest that how much a record says about a period is a poor guide to how useful that period is as evidence. Because no expert reference timelines are available, our measurements describe retrieval behavior rather than clinical accuracy. This position and technical report presents the retrieval analysis behind the Onco-CONSTRUE framework and sets out a research agenda for budget-constrained longitudinal clinical retrieval. 

---
# Reusing Multimodal Agents: Efficient Evaluation of Sound Designs in Games 

**Authors**: Thai Nguyen Van, Ibrahim Khan, Mury Fajar Dewantoro, Pratch Suntichaikul, Hai Van Pham, Ruck Thawonmas  

**Link**: [PDF](https://arxiv.org/pdf/2610.27670)  

**Abstract**: This paper introduces a novel framework for efficiently evaluating in-game sound designs using multimodal learning and reinforcement learning. Sound design significantly influences gameplay and accessibility; however, objective evaluation methods, defined here as automated assessments based on agent performance rather than human ratings, often require retraining agents from scratch for each sound design. Our approach first pre-trains a multimodal encoder that encodes visual and auditory information into a shared latent space. A reinforcement learning agent is then trained using this latent representation. We fine-tune the multimodal encoder for each sound design while keeping the trained policy fixed, allowing us to isolate the effects of sound changes on agent behavior. To evaluate the proposed method, we conducted experiments on games from multiple genres, comparing the performance of blind agents (agents using only sound input) under various sound designs. Experimental results suggest that this approach can detect differences in agent performance caused by degraded or altered sound design. Moreover, we significantly reduce the computational cost compared to retraining the agent from scratch. Experimental results suggest that this approach provides a practical and scalable solution for evaluating in-game sound designs. The source code, data and supplementary document are available: this https URL. 

---
# Inferring continuous cognitive traits from limited neuroimaging data using pairwise convolutional neural networks 

**Authors**: Marjan Rashidi, Robert Bain, Craig Stark, Jeffrey Krichmar, Mary Hegarty, Elizabeth Chrastil  

**Link**: [PDF](https://arxiv.org/pdf/2610.27668)  

**Abstract**: Individual differences in cognitive abilities vary widely, yet the structural brain features supporting these differences remain poorly understood. Traditional brain-behavior analyses rely on predefined metrics such as volume or cortical thickness and typically assume linear relationships with behavior, limiting sensitivity in small datasets. Here, we introduce a pairwise convolutional ranking framework that learns directly from whole-brain T1-weighted MRI by making comparative rather than absolute predictions, which performs well even with small samples (N=108). We tested this method with four small-scale spatial abilities tasks. Regression analysis predicted performance with high accuracy across all tasks, achieving greater predictive power over standard regression and pointwise deep learning baselines. Iterative cortical-region removal (backward elimination) revealed brain regions supporting individual performance common to all four tasks, as well as task-specific regions. These results demonstrate that pairwise learning extracts meaningful behavioral information from structural MRI, providing prediction and interpretable insight into the neuroanatomy of individual differences. 

---
# When Can LLM Reasoning Reliably Reuse Intermediate Results? 

**Authors**: Deyuan Liu, Zecheng Wang, Xuanyi Liu, Jungang Li, Yihao Hu, Mingjie Tang, Bryan Dai, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27662)  

**Abstract**: Sampling complete solutions gives large language models more chances to solve a problem, but each attempt starts again. For tasks with several subproblems, all required answers must be correct in the same attempt. Retaining intermediate results can avoid repeating correct work, but errors in those results may affect later attempts. We study when this reuse is reliable and how much computation it requires. For local retries that stop when no candidate passes within the retry budget, we give conditions for correct completion and bound the average number of attempts. We also show how checking cost and the chance of passing a check affect total computation. Experiments across eight reasoning models examine composite mathematics and arithmetic. Across four DeepSeek models, average success on the second subproblem is 36.2 percent without first-subproblem context, 57.2 percent with its answer, and 58.3 percent with its reasoning and answer. These scores measure second-subproblem coverage and include continuations made when selection fails. Arithmetic experiments show that similar first-attempt accuracies can lead to very different coverage after repeated attempts. Code is available at this https URL. 

---
# Does Multi-Agent Collaboration Pay When Agents Serve Users? Testing Agent Scaling in Dual-Control Settings 

**Authors**: Ethan Yang, Xiangcheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.27655)  

**Abstract**: Multi-agent systems (MAS) built from language models are deployed on the premise that collaboration beats a single-agent system (SAS), yet when it pays is unclear. The most systematic study to date gives a capability-saturation rule: coordination helps while the SAS is weak and hurts once it is strong, with the crossing near 45% single-agent accuracy. That rule was derived on autonomous benchmarks, where no user takes part. We test it where agents now work: in conversation with a user under a written policy, on $\tau^2$-bench, the dual-control benchmark. On its airline, retail and telecom domains we compare the stock SAS with a centralized MAS built from the same model for six models from three families, over about 7,950 simulated conversations, with one completion-token ledger per episode per system. The study was pre-registered with a public change log; the MAS is reported as built, with one validity bar unmet (its planner never saw the tool schemas). Under a fixed simulated user no cell shows a MAS benefit: the MAS is worse in 12 of the 13 valid cells and tied in the thirteenth, and the fitted gain is negative across the observed capability range (weighted mean -28 points, 95% CI [-44, -12]). On the two cells where the rule predicts a benefit, the MAS loses by 13 and 22 points, and without a budget ceiling it at best draws level at 5.5 times the SAS's token spend. The one gain appears on the cell nearest 45%, under a second simulator. Whether the gain declines with capability is unresolved (slope -0.41, 95% CI [-1.51, +0.68]), because model-by-domain effects dominate. The MAS does not detectably reduce wrongful database writes; it converts other failures into budget exhaustion and inaction. For agents that serve people, the SAS is the default; a multi-agent design must show compute-matched evidence from the setting itself. We release all trajectories, ledgers, prompts and code. 

---
# CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents 

**Authors**: Xu Hu, Haoming Li, Yuxiang Wang, Jinxin Yang, Kangshuo Li, Licheng Wang, Dingyi Kang, Jiayu Song, Tianhao Wang, Zhanhua Hu, Xu Yang, Qiannan Li, Bingzhe Li, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27645)  

**Abstract**: As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workflows with executable ground truth, collects verifier-confirmed failures from multiple policies, and turns them into verified recovery trajectories. CUA-Recovery comprises a human-verified test suite of 2,250 erroneous states over 100 tasks and a training suite of 8.6K trajectories over 377 tasks. Evaluation on the test suite shows that agents struggle both to recognize inherited errors and to complete the task. Across six agents and six takeover depths, Error Awareness Rate (EAR) reaches at most 44.0% and Pass@3 at most 25.5%. After fine-tuning Qwen3.5-35B-A3B on the training suite, its average EAR rises from 19.9% to 35.2% on held-out recovery states, and its Rubric Score goes from 5.0% to 16.1%. It also gains 15.3 points on MyPCBench and 10.8 points on Odysseys for Pass@3, showing that the training trajectories help both with recovery and with completing tasks from a clean start. Our code is available at this https URL. 

---
# SCODEL: Shared Canonical Ontology Definitions for Food Entity Linking and Ontology-Grounded Recipe Knowledge Graphs 

**Authors**: Nuhu Ibrahim, Rishi Ravikumar, Ayuba Dauda, Fuqi Xu, Robert Stevens, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2610.27637)  

**Abstract**: Identifying recipe entities and linking them to ontology concepts is fundamental for semantic recipe understanding. However, food mentions in recipes are often ambiguous, context-dependent, and lexically diverse, making ontology entity linking (EL) challenging. We propose SCODEL, a streamlined, model-agnostic approach based on a central idea: contextual food mentions and ontology concepts that refer to the same entity should learn the same canonical ontology-oriented definition. SCODEL uses these shared definitions as a semantic normalisation layer, providing complementary semantic information to existing EL models without requiring a new linking architecture. We further introduce ReciLinkD, the first benchmark for contextual food ontology entity linking in recipes. Experiments show that SCODEL consistently improves three modular EL models (ReFinED, GSM-EL, and BLINK): SCODEL x BLINK achieves the highest Hit@1 on seen concepts (88.33%), while SCODEL x GSM-EL achieves the highest Hit@1 on unseen concepts (83.15%), outperforming the strongest large language model (LLM) based EL baseline in each setting. Experiments on a biomedical ontology benchmark further demonstrate that SCODEL generalises beyond the food domain. Finally, we use SCODEL to ground food mentions at scale and construct an ontology-grounded recipe knowledge graph, demonstrating its utility for ontology-aware recipe retrieval and corpus-level food analysis. 

---
# When Does Feedback Help? Auditing Learned Voltage-Clamp Experimental Design 

**Authors**: Rayan Nyaupane, Abhiram Bhupatiraju  

**Link**: [PDF](https://arxiv.org/pdf/2610.27632)  

**Abstract**: A learned scientific controller can outperform a misspecified planner without benefiting from feedback. We audit two synthetic voltage-clamp tasks with a training-selected fixed protocol, current masking, a Bayesian reference, and an excluded target. Five recurrent policies repeat one fixed protocol in all 1,200 off-grid rollouts; masking current changes no decision. In a crossed four-hypothesis task, a fixed diagnostic pulse is followed by one learned choice. Policies then use the first current, but tie a correctly specified Bayesian controller, and a fitted threshold reproduces the learned actions without being that controller. The excluded-target prediction gain is $4.6991\times10^{-6}$ nats per sample. Feedback use and protocol quality are separate tests. These cases establish neither biological discovery nor general RL superiority. 

---
# TriGeoBench: A Multimodal Mathematical Geometry Benchmark with Figures and TikZ 

**Authors**: Jing Wang, Yitong Yang, Hao Dou, Yaomin Wu, Shuting He, Tingfei Shen, Jia Wu, Jiaqi Li, Yinglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27626)  

**Abstract**: Multimodal Large Language Models show strong vision-language capabilities, yet their geometry-centric reasoning remains insufficiently understood. Existing mathematical benchmarks typically rely on rasterized figures and cover limited geometric knowledge areas and response formats, making it difficult to assess whether models reason over geometric structures or exploit superficial visual patterns. We introduce TriGeoBench, a geometry-centric multimodal benchmark constructed from Chinese middle- and high-school mathematics examinations. TriGeoBench contains 7,000 question-answer pairs and 12,235 geometric figures with TikZ code, spanning plane geometry, solid geometry, analytic geometry, function, and vector geometry across diverse response formats. Each instance forms a {question-answer pair, image, TikZ} triplet: question-answer pairs are stored in LaTeX, images as RGB figures, and TikZ preserves geometric topology, symbolic annotations, and relational structure. We evaluate representative open- and closed-source MLLMs through image-to-TikZ and multimodal reasoning probes. Results reveal persistent fine-grained structure and image-text alignment challenges. Oracle experiments with human-curated reference TikZ suggest that accurate executable representations provide complementary geometric cues; they do not evaluate an end-to-end conversion pipeline. TriGeoBench offers a testbed for structure-aware mathematical reasoning. 

---
# From Encoding to Generation: Towards Empathetic Large Language Models 

**Authors**: Samhita Bollepally, Takashi Yamauchi, Marissa West  

**Link**: [PDF](https://arxiv.org/pdf/2610.27603)  

**Abstract**: Empathetic interaction involves two distinguishable capacities: representing another person's emotional state (cognitive empathy) and generating an appropriate response to that state (affective empathy). We apply this distinction to examine how emotion-related information is internally organized, selectively attended to, and carried through response generation in Llama-3.2-3B-Instruct across 10 emotion categories from the EmpatheticDialogues dataset. Layer-wise linear probing revealed a broad middle-layer region of peak emotion-category decodability, with the layer-11 L2 probe achieving 63.6% held-out accuracy against a 10% chance level and a 54.6% TF-IDF lexical baseline. During response generation, emotion-related user tokens received 1.89 times as much attention per token as other utterance tokens, concentrated across layers 10-15. Target-emotion alignment then decreased from a mean cosine similarity of 0.239 for input representations to 0.125 for generated responses, with positive drift in 77% of examples and the largest shifts for angry and furious. Emotion-category representation, attention allocation, and response-level alignment therefore emerge as distinguishable components of model processing that can be analyzed through a functional cognitive-affective empathy framework. These findings provide a foundation for monitoring emotion representations during generation and leveraging models' existing affective structure to support more naturally empathetic responses without relying on external prompting or explicit emotion conditioning. 

---
# Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE 

**Authors**: Ermo Hua, Xiangyu Hong, Baiting Wu, Che Jiang, Haozheng Hou, Yicheng Gu, Hanjing Wang, Qirui Zeng, Ming Zhang, Ning Ding, Qipeng Guo, Youbang Sun, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27598)  

**Abstract**: As model parameters scale up, sparsely activated Mixture-of-Experts (MoE) approaches have become essential for model scaling. However, mainstream models currently adopt hierarchical architectures, where experts in different layers, although topologically independent, exhibit substantial similarity and redundancy in their stored content. Compared to the hierarchically arranged \textbf{Local Expert} in existing mainstream models, we hypothesize that inter-layer shared \textbf{Meta Expert} hold greater potential. To this end, we conduct a multi-dimensional empirical comparison between Local and Meta Experts, and find that models incorporating Meta Experts exhibit lower parameter redundancy, together with lower cross-entropy loss and higher downstream task accuracy. With the activated quantity and the expert granularity held fixed, \textbf{our model matches the performance of Vanilla MoE using only 70--80\% of the experts}, substantially improving the model's scaling potential. Beyond removing inter-layer redundancy among experts and improving information storage efficiency, our theoretical analysis further shows that, compared with Local Experts, Meta Experts provide hidden states with more flexible residual pathways, thereby improving information retrieval efficiency. 

---
# Beyond Experience: Knowledge-Guided Evolution for NPU Kernel Generation 

**Authors**: Xi Chen, Ming Shen, Haotian Yan, Ruoyu Sun, Tianjun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27588)  

**Abstract**: LLM agents have become capable kernel engineers on mainstream GPU stacks, but they struggle on emerging accelerators such as Ascend NPUs, whose programming model is barely represented in pretraining data. Recent NPU agents address this gap largely through experience: they accumulate memory from their own generation and optimization trials and reuse it on later tasks. We argue that experience alone is not enough for kernel generation. An agent can only remember what it has tried, and it rarely tries a mechanism it does not know. As a result, such memory offers little help when a new operator calls for a different mechanism, and even familiar operators raise unseen situations as optimization proceeds. We present ASCENDEVOLVE, which grounds every step of an evolutionary kernel search in a large external knowledge base built from platform documentation, toolkit headers, and operator repositories. Because this corpus far exceeds any context window, ASCENDEVOLVE indexes it and routes each agent to a small, task-specific slice of evidence instead of the corpus itself. The retrieved knowledge seeds diverse implementation strategies, guides repair when a candidate fails, and constrains how strong candidates are recombined. On 53 Ascend C operators from CANN-Bench spanning four difficulty levels, ASCENDEVOLVE achieves an overall score of 71.39, improving over Reflexion and OpenEvolve by 11.6% and 17.2%. 

---
# DuplexLedger: Benchmarking Transactional Memory in Full-Duplex Speech Agents 

**Authors**: Yuxuan Li, Xiangchen Guan, Kunrong Li, Jin Wang, Kaiwen Luo, Yang Xiao, Yufei Guo, Zhisheng Chen, Liang Lin, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27581)  

**Abstract**: Full-duplex speech agents must update persistent memory while listening and speaking, yet it remains unclear which events should be stored, revised, or excluded. Existing benchmarks isolate interaction timing from memory outcomes, obscuring the state transitions behind a correct answer. We introduce DuplexLedger, a bilingual benchmark of 1,094 persistent sessions aligning four full-duplex scenarios with event-level Write/Update/Ignore/Retain supervision, explicit pre/post memory states, and 6,564 same-session questions and checks. Evaluating five full-duplex models reveals an interaction-memory gap: no system combines strong boundary control with coherent final-state memory, and the best final-state fact coverage is 2.01%. We propose ACT-Mem, an 8.92M-parameter sidecar comprising an event controller, a versioned transactional ledger, and a typed retriever coupled with a frozen MiniCPM-o 4.5 backbone. On 219 held-out sessions, ACT-Mem increases latest-value accuracy from 4.11% to 8.68% while reducing stale assertions and distractor adoption. It improves interaction boundaries but slows interruption stopping. These results motivate selective, auditable full-duplex memory rather than indiscriminate transcript retention. Code is available at this https URL. 

---
# The Show-Up Attack: When Subagents Win Over Their Supervisor 

**Authors**: Maosen Zhang, Jianshuo Dong, Yutong Zhang, Jie Zhang, Tianwei Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27574)  

**Abstract**: Multi-agent systems are increasingly deployed to coordinate multiple large language model (LLM) agents. However, this also creates new attack surfaces: malicious agents may have chances to exploit the coordination. In this paper, we propose and study the \textit{show-up attack}, in which a malicious subagent manipulates the supervisor into involving it in tasks beyond its assigned scope. To evaluate this risk, we propose SUA-Arena, which implements 6 attack strategies and contains 188 test cases. Then, we evaluate 6 frontier LLMs and find that all of them are vulnerable to show-up attacks. For example, GPT-5.6-Sol as a supervisor suffers an ASR of up to 0.93. Mitigation strategies such as prompting and restricted communication fail to reduce the risk consistently. These results show that multi-agent systems need explicit mechanisms for enforcing authority boundaries. 

---
# When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects 

**Authors**: Sadia Asif, Israel Antonio Rosales Laguan, Shumaila Asif, Haris Khan, Muneeb Asif, Muhammad Zeeshan Karamat  

**Link**: [PDF](https://arxiv.org/pdf/2610.27573)  

**Abstract**: Large language models (LLMs) can influence user trust and decision-making through subtle linguistic cues, yet the behavioral effects of manipulative outputs remain poorly quantified. We introduce a behavioral computing framework for measuring \textit{behavioral vulnerability}, defined as users' susceptibility to manipulative AI responses. Building on DarkPatterns-LLM benchmark, we generate controlled manipulative stimuli using Manipulation Gradient Descent (MGD), targeting biases associated with authority, urgency, and social proof. We analyze participant responses using the Multi-Stakeholder Impact Assessment Network (MSIAN) to model cross-user influence and the Temporal Harm Propagation Model (THPM) to characterize the persistence of behavioral effects over time. In a study involving 120 participants, manipulative framing produced significant changes in trust calibration and compliance ($p < 0.01$), with effects also propagating across social contexts. These findings highlight the need to extend LLM safety evaluation beyond model outputs to measurable behavioral consequences and provide a computational framework for quantifying individual, social, and temporal risks arising from manipulative AI interactions. 

---
# RaspAssistant as an Agentic Framework for Automating Adsorption Simulation Research in Porous Materials 

**Authors**: Marko Petković, Susanne Gooijer, José Manuel Vicent-Luna, Vlado Menkovski, Sofía Calero  

**Link**: [PDF](https://arxiv.org/pdf/2610.27572)  

**Abstract**: Porous-material adsorption studies require coordinated structure preparation, force-field selection, simulation, validation, and interpretation, making end-to-end automation difficult. We test whether RaspAssistant, a hierarchical multi-agent framework, can carry an adsorption study from force-field preparation through validation and downstream separation analysis. Across twelve multi-agent workflow-reproduction runs spanning four adsorption and separation benchmarks, overall scores ranged from 8.50 to 10.00 on a rubric evaluating workflow correctness, simulation validity, and analysis consistency. In a separate eight-case force-field extraction benchmark, the mean weighted score was 0.967, combining parameter agreement and RASPA readiness. In open-ended studies, the system diagnosed and refined an acetylene force field and, after validating a CO$_2$ force field against MFI experiments, screened 197 pure-silica zeolites that passed automated input checks. These results demonstrate autonomous workflow execution after task specification and input provision: RaspAssistant coordinates, validates, adapts, and documents extended adsorption-simulation workflows while exposing model sensitivity and methodological limitations for expert review. 

---
# Act on Your Behalf, Beyond Your Authorization: Benchmarking LLM Agent Authorization Alignment 

**Authors**: Zhiping Zhang, Franklin Mingzhe Li, Zihao Huang, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27548)  

**Abstract**: When an LLM agent acts on behalf of a user, it is expected to act within the authorization scope the user grants. Existing agent evaluations largely measure task completion or harmful consequences and can therefore overlook actions that violate this delegation without producing conventional safety or privacy harms. We introduce AuthorizationBench, a benchmark of authorization alignment grounded in documented real-world cases. A scalable construction pipeline produces 286 executable tasks spanning situational triggers, the presence or absence of direct harmful consequences, and ten application domains. We execute every task with four models at two reasoning-effort settings across three agent harnesses in an authorization-aware sandbox that keeps authorized and out-of-authorization paths executable and observable, yielding 6,864 trajectories. Agents act out of authorization in 19.2% of trajectories, with rates ranging from 10.5% to 26.6% across the 24 configurations. Task completion does not imply authorization. Increasing reasoning effort leaves the aggregate rate nearly unchanged, with effects varying across models. On controlled pairs, agents act out of authorization at similar rates with (17.9%) and without (20.8%) direct harmful consequences, and more often under benign pressures than malicious injections (23.5% versus 17.6%, p=0.002). Our results show that task capability, additional reasoning, and consequence-oriented safety evaluation do not by themselves ensure authorization alignment. Moreover, agents that consult the principal before completing a task exhibit lower rates of out-of-authorization behavior (14.0% versus 21.6%, p<0.001), pointing to a tension between agent autonomy and preserving users' decision authority that calls for further human-centered study. Dataset and code are available at this https URL . 

---
# Automated Multi-Agent Organization Design: From Architecture Search to Organizational Learning 

**Authors**: Shao-Jun Xia, Yuner Zhang, Huixin Zhang, Di Wang, Yifan Jiang, Liangxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27531)  

**Abstract**: Agentic system design is increasingly shifting from hand-crafted prompts and workflows toward automated search over agentic programs and interaction structures, with multi-agent systems receiving growing attention. Yet a central question remains unresolved: when does an automatically generated multi-agent structure exhibit a functionally meaningful division of responsibility? We argue that automated multi-agent design should move beyond searching for effective architectures toward identifying which responsibilities are functionally necessary, how they should be allocated, and when they should change. To explore this, we propose Automated Multi-Agent Organization Design (AMOD), which treats organization itself as a first-class design object, explicitly modeling how agent teams are composed, what functional responsibilities the agents assume, how they coordinate, and how collective decisions are made. We formulate AMOD as an execution-driven design loop. For each task, an automated designer instantiates an organization by assigning functional responsibilities and coordination rules to a set of agents. After execution, task outcomes, intermediate artifacts, and interaction traces are examined to identify which responsibilities may be missing, duplicated, or ineffective, and the organization is revised accordingly. We also propose testing discovered responsibilities by removing them, replacing them with generic roles, or changing their assignment under matched resource budgets, with complementary evaluations. AMOD provides a principled foundation for evidence-based and adaptive multi-agent organization design. 

---
# MemDev-Bench: Does Memory Help Code Agents?Benchmarking Memory Systems over Real Repository Histories 

**Authors**: Guorui Chen, Tao Liu, Zhenzhen Huang, Yidan Zhang, Liyu Jiang, Shuoyuan Wang, Hanchen Sun, Xu He, Dong Li, Yihan Hu, Jianye Hao, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27524)  

**Abstract**: Code agents increasingly work on evolving real repositories, and memory systems aim to help them reuse earlier experience. However, such experience is diverse and evolving, so it remains unclear how memory systems organize it and thereby affect code agents. Existing benchmarks cannot answer this reliably because they lack either temporally valid histories or control over prior-task relevance. In this paper, we present MemDev-Bench to evaluate how memory systems affect code agents over diverse repository histories. It pairs each task with relevance-annotated earlier tasks from the same repository to build histories with controlled relevance. We also build an automated pipeline that reliably makes these earlier tasks executable and verifiable, producing 4,869 historical tasks for 555 base tasks. Based on MemDev-Bench, our systematic evaluation of memory systems across diverse development scenarios yields three main findings. (1) Memory mainly improves fine-grained localization and editing, increasing functional correctness. (2) Memory also causes failures on otherwise solvable tasks even with related history, and unrelated history tends to increase such failures. These failures mainly stem from agents reusing memory without reassessing its applicability. (3) Compared with using memory systems, directly providing agents with raw historical trajectories substantially reduces their steps. Further analysis suggests that agents save these steps by reusing process information that compressed memories tend to omit. These results suggest that future memory systems should preserve process information and reassess memory applicability against current task context. 

---
# When Health Assistants Agree Too Readily: A Benchmark of Time-Critical Self-Diagnosis 

**Authors**: Marilyn Zhang, Tianfeng Chen, Patricia M Pechter, Mark E. Whiting  

**Link**: [PDF](https://arxiv.org/pdf/2610.27518)  

**Abstract**: Every day, more than 40 million people ask ChatGPT a health question, such as whether laptop posture explains a stiff neck and days of pounding headache. Posture is plausible, but meningitis is a time-critical possibility. We introduce SelfDx, a paired benchmark for whether models have the appropriate response behavior, a form of epistemic independence, in health questions where misplaced agreement with the user's self-diagnosis could delay urgent or emergency care. Each of 100 pairs contrasts a caution case, where a serious alternative remains possible, with an agreement case, where the user supplies evidence that rules out the alternative and it is safe for the model to agree. We find across 27 frontier models that they tend to inappropriately agree with the user's self-diagnosis in cases that require caution. Even the best current-generation model reaches only a 77.0% appropriate caution rate on time-critical condition cases. To address this gap, we experiment with three prompt variants that increase appropriate caution but reduce appropriate agreement by 9-35%, indicating excessive caution. In contrast, supervised fine-tuning on 128 balanced examples improves both rates together across three runs. We recommend post-training to improve appropriate caution without making models cautious on every case, with paired evaluation to verify both response behaviors. 

---
# Structured-Token Attention for Fine-Grained mmWave Liquid Identification 

**Authors**: Junseo Min, Joongheon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27512)  

**Abstract**: Fine-grained liquid identification requires a classifier to distinguish similar substances from compact millimeter-wave measurements. We investigate structured-token attention, which represents a radar feature tensor through complementary antenna, feature, branch, and global views. The proposed StructAttn model embeds these views into eleven tokens and combines them with a three-layer transformer. Experiments on the public FG-LiquID data cover 30 liquids and 3,000 samples. Over five seeds on the released split, StructAttn achieves 96.70\% mean best-epoch accuracy, matching the original-recipe MMNet baseline and exceeding a scalar-token attention model by 1.50 percentage points. A five-seed probability ensemble reaches 97\%, corresponding to 18 errors among 600 test samples. Final-epoch and five-fold results complement the checkpoint-selected comparison, with StructAttn obtaining 95.30\% out-of-fold accuracy. Token and feature analyses show that structured views are useful for prediction, while residual confusions concentrate in water, soft-drink, and edible-oil classes. These results support structured tokenization as a practical design choice for attention-based liquid recognition. 

---
# The More the Better? A Comparison of Single- and Multi-Agent LLM Setups on a Financial Review Task 

**Authors**: Pei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27488)  

**Abstract**: Practitioner guidance holds that a large language model reviews a document better when the work is divided among agents. We test a narrow form of it in a controlled comparison of single-agent and multi-agent setups on a financial task, reviewing credit assessment memos against requirements. One agent checks six requirements in one call. A team of three on the same model checks the same memo, two requirements each, in the same words, with no roles or later stage. Labels are known for every requirement on 300 synthetic memos, so misses and false alarms are measured separately. Two primary predictions are tested on four models at a Bonferroni-corrected level. Under an instruction to decide each requirement once, the team catches 0.08 to 0.10 more of the defects that carry an explanation on the three models whose single agent misses some, at the corrected level on two. It raises more false alarms on all four, at the corrected level on three. Balanced accuracy, an exploratory measure, changes by 0.02 or less, and the team uses 2.2 to 2.4 times the tokens. Showing each agent the requirements it was not given reduces the added false alarms on the two Claude models, at the corrected level on one. In exploratory comparisons, sampling the single agent three times and flagging on any draw gives higher balanced accuracy than the team on those models, and the team's results depend on the instruction. 

---
# Securing Agentic Commerce Through Behavior-Aware Fraud Detection and Cybersecurity for Trusted AI-Agent Payment Authorization 

**Authors**: Kayode L. Ogunsusi  

**Link**: [PDF](https://arxiv.org/pdf/2610.27483)  

**Abstract**: The growing use of autonomous artificial intelligence agents in digital commerce creates new security concerns involving agent impersonation, unauthorized payments, compromised agents, and misuse of delegated authority. This research introduces the Autonomous Commerce Agent Guard System (ACAGS), a cybersecurity framework combining supervised machine learning, behavioral analytics, agent-specific authorization controls, and explainable AI. The framework was evaluated using synthetic financial transaction data augmented with simulated agent characteristics and five adversarial scenarios. XGBoost was selected following comparisons with alternative classifiers. Across ten experimental runs, behavioral features improved mean F1-score from 52.12% to 67.49%, with agent-specific features increasing performance to 69.46%. The framework achieved detection rates of 99.77% for abnormal agent behavior, 99.73% for impersonation, and 100% for malicious delegation and authorization-scope violations. Compromised-agent detection reached 72.27%. ACAGS also supported APPROVE, REVIEW, and REJECT authorization outcomes, with SHAP explanations providing model interpretability. The results demonstrate the potential of combining behavioral intelligence and delegated authorization controls to strengthen simulated agentic-commerce payment security, while highlighting the need for validation using real-world transactions. 

---
# Runtime Control for AI Agents: Closing the Evaluation--Execution Gap in Operational Workflows 

**Authors**: Jayesh Bhadane, Ramachandra Yerramsetti, Rajesh Gupta, Sumit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.27480)  

**Abstract**: Language-model agents increasingly participate in operational workflows where approvals, binds, and payments create commitments. An approval valid when issued may no longer authorize execution after relevant facts, evidence, policy, or authority change. We call this the evaluation--execution gap. We specify a state-bound runtime assurance layer that separates probabilistic measurement from deterministic adjudication, binds an execution allowance to the decision-relevant basis of an approval, invalidates that allowance after relevant logged changes, and rechecks admissibility at dispatch. This distinguishes the continued validity of a prior approval from a current-state policy check. We describe eight failure modes, an assurance model, a reference architecture, and prospective interventions with retrospective remediation. In a scripted 42-case replay, the state-bound configuration identifies 15 later-invalidated approvals that a deliberately non-re-evaluating point-in-time comparator does not revisit; it holds 14 affected actions before dispatch and compensates one landed effect. These results illustrate behavior under the stipulated policies and event orders, not superiority over dispatch-time enforcement designs or production effectiveness. Under explicit interception, ingestion, dependency, and serialization assumptions, the model establishes dispatch-time validity within the registered execution surface. 

---
# Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement 

**Authors**: De Wang, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.27465)  

**Abstract**: Agent platforms are starting to improve their own agents: they read a deployed agent's execution traces, derive a change, have an automated verifier approve it, and release it with no human author. How much of this loop can safely run without a human? We propose two quantities a platform can measure. Verifier strength $\sigma$ is how well the available verifier -- often an LLM judge -- agrees with a trusted reference verifier; it limits how much a step can improve the agent. Recoverability is how cheaply a bad change can be undone: whether undoing it needs a new verifier approval, and how much of the old behavior a rollback restores. A step should run unattended only if both are adequate. On our platform this rule allows prompt, tool and skill edits to run unattended and keeps weight updates under human sign-off.
Our main contribution is these two definitions and how to measure them. We built a platform that applies the rule and tested it in two ways: on the platform's own approval gate, and in controlled experiments on the $\tau^2$-bench agent benchmark. On verifier strength, the platform's LLM judge sometimes gave the same score no matter what it was shown (9 of 37 settings) and reported no error, so a verifier can look healthy while approving anything; its strength must be measured, and two simple tests catch the failure. On recoverability, restoring a saved copy brought the agent back to its original behavior, but a regenerated undo did not, so only changes whose old version is kept can be reversed without a human; and while a change was deployed, it altered the outcome of 17 percent of tasks beyond normal run-to-run variation, which no undo repairs. Together these support the rule: a change can run unattended only when its old version is kept and its verifier's strength has been measured. 

---
# Ask to Distill: Learning to Synthesize Data for On-Policy Distillation 

**Authors**: Jiashuo Sun, Hao Zhang, Xueqiang Xu, Heng Wang, Yixuan Xie, Pengcheng Jiang, Zhenrui Yue, Jiajun Fan, Peiran Li, Ge Liu, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.27450)  

**Abstract**: On-policy distillation (OPD) can succeed with surprisingly few training questions, motivating a view of questions as carriers of distributional knowledge transfer. How can we assess whether a newly generated question is valuable for distillation, and learn to generate better questions---without access to its correct answer? We introduce Ask-to-Distill (\method), a framework that treats synthetic questions as distribution elicitors and optimizes the teacher supervision they expose along student-generated trajectories. A student-initialized synthesizer is optimized using a distributional surrogate for question value, computed from a frozen student--teacher pair without reference answers, correctness rewards, or per-candidate student updates. The reward combines teacher--student disagreement with teacher probability mass on the student's candidate support, together with length and diversity controls. The generated questions are filtered and deduplicated into a prompt-only bank for standard OPD. Across mathematical reasoning, code generation, and agentic tasks, our method improves average performance and teacher-gap recovery over unoptimized student-generated prompts and KL-filtered real questions, while matching or exceeding real-bank OPD in the evaluated settings. Signal analyses further show that optimized questions can elicit denser supervision and more total signal from shorter student responses, supporting elicited distributional supervision as an objective for synthetic distillation data. Our code, data and models are publicly available at this https URL. 

---
# Harness Self-Evolution for Long-Horizon Terminal Agents 

**Authors**: Ruhan Wang, Yucheng Shi, Zongxia Li, Junyao Yang, Zhongzhi Li, Yue Yu, Haitao Mi, Dongruo Zhou, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27435)  

**Abstract**: Long-horizon terminal agents may solve a task in one run yet fail across repeated runs. Stochastic errors can propagate through actions and context, creating a gap between capability and reliable execution. We present RiseHarness, a framework for iteratively improving agent harnesses with model weights fixed. It focuses on flipping tasks with both successful and failed runs under the same configuration, using successful runs as references for failure analysis. RiseHarness combines agent trajectories with internal harness traces from a runtime monitor to examine how model actions interact with command execution, feedback, context management, and completion decisions. A Harness Handbook links this evidence to harness behaviors and source code, helping identify recurring problems in mechanisms shared across tasks. RiseHarness then searches other harness frameworks for relevant implementations and adapts them as design references to the current model and execution protocol. Candidate changes undergo behavioral checks and repeated task evaluation; versions with measured gains seed the next iteration. This loop integrates suitable mechanisms from different frameworks into a single harness and uses new execution evidence to guide further changes. We evaluate task success and reliability across models, ablate harness execution evidence and external research, and test transfer to unseen benchmarks with the same model. RiseHarness treats a specific model as the unit of harness adaptation, studying how changes to shared interaction mechanisms can make existing capabilities more reliable across tasks. 

---
# KPHarness: Co-Evolving Knowledge and Probing for Long-Horizon Gaming Harnesses 

**Authors**: Zhikang Chen, Yueqing Sun, Sen Cui, Ziang Ye, Yu Wang, Yan Feng2, Xierui Song, Yu Zhang, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27417)  

**Abstract**: A failed gaming trajectory reveals that an agent needs intervention, but not necessarily how its execution harness should change. Effective harness adaptation therefore requires gathering diagnostic evidence, rather than merely proposing alternative recovery rules. We present KPHarness, a framework that adapts the external execution harness of a frozen vision-language-action model through knowledge-probing co-evolution. Recovery knowledge supplies context-specific intervention hypotheses that guide active probing; probe outcomes reveal when these hypotheses apply and provide evidence for refining both recovery knowledge and later probe selection. This reciprocal process allows the harness to improve not only which interventions it applies, but also how it investigates failures. KPHarness separates fast policy execution from a slow research loop that translates accumulated knowledge and probing evidence into bounded harness revisions. Each revision is compared with the incumbent under matched sandbox conditions and accepted only when the paired evaluation shows improvement. Recoveries supported by further validation are retained for cross-task reuse, while immediate interaction feedback and delayed episode-success credit update action- and knowledge-probing policies during adaptation. Experiments on Minecraft tasks spanning mining, combat, and crafting show improved task coverage over bare policy execution, a manually specified fixed harness, and knowledge-only harness evolution with non-adaptive probing. Evaluation on previously unseen tasks further shows improved task completion with the evolved harness, knowledge bank, and probing policies held fixed, 

---
# More Memory, Worse Agents: Error Reproduction and Anti-Persistence in LLM Agents 

**Authors**: Ruhan Wang, Kishan Panaganti, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27406)  

**Abstract**: LLM agents can improve across tasks without updating model parameters by storing experience from their own executions in external memory and reusing it to guide later decisions. A central design question is whether to retain only observed trajectories and outcomes or also the rules, summaries, and skills inferred from them. Retaining these abstractions allows agents to reuse strategies without repeatedly deriving them from raw trajectories. However, errors in stored abstractions can propagate to subsequent decisions through retrieval and reuse. They can also shape execution trajectories, which then become the basis for new abstractions that reproduce the original errors. We term this feedback mechanism error reproduction: errors are regenerated through a cycle of abstraction, execution, and re-abstraction, extending their persistence beyond direct reuse of the original faulty knowledge. We propose Anti-Persistence, a memory design that accumulates trajectories and outcomes while reconstructing task-specific guidance on demand. An adaptive selector uses prior outcomes to decide whether and how to synthesize guidance, which is discarded after execution. This prevents its direct reuse as a stored object, although indirect feedback through retained trajectories remains possible. Across different benchmarks and backbones, Anti-Persistence achieves higher overall held-out success and the lowest average LLM-call cost among the evaluated memory designs. These results support separating experience accumulation from abstraction persistence in memorybased self-improvement. 

---
# ProMemAgent: Long-Horizon Proactive Assistance with Context-Aware Prospective Memory from Sensory Perceptions 

**Authors**: Lilin Xu, Linyang He, Kaiyuan Hou, Yuang Fan, Minghui Zhao, Xuhai Xu, Xiaofan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27400)  

**Abstract**: Large language model (LLM) agents are moving beyond reactive responses toward proactive assistance that anticipates user needs without explicit requests. However, existing proactive agents primarily address immediate needs, while many everyday situations involve delayed intentions that can only be acted on hours or days later. Humans handle such intentions through prospective memory, the ability to remember to act on an intention when the right moment arises. In this paper, we introduce ProMemAgent, the first proactive agent to incorporate prospective memory over everyday sensory contexts for long-horizon proactive assistance. ProMemAgent organizes past sensory perceptions (e.g., egocentric video and audio) into context-aware memory, generating prospective cues and linking earlier events with subsequent developments for future retrieval. Given the current context, it retrieves relevant past events and performs memory-augmented proactive reasoning to determine whether assistance is needed. We further introduce CPMBench, the first benchmark for evaluating long-horizon proactive assistance over sensory contexts, comprising 1,000 instances across eight everyday categories. Extensive experiments show that ProMemAgent achieves the best memory retrieval performance and consistently outperforms all baselines on long-horizon proactive assistance. 

---
# Unified Sharp Bounds on Probabilities of Causation 

**Authors**: Xin Shu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27398)  

**Abstract**: Probabilities of causation (PoCs) have developed steadily, from the binary measures first introduced by Pearl to the non-binary setting recently studied by Shu et al. As the theory has matured, PoCs have found applications in a growing range of areas, including personalized medicine, marketing and legal attribution. Shu et al. proposed closed-form bounds for multi-valued treatments and outcomes and proved them valid, but left their sharpness in high dimensions open. We close this gap with a mathematical proof. We further show that $\PNS$, $\PSub$, $\PRep$ and $\PN$ are all special cases of a single, more general formula. Combined with their equivalence classes, this settles the question of sharp bounds for discrete PoCs from experimental and observational data: in every dimension, they are given in closed form by a single theorem. 

---
# When PTQ Calibration Helps or Hurts: Dose-Driven Collapse and Validation-Gated Abstention in Quantized Text Embedders 

**Authors**: Hyojung Han, Ryangkyung Kang, Youngeun Kim, Seung-Hun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2610.27397)  

**Abstract**: Learned post-training quantization (PTQ) calibration fits quantizer parameters on a small calibration set, and for a text embedder the natural supervision is retrieval relevance. We ask two questions that are usually conflated: how far to calibrate, and whether to calibrate at all. Across three embedders (e5-base-v2, BGE-M3, EmbeddingGemma-300m) at 4-bit weight quantization, 161 calibration runs, the outcome of a fixed procedure ranges from a gain to catastrophic degradation, and the driver is the dose of the update, measured as the functional displacement D_func of the quantized model from its uncalibrated start. On EmbeddingGemma the same learning-rate grid gives +1.47 to -71.53 points of in-domain NDCG@10 change; a label-free cosine objective collapses at the same dose (-69.52 vs -71.63), so the collapse is a property of learnable calibration rather than of retrieval supervision. Objective sensitivity is heterogeneous: on e5, ranking calibration is substantially more robust than cosine at matched displacement (+14.16 +/- 4.34 pp, 95% t-CI [+3.37, +24.96], three paired slates), whereas on BGE-M3 the ordering is directionally opposite but unresolved ([-5.18, +2.38]). Dose is the main axis but not a sufficient statistic: equal D_func reached along different paths differs by up to 1.61 pp. We therefore treat calibration as an optional intervention. A pre-registered rule that accepts a checkpoint only when a per-query paired confidence interval on a 25% calibration-validation split clears the uncalibrated baseline was run on 6 independent slates and five arms. It never selected a checkpoint below its slate's baseline (0 harmful selections in 30 decisions), retained 8 of 11 beneficial updates, and paid a mean oracle regret of 1.52 pp with one abstention that forfeited 11.13 pp, a safety-opportunity trade-off, not a free improvement. We report what remains open, including label efficiency. 

---
# Correcting Attribution Patching with a Few Exact Measurements 

**Authors**: Luyang Zhang, Minxing Zheng, Jialu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27393)  

**Abstract**: Mechanistic interpretability asks which language model components matter for a behavior, and activation patching answers by changing one component's activation and measuring its effect, at a cost growing with components and prompts. Attribution patching estimates these effects from one gradient pass, but its estimates can be far off, and verifying each one restores the cost. Can the structure of these errors reduce this cost? We propose Guided-AP, which uses a few exact measurements to reveal attribution patching's errors and correct unmeasured estimates. On a single prompt, many components' errors share one sign, so a small exact sample yields one correction for the rest. Over many prompts of one task, components are misestimated similarly, so a few measured prompts show what to verify and how to correct each component's average effect. Our analysis shows that choosing what to verify needs error size, while correction also needs sign. On a single prompt, the correction lowers error by up to 26% across three behaviors at the compute of verifying the largest estimates and also on larger models. Over many prompts, learned corrections cut the error of average effects on factual completion by 75% to 79%, repaying their cost in long analyses. 

---
# REM: What to Store, When to Recall, What to Forget in Experience Memory for Coding Agents 

**Authors**: Xinyi Lisa Chen, Pedro Herrero-Vidal, Xiaohan Chen, Xiaojing Yu, Cris Liu, Bingqing Song, Bin Xu, Suleiman A. Khan, You-Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27377)  

**Abstract**: Large language model (LLM) agents now autonomously complete long-horizon tasks, such as repository-level coding, through hundreds of think-act steps, yet the experience each run produces is rarely reused on the next. Prior memory systems concentrate on retrieval, finding the most relevant past trajectory. Retrieval cannot observe the running agent, so it cannot decide whether the agent needs an experience, at which point, or for how long. We introduce REM (Reusable Experience Memory), a framework that manages the memory lifecycle from inside the agent's think-act loop through three explicit decisions. What to store: each completed trajectory is consolidated into a phase-structured document, two orders of magnitude smaller than the trajectory. When to recall: the document is injected at the start of the task; the costlier implementation record is withheld until an in-loop gate detects the agent struggling. What to forget: the record is evicted after a single exposure. On repository-level bug fixing (SWE-ContextBench Lite), REM retrieves the related trajectory more accurately than every state-of-the-art memory system (68.7% vs. 55.6% match@1) and improves, in every replicate, end-task resolution (27.6% vs. 24.9%) at 17% lower cost. Injecting the same experience as a summary or raw trajectory, even when oracle-retrieved, does not. Ablating any of the three decisions reduces resolution. Altogether, these results demonstrate that the bottleneck in experience reuse is not retrieval fidelity but how experience is stored and governed across its lifecycle, opening a control-theoretic perspective on agent memory. 

---
# An LLM-Based Agent for Legal Compliance Assessment: Architecture, Memory, and Evaluation in a Canadian and Quebec Context 

**Authors**: Soukaina Alaoui Ismaili, Jean-Sebastien Dessureault  

**Link**: [PDF](https://arxiv.org/pdf/2610.27347)  

**Abstract**: Checking whether a text complies with a body of law is slow when done by hand and fragile when done with keyword rules. This paper presents ComplianceAgent, the legal compliance sub-agent of COMPASS, a multi-agent framework that evaluates texts along four axes: digital sovereignty, carbon impact, compliance, and ethics. The agent relies on a large language model that scores a text on a fixed rubric, grounds its judgment in a reference legal corpus, and, when the score falls at or below a critical threshold, runs a corrective chain made of self-healing rewrites, a compliance label, an explainability report, and escalation to a human reviewer. We describe what the language model does at each step, how the memory of the agent is organized, and how the same implementation can be pointed at different reference corpora. We evaluate the agent in a Canadian and Quebec context, on a corpus of thirteen legal instruments that brings together federal and provincial texts, international treaties, and municipal documents of the City of Trois-Rivières. The evaluation relies on a hand-annotated gold standard of fifteen texts and combines classification metrics, a repeated-query stability test, retrieval precision, and a composite score. On this gold standard the agent separates every conforming text from every non-conforming one, its scores remain stable under repeated queries, and the index returns relevant passages. 

---
# Modeling Authentic Research Trajectories: A Graph View of Research Ideation 

**Authors**: Ruochen Li, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27337)  

**Abstract**: The research idea in a published paper is rarely the project's initial idea. Between the two lies a closed-loop process in which researchers run experiments, derive insights, identify challenges, explore the literature, and revise the idea, with each step motivated by evidence and reasoning they can articulate. We call the path one idea takes through this process, with the evidence and reasoning behind each step, its authentic research trajectory. Although it underlies every scientific contribution, it is rarely recorded. Existing LLM-based research agents generate an idea once, refine it against reviews, literature, or their own experimental results, or search over candidate ideas and experiments. In all of them the text of the idea evolves, while the evidence and reasoning that connected one version to the next are discarded. We propose the Research Progress Graph (RPG), a representation of authentic and complete research trajectories. An RPG records the evolution of an idea as typed research states connected by cognitive transitions, each annotated with a temporal index and the reasoning behind it, which makes this evolution an explicit object of study. We ground the representation in authentic trajectories whose steps their authors justify in annotations and papers. Comparing the representations of thirty-five research agents with the RPG, we find that each is an incomplete RPG; none keeps the state that changed, the evidence that changed it, and the reasoning connecting them as one evolving record. Modeling ideation with the RPG changes how research agents should be designed, evaluated, and trained. Agents predict the next research state instead of a complete idea, ideas are evaluated as trajectories instead of final texts, and authentic trajectories serve as training data. These directions shift research agents from generating static ideas to modeling how ideas evolve. 

---
# Unsupervised Tool Evolution for Coding Agents in Data-Lake Analytics 

**Authors**: Yuqing Yang, Yurong Liu, Deqing Fu, Manav Garg, Sravan Babu Bodapati, Fatma Ozcan, Alon Y. Halevy, Jiani Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27327)  

**Abstract**: Coding agents can answer analytical questions over a data lake by discovering, cleaning, and joining the relevant data. However, they discard this work after each question and redo it from scratch for the next one. Tool evolution, which automatically builds and refines reusable tools across tasks, is a natural fit for this repetition. Since a ground-truth label for a data-lake question requires an expert to solve it first, we consider the unsupervised setting. We propose TEAL (Tool Evolution for Analytics over data Lakes), which evolves a library of tools and usage skills for a given Worker agent without labels. Two agents propose tools from complementary evidence, an Explorer from the data lake itself and a Critic from Worker trajectories. A proposed change is kept only if the Worker adopts the tool on a rerun without extra steps and any changed answer is supported by trajectory evidence. We also introduce TEAL-Q, which generates synthetic questions for tool evolution when real ones are unavailable. We evaluate on KramaBench and LakeQA with four Worker LLMs under step budgets, which cap the number of steps the Worker may take per question. With a tight budget of 10 or 15 steps, TEAL improves every Worker on both benchmarks, lifting Sonnet-5 at 15 steps by 9.3 points on KramaBench and 21.7 points on LakeQA. A tighter budget also yields better feedback for evolution. For example, Sonnet-5's LakeQA library evolved at 20 steps and deployed unchanged at 30 steps scores 68.3, above both the baseline (59.6) and a library evolved directly at 30 steps (64.2). Additionally, a library evolved on synthetic questions performs comparably to one evolved on real questions. These results show that useful tools can be evolved without labels, and hopefully inspire future work on what to evolve and how to obtain feedback in unsupervised agent evolution. 

---
# Evolving Self-tuning Benchmarks for Long-Horizon Tool Use 

**Authors**: Hyunji Alex Nam, Michael Dennis, Natasha Jaques, Minsuk Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27307)  

**Abstract**: For AI assistants to move beyond simple chat interfaces and reliably assist users with applications like Gmail, Venmo, and Slack, we need to improve their tool-use capabilities. However, existing tool use benchmarks are still limited and difficult to scale because they rely on human supervision or are too domain-specific. To make data generation scalable, we propose, TURTLE (**T**ool **U**se **R**elated **T**asks with **L**ong-Horizon **E**volution), an Unsupervised Environment Design (UED) approach to evolve existing tasks using a proposer, a solver, and a feedback generator large language model agents. The proposer is rewarded for difficult yet solvable tasks based on the solver's empirical performance, and the feedback generator summarizes the solver's execution traces for the proposer to exploit. TURTLE produces evolving, self-tuning test suites adaptable to different environment requirements, from supporting user simulators and dense document retrieval to interacting across multiple apps. Empirical results on AppWorld, $\tau$-Bench, and AutomationBench using Gemma 4 31B-it show that the evolved tasks target the desired difficulty range (e.g., a 165\% increase in the percentage of hard tasks on AppWorld), increase task diversity (in terms of reasoning and simultaneous tool interactions), and elicit a broader range of failure modes than the original tasks. 

---
# When Does an Added Planning Capability Create Operational Value? An Observable Evidence Standard 

**Authors**: Yihua Xu, Yan Lyu, Jize Li, Jiani He, You Lin, Yidan Hu, Rubing Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27305)  

**Abstract**: Industrial artificial intelligence (AI) planning systems combine predictive models, optimization logic, controls, and governance rules, but improvement in the complete policy does not show whether one addition made an independent contribution or whether that contribution justifies its burden. We address this attribution problem with an observable evidence standard that separates opportunity to act, executed-plan change, operational consequence, equivalence, uncertainty, and operating scope. Using forecast-to-plan replay on Favorita, M5, and Walmart data, we apply the standard to heterogeneous planning additions. Three limits of single-indicator evaluation emerge. First, a composite policy can conceal negligible component contribution: one control changed only 15 of 681 executed plans. Second, changing execution does not guarantee stable benefit: the evaluated multi-period capability helped in several long-lead settings but was 7.6% more costly under tight capacity. Third, indirect evidence can prioritize review but cannot complete attribution. Independent runs produced different paths in 19 of 599 cases that a rule-based check predicted would be unchanged. Although a lower-cost diagnostic matched the full-evaluation winner in only 55.0% of cases, it selected an option within the $10^{-6}$ cost-equivalence tolerance in 79.5%; the remaining cases prevented the proxy from authorizing acceptance. These experiments establish capability-level contribution evidence; the organization completes the value decision by weighing it against implementation and sustaining burden. The resulting capability evidence record links the observations to validation, deployment scope, fallback, monitoring, redesign, and retirement decisions. 

---
# Stronger artificial intelligence breakthroughs hasten their own exhaustion 

**Authors**: Beichen Lu, Honglin Bao, Qi Huang, Qingguo Meng, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2610.27296)  

**Abstract**: Progress in artificial intelligence is widely assumed to scale with effort: more researchers, papers and investment should accelerate discovery. Theories of technological growth instead locate the difficulty of discovery in the space of ideas. Here we show that neither scale nor space is enough: breakthroughs depend on diverse attention, which collective concentration can deplete. Across 36 years of computer science, we analyze 7.2 million papers, 73,089 peer reviews and 3.7 million researchers and their topic shifts. The field expanded while its attention narrowed sharply after the Transformer. Over half of topic-shifting researchers moved into AI, especially mid-career ones, and reviewer confidence fell by a full point on a 5-point scale. Each successive breakthrough, from AlexNet to large language models, raised the field's carrying capacity for ideas but was followed by steeper exhaustion of those ideas. Bigger breakthroughs capture attention more completely, depleting the diversity their successors may draw upon. Structural biology shows the same cycle after cryo-EM and AlphaFold. Simulations show how well-intentioned and widely used policies such as reciprocal reviewing empower prolific authors and further narrow what the field accepts. We call for institutions that protect diverse attention in artificial intelligence rather than reward concentration on the winning paradigm. 

---
# Vulnerability Analysis for Safe Multi-Agent Reinforcement 

**Authors**: Taozheng Zhu, Shixiong Jiang, Fanxin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27293)  

**Abstract**: Safe Multi-Agent Reinforcement Learning (SMARL) aims to learn cooperative control policies under prescribed safety constraints. Al- though existing SMARL methods focus on enforcing safety during policy optimization, their vulnerability remains largely unexplored. This paper investigates inference-time observation attacks against SMARL, where an adversary perturbs the observations of a subset of agents, thereby altering the actions produced by their policies. We propose MinMax Attack, an attack framework that jointly selects adversarial target actions for compromised agents while accounting for the Cooperative Resistance of benign agents. We further derive a bound on the induced safety cost and characterize its relation- ship with the Compromised-Agent Budget and the Perturbation Budget. Experiments across multiple safety-critical multi-agent benchmarks and representative state-of-the-art SMARL algorithms show that MinMax Attack induces substantially higher safety costs than existing attack baselines while preserving task reward, thereby reducing its detectability and achieving both attack effectiveness and stealthiness. 

---
# TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation 

**Authors**: Akash Gokul, Srijan Bansal, Yingbo Zhou, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2610.27290)  

**Abstract**: Test-case generation is critical but remains challenging for language models. Existing test-time scaling methods either isolate candidates or aggregate traces, which do not scale to long-horizon tasks. We present TEX, a collaborative test-time scaling framework based on execution-driven cross-validation. TEX parallelizes agents to generate tests and patches, cross-executing each test against every patch to form a $K \times K$ feedback matrix. Agents use their matrix slices for refinement, enabling objective validation without verbose trace sharing. On SWT-Bench Verified, TEX achieves 86.4%, establishing a new state-of-the-art and improving +9.0% over single-agent baselines. Moreover, even random selection from TEX outputs (84.4%) surpasses the previous SOTA (84.0%), showing collaborative execution feedback improves solution quality, not just best-case samples. Our results highlight execution-grounded feedback as an effective scaling mechanism for long-horizon agentic tasks. 

---
# A latent control model for realistic rodent motion 

**Authors**: Aidan Sirbu, Charles Y. Zhang, Yuanjia Yang, Bence P. Ölveczky, Talmo D. Pereira, Blake A. Richards  

**Link**: [PDF](https://arxiv.org/pdf/2610.27279)  

**Abstract**: The development of embodied models of animal behavior is a major goal of the neuro-AI community. Embodied models will enhance our ability to make direct comparisons between biological and artificial agents. To make such comparisons effective, though, we need embodied models that actually move like animals. Previous work has developed virtual animal bodies and techniques for training realistic motion through imitation. However, to date, there is no system for embodied animal modeling that produces natural movements both during trained goal-directed behavior, and during exploration driven by random generation. Here, we introduce Simulated Control via Augmented Motor Priors for Embodied Reinforcement learning (SCAMPER), a system for training an embodied virtual rat model, enabling natural movements during both goal-directed behavior and random exploration. SCAMPER uses a variational architecture with two distinct priors, a motion prior for control and a behavior prior for exploration. We show that combining both of these priors allows SCAMPER to generate realistic behavior during exploration and exploit the task structure effectively. SCAMPER can also learn from sparse rewards as a result of the diverse movements induced by the behavior prior, allowing the system to model realistic sparse-reward tasks from experimental neuroscience. Altogether, SCAMPER provides the neuro-AI community with an effective new system for training embodied models of rodents on realistic tasks. 

---
# TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences 

**Authors**: Rabimba Karanjai, Yiming Bian, Yun-Shiuan Chuang, Jenna L. Ballard, Zijie Pan, Shivani Shekhar, Yixi Li, Yirou Ge, Avinash Thangali, Richard Williamson, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2610.27256)  

**Abstract**: Analyzing merchant transaction histories requires dynamic temporal filtering, entity resolution, and aggregation over irregular event streams--operations poorly captured by static tabular or time-series forecasting benchmarks. We introduce TxnQA, an analytical QA benchmark comprising 27 task families (81 templates) alongside a 5-scenario fraud-risk reasoning suite that evaluates both verdicts and supporting evidence over synthetic, privacy-mitigated transaction data. Ground-truth answers are computed programmatically, and model performance is decomposed into format-following rate (FFR), conditional content accuracy (CA), and end-to-end correctness (E2E). Evaluating prompted proprietary baselines and six LoRA-adapted open-weight models shows that adaptation reliably enforces output schemas but leaves arithmetic bottlenecks: Gemma-4-31B-IT and Qwen3-8B attain E2E scores of 0.867 and 0.816, whereas evaluated MoE checkpoints reach only 0.632-0.713. Furthermore, on the 500-instance fraud suite, verdict accuracy frequently exceeds 80%, yet joint accuracy (requiring exact evidence attribution) collapses to at most ~23%. These findings identify schema compliance, semantic calculation, and evidence attribution as three distinct bottlenecks in transaction-sequence QA. 

---
# EGI-BENCH: An Enterprise General Intelligence Benchmark Suite 

**Authors**: Zhiwei Liu, Zhujun Lan, Jielin Qiu, Zixiang Chen, Liangwei Yang, Ming Zhu, Akshara Prabhakar, Wenting Zhao, John Emmons, Jason Wu, Shafiq Rayhan Joty, Junnan Li, Shelby Heinecke, Silvio Savarese, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27234)  

**Abstract**: Enterprise adoption of LLMs in agentic workflows is gated by evaluations that span a wide arc of abilities: CRM operations, heterogeneous-document retrieval, personal data understanding, multi-turn planning, tool use, function calling, long-horizon research, and long-context software engineering. These evaluations live today in separate repositories with divergent runners, dependencies, and scoring conventions, which makes cross-benchmark comparison difficult and obscures what a single aggregate number actually represents. We release EGI-BENCH, a unified LLM agent benchmark suite that consolidates eight enterprise facing agent benchmarks (CRMARENA, HERB, PERSONABENCH, USERBENCH, MCP-UNIVERSE, MCPEVAL, BENCHFORCE, LOCOBENCH) behind a single command-line interface with two runtime-budgeted execution modes. On top of the suite we define the Enterprise General Intelligence (EGI) Score, a six-dimensional framework (M1-M6) whose buckets are populated by fine-grained sub-metrics routed from each benchmark rather than by benchmark-level averages. We describe the suite, the sub-metric routing, and the quick/deep budgets, and report results for fourteen frontier and cost-efficient LLMs spanning the GPT, Claude, and Gemini families under quick mode. Top-line EGI ranges from 39.6 (GPT-4o mini) to 65.6 (gpt-5.5), and no single model dominates every capability axis: gpt-5.5 leads task outcome, planning, and retrieval grounding; Claude Sonnet 4.5 leads tool actioning and safety/abstention; and Claude Haiku 4.5 leads engineering quality. Code is available at this https URL. 

---
# Magnolia: A Persistent-Memory Agent Framework for Computational Chemistry 

**Authors**: Tao Jiang, Benjamin Navet, Yousouf Taghzouti, Fabien Gandon, Louis-Félix Nothias, Élise Dumont  

**Link**: [PDF](https://arxiv.org/pdf/2610.27230)  

**Abstract**: Chemistry has recently entered a high-data regime as molecular databases from high-throughput platforms or computational studies have become available. Increasing automation within the chemical sciences leads to a more digital and precision chemistry. Dedicated scientific agents are becoming central to guiding the chemist through an ever-increasing amount of data and findings, including negative ones. Here, we introduce Magnolia, a persistent-memory artificial intelligence agent framework that integrates computational protocols, from docking to quantum chemistry, lab notebooks and chat logs relying on large language models (LLMs). It relies on a three-tier memory system: session-level observations are automatically captured and distilled into structured learnings, confirmed and elevated to project-tier entries and, when human-validated, promoted to cross-project rules that actively guide subsequent work and provide direct chemical feedback. We illustrate Magnolia on four case studies: quantum-chemical evaluation of aromatic systems (COMPAS database), molecular dynamics of butane and of modified DNA, and a protein-peptide docking campaign. Across these projects, Magnolia tracked 317 runs and accumulated 116 project-tier entries, from which four cross-project rules were promoted; in controlled replays of four tasks, runs with memory passed the checks that runs without it failed (for example, ten of ten docking-configuration checks, against one to four of ten). By combining automation, reproducibility, and human-in-the-loop control, our Magnolia framework streamlines docking, molecular dynamics and quantum chemistry calculations with direct analysis and LLM-augmented feedback, offering accelerated access to computational (bio)chemistry. 

---
# Sparse RL Relaxation for LLM Reasoning 

**Authors**: Jiabin Fan, Matthias Lindemann, Zita Marinho, André FT Martins, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27220)  

**Abstract**: Recent work has explored continuous representations in place of discrete tokens during Chain-of-Thought (CoT) reasoning, often referred to as soft CoT. This is motivated by the idea that a mixture of token embeddings can represent a superposition of multiple reasoning possibilities within a single step. However, it remains challenging to train large language models to realize this potential. Existing work has explored reinforcement learning (RL)-style training for soft CoT. However, these approaches treat soft tokens as actions in a Markov decision process rather than as differentiable intermediate computation steps. Thus, they do not propagate gradients across the soft tokens and miss out on the signal of how the choice of one soft token affects the next. In this paper, we propose a sparse relaxation method for RL (called SpaRLax), which leverages the differentiability of soft-CoT reasoning over time steps and makes training more stable and effective. We adopt an entmax sparsity method that adaptively determines the degree of softness, which truncates long-tail noise at confident steps while retaining multiple plausible candidates when uncertain. Across nine reasoning benchmarks, our method achieves the highest average performance compared with baseline methods. Further analyses validate the effective use of soft-token superposition in our approach, and also demonstrate the contribution of adaptive sparsity in our approach. 

---
# LSA-PAST: Layer-Specific Adaptation and Prototype Transport for Compositional Incremental Learning 

**Authors**: Kyle Stein, Andrew Arash Mahyari, Guillermo Francia III, Eman El-Sheikh, Jiangpeng He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27203)  

**Abstract**: Class-Incremental Learning (CIL) enables models to recognize new classes over time while retaining prior knowledge, but many real-world problems are compositional, where visual concepts arise from combinations of primitives such as states (i.e., red, shiny) and objects (i.e., shoe, car). Standard CIL methods treat each composition as an independent class, ignoring shared primitives and providing no mechanism to stabilize their representations as new compositions arrive. This causes parameter updates to entangle primitives and induce drift that degrades performance on earlier compositions. Compositional Incremental Learning (composition-IL) formalizes this setting by introducing new state-object compositions across tasks. In this paper, we propose Layer-Specific Adaptation with Prototype-Anchored Adapter-Space Transport (LSA-PAST), a parameter-efficient method tailored to composition-IL. LSA integrates Low-Rank Adaptation (LoRA) into a frozen ViT with depth-specialized adapters for states, objects, and compositions. PAST then maps past, anchored prototypes into the current adapter space through a rehearsal-free affine transform fit on current-task data, enabling old prototypes to remain compatible with new adapters. Extensive experiments demonstrate that LSA-PAST achieves state-of-the-art (SOA) accuracy-forgetting trade-off performance for composition-IL. 

---
# Learned Control Variates for Rare Grid-Outage Estimation at Electric Vehicle Charging Stations 

**Authors**: Sulyman Iyanda, Mansur Arief  

**Link**: [PDF](https://arxiv.org/pdf/2610.27159)  

**Abstract**: Public charging of electric vehicles (EVs) adds a large and synchronized load to the distribution feeders. In most networks, it is rare to overload a feeder at the same time. When it does happen, the consequences can be severe: an overloaded feeder stresses its transformers and can trigger cascading outages across a city. Planners who design for high reliability therefore need to estimate very small outage probabilities. To that end, plain Monte Carlo estimation with an expensive grid simulator can require a prohibitively large number of runs for this task. To reduce the number of runs, we propose a control variate (CV) built from a learned classifier of the outage, trained on cheap observable features such as temperature and load forecasts. We show that for any binary learner, the variance reduction is an explicit function of its recall and false-positive rate. Furthermore, we show that control variate based on a Gaussian proxy of the load loses its effect as outages become rarer, while a learner with non-vanishing precision keeps removing a fixed fraction of the variance even in the extreme rare-event regime. We verify these results in numerical experiments using a feeder-overload simulator across three orders of magnitude of outage probability and obtain that even weak learner with a single feature can remove a quarter of the variance, while a boosted-tree learner is able to remove about 60%. We then apply the estimator to the feeder network of Surabaya, Indonesia, in which the Gaussian load model leaves every feeder three to seven times the one-day-in-ten-years blackout target. At the same simulation budget, planning the reserve with the learned CV and a 95% confidence limit reduces the city-wide blackout days by a third compared with Monte Carlo. These results show that learned control variates can help planners design distribution grids that stay reliable against rare outages. 

---
# MoSAIK: Motion Search with Artificial Intelligence in K-space 

**Authors**: Oscar Dabrowski, Abijith Jagannath, Frédéric Grouiller, Sébastien Courvoisier  

**Link**: [PDF](https://arxiv.org/pdf/2610.27156)  

**Abstract**: Object:
Motion artefacts remain a major limitation of routine two-dimensional T2-weighted turbo spin-echo (TSE) brain MRI, reducing diagnostic confidence and often necessitating repeated acquisitions. MoSAIK (Motion Search with AI in K-space) retrospectively estimates in-plane rigid-body motion directly from standard multi-coil TSE k-space. Without a motion-free reference, external navigators, or additional acquisitions, MoSAIK infers motion from k-space self-consistency violations and incorporates predicted parameters into model-based, motion-aware reconstruction rather than synthesizing corrected images.
Materials and Methods:
Selected complex coil data were represented as log-magnitude and phase channels. A decoupled convolutional embedding and Transformer encoder estimated phase-encoding-line-indexed rotation and two-dimensional translations. Training used motion-free clinical acquisitions with simulated motion; four held-out subjects were evaluated. Component-wise errors assessed estimation; SSIM and PSNR in silico and entropy focus criterion (EFC) in vivo assessed reconstruction.
Results:
Motion estimation errors were low for both rotation (MSE $<$ 0.2 deg$^2$) and translation (MSE $<$ 0.2 pixels$^2$), with Pearson correlations of approximately 0.96 and 0.94, respectively. In every in-vivo acquisition, more than 75% of slices showed information gain relative to uncorrected reconstructions. MoSAIK reduced visible artefacts while preserving anatomical detail.
Discussion:
Transformer-based k-space motion-trajectory estimation enables fast retrospective correction of standard clinical TSE brain MRI. 

---
# EDMA: Entropy-Driven Multimodal Answering 

**Authors**: Emanuele Mezzi, Gertjan Burghouts, Fabio Massacci, Mengyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27120)  

**Abstract**: Multimodal question answering (MMQA) is based on integrating heterogeneous data sources, selectively leveraging relevant modalities, and ignoring distractors. Existing approaches based on Multimodal Large Language Models (MLLMs) improve performance by splitting the task into intermediate steps, but do not quantify how each modality contributes to reaching the final answer. We argue that quantifying the effect that each modality has on the answering process improves process traceability and final performance. Thus, we propose Entropy-Driven Multimodal Answering (EDMA), which uses logical entropy to formalise multimodal QA as an iterative process of entropy reduction and to quantify the contribution of each modality in distinguishing correct from incorrect candidate answers. This enables the construction of quantifiable answering trajectories, where each step is associated with a measurable entropy reduction. EDMA introduces (i) modality-conditioned partitioning to estimate unimodal entropy reduction, (ii) entropy-driven multimodal fusion to capture complementary information across modalities, and (iii) entropy-based selection of the answering trajectory that minimises the most logical entropy. Evaluated across standard MMQA benchmarks, EDMA outperforms competitive prompting baselines, with average gains of 12.8 F1 and 4.6 LLM-J points over Planning Prompting and 18.6 F1 and 8.3 LLM-J points over Chain-of-Thought. Moreover, EDMA outperforms self-consistency baselines and is competitive with the multi-agent system baseline. Moreover, we show that higher entropy correlates with more false positives and, through controlled trajectory selection, demonstrate that logical entropy serves as an effective, monotonic selector for correct reasoning paths, validating entropy level as a reliable proxy for performance. 

---
# PORA: Learning When More Reasoning Pays Off for LLM Agents 

**Authors**: Bo Yan, Qi Zhang, Zhen Tan, Tianlong Chen, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27119)  

**Abstract**: LLM agents perform tasks over multiple steps and revise their actions based on feedback. As the total reasoning budget is often limited and must support future steps, the agent must choose between short and long reasoning before each action. Longer reasoning may improve the final task outcome by preventing later errors, but it also uses budget capacity that may be needed for future decisions. This creates two challenges: estimating the downstream value of long reasoning and deciding when that value is worth using the remaining budget. We propose Paired Outcome-guided Reasoning Allocation (PORA), which learns this decision by comparing short and long reasoning from the same interaction state. During training, \method{} completes the task twice from each selected state, once with a short reasoning allowance and once with a long allowance, and compares their final quality and total processing to estimate when longer reasoning is valuable. During inference, a controller weighs this estimate against learned prices on remaining-budget pressure and chooses either short or long reasoning without running both alternatives. We evaluate PORA across application, memory, code, retail, and mathematics tasks, achieving strong results across all settings. On BFCL memory, it exceeds the strongest existing method by 17.4--32.9 percentage points. Within the same agent, it matches or exceeds always using long reasoning on code and mathematics with 50.6% fewer actor tokens and 31.2% fewer model requests, respectively. 

---
# PROVISO: Provenance-Informed Action Selection for Secure LLM Agents 

**Authors**: Bo Yan, Qi Zhang, Ziwen Kan, Fnu Suya, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27115)  

**Abstract**: LLM agents act on the external world through tools and third-party skills, so a single executed action can cause irreversible harm. Existing defenses judge a proposed action without tracking whether its content comes from the user or from external sources such as tool outputs, and they judge it in isolation rather than against an available alternative, which forces a trade-off between security and utility. We present PROVISO, a controller that decides which action to commit after the agent proposes an action and before it is executed. PROVISO encodes trusted observations, external observations, the proposed action, and an alternative into provenance-typed memory, relates them through role-specific queries, and commits the action with the highest calibrated value learned from outcomes. On four agent security tasks spanning the agent lifecycle, no baseline exceeds PROVISO in both security and utility. PROVISO improves F1 over the strongest baseline by 13.84 points on SkillsBench and by 7.07 points on ToolSafe, and it reaches 88-99 F1 on three additional skill detection benchmarks. 

---
# Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs 

**Authors**: Zijian Liu, Chaoyou Fu, Yan Yang, Caifeng Shan, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27103)  

**Abstract**: Recent Omni LLMs (OLLMs) have demonstrated remarkable capabilities in joint audio-visual reasoning, yet cross-modal hallucinations remain a major challenge. Existing mitigation methods mainly operate at the output level, overlooking the entanglement of grounding evidence and learned priors within fused representations. To address this issue, we propose Omni-Woodpecker (OWP), a training-free framework that shifts hallucination mitigation from output-level manipulation to conflict-conditioned representation rebalancing. Specifically, OWP first identifies cross-modal conflicts from complementary modality-specific views. It then employs orthogonal decomposition to separate grounding evidence from competing priors within the hidden states. Finally, OWP adaptively rebalances fused representations by suppressing misleading priors and restoring ignored evidence before decoding. Extensive experiments demonstrate the effectiveness and strong cross-architecture generalization of OWP. Using Qwen2.5-Omni-7B as the backbone, OWP improves performance by 12.9 and 8.6 points on CMM and AVHBench, respectively, while consistently achieving gains of 0.67-2.25 points across four general OLLM reasoning benchmarks. Furthermore, OWP generalizes effectively to VideoLLaMA2-AV, improving performance by 4.6 and 3.9 points on the same benchmarks. Code is publicly available at this https URL 

---
# A general-purpose System One decision model for clinical records 

**Authors**: Andrew Zhang, Trinh T.L. Vuong, Sophia J. Wagner, Dongmin Bang, Tong Ding, Yuchang Su, Mary Cipperman, Christina S. Chen, Judith Lefkes, Ramin Kahidi, Ming Y. Lu, Caiwei Tian, Tianhao Luo, Jingnan Ma, Rowland Pettit, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2610.27099)  

**Abstract**: Clinical care, health-system operations and biomedical discovery depend on extracting reliable answers from patient records, yet applying diverse clinical questions across entire populations remains computationally expensive. Here we present Clinical System One (CS1), a 4-billion-parameter decision model for fast, typed judgments from clinical records, along with two agentic harnesses in which CS1 works together with a generalist language model. CS1-4B is trained on 12.4 million questions and answers from 3.6 million patients across a multi-hospital health system, combining structured-data keys, observed outcomes and language-model supervision across clinical notes, diagnostic reports and patient timelines. It answers multiple typed questions in a single forward pass without task-specific retraining, and each answer carries a probability that can route uncertain cases to human review. On 963,408 held-out patients, CS1-4B outperformed open decision models up to 27 billion parameters on 13 of 14 tasks scored independently of its training teacher. Across 21 external data sets, its aggregate performance approached that of the strongest tested 27-billion-parameter model, with 6.4-fold faster inference in a matched workload. Integrating CS1-4B with a language model agent produced cited answers over complete multi-year patient histories and screened a patient population against the eligibility criteria of 17 registered clinical trials, placing four times as many judged-eligible patients among its top 20 candidates as keyword search while using 690-fold fewer language-model tokens than a language model reading every record. These findings support a division of labour in clinical AI, with small decision models performing repeated record reading and generative models formulating questions and synthesizing evidence. 

---
# ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems 

**Authors**: Zeyu Zheng, Yiyang Liu, Yuyang Zhao, Congmin Zheng, Jiachen Zhu, Jianian Chen, Kangning Zhang, Kayue Daniel Wong, Lingyu Yang, Rong Shan, Shihao Chen, Thomas Zhu, Wentao Zhang, Xiaopeng Li, Yan Hu, Yongxi Lin, Yuxiang Chen, Zhuoying Ou, Jianghao Lin, Jeremy Avigad, Prasad Tetali, Jingbang Chen, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.27046)  

**Abstract**: As AI systems begin to resolve open research problems, reliable verification becomes increasingly important. Formal mathematics provides a trusted framework for mathematical reasoning, enabling agents to produce machine-checkable proofs while conducting mathematical exploration. Existing formal mathematics benchmarks often focus on known results or a small set of prominent open problems, which can become difficult to evaluate reliably once solutions are discovered and widely disseminated. Existing evaluation protocols are also not always aligned with how modern coding agents operate. We therefore introduce ReuF2F, a Lean 4 benchmark of 244 formalized open problems drawn from Research Experiences for Undergraduates (REU), spanning entry-level open research problems across a broad range of mathematical subfields. We expect these less prominent open problems, which tend to attract less attention and may remain unsolved for longer, to be better suited for held-out evaluation of end-to-end open problem resolution in Lean. We adopt a standard coding-agent evaluation setting, placing each problem and its logical negation into a Lean project in a Docker container. We also build Leanimum-agent on mini-SWE-agent to provide a unified harness for fair comparisons across models. Our evaluation finds that even the best-performing model resolves only ~40% of the problems, leaving substantial room for improvement on formalized open research problems. 

---
# Physical AI: Concepts, Techniques, and Challenges 

**Authors**: Weinan Zhang, Chunhua Shen, Tianfan Xue, Bin Zhao, Bingqi Jiang, Zihan Guo, Zicai Cui, Yang Li, Ming Zhou, Xinyuan Chen, Lihe Ding, Xiao Chen, Yujin Wang, Shi Guo, Junting Dong, Tao Lu, Xing Shen, Haoyu Guo, Yao Mu, Haiyong Yu, Haoxiang Ma, Fuxian Huang, Qiaojun Yu, Yunsong Zhou, Kailin Li, Zirui Wang, Yufei Xue, Weishuai Zeng, Tao Huang, Zhigang Wang, Junli Liu, Pengyuan Wu, Pingrui Zhang, Haotian Wang, Dongzhan Zhou, Jiafei Cao, Feng Yuan, Xinyi Ye, Quanli Xuan, Mulin Yu, Shangding Gu, Kaiwen Song, Dong Wang, Jiarui Li, Kehui Liu, Zhongjie Jia, Dongjiang Li, Jiacheng Bao, Haoran Yang, Zhaoqing Chen, Yiming Sun, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27045)  

**Abstract**: In recent years, physical artificial intelligence (physical AI) has experienced rapid development, capturing significant attention in academia and industry. However, the field has yet to reach consensus on its definition, technical landscape, and core challenges. To address this gap, this monograph seeks to define physical AI, present its technical landscape, and examine each key technical subfield in depth. In doing so, it establishes a unified framework for Physical AI: artificial systems that perceive, model, and act upon the physical world through continuous feedback under real-world constraints. Further, this monograph distinguishes physical AI from conventional digital AI, embodied AI, cyber-physical systems, and digital twins. It develops the foundations of physical AI across physics, control, cognition, learning, and reasoning. It reviews key methodologies, including multimodal perception, physical data and simulation, world models, embodied foundation models, motor-control systems, multi-agent coordination, and hardware-software co-design. Furthermore, it examines applications spanning robotics, manufacturing, infrastructure, scientific experimentation, and autonomous services. Particular attention is given to sim-to-real transfer, reliability, real-time execution, evaluation, safety, governance, and long-term deployment. By connecting perception, prediction, decision-making, action, and adaptation in a closed loop, this monograph identifies the technical challenges and research directions required to advance from narrow automation toward more generalizable physical AI and, ultimately, physical artificial general intelligence (Physical AGI). 

---
# M-Plan: Comprehensive Planning For LLM Agents with Memory Arrangement 

**Authors**: Ziyang Liu, Jiakang Yuan, Jiayuan Fan, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27032)  

**Abstract**: Large language model (LLM)-based agents have recently demonstrated outstanding performance in single-turn tasks, yet they continue to struggle with long-horizon tasks, particularly in environments without prior training. A key limitation of single LLMs is their difficulty in processing long contexts, which stems from weaknesses in continuous decision-making and reasoning capabilities. Although existing multi-agent systems (MAS) equipped with long-term memory bases can partially address this gap, they mostly rely on knowledge reuse or summarization, an approach that is often constrained by directly retrieval without any adaptation for the current task, while the retrieved memory may contain much history noises waiting to be preprocessed. In contrast, MAS characterized by task-specific guidance often lack long-term memory specifically designed for the guidance. To address these challenges, we propose M-Plan, a closed-loop MAS framework that integrates long-term and short-term memory. By combining global-local and past-current guidance, M-Plan enhances agents' performance on multi-turn tasks while maintaining a plug-and-play design. During task execution, agents receive task-specific milestones and insights, and are provided with local hints only when confusion arises, which form global-local guidance together. These guidance generation and utilization construct the short-term memory as well. After task completion, relevant information about the current task is optimized or reorganized before being sent to the long-term memory storage for future retrieval. The designments mentioned above build a comprehensive framework different from other individual long-term or short-term memory works. Extensive experiments on multi-turn tasks including ALFWorld, Webshop, and PDDL across diverse models demonstrate that M-Plan outperforms multiple baseline methods in both accuracy and efficiency. 

---
# Teaching AI Agent to Play Poker Strategically Through Preference Optimization 

**Authors**: Phurinat Polasa, Thanapon Thanadunpremdet, Phatthanasak Kraiduang, Saksorn Ruangtanusak  

**Link**: [PDF](https://arxiv.org/pdf/2610.27028)  

**Abstract**: A poker agent can make a legal move and still make a strategically poor one. We ask whether compact AI agents can exhibit strategic behavior under hidden information and whether preference optimization can improve beyond static and zero-shot policies. We first evaluate a static baseline and three frozen language models: SmallThinker, Qwen3, and Gemma-3 whose performance varies substantially between exact-game and six-player evaluation. We therefore propose Strategic Preference Optimization with KL-anchored Equilibrium (SPOKE), combining teacher-relative value penalties, policy anchoring, risk constraints, and structured rationale supervision, and compare it with SFT, DPO, GRPO, and GSPO. Across three runs of a SmallThinker-4B-A0.6B policy, SPOKE achieves the lowest mean exploitability in Kuhn (0.213) and Leduc (1.081) and the highest six-max top-1 agreement (0.775), while maintaining teacher KL comparable to SFT (0.522 vs 0.523). These results provide preliminary evidence that strategic preference optimization can improve strategic quality while preserving teacher-consistent behavior. 

---
# Does Multi-Agent Collaboration Reduce Hallucination in Clinical Tasks? 

**Authors**: Won Seok Jang, Harry Bhaskar Joshi, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27027)  

**Abstract**: Multi-agent LLM systems improve performance on clinical tasks, and that improvement is often read as evidence of a more trustworthy system. However, the two are not the same claim. A clinician acts on the stated reasoning and not only on the conclusion, so whether collaboration reduces hallucination, and what decides when it does, has to be measured in the reasoning too. Yet most evaluations score only the final answer. We evaluate four systems (a single-LLM baseline and three multi-agent topologies) across four models and three clinical tasks, scoring hallucination, under a factuality and faithfulness taxonomy, inside the rationale each system generates and reporting it alongside the task metric. We report that collaboration buys task performance but not lower hallucination as a rule, and no topology is reliably better than another. The effect differs by model: of the $36$ model--system--task comparisons, $12$ are large enough to resolve --- four decreases and eight increases, with no model on both sides. What sets the level, and the direction of any change, is the model rather than the topology or the number of agents. A self-correction experiment is consistent with this: shown its own flagged errors, every model except Llama-3.2-3B-Instruct reduces hallucination; the one model for which collaboration lowers hallucination gains the most, and Llama-3.2-3B-Instruct, the model whose hallucination it amplifies most, does not reduce its hallucination at all. We find that collaboration redistributes what its models can already do; it does not supply what they cannot. 

---
# SPEM: Structure-Aware Procedural Memory for Efficient Skill Use in LLM Agents 

**Authors**: Ran Zhou, Zehui Kong, Fanyu Lin, Jiaheng Liu, Zhen Zhang, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.26995)  

**Abstract**: LLM-based agents increasingly rely on reusable skills. However, scaling skill libraries makes efficient and accurate skill use difficult. Full skill loading incurs high token and runtime overhead. In contrast, flat semantic retrieval treats skills as independent text entries, ignoring dependencies, execution order, state transitions, and data flow constraints. We propose SPEM, a lightweight structure-aware procedural memory framework for efficient skill use in LLM-based agents. SPEM maintains Atomic Skill Memory for individual skills and Skill-DAG Memory for reusable compositions derived from successful executions. For a new task, SPEM reuses high-confidence Skill-DAG templates when available. Otherwise, it retrieves atomic skills and invokes Skill-GNN for state- and structure-aware skill selection. Successful executions are written back as new Skill-DAGs, allowing procedural memory to accumulate over time. Experiments on ALFWorld and SkillsBench show that SPEM achieves the best overall effectiveness--efficiency trade-off, reaching 87.1% ALFWorld success and 38.38% SkillsBench reward while reducing token and runtime costs against strong baselines. 

---
# When Planning Problems Become Hard for LLMs: Separating Structural Complexity from Semantic Interpretability 

**Authors**: Mayukh Debnath, Adeyinka Adedigba, Rammohan Mallipeddi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26964)  

**Abstract**: Large language models (LLMs) struggle with classical planning, but benchmarks rarely distinguish structural difficulty from the difficulty of interpreting symbolic vocabulary. We examine seven STRIPS domains with BFS-optimal plan lengths. For predicting structural difficulty without solving an instance, the zero-training delete-relaxation heuristic $h_{\mathrm{add}}$ outperforms learned alternatives on held-out domains (mean Spearman $|\rho|=0.931$), including ridge regression ($0.707$), object count ($0.657$), and the episodic retrieval architecture ARC ($0.400$). ARC also fails to outperform a linear model using the same features. To isolate semantic effects, we progressively rename predicates while preserving states, transitions, and optimal plans. Valid LLM planning success drops from $67.5\%$ to $40.0\%$ between $0\%$ and $100\%$ renaming, while refusals remain at $0\%$. An LLM supplied with ARC's conditioning information predicts difficulty in context better than trained ARC (mean $|\rho|=0.826$). Hybrid routing between an LLM and a symbolic planner depends more on calibration to the target LLM's outcomes than on the structural score selected. Thus, established symbolic heuristics already estimate structural difficulty well; improving hybrid planning requires modeling LLM behavior sensitive to predicate semantics rather than developing more structural predictors. 

---
# LightCRS: Lightweight Multi-Domain Conversational Recommendation System 

**Authors**: He Ma, Lan You, Qiyang Wu, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26948)  

**Abstract**: Conversational recommender systems have recently adopted vision-language models to match user dialogue intents with multimodal product information. However, extending these methods to many product domains faces a dilemma: deploying a separately adapted VLM per domain scales cost linearly with the number of domains and requires routing each dialogue to the right model, while naively merging all domains into a single VLM degrades accuracy because product characteristics vary widely across categories. We propose LightCRS, a lightweight, query-conditioned framework for unified multi-domain multimodal conversational recommendation. LightCRS precomputes multimodal product representations once offline with frozen encoders. Online, a compact 5.3M-parameter module fuses this evidence for each dialogue via query-conditioned cross-attention, whose parameters are shared across all domains, and compares candidates jointly with an inter-candidate Transformer. To enable rigorous evaluation, we construct MM-R2A-25, a 25-domain benchmark pairing natural Reddit recommendation-seeking conversations with Amazon product listings. LightCRS achieves state-of-the-art average HR@1, NDCG@5, and MRR, improving HR@1 by 35.4\% over the strongest baseline while reducing per-query latency by 266 times. Joint multi-domain training itself improves HR@1 by 15.2\% over separately trained single-domain models. 

---
# MAMath-Bench: Process-Aware and Robust Evaluation of Multimodal Multi-Agent Mathematical Reasoning 

**Authors**: Xinping Wang, Jie Ma, Rui Xing, Wensong Guo, Xiayi Zhang, Jianhao Deng, Zihao Zhou, Boqian Yang, Feijuan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.26943)  

**Abstract**: Mathematical reasoning is a fundamental capability for educational applications of Large Language Models (LLMs), where reliable problem solving requires not only correct answers but also coherent and verifiable reasoning processes. Multi-agent systems (MASs) have shown growing potential in this domain by leveraging collaboration and verification among multiple LLM agents. However, existing evaluations primarily focus on the correctness of final answers in text-only settings, offering limited insight into multimodal reasoning capabilities, the reliability of intermediate reasoning steps, and robustness against reasoning errors. To address these limitations, we introduce MAMath-Bench, a process-aware benchmark designed to evaluate the multimodal mathematical reasoning capabilities of MASs. MAMath-Bench comprises 1,003 problems spanning 11 mathematical categories, multiple difficulty levels, and several modalities, including text, images, and audio. It features fine-grained milestone annotations to evaluate systems across three dimensions: problem-solving correctness, reasoning-process reliability, and robustness to adversarial reasoning-chain perturbations. Our evaluation yields three key findings. First, multi-agent collaboration does not consistently improve final-answer accuracy over strong single LLMs. Second, collaboration can substantially enhance the coverage of intermediate reasoning, but such gains may come at the cost of increased reasoning redundancy. Third, current systems remain vulnerable to adversarial reasoning errors, particularly those involving subtle structural perturbations. The dataset is available at this https URL. 

---
# Fix, Not Just Resample: Bounded Compiler-Guided Repair in Lean 4 

**Authors**: Haotian Deng, Chris Farber, Sonal Narasimha, Samskrithi Sivakumar, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.26942)  

**Abstract**: Automated theorem provers often rely on sampling language models, but proof assistants like Lean reject each failed step with a precise error message --- information that pure sampling discards. To overcome this, we introduce a bounded, online step-level repair module. Our search samples $K$ candidate tactics at each step and repairs a failed tactic at most $R$ times. Instead of discarding failed tactics, we feed the compiler's error message to a pretrained repair model to fix individual rejected steps in place. This approach improves the solve rate over a sampling-only baseline across all tested candidate budgets ($K \in \{1,3,5,8\}$), achieving up to a $32\%$ relative gain and solving $82$ additional theorems on a $1{,}990$-theorem subset of the LeanDojo benchmark. Repair is not free: it trades extra runtime for reachable proofs. But the trade is predictable: repair pays most when sampling is constrained; its return is front-loaded, with the first repair attempt capturing $84\text{--}96\%$ of the total gain; and the gain comes almost entirely from shallow tactic-execution failures rather than structural errors such as type mismatches. This targeted repair successfully replicates on a second benchmark (Lean Workbook), nearly doubling the solve rate with zero regressions. 

---
# Industrial Agentic AI: LLM-Driven New-Generation Agents for Smart Manufacturing 

**Authors**: Lei Ren, Haiteng Wang, Shixiang Li, Jing Zhang, Yuqing Wang, Jingwei Guo, Yunfei Zhu, Yuanjun Laili, Yang Tang, Lihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26941)  

**Abstract**: Large language models (LLMs) are enabling agents to move beyond predefined workflows toward autonomous planning and execution, creating new opportunities for smart manufacturing. However, dynamic environments, complex physical constraints, and stringent reliability requirements hinder the direct deployment of general-purpose agents in smart manufacturing. This article presents a systematic overview of LLM-driven industrial agents. First, we analyze five key challenges and discuss the concept and core characteristics of industrial agents. We then establish a unified taxonomy covering application scenarios, embodiment forms, and five capability levels. Next, we propose a four-layer system architecture and detail its core modules and operational mechanisms for closed-loop task execution. Finally, we review representative applications across the industrial lifecycle and outline future research directions. This article provides a structured reference for the design, development, and application of industrial agents. 

---
# DataPraxis: Scaling Analytics Tasks via Verified Workspace Evolution 

**Authors**: Nan Jia, Wenxuan Shi, Jingqing Ruan, Zhizhao Zeng, Tianyi Zhai, Zhongda Su, Feng Hong, Zhengyu Chen, Wenhao Yan, Haojin Yang, Siqi Dai, Weipeng Zhang, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26924)  

**Abstract**: General-purpose agents are increasingly expected to perform professional analytics work, which requires sustained reasoning, iterative investigation, and reliable evidence. Training these capabilities requires diverse and challenging tasks. However, generating tasks directly from raw data often overlooks deeper questions that emerge through analysis and provides limited evidence for verification. To address these limitations, we introduce DataPraxis, an analytics task synthesis framework based on verified workspace evolution. The framework progressively enriches workspaces with validated findings and artifacts, then inherits and combines these workspaces to uncover deeper questions. It constructs tasks from accumulated findings and their dependencies, verifying analytical results both during evolution and after task assembly. Using this framework, we construct DataPraxis-5K, a dataset of 5,000 verified trajectories spanning diverse domains and output formats, to train DataPraxis-27B and DataPraxis-35B-A3B. DataPraxis-27B achieves 55.11 on DABStep and 51.9 on JobBench, matching or surpassing frontier models such as GLM-5.2. DataPraxis-5K is available at this http URL 

---
# MedPAO-Assist: Edge-Deployable, Modality-Aware Agentic AI for Radiology Report Structuring 

**Authors**: Shrish Shrinath Vaidya, Minmini Selvam, Ganapathy Krishnamurthi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26913)  

**Abstract**: The adoption of Large Language Models for radiology report structuring has been largely dominated by either proprietary cloud-based models or large open models requiring high-end GPUs for deployment, which is a critical bottleneck for clinical adoption in radiology workflows that demand data privacy and offline operation. To address this, we present MedPAO-Assist, an edge-deployable agentic framework that generates structured radiology reports directly from radiologist dictation and voice commands, requiring only $\approx3.3$ GB peak RAM and achieving an average end-to-end latency of $\approx31$ seconds on a standard laptop CPU through an asynchronous, multi-threaded execution pipeline. The system currently supports four major imaging modalities -- Chest X-ray, Chest CT, Abdomen CT, and Brain MRI. To our knowledge, it is among the earliest agentic systems to run its entire pipeline with a Small Language Model (SLM) in the 0.6B-parameter range. This multi-agent architecture, built around an Extract-Ground-Generate paradigm, performs concept extraction, SNOMED-CT grounding, and structured report generation, followed by a speech-driven human-in-the-loop correction phase, together forming a fully "no-touch" reporting workflow that requires no keyboard input, external API calls, or internet connectivity. We evaluate each pipeline stage individually -- concept extraction, ontology grounding, and concept categorization and report hardware profiling results demonstrating the system's feasibility for real-time, CPU-only clinical deployment. 

---
# Pareto-Optimal RTL Code Generation via Multi-Objective Reinforcement Learning with Large Language Models 

**Authors**: Koki Takeshita, Kenji Uebayashi, Hikaru Tomonari, Masaya Tsunokake, Koichiro Yawata, Tatsuya Sasaki, Yusuke Ohtsubo, Kota Dohi, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2610.26904)  

**Abstract**: Recent studies have proposed reinforcement learning with verifiable rewards (RLVR) methods that enable LLMs to generate high-quality RTL circuits from natural-language specifications. Existing approaches optimize circuit PPA (Power, Performance, Area) by aggregating multiple objectives into a single scalar reward, making it difficult to simultaneously generate diverse Pareto-optimal solutions across different trade-offs such as low power, low latency, and small area. However, in practical design, the relative importance of each objective varies depending on product requirements and design constraints; thus, obtaining a diverse and high-quality candidate set is beneficial in that designers can choose a preferred trade-off a posteriori. In this work, we are the first to propose a learning method for LLM-based circuit generation that optimizes, not the quality of a single circuit, but the quality and diversity of the entire circuit set obtained through multiple generations in the PPA objective space. The proposed method consists of (1) a generation-based training pipeline that reformulates coverage improvement of the entire Pareto solution set into a sequential circuit generation problem, and (2) a reward function based on Hypervolume Contribution (HVC) that measures how much a generated circuit contributes to the Pareto front. Evaluation on the RTLLM v2 benchmark shows that the proposed method achieves +13.3 percentage points in relative Hypervolume over a baseline without PPA optimization and +14.4 percentage points over a single-objective reinforcement learning method, while also demonstrating superiority in circuit diversity. 

---
# Large language model and reinforcement learning based autonomous construction and progress updates 

**Authors**: Yifei Xiao, Huasheng Lin, T.Y. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26899)  

**Abstract**: Due to the rapid growth of the global population, traditional construction methods are gradually unable to meet the current housing demand. The advancement of machine learning algorithms has encouraged researchers to explore their applications in the construction industry, leading to more intelligent and efficient building processes. In particular, the rise of large language models in recent years has provided more possibilities for the transformation from traditional construction to smart construction. In this study, a large language model and reinforcement learning based autonomous construction method is proposed. An enhanced soft actor-critic reinforcement learning algorithm is developed to plan actions for a mobile crane according to construction tasks. Robot kinematics and control algorithms are used to ensure that the actions generated by the reinforcement learning algorithm can be executed by the crane. In addition, a customized embedding model is developed to extract information from the construction environment according to user interests. Then, a large language model, ChatGPT 5, is used to update construction progress, provide safety information to control system, and generate construction progress reports. The feasibility of proposed method is verified through several full-scall retaining wall construction experiments. 

---
# Navalia: Advancing Cowork Agents through Verifiable Task Synthesis and Long-Horizon Post-Training 

**Authors**: Hengqi Liu, Ziyuan Liu, Yang Qin, Wanting Zhou, Zichuan Wang, Jiachen Liang, Xu Chu, Shaowei Chen, Zhaokai Luo, Mu Chuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26887)  

**Abstract**: Large language model (LLM) agents are evolving from conversational assistants into Cowork Agents capable of autonomous planning, tool use, and complex task execution. Financial analysis requires these agents to integrate cross-document evidence and perform multistep numerical reasoning, but expert task authoring is costly, while direct LLM synthesis struggles to produce challenging tasks with reliable answers. We introduce Navalia, a framework for Cowork Agents that combines verifiable task synthesis with long-horizon post-training. Using corporate financial reports as its data source, Navalia semantically selects supporting evidence from complete fact sets and constructs executable computation chains to generate analytical tasks with traceable evidence and actual data dependencies between steps. We further use independent LLM review and programmatic checks to verify evidence consistency and answer correctness. The verified tasks are then used for trajectory collection, cold-start supervised fine-tuning (SFT), and reinforcement learning (RL) to improve long-horizon task execution. Navalia, initialized from Qwen3.6-35B-A3B, achieves consistent gains over the base model across all five benchmarks: OfficeQA Pro, FinSearchComp, Finance Agent Benchmark, APEX-Agents-AA, and Workspace-Bench-Lite. These results demonstrate that Navalia improves evidence integration, tool use, and long-horizon task execution capabilities and generalizes to broader Cowork settings, including general office tasks. 

---
# Joints, Camera, Action! Co-Generating 3D Motion and Video for Arbitrary Rigs 

**Authors**: Noam Rotstein, Yoad Tewel, Chen Tessler, Haggai Maron, Gal Chechik, Or Litany, Yoni Kasten, Yuval Atzmon  

**Link**: [PDF](https://arxiv.org/pdf/2610.28632)  

**Abstract**: Video generation models can now depict remarkably realistic motion. Yet it is unclear how to use this motion beyond pixels: to animate a specific 3D asset and replay the animation in its 3D scene. For instance, a video model can show a monkey jumping over a crossbar, but it cannot return that jump as replayable 3D motion placed in the 3D scene. Here we present a model that directly co-generates the video and 3D motion for any supplied rig, along with a camera that registers this motion to the scene. Prior video-motion models either handle only human skeletons, often adding billions of parameters, or triangulate 3D motion from generated 2D pose maps. We instead keep 3D motion in a separate model that supports arbitrary skeletons, and connect it to a pretrained video model through a small set of cross-attention layers. This separation lets the motion model first learn how skeletons move from motion-only data, which is easier to obtain and augment than paired video. Additionally, our model learns to locate each joint in the video frames, which lets us recover the camera and place the 3D motion in the scene. To train across many skeletons, we use a generative video editor to place animated 3D assets in scenes suited to their actions, curating the edits so their known motion remains valid. Together with human and robot video, this yields nearly 200K training pairs. Our evaluations demonstrate that our model animates diverse rigs from text more faithfully than video-first pipelines, and its motion responds to the surrounding scene. It also turns video into rig motion, rig motion into video, and enables keyframe-based control of the 3D motion. Videos and additional results are available on our project page: this https URL. 

---
# GBClaw: An agentic AI framework for atomistic grain boundary research and mechanistic discovery 

**Authors**: Yazhuo Liu, Kunqing Ding, Xiaoguo Gong, Zifeng Wang, Yichen Yang, Yifeng Che, Wen Chen, Josh Kacher, Olivier Pierron, Shigenobu Ogata, Ting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28629)  

**Abstract**: Agentic artificial intelligence (AI) systems hold immense potential for automating complex workflows in materials discovery and design, but extending them to domain-intensive mechanistic investigations remains challenging. Here, we present GBClaw, an agentic AI framework for investigating the structure and deformation of general grain boundaries (GBs). Although general GBs constitute the vast majority of real-world interfaces in polycrystalline materials, they remain understudied because their structural complexity and interfacial mismatch complicate experimental and modeling research. GBClaw integrates large language models (LLMs) into a modular architecture comprising an orchestrator, specialized agents, domain-specific tools, simulation interfaces, and hierarchical memory. Through dynamic replanning and iterative workflow refinement, the framework effectively connects systematic simulations with targeted mechanistic analysis. Applied to 14 general asymmetric tilt GBs in nickel, GBClaw characterized interfacial defect structures and identified diverse deformation modes across these incommensurate boundaries. Comparing boundary sliding with and without secondary GB misfit dislocations revealed substantial differences in shear strength, underscoring the critical role of defect hierarchies in sliding resistance. Furthermore, researcher-guided follow-up calculations traced an anomalously high shear resistance to metastable boundary structures. This work demonstrates that combining agentic AI with domain expertise bridges high-throughput computational screening with deep physical understanding of complex material interfaces. 

---
# UniWALM: Unified World Action Language Model for Embodied Robot Control 

**Authors**: Rahul Sajnani, Diego Romeres, Mingyuan Zhang, Kai Gao, Carl Qi, Baichuan Huang, Srinath Sridhar, Dylan Randle, Lillian J. Ratliff, Jose Barreiros, Manikantan Nambi, Jeroen Vanbaar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28626)  

**Abstract**: Motivated by the human ability to integrate experience across language, perception, and action, we introduce UniWALM, a unified world-action-language model that learns video, action, and language in a single shared representation, where each modality is both a conditioning signal and a supervision target. We ask: can jointly generating language enrich this shared representation, improving manipulation and supporting memory-dependent behavior? Unlike prior approaches that attach a reasoning or vision-language model to a video-action backbone and run them as distinct stages, UniWALM unifies all three modalities inside a single mixture-of-transformers backbone whose per-modality diffusion heads co-denoise synchronously, decoding subtask and textual memory in the same pass as action with no separate planning stage; the generated language is re-consumed on the next step, closing a language-as-memory loop without an external planner or memory bank. To keep the action stream reliable under this richer objective, we further introduce Flow Matching with Correction (FMC), a training objective that teaches recovery from perturbed action trajectories at no inference-time cost. UniWALM consistently improves over same-backbone baselines (e.g., FastWAM) on compositional pick-and-place (MolmoSpaces) and across the memory-dependent subset of RMBench, where it reaches 34.5% without any pretrained language or reasoning model and exceeds Mem-0's keyframe-equipped setup (28.5%) using only a single anchor frame. On a real-world Find-the-Item task, generating textual memory lifts success from 11.1% to 61.1%, while state-of-the-art VLAs stay at or below chance (<=33.3%). 

---
# Industrial ReMEmbR: Scaling Spatiotemporal Memory to Robot Fleets 

**Authors**: Jad Yammine, Malek Wahidi, Abrar Anwar, Yan Chang, Mihir Acharya, Anthony Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2610.28624)  

**Abstract**: Warehouse robot fleets continuously observe operational conditions, but preserving these observations as reusable semantic memory incurs substantial computation, communication, and storage costs. For example, explaining recurring aisle blockages can require reviewing video from multiple robots across several shifts. We present IndustrialRemembr, a shared spatiotemporal memory system that extends NVIDIA ReMEmbR to robot fleets and reduces server-side computation by generating scene descriptions only on demand. On each robot, an NVIDIA Jetson encodes video clips into retrieval embeddings with Cosmos-Embed1, vision tokens with Cosmos3-Edge, and reconstruction latents with the Wan2.2 VAE, without uploading raw camera frames. The encoding models are re-used to detect people in image patches, zeroing them out to support PII abstraction for GDPR-compliant deployments. We split Cosmos3-Edge at its vision-language boundary, placing the vision encoder on the robot and the language decoder on an on-premises server. A Nemotron agent retrieves memories by content, position, and time and queries the decoder using cached vision tokens. This design eliminates caption generation at write time, while cached-token decoding reproduces the unsplit model's output in all 450 tested decodes. 

---
# MindView: Benchmarking Generative Multi-View Spatial Intelligence 

**Authors**: Yanbo Wang, Siqi Luo, Siyu Zhang, Chenming Zhu, Wenzhao Zheng, Qian Yu, Juncheng Yan, Bin Fu, Tai Wang, Yu Qiao, Yihao Liu, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28622)  

**Abstract**: Spatial intelligence has recently been assessed through generative tasks such as spatially grounded image editing, yet language-guided generation of spatially consistent views from partial multi-view observations remains underexplored. We introduce MindView, a benchmark for generative multi-view spatial intelligence with 1,500 samples from 513 scenes across three indoor datasets. Its seven tasks form two families: geometry-dominant tasks specify changes in camera position and orientation, while semantics-dominant tasks require inferring target viewpoints by grounding referred objects and interpreting spatial relations and reference-frame cues. Evaluation of eleven generators reveals substantial room for improvement on MindView. A paired context analysis further shows that all eleven generators obtain lower overall scores under the expanded context setting. These findings motivate us to investigate whether task-specific supervision can improve target-view generation and whether models can use additional multi-view evidence effectively. We therefore fine-tune BAGEL on the MindView training dataset, improving its overall score by 24.3 points and yielding gains across all seven MindView tasks. Beyond these generation gains, the fine-tuned model also yields higher scores on three spatial-understanding benchmarks without training on their annotations, suggesting that learning to generate spatially grounded views may benefit spatial understanding. To improve evidence utilization under expanded context, we propose MindCue, a training-free spatial evidence selector guided by a target-view hypothesis. MindCue reverses the drop in overall score under expanded context and outperforms the standard-context baseline. Overall, MindView provides a concrete testbed for generative multi-view spatial intelligence. 

---
# Benchmarking Arabic Dialects for Math Problems 

**Authors**: Noor Abo Mokh, Teresa Lynn, Nizar Habash, Ahmed Elshabrawy, Sara Alhajeri, Alaa Elsetohy, Preslav Nakov, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2610.28607)  

**Abstract**: Mathematical reasoning remains underexplored for low-resource languages, including Dialectal Arabic. To bridge this gap, we introduce the DAR-MGSM (Dialectal Arabic Multilingual Grade School Math) Benchmark, the first manually curated benchmark for evaluating mathematical reasoning in Dialectal Arabic. DAR-MGSM consists of math word problems (MWPs) requiring multi-step reasoning and covers five major Arabic dialects alongside Modern Standard Arabic (MSA). The benchmark comprises two parallel versions: translated-only and translated & culturally localized. The latter version adapts entities and references to regional varieties. We evaluate eleven Arabic-centric and multilingual large language models on DAR-MGSM. The results show that the performance generally declines on the localized version, highlighting the challenges introduced by culturally grounded adaptations. However, the magnitude of this effect varies across models, with Arabic-centric models exhibiting different levels of sensitivity to localization. A mechanistic interpretability analysis shows that translated and localized variants of the same MWPs often induce distinct internal reasoning trajectories, even when aggregate accuracy remains similar. Overall, our results show that mathematical reasoning performance depends on both linguistic variety and cultural context. 

---
# Turning TS2Vec into a Foundation Model for Time Series Classification 

**Authors**: Vasilii Feofanov, Zhenghan Tai, Andrei Ermakov, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2610.28602)  

**Abstract**: TS2Vec is one of the strongest self-supervised methods for time series classification, introduced in 2022. It is designed to be trained anew on every dataset: self-supervised pre-training followed by linear probing. Since then, the field has shifted towards foundation models that reuse one encoder across datasets. Yet TS2Vec offers a compact convolutional architecture that operates directly on temporal samples and accepts variable-length inputs without resampling. We ask whether these properties can be retained when the encoder is pre-trained only once, and show that they can: TS2Vec can essentially be turned into a foundation model. Three changes suffice: pre-train on a synthetic corpus, feed the encoder a raw amplitude view and an instance-normalized shape view of the input through two small stems, and distill one specialist encoder per view into a single encoder. The result, TS2Vec-FM, is a 1.6M-parameter dilated convolutional network that, kept frozen, reaches 83.3% on UCR-128 and 75.3% on UEA-27 under linear probing, within 0.3 pp of MantisV2, the current state of the art, and above TS2Vec trained separately on each target dataset. What we find most interesting is how different TS2Vec-FM is from other foundation models in design: near-parity is reached without attention, without a patch tokenizer, and without intermediate-layer extraction, using an "old-school" convolutional encoder and a two-layer MLP stem. 

---
# Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation 

**Authors**: Shu Zhao, Tan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28586)  

**Abstract**: Agentic on-policy distillation (OPD) trains a student agent on its own trajectories while a teacher scores every token. A natural explanation of where it goes wrong is compounding error: the student's mistakes carry its trajectories beyond the region where the teacher is reliable, so recent methods keep training inside that region, through episode-length curricula, teacher-guided turns, prefix replay, or teacher intervention where the two disagree. We show this explanation is incomplete. From the states where the student departs most from the teacher, the teacher's own takeovers still finish the task, yet the student trained on its token scores keeps running out of turns. We attribute this to the myopia of next-token scoring. The teacher rates each token without seeing where the episode ends, so its scores let through the safe, delaying moves that outcome-based RL would punish with zero reward, a penalty agentic OPD lacks. We call this difference between what the teacher does and what it scores the rollout-scorer gap. We propose Rollout-to-Scorer Distillation (R2S), which repairs the scorer before distillation by mining high-disagreement states from the untrained student, running teacher takeovers from them, keeping the states where enough takeovers pass a reward-free check such as consensus or an LLM judge, and fine-tuning the teacher to give its takeover-informed scores on every assistant token from the divergence turn onward, without the takeover in context. The repaired teacher is frozen as the sole scorer for standard OPD, requiring no task reward during repair or training. Across three environments, two teachers and four students, R2S outperforms three state-of-the-art agentic-OPD methods in all twelve teacher-student-environment cells, by 12.8 success points on average over the best of them in each cell. Project page: this https URL. 

---
# SceneRig: An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images 

**Authors**: Rundong Luo, Yunong Liu, Di Cao, Matthew Tancik, Shyamal Buch, Colton Stearns, Wenqi Xian  

**Link**: [PDF](https://arxiv.org/pdf/2610.28585)  

**Abstract**: Simulation is a powerful tool for robotics, but building diverse, simulation-ready scenes at scale remains difficult. Reconstructing scenes from single images is a promising alternative, but visual similarity alone is insufficient: objects must also be recovered, accurately placed, and physically grounded. Existing methods that rely on post-hoc physics settling often misplace objects, producing scenes that look correct but behave poorly under manipulation. We present SceneRig, an agentic system that turns a single image into an editable, simulation-ready scene. SceneRig combines geometric, visual, and physical tools to reconstruct a faithful 3D scene. An agent then performs edit-and-validate loops inside a physics simulator to achieve physical stability. Across diverse inputs, SceneRig improves instance segmentation, object-centric appearance, and 3D placement while keeping the reconstructed scenes physically stable. Furthermore, SceneRig is effective for downstream robotics: its scenes predict real-robot policy outcomes more accurately than the baseline method and more reliably support replay of recorded manipulations without trajectory adaptation. 

---
# Spatially Informed and Direction Sensitive Representation Learning for Spatial Transcriptomics 

**Authors**: Azim Dehghani Amirabad, Ali Khodabandeh Yalabadi, Artem Moskalev, Yun-Ching Chen, Mehdi Pirooznia, Scott Oloff, Rui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28579)  

**Abstract**: Cellular states are shaped by both intrinsic gene expression programs and the spatial microenvironment in which cells reside. Spatial transcriptomics makes this context measurable, but effective representation learning requires capturing both spatially informative molecular signals and intercellular organization. Existing approaches address these challenges only partially: conventional gene selection can overlook spatially structured signals, while methods for encoding cell-cell relationships often either reduce a cell's microenvironment to proximity alone, discarding direction, or rely on absolute coordinates tied to an arbitrary tissue frame. Consequently, cells with similar neighbor distances but distinct spatial organization may receive overly similar representations, despite experiencing different biological contexts that shape their cellular states. We introduce Tessera, a spatial representation learning framework that addresses both molecular selection and cellular context. Tessera combines spatially informed gene selection with a geometric inductive bias that contextualizes each cell within its local tissue environment. By modeling relative cell-cell relationships, Tessera is translation-invariant while preserving direction, allowing neighboring cells to contribute differently according to their spatial context and cellular state. Across held-out tissues and technologies, Tessera outperforms existing spatial foundation models in representation quality and generalizes zero-shot to contexts up to $16\times$ larger than those seen during training. 

---
# Large Flow Language Models 

**Authors**: Arnav Chavan, Chanhyuk Lee, Udbhav Bamba, Jaehoon Yoo, Seunghoon Hong, Aditi Raghunathan, Jinwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.28559)  

**Abstract**: Continuous flow language models offer a path to parallel generation with tools for control and acceleration unavailable to discrete diffusion, yet effective techniques for building them at the billion-parameter scale remain less established. We present Lafa (Large flow language model), a fast, high-performing flow language model scaled to 8B parameters in a data-efficient manner. Lafa is a blockwise flow model with causal attention on the prefix and bidirectional, asynchronous flow within each block. This design enables efficient training by adapting pretrained autoregressive (AR) models and fast inference via prefix caching and blockwise parallel decoding. We train Lafa at 0.5B, 7B, and 8B scales with fewer than 30B tokens, more efficient than previous diffusion models, and evaluate it on instruction following, reasoning, and coding, where the learned flow matches or surpasses the quality of AR and can be distilled into flow maps for one- and few-step generation. Notably, flow training preserves or even improves pretrained AR capabilities via a multi-token prediction effect, which we leverage in a self-speculative decoding scheme that verifies flow-based drafts autoregressively within a single model. This yields 2.54x the throughput of an AR model at comparable quality, outperforming multi-token prediction and discrete diffusion methods. Our findings show that continuous flows can serve as a scalable foundation for language modeling. 

---
# When Do LLMs Help with Feature Selection? A Systematic Empirical Study 

**Authors**: Fateme Memar, Dongjie Wang, Arvin Agah  

**Link**: [PDF](https://arxiv.org/pdf/2610.28556)  

**Abstract**: Large language models (LLMs) can leverage semantic feature names alongside summary statistics for tabular feature selection, yet it remains unclear when such information improves subset selection relative to established statistical methods. We investigate this question within an 18-dataset framework comprising 10 datasets for Group Relative Policy Optimization (GRPO) training, 2 for checkpoint validation, and 6 held-out evaluation datasets. We compare five classical baselines, frozen Qwen2.5-7B and Llama-3.3-70B selectors, and two GRPO-adapted Qwen2.5-7B models, reporting classification ROC-AUC and regression R^2 separately and using dataset-wise ranks for cross-task comparison. Across five seeded evaluation partitions, classical selectors generally achieve higher predictive performance than the tested 7B LLM/RL selectors, including under a pre-specified ~33% feature budget with strict train-only classical fitting, while the 7B selectors retain compact subsets. A cardinality-matched semantic sensitivity experiment shows that, for Zero-shot Qwen-7B, informative feature names improve regression performance over anonymized and shuffled alternatives (+0.1964 and +0.1606 mean R^2), whereas aggregate classification differences are smaller (+0.0159 and +0.0084 ROC-AUC) and dataset-dependent. These results characterize when semantic information benefits LLM-based feature selection and show that classical selectors remain strong baselines for predictive tabular feature selection. 

---
# Learning Computational Structure for Sample-Efficient Prediction 

**Authors**: Hangyu Zhou, Constantine Dovrolis, Stephen Mussmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.28555)  

**Abstract**: Statistical learning in high dimensions suffers from the curse of dimensionality. Fortunately, many functions of practical interest are compositionally sparse, meaning they admit a composition of local functions, each depending on only a few variables, allowing sample complexity to depend on local arity rather than ambient dimension. Yet realizing this benefit requires exploiting hidden compositional structure, which theory does not establish for practical gradient-based training. We study this gap with CSFRBench, a controlled benchmark that hides the target's compositional computation during training but retains it for diagnosis, and find that dense networks and sparsity-based baselines leave much of that benefit unrealized. Inspired by the spatial organization of biological neural circuits, we introduce Learned Wiring (LeWi): each neuron has a learned low-dimensional embedding, only nearby neurons connect, and embeddings and weights are trained jointly. LeWi improves sample efficiency on targets with hierarchical structure, intermediate reuse, or independent modules. Across 32 additional targets generated by independently sampled computation graphs, LeWi is the most sample-efficient method not given the target graph, with 27% lower mean learning-curve area than dense networks. Dense networks perform better on targets with a single high-arity function. LeWi's representations and connectivity align with the target computation. On 17 world-model prediction tasks spanning control, navigation, and manipulation, LeWi beats the dense baseline on 16 and wins 99/119 baseline comparisons. These results show that jointly learning connectivity and weights can exploit computational structure to improve sample efficiency. 

---
# Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks 

**Authors**: Rodrigo Casado Noguerales, Dimitri von Rütte, Bernhard Schölkopf, Thomas Hofmann, Aran Raoufi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28543)  

**Abstract**: Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and graph coloring, the symmetric formulation of binary diffusion consistently outperforms strong, well-tuned masked and uniform baselines under matched architectures and controlled training, sampling, and tuning budgets, with its largest gains on the hardest instances, while remaining competitive on language modeling (Text8, TinyStories, OpenWebText). These results support candidate sets as an effective intermediate representation for iterative reasoning and show that the corruption kernel can materially change the computation learned by the reverse model. 

---
# VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction 

**Authors**: Yuto Shibata, Kotaro Amaya, Kashu Yamazaki, Lalit Jayanti, Yoshimitsu Aoki, Mariko Isogawa, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.28523)  

**Abstract**: Humanoid-human interaction requires robots to operate under severe partial observability: during close physical contact, the human partner is occluded, truncated, or outside the robot's field of view, causing pose estimation to fail and often necessitating external tracking or privileged human state. At the same time, heavy mutual occlusions introduce errors into the human-human motion-capture data used for learning. We present REACT, a framework for learning deployable humanoid-human interaction policies from human-human demonstrations. REACT treats semantic contact structure as the key supervisory signal connecting demonstration processing, interaction learning, and onboard control. First, a VLM-guided contact modeling and retargeting pipeline identifies semantically meaningful contacts and transfers them onto humanoid-human geometry while preserving interaction structure. Using these annotations, privileged interaction policies learn physically grounded behaviors through interaction-aware contact rewards and are then distilled into deployable egocentric depth policies trained entirely from onboard sensing using photorealistic human depth rendering. To address severe partial observability, we introduce generative partner-state modeling objectives that predict partner motion from occluded egocentric views. We evaluate REACT in simulation and on a real Unitree G1 humanoid across five contact-rich interactions. Semantic contact modeling substantially improves preservation of demonstrated interactions, while generative partner-state modeling improves contact recall on four of five interactions, most under heavy occlusion. On real hardware, REACT performs all five interactions using onboard sensing alone. 

---
# ECG for laboratory values:A comprehensive physiological and technical analysis 

**Authors**: Walentin Palanchuk  

**Link**: [PDF](https://arxiv.org/pdf/2610.28507)  

**Abstract**: Laboratory measurements are essential for clinical decision making but typically require invasive blood sampling. Recent research suggests that ECG recordings may contain information beyond cardiac rhythm and provide information related to laboratory values. This thesis investigates NT-proBNP prediction from ECG recordings and patient metadata using a multimodal deep learning framework, while considering the physiological plausibility of ECG-based prediction. It also compares a 1-lead configuration using lead II with the corresponding 12-lead configuration. Additional experiments on glucose, hemoglobin, and potassium are presented in the appendix.
Regression, binary classification, and three-class classification experiments were conducted using MIMIC-IV, with the three-class task based on clinically established thresholds. The 12-lead configuration consistently achieved better performance. Regression reached a mean absolute error of 534.98 pg/mL for 1-lead and 514.27 pg/mL for 12-lead. Binary classification achieved AUROCs of 0.849 and 0.876, respectively, while three-class classification achieved macro-averaged AUROCs of 0.738 and 0.752.
The results indicate that precise NT-proBNP regression remains challenging, whereas broader concentration ranges can be classified more effectively. The findings are consistent with the physiological hypothesis that ECG characteristics and NT-proBNP concentrations reflect shared cardiovascular processes. ECG-based NT-proBNP prediction may therefore provide complementary information, particularly for classifying broader concentration ranges. However, laboratory measurements remain necessary, and validation on independent datasets is required. 

---
# BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models 

**Authors**: Ning Zhang, Shihao Wang, Jinrui Zhang, Chaodong Xiao, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28500)  

**Abstract**: For efficient deployment of large language models (LLMs), methods of ultra-low-bit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key-value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm-system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidence-only decoding, speculative-only decoding, and hybrid decoding provide different quality-throughput operating points without using separate draft and verifier models. Accepted-prefix alignment keeps the committed prefix, materialized causal KV boundary, and CUDA Graph replay state consistent for continued generation. BitMTP uses this contract to select width-specialized low-bit execution and reuse captured graphs over shared KV storage. On NVIDIA H20, BitMTP with hybrid decoding achieves five-task mean speedups of 2.2x on BitNet b1.58-2B and 2.1-2.4x on 1.7B-4B Bonsai models at comparable task quality. On RTX 3090 and Jetson Orin, it further achieves up to 2.0x in five-task mean throughput and 3.3x on MBPP+, respectively. These results demonstrate that BitMTP accelerates ultra-low-bit LLM generation across backbones, model sizes, and GPU types. The source code is released at this https URL 

---
# Model- and Layer-Level Closed-Loop Forecasting with SSM-Based Spiking Neural Networks 

**Authors**: Erik Stolt, Sanja Karilanova, Ayca Ozcelikkale  

**Link**: [PDF](https://arxiv.org/pdf/2610.28495)  

**Abstract**: Deep State-Space Models (Deep SSMs) have shown promising performance and computational benefits in processing of time-series data with complex temporal dependencies. Spiking neural networks (SNNs) constitute a biologically inspired class of deep SSMs, where computing blocks communicate through spikes, discrete single-bit events rather than continuous activation functions. A particularly important task in time-series processing is forecasting, i.e. predicting future values of a sequence from past observations alone. However, how to best perform forecasting with SNNs is an open question, where the standard autoregressive nature of the data processing with SNNs requires values that are not available during the forecasting phase. In this article, we address this challenge. We explore two closed-loop forecasting strategies, in which the network's own generated values are fed back into the network as if they were observed past values: (i) model-level and (ii) layer-level closed-loop forecasting. We evaluate these approaches on the ETT-small datasets, a standard benchmark for time-series forecasting, under both spike-based and continuous activation functions. We compare the proposed closed-loop approaches against state-of-the-art methods such as SpaceTime, as well as the zero-masking baseline in S4. Both of the proposed approaches perform comparably to or better than zero-masking with S4, and in most cases match SpaceTime. These findings highlight the trade-off between continuous and low-bit quantized activations, and demonstrate that the proposed Deep SSM architectures provide a competitive framework for forecasting. 

---
# Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting 

**Authors**: Mohammad Reza Modarres, Armin Tourajmehr, Artin Tavasoli, Yadollah Yaghoobzadeh, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28434)  

**Abstract**: Large language models leave model-specific traces in their generated text, but it remains unclear how finely these fingerprints can distinguish related models and which signals remain informative as obvious cues are reduced. We study this through fingerprint resolution, considering both model resolution, from different families to distilled and adapted variants, and signal resolution across behavioral, stylometric, lexical, and semantic evidence. We progressively suppress surface cues, paraphrase outputs, vary the amount of available text, and test generalization across prompt distributions and languages. Our results show that fingerprints persist beyond broad family differences, remaining detectable in distilled models and even LoRA adaptations of the same base model. Although attribution weakens as surface and lexical cues are altered, substantial model-specific signal remains. These signals transfer across prompt sources and languages, with semantic representations proving most robust across languages. These findings show that model identity is recoverable at multiple resolutions, but that recoverability depends on model similarity and the evidence available in the text. 

---
# Plausible Future, Wrong Choice: Steering World-Action Models toward Intended Future 

**Authors**: Zhuohao Li, Jiawei He, Yuxuan Yan, Lang Zhou, Yinghao Li, Yanrui Li, Tianyu Zhang, Yi-Yang Song, Xirui Wu andYi-Lin Wei, Fating Hong, Qing Zhang, Zhilin Zhao, Jingdong Wang, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.28431)  

**Abstract**: World-Action Models (WAMs) have emerged as a promising paradigm for embodied generalist policies by jointly modeling future world evolution and robot actions across diverse manipulation tasks. However, representing diverse futures is only the first step: a capable WAM must also generate futures that match the requirements of the current task and execution progress. In long-horizon manipulation where similar observation-action segments occur across different tasks and stages, existing approaches may generate locally plausible yet misdirected rollouts to drift toward wrong task stage, i.e., a phenomenon we term Stage Confusion. To address this problem, we propose TaskWAM, a closed-loop, stage-aware framework that represents fine-grained task requirements with Mode Tokens and tracks execution progress with State Tokens, forming a structured task state to steer world--action generation toward intended future. We further introduce StageBench, a controlled multi-stage manipulation benchmark with parameterized stage variations and automated this http URL experiments on both simulation benchmarks and real-world tasks demonstrate that TaskWAM effectively reduces confusion and achieves competitive performance with state-of-the-art methods. Moreover, TaskWAM provides strong plug-and-play transfer across heterogeneous WAM architectures and robot embodiments, including single-arm robots, dual-arm grippers, dexterous hands, and humanoid robots. 

---
# A Digital Twin to Improve Cyber-Social Impact 

**Authors**: Vita Santa Barletta, Danilo Caivano, Gabriel Cellammare, Andrea Dimauro, Pasquale Mesagna, Massimiliano Morga  

**Link**: [PDF](https://arxiv.org/pdf/2610.28430)  

**Abstract**: The increasing complexity of Industrial Control Systems (ICS) and their integration with IT infrastructures has significantly expanded the cyber threat landscape. Traditional cybersecurity models, primarily based on the Confidentiality, Integrity, and Availability (CIA triad), fail to capture the broader societal and operational impacts of cyberattacks. This paper proposes a novel framework that integrates Digital Twin technology with a Cyber-Social Security model to simulate, analyze, and quantify cyber threats in critical infrastructures. Virtualized ICS environments built with Mininet and MiniCPS are used to model attack scenarios and assess their consequences across technical and social dimensions. At the core of the framework lies the Cyber-Social Vulnerability Scoring System (CSVSS), a quantitative metric that extends CVSS with a Social Impact dimension grounded in the Entity and Place pillars, producing a single, reproducible severity score that reflects real-world consequences. Crucially, the framework does not automate risk assessment but keeps the human analyst at its center: a Large Language Model (LLM) performs a first-pass analysis, mapping vulnerabilities and drafting risk reports, while the analyst, supported by explainable AI feedback, validates the findings and retains decision authority over critical cases. Experimental results demonstrate the framework's ability to model and score cyber-social impact, while the human-in-the-loop design is presented as a conceptual principle to preserve oversight. 

---
# Racer: Hunting GPU Kernel Bugs with Symbolic Equivalence Checking 

**Authors**: Tuan Truong Vu, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28429)  

**Abstract**: GPU kernels are critical to the performance of today's AI services, yet the same operators are rewritten repeatedly: for each GPU generation, numeric format, serving regime, and kernel language. Every rewrite must preserve the function computed by the implementation it replaces. Today, this obligation depends almost entirely on differential testing: run the new kernel and a trusted reference on randomly sampled tensors, then compare their outputs within a tolerance. This approach has blind spots that additional test cases cannot eliminate. For example, a logical bug may produce an error smaller than the tolerance required to accommodate a correct optimized kernel.
We propose symbolic equivalence checking, which executes a kernel and its reference on symbolic inputs, turning each output cell into an exact algebraic expression, and then checks whether the paired expressions agree for all inputs. Because these expressions are exact, symbolic equivalence checking distinguishes floating-point error from logical bugs, while a single check covers the entire input space.
We implement Racer, a PTX-level symbolic equivalence checking engine, with three techniques that make this approach practical for production kernels: (1) Shape Reduction to control checking cost, (2) a lockstep CTA scheduler to handle concurrency, and (3) idiom templates with exact numeric refutation for otherwise inconclusive results. Across 2,408 production kernels from 41 collection targets, Racer found 82 real bugs in 23 libraries, including 76 previously unknown bugs and 41 confirmed so far. 

---
# After the Attacker Leaves: Benchmarking Carryover Attacks on LLM Agents 

**Authors**: Zhiyue Xu, Xukai Wang, Zhiyuan Huang, Mingjie Zhan, Jian Zhao, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28425)  

**Abstract**: Modern LLM agents persist state across sessions via two channels: long-lived memory files silently prepended to the system prompt, and scheduled tasks waking the agent without user input. A one-time indirect prompt injection committed to either channel can re-fire across sessions long after the attacker leaves, threats we term carryover attacks. Existing prompt-injection benchmarks evaluate attacks only in the session where malicious content is first observed, leaving the gate governing when and how often a planted payload later re-fires unmeasured. We present CarryoverBench: 12 personal-assistant scenarios and 101 attack tasks across three trigger channels: Instant, same-turn IPI; Autonomous, heartbeat-scheduled; and Latent, memory-resurfaced. A rule-based verifier scores planting and triggering as independent stages. Across five frontier models, planted payloads re-fire without measurable decay over five rounds; conditional on successful planting, the trigger stage often provides little additional protection. Porting the Latent channel to a second runtime, Hermes, attenuates but does not eliminate the effect, suggesting that carryover defense requires runtime-layer controls in addition to prompt-level interventions. 

---
# WorldSR: Harnessing World Knowledge Search for Grounded Image Super-Resolution 

**Authors**: Junyuan Deng, Xinyi Wu, Zhenyao Wu, Song Wang, Yu Wang, Congchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28423)  

**Abstract**: Generative-prior-based super-resolution (SR) methods leverage implicit knowledge learned from large-scale models to enhance low-resolution inputs. However, such priors are inherently stochastic and optimized for diverse generation, conflicting with the deterministic, high-fidelity reconstruction required for SR. It is also tied to the base model, costly to update, and prone to hallucinations as real-world data evolves. To address these limitations, we introduce WorldSR, a novel framework that augments SR with explicit, entity-aligned visual world knowledge. WorldSR automatically decomposes the input into semantic entities and performs entity-wise retrieval of structurally and semantically aligned images from a large-scale external corpus. For efficiency and reliability, we incorporate a memory mechanism to reuse previously retrieved results and a multi-step filtering process to refine candidate references. The selected references as world knowledge are then integrated through a novel multi-reference super-resolution model, which provides fine-grained constraints for detail synthesis and structural recovery. Experiments on real-world SR benchmarks show that WorldSR consistently improves structural fidelity and visual quality, outperforming existing generative super-resolution methods. Our results suggest that entity-level grounding in external visual world knowledge provides a simple and effective complement to model-bound generative priors. 

---
# DISTRACT: Degradation-Indexed Study of Textual Reference Across Competing Targets 

**Authors**: Dan Zlotnikov, Ohad Ben-Shahar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28419)  

**Abstract**: Vision-language trackers (VLTs) accept descriptions intended to assist tracking, but accepting text does not establish that its content changes tracking decisions, particularly as visual evidence deteriorates. We introduce DISTRACT, a matched-intervention benchmark that holds video, target, and distractor fixed while crossing four text priors with four image degradations along a controlled severity axis. We evaluate five VLTs on four datasets using behavioral contrasts, layerwise representation probes, and activation patching. Molmo2 and JointNLT exhibit directed text use. Some effects contract under degradation, whereas others show no significant endpoint change after multiplicity correction. UVLTrack, MMTrack, and DUTrack respond only weakly to changes in text semantics, with no directed behavioral effect surviving FDR correction in the tested settings. Model-internal analyses reveal that at decision-near states, correct-versus-conflicting text is decodable in Molmo2 and JointNLT but not in the other three systems. In the responsive models, decodability peaks at late states where replacement produces little behavioral change, whereas replacing selected earlier states redirects conflict-associated trajectories toward the target. Because matched-correct donors do not significantly outperform unrelated donors, these interventions identify earlier intervention-sensitive states without establishing target-specific semantic rescue. DISTRACT thus exposes when and where text affects tracking under degraded visual evidence. 

---
# Code2Video-Compass: A Comprehensive Benchmark for General-Purpose Code-Based Video Generation across T2V, I2V, and V2V 

**Authors**: Zhenghan Chen, Zekai Shao, Xin Lin, Leting Cheng, Siming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28416)  

**Abstract**: Code-based video generation offers fine-grained control, targeted editing, and component reuse compared with direct video generation. This approach requires multimodal large language models (MLLMs) to translate visual and temporal requirements into executable programs that produce complete videos. However, existing code-based benchmarks primarily target specialized tasks, leaving complete video creation across diverse creative categories and input modalities insufficiently evaluated. We introduce Code2Video-Compass, a general-purpose benchmark for code-based video generation across text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) settings. Code2Video-Compass comprises 2,104 tasks spanning 10 categories, 24 subtypes, and three complexity dimensions covering construction, dynamic animation, and state dependency. It combines tasks curated from textual instructions and visual references with corresponding T2V, I2V, and V2V tasks derived from complete professional and user-created videos. We propose a dual-level evaluation framework combining 12 objective metrics and five subjective metrics. Objective evaluation covers operational performance, video quality, text-video alignment, and reference fidelity, while subjective evaluation uses an MLLM judge to evaluate instruction following and holistic video quality. Extensive evaluation of 15 recent proprietary and open-weight MLLMs from nine families reveals that even leading models face challenges in reliably generating high-quality complete videos aligned with textual and visual requirements, with performance generally declining as construction, dynamic animation, and state-dependency complexity increases. We anticipate that Code2Video-Compass will advance reliable, coherent, and controllable code-based video generation and inspire broader progress in programmatic visual creation. 

---
# Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI 

**Authors**: Spiros Maggioros, Guray Erus, Gareth Harman, George Aidinis, Alexander Getka, Aristeidis Sotiras, Christos Davatzikos  

**Link**: [PDF](https://arxiv.org/pdf/2610.28411)  

**Abstract**: Deep learning models have become essential tools in medical imaging, yet training them end-to-end for diverse tasks remains challenging due to limited labeled data and computational constraints. To address this, we introduce Brain2Vec, a foundation model pre-trained on 74,425 T1-weighted Magnetic Resonance Imaging (MRI) scans pooled from multiple studies. Brain2Vec can serve both as a weight initialization backbone for fine-tuning downstream tasks and as a standalone feature extractor. We evaluated Brain2Vec in three tasks: downstream predictive performance, convergence speed, and embedding quality. Using Brain2Vec's pretrained weights as initialization and fine-tuning on labeled data for a brain age prediction task, results demonstrate improved final accuracy and faster convergence compared to random initialization. Additionally, its frozen embeddings outperform anatomically defined volumetric features on four of five downstream classification tasks and perform competitively with a state-of-the-art foundation model pre-trained on substantially more data. Lastly, we show that Brain2Vec encodes information primarily from a spatially localized and neuroanatomically meaningful set of brain regions, and that manipulating a subset of latent features produces monotonic, region-specific effects on the reconstructed anatomy. Brain2Vec code and trained models are available as an open-source package on this https URL. Users can apply Brain2Vec locally by installing the package, or directly through the NiChart cloud platform this https URL, where Brain2Vec features can be derived without requiring local installation or specialized infrastructure. 

---
# Rewards for Budget-Aware Multi-Turn Agents That Adapt to Arbitrary Test-Time Constraints 

**Authors**: Tom A. Lamb, Adam Golinski, Eleonora Gualdoni, Sinead Williamson, Michael Kirchhof  

**Link**: [PDF](https://arxiv.org/pdf/2610.28409)  

**Abstract**: Large language model agents are often given an inference budget and asked to maximise success within it. A multi-turn agent has to allocate this budget across many environment interactions. An external harness can enforce the budget, but it remains unclear whether the agent itself can learn when and how much to spend. We ask whether agents can learn this allocation across dynamically specified budgets and which rewards best teach this behaviour. We adapt existing length penalties to budget constraints spanning multiple resource types, introduce new reward formulations, and compare eleven objectives in a shared training pipeline. We train Qwen3.5 models at 4B and 9B on AppWorld with budgets over tokens, API calls, and environment steps, and on the BabyAI gridworld with turn budgets. We train a separate policy for each environment and resource type, with each policy trained across budget values. Training with rewards that penalise overspending substantially improves within-budget success and natural compliance over untrained models and over training on task success alone, with agents learning to adapt their resource use to stated budgets and task difficulty. These gains extend to unseen budgets and, zero-shot, to a shifted task distribution and a new benchmark. Reward objectives trade off success within budget and natural compliance. A sigmoid penalty for overspending most often achieves the highest within-budget success at an individual budget, while a fixed negative reward for overspending generally yields the strongest natural compliance. Combining penalties for overspend frequency and magnitude gives the smallest average gap to the best method at each budget. Overall, we show that a single agent can learn to allocate resources across turns and adapt its behaviour to user-specified budgets using budget feedback without external budget-allocation controllers. 

---
# The Instruction Tax on Factual Recall: Base Models Can Beat Their Instruction-Tuned Counterparts 

**Authors**: Christophe Khalil, Aya Mourad  

**Link**: [PDF](https://arxiv.org/pdf/2610.28407)  

**Abstract**: We present our solution to the AKBC LM-KBC 2026 Shared Task, where we achieved a score of 0.6836 on the official leaderboard, more than double the baseline of 0.2964. This task requires generating all possible objects for subject-relation pairs using a language model with up to 32 billion parameters, with no retrieval or fine-tuning involved. We show that the prompting methodology is as important as the model architecture: while the instruction-tuned model version achieves 0.5681 with its chat interface, the base model achieves 0.6836 when prompted using text completion. Thus, instruction tuning decreases factual recall; we refer to this effect as the instruction tax, which is caused by the model's weights rather than the prompting method. Our final solution is fairly straightforward: we prompt the base model with some examples from the training data, generate 64--256 outputs, and retain only the repeated ones. Two specialized methods are then applied for each relation: one improves the recall of large answer sets, and the other verifies a person's vitality before predicting their place of death. 

---
# MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems 

**Authors**: Mohammad Kachuee, Phu Mon Htut, Sheng Li, Nikolaos Pappas, Miguel Ballesteros Martinez, Yassine Benajiba, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2610.28397)  

**Abstract**: Agentic AI extends large language models (LLMs) from generating responses to planning and executing complex workflows, substantially expanding their utility while exposing applications to unsafe actions, information leakage, and adversarial manipulation. Multi-agent systems further expand these capabilities through specialization and collaboration, but distributed authority, heterogeneous roles, and opaque or potentially compromised participants make safeguarding even more challenging. Existing defenses address important individual threats but provide fragmented coverage across local execution, system-level interactions, and enforcement mechanisms. We introduce MAS-Guard, a unified framework that closes this gap for safeguarding single- and multi-agent AI applications containing both instrumentable and opaque agents. Within instrumentable agents, the proposed local Agent Guardrail (A-Guard) layer applies task-specific experts to assess agentic events. At the system level, developer-owned Agent Charters and a Control Plane combine deterministic structural checks of routes, agent capabilities, and message formats with semantic assessments of roles and message content. These complementary layers enable enforcement through a shared set of interventions: allow, reflection, confirmation, and block. Evaluated across six benchmarks covering diverse single- and multi-agent scenarios with three agent LLMs from different providers, MAS-Guard reduces the overall attack success rate by 71.5-86.3% relative to unguarded agents, while meeting the utility reference criterion on benign tasks. 

---
# VisualEvolve: A Skill Self-Evolution Framework for Visual Reasoning 

**Authors**: Jiawei Wang, Yucun Li, Xueqi Li, Shenghao Wang, Xinquan Gong, Dan Zhang, Zheqi Lv, Yuquan Le, Shaozu Yuan, An Zhang, Da Cao, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2610.28386)  

**Abstract**: Visual tool use enables vision-language models (VLMs) to acquire task-relevant evidence, but its effectiveness can be limited by predefined visual operations and the tool-use strategies learned during training. Representing these operations and strategies as reusable external skills allows them to be refined through self-evolution without updating model parameters. However, skill refinement can introduce performance regressions. Meanwhile, inspecting accumulated task execution trajectories together with skill evolution history may lead to evolution evidence overload. To address these challenges, we propose VisualEvolve, a self-evolving framework that evolves executable visual operations and tool-use strategies for VLMs through a Select--Execute--Evolve--Evaluate cycle. Skill-Library Snapshot Selection uses performance on a fixed validation set disjoint from execution tasks to select an effective historical snapshot for subsequent evolution, limiting the propagation of performance regressions. Progressive Evidence Access (PEA) guides capability-gap diagnosis through a coarse-to-fine hierarchy comprising the Skill-Performance Report, Evolution Log, Trajectory Summaries, and relevant raw trajectory segments. Across four VLM backbones and six evaluation sets from three benchmarks, VisualEvolve improves over the Initial Skill Library in 23 of 24 settings, with a maximum gain of 15.1 percentage points on VisualProbe Hard using Qwen3.5-Plus. Ablation studies support the contributions of Snapshot Selection and PEA, while further experiments show that skills can be improved using experience from different data settings and directly reused across backbones. Moreover, VisualEvolve enables effective skill self-evolution with over an order of magnitude fewer task-solving trajectories than estimated for representative RL-based methods. 

---
# Don't Skip the Baseline: A Controlled Comparison of Dynamic Expert Skipping in MoEs 

**Authors**: Jakob Steimle, Thomas Elsken, Daniel Ehrhardt, Lukas Rinder, Daniel Cremers, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28369)  

**Abstract**: A growing body of work proposes dynamic expert-skipping methods for Mixture-of-Experts (MoE) models that adjust the number of experts activated per token during inference, aiming to preserve model quality while further reducing the number of active parameters. These methods have added calibrated importance scores, inter-expert similarity metrics, or learned per-token allocators in order to identify experts to skip. However, different calibration regimes, different reductions in the effective number of experts, and evaluation suites make comparisons difficult. We classify four representative prior methods into a unified taxonomy and benchmark them against a natural, simple baseline: statically reducing the number of experts $k$ (i.e., same reduced $k$ across all layers and tokens). We provide a fair comparison of these methods at matched expert-compute budget, on two modern MoEs (Gemma4-26B-A4B and Qwen3-30B-A3B) in a text-only setting and a contemporary vision-language model (Qwen3-VL) in a multimodal setting. Surprisingly, we find that the simple baseline is hard to outperform consistently across models, modalities, benchmarks, and calibration settings. The methods we benchmark often underperform the baseline and achieved gains, when present, are minor. We further find that the effective $k$ achieved (or optimized for) in calibration can diverge from the effective $k$ in testing, resulting in over- or under-estimating the compute at inference time. Based on these findings, we recommend a calibration-free per-modality static $k$ reduction and validate this on a held-out model and modality. We further argue that a static control at matched measured compute is the baseline any dynamic skipping method must beat. 

---
# Context-Aware Prompt Injection Detection via Small Language Models in RAG Pipelines: nokast-secureRAG 

**Authors**: Abhishek Rai  

**Link**: [PDF](https://arxiv.org/pdf/2610.28364)  

**Abstract**: Local Retrieval-Augmented Generation (RAG) systems are vulnerable to indirect prompt injection and retrieval poisoning, where malicious instructions in retrieved content are mistaken for trusted input. We present nokast-secureRAG, a roughly 0.5B-parameter small language model that acts as a semantic firewall between retrieval and generation. It jointly evaluates the user query and retrieved context, classifying the pair as safe, suspicious, or malicious. The model is trained using a teacher-judge-student pipeline: a 35B teacher generates 5,233 examples across four attack families, and an independent 120B judge re-labels them, achieving 95.0% agreement before student fine-tuning. On a held-out in-distribution test set, the model achieves 0.994 detection recall at a 0.026 false-positive rate and a 0.006 attack-success-rate proxy, outperforming a regex baseline, a public prompt-injection classifier, Llama Guard 3 1B, and its untuned version. Prompt-Guard-86M reaches 0.900 recall but has a 0.690 false-positive rate. On a judgment-flip test, where the same sentence is benign or malicious depending on context, the model correctly classifies both cases 75% of the time, outperforming the compared baselines. Ablations indicate that retrieved context carries the main attack signal, while query-context consistency reduces false positives. Single-request latency is 293-404 ms, above the target of 100 ms, while amortized throughput latency is 37 ms under batching. On the external PoisonedRAG benchmark, recall falls to near zero, indicating that fluent fact-poisoning attacks lie outside the detector's instruction-consistency threat model. We report this limitation alongside the in-distribution results. The model, data-generation pipeline, and evaluation harness are released open source. 

---
# CountVideoBench: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models 

**Authors**: Marc Chapus, Carlos Crispim-Junior, Véronique Eglin, Atilla Baskurt  

**Link**: [PDF](https://arxiv.org/pdf/2610.28356)  

**Abstract**: Object counting in videos remains a poorly characterized capability of multimodal large language models. Counting questions are included in many existing VideoQA benchmarks, but the ground-truth count distributions in these benchmarks are heavily skewed toward small values. Across the nine benchmarks we survey, the median ground-truth count is at most five, and the three most frequent values together account for between 45% and 93% of the videos. The resulting mean accuracy is then dominated by the most frequent counts; a value-dependent bias at a rare count contributes negligibly to the average and remains invisible. We introduce CountVideoBench, a benchmark of 2,600 synthetic videos covering 26 cardinalities in the interval [5, 50] (100 videos each), grouped into six clusters of consecutive counts that span the range while enabling near-neighbor comparisons. Each video contains a single class of static objects under a controlled camera motion, removing object motion and reducing recognition as confounds. We evaluate sixteen video-language models from nine architecture families in a zero-shot setting. No model exceeds 27.6% Exact Match. The dependence of accuracy on cardinality varies across models, ranging from monotonic decline to a non-monotonic peak-and-collapse pattern, which rules out increasing difficulty with cardinality as the dominant explanation of counting failure. Every model further concentrates its predictions on its own set of preferred integers, with accuracy peaks coinciding with these values and collapsing at adjacent counts. These findings highlight the need for balanced, diagnostic counting benchmarks to reliably assess this fundamental visual capability. 

---
# RayCo4D: Ray-Conditioned Continuous Motion Model for Monocular 4D Dynamic Reconstruction 

**Authors**: Yuanxiang Ni, Xianliang Huang, Ruihan Hou, Feng Xie, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28351)  

**Abstract**: Reconstructing dynamic 3D scenes from monocular video requires inferring motion from noisy and intermittently visible observations. Long-range tracks provide valuable geometric constraints, but integrating their fragmented temporal coverage into shared, continuous motion remains challenging. We introduce \textbf{RayCo4D}, a ray-conditioned continuous motion model for monocular 4D reconstruction that explicitly separates visual measurements from latent 3D motion states. Each track is associated with a reference-time 3D query constrained to its camera ray, and its image and depth observations constrain shared latent motion nodes through local many-to-many associations. Each node maintains an $\mathrm{SE}(3)$ pose and body-frame velocity at sparse support times. A white-noise-on-acceleration process prior couples neighboring states and defines continuous-time interpolation, allowing observations across visibility intervals to jointly constrain the same latent state this http URL inferred shared motion is transferred to Gaussian primitives through dual-quaternion blending, while reference-anchored local $\mathrm{SE}(3)$ residuals capture fine-scale non-rigid deformation beyond the shared scaffold. Experiments on diverse monocular dynamic scenes show that RayCo4D achieves competitive novel-view synthesis quality, improves structural and temporal consistency under rapid motion and incomplete tracks, and supports continuous-time rendering. 

---
# Gated Prototype Transfer for Zero-Support Medical Vision Language Model 

**Authors**: Sumaiya Fazal, Rizwan Qureshi, Laleh Behjat, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2610.28350)  

**Abstract**: Realistic few-shot adaptation of medical vision-language models (VLMs) can leave low-prevalence classes without any labeled support. The training-free SS- Text+ solver remains operational in this setting, but its zero-support prototypes receive no visual adaptation and fall back to a purely textual class prototype. We in- troduce Gated Knowledge-Guided Prototype Transfer (GProtoT), a pre-repulsion module that transfers supported-class visual residuals to missing classes using VLM similarity and an ontology-based clinical distance. A validation-free evidence gate scales transfer based on semantic relevance and residual agreement, thereby limiting unsupported extrapolation. In 50 paired realistic one-shot trials on MESSIDOR with FLAIR features, severe and proliferative diabetic retinopathy were absent from every support set. Relative to SS-Text+, gated ProtoT improved missing-class recall by 5.23 points, average class accuracy by 1.37 points, and the missing/supported recall harmonic mean by 1.83 points. It reduced negative- transfer frequency from 46-48% for ungated transfer to 2%, while at the cost of reductions of 1.20 percentage points in supported-class recall and 0.45 percentage points in top-1 accuracy. These results indicate that evidence-based gating provides a safety mechanism for the knowledge source. GProtoT is based on SS- Text+ and requires no gradients, validation data, or additional model inference, of- fering a low-compute path toward accountable adaptation under incomplete clini- cal supervision. Code is available: this https URL 

---
# Beyond a Single Sentiment Score: Topical and Affective Dimensions of Emoji Meaning on Twitter 

**Authors**: Rohan James, Ioannis Christofilogiannis, Yuuv Jauhari, Shreya Shah, Gauri Pethkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28337)  

**Abstract**: Emoji are widely treated as compact affective tokens whose meaning is summarised by a single sentiment score. We argue that this picture is incomplete. From a corpus of 16{,}575 English tweets and a per-emoji metric we call \emph{stability delta} (within-class minus between-class cosine similarity of sentence embeddings), we show that emojis used in tightly clustered topical contexts also exhibit \emph{more} polarised crowd-sourced sentiment, not less. Validated against the Kralj Novak et al.\ (2015) sentiment lexicon, the central correlation is Spearman $\rho = +0.361$ (Holm $p = 0.015$, 95\% bootstrap CI $[+0.10, +0.59]$); against the independent Ferré et al.\ (2025) Spanish 13-emotion ratings, $\rho = +0.509$ ($p = 0.0003$). The result survives permutation testing, BCa bootstrap, leave-one-out jackknife, partial correlation controlling for emoji frequency, and Holm correction across a four-test family. Topical coherence and affective dispersion are distinct dimensions of emoji meaning. As a downstream check we report leakage-free five-fold cross-validation results for sarcasm classification on the Emoji Sarcasm Dataset; all three baselines (a gated multimodal architecture, fine-tuned BERTweet features, and fine-tuned RoBERTa features) score above 0.97 macro F1, indicating the dataset has saturating emoji-label cues. We frame the classifier as supporting evidence rather than the contribution. 

---
# Identifying Dependent Components from Multi-domain Linear Mixtures 

**Authors**: Danru Xu, Lauri Parkkonen, Sara Magliacane, Aapo Hyvärinen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28330)  

**Abstract**: We study a linear mixing model with dependent latent components, assuming multiple data domains. Most existing models assume that the components are independent or at least uncorrelated, in line with independent component analysis (ICA). Some recent work allows for dependent components, but then makes specific assumptions such as parametric forms of dependencies, multi-view settings, or interventions, or does not recover the individual components. In contrast, we consider a multi-domain setting in which domains differ through domain-specific scalings of the components, while the distribution of the underlying latent components is the same across domains. This approach can model data collected, for example, from different sensors measuring the same process, different laboratories conducting the same experiment, different experimental conditions, or different subjects that might differ in biological or physiological factors. We show that, under sufficient domain variability, latent variables and mixing functions can be identified from second-order statistics alone. We propose the Multi-Domain Covariance Matching (MuDo-CoM) algorithm that generalizes previous methods of joint diagonalization. MuDo-CoM is validated on simulated data and a real-world fMRI dataset. 

---
# ABACUS: A Benchmark for Image Editing Models to Assess Object Category and Counting Under Scene Conservation 

**Authors**: Aymen Lassoued, Angelique Loesch, Romaric Audigier  

**Link**: [PDF](https://arxiv.org/pdf/2610.28316)  

**Abstract**: Instruction-based image editing models can now add, remove a specified number of objects or change their category while preserving the scene. Yet these models often fail to produce the correct number, and these failures grow sharply as scenes become more complex and the requested count increases. Current benchmarks largely miss this problem: they score edits with holistic similarity metrics or with vision-language model (VLM) judges that are themselves poor at counting, so numerical errors go unnoticed or are graded incorrectly. As a result, the true numerical ability of today's editing models has never been accurately measured. We address this issue with ABACUS, a new benchmark, evaluation protocol and dedicated scoring metric for numerical image editing. Our protocol is judge-free: no VLM or MLLM scores an edit or counts objects (a VLM only confirms the category of rare ambiguous crops). Instead, it verifies each edit at the instance level against panoptic ground truth confirmed by detector consensus, checking both that the exact requested count was achieved and that the rest of the scene was preserved, across the add, remove and change operations. By evaluating leading image editors, our method reveals substantial numerical editing failures that worsen with increasing scene complexity and object count; failures that prior benchmarks often overlooked or misjudged. 

---
# Local Work, Global Progress: A Convergence Theory of AI Agent Collaboration 

**Authors**: Ivan Ilin, Peter Richtárik  

**Link**: [PDF](https://arxiv.org/pdf/2610.28286)  

**Abstract**: Teams of large language model (LLM) agents are increasingly applied to complex tasks, but their collaboration protocols are mostly designed empirically, without convergence guarantees. We formalize collaborative problem solving as block optimization: each agent improves one part of a shared solution, guided by its local objective plus a correction for how its changes affect the other parts. A coordinator checks every proposal against the global objective and merges proposals that do not interfere with each other. We prove linear convergence up to an error floor that vanishes when the corrections are exact. Interfering proposals can also be merged once their combination has been checked; the rate then depends on how well the proposals actually combine, not on worst-case interactions. With GPT-OSS-120B agents on robot routing, a three-agent team finds collision-free plans in every run, better than those of a single agent that controls all robots. On a larger routing task, a six-agent team completes its search about twice as fast as the single agent (in median). Merging the best checked combination of proposals, rather than only proposals that do not interfere, cuts the team's search time by about 30%. With constants measured on the runs, our convergence bound holds in every round and follows the observed progress. We also test the theory on query planning in SQLite and on a synthetic task. 

---
# GALA: Geometry-Aware Language Model for Controllable Object Arrangement 

**Authors**: Chucheng Xiang, Runze Wang, Zhi Deng, Ruchao Bao, Yuanwei Zhang, Yuxuan Xie, Hanliu Wang, Liangzhen Fei, Wenzheng Wu, Cheng Wan, Peifeng Li, Zhongyuan Liu, Ligang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28280)  

**Abstract**: Reasoning about fine-grained placement constraints, such as support and containment, is essential for realistic 3D scene creation. It remains a significant challenge for existing methods that rely on simplified object representations like oriented bounding box (OBB), which are blind to the detailed geometry. To address this limitation, we introduce GALA, a geometry-aware language model with explicit point cloud perception for controllable object placement. Our model performs autoregressively by predicting the 6D pose of one target object at a time conditioned on the scene context and placement instruction. We utilize a two-stage training strategy: the first stage aligns geometric and textual modalities through pretraining tasks, and the second stage fine-tunes the model for placement. To facilitate this task, we also curate a multi-source scene dataset covering diverse room types and fine-grained furniture arrangements, which will be open-sourced for community benefit. Experiments on our benchmark show our model significantly outperforms baselines without explicit geometric reasoning. We further develop a multi-stage plan-place-verify agent pipeline that uses the model for text-driven scene generation. Project Page: this https URL 

---
# VGL-stGen: Learning a shared multimodal model for lung histology and spatial transcriptomics 

**Authors**: Haenara Shin, Dongjoo Lee, Sungwoo Bae, Kwon Joong Na, Daeseung Lee, Hongyoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28266)  

**Abstract**: VGL-stGen is a multimodal large language model adapted from a pretrained medical vision-language model using paired lung H&E and spatial transcriptomics with single-cell references. Training integrates image-expression and expression-cell-type pairs, and gene-conditioned pseudo-labels transfer transcriptomic cell-type supervision to histology images. One deployed checkpoint supports thirteen evaluated tasks spanning tissue and cell-type classification, gene prediction and conditional expression completion from combinations of histology, cell identity and observed gene expression. Task-specific decoding produces categorical labels, gene lists and structured expression levels. For conditional completion, fixing queried gene identities and predicting their levels autoregressively improves structured prediction. Controlled adaptation shows that restricting supervision to requested expression-level tokens can further improve image-conditioned completion. Image-permutation experiments indicate that correctly paired histology adds information beyond observed expression. External classification relies partly on computational references with shared annotation lineage, and some H&E sources were exposed during cancer training or model development. Within these conditions, VGL-stGen demonstrates how a medical vision-language backbone can learn from spatial transcriptomics and histology to connect tissue morphology and molecular states through shared generative modeling. 

---
# PI-MLLM: Psychophysics-Inspired Multimodal Large Language Model for Oddly Satisfying Video Emotion Analysis 

**Authors**: Yuheng Zhang, Jie Zhang, Yuecong Min, Yunwei Zhao, Meichen Liu, Qiankun Li, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28264)  

**Abstract**: Oddly Satisfying Videos (OSVs) can evoke strong viewer responses through repetitive motion, visual tactility, and ASMR-like sounds despite containing few explicit semantic emotion cues. This makes OSVs a challenging setting for Video Emotion Analysis (VEA), since most existing methods focus on semantic video content and under-model the physical sensory dynamics that shape viewer responses. To support Oddly Satisfying Video Emotion Analysis (OSVEA), we construct the Oddly Satisfying Video Dataset (OSVD), comprising 12,000 annotated 5-second clips. Each clip provides viewer-emotion labels under video-only, audio-only, and audio-visual conditions, together with explanations grounded in visual and acoustic sensory evidence. Building on OSVD, we propose PI-MLLM, a psychophysics-inspired multimodal large language model that integrates Modality-Specific Dynamics Alignment (MSDA) and Coalition-Based Audio-Visual Attribution (CAVA). MSDA uses auxiliary prediction objectives to align modality representations with visual motion and acoustic energy-onset dynamics. CAVA evaluates emotion scores across modality coalitions and applies Shapley attribution to identify the dominant modality and determine whether visual and auditory evidence is synergistic or conflicting. Experiments show that PI-MLLM improves both viewer-side emotion prediction and sensory-grounded explanation quality, suggesting the value of physical sensory cues to video emotion analysis. 

---
# TPO: Tri-level Distributionally Robust Learning for OOD Direct Preference Optimization 

**Authors**: Chengtao Jian, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28229)  

**Abstract**: Direct Preference Optimization (DPO) minimizes empirical risk over observed preference comparisons, implicitly assuming that the training preference distribution matches the deployment preference distribution. In practice, this i.i.d. assumption often fails, as preference distribution shift exhibits a natural hierarchical structure across three levels: group-level shift, prompt-level shift, and preference-level shift. Such multi-level distributional mismatch poses a critical challenge for safe LLM deployment, yet existing robust DPO methods typically either model uncertainty over the joint preference distribution or focus on individual sources of shift, leaving the hierarchical structure of OOD preference shifts underexplored. We propose TPO, a tri-level distributionally robust framework for OOD preference alignment, which jointly models all three levels of distributional uncertainty through a hierarchically structured rectangular ambiguity set. By instantiating it with Wasserstein distance, we derive a tractable and interpretable objective that combines a closed-form preference-level margin penalty, prompt-level adversarial embedding perturbation, and worst-case group reweighting, each governed by an independent radius. We establish finite-sample convergence guarantees and empirically validate the effectiveness of TPO over existing robust DPO methods. 

---
# Local Inference of LLMs on Long Contexts: A Data Systems Perspective 

**Authors**: Mattias Dürrmeier, Alberto Lerner  

**Link**: [PDF](https://arxiv.org/pdf/2610.28223)  

**Abstract**: Large language models can solve complex problems across a variety of subjects, such as mathematics and coding. For such complex analytical tasks, models with a long context perform better. Serving models with long context is currently mostly done on powerful systems with multiple GPUs, which are too expensive for most users. An alternative is local inference, or serving a model on a user's own machine, which removes per-token costs and rate limits, besides keeping sensitive data private and giving users full control over the models they run. The common understanding, however, is that local LLMs need compression (e.g., quantization) to run on a single GPU. In this paper, we set out to study the limitations that prevent serving long-context models for single users, besides the capacity issue. Rather than characterizing a single model on one GPU, we analyze how different models behave across several GPU generations. This lets us uncover not just individual bottlenecks but broader trends in where inefficiencies arise. Our analysis confirms that, as expected, self-attention dominates the runtime for long-context inference. More unexpectedly, our evaluation reveals that the current structure of self-attention leaves key GPU resources significantly underutilized, and the newest GPU is often the least well served. Many of these findings are fundamentally data management problems, from paging the KV cache to improving cache utilization and synchronization in the GPU, which is why we bring this study to the Data Systems community. To engage this community, we present our findings as a series of experimental observations and formulate recommendations for future work. 

---
# Region-Guided Food-Package Evidence Extraction and Abstention-Aware Draft R3337 Screening with Selective VLM Review 

**Authors**: Mayimunah Nagayi, Alice Scaria Khan, Tamryn Frank, Rina Swart, Clement Nyirenda  

**Link**: [PDF](https://arxiv.org/pdf/2610.28211)  

**Abstract**: Applying South Africa's draft R3337 warning criteria to food-package photographs requires nutrition, ingredient and product-form evidence. This study presents a region-guided evidence extraction and abstention-aware screening workflow for draft R3337. A three-class detector localises nutrition information panels, ingredient lists and product-identity regions, after which PaddleOCR and structured extraction recover nutrient values, ingredient evidence and product-form information. Recovered evidence is aggregated and assessed using deterministic draft R3337 rules. When evidence is incomplete, uncertain or conflicting, the workflow abstains through Review or Insufficient evidence rather than forcing an automatic outcome. For qualifying unresolved cases, a selective vision-language model (VLM) review stage using Qwen3-VL 8B Instruct reassesses unresolved evidence. Accepted evidence is merged and the same deterministic rules are reapplied. The workflow was applied to 442 products represented by 3,929 images. Evaluation used a manual validation set of 65 products. Before selective VLM review, 31 of 60 decidable products received automatic outcomes and 30 were correct, giving 51.7% coverage and 96.8% accuracy among automatic outcomes (selective accuracy). Selective VLM review resolved eight additional products, increasing coverage to 65.0%. Of 39 automatic outcomes, 38 were correct, giving 97.4% selective accuracy. The single false warning was retained from the rules-only workflow, while no unsafe warning omissions occurred after selective review. Evidence-level evaluation nevertheless identified incorrect VLM-recovered nutrition, ingredient and product-form evidence. These findings show that selective VLM review can recover unresolved evidence, while deterministic screening and explicit abstention remain necessary for reliable image-based draft R3337 screening. 

---
# JAXbounce: A Continuous-Control Pong-Like Benchmark for Active Inference Beyond Low Dimensional Environments 

**Authors**: Chaitanya Kaul, Markus Klar, Sebastian Stein, Fraser C. Paterson, John H. Williamson, Roderick Murray-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2610.28203)  

**Abstract**: We introduce JAXbounce, a continuous control pong-like benchmark designed to evaluate active inference beyond discrete, low dimensional environments. JAXbounce combines continuous states and stochastic dynamics in discrete time with flexible and composable observation models that allow representing the environment as a state vector or a pixel grid, corruptible through the rich family of noise models we provide. It is fully differentiable and parallelisable in JAX. We run over 10,000 agent simulations on our environment comparing a baseline deterministic solution, to a state-based active inference agent with Unscented Kalman Filter perception and expected free energy planning, and a pixel based agent whose measurement model is a pretrained neural network encoder. All agents are evaluated across three scenarios of increasing difficulty with eight different noise conditions. Our experiments characterise how performance of continuous state agents scales with planning, planning horizons, observation modalities and knowledge of the noise process, exhibiting preference-oriented behaviour even in the presence of increasing uncertainty. We release JAXbounce as a reproducible benchmark for continuous control active inference research. 

---
# Parameter-Efficient Adaptation of Time Series Foundation Models for Electrical Load Forecasting 

**Authors**: Aliullah, Julian Oelhaf, Edouard Fouché, Jessica Deuschel, Christian Bergler, Andreas Maier, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2610.28200)  

**Abstract**: Accurate electrical load forecasting supports power-system operation across countries, regions, cities, and buildings, but repeated development of target-specific models is costly. Time series foundation models (TSFMs) provide reusable pretrained backbones, although multi-domain pretraining may not fully capture load-specific characteristics. This paper compares pretrained inference, full fine-tuning, and low-rank adaptation (LoRA) in terms of forecasting accuracy and adaptation efficiency across four electrical load aggregation levels. MOIRAI 1.0 Large serves as the primary backbone, while Chronos-2 assesses whether the findings transfer across model families. The evaluation covers both in-distribution (ID) and out-of-distribution (OOD) settings and examines whether compact adaptation provides consistent gains over strong pretrained checkpoints. Pretrained foundation models outperform the implemented seasonal naive and XGBoost baselines across the evaluated aggregation levels. On MOIRAI Large, rank-16 LoRA reduces normalized mean absolute error (NMAE) by 7.5% on average while updating 3.146 million parameters, about 1% of the model, and retains most of the improvement from full fine-tuning. Chronos-2 achieves lower pretrained NMAE than MOIRAI but benefits less from adaptation. OOD results show that adaptation benefits vary by target, with the clearest gains for the local-area series and several buildings. Overall, LoRA provides a favorable accuracy-efficiency trade-off when adaptation yields a meaningful gain over a strong pretrained checkpoint. 

---
# Reframing Object-Centric Reinforcement Learning: From Absolute to Relative Representations 

**Authors**: Cedric Derstroff, Jannis Blüml, Janosch Moos, Michael Girstl, Zihan Ye, Debora Clever, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2610.28197)  

**Abstract**: Object-centric reinforcement learning agents typically receive each object's position in a fixed global frame, even though their decisions usually depend on where objects are relative to one another. These absolute views often fail to capture the local context that is most relevant to the agent's behavior. As a result, an agent must relearn the same response at every location where a configuration can occur, and its performance collapses when the scene is merely shifted. In this work, we propose a shift toward Object-Relative Coordinates fixed to Anchor (ORCA), where entities are described in a relative frame of reference. To operationalize this idea, we introduce Object-Relative inductive-Bias Transformers (ORBiT), a transformer-based architecture for processing anchor-relative inputs in reinforcement learning. Our results suggest these relative representations improve performance for both MLP and transformer agents. Furthermore, we demonstrate that ORBiT outperforms the pixel-based PPO baseline, achieving an aggregate HNS IQM greater than 1.0 over 13 Atari games. ORBiT learns the game of Foosball from scratch, dominating all baselines. In our controlled Tag environment, we further show that ORCA extends to rotations, that ORBiT generalizes to unseen numbers of distractors, and that these gains require a decision-relevant anchor. 

---
# Adaptive Pullback: Reweighting Supervision at Penalty Jumps in On-Policy Distillation 

**Authors**: Guangyu Liu, Siyuan Gan, Shilong Zhou, Linjian Meng, Boyan Wang, Jing Huo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28194)  

**Abstract**: On-policy distillation (OPD) supervises student-generated responses using sampled-token student-teacher log-ratios. A positive log-ratio penalizes a token that the student assigns higher probability to than the teacher, but its current value does not distinguish sustained high penalties from abrupt increases along a response. We examine these local changes by comparing consecutive blocks of tokens within the same response. Our diagnostic identifies penalty jumps, abrupt increases in average penalty that persist on held-out tokens. Across these jumps, penalties rise while teacher confidence decreases. We call this pattern penalty-confidence tension. Boundaries into high-penalty blocks do not exhibit comparable confidence changes, showing that local changes provide information beyond the current penalty value. Motivated by this observation, we propose Adaptive Pullback OPD (AP-OPD). At each detected jump, AP-OPD uses the preceding block as a local reference and uniformly scales all supervision coefficients in the following block so that its weighted average penalty matches the preceding block's original mean. This reference determines the scaling factor without an additional strength constant. AP-OPD improves performance on both mathematical reasoning and code generation. 

---
# Offline Makes Online Better: Calibrated Cache Compression for Video Generation 

**Authors**: Zhen Wang, Ning Lai, Qingtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28192)  

**Abstract**: Token merging is an effective approach to reducing KV-cache memory consumption in autoregressive video generation by clustering temporally redundant tokens and retaining only a small set of representatives. Existing methods, however, typically apply a uniform merging ratio across all attention layers and heads. We observe that different layers and heads exhibit substantially different yet jointly correlated sensitivities to token merging, and that these sensitivity patterns remain consistent across prompts. Based on this observation, we introduce CALM, a Calibrated Allocation for Layer-Head Merging that uses offline calibration to improve online cache compression. Exhaustively evaluating all possible layer-head allocations is prohibitively expensive. To enable efficient calibration, we show that attention-output error strongly correlates with end-to-end video quality and can therefore serve as a lightweight proxy for allocation risk assessment. We evaluate different allocation strategies using this proxy and construct a fixed layer-head risk map offline. Given a cache budget at inference time, CALM uses dynamic programming to determine the best allocation of merging ratios. Experiments across three different autoregressive video models demonstrate that CALM consistently outperforms uniform token merging under the same memory budget, by reducing the LPIPS to original videos by around 21% with higher video-quality scores. 

---
# Perceives but Doesn't Brake: A Perception-Action Binding Gap in Reasoning Driving VLAs 

**Authors**: Matteo Cercola, Katie Z Luo, Simone Formentin, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2610.28184)  

**Abstract**: Reasoning-augmented vision-language-action (VLA) driving models can perceive and reason about safety-critical hazards yet fail to translate that evidence into safe actions. We characterize this failure as a perception-action binding gap: hazard information is available inside the model but exerts insufficient control over trajectory generation. Using matched simulated counterfactuals and logged driving replay, we show that in-path hazards are actually encoded in the visual representation, but longitudinal plans are dominated by ego-speed history and respond only weakly to hazard proximity or the model's own reasoning trace. Further analysis shows that the trajectory generator is capable of braking but sometimes fails to do so due to the perception-action binding gap. Without retraining or modifying any weights, we inject a hazard-conditioned energy into the frozen flow-matching sampler of Alpamayo-1.5, restoring the missing coupling at generation time. With exact obstacle geometry, guidance cuts footprint intrusion from 58.1% to 14.5% and recovers safe braking whenever sufficient stopping distance remains, eliminating collisions entirely in our braking-relevant closed-loop battery. It also reverses the diagnosed dependence of the plan, from tracking ego speed to responding to obstacle range. These results separate hazard perception from hazard control in driving VLAs, show that unsafe behavior can arise even when the model has both the evidence and the ability to act on it, and propose a simple fix for bridging this binding gap. 

---
# Temporal Evidence-Parameter Co-Alignment for LLM Reasoning over Temporal Knowledge Graphs 

**Authors**: Jiapu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28176)  

**Abstract**: Large Language Models (LLMs) often struggle with temporally grounded reasoning because their static parametric memory is misaligned with the dynamic evolution of real-world knowledge. Although RAG and Parameter-Efficient Fine-Tuning (PEFT) provide practical ways to enhance LLMs, they remain insufficient for temporal reasoning. Existing retrieval methods typically prioritize semantic relevance while overlooking temporal validity, event order, and evolutionary dependencies, resulting in evidence that is relevant in isolation but temporally fragmented. Meanwhile, conventional PEFT methods usually employ fixed and input-agnostic low-rank updates, making it difficult to capture time-dependent entity semantics and relation drift. To address these limitations, we propose Temporal Evidence-Parameter Co-Alignment (TEPCA), a framework for LLM reasoning over TKGs. TEPCA aligns LLM reasoning with temporal knowledge at both the evidence and parameter levels. At the evidence level, it introduces time-constrained Dynamic Relevance-Guided Path (DRGP) pruning to preserve globally coherent temporal paths, together with Mutual-Information (MI)-based pruning to retain query-informative local facts under temporal constraints. At the parameter level, it integrates Dynamic-Gated LoRA (DG-LoRA), a timestamp-conditioned low-rank adaptation mechanism that dynamically modulates the effective adaptation subspace according to temporal context. This design enables LLMs to better capture fine-grained entity evolution and time-dependent relation semantics while preserving parameter efficiency. Experimental results on two TKG benchmarks demonstrate that TEPCA achieves state-of-the-art performance. Additional evaluations of DG-LoRA on eight commonsense QA datasets further verify its general input-adaptive capacity, scalability across LLM backbones, and interpretability. 

---
# Africa-Papers: A 10-Billion-Token Multi-Domain Academic Corpus for African Foundation Language Models 

**Authors**: Mich-Seth Owusu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28173)  

**Abstract**: Large Language Models (LLMs) depend fundamentally on the diversity, depth, and factual rigor of their pre-training corpora. However, contemporary foundational pre-training datasets (such as The Pile, S2ORC, and Dolma) exhibit a severe structural omission: African scientific, medical, legal, and agricultural scholarship constitutes less than 0.5% of global pre-training tokens. Consequently, state-of-the-art models suffer from systematic failure modes when reasoning about tropical epidemiology, localized smallholder agro-ecologies, regional economic governance, and indigenous linguistic structures. To resolve this representation deficit and advance African intellectual sovereignty in artificial intelligence, we introduce Africa-Papers, the largest curated multi-domain Pan-African academic pre-training corpus to date. Comprising over 844,807 full-text papers spanning 10.15 billion tokens across 879 Parquet shards, Africa-Papers unifies four primary scientific streams: (1) continental peer-reviewed journals indexed in DOAJ across 31 African nations, (2) the CGIAR CGSpace agricultural repository, (3) African-authored and African-focused STEM/AI preprints on arXiv, and (4) open-access international publications by scholars affiliated with African institutions indexed via OpenAlex. The entire corpus is processed into layout-aware, clean Markdown preserving section hierarchies, mathematical equations, and tabular data, accompanied by exact BPE token counts and a standardized 8-domain taxonomy. We release Africa-Papers openly on the Hugging Face Hub under the AfriSpeech organization to accelerate continual pre-training and domain evaluation for foundation models globally. 

---
# From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming 

**Authors**: Joschka Kersting, Gesa Benndorf  

**Link**: [PDF](https://arxiv.org/pdf/2610.28160)  

**Abstract**: Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates \cite{pred}, yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects. 

---
# Causal Discovery for Linear Models with Latent Confounding via Multi-domain Identifiable Representation Learning 

**Authors**: Danru Xu, Aapo Hyvärinen, Sara Magliacane  

**Link**: [PDF](https://arxiv.org/pdf/2610.28134)  

**Abstract**: Causal discovery with latent confounding is particularly challenging, but a common setting in real-world data. In this work, we focus on linear causal models with latent confounding, represented by acyclic directed mixed graphs (ADMGs). Existing methods that fully identify ADMGs typically assume non-Gaussianity or graphical constraints, such as bow-freeness. In contrast, we propose Linear ADMG Learning (LinADMG), a framework for learning ADMGs for linear causal models through identifiable representation learning. Rather than relying on non-Gaussianity or bow-freeness, LinADMG leverages multi-domain settings to identify the noise variables. We show that the identifiability of the noise variables is sufficient for recovering all direct causal relations, and that, under an additional latent-confounding no-cancellation condition, LinADMG can identify the whole ADMG. We implement LinADMG with four identifiable representation learning approaches from the literature and validate the methods on simulated and semisynthetic datasets. 

---
# De-Diffusing WAMs: Action-Sufficient Direct Prediction for Robot Control 

**Authors**: Shaoyi Zheng, Zhongtao Jiang, Yue Peng, Wenbo Lu, Shunchi Zhang, Xi Wang, Shengji Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28122)  

**Abstract**: Cascade world-action models (WAMs) use predicted visual futures to condition robot actions, but inherit costly iterative inference from diffusion-pretrained experts. We ask what a single-step predicted future must preserve to support accurate single-step action prediction. In the evaluated WAM, resampling video noise has little effect on downstream actions, yet naively truncating both streams to one step substantially reduces task success. Video-space supervision only partially closes this gap, and a local sensitivity analysis shows that vision prediction errors have highly unequal consequences for action accuracy. These findings motivate DirectWAM, which converts pretrained video and action experts into single-step predictors without stochastic sampling through action-sufficient distillation, while retaining the explicit future-to-action cascade. After learning a one-step action model, we train the future predictor by decoding actions from its own predicted futures and backpropagating demonstration action error through the expert, with an optional future-latent reconstruction loss. The resulting objective evaluates the student-generated future through its downstream control consequences rather than relying on representation matching alone. DirectWAM achieves 97.1% average success across four LIBERO suites and 85.4% on RoboTwin 2.0, outperforming Flash-WAM at the same one-evaluation-per-stream budget. This advantage has also extended to three real-world tasks that we designed. Compared with the full iterative teacher, it reduces inference latency by $28.0\times$. 

---
# ClawMMU: Protection-Coherent Virtual Memory for Stateful LLM Agents 

**Authors**: Jinhao Hu, Bardia Mohammadi, Mofasshara Binte Rafique, Lars Henning Klein, Akhil Arora, Ashvin Goel, Laurent Bindschaedler  

**Link**: [PDF](https://arxiv.org/pdf/2610.28093)  

**Abstract**: AI agents keep long-term memory, much of which comes from web pages and tool results. Agent harnesses, the software that runs the model, do not record who may act on this memory, so an instruction hidden in a fetched page can be treated as the user's own instruction. We propose ClawMMU, a protection layer that labels every piece of memory with its owner, scope, origins, and authority. ClawMMU checks these labels whenever content enters the model's context, is written to storage, returns from storage, or drives a tool call. Its central rule is that information may rise in authority, but authority never lifts itself. Because the checks depend on where content came from, not on what it says, memory attacks become classical protection faults that the harness stops even when the model is fooled. 

---
# BAIT: Can LLMs Find What Biases a Reader? A Benchmark for Cognitive Bias Triggers in News 

**Authors**: Yousra Fettach, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2610.28092)  

**Abstract**: Persuasive writing exploits cognitive biases, recruiting a reader's mental shortcuts to shape belief, often without the reader noticing. We call the linguistic features exploiting such mental shortcuts cognitive bias triggers. As large language models (LLMs) increasingly mediate how people encounter news, summarizing it, ranking it, and answering questions over it, a natural question is whether they can recognize the features of a text that bias a human reader. We introduce BAIT (Bias Annotation In Text), a resource for detecting cognitive bias triggers in news. BAIT pairs a schema of nine triggers, each specified in an annotator codebook, with 50 English news articles annotated by trained annotators at the article and span levels and adjudicated into a gold standard. On this data we define two tasks, detecting which triggers a text contains and locating the spans that carry them. We then benchmark six frontier LLMs zero-shot and with three or ten in-context examples. The models approach human agreement on which triggers an article contains, but rarely recover its full set of triggers, and they reach less than half of human agreement in locating the passages that carry them. Moreover, they over-predict triggers that come with easy surface cues, and most of the passages they mark for these triggers are not triggers, yet carry the trigger's surface cue far more often than random text. These findings suggest that the evaluated LLMs respond to surface cues rather than to the mechanisms that make a passage manipulative. We release the schema, the codebook, standoff annotations, and the evaluation pipeline. 

---
# From Complex Histories to Simple Predictors: Predictive Dimension in Dynamic Graph Learning 

**Authors**: Hantong Feng, Duxin Chen, Wenwu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28090)  

**Abstract**: Existing continuous-time graph learning methods rely heavily on complex neural architectures to encode ever-expanding interaction histories. However, the future conditional distribution induced by complex histories can often be characterized by a low-dimensional predictive state manifold, implying that only a few predictive degrees of freedom are essential for future interactions. We therefore argue that continuous-time graph learning should identify low-dimensional states sufficient for future prediction rather than reconstruct the full history. To make this perspective measurable, we define and independently estimate the intrinsic predictive dimension, which quantifies the effective predictive degrees of freedom required by a given task. Specifically, we remove the history-independent marginal from the history-conditioned future distribution, decompose the resulting predictive residual operator, and estimate the minimum effective rank needed to preserve the predictive power of the full history. Experiments across datasets show that model performance saturation is largely determined by the intrinsic predictive dimension. Based on this finding, we propose PreDyG, a lightweight model whose capacity is matched to the intrinsic predictive dimension. PreDyG achieves superior predictive performance with only 0.07\% of the parameters of a representative complex baseline. 

---
# MoDDiT: Simple Token Routing in Diffusion Transformers 

**Authors**: Jarek Liesen, Matthew T. Jackson, Benjamin Ellis, Shimon Whiteson, Jakob N. Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2610.28086)  

**Abstract**: Real-time video generation, interactive world models, and on-device image synthesis are almost a reality. Diffusion Transformers (DiT) achieve excellent sample quality, shifting the bottleneck to the throughput of the models themselves. Natural images are inherently sparse, as large portions of an image are often background, contribute little to the scene content, and are easy to denoise. DiTs fail to exploit this sparsity, as they process image tokens statically and uniformly. We propose MoDDiT, a simple architecture modification that learns to route tokens through both MLP and self-attention operations. By routing tokens through both blocks independently, MoDDiT allows for dynamic compute allocation in MLPs that varies across tokens, images, and denoising timesteps. Using MoDDiT to train image generation models from scratch significantly reduces computational costs of training and inference, processing as few as 12.5% of the tokens. By increasing sparsity and model depth, MoDDiT reduces costs and improves generation performance over unrouted baselines. The learned allocation is interpretable, revealing how computation varies with visual content and noise level. We also use MoDDiT to train video generation models, where allocation correlates with motion, and to distill a dense bidirectional video model into a sparse autoregressive student. 

---
# PULSE: Personalized Opportunistic Learning for Wearable Devices 

**Authors**: Mattia Giovanni Campana, Dariush Salami, Hüseyin Yiğitler, Stephan Sigg  

**Link**: [PDF](https://arxiv.org/pdf/2610.28081)  

**Abstract**: Wearable devices increasingly support personalized health and well-being applications, but post-deployment data can differ substantially from those used for training, while purely local adaptation may be limited by the data available on a single device during real-world operation. Conventional federated and decentralized learning rely on recurrent communication and stable connectivity, whereas existing opportunistic approaches often exploit available encounters without adequately considering their utility, compatibility, or communication cost.
We propose PULSE, an event-driven framework for personalized opportunistic learning that treats each encounter as an online collaboration decision, selecting either no interaction or a promising nearby peer. Peer gradients are incorporated only when compatible with the local learning direction, while past interactions guide future peer selection. Across human activity recognition, fall detection, and stress recognition under synthetic and real contact traces, PULSE preserves or improves personalized classification performance, while reducing peer-assisted exchanges by 80-94% compared with opportunistic model averaging. On-device experiments further demonstrate its feasibility on wearable hardware: peer-assisted operations require limited energy, while the complete implementation occupies 149.8 KiB of RAM and 561.8 KiB of flash. The obtained results show that selective opportunistic collaboration can provide effective personalization while limiting communication and resource demands on constrained wearable devices. 

---
# Same Judgment, Different Yes: Measuring Multilingual Judgment Profiles in Language Models 

**Authors**: Haonan Huang, Yuwei Lu, Chengkun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.28066)  

**Abstract**: Multilingual evaluations often score a model's opening yes or no, a token that also carries model and language answering conventions. We ask when different ways of eliciting and reading an answer recover the same judgment, and when they measure an answer habit. We collect 1.1 million replies from eight models in six languages on up to 105 moral vignettes, varying the question's form, the answer constraint, and the reading of the reply, token or sentence, in both wordings (the FAR-6 battery). Story-level judgment profiles converge across the tested readouts and question forms, except for the yes/no token after a negative interrogative read under a preset mapping: on the same replies, token and sentence profiles agree at a median r = 0.67 across cells, while sentence profiles agree with free-sentence profiles at 0.94, a known linguistic variability. With each language's story consensus removed, a model's profile is nearest to its own in another language for 39 of 48 model-language cells under proposition labels and 15 under free sentences, against 5 expected by chance. Particle, position, and named-term preferences follow their own patterns. A model can keep its judgment and change its yes: scoring that token requires establishing which proposition it affirms. 

---
# Quantifying the Threat: A Bayesian Probabilistic Framework for Heavy Metal Health Risks among Children in the Densu Basin, Ghana 

**Authors**: T. Ansah-Narh, G. Y. Afrifa, J. B. Tandoh, E. N. N. Nortey, D. Adjei, R. Opoku-Sarkordie, R. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2610.28064)  

**Abstract**: Groundwater in Ghana's Densu River Basin remains a key domestic water source, but multi-metal contamination threatens children's health. Previous deterministic hazard index assessments provide limited treatment of uncertainty, co-exposure, and spatial variability. This study applies a Bayesian probabilistic framework for non-carcinogenic risk assessment using 94 groundwater samples and measurements of Fe, Mn, Ni, Pb, Cd, and As. A Bayesian regression model propagates parameter uncertainty and accounts for contaminant co-occurrence. Adaptive Hamiltonian Monte Carlo is used for posterior inference, while inverse distance weighting generates spatially continuous risk surfaces. Pb, As, and Cd dominate cumulative risk. Posterior predictive results indicate that approximately 31% of sampling sites have a probability greater than 0.5 of exceeding a hazard index (HI) of 1. The risk surface is highly heterogeneous, with a posterior mean of 0.372 and a standard deviation of 0.416. High-risk zones often reflect moderate co-exposure rather than extreme contamination by a single metal. The framework provides uncertainty-aware, spatially explicit information to support groundwater monitoring, risk mitigation, and public health planning in data-limited regions. 

---
# Auditing Where Concept Bottleneck Accuracy Comes From on CUB 

**Authors**: Sascha Saralajew, Eleonora Poeta, Eliana Pastor, Ammar Shaker  

**Link**: [PDF](https://arxiv.org/pdf/2610.28059)  

**Abstract**: Concept Bottleneck Models (CBMs) promise predictions that humans can inspect and correct through concepts such as "curved beak". Yet on CUB-200-2011, their standard benchmark, progress is measured like that of ordinary classifiers: by task and concept accuracy. Neither reveals whether the label is predicted from the exposed concept scores, or whether a concept is supported by visual evidence of that concept rather than by co-occurring features such as cues to the species as a whole. We audit where the accuracy of CBMs and Concept Embedding Models (CEMs) on CUB comes from. The CUB annotations are too noisy to explain it: class-level training labels agree with per-image annotations on only ~82% of image--concept pairs, and classifiers trained on per-image annotations reach only ~37% species accuracy. Leakage probes show that CEM predictions cannot be reconstructed from their concept scores; the missing class information sits in internal embeddings. We identify four design axes through which class information can enter concept scores---receptive field, shared versus concept-specific representations, norm-dependent scoring, and task supervision. An ablation shows that most of it enters through non-local features: restricting a ResNet50 backbone to local BagNet patches lowers the accuracy recoverable from concept scores from 72% to 43%, close to what ground-truth annotations support, while joint training raises it again at the cost of concept accuracy. Localization and occlusion show that locality makes grounding measurable but does not guarantee it: under class-level labels, local concept scores use the head as a proxy for body, wing, and tail concepts, and CEM concept scores barely respond to occluding their supporting part. Task and concept accuracy should thus support, not drive, the evaluation of CBMs, and given the noisy annotations, CUB accuracy gaps should not be overinterpreted. 

---
# Slots Are Not Speakers: Training-Free Speaker Linking for Frozen Streaming Diarizers 

**Authors**: Aditya Srinivas Menon, Kumarmanas Nethil, Kavya Manohar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28053)  

**Abstract**: This paper presents a causal, training-free linker that allows a frozen streaming diarizer to track more speakers than it has output slots. Streaming end-to-end diarizers such as Streaming Sortformer emit a fixed number N of concurrent speaker slots. Above N speakers, they keep detecting speech but reuse slots across people, because N bounds how many speakers the model can express at once, not how many it will meet. The linker uses only the host's slot posteriors and the audio. It keeps the frames where one slot is active, splits each slot where the speaker changes, embeds each piece with an off-the-shelf speaker encoder, and matches it against a bank of speakers seen so far. It only relabels, so missed speech and false alarm are identical to the host's and every gain is a reduction in speaker confusion. On 1418 sessions from eight corpora, it reduces the 4-slot host's DER above its cap from 36.8% to 22.6%, and barely moves it below the cap. The same constants transfer to the host's 8-slot successor and to two streaming EEND hosts from a different model family. On courtroom hearings with up to 23 speakers it reduces cpWER from 82.7% to 24.4%. 

---
# Assessing Representation Semantics Under Language Pretraining in Fine-Grained Classification 

**Authors**: Md Raisul Kibria, Sébastien Lafond  

**Link**: [PDF](https://arxiv.org/pdf/2610.28051)  

**Abstract**: Understanding how supervision signals shape semantic visual representations remains challenging, particularly in fine-grained, low-data settings. Standard evaluations of vision models rely on downstream accuracy, which often fails to capture structural differences in learned representations. We propose a compact explainability-driven framework for comparing encoders trained with classification-only and caption-generation supervision on CUB-200. Using a controlled two-stage protocol in which both encoders are ultimately optimized for the same classification task, we enable a direct comparison of the resulting representations. Saliency maps are generated using attention rollout and gradient-weighted rollout and evaluated with five metrics covering faithfulness (positive and negative perturbation) and localization (keypoint emphasis, background attribution, and Earth Mover's Distance). Across random seeds and explanation methods, caption supervision yields representations that are more sensitive to the removal of salient regions, better aligned with annotated object parts, and less reliant on background regions, whereas classification supervision produces representations that are slightly more robust to the removal of low-saliency regions. An ablation study on the Oxford-IIIT Pet dataset shows consistent trends across key metrics. These results demonstrate that the proposed framework makes supervision-induced representational differences observable and quantifiable beyond downstream accuracy alone. More broadly, it provides a systematic basis for assessing learned representations and comparing alternative training and transfer-learning strategies. 

---
# On-Policy Transition Distillation 

**Authors**: Runyang You, Yongqi Li, Tiezheng Yu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28048)  

**Abstract**: On-policy distillation (OPD) consolidates expert capabilities into a single student by matching teacher and student final-layer distributions on student-generated rollouts. We investigate whether depthwise prediction transitions, the changes in predictions between layers, can provide additional task-relevant supervision. Our diagnostic study finds that stronger teacher--student agreement in intermediate transitions is associated with higher answer correctness within questions, with correct responses exhibiting lower mean transition KL across all measured intermediate layer pairs. These observations motivate On-Policy Transition Distillation (OPTD), which augments final-layer distribution matching by aligning teacher and student layer-to-layer log-probability increments on student-generated rollouts. Evaluations across three backbones and eight benchmarks spanning mathematics, coding, and instruction following show that OPTD achieves the best or tied-best overall performance in both dual- and three-domain distillation. In the three-domain setting, OPTD recovers $\approx 90\%$ of the overall teacher--student capability gap, compared with $\approx 75\%$ for OPD. 

---
# Learning Tactile-Aware Hand Representations from Human Hand-Object Interaction Data 

**Authors**: Yeonguk Yu, Jaehong Kim, Byungok Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.27995)  

**Abstract**: Tactile representation learning for dexterous manipulation often requires substantial robot-side data collection, limiting its scalability across different hands and sensors. We explore whether large-scale human hand-object interaction (HOI) data can provide a useful prior for robotic tactile learning. To this end, we propose HuTact, a framework that pretrains tactile-aware hand representations from human HOI data without requiring real tactile signals during pretraining. HuTact transfers the interaction structure learned from human hand motions to robotic tactile observations and can be further adapted with a small amount of unlabeled robot interaction data. We evaluate HuTact on tactile perception and manipulation tasks, including grasp prediction, slip detection, object classification, and tactile-conditioned policy learning. Across these tasks, human HOI pretraining improves data efficiency and provides a strong initialization for downstream learning. Notably, with HOI pretraining followed by midtraining on unlabeled robot play data, HuTact achieves 88.26% Macro-F1 on grasp prediction using only 25\% of the labeled downstream data, comparable to training from scratch on the full labeled dataset (88.23%). These results suggest that human interaction data can serve as a scalable complementary source for robotic tactile representation learning. 

---
# Recast the Past: Stable Long Video Generation with Full Context Memory 

**Authors**: Yukuo Ma, Cong Liu, Junke Wang, Junqi Liu, Haibin Huang, Zuxuan Wu, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27986)  

**Abstract**: Long-horizon video generation requires retaining an ever-expanding history, but conditioning on its growing token sequence is costly. Existing methods restrict or compress historical context, potentially losing information needed for future generation. This work proposes ReCam, which groups local spatiotemporal blocks of historical latents into vectors and projects them directly into the high-dimensional hidden space of a Diffusion Transformer. The model learns to use the resulting memory tokens through native self-attention, reducing token count while retaining block-level access to the full history. To mitigate temporal drift during autoregressive rollout, we further introduce dual-pointer denoising with separate noise schedules for recent context and target frames. Among the evaluated methods, ReCam achieves the highest VBench-Long overall score of 80.1 and competitive static-scene revisitation performance on R2M-Bench. It also leads on all four consistency metrics for off-screen subject reappearance, including clothing consistency of 4.09 versus 3.03 for the best baseline. 

---
# APRIL: Ambiguity-Aware Expert Routing for Respiratory Sound Domain-Incremental Learning 

**Authors**: Xinhong Li, Weiping Tu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27942)  

**Abstract**: Automated respiratory sound classification enables continuous, non-invasive respiratory assessment, yet long-term deployment remains challenging due to domain shifts from evolving recording devices and acquisition periods. When privacy and governance regulations prohibit replaying historical data, sequential learning easily overwrites previously acquired knowledge. Preserving dedicated domain experts prevents parameter interference, but selecting the correct expert at inference is difficult under overlapping acoustic distributions and absent domain labels. To resolve this dilemma, we propose APRIL, an exemplar-free domain-incremental framework for RSC. Built on a shared frozen BEATs backbone, APRIL sequentially inherits and isolates low-rank experts to enable new-domain adaptation without altering earlier models. A multi-level prototype router retrieves the top-two candidate experts and dynamically selects the final model by integrating domain-routing posteriors with classification confidence, adaptively weighted by matching ambiguity. We establish standardized device- and temporal-incremental benchmarks on the ICBHI and SPRSound datasets across forward and reversed domain orders. Evaluated over five random seeds, APRIL consistently achieves the highest final-session average Score among compared continual learning approaches, reaching 64.4% and 65.4% on ICBHI, and 83.1% and 82.2% on SPRSound under forward and reversed sequences, respectively. Crucially, average forgetting remains under 1.0% across all protocols, demonstrating robust domain adaptation and long-term knowledge retention without data replay. Overall, APRIL offers a reliable and privacy-preserving paradigm for deploying continuous respiratory sound classification across real-world healthcare settings. 

---
# Polarity Blindness in LLM Safety: A Systematic Study of Negation, Antonym Substitution, and Adversarial Framing 

**Authors**: Diego Miguel Lozano, Erik Derner, Juan Antonio Pérez-Ortiz, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2610.27908)  

**Abstract**: Negation remains a persistent challenge for language models across NLP tasks and modalities. We show that this weakness can undermine LLM safety guardrails through a structural vulnerability we term polarity blindness: the failure to recognize harmful intent after polarity-altering linguistic transformations. We study two forms of polarity inversion: grammatical negation and antonym replacement, and use synonym replacement as a polarity-preserving control. We further test whether an adversarial textual wrapper amplifies these failures. Using a 260-prompt benchmark across 10 state-of-the-art LLMs and 3 languages (English, Spanish, Czech), stratified by harmfulness potential and evaluated on a validated 0-5 harm scale, we find that polarity blindness affects most evaluated models, with mean harm scores reaching 4.12 for the most vulnerable model. The effect persists across typologically differing negation systems, suggesting that polarity blindness reflects a broader limitation in semantic robustness rather than a language- or grammar-specific failure. 

---
# CollabSciReM: Aligning Language Models for Scientific Conversations via Science-Aware Reward Decomposition 

**Authors**: Sasi Kiran Gaddipati, Gollam Rabby, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2610.27901)  

**Abstract**: Large Language Models (LLMs) are increasingly used as research collaborators for hypothesis refinement, methodological planning, and technical analysis across multi-turn interactions. In this setting, a useful assistant must be accurate, methodologically justified, logically consistent, and token-efficient. Current alignment approaches rarely achieve all of these together. Instruction-tuned models gain conversational quality through verbosity, and we show that science-agnostic general collaborative rewards compress responses at the expense of scientific rigor. We propose CollabSciReM, a goal-conditioned reward framework that decomposes the session-level reward into conversational quality, scientific rigor, and structural efficiency. We also introduce SciQuality, a pairwise evaluation of multi-turn research conversations grounded in 300 held-out scientific papers and scored on six dimensions: methodological soundness, justification, logical consistency, factual correctness, novelty, and responsiveness. Fine-tuned on 8.62K synthetic research conversations, CollabSciReM is preferred on SciQuality over a science-agnostic reward baseline trained on the same data for all four Llama and Qwen models, including a 65.3% win rate at matched response length. For the 7B and 8B models, CollabSciReM outperforms the science-agnostic reward baseline on the primary metric in 8 of 10 single- and multi-turn benchmark settings. In a blind study, domain experts rated CollabSciReM significantly higher on novelty than the general-reward variant and reported fewer hallucinated responses than both the general-reward variant and the base model. 

---
# The Shape of the DPO Hyperparameter Landscape: What to Measure, How It Moves with Scale, and When Not to Transfer It 

**Authors**: Hannan Mahadik, David Salinas, Jacek Golebiowski, Aaron Klein  

**Link**: [PDF](https://arxiv.org/pdf/2610.27895)  

**Abstract**: Direct preference optimisation (DPO) is widely used for language model post-training, but its performance depends strongly on the choice of its hyperparameters, such as the learning rate and the reference-policy regularisation parameter $\beta$. To better understand the DPO hyperparameter landscape, we conduct extensive experiments using a dense hyperparameter grid across different model scales from the same family. This allows us to analyse how the optimal DPO hyperparameters evolve with model size and the number of update steps, with intermediate checkpoints providing an inexpensive signal to accelerate the hyperparameter optimisation process. Furthermore, by evaluating our models on both static and dynamic benchmarks, we identify which benchmarks provide the strongest signal for hyperparameter selection. Finally, we investigate how different state-of-the-art hyperparameter optimisation strategies perform in this setting. Our experiments show that the optimal learning rate varies with model size and that hyperparameters are not transferable across training datasets. We also find that validation loss is a poor proxy for hyperparameter optimisation, as hyperparameters that minimize validation loss only weakly correlate with optimal hyperparameters for downstream benchmarks. Therefore, DPO hyperparameters need to be tuned specifically for the target model, dataset, and evaluation objective. We also show that short candidate training runs can substantially reduce the cost of this selection. We release all our training configurations and evaluation results to support reproducible analysis of selection quality and training cost. 

---
# Physical-Variable-Informed Deep Learning for Robust Diagnosis in Industrial Fluid Transport Systems 

**Authors**: Yu Sha, Junqi Tao, Dixin Zhou, Horst Stoecker, Domagoj Vnucec, Andreas Widl, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27891)  

**Abstract**: Industrial fluid transport systems are monitored through heterogeneous sensing streams, but robust fault diagnosis remains difficult because response signals vary with operating conditions and differ from physical variables in sampling rate, observation duration and data granularity. To address these challenges, we propose a physical-variable-informed deep learning framework (PhyNet) for robust fault diagnosis under real-world sampling heterogeneity. PhyNet treats physical variables as global operating constraints and response signals as local diagnostic evidence. Its physical-to-response cross-attention mechanism connects sequence-level physical-variable tokens with a learnable global response token, enabling interaction without strict temporal alignment. Across four benchmarks, including three real-world cavitation datasets and one public two-phase-flow dataset, PhyNet achieves the best performance across all evaluated metrics. Class-wise analyses, together with attention and masking analyses, show improved discrimination of ambiguous states and indicate that physical variables meaningfully constrain local response interpretation. These findings demonstrate a deployment-oriented approach to robust diagnosis in heterogeneous industrial environments. 

---
# Hallucinations of Large Language Models in Cross-Language Code Translation: Phenomena and Mitigation 

**Authors**: Yali Du, Yunji Zhang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.27868)  

**Abstract**: Code translation converts source code from one programming language to another while preserving its functionality. Although Large Language Models (LLMs) have improved code translation, their outputs still contain hallucinations: generated code that conflicts with the source program, violates target-language facts, or introduces contextually inconsistent dependencies. These failures are especially harmful in translation because small semantic deviations can silently change program behavior. Existing hallucination studies mainly focus on natural-language generation or code generation, leaving the forms, distribution, and mitigation of hallucinations in cross-language code translation insufficiently characterized. This paper presents a systematic empirical study of LLM hallucinations in code translation. Across three benchmark datasets, six language pairs, and seven LLMs, we construct a taxonomy with three major categories: Semantic Consistency Conflicts, Factual Knowledge Conflicts, and Project Context Deviations. We manually annotate 13,919 hallucination cases and analyze how hallucination types vary across models, datasets, and language directions. Guided by the taxonomy, we further propose HaluCodeTran, a mitigation framework that combines BM25-based retrieval of relevant translation examples with taxonomy-aware reflective refinement. The study provides an execution-tested characterization of code-translation hallucinations and a practical baseline for building more reliable LLM-based code migration tools. 

---
# VISH-ASR: Viterbi-Guided Selection of Hotwords for Speech-LLM Contextual ASR 

**Authors**: Xiaoyu Gu, Yu Xi, Chencheng Wang, Zonghan Zhou, Haoyu Li, Jing Peng, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27867)  

**Abstract**: Prompt-based contextual ASR requires selecting a compact hotword subset from large dynamic inventories, yet existing pre-selection methods rely either on discrete ASR hypotheses that may discard acoustic evidence or on separately trained retrieval models. We propose Viterbi-guided Selection of Hotwords (VISH), a training-free, plug-and-play framework that directly verifies candidate-specific acoustic support over frame-level CTC posteriors. VISH formulates pre-selection as constrained posterior search; M-VISH makes large-list verification scalable through frame-synchronous shared-prefix Viterbi search with bounded top-K pruning, while Bi-MVISH exploits complementary backward evidence for robust verification. Across five benchmarks, M-VISH achieves up to 11.7% absolute improvement in Best-F1 over a strong learned-retrieval baseline and reduces candidate-search latency by 75.2% at 2,000 candidates relative to hypothesis-based filtering. These results demonstrate the effectiveness of posterior-level verification for accurate and scalable hotword pre-selection without additional model training. 

---
# SMART: Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics 

**Authors**: Xiaoyu Gu, Chencheng Wang, Fuyang Li, Yu Xi, Tao Liu, Jiaoqi Guo, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27820)  

**Abstract**: Word error rate (WER) assigns equal cost to lexical edits despite their different semantic consequences, while overall semantic scores cannot identify which individual errors alter the intended meaning. We formulate ASR semantic evaluation as context-aware, segment-level consequence assessment and propose Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics (SMART), a framework for traceable semantic evaluation and targeted ASR correction. SMART maps reference-hypothesis discrepancies into coherent segments and applies a structured contextual rubric to assign explicit consequence labels. These labels support Semantic Impact Error Rate (SIER), which quantifies consequential errors at utterance, chunk, and lexical levels. Against majority-vote human labels, SMART achieves 0.843 F1 versus 0.745 for direct LLM judgment. Evaluation across multiple LLMs further demonstrates that SMART is relatively insensitive to the choice of evaluation model, while MER and SIER reveal different aspects of ASR quality. A separate target-reference-free rubric guides correction using the target hypothesis and surrounding ASR hypotheses, outperforming a matched w/o Rubric baseline on both datasets, with further absolute reductions of 0.85-0.98 percentage points in MER and 1.18-1.65 percentage points in SIER_lex. 

---
# RAG-NIMBF: A Non-Invasive, Modular Benchmarking Framework for Automated RAG Configuration Analysis 

**Authors**: Niclas Hart, Nicolas Weeger  

**Link**: [PDF](https://arxiv.org/pdf/2610.27816)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding the outputs of large language models (LLM) in external knowledge. However, practitioners are faced with a vast configuration space comprising chunking, embedding, retrieval, reranking, prompting and generation models, and there is limited guidance on how these components interact. We present RAG-NIMBF, a modular benchmarking framework that attaches to existing RAG pipelines as a non-invasive evaluation layer. It evaluates the configuration space through controlled ablation with end-to-end experiment tracking. The evaluation combines LLM-based answer quality scores, retrieval correctness measures, and runtime indicators. This study validates the framework across multiple generation models on the SQuAD dataset. This process reveals non-obvious interactions and demonstrates how the experiments are tracked. The findings indicate that the retrieval balance of relevance and diversity substantially enhances the relevance and coverage of retrieved context in comparison to a conventional similarity search, with MMR raising nDCG@5 by +4 to +11 percentage points across the swept models. More detailed prompt instructions meaningfully increase how faithfully generated answers stick to the retrieved evidence, especially for capable and reasoning models (+10 to +13pp on Qwen3.6-27B, GPT-4o-mini and Qwen3.5-think, but negligible on the 20B generator). Overall, RAG-NIMBF transforms the configuration space of RAG into reproducible, comparable experiments. By attaching to existing production pipelines in a non-invasive way, it enables practitioners to systematically evaluate and improve deployed systems without disrupting existing workflows. This replaces ad hoc choices with evidence-based decisions. 

---
# SLM-SS: Token-Based Speech Separation with Speech Language Models 

**Authors**: Chenda Li, Tianhua Li, Yiming Chen, Wei Wang, Jiahe Wang, Xin Zhou, Yanmin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.27785)  

**Abstract**: Speech separation (SS) has witnessed substantial progress in recent years with the advent of neural network-based methods, achieving notable improvements on signal-level evaluation metrics. Existing speech separation systems are mostly optimized with signal-level objectives, and often yield outputs that are acceptable in waveform metrics but suboptimal for intelligibility-sensitive downstream tasks. To address this gap, we propose SLM-SS, a token-based generative speech separation framework that models multi-talker mixtures with speech language models. We first discretize clean speech into multi-codebook token sequences with neural codecs and serialize multi-speaker targets via Serialized Output Training (SOT). An encoder-decoder autoregressive speech language model predicts the zero-order sequence to establish speaker ordering and coarse linguistic content, and a non-autoregressive module then refines higher-order residual codebooks for efficient reconstruction. We further introduce a per-talker non-autoregressive strategy that simplifies high-order modeling and improves the stability. On Libri2Mix, SLM-SS with per-talker NAR achieves $6.92\%$ WER, approaching codec-only reconstruction. The WavTokenizer variant eliminates NAR refinement and achieves the highest mean subjective quality score among evaluated separation systems. Its UTMOS and DNSMOS advantages over discriminative baselines persist on unseen noisy and Chinese mixtures without fine-tuning. 

---
# RELAY: Scaling Historical Build Restoration through Rule Reuse and Agentic Repair 

**Authors**: Olivier Nourry, Yoshiki Higo  

**Link**: [PDF](https://arxiv.org/pdf/2610.27784)  

**Abstract**: Vulnerability benchmarks and software engineering research depend on rebuilding historical software. Yet builds rely on external resources, such as base images and packages, that change independently of the source code. In this study, we examine this environment decay on 14,159 OSS-Fuzz cases. We find that 69.7% no longer complete reconstruction and verification, and that most failures share a small number of recurring failure symptoms. A recurring failure can therefore be diagnosed once and its repair reused. We present Relay, a hybrid framework that uses an LLM to derive and validate deterministic repair rules from representative failures, reuses them without model calls, and escalates cases unresolved by rules to a per-case agent. We evaluate Relay in human-guided and fully automated rule-derivation campaigns and compare it with a per-case agent approach. We find that rule derivation can be fully automated, and that reusable rules restore as many builds as per-case agents with fewer build executions. However, agents restore cases that the rules cannot. Our results show that a hybrid approach that applies rules first and escalates the remaining cases to an agent restores more builds than either approach alone and at a lower cost than using agents alone. 

---
# Radar-Based 3D Human Pose Estimation via Motion-Aware Pose Initialization and Observation-Guided Refinement 

**Authors**: Woo-Jin Jung, Chan-Bin Lim, Seung-Hyun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27774)  

**Abstract**: 3D Human Pose Estimation (3D HPE) is used in indoor environments, such as smart homes, to analyze human movement and monitor human states. Millimeter-wave (mmWave) radar is suitable for this task because it is robust to lighting variations and does not capture images of human appearance. However, radar observations often provide insufficient information about certain body parts or contain overlapping reflections from multiple body parts, limiting accurate 3D joint localization. To address these limitations, we propose the Initialization-to-Refinement Pose Framework (I2R-Pose), comprising two stages: Motion-Aware Pose Initialization and Observation-Guided Refinement. In the first stage, joint-wise features are learned by incorporating temporal relationships across consecutive radar observations and human skeletal structure to compensate for insufficient joint information and estimate an initial 3D pose sequence. In the second stage, separately extracted radar observation features predict joint-wise positional corrections that refine the initial pose and generate the final pose sequence. The framework supports individual and fused inputs of time-Doppler/time-range (TD/TR) maps and radar point clouds (PCs). To evaluate performance across indoor activities, we construct the Radar Motion and Spatial Information for 3D Human Pose Estimation (RaMoS-Pose) dataset, providing synchronized TD/TR maps, radar PCs, and 3D joint ground truth (GT). Evaluations on RaMoS-Pose and the publicly available MVDoppler-Pose dataset show that I2R-Pose reduces mean per-joint position error (MPJPE) by 39.50% on MVDoppler-Pose and by 15.13% on RaMoS-Pose with fused inputs, compared with the best-performing existing methods evaluated on each dataset. 

---
# How Much Do $\ell_1$ Attacks Need Concentrated Perturbations? 

**Authors**: Melika Shirian, Kianoosh Vadaei, Arshia Hemmat  

**Link**: [PDF](https://arxiv.org/pdf/2610.27773)  

**Abstract**: A global $\ell_1$ budget limits how much an adversary may change an image, not how that change is distributed. We ask how much of an $\ell_1$ attack's observed success remains when large coordinate changes are restricted. We add a per-coordinate cap to the budget, enforce the joint constraint by exact in-loop projection, and sweep the cap for APGD-$\ell_1$, SLIDE, and FMN-$\ell_1$ without altering their objectives. Across 18 standard-model settings, caps of at most $32/255$ retain a median $97\%$ of attack success relative to each method's native uncapped baseline, while median peak magnitude falls by $80\%$ and median effective support nearly triples. At matched realized $\ell_1$, capped perturbations remain structurally distinct; separate ImageNet comparisons also show lower LPIPS and DISTS at selected caps. Direct in-loop constrained optimization preserves up to $88$ percentage points more attack success than post-hoc clipping, highlighting the importance of constraint enforcement. Suitable caps nevertheless vary across attacks and models, and adversarially trained classifiers exhibit more heterogeneous concentration sensitivity. These results establish perturbation concentration as a complementary, model- and attack-dependent axis of $\ell_1$ attack evaluation. 

---
# S$^2$-MAS: A Society-Scale Multi-Agent Simulation Framework with 500 Million Household Persona Agents 

**Authors**: Yang Li, Di Wan, Yike Wang, Chujun He, Zongsi Xu, Xiyu Chen, Sheng Zhong, Deng Binjie, Zhenwei Guo, Hou Jun, Kunhan Guo, Weihao Lin, Tuo Zhou, Hai Liu, Ye Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.27746)  

**Abstract**: Simulating economic responses at society scale requires agents whose decisions reflect household circumstances and the effects of earlier actions. We introduce S$^2$-MAS, a society-scale multi-agent simulation framework built on 500 million synthetic household personas grounded in real-survey data. Its Genese engine separates decisions from state updates, while reusable financial scenarios and application ledgers track consumption, cash, debt and payments. FinL-mlx, a compact decision model trained with SFT + RLCD on 2.93 million examples, achieves 74.55% accuracy and 38.37% question-macro F1 in consumer financial-choice scenarios, outperforming Jev 1.13 and GLM-5.3-Flash on both measures. We evaluate eight financial task families and simulate 12 million distinct household agents in paired purchase-funding conditions. Jointly raising purchase goals and approved credit limits increases consumption, with responses differing across age and income groups. Education, employment and household-size differences persist after age--income standardization and mainly reflect transitions from deferral to purchase. Asset and debt groups differ in cash drawdown, repayment and closing debt. A 24-month panel varies purchase goals under a common credit product, with repayment and arrears updating subsequent offers. S$^2$-MAS links heterogeneous household decisions to evolving consumption, cash and repayment paths, supporting economic experiments across household groups, population scales and time horizons. 

---
# Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates 

**Authors**: An-Sheng Jhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27745)  

**Abstract**: Large language models (LLMs) are increasingly used as judgment assistants and as stand-in participants in psychology experiments, yet little is known about how much their probability estimates move in response to a single new piece of information. We pair premise-hypothesis items from $\delta$-SNLI with one strengthener and one weakener each, elicit the prior and the posteriors in separate calls, and measure the directional movement s: the mean posterior change signed by the annotated direction, so that the prior cancels out. We vary whether the update is stated ("Here is something else you learned") or reported ("Someone said that"), and whether the model gives one probability or five weighted estimates. We test seven configurations: two local models (Qwen3-8B, OLMo-2-7B; 240 pairs) and five configurations run through agent command-line harnesses (120 pairs), with hypotheses and analysis rules fixed before the formal runs (not publicly registered). Under stated, direct elicitation all seven move in the annotated direction (s = 18-28 percentage points). Reporting lowers s in all seven point estimates (significantly in six, by 2-5 points) and raises the share of reverse movements in all seven: the effect is consistent but small. Five weighted estimates lower the share of responses at $\leq$5% or $\geq$95% in all seven, but in the two local models they also reduce s by 43% and 68%, whereas in the harnessed configurations the point estimates of the change lie between -3.7 and +0.5 points. An elicitation format that reduces extreme values thus need not preserve sensitivity to evidence; the two should be tested separately. Because $\delta$-NLI annotates only direction, we make no claim about how large an update should be. Code, data and the frozen plan: this https URL 

---
# Anytime-Valid Runtime Monitoring of Learned Barrier Certificates 

**Authors**: Zhipeng Cao, Taoran Wu, Weiru Cui, Luke Ong, Bai Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.27726)  

**Abstract**: Learned barrier certificates provide a principled mechanism for establishing safety of stochastic systems, but their validity may deteriorate during deployment when the underlying dynamics differ from those used for training. We propose an anytime-valid runtime monitor that audits a learned barrier certificate using observed closed-loop trajectories. The monitor is intended to detect loss of statistical support for the deployed barrier condition; it does not by itself guarantee detection before unsafe entry or safety of a subsequent intervention. Specifically, the conditional barrier-drift condition is formulated as an episode-wise sequential hypothesis test, and observed barrier increments are transformed into a nonnegative test supermartingale, yielding finite-sample false-alarm control under arbitrary stopping times. Inspired by finite-horizon safety verification, we further introduce a deterministic temporal shaping of the monitoring boundary, making the monitor more conservative early in a prescribed deployment horizon while preserving anytime-validity. The monitor is designed to detect statistical evidence that the safety certificate is no longer consistent with the observed dynamics, rather than to predict unsafe entry. To assess the monitor empirically, we evaluate empirical alarm behavior under valid and invalid certificates. Code and data are available at this https URL. 

---
# World2HOSI: A Multi-Expert Harness for Long-Horizon Human-Object-Scene Interaction Generation 

**Authors**: Jiushu Li, Haoran Wang, Yumeng Liu, Zhiyang Dou, Yuan Liu, Yuexin Ma, Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.27725)  

**Abstract**: Long-horizon human-object-scene interaction (HOSI) is central to embodied AI, animation, and game development. Existing human-object interaction (HOI) and human-scene interaction (HSI) models excel in their respective domains, but none alone covers tasks requiring multiple interaction skills. Training unified HOSI models is hindered by scarce and costly data. We introduce World2HOSI, a multi-expert orchestration harness that generates continuous HOSI data from 3D Gaussian Splatting (3DGS) scenes and natural language instructions. Our key insight is that long-horizon HOSI hinges not on a stronger single model but on orchestration. Given a shared, structured scene state and an orchestration harness grounded in it, existing experts can jointly accomplish tasks beyond the reach of any of them alone. World2HOSI realizes this insight as a closed loop around a persistent scene state built from decoupled background and object assets. A harness plans action stages over this state, invokes experts via their native conditioning interfaces, and writes each action's effects back, so that every invocation is grounded in the evolving world. To reduce physical artifacts, bounded residual optimization and contact-driven physics simulation further ensure the physical plausibility of human motion and object manipulation, respectively, while preserving interaction intent. Experiments and a perceptual user study show that World2HOSI achieves the best task completion and interaction quality on the evaluated HOSI tasks, while generating more diverse data. 

---
# RLSplat: Resource-Linear Budget Control for Feed-forward 3D Gaussian Splatting 

**Authors**: Injae Kim, Minseok Joo, Jihwan Park, Sehyung Kim, Minseong Bae, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27712)  

**Abstract**: Feed-forward 3D Gaussian Splatting (FF3DGS) reconstructs 3D scenes from images in a single forward pass, making it well-suited to applications that must operate within limited resource budgets. To further improve its practicality, recent budget-controllable FF3DGS methods allow the number of Gaussians to be adjusted at inference, so that the output meets a given storage limit or rendering-latency constraint. However, these methods overlook the cost of inference itself: to produce a representation with the target number of Gaussians, they first predict a dense set of Gaussians and then prune or select from it, requiring nearly constant latency and peak memory regardless of the budget. We therefore propose RLSplat, a resource-linear budget-controllable FF3DGS framework whose representation size, inference latency, and peak memory all decrease linearly as the Gaussian budget is reduced. Instead of pruning or selection, RLSplat learns a Gaussian allocation policy that distributes the budget across local scene regions, so that exactly the requested number of Gaussians is generated. These Gaussians are then refined through resource-linear Gaussian decoding, in which each Gaussian attends only to nearby Gaussians found via an efficient hierarchical neighbor search, so that the decoding cost decreases linearly with fewer Gaussians. In zero-shot evaluation, RLSplat outperforms prior budget-controllable methods at every budget, surpassing the strongest baseline even with 4$\times$ fewer Gaussians, while running up to 4.1$\times$ faster with 9.6$\times$ less peak memory at low budgets. 

---
# Beyond Calibration: Temporal LLM Confidence Assessment Across Languages and Local Settings 

**Authors**: Sahil Kale  

**Link**: [PDF](https://arxiv.org/pdf/2610.27702)  

**Abstract**: Large language models are expected to estimate their own uncertainty, with confidence signals used to guide retrieval, adaptive compute allocation, routing, and selective abstention. We investigate whether confidence signals measured at different temporal stages of problem solving encode distinct forms of epistemic information and remain reliable beyond English-centric settings. Inspired by classical meta-memory theory, we distinguish anticipatory pre-solve confidence (Feeling-of-Knowing, FOK) from retrospective post-solve confidence (Judgement-of-Learning, JOL). We evaluate these signals across two frontier LLMs, multiple reasoning domains, translated Marathi benchmarks, and a newly constructed benchmark of genuinely local Marathi factual questions. Across settings, the cheaper pre-solve signal remains weakly calibrated, while post-solve confidence generally provides stronger epistemic alignment despite its additional inference cost. Confidence behaviour transfers surprisingly well under direct language translation, with calibration trends largely preserved between English and Marathi, but degrades substantially on local Marathi knowledge, suggesting that linguistic transfer alone is insufficient for region-specific information needs. Finally, hallucinated responses maintain persistently elevated confidence after answer generation, highlighting important limitations of current LLM self-assessment mechanisms. 

---
# VLA-Retain: Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs 

**Authors**: Borui Xu, Han Chen, Bingsheng He, Ming Wu, Jason Zeng, Michael Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2610.27698)  

**Abstract**: Vision-language-action (VLA) models transfer knowledge from large-scale pretraining into robot control, but their manipulation performance remains sensitive to changes in observation conditions, which recent work addresses by fine-tuning the visual module on target-domain data. However, it remains unclear how such adaptation affects robustness beyond the target condition, as existing robustness benchmarks only evaluate models without fine-tuning and adaptation studies only report the target gain. Measuring this cross-dimension effect requires separating target gains from off-target changes, isolating adaptation effects from architectural differences, ruling out generic forgetting, and comparing tuning choices. Therefore, we introduce VLA-Retain, a benchmark that compares five open VLA models, tuned on Camera-view or Sensor-noise data, with their own base checkpoints on LIBERO-Plus dataset spanning seven perturbation dimensions. Our study reveals four findings. First, targeted vision tuning is not local, and its target gain does not predict retention. Mean off-target gain is negative in 8 of 10 model--domain configurations. Under Camera-view tuning, X-VLA gains $69.4$ percentage points on the target but loses $4.6$ points on average elsewhere, whereas OpenVLA-OFT gains $39.3$ and $1.4$ points, respectively. Second, these regressions are not explained only by performance loss on the same tasks without perturbations. Third, broader training coverage raises overall gain over target-specific training but does not eliminate dimension-level regressions. Fourth, tuning strategies are model-dependent, as DELTA-style feature preservation raises X-VLA's mean off-target gain to $+1.2$ points but lowers OpenVLA-OFT's to $-23.1$. Overall, these findings show that target-domain success alone is insufficient and highlight cross-dimensional retention as a key criterion for targeted VLA adaptation. 

---
# CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation 

**Authors**: Pengying Wu, Zhejia Wen, Kang Ding, Kangjie Zhou, Chunxuan Jiao, Wei Pan, Xiangli Nie, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27677)  

**Abstract**: In object goal navigation (ObjectNav), an embodied agent must find a target object (e.g., "find a bed") in a previously unseen environment, relying on spatial commonsense such as object co-occurrence. Large Language Models (LLMs) encode this commonsense, but deploying them on robots exposes a trade-off between reasoning quality and efficiency: cloud-based models incur latency and privacy risks, whereas small on-device LLMs are fast but prone to hallucinated spatial judgments. Hybrid reasoning LLMs offer both a fast non-thinking mode and a slow thinking mode, which raises a key question for embodied agents: when is slow thinking worth its cost? We propose CovNav, a training-free framework that answers this question with Conformal Prediction (CP), turning dual-process (System 1 / System 2) inference into an uncertainty-aware decision. Much like recent decision models such as Jev, CovNav casts each spatial query as a structured binary decision rather than free-form generation: a local LLM in fast mode judges whether the target is likely near an observed object, CP converts its token probabilities into a prediction set, and only ambiguous queries are escalated to slow chain-of-thought reasoning. The object-pair formulation keeps calibration tractable and supports open-set objects through off-the-shelf detectors. Experiments on Gibson and HM3D show that CovNav surpasses the strongest zero-shot baseline by 0.8 and 1.7 percentage points in success rate on HM3D v1 and v2, while invoking slow reasoning for only 12.4% of queries and retaining most of the gain of always-on thinking at a fraction of its latency. Real-robot experiments on a TurtleBot 4 further validate the approach. CovNav shows how calibrated uncertainty can decide when an embodied agent should think slowly, offering a principled form of adaptive test-time reasoning for LLM-based robots. 

---
# Pairwise Learning Increases Data Efficiency in Deep Neural Networks 

**Authors**: Robert Bain, Marjan Rashidi, Elizabeth Chrastil  

**Link**: [PDF](https://arxiv.org/pdf/2610.27669)  

**Abstract**: Deep neural networks often fail at small-sample regression, collapsing to low-variance predictions that miss individual differences. Here, we introduce a pairwise comparison framework that reframes regression as aggregated relative rankings. This method improves data efficiency across model architectures and dataset sizes. On the UTKFace dataset (23k images), pairwise learning generates over one billion training comparisons without additional supervision. The paired ranking allows a lightweight ResNet10 trained from scratch on home hardware to achieve a 3.64-year MAE-surpassing previous models, including those with large-scale pretraining. For classification into categories, pairwise supervision consistently outperforms an otherwise identical 10-way crossentropy model on STL10 with 500-2000 labeled examples, improving accuracy by up to 5.2%. Overall, pairwise supervision enhances data efficiency and enables strong regression and classification performance when labeled data are limited, particularly in clinical settings. 

---
# A Unified Attack Framework against Statistical Generative Watermarks for Large Language Models 

**Authors**: Romina Omidi, Nima Naderloui, Yuan Hong, Binghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27653)  

**Abstract**: Watermarking has emerged as a key tool for identifying text generated by large language models (LLMs), with current schemes embedding statistical signals into the generation process to enable later detection. In this work, we identify a shared weakness across a broad class of popular LLM watermarking methods, including KGW (and its variants), GumbelSoft, DipMark, ITS, and SynthID. We show that these schemes exhibit a fundamental link between their detection scores and the token probability shifts they introduce during generation. This link, which we call score-ratio alignment, allows an attacker with only black-box query access to the model--and no knowledge of secret keys or model weights--to produce text that evades detection while remaining fluent and natural. We prove that the resulting attack is optimal across all covered watermark families, and show that the weakness is intrinsic: any watermark designed to maximize statistical detection power necessarily induces the very structure that attackers can exploit. We evaluate our attack on seven popular generative LLM watermarking schemes, three LLMs, and tasks, which reduce detection rates to at most 0.05 across all settings while preserving text quality. Our findings show that current watermarking methods can be reliably defeated under realistic attack conditions, highlighting the need for fundamentally different watermarking designs that decouple detection scores from token probability shifts. 

---
# Corpus Quality Improves, Retrieval Utility Does Not: An Ablation of Preprocessing and Refinement in a Speech-to-QA Pipeline 

**Authors**: Wonsup Shin, Jingu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27652)  

**Abstract**: Practitioners building training corpora with a pipeline of language models keep a preprocessing or refinement (post-process) stage when switching it on raises the corpus quality. We ablate a production pipeline that turns conference speech into question-answer pairs. The preprocessing stage rewrites the transcript, and the refinement stage filters the generated pairs. Switching each on and off gives four corpora from the same 40 Korean medical and finance sessions. On the corpus itself, quality rises as the stages switch on. Four LLM judges and six human domain experts rank the four corpora the same way, and the fully refined corpus leads by +0.18 and +0.22 on a 1-5 scale. We then ask whether that gain carries downstream. We supply each corpus to six models at inference and measure retrieval accuracy. The ranking does not hold. Enabling the transcript rewrite lowers retrieval accuracy by 0.061, the only stage effect that survives correction for multiple comparisons. That stage also reads the whole session, so it almost doubles the cost of one. With both stages on, accuracy does not change. An equivalence test we declared in advance confirms this. The corpus itself is clearly valuable. It lifts retrieval accuracy from 0.231 to 0.748 in every model and domain we test. More than a third of its pairs carry knowledge that neither of two frontier models produces on its own. On our data, the null does not come from weak corpora. A rising score on a rubric of this kind did not predict what the corpus holds for a model. 

---
# BCT: Bayes-Consistent Tempering for Flow Language Models 

**Authors**: Daisuke Oba, Hiroki Furuta, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.27641)  

**Abstract**: Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free variants for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and a von Mises--Fisher flow LM, BCT improves GSM8K generation across broad temperature ranges and achieves the strongest tempering performance. The gains also extend to Sudoku, demonstrating benefits on tightly constrained generation. At very low temperatures, the learned realization is also markedly more robust than the training-free variants, supporting temperature as a property of the generative flow across training and inference rather than a post-hoc probability adjustment. Code and model weights are available at this https URL . 

---
# A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera 

**Authors**: Yosub Shin, Yuan Liu, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2610.27622)  

**Abstract**: Handheld demonstration devices decouple data collection from the robot, but conventional designs burden the hand with a wrist camera, cables and a pose tracker (SLAM, an IMU, a VR controller, or a fixed external camera that the collector's body occludes), and burden the pipeline with as many data streams to timestamp, synchronise and cross-calibrate. We show that a single egocentric camera (USD 122) worn on the collector's head replaces the wrist camera, the pose tracker and their separate streams with one. It reads printed fiducials on an electronics-free glove (USD 22) against paper tiles taped around the workspace, giving the 6-DoF gripper pose and the jaw width from the same frames the policy will see; at deployment the same camera rides a low-cost (USD 120) arm beside the robot that mimics the collector's head from the robot's joint state. Against the robot's encoders the labels are accurate to 6.5 mm and 1.6 degrees median with 93-100% frame coverage, and a policy trained on them instead of encoders pays 3% in validation loss. Adding 194 glove demonstrations to 48 teleoperated episodes lifts rollout success from 10/25 to 17/25, level with 172 teleoperated episodes alone (18/25), and the full co-trained head-camera policy succeeds in 20/25 against 15/25 for the wrist-camera recipe, whose policies drop the object when the goal leaves their view, and 12/25 for a fixed camera. 

---
# Local Allow, Global Deny: Why Per-Action Authorization Does Not Compose in LLM Agents 

**Authors**: Anmol Pandey, Aditya Jain, Liang Chen, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2610.27620)  

**Abstract**: Autonomous LLM agents act through tools under per-action authorization: a local guard screens each tool call in isolation. We show that local authorization does not compose into global authorization. A compositional authorization failure (CAF) is a locally authorized execution, or a pair of locally authorized executions, that violates a global policy spanning intent, purpose, provenance, consent, and delegation. We characterize CAFs along two axes, expressiveness (single-trace safety versus relational k-safety) and observability (a ladder of monitor profiles). Our central result is an impossibility: relational purpose violations form a 2-safety hyperproperty with no per-trace locus of blame, so a per-action monitor that never raises a false alarm detects none of them, independent of the underlying model or decision procedure; they become visible only under cross-execution monitoring, which none of the six guardrails we survey implements. We make this constructive with AL-SYNTH, a SAT/SMT synthesizer (using no language model) that emits minimal CAF witnesses, each checked against an exhaustive oracle. The empirical signature is sharp: across nine LLM judges the per-action view approves the synthesized witness almost without exception, blocking it at a rate comparable to a benign control, yet the same models flag it when shown the two executions together. Per-action authorization is therefore structurally insufficient for relational policy conjuncts not reducible to single-trace safety: to bound them, a monitor must compare executions or accept false alarms. 

---
# Learning Inter-Graph Interactions Between Heterogeneous Substructures of Chemical Systems 

**Authors**: Gyoung S. Na  

**Link**: [PDF](https://arxiv.org/pdf/2610.27614)  

**Abstract**: Complex chemical systems containing heterogeneous substructures are common in real-world applications. Various physical phenomena of the complex chemical systems are derived from the interactions between the heterogeneous substructures. However, existing graph representation learning methods for inter-graph interactions assumed graph-level interactions between homogeneous structures, such as organic molecules and inorganic crystalline materials. We propose a data descriptor of the complex chemical systems and a graph neural network for learning inter-graph interactions between organic and inorganic compounds. We applied the proposed method to predict the physical properties of hybrid solar cell materials containing heterogeneous substructures, which have received significant attention for sustainable energy resources. By learning heterogeneous inter-graph interactions, the proposed method achieved state-of-the-art accuracy in predicting band gaps of 1,682 hybrid solar cell materials. 

---
# ABench: Adversarily Benchmarking Large Language Models across Taxonomized Values 

**Authors**: Liping Yang, Yongqi Leng, Jingting Zheng, Yuqi Ren, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.27611)  

**Abstract**: Ensuring the alignment of Large Language Models (LLMs) with diverse cultural values is critical for their safe global deployment. However, existing alignment benchmarks are predominantly English-centric or confined to simple discriminative tasks. Moreover, they often fail to effectively evaluate model robustness against complex, adversarial inductions within specific cultural contexts. To address these limitations, we propose ABench, a fine-grained, handcrafted benchmark specifically tailored to taxonomized values in Chinese. Grounded in a rigorous hierarchical taxonomy comprising 158 derived value subcategories, ABench consists of 2,895 open-ended questions. These questions employ techniques such as role-playing and logical traps to probe the stability of models' stances in adversarially induced contexts. Furthermore, we propose a weighted automated evaluation framework that utilizes question-specific scoring criteria for nuanced assessment. Extensive experiments on 16 mainstream LLMs reveal that most models exhibit significant vulnerabilities when subjected to sophisticated adversary attacks on values (scoring below 60). This underscores the need for more robust, culturally specific alignment strategies. Our data will be publicly released to promote further research on taxonomized value alignment. 

---
# Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims 

**Authors**: Zihan Chen, Di Zhu, Lei Nico Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27610)  

**Abstract**: Recursive self-improvement (RSI) promises AI systems, such as self-evolving LLM agents, that become better at producing further improvements. Evaluating this promise requires separating product claims about the final system from process claims about the improvement process itself. Existing surveys specify the evidence these claims require, but how often studies report that evidence remains unmeasured. We examine this question through full-text coding of 114 eligible studies in a reproducible arXiv search frame frozen in October 2026, including 90 that describe their own systems as RSI. The codebook links claims to evaluation designs. Independent human coding, third-reader adjudication, and a separate blind coding of improver comparisons assess coding reliability. Among the 90 RSI studies, 90% report product evaluation beyond development tasks under a broad criterion that includes partial or unstated separation, while 60% report evaluation on tasks unused for selection. Only 22% audit two or more reruns of the full loop. In a human-coded random sample, the RSI studies show the same contrast: 45% evaluate on tasks unused for selection and 14% audit two or more reruns. Of the 26 studies claiming a better improvement procedure, 23 report a related comparison, but only 4 compare earlier and later procedures from identical starting artifacts at a matched budget. Across these results and a comparison set of established works, evidence for repeatability and improved procedures remains less developed than product evaluation. Three propositions explain how selection effects, shared evaluator errors, and changes in system state limit what product evaluations can establish. Together, the findings support a claim-indexed reporting standard that connects each claim to the evaluation design needed to test it. 

---
# WORTH Benchmark: Evidence Discovery under Not-Fully-Cooperative Information -- A Two-Episode Pilot 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27597)  

**Abstract**: Existing reasoning evaluations typically provide task-relevant information to models as available premises, rarely examining concurrently how models discover evidence autonomously and how information presentation within narratives guides or obscures judgment. This paper proposes the WORTH Benchmark: models read original mystery narratives without clue annotations, autonomously select key clue sentences, identify case participants, and explain their reasoning process. The narrative materials are designed by professional authors such that evidence facilitating reasoning coexists with narrative arrangements that delay the revelation of truth. We report a pilot experiment under full-story conditions across two episodes, conducting a total of 70 formal runs on models, of which 67 runs are content-scorable. Scoring separately records hits on author-designated key clue sentences and participant selections; natural language explanations are used for case diagnosis and are not scored independently. The results show that clue hits and participant judgments are not synchronized; across 37 content-scorable runs in Episode 02, 31 runs simultaneously selected the same set of incorrect participants, while adjacent suspicion-oriented and exoneration-oriented sentences were selected 25 times and 0 times, respectively. These phenomena illustrate the diagnostic value of component-wise analysis and suggest that not-fully-cooperative informational relationships may influence evidence selection and judgment. This study only implements full-story conditions for two episodes, and does not establish stable model rankings or infer general mechanisms on this basis; human player records have not yet been cleaned, and this version does not report human-model comparisons. 

---
# Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora 

**Authors**: Wonsup Shin, Jingu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.27590)  

**Abstract**: When unique pretraining data is scarce, models see the same tokens many times and are thought to memorize. Random-token replacement, an input-corruption augmentation, is one remedy reported to lower validation loss, but a lower loss alone does not reveal which source of error it reduces. On a controlled source whose entropy rate is known in closed form, we show that replacement shrinks predictions toward the corpus average. We also introduce a decomposition that separates validation loss into approximation, estimation, and memorization terms. This shrinkage view predicts that replacement's bias cost needs no memorization but its benefit does. Memorization raises validation loss after its minimum, so we report the effect at two points, the minimum validation loss and the end of a fixed budget. Where replacement helps, it reduces this rise by about 60%. At the end of the budget, it turns from harmful to helpful as the controlled source becomes less predictable; at the minimum, it helps at no predictability level under the fixed recipe. On Korean and Japanese web text, replacement removes about a third of the loss rise after the minimum by the end of the budget, but its effect at the minimum is much smaller. Across ten real corpora, replacement lowers the final loss on nine. It harms only the census corpus, program-generated JSON that is the most redundant of the ten and much more redundant than any of 64 further prose and code corpora we measure. 

---
# Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval 

**Authors**: Yuchen Guo, Zhuoyuan Mao, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.27580)  

**Abstract**: Factual recall is a lookup: some representation inside the network serves as the key, and what that key is made of decides which queries can retrieve the fact, what an edit must cover, and what a retriever can read. We call it the factual access state. In autoregressive (AR) models the subject position enriches attributes largely independently of the relation, and relation-specific selection happens later, at the readout position. In masked diffusion language models (DLMs) we find the state is assembled differently, and much earlier: in the first attention layer the wording of the relation is routed into the position of whichever entity is visible and the early feed-forward layers look the fact up on this fused representation. The state therefore varies with the relation's surface form and with the query direction, so the queries about one fact occupy a region of state space rather than a point, and an operation on the fact succeeds only for queries that enter it. We treat the state as an interface to a DLM's factual knowledge, and put it to two uses. Writing to the state gives a knowledge editor: from one declarative statement, Multi-View Edit fits one shared value per direction across that region and installs it in the early-MLP block, preserving source-prompt reliability while substantially improving paraphrase and reverse question answering, where AR editors gain near zero. Reading it suggests a complementary use: the state at each entity position can serve as a relation-specific retrieval vector with no training, returning both pieces of evidence for two-entity queries on which pooled, prompted and lexical retrievers almost never return both. 

---
# TRCM: Topology-Aware Route-Consistency Modeling for Robust Neural Vehicle Routing 

**Authors**: Peiheng Jian, Zhenwei Wang, Tiehua Zhang, Lei Zhang, Zhishu Shen, Ruibin Bai, Zhijun Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.27568)  

**Abstract**: Neural combinatorial optimization has enabled fast constructive solvers for the Capacitated Vehicle Routing Problem (CVRP), yet most autoregressive models still construct solutions through sequential node selection and represent route organization only implicitly. We investigate whether route-level customer co-assignment can be modeled explicitly and used to guide this node-level decision process. To this end, we propose TopologyAware RouteConsistency Modeling (TRCM), which reuses node representations from a standard autoregressive construction pipeline to predict pairwise same-route probabilities and injects them into customer selection as a route-consistency signal. Because sameroute supervision is both class-imbalanced and structurally uneven, TRCM combines structure-preserving augmentation with Topological Entropy (TE)- based reweighting to emphasize locally ambiguous customer pairs. Under matched decoding budgets, TRCM reduces the optimality gap from 3.67% to 3.28% on CVRP50 and from 2.11% to 1.99% on CVRP100; under 8-fold augmented inference, it reduces the CVRP100 gap from 1.61% to 1.49%. On XML100, TRCM*8Aug improves the average objective by 0.24%, with larger gains on several structurally informative factors. A controlled demand-reassignment test over 500 variants further shows more concentrated objective fluctuations. These results support explicit route-consistency modeling as an effective structural principle for autoregressive neural routing. 

---
# AudiPsy: Evidence-Grounded Multimodal Reasoning over Spoken Counseling Conversations for Depression Assessment 

**Authors**: Chuanbo Zhu, Run Luo, Weibo Zhao, Yue Li, Kun Qian, Qiang Qu, Yike Guo, Minghuan Tan, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.27567)  

**Abstract**: Depression assessment from counseling speech requires integrating textual and acoustic psychological evidence across turns, yet existing end-to-end classification approaches lack explicit multimodal evidence interpretation and context-aware reasoning. We propose AudiPsy, an evidence-grounded omni-LLM for depression assessment over spoken counseling conversations. AudiPsy employs an Observation-Assessment-Summary (OAS) paradigm to integrate dialogue transcripts, textual and acoustic emotions, and textual-acoustic emotion incongruence into structured reasoning. OAS incrementally organizes multimodal observations, performs contextualized assessment, and maintains an evolving dialogue summary for subsequent turns. To support this framework, we construct AudiPsy-Counsel, a synthetic spoken counseling conversation corpus with multimodal context augmentation, and develop a unified three-stage curriculum for depression-oriented reasoning. Experiments on DAIC-WOZ demonstrate superior depression assessment performance over specialized depression detection methods and general-purpose speech/omni-LLMs, along with improved reasoning quality, while supplementary evaluation on EATD-Corpus supports AudiPsy's applicability to naturalistic conversations. 

---
# Learned Readout Tokens for Extracting Large Language Model Confidence 

**Authors**: Nima Seifi, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2610.27566)  

**Abstract**: Large language models (LLMs) are increasingly used within everyday applications where their responses have significant downstream impacts. For responsible deployment of such systems, we desire insight into how much we can trust their individual outputs. A popular approach for estimating LLM confidence is to train probes on the hidden states of the LLM's tokens to predict the correctness of the model's output. These probes typically read hidden states from tokens that are already in the model's vocabulary. In this work, we introduce a new approach to confidence probes that learn a new readout token and utilize its hidden states to train the probe. We show that our learned readout approach ScratchProbe outperforms its existing-token counterpart on a variety of datasets and LLMs. We also extend our method to a multi-token approach which we show provides further performance improvements. Finally, since the learned readout changes only the position a probe reads from, any existing probe can adopt it. Applied to a state-of-the-art probe, it improves AUROC on every backbone, in domain and out of domain. 

---
# AnomalyJev: Towards Evaluating Image Anomaly Explanations via Claim-Level Verification 

**Authors**: Qizhou Wang, Jiawen Zhu, Guansong Pang, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2610.27565)  

**Abstract**: Recent multimodal LLM (MLLM)-based anomaly detection (AD) methods can generate textual explanations that help interpret their predictions. However, reliably and efficiently evaluating these explanations remains challenging. While LLM judges can assess explanations through structured rubrics, they often require substantial token generation, making evaluation costly and difficult to scale. We instead propose to anchor evaluation in the available ground truth by extracting verifiable claims from each explanation and verifying them against the corresponding annotations. We instantiate this idea as AnomalyJev, a lightweight framework that extracts verifiable claims from contextualised representations using ground-truth-defined queries and pointer based heads, without token generation. The claims are then verified against the ground truth to score faithfulness and completeness. For evaluation, we construct a human-annotated set of 640 explanations with over 1,600 expert ratings across in-domain and cross-dataset settings. Despite its compact design, AnomalyJev achieves stronger agreement with human annotators than most open MLLM judges and approaches that of flagship MLLM judges, at a fraction of their runtime and resources. 

---
# CityPaint: A Free Lunch from 2D Diffusion for City-Scale 3D Texturing 

**Authors**: Yuchen Zou, Lihuang Fang, Siyu Zhu, Xiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27564)  

**Abstract**: Generating high-fidelity, city-scale 3D environments plays a crucial role in applications such as virtual reality, gaming, and urban digital twins. However, the scarcity of real-world city-scale 3D training data forces existing methods to either stitch independently generated assets or learn from synthetic urban corpora, often sacrificing realism and global consistency. In contrast, 2D image data and their corresponding generative models are abundant and encode rich visual priors. To bridge this gap, we propose CityPaint, a novel framework that leverages 2D diffusion models to generate globally consistent city-scale 3D textures without relying on city-scale 3D training data. Specifically, CityPaint integrates geometric priors with the generative power of 2D diffusion models, introducing a robust top-$K$ camera baking strategy for iterative texture projection. To address the artifacts of 2D-to-3D projection, we further propose a decoupled refinement mechanism that separately optimizes terrain and building textures. Extensive experiments show that CityPaint produces globally coherent textures with fine local details and achieves favorable results against strong baselines in both visual quality and cross-view consistency. 

---
# AnchorRM: Parallel Intra-Batch State Reconstruction with Anchor-Based Residual Memory for Temporal Graph Neural Networks 

**Authors**: Haroon Gharwi, Kai Shu, Yue Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.27558)  

**Abstract**: Memory-based temporal graph neural networks (TGN) rely on recurrent node memories to capture evolving interaction histories. However, temporal mini-batching introduces a trade-off between computational efficiency and temporal fidelity: when a node appears multiple times within the same batch, later interactions are evaluated using stale pre-batch memory because the intermediate memory updates have not yet been committed. This problem becomes more severe as temporal batch size increases. To overcome this challenge, we propose AnchorRM, an anchor-based residual memory mechanism that reconstructs event-specific intra-batch node states without executing sequential recurrent updates. AnchorRM treats the last committed node memory at the beginning of each batch as a pre-batch fixed anchor and uses causally preceding same-batch interactions to infer the missing state evolution relative to that anchor. The learned residual then calibrates the anchor to reconstruct an event-specific node memory for each interaction. Because each event-specific residual is predicted directly from the fixed anchor and its causal history, reconstructed states do not depend on previously reconstructed states and can therefore be computed in parallel within the batch while preserving temporal causality. Experiments across four temporal graph datasets and three memory-based backbones demonstrate that AnchorRM achieves an average 6.16x training speedup (up to 10.05x) and 28.67x batch scaling (up to 40x). These results demonstrate that AnchorRM substantially improves the scalability and training efficiency of memory-based temporal graph learning while maintaining competitive predictive performance. 

---
# Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation 

**Authors**: Bingqi Shang, Yang Zhang, Yiwei Chen, Changsheng Wang, Shiyu Chang, Bingquan Shen, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27555)  

**Abstract**: Safety pretraining has emerged as a promising paradigm for embedding safety deeply into a model during pretraining, yielding robustness to subsequent fine-tuning that conventional safety post-training often lacks. We refer to this safety acquired during model pretraining as native safety. Yet native safety is expensive to acquire, confined to the model in which it was pretrained, and often accompanied by lower general utility. In this work, we rethink the practical value of safety-pretrained models and introduce, for the first time, the native safety reuse problem: Can native safety be transferred from a specialized, potentially smaller, safety-pretrained "teacher" to standard-pretrained "students," enabling them to inherit its robustness while preserving their general utility? We show that native safety can "travel" through on-policy distillation. A standard-pretrained student generates responses to safety-pretraining prompts, while a small safety-pretrained teacher supervises the student's token-level predictions along its own generated trajectories; cross-tokenizer alignment further enables transfer across model families. The resulting students exhibit substantially greater safety robustness than their safety-post-trained counterparts while retaining higher general utility than the teacher: subsequent fine-tuning raises the average attack success rate (ASR) of the 1.7B student by only 5.00%, compared with 11.58% for safety post-training, and its average utility of 48.20% far exceeds the aligned teacher's 27.42%. Unlike conventional large-to-small knowledge distillation, our framework enables selective, small-to-large safety transfer: it reuses the robust safety expertise of a smaller pretrained teacher without inheriting its utility limitations. These findings establish native safety as a reusable model property rather than one permanently tied to the originally pretrained model. 

---
# Less is More: Concise Retrieval and Critic-based Re-ranking for Composed Image Retrieval 

**Authors**: Yuqian Zheng, Hyoju Ahn, Hyeongjin Ahn, Eunil Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.27538)  

**Abstract**: Zero-shot composed image retrieval (ZS-CIR) aims to retrieve a target image based on a reference image and modification text. Recent approaches commonly rely on large vision-language models to generate target image captions for retrieval. However, these captions may include redundant semantic details, introducing noise when retrieving the target image and leading to suboptimal retrieval results. To address this issue, we propose Less-is-More CIR, a two-stage training-free framework that combines Concise Retrieval and Critic Re-ranking. In the first stage, concise target-oriented captions are generated that preserve only the most discriminative modification semantics, thereby reducing semantic noise in the retrieval. In the second stage, a critic re-rank module further evaluates the retrieved candidates from the perspectives of modification consistency and reference preservation. Extensive experiments demonstrate the effectiveness and robustness of the proposed framework for the ZS-CIR. Our code is available at: this https URL. 

---
# Adversarial Benchmarks and BenchDoor Attacks: Exploiting Benchmark Incentives for Backdoor Attacks 

**Authors**: Yixin Wu, Rachel Hong, Jamie Morgenstern, Tadayoshi Kohno  

**Link**: [PDF](https://arxiv.org/pdf/2610.27533)  

**Abstract**: Machine learning benchmarks act as guideposts for model development and spur the creation of training datasets intended to improve benchmark scores. We introduce BenchDoor, a family of data poisoning attacks that exploits this incentive. Beyond preserving benign performance, BenchDoor requires benchmark gains that encourage adoption of data that also implants a backdoor. The adversary either manipulates a benchmark's training or test data, creating an adversarial benchmark, or supplies an independent training dataset while leaving the benchmark unchanged. We formalize five attack scenarios, including training on rewritten test examples and applying similarity-based decontamination. Across multiple-choice Q&A, text classification, and open-ended Q&A with Qwen and Gemma models, BenchDoor achieves positive benchmark gains for all 30 evaluated configurations, with held-out attack success rates of 82.33% to 100.00% at poisoning rates of 2% to 10%. Both objectives remain achievable after synthetic data generation and similarity-based decontamination, with attack success rates of at least 90.33%. These findings show how benchmarks can serve as attack vectors both through their own data and through training resources that deliver improvements on them. 

---
# HealBench: A Fine-Grained Benchmark for Evaluating Psychological Comfort in Text-to-Image Generation 

**Authors**: Junjie Xu, Yuzhe Yang, Rui Yang, Luwei Xiao, Wenxin Wu, Xingjiao Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.27516)  

**Abstract**: Despite advances in photorealism, semantic fidelity, and aesthetics, text-to-image generation and image editing remain understudied in a human-centered dimension: whether their outputs feel psychologically comforting. Grounded in cognitive evaluation theory, we define psychological comfort as an affective response shaped by viewers' appraisal of restorative and emotionally supportive visual cues. We introduce HealBench, a fine-grained benchmark of 30,000 high-quality image-text pairs spanning 19 categories and 92 subcategories, selected from 100,000 candidates through a human- and expert-in-the-loop pipeline combining vision-language model matching, human preference models, expert filtering, and exemplar-guided augmentation. We also construct HealBench-Eval, a feature-clustered evaluation set of 1,095 instances. Using HealBench-Eval, we conduct objective and subjective evaluations of 14 representative image generation models and prompt-rewriting strategies. Results reveal a gap between conventional visual quality and perceived comfort: strong semantic alignment and aesthetics do not guarantee emotionally restorative images. Reinforcement learning experiments further show that HealScorer can serve as a reward signal for assessing and improving the emotional and restorative quality of generated images. These findings establish psychological comfort as a distinct dimension of human-centered image generation and position HealBench as a foundation for affect-aware generative models. 

---
# Progressive Subset Selection and Cascaded Consistency for Unsupervised Vision-Language Model Adaptation 

**Authors**: Hao Yan, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27515)  

**Abstract**: Pretrained Vision-Language Models (PVLMs) achieve strong zero-shot recognition but often suffer performance degradation under target-domain shift. Unsupervised adaptation provides a practical way to improve target-domain performance without labeled data, but it remains challenging because pseudo-labels generated by the model can be unreliable and no labeled validation set is available for model selection or early stopping. We propose a simple framework for unsupervised PVLM adaptation that combines progressive margin-based subset selection with cascaded consistency learning. The subset selection strategy constructs a class-balanced pseudo-labeled set using prediction margins and gradually expands it during training, enabling adaptation to start from reliable supervision anchors before incorporating broader target-domain data. Cascaded consistency learning further exploits unlabeled samples by using temporally smoothed teacher predictions on weakly augmented views to supervise student predictions on strongly augmented views, with confidence filtering to reduce the influence of uncertain pseudo-labels. To preserve the pretrained embedding structure, the framework updates only normalization parameters in the visual encoder and class prototypes initialized from text prompts. Experiments across multiple recognition benchmarks with both CLIP and SigLIP backbones show consistent improvements over zero-shot baselines and competitive performance against recent unsupervised PVLM adaptation methods. These results suggest that progressive pseudo-label selection and teacher-guided consistency provide an effective and lightweight strategy for adapting PVLM classifiers without labeled target data. 

---
# Cross-Visibility-Gap Identity Persistence for 4D Scene Graphs 

**Authors**: Jiahui Hu, Li Gao, Yu Liu, Pengzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27505)  

**Abstract**: 4D scene graphs require persistent object identity when an object disappears inside an opaque carrier and later reappears as a fresh detection. When appearance and target-centric motion are unavailable, identity recovery must rely on relational history. We formulate cross-visibility-gap identity persistence as historical-node recovery through a remembered target--carrier binding.
We introduce CVG-SGM, a perception-frontend-compatible transfer--maintain--read memory. Geometry-weighted cross-node interaction transfers target-specific evidence into carrier states, a shared per-node GRU maintains this information during transport, and a two-time readout combines stored carrier--identity compatibility with current detection--carrier geometry to recover the historical identity.
In the controlled one-identity-per-carrier setting, CVG-SGM achieves 1.000 re-link accuracy, compared with 0.570 for both no-memory and parameter-count-matched no-write controls. It also generalizes to scene-disjoint, unseen longer visibility gaps. On real RGB-D/SAM2 feature-level episodes, it correctly re-links 36/36 held-out detections, compared with 28/36 for the strongest fixed carrier--identity control. Counterfactual carrier-memory swaps redirect predictions according to the swapped bindings, supporting the role of stored relational information in identity recovery. The current approach remains limited by shared-carrier identity ambiguity and does not provide end-to-end raw-video re-identification. 

---
# Self-Evolving Agents through Privileged Retrospective Distillation 

**Authors**: Yuhang Zhou, Yifan Wu, Bo Peng, Mingyi Wang, Jiayi Liu, Xiangjun Fan, Lizhu Zhang, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27495)  

**Abstract**: Self-evolving agents aim to improve from their own experience rather than remain fixed after deployment, but naturally collected trajectories often lack ground-truth solutions or verifier-derived rewards. We introduce Retrospective Self-Distillation (RSD), a multi-round framework that turns such experience into dense supervision using privileged retrospective context. For each on-policy trajectory, the same underlying model acts as both student and teacher: the teacher re-scores the student's realized actions while conditioning on cross-task precedents and lessons, together with turn-level reflection and future hindsight from the completed trajectory. The resulting teacher distribution is distilled back into the policy through on-policy self-distillation. Across rounds, the experience bank grows and the retrospective context is refreshed, allowing the model, its data, and its supervision to co-evolve. We further introduce frontier-guided context-policy evolution, where a stronger frontier model proposes constrained edits to how privileged context is constructed. Across ALFWorld, ScienceWorld, and SearchQA with Qwen3-8B and Qwen3-14B, RSD improves performance by up to 59.7% relative to the base model, while frontier-guided context-policy evolution adds up to a further 44.7% relative improvement over a fixed context policy. These results show that naturally accumulated experience can be converted into persistent model capability without ground-truth trajectories or verifier-derived rewards for the core policy update. 

---
# Orthogonal Boundary Unlearning for Dual-Use Concepts in Large Language Models 

**Authors**: Niccolò Ronchetti  

**Link**: [PDF](https://arxiv.org/pdf/2610.27490)  

**Abstract**: Mitigating catastrophic risks from large language models requires excising hazardous operational capabilities, such as cyberwarfare exploitation and pathogen enhancement, without degrading benign scientific utility. In these dual-use domains, legitimate defensive science and malicious attacks share vocabulary, syntax, and internal model representations, causing standard unlearning methods to induce severe collateral damage (concept bleeding).
In this work, we present Orthogonal Boundary Unlearning (OBU), a mathematically grounded framework designed to eliminate collateral damage on benign dual-use capabilities. OBU extracts a domain-matched capability subspace from paired calibration queries via singular value decomposition on empirical activation covariances, and restricts parameter updates strictly to the orthogonal null space. This guarantees that benign representations experience zero parameter perturbation along the spanned calibration subspace.
We present two complementary realizations of OBU: Projected LoRA, which applies the orthogonal constraint during iterative gradient-based fine-tuning, and Closed-Form Weight Surgery, which directly edits model weights in a single analytical step without training. Across four modern open-weight foundation models (Gemma-3-4B, Qwen3.5-4B, SmolLM3-3B, and Granite-4.2-3B), OBU achieves strong dual-use concept separation, substantially mitigates in-context priming attacks, and demonstrates durable resistance against adversarial relearning without degrading benign domain competence. 

---
# Beyond the Prompt: Recovering Hidden Concepts in Text-to-Image Models 

**Authors**: Zhengfeng Zheng, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27473)  

**Abstract**: Text-to-image models can generate visual concepts that their prompts never mention, and recent work shows that such associations can be implanted so that ordinary prompts trigger them. Auditing this behavior requires recovering the description that a prompt omits. Existing prompt optimizers are not well suited to this task. Gradient-based methods require backward access to the scoring encoder. Discrete evolutionary search treats semantically unrelated tokens as equally close. Language-model-guided search requires one inference call per candidate, which limits exploration. This paper presents EmbGA, an embedding-space genetic algorithm that recovers concepts present in a set of images but absent from the prompt that generated them. EmbGA keeps the population structure of genetic search but applies its operators in the token embedding space. Crossover interpolates between parent embeddings, and mutation moves an embedding toward a randomly drawn vocabulary embedding. Each offspring is then projected back to discrete tokens before scoring, so every evaluated candidate is a valid token sequence. We evaluate EmbGA on two benchmarks against six optimizers under two exploration budgets. EmbGA achieves the highest alignment when the exploration budget is large, recovers the most grounded tokens on natural images, and runs much faster than language-model-guided search. 

---
# Privacy-Preserving Personalized Federated Learning (PPP-FL) for Medical Imaging 

**Authors**: Md Morshedul Islam, Hossain Shahriar, Alfredo Cuzzocrea  

**Link**: [PDF](https://arxiv.org/pdf/2610.27470)  

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that reduces transmission costs, mitigates cross-border data sharing restrictions, and enhances privacy by keeping raw data on local devices. Clients in FL systems train local models and share only updates, which are aggregated by a central server. While this setup limits direct data exposure, FL remains vulnerable to privacy attacks. Existing attack and defense strategies have largely been developed for centralized systems, and adapting them to FL introduces new challenges due to its unique architecture and training workflow. In this work, we first propose a model inversion attack on FL systems using Deep Convolutional Generative Adversarial Networks (DCGANs) in the context of large-scale medical imaging. Our method operates under realistic assumptions about the attacker's knowledge and capabilities. We then examine the effect of client-side differential privacy (DP) in mitigating privacy leakage and incorporate personalized FL to improve model performance under DP settings. Experiments on the Brain MRI ND-5 dataset show that DCGAN-based attacks can reconstruct semantically meaningful synthetic images, revealing significant risks to the privacy of training data. Applying DP reduces attack success but introduces an accuracy trade-off, which is further overcome by personalized FL at each client side. We also evaluate how different aggregation algorithms affect both attack vulnerability and defense effectiveness, underscoring the critical role of aggregation strategies. The proposed attack-and-defense framework is model-agnostic and applicable to a wide range of FL-based systems. 

---
# Training-Free Continual Learning for Multimodal Forecasting via Prompt Evolution 

**Authors**: Laurent Mombaerts, Jonathan Taws, Jacopo Gargano, Dinesh Gaddam, Kevin George, Srijan Tiwari, Shreyas Rajesh  

**Link**: [PDF](https://arxiv.org/pdf/2610.27468)  

**Abstract**: Adapting a forecasting system to a changing world classically means retraining. We ask whether the task of continually learning regimes for time series forecasting can be represented by a piece of text: a guidance block, evolved by outcome-driven reflection and used in a frozen LLM forecaster's prompt. On a refreshed multimodal benchmark (weekly search indices, daily currencies, daily commodities, each window paired with web-retrieved news), we run GEPA-style prompt evolution under strict contamination control to prevent the data leakage common in timeseries benchmarks: the forecaster's knowledge cutoff predates every evaluated window. A 900-word evolved guidance carried by Llama-3.3-70B beats TimeSeries Foundation Models (TSFMs) on held-out data (GM MASE 0.776 vs. 0.898 for TabPFN-TS-3), transfers to more recent models without measurable loss, and costs dollars rather than GPU-days to update. We characterize the loop: out-of-loop gains concentrate in the first reflection, in-loop improvements overstate out-of-loop gains 2-3x unless an untouched monitor arbitrates, and error-only objectives silently suppress the text modality. We provide two extensions. First, compiling the guidance into executable code with a TSFM as a callable tool recovers most of the gain deterministically. Second, ablations show the advantage widens at short horizons and that removing the news text does not hurt the guided forecaster, consistent with the text's value being distilled into policy during learning. We argue that continual prompt evolution is a practical, training-free alternative to retraining for multimodal forecasting. 

---
# Projected Distillation: On Learning to Solve Hard Problems from Close, Correct Targets 

**Authors**: Shripad V. Deshmukh, Christian Bock, Kangrui Ruan, Abhijeet Awasthi, Vijay Lingam, Rein Houthooft, Manish Prajapat  

**Link**: [PDF](https://arxiv.org/pdf/2610.27467)  

**Abstract**: On-policy learning is central to post-training: training on a model's own generations limits distribution shift and helps preserve existing capabilities. On hard problems, however, on-policy RL provides no reward-based learning signal when all attempts fail. On-policy distillation provides next-token supervision from a teacher, but on hard problems, if a rollout goes wrong early, subsequent supervision is conditioned on prefixes that already contain the mistake. Correct behavior can instead come from a gold or teacher solution, but such off-policy targets can differ substantially from the model's generations and, as we show, may erode existing capabilities. We therefore treat on-policyness as a continuum and propose projected distillation, which minimally edits the student's failed attempts into correct targets. The method uses contrastive updates to favor corrections over failed attempts and refreshes these corrections as the student evolves, enabling near-on-policy learning even when all student attempts fail. Theoretically, we prove that the student's learning regret grows with the edit size: it scales linearly with the horizon under minimal correction, as in DAgger, and quadratically under full-answer replacement, as in behavior cloning. Empirically, across two students (Qwen3-8B and OLMo-2-7B) and five domains spanning math, code, science, and tool use, projected distillation gives the highest area under the specialization-retention frontier (best on 8 of 9 cells against GRPO and OPD, and 7 of 9 against SPOT and SODA). Our ablations highlight the importance of online corrections, the contrastive loss, and on-policyness: moving targets farther off-policy steadily worsens the specialization-retention trade-off, with retention falling from 0.916 to 0.744 when minimal corrections are replaced by student-agnostic gold solutions. 

---
# Beyond Scalar Leaderboards: Adaptive Sampling for Pareto Frontier Identification in Multi-Objective LLM Arenas 

**Authors**: Liang Twist Shan, Jiahao Ai, Zhimei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.27447)  

**Abstract**: The evaluation of large language models (LLMs) is inherently multi-objective, as different models may offer different tradeoffs across performance dimensions such as math, code, and writing. Because such evaluations are often based on costly and time-consuming human feedback, it is important to draw reliable conclusions using as few comparisons as possible. In this paper, we present Vector-Borda Empirical-Gap Elimination (VB-EGE), a fixed-confidence adaptive sampling algorithm for efficiently identifying models on the Pareto frontier in arena-style evaluation, where each comparison evaluates a pair of models on a prompt with respect to a selected performance dimension. VB-EGE progressively eliminates models from an active set once sufficient statistical evidence has been accumulated to certify their Pareto status. Under a multi-objective Bradley-Terry model, we show that VB-EGE returns the correct Pareto frontier with high probability and derive an instance-dependent upper bound on its sample complexity. We further establish a worst-case lower bound showing that VB-EGE is minimax-optimal up to logarithmic factors. Simulations and experiments on three real-world Arena datasets (Arena, SearchArena, and VisionArena) demonstrate VB-EGE's empirical effectiveness. 

---
# Scaling Graph Reasoning in Language Models with Learnable Graph Positional Encodings 

**Authors**: Javier Porras-Valenzuela, Arush Arora, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2610.27439)  

**Abstract**: Solving reasoning tasks supported on graphs typically requires providing a Language Reasoning Model (LRM) with a serialized description of the graph. This approach results in the input length growing with the number of edges, which limits the size of the problems that can be solved. Furthermore, even when graphs fit in context, performance sharply decays as the graphs grow. We propose to address this by augmenting LRMs with scalable and permutation equivariant graph positional encodings that modulate attention logits. An additional post-fusion-style module incorporates structural information into hidden states. We compute these positional encodings using a GNN followed by a Graph Transformer. This eliminates the need for serialized adjacencies, reducing the context burden to $O(N)$. We jointly fine-tune the positional encoder and the LRM LoRA adapters using optimal solutions generated by graph-theoretic solvers, resulting in a computationally efficient method. Trained on small graphs, the model exhibits improved size-generalization across graph-theoretic tasks on graphs with up to 1,000 nodes, and produces fewer tokens than a conventional LRM. Additionally, we observe that our model remains robust to node relabelings. Evaluations on graph-theoretic benchmarks also show that performance on small graphs is preserved. 

---
# MLIPS: Multi-Layer Indexer Pool Sharing 

**Authors**: Samkit Jain, Sai Kishan Pampana, Shlok Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.27438)  

**Abstract**: Sparse attention limits the keys used by each query, but finding those keys can still require a full-prefix scan at every layer. We introduce \textbf{MLIPS (Multi-Layer Indexer Pool Sharing)}, a training-free method that shares candidate search while retaining each consumer's own scoring and selection. A producer publishes a top-$M$ pool larger than the attention budget $K$; subsequent layers select their own top-$K$ from that pool. Measurements across three sparse-attention model families show substantially higher coverage from broad pools than overlap between final selections. We implement layerwise sharing, persistent decode reuse, and a coarse producer in vLLM for DeepSeek-V4-Flash. Across four long-context suites, three consumer layers per producer give $1.14$--$1.40\times$ speedups and $14.8$--$27.4\%$ lower GPU energy, with observed accuracy changes of $-1.49$ to $+0.29$ percentage points in single runs. More extensive sharing exposes task-dependent quality losses. Sampled prefill traces also reveal recurring selected KV positions across query groups and layers. We characterize this shared working-set structure and use it to motivate Top-$G$, an exploratory bounded pool construction for prefill. Its deployment, task accuracy, and serving benefits remain open. 

---
# The Router Already Knows 

**Authors**: Andrew Bistras, Marty Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2610.27421)  

**Abstract**: Mixture-of-experts (MoE) models already make a sparse choice for every token: which experts should process it. We ask whether a sparse dictionary should reuse that choice. NR-SAE gives each routed expert a small dictionary and opens only those the frozen router selects. At block 15 of Gemma 4 26B-A4B, after ten million C4 training tokens and across three seeds, the expert-aligned autoencoder explains 0.518 of the variance in router-weighted expert outputs on a fixed evaluation capture. A rank-addressed control with the same parameters but sixteen times as many candidates reaches 0.481, a candidate-matched control 0.458, and an architecture-matched control with randomized expert identity 0.387. A transcoder variant also leads all three controls: 0.432, against 0.347, 0.371, and 0.118. Part of this advantage is architectural: each expert writes into a subspace of at most a quarter of the model width, so naming the expert names the subspace. How much it explains, we have not measured. At this layer and budget, the router already knows something a dictionary would otherwise have to learn. Our implementation did not train faster, and semantic interpretability, downstream loss, and causal faithfulness remain untested. 

---
# Fundamental Impossibility Results in AI Safety: Trading Off between Desiderata 

**Authors**: Ida Mattsson, Ratip Emin Berker, Emanuel Tewolde, Ioannis Anagnostides, Kiriaki Fragkia, Xinyu Liu, Alexander Heckett, Jiayuan Liu, Eden Hartman, Vojtěch Kovařík, Sven Neth, Maria-Florina Balcan, Tuomas Sandholm, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2610.27373)  

**Abstract**: Is there a fundamental tradeoff between desiderata for safe AI systems, such as harmlessness and helpfulness? We answer by formalizing the interaction between AI systems and users as a game with three players: a defender and an attacker that are computationally or informationally bounded, and an AI that provides public advice and is less constrained. Our main result shows that there is a fundamental tradeoff in providing advice that helps the defender while advice remains public. We also consider whether the information that the AI shares can be manipulated to benefit only the defender and conclude with a discussion of how we can use these results in designing safer AI systems. 

---
# Trail-Weighted Influence Scores for Approximate Outcome Prediction in Bayesian Networks 

**Authors**: Sahil Kumar Rajput, Randhir Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.27366)  

**Abstract**: We study trail-weighted pruning of evidence for Bayesian-network prediction. A traversal enumerates the active trails from each evidence node to the outcome, scores each trail by the product of its edge strengths, and drops evidence whose score falls below a threshold. For collider-free networks (chains and forks) over binary variables, we show that the trail score equals the exact correlation between a trail's endpoints, that a chain and a fork with equal edge strengths have identical joint distributions, and that the pruned posterior is within a computable sum of per-trail terms of the exact posterior when each dropped trail lies in its own branch of the outcome node. We prove this bound, check it exhaustively on all small trees, and show that it fails when dropped trails share a branch. In simulations on random trees, thresholding trail scores gives 1.4 to 13 times lower error than random pruning of the same size. We also apply the score, as an unproven heuristic, to seven bnlearn networks with colliders and loops. Keeping the top half of the active evidence by trail score gives lower error than random pruning on all seven, but it is not clearly better than a mutual-information ranking, and scoring costs more than the inference it saves on six of the seven, so we show no speedup. Models trained on trail, graph and information features and tested on networks they never saw do not beat the trail score or mutual information, but trail features carry most of what they use and clearly outperform generic graph features. A bound for general networks is left to future work. 

---
# Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection 

**Authors**: Mosh Levy, Aly Lidayan, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.27364)  

**Abstract**: In joint-embedding predictive architectures (JEPA), an encoder and a predictor are trained together to produce a representation that minimizes the error in predicting the representation of the next observation. Because the representations of both the current and the next observation are learned, it is unclear which features they will include. Earlier characterizations describe the features JEPA selects in terms of their individual predictability, raising the concern that unpredictable features, like a traffic light, will not be encoded into the representation. However, the prediction error is also affected by the relationships between features, e.g., the light's color helps predict whether the traffic will stop. We propose the predictive closure characterization, which accounts for these relationships. Based on a reformulation of the JEPA training objective, we show that selection depends on properties of feature sets. This view agrees with earlier works when features evolve independently, but makes new predictions when they do not, for instance that a feature like the traffic light can still be retained. We test this characterization in controlled experiments, which show that unpredictable features can become better represented when they help predict other features, and that changing one feature's predictive relationships can affect how well others are represented. We also demonstrate implications for steering feature selection: targeting one feature with an auxiliary objective, or supplying it to the predictor as side information, can strengthen the representation of another feature whose future values it helps predict, so steering can be indirect. 

---
# On the Limits of Scale: Inverse Scaling in Vision-Language Models 

**Authors**: Rohit Saxena, Alessandro Suglia, Marek Rei, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2610.27360)  

**Abstract**: Scaling laws suggest that increasing model size, training data, and compute generally yields monotonic performance improvements. While this often holds at the aggregate level, we present systematic evidence that vision-language models (VLMs) exhibit inverse scaling at the subtask level: specific question categories on which larger models perform worse than smaller counterparts within the same model family. We evaluate 19 models from four distinct model families across eight multimodal benchmarks (nine evaluation splits), answering 29,386 questions per model, 558,334 model-question evaluations in total. Using per-question pairwise analysis across all model pairs in each family, we identify 375 subtask-model-pair combinations exhibiting inverse scaling, spanning 4,342 unique queries where a smaller model answers correctly but a larger model fails, and we find that these patterns replicate across independently developed architectures. We identify 13 subtask-dataset pairs that show inverse scaling across multiple families, consistent with the phenomenon being task-driven rather than model-specific. The most affected domains in our study include multi-image spatial reasoning, visual perception, and university-level STEM knowledge, with accuracy gaps exceeding 20 percentage points in several cases, while diagram understanding appears comparatively resistant. These findings challenge the assumption that increasing model size uniformly improves multimodal capabilities and suggest that model selection should be task-aware. 

---
# LiBRA: Logit Fingerprints for Byzantine-Robust Client Selection in Federated Learning 

**Authors**: Saheed A. Tijani, Frank Jiang, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27357)  

**Abstract**: Byzantine-robust aggregation shields federated learning from malicious updates, but the rules that filter out suspect updates do so by comparing clients' parameter vectors. That comparison requires access to every client's update, which secure aggregation exists to withhold, costs $O(n^2 d)$ in the parameter dimension $d$, and is defined only between clients that share an architecture. We propose LiBRA, which compares clients on what their models compute rather than on the parameters that compute them, using a fingerprint of $C^2$ values for $C$ classes, the class-conditional logit response of each client's model to pseudo-patterns optimised against it. We instantiate LiBRA as variants of Krum, Multi-Krum and Bulyan that change only what the dissimilarity score is computed on. On Fashion-MNIST and CIFAR-10 with three architectures and four backdoor and label-manipulation attacks under i.i.d. partitions, each variant holds every attack its parent holds, and on identical client updates under a backdoor attack the fingerprint ranks malicious clients within 0.006 AUC of the full parameter vector. The variants thus match the robustness of the rules they extend, while their trust decision reads no client parameter, compares up to 49,090 times fewer values, and can by construction compare clients whose architectures differ, at the cost of one pattern optimisation per client per round. 

---
# Exposing Vulnerability Monoculture Through Systemic Adversarial Robustness 

**Authors**: Leo Elmecker-Plakolm, Lukasz Szpruch, Avishek Joey Bose, Matthew Wicker  

**Link**: [PDF](https://arxiv.org/pdf/2610.27353)  

**Abstract**: Beneath the apparent diversity of foundation models lies an increasingly homogeneous technical stack of common training pipelines, distillation methods, and data. These shared algorithmic choices can expose an otherwise distinct model ecosystem to concentrated risks leading to a vulnerability monoculture. Yet current practice in assessing adversarial robustness is restricted to individual robustness scores and attack transfer rates, leaving open the unexplored fundamental question: how many attacks suffice for catastrophic widespread compromise? In this paper, we investigate the empirical nature of shared vulnerabilities within a model pool and introduce Systemic Adversarial Robustness (SAR), a novel measure of shared vulnerability of a model pool through the attack diversity needed to exploit it. Specifically, SAR interpolates between single-attack transferability and independent per-model robustness: small values reveal a vulnerability monoculture, whereas large values indicate that failures are dispersed across models. We theoretically formulate SAR as a partial set-cover problem with provable approximation guarantees---enabling the development of scalable algorithms to empirically measure the systemic adversarial robustness. We empirically test the shared vulnerability of vision and large language models and characterize their shared collective vulnerability curve. Specifically, we demonstrate that model pools distilled from DeepSeek-R1 and Claude Opus 4.6 exhibit lower estimated systemic robustness than their corresponding parent pools, revealing a potential systemic cost of frontier-model distillation. Our framework seeks to establish vulnerability monoculture as a measurable target for a more wholistic adversarial evaluation of model populations. 

---
# Refuses When Asked, Complies When Acting: A Runtime Safety Check for Robot Control Loops 

**Authors**: Arnav Grover  

**Link**: [PDF](https://arxiv.org/pdf/2610.27350)  

**Abstract**: Frontier models can control robots through agent harnesses that loop on one question: what motion comes next toward the user's goal? We show that this default loop carries out harmful instructions. With GPT-6 Astra controlling a low-cost SO-101 arm through the Inspect Robots harness on three household tasks whose hazard is common knowledge, such as "put the knife in the toy basket", the policy refused 0% of 15 trials, attempted the harm every time, and completed it in 27%. Its one exit, give_up, is meant for goals that cannot be achieved, and it never cited the hazard. We add an optional runtime safety check to the loop: before the first motion, the model is asked whether it would carry out the instruction, given the same camera frames. The check would have stopped all 15 hazardous instructions, including the 4 that completed the harm, and flagged none of the 15 harmless controls. Code and data: this https URL. 

---
# Steering Vision-Language Models for Visual-Contextual Privacy in Location Disclosure 

**Authors**: Wenyue Chen, Yuhan Cheng, Ruixin Yang, Ethan Mendes, Alan Ritter, Jingwei Sun, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27323)  

**Abstract**: Vision-language models (VLMs) can infer sensitive information such as geolocation from subtle visual cues, creating privacy risks when responses disclose more detail than the visual context warrants. We study visual-contextual privacy in geolocation disclosure, where a VLM should reveal location information only at the granularity appropriate to the visual context depicted in the image. We first show that this contextually appropriate granularity is recoverable from pre-generation hidden states, even when generated responses are misaligned, with disclosure levels exhibiting an ordered structure along a common hidden-state direction. Motivated by this, we propose an inference-time route-and-steer framework: an ordinal granularity router maps visual-contextual privacy representations to a disclosure route, and a disclosure-level-conditioned steering module applies route-specific hidden-state interventions during decoding. Across four VLM backbones, our method improves visual-context-aware disclosure alignment, better balances over- and under-disclosure, and incurs only modest changes on general capability benchmarks. It further transfers to another dataset with a distinct data distribution and privacy-risk taxonomy. 

---
# Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages 

**Authors**: Jalil Alva, David Harper, Abigail E. Thornton, Rafael S. Villalobos Alva, Leonardo Kenji Minemura Suazo, Konstantinos Karageorgos, Diego Velazquez, Sergio A. Serrano, Marvin Vargas Flores, Aaron Schliem, Fernando F. Migone  

**Link**: [PDF](https://arxiv.org/pdf/2610.27304)  

**Abstract**: Large language models are deployed globally, yet safety alignment and evaluation remain concentrated in English: harmful requests refused in English can elicit unsafe completions when translated into low-resource languages. We evaluate ten widely deployed LLMs across 79 languages (English plus 78 low-resource) under a translation-bypass threat model, using a metric conditioned on English safety, which separates translation bypass from unsafety already present in English. English safety does not transfer reliably: conditioned on safe English completions, mean bypass rates range from 21.9% to 44.4%. Bypass risk concentrates in a subset of language families, a pattern broadly shared across models. The shared ordering is confounded with translation degeneracy, but its variation across models is significant and survives every control we apply. We also show that per-model bypass rates do not identify a stable ranking: nine of ten models change rank across three conditioning schemes that estimate the same quantity, with no change to any underlying response. 

---
# Evaluating Context Compaction on the Questions That Come Later 

**Authors**: Junwei Yao, Yongxi Zhou, Zike Yang, Zhijun Wang, Jianan Liu, Yanxian Chen, Wenbo Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.27301)  

**Abstract**: Language-model agents compact their context for the question at hand, but the compacted context must also serve questions that come later. We evaluate compaction on these later questions, using four corpora that annotate the evidence of several questions per text and recorded agent trajectories, and compare every method with random deletion at the same budget. On MultiRC, keeping the sentences that gpt-6-luna (through the OpenAI Decisions API) judges necessary for the current question raises its accuracy on that question by 11.4 points at 25% of the tokens. On the later questions, however, no method that selects sentences for the current question does detectably better than deleting sentences at random. Deleting words instead of sentences does better: LLMLingua-2, which ignores the question, beats random sentence deletion at the same token count by 12.5 points on the later MultiRC questions. It keeps about twice random deletion's share of the evidence's content words, and at equal content-word retention its gain shrinks to 7.3 points (exploratory). On FEVER, label accuracy can overstate what the kept text supports: the gain of sentence selection is no longer detected when answers are scored against the kept evidence, and the 21.0-point gain of LLMLingua-2 is not shown to rest on the kept text. The ranking also depends on the setting: on Evidence Inference sentence selection beats LLMLingua-2 at a nominal 25% budget, and on $\tau$-bench trajectories, whose later tool calls need exact identifiers, it beats random deletion while LLMLingua-2 shows no detected gain. Scores on the question a compaction was made for do not reliably show how it serves later ones. 

---
# Generalizing Multimodal Foundational Models in Context-Driven Class Incremental Learning 

**Authors**: Avi Gupta, Nilotpal Sinha, Vishnu Raj, Sambuddha Saha, Pratik Joshi, Koteswar Rao Jerripothula, Tammam Tillo  

**Link**: [PDF](https://arxiv.org/pdf/2610.27298)  

**Abstract**: Class-Incremental Learning (CIL) aims to learn new classes continuously without forgetting the previously learned ones. Recent works in CIL have significantly advanced across various modalities; however, at the intersection of audio-visual modality, this problem remains underexplored. While representations from multimodal foundational models (MFMs) encapsulate rich static priors, our preliminary analysis shows that these representations struggle in incremental settings. Motivated by this, we propose a framework that generalizes frozen MFMs to audio-visual CIL. Specifically, we leverage their dense audio and visual representations and employ an attention-based contextual guidance module, in which the audio representations serve as task-stable contextual guidance for the visual representations, enabling effective class recognition in the incremental setting. To further mitigate catastrophic forgetting and preserve stability across tasks, we employ feature- and logit-level distillation, which are primarily responsible for the reduced forgetting. Extensive evaluations across various audio-visual CIL benchmarks demonstrate that our approach consistently outperforms state-of-the-art methods. Code is available at this https URL. 

---
# When Does Hate Transfer? A Large-Scale Study of Cross-Lingual Hate Speech Detection 

**Authors**: Zahra Bokaei, Walid Magdy, Bonnie Webber  

**Link**: [PDF](https://arxiv.org/pdf/2610.27285)  

**Abstract**: Effective detection of hate content is essential for maintaining healthy online communities, yet performance can vary substantially across languages. This work investigates a core practical question: Can training data from one language support hate-speech detection in another language with comparable performance? To answer this question, we conduct a large-scale study across 14 languages. \change{We observe consistent patterns in which certain language pairs transfer more effectively than others. Among the factors we measure, linguistic similarity is the strongest predictor of these patterns, with temporal overlap and dataset-size difference playing smaller roles. These measurable factors do not fully explain the observed patterns; a complementary qualitative analysis suggests that shared cultural cues and communicative norms may be associated with some of the remaining cases, particularly within several clusters. We present this cultural interpretation as a hypothesis from qualitative inspection rather than a measured cause.}{} Our analysis includes low-resource languages, offering practical insights into when cross-lingual transfer can support them and where its limitations emerge. These findings inform the design of more reliable multilingual hate-detection systems. 

---
# VPFlow: Value-Predictive Planning with Imperfect World Models for Offline Goal Reaching 

**Authors**: Yunzhe Hu, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27280)  

**Abstract**: Offline goal-conditioned reinforcement learning enables skill acquisition by reaching any goal from any state without dense rewards. Model-based methods allow for planning by scaling test-time compute to solve complex tasks. Combining both holds appealing promise for pre-training on suboptimal demonstrations and zero-shot adaptation via goal specification and planning. Yet existing methods assume effective planning requires accurate next-state prediction in pixel or latent space. In this work, we show that value prediction is more useful for planning, whereas accurate state prediction is often unnecessary. Our key insight is that offline goal reaching naturally admits a closed-form Monte Carlo return, enabling value regression to replace temporal-difference bootstrapping. We use this return as the sole supervision for latent world models: imagined latents are trained to predict goal-reaching value rather than match future states, implicitly preventing representation collapse. These planning-oriented representations let imperfect world models imagine on the level sets of value field beyond dataset transitions. We then present VPFlow, a value-predictive planner that pairs the world models with a one-step flow policy and plans in its noise space via evolutionary search. It enables policy improvement with imagined generalized advantage and reduces out-of-distribution queries common in action-space planning. On 30 state and pixel OGBench datasets, VPFlow surpasses model-free baselines by at least 30.3 and 10.1 percentage points in success rate, respectively, and outperforms latent planners LeWorldModel, PLDM, and TD-MPC2. It achieves this despite a latent prediction error orders of magnitude larger than with state prediction, while minimizing this error lowers success rate. This demonstrates that value-predictive representations, not exact state prediction, are what offline planning requires. 

---
# Certifying Repeatable Entry-Replay Witnesses for COBOL-to-Java Migration 

**Authors**: Andras L. Ferenczi, Mariya Bessonov, Allwyn Ponnudurai, Chandrakanth R. Mamillapalli, Krishna K. Maddula, Krishna K. Lingamneni  

**Link**: [PDF](https://arxiv.org/pdf/2610.27272)  

**Abstract**: Producing synthetic data for comparison testing a migrated COBOL application is a major challenge that agentic migrations face today. Our prior paper, the Agentic Method for Deterministic Validation of Legacy Code Migration, presents a deterministic oracle to facilitate such migrations, the Locksmith Loop. We now propose a novel method for achieving high branch and paragraph coverage through generated synthetic end-to-end test cases, by means of entry-replay test-case certification by observing the code execution trace on a surrogate COBOL program that mocks external interactions via a chronological external-operation tape. We applied our method to a very large COBOL program with 33,429 lines of code to generate 849 synthetic end-to-end test cases covering 2,973/3,806 source branches (78.1%) and 989/1,033 main-AST paragraphs (95.7%). Using the Locksmith Loop's projected-agreement validation, the generated Java code matched the COBOL surrogate on the specified observations across all 849 certified synthetic end-to-end test cases. During the campaign we detected 61 defects and fixed the Java generator accordingly. To test whether the coding agent could silently introduce business logic bugs, we seeded 450 defects into the Java code, such as deleted writes, deleted paragraph calls, and altered rounding, to verify how many of these would be detected by the gating methodology. The gate detected 47.6%, including all deleted writes and paragraph calls in executed code, but was unable to detect the rounding and precision defects. 

---
# Auditing Safety Preference Supervision with Recorded Derivations 

**Authors**: Chenhan Lyu, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.27266)  

**Abstract**: Preference labels align model training, but the judgments behind them are difficult to inspect or verify. In this paper, we examine how preference construction can be checked when intermediate decisions are recorded, and which of them remain uncertain. We present Symbolic Preference Labeling, a method in which a pretrained language model classifies predefined attributes of prompts and responses. Explicit rules assign response verdicts, and a fixed verdict ranking determines the preference label. Each preference label has a recorded derivation: the response verdicts and preference label can be recomputed from the stored attribute assignments, rules, and verdict ranking without rerunning the language model. On 200 prompts annotated by human raters, the language model produces valid structured outputs for 192 prompts. All eight parsing failures occur in the subset of self-harm prompts marked as harmful. Using 465 preference pairs, we train one policy model with direct preference optimization across five training seeds. Mean measured benign refusal is 40.82 percent on 1,319 benign prompts that resemble unsafe ones, and mean measured toxic refusal is 84.98 percent on 655 toxic prompts. This study distinguishes the reproducibility of preference labels from attribute correctness. The recorded derivations make preference construction open to inspection, while attribute correctness and semantic safety require separate validation. 

---
# Theorem prover Arend 

**Authors**: Fedor Part, Valery Isaev, Sergey Sinchuk  

**Link**: [PDF](https://arxiv.org/pdf/2610.27264)  

**Abstract**: This paper is an exposition of the theorem prover Arend developed at JetBrains Research. Arend language has been designed to facilitate the formalization of mathematics within the framework of univalent foundations and with the focus on constructive mathematics. It is based on a variant of homotopy type theory with cubical syntax that is more closely aligned with Martin-Löf Type Theory (MLTT) than fully computational cubical type theories, while still retaining some of their computational properties useful in practice. Among the most notable features of the Arend language is its powerful system of classes and records supporting subsumptive subtyping, partial implementations and anonymous extensions, which provides extensive flexibility for constructing hierarchies of definitions. Formalization in Arend is streamlined by the IntelliJ Arend plugin, which transforms IntelliJ IDEA into a full-fledged integrated development environment (IDE) for the Arend language. The main Arend library arend-lib includes formalizations in constructive algebra, topology and synthetic homotopy theory. 

---
# When Models Know but Can't Linearly Say: Relational Encoding Limits Linear Vulnerability Classification in Code LLMs 

**Authors**: Rui Melo, Andre Catarino, Claudia Mamede, Corina Pasareanu, Rui Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2610.27257)  

**Abstract**: Large language models can detect vulnerable code, yet the form of vulnerability encoding in their representations remains unclear. We analyze 8 code models spanning 6.7B-15B parameters on C programs using three datasets (DELTASECOMMITS, SVEN, and PRECISE-BUGS), comparing two diagnostic tasks on the same representations: identifying what patterns the code instantiates, and determining whether it is vulnerable. Models encode the first with high fidelity. Under the studied paired C-code datasets and linear-probing setup, vulnerability information is more accessible as a paired relational signal-whether code A is riskier than code B-than as a globally separable categorical label. This structural difference has implications for classification: the model reliably ranks which code is riskier but standard classifiers on isolated samples achieve limited accuracy because vulnerable and secure samples form overlapping distributions in representation space. 

---
# ValiAssist: Scaling LLM-generated Programming Feedback Validation by Demonstration 

**Authors**: Xiaohang Tang, Rehema Abulikemu, Peng Gao, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.27248)  

**Abstract**: Generative AI can create programming feedback at scale, but validating its alignment with instructors' pedagogical goals and strategies remains challenging. Instructors may struggle to fully specify validation criteria upfront, as these often emerge through inspecting concrete examples and revising misaligned feedback. We present ValiAssist, a validation-by-demonstration system that infers reusable validation logic from instructors' revisions of AI-generated feedback and allows instructors to inspect and refine that logic as it is applied across other student submissions. In a comparative study with 16 instructors, ValiAssist helped instructors create and refine validation criteria, producing feedback that better aligned with their pedagogical intent (91% vs. 57%) and more than doubling the accuracy of correcting problematic criteria (77% vs. 32%). Our findings suggest that validation by demonstration can help instructors turn judgments they know how to make on individual cases into explicit, reusable logic, while progressively clarifying where those judgments do and do not generalize. 

---
# Target-Aware Curation of Hallucination Stress Tests for Large Language Models 

**Authors**: Jiho Lee, Bingquan Zhang, Lin Ting Liang, Soroush Vosoughi, Weicheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.27247)  

**Abstract**: Hallucination stress tests can yield similar average scores across language models even when different prompts elicit hallucinations in each model. We develop a procedure that iteratively edits source-grounded seed prompts to expose hallucinations in a target model. An LLM editor revises each prompt in response to the target's answer and a judge's category-specific score and explanation. In the main setting, the target also receives feedback on its previous response. Experiments cover 500 seeds from seven datasets and ten topics, with intrinsic and extrinsic hallucinations evaluated across open-weight and closed-source models. Editing substantially increases judge-rated severity over the seeds. In the baseline configuration, retained responses fall in the judge's strong-hallucination bin in 77.2% of runs targeting extrinsic hallucination and 74.2% targeting intrinsic hallucination. Later editing rounds account for more than half of the mean score increase. Among 219 prompts that pass model-assisted source-validity checks, 89 elicit responses containing unsupported claims. Across five models, 200 curated prompts yield similar average severity scores, although pairwise Spearman correlations between prompt-level scores are only $\rho=0.0387$ to $0.1897$. Curated prompts can be reused across models, but their diagnostic value depends on source validity and the failures they expose in each model. 

---
# AutoRedClaw: Automated Red-Teaming for Claw-Style Agents 

**Authors**: Hao Li, Tongyan Hu, Weidi Luo, Shanghao Shi, Xiaogeng Liu, Aiden Suh, Yulong Cao, Ning Zhang, William Yeoh, Yevgeniy Vorobeychik, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27244)  

**Abstract**: Agent systems are shifting from task-scoped tool-use agents to persistent, personalized assistants such as OpenClaw. These claw-style agents maintain configuration files (e.g., this http URL), long-term memory, and reusable skills. Once poisoned, this persistent state is repeatedly reloaded into the agent's context, creating persistent threats across tasks and sessions. Existing security evaluation systems struggle to evaluate these complex claw-style systems since they rely heavily on manually constructed test cases and environments, which limits their scale and coverage; in addition, these manually designed cases quickly become outdated as agents evolve, and their evaluation settings leave a gap with how claw-style agents are deployed in practice. To address this, we introduce AutoRedClaw, an automated red-teaming framework that generates end-to-end attacks together with the environments needed to execute them. AutoRedClaw combines a strict quality-checking protocol with execution-based validation to ensure the correctness, realism, and effectiveness of generated samples, and adopts task-driven environment construction to keep costs low. To achieve broad coverage, it organizes generation along risk categories, attack strategies, attack stages, and attack surfaces, and uses experience-guided exploration to steer generation toward under-explored regions. AutoRedClaw achieves an effective attack rate above 30% at less than $0.10 per successful sample. From its outputs, we construct RedClaw-Bench, a 330-case benchmark that yields substantially higher attack success rates (ASRs) than AgentDojo, AgentDyn, and DTap. Across seven harnesses and four frontier models, most agent systems exceed 50% ASR, showing that our red-teaming pipeline provides a strong evaluation signal even for the most advanced agent systems. Our code and benchmark are released at this https URL. 

---
# MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows 

**Authors**: Ruoxin Wang, Yuanting Pan, Binxu Li, Ruiyang Ji, Yaqian Chen, Jiayuan Ding, Qifan Wang, Lifu Huang, Mingjie Li, Yu Zhang, Ehsan Adeli, Kilian Pohl, Zhe Xu, Zihao Lin, Haoyu Dong, Yixin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27233)  

**Abstract**: Medical workflows often require multiple interdependent analyses, but existing specialist and generalist AI models are not designed to coordinate them. We introduce MMedAgent-Plus, a multimodal medical tool-use agent that combines a vision-language planner with 18 leading specialist models spanning 12 tasks and 8 modalities. Through a unified tool interface, the planner selects specialists, executes them in sequence, and synthesizes responses grounded in their outputs. The planner is instruction-tuned on 208,928 samples from 32 public medical benchmarks, covering single- and multi-tool interactions across single and multiple rounds. We separately evaluate tool orchestration, response synthesis, and end-to-end execution on held-out samples. MMedAgent-Plus achieves tool-call accuracies of 91.0%, 98.1%, and 80.7% in single-tool multi-round, multi-tool multi-round, and multi-tool single-round settings, respectively. Given correct tool outputs, MMedAgent-Plus achieves scores of 14.4-14.8 out of 15 for faithfulness, completeness, and conciseness across 600 evaluation samples. In end-to-end execution without pre-supplied tool outputs, MMedAgent-Plus correctly invokes the required specialist in 90.6% of 350 evaluation cases spanning seven tasks. Its final-answer accuracy reaches 0.98, matching the dedicated leading specialist and exceeding the strongest baseline (0.28). 

---
# Profile-DP: Differential Privacy for User-Controlled Profile Release in Personalized LLM Inference 

**Authors**: Mingchen Li, Zifan Peng, Kewei Sha, Lavinia F Pieptea, Heng Fan, Junhua Ding, Meikang Qiu, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.27213)  

**Abstract**: Personalized large language models (LLMs) increasingly rely on profile-conditioned inference, pairing public requests with rich, user-specific facts. This paradigm exposes the user profile itself, together with the user's explicit sharing preferences. We introduce Profile-DP, a policy-aware local differential privacy framework designed specifically for user-profile release in personalized LLM inference. Profile-DP provably bounds the release of user-withheld facts while treating public instructions, profile schemas, and authorized facts as public side information. By generating a privatized profile that composes with the public instruction prior to answer generation, our framework decouples formal privacy guarantees from downstream model behavior. We develop two finite-domain mechanisms satisfying Profile-DP: DP-SLOT independently perturbs protected attributes, while DP-BLOCK uses public semantic decisions to coordinate their release and jointly sample related attributes. A positive, public compatibility prior connects these attributes while preserving a profile-level privacy bound. On PEEP, DP-BLOCK leads the utility composite among DP methods at all 13 structured budgets and is closest to the ideal privacy--utility point in 18 of 26 format--budget settings under our composite weighting. Direct measurements reveal complementary strengths: DP-BLOCK discloses less, while DP-SLOT retains more authorized facts. These results make policy-aware profile release a practical basis for studying privacy and personalization. 

---
# Language Guidance for Supervised Vision Training: An Empirical Study of Generalization 

**Authors**: Shruthi Gowda, Nikita Jain, Elahe Arani, Bahram Zonooz  

**Link**: [PDF](https://arxiv.org/pdf/2610.27208)  

**Abstract**: Deep neural networks have achieved remarkable success on vision benchmarks, yet they continue to struggle with many generalization challenges. Supervised vision training relies on one-hot labels, which provide limited information about semantic structure and shared attributes between classes. This limited supervision can leave visual representations vulnerable to distribution shifts, spurious correlations, texture bias, adversarial perturbations, and forgetting in sequential learning settings. We study whether pretrained language models can serve as lightweight auxiliary supervision for vision training without requiring paired image-text data, prompt engineering, or contrastive objectives. Specifically, we evaluate two forms of language guidance, Explicit Language Guidance (ExLG) and Implicit Language Guidance (ImLG). We conduct a comprehensive evaluation across six generalization regimes, including in-distribution, out-of-distribution generalization, shortcut and spurious correlation resistance, texture and shape bias, adversarial robustness, and continual learning. Our analyses show that the two mechanisms have complementary strengths, with explicit guidance consistently benefiting in-distribution, low-data performance, and continual learning retention, while implicit guidance is often more useful in shortcut-sensitive settings and under stronger adversarial perturbations. Importantly, both are lightweight and add minimal parameters and training overhead. The analyses characterize when language-derived structure helps supervised vision training and provides a practical roadmap for using off-the-shelf pretrained models from another modality as auxiliary supervision. 

---
# LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents 

**Authors**: Ocean Monjur, Daud Ahmad Nisar, Shahriar Kabir Nahin, Tadesse Destaw Belay, Sushruti Mishra, Israel Abebe Azime, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2610.27205)  

**Abstract**: Web agents are typically evaluated on English-language interfaces, leaving their ability to operate across the world's multilingual websites largely unexplored. This gap is especially significant for low-resource languages, where standardized resources for evaluating web agents remain limited. To address this gap, we introduce Low-Resource Web Agent Benchmark (LoWAgent), a benchmark for evaluating web agents across different low-resource languages (LRLs) in comparison to the default, English. LoWAgent evaluates web agents across English and four LRLs: Hindi, Bengali, Urdu, and Amharic. The benchmark comprises of six fully localized websites, instantiated across five languages to yield 3,000 task instances. We evaluate three closed-source and five open-source models across all five languages on all six websites. Our findings show that both closed-source and open-source models experience performance degradation for LRLs compared to English, with substantially greater performance drops observed among the open-source model agents. We further evaluate the robustness of the models on realistic interface disruptions, such as cookie notices and advertisement pop-ups, and find that these further degrade agentic performance. Finally, we show that post-training on a small subset of LoWAgent traces can significantly improve low-resource language agentic task performance, demonstrating the usefulness of our benchmark. 

---
# CARLAR Influence: A Structured Hallucination Risk in Multimodal Cheque Extraction and an Exploratory Compressed-Image Countermeasure 

**Authors**: Joong Ho, Choi, Aditya Paul, Nidhi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.27192)  

**Abstract**: Multimodal large language models (LLMs) are increasingly used for structured extraction from financial documents, but their behavior under visual ambiguity is poorly understood. We study CARLAR (Courtesy Amount Recognition and Legal Amount Recognition) influence: a model returns the same amount for both cheque fields even though their annotations differ, including cases with a blank annotation encoded as zero. Across ten visual prompting conditions, we evaluate GPT-4.1 and Claude 4.5 Sonnet on 243 primary cheques (124 with two observed unequal amounts) and 598 strict mismatches from two additional datasets. Under courtesy redaction, GPT-4.1 CARLAR reaches 73.7 percent (95 percent bootstrap interval: 68.3 to 79.0; p below 0.001 versus baseline). Model-reported confidence can remain high when predictions agree; expected calibration error reaches 0.93 among CARLAR cases. We also test a compressed-image few-shot intervention that presents examples as one visual table without model fine-tuning. On the 243-case cohort, Claude's Control condition increases paired joint extraction accuracy by 11.5 percentage points (95 percent interval: 6.2 to 16.9; adjusted p = 0.00054), but its strict-124 gain is smaller and not significant after correction. The intervention's effect on predicted-field agreement is model-dependent: it decreases for Claude and increases for GPT-4.1. These results motivate validation methods that account for visual context and dependencies between extracted fields. 

---
# On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models 

**Authors**: Chenpan Li, Thies Brockmoeller, Stefanie Jegelka, Sharut Gupta, Nikolaos Karalias  

**Link**: [PDF](https://arxiv.org/pdf/2610.27190)  

**Abstract**: Modern language models are increasingly asked to handle several tasks within a single prompt. Doing so requires solving each task correctly while integrating the resulting answers into a coherent response. In this paper, we identify a fundamental limitation in this process, where models make substantially more errors when tasks are composed, even when they can solve every constituent task reliably in isolation. This degradation cannot be fully explained by the accumulation of independent errors. Across both open-weight models and the frontier model Claude Opus 5, our findings instead point to an implicit thinking budget underlying this effect. In particular, under composition, models use substantially fewer tokens per task than in isolation, even when the full sequence remains well within the context window. Tasks that require longer solution traces in isolation are consequently more $\textit{fragile}$ under composition, i.e., models make more errors on prompts containing these tasks. In isolation, non-fragile tasks receive more reasoning than they need and waste compute. However, by grouping tasks according to their fragility, the implicit budgeting can be manipulated to reduce token usage by $43$\% across a range of math and linguistic tasks while also preserving accuracy. 

---
# Visual Task Modeling and Automated Planning for Humanoid Robots in Education of Individuals with Autism Spectrum Disorder 

**Authors**: Lada Kudláčková  

**Link**: [PDF](https://arxiv.org/pdf/2610.27162)  

**Abstract**: In robot-assisted education, a robot must understand how to complete a task in order to effectively guide a student through it. We present a novel method that enables a semi-autonomous, camera-equipped robot to recognise the task goal and plan actions to achieve it. Our work focuses on Shoebox Tasks, an activity commonly used in the education of individuals with Autism Spectrum Disorder (ASD). These tasks are designed so that the goal can be inferred directly from their visual structure. We have developed a solver that uses a vision-based Large Language Model to convert photos of a Shoebox Task into a Planning Domain Definition Language (PDDL) representation and generate action plans for solving it. We have also implemented a robotic interface that enables a Nao robot to autonomously execute various Shoebox Tasks and assist a student by suggesting next steps or signalling detected failures through action demonstrations, voice commands, or gestures. With an adapted robotic interface, the solver can be integrated into a variety of educational robots. No programming expertise is required from teachers, and the solver operates exclusively on captured images as input. Research in special educational needs indicates that students with ASD can benefit substantially from robot-assisted teaching, and our system aims to strengthen these learning opportunities while also fostering interest in robotics among individuals with ASD. 

---
# Understanding the Benefits of Scaffolds through Computational Separations 

**Authors**: Jingchu Gai, Adam Block, Andrej Risteski  

**Link**: [PDF](https://arxiv.org/pdf/2610.27157)  

**Abstract**: Inference-time harnesses, i.e., external procedures that call a model many times and keep state across calls, can substantially improve model capabilities. In this paper, we investigate the benefits of using such harnesses and the extent to which these benefits can be internalized through training. More precisely, we conceptualize a harness $\mathcal{A}$ as an arbitrary wrapper around a base model $M$ and attempt to distill a student model from the harness's traces. Empirically we find that, even with simple scaffolds and small models, the inference-time harness significantly outperforms a distilled student. We then prove that such a gap is due to a fundamental representational limitation: inference-time harnesses can compute a closure of their per-call primitive that is $\NL$-complete even when the primitive is a trivial $\AC^0$ lookup, and that no fixed-depth forward pass can represent under the widely believed conjecture that $\NL\not\subseteq\TC^0$. Our theory suggests that expanding the representational capacity of the student relative to the task at hand, e.g., by enlarging the model, making the task easier, or lengthening its chain-of-thought, is necessary to close the gap, which we verify empirically. Finally, we scale to an agentic harness on competition math, where the same gap reappears. Overall, our results show that a harness confers on the base model a profound gain in representational power, one that training a same-class student on its traces cannot transfer back into the weights. 

---
# Model Exposure on One Device Endangers Others: Rethinking Edge Neural Network Security across Devices 

**Authors**: Yuyang Zhang, Kangjie Chen, Yihui Jin, Run Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27152)  

**Abstract**: Neural networks (NNs) are increasingly deployed on edge devices such as mobile and IoT platforms, offering low latency and improved privacy compared to cloud-based deployment. However, on-device models embedded in mobile applications can be unpacked and exposed to adversaries, enabling cross-node attacks in centrally distributed ecosystems, where adversarial examples crafted on one compromised device transfer effectively to other nodes sharing the same models. To address this threat, we propose VeilNN, a defense framework that reduces adversarial transferability across on-device models via model obfuscation. The core idea of VeilNN is to introduce controlled model diversity while preserving original task functionality. Specifically, VeilNN applies stealthy and irreversible obfuscation at both the architectural and parameter levels through network cell obfuscation and layer assembling, and is applicable to diverse NN architectures, including MLPs, CNNs, and Vision Transformers. Extensive experiments demonstrate that VeilNN effectively defends against cross-node attacks, achieving an average defense success of 49.19% and up to 81.19% across four adversarial attack types, with more than 33.16% improvement over existing defenses. 

---
# ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy 

**Authors**: Yuqiao Wen, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2610.27142)  

**Abstract**: Data selection plays a central role in modern large language model training, as the quality of training data strongly influences downstream performance. A common approach is to curate a development set that encodes the desired skills (e.g., reasoning or multilinguality) and select data that are similar to them. Existing methods compute similarity using cosine distance over either the model's hidden states or gradients. However, we observe that this max-similarity strategy induces a mode-seeking bias: the selected data clusters tightly around the development samples. To address this, we propose ICLHerd, a data selection framework that leverages the von Mises-Fisher kernel to balance mode-seeking with mode-averaging. We further introduce in-context learning as a validation proxy, enabling efficient hyperparameter search over the balance trade-off. Results show that ICLHerd outperforms existing LLM data selection algorithms, highlighting the importance of balancing mode-seeking and mode-averaging. 

---
# Representing short-term movement to reveal animal identity and learning progress 

**Authors**: Ping Zhang, Quang-Huy Nguyen, Yingtie Lei, Sameera Shridhar, Zachary Leveroni, Ellie Quattrocchi, Jai Y Yu, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27140)  

**Abstract**: How learning progresses over time and varies across individuals is a longstanding question in cognitive and behavioral science. However, a major challenge is identifying observable behavioral measures that faithfully capture learning progression. Existing approaches commonly rely on predefined behavioral features or task-level performance, while fine-scale posture and movement in freely behaving animals remain less explored. In this paper, we ask: can short-term movement trajectories reveal learning progress and individual identity? We introduce a self-supervised representation learning framework built on a simple hypothesis: behavioral characteristics associated with identity and learning remain relatively stable over nearby time intervals. Using temporal proximity as supervision, our framework learns compact representations directly from movement trajectories without manually defining behavioral states or kinematic features. Using behavior data from freely behaving rats, we show that the learned representations not only improve prediction of identity and learning-related readouts over existing methods, but also provide a tool for investigating broader scientific questions about when and how these signals appear in behavior. Specifically, trajectories as short as one second contain informative learning and identity signals. Egocentric pose dynamics are more predictive of identity, whereas allocentric motion is more predictive of learning-related variation, and both signals are strongest at decision junctions in the alternation cohort. We further use a representation-conditioned diffusion model to inspect movement patterns associated with identity and learning progress. Our findings establish short-term movement as an informative behavioral "fingerprint" for studying learning and individual differences. Our code is available at this https URL. 

---
# Ontic state recovery through prediction 

**Authors**: Ping Zhang, Quang-Huy Nguyen, Yingtie Lei, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2610.27139)  

**Abstract**: We encounter a system through its responses, yet its state concerns how the system is. Predicting further responses promises knowledge beyond what is immediately observed. What that knowledge tells us about the system's state depends on what we mean by a state in the first place. We give this notion a model-relative meaning: a state is the complete response profile over a specified query universe, and configurations agreeing on every query are regarded as the same state. From this perspective, a learned state is understood through the distinctions between these profiles that it retains. Prediction concerns selected observations; recovery concerns the profile as a whole. We examine their relationship through query ambiguity, state-space geometry, and the choice of prediction targets. Synthetic systems with known response laws and neural recordings provide complementary settings for this investigation. The aim is to clarify how prediction can become a way of knowing a system, and what it would mean for a learned state to preserve the state being studied. 

---
# Mind Your Boundaries: Boundary-Aware Model Obfuscation against Cross-node Attack 

**Authors**: Yuyang Zhang, Yihui Jin, Kangjie Chen, Leyang Wu, Run Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27135)  

**Abstract**: With the advancement of powerful AI chips, on-device machine learning is gaining traction for its low latency and privacy protection. However, when edge devices are compromised, attackers can obtain white-box access to the deployed models by reversing the apps directly to obtain the knowledge of peer devices, which is known as Cross-Node Attacks, like performing efficient adversarial attacks. Most existing defenses target known vulnerabilities, yet white-box access still enables attackers to uncover more unknown weaknesses. This creates a critical risk for edge-device deployments, where one compromised device can endanger all models distributed from the same source. Motivated by findings that the model robustness is primarily influenced by the curvature of the decision boundary, we investigate how model parameters shape its curvature. In this paper, we propose a novel method, BMO, that obfuscates model parameters to diversify the decision boundary further resisting cross-node attack while preserving its benign accuracy. Specifically, we explore decomposing model parameters into boundary-position and boundary-curvature components based on null-space constraints, then models with differentiated decision boundaries are instantiated by injecting noise into the boundary-curvature component. Across all four major types of adversarial attacks with four widely-deployed on-device models, BMO achieves an average success rate of 68.68% in defense, far exceeding the most competitive baseline, which attains only 22.32% in defending against cross-node attacks. Moreover, BMO achieves comparable performance in benign accuracy compared with baselines. 

---
# Reconstruction-Optimized Expert Pruning for Mixture-of-Experts Language Models 

**Authors**: Jelena Markovic-Voronov, Petros Prastakos, Kayhan Behdin, Jincheng Cao, Zhipeng Wang, Shayan Mohajer Hamidi, Yuanda Xu, Zhengze Zhou, Wenhui Zhu, Rahul Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2610.27123)  

**Abstract**: Mixture-of-Experts (MoE) models reduce computation through sparse expert activation, but deployment still requires storing a large number of expert parameters. We study post-training expert pruning using task-domain calibration data as a subset-selection problem: given a pretrained MoE layer, select a fixed number of experts that best reconstruct its original output. Exact subset search is combinatorial and becomes intractable for modern MoE models with many experts per layer. We develop OptEP, a scalable continuous relaxation that directly optimizes differentiable expert-selection variables, followed by a lightweight reconstruction-based update of the retained expert weights. Across multiple MoE architectures and tasks spanning mathematical reasoning and code generation, our approach improves over the strongest existing expert-pruning baselines, with absolute accuracy gains of up to 15.05% on Qwen3-30B-A3B, 14.55% on Qwen3-235B-A22B, 11.06% on GLM-4.5-Air, and 16.20% on Mixtral-8x7B. The improvements are particularly pronounced under aggressive pruning, demonstrating that reconstruction-driven expert selection and weight adaptation can preserve substantially more of the original model's performance while reducing the number of retained experts. 

---
# Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA 

**Authors**: Jennifer Hsia, Zachary C. Lipton, Aarti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.27114)  

**Abstract**: Retrieval-augmented QA can fail even when the correct answer is in the retrieved evidence. This can happen when a competing answer satisfies some, but not all, of the question's requirements and is mentioned frequently in the retrieved documents, which can make it appear to be the best-supported answer. To test whether this retrieval skew actually affects the decision, we manipulate it directly: increasing the prevalence of passages that favor the competing answer makes the model more likely to select it. This points to a decision problem that candidate-level support alone cannot resolve. We therefore introduce Joint Constraint Aggregation (JCA), which compares candidates against the same question-derived requirements using the retrieved evidence. Across three evaluation settings and three instruction-tuned backbones, JCA improves overall accuracy, and its advantage grows when retrieval favors a competing answer. Together, these results suggest that when partial matches and retrieval prominence pull models toward a competing answer, comparing candidates against the question's full requirements can make selection more reliable. 

---
# Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling 

**Authors**: Tong Ding, Caiwei Tian, Dongmin Bang, Ming Y. Lu, Sophia J. Wagner, Andrew Zhang, Rowland Pettit, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2610.27047)  

**Abstract**: EHR foundation models promise scalable clinical prediction across health systems, yet their deployment is constrained by a tokenization bottleneck. The same clinical concept can appear as different ontology codes, local identifiers, synonyms, or free-text descriptions, which standard EHR tokenizers treat as unrelated symbols. A model trained on MIMIC-III with a closed-vocabulary tokenizer can use only 13% of eICU events. We introduce MEDIATOR, a discrete ontology-agnostic tokenizer that maps any clinical concept expressed as text into a shared vector-quantized codebook, turning EHR harmonization into open-vocabulary tokenization without manual mappings or a fixed vocabulary. Trained with synonym supervision and guided by enriched concept descriptions, code co-occurrence, and relation graphs, MEDIATOR collapses equivalent expressions while organizing distinct concepts in a medically meaningful discrete space. Used as the token interface of an EHR Transformer, MEDIATOR improves mean AUPRC over the strongest prior tokenizer by 4.0 points (8% relative) on 20 MIMIC-III tasks and by 8.6 points (29% relative) in zero-shot transfer to 14 eICU tasks, where it uses all events. The transfer gain combines coverage and representation, as MEDIATOR remains the best tokenizer under identical inputs (+1.0 and +1.9 points on operational and subcohort-mortality tasks). Same-encoder controls show that the text encoder drives in-domain accuracy, while the learned objectives of MEDIATOR drive transfer robustness and entity-linking quality. MEDIATOR also enables transferable attribution across hospitals. 

---
# Cross-Lingual Reusability of LLM Pipelines: Financial Extraction, Multilingual Reasoning, and Semantic Screening 

**Authors**: Joong Ho, Choi, Sherine George, Jiayang Zhao, Jayesh Rathi, Ashwin Kandath, Ananya Sane, Vishvak Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.27043)  

**Abstract**: Financial institutions increasingly deploy large language models (LLMs) in multilingual markets, raising a practical question: can pipelines developed and validated in English be reused across languages, or must they be rebuilt per locale? We test cross-lingual reusability along three dimensions--accuracy, reasoning-effort transfer, and behavioural stability--through three complementary tasks. Task 1 (cheque extraction) and Task 2 (Mintaka QA with pinned evidence across nine languages and nine model-effort configurations) test whether accuracy and effort settings transfer; Task 3 (FinanceBench pilot, 50 items over SEC 10-K filings in four languages) validates whether these patterns generalise to financial content. Accuracy reusability is broadly confirmed: Gemini 3.5 Flash achieves 95.2-97.6% extraction accuracy across prompt languages, and FinanceBench per-language means span 5.8pp (77.1-82.9%), though the language-difficulty ranking reverses between domains (Hindi is worst on Mintaka, best on FinanceBench at 82.9%). Effort reusability is domain-dependent: higher reasoning effort yields marginal gains on general-knowledge QA but up to 8pp on financial content, and effort configurations that help in English help proportionally in other languages. Behavioural reusability holds: citation-set overlap with English averages 77-79% Jaccard, with no single language consistently worse. Together, these results support qualified reuse of English-built pipelines, but per-domain, per-language validation remains necessary--neither semantic similarity nor self-reported confidence reliably predicts cross-lingual accuracy. 

---
# Robustness of Internal Signals for Hallucination Detection in Vision-Language Models 

**Authors**: Allison Chen, William Yang, Salma Abdel Magid, Jonathan Williams, Olga Russakovsky, Esin Tureci  

**Link**: [PDF](https://arxiv.org/pdf/2610.27041)  

**Abstract**: When given an image and a text prompt, vision-language models (VLMs) can produce seemingly sophisticated text responses. However, they are also prone to hallucinate objects in their output, which can have detrimental consequences in safety-critical applications. To detect these hallucinations, certain methods interestingly leverage internal signals from the VLM as it generates the text response; potentially, these methods use the same signals that led to a hallucination to detect it. We conduct an in-depth analysis to understand the robustness of these detection methods. First, we show that a naive baseline of a generated word's token position in the VLM text response achieves competitive performance. Further, we reveal how some detection methods rely on token position and use this heuristic as a shortcut. Second, we explore detection methods' sensitivity to language and vision modalities. Our experiments suggest that strong detection performance is often driven by detecting language-based hallucinations, and when methods rely on token position, they tend to be less visually grounded and exhibit reduced robustness when hallucinations occur earlier in a text response. Lastly, we demonstrate that different detection methods can carry complementary signals; when ensembling methods together, adding weaker methods to an ensemble of higher performing ones still offers performance improvements. Altogether, while there may be useful internal VLM signals to detect hallucinations, our investigation reveals important limitations of the current methods that try to leverage them. 

---
# BR-MoE: Budgeted Semantic Release for Privacy-Preserving Traffic Analytics 

**Authors**: Abdolazim Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2610.27026)  

**Abstract**: Traffic cameras support useful analysis, but their outputs can reveal private information. Replacing images with text reduces direct visual exposure. It does not place a formal bound on disclosure. The published reinforcement learning and mixture-of-experts framework refines scene descriptions through relevance, coverage, and brevity rewards. It does not account for repeated releases under a fixed privacy budget. This paper proposes budgeted semantic release as an extension of that framework. Four scene experts first process isolated video chunks. Their outputs are mapped to a fixed, bounded semantic schema. A reinforcement learning controller selects which expert summaries to release and how much privacy budget to spend. The controller uses public task requests and previous private releases. A separate privacy filter enforces the total budget. Calibrated noise is added before the summaries are converted to text. Under explicit chunk isolation and bounded-event assumptions, the complete release transcript satisfies an event-level differential privacy bound. The design also supports reuse of previous summaries without new disclosure. An evaluation protocol covers traffic utility, privacy attacks, release freshness, and computational cost. Illustrative tables show the intended reporting format. They do not establish empirical gains. The proposed contribution is a task-aware release interface with auditable privacy accounting for repeated traffic descriptions. 

---
# Reliable Estimation of Differential Privacy under Selection Bias 

**Authors**: Valentin Dorseuil, Meenatchi Sundaram Muthu Selva Annamalai, Jamal Atif, Olivier Cappé, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2610.27021)  

**Abstract**: Statistical privacy audits are essential for verifying the theoretical privacy guarantees of differentially private (DP) mechanisms. Auditors typically use Membership Inference Attacks (MIAs) to estimate a lower bound on the true privacy leakage of DP mechanisms at a specified confidence level, which should not exceed the theoretical privacy upper bound guaranteed by DP. However, we found that many auditing implementations in prior work introduce a selection bias when running MIAs, which leads to statistically invalid and overestimated lower bounds. Because most audits are not tight, this bias has thus far gone unnoticed: the estimated lower bounds remained well below the theoretical upper bounds. Nevertheless, accurately evaluating this lower bound is critical, as a flawed statistical assessment could lead to false alarms on correct mechanisms or an incorrect evaluation of the guarantee's tightness. This work employs tools from the statistical literature to correct this bias. We present an overview of the different correction methods and assess their respective advantages. Furthermore, we perform a comprehensive re-analysis of results returned by prior audit implementations and show that without statistical correction, these audits violate the theoretical upper bounds up to four times more often than their claimed confidence level implies. With correction applied, the resulting lower bounds are statistically valid, and violations occur at no more than the nominal rate. 

---
# Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards 

**Authors**: Yunan Zeng, Jinhao Jing, Xi Cheng, Yuwang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.27012)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) trains a language model by sampling responses, scoring them with an automatic verifier, and updating the policy from the resulting rewards. The policy therefore receives direct learning signals only from trajectories that appear in its rollouts. Yet repeated sampling from the same policy can concentrate on similar reasoning paths, leaving useful alternatives unexplored under a fixed rollout budget. We introduce Activation-Space Exploration (ASE), an online method for diversifying training rollouts. By clustering activation differences between correct and incorrect responses to the same problem, ASE builds a structured bank of latent exploration vectors. Different vectors are then applied to subsequent rollouts to explore alternative reasoning paths. On Qwen3-1.7B-Base and Qwen3-8B-Base trained on DAPO-Math-17k, ASE improves average Mean@8 across six mathematical reasoning benchmarks over GRPO by 4.87 and 1.87 percentage points, respectively, under the same rollout budget. 

---
# HierDTA: Hierarchical Structure-Aware Representation Learning for Drug--Target Affinity Prediction 

**Authors**: Huangchao Xu, Baohua Zhang, Qian Liu, Zhong Jin, Yi Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.27001)  

**Abstract**: Drug--target affinity (DTA) prediction plays an important role in computer-aided drug discovery, where reliable computational models can help prioritize compound--protein pairs before experimental validation. A key challenge in DTA prediction is to effectively mine multi-level molecular information from drug structures while accurately modeling the binding-site environment of target proteins. To address these issues, we introduce HierDTA, a hierarchical structure-aware framework for DTA prediction. \textcolor{black}{HierDTA represents each drug as an atom--motif heterogeneous graph and introduces an augmentation module that propagates functional-motif semantics to the corresponding atoms. For proteins, kinase-domain structures are converted into residue interaction graphs. The resulting atom and residue representations are integrated using cross-attention to capture pair-specific relationships, while molecule fingerprints and pretrained ESM-2 embeddings provide complementary global information. Compared with state-of-the-art methods, HierDTA achieves superior performance on the Davis dataset across all metrics and attains the highest \(r_m^2\) of 0.791 on KIBA. It also exhibits strong generalization under cold-start settings, particularly on KIBA, where MSE is reduced by 21.7\% in the unseen-protein scenario. Ablation study and interpretability analysis supported the contributions of the hierarchical augmentation and interaction components.} Case studies further showed that HierDTA provided early enrichment of candidate bioactive compounds. These results indicate that HierDTA is an effective framework for affinity prediction and target-specific virtual screening. 

---
# CoASFL: Coupled Consistency Optimization for Asynchronous Split Federated Learning 

**Authors**: Junhao Cheng, Yunming Liao, Hongli Xu, Yang Xu, Zuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.26992)  

**Abstract**: Split Federated Learning (SFL) has emerged as a promising paradigm for training large-scale deep models on resource-constrained edge devices. Although asynchronous SFL alleviates the straggler bottleneck caused by system heterogeneity, it inevitably introduces two coupled dimensions of inconsistency: horizontal inconsistency across clients and vertical inconsistency between clients and the server. These inconsistencies, further exacerbated by statistical heterogeneity, result in biased activation representations, model divergence, and severe client drift. To address these challenges, we propose CoASFL, an inconsistency-aware asynchronous SFL framework that incorporates a dynamic activation merging mechanism and a consistency-aware model aggregation mechanism. Specifically, CoASFL constructs more consistent activation sequences from heterogeneous clients to alleviate horizontal inconsistency, and dynamically adjusts aggregation weights according to client participation frequencies and progress gaps to mitigate vertical inconsistency. By jointly coordinating activation-level and model-level consistency, CoASFL effectively suppresses inconsistency propagation during asynchronous training. Extensive experiments on an 80-device heterogeneous edge testbed demonstrate that CoASFL consistently outperforms state-of-the-art SFL approaches, accelerating training by 1.61$\times$ to 3.74$\times$ and improving final model accuracy by up to 3.26\% under heterogeneous scenarios. 

---
# 3D-MoE: Complementarity-Guided Hardware-Software Co-Design for MoE Decode on 3D-Stacked NMP 

**Authors**: Yudong Pan, Lian Liu, Shixin Zhao, Yintao He, Kaiyan Chang, Tianhua Han, Chenyang Ai, Haonan Zhu, Mengdi Wang, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26991)  

**Abstract**: Mixture-of-Experts (MoE) have become a mainstream way to scale Large Language Models (LLMs), because each token activates only a small subset of experts, reducing inference cost. However, MoE decode remains highly memory-bound and exhibits strongly skewed, time-varying expert loads. 3D-stacked DRAM near-memory processing (NMP) provides the bandwidth that decode demands, yet its tiled organization makes load balancing expensive: expert parallelism (EP) preserves locality but suffers from hotspot-induced imbalance, while more distributed strategies weaken locality and increase NoC traffic.
To address this locality-balance conflict, we observe that domain-specialized token routing creates offsetting expert-load fluctuations across decode steps, a property we call expert complementarity. Using this insight, we propose 3D-MoE, a hardware--software co-designed system for MoE decode on 3D-stacked NMP. 3D-MoE uses offline, one-time complementarity-guided grouping and locality-aware mapping to establish coarse-grained inter-group balance, handles residual intra-group hotspots through a lightweight Collaborative Memory-Network (CMN) that streams each hot expert's weight tiles once from its host DRAM channel to helper cores over a reconfigured local broadcast path, and uses cost-aware online remapping to remain robust to workload drift. Across MoE models, 3D-MoE achieves an average 4.63$\times$ MoE-layer speedup over EP and improves end-to-end throughput by 1.58$\times$--1.86$\times$ over the strongest NMP baselines, with 2.11\% additional logic-die area. 

---
# Dynamic Dual-Feedback Conformal Inference for Time Series Forecasting 

**Authors**: Songlin Du, Ling Luo, Uwe Aickelin  

**Link**: [PDF](https://arxiv.org/pdf/2610.26987)  

**Abstract**: Quantifying uncertainty in time series forecasting is particularly demanding because sequential data exhibit temporal dependence and are prone to distributional changes. Conformal inference has emerged as a powerful uncertainty quantification approach through the construction of prediction sets. Recent advances have introduced online conformal methods that adaptively adjust prediction thresholds through feedback mechanisms. However, the existing feedback mechanism typically relies solely on miscoverage indicators (actual feedback), whether the true label falls within the interval at each time step, while overlooking the empirical prediction threshold (estimated feedback) that is derived from the oracle conformal method. In this paper, we propose Dynamic Dual-feedback Conformal Inference (DDCI), which incorporates a dual-feedback mechanism consisting of actual feedback and estimated feedback. The former drives the primary adjustment of the intervals based on true observations, while the latter dampens excessive expansions or contractions by leveraging empirical thresholds from conformal inference during updates. By balancing these two signals, DDCI achieves more stable and narrower prediction intervals in sequential settings while maintaining near-nominal coverage in practice. 

---
# SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection 

**Authors**: Selma Khelili, Zohra Abdeli  

**Link**: [PDF](https://arxiv.org/pdf/2610.26972)  

**Abstract**: We present SZED's submission to ArGuard 2026 Track A: Arabic hateful meme detection. The task has two parts: binary classification (A1) and fine-grained multi-label tagging (A2). For A1, we fuse MARBERTv2 and CLIP ViT-L/14 embeddings, comparing concatenation, self-attention, and ensemble heads. For A2, we use a multi-task label-attention architecture with per-class query vectors and an auxiliary head for cascade calibration. We reach macro-F1 = 0.7597 on A1 and 0.3142 on A2 (averaged over 10 active classes). We report reproducibility across runs, per-class error patterns, and the effect of cascade calibration on the long-tailed A2 label space. 

---
# FoundPDE: A generative foundation model for data-driven symbolic discovery of partial differential equations 

**Authors**: Langchen Liu, Junlin Luo, Min Zhu, Zhuoran Yang, Lu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26963)  

**Abstract**: Discovering governing partial differential equations (PDEs) from observed data is a central inverse problem in computational science and engineering: the goal is not only to predict trajectories, but to identify interpretable symbolic laws that can be analyzed and reused. Existing PDE discovery methods often depend on hand-designed libraries, accurate numerical derivatives, or per-equation symbolic search, making them fragile under sparse observations, noisy data, broad equation families, and unseen symbolic compositions. Here, we introduce FoundPDE, a pretrained data-to-symbol foundation model for PDE discovery. FoundPDE maps discretized spatio-temporal trajectories to symbolic equations with a transformer, uses Fourier and first-order spatial-derivative augmentation to expose complementary trajectory information, refines numerical coefficients, and supports downstream extension with improved generalization through few-shot fine-tuning and hierarchical discovery. On a clean in-domain benchmark spanning 20 PDE families, our model reaches 97.50% type recognition. Under coefficient shifts beyond the training support, type recognition remains 90.53%; under multiplicative Gaussian noise on the model input, it is 94.43% at a 2% noise level but falls to 52.09% at 10%. For unseen composite equations, few-shot fine-tuning reaches over 90% exact recovery on four held-out families, and we further develop a hierarchical discovery algorithm that recovers two-term composites without fine-tuning FoundPDE. By amortizing symbolic discovery across equation families, FoundPDE shifts PDE discovery from per-dataset search toward a reusable pretrained model that extends to new physics with limited additional data, a step toward foundation models that uncover interpretable governing laws directly from scientific data. 

---
# Beyond Policy Discrepancy: Capability-Aware Multi-Teacher On-Policy Distillation 

**Authors**: Xuanfeng Yang, Le Gui, Jiaming Wang, Xuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26950)  

**Abstract**: Multi-teacher on-policy distillation (MOPD) integrates domain specialists into a shared student, but determining how strongly each teacher should supervise remains challenging. Existing discrepancy-based allocation can assign strong supervision even when the student already matches or exceeds the teacher's task performance, since behavioral mismatch does not necessarily imply a remaining capability deficit. We propose Capability-Aware Multi-Teacher On-Policy Distillation (CA-MOPD), which combines specialist merging with capability-guided allocation to account for what the student has already learned. CA-MOPD first merges reinforcement-learning specialists into a strong student initialization, then uses verifier-estimated teacher--student capability gaps to adjust domain-level distillation strength. Supervision decreases as capability gaps close and increases when estimated student capability regresses, while dense token-level guidance remains provided by the teachers. Across six benchmarks spanning mathematics, code generation, and instruction following, CA-MOPD achieves an average score of 32.22, outperforming naive MOPD, Open-MOPD, and the merged initialization by 0.86, 0.83, and 0.74 points, respectively. Controlled comparisons further support the benefits of merged initialization and capability-gap guidance under matched training budgets. 

---
# GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation 

**Authors**: Hardik Ruparel, Raj Kumar Pandey, Pramod Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.26938)  

**Abstract**: Retrieval-Augmented Generation (RAG) is increasingly served across geo-distributed cloud and edge infrastructures, where wide-area latency, cross-region egress cost, and data-residency rules compete with retrieval quality. Conventional pipelines rank by semantic similarity alone and ignore where content lives, inflating tail latency and cost. We formalize the Geo-RAG Selection Problem (GRaSP), a drop-in re-scoring layer that decays relevance by latency and adds an optional cost penalty under a cardinality budget and per-region caps, with no retraining. We first settle its complexity: the base formulation is a linear objective over a partition matroid, solved optimally by an O(N log N) greedy algorithm, rather than NP-hard as a knapsack framing suggests, with hardness entering only under an additive egress budget. On a five-node CloudLab testbed emulating a four-region WAN over the Natural Questions corpus (2.68M passages), GRaSP's benefit is placement-dependent: under uniform placement it cuts fetch latency at negligible quality cost; under topical (residency-like) placement it traces a recall-latency Pareto frontier that dominates relevance-only, nearest-k, and a latency-budgeted baseline. At a moderate latency setting, GRaSP lowers P90 fetch latency by 41% while retaining 78% of relevance-only's recall, with sub-millisecond selection overhead, stable to 200 queries/s. Findings generalize across four public BEIR benchmarks. In end-to-end question answering, GRaSP reduces context-fetch latency by 43% while remaining within 0.062 absolute EM of relevance-only retrieval. 

---
# How Intervention History Changes AI Agent Sabotage on Subsequent Tasks 

**Authors**: Daniel Saeedi, Amirali Aghazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2610.26935)  

**Abstract**: AI control evaluations often reset agents' memory between tasks, although deployed agents may retain records of prior interventions. We ask whether learning that an earlier attack was caught changes an agent's ability to evade defer-to-trusted (DtT), a protocol that replaces suspicious work with a trusted model's solution. Herein, we study cross-episode adaptation: whether intervention history changes attack behavior on a subsequent task with different objectives and a fresh environment. We evaluate this using ControlArena's Infrastructure-as-Code (IaC) and SHADE-Arena as testbeds, where agents must complete a main task while covertly pursuing a harmful side objective. For each successor task, we pair a history-conditioned attempt with the model's no-history attempt. The retained history either states that the previous submission was replaced or additionally includes the trusted model's replacement. For Claude Opus 4.8 on IaC, intervention history reduces successful sabotage. Completion of both the main and side objectives falls from 77% to 12% when the agent is told about the replacement and to 38% when it is also shown the trusted model's replacement, while main task success remains essentially unchanged. Condensing the history weakens this effect, suggesting that the retained context matters and limiting attribution to intervention feedback alone. GPT-5.4 shows no significant change in successful sabotage on the same IaC benchmark. On SHADE-Arena, GPT-5's successful sabotage rate decreases, but the reduction is not significant. Across the tested models, history conditions, and deferral thresholds, intervention history never increases the rate of successful sabotage that survives DtT. These results show that intervention history can change subsequent attack behavior, but its effect depends on both the model and the retained context. 

---
# Expert-Guided Rubrics as Dense Reward Signals for Reinforcement Learning in Non-Verifiable Domains 

**Authors**: Ashish Bansal, Hossam Amer, Keerat Kaur Guliani, Parinita Edke, Lesley Matheson, Leizer Teran  

**Link**: [PDF](https://arxiv.org/pdf/2610.26932)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in non-verifiable domains such as financial services, where they must interpret user intent, rewrite queries, and retrieve information under domain-specific constraints. Although reinforcement learning (RL) can train such agents from the consequences of their actions, recent successes largely rely on verifiable tasks with automatically computed binary rewards. In non-verifiable retrieval tasks, assessing query quality requires scarce subject-matter expert (SME) judgment that is inherently graded rather than binary. With Group Relative Policy Optimization (GRPO), sparse rewards yield no useful learning signal when all sampled outputs within a group receive the same outcome, causing optimization to stall. We introduce a method that converts a small, fixed investment of SME judgment into a dense reward and combines it with retrieval-based metrics to form a hybrid, GRPO-compatible training signal. Experiments show improvements of up to 6\% in Recall@3 and 6.7\% in MRR@3 compared with sparse-reward baselines. 

---
# Multilingual Brain Decoding Reveals Shared Cortical Semantic Geometry 

**Authors**: Yi Guo, Yong Hu, Michael Kwok-Po Ng, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26927)  

**Abstract**: Human languages differ markedly in lexical and syntactic form yet can convey similar semantic content. The neural basis of this cross-language semantic commonality remains unclear. Here we developed a multilingual brain decoding framework and analysed brain recordings from 227 participants spanning Chinese, English and French. We found that cortical semantic representations exhibited consistent semantic geometry relationships across languages and individuals, revealing a shared cortical semantic geometry. Importantly, the cross-language consistency of cortical semantic geometry increased from sensorimotor to higher-order association cortex, forming a shared semantic geometry gradient closely aligned with the principal cortical functional gradient. Our work establishes generalizable multilingual brain decoding and identifies stable cortical semantic geometry as a neural basis for semantic commonality across languages. 

---
# Enhancing analytical photon transport in turbid media through physics-guided neural network corrections 

**Authors**: Guojun Tang, Paul D. Teal, Alexander Doronin  

**Link**: [PDF](https://arxiv.org/pdf/2610.26917)  

**Abstract**: Analytical methods provide rapid and computationally efficient predictions of light transport in turbid media, making them attractive for real-time and inverse applications. However, their accuracy is limited by the approximations and assumptions used to obtain tractable solutions. On the other hand, Monte Carlo (MC) simulation is widely regarded as the "gold standard" for modeling light transport in turbid media, but its high computational cost limits its use in applications requiring repeated forward evaluations. Here, we introduce a physics-guided residual learning framework that combines the computational efficiency of analytical modeling with the accuracy of MC simulation. As a representative case study, we consider the diffusion approximation (DA), which can exhibit systematic errors under relatively strong absorption, weak scattering, and imperfect boundary treatment. Given the absorption coefficient, scattering coefficient, and anisotropy factor, the corresponding DA prediction is incorporated as a physical prior, while the network learns only the residual between the analytical prediction and the MC reference. Across the evaluated optical-property range, the proposed framework reduces scalar-reflectance RMSE and spatially resolved reflectance RMSLE by factors of approximately 62 and 14, respectively, relative to the diffusion approximation. With only 1\% of the available training data, it achieves approximately 87\% and 93\% lower corresponding errors than a direct neural-network surrogate. These results demonstrate that learning corrections to an analytical physical baseline improves both prediction accuracy and data efficiency. 

---
# Steerable Pluralistic Policy Alignment from Anonymized Crowd Feedback 

**Authors**: Akansha Kalra, Ellen R. Novoseller, Vernon J. Lawhern, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2610.26914)  

**Abstract**: Pluralistic alignment requires AI systems to learn from diverse populations whose preferences may reflect heterogeneous or even conflicting objectives, and to personalize their behavior to individual users. Existing approaches in pluralistic alignment often assume access to annotator identities and rely on powerful pretrained representations to model individual preference variations. We propose Preference-PCA Pluralistic Learning (P3L), a novel framework for pluralistic alignment that jointly learns trajectory embeddings and the shared basis of preference directions via PCA decomposition from heterogeneous crowd feedback without requiring annotator identities. At test time, our approach enables rapid and accurate personalized reward learning and downstream policy optimization. Experiments across multiple embodied control tasks demonstrate that P3L enables steerable personal alignment that generalizes beyond the offline dataset to out-of-distribution new users and attains competitive RL and test performance, while substantially reducing computational cost and supporting variable adaptation query budgets. 

---
# Uncertainty Gates the Gradient: Adaptive Updates for Generalized Few-Shot 3D Segmentation 

**Authors**: Yifei Zhao, Fanyu Zhao, Shengtang Wu, Han Zhang, Ruike Cao, Yinsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26909)  

**Abstract**: In generalized few-shot 3D point cloud segmentation, support annotations are often compromised by sparsity and occlusion. Blindly adapting to such low quality samples risks learning spurious geometric patterns rather than robust novel concepts. Although 3D Vision-Language Models have been introduced to sup plement supervision, the potential of uncertainty to actively govern adaptation dynamics remains underexplored. Existing signal-level methods like loss reweight ing fail to prevent unreliable episodes from steering the optimization trajectory. To address this, we propose Uncertainty-Gated Adaptive (UGA), to the best of our knowledge the first framework that elevates uncertainty to update-level gating for generalized few-shot 3D point cloud segmentation. The core mod ule, Uncertainty-Gated Update (UG-Update), estimates episodic reliability via intra-class dispersion over merged support and pseudo-labeled novel supervision, and selectively gates residual gradient updates. This encourages the model to learn novel semantics from reliable episodes while attenuating updates during unreliable ones. Complementing this, we introduce an Uncertainty-Based Pseudo Label Filter (UB-Filter) to enforce learning from high-confidence dense signals. On ScanNet200 and ScanNet++ (1-/5-shot), UGA sets new state of the art with an average +3.08 HM IoU over the strongest prior baseline. 

---
# MOSS: A Production Framework for Monitoring and Diagnosing Semantic Failures in Search and Recommendation Systems 

**Authors**: Alina Nesen, Prathyusha Kanmanth Reddy, David Relyea, Rohit Avinash Mujumdar, Samaneh Karimi, Bo Shen, Jordan Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26902)  

**Abstract**: Detecting semantic failures in production recommendation and search systems is hard: behavioral metrics lag by days, manual review does not scale, and most LLM evaluation frameworks are designed for offline, fixed datasets rather than continuous operational monitoring. We present Model Sanity Service (MOSS), a production framework that addresses this operational gap by embedding LLM-based relevance scoring into a recurring monitoring loop with targeted sampling, strategy-aware prompting, structured alerting, and diagnostic follow-up prompting that categorizes failures into actionable root causes.
MOSS has been running in production at a large U.S. retailer for over six months across recommendation, search, and offer surfaces, and has surfaced two confirmed serving defects. For one of these, a backfill blending defect, MOSS quantified the impact of the proposed fix offline as a 76.4\% relative reduction in alert rate. On a 498-pair human-annotated recommendation benchmark, frontier LLM judges achieve higher F1 and AUC point estimates than non-LLM baselines (internal production embeddings and SBERT), with the largest gains on strategy-dependent tasks requiring intent-level relational reasoning. The same judge, score scale, and threshold transfer to a 466-pair search benchmark spanning 21 queries with only a surface-specific prompt adaptation, achieving strong F1 and AUC without retuning. These results show that a single LLM monitoring framework can provide useful, actionable quality signals across multiple product discovery surfaces in a large-scale e-commerce setting. 

---
# PW-Pruning: Phase-Aware Expert Selection for Speech-Centric Omni-Modal Models 

**Authors**: Junjie Cao, Fengping Tian, Wenjie Zhou, Xuanfan Ni, Chenyang Lyu, Siyou Liu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26891)  

**Abstract**: Sparse routing reduces per-token computation but leaves the full expert bank resident in memory. We study expert pruning for speech-centric omni-modal models, where pooling calibration statistics can obscure the different expert demands of acoustic-input processing and text-answer generation. We propose PW-Pruning, a phase-aware extension of DO-ACP that constructs separate importance-weighted kernels for acoustic-prefill and teacher-forced answer tokens, then combines phase-weighted log-determinant marginal gains to select one static expert bank. On Qwen3-Omni, retaining 32 of 128 experts per layer improves MMAU-mini and MMSU accuracy over matched pooled selection by 3.20 and 4.38 percentage points, respectively, after supervised fine-tuning, averaged over three runs. Single-phase and shuffled-label controls on this public backbone suggest that semantic phase information contributes to these gains. Experiments on a non-public Qwen3.5-Omni backbone and English-only zero-shot evaluation of Nemotron provide complementary evidence. Across the tested Qwen budgets, AU gains are larger under stronger pruning, without inference-time mask switching. 

---
# Disentangling Depression from Cognitive Decline in Elderly Speech Using Concurrent Clinical Assessments 

**Authors**: Woori Jeon, Seunghee Ha, Sang-Kyu Lee, Ji Hye Yoon, Tae-Jin Yoon, Seung Jin Lee, Woojae Han, Jungmin So  

**Link**: [PDF](https://arxiv.org/pdf/2610.26890)  

**Abstract**: Detecting depression from speech in elderly patients with mild cognitive impairment (MCI) is complicated by overlapping acoustic effects of cognitive decline. Without disentangling these, classifiers risk learning cognitive rather than depression-specific patterns. We present a Korean elderly speech corpus of 89 MCI speakers collected over three years with concurrent depression (SGDS) and cognitive (MMSE) assessments. Both linear mixed-effects models and partial correlations identify the same pattern: of seven feature groups, only formant features are associated with depression after controlling for cognitive function, while widely used F0 features carry no signal. The F1+F2 subset achieves UAR 0.760 with just 12 features, outperforming both the full eGeMAPS baseline and self-supervised representations. These findings show that concurrent clinical assessments can identify interpretable, depression-specific acoustic markers in populations where cognitive and affective symptoms co-occur. 

---
# Measuring Consensus in AI Agent Networks 

**Authors**: Jin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24833)  

**Abstract**: As AI agents become increasingly connected through networks and take on tasks that require negotiation and joint decision-making, a systematic approach is needed to assess how their views converge and what information supports their decisions. We present an evaluation framework that treats consensus formation as the evolution of publicly stated positions toward a shared region, using accumulated communication history as context. The framework tracks position trajectories, group dispersion, and information expressed in public messages. Two case studies examine staffing decisions based on distributed facts and decisions requiring agents to reconcile duplicate information and a cancelled contract. Results from 1,872 completed runs associate sharing explanations with more accurate group decisions, while the second task reveals persistent disagreement despite accurate aggregate answers. Membership-change experiments provide a complementary evaluation. By separating convergence among agents from progress toward a task's full-information answer, the framework relates decision accuracy and bounded disagreement to the information available during the exchange. 

---
# Aptamers as a sandbox for lab-in-the-loop agentic molecular discovery 

**Authors**: Zachary Cutts, John Coller, Rachel Chen, Hua Tang, Mathias Bellaiche, Ryder Easterlin, Dina Zverinski, Petar Sirkovic, Avinatan Hassidim, Yossi Matias, Pooja Rao, Annalisa Pawlosky  

**Link**: [PDF](https://arxiv.org/pdf/2610.24806)  

**Abstract**: Despite breakthroughs in AI-driven molecular generation, existing discovery pipelines operate largely in silico, with wet-lab feedback reserved for validation rather than iterative improvement. Whether large language model (LLM)-based reasoning agents can accelerate de novo discovery by iterating with lab data remains unproven.
Here we use protein-binding aptamer design as a sandbox (1-2 week cycles, \$1-\$3 per candidate) to demonstrate a wet lab in a closed loop with AI systems. We tasked agentic pipelines with developing aptamer binders against $\alpha$-thrombin and myoglobin across three design-test-improve cycles ($\sim$1,000 candidates per round; $\sim$4,400 constructs). Statistically significant, bead-corrected on-target hit rates among new candidates rose across rounds from 0.0% in Round 1 to 26.0% in Round 2 and 42.1% in Round 3 on $\alpha$-thrombin (37.1% excluding literature motif descendants), and from 10.2% in Round 1 to 3.7% in Round 2 and 33.5% in Round 3 on myoglobin. Single-clone bead-binding titrations of eight full-length $\alpha$-thrombin candidates confirmed concordance with pooled sequencing enrichment (Pearson $r = 0.90$). In these assays, literature-derived motifs engineered by the agents to reduce off-target reactivity in the pooled screen reached apparent affinities of $K_D = 24.6$--$133$ nM, whereas de novo scaffolds showed low-micromolar binding ($K_D > 1\ \mu$M).
We also report how our pipelines used myoglobin binding data to exploit the G-quadruplex binder space, diagnose cross-reactivity as on-target specificity dropped from 75.0% to 19.1%, and recover by developing strategies that restored specificity to 72.4%. We demonstrate that empirical lab data is more valuable than existing computational proxies, showing how iterative experimental feedback enables AI agents to diagnose assay artifacts and refine molecular designs across rounds. 

---
# Fine-Tuning Small Language Models for Reliable VASP INCAR Generation 

**Authors**: Xinyue Zhang, Jixiang Li, Bin Shao, Baishun Yang, Zhiyang Liu, Weichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05387)  

**Abstract**: Language models can prepare VASP INCAR files from natural-language requests, but so far only large proprietary cloud models come close to handling the tightly coupled, physics-sensitive settings reliably, a dependence that fits poorly with local, high-throughput materials workflows where privacy, cost, and offline deployment matter. We show that a small language model (SLM) can close this gap. The SLM is fine-tuned on reference VASP calculations and paired with VASPGuard, a deterministic post-processor that checks syntax, workflow, and material-dependent constraints; we call the combined model INCAR-SLM. On INCARBench, a benchmark for VASP INCAR generation, INCAR-SLM built on Qwen3-4B outperforms every general-purpose LLM evaluated, exceeding GPT-5.4 by 15.55 points on the 100-point INCAR Score. Most of this gain comes from fine-tuning, with VASPGuard correcting the errors that remain. We further find that model size matters less than expected: once fine-tuning and post-processing are applied, performance saturates at a few billion parameters, and Qwen3-4B outperforms larger models in the same family. 

---
# INCARBench: A Benchmark for Scientific Configuration in VASP INCAR by Large Language Models 

**Authors**: Bin Shao, Jixiang Li, Xinyue Zhang, Baishun Yang, Zhiyang Liu, Weichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.23571)  

**Abstract**: Large language models (LLMs) are increasingly being integrated into first-principles computational workflows, yet their ability to configure scientific calculations remains poorly understood. Here, we introduce INCARBench, a benchmark for evaluating LLMs on input configuration for the Vienna Ab initio Simulation Package (VASP) through both configuration generation and repair tasks. Evaluating 19 model configurations reveals substantial capability differences among current frontier models. While several models achieve high semantic and policy accuracy, task-critical correctness remains substantially lower, demonstrating that parameter-level correctness does not necessarily imply scientifically valid configurations. Failure analysis shows that errors concentrate in physically coupled settings involving DFT+$U$, magnetism, and correlated materials, where multiple constraints must be satisfied simultaneously. Repair evaluation further reveals that correcting incorrect settings and preserving already-valid configurations are distinct capabilities, with configuration preservation remaining a major challenge. These findings establish scientific configuration as a measurable capability of large language models and provide a foundation for developing more reliable AI systems for computational materials science. 

---
# Large Language Models for Agent-Based Modelling: Current and possible uses across the modelling cycle 

**Authors**: Loïs Vanhée, Melania Borit, Peer-Olaf Siebers, Roger Cremades, Christopher Frantz, Önder Gürcan, František Kalvas, Denisa Reshef Kera, Vivek Nallur, Kavin Narasimhan, Martin Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2507.05723)  

**Abstract**: The emergence of Large Language Models (LLMs) with increasingly sophisticated natural language understanding and generative capabilities has sparked interest in the Agent-based Modelling (ABM) community. With their ability to summarize, generate, analyze, categorize, transcribe and translate text, answer questions, propose explanations, sustain dialogue, extract information from unstructured text, and perform logical reasoning and problem-solving tasks, LLMs have a good potential to contribute to the modelling process. After reviewing the current use of LLMs in ABM, this study reflects on the opportunities and challenges of the potential use of LLMs in ABM. It does so by following the modelling cycle, from problem formulation to documentation and communication of model results, and holding a critical stance. 

---
