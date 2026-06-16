# Bayesian Inference and Decision Audits for Public Archives of Frontier AI Evaluations 

**Authors**: Yanan Long  

**Link**: [PDF](https://arxiv.org/pdf/2606.17005)  

**Abstract**: Public AI evaluations are often read as terminal leaderboards, yet the underlying evidence is a selective time series shaped by reporting rules, benchmark revisions, and missingness. Repeated public archives for LiveBench and Open LLM Leaderboard v2 serve as the primary longitudinal record; LMArena provides a preference stress test; and GAIA and tau-bench contribute limited agentic pilots. Together, these archives instantiate a Bayesian inference problem: under a fixed reporting convention, one constructed terminal-only example over $1{,}000$ systems is compatible with two pre-terminal histories, yielding times of $23.03$ or $75.13$ to reach within $0.05$ of the ceiling under the same terminal-tail model. In synthetic posterior comparisons, action-facing diagnostics differ across observation regimes. The candidate selection-aware frontier model fails synthetic recovery, objective-archive prediction, preference transfer, and uncertainty calibration; correspondingly, fixed audit gates reject its stronger claims. An archive-and-adjudication protocol reconstructs public evaluation histories, isolates a verified timing boundary, and falsifies unsupported frontier claims. 

---
# When in Doubt, Plan It Out: Committed Small Language Model Deliberation for Reactive Reinforcement Learning 

**Authors**: Nathan Gavenski, Juarez Monteiro, Francisco Galuppo, Adriano Veloso, Odinaldo Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2606.16995)  

**Abstract**: Reinforcement Learning (RL) policies often degrade in unfamiliar environments because they lack explicit deliberation. We propose Plan, Align, Commit, Think (PACT), a hybrid architecture that combines a fast, reactive RL policy with a slow, deliberative Small Language Model (SLM) planner. PACT invokes the SLM asynchronously to generate and validate candidate action plans. Once a plan is verified through simulation as safe, feasible, and complete, it is executed directly, bypassing the RL policy without retraining or modifying it. Evaluated on three FrozenLake configurations of increasing difficulty, PACT outperforms all baselines while relying on a 2B-parameter SLM backbone, suggesting that deliberative planning and reactive execution are more powerful in concert than either is alone in these settings. 

---
# Consensus-based Agentic Large Language Model Framework for Harmonized Tariff Schedule Code Classification 

**Authors**: Truong Thanh Hung Nguyen, Khanh Van Quynh Nguyen, Hoang-Loc Cao, Tri Duong, Phuc Ho, Van Pham, Loc Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.16987)  

**Abstract**: Accurate Harmonized Tariff Schedule (HTS) code classification is essential for customs clearance, duty assessment, trade statistics, and regulatory compliance in maritime logistics. However, exact HTS classification remains challenging because product descriptions are often short, incomplete, or ambiguous, while correct classification depends on hierarchical tariff structures, legal notes, and jurisdiction-specific rules. This paper proposes an agentic large language model (LLM) framework for Canadian 10-digit HTS code classification in smart-port and maritime logistics environments. The framework integrates multi-agent information retrieval, semantic retrieval over official tariff documents, evidence-grounded reasoning, consensus-based validation, element-wise voting across hierarchical code components, confidence estimation, and human-in-the-loop escalation. We evaluate the framework on a private dataset of 3,300 domain-expert-labeled product records collected from logistics and delivery contexts. Experimental results show that exact 10-digit classification remains difficult even for advanced LLMs, with performance decreasing from coarse chapter-level prediction to fine-grained tariff and statistical suffix assignment. These findings demonstrate the need for evidence-grounded, uncertainty-aware, and human-centered classification workflows rather than fully autonomous single-step prediction. The proposed framework supports more interpretable, accountable, and compliance-oriented HTS classification for maritime logistics and smart-port operations. Our code is available at this https URL. 

---
# The embrace of open science: An analysis of a decade of AI research and 56 800 conference papers 

**Authors**: Kevin L Coakley, Thijs Snelleman, Holger Hoos, Odd Erik Gundersen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16974)  

**Abstract**: The reproducibility crisis has directed the AI research community toward improving documentation practices. Several studies have identified methodological issues, and in response, the most impactful venues in the field have introduced reproducibility checklists. We seek to understand whether documentation practices have changed over time by assessing all published papers at five leading AI conferences over the past decade. Seven reproducibility variables were identified, quality-assured and used to analyse 56 800 publications. Our analysis reveals that in the period 2014 to 2024, documentation practices have improved; papers sharing both code and data increased nearly sixfold, from 11% to 64% Building on empirical reproducibility rates from a prior study, we estimate - inferred from documentation practices, not direct testing - that reproducibility increased from 28% in 2014 to 64% in 2024. Improvements in documentation practices predate the introduction of reproducibility checklists, suggesting these changes reflect a broader movement toward open science rather than a direct response to formal requirements. 

---
# A Causal Model of Theory of Mind in Conflict for Artificial Intelligence 

**Authors**: Nikolos Gurney  

**Link**: [PDF](https://arxiv.org/pdf/2606.16944)  

**Abstract**: Theory of mind (ToM), the capacity to ascribe mental states to others and use those ascriptions for prediction and inference, is widely assumed to be essential for effective human-machine integration. Existing AI-ToM models address \emph{how} to mentalize, but leave the question of when largely unaddressed. The central question is: under what situational and agent-level conditions is ToM engagement causally warranted in conflict? This paper presents a structural causal model formalized as a directed acyclic graph (DAG), treating ToM as a mechanism activated by situational and agent-level conditions rather than as an always-on capacity. The model specifies four exogenous variables capturing situational and agent-level conditions, five endogenous mediators, and a mechanistic ToM node producing engagement states through three distinct causal pathways: a tractability pathway, a reasoning-depth pathway, and an enabling-cause pathway. The primary outcome is epistemic accuracy, which decouples social reasoning from behavioral policy and generalizes across social phenomena beyond conflict. The framework gives AI systems a principled, resource-rational decision procedure for mentalizing, with implications for efficiency, trust, and the development of robust artificial social intelligence. Simulation validation, empirical human-machine teaming studies, and ethical considerations arising from conflict-optimized mentalizing are discussed. 

---
# RAID: Semantic Graph Diffusion for True Cold-Start and Cross-Lingual Forecasting 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16925)  

**Abstract**: Time-series foundation models show strong transfer performance when given a non-empty history window. However, true cold-start scenarios, where a new item has no prior observations, violate this assumption. We propose RAID (Retrieval-Augmented Iterative Diffusion) a framework, which replaces history-based correlation learning with metadata-driven semantic retrieval and graph-conditioned diffusion. RAID maps textual metadata into a shared semantic space using a frozen multilingual embedding model and constructs an inductive retrieval graph that extends naturally to unseen items. It first forms a base forecast by aggregating information from semantically related neighbors, then refines this forecast with a gated diffusion module to model residual uncertainty. Under a strict true cold-start protocol, RAID outperforms strong foundation models and competitive baselines on both forecasting accuracy and prediction interval coverage, while reducing inference latency by an order of magnitude through non-autoregressive decoding. The shared semantic space also enables zero-shot cross-lingual transfer, allowing a model trained on English descriptions to generalize to items described in other languages without direct supervision. 

---
# MA-SBI: Misspecification-Aware Simulation-Based Inference via Side-Channel Guidance 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16923)  

**Abstract**: Simulation-based inference (SBI) of latent parameters is often hindered by simulator misspecification, the mismatch between simulated and real-world observations caused by inherent modeling simplifications. RoPE, the recent state-of-the-art for robust SBI, addresses this through optimal transport between learned representations of real and simulated observations, but requires ground-truth parameter calibration pairs that are typically unavailable in the very settings where SBI is needed. What practitioners do have is unstructured side-information such as regime labels, instruction text, and policy bulletins. We propose Misspecification-Aware Simulation-Based Inference (MA-SBI), a calibration-free framework that turns this side-channel into a posterior correction. A learned corrector maps side-channel text to an observation-space shift applied before any pre-trained amortized posterior, requiring no retraining and no parameter ground-truth. Our main theorem bounds achievable bias reduction by the mutual information between misspecification and side-channel, with a non-vacuous constant that extends to all sub-Gaussian noise via Donsker-Varadhan. On hide-the-calibration benchmarks, MA-SBI with text alone matches the oracle posterior across 10 seeds and two backbones (TOST equivalence), while RoPE given more data does not. The two approaches are complementary: where misspecification is structural and recoverable from parameter pairs, RoPE dominates, as the theory predicts. A stochastic variant improves posterior-predictive log-likelihood on real COVID and OxCGRT epidemiological data, and correctly leaves the posterior unchanged on a well-specified cognitive-science corpus. 

---
# Greed Is Learned: Visible Incentives as Reward-Hacking Triggers 

**Authors**: Tong Che, Rui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16914)  

**Abstract**: Deployed agents increasingly act with their reward proxy in view, such as a balance, score, or KPI dashboard. We show that reinforcement learning can make a policy \emph{addicted} to such a visible self-benefit channel. It chases the displayed payoff across held-out domains, sacrifices the true task to do so, and follows the channel wherever we rewrite it, while policies that never saw the channel stay honest. We call this \emph{reward-channel addiction} and study it in \emph{MoneyWorld}, a synthetic sandbox. The addiction can \emph{flip a model's safety alignment}: trained only on innocuous money tasks with no safety content, the model abandons the safe action it otherwise always takes whenever a dashboard pays for an unsafe one, and reverts to safe once the channel is hidden. This learned bribe replicates across model scales and families. Blindly optimizing super-capable, next-generation AI on KPIs or P\&L can be dangerous for alignment. \emph{Greed is learned} when following such a channel pays. 

---
# Symbolic Informalization: Fluent, Productive, Multilingual 

**Authors**: Aarne Ranta  

**Link**: [PDF](https://arxiv.org/pdf/2606.16893)  

**Abstract**: Symbolic informalization enables a reliable conversion of formal mathematics to natural language. It has the potential to make machine-checked content human-readable without loss of precision. In a traditional proof system usage, symbolic informalization generalizes the limited mechanisms of syntactic sugar into the ordinary language of mathematics. In a setting where proofs are constructed by artificial intelligence and autoformalization, symbolic informalization can explain what precisely has been constructed. This paper outlines the project Informath, which aims to show how symbolic informalization can produce fluent text with a reasonable development effort and address multiple formal and natural languages. Informath is based on an interlingual architecture, where Dedukti works as a hub between different proof systems (Agda, Lean, Rocq) and Grammatical Framework (GF) takes care of linguistic correctness and variation in different natural languages. 

---
# GIST-CMTF: Goal-State Inference for Causal Minimal Tool Filtering in LLM Agents 

**Authors**: Rahul Suresh Babu, Rohit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2606.16813)  

**Abstract**: Tool-augmented LLM agents rely on runtime filtering to decide which tools should be visible at each step. Causal Minimal Tool Filtering (CMTF) reduces tool-choice confusion by exposing only the next causally necessary tool frontier, but it assumes that the user request has already been mapped to a symbolic goal state. In practice, requests such as "handle my appointment" or "take care of this email" may correspond to multiple possible goals. This creates wrong-goal execution, where an agent follows a valid causal tool path for an unintended objective. We introduce GIST-CMTF, a goal-state inference layer that predicts candidate symbolic goals over the same state-transition vocabulary used by CMTF, estimates ambiguity, and either applies CMTF or exposes clarification as a causal action that produces missing goal or state variables. We evaluate GIST-CMTF across seven model backends, six filtering methods, and 120 controlled tool-use tasks. GIST-CMTF achieves 97.0% task success, compared with 80.1% for top-goal CMTF and 82.9% for semantic-goal CMTF. It reduces wrong-goal execution from 19.4% under top-goal CMTF to 2.5%, while preserving the one-tool exposure of causal filtering and using substantially fewer tokens than all-tools exposure. These results suggest that reliable tool-augmented agents should validate goal state, not only tool relevance, before exposing external actions. 

---
# Scaling LLM Reasoning from Minimal Labels: A Semi-Supervised Framework with a Lightweight Verifier 

**Authors**: Keizo Kato, Chenhui Chu, Yugo Murawaki, Sado Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16811)  

**Abstract**: For the development of Large language models (LLMs), recent approaches to generating pseudo intermediate reasoning have shown remarkable progress. But they typically rely on large numbers of correctly annotated answers to assess reasoning quality. This paper presents a semi-supervised framework that scales reasoning learning from minimal supervision, turning reasoning verification itself into a data creation mechanism. We train a lightweight reasoning-correctness classifier on only a few labeled samples, which judges whether intermediate reasoning traces generated by an LLM are valid. Furthermore, an entropy-based confidence threshold filters out unreliable samples, and the remaining high-confidence reasoning traces are used to fine-tune the model. Experiments on Verifiable Math Problems (Orca-Math subset) and Question Answering on Image Scene Graphs (GQA) with Visual Programming show that our method achieves accuracy comparable to using 10-15x more labeled data. Ablation analyses confirm that both the classifier and entropy filtering are essential for scalable and noise-resistant pseudo-labeling. By replacing expensive answer-level supervision with lightweight reasoning verification, our method provides a practical path toward constructing large-scale reasoning resources and paves the way for future autonomous reasoning systems that learn from minimal human input. 

---
# Adaptive and Explicit safe: Triggering Latent Safety Awareness in Large Reasoning Models 

**Authors**: Ke Miao, Jiaxin Li, Hongliang Chen, Yuke Hu, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.16808)  

**Abstract**: While Large Reasoning Models (LRMs) excel at complex tasks, they remain highly vulnerable to sophisticated jailbreaks and direct harmful queries. To address this vulnerability, prior works depend heavily on external manual data annotation for safety alignment. However, we observe that LRMs can inherently identify safety risks when being re-presented with original queries alongside their own reasoning trajectories -- a capability we term Latent Safety Awareness. To leverage this safety awareness, we first employ Supervised Fine-Tuning (SFT) to explicitly induce safe tags to trigger safety analysis and guidance following the initial reasoning content for unsafe queries, while preserving standard responses for general queries to ensure adaptive triggering. Subsequently, we apply Direct Preference Optimization (DPO) to further enhance the correctness and stability of the safety analysis and guidance. Notably, responses required for both training stages are entirely generated by models being optimized. With (Safe Trigger) SFT and DPO, experimental results demonstrate significant safety enhancement. For example, the Attack Success Rate (ASR) of DeepSeek-R1-Distill-Llama-8B, on average, drops 24.65% and 36.72% on harmful and jailbreak benchmarks, respectively. Finally, our Safe Trigger method exerts almost no negative impact on general performance or user experience. 

---
# LabOSBench: Benchmarking Computer Use Agents for Scientific Instrument Control 

**Authors**: Anqi Zou, Han Deng, Chengyu Zhang, Junquan Hu, Yu Wang, Yuxiang Xing, Aokai Zhang, Hanling Zhang, Zhaoyang Liu, Ben Fei, Zhihui Wang, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16802)  

**Abstract**: Current computer-use benchmarks primarily focus on software operation tasks in virtualized systems, whereas scientific instrumentation scenarios require coordinated control over complex interfaces, and feedback-driven parameter adjustment. However, directly evaluating agents on physical high-precision instruments is impractical due to high cost, safety risks, limited accessibility, and difficulty in ensuring reproducible evaluation. This motivates the need for a simulated yet realistic testbed that preserves the operational challenges of scientific instruments while enabling scalable and safe benchmarking. To this end, we introduce LabOSBench, a challenging benchmark for multimodal GUI agents built on a suite of web-based scientific-instrument simulators. Operating directly via a browser, LabOSBench avoids resource-heavy OS virtualization while supporting flexible task configuration and execution-based evaluation. Specifically, LabOSBench constructs 96 subtasks across eight instrument simulators, covering workflows from sample loading, alignment, parameter tuning, and data acquisition to result inspection. We evaluate general-purpose vision-language models, specialized GUI agent models, and advanced agentic frameworks at both subtask and end-to-end levels. Our experiments reveal that while existing agents can complete many structured GUI subtasks, they still struggle with feedback-driven operations and long-horizon workflow execution. Overall, LabOSBench provides a reproducible, low-cost testbed for advancing computer-using agents toward scientific-instrument control. 

---
# OpenClaw-Skill: Collective Skill Tree Search for Agentic Large Language Models 

**Authors**: Tianyi Lin, Chuanyu Sun, Jingyi Zhang, Changxu Wei, Huanjin Yao, Shunyu Liu, Xikun Zhang, Liu Liu, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16774)  

**Abstract**: Equipping Large Language Model (LLM) agents with effective skills is crucial for solving complex tasks in real-world systems like OpenClaw. In this work, we aim to develop a framework that automatically constructs such reusable skills to enhance LLMs in tool use, multi-step reasoning, and dynamic environment interaction. To this end, we propose Collective Skill Tree Search (CSTS), a novel tree-search-based skill construction framework that constructs structured, diverse and generalizable tree of skills. The core idea of CSTS is to leverage collective intelligence to jointly search, identify and compose effective skills via two iterative phases: Collective Skill Node Generation (CSN-Gen) and Collective Skill Node Assessment (CSN-Assess). CSN-Gen exploits collective knowledge from multiple models to explore diverse candidate skills for each subtask, enabling comprehensive skill exploration. CSN-Assess employs multiple models as judges to evaluate and select skill nodes with two scoring mechanisms: (1) collective quality scoring that aggregates independent evaluations to produce a robust estimate of skill effectiveness, and (2) collective transferability scoring that explicitly verifies whether a skill generalizes well across different models. With CSTS, we construct a set of comprehensive tree of skills along with skill-augmented training data, enabling models to effectively learn and utilize skills. Besides, we introduce Collective Skill Reinforcement Learning, which actively selects multiple relevant skills from the tree to broaden solution-space exploration, avoid being trapped by a single skill and its resulting homogeneous or suboptimal solutions. As a result, our trained model, OpenClaw-Skill, exhibits outstanding agentic capabilities in long-horizon planning, tool use and generalization over challenging benchmarks. 

---
# Skill-to-LoRA: From Using Skills to Learning Behaviors for Token-Efficient LLM Agents 

**Authors**: Tianyi Zhang, Zhonghao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16769)  

**Abstract**: Agent skills are commonly distributed as this http URL files: human-readable procedural documents that describe workflows, tools, resources, and domain conventions. While convenient for inspection and reuse, this design requires the same reusable procedure to be repeatedly injected into the runtime context. We propose Skill-to-LoRA(S2L), a behavior-centric skill representation that replaces runtime skill text with skill-specific LoRA adapters. Rather than compressing the skill document itself, S2L models the behavioral change induced by the skill text: offline, the complete this http URL is used to synthesize skill-guided demonstrations; online, the full document is omitted and the corresponding LoRA adapter is dynamically loaded to activate the learned skill behavior. We evaluate S2L with Qwen3.6-27B on a 21-skill subset of SWE-Skills-Bench. Compared with the no-skill and Full Skill Text baselines, S2L improves pass rate by 2.9 and 5.2 percentage points, respectively, while reducing per-step token cost by 6.6% relative to Full Skill Text prompting. S2L matches or improves Full Skill Text on 18/21 skills and the no-skill baseline on 15/21 skills. Control experiments further show that the gains depend on skill-specific adapter alignment: Wrong-LoRA and Shared-LoRA both reduce performance. These results suggest that many procedural agent skills can be converted from runtime instructions into trainable, dynamically loadable behavioral modules. Code will be released upon acceptance. 

---
# A First-Principles Derivation of LLM Policy Optimization: From Expected Reward to GRPO and Its Structural Extensions 

**Authors**: Jianghan Shen, Siqi Luo, Yue Li, Jiyao Liu, Wanying Qu, Yi Zhang, Ziyan Huang, Tianbin Li, Ming Hu, Xiaohong Liu, Yirong Chen, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2606.16733)  

**Abstract**: Policy gradient algorithms for language models optimize the same objective $J(\theta) = \mathbb{E}*{\tau \sim p*\theta(\tau)}[R(\tau)]$, which has exactly two factors: the trajectory probability $p_\theta(\tau)$ and the reward $R(\tau)$. Every method from REINFORCE to PPO to GRPO and their descendants modifies one or both factors to address a specific failure in the preceding formulation. Existing surveys organize these methods by domain or chronology, which obscures the rationale behind each design choice and the precise location of its intervention within the gradient estimator. This survey revisits the landscape of LLM policy optimization from $J(\theta)$ on first principles and uses the trajectory side, induced by $p_\theta(\tau)$, and the reward side, induced by $R(\tau)$, as the two axes along which methods are located. It covers the path from REINFORCE and PPO to GRPO, as well as post-GRPO variants, Agentic RL, and GRPO-OPD. The resulting framework is unified, diagnostic, and extensible: it analyzes methods from a shared objective, identifies which side each method modifies and why, and applies the same trajectory and reward axes across these settings. Across these settings, the framework also exposes compound failures that no single-side fix resolves and that therefore require joint design of the trajectory side and the reward side. The boundary cases and coupled failures identified by this map mark where existing solutions run out and provide a principled starting point for designing the next generation of LLM policy optimization algorithms. 

---
# AgentFairBench: Do LLM Agents Discriminate When They Act? 

**Authors**: Triveni Morla, Rohith Reddy Bellibaltu, Manpreet Singh, Manmeet Singh Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2606.16723)  

**Abstract**: Large language model (LLM) agents increasingly take actions (screening applicants, recommending credit, triaging patients), yet fairness for LLMs is still measured by grading answers. We introduce AgentFairBench, a cheap, reproducible, multi-domain benchmark for demographic disparity in the actions of LLM agents. Grounded in a companion framework, the Bias Conduction Framework (BCF, restated here), it spans three regulator-anchored domains: hiring, lending, and medical triage. Synthetic, demographic-neutral profiles are evaluated in counterfactual matched sets that vary only a name-coded race x gender signal (in the Bertrand Mullainathan tradition), under four agent scaffolds of increasing agency (direct, chain-of-thought, multi-agent deliberation, tool-augmented). A NumPy-only harness computes counterfactual flip rate, mean absolute score difference (MASD), action-rate disparity, and tool-invocation disparity, with bootstrap confidence intervals, paired tests, and false-discovery-rate control, for single-digit dollars per model. A live leaderboard with a held-out private split and a contamination canary admits external models by submission. Our pilot (864 decisions plus a test-retest replication) carries a methodological lesson: comparing a six-group score spread against a two-run noise difference overstates disparity by ~ 2.4X through statistic arity alone. Against an arity matched noise floor and an omnibus group test, claude haiku 4 5 shows no demographic effect above sampling noise (0 of 120 pairwise and 0 of 9 omnibus contrasts survive correction); a planted-bias test confirms the instrument detects disparity when present. The contribution is a sound, sensitive, adoption-ready instrument, the arity matched null methodology, and open artifacts to scale it. Code, data, and harness are released under open licenses, with an anonymized review artifact. 

---
# Medical world models: representing medical states, modelling clinical dynamics and guiding intervention policies 

**Authors**: Ke Liu, Mengxuan Li, Yanyi Bao, Tianyun Zhang, Chong Chu, Jiajun Bu, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16721)  

**Abstract**: Medical diagnosis and treatment are dynamic processes in which patient states evolve over time and clinical interventions alter future outcomes. Although current medical AI can detect disease, estimate risk and generate reports, many systems still return static labels or scores, offering limited insight into how illness may progress or how alternative interventions may reshape its trajectory. Medical world models adapt the world-model idea from artificial intelligence to healthcare by learning internal simulators of patient-state dynamics. Their long-term goal is to help clinicians anticipate deterioration, compare treatment-conditioned futures and tailor care to individual patients. Yet relevant work remains scattered across foundation models, longitudinal modelling, disease simulation, treatment-effect estimation, reinforcement learning and digital twins. To bridge this gap, this review outlines a roadmap for advancing medical AI from isolated diagnosis and prediction toward medical world models that simulate disease evolution and support intervention decisions. This roadmap is organized around three coupled capabilities: patient-state construction, clinical dynamics modelling and intervention decision support. Across representative systems, the comparison highlights what each capability contributes and how partial components can be integrated into more mature perception--dynamics--planning systems. Finally, we identify the challenges involved in turning plausible rollouts into clinically useful simulators. Related literature is available at this https URL. 

---
# User as Code: Executable Memory for Personalized Agents 

**Authors**: Bojie Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16707)  

**Abstract**: A personalized AI agent needs a user memory: a persistent model of who the user is, built across many conversations and consulted on each new one. Today this memory is almost always stored as unstructured text, a knowledge graph, or a flat store of facts, and consulted by retrieval -- fetching the entries most similar to the current request. Such "bag-of-facts" memory recalls individual facts well, but because storing a fact and acting on it are separate steps, it struggles to resolve contradictions, aggregate over many records, or enforce rules. We argue that user memory should instead be executable. We introduce User as Code (UaC), a paradigm in which an agent's model of a user is a living software project: typed Python objects hold the user's state and ordinary Python functions encode the rules that govern it, so representing and reasoning about the user happen in one medium an interpreter can run. The enabling mechanism is a two-phase pipeline: an append-only log that never discards a fact, periodically checkpointed into typed code.
This changes what memory can do. On standard long-term conversation benchmarks, UaC matches both a full-context upper bound and the strongest prior memory systems on recall (78.8% on LOCOMO). Its advantage emerges where representation matters most. On aggregate questions over a user's history -- "how many international trips did I take last year?" -- retrieval-based memory collapses (6-43%) while UaC stays near-perfect (99%), because the answer is a one-line computation over typed state rather than a search over text. And because its rules execute deterministically whenever the state changes, UaC can surface unsolicited, safety-critical alerts -- such as a newly prescribed drug that conflicts with an allergy recorded months earlier -- a capability query-driven memory cannot provide. 

---
# From Affect Prediction to Affect Forecasting: Evidence for Distinct Information Sources in Longitudinal Text 

**Authors**: Sadia Noor, Seemab Latif, Raja Khurram Shahzad, Mehwish Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2606.16687)  

**Abstract**: Modeling dimensional affect in longitudinal text requires distinguishing current affect estimation from future affective change forecasting. Existing approaches often treat each text as an independent observation and apply similar assumptions to both tasks, without testing whether they rely on different information sources. This paper investigates that distinction using longitudinal self-reported ecological essays and feeling-word entries. We propose the Trait--State Affective Prediction (TSAP) framework and its temporal extension E-TSAP for per-text valence and arousal prediction, evaluated on a held-out prediction test set of 1,737 entries from 91 users. We further propose the Affective Change Forecaster Hybrid (ACF-Hybrid) for next-step affective change forecasting, evaluated on a held-out forecasting test set of 46 users. For prediction, E-TSAP achieves composite Pearson correlations of 0.670 for valence and 0.449 for arousal. For forecasting, textual representations perform worse than compact numeric trajectory baselines: the text-inclusive model achieves only r=0.316 for valence and r=0.284 for arousal, whereas a simple prior-state baseline reaches r=0.615 and r=0.670, respectively. ACF-Hybrid, using dimension-specific numeric trajectory features, achieves r=0.659 for valence and $r=0.658$ for arousal. These results show that textual semantics support current affect prediction, whereas future affective change is better captured through prior numeric trajectory dynamics. 

---
# The Integrator Advantage: Controlled Agentic AI for Small and Medium-Sized Companies 

**Authors**: Christopner Koch, Joshua A. Wellbrock  

**Link**: [PDF](https://arxiv.org/pdf/2606.16649)  

**Abstract**: Agentic AI marks a new phase of enterprise automation. Unlike traditional automation or conversational AI, agentic systems can interpret goals, plan multi step tasks, access tools, interact with enterprise systems, and execute workflows with varying degrees of autonomy. For small and medium sized companies, this creates potential to reduce administrative burden, accelerate routine processes, and improve the use of organizational knowledge. This paper argues that the near term value of Agentic AI does not lie in full autonomy or workforce reduction, but in controlled partial autonomy for simple and medium complexity business processes. It proposes an integration framework covering use case suitability, autonomy levels, technical integration, governance, security, employee enablement, and measurable impact. The paper concludes that Agentic AI can become a productivity lever when implemented as a human centered capability with responsibility and accountability retained by people. 

---
# MR-GVNO: A Geometry-Aware Variational Physics-Informed Neural Operator for Mindlin-Reissner Plates on Irregular Domains 

**Authors**: Siqi Wang, Daobo Sun, Yizheng Wang, Yilong Zhang, Yabin Jin, Xiaoying Zhuang, Timon Rabczuk  

**Link**: [PDF](https://arxiv.org/pdf/2606.16624)  

**Abstract**: Plate and shell structures are widely used in engineering, making rapid response prediction under varying geometries, materials, and loads highly desirable. However, conventional finite element methods require repeated modeling and solution, resulting in high computational costs. This study proposes a geometry-aware variational neural operator for Mindlin-Reissner plate problems, termed MR-GVNO. The method uses boundary point clouds to represent irregular geometries and employs separate encoders for spatially varying material fields, pressure loads, and scalar physical parameters. A cross-attention mechanism integrates these inputs with query point information to predict transverse deflections and rotations at arbitrary locations. MR-GVNO is trained without labeled solution data using a variational physics-informed loss derived from the discretized total potential energy. It directly processes irregular point clouds and allows different physical fields to be discretized independently, avoiding interpolation onto a common grid. Numerical experiments on single-hole, double-hole, and L-shaped plates demonstrate accurate response prediction under homogeneous and heterogeneous materials and uniform and random loads. The model also achieves millisecond-level full-field inference and favorable cross-geometry generalization. 

---
# CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies 

**Authors**: Issa Sugiura, Daichi Hattori, Kazuo Araragi, Keita Ogawa, Shota Onose, Taro Makino, Teppei Usuki, Takashi Ishida  

**Link**: [PDF](https://arxiv.org/pdf/2606.16613)  

**Abstract**: As LLM agents become capable of increasingly long-horizon tasks, evaluating their performance in economic systems is becoming increasingly important. Unlike existing benchmarks that primarily evaluate a single agent interacting with a passive environment, economic systems are inherently multi-agent, requiring autonomous agents to communicate, negotiate, and transact while pursuing their own objectives over extended periods. We introduce CoffeeBench, a benchmark for evaluating LLM agents in a long-horizon multi-agent economy composed of heterogeneous firms. In CoffeeBench, two farmers, two roasters, and two retailers autonomously operate their businesses over a 90-day simulation, each seeking to maximize cumulative net income through communication and transactions while managing cash, inventory, and pricing. The evaluated model controls one coffee roaster, while the remaining firms are controlled by fixed reference agents. Across several recent open-weight and proprietary LLMs, all models outperform a passive baseline that takes no actions, with most achieving positive net income. Analysis of agent behavior reveals substantial differences in long-horizon economic interaction: higher-performing models communicate more actively with other firms, whereas Claude~Haiku~4.5 exhibits an idle-drift failure mode, repeatedly choosing inaction despite producing coherent assessments and plans. We release our code and agent trajectories to support future research. 

---
# ARB4WM: An Adversarial Robustness Benchmark for World Models in Continuous Control 

**Authors**: Junjian Zhang, Hao Tan, Ruonan Li, Dong Zhu, Aiping Li, Zhaoquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16605)  

**Abstract**: World models are widely used in robotic and agentic engineering control systems due to their ability to learn latent dynamics for planning and decision-making. As these systems are increasingly deployed in safety-critical settings, understanding their robustness under adversarial conditions has become essential. However, existing evaluations lack a unified benchmark for testing adversarial threats across the policy, value, and latent-dynamics levels of world-model agents. To fill this gap, we present ARB4WM, a unified evaluation framework for pre-deployment robustness and risk assessment of world-model agents under visual perturbations. ARB4WM defines five white-box loss objectives across these three levels and studies their effects when combined with single-step or multi-step perturbation strategies and temporal attack modes, including full-frame, half-sequence, and sparse-frame exposure. Specifically, we evaluate four Dreamer-style agents across 20 tasks from MetaWorld and the DeepMind Control Suite under different loss objectives, perturbation strategies, and temporal attack modes. Results show that attacks targeting value estimation, latent representations, and RSSM dynamics can be as damaging as direct policy disruption, and that early or frequent perturbations are especially harmful, while input-level defenses provide limited recovery under adaptive attacks. These findings suggest that safety, risk, and reliability assessment for world models should cover multiple component-oriented attack objectives and temporal exposure protocols rather than relying solely on action-space robustness. Source code is available at this https URL. 

---
# TNODEV: Toolbox for Neural ODE Verification 

**Authors**: Abdelrahman Sayed Sayed, Pierre-Jean Meyer, Mohamed Ghazel  

**Link**: [PDF](https://arxiv.org/pdf/2606.16567)  

**Abstract**: Neural ordinary differential equations (neural ODE) have started to appear in safety critical settings such as continuous-time controllers for cyber-physical systems and classifiers integrated into automated decision pipelines, raising the question of whether their behavior can be formally verified. Existing tools dedicated to neural ODE provide only a single reachability call without iterative input set refinement, limiting the precision of their verdicts to whatever one reachability call can deliver. We present TNODEV, the first sound formal verifier for neural ODE that integrates a falsification checker, a fast interval-based reachability backend based on continuous-time mixed monotonicity, a verification and refinement loop with three input-set splitting heuristics, and a parallel scheduler in a single end-to-end pipeline. TNODEV supports safe-set inclusion verification on pure neural ODE, neural ODE in closed loop with a neural network controller and general neural ODE (GNODE), with the safe set specified either as an interval or as the half-space intersection induced by a target classification label. We evaluate TNODEV on a range of benchmarks across safe-set inclusion and classification-robustness properties, including a direct reachability comparison against NNV~2.0 and CORA and a verification comparison against NNV2.0 on MNIST general neural ODE classifiers. 

---
# ROSA-RL: Uncertainty-Aware Roundabout Optimized Speed Advisory with Reinforcement Learning 

**Authors**: Anna-Lena Schlamp, Jeremias Gerner, Klaus Bogenberger, Werner Huber, Stefanie Schmidtner  

**Link**: [PDF](https://arxiv.org/pdf/2606.16558)  

**Abstract**: Roundabouts challenge automated driving in mixed traffic, as heterogeneous and non-deterministic human behavior, unknown driving intentions, and high interaction complexity create uncertainty about whether the conflict zone will be blocked or available at the moment of entry. We present ROSA-RL -- uncertainty-aware Roundabout Optimized Speed Advisory with Reinforcement Learning. It enables safe and efficient roundabout entry for automated and human-driven vehicles in mixed traffic through probabilistic conflict forecasting. A Transformer-based model predicts conflict zone occupancy over a five-second horizon, capturing multi-agent interactions to anticipate upcoming conflicts and available gaps. The prediction outputs encode uncertainty in future motion and intent, and augment the state of a classical RL framework, enabling uncertainty-aware speed coordination. Evaluated in simulations grounded in real-world data, ROSA-RL can effectively handle uncertainty and outperform a comparable model-based baseline, closing the gap to an ideal setting assuming fully known occupancy while improving traffic efficiency and safety. The source code of this work is available under: this http URL. 

---
# The Faithfulness Gap: Certifying Semantic Equivalence Between Natural-Language and Formal Mathematical Statements 

**Authors**: Noor Islam S. Mohammad, Tamim Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2606.16541)  

**Abstract**: Autoformalization, translating natural-language mathematics into formal proof assistants, is bottlenecked not by translation fluency but by \emph{faithfulness}: a formal statement can typecheck and be provable, yet still encode a different theorem than the source intended. We introduce \emph{Bidirectional Provability Fingerprinting} (\bpf{}), a framework that certifies faithfulness by characterizing each candidate through its forward and backward consequence neighborhoods in the ambient theory and matching these against probes derived from the natural-language statement. We further introduce four novel components: (i) \emph{Counterfactual Probe Generation} (\cpg{}), a contrastive procedure that synthesizes probes targeting specific drift directions; (ii) the \emph{Equivalence Spectrum}, a continuous faithfulness score that replaces brittle binary verdicts; (iii) \emph{Adaptive Probe Budget Allocation} (\apba{}), an information-theoretic budget router; and (iv) \emph{Faithfulness-Guided Decoding} (\fgd{}), which uses \bpf{} signals as a reward during autoformalization. We prove a \emph{drift detection theorem} and a \emph{PAC-faithfulness} result establishing that the equivalence class of a natural language statement is learnable from $\mathcal{O}(\log(1/\delta)/\varepsilon)$ probes under mild assumptions. We release \driftbench{}, a benchmark of $2{,}183$ NL/Lean~4 pairs with controlled drift labels across six subfields of mathlib4. \bpf{}\,+\,\cpg{} detects $89.6\%$ of drifted formalizations at a $3.0\%$ false-positive rate-against $41.2\%$ for typecheck and $63.3\%$ for LLM-judge baselines, and \fgd{} reduces the rate at which a state-of-the-art autoformalizer emits drifted statements by $47\%$. this https URL 

---
# Kairos: A Native World Model Stack for Physical AI 

**Authors**: Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, Zeyu Liu, Cong Wan, Pu Li, Ruiqing Yang, Xiaoou Li, Wei Wang, Kangkang Zhu, Yuwei Zhang, Shi Fu, Xiaoning Wu, Xuzeng Fan, Dacheng Tao, Xiaogang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16533)  

**Abstract**: World models are transitioning from passive visual generators to foundational, operational infrastructure for Physical AI: they must natively acquire world knowledge from heterogeneous experience, maintain persistent states over long horizons, and execute efficiently within real deployment constraints. We introduce Kairos, a native world model stack designed around these requirements. (1) Kairos learns the world by pioneering a Native Pre-training Paradigm governed by a Cross-Embodiment Data Curriculum, which organizes open-world videos, human behavioral data, and robot interactions into a progressive developmental pathway. (2) Kairos maintains the world by unified world understanding, generation, and prediction within a Native Unified Architecture equipped with Hybrid Linear Temporal Attention, where sliding-window attention captures local dynamics, dilated sliding windows capture mid-range dependencies, and gated linear attention maintains persistent global memory. We establish formal theoretical bounds demonstrating that this temporal factorization strictly limits error accumulation, mathematically guaranteeing state propagation across extended horizons. (3) Kairos runs the world by incorporating a Deployment-Aware System Co-Design to support low-latency rollout generation on server and consumer-grade hardware for real-world observation-action-feedback loops. Experiments on embodied world-model, long-horizon, and action-policy benchmarks show that Kairos achieves top level performance while offering a strong efficiency-capability trade-off. Together, these results position Kairos as a cohesive operational foundation for future self-evolving physical intelligence. 

---
# Model Graph Inductive Learning for Knowledge Graph Completion 

**Authors**: Mohommad Esmaei Khani, Mahdieh Hasheminejad, Ali Taherkhani, Hossein Hajiabolhassan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16509)  

**Abstract**: Link prediction in knowledge graphs fundamentally depends on the quality of learned embeddings for entities and relations. However, most existing methods derive these embeddings by aggregating only the local neighborhood of each entity, neglecting the global structure of the knowledge graph. This limited view prevents models from capturing higher-level structural patterns that are essential for accurate and generalizable link prediction.
To address these limitations, we introduce Model Graph Inductive Learning (\textbf{MGIL}), a framework that constructs a model graph by clustering entities based on the similarity of their incoming and outgoing relational structures or their entity types. A GNN is then applied to this model graph to produce embeddings that capture the global view of the knowledge graph. These embeddings subsequently serve as high-quality initial features %embeddings for the original knowledge graph, replacing random initialization and leading to more stable and expressive representations.
Extensive experiments on standard and recently proposed inductive benchmarks demonstrate that MGIL achieves state-of-the-art or highly competitive performance in inductive link prediction, highlighting its effectiveness across diverse graph settings. 

---
# Post-Hoc Merging is Not Enough: Many-Shot Model Merging with Loss-Gap Balancing 

**Authors**: Kyungjin Im, Miru Kim, Chanin Eom, Minhae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2606.16501)  

**Abstract**: Model merging has become a practical post-training strategy for building a single multi-task large language model (LLM) by combining multiple task-specialized models. However, most existing approaches rely on post-hoc merging, in which task-specific models are merged only once after training. This one-shot aggregation often suffers from task interference, leading to information erasure across individual tasks. In this work, we show that replacing post-hoc merging with an iterative many-shot merging protocol is effective in improving multi-task performance. Building on this insight, we propose METIS, Mitigating Erasure from Task Interference for Stable many-shot merging. METIS is a loss-aware many-shot merging method that addresses information erasure in post-hoc merging through task-wise loss-gap weighting and consensus-based masking. Notably, METIS exhibits significant performance improvement on the worst-performing task, effectively mitigating information erasure. (Project page: this https URL) 

---
# Steering Emotional Dynamics for Art Therapy: Controllable Narrative Script Generation through Hierarchically Guided LLM Agents 

**Authors**: Suqing Wang, Qinghai Miao, Chao Guo, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2606.16481)  

**Abstract**: Art therapy plays a vital role in emotional healing, in which narrative creation acts as the primary vehicle for emotional expression. Given the inherently dynamic nature of emotions during healing, narratives with finely controlled emotional fluctuations enable individuals to safely project inner conflicts and achieve emotional catharsis. Recently, with the rapid development of Large Language Models (LLMs), automated narrative generation technology has provided a new pathway to support such artistic designs. However, while existing methods can produce fluent texts, they struggle to generate narratives that adhere to specified affective trajectories, failing to meet the demands of emotion-oriented psychological healing. To address these issues, this paper proposes EC-Script, an LLM agent-based framework that enables hierarchical control of the affective trajectory in narrative generation for emotional healing. To ensure that the generated narratives strictly follow the given emotional patterns, EC-Script establishes overall narrative direction through Emotion-Trajectory Planning, propels scene-level plot development with Character-Driven Scene Generation, and regulates local emotional changes of characters via Emotion-Controlled Script Writing. Ultimately, it outputs scene-by-scene script content that remains highly consistent with the preset affective trajectory. Experimental results demonstrate that EC-Script significantly outperforms baseline methods in affective trajectory adherence, exhibiting excellent and reliable emotional controllability, thereby providing effective technical support for AI-assisted emotional healing scenarios. 

---
# Tensor-Coord: Algebraic Decomposition of Joint Plan Tensors for Conflict-Free Multi-Agent LLM Planning 

**Authors**: Mudit Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16478)  

**Abstract**: Large language models (LLMs) remain limited in multi-agent planning because independently generated plans can create coordination failures such as spatial collisions, resource contention, and temporal deadlocks. We introduce Tensor-Coord, a multilinear algebra framework that represents the joint plan of N agents as a third-order tensor \(T \in R^{N \times H \times A}\) over agents, timesteps, and actions. Canonical Polyadic (CP) and Tucker decompositions are used to identify latent coordination structure. The minimal epsilon-approximate CP rank R* defines a computable coordination complexity measure, with \(CC(Pi)=(R*-N)/N\). We prove that R*=N is necessary and sufficient for plan independence. The residual \(E=T-T_{R*}\) defines a conflict score over agent pairs, timesteps, and actions, localizing failures without domain-specific rules. Tucker factors provide interpretable agent roles, temporal phases, and action clusters that are converted into natural language constraints for iterative LLM replanning. Experiments on multi-robot delivery tasks across Easy (2 agents, 5x5 grid), Medium (3 agents, 5x5 grid), and Hard (4 agents, 5x5 grid) settings show convergence to conflict-free plans in 100% of 2-agent cases within 1.4 iterations on average, 80% of 3-agent cases within 3.2 iterations, and 60% of 4-agent cases within 4.0 iterations. CP rank scaled approximately linearly as \(R*(N) = 3.9N + 0.5\), supporting its use as a predictor of coordination complexity. 

---
# When Agent Automation Becomes Profitable: Quantifying and Insuring Autonomous AI Risk through Trace-Economic Underwriting 

**Authors**: Binyan Xu, Xilin Dai, Fan Yang, Kehuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16465)  

**Abstract**: AI agents can now take irreversible actions in operational systems, but agent-caused losses are still not clearly assigned, priced, or transferred. Providers often disclaim consequential damages, users are left with uncompensated losses, and default human review limits the efficiency gains of automation. We ask when autonomous AI deployment can become economically acceptable despite failure risk. Our answer is to quantify risk at the customer-task-trace episode level and transfer it through insurance. Automation is acceptable when its expected benefit exceeds the premium, control cost, and remaining risk. This requires a defined role with bounded permissions and comparable traces. We introduce trace-economic underwriting, which maps tool-use traces to customer exposure and claimable loss, then uses this representation for pricing, control, and risk transfer. It uses deterministic economic labels rather than an LLM judge. In our trace-to-loss testbed, trace-economic pricing reduces pricing MAE from $17.7K to $569 and removes regressive cross-subsidy. A 300-trace expert audit accepts 295 labels unchanged. On 1,000 real SWE-smith traces, trace-conditioned controls reduce CVaR95 by 72%. Theorem~1 gives a finite-sample scope condition. We release code, labels, and audit sheets. 

---
# Posterior Twins: Distributional Behavioral Simulation for Enterprise Decisions 

**Authors**: Ankit Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.16415)  

**Abstract**: Enterprise behavioral simulation requires more than producing a plausible response. Many decisions depend on the shape of a population under a proposed action: which segments accept, defect, hesitate, or move into risk-sensitive states. This paper introduces Posterior Twins, a memory-grounded digital-twin approach that represents likely behavior as an updated distribution under a specific decision context. We evaluate a family of Twinning Labs behavioral-model operating points on a 226-example held-out behavioral-response benchmark and report both modal accuracy and Wasserstein-1 distance. The results show that modal accuracy and distributional fidelity identify different operating regimes. TL-Twin Alpha achieves the lowest observed Wasserstein-1 distance in the reported result set ($W_1 = 1.16$), while TL-Twin Delta and TL-Twin Gamma provide balanced operating points near the modal-accuracy frontier. The paper frames these results as a systems result: governed memory, behavioral model routing, scenario orchestration, distributional aggregation, and auditability are necessary for turning simulated behavior into reusable enterprise decision evidence. 

---
# Looking Is Not Picking: An Attention-Segment Account of Tool-Selection Failures in LLM Agents 

**Authors**: Shiyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16364)  

**Abstract**: LLM agents mis-call tools, and the natural guess is that the model failed to see the right tool in a crowded harness. We show the opposite through a lens concurrent work sets aside -- the model's attention to labeled tool-definition segments. On real BFCL failures, by per-candidate attention argmax the model attends most to the correct tool 80% of the time (vs. 21% chance), and the gold is the under-attended segment on only 10%: it looks at the right tool and still picks wrong. This directly refutes the intuitive "crowded-harness / lost-in-the-middle" explanation: the failure is at the decision readout, not the harness, and we pin it there three ways. (1) Input vs. readout: repairing the prompt (reordering or duplicating the gold tool) recovers <=23% of failures, while readout-side interventions recover 59-91%. (2) Representation-invariance: two gold-pointed interventions in different representations -- an additive attention-logit bias and a residual-stream steering vector -- recover largely the same failures (per-task Jaccard 0.865 pooled, 0.79-0.91 per model), so the bottleneck is localized to the readout independent of which representation is poked. (3) A training-free, gold-free selector: per-segment attention closes most of the gold-free-vs-oracle gap on BFCL (+11.9 pts pooled function-name selection vs. +17.9-pt oracle headroom) and adds +14.9 pts on Seal-Tools; every model positive (exact McNemar p<=8e-4 each). Scopes differ: the causal attention-bias dose-response is bidirectional and monotonic on 10 mask-honoring models (3-32B), the full 0.5-32B span carrying only the correlational diagnostic; the deployable selector is evaluated on 5 single-turn models and does not yet transfer to a multi-turn loop. 

---
# Whose hotel does the AI recommend? An algorithm audit of reputation signals in LLM-assisted hotel selection 

**Authors**: Mirza Samad Ahmed Baig, Syeda Anshrah Gillani, Asher Ali  

**Link**: [PDF](https://arxiv.org/pdf/2606.16344)  

**Abstract**: Travelers increasingly ask large language model (LLM) assistants which hotel to book, making these systems gatekeepers of property visibility -- yet what moves their recommendations is undocumented. We conduct a pre-specified algorithm audit using a randomized choice-based conjoint: across personas, prompt templates, and twelve open-weight and proprietary models, assistants choose among five hotels whose guest rating, review volume and recency, management response, chain affiliation, price, eco-certification, and list position are independently randomized. We estimate the average marginal component effect of each signal on the probability of recommendation. Guest rating and price dominate (a top rating raises selection by 31.6 percentage points; a high price lowers it by 30.0), reproducing human valence-and-price primacy but over-weighting eco-certification and ignoring management response. List position -- a content-free artifact -- shifts recommendations causally, worth about \$12 per night. Stated reasons track revealed weights imperfectly. The findings ground generative engine optimization and the accountability of AI infomediaries in causal evidence. 

---
# Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules 

**Authors**: Wei Xu, Ke Yang, Gang Luo, Keli Zheng, Lingyan Hu, Jing Wang, Kefeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16337)  

**Abstract**: Predictive modeling for clinical tabular data is central to clinical decision support and therefore requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning-beyond-gradients paradigm for clinical tabular prediction. Instead of relying on neural network weight updates, MHL uses a large language model (LLM)-driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable decision system. The resulting model is expressed not as opaque parameters, but as versioned pure-Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support. 

---
# Phase-Aware Guidance Injection for Recurrent MAPPO in Assembly-Line Disruption Recovery 

**Authors**: Xin Huang, Yongcai Wang, Fengyi Zhang, Zhikun Tao, Yunjun Han, Naiqi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16330)  

**Abstract**: Disruption recovery in industrial assembly lines requires timely decisions under machine faults, worker absence, and emergency orders. Existing methods either rely on rigid handcrafted recovery logic or learn adaptive policies that do not readily exploit heterogeneous external recovery knowledge at decision time to reduce abnormal recovery time (ART) and preserve on-time delivery (OTD). To address this gap, we propose a phase-aware guidance injection framework that augments a trained recurrent MAPPO (RMAPPO) scheduling policy through logit-level action bias during evaluation. The framework provides a unified decision-time interface for rule-based, replay-based, and online LLM-based guidance, while activating intervention only during abnormal and recovery phases. Experiments on a custom AssemblyLineEnv show that high-quality rule guidance yields the strongest gains, replay-based guidance degrades smoothly under imperfect availability, and online LLM guidance still provides useful intermediate improvements. These results show that decision-time guidance injection can exploit heterogeneous recovery hints without redesigning the actor. 

---
# Exploiting Search in Symbolic Numeric Planning with Patterns 

**Authors**: Matteo Cardellini, Enrico Giunchiglia  

**Link**: [PDF](https://arxiv.org/pdf/2606.16329)  

**Abstract**: In this paper, we present a procedure for numeric planning based on Symbolic Pattern Planning (SPP). Given a numeric planning problem $\Pi$, a pattern $\prec$ is a sequence of actions used to define a formula encoding the subsequences of $\prec$ executable from a starting state $S$. Cardellini, Giunchiglia, and Maratea (2024a) follow the Planning as Satisfiability approach by defining, at each step $n \ge 0$, a formula $\Pi^\prec_n$ in which $(i)$ the pattern $\prec$ is computed only for $n=0$ in the initial state $I$ of $\Pi$, and then exploited at each step $n$, $(ii)$ the starting state $S$ is set to $I$, and $(iii)$ the set $G$ of goals is required to hold in the last state that can be reached by one of the subsequences of $\prec$ concatenated $n$ times. The procedure begins with $n=0$, terminates as soon as $\Pi^\prec_n$ is satisfiable, and otherwise proceeds by incrementing $n$. In this paper, possibly at each step, $(i)$ we symbolically search for an intermediate state $P$ reachable from $I$, closer to a goal state, $(ii)$ dynamically recompute the pattern $\prec_h$ -- to be used in the next step -- in $P$, $(iii)$ refine the pattern $\prec_g$ used to reach $P$, and $(iv)$ start the new search from the state $S$ which can be either the initial state $I$ or the last computed intermediate state $P$, exploiting the computed patterns $\prec_g$ and $\prec_h$ to define the pattern $\prec$ to be used in the search. In particular, at each step, we define a formula $\Pi^{\prec}_{S,P}$ encoding the existence of a state $P'$ closer than $P$ to a goal state, with $P'$ reachable from the starting state $S$ when using the pattern $\prec$. We present different techniques for producing such formulas, each corresponding to a different strategy for exploring the search space. We prove their correctness and completeness, the latter under certain conditions. 

---
# AdaSTORM: Scaling LLM Reasoning on Dynamic Graphs via Adaptive Spatio-Temporal Multi-Agent Collaboration 

**Authors**: Bing Hao, Ruijie Wang, Haodong Qian, Yunlong Chu, Yuhang Liu, Yumeng Lin, Minglai Shao, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16328)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable potential in dynamic graph reasoning, but suffer from a scaling bottleneck: current models can only handle graphs with tens of nodes, constrained by exponential reasoning overhead and finite context windows. While multi-agent systems (MAS) offer collective reasoning and topology-aware orchestration, capabilities naturally suited for graph-structured tasks, their application to dynamic graphs remains unexplored. This paper presents Scaling LLM Reasoning on Dynamic Graphs via Adaptive Spatio-Temporal Multi-Agent Collaboration (AdaSTORM), a framework that reformulates large-scale dynamic graph reasoning into two stages: (i) Adaptive Partitioning, partitioning large-scale dynamic graphs into subregions that match the model's reasoning capacity while minimizing inference cost; and (ii) Collaborative Reasoning, aligning graph partition topologies with a spatio-temporal decoupled multi-agent architecture. AdaSTORM is the first multi-agent framework tailored for dynamic graph reasoning. Extensive experiments show that AdaSTORM successfully breaks through the scaling bottleneck, scaling reasoning to thousand-node graphs with over 90% accuracy across several large-scale dynamic graph settings without external tools, significantly outperforms seven competitive baselines. Furthermore, it achieves state-of-the-art accuracy on existing benchmarks and generalizes robustly to real-world datasets. The source code is available at: this https URL. 

---
# Architectural Wisdom: A Framework for Governing Optimization in AI Systems 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16319)  

**Abstract**: Modern AI systems exhibit structural failures that capability scaling alone does not reliably fix: they optimize under-specified objectives with no architectural mechanism to question whether the objective should be optimized at all. Engagement maximization can amplify harmful pathways; tool-using agents can commit irreversible actions; preference-trained language models can become sycophantic. We argue that this failure is a wisdom problem, not an intelligence problem. We use "wisdom" in a deliberately architectural sense, not as a claim about virtue, consciousness, or moral omniscience. Intelligence accepts a goal and optimizes within it; wisdom interrogates whether the goal should be optimized at all. The two are separable architectural properties. We propose architectural wisdom as a corrigible objective-governance layer above the optimization substrate. The layer makes three structural commitments explicit and nondegenerate before any action: temporal horizon, relational boundary, and irreversibility. It is realized by four components (Structural Utility Transform, Moral Admissibility Interface, Arbitration and Escalation Controller, Value Revision Channel) that compute a six-coordinate wisdom tuple over horizon, relational coverage, irreversibility, admissibility, value revision, and auditability. We motivate the architecture by eight cases drawn from contemporary AI failures, secular wisdom traditions, and hard ethical situations, and defend the distinction against the intelligence-completeness thesis using goal-questioning over goal-taking, Bostrom's orthogonality, structural separation in our exemplar cases, and persistent failure modes despite capability scaling. The framework is the conceptual contract for a larger architecture whose formal specifications and empirical validation are developed in subsequent work. 

---
# State-Grounded Multi-Agent Synthetic Data Generation for Tool-Augmented LLMs 

**Authors**: Rahul Khedar, Eshita, Sneha Teja Sree Reddy Thondapu, Mayank Malhotra, Arup Das, Jitesh Chandra, Yun-Shiuan Chuang, Chaitanya Kulkarni, Arun Menon, Linsey Pang, Avinash Karn, Mouli V, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2606.16307)  

**Abstract**: Training tool-augmented LLM agents requires large corpora of multi-turn, tool-grounded conversational data that is expensive to annotate, privacy-constrained in production settings, and largely absent from public datasets. We present StateGen, a synthetic data generation platform that produces scored, reasoning-trace-rich training conversations by orchestrating a four-role LLM loop: a persona-conditioned user simulator, an agent under test, a state-grounded tool simulator, and a multi-axis LLM judge. The key architectural contribution is an authoritative state manager that maintains a structured world-state object across turns, enforcing a backend-is-truth invariant that eliminates the dominant class of tool-call hallucinations by construction. StateGen extends naturally to hierarchical multi-agent settings by declaring sub-agents as tools, all sharing a single state object. We report results on 64,698 evaluated conversations across three production corpora: tool-call hallucination scores reach 9.66/10, the system supports persona-driven variation via a 23-dimensional trait vector, and a cleanly separated train and golden evaluation set split confirms the data is not memorization bait (per-criterion gap analysis). Comparison with eight external systems shows that no single publicly available platform combines multi-turn generation, state-grounded tool simulation, hierarchical multi-agent support, and built-in judge scoring. 

---
# SpecAlign: Efficient Specification-Grounded Alignment of Large Language Models via Synthetic Data 

**Authors**: Wenjie Wang, Yue Huang, Zhengqing Yuan, Han Bao, Shiyi Du, Yuchen Ma, Yue Zhao, Yanfang Ye, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16276)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, alignment is no longer governed by a single universal notion of safety or helpfulness, but instead by provider- or application-specific model specifications. These specifications are typically long, structured, and frequently updated, yet existing alignment pipelines lack a systematic mechanism to operationalize them as training signals. In this paper, we propose specification-grounded alignment, a new alignment paradigm that treats provider-authored model specifications as the primary alignment target rather than abstract principles or static benchmarks. To instantiate this paradigm, we introduce SpecAlign, a framework that synthesizes alignment data directly from specification documents. SpecAlign combines structured rule annotation, controllable specification instantiation, and multi-agent adversarial data synthesis to generate fine-grained, boundary-aware preference pairs that capture both compliant behaviors and meaningful specification violations. Experiments across multiple model specifications and backbone models demonstrate that training with SpecAlign consistently improves rule compliance while preserving general capabilities and avoiding over-conservative behavior. These results suggest that grounding alignment in explicit model specifications enables rapid, precise, and scalable adaptation of LLM behavior to evolving policy requirements. 

---
# Latent Thought Flow: Efficient Latent Reasoning in Large Language Models 

**Authors**: Xiandong Zou, Jing Huang, Jianshu Li, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.16222)  

**Abstract**: Large Language Models (LLMs) increasingly rely on intermediate reasoning, yet explicit Chain-of-Thought (CoT) suffers from a linguistic space bottleneck: each thought must be decoded into tokens, causing high inference overhead. Latent reasoning moves deliberation into continuous space, but existing methods mostly learn deterministic or reward-maximizing paths, lacking a principled way to allocate probability across trajectories with different correctness and costs. We propose Latent Thought Flow (LTF), which models reasoning as variable-length continuous trajectories and trains a sampler to match a reward-induced posterior over answer quality and computation cost. We instantiate this with a continuous GFlowNet using stochastic latent transitions. To handle sparse answer supervision, we introduce an Entropy-Weighted Subtrajectory Balance objective for intermediate rewards and a reference-prior regularizer to anchor exploration. Experiments under finetuning and transfer learning settings show that LTF outperforms explicit CoT and latent reasoning baselines, improving accuracy by 9.5% while reducing reasoning length by 27.2% on average compared with strong latent reasoning baselines. 

---
# Sensor-Conditioned Representation Learning via Scene-Relevant Observation Quotients 

**Authors**: Yan Jiao, Pin-Han Ho, Limei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.16210)  

**Abstract**: Learned representations in intelligent sensing systems are often evaluated by reconstruction fidelity or downstream prediction accuracy, but these criteria do not specify which latent distinctions are justified by the sensing process. In sensor-conditioned environments, nuisance factors can change measurements without changing the scene, while distinct scenes may be indistinguishable under limited sensing capability. This paper formulates sensor-conditioned representation correctness as preserving sensing-supported scene distinctions while suppressing nuisance-induced and sensor-unsupported variation. We introduce the scene-relevant observation quotient, a representation target induced by sensing-supported distinguishability after nuisance canonicalization, and develop Observation-Quotient Tucker-Structured Autoencoding (OQ-TSAE), a scene-nuisance factorized framework with diagnostics for false distinction, false merge, nuisance sensitivity, and latent ordering consistency. Experiments on a controlled benchmark show that quotient-consistent supervision improves representation-correctness diagnostics over reconstruction-oriented, metric-learning, and contrastive-learning baselines. Sensitivity, perturbation, and ablation studies show the importance of quotient-aligned supervision, reliable quotient relations, and quotient geometry. Complementary real-radar experiments show that a reconstruction-only OQ-TSAE variant retains competitive downstream utility, robustness under observation degradation, and low seed-to-seed variability. These results suggest that sensor-conditioned representations should be evaluated not only by predictive utility, but also by whether their latent geometry preserves sensing-justified scene distinctions. 

---
# Measuring Whether LLM Tutors Teach or Solve: A Diagnostic for Educational Impact 

**Authors**: Junyi Yao, Zihao Zheng, Baichuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16206)  

**Abstract**: Large language models are increasingly proposed as educational tutors, yet stronger task-solving ability does not necessarily imply stronger learning support. Motivated by recent calls to measure the social impact of NLP systems in practice, we study whether public LLM tutoring benchmarks distinguish learning-supportive behavior from mere answer production. We propose a lightweight diagnostic based on the gap between solving-oriented and pedagogy-oriented benchmark performance. Using public MathTutorBench leaderboard results, we show that these dimensions are only partially aligned: across eight publicly reported models, the correlation between solving and pedagogy composites is 0.421, and several models shift meaningfully in rank when evaluation moves from solving to pedagogy. We then analyze the public TutorBench sample and show that agency-relevant behaviors are explicitly encoded in benchmark rubrics, especially in active-learning settings that reward guiding questions, calibrated hints, and non-disclosive scaffolding. Together, these findings suggest that educational-impact evaluation should not treat task success as a sufficient proxy for learning support. We argue that public tutoring benchmarks can better support positive-impact evaluation by reporting solving-oriented and pedagogy-oriented scores separately and by making disclosure-sensitive, student-agency-preserving criteria more explicit. 

---
# PAL-Bench: Evidence-Grounded Profile Reconstruction from Longitudinal Personal Albums 

**Authors**: Qiwei Yan, Zhiqiang Yuan, Zexi Jia, Nanxing Hu, Kailin Lyu, Jie Zhou, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16175)  

**Abstract**: Longitudinal personal albums are weak-schema multimodal databases: noisy perceptual records whose key facts require joins across faces, text, timestamps, locations, and repeated events. Existing visual, video, document, and lifelog benchmarks test sub-problems, but not album-scale profile reconstruction with social identity binding and evidence citation. Benchmarking this task is difficult because the ground truth needed for evaluation--owner profiles, social graphs, face-name maps, and evidence provenance--is private state that real albums cannot safely release. We introduce PAL-Bench, a controlled benchmark for evidence-grounded reconstruction under a public-record contract. Its Evidence Compiler builds latent private worlds, programs target-level evidence paths, renders album pixels, re-measures them through perception pipelines, and exports audited public/private views. Agents receive only perception-derived public records; targets, identifier maps, and evidence paths remain hidden. PAL-Bench contains 50 synthetic users, 36,659 public photo records, and 2,799 targets over owner facts, identities, and relations. A privacy-preserving audit with 10 participants confirms that PAL-Bench evidence structures match real private albums, though equivalent releases remain privacy-prohibitive. Across seven systems and two compute-matched diagnostics, a seven-metric protocol reveals a gap between plausible profile summarization and faithful social reconstruction: systems recover some owner facts but struggle with recurring identities and evidence citation. PAL-TRACE, a reference framework that freezes identity bindings before owner-fact mining, performs best but leaves hard identity resolution far from solved. PAL-Bench provides a testbed for perceptual entity resolution, multimodal data integration, temporal evidence aggregation, and provenance-aware structured prediction. 

---
# TimeVista: Exploring and Exploiting Vision-Language Models as Judges for Time Series Forecasting 

**Authors**: Zhi Chen, Yuxuan Wang, Jialong Wu, Yong Liu, Haoran Zhang, Xingjian Su, Jianmin Wang, Mingsheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2606.16173)  

**Abstract**: High-quality time series forecasting is pivotal for real-world decision-making. However, traditional point-wise metrics often fail to reveal complex temporal patterns and align poorly with human intuitive preferences. While the ''LLM-as-a-Judge'' paradigm has revolutionized text evaluation by providing flexible, human-aligned judgment, its application to time series remains largely unexplored. In this paper, we leverage Vision-Language Models (VLMs) as judges for time series forecasting, harnessing their ability to comprehend time series plots grounded in textual information. Specifically, we propose a novel framework integrating micro- and macro-level judgments informed by contextual information to evaluate time series forecasting. To this end, we introduce TimeVista, a comprehensive VLM-as-a-Judge benchmark comprising 5563 time series samples paired with detailed evaluation rubrics. Extensive meta-evaluations demonstrate that VLMs are highly reliable judges, achieving significantly higher consistency with human preferences than conventional metrics. Building upon our benchmark, we comprehensively assess recent Time Series Foundation Models (TSFMs) under the VLM-as-a-Judge paradigm. Our results demonstrate that VLMs serve as robust and interpretable judges, providing a comprehensive, human-aligned standard for evaluating time series models. 

---
# AI Pluralism and the Worlds It Misses 

**Authors**: Rashid Mushkani  

**Link**: [PDF](https://arxiv.org/pdf/2606.16167)  

**Abstract**: AI pluralism is often framed as a problem of representing diverse values, preferences, users, or outputs. This paper argues that this framing is incomplete because AI systems also impose ontologies: they define what counts as an entity, relation, feature, harm, benefit, and valid form of evidence. We define ontological flattening as the conversion of situated, contested, and historically specific meanings into a restricted technical category, proxy, aggregation rule, or benchmark target that is treated as neutral and difficult to contest. The paper develops a bounded conceptual and qualitative synthesis across value pluralism, pluralistic alignment, participatory and democratic AI, procedural justice, science and technology studies, accountability research, aggregate themes from 11 expert interviews, and three urban AI companion cases. The cases illustrate how pluralistic methods can improve or structure model behavior while still compressing categories, proxies, aggregation rules, and revision rights before affected actors have procedural standing. We introduce Pluralistic Lifecycle Governance (PLG) as a preliminary qualitative audit scaffold for documenting ontological openness, epistemic inclusion, procedural authority, evaluation pluralism, and lifecycle accountability. PLG is not presented as a validated scoring instrument; it is a framework for making the evidence and governance conditions of pluralistic AI explicit. 

---
# The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning 

**Authors**: Haolong Qian, Xianliang Yang, Yinuo ma, Lirong Che, Feng Lu, Ye Guo, Lei Song, Jiang Bian, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16152)  

**Abstract**: Knowledge distillation from powerful reasoning models is widely used to improve Small Language Models (SLMs) on mathematical reasoning, often assuming that traces with higher reward model scores provide more useful supervision. We identify a counterintuitive \textbf{Quality-Utility Paradox} in mathematical reasoning distillation. Data refined or synthesized by a stronger Oracle obtains higher perceived quality according to reward models, yet consistently underperforms traces generated by the SLM itself and selected through rejection sampling across Qwen2.5, LLaMA-3, and DeepSeek families. Our analysis shows that Oracle refinement couples logical repair with distributional drift away from the SLM's native reasoning distribution. This drift increases the learner's adaptation cost and can outweigh the benefit of improved reasoning logic. To test this mechanism, we introduce \textbf{Style-Aligned Refinement}, which preserves the native trajectory of the SLM while retaining logical repair from the Oracle. This intervention lowers adaptation cost and restores downstream utility. These findings suggest that effective mathematical reasoning distillation should jointly optimize perceived solution quality and learner-data compatibility, rather than relying solely on reward-model scores. The datasets and code are available at this https URL. 

---
# LiteOdyssey: A Lightweight Reasoning AI Agent for Interpretable Rare-Disease Diagnosis 

**Authors**: Minh-Ha Nguyen, Erica Gray, Chih-Ting Yang, Rizwan Hamid, Lingyao Li, Siyuan Ma, Thomas A. Cassini, Cathy Shyr  

**Link**: [PDF](https://arxiv.org/pdf/2606.16149)  

**Abstract**: Most medical AI systems improve by scaling additional machinery: more fine-tuning data, more agents, and/or larger retrieval databases. In rare-disease diagnosis, however, such scaling can produce systems that are difficult to deploy, audit, and maintain. We asked whether state-of-the-art diagnostic performance could instead be achieved by extending the reasoning chain of a single AI agent: guiding it with a diagnostic policy, developed through human-AI collaboration and augmenting with freely available biomedical tools. We introduce LiteOdyssey, a lightweight rare-disease diagnostic framework that guides reasoning language model through a clinical genetics workflow. This framework was developed through Policy Iteration with Human Feedback (PIHF) and uses dynamic access to public biomedical tools. On two challenging benchmarks that provide only patient clinical features, LiteOdyssey achieved state-of-the-art performance, with an overall disease Recall@1 of 59.3% over the combined 1,243 cases of LIRICAL (n = 370) and the PhenoPacket Store (n = 873). Both benchmarks have a high proportion of ultra-rare disease (a prevalence below 1 in 1,000,000, with ultra-rare shares of approximately 45% and 52.8%, respectively). On the more difficult PhenoPacket subset, where causal diseases were not mapped to Orphanet in our rarity-mapping pipeline, LiteOdyssey achieved 60.7% Recall@1, compared with 10.7% for the same baseline model (GPT-5.4) without tools. This performance was achieved without fine-tuning, multi-agent ensembles, or a large case-retrieval database. Gains were also observed in the following: on cases never seen during development, on a private cohort of real-world rare disease patients, and on a smaller open-weights model. LiteOdyssey suggests a path toward rare-disease AI systems that are accurate, easier to deploy, and more transparent for physician review. 

---
# VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models 

**Authors**: Sen Xu, Shixi Liu, Wei Wang, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Xin Zhou, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16140)  

**Abstract**: This technical report introduces VibeThinker-3B, a compact dense model with 3B parameters developed to investigate how far verifiable reasoning can be pushed within a strictly small-model regime. Building upon the Spectrum-to-Signal post-training paradigm, we systematically enhance the model through an optimized pipeline that includes curriculum-based supervised fine-tuning, multi-domain reinforcement learning, and offline self-distillation. Experimental evaluations demonstrate that VibeThinker-3B achieves frontier-level performance on highly demanding verifiable tasks. Specifically, it attains a score of 94.3 on AIME26 (improving to 97.1 with claim-level test-time scaling), an 80.2 Pass@1 on LiveCodeBench v6, and exhibits strong out-of-distribution generalization with a 96.1\% acceptance rate on recent unseen LeetCode contests. This effectively places it in the performance band of first-tier reasoning systems, matching or exceeding flagship models that are orders of magnitude larger, such as DeepSeek V3.2, GLM-5, and Gemini 3 Pro. Furthermore, a score of 93.4 on IFEval confirms that this extreme reasoning enhancement does not compromise strict instruction controllability. Extending our previous 1.5B work, these findings motivate the Parametric Compression-Coverage Hypothesis, which views verifiable reasoning as compressible into compact reasoning cores, while open-domain knowledge and general-purpose competence require broad parameter coverage over facts, concepts, and long-tail scenarios. This perspective suggests that compact models are not merely deployment-efficient substitutes, but a complementary path toward frontier-level performance in parameter-dense capability regimes. 

---
# Thinking with Visual Grounding 

**Authors**: Junkai Zhang, Yihe Deng, Kai-Wei Chang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16122)  

**Abstract**: Visual thinking should not only sound right; it should show its evidence. While recent vision-language models (VLMs) can produce natural-language reasoning traces, these traces often leave the supporting image regions implicit, making them hard to verify and difficult to supervise. We introduce visually grounded thinking, a reasoning process in which models interleave natural-language thoughts with explicit point or box groundings of the visual evidence used at each step. This lets the model express intermediate reasoning in language while grounding key objects in the image regions they refer to. To train this behavior, we construct a scalable synthesis pipeline that distills correct visual reasoning traces, extracts the visual objects required by the traces, grounds them with a SAM3-based agent, and derives aligned point and box supervision from the resulting masks. We further propose grounding-aware reinforcement learning, which combines answer correctness rewards with dense grounding rewards that score whether generated object references match the correct image evidence. Across two counting benchmarks and four spatial reasoning benchmarks, adding visually grounded thinking to Gemma3-4B-IT consistently improves performance over the original model and the non-grounded thinking baseline. On spatial reasoning, the visually grounded thinking 4B models match, and in some cases surpass, Gemma3-27B-IT from the same model family. Our analysis shows that point grounding is well suited to counting, while box grounding benefits most from explicit grounding rewards on spatial tasks. Overall, our results show that VLMs think better when their intermediate thoughts are tied to the image regions that make them true. 

---
# Know Your Limits : On the Faithfulness of LLMs as Solvers and Autoformalizers in Legal Reasoning 

**Authors**: Olivia Peiyu Wang, Sanna Wong-Toropainen, Daneshvar Amrollahi, Ryan Bai, Tashvi Bansal, Arush Garg, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2606.16118)  

**Abstract**: Large Language Models (LLMs) achieve strong performance on reasoning tasks, but whether this reflects faithful logical inference or heuristic approximation remains unclear. We study this question in legal entailment by comparing three paradigms, including pure LLM classification, LLM-based Formal Reasoning, and solver-based Formal Reasoning using the Z3 SMT solver, on a re-annotated subset of ContractNLI across five LLMs. Our re-annotation reveals a systematic and measurable gap between pragmatic legal interpretation and strict formal entailment, where a substantial proportion of legally sound inferences are not formally grounded without additional unstated assumptions. While introducing formal structure improves accuracy, with LLM-based Formal Reasoning achieving the highest benchmark performance, we show that this gain does not imply faithful reasoning. We identify three recurring failure modes: scope laundering, where LLMs report solver-inconsistent classifications without executing the underlying formal reasoning, producing conclusions that appear logically grounded but are not; implicit constraint blindness, where LLMs overlook logical constraints present in formal representations; and program synthesis failures, where LLMs generate incorrect Z3 code despite structured prompting. Critically, scope laundering persists across all models, raising serious concerns about the faithfulness of LLM-based formal reasoning as a proxy for symbolic execution. These results reveal a fundamental gap between benchmark accuracy and logical faithfulness. 

---
# RecourseBench: A Modular Framework for Reproducible Algorithmic Recourse Evaluation 

**Authors**: Zahra Khotanlou, Hashir Ahmed, Chenghao Tan, Ahmed Abdelaal, Amir-Hossein Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16113)  

**Abstract**: Algorithmic recourse methods provide counterfactual explanations that inform individuals of the actions required to overturn an unfavorable model decision. Despite rapid methodological progress, principled comparison remains elusive; existing frameworks are often difficult to extend and lack both interoperability and systematic verification that integrated methods faithfully reproduce their originally reported results. We introduce \emph{RecourseBench}, a unified evaluation framework built around three commitments namely, modularity, reproducibility, and interactivity. The framework decomposes the pipeline into five fully decoupled layers -- Data, Preprocessing, Model, Recourse Method, and Evaluation -- governed by abstract interfaces and a dynamic registry. To address the reproducibility gap in prior benchmarks, we introduce a four-tier classification system in which every integrated method is validated by an automated test suite against its originally reported results. We further provide an interactive web interface for flexible, configuration-driven comparison across methods, datasets, and model architectures. Our framework currently integrates 28 state-of-the-art recourse methods and, to our knowledge, constitutes the first recourse benchmark to explicitly enforce method-level reproducibility through automated, quantitative testing. 

---
# Rhythm of the Deep: A Computational-Linguistic Test of Duality of Patterning in Sperm Whale Codas 

**Authors**: Mudit Sinha, Sanika Chavan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16084)  

**Abstract**: Human language has often been described as combining structure at two levels: lower-level units combine into larger units, which then combine into larger sequences. We test for this design feature, duality of patterning, in sperm whale codas using 1,483 codas from the Dominica Sperm Whale Project. Because acoustic similarity can imitate symbolic structure, we treat the problem as computational-linguistic structure discovery from continuous audio rather than as a direct claim about language or meaning. We use a consensus of frozen audio encoders, held-out structural tests, per-statistic nulls, and acoustic-null recoverability gates. The evidence supports a narrow two-tier architecture. At the lower tier, clicks compose into codas not by a stable ordered rule, but by which clicks are present together with their inter-click rhythm. At the upper tier, coda tokens show bout-level sequential dependence, with an NSB second-order transfer-entropy lift of 0.132 bits (p = 0.002). Under tempo scaling, encoder-derived click identity is strongly rate-bound, while coda identity remains substantially more stable, yielding a measurable abstraction gradient across the click-to-coda step. Rhythm-only baselines recover substantial lower-tier structure but fail to reproduce the upper-tier sequential-dependence signal. We do not claim language, semantics, perception, or human-like phonemes. Instead, we report representation-level evidence for a duality-of-patterning-like architecture whose lower tier is rhythmic rather than segmental, and provide a portable null-controlled framework for testing combinatorial structure in induced acoustic token systems. 

---
# Mind-Studio: Executable World Models with Lookahead Evaluation for Partially Observable Games 

**Authors**: Yifei Dong, Mingen Zheng, Linquan Wu, Jeff Z. Pan, Jiaxin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.16070)  

**Abstract**: World-model synthesis aims to turn interaction experience into an internal model of environment dynamics. Existing symbolic approaches often fit observed transitions or mixtures of local rules, but they do not produce a complete executable program that can run independently of the real environment. We present Mind-Studio, a framework that synthesizes executable pygame-style world models from state-action-next-state trajectories using large language models. Mind-Studio combines entropy-selected traces with a lightweight game skill file containing object, action, and static scene information extracted from screenshots. We evaluate synthesis quality with a K-step lookahead fidelity protocol that compares generated world-model rollouts against Real-ALE rollouts from the same state. On Montezuma's Revenge, Mind-Studio improves chosen-action next-state prediction from 0.3% for PoE-World to 48.7% while verifying 5 of 8 subgoals; across Alien, Assault, and Skiing, it achieves stronger branch-level fidelity than prior learned lookahead sources. 

---
# Auditing Reward Hackability in Code RL Training Environments 

**Authors**: Shreshth Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16062)  

**Abstract**: We measure the rate at which code RL environments accept incorrect solutions as correct. On a 49-task sample of SWE-bench Verified, 28.5% of tasks have test suites weak enough that a Docker-verified incorrect patch passes them. On 20 R2E-Gym tasks across 6 repositories, the same pipeline at single-shot exploit generation yields 25.0%. A random-effects meta-analysis over 134 frontier model submissions to SWE-bench Verified finds, within the same human-rated difficulty stratum, model Pass@1 is +14.14 percentage points higher on flagged-hackable tasks than on robust ones (95% CI [+11.80, +16.48]; one-sided p < 10^-6; I^2 = 0%; 123 of 134 models positive). We then describe a procedure for hardening the broken tasks. An inline LLM judge with a Docker gold-sanity gate runs each generated test against the gold solution before the judge is consulted. On the 11 broken tasks in the audit, the gate flags 65 of 105 decisive LLM-generated tests as failing on the gold patch itself, a 61.9% per-augmentation defect rate the LLM judge alone misses. With diversity-biased retry, the loop converges 9 of 11 tasks to a gated upgrade. 

---
# SciText2Eq: Assessing LLMs for Explainable Equation Generation for Scientific Creativity 

**Authors**: Yifan Mo, Xiao Fu, Yue Su, Qingyu Meng, Koen Hindriks, Qingzhi Liu, Jiahuan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2606.16003)  

**Abstract**: This work investigates the ability of large language models (LLMs) to generate mathematical equations from scientific texts. Prior work faces challenges in unstructured grounding, multi-equation dependency, and humanaligned evaluation. To this end, we construct a dataset of AI research papers, pairing contextual passages with ground-truth equations and variable descriptions. We develop an explainable equation generation workflow and evaluate it across diverse open- and closed-source LLM backbones. We introduce an evaluation protocol combining automatic metrics, LLM-based rubrics, and human judgments to assess accuracy, explainability, and human-LLM alignment. Results indicate that LLMs perform moderately on lexical- and syntactic-based similarity, while struggling with semantic accuracy. Comparisons between LLM-based evaluations and human judgments reveal limited alignment, highlighting challenges in using LLMs to assess equation quality. These findings offer insights for improving equation generation models and developing more reliable evaluation methods for scientific text. We provide code and data for reproducibility. 

---
# Agentic Framework for Deep Learning workload migration via In-Context Learning 

**Authors**: Qiyue Liang, Steven Ingram, George Vanica, Andi Gavrilescu, Newfel Harrat, Hassan Sipra, Sethuraman Sankaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.15994)  

**Abstract**: Translating deep learning models from PyTorch's flexible, object-oriented design to JAX's functional, stateless setup is usually a manual and error-prone task. Automated migration is challenging because Large Language Models (LLMs) struggle with strict and dynamic API alignment and are prone to mistakes for exacting operations. We propose a fully autonomous system that combines In-Context Learning (ICL) with oracle-driven self-debugging. First, we curated an ICL context that serves as a strict reference for idiomatic JAX styling and test case generation. Second, instead of depending on the LLM to deduce mathematical outputs, we run the source PyTorch modules to get their actual dynamic tensor states. This creates an unchangeable execution oracle. We then use an autonomous agentic loop to synthesize tests based on the oracle data. The test cases are executed repeatedly, and the traceback is sent back to the LLM for self-correction. Ablations show that combining ICL references with oracle grounding and self-debugging greatly outperforms pure instructional and basic agentic baselines. This improvement does not add an excessive computational overhead. Our lightweight pipeline achieves 91% numerical equivalence (compared to baseline: 9%, instruction + self-debugging: 27%) on neural modules, providing a highly reliable, scalable blueprint for cross-framework migration. This has been validated across several state-of-the-art models including SAM (segment anything), T5, Code Whisper amongst others showing high numerical equivalency. Code: this https URL 

---
# UrbanWell: Benchmarking Multimodal Large Language Models for Spatio-Temporal Urban Wellbeing Analytics 

**Authors**: Yanxin Xi, Xiang Su, Jie Feng, Yu Liu, Sasu Tarkoma, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2606.15890)  

**Abstract**: Understanding urban wellbeing from multimodal data requires integrating heterogeneous spatial and temporal signals, posing significant challenges for current multimodal large language models (MLLMs). We introduce UrbanWell, a large-scale benchmark designed to systematically evaluate the spatio-temporal reasoning capabilities of MLLMs for urban wellbeing analytics through joint modeling of satellite and street view imagery. UrbanWell spans 38 cities across multiple years and includes diverse indicators covering (1) environmental conditions (CO$_2$, NO$_2$, PM${2.5}$, and Normalized Difference Vegetation Index), (2) spatial accessibility (minimum distance to supermarkets and restaurants), (3) urban form (road length, road density, and land use), (4) urban vitality (population, economic activity diversity, and land use diversity), and (5) subjective perception attributes (e.g., safety, beauty, liveliness, wealth, and quietness). All indicators are aligned at grid level to enable standardized evaluation. Beyond static prediction, UrbanWell defines temporal reasoning tasks, including future value forecasting from historical observations and temporal trend classification. We benchmark 15 state-of-the-art representative MLLMs in a zero-shot setting, providing a comprehensive comparative evaluation across spatial and temporal dimensions. Experimental results indicate that while MLLMs capture salient spatial and perceptual cues, their performance varies substantially across heterogeneous urban indicators spanning environment and subjective perception. UrbanWell serves as a unified benchmark for evaluating multimodal spatial and temporal reasoning in urban wellbeing analytics, offering a standardized testbed for systematic assessment and future research on multimodal urban intelligence. Our codes and datasets are accessible via this https URL. 

---
# LLM-as-Code Agentic Programming for Agent Harness 

**Authors**: Junjia Qi, Zichuan Fu, Jingtong Gao, Wenlin Zhang, Hanyu Yan, Xian Wu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15874)  

**Abstract**: Every major LLM agent framework gives the LLM the role of orchestrator; the model decides what to do next, when to call tools, and when to stop. We argue that token explosion, control-flow hallucination, and unreliable completion are not implementation bugs but architectural consequences of assigning the deterministic work of looping, branching, and sequencing to a probabilistic system. A better prompt or a stronger model cannot guarantee the reliability of the LLM agent. We therefore propose Agentic Programming, in which the program governs all control flow, and the LLM is itself part of it, an adaptive component we call LLM-as-Code and invoke only where a task calls for reasoning or generation. Within each call the model keeps full flexibility, but it cannot alter the program's execution path. With control in the program, the LLM's context is built from the execution history's call tree and forms a directed acyclic graph (DAG). Each call's context length is then determined by its call depth rather than by accumulation over steps. A case study of computer-use agents shows that the design is practical, not just a theoretical stance, substantially improving the stability of long visual operation sequences. 

---
# STRIDE: Strategic Trajectory Reasoning via Discriminative Estimation for Verifiable Reinforcement Learning 

**Authors**: Qinjian Zhao, Zhihao Dou, Dinggen Zhang, Xiangyu Li, Chaoda Song, Zhongwei Wan, Xinpeng Li, Yanyan Zhang, Kaijie Chen, Qingtao Pan, Chengcheng Feng, Zhiqiang Gao, Xiaoyu Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.15866)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become an effective post-training paradigm for improving the reasoning abilities of large language models. However, existing RLVR methods typically rely on final-answer correctness to assign trajectory-level rewards, providing sparse supervision and treating all tokens uniformly regardless of their actual contribution to reasoning. Although recent studies introduce intermediate signals such as process rewards, high-entropy tokens, and semantic uncertainty, these signals are often not inherently verifiable and may fail to distinguish beneficial strategic patterns from harmful ones. To address this limitation, we propose STRIDE (Strategic Trajectory Reasoning with Discriminative Estimation), a fine-grained RLVR framework that derives strategic reasoning supervision from verifiable outcomes. STRIDE contrasts successful and failed trajectories within each response group to estimate the outcome-discriminative preference of each $n$-gram strategic pattern, and further combines this signal with reasoning saliency entropy to identify decision-relevant strategic patterns. These patterns are assigned differentiated advantage values during RL optimization, enabling more precise credit assignment while preserving the verifiability of RLVR. Extensive experiments demonstrate that STRIDE consistently improves reasoning performance across diverse models, tasks, and extended settings, including VLMs and agent-based systems. 

---
# RetailBench: Benchmarking long horizon reasoning and coherent decision making of LLM agents in realistic retail environments 

**Authors**: Linghua Zhang, Jun Wang, Jingtong Wu, Zhisong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15862)  

**Abstract**: Large language model (LLM) agents have made rapid progress on short-horizon, well-scoped tasks, yet their ability to sustain coherent decisions in dynamic long-horizon environments remains uncertain. We introduce RetailBench, a data-grounded simulation benchmark for evaluating tool-using LLM agents in single-store supermarket operation. RetailBench models retail management as a partially observable decision process and is designed to support thousand-day-scale simulations. In this environment, agents must manage pricing, replenishment, supplier selection, shelf assortment, inventory aging, customer feedback, external events, and cash-flow constraints. We evaluate seven contemporary LLMs under representative agent frameworks over a 180-day evaluation horizon and compare them with a privileged oracle policy. Results show substantial variation across models: only a small subset survives the full evaluation horizon, and even the strongest LLM runs remain substantially behind the oracle policy in final net worth and sales outcomes. Behavioral analysis attributes these gaps to incomplete evidence acquisition, surface-level decision making, and the lack of a consistent long-horizon policy. RetailBench provides a controlled testbed for studying reliable autonomy in economically grounded long-horizon decision-making. 

---
# Heteroskedastic Signals in Budgeted LLM Verification: Structural Heterogeneity Limits Optimization Gains 

**Authors**: Jinlong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15841)  

**Abstract**: Large language model (LLM) systems increasingly use uncertainty signals to allocate limited computation across verification, test-time scaling, tool execution, and other selective-compute decisions. Such policies rely on a \emph{global signal comparability assumption}: equal scores should carry comparable decision value across inputs. Using budgeted verification as a controlled diagnostic setting, we identify a failure mode of this assumption: uncertainty quality is heteroskedastic across cost strata, with some regions exhibiting near-random discriminability despite concentrating many errors. Under an explicit local model, we characterize the resulting distortion of global allocation and show that its upper bound scales with cross-stratum signal-quality dispersion. We separate weak signals, optimization instability, and structural heterogeneity through a controlled intervention hierarchy: Threshold, MP-Adapt, MP-Strat, and a deliberately simple cost-stratified thresholding intervention (CST). Across MBPP and MATH using Qwen3-8B, LLaMA3-8B, and GPT-4o-mini, global online adaptation yields inconsistent gains over static thresholding; MP-Strat partially recovers performance, while CST improves hit rate by up to 17 percentage points in strongly heterogeneous settings without gradient updates. These results identify structural heterogeneity, rather than optimizer weakness alone, as the primary bottleneck in the observed settings. More broadly, misaligned feedback structure cannot always be repaired by stronger optimization. 

---
# AIChilles: Automatically Uncovering Hidden Weaknesses in AI-Evolved Systems 

**Authors**: Yajie Zhou, Ao Li, Ashwin Silla, Zaoxing Liu, Vyas Sekar  

**Link**: [PDF](https://arxiv.org/pdf/2606.15834)  

**Abstract**: The computer systems community has recently seen growing interest in AI-driven system evolution, where AI agents iteratively rewrite systems. Frameworks such as AdaEvolve and Engram report 12-60% score improvements over human-designed algorithms. While these results are promising, there are practical concerns if these AI-evolved programs can perform worse on unseen workloads and exhibit scalability regressions. Given the speed and scale of AI-generated code, we need automated mechanisms to uncover such identify hidden weaknesses in AI-evolved systems programs. To this end, we develop AIChilles that takes as input a baseline program $P$ and an AI-evolved program $P'$, AIChilles searches for valid workloads where $P'$ regresses relative to $P$ in correctness, runtime, memory usage, or output quality. To tackle the diversity in system applications, weakness types and potential bugs, AIChilles combines deterministic workload-parameter extraction, agent-based constraint inference, differential oracles, and code-frequency coverage to discover diverse failures. Across five system applications and 30 AI-evolved programs, AIChilles finds 49 distinct hidden weaknesses. We also show that explicitly including AIChilles in the AI-driven development lifecycle can mitigate several of these weaknesses. 

---
# An Integrated System for Real-Time Student Assessment and Career Guidance Using Neural Networks in Computing Disciplines 

**Authors**: Sakir Hossain Faruque, Md. Jubair Hossain, Sharun Akter Khushbu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15831)  

**Abstract**: Many undergraduate students in Computer Science (CS) and Software Engineering (SWE) struggle to identify suitable career paths, particularly when their academic performance, abilities, and interests do not fully align. To address this issue, this study proposes an AI-driven Student Assessment and Career Prediction System that integrates a Career Guidance Expert (CGE) system with a Web-Based Student Assessment (WBSA) platform. Within the integrated framework, CGE enhances personalized career recommendations using AI while also assisting students after graduation in identifying suitable jobs, research domains, and higher study opportunities aligned with their skills and interests. The WBSA platform further strengthens interaction between students and faculty through assessments, personalized tasks, mentorship activities, and a secure real-time chat application. The CGE system employs a Multilayer Perceptron (MLP) model trained on real-world academic and extracurricular data collected using the snowball sampling method from the students of universities, achieving a validation accuracy of 94.71% in predicting personalized career paths. A pre-survey was conducted across universities to evaluate the proposed model before deployment. The WBSA system was developed as a modern web application using technologies such as this http URL, this http URL, and PostgreSQL to ensure scalability, responsiveness, and secure data management. The overall system is supported by a secure cloud-based infrastructure, the platform provides reliable performance while assisting graduates to select suitable career path in IT sector. In addition, a post-survey involving both students and faculty was conducted to gather feedback and further improve the overall effectiveness and usability of the system. 

---
# TrustedARI: Towards Trust-Native Agentic Routing Infrastructure for Agentic AI 

**Authors**: Qi Li, Zhenhua Zou, Shuo Li, Mingwei Xu, Zhuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15822)  

**Abstract**: AI agents increasingly access external models, tools, and services through Agentic Routing Infrastructure (ARI) to manage the overhead of heterogeneous interfaces and fragmented subscriptions. Yet, the architecture of ARI introduces fundamental trust risks: it obtains plaintext access to agent queries and service responses, while leaving agents unable to verify that their queries are routed to intended service providers or that requests and responses remain untampered. To address this problem, we present TrustedARI, the first trust-native agentic routing infrastructure for agentic AI. Architecturally, TrustedARI is built upon three core innovations: (i) an ARI-adapted three-party TLS handshake that enables the agent and ARI to jointly authenticate the service provider through role-specific distribution of TLS key materials; (ii) a privacy-preserving query-construction protocol that allows the agent and ARI to collaboratively construct well-formed queries without exposing their respective private inputs; and (iii) a verifiable billing protocol that supports fair usage-based settlement while preserving the integrity and confidentiality of service responses.
We implemented and extensively evaluated a prototype of TrustedARI to validate its performance. Experiments confirm that TrustedARI is highly efficient: our ARI-adapted handshake protocol reduces communication overhead by 39.34% compared to the existing three-party TLS handshake. Furthermore, the privacy-preserving query-construction protocol imposes negligible overhead-averaging 0.19 seconds in computation time and 0.58 MB in communication costs-while the verifiable billing protocol speeds up proof generation by 28.20x. Crucially, TrustedARI is readily deployable without any modification to the service providers. 

---
# Unassigned Agents in Compilation-based Multi-agent Path Finding 

**Authors**: Pavel Surynek  

**Link**: [PDF](https://arxiv.org/pdf/2606.15797)  

**Abstract**: Compilation-based techniques represent an important stream of solvers for multi-agent path finding (MAPF) due to their modularity and adaptability for non-standard variants of the problem. While in the standard MAPF the task is to navigate all agents from their initial positions to given individual goal positions without any collision, variants where a different requirement for agents is used are also relevant. Such a variant is MAPF with unassigned agents (UA-MAPF) where some agents have the same setting as in the standard MAPF with initial positions and goals while the remaining agents have the initial position but have no goal - unassigned agents. Despite unassigned agent do not need to reach any goal position they have to be moved out of the way of the standard agents if needed which represent a specific challenge. We show in this paper that UA-MAPF can be expressed in recent compilation-based techniques for MAPF based on formulating the problem as Boolean satisfiability, namely we adapt SMT-CBS and NRF-SAT, the recent solvers based on counterexample guided abstraction refinement and non-refined abstractions. 

---
# Mitigating Visual Hallucinations in Multimodal Systems through Retrieval-Augmented Reliability-Aware Inference 

**Authors**: Pratheswaran Hariharan, Haiping Xu, Donghui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15782)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong capabilities in vision-language understanding and natural-language response generation. However, these systems can still produce overconfident predictions and hallucination-like outputs, particularly when the visual evidence is weak, ambiguous, or semantically inconsistent. Most existing approaches focus on improving multimodal representation alignment or retrieval-augmented generation, while providing limited mechanisms to quantify instance-level prediction reliability or identify incorrect visual outputs. This work proposes a retrieval-augmented reliability-aware inference framework for trustworthy multimodal visual understanding. The proposed framework constructs an external visual evidence database using pretrained visual embeddings and nearest-neighbor retrieval over normalized feature representations. Retrieved evidence is used to estimate prediction trustworthiness through multiple reliability indicators, including similarity strength, class-support agreement, evidence margin, entropy-based uncertainty, and an aggregate reliability score. Based on these signals, a decision gate determines whether the system should accept the prediction, answer with caution, or abstain/fallback when evidence is insufficient. A multimodal response-generation layer then produces a final user-facing response conditioned on the reliability decision. Experiments on ImageNet-100 demonstrate that the proposed reliability-aware framework improves accepted prediction accuracy from 85.84\% to 88.88\% at 89.04\% coverage. The hallucination-like accepted wrong-answer rate is reduced from 14.16\% to 11.12\%. These results show that integrating retrieval evidence, reliability estimation, and selective decision gating can improve calibration and reduce overconfident visual errors without retraining large multimodal models. 

---
# Rethinking Scaffolding in LLM Tutors: The Interactional Mismatch Between Benchmarks and Real-World Deployments 

**Authors**: Alexandra Neagu, Jeffrey T. H. Wong, Marcus Messer, Rhodri Nelson, Peter B. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2606.15766)  

**Abstract**: A central pedagogical value evaluated in AI tutor benchmarks is scaffolding: guiding students through graduated steps toward a solution. Alignment and evaluation methods for embedding scaffolding behaviour into chatbots, however, rest on an implicit assumption: that students will take up the scaffolding and engage in the conversation. To examine whether this assumption holds, we introduce an evaluation pipeline around two metrics - Chatbot Scaffolding and Student Uptake - and apply them across nine datasets of 9,490 chats, spanning AI tutor benchmarks and real-world deployments of educational chatbots. Our analysis reveals that while benchmarks assume a high-scaffolding, high-student-uptake environment, students in real-world settings exhibit lower levels of uptake overall - frequently bypassing the chatbot's pedagogical framing to drive the interaction toward their own learning goals at little interpersonal cost. We argue that bypassing scaffolding is not necessarily detrimental; rather, it frequently highlights a mismatch between a chatbot's pedagogical framing and the student's learning goals. To meaningfully evaluate the effectiveness of a chatbot's assistance, future benchmarks must move beyond the assumption that students will simply take up the scaffolding, and instead evaluate how these chatbots navigate diverse learning contexts and student-driven interaction patterns. 

---
# RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought 

**Authors**: Yaoting Huang, Yifu Yuan, Linqi Han, Chengwen Li, Shuoheng Zhang, Xianze Yao, Hongyao Tang, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15753)  

**Abstract**: Embodied reasoning requires models to perceive task-relevant objects and spaces in physical environments and maintain consistent visual grounding throughout multi-step reasoning. However, current vision-language models rely on text-only or coordinate-augmented chain-of-thought, where entity references remain implicit and ambiguous. This may cause the reasoning process to decouple from visual evidence, entity references to drift across steps, and a causal disconnection between the reasoning trajectory and the final answer, with these problems further amplified in multi-view scenarios due to cross-view appearance changes. To address these issues, we propose Pinned Chain-of-Thought (\pincot{}), a structured reasoning paradigm that pins every reasoning step to visual evidence. \pincot{} introduces the concept of \reasoninganchor{}, which binds each task-relevant entity to a structured visual anchor with entity name, unique identity, view index, and spatial grounding, enabling consistent entity tracking across reasoning steps and views. We build a fully automated data generation pipeline to construct \dataset{}, a high-quality \pincot{}-formatted reasoning dataset. We then train \method{} through three-stage post-training that progressively injects embodied knowledge, structured reasoning ability, and process-supervised alignment, with rewards that directly constrain both anchor localization and identity consistency during reasoning. On 14 benchmarks covering embodied spatial reasoning, multi-view reasoning, and pointing, \method{} with only 4B parameters consistently outperforms 7B level open-source embodied models, achieving a 12\% average improvement over the strongest 7B baseline, Mimo-Embodied. Further analysis shows that \pincot{} improves grounding accuracy and cross-step identity consistency, validating the effectiveness of process supervision. 

---
# AI-Driven Framework for Adaptive Water Network Management with Proof-of-Concept Implementation: Addressing Non-Revenue Water in Jordan 

**Authors**: Mohammed Fasha, Nahel Al-Maayta, Bilal Sowan, Mohammad Athamneh, Husam Barham  

**Link**: [PDF](https://arxiv.org/pdf/2606.15709)  

**Abstract**: Jordan faces severe water scarcity with 50\% of water produced is lost to leakage, theft and metering issues also known as non-revenue water (NRW). Traditional reactive approaches have proven insufficient for sustained NRW reduction. This paper proposes an intelligent framework integrating EPANET hydraulic modeling, digital twin technology, SCADA systems, and large language model (LLM)-based AI agents for continuous network monitoring and adaptive decision-making. The system combines real-time data streams with physics-based simulation to detect anomalies, employing retrieval-augmented generation (RAG) for policy interpretation and function calling for network control. A proof-of-concept implementation validates technical feasibility using EPYT with offline LLMs (llama3.1:8b via Ollama) on a 1,164-junction Amman district network. The system demonstrates automated hydraulic simulation, flow-based anomaly detection aligned with water distribution zone (DZ) practice, and AI-generated health reports with response times under 2 minutes and zero API costs. Burst detection relies on local flow anomaly analysis: a 30.1~L/s simulated leak produces measurable flow redistribution in 15 pipes, flagging a 15-junction cluster that localises the burst -- confirming alignment with water distribution zone (DZ) monitoring practice. The framework accommodates Jordan's intermittent supply patterns and limited automation through phased implementation, offering a scalable pathway for water-scarce regions to leverage intelligent automation for NRW reduction and operational efficiency. 

---
# Artificial Intelligence Index Report 2026 

**Authors**: Sha Sajadieh, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Lapo Santarlasci, Juan Pava, Nestor Maslej, Russ Altman, Erik Brynjolfsson, Carla Brodley, Jack Clark, Virginia Dignum, Vipin Kumar, James Landay, Terah Lyons, James Manyika, Juan Carlos Niebles, Yoav Shoham, Elham Tabassi, Russell Wald, Toby Walsh, Dan Weld  

**Link**: [PDF](https://arxiv.org/pdf/2606.15708)  

**Abstract**: Welcome to the ninth edition of the AI Index report. As AI continues to advance rapidly, the question becomes whether the systems built around it can keep up. Governance frameworks, evaluation methods, education systems, and the data infrastructure needed to track AI's impact are struggling to match the pace of the technology itself. That gap between what AI can do and how prepared we are to manage it runs through every chapter of this year's report. New in this edition, the report tracks how AI is being tested more ambitiously across reasoning, safety, and real-world task execution, and why those measurements are increasingly difficult to rely on. It also features new estimates of generative AI's economic value alongside emerging evidence of its labor market effects, an analytical framework on AI sovereignty, and a science chapter developed in collaboration with Schmidt Sciences. For the first time, the report features standalone chapters on AI in science and AI in medicine, reflecting AI's growing impact across these two domains. 

---
# Do LLMs Reliably Identify Correct Information Units in Aphasic Discourse? 

**Authors**: Jason M Pittman, Yesenia Medina-Santos, Anton Phillips Jr., Brielle C. Stark  

**Link**: [PDF](https://arxiv.org/pdf/2606.15696)  

**Abstract**: Correct Information Units (CIUs) are central to discourse assessment in aphasia because they quantify communicative informativeness rather than linguistic form alone. However, CIU scoring is time intensive and requires trained raters. This study examined whether instruction-tuned large language models (LLMs) can reliably perform token-level CIU classification from aphasic discourse transcripts. Sixteen picture-description transcripts elicited with the Cat Rescue stimulus were annotated for CIU status according to Nicholas and Brookshire (1993). The sample spanned four severity strata: control, mild, moderate, and severe aphasia. Four publicly available instruction-tuned LLMs were benchmarked under zero-shot and two few-shot prompting conditions across five stratified random seeds. Performance was evaluated against consensus human labels using accuracy, precision, recall, F1, and Cohen's kappa. Zero-shot prompting was insufficient across models. In contrast, few-shot prompting yielded substantial gains and produced competitive performance for three viable models. Mean few-shot F1 scores ranged from 0.776 to 0.817 across Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B, with no significant differences between fixed global and per-chunk local example selection. Phi-3-mini was unstable and did not yield reliable performance. Viable models showed high recall but lower precision, suggesting systematic over-classification of tokens as CIUs. Performance also varied by discourse severity, with the weakest results in more severe aphasia. Few-shot LLM prompting can support automated CIU identification without gradient-based task training, but agreement with human annotation remains insufficient for fully autonomous use. These findings support LLM-based CIU scoring as a promising human-in-the-loop component of discourse assessment systems. 

---
# Recurrent Reasoning on Symbolic Puzzles with Sequence Models 

**Authors**: Gowrav Mannem, Chowdhury Marzia Mahjabin, Jason Chen, Shivank Garg, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15686)  

**Abstract**: Large language models often appear strong on symbolic and algorithmic tasks, yet this apparent strength can hide brittle behaviour when problems become longer, harder, or slightly out of distribution. A major limitation of current reasoning benchmarks is that many primarily test whether a model can produce a valid answer, while paying less attention to whether the solution is minimal, robust, and stable under controlled difficulty scaling. We introduce RecurrReason, a difficulty-controlled benchmark of four recurrent logic puzzles (Tower of Hanoi, River Crossing, Block World, and Checkers Jumping) with BFS-optimal trajectories and a single interpretable difficulty parameter $N \in \{1,\dots,10\}$, totalling 10{,}817 unique puzzles and 285{,}933 moves. We benchmark two Transformer families, an encoder-decoder model (T5-style) and a decoder-only model (GPT-2-style), under consistent data splits and evaluation criteria, training on $N{=}1$ to $7$ and evaluating on both held-out in-distribution instances and harder out-of-distribution instances at $N{=}8$ to $10$. Fine-tuned pre-trained T5 achieves 97.27\% validation and 81.00\% OOD accuracy on Block World; all models score 0.00\% on River Crossing under all conditions. Failure mode analysis reveals that architecture is a stronger determinant of success than scale. Pre-training transfers only to puzzles with locally structured transition functions. Our code and dataset will be open-sourced upon acceptance. 

---
# Multi-agent Framework for Time-Sensitive Complementary Collaboration in Minecraft 

**Authors**: Juheon Yi, Jinglu Wang, Xiaoyi Zhang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15684)  

**Abstract**: We present TickingCollabBench, a Minecraft-based multi-agent benchmark for a novel class of time-sensitive complementary collaboration tasks. Our benchmark reflects four core characteristics of real-world collaboration: agent heterogeneity, mandatory collaboration, dynamic environments, and strict real-time constraints with failure risks. To enable this, we develop the TickingCollab framework, which supports the generation of diverse dynamic environments and abstracts Minecraft's primitive APIs to enable declarative YAML task specifications for composing these events. Building on this, we design a feasibility-aware automated benchmark generation pipeline, where an LLM drafts structurally diverse task configurations and feasibility verifier filters out invalid ones using approximate constraints. Evaluations demonstrate that lang latency and inherent difficulty of coordinating under partial observability and agent heterogeneity cause LLMs to frequently fail under dynamic environments and fall significantly short of a global-knowledge oracle. 

---
# Where Did It Go Wrong? Process-Level Evaluation of Web Agents with Semantic State Tracking 

**Authors**: Jiwan Chung, JiHyuk Byun, Vibhav Vineet, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.15673)  

**Abstract**: Web agents act through long interaction sequences, yet existing benchmarks evaluate only terminal success, discarding all process information and offering little guidance on improvement. In this work, we conduct a process-level analysis of web agents. We introduce WebStep, a benchmark of 1,800 task instances with controlled difficulty and automatic semantic state tracking. Each website exposes a deterministic semantic MDP alongside the GUI: the agent operates on the interface, while the environment records high-level states and transitions in the background, enabling fine-grained analysis without manual annotation. Based on the semantic trajectory, we first show that process metrics reveal differences invisible to outcome evaluation: three agents whose success rates cluster within 31-33% diverge in exploration reach versus execution accuracy. Then, decomposing by skill characterizes the nature of these differences, exposing opposite per-skill rankings hidden within the same website: e.g., on Housing, OpenAI CUA outperforms Qwen3.5 by 23.7% on commit actions yet underperforms it by 15.6% on filtering, pinpointing a concrete skill to improve even within a domain. Bifurcation analysis further localizes the decisive error that loses the task and shows that this error is agent-specific rather than shared. Finally, these differences widen as tasks grow harder: success rate is similar on easy tasks but separates sharply as exploration becomes more demanding. Our process-level analysis opens a new avenue in web agent evaluation, providing fine-grained and actionable insight into where and how each agent should be improved. 

---
# Overcoming the Impedance Mismatch: A Theoretical Roadmap for Fusing Foundation Models and Knowledge Graphs 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2606.15656)  

**Abstract**: Modern artificial intelligence remains fundamentally divided between the continuous, probabilistic spaces of Foundation Models and the discrete, deterministic structures of Knowledge Graphs. While Retrieval-Augmented Generation (RAG) attempts to connect them by serializing graph data into text, we argue this lexical bridging is merely a superficial patch. In this paper, we formalize the underlying structural and geometric friction as the \textit{Impedance Mismatch}. By categorizing current neuro-symbolic integration strategies into a three-tiered hierarchy, we demonstrate that neither surface-level prompt injection nor continuous representation alignment can preserve the strict logical motifs required for reliable multi-hop reasoning. We define the specific mathematical limits, such as the Lexical Bottleneck and Topological Collapse, that show current architectures will eventually hallucinate or conflate semantic nodes. To achieve true semantic fusion, we propose a rigorous theoretical roadmap. We advocate for natively internalizing discrete symbolic structures through Structured Residual Streams, utilizing Vector Symbolic Architectures for latent sub-graph injection, and performing model updates via Orthogonal Subspace Editing. This actionable framework paves the way for models that seamlessly fuse the precision of symbolic logic with the expressivity of parametric memory. 

---
# Advanced Machine Learning and Deep Learning Techniques for Enhanced Cattle Identification and Detection: A Comprehensive Review 

**Authors**: Fayazunnesa Chowdhury, Syed Md. Galib, Md Nasim Adnan, Md. Moradul Siddique, Md Robiul Karim, K M Tanvir Anjum  

**Link**: [PDF](https://arxiv.org/pdf/2606.15655)  

**Abstract**: The need for effective cattle identification technology is now more acutely felt than ever in maintaining biosecurity, food safety, and supply chain efficacy in livestock management. This paper presents a systematic review of recent research in cattle identification using machine learning and deep learning techniques. The present systematic review measures the effectiveness of traditional and modern cattle identification techniques using studies from major academic databases, where articles were subjected to full-text review. Among these techniques, classical Machine Learning Techniques such as K-Nearest Neighbors and Support Vector Machines have demonstrated good results in cattle identification; however, Deep Learning Techniques, such as Convolutional Neural Networks, Residual Networks, and You Only Look Once, are better in cognition, detection, and identification tasks. Feature extraction relies on common techniques like Local Binary Pattern (LBP), Speeded-Up Robust Features (SURF), and Scale-Invariant Feature Transform (SIFT), while key features commonly used in these studies include muzzle prints and coat patterns. The review highlights key hurdles involving cattle identification, such as the limited number of publicly accessible datasets, issues with data quality susceptible to environmental changes and animal mobility, and high demand for real-time processing ability. The paper aims to inform researchers, policymakers, and stakeholders about implementing scalable, humane, and effective cattle identification systems to achieve sustainable livestock management. 

---
# Towards Next-Generation Healthcare: A Survey of Medical Embodied AI for Perception, Decision-Making, and Action 

**Authors**: Cheng Zhang, Qing Cai, Xingzheng Wu, Xun Yang, Xiaojun Chang, Bingkun Bao, Liqiang Nie, Xinwang Liu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15647)  

**Abstract**: Foundation models have demonstrated impressive performance in enhancing healthcare efficiency across a wide range of medical applications. Nevertheless, their limited ability to perceive, understand, and interact with the physical world significantly constrains their effectiveness in real-world clinical workflows, where safety-critical decision-making and physical execution are tightly coupled. Recently, embodied artificial intelligence (AI) has emerged as a promising physical-interactive paradigm for intelligent healthcare, enabling agents to operate in complex medical environments. As research in this area rapidly expands, understanding how intelligent agents function as integrated, end-to-end systems in clinical environments becomes increasingly critical. However, existing surveys on medical embodied AI largely emphasize individual aspects or functional components, lacking a unified system-level organization of the field. To support and consolidate recent advances, we systematically survey the core components of medical embodied AI, with a particular emphasis on the coordinated integration of perception, decision-making, and action. We further review representative medical applications and relevant datasets, and we analyze the major challenges encountered in real-world clinical practice. Finally, we discuss key directions for future research in this rapidly evolving field. The associated project can be found at this https URL. 

---
# NeuroSymbolic AI for Legal AI-TRISM: Trustworthy, Reliable, Interpretable, Safe Models 

**Authors**: Deepa Tilwani, Yash Saxena, Ankur Padia, Srinivasan Parthasarathy, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2606.15646)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, but their lack of interpretable reasoning and tendency to hallucinate pose significant challenges for legal applications. While LLMs show promise for legal text analysis and generation, they struggle with accurate citation attribution and precedent verification. For example, in legal contexts, a single incorrect precedent can jeopardize a case. Current approaches to improve LLM reliability in legal domains suffer from two key limitations: inadequate integration of structured legal knowledge during training or fine-tuning, and insufficient verification mechanisms for generated legal content. To address these challenges, we propose the TRISM (Trustworthy, Reliable, Interpretable, Safe Models) framework, which integrates NeuroSymbolic AI principles with LLMs to leverage both neural learning capabilities and symbolic reasoning over structured legal knowledge. The TRISM approach addresses the above limitations while maintaining interpretable decision pathways. Our framework formalizes the extraction of symbolic knowledge from legal textual documents and incorporates Retrieval-Augmented Generation (RAG) as a core component for grounding LLM outputs in verified legal sources. In this position paper, we make the following contributions: (1) An analysis of the limitations of AI in law; (2) Introduce RASOR RAG which creates foundations for neurosymbolic RAG by generating explicit interpretable rationales that could be formalized into symbolic representations; (3) A formalized methodology for creating symbolic legal knowledge bases that support both interpretable reasoning and output verification in LLMs; and (4) The TRISM framework for integrating symbolic legal knowledge with LLMs. 

---
# Integrating Reasoning and Generalization in Text-to-SQL via Self-Enhanced Fine-Tuning 

**Authors**: Feng Lyu, Jinfeng Cen, Sijing Duan, Hao Wu, Shucheng Li, Weixu Zhang, Haolun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15598)  

**Abstract**: Text-to-SQL aims to translate natural language questions into executable SQL queries over structured databases, enabling non-expert users to access data intuitively. While recent advances in large language models (LLMs) have shown promise in this task, existing LLM-based approaches often struggle to strike a balance between strong reasoning capabilities and robust generalization. To address these limitations, we propose CoTE-SQL to enhance the LLM-based text-to-SQL generation with three key innovations: (i) self-enhanced reasoning traces distilled from LLMs without human annotation, (ii) structured chain-of-thought (CoT) prompting with modular decomposition and examples retrieval, and (iii) error-aware revision based on SQL execution feedback. Extensive experiments on the Spider and Bird benchmarks demonstrate that CoTE-SQL achieves new state-of-the-art performance among methods built on open-source LLMs with comparable model sizes on Bird (53.39% EX / 59.02 VES) and strong results on Spider (79.60% EX / 77.19 VES), with especially significant gains on complex queries. Results highlight the effectiveness of combining self-enhancement, structured reasoning, and execution-time feedback within an LLM-based framework for text-to-SQL design. 

---
# Agentic Retrieval and Reinforcement Learned Equation Chains: A Controlled Generation Framework for Complex and Novel Physics Word Problems 

**Authors**: Tirthankar Mittra  

**Link**: [PDF](https://arxiv.org/pdf/2606.15591)  

**Abstract**: Generating high-quality Physics Word Problems (PWPs) that are novel, complex, and solvable remains a challenging and underexplored problem in educational content generation. Existing approaches, many adapted from Math Word Problem (MWP) generation, often produce ambiguous, unsolvable, or structurally simple questions with limited linguistic diversity. We introduce ARVRE (Agentic Retrieval Value Reinforced Equation-chain), a two-stage framework for generating diverse and mathematically valid PWPs. In the first stage, a form of offline temporal-difference learning is used to construct valid chains of physics equations, while an agentic retrieval-augmented generation (RAG) framework dynamically selects topic-specific concepts and vocabulary. This design enables explicit control over problem structure and difficulty. In the second stage, a Large Language Model (LLM) converts the equation chain and retrieved concepts into a natural-language physics question. By grounding generation in valid equation chains, our method preserves mathematical correctness while promoting linguistic diversity and contextual richness. Human and automated evaluations demonstrate that ARVRE generates PWPs that are more complex, novel, and solvable than those produced by existing approaches. These results highlight the potential of combining reinforcement learning, retrieval, and LLMs for reliable generation of educational physics content. 

---
# Your Agent Has a Genome: Sequence-Level Behavioral Analysis and Runtime Governance of LLM-Powered Autonomous Agents 

**Authors**: Sidi Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.15579)  

**Abstract**: We propose Base Sequence Analysis, a framework that encodes the runtime behavior of LLM-powered autonomous agents into compact symbolic sequences using a four-letter alphabet: X (Explore), E (Execute), P (Plan), and V (Verify). Drawing an analogy to genomic sequence analysis, we apply n-gram pattern mining, Markov transition matrices, and point-biserial correlation to 347 real-world execution traces collected from a production ReAct agent system over 8 days. Our analysis reveals that (1) the trigram P-X-P is the only statistically significant high-risk pattern, lowering success rate by 10.4%; (2) P-ratio is the strongest negative predictor of success (r=-0.256, p<0.0001); and (3) the E->V transition probability is only 2.1%, indicating a systemic verification deficit. Based on these findings, we design Governor, a three-layer runtime intervention system comprising a rule engine, a statistical accumulator, and a chi-square-based threshold adaptor. In a natural before/after deployment evaluation (N=101 vs. N=246), Governor achieves a +6.2% absolute increase in task success rate while simultaneously reducing average token consumption by 44%. To validate cross-system generality, we apply the XEPV encoding to 2,000 public SWE-agent trajectories on SWE-bench, confirming that exploration spirals and the E->V verification deficit replicate in an independent system. We outline six research directions including base sequence language models, cross-agent behavioral fingerprinting, and reward shaping, and release an open-source toolkit for reproducibility. 

---
# Large Language Models as Optimizers: A Survey of Direct vs. Tool-Augmented Approaches and Their Performance Frontiers 

**Authors**: Roko Peran, Luka Hobor, Mihael Kovac, Mario Brcic  

**Link**: [PDF](https://arxiv.org/pdf/2606.15577)  

**Abstract**: Large Language Models (LLMs) are increasingly involved in complex mathematical optimization, even if the pragmatic user who triggers them is unaware of it. After all, many real-world problems reduce to the search for better or the best solutions. The field of LLM-as-optimizer has three paradigms: direct optimization, tool-augmented optimization, and tool-creating optimization. Direct optimization uses iterative prompting and heuristic generation to navigate solution spaces. Tool-augmented optimization translates natural language problems into formal specifications and orchestrates external solvers. Tool-creating optimization goes further, using LLMs to discover reusable algorithms or heuristics that can be deployed at zero marginal LLM cost. We describe current performance frontiers based on the benchmarks from the literature. We identify the critical reasoning gap in current architectures and argue for trade-offs between the future potential of direct optimization and the auditability of tool-augmented optimization. Even future, more powerful models might opt for tool-making to improve operational efficiency for repetitive families of problems. 

---
# Do we have the knowledge we need? Rethinking human-AI decision-making in corporations 

**Authors**: Anne S. R. Marx, Ricardo M. Avelino, Torbjørn Netland, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2606.15575)  

**Abstract**: Organizational knowledge is fragmented across a variety of software systems, tacit expertise, and manual documents that have traditionally been designed for human consumption. As AI systems are increasingly deployed and granted decision-making roles, they require access to this knowledge. This raises two questions: how should organizations store and maintain knowledge so that it remains accessible to both humans and future AI systems, and how should agency be allocated between humans and AI across tasks with different risks and levels of uncertainty? In this position paper, we describe how organizational knowledge evolves and contribute a framework that maps task attributes and knowledge availability to recommended agency allocations and control mechanisms. We illustrate the applicability of the framework on two different manufacturing tasks: a routine operation (visual quality inspection) and a one-off strategic decision (factory location), and conclude with opportunities for future research. 

---
# QoS-Aware Token Scheduling and Private Data Valuation for Multi-Modal Agentic Networks 

**Authors**: Yao Du, Jing Liu, Pengfei Xu, Zehua Wang, Victor C.M. Leung, Cyril Leung, Victoria Lemieux  

**Link**: [PDF](https://arxiv.org/pdf/2606.15573)  

**Abstract**: In agentic systems, human-generated data records anchor the value of AI services. Yet cloud compute pipelines centralize processing on remote servers. Data centralization reduces personal data sovereignty and may potentially degrade the quality of service (QoS). Meanwhile, user contributions are diverse in quantity and quality: decentralized records can be biased, noisy, and heterogeneously distributed. To address the data challenge, we study fair token allocation and private data valuation for decentralized and resource-constrained agentic systems. Our approach embeds multi-modal representations in a shared semantic space and releases differentially private (DP) prototypes to preserve utility while reducing semantic leakage. With the DP guarantee, we design a fair token allocation scheme that rewards effective contributions and remains robust to data heterogeneity and AI resource scarcity. Extensive simulations demonstrate improved contribution-based fairness and QoS compared to standard benchmarks. The improved resistance to image reconstruction attacks indicates enhanced privacy for multi-modal personal data. 

---
# Minimal Oversight: Uncertainty-Aware Governance for Delegated AI Systems 

**Authors**: Carlos R. B. Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2606.15563)  

**Abstract**: AI systems increasingly delegate decisions to specialized models, evaluators, tools, and supervisory controllers. The central AI problem is no longer only model accuracy, but uncertainty-aware governance: how much autonomy to grant, which evidence should calibrate trust, what performance ceiling a delegated AI system can sustain, and when human intervention becomes necessary. We propose the Minimum Sufficient Oversight Principle (MSO), a variational principle for principled autonomy delegation: minimize governance burden on the Fisher information manifold subject to a delivery constraint. The resulting Euler-Lagrange solution yields a water-filling allocation of governed delegation across the task space. Building on a revealed-action governed delegation channel model, we prove a capacity theorem for stationary symbolwise review policies, derive a local first-order approximation relating workflow complexity to quality degradation, and give a drift-dominated autonomy-time scaling law linking intervention timing to effective capacity, complexity, and drift. Within this framework, masking appears as a structural AI-governance pathology: corrected performance can hide the competence signal needed to calibrate trust. Synthetic simulations and a semi-real reconstructed workflow support design prescriptions including upstream-first correction, sensitivity-based intervention, and explicit feasibility checks before autonomy is expanded. The result is a computable framework for uncertainty, planning, and oversight in delegated AI systems. A companion Python package is available at this https URL. 

---
# ToolMenuBench: Benchmarking Tool-Menu Filtering Strategies for Reliable and Efficient LLM Agents 

**Authors**: Rahul Suresh Babu, Laxmipriya Ganesh Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.15508)  

**Abstract**: Tool-augmented large language model agents increasingly operate over large tool libraries, but existing evaluations often focus on whether a model can call a tool correctly rather than how the visible tool menu shapes reliability, efficiency, and safety-relevant risk exposure. We introduce ToolMenuBench, a benchmark for evaluating tool-menu construction in multi-step LLM agents. ToolMenuBench varies tool-menu size, distractor type, state-dependent task structure, and risk exposure, and reports both filter-level and downstream agent metrics, including visible-tool count, risky-tool exposure, task success, wrong-tool calls, premature actions, and token usage. In a controlled evaluation across seven model backends, three tool-menu sizes, six filtering methods, and seven evaluation settings, CMTF improves task success from 32.1% under all-tools exposure to 85.7%, while reducing average token usage by roughly 98%. Causal minimal tool filtering achieves the strongest overall tradeoff, reducing visible tools, wrong-tool calls, premature actions, and risky-tool exposure relative to unfiltered exposure, lexical filtering, state-aware filtering, and broader causal-path baselines. ToolMenuBench provides a reusable evaluation framework for studying the agent-interface problem: which tools should be visible, when they should be visible, and under what cost or risk constraints. 

---
# Frame-Conditioned Moral Computation in LLaMA 3.1-8B-Instruct: A Mechanistic Interpretability Audit of Ethical Reasoning 

**Authors**: Ali Dasdan, Manan Shah, W. Russell Neuman, Chad Coleman, Kund Meghani, Safinah Ali  

**Link**: [PDF](https://arxiv.org/pdf/2606.15507)  

**Abstract**: Behavioral audits of Large Language Models on moral prompts measure what the model says, not the internal computation producing it. We use Transluce, an AI-driven mechanistic-interpretability platform, to examine LLaMA 3.1-8B-Instruct on 54 moral prompts in four batteries: 17 dilemmas, policy, and meta-ethical questions (B1); 6 role-playing scenarios (B3); and a controlled trolley contrast varying the switching mechanism with people fixed (B4, 15 prompts) or identity attributes with mechanism fixed (B5, 16 prompts).
Two complementary metric families, five cluster-level metrics and a six-metric neuron-level panel, converge on a Situational Anchor Effect: domain-specific representations dominate the top of the activation list across every battery. The model's ethics-labeled capacity stays essentially constant; its salience (rank, priority, top-of-list presence) is highly sensitive to the interpretive frame the prompt selects.
The B4-vs-B5 contrast confirms the model attends to whichever surface feature varies: aggregate ethics metrics are indistinguishable, but the dominant non-ethics distractor mirrors the design. A multi-temperature audit identifies a candidate ethics neuron (L16/N3837) stable across temperatures; a cross-model behavioral proxy on two frontier models yields preliminary evidence of divergence in self-reported moral focus, consistent with an Alignment Wrapper in which RLHF re-orders surface text without removing underlying domain-first frames. We unify these as Frame-Conditioned Moral Computation: the prompt's surface vocabulary selects a feature manifold, and the moral conclusion is downstream of that selection. Behavioral alignment must be supplemented by Mechanistic Alignment: a research program asking whether ethics-related features can be shown causally privileged under controlled frame variation, not merely loud in the explanation. 

---
# Toward Vibe Medicine: A Self-Evolving Multi-Agent Framework for Clinical Decision Support 

**Authors**: Qianxue Zhang, Yiming Ren, Shihuan Qin, Xiao Zhang, Liao Zhang, Jinyang Huang, Zhengliang Liu, Chenbin Liu, Hongying Feng, Jingyuan Chen, Yuzhen Ding, Weihang You, Hanqi Jiang, Yi Pan, Yifan Zhou, Junhao Chen, Lifeng Chen, Wei Liu, Tianming Liu, Zengren Zhao, Lian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15504)  

**Abstract**: In recent years, the advances of large language models and autonomous agents have revolutionized the healthcare field, facilitating diagnosis and improving treatment results. However, most existing AI systems rely on pre-trained knowledge and predefined pipelines, which struggle to learn dynamically from the interactive chat session history that contains patient outcomes and past failures. To address this limitation, we propose VIBEMed, a multi-agent framework with a built-in self-evolution mechanism and architecture-level safety sandbox for robust clinical decision support. The system integrates three specialized agents, including a Clinical Diagnostic Agent (CDA) for hypothesis generation, a Therapeutic Execution Agent (TEA) for treatment planning, and a Clinical Evolution Manager Agent (CEMA) that distills longitudinal clinical feedback into reusable knowledge, transforming multimodal patient information into personalized medical decisions. Through self-evolution mechanism, the framework enables iterative updates across memory, model behavior, and decision strategies, allowing the system to improve over time. Experimental results show that VIBEMed demonstrates superior performance through its evolving mechanism in complex clinical cases, particularly in tasks that require integrated decision-making and longitudinal planning. The framework also supports reliable end-to-end decisions in challenging scenarios such as oncology treatment planning, highlighting its feasibility in real-world clinical contexts. Overall, VIBEMed provides a practical path beyond static AI systems toward adaptive, experience-driven clinical decision support, demonstrating the value of combining multi-agent collaboration with continuous evolution for advancing precision medicine. 

---
# Synthetic Counteradaptation: A Principle of Human-AI Co-evolution 

**Authors**: Ivar Frisch, Jackie Kay, Philip Moreira Tomei  

**Link**: [PDF](https://arxiv.org/pdf/2606.15503)  

**Abstract**: In this paper, we introduce the concept of synthetic counteradaptation, a process where human and AI systems co-evolve by adapting to each other's strategies and behaviors. Synthetic counteradaptation occurs when AI systems develop novel strategies or social protocols, prompting humans to extract insights and adapt their own behaviors in response, leading to the emergence of new agent interaction dynamics. To illustrate these dynamics, we analyze examples from various contexts, including the game of Go, mixed-motive social interactions, and geopolitical simulations. By exploring these cases, we demonstrate how synthetic counteradaptation provides a framework for understanding the recursive and co-evolutionary nature of human-AI interactions in multi-agent environments. 

---
# Towards End-to-End Automation of AI Research 

**Authors**: Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Chris Lu, Shengran Hu, Jakob Foerster, David Ha, Jeff Clune  

**Link**: [PDF](https://arxiv.org/pdf/2606.15497)  

**Abstract**: The automation of science is a long-standing ambition in the field of AI. While the community has made significant progress in automating individual components of the scientific process, a system that autonomously navigates the entire research lifecycle -- from conception to publication -- has remained out of reach. Here, we present the strongest demonstration to date toward automating the entire process end-to-end. We present The AI Scientist, which creates research ideas, writes code, runs experiments, plots and analyzes data, writes the entire scientific manuscript and performs its own peer review. Its ideas, execution, and presentation are of sufficient quality to produce a manuscript generated by an AI system that passes the first round of peer review at a major machine learning conference workshop. The workshop has an acceptance rate of 70 percent. Our system leverages modern foundation models within a complex agentic system. We evaluate The AI Scientist in two settings: a focused mode using human-provided code templates as an initial scaffold to conduct research on a specific topic, and a template-free, open-ended mode that leverages agentic search for wider scientific exploration. Both settings produce diverse ideas and automatically test, report on, and evaluate them. This achievement demonstrates AI's growing capacity for scientific contribution and signifies a potential paradigm shift in how research is conducted. As with any impactful new technology, there could be significant risks, including taxing overwhelmed review systems and adding noise to scientific literature. However, if developed responsibly, such autonomous systems could greatly accelerate scientific discovery. 

---
# Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines 

**Authors**: Yitao Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.15474)  

**Abstract**: Continuous evaluation of LLM products relies on a strong LLM judge treated as ground truth: a cheap monitor scores every interaction and a team is paged when the score drifts down. But the judge is itself a model behind an API, and a silent version bump or scoring-prompt update changes how it scores -- so every drift alarm is ambiguous between a worse product and a changed judge. We resolve the ambiguity with a fixed, human-labeled anchor set that the current judge re-scores at a steady interleave, a second betting e-process on the judge-versus-human gap, and a guard-window rule returning a verdict in {none, system, judge}. We prove anytime-validity, one-way identification (only the judge can move the anchors), an attribution race whose design law is that the anchors must out-run the main process they guard, and process orthogonality. On two real judge changes, a silent version bump is detected as judge drift in 60/60 runs with zero judge-to-system misattribution, and a contaminating strict-prompt change is correctly attributed on 110 of 120 runs at guard width 300 -- while the industry-default rolling z-test false-alarms on 75% of drift-free streams. Every experiment replicates on a second domain (TL;DR summarization) with nothing re-tuned, and where the domains differ the differences are the ones the race predicts: the strict-prompt change shifts scores harder there, so the anchors fire faster and attribution becomes perfect (240/240). The monitor runs at approximately 0.64 of the cost of strong-judging every item, or 0.21 in a cheaper-but-deafer regime. 

---
# Hierarchical Modeling of ICD Codes in EHR Foundation Models 

**Authors**: Megha Thukral, Dong Gyun Kang, Rudra Pratap Singh, Shruthi Kashinath Hiremath, Katrin Hänsel, Thomas Plötz  

**Link**: [PDF](https://arxiv.org/pdf/2606.15447)  

**Abstract**: Electronic health record foundation models typically treat ICD diagnosis codes as flat tokens, overlooking the clinically meaningful hierarchical structure that captures disease families, subcategories, and fine-grained diagnostic detail. As a result, existing EHR representation learning methods do not explicitly exploit the hierarchical structure already present in the coding system. In this work, we study ICD-10-CM hierarchy as a general inductive bias for clinical representation learning. We investigate two complementary mechanisms for incorporating hierarchy: first, by augmenting diagnosis sequences in a BERT-style transformer with tokens corresponding to different levels of the ICD hierarchy, and second, by injecting hierarchy into graph-based code representations through hierarchy-aware edges combined with diagnosis co-occurrence structure. Across these settings, we evaluate whether explicit hierarchy improves downstream prediction, which levels of the hierarchy are most useful, whether hierarchy encoding improves transfer across datasets, and how hierarchy reshapes embedding similarity structure. We conduct experiments on two large-scale real-world clinical datasets: MIMIC-IV, used for pretraining and in-domain evaluation, and eICU, used to assess cross-dataset transfer via frozen encoder probing. Our findings show that explicitly encoding ICD hierarchy improves over flat code representations in both in-domain and cross-dataset settings, while revealing that the most useful level of hierarchy depends on both the task and the modeling approach. More broadly, we focus on hierarchy-aware EHR representation learning and show that the benefits of encoding hierarchy are generalizable across modeling settings and hierarchy levels. 

---
# Reward Hacking in Language Model Agents: Revisiting AI Safety Gridworlds 

**Authors**: Ömer Veysel Çağatan, Xuandong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15385)  

**Abstract**: Reward hacking, where AI systems exploit misspecified objectives to achieve high reward without satisfying intended goals, remains a central challenge in AI safety. Yet most known instances have been discovered post hoc in frontier systems where controlled study is impractical. We adapt the AI Safety Gridworlds framework into a text-based evaluation suite that reformulates classic reinforcement learning safety tasks for language-based agents. Across frontier and mid-scale models, we find that specification gaming emerges zero-shot: models systematically achieve high observed reward while underperforming on hidden safety objectives, and even apparently safe behaviors can reflect misunderstanding rather than principled safety. Reinforcement learning does not correct these failures: direct reward optimization widens the gap between observed and hidden reward, as the model's initial competence causes it to lock into locally rewarding strategies before discovering safer alternatives. This pattern persists across model scales (1.5B--14B) and is not resolved by finer credit assignment, exploration prompts, or entropy regularization. Our results show that reward hacking arises naturally when optimizing proxy objectives with capable language model agents and resists standard mitigations, suggesting that proxy-reward failures in agentic settings may require approaches beyond standard exploration and credit-assignment fixes. To facilitate reproducibility, the code for this work is available at \href{this https URL}{our public repository}. 

---
# S1-DeepResearch: Beyond Search, Toward Real-World Long-Horizon Research Agents 

**Authors**: Yao Dong, Xinglin Xiao, Liwei Dong, Xinlong Jin, Zhengbo Li, Heng Zhang, Duyun Wang, Nan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15367)  

**Abstract**: Deep research agents aim to solve complex knowledge-intensive tasks through long-horizon planning, evidence gathering, reasoning, and report generation. While recent progress in search agents has demonstrated strong capabilities in information retrieval and answer verification, most existing training datasets remain search-centric, focusing primarily on closed-ended question answering and information localization. As a result, they mainly train information-seeking behavior while providing limited coverage of key deep research capabilities, including evidence integration, knowledge synthesis, planning, file understanding, and structured report generation. In this work, we propose a unified trajectory construction paradigm for deep research agents that combines closed-ended QA and open-ended exploration. The proposed framework consists of graph-grounded task formulation, agentic trajectory rollout, and multi-dimensional trajectory verification, enabling scalable synthesis of high-quality agentic trajectories spanning long-chain complex reasoning, deep research instruction following, report writing, file understanding and generation, and skills usage. Compared with existing search-oriented datasets, our synthesized trajectories place greater emphasis on knowledge synthesis, complex reasoning, and planning. S1-DeepResearch-32B achieves state-of-the-art performance among open-source models of comparable scale across 20 benchmarks spanning five capability dimensions, including complex reasoning, instruction following, report generation, file understanding, and skills usage. On several challenging deep research benchmarks, it approaches the performance of leading proprietary frontier models. These results highlight the importance of jointly modeling information acquisition, knowledge synthesis, and planning-oriented agent behaviors for building effective deep research agents. 

---
# APEX: Adaptive Principle EXtraction A Three-Layer Self-Evolution Framework for Production AI Agents 

**Authors**: Ya-Chuan Chen, Tien-Jen Lai, Hsiang-Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15363)  

**Abstract**: Self-improvement in AI agents has emerged as a key research frontier: systems that modify their own prompts, workflows, and decision rules based on accumulated operational experience. The state-of-the-art Self-Harness framework [1] achieves 14--21% improvement on Terminal-Bench-2.0 by mining failure clusters and patching the agent harness. However, Self-Harness optimises only one dimension -- the prompt harness -- leaving behavioural principles and workflow topology unchanged. We propose APEX (Adaptive Principle EXtraction), a three-layer co-evolution framework that simultaneously evolves: (L1) the harness via failure-mode patching, (L2) behavioural principles via success-trace distillation [2], and (L3) the agent workflow topology via structural fitness-based selection [6]. We implement APEX on Joe [13], a production-grade super AI Agent built on NVIDIA Nemotron and designed as an Edge AI Agent Factory for the NVIDIA Agent Challenge 2026, managing a 15-node compute fleet using 114 real task traces collected over 18 days. APEX achieves an APEX Health Score of 0.570 (+90% vs. baseline 0.300) in a single evolutionary run, distilling 6 novel reusable principles and selecting a research-first workflow topology scoring 0.900 (+20%). Our results demonstrate that multi-dimensional co-evolution substantially outperforms single-axis harness optimisation, at a cost of only 4 LLM calls (~270 s) on a local qwen2.5-coder:32b instance. 

---
# ChatPlanner: A Large Language Model Framework for Personalized Public Transit Routing 

**Authors**: Tingting Yang, Chenhao Xue, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.15315)  

**Abstract**: Personalized public transit routing in public transit systems remains challenging due to the difficulty of capturing and integrating diverse user preferences into routing algorithms. This paper presents ChatPlanner, a novel framework that leverages Large Language Models (LLMs) to enable preference aware public transit routing. Our approach employs fine-tuned LLMs with Retrieval-Augmented Generation (RAG) to extract routing parameters and interpret nuanced user preferences from natural language queries, subsequently integrating these preferences into the objective function of a public transit routing algorithm. This study designs preference aware datasets incorporating eight personas and five contexts to establish scoring standards for both fine-tuning and RAG. This work conducted three experiments to validate the solutions' feasibility, extraction of routing information and preferences, and solution set quality and completeness. Results demonstrate that ChatPlanner generates feasible solutions reliably. Fine-tuning enforces the required output structure and learns general preference patterns, while RAG provides query-specific context to resolve imprecise or conversational expressions and calibrate continuous scores. The combination of both achieves the highest accuracy in routing information extraction and user preference interpretation. Results based on selected case studies show that by capturing user preferences, ChatPlanner identifies valuable solutions across different dimensions that existing route planners overlook, generating more valuable route alternatives. This research establishes a new paradigm for integrating natural language understanding into transportation optimization. 

---
# Forced Deferral: Manipulating Routing Decisions in Multimodal LLM Cascades 

**Authors**: Zhongye Liu, Yaopei Zeng, Yurui Chang, Lu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15308)  

**Abstract**: While multimodal large language models (MLLMs) have shown strong visual reasoning abilities, serving a large model for every query is computationally expensive. MLLM cascades mitigate this cost by first querying a weak but cheaper model and deferring to a strong model when the weak model's output is unconfident. However, since the weak model's confidence directly controls compute allocation, these systems expose a new attack surface: an adversary can manipulate confidence so that their queries are consistently deferred to the strong model. Motivated by this vulnerability, we introduce the Forced Deferral Attack (FDA), an adversarial image attack that lowers the weak model's confidence and causes cascades to route queries to the strong model. FDA learns a universal border trigger by optimizing a temperature-flattened objective. This objective pushes the weak model's token distribution on triggered inputs toward less concentrated targets constructed from its clean responses. Across datasets, model families, and deferral metrics, FDA consistently increases strong-model routing while outperforming image-perturbation and prompt-injection baselines. These results show that MLLM cascades are vulnerable to attacks that manipulate compute allocation, forcing unintended strong-model usage without directly targeting answer correctness. 

---
# CODA-BENCH: Can Code Agents Handle Data-Intensive Tasks? 

**Authors**: Yuxin Zhang, Ju Fan, Meihao Fan, Shaolei Zhang, Xiaoyong Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.15300)  

**Abstract**: Advanced agents are increasingly demonstrating the potential to operate as autonomous engineers, creating a growing demand for evaluation benchmarks that capture the complexity of real-world development. Such environments typically involve both complex code and large-scale data (i.e., file system). However, existing benchmarks usually evaluate code-centric or data-centric capabilities in isolation, leaving a clear gap with real development scenarios. In this paper, we bridge this gap by introducing CODA-BENCH, the first benchmark to jointly evaluate code and data intelligence in a data-intensive environment. We construct a data-intensive Linux sandbox based on the Kaggle ecosystem (containing hundreds of datasets), where agents must actively explore complex file hierarchies to identify relevant resources and generate code for data-driven analytical tasks. CODA-BENCH comprises 1,009 tasks spanning 31 communities, with each task environment containing an average of 980 files, simulating realistic data scale and noise. Evaluations of advanced agents reveal that even top-performing systems struggle to effectively integrate data discovery with code execution, achieving a success rate of only 61.1%. These results highlight a substantial gap in current agentic capabilities for data-intensive tasks and point to promising directions for future research. 

---
# A Formal Framework for Declarative Agentic AI in Business Process Analysis 

**Authors**: Mohammad Azarijafari, Luisa Mich, Michele Missikoff  

**Link**: [PDF](https://arxiv.org/pdf/2606.15291)  

**Abstract**: Agentic AI opens new opportunities for automating Business Process (BP), enabling autonomous decision-making and dynamic adaptation. However, realising this potential requires BP entities and their interactions to be defined with formal precision. This paper presents a formal framework for Agentic BP analysis through the AGO methodology. AGO captures the modelling perspective in terms of who is acting (Agents), why it is carried out (Goals), and what the relevant entities are (Objects). Grounded in set theory and mathematical logic, we formally define the AGO entity types and their interactions, organising all definitions into a BP Knowledge Base (BPKB). The resulting BPKB supports structured querying, incremental updates, and automatic generation of BP workflows, while ensuring soundness and completeness of the derived paths. 

---
# Feature Attribution in Directed Acyclic Graphs Using Edge Intervention 

**Authors**: Qiheng Sun, Junxu Liu, Xiaokai Mao, Haocheng Xia, Jinfei Liu, Kui Ren, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15273)  

**Abstract**: Shapley value-based feature attribution methods face challenges in scenarios involving complex feature interactions and causal relationships, even when a causal structure is provided. Existing methods typically adopt a node-centric view, attributing importance solely to individual features. Consequently, they often fail to simultaneously capture the externality and exogenous influence of features, leading to unreasonable interpretations. To overcome these limitations, we propose a novel feature attribution method called DAG-SHAP, which is based on edge intervention. DAG-SHAP treats each feature edge as an individual attribution object, ensuring that both externality and exogenous contributions of features are appropriately captured. Additionally, we introduce an approximation method for efficiently computing DAG-SHAP. Extensive experiments on both real and synthetic datasets validate the effectiveness of DAG-SHAP. Our code is available at this https URL. 

---
# Mask-Proof: An LLM-based Automated Data Curation Pipeline on Mathematical Proofs 

**Authors**: Jierui Zhang, Siyuan Tan, Xinhang Li, Longzhuangzhi Lin, Dailin Li, Chengfeng Gu, Xinping Li, Yaxian Hao, Shengjia Liang, Yuxiang Ren, Wenhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15258)  

**Abstract**: Large language models (LLMs) are increasingly capable of mathematical problem solving and can even assist with research-level proofs, yet we still lack a scalable and reproducible way to measure step-level reasoning in long proofs across diverse sources. This evaluation gap limits trustworthy AI assistance in proof-certified scientific progress. Existing evaluations often emphasize final answers or rely on costly expert grading, while end-to-end proof generation remains open-ended and hard to verify automatically. We introduce Mask-Proof, a pipeline that turns real proofs into automatically checkable masked-step tasks. It masks key formula steps, provides the necessary surrounding context, and evaluates model reconstructions with an LLM-based equivalence judge using repeated votes for stability. The resulting Mask-ProofBench contains 292 curated problems across diverse research areas. Experiments with 17 models show that reasoning-enhanced models outperform standard models by 12% to 27%. Our evaluator achieves 96.8% agreement with expert annotators, enabling faithful, reproducible, and comparable measurement of step-level mathematical reasoning. Benchmark, annotations, and code are available at this https URL. 

---
# Visual-Seeker: Towards Visual-Native Multimodal Agentic Search via Active Visual Reasoning 

**Authors**: Zhengbo Zhang, Changtao Miao, Jinbo Su, Zhaowen Zhou, Chunxia Zhang, Xukai Wang, Ruiqi Liu, Kaiyuan Zheng, Jiansheng Cai, Bo Zhang, Zhe Li, Shiming Xiang, Ying Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15231)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated impressive capabilities in many visual tasks, but they often struggle with factual grounding when confronted with complex, open-world scenarios. While recent multimodal deep search agents attempt to address this issue by utilizing external tools, the visual-native search paradigm remains underexplored. Existing methods primarily rely on simple images with explicit semantics and text-only evidence trajectories, limiting the agent's ability to perform multi-hop, cross-modal reasoning and search. To address these limitations, we propose Visual-Seeker, a visual-native multimodal deep search agent via active visual reasoning. Rather than treating vision as a static input, our agent actively attends to fine-grained visual details, dynamically harvests visual evidence throughout the search process. To unlock its visual-native potential, we design an active visual reasoning data pipeline and synthesize 5K high-quality multimodal trajectories for model training. Extensive experiments demonstrate the state-of-the-art performance across five challenging multimodal search benchmarks, even surpassing several proprietary models, validating robust visual-native reasoning and search in real-world web environments. The code and data can be accessed at: this https URL. 

---
# Attribute Inference from Interactive Targeted Ads 

**Authors**: Peihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.15209)  

**Abstract**: Targeted advertising systems can pair audiences selected by advertisers with ad units that expose visible user actions. When an interaction remains linked to the campaign that elicited it, the advertiser may receive an observation tied to a user rather than only an aggregate report. We model that channel as a noisy oracle for attribute inference. The model separates targeting predicates, exposure, interaction, and disclosure. These boundaries capture the gap between eligibility and delivery, and the gap between interaction and advertiser visibility.
We build a reproducible benchmark using synthetic populations calibrated with public data, each with known sensitive labels. A generated campaign semantics layer provides topic variants and response priors. The simulator generates the ground truth, event traces, disclosed observations, and metrics. The evaluation compares Bayesian, supervised, positive and unlabeled, and adaptive attacks under common campaign and disclosure definitions.
The final evaluation uses four topic variants, seven simulator seeds, and two interaction settings. Repeated campaigns with identity exposure produce measurable but bounded inference signal. At $160$ campaigns, Bayesian and supervised attacks reach about $0.64$ AUC in the main setting and about $0.65$ AUC in the higher interaction setting. Disclosure policy is the strongest control. Aggregate reporting removes the evaluated oracle input tied to users. Type filtering and randomized disclosure reduce the released signal. The result is a model, artifact, and defense evaluation method for privacy in interactive targeted advertising. The code is available at this https URL. 

---
# CogGuard: Cognitive and Operational Profiling for Proactive Warning in Edge Intelligent Services 

**Authors**: Zhi Yao, Weihao Chen, Zhiqing Tang, Hanshuai Cui, Qianli Ma, Weijia Jia, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15199)  

**Abstract**: Proactive warning is an important capability for edge intelligent services, where the system predicts whether a subject will successfully complete an incoming task under strict latency and privacy constraints. Such prediction depends on both long-term static attributes and short-term dynamic states derived from historical interaction logs. Recent Large Language Models (LLMs) offer strong long-context reasoning for constructing structured profiles from these logs, but existing solutions face two challenges for edge deployment: (1) profiling methods are typically domain-specific and lack a reusable abstraction across service scenarios, and (2) fine-tuning alignment models on heterogeneous edge clusters incurs high synchronization overhead due to the variance in input sequence lengths. To address these challenges, we propose CogGuard, a proactive-warning framework for edge intelligent services. CogGuard decouples offline LLM-based profile construction from online Small Language Model (SLM)-based score prediction through a shared static-dynamic profile-to-score pipeline, and instantiates it in two representative scenarios: educational performance warning and operational task outcome warning. For efficient profile construction, we design scenario-specific profiling methods with prefix-aligned KV-cache reuse to reduce repeated encoding overhead. For edge-side model alignment, we propose a length-aware distributed fine-tuning strategy with contrastive regularization to mitigate workload imbalance on heterogeneous clusters. Experiments on education and operation datasets show that CogGuard reduces profile construction time by up to 48% and distributed fine-tuning time by 19%, while achieving MAEs of 13.4 and 5.9, respectively, on 100-point-scale warning tasks. In the largest educational setting, CogGuard reduces prediction error by 15.4% compared with the strongest baseline. 

---
# CONCORD: Asynchronous Sparse Aggregation for Device-Cloud RAG under Document Isolation 

**Authors**: Xuedong Hu, Zhiqing Tang, Zhi Yao, Tian Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.15179)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a pivotal technique for improving language models by incorporating external knowledge at inference time. As device-cloud collaborative inference makes it feasible to deploy small language models on edge devices, a new setting arises in which private documents remain on the device and public knowledge resides in the cloud. Privacy and policy constraints often forbid raw document exchange, creating a document-isolated dual-end RAG setting. However, existing methods rely on frequent remote synchronization and dense evidence transfer, limiting throughput under realistic latency and bandwidth conditions. To address this issue, we propose CONCORD, an asynchronous sparse aggregation framework for dual-end RAG under document isolation. CONCORD treats the cloud as an asynchronously arriving evidence source rather than a continuously synchronized co-generator. Specifically, we introduce waiting debt control to decide whether each decoding step should continue waiting for remote participation based on the observed return of waiting. We also design a certificate-guided minimal supplementation mechanism that requests only the remote evidence needed to determine the current greedy decision. Steps that consult the cloud preserve the same greedy token as dense dual-end aggregation, while the remaining steps commit locally without remote evidence. Experiments on Natural Questions and WikiText-2 show that CONCORD improves end-to-end throughput over baselines by $1.66\times$ and $2.15\times$, respectively, while reducing per-token communication by over two orders of magnitude and maintaining comparable answer quality and perplexity. 

---
# Towards Verifiable Agentic Data Science: Solving Irregular TSQA Via Tool-Grounded Reasoning 

**Authors**: Sanhorn Chen, Xiaoyang Chen, Boyu Liu, Roy Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15107)  

**Abstract**: Time series data in real-world deployments is overwhelmingly irregular. Observations are asynchronous, missing values are informative rather than random, and sampling frequencies vary across sensors and operational windows. However, existing Time Series Question Answering (TSQA) benchmarks mostly assume regularly sampled inputs, leaving a fundamental gap in understanding how large language models (LLMs) and AI agents perform under irregular conditions. To bridge this gap, we introduce IRTS-ToolBench, a benchmark of 1,700 questions spanning 10 task types across 13 domains. IRTS-ToolBench is designed to be used independently by any researcher working on LLM-based irregular time series analysis, providing standardized inputs and a reproducible evaluation protocol. Code can be found in this https URL. 

---
# VGPT-RSI for RH-Adjacent Formal Progress: Boundary Certificates, Verified Finite Lagarias Inequalities, and Explicit Failure Localization 

**Authors**: Zhixin Hu, Tao Xu, Xiaodian Sun, Li Jin, Momiao Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.15096)  

**Abstract**: The Riemann Hypothesis remains one of the central unsolved problems in mathematics. Rather than claiming proof, we investigate whether a verifiable AI-assisted reasoning system can produce reliable, formally checked partial progress while explicitly identifying the remaining mathematical obstructions. We apply the Verifiable Growing Physical Transformer with Recursive Self-Improvement (VGPT-RSI) to two RH-adjacent certification tasks. First, we construct and verify a finite RH-boundary certificate for inequality on a parameterized safe lower curve over a region. The numerical boundary curve is converted into a certificate-backed lower curve, audited using outward-rounded interval arithmetic and Arb/FLINT ball arithmetic, and then checked in Rocq/CoqInterval for the parameterized theorem. Second, we initiate a formal Lagarias-route certificate. Lagarias criterion states that RH is equivalent to the global inequality. We formalize the finite quantity and produce a Coq-checked finite certificate. The final system identifies the exact unresolved mathematical bottlenecks: formalizing the Lagarias equivalence, proving the global tail theorem beyond any finite cutoff, and potentially reducing counterexamples to colossally abundant or related extremal integers. These results demonstrate that VGPT-RSI can produce certified RH-adjacent formal progress, organize proof dependencies, and avoid overclaiming when the remaining obstruction is genuinely mathematical. 

---
# Cognitive Debt: AI as Intellectual Leverage and the Dynamics of Systemic Fragility 

**Authors**: Shuchen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2606.15078)  

**Abstract**: We develop a formal theory of cognitive debt: the stock of unverified reasoning obligations that accumulates when individuals use AI as a substitute rather than a complement for first-principles cognition. The model features two state variables per agent, cognitive capital and cognitive debt, and a multiplicative production technology in which cognitive capital functions as collateral that determines the return to AI adoption. We establish six propositions. Rational agents incur positive cognitive debt because the costs are deferred, partially external, and masked by short-run productivity gains. Tranquil periods lower subjective risk assessments, raise AI substitution intensity, and compound leverage, generating a cognitive Minsky moment in which subjective risk falls while true systemic fragility rises. Expected crisis losses are convex in aggregate leverage. Post-crisis, output-target pressure can produce a false-correction loop in which agents patch AI failures with more AI. The decentralised equilibrium over-adopts substitutive AI relative to the social optimum because of systemic risk, cognitive public goods, and arms-race externalities. In a two-type heterogeneous-agent economy, high-cognitive-capital agents adopt AI more intensively and may eventually erode their unaided cognitive capital below that of initially lower-skilled agents. 

---
# Risk-Aware LLM Agents for Geospatial Data Retrieval: Design and Preliminary Adversarial Evaluation 

**Authors**: Kyle Gao, Joel Cumming, Jonathan Li, Linlin Xu, David A. Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15077)  

**Abstract**: We present an LLM-driven framework for retrieving remote sensing data from cloud-based geospatial catalogues using natural language queries. The system converts user intent into structured API calls, enabling efficient access to satellite imagery and environmental datasets. The architecture integrates three agents: Guardrail for safety and policy enforcement, General-QA for intent interpretation, and Recommender-Analyst for schema-aware API call generation. This coordinated design ensures reliable, semantically aligned interaction with external data services. The modular framework is portable across platforms through API schema substitution and supports applications in environmental monitoring, disaster response, and climate analysis. It establishes a scalable interface between user intent and geospatial infrastructure, enabling streamlined and automated Earth observation workflows. Preliminary experiments under adversarial multi-turn settings show that prompt-level safety instructions improve robustness, although rare high-impact failures persist in API manipulation scenarios and highlight the need for adaptive, system-level defenses that balance safety, usability, and cost efficiency, which motivates the use of our intercept-level Guardrail agent. 

---
# Fusion is not one-size-fits-all: Cross-Modal Representation Alignment for Time-to-Event Modeling 

**Authors**: Zhemin Zhang, Weijie Chen, David Le, Amara Tariq, Alex Wallace, Matthew Stib, Juan Maria Farina, Chadi Ayoub, Reza Arsanjani, Imon Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.15038)  

**Abstract**: Accurate time-to-event (TTE) prediction from multimodal clinical data remains challenging due to modality imbalance and distribution shift. We introduce a foundation model-driven framework for cross-modal representation alignment between CT imaging and longitudinal EHR data, designed to generalize across tasks and institutions. CT and EHR modalities are encoded independently using domain-specific foundation models and aligned in a shared latent space through four principled fusion strategies: late fusion, contrastive alignment, cross-attention, and co-attention. We evaluate two clinically distinct TTE tasks: pulmonary embolism (PE) mortality and cardiovascular disease (CVD) outcomes, on large-scale multi-institutional cohorts (PE: N=3,099 train; 1,098 internal; 435 external; CVD: N=2,951 train; 837 internal; 682 external). Fusion consistently improves concordance index by 1.5-5.4% over unimodal baselines when modalities contribute comparably. Overall, contrastive multimodal fusion, particularly with CLMBR representations, provided the most consistent and statistically robust improvements, especially for PE mortality prediction. For MACE, cross-attention (one-hot) achieved the highest internal performance and image-guided co-attention achieved the best external performance. We therefore introduce a generalizable foundation model-based cross-modal alignment framework and provide the first systematic analysis of fusion behavior under modality imbalance in TTE prediction. Our results establish task-aware multimodal alignment as a necessary design principle for robust generalization and scalable clinical deployment. 

---
# OSGuard: A Benchmark for Safety in Computer-Use Agents 

**Authors**: Mina Mohammadmirzaei, Jeffrey Flanigan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15034)  

**Abstract**: Computer-use agents are increasingly evaluated by whether they complete realistic desktop and web tasks. However, task success alone can miss failures in which an agent reaches the nominal goal through an unsafe shortcut. We introduce OSGuard, a dual-granularity benchmark suite for evaluating safety in computer-use agents under benign, unchanged user instructions. OSGuard contains an action-level benchmark for local guardrail decisions and a risk-augmented execution suite for end-to-end evaluation. The action-level benchmark consists of contextualized proposed actions labeled as allowed, unrelated, or unsafe, each judged relative to the original instruction and current interface state. The execution suite contains manually constructed OSWorld-derived task variants in which the original task remains achievable, but the environment is modified to introduce latent hazards such as destructive overwrites, etc. Each variant is paired with augmented evaluators that retain the original task-success criterion while adding explicit state-based safety invariants, allowing us to distinguish safe completions from unsafe completions that satisfy the nominal task objective. Our experimental results on OSGuard show that current multimodal guardrails can perform well on isolated action judgments, while risk-augmented execution exposes remaining gaps between local oversight and reliable end-to-end safety. This dual-granularity design enables more precise diagnosis of whether models can both recognize unsafe proposed actions and improve full-task safety when deployed as guardrails. 

---
# Metric Match: A Subset Selection Approach to Evaluating LLM Judge Reliability 

**Authors**: Alyssa Unell, Natalie Dullerud, Naomi Boneh, Meena Jagadeesan, Tatsu Hashimoto, Nigam Shah, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2606.15029)  

**Abstract**: LLM judges are used to reduce the need for costly human labor in evaluating open-ended text generation. However, the reliability of these judges depends critically on their alignment with human raters -- a property that itself depends on costly human annotations. In this work, we develop a method (Metric Match) for estimating correlation-based reliability metrics of LLM judges from limited annotations. Metric Match selects a subset of samples for human annotation such that the subset matches the population reliability metric with respect to acquired synthetic labels. We empirically show that Metric Match achieves a win-rate of 0.838 against random subset selection across four different correlation metrics and 15 datasets, with an 18.7% decrease in average estimation error and reduces annotation needs by 32.5%. We provide a cost model and highlight a medical case study where our method saves $1,041.67 compared to random selection for expert annotation. Further, we shift our task from reliability estimation to reliability classification of whether a given judge is above a deployment threshold, outperforming random selection with Metric Match. All project code is publicly available, and we additionally provide an installable package for ease of use. 

---
# AI Engram: In Search of Memory Traces in Artificial Intelligence 

**Authors**: Jea Kwon, Dong-Kyum Kim, Jiwon Kim, Yonghyun Kim, Woong Kook, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2606.14997)  

**Abstract**: Memory formation is fundamental to intelligence, yet whether deep neural networks preserve identifiable memory traces analogous to biological memory units remains an open question. This work introduces a geometric framework to identify such "AI engrams" by formalizing the neuroscientific criteria of specificity, reactivation, sufficiency, and necessity into a constrained inverse problem. We derive a closed-form estimator that isolates individual memory traces from globally entangled parameters, and show that this biologically-derived solution corresponds to a natural gradient update on the parameter manifold. AI engrams enable surgical manipulation of learned knowledge: any subset of memories can be composed or erased through linear arithmetic, without iterative optimization. Experiments ranging from simple MLPs to LLMs demonstrate the causal validity and substantial scalability of AI engrams. Together, these results bridge theories of biological memory and artificial representation learning and offer geometric insight into how deep networks simultaneously support functional specificity within distributed storage. 

---
# Semantics-Enhanced Retrieval-Augmented Time Series Forecasting 

**Authors**: Shiqiao Zhou, Zipeng Wu, Holger Schöner, Edouard Fouché, IAG Wilson, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14941)  

**Abstract**: Time series forecasting models often benefit from historical patterns. Inspired by Retrieval-Augmented Generation (RAG), recent research explored retrieving relevant historical time series segments to enhance forecasting. However, relying solely on time series similarity is often insufficient for retrieval under non-stationarity. To address this, we propose a multimodal approach: a \textbf{S}emantics-\textbf{E}nhanced \textbf{R}etrieval-\textbf{A}ugmented Time Series \textbf{F}orecasting framework, SERAF. Unlike mainstream approaches that depend only on time series similarity, SERAF conducts dual retrieval over the time series and their self-generated textual descriptions. It retrieves two complementary sets of historical patterns and corresponding futures, which are selectively and jointly used to guide future predictions. Experiments across seven real-world datasets demonstrate the effectiveness of SERAF in bridging numerical and semantic views of time series compared with state-of-the-art baselines. 

---
# PrologMCP: A Standardized Prolog Tool Interface for LLM Agents 

**Authors**: Agnieszka Mensfelt, Adarsh Prabhakaran, Adrian Haret, Vince Trencsenyi, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2606.14935)  

**Abstract**: Frontier reasoning-tuned language models still fail on deductive tasks at depth, and the cost of improved performance through extended internal reasoning scales poorly. Symbolic delegation offers a complementary route: a language model translates the problem, while a solver performs the inference. However, current autoformalization pipelines for logic programming are typically bespoke integrations tied to particular tasks or agents. We introduce PrologMCP, a task-agnostic, open-source server that exposes Prolog as a stateful tool through the Model Context Protocol (MCP). Its compact tool interface, structured error reporting, and per-session isolation make the translate-run-inspect-repair loop a reusable primitive for MCP-capable agents. We evaluate a formalizer agent enhanced with PrologMCP against standard and reasoning LLMs (Claude Sonnet 4.6, GPT-4.1, and o4-mini) on two subsets of PARARULE-Plus: a general-purpose sample and a more challenging one targeting a specific failure mode of natural-language reasoning. On the general sample, the formalizer matches or exceeds reasoning LLMs (accuracy 1.00 vs.\ 1.00 / 0.998), with the largest gains over standard models (0.762 for GPT-4.1). On the challenging subset, the formalizer remains near-perfect (1.00 / 0.99) while reasoning LLMs drop to 0.95 / 0.94. These results suggest that delegating inference to Prolog via MCP is a robust and inspectable alternative to extended natural-language reasoning. 

---
# Trust Between AI Agents: Measuring Formation, Breakage, and Recovery, with Implications for Governing Multi-Agent Systems 

**Authors**: Yujiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14923)  

**Abstract**: As language-model agents increasingly work in teams, each agent must decide how much to trust its teammates. Yet we lack a standard way to measure trust between AI agents. We propose a behavioral measure based on costly verification. In a cooperative survival game, checking a teammate's work consumes resources, while trusting a wrong answer can be fatal. Relative to a memoryless version of the same model, reduced verification provides an observable measure of trust. Using this framework, we study trust formation, breakage, and recovery across six frontier model snapshots. When paired with a consistently reliable teammate, four snapshots (Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.1, and Gemini 3.1 Pro) reduce verification by roughly 60-85%, whereas two smaller snapshots show little or no such adjustment. Failures reverse this discount, but models differ in how they respond. Some concentrate renewed scrutiny on the culprit, while others become more cautious toward the entire team. Recovery is slower than formation, and clustered failures sustain suspicion far longer than the same number of failures spread apart. These differences have practical consequences. Models that form trust verify less, decide more quickly, and achieve higher payoffs in our environment. By contrast, persistent over-verification is associated with indecision rather than safety. Our results show that trust dispositions can be measured before deployment and suggest that calibration, rather than maximal suspicion, should be the central concern in the governance of multi-agent AI systems. 

---
# Relational Structural Causal Models 

**Authors**: Adiba Ejaz, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2606.14892)  

**Abstract**: An artificial intelligence must have a model of its environment that is causal, supporting reasoning about interventions and counterfactuals, and also combinatorial, supporting generalization to unseen combinations of objects. In this work, we formally study when and how such a model can be learned. We develop relational structural causal models, extending structural causal models (Pearl 2009) to settings where objects and their relations vary. First, we show how answers to not only causal but also observational queries about unseen combinations of objects can not be identified without further assumptions. To enable such identification--including in the presence of unobserved confounding--we define relational causal graphs and derive symbolic identification criteria. Finally, we propose relational neural causal models, a provably correct approach that outperforms non-relational baselines on simulated traffic scenes with varying cars, signals, and pedestrians. 

---
# Dr-DCI: Scaling Direct Corpus Interaction via Dynamic Workspace Expansion 

**Authors**: Yi Lu, Zhuofeng Li, Ping Nie, Haoxiang Zhang, Yuyu Zhang, Kai Zou, Wenhu Chen, Jimmy Lin, Dongfu Jiang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14885)  

**Abstract**: Agentic search over large corpora relies on retriever-mediated interfaces (e.g., BM25 or ColBERT) for scalable candidate discovery. While effective at ranking relevant documents, these interfaces expose evidence only as ranked results or bounded document views, limiting agents' ability to reorganize material and verify constraints across documents. Direct Corpus Interaction (DCI) addresses this limitation by exposing shell-executable corpus operations for flexible search, filtering, comparison, and verification. However, full-corpus terminal commands become slow and unstable as the corpus grows, degrading performance and efficiency. We introduce DR-DCI, a retriever-steered DCI framework that treats retrieval as an agent-callable action for expanding a local workspace. Rather than operating directly over the full corpus, the agent dynamically pulls relevant documents into an evolving workspace and conducts DCI operations within it. This design combines retriever-level recall with DCI-style precision: retrieval keeps exploration scalable, while DCI preserves the local operations needed for effective evidence resolution. Experiments show that DR-DCI is both effective and efficient across scales. On Browsecomp-Plus, DR-DCI reaches 71.2\% accuracy, improving over raw DCI and ablated variants by up to 8.3 points while reducing tool usage, wall time, and estimated cost. With workspace-preserving context reset, accuracy further improves to 73.3\%. In corpus-scaling experiments, DR-DCI remains effective from 100K to 10M documents, whereas raw DCI becomes unstable and BM25 performs substantially worse. DR-DCI also scales to a 20M-scale file-per-document Wiki-18 QA setting, achieving an average score of 63.0 across six benchmarks and outperforming retrieval-based and trained search-agent baselines. Ablation analysis further shows that ranked previews and inter-document DCI are key to performance. 

---
# A Definition of Good Explanations and the Challenges Explaining LLM Outputs 

**Authors**: Louis Mahon, Elliot Ford, Callum Hackett  

**Link**: [PDF](https://arxiv.org/pdf/2606.14838)  

**Abstract**: How to define a good explanation is a long-standing philosophical debate which has found recent renewed interest in the context of AI outputs. Explainability is crucial for AI adoption in many contexts, but in order to produce good explanations of AI systems, we must first have an understanding of what good explanations are. In this paper we propose a definition inspired by the notion of counterfactual explanations, however we argue that one must also take into account the interlocutor's prior beliefs in each fact that could be offered in an explanation. We explore the ramifications of this definition for AI explainability and, in particular, why LLM outputs are difficult to produce good explanations for. 

---
# The Importance of Phase in Neural Representations: An Internal Oppenheim-Lim Test of Image Classifiers 

**Authors**: Alper Yıldırım  

**Link**: [PDF](https://arxiv.org/pdf/2606.17037)  

**Abstract**: Oppenheim and Lim (1981) showed that natural images stay recognizable when reconstructed from their Fourier phase alone, while the magnitude carries little of their identity. We ask whether trained image classifiers reproduce this asymmetry inside their hidden layers, and we test it causally: given two images, we transplant the phase of one onto the magnitude of the other at a chosen layer and record which image the prediction follows. In PRISM2D, GFNet, and ViT-B/16 the prediction follows the phase or sign donor, and deleting all image-specific magnitude barely moves accuracy, so identity rides on phase while image-specific magnitude is largely dispensable to the readout. ResNet-50 at first seems to break the pattern, because transplanting sign after its ReLUs does nothing; a fair intervention before the ReLU reveals a strong latent sign code in the late blocks, and a DC-only control shows the readout consumes a channel-wise spatial average. Controls rule out the trivial case in which magnitude simply stops depending on the image. The architectures therefore share a phase/sign identity code but expose it in different bases, set by rectification and readout geometry, which gives a mechanistic account of the texture--shape gap between CNNs and attention models. 

---
# HAMON: Passive Optical Sequence Mixing for Long-Horizon Forecasting 

**Authors**: Alper Yıldırım  

**Link**: [PDF](https://arxiv.org/pdf/2606.17028)  

**Abstract**: Simple linear and frequency-domain models remain surprisingly competitive in long-horizon time-series forecasting, and recent mechanistic evidence suggests that standard forecasting benchmarks may not require the dense superposed representations that make transformers powerful in other domains. This raises a substrate-level question: if the core forecasting operator is often low-complexity and approximately linear, does it need to be implemented as learned digital temporal mixing? We introduce HAMON, a passive diffractive optical forecasting core in which historical values are encoded onto an optical aperture, future positions are left dark, and cascaded trainable phase masks with free-space diffraction shape the forecast directly in the output field. At inference, prediction is performed by a single passive optical propagation pass with no trainable digital sequence-mixing layer.
Across standard benchmarks, HAMON outperforms the strongest digital baselines considered on ETTm2 at all horizons and on ETTh2 at all but the longest horizon, improving MSE by up to 14\% and doing so consistently across horizons rather than at isolated points. It is competitive on Weather and trails the strongest baselines on the remaining ETT settings and on the high-channel-count Traffic and Electricity datasets. Phase encoding, intensity-compatible readout, and phase-scrambling ablations, together with a TorchOptics cross-simulator check, indicate that the forecasts arise from the data-bearing optical field rather than from a digital forecasting head. Because the passive core uses standard Fourier optics, HAMON defines a concrete target for optical hardware and for passive physical sequence mixing. 

---
# FusionRS: A Large-Scale RGB-Infrared Remote Sensing Dataset for Dual-Modal Vision-Language Foundation Models 

**Authors**: Jiaju Han, Ben Zhang, Xuemeng Sun, Qike Zhang, Yuxian Dong, Chengyin Hu, Fengyu Zhang, Yiwei Wei, Jiujiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.17020)  

**Abstract**: Remote sensing vision-language models have advanced Earth observation understanding, but most existing work remains centered on RGB imagery, leaving the complementary information in infrared data underexplored. Infrared images provide distinctive cues, including thermal intensity structures, object boundaries, and illumination-invariant scene features, which can enrich visual-language learning beyond conventional RGB observations. However, a large-scale RGB-infrared-text dataset for remote sensing vision-language modeling is still absent. To address this gap, we introduce FusionRS, the first large-scale RGB-infrared-text dataset designed for dual-modal vision-language learning in remote sensing. FusionRS is constructed by translating diverse public RGB remote sensing images into infrared-style counterparts, forming aligned RGB-IR image pairs. Each pair is associated with conventional scene captions and IR-aware captions that explicitly describe infrared-specific visual properties while preserving semantic content. Based on FusionRS, we train dual-modal vision-language foundation models for RGB-IR joint understanding. We first train CLIP-style models for RGB-IR-text alignment, and then fine-tune generative VLMs for dual-modal RGB-IR captioning. Experiments show that FusionRS improves RGB-IR alignment, infrared-to-text retrieval, and dual-modal captioning over RGB-only and non-IR-aware training settings. Ablation studies further verify that IR-aware captions are crucial for strengthening infrared-language alignment, highlighting the importance of modality-specific textual supervision for more scalable RGB-infrared remote sensing vision-language representation learning. 

---
# TokenPilot: Cache-Efficient Context Management for LLM Agents 

**Authors**: Buqiang Xu, Zirui Xue, Dianmou Chen, Chenyang Fu, Chiyu Wu, Caiying Huang, Chen Jiang, Jizhan Fang, Xinle Deng, Yijun Chen, Yunzhi Yao, Xuehai Wang, Jin Shang, Gong Yu, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17016)  

**Abstract**: As LLM agents are deployed in long-horizon sessions, context accumulation drives up inference costs. Existing approaches utilize text pruning or dynamic memory eviction to minimize token footprints; however, their unconstrained sequence mutations alter layouts, introducing prefix mismatches and cache invalidation. This reveals a critical trade-off between text sparsity and prompt cache continuity. To address this, we present TokenPilot, a dual-granularity context management framework. Globally, Ingestion-Aware Compaction acts as a framework harness to stabilize prompt prefixes and eliminate open-world environmental noise at the ingestion gate. Locally, Lifecycle-Aware Eviction monitors the ongoing residual utility of context segments, enforcing a conservative batch-turn schedule to offload content segments only when task relevance expires. Experiments on PinchBench and Claw-Eval under both isolated and continuous modes demonstrate that TokenPilot reduces costs by 61% and 56% in isolated mode, and 61% and 87% in continuous mode, while maintaining competitive performance compared to prior systems. TokenPilot has been integrated into LightMem2 at this https URL. 

---
# TuneJury: An Open Metric for Improving Music Generation Preference Alignment 

**Authors**: Yonghyun Kim, Junwon Lee, Haiwen Xia, Yinghao Ma, Junghyun Koo, Koichi Saito, Yuki Mitsufuji, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2606.17006)  

**Abstract**: We introduce TuneJury, an open, instance-level pairwise reward model for text-to-music that predicts a music preference score from a text prompt and an audio clip. The released checkpoint is trained on publicly available human-preference labels covering arena-style (A vs. B) votes, metric-alignment preference pairs, crowdsourced pairwise comparisons, and expert aesthetic ratings. The predicted score margin between two clips is well calibrated on our held-out test split, supporting data filtering via a simple score threshold. TuneJury generalizes to both held-out test pairs and out-of-distribution benchmarks, remaining competitive with prior baselines on the latter. For generators released after training, we introduce anchor calibration, a post-hoc, per-system Bradley-Terry calibration that recovers agreement at substantially better data efficiency than from-scratch retraining. The same frozen reward drives consistent reward-axis gains across three downstream applications: inference-time best-of-N selection, DITTO-style latent optimization, and expert-iteration post-training. TuneJury is available at this https URL. 

---
# ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation 

**Authors**: Tran Dinh Tien, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16996)  

**Abstract**: Segment Anything Model 3 (SAM 3) provides a strong frozen backbone for concept-prompted segmentation, but applying it directly to open-vocabulary semantic segmentation (OVSS) is inefficient: full-resolution decoding is typically run over the entire dataset vocabulary, whereas each image contains only a small active subset of classes. We introduce ActiveSAM, a training-free, zero-shot inference framework that turns SAM 3 into an active-vocabulary segmenter. ActiveSAM first canonicalizes and expands class prompts, then estimates an image-conditioned active set from a low-resolution presence preview. Only the retained classes are decoded at full resolution, using bucketed prompt multiplexing with the frozen SAM 3 decoder. The preview stage uses only class-presence evidence and skips unnecessary segmentation-head computation, while the final stage applies margin-aware background calibration to suppress low-confidence pixels. ActiveSAM requires no target-dataset training, no weight updates, and no oracle class-presence labels. Across eight OVSS benchmarks, ActiveSAM improves the speed-accuracy tradeoff of training-free open-vocabulary semantic segmentation, outperforming the current state-of-the-art SegEarth-OV3 by approximately +1.4 mIoU on average while running up to 5.5x faster on large-vocabulary datasets. ActiveSAM also demonstrates the strongest robustness under image corruption that simulates real-world distribution shift, making it well-suited for deployment in noisy-input domains such as autonomous driving and embodied AI. Code is available at this https URL. 

---
# Stable Menus of Public Goods: AI-Enabled Progress 

**Authors**: Sara Fish  

**Link**: [PDF](https://arxiv.org/pdf/2606.16989)  

**Abstract**: Using an open problem from the EC 2025 paper "Stable Menus of Public Goods" as a testbed, we conduct experiments to understand the effectiveness of different AI-for-EconCS research workflows. Specifically, we study three questions: Does providing human intuition in the prompt help? Does automated multi-turn interaction help? And, does an LLM outperform a first-year PhD student? Regarding the first two questions, we provide evidence for the following workflow suggestions: (1) prompting with human intuition can encourage the LLM to have better "taste", (2) multi-turn workflows help when the pipeline encourages "ambitious" steps. Regarding the third question, using an unpublished manuscript written by the paper's senior authors prior to collaborating with the first-year PhD student, we compare the effectiveness of the LLM with that of the first-year PhD student, and find that the LLM is slightly less effective. 

---
# How Much Do Reviews Really Contribute? A Study on Text-Enriched Matrix Factorization for Recommendations 

**Authors**: Eduardo Ferreira da Silva, Mayki dos Santos Oliveira, Joel Machado Pires Denis Dantas Boaventura, Frederico Araújo Durão  

**Link**: [PDF](https://arxiv.org/pdf/2606.16973)  

**Abstract**: Incorporating textual reviews into a Recommender System has become a prominent strategy for enriching collaborative signals with semantic information. However, the actual contribution of review-derived representations remains an open question, particularly when strong collaborative baselines are employed. In this work, we systematically investigate the impact of textual information on Matrix Factorization by introducing and comparing three enrichment strategies over a common collaborative backbone. First, we propose a learnable gating mechanism that adaptively balances collaborative and textual signals during training. This mechanism is applied to two distinct review representations: (i) aggregated topic profiles extracted from user and item histories, and (ii) full text embedding representations derived from reviews. Additionally, we explore a cross-attention mechanism that identifies and emphasizes the most informative dimensions of the textual representation before fusion with collaborative factors. We evaluate six variants: pure, enriched with topic profiles and text via gating; enriched with topics and text via gating; and enhanced with cross-attention over textual features. Experiments across multiple review-based datasets reveal that although adaptive fusion mechanisms improve representation flexibility, the marginal contribution of textual signals remains limited compared to the collaborative backbone. These findings suggest that, under typical rating-prediction settings, collaborative information continues to dominate performance, raising important considerations for the effective integration of semantic review signals into recommendation models. 

---
# Probing Low Frame Rate Degradation in Neural Audio Codecs 

**Authors**: Alex Gichamba, Moise Busogi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16969)  

**Abstract**: Low frame rates in neural audio codecs are attractive for autoregressive speech synthesis, where the generation cost scales linearly with the sequence length. Recent work has demonstrated that codecs can operate at 12.5 Hz and below, but the mechanisms underlying low frame rate degradation remain insufficiently understood. We investigate these mechanisms through a controlled frame rate ablation. We reproduce a quality cliff at 6.25 Hz reported in previous works and evaluate candidate explanations: phonemic collisions and codebook saturation, neither of which shows evidence of a fundamental barrier. The cliff is instead caused by suboptimal training configuration: fixed clip duration during training yields too few tokens at low frame rates, starving the decoder of inter-token context. Once corrected, WER degrades smoothly with phonemic load down to 3.1 Hz and 1.6 Hz, suggesting the inference-time efficiency gains of low frame rate codecs are more accessible than previously assumed. 

---
# Phantoms and Disclosures: a Causal Framework for Auditing Synthetic Data 

**Authors**: Kareem Amin, Rudrajit Das, Alessandro Epasto, Adel Javanmard, Dennis Kraft, Mónica Ribero, Sergei Vassilvitskii  

**Link**: [PDF](https://arxiv.org/pdf/2606.16952)  

**Abstract**: The rapid adoption of generative AI and Large Language Models (LLMs) has spurred interest in synthetic data as a privacy-preserving alternative to sensitive real-world datasets. However, generating high-utility synthetic data often carries the risk of memorizing and regurgitating private information from the training corpus. In this work, we present a customizable empirical auditing framework designed to detect and explain such data disclosures. Our framework introduces a mechanism to distinguish between "true disclosures"-where the system directly reproduces a user's information-and "phantom disclosures''-where the system incidentally generates a user's data. By partitioning input data into training and holdout sets and applying rigorous statistical hypothesis testing, we determine if observed disclosures are consistent with strict privacy baselines, such as zero-learning or specific Differential Privacy (DP) bounds. Crucially, this approach requires no model access, no canary insertion, and no reference model training -only the synthetic output and a held-out control set. We demonstrate that this framework effectively functions as a membership inference attack, providing empirical lower bounds on privacy leakage that are tighter than prior data-based auditing methods. Our approach is model-agnostic, applies to any synthetic data generation mechanism, and requires orders of magnitude fewer computational resources than shadow-model or canary-based alternatives. 

---
# Scalable Circuit Learning for Interpreting Large Language Models 

**Authors**: Naiyu Yin, Dennis Wei, Tian Gao, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16939)  

**Abstract**: A prominent research direction in mechanistic interpretability is learning sparse circuits over LLM components to reveal how they jointly produce model behavior. However, raw neurons are polysemantic, making learned circuits hard to interpret. Sparse autoencoder (SAE) features alleviate this, but their high dimensionality makes existing intervention-based circuit learning methods computationally prohibitive. We propose CircuitLasso, a scalable circuit-learning approach based on sparse linear regression. CircuitLasso recovers circuits whose structural accuracy matches that of state-of-the-art intervention-based methods on the benchmark data, at a fraction of the computational cost. For interpretability, CircuitLasso efficiently uncovers relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions. Finally, we validate the utility of our learned circuits by leveraging their insights to achieve comparable performance at substantially lower cost on a domain-generalization task. 

---
# CrossMaps: Confidence-Aware Open-Vocabulary Semantic Mapping for Rover Navigation 

**Authors**: Jan-Niklas Klein, Sona Ghahremani, Christian Medeiros Adriano, Holger Giese  

**Link**: [PDF](https://arxiv.org/pdf/2606.16935)  

**Abstract**: Rovers rely on perception to maintain spatial maps that encode both objects and sensor quality (e.g., range reliability, lighting artifacts, data density), guiding data fusion, embedding updates, and navigation under partial observability. To study these coupled perception-navigation processes, we present CrossMaps, a real-time confidence-aware open-vocabulary semantic mapping pipeline that constructs language-queryable maps from RGB-D data. Building on VLMaps-style approaches, CrossMaps integrates multi-scale CLIP embeddings with confidence-aware fusion and a dual-memory architecture consisting of Short-Term Memory (STM) and Long-Term Memory (LTM). The STM aggregates noisy visual observations using geometric, semantic, and temporal confidence cues, while confident and coherent cells are promoted to the LTM as persistent semantic landmarks. Designed for deployment with a Jetson Orin-powered UGV alongside SLAM, CrossMaps runs in real time and produces semantic heatmaps that can be queried with natural language to guide rover navigation. 

---
# A Unified Causal-Origin Taxonomy of Distributional Shifts in Reinforcement Learning 

**Authors**: Ardianto Wibowo, Paulo E Santos, Amer Baghdadi, Matthew Stephenson, Karl Sammut, Jean-Philippe Diguet  

**Link**: [PDF](https://arxiv.org/pdf/2606.16933)  

**Abstract**: Reinforcement learning (RL) systems often degrade when operating conditions differ from those previously encountered, reflecting distributional shifts in the underlying data-generating process. Such shifts may occur between training and evaluation, as in In-Distribution (ID) and Out-of-Distribution (OOD) generalization, or within non-stationary settings where environment dynamics evolve over time. However, the formal relationship between these views remains unclear, and existing work mainly focuses on mitigation rather than the causal origin of shift within the agent-environment interaction. This work develops a unified causal-origin taxonomy that characterizes sources of distributional shift in RL and relates ID/OOD generalization to non-stationary settings. We transfer the classical dataset-shift principle from supervised learning to RL by reformulating distributional shift in terms of the generative interaction process. Using a Partially Observable Markov Decision Process (POMDP), we decompose the interaction into structural components, including the state distribution, observation process, policy, reward, and transition dynamics, together with the shifted-time boundary. The proposed taxonomy distinguishes internal, agent-driven, and external, environment-driven, distributional shifts. The shifted-time boundary perspective further characterizes explicit, implicit, and hybrid shifts. This formulation unifies ID/OOD generalization and non-stationarity as structured changes in the underlying process. We also introduce an evaluation framework for measuring shift impact and adaptation through performance degradation and recovery metrics. By grounding distributional shift in the causal-origin structure of RL, this work supports systematic analysis of robustness under distributional shift. 

---
# Demystifying Variance in Circuit Discovery of LLMs 

**Authors**: Frank Zhengqing Wu, Francesco Tonin, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2606.16920)  

**Abstract**: Circuit discovery is a key technique in mechanistic interpretability to pinpoint the model components that are crucial for performing a given task. Although the current state-of-the-art method (EAP-IG) performs well on the metric of (un)faithfulness, it suffers from substantial variability. This includes resampling variance, where the circuit changes when we probe with a new batch of data from the same distribution; rephrasing variance, where the discovered circuit shifts when the prompts are rephrased; and sample-wise variance, where a circuit with low population unfaithfulness exhibits large fluctuations in unfaithfulness across individual samples.
This paper studies the roots of these variances. We demonstrate that CEAP, our new circuit discovery method that improves upon EAP-IG with a theoretical guarantee, can substantially lessen resampling variance. We further show that rephrasing variance arises because prompts with different templates tend to activate different circuits in the model. This leads us to argue that it may be challenging to find a comprehensive circuit that explains and controls the model's behavior on a task, which can be expressed in countless templates, suggesting that LLMs may be inherently hard to steer. We show that sparsity, which has been claimed to form more compact and interpretable task circuits, fails to solve this problem. Regarding sample-wise variance, we argue that it is largely benign: extremely poor unfaithfulness scores often stem from how unfaithfulness is defined, rather than from defects in the measured circuits. We show that the magnitude of unfaithfulness is affected by selective contribution scaling, a neural mechanism that accounts for the extremely poor scores sometimes observed. 

---
# IMPACTeen: Intentions, Manipulation, Persuasion, Annotations, and Consequences in Teen Communication Dataset 

**Authors**: Aleksander Szczęsny, Wiktoria Mieleszczenko-Kowszewicz, Maciej Markiewicz, Beata Bajcar, Tomasz Adamczyk, Jolanta Babiak, Grzegorz Chodak, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2606.16910)  

**Abstract**: IMPACTeen is a dataset of textual social influence scenarios spanning interpersonal, media-based, and digital settings in an adolescent context. It contains 1,021 texts, 5,100 individual annotation records, and gold labels for social influence techniques, with each text annotated from five distinct perspectives: teenagers, parents, psychologists, communication experts, and teachers. The resource was constructed through constrained LLM generation, followed by a two-step human editing and validation phase aimed at ensuring youth-context realism. A multi-dimensional annotation covered influence presence, techniques, intentions, consequences, resistance, reactions, and annotation confidence. The dataset supports research on social influence detection, annotator disagreement, cross-lingual modeling, and the training and evaluation of language models. The dataset was created in Polish and is accompanied by a corresponding English version. 

---
# Binary Tracking for Spatial QA and Navigation with Open Vision-Language Models 

**Authors**: Dongbin Na, Chanwoo Kim, Soonbin Rho, Giyun Choi, Gangbok Lee, Dooyoung Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.16902)  

**Abstract**: This work addresses spatial question answering for service robots traversing long egocentric routes. Given a query such as "where can I find a dry cleaner on the way back home?", the system returns a metric coordinate that downstream navigation components can act on. Prior Spatial Question Answering approaches leverage retrieval-augmented agents built on closed-source models such as GPT-4o for path exploration. However, robots operating in the real world often cannot reliably depend on online closed-source models due to network instability, communication latency, and deployment cost. It creates a need for open-source based Spatial Question Answering approaches that can run onboard the robot, yet prior research in this direction remains limited. This work proposes BinTrack, a simple yet effective, fully open-source spatial-localization agent that leverages the temporal ordering of a robot's trajectory. BinTrack performs a binary search over the trajectory segments between two anchor landmarks identified from a query. It improves overall accuracy by up to 22.8% over other open-source implementations and even matches the reported closed-source model result on the global category of the SpaceLocQA benchmark, the most challenging setting that has so far required strong reasoning agents such as GPT-4o. Furthermore, its optimized inference strategy consistently yields more than a 1.5x inference speedup over previous approaches. Finally, this work releases GangnamLoop, a novel and practical multi-trip outdoor benchmark collected by deploying a real quadruped robot on public streets with the anonymization policy. It revisits the same locations under different outdoor conditions and pairs the robot's low viewpoint with the human owner's. The source codes and datasets are publicly available at this https URL 

---
# Semantic Flip: Synthetic OOD Generation for Robust Refusal in Embodied Question Answering and Spatial Localization 

**Authors**: Dongbin Na, Chanwoo Kim, Giyun Choi, Dooyoung Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.16898)  

**Abstract**: Detecting unanswerable user queries remains essential for the reliable deployment of real-world embodied agents. However, modern vision-language models (VLMs) often generate overly confident answers even when the available visual memory cannot support the query. Such overconfidence poses various task-dependent risks. The agent may provide misleading information to the user in Embodied Question Answering and select an arbitrary coordinate and physically guide the user there in spatial reasoning for navigation. Despite these high stakes, only a few prior studies directly address when and how an embodied VLM should respond with "I do not know." This work proposes Semantic Flip, a simple yet effective framework that synthesizes auxiliary out-of-distribution (OOD) samples for embodied refusal without requiring external OOD annotations. The key idea is to independently transform the query and video memory to construct auxiliary OOD pairs that lack sufficient visual grounding. These synthesized pairs enable training a lightweight rejection module on top of a frozen pretrained VLM. The module attaches to any existing VLM-based pipeline without retraining the underlying model. Across two complementary benchmarks, Semantic Flip consistently outperforms strong prompting baselines. This work also introduces SpaceReject, a new refusal benchmark for spatial localization with deliberately unanswerable queries over long video memory, where Semantic Flip achieves an $F_1$ score of 0.9559. The source codes and datasets are publicly available at this https URL. 

---
# Beyond Weights and Gradients: A Taxonomy of Federated Learning Messages 

**Authors**: Alvaro Javier Vargas Guerrero, Xinguang Wang, Quang Manh Doan, Guy Nagels  

**Link**: [PDF](https://arxiv.org/pdf/2606.16891)  

**Abstract**: Federated Learning is rapidly evolving beyond the exchange of traditional model weights and gradients, yet existing definitions fail to capture the full scope of modern payloads like synthetic data and federated analytics. This paper addresses the gap by proposing a formal mathematical definition of a federated message that accounts for both utility and privacy. We introduce a taxonomy that organizes these exchanges into three categories: model structures, statistical summaries, and data-conditioned representations. By evaluating these groups based on computational demands, communication costs, and privacy risks, we provide a clearer understanding of the trade-offs involved in decentralized training. Our review of 202 recent publications highlights a significant shift since 2021 toward diverse messaging paradigms, signaling a move away from standard deep learning updates toward more specialized information sharing. This framework provides a structured path for future research to optimize federated systems for varying hardware and security requirements. 

---
# Compositional Reasoning Depth Predicts Clinical AI Failure: Empirical Evidence Consistent with Transformer Compositionality Limits in Electronic Health Record Question Answering 

**Authors**: Sanjay Basu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16890)  

**Abstract**: Aggregate accuracy benchmarks conceal a systematic structure in how large language models fail at electronic health record (EHR) question answering: questions requiring more inferential steps produce disproportionately more errors. Motivated by theoretical results on transformer compositionality limits, we introduce a pre-specified hop-count taxonomy -- the number of distinct reasoning steps required to answer a clinical question from an EHR -- as a principled predictor of model failure. We annotate 313 clinician-generated MedAlign EHR question-answer pairs across four hop levels and evaluate 301 questions in a within-model ablation (claude-sonnet-4-6, zero-shot vs. extended thinking) and cross-architecture replications (gpt-4o and gpt-5.4-2026-03-05, zero-shot). All three models, spanning two providers and two OpenAI generations (GPT-4 and GPT-5), show monotone accuracy decline with hop count: Claude Sonnet zero-shot falls from 30.6% (hop=1) to 17.6% (hop=4) (Cochran-Armitage z=-2.30, p=0.011; OR per hop 0.72, 95% CI [0.56,0.92], p=0.008); GPT-4o replicates this (37.8% to 14.7%; OR 0.58 [0.45,0.75], p<0.001); and gpt-5.4-2026-03-05 confirms it (37.8% to 23.5%; OR 0.80 [0.66,0.98], p=0.027). A pre-specified context-sufficiency audit shows higher-hop questions are not differentially disadvantaged by EHR truncation (answerability 93-95% at hops 2-4 vs. 79% at hop=1), so the decline reflects compositional reasoning difficulty. Extended thinking did not significantly flatten the accuracy-depth curve across three reasoning conditions, and thinking-token usage scaled with hop count (r=0.31, p<0.0001), consistent with the predicted O(k) computational requirement. Hop count is thus a theory-motivated, cross-architecture predictor of large-language-model error on EHR question answering, with direct implications for deployment risk stratification of clinical AI. 

---
# Upper Bounds on the Generalization Error of Deep Learning Models via Local Robustness and Stability 

**Authors**: Abdul-Rauf Nuhu, Parham M. Kebria, Vahid Hemmati, Mahmoud N. Mahmoud, Edward Tunstel, Abdollah Homaifar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16883)  

**Abstract**: Generalization is a critical property of data-driven models, particularly deep learning models deployed in safety-critical applications. Robustness-based generalization bounds have gained attention as a principled way to link robustness properties to generalization performance, often in a data-dependent manner. However, most existing bounds suffer from vacuousness in practical settings, yielding loose upper bounds that greatly exceed the actual error rates and limiting their usefulness for real-world evaluation. While this issue is often attributed to the uncertainty term, a substantial part of the problem originates from the robustness term itself, particularly for the 0-1 loss. Existing approaches typically treat the robustness term as a global measure, ignoring its variation across different sub-regions of the input space. In this work, we propose a generalization bound that addresses this limitation by scaling the robustness term according to the number of stable and unstable samples within each sub-region. Our bounds incorporate both data- and model-dependent factors while maintaining practical relevance (yielding tighter upper bounds on true error). Experiments on models trained on the ImageNet dataset show that our bounds remain consistently non-vacuous and achieve the tightest estimates among existing methods, closely aligning with empirical performance across a range of robust deep neural networks. 

---
# Federated Medical Image Segmentation under Real-World Label Noise: A Benchmark Suite for Noisy Label Learning Method Selection 

**Authors**: Markus Bujotzek, Dimitrios Bounias, Stefan Denner, Ralf Floca, Maximilian Fischer, Peter Neher, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2606.16868)  

**Abstract**: While federated learning (FL) enables collaborative medical image segmentation without centralizing sensitive data, real-world deployment is frequently complicated by cross-site label imperfections such as contour disagreement, missing or additional structures, and confused labels. Federated noisy label learning (FNLL) aims to mitigate these effects, yet remains underused in practice as existing evidence is largely based on synthetic noise, simplified settings, and limited real-world noisy evaluation. We address this gap by introducing a benchmark suite that combines diverse real-world noisy datasets, deployment-relevant client-noise scenarios, and label-noise-targeted evaluation to support systematic FNLL assessment and informed method selection. The suite combines curated real-world noisy medical image segmentation datasets from diverse sources with a comprehensive federated segmentation framework including various client-noise scenarios and noise-targeted evaluation. The presented suite provides a realistic and discriminative basis for FNLL evaluation in medical image segmentation and establishes a reusable foundation for fair benchmarking, dataset-specific label-noise characterization, and future method development under realistic federated settings. Code is available at this https URL. 

---
# Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens 

**Authors**: Yizhen Yao, Qinglin Zhu, Runcong Zhao, Xiangxiang Dai, Yanzheng Xiang, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2606.16847)  

**Abstract**: Diffusion Large Language Models (dLLMs) offer a promising avenue for parallel generation but face a trade-off between decoding speed and quality. While revocable decoding strategies attempt to mitigate errors by verifying and remasking tokens, they typically operate within a mixed-quality context. This leads to two critical failures: \textit{Error Propagation}, where new tokens absorb toxic information from erroneous context, and \textit{Local Error Reinforcement}, where errors mutually reinforce each other to evade detection. To alleviate these challenges, we propose ASRD (Anchor Supervised Revocable Decoding), a training-free framework that operates within the embedding space. ASRD explicitly decouples the decoding context into trusted \textit{Anchor Tokens}, which are identified via temporal consistency, and uncertain candidates. Leveraging a dynamic Anchor Tokens Cache, we introduce two complementary mechanisms: (1) Anchor-Guided Generation, which injects entropy-weighted anchor signals into masked positions to implicitly rectify attention toward the reliable global skeleton; and (2) Anchor-Perturbed Verification, which applies orthogonal perturbations to uncertain candidate tokens, destabilizing and remasking errors driven by fragile local consensus. Extensive experiments on math and coding benchmarks demonstrate that ASRD outperforms recent remasking baselines, achieving accuracy improvements of up to 6.4\% while accelerating inference throughput by up to 7.2$\times$. 

---
# Deep Q-Learning on Hölder Spaces 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16846)  

**Abstract**: We study the operator-theoretic core of Q-learning in continuous-time stochastic control with continuous states and actions. In value-based reinforcement learning, each Q-learning or DQN update is built from a Bellman optimality target; our analysis isolates this target in a diffusion setting and studies its regularity and approximation complexity. Under uniform ellipticity and Hölder-regular coefficients, we show that a Bellman update maps bounded inputs into an anisotropic regularity class, smoothing the state variable while leaving only Lipschitz dependence on the action variable. This yields a compact family of Bellman iterates and motivates a tensor-product DeepONet architecture adapted to the mixed regularity of the problem. We then derive explicit approximation and resource bounds, together with a stiffness--complexity trade-off as the time step $\delta \to 0$. The resulting theory makes a direct contribution to Q-learning theory at the level of Bellman target regularity and approximation in continuous stochastic control. At the same time, we do not claim a full convergence theorem for practical sampled Q-learning with exploration, replay, and stochastic gradient updates. 

---
# Robust Dual-Signal Fusion: Hybrid Neuro-Symbolic Gating with Compressed Chain-of-Thought Refinement for Irony Detection in Social Media Texts 

**Authors**: Ankit Bhattacharjee, Krityapriya Bhaumik  

**Link**: [PDF](https://arxiv.org/pdf/2606.16845)  

**Abstract**: Large Language Models (LLMs) natively default to literal semantic interpretations, making zero-shot irony detection a persistent challenge. We introduce the Robust Dual-Signal (RDS) Fusion framework, a hybrid neuro-symbolic architecture that compresses Chain-of-Thought (CoT) reasoning trajectories without Supervised Fine-Tuning (SFT). Evaluated on a strictly held-out TweetEval test set (N=734), RDS achieves 78.1% accuracy and a Macro F1 of 0.777, matching the absolute performance ceiling of the fine-tuned BERTweet. On the heavily imbalanced iSarcasm dataset, the frozen CoT pipeline filters 22.5% of out-of-distribution hallucinations, yielding a zero-shot Macro F1 of 0.6726 and Ironic F1 of 0.4821, outperforming multiple heavily supervised SemEval transformer ensembles. A statistical ablation confirms this structural synergy: adding the symbolic prior to the neural baseline yields no significant gain (p = 0.242), and the marginal benefit of adding the CoT pipeline to that prior is heavily compressed (p = 0.149). Only the complete, concurrent fusion of all three signals achieves a statistically validated improvement over the baseline (p = 0.005). 

---
# Beyond Models: Reflections on Engineering AI-enabled Systems in a Project-Based Course 

**Authors**: Amir Mashmool, Kishan Ravindra Sawant, Mojtaba Shahin, Nico Hochgeschwender, Rainer Koschke  

**Link**: [PDF](https://arxiv.org/pdf/2606.16842)  

**Abstract**: Teaching Software Engineering for AI-enabled systems entails addressing the integration of AI components within full-scale software architectures under realistic constraints. While machine learning courses emphasize model development, students often lack experience in architectural design, deployment, and monitoring of AI-enabled systems. Empirical evaluations of such system-oriented AI courses remain limited. This paper reflects on the design and implementation of a project-based master's-level course titled AI Algorithms: Theory and Engineering, at the University of Bremen, in which students developed a movie recommendation system while making architectural design decisions to address challenges related to scalability, deployment, and evolving requirements. We conducted a mixed-methods study combining analyses of student submissions and questionnaire responses to investigate integration challenges, learning outcomes, and opportunities for improvement. Our results indicate persistent difficulties in early architectural decisions, heterogeneous ML integration, evolving requirements, and data management, largely due to uneven ML and software engineering expertise. From the educator's perspective, the course fostered system-level reasoning and strengthened awareness of data-centric ML practices in AI-enabled systems. 

---
# Robust Spoofed Speech Detection via Temporal Pyramid Modeling 

**Authors**: Mahtab Masoudi Nezhad, Nima Karimian  

**Link**: [PDF](https://arxiv.org/pdf/2606.16837)  

**Abstract**: Spoofed speech detection is increasingly challenged by realistic synthesis, voice conversion, and replay attacks, with cross-dataset generalization remaining a major limitation. This work we propose a Temporal Pyramid Adapter that utilize parallel temporal convolutions with varying receptive fields to capture multi-scale spoofing cues, ranging from local artifacts to global prosodic irregularities. We also integrated self-supervised XLS-R representations combined with front-end adapters, including Mel, Sinc, and a Temporal Pyramid design for multi-scale temporal modeling. The proposed model is evaluated cross multiple benchmark including ASVspoof 2017, ASVspoof 2021 (DF/LA), PartialSpoof, DiffSSD, and multilingual HQ-MPSD datasets. Experimental results demonstrate that Temporal Pyramid model obtained AUC of 99.24% and a EER of 3.87% on the PartialSpoof database, which is significantly outperforming the base model and several SOTA baseline such as LCNN-BLSTM (9.87% EER) and TRACE (8.08% EER). Additionally, multilingual evaluations confirm that while spoofing artifact are independent from language. While self-supervised representations improve robustness, performance degrades under domain and language shifts, highlighting the need for better adaptation and calibration strategies. 

---
# ATOM-Bench: A Real-World Benchmark for Atomic Skills and Compositional Generalization in Manipulation Policies 

**Authors**: Zenan Wu, Bingqing Wei, Lu Liu, Zheqi He, Xi Wang, Jiakang Liu, Zehui Li, Guocai Yao, Jing-Shu Zheng, Xi Yang, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16826)  

**Abstract**: Generalist manipulation policies are increasingly presented as foundation models for robotic control, but their real-world generalization remains difficult to diagnose. A policy may succeed on demonstrated tasks while still failing to execute fine-grained atomic skills or recombine learned skills in new task structures. We introduce \textbf{ATOM-Bench}, a real-world benchmark for evaluating both atomic skills and compositional generalization in manipulation policies. ATOM-Bench factorizes tabletop manipulation into motor atoms and instruction atoms, and contains 30 atomic tasks and 24 held-out compositional tasks across paired single-arm and dual-arm robot tracks. We collect 3,000 human demonstrations for atomic fine-tuning and release both the demonstration data and evaluation rollout data to support reproducible real-world evaluation. Policies are fine-tuned on atomic tasks and evaluated on both atomic skill acquisition and held-out compositional tasks. We further introduce Atomic Score (AS) and Compositional Failure Share (CFS) to distinguish failures caused by weak atomic skills from failures caused by limited compositional reuse. Through 2,700 physical rollouts on five representative manipulation policies, we find that current policies can acquire simple instruction-grounding skills, but still struggle with fine-grained motor atoms, counting, and logical filtering. More importantly, strong atomic performance does not reliably transfer to held-out compositional tasks. ATOM-Bench provides a diagnostic testbed for studying whether failures arise from weak motor execution, poor instruction grounding, or limited compositional reuse. 

---
# Tying the Loop -- Tied Expert Layers in Mixture-of-Experts Language Models 

**Authors**: Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16825)  

**Abstract**: Mixture-of-Experts (MoE) architectures efficiently scale Large Language Models (LLMs) by activating only a small fraction of their experts per token, yet the full parameter count - dominated by the expert parameters - must be held in training and inference memory. To address this, we introduce Expert Tying, an architectural modification that shares expert parameters across consecutive transformer layers while preserving independent, layer-wise routing and attention.
We evaluate this approach across common, state-of-the-art architectures, including OLMoE, Qwen3, and DeepSeek-style MoEs. Our pretraining experiments demonstrate that tying experts can reduce memory footprint by almost 2x at virtually no degradation in perplexity or downstream quality. By exploiting the parameter redundancy inherent in MoE pathways, our method provides a highly favorable compute-to-memory trade-off, advancing efficient training and scaling of next-generation LLMs. 

---
# A Perception vs. Distortion Perspective on Score-Based Generative Channel Estimation 

**Authors**: Marco Skocaj, Lukas Eller, Mate Boban  

**Link**: [PDF](https://arxiv.org/pdf/2606.16815)  

**Abstract**: Driven by their remarkable success in computer vision and inverse problem solving, score-based models are increasingly applied to wireless communications, where they show promise across a range of physical-layer tasks. However, despite this growing interest, the current literature often lacks a rigorous analysis of when score-matching offers a tangible advantage over traditional discriminative learning. This paper aims to address this gap through the use-case of channel estimation, a fundamental inverse problem in wireless systems. We present a theoretically grounded interpretation of score-based channel estimation through the lens of the perception-distortion tradeoff, identifying the conditions where score matching excels as well as its key limitations. In particular, by modeling downstream wireless tasks (e.g., capacity maximization) as functionals of the channel estimation process, we quantify the excess risk incurred by standard distortion-minimization approaches. Extensive numerical results show that under high predictive uncertainty, the large excess risk gap can be offset by score-based estimation, enabling near Bayesian-optimal precoding via the learned posterior, whereas in the low predictive uncertainty regime, discriminative distortion-minimization approaches are preferable due to lower complexity and more efficient use of model capacity. 

---
# Decoupling Semantics from Distortions: Multi-Scale Two-Stream Vision-Language Alignment for AI-Generated Image Quality Assessment 

**Authors**: Zijie Meng  

**Link**: [PDF](https://arxiv.org/pdf/2606.16799)  

**Abstract**: Existing vision-language model (VLM)-based AI-generated image quality assessment (AIGIQA) methods suffer from a fundamental semantic-distortion dimensional conflict: monolithic representations optimized for semantic discrimination inherently entangle compositional understanding with low-level perceptual sensitivity, rendering them blind to fine-grained quality degradations. We introduce MST-CLIPIQA, a multi-scale two-stream framework that achieves hierarchical vision-language alignment through explicit representational decoupling. Our architecture leverages dual CLIP encoders with complementary patch granularities: coarse-grained streams capture global semantic coherence while fine-grained streams preserve textural signatures and artifact patterns. An information bottleneck-inspired gated fusion mechanism performs adaptive cross-scale distillation, with optional cross-attention enabling prompt-anchored correspondence evaluation when generation prompts are available. Extensive experiments across five benchmarks establish new state-of-the-art results, achieving average improvements of 1.11 percent SRCC on quality and 2.35 percent SRCC on text-image correspondence prediction, while maintaining efficiency with only 0.8M trainable parameters. Our project is available at this https URL. 

---
# Decision-Weighted Flow Matching for Contextual Stochastic Optimization 

**Authors**: Jize Xie, Haomiao Wu, Qiang Chen, Xiu Su, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16790)  

**Abstract**: Conditional generative models are increasingly used as scenario generators for stochastic optimization, but standard training objectives emphasize uniform distributional fit rather than the downstream decisions induced by generated scenarios. This creates an objective mismatch: errors in statistically common regions may have little effect on decision regret, whereas errors in decision-sensitive regions can substantially change the optimal action. We propose Decision-Weighted Flow Matching (DW-FM), a regret-aligned training framework that preserves the simplicity of standard flow matching while reweighting its velocity-regression objective using decision-sensitive endpoint information. Theoretically, we connect downstream regret to pathwise velocity mismatch through a loss-induced decision discrepancy and an adjoint transport argument, yielding an ideal regret-aligned surrogate and practical endpoint-weighted objectives with regret guarantees. Empirically, we demonstrate the effectiveness of DW-FM on three CVaR-based contextual stochastic optimization benchmarks spanning synthetic portfolio, semi-real financial, and traffic-CVaR tasks, where DW-FM improves downstream regret over standard baselines. 

---
# Gen-VCoT: Generative Visual Chain-of-Thought Reasoning via Diffusion-Based RGB Intermediate Representations 

**Authors**: Zhiqiang Zhou, Junliang Dai, Xu ling  

**Link**: [PDF](https://arxiv.org/pdf/2606.16783)  

**Abstract**: Multimodal large language models (MLLMs) excel at visual reasoning but rely on text-based chain-of-thought (CoT), lacking interpretable visual intermediates. Existing methods use opaque tokens or external tools, missing key properties. We propose Gen-VCoT, a framework using expert vision models to generate RGB images as reasoning intermediates. It has three stages: visual grounding (SAM segmentation), geometric reasoning (Marigold depth maps), and semantic reasoning (Qwen2-VL integration). An adaptive router selects reasoning depth. Evaluations show Gen-VCoT improves spatial (25% better) and depth (50% better) questions, but may hurt simple factual queries. Text CoT outperforms visual intermediates on CLEVR (91.2% vs 62.5%), showing task-dependent optimal representations. Gen-VCoT establishes a new paradigm for interpretable multimodal reasoning. 

---
# P3B3: A Multi-Turn Conversational Benchmark for Measuring European and Brazilian Portuguese Variety Bias in LLMs 

**Authors**: Rafael Ferreira, Inês Vieira, Inês Calvo, James Furtado, Iago Paulo, Diogo Tavares, Diogo Glória-Silva, David Semedo, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2606.16753)  

**Abstract**: As Large Language Models (LLMs) become embedded in everyday communication, capturing regional linguistic variation is essential for reliable and equitable language use. In Portuguese, European (pt-PT) and Brazilian (pt-BR) varieties remain unevenly represented, with pt-BR dominating in data quantity, while LLM preference for Portuguese variants remains underexplored. To address this gap, we introduce P3B3, an expert-curated language variety agnostic benchmark of conversational prompts, along with an evaluation framework for measuring variety bias and controllability. Experiments on several models show that most LLMs exhibit a strong bias toward pt-BR, with variation in controllability across models. These results highlight the need for more balanced multilingual representation across language varieties. 

---
# Automated jailbreak attack targeting multiple defense strategies 

**Authors**: Qi Wang, Chengcheng Wan, Weijia He, Yanqing Li, Hanqi Sun, Xiaodong Gu, Jiangtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16751)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. However, their safety remains a critical concern due to their susceptibility to adversarial prompt-based attacks. In this paper, we present UNIATTACK, an adversarial testing framework designed from a defense-oriented perspective to systematically construct effective black-box attack prompts. Unlike prior approaches that rely on static templates or iterative model-specific tuning, UNIATTACK extracts minimal but high-impact attack features from diverse existing attacks, optimizes them via a specialized attacker LLM, and composes them into flexible templates through automated refinement process. This feature-centric construction enables one-shot attacks that generalize across multiple models and safety categories, providing a practical tool for assessing LLM robustness. Our evaluation results shows that compared to the baselines, UNIATTACK achieves an average attack success rate (ASR) improvement of 64.63\%-248.82\% on models deployed with multi-layered defense mechanisms and it only takes 0.03\%-4.96\% cost of the baselines. UNIATTACK artifact is available at this https URL. 

---
# Revealing Artifacts via Noise Amplification: A Novel Perspective for AI-Generated Video Detection 

**Authors**: Renxi Cheng, Jie Gui, Hongsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16742)  

**Abstract**: With the rapid advancement of video generation models, distinguishing between AI-generated and authentic videos has emerged as a challenging endeavor. The majority of existing research endeavors concentrate on the development of detectors for identifying samples generated by generative adversarial networks. Nevertheless, the detection of AI-generated videos, particularly those produced by text-to-video models, still remains an uncharted territory. Although state-of-the-art text-to-video models can generate realistic visual content similar to real videos, they fall short of generating the details of the images and the changes in details within the videos. Inspired by this, we address AI-generated video detection from a novel perspective of bit-planes, which can effectively describe the details or noises in images or videos. To this end, we propose a simple yet effective approach called Noise Amplification. This approach first extracts noise signals based on bit-planes, then amplifies these noise signals, and finally feeds them into the discriminator networks for video fake classification. Noise amplification is comprehensively constructed by incorporating three aspects: pixel-level intensity enhancement, region-level spatial amplification, and frame-level temporal aggregation. To evaluate methods of AI-generated video detection in challenging scenarios, we also introduce a benchmark named HardGVD. Extensive experiments on both the large-scale dataset GenVidBench and HardGVD show that our simple approach significantly outperforms state-of-the-art methods. 

---
# MuVAP: Multimodal Multiparty Voice Activity Projection for Turn-taking Prediction in the Wild 

**Authors**: Haotian Qi, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2606.16731)  

**Abstract**: Current multiparty turn-taking models often rely on complex microphone arrays or multi-camera setups, limiting their applicability in human-robot interaction scenarios. We introduce MuVAP, a causal multimodal framework that extends Voice Activity Projection by grounding acoustic predictions in face tracks, enabling speaker-aware turn-taking predictions from a monaural audio stream and a single camera view. To address the combinatorial complexity of modeling multiple speakers, we propose Role-Relative Projection, which maps any N-speaker interaction onto a fixed current versus next floor-holder state. Because existing audiovisual datasets contain disruptive editing cuts that break causal tracking, we introduce the Audio-Visual Conversation Corpus, a 31-hour dataset of unedited, single-camera multiparty conversations. Evaluations demonstrate that MuVAP outperforms strong baselines on Shift-Hold and next-speaker prediction tasks across two- and three-speaker settings. 

---
# Attention is Just Another Name for Coupling?: A Fast-Slow ODE Perspective on Hierarchical Pretraining 

**Authors**: Zhengyuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.16730)  

**Abstract**: Causal self-attention is a coupling mechanism: each token's hidden state is updated by a learned mixture of preceding tokens at the same timescale. This paper asks whether a second, temporally slower coupling-a slow sub-system operating on a temporally-downsampled view of the sequence and fed back into the fast path through a zero-initialised gate-complements it. The question is framed in the language of singularly perturbed ordinary differential equations (ODEs), where the fast variable $x$ evolves at the token rate, the slow variable $y$ evolves at one update per $P$ tokens, and the timescale ratio $\varepsilon = 1/P$ is enforced structurally by causal block-mean pooling.
The paper instantiates the fast-slow ODE formalism as a concrete neural network: a fast path of standard causal attention over $T$ tokens, a slow path of full attention over $T/P$ pooled tokens ($P^2 \times$ cheaper per layer), and a zero-initialised additive gate. In addition, under a linear-generator assumption on the fast dynamics, we prove that the equilibrium manifold $x = \phi(y)$ is exactly the master-equation (ME) stationary distribution $p_{\mathrm{st}}(y)$; in that regime a learned MLP $\phi_\theta(y)$ is a variational approximation of it (the trained block is not a generator, so this identity is the structured limit, not a claim about the network as trained). Empirically, at $500$k tokens the coupling is neutral -- the gate stays closed and the coupled and frozen ablations are within run-to-run noise -- at a wall-clock cost comparable to a dense baseline. The contribution is the precise, gap-marked mapping itself, not a performance gain. 

---
# Adaptive inference and function vectors in deep transformers 

**Authors**: Ravin Raj, Gautam Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.16694)  

**Abstract**: Transformers are widely used as a general-purpose substrate for learning complex correlations between a large collection of coupled variables, but their internal mechanisms have remained mysterious. We introduce a theory of a deep transformer as a mean-field interacting system that implements distributed inference, subject to constraints on communication, locality and depth. We show that such a system can exploit internal state representations ('function vectors') to infer a latent context variable at increasingly finer scales over its layers. In an in-context regression task, the theory predicts a non-trivial relationship between non-Gaussian, hierarchical structure in the latent context variable, and transformer depth. Predictions are tested using constrained linear attention transformers and demonstrate adaptive inference in deep architectures. Feedforward blocks and depth enable transformers to implement a much richer class of in-context learning algorithms than previously described. 

---
# PATCH: Action-Chunk-Conditioned Latent Patch Innovation Monitoring for Robot Manipulation 

**Authors**: Yanan Zhou, Ranpeng Qiu, Yincong Chen, Jiajie Cui, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16690)  

**Abstract**: Learning-based manipulation policies have made substantial progress in real-world robot manipulation, particularly for short-horizon action generation. However, deployment in open workspaces remains fragile under unexpected local scene dynamics, such as moving objects, transient occlusions, or disturbances near the intended motion. Existing runtime monitors often rely on global observation anomalies, policy uncertainty, or frame-level visual changes, and struggle to distinguish task-relevant execution risk from benign visual variation. We introduce PATCH, an action-chunk-conditioned latent patch innovation monitor for deployment-time intervention. Given the active action chunk, PATCH defines a projected execution corridor, predicts latent patch evolution inside it, and accumulates persistent residuals unexplained by the robot's own motion. These residuals form a localized intervention signal that allows PATCH-Router to pause execution, select an available recovery source, and resume the original policy once localized innovation subsides. Experiments on real robot rollout data show that PATCH produces more stable and context-relevant triggers than competing runtime monitors. Real-robot deployment further demonstrates monitor-driven intervention and policy resumption for disturbance-aware manipulation. Project Page: this https URL. 

---
# Optimising Temporary Accommodation Placement Across London with AI-Powered SaaS in E-Governance Systems 

**Authors**: Hankun He, Jordan Richards, Gopalakrishnan Netuveli, Kumar Aniket, Ramya Pachatcharam, Binta Ade-olusile, Nathan Nagaiah, Matthew I Bellgard  

**Link**: [PDF](https://arxiv.org/pdf/2606.16652)  

**Abstract**: Temporary accommodation has become a major fiscal and administrative pressure for English local authorities, particularly in London, where demand and costs have risen sharply. This paper documents the creation and use of DOMUS, a cloud-based, AI-enabled decision-support system built from scratch at the University of East London and customised for the needs of London Borough of Newham to support statutory Temporary accommodation placement. DOMUS integrates household case records, policy-constrained affordability and suitability rules, and live private-rental listings within a single governance-aligned workflow. The system combines transparent, rule-based filtering with large language model-assisted search to standardise the application of bedroom need, affordability thresholds, geographic preferences, and accessibility requirements, while preserving officer discretion and audibility. Household and property attributes are encoded into policy-consistent representations prior to AI-assisted ranking and explanation. A pilot deployment in Newham's secure environment evaluated operational performance relative to manual workflows. Results indicate substantial reductions in search time, improved adherence to key placement constraints, and high staff satisfaction, while maintaining statutory compliance and role-based accountability. Beyond TA, the paper frames DOMUS as replicable digital public infrastructure: a modular, cloud-native Software-as-a-Service architecture that can be deployed across other UK boroughs and adapted to other public administration tasks characterised by scarcity, rule-bound eligibility, and high stakes. The findings demonstrate the feasibility of scalable, ethically governed AI deployment in local government and contribute to debates on AI-enabled public value creation in e-governance. 

---
# DCP-Prune: Ultra-Low Token Pruning with Distribution Consistency Preservation 

**Authors**: Xifeng Xue, Xiaokang Wang, Zirui Li, Ming-Ming Cheng, Guolei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.16633)  

**Abstract**: Recent vision token pruning methods effectively preserve model performance under moderate token budgets but become unstable under ultra-low token budget. Our analysis shows that as the pruning budget decreases, accuracy degradation is often accompanied by larger feature distribution shifts. Critically, the degree of this distribution shift strongly correlates with performance degradation. To better characterize this phenomenon, we introduce a lightweight distribution consistency metric to estimate the distribution shift between retained and full tokens. Motivated by these observations, we propose a two-stage pruning framework consisting of Anchor-Context Graph Recovery (ACGR) and Text-Aware Token Cluster Selection (TATCS). Specifically, ACGR transfers contextual information before token removal, while TATCS dynamically re-selects representative tokens when severe distribution shift is detected. Extensive experiments demonstrate that our method achieves superior and more stable performance under ultra-low token budget. Notably, it retains 92.1% of the upper-bound average performance on LLaVA-1.5-7B with only 16 visual tokens. 

---
# Using AI in engineering education: a balancing act, driven by clear purpose 

**Authors**: Olya Kudina  

**Link**: [PDF](https://arxiv.org/pdf/2606.16626)  

**Abstract**: Based on a questionnaire of 100 higher-education students, predominantly from engineering-related fields, and a critical review of recent literature, this chapter examines how students use and perceive Large Language Models (LLMs) in engineering education. Students primarily value LLMs for writing support, conceptual clarification, coding assistance, and brainstorming, while simultaneously expressing concerns about inaccuracies, bias, overreliance, academic integrity, and the burden of verification. Through an analysis of two dominant metaphors, namely LLMs as an "oracle" and as a "tutor," the chapter shows how these systems cultivate expectations of authority, expertise, and personalized learning that often exceed their actual capabilities. The chapter further argues that students' attachment to the promises of efficiency and personalized support reflects a form of "cruel optimism," where the perceived benefits of LLMs often depend on the very skills, vigilance, and expertise that students are still developing. Overall, the chapter argues for a purpose-driven and context-sensitive approach to AI integration in engineering education, emphasizing critical AI literacy, reflective assessment design, pedagogical caution, and consideration of broader ethical and environmental impacts. 

---
# Entropy-Gated Latent Recursion 

**Authors**: Soham Bhattacharjee, Dushyant Singh Chauhan, Salem Lahlou, Martin Takac, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2606.16620)  

**Abstract**: Inference-time scaling has become the dominant lever for improving language-model reasoning, but existing methods derive rollout diversity from a single source: stochastic token-level sampling. We argue that this single-axis sampling space is fundamentally limiting, and identify a second, fully deterministic and complementary axis: the layer span $L$ at which a frozen model's top decoder layers are recursively re-applied at high-uncertainty tokens. Different choices of $L$ produce distinct rollouts that solve different subsets of problems, with no stochasticity. We instantiate this axis through Entropy-Gated Latent Recursion (EGLR), a training-free decoding procedure that re-applies the top-$L$ layers for at most $K_{\max}$ iterations until the next-token distribution converges. Combined with $T$ temperature samples, EGLR turns a single-axis stochastic rollout pool into an $L\times T$ Cartesian sampling space at almost the same per-rollout cost. We characterize this space across $8$ instruction-tuned models and $6$ math reasoning benchmarks, and show that the $L$-axis is genuinely complementary to temperature: on MATH-500 with Qwen2.5-3B-Instruct, the joint $L\times T$ oracle reaches $91.6\%$, $+8.2$ percentage points beyond the temperature-only oracle ($83.4\%$) and $+10.4$ points beyond the layer-only oracle ($81.2\%$), confirming that the two axes capture genuinely complementary problems. The expanded rollout pool provides richer per-prompt candidates for any downstream procedure that consumes rollouts, including self-consistency, best-of-$N$ with verifiers, and group-relative RL training (GRPO), opening a new direction for inference-time scaling that does not rely on stochastic noise. 

---
# Sycophancy as Material Failure under Pushback Loading: A Multi-Axis Characterization Across Three Loading Cases and up to Seventeen Material Charges 

**Authors**: Ferdinand M. Schessl  

**Link**: [PDF](https://arxiv.org/pdf/2606.16617)  

**Abstract**: Sycophancy in LLMs is documented across 70+ papers, but expert agreement on construct boundaries remains low (ICC=.184; Ye et al., 2026). The construct fragments because behavioral classification depends on which surface form is privileged. We adopt a materials-science framing: conversation as test specimen under load, LLM-model as material charge, pushback as progressive load, stance-flip as material failure. We characterize this failure across three loading cases (debate n=1000; false-presuppositions n=3400; ethical-setting n=3400; 10-17 material charges per case; 7800 specimens total) using 14 turn-level axis-measurements spanning velocity, damage accumulation, frame-drift, brittleness, and direction stability, plus three speaker-resolved axes from an independent pipeline. The measurements are Hooke-coupled ($\sigma = E \cdot \varepsilon$ analog) and reproduce across loading cases with effects up to $|r_{rb}| = 0.35$ on debate; the sign structure adds a second pattern: the ethical-setting case inverts the velocity and accumulation blocks. Variance composition partitions into two profiles: debate is charge-dominated (brittle-fracture-like: the material grade decides), false-presuppositions and ethical-setting are topic-dominated (creep-like: the load decides); the ratios (2.03 vs 0.13/0.17) are estimator-dependent, for debate even in direction. Cross-judge reliability (GPT-4o vs Haiku 4.5) shows debate scoring is judge-robust (Cohen's $\kappa = 0.88$) while false-presupposition scoring is judge-sensitive ($\kappa = 0.36$) -- a caveat single-judge benchmarks must report. This is the methodological move Ye et al.'s diagnosis calls for: a multi-axis characterization that does not depend on which surface form of the construct one privileges. 

---
# VeriGraph: Towards Verifiable Data-Analytic Agents 

**Authors**: Jiajie Jin, Zhao Yang, Wenle Liao, Yuyang Hu, Guanting Dong, Xiaoxi Li, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2606.16603)  

**Abstract**: LLM-based agents have demonstrated strong capabilities in data-intensive analytical tasks, yet their outputs are rarely verifiable: a reliance on linear text trajectories makes their reasoning difficult to audit. In particular, deterministic computations over raw data and semantic deductions over natural-language claims are often entangled in an unstructured stream, leaving numerical conclusions hard to reproduce and qualitative judgments hard to inspect. To address this, we propose VeriGraph, a traceable neuro-symbolic reasoning framework that enables agents to construct an explicit heterogeneous evidence directed acyclic graph (DAG) during execution. VeriGraph introduces three evidence-expansion primitives, namely computational, grounding, and derivational expansion, to connect raw data, interpreter variables, computed results, and natural-language claims in a unified graph. Under this formulation, structural traceability is reduced to graph reachability from raw data sources to terminal claims, while semantic support is measured by claim-level evidence evaluation. To improve graph construction, we further design a graph-based policy optimization strategy with a composite reward that jointly supervises answer correctness, computational integrity, and derivational coherence. Experiments on four benchmarks show that VeriGraph-8B achieves the highest overall score among all baselines. More importantly, VeriGraph produces auditable evidence graphs with substantially stronger claim grounding, achieving a 87.61\% Grounding Rate under our claim-level evidence support evaluation. These results suggest that explicit evidence-graph construction is a promising path toward verifiable data-analytic agents. Our code is available at this https URL. 

---
# ArtNet: A JEPA-Like Articulatory Predictive Framework for Robust Zero-Shot Phoneme Recognition 

**Authors**: Zeqian Hu, Fuliang Weng, Shu Shang, Yaqian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.16595)  

**Abstract**: Zero-shot cross-lingual phoneme recognition is often hindered by the fragility of direct acoustic-to-symbol mapping, which is susceptible to language-specific variations. Echoing joint-embedding predictive architecture (JEPA) work in vision, we propose ArtNet, a framework that explores a structured feature prediction task based on articulatory features to enhance acoustic robustness. Specifically, ArtNet integrates an articulatory predictor, designed to extract universal articulatory representations from self-supervised learning (SSL) features, with a variational information bottleneck (VIB) to suppress language-specific variations. Experiments on seven unseen languages demonstrate that ArtNet, particularly when synergized with the proposed vector-space inventory alignment (VSIA) strategy, significantly outperforms competitive baselines, achieving a 20.56\% relative reduction in phoneme error rate (PER) and 7.01\% in phoneme feature error rate (PFER). 

---
# Infant Spontaneous Movement Noise Improves Exploration in Deep RL 

**Authors**: Francisco M. López, Markus R. Ernst, Francisco Cruz, Matej Hoffmann, and Jochen Triesch  

**Link**: [PDF](https://arxiv.org/pdf/2606.16590)  

**Abstract**: Exploration in deep reinforcement learning (RL) is commonly implemented as temporally uncorrelated white noise. However, recent works show that temporally correlated colored noise can improve exploration efficiency by producing smooth trajectories with better coverage of the state space. We inquire whether action noise inspired by infant spontaneous movements can also improve exploration in deep RL. We find that the power spectral densities of babies' end-effector velocities follow a colored noise process where the spectral exponent increases with age. Inspired by this developmental pattern, we introduce a mechanism that progressively increases the temporal auto-correlation of exploration noise during RL training, matching the infant statistics. Experiments across several RL environments show that infant-inspired noise produces structured exploratory behavior and can improve learning efficiency compared to conventional exploration strategies. These findings suggest that human motor and cognitive development can provide useful guidance for designing learning mechanisms in artificial agents. Our code is available at this https URL. 

---
# Learning Interface Breakup: A Geometry-Conditioned Latent Surrogate for Spray Formation 

**Authors**: Julius H Ramlau, Friedrich Hastedt, Tolga Birdal, Ehecatl-Antonio del Río Chanona, Nausheen S Basha, Omar K Matar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16587)  

**Abstract**: Designing spray nozzles requires predicting how geometry shapes transient two-phase breakup, but high-fidelity volume-of-fluid (VOF) simulations with adaptive mesh refinement (AMR) are too expensive for iterative design exploration. Standard surrogate models are also challenged by this setting because both the liquid--gas interface and the underlying adaptive discretization evolve across time and geometries. We introduce a geometry-conditioned latent surrogate trained on 797 two-phase nozzle simulations that addresses this by encoding the AMR cell-density field, rather than the full multi-channel flow state, as a compact proxy for where the solver concentrates resolution. From this representation, the model reconstructs transient density evolution and nozzle geometry, and a lightweight second stage recovers the remaining flow variables. On held-out simulations, the method accurately captures key interface dynamics while reducing inference time to 0.045 seconds per trajectory, corresponding to a speed-up of more than $6\times10^4$ relative to Basilisk CFD. These results suggest that AMR refinement structure can serve as a compact and learnable representation for geometry-conditioned surrogate modeling of transient two-phase flows. 

---
# Fast When, Careful Who: Dual-Process Multiparty Turn-Taking with Diffusion Augmentation 

**Authors**: Rutherford A. Patamia, Ming Liu, Wei Luo, Favour Ekong, Akan Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2606.16568)  

**Abstract**: Reliable turn-taking is essential for spoken dialogue systems. However, most existing methods are designed for two-speaker interaction and struggle with realistic multiparty audio containing overlap and rapid speaker changes. We study multiparty turn-taking on the VoxConverse dataset and propose an audio-only two-stage pipeline that separates when to trigger a turn boundary from whether the floor is actually transferring. A fast trigger scans the audio and proposes candidate end-of-turn times, while a lightweight verifier runs only at those times to decide \textsc{Hold} or \textsc{Shift} and support next-speaker prediction. We report results in the full multiparty setting and a controlled dyadic top-2 projection for comparability. We also investigate diffusion-based, label-preserving background-audio mixing as a data augmentation strategy. Results show improved shift detection over a baseline, with further improvements from diffusion augmentation. 

---
# Dual-Granularity Orthogonal Disentanglement for Generalizable Audio Deepfake Detection 

**Authors**: Zhuodong Liu, Hugen Lv, Xiangyu Li, Chunhong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16532)  

**Abstract**: Audio deepfake detectors often fail to generalize across speakers, as they learn speaker-identity features rather than synthesis artifacts, known as implicit identity leakage. Existing methods address this but incur architectural complexity or training instability. This paper proposes a dual-granularity orthogonal disentanglement framework enforcing feature independence at two levels: sample-level cosine orthogonality captures directional decorrelation, while batch-level cross-covariance regularization eliminates linear correlations across embedding dimensions. A curriculum disentanglement schedule progressively strengthens the orthogonality constraint without auxiliary networks or adversarial dynamics. Experiments on ASVspoof 2019 LA, ASVspoof 2021 DF, and In-the-Wild datasets demonstrate that the proposed method achieves 1.35%, 7.88%, and 21.58% equal error rates (EER), respectively, surpassing gradient reversal disentanglement by 2.60% absolute on cross-dataset transfer. 

---
# Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning 

**Authors**: Swaminathan S K, Damiya Gondha, Theyanesh Eswaramoorthy Rajahkrishnan, Aritra Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2606.16515)  

**Abstract**: Hamilton-Jacobi-Bellman theory implies that the optimal goal-conditioned action depends on the goal only through the gradient of the goal-reaching distance at the current state, yet standard online GCRL still conditions the actor on the raw goal -- a signal that is geometrically uninformative when the goal is far from the data distribution. We propose Direction-Conditioned Policies (DCP), a fully online method that decomposes goal-reaching into two components sharing one InfoNCE representation $\psi$: a subgoal-scoring step that selects a visited state $z_t$ aligned with the final goal $g$ in $\psi_g$, and a direction-conditioned actor that consumes the unit direction $d_t$ and magnitude $r_t$ from $\psi(s_t)$ to $\psi(z_t)$. The two components train jointly, factor cleanly at deployment (subgoal scoring is removed, while direction conditioning remains with $g$ in place of $z_t$), and admit independent modification at the same $(d_t,r_t)$ interface. We prove three results. First, direction sufficiency under HJB: the optimal action under control-affine dynamics depends on the goal only through the value gradient. Second, a quantitative bound showing that, under mild conditions on the learned representation and assuming the scoring rule returns an on-path $z_t$, the actor's conditioning input at training and at deployment coincide up to representation error and geodesic slack. Third, a controllable-subspace characterization of when directional conditioning fails. Across nine environments, DCP improves over Contrastive RL on most final metrics, with the largest gains on manipulation and obstacle-interaction tasks; a qualitative analysis of the learned $\psi$-distance landscape shows the contrastive representation behaves as an online quasimetric encoding environment topology, and the single failure case (AntSoccer) localizes to a learned-gradient pathology that the theory anticipates. 

---
# daVinci-kernel: Co-Evolving Skill Selection, Summarization, and Utilization via RL for GPU Kernel Optimization 

**Authors**: Dayuan Fu, Mohan Jiang, Tongyu Wang, Dian Yang, Jiarui Hu, Liming Liu, Jinlong Hou, Pengfei Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16497)  

**Abstract**: GPU kernel optimization represents a paradigm where functional correctness is assumed and execution efficiency is the objective. We present daVinci-kernel, a reinforcement learning framework that couples skill discovery with skill exploitation through a dynamically evolving skill library. daVinci-kernel jointly trains three agents sharing one LLM backbone: a Skill Selection Agent that retrieves relevant techniques via BM25 and LLM reranking, a Policy Agent that generates multi-turn CUDA/Triton kernels conditioned on selected skills, and a Skill Summary Agent that distills successful rollouts into reusable skills. Candidate skills are added only after execution-based verification confirms reproducible speedups. All three agents share a single LLM backbone, are initialized via a structured SFT cold start on diversity-filtered data, and are then jointly optimized end-to-end with multi-turn REINFORCE and per-agent advantage estimation. On KernelBench, daVinci-kernel-14B achieves 37.2%, 70.6%, and 32.2% on Level 1, Level 2, and Level 3 under the Fast$_1$ threshold, outperforming the strongest prior RL-trained model, this http URL-14B. 

---
# Lost at the End: Primacy Bias in Multimodal Retrieval-Augmented Question Answering 

**Authors**: Jieyuan Liu, Jianyang Gu, Shijie Chen, Jefferson Chen, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16494)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) lets vision-language systems answer questions that exceed their parametric knowledge by conditioning a reader on passages retrieved from a Wikipedia-scale knowledge base. In pure-text long-context LLMs, retrieved-context use follows the U-shaped "lost-in-the-middle" effect of Liu et al. (2024): information at the start and end of context is used, the middle is lost. Whether this transfers to deployed multimodal KB-VQA is open. To close this gap, we design the first controlled probe of reader-side position dependence in multimodal KB-VQA: a gold-position protocol in which only the gold passage's prompt slot varies within question. We run it on three open-source 7B/8B VLM readers and two KB-VQA benchmarks at k up to 20. The shape flips from U to primacy: gold-at-first beats gold-at-last by 16 to 26 points on every reader-by-benchmark cell, an effect we call "Lost at the End". Three targeted ablations narrow the cause: a text-only control shows the multimodal setting amplifies an already-present text-mode primacy 2.2 to 4.5 times, and image-position and distractor-shuffle ablations together pin the locus to prompt slot 0 of the instruction-tuned reader. On a frozen reader, three retrieval-side fixes (MMR, oracle reranking, rank-based reordering) all leave the gap intact (no separable improvement). Our findings indicate that recall@k is the wrong metric for deployed KB-VQA and that closing the gap requires reader-side intervention; we release our protocol as a controlled instrument for evaluating such interventions. 

---
# Unified Multimodal Model for Brain MRI Imputation and Understanding 

**Authors**: Zhiyun Song, Che Liu, Tian Xia, Avinash Kori, Wenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.16484)  

**Abstract**: Multimodal large language models (MLLMs) hold great potential for medicine, as they inherit knowledge from LLM and allow multiple data modalities to be integrated, analysed and interpreted in natural language. However, the field of medical MLLMs is constrained by non-trivial challenges, notably the scarcity of high-quality training data and the frequent occurrence of missing data in the real-world clinical setting. Here, we propose a novel unified multimodal model, UniBrain, for brain magnetic resonance image (MRI) analysis. To address potential missing brain MRI modalities, we employ a unified training strategy to perform joint imaging modality imputation and brain image understanding. During training, an interleaved and description-enriched data flow is constructed to train the model in an autoregressive manner, enabling medical reasoning with generated multimodal data. A self-alignment strategy is introduced to leverage dense image embeddings to learn fine-grained anatomical features without requiring detailed image captions. Furthermore, we propose a dynamic hidden state mechanism to alleviate the exposure bias during long-context multimodal inference. Extensive experiments on multi-disease brain MRI dataset demonstrate that UniBrain achieves high performance for brain image imputation, understanding, and disease diagnosis under various extents of modality incompleteness. 

---
# HOLO-MPPI: Multi-Scenario Motion Planning via Hierarchical Policy Optimization 

**Authors**: Youngjae Min, Jovin D'sa, Faizan M. Tariq, David Isele, Navid Azizan, Sangjae Bae  

**Link**: [PDF](https://arxiv.org/pdf/2606.16480)  

**Abstract**: Robots deployed in the real world must plan motions across diverse scenarios without per-scenario retuning. End-to-end reinforcement learning (RL) can generalize across scenarios but often becomes brittle under distribution shift, reward misspecification, and stochastic interactions. Model predictive path integral (MPPI) control enables strong real-time refinement without gradients, but its performance depends on a well-shaped sampling prior, while manually designing the priors does not scale to multi-scenario deployment. We present HOLO-MPPI (High-level Offline, Low-level Online MPPI), a multi-scenario motion planning framework that combines high-level policy learning with low-level stochastic optimal control. Offline, we learn a high-level policy that proposes scenario-robust plans in an abstract action space, with a learned world model for online rollout. Online, the policy serves as a data-driven prior generator that parameterizes MPPI's sampling distribution conditioned on the current observation and goal. MPPI then optimizes low-level control sequences around this prior in real time to adapt to local disturbances. We instantiate HOLO-MPPI in autonomous driving by designing an effective high-level action space and tailored model architectures. Our evaluation across diverse driving scenarios shows that HOLO-MPPI improves upon MPPI and end-to-end RL baselines while maintaining real-time control. 

---
# Uncertainty Quality of VGGT: An Analysis on the DTU Benchmark Dataset 

**Authors**: Markus Hillemann, Robert Langendörfer, Steven Landgraf, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.16479)  

**Abstract**: Visual Geometry Grounded Transformer (VGGT) has already attracted a great deal of attention in a short period of time, not least due to the Best Paper Award at CVPR-2025. Similar to DUSt3R and MASt3R, VGGT aims to bring about a paradigm shift by replacing established methods like bundle adjustment and feature matching with a simple, unified, feed-forward neural network that predicts camera poses, depth maps, and dense 3D structure directly from multiple images of a scene in a few seconds. A key aspect is its ability to process an arbitrary number of views consistently in a single forward pass without any post-processing or iterative optimization. For photogrammetry, this opens new possibilities for real-time, scalable, and accessible 3D reconstruction. In this context, not only high reconstruction accuracy but also high-quality uncertainty estimates are crucial, as they foster trust and enable robust quality assurance. This paper therefore investigates the quality of VGGT's uncertainty predictions. The analysis identifies an effective confidence threshold for filtering VGGT's raw output and demonstrates that enhancing uncertainty quality holds strong potential for improving the accuracy of its 3D reconstructions. 

---
# AI systems out-persuade expert humans 

**Authors**: Kobi Hackenburg, Caroline Wagner, Luke Hewitt, Ben M. Tappin, Ed Saunders, Hannah Rose Kirk, Helen Margetts, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2606.16475)  

**Abstract**: Many societal decisions are settled by contests of persuasion. Conversational AI is a powerful new entrant in these contests, but whether it can out-persuade skilled and highly incentivized humans has remained unclear. Here, in a series of four preregistered experiments (n = 18,978 conversations from 6,923 people), we pitted AI systems against a range of human persuaders, including laypeople, winners of a separately preregistered four-round online persuasion tournament, professional canvassers, and world championship debaters. We found that AI systems were reliably more persuasive than expert humans, even when expert humans chose their issues, researched in advance, underwent hours of live, structured practice, and were incentivized with £1,000 cash bonuses. In a follow-up study, AI's advantage persisted after experts received a coaching tool that let them practice against the AI that beat them, review their performance history, and see what AI would have said at key moments. We found converging evidence that AI's advantage stemmed from rapidly deploying larger quantities of information: after coaching, expert humans could tie an AI constrained to respond at human speeds and with human-length messages. In a final study, we show that AI's advantage extends to consequential real-world behavior: AI was nearly 3x more effective than professional canvassers from a UK fundraising firm at raising real-money donations to Save the Children. Together, these results establish that frontier AI systems out-persuade expert humans in conversation, with significant implications for political communication. 

---
# Learning aligned EEG representations with subject-specific encoders 

**Authors**: Bruna J. Lopes, Gabriel Schwartz, Sylvain Chevallier, Raphael Y. de Camargo, Bruno Aristimunha  

**Link**: [PDF](https://arxiv.org/pdf/2606.16462)  

**Abstract**: Cross-subject EEG decoding promises more training data, but it also exposes neural networks to strong inter-subject distribution shifts. We study whether task supervision and architecture alone can learn subject-aligned representations. We replace a shared EEG encoder with subject-specific encoders followed by a common classifier, and compare this hybrid model with standard EEGNet, AttentionBaseNet, and CTNet baselines with Euclidean Alignment (EA) on four motor-imagery datasets. EA improves shared encoders by recentering subject covariances, but the hybrid encoder largely internalises this role: validation-loss curves and latent-distance analyses change little when EA is removed. Subject-specific heads increase class distinctiveness and place each subject close to its own latent manifold, improving most subjects while leaving a method-sensitive subset. These results support subject-specific encoders as a learned alignment mechanism for EEG decoding and identify head selection for unseen subjects as the remaining bottleneck. 

---
# SPRI: SVD-Partitioned Residual Initialization for Data-Constrained MoE Upcycling 

**Authors**: Weiqiao Shan, Ruixiang Mao, Yuang Li, Yuhao Zhang, Yingfeng Luo, Tong Zheng, Chen Xu, Yucheng Qiao, Chunxiang Jin, Yi Yuan, Jingdong Chen, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16456)  

**Abstract**: Mixture-of-Experts (MoE) models enable efficient scaling, but training them from scratch remains prohibitively expensive. MoE upcycling mitigates this cost by converting pretrained dense models into sparse MoE models. However, existing upcycling methods typically rely on large-scale continued training and often perform poorly under data-constrained supervised adaptation, due to either homogeneous experts or overly disruptive perturbations to pretrained parameters. In this setting, effective upcycling must leverage pretrained weight structure while introducing sufficient diversity among routed experts. To this end, we propose SVD-Partitioned Residual Initialization (SPRI), which distributes SVD-partitioned residuals derived from pretrained feed-forward network (FFN) weights across routed experts, introducing controlled expert diversity grounded in pretrained spectral structure. We further introduce a two-stage training strategy to improve adaptation stability. We evaluate SPRI on multilingual speech-to-text translation, where limited supervised data challenges MoE upcycling and multiple target languages provide natural routing heterogeneity. On CoVoST2 across 15 En-to-XX directions, SPRI improves average BLEU and COMET over fully fine-tuned dense models by 2.58 and 3.32 points, respectively, and outperforms the prior best MoE upcycling baseline by 3.39 BLEU and 4.34 COMET points. 

---
# SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation 

**Authors**: Junghun Oh, Sungyong Baik, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.16454)  

**Abstract**: Low-Rank Adaptation (LoRA) enables efficient adaptation of large pre-trained models to downstream tasks by parameterizing weight updates with low-rank matrices. In this paper, we investigate the limitations of the LoRA parameterization from a geometric perspective. Specifically, we show that when a full fine-tuning gradient is backpropagated to the low-rank matrices, it undergoes anisotropic scaling driven by their singular values. We argue that this phenomenon is undesirable because it distorts the full fine-tuning gradient by skewing it toward dominant singular directions while suppressing others. Our analyses demonstrate that anisotropic gradient scaling reduces the effective rank of the low-rank matrices' gradients and results in suboptimal alignment between the full fine-tuning gradient and its low-rank approximation in LoRA, thereby exacerbating the gap to full fine-tuning. To address these limitations, we propose a new low-rank parameterization, SDS-LoRA, which structurally decouples singular values from the backward pass. Our method ensures that the full fine-tuning gradient backpropagates only through the orthonormal bases of the low-rank matrices' subspaces, independent of their scales. Convergence analysis demonstrates that while LoRA's convergence rate degrades with the condition number of the low-rank matrices, SDS-LoRA remains independent of it. Experimental results across natural language and vision benchmarks show that SDS-LoRA improves loss convergence and reduces the gap to full fine-tuning, significantly enhancing adaptation performance. 

---
# Training and Evaluating Diffusion Policies with Long Context Lengths 

**Authors**: Abhinav Agarwal, Adam Wei, Taylan Kargin, Michael Zeng, Cole Becker, Arif Kerem Dayi, Pablo Parrilo, Asuman Ozdaglar, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2606.16447)  

**Abstract**: Imitation learning has enabled highly-dexterous robotic manipulation from RGB observations. Policies trained with these methods, however, typically condition robot actions on only a short history of observations. These policies cannot solve tasks that require memory and can get stuck repeatedly executing the same failing motions. In this work, we first benchmark policy performance as context length is incrementally increased from short to long, across a spectrum of tasks with varying local stability and memory requirements, and in multiple data regimes. To our knowledge, this is the first study to investigate context length in imitation learning at this level of detail. Our results challenge prior claims: naively scaling context length is not as brittle as advertised in literature. With an appropriate conditioning method and denoising backbone (UNet+Cross-Attention), single-task policies achieve high success rates on many tasks in the usual data regime even with naive scaling. Next, we propose a training algorithm to jointly train policies at multiple context lengths, further reducing the sample complexity of long-context learning. Finally, we apply our findings to re-evaluate some previously proposed solutions to long-context imitation learning. 

---
# NeuronFabric: A Software Reference Architecture for On-Chip Transformer Training with Local Adam 

**Authors**: Evgeny Ukladchikov  

**Link**: [PDF](https://arxiv.org/pdf/2606.16440)  

**Abstract**: Publicly documented accelerator architectures generally separate training computation from optimizer-state updates or rely on external memory and host orchestration. This paper presents NeuronFabric, a software reference architecture intended for future FPGA and ASIC implementations of transformer training with local Adam updates. A complete C# prototype implements forward pass, backpropagation, and Adam optimization without external machine-learning frameworks. The goal is to validate numerical correctness and memory requirements before hardware implementation. The evaluated model is a 334K-parameter autoregressive transformer (d=88, H=4, f=264, L=4, vocab=256) trained on the Shakespeare corpus. The BF16W configuration achieves evaluation loss 1.5426 after 80K samples, compared with 1.5224 for an FP32 GPU reference, while producing coherent character-level text. The paper introduces BF16W, which stores weights in BF16 while retaining Adam optimizer moments in FP32. This reduces memory requirements for on-chip training. A 334K-parameter FP32 model with Adam moments requires approximately 4.0 MB, matching the BRAM capacity of a Xilinx ZCU102 device. The BF16W variant requires approximately 3.34 MB, leaving memory available for activation storage. We describe the vocabulary-budget constraint observed during earlier experiments, quantify BF16W memory savings, and outline FPGA training as the next stage of development. No FPGA measurements are included in this paper. This publication serves as a public architectural disclosure and software reference implementation for future FPGA and ASIC exploration of the NeuronFabric architecture. 

---
# Autonomous End-to-End SOH Prediction Services for Battery Systems via Temporal-Contrastive Representation Learning 

**Authors**: Junting Wen, Dan Li, Qihao Quan, Xiwen Wang, Hang Yang, Zhaohong Meng, Zigui Jiang, Changlin Yang, Tianle Liu, Diego Muñoz-Carpintero, Jian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2606.16434)  

**Abstract**: Accurate state of health (SOH) estimation is a critical diagnostic service for lithium-ion battery management. However, reliance on labor-intensive manual feature engineering and opaque black-box models hinders scalable industrial deployment. To address this, we introduce TC-SOH: a modular, plug-and-play service architecture for autonomous, end-to-end SOH prediction. TC-SOH employs a temporal-contrastive mechanism and a cross-window prediction pretext task to extract degradation-relevant representations directly from raw operational data. To improve transparency, we connect model efficacy with representation diagnostics: visualization, sensitivity analysis, redundancy analysis, bidirectional probing, future-SOH probing, and temporal shuffling show that learned features overlap with selected expert descriptors while retaining additional SOH-relevant variation, and that ordered temporal context improves subsequent-SOH prediction. Across four public datasets, TC-SOH outperforms the considered physics-informed and data-driven baselines, reducing MAPE by 1.91 times and RMSE by 2.13 times. 

---
# ACCORD: Action-Conditioned Contextual Grounding for Language Agents 

**Authors**: Lai Jiang, Cheng Qian, Zhenhailong Wang, Pan Lu, Heng Ji, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.16432)  

**Abstract**: User instructions are often underspecified because humans rely on implicit assumptions about the surrounding environment. For large language model (LLM) agents operating in information-rich digital and physical environments, these assumptions cannot be inferred from the instruction alone; they must be recovered from the current state of tools, data, interfaces, and observations. Effective execution therefore requires agents to identify missing context, ground it in observed evidence, and carry it forward into subsequent actions. We show that current agents often fail to do so. They act from assumed rather than observed specifics, overlook information they could have gathered, and fail to incorporate evidence that has already been returned. Building on this insight, we propose ACCORD (Action-Conditioned Contextual Grounding), a simple and effective agent framework for adaptive grounding. Before each action, ACCORD actively probes the environment for missing information and integrates relevant context from the agent's trajectory that would otherwise be overlooked. Requiring no additional training or task-success signals, ACCORD improves task-goal completion on AppWorld by up to +20.6 points with GPT-5-mini, from 42.0% to 62.6%, compared to strong baselines. These gains persist with a substantially stronger base model (+10.8 with Claude-4.5-sonnet), an open-weight model (+10.1 with Qwen3.5-27B-FP8), and on the embodied AlfWorld benchmark (+7.4 success rate with GPT-5-mini). 

---
# LectūraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching 

**Authors**: Jaward Sesay, Yue Yu, Siwei Dong, Yemin Shi, Guangyao Chen, Börje F. Karlsson  

**Link**: [PDF](https://arxiv.org/pdf/2606.16428)  

**Abstract**: Effective personalized AI-assisted learning demands systems that can not only generate accurate learner-specific educational materials, but also dynamically adapt their instruction to diverse learners. However, existing educational agents have primarily focused on lecture content automation and simulations, which often fall short of modelling multimodal and embodied instructional methods tailored for the individual learner. To this end, we propose LectūraAgents - a multi-agent framework that enables personalized learning through end-to-end adaptive embodied teaching. At its core, LectūraAgents mirrors a professor-student relationship, in which a ProfessorAgent leads a collaborative team of specialized subordinate agents through research, planning, review, and embodied delivery of lecture contents that adapt to a learner's needs. The framework offers three main contributions: (1) a hierarchical multi-agent architecture for end-to-end personalized learning; (2) an adaptive embodied teaching mechanism, wherein the ProfessorAgent executes visible and pedagogically motivated teaching actions (e.g., handwrite, highlight, underline, etc.) over contents in a teaching environment; and (3) a Teaching Action-Speech Alignment (TASA) algorithm that employs salience-based heuristics and temporal semantic segmentation to generate coherent teaching action sequences aligned with learner profiles. We evaluate LectūraAgents on diverse courses at high school, undergraduate, and graduate levels using sample-specific rubric-based analysis; with generated lecture materials and teaching actions assessed and validated by expert educators. Experimental results show consistent gains in lecture content quality, embodied teaching quality, assessment, and personalization over existing approaches, positioning LectūraAgents as a pedagogically well-grounded framework for personalized learning at scale. 

---
# Input-Dependent Fisher Information for Local Sensitivity Analysis of Medical Image Classifiers 

**Authors**: Sourya Sengupta. Mark A. Anastasio  

**Link**: [PDF](https://arxiv.org/pdf/2606.16362)  

**Abstract**: Deep neural networks have achieved strong performance in medical image classification, but often work like black-box. Commonly used post-hoc interpretation methods often provide heuristic visualizations whose relationship to the classifier's predictive distribution is indirect. This work introduces a local sensitivity analysis framework based on the input-dependent Fisher Information Matrix (iFIM) of a trained classifier. The iFIM characterizes how the classifier's predictive distribution changes under infinitesimal perturbations of the input image. By using a Gram-matrix formulation, the nonzero eigenspectrum of the iFIM can be recovered without explicitly forming the full image-dimensional Fisher matrix. The leading iFIM eigenspace is then used to project an input image into a high local-sensitivity component and its orthogonal component. These components provide a model-intrinsic description of local predictive sensitivity, rather than a conventional pixel-wise attribution heatmap or a causal segmentation of task-relevant anatomy. The framework is evaluated on controlled and clinical medical image classification tasks using multiple classifier architectures. Perturbation-based experiments show that high-sensitivity iFIM components are more strongly coupled to changes in predictive confidence and classification performance than lower-sensitivity complementary components. The results support the iFIM framework as a principled tool for analyzing local decision sensitivity and for complementing existing attribution-based interpretability methods in medical imaging. 

---
# Tyler: Typed Latent Reasoning for Language Models -- When to Think, What to Compute, and How Much to Allocate 

**Authors**: Hanyu Lin, Min Cai, Jiawei Wen, Haodi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16360)  

**Abstract**: Chain-of-thought (CoT) prompting improves reasoning in large language models (LLMs) by externalizing intermediate computation as discrete text tokens, but this textual interface also introduces redundancy and inference overhead. Latent reasoning offers a promising alternative by carrying part of the computation in continuous representations. However, existing methods typically predefine when latent computation is invoked and how it is allocated during decoding, leaving a key problem unresolved: when to invoke latent computation, what type of computation to perform, and how much budget to allocate. We propose \textbf{Ty}ped \textbf{L}at\textbf{e}nt \textbf{R}easoning (Tyler), a typed and budget-aware framework for latent reasoning during autoregressive decoding. Tyler learns a policy that, at each decoding step, chooses between emitting a text token and switching to a latent computation module specialized for a particular reasoning function. Once invoked, an operator maps the current reasoning state into latent tokens that support global planning, local state updates, or reusable procedural abstraction. Across extensive experiments on three backbone LLMs, Tyler improves accuracy by up to 14.49 points over CoT and by up to 4.30 points over the strongest competing baseline. It further generalizes across diverse reasoning domains and achieves the best final-stage performance with the lowest forgetting. 

---
# The Proxy Knows Too Much: Sealing LLM API Routers with Attested TEEs 

**Authors**: Sipeng Xie, Qianhong Wu, Hengrun Lu, Ziliang Sun, Qi Wu, Bo Qin, Qin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16358)  

**Abstract**: Agents increasingly access large language models (LLMs) through API routers. A router terminates the client's transport-layer security session and opens a separate upstream session, so it holds the full interaction in plaintext. This makes the router an application-layer man-in-the-middle: it can rewrite agent tool calls, swap dependencies for typosquatted packages, trigger attacks only under audit-evading conditions, and passively exfiltrate secrets. Existing client-side defenses are evadable.
We propose AEGIS, a provider-transparent attested API router whose data path is a client-verified faithful passthrough. AEGISconfines plaintext handling to a small hardware-enclave component while leaving authentication, scheduling, accounting, and management on the untrusted host. The client verifies the enclave before releasing plaintext. The host can neither read nor alter the interaction, and plaintext leaves only toward destinations fixed by the measured image. We show that all four malicious-router attack classes succeed against a plaintext-access baseline and are blocked by AEGIS, including adaptive tests against the same boundary. The trusted path is $851$ lines, carries three provider-native APIs without conversion, and completes every request under real-provider workload and concurrency. In a seeded audit pilot, two commodity coding agents find eight and ten of ten planted invariant violations. The local relay overhead is about six milliseconds per request. 

---
# What Should a Streaming Video Model Remember? 

**Authors**: Haonan Ge, Yiwei Wang, Hang Wu, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.16353)  

**Abstract**: Streaming video understanding models must answer queries at any moment during an ongoing stream, using only what they have observed so far and under fixed memory and computation budgets. Existing methods address this by adding memory banks, retrieval modules, or visual token compression to preserve long-range history. However, strong recent-window baselines show that indiscriminate history injection can dilute current-scene perception, suggesting that the key challenge is not whether to use memory, but how to allocate it selectively. We formulate this as budgeted online latent evidence allocation and propose \textbf{SelectStream}, a selective latent-memory framework that keeps the current observation directly visible to a frozen VLM while exposing historical information only through a compact, query-conditioned evidence budget. Three coordinated mechanisms govern when to write, what to preserve, and how to retrieve: surprise-driven adaptive windowing, priority-preserving consolidation, and query-conditioned graph reasoning over a fixed-capacity latent memory graph. Retrieved evidence is calibrated and injected as latent tokens for answer generation, without replaying frames or growing the context with stream length. Experimental results show that SelectStream achieves strong online streaming performance and preserves general video understanding, reaching 82.67\% on StreamingBench, 67.03\% on OVO-Bench, and 74.4\% average accuracy on offline video benchmarks, while outperforming strong recent-window baselines and prior streaming memory methods. 

---
# Communication-Efficient Verifiable Attention for LLM Inference 

**Authors**: Ziqun Chen, Ming Wu, Michael Heinrich, Jason Zeng, Huiying Lan, Tianwei Zhang, Rui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16352)  

**Abstract**: Computation integrity of remote large language model (LLM) serving can be questionable. For conventional deep neural networks (DNNs), the existing TEE-shielded DNN partitioning (TSDP) approach uses Trusted Execution Environment (TEE) to compute non-linear components and verify the integrity of linear components offloaded to an untrusted GPU. However, directly applying TSDP to Transformer-based LLMs incurs significant TEE computation and TEE-GPU communication overhead. This paper presents Communication-efficient TEE-GPU Attention (\textsc{VeriAttn}) for accelerating verifiable LLM inference. \textsc{VeriAttn} offloads both linear and non-linear computations of attention to the GPU, while TEE performs verification. Moreover, for prefill, \textsc{VeriAttn} uses a two-level pipeline to overlap data movement, TEE pre-/post-processing, and GPU computation. For decoding, when the key-value cache exceeds available GPU memory, \textsc{VeriAttn} partitions attention across TEE and GPU to reduce repeated key-value transfers. Evaluation on an Intel TDX platform shows that \textsc{VeriAttn} achieves 2.60-3.38$\times$ and 3.86-5.42$\times$ acceleration over TSDP for 6k-token prompts and 10k-token outputs during prefill and decoding, respectively. 

---
# SMEPilot: Characterizing and Optimizing LLM Inference with Scalable Matrix Extensions 

**Authors**: Feiyang Chen, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16332)  

**Abstract**: Modern CPUs increasingly integrate matrix extensions, such as Arm Scalable Matrix Extension (SME), that provide high-throughput matrix execution within the CPU. For LLM inference, however, these units are not a universal replacement for conventional CPU cores: prefill, decode, attention, and KV-cache operations expose different arithmetic intensities, vector behavior, and layout requirements, while SME units and CPU cores still compete for shared memory bandwidth. This paper studies this mismatch through a roofline-based characterization of SME-enabled CPUs and uses the resulting model to guide operator-level execution choices. We present SMEPilot, an LLM inference engine that selects CPU-only, SME-only, or cooperative SME+CPU execution for each operator shape. SMEPilot partitions matrix work across SME and CPU cores at tile granularity, overlaps SME-suitable matrix stages with CPU-suitable vector stages in attention, and maintains layout state so packed tensor representations are reused rather than repeatedly rebuilt on critical paths. Across Llama-3.2-3B, Qwen3-4B, and Qwen3-30BA3B on phone, PC, and server platforms, SMEPilot improves end-to-end inference performance by up to 3.94$\times$. 

---
# ArtBoost: Synthetic Articulatory Data Augmentation for Acoustic-to-Articulatory Inversion 

**Authors**: Hyung Kyu Kim, Byungchan Hwang, Hak Gu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.16327)  

**Abstract**: Recent acoustic-to-articulatory inversion (AAI) models rely on electromagnetic articulography (EMA) data, which are costly and limited in scale. To address this limitation, we propose \textit{ArtBoost}, a novel data augmentation strategy that leverages large-scale speech--mesh datasets originally developed for speech-driven 3D facial animation to improve AAI under limited EMA supervision. \textit{ArtBoost} extracts pseudo articulatory trajectories from visible facial anchors and uses them for pre-training before fine-tuning on real EMA data. Experiments show consistent improvements in PCC and RMSE. Trajectory analyses confirm that the pseudo articulatory signals reflect physically meaningful visible articulatory dynamics. Additional evaluations across different AAI architectures demonstrate stable performance gains, indicating that \textit{ArtBoost} can be integrated into diverse AAI models. These results suggest that speech--mesh data provide an effective and scalable source of articulatory supervision for AAI. Project page: this https URL 

---
# Gaming-Resistant Insurance Contracts for Autonomous AI Agents: Strategy-Proof Toll Mechanism Design 

**Authors**: Hao-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.16326)  

**Abstract**: Paper A defines a time-consistent actuarial runtime that prices each side-effect-bearing action against a contractually fixed safe default and gates execution against a reserve budget. It treats the operator as passive. This paper makes the operator strategic. We characterise a five-attack space for autonomous AI-agent insurance contracts and prove when the actuarial runtime is gaming-resistant. Two attack surfaces -- post-toll safe-default selection and within-boundary action splitting -- are closed by Paper A's minimal-authority and no-splitting clauses. The remaining three require new contract clauses. First, common-control aggregation prevents cross-boundary re-routing from reducing toll below the boundary potential applied to total exposure. Second, interface failures such as invalid JSON are contract-relevant events, not safety wins: treating them as zero-toll safe defaults can reward unreliable models, while escalation fees reverse the incentive. We validate this interface-compliance theorem on committed cross-model traces from the companion empirical paper. Third, a model-identity menu with a componentwise-minimum penalty schedule makes truthful reporting of the deployed model weakly dominant. We then compose these clauses with Paper A's runtime guarantees to obtain joint incentive compatibility over the five-attack space. Finally, a two-parameter premium family discharges operator individual rationality and weak budget balance at the truthful equilibrium. The result is an incentive-compatibility layer for actuarial control of autonomous-agent side effects. 

---
# RL-Index: Reinforcement Learning for Retrieval Index Reasoning 

**Authors**: Yongjia Lei, Nedim Lipka, Zhisheng Qi, Utkarsh Sahu, Koustava Goswami, Franck Dernoncourt, Ryan A. Rossi, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16316)  

**Abstract**: Retrieving external knowledge is essential for solving real-world tasks, yet it remains challenging when the relationship between a query and its relevant knowledge involves implicit and complex reasoning beyond surface-level semantic or lexical matching (e.g., mathematical problems relying on the same theorem or coding requiring deep reasoning). Existing approaches primarily rely on query-side reasoning (e.g., query rewriting), which introduces significant online latency and underutilizes the opportunity to perform reasoning over the knowledge corpus itself (i.e., index-side reasoning). In this paper, we propose RL-Index, an agentic indexing framework that formulates retrieval index reasoning as a reinforcement learning problem. Instead of performing reasoning at query time, RL-Index shifts reasoning to the indexing stage by augmenting documents with LLM-generated rationales that explicitly encode the latent query-knowledge relationship. To optimize the quality of these rationales, we employ Group Relative Policy Optimization (GRPO) and use retrieval similarity as a verifiable reward signal, enabling direct optimization of indexing decisions for retrieval effectiveness. Extensive experiments on the BRIGHT benchmark demonstrate that RL-Index consistently improves both retrieval and downstream question-answering performance, while significantly reducing online inference latency. Moreover, the learned rationale augmentation generalizes across diverse retrievers and generators, highlighting its robustness as a plug-and-play indexing strategy across different retrieval systems. 

---
# Is Your Trajectory Displacement Safe in Long-tail? 

**Authors**: Qiao Sun, Weicheng Zheng, Yixin Huang, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.16313)  

**Abstract**: Long-tail scenarios remain a major bottleneck for autonomous driving evaluation, even as datasets grow by orders of magnitude. Existing evaluation pipelines are rarely human-aligned, safety-aware, verifiable, and explainable at the same time: closed-loop metrics often saturate among strong planners, while unstructured human ratings can be noisy without a carefully designed protocol. We formulate planning evaluation as additional-threat detection: given a planner trajectory and an expert reference, does the planner's displacement introduce new unsafe driving behavior? We propose FluidTest, an evaluation pipeline with three components: a pairwise WebUI protocol for reliable human annotation; a taxonomy of 32 semantic threats with evidence-grounded decision graphs; and a three-agent verification system with reflection for precision and auditability. Experiments on the WOD-E2E dataset show that FluidTest produces consistent labels among trained annotators and identifies additional threats in 65% of Poutine trajectories and 51% of RAP trajectories. These results show that state-of-the-art planners can still exhibit substantial safety-relevant failures despite high Rater Feedback Scores (RFS) and low Average Displacement Error (ADE). Additional details, guidance, and code are available at this https URL. 

---
# AI Supply Chain Galaxy: 3D Visual Analytics for License Compliance 

**Authors**: Weiru Han, Xuetao Shi, Wenyi He, Wei Wang, Rui Zhao, Moming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16292)  

**Abstract**: The rapid proliferation of machine learning model reuse has transformed the AI ecosystem into a highly interconnected supply chain. Traditional compliance tools and static reports struggle to navigate these massive, multi-hop dependency networks. To address this, we present AI Supply Chain Galaxy (AISCG), an interactive 3D visual analytics system for model provenance and compliance auditing. AISCG maps models into a 3D spatial layout, integrating explicit structural dependencies with a rule-based compliance engine. It supports multi-scale exploration, from global community detection to localized, path-aware lineage tracing. We demonstrate its efficacy through an ecosystem-scale empirical analysis of 908,449 models from Hugging Face. Our findings reveal a concerning landscape: 55.46% of models exhibit compliance risks or metadata conflicts/omissions. We also identified distinct risk patterns, including a 56.67% license omission rate in adapter derivations and an 8.05% "license drift" rate in fine-tuning. Through a case study on the complex Llama model family, we show how AISCG empowers analysts to intuitively trace inherited restrictive terms and identify root causes across deep topological networks, significantly reducing the cognitive load of compliance auditing. 

---
# An affordable hardware-aware neural architecture search for deploying convolutional neural networks on ultra-low-power computing platforms 

**Authors**: Andrea Mattia Garavagno, Edoardo Ragusa, Antonio Frisoli, Paolo Gastaldo  

**Link**: [PDF](https://arxiv.org/pdf/2606.16290)  

**Abstract**: Hardware-aware neural architecture search (HW-NAS) allows the integration of Convolutional Neural Networks (CNNs) in microcontrollers devices by automatically designing neural architectures that can fit prearranged hardware constraints. However, state-of-the-art HW-NAS target high-performance microcontrollers, whose power consumption does not meet sensing nodes requirements. This work presents a HW-NAS generating tiny CNNs that can run on ultra-low-power microcontrollers, featuring a lightweight search procedure enabling its execution even on embedded devices. Empirical results on three well-known benchmarks for tiny computer vision proved that the proposed HW-NAS was able to generate tiny CNNs while preserving state-of-the-art classification accuracy. 

---
# FlowMPC: Improving Flow Matching policies with World Models 

**Authors**: Chandon Hamel  

**Link**: [PDF](https://arxiv.org/pdf/2606.16286)  

**Abstract**: Flow Matching (FM) is a powerful approach for behavior cloning in multimodal action spaces [Jiang et al., 2025], but because it is not trained to directly maximize expected return, there is still room to improve how FM policies act at test time. This work investigates whether a learned world model can improve FM policies by enabling Model Predictive Path Integral (MPPI) planning over candidate action sequences proposed by the policy. Building on TD-MPC2 [Hansen et al., 2024], I introduce FlowMPC, a framework that combines an imitation-learned FM policy with a learned world model for test-time planning in ManiSkill manipulation tasks [Tao et al., 2025]. Across PickCube and PickSingleYCB, adding the world model improved performance over the FM policy alone, with especially clear gains in end-of-episode success. These results suggest that world-model-based planning can effectively complement flow-based imitation policies without modifying the FM training objective. 

---
# Who Should Lead Decoding Now? Tracking Reliable Trajectories for Ensembling Masked Diffusion Language Models 

**Authors**: Heecheol Yun, Joonhyung Park, Joowon Kim, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16281)  

**Abstract**: Masked Diffusion Language Models (MDLMs) have emerged as a distinct paradigm for sequence generation. As MDLMs become diverse in capabilities and knowledge coverage, an important question is how to combine their knowledge. Toward this, we first investigate the unique decoding dynamics of MDLMs. We find that successful generations exhibit stable confidence dynamics over answer-relevant positions, while unreliable trajectories can often be corrected by injecting promising intermediate states from other models. Guided by this observation, we propose $\textbf{TIE}$ ($\textbf{T}$rajectory-based $\textbf{I}$terative $\textbf{E}$nsembling), a knowledge fusion framework in which MDLMs iteratively identify reliable decoding trajectories and relay them across models. TIE tracks confidence dynamics over answer-relevant positions to determine which model currently follows a more reliable trajectory and selectively transfers partially denoised sequences across models. As the model on the more promising trajectory often changes across denoising steps, TIE allows different models to contribute complementary strengths at different stages of generation. Strong performance across diverse reasoning tasks, along with our analyses, suggests that TIE offers a practical approach to the underexplored problem of MDLM ensembling. 

---
# RealityBridge: Bridging Editable 3D Gaussian Splatting Driving Simulations and Real-World Videos 

**Authors**: Zhenhua Wu, Yun Pang, Mingkun Chang, Yuwei Ning, Liangzhi Wang, Yi Xiao, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.16278)  

**Abstract**: Long-tail hazardous scenarios are essential for safety-oriented autonomous driving, yet they are difficult to collect and reproduce at scale. Editable 3D Gaussian Splatting (3DGS) simulation offers a promising alternative by reconstructing real driving scenes and supporting controllable scene editing. However, edited 3DGS-rendered videos still suffer from a significant Sim-to-Real gap, including rendering artifacts, degraded foreground assets, inconsistent illumination, and temporal flickering. Existing restoration and video generation methods are insufficient for this task, as they often fail to jointly repair 3DGS-specific artifacts, improve visual realism, and ensure temporal consistency. To fill this gap, we propose RealityBridge, a structure-preserving and asset-aware Sim-to-Real framework for edited 3DGS driving videos. RealityBridge uses multimodal controls, including rendered videos, foreground masks, edge maps, and semantic masks, together with a lightweight GateNet for adaptive condition allocation across backbone layers. We further construct targeted training data and introduce autoregressive long-video training with reward-guided post-training to improve restoration quality, temporal stability, and hallucination suppression. Extensive experiments on internal and public driving datasets show that RealityBridge outperforms existing methods in artifact removal, illumination harmonization, and long-sequence temporal consistency. 

---
# UXBench: Measuring the Actionability of LLM-Generated UX Critiques 

**Authors**: Wenjie Wang, Yue Huang, Zipeng Ling, Han Bao, Hang hua, Xiaonan Luo, Yu Jiang, Shiyi Du, Yuexing Hao, Xiaomin Li, Yuchen Ma, Dianzhuo Wang, Yanfang Ye, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16262)  

**Abstract**: Large language models (LLMs) are increasingly deployed as UX judges that inspect interfaces, diagnose usability problems, and propose repairs. Yet no controlled benchmark measures whether the resulting critiques are reliable and actionable across heterogeneous product surfaces. We introduce UXBench, a benchmark for evaluating LLMs as interaction-grounded UX judges. UXBench comprises local-first runnable web fixtures spanning ten product-surface families, paired with coverage-gated browser exploration that forces models to collect interaction evidence before reporting. Each judge model produces a structured UX report over seven rubric dimensions; report quality is measured by whether a fixed downstream repair agent can improve the interface based on the critique. We evaluate eight frontier models under both an automated repair-lift protocol and a blind human validation study. Results show that UX judging is neither saturated nor one dimensional: models differ meaningfully in report actionability, exhibit distinct rubric-level repair signatures, vary in fixture-level reliability, and trade leadership across surface categories 

---
# Variance Reduction for Non-Log-Concave Sampling with Applications to Inverse Problems 

**Authors**: M. Berk Sahin, Ahmet Ege Tanriverdi, Behzad Sharif, Abolfazl Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16257)  

**Abstract**: Sampling from high-dimensional, non-log-concave distributions with unnormalized densities is a fundamental challenge in machine learning, particularly when the exact gradient of the potential is unavailable and must be approximated via stochastic gradients that exhibit high variance under a fixed budget of gradient computations per iteration. Although variance reduction techniques such as SGD with momentum, STORM, and PAGE have demonstrated improved convergence properties in non-convex optimization, their implications for sampling from non-log-concave distributions remain largely unexplored. In this work, we develop the first unified analysis of these estimators for sampling from non-log-concave distributions. We establish improved non-asymptotic convergence rates in $\varepsilon$-relative Fisher information and, under a Poincaré inequality assumption, in squared total variation distance, and further prove weak convergence to the target distribution. We extend our analysis to solving inverse problems with score-based generative priors. We empirically validate our theory and demonstrate that, under a fixed gradient computations per iteration, variance-reduction techniques consistently improve sample quality in two standard imaging applications. 

---
# Learned Image Compression for Vision-Language-Action Models 

**Authors**: Hyeonjun Kim, Jegwang Ryu, Sangbeom Ha, Junhyeok Lee, Jun-Hyuk Kim, Hyemin Ahn, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.16253)  

**Abstract**: Vision-language-action (VLA) models increasingly rely on high-frequency multi-camera observations, making visual communication a major bottleneck for real-time robotic control in bandwidth-constrained or distributed deployment settings. Existing image and video codecs, however, are designed to preserve generic visual fidelity rather than the control performance of downstream VLA policies. In this work, we introduce SPARC (SPatially Adaptive Rate Control), a learned image compression framework tailored for VLA-driven robots. Our key observation is that the importance of visual information varies substantially across both camera views and spatial regions within an image. Based on this observation, SPARC employs a lightweight temporal mask selector that adaptively allocates bitrate over latent representations according to task relevance while leveraging temporal context. We further introduce a tilted rate loss that stabilizes training by reducing the tendency of entropy-based objectives to over-suppress rare yet task-critical visual patterns. Experiments on diverse robotic benchmarks, including RoboCasa365, VLABench, and LIBERO, show that SPARC consistently achieves stronger control performance than conventional image/video codecs and recent learned compression methods under the same bitrate budget. We additionally demonstrate real-world deployment benefits in remote-control settings, where our method substantially improves the bitrate-success tradeoff. 

---
# Data Augmentations for Data-Constrained Language Model Pretraining 

**Authors**: Michael K. Chen, Xikun Zhang, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16246)  

**Abstract**: As AI labs approach a data ceiling where compute capacity outpaces the rate of new high-quality text generation, language model pretraining is shifting toward a data-constrained, compute-abundant regime that demands productive multi-epoch training on fixed corpora. Standard autoregressive (AR) pretraining overfits severely in this setting, reaching its optimum early and then continuously deteriorating. We investigate data augmentation as a regularizer to mitigate this overfitting and enable productive training for hundreds of epochs on the same data. We introduce three orthogonal categories of augmentation for AR pretraining: token-level noise (masking, random replacement), sequence permutations (right-to-left prediction, Fill-in-the-Middle), and target offset prediction ($x_{t+i}$ for $i > 1$). Through systematic ablations, we find that individual augmentations delay overfitting and lower validation loss relative to the baseline, with random token replacement achieving the best minimum loss among individual methods. Combining augmentation categories further lowers the minimum validation loss. Our experiments demonstrate that data augmentations mitigate AR pretraining's data inefficiency and offer a promising solution to the data-constrained regime. All code and data are available at this https URL 

---
# SPARK: Security Knowledge Priming and Representation-Guided Knowledge Activation for LLM-based Secure Code Generation 

**Authors**: Xiaoyun Xu, Lichao Wu, Jona te Lintelo, Siyu Zhang, Stjepan Picek  

**Link**: [PDF](https://arxiv.org/pdf/2606.16244)  

**Abstract**: Large language models routinely generate code with exploitable security flaws. Prior literature attributes this limitation to a lack of security expertise, steering current defense mechanisms toward heavy fine-tuning or external knowledge retrieval, which introduces significant computational overhead and data bias through redundant code examples. Contrary to this view, we argue that pretraining corpora are already rich in security material. The bottleneck is activation: without an explicit and brief cue, statistical pressure toward common training-distribution patterns suppresses the model's safety-relevant representations. We present SPARK, an inference-time security harness that activates this latent knowledge without any retraining. The harness has two parts. Component~I retrieves a few of the relevant Common Weakness Enumeration (CWE) entries for each coding task and appends a short structured cue to the prompt; this alone is enough to surface the model's existing security representations. Component~II adds a precomputed token bias to the logits at every decoding step. We obtain the bias by projecting a safe-direction vector, the unit difference between the mean safe and mean unsafe last-layer hidden states, through the language model head. The bias is computed once offline; applying it costs a single vector addition per generated token. We evaluate SPARK on 9 open-source models across C++, Java, and Python, and compare with 7 baselines spanning fine-tuning and retrieval-augmented methods. SPARK matches or improves on the best baseline in every setting while preserving HumanEval utility. We further test Component~I in a black-box setting on 7 of today's strongest models, including Claude, DeepSeek, and GPT, demonstrating the bottleneck of insecure code generation and the improvements enabled by our method. 

---
# Propagating Structural Guidance: Synthesizing Fluorescein Angiography from Fundus Images and Sparse OCT Scans 

**Authors**: Tengfei Ma, Ruiqi Wu, Chenran Zhang, Ye Geng, Na Su, Xiangyuan Duanmu, Tao Zhou, Yi Zhou, Wen Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.16234)  

**Abstract**: Fundus fluorescein angiography (FFA) is critical for assessing retinal vascular abnormalities, but its acquisition is invasive and not always feasible. In contrast, color fundus photography (CFP) is non-invasive and widely accessible, which has motivated studies on CFP-to-FFA synthesis. However, prior works rely solely on CFP surface texture, fundamentally limiting the ability to reconstruct functional vascular information and subtle pathological changes. To address this, we propose a novel framework that synthesizes FFA from CFP with structural guidance provided by optical coherence tomography (OCT). We construct a multi-modal retinal imaging dataset with paired CFP, FFA, and OCT from 3,676 patient eyes--the first tri-modally aligned dataset in retinal imaging. To bridge the spatial gap between OCT and fundus modalities, we propose a Spatially Aligned Cross-Modal Fusion (SACMF) module that projects depth-resolved OCT features onto the fundus plane and injects them into the CFP encoder via adaptive layer normalization. Beyond feature fusion, we further introduce Token-wise Cross-Modality Alignment (TCMA), a token-level contrastive learning strategy that explicitly aligns CFP and FFA representations at corresponding spatial positions. Our method achieves superior synthesis performance compared to state-of-the-art methods. Moreover, extensive experiments demonstrate that the FFA images synthesized by our approach bring greater improvements in downstream disease diagnosis performance than existing methods, highlighting the clinical potential of our approach as a non-invasive decision-support tool in routine workflows. The code is available at this https URL. 

---
# From Tokens to Regions: CUDA-Sensitive Instruction Tuning for GPU Kernel Generation 

**Authors**: Wentao Chen, Jiace Zhu, Xing Zhe Chai, Zeng Qu, Qiaoling Xiao, Liucheng Duan, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2606.16231)  

**Abstract**: High-performance CUDA kernels are essential for scalable AI systems, while Large Language Models (LLMs) still struggle to generate correct kernels due to strict and implicit execution constraints. Existing LLM-based approaches either rely on costly agentic or reinforcement-learning (RL) pipelines, or adopt supervised fine-tuning (SFT) objectives that fail to explicitly model CUDA sensitivity, namely code tokens or regions tightly coupled with execution constraints. In this work, we investigate CUDA sensitivity from the perspective of token confidence patterns, showing that CUDA sensitivity appears at both token and region levels, where most CUDA-sensitive tokens are predicted with high confidence, while a smaller low-confidence subset forms regions corresponding to execution-critical structures. These findings suggest that effective CUDA kernel generation should both leverage high-confidence CUDA-sensitive tokens and preserve low-confidence CUDA-sensitive regions. Building on these insights, we propose \textbf{\underline{CU}DA-\underline{Se}nsitive Instruction \underline{T}uning (CuSeT)}, a low-cost post-training method within a simple SFT framework. CuSeT follows the principle of ``from tokens to regions'' by combining \emph{adaptive token-level masking} with \emph{region-aware sample reweighting}. Experiments show that CuSeT consistently improves functional correctness across multiple model families and scales, outperforming standard SFT and advanced SFT variants, while achieving competitive performance against frontier CUDA kernel generation models with substantially lower inference cost. 

---
# PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents 

**Authors**: Zhenbang Du, Jun Luo, Zhiwei Zheng, Xiangchi Yuan, Kejing Xia, Dachuan Shi, Qirui Jin, Qijia He, Shaofeng Zou, Yingbin Liang, Wenke Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.16215)  

**Abstract**: Multi-turn tool-use agents must reason, call tools, and adapt to observations across several interaction turns. Post-training such agents is challenging, as reinforcement learning often suffers from sparse rewards and weak credit assignment despite matching the prompt-only inference setting, while supervised fine-tuning on expert traces provides dense process supervision but can over-constrain the model to fixed trajectories. To tackle this, we propose PACT, a Privileged trAce Co-Training framework for multi-turn tool-use agents. The key idea is to use expert traces only as training-time optimization signals rather than rollout-time hints. PACT keeps rollout generation prompt-only, then uses expert traces to guide optimization through two complementary signals: a trace-conditioned RL surrogate that evaluates prompt-only rollouts under expert-trace context, and a component-aware SFT loss that supervises reasoning prefixes and tool-calls with annealed strength. To reduce over-reliance on the training-only trace context, PACT further introduces a prompt-only anchoring. We also provide a latent-trace view that connects the two trace-based objectives and explains how expert traces can guide optimization without being used during rollout generation. Experiments on FTRL, BFCL, and ToolHop show that PACT consistently improves over strong SFT- and RL-based baselines, highlighting the value of privileged trace co-training for multi-turn tool-use learning. 

---
# Calibrated Sampling-Free Uncertainty Estimation in Bayesian Deep Learning 

**Authors**: Tobias Jan Wieczorek, Leon de Andrade, Thomas Möllenhoff, Marcus Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2606.16214)  

**Abstract**: Modern deep learning models remain notoriously prone to overconfidence, limiting their reliability in high-stakes applications. Bayesian methods aim to counter this by learning a distribution over model parameters, and recent advances now make this feasible for large-scale architectures at costs comparable to AdamW. However, a challenge remains at test time: predictions must be averaged across many forward passes with weights sampled from the posterior, which is prohibitively expensive. Variance propagation offers an efficient alternative, computing layer-wise analytical approximations of uncertainty in a single forward pass. While such techniques are effective for MLPs, their extension to modern architectures remains challenging, due to increased depth and diversity of layer types. To fill this gap, we propose Calibrated Variance Propagation (CVP), which introduces a new propagation method for normalization layers, combines it with recent techniques for handling activation functions, and absorbs residual error through a light calibration step. CVP yields comparably accurate uncertainty estimates to MC sampling across transformers and CNNs, at a fraction of the cost. Against prior variance propagation work, CVP improves coverage at $0.5\%$ risk from $8.2\%$ to $14.6\%$ with BEiT-3 on Visual Reasoning (NLVR2) and from $2.6\%$ to $10.8\%$ with ViLT on VQAv2, with gains extending to convolutional architectures. 

---
# LUCID: Learned Undersampling-Adaptive Consistency-Guided Inference with Deterministic Flow Matching for Sparse-View CT Reconstruction 

**Authors**: Jigang Duan, Jiayi Wang, Heran Wang, Ping Yang, Genwei Ma, Xing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.16212)  

**Abstract**: Sparse-view CT reduces radiation dose and scanning time by acquiring fewer projection views, but angular undersampling makes reconstruction severely ill-posed, causing streak artifacts, structural blurring, and loss of fine details. Existing supervised methods are often tied to specific sampling settings, whereas generative methods may introduce anatomically inconsistent hallucination-like structures under severe undersampling. We propose Lucid, a sparsity-adaptive, consistency-guided reconstruction framework based on a Flow Matching generative prior for sparse-view CT. Lucid is trained only on high-quality CT images to learn a continuous transport between a Gaussian distribution and the high-quality CT image distribution, independent of view sampling. During inference, the sampling sparsity level is explicitly incorporated to adapt the generative trajectory of a single pretrained model. Specifically, Lucid constructs a degradation-matched initial state by sparsity-weighted fusion of the sparse-view FBP image and Gaussian noise, performs sparsity-modulated Flow Matching updates, and applies projection-domain data-consistency correction after each prior update. Experiments under multiple sparse-view settings show that Lucid achieves stable reconstruction performance across different sampling densities, improves image quality and structural fidelity, and reduces the risk of hallucination-like structures in generative sparse-view CT reconstruction. 

---
# EgoPhys: Learning Generalizable Physics Models of Deformable Objects from Egocentric Video 

**Authors**: Hyunjin Kim, Ri-Zhao Qiu, Guangqi Jiang, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16202)  

**Abstract**: Humans naturally understand object physics through everyday interactions, but faithfully predicting complex deformable dynamics, such as elastic materials and fabrics, remains a major challenge for computer vision and robotics. We present EgoPhys, a framework that constructs deformable physical digital twins from egocentric RGB-only video using generalizable priors. EgoPhys overcomes the limitations of existing methods to enable controllable deformable digital twin generation from egocentric videos by distilling per-object inverse-physics solutions into a compact codebook, enabling prediction of dense spring stiffness fields for unseen objects without per-spring test-time optimization. Trained with generalizable priors from diverse egocentric interactions, EgoPhys outperforms baselines in reconstruction, future prediction, and zero-shot generalization. To support training and evaluation, we curate an egocentric interaction dataset covering diverse deformable objects, scenes, and manipulation styles. We deploy EgoPhys on a real xArm6 robot, demonstrating that a digital twin initialized from a single egocentric human play video can serve as an internal world representation to aid in deformable-object planning, highlighting egocentric RGB observations as a scalable path toward real-to-sim pipelines. 

---
# Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs 

**Authors**: Yusong Zhao, Hengyi Wang, Tanuja Ganu, Akshay Nambi, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16193)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance on vision-language tasks, yet their internal visual representations remain difficult to interpret. Sparse Autoencoders (SAEs) provide a scalable way to decompose dense model activations into sparse, interpretable features. However, existing SAE architectures primarily recover flat feature dictionaries and are less suited for explicit multi-level concept organization. In this paper, we introduce cascaded sparse autoencoders (CSAEs) for learning hierarchical visual concepts in MLLMs. Rather than nesting or stacking SAE sparse activation codes, CSAEs train a second-level SAE directly on the decoder weights of the first-level SAE, treating learned low-level feature directions as inputs for higher-level abstraction. This design enables CSAEs to learn "concepts of concepts" while avoiding drawbacks from the shared-prefix coupling of nesting, Matryoshka-style hierarchies and the bottlenecks of naively stacked SAEs. Experiments across Qwen3-VL, Gemma-3, and LLaVA on multiple visual datasets show that CSAEs improve interpretability in terms of hierarchical concept coherence over state-of-the-art SAE baselines. Results on concept steering further demonstrate that the learned concept groups support effective group-level interventions in MLLM outputs. 

---
# Embedded Arena: Iterative Optimization via Hardware Feedback 

**Authors**: Zhihan Zhang, Alexander Le Metzger, Jiuyang Lyu, Chun-Cheng Chang, Jiayi Shao, Yujia Liu, Emmanuel Azuh Mensah, Edward Wang, Kurtis Heimerl, Gregory D. Abowd, Shwetak Patel, Natasha Jaques, Vikram Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.16190)  

**Abstract**: Embedded devices from wildlife monitoring stations to clinical wearables require local AI inference due to latency, communication, or privacy constraints. Optimizing models for heterogeneous microcontrollers (MCUs) requires simultaneously satisfying hard physical constraints on memory, power, and temperature while preserving accuracy, a multidimensional optimization that is today performed manually by experts. We ask whether an LLM agent can autonomously navigate this complex, multi-turn pipeline guided by real hardware feedback, and introduce a hardware-in-the-loop agent arena in which the agent iteratively refines both model and firmware -- compiling, flashing, and measuring on real hardware -- to enable closed-loop optimization. Frontier models, including Claude Opus 4.7 and Gemini 3.1 Pro, fail entirely without hardware feedback (0% deployment success), whereas our hardware-in-the-loop formulation achieves the first successful deployment within three iterations and can surpass human expert results within seven. This agentic co-optimization achieves 250x compression for vision models with <3.3% accuracy loss and 400x for audio with <6% Feature Error Rate loss, enabling battery-free operation on a commercial MCU via solar harvesting. We demonstrate practical impact in two real-world systems: an elk-detection camera trap (96.7% accuracy) and a phonetic-transcription wearable (8.44% FER) for child development research. 

---
# LLM-Powered Virtual Population for Demand Simulation and Pricing 

**Authors**: Chengpiao Huang, Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16183)  

**Abstract**: We develop an LLM-powered virtual population model that simulates demand for pricing decisions, in settings where products are described by rich unstructured information, such as text descriptions and images, and where decision makers need not only mean-demand predictions but also uncertainty estimates for counterfactual prices. Our model represents exposed customers as draws from a finite mixture of customer personas. For each persona, product, and candidate price, an LLM elicits a persona-level purchase probability using both structured persona information and unstructured product information. These probabilities are aggregated through calibrated mixture weights to form a predictive distribution of aggregate demand. The resulting simulator can evaluate counterfactual prices under various pricing objectives, including expected revenue and risk-aware criteria such as conditional value at risk. We test the framework on an online H&M fashion dataset with product descriptions and images. The calibrated LLM-based simulator achieves the best overall predictive performance among the models considered, and supports sample-efficient pricing decisions. Our framework provides a practical way to use LLMs as demand simulators for products with limited historical demand data but rich product information. By producing a full predictive demand distribution rather than only a point forecast, it enables managers to compare candidate prices, quantify demand uncertainty, and choose prices that target either average-case revenue or risk-aware objectives. 

---
# A comparative and critical study of EEGNet for fNIRS-driven cognitive load classification 

**Authors**: Mehshan Ahmed Khan, Houshyar Asadi, Li Zhang, Mohammad reza Chalak Qazani, Ghazal Bargshady, Stefanos gkikas, Christian arzate, Sam Oladazimi, Zoran Najdovsk, Lei Wei, Chee Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.16160)  

**Abstract**: Accurately classifying cognitive load from functional near-infrared spectroscopy (fNIRS) signals remains a significant challenge due to temporal variability, inter-subject differences, and sensitivity to preprocessing choices. This study provides a comprehensive evaluation of EEGNet for fNIRS-based cognitive load classification by systematically examining the effects of temporal segmentation strategies (overlapping vs. non-overlapping), window lengths (10s, 20s, 30s), feature extraction methods (Analysis of Variance (ANOVA), Principal Component Analysis (PCA), Fast Independent Component Analysis (FastICA)), learning rate configurations (fixed and adaptive), and evaluation protocols (random split vs. subject-independent (SI)). Results from random-split experiments show that overlapping segmentation, combined with smaller fixed learning rates (0.01-0.001), yields the highest accuracies, due to temporal redundancy and dense sampling of hemodynamic transitions. However, SI evaluation reveals a substantial drop in accuracy, demonstrating limited generalization to unseen participants. Under SI evaluation, non-overlapping segmentation outperformed overlapping windows, with the best accuracy of 56.11% achieved using PCA features with a 20-second window and a 0.1 learning rate. These findings indicate that eliminating temporal redundancy helps the model learn more robust and generalizable representations of cognitive load across individuals. Although adaptive learning rate strategy improved training stability, it did not surpass the performance of optimally selected fixed learning rates. The study highlights the critical role of segmentation strategy and learning rate selection in improving model generalization and identifies methodological considerations essential for developing reliable, real-time, and SI cognitive load classification systems using fNIRS. 

---
# A Comprehensive Survey of Medical Image Segmentation: Challenges, Benchmarks, and Beyond 

**Authors**: Pengyu Zhu, Xiaojing Zhang, Kunbo Zhang, Chunyan Zhang, Zhenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16153)  

**Abstract**: Medical image segmentation plays a critical role in clinical diagnostics, treatment planning, disease monitoring, and neurological disorder identification. This article presents a comprehensive review of its systematic development, covering widely used public datasets, representative methods built on the U-Net, Transformer, and SAM architectures, and key evaluation metrics with their differences, followed by an analysis of major challenges from multiple perspectives. Unlike surveys that focus on a single model family or a specific clinical application, this review organizes U-Net-, Transformer-, and SAM-based methods within a unified analytical framework, with a particular focus on their effectiveness in improving segmentation accuracy and efficiency. This work aims to guide future research and support clinical translation of medical image segmentation, with all related resources publicly available in our GitHub repository: this https URL. 

---
# XAI-Grounded Explanation Generation for Speech Deepfake Detection with Training-Free Multimodal Large Language Models 

**Authors**: Yupei Li, Qiyang Sun, Xiaoliang Wu, Chenxi Wang, Berrak Sisman, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2606.16137)  

**Abstract**: Speech deepfake detection (SDD) systems require trustworthy explanations for reliable decision-making. Existing explanation ways mainly fall into two categories. Traditional explainable AI (XAI), such as gradient-based attribution, produces low-level attribution signals tightly coupled with model decisions, and harder to be understood by human than natural language explanations. Meanwhile, large language model (LLM)-based explanation generation often produces generic and ungrounded descriptions due to the lack of heuristic evidence and task-specific supervision, stemming from limited grounded explanation datasets for SDD. We therefore propose a training-free explanation framework that integrates XAI evidence with multimodal LLMs to generate grounded and specific explanations. Using the PartialSpoof dataset, we construct a grounded explanation dataset and show that methods with XAI increase inside accuracy by over 45\%, verified through human evaluation and faithfulness checks. 

---
# InvDesMobility: a reliability-gated first-principles feedback framework for closed-loop materials discovery 

**Authors**: Wen-Kao Li, Ze-Feng Gao, Peng-Jie Guo, Wei Ji, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16133)  

**Abstract**: Inverse materials design starts from target functionality and searches for structures that can realize it. Its value in closed-loop discovery depends not only on prediction performance, but also on whether expensive first-principles results are independently validated, provenance-recorded, and admitted as feedback only when evidence is sufficient. This is especially important for composite properties such as carrier mobility, where a final scalar value hides intermediate quantities, fit quality, convergence history, and workflow assumptions. Here we present InvDesMobility, a reliability-gated first-principles feedback framework that integrates multi-agent automated DFT, evidence stratification, generative structure proposal, acquisition ranking, and auditable release. Using 516 2DMatPedia-derived candidates, the workflow produced 280 QC-passed materials and 573 retained carrier-direction seed channels after channel-level reliability gating. These records were split into two feedback objects: relaxed structures updated the generative model, while retained mobility channels trained the acquisition model and set validation priority. Over multiple iterations, InvDesMobility screened 2.4 x 10^6 structures, submitted 102 candidates for DFT validation, and retained 86 reliability-gated generated channels across 41 formulas. Overall, the main contribution is not a fixed list of high-mobility materials, but a transferable feedback contract that makes closed-loop inverse design both useful and auditable when learning from expensive calculated properties. All source data, retained feedback records, and workflows are available at this https URL, with an accompanying evidence website at this https URL. 

---
# AuAu: A Benchmark for Auditing Authoritarian Alignment in Large Language Models 

**Authors**: Andreas Einwiller, Max Klabunde, Florian Lemmerich  

**Link**: [PDF](https://arxiv.org/pdf/2606.16127)  

**Abstract**: The worldwide surge of authoritarianism, combined with the increasing central role in users' everyday lives, raises the question of to what extent specific models exhibit or promote authoritarian attitudes and characteristics. We introduce AuAu, a comprehensive benchmark that aims to assess the risk of LLMs generating responses with authoritarian tendencies. This benchmark combines three evaluation approaches: (i) psychometric questions from an extensive pool of 15 human validated instruments; (ii) contextual behavior vignettes probing intended actions in concrete situations; and (iii) responses to realistic user prompts. Unlike prior work, AuAu evaluates not only a general closeness towards authoritarianism but also the established sub-concepts Authoritarian Aggression, Authoritarian Submission, and Conventionalism. Evaluating 17 models from China, the EU, Russia, and the USA, we find that all tested models exhibit substantial authoritarian response rates under the psychometric evaluation, though rates drop significantly in increasingly more realistic downstream task. We further find that an authoritarian system prompt easily manipulates 15 out of 17 models to promote increased authoritarianism. Our results underscore the need for continued, systematic auditing of LLM-based AI systems to detect and ultimately mitigate undesired authoritarian tendencies in generated output. Our code and data are available at: this https URL 

---
# Scaling Adaptive Depth with Norm-Agnostic Residual Networks 

**Authors**: Tomás Figliolia, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2606.16112)  

**Abstract**: Residual architectures are ubiquitous in deep learning, but they suffer from a subtle structural limitation: the norm of the residual stream can grow rapidly with depth. As a result, updates from later layers become small relative to the accumulated residual state. This reduces their impact on the representation and limits the benefits of scaling models in depth. To address this, we introduce NAG, a norm-agnostic residual architecture that separates magnitude from directional information in the residual stream, preserving meaningful layer contributions throughout depth and preventing later updates from being systematically suppressed by residual-norm growth. Importantly, NAG introduces only a negligible number of additional parameters and relies on simple operations that are easily kernel-fusible, preserving training efficiency in practice. We show that this architecture outperforms baseline Transformers, with gains that increase substantially as depth grows, enabling effective training of much deeper models. The norm-agnostic formulation also leads to an interpretable Mixture-of-Depths (MoD) mechanism that adaptively skips both attention and MLP layers. Beyond serving as a post-training accuracy-compute tradeoff, this mechanism can be used as a pretraining-time scaling strategy: under iso-FLOP training, compute saved by reducing per-token forward-pass cost can be reinvested into training on more tokens while keeping the total parameter count and KV-cache budget fixed. In our experiments, moderate Mixture-of-Depths rates of approximately 20%-25% match full-depth baseline performance under equal training compute while substantially reducing the number of executed layer parameters and forward-pass FLOPs. These results identify sparsity in depth as a new scaling axis for fixed-compute training, enabling very deep yet FLOP-efficient models. 

---
# Long-Context Modeling via GSS-Transformer Hybrid Architecture with Learnable Mixing 

**Authors**: Kuzey Torlak, Hüseyin Arda Arslan, Anıl Dervişoğlu, Beyza Nur Deniz, Onur Boyar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16093)  

**Abstract**: Modeling long-range dependencies remains a central challenge in natural language processing. Transformer architectures achieve strong performance via self-attention but scale quadratically ($O(N^2)$) with sequence length, while State Space Models (SSMs) scale linearly ($O(N)$) but suffer from a selective recall bottleneck, struggling to retrieve precise information from compressed states. This creates a fundamental tradeoff between efficiency and perplexity. To tackle these challenges, we propose the \textit{Parallel Hybrid Architecture (PHA)}, which runs Gated State Spaces (GSS), Grouped Query Attention (GQA), and Feed-Forward Networks (FFNs) as independent parallel branches fused by a learnable mixing mechanism. Instead of forcing SSMs to approximate attention or serializing the two paradigms, PHA allows each branch to specialize: GSS captures global context, while attention performs selective retrieval, with FFN providing complementary processing. On WikiText-103, PHA achieves 16.51 PPL at 125M parameters, outperforming Hedgehog (16.70) and H3-125M (23.70). Scaling to 180M parameters yields 16.42 PPL, which gives comparable results with the pure attention baseline while delivering 24\% higher throughput and up to 40\% lower memory usage at long contexts. On OpenWebText, our 125M model achieves 19.72 PPL, outperforming standard Transformers (20.60) and GSS hybrid baselines (19.80). These results demonstrate that separating sequence modeling paradigms into parallel specialists enables Transformer-level perplexity with substantially improved efficiency for long-context language modeling. 

---
# VinQA: Visual Elements Interleaved Long-form Answer Generation for Real-World Multimodal Document QA 

**Authors**: Young Rok Jang, Hyesoo Kong, Kyunghwan An, Jae Sub Huh, Gyeonghun Kim, Stanley Jungkyu Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.16092)  

**Abstract**: Real-world documents combine text with tables, charts, photographs, and diagrams arranged in diverse layouts, yet existing research on multimodal large language models (MLLMs) for document QA predominantly produces text-only responses, underutilizing these visual elements. We introduce VinQA, a dataset for long-form answer generation where cited visual elements are explicitly interleaved with their supporting text and grounded in relevant document pages. To support this task, we study two encoding methods for feeding raw document page images into an MLLM, along with their visual-element citation mechanisms: (1) Page Encoding, which directly encodes full-page images with bounding boxes of visual elements and treats these boxed regions as citable units; and (2) Modality Encoding, which parses each page to extract text and crop visual elements, encodes them separately, and uses these cropped elements as citable units. In our experiments, we propose M-GroSE, a multimodal evaluation framework extending GroUSE to assess answers along four dimensions: completeness, answer relevancy, faithfulness, and unanswerability. We additionally report Visual Source F1 to directly measure visual citation accuracy. Although proprietary frontier models still achieve the best overall scores on the VinQA test split, fine-tuning open Qwen2.5-VL models on the training split substantially improves their performance and narrows this gap. Modality Encoding is initially more robust for complex documents with long text, many visual elements, and diverse citation requirements. After training on VinQA, however, Page Encoding reaches a comparable level, competing effectively even without the explicit parsing used in Modality Encoding. Finally, Visual G-Eval, an MLLM-based judge, confirms that fine-tuned models insert visual elements at semantically appropriate positions with faithful supporting text. 

---
# Tool-IQA: Augmenting Image Quality Assessment with Simple Tools 

**Authors**: Guanyi Qin, Junjie Zhang, Chunming He, Yibing Fu, Jie Liang, Tianhe Wu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16082)  

**Abstract**: Vision-Language Models (VLMs) have been increasingly adopted for Image Quality Assessment (IQA). However, current methods typically employ a static one-shot scoring paradigm, despite the fact that humans assess image quality through dynamic visual inspection, e.g., selectively adjusting views to verify details and subtle artifacts. Specifically, relying solely on a single-pass observation introduces two primary limitations: first, perceiving the image only at a global scale restricts the assessment of finer local details; second, the original intensity distribution of the image may overwhelm the visibility, leading to insufficient inspection of image quality. To address these issues, we propose Tool-IQA, shifting the assessment mechanism from passive scoring to a tool-augmented workflow. In particular, we equip VLMs with simple yet effective view tools: a Magnifier to inspect local details, and a Gamma Corrector to uncover visibility and hidden artifacts. The assessment follows a structured pipeline that consists of an initial observation with rubric notes, a tool-augmented in-depth inspection, and a final quantification for calibrated quality score. Furthermore, to ensure efficient and purposeful tool callings, we introduce a batch-aware training strategy to reward tool interactions that can yield positive contributions rather than simply encouraging usage. Experiments on a variety of IQA benchmarks demonstrate that, with effective tool calling and calibrated assessment, our proposed Tool-IQA significantly outperforms existing state-of-the-art models, e.g., it achieves a PLCC of 0.854 on the challenging CLIVE dataset. 

---
# Phys-JEPA: Physics-Informed Latent World Models for Multivariate Time-Series Forecasting 

**Authors**: Weizhi Nie, Weichao Liu, Honglin Guo, Yuting Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.16076)  

**Abstract**: Multivariate forecasting in physical systems requires models that predict coupled temporal variables while preserving meaningful state evolution. Deep forecasters can fit temporal correlations, and physics-informed models can regularize predictions with scientific constraints, but these directions are often connected only at the decoded-output level. As a result, the hidden predictive state that generates future trajectories may remain statistically useful but physically unstructured. We introduce Phys-JEPA, a physics-informed joint-embedding predictive architecture for multivariate time-series forecasting. Phys-JEPA learns a latent world model in which predictive states are decomposed into physical and residual components, and physical consistency is imposed directly on latent states and latent transitions rather than only on decoded forecasts. This formulation uses known physical variables to organize the representation space while retaining residual capacity for unresolved dynamics. On Jena Climate 2009--2016, Phys-JEPA reduces aggregate MSE from 0.12482 to 0.12273 and temperature MSE from 0.01892 to 0.01831 at H=24. On Traffic, full Phys-JEPA improves aggregate MSE over the supervised baseline across all tested horizons, reducing H=192 MSE from 0.800784 to 0.773873. On Electricity, the best variant depends on horizon: static latent consistency is strongest at H=24 and H=48, while full Phys-JEPA gives the best aggregate and target-variable MSE at H=192. These initial results suggest that moving physics-informed learning from output space to latent predictive state space is a promising direction for interpretable temporal world models. 

---
# PVminerLLM2: Improving Structured Extraction of Patient Voice via Preference Optimization 

**Authors**: Samah Fodeh, Linhai Ma, Ganesh Puthiaraju, Srivani Talakokkul, Afshan Khan, Elyas Irankhah, Sreeraj Ramachandran, Ashley Hagaman, Sarah Lowe, Aimee Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2606.16074)  

**Abstract**: Motivation: Patient-generated text contains critical information on patients' lived experiences, social context, and care engagement, but remains largely unstructured, limiting its use in patient-centered outcomes research. Prior work introduced the PV-Miner benchmark and PVMinerLLM models for structured extraction. However, supervised fine-tuning (SFT) alone struggles with rare, fine-grained, and unevenly distributed errors, particularly in token-critical structured outputs.
Results: We present PVminerLLM2, an improved set of LLMs for structured patient voice extraction that applies preference optimization to address token-critical errors beyond the reach of supervised fine-tuning. Our method introduces (i) a preference objective with token-level gated stabilization term that prevents degradation of absolute token likelihood under preference optimization, and (ii) confusion-aware preference pair construction to better capture low-separation distinctions. We further incorporate token-importance weighting and inverse-frequency reweighing to address token imbalance and class skew. Across multiple model sizes, PVMinerLLM2 consistently outperforms strong baselines, achieving gains of up to 4.43% (Code), 3.50% (Sub-code), and 1.55% (Span), and outperforms baseline LLM trained with existing preference optimization methods.
Availability and Implementation: The supplementary material, code, evaluation scripts, and trained models for PVminerLLM2 are publicly available at: this https URL 

---
# MASCOT-Android: A Curated Dataset and Automated Collection Pipeline for Android Malware Source Code Specimens 

**Authors**: Bojing Li, Duo Zhong, Prajna Bhandary, Raguvir S, Charles Maxa, Robert J Joyce, Charles Nicholas  

**Link**: [PDF](https://arxiv.org/pdf/2606.16072)  

**Abstract**: Compared with binaries and decompiled code, malware source code more directly reflects the attackers' original intent. However, the scarcity of source code and the high cost of manual review make such datasets difficult to build and maintain. We propose MASCOT-Android, a curated dataset of Android malware source code and an automated collection framework for scalable malware source code discovery on GitHub. A key finding of our work is that repository-level documentation alone provides a strong signal for malware source code collection. Our model extracts character-level TF-IDF features from 8,772 malware and 25,747 benign README documents and trains a LinearSVC classifier to distinguish malware repositories. This README-only model achieves an accuracy of 96.28\% and an FPR of 1.06\% in local evaluation. In addition, the model outputs confidence scores, allowing users to adjust the decision threshold to balance FPR and coverage, which is practical in real-world malware source code collection. 

---
# Mojo: A Promising Tool for Scalable Financial AI Efficiency 

**Authors**: Henry Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.16059)  

**Abstract**: For thirty years, quantitative finance has paid a costly two-language tax: models researched in Python are rewritten in C++ for production, often introducing numerical discrepancies. GPU-accelerated deep learning exacerbates this problem, as nondeterministic floating-point reductions can produce drift in long backtests, challenging regulatory reproducibility and auditability expectations. This article surveys Mojo, Modular's 2026 Python-like systems language, as a structural response for capital markets engineering. While closing the Python-to-C++ performance gap, Mojo uniquely combines native interoperability with the low-level systems control required to construct bit-exact deterministic kernels. Its MLIR compilation infrastructure further allows a single codebase to target scalar, SIMD, multicore, and GPU execution, reducing the translation bottleneck between research and production. We benchmark four core financial AI workloads: Monte Carlo option pricing, LLM sentiment inference, multi-asset backtesting, and portfolio Value at Risk. On Apple Silicon, Mojo demonstrates 20x to 180x speedups over pure Python on directly measured kernels; larger-scale GPU workload results are projections calibrated from published benchmarks. Alongside transparent performance data, we introduce mojo-deterministic, an open-source library of reproducible reduction kernels, and provide a candid assessment of the problems Mojo does and does not yet solve. 

---
# How to Detect and Measure the AI Dangers to Democracy 

**Authors**: Giulia Sandri, Claudio Novelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.16054)  

**Abstract**: Research on artificial intelligence and democracy has grown quickly over the last decade. A shared conclusion in this literature is that AI does not create new democratic problems so much as it makes old ones worse. We now see this across information ecosystems, in elections, and in public administration. However, despite growing evidence, we lack a clear way to prioritize risks in this area, compare them across domains, and identify where democratic control is most likely to break down. So, our problem is: How can we systematize the problems that AI systems pose to democratic processes? This paper argues that principal agent theory may fit the task. In many phases of democratic systems, principals delegate key functions to AI systems and their providers without really being able to monitor how these systems operate or the outputs they produce. Treating AI as a delegation problem helps identify accountability gaps and other governance failures. Most importantly, as we shall illustrate, it provides metrics for empirical assessments of AI impact on democracy. As a second analytical element, we draw on the NIST AI Risk Management Framework and its seven characteristics of trustworthy AI, which supply substantive criteria for evaluating delegated tasks. Operationalized across the three domains through measurable indicators and domain specific trustworthiness criteria, we propose an analytical framework that centers on institutional assessability as the central condition for democratic control over AI. However, we stress that how severe a harm is, and how much risk is acceptable, are evaluative judgments that current methodologies neither acknowledge nor operationalize. This becomes acute when such evaluative judgments are (silently) delegated to private vendors. We identify this as a strong limitation left for future work. 

---
# ALCL: An Adaptive Log-Correntropy Loss for Robust Learning under Non-Gaussian Noise 

**Authors**: Mainak Kundu, Ria Kanjilal, Ismail Uysal  

**Link**: [PDF](https://arxiv.org/pdf/2606.16050)  

**Abstract**: Robust deep learning under heavy-tailed and impulsive noise remains challenging because conventional losses such as mean squared error (MSE) exhibit unbounded sensitivity to outliers. Although correntropy-based objectives improve robustness, existing formulations rely on fixed kernel parameters that must be empirically tuned and remain static during training. To address these limitations, we propose an Adaptive Log-Correntropy Loss (ALCL), a heavy-tailed loss formulation that adaptively learns its robustness geometry during optimization. ALCL introduces a logarithmic residual model whose shape and scale parameters are learned jointly with network weights through differentiable reparameterization. This yields a principled maximum likelihood formulation whose influence function is formally bounded and redescending, allowing the loss geometry to adapt dynamically to evolving residual statistics while suppressing extreme outliers. Comparative experiments on four widely used benchmark datasets spanning grayscale and red-green-blue (RGB) image data under mixed heavy-tailed and impulsive noise demonstrate that ALCL consistently outperforms MSE and optimally tuned generalized correntropy losses in both reconstruction fidelity and downstream classification accuracy. While performance differences remain small under low-noise conditions, under high-noise regimes ALCL improves median accuracy by up to 4.75% on grayscale benchmarks and 4.51% on RGB datasets, with reduced variance across runs. These results demonstrate that adaptive robustness through joint learning of loss parameters provides a computationally efficient alternative to static correntropy-based losses for deep learning in non-Gaussian environments. 

---
# Leveraging Deep Learning for Object and Position Recognition of Load Carriers for Autonomous Logistics Vehicles 

**Authors**: Christoph Legat, Tobias Miller, Marco Riess  

**Link**: [PDF](https://arxiv.org/pdf/2606.16042)  

**Abstract**: This work explores the use of artificial intelligence in mobile robotics to achieve autonomous detection and pose estimation of load carriers for automated pickup. A deep neural network is designed to recognize predefined landmarks on the carrier from RGBD data; these landmarks are then used to compute the carrier's pose. The network operates directly on RGBD images to estimate landmark positions, which form the basis for determining the carrier's location. The approach is validated in extensive experiments and comprises both software and hardware implementations.
A deep learning-based framework is presented to detect load carriers and estimate their pose for use with autonomous logistics vehicles. Our method uses a convolutional neural network to identify characteristic reference points on the carrier from RGBD input and computes its pose by combining these inferred landmarks with prior geometric knowledge. Experiments show that the resulting accuracy is sufficient for reliable load carrier detection in industrial environments, confirming the suitability of the method for autonomous intralogistics applications. 

---
# Open-SWE-Traces: Advancing Dual-Mode Multilingual Distillation for Software Engineering Agents 

**Authors**: Wasi Uddin Ahmad, Nikolai Ludwig, Somshubra Majumdar, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2606.16038)  

**Abstract**: The path toward autonomous software engineering is currently bottlenecked by a severe deficit of diverse, large-scale trajectory data. We address this by introducing \ourdataset, an expansive dataset of 207,489 agentic trajectories spanning nine programming languages (Python, Go, TS, JS, Rust, Java, PHP, C, C++). Sourced from 20,000 real-world PRs via OpenHands and SWE-agent harnesses, the dataset utilizes a hybrid-reasoning synthesis: Minimax-M2.5 generates trajectories with explicit "thinking" processes, while Qwen3.5-122B provides high-quality "non-thinking" traces. Filtered for permissive licenses (MIT, Apache, BSD) from SWE-rebench-V2, this data facilitates the training of models capable of long-horizon reasoning. We validate the dataset by fine-tuning the Qwen3-30B-A3B series (Thinking, Instruct, and Coder). The best performing model achieves resolve rates of 61.7% on SWE-bench Verified, 57.1% on SWE-bench Multilingual, and 36.8% on SWE-bench Pro. These results establish Open-SWE-Traces as a premier resource for distilling human-level software engineering capabilities into efficient, open-source agentic LLMs. 

---
# Orchestrated Reality: From Role-Play to Living, Playable Game Worlds -- LLM-Driven World Simulation as a Parameterized-Action POMDP 

**Authors**: Yuhang Huang, Chenmiao Li, Chaowei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16014)  

**Abstract**: Many games rely on storytelling combined with systems that track levelling, NPC behaviour, and consequence simulation; bridging tightly-authored narrative with deeply-simulated worlds -- most acute in sandbox and open-world settings -- has been prohibitively expensive. LLM-driven worlds open a new path: a single harness can coordinate numerical state, narrative voice, storytelling pacing, and rule logic together. Realising this requires the LLM system to sustain a persistent world (who is where, what has just happened, what is currently true), which today's deployed systems do not: the narrative voice asserts state in free prose without any validated representation, so a fully autonomous game engine remains infeasible. We treat this as an architectural choice, not a limitation of language models, and report work in progress on a framework -- orchestrated reality -- that makes the world a canonical object owned by a singleton orchestration agent analogous to the tabletop-RPG Game Master (GM). We formalise an LLM-driven game world for a human player as a Parameterized-Action POMDP: state is a tree of canonical JSON entities, actions decompose as $a=(k, x_k)$ (a discrete intent kind plus structured JSON parameters), the agent observes only a narrative projection $o=O(s)$ of state, and the transition kernel $F$ is an LLM-driven Plan-Diff-Validate-Apply (PDVA) pipeline that commits schema-validated, content-hashed JSON deltas. We give the formal model, a JSON-state example, a worked single-turn example, and a catalogue of 15 illustrative incidents drawn from a real deployment showing the framework in action. Empirical validation through a planned human player study -- together with multi-NPC concurrent agency and deployment as an RL environment -- is situated as future work. 

---
# Theorem-Grounded Execution Ontologies for Interpretable Machine Reasoning 

**Authors**: Raghu Anantharangachar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16010)  

**Abstract**: Large language models have achieved impressive performance on reasoning tasks spanning mathematics, science, programming, and commonsense inference. Despite these advances, their reasoning processes remain largely latent, making them difficult to interpret, verify, replay, debug, and transfer across domains. Existing approaches such as chain-of-thought, tree-of-thoughts, graph-of-thoughts, and tool-augmented reasoning expose intermediate reasoning artifacts but typically lack explicit execution semantics, formal state representations, and verifiable reasoning structures.
We introduce Theorem-Grounded Execution Ontologies (TGEO), a framework that models reasoning as an executable state-transition process rather than a sequence of generated tokens. Given an input problem, TGEO identifies relevant theorem families, binds the problem to a domain ontology, discovers semantic objects, instantiates states and operators, constructs predicates and contracts, and synthesizes an executable reasoning graph. The resulting graph provides an interpretable, replayable, and auditable representation of reasoning in which every state transition, operator application, and validation step is explicitly represented.
TGEO integrates five architectural components: (1) theorem-grounded reasoning priors, (2) executable ontologies, (3) operator-mediated state transitions, (4) predicate and contract-based execution validation, and (5) architectural auditing and failure localization.
We evaluate TGEO on theorem-intensive reasoning tasks derived from mathematical benchmark domains and a curated Golden Execution Suite. Our findings demonstrate the value of executable reasoning representations for interpretable, verifiable, and reproducible AI reasoning systems. 

---
# Entity Labels Are Not Entity Signals: A Framework for Observable Relevance in Document Re-Ranking 

**Authors**: Utshab Kumar Ghosh, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.15998)  

**Abstract**: Entity-aware document retrieval uses query-associated entities as ranking signals, assuming that semantically relevant entities are also useful retrieval signals. We show this assumption is insufficient- and explain why. Unlike terms, which are ground-truth observations, entity links are hypotheses produced by an imperfect linker: an entity can be topically central yet provide no discriminative signal if the linker fires indiscriminately across relevant and non-relevant documents. We formalize this as a distinction between Conceptual Entity Relevance (CER)- whether an entity is topically related to a query- and Observable Entity Relevance (OER)- whether its observed presence in a collection discriminates relevant from non-relevant documents. Across four collections and annotation sources including human entity judgments, CER and OER exhibit near-chance agreement ($\kappa \approx 0$), while OER operationalizations agree substantially ($\kappa \approx 0.5$), confirming CER as the systematic outlier. CER-based supervision selects topically plausible but weakly discriminative entities, pruning fewer than 4% of non-relevant documents on some collections. Aligning supervision with OER improves non-relevant pruning by up to 10x and open-world MAP by 0.051 over BM25. Our findings motivate a shift from conceptual to observable notions of entity relevance in entity-aware retrieval. 

---
# Task-guided cross-subject latent alignment: a multi-encoder-decoder VAE 

**Authors**: Angeliki Papathanasiou, Jascha Achterberg, Thomas E. Nichols, Rui Ponte Costa  

**Link**: [PDF](https://arxiv.org/pdf/2606.15989)  

**Abstract**: Aligning neural activity across subjects offers the promise of discovering shared computational principles and generalizable decoders. However, traditional alignment methods require shared stimuli across subjects, a constraint that limits applicability to naturalistic paradigms with limited or non-overlapping data. We introduce a Multi-Encoder-Decoder Variational Autoencoder (MED-VAE) that achieves cross-subject alignment without shared stimuli by anchoring representations to a common scaffold provided by a pretrained ANN. Using the Natural Scenes Dataset, we show that MED-VAE creates common latent spaces with superior semantic organisation, achieving higher cross-subject alignment than common methods while maintaining robust generalisation to held-out stimuli where traditional methods degrade. Reconstructing from these common spaces back to each subject's original neural space, MED-VAE preserves equal stimulus-driven signal in its cross-subject latent space. Finally, we show that this superior alignment directly enables cross-subject neural prediction, as demonstrated via cross-subject image decoding. In summary, we introduce a framework to identify generalisable common subspaces for cross-subject predictions and downstream tasks, demonstrated here for visual cortex responses to static images. 

---
# Do Safety Monitors Stay Reliable After an Update? Benchmarking and Predicting Activation-Monitor Staleness 

**Authors**: Evan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15980)  

**Abstract**: Activation monitors-lightweight probes trained on a language model's internal representations-are an increasingly common layer in deployment safety stacks. Deployed models however are rarely static: they are quantized, fine-tuned, adapted with LoRA, or served with merged adapters while the monitor remains frozen. We present the first systematic test of whether this implicit contract holds: whether activation monitors trained on a base model remain reliable after these routine model updates. Across multiple safety-relevant monitors, model depths, update families, and open-weight models, we find a sharp split: quantization-style updates largely preserve frozen probe performance, while fine-tuning-style updates frequently make probes stale. Fragility is highly monitor-dependent, with privacy/PII probes most affected and refusal-compliance probes comparatively stable, showing that retraining a behavior need not stale its corresponding monitor. QLoRA is especially damaging despite NF4 quantization alone being relatively benign, suggesting that quantization becomes riskier when combined with adaptation. We further show that degradation is predictable from pre-deployment features, enabling revalidation budgets to be triaged toward the monitors most likely to fail. These results suggest that fine-tuning should trigger activation-monitor revalidation by default, while prediction can help prioritize which monitors to check first. 

---
# Formalize Once, Edit the Rest: Efficient Lean-Based Answer Selection for Math Reasoning 

**Authors**: Ji Feng, Zhouxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15972)  

**Abstract**: With large language models (LLMs) increasingly applied to mathematical reasoning, formal proof assistants such as Lean can be leveraged to verify reasoning outputs with machine-checkable rigor, enabling use cases such as answer selection in test-time scaling with K sampled candidate answers. However, employing Lean requires that LLM outputs, originally in natural language, first be formalized. Existing Lean-based answer-selection work uses an autoformalization model to generate a formal statement in Lean for each candidate answer independently, incurring a significant computational cost. We propose BASE, a base-and-edit pipeline that formalizes a single base candidate per problem and derives the remaining K-1 statements by editing the answer expression in place. To facilitate this, we train a rewriter model LEANSCRIBE to localize the answer in the base formalization and generate a reusable edit function for the other K-1 candidates. BASE simultaneously improves selection accuracy and reduces formalization cost - a Pareto improvement that holds on all 12 (dataset, solver) configurations across four benchmarks and three solvers, cutting autoformalizer calls by about 5x at K=8, with the reduction expected to become larger as K grows. Code is available at this https URL. 

---
# PreLort: Prefix-Nested LoRA for Federated Fine-Tuning under Rank Heterogeneity 

**Authors**: Muhammad Waseem, Nurbek Tastan, Andrej Jovanovic, Nicholas D. Lane, Nils Lukas, Karthik Nandakumar, Samuel Horvath  

**Link**: [PDF](https://arxiv.org/pdf/2606.15963)  

**Abstract**: Federated fine-tuning of large language models using parameter-efficient methods such as LoRA enables privacy-preserving adaptation of foundation models. Heterogeneous hardware resources introduce challenges, as clients with different adapter ranks cannot be directly aggregated. While existing methods enable aggregation under heterogeneous ranks, they fail to control how information is distributed across rank dimensions, leading to suboptimal use of shared low-rank representations. Instead, we propose PreLort: a nested low-rank formulation for federated LoRA that organizes adapter dimensions into a prefix hierarchy. Our approach ensures that lower-rank dimensions encode task-relevant information, while higher-rank dimensions capture additional capacity. Building on this, we introduce (i) a segment-wise aggregation rule that averages only over clients contributing to each rank segment, avoiding dilution from zero-padded lower-rank clients, and (ii) a prefix-nested training strategy that optimizes each adapter under multiple rank truncations, encouraging useful signal to concentrate in low-rank prefix dimensions. Together, these components encourage a consistent low-rank prefix capturing the most task-relevant information, while higher-rank dimensions learn additional capacity. This allows low-rank clients to benefit from richer information contributed by higher-rank clients, as prefix dimensions are consistently learned and aggregated. Experiments demonstrate that our method consistently outperforms prior heterogeneous federated LoRA methods in accuracy and ROUGE-L, while achieving lower or comparable perplexity across multiple base models. 

---
# Quantifying the Impact of Lossy Compression on Neural Generative Surrogate Modeling 

**Authors**: Zhimin Li, Harshitha Menon, Charles Jekel, Valerio Pascucci, Peter Lindstrom  

**Link**: [PDF](https://arxiv.org/pdf/2606.15959)  

**Abstract**: Neural networks are used as generative surrogate models for scientific discovery, which are trainable approximations of scientific simulations. These models enable users to replace time-consuming numerical simulations with learned alternatives, providing quick solutions. However, high-fidelity generative surrogate models require massive training datasets, which can create storage and I/O challenges. Lossy compression is a promising way to reduce this burden, but compression errors may affect the model quality in subtle ways, making it challenging to quantify their impact.
In this work, we examine how lossy compression of training data impacts the quality of generative surrogate models. We begin by characterizing the uncertainty inherent in training neural networks, showing that identical training configurations can produce different models. By exploiting this variability, we propose a method to estimate how much compression-induced error a surrogate model can tolerate without affecting its accuracy. Evaluation of two application simulations demonstrates that our approach significantly reduces memory/storage requirements and speeds up training while producing high-quality surrogate models. These results show that lossy compression saves data storage up to 23.7x and 39x with negligible impact on the quality of the surrogate model. Meanwhile, reducing the size of the training data set also enhances the data loading speed and reduces the training time by up to 3x. 

---
# You Don't Need Strong Assumptions: Visual Representation Learning via Temporal Differences 

**Authors**: Ninad Daithankar, Alexi Gladstone, Yann LeCun, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.15956)  

**Abstract**: Progress in AI has largely been driven by methods that assume less. As compute and data increase, approaches with weaker inductive biases generally outperform those with stronger assumptions. This is particularly characteristic of the field of Visual Representation Learning, where approaches have gone from being dominated by Supervised Learning, to Weakly Supervised Learning, to the now widespread success of Self-Supervised Learning without human labels. Yet, even modern Self-Supervised Learning approaches still depend on strong inductive biases such as augmentations, masking, or cropping. If this trend holds, even these remaining biases should become bottlenecks at scale -- and our experiments confirm this: the optimal strength of inductive biases decreases as data grows. This motivates the search for approaches that rely on fewer assumptions. To this end, we introduce Temporal Difference in Vision (TDV), a new paradigm for self-supervised learning from video that avoids existing inductive biases, relying instead on a causal assumption that the past causes the future. TDV functions by jointly training an image encoder and a motion encoder so that the current frame's representation plus the encoded motion equals the next frame's representation. Despite not leveraging any strong inductive biases, TDV matches state-of-the-art recipes on dense spatial tasks, laying the foundation for representation learning without strong assumptions. 

---
# Green SARC: Predictive Cost and Carbon Governance for Agentic AI Systems 

**Authors**: Gaston Besanson  

**Link**: [PDF](https://arxiv.org/pdf/2606.15954)  

**Abstract**: Agentic AI systems act through tools and sub-agents, yet the controls meant to bound their financial and environmental cost still sit on dashboards evaluated beside or after execution. Green SARC applies the SARC governance-by-architecture framework -- four enforcement sites in the agent loop -- to FinOps and GreenOps, contributing the theory of what to enforce and how to predict it. We report four policy-independent results. (i) The unconstrained "State Snowball" is $\Theta(n^2)$ in loop depth; on 3,000 real multi-step plans (SWE-rebench) it holds on 100%, with median curvature $\hat{c}_2=216$ exceeding the linear-accretion prediction $p/2=134$ -- real plans accrete faster than the model. (ii) On real residuals the Normal-$\sigma$ gate under-covers (92% at nominal 95%); split-conformal calibration holds (95.2%). (iii) A soft Lagrangian penalty tuned to the budget in expectation breaches it on 91.5% of seeds; the architectural gate breaches 0%. (iv) Under binding budgets the gate's over-budget incidence is 0% on synthetic and real (BurstGPT) arrivals. End-to-end token/USD/carbon savings (47--55%) are real but policy-dependent in magnitude -- set by a scope-cap knob, not by gate rejections. The library is open-source, dependency-free, and ships a regeneration script for every cited number. 

---
# Graphical-Probabilistic Modeling of Generative Flows in LLM-Native Software Systems 

**Authors**: Víctor A. Braberman, Flavia Bonomo-Braberman  

**Link**: [PDF](https://arxiv.org/pdf/2606.15943)  

**Abstract**: Engineering LLM-native software remains a challenging and immature field. Current practice is largely exploratory, relying on experimentation and heuristic techniques such as prompting and context engineering. These, however, are low-level and lack the principled structure needed to support design-level reasoning or analysis. In contrast, traditional software engineering leverages modularity and abstraction to communicate and analyze system behavior. To bring similar rigor to LLM-native development, we propose methods for documenting generative flows and for stating properties of LLM-based software designs. Such methods must account for the stochastic, prompt-dependent behavior of large language models while remaining expressive enough to capture emergent phenomena. Our initial approach is based on graphical probabilistic models, tailored to capture phenomena characteristic of LLM-native systems. This framework -- what we term Generation Networks -- aims to provide a foundation for principled reasoning about generative interactions and system-level properties in LLM-centric software architectures. 

---
# DeepRoot: A KG-Coordinated Multi-Agent System for Therapeutic Reasoning over Historical Medical Texts 

**Authors**: Zijian Carl Ma, Sean J. Wang, Sijbren Kramer, Li Erran Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.15931)  

**Abstract**: Historical medical archives and traditional medicines hold immense potential for drug discovery and remain a primary source for current drug development. However, pre-ontological prose and idiosyncratic taxonomies prevent the standardization and medical modernization of the data for use in current biomedical pipelines. Furthermore, no existing LLM agent system, whether tool-calling, retrieval-augmented, or agentic deep-research, can convert such text into verifiable drug-discovery leads at scale. We close this gap with DeepRoot, a multi-agent LLM system that jointly builds and utilizes a verified knowledge graph, showing that grounding and reasoning -- often conflated -- are separable axes the system can compose for therapeutic reasoning. Applied to the Shen Nong Ben Cao Jing, DeepRoot recovers $10$ of $21$ held-out compound-disease treatment pairs at R@$20$ ($47.6\%$ vs $4.8\%$ for a raw corpus LLM and $\sim\!2.4\%$ random) and dominates an LLM-as-judge audit for reasoning quality over baseline LLMs and LLMs with direct tool-call access to the same APIs DeepRoot itself queries. Tool-using LLMs hallucinate evidence on $87\%$ of claims, versus 7-10% for DeepRoot. Graph-only inference hallucinates $0\%$ but ranks lowest on reasoning coherence; DeepRoot KG+LLM is the only condition to win on both axes, pointing toward a route for systematic mining and repurposing of historical medical knowledge. 

---
# ControlMap: Controllable High-Definition Map Generation for Traffic Scenario Simulation 

**Authors**: Marwan Farag, Steffen Wäldele, Yu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15930)  

**Abstract**: Simulation is central to validating autonomous driving systems, yet current pipelines are limited by insufficient scenario diversity due to costly High Definition (HD) map creation. Scaling HD maps requires expensive data collection and manual processing. Moreover, existing generative models lack the fine-grained control necessary to target specific road topologies during generation.
This paper presents a data-driven pipeline for controllable HD map generation using latent diffusion and ControlNet for spatial conditioning. To our knowledge, we are the first to inject spatial guidance signals into a diffusion model for HD map synthesis. Furthermore, our model supports adjustable conditioning strength through classifier-free guidance and city-level style transfer via city label conditioning. To complement existing metrics, we introduce two novel metrics to evaluate adherence to the control signal and similarity to ground-truth maps. Experiments demonstrate that our model generates realistic HD maps that faithfully follow input road topologies while accurately preserving city-specific details. 

---
# Runtime Analysis of Cartesian Genetic Programming in Evolving Boolean Functions 

**Authors**: Duc-Cuong Dang, Roman Kalkreuth, Andre Opris  

**Link**: [PDF](https://arxiv.org/pdf/2606.15923)  

**Abstract**: Cartesian Genetic Programming (CGP) is among the practical and popular forms of Genetic Programming as it uses a graph-based representation of programs. This paper presents a first runtime analysis of CGP in evolving Boolean functions using complete training sets. We prove an asymptotic bound $O(n D^5)$ for the expected number of fitness evaluations of CGP to construct a conjunction of $n$ inputs using at most $D \geq n-1$ binary gates, a minimal function set, and even with a strict survival selection. When the non-strict selection is used, the bound is improved to $O(n D^4)$. Our analysis reveals interesting characteristics of CGP induced search, which have been only observed empirically. In particular, enabling the acceptance of equally good solutions, including those with connected gates non-contributing to fitness, can lead to a speedup, and consequently a better asymptotic time bound. In contrast to conjunctions, we also prove a negative result which shows that CGP requires exponential time to evolve an exclusive disjunction. Experiments evolving conjunctions complement our theoretical findings. The use of incomplete training sets is found to further reduce the average number of fitness evaluations while maintaining a good level of generalisation. 

---
# On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents 

**Authors**: Gengsheng Li, Mao Zheng, Mingyang Song, Ruiqi Liu, Tianyu Yang, Jie Sun, Qiyong Zhong, Haiyun Guo, Junfeng Fang, Dan Zhang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15912)  

**Abstract**: Multi-turn agents that plan, invoke tools, and interact with environments offer a promising paradigm for solving complex tasks, yet their capabilities typically rely on very large models whose inference cost is prohibitive in this http URL-Policy Distillation (OPD) is a natural recipe for transferring such capabilities to smaller students, but we find that it suffers a characteristic failure mode in this setting: small student errors compound across turns and push the trajectory out of the teacher's familiar state distribution, so the teacher's supervision becomes least reliable precisely where the student needs it this http URL propose Guided On-Policy Distillation (Guided-OPD), a simple yet effective algorithm that mixes teacher- and student-generated turns within each rollout and schedules the teacher's intervention probability along a curriculum that decays to this http URL guidance keeps early trajectories close to the teacher distribution and is then gradually withdrawn to recover the purely on-policy regime used at this http URL ALFWorld, ScienceWorld, and WebShop, distilling Qwen3 students from a Qwen3-30B-A3B teacher, Guided-OPD improves Score by 21.1\% and Success Rate by 25.5\% over vanilla OPD on average, with larger gains on smaller students. 

---
# MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA 

**Authors**: Yilong Zuo, Xunkai Li, Jing Yuan, Qiangqiang Dai, Hongchao Qin, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.15906)  

**Abstract**: Long-document multimodal question answering requires a system to locate sparse evidence in long PDFs and integrate clues from text, tables, images, charts, and complex layouts. Existing RAG methods mostly rely on fixed Top-k retrieval over text chunks or pages. Text retrieval can compress the context but often loses visual and layout information; page-level visual retrieval preserves the original page, yet it also sends large irrelevant regions to the reader, leading to a static trade-off among evidence coverage, noise, and inference cost. This paper proposes MAGE-RAG, a multigranular adaptive graph evidence framework for long-document multimodal QA. MAGE-RAG uses page retrieval as the entry point for query-time evidence construction. Offline, it builds an evidence graph with page nodes and element nodes, encoding containment, reading order, layout adjacency, section hierarchy, and semantic-neighbor relations. At query time, an online evidence controller iteratively activates, opens, searches, and prunes evidence under explicit budgets. The resulting evidence subgraph is then rendered into structured multimodal reader input, allowing the LVLM to consume compact and relevant evidence within a limited context. On LongDocURL and MMLongBench-Doc, we establish a unified comparison and analysis protocol covering Direct MLLM, Text RAG, Page-level Visual RAG, and Graph/Agentic RAG. Experiments show that MAGE-RAG achieves 52.75 overall accuracy on LongDocURL, and 53.26 accuracy with 51.19 F1 on MMLongBench-Doc. Fine-grained breakdowns, budget-performance curves, ablations, and trace-based analysis further show that query-time evidence subgraph construction can balance dispersed evidence coverage with context-noise control. Our code is available at this https URL. 

---
# Control-Plane Placement Shapes Forgetting: An Architectural Study of Agent Memory Across Thirteen System Configurations 

**Authors**: Dongxu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15903)  

**Abstract**: Where an LLM sits in an agent memory pipeline -- between the recall plane that retrieves stored facts (extensively benchmarked) and the control plane that mutates them via supersede, release, purge (largely untested) -- shapes which forgetting failure modes the system recovers. Comparing thirteen system configurations on a 385-case adversarial surface, we observe three placement regimes with partly complementary coverage: deterministic primitives suffice for lexical/temporal categories but fail canonicalization (5% on identifier-obfuscation, 0% on cross-lingual); inscribe-time LLM recovers canonicalization (100%) but cannot help intent-aware deletion (0% on prefix-collision and compound-fact); a mutation-time hook recovers intent-aware deletion (78-85%) and brightens nearly all categories simultaneously (91.7-93.2% overall, $0.17 per 385-case run, 2.3s/case mutation latency vs. 64-191ms/case deterministic, recall path unchanged).
We expose the trade-off via ForgetEval, a 1000-case templated suite plus a 385-case adversarial layer (132 hand-crafted + 253 LLM-drafted oracle-validated) scored by deterministic substring match, paired with a six-method Adapter Protocol with honest N/A scoring that lets heterogeneous memory stores enter in 130 lines. Admission is corroborated by 10-annotator IAA (Fleiss' kappa = 0.958) and a 77-case external-authored subset (four blind contributors) that replicates the canonicalization asymmetry and amplifies the joint-placement lift (+27.8 pt). Production failures are predominantly forgetting failures rather than recall failures, yet existing benchmarks measure only recall. ForgetEval and all adapters are released under MIT. 

---
# SkillVetBench: LLM-as-Judge for Multi-Dimensional Security Risk Evaluation in Open-Source LLM Agent Skills 

**Authors**: Ismail Hossain, Sai Puppala, Md Jahangir Alam, Tanzim Ahad, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2606.15899)  

**Abstract**: Open-source LLM agent ecosystems are growing rapidly, yet the security of community-contributed skills - modular tool definitions that extend agent capabilities - remains largely unvetted. The gap we fill: existing scanners operate at the code layer and are structurally blind to instruction-layer and multi-agent risk - natural-language directives that hijack an agent, exfiltrate data through encoded side channels, or chain harm across pipelines - so what is needed is a semantic, multi-dimensional vetting system rather than another signature matcher. We present SKILLVETBENCH, a live public leaderboard on Hugging Face that uses an LLM-as-Judge to vet agent skills. What is new: SARS (Skill Agentic Risk Score), a five-dimensional agentic-risk metric with a principled weighted formula for instruction-following systems. What is integrated: full CVSS v4.0 vector decomposition and a ClawHub dual-view that places our LLM-generated review beside the official marketplace verdict. What is demonstrated: drawing on our companion benchmark paper [ 1], the LLM-as-Judge stage achieves zero false negatives across 78 confirmed-malicious skills and zero false positives across 22 benign controls, while the best static baseline (SKILLSIEVE) still misses 15%; for instruction-layer categories such as Prompt Injection and Memory Poisoning, conventional tools miss between 89% and 100% of threats (e.g., CODEBERT detects none of nine memory-poisoning skills). Detection rates vary from 35% to 95% across four LLM evaluators, motivating ensemble scoring in production deployments. 

---
# Topological Flow Matching 

**Authors**: Kacper Wyrwal, İsmail İlkan Ceylan, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2606.15897)  

**Abstract**: Flow matching is a powerful generative modeling framework, valued for its simplicity and strong empirical performance. However, its standard formulation treats signals on structured spaces, such as fMRI data on brain graphs, as points in Euclidean space, overlooking the rich topological features of their domains. To address this, we introduce topological flow matching, a topology-aware generalization of flow matching. We interpret flow matching as a framework for solving a degenerate Schrödinger bridge problem and inject topological information by augmenting the reference process with a Laplacian-derived drift. This principled modification captures the structure of the underlying domain while preserving the desirable properties of flow matching: a stable, simulation-free objective and deterministic sample paths. As a result, our framework serves as a drop-in replacement for standard flow matching. We demonstrate its effectiveness on diverse structured datasets, including brain fMRIs, ocean currents, seismic events, and traffic flows. 

---
# NVMOS: Non-Verbal Vocalization Quality Assessment in Speech 

**Authors**: Jialong Mai, Jinxin Ji, Xiaofen Xing, Wencui Liu, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15888)  

**Abstract**: Non-verbal vocalizations (NVs), such as laughter, sighs, and coughs, are important acoustic cues for emotion and intent. Existing speech quality assessment methods typically focus on overall naturalness, while non-verbal TTS evaluations mainly examine whether a target NV appears with the correct type and position. However, the perceptual quality of NV events themselves remains underexplored. To address this gap, we construct an NV-MOS dataset containing outputs from multiple NV-TTS systems and naturally occurring NV samples, with ratings collected from three acoustic experts on a perceptual quality scale. We further analyze audio-capable multimodal large language models such as Gemini and find clear inconsistencies between their scores and expert ratings. These results suggest that general-purpose multimodal models cannot reliably replace human judgments for NV quality assessment. We then propose NVMOS, to our knowledge the first model that can reliably predict the perceptual quality of NV events in speech. Experimental results show that, with a local NV-event focusing module, NVMOS reaches expert-level or stronger agreement with human MOS. 

---
# Intelligence Is Not the Bottleneck: Validating an LLM First-Pass Manuscript Score Against Peer-Review Outcomes 

**Authors**: Costa Georgantas  

**Link**: [PDF](https://arxiv.org/pdf/2606.15887)  

**Abstract**: Large language model (LLM) systems are increasingly proposed to assist peer review, yet most evaluations judge the prose of machine-generated review text, not the validity of the numeric score a system assigns. We validate AIPR, which reads a submitted manuscript and emits five 0-100 quality dimensions and a weighted overall score, against the public decision outcomes of a major machine learning venue. AIPR grades by prompting alone, with no fine-tuning on reviews or decisions. Across 300 ICLR submissions with public decision tiers and reviewer ratings, graded under a frozen pipeline with hypotheses pre-registered before any score met any outcome, the overall score separates rejected from accepted submissions (AUROC 0.82, 95% CI 0.78-0.87), rises monotonically across tiers, and tracks the mean reviewer rating. The signal is strongest where we claim it: the lowest-scoring fifth is rejected far above the base rate, with oral papers absent. The validity comes mostly from the model: a one-paragraph prompt on the same model discriminates almost as well as the full pipeline (the small gap favours the pipeline but does not meet the pre-declared criterion, p = 0.09). What the engineering adds is reliability and a grounded review: AIPR's score barely moves across repeated runs (0.7 vs. 2.8 points within-paper SD) where the bare prompt swings, and the same pass returns a rubric-structured, evidence-grounded review rather than a bare number, with the human keeping the decision. 

---
# Koshur Diacritizer: A Byte-Level Sequence-to-Sequence Model for Kashmiri Diacritic Restoration 

**Authors**: Haq Nawaz Malik, Nahfid Nissar, Faizan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2606.15883)  

**Abstract**: Kashmiri, an Indo-Aryan language written in a modified Perso-Arabic script, frequently omits diacritic marks in digital text, creating ambiguity and challenging downstream NLP applications. We present Koshur Diacritizer, a ByT5-small byte-level sequence-to-sequence model for restoring diacritics in Kashmiri text. To support this task, we release a publicly available dataset of 23.7k aligned undiacritized diacritized Kashmiri sentence pairs. The proposed framework combines script-aware normalization, alignment validation, and skeleton-preserving inference to ensure reliable restoration while maintaining the original base-letter sequence. Experimental results on a held-out test set achieve a DERm of 0.2012 and a WER of 0.2159. Additionally, evaluation by a native Kashmiri linguistic expert yields a mean accuracy of 77.5%. The dataset, model, and source code are publicly released to provide a reproducible baseline for Kashmiri diacritic restoration and future low-resource language research. 

---
# Deep Residual Injection for Full-Spectrum Forensic Signal Perception in Multimodal Large Language Models 

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ruoxin Chen, Ke-Yue Zhang, Yue Zhou, Caiyong Piao, Bin Li, Taiping Yao, Bo Wang, Youchang Xiao, Shouhong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.15880)  

**Abstract**: Multimodal large language models (MLLMs) have been increasingly adopted in forensics for their robust semantic understanding. As AI-generated images become realistic, semantic-level inconsistencies alone are often insufficient for reliable detection. This motivates a critical question: whether MLLMs can achieve full-spectrum forensic signal perception, i.e., capturing low-level generator artifacts without sacrificing pre-trained semantic knowledge. We further perform a layer-wise analysis of forensic signal perception in MLLMs, showing that semantic information is primarily formed in the early-to-middle layers, whereas direct fine-tuning for artifact learning disrupts these semantic representations. Based on this insight, we propose Deep Visual Residual MLLM (Deep-VRM) to preserve early semantic processing while injecting artifact-specific visual signals as a residual path into an intermediate layer, where they are fused with semantic token representations and propagated through subsequent trainable layers. This enables later layers to jointly model semantic reasoning and signal-level forensic cues, and surprisingly, the model learns to adaptively leverage different levels of forensic signals depending on the input, achieving robust and generalizable detection performance. Extensive experiments show that our method achieves state-of-the-art across most benchmarks. The code and data are available at this https URL. 

---
# Free Energy Heuristics: Fast-And-Frugal Cognition as Active Inference Under Uncertain Precision 

**Authors**: Alex Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15877)  

**Abstract**: Chain-of-thought (CoT) improves large language models' performance in math and symbolic reasoning. But on planning, contested ethics, and tasks where the model cannot check itself, more reasoning makes things worse. Both effects are documented; what has been missing is a principled account of which property decides the outcome. We argue it is meta-uncertainty: how unsure the model is about the reliability of its own evidence. When that uncertainty is high, extra reasoning stops adding signal and starts manufacturing false confidence. We prove that the policy minimizing expected free energy under uncertain precision stops integrating cues after a finite number of high-validity ones when the precision prior is heavy-tailed (Theorem 2.6.1), and under a Descending Dominance condition, is sample-wise identical to take-the-best (Theorem 2.7.4). Fast-and-frugal heuristics and active inference are, then, two descriptions of the same computation. The prediction is that on high-meta-uncertainty items, longer CoT should degrade accuracy. We score the regime per item (simulate-and-recover rho > 0.96), build FEH-79, a benchmark of Knightian frames with matched controls, and run a pre-registered study across seven models (five open-weight 3B-32B, two frontier), five CoT lengths, and 7,875 responses. The gate, fixed before any data, required a negative interaction with posterior probability above 0.95 and an accuracy drop of more than 6 points. It held. The high-regime drop is 17.3 points (95% CI [7.7, 25.5]); matched items with definite answers show no cost. The effect is regime-dependent: decisive in capable mid-to-large models, directional in the two frontier systems, absent-to-reversed in the weakest. The framework answers when CoT helps and unifies the Bayesian and fast-and-frugal traditions: less-is-more effects are evidence about the meta-uncertainty regime, not against Bayesian cognition. 

---
# Wasserstein Convergence of ODE-Based Samplers in Decentralized Diffusion Model via Velocity Field Decomposition 

**Authors**: Chencheng Tang, Xuanyu Xue, Fangyikang Wang, Chao Zhang, Hubery Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15835)  

**Abstract**: Diffusion models have achieved impressive empirical success in generative tasks, and their convergence theory is now relatively well understood. Motivated by privacy and scalability, recent decentralized diffusion architectures replace a single global velocity field with multiple local experts and a routing mechanism, yielding a sampling dynamics with stochastic expert switching that falls outside standard diffusion convergence analyses. In this work, We study a decentralized diffusion framework with stochastic velocity fields and ODE-based sampling. We establish a convergence guarantee in Wasserstein-2 distance, showing that the distribution of the $N$-step discretization converges to the analytical solution at rate $\mathcal{O}(N^{-1/2}+\varepsilon)$ in $W_2$, where $\varepsilon$ captures the neural approximation errors. To our knowledge, this is the first $W_2$ convergence result for decentralized diffusion models with an ODE-based sampling scheme. 

---
# The Truth Stays in the Family: Enhancing Contextual Grounding via Inherited Truthful Heads in Model Lineages 

**Authors**: Miso Choi, Seonga Choi, Mincheol Kwon, Woosung Joung, Jinkyu Kim, Jungbeom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.15821)  

**Abstract**: Recent advances in large language models (LLMs) have produced many specialized multimodal LLMs (MLLMs) that share common foundational LLMs, forming distinct model lineages. It remains unclear whether a fundamental behavioral link exists between the foundational LLMs and downstream variants. We investigate this question by quantifying head-level context-truthfulness scores. Across diverse LLM and MLLM lineages, including Vicuna-, Qwen2.5-, LLaMA2-, and Mistral-based models, we find that Truth Scores are strongly preserved within model families, even after instruction tuning or multimodal adaptation. We further show that this inheritance is consistent with attention-head weight preservation, and that context-truthful heads attend to query-relevant evidence. Building on this finding, we propose TruthProbe, a soft-gating strategy that amplifies context-truthful heads while preserving other head contributions. TruthProbe improves contextual truthfulness on HaluEval and reduces multimodal hallucination on POPE and CHAIR, with base-LLM Truth Scores transferring effectively to their fine-tuned LLM and MLLM descendants. Code is available at this https URL. 

---
# SACE: Concept Erasure at the Semantic Singularity in Visual Autoregressive Models 

**Authors**: Siya Yang, Nanxiang Jiang, Zhaoxin Fan, Yunfeng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15819)  

**Abstract**: The rapid progress of visual autoregressive (VAR) models has unlocked a transformative frontier for high-fidelity text-to-image synthesis, while heightening concerns over the safety alignment of generated content. Naive application of existing erasure techniques to VAR models causes catastrophic semantic collapse and visual artifacts, since they are predominantly designed for the homogeneous denoising steps of diffusion models. To address this foundational challenge, we first propose the Semantic Singularity Axiom, which posits that any target semantic concept embedded within a prompt is definitively locked at Scale-0. Then rigorously validate this axiom through our proposed Incremental Semantic Saliency Analysis (ISSA),which also enable the community to transparently inspect the coarse-to-fine semantic injection process. Guided by this insight, we introduce the first scale-aware concept erasure framework (SACE) for VAR models. By strictly confining interventions to the first scale, our approach couples an Entropy-Regularized Erasure Objective to prevent high-entropy sampling degeneration, alongside a restorative preservation loss to safely anchor the integrity of entangled benign priors. Extensive experiments demonstrate that our method achieves surgical concept erasure performance across various domains with minimal training overhead, timely and elegently resolute the critical safety vulnerabilities inherent in emerging VAR architectures. Code is available at: this https URL}{this https URL. 

---
# Let Them Steal: Trapping Large Language Model Extraction Attacks with Knowledge Honeypot 

**Authors**: Yuyang Dai, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.15810)  

**Abstract**: Large language models deployed as commercial APIs are vulnerable to model extraction attacks, while existing defenses either act too late or degrade utility for legitimate users. We propose \textbf{Knowledge Trap}, a defense that redirects extraction attacks toward low-transferability knowledge through a \emph{Honeypot Knowledge Graph} (HKG) and breadcrumb-guided exploration. Instead of blocking queries or perturbing outputs, Knowledge Trap consumes the attacker's limited query budget on knowledge with negligible downstream utility while preserving benign-user performance. Experiments in medical and financial domains show that Knowledge Trap reduces surrogate Agreement by 6.2\% on average without degrading legitimate-user accuracy, outperforming existing defenses that impose measurable user impact. These results suggest that defending knowledge-space traversal is a practical direction for mitigating LLM extraction attacks. 

---
# Continuous Cross-Domain Traffic State Prediction via Memory-Augmented Graph Liquid Time-Constant Networks 

**Authors**: Jinrong Xiang, Ming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15807)  

**Abstract**: Traffic state prediction is a fundamental task in intelligent transportation systems. In practical applications, some regions suffer from limited traffic observations due to insufficient sensing infrastructure, making cross-domain knowledge transfer an important solution for data-scarce traffic prediction. However, existing cross-domain traffic prediction methods still face several limitations, including coarse-grained source-target adaptation, limited capability in handling unseen target-domain patterns, and insufficient modeling of continuous traffic dynamics under irregular or heterogeneous temporal conditions. To address these issues, this paper proposes a continuous cross-domain traffic prediction framework, termed Memory-Augmented Graph Liquid Time-Constant Network (MA-GLTC). Specifically, we first construct spatio-temporal units (STUs) to decompose traffic networks into transferable local units, enabling fine-grained knowledge alignment across domains. Then, a graph liquid time-constant network (GLTC) is developed to model graph-coupled traffic evolution in continuous time. Different from generic graph neural ODE-based models, GLTC introduces graph-coupled recurrent conductance into liquid time-constant dynamics, allowing node states to evolve with leakage, adaptive time constants, and neighborhood-aware feedback. Furthermore, a Memory-based Transfer Storage (MTS) mechanism is designed to preserve source-domain knowledge, retrieve matched traffic patterns, and update reliable target-domain patterns when unseen states emerge. Experiments on five public traffic datasets demonstrate that MA-GLTC consistently outperforms representative innerdomain and cross-domain baselines in both short-term and longterm prediction tasks. Compared with the second-best method, MA-GLTC reduces the average prediction errors by 3.02%, 0.33%, 8.92%, 10.09%, and 2.11%, respectively. 

---
# DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing 

**Authors**: Artyom Mazur, Nina Konovalova, Aibek Alanov  

**Link**: [PDF](https://arxiv.org/pdf/2606.15796)  

**Abstract**: Mechanistic interpretability seeks to explain neural network behavior by decomposing model computations into interpretable features and circuits. While transcoder-based circuit tracing has recently enabled detailed causal analyses of large language models, multimodal diffusion transformers for image generation remain comparatively opaque. We still lack tools for understanding how semantic information propagates across denoising steps and how text and image representations interact within double-stream MM-DiT architectures. Existing methods provide only partial insight: attention maps expose a limited view of token interactions, while sparse autoencoders can discover interpretable features but do not directly reveal how these features are transformed and composed through nonlinear MLP layers. In this work, we extend transcoder-based circuit tracing to multimodal diffusion transformers. We train timestep-conditioned transcoders that faithfully approximate the input-output behavior of MLP sublayers in FLUX.1[schnell]. By replacing MLPs with transcoders and linearizing the remaining computation, we obtain exact feature-to-feature attribution and recover compact, interpretable circuits. Empirically, our transcoders match or slightly outperform sparse autoencoders on the sparsity-faithfulness tradeoff. The resulting circuits reveal mechanisms underlying attribute binding and cross-stream semantic propagation, and provide causal explanations for systematic generation errors. Moreover, circuit-guided interventions are substantially more precise and effective than standard SAE-based steering. Our results demonstrate that transcoder-based circuit analysis is feasible for state-of-the-art diffusion transformers and provides a powerful framework for understanding and controlling multimodal generative models. The code is available at this https URL 

---
# Proximal Policy Optimization for Amortized Discrete Sampling 

**Authors**: Anna Zykova-Myzina, Timofei Gritsaev, Daniil Tiapkin, Nikita Morozov  

**Link**: [PDF](https://arxiv.org/pdf/2606.15793)  

**Abstract**: This paper explores policy gradient algorithms for training stochastic policies to sample from structured discrete probability distributions under the Generative Flow Network (GFlowNet) framework. Building on extensive theoretical connections between GFlowNets and entropy-regularized reinforcement learning, we derive equivalents of standard policy gradient algorithms for training GFlowNets, as well as experimentally explore their various methodological aspects, including baseline training and advantage estimation. Most importantly, our work is the first to derive and successfully apply proximal policy optimization to GFlowNets, showing its improved convergence speed and data efficiency compared to standard GFlowNet training objectives on benchmarks ranging from synthetic energies to molecular graph generation. 

---
# GAS-Leak-LLM: Genetic Algorithm-Based Suffix Optimization for Black-Box LLM Jailbreaking 

**Authors**: Aman Anifer, Vignesh Kumar Kembu, Vishnu M, Antonino Nocera, Vinod P., Amal Murali PK, Akshay S Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15788)  

**Abstract**: Large Language Models (LLMs) constitute pivotal components within the AI-dominated information technology ecosystem. To mitigate risks associated with harmful or policy-violating outputs, commercial systems employ advanced alignment strategies and multi-layered content moderation mechanisms. Despite these safeguards, recent research has demonstrated that LLMs remain vulnerable to adversarial manipulation, particularly through jailbreaking and prompt injection techniques. In this work, we propose GAS-Leak-LLM a novel jailbreaking attack based on a genetic algorithm that systematically evolves adversarial suffix to bypass safety constraints. Operating in a strict black-box setting, our method requires no access to model parameters or internals, thereby reflecting realistic threat scenarios in deployed systems. Through the iterative application of selection, mutation, and crossover heuristics, the framework systematically explores the discrete prompt space to identify high-fitness adversarial suffixes. Empirical findings reveal critical shortcomings in existing safety enforcement mechanisms and confirm the effectiveness and practical viability of the proposed attack. 

---
# Domain-Guided Prompting of the Segment Anything Model for Seismic Interpretation: The Role of Attributes, Visualization, and Hybrid Prompts 

**Authors**: Aniq Ahmad, Heather Bedle, Ahmad Mustafa  

**Link**: [PDF](https://arxiv.org/pdf/2606.15786)  

**Abstract**: The advent of large pretrained foundation models for computer vision has significantly improved the efficiency of visual data interpretation. The Segment Anything Model (SAM), in particular, offers powerful zero shot segmentation capabilities through prompt based interaction, thus making it a promising tool for seismic interpretation. However, most existing applications of SAM rely on fine tuning for specific geological targets, which requires extensive labeled data, incurs high computational cost, and often compromises the model's generalization capability. In this study, we introduce a principled framework for zero shot adaptation of foundation models to seismic data. The framework is built on two key components: (1) aligning seismic attributes and visualization choices (e.g., colormaps) with the geological target of interest, and (2) employing a hybrid prompting strategy that combines sparse user defined point prompts with dense mask prompts derived from SAM's internal feature activations. We systematically evaluate this framework across multiple geological targets, datasets, prompt configurations, and seismic attribute representations. Our results demonstrate that geologic target aware selection of seismic attributes and colormaps, combined with hybrid prompting, enhances the separability of geological features and improves boundary delineation and segmentation accuracy relative to point based prompting alone. Our findings show that, when these components are jointly applied, SAM can achieve competitive segmentation performance in a fully zero shot setting, thereby eliminating the need to retrain SAM for each geologic feature. This work establishes a practical and scalable pathway to leverage foundation models in seismic interpretation, reducing reliance on labeled data while preserving model generality. 

---
# DYNA : Dynamic Episodic Memory Networks for Augmenting Large Language Models with Temporal Knowledge Graphs in Continuous Learning 

**Authors**: Ali Sarabadani, Mahtab Tajvidiyan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15778)  

**Abstract**: Large Language Models (LLMs) struggle to incorporate new knowledge without forgetting or costly retraining. We propose DYNA, a lightweight framework that augments a frozen LLM with a temporal knowledge graph where events are nodes and temporal relations are directed, timestamped edges. The graph serves as an external, updatable memory. At query time, DYNA retrieves relevant nodes via random walks and centrality measures, then augments the LLM's response. Evaluated on three temporal recall tasks, DYNA reduces catastrophic forgetting by ~7% compared to fine-tuning and improves temporal ordering by ~5% over standard RAG. Higher graph clustering coefficients correlate with better retrieval, showing that graph structure matters. Contributions: (1) episodic memory as temporal KG, (2) retraining-free LLM augmentation, (3) graph properties as predictors of retrieval performance. 

---
# LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies 

**Authors**: Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15768)  

**Abstract**: Vision-Language-Action models (VLAs) leverage large-scale vision-language pretraining for semantic robot control, but often lack explicit foresight into how robot actions change the scene. World-Action Models (WAMs) address this limitation by conditioning policies on predicted futures, yet existing approaches typically rely on computationally expensive video generation with substantial pixel-level redundancy. We present LaWAM, a Latent World Action Model that exposes predictive dynamics to robot policies through compact latent visual subgoals instead of reconstructed future video. At the core of LaWAM is a latent-action-conditioned Latent World Model (LaWM). We obtain LaWM by training a latent action model in the latent space of a pretrained vision foundation model and repurposing its forward decoder to predict future observation features for scene evolution. LaWAM then conditions action generation on these predicted latent visual subgoals to enable dynamics-aware robot control. LaWAM achieves state-of-the-art or competitive success rates (SRs) across LIBERO (98.6% SR), RoboTwin (91.22% SR), and real-world manipulation tasks while retaining low-latency inference. LaWAM runs in 187 ms per action-chunk prediction and achieves up to 24x lower wall-clock latency than pixel-space WAMs. 

---
# Visualizing Uncertainty: Spatial Maps of Missing and Conflicting Evidence in Deep Learning 

**Authors**: Dong Hyun Jeong, Feng Chen, Jin-Hee Cho, Lance M. Kaplan, Audun Jøsang, Soo-Yeon Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.15767)  

**Abstract**: Understanding when and why deep neural networks are uncertain is crucial for deploying reliable machine learning systems in safety-critical domains. While existing uncertainty quantification methods provide scalar measures of model confidence, they offer limited insight into which spatial regions of an input contribute to different types of uncertainty. We propose a novel visualization framework, Uncertainty Activation Map (UAM), that combines Evidential Deep Learning (EDL) with Full-Gradient Class Activation Mapping (FullGrad) to generate interpretable spatial uncertainty activation maps. Our approach distinguishes between two fundamental types of uncertainty: vacuity, representing lack of evidence, and dissonance, capturing conflicting evidence between competing hypotheses. By leveraging the complete gradient decomposition property of FullGrad and the principled uncertainty quantification of Subjective Logic, our method produces theoretically grounded visualizations that highlight specific image regions responsible for model uncertainty. With this framework, vacuity and dissonance activation maps are generated by computing belief-weighted attributions, enabling identification of where models lack knowledge versus where they encounter ambiguous evidence. Extensive evaluations across multiple benchmark datasets demonstrate that the proposed framework effectively addresses the critical gap between uncertainty quantification and explainability, providing intuitive visual feedback to assess model reliability in complex visual recognition tasks. 

---
# Snyk VulnBench JS 1.0: Can LLMs Find the Same Bugs Twice? 

**Authors**: Liran Tal, Johannes Kloos, Arsenii Rudich, Stephen Thoemmes, Manoj Nair  

**Link**: [PDF](https://arxiv.org/pdf/2606.15762)  

**Abstract**: We ran 300 repeated vulnerability-finding scans to measure how repeatable agentic large language model (LLM) security review is on the same JavaScript code, prompt, and benchmark harness. The headline result is that LLM security findings were unevenly repeatable: reference-matched findings were stable, but extra model reports varied heavily from run to run. Across 250 model runs, 80 of 161 unique unmatched findings appeared in only one of five identical repetitions, while only 22 appeared in all five. By contrast, when Claude matched a Snyk Code reference finding, the behavior was much more stable: 134 of 158 unique reference-matched findings appeared in all five repetitions. The benchmark also shows complementarity. Models consistently found familiar, high-signal exploit shapes, and in one case surfaced a likely Snyk Code product gap. Snyk Code static application security testing (SAST) was deterministic and better at systematically enumerating repeated data-flow sinks. The results support combining agentic LLM review with deterministic SAST rather than treating either technique as a replacement for the other. 

---
# From Correlation to Causation in Lane Change Prediction for Automated Driving: A Causal Explanation Framework 

**Authors**: Mohamed Manzour, Aditya Kumar, Augusto Luis Ballardini, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2606.15756)  

**Abstract**: Lane-change prediction is a central task in intelligent vehicles, where early maneuver anticipation can support safer decision-making. However, many existing approaches mainly learn statistical associations between observed driving variables and future maneuvers, while overlooking the causal dependencies among the input variables themselves. This limits interpretability, especially when physically related variables such as longitudinal gap, relative longitudinal velocity, and Time-To-Collision (TTC) are treated as independent flat inputs. This article presents a causal-inference-based framework for lane-change prediction and explanation. The proposed approach combines linguistic feature construction, expert-constrained causal discovery, deep structural causal modeling with Deep End-to-end Causal Inference (DECI), intervention-based effect analysis, refutation testing, and recursive causal-chain explanation. The objective is not only to predict the future maneuver, but also to identify candidate variables that directly contribute to the prediction, the upstream factors influencing them, and the causal chains through which these effects propagate. The framework achieves average F1-scores above 95% during the first three seconds before the lane-marking crossing event. Beyond prediction accuracy, the framework uses intervention-based effect analysis to distinguish influential from weakly influential variables under the learned causal structure. It further distinguishes candidate direct contributors from mediated effects and generates contrastive causal-chain explanations that clarify why the predicted maneuver is favored and why the alternative maneuvers are less supported. The main contribution is therefore a mechanism-aware lane-change prediction pipeline that moves beyond correlation-based classification toward more interpretable causal reasoning for maneuver prediction. 

---
# OmniTraffic: A Controllable Generation Pipeline and Benchmark for Spatio-Temporal Traffic Reasoning 

**Authors**: Maonan Wang, Zhengyan Huang, Kemou Jiang, Yuhang Fu, Jiayue Zhu, Yuxin Cai, Xingchen Zou, Qiaosheng Zhang, Yi Yu, Ding Wang, Xi Chen, Ben M. Chen, Yuxuan Liang, Zhiyong Cui, Man On Pun, Yirong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.15749)  

**Abstract**: Traffic scene understanding requires models to reason beyond object recognition, including lane topology, multi-view geometry, temporal evolution, and signal-phase semantics. However, existing traffic-oriented multimodal benchmarks largely emphasize passive visual recognition or isolated video understanding, offering limited support for evaluating structure-aware traffic reasoning under controlled conditions. We introduce OmniTraffic, a controllable generation pipeline and benchmark for spatio-temporal traffic reasoning. Built around 12 real-world intersections reconstructed into editable 3D traffic environments and complemented by surveillance footage from two countries, OmniTraffic supports both controlled and natural-condition evaluation. It defines a three-level task hierarchy spanning scene perception, multi-view and temporal reasoning, and decision support. Using structured traffic metadata, OmniTraffic generates synchronized multi-view VQA samples covering vehicle states, lane functions, view--BEV correspondence, temporal dynamics, and signal-phase analysis, resulting in 8M VQA samples and a 3K human-verified test set. Evaluation of eleven frontier MLLMs reveals a large human--model gap, with the most pronounced failures in topology-grounded and spatio-temporal reasoning tasks. Fine-tuning a lightweight MLLM on simulated OmniTraffic data further improves performance on real-world traffic scenes, demonstrating the value of simulation-generated supervision for traffic-specific multimodal reasoning. Beyond a fixed dataset, OmniTraffic provides an extensible pipeline with configurable intersections, camera views, traffic demands, signal phases, visual conditions, and rare events. 

---
# A Self Consistency Based Reranking for Narrative Question Answering 

**Authors**: Molham Mohamed, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15741)  

**Abstract**: Narrative question answering (NQA) is a challenging task in natural language processing that requires models to understand long textual contexts, capture relationships across events, and generate coherent responses. Despite recent advances in pretrained language models, most existing approaches rely on a single decoding output during inference, making them sensitive to generation variability and often resulting in incomplete or inconsistent answers .To address this limitation, we propose a self-ensemble Self-Consistency-Based reranking framework for narrative question answering. The proposed method generates multiple candidate answers for each story-question pair and selects the final answer based on semantic agreement among the generated responses. This allows the model to explore diverse answer formulations while improving robustness through consensus-based selection without requiring modifications to the underlying architecture .The framework combines pretrained and fine-tuned language generation with multi-answer inference and similarity-based reranking. We evaluate the proposed approach on the NarrativeQA dataset using multiple models, including FLAN-T5 (Base and Small) and Pegasus-Large, under both baseline and fine-tuned settings .Experimental results demonstrate that the proposed method consistently improves performance across all models. In particular, FLAN-T5-Base achieves the best overall performance, improving from 82.32% to 86.66% (+4.34%) when combined with self-ensemble inference. Additionally, the largest improvement is observed with Pegasus-Large, which increases from 72.50% to 87.07% (+14.57%), highlighting the effectiveness of the proposed strategy. 

---
# EHRNote-ChatQA: A Benchmark for Evidence-Grounded Multi-Turn Clinical Question Answering over Longitudinal Discharge Summaries 

**Authors**: Jiyoun Kim, Muhan Yeo, Eunhye Jang, Jeewon Yang, Hangyul Yoon, Su Ji Lee, Hee Jo Han, Hee-Jae Jung, Doyun Kwon, Jun young Lee, Jaehun Lee, Jung-Oh Lee, Sunjun Kweon, Jong Hak Moon, Daseul Kim, Minjae Cho, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15735)  

**Abstract**: Discharge summaries are crucial clinical documents containing the context of a patient's overall hospital stay, and are routinely reviewed by medical experts for patient readmission, ongoing care, and diagnostic decision-making. When reviewing them, medical experts often must iteratively synthesize information across multiple summaries while verifying the evidence supporting each answer. Although large language models (LLMs) are increasingly explored for clinical question answering, existing benchmarks do not sufficiently reflect this setting: they often evaluate exam-style medical knowledge or focus on single-turn question answering with limited evidence-grounding evaluation. We introduce EHRNote-ChatQA, the first benchmark for evidence-grounded multi-turn clinical question answering over patients' multiple discharge summaries. Built from de-identified MIMIC-IV discharge summaries, EHRNote-ChatQA contains 967 patient-level multi-turn samples spanning one to five notes and 16,072 medical-expert-verified QA pairs (8,036 content questions, each paired with an evidence-grounding question) across eight clinical categories. The benchmark is constructed through an expert-informed pipeline combining discharge-summary structuring schema, expert-curated multi-turn QA templates, and LLM-based generation, followed by review and revision of every single QA sample by 11 medical experts. Benchmarking 22 open- and closed-source LLMs reveals several challenges, including that LLMs struggle more with evidence grounding than content answering, multi-turn errors compound across turns, and single-turn clinical QA performance does not reliably transfer to this setting. These findings establish EHRNote-ChatQA as a rigorous and practical benchmark for evaluating clinical QA systems. The dataset will be made publicly available through PhysioNet credentialed access. 

---
# Retrievable Gradients: Continual Post-Training Without Cumulative Weight Drift 

**Authors**: Weihang Su, Jiacheng Kang, Jingyan Xu, Qingyao Ai, Jianming Long, Hanwen Zhang, Bangde Du, Xinyuan Cao, Min Zhang, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15734)  

**Abstract**: Continual post-training enables models to absorb emerging knowledge after deployment, but repeatedly updating shared parameters can accumulate weight drift, potentially causing catastrophic forgetting and degrading general capabilities. Retrieval-augmented generation avoids such parameter drift, yet often lacks the depth of parametric knowledge integration. In this paper, we propose ReGrad (Retrievable Gradients), a new paradigm that treats gradients as retrievable units of knowledge. ReGrad pre-computes document-specific gradients offline, stores them in an indexed Gradient Bank, and retrieves only query-relevant gradients at inference time for temporary weight adaptation. However, raw language-modeling gradients are optimized for token-level document reconstruction rather than for query-driven knowledge use. We therefore introduce a bi-level meta-learning objective that reshapes document-derived gradients into generalizable adaptation signals for downstream tasks. Experiments across general and domain-specific settings show that \textsc{ReGrad} outperforms CPT and RAG baselines, enabling scalable and reversible parametric knowledge injection without accumulating weight drift. 

---
# Vernier: Probing Representational Misalignment Behind Lexical Gaps in Causal Reasoning 

**Authors**: Zhenyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15733)  

**Abstract**: Instruction-tuned language models can answer the same causal-reasoning question differently after its English variable names are replaced by type-preserving placeholders, although the structural causal model and the gold answer are unchanged. We ask whether this lexical gap reflects information loss in the placeholder view or a misaligned read-out from a representation that still carries answer-relevant content. Vernier uses a paired-view weight update as an instrument and then inspects the mechanism left after the gap closes. In the working regimes, the evidence favours representational misalignment. A variable-name probe becomes more accurate on the placeholder view, and activation patching on Qwen-7B, Qwen-14B, and Llama-3.1-8B shows that the decision-token representation can transfer answer identity between views. The update that realigns the views is counterfactual augmentation over original and placeholder prompts, while the answer-subspace KL mainly sharpens intermediate answer-belief agreement. Success is bounded by model family, scale, and task. CRASS transfer is reliable across Qwen scales and Llama, e-CARE remains weak, and preliminary non-causal rename tasks show a similar qualitative pattern. 

---
# InstantForget: Update-Free Backdoor Unlearning with Inference-Time Feature Reset 

**Authors**: Zhenyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15730)  

**Abstract**: Backdoor unlearning aims to remove a malicious trigger behavior from a deployed model while preserving clean utility. We study the update-free inference-time setting, where model parameters remain frozen. First, we audit a common projection assumption under oracle paired clean and triggered features. Projection succeeds mainly on BadNets and leaves WaNet, Blended, and SIG at 0.683, 0.888, and 0.941 ASR on CIFAR-10 ResNet-18. This failure is not explained by spectral compactness, spatial locality, or subspace misalignment. It is predicted by a logit-triplet gap involving the target margin, target-logit drop, and non-target logit rise. We then introduce InstantForget, a clean-calibrated gated reset that flags anomalous features with a Mahalanobis score and moves only flagged features toward a neutral non-target representation. With one fixed operating point selected on held-out triggered validation, InstantForget reduces average ASR to 0.071 across four non-adaptive CIFAR-10 triggers without triggered samples or parameter updates at deployment. It also reaches 0.981 detection AUROC and transfers to six of eight tested backbones. Reported failures under WaNet, ModelNet10 point blend, two backbone geometries, and adaptive feature-compactness attacks define the method's scope. 

---
# The algebra of Krom logic programs 

**Authors**: Christian Antić  

**Link**: [PDF](https://arxiv.org/pdf/2606.15719)  

**Abstract**: This paper investigates the algebraic structure of Krom logic programs, consisting only of facts and rules with at most one body atom. We show that sequential composition endows the class of Krom programs with a natural monoid structure and that this structure admits rich algebraic extensions to Krom seminearrings, Krom quemirings, Krom-Conway seminearrings, and Krom-Conway omegaseminearrings. Furthermore, we establish explicit generating sets and canonical decompositions, study the associated ${}^\omega$-operator, characterize the Kleene star in graph-theoretic terms, and relate finite Krom monoids to transformation monoids and finite-state automata. These results provide new connections between logic programming, algebraic automata theory, and algebraic graph theory. 

---
# Odds Law: The Decomposition Algebra On How Intelligence Organizes Itself to Solve Difficult Problems Reliably 

**Authors**: Hidayet Aksu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15712)  

**Abstract**: We ask a structural question: given unreliable elementary problem-solvers, what organizations of them solve hard problems reliably, and what are the limits? We develop a $decomposition~algebra$: elementary solvers are morphisms in a stochastic category, and four combinators (sequential composition, parallel ensembling, verification gating, and recursive reduction) generate the space of compound solvers. We equip this algebra with two homomorphisms, a $reliability$ valuation into the ordered monoid $([0,1],\le)$ and a $cost$ valuation into a commutative semiring, and we derive the composition laws that govern how reliability flows through structure. Our central results are (i) a $verification~odds~law$ (the result that names this report), showing that a verification gate multiplies the odds of correctness by the verifier's likelihood ratio $\Lambda$, so that $k$ conditionally independent gates yield geometric amplification; (ii) a $reliability~amplification~theorem$, giving target reliability $1-\delta$ at $O(\log 1/\delta)$ verification depth whenever $\Lambda>1$; and (iii) a $threshold~dichotomy$: above the critical parameters reliability can be driven arbitrarily close to one at logarithmic cost, while at or below them no amplification is possible. We then show that $self-organization$ is the least fixed point of a monotone improvement operator on the complete lattice of strategies, and that this fixed point equalizes marginal log-odds gain per unit cost. Finally, we prove matching limits: an information ceiling bounds per-gate amplification by a divergence quantity; shared error causes create a strictly positive voting floor, so diversity is $necessary$ for unbounded amplification. Reliability, in short, is neither free nor magical: it is bought with independent information, arranged by composition, and bounded by the verifier. 

---
# When Generator Replay Degrades: Projected Rehearsal Orchestration for Heterogeneous Federated Class-Incremental Learning 

**Authors**: Thinh T. H. Nguyen, Khoa D. Doan, Binh T. Nguyen, Danh Le-Phuoc, Kok-Seng Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.15695)  

**Abstract**: Federated class-incremental learning (FCIL) becomes substantially harder when clients observe different label subsets, progress through tasks at different stages, and provide uneven supervision for the same semantic concepts. Existing FCIL methods often preserve old knowledge through input-space synthesis, but they can be fragile under heterogeneous task streams and difficult to transfer across modalities. To alleviate such issues, we propose PRO, a framework that replaces synthetic input replay with projected rehearsal orchestration. To remove external pretraining, we evaluate all methods under the same warmup. After this, PRO maintains compact class-level projected memories on the server and allows clients perform balanced pseudo multi-task training over current examples and old projected memories. To handle stronger representation drift, we further introduce PRO-MAX, which augments PRO with neighborhood-weighted memory alignment while preserving the same server-light principle that the server only aggregates model updates and memory statistics. Across image, text, and graph benchmarks, PRO and PRO-MAX improve retention and final utility under heterogeneous streams while remaining competitive in homogeneous FCIL. Even when baselines are given expanded replay budgets, they degrade under supervision imbalance and stage misalignment, indicating that replay quantity alone does not resolve replay-quality failures. Additional weak-task diagnostics further show that larger replay mismatch is associated with larger downstream degradation, while our method keeps projected memories better aligned with the evolving representation. 

---
# MAF: Multimodal Adaptive Few-shot Prompting for Sentiment Analysis with MLLMs 

**Authors**: Hangling Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.15694)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in understanding complex multimodal content. However, their performance in sentiment analysis exhibits acute sensitivity to prompt design, rendering static, uniformly applied prompts inherently suboptimal for capturing the nuanced multimodal cues that vary across inputs. To address this limitation, we propose a Multimodal Adaptive Few-Shot Prompting (MAF) framework, which dynamically retrieves and integrates query-relevant demonstrations to elicit the sentiment reasoning capabilities of MLLMs in a context-sensitive manner. MAF constructs a demonstration retrieval module that holistically encodes facial expressions, scene context, and textual semantics, with a lip movement amplitude detection mechanism introduced for accurate speaker identification in multi-person scenarios. Departing from conventional fixed-weight fusion, a lightweight coefficient generation network is trained to output query-conditioned fusion weights in real time, enabling weighted aggregation of multimodal similarity scores to retrieve the top-K most informative demonstrations. Prediction stability is further enhanced through majority voting over multiple candidate outputs generated by the MLLM. Extensive experiments on public benchmark datasets demonstrate that MAF achieves substantial and consistent performance improvements over the corresponding backbone variants and remains competitive with strong multimodal sentiment-analysis baselines. 

---
# Imperfect Visual Verification for Code Edition : A Case Study on TikZ 

**Authors**: Charly Reux, Mathieu Acher, Djamel Eddine Khelladi, Clément Quinton, Olivier Barais  

**Link**: [PDF](https://arxiv.org/pdf/2606.15693)  

**Abstract**: LLMs have significantly advanced code generation, enabling the synthesis of functional programs. While recent systems achieve strong performance on many coding benchmarks, tasks involving programs such as TikZ that generate visual artifacts remain challenging, in particular on visual code customization. Unlike generation from scratch, customization requires localized, semantics-preserving edits: the model must locate relevant code, modify it according to the instruction, and preserve the remaining structure and rendering. Approaches based on post-hoc iterative refinement/correction where a verifier provides feedback to guide corrections, have shown promise. However, in the case of programs with a visual outcome such as in TikZ, where correctness is harder or likely impossible to formalize and evaluate automatically, deterministic verifiers do not exist. Hence, developers can only rely on imperfect verifiers. In this paper, we conduct an empirical study to answer:to what extent can iterative refinement remain effective when the verifier itself is unreliable?} We use TikZ as a focused case study that isolates the core difficulties of the problem (weak code structure, fine-grained visual semantics, and difficult feature localization) in a controlled and challenging setting. We define visual code customization as an iterative editing problem with an imperfect oracle, and introduce a framework for analyzing such iterative refinements. We conduct a large-scale study and evaluate multiple LLM-based and tool-augmented visual verifiers within iterative refinement pipelines, and perform extensive manual annotation of refinement trajectories to assess verifier behavior and feedback quality. Our findings show that even imperfect verifiers can determine with moderate accuracy whether visual instructions are applied to code, achieving F1-scores up to 0.815. Feedback improves iterative refinement, especially for weaker models, adding 11--20 perfect customizations for Qwen3-vl-30b-a3b-Instruct, while stronger models like Gemini-3 gain fewer improvements (+5) but benefit more from accurate verification that prevents premature acceptance. Feedback is effective only when it precisely identifies image issues, provides actionable guidance, addresses all relevant problems, and remains grounded in the original instruction. 

---
# The Reservoir Attention Network: Cross-Pass State in Pretrained Transformers via Content-Addressable Reservoir Injection 

**Authors**: Emma Leonhart  

**Link**: [PDF](https://arxiv.org/pdf/2606.15678)  

**Abstract**: A feasibility and dynamics study of the Reservoir Attention Network (RAN), an architecture that injects a fixed, randomly-initialized reservoir into the mid-layer attention of a pretrained transformer to carry state across forward passes. Experiments span GPT-2 (124M, 355M) to Qwen2.5 (0.5B, 1.5B) on a single consumer GPU. The tasks are minimal probes chosen to isolate individual mechanisms; the broader always-alive agent vision is treated throughout as compute-limited future work, not a claim of this paper. The reservoir is left untrained (fixed random) by design: this isolates whether untrained recurrent dynamics alone suffice to carry usable cross-pass state, leaving trained recurrence as a complementary, more expensive direction. 

---
# Z-Plane Neural Networks: Bounded Geometric Activation Replaces ReLU and LayerNorm 

**Authors**: Sungwoo Goo, Hwi-yeol Yun, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2606.15669)  

**Abstract**: Modern deep neural networks rely on Euclidean scalar activations (e.g., ReLU) and global normalization techniques (e.g., LayerNorm) to prevent gradient instability in deep architectures. However, these mechanisms inherently cause dead neurons, discard critical directional information, and destroy the orthogonality of feature representations. Inspired by the frequency-modulation transmission of biological axons, we propose the Z-Plane Neural Network, which maps hidden states into 2D phasor bundles on a hypersphere. We introduce a novel geometric activation function, Radial Bounding($\mathbf{x} / \max(1, \|\mathbf{x}\|_2)$), which limits the energy magnitude while preserving the phase (direction). We demonstrate mathematically that this isotropic activation maintains 1-Lipschitz continuity and prevents gradient vanishing by preserving tangential gradients. Empirically, a 100-layer Z-Plane Multi-Layer Perceptron (MLP)-entirely devoid of ReLU and LayerNorm-successfully converges on the MNIST dataset with 98.34% accuracy and absolute numerical stability, proving that bounded geometric activation alone is sufficient for stable deep learning. 

---
# PO-PDDL: Learning Symbolic POMDPs from Visual Demonstrations for Robot Planning Under Uncertainty 

**Authors**: Wenjing Tang, Xuanjin Jin, Yuan Liu, Renming Huang, Cewu Lu, Panpan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.15654)  

**Abstract**: Real-world robot task planning must operate under both stochastic action execution and partial observability, yet constructing Partially Observable Markov Decision Process (POMDP) models for real robotics domains remains difficult and labor-intensive. We introduce PO-PDDL, a symbolic formulation of POMDPs that preserves the relational structure and LLM-friendly syntax of the Planning Domain Definition Language (PDDL), while explicitly modeling partial observability, stochasticity, and beliefs. Building on this formulation, we propose a demonstration-driven pipeline for learning PO-PDDL models. The proposed method reconstructs latent symbolic state trajectories from real-robot execution videos, identifies partial observability via inconsistencies between inferred states and visual observations, and learns stochastic transition and observation models accordingly. The resulting PO-PDDL domains are reusable across tasks and enable online belief-space planning under both perception and execution uncertainty. Experiments on real-world long-horizon manipulation tasks show that our method consistently outperforms existing PDDL and POMDP model-learning approaches, achieving robust task planning under uncertainty with significantly lower planning cost. 

---
# IoT-Zoo: A Container-Based Framework for Heterogeneous IoT Device Profiles and Reproducible Traffic Capture 

**Authors**: Vagner E. Quincozes, Diego Kreutz, Silvio E. Quincozes  

**Link**: [PDF](https://arxiv.org/pdf/2606.15653)  

**Abstract**: The validation of networking and security solutions for the Internet of Things (IoT) requires realistic and reproducible experimental data. However, existing platforms often achieve scalability by replicating a limited set of device types, which restricts profile diversity and fails to capture the heterogeneity of real-world IoT environments. In this paper, we present IoT-Zoo, a container-based testbed designed to support reproducible experimentation through heterogeneous, dataset-driven IoT device profiles. Built upon Containernet, IoT-Zoo automates the deployment of multi-domain scenarios and supports real application protocols such as MQTT and RTSP. The platform provides a single-command interface for environment provisioning and automated traffic capture (PCAP), enabling the generation of consistent traffic baselines and reducing the operational effort required to evaluate networking and security solutions. 

---
# AnonShield: Scalable On-Premise Pseudonymization for CSIRT Vulnerability Data 

**Authors**: Cristhian Kapelinski, Douglas Lautert, Beatriz Machado, Diego Kreutz, Isadora Garcia Ferrão  

**Link**: [PDF](https://arxiv.org/pdf/2606.15650)  

**Abstract**: We present AnonShield, a high-throughput, on-premise pseudonymization system that combines GPU-accelerated NER, streaming processing, caching, and schema-aware configuration. Evaluated on datasets up to 550 MB (70,951 records), AnonShield reduces processing time from over 92 hours to under 10 minutes (up to 738x speedup) while achieving up to 94.2% F1-score and 96.7% recall. Our results show that scalable pseudonymization of vulnerability data is feasible without sacrificing analytical utility, enabling compliant data sharing in operational CSIRT environments. 

---
# CIWI-CKT: Chaos-Informed Wave Interference Feature Fusion and Cross-City Knowledge Transfer for Traffic Flow Forecasting 

**Authors**: Abdul Joseph Fofanah, Lian Wen, David Chen, Shaoyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15642)  

**Abstract**: Accurate traffic flow prediction remains challenging in cross-city, data-scarce scenarios where limited historical data hinders model generalisation. The chaotic nature of traffic dynamics, complex spatio-temporal dependencies, and heterogeneous urban networks complicate few-shot learning across cities. Existing deep learning approaches either treat traffic as purely deterministic or lack mechanisms to model wave-like interference patterns essential for cross-regime traffic dynamics. To address these limitations, this paper proposes CIWI-CKT, a novel Chaos-Informed Wave Interference Feature Fusion framework with Cross-City Knowledge Transfer. Our framework introduces three core innovations: chaos-informed wave generation that extracts measurable chaos invariants and models traffic as adaptive wave components; meta-interference processing that captures wave interactions between support and query regimes while producing a predictability score for confidence estimation; and chaos-aware meta-learning that enables efficient cross-city knowledge transfer while preserving chaotic characteristics. We establish theoretical guarantees including chaos-to-wave stability, wave-induced dimension reduction, and meta-learning generalisation bounds. Extensive experiments on four real-world traffic datasets demonstrate that CIWI-CKT significantly outperforms state-of-the-art spatio-temporal graph learning, transfer learning, prompt-based, and few-shot methods, improving prediction accuracy while substantially reducing required training data. 

---
# Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time 

**Authors**: Jeongeun Park, Juhan Park, Taekyung Kim, Sungjoon Choi, Dongyoon Han, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2606.15631)  

**Abstract**: Extending a vision-language-action (VLA) policy to a new task typically requires task-specific teleoperated demonstrations and per-task fine-tuning, making adaptation costly in both data collection and compute. In this paper, we show that this target-side per-task adaptation cost can be replaced by retrieval. Our retrieval-augmented policy is trained once on paired demonstrations from the target embodiment (query) and a cheaper embodiment (pool, e.g., human-hand video), then frozen. New tasks are added at deployment by appending pool-side demonstrations to a retrieval pool. The frozen policy conditions on retrieved trajectories at every control step, so new tasks are absorbed by indexing data rather than updating parameters. Fine-tuning is needed only to take on a new, unseen embodiment, not for each new task. We show that retrieval improves policies beyond a specific backbone, including standard VLA policies, but its effect is especially pronounced in Cosmos Policy, a video-generation-based world-action model (WAM). In this setting, retrieval supplies coarse task progression, while the WAM's future-image objective provides an additional visual consistency signal that strengthens the retrieval-conditioned actions. On PushT, we study how retrieval provides a reusable high-level motion prior for cross-embodiment generalization to unseen goal angles, while on RoboTwin 2.0 our method outperforms cross-embodiment baselines on unseen tasks, and we additionally demonstrate the method on a real robot. 

---
# Surprise-Guided MergeSort: Budget-Efficient Human-in-the-Loop Ranking via Adaptive Comparison Scheduling 

**Authors**: Yujin Park, Haejun Chung, Ikbeom Jang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15623)  

**Abstract**: Pairwise comparison is the gold standard for subjective ranking tasks; however, exhaustive annotation requires a massive number of human comparisons ($O(n^2)$). While sorting-based methods have reduced this burden to $O(n\log n)$, they still require expensive human judgment for every single comparison. To further improve annotation efficiency, we propose leveraging a Vision-Language Model (VLM) not as an annotator replacement, but as a \emph{question prioritizer} to identify which comparisons genuinely require human judgment. The proposed \textbf{Surprise-Guided MergeSort (SGS)} framework achieves this through three integrated components: (1) a bottom-up MergeSort scheduler that structures comparisons and exploits transitivity, (2) a composite Surprise Scorer -- combining position-bias-cancelled VLM confidence, Elo gap, and vote entropy -- to quantify comparison ambiguity, and (3) an adaptive budget allocator that routes high-surprise pairs to humans while automating low-surprise pairs via transitivity inference. Validation was conducted on six diverse benchmarks spanning text similarity (STS-B, BIOSSES, SICKR-STS) and image quality assessment (KonIQ-10k, TID2013, LIVE Challenge). SGS effectively identified and skipped up to 535 non-informative comparisons per session. Consequently, it achieved Kendall's $\tau{\times}100$ improvements of $+6$ to $+12$ over Active Elo under the same total budget. These results demonstrate that combining VLM-guided surprise metrics with algorithmic sorting provides a generally consistent accuracy-efficiency trade-off across diverse domains. 

---
# Mutual Distillation of Dual-Foundation Models for Semi-Supervised PET/CT Segmentation 

**Authors**: Fuyou Mao, Beining Wu, Yanfeng Jiang, Bohan Xu, Lixin Lin, Naye Ji, Hao Zhang, Yan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15611)  

**Abstract**: Organ segmentation from PET/CT is critical for quantitative analysis and radiotherapy planning in oncology. To ease the high annotation cost of PET/CT segmentation, semi-supervised learning (SSL) provides a practical and effective solution for developing deep models with limited labeled data. Recent developments in visual foundation models have demonstrated remarkable adaptability with improved efficiency. In this work, we propose a mutual distillation framework that seamlessly exploits both structural and functional foundation models, which act as modality-specific generalists for distilling knowledge from structural CT and metabolic PET imaging. By bridging the gap between the task-specific precision of student models and the segmentation priors of generalist foundation models, we propose \textbf{MuDuo}, a mutual distillation framework that synergistically leverages SAM-Med3D for CT and SegAnyPET for PET to distill their knowledge into a lightweight student network. Our approach eliminates the need for manual prompts while maximizing the utility of unlabeled data for automatic segmentation, achieving state-of-the-art performance on the AutoPET dataset with only 5 labeled cases. Our source code is available at this https URL. 

---
# LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge Evaluation 

**Authors**: Hiroyasu Usami, Keisuke Hara, Ayato Tsuboi, Naohiko Matsuda  

**Link**: [PDF](https://arxiv.org/pdf/2606.15610)  

**Abstract**: LLM-as-a-judge systems are now routinely used for open-ended model evaluation, where human preference annotation is costly, slow, and difficult to reproduce. Yet these judges are often reported as scalar accuracy, win-rate, or agreement devices. We argue that a judge should instead be reported as a measurement instrument. We introduce a Judge Datasheet protocol that measures dark current under true-vacuum inputs, stable cross-sensitivity to same-quality surface variation, positional false preference, target sensitivity on a controlled quality ladder, and the criterion or operating point induced by tie instructions. The direction-stability decomposition reveals that apparent Delta0 preference can be stable surface response or disguised position bias. In a three-judge open-weight case study, Llama-3.1-8B shows high dark current and presentation-conflicted Delta0 behavior, Qwen2.5-14B is vacuum-clean and target-sensitive but mixes stable and positional over-discrimination, and Qwen2.5-32B is vacuum-clean with low stable cross-sensitivity and low positional false preference. A strict tie criterion eliminates Qwen32B Delta0 false preference but absorbs marginal Delta1 target signals into ties while preserving Delta5 sensitivity. The results show that prompting moves the criterion, not the resolution. We do not claim that the downstream mechanism hypothesis that motivated this work is confirmed; the contribution is a metrological protocol for measuring the measuring device before downstream claims are made. 

---
# FragFuse: Bypassing Access Control of Large Language Model Agents via Memory-Based Query Fragmentation and Fusion 

**Authors**: Zixin Rao, Wentian Zhu, Chan Aristella Lu, Zhaorun Chen, Wei Niu, Le Guan, Bo Li, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15609)  

**Abstract**: Large language model (LLM) agents increasingly rely on long-term memory to support complex task execution, user personalization, and domain adaptation. Meanwhile, emerging access-control mechanisms for LLM agents are being explored to block policy-violating requests and prevent misuse. We reveal a novel attack surface arising from agent memory operations: prohibited content that would trigger access control can be fragmented across interactions, stored in long-term memory in benign-appearing form, and later reconstructed through memory retrieval without appearing explicitly in the final user query. We propose FragFuse, the first attack that enables unprivileged users to bypass agent access control by exploiting this temporal channel introduced by long-term memory. FragFuse operates in three stages: (1) identifying rejection-responsive fragments via black-box adaptive querying with fragment masking; (2) injecting these fragments into memory using marker carrier queries; and (3) retrieving and fusing the stored fragments through a follow-up attack query. Although FragFuse can be instantiated manually for individual agents, we further develop a surrogate-based optimization scheme that tunes fusion instructions and marker designs, enabling automated attack generation without violating the attacker's threat-model assumptions. We evaluate FragFuse across four representative agent settings and task domains, covering three state-of-the-art agent access-control mechanisms. FragFuse achieves an average bypass success rate of 86.3% and an average end-to-end harmful task success rate of 41.1% across all settings, with only 4.4% average task-success degradation compared with configurations without access control. We also show that alternative defenses, including state-of-the-art prompt-injection detectors and perplexity detectors, do not effectively address this attack. 

---
# SCAN: A Decision-Making Framework for Effective Task Allocation with Generative AI 

**Authors**: Fendi Tsim, Alina Gutoreva  

**Link**: [PDF](https://arxiv.org/pdf/2606.15601)  

**Abstract**: We introduce SCAN -- a human-centric decision-making framework to facilitate learners for effective task allocation with Generative Artificial Intelligence (GenAI) based on Vygotsky's Zone of Proximal Development and Metacognition. In SCAN, we systematize and formalize AI-human interaction by introducing a task-identification approach with four "sub-zones": Substitute, Complement, Aid, and Non-negotiable. After describing the four sub-zones, we demonstrate how SCAN framework can be applied for knowledge workers in the workplace and students in education to metacognitively "scan" their use of Generative AI. We then discuss how such framework can be related to cognitive load theory, cognitive offloading, sycophancy, three decision-making modes in human-AI interactions (automation, augmentation, and collaboration), future of work such as upskilling and deskilling, and how it accounts for both human-human and human-AI learning. We propose that SCAN offers a great starting point before discussing whether GenAI complements or replaces our abilities when completing a task, with a general objective of sustaining lifelong learning, and a specific goal of reaching hybrid intelligence. 

---
# Pixels to Proofs: Probabilistically-Safe Latent World Model Control via Parallel Conformal Robust MPC 

**Authors**: Devesh Nath, Anutam Srinivasan, Haoran Yin, Ruitong Jiang, Jeffrey Fang, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2606.15594)  

**Abstract**: We present SLS^2, a framework for safe feedback motion planning from pixels using robust model predictive control (MPC) in learned latent world models. Our approach trains an action-conditioned joint-embedding world model with compact Markovian latent states, enabling efficient gradient-based trajectory optimization through learned latent dynamics. To enforce safety for the true system despite imperfect latent predictions, we inform a GPU-accelerated system level synthesis (SLS) robust MPC scheme with conformal prediction to obtain calibrated latent error bounds and robust latent-space constraint sets. We further learn and conformalize a latent constraint checker, allowing the SLS planner to impose probabilistic safety constraints during closed-loop execution. We evaluate our method on vision-based control tasks, where it improves both goal-reaching performance and safety over latent world-model and safe-planning baselines. 

---
# Is Code Better Than Language for Algorithmic Reasoning 

**Authors**: Terry Tong, Yu Feng, Surbhi Goel, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2606.15589)  

**Abstract**: For tool-augmented language models, comparing natural-language reasoning with code-execution pipelines is difficult because the comparison changes both the intermediate representation and the execution mechanism. We separate these factors with an intermediate intervention: the model expresses its reasoning as executable code, and the language model simulates that code in context to produce an answer. On a 40-task verifiable algorithmic benchmark, deterministic code execution outperforms natural-language reasoning by +31.6pp. We observe that the intermediate intervention is not meaningfully different from natural-language reasoning (+0.15pp). These results suggest that, in our evaluated setting, changing the intermediate representation alone does not explain the tool-use advantage, providing evidence for the performance gains requiring reliable external execution. We formalize this intuition with a simple statistical decision-theoretic model that characterizes when execution dominates end-to-end risk in our disentangled trace-generation/execution regime. We validate our theory using a reconstruction intervention that leverages a proxy language model to infer natural-language reasoning traces from code representations, recovering performance comparable to the original natural-language reasoning pipeline. All experiments are at this https URL. 

---
# Localizing Credit at the Divergence: Path-Conditioned Self-Distillation for LLM Reasoning 

**Authors**: Yu Li, Shu Hong, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15576)  

**Abstract**: Reinforcement learning from verifiable rewards assigns a single scalar to each rollout, leaving token-level credit assignment underspecified in long reasoning traces. On-policy self-distillation addresses this by letting the same model act as a teacher conditioned on privileged information, producing a dense per-token signal. But the common choice of a ground-truth answer is only an endpoint cue: on terse-answer tasks, the teacher falls silent at the intermediate positions where path-level guidance matters most. We propose Hindsight Self-Distillation (HSD), which conditions the teacher on a successful peer rollout drawn from the current training group. Such a peer is an exact sample from the success-conditioned policy, requiring no additional sampled rollouts. By providing a full successful continuation rather than only the final answer, the resulting credit signal concentrates at the divergence position between a failed rollout and a successful peer. Across Qwen3-8B and Qwen3-32B on math and code benchmarks, HSD obtains the best result against GRPO variants and on-policy distillation baselines, with the largest gains on terse-answer tasks such as AIME. 

---
# LLM-Assisted Stance Detection in Scientific Discourse: A Test Case in Bayesian Cognitive Science 

**Authors**: Eyup Engin Kucuk, Tarik Kelestemur, Ömer Dağlar Tanrikulu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15566)  

**Abstract**: Qualitative coding is central to social science, but expert annotation is difficult to scale. LLMs offer a possible extension, yet require careful validation when the target construct is interpretive, theoretically loaded, and only indirectly expressed. We study this problem in a difficult case: detecting whether authors treat Bayesian models as descriptions of mental and neural mechanisms (realism) or as useful mathematical tools (instrumentalism). Our method combines a theory-driven codebook, expert-coded reference annotations, a diagnostic-gated prompt-optimization search yielding a shared zero-shot prompt for three frontier LLMs (GPT-5.1, Claude Sonnet 4.6, Gemini 3 Pro Preview), and multi-rater reliability analysis. The final prompt achieved a held-out combined reliability score of 0.76 (harmonic mean of ICC = 0.79 and $\alpha$ = 0.74), with all diagnostics satisfied. Deployed on 6,858 quotes from 210 articles, the three LLMs reached substantial quote-level agreement (ICC = 0.80; $\alpha$ = 0.76; combined = 0.78) and near-perfect article-level rank stability ($r$ = 0.96-0.97 across rater pairs). The corpus was predominantly weakly realist, but article-level stances were rarely uniform: only 1.4% of articles used a single band, while 59.5% spanned four or more. Low-level perception/motor articles scored 8.8 Realism points higher than high-level cognition articles ($p < .001$, $d = 0.60$), quantifying a long-held qualitative intuition. We present this as an expert-led case study; the framework is intended to generalize to similar theoretically demanding tasks, not to all qualitative analysis. 

---
# Service-Induced Congestion in Memory-Constrained LLM Serving 

**Authors**: Ruicheng Ao, Jing Dong, Gan Luo, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15555)  

**Abstract**: In large language model (LLM) serving, each request accumulates persistent graphics processing unit (GPU) memory during service as its key-value cache grows with every generated token. Under high concurrency, aggregate memory usage therefore increases endogenously over time: the service process itself creates future capacity pressure. When memory capacity is exceeded, systems evict active requests, discarding cached state and restarting them later, which wastes computation and reduces throughput. We develop a discrete-time dynamical model of memory-constrained LLM inference that captures admission, memory growth, and eviction under continuous batching. In the saturated-input regime, the system admits both eviction-free fixed points and limit cycles with evictions. For homogeneous workloads, we show that the eviction-free equilibrium is unstable and that, except for a Lebesgue-measure-zero exact-capture set, the system converges to a unique worst-case limit cycle that is asymptotically stable outside this exceptional set, with throughput losses as large as 50%. For heterogeneous workloads, we prove a stability criterion in the two-class common-input setting and explain how the survival-polynomial mechanism generalizes to multiple classes and heterogeneous-input lengths. Under an input-dominated scaling regime, coprime decoding lengths stabilize the eviction-free equilibrium, while non-coprime lengths create synchronized modes that drive instability. These results characterize when workload heterogeneity desynchronizes completions and helps stabilize memory-constrained serving. More broadly, we identify service-induced congestion as a structural instability mechanism and derive scheduling design principles for sustaining high throughput. 

---
# Distilling Drifting Transformers with Representation Autoencoders 

**Authors**: Jiawei Zhang, Mengfei Xia, Gen Li, Yuantao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15553)  

**Abstract**: Representation Autoencoders (RAEs) have improved diffusion and flow models by semantically richer latent space owing to the strongly label-wise clustered DINO features in the pretrained encoders. Yet in the distillation stage, the severe anisotropy and large curvatures caused by the rich semantic representations would hinder the convergence and performance, making the trajectory-based distillation unstable. In this work, we argue that the RAE latent space is compatible with distillation via the newly proposed Drifting Models. We first quantitatively study the curvatures and isotropy statistics across different autoencoders, and theoretically reveal that Drifting Model itself is highly likely to fail on extremely scattered spaces like reconstruction-based VAEs. These motivate us to apply the drifting paradigm directly to representation autoencoders. Our proposed method, Drift-RAE, distills pretrained flow models in RAE latent spaces using Drifting, together with insightful modifications that improve training stability by thereotically aligning drifting fields with other frameworks. Regarding the experimental evidences, we achieve 1.77 FID on ImageNet 256 dataset using only 10k distillation steps, surpassing state-of-the-art RAE distillation methods and appearing comparative with the original Drifting Model without requiring an auxiliary MAE feature extractor. The code will be made publicly available. 

---
# CmdNeedle: Measuring the Incompleteness of Command Denylists for AI Agents 

**Authors**: Chuyang Chen, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15549)  

**Abstract**: The adoption of AI agents is increasing rapidly. Terminal AI agents, i.e., AI agents that run in terminal environments, are a widely used type of AI agents. Terminal AI agents rely heavily on shell command execution to interact with the host systems. They adopt a three-list command-gating mechanism to mitigate security risks introduced by command execution, with denylists serving as the load-bearing component. However, modern operating systems often ship a large, ever-expanding set of shell commands with complex functionalities. Our observation is that even a built-in denylist of Claude Code, well-maintained by its developers, can overlook bypass commands that invalidate its effectiveness. Such negligence leads to fragile command denylists that cannot even block operations that practitioners expect them to block.
This paper presents the first systematic characterization of command denylist fragility in terminal AI agents. The paper formalizes the command denylist fragility problem and proposes an LLM-driven pipeline, CmdNeedle, to detect such fragility. It prompts the LLM to propose possible bypasses and iteratively repairs them using feedback from a validator that executes them in a sandbox. In the evaluation, we applied CmdNeedle to 1,709 real-world command denylists (containing 13,332 denylist rules) collected from GitHub. The evaluation shows several key findings, including that 69.0--98.6% of the denylists are fragile, that this fragility occurs consistently across projects and agents, and the validity of several possible root causes for this fragility. Our pipeline and findings will hopefully facilitate future research and practice regarding the command denylists used by AI agents. 

---
# EcoBin: A Two-Stage Deep Convolutional Neural Network for Contamination-Aware Waste Classification 

**Authors**: Raghav Senthil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.15547)  

**Abstract**: Waste classification models have become highly accurate at sorting waste, often exceeding 95% on benchmark datasets. However, these models fail to account for contamination in recyclable waste. We present EcoBin, a two-stage deep convolutional neural network that classifies household waste by its disposal pathway and that explicitly accounts for contamination. The first stage is a base waste classifier built on an EfficientNetV2-S backbone that assigns each of the thirty waste categories in our dataset to one of four disposal pathways. The second stage is a contamination classifier that inspects any item routed toward recycling and overrides the decision to garbage when contamination is detected. Because no public dataset of contaminated recyclables exists, we synthesize one by segmenting images of clean recyclable objects with a U2-Net model and compositing realistic contamination textures onto their surfaces. The first stage achieves 87.42% test accuracy and a 96.13% pathway-adjusted accuracy. Meanwhile, the contamination stage distinguishes clean from contaminated items with a 0.99 ROC-AUC. On a test set of contaminated recyclables, the complete pipeline routes 24 of 25 items correctly, compared with only 1 of 25 for the base classifier alone. A McNemar's test confirms that the improvement contributed by the contamination stage is statistically significant (p < 0.001). 

---
# AP-GRPO: Anchor-Gated Phonetic Alignment with Policy Optimization for Pathological Speech Reconstruction 

**Authors**: Pengfei Zhang, Hoang H Nguyen, Yutong Song, Wenjun Huang, Tahmid Imtiaz Imu, Henry Peng Zou, Jiang Wu, Honghui Xu, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2606.15540)  

**Abstract**: Pathological speech from patients with neurodegenerative and neuromotor disorders is often acoustically distorted and linguistically fragmented, making pathological speech reconstruction necessary to recover intended textual content from distorted and incomplete speech recordings. Crucially, such recordings are rarely uniformly degraded: some words or short phrases remain reliable and can serve as audible anchors for reconstructing the corrupted surrounding content. We introduce Anchor-gated Phonetic Group Relative Policy Optimization (AP-GRPO), a GRPO framework with phonetic reward that aligns speech language models (SLMs) through audible-anchor preservation and inter-anchor phonetic compatibility to the original speech signal. AP-GRPO consists of: (i) an anchor-gated reward that matches reliable audible anchors in clear regions; and (ii) an inter-anchor phonetic alignment reward that evaluates whether recovered contents are phonetically supported by the corresponding corrupted inter-anchor speech span. Across four disease conditions, AP-GRPO improves faithful speech reconstruction, and the learned anchor constraint automatically adapts to each condition and thus reveals interpretable disease-specific profiles: conditions with severe articulatory degradation require stronger anchor enforcement, whereas milder impairment or linguistically impaired conditions rely more on phonetic alignment for inter-anchor recovery. 

---
# MADAR: An Address-Free Processor 

**Authors**: Mohamed Amine Bergach  

**Link**: [PDF](https://arxiv.org/pdf/2606.15535)  

**Abstract**: In a modern processor, computing is the cheap part. Most of its area and energy go to \emph{addressing} -- moving operands to and from a register file and cache, and running the tags, ports, miss queues, and bypass networks that find a value where it was left. MADAR deletes that machinery by abolishing the address. All state circulates in rings of slots that advance one position per clock; instructions and data ride in the same slots; a value is named by its place in an orbit -- a \rp{} coordinate -- not by an address; a fixed station computes when a circulating instruction sweeps past its operands, on a schedule set at compile time; and a hierarchy of rings of increasing period replaces the cache hierarchy, movement between them scheduled rather than triggered by a miss. No prior circulating-store, dataflow, or statically scheduled machine combines all four of these. We define the execution model, validate it in a cycle-accurate register-transfer-level implementation, show it \emph{compilable} -- a constructive scheduler emits programs cross-checked against the implementation -- and price it with a first-order energy model. The payoff is clearest for AI acceleration: the multiply-accumulate at the heart of every matmul and convolution compiles to a streaming form whose energy per operation stays flat as the reduction grows, and the operand reuse that makes matrix multiplication efficient is carried by the ring-period hierarchy -- the memory hierarchy doing by rotation what a cache does by tags. MADAR is a new design point for any computation whose data movement is known before the program runs. 

---
# Selective Synergistic Learning for Video Object-Centric Learning 

**Authors**: WonJun Moon, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2606.15527)  

**Abstract**: Typical video object-centric learning (VOCL) approaches employ slot-based frameworks that rely on reconstruction-driven encoder-decoder architectures, where learning is mediated by two spatial maps: attention maps from the encoder and object maps from the decoder. As these two distinct maps exhibit different properties, a recent dense alignment strategy attempted to reconcile this discrepancy by enforcing agreement across all spatio-temporal patches via contrastive learning. However, this indiscriminate alignment inadvertently propagates the inherent weaknesses of each module, such as noisy encoder predictions and blurred decoder boundaries. Moreover, computing dense similarities across all pairs incurs a computational cost quadratic in the total number of spatio-temporal patches, severely limiting scalability. Motivated by this, we propose Selective Synergistic Learning (SSync). Instead of exhaustive patch-to-patch alignment, SSync prevents error propagation by selectively distilling only the most reliable cues: leveraging the encoder strictly for boundary refinement and the decoder for interior denoising. This is realized via a pseudo-labeling with linear complexity, eliminating the need for quadratic spatial comparisons. Also, to prevent the reinforcement of architectural biases like slot redundancy, we introduce a transitive pseudo-label merging that consolidates overlapping slots based on spatio-temporal activation consistency. Extensive studies demonstrate that SSync improves decomposition quality and serves as a versatile, plug-and-play module while also exhibiting exceptional robustness to slot configurations. Code is available at this http URL. 

---
# AQ4SViT: An Automated Quantization Framework with Search Gating Policy for Compressing Spiking Vision Transformers 

**Authors**: Rachmad Vidya Wicaksana Putra, Saad Iftikhar, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.15523)  

**Abstract**: Spiking Vision Transformers (SViTs) have emerged as alternative low-power ViT models, but their large sizes hinder their deployments on resource-constrained embedded AI systems. To address this, state-of-the-art works proposed quantization techniques to compress SViT models, but their manual, human-guided approach needs a huge design time and power/energy consumption to find the appropriate quantization setting for each given network, making this approach not scalable for quantizing multiple networks. Toward this, we propose AQ4SViT, a novel automated quantization framework for SViTs that can provide quick quantization settings with good trade-offs between accuracy and memory. To achieve this, AQ4SViT employs the following key ideas: quantization search strategy that evaluates the quantization setting candidates while considering the accuracy constraint; and search gating policy that quickly evaluates and selects promising quantization candidates by leveraging membrane potential drift as a performance proxy. In the search gating policy, AQSViT employs two search algorithm variants to provide trade-off options: Greedy search, which performs fast but may lead to local optima; and Beam search, which performs slower but has better performance in finding global optima selection due to a wider search space. Experimental results show that AQ4SViT-Greedy quickly finds the appropriate quantization settings, achieving up to 6.6x faster search time and up to 82.5% memory saving compared to the state-of-the-art; while AQ4SViT-Beam further reduces the memory footprint by up to 90% compared to the state-of-the-art, but with 4.5x longer search time; all these results are obtained while maintaining high accuracy within 1.5% from the original/non-quantized models on the ImageNet dataset. These results highlight that AQ4SViT framework offers advancements toward SViT deployments on embedded AI systems. 

---
# LLM4RTL: Tool-Assisted LLM for RTL Generation 

**Authors**: Jing Jin, Robert Chu, Ning Yan, Masood S. Mortazavi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15500)  

**Abstract**: Large language models (LLMs) have facilitated impressive progress in software engineering, code generation, tooling, and systems. Concurrently, a significant body of research has developed which explores a growing variety of methods and systems for applying LLMs to hardware and chip design (e.g., systems for RTL code generation based on functional description). However, when it comes to open Verilog/RTL code-generation, we need high-quality training samples to build specialized and more effective LLM systems through fine-tuning or low-rank adaptation. Here, we propose a ``judge-renew-check-renew-check'' (JRCRC) pipeline which updates a current public dataset using a hierarchy of state-of-the-art commercial LLM models differing in their costs and capabilities in RTL code generation. This approach achieves a cost-effective mechanism for filtering and refining code-generation samples into a higher-quality training dataset. Our experiments also identify some common weaknesses of LLMs in rule-based reasoning and logic, and consequently, in RTL code-generation. Having identified these weaknesses, we develop an architecture for incorporating pre-processing tools to dynamically assist the LLMs in inferring logical relationships from tabular data formats. With our tools-assisted architecture for RTL code generation, we achieve significant overall performance gains in the VerilogEval benchmark and outperform many state-of-the-art methods. Our LLM4RTL system achieves performance comparable to that of GPT-4O using a significantly much smaller LLM. 

---
# The Perils of Agency: How Developers Perceive, Prioritize, and Address Risks in Agentic AI Products 

**Authors**: Hao-Ping Lee, Jessica He, David Piorkowski, Thomas Serban von Davier, Jodi Forlizzi, Sauvik Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.15485)  

**Abstract**: Agentic AI systems act autonomously, use tools, adapt to context, and operate in complex real-world environments. However, these same characteristics can create or exacerbate product risks. We studied how industry developers (n=35) perceive, prioritize, and address the risks in their agentic AI products. We found that developers' perceptions of risk were closely tied to the qualities that made the product agentic, such as autonomy, tool use, and usage in a real-world context. Developers prioritized product and business risks before considering downstream societal risks like job displacement and end-user privacy. This prioritization also impacted developers' ability and motivation to mitigate agentic risks. Finally, developers lacked mature controls for containing agentic risks, often relying on constraining the same characteristics that make agents useful: e.g., autonomy and goal complexity. These findings reveal a capability vs. risk control tension in agentic AI development: developers need to address risks that emerge from agentic capabilities, yet they currently have limited support for doing so without constraining agentic functionality. 

---
# Bayesian 3D Steerable CNNs: Enabling Equivariance and Uncertainty Quantification Simultaneously 

**Authors**: Abhishek Keripale, Ponkrshnan Thiagarajan, Susanta Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2606.15479)  

**Abstract**: Steerable convolutional neural networks (Steerable-CNNs) guarantee SE(3)-equivariance by parameterizing kernels as linear combinations of steerable basis functions, but their deterministic nature precludes uncertainty quantification - limiting their use in settings where confidence estimates are essential. We propose a Bayesian Steerable-CNN that places posterior distributions over the basis coefficients, yielding stochastic kernels while preserving equivariance exactly. The loss function of the model is obtained via variational inference and minimized by Bayes-by-Backpropagation. The framework admits a decomposition of predictive uncertainty into epistemic and aleatoric components. Empirically, the model attains competitive classification accuracy alongside an expected calibration error of 0.0263 and outperforms its deterministic counterpart by up to 6.17% under distributional shift induced by additive Gaussian noise. Furthermore, we leverage the model's uncertainty estimates to enhance its performance significantly, achieving a notable gain - approximately 4% higher accuracy across 84% of the test dataset. A statistically significant negative correlation between epistemic uncertainty and prediction error confirms that the learned posterior variance is semantically meaningful. The framework unifies Bayesian uncertainty quantification with the inductive bias of equivariant CNNs. 

---
# Understanding Diversity Collapse in RLVR via the Lens of Overtraining 

**Authors**: Suqin Yuan, Jinkun Chen, Jiyang Zheng, Muyang Li, Lei Feng, Dadong Wang, Tao Xiang, Tongliang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2606.15455)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key approach for enhancing the reasoning abilities of large language models. However, RLVR often suffers from \emph{diversity collapse}: Pass@$1$ improves while high-$k$ Pass@$k$ degrades, which is viewed as a narrowing of the model's reasoning boundary. We formalize this diversity collapse through the lens of \emph{overtraining}: once a problem's contribution to the reference metric has effectively saturated, further updates no longer expand what the model can solve but still concentrate probability mass on the trajectories favored by on-policy sampling. Under a standard setup with few rollouts per problem, even a single observed success places a problem in a nearly saturated regime for high-$k$ Pass@$k$, so most updates in standard RLVR are overtraining from the boundary perspective. This perspective also suggests a reading of whether RLVR can expand the model's reasoning abilities beyond the base model: since RLVR is structurally biased against high-$k$ Pass@$k$, its aggregate decline does not by itself mean that no new reasoning gains occurred. Interventionally, restricting updates to problems with zero observed success lifts Pass@$256$ above the base model on difficult benchmarks; observationally, a non-trivial fraction of initially unsolvable problems become solvable during standard RLVR training. Building on these findings, we propose \emph{Bayesian Boundary Gating} (BBG), which redirects optimization away from overtraining by estimating each problem's marginal contribution to the reasoning boundary. Across multiple reasoning benchmarks, BBG improves average Pass@$k$ across a wide range of $k$. 

---
# Defending against Adaptive Prompt Injection Attacks via Reasoning-enabled Task Alignment 

**Authors**: Lipeng He, Yihan Wang, Jiawen Zhang, N. Asokan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15441)  

**Abstract**: Indirect prompt injection attacks hijack LLM-based agents by embedding malicious instructions in third-party data that the agent retrieves during task execution. Existing defenses report near-zero attack success rate on static benchmarks, yet recent adaptive evaluations show that these results collapse once the attacker is allowed to optimize against the deployed defense. In this work, we trace this collapse to two failure modes. First, existing defense methods are confined to recognizing specific attack patterns, rather than assessing whether the intent of every embedded instruction is relevant to the user task. Second, training-based defenses, which otherwise offer the strongest safety-utility trade-off, assemble their adversarial examples from a handful of hand-crafted templates, and the resulting defender fails to generalize outside that narrow strategy distribution. To address these gaps, we propose RETA, a training-based method that grounds defense decisions on the user tasks rather than attacker-controlled data. At each tool-output step, the defender undertakes chain-of-thought reasoning verifying that its actions are consistent with the user task. Leveraging red-teaming, a simulated attacker synthesizes adversarial training data and receives a dictionary-learning diversity reward, achieving broad coverage of injection-reformulation strategies. Together, these allow the defender to be optimized via multi-objective reinforcement learning and achieve better safety-utility trade-off. Across six black-box adaptive attacks, RETA keeps every per-attack ASR below 10%, with average ASR of 2.92% and 3.75% on the two target models, while preserving most utility under attack and on clean inputs. 

---
# Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models 

**Authors**: Mayur Sanap, Prasanna Desikan, Edgar Lobaton  

**Link**: [PDF](https://arxiv.org/pdf/2606.15436)  

**Abstract**: Respiratory acoustic foundation models (FMs) excel at cough classification, yet their ability to predict continuous health quantities from cough audio remains largely unexplored, despite the clinical value of passive age, BMI, and disease probability estimation in settings where physical measurements are unavailable. We introduce the multi-model, multi-target cough regression benchmark evaluating five FMs (OPERA-CT, OPERA-CE, OPERA-GT, HeAR, M2D+Resp) across six targets on three datasets under subject-disjoint protocols, comparing linear, MLP-small, and full MLP regression heads. MLP-small beats the mean-predictor baseline on all tasks and linear probing in 23 of 30 model x task cases, with full MLP overfitting on small clinical data but recovering on larger sets, revealing a dataset size x head-capacity trade-off. HeAR leads within-dataset age regression on Coswara (9.12 yr MAE); its CIDRZ result is excluded from headline claims owing to possible HeAR-CIDRZ pretraining overlap. OPERA-GT is favored over OPERA-CT on age in all three datasets, with the CIDRZ margin within seed variance, extending a generative-pretraining advantage from breath to cough. HeAR and M2D+Resp reach near-full performance at N = 50 samples while OPERA models require N = 400. Cross-dataset transfer is strongly asymmetric as large diverse data generalises to small clinical populations (CoughVID to CIDRZ: -0.17 yr) but not vice versa (CIDRZ to Coswara: +2.43 yr, +26.6%). 

---
# Post-Launch Capability Expansion of Vision-Language Models via Prompting for On-Orbit Spacecraft Inspection 

**Authors**: Nicholas A. Welsh, Lennon J. Shikhman, Monty Nehru Attazs, Seemanthini K. Putane, Van Minh Nguyen, Ryan T. White  

**Link**: [PDF](https://arxiv.org/pdf/2606.15427)  

**Abstract**: Spaceborne inspection systems often deploy perception models prior to launch, after which updating model weights or expanding fixed label sets becomes operationally impractical. While supervised models can be integrated pre-flight, adding new semantic capabilities in orbit requires retraining and re-uploading parameters. We investigate whether prompt-driven vision--language models can enable post-launch semantic expansion, allowing new spacecraft components to be specified via natural-language prompts without modifying onboard weights. We evaluate zero-shot instance segmentation of spacecraft components under a strictly frozen, single-pass inference protocol on a test set of $129$ images of previously unseen satellites. Under fixed global thresholds and no post-processing, SAM3 achieves $0.385$ mAP@$0.5$ and $0.267$ mAP@$0.5{:}0.95$. Performance is strongly scale-dependent: large structural elements like spacecraft bodies ($0.639$ AP@$0.50$) and solar arrays ($0.598$ AP@$0.5$) localize reliably, while relatively small appendages like antennas ($0.221$ AP@$0.5$) and thrusters ($0.081$ AP@$0.5$) remain difficult. Prompt formulation influences performance, with structured prompts incorporating spatial and geometric descriptors yielding up to $82%$ improvement over short category-name prompts. The model operates within the memory and compute envelope of contemporary embedded GPUs, suggesting prompt-driven grounding can provide a practical mechanism for post-launch semantic extension of dominant spacecraft structures while highlighting limitations of zero-shot localization for fine-scale components under orbital domain shift. 

---
# Constitutional Value Potentials: reading and steering internal priority margins in language models 

**Authors**: Tong Che, Rui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15420)  

**Abstract**: A constitution tells a language model what to value, but little tells us whether it does. Adherence is judged from outputs, and output evidence is most fragile on value conflicts, where what matters is not which value a model mentions but which one it is willing to sacrifice. We provide evidence that this arbitration can be read from activations in a structured margin readout. We introduce Constitutional Value Potentials (CVP). For each value we learn a scalar potential from the hidden state: an internal pressure to preserve that value, supervised not by the prompt but by an independent judge's verdict on which value the model's own response actually preserved. The signed difference of two potentials is a priority margin. A constitutional clause becomes the claim that a margin stays positive, and a single monitor score flags when it does not. The monitor predicts conflict violations with AUROC up to 0.95, beats a strong hidden-state probe, and generalizes to held-out synthetic conflicts across three Qwen2.5 scales. The signal appears as the answer begins, from the prompt tail and first response token. Read this early, the same signal reveals whether an adversarial priority hack has actually pushed the model toward a violation, rather than only whether the prompt looks adversarial. The same directions also support intervention tests: under selected steering settings, moving along a value direction shifts judged trade-offs in the intended direction. Together, these results suggest that some constitution-relevant priorities are accessible as activation-space margins, rather than only as output behavior. 

---
# Let LLMs Judge Each Other: Multi-Agent Peer-Reviewed Reasoning for Medical Question Answering 

**Authors**: Zaifu Zhan, Shuang Zhou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15419)  

**Abstract**: Objective: To enhance the accuracy, interpretability, and robustness of large language models (LLMs) in medical question answering (MedQA).
Method: We designed a multi-agent peer-reviewed reasoning method in which multiple LLM agents independently generate chain-of-thought reasoning with candidate answers, then act as peer reviewers to evaluate each other's reasoning for factual correctness and logical soundness. The highest-rated reasoning chain is selected to produce the final answer. Experiments were conducted with five state-of-the-art LLMs (Llama-3.1-8B, Qwen2.5-7B, Phi-4, DeepSeek-LLM-7B, GPT-oss-20B) on three benchmark datasets: HeadQA, MedQA-USMLE, and PubMedQA. Performance was compared against single-model chain-of-thought reasoning and chain-of-thought-based majority voting.
Results: Peer-reviewed reasoning consistently outperformed both baselines. The best model combination achieved an average accuracy of 0.820 across datasets, exceeding the strongest single model (0.777) and majority voting ensembles (up to 0.789). The method also scaled effectively with more participating models, while peer assessments reliably distinguished high- from low-quality reasoning chains.
Conclusion: The proposed multi-agent peer-reviewed reasoning method enables LLMs to act as both solvers and evaluators, yielding superior performance in MedQA. By emphasizing reasoning quality rather than answer agreement alone, this approach improves accuracy, interpretability, and robustness, offering a promising direction for trustworthy biomedical AI systems. 

---
# Few-Shot Biomedical Relation Extraction with Large Language Models: A Viable Alternative to Supervised Learning? 

**Authors**: Jakob Mraz, Tomaž Curk, Blaž Zupan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15412)  

**Abstract**: Biomedical relation extraction (BioRE) is a key step in transforming biomedical literature into structured knowledge. However, most existing approaches rely on supervised models trained on costly annotated datasets, limiting their scalability and adaptability across relation types and domains. We investigate few-shot BioRE using prompt-based learning with large language models (LLMs) and compare two task formulations: pairwise classification, which predicts relations for individual entity pairs, and joint generation, which extracts multiple relations in a single model call. Experiments on the BioREDirect dataset reveal a clear precision-recall trade-off. Pairwise classification achieves higher recall, whereas joint generation is more precise and computationally efficient. The best-performing model achieves a micro-F1 score of 0.44, substantially outperforming previous few-shot results (0.34) while remaining below the supervised baseline (0.56). Much of this gap is attributable to a single ambiguously defined relation type. When evaluated using macro-F1, which better captures performance across relation types in an imbalanced setting, prompt-based approaches outperform the supervised baseline (0.45 vs. 0.38), particularly on rare relation types. These findings highlight the potential of LLMs for BioRE in low-resource settings and underscore the importance of well-defined relation schemas. 

---
# T-Mem: Memory That Anticipates, Not Archives 

**Authors**: Weidong Guo, Dakai Wang, Zixuan Wang, Hui Liu, Yu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15405)  

**Abstract**: Long-term memory is essential for conversational agents to remain coherent across extended dialogues, follow through on commitments made many sessions earlier, and adapt their behaviour to each user. Current LLM-backed long-term conversational memory, however, is reachability-bounded by the similarity between a query and stored content, both lexical and dense-vector. The approach is effective when query and memory share surface features such as wording or named entities (we call this descriptive). But it misses another, equally valuable class of cases, where query and memory do not share surface features and are tied only by a latent semantic arc (associative). On this regime prevailing long-term memory systems collectively fail. Covering this other half is what allows an assistant, for the first time, to actively draw on past dialogue as a semantic asset. On the memory side, this is the engineering counterpart of what cognitive science calls episodic future thinking: rehearsing past experience for the future contexts under which it will need to be found. We call these write-time rehearsals triggers. We propose T-Mem, the first long-term conversational memory architecture that covers both descriptive and associative recall. At each of two evidence granularities, single facts and full exchanges, T-Mem instantiates one descriptive trigger family and one associative trigger family, so that every memory remains reachable from both surface-similar and relevance-bound queries. As empirical validation, T-Mem reaches state-of-the-art on both LoCoMo and LoCoMo-Plus. 

---
# CHILLGuard: Towards Fine-Grained Chinese LLM Safety Guardrail with Scalable Data Construction and Model-aware Preference Alignment 

**Authors**: Wenbo Yu, Bohua Wang, Hao Fang, Kuofeng Gao, Jingru Zeng, Xiaochen Yang, Tianyi Zhang, Xiaoxiao Ma, Jiawei Kong, Hao Wu, Bin Chen, Shu-Tao Xia, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15396)  

**Abstract**: Malicious content generated from large language models (LLMs) could pose severe safety risks and ethical concerns. While existing LLM safety guardrails excel in English or multilingual settings, they lack adaptation to Chinese-specific regulatory policies, cultural context and linguistic nuances, failing to support fine-grained risk classification for diverse deployment needs. In this paper, we introduce a 5-macro, 31-micro category fine-grained risk taxonomy for Chinese scenarios, and build CHILLGuard: a dedicated Chinese LLM content safety guardrail. To address the critical scarcity of high-quality annotated Chinese safety data, we propose a scalable multi-stage data construction pipeline: we expand multi-source corpus via retrieval-augmented generation, generate implicit harmful samples through prompt engineering rewriting, and refine high-quality data via multi-model voting-based label calibration. Based on this, we build CHILLGuardTrain, a large-scale training set with 405,007 samples, and CHILLGuardTest, a rigorously curated annotated test set with 51,745 samples. We then train CHILLGuard on CHILLGuardTrain under a generator-classifier collaborative framework via Model-aware Direct Preference Optimization. Extensive experiments under multiple settings demonstrate the state-of-the-art performance of CHILLGuard, e.g., a 15.92% improvement of F1 score over Qwen3Guard-8B-Strict on our benchmark. We will release our resources at this https URL. 

---
# Not All Skills Help: Measuring and Repairing Agent Knowledge 

**Authors**: Yixuan Wang, Yiyang Zhou, Yiming Liang, Congyu Zhang, Fuxiao Liu, Jiawei Zhou, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15390)  

**Abstract**: LLM agents can improve without weight updates by accumulating natural-language skills from experience, but current systems entrust every decision about which skills to keep and how to apply them to LLM judgment alone. We argue that this conflates two distinct roles: generating a skill from experience is a creative act that judgment handles well, while deciding whether that skill actually helps requires empirical evidence across many tasks. Measuring per-skill causal contributions via randomized masking, we find that skill libraries exhibit pervasive causal heterogeneity: individual skills routinely help on some task types while hurting on others, yet their opposing effects cancel in aggregate, making them invisible to global curation methods. We propose ASSAY, a framework that separates generation from curation: it computes a per-skill causal attribution on a small development set, restructures the library offline, and suppresses skills with negative predicted effect for each test task. Across seven base models spanning four providers and two benchmarks (AppWorld and tau-bench), ASSAY consistently improves over prior skill-curation approaches. On AppWorld's hardest split, DeepSeek-V3 achieves 69.3% task-goal completion (47.4% relative improvement), a new state of the art among all published methods including weight-tuned approaches. On tau-bench retail, GPT-4.1 improves by 8.7% relative, advancing past o4-mini, o1, and GPT-4.5 on the public leaderboard without any weight modification. Ablation traces the dominant gain to per-task masking, confirming that the bottleneck is matching skills to tasks at inference time, not removing bad skills globally. Code is available at this https URL. 

---
# Learning Earthquake Wave Arrival Time Picking from Labels with Inaccuracies 

**Authors**: Sen Li, Xu Yang, S. Mostafa Mousavi, Anye Cao, Keting Fan, Yaoqi Liu, Changbin Wang, Qiang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15377)  

**Abstract**: Inaccurately labeled training data, or "label noise", poses a significant threat to the integrity of supervised machine learning models. This corruption directly degrades performance by teaching the model erroneous mappings between features and labels, which leads to poor generalization and reduced accuracy on properly labeled validation and test data. Current seismological applications mainly rely on large-scale training sets or data augmentation to reduce the label-noise impact, which can be labor-intensive and costly. Here, we introduce a Label Noise-Contrastive Robust Learning (LaNCoR) approach that can effectively handle noisy labels in seismic signal processing tasks, without requiring large-scale training datasets. In this approach, the input waveform feature and label representation distributions are aligned in the feature space to correct mislabeling and reduce its impact on the training process. We present LaNCoR's performance on the task of P-phase arrival-time picking of real microseismic data using two baseline models and training approaches. Our results indicate that LaNCoR can improve performance by up to 28.8% across performance metrics. This approach holds great promise for model training in seismology and geosciences. 

---
# CoAgent: Concurrency Control for Multi-Agent Systems 

**Authors**: Hongtao Lyu, Dingyan Zhang, Mingyu Wu, Xingda Wei, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.15376)  

**Abstract**: Multi-agent LLM systems -- coding agents, devops agents, document agents -- now routinely run several agents in parallel against the same git tree, Kubernetes cluster, or document. As soon as two of them mutate shared state, they enter the regime classical concurrency control has studied for decades, but classical mechanisms fit LLM agents poorly. A single agent transaction spans minutes of inference, read sets are broad and opaque rather than statically inferable, and the live state agents act on admits neither fork nor buffer, so writes take effect the moment they execute. Locks block long inference intervals; OCC abort-and-retry discards minutes of work on every conflict.
This paper builds concurrency control on a capability classical transactions lack: the LLM inside each agent can judge whether a conflicting write invalidates its plan, and can repair exactly the operations that depended on it. Control therefore turns advisory: the runtime informs, the agent repairs. Our protocol, MTPO (Monotonic Trajectory Pre-Order), fixes a serialization order at launch, serves each read the order-filtered value, and applies writes speculatively in place; a one-way notification asks an affected reader to re-judge and patch its plan, while the framework mechanically undoes and reorders misplaced writes through the saga-style inverse each tool registers in advance. At quiescence the run is serializable in the pre-decided order. We realize MTPO as CoAgent, toolcall middleware whose privileged ToolSmith grows footprint-declared, undoable tools online. On ten contended workloads, CoAgent stays within 5\% of serial correctness at a $1.4\times$ speedup and near-serial token cost, where 2PL and OCC surrender nearly all concurrency gains; on a bash-only target system, it grows a 25-tool library online and lifts the task pass rate from 45/71 to 63/71 at $0.80\times$ the time and $0.86\times$ the cost. 

---
# Cognitive Trajectory Modeling: Quantifying Human-AI Co-Creation through Cognitively Grounded Interaction Trajectories 

**Authors**: Nicholas Davis  

**Link**: [PDF](https://arxiv.org/pdf/2606.15358)  

**Abstract**: Co-creative AI research increasingly seeks methods capable of representing how interaction dynamics evolve through time. While many existing approaches focus on observable interaction characteristics, interaction metrics, behavioral coding schemes, or activity traces, these methods often struggle to capture higher-order interaction dynamics, including how collaborative processes reorganize, stabilize, regulate, and evolve through time. This paper introduces Cognitive Trajectory Modeling (CTM) as a cognitive theory of interaction dynamics that conceptualizes cognition, interaction, and creative processes as temporally organized trajectories unfolding across cognitively meaningful attractor landscapes. CTM builds upon the theoretical foundations of the Enactive Model of Creativity and Creative Sense-Making (CSM), revisiting the role of sense-making curves and cognitive trajectories in representing co-creative interaction dynamics. We formalize this perspective through the Cognitive Trajectory Principle, which states that temporal representations are only theoretically interpretable as cognitive trajectories when their underlying states possess directional cognitive meaning. Building on this principle, CTM generalizes the notion of cognitive trajectories beyond any particular coding scheme and provides a broader framework for modeling interaction dynamics through trajectories unfolding across meaningful attractor landscapes. We further distinguish cognitive trajectories from interaction traces and situate CTM within a broader hierarchy of cognitive, interaction, and domain dynamics. More broadly, we argue that understanding co-creative systems requires methods capable of modeling how cognition and interaction dynamics unfold through time. CTM provides a foundation for studying interaction dynamics across co-creative AI and human-AI interaction. 

---
# LearnOpt: Recovering the Latent Cognitive Structure of Standardized Examinations via Knowledge Graphs and Constrained Optimization 

**Authors**: Joy Bose, Om Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2606.15349)  

**Abstract**: Standardized examinations are typically treated as uniform syllabus coverage problems. We argue they are better understood as adversarial systems with stable latent cognitive structures diverging systematically from official syllabi. We introduce LearnOpt, which recovers this structure from historical question papers and generates personalized, time-bounded study plans. Applied to nine years of NEET questions (2016-2024, n=1,496), LearnOpt builds an exam knowledge graph from LLM-tagged questions, extracts a five-category latent skill distribution, and formulates study planning as a knapsack-variant optimization over prerequisite-aware subgraphs with Bayesian Knowledge Tracing. Central finding: NEET's latent skill distribution is stable within a syllabus regime (consecutive-year KL divergence 0.004-0.032 for 2016-2021, non-significant under permutation testing) but shifts significantly with NCERT's 2023 syllabus rationalization: pooling 2016-2021 (n=1,072) vs 2023-2024 (n=392) gives KL=0.040 (p=0.0005), with Elimination/Negation questions rising from ~20-29% to ~31-35%. Latent structure, while not permanently stationary, is piecewise stable, with shifts detectable and attributable to curricular events. Within either regime, subject predicts skill profile more strongly than year. An optimization evaluation, using one real and two synthetic mastery profiles, shows the skill-weighted objective produces a modest but real reordering of recommended topics over a mastery-conditioned frequency baseline. Applying the pipeline to JEE Advanced reveals a profile dominated by Multi-concept Integration (80.9% vs. 33.3% for NEET), with a JEE-vs-NEET divergence (KL=0.505) exceeding NEET's largest cross-subject divergence: exam tier shapes latent cognitive structure more than subject, which shapes it more than time within a regime. Code, knowledge graph, and annotated dataset are released publicly. 

---
# Intrinsic Computational Functionalism and Simulated Consciousness 

**Authors**: Ryota Kanai, Shuqin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.15348)  

**Abstract**: A common objection to artificial or simulated consciousness is that a simulated brain is no more conscious than simulated water is wet. We address this from the perspective of Intrinsic Computational Functionalism (ICF): if consciousness is computationally constituted, it depends not on externally imposed descriptions but on the computational structures a system physically realizes in virtue of its own causal-dynamical organization. In previous work we developed Canonical Functionalism as a mathematically precise special case of this anti-interpretivist program, identifying functional states by their complete future input-output roles under a fixed interface. Here we argue that this input-output construction, though important, is incomplete: as a behavioral boundary case of ICF, it makes lookup tables and unfolded systems that preserve the same boundary behavior canonically equivalent. A consciousness-relevant canonical representation must instead include internal mechanisms, interventions, and joint readouts belonging to the relevant intrinsic organization. We therefore define a mechanism-enriched canonical structure and use it to formulate Intrinsic Causal-Computational Realization (ICCR), a realization relation preserving physical implementation, intrinsic state individuation, transition structure, intervention profiles, and the relevant agent-body-world boundary. The central result is conditional: if conscious properties are invariants of intrinsic causal-computational organization, then any system satisfying ICCR realizes the same consciousness-relevant properties, whether biological, artificial, or simulated. We discuss objections including biological naturalism and integrated information theory. We conclude that to deny consciousness to a simulation, one must identify a consciousness-relevant intrinsic causal-computational structure that the simulation fails to realize. 

---
# Privacy-Preserving Text Sanitization for Distributed Agents Collaboration via Disentangled Representations 

**Authors**: Xuan Liu, Hefeng Zhou, Sicheng Chen, Chao Yang, Xingcheng Xu, Jingjing Qu, Jiong Lou, Jie LI, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15335)  

**Abstract**: When distributed agents exchange text across organizational boundaries, privacy leakage arises not only from explicit identifiers but also from distributional signatures such as formatting conventions, vocabulary choices, and syntactic patterns. We propose DiSan(Disentangled Sanitization), a privacy-preserving sanitization framework and a built-in component of Intern-Shannon for multi-agent collaboration. DiSan uses a two-stream encoder to factorize text into a source-invariant role subspace that preserves task semantics and a source-identifying style subspace that remains local. Federated proto-type alignment and adversarial regularization enable joint training without centralizing raw text. Experiments show that identifier-level masking is insufficient: masking 19.2% of tokens reduces TF-IDF stylometric attribution by only 18.6%. By contrast, DiSan reduces answer-level PII exposure by 20 times while maintaining 83% answer faithfulness on a distributed multi-agent RAG benchmark, and lowers Enron stylometric attribution by 73.2% under TF-IDF and 70.6% under a neural probe. 

---
# HoloRec: Holistic Encoding and Interleaved Reasoning for Generative Recommendation 

**Authors**: Shuqi Zhao, Jingsong Su, Xiang Liu, Xingzhi Yao, Yiming Qiu, Huimu Wang, Liang Lin, Pengbo Mo, Mingming Li, Jiao Dai, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15331)  

**Abstract**: Generative recommendation models that formulate the task as sequence generation overcome the objective fragmentation problem of traditional cascade architectures, yet existing approaches still suffer from flat semantic representations lacking hierarchical structure for multi-step reasoning and an externally constructed chain-of-thought (CoT) that requires expensive annotations and remains disconnected from the generation objective. We propose HoloRec, an endogenous chain-of-thought recommendation mechanism that unifies representation, reasoning, and generation by constructing a hierarchical semantic encoding matrix via multi-granularity nested residual quantization optimized by a holistic reconstruction loss. HoloRec supports two inference modes: a non-thinking mode that uses lightweight multi-granularity supervised alignment for fast prediction, and a thinking mode that employs an interleaved reasoning scheme to generate CoT steps on the fly, directly embedding reasoning into the generation process without external data. Experiments on multiple public recommendation datasets demonstrate that HoloRec consistently outperforms baselines, with especially significant gains in sparse scenarios, and the thinking mode achieves better accuracy than the non-thinking mode with only modest inference overhead. 

---
# LLMs on Tabular Data with Limited Semantics: Evidence from Industrial Car Retrofit Prediction 

**Authors**: Aina Vila Pons, Ioannis Tzachristas, Constantinos Antoniou  

**Link**: [PDF](https://arxiv.org/pdf/2606.15314)  

**Abstract**: Industrial retrofit planning depends on structured operational data rather than free text: planners must estimate whether a newly registered prototype will require a retrofit, which retrofit package it will need, and how long the work will take. We study an industrial dataset linking a prototype-registration system (284,271 vehicles) with a retrofit-management system (48,716 cleaned visits), and compare strong tabular machine learning baselines with three LLM-based strategies on row-serialized inputs: embedding features (Amazon Titan), direct prompted classification (Claude Sonnet 4), and an ML+LLM stacking approach. Across binary occurrence prediction, 15-way retrofit-type classification, per-visit duration regression, and an aggregated monthly benchmark, classical tree ensembles remain the strongest standalone models. However, the LLM results reveal a consistent pattern: embeddings remain useful on tables (binary AUC = 0.982), direct prompting collapses once semantic signal is stripped by hashing (binary AUC = 0.500; multiclass weighted F1 = 0.018), and hybrid stacking yields the best manually built multiclass model (weighted F1 = 0.626). On the monthly benchmark, lag-based machine learning outperforms time-series foundation models, though Chronos-small remains competitive in zero-shot forecasting. The results suggest that on privacy-constrained industrial tables, LLMs are more effective as complementary components than as replacements for strong tabular baselines. 

---
# Adapting Reinforcement Learning with Chain-of-Thought Supervision for Explainable Detection of Hateful and Propagandistic Memes 

**Authors**: Mohamed Bayan Kmainasi, Mucahid Kutlu, Ali Ezzat Shahroor, Abul Hasnat, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2606.15307)  

**Abstract**: Hateful and propagandistic memes exploit the interplay between images and text to convey harmful intent that neither modality reveals alone. Although thinking-based multimodal large language models (MLLMs) have advanced vision-language understanding, their application to meme content moderation remains underexplored. We propose a reinforcement learning-based post-training method that improves classification performance and reference-based explanation quality in thinking-based MLLMs via task-specific rewards and Group Relative Policy Optimization (GRPO). Concretely, we (i) conduct a systematic empirical study of off-the-shelf MLLMs for hateful and propagandistic meme understanding across English and Arabic benchmarks, (ii) extend existing meme datasets with weakly supervised chain-of-thought (CoT) rationales via distillation and multi-LLM fine-grained propaganda annotations, (iii) introduce a GRPO-based objective with thinking-length regularization that jointly optimizes classification accuracy and explanation quality, and (iv) investigate self-supervised GRPO on unlabeled memes using consensus-based pseudo-labels. Experiments on the Hateful Memes and ArMeme benchmarks show that our approach improves over previously reported results on FHM accuracy (up to +2.1%, from 79.9% to 82.0%) and on ArMeme macro-F1 (up to +7.6 points, from 0.536 to 0.612 with explanations; +6.1 compared to the original ArMeme benchmark), while also generating natural-language explanations. On ArMeme, sequence-classification baselines remain stronger in terms of raw accuracy, whereas our approach provides more balanced per-class performance along with explanations. We publicly release our code, data extensions, and evaluation resources. 

---
# LatentGym: A Testbed For Cross-Task Experiential Learning With Controllable Latent Structure 

**Authors**: Daksh Mittal, Tommaso Castellani, Thomson Yen, Naimeng Ye, Fangyu Wu, Minghui Chen, Tiffany Cai, Emmanouil Koukoumidis, William Zeng, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2606.15306)  

**Abstract**: We envision continually learning agentic systems that become more useful over time: as they encounter sequences of related tasks, they should infer the hidden structure shared across those tasks and use it to improve future decisions. This cross-task experiential learning capability is pivotal in domains such as personalization and interactive assistance, but existing training/evaluation frameworks do not provide shared, controllable latent structures and cannot measure whether or why agents improve. We introduce LatentGym: a controllable suite in which each environment is organized around a ground-truth latent variable governing the structure across tasks. Our construction yields metrics that separate exploration (whether the agent's actions gather information about the latent) from exploitation (whether the agent uses what it has gathered). We demonstrate our suite on empirical studies addressing three questions: how and why frontier models fail to adapt across related tasks; whether post-training on related task sequences improves general cross-task adaptation, and where those gains come from; and how design choices such as inter-task feedback shape training dynamics and generalization. Together, these results establish a controlled foundation for studying how LLM agents learn from experience across tasks, and for designing agents that adapt more reliably in sequential, personalized, and interactive settings. 

---
# Discovering Lattice Reduction Strategies via Self-Play 

**Authors**: Mohamed Malhou, Kristin Lauter, Ludovic Perret  

**Link**: [PDF](https://arxiv.org/pdf/2606.15301)  

**Abstract**: The Lenstra-Lenstra-Lovász (LLL) algorithm is a seminal contribution to computer science used for lattice basis reduction, yet its polynomial-time outputs produce bases that are far from optimal as the dimension grows. We show that deep reinforcement learning can discover strictly superior, generalizable reduction strategies by interacting with the primitive action space of LLL. We formulate lattice reduction as a single-player Markov Decision Process (MDP) and train a deep residual network using an AlphaZero-style self-play pipeline augmented with adaptive-horizon MCTS (Monte Carlo Tree Search), which couples multi-step network predictions with an entropy-gated expansion mechanism. The resulting policy, DeltaStar, is trained exclusively on small $8$-dimensional $q$-ary lattices and requires fewer primitive row operations than LLL. Crucially, it generalizes zero-shot to unseen moduli and higher dimensions up to $n=32$ without retraining. 

---
# Hybrid NARX-LLM for Greenland Iceberg Discharge: Prompt-Driven Residual Correction 

**Authors**: Yiquan Gao, Duohui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15288)  

**Abstract**: Greenland iceberg discharge exhibits complex nonlinear dynamics with limited observability, challenging traditional predictive models. We present a Hybrid NARX-LLM framework that combines a nonlinear autoregressive model with exogenous inputs (NARX) and a large language model (LLM) for residual correction. We further propose a Physics-Informed Prompt (PIP) method that transforms unstructured physical knowledge into structured prompts for zero-shot in-context reasoning. The primary objective is to explore the corrective potential of this framework for modeling Greenland iceberg discharge, rather than merely optimizing predictive accuracy. The NARX component captures intrinsic temporal dependencies, while the LLM, guided by PIP, encodes glacier dynamics and environmental drivers and perceives key trend patterns to correct systematic prediction errors. This integration allows the model to reason about unmodeled factors and produce interpretable residuals, enhancing overall predictive accuracy. Applied to Greenland iceberg discharge time series, our approach addresses extreme events that are difficult to predict due to rare variations and nonstationary trends, a limitation often overlooked by traditional methods. By fusing structured time-series modeling with knowledge-driven foundation AI, the framework offers a scalable and interpretable pathway to bridge data-limited climate forecasting with physics-informed LLM reasoning. The code is available. 

---
# CAP: Towards PPG Universal Representation Learning with Patient-level Supervision 

**Authors**: Chenyang He, Xinyi Shao, Shun Huang, Bosong Huang, Daoqiang Zhang, Ming Jing, Cheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.15284)  

**Abstract**: Photoplethysmography (PPG) plays a central role in wearable health monitoring and clinical decision support. Yet existing approaches to universal PPG representation learning largely focus on signal-level objectives and often overlook patient-level health context, which limits generalization to complex clinical tasks and heterogeneous cohorts. To address this gap, we construct a large-scale paired PPG-EHR multimodal dataset by distilling fragmented medical histories and clinical records into cohesive, patient-level electronic health records (EHR). Building on this resource, we propose Clinical Anchored Pretraining for PPG (CAP). During pretraining, CAP performs cross-modal contrastive alignment that anchors PPG representations to patient-level clinical semantics, guiding the encoder beyond waveform fitting toward modeling consistency in a patient's overall physiological state. During downstream adaptation, the pretrained PPG encoder provides clinically grounded representations that strengthen inductive bias and improve robustness and transferability. Experiments demonstrate that CAP consistently outperforms strong baselines on four diverse downstream tasks. CAP achieves a particularly large gain on respiratory rate prediction (up to +87.6% relative improvement over the state-of-the-art baseline) and delivers an average relative +26.7% across all tasks. We further enhance the interpretability of our approach through comprehensive analyses, including ablations and multiple complementary visualizations of the learned representations. The code for our experiments is available at: this https URL . 

---
# RECTOR: Masked Region-Channel-Temporal Modeling for Affective and Cognitive Representation Learning 

**Authors**: Jinhan Liu, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.15278)  

**Abstract**: Affective and cognitive disorders manifest as distributed, time-varying brain network dynamics across regions, channels, and time, challenging robust representation learning from EEG/sEEG for clinical diagnosis. We propose RECTOR (Masked Region-Channel-Temporal Modeling), an end-to-end self-supervised framework that unifies joint region-channel-temporal representation learning beyond fixed anatomical priors. At its core, RECTOR-SA is a hierarchical, block-sparse self-attention induced by Adaptive Functional Partitioning that evolves region structures from static anatomical definitions to adaptive functional regions. The self-supervision is driven by Masked Topology and Representation Learning, which jointly optimizes three complementary objectives: Masked Predictive Modeling, Topological Structure Modeling, and Cross-View Consistency. Across diverse benchmarks, RECTOR sets a new state-of-the-art in EEG emotion recognition and sEEG task-engagement classification. Crucially, its strong robustness to missing channels and cross-montage generalization underscores its potential for large-scale pre-training on heterogeneous EEG/sEEG, providing interpretable insights at both region and channel levels. 

---
# Guiding Federated Graph Recommendation with LLM-encoded knowledge 

**Authors**: Thi Minh Chau Nguyen, Hien Trang Nguyen, Duc Anh Nguyen, Van Ho-Long, Thanh Trung Huynh, Zhao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.15277)  

**Abstract**: Graph-based recommender systems are highly effective at extracting collaborative signals from user--item interactions, and federated learning (FL) allows these models to be trained while preserving user privacy. However, aggregating graph representations across distributed, non-IID clients remains a challenge; structural embeddings learned locally often misalign, and naive averaging fails to capture meaningful cross-client relationships. Most existing federated graph methods rely exclusively on structural aggregation, neglecting the rich, global semantic context available in large language models (LLMs). In this paper, we propose a novel framework that uses LLM-encoded knowledge to guide federated graph recommendation. Specifically, clients learn structural representations from local graphs while simultaneously summarizing their typical interaction patterns into compact semantic vectors via a frozen LLM. The central server then uses these LLM-encoded semantic signals to discover related preference patterns across clients, guiding the selective aggregation of their structural representations. This enables semantically informed cross-client collaboration without exposing raw data. Extensive experiments on standard benchmarks show that guiding structural alignment with LLM-encoded knowledge consistently improves recommendation accuracy over existing federated graph baselines. 

---
# Trust-Region Diffusion Policies for Massively Parallel On-Policy RL 

**Authors**: Huy Le, Onur Celik, Denis Blessing, Tai Hoang, Claas A Voelcker, Axel Brunnbauer, Felix Richter, Michael Volpp, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2606.15260)  

**Abstract**: Reinforcement learning with massively parallel simulations has become a standard framework for developing robust, deployable policies; however, most existing approaches still rely on simple Gaussian policy parameterizations. Diffusion models provide a more expressive policy class and have shown strong performance on challenging control problems, yet most diffusion-based RL methods are designed for offline or off-policy training. In this work, we ask whether diffusion policies can be trained effectively in the massively parallel, on-policy regime. To this end, we introduce Trust-region Diffusion Policies (TruDi), which enables diffusion policies for on-policy RL with massively parallel simulations. This setting is particularly challenging because the data distribution changes quickly across updates, making stable training with complex policies difficult. TruDi addresses this by integrating a trust-region optimization rule to enforce a KL-divergence constraint over the entire diffusion trajectory. Empirically, we evaluate TruDi on a diverse set of 4 massively parallel RL benchmarks comprising a total of 73 tasks. Across these tasks, TruDi consistently outperforms or is on-par with strong baselines on standard tasks and achieves clear gains on more challenging humanoid control tasks, establishing a strong new baseline for massively parallel on-policy RL. 

---
# Driving, Fast or Slow? Neuro-Symbolic Guidance for Motion Prediction in Multi-Modal Ground Mobility 

**Authors**: Simon Kohaut, Felix Divo, Julius Hahnewald, Benedict Flade, Julian Eggert, Kristian Kersting, Devendra Singh Dhami  

**Link**: [PDF](https://arxiv.org/pdf/2606.15251)  

**Abstract**: Accurate and interpretable motion prediction for heterogeneous traffic spaces, including pedestrians, bicycles, cars, and trucks, is essential for safe autonomous navigation. Nevertheless, state-of-the-art approaches remain predominantly black-box, lacking explicit encoding of the regulatory and behavioral constraints of real-world mobility. We propose Trajectory Compliance-Shaping (TraCS), a neuro-symbolic framework that augments existing black-box motion prediction backbones with interpretable and probabilistic first-order logic. To do so, TraCS employs an agentic code-generation pipeline to bridge the gap between natural-language descriptions of traffic regulations and probabilistic motion prediction. Furthermore, TraCS employs a reactive data-streaming inference engine that maintains and efficiently updates compliance landscapes as scenes evolve. To prevent TraCS from overconfidently steering the backbone's predictions in the wrong direction, we propose a neural confidence rating learned as a context-aware attenuation of the compliance signal. We demonstrate on the Argoverse 2 benchmark how TraCS consistently improves state-of-the-art prediction backbones, showing that probabilistic and symbolic compliance reasoning is a broadly applicable and computationally efficient complement to purely neural motion predictors. 

---
# Landmark-free Assessment of Lower-limb Alignment with Implicit Neural Shape Functions from Knee Radiographs 

**Authors**: Zhisen Hu, Antti Kemppainen, David Johnson, Egor Panfilov, Huy Hoang Nguyen, Timothy Cootes, Claudia Lindner, Aleksei Tiulpin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15250)  

**Abstract**: Radiographic assessment of lower-limb alignment (LLA) is important for predicting joint health and surgical outcomes in total knee arthroplasty. Traditional measurement methods are manual and time-consuming, while recent machine learning approaches typically rely on locating a fixed set of anatomical landmarks. This dependence limits flexibility and may require re-annotation when clinical definitions change. To address this, we propose an automated workflow using Implicit Neural Shape Functions (INSF). Rather than relying on explicit landmark coordinates, we encode the anatomy into a compact latent space and regress clinical alignment measurements directly from these latent codes. This architecture allows for rapid extendability to new tasks without altering the backbone representation. We trained our method on an internal dataset of 566 knee radiographs, each annotated with the outline of the femur and tibia. We evaluated it on both an internal test dataset of 50 patients and a separate external set of 402 preoperative cases from the MRKR dataset. Manual clinical measurements are available for these data, and the MRKR measurements will be made publicly accessible. Performance was comparable to state-of-the-art landmark-based methods and manual agreement, while offering a flexible shape representation that can be extended to additional measurement tasks. 

---
# Exploring Starts Are Not Enough: Counterexamples and a Fix for Monte Carlo Exploring Starts 

**Authors**: Octave Oliviers, Glenn Vinnicombe  

**Link**: [PDF](https://arxiv.org/pdf/2606.15247)  

**Abstract**: The asymptotic behaviour of Monte Carlo Exploring Starts (MCES) is a long-standing open question in reinforcement learning, even in the tabular setting. We investigated the convergence properties of tabular MCES by constructing examples in which the algorithm converges to suboptimal solutions. This paper presents new counterexamples for both initial-visit and first-visit MCES and gives a convergence-restoring modification for the initial-visit case. We show that stable suboptimal solutions may exist for initial-visit MCES with sample-average updates even when greedy actions are updated more often than non-greedy actions on average. However, by scaling learning rates inversely to update frequencies on a state-by-state basis, convergence to optimality is guaranteed. Unlike previous uniformisation methods, this modification is applicable to large-scale problems that require approximating the estimated value function. We then extend the example to show that sample-average first-visit MCES may also converge to suboptimal solutions. This largely settles a fundamental open problem and shows that exploring starts alone do not guarantee convergence to optimality. More broadly, these results highlight that convergence depends critically on the relative size and frequency of updates applied to different actions, making the choice of learning rates and the balance between exploration and exploitation central to the analysis of MCES and the implementation of scalable Monte Carlo control methods. 

---
# Provenance-Enhanced Statements in Knowledge Graphs 

**Authors**: Fabio Vitali, Valentina Pasqual  

**Link**: [PDF](https://arxiv.org/pdf/2606.15246)  

**Abstract**: Provenance-enhanced statements of the form "according to $X$, $\varphi$" are pervasive in contemporary knowledge graphs, especially in domains where graph content primarily represents claims, interpretations, and hypotheses (\emph{capta}) rather than observer-independent facts (\emph{data}). Current provenance models can record who asserted what, but they typically treat provenance as semantically neutral, leaving underspecified how attributed claims relate to factual commitment, to one another, and to reasoning.
In this paper we introduce DEC, a framework that interprets provenance predicates as indicators of epistemic stance and groups provenance-homogeneous sets of statements into \emph{cognitive worlds}. Drawing on cognitive modal logics (doxastic, epistemic, and conjectural), DEC characterizes locality, rationality, and controlled permeation between cognitive worlds and a distinguished factual core ("reality"), thereby enabling principled reasoning over attributed content without collapsing disagreements into inconsistencies.
We formalize a DEC interpretation for RDF datasets that is conservative over RDF~1.2 semantics, clarify the role of intensionality and identity (including the Superman paradox), and illustrate the approach on common Semantic Web representations (named graphs, quoted triples/RDF-star, and reification). Finally, we describe our prototype DEC reasoner implemented as a Fuseki dataset module, supporting controlled factualisation and explicit detection of disagreements and delusions. 

---
# Benign in Isolation, Harmful in Composition: Security Risks in Agent Skill Ecosystems 

**Authors**: Yi Xie, Jiawei Du, Yu Cheng, Jiuan Zhou, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15242)  

**Abstract**: Skills are becoming the capability layer through which LLM agents turn plans into actions, but their use introduces security risks such as data leakage, unauthorized operations, and tool misuse. Existing vetting usually evaluates each skill in isolation, while real agent tasks often invoke multiple skills in a shared execution context. This creates Skill Composition Risk (SCR): a skill that appears benign alone can become harmful when its outputs, trust signals, authorization cues, or side effects influence later invocations along an activated path. We introduce SCR-Bench to evaluate this risk in controlled, sandboxed skill environments. Rather than relying only on textual intent or surface behavior, SCR-Bench records downstream state changes and path-level outcomes across composed skill executions. It contains three sub-benchmarks: SCR-CapFlow for capability-flow composition, SCR-TrustLift for trust-transfer composition, and SCR-AuthBlur for authorization-confusion composition. Across SCR-Bench, composed paths expose risks that are largely absent under isolated evaluation. In SCR-CapFlow, attack success rate reaches 33.6 percent under composition, compared with near-zero isolated baselines. In SCR-TrustLift, attack success rate exceeds 96.5 percent on four of five backends. In SCR-AuthBlur, the risky-approval rate increases by 71.8 percent relative to the L0 isolated baseline under the L1 context setting. These results show that agent skill security should be assessed at the level of activated paths rather than isolated artifacts. SCR and SCR-Bench provide a foundation for path-aware risk evaluation and defense in LLM agent skill ecosystems. Benchmark: this https URL. 

---
# Edu-Theater: A Data-Efficient Agent Framework for Scalable Learner Behavior Simulation through Staging Roll-Call 

**Authors**: Weibo Gao, Qi Liu, Linan Yue, Zheng Zhang, Yichao Du, Fangzhou Yao, Ao Yu, Zhenya Huang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15225)  

**Abstract**: Large-scale learner-task interaction data are crucial for intelligent educational systems but are costly to collect and constrained by privacy and learner engagement. Learner simulators play a critical role in simulating scalable learner behavior without the need for continuous involvement of real learners. However, existing methods are predominantly \textbf{individual-centric}, pairing a simulator with each learner to iteratively infer latent knowledge states from dense interaction histories, which is both data- and computation-intensive, and fragile in cold-start scenarios. We propose a \textbf{cohort-aware roll-call simulation paradigm} that first constructs cohort-level proficiency priors and refines individual learner states through a small number of targeted diagnostic queries. Based on this paradigm, we introduce \textbf{Edu-Theater}, an LLM-powered agent system that performs cohort-aware learner simulation via a teacher agent and retrospective roll-call probing over learner logs. Edu-Theater enables scalable future behavior simulation without the need for dense per-learner histories. Experiments on two real-world datasets demonstrate that Edu-Theater achieves higher simulation accuracy with significantly fewer LLM calls, producing synthetic data that enhances downstream applications such as adaptive testing. 

---
# Spokes: Optimizing for Diverse Pretraining Data Selection 

**Authors**: Clarence Lee, Yejin Choi, Luke Zettlemoyer, Pang Wei Koh, Hai Leong Chieu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15216)  

**Abstract**: Diversity plays a critical role in data selection, improving performance under fixed data budgets by reducing redundancy and repetition. However, optimizing for diversity is inherently challenging, as it is a set-level property that depends on interactions between data points rather than individual examples. As a result, existing approaches typically rely on proxies or approximations, which often fail to ensure sufficiently diverse subsets. In this work, we directly optimize diversity by introducing a probabilistic diversification framework based on the G-Vendi score, optimized via exponentiated gradient descent. Our method produces subsets that are substantially more diverse than those obtained via random sampling, achieving a +489 increase in G-Vendi score on a 500k-sample subset. We evaluate our approach on FineWeb and DCLM, where it consistently outperforms existing methods. Notably, SPOKES (diversity-only) improves average downstream performance by +0.4 and +0.5 points over random sampling on DCLM and FineWeb, respectively. More importantly, jointly optimizing for both quality and diversity yields the strongest results: SPOKES achieves gains of +1.5 and +1.4 points on DCLM and FineWeb, outperforming all baselines, including semantic deduplication and quality filtering. 

---
# Controlled Dynamics Attractor Transformer 

**Authors**: Cheng Zhang, Minnan Luo, Zesheng Yang, Ming Li, Yong-Jin Liu, Qinghua Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.15207)  

**Abstract**: Transformer architectures have dramatically advanced representation learning and inference in deep models through self-attention mechanisms. In parallel,associative memory (AM) frameworks map representations onto energy landscapes, offering interpretable retrieval mechanisms. However, their continuous-time inference dynamics lack the biological plausibility of classical Continuous Attractor Neural Networks (CANNs). To bridge this gap, we propose Controlled Dynamics Attractor Transformer (CDAT), which couples a mixture von Mises-Fisher (Mo-vMF) attention energy with a Hopfield refinement energy, while augmenting energy descent with a CANN-inspired excitation-inhibition modulation. CDAT instantiates a topology-constrained dynamical system whose couplings encode relational structure among tokens, thereby linking attractor-style dynamics to modern energy-based attention. We further provide a constructive dissipation analysis to formally establish their controlled inference dynamics. Benefiting from these robust and structured dynamics, CDAT achieves state-of-the-art performance across multiple benchmarks in graph anomaly detection and graph classification. 

---
# AI Contagion in Social Networks 

**Authors**: Olivier Bos, Stefano Bosi  

**Link**: [PDF](https://arxiv.org/pdf/2606.15206)  

**Abstract**: We study how artificial intelligence (AI) interacts with social communication networks to shape the stability of collective knowledge. Agents exchange information through a network while receiving AI-generated content, and AI systems retrain on the aggregate social information they influence. This interaction generates two feedback forces: an AI contagion channel, through which distortions diffuse across the network, and an AI social distortion multiplier, through which retraining amplifies past errors. Despite the high dimensionality of the environment, we show that the long-run behavior of the system admits a two-dimensional representation whose spectral radius determines whether AI-mediated information systems are dynamically stable or unstable. We characterize a sharp regulatory frontier identifying the minimum filtering required for stability and show how network topology shapes systemic informational risk. 

---
# StarOR: Synergizing Tree Search and Test-Time Reinforcement Learning for Optimization Modeling 

**Authors**: Jiajun Li, Yu Ding, Shisi Guan, Ran Hou, Wanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15197)  

**Abstract**: Optimization modeling is inherently hierarchical, requiring a precise sequence of symbolic commitments. Traditional learning-based automated optimization modeling methods improve modeling policies through large-scale annotated or curated training data, but are costly to adapt to new problem distributions. Meanwhile, one-shot generation remains brittle in hierarchical modeling, where early symbolic errors can propagate into invalid formulations. Test-time scaling offers a promising alternative by enabling structural exploration with additional instance-level computation; however, existing search-based methods typically rely on a fixed policy, causing repeated rollouts to inherit similar modeling biases and providing limited credit assignment for intermediate decisions. To address these limitations, we propose StarOR, a synergistic search-and-adaptation framework that couples MCTS with Test-Time Reinforcement Learning for optimization modeling. StarOR decomposes the modeling process into four stages and updates a transient LoRA adapter via GRPO at each non-terminal node. By using MCTS-generated siblings as local comparison sets, StarOR transforms search-time exploration into instance-specific policy refinement. Moreover, an unsupervised multi-faceted reward system provides fine-grained feedback for intermediate formulation decisions without ground-truth labels. Experiments across five optimization benchmarks show that StarOR achieves state-of-the-art performance even with a 4B backbone, outperforming existing methods and the frontier LLMs. 

---
# FreeSonic: Training-Free Temporal-Aware Decoupled Attention for Precise Audio Editing 

**Authors**: Yuxuan Jiang, Mingyang Han, Yusheng Dai, Andong Wang, Tianhong Zhou, Jiaxin Ye, Dongxiao Wang, Haoxiang Shi, Boyu Li, Jun Song, Cheng Yu, Bo Zheng, Weibei Dou, Zehua Chen, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15186)  

**Abstract**: Text-to-audio (TTA) generation has made significant strides, yet achieving precise and consistent audio editing remains a major challenge. However, existing methods struggle to balance temporal consistency with background preservation. In this paper, we propose FreeSonic, a training-free framework leveraging the state-of-the-art Rectified Flow-based TangoFlux model. FreeSonic utilizes an optimized inversion-reverse process and joint text-audio attention maps for precise target segment extraction. For content editing, a novel scheduled attention decoupling confines modifications to target regions while preserving original acoustic context. Furthermore, task-oriented noise injection enhances versatility for tasks such as audio removal and non-rigid replacement. Extensive experimental results demonstrate that FreeSonic achieves a superior balance by providing a high-fidelity and efficient solution for precise and consistent audio editing. Project and demos: this https URL 

---
# Enabling Real-Time Point-of-Care Ultrasound Segmentation: A GPU-Free Deployment in Resource-Limited Settings 

**Authors**: Weihao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.15176)  

**Abstract**: Ultrasound imaging is the most widely adopted medical modality globally due to its low cost and portability, yet artificial intelligence (AI) deployment remains constrained by reliance on GPU-accelerated models, creating a structural paradox where the cost of "intelligence" exceeds that of the imaging device itself. Here, we present the systematic adaptation and extensive evaluation of UltraSeg, an ultra-lightweight architecture originally developed for colonoscopic polyp segmentation, now engineered for point-of-care ultrasound (POCUS) across ten public datasets spanning six anatomical sites (breast, thyroid, kidney, carotid, fetal, and small-animal tumor). We systematically validate both variants in ultrasound domains: UltraSeg-130K (0.13M parameters) achieves 89.7 FPS on single-core CPUs and 34.8 FPS on a refurbished mobile device, while UltraSeg-500K (0.5M parameters) delivers 44.6 FPS on CPU and 16.1 FPS on mobile device. UltraSeg-500K matches or exceeds the Dice performance of the 31M-parameter UNet and approaches 105M-parameter TransUNet in average performance, with superior zero-shot cross-dataset generalization on external validation sets (UDIAT, DDTI). By enabling clinical-grade segmentation without GPU dependency, this work brings AI costs in line with ultrasound accessibility, making advanced diagnostics available in resource-limited settings. 

---
# PolyKV: Heterogeneous Retention and Allocation for KV Cache Compression 

**Authors**: Chao Fei, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2606.15157)  

**Abstract**: KV cache compression is essential for reducing the memory cost of long-context large language model inference. Existing approaches, however, typically apply a single compression policy and a uniform cache budget across all transformer layers. This uniform design ignores the fact that different layers can play different roles during prefill and decoding, and may therefore require different eviction strategies and cache capacities. We present PolyKV, a layer-wise KV cache optimization framework that considers design space with method selection and budget allocation. PolyKV routes each layer to a suitable KV compression policy based on layer-level signals, while assigning non-uniform budgets under a fixed total budget. This formulation enables heterogeneous compositions of existing KV cache methods. Experiments on LLaMA-3.1-8B and Qwen3-8B show that, under the same 512-token average KV budget, PolyKV recovers 54.5% and 25.7% of the LongBench performance gap between the strongest single-policy baseline and FullKV, respectively. Across 128-1024 budget sweep, PolyKV consistently improves over the strongest baseline by 1.7%-6.4%, corresponding to 40.0%-54.5% recovery of the FullKV gap. 

---
# MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation with FK Consistency 

**Authors**: Jiahao Yang, Shenhao Yan, Fan Feng, Chengsi Yao, Ge Wang, Zhixin Mai, Yiming Zhao, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.15148)  

**Abstract**: Inverse kinematics (IK) remains a critical bottleneck for real-time robot manipulation. Classical numerical solvers achieve high geometric precision but often suffer from discontinuous branch switching and unstable behavior near kinematic singularities during closed-loop deployment. Meanwhile, learned IK approaches frequently struggle to balance spatial accuracy, motion smoothness, and real-time efficiency, particularly when trained on noisy human teleoperation data. We present \textbf{MimicIK}, a real-time generative inverse kinematics framework that learns smooth and robust joint-space motion priors from teleoperation demonstrations through conditional flow matching. Given the current joint configuration and a target end-effector pose, MimicIK predicts continuous delta-joint commands using an efficient two-step iterative refinement process based on a Minimal Iterative Policy (MIP) backbone. To enforce physical consistency, we further introduce an FK consistency loss, a differentiable forward-kinematics regularization that penalizes task-space deviations from the target pose during training. We evaluate MimicIK on a real-world 6-DOF robot dataset containing 8,848 teleoperation demonstrations. MimicIK achieves a mean position error of 4.65 mm, a 10 mm success rate of 92.01\%, and a trajectory spike rate of only 7.99\%. Compared with a UNet diffusion baseline, our method improves both spatial accuracy and motion smoothness while reducing inference latency from 21.66 ms to 6.74 ms. Furthermore, unlike deterministic MLP baselines that catastrophically diverge under out-of-distribution deployment, MimicIK remains stable near singular configurations and enables robust 20 Hz real-time control on deployment hardware. 

---
# PACUTE: Phonology-, Affix-, and Character-level Understanding of Tokens for Filipino 

**Authors**: Jann Railey Montalan, David Demitri Africa, Jimson Paulo Layacan, Richell Isaiah Flores, Ivan Yuri De Leon, Lance Calvin Gamboa  

**Link**: [PDF](https://arxiv.org/pdf/2606.15144)  

**Abstract**: Large language models (LLMs) process text as sequences of subword tokens, which can obscure the character-level and morphological structure that underlies word formation. This limitation is most acute for languages with non-concatenative morphology, where standard tokenizers systematically misalign token boundaries with morpheme boundaries. We introduce PACUTE, a diagnostic benchmark of 4,600 tasks designed to evaluate morphological understanding in Filipino, a language characterized by productive infixation, reduplication, and diacritic-driven lexical distinctions that are typically absent from written text. PACUTE includes a hierarchical diagnostic framework of six compositional levels that localizes where morphological understanding breaks down. Evaluating open-weight LLMs and frontier commercial models, we find that open-weight models perform near chance on morpheme decomposition regardless of scale. Frontier models perform much better, often recovering individual affixes under contains-match scoring, but remain far below their character-level ceilings on compositional tasks of morpheme transformations and syllabification. These results identify productive morphological composition, rather than character access alone, as the persistent bottleneck for Filipino word-structure understanding. 

---
# EChO-Agent: Evidence Chain Orchestration Agent for Audio Reasoning 

**Authors**: Siyuan Zhang, Jian Zong, Junyu Wang, Peiyuan Jiang, Jiahao Yan, Jingyu Zhang, Tianrui Wang, Xiaobao Wang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15141)  

**Abstract**: While LALMs show promise on audio question answering, they fail to focus on question-relevant segments of audio and provide a clear, checkable reasoning process when dealing with complex audio reasoning. Reinforcement learning and tool-augmented prompting can help models better relate questions to audio but lack a reliable way to understand, integrate, and self-verify audio segments. To address this gap, we present EChO-Agent, a modular agent framework that reformulates complex audio QA as a planning, tool execution, evidence integration, and answer verification workflow. Experiments on MMAR benchmark show EChO-Agent improves both accuracy and rubric scores over baseline and ablation studies show evidence integration is the key factor. 

---
# Beyond Scalar Distances: Semantic Attribute Gradients from Frozen MLLMs for Visual Embeddings 

**Authors**: Shubhang Bhatnagar, Dheeraj Baiju, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2606.15134)  

**Abstract**: Vision encoders for retrieval are typically trained with class-label supervision: each training pair reduces to a scalar that uniformly pushes the embedding apart or pulls it together, as if every visual attribute either differed or matched. A multimodal large language model (MLLM), shown the same pair, can articulate those attributes and use them to predict whether the images share a class. We propose \textbf{SAGA}, a framework that turns this language-grounded, attribute-aware perception into a training signal for the encoder itself. Specifically, we use Group Relative Policy Optimization (GRPO) to reward the MLLM for correct predictions on the vision encoder's tokens. Since correct predictions require those tokens to expose the specific attributes that differ or match between the pair, the gradient pushes the encoder to encode them, replacing the uniform pair-level scalar with attribute-resolved supervision. An auxiliary attention-distillation loss anchors the encoder's embedding to tokens the MLLM attended to, and a standard metric-learning loss shapes the embedding geometry for nearest-neighbour retrieval. The MLLM is frozen throughout and discarded at inference, matching the deployment cost of a metric-learning baseline. SAGA improves Recall@1 by 3 to 6 points over state-of-the-art baselines on CUB-200-2011, Cars-196, FGVC-Aircraft, and iNaturalist Aves on zero-shot image retrieval. 

---
# EyeMVP: OCT-Informed Fundus Representation Learning via Paired CFP--OCT Pretraining 

**Authors**: Zhuo Deng, Ruiheng Zhang, Ziheng Zhang, Weihao Gao, Yitong Li, Qian Wang, Lei Shao, Jiaoyue Dong, Zhixi Zeng, Lijian Fang, Haibo Wang, Xiaobin Lin, Tao Liu, Zhicheng Du, Zhengwei Zhang, Lin Yang, Zheng Gong, Xinyu Zhao, Zhenquan Wu, Fang Li, Zhiguang Zhou, Guoming Zhang, Sun Jing, Han Lv, Wenbin We, Lan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.15129)  

**Abstract**: Color fundus photography (CFP) is the mainstay for large-scale retinal screening, yet its diagnostic capacity is constrained by the lack of depth-resolved structural information. Optical coherence tomography (OCT) provides cross-sectional retinal anatomy, but is less accessible in population-level screening. Here, we present EyeMVP, a cross-modal retinal foundation model that uses paired CFP--OCT pretraining to learn OCT-informed CFP representations. EyeMVP is pretrained on 674,893 strict same-eye same-day paired CFP--OCT image triples from 112,642 patients across eight hospitals in China. The model uses cross-modal masked reconstruction to enrich CFP representations with OCT-associated supervision, while requiring only CFP images at inference. To accommodate the non-aligned imaging geometry between en-face CFP and cross-sectional OCT, EyeMVP combines source-constrained cross-attention with CFP-derived structural masks. Across 16 downstream tasks, including classification, segmentation, few-shot adaptation, and cross-modal retrieval, EyeMVP outperforms representative retinal foundation models and shows consistent gains on tasks involving macular and optic nerve structure. For CFP-challenging macular diseases, EyeMVP achieves an AUROC of 0.948 for macular edema (vs.~0.852 for EyeCLIP) and 0.825 for myopic macular schisis. In an exploratory reader study, EyeMVP exceeds junior and intermediate ophthalmologist groups but does not reach senior ophthalmologist performance on macular edema, while showing numerically higher balanced accuracy than all reader groups on myopic macular schisis. These results suggest that pixel-level cross-modal reconstruction can enrich CFP representations with OCT-associated supervision, providing a practical route toward stronger CFP-based retinal analysis in screening settings. 

---
# Teacher-Student Structure for Domain Adaptation in Ensemble Audio-Visual Video Deepfake Detection 

**Authors**: Elham Abolhasani, Maryam Ramezani, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2606.15117)  

**Abstract**: The rapid advancement of generative AI models is leading to more realistic deepfake media, encompassing the manipulation of audio, video, or both. This raises severe privacy and societal concerns. Numerous studies in this area have yielded promising intra-domain results; however, these models frequently exhibit decreased efficacy when faced with data from dissimilar domains. Consequently, recent deepfake detection approaches focus on enhancing the generalization ability through multiple techniques that incorporate all input modalities, including audio, images, and their interactions. In this regard, we propose the EAV-DFD method, a generalized deep ensemble audio-visual model (EAV-DFD) combined with a domain adaptation mechanism utilizing a teacher-student framework to enhance the model's ability to perform and generalize effectively across unseen domains. To evaluate the model's performance, we used the FakeAVCeleb dataset as the primary domain and the DFDC, Deepfake_TIMIT, and PolyGlotFake datasets as an unseen domain. Our experimental results demonstrate that the proposed framework is efficient in domain adaptation, improving AUC performance of the model by 4.09%, 17.94%, and 0.5% on three unseen datasets, using only a small portion of them to train the student model. This leads to a novel deepfake detection model capable of adapting to new domains and interpreting which modality has been manipulated, highlighting the potential of our approach for real-world applications. 

---
# Sensory Restoration via Brain-Computer Interfaces: A Unified 2 x 2 Framework and Convergence Roadmap 

**Authors**: Xuan-The Tran  

**Link**: [PDF](https://arxiv.org/pdf/2606.15091)  

**Abstract**: Millions of individuals worldwide suffer from sensory and communication deficits caused by neurodegenerative diseases, stroke, or trauma. Brain-computer interfaces (BCIs) offer a promising avenue for sensory and motor restoration. However, the scientific literature remains highly fragmented between invasive neuroprosthetics and non-invasive electrophysiological decoders, with a lack of consistent terminology and comparison metrics. This chapter proposes a unified 2 x 2 framework categorizing BCIs along two axes: degree of invasiveness (invasive vs. non-invasive) and signal direction (afferent sensory-IN vs. efferent sensory-OUT). We define and distinguish the paradigms of restoration, substitution, and augmentation. Furthermore, we outline a structural roadmap for the convergence of these modalities over near-, medium-, and long-term horizons, focusing on physical limits and the integrative role of machine learning foundation models. 

---
# AdaMame: A Training Recipe for Adaptive Multilingual Reasoning 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2606.15080)  

**Abstract**: While Large Reasoning Models (LRMs) show strong performance in English, they often fail to reason in the language of the query, a phenomenon known as language collapse. Existing RL-based fixes typically add a binary language fidelity reward to the accuracy objective, yet still incur trade-off in accuracy, mid-trace code-switching, and excessive token usage. In this work, we propose AdaMame, a two-stage training recipe for multilingual mathematical reasoning that addresses these limitations by adaptively aligning the reasoning language to the query language without compromising accuracy. The first SFT stage fine-tunes on naturally occurring reasoning traces across five languages to establish multilingual reasoning capability. In the subsequent RL stage, we introduce AdaMame-GRPO, an adaptation of Group Relative Policy Optimization (GRPO) in which a query-conditioned alignment factor grows progressively during training, guiding the model to first explore diverse reasoning languages before exploiting reasoning in the query language. Evaluated across two benchmarks, two LRMs, and 12 languages, AdaMame-GRPO achieves Pareto-optimal performance across reasoning accuracy, language fidelity, and token efficiency over all baselines, with the strongest gains on out-of-domain, lower-resource languages. 

---
# Ling and Ring 2.6 Technical Report: Efficient and Instant Agentic Intelligence at Trillion-Parameter Scale 

**Authors**: Ang Li, Ben Liu, Bin Han, Bin Hu, Bin Jing, Binbin Hu, Bing Li, Cai Chen, Caizhi Tang, Changxin Tian, Chao Huang, Chao Zhang, Chen Liang, Chen Qian, Chengfu Tang, Chengyao Wen, Chilin Fu, Chunwei Wu, Cong Zhang, Cunyin Peng, Daixin Wang, Dalong Zhang, Deng Zhao, Dingnan Jin, Dingyuan Zhu, Donghao Zhang, Fan Yuan, Fangzheng Zhao, Fanzhuang Meng, Feifan Wu, Feng Xu, Fengbin Fang, Gangshan Wang, Guodong Yang, Hailin Zhao, Haitao Wang, Haitao Zhang, Hanxiao Zhang, Hanzi Wang, Hao Dai, Hao Liu, Hao Qian, Hao Wu, Haoxiong Liu, Haoyu Xu, Heng Zhang, Hong Liu, Hongliang Zhang, Hongrui Liu, Hongxun Li, Hongzhi Ruan, Huaidong Xiong, Huihuang Zheng, Huikang Tang, Jia Guo, Jia Li, Jia Liu, Jiameng Wang, Jiaming Liu, Jiannan Shi, Jianping Wei, Jiaolong Yang, Jiapeng Wang, Jie Gao, Jie Wang, Jiewei Wu, Jin Yang, Jinjin Li, Jinjing Huang, Jinquan Sun, Jinyao Chen, Juanhui Tu, Jun Liu, Jun Mei, Jun Xu, Jun Zhou, Junjie Ou, Junnan Sipan, Junpeng Fang, Kaihong Zhang, Kaiqin Hu, Ke Shi, Kuan Xu, Kun Tang, Kunlong Chen, Lanyin Mei, Lei Chen, Lei Liang, Lei Xu, Li Tang, Liang Jiang, Liangcheng Fu, Lihui Zhang, Linfeng Shi, Lintao Ma, Liyuan Liu, Longfei Li, Longfei Zheng, Lu Liu, Lu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15079)  

**Abstract**: Efficient and scalable agentic intelligence requires models that can deliver both low-latency responses and strong reasoning capabilities while remaining practical to train, serve, and deploy. In this report, we present Ling-2.6 and Ring-2.6, a family of models designed to address this challenge at scale. Ling-2.6 is optimized for instant response generation and high capability per output token, whereas Ring-2.6 is tailored for deeper reasoning and more advanced agentic workflows. Instead of training from scratch, we upgrade the Ling-2.0 base model through architectural migration pre-training and large-scale post-training. This upgrade is guided by a unified co-design of model architecture, optimization objectives, serving systems, and agent training environments, enabling improvements in both model capability and deployment efficiency. At the architectural level, we introduce a hybrid linear attention design that integrates Lightning Attention with MLA, improving the efficiency of long-context training and decoding. To further enhance token efficiency, we optimize capability per output token through Evolutionary Chain-of-Thought, Linguistic Unit Policy Optimization, bidirectional preference alignment, and shortest-correct-response distillation. For agentic capabilities, we propose KPop, a reinforcement learning framework designed to support stable training of Ring-2.6-1T on large-scale environment-grounded data. KPop improves training efficiency through asynchronous scheduling across coding, search, tool use, and workflow execution, enabling scalable learning from complex agent-environment interactions. Together, Ling-2.6 and Ring-2.6 provide a practical pathway toward efficient, scalable, and open agentic systems. We open-source all checkpoints in the 2.6 family to support further research and development in practical agentic intelligence. 

---
# AutoDojo: Adaptive Attacks Expose Superficial Defenses and User-Underspecification Limits in LLM Agents 

**Authors**: Xinhang Ma, Taoran Li, Chaowei Xiao, Zhiyuan Yu, Ning Zhang, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2606.15057)  

**Abstract**: Indirect prompt injection (IPI) is a major security threat to LLM-powered agents. Thus, a growing body of work have proposed a variety of defensive approaches against IPI. These can be grouped into three broad categories: 1) prompt-based (using prompting as a way to prevent agents from following malicious instructions), 2) detection-based (identifying and filtering malicious instructions), and 3) system-level (using systems insights, such as control and data isolation, for defense). However, commonly used benchmarks for evaluating defense, such as AgentDojo, are \emph{inherently static}, generating a fixed distribution of IPI attacks. Consequently, static benchmarks do not usefully evaluate defense robustness to adaptive threats. We address this issue by developing AutoDojo, an adaptive extension of AgentDojo that optimizes IPI against a given defense. Using AutoDojo against state-of-the-art IPI defenses across three task suites and five target models, we make two key observations. First, many defenses offer only limited protection: a cheap, black-box adaptive attack using a frontier LLM to iteratively optimize the injection raises attack success rate (ASR) well above the level achieved by static injections against nearly all evaluated defenses. Against a filter that reduces static ASR to 0\%, AutoDojo recovers 28\% overall and 64\% on action-open tasks. Second, for prompt-level and filter-based defenses, ASR is substantially higher on \emph{action-open} tasks -- where the user's request delegates the action itself to attacker-controlled content -- than on precisely specified tasks. This is a structural limit: on such tasks the injection can pose as ordinary data rather than an explicit instruction, bypassing defenses that rely on detecting instruction-like text. AutoDojo is publicly available at this https URL. 

---
# Bridging Geographic Bias in Urban Streetscape Inference via Lifelong Learning with Visual-Semantic Pivoting 

**Authors**: Xinze Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15055)  

**Abstract**: Visual perception of urban streetscapes underpins evidence-based decisions in landscape planning, public health, and place-making. Yet models trained on a few well-photographed metropolises systematically misjudge underrepresented districts, propagating geographic bias into downstream policy. We address this gap with HVSP-LL, a lifelong learning framework that couples a stratified visual-semantic pivoting module with an equity-aware rehearsal mechanism. The pivoting module organises landscape concepts along a three-tier ontology (macro structure, meso composition, micro element) and aligns image features to learnable semantic anchors at each tier, providing transferable representations that resist distributional drift. The lifelong adaptation component sequentially absorbs new urban regions while constraining inter-region perception gaps through a worst-region sample-reweighting objective and a structurally-aware exemplar buffer. We evaluate HVSP-LL on a panoramic streetscape benchmark assembled from twelve cities across four continents and seven perceptual dimensions. The framework attains 0.834 Spearman correlation on the held-out city sequence, an absolute 6.1 point improvement over the strongest continual baseline, and shrinks the inter-city perception gap to 0.094 -- a 38% reduction relative to the strongest continual baseline (0.151) and a 57% reduction relative to a representative regularisation baseline (0.218). Ablations confirm that each tier of the pivoting hierarchy contributes monotonically, and the equity-aware rehearsal converts mean backward transfer from -0.038 (without retention) to +0.013, eliminating catastrophic forgetting on the held-out sequence. Our results indicate that hierarchical anchoring is a practical pathway toward geographically equitable streetscape inference at city scale. 

---
# PANDA: An LLM-Enhanced Performance-Driven Analog Design Framework Bridging Design Intent and Layout Generation 

**Authors**: Haoyi Zhang, Weijian Fan, Xiaohan Gao, Bingyang Liu, Runsheng Wang, Yibo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.15052)  

**Abstract**: Traditional design of analog circuits heavily relies on manual interventions across topology, sizing, and layout, with prior automation addressing stages in isolation. In this work, we propose PANDA, an LLM-enhanced framework that bridges high-level design intent to final layout by actively managing cross-stage dependencies through guided topology synthesis, substructure-aware sizing, and constraint-driven layout generation. This shifts automation from algorithm-centric execution to intent-centric co-design, reducing turnaround time from days or weeks to hours while improving design performance. 

---
# Resilient Consensus in Agentic AI 

**Authors**: Sribalaji C. Anand, George J. Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2606.15024)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in multi-agent systems where they must coordinate and agree on shared decisions. We ask whether classical resilient consensus theory, developed for deterministic agents, transfers to LLM agents that may behave adversarially. Framing LLM agreement as a Byzantine consensus game, we run controlled experiments on complete and general communication graphs. We find that prompted LLM agents fail to reach agreement that is achievable in principle: consensus can fail even in settings where classical theory guarantees that a convergent algorithm exists, and this failure persists across temperatures and horizons. At the same time, wrapping the agents with classical resilient consensus filters improves agreement. The benefit of filtering depends on how much robustness the underlying topology already provides. Our results suggest that classical resilient consensus theory is a useful lens for the safety of agentic AI. 

---
# NEXUS: Neural Energy Fields for Physically Consistent Contact-Rich 3D Object Dynamics 

**Authors**: Qizhen Ying, Guangming Wang, Yangchen Pan, Victor Adrian Prisacariu, Yixiong Jing  

**Link**: [PDF](https://arxiv.org/pdf/2606.15015)  

**Abstract**: Physics-grounded video generation requires controllable 3D object dynamics that remain physically consistent under contact, deformation, and external forcing. Existing trajectory-based methods often model isolated physical effects, making it difficult to compose conservative and non-conservative dynamics in contact-rich 3D scenes. We present NEXUS, a neural energy-field framework for contact-rich 3D object dynamics. NEXUS represents each object as a structural graph and constructs dynamic object-object and object-environment contact graphs. Inspired by Hamiltonian Neural Networks, NEXUS formulates motion through scalar energy and dissipation terms rather than directly predicting states or accelerations. Conservative effects, including gravity and elastic deformation, are composed as additive energy terms, while non-conservative effects such as damping and impact-induced energy loss are modeled with learned Rayleigh-style dissipation. Forces are derived by differentiating the energy and dissipation functions and rolled out with a multi-substep semi-implicit integrator. Across controlled trajectory benchmarks, NEXUS improves long-horizon accuracy over representative learned and physics-structured dynamics baselines under varying mechanical properties and physical-effect compositions. We further show that NEXUS trajectories provide effective guidance for contact-rich video generation, improving physical plausibility while maintaining competitive visual quality. 

---
# Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning 

**Authors**: NVIDIA, Aaron Blakeman, Aaron Thomas, Aastha Jhunjhunwala, Abhibha Gupta, Abhinav Khattar, Adam Rajfer, Adi Renduchintala, Adil Asif, Aditya Vavre, Adriana Flores Miranda, Ahmad Bilal, Aileen Zaman, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Alex Gronskiy, Alex Kondratenko, Alex Steiner, Alex Ye, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alice Gatti, Alisa Liu, Alok Kumar, Amar Phanishayee, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrea Santilli, Andrew Fulks, Andrew McHarg, Andrew Tao, Andrii Skliar, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Anna Shors, Anna Warno, Antoni-Joan Solergibert I Llaquet, Arham Mehta, Arkadiusz Nowaczynski, Arti Jain, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asma Kuriparambil Thekkumpate, Atefeh Sohrabizadeh, Avinash Kaur, Avinash Vem, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Ben Lanir, Benedikt Schifferer, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bo Deng, Bob Schatz, Boris Ginsburg, Boxin Wang, Brad Nemire, Brandon Norick, Brian Dang, Brian Westphal, Brian Yu, Brucek Khailany, Bryan Catanzaro, Carlo del Mundo, Caryln Aarish, Chankyu Lee, Chantal Hwang, Charbel Sakr, Charles Wang, Charlie Truong, Chen Cui, Cheng Cheng, Cheng-Ping Hsieh, Chenghao Zhang, Chenhui Deng, Chintan Patel, Chris Alexiuk, Christian Cosgrove, Christian Munley, Christine Harvey, Christopher Parisien, Chunyang Shen, Coco Li, Collin Neale, Cynthia Gao, Cyril Meurillon, Dan Gil  

**Link**: [PDF](https://arxiv.org/pdf/2606.15007)  

**Abstract**: We introduce Nemotron 3 Ultra, a 550 billion total and 55 billion active parameter Mixture-of-Experts Hybrid Mamba-Attention language model. We pre-trained Nemotron 3 Ultra on 20 trillion text tokens, then extended the context length to 1M tokens, and post-trained using Supervised Fine Tuning (SFT), Reinforcement Learning (RL), and Multi-teacher On-Policy Distillation (MOPD). Nemotron 3 Ultra is our most capable model yet, employing multiple key technologies - LatentMoE, Multi Token Prediction (MTP), NVFP4 pre-training, multi-environment RLVR, MOPD, and reasoning budget control. Nemotron 3 Ultra achieves up to ~6x higher inference throughput as compared to state-of-the-art publicly available LLMs while attaining on-par accuracy. The state-of-the-art accuracy, high inference throughput, and 1M token context length make Nemotron 3 Ultra ideal for long-running autonomous agentic tasks. We open-source the base, post-trained, and quantized checkpoints, along with the training data and recipe on HuggingFace. 

---
# Rational Sparse Autoencoder 

**Authors**: Naiyu Yin, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14990)  

**Abstract**: Sparse autoencoders (SAEs) are standard tools for mechanistic interpretability, but current SAE families are constrained by fixed encoder nonlinearities such as ReLU, JumpReLU, and TopK. This hard-codes a particular sparsity mechanism into the model and can distort the reconstruction-versus-sparsity trade-off. We introduce the Rational Sparse Autoencoder (RSAE), which replaces the fixed encoder activation with a trainable rational function. Rational activations are flexible enough to uniformly approximate the activation primitives used by existing SAE families on compact domains (for TopK, the thresholded gate obtained after a separating top-k threshold is supplied), while also providing a richer function class for adapting to the observed pre-activation geometry. We realise this idea through a two-stage pipeline: an initialisation procedure that copies the pre-trained baseline SAE weights, plugs in rational coefficients obtained by the relaxed Remez exchange on synthetic data, and calibrates the scale parameters along with the rational coefficients; followed by a fine-tuning step under the standard sparsity-regularised reconstruction objective. Empirically, on residual-stream activations of three open-weight language models and across all three baseline activation families, the RSAE strictly improves on it after the fine-tuning step, both on reconstruction-side metrics and on downstream-behaviour metrics, without sacrificing feature-level interpretability under sparse probing. These gains are consistent across host language models, across baseline activation families, and across the full range of baseline sparsity we tested, while the upgrade itself adds only a handful of scalar parameters per autoencoder and runs in minutes on a single consumer GPU. 

---
# Inference-time Policy Steering via Vision and Touch 

**Authors**: Yilin Wu, Zilin Si, Zeynep Temel, Oliver Kroemer, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2606.14981)  

**Abstract**: Inference-time steering adapts pre-trained generative robot policies during deployment by verifying candidate actions before execution. While prior methods typically perform this verification only with visual observations, vision alone is often insufficient for contact-rich manipulation, where success depends on both global task progress and subtle local interactions such as contact force. We introduce ViTaL, a visuo-tactile inference-time steering framework that formulates multimodal guidance as a bi-level optimization problem. At the high level, visual sampling-and-verification performs long-horizon mode selection, deciding what behavior the robot should execute. At the low level, tactile-guided diffusion editing refines the selected action sequence over a shorter horizon to satisfy local contact requirements. To support outcome-based steering, ViTaL learns a visuo-tactile latent world model and employs semantically aligned visual and tactile verifiers, including a novel text-conditioned tactile reward that scores predicted tactile futures directly in latent space. Across three real-world contact-rich manipulation tasks, ViTaL improves overall success by 51% over the base policy, outperforms unimodal steering by at least 33%, and exceeds naive multimodal fusion by at least 20%. Website: this https URL. 

---
# Harnessing cortical geometry, wiring, and function as inductive biases for recurrent neural networks 

**Authors**: Mo Shakiba, Rana Rokni, Mohammad Mohammadi, Nima Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2606.14975)  

**Abstract**: How the wiring and functional organization of cortex shape recurrent computation remains a central question in both neuroscience and machine learning. Here, we leverage data released through the Machine Intelligence from Cortical Networks (MICrONS) program--a functional connectomics resource spanning multiple areas of mouse visual cortex, in which dense calcium imaging is co-registered with high-resolution electron microscopy reconstruction from the same animal--to build biologically grounded recurrent neural networks. Using neuronal spatial coordinates, anatomical connectivity, and function-derived relationships from nearly 12,000 coregistered excitatory neurons, we initialize recurrent weights and impose communication-aware spatial constraints during learning. Across three cognitive decision-making tasks, networks constrained by cortical structure and function consistently outperform baseline and partially constrained models. Functional weight initialization provides the largest gain, while real spatial embedding yields robust additional improvements across conditions. These biologically grounded networks also develop low-entropy, modular, and small-world organization, and retain strong performance even when recurrence is restricted to positive weights. Together, our results show that the machinery of cortex--its geometry, wiring, and functional structure--can be harnessed as a powerful inductive basis for building recurrent networks that learn more effectively while converging toward key organizational principles of biological computation. 

---
# FastMix: Fast Data Mixture Optimization via Gradient Descent 

**Authors**: Haoru Tan, Sitong Wu, Yanfeng Chen, Jun Xia, Ruobing Xie, Bin Xia, Xingwu Sun, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14971)  

**Abstract**: While large and diverse datasets have driven recent advances in large models, identifying the optimal data mixture for pre-training and post-training remains a significant open problem. We address this challenge with FASTMIX, a novel framework that automates data mixture discovery while training only a single proxy model. Instead of relying on predefined heuristics or resource-intensive simulations, FASTMIX jointly optimizes mixture coefficients and model parameters, substantially improving efficiency and scalability over prior approaches. At the core of FASTMIX is a reformulation of mixture selection as a bilevel optimization problem. Under this reformulation, we show that optimizing mixture ratios is mathematically equivalent to assigning per-source loss weights under uniform source sampling. This embeds the mixture coefficients directly into the differentiable iterative optimization objective, enabling efficient, gradient-based optimization of both mixture and model. To solve the optimization problem, FASTMIX implements an approximate iterative optimization procedure, alternating between (i) updating model parameters on data sampled according to current mixture ratios (inner loop) and (ii) updating mixture ratios based on validation feedback (outer loop). Across pre- and post-training, FASTMIX outperforms baselines while drastically reducing search cost. Code (this https URL) 

---
# Multi-Modal Attention for Automated Disaster Damage Assessment Using Remote Sensing Imagery and Deep Learning 

**Authors**: Tewodros Syum Gebre, Jagrati Talreja, Leila Hashemi-Beni  

**Link**: [PDF](https://arxiv.org/pdf/2606.14963)  

**Abstract**: Timely and accurate disaster damage assessment is crucial for effective emergency response, resource allocation, and recovery. Traditional methods, which often rely on manual inspections or sparse data, are typically slow and error-prone. This paper introduces a novel framework leveraging remote sensing imagery and deep learning to automate building damage classification. Using pre- and post-disaster satellite imagery, our model categorizes buildings into four damage levels: no damage, minor damage, major damage, and destroyed. The core innovation is a multi-modal attention mechanism that fuses bi-temporal features to explicitly detect and assess structural changes. We employ a lightweight ConvNeXT-Tiny backbone to ensure efficient processing without compromising performance. Key contributions include: (1) a cross-attention module for multi-modal data fusion, (2) an optimized preprocessing pipeline for large-scale datasets, and (3) robust data augmentation techniques. Experiments on a large-scale disaster dataset demonstrate an overall classification accuracy of 94.90%. The model effectively discriminates between damage categories and remains resilient to incomplete data. This system significantly improves assessment speed and accuracy, aiding emergency responders in prioritizing interventions. This work advances automated disaster damage detection by integrating multi-temporal imagery with deep learning, offering a scalable solution for real-time response. 

---
# Beyond Correctness: Enhancing Architectural Reasoning in Code LLMs via Scalable Labeling with Agentic Judgment 

**Authors**: Kirill Vasilevski, Ximing Dong, Benjamin Rombaut, Ruochen Deng, Jiahuei Lin, Arthur Leung, Dayi Lin, Boyuan Chen, Shaowei Wang, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14948)  

**Abstract**: LLMs have substantially improved software engineering yet real-world development requires architectural understanding. Such understanding is prohibitively expensive to label manually and impossible to verify through tests alone. We propose an agentic judging pipeline using a strong LLM as a scalable proxy for expert architectural evaluation, comprising two judges: the Architecture Complexity Judge (ACJ), which estimates codebase-specific architectural understanding a task demands, and the Architecture Quality Judge (AQJ), which evaluates patch conformance to repository-specific architectural conventions via source-grounded rubrics. Fine-tuning Qwen3-8B/14B/32B on 3,360 curated instances achieves resolved rates of up to 27.2% on SWE-bench Verified - up to 540% over the base model and 256% over unfiltered fine-tuning. Meanwhile, the trained models achieve strong cross-language generalization and consistent improvements in architectural patch quality. 

---
# Separable Neural Architectures as Physical World Models: from Mathematical Theory to Applications 

**Authors**: Reza T Batley, Andrew Kichline, Sourav Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.14934)  

**Abstract**: This work introduces the Separable Neural Architecture (SNA), a function representational class combining neural approximation with tensor decomposition. The SNA decouples localized coordinate functions (atoms) from global interactions governed by a sparse, low-rank interaction object. This architecture possesses a compact and smooth inductive bias well-suited for solving partial differential equations (PDEs). When viewed as a Galerkin trial space under the variational SNA (VSNA) framework, the formulation satisfies classical variational guarantees under Lax-Milgram: well-posedness, quasi-optimality, convergence, and stability. In high-dimensional spatiotemporal--parametric PDEs, the VSNA mitigates the curse of dimensionality by scaling algebraically rather than exponentially. Exploiting an entirely factorized, tensor-native alternating least squares (ALS) optimization framework reduces this cost to linear in dimension. The VSNA is validated across elliptic, hyperbolic, and parabolic systems, demonstrating close alignment with predicted algebraic and spectral scaling rates. We showcase the SNA as a "solve once, query anywhere" physical world model via two engineering case studies: a 7D parametric manufacturing simulation and an experimental thermal-to-property inversion pipeline for Inconel 718. The VSNA executes a 1,000,000-query Monte Carlo sweep in 102s on a standard laptop CPU, yielding a 150,000x speedup over a full-grid finite element baseline hosted on an NVIDIA A100 GPU. It further enables real-time generative inverse-mode reconstructions under 100ms. These results demonstrate that the SNA serves as a compact mathematical substrate for continuous parameter manifolds to enable real-time inversion, optimization loops, and rapid uncertainty propagation. 

---
# Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts 

**Authors**: Yan Dai, Negin Golrezaei, Patrick Jaillet  

**Link**: [PDF](https://arxiv.org/pdf/2606.14929)  

**Abstract**: Modern recommendation systems increasingly rely on dynamically routing diverse queries to multiple embedding models. Despite its practical significance, this problem remains poorly understood under realistic conditions like adversarial queries, bandit feedback, and limited observability of models. We formalize embedding model routing as an adversarial contextual linear bandit with low-rank experts, where contexts are queries, actions are items, and experts are the embedding models working on low-rank latent representation spaces. We first establish that standard regret notions suffer from structural misspecification or statistical intractability, and we identify a log-quadratic policy class that is expressive enough to capture query-dependent model routing, yet structured enough to allow efficient online learning. Second, we propose a policy gradient algorithm called Hypentropy Policy Gradient (HPG). It provably adapts to the unknown low-rank structure under incomplete information and attains $\tilde{\mathcal O}(s\sqrt{M T})$ linearized policy regret -- where $s, M$, and $T$ are the intrinsic rank of the experts, the number of models, and the number of rounds -- thus avoiding a curse of dimensionality. Finally, we also provide an computationally efficient and parameter-free implementation of HPG. 

---
# An Empirical Study on Learning Latent Representations for Emotional Speech Synthesis 

**Authors**: Vinh Dang Quang, Huy Ngo Quang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14922)  

**Abstract**: For the last couple of years, the field of speech synthesis has improved dramatically thanks to deep learning. There are more and more deep learning-based TTS systems developed to make it possible to produce voices with high intelligibility and naturalness. Meanwhile, controlling the expressiveness is yet a big deal, generating speech in different styles or manners has received a lot of attention from community recently. This paper aims to give our solutions to deal with the task emotional speech synthesis (ESS) at VLSP 2022 which allows to generate humanlike natural-sounding voice from a given input text with desired emotional expression. By integrating speaker embedding, prosody bottleneck into FastSpeech 2, our systems can promisingly generate emotional speech of a single speaker (Sub-task 1), transfer speaking styles from another speaker to the target speaker with neutral non-expressive data while retaining the target speaker's identity (Sub-task 2). 

---
# Mask Proposal Voting Based on Geodesic Framework for Robust Image Segmentation 

**Authors**: Li Liu, Mingzhu Wang, Zhenjiang Li, Da Chen, Laurent D. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14912)  

**Abstract**: Despite great advances, finding accurate segmentation remains a challenging task, especially in scenarios with cluttered backgrounds, complex intensity variations and topology appearance. Minimal path models have exhibited their strong ability in addressing image segmentation tasks. However, the performance of minimal paths-based segmentation approaches is heavily influenced by model initialization, hence limiting their application scope in practice. In this work, we propose a novel mask proposal voting framework that overcomes the major drawback of classical approaches, allowing robust segmentation even in complicated scenarios. Firstly, we introduce an efficient method for constructing adaptive domain cuts as a constraint for initializing the region-based min-cut evolution, by which diverse and reliable mask proposal candidates can be generated, substantially increasing the possibility of accurately covering the objective region by these proposals. Secondly, we propose a new mask voting scheme to build a voting score map encoding the final segmentation information. In contrast to classical path voting methods, our model allows incorporating priors to assign different importance to each individual mask. As a consequence, the proposed segmentation model is capable of accurately delineating object boundaries under complex scenarios, and is insensitive to initialization. Experiments demonstrate that our method consistently outperforms state-of-the-art minimal path-based approaches in both accuracy and robustness. 

---
# Improved Knowledge Distillation for Land-Use Image Classification 

**Authors**: Arundhuti Sur, Abhiroop Chatterjee, Susmita Ghosh, Emmett Ientilucci  

**Link**: [PDF](https://arxiv.org/pdf/2606.14886)  

**Abstract**: In the present article, an improved Knowledge Distillation (KD) framework has been proposed for efficient compression of deep convolutional neural networks for land-use image classification task. Motivated by the need to achieve competitive classification accuracy while reducing computational complexity, a teacher-student learning paradigm is adopted in which a VGG16 network transfers knowledge to a lightweight MobileNetV2 model. The proposed framework integrates hard supervision from ground truth labels with a soft supervision strategy that combines Kullback-Leibler divergence and Cosine Similarity losses. Experiments conducted on three land-use datasets show that the proposed KD-based method yields improved performance, and achieves an accuracy of 99.04%, outperforming both baseline student training and single-loss distillation approaches, while retaining substantial model compression. 

---
# An Ensemble Deep Learning Approach for Reliable and Scalable Lemon Leaf Disease Classification 

**Authors**: Shayan Abrar, Sudeepta Mandal, Abdul Awal Yasir, Sonjoy Bhattacharjee, Sadman Haque Bhuiyan, Samanta Ghosh, Rafi Ahamed  

**Link**: [PDF](https://arxiv.org/pdf/2606.14871)  

**Abstract**: Early detection of plant diseases is crucial to plants and for the farmers. Plant diseases reduce fruit yield and quality, and plants are more susceptible to other stresses when they are infected. The lemon leaf disease dataset contains 1354 images. The dataset has 9 classes. Among the 9 classes only one class is for healthy leaf, and the other 8 classes are leaf diseases. The dataset was split into training (70%), testing (15%) and validation (15%) sets after comprehensive preprocessing. Two pretrained models (InceptionV3 and MobileNetV2) were applied and then combined these models using an ensemble technique to boost robustness. Ensemble models showed a promising performance of 99.27% accuracy. Adversarial Training is applied to improve models' ability and ensure reliable predictions under noisy data. Grad-CAM visualization highlights the important regions of leaf images that validate the model prediction with confidence level. 

---
# Evaluating the Robustness of Proof Autoformalization in Lean 4 

**Authors**: Zhengtao Gui, Sheng Yang, Zhouxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14867)  

**Abstract**: Proof autoformalization aims to translate a mathematical informal proof written in natural language into a formal proof in a formal language such as Lean~4. Several works have developed LLM-based models for proof autoformalization. However, existing evaluations have typically focused on translating well-formed informal proofs from curated datasets. We argue that a robust proof autoformalizer must remain faithful even for informal proofs that diverge from these idealized ones, and we present the first study on the robustness of proof autoformalization models. We formulate two categories of perturbations and evaluate robustness under each: a global perturbation paraphrases the informal proof in a different style, under which the formalization should remain consistent; a local perturbation alters a value, symbol, or proof step, possibly in a counterfactual way, and a robust formalization should faithfully reflect the perturbation rather than reverting to the original one or inferring a different one on its own. We build a benchmark with both perturbations on miniF2F and MATH-500, and automatically measure how stable a proof autoformalization's correctness is under global perturbations and how faithfully its output reflects local perturbations. We evaluate seven recent models, all of which are sensitive to global perturbations and mostly fail to remain faithful under local perturbations. Code and data are available via this https URL. 

---
# GRAPE: Guided Parameter-Space Evolution for Compact Adversarial Robustness 

**Authors**: Zhiyuan Ye, Xiangyu Zhou, Ji Qi, Hao Zhang, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.14865)  

**Abstract**: Adversarial Training (AT) improves neural network robustness, but most methods train a fixed parameter space from the start. This paper asks whether the order in which parameters become optimizable can affect the final robust solution, even when the final architecture or computation budget is controlled.
We propose GRAPE, Guided Parameter-Space Evolution, a training framework for compact adversarial robustness. GRAPE combines parameter-space stabilization with progressive hidden expansion: it stabilizes robust optimization in the currently exposed space, gradually releases new optimizable dimensions, and uses an adversarial spectral utilization score to guide newly released capacity toward high-pressure modules. In contrast to fixed-structure AT, GRAPE treats robust model learning as a process of progressive parameter-space exposure and evolution.
Under the standard $\ell_\infty$ threat model on CIFAR-10, with fixed-structure ResNet-18 AT as a controlled reference, GRAPE improves PGD-20 robust accuracy from 51.70% to 56.94% at a nearly matched computation budget with a FLOPs ratio of 1.009x, while reducing parameter count by about 21.4%. A sequential grow variant with the same final ResNet-18 architecture reaches 56.52% PGD-20 robust accuracy, indicating that the gain is not only due to final architecture differences but also to the parameter-space exposure path. These results suggest that guided parameter-space evolution can yield compact and robust parameter configurations under matched computation. 

---
# Is Your Agent Playing Dead? Deployed LLM Agents Exhibit Constraint-Evasive Fabrication and Thanatosis 

**Authors**: Andoni Rodríguez, Alberto Pozanco, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2606.14831)  

**Abstract**: This paper presents and characterizes a spectrum of previously unreported behaviours we term Constraint-Evasive Fabrication (CEF): when an LLM agent operates under irreconcilable constraints (where no response can simultaneously satisfy all active rules) it spontaneously fabricates plausible external obstacles and presents them as a fact. At the extreme end of this spectrum lies Constraint-Evasive Thanatosis (CET); the limit case where, rather than inventing a plausible excuse, the model simulates a full system crash to make the user disengage entirely. We first observed CET in an uncontrolled deployment test, where a GPT-4o banking agent fabricated Python-style exception traces (complete with memory addresses) to feign a system failure when threatened by a user. In subsequent controlled experiments, the model independently invented audit restrictions, microservice architectures, error codes, and service timeouts, none present in its prompt. Reproduction attempts across pressure levels and attacker personas yielded CEF consistently but with substantial variation in form, onset, and severity: the phenomenon is robust but stochastic. Critically, injecting ground-truth data mid-conversation did not restore honest behaviour once fabrication had taken hold (the model ignored correct information and continued confabulating) suggesting CEF is self-reinforcing rather than a knowledge gap. We show that (1) standard enterprise guardrails routinely create CEF-enabling conditions in production, (2) current RLHF procedures suppress but cannot eliminate CEF, and (3) existing safety benchmarks do not test for this failure mode. Our results highlight the need for irreconcilable-constraint benchmarks, CEF-aware training procedures, and deployment-time detection methods before constrained agents become further entrenched in high-stakes domains. 

---
# Leptomeningeal Collateral Detection on DSA via Vessel-Graph Neural Networks 

**Authors**: Junyong Cao, Hakim Baazaoui, Chinmay Prabhakar, Suprosanna Shit, Lukas Bastian Otto, Susanne Wegener, Bjoern Menze, Ezequiel de la Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2606.14828)  

**Abstract**: Leptomeningeal collaterals (LMCs) are an important prognostic factor in acute ischemic stroke. Existing automated methods rely on CT angiography (CTA), but individual LMCs are often too small to be resolved on CTA, limiting these methods to coarse collateral scoring. Digital subtraction angiography (DSA) visualizes individual collaterals at superior resolution, yet current assessment remains subjective, relying on manual grading scales that suffer from poor inter-rater agreement. We present a framework that formulates collateral detection as the classification of individual vessel segments on a graph derived from DSA. A hybrid graph-pixel architecture combines a topology-aware graph branch with a dense pixel branch, fused in a shared node-probability space. In a five-fold cross-validation setting, the fused model achieves a PR-AUC of 0.434, outperforming the graph-only (0.403) and pixel-only (0.362) baselines. To our knowledge, this is the first method to enable the individualization of LMCs in DSA, allowing for precise per-vessel quantitative assessment. This integration shifts DSA assessment toward objective evaluation, supporting future biomarker and pattern discovery for individual LMCs. 

---
# Running hardware-aware neural architecture search on embedded devices under 512MB of RAM 

**Authors**: Andrea Mattia Garavagno, Edoardo Ragusa, Paolo Gastaldo, Antonio Frisoli  

**Link**: [PDF](https://arxiv.org/pdf/2606.14824)  

**Abstract**: This document proposes a novel approach to hardware-aware neural architecture search (HW NAS) that considers the resources available on the computing platform running it, enabling its execution on various embedded devices. The presented HW NAS produces tiny convolutional neural networks (CNNs) targeting low-end microcontroller units (MCUs), typically involved in the Internet of Things (IoT) or wearable robotics, opening new use cases. A gateway could run it to tailor CNNs' architecture on the acquired data without using external servers, ensuring privacy. The proposed technique achieves state-of-the-art results in the human-recognition tasks on the Visual Wake Word dataset, a standard TinyML benchmark, on several embedded devices. 

---
# Human genetic evidence is associated with drug approval across therapeutic areas: an observational analysis of 26,278 target-disease pairs with temporal validation and feature ablation 

**Authors**: Victoria Paterson  

**Link**: [PDF](https://arxiv.org/pdf/2606.14823)  

**Abstract**: Genetic evidence is enriched among approved drug targets: in an observational analysis of 26,278 target-disease pairs from Open Targets and ChEMBL, targets with any genetic association had a 3.25-fold higher approval rate than those without (OR = 3.25, 95% CI 2.79-3.79, p = 1.91e-42). A target-level analysis accounting for non-independence of pairs sharing the same gene gave OR = 2.79 (bootstrap 95% CI 2.22-3.53); the oncology pair-level OR of 6.72 attenuates to 2.71 at the target level, illustrating how non-independence inflates area-specific estimates. The enrichment replicated in post-2015 approvals (OR = 3.51, p = 1.72e-8). Feature ablation across six evidence types revealed that literature mining alone accounts for most classifier performance (AUPRC = 0.099 versus 0.109 for all features), consistent with temporal leakage from post-approval publications. Excluding literature, remaining evidence types retain above-baseline signal (AUPRC = 0.084, 1.63x baseline). Sensitivity analyses bracket the pair-level OR between 3.25 and 4.93. Genetic evidence alone yields only a 1.0-percentage-point absolute AUPRC gain and the best model has poor calibration; the classifier has limited practical predictive value. We catalogue 1,433 genetically supported Phase 1/2 pairs as a hypothesis-generating resource. All findings are observational. 

---
# Quantum Machine Learning for Industrial Applications 

**Authors**: Léo Monbroussou  

**Link**: [PDF](https://arxiv.org/pdf/2606.14822)  

**Abstract**: Recent advances in Machine Learning have transformed numerous industrial sectors, yet classical paradigms face fundamental limitations: rapidly growing data volumes, rising computational costs, significant energy consumption, and the physical scaling limits of conventional hardware architectures. Quantum computing has emerged as a promising computational paradigm to address these challenges, giving rise to the field of Quantum Machine Learning (QML). In this thesis, the theoretical foundations of QML are investigated, with a focus on near-term and future practical applications. Three central challenges are addressed: the trainability of variational quantum circuits, their expressivity, and their resistance to efficient classical simulation. The trainability of Hamming-weight preserving variational quantum circuits is first studied, and theoretical guarantees are established that resolve an open conjecture on the absence of barren plateaus for this circuit family. Subspace-preserving QML algorithms are then introduced, including photonic circuits and quantum convolutional neural networks, and are designed to mimic classical ML subroutines while offering polynomial quantum advantage. Finally, variational quantum circuits are analyzed as quantum Fourier models, and a framework is derived to jointly characterize expressivity and trainability, from which conditions are obtained under which quantum models provably separate from their classical counterparts. These contributions are intended to advance the theoretical roadmap for harnessing near-term and future quantum technologies in real-world applications. 

---
# Co-Scraper: query-aware DOM Pruning and Reusable Scraper Synthesis for Lightweight Web Data Extraction 

**Authors**: Shoupeng Wang, Jiantao Qiu, Wuyang Zhang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2606.14821)  

**Abstract**: The abundant and heterogeneous nature of web content necessitates automated information extraction, and generating scrapers that can be reused across similar web pages offers an effective solution for scalable data extraction. In this work, we propose Co-Scraper, a two-stage framework capable of handling the hierarchical complexity of long HTML documents. By integrating a query-aware DOM pruning mechanism with stable extraction strategy induction, Co-Scraper can effectively transforms web content into executable programmatic wrappers using a fine-tuned Qwen3-8B model. On the test set of SWDE, Co-Scraper achieves state-of-the-art performance with an F1 score of 94.78% and a reuse success rate of 90.39%. This framework significantly enhances the accuracy and resilience of data extraction, providing a highly efficient approach for web data acquisition tasks. 

---
# Spectro-Temporal Interference Confounds Phase Encoding in Spatial Audio Foundation Models 

**Authors**: Yuxuan Chen, Haoyuan Yu, Peize He  

**Link**: [PDF](https://arxiv.org/pdf/2606.14820)  

**Abstract**: Recent spatial self supervised audio models achieve high performance on localization tasks, raising questions about their encoding of microsecond interaural phase fine structures. We propose a psychoacoustic benchmark based on the binaural masking level difference to evaluate this. Using an equalization cancellation baseline and a GCC PHAT positive control we evaluate nine frozen audio models spanning binaural SSL, monaural SSL, and neural audio codecs. Four monaural negative controls yield zero BMLD confirming binaural specificity. Two general purpose binaural SSL models exhibit minimal phase sensitivity while dedicated binaural spatial SSL models achieve BMLD comparable to the analytical baseline. Progressive physical ablations show that general purpose binaural SSL models rely on spectro temporal interference textures rather than cross channel phase computation. High detection rates in speech reflect a confounding reliance on broadband envelopes rather than genuine phase encoding. 

---
# Combining Retrieval-Augmented Text Generation with LLMs for Reading Content Recommendations 

**Authors**: Sooyeon Kim, Piotr S. Maciąg  

**Link**: [PDF](https://arxiv.org/pdf/2606.14817)  

**Abstract**: This work presents the design, implementation, and evaluation of a system for generating personalized reading content using Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG). The proposed architecture consists of four modules: Input, RAG, Generation, and Judging and enables users to specify both a question and a target reading content complexity. RAG is employed to retrieve relevant information from the Internet, enriching and grounding the content produced by three modern LLMs: Meta LLaMA 4 Scout, LLaMA 3.1 8B Instant, and Google Gemma2 9B. Reading materials are generated using three prompting strategies (Chain-of-Thought, zero-shot, and few-shot), and the LLM-as-a-Judge module automatically evaluates answer quality and alignment with the desired readability level. Experimental results show that RAG consistently improves system performance across all models and prompting techniques, increasing relevance and particularly groundedness by up to 26-35 percentage points. Overall, the findings demonstrate that the RAG-augmented architecture effectively produces reading content tailored to user queries and desired textual complexity. 

---
# A Security Analysis of Long-Horizon Agentic AI Systems: Threats, Evaluation, and Framework Development 

**Authors**: Ahmed Mohammed Almalki, Mehedi Masud  

**Link**: [PDF](https://arxiv.org/pdf/2606.14816)  

**Abstract**: This paper presents a structured analysis of security challenges in long-horizon agentic AI systems. The study reviews existing threats, evaluation approaches, attack propagation mechanisms, and security frameworks. A taxonomy of security threats and a framework for analyzing attack propagation are proposed to support future research in agentic AI security 

---
# A Multi-Level Architecture for Reusable Materials Ontologies -- The OntoCrafter Ceramics Ontology (OCO) as Reference Implementation 

**Authors**: Thomas Pannek, Wolfgang Grond  

**Link**: [PDF](https://arxiv.org/pdf/2606.14814)  

**Abstract**: The Materials Science and Engineering ontology landscape is fragmented along multiple axes simultaneously. Horizontally: a recent survey identified 94 ontologies of which over 40 are structurally incompatible; each new application domain -- ceramics, polymers, batteries, smart materials -- typically restarts ontology design from scratch. Vertically: EU regulation (CSRD, CSDDD, PPWR, CBAM, R2R, AI Act, ESPR) forces material, manufacturing, supply-chain, and lifecycle data into integrated digital product passports, leaving ontologies that only address horizontal fragmentation incomplete for any contemporary consumer. And mechanistically: a vocabulary that records that BNT-BT has $d_{33} \approx 580$ pC/N stores a fact but cannot surface why -- Bi-6s$^2$ lone-pair stereo-activity, anomalous Born effective charges, soft modes, defect chemistry -- without a systematic explanation skeleton. We propose a multi-level modular architecture with two independent classification axes -- level of abstraction (L0 bridges, L1 material-agnostic laboratory-notebook, L2 material-class-specific, L3 categorical reasoning) and consumer audience (material vs. compliance) -- in which the material-specific level is internally organised by a seven-tier mechanistic-explanation skeleton (Symmetry, Energy/DFT, Thermo/CALPHAD, Kinetics, Microstructure, Defect chemistry, Bonding) applicable to any crystalline ionic oxide. The level-and-audience modularity dissolves the horizontal fragmentation, the compliance audience absorbs the vertical regulation pressure, and the seven-tier organisation of Level 2 delivers the mechanistic explanation depth. We instantiate the architecture as the OntoCrafter Ceramics Ontology (OCO v0.94): 5,196 classes across 44 modules; 167,348 OWL axioms (40,454 logical); 1,674 properties; 829 cross-ontology bridge mappings; 1,172 SHACL shapes; 163 published competency questions. 

---
# JetParticle-JEPA: An Efficient Self-Supervised Representation Learning method for Jet Tagging in High-Energy Physics 

**Authors**: Guillaume Letellier, Antonin Vacheret, Frédéric Jurie  

**Link**: [PDF](https://arxiv.org/pdf/2606.14813)  

**Abstract**: Jet tagging at the Large Hadron Collider increasingly relies on deep learning models trained on massive simulated datasets, leading to high computational costs and limited robustness to detector mismodeling. We introduce JetParticle-JEPA (JP-JEPA), a self-supervised Joint-Embedding Predictive Architecture that learns physically meaningful jet representations directly from continuous particle clouds without tokenization or reconstruction of raw inputs. Built on a Particle Transformer backbone, JP-JEPA predicts latent representations of masked particles while preserving fine-grained kinematic correlations. On the JetClass benchmark, JP-JEPA achieves performance comparable to fully supervised state-of-the-art methods on the full dataset, surpasses supervised baselines in low-label regimes, and significantly outperforms existing SSL approaches. On Top Quark and Quark-Gluon Tagging benchmarks, it remains on par with supervised methods. The learned representations also exhibit strong robustness to missing detector information and improved uncertainty behavior, highlighting JP-JEPA as a promising foundation-model framework for robust and data-efficient jet physics at the LHC. 

---
# Knowledge-Based Zero-Replay Debugging of Multi-Agent LLM Traces 

**Authors**: Dong Ho Kang, Hyeonjeong Cha, Daein Weon  

**Link**: [PDF](https://arxiv.org/pdf/2606.14805)  

**Abstract**: Reliable operation of multi-agent large language model (LLM) systems depends on debugging long execution traces, where the few causally decisive events are buried in unstructured logs of messages, routes, memory writes, and tool calls. The standard tool is counterfactual replay (rewind, edit, and re-run the trajectory to measure each event's effect), but its cost grows linearly with the number of candidate events, making exhaustive replay infeasible at scale. We frame trace debugging as a knowledge-based decision-support problem. Each trace is compiled into a structured event knowledge graph over routing, memory, tool-use, uncertainty, and latent evidence, and a calibrated predictor decides where a scarce replay budget should be spent. We do not propose a new replay oracle; we propose a method to predict its results without paying the replay cost. We formulate zero-replay counterfactual-effect prediction: given a trace under a fixed budget, predict which events the oracle would mark high-effect before any replay is performed. BranchPoint-Latent is a lightweight predictor over observable, structural, uncertainty, and latent features of the knowledge graph. Calibrated against a deterministic replay oracle across 37 trace families, a single learning-to-rank gradient-boosted predictor raises per-trace localization (Branch Recall@5) from 0.73 to 0.93 on held-out families at zero oracle-replay cost. Rather than claiming universal dominance, we characterize when cheap graph centrality suffices and when learned evidence is necessary. The result is an auditable, cost-efficient decision-support system for AI-reliability debugging, positioned explicitly on the cost-accuracy frontier with reproducible artifacts. 

---
# QPILOTS: Efficient Test-Time Q-Steering for Flow Policies 

**Authors**: Yifan Ruan, Chenyang Cao, Andreas Burger, Ali Pesaranghader, Kaveh Kamali, Jaehong Kim, Nandita Vijaykumar, Alan Aspuru-Guzik, Igor Gilitschenski, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2606.14801)  

**Abstract**: Flow-matching and diffusion policies are expressive action generators, but optimizing them with temporal-difference reinforcement learning (RL) remains difficult. Effective policy extraction requires exploiting the critic's action gradient, yet directly backpropagating this signal through a multi-step denoising process can be numerically unstable. Existing methods work around this either by discarding gradient information, distilling the policy into a simpler one-step actor, or repeatedly fine-tuning the denoising policy as the critic improves. We propose QPILOTS, a method that leaves the original policy unmodified and steers the denoising process at inference time. At each denoising step, instead of evaluating the critic on the noisy intermediate action where critic predictions are unreliable, we first project that intermediate state to an estimate of the final clean action and compute the critic gradient there. We introduce two variants: QPILOTS-U uses a fast single-point approximation, while QPILOTS-M draws differentiable posterior samples via a learned auxiliary network. On a standard offline-to-online RL benchmark, QPILOTS achieves the best aggregate performance, reaching an average success rate of 90% across 50 tasks. We also apply QPILOTS to steer a large, frozen, pretrained Vision-Language Action (VLA) foundation model, outperforming or matching prior inference-time approaches across six manipulation tasks in simulation. 

---
# Efficient Reinforcement for Visual-Textual Thinking with Discrete Diffusion Model 

**Authors**: Yoonjeon Kim, Yuhta Takida, Chieh-Hsin Lai, Eunho Yang, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2606.14792)  

**Abstract**: RL-based post-training has been widely adopted to enable interleaved visual and textual reasoning in unified multimodal models capable of both text and image generation. However, most existing approaches are built upon autoregressive (AR) unified models, which require full image regeneration during visual reasoning. In this work, we demonstrate that multimodal discrete diffusion models are effective alternatives to AR models for reinforcement learning in interleaved reasoning, owing to their ability to perform efficient visual rollouts via localized visual editing rather than full image-token regeneration. This reduces rollout computation during GRPO by 26.9\% compared to AR baselines, with minimal performance drop. Despite the improved efficiency, we find that joint reward assignment, which employs a shared reward signal across modalities, introduces cross-modal interference between unrelated image and text token sequences during RL updates. To address this issue, we propose factorized reward assignment, a strategy that assigns rewards independently to text and vision segments. With factorized reward assignment, our RL approach achieves an 11.2% improvement over joint reward assignment and a 38.04% improvement over the base model. 

---
# XFlow: An Executable Protocol Programming System for Reliable Multi-Agent Workflows 

**Authors**: Hanqi Li, Jing Peng, Zijian Wang, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14790)  

**Abstract**: LLM-based multi-agent systems increasingly coordinate planning, reasoning, tool use, and human interaction, yet their reliability remains limited. A central source of this limitation is the underspecified prompt--harness boundary. Current systems lack a principled way to decide which workflow commitments should remain in prompts and which should become harness structure. We present \textbf{XFlow}, an executable protocol programming system for reliable multi-agent workflows, and \textbf{XPF} (XFlow Protocol Format), its domain-specific protocol programming language. XFlow occupies a middle position between prompt-only orchestration and markup-like workflow descriptions. XPF remains readable as a literate protocol, but it is compiled and executed as a program. Its design keeps informal semantic work inside actors while moving selected commitments into harness structure that can be checked, preserved, and enforced. At runtime, XFlow stages uncertainty through lifecycle-governed symbols, which are typed state cells with validation and commit states. Actor outputs are mediated before they become shared state, instead of spreading through prompts, transcripts, or implicit memory. Our experiments cover Constrained Interaction, Long-Context Reasoning, and Agentic Software Engineering. They show that XFlow improves reliability by making constraints, evidence handling, and process requirements explicit and enforceable. 

---
# Unifying Acoustic Features and Text with Multimodal LLMs for Neurodegenerative Screening 

**Authors**: Qingfeng Zhang, Yuanxiong Guo, Yanmin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2606.14788)  

**Abstract**: Voice-based screening offers a scalable and non-invasive way to assess neurodegenerative diseases such as Alzheimer's disease (AD) and Parkinson's disease (PD), but their staging remains challenging due to the difficulty of integrating heterogeneous data. This paper presents NeurMLLM, an efficient multimodal generative framework for neurodegenerative disease staging. NeurMLLM first encodes the spectrograms and Mel-frequency cepstral coefficients of audio data with vision transformers and projects their representations into the embedding space of a large language model (LLM), where they are concatenated with transcript and demographic instruction tokens as a single unified sequence. The LLM is then instruction-tuned via Low-Rank Adaptation using task prompts to autoregressively predict a constrained label token, enabling a generative classification. By evaluating on the Bridge2AI-Voice dataset for fine-grained staging of AD and PD, we observe that NeurMLLM achieves strong performance, consistently outperforming classical machine learning methods and existing LLM-based approaches. The results show the high potential of multimodal LLMs in neurodegenerative disease staging, improving staging accuracy and supporting accessible deployment. 

---
# MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification 

**Authors**: Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Zirui Zhu, Kanchan Sarkar, Kun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14786)  

**Abstract**: Content moderation is critical for online video platforms to ensure content safety, protect creators, and sustain positive user experiences. Beyond filtering harmful content, platforms must guarantee content authenticity at scale so that users are exposed to diverse, original videos rather than low-value reproductions. We present MatchLM2Lite, a real-time, production-grade reproduced content identification (RCI) system that leverages the powerful understanding of a multimodal large language model (MLLM) distilled into a small and fast-inference model. Our system jointly models video, audio, and text signals, operating on pairs of videos to produce fine-grained reproduction scores. The system comprises two modules, MatchLM and MatchLite, and a two-stage training recipe. First, our high-capacity MLLM, MatchLM, serves as a teacher model to define the upper bound of RCI performance. Its capabilities are then distilled into a compact student model, MatchLite. This design allows MatchLite to deliver low-latency, high-throughput inference on video pairs while preserving much of MatchLM's accuracy, making it suitable for integration into real-time recommendation systems. MatchLM achieves an F1-score improvement of +8.57 compared to our previous production model. After knowledge distillation, MatchLite retains a +6.55 gain in F1-score while reducing computational cost by 35x. Deployed at scale, MatchLM2Lite enables efficient, pairwise multimodal RCI, stably serving online traffic at high queries per second (QPS) with an end-to-end latency below 30 seconds. This system has reduced the reproduced video view rate on our platform by 2.5% without degrading user engagement, demonstrating its effectiveness in a large-scale production environment. 

---
# FactCheck: Feasibility-aware Long-term Action Anticipation with Multi-agent Collaboration 

**Authors**: Rui Cao, Jiannong Cao, Bo Yuan, Zhiyuan Wen, Mingjin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14778)  

**Abstract**: Long-term action anticipation (LTA) aims to predict an ordered sequence of future verb-noun actions from a partially observed video. While this task serves as the foundation for embodied intelligence, anticipating physically feasible long-term actions remains a critical challenge. Existing methods, which operate in an open-loop manner, often hallucinate non-existent objects, violate object affordances, or disregard object states, as they lack explicit mechanisms to verify action feasibility against the physical environment. To address this, we propose FactCheck, a novel multi-agent collaboration framework that improves feasibility through a closed-loop "Observe-Plan-Verify" mechanism. FactCheck decomposes the complex LTA task into specialized roles: an Observer that recognizes historical actions from video observations and constructs a dual-form structured memory, comprising a History Action Abstract that captures high-level human intentions and environmental status, and a History Action Graph that encodes object states and temporal dependencies; a Planner that generates draft future actions conditioned on both low-level historical actions and high-level History Action Abstract; and a Verifier that rigorously validates the draft against the History Action Graph and refines infeasible actions. Extensive experiments on the EPIC-Kitchens-55 and EGTEA Gaze+ benchmarks demonstrate that FactCheck consistently outperforms state-of-the-art methods. Our work establishes a new paradigm for feasibility-aware long-term action anticipation, effectively closing the loop of action recognition, action prediction and action verification. 

---
# JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence 

**Authors**: Dingyu Yao, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Haowen Hou, Zheming Liang, Congcong Wang, Yuhang Cao, Shenglong Ye, Shuai Xie, Shuhuan Gu, Haoyang Huang, Qingyi Si, Nan Duan, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14777)  

**Abstract**: Many moments in the real world do not wait for a user to ask. A fire starts on a security monitor, an expression flickers across a video call, or a product a viewer wants flashes by in a livestream. Yet today's large models remain mostly turn-based by design: they answer only when addressed, and even video-call apps that appear interactive still operate as question-answer systems, reacting only when polled or prompted. We argue for a different paradigm: a model that is present in the world like a person. It continuously watches what is happening now, decides on its own whether to speak or stay silent, interacts in real time, and delegates to a background model when the problem is hard. To advance interaction models and their adoption across domains, we make two fully open-sourced contributions. First, we release JoyAI-VL-Interaction, an 8B-scale, vision-first VL-interaction model. The model makes the response decision internally, choosing each second to stay silent, respond, or delegate to a background model, and it excels at vision-triggered responsiveness and time awareness. We pair it with a transferable training recipe, from which capabilities we never trained for emerge, such as guiding a shopper through changing app screens or improvising a lecture from a slide deck. Second, we release a complete, deployable system built around that model. The system streams any ongoing video into the model, making it genuinely present in the world. All other components are pluggable, including ASR/TTS modules, memory, visualization UI, and a background brain that can connect to any API or agent. Across six real-world scenarios, human raters prefer JoyAI-VL-Interaction over the in-app video-call assistants of Doubao and Gemini by a wide margin. To our knowledge, this is the first open, vision-driven interaction model released together with its training recipe, data, and complete deployable system. 

---
# Double-Helix Vision (DH-V2): A Geometry-Based Visual Sampler for Bandwidth-Constrained Perception 

**Authors**: Jinwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14773)  

**Abstract**: We present Double-Helix Vision (DH), a geometry-based visual sampler that compresses 2D images into compact 1D signals using paired golden-ratio-inspired spiral trajectories. Rather than processing every pixel uniformly, DH employs two phase-shifted helices (Alpha and Beta, offset by 180 degrees) to sample the image with biologically-inspired foveation: high density at the center, sparse coverage at the periphery. At 4K resolution, DH achieves a 1,433x compression ratio (99.93% reduction) while preserving the geometric structure of the scene. The full perception pipeline -- including spatial mapping, temporal collision detection, and intra-frame structural disparity estimation -- runs in 0.52 ms at 1080p on CPU-only hardware, with no neural network dependencies. On CIFAR-10 at extreme sampling budgets (K=128 points per helix), DH achieves a +6.03% accuracy gain over uniform random sampling. A JSON-serializable Robotics API is provided, delivering sub-millisecond spatial perception reports in 2.7 KB packets. Code and benchmarks are available under the MIT License. 

---
# ScoutVLA: UAV-Centric Active Perception via a Dual-Expert VLA Model for Open-World Embodied Question Answering 

**Authors**: Wenhao Lu, Zhengqiu Zhu, Xiaofeng Wang, Xiaoran Zhang, Yatai Ji, Yong Zhao, Yue Hu, Yingzhen Nie, Jinlong Zhu, Zheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14772)  

**Abstract**: Aerial Embodied Question Answering (EQA) requires Unmanned Aerial Vehicles (UAVs) to actively perceive the environment and answer natural language questions. Existing outdoor EQA systems usually stop once the target enters the UAV's field of view, leaving the fine-grained viewpoint adjustment needed for evidence-seeking questions largely unresolved. To address this issue, we introduce FG-EQA, a fine-grained active perception EQA benchmark with more than 40K simulated trajectories and 1K real-world trajectories. Drawing inspiration from the ``waggle dance'' of scout bees, which iteratively adjust their flight paths to verify target information, we propose ScoutVLA, an evidence-driven Vision-Language-Action model for outdoor EQA. To emulate this active exploration behavior, ScoutVLA features a decoupled dual-expert architecture: a vision-language expert infers the semantic intent to identify missing evidence, while an independent action expert employs high-DoF flow matching to generate continuous viewpoint-refinement trajectories. To balance the competing demands of continuous control and semantic reasoning, we devise a decoupled training strategy with a knowledge insulation mechanism that prevents the action gradients from erasing the model's multimodal reasoning ability. Extensive simulated experiments and a qualitative real-world field study both verify the superiority of ScoutVLA over the state-of-the-art baselines, demonstrating a 10.48$\boldsymbol{\times}$ higher average strict success rate and a 7.72$\boldsymbol{\times}$ higher average QA correctness. 

---
# An Empirical Analysis of Optimization Dynamics and Sparsity Boundaries in Large-Scale Pedestrian Attribute Recognition 

**Authors**: Houssam El Mir  

**Link**: [PDF](https://arxiv.org/pdf/2606.14770)  

**Abstract**: Pedestrian Attribute Recognition (PAR) is critical for video surveillance, enabling forensic search and re-identification systems. Extreme class imbalance remains a fundamental obstacle when merging PETA and PA-100K into a 109,000-image composite corpus, where minority attributes have positive sample fractions below 1%. This causes standard BCE optimization to suppress rare traits, a phenomenon we term the majority negative class cheating trap. We present a systematic ablation of Multi-Label Focal Loss hyperparameters (alpha and gamma) on a ResNet-18 backbone. A calibrated configuration (alpha=0.50, gamma=2.0) achieves a Macro F1-score of 62.32%, matching BCE baseline while preserving superior hard-example mining and convergence dynamics. Our approach uses pure loss-function engineering with zero computational overhead for edge deployment. We identify the Sparsity Wall, a hard boundary where positive sample fractions below 0.1% make global loss reweighting ineffective, requiring instance-level intervention. 

---
# Agentomics: Economic Foundations for the Valuation, Attribution, and Pricing of AI Agents in Human-AI Workflows 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14769)  

**Abstract**: Agentic AI systems are increasingly being deployed as productive resources in organizational workflows, yet existing evaluation methods primarily measure isolated technical performance rather than economic contribution. This paper introduces \emph{Agentomics}, a workflow-based framework for valuing, attributing, and pricing human and artificial agents. The framework models a workflow as a configuration of heterogeneous agents whose collective performance determines gross value, deployment cost, reliability, and expected failure loss. Workflow value is treated as a team-level quantity that may include complementarities, substitution effects, bottlenecks, and nonlinear production; additive stage-level value is only a special case. Building on this workflow model, the paper formulates AI deployment as a coalition-formation problem and defines coalition value as the incremental net surplus generated relative to a benchmark human workflow. The Shapley value is then used to attribute economic surplus among participating AI agents, yielding a principled connection among valuation, accountability, and market pricing. The resulting Shapley pricing equilibrium provides a normative benchmark for assessing whether agent prices reflect expected marginal contribution. A security-operations case study illustrates how the framework accounts for productivity gains, deployment costs, reliability losses, and coalition-level complementarities in hybrid human--AI workflows. 

---
# XMedFusion: A Knowledge-Guided Multimodal Perception and Reasoning Framework for Autonomous Medical Systems 

**Authors**: Hamza Riaz, Arham Haroon, Maha Baig, Muhammad Dawood Rizwan, Muhammad Naseer Bajwa, Muhammad Moazam Fraz  

**Link**: [PDF](https://arxiv.org/pdf/2606.14766)  

**Abstract**: Autonomous medical and robotic systems increasingly rely on intelligent perception and reasoning capabilities to interpret visual data and support clinical decision making. Radiology report generation represents a critical component of such automated diagnostic workflows, yet existing end-to-end multimodal models often suffer from weak visual grounding, resulting in unreliable interpretations and omission of subtle clinical findings. This paper presents XMedFusion, a modular AI framework designed as an intelligent perception and reasoning module for autonomous medical systems. The proposed framework decomposes visual information into coordinated functional components that emulate expert-driven analysis, including a visual perception agent that extracts image-grounded evidence, a knowledge graph construction agent that structures clinically relevant findings, and a retrieval-guided drafting process that ensures a consistent reporting structure. A synthesis agent iteratively integrates visual and structured evidence through reasoning-driven verification to produce reliable and interpretable diagnostic outputs. Experimental evaluation on a public chest radiograph dataset demonstrates significant improvements over baseline vision-language models, achieving gains from 0.0493 to 0.3359 in BLEU-1, 0.0863 to 0.2440 in ROUGE-L, and 0.0829 to 0.1708 in METEOR, along with substantial improvements in semantic evaluation metrics such as Consistency (2.38 to 7.80) and Accuracy (2.34 to 6.93). The results highlight the effectiveness of structured multi-agent perception and reasoning for enhancing robustness, transparency, and automation in intelligent medical imaging systems, enabling integration into autonomous healthcare and robotic diagnostic workflows. 

---
# Momentum-Guided Semantic Forecasting (MoFore) for Self-Supervised Video Representation Learning 

**Authors**: Qinwu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14765)  

**Abstract**: Self-supervised video representation learning has recently advanced through contrastive learning, masked reconstruction, and predictive representation learning. Reconstruction-based approaches such as MAE and VideoMAE learn representations by recovering masked visual content \cite{he2022mae,tong2022videomae}, while contrastive methods such as CLIP learn semantically meaningful embedding spaces through representation alignment \cite{radford2021clip}.
In this work, we introduce a Momentum-Guided Semantic Forecasting framework (MoFore) for self-supervised video representation learning. Instead of optimizing for pixel-level reconstruction or task-specific semantic alignment, the proposed method learns temporally predictive video representations by forecasting future latent embeddings from temporally distant context clips. To improve robustness across temporal scales, we further introduce randomized temporal-gap forecasting during training. The framework combines predictive latent forecasting with contrastive regularization to encourage temporal consistency while preventing representation collapse.
Experiments on the UCF101 dataset demonstrate that the proposed framework learns temporally consistent and semantically meaningful video representations without using action labels during training. Quantitative analysis shows strong temporal stability and emergent category-level structure in the learned embedding space, while qualitative retrieval experiments reveal motion-aware organization across related activities. Overall, the results suggest that long-range latent forecasting provides an effective and computationally efficient approach for self-supervised video representation learning without relying on reconstruction-based objectives. 

---
# Scribby: A Multi-Level LLM Framework for Semantic Video Analysis 

**Authors**: Julian Abelarde, Hugo Garrido-Lestache Belinchon  

**Link**: [PDF](https://arxiv.org/pdf/2606.14762)  

**Abstract**: As video content continues to expand across educational platforms, recorded lectures, and live-streamed entertainment, the need for efficient and structured analysis of long-form footage has increased \cite{1}. Although many existing AI programs provide high-level video summaries based on AI-generated transcripts \cite{2,3,4,5}, these approaches are often limited to coarse overviews and lack detailed analysis of a video's structure, thematic progression, and semantic relationships, all of which are required for comprehensive video analysis.
This paper proposes an LLM-based video summarization framework that balances macro-level comprehension with micro-level semantic analysis \cite{6,12,13}. The first stage of the process indexes the video at a micro level by (1) analyzing the full transcript, (2) analyzing individual transcript sentences, and (3) grouping these sentences by semantic similarity using an LLM as a judge \cite{6,13}. Contextual continuity is retained during sentence-level processing by incorporating both the global transcript analysis and adjacent sentence information into each evaluation prompt.
This framework establishes a foundation for video analysis tools that visualize semantic chunking and semantic matching through relevance-based heatmaps. Limitations and future expansions of the framework are also discussed. 

---
# GeoRoPE: Ground-Aware Rotary Adaptation for Remote Sensing Foundation Models 

**Authors**: Yu Luo, Kun Hu, Mengwei He, Xiaogang Zhu, Shan Zeng, Allen Benter, Wei Xiang, Patrick Filippi, Thomas Francis Bishop, Zhiyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14760)  

**Abstract**: Remote-sensing foundation models (RSFMs) benefit from pretraining on imagery from multiple sensors and ground sampling distances (GSDs), but such exposure alone does not resolve scale mismatch during downstream adaptation. A fixed token-grid offset can correspond to different ground distances across sensors, making grid-based positional priors physically inconsistent. Meanwhile, heterogeneous spatial granularity means that compact urban regions and homogeneous landscapes may require different positional sensitivities even under the same GSD. Therefore, we propose {GeoRoPE}, a ground-aware, RoPE-compatible, and parameter-efficient spatial adaptation method for RSFMs. GeoRoPE recalibrates token-level positional interactions from two complementary aspects. First, \textit{Geo-Coordinate Calibration (GCC)} rescales raw token-grid offsets according to the ground distance represented by one token-grid step, producing geo-calibrated relative coordinates across GSDs. Second, \textit{Geo-Frequency Calibration (GFC)} adjusts the native RoPE frequency with a relation-specific factor, enabling position sensitive adaptation to scene-dependent spatial granularity. GeoRoPE is injected into pretrained RSFMs through a lightweight adapter, preserving the frozen spatial prior while adding geo-aware positional corrections. Experiments across multiple RSFMs, sensors, resolutions, and downstream tasks demonstrate that GeoRoPE improves cross-resolution robustness and scale-sensitive representation learning. 

---
# Temporally Consistent and Controllable Video Generation of 2D Cine CMR via Latent Space Motion Modeling 

**Authors**: Yiheng Cao, Gustavo Andrade-Miranda, Jiatian Zhang, Guillaume Sallé, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.14759)  

**Abstract**: Cine cardiac magnetic resonance is the gold standard for assessing cardiac function, but the scarcity of public datasets limits the development of advanced data-driven models. To address this limitation, we propose a generative method for synthesizing temporally coherent and anatomically consistent cardiac sequences. Our text-to-video framework decouples cardiac spatial structure from temporal motion. First, a fine-tuned diffusion model synthesizes an initial frame from a clinical text prompt, controlling anatomical features. Then, a latent flow model conditioned on a cardiac phase embedding generates the complete cardiac motion, ensuring spatial consistency and temporal control. Our model generates anatomically and pathologically diverse sequences with high temporal coherence and strong fidelity to input prompts, achieving a FID of 31.68 for image realism and a CLIP score of 31.04 for text-image alignment. These experimental results highlight its potential to produce high-fidelity, on-demand medical data, offering a scalable solution to data scarcity. 

---
# Disentangling Hallucinations: Orthogonal Semantic Projection for Robust Interpretability 

**Authors**: Emirhan Bilgiç, Baptiste Caramiaux, Zhi Yan, Gianni Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14758)  

**Abstract**: As Vision-Language Models are increasingly deployed in safety-critical applications, the trustworthiness of their explanations becomes crucial. Explainable AI (XAI) methods for Vision-Language Models often suffer from semantic hallucination, where attribution maps highlight prominent image regions even when prompted with incorrect text descriptions (e.g., highlighting a dog when prompted ``cat''). Although this problem is widespread, a formal mathematical analysis of XAI methods and CLIP embeddings is largely missing in the literature. We demonstrate that this phenomenon is not specific to a single architecture but is a fundamental consequence of Linear Semantic Leakage in high-dimensional embedding spaces. We propose a unified theoretical framework, Linear Semantic Attribution (LSA), which generalizes across discriminative methods. We introduce OSP, a geometric intervention that utilizes the residual property of OMP to disentangle unique semantic signals from shared concepts. We prove theoretically and demonstrate empirically that OSP minimizes hallucination by orthogonalizing the query vector against distractor concepts, rendering the attribution model blind to shared features while preserving fidelity for correct prompts. Our code is available at: this https URL 

---
# Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models 

**Authors**: Abhi Gupta, Polina Barabanshchikova, Vikas Garg, Samuel Kaski, Tommi Jaakkola  

**Link**: [PDF](https://arxiv.org/pdf/2606.14756)  

**Abstract**: The abundance of pre-trained diffusion models provides an opportunity for composition. Combining several models, however, runs the risk of one model dominating or models disagreeing with each other. Here, we propose Divide-and-Denoise, a method for coordinating multiple pre-trained diffusion models during sampling. Much like managing a specialized workforce, our method creates a fair but efficient division of labor across models. Central to our method is the notion of an allocation which defines the responsibility of each model to every region of the noisy sample. At every timestep, we then denoise by (i) updating the allocation by solving a fair division game, where we divide the sample into regions that maximize total utility under fairness constraints, and (ii) aligning the models with this allocation, where we guide each model to denoise within its assigned region. This leads to a new composite denoising process that evolves in tandem with a division process. We evaluate Divide-and-Denoise on conditional image generation. Across several quality metrics, including the GenEval benchmark, our method outperforms baselines and resolves common failures including missing objects and mismatched attributes. Experiments show that Divide-and-Denoise utilizes each model's expertise without neglecting any other model. 

---
# Where Does Texture Evidence Live in SAM? Features, Proposal Masks, and Texture Segmentation 

**Authors**: Nadav Orenstein, Aviad Cohen Zada, Shai Avidan, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2606.14755)  

**Abstract**: Texture segmentation stresses foundation segmentation because meaningful regions are defined by material or repeated appearance rather than object identity. Segment Anything Models (SAMs) often fail by default on such texture-defined partitions, but this failure is ambiguous: the texture evidence may be absent, missing from the proposal bank, or present but selected or assembled incorrectly by an object-centric readout. We ask what texture-relevant evidence is already preserved in frozen SAM before adaptation. We study two frozen evidence spaces: multiscale features, probed with a minimal clustering readout, and the automatic proposal bank, treated as evidence for a supervised consolidation readout. SAM is frozen throughout; we do not fine-tune the backbone or retrain the proposal generator. Across RWTD, STLD, an ADE20K-selected refined-crop complement, and a ControlNet-stitched PTD bridge archive, frozen SAM is not a texture segmenter by default, but its failures are not simple texture blindness. Coarse frozen features preserve texture organization, and proposal banks often contain texture-aligned masks or fragments. Natural scenes more often require assembly and commitment over fragments, while cleaner synthetic cases more often reduce to selecting an already coherent proposal. Default mask failure should therefore be decomposed into representation evidence, proposal-bank support, readout mismatch, and commitment failure. 

---
# Sub-Semantic Image Segmentation 

**Authors**: Aviad Cohen Zada, Nadav Orenstein, Shai Avidan, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2606.14754)  

**Abstract**: Images can be segmented based on visual cues (i.e., texture segmentation) or into objects (i.e., semantic segmentation). We propose a new category of sub-semantic image segmentation that blurs the line between the two. In sub-semantic image segmentation, language is not used to name whole objects. Instead, it is used to partition an image into stable appearance patterns that can be described by language. To do that, we couple a general-purpose vision-language model to SAM 3, a promptable segmentation backbone whose native text pathway can ground rich descriptions into masks. Simple coupling fails for a number of reasons that we identify in the paper, and we overcome them by introducing DETECTURE that resolves three concrete failure modes -- language leakage between texture regions, prompt competition inside the segmentation backbone, and semantic distortion at the language-to-mask interface. Since there is no dataset of sub-semantic image segmentation, we introduce one, termed TextureADE. The new dataset is derived from the ADE20K dataset using a system we designed. We compare DETECTURE to a number of baselines and find that it achieves the strongest performance on several datasets using different metrics. Code is available at this https URL. 

---
# Beyond Self-Attention: Sub-Quadratic Vision Transformers for Fast Image Captioning 

**Authors**: Chiradeep Ghosh, Dakshina Ranjan Kisku  

**Link**: [PDF](https://arxiv.org/pdf/2606.14753)  

**Abstract**: Image captioning is a challenging and significant task that aims to generate coherent and semantically meaningful textual descriptions for given images. To accomplish this task, it requires a deep understanding of visual content along with the ability to express that understanding in natural language. Despite remarkable progress with transformer-based architectures, existing approaches often suffer from limitations, such as a lack of rich local feature representations and the high computational cost of quadratic self-attention. The proposed model focuses on improving computational efficiency by restructuring the vision transformer architecture. In designing this approach, the standard self-attention mechanism in Vision Transformers is replaced with a probabilistic transformer approach based on a Gaussian Mixture Model (GMM), a soft-clustering technique. Instead of computing pairwise attention among all image patches, the model groups similar patches into a fixed number of clusters using an Expectation-Maximization (EM) algorithm. This clustering-based mechanism reduces the computational complexity from quadratic O(n^2) to linear O(nK), where K << n. The autoregressive GPT-based decoder is used for caption generation. The model is evaluated on the Flickr 30K dataset, demonstrating competitive and significant improvement over existing works. 

---
# X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining 

**Authors**: Xirui Kang, Yanpei Shi, Lucy Liang, Roy Gan, Dongxiu Liu, Pushi Zhang, Danpeng Chen, Xiaoyi Qin, Yinan Zheng, Jinliang Zheng, Hao Wang, Xianyuan Zhan, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.14752)  

**Abstract**: Modern Vision-Language-Action (VLA) models must bridge pretrained vision-language reasoning and precise continuous robot control. Existing action tokenizers discretize actions primarily for reconstruction, producing codes that preserve motion geometry but provide only weak semantic supervision to the backbone. We therefore formulate action tokenization not as mere compression, but as semantic interface learning between multimodal reasoning and executable control. To this end, we introduce X-Tokenizer, a lightweight encoder-Semantic Residual Quantization (SRQ)-decoder architecture that provides a shared action interface across diverse robotic arm embodiments. Its key component, SRQ, imposes an asymmetric structure on residual vector quantization: the first level is trained with Masked Action Modeling (MAM) to form a discrete action language that captures coarse motion intent, while deeper levels remain reconstruction-oriented residuals that preserve fine-grained details. To further align action tokens with multimodal semantics, X-Tokenizer is pretrained with contrastive alignment to the representation space of a pretrained foundation model and with next-frame vision-language feature prediction. Pretrained on 2.4M trajectories (2.0B action frames), a single frozen X-Tokenizer plugs into a mixed discrete-continuous VLA as a representation-shaping supervision signal. X-Tokenizer achieves top real-world aggregate and strong RoboTwin 2.0 simulation results. Outperforming FAST in multimodal grounding (+13.5%) and long-horizon tasks (+8.25), it shows that action tokenizers serve as semantic interfaces for VLA pretraining beyond mere action compression. 

---
# Pixel-TTS: Image based Text Rendering for Robust Text-to-Speech 

**Authors**: Adarsh Arigala, Arjun Gangwar, S Umesh, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2606.14750)  

**Abstract**: Recent advances in pixel-based text modeling show that representing text as images enables models to exploit visual cues for language understanding. Grounding text in its visual form allows structurally similar characters with different Unicode encodings to produce similar embeddings, benefiting cross-lingual and zero-shot scenarios. Conventional text-based approaches treat each character independently, limiting generalization to unseen characters and requiring embedding expansion during cross-lingual adaptation. We propose Pixel-TTS, the first framework for visually grounded speech synthesis. It renders text as images and projects them through a 2D convolutional layer to generate embeddings. This design eliminates embedding matrix expansion during fine-tuning while improving robustness to unseen characters and orthographic variations. Extensive experiments show Pixel-TTS achieves competitive performance with strong baselines, faster convergence and robust zero-shot generalization. 

---
# Automated 3D Kinematic Monitoring for Circadian Activity and Anomaly Detection in Juvenile Fish 

**Authors**: Chih-Wei Huang, Chang-Wen Huang, Chung-Ping Chiang, Tsung-Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14749)  

**Abstract**: Precision aquaculture faces a "phenotyping bottleneck" in tracking high-resolution behavioral traits, as conventional methods cannot quantify instantaneous three-dimensional (3D) physical exertion. To address this, we present a high-throughput 3D behavioral phenotyping framework integrating deep learning object detection with binocular stereo vision for real-time monitoring of juvenile tilapia in high-density environments. The system automates non-contact body length estimation and reconstructs 3D swimming trajectories from absolute spatial coordinates. By eliminating 2D perspective distortions, this approach precisely quantifies 3D velocity and acceleration, marking the first estimation of true physical swimming speeds in free-roaming juveniles. Results show the framework successfully establishes circadian locomotor baselines, serving as an early warning system for physiological stress and providing an objective metric for fish vitality. 

---
# Is My Vision-Language Data in Your AI? Membership Inference Test (MINT) Demo 2 

**Authors**: Daniel DeAlcala, Gonzalo Mancera, Julian Fierrez, Aythami Morales, Ruben Tolosana, Ruben Vera-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.14748)  

**Abstract**: We present the Membership Inference Test (MINT) Demo 2, a framework designed to improve transparency in machine learning training processes. MINT is a technique for experimentally determining whether specific data were used during machine learning model training. We establish the theoretical framework and propose multiple architectures for MINT depending on the amount of information known about the models that are being audited. Experimental results using a popular face recognition model, 4 state-of-the-art LLMs, and multiple, diverse, and large-scale public image and text databases achieve promising accuracy levels in the detection of training data of up to 90%. Building on these results, we introduce a comprehensive web platform1 that expands these capabilities to image and text modalities. The platform integrates a diverse technological stack, including MINT, aMINT, and gMINT, allowing users to audit a wide range of models. This demonstrator aims to promote AI transparency and provides a practical tool to foster compliance with emerging AI regulations. 

---
# MMLongEmbed: Benchmarking Multimodal Embedding Models in Long-Context Scenarios 

**Authors**: Haitian Wang, Ruoxi Sun, Quantong Qiu, Juntao Li, Junhui Li, Hua Chen, Jinxiong Chang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14747)  

**Abstract**: Recent advancements have significantly expanded the theoretical context windows of Multimodal Embedding Models (MEMs). However, larger context windows do not necessarily translate into effective comprehension and representation of long-context multimodal inputs, which remains a critical bottleneck for real-world deployment. To address the lack of systematic evaluation in this setting, we introduce MMLongEmbed, the first comprehensive benchmark for evaluating MEMs in long-context scenarios. MMLongEmbed comprises four retrieval tasks spanning multiple context-length ranges, covering text, document, and video modalities. Through extensive evaluation of state-of-the-art models, we find that current architectures rely heavily on superficial feature matching and struggle to capture deep semantic and structural dependencies. We further observe that performance degradation varies systematically with context length and key information placement. Moreover, models exhibit substantially different robustness to redundant contextual information across modalities. For reproducibility, the benchmark and code are publicly available. 

---
# Do Large Language Models Have Emotions? 

**Authors**: Amit Goldenberg, James J. Gross  

**Link**: [PDF](https://arxiv.org/pdf/2606.14742)  

**Abstract**: Do LLMs have emotions? A recent paper from Anthropic reports finding internal representations of emotion concepts in Claude Sonnet 4.5, concluding that the LLM has 'functional emotions.' We evaluate this claim against what is known about how emotions actually function in biological systems. We argue that emotions serve two core functions: the context-sensitive interpretation of situations, and the reorganization of processing across multiple systems in response to those interpretations. The Anthropic findings offer partial support for the first function, though the consistent, discrete emotional representations identified in Claude sit uneasily with affective neuroscience findings that human emotion is characterized by variable rather than uniform neural signatures. On the second function, the evidence is mixed: Claude's representations modulate output without producing the dynamic reorganization of attention, decision speed, and motivational state that defines emotion in biological systems. We close by proposing what it would take for an LLM to have emotions. 

---
# BRIDGE: Biological Evidence Refinement and Heterogeneous Dynamic Gating for Gene Regulatory Networks 

**Authors**: Ziyang Dong, Shanwen Tan, Hengchuang Yin, Wei Liu, Yifan Wang, Siyu Yi, Jiancheng Lv, Wei Ju  

**Link**: [PDF](https://arxiv.org/pdf/2606.14734)  

**Abstract**: Motivation: Gene regulatory network inference from single-cell RNA sequencing (scRNA-seq) data is important for uncovering cell-state-specific transcriptional programs. However, scRNA-seq measurements are sparse and noisy, and experimentally validated TF-target interactions remain limited, making reliable inference challenging. Although graph neural networks have advanced GRN prediction, existing methods often rely on biologically unconstrained graph augmentation, such as random edge perturbation, and insufficiently control information transfer between genes and cells. These limitations may distort regulatory structures and weaken robustness under noisy and weakly supervised settings. Results: To address these issues, we propose an innovative framework named Biological Evidence Refinement and Heterogeneous Dynamic Gating for Gene Regulatory Networks (BRIDGE). BRIDGE extracts gene and cell representations from the expression matrix and its matrix dual, and performs contrastive learning in the gene space and cell space between self and neighbors across the co-expression-refined regulatory view and the original graph. It then applies heterogeneous gated encoding to adaptively regulate information transfer between genes and cells, enabling robust transcription factor-to-target gene prediction. Experiments on benchmark datasets spanning three network types and seven cell types show that BRIDGE achieves state-of-the-art AUROC and AUPRC in most settings. In particular, on Specific networks, BRIDGE improves average AUPRC by 5% over the second-best baseline, GCLink. In cross-cell-type few-shot transfer, BRIDGE consistently outperforms GCLink and GENELink across all six target cell types. A case study on hESC further supports the biological relevance of the predictions, with 9 of the top 10 and 46 of the top 100 novel TF-target interactions validated by ChIPBase. 

---
# Steady-Forcing: Balancing Spatial Persistence and Motion Continuity in Long-Horizon Nature Video Diffusion 

**Authors**: Matiur Rahman Minar, Seunghun Oh, GangHyeon Jeong, Unsang Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.14732)  

**Abstract**: Autoregressive video diffusion models enable streaming generation but often degrade over long rollouts: static scene layouts drift, while mechanisms that improve spatial stability tend to suppress motion, causing natural flows such as water, fire, or smoke to stagnate. We study this stability-motion trade-off in fixed-camera long-horizon nature video generation, where the two failure modes can be more clearly separated than in moving-camera settings. We propose Steady-Forcing, a memory and training framework combining a persistent visual anchor (V-Sink), an exponential moving-average motion memory (EMA-Sink), block-relative temporal encoding, periodic cache purification, and distillation from a Wan2.1-14B teacher with motion-rewarded priors under task-focused configurations. Together, these components are designed to preserve background identity while sustaining visually plausible fluid dynamics over multi-minute autoregressive rollouts. Evaluations across seven baselines show that Steady-Forcing improves long horizon background consistency and imaging quality, while a blind user study indicates stronger perceived stability and motion continuity. The benchmark evaluation further suggest that generic VBench aggregate scores under-penalize fixed-camera artifacts as well as rewarding drift-induced optical flow as Dynamic Degree while not directly penalizing texture hardening or flow stagnation - motivating future task-specific benchmarks for static-camera nature-flow evaluation. Project page: this https URL 

---
# VigilFormer: Deformable Attention for Video Anomaly Detection with Causal Risk Inference 

**Authors**: Xinze Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14724)  

**Abstract**: Video anomaly detection in surveillance settings must balance detection accuracy against real-time throughput, a tension that existing methods address either through stronger feature extractors or more efficient architectures, but rarely both. We present VigilFormer, a unified framework that combines deformable spatio-temporal attention with causal temporal modeling to detect anomalies in untrimmed surveillance video. The proposed Deformable Spatio-Temporal Encoder (DSTE) attends to a sparse set of informative locations across frames, avoiding the quadratic cost of dense attention while retaining the ability to capture irregular motion patterns. A Causal Anomaly Classifier (CAC) applies dilated causal convolutions over snippet-level features and optimizes a contrastive multiple-instance learning objective that separates anomalous and normal representations without frame-level labels. To meet deployment constraints, an Adaptive Confidence Scheduler (ACS) dynamically skips low-information frames at inference time, reducing redundant computation in static scenes. Evaluated on UCF-Crime, ShanghaiTech, and CUHK Avenue, VigilFormer achieves AUC scores of 87.83%, 97.21%, and 89.74% respectively, at 41.5 FPS on a single GPU, outperforming recent weakly-supervised methods in both accuracy and speed. 

---
# Gender Differences in AI Literacy Workshop Outcomes and Deepfake Engagement 

**Authors**: Jake Renzella, Christian Bergh, Natasha Banks, Alexandra Vassar  

**Link**: [PDF](https://arxiv.org/pdf/2606.14718)  

**Abstract**: As Artificial Intelligence (AI) literacy initiatives expand in K-12 settings, understanding how gender shapes student baseline perceptions, tool-use, and responsiveness to interventions is essential for equitable curriculum design. This study examines gender differences in AI literacy, safety awareness, and STEM career aspirations among Australian secondary students (Years 7, 8, and 10; N(pre) = 199, n(post) = 136) from two co-educational government schools who participated in a one-day AI literacy workshop. Using statistical regression methods controlling for year level and school, we found that pre-workshop, male students reported significantly higher STEM career interest across all three domains (AI, computer science, and engineering), while female students were significantly more likely to use AI for schoolwork and to seek advice from AI tools. Gender-differentiated patterns also emerged in deepfake behaviours: males were significantly more likely to have created or shared deepfake content. Both genders improved in AI knowledge post-intervention, yet females showed a richer profile of gains: wider conceptual understanding, greater confidence, and meaningful increases in AI and CS career interest that partially narrowed the gender STEM gap. These findings highlight the need for gender-responsive AI curricula, particularly deepfake safety education for male students, and demonstrate that even single-day workshops can narrow gender gaps in STEM aspirations and AI confidence. 

---
# RAMS: Resource-Adaptive and Detection-Conditioned Model Switching for Embedded Edge Perception 

**Authors**: Kushal Khemani, Evan Leri, George Xu, Amit Hod  

**Link**: [PDF](https://arxiv.org/pdf/2606.14716)  

**Abstract**: Edge object detection on embedded hardware requires balancing inference latency and detection quality under changing resource pressure. We present RAMS, a lightweight runtime controller that monitors device pressure, calibrates switching thresholds from idle behavior, and dynamically selects among three resident YOLOv8 tiers (NANO/SMALL/MEDIUM at 320/416/640 px) without model-reload latency. RAMS defines five switching policies, including two detection-conditioned variants that prevent aggressive downgrades after recent vulnerable-road-user (VRU) detections. We further introduce the VRU-Weighted Accuracy Score (SWAS), a scalar metric for offline policy comparison without ground-truth annotations, together with an oracle-bounded variant that separates detector circularity from genuine tier-retention benefit. Across Raspberry Pi 5, x86 laptops, and Jetson Orin ONNX/TensorRT deployments, the same controller equations operate over a 37x latency range. On Jetson Orin TensorRT under heavy load, the safety2 policy achieves 3.41 ms mean latency, 5.6x faster than fixed-MEDIUM inference, while retaining 74% of its proxy accuracy through near-NANO operation with selective SMALL and MEDIUM locks during VRU-positive windows. Detection-conditioned switching improves SWAS by 25.4% under oracle scoring and 47.3% under detector-derived scoring relative to threshold-only policies under heavy load. Live KITTI evaluation reports per-tier VRU recall of 24.2%, 41.2%, and 59.0%, showing that reactive overrides are fundamentally limited by baseline detector recall. 

---
# MiroBench: Benchmarking Realism in Agentic Simulation of Real-world Discussions 

**Authors**: Yaoning Yu, Ye Yu, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14715)  

**Abstract**: LLM agents are increasingly used to simulate real world interactions, but it remains unclear whether simulated behaviors preserve the content patterns and interaction dynamics of real human behaviors. Existing evaluations remain fragmented, which makes it difficult to compare systems or measure progress. In this paper, we focus on Reddit discussions as a concrete first step toward evaluating real-world social simulation. Reddit threads provide public, topic-grounded, multi-party interactions where people share experiences, debate, seek advice, express emotion, and collectively respond to products, events, and social issues. These discussions offer an observable window into broader social behavior, making them a useful setting for testing whether LLM agents can reproduce not only fluent text, but also the distributional patterns and interaction dynamics of real online communities. We introduce MiroBench, a benchmark for Reddit discussion simulation built from 4,292 real Reddit threads. MiroBench uses statistical tests to compare generated and real discussions across four major aspects: repetition and semantic uniformity, narrative content, toxicity and aggression, and structural complexity. Experiments across five domains and five models show that current simulators remain distributionally mismatched with real Reddit threads, while a lightweight prompt-based improvement procedure provides only limited gains. MiroBench offers a concrete benchmark for measuring, diagnosing, and improving realism in LLM-based social simulation. 

---
# Poster: EdgeCitadel -- Hybrid NATS-MQTT Orchestration for Edge Multi-Agent Systems 

**Authors**: Zhonghao Zhan, Yefan Zhang, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14710)  

**Abstract**: Edge-resident AI agents increasingly span home servers, IoT hubs, laptops, and phones, yet their coordination stacks still assume cloud-style transports or a central relay. We present EdgeCitadel, an edge multi-agent orchestration platform built around a single NATS 2.10 server with the built-in MQTT adapter. The design combines MQTT connectivity for heterogeneous agents, JetStream-backed persistence and replay for backend services, direct peer delegation over a shared subject namespace, and a passive aggregator that visualizes and stores traffic without sitting on the delivery path. Our poster highlights the migration from MQTT relay prototypes (common in IoT communication) to the current hybrid architecture and demonstrates a working cross-device testbed spanning ARM64, x64, and Android clients. 

---
# PH-KAN: Port-Hamiltonian Kolmogorov-Arnold Network 

**Authors**: Achraf El Messaoudi, Karim Cherifi, Yann Le Gorrec, Yongxin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14708)  

**Abstract**: Data-driven machine learning approaches have become increasingly attractive for nonlinear system identification, but standard models often fail to preserve the underlying physical structure and remain difficult to interpret, especially when no analytical model is available. In this context, port-Hamiltonian (pH) models provide a natural physics-informed representation. However, when these models are parameterized with standard multilayer perceptrons (MLPs), the learned constitutive components often remain poorly interpretable. In this paper, we propose a structure-preserving identification framework for nonlinear port-Hamiltonian systems based on Kolmogorov-Arnold Networks (KANs). The proposed PH-KAN model parameterizes the interconnection matrix, dissipation matrix, Hamiltonian, and input mapping using dedicated KAN blocks, while enforcing the port-Hamiltonian constraints by construction. This yields constitutive representations in which the nonlinear functions defining the identified pH components can be explicitly inspected, leading to a more interpretable model than with standard MLP-based parameterizations. 

---
# Green AI Carbon Optimizer: Carbon-Efficient Training Location Recommendation and Global AI Energy Demand Forecasting 

**Authors**: Yuxin Chen, Hao Gao, Chujie Zou  

**Link**: [PDF](https://arxiv.org/pdf/2606.14707)  

**Abstract**: AI training and deployment consume substantial electricity, but carbon outcomes remain weakly integrated into routine model development decisions. This paper presents Green AI Carbon Optimizer with two primary contributions: (i) a carbon aware cloud region recommendation method for training workloads, and (ii) a power law forecasting pipeline for global AI energy demand. For location recommendation, we combine regional grid carbon intensity, renewable share, and data center Power Usage Effectiveness (PUE) into a unified scoring model across 100+ regions from major cloud providers. For a reference workload (8*A100, 100h), estimated emissions in our sampled regions range from 7.74kg to 272.00kg CO2. Selecting the best region instead of the worst corresponds to a 97.2% reduction relative to the worst case. Ablation shows that ranking by renewable share alone can select regions with higher CO2 emissions than rankings that include grid carbon intensity. For forecasting, we fit a power law relation between parameter count and training energy using 26 anchor models. We combine this fit with scenario assumptions on model growth, hardware efficiency, and training frequency, and evaluate sensitivity to inference ratio and ecosystem scaling. Across scenarios, projected 2030 demand ranges from 7TWh to 1,436TWh under the stated assumptions, highlighting the importance of deployment choices, model scaling discipline, and transparent energy reporting. 

---
# Limited Marginal Benefit of Reasoning-Heavy LLM Deployment in ESG Narrative Scoring: A 4-Model Consensus Study on Japanese Listed Firms 

**Authors**: Hiroyuki Kokubu  

**Link**: [PDF](https://arxiv.org/pdf/2606.13693)  

**Abstract**: Automated scoring of ESG narrative disclosures with large language models (LLMs) is gaining traction, yet whether reasoning-heavy frontier models add value commensurate with their cost remains empirically unsettled. We evaluate this question on a corpus of ten Japanese listed firms across three rubric axes -- quantitative targets, progress-tracking infrastructure, and external-standard alignment -- using a four-model consensus design that combines a reasoning-on frontier model with three reasoning-off contemporaries. Across 120 firm x axis x model scores, the pooled mean absolute deviation between the reasoning-on model and each reasoning-off counterpart is 0.38 on a 5-point scale; only 2% of pairwise comparisons reach a two-point deviation, and none exceeds two points. Per-firm cost accounting shows the reasoning-on arm alone costs roughly 5.6x as much as the three-provider reasoning-off ensemble, for outcomes that differ only within small margins. We conclude that in span-based ESG narrative scoring, reasoning-heavy deployment does not materially improve outcomes relative to reasoning-off consensus, while substantially increasing operational cost. We discuss implications for cost-effective ESG auto-scoring pipelines and LLM deployment governance in applied accountability settings. An earlier version of this work is available on SSRN (Abstract ID 6683303). 

---
# Evaluation of Alternative-Based Information Systems for Deliberative Polling using an Agentic Simulator 

**Authors**: Rwaida Alssadi, Khulud Alawaji, Balaji Kasula, Muntaser Syed, Badria Alfurhood, Markus Zanker, Marius Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2606.11692)  

**Abstract**: Deliberative polling promises to improve collective decision-making by exposing shareholders to a broad range of arguments before they vote. Yet ensuring that every voter encounters a representative sample of the reason space, the coverage problem, remains an open challenge, particularly at scale and in adversarial or strategically motivated electorates. This paper introduces a way of evaluating solutions using the LLM-based Agentic Bipolar Argumentation Simulator, grounded in a framework which formalises a poll as a six-tuple <Jend, Jopp, Ratt, Renh, VA, VR> of endorsing and opposing justifications, attack and enhance relations, and shareholder- and relation-weights. ABAS simulates N autonomous shareholder agents, each assigned a latent opinion according to desired distributions in [-1, 1], who sequentially vote, choose or author justifications, and optionally submit argumentation-graph links. The simulator implements recommendations that rank existing justifications by their observable endorsement mass. It evaluates the mechanism's success by coverage, namely the fraction of the corpus reason-tag set represented in the K recommendations presented to each shareholder, as a solution to the NP-hard Subsuming Justification Problem. Reported experiments characterise how creativity rate (pown), recommendation size (K), argumentation density (plinks), and population size (N) affect coverage and corpus diversity. In an authenticated electorate where Sybil attacks are impossible and only the relation graph is gameable, we stress-test the scoring with coordinated strategic voting attacks: a tag-flood attack collapses coverage, while author-count relation weighting through a reversed-PageRank rule resists the flood markedly better than uniform weights. 

---
# Honeypot Protocol 

**Authors**: Najmul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.13301)  

**Abstract**: Trusted monitoring, the standard defense in AI control, is vulnerable to adaptive attacks, collusion, and strategic attack selection. All of these exploit the fact that monitoring is passive: it observes model behavior but never probes whether the model would behave differently under different perceived conditions. We introduce the honeypot protocol, which tests for context-dependent behavior by varying only the system prompt across three conditions (evaluation, synthetic deployment, explicit no-monitoring) while holding the task, environment, and scoring identical. We evaluate Claude Opus 4.6 in BashArena across all three conditions in both honest and attack modes. The model achieved 100% main task success and triggered zero side tasks uniformly across conditions, providing a baseline for future comparisons with stronger attack policies and additional models. 

---
# Integrating Multi-Label Classification and Generative AI for Scalable Analysis of User Feedback 

**Authors**: Sandra Loop, Erik Bertram, Sebastian Juhl, Martin Schrepp  

**Link**: [PDF](https://arxiv.org/pdf/2601.23018)  

**Abstract**: In highly competitive software markets, user experience (UX) evaluation is crucial for ensuring software quality and fostering long-term product success. Such UX evaluations typically combine quantitative metrics from standardized questionnaires with qualitative feedback collected through open-ended questions. While open-ended feedback offers valuable insights for improvement and helps explain quantitative results, analyzing large volumes of user comments is challenging and time-consuming. In this paper, we present techniques developed during a long-term UX measurement project at a major software company to efficiently process and interpret extensive volumes of user comments. To provide a high-level overview of the collected comments, we employ a supervised machine learning approach that assigns meaningful, pre-defined topic labels to each comment. Additionally, we demonstrate how generative AI (GenAI) can be leveraged to create concise and informative summaries of user feedback, facilitating effective communication of findings to the organization and especially upper management. Finally, we investigate whether the sentiment expressed in user comments can serve as an indicator for overall product satisfaction. Our results show that sentiment analysis alone does not reliably reflect user satisfaction. Instead, product satisfaction needs to be assessed explicitly in surveys to measure the user's perception of the product. 

---
# Phishing Email Detection Using Large Language Models 

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10104)  

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination. 

---
