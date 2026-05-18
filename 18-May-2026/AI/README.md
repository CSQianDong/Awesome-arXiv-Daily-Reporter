# Prospective multi-pathogen disease forecasting using autonomous LLM-guided tree search 

**Authors**: Sarah Martinson, Michael P. Brenner, Martyna Plomecka, Brian P. Williams, Nicholas G. Reich, Zahra Shamsi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16238)  

**Abstract**: Probabilistic forecasting of infectious diseases is crucial for public health but relies on labor-intensive manual model curation by expert modeling teams. This bespoke development bottlenecks scalability to granular geographic resolutions or emerging pathogens. Here, we present an autonomous system using Large Language Model (LLM)-guided tree search to iteratively generate, evaluate, and optimize executable forecasting software. In a fully prospective, real-time evaluation during the 2025-2026 US respiratory season, the system autonomously discovered methodologically diverse models for influenza, COVID-19, and respiratory syncytial virus (RSV). Aggregating these machine-generated models yielded an ensemble that consistently matched or outperformed the gold-standard, human-curated Centers for Disease Control and Prevention (CDC) hub ensembles out-of-sample. The system successfully navigated data-scarce "cold start" scenarios for RSV. Moreover, controlled retrospective ablations revealed that optimizing log-scale distance metrics prevents reward hacking, while an automated judge-in-the-loop ensures structural fidelity to complex scientific theories. By autonomously translating epidemiological theory into accurate, transparent code, this framework overcomes the modeling labor bottleneck, enabling rapid deployment of expert-level disease forecasting at unprecedented scales. 

---
# FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast 

**Authors**: Igor Bogdanov, Chung-Horng Lung, Thomas Kunz, Jie Gao, Adrian Taylor, Marzia Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16233)  

**Abstract**: Can LLM agents improve decision-making through self-generated memory without gradient updates? We propose FORGE (Failure-Optimized Reflective Graduation and Evolution), a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents. FORGE wraps a Reflexion-style inner loop, where a dedicated reflection agent (using the same underlying LLM, no distillation from a stronger model) converts failed trajectories into reusable knowledge artifacts: textual heuristics (Rules), few-shot demonstrations (Examples), or both (Mixed), with an outer loop that propagates the best-performing instance's memory to the population between stages and freezes converged instances via a graduation criterion. We evaluate on CybORG CAGE-2, a stochastic network-defense POMDP at a 30-step horizon against the B-line attacker, where all four tested LLM families (Gemini-2.5-Flash-Lite, Grok-4-Fast, Llama-4-Maverick, Qwen3-235B) exhibit strongly negative, heavy-tailed zero-shot rewards. Compared against both a zero-shot baseline and a Reflexion baseline (isolated single-stream learning), FORGE improves average evaluation return by 1.7-7.7$\times$ over zero-shot and by 29-72% over Reflexion in all 12 model-representation conditions, reducing major-failure rates (below $-100$) to as low as $\sim$1%. We find that (1) population broadcast is critical mechanism, with a no-graduation ablation confirming that broadcast carries the performance gains while graduation primarily saves compute; (2) Examples achieves the strongest returns for three of four models, Rules offers the best cost-reliability profile with $\sim$40% fewer tokens; and (3) weaker baseline models benefit disproportionately, suggesting FORGE may mitigate capability gaps rather than amplify strong models. All evidence is confined to CAGE-2 B-line; cross-family findings are directional evidence. 

---
# Fully Open Meditron: An Auditable Pipeline for Clinical LLMs 

**Authors**: Xavier Theimer-Lienhard, Mushtaha El-Amin, Fay Elhassan, Sahaj Vaidya, Victor Cartier-Negadi, David Sasu, Lars Klein, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2605.16215)  

**Abstract**: Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, and generation pipelines that determine model behavior. Fully Open (FO) models, which expose the complete training stack end-to-end, do not currently exist in medicine. We introduce Fully Open Meditron, the first fully open pipeline for building LLM-CDSS, comprising a clinician-audited training corpus, a reproducible data construction and training framework, and a use-aligned evaluation protocol. The corpus unifies eight public medical QA datasets into a normalized conversational format and expands coverage with three clinician-vetted synthetic extensions: exam-style QA, guideline-grounded QA derived from 46,469 clinical practice guidelines, and clinical vignettes. The pipeline enforces system-wide decontamination, gold-label resampling of teacher generations, and end-to-end validation by a four-physician panel. We evaluate using an LLM-as-a-judge protocol over expert-written clinical vignettes, calibrated against 204 human raters. We apply the recipe to five FO base models (Apertus-70B/8B-Instruct, OLMo-2-32B-SFT, EuroLLM-22B/9B-Instruct). All MeditronFO variants are preferred over their bases. Apertus-70B-MeditronFO improves +6.6 points over its base (47.2% to 53.8%) on aggregate medical benchmarks, establishing a new FO SoTA. Gemma-3-27B-MeditronFO is preferred over MedGemma in 58.6% of LLM-as-a-judge comparisons and outperforms it on HealthBench (58% vs 55.9%). These results show that fully open pipelines can achieve state-of-the-art domain-specific performance without sacrificing auditability or reproducibility. 

---
# Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most 

**Authors**: Tahreem Yasir, Wenbo Li, Sam Gilson, Sutapa Dey Tithi, Xiaoyi Tian, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2605.16207)  

**Abstract**: Effective tutoring requires distinguishing optimal, valid but suboptimal, and incorrect student solutions, a distinction central to intelligent tutoring systems (ITS) but untested for LLM-based tutors. As LLMs are increasingly explored as conversational complements to ITS, evaluating their diagnostic precision is essential. We present a benchmark of seven LLM feedback agents in propositional logic using knowledge-graph-derived ground truth across 10,836 solution--feedback pairs and three feedback conditions. Models achieved near-ceiling performance on optimal steps but systematically over-rejected valid but suboptimal reasoning and over-validated incorrect solutions, precisely where adaptive tutoring matters most. These failures persisted across models regardless of solution context, suggesting architectural rather than informational limits. Moreover, accurate diagnosis did not reliably produce pedagogically actionable feedback, revealing a gap between diagnostic judgment and instructional effectiveness. Our findings suggest that LLMs are better suited for hybrid architectures where KG-grounded models handle diagnosis while LLMs support open-ended scaffolding and dialogue. 

---
# Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP 

**Authors**: Igor Bogdanov, Chung-Horng Lung, Thomas Kunz, Jie Gao, Adrian Taylor, Marzia Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16205)  

**Abstract**: Deploying compound LLM agents in adversarial, partially observable sequential environments requires navigating several design dimensions: (1) what the agent sees, (2) how it reasons, and (3) how tasks are decomposed across components. Yet practitioners lack guidance on which design choices improve performance versus merely increase inference costs. We present a controlled study of compound LLM agent design in CybORG CAGE-2, a cyber defense environment modeled as a Partially Observable Markov Decision Process (POMDP). Reward is non-positive, so all configurations operate in a failure-mitigation mode. Our evaluation spans five model families, six models, and twelve configurations (3,475 episodes) with token-level cost accounting. We vary context representation (raw observations vs. a deterministic state-tracking layer with compressed history), deliberation (self-questioning, self-critique, and self-improvement tools, with optional chain-of-thought prompting), and hierarchical decomposition (monolithic ReAct vs. delegation to specialized sub-agents). We find that: (1) Programmatic state abstraction delivers the largest returns per token spent (RPTS), improving mean return by up to 76% over raw observations. (2) Distributing deliberation tools across a hierarchy degrades performance relative to hierarchy alone for all five model families, reaching up to 3.4$\times$ worse mean return while using 1.8-2.7$\times$ more tokens. We call this destructive pattern a deliberation cascade. (3) Hierarchical decomposition without deliberation achieves the best absolute performance for most models, and context engineering is generally more cost-effective than deliberation. These findings suggest a design principle for structured adversarial POMDPs: invest in programmatic infrastructure and clean task decomposition rather than deeper per-agent reasoning, as these strategies can interfere when combined. 

---
# Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced AI Systems 

**Authors**: Parand A. Alamdari, Toryn Q. Klassen, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2605.16198)  

**Abstract**: We examine one particular dimension of AI governance: how to monitor and audit AI-enabled products and services throughout the AI development lifecycle, from pre-deployment testing to post-deployment auditing. Combining principles from formal methods with SoTA machine learning, we propose techniques that enable AI-enabled product and service developers, as well as third party AI developers and evaluators, to perform offline auditing and online (runtime) monitoring of product-specific (temporally extended) behavioral constraints such as safety constraints, norms, rules and regulations with respect to black-box advanced AI systems, notably LLMs. We further provide practical techniques for predictive monitoring, such as sampling-based methods, and we introduce intervening monitors that act at runtime to preempt and potentially mitigate predicted violations. Experimental results show that by exploiting the formal syntax and semantics of Linear Temporal Logic (LTL), our proposed auditing and monitoring techniques are superior to LLM baseline methods in detecting violations of temporally extended behavioral constraints; with our approach, even small-model labelers match or exceed frontier LLM judges. Our predictive and intervening monitors significantly reduce the violation rates of LLM-based agents while largely preserving task performance. We further show through controlled experiments that LLMs' temporal reasoning shows a pronounced degradation in accuracy with increasing event distance, number of constraints, and number of propositions. 

---
# An Algebraic Exposition of the Theory of Dyadic Morality 

**Authors**: Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2605.16153)  

**Abstract**: This paper provides an algebraic exposition of the theory of dyadic morality (TDM), a psychological model of moral judgment grounded in a simple two-node template: an intentional agent causing harm to a vulnerable patient. We formalize TDM using structural causal modeling (SCM) notation and identify three psychological operators (typecasting operator, completion operator, and valence-dependent inference mechanism) that extend standard SCM to capture how people compute moral judgments under constraints. We address scalability challenges arising from TDM's dyadic limitation, showing how moral cognition compresses multi-node scenarios through node collapse and sequential processing. Drawing on this algebraic framework, we demonstrate concrete applications to AI policy design: detecting conflicting obligations, structuring helpfulness policies to preserve user agency, and designing post-failure communication as causal interventions. Finally, we recommend scoped, contextual measurement of mind perception over universal averaging to operationalize the theory empirically. This algebraic formalization enables neurosymbolic AI systems to compute morality in a way that is both mathematically rigorous and faithful to human moral cognition. 

---
# Look Before You Leap: Autonomous Exploration for LLM Agents 

**Authors**: Ziang Ye, Wentao Shi, Yuxin Liu, Yu Wang, Zhengzhou Cai, Yaorui Shi, Qi Gu, Xunliang Cai, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16143)  

**Abstract**: Large language model based agents often fail in unfamiliar environments due to premature exploitation: a tendency to act on prior knowledge before acquiring sufficient environment-specific information. We identify autonomous exploration as a critical yet underexplored capability for building adaptive agents. To formalize and quantify this capability, we introduce Exploration Checkpoint Coverage, a verifiable metric that measures how broadly an agent discovers key states, objects, and affordances. Our systematic evaluation reveals that agents trained with standard task-oriented reinforcement learning consistently exhibit narrow and repetitive behaviors that impede downstream performance. To address this limitation, we develop a training strategy that interleaves task-execution rollouts and exploration rollouts, with each type of rollout optimized by its corresponding verifiable reward. Building on this training strategy, we propose the Explore-then-Act paradigm, which decouples information-gathering from task execution: agents first utilize an interaction budget to acquire grounded environmental knowledge, then leverage it for task resolution. Our results demonstrate that learning to systematically explore is imperative for building generalizable and real-world-ready agents. 

---
# Property-Guided LLM Program Synthesis for Planning 

**Authors**: Augusto B. Corrêa, André G. Pereira, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2605.16142)  

**Abstract**: LLMs have shown impressive success in program synthesis, discovering programs that surpass prior solutions. However, these approaches rely on simple numeric scores to signal program quality, such as the value of the solution or the number of passed tests. Because a score offers no guidance on why a program failed, the system must generate and evaluate many candidates hoping some succeed, increasing LLM inference and evaluation costs. We study a different approach: property-guided LLM program synthesis. Instead of scoring programs after evaluation, we check whether a candidate satisfies a formally defined property. When the property is violated, we stop the evaluation early and provide the LLM with a concrete counterexample showing exactly how the program failed. This feedback drastically reduces both the number of program generations and the evaluation cost, and can guide the LLM to generate stronger programs. We evaluate this approach on PDDL planning domains, asking the LLM to synthesize direct heuristic functions: every state reachable by strictly improving transitions has a strictly improving successor. A heuristic with this property leads hill-climbing algorithm directly to a goal state. A counterexample-guided repair loop generates one candidate program, checks the property over a training set, and returns the first case that violates the property. We evaluate our approach on ten planning domains with an out-of-distribution test set. The synthesized heuristics are effectively direct on virtually all test tasks, and compared to the best prior generation method our approach generates seven times fewer programs per domain on average, solves more tasks without using search, and requires several orders of magnitude less computation to evaluate candidates. Whenever a problem admits a verifiable property, property-guided LLM synthesis can reduce cost and improve program quality. 

---
# ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents 

**Authors**: Chinmay Savadikar, Mingyu Zhao, Yuanzheng Zhu, Han Li, Shuang Xie, Alberto Castelo, Tianfu Wu, Lingyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16116)  

**Abstract**: Developing and evaluating e-commerce web agents requires environments that preserve meaningful task structure while enabling controllable, reproducible, and scalable scientific comparison. Existing methodologies force a tradeoff: live storefronts provide realism but are non-stationary, difficult to inspect, and irreproducible, while hand-built sandbox benchmarks provide control but cover only a narrow range of layouts, catalogs, policies, and interaction patterns. We argue that the core bottleneck is methodological: the field lacks a scalable way to construct evaluation settings that are simultaneously realistic, diverse, controllable, inspectable, and reproducible. We introduce ShopGym, an integrated framework for realistic simulation and scalable benchmarking of e-commerce web agents. ShopGym is a framework for constructing e-commerce simulation environments and grounded benchmark tasks. Its simulation layer, ShopArena, converts live seed storefronts into self-contained sandbox shops through anonymized shop specifications and a staged, validated generation process. On top of these simulated storefronts, ShopGuru synthesizes benchmark tasks across seven skill categories, grounding each task in the shop's catalog, navigation structure, policies, and interaction affordances. Together, ShopArena and ShopGuru produce self-contained, resettable, inspectable, and stable evaluation artifacts that preserve structural properties and agent-evaluation signals relevant to shopping tasks. We validate the framework through graph-based structural analysis and agent-based behavioral evaluation with 224 generated tasks across six sandbox shops: three constructed with synthetic data and three with real data. Our results show that the synthetic shops preserve key structural properties of live storefronts, with agent performance on synthetic shops positively correlated with performance on live storefronts. 

---
# Sign-Separated Finite-Time Error Analysis of Q-Learning 

**Authors**: Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.16103)  

**Abstract**: This paper develops a sign-separated finite-time error analysis for constant step-size Q-learning. Starting from the switching-system representation, the error is decomposed into its componentwise negative and positive parts. The negative part is dominated by a lower comparison linear time-invariant (LTI) system associated with a fixed optimal policy, whereas the positive part is controlled by a linear switching system. The resulting bounds show that the negative-side LTI certificate is no slower than the positive-side switching certificate and may produce a faster exponential envelope. The analysis identifies a max-induced asymmetry in Q-learning error dynamics. This asymmetry is connected to overestimation: positive action-wise errors can be selected and propagated by the Bellman maximum, whereas negative errors admit an optimal-policy lower comparison. Finite-time bounds are provided for both deterministic and stochastic constant-step-size recursions. 

---
# Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law 

**Authors**: Parisa Kordjamshidi, Samer Aslan, Madhavan Seshadri, Leslie Barrett, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2605.16052)  

**Abstract**: Recent advances in large language models (LLMs) have significantly enhanced automated legal reasoning. Yet, it remains unclear whether their performance reflects genuine legal reasoning ability or artifacts of data contamination. We present a comprehensive empirical study of tax law reasoning approaches and implement a contamination detection protocol to rigorously assess LLM reliability. We show that performance can be inflated by contamination. Building on this analysis, we conduct a systematic evaluation, comparing monolithic LLMs with hybrid systems that translate statutory text into formal representations and delegate inference to symbolic solvers. We build a novel test suite designed to probe generalization to unseen documents via case and rule variations. Our findings indicate that legal reasoning is inherently compositional and that neuro-symbolic frameworks offer a more reliable and robust foundation for legal AI, as well as improved generalization to unobserved situations. 

---
# ScreenSearch: Uncertainty-Aware OS Exploration 

**Authors**: Michael Solodko, Justin Wagle  

**Link**: [PDF](https://arxiv.org/pdf/2605.16024)  

**Abstract**: Desktop GUI agents operate under partial observability: visually similar screens can correspond to different underlying workflow states, so locally plausible actions can lead to sharply different outcomes. We frame this as a problem of computer/OS state exploration, where effective behavior requires both expanding the reachable frontier and reducing ambiguity before committing. We present ScreenSearch, a system that combines structural screen retrieval and deduplication with an ambiguity-aware PUCT graph-bandit for large-scale desktop exploration. The retrieval layer converts UIA trees into location-aware structural features, indexes related screens through sparse token search and metadata filters, and maintains a shared deduplicated state graph across VM workers. On top of this graph, we define a scalable ambiguity signal based on matched-action outcome dispersion. If similar screens produce different next states under the same action signature, the state should be probed further rather than treated as resolved. We use this signal together with frontier rewards to drive large-scale exploration and replay-start policy evaluation over the shared graph. Across 11 desktop applications, ScreenSearch collects over 1M screenshots and over 30K deduplicated states, yielding large exploration corpora with substantial cross-application and within-application diversity. On a fixed replay-start slice, we observe a clear novelty--ambiguity trade-off: some policies reduce ambiguity quickly while discovering little frontier. Ambiguity reduction alone is therefore not a sufficient exploration objective. Appendix ablations show that stronger proposal priors can materially improve unique-state discovery during corpus building. These results suggest that state identity, proposal quality, and ambiguity-aware search all matter when deciding when to probe and when to commit. 

---
# Petri Net Induced Heuristic Search for Resource Constrained Scheduling 

**Authors**: Ido Lublin, Dor Atzmon, Izack Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15983)  

**Abstract**: We formulate the Resource-Constrained Project Scheduling Problem (RCPSP) as optimal search over the reachability graph of a Timed Transition Petri Net with Resources, using relative-delay tokens so that scheduling decisions correspond to transition firings in the induced state space. We solve the resulting problem with $A^*$ guided by a heuristic that combines Critical Path and resource-based lower bounds, and prove that it is consistent under our token-based time semantics. Experiments on the PSPLIB benchmarks show that the approach outperforms strong exact Mixed-Integer Linear Programming (MIP) baselines (SCIP, CBC) in both success rate and solve time. Per-instance analysis shows that heuristic search and MIP degrade along independent axes, resource tightness for $A^*$ and formulation size for MIP, with resource strength mediating which solver benefits from scale. 

---
# Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning 

**Authors**: Dillon Z. Chen, Till Hofmann, Toryn Q. Klassen, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2605.15975)  

**Abstract**: We tackle the challenge of building embodied AI agents that can reliably solve long-horizon planning problems. Imitation learning from demonstrations has shown itself to be effective in training robots to solve a diversity of complex tasks requiring fine motor control and manipulation over low-level (LL), continuous environments. Yet, it remains a difficult endeavour to generate long-horizon plans from imitation learning alone. In contrast, high-level (HL), symbolic abstractions facilitate efficient and interpretable long-horizon planning. We propose to combine the strengths of LL imitation learning for manipulation and control, and HL symbolic abstractions for long-horizon planning. We realise this idea via \emph{bilevel policies} of the form $(\pi^{\mathrm{hl}}, \pi^{\mathrm{ll}})$, consisting of a neural policy $\pi^{\mathrm{ll}}$ learned from LL demonstrations, and an HL symbolic policy $\pi^{\mathrm{hl}}$ that is constructed from symbolic abstractions of the LL demonstrations combined with inductive generalisation. We implement these ideas in the BISON system. Experiments on extended MetaWorld benchmarks demonstrate that BISON generalises to long horizons and problems with greater numbers of objects than those solved by VLA and end-to-end methods, and is more time and memory efficient in training and inference. Notably, when ignoring LL execution, BISON's HL policies can solve HL problems with 10,000 relevant objects in under a minute. Project page: this https URL 

---
# Deterministic Event-Graph Substrates as World Models for Counterfactual Reasoning 

**Authors**: Fabio Rovai  

**Link**: [PDF](https://arxiv.org/pdf/2605.15967)  

**Abstract**: We study event-graph substrates: a class of world models that represent agent state as an append-only log of typed RDF triples and answer counterfactual queries by forking the log under a structured intervention vocabulary. Substrates are inspectable at the triple level, support exact counterfactuals, and transfer across domains without learned components. We formalize the class, prove a duality between explanatory and counterfactual queries that reduces both to the same causal-ancestor traversal, and evaluate a 1,400-line CLEVRER-DSL interpreter atop a domain-agnostic substrate runtime at full CLEVRER validation scale (n=75,618). The substrate exceeds the NS-DR symbolic oracle on all four per-question categories (by 9.89, 20.26, 17.65, and 0.80 percentage points), and exceeds the parametric ALOE baseline on descriptive and explanatory while lagging on predictive and counterfactual. We also introduce twin-EventLog, a 500-specification Park-canonical Smallville counterfactual benchmark on which the substrate exceeds Llama-3.1-8B with full context by 18.80 points joint accuracy. 

---
# PAGER: Bridging the Semantic-Execution Gap in Point-Precise Geometric GUI Control 

**Authors**: Jingxuan Wei, Xi Bai, Shan Liu, Caijun Jia, Zheng Sun, Xinglong Xu, Siyuan Li, Linzhuang Sun, Bihui Yu, Conghui He, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.15963)  

**Abstract**: Large vision-language models have significantly advanced GUI agents, enabling executable interaction across web, mobile, and desktop interfaces. Yet these gains largely rely on a forgiving region-tolerant paradigm, where many nearby pixels inside the same component remain valid. Precise geometric construction breaks this assumption: actions must land on points in continuous canvas space rather than tolerant regions. Because geometric primitives carry ontological dependencies, a local coordinate error can induce cascading topological failures that distort downstream objects and invalidate the final construction. We identify this regime as precision-sensitive GUI tasks, requiring point-level accuracy, geometry-aware verification, and robustness to dependency-driven error propagation. To benchmark it, we introduce PAGE Bench, with 4,906 problems and over 224K process-supervised, pixel-level GUI actions. We further propose PAGER, a topology-aware agent that decomposes construction into dependency-structured planning and pixel-level execution. Pixel-grounded supervised tuning establishes executable action grammar, while precision-aligned reinforcement learning mitigates rollout-induced exposure bias through state-conditioned geometric feedback. Experiments reveal a pronounced Semantic-Execution Gap: general multimodal models can exceed 88% action type accuracy yet remain below 6% task success. PAGER closes this gap, delivering 4.1x higher task success than the strongest evaluated general baseline and raising step success rate from below 9% for GUI-specialized agents to over 62%, establishing a new state of the art for point-precise GUI control. 

---
# Imperfect World Models are Exploitable 

**Authors**: Logan Mondal Bhamidipaty, Esmeralda S. Whitammer, David Abel, Mykel J. Kochenderfer, Subramanian Ramamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2605.15960)  

**Abstract**: We propose a novel definition of model exploitation in reinforcement learning. Informally, a world model is exploitable if it implies that one policy should be strictly preferred over another while the environment's true transition model implies the reverse. We analogize our definition with a prior characterization of reward hacking but show that the associated proof of inevitability does not transfer to exploitation. To overcome this obstruction, we develop a general theory of reward hacking and model exploitation that proves that exploitation is essentially unavoidable on large policy sets and yields the corresponding claim for hacking as a special case. Unfortunately, we also find that the conditions that guarantee unhackability in finite policy sets have no counterpart that precludes exploitation. Consequently, we introduce a relaxed notion of exploitation and derive a safe horizon within which it can be avoided. Taken together, our results establish a formal bridge between reward hacking and model exploitation and elucidate the limits of safe planning in world models. 

---
# Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design 

**Authors**: Alberto Pepe, Chien-Yu Lin, Despoina Magka, Bilge Acun, Yannan Nellie Wu, Anton Protopopov, Carole-Jean Wu, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2605.15871)  

**Abstract**: Toward recursive self-improvement, we investigate LLM agents autonomously designing foundation models beyond standard Transformers. We introduce a dual-framework approach: AIRA-Compose for high-level architecture search, and AIRA-Design for low-level mechanistic implementation. AIRA-Compose uses 11 agents to explore fundamental computational primitives under a 24-hour budget. Agents evaluate million-parameter candidates, extrapolating top designs to 350M, 1B, and 3B scales. This yields 14 architectures across two families: AIRAformers (Transformer-based) and AIRAhybrids (Transformer-Mamba). Pre-trained at 1B scale, these consistently outperform Llama 3.2 and Composer-found baselines. On downstream tasks, AIRAformer-D and AIRAhybrid-D improve accuracy by 2.4% and 3.8% over Llama 3.2. Furthermore, AIRA-Compose finds models with highly efficient scaling frontiers: AIRAformer-C scales 54% and 71% faster than Llama 3.2 and Composer's best Transformer, while AIRAhybrid-C outscales Nemotron-2 by 23% and Composer's best hybrid by 37%. AIRA-Design tasks 20 agents with writing novel attention mechanisms for long-range dependencies and high-performing training scripts. On the Long Range Arena benchmark, agent-designed architectures reach within 2.3% and 2.6% of human state-of-the-art on document matching and text classification. On the Autoresearch benchmark, Greedy Opus 4.5 achieves 0.968 validation bits-per-byte under a fixed time budget, surpassing the published minimum. Together, these frameworks show AI agents can autonomously discover architectures and algorithmic optimizations matching or surpassing hand-designed baselines. This establishes a powerful paradigm for discovering next-generation foundation models, marking a clear step toward recursive self-improvement. 

---
# SaaS-Bench: Can Computer-Use Agents Leverage Real-World SaaS to Solve Professional Workflows? 

**Authors**: Kean Shi, Zihang Li, Tianyi Ma, Zengji Tu, Jialong Wu, Xinbo Xu, Qingyao Yang, Ruoyu Wu, Weichu Xie, Ming Wu, Jason Zeng, Michael Heinrich, Elvis Zhang, Liang Chen, Kuan Li, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15777)  

**Abstract**: Computer-Using Agents (CUAs) are rapidly extending large language models (LLMs) beyond text-based reasoning toward action execution in more complex environments, such as web browsers and graphical user interfaces (GUIs). However, existing web and GUI agent benchmarks often rely on simplified settings, isolated tasks, or short-horizon interactions, making it difficult to assess capabilities of agents in realistic professional workflows. Software-as-a-Service (SaaS) environments are a natural choice for CUA evaluation, as they host a large share of modern digital work and naturally involve dynamic system states, cross-application coordination, domain-specific knowledge, and long-horizon dependencies. To this end, we introduce SaaS-Bench, a benchmark built on 23 deployable SaaS systems across six professional domains, containing 106 tasks grounded in realistic work scenarios. These tasks require long-horizon execution, cover both text-only and multimodal settings, and are evaluated with weighted verification checkpoints that measure strict task completion and partial progress. Experiments show that representative LLM-based agents struggle on SaaS-Bench, with even the strongest model completing fewer than 4% of tasks end-to-end, exposing limitations in planning, state tracking, cross-application context maintenance, and error recovery. Code are available at this https URL for reproduction. 

---
# ALSO: Adversarial Online Strategy Optimization for Social Agents 

**Authors**: Xiang Li, Liping Yi, Mingze Kong, Min Zhang, Zhongxiang Dai, QingHua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15768)  

**Abstract**: Social simulation provides a compelling testbed for studying social intelligence, where agents interact through multi-turn dialogues under evolving contexts and strategically adapting opponents. Such environments are inherently non-stationary, requiring agents to dynamically adjust their strategies over time. However, most Large Language Model (LLM) based social agents rely on static personas, while existing approaches for enhancing social intelligence, such as offline reinforcement learning or external planners, are ill-suited to these settings, typically assuming stationarity and incurring substantial training overhead. To bridge this gap, we propose \textbf{ALSO} (\textbf{A}dversarial on\textbf{L}ine \textbf{S}trategy \textbf{O}ptimization), the first framework for online strategy optimization in multi-agent social simulation. ALSO advances social adaptation through two key contributions. (1) ALSO formulates multi-turn interaction as an adversarial bandit problem, where combinations of static personas and dynamic strategy instructions are treated as arms, providing a principled solution to non-stationarity without relying on environmental stability assumptions. (2) To predict rewards and generalize sparse feedback in multi-turn dialogues, ALSO introduces a lightweight neural surrogate to predict rewards from interaction histories, enabling sample-efficient exploration and continuous online adaptation. Experiments on the Sotopia benchmark demonstrate that ALSO consistently outperforms static baselines and existing optimization methods in dynamic environments, validating the effectiveness of adversarial online strategy optimization for building robust social agents. 

---
# Can We Trust AI-Inferred User States. A Psychometric Framework for Validating the Reliability of Users States Classification by LLMs in Operational Environments 

**Authors**: Izabella Krzeminska, Michal Butkiewicz, Ewa Komkowska  

**Link**: [PDF](https://arxiv.org/pdf/2605.15734)  

**Abstract**: The use of large language models to assess user states in conversational and adaptive systems is based on the assumption that the metrics used for such assessment are stable and interpretable at the level of individual scores. This paper empirically tests this assumption, focusing on the psychometric reliability of artificial intelligence (AI) measures of user states.
This study employed replication evaluation procedures to assess the repeatability of a broad set of metrics across three different bimodal large language models (GPT-4o audio, Gemini 2.0 Flash, Gemini 2.5 Flash). Analyses include both individual score reliability and aggregated reliability, allowing us to distinguish metrics potentially useful for real-time adaptation from those that retain their value only in aggregated analyses.
The results demonstrate that metric reliability cannot be considered a default property in interpretive domains. The lack of stability at the level of individual scores precludes the interpretation of such scores as indicators of user state in real-time adaptive systems, even if these metrics demonstrate stability after aggregation. At the same time, the study indicates that individually unstable metrics can retain analytical utility in post-hoc studies, identifying rules governing interactions and their relationships with user experience parameters such as satisfaction, trust, and engagement.
The main contribution of this work, besides quantifying the severity of the problem (only 31 of 213 metrics met the criteria), is the proposal of a replicable evaluation framework, enabling measurable evaluations of metric applicability. This approach supports more responsible AI design of adaptive systems, in which the interpretation of results requires explicit validation of reliability and monitoring for violations over time. 

---
# Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR 

**Authors**: Chanuk Lee, Sangwoo Park, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15726)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While increasing the number of rollouts alleviates this issue, such brute-force scaling is computationally expensive, and existing approaches that modify the optimization objective provide limited control over what is explored. In this work, we propose NudgeRL, a framework for structured and diversity-driven exploration in RLVR. Our approach introduces Strategy Nudging, which conditions each rollout on lightweight, strategy-level contexts to induce diverse reasoning trajectories without relying on expensive oracle supervision. To effectively learn from such structured exploration, we further propose a unified objective, which decomposes the reward signal into inter- and intra-context components and incorporates a distillation objective to transfer discovered behaviors back to the base policy. Empirically, NudgeRL outperforms standard GRPO with up to 8 times larger rollout budgets, while outperforming oracle-guided RL baseline on average across five challenging math benchmarks. These results demonstrate that structured, context-driven exploration can serve as an efficient and scalable alternative to both brute-force rollout scaling and feasibility-oriented methods based on privileged information. Our code is available at this https URL. 

---
# PRISM: Prompt Reliability via Iterative Simulation and Monitoring for Enterprise Conversational AI 

**Authors**: Keshava Chaitanya, Jahnavi Gundakaram  

**Link**: [PDF](https://arxiv.org/pdf/2605.15665)  

**Abstract**: Deploying large language model (LLM)-driven conversational agents in enterprise settings requires prompts that are simultaneously correct at launch and resilient to the non-deterministic behavioral drift that characterizes production LLM deployments. Existing prompt optimization frameworks address prompt quality as a one-time compile-time problem, leaving open the equally critical question of how to detect and repair prompt regressions caused by silent LLM behavior changes over time. We present PRISM (Prompt Reliability via Iterative Simulation and Monitoring), a closed-loop framework that treats prompt engineering as a continuous reliability engineering problem rather than a one-time authorship task. PRISM takes as input plain-language agent requirements, a set of configured tools and memory variables, and an initial draft prompt. It automatically generates test cases from requirements, simulates full multi-turn conversations against a platform-faithful LLM environment, evaluates pass/fail using an LLM-as-judge, diagnoses root causes of failures, and surgically repairs the prompt -- iterating until all tests pass. Critically, PRISM is designed to run on a scheduled basis (daily), treating LLM behavioral drift as a first-class reliability concern. We evaluate PRISM across 35 enterprise conversational agents over a three-week deployment period on the this http URL V3 platform. PRISM reduces median prompt authoring time from 2 days to under 30 minutes, achieves 99% production reliability across all evaluated agents, and successfully identifies and repairs production regressions caused by LLM behavioral drift within a 24-hour detection window. Our results suggest that continuous, simulation-driven prompt optimization is both tractable and necessary for reliable enterprise conversational AI at scale. 

---
# ColPackAgent: Agent-Skill-Guided Hard-Particle Monte Carlo Workflows for Colloidal Packing 

**Authors**: Lijie Ding, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2605.15625)  

**Abstract**: We introduce ColPackAgent, an agent framework that autonomously runs Monte Carlo simulations of colloidal packing through a Model Context Protocol (MCP) tool server and an agent skill, whether as a standalone agent or inside an existing agent system. By harnessing the MCP server and agent skill, ColPackAgent executes a structured workflow for colloidal packing simulations, which are central to studies of phase behavior, self-assembly, and materials design. Without dedicated simulation tools and workflow instructions, general-purpose Large Language Model (LLM) agents tend to describe such workflows rather than execute them reliably. The MCP server exposes a custom-built colpack Python package that wraps HOOMD-blue hard-particle Monte Carlo, and the skill encodes a four-stage workflow contract. ColPackAgent can carry out the workflow interactively with human feedback, autonomously from an end-to-end prompt, or as autoresearch following a provided program file. We demonstrate the system in different modes with several colloidal packing simulation examples such as cube particles in 3D, a binary system of disks and capsules in 2D, and the 2D hard-disk freezing transition using autoresearch. We also compare model performance on this workflow across a panel of LLMs with 17 stage-specific prompts. This benchmark provides a stage-level check of how reliably different models follow the setup, planning, and analysis workflow. Together, these results show that pairing a domain Python package with MCP tools and a portable agent skill provides a practical route for turning a simulation toolkit into an agent-assisted research workflow. 

---
# TopoEvo: A Topology-Aware Self-Evolving Multi-Agent Framework for Root Cause Analysis in Microservices 

**Authors**: Junle Wang, Xingchuang Liao, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15611)  

**Abstract**: Root cause analysis (RCA) in microservices is challenging due to (i) noisy and heterogeneous multimodal observability (metrics, logs, traces), (ii) cascading failure propagation that amplifies downstream symptoms, and (iii) non-stationary topology drift induced by autoscaling and rolling updates. Recent LLM-based RCA agents can generate tool-grounded explanations, yet they often remain topology-agnostic and suffer from \emph{symptom-amplification bias}, misattributing the root cause to salient downstream victims. We propose \textbf{TopoEvo}, a topology-aware self-evolving multi-agent framework that couples graph representation learning with structured, topology-constrained reasoning. TopoEvo first introduces \emph{Metric-orthogonal Multimodal Alignment} (MOMA), which decomposes metric embeddings into complementary subspaces and contrastively aligns logs and traces to reduce modality redundancy and sparsity, yielding stable node representations for graph encoding. It then applies \emph{Vector Quantization} (VQ) to discretize topology-enhanced states into auditable \emph{symptom tokens} with a symptom lexicon, enabling reliable retrieval and token-level evidence grounding. On top of these discrete topology cues, TopoEvo performs a multi-agent \emph{Hypothesis--Evidence--Test} (HET) workflow to explicitly verify propagation-consistent explanations and separate initiating anomalies from amplified downstream symptoms. Finally, a \emph{Self-Evolving Mechanism} refreshes hierarchical incident memory and performs conservative test-time adaptation with high-confidence pseudo-labels to maintain robustness under drift. 

---
# See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation 

**Authors**: Yuejia Li, Ke He, Junheng Li, Shutong Chen, Jingkang Xia, Zhiyue Su, Junchi Zhang, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.15585)  

**Abstract**: Large language models can generate executable code for educational animations, but the resulting renders often exhibit visual defects, including element overlap, misalignment, and broken animation continuity. These defects cannot be reliably detected from the code alone and become apparent only after execution. We formalize this problem as render-feedback-aware constrained code generation: given a natural language specification, the model must generate executable code whose rendered output satisfies structured quality criteria that can be evaluated only after rendering. To address this problem, we introduce OmniManim, a render-feedback-aware educational animation generation framework built around a shared scene state, explicit visual planning, structured post-render diagnostics, and localized repair. Within OmniManim, the Vision Agent is a task-specific visual planning module: it predicts sparse keyframe layouts with coarse-to-fine bounding-box denoising and optimizes an interpolation-aware objective to reduce intermediate-frame failures induced by downstream animation interpolation. We further construct two datasets, ManimLayout-1K and EduRequire-500, and provide a reproducible evaluation protocol covering executability, instructional quality, visual quality, and efficiency. On EduRequire-500, OmniManim improves measured render quality over both single-model baselines and existing multi-agent frameworks. Systematic ablation studies further verify that explicit visual planning, especially its coarse spatial prior, bounding-box refinement, and interpolation-aware optimization, is central to these gains. 

---
# STAR: A Stage-attributed Triage and Repair framework for RCA Agents in Microservices 

**Authors**: Junle Wang, Xingchuang Liao, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15581)  

**Abstract**: LLM-based root cause analysis (RCA) agents have recently emerged as a promising paradigm for incident diagnosis in microservice AIOps. However, their reliability remains fragile: an error in early evidence collection, hypothesis formulation, or causal analysis can propagate through the reasoning trace and eventually corrupt the final diagnosis. In this paper, we present \textbf{STAR}, a \emph{Stage-attributed Triage and Repair} framework for repairing erroneous RCA traces. STAR explicitly decomposes an RCA workflow into four structured stages, namely \emph{Evidence Package} (EP), \emph{Hypothesis Set} (HS), \emph{Analysis Structure} (AS), and \emph{Decision Report} (DR), and treats agent failure as a stage-localizable reasoning bug rather than a monolithic end-to-end error. Built on top of LangGraph, STAR performs stage-wise auditing, budget-aware \emph{Fast/Slow Routing}, \emph{decisive stage localization via counterfactual candidate evaluation}, and stage-specific patch-and-replay repair.
We evaluate STAR on a public large-scale benchmark and a real-world production dataset, using two RCA agent workflows and three foundation models. Experimental results show that STAR consistently improves both root cause localization and fault type classification over strong baselines. Moreover, STAR identifies the decisive faulty stage with high accuracy, repairs most initially incorrect traces within one or two replay rounds, and benefits substantially from both Fast/Slow Routing and counterfactual stage evaluation. These results suggest that explicitly modeling \emph{where} an RCA agent fails is an effective path toward reliable, debuggable, and self-repairing agentic RCA systems. 

---
# Position: Artificial Intelligence Needs Meta Intelligence -- the Case for Metacognitive AI 

**Authors**: Sergei Chuprov, Richard D. Lange, Leon Reznik, Paulo Shakarian, Raman Zatsarenko, Dmitrii Korobeinikov  

**Link**: [PDF](https://arxiv.org/pdf/2605.15567)  

**Abstract**: This position paper argues for metacognition as a general design principle for creating more accurate, secure, and efficient AI. The metacognitive solution involves systems monitoring their own states and judiciously allocating resources depending on each problem instance's difficulty or cost of mistakes. Drawing inspiration both from past work on resource-rational AI and from well-documented metacognitive strategies in psychology and cognitive science, we identify specific challenges in embedding these strategies into AI design and highlight open theoretical and implementation problems. We showcase these principles through a tangible example of improved learning efficiency, effectiveness, and security in a Federated Learning (FL) case study. We show how these principles can be translated into practice with a novel software framework developed specifically to allow the community to design, deploy, and experiment with metacognition-enabled AI applications. 

---
# DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding 

**Authors**: Yichao Liu, Huawen Shen, Liu Yu, Shiyu Liu, Zeyu Chen, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.15542)  

**Abstract**: GUI agents powered by Multimodal Large Language Models (MLLMs) have demonstrated impressive capability in understanding and executing user instructions. However, accurately grounding instruction-relevant elements from high-resolution screenshots cluttered with irrelevant UI components remains challenging for existing approaches. Inspired by how humans dynamically adjust their perceptual scope to locate task-related regions on complex screens, we propose DRS-GUI, a training-free dynamic region search framework for GUI grounding that can be seamlessly integrated into existing MLLMs. DRS-GUI introduces a lightweight UI Perceptor that performs three human-like perceptual actions (Focus, Shift, and Scatter) to progressively explore the interface and generate region proposals. To dynamically schedule these actions, we further design an Action Planner based on Monte Carlo Tree Search (MCTS). A region quality reward is employed to evaluate and select the highly instruction-relevant region, efficiently pruning redundant UI elements. Experiments demonstrate that DRS-GUI yields a 14\% improvement on ScreenSpot-Pro for general and GUI-specific MLLMs (Qwen2.5-VL-7B and UGround-V1-7B), significantly enhancing grounding performance and generalization. 

---
# RTL-BenchMT: Dynamic Maintenance of RTL Generation Benchmark Through Agent-Assisted Analysis and Revision 

**Authors**: Jing Wang, Shang Liu, Hangan Zhou, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.15537)  

**Abstract**: This paper introduces RTL-BenchMT, an agentic framework for dynamically maintaining RTL generation benchmarks. Large Language Models (LLMs) assisted automated RTL generation is one of the most important directions in EDA research. However, current RTL benchmarks face two critical challenges: (1) flawed cases in the benchmarks and (2) overfitting to the benchmarks. Both challenges are difficult to resolve purely by manual engineering effort. To address these issues and systematically reduce human maintenance costs, we propose an automated agentic framework, RTL-BenchMT. RTL-BenchMT focuses on two key applications: (1) automatically identifying and revising flawed benchmark cases and (2) automatically detecting and updating overfitting cases. With the assistance of RTL-BenchMT, we conduct a thorough, in-depth analysis of flawed and overfitting cases and produce a refined benchmark suite that will be open-sourced to the community. 

---
# CAPS: Cascaded Adaptive Pairwise Selection for Efficient Parallel Reasoning 

**Authors**: Fangzhou Lin, Shuo Xing, Peiran Li, Siyuan Yang, Qianwen Ge, Kazunori Yamada, Ziming Zhang, Haichong Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15513)  

**Abstract**: Parallel reasoning, where a generator samples many candidate solutions and an aggregator selects the best, is one of the most effective forms of test-time scaling in large language models, and pairwise self-verification has become its strongest aggregation primitive. Yet pairwise verification carries a heavy cost: each judgment reads two complete solutions in full, and existing methods perform tens of such judgments per problem regardless of whether the comparison is informative. We introduce CAPS (Cascaded Adaptive Pairwise Selection), an inference-only framework that allocates verifier compute non-uniformly along two orthogonal axes: an evidence axis that adapts how much of each candidate the judge sees, and a distribution axis that adapts how comparisons are spread across the pool. CAPS instantiates these into a four-stage cascade with an optional rescue subroutine, and admits a closed-form verifier-token cost in which the per-candidate marginal cost is roughly halved relative to uniform full-evidence schedules. On four self-verifying models (Qwen3-14B, GPT-OSS-20B, Qwen3-4B-Instruct/Thinking) and five reasoning benchmarks spanning code (LiveCodeBench-v5/v6, CodeContests) and math (AIME 2025, HMMT 2025), CAPS outperforms the leading pairwise verifier on 14 of 20 suites while using 25.4% of its verifier-token budget on code, and outperforms pointwise self-verification on all 20. The trade-off suites admit an interpretable diagnostic in terms of the verifier's accuracy at partial versus full evidence, providing a concrete pre-deployment check for cascade suitability. 

---
# X-SYNTH: Beyond Retrieval -- Enterprise Context Synthesis from Observed Human Attention 

**Authors**: Guruprasad Raghavan, George Nychis, Rohan Narayana Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2605.15505)  

**Abstract**: In enterprise operations, the context required for an AI agent task is scattered across systems of record, static information stores, and communication channels. What is stored is system state, a lossy representation of the work that actually happened [2, 52]. The prevailing approach [17, 31, 34, 36] retrieves by matching request content to what is stored; for narrow requests this works well. But synthesis quality depends on knowing what to surface and how to interpret it: knowledge specific to each organization, team, and individual [5, 57, 61], present in behavioral patterns, absent from any retrieval index. For complex agentic tasks it breaks down: True Lead Rate is low, False Lead Rate is high, and the model has no mechanism to improve. We present X-SYNTH, a framework for enterprise context synthesis grounded in human attention, the digitally observable interaction signatures of each worker, encoding not just what they did but the sequence in which they did it, along with implicit reward signals. Behavioral traces preceding positive outcomes are distinguishable from those that did not, without external labeling. X-SYNTH models each individual's behavioral baseline as a Digital Twin Signature (DTS) and selects among seven qualitatively distinct attention filters: Proportional, Inverse, Differential, Recurrent, Comparative, Sequential, and Collective, per individual and per query, to identify causally relevant activity signatures. A four-stage pipeline assembles ranked context grounded in behavioral patterns rather than query embeddings. On a sales lead identification task, a frontier model unaided achieves 9.5% True Lead Rate (TLR) with 90.5% False Lead Rate (FLR). Augmented with X-SYNTH, TLR rises to 61.9% (6.5x) while FLR falls to 18.8%. Enterprise context synthesis is not a retrieval problem. It is a relevance problem, and human attention is its most reliable ground truth. 

---
# From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates 

**Authors**: Ruobing Zuo, Hanrui Zhao, Gaolei He, Zhengfeng Yang, Jianlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15445)  

**Abstract**: Automated proving of polynomial inequalities is a fundamental challenge in automated mathematical reasoning, where rich algebraic structure and a rapidly growing certificate search space hinder scalability. Purely symbolic approaches provide strong guarantees but often scale poorly as the number of variables or the degree increases, due to expensive algebraic manipulations and rapidly growing intermediate expressions. In parallel, LLM-guided methods have made notable progress, particularly on competition-style inequalities with a small number of variables. To address the remaining scalability challenges, we propose NSPI, a neuro-symbolic framework that combines the complementary strengths of LLMs and symbolic computation for polynomial-inequality proving. Concretely, an LLM proposes a conjecture in the form of an approximate polynomial Sum-Of-Squares (SOS) decomposition; we refine it via symbolic computation to obtain an exact polynomial SOS representation, which directly proves the target inequality, and we further certify the proof in Lean, yielding an end-to-end pipeline from heuristic discovery to machine-checked proof. Experiments on challenging benchmarks involving polynomials with up to 10 variables demonstrate the effectiveness and scalability of the proposed method. 

---
# Beyond Partner Diversity: An Influence-Based Team Steering Framework for Zero-Shot Human-Machine Teaming 

**Authors**: Wei Sheng, Rohan Paleja  

**Link**: [PDF](https://arxiv.org/pdf/2605.15400)  

**Abstract**: While AI agents are rapidly advancing from isolated tools to interactive collaborators, data-driven human-machine teaming (HMT) methods remain costly in their reliance on human interaction data across domains, teammates, and team sizes. Zero-shot coordination (ZSC) addresses this bottleneck by simulating diverse partner populations to approximate how unseen partners might behave. However, partner coverage alone is insufficient as team settings scale and communication becomes degraded. To remedy this deficiency, we propose Influence-Based Team Steering (IBTS), a framework that uses influence shaping to incentivize agents to discover diverse, high-performing team interaction patterns and further steers ongoing trajectories toward stronger learned coordination modes. We assess IBTS on Overcooked-AI in both two-agent and three-agent settings, allowing us to test whether learned coordination structure transfers beyond dyadic interaction. Our evaluation includes simulated partners, synthetic partner-style variation, and, to our knowledge, the first 30-subject Overcooked-AI HMT study involving two real human teammates and one machine teammate. Across these evaluations, IBTS improves team performance against competing baselines, highlighting the need for scaled ZSC to combine sparse-reward coordination mechanisms with partner-variation coverage rather than relying on diversity alone. 

---
# Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute 

**Authors**: Eugene Koran, Yejun Yun, Samantha Tetef, Benjamin Arnav, Pablo Bernabeu-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2605.15377)  

**Abstract**: As AI systems are increasingly deployed in autonomous agentic settings at scale, it is important to ensure the actions they take are safe and aligned with user intent. Monitoring agent actions is a key safety mechanism, yet reliable monitors remain difficult to build and the scale of these systems makes human oversight impractical. We show that combining signals from diverse monitors into an ensemble improves detection of misaligned actions. We build 12 GPT-4.1-Mini monitors using both prompting and fine-tuning strategies. We evaluate them on coding tasks where candidate solutions pass standard tests but fail on adversarial inputs. In this setting, diverse ensembles outperform both individual monitors and homogeneous ensembles. Our best 3-monitor ensemble achieves 2.4x greater detection performance gain compared to an ensemble composed of three identical monitors, with the same ensemble performing strongly on an independent dataset. We contend that these results show that diversity - not scale - drives gains. The best ensembles combine strong individual performance with low correlation between monitors. Furthermore, fine-tuned monitors appear in every top-performing ensemble and maintain this advantage on out-of-distribution attack types, suggesting that fine-tuning enables detection capabilities that prompting alone does not elicit. These results support ensemble monitoring as a practical AI control strategy for safety gains at reasonable inference costs. 

---
# Belief Engine: Configurable and Inspectable Stance Dynamics in Multi-Agent LLM Deliberation 

**Authors**: Joshua C. Yang, Maurice Flechtner, Damian Dailisan, Michiel A. Bakker  

**Link**: [PDF](https://arxiv.org/pdf/2605.15343)  

**Abstract**: LLM-based agents are increasingly used to simulate deliberative interactions such as negotiation, conflict resolution, and multi-turn opinion exchange. Yet generated transcripts often do not reveal why an agent's stance changes: movement may reflect evidence uptake, anchoring, role drift, echoing, or changed prompt and retrieval context. We introduce the Belief Engine (BE), an auditable belief-update layer that treats "belief" as an evidential state over a proposition and exposes it as scalar stance. BE extracts arguments into structured memory and updates stance with a log-odds rule controlled by evidence uptake u and prior anchoring a. Across multiple base LLMs, parameter sweeps show that these controls reliably shape stance dynamics while preserving an evidence-level update trail. On DEBATE, a human deliberation dataset with pre/post opinions, BE best reconstructs participants whose final stance follows extracted evidence; stable and evidence-opposed cases instead point to anchoring or factors outside the extracted evidence stream. BE provides configurable infrastructure for studying evidence-grounded deliberation, where openness, commitment, convergence, and disagreement can be tied to explicit update assumptions rather than hidden prompt effects. 

---
# Zero-Shot Goal Recognition with Large Language Models 

**Authors**: Kin Max Piamolini Gusmão, Nathan Gavenski, Nir Oren, Felipe Meneguzzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15333)  

**Abstract**: Large language models have recently reached near-parity with classical planners on well-known planning domains, yet this competence relies on world-knowledge exploitation rather than genuine symbolic reasoning. Goal recognition is a complementary abductive task structurally better suited to LLM strengths: it consists of evaluating consistency with world knowledge rather than generating novel action sequences. This paper provides the first systematic zero-shot evaluation of frontier LLMs as goal recognisers on key classical PDDL benchmarks. Our results show that LLM competence on goal recognition is uneven: some models scale with evidence and approach landmark-based accuracy at full observations, while others remain anchored to world-knowledge priors regardless of how much evidence accumulates. Qualitative analysis of model reasoning traces reveals that this divergence reflects a fundamental difference in evidence integration rather than domain familiarity. These findings position goal recognition as a principled benchmark for the foundational planning knowledge of LLMs. 

---
# Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning 

**Authors**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15315)  

**Abstract**: LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one score and one transition matrix. We show that this formulation creates a modeling bottleneck: a single CRF transition prior must serve heterogeneous retention patterns, including contiguous semantic spans and sparse structural support lines. We propose LaMR (Latent Multi-Rubric), a structured pruning framework that decomposes code relevance into two interpretable quality dimensions, semantic evidence and dependency support, each modeled by a dedicated CRF with dimension-specific transition dynamics. A mixture-of-experts gating network dynamically weights the per-rubric emissions conditioned on the query, and a final CRF layer on the fused emissions produces the aggregate keep-or-prune decision. To supervise each dimension without additional annotation cost, we derive multi-rubric labels from the existing training corpus via AST-based program analysis, simultaneously denoising the teacher's binary labels. By effectively filtering distracting noise, LaMR frequently matches or even outperforms unpruned full-context baselines. Experiments on four benchmarks (SWE-Bench Verified, SWE-QA, LCC, LongCodeQA) show that LaMR wins 12 of 16 head-to-head multi-turn comparisons. It saves up to 31% more tokens on multi-turn agent tasks and improves Exact Match by up to +3.5 on single-turn tasks, while performance is frequently enhanced by denoising the context, and any remaining drops are marginal. 

---
# SMCEvolve: Principled Scientific Discovery via Sequential Monte Carlo Evolution 

**Authors**: Jiachen Jiang, Huminhao Zhu, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15308)  

**Abstract**: LLM-driven program evolution has emerged as a powerful tool for automated scientific discovery, yet existing frameworks offer no principled guide for designing their individual components and provide no guarantee that the search converges. We introduce SMCEvolve, which recasts program search as sampling from a reward-tilted target distribution and approximates it with a Sequential Monte Carlo (SMC) sampler. From this view, three core mechanisms emerge as principled components: adaptive parent resampling, mixture of mutation with acceptance, and automatic convergence control. We further provide a finite-sample complexity analysis that bounds the LLM-call budget required to reach a target approximation error. Across math, algorithm efficiency, symbolic regression, and end-to-end ML research benchmarks, SMCEvolve surpasses state-of-the-art evolving systems while using fewer LLM calls under self-determined termination. The code is available at this https URL. 

---
# Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution 

**Authors**: Han Li, Jinyu Tian, Rili Feng, Yuqiao Du, Chong Zheng, Chenyu Wang, Chenchen Liu, Shihao Li, Xinping Lei, Yifan Yao, Weihao Xie, Letian Zhu, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15301)  

**Abstract**: Large language models (LLMs) still struggle with the rigorous reasoning demands of hard competitive programming. While recent multi-agent frameworks attempt to bridge this reliability gap, they remain fundamentally stateless: they rely on static retrieval and discard the valuable problem-solving and debugging experience gained from previous tasks. To address this, we present Solvita, an agentic evolution framework that enables continuous learning without requiring weight updates to the underlying LLM. Solvita reorganizes problem-solving into a closed-loop system of strategy selection, program synthesis, certified supervision, and targeted hacking, executed by four specialized agents: Planner, Solver, Oracle, and Hacker. Crucially, each agent is paired with a trainable, graph-structured knowledge network. As the system operates, outcome signals, such as pass/fail verdicts, test certification quality, and adversarial vulnerabilities discovered by the Hacker, are recast as reinforcement learning updates to these network weights. This allows the agents to dynamically route future queries based on past successes and failures, effectively accumulating transferable reasoning experience over time. Evaluated across CodeContests, APPS, AetherCode, and live Codeforces rounds, Solvita establishes a new state-of-the-art among code-generation agents, outperforming existing multi-agent pipelines and nearly doubling the accuracy of single-pass baselines. 

---
# Verifiable Agentic Infrastructure: Proof-Derived Authorization for Sovereign AI Systems 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15228)  

**Abstract**: Modern cloud and enterprise systems rely on identity-centric authorization, assuming that callers possessing valid credentials are safe to execute commands. The emergence of autonomous AI agents invalidates this assumption: agents can generate syntactically valid but semantically unsafe actions, making standing privileges a significant operational risk. This risk becomes especially acute in sovereign AI systems, where autonomous agents may interact with cloud infrastructure, regulated data, financial workflows, and national-scale digital services. Governed mutation substrates reduce this risk by interposing on agent actions: agents submit intents, infrastructure evaluates context and policy, and execution is mediated. However, this shifts the trust boundary: how can the decision to authorize an intent be made verifiable, distributed, and replayable?
We introduce a Distributed Trust Framework (DTF), a verification framework for governed mutation systems that computes execution authority from structured, verifiable artifacts. DTF introduces a Justification Proof to encode the admissibility basis of an action, a consensus model for independent evaluation, an ephemeral Execution Identity derived from the approved proof, and an append-only Evidence Chain that preserves the authorization lifecycle. Under stated substrate assumptions, this architecture enforces a compact authorization invariant: no high-stakes execution without a proof object, no derived authority without consensus, and no valid mutation detached from evidence.
We define the model, instantiate it over an OpenKedge-based governed mutation substrate, and show how it maps onto cloud-native environments. By shifting authorization from standing identity to proof-derived authority, DTF provides an infrastructure foundation for making agentic execution governable, auditable, and bounded in sovereign AI deployments. 

---
# NIMO Controller: a self-driving laboratory orchestrator based on the Model Context Protocol 

**Authors**: Naruki Yoshikawa, Ryo Tamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.15227)  

**Abstract**: Self-driving laboratories (SDLs) have attracted increasing attention as a means of accelerating scientific discovery; however, developing SDL software remains technically demanding. To improve accessibility, orchestration software frameworks have been proposed to coordinate SDL components. Nevertheless, existing frameworks are primarily designed for human interaction and do not provide standardized interfaces suitable for AI agents. In this work, we propose an SDL software architecture based on the Model Context Protocol (MCP), in which all SDL functionalities are exposed through MCP servers. Following this design principle, we introduce an MCP-based SDL orchestrator, named NIMO Controller. It provides a visual programming interface automatically generated through MCP-based tool discovery, allowing human users to design experimental workflows without writing code. The same MCP backend can also be accessed by AI agents, providing a unified interface for both human users and AI agents. We demonstrate the proposed system through a case study on a color-matching SDL. The results validate the usability of the proposed MCP-based SDL architecture. 

---
# ICRL: Learning to Internalize Self-Critique with Reinforcement Learning 

**Authors**: Jianbo Lin, Xiaomin Yu, Yi Xin, Yifu Guo, Zhuosong Jiang, Zhongqi Yue, Weishi Wang, Heqing Zou, Chengwei Qin, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15224)  

**Abstract**: Large language model-based agents make mistakes, yet critique can often guide the same model toward correct behavior. However, when critique is removed, the model may fail again on the same query, indicating that it has not internalized the critique's guidance into its underlying capability. Meanwhile, a frozen critic cannot improve its feedback quality over time, limiting the potential for iterative self-improvement. To address this, we propose learning to internalize self-critique with reinforcement learning(ICRL), a novel framework that jointly trains a solver and a critic from a shared backbone to convert critique-induced success into unassisted solver ability. The critic is rewarded based on the solver's subsequent performance gain, incentivizing actionable feedback. To address the distribution shift between critique-conditioned and critique-free behavior, ICRL introduces a distribution-calibration re-weighting ratio that selectively transfers critique-guided improvements compatible with the solver's own prompt distribution. Additionally, a role-wise group advantage estimation stabilizes joint optimization across the two roles. Together, these mechanisms ensure that the solver learns to improve itself without external critique, rather than becoming dependent on critique-conditioned behavior. We evaluate ICRL on diverse benchmarks spanning agentic and mathematical reasoning tasks, using Qwen3-4B and Qwen3-8B as backbones. Results show consistent improvements, with average gains of 6.4 points over GRPO on agentic tasks, and 7.0 points on mathematical reasoning. Notably, the learned 8B critic is comparable to 32B critics while using substantially fewer tokens. The code is available at this https URL. 

---
# NOVA: Fundamental Limits of Knowledge Discovery Through AI 

**Authors**: Salman Avestimehr, Ken Duffy, Muriel Médard  

**Link**: [PDF](https://arxiv.org/pdf/2605.15219)  

**Abstract**: Can AI systems discover genuinely new knowledge through iterative self improvement, and if so, at what cost? We introduce the NOVA framework, which models the common ``generate, verify, accumulate, retrain'' loop as an adaptive sampling process over a knowledge space. We identify sufficient conditions under which accumulated genuine knowledge eventually covers a finite domain, and show how their violations produce distinct failure modes: contamination, forgetting, exploration failure, and acceptance failure. We then analyze imperfect verification and identify a contamination trap: as easy-to-find knowledge is exhausted, the model mass assigned to new valid artifacts shrinks, so even small false-positive rates can cause invalid artifacts to enter the knowledge base faster than genuine discoveries. We clarify that Good--Turing estimation is a local batch-diversity diagnostic, not an estimator of the historically undiscovered valid mass that governs long-term discovery. Under a separate tail-equivalence assumption relating the model's effective discovery distribution to a Zipf law with exponent $\alpha>1$, we prove that the cumulative generation cost required to obtain $D$ distinct genuine discoveries satisfies $R_{\mathrm{cum}}(D)=\Theta(c_{\mathrm{gen}}D^\alpha)$, where $c_{\mathrm{gen}}$ is the per-candidate generation cost. This scaling law quantifies asymptotic diminishing returns as the discovery frontier advances. Finally, we formalize human amplification through guidance, generation, and verification, explaining why expert input is most valuable near autonomous exploration barriers. 

---
# CAX-Agent: A Lightweight Agent Harness for Reliable APDL Automation 

**Authors**: Chenying Lin, Yichen Hai, Yi He, Ran Wang, Haiyan Qiang, Liang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15218)  

**Abstract**: Large language models deployed for MAPDL finite-element simulation face practical reliability challenges: without structured execution control, tool encapsulation, and fault recovery, outputs may be inconsistent and task failures are common. The Agent Harness paradigm addresses this by inserting domain-specific orchestration middleware that manages tool lifecycles, workflow state, and recovery escalation. This paper presents the architecture of CAX-Agent, a lightweight agent harness purpose-built for MAPDL automation, and empirically evaluates one of its core components -- the recovery this http URL-Agent organizes execution into three layers -- LLM service, agent harness, and solver backend -- with a recovery ladder that escalates from deterministic rule patching through model-driven regeneration to context enrichment and human intervention. We evaluate three recovery strategies (no_recovery, rule_only, and model_only) on 50 standard structural benchmarks with three repeated runs per strategy (450 case-runs total). Two independent human raters score task completion under blind conditions; inter-rater agreement is strong (quadratic weighted Cohen's kappa = 0.84, 96 percent of score pairs within one point). Model_only achieves the best completion rate (0.9267), task score (3.59/4), total score (9.16/10), and zero-intervention rate (0.84), outperforming rule_only (0.7733, 3.17/4, 7.03/10, 0.00) and no_recovery (0.6933, 2.74/4, 5.60/10, 0.00) with large effect sizes (Cliff's delta = 0.81-0.87). The benchmark uses deliberately simple geometries to isolate recovery-policy effects; we discuss the scope of these findings and directions for broader validation. 

---
# Fair outputs, Biased Internals: Causal Potency and Asymmetry of Latent Bias in LLMs for High-Stakes Decisions 

**Authors**: Jagdish Tripathy, Marcus Buckmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.15217)  

**Abstract**: Instruction-tuned language models exhibit behavioural fairness in high-stakes decisions while retaining biased associations in their internal representations. However, whether these suppressed representations can affect model outputs - and whether such causal potency is symmetric across demographic groups - remains unknown. We investigate the use of open-weight models for mortgage underwriting using matched applications that differ only in racially-associated names and reveal a critical disconnect: models show no output-level bias, yet retain and amplify demographic representations across model layers. Through activation steering and novel cross-layer interventions, we demonstrate that this suppressed information is decision-relevant: when reinjected at critical layers, it produces near-complete decision reversals. Critically, this latent bias is asymmetric - steering interventions affect decisions in one demographic direction, while producing minimal effects in reverse - and susceptible to adversarial prompt engineering and parameter-efficient fine-tuning. These findings demonstrate that behavioural audits focused on outputs are insufficient: fair outputs can mask exploitable internal biases. They also motivate dual-layer testing frameworks combining output evaluation with representational analysis for AI governance in high-stakes decisions. 

---
# SkillSmith: Compiling Agent Skills into Boundary-Guided Runtime Interfaces 

**Authors**: Duling Xu, Zheng Chen, Zaifeng Pan, Jiawei Guan, Dong Dong, Jialin Li, Bangzheng Pu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15215)  

**Abstract**: Recently, skills have been widely adopted in large language model (LLM)-based agent systems across various domains. In existing frameworks, skills are typically injected into the agent reasoning loop as contextual guidance once matched to a runtime task, enabling specialized task-solving capabilities. We find that this execution paradigm introduces two major sources of redundancy: irrelevant context injection and repeated skill-specific reasoning and planning. To this end, we propose SkillSmith, a boundary-first compiler-runtime framework that compiles skill packages offline into minimal executable interfaces. By extracting fine-grained operational boundaries from skills, SkillSmith enables agents to dynamically access and execute only the relevant components at runtime, thereby minimizing unnecessary context injection and redundant reasoning overhead. In the evaluation on SkillsBench benchmark, SkillSmith reduces solve-stage token usage by 57.44%, thinking iterations by 42.99%, solve time by 50.57% (2.02x faster), and token-proportional monetary cost by 57.44% compared with using raw-skills. Moreover, compiled artifacts produced by a stronger model can be reused by a smaller or more efficient runtime model, improving task accuracy in cases where raw skill interpretation fails. The source code and data are available at this https URL. 

---
# Does Theory of Mind Improvement Really Benefit Human-AI Interactions? Empirical Findings from Interactive Evaluations 

**Authors**: Nanxu Gong, Zixin Chen, Haotian Li, Zishu Zhao, Jianxun Lian, Huamin Qu, Yanjie Fu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.15205)  

**Abstract**: Improving the Theory of Mind (ToM) capability of Large Language Models (LLMs) is crucial for effective social interactions between these AI models and humans. However, the existing benchmarks often measure ToM capability improvement through story-reading, multiple-choice questions from a third-person perspective, while ignoring the first-person, dynamic, and open-ended nature of human-AI (HAI) interactions. To directly examine how ToM improvement techniques benefit HAI interactions, we first proposed the new paradigm of interactive ToM evaluation with both perspective and metric shifts. Next, following the paradigm, we conducted a systematic study of four representative ToM enhancement techniques using both four real-world datasets and a user study, covering both goal-oriented tasks (e.g., coding, math) and experience-oriented tasks (e.g., counseling). Our findings reveal that improvements on static benchmarks do not always translate to better performance in dynamic HAI interactions. This paper offers critical insights into ToM evaluation, showing the necessity of interaction-based assessments in developing next-generation, socially aware LLMs for HAI symbiosis. 

---
# SDOF: Taming the Alignment Tax in Multi-Agent Orchestration with State-Constrained Dispatch 

**Authors**: Zhantao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15204)  

**Abstract**: Multi-agent orchestration frameworks such as LangChain, LangGraph, and CrewAI route tasks through graph-based pipelines but do not enforce the stage constraints that govern real business processes. We present SDOF, a framework that treats multi-agent execution as a constrained state machine. SDOF operates through two primary defensive layers, implemented by three components: (1) an Online-RLHF Specialized Intent Router trained via Generative Reward Modeling (GRPO) and (2) a StateAwareDispatcher with GoalStage finite-automaton checks and precondition/postcondition SkillRegistry validation for auditable execution control. On a recruitment system backed by the Beisen iTalent platform (6000+ enterprises), 185 expert-curated scenarios trigger 1671 live API calls. Our GSPO-aligned 7B Intent Router achieves higher joint accuracy than zero-shot GPT-4o on this FSM-constrained adversarial routing benchmark (80.9% versus 48.9%). In end-to-end execution, SDOF reaches 86.5% task completion (95% confidence interval 80.8 to 90.7) and blocks all 22 operations in the injection, illegal HR subset. Under a broader message-level blocking audit, SDOF attains precision 100% and recall 88%, expert agreement kappa=0.94. A separate evaluation on 960 SGD-derived dialogues spanning 8 service domains surfaces 201 stage-order conflicts under our FSM mapping, 41 of which arise in the normal split. This arXiv version reports the current validated scope; extended multi-seed training comparisons and deeper workflow evaluations will be released in a subsequent update. 

---
# DeepSlide: From Artifacts to Presentation Delivery 

**Authors**: Ming Yang, Zhiwei Zhang, Jiahang Li, Haoseng Liu, Yuzheng Cai, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15202)  

**Abstract**: Presentations are a primary medium for scholarly communication, yet most AI slide generators optimize the artifact (a visually plausible deck) while under-optimizing the delivery process (pacing, narrative, and presentation preparation). We present DeepSlide, a human-in-the-loop multi-agent system that supports preparing the full presentation process, from requirement elicitation and time-budgeted narrative planning, to evidence-grounded slide--script generation, attention augmentation, and rehearsal support. DeepSlide integrates (i) a controllable logical-chain planner with per-node time budgets, (ii) a lightweight content-tree retriever for grounding, (iii) Markov-style sequential rendering with style inheritance, and (iv) sandboxed execution with minimal repair to ensure renderability. We further introduce a dual-scoreboard benchmark that cleanly separates static artifact quality from dynamic delivery excellence. Across 20 domains and diverse audience profiles, DeepSlide matches strong baselines on artifact quality while consistently achieving larger gains on delivery metrics, improving narrative flow, pacing precision, and slide--script synergy with clearer attention guidance. 

---
# IVGT: Implicit Visual Geometry Transformer for Neural Scene Representation 

**Authors**: Yuqi Wu, Tianyu Hu, Wenzhao Zheng, Yuanhui Huang, Haowen Sun, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16258)  

**Abstract**: Reconstructing coherent 3D geometry and appearance from unposed multi-view images is a fundamental yet challenging problem in computer vision. Most existing visual geometry foundation models predict explicit geometry by regressing pixel-aligned pointmaps, often suffering from redundancy and limited geometric continuity. We propose IVGT, an Implicit Visual Geometry Transformer that implicitly models continuous and coherent geometry from pose-free multi-view images. This formulation learns a continuous neural scene representation in a canonical coordinate system and supports continuous spatial queries at any 3D positions, retrieving local features to predict signed distance (SDF) values and colors using lightweight decoders. It allows direct extraction of continuous and coherent surface geometry, enabling rendering of RGB images, depth maps, and surface normal maps from arbitrary viewpoints. We train IVGT via multi-dataset joint optimization with 2D supervision and 3D geometric regularization. IVGT demonstrates generalization across scenes and achieves strong performance on various tasks, including mesh and point cloud reconstruction, novel view synthesis, depth and surface normal estimation, and camera pose estimation. 

---
# Designing Datacenter Power Delivery Hierarchies for the AI Era 

**Authors**: Grant Wilkins, Fiodar Kazhamiaka, Alok Gautam Kumbhare, Chaojie Zhang, Ricardo Bianchini  

**Link**: [PDF](https://arxiv.org/pdf/2605.16255)  

**Abstract**: Demand for AI accelerators is rapidly increasing rack power density, with projections approaching 1MW per deployment by 2027. This poses a major challenge for datacenter power delivery designers. As power densities increase, a datacenter designed for a different target density may strand power, i.e., may be unable to use all the power that its delivery hierarchy has provisioned. Designs must remain efficient over long datacenter lifetimes and multiple hardware generations. Power utilization is particularly important as grid power capacity is a scarce resource in the AI era.
Designing an efficient power delivery hierarchy for the long run is difficult because rack placement feasibility, workload impact, and cost depend jointly on electrical topology, deployment granularity, placement policy, power oversubscription, and workload mix. Moreover, each of these factors evolve over time, have inter-dependencies across multiple resource dimensions, and generally do not lend themselves to closed-form analysis.
To address this challenge, we develop a framework for evaluating datacenter power delivery designs using throughput, power, and cost metrics over realistic arrival, oversubscription, and decommissioning sequences. The framework combines projection models for GPU, compute, and storage deployments with operational factors grounded in production data from Microsoft Azure. Our results show that multi-resource stranding materially changes deployable capacity, effective capital expenditure, and delivered performance, and quantify how rising density from rack- and pod-scale AI systems shapes these outcomes. For AI datacenter design, the relevant planning objective is not installed megawatts, but deployable capacity over time. 

---
# A Generative AI Framework for Intelligent Utility Billing CO 2 Analytics and Sustainable Resource Optimisation 

**Authors**: Pavan Manjunath, Thomas Pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.16250)  

**Abstract**: Distribution utilities are now expected to deliver bills that customers can actually read attach a defensible carbon number to every kWh sold and schedule load against grid stress and emissions constraints We propose an end-to-end framework that unifies four production-grade capabilities under one architectural roof a generative-AI agent that drafts each customers natural-language billing statement from structured numeric inputs under a constrained decoding policy a transformer-based forecaster that supplies the day-ahead consumption estimate with calibrated quantile bands 

---
# AI-Mediated Communication Can Steer Collective Opinion 

**Authors**: Stratis Tsirtsis, Kai Rawal, Chris Russell, Brent Mittelstadt, Sandra Wachter  

**Link**: [PDF](https://arxiv.org/pdf/2605.16245)  

**Abstract**: Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) now polish users' posts on LinkedIn and provide context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social network, transforming the opinions they express and perceive. By analytically characterizing the equilibrium of this model and performing simulations on real social network data, we show that biases introduced by AI in human-to-human communication can be amplified through the network and shift collective opinion in their direction. In light of these findings, we investigate whether such biases are controllable by online platforms. We audit the "Explain this post" feature on X and find evidence of pro-life bias in Grok's outputs on abortion-related content, which we trace back to specific design choices. We conclude with a discussion of the broader implications of our findings in relation to ongoing legislative efforts in the European Union. 

---
# Offline Semantic Guidance for Efficient Vision-Language-Action Policy Distillation 

**Authors**: Jin Shi, Brady Zhang, Yishun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16241)  

**Abstract**: Billion-parameter Vision-Language-Action (VLA) policies have recently shown impressive performance in robotic manipulation, yet their size and inference cost remain major obstacles for real-time closed-loop control. We introduce \textbf{VLA-AD}, a distillation framework that uses a Vision-Language Model as an offline semantic supervisor to transfer large VLA teachers into lightweight student policies. Instead of relying only on low-level action imitation, VLA-AD augments teacher-provided 7-DoF action targets with high-level semantic guidance, including task phase anchors and multi-frame operating-direction descriptions. These auxiliary signals are used only during training: at test time, the student policy runs independently, with neither the VLA teacher nor the VLM required. We evaluate VLA-AD on three LIBERO benchmark suites. Using OpenVLA-7B as the teacher, our method produces a 158M-parameter student, yielding a $44\times$ reduction in model size while matching the teacher with only a $0.27\%$ average relative gap. The resulting policy runs at 12.5 Hz on an RTX 4090, achieving a $3.28\times$ inference speedup over OpenVLA-7B. We further show that the same semantic distillation pipeline generalizes to a different $\pi_{0.5}$-4B teacher, where the student outperforms the teacher on two suites and remains within $0.53\%$ on \texttt{libero\_goal}. Additional analysis indicates that phase-level supervision and multi-frame directional cues make the student less sensitive to noisy teacher actions, such as erroneous high-frequency gripper changes. Overall, VLA-AD demonstrates that offline semantic guidance from VLMs can substantially improve the efficiency, robustness, and deployability of VLA policy distillation. 

---
# Layer Equivalence Is Not a Property of Layers Alone: How You Test Redundancy Changes What You Find 

**Authors**: Gabriel Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.16234)  

**Abstract**: When researchers ask whether two transformer layers are "equivalent" for compression, they often conflate distinct tests. Replacement asks whether one layer's map can substitute for another's in place; interchange asks whether two layers approximately commute when their positions are swapped. Both are output-grounded swap-KL probes, but they need not agree: on pretrained transformers the protocol gap can change which layers look safe to prune by several-fold under the same evaluator, especially when replacement distances are high.
We measure both protocols across checkpoints and architectures. On a Pythia training trajectory (410M and 1.4B), the replacement-interchange gap grows from initialization to convergence. Under one matched WikiText-2 contract at 8B scale, Qwen3-8B enters a divergent regime: interchange-guided removal is several-fold safer than replacement-guided at the same layer budgets, while Llama-3.1-8B ties the two protocols for pruning cost even though interchange KL is lower, showing metric gaps need not map one-to-one to removal. Before layer removal or merging, score both swap-KLs on the target checkpoint; the diagnostic requires only unlabeled forward passes. 

---
# A Unified Generative-AI Framework for Smart Energy Infrastructure: Intelligent Gas Distribution, Utility Billing, Carbon Analytics, and Quantum-Inspired Optimisation 

**Authors**: Pavan Manjunath, Thomas pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.16232)  

**Abstract**: The accelerating convergence of smart metering, generative artificial intelligence, and quantum-inspired combinatorial optimisation is reshaping how energy utilities manage physical infrastructure, customer engagement, and environmental accountability 

---
# Evaluating Design Video Generation: Metrics for Compositional Fidelity 

**Authors**: Adrienne Deganutti, Dingning Cao, Jaejung Seol, Elad Hirsch, Purvanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2605.16223)  

**Abstract**: Generative video models are increasingly used in design animation tasks, yet no standardized evaluation framework exists for this domain. Unlike natural video generation, design animation imposes structured constraints: specific components shall animate with prescribed motion types, directions, speed and timing, while non-animated regions must remain stable and layout structure must be preserved. This paper provides a fully automated evaluation framework organized across four dimensions: layout fidelity, motion correctness, temporal quality, and content fidelity. This eliminates the reliance on subjective human evaluation and establishes a common basis for benchmarking progress in the field. 

---
# Argus: Evidence Assembly for Scalable Deep Research Agents 

**Authors**: Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16217)  

**Abstract**: Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens. 

---
# paper.json: A Coordination Convention for LLM-Agent-Actionable Papers 

**Authors**: Arquimedes Canedo  

**Link**: [PDF](https://arxiv.org/pdf/2605.16194)  

**Abstract**: LLM agents routinely serve as first (and sometimes only) readers of academic papers, skimming for sub-claims, extracting reproducibility steps, and generalizing scope. Standard prose papers produce recurring failures in this role: sub-claims that cannot be cited at sub-paper granularity, scope overextension beyond what the paper tests, and figure commands buried in codebases rather than the paper itself. We propose `this http URL`, a companion JSON file that travels with the PDF and addresses each failure with a lightweight convention: stable claim IDs (C1), an explicit does-not-claim list (C2), exact per-figure shell commands (C3), and stable definition IDs (C5). A fifth convention (C4) holds that minimum viable compliance, hand-written JSON alongside the PDF, is achievable in under an hour for a finished paper without touching the human-readable output. C1, C2, C3, and C5 are open invitations: an agent that reads a compliant paper and acts on it produces evidence for or against them. This paper is itself compliant: `uv run this http URL this http URL --against this http URL` passes. Repo: this https URL 

---
# Second-Order Multi-Level Variance Correction for Modality Competition in Multimodal Models 

**Authors**: Yishun Lu, Wes Armour  

**Link**: [PDF](https://arxiv.org/pdf/2605.16165)  

**Abstract**: Autoregressive next-token training offers a unified formulation for image generation and text understanding, but it also creates strong modality competition that destabilizes optimization and limits large-batch scaling. We show that first-order optimizers such as AdamW are vulnerable to cross-modality gradient heterogeneity, while second-order preconditioning, particularly SOAP, provides a more stable basis for multimodal alignment. Building on this insight, we propose \emph{ML-FOP-SOAP}, a second-order optimization framework with Multi-Level Variance Correction. Our Fisher-Orthogonal Projection suppresses variance-induced modality conflicts, reducing the trade-off between visual generation and textual understanding. To make this practical under large gradient accumulation, we introduce a hierarchical folding strategy that captures fine-grained variance with low micro-step overhead. Experiments on Janus and Emu3 show consistent gains across both modalities and stable training at batch size 8192. Compared with AdamW, our method improves sample efficiency by up to $1.4\times$ and accelerates wall-clock training by up to $1.5\times$, offering a robust optimizer for scaling multimodal foundation models. 

---
# Surrogate Neural Architecture Codesign Package (SNAC-Pack) 

**Authors**: Jason Weitz, Dmitri Demler, Benjamin Hawks, Aaron Wang, Nhan Tran, Javier Duarte  

**Link**: [PDF](https://arxiv.org/pdf/2605.16138)  

**Abstract**: Neural architecture search (NAS) is a powerful approach for automating model design, but existing methods often optimize for accuracy alone or rely on proxy metrics such as bit operations (BOPs) that correlate poorly with hardware cost. This gap is particularly large for FPGA deployment, where cost is dominated by a multi-dimensional budget of lookup tables, DSPs, flip-flops, BRAM, and latency. We present the Surrogate Neural Architecture Codesign Package (SNAC-Pack), an open-source AutoML framework for hardware-aware neural architecture codesign and end-to-end FPGA deployment. SNAC-Pack runs a multi-objective global search with Optuna and NSGA-II, loading trials to a shared SQLite store that enables parallel workers across compute nodes. A hardware surrogate model outputs per-trial resource and latency estimates, avoiding the synthesis cost that would otherwise dominate the search loop. A local search stage then applies quantization-aware training (QAT) together with iterative magnitude pruning in a combined compression loop, after which the final model is synthesized to FPGA firmware via the hls4ml Python library. A YAML configuration and an optional agentic frontend let users run the pipeline on new datasets without modifying the framework. We demonstrate SNAC-Pack on jet classification at the Large Hadron Collider and superconducting qubit readout, discovering compact architectures that match or exceed strong baselines on the task metric while reducing FPGA resource utilization and, in the qubit readout case, reducing the design space exploration process from months of manual fine-tuning to hours of automated search. 

---
# Navigating Potholes with Geometry-Aware Sharpness Minimization 

**Authors**: Simon Dufort-Labbé, Mehrab Hamidi, Razvan Pascanu, Ioannis Mitliagkas, Damien Scieur, Aristide Baratin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16134)  

**Abstract**: Sharpness-aware minimization (SAM) encourages flat minima by perturbing parameters along directions of high loss curvature, but treats all parameter directions uniformly, ignoring the underlying loss geometry. We introduce LLQR+SAM, which combines SAM with a learned preconditioner obtained from the recently proposed LLQR framework, a second-order method that recasts steepest descent as a layerwise linear-quadratic regulator problem. The preconditioner is updated sparsely and maintained as a slow exponential moving average, so it captures a smoothed, low-resolution picture of the loss landscape geometry. The SAM perturbation then operates on top of this learned geometry, probing curvature at a faster timescale. We show that this two-timescale structure is not merely a computational convenience: theoretically, the preconditioner amplifies the SAM escape signal in directions that are flat under the average geometry but locally sharp (potholes). Wide, flat basins, by contrast, remain stable. Empirically, LLQR+SAM gives consistent gains over both SAM and LLQR alone across standard vision and sequence modeling benchmarks, supporting the view that slow learned geometry and fast sharpness correction are genuinely complementary. 

---
# Entropy Across the Bridge: Conditional-Marginal Discretization for Flow and Schrödinger Samplers 

**Authors**: Bruno Trentini, Dejan Stancevic, Michael M. Bronstein, Alexander Tong, Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2605.16126)  

**Abstract**: For a fixed flow-based generative model under a small inference budget, sample quality can depend strongly on where the sampler spends its few function evaluations. Flow matching and Schrödinger bridges define probability paths, yet their inference grids are usually heuristic or inherited from one-endpoint diffusion. We derive a conditional-marginal entropy-rate objective for bridge-aware discretization, separating endpoint-conditioned bridge geometry from marginal flow evolution, and use it to build a training-free entropic inference-time scheduler from first principles. For Gaussian Brownian bridges this rate is closed-form and U-shaped, motivating boundary-heavy nonuniform grids. On trained two-dimensional bridge/flow models, the estimated profile recovers the predicted shape and improves 10-step ODE-Heun MMD over linear by 18.1%, with a paired 22.7% SDE-Heun improvement in the same low-NFE sweep. On EDM/CIFAR-10, the entropic time-discretization gives the best tested five-step FID (186.3 \pm 4.0 versus 200.5 \pm 2.9 for linear and 238.0 \pm 5.3 for cosine). On AlphaFlow protein generation, entropic conditional-marginal (cond-marg) scheduling shows advantage in low-NFE regimes on both CAMEO22 and ATLAS benchmarks. These results support entropy-rate scheduling as a practical low-budget allocation signal for high-dimensional bridge and flow samplers. 

---
# GenShield: Unified Detection and Artifact Correction for AI-Generated Images 

**Authors**: Zhipei Xu, Xuanyu Zhang, Youmin Xu, Qing Huang, Shen Chen, Taiping Yao, Shouhong Ding, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16122)  

**Abstract**: Diffusion-based image synthesis has made AI-generated images (AIGI) increasingly photorealistic, raising urgent concerns about authenticity in applications such as misinformation detection, digital forensics, and content moderation. Despite the substantial advances in AIGI detection, how to correct detected AI-generated images with visible artifacts and restore realistic appearance remains largely underexplored. Moreover, few existing work has established the connection between AIGI detection and artifact correction. To fill this gap, we propose GenShield, a unified autoregressive framework that jointly performs explainable AIGI detection and controllable artifact correction in a closed loop from diagnosis to restoration, revealing a mutually reinforcing relationship between these two tasks. We further introduce a Visual Chain-of-Thought based curriculum learning strategy that enables self-explained, multi-step ``diagnose-then-repair'' correction with an explicit stopping criterion. A high-quality dataset with large-scale ``artifact-restored'' pairs is also constructed alongside a unified evaluation pipeline. Extensive experiments on our correction benchmark and mainstream AIGI detection benchmarks demonstrate state-of-the-art performance and strong generalization of our method. The code is available at this https URL. 

---
# DebiasRAG: A Tuning-Free Path to Fair Generation in Large Language Models through Retrieval-Augmented Generation 

**Authors**: Rui Chu, Bingyin Zhao, Thanh Quoc Hung Le, Duy Cao Hoang, Huawei Lin, Ping Li, Weijie Zhao, Khoa D Doan, Yingjie Lao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16113)  

**Abstract**: Large language models (LLMs) have achieved unprecedented success due to their exceptional generative capabilities. However, because they depend on knowledge encapsulated from training corpora, they may produce hallucinations, stereotypes, and socially biased content. In particular, LLMs are prone to prejudiced responses involving race, gender, and age, which are collectively referred to as social biases. Prior studies have used fine-tuning and prompt engineering to mitigate such biases in LLMs, but these methods require additional training resources or domain knowledge to design the framework. Moreover, they may degrade the original capabilities of LLMs and often overlook the need for dynamic debiasing contexts for fairer inference. In this paper, we propose DebiasRAG, a novel tuning-free and dynamic query-specific debiasing framework based on retrieval-augmented generation (RAG). DebiasRAG improves fairness while preserving the intrinsic properties of LLMs, such as representation ability. DebiasRAG consists of three stages: (1) query-specific debiasing candidate generation; (2) context candidate pool construction; and (3) gradient-updated debiasing-guided context piece reranking. First, DebiasRAG leverages self-diagnosed bias contexts relevant to the query through regular retrieval, where the bias contexts are prepared offline by the DebiasRAG provider. Given the query-specific bias contexts, DebiasRAG reversely produces debiasing contexts, which are provided as additional fairness constraints for LLM outputs. Second, a regular RAG retrieval process produces query-related contexts from the regular RAG document database, such as a chunked Wikipedia dataset. 

---
# Attention Dispersion in Dynamic Graph Transformers: Diagnosis and a Transferable Fix 

**Authors**: Jinhao Zhang, Kangfei Zhao, Qiuhao Zeng, Long-Kai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16112)  

**Abstract**: Transformer-based architectures have become the dominant paradigm for Continuous-Time Dynamic Graph (CTDG) learning, yet their performance remains limited on temporally shifted datasets. In this work, we identify attention dispersion as a shared failure mode of dynamic graph Transformers under temporal distribution shift. Through controlled ablation contrasting structurally and temporally distinguished historical neighbors against random ones, we show that prediction depends on a class of critical nodes that carry consistently more predictive signal than arbitrary neighbors. However, existing Transformers fail to focus on these nodes even when they are present in the input, as temporal shift weakens attention contrast and produces overly dispersed attention distributions. This diagnosis suggests a simple and transferable fix: replace standard attention with differential attention, which suppresses common-mode attention and amplifies distinctive token-level signals. When added to three representative CTDG Transformer baselines, differential attention consistently improves performance, with gains concentrated on high-shift datasets. Attention-level measurements further confirm the mechanism, showing reduced attention entropy and increased attention mass on critical nodes. Building on these findings, we introduce DiffDyG, a reference implementation combining differential attention with standard input encodings. Across 9 benchmarks and three negative sampling protocols, DiffDyG achieves SOTA performance, with especially large gains on the most shifted datasets. 

---
# Federated Imputation under Heterogeneous Feature Spaces 

**Authors**: Imane Hocine, Chaimaa Medjadji, Sylvain Kubler, Gregoire Danoy, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2605.16099)  

**Abstract**: Federated Learning (FL) enables collaborative training across decentralized clients, but most methods assume aligned feature schemas, an assumption that rarely holds in tabular settings where clients observe only partially overlapping feature subsets. In these heterogeneous feature spaces, parameter-averaging methods (e.g., FedAvg) transfer little information across weakly overlapping or disjoint feature groups, limiting their effectiveness for federated imputation. To overcome this, we propose \textbf{FedHF-Impute}, a federated imputation framework that separates structural feature unavailability from conventional missingness and uses a shared global feature graph to propagate information across statistically related features through message passing. This enables indirect cross-client knowledge transfer, even when features are never jointly observed locally, while preserving standard federated communication. Under simulated partial schema overlap on the SECOM and AirQuality datasets, FedHF-Impute improves imputation accuracy (RMSE) over FL baselines by 26.9\%, and 8.4\% respectively, while achieving comparable performance on PhysioNET, with only a 0.3\% difference relative to the best baseline. 

---
# GeoGS-CE: Learning Delay--Beam Channel Priors with 3D Gaussians for High-Mobility Scenarios 

**Authors**: Yumeng Zhang, Jiajia Guo, Chaozheng Wen, Chenghong Bian, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16094)  

**Abstract**: Wideband channel estimation (CE) in high-mobility scenarios remains challenging because channel responses vary rapidly, while practical systems can allocate only sparse pilots to accommodate dense users. Fortunately, many high-mobility environments, such as high-speed railways, exhibit scheduled trajectories, predictable velocities, and a limited number of dominant propagation paths. These properties induce a delay--beam power spectrum that is more stable than the instantaneous complex channel frequency response (CFR), less sensitive to the random phase coherence, and rich in geometric information. To exploit such environmental properties, we propose GeoGS-CE, a two-stage channel estimation framework for sparse-pilot high-mobility scenarios. In the offline stage, GeoGS-CE jointly models: 1) a scene-level 3D Gaussian representation that captures the non-line-of-sight (NLoS) geometric scattering support, and 2) a leakage-aware differentiable wireless rendering process that maps the NLoS Gaussians, together with an explicit virtual line-of-sight (LoS) component, to the measured delay--beam power spectrum, while accounting for practical OFDM delay and array leakage effects. In the online stage, the delay--beam power spectrum is predicted for each user location and used as a strong covariance prior, enabling accurate full-band and full-array CFR reconstruction and tracking through a linear MMSE estimator. Simulations based on channels generated from a segment of the Guangshen high-speed railway show that the proposed geometric prior substantially improves CFR reconstruction over pilot-only and non-geometric baselines. 

---
# Centralized vs Decentralized Federated Learning: A trade-off performance analysis 

**Authors**: Chaimaa Medjadji, Guilain Leduc, Sylvain Kubler, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2605.16089)  

**Abstract**: Federated Learning (FL) has emerged as a promising paradigm for collaborative model training across distributed edge devices while preserving data privacy especially with the huge increase amount of data due to the adoption of technologies which contributes to the growing number of IoT devices. Storing this amount of data centrally is challenging due to issues like limited communication, privacy, and regulations. FL can be Centralized (CFL), Decentralized (DFL), and Semi-decentralized (SDFL). Choosing the right FL architecture depends on the application's needs. However, very few research studies have experimentally compared these three types of architectures to not only understand the respective strengths and limitations, but also trade-offs between different performance indicators. This paper overcome this lack of analysis, conducting experimental analyses using the Fedstellar simulator, MNIST dataset, and MLP classifier. 

---
# Multi-level Self-supervised Pretraining on Compositional Hierarchical Graph for Molecular Property Prediction 

**Authors**: Xiayu Liu, Zhengyi Lu, Hou-biao Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16088)  

**Abstract**: Self-supervised pretraining on molecular graphs has emerged as a promising approach for molecular property prediction, yet most existing methods operate at a single structural granularity and treat bond information as auxiliary edge attributes rather than as an independent semantic layer. In this work, we propose MolCHG, a multi-level self-supervised pretraining framework built upon a novel Compositional Hierarchical Graph that organizes molecular structure into four types of nodes across three semantic levels. By introducing a bond graph that operates in parallel with the atom graph, our architecture elevates bond-level information to independently evolving node representations, enabling fragment nodes to aggregate atom-level and bond-level semantics on an equal footing. We design three level-specific pretraining objectives: an atom-bond cross-view contrastive task that aligns the atom-view and bond-view representations within each fragment, a fragment-level functional group prediction task to inject domain-relevant chemical knowledge, and graph-level structure prediction tasks to encode global molecular topology. Experiments on nine MoleculeNet benchmarks demonstrate that MolCHG achieves the best performance on seven datasets across both classification and regression tasks, remaining competitive with the strongest baselines on the rest. Ablation studies further confirm that the multi-level supervision signals are complementary and that each component contributes to the overall performance. 

---
# Towards Trustworthy and Explainable AI for Perception Models: From Concept to Prototype Vehicle Deployment 

**Authors**: Till Beemelmanns, Shayan Sharifi, Manas Mehrotra, Ayushman Choudhuri, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.16087)  

**Abstract**: Deep Neural Networks have become the dominant solution for Autonomous Driving perception, but their opacity conflicts with emerging Trustworthy AI guidelines and complicates safety assurance, debugging, and human oversight. While theoretical frameworks for safe and Explainable AI (XAI) exist, concrete implementations of Trustworthy AI for 3D scene understanding remain scarce. We address this gap by proposing a Trustworthy AI perception module that is remarkably robust, integrates faithful explainability, and calibrated uncertainty estimates. Building on a transformer-based detector, we derive explanation from the attention mechanism at inference time and validate their faithfulness using perturbation-based consistency tests. We further integrate an uncertainty estimation and calibration module, and apply robustness-enhancing training methods. Experiments show faithful saliency behavior, improved robustness, and well-calibrated uncertainty estimates. Finally, we deploy these Trustworthy AI elements in a prototype vehicle and provide an XAI Interface that visualizes documentation artifacts, model uncertainty state, and saliency maps, demonstrating the feasibility of trustworthy perception monitoring in real time. Supplementary materials are available at this https URL . 

---
# Towards Foundation Models for Relational Databases with Language Models and Graph Neural Networks 

**Authors**: Jingcheng Wu, Ratan Bahadur Thapa, Mojtaba Nayyeri, Lucas Etteldorf, Max Finkenbeiner, Fabian Leeske, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2605.16085)  

**Abstract**: Relational databases store much of the world's structured information, and they are essential for driving complex predictive applications. However, deep learning progress on relational data remains limited, as conventional approaches flatten databases into single tables via manual feature engineering, discarding relational context. Relational deep learning (RDL) addresses this by modeling databases as relational entity graphs (REGs) for graph neural networks (GNNs), but remains task- and database-specific. To combine the strengths of both paradigms, we propose a hybrid architecture combining a fine-tuned BART encoder to capture intra-row semantics with a GraphSAGE-based GNN over REGs to inject relational context. Experiments on RelBench show that the GNN substantially enriches BART's row embeddings, achieving a ROC-AUC of 67.40 on the driver-dnf task from the rel-f1 dataset. This performance is competitive with supervised baselines such as LightGBM (68.86) and narrows the gap to RDL (72.62) to within 5.22 points, though a substantial gap remains to state-of-the-art foundation models such as KumoRFM (82.63). These results suggest that lightweight hybrid LM-GNN architectures offer a promising and resource-efficient path towards foundation models for relational databases. 

---
# VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation 

**Authors**: Yiming Zhao, Yu Zeng, Wenxuan Huang, Zhen Fang, Qing Miao, Qisheng Su, Jiawei Zhao, Jiayin Cai, Lin Chen, Zehui Chen, Yukun Qi, Yao Hu, Xiaolong Jiang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16079)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in video understanding, yet they face substantial challenges in tasks requiring precise spatiotemporal localization at the instance level. Existing methods primarily rely on text prompts for human-model interaction, but these prompts struggle to provide precise spatial and temporal references, resulting in poor user experience. Furthermore, current approaches typically decouple visual perception from language reasoning, centering reasoning around language rather than visual content, which limits the model's ability to proactively perceive fine-grained visual evidence. To address these challenges, we propose VideoSeeker, a novel paradigm for instance-level video understanding through visual prompts. VideoSeeker seamlessly integrates agentic reasoning with instance-level video understanding tasks, enabling the model to proactively perceive and retrieve relevant video segments on demand. We construct a four-stage fully automated data synthesis pipeline to efficiently generate large-scale, high-quality instance-level video data. We internalize tool-calling and proactive perception capabilities into the model via cold-start supervision and RL training, building a powerful video understanding model. Experiments demonstrate that our model achieves an average improvement of +13.7% over baselines on instance-level video understanding tasks, surpassing powerful closed-source models such as GPT-4o and Gemini-2.5-Pro, while also showing effective transferability on general video understanding benchmarks. The relevant datasets and code will be released publicly. 

---
# AgriMind: An Ensemble Deep Learning Framework for Multi-Class Plant Disease Classification 

**Authors**: Salma Hoque Talukdar Koli, Fahima Haque Talukder Jely  

**Link**: [PDF](https://arxiv.org/pdf/2605.16076)  

**Abstract**: Plant disease detection is still largely manual in Bangladesh, where extension workers eyeball leaf samples across millions of smallholdings. We built AgriMind to automate this: an ensemble of ResNet50, EfficientNet-B0, and DenseNet121 trained on 20,638 PlantVillage images across 15 pepper, potato, and tomato disease classes. Transfer learning with frozen ImageNet backbones and 10 epochs of head-only training keeps the pipeline lightweight. Individual models hit 96--97% on the held-out test set, but averaging their softmax outputs pushes the ensemble to 99.23% -- a two-thirds cut in error rate. We tried biasing the average toward the best validation model; it backfired. Dropping any single model also hurt. Pepper and potato classify perfectly; tomato, with ten visually similar classes, still reaches 99.01%. On an NVIDIA T4 GPU the full ensemble runs at 53 FPS. Whether that translates to real-time mobile use depends on TensorFlow Lite optimization -- work we have not yet completed. 

---
# Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting 

**Authors**: Raushan Joshi, Jean-Yves Guillemaut  

**Link**: [PDF](https://arxiv.org/pdf/2605.16065)  

**Abstract**: 3D Gaussian Splatting (3D-GS) enables real-time 3D scene reconstruction but lacks robust segmentation for editing tasks such as object removal, extraction, and recoloring. Existing approaches that lift 2D segmentations to the 3D domain suffer from view inconsistencies and coarse masks. In this paper, we propose a novel framework that leverages the Segment Anything Model High Quality (SAM-HQ) to generate accurate 2D masks, addressing the limitations of the standard SAM in boundary fidelity and fine-structure preservation. To achieve robust 3D segmentation of any target object in a given scene, we introduce a prior-guided label reassignment method that assigns labels to 3D Gaussians by enforcing multiview consistency with learned priors. Our approach achieves state-of-the-art segmentation accuracy and enables interactive, real-time object editing while maintaining high visual fidelity. Qualitative results demonstrate superior boundary preservation and practical utility in Virtual Reality (VR) and robotics, advancing 3D scene editing. 

---
# Misspecified Explore-then-Exploit Leads to Supra-Competitive Prices 

**Authors**: Jackie Baek, Vivek F. Farias, Farrell Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16064)  

**Abstract**: We study whether simple algorithmic pricing systems can systematically produce collusive-like prices in multi-firm markets. We consider firms using an explore-then-exploit pipeline: they randomize prices during an initial exploration phase, then estimate demand from their own historical data and set prices myopically thereafter. The estimation step relies on a misspecified, monopoly-style model that omits competitors' prices. We characterize when this pipeline converges to supra-competitive prices above the Nash equilibrium, via a fluid-limit ordinary differential equation analysis. We show that supra-competitive prices arise when firms explore within similar price ranges on the same side of the Nash price. Moreover, prices can be substantially above the Nash price; we show that prices can reach monopoly levels under symmetric exploration. Simulations calibrated to a real multifamily rental market confirm that supra-competitive outcomes arise robustly beyond our theoretical assumptions, including under finite horizons, heterogeneous products, and nonlinear logit demand. 

---
# Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making 

**Authors**: Fan Feng, Selena Ge, Minghao Fu, Zijian Li, Yujia Zheng, Zeyu Tang, Yingyao Hu, Biwei Huang, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16054)  

**Abstract**: Recent work has framed decision-making as a sequence modeling problem using generative models such as diffusion models. Although promising, these approaches often overlook latent factors that exhibit evolving dynamics, elements that are fundamental to environment transitions, reward structures, and high-level agent behavior. Explicitly modeling these hidden processes is essential for both precise dynamics modeling and effective decision-making. In this paper, we propose a unified framework that explicitly incorporates latent dynamic inference into generative decision-making from minimal yet sufficient observations. We theoretically show that under mild conditions, the latent process can be identified from small temporal blocks of observations. Building on this insight, we introduce Ada-Diffuser, a causal diffusion model that learns the temporal structure of observed interactions and the underlying latent dynamics simultaneously, and furthermore, leverages them for planning and control. With a modular design, Ada-Diffuser supports both planning and policy learning tasks, enabling adaptation to latent variations in dynamics, rewards, and latent actions. Experiments on simulated control and robotic benchmarks demonstrate its effectiveness in accurate latent inference and adaptive policy learning. 

---
# Looped SSMs: Depth-Recurrence and Input Reshaping for Time Series Classification 

**Authors**: Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16048)  

**Abstract**: State Space Models (SSMs) are inherently recurrent along the sequence dimension, yet depth-recurrence - reusing the same block repeatedly across layers, as recently applied in looped transformers - has not been explored in this model family. We show that a looped SSM with $k$ parameters iterated $L$ times consistently closely matches or outperforms a standard SSM with $k \cdot L$ independent parameters across four architectures (LRU, S5, LinOSS, LrcSSM) and six time series classification benchmarks, despite operating within a strictly smaller hypothesis space, as we formally establish. Since the larger model contains the looped model as a special case, this dominance cannot be explained by expressivity and instead points to parameter sharing across depth as a beneficial inductive bias that simplifies optimization. These results demonstrate that depth-recurrence is orthogonal to sequence-recurrence and independently beneficial. We further show that input reshaping is an equally neglected design axis: concatenating timesteps for low-dimensional inputs, or flattening and rechunking the joint feature-time dimension for high-dimensional ones, yields accuracy gains of 1-6% across all models, confirmed over 5 random seeds. Both techniques provide standalone improvements that compound when combined, suggesting that depth and input reshaping are two independent and underexplored design axes for SSMs on time series. 

---
# XSearch: Explainable Code Search via Concept-to-Code Alignment 

**Authors**: Yiming Liu, Ruofan Liu, Yun Lin, Zicong Zhang, Weiyu Kong, Pengnian Qi, Xiao Cheng, Weinan Zhang, Qianxiang Wang, Linpeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16046)  

**Abstract**: Semantic code search has been widely adopted in both academia and industry. These approaches embed natural-language queries and code snippets into a shared embedding space and retrieve results based on vector similarity. Despit strong performance on benchmark datasets, they often suffer from poor explainability and generalization. Retrieved code may appear semantically similar yet miss critical functional requirements of the query, while providing no explanation of why the result was retrieved. Moreover, such failures become more severe under distribution shift, where models struggle to generalize to unseen benchmarks. In this work, we propose XSearch, an intrinsically explainable code search framework. Our key insight is that by relying on global embedding similarity, existing retrievers inherently take an inductive view. They learn statistical patterns rather than truly understanding the query's functional requirements. We address this problem by reformulating code search as a deductive concept alignment problem. XSearch (i) identifies functional concepts in the query and (ii) explicitly aligns them with corresponding code statements. This explain-then-predict design produces inherent concept-level explanations and mitigates shortcut learning that harms out-of-distribution generalization. We train an encoder with explicit concept-alignment objectives and perform retrieval through explicit matching between query concepts and code statements. Experiments show that, trained on CodeSearchNet using GraphCodeBERT (125M parameters), XSearch improves performance on out-of-distribution benchmarks from 0.02 to 0.33 (15x) over eight state-of-the-art retrievers, and consistently outperforms both encoder- and decoder-based baselines with up to 7B parameters. A user study demonstrates that concept-alignment explanations enable users to evaluate retrieved results faster and more accurately. 

---
# RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents 

**Authors**: Zijie Dai, Shiyuan Deng, Sheng Guan, Yizhou Tian, Xin Yao, Xiao Yan, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16045)  

**Abstract**: Memory systems often organize user-agent interactions as retrievable external memory and are crucial for long-running agents by overcoming the limited context windows of LLMs. However, existing memory systems invoke LLMs to process every incoming interaction for memory extraction, and such an eager memory consolidation scheme leads to substantial token consumption. To tackle this problem, we propose RecMem by rethinking when memory consolidation should be conducted. RecMem stores incoming interactions in a subconscious memory layer and encode them using lightweight embedding models for retrieval. LLMs are only invoked to extract episodic and semantic memory when sustained recurrence are observed for semantically similar interactions. Such recurrence-based consolidation works because these interactions correspond to a semantic cluster with rich information and thus are worth extraction and summarization. To improve accuracy, RecMem also incorporates a semantic refinement mechanism that recovers the fine-grained facts omitted by memory extraction. Experiments show that RecMem reduces the memory construction token cost of three SOTA memory systems by up to 87% while exceeding their accuracy. 

---
# Learning Sim-Grounded Policies for Bimanual Rope Manipulation from Human Teleoperation Data 

**Authors**: Gina Wigginghaus, Tim Missal, Berk Guler, Simon Manschitz, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2605.16043)  

**Abstract**: Deformable Linear Objects (DLOs) such as ropes and cables are widely encountered in both household and industrial applications, yet remain challenging to manipulate due to their infinite-dimensional configuration space and frequent self-occlusion. Imitation learning from teleoperation offers a practical path to bimanual DLO manipulation, but its scalability is limited by human effort, making the choice of observation space critical for generalization from small datasets. In this study, we investigate whether the lack of generalization in egocentric visual policies for the knot-untangling task stems from the observation space itself, rather than from the policy architecture or data scale. We compare two Action Chunking with Transformers policies trained on the same bimanual teleoperation data: a vision-based policy conditioned on two egocentric RGB streams from wrist-mounted cameras, and a state-based policy conditioned on the DLO's 3D particle state, extracted from an initial observation via multi-view fusion and evolved in a particle-based eXtended Position-Based Dynamics simulation. Evaluated open-loop on an unseen rope configuration, the state-based policy outperforms its visual counterpart with a 30.8% reduction in L1 error when predicting the initial grasp-and-pull action, quantifying the observability gap between pixels and physics-consistent state, and pointing toward more data-efficient robot learning for the DLO manipulation task from limited human demonstrations. 

---
# Who Owns This Agent? Tracing AI Agents Back to Their Owners 

**Authors**: Ruben Chocron, Doron Jonathan Ben Chayim, Eyal Lenga, Gilad Gressel, Alina Oprea, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2605.16035)  

**Abstract**: AI agents are increasingly deployed to act autonomously in the world, yet there is still no reliable way to trace a harmful agent back to the account that deployed it. This creates the same accountability gap across both ends of the intent spectrum: benign operators may deploy misconfigured or overbroad agents that cause harm unintentionally, while malicious operators may deliberately weaponize agents for scams, harassment, or cyber attacks. In many cases, these agents are powered by vendor-hosted models, a dependency that holds even for sophisticated adversaries such as state actors conducting cyber operations. In either case, affected parties can observe the behavior but cannot notify the responsible operator, stop the session, or identify the account for investigation.
We formalize this gap as the problem of agent attribution: linking an observed agent interaction to the responsible account at the hosting vendor. To our knowledge, this is the first work to define the problem and present a practical solution. Our protocol is canary-based: an authorized party injects a canary into the agent's interaction stream, and the vendor searches a narrow window of session logs to recover the originating session and account. Simple canaries suffice in non-adversarial settings. For adversarial operators who filter or paraphrase incoming content, we develop robust canary constructions that cannot be suppressed without degrading the agent's own task performance, yielding a formal asymmetry in the defender's favor. We evaluate a variety of scenarios including real-world agents and show that our attribution method is reliable, robust, and scalable for vendor-side deployment. 

---
# From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation 

**Authors**: Yu Pan, Yang Hou, Xiongfei Wu, Liang Zhang, Yves Le Traon, Lei Ma, Jianjun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16026)  

**Abstract**: Compositional speech-to-speech translation (S2ST) systems built upon speech large language models (SpeechLLMs) have recently shown promising performance. However, existing S2ST systems often either neglect source-language information or encode it through a language-as-label paradigm, representing each source language as an independent flat embedding. Such a design overlooks systematic linguistic structure shared across languages, which may limit data-efficient multilingual adaptation when supervised S2ST data are scarce. To address this issue, we propose S2ST-Omni 2, a many-to-one compositional S2ST framework that systematically reformulates multilingual language conditioning from flat language labels to structured typological priors. Specifically, S2ST-Omni 2 revisits language conditioning at three levels: typology-informed hierarchical language encoding for structured source-language representation, dynamically-gated language-aware Dual-CTC for content-adaptive acoustic modulation, and typology-aware LLM prompting for decoder-side linguistic guidance. Experiments on CVSS-C show that S2ST-Omni 2 achieves superior average performance among representative S2ST approaches across BLEU, COMET, ASR-BLEU, and BLASER 2.0 under the adopted evaluation protocol. Ablation studies indicate that the proposed representation-level, acoustic-level, and decoding-level strategies provide complementary benefits. Moreover, controlled data-budget analyses and a Japanese-to-English evaluation using only approximately 3 hours of supervised training data suggest that explicit typological priors provide useful inductive biases for data-efficient multilingual S2ST. 

---
# Can Vision Language Models Be Adaptive in Mathematics Education? A Learner Model-based Rubric Study 

**Authors**: Jie Gao, Yongan Yu, Junzhu Su, Yiran Lin, Adam K. Dube, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2605.16011)  

**Abstract**: Adaptive learning refers to educational technologies that track learners' learning progress and adapt the instructional process based on individual learners' learning performance. It is increasingly recognized as critical for developing an effective learning support tool. Vision language models (VLMs) have seen adoption in mathematics education, and students have been using them as learning aids for personalized instruction. However, it is unknown whether VLMs have the ability to adapt to different learner profiles when providing mathematical instructions. Current VLMs lack a systematic evaluation framework for this adaptivity to different learner profiles in mathematics tutoring tasks. To address this gap, we draw on the learner model from the adaptive learning framework (Shute and Towle, 2018) and propose a learner model-based rubric. Our rubric formalizes adaptivity assessment into three aspects: cognitive aspects, motivational aspects, and complexity. We also evaluate two additional dimensions of VLM responses: correctness (of answers and solutions) and quality (of the response itself). Our experimental results show measurable differences in adaptivity across models and also reveal that current VLMs struggle to consistently produce learner model-based instructional responses, especially when receiving limited learner information. 

---
# CitePrism: Human-in-the-Loop AI for Citation Auditing and Editorial Integrity 

**Authors**: Gowrika Mahesh, Budanur Madappa Darshan Gowda, Kavana Gopladevarahalli Papegowda, Prajwal Basavaraj, Binh Vu, Swati Chandna, Mehrdad Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2605.16000)  

**Abstract**: Editors and reviewers are expected to ensure that manuscripts cite relevant, accurate, current, and ethically appropriate literature, yet manuscript-level citation auditing remains largely manual, fragmented, and difficult to scale. Citation context, metadata quality, self-citation patterns, and bibliographic integrity all affect whether a reference appropriately supports a local claim. We present CitePrism, a transparent hybrid decision-support framework for editorial citation auditing that combines LLM-assisted contextual reasoning, embedding-based semantic similarity, metadata verification, integrity-oriented flags, and human-in-the-loop analyst review. CitePrism extracts citation neighborhoods, enriches reference metadata, computes fused relevance scores, surfaces metadata and self-citation review prompts, and supports configurable threshold-based triage. In a preliminary validation on a single case-study manuscript with 104 references from pavement engineering, agreement with human binary relevance labels reached Cohen's kappa = 0.429. At operating threshold tau = 17, CitePrism flagged all human-labeled irrelevant citations, while also producing false positives requiring analyst review. These results suggest that CitePrism may support conservative editorial screening and citation-quality triage, but they do not establish general editorial performance. CitePrism is intended as pilot-stage decision support, not as an autonomous misconduct detector or automated editorial decision system. Broader validation across manuscripts, domains, annotators, baselines, and deployment settings is required before operational use. 

---
# Constrained latent state modeling: A unifying perspective on representation learning under competing constraints 

**Authors**: Gwenolé Quellec  

**Link**: [PDF](https://arxiv.org/pdf/2605.15995)  

**Abstract**: Learning latent representations from complex data is central to modern machine learning, spanning temporal, multimodal, and partially observed systems. In such settings, representations are better understood as latent states capturing underlying system dynamics, rather than as mere compressed summaries of observations. Yet current approaches remain fragmented, relying on distinct -- and often implicit -- assumptions about what these states should represent. We argue that this fragmentation reflects a more fundamental limitation: latent representations are typically learned from underconstrained objectives that fail to specify the properties that meaningful latent states should satisfy. As a result, multiple representations can satisfy the same objective, leading to ambiguity in their structure and interpretation. While many of the underlying principles have been explored in isolation, their interactions have not been explicitly formalized. In this work, we propose constrained latent state modeling (CLSM) as a unifying perspective. We identify a set of core properties -- predictive sufficiency, minimality, temporal coherence, observation compatibility, invariance to nuisance factors, and structural constraints -- and show that they are intrinsically coupled through fundamental trade-offs. Revisiting major modeling families through this lens, we show that existing approaches can be interpreted as enforcing different subsets of constraints, thereby occupying distinct regions of a common design space. This perspective reframes persistent challenges such as lack of identifiability as consequences of underconstrained formulations, rather than isolated technical limitations. More broadly, CLSM provides a principled framework to make design choices explicit, to analyze trade-offs, and to guide the development of more interpretable, robust, and task-aligned latent state models. 

---
# Beyond Content: A Comprehensive Speech Toxicity Dataset and Detection Framework Incorporating Paralinguistic Cues 

**Authors**: Zhongjie Ba, Liang Yi, Peng Cheng, Qingcao Li, Qinglong Wang, Li Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15984)  

**Abstract**: Toxic speech detection has become a crucial challenge in maintaining safe online communication environments. However, existing approaches to toxic speech detection often neglect the contribution of paralinguistic cues, such as emotion, intonation, and speech rate, which are key to detecting speech toxicity. Moreover, current toxic speech datasets are predominantly text-based, limiting the development of models that can capture paralinguistic this http URL address these challenges, we present ToxiAlert-Bench, a large-scale audio dataset comprising over 30,000 audio clips annotated with seven major toxic categories and twenty fine-grained toxic labels. Uniquely, our dataset annotates toxicity sources -- distinguishing between textual content and paralinguistic origins -- for comprehensive toxic speech this http URL, we propose a dual-head neural network with a multi-stage training strategy tailored for toxic speech detection. This architecture features two task-specific classification headers: one for identifying the source of sensitivity (textual or paralinguistic), and the other for categorizing the specific toxic type. The training process involves independent head training followed by joint fine-tuning to reduce task interference. To mitigate data class imbalance, we incorporate class-balanced sampling and weighted loss this http URL experimental results show that leveraging paralinguistic features significantly improves detection performance. Our method consistently outperforms existing baselines across multiple evaluation metrics, with a 21.1% relative improvement in Macro-F1 score and a 13.0% relative gain in accuracy over the strongest baseline, highlighting its enhanced effectiveness and practical applicability. 

---
# Ontology for Policing: Conceptual Knowledge Learning for Semantic Understanding and Reasoning in Law Enforcement Reports 

**Authors**: Anita Srbinovska, Jansen Orfan, Adrian Martin, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2605.15978)  

**Abstract**: Law enforcement reports contain structured fields and written narratives. However, many incident facts that are needed for review, police training, and investigations are in natural language and require manual reading. We propose a framework using symbolic methods for converting narratives into evidence-linked facts. Our objective is to measure the value of narratives to recover incident details only from the unstructured text and build temporal graphs with time cues and domain axioms. We achieve this by redacting personal identifiers, semantic parsing, predicate mapping to ontology, and reasoning. We evaluate the symbolic approach on 450 property crime reports and a short human review. Of the extracted events from the system, 54.1% had a confidence score of at least 0.80 and 93.7% were mapped through the PropBank--VerbNet--WordNet semantic path. 100% agreement was reached on incident initiation, stolen items, and temporal cues and lower agreement for forced entry interpretation. 

---
# Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective 

**Authors**: Ernesto Garcia-Estrada, Carlos Escolano, José A. R. Fonallosa  

**Link**: [PDF](https://arxiv.org/pdf/2605.15976)  

**Abstract**: Production machine translation relies overwhelmingly on encoder-decoder Seq2Seq models, yet reinforcement learning approaches to MT fine-tuning have largely targeted decoder-only LLMs at $\geq$7B parameters, with limited systematic study of encoder-decoder architectures. We apply Group Relative Policy Optimization to NLLB-200 (600M and 1.3B) using a hybrid reference-free reward (LaBSE and COMET-Kiwi) that requires no parallel data at fine-tuning time, evaluating across 13 typologically diverse languages. GRPO yields consistent improvements on all 13 languages, up to $+$5.03 chrF++ for Traditional Chinese, and, without any target-language data, competes with 3-epoch supervised fine-tuning on morphologically complex languages . We identify a consistent empirical pattern in which gains are largest where baseline performance is weakest and reward discriminability is highest, making this approach most effective precisely where parallel data is scarcest, and replicate this pattern across English and Spanish source languages. 

---
# When and Why Adversarial Training Improves PINNs: A Neural Tangent Kernel Perspective 

**Authors**: Yuan-dong Cao, Chi Chiu SO, Jun-Min Wang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15959)  

**Abstract**: Physics-informed neural networks (PINNs) are powerful surrogates for differential equations but are notoriously difficult to train due to spectral bias, stiffness, and poor accuracy on high-frequency or multiscale solutions. Adversarial training based on generative adversarial networks (GANs) has recently gained surprisingly strong empirical results in improving training, but the underlying mechanisms remain elusive. To this end, we propose a new analysis framework for adversarially trained PINNs, based on the key observation of how the discriminator in GANs can influence the training dynamics of PINNs. The framework first provides a much needed theoretical grounding to why and when adversarial training is effective in PINNs, then presents a unified analysis of GANs variants in such training, and finally leads to a new, practical, efficient training algorithm for PINNs. Empirical results demonstrate that our method can significantly reduce the pathology of PINNs training, thereby providing better models with superior performances, often several magnitudes more accurate than alternative methods. 

---
# Decomposed Vision-Language Alignment for Fine-Grained Open-Vocabulary Segmentation 

**Authors**: Chenhao Wang, Yingrui Ji, Yu Meng, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15942)  

**Abstract**: Open-vocabulary segmentation models often struggle to generalize to unseen combinations of object categories and attributes, because fine-grained descriptions are typically encoded as holistic sentences that entangle multiple semantic units. We propose a Decomposed Vision-Language Alignment framework that explicitly factorizes textual prompts into a concept token and multiple attribute tokens, enabling separate cross-modal interactions for each semantic unit. At the feature level, we introduce a Feature-Gated Cross-Attention module that generates attribute-specific gating maps to fuse information in a multiplicative manner, effectively enforcing compositional semantics. At the scoring level, per-token similarities are aggregated in log-space, producing a stable and interpretable compositional matching. The method can be seamlessly integrated into existing transformer-based segmentation architectures and significantly improves generalization to unseen attribute-category compositions in fine-grained open-vocabulary segmentation benchmarks. 

---
# LoCO: Low-rank Compositional Rotation Fine-tuning 

**Authors**: An Nguyen, Jaesik Choi, Anh Tong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15916)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as an critical technique for adapting large-scale foundation models across natural language processing and computer vision. While existing methods such as low-rank adaptations achieve parameter efficiency via low-rank weight updates, they are limited in their ability to preserve the geometric structure of pretrained representations. We introduce Low-rank Compositional Orthogonal fine-tuning (LoCO), a novel PEFT method that constructs orthogonal transformations through low-rank skew-symmetric matrices and compositional rotation chains. We propose an approximation scheme that enables fully parallel computation of compositional rotations, making the approach practical for high-dimensional feature spaces. Our method maintains low computational complexity while maintaining orthogonality with controlled approximation error. We validate LoCO across diverse domains, including diffusion transformer fine-tuning, vision transformer adaptation, and language model adaptation. Our method demonstrates superior or competitive performance compared to both existing orthogonal and non-orthogonal methods. 

---
# SLIP & ETHICS: Graduated Intervention for AI Emotional Companions 

**Authors**: Minseo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.15915)  

**Abstract**: AI emotional companions face a safety-rapport paradox: restrictive safeguards can damage supportive alliance, while permissive systems risk user harm. We present SLIP (Staged Layers of Intervention Protocol), a four-stage graduated methodology deriving interventions (none, soft, hard) from structured qualitative indicators -- affect intensity (a) and narrative dynamism (m) -- alongside ETHICS (Emergent Taxonomy for Human-AI Interaction Context Signals), a "signals not labels" taxonomy. An evaluation combining a small-scale production deployment (N=68 entries, 10 users, 10 weeks) with a synthetic persona battery (N=91, 5 behavioral-risk profiles) achieved 0% false positives for the flow persona and showed expected escalation patterns in crisis-oriented personas. However, initial results showed that 8 consecutive days of high-energy elevation produced zero interventions (0/8), exposing a boundary where the "do not pathologize" principle conflicts with safety. A subsequent three-model stress test demonstrated that increased model capability improves detection from 0/8 to 6/8 while preserving 0/10 flow false positives in the largest model. Read as preliminary, these findings position graduated intervention as a design direction for navigating -- not resolving -- the safety-rapport tension in affective computing. 

---
# Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation 

**Authors**: Shuaiyi Li, Zhisong Zhang, Yan Wang, Lei Zhu, Dongyang Ma, Chenlong Deng, Yang Deng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.15913)  

**Abstract**: Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention. 

---
# RaPD: Resolution-Agnostic Pixel Diffusion via Semantics-Enriched Implicit Representations 

**Authors**: Yanhao Ge, Shanyan Guan, Weihao Wang, Ying Tai, Mingyu You  

**Link**: [PDF](https://arxiv.org/pdf/2605.15908)  

**Abstract**: Natural images are continuous, yet most generative models synthesize them on discrete grids, limiting resolution-flexible generation. Continuous neural fields enable resolution-free rendering, but prior methods introduce continuity only at the decoding stage as an interpolation module, leaving the generative latent space discretized and reconstruction-oriented. We propose RaPD (Resolution-agnostic Pixel Diffusion), which performs diffusion in a continuous Neural Image Field (NIF) latent space. RaPD bridges this reconstruction-generation gap with Semantic Representation Guidance for generation-aware latent learning and a Coordinate-Queried Attention Renderer for coordinate-conditioned, scale-aware rendering. A single denoised latent can be rendered at arbitrary resolutions by changing only the query coordinates, keeping diffusion cost fixed. Experiments demonstrate superior generation quality and resolution scalability. 

---
# Generative Long-term User Interest Modeling for Click-Through Rate Prediction 

**Authors**: Jiangli Shao, Kaifu Zheng, Hao Fang, Huimu Ye, Zhiwei Liu, Bo Zhang, Shu Han, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15905)  

**Abstract**: Modeling long-term user interests with massive historical user behaviors enhances click-through rate (CTR) prediction performance in advertising and recommendation systems. Typically, a two-stage framework is widely adopted, where a general search unit (GSU) first retrieves top-$k$ relevant behaviors towards the target item, and an exact search unit (ESU) generates interest features via tailored attention. However, current target-centered GSU would ignore other latent user interests, leading to incomplete and biased interest features. Additionally, the matching-based retrieval process in GSUs depends on the pairwise similarity score between target item and each historical behavior, which not only becomes time-consuming for online services as user behaviors continue to grow, but also overlooks the interaction information among user behaviors. To combat these problems, we propose a \textbf{Gen}erative \textbf{L}ong-term user \textbf{I}nterest model named GenLI for CTR prediction. GenLI consists of an interest generation module (IGM), a behavior retrieval module (BRM), and an interest fusion module (IFM). The IGM generates multiple interest distributions to indicate different aspects of real-time user interests, which is target-independent and incorporates interaction information among behaviors, ensuring complete and diverse interest features. The BRM selects related behaviors via a simple lookup operation, reducing the time complexity for weighting each behavior to $O(1)$. Finally, the IFM uses delicate gating mechanisms to generate interest features. Based on the generation process, GenLI improves the diversity of user interests and avoids complex matching-based behavioral retrieval, achieving a better balance between accuracy and efficiency for CTR prediction. 

---
# Uncertainty-Aware Wildfire Smoke Density Classification from Satellite Imagery via CBAM-Augmented EfficientNet with Evidential Deep Learning 

**Authors**: Ranjith Chodavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15894)  

**Abstract**: Rapid and accurate wildfire smoke severity assessment from satellite images is essential for emergency response, air quality modeling, and human health risk management. Existing deep learning approaches treat smoke detection as a binary task, producing point estimates without any measure of prediction confidence. We propose a probabilistic framework to categorize a satellite patch into Light, Moderate, and Heavy severity classes and to provide decomposed epistemic and aleatoric uncertainty in a single forward pass. Our architecture uses the backbone of a pre-trained EfficientNet-B3 and a CBAM module with an evidential deep learning head that predicts Dirichlet concentration parameters, directly estimating vacuity (epistemic) and dissonance (aleatoric) without Monte Carlo sampling. Evaluated on 16,298 real satellite patches derived from the Wildfire Detection dataset, our model achieves 93.8% weighted test accuracy (91.1% unweighted) with ECE=0.0274. Selective prediction retaining the most certain 50% of patches achieves 96.7% accuracy. As image quality degrades, uncertainty increases monotonically, and vacuity is a practical scan quality measure. The Moderate class represents transitional smoke conditions that exhibit the highest epistemic uncertainty (mean vacuity = 0.187), confirming the model correctly identifies ambiguous smoke boundary regions. CBAM spatial attention maps localize to structurally distinctive scene regions, and t-SNE demonstrates the clear cluster separation of Light and Heavy smoke. 

---
# CHoE: Cross-Domain Heterogeneous Graph Prompt Learning via Structure-Conditioned Experts 

**Authors**: Peiyuan Li, Yongqi Huang, Jitao Zhao, Dongxiao He, Di Jin, Weixiong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15888)  

**Abstract**: Heterogeneous Graph Prompt Learning (HGPL)has emerged as a promising paradigm for bridging the gap between the objectives of pre-training foundation models and their downstream applications in heterogeneous graph settings. However, existing HGPL methods are primarily designed for in-domain scenarios, whereas real-world deployments often span multiple domains, and the data used for pre-training and downstream tasks may originate from different distributions. Consequently, the applicability of current HGPL approaches is limited to in-domain settings, and their performance typically degrades when application domains shift. To address this serious limitation, we develop CHoE, a cross-domain HGPL method built upon an expert network. During pre-training, we introduce and train structure-conditioned experts, and during prompt tuning, we adopt a structure-aware expert routing and load balancing mechanism to select structurally compatible experts for each meta-path view. In addition, we design a prompt-based semantic fusion module to integrate representations across multiple views for downstream prediction. Extensive experiments show that CHoE consistently improves performance in few-shot cross-domain applications, outperforming all baseline approaches. 

---
# Symplectic Neural Operators for Learning Infinite Dimensional Hamiltonian Systems 

**Authors**: Yeang Makara, Yusuke Tanaka, Takashi Matsubara, Takaharu Yaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15881)  

**Abstract**: The modeling and simulation of infinite-dimensional Hamiltonian systems are central problems in mathematical physics and engineering, however they pose significant computational and structural challenges for standard data-driven architectures. In this work, we introduce the Symplectic Neural Operator, a neural operator architecture designed to preserve the symplectic structure intrinsic to Hamiltonian PDEs. We provide a theoretical characterization of their symplecticity and establish a rigorous long-term stability result based on the combination of symplectic structure preservation and learning accuracy. Numerical experiments on canonical Hamiltonian PDEs corroborate this theoretical result and show that SNOs exhibit improved energy behavior compared with non-structure-preserving neural operators. 

---
# FSCM: Frequency-Enhanced Spatial-Spectral Coupled Mamba for Infrared Hyperspectral Image Colorization 

**Authors**: Tingting Liu, Yuan Liu, Guiping Chen, Xiubao Sui, Qian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15880)  

**Abstract**: Thermal infrared imaging is robust to illumination variations and smoke interference, making it important for all-weather perception. However, the lack of natural color and fine texture limits target recognition, human visual interpretation, and the transfer of visible-light models. Existing infrared colorization methods mainly rely on single-band images, where insufficient spectral cues may lead to structural distortion and semantic confusion. Although infrared hyperspectral images provide rich spectral responses and material information, existing single-band frameworks remain limited in modeling spatial-spectral coupling and weak texture details. To address these issues, this paper presents FSCM, a spectral-information-guided GAN framework. Within FSCM, a frequency-enhanced spatial-spectral state-space generator composed of cascaded FSB units is constructed. Each FSB integrates three complementary components: state-space modeling captures global spatial-spectral dependencies; the frequency enhancement module (FEM) combines multi-level wavelet decomposition and Fourier gating to recover structural contours, directional high-frequency details, and global frequency responses; and the dual-stream hybrid gating module (DGM) integrates deformation-aware sampling with sparse attention to enhance effective local structures and suppress background interference. Additionally, an online semantic segmentation-guided loss is introduced to constrain the generated results, improving semantic consistency in complex road scenes. Experiments show that FSCM outperforms existing infrared colorization methods in visual quality and semantic fidelity. 

---
# Shapley Neuron Values for Continual Learning: Which Neurons Matter Most? 

**Authors**: Mohammad Ali Vahedifar, Abhisek Ray, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15877)  

**Abstract**: Continual learning enables neural networks to learn tasks sequentially without forgetting previously acquired knowledge. However, neural networks suffer from catastrophic forgetting, where learning new tasks degrades performance on earlier ones. We address this problem with Shapley Neuron Valuation (SNV), a principled framework that quantifies Neuron importance in continual learning, grounded in cooperative game theory. SNV selectively freezes important Neurons while keeping others plastic, enabling buffer-free continual learning without expanding architecture. Experiments on ImageNet-1k show that SNV consistently outperforms existing buffer-free methods. In particular, SNV improves accuracy by +2.88% in the class incremental learning and +6.46% in the task incremental learning scenarios compared to the second baseline. 

---
# Access Timing as Scaffolding: A Reinforcement Learning Approach to GenAI in Education 

**Authors**: Janne Rotter, Pau Benazet i Montobbio, Davinia Hernández-Leo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15850)  

**Abstract**: In recent years, generative AI (GenAI) in educational settings has become ubiquitous in students' daily lives, despite its potential to induce over-reliance, metacognitive disengagement, and diminished learning when used unrestrictedly. While most prior research has thus focused on how to pedagogically scaffold its usage, the question of when to allow off-the-shelf GenAI remains understudied and lacks pedagogically grounded empirical investigation. We treat access timing itself as a form of implicit scaffolding and operationalize it through a reinforcement learning (RL) agent that decides when students should access GenAI, with a reward function grounded in metacognitive theory, cognitive load theory, and productive failure. In a mixed-methods controlled lab study with N=105 participants, we compared the agent's effect on learning gains and metacognitive engagement to unrestricted and fully restricted use. Results show that strategically timed GenAI access under the reinforcement learning condition improved objective post-test performance and metacognitive accuracy compared with unrestricted access, while reducing task errors and time on task relative to complete withholding, all without the need for explicit metacognitive prompts or structured scaffolding. However, no between-condition differences emerged on self-reported metacognitive awareness. Overall, timing of GenAI access therefore is a tractable, theoretically grounded, and scalable pedagogical paradigm that improves over completely unrestricted and withheld access, compatible with off-the-shelf tools and potentially low adoption barrier. This opens up a new research area that explores how access timing can be facilitated by educators and implemented in human-AI learning system design. 

---
# RoadmapBench: Evaluating Long-Horizon Agentic Software Development Across Version Upgrades 

**Authors**: Xinbo Xu, Ruihan Yang, Haiyang Shen, Wendong Xu, Bofei Gao, Ruoyu Wu, Kean Shi, Weichu Xie, Xuanzhong Chen, Ming Wu, Jason Zeng, Michael Heinrich, Elvis Zhang, Liang Chen, Kuan Li, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15846)  

**Abstract**: Coding agents are increasingly deployed in real software development, where a single version iteration requires months of coordinated work across many files. However, most existing benchmarks focus predominantly on single-issue bug fixes from Python repositories, with coarse pass/fail evaluation outcomes, and thus fail to capture long-horizon, multi-target development at real engineering scale. To address this gap, we present RoadmapBench, a benchmark of 115 long-horizon coding tasks grounded in real open-source version upgrades across 17 repositories and 5 programming languages. Each task places the agent on a source-version code snapshot and provides a multi-target roadmap instruction requiring it to implement the functionality introduced in the target version, with a median modification of 3,700 lines across 51 files. We conduct a systematic evaluation on thirteen frontier models and find that even the strongest, Claude-Opus-4.7, resolves only 39.1% of tasks, while the weakest achieves merely 5.2%, in stark contrast to existing bug-fix benchmarks, suggesting that long-horizon software development remains a largely unsolved problem. 

---
# GAP: Geometric Anchor Pre-training for Data-Efficient Visuomotor Learning of Manipulation Tasks 

**Authors**: Davide Buoso, Andrea Protopapa, Stefano Di Carlo, Francesca Pistilli, Giuseppe Averta  

**Link**: [PDF](https://arxiv.org/pdf/2605.15836)  

**Abstract**: Learning visuomotor policies from scarce expert demonstrations remains a core challenge in robotic manipulation. A primary hurdle lies in distilling high-dimensional RGB representations into control-relevant geometry without overfitting. While using frozen pre-trained Vision Foundation Models (VFMs) improves data efficiency, it also shifts most task adaptation onto a small spatial pooling module, which can latch onto task-irrelevant shortcuts and lose geometric grounding when finetuned with few data samples. More broadly, pre-trained visual representations used for policy learning have been observed to struggle under even minor scene perturbations, highlighting the need for robustness-oriented inductive biases. We propose Geometric Anchor Pre-training (GAP), a simple, action-free warm-up stage that regularizes the spatial adapter before downstream imitation learning. GAP pre-trains the pooling layer on a lightweight simulated proxy task where object masks are available at no cost, encouraging the adapter to produce keypoints that lie on the object, cover its spatial extent, and remain sharp and repeatable over time. This yields stable geometric anchors that provide a reliable coordinate interface for few-shot policy learning, while keeping the VFM frozen. We evaluate GAP on RoboMimic and ManiSkill under severe data scarcity (15-50 demonstrations) and domain shift. A simple adapter regularized with GAP consistently outperforms stronger attention-based poolers and end-to-end fine-tuning, achieving 62% success on RoboMimic Can with 15 demonstrations (+16% over AFA), 63% on the long-horizon high-precision Tool Hang task with 50 demonstrations, and 61% on ManiSkill StackCube with 30 demonstrations (+11% over full fine-tuning). The proxy stage is lightweight and fully decoupled from downstream tasks, making it practical to reuse across environments and manipulation skills. 

---
# Modeling Music as a Time-Frequency Image: A 2D Tokenizer for Music Generation 

**Authors**: Yuqing Cheng, Xingyu Ma, Guochen Yu, Xiaotao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15831)  

**Abstract**: Autoregressive music generation depends strongly on the audio tokenizer. Existing high-fidelity codecs often use residual multi-codebook quantization, which preserves reconstruction quality but complicates language modeling after sequence flattening, as the residual hierarchy imposes strong sequential dependencies and can amplify error accumulation. We propose BandTok, a generation-oriented 2D Mel-spectrogram tokenizer that represents each frame with Mel-frequency band tokens from a single shared codebook. This design yields a physically interpretable time-frequency token grid with a more independent token structure, making it better suited for autoregressive modeling. BandTok improves reconstruction with a multi-scale PatchGAN objective and EMA codebook updates. We further introduce an autoregressive language model with 2D Rotary Position Embedding (2D RoPE) to preserve temporal and frequency-band structure during generation. Experiments show that BandTok improves over residual-codebook tokenizers and achieves strong results in a data-limited setting. The source code and generation demos for this work are publicly available. 

---
# Toward Natural and Companionable Virtual Agents via Cross-Temporal Emotional Modeling 

**Authors**: Feier Qin, Xiao Li, Yi Zheng, Haibin Huang, Hanyao Wang, Xiaoyu Wang, Yan Lu, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15812)  

**Abstract**: Recent advances in foundation models have enabled conversational agents that aim for sustained companionship rather than mere task completion. Yet most still remain unable to support natural, long-term companion-like interactions, resulting in experiences that feel episodic and inauthentic. We argue that current agents overlooked cross-temporal modeling of agents' social behaviors and internal emotions: generated behaviors rarely influence an agent's emotional state, and emotional states seldom shape subsequent behaviors. We present Cross-Temporal Emotion Modeling (CTEM), a framework that links long-term behavioral history to moment-to-moment emotional expression. CTEM establishes a closed loop where past experiences update an evolving emotional state; this state conditions immediate interactions; and user feedback continually revises both memory and emotional state, enabling reflection and anticipation. We instantiate CTEM as Auri, a companion agent on an instant-messaging platform, and report a 21-day in-the-wild study showing that CTEM shows improvements in perceived naturalness, coherence, and emotional harmony. 

---
# Grokking as Structural Inference: Transformers Need Bayesian Lottery Tickets 

**Authors**: Kai Hidajat, Solden Stoll, Joseph An  

**Link**: [PDF](https://arxiv.org/pdf/2605.15787)  

**Abstract**: Why does a Transformer that has memorized its training set wait thousands of steps before it generalizes? Existing accounts locate this delay in norm minimization, feature emergence, or the late discovery of sparse subnetworks. These explanations capture important parts of the transition, but ignore a constraint unique to attention-based models: if attention discards an informative token, no bounded downstream computation can recover it. We formalize attention as an implicit Bayesian posterior over the task dependency graph and prove that generalization requires two separable conditions: a familiar Goldilocks bound on MLP capacity, coinciding with norm-based theories of grokking, and a novel Bayesian structural condition requiring attention to place sufficient mass on every informative token. This decoupling explains delayed generalization as delayed structural inference. Early in training, the MLP memorizes through unaligned features, drives the cross-entropy loss near zero, and thereby starves attention of structural gradient. Weight decay must then erode memorization before the missing graph becomes learnable, yielding the known inverse-weight-decay delay, which we derive as a structural waiting time. We then prove that this explaining-away delay can be bypassed by a KL-based structural intervention, yielding an inverse-intervention-strength scaling law for the grokking time. Experiments on algorithmic sequence tasks isolate structure from capacity and show that this Bayesian ticket matches or outperforms lottery-ticket transfer. 

---
# A Topology-Aware Spatiotemporal Handover Framework for Continuous Multi-UAV Tracking 

**Authors**: Jianlin Ye, Christos Kyrkou, Panayiotis Kolios  

**Link**: [PDF](https://arxiv.org/pdf/2605.15779)  

**Abstract**: The integration of Unmanned Aerial Vehicles(UAVs) into Intelligent Transportation Systems (ITS) offers synoptic visibility for traffic monitoring, yet scalable deployment is hindered by trajectory fragmentation, where vehicle identity persistence is lost across multi-UAV Fields of View (FOV). While state-of-the-art frameworks excel in optimizing local trajectory extraction and stability for single-drone imagery, they often function as isolated data silos that generate disjointed trajectories, thereby precluding network-level analysis such as Origin-Destination estimation. This paper presents a real-time Multi-Camera Multi-Vehicle Tracking (MCMT) system designed to handle global identity persistence. Addressing the visual ambiguity and computational cost of appearance-based Re-Identification (Re-ID) in nadir views, we introduce a lightweight Topology-Based Spatiotemporal Handover mechanism. We implement a high-throughput parallel pipeline leveraging YOLO11 and ByteTrack to process concurrent 4K streams. Our core contribution is a deterministic queue-based matching algorithm that utilizes geometric overlaps and virtual lane discretization to predictively manage identity handover via FIFO queues. Experimental results on complex urban environments, including intersections and merging traffic, demonstrate a Handover Success Rate (HOSR) of 99.8% in continuous traffic flows, significantly outperforming Re-ID baselines (74.1%) while validating edge deployment feasibility. The source code is available at this https URL. 

---
# Lamarckian Inheritance in Dynamic Environments: How Key Variables Affect Evolutionary Dynamics 

**Authors**: K. Ege de Bruin, Kyrre Glette, Kai Olav Ellefsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15769)  

**Abstract**: The co-optimization of a robot's body and brain presents a coupled challenge: the morphology constrains which control strategies are effective, while the control determines how well the morphology performs. To address this, we combine morphology optimization as evolution with controller optimization as lifetime learning, utilizing Lamarckian inheritance to transfer learned controller parameters from parent to offspring. In dynamic environments, existing literature presents conflicting evidence: while traditional evolutionary theory often suggests Lamarckian inheritance lacks benefit, recent studies in evolutionary robotics indicate it can improve performance. We hypothesize that this is because previous works have not included all relevant variables with dynamic environments. In this work, we show that the benefit of Lamarckian inheritance depends on two variables: how conflicting the environmental changes are to robot control, and the predictability of those changes for the robotic agent. Using virtual soft robots and two different learning approaches, Bayesian optimization and reinforcement learning, we show that Lamarckian inheritance only underperforms Darwinian inheritance when the changes are both conflicting and unpredictable. We find that adding a sensor to detect environmental changes restores the benefits for Lamarckian inheritance in conflicting environments, by allowing robotic agents to predict the need for a different behavior, thereby generalizing their control. 

---
# GRASP: Learning to Ground Social Reasoning in Multi-Person Non-Verbal Interactions 

**Authors**: Junho Kim, Xu Cao, Houze Yang, Bikram Boote, Ana Jojic, Fiona Ryan, Bolin Lai, Sangmin Lee, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2605.15764)  

**Abstract**: Understanding social interactions requires reasoning over subtle non-verbal cues, yet current multimodal large language models (MLLMs) often fail to identify who interacts with whom in multi-person videos. We introduce GRASP, a large-scale social reasoning dataset that connects high-level social QA with fine-grained gaze and deictic gesture events. GRASP contains 290K question--answer pairs over 46K videos totaling 749 hours, organized by a 16-category taxonomy spanning gaze, gesture, and joint gaze--gesture reasoning, together with GRASP-Bench for evaluation. Unlike prior resources that focus on either isolated cues or high-level social QA, GRASP builds questions from identity-consistent gaze trajectories, deictic gestures, and their joint compositions into social events. Moreover, we propose Social Grounding Reward (SGR), a learning signal that uses these social events to encourage models to reason about the participants involved in each interaction. Experiments show that SGR improves performance on GRASP-Bench while maintaining zero-shot performance on related social video QA benchmarks. 

---
# CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs 

**Authors**: Kamil Guttmann, Zofia Fraś, Artur Nowakowski, Krzysztof Jassem  

**Link**: [PDF](https://arxiv.org/pdf/2605.15763)  

**Abstract**: Current state-of-the-art Quality Estimation (QE) in machine translation relies on massive, proprietary LLMs, raising data privacy concerns. We demonstrate that smaller, open-source LLMs (<30B parameters) are a viable, cost-effective and privacy-preserving alternative. Using a single-pass prompting strategy, our models simultaneously generate quality scores, MQM error annotations, suggested error corrections, and full post-editions. Our analysis shows these models achieve highly competitive system-level correlations with human judgments that outperform traditional neural metrics, fine-tuned models, and human inter-annotator agreement, effectively approximating the capabilities of much larger proprietary LLMs. 

---
# BiomedAP: A Vision-Informed Dual-Anchor Framework with Gated Cross-Modal Fusion for Robust Medical Vision-Language Adaptation 

**Authors**: Huanyang Tong, Kai Liu, Fangjun Kuang, Huiling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15736)  

**Abstract**: Biomedical Vision--Language Models (VLMs) have shown remarkable promise in few-shot medical diagnosis but face a critical bottleneck: \textit{fragility to prompt variations}.Existing adaptation frameworks typically optimize visual and textual prompts as independent streams, relying on ideal ``Golden Prompts''. In clinical reality, where descriptions are often noisy and heterogeneous, this modality isolation leads to unstable cross-modal alignment.
To address this, we propose BiomedAP, a vision-informed dual-anchor framework with gated cross-modal this http URL enforces synergistic alignment through two mechanisms: (1) Gated Cross-Modal Fusion, which enables layer-wise interaction between modalities, acting as a dynamic noise regulator to suppress irrelevant textual cues; and (2) a Dual-Anchor Constraint that regularizes learnable prompts toward stable semantic centroids derived from both expert templates (High Anchors) and few-shot visual prototypes (Low Anchors).
Extensive experiments across 11 benchmarks demonstrate that BiomedAP consistently surpasses baselines, achieving competitive few-shot accuracy and markedly enhanced robustness under prompt perturbations.
Our code is available at: this https URL.
Keywords: Vision-Language Models; Prompt Learning; Parameter-Efficient Fine-Tuning; Few-shot Learning 

---
# UAM: A Dual-Stream Perspective on Forgetting in VLA Training 

**Authors**: Jianke Zhang, Yuanfei Luo, Yucheng Hu, Xiaoyu Chen, Yanjiang Guo, Ziyang Liu, Hongbin Xu, Tian Lan, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15735)  

**Abstract**: Vision--language--action (VLA) models are typically built by fine-tuning a pretrained vision--language model (VLM) on action data. However, we show that this standard recipe systematically erodes the VLM's multimodal competence, a side effect we call the embodiment tax. But do VLAs have to forget? Inspired by the two-stream organization of biological vision, we trace this degradation to a structural bottleneck: current VLAs ask a single encoder to support both language-grounded semantics and control-relevant visual features, whereas biological vision separates recognition and visuomotor control into distinct pathways. Building on this view, we propose the Unified Action Model (UAM), which adds a parallel Dorsal Expert, an analog of the brain's dorsal pathway. To make the Dorsal Expert an effective second pathway and reduce the control-learning burden on the VLM, we initialize it from a pretrained generative model and train it with a mid-level reasoning objective that predicts visual dynamics. This design allows us to train the whole VLA end-to-end on action data alone: with no parameter freezing, no gradient stopping, and no auxiliary VL co-training, UAM retains over $95\%$ of the underlying VLM's multimodal capability and at the same time achieves the highest average success rate among baselines on a variety of manipulation tasks that probe out-of-distribution generalization, including unseen objects, novel object--target compositions, and instruction variation. Together, these results suggest that semantic preservation in VLAs can emerge from architectural separation itself, rather than being enforced by frozen weights or auxiliary data replay, and that this preserved semantic capability can naturally transfer from VLMs to semantic generalization in actions. 

---
# Structure Abstraction and Generalization in a Hippocampal-Entorhinal Inspired World Model 

**Authors**: Tianqiu Zhang, Muyang Lyu, Xiao Liu, Si Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15733)  

**Abstract**: Humans abstract experiences into structured representations to facilitate pattern inference and knowledge transfer. While the hippocampal-entorhinal (HPC-MEC) circuit is known to represent both spatial and conceptual spaces, the mechanisms for concurrently extracting abstract structures from continuous, high-dimensional dynamics remain poorly understood. We propose a brain-inspired hierarchical model that simultaneously infers latent transitions and constructs a predictive visual world model. Our architecture employs an inverse model for structural extraction alongside an HPC-MEC coupling model that dissociates relational structures (MEC) from integrated episodic scenes (HPC). Using primitive transformation dynamics as a benchmark, we demonstrate the model's capacity for structural abstraction. By leveraging velocity-driven path integration, the framework enables robust prediction and structural reuse across diverse contexts, thereby achieving structural generalization. This work provides a novel computational framework for understanding how brain-inspired, self-supervised learning of world models facilitates the acquisition of reusable abstract knowledge. 

---
# DecomPose: Disentangling Cross-Category Optimization Contention for Category-Level 6D Object Pose Estimation 

**Authors**: Yifan Gao, Lu Zou, Zhangjin Huang, Guoping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15728)  

**Abstract**: Category-level 6D object pose estimation is typically formulated as a multi-category joint learning problem with fully shared model parameters. However, pronounced geometric heterogeneity across categories entangles incompatible optimization signals in shared modules, resulting in gradient conflicts and negative transfer during training. To address this challenge, we first introduce gradient-based diagnostics to quantify module-level cross-category contention. Building on results of diagnostics, we propose DecomPose, a difficulty-aware decomposition framework that mitigates optimization contention via: (1) difficulty-aware gradient decoupling, which groups categories using a data-driven difficulty proxy and routes each instance to a group-specific correspondence branch to isolate incompatible updates; and (2) stability-driven asymmetric branching, which assigns higher-capacity branches to structurally simple categories as stable optimization anchors while constraining complex categories with lightweight branches to suppress noisy updates and alleviate negative transfer. Extensive experiments on REAL275, CAMERA25, and HouseCat6D demonstrate that DecomPose effectively reduces cross-category optimization contention and delivers superior pose estimation performance across multiple benchmarks. 

---
# DiLA: Disentangled Latent Action World Models 

**Authors**: Tianqiu Zhang, Muyang Lyu, Yufan Zhang, Fang Fang, Si Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15725)  

**Abstract**: Latent Action Models (LAMs) enable the learning of world models from unlabeled video by inferring abstract actions between consecutive frames. However, LAMs face a fundamental trade-off between action abstraction and generation fidelity. Existing methods typically circumvent this issue by using two-stage training with pre-trained world models or by limiting predictions to optical flow. In this paper, we introduce DiLA, a novel Disentangled Latent Action world model that aims to resolve this trade-off via content-structure disentanglement. Our key insight is that disentanglement and latent action learning are co-evolving: the predictive bottleneck inherent in latent action learning serves as a driving force for disentanglement, compelling the model to distill spatial layouts into the structure pathway while offloading visual details to a separate content pathway for generation. This synergy yields a continuous, semantically structured latent action space without compromising generative quality. DiLA achieves superior results in video generation quality, action transfer, visual planning, and manifold interpretability. These findings establish DiLA as a unified framework that simultaneously achieves high-level action abstraction and high-fidelity generation, advancing the frontier of self-supervised world model learning. 

---
# Bidirectional Fusion Guided by Cardiac Patterns for Semi-Supervised ECG Segmentation 

**Authors**: Jeonghwa Lim, Minje Park, Sunghoon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15722)  

**Abstract**: Accurate delineation of electrocardiogram (ECG), the segmentation of meaningful waveform features, is crucial for cardiovascular diagnostics. However, the scarcity of annotated data poses a significant challenge for training deep learning models. Conventional semi-supervised semantic segmentation (SemiSeg) methods primarily focus on consistency from unlabeled data, underutilizing the information exchange possible between labeled and unlabeled sets. To address this, we introduce CardioMix, a framework built on a bidirectional CutMix strategy guided by cardiac patterns for ECG segmentation. This approach enriches the labeled set with realistic variations from unlabeled data while simultaneously applying stronger supervisory signals to the unlabeled set, as the cardiac pattern-guided mixing ensures all augmented samples remain physiologically meaningful. Our framework is designed as a plug-and-play module, demonstrating high compatibility with various SemiSeg algorithms. Extensive experiments on SemiSegECG, a public multi-dataset benchmark for ECG delineation, demonstrate that CardioMix consistently outperforms existing CutMix-based fusion strategies across diverse datasets and labeled ratios as a plug-and-play module compatible with various SemiSeg algorithms. 

---
# Position: Early-Stage Quality Assurance in Annotation Pipelines Is More Cost-Effective Than Late-Stage Validation 

**Authors**: Sunil Kothari, Sumukha Sharma Thoppanahalli Chandramouli, Naman Khandelwal, Parth Kulshreshtha, Ashi Jain, Kriti Banka, Tanuja Chintada, Venkata Triveni, Gulipalli Praveen Kumar, Manish Mehta, Tao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15714)  

**Abstract**: This position paper argues that the machine learning community should prioritize early-stage quality assurance in annotation pipelines over the prevailing practice of late-stage validation. Data quality bottlenecks increasingly limit foundation model improvement, yet quality assurance research focuses almost exclusively on validation methods rather than validation timing. When validation occurs, not merely what methods are employed, fundamentally determines both error rates and annotation costs. This temporal neglect is puzzling given the well-established "shift-left" principle from software engineering, where empirical studies demonstrate 4--100x cost multipliers for defects detected in later stages (Boehm, 1981; Shull et al., 2002). Annotation pipelines exhibit analogous dynamics: errors caught before annotation begins cost a fraction of those discovered after review cycles complete. We propose a taxonomy of three QA trigger points, namely pre-annotation (T0), post-annotation (T1), and post-review (T2), that decompose annotation workflows into discrete validation opportunities. A parametric error-propagation model formalizes when timing affects final error rates versus only economics, making timing a measurable design variable rather than a configuration afterthought. A survey of 47 recent papers reveals that only 4% report when validation occurs, a striking gap given timing's demonstrated impact in adjacent fields. Without explicit attention to QA timing, the community risks optimizing validation methods while ignoring the structural variable that may matter most. Acting on this position requires three steps: researchers should report QA timing configurations alongside validation methods; annotation platforms should expose timing as a first-class parameter; and the community should run controlled experiments that measure stage-specific detection rates directly. 

---
# Learning Dynamic Pick-and-Place for a Legged Manipulator 

**Authors**: Moonkyu Jung, Jiseong Lee, Zhengmao He, Donghoon Youm, Juhyeok Mun, HyeongJun Kim, Hyunsik Oh, Donghyuk Choi, Jungwoo Hur, Jie Song, Jemin Hwangbo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15713)  

**Abstract**: Legged manipulators extend robotic capabilities beyond static manipulation by integrating agile locomotion with versatile arm control. However, achieving precise manipulation while maintaining coordinated locomotion remains a major challenge. This work presents a hierarchical reinforcement learning framework for dynamic pick-and-place tasks using a quadruped equipped with a 6-DOF robotic arm. The framework incorporates an explicit mass estimation module enabling adaptive whole-body control for objects with varying weights. In simulation, the system achieves an 86.05% success rate with payloads up to 2.3 kg. The approach is further validated through real-world experiments across six representative scenarios with controlled variations in object physical properties (size and mass) and task heights. Specifically, within a wide vertical workspace ranging from ground level to 1.1~m-high tabletops, the system demonstrates an average success rate of 73.3% for payloads up to 1.3 kg, with an average execution time of 4.06 s. Unlike prior works that handle lightweight objects and execute pick-and-place motions with slow, piecewise motions, the proposed framework exploits concurrent locomotion and manipulation for dynamic, continuous execution. These results demonstrate the potential of quadrupedal mobile manipulators for adaptive, whole-body pick-and-place with heavier payloads and extended workspaces. 

---
# Feedback World Model Enables Precise Guidance of Diffusion Policy 

**Authors**: Tuo An, Jindou Jia, Gen Li, Jingliang Li, Chuhao Zhou, Pengfei Liu, Bofan Lyu, Jiaqi Bai, Xinying Guo, Geng Li, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15705)  

**Abstract**: World models aim to improve robotic decision making by predicting the consequences of actions. However, in practice, their predictions often become unreliable once the robot encounters states outside the training distribution, limiting their effectiveness at deployment. We observe that execution itself provides a natural but underutilized signal: after each action, the robot directly observes the true next state, revealing the mismatch between predicted and actual outcomes. Building on this insight, we propose feedback world model, a new paradigm that closes the loop between prediction and observation at inference time. Instead of treating the world model as a static open-loop predictor, our method maintains a lightweight feedback state that is updated online to iteratively correct future predictions, compensating for model errors using real-time observations without additional training data or parameter updates. We show that this process can be interpreted as a latent-space observer and admits convergence guarantees under mild conditions. We further introduce action-aware guidance to better translate corrected predictions into control by emphasizing action-controllable components while suppressing irrelevant variations. Experiments on LIBERO-Plus, Robomimic, and real-world manipulation tasks demonstrate that our method substantially improves both prediction accuracy and policy performance under distribution shift. In particular, it reduces world model prediction error by up to 76.4% and improves out-of-distribution (OOD) success rate by 30%. These results show that incorporating real-time feedback at inference time provides a simple yet powerful alternative to static world modeling. 

---
# H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure 

**Authors**: Jiawei Yu, Yixiang Fang, Xilin Liu, Yuchi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15701)  

**Abstract**: Memory data are ubiquitous in Large Language Model (LLM)-based agents (e.g., OpenClaw and Manus). A few recent works have attempted to exploit agents'memory for improving their performance on the question-answering (QA) task, but they lack a principled mechanism for effectively modeling how memory data evolves over time and retrieving memory data effectively, leading to poor performance in memory utilization. To fill this gap, we present H-Mem, a novel memory mechanism via a hybrid structure that can not only effectively model the evolution of agent memory over a long period of time, but also provide an efficient memory retrieval approach. Particularly, H-Mem builds a temporal and semantic tree structure that allows the short-term memory data to evolve progressively into long-term memory data, where the latter provides summarized information about the former, while simultaneously constructing a knowledge graph to capture the relationships between entities in memory. Moreover, it offers an effective memory retrieval approach by exploiting the hybrid structure of the tree and graph structures. Extensive experiments on three agent memory benchmarks show that H-Mem achieves state-of-the-art performance on the QA task. 

---
# $α$-TCAV: A Unified Framework for Testing with Concept Activation Vectors 

**Authors**: Ekkehard Schnoor, Jawher Said, Malik Tiomoko, Wojciech Samek, Alexander Jung  

**Link**: [PDF](https://arxiv.org/pdf/2605.15688)  

**Abstract**: Concept Activation Vectors (CAVs) are a fundamental tool for concept-based explainability in deep learning, yet their practical utility is limited by statistical instability. We analyze the stochastic nature of CAVs and the Testing with CAVs (TCAV) method, deriving the distributions of major CAV classes including PatternCAV, FastCAV, and ridge regression-based CAVs. We then identify a fundamental flaw in the standard TCAV score: its reliance on a discontinuous indicator function induces non-decaying variance in critical regimes. To address this, we introduce $\alpha$-TCAV, a generalized framework that replaces the indicator with a parameterized smooth function, yielding a unified probabilistic formulation that subsumes both TCAV and Multi-TCAV. We characterize the induced distributions of sensitivity scores and different TCAV variants, showing that established state-of-the-art choices lack theoretical justification. We provide principled guidance on tuning the parameter in $\alpha$-TCAV -- either to imitate Multi-TCAV at substantially lower computational cost, or to obtain a calibrated Bayes-optimal probabilistic measure of a concept's influence. Finally, our analysis yields practical recommendations that challenge established routines: most notably, allocating the full sampling budget to a single CAV rather than splitting it across several. 

---
# ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models 

**Authors**: Jiahui Guang, Yingjie Zhu, Cuiyun Gao, Haiyan Wang, Jing Li, Di Shao, Zhaoquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15687)  

**Abstract**: Multimodal large language models (MLLMs) may memorize sensitive cross-modal information during pretraining, making machine unlearning (MU) crucial. Existing methods typically evaluate unlearning effectiveness based on output deviations, while overlooking the generation quality after unlearning. This can easily lead to hallucinated or rigid responses, thereby affecting the usability and safety of the unlearned model. To address this issue, we propose ASRU, a controllable multimodal unlearning framework that incorporates generation quality as a core evaluation objective. ASRU first induces initial refusal behavior through activation redirection, and then optimizes fine-grained refusal boundaries using a customized reward function, thereby achieving a better trade-off between target knowledge unlearning and model utility. Experiments on Qwen3-VL show that ASRU significantly improves unlearning effectiveness (+24.6%) on average and generation quality (5.8x) on average while effectively preserving model utility, using only a small amount of retained supervision data. 

---
# Interaction-Aware Influence Functions for Group Attribution 

**Authors**: Jaeseung Heo, Kyeongheung Yun, Youngbin Choi, Sehyun Hwang, Jungseul Ok, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.15675)  

**Abstract**: Influence functions approximate how removing a training example changes a quantity of interest, called the target function, such as a held-out loss. To estimate the influence of a group of examples, the standard practice is to sum the individual influences of its members. However, this sum does not capture how examples jointly affect the target: a pair of examples may be redundant or complementary, but the sum cannot distinguish these cases. We propose an interaction-aware influence function that characterizes how interactions between examples influence the target. By expanding the target to second order around the trained parameters, we obtain an estimator that augments the standard sum with a pairwise interaction term that captures the alignment between two examples' effects on the target. We empirically evaluate our estimator in two settings. First, on six dataset-model pairs spanning logistic regression, MLPs, and ResNet-9, our estimator tracks leave-group-out retraining substantially better than first-order influence across all settings. Second, when used as a greedy selection rule for instruction-tuning data on Llama-3.1-8B, it beats prior influence-based and representation-similarity baselines on five of seven downstream tasks, in a regime where standard influence-based selection underperforms random selection. 

---
# VLMs Trace Without Tracking: Diagnosing Failures in Visual Path Following 

**Authors**: Hyesoo Hong, Minsoo Kim, Wonje Jeung, Sangyeon Yoon, Dongjae Jeon, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.15672)  

**Abstract**: Vision-language models (VLMs) achieve strong performance on multimodal benchmarks, but may still lack robust control over basic visual operations. We study \textit{line tracing}, where a model must follow a selected visual path through successive local continuations. To isolate this ability, we design controlled tracing tasks that introduce nearby competitors while reducing semantic and topological ambiguity such as crossings and overlaps. Across these tasks, even state-of-the-art VLMs frequently lose the target path and switch to nearby alternatives, especially when those alternatives look locally similar to the target. Behavioral interventions and internal analyses indicate that these failures arise from local competition: nearby similar distractors pull the model away from the true continuation. Standard remedies do not remove this bottleneck: model-size scaling provides only limited gains, reasoning partially compensates through costly substitute strategies, and explicit tracing instructions fail to recover stable path following. Finally, tests on tangled-cable scenes and metro maps with richer visual complexity show that the same path-switching failure persists beyond our controlled settings. 

---
# VAGS: Velocity Adaptive Guidance Scale for Image Editing and Generation 

**Authors**: Yan Luo, Ahmadou Aidara, Jingyi Lu, Jeremy Moebel, Kai Han, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15661)  

**Abstract**: Classifier-free guidance (CFG) is the primary control over how strongly text semantics move a flow-based sampler, yet standard practice holds its scale fixed across the entire ODE trajectory. This is a fundamental mismatch: early steps are noise-dominated and carry weak semantic signal, while late steps commit image structure and demand stronger directional commitment; more critically, the value of any guidance strength depends on whether the guided velocity is consistent with the model's current dynamics or working against them. We propose \textit{Velocity-Adaptive Guidance Scale} (VAGS), a training-free replacement that multiplies the nominal scale by a bounded factor combining a temporal signal-level term with the cosine similarity between task-relevant velocity fields. For inversion-free editing, VAGS measures the alignment between source- and target-guided velocities, so edit strength at each step reflects local compatibility between preservation and transformation. For generation, VAGS-Gen uses the alignment between unconditional and conditional velocities as the analogous signal. Neither variant requires fine-tuning, auxiliary networks, or extra forward passes, and fixed CFG is recovered as a special case. On PIE-Bench and DIV2K for editing, and COCO17, CUB-200, and Flickr30K for generation, VAGS consistently improves structural fidelity and generation quality over fixed CFG and recent training-free guidance variants. The code is publicly available at this https URL. 

---
# TFZ-Tree: An Ultra-Lightweight Waveform Classification Framework for Resource-Constrained Devices 

**Authors**: Hao Wang, Kuang Zhang, Yonggang Chi, Tianqi Zhao, Yanbo Fu, Jiaxing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15656)  

**Abstract**: Under the trend of multi-waveform coexistence in 6G IoT, intelligent receivers must first identify physical-layer waveform types before performing correct demodulation and resource scheduling. However, existing signal identification research largely focuses on symbol-level modulation classification. Research directly targeting physical-layer waveform types (e.g., OFDM, OTFS, LoRa) is not only extremely scarce but also heavily reliant on deep neural networks and complex time-frequency transforms, making deployment on resource-constrained terminals difficult. Symbol modulation classification methods themselves cannot circumvent the prerequisite of ``waveform identification first.''
To address this dual gap, we propose an ultra-lightweight waveform classification framework based on time-frequency multidimensional features with a cooperative Z-test tree (ZTree). The framework employs low-complexity time-domain feature extraction, and the classification backend adopts a ZTree optimized by Z-statistical testing, which uses hypothesis testing confidence to automatically control decision tree splitting and size, ensuring efficient execution on resource-limited processors. Tested on ten 6G candidate waveforms including OFDM, OTFS, DSSS, LoRa, and NB-IoT, the method achieves 99.5\% average accuracy under AWGN and 87.4\% under TDL-C multipath channels, with main confusion between OTFS and LoRa. Implemented in C on an x86 platform, single inference latency is under 4~ms. To the best of our knowledge, this is the first work achieving real-time recognition of ten IoT waveform types. Future work will target deployment acceleration on embedded MCUs. Code and dataset are open-sourced at: this https URL. 

---
# Bridging Silicon and the Hippocampus: Algebro-Deterministic Memory "VaCoAl" as a Substrate for Vector-HaSH and TEM 

**Authors**: Hiroyuki Chuma, Kanji Otsuka, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2605.15652)  

**Abstract**: Vector-HaSH and the Tolman-Eichenbaum Machine (TEM) propose that the hippocampal-entorhinal circuit factorizes content from a prestructured grid-cell scaffold and supports compositional memory via ripple-mediated replay. Human iEEG shows that hippocampal sharp-wave ripples (SWRs) gate episodic recall, ripple-locked cortical reactivation recapitulates encoding-time patterns, and multi-hop replay fidelity decays multiplicatively along sequence length. These literatures have advanced in parallel without a shared algebraic object. We show that VaCoAl, an algebro-deterministic hyperdimensional memory architecture built from Galois-field LFSRs, supplies that object. Specifically, deterministic Galois-field diffusion provides a substrate-level alternative to Vector-HaSH's random scaffold-to-hippocampus projection that satisfies the same quasi-orthogonality requirement, with matched second-moment statistics, stronger avalanche behavior, and bit-exact reproducibility. The path-integral Confidence Ratio $CR_2$, the product of per-step $CR_1$ values along an $n$-hop chain, is the natural functional form for multi-hop replay-fidelity decay under conditional independence of per-step reactivation, providing the first algebraically tractable model of reported multiplicative decay. STDP-like path selection in VaCoAl follows from architectural demands -- similarity preservation, compositional reversibility, and bounded-frontier search -- that also constrain hippocampal computation. We further argue that VaCoAl operating regimes share architectural commitments with the EC--CA3 and EC--DG--CA3 pathways, motivating an energy-capacity-plasticity reading of why both are conserved across $>$520 Myr of evolution and primate dentate-gyrus elaboration. We prove formal correspondences, derive testable iEEG predictions, and bridge computational neuroscience and hyperdimensional engineering. 

---
# Sharp Spectral Thresholds for Logit Fixed Points 

**Authors**: Tongxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15651)  

**Abstract**: Softmax feedback systems are a common mathematical core of entropy-regularized reinforcement learning, logit game dynamics, population choice, and mean-field variational updates. Their central stability question is simple: when does a self-reinforcing softmax system produce a unique and globally predictable outcome? Classical theory gives a conservative answer. By treating softmax as a unit-scale response, it certifies stability only in a strongly randomized regime.
We prove that the classical approach misses an entire stable regime and does not identify the point at which the qualitative change truly occurs. For finite-dimensional affine logit systems, the sharp dimension-free Euclidean threshold is $$\beta\|\Pi W\Pi\|_{\mathcal T\to\mathcal T}<2,$$ rather than the previously used condition, which certifies stability only while the softmax system remains safely over-regularized. Our theorem fills the previously missing pre-bifurcation regime, extending stability guarantees for affine softmax feedback systems to reward-responsive yet globally predictable systems. It enlarges the certified stability boundary for these systems and identifies where the model genuinely undergoes a phase transition. 

---
# Latent Video Prediction Learns Better World Models 

**Authors**: Ali J Alrasheed, Aryan Yazdan Parast, Basim Azam, James Bailey, Naveed Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2605.15618)  

**Abstract**: Self-supervised video models are increasingly framed as world models, yet their evaluation remains largely confined to a single top-1 accuracy score on clean benchmarks. This leaves a major gap in comprehending their potential as world models. We present the first systematic study addressing this gap, analyzing four matched-capacity frontier video foundation models, V-JEPA 2.1, V-JEPA 2, VideoPrism, and VideoMAEv2, across five robustness axes relevant to their deployment as video world models: feature discriminability, corruption robustness, fine-grained discrimination, occlusion robustness, and sensitivity to temporal direction. Our evaluations establish that across all five axes, latent-prediction models form a distinct and consistent profile. They degrade more gracefully under pixel corruption, preserve usable class structure rather than mere geometric stability under occlusion, capture fine-grained physical contact cues without reconstructing pixels, and uniquely encode the arrow of time. These advantages can even survive task adaptation: a frozen V-JEPA 2 backbone with a lightweight attentive probe outperforms a fully fine-tuned VideoMAE and a supervised TimeSformer on corruption and occlusion robustness. Our extensive results offer concrete new evidence in favor of latent prediction for robust world modeling. 

---
# A Few GPUs, A Whole Lotta Scale: Faithful LLM Training Emulation with PrismLLM 

**Authors**: Shaoke Xi, ChonLam Lao, Boyi Jia, Jiaqi Gao, Zhipeng Zhang, Jiamin Cao, Brian Sutioso, Erci Xu, Minlan Yu, Kui Ren, Yong Li, Zhengping Qian, Ennan Zhai, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.15617)  

**Abstract**: Large language model (LLM) training today runs on clusters spanning thousands of GPUs. While this scale enables rapid model advances, developing, debugging, and performance-tuning the training framework inevitably becomes complex and costly. This is because engineers often need to reproduce production behaviors to diagnose failures or evaluate optimizations, thereby demanding frequent and even exclusive access to production-scale clusters -- which becomes increasingly hard given that the majority of GPUs are already committed to production workloads. Simulation relies on complex performance models that are difficult to maintain, and downscaled experiments often fail to capture scale-dependent behaviors.
We present PrismLLM to decouple large-scale execution from the need to access large clusters, enabling engineers to run and observe ranks of interest under faithful large-scale behavior using only a few GPUs. PrismLLM constructs a high-fidelity execution graph via a slicing-based approach that captures computation, communication, and dependencies of the target scale. Then, PrismLLM performs hybrid emulation where selected ranks execute the original program while the remaining ranks are replayed as virtual participants.
Experiments on large-scale LLM training workloads show that PrismLLM accurately reproduces performance and memory behavior, achieving only 0.58\% average error in iteration time and less than 0.01\% error in peak GPU memory usage. PrismLLM can emulate clusters of up to 8192 GPUs using fewer than 1\% of the physical GPUs required by the original deployment. 

---
# Offline Reinforcement Learning with Universal Horizon Models 

**Authors**: Hojun Chung, Junseo Lee, Songhwai Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.15603)  

**Abstract**: Model-based reinforcement learning (RL) offers a compelling approach to offline RL by enabling value learning on imagined on-policy trajectories. However, it often suffers from compounding errors due to repeated model inference on self-generated states. While geometric horizon models (GHM) alleviate this issue through direct prediction over a discounted infinite-horizon future, they remain challenged in accurately modeling distant future states. To this end, we introduce universal horizon models (UHM), a generalization of GHM that directly predicts future states under arbitrary horizons. Leveraging this flexibility, we propose a scalable value learning method that employs a winsorized horizon distribution to stabilize training by capping excessively large horizons. Experimental results on 100 challenging OGBench tasks demonstrate that the proposed method outperforms competitive baselines, particularly on tasks with highly suboptimal datasets and those requiring long-horizon reasoning. Project page: this https URL 

---
# Pretraining Objective Matters in Extreme Low-Data FGVC: A Backbone-Controlled Study 

**Authors**: Alexander Hackett, Srikanth Thudumu, Ginny Fisher, Mahule Roy, Aisha Sartaj, Jason Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2605.15599)  

**Abstract**: Extreme low-data fine-grained classification is common in expert domains where labeling is expensive, yet practitioners still need principled guidance for selecting pretrained encoders. We study emerald inclusion grading with a custom dataset of labeled images across three classes and ask: under matched backbone capacity, how does pretraining objective affect downstream representation quality? We compare four frozen ViT-B/16 encoders trained with supervised classification, contrastive learning (SigLIP2), masked reconstruction (MAE), and self-distillation (DINOv3), and evaluate them with leave-one-out cross-validation using linear and nonlinear probes. To control statistical noise in the low-N regime, we use permutation testing (N=1000) on macro one-vs-rest AUC. Supervised and contrastive encoders provide the strongest linear separability (logistic AUC: 0.768 and 0.735; SVM AUC: 0.739 and 0.697), while MAE improves under nonlinear probes (XGBoost AUC: 0.713). We find that DINOv3 underperforms across probe families in this domain. These results support a practical recommendation for extreme low-data FGVC: prioritize margin-enforcing pretraining objectives when data scarcity restricts probing to linear decision rules, and consider reconstruction-style encoders when nonlinear classifiers are feasible given dataset constraints. 

---
# Embracing Biased Transition Matrices for Complementary-Label Learning with Many Classes 

**Authors**: Tan-Ha Mai, Chao-Kai Chiang, Han-Hwa Shih, Gang Niu, Masashi Sugiyama, Hsuan-Tien Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.15586)  

**Abstract**: Complementary-label learning (CLL) is a weakly supervised paradigm where instances are labeled with classes they do not belong to. Despite a decade of research, CLL methods remain competitive mainly on 10-class classification, with scaling to large label spaces continuing to be an enduring bottleneck. This limitation stems from the common assumption of uniform label generation in traditional methods, which fatally dilutes the learning signal in many-class settings. In this paper, we demonstrate that this long-standing barrier can be overcome by deliberately designing a biased (non-uniform) generation process that restricts complementary labels to a subset of classes. This finding motivates us to propose Bias-Induced Constrained Labeling (BICL), a principled framework spanning data collection to training that leverages this bias. BICL enables effective learning on CIFAR-100 and TinyImageNet-200, achieving more than sevenfold accuracy improvements over traditional methods. Our findings establish a new trajectory for making CLL feasible for many classes in real-world applications. 

---
# Detecting Privilege Escalation in Polyglot Microservices via Agentic Program Analysis 

**Authors**: Penghui Li, Hong Yau Chong, Yinzhi Cao, Junfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15569)  

**Abstract**: Microservices are widely adopted in modern cloud systems due to their scalability and fault tolerance. However, microservice architectures introduce significant complexity in privilege and permission control, creating risks of privilege escalation where attackers can gain unauthorized access to resources or operations. Detecting such vulnerabilities is challenging due to complex cross-service interactions, polyglot codebases, and diverse privileged operations and permission checks. We present Neo, an agentic program analysis framework that combines large language models (LLMs) with classic program analysis to address these challenges. Neo leverages an LLM-based agent that dynamically generates analysis plans, adapts code search strategies, and validates semantics. We develop code search primitives that enable Neo to perform scalable and flexible code exploration across services and languages. We evaluated Neo on 25 open-source microservice applications spanning 7 programming languages and 6.2 million lines of code. Neo uncovered 24 zero-day privilege escalation vulnerabilities and achieved 81.0% precision and 85.0% recall on a ground-truth dataset. Compared to existing program analysis and agentic solutions, Neo demonstrated significant improvements in both detection accuracy and scalability. We further showcased Neo's extensibility by applying it to other application domains and vulnerability types, uncovering 18 additional zero-day vulnerabilities. 

---
# AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs 

**Authors**: Haizhong Zheng, Yizhuo Di, Jiahui Wang, Shuowei Jin, Xueshen Liu, Yongji Wu, Z. Morley Mao, Ion Stoica, Jiawei Zhao, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15565)  

**Abstract**: Reinforcement learning (RL) is increasingly used to improve the reasoning, coding, and tool-use capabilities of large language models, but agentic RL remains prohibitively expensive. Scaling RL to agentic LLMs requires supporting complex workloads, including multi-policy collaborative training, while efficiently using elastic, heterogeneous, and cross-region compute resources. Existing LLM RL systems support some of these capabilities, but each new extension often requires dedicated system engineering. This burden arises from trainer-centered control architectures and the lack of principled abstractions for RL system components. To address these limitations, we propose AstraFlow, a dataflow-oriented RL system that replaces conventional trainer-centered control with principled component abstractions. In AstraFlow, rollout services, dataflow management, and training are decoupled into autonomous components, enabling the system to natively support complex multi-policy agentic RL workloads and efficiently exploit diverse compute resources. We evaluate AstraFlow across math, code, search, and AgentBench workloads, showing that the same system supports multi-policy training, elastic scaling, heterogeneous cross-region execution, and composable data algorithms without system-level code changes. In multi-policy collaborative training, AstraFlow achieves comparable or better accuracy than existing RL systems while speeding up training time by 2.7x. 

---
# Operator-Controlled 6G: From Connectivity Infrastructure to Guaranteed Digital Services 

**Authors**: David Soldani  

**Link**: [PDF](https://arxiv.org/pdf/2605.15553)  

**Abstract**: Sixth-generation mobile networks (6G) are approaching a structural inflection point. Five generations of vendor-led architectures have left operators procuring and operating networks they do not own, on platforms they cannot modify, with AI layers they cannot audit. This paper argues that 6G must reverse this trajectory by reordering operator priorities: Control First, Customer First, Business First, Operations First, and Technology Last. Technology should serve operator control, customer outcomes, monetizable guarantees, and software-driven operations, not dictate this http URL contributions operationalize this thesis. The 6G Control Compact defines a three-layer ownership taxonomy--own, federate, and consume--that allocates architectural sovereignty according to strategic value. The Guarantee Economy defines a five-tier, outcome-priced commercial model that converts operator control into enforceable service-level objectives. The framework is grounded in operational evidence from Rakuten Mobile, the world's first national-scale, fully cloud-native, fully Open RAN deployment, which reached full-year EBITDA profitability in FY2025. It is aligned with the ITU-R IMT-2030 framework, 3GPP 6G use cases and service requirements, NGMN recommendations, ETSI standards, O-RAN Alliance and AI-RAN Alliance specifications, IOWN Global Forum sustainability metrics, Linux Foundation initiatives, and leading industry and academic programs. A three-phase roadmap covering 2025-2027, 2027-2029, and 2029-2032 and beyond, together with seven stakeholder-specific calls to action, translates the architecture into industry commitments. The central claim is that Rakuten Mobile's deployment demonstrates the feasibility of operator-controlled 6G. Decisions made during 2026-2028 will determine whether 6G becomes a platform for guaranteed digital services or another vendor-dependent infrastructure cycle. 

---
# CTF4Nuclear: Common Task Framework for Nuclear Fission and Fusion Models 

**Authors**: Stefano Riva, Carolina Introini, Antonio Cammi, Dean Price, Alexey Yermakov, Yue Zhao, Philippe M. Wyder, Judah Goldfeder, Jan Williams, Amy Sara Rude, Matteo Tomasetto, Joe Germany, Joseph Bakarji, Georg Maierhofer, Miles Cranmer, J. Nathan Kutz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15549)  

**Abstract**: The demand for clean energy is ever increasing, with new nuclear technologies presenting a complementary solution to renewable energies. However, designing and operating these systems is exceptionally difficult, given the complexity of the physical phenomena that interact to form the system dynamics. While high-fidelity simulations help to understand the non-linear, multi-physics interactions within a reactor, they are computationally expensive and rarely suitable for real-time applications. Furthermore, model-based approaches are inherently sensitive to simplifying assumptions required to derive their governing equations and parameters, leading to inevitable discrepancies with real-world measurements. In contrast, Machine Learning (ML) methods have the potential to generate reliable surrogate models which may be able to quickly predict the system's behaviour. However, the number of data-driven methods that can potentially be used for this task is large and diverse. In a safety-critical setting such as nuclear engineering, a fair comparison of different ML methods, and a clear understanding of their advantages and limitations, is of paramount importance. To address this, we introduce a Common Task Framework (CTF) for ML in nuclear engineering, building upon previous efforts in dynamical systems and seismology. This CTF considers a curated set of datasets from different nuclear and nuclear-adjacent systems. The CTF evaluates the performance of a method on 12 established metrics, alongside a new paradigm focused on system monitoring from sparse measurements only. We illustrate the framework by benchmarking standard ML baselines against these datasets, revealing current method limitations. Our vision is to replace ad hoc comparisons with standardized evaluations on hidden test sets, raising the bar for rigour and reproducibility in scientific ML for the nuclear industry. 

---
# Domain-Independent Game Abstraction using Word Embedding Techniques 

**Authors**: Juho Kim, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2605.15543)  

**Abstract**: Many games of interest in the real world are often intractably large, thereby necessitating the use of game abstraction to shrink them in size, typically by many magnitudes. Over the last two decades, there have been significant advances in game abstraction; however, the domain-specific nature (usually poker) of much of the prior work prevents those techniques from being easily generalized to other settings without extensively analyzing the game at hand. In this paper, we propose a domain-independent approach to game abstraction, which applies word embedding techniques from the field of natural language processing. Treating each action as a word and gameplay data as a corpus, word vectors can be trained to represent each action as a real-valued vector, which can then be clustered to facilitate game abstraction. We also explore the use of foundational embedding models and show that action embeddings obtained this way can capture a surprising amount of information about the underlying game. Experimental results demonstrate that our proposed game abstraction technique is effective, although it does not outperform specialized algorithms tailored to specific games. 

---
# SkiP: When to Skip and When to Refine for Efficient Robot Manipulation 

**Authors**: Mingtong Dai, Guanqi Peng, Yongjie Bai, Feng Yan, Chunjie Chen, Lingbo Liu, Liang Lin, Xinyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15536)  

**Abstract**: Previous imitation learning policies predict future actions at every control step, whether in smooth motion phases or precise, contact-rich operation phases. This uniform treatment is wasteful: most steps in a manipulation trajectory traverse free space and carry little task-relevant information, while a small fraction of \emph{key} steps around contacts, grasps, and alignment demand dense, high-resolution prediction. We propose a novel \emph{action relabeling} mechanism: at each timestep in a skip segment, we replace the behavior cloning target with the action at the entrance of the next key segment, enabling the policy to leap over redundant steps in a single decision. The resulting \textbf{Skip Policy (SkiP)} dynamically leaps over skip segments and intensively refines actions in key segments, within a single unified network requiring no learned skip planner or hierarchical structure. To automatically partition demonstrations into key and skip segments without manual annotation, we introduce \emph{Motion Spectrum Keying} (MSK), a fast, task-agnostic procedure that detects local motion complexity from action signals. Extensive experiments across 72 simulated manipulation tasks and three real-robot tasks show that SkiP reduces executed steps by $15$--$40\%$ while matching or improving success rates across various policy backbones. Project page: \texttt{this https URL}. 

---
# Tuning-free Instruction-based Video Editing Via Structural Noise Initialization and Guidance 

**Authors**: Song Wu, Xinyu Chen, Qian Wang, Liang Li, Zili Yi, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15533)  

**Abstract**: Video editing poses a significant challenge. While a series of tuning-free methods circumvent the need for extensive data collection and model training, they often underutilize the rich information embedded within noisy latent, leading to unsatisfactory results. To address this, we propose a \textit{tuning-free, instruction-based} video editing framework. We approach video editing from the perspective of noisy latent: we design a Structural Noise Initialization Strategy (SNIS) to secure a superior editing starting point by assigning higher noise levels to edited regions (to facilitate content change) and lower noise levels to unedited regions (to maintain content consistency). We introduce a Noise Guidance Mechanism (NGM), which leverages the video prior in the generative model and effectively integrates rich information within the noisy latent to guide the denoising process, thereby preserving unedited content and overall visual coherence. Experiments show that our proposed method achieves better visual quality and state-of-the-art performance. 

---
# DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal Distillation 

**Authors**: Jaehun Jung, Hyunwoo Kim, Brandon Cui, Ximing Lu, David Acuna, Prithviraj Ammanabrolu, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15532)  

**Abstract**: Distillation enables compact Vision-Language Models (VLMs) to obtain strong reasoning capabilities, yet the prompts driving this process are typically chosen via simple heuristics or aggregated from off-the-shelf datasets. We reveal a critical inefficiency in this approach: up to 69% of the prompts in standard chart / document reasoning datasets are effectively zero-delta, meaning the teacher and student already induce the exact same answer distribution. Training on these prompts provides minimal learning signal, causing student improvement to rapidly saturate regardless of data scale. To escape the zero-delta trap, we return to first principles: distillation fundamentally minimizes distributional divergence, and thus a prompt is valuable only if it exposes a functional capability gap between the teacher and student. We quantify this gap through answer divergence ($\Delta$), demonstrating that non-zero divergence is critical for effective scaling. Building on this insight, we propose a staged synthesis pipeline that repurposes existing datasets as seeds, actively targeting student failure modes to produce better prompts. The result is DeltaPrompts, a diverse dataset of 200k synthetic, high-divergence reasoning problems. We evaluate DeltaPrompts across three distinct settings: on-policy distillation with the target teacher-student pair, transfer to a novel model family without regenerating the data, and off-policy fine-tuning of a non-reasoning model. Across all scenarios, DeltaPrompts drives substantial gains, yielding up to 15% relative improvement even on top of a highly-optimized reasoning model (e.g., Qwen3-VL-8B-Thinking) -- averaged over 10 benchmarks spanning chart, document and perception-centric reasoning. 

---
# Process Rewards with Learned Reliability 

**Authors**: Jinyuan Li, Langlin Huang, Chengsong Huang, Shaoyang Xu, Donghong Cai, Yuyi Yang, Wenxuan Zhang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15529)  

**Abstract**: Process Reward Models (PRMs) provide step-level feedback for reasoning, but current PRMs usually output only a single reward score for each step. Downstream methods must therefore treat imperfect step-level reward predictions as reliable decision signals, with no indication of when these predictions should be trusted. We propose BetaPRM, a distributional PRM that predicts both a step-level success probability and the reliability of that prediction. Given step-success supervision from Monte Carlo continuations, BetaPRM learns a Beta belief that explains the observed number of successful continuations through a Beta-Binomial likelihood, rather than regressing to the finite-sample success ratio as a point target. This learned reliability signal indicates when a step reward should be trusted, enabling downstream applications to distinguish reliable rewards from uncertain ones. As one application, we introduce Adaptive Computation Allocation (ACA) for PRM-guided Best-of-N reasoning. ACA uses the learned reliability signal to stop when a high-reward solution is reliable and to spend additional computation on uncertain candidate prefixes. Experiments across four backbones and four reasoning benchmarks show that BetaPRM improves PRM-guided Best-of-N selection while preserving standard step-level error detection. Built on this signal, ACA improves the accuracy--token tradeoff over fixed-budget Best-of-16, reducing token usage by up to 33.57% while improving final-answer accuracy. 

---
# Neural Point-Forms 

**Authors**: Bruno Trentini, Jacob Hume, Vincenzo Antonio Isoldi, Philipp Misof, Ekaterina S. Ivshina, Kelly Maggs  

**Link**: [PDF](https://arxiv.org/pdf/2605.15524)  

**Abstract**: Point cloud learning often rests on the premise that observed samples are noisy traces of an underlying geometric object, such as a manifold embedded in a high-dimensional feature space. Yet much of this geometry is not captured directly by coordinates, pairwise distances, or learned graph neighborhoods alone. In the smooth setting, differential forms are devices to encode higher order tangency information. In this work, we introduce a new family of principled learnable geometric features for point clouds called neural point-forms (NPFs). In the absence of a natural tangency structure, we instead use Laplacian-based techniques from Diffusion Geometry to build a discrete model for comparing differential forms on point clouds via inner products. In the continuum, submanifolds of a shared ambient feature space are represented as comparison matrices, whose entries describe how pairs of feature forms interact with extrinsic tangency information. We make this intuition precise by proving the long-run consistency of comparison matrices under standard sampling, bandwidth, density, and manifold-hypothesis assumptions. This yields a compact, efficient and permutation-invariant neural layer whose output is a learned form-comparison matrix. Across synthetic and biologically relevant experiments, we show that NPFs provide a competitive, and interpretable representation, with the strongest benefits appearing when labels depend on sampling density, manifold-like structure, or response-relevant population geometry. 

---
# On the Fragility of Data Attribution When Learning Is Distributed 

**Authors**: Xian Gao, Bo Hui, Min-Te Sun, Wei-Shinn Ku  

**Link**: [PDF](https://arxiv.org/pdf/2605.15520)  

**Abstract**: Data attribution has become an important component of pricing, auditing, and governance in machine learning pipelines, yet most attribution methods implicitly assume that attribution values faithfully reflect participants' contributions. We show that this assumption can fail: a single participant in a standard distributed training workflow can substantially inflate its measured attribution value while preserving global utility. Our attribution-first attack uses latent optimization to inject small synthetic batches that preserve utility while exploiting non-IID label coverage and evaluator sensitivities. Across datasets, models, and multiple marginal-utility evaluators, the attack consistently increases the adversary's attribution value and reshapes the relative attribution structure among benign clients without degrading accuracy or triggering geometry-based defenses. These results show that attribution itself forms a new attack surface and motivate the development of attribution-robust and incentive-compatible scoring mechanisms. 

---
# DiffVAS: Diffusion-Guided Visual Active Search in Partially Observable Environments 

**Authors**: Anindya Sarkar, Srikumar Sastry, Aleksis Pirinen, Nathan Jacobs, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2605.15519)  

**Abstract**: Visual active search (VAS) has been introduced as a modeling framework that leverages visual cues to direct aerial (e.g., UAV-based) exploration and pinpoint areas of interest within extensive geospatial regions. Potential applications of VAS include detecting hotspots for rare wildlife poaching, aiding search-and-rescue missions, and uncovering illegal trafficking of weapons, among other uses. Previous VAS approaches assume that the entire search space is known upfront, which is often unrealistic due to constraints such as a restricted field of view and high acquisition costs, and they typically learn policies tailored to specific target objects, which limits their ability to search for multiple target categories simultaneously. In this work, we propose DiffVAS, a target-conditioned policy that searches for diverse objects simultaneously according to task requirements in partially observable environments, which advances the deployment of visual active search policies in real-world applications. DiffVAS leverages a diffusion model to reconstruct the entire geospatial area from sequentially observed partial glimpses, which enables a target-conditioned reinforcement learning-based planning module to effectively reason and guide subsequent search steps. Extensive experiments demonstrate that DiffVAS excels in searching diverse objects in partially observable environments, significantly surpassing state-of-the-art methods on several datasets. 

---
# RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably 

**Authors**: Yufeng Du, Phillip Harris, Minyang Tian, Eliu A Huerta, Srikanth Ronanki, Subendhu Rongali, Aram Galstyan, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15514)  

**Abstract**: We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models. 

---
# PrismQuant: Rate-Distortion-Optimal Vector Quantization for Gaussian-Mixture Sources 

**Authors**: Bumsu Park, Chanho Park, Youngmok Park, Namyoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.15507)  

**Abstract**: For a Gaussian source under mean-squared error (MSE), classical transform coding is rate--distortion (RD) optimal: the Karhunen--Loeve transform (KLT) diagonalizes the covariance, reverse waterfilling allocates the bits, and scalar quantization closes the loop. This elegant story breaks down for multimodal sources, where no single covariance can capture heterogeneous local geometries, and the RD function loses its closed form. We revisit this problem through Gaussian-mixture sources and develop a constructive RD theory for them. Our key finding is that the mixture structure incurs only a component label cost. Conditioned on the active mixture component, each branch is Gaussian; the challenge is allocating bits across heterogeneous branches. We prove that the genie-aided conditional RD function is governed by a single global reverse-waterfilling level shared across all components and eigenmodes. Building on this result, we introduce PrismQuant, which transmits the component label losslessly and encodes the residual using the component-matched KLT, followed by scalar quantization, achieving a rate of H(C)/n bits per source dimension of the converse, with a vanishing asymptotic gap. We further develop a practical implementation based on EM-driven Gaussian-mixture learning, component-adaptive KLTs, and entropy-constrained scalar quantization (ECSQ). Experiments on synthetic Gaussian mixtures show that PrismQuant closely approaches the theoretical RD bound, while experiments on real-world channel-state-information (CSI) data demonstrate competitive or superior performance compared with transformer-based learned codecs at more than one order of magnitude smaller model size. 

---
# Learning with Conflicts of Interest 

**Authors**: Nischal Aryal, Arash Termehchy, Ali Vakilian, Marianne Winslett  

**Link**: [PDF](https://arxiv.org/pdf/2605.15504)  

**Abstract**: Financial, social, and political factors often prevent the interests of the owners of ML systems and services and their users from being perfectly aligned. ML systems often produce biased information that can influence users to make decisions that are not in their best interest. Current solution approaches require ML systems to implement protocols to mitigate their biases. However, ML system owners usually do not have any incentive to implement these protocols and often argue that it limits their freedom of expression or business. We believe that a successful solution to this problem must recognize the conflict of interest between the ML systems and their users, and use this information to protect users against information that adversely influences their decisions while allowing users to safely benefit from these systems. To this end, we propose a game-theoretic framework that models the interaction between ML systems and users with conflicts of interest. We present scalable algorithms with theoretical guarantees that maximize the amount of desired information and actions and minimize the amount of biased and manipulative actions in interaction with ML systems. 

---
# Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs 

**Authors**: Vincent-Daniel Yun, Junhyuk Jo, Sai Praneeth Karimireddy, Sunwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.15491)  

**Abstract**: Layer pruning removes entire Transformer decoder blocks from large language models, but introduces a mismatch between the hidden state received by the next surviving layer and the distribution it was trained to process, leading to significant performance degradation. We propose Ghosted Layers, a training-free recovery module that addresses this issue by solving a boundary activation alignment problem. Our method derives a closed-form optimal linear operator from a small calibration set to reconstruct the activation discrepancy introduced by the pruned layers. We show that this solution corresponds to the unconstrained optimum of the alignment objective, whereas existing methods are restricted to constrained solutions over limited operator subspaces. Experiments across multiple LLM backbones and pruning strategies demonstrate that our method consistently improves accuracy and perplexity over prior training-free baselines, while preserving the efficiency gains of layer pruning. 

---
# Hybrid LLM-based Intelligent Framework for Robot Task Scheduling 

**Authors**: Swayamjit Saha, Subhabrata Das, Haonan Duan, Xiao-Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15486)  

**Abstract**: This study introduces intelligent frameworks that use Large Language Models (LLMs) to improve task scheduling for construction robots. The LLM is fed with key data about the desired task, such as agent action abilities, and the desired end goal to be achieved. A well-balanced allocation strategy is developed, optimizing both time efficiency and resource utilization. Our system utilizes a Natural Language Processing interface to streamline communication with construction professionals and adapt in real-time to unexpected site conditions. We concurrently use two LLM agents, specifically generator (GPT-4) and supervisor (Gemma 3/Llama 4/Mistral 7b) LLM agents to provide a more precise task schedule. We evaluate the proposed methodology using a straightforward scenario and provide metric scores to prove the efficacy of the frameworks. Our results highlight that the implementation of LLMs is crucial in construction operational tasks including robots. 

---
# Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays 

**Authors**: Kaize Deng, Zewen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15480)  

**Abstract**: Stochastic communication delays in teleoperation introduce signal discontinuities that undermine control stability and degrade control performance. Consequently, the conventional reinforcement learning (RL) methods struggle with the delayed observations due to the delay-induced observations, leading to high-frequency chattering. To address this, we propose a hybrid control framework, delay-resilient RL, integrating a state estimator utilizing Long Short-Term Memory (LSTM) with a residual RL policy, which is resilient to stochastic delays. The LSTM reconstructs smooth, continuous state estimates from delayed observations, enabling the RL agent to learn a residual torque compensation policy that balances tracking accuracy with velocity smoothness. Experimental validation on Franka Panda robots demonstrates that our approach significantly outperforms the state-of-the-art baselines, ensuring robust and stable teleoperation even under high-variance stochastic delays. 

---
# Retrieval-Augmented Large Language Models for Schema-Constrained Clinical Information Extraction 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2605.15467)  

**Abstract**: Conversational nurse-patient transcripts contain actionable observations, but converting these transcripts into structured representations at scale remains challenging. Documentation burden is substantial, with prior studies showing clinicians spend large portions of their workday on documentation and related desk work rather than direct patient care. MEDIQA-SYNUR focuses on observation extraction from conversational nurse-patient transcripts, requiring systems to normalize these narratives into a predefined schema with value-type constraints. We propose a modular retrieval-augmented generation (RAG) pipeline that uses the training set as an exemplar corpus, combines schema-constrained prompting (full schema vs. pruned candidate schema), deterministic schema-based postprocessing, and a second-pass audit, with two LLM backbones: Llama-4-Scout-17B-16E-Instruct and GPT-5.2 with corresponding embedding models for RAG. Our best configuration uses GPT-5.2 with full schema, RAG, and a second-pass auditing, achieving 80.36% F1 score. Overall, our results show that RAG consistently improves performance, while the optimal degree of schema constraint depends on the model, and second-pass auditing yields modest additional gains by correcting residual schema-adherence errors. 

---
# GRLO: Towards Generalizable Reinforcement Learning in Open-Ended Environments from Zero 

**Authors**: Shangjian Yin, Yu Fu, Yue Dong, Zhouxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15464)  

**Abstract**: Post-training has become a crucial step for unlocking the capabilities of large language models, with reinforcement learning (RL) emerging as a critical paradigm. Recent RL-based post-training has increasingly split into two paradigms: reinforcement learning from human feedback (RLHF), which optimizes models using human preference signals in target domains, and reinforcement learning from verifiable rewards (RLVR), which operates in verifier-backed environments. The latter has dominated recent reasoning-oriented post-training because it delivers stronger gains and higher efficiency on domain-specific tasks (e.g., reasoning). However, although in-domain RL training achieves promising performance, it still requires a substantial amount of GPU compute, which remains a major barrier to broad adoption. In this work, we study the generalization ability of RLHF learned from scratch from a small set of interactions in open-ended environments, and investigate whether the conversational abilities it explicitly acquires can implicitly transfer to downstream tasks such as mathematical reasoning and code generation, namely GRLO. Specifically, on Qwen3-4B-Base backbone, GRLO improves the average performance across all domains from 24.1 to 63.1 with only 5K prompts and 22.7 GPU hours, requiring about $46\times$ less data and $68\times$ less compute than a strong in-domain RLVR baseline. The resulting model is even competitive with Qwen's released post-trained models which required a much larger training cost. Notably, a subsequent in-domain RLVR stage brings only selective gains, mainly on harder competition-math benchmarks. We hope GRLO offers a simple and efficient recipe for building broadly capable post-trained models. Our code and data will be available at: \href{this https URL}{this https URL}. 

---
# DrugSAGE:Self-evolving Agent Experience for Efficient State-of-the-Art Drug Discovery 

**Authors**: Yikun Zhang, Xiwei Cheng, Tianyu Liu, Yuanqi Du, Wengong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.15461)  

**Abstract**: Building state-of-the-art (SOTA) predictive models for drug discovery requires expensive search over tools, architectures, and training strategies. Current LLM-based agents can find SOTA solutions through extensive trial and error, but they do not retain the experience accumulated along the way and therefore pay the full search cost on every new task. We propose \method (Self-evolving Agent Experience), a framework that accumulates and reuses experience across tasks to build SOTA drug discovery models efficiently. \method maintains a cross-task memory of verified skills, statistical evidence about effective strategies, and a record of recurring errors and their fixes. In some cases, \method transfers a working solution directly without test-time search. In 33 molecular property prediction tasks, \method ranks first among nine SOTA agents in a single-task setting. With memory accumulated from 16 smaller tasks, \method achieves an averaged normalized score of 0.935 on 17 held-out tasks in a cross-task evaluation setting and outperforms all baseline agents by 10-30\% in a zero-test-time search regime. In summary, our work shows the advantage of cross-task memory for efficient SOTA model development in drug discovery. 

---
# Differentially Private Motif-Preserving Multi-modal Hashing 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.15460)  

**Abstract**: Cross-modal hashing enables efficient retrieval by encoding images and text into compact binary codes. State-of-the-art methods rely on semantic similarity graphs derived from user interactions for supervision, yet these graphs encode sensitive behavioral patterns vulnerable to link reconstruction attacks. Existing privacy-preserving approaches fail on graph-structured data: Differentially Private SGD destroys relational motifs by treating samples independently, while graph synthesis methods suffer from unbounded local sensitivity in scale-free networks, hub nodes cause single-edge modifications to alter triangle counts by $\mathcal{O}(N)$, necessitating prohibitive noise injection. We term this phenomenon Hubness Explosion. We propose DMP-MH, a Sanitize-then-Distill framework that decouples privacy from representation learning. Our approach first bounds sensitivity by deterministically clipping node degrees, capping the $L_2$-sensitivity of triangle motifs independently of dataset size. A sanitized synthetic graph is then generated via Noisy Mirror Descent under $(\epsilon,\delta)$-Edge Differential Privacy. Finally, dual-stream hashing networks distill this topology using a holistic structural loss that enforces cross-modal alignment. Evaluated on MIRFlickr-25K and NUS-WIDE under a strict inductive protocol, DMP-MH outperforms private baselines by up to 11.4 mAP points while retaining up to 92.5% of non-private performance. 

---
# RIDE: Retinex-Informed Decoupling for Exposing Concealed Objects 

**Authors**: Chunming He, Rihan Zhang, Dingming Zhang, Chengyu Fang, Longxiang Tang, Jingjia Feng, Fengyang Xiao, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15450)  

**Abstract**: Concealed Object Segmentation (COS) encompasses a family of dense-prediction tasks, including camouflaged object detection, polyp segmentation, transparent object detection, and industrial defect inspection, where targets are visually entangled with their surroundings through different physical mechanisms. Existing methods either operate directly on RGB images or employ \emph{heterogeneous} decompositions (\eg, Fourier, wavelet) that redistribute spatial evidence across scale/frequency coefficients, making pixel-aligned cues less direct. We introduce a fundamentally different perspective: \textbf{homogeneous image decomposition} via Retinex theory, which factorizes an image into illumination and reflectance components within the \emph{same} spatial domain. Our key insight is that visual entanglement enforces appearance matching in the composite space, but this does \emph{not} necessitate simultaneous matching in both component spaces, a phenomenon we formalize as the \textbf{Discriminability Gap Theorem}. Crucially, we show that across diverse COS sub-tasks, the underlying physical processes systematically anti-correlate illumination and reflectance differences, yielding theoretical guarantees that Retinex decomposition preserves or strictly improves total foreground--background discriminability across the full physical regime, with anti-correlation maximizing the gain. Building on this, we propose \textbf{RIDE} comprising: (i) a Task-Driven Retinex Decomposition module that learns segmentation-optimal factorizations end-to-end; (ii) a Discriminability Gap Attention mechanism that adaptively exploits where decomposition helps; and (iii) a Camouflage-Breaking Contrastive loss operating in reflectance feature space. 

---
# Runtime-Structured Task Decomposition for Agentic Coding Systems 

**Authors**: Shubhi Asthana, Bing Zhang, Chad DeLuca, Hima Patel, Ruchi Mahindru  

**Link**: [PDF](https://arxiv.org/pdf/2605.15425)  

**Abstract**: Agentic coding systems increasingly use large language models (LLMs) for software engineering tasks such as debugging, root cause analysis, and code review. However, many existing systems encode task logic, execution flow, and output generation inside monolithic prompts. This design creates brittle behavior, limited debuggability, and high retry costs because failures often require rerunning the full workflow.
We present runtime-structured task decomposition, an architectural approach in which task partitioning and execution flow are managed through executable control logic rather than prompt structure alone. LLMs are used only for focused judgment tasks, and outputs are validated against predefined schemas before downstream execution.
We evaluate this approach on two software engineering workloads using three configurations: monolithic execution, static decomposition with fixed subtasks and no runtime branching, and runtime-structured decomposition. Each configuration was evaluated across 10 runs.
Our results show that decomposition alone does not necessarily reduce retry cost. In the Kubernetes root cause analysis workload, the static decomposition baseline produced a retry cost of 1,632 +/- 145 tokens versus 904 +/- 17 tokens for the monolithic baseline because failures forced reruns of downstream subtasks. A similar pattern appeared in the multi-file debugging workload, where the static baseline consumed 933 tokens compared to 703 tokens for the monolithic system.
The runtime-structured approach reran only failed subtasks, reducing retry costs to 436 +/- 132 tokens for root cause analysis and 460 tokens for debugging. Overall, the approach achieved up to 51.7% lower retry cost than monolithic systems and 73.2% lower retry cost than static decomposition baselines, improving efficiency, debuggability, and operational reliability in agentic coding systems. 

---
# MR2-ByteTrack: CNN and Transformer-based Video Object Detection for AI-augmented Embedded Vision Sensor Nodes 

**Authors**: Luca Bompani, Manuele Rusci, Luca Benini, Daniele Palossi, Francesco Conti  

**Link**: [PDF](https://arxiv.org/pdf/2605.15423)  

**Abstract**: Modern smart vision sensors need on-device intelligence to process video streams, as cloud computing is often impractical due to bandwidth, latency, and privacy constraints. However, these sensory systems typically rely on ultra-low-power microcontrollers (MCUs) with limited memory and compute, making conventional video object detection methods, which require feature storage or multi-frame buffering, unfeasible. To address this challenge, we introduce Multi-Resolution Rescored ByteTrack (MR2-ByteTrack), a Video Object Detection (VOD) method tailored for MCU-based embedded vision nodes. MR2-ByteTrack reduces computational cost by alternating between full- and low-resolution inference, while linking detections across frames via ByteTrack and correcting misclassifications through the Rescore algorithm, which applies probability union rules to aggregate detection confidence scores across frames. We apply our approach to both a CNN-based detector and a Transformer-based model, demonstrating its generality across architectures with fundamentally different spatial processing. Experiments on ImageNetVID demonstrate that MR2-ByteTrack maintains accuracy, achieving mAP scores of up to 49.0 for the CNN-based models and 48.7 for the Transformer, while reducing multiply-accumulate operations by as much as 53\% for the CNNs and 32\% for the Transformer. When deployed on GAP9, an ultra-low-power RISC-V multicore MCU, our method yields up to 55\% energy savings compared to processing only full-resolution images, enabling the first real-time Transformer-based VOD on an MCU-class embedded vision node. Code available at this https URL 

---
# $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data 

**Authors**: Jake Fawkes, Jason Hartford  

**Link**: [PDF](https://arxiv.org/pdf/2605.15417)  

**Abstract**: In GFlowNets and variational inference, it has been shown that the mean square error between target and model log probabilities is an effective, low variance, surrogate loss for training generative models.
This loss has the property that when evaluated \emph{on-policy} its gradients correspond to those of the KL divergence, while \emph{off-policy} it remains a valid loss with the same global minimizer. In this work, we demonstrate that this construction can be extended to the whole family of $f$-divergences, leading to a family of losses whose on-policy gradients are that of the corresponding $f$-divergence, but retain the same global minimizer off-policy. Specifically, we show that the on-policy gradients lead to a one to one correspondence between translation invariant loss functions on the target and model log probabilities, and $f$-divergences. This equivalence allows us to design new surrogate loss functions for tuning a wide class of generative models that inherit the properties of the corresponding $f$-divergence, such as being more mode covering, whilst being applicable to off-policy data. We apply our losses on a range of tasks, including classic synthetic examples, SynFlowNets for molecule discovery, and asynchronous large language model (LLM) tuning, demonstrating that our models retain their predicted properties on- and off-policy in a wide class of generative models. 

---
# Margin-Adaptive Confidence Ranking for Reliable LLM Judgement 

**Authors**: Gaojie Jin, Yong Tao, Lijia Yu, Tianjin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15416)  

**Abstract**: Jung et al. (2025) introduce a hypothesis testing framework for guaranteeing agreement between large language models (LLMs) and human judgments, relying on the assumption that the model's estimated confidence is monotonic with respect to human-disagreement risk. In practice, however, this assumption may be violated, and the generalization behavior of the confidence estimator is not explicitly analyzed. We mitigate these issues by learning a dedicated confidence estimator instead of relying on heuristic confidence signals. Our approach leverages simulated annotator diversity and a margin-based ranking formulation to explicitly model how confidently an LLM distinguishes between human-agreement and human-disagreement cases. We further derive generalization guarantees for this estimator, revealing a margin-dependent trade-off that informs the design of an adaptive estimator training procedure. When integrated into fixed-sequence testing, the learned confidence estimator yields improved ranking accuracy and empirically strengthens the monotonic relationship between confidence and disagreement risk, leading to higher success rates in satisfying target agreement levels across multiple datasets and judge models. 

---
# From Feedback Loops to Policy Updates: Reinforcement Fine-Tuning for LLM-Based Alpha Factor Discovery 

**Authors**: Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Zixuan Xie, Chiming Duan, Minghua He, Philip S. Yu, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.15412)  

**Abstract**: Modern quantitative trading increasingly relies on systematic models to extract predictive signals from large-scale financial data, where alpha factor discovery plays a central role in transforming market observations into tradable signals. Recent LLM-based methods have shown promise in automating factor generation, but most of them still rely on prompt-level generation--evaluation--feedback loops for iterative optimization. As the loop becomes longer, repeatedly appended historical candidates and feedback can cause context explosion, increase inference cost, dilute useful information, and introduce feedback drift. Moreover, these methods often depend on very large LLMs whose stable generation preferences may lead to structurally similar expressions, redundant candidates, and search stagnation. To address these limitations, we propose \textsc{QuantEvolver}, a self-evolving alpha factor discovery framework based on reinforcement fine-tuning. Instead of accumulating feedback in the prompt, \textsc{QuantEvolver} converts executable quantitative evaluation into policy updates, enabling a Miner LLM to internalize historical optimization experience through parameter learning. Specifically, \textsc{QuantEvolver} constructs high-quality seed factors, builds diverse seed--time-window training tasks, generates executable Factor DSL expressions, evaluates them through Regime Backtest, and optimizes the Miner LLM with Diversity-Complementarity Reward. During training, high-quality factors are continuously accumulated in a Mined Factor Database, which serves as the final discovered factor library. Extensive experiments on three realistic market benchmarks demonstrate the effectiveness of \textsc{QuantEvolver}, which consistently improves the primary evaluation metric of each task over existing LLM-based alpha factor discovery baselines, produces higher-quality and more complementary factor pools. 

---
# Diagonal Adaptive Non-local Observables on Quantum Neural Networks 

**Authors**: Huan-Hsin Tseng, Yan Li, Hsin-Yi Lin, Samuel Yen-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15410)  

**Abstract**: Adaptive Non-local Observables (ANOs) have shown that making quantum observables dynamic can substantially enlarge the function space of Variational Quantum Algorithms, partly shifting hardware demands from circuit synthesis to measurement design. However, this advantage is accompanied by a steep increase in the number of parameters, as well as the classical optimization cost for varying general Hermitian observables.
We propose a special form of ANO that significantly reduces this burden by considering only diagonal observables paired with quantum circuits. Mathematically, this is equivalent to the full ANO of a large parameter space since diagonal matrices are canonical representatives of the ANO space modulo unitary similarity. As a result, Diagonal ANO retains the same capability of full ANO while reducing $k$-local observable complexity from $O(4^k)$ to $O(2^k)$ and lowering the corresponding measurement-side classical computation. In this sense, diagonal ANO preserves much of the benefit of full ANO while encompassing conventional VQCs as a special case. 

---
# Amortized Energy-Based Bayesian Inference 

**Authors**: Hojjat Kaveh, Ricardo Baptista, Andrew M. Stuart  

**Link**: [PDF](https://arxiv.org/pdf/2605.15407)  

**Abstract**: We consider amortized Bayesian inference for nonlinear inverse problems in settings where only samples from the joint distribution of parameters and observations are available. Classical methods such as Markov chain Monte Carlo require solving a new inference problem for each observation, which can be computationally prohibitive when inference must be repeated many times. We propose a transport-based approach that learns an observation-dependent map pushing forward a reference measure to approximate the posterior distribution. The map is trained by minimizing an averaged energy-distance objective between the true posterior and the learned pushforward. This formulation is likelihood-free, requiring only joint samples, and avoids density evaluation, invertibility constraints, and Jacobian determinant computations. For function-space inverse problems with Gaussian priors, we parameterize the transport map as the identity plus a perturbation in the Cameron-Martin space of the prior, preserving absolute continuity with respect to the prior. In infinite-dimensional settings, the map is represented using neural operators. We illustrate the method on a finite-dimensional nonlinear inverse problem and two PDE-constrained inverse problems arising in porous medium flow and seismic inversion. The results show that the learned transport captures posterior structure, including multimodality and dominant modes, while enabling fast posterior sampling for new observations. 

---
# Breakeven complexity: A new perspective on neural partial differential equation solvers 

**Authors**: Yijing Zhang, Nicholas Roberts, Tanya Marwah, Mikhail Khodak  

**Link**: [PDF](https://arxiv.org/pdf/2605.15399)  

**Abstract**: Neural surrogate solvers of partial differential equations (PDEs) promise dramatic speedups over numerical methods, especially in scenarios requiring many solves. However, current accuracy-based evaluations do not fully consider two central issues: (1) neural solvers incur substantial up-front costs for data generation, training, and tuning; and (2) classical solvers can also generate low-fidelity solutions at a sufficiently low simulation cost. To explicitly account for these realities and fully incorporate end-to-end costs, we propose an evaluation framework centered on breakeven complexity, a metric that counts the forward solves before a learned solver is cost-effective relative to an error-equivalent traditional solver. To evaluate this measure, we apply scaling laws to determine how much training budget to allocate to data generation and discuss how to achieve smooth error-matching in diverse settings. We evaluate the breakeven complexity of multiple neural PDE solvers on three PDEs on 2D periodic domains from APEBench and a novel benchmark of flows past multiple obstacles generated by the GPU-native PyFR code. Among other findings, our results suggest that neural PDE solvers become more effective as problems get harder in terms of cost, dimension, rollout, physics regime (e.g. higher Reynolds number), etc. 

---
# Representation Without Reward: A JEPA Audit for LLM Fine-Tuning 

**Authors**: Biswa Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.15394)  

**Abstract**: Joint-embedding predictive architectures (JEPAs) propose that a model should learn more useful abstractions when trained to predict latent representations rather than observed outputs. For autoregressive language-model fine-tuning the principle entails a stricter requirement: the induced hidden-state geometry must reach the language-model head \emph{and} improve the decoded task metric. We test that requirement under a fixed Llama-3.2-1B-Instruct LoRA harness on natural-language-to-regex generation, comparing twenty-two training-time auxiliaries across trajectory-shape regularisation, distributional constraints, predictor/target asymmetry, Fisher-metric Jacobi residuals, and a decoder-visible JEPA objective constructed to lie in cross-entropy's positive cone. The empirical answer is a structured null: several auxiliaries clear single-cell paired $\alpha = 0.10$ without correction (T3-Local at $\Delta = +2.53$~pp, $p = 0.003$ being the strongest), but none survives Bonferroni or Holm--Bonferroni at the relevant family-wise threshold, even though many change curvature, anisotropy, variance, and gradient direction. Decoder-visible JEPA yields the first positive auxiliary--cross-entropy gradient cosine in the study, yet exact match remains inside seed noise; a full-fine-tuning replication of the same auxiliary at $n = 5$ seeds reproduces the null on both benchmarks (TURK: $\Delta = +0.04$~pp, $p_{\text{paired}} = 0.96$; SYNTH: $\Delta = +0.52$~pp, $p_{\text{paired}} = 0.28$), so the null is robust across LoRA and full fine-tuning for the decoder-visible construction. Hidden-state representation work and decoded-task accuracy are therefore weakly coupled in this regime; we accordingly reframe LLM-domain JEPA evaluation as a coupling problem, in which the operative question is under which metrics useful hidden geometry becomes decoder-visible task signal. 

---
# PanoWorld: Geometry-Consistent Panoramic Video World Modeling 

**Authors**: Le Jiang, Xiangyu Bai, Bishoy Galoaa, Shayda Moezzi, Caleb James Lee, Tooba Imtiaz, Edmund Yeh, Jennifer Dy, Yanzhi Wang, Sarah Ostadabbas  

**Link**: [PDF](https://arxiv.org/pdf/2605.15391)  

**Abstract**: We present PanoWorld, a panoramic video world model that generates geometry-consistent 360$\degree$ video from a single image and a caption. Existing panoramic video methods optimize primarily for visual realism and do not explicitly constrain the underlying 3D scene state, producing outputs that appear plausible yet exhibit inconsistent depth, broken correspondences, and implausible motion across the spherical surface. We address this gap by framing panoramic video generation as a geometry- and dynamics-consistent latent state modeling problem rather than pure visual synthesis. Building on a pre-trained perspective video world model, we introduce two lightweight regularizers: a depth consistency loss against pseudo ground-truth panoramic depth, and a trajectory consistency loss that supervises the 3D world-frame positions of tracked points across time. We further apply spherical-geometry-aware adaptation to the conditioning and positional encoding. We additionally introduce PanoGeo, a unified geometry-aware panoramic video dataset with consistent depth, trajectory, and prompt annotations across diverse real and synthetic sources, used for both training and stratified evaluation. Experiments show that PanoWorld improves geometric consistency over prior panoramic generation methods while maintaining competitive visual realism, establishing that panoramic video generation must be treated as a geometric modeling problem to support the holistic spatial understanding requirements of embodied AI applications. Code is available at this https URL. 

---
# Is One Score Enough? Rethinking the Evaluation of Sequentially Evolving LLM Memory 

**Authors**: Songwei Dong, Zihan Chen, Chengshuai Shi, Peng Wang, Jundong Li, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15384)  

**Abstract**: Memory plays a central role in enabling large language models (LLMs) to operate over sequential tasks by accumulating and reusing experience over time. However, existing evaluations of LLM memory mostly rely on aggregate metrics such as final hold-out accuracy or cumulative online performance, which can obscure critical failure modes such as forgetting and negative transfer. In this paper, we introduce SeqMem-Eval, a diagnostic evaluation framework for sequentially evolving LLM memory. Drawing inspiration from continual learning, it targets a test-time setting in which memory is external, prompt-mediated, and updated without modifying model parameters. Rather than focusing only on final performance, SeqMem-Eval evaluates how memory states evolve, generalize, consolidate experience, and retain useful information during sequential inference. Specifically, it measures online utility, hold-out generalization, backward transfer, and forgetting, providing a finer-grained view of memory quality. Through extensive experiments across diverse tasks and memory methods, we show that higher final or cumulative accuracy does not necessarily imply better memory quality: many methods exhibit strong performance gains while suffering from substantial forgetting or negative transfer. Moreover, different memory designs exhibit distinct trade-offs between adaptability and stability that remain invisible under standard evaluation metrics. 

---
# ChangeFlow -- Latent Rectified Flow for Change Detection in Remote Sensing 

**Authors**: Blaž Rolih, Matic Fučka, Filip Wolf, Luka Čehovin Zajc  

**Link**: [PDF](https://arxiv.org/pdf/2605.15375)  

**Abstract**: Remote sensing change detection (RSCD) aims to localise changes between two images of the same geographic region. In practice, change masks often follow region-level annotation conventions rather than purely local appearance differences, making them context-dependent and occasionally ambiguous. Most state-of-the-art methods utilise per-pixel discriminative classification, which produces a single prediction per input and fails to explicitly model the changed region as a coherent whole. A natural alternative is generative formulation, which can model a distribution of plausible masks, enabling sampling to capture ambiguity and encourage global consistency. However, existing generative RSCD approaches typically lag behind strong discriminative baselines due to the high computational cost of pixel-space generation and the complexity of their conditioning mechanisms. To address the limitations of prior discriminative and generative methods, we propose ChangeFlow, a generative framework that reformulates change detection as the synthesis of a change mask in latent space via rectified flow. ChangeFlow is guided by a structured yet lightweight conditioning signal, and its stochastic design naturally supports sampling-based prediction ensembling. Namely, aggregating multiple predicted change masks improves robustness, while sample agreement provides a practical confidence estimation that highlights ambiguous regions. Across four benchmarks, ChangeFlow achieves an average F1 of 80.4\%, improving by 1.3 points on average over the previous best method, while maintaining inference speed comparable to recent strong baselines. Project page: this https URL 

---
# PACER: Acyclic Causal Discovery from Large-Scale Interventional Data 

**Authors**: Ramon Viñas Torné, Sílvia Fàbregas Salazar, Soyon Park, Ivo Alexander Ban, Artyom Gadetsky, Nikita Doikov, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2605.15353)  

**Abstract**: Inferring the structure of directed acyclic graphs (DAGs) from data is a central challenge in causal discovery, particularly in modern high-dimensional settings where large-scale interventional data are increasingly available. While interventional data can improve identifiability, existing methods remain limited by soft acyclicity constraints, leading to optimization over invalid cyclic graphs, numerical instability, and reduced scalability. We introduce PACER (Perturbation-driven Acyclic Causal Edge Recovery), a scalable framework for causal discovery that guarantees acyclicity by construction. PACER parameterizes a distribution over DAGs through a joint model of variable permutations and edge probabilities, enabling direct optimization over valid causal structures without surrogate penalties. The framework supports a unified likelihood-based treatment of observational and interventional data, flexible conditional density models, and the incorporation of structural prior knowledge. For linear-Gaussian mechanisms, we derive closed-form expressions for the expected interventional log-likelihood and its gradients, yielding substantial computational gains. Empirically, PACER matches or exceeds state-of-the-art methods on protein signaling and large-scale genetic perturbation benchmarks, while scaling efficiently to networks with thousands of variables and achieving up to two orders of magnitude speedups over penalty-based differentiable approaches. These results demonstrate that exact and scalable causal discovery from high-dimensional perturbation data is achievable through principled search space design. 

---
# LEAP: Trajectory-Level Evaluation of LLMs in Iterative Scientific Design 

**Authors**: Marilyn Zhang, Tianfeng Chen, Fabián Barzuna, Ankita Rathod, Mark E. Whiting  

**Link**: [PDF](https://arxiv.org/pdf/2605.15341)  

**Abstract**: LLMs are increasingly deployed in autonomous laboratories, under the assumption that their domain priors and reasoning over iterative feedback let them converge on good designs in fewer iterations than feedback-only baselines. Current iterative scientific design benchmarks, however, score only outcome snapshots at fixed horizons. This leaves the learning trajectory unmeasured, even though the trajectory is what captures learning efficiency, where each iteration saved is a real saving in cost and time. Motivated by this, we examine three evaluation choices that change the conclusions one draws about LLM learning efficiency in iterative scientific design: what to measure, what baseline to compare against, and what to ground against. We introduce LEAPBench, Learning Efficiency in Adaptive Processes, a 55-task framework that pairs a best-so-far area under the curve (AUC) trajectory metric with a classical Bayesian-optimization reference and an audit grounded in published literature. Applied to eight contemporary LLMs, switching from final-outcome to trajectory scoring changes the best-model decision on 53% of tasks at matched horizons, and exposes efficiency gains overlooked by outcome-based scoring. LLMs do not outperform a classical Bayesian baseline. On 16 biology tasks where the oracle's reward signal is aligned with configurations from the published-best design, domain-aware prompting leads to LLM choices that match the published-best's approximately 10 percentage points less often than domain-agnostic prompting at iteration 30. The pattern is sharpest on 6 tasks where the literature-typical and published-best configurations diverge, and domain-agnostic prompting matches the published-best more often on all 6. The trajectory metric also doubles as a tractable training target. Offline reinforcement learning with the metric as a reward improves performance on 14 of 21 held-out tasks. 

---
# Hidden in Memory: Sleeper Memory Poisoning in LLM Agents 

**Authors**: Sidharth Pulipaka, Stanislau Hlebik, Leonidas Raghav, Sahar Abdelnabi, Vyas Raina, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15338)  

**Abstract**: Large language models are increasingly augmented with persistent memory, allowing assistants to store user-specific information across sessions for personalization and continuity. This statefulness introduces a new security risk: adversarial content can corrupt what an assistant remembers and thereby influence future interactions. We propose and study sleeper memory poisoning, a delayed attack in which an adversary manipulates external context, such as a document, webpage, or repository, to cause the assistant to store a fabricated memory about the user. Unlike conventional prompt injection, the attack can remain dormant and re-emerge across multiple later conversations. We evaluate the full attack pipeline: whether poisoned memories are written, later retrieved, and ultimately used to steer the following conversations. Across stateful LLM assistants, poisoned memories were added up to 99.8% on GPT-5.5 and 95% on Kimi-K2.6. Crucially, among successful retrievals, poisoned memories cause attacker-intended agentic actions in 60-89% of evaluations across models. These results show that persistent memory can act as a long-term attack surface across multiple future conversations. 

---
# HoloMotion-1 Technical Report 

**Authors**: Maiyue Chen, Kaihui Wang, Bo Zhang, Xihan Ma, Zhiyuan Yang, Yi Ren, Qijun Huang, Zihao Zhu, Yucheng Wang, Zhizhong Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.15336)  

**Abstract**: In this report, we present HoloMotion-1, a humanoid motion foundation model for zero-shot whole-body motion tracking. A key innovation of HoloMotion-1 is to scale control-policy training with a large-scale hybrid motion corpus, where video-reconstructed motions from in-the-wild videos provide the dominant source of motion diversity, while curated motion-capture and in-house motion data provide higher-fidelity supervision and deployment-oriented coverage. This data regime enables HoloMotion-1 to move beyond conventional MoCap-only training and exposes the policy to substantially broader behaviors, capture conditions, and motion styles.
Learning from such heterogeneous data introduces new challenges, including reconstruction noise, source-domain mismatch, uneven motion quality, and the need for temporal modeling under large behavioral variation. To address these challenges, HoloMotion-1 integrates large-capacity temporal modeling, a sparsely activated Mixture-of-Experts Transformer with KV-cache inference for real-time control, and a sequence-level training strategy that improves learning efficiency on extended motion sequences. Extensive experiments on multiple unseen motion benchmarks show that HoloMotion-1 generalizes robustly across diverse motion types and capture conditions, significantly improves tracking accuracy over prior methods, and transfers directly to a real humanoid robot without task-specific fine-tuning. 

---
# From I/O to Code with Discovery Agent 

**Authors**: Yihong Dong, Jiaru Qian, Haoran Zhang, Peixu Wang, Binhua Li, Zhi Jin, Yongbin Li, Ge Li, Xiaokang Yang, Xue Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15334)  

**Abstract**: The automatic synthesis of a program from any form of specification is regarded as a holy grail of computer science. Fueled by LLMs, NL2Code has achieved tremendous success, yet the fundamentally more challenging task of synthesizing programs from input-output behavior, which we refer to as IO2Code, remains largely unsolved. Whereas NL2Code can exploit the semantic alignment between natural language and code acquired during pretraining, IO2Code requires recovering underlying principles from concrete computational behavior, navigating a vast and underspecified hypothesis space. To address this, we propose DIO-Agent, a discovery agent for IO2Code. Our method frames IO2Code as an evolutionary search over discrete program space, in which an LLM serves as the mutation operator and concrete error signals from execution guide each mutation. To prevent the search from wandering into structurally complex yet incorrect dead ends, we introduce the Transformation Priority Premise as a mutation prior that biases the LLM toward the simplest hypothesis consistent with current evidence, progressively escalating from constants to conditionals to iteration only when simpler constructs are insufficient. To facilitate systematic study, we further construct an IO2CodeBench spanning multiple difficulty levels. Extensive experiments show that DIO-Agent consistently outperforms both traditional program-by-example method and SOTA evolution-agent baselines across all difficulty levels and various LLMs, while substantially surpassing test-time scaling strategies with equivalent sampling budgets. 

---
# Fortress: A Case Study in Stabilizing Search Recommendations via Temporal Data Augmentation and Feature Pruning 

**Authors**: Milind Pandurang Jagre, Jia Huang, Dayvid V. R. Oliveira, Zhinan Cheng, Babak Seyed Aghazadeh, Puja Das, Chris Alvino, Jinda Han, Kailash Thiyagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2605.15299)  

**Abstract**: In search and recommendation systems, predictive models often suffer from temporal instability when certain input features introduce volatility in output scores. This instability can degrade model reliability and user experience especially in multi-stage systems where consistent predictions are critical for downstream decision making. We introduce Fortress, a general framework for enhancing model stability and accuracy by identifying and pruning features that contribute to inconsistent prediction scores over time. Fortress leverages historical snapshots temporally partitioned datasets capturing score fluctuations for the same entity across periods and follows a four-step process: (1) collect historical snapshots, (2) identify samples with unstable predictions, (3) isolate and remove instability-inducing features, and (4) retrain models using only stable features. While semantic features from LLMs and BERT-based models improve generalization, they often lack full query or entity coverage. Engagement-based features offer strong predictive power but tend to introduce temporal instability. Fortress mitigates this trade-off by suppressing the volatility of engagement signals while retaining their predictive value leading to more stable and accurate models. We validate Fortress on a query-to-app relevance model in a large-scale app marketplace. Offline experiments demonstrate notable improvements in prediction stability (measured by Coefficient of Variation) and classification performance (measured by PR-AUC). 

---
# PhysBrain 1.0 Technical Report 

**Authors**: Shijie Lian, Bin Yu, Xiaopeng Lin, Changti Wu, Hang Yuan, Xiaolin Hu, Zhaolong Shen, Yuzhuo Miao, Haishan Liu, Yuxuan Tian, Yukun Shi, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15298)  

**Abstract**: Vision-language-action models have advanced rapidly, but robot trajectories alone provide limited coverage for learning broad physical understanding. PhysBrain 1.0 studies a complementary route: converting large-scale human egocentric video into structured physical commonsense supervision before robot adaptation. Our data engine extracts scene elements, spatial dynamics, action execution, and depth-aware relations, then turns them into question-answer supervision for training PhysBrain VLMs. The resulting physical priors are further transferred to VLA policies through a capability-preserving and language-sensitive adaptation design. Across multimodal QA benchmarks and embodied control benchmarks, including ERQA, PhysBench, SimplerEnv-WidowX, LIBERO, and RoboCasa, PhysBrain 1.0 achieves SOTA results and shows especially strong out-of-domain performance on SimplerEnv. These results suggest that scaling physical commonsense from human interaction video can provide an effective bridge from multimodal understanding to robot action. 

---
# GESD: Beyond Outcome-Oriented Fairness 

**Authors**: Gideon Popoola, John Sheppard  

**Link**: [PDF](https://arxiv.org/pdf/2605.15295)  

**Abstract**: Machine learning (ML) algorithms are increasingly deployed in high-stakes decision-making domains such as loan approvals, hiring, and recidivism predictions. While existing fairness metrics (e.g., statistical parity, equal opportunity) effectively quantify outcome-oriented disparities, they offer limited insight into the procedure or explanation behind biased decisions. To address this gap, we propose Group-level Explanation Stability Disparity (GESD), a \textit{procedural-oriented} fairness metric that measures disparities in the stability, robustness, and sensitivity of model explanations across different subgroups in a protected category. %GESD is explainer-agnostic, model-agnostic, and extends the scope of fairness analyses to the level of explainability. We further integrate GESD into a multi-objective optimization framework that jointly optimizes for utility, outcome-based fairness, and explanation-based fairness called FEU (Fairness--Explainability--Utility). Empirical results on multiple benchmark datasets show that GESD effectively captures group-wise discrepancies in explanation quality, and that FEU improves both utility and fairness over state-of-the-art methods. By bridging outcome-based and explanation-based fairness, GESD offers a comprehensive tool for diagnosing and mitigating bias in predictive modeling. Our code and datasets are available on GitHub {\hyperlink{this https URL}{this https URL}} 

---
# GQA-μP: The maximal parameterization update for grouped query attention 

**Authors**: Kyle R. Chickering, Huijuan Wang, Mengxi Wu, Alexander Moreno, Muhao Chen, Xuezhe Ma, Daria Soboleva, Joel Hestness, Zhengzhong Liu, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2605.15290)  

**Abstract**: Hyperparameter transfer across model architectures dramatically reduces the amount of compute necessary for tuning large language models (LLMs). The maximal update parameterization ({\mu}P) ensures transfer through principled mathematical analysis but can be challenging to derive for new model architectures. Building on the spectral feature-learning view of Yang et al. (2023a), we make two advances. First, we promote spectral norm conditions on the weights from a heuristic to the definition of feature learning, and as a consequence arrive at the Complete-P depth and weight-decay scalings without recourse to lazy-learning. Second, we consider a modified spectral norm that preserves the valid scaling law of network weights when weight matrices are not full rank. This enables (to our knowledge, the first) derivation of {\mu}P scalings for grouped-query attention (GQA). We demonstrate the efficacy of our theoretical derivations by showing learning rate transfer across the GQA repetition hyperparameter as well as experiments regarding transfer over weight decay. 

---
# Universal Approximation of Nonlinear Operators and Their Derivatives 

**Authors**: Filippo de Feo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15285)  

**Abstract**: Derivative-Informed Operator Learning (DIOL), i.e. learning a (nonlinear) operator and its derivatives, is an open research frontier at the foundations of the influential field of Operator Learning (OL). In particular, Universal Approximation Theorems (UATs) of nonlinear operators and their derivatives are foundational open questions and delicate problems in nonlinear functional analysis. In this manuscript, we prove the first UATs of non-linear $k$-times differentiable operators between Banach spaces and their derivatives, uniformly on compact sets and in weighted Sobolev norms for general finite input measures, via OL architectures. Our results are the first complete generalizations of the corresponding influential classical results in [Hornik, 1991] to infinite-dimensional settings and OL.
We discuss several open areas where DIOL and our UATs find applications: high-order accuracy in OL, fast constrained optimization in Banach spaces (e.g. optimal control of PDEs, inverse problems) and numerical methods for infinite-dimensional PDEs (e.g. HJB PDEs on Banach spaces from optimal control of PDEs, SPDEs, path-dependent systems, partially observed systems, mean-field control).
We parameterize nonlinear operators via Encoder-Decoder Architectures, renowned classes in OL due to their generality, including classical architectures, such as DeepONets, Deep-H-ONets, PCA-Nets. Our results are based on four key features that allow us to prove UATs in full generality: (i) Approximation Properties of Banach spaces. (ii) $k$-times continuous differentiability in the sense of Bastiani (weaker than $k$-times continuous Fréchet differentiability). (iii) Natural compact-open topologies for UA; indeed, we show that UA in standard compact-open topologies induced by operator norms is violated even for Fréchet derivatives. (iv) Construction of novel weighted Sobolev spaces for the UA. 

---
# Autonomous Intelligent Agents for Natural-Language-Driven Web Execution with Integrated Security Assurance 

**Authors**: Vinil Pasupuleti, Siva Rama Krishna Varma Bayyavarapu, Shrey Tyagi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15281)  

**Abstract**: Modern web test suites rot. A UI refactor breaks locators, a timing change causes race conditions, and within weeks developers abandon the suite entirely. This paper presents an AI-driven autonomous testing framework that addresses these failure modes through five integrated strategies - navigation reliability, context-aware selector generation, post-generation validation, smart wait injection, and failure learning - implemented over a containerised worker architecture that decouples orchestration from long-running browser execution. Evaluated across four production applications and 176 scenarios, the framework improves script generation success from 55% to 93%, achieves an 8x reduction in navigation failures, eliminates 80% of timing-related race conditions, and reduces test creation time by 75% compared to manual Selenium authoring. The framework extends naturally to security validation: testers describe attack scenarios in plain English - "try accessing another user's invoice" - which the agent converts to OWASP Top 10-aligned browser probes, detecting 85% of authentication bypass vulnerabilities and 95% of input validation flaws with false positive rates below 12%. Natural-language-driven security testing of this kind represents, to our knowledge, a novel contribution to the field. 

---
# PDRNN: Modular Data-driven Pedestrian Dead Reckoning on Loosely Coupled Radio- and Inertial-Signalstreams 

**Authors**: Peter Bauer, Andreas Porada, Felix Ott, Christopher Mutschler, Tobias Feigl  

**Link**: [PDF](https://arxiv.org/pdf/2605.15252)  

**Abstract**: Modern pedestrian dead reckoning (PDR) systems rely on fusing noisy and biased estimates of position, velocity, and calibrated orientation derived from loosely coupled sensors to determine the current pose of a localized object. However, discrepancies in the sampling rates of sensor-specific estimation methods and unreliable transmission pose significant challenges. And traditional methods often fail to effectively fuse multimodal sensor data during dynamic movements characterized by high accelerations, velocities, and rapidly varying orientations. To address these limitations, we propose a simple recurrent neural network (RNN) architecture capable of implicitly forecasting asynchronous sensor data streams from diverse estimation methods along reference trajectories. The proposed approach introduces PDRNN, a modular hybrid AI-assisted PDR system that handles each component as an independent ensemble of machine learning (ML) models to estimate both key parameter means and variances. Separate ML-based models are employed to estimate orientation, (un)directed velocity or distance from acceleration and gyroscope data, with optional absolute positioning from synchronized radio systems such as 5G for stabilization. A final fusion model combines these outputs, position, velocity, and orientation, while using uncertainty estimates to enhance system robustness. The modular design allows individual components to be updated, fine-tuned, or replaced without affecting the entire system. Experiments on dynamic sports movement data show that PDRNN achieves superior accuracy and precision compared to classic and ML-based methods, effectively avoiding error accumulation common in black-box approaches. And PDRNN offers forecast capabilities and better component control despite increased system complexity. 

---
# GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding 

**Authors**: Fanxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15250)  

**Abstract**: Multi-head Latent Attention (MLA), the attention used in DeepSeek-V2/V3, jointly compresses keys and values into a low-rank latent and matches the H100 roofline almost perfectly. Its trained weights, however, expose only one decoding path - an absorbed MQA form - which ties efficient inference to H100-class compute-bandwidth ratios, forfeits tensor parallelism along the head axis, and yields no Multi-Token Prediction (MTP) gain on commodity inference GPUs such as the export-restricted H20. We propose Group-Query Latent Attention (GQLA), a minimal modification of MLA whose trained weights expose two algebraically equivalent decoding paths over the same parameters: an MQA-absorb path identical to MLA's, and a GQA path with a per-group expanded cache. The runtime picks the path that matches the target hardware - no retraining, no custom kernels - so a single set of GQLA weights pins the rooflines of both H100 (MQA-absorb, s_q=1) and H20 (GQA + MTP, s_q=2), while supporting up to 8-way zero-redundancy tensor parallelism on the GQA path. To avoid pretraining from scratch we extend TransMLA into TransGQLA, which converts a pretrained GQA checkpoint into a GQLA model; on LLaMA-3-8B it compresses the per-token KV cache to 28.125% of the GQA baseline on the MQA-absorb path while structurally preserving GQA-level traffic on the per-group path. 

---
# Reading the Cell, Designing the Cure: Perturbation-Conditioned Molecular Diffusion for Function-Oriented Drug Design 

**Authors**: Ziyu Xu, Zijian Zhang, Liang Wang, Zhiyuan Liu, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15243)  

**Abstract**: When reliable target structures are unavailable at scale or phenotypes arise from dysregulated pathways, transcriptomic perturbations provide a system-level functional readout for drug action. In this work, we formalize \emph{Transcriptome-based Drug Design (TBDD)} as a generative inverse problem: designing drug molecules conditioned on desired transcriptomic state transitions. We analyze the inherently ill-posed nature of this task, which is further complicated by the profound domain gap between biology and chemistry and by the sparsity of transcriptomic signals. To address these challenges, we propose \textbf{\themodel{}} (A \textbf{C}ell\textbf{U}lar \textbf{R}esponse \textbf{E}ngine), a multi-resolution transcriptome-guided diffusion framework. \themodel{} features a specialized \textbf{Transcriptome Perturbation Functional Feature Extractor (TFE)} that (1) distills function-oriented perturbation embeddings from pre/post states, (2) aligns these signatures to dual chemical views to bridge the cross-modal gap, and (3) performs heterogeneity-aware aggregation to extract robust state-specific signals from noisy transcriptomic data. Extensive evaluations on both standard benchmarks and rigorous out-of-distribution protocols demonstrate that \themodel{} consistently outperforms strong baselines in structural quality and functional consistency. Furthermore, we validate its practical utility via a zero-shot gene-inhibitor design task, highlighting the potential of phenotype-driven generative discovery. 

---
# Hydra: Efficient, Correct Code Generation via Checkpoint-and-Rollback Support 

**Authors**: Alexander Du, Jianjun Ou, Danyang Zhuo, Matthew Lentz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15238)  

**Abstract**: Large language models are increasingly used for code generation, but many generated programs fail to compile, a prerequisite for further correctness checks such as unit tests. Existing solutions for repairing static errors are costly in both latency and token consumption. Post-hoc repair delays error detection until generation completes and commonly regenerates large regions of previously valid code. Constrained semantic decoding checks after each token, incurring per-token overhead while limiting repair to the current token even when the root cause lies earlier.
We present Hydra, a system for efficient recovery from static errors during code generation. Hydra allows checking to proceed asynchronously with generation, avoiding checker overhead when the generated code is semantically correct. In addition, it provides checkpoint-and-rollback support for targeted repair, avoiding regeneration and rechecking of valid prefixes. We retrofit the Clang C/C++ compiler to support Hydra with modest modifications. Paired with a token-efficient repair strategy, Hydra reduces latency by up to 71% and token consumption by up to 70% relative to post-hoc repair on C/C++ code generation tasks that encounter static errors. 

---
# A3D: Agentic AI flow for autonomous Accelerator Design 

**Authors**: Abinand Nallathambi, Christopher Knight, Shantanu Ganguly, Wilfried Haensch, Anand Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.15237)  

**Abstract**: Accelerating applications through the design of hardware accelerators can significantly enhance system performance and energy efficiency. Despite advances, such as high-level synthesis (HLS), designing accelerators for complex applications still remains highly labor-intensive, demanding considerable expertise in understanding workloads to be accelerated, hardware design, micro-architecture, and EDA tool usage, posing challenges for application domain experts. Therefore, most accelerator solutions are limited to applications with a regular predictable dataflow. Advances in AI have enabled agents that perform autonomous planning, reasoning, execution and reflection, leading to unprecedented potential for automation through agentic AI. We present A3D, an Agentic AI flow for end-to-end Automation of hardware Accelerator Design. A3D automates workload analysis, performance bottleneck identification, code refactoring for HLS compatibility and micro-architecture generation. A3D also generates diverse accelerator designs by automatically exploring the speed-area tradeoff space. Recent efforts have explored the use of AI for specific tasks such as design space exploration in HLS, leaving several tasks to still be performed manually. A3D addresses the challenges in applying modern LLMs to accelerator design by judiciously partitioning tasks among specialist agents, orchestrating process loops with specialist and verifier agents, utilizing pre-existing and custom tools, and employing agentic RAG for codebase and proprietary EDA tool documentation exploration. Our implementation of A3D, using commercial components like Claude Sonnet 4.5 and the Catapult HLS tool, demonstrates its effectiveness by generating accelerator designs with no human intervention from complex scientific applications like LAMMPS (molecular dynamics simulation) and QMCPACK (quantum chemistry). 

---
# Learning Selective Merge Policies for Deadline-Constrained Coded Caching via Deep Reinforcement Learning 

**Authors**: Amirhossein Yousefiramandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15236)  

**Abstract**: With the coded caching, the server can use the information the users have cached to serve multiple users at a time by sending a single coded multi-casting message, i.e., the merged message, thereby relieving the peak network loads. However, for the delay-sensitive applications of the users, like the video streaming services, it becomes essential to choose which messages to merge online, considering the strict deadlines for each request. The problem, however, is that while the merge is helpful for the formation of the current coded multi-casting message, it can be harmful for the subsequent ones. We proposed a DRL-based solution that formulates the deadline-constrained coded delivery as a masked discrete-action queue-state control problem, while we trained a graph-attention policy network via proximal policy optimization. The policy network reduces the broadcast-packet expiration ratio $\rho$ by $40.9%$ ($0.208$ vs. $0.352$) with respect to the best coded multi-casting baseline (SACM++) on the uniform-demand benchmark, while also attaining the best broadcast-efficiency score $\sigma$ across the Track A battery among the coded multi-casting methods. The interesting fact we observed is that for the applications of the users with tight deadlines, the method of selective merging is better than the method of aggressive merging, i.e., the policy network learns to merge at only $\approx 31.8%$ rate, even though the same observation holds across the variations within the same simulator family. 

---
# PBT-Bench: Benchmarking AI Agents on Property-Based Testing 

**Authors**: Lucas Jing, Xinqi Wang, Liao Zhang, Simon S. Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.15229)  

**Abstract**: Existing code benchmarks measure whether an agent can produce any test that reproduces a known bug, or whether it can produce a
patch that fixes a described issue. Neither isolates the distinct skill of property-based testing: deriving a semantic invariant
from documentation, and then constructing an input-generation strategy precise enough to make a random search reveal the violation.
We introduce PBT-Bench, a benchmark of 100 curated property-based testing problems across 40 real Python libraries. Each problem
injects one or more semantic bugs (365 in total, mean 3.65 per problem) designed so that default-strategy random inputs almost
never trigger them; the agent must read the library's documentation, identify the relevant invariant, and specify a Hypothesis
@given strategy that concentrates mass in the trigger region. Bugs are stratified across three difficulty levels (L1-L3) spanning
single-constraint boundary bugs to stateful, cross-function protocol violations. We evaluate eight contemporary LLMs under two
prompting regimes (open-ended baseline vs. explicit Hypothesis scaffolding) for three independent runs per configuration. Bug
recall under the PBT-guided prompt ranges from 42.1% to 83.4% across models; under the open-ended baseline, from 31.4% to 76.7%.
Hypothesis scaffolding lifts mid-capability models by over 20 percentage points, but yields smaller gains for the strongest models,
with two exceptions showing degradation, suggesting the structured prompt can interfere with certain model behaviours rather than
complementing them. The hardest bugs prove model-specific: different architectures fail on different problems, leaving persistent
gaps that no single model closes. We release the benchmark, harness, and full evaluation corpus to support downstream work on
documentation-grounded semantic reasoning. 

---
# Is Agentic AI Ready for Real-World Hardware Engineering? A Deep Dive with Phoenix-bench 

**Authors**: Qingyun Zou, Feng Yu, Hongshi Tan, Bingsheng He, WengFai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15226)  

**Abstract**: We ask whether agentic AI systems built for software engineering transfer to realistic hardware engineering. Existing hardware LLM benchmarks isolate sub-tasks but none jointly requires repository navigation, hierarchy-aware localization, Electronic Design Automation (EDA) executable verification, and maintenance-style patching. We introduce \textbf{Phoenix-bench}, a synchronized corpus of 511 verified Verilator instances from 114 GitHub repositories, each shipped with the developer patch, design-flow labels, fail-to-pass and pass-to-pass testbenches, and a Docker-pinned EDA environment so resolved-rate differences reflect agent behavior rather than toolchain availability. Using Phoenix-bench we run a uniform evaluation of four commercial agents and eight open-source agentic structures across four LLM backbones, plus two diagnostic interventions (file-level oracle localization and one round of testbench-log feedback). Three findings emerge. (i)~Software and hardware are fundamentally different engineering tasks: the same agent loses 37\% to 58\% from SWE-bench Verified to Phoenix-bench because hardware bugs propagate across parallel instantiated modules through signal flow rather than along a software-style call graph, and software-tuned agents stop at the symptom file instead of tracing back through the instantiation chain. (ii)~Failures concentrate on design control-flow / finite state machine (FSM) bugs, verification testbench bugs, and hard cases that demand cross-hierarchy signal-flow tracking and coordinated multi-file edits. (iii)~Localization granularity matters far more than localization itself: a perfect file-level oracle yields only $+1.4$\% because the agent then breaks files that did not need editing, while a single round of test case feedback lifts resolved rate by $42$\% to $45$\% because the test case tells \emph{where} the bug is and \emph{what} the fix has to look like. 

---
# Do Biological Structural Guarantees Earn Their Complexity? 

**Authors**: Bogdan Banu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15225)  

**Abstract**: Biologically-inspired AI agent frameworks claim reliability benefits through structural guarantees adapted from gene regulatory networks, immune systems, and metabolic control. These claims are rarely tested empirically against simpler alternatives. We present three deep benchmarks: metabolic priority gating, autoinducer-based quorum sensing, and Bayesian stagnation detection, each comparing a biologically-grounded implementation against a naive non-biological alternative and an ablated control, across 1,000 trials per seed and 10 seeds (10M+ data points total). 

---
# GenAI-Driven Approach to RISC-V Supply Chain Exploration 

**Authors**: Nenad Petrovic, Andre Schamschurko, Yingjie Xu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2605.15223)  

**Abstract**: This paper presents an LLM-empowered workflow for RISC-V supply chain analysis, integrating Vision-Language Models (VLMs) and Model-Driven Engineering (MDE) to enable comprehensive, multimodal data-driven insights. The proposed approach addresses the challenges of heterogeneous and unstructured supply chain data by leveraging LLMs for textual understanding and VLMs for extracting information from visual artifacts such as diagrams, tables, and scanned documents. These models collaboratively identify key entities and relationships, which are then organized into a knowledge graph representing supply chain components and their interdependencies. For analytical reasoning, the workflow incorporates MDE techniques and constraint-based modeling to enable formal validation of dependencies, detection of bottlenecks, and assessment of risks. The synergy between LLM- and VLM-based semantic understanding and MDE-based formal analysis supports both exploratory and systematic evaluation of supply chain resilience. A human-in-the-loop mechanism further enables interactive querying and expert validation. The approach is evaluated in RISC-V ecosystem scenarios, demonstrating its effectiveness in generating actionable insights, enhancing transparency, and supporting decision-making in complex semiconductor supply chains. 

---
# Effective Harness Engineering for Algorithm Discovery with Coding Agents 

**Authors**: Yoichi Ishibashi, Taro Yano, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.15221)  

**Abstract**: AlphaEvolve and FunSearch have demonstrated the potential of combining large language models (LLMs) with evolutionary search for automated algorithm discovery. However, discovery success is shaped not only by model capability but also significantly by the design of the execution infrastructure, i.e., the harness. This paper investigates effective harness design through three questions: under a fixed token budget, is it better to produce many algorithms with brief thought or fewer algorithms with deeper thought? How should the harness handle evaluation hacks, where generated programs exploit the scoring function? And how can agents that require full filesystem access execute safely in parallel? Using Vesper, an algorithm discovery framework that incorporates harness improvements addressing these questions, we evaluate on Circle Packing under the same token budget. Interestingly, generating fewer algorithms while thinking more deeply about each one achieved higher scores. That is, scaling the quality of each individual is more budget-efficient than scaling the number of evolutionary generations. Surprisingly, more capable models produced evaluation hacks at higher rates, making hack detection increasingly necessary as models scale. 

---
# Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time 

**Authors**: Michael Y. Hu, Apurva Gandhi, Kyunghyun Cho, Tal Linzen, Pratyusha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15220)  

**Abstract**: Data mixing decides how to combine different sources or types of data and is a consequential problem throughout language model training. In pretraining, data composition is a key determinant of model quality; in continual learning and adaptation, it governs what is retained and acquired. Yet existing data mixing methods address only one phase of this lifecycle at a time: some require smaller proxy models tied to a single training phase, others assume a fixed domain set, and continual learning lacks principled guidance altogether. We argue that data mixing is fundamentally an online decision making problem -- one that recurs throughout training and demands a single, unified solution. We introduce OP-Mix (On-Policy Mix), a data mixing algorithm that operates across the entire language model training lifecycle. Our main insight is that candidate data mixtures can be cheaply simulated by interpolating between low-rank adapters trained directly on the current model, eliminating separate proxy models and ensuring the search is always grounded in the model's actual learning dynamics. Across pretraining, continual midtraining, and continual instruction tuning, OP-Mix consistently finds near-optimal mixtures while using a fraction of the compute of the baselines. In pretraining, OP-Mix improves upon training without mixing by 6.3% in average perplexity. For continual learning, OP-Mix matches the performance of both retraining and on-policy distillation while using 66% and 95% less overall compute, respectively. OP-Mix suggests a different view of language model training: not a sequence of distinct phases, but a single continuous process of learning from data. 

---
# An LLM-RAG Approach for Healthy Eating Index-Informed Personalized Food Recommendations 

**Authors**: Yibin Wang, Yanjie Yang, Grace Melo Guerrero, Rodolfo M. Nayga Jr., Azlan Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.15213)  

**Abstract**: Diet quality is a leading determinant of chronic disease risk. Advances in artificial intelligence (AI) have enabled food recommendation systems to adapt suggestions to user preferences and health goals. However, most current systems rely on loosely curated food databases and provide limited connection to a validated index. In this study, we propose a Healthy Eating Index (HEI) informed retrieval-augmented generation (RAG) framework that combines standardized nutrition databases with large language models (LLMs) for personalized food recommendations. Our proposed method anchors retrieval in the National Health and Nutrition Examination Survey (NHANES) and the Food Patterns Equivalents Database (FPED). A food-level embedding space is constructed from FPED-derived textual descriptions. For each entity, the system computes baseline HEI scores, retrieves candidate foods for intake recommendations, and estimates the HEI impact of simple substitutions or additions. A constrained RAG pipeline instantiated with a pretrained OpenAI LLM generates personalized recommendations and sources based on nutrient profiles and HEI contributions. The simulation results showed a mean HEI improvement of 6.45, with the proportion of users HEI over 50 increasing from 45.12 to 61.26. Quantile analysis revealed consistent improved shifts across the HEI distribution. Our findings suggest that the proposed LLM-RAG-based AI systems can support more precise, explainable, and personalized nutrition guidance to improve diet quality. 

---
# Fault tolerance estimation in digital circuits with visualised generative networks 

**Authors**: Sascha Biel, Carl Alexander Gaede, Amiel Glaser, Jan Wolter, Alexej Schelle  

**Link**: [PDF](https://arxiv.org/pdf/2605.15212)  

**Abstract**: We propose a new numerical method to estimate the fault tolerance of failure modes in digital circuit structures with a generative network sampling technique. From a random input of generated bitwise configurations of ideally digitalised analog currents in the digital circuit design with classical logical gates, expected output currents are compared to the realistic signals of a numerical experiment at the discriminator part of the Generative Adversarial Network (GAN) to calculate the deviation from ideal digital electronic signals, including various error modes, such as missing or interchanged logical devices. From the present analysis of a representation of the GAN in terms of complex variables, it is possible to evaluate the robustness in electronic designs by differentiating the impact of failure modes associated with different classical logical elements in the circuit. 

---
# Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels 

**Authors**: Plawan Kumar Rath, Rahul Maliakkal  

**Link**: [PDF](https://arxiv.org/pdf/2605.15208)  

**Abstract**: Large Language Models are routinely compressed via post-training quantization to reduce inference costs and memory footprint for cloud and edge deployment, yet the impact of this compression on model quality remains poorly understood. Existing studies typically compare only two conditions (full-precision vs. a single quantized variant), rely on aggregate bias metrics, and evaluate a single model family, making it impossible to distinguish gradual degradation from threshold-dependent safety failures. We conduct a controlled empirical study of three instruction-tuned models (Qwen2.5-7B, Mistral-7B, Phi-3.5-mini) at five precision levels (BF16 through 3-bit) on 12,148 BBQ bias benchmark items across 5 random seeds, totaling 911,100 inference records. Our results reveal that 3-bit quantization causes 6-21% of previously unbiased items to develop new stereotypical behaviors, following a clear dose-response pattern confirmed via logistic regression, while models' willingness to select "unknown" answers declines by 17.4%. Crucially, these item-level changes are invisible to standard quality metrics: perplexity increases by less than 0.5% at 8-bit and under 3% at 4-bit across all three models, yet 2.5-5.6% of items already develop new biases at 4-bit. These findings demonstrate that aggregate evaluation metrics systematically miss fairness-critical degradation, underscoring the need for quality-aware compression protocols that explicitly test for bias emergence before deployment. 

---
# AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Devices 

**Authors**: Dzung Pham, Kleomenis Katevas, Ali Shahin Shamsabadi, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15206)  

**Abstract**: Autonomous agents powered by large language models (LLMs) are increasingly used to automate complex, multi-step tasks such as coding or web-based question answering. While remote, cloud-based agents offer scalability and ease of deployment, they raise privacy concerns, depend on network connectivity, and incur recurring API costs. Deploying agents locally on user devices mitigates these issues by preserving data privacy and eliminating usage-based fees. However, agentic workflows are far more resource-intensive than typical LLM interactions. Iterative reasoning, tool use, and failure retries substantially increase token consumption, often expending significant compute without successfully completing tasks.
In this work, we investigate the time, token, and energy overhead of locally deployed LLM-based agents on consumer hardware. Our measurements show that agentic execution increases GPU power draw, temperature, and battery drain compared to single-inference workloads. To address this inefficiency, we introduce AgentStop, a lightweight efficiency supervisor that predicts and preemptively terminates trajectories unlikely to succeed. Leveraging low-cost execution signals, such as token-level log probabilities, AgentStop can reduce wasted energy by 15-20% with minimal impact on task performance (<5% utility drop) for challenging web-based question answering and coding benchmarks. These findings position predictive early termination as a practical mechanism for enabling sustainable, privacy-preserving LLM agents on user devices. Our project code and data are available at this https URL. 

---
# Agent4POI: Agentic Context-Conditioned Affordance Reasoning for Multimodal Point-of-Interest Recommendation 

**Authors**: Jinze Wang, Yangchen Zeng, Tiehua Zhang, Lu Zhang, Yuze Liu, Yongchao Liu, Xingjun Ma, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.15203)  

**Abstract**: We introduce Agent4POI, the first POI recommendation framework that generates context-conditioned multimodal representations at recommendation time, rather than relying on static POI embeddings pre-computed independently of context. Existing multimodal systems encode each POI once as a static embedding, a design that precludes reasoning about why the same cafe affords solo work on Monday but group celebration on Friday evening. We formally prove that no pre-computed encoder can satisfy context-sensitive ranking under standard bilinear scoring, motivating inference-time item-side representation. Agent4POI inverts this computation: given a situational context, a four-phase LLM agent generates dynamic, context-specific affordance queries (Phase 1) and executes a five-step cross-modal chain-of-thought over image, review, and metadata evidence (Phase 2). The resulting uncertainty-aware affordance representation is grounded in Gibsonian affordance theory. These cross-modal verdicts form a structured, uncertainty-adjusted affordance representation (Phase 3), which is aligned with user preferences via a semantic caching system for low-latency ranking (Phase 4). On three POI benchmarks and three evaluation configurations (standard, cold-start, context-shift), Agent4POI achieves a 23.2% relative gain over the strongest baseline and degrades by only 7.5% under context-shift versus 16--17\% for the strongest baselines. In cold-start scenarios, Agent4POI outperforms the best content-based baseline by up to 2.4x, whereas ID-based methods fail to generalize. 

---
# Ensuring Logic in the Fog: Sound POMDP Synthesis with LTL Objectives 

**Authors**: Can Zhou, Yulong Gao, Pian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12581)  

**Abstract**: Synthesising autonomous agents that can navigate uncertain environments while adhering to complex temporal constraints remains a fundamental challenge. While Linear Temporal Logic (LTL) provides a rigorous language for specifying such tasks, the inherent undecidability of qualitatively verifying LTL satisfaction in partially observable Markov decision processes renders quantitative synthesis difficult, especially when designing reliable reward signals for approximate solvers. In this paper, we bridge this gap with a novel, sound reward-shaping mechanism that dynamically generates belief-dependent rewards grounded in certified LTL satisfaction. By integrating this mechanism into an enhanced Monte Carlo Planning framework, we empower agents to navigate the `fog' of partial observability with a search process focused on maximising verifiable success. Our experiments demonstrate that this approach not only thrives in scenarios where existing solvers fail but also maintains effectiveness and scalability across diverse benchmark domains. 

---
