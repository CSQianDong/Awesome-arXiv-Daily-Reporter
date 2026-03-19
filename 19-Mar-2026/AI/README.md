# AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse 

**Authors**: Zhang Zhang, Shuqi Lu, Hongjin Qian, Di He, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18000)  

**Abstract**: Building LLM-based agents has become increasingly important. Recent works on LLM-based agent self-evolution primarily record successful experiences as textual prompts or reflections, which cannot reliably guarantee efficient task re-execution in complex scenarios. We propose AgentFactory, a new self-evolution paradigm that preserves successful task solutions as executable subagent code rather than textual experience. Crucially, these subagents are continuously refined based on execution feedback, becoming increasingly robust and efficient as more tasks are encountered. Saved subagents are pure Python code with standardized documentation, enabling portability across any Python-capable system. We demonstrate that AgentFactory enables continuous capability accumulation: its library of executable subagents grows and improves over time, progressively reducing the effort required for similar tasks without manual intervention. Our implementation is open-sourced at this https URL, and our demonstration video is available at this https URL. 

---
# RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy 

**Authors**: Zhenhang Yuan, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.17831)  

**Abstract**: LLM agents often fail in closed-world embodied environments because actions must satisfy strict preconditions -- such as location, inventory, and container states -- and failure feedback is sparse. We identify two structurally coupled failure modes: (P1) invalid action generation and (P2) state drift, each amplifying the other in a degenerative cycle. We present RPMS, a conflict-managed architecture that enforces action feasibility via structured rule retrieval, gates memory applicability via a lightweight belief state, and resolves conflicts between the two sources via rules-first arbitration. On ALFWorld (134 unseen tasks), RPMS achieves 59.7% single-trial success with Llama 3.1 8B (+23.9 pp over baseline) and 98.5% with Claude Sonnet 4.5 (+11.9 pp); of the 8B gain, rule retrieval alone contributes +14.9 pp (statistically significant), making it the dominant factor. A key finding is that episodic memory is conditionally useful: it harms performance on some task types when used without grounding, but becomes a stable net positive once filtered by current state and constrained by explicit action rules. Adapting RPMS to ScienceWorld with GPT-4 yields consistent gains across all ablation conditions (avg. score 54.0 vs. 44.9 for the ReAct baseline), providing transfer evidence that the core mechanisms hold across structurally distinct environments. 

---
# Governed Memory: A Production Architecture for Multi-Agent Workflows 

**Authors**: Hamed Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17787)  

**Abstract**: Enterprise AI deploys dozens of autonomous agent nodes across workflows, each acting on the same entities with no shared memory and no common governance. We identify five structural challenges arising from this memory governance gap: memory silos across agent workflows; governance fragmentation across teams and tools; unstructured memories unusable by downstream systems; redundant context delivery in autonomous multi-step executions; and silent quality degradation without feedback loops. We present Governed Memory, a shared memory and governance layer addressing this gap through four mechanisms: a dual memory model combining open-set atomic facts with schema-enforced typed properties; tiered governance routing with progressive context delivery; reflection-bounded retrieval with entity-scoped isolation; and a closed-loop schema lifecycle with AI-assisted authoring and automated per-property refinement. We validate each mechanism through controlled experiments (N=250, five content types): 99.6% fact recall with complementary dual-modality coverage; 92% governance routing precision; 50% token reduction from progressive delivery; zero cross-entity leakage across 500 adversarial queries; 100% adversarial governance compliance; and output quality saturation at approximately seven governed memories per entity. On the LoCoMo benchmark, the architecture achieves 74.8% overall accuracy, confirming that governance and schema enforcement impose no retrieval quality penalty. The system is in production at this http URL. 

---
# Facts as First Class Objects: Knowledge Objects for Persistent LLM Memory 

**Authors**: Oliver Zahn, Simran Chana  

**Link**: [PDF](https://arxiv.org/pdf/2603.17781)  

**Abstract**: Large language models increasingly serve as persistent knowledge workers, with in-context memory - facts stored in the prompt - as the default strategy. We benchmark in-context memory against Knowledge Objects (KOs), discrete hash-addressed tuples with O(1) retrieval. Within the context window, Claude Sonnet 4.5 achieves 100% exact-match accuracy from 10 to 7,000 facts (97.5% of its 200K window). However, production deployment reveals three failure modes: capacity limits (prompts overflow at 8,000 facts), compaction loss (summarization destroys 60% of facts), and goal drift (cascading compaction erodes 54% of project constraints while the model continues with full confidence). KOs achieve 100% accuracy across all conditions at 252x lower cost. On multi-hop reasoning, KOs reach 78.9% versus 31.6% for in-context. Cross-model replication across four frontier models confirms compaction loss is architectural, not model-specific. We additionally show that embedding retrieval fails on adversarial facts (20% precision at 1) and that neural memory (Titans) stores facts but fails to retrieve them on demand. We introduce density-adaptive retrieval as a switching mechanism and release the benchmark suite. 

---
# From Virtual Environments to Real-World Trials: Emerging Trends in Autonomous Driving 

**Authors**: A. Humnabadkar, A. Sikdar, B. Cave, H. Zhang, N. Bessis, A. Behera  

**Link**: [PDF](https://arxiv.org/pdf/2603.17714)  

**Abstract**: Autonomous driving technologies have achieved significant advances in recent years, yet their real-world deployment remains constrained by data scarcity, safety requirements, and the need for generalization across diverse environments. In response, synthetic data and virtual environments have emerged as powerful enablers, offering scalable, controllable, and richly annotated scenarios for training and evaluation. This survey presents a comprehensive review of recent developments at the intersection of autonomous driving, simulation technologies, and synthetic datasets. We organize the landscape across three core dimensions: (i) the use of synthetic data for perception and planning, (ii) digital twin-based simulation for system validation, and (iii) domain adaptation strategies bridging synthetic and real-world data. We also highlight the role of vision-language models and simulation realism in enhancing scene understanding and generalization. A detailed taxonomy of datasets, tools, and simulation platforms is provided, alongside an analysis of trends in benchmark design. Finally, we discuss critical challenges and open research directions, including Sim2Real transfer, scalable safety validation, cooperative autonomy, and simulation-driven policy learning, that must be addressed to accelerate the path toward safe, generalizable, and globally deployable autonomous driving systems. 

---
# MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment 

**Authors**: Yusen Wu, Yiran Liu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17694)  

**Abstract**: In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), leveraging the inherent generalization capabilities of large-sacle models to establish a unified simulation framework applicable to cross-domain and cross-category scenarios. Central to our approach is a preference learning paradigm in which LLMs are economically aligned via post-training on extensive, heterogeneous transaction records across diverse product categories. This methodology enables the models to internalize and transfer latent consumer preference patterns, thereby mitigating the data sparsity issues prevalent in individual categories. To enhance simulation stability, we implement a mean-field mechanism designed to model the dynamic interactions between the product environment and customer populations, effectively stabilizing sampling processes within high-dimensional decision spaces. Furthermore, we propose a multi-agent discussion framework wherein specialized agents collaboratively process extensive product information. This architecture distributes cognitive load to alleviate single-agent attention bottlenecks and captures critical decision factors through structured dialogue. Experiments demonstrate that our framework achieves significant improvements in product selection accuracy, purchase quantity prediction, and simulation stability compared to existing economic and financial LLM simulation baselines. Our results substantiate the potential of large language models as a foundational pillar for high-fidelity, scalable decision simulation and latter analysis in the real economy based on foundational database. 

---
# Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents 

**Authors**: Mohsen Arjmandi  

**Link**: [PDF](https://arxiv.org/pdf/2603.17683)  

**Abstract**: Large language model (LLM) agents deployed in unknown environments must learn task structure at test time, but current approaches require thousands of interactions to form useful hypotheses. We present Sensi, an LLM agent architecture for the ARC-AGI-3 game-playing challenge that introduces structured test-time learning through three mechanisms: (1) a two-player architecture separating perception from action, (2) a curriculum-based learning system managed by an external state machine, and (3) a database-as-control-plane that makes the agents context window programmatically steerable. We further introduce an LLM-as-judge component with dynamically generated evaluation rubrics to determine when the agent has learned enough about one topic to advance to the next. We report results across two iterations: Sensi v1 solves 2 game levels using the two-player architecture alone, while Sensi v2 adds curriculum learning and solves 0 levels - but completes its entire learning curriculum in approximately 32 action attempts, achieving 50-94x greater sample efficiency than comparable systems that require 1600-3000 attempts. We precisely diagnose the failure mode as a self-consistent hallucination cascade originating in the perception layer, demonstrating that the architectural bottleneck has shifted from learning efficiency to perceptual grounding - a more tractable problem. 

---
# VeriGrey: Greybox Agent Validation 

**Authors**: Yuntong Zhang, Sungmin Kang, Ruijie Meng, Marcel Böhme, Abhik Roychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.17639)  

**Abstract**: Agentic AI has been a topic of great interest recently. A Large Language Model (LLM) agent involves one or more LLMs in the back-end. In the front end, it conducts autonomous decision-making by combining the LLM outputs with results obtained by invoking several external tools. The autonomous interactions with the external environment introduce critical security risks.
In this paper, we present a grey-box approach to explore diverse behaviors and uncover security risks in LLM agents. Our approach VeriGrey uses the sequence of tools invoked as a feedback function to drive the testing process. This helps uncover infrequent but dangerous tool invocations that cause unexpected agent behavior. As mutation operators in the testing process, we mutate prompts to design pernicious injection prompts. This is carefully accomplished by linking the task of the agent to an injection task, so that the injection task becomes a necessary step of completing the agent functionality. Comparing our approach with a black-box baseline on the well-known AgentDojo benchmark, VeriGrey achieves 33% additional efficacy in finding indirect prompt injection vulnerabilities with a GPT-4.1 back-end.
We also conduct real-world case studies with the widely used coding agent Gemini CLI, and the well-known OpenClaw personal assistant. VeriGrey finds prompts inducing several attack scenarios that could not be identified by black-box approaches. In OpenClaw, by constructing a conversation agent which employs mutational fuzz testing as needed, VeriGrey is able to discover malicious skill variants from 10 malicious skills (with 10/10= 100% success rate on the Kimi-K2.5 LLM backend, and 9/10= 90% success rate on Opus 4.6 LLM backend). This demonstrates the value of a dynamic approach like VeriGrey to test agents, and to eventually lead to an agent assurance framework. 

---
# Per-Domain Generalizing Policies: On Learning Efficient and Robust Q-Value Functions (Extended Version with Technical Appendix) 

**Authors**: Nicola J. Müller, Moritz Oster, Isabel Valera, Jörg Hoffmann, Timo P. Gros  

**Link**: [PDF](https://arxiv.org/pdf/2603.17544)  

**Abstract**: Learning per-domain generalizing policies is a key challenge in learning for planning. Standard approaches learn state-value functions represented as graph neural networks using supervised learning on optimal plans generated by a teacher planner. In this work, we advocate for learning Q-value functions instead. Such policies are drastically cheaper to evaluate for a given state, as they need to process only the current state rather than every successor. Surprisingly, vanilla supervised learning of Q-values performs poorly as it does not learn to distinguish between the actions taken and those not taken by the teacher. We address this by using regularization terms that enforce this distinction, resulting in Q-value policies that consistently outperform state-value policies across a range of 10 domains and are competitive with the planner LAMA-first. 

---
# Informative Semi-Factuals for XAI: The Elaborated Explanations that People Prefer 

**Authors**: Saugat Aryal, Mark T. Keane  

**Link**: [PDF](https://arxiv.org/pdf/2603.17534)  

**Abstract**: Recently, in eXplainable AI (XAI), $\textit{even if}$ explanations -- so-called semi-factuals -- have emerged as a popular strategy that explains how a predicted outcome $\textit{can remain the same}$ even when certain input-features are altered. For example, in the commonly-used banking app scenario, a semi-factual explanation could inform customers about better options, other alternatives for their successful application, by saying "$\textit{Even if}$ you asked for double the loan amount, you would still be accepted". Most semi-factuals XAI algorithms focus on finding maximal value-changes to a single key-feature that do $\textit{not}$ alter the outcome (unlike counterfactual explanations that often find minimal value-changes to several features that alter the outcome). However, no current semi-factual method explains $\textit{why}$ these extreme value-changes do not alter outcomes; for example, a more informative semi-factual could tell the customer that it is their good credit score that allows them to borrow double their requested loan. In this work, we advance a new algorithm -- the $\textit{informative semi-factuals}$ (ISF) method -- that generates more elaborated explanations supplementing semi-factuals with information about additional $\textit{hidden features}$ that influence an automated decision. Experimental results on benchmark datasets show that this ISF method computes semi-factuals that are both informative and of high-quality on key metrics. Furthermore, a user study shows that people prefer these elaborated explanations over the simpler semi-factual explanations generated by current methods. 

---
# When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution 

**Authors**: Yi Nian, Haosen Cao, Shenzhe Zhu, Henry Peng Zou, Qingqing Luan, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17445)  

**Abstract**: When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? Multi-agent language systems increasingly rely on structured interactions such as delegation and iterative refinement, yet the final output often obscures the underlying interaction topology and agent contributions. We introduce IET (Implicit Execution Tracing), a metadata-independent framework that enables token-level attribution directly from generated text and a simple mechanism for interaction topology reconstruction. During generation, agent-specific keyed signals are embedded into the token distribution, transforming the text into a self-describing execution trace detectable only with a secret key. At detection time, a transition-aware scoring method identifies agent handover points and reconstructs the interaction graph. Experiments show that IET recovers agent segments and coordination structure with high accuracy while preserving generation quality, enabling privacy-preserving auditing for multi-agent language systems. 

---
# Proactive Knowledge Inquiry in Doctor-Patient Dialogue: Stateful Extraction, Belief Updating, and Path-Aware Action Planning 

**Authors**: Zhenhai Pan, Yan Liu, Jia You  

**Link**: [PDF](https://arxiv.org/pdf/2603.17425)  

**Abstract**: Most automated electronic medical record (EMR) pipelines remain output-oriented: they transcribe, extract, and summarize after the consultation, but they do not explicitly model what is already known, what is still missing, which uncertainty matters most, or what question or recommendation should come next. We formulate doctor-patient dialogue as a proactive knowledge-inquiry problem under partial observability. The proposed framework combines stateful extraction, sequential belief updating, gap-aware state modeling, hybrid retrieval over objectified medical knowledge, and a POMDP-lite action planner. Instead of treating the EMR as the only target artifact, the framework treats documentation as the structured projection of an ongoing inquiry loop. To make the formulation concrete, we report a controlled pilot evaluation on ten standardized multi-turn dialogues together with a 300-query retrieval benchmark aggregated across dialogues. On this pilot protocol, the full framework reaches 83.3% coverage, 80.0% risk recall, 81.4% structural completeness, and lower redundancy than the chunk-only and template-heavy interactive baselines. These pilot results do not establish clinical generalization; rather, they suggest that proactive inquiry may be methodologically interesting under tightly controlled conditions and can be viewed as a conceptually appealing formulation worth further investigation for dialogue-based EMR generation. This work should be read as a pilot concept demonstration under a controlled simulated setting rather than as evidence of clinical deployment readiness. No implication of clinical deployment readiness, clinical safety, or real-world clinical utility should be inferred from this pilot protocol. 

---
# From Digital Twins to World Models:Opportunities, Challenges, and Applications for Mobile Edge General Intelligence 

**Authors**: Jie Zheng, Dusit Niyato, Changyuan Zhao, Jiawen Kang, Jiacheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17420)  

**Abstract**: The rapid evolution toward 6G and beyond communication systems is accelerating the convergence of digital twins and world models at the network edge. Traditional digital twins provide high-fidelity representations of physical systems and support monitoring, analysis, and offline optimization. However, in highly dynamic edge environments, they face limitations in autonomy, adaptability, and scalability. This paper presents a systematic survey of the transition from digital twins to world models and discusses its role in enabling edge general intelligence (EGI). First, the paper clarifies the conceptual differences between digital twins and world models and highlights the shift from physics-based, centralized, and system-centric replicas to data-driven, decentralized, and agent-centric internal models. This discussion helps readers gain a clear understanding of how this transition enables more adaptive, autonomous, and resource-efficient intelligence at the network edge. The paper reviews the design principles, architectures, and key components of world models, including perception, latent state representation, dynamics learning, imagination-based planning, and memory. In addition, it examines the integration of world models and digital twins in wireless EGI systems and surveys emerging applications in integrated sensing and communications, semantic communication, air-ground networks, and low-altitude wireless networks. Finally, this survey provides a systematic roadmap and practical insights for designing world-model-driven edge intelligence systems in wireless and edge computing environments. It also outlines key research challenges and future directions toward scalable, reliable, and interoperable world models for edge-native agentic AI. 

---
# Towards Safer Large Reasoning Models by Promoting Safety Decision-Making before Chain-of-Thought Generation 

**Authors**: Jianan Chen, Zhifang Zhang, Shuo He, Linan Yue, Lei Feng, Minling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17368)  

**Abstract**: Large reasoning models (LRMs) achieved remarkable performance via chain-of-thought (CoT), but recent studies showed that such enhanced reasoning capabilities are at the expense of significantly degraded safety capabilities. In this paper, we reveal that LRMs' safety degradation occurs only after CoT is enabled, and this degradation is not observed when CoT is disabled. This observation motivates us to consider encouraging LRMs to make safety decisions before CoT generation. To this end, we propose a novel safety alignment method that promotes the safety decision-making of LRMs before starting CoT generation. Specifically, we first utilize a Bert-based classifier to extract safety decision signals from a safe model (e.g., a CoT-disabled LRM) and then integrate these signals into LRMs' safety alignment as auxiliary supervision. In this way, the safety gradients can be backpropagated to the LRMs' latent representations, effectively strengthening the LRMs' safety decision-making abilities against CoT generation. Extensive experiments demonstrate that our method substantially improves the safety capabilities of LRMs while effectively maintaining LRMs' general reasoning performance. 

---
# A Progressive Visual-Logic-Aligned Framework for Ride-Hailing Adjudication 

**Authors**: Weiming Wu, Zi-Jian Cheng, Jie Meng, Peng Zhen, Shan Huang, Qun Li, Guobin Wu, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.17328)  

**Abstract**: The efficient adjudication of responsibility disputes is pivotal for maintaining marketplace fairness. However, the exponential surge in ride-hailing volume renders manual review intractable, while conventional automated methods lack the reasoning transparency required for quasi-judicial decisions. Although Multimodal LLMs offer a promising paradigm, they fundamentally struggle to bridge the gap between general visual semantics and rigorous evidentiary protocols, often leading to perceptual hallucinations and logical looseness. To address these systemic misalignments, we introduce RideJudge, a Progressive Visual-Logic-Aligned Framework. Instead of relying on generic pre-training, we bridge the semantic gap via SynTraj, a synthesis engine that grounds abstract liability concepts into concrete trajectory patterns. To resolve the conflict between massive regulation volume and limited context windows, we propose an Adaptive Context Optimization strategy that distills expert knowledge, coupled with a Chain-of-Adjudication mechanism to enforce active evidentiary inquiry. Furthermore, addressing the inadequacy of sparse binary feedback for complex liability assessment, we implement a novel Ordinal-Sensitive Reinforcement Learning mechanism that calibrates decision boundaries against hierarchical severity. Extensive experiments show that our RideJudge-8B achieves 88.41\% accuracy, surpassing 32B-scale baselines and establishing a new standard for interpretable adjudication. 

---
# ShuttleEnv: An Interactive Data-Driven RL Environment for Badminton Strategy Modeling 

**Authors**: Ang Li, Xinyang Gong, Bozhou Chen, Yunlong Lu, Jiaming Ji, Yongyi Wang, Yaodong Yang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.17324)  

**Abstract**: We present ShuttleEnv, an interactive and data-driven simulation environment for badminton, designed to support reinforcement learning and strategic behavior analysis in fast-paced adversarial sports. The environment is grounded in elite-player match data and employs explicit probabilistic models to simulate rally-level dynamics, enabling realistic and interpretable agent-opponent interactions without relying on physics-based simulation. In this demonstration, we showcase multiple trained agents within ShuttleEnv and provide live, step-by-step visualization of badminton rallies, allowing attendees to explore different play styles, observe emergent strategies, and interactively analyze decision-making behaviors. ShuttleEnv serves as a reusable platform for research, visualization, and demonstration of intelligent agents in sports AI. Our ShuttleEnv demo video URL: this https URL 

---
# Physics-informed offline reinforcement learning eliminates catastrophic fuel waste in maritime routing 

**Authors**: Aniruddha Bora, Julie Chalfant, Chryssostomos Chryssostomidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.17319)  

**Abstract**: International shipping produces approximately 3% of global greenhouse gas emissions, yet voyage routing remains dominated by heuristic methods. We present PIER (Physics-Informed, Energy-efficient, Risk-aware routing), an offline reinforcement learning framework that learns fuel-efficient, safety-aware routing policies from physics-calibrated environments grounded in historical vessel tracking data and ocean reanalysis products, requiring no online simulator. Validated on one full year (2023) of AIS data across seven Gulf of Mexico routes (840 episodes per method), PIER reduces mean CO2 emissions by 10% relative to great-circle routing. However, PIER's primary contribution is eliminating catastrophic fuel waste: great-circle routing incurs extreme fuel consumption (>1.5x median) in 4.8% of voyages; PIER reduces this to 0.5%, a 9-fold reduction. Per-voyage fuel variance is 3.5x lower (p<0.001), with bootstrap 95% CI for mean savings [2.9%, 15.7%]. Partial validation against observed AIS vessel behavior confirms consistency with the fastest real transits while exhibiting 23.1x lower variance. Crucially, PIER is forecast-independent: unlike A* path optimization whose wave protection degrades 4.5x under realistic forecast uncertainty, PIER maintains constant performance using only local observations. The framework combines physics-informed state construction, demonstration-augmented offline data, and a decoupled post-hoc safety shield, an architecture that transfers to wildfire evacuation, aircraft trajectory optimization, and autonomous navigation in unmapped terrain. 

---
# InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning 

**Authors**: Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17310)  

**Abstract**: Large Language Models (LLMs) with extended reasoning capabilities often generate verbose and redundant reasoning traces, incurring unnecessary computational cost. While existing reinforcement learning approaches address this by optimizing final response length, they neglect the quality of intermediate reasoning steps, leaving models vulnerable to reward hacking. We argue that verbosity is not merely a length problem, but a symptom of poor intermediate reasoning quality. To investigate this, we conduct an empirical study tracking the conditional entropy of the answer distribution across reasoning steps. We find that high-quality reasoning traces exhibit two consistent properties: low uncertainty convergence and monotonic progress. These findings suggest that high-quality reasoning traces are informationally dense, that is, each step contributes meaningful entropy reduction relative to the total reasoning length. Motivated by this, we propose InfoDensity, a reward framework for RL training that combines an AUC-based reward and a monotonicity reward as a unified measure of reasoning quality, weighted by a length scaling term that favors achieving equivalent quality more concisely. Experiments on mathematical reasoning benchmarks demonstrate that InfoDensity matches or surpasses state-of-the-art baselines in accuracy while significantly reducing token usage, achieving a strong accuracy-efficiency trade-off. 

---
# Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations 

**Authors**: Haozheng Luo, Yimin Wang, Jiahao Yu, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17305)  

**Abstract**: We propose CRAFT, a red-teaming alignment framework that leverages model reasoning capabilities and hidden representations to improve robustness against jailbreak attacks. Unlike prior defenses that operate primarily at the output level, CRAFT aligns large reasoning models to generate safety-aware reasoning traces by explicitly optimizing objectives defined over the hidden state space. Methodologically, CRAFT integrates contrastive representation learning with reinforcement learning to separate safe and unsafe reasoning trajectories, yielding a latent-space geometry that supports robust, reasoning-level safety alignment. Theoretically, we show that incorporating latent-textual consistency into GRPO eliminates superficially aligned policies by ruling them out as local optima. Empirically, we evaluate CRAFT on multiple safety benchmarks using two strong reasoning models, Qwen3-4B-Thinking and R1-Distill-Llama-8B, where it consistently outperforms state-of-the-art defenses such as IPO and SafeKey. Notably, CRAFT delivers an average 79.0% improvement in reasoning safety and 87.7% improvement in final-response safety over the base models, demonstrating the effectiveness of hidden-space reasoning alignment. 

---
# Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures 

**Authors**: Young Bin Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.17244)  

**Abstract**: While individual components for AI agent memory exist in prior systems, their architectural synthesis and formal grounding remain underexplored. We present Kumiho, a graph-native cognitive memory architecture grounded in formal belief revision semantics. The structural primitives required for cognitive memory -- immutable revisions, mutable tag pointers, typed dependency edges, URI-based addressing -- are identical to those required for managing agent-produced work as versionable assets, enabling a unified graph-native architecture that serves both purposes. The central formal contribution is a correspondence between the AGM belief revision framework and the operational semantics of a property graph memory system, proving satisfaction of the basic AGM postulates (K*2--K*6) and Hansson's belief base postulates (Relevance, Core-Retainment). The architecture implements a dual-store model (Redis working memory, Neo4j long-term graph) with hybrid fulltext and vector retrieval. On LoCoMo (token-level F1), Kumiho achieves 0.565 overall F1 (n=1,986) including 97.5% adversarial refusal accuracy. On LoCoMo-Plus, a Level-2 cognitive memory benchmark testing implicit constraint recall, Kumiho achieves 93.3% judge accuracy (n=401); independent reproduction by the benchmark authors yielded results in the mid-80% range, still substantially outperforming all published baselines (best: Gemini 2.5 Pro, 45.7%). Three architectural innovations drive the results: prospective indexing (LLM-generated future-scenario implications indexed at write time), event extraction (structured causal events preserved in summaries), and client-side LLM reranking. The architecture is model-decoupled: switching the answer model from GPT-4o-mini (~88%) to GPT-4o (93.3%) improves end-to-end accuracy without pipeline changes, at a total evaluation cost of ~$14 for 401 entries. 

---
# Draft-and-Prune: Improving the Reliability of Auto-formalization for Logical Reasoning 

**Authors**: Zhiyu Ni, Zheng Liang, Liangcheng Song, Chenrui Cao, Xian Zhang, Alberto Sangiovanni-Vincentelli, Pierluigi Nuzzo  

**Link**: [PDF](https://arxiv.org/pdf/2603.17233)  

**Abstract**: Auto-formalization (AF) translates natural-language reasoning problems into solver-executable programs, enabling symbolic solvers to perform sound logical deduction. In practice, however, AF pipelines are currently brittle: programs may fail to execute, or execute but encode incorrect semantics. While prior work largely mitigates syntactic failures via repairs based on solver feedback, reducing semantics failures remains a major bottleneck. We propose Draft-and-Prune (D&P), an inference-time framework that improves AF-based logical reasoning via diversity and verification. D&P first drafts multiple natural-language plans and conditions program generation on them. It further prunes executable but contradictory or ambiguous formalizations, and aggregates predictions from surviving paths via majority voting. Across four representative benchmarks (AR-LSAT, ProofWriter, PrOntoQA, LogicalDeduction), D&P substantially strengthens AF-based reasoning without extra supervision. On AR-LSAT, in the AF-only setting, D&P achieves 78.43% accuracy with GPT-4 and 78.00% accuracy with GPT-4o, significantly outperforming the strongest AF baselines MAD-LOGIC and CLOVER. D&P then attains near-ceiling performance on the other benchmarks, including 100% on PrOntoQA and LogicalDeduction. 

---
# AI Scientist via Synthetic Task Scaling 

**Authors**: Ziyang Cai, Harkirat Behl  

**Link**: [PDF](https://arxiv.org/pdf/2603.17216)  

**Abstract**: With the advent of AI agents, automatic scientific discovery has become a tenable goal. Many recent works scaffold agentic systems that can perform machine learning research, but don't offer a principled way to train such agents -- and current LLMs often generate plausible-looking but ineffective ideas. To make progress on training agents that can learn from doing, we provide a novel synthetic environment generation pipeline targeting machine learning agents. Our pipeline automatically synthesizes machine learning challenges compatible with the SWE-agent framework, covering topic sampling, dataset proposal, and code generation. The resulting synthetic tasks are 1) grounded in real machine learning datasets, because the proposed datasets are verified against the Huggingface API and are 2) verified for higher quality with a self-debugging loop. To validate the effectiveness of our synthetic tasks, we tackle MLGym, a benchmark for machine learning tasks. From the synthetic tasks, we sample trajectories from a teacher model (GPT-5), then use the trajectories to train a student model (Qwen3-4B and Qwen3-8B). The student models trained with our synthetic tasks achieve improved performance on MLGym, raising the AUP metric by 9% for Qwen3-4B and 12% for Qwen3-8B. 

---
# How Clued up are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment 

**Authors**: Rebecca Ansell, Autumn Toney-Wails  

**Link**: [PDF](https://arxiv.org/pdf/2603.17169)  

**Abstract**: Deducing whodunit proves challenging for LLM agents. In this paper, we implement a text-based multi-agent version of the classic board game Clue as a rule-based testbed for evaluating multi-step deductive reasoning, with six agents drawn from GPT-4o-mini and Gemini-2.5-Flash. We further investigate whether fine-tuning on structured logic puzzles transfers to improved in-game reasoning and gameplay. Across 18 simulated games, agents achieve only four correct wins, indicating difficulty in maintaining consistent deductive reasoning over the course of a full game. Additionally, we find that fine-tuning does not reliably improve performance and, in some cases, appears to increase reasoning volume without improving reasoning precision. 

---
# Cascade-Aware Multi-Agent Routing: Spatio-Temporal Sidecars and Geometry-Switching 

**Authors**: Davide Di Gioia  

**Link**: [PDF](https://arxiv.org/pdf/2603.17112)  

**Abstract**: A common architectural pattern in advanced AI reasoning systems is the symbolic graph network: specialized agents or modules connected by delegation edges, routing tasks through a dynamic execution graph. Current schedulers optimize load and fitness but are geometry-blind: they do not model how failures propagate differently in tree-like versus cyclic regimes. In tree-like delegation, a single failure can cascade exponentially; in dense cyclic graphs, failures tend to self-limit. We identify this observability gap, quantify its system-level cost, and propose a lightweight mitigation.
We formulate online geometry control for route-risk estimation on time-indexed execution graphs with route-local failure history. Our approach combines (i) a Euclidean spatio-temporal propagation baseline, (ii) a hyperbolic route-risk model with temporal decay (and optional burst excitation), and (iii) a learned geometry selector over structural features. The selector is a compact MLP (9->12->1) using six topology statistics plus three geometry-aware signals: BFS shell-growth slope, cycle-rank norm, and fitted Poincare curvature. On the Genesis 3 benchmark distribution, adaptive switching improves win rate in the hardest non_tree regime from 64-72% (fixed hyperbolic variants) to 92%, and achieves 87.2% overall win rate.
To measure total system value, we compare against Genesis 3 routing without any spatio-temporal sidecar, using only native bandit/LinUCB signals (team fitness and mean node load). This baseline achieves 50.4% win rate overall and 20% in tree-like regimes; the full sidecar recovers 87.2% overall (+36.8 pp), with +48 to +68 pp gains in tree-like settings, consistent with a cascade-sensitivity analysis. Overall, a 133-parameter sidecar substantially mitigates geometry-blind failure propagation in one high-capability execution-graph system. 

---
# Transformers are Bayesian Networks 

**Authors**: Gregory Coppola  

**Link**: [PDF](https://arxiv.org/pdf/2603.17063)  

**Abstract**: Transformers are the dominant architecture in AI, yet why they work remains poorly understood. This paper offers a precise answer: a transformer is a Bayesian network. We establish this in five ways.
First, we prove that every sigmoid transformer with any weights implements weighted loopy belief propagation on its implicit factor graph. One layer is one round of BP. This holds for any weights -- trained, random, or constructed. Formally verified against standard mathematical axioms.
Second, we give a constructive proof that a transformer can implement exact belief propagation on any declared knowledge base. On knowledge bases without circular dependencies this yields provably correct probability estimates at every node. Formally verified against standard mathematical axioms.
Third, we prove uniqueness: a sigmoid transformer that produces exact posteriors necessarily has BP weights. There is no other path through the sigmoid architecture to exact posteriors. Formally verified against standard mathematical axioms.
Fourth, we delineate the AND/OR boolean structure of the transformer layer: attention is AND, the FFN is OR, and their strict alternation is Pearl's gather/update algorithm exactly.
Fifth, we confirm all formal results experimentally, corroborating the Bayesian network characterization in practice. We also establish the practical viability of loopy belief propagation despite the current lack of a theoretical convergence guarantee.
We further prove that verifiable inference requires a finite concept space. Any finite verification procedure can distinguish at most finitely many concepts. Without grounding, correctness is not defined. Hallucination is not a bug that scaling can fix. It is the structural consequence of operating without concepts. Formally verified against standard mathematical axioms. 

---
# Generative AI-assisted Participatory Modeling in Socio-Environmental Planning under Deep Uncertainty 

**Authors**: Zhihao Pei, Nir Lipovetzky, Angela M. Rojas-Arevalo, Fjalar J. de Haan, Enayat A. Moallemi  

**Link**: [PDF](https://arxiv.org/pdf/2603.17021)  

**Abstract**: Socio-environmental planning under deep uncertainty requires researchers to identify and conceptualize problems before exploring policies and deploying plans. In practice and model-based planning approaches, this problem conceptualization process often relies on participatory modeling to translate stakeholders' natural-language descriptions into a quantitative model, making this process complex and time-consuming. To facilitate this process, we propose a templated workflow that uses large language models for an initial conceptualization process. During the workflow, researchers can use large language models to identify the essential model components from stakeholders' intuitive problem descriptions, explore their diverse perspectives approaching the problem, assemble these components into a unified model, and eventually implement the model in Python through iterative communication. These results will facilitate the subsequent socio-environmental planning under deep uncertainty steps. Using ChatGPT 5.2 Instant, we demonstrated this workflow on the lake problem and an electricity market problem, both of which demonstrate socio-environmental planning problems. In both cases, acceptable outputs were obtained after a few iterations with human verification and refinement. These experiments indicated that large language models can serve as an effective tool for facilitating participatory modeling in the problem conceptualization process in socio-environmental planning. 

---
# Unified Spatio-Temporal Token Scoring for Efficient Video VLMs 

**Authors**: Jianrui Zhang, Yue Yang, Rohun Tripathi, Winson Han, Ranjay Krishna, Christopher Clark, Yong Jae Lee, Sangho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.18004)  

**Abstract**: Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning. 

---
# Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models 

**Authors**: Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2603.18002)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: this https URL 

---
# Toward Scalable Automated Repository-Level Datasets for Software Vulnerability Detection 

**Authors**: Amine Lbath  

**Link**: [PDF](https://arxiv.org/pdf/2603.17974)  

**Abstract**: Software vulnerabilities continue to grow in volume and remain difficult to detect in practice. Although learning-based vulnerability detection has progressed, existing benchmarks are largely function-centric and fail to capture realistic, executable, interprocedural settings. Recent repo-level security benchmarks demonstrate the importance of realistic environments, but their manual curation limits scale. This doctoral research proposes an automated benchmark generator that injects realistic vulnerabilities into real-world repositories and synthesizes reproducible proof-of-vulnerability (PoV) exploits, enabling precisely labeled datasets for training and evaluating repo-level vulnerability detection agents. We further investigate an adversarial co-evolution loop between injection and detection agents to improve robustness under realistic constraints. 

---
# TDAD: Test-Driven Agentic Development - Reducing Code Regressions in AI Coding Agents via Graph-Based Impact Analysis 

**Authors**: Pepe Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2603.17973)  

**Abstract**: AI coding agents can resolve real-world software issues, yet they frequently introduce regressions, breaking tests that previously passed. Current benchmarks focus almost exclusively on resolution rate, leaving regression behavior under-studied. This paper presents TDAD (Test-Driven Agentic Development), an open-source tool and benchmark methodology that combines abstract-syntax-tree (AST) based code-test graph construction with weighted impact analysis to surface the tests most likely affected by a proposed change. Evaluated on SWE-bench Verified with two local models (Qwen3-Coder 30B on 100 instances and Qwen3.5-35B-A3B on 25 instances), TDAD's GraphRAG workflow reduced test-level regressions by 70% (6.08% to 1.82%) and improved resolution from 24% to 32% when deployed as an agent skill. A surprising finding is that TDD prompting alone increased regressions (9.94%), revealing that smaller models benefit more from contextual information (which tests to verify) than from procedural instructions (how to do TDD). An autonomous auto-improvement loop raised resolution from 12% to 60% on a 10-instance subset with 0% regression. These findings suggest that for AI agent tool design, surfacing contextual information outperforms prescribing procedural workflows. All code, data, and logs are publicly available at this https URL. 

---
# Specification-Aware Distribution Shaping for Robotics Foundation Models 

**Authors**: Sadık Bera Yüksel, Derya Aksaray  

**Link**: [PDF](https://arxiv.org/pdf/2603.17969)  

**Abstract**: Robotics foundation models have demonstrated strong capabilities in executing natural language instructions across diverse tasks and environments. However, they remain largely data-driven and lack formal guarantees on safety and satisfaction of time-dependent specifications during deployment. In practice, robots often need to comply with operational constraints involving rich spatio-temporal requirements such as time-bounded goal visits, sequential objectives, and persistent safety conditions. In this work, we propose a specification-aware action distribution optimization framework that enforces a broad class of Signal Temporal Logic (STL) constraints during execution of a pretrained robotics foundation model without modifying its parameters. At each decision step, the method computes a minimally modified action distribution that satisfies a hard STL feasibility constraint by reasoning over the remaining horizon using forward dynamics propagation. We validate the proposed framework in simulation using a state-of-the-art robotics foundation model across multiple environments and complex specifications. 

---
# VideoAtlas: Navigating Long-Form Video in Logarithmic Compute 

**Authors**: Mohamed Eltahir, Ali Habibullah, Yazan Alshoibi, Lama Ayash, Tanveer Hussain, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.17948)  

**Abstract**: Extending language models to video introduces two challenges: representation, where existing methods rely on lossy approximations, and long-context, where caption- or agent-based pipelines collapse video into text and lose visual fidelity. To overcome this, we introduce \textbf{VideoAtlas}, a task-agnostic environment to represent video as a hierarchical grid that is simultaneously lossless, navigable, scalable, caption- and preprocessing-free. An overview of the video is available at a glance, and any region can be recursively zoomed into, with the same visual representation used uniformly for the video, intermediate investigations, and the agent's memory, eliminating lossy text conversion end-to-end. This hierarchical structure ensures access depth grows only logarithmically with video length. For long-context, Recursive Language Models (RLMs) recently offered a powerful solution for long text, but extending them to visual domain requires a structured environment to recurse into, which \textbf{VideoAtlas} provides. \textbf{VideoAtlas} as a Markov Decision Process unlocks Video-RLM: a parallel Master-Worker architecture where a Master coordinates global exploration while Workers concurrently drill into assigned regions to accumulate lossless visual evidence. We demonstrate three key findings: (1)~logarithmic compute growth with video duration, further amplified by a 30-60\% multimodal cache hit rate arising from the grid's structural reuse. (2)~environment budgeting, where bounding the maximum exploration depth provides a principled compute-accuracy hyperparameter. (3)~emergent adaptive compute allocation that scales with question granularity. When scaling from 1-hour to 10-hour benchmarks, Video-RLM remains the most duration-robust method with minimal accuracy degradation, demonstrating that structured environment navigation is a viable and scalable paradigm for video understanding. 

---
# CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention 

**Authors**: Zhongzhu Zhou, Fengxiang Bie, Ziyan Chen, Zhenyu Zhang, Yibo Yang, Junxiong Wang, Ben Athiwaratkun, Xiaoxia Wu, Shuaiwen Leon Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.17946)  

**Abstract**: Converting pretrained attention modules such as grouped-query attention (GQA) into multi-head latent attention (MLA) can improve expressivity without increasing KV-cache cost, making it attractive for efficient inference. However, many practical conversion baselines rely on weight-only low-rank approximations (e.g., SVD-style initializations) and uniform rank allocation. They focus on minimizing the difference between weight matrices rather than on how those weights affect input activations, ignore the covariance structure of activations, and enforce uniform rank across layers, causing activation drift and degraded attention fidelity. To address these issues, we propose CARE, a Covariance-Aware, Rank-Enhanced MLA conversion pipeline under a fixed KV width. CARE introduces three key steps: (i) activation-preserving factorization, which aligns the approximation with the actual input activations rather than just the weights; (ii) adjusted-rank allocation, which spreads a fixed KV budget across layers by giving more capacity to layers that need it most; and (iii) KV-parity mapping, which reparameterizes the converted K and V to fit the MLA format while keeping the KV-cache size unchanged. Our method outperforms a uniform-rank SVD baseline on Qwen3-4B/30B-A3B-Instruct-2507 and Llama-3.1-8B/70B-Instruct, reducing one-shot perplexity by up to 215x and improving mean accuracy by up to 1.70x at matched KV budgets. With a brief post-SVD healing fine-tune, we fully recover the original model's accuracy. 

---
# IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia 

**Authors**: Priyaranjan Pattnayak, Sanchari Chowdhuri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17915)  

**Abstract**: As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt.
Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices.
Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms. 

---
# Differential Privacy in Generative AI Agents: Analysis and Optimal Tradeoffs 

**Authors**: Ya-Ting Yang, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17902)  

**Abstract**: Large language models (LLMs) and AI agents are increasingly integrated into enterprise systems to access internal databases and generate context-aware responses. While such integration improves productivity and decision support, the model outputs may inadvertently reveal sensitive information. Although many prior efforts focus on protecting the privacy of user prompts, relatively few studies consider privacy risks from the enterprise data perspective. Hence, this paper develops a probabilistic framework for analyzing privacy leakage in AI agents based on differential privacy. We model response generation as a stochastic mechanism that maps prompts and datasets to distributions over token sequences. Within this framework, we introduce token-level and message-level differential privacy and derive privacy bounds that relate privacy leakage to generation parameters such as temperature and message length. We further formulate a privacy-utility design problem that characterizes optimal temperature selection. 

---
# scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns 

**Authors**: Sergey V. Samsonau  

**Link**: [PDF](https://arxiv.org/pdf/2603.17893)  

**Abstract**: Methodology bugs in scientific Python code produce plausible but incorrect results that traditional linters and static analysis tools cannot detect. Several research groups have built ML-specific linters, demonstrating that detection is feasible. Yet these tools share a sustainability problem: dependency on specific pylint or Python versions, limited packaging, and reliance on manual engineering for every new pattern. As AI-generated code increases the volume of scientific software, the need for automated methodology checking (such as detecting data leakage, incorrect cross-validation, and missing random seeds) grows. We present scicode-lint, whose two-tier architecture separates pattern design (frontier models at build time) from execution (small local model at runtime). Patterns are generated, not hand-coded; adapting to new library versions costs tokens, not engineering hours. On Kaggle notebooks with human-labeled ground truth, preprocessing leakage detection reaches 65% precision at 100% recall; on 38 published scientific papers applying AI/ML, precision is 62% (LLM-judged) with substantial variation across pattern categories; on a held-out paper set, precision is 54%. On controlled tests, scicode-lint achieves 97.7% accuracy across 66 patterns. 

---
# RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference 

**Authors**: Arpit Singh Gautam, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2603.17891)  

**Abstract**: Post training quantization is essential for deploying large language models (LLMs) on resource constrained hardware, yet state of the art methods enforce uniform bit widths across layers, yielding suboptimal accuracy efficiency trade offs. We present RAMP (Reinforcement Adaptive Mixed Precision), an off policy Soft Actor Critic framework that learns per layer bit width assignments to minimize perplexity under a global bit budget. The policy conditions on an 11 dimensional embedding of activation statistics, weight properties, and structural descriptors, enabling zero shot transfer across model families and scales. To enable stable sub 4 bit quantization, we introduce Scale Folding, a preconditioning technique that migrates activation outliers into weights via per channel scaling and normalization layer compensation. A quality prioritized reward with asymmetric penalties and budget cliffs drives rapid convergence. On Llama 2 7B, RAMP achieves 5.54 perplexity at 3.68GB (3.65 effective bits), outperforming uniform 4 bit AWQ (5.60 at 3.90 GB) and GPTQ by 6% in size and 1% to3% in quality. Critically, a policy trained only on Llama 2 7B generalizes zero shot to Llama 2 13B and Mistral 7B, often surpassing target specific training, supporting the hypothesis that quantization sensitivity is primarily architectural. The HALO pipeline exports allocations to GGUF format for kernel free inference on CPUs, GPUs, and edge devices, retaining 99.5% of FP16 commonsense reasoning performance. 

---
# AI-Assisted Goal Setting Improves Goal Progress Through Social Accountability 

**Authors**: Michel Schimpf, Julian Voigt, Thomas Bohné  

**Link**: [PDF](https://arxiv.org/pdf/2603.17887)  

**Abstract**: Helping people identify and pursue personally meaningful career goals at scale remains a key challenge in applied psychology. Career coaching can improve goal quality and attainment, but its cost and limited availability restrict access. Large language model (LLM)-based chatbots offer a scalable alternative, yet the psychological mechanisms by which they might support goal pursuit remain untested. Here we report a preregistered three-arm randomised controlled trial (N = 517) comparing an AI career coach ("Leon," powered by Claude Sonnet), a matched structured written questionnaire covering closely matched reflective topics, and a no-support control on goal progress at a two-week follow-up. The AI chatbot produced significantly higher goal progress than the control (d = 0.33, p = .016). Compared with the written-reflection condition, the AI did not significantly improve overall goal progress, but it increased perceived social accountability. In the preregistered mediation model, perceived accountability mediated the AI-over-questionnaire effect on goal progress (indirect effect = 0.15, 95% CI [0.04, 0.31]), whereas self-concordance did not. These findings suggest that AI-assisted goal setting can improve short-term goal progress, and that its clearest added value over structured self-reflection lies in increasing felt accountability. 

---
# Differential Attention-Augmented BiomedCLIP with Asymmetric Focal Optimization for Imbalanced Multi-Label Video Capsule Endoscopy Classification 

**Authors**: Podakanti Satyajith Chary, Nagarajan Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2603.17879)  

**Abstract**: This work presents a multi-label classification framework for video capsule endoscopy (VCE) that addresses the extreme class imbalance inherent in the Galar dataset through a combination of architectural and optimization-level strategies. Our approach modifies BiomedCLIP, a biomedical vision-language foundation model, by replacing its standard multi-head self-attention with a differential attention mechanism that computes the difference between two softmax attention maps to suppress attention noise. To counteract the skewed label distribution, where pathological findings constitute less than 0.1% of all annotated frames, a sqrt-frequency weighted sampler, asymmetric focal loss, mixup regularization, and per-class threshold optimization are employed. Temporal coherence is enforced through median-filter smoothing and gap merging prior to event-level JSON generation. On the held-out RARE-VISION test set comprising three NaviCam examinations (161,025 frames), the pipeline achieves an overall temporal mAP@0.5 of 0.2456 and mAP@0.95 of 0.2353, with total inference completed in approximately 8.6 minutes on a single GPU. 

---
# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval 

**Authors**: Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2603.17872)  

**Abstract**: Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Corrective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI. 

---
# Procedural Generation of Algorithm Discovery Tasks in Machine Learning 

**Authors**: Alexander D. Goldie, Zilin Wang, Adrian Hayler, Deepak Nathani, Edan Toledo, Ken Thampiratwong, Aleksandra Kalisz, Michael Beukman, Alistair Letcher, Shashank Reddy, Clarisse Wibault, Theo Wolf, Charles O'Neill, Uljad Berdica, Nicholas Roberts, Saeed Rahmani, Hannah Erlebach, Roberta Raileanu, Shimon Whiteson, Jakob N. Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2603.17863)  

**Abstract**: Automating the development of machine learning algorithms has the potential to unlock new breakthroughs. However, our ability to improve and evaluate algorithm discovery systems has thus far been limited by existing task suites. They suffer from many issues, such as: poor evaluation methodologies; data contamination; and containing saturated or very similar problems. Here, we introduce DiscoGen, a procedural generator of algorithm discovery tasks for machine learning, such as developing optimisers for reinforcement learning or loss functions for image classification. Motivated by the success of procedural generation in reinforcement learning, DiscoGen spans millions of tasks of varying difficulty and complexity from a range of machine learning fields. These tasks are specified by a small number of configuration parameters and can be used to optimise algorithm discovery agents (ADAs). We present DiscoBench, a benchmark consisting of a fixed, small subset of DiscoGen tasks for principled evaluation of ADAs. Finally, we propose a number of ambitious, impactful research directions enabled by DiscoGen, in addition to experiments demonstrating its use for prompt optimisation of an ADA. DiscoGen is released open-source at this https URL. 

---
# How do LLMs Compute Verbal Confidence 

**Authors**: Dharshan Kumaran, Arthur Conmy, Federico Barbero, Simon Osindero, Viorica Patraucean, Petar Velickovic  

**Link**: [PDF](https://arxiv.org/pdf/2603.17839)  

**Abstract**: Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed - just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents - token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B and Qwen 2.5 7B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration. 

---
# Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control 

**Authors**: Zunzhe Zhang, Runhan Huang, Yicheng Liu, Shaoting Zhu, Linzhan Mou, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17834)  

**Abstract**: Diffusion models and flow matching have become a cornerstone of robotic imitation learning, yet they suffer from a structural inefficiency where inference is often bound to a fixed integration schedule that is agnostic to state complexity. This paradigm forces the policy to expend the same computational budget on trivial motions as it does on complex tasks. We introduce Generative Control as Optimization (GeCO), a time-unconditional framework that transforms action synthesis from trajectory integration into iterative optimization. GeCO learns a stationary velocity field in the action-sequence space where expert behaviors form stable attractors. Consequently, test-time inference becomes an adaptive process that allocates computation based on convergence--exiting early for simple states while refining longer for difficult ones. Furthermore, this stationary geometry yields an intrinsic, training-free safety signal, as the field norm at the optimized action serves as a robust out-of-distribution (OOD) detector, remaining low for in-distribution states while significantly increasing for anomalies. We validate GeCO on standard simulation benchmarks and demonstrate seamless scaling to pi0-series Vision-Language-Action (VLA) models. As a plug-and-play replacement for standard flow-matching heads, GeCO improves success rates and efficiency with an optimization-native mechanism for safe deployment. Video and code can be found at this https URL 

---
# Text-to-Stage: Spatial Layouts from Long-form Narratives 

**Authors**: Jefferson Hernandez, Swarnadeep Saha, Chenxi Whitehouse, Sanjeel Parekh, Calvin Murdock, Yuliang Li, W. Owen Brimijoin, Vamsi Krishna Ithapu, Ishwarya Ananthabhotla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17832)  

**Abstract**: In this work, we probe the ability of a language model to demonstrate spatial reasoning from unstructured text, mimicking human capabilities and automating a process that benefits many downstream media applications. Concretely, we study the narrative-to-play task: inferring stage-play layouts (scenes, speaker positions, movements, and room types) from text that lacks explicit spatial, positional, or relational cues. We then introduce a dramaturgy-inspired deterministic evaluation suite and, finally, a training and inference recipe that combines rejection SFT using Best-of-N sampling with RL from verifiable rewards via GRPO. Experiments on a text-only corpus of classical English literature demonstrate improvements over vanilla models across multiple metrics (character attribution, spatial plausibility, and movement economy), as well as alignment with an LLM-as-a-judge and subjective human preferences. 

---
# CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents 

**Authors**: Lintang Sutawika, Aditya Bharat Soni, Bharath Sriraam R R, Apurva Gandhi, Taha Yassine, Sanidhya Vijayvargiya, Yuchen Li, Xuhui Zhou, Yilin Zhang, Leander Melroy Maben, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2603.17829)  

**Abstract**: A prerequisite for coding agents to perform tasks on large repositories is code localization - the identification of relevant files, classes, and functions to work on. While repository-level code localization has been performed using embedding-based retrieval approaches such as vector search, recent work has focused on developing agents to localize relevant code either as a standalone precursor to or interleaved with performing actual work. Most prior methods on agentic code search equip the agent with complex, specialized tools, such as repository graphs derived from static analysis. In this paper, we demonstrate that, with an effective reinforcement learning recipe, a coding agent equipped with nothing more than a standard Unix terminal can be trained to achieve strong results. Our experiments on three benchmarks (SWE-Bench Verified, Pro, and Lite) reveal that our models consistently achieve superior or competitive performance over 2-18x larger base and post-trained LLMs and sometimes approach performance provided by closed models like Claude Sonnet, even when using specialized scaffolds. Our work particularly focuses on techniques for re-purposing existing coding agent environments for code search, reward design, and RL optimization. We release the resulting model family, CodeScout, along with all our code and data for the community to build upon. 

---
# FailureMem: A Failure-Aware Multimodal Framework for Autonomous Software Repair 

**Authors**: Ruize Ma, Yilei Jiang, Shilin Zhang, Zheng Ma, Yi Feng, Vincent Ng, Zhi Wang, Xiangyu Yue, Chuanyi Li, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17826)  

**Abstract**: Multimodal Automated Program Repair (MAPR) extends traditional program repair by requiring models to jointly reason over source code, textual issue descriptions, and visual artifacts such as GUI screenshots. While recent LLM-based repair systems have shown promising results, existing approaches face several limitations: rigid workflow pipelines restrict exploration during debugging, visual reasoning is often performed over full-page screenshots without localized grounding, and failed repair attempts are rarely transformed into reusable knowledge. To address these challenges, we propose FailureMem, a multimodal repair framework that integrates three key mechanisms: a hybrid workflow-agent architecture that balances structured localization with flexible reasoning, active perception tools that enable region-level visual grounding, and a Failure Memory Bank that converts past repair attempts into reusable guidance. Experiments on SWE-bench Multimodal demonstrate FailureMem improves the resolved rate over GUIRepair by 3.7%. 

---
# ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation 

**Authors**: Dmitriy Rivkin, Parker Ewen, Lili Gao, Julian Ost, Stefanie Walz, Rasika Kangutkar, Mario Bijelic, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2603.17812)  

**Abstract**: Recent video diffusion models achieve high-quality generation through recurrent frame processing where each frame generation depends on previous frames. However, this recurrent mechanism means that training such models in the pixel domain incurs prohibitive memory costs, as activations accumulate across the entire video sequence. This fundamental limitation also makes fine-tuning these models with pixel-wise losses computationally intractable for long or high-resolution videos. This paper introduces ChopGrad, a truncated backpropagation scheme for video decoding, limiting gradient computation to local frame windows while maintaining global consistency. We provide a theoretical analysis of this approximation and show that it enables efficient fine-tuning with frame-wise losses. ChopGrad reduces training memory from scaling linearly with the number of video frames (full backpropagation) to constant memory, and compares favorably to existing state-of-the-art video diffusion models across a suite of conditional video generation tasks with pixel-wise losses, including video super-resolution, video inpainting, video enhancement of neural-rendered scenes, and controlled driving video generation. 

---
# Dropout Robustness and Cognitive Profiling of Transformer Models via Stochastic Inference 

**Authors**: Antônio Junior Alves Caiado, Michael Hahsler  

**Link**: [PDF](https://arxiv.org/pdf/2603.17811)  

**Abstract**: Transformer-based language models are widely deployed for reasoning, yet their behavior under inference-time stochasticity remains underexplored. While dropout is common during training, its inference-time effects via Monte Carlo sampling lack systematic evaluation across architectures, limiting understanding of model reliability in uncertainty-aware applications.
This work analyzes dropout-induced variability across 19 transformer models using MC Dropout with 100 stochastic forward passes per sample. Dropout robustness is defined as maintaining high accuracy and stable predictions under stochastic inference, measured by standard deviation of per-run accuracies. A cognitive decomposition framework disentangles performance into memory and reasoning components. Experiments span five dropout configurations yielding 95 unique evaluations on 1,000 samples.
Results reveal substantial architectural variation. Smaller models demonstrate perfect prediction stability while medium-sized models exhibit notable volatility. Mid-sized models achieve the best overall performance; larger models excel at memory tasks. Critically, 53% of models suffer severe accuracy degradation under baseline MC Dropout, with task-specialized models losing up to 24 percentage points, indicating unsuitability for uncertainty quantification in these architectures. Asymmetric effects emerge: high dropout reduces memory accuracy by 27 percentage points while reasoning degrades only 1 point, suggesting memory tasks rely on stable representations that dropout disrupts. 84% of models demonstrate memory-biased performance.
This provides the first comprehensive MC Dropout benchmark for transformers, revealing dropout robustness is architecture-dependent and uncorrelated with scale. The cognitive profiling framework offers actionable guidance for model selection in uncertainty-aware applications. 

---
# Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients 

**Authors**: Ziwei Xiang, Fanhu Zeng, Hongjian Fang, Rui-Qi Wang, Renxing Chen, Yanan Zhu, Yi Chen, Peipei Yang, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17809)  

**Abstract**: Large Vision Language Models (LVLMs) have achieved remarkable success in a range of downstream tasks that require multimodal interaction, but their capabilities come with substantial computational and memory overhead, which hinders practical deployment. Among numerous acceleration techniques, post-training quantization is a popular and effective strategy for reducing memory cost and accelerating inference. However, existing LVLM quantization methods typically measure token sensitivity at the modality level, which fails to capture the complex cross-token interactions and falls short in quantitatively measuring the quantization error at the token level. As tokens interact within the model, the distinction between modalities gradually diminishes, suggesting the need for fine-grained calibration. Inspired by axiomatic attribution in mechanistic interpretability, we introduce a fine-grained quantization strategy on Quantization-aware Integrated Gradients (QIG), which leverages integrated gradients to quantitatively evaluate token sensitivity and push the granularity from modality level to token level, reflecting both inter-modality and intra-modality dynamics. Extensive experiments on multiple LVLMs under both W4A8 and W3A16 settings show that our method improves accuracy across models and benchmarks with negligible latency overhead. For example, under 3-bit weight-only quantization, our method improves the average accuracy of LLaVA-onevision-7B by 1.60%, reducing the gap to its full-precision counterpart to only 1.33%. The code is available at this https URL. 

---
# EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards 

**Authors**: Ruixiang Wang, Qingming Liu, Yueci Deng, Guiliang Liu, Zhen Liu, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2603.17808)  

**Abstract**: Video generative models are increasingly used as world models for robotics, where a model generates a future visual rollout conditioned on the current observation and task instruction, and an inverse dynamics model (IDM) converts the generated frames into executable robot actions. However, current video world models lack explicit executability constraints. As a result, visually coherent rollouts may still violate rigid-body and kinematic consistency, producing unstable or infeasible control commands when decoded by an IDM. We refer to this mismatch between visual generation and physically executable control as the executability gap. While this gap can be mitigated at inference time using techniques such as rejection sampling, such approaches are inefficient due to the high cost of video generation. In this paper, we leverage the executability gap as a training signal and introduce Executable Video Alignment (EVA), a reinforcement-learning post-training framework for aligning video world models. EVA trains an inverse dynamics model on real robot trajectories and repurposes it as a reward model that evaluates generated videos through the action sequences they induce, encouraging smooth motions measured by velocity, acceleration, and jerk while penalizing actions that violate embodiment constraints. Importantly, the reward remains informative even when generated videos contain severe visual artifacts, since such artifacts typically translate into unstable or out-of-bound actions. Experiments on the RoboTwin benchmark and a real bimanual robot show that EVA reduces embodiment-specific artifacts in generated rollouts and improves downstream task execution success. 

---
# RangeAD: Fast On-Model Anomaly Detection 

**Authors**: Luca Hinkamp, Simon Klüttermann, Emmanuel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2603.17795)  

**Abstract**: In practice, machine learning methods commonly require anomaly detection (AD) to filter inputs or detect distributional shifts. Typically, this is implemented by running a separate AD model alongside the primary model. However, this separation ignores the fact that the primary model already encodes substantial information about the target distribution. In this paper, we introduce On-Model AD, a setting for anomaly detection that explicitly leverages access to a related machine learning model. Within this setting, we propose RangeAD, an algorithm that utilizes neuron-wise output ranges derived from the primary model. RangeAD achieves superior performance even on high-dimensional tasks while incurring substantially lower inference costs. Our results demonstrate the potential of the On-Model AD setting as a practical framework for efficient anomaly detection. 

---
# A Dual Certificate Approach to Sparsity in Infinite-Width Shallow Neural Networks 

**Authors**: Leonardo Del Grande, Christoph Brune, Marcello Carioni  

**Link**: [PDF](https://arxiv.org/pdf/2603.17785)  

**Abstract**: In this paper, we study total variation (TV)-regularized training of infinite-width shallow ReLU neural networks, formulated as a convex optimization problem over measures on the unit sphere. Our approach leverages the duality theory of TV-regularized optimization problems to establish rigorous guarantees on the sparsity of the solutions to the training problem. Our analysis further characterizes how and when this sparsity persists in a low noise regime and for small regularization parameter. The key observation that motivates our analysis is that, for ReLU activations, the associated dual certificate is piecewise linear in the weight space. Its linearity regions, which we name dual regions, are determined by the activation patterns of the data via the induced hyperplane arrangement. Taking advantage of this structure, we prove that, on each dual region, the dual certificate admits at most one extreme value. As a consequence, the support of any minimizer is finite, and its cardinality can be bounded from above by a constant depending only on the geometry of the data-induced hyperplane arrangement. Then, we further investigate sufficient conditions ensuring uniqueness of such sparse solution. Finally, under a suitable non-degeneracy condition on the dual certificate along the boundaries of the dual regions, we prove that in the presence of low label noise and for small regularization parameter, solutions to the training problem remain sparse with the same number of Dirac deltas. Additionally, their location and the amplitudes converge, and, in case the locations lie in the interior of a dual region, the convergence happens with a rate that depends linearly on the noise and the regularization parameter. 

---
# CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution 

**Authors**: Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Gaiyang Han, Wanqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17775)  

**Abstract**: Label-free reinforcement learning enables large language models to improve reasoning capabilities without ground-truth supervision, typically by treating majority-voted answers as pseudo-labels. However, we identify a critical failure mode: as training maximizes self-consistency, output diversity collapses, causing the model to confidently reinforce systematic errors that evade detection. We term this the consensus trap. To escape it, we propose CoVerRL, a framework where a single model alternates between generator and verifier roles, with each capability bootstrapping the other. Majority voting provides noisy but informative supervision for training the verifier, while the improving verifier progressively filters self-consistent errors from pseudo-labels. This co-evolution creates a virtuous cycle that maintains high reward accuracy throughout training. Experiments across Qwen and Llama model families demonstrate that CoVerRL outperforms label-free baselines by 4.7-5.9\% on mathematical reasoning benchmarks. Moreover, self-verification accuracy improves from around 55\% to over 85\%, confirming that both capabilities genuinely co-evolve. 

---
# Attention Sinks Induce Gradient Sinks 

**Authors**: Yihong Chen, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17771)  

**Abstract**: Attention sinks and massive activations are recurring and closely related phenomena in Transformer models. Existing studies have largely focused on the forward pass, making it unclear whether their connection is direct or mediated by a training-time mechanism. We study this question from the perspective of backpropagation. Empirically and theoretically, we show that under causal mask, attention sinks can induce pronounced gradient concentration, which we term gradient sinks. Furthermore, in pre-norm architectures with RMSNorm, massive activations can be understood as an adaptive response to this localized gradient pressure during training. To test this hypothesis, we introduce V-scale, a modification that adjusts value-path backpropagated gradients. In pretrained V-scale models, attention sinks are preserved whereas massive activations are suppressed. These results support the interpretation that gradient sink is a key training-time mediator linking attention sinks and massive activations. 

---
# Harm or Humor: A Multimodal, Multilingual Benchmark for Overt and Covert Harmful Humor 

**Authors**: Ahmed Sharshar, Hosam Elgendy, Saad El Dine Ahmed, Yasser Rohaim, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17759)  

**Abstract**: Dark humor often relies on subtle cultural nuances and implicit cues that require contextual reasoning to interpret, posing safety challenges that current static benchmarks fail to capture. To address this, we introduce a novel multimodal, multilingual benchmark for detecting and understanding harmful and offensive humor. Our manually curated dataset comprises 3,000 texts and 6,000 images in English and Arabic, alongside 1,200 videos that span English, Arabic, and language-independent (universal) contexts. Unlike standard toxicity datasets, we enforce a strict annotation guideline: distinguishing \emph{Safe} jokes from \emph{Harmful} ones, with the latter further classified into \emph{Explicit} (overt) and \emph{Implicit} (Covert) categories to probe deep reasoning. We systematically evaluate state-of-the-art (SOTA) open and closed-source models across all modalities. Our findings reveal that closed-source models significantly outperform open-source ones, with a notable difference in performance between the English and Arabic languages in both, underscoring the critical need for culturally grounded, reasoning-aware safety alignment. \textcolor{red}{Warning: this paper contains example data that may be offensive, harmful, or biased.} 

---
# SARE: Sample-wise Adaptive Reasoning for Training-free Fine-grained Visual Recognition 

**Authors**: Jingxiao Yang, DaLin He, Miao Pan, Ge Su, Wenqi Zhang, Yifeng Hu, Tangwei Li, Yuke Li, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17729)  

**Abstract**: Recent advances in Large Vision-Language Models (LVLMs) have enabled training-free Fine-Grained Visual Recognition (FGVR). However, effectively exploiting LVLMs for FGVR remains challenging due to the inherent visual ambiguity of subordinate-level categories. Existing methods predominantly adopt either retrieval-oriented or reasoning-oriented paradigms to tackle this challenge, but both are constrained by two fundamental limitations:(1) They apply the same inference pipeline to all samples without accounting for uneven recognition difficulty, thereby leading to suboptimal accuracy and efficiency; (2) The lack of mechanisms to consolidate and reuse error-specific experience causes repeated failures on similar challenging cases. To address these limitations, we propose SARE, a Sample-wise Adaptive textbfREasoning framework for training-free FGVR. Specifically, SARE adopts a cascaded design that combines fast candidate retrieval with fine-grained reasoning, invoking the latter only when necessary. In the reasoning process, SARE incorporates a self-reflective experience mechanism that leverages past failures to provide transferable discriminative guidance during inference, without any parameter updates. Extensive experiments across 14 datasets substantiate that SARE achieves state-of-the-art performance while substantially reducing computational overhead. 

---
# Machine Learning for Network Attacks Classification and Statistical Evaluation of Machine Learning for Network Attacks Classification and Adversarial Learning Methodologies for Synthetic Data Generation 

**Authors**: Iakovos-Christos Zarkadis, Christos Douligeris  

**Link**: [PDF](https://arxiv.org/pdf/2603.17717)  

**Abstract**: Supervised detection of network attacks has always been a critical part of network intrusion detection systems (NIDS). Nowadays, in a pivotal time for artificial intelligence (AI), with even more sophisticated attacks that utilize advanced techniques, such as generative artificial intelligence (GenAI) and reinforcement learning, it has become a vital component if we wish to protect our personal data, which are scattered across the web. In this paper, we address two tasks, in the first unified multi-modal NIDS dataset, which incorporates flow-level data, packet payload information and temporal contextual features, from the reprocessed CIC-IDS-2017, CIC-IoT-2023, UNSW-NB15 and CIC-DDoS-2019, with the same feature space. In the first task we use machine learning (ML) algorithms, with stratified cross validation, in order to prevent network attacks, with stability and reliability. In the second task we use adversarial learning algorithms to generate synthetic data, compare them with the real ones and evaluate their fidelity, utility and privacy using the SDV framework, f-divergences, distinguishability and non-parametric statistical tests. The findings provide stable ML models for intrusion detection and generative models with high fidelity and utility, by combining the Synthetic Data Vault framework, the TRTS and TSTR tests, with non-parametric statistical tests and f-divergence measures. 

---
# Eye image segmentation using visual and concept prompts with Segment Anything Model 3 (SAM3) 

**Authors**: Diederick C. Niehorster, Marcus Nyström  

**Link**: [PDF](https://arxiv.org/pdf/2603.17715)  

**Abstract**: Previous work has reported that vision foundation models show promising zero-shot performance in eye image segmentation. Here we examine whether the latest iteration of the Segment Anything Model, SAM3, offers better eye image segmentation performance than SAM2, and explore the performance of its new concept (text) prompting mode. Eye image segmentation performance was evaluated using diverse datasets encompassing both high-resolution high-quality videos from a lab environment and the TEyeD dataset consisting of challenging eye videos acquired in the wild. Results show that in most cases SAM3 with either visual or concept prompts did not perform better than SAM2, for both lab and in-the-wild datasets. Since SAM2 not only performed better but was also faster, we conclude that SAM2 remains the best option for eye image segmentation. We provide our adaptation of SAM3's codebase that allows processing videos of arbitrary duration. 

---
# Can Blindfolded LLMs Still Trade? An Anonymization-First Framework for Portfolio Optimization 

**Authors**: Joohyoung Jeon, Hongchul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.17692)  

**Abstract**: For LLM trading agents to be genuinely trustworthy, they must demonstrate understanding of market dynamics rather than exploitation of memorized ticker associations. Building responsible multi-agent systems demands rigorous signal validation: proving that predictions reflect legitimate patterns, not pre-trained recall. We address two sources of spurious performance: memorization bias from ticker-specific pre-training, and survivorship bias from flawed backtesting. Our approach is to blindfold the agents--anonymizing all identifiers--and verify whether meaningful signals persist. BlindTrade anonymizes tickers and company names, and four LLM agents output scores along with reasoning. We construct a GNN graph from reasoning embeddings and trade using PPO-DSR policy. On 2025 YTD (through 2025-08-01), we achieved Sharpe 1.40 +/- 0.22 across 20 seeds and validated signal legitimacy through negative control experiments. To assess robustness beyond a single OOS window, we additionally evaluate an extended period (2024--2025), revealing market-regime dependency: the policy excels in volatile conditions but shows reduced alpha in trending bull markets. 

---
# Objective Mispricing Detection for Shortlisting Undervalued Football Players via Market Dynamics and News Signals 

**Authors**: Chinenye Omejieke, Shuyao Chen, Xia Cui  

**Link**: [PDF](https://arxiv.org/pdf/2603.17687)  

**Abstract**: We present a practical, reproducible framework for identifying undervalued football players grounded in objective mispricing. Instead of relying on subjective expert labels, we estimate an expected market value from structured data (historical market dynamics, biographical and contract features, transfer history) and compare it to the observed valuation to define mispricing. We then assess whether news-derived Natural Language Processing (NLP) features (i.e., sentiment statistics and semantic embeddings from football articles) complement market signals for shortlisting undervalued players.
Using a chronological (leakage-aware) evaluation, gradient-boosted regression explains a large share of the variance in log-transformed market value. For undervaluation shortlisting, ROC-AUC-based ablations show that market dynamics are the primary signal, while NLP features provide consistent, secondary gains that improve robustness and interpretability. SHAP analyses suggest the dominance of market trends and age, with news-derived volatility cues amplifying signals in high-uncertainty regimes. The proposed pipeline is designed for decision support in scouting workflows, emphasizing ranking/shortlisting over hard classification thresholds, and includes a concise reproducibility and ethics statement. 

---
# WeatherReasonSeg: A Benchmark for Weather-Aware Reasoning Segmentation in Visual Language Models 

**Authors**: Wanjun Du, Zifeng Yuan, Tingting Chen, Fucai Ke, Beibei Lin, Shunli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17680)  

**Abstract**: Existing vision-language models (VLMs) have demonstrated impressive performance in reasoning-based segmentation. However, current benchmarks are primarily constructed from high-quality images captured under idealized conditions. This raises a critical question: when visual cues are severely degraded by adverse weather conditions such as rain, snow, or fog, can VLMs sustain reliable reasoning segmentation capabilities? In response to this challenge, we introduce WeatherReasonSeg, a benchmark designed to evaluate VLM performance in reasoning-based segmentation under adverse weather conditions. It consists of two complementary components. First, we construct a controllable reasoning dataset by applying synthetic weather with varying severity levels to existing segmentation datasets, enabling fine-grained robustness analysis. Second, to capture real-world complexity, we curate a real-world adverse-weather reasoning segmentation dataset with semantically consistent queries generated via mask-guided LLM prompting. We further broaden the evaluation scope across five reasoning dimensions, including functionality, application scenarios, structural attributes, interactions, and requirement matching. Extensive experiments across diverse VLMs reveal two key findings: (1) VLM performance degrades monotonically with increasing weather severity, and (2) different weather types induce distinct vulnerability patterns. We hope WeatherReasonSeg will serve as a foundation for advancing robust, weather-aware reasoning. 

---
# Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models 

**Authors**: Jaemin Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.17677)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factual grounding by incorporating external knowledge into language model generation. However, when retrieved context is noisy, unreliable, or inconsistent with the model's parametric knowledge, it introduces retrieval-prior conflicts that can degrade generation quality. While this problem has been studied in autoregressive language models, it remains largely unexplored in diffusion-based language models, where the iterative denoising process introduces unique challenges for integrating retrieved context. In this work, we propose Adaptive Retrieval-Augmented Masked Diffusion (ARAM), a training-free adaptive guidance framework for Masked Diffusion Models (MDMs) in RAG settings. ARAM dynamically calibrates the guidance scale during denoising according to the Signal-to-Noise Ratio (SNR) of the distributional shift induced by retrieved context. Intuitively, the model strengthens guidance when the retrieved context provides reliable corrective evidence and suppresses it when the contextual signal is noisy or non-supportive. Extensive experiments on multiple knowledge-intensive QA benchmarks show that ARAM improves overall QA performance over competitive RAG baselines. 

---
# Inhibitory normalization of error signals improves learning in neural circuits 

**Authors**: Roy Henha Eyono, Daniel Levenstein, Arna Ghosh, Jonathan Cornford, Blake Richards  

**Link**: [PDF](https://arxiv.org/pdf/2603.17676)  

**Abstract**: Normalization is a critical operation in neural circuits. In the brain, there is evidence that normalization is implemented via inhibitory interneurons and allows neural populations to adjust to changes in the distribution of their inputs. In artificial neural networks (ANNs), normalization is used to improve learning in tasks that involve complex input distributions. However, it is unclear whether inhibition-mediated normalization in biological neural circuits also improves learning. Here, we explore this possibility using ANNs with separate excitatory and inhibitory populations trained on an image recognition task with variable luminosity. We find that inhibition-mediated normalization does not improve learning if normalization is applied only during inference. However, when this normalization is extended to include back-propagated errors, performance improves significantly. These results suggest that if inhibition-mediated normalization improves learning in the brain, it additionally requires the normalization of learning signals. 

---
# Post-Training Local LLM Agents for Linux Privilege Escalation with Verifiable Rewards 

**Authors**: Philipp Normann, Andreas Happe, Jürgen Cito, Daniel Arp  

**Link**: [PDF](https://arxiv.org/pdf/2603.17673)  

**Abstract**: LLM agents are increasingly relevant to research domains such as vulnerability discovery. Yet, the strongest systems remain closed and cloud-only, making them resource-intensive, difficult to reproduce, and unsuitable for work involving proprietary code or sensitive data. Consequently, there is an urgent need for small, local models that can perform security tasks under strict resource budgets, but methods for developing them remain underexplored. In this paper, we address this gap by proposing a two-stage post-training pipeline. We focus on the problem of Linux privilege escalation, where success is automatically verifiable and the task requires multi-step interactive reasoning. Using an experimental setup that prevents data leakage, we post-train a 4B model in two stages: supervised fine-tuning on traces from procedurally generated privilege-escalation environments, followed by reinforcement learning with verifiable rewards. On a held-out benchmark of 12 Linux privilege-escalation scenarios, supervised fine-tuning alone more than doubles the baseline success rate at 20 rounds, and reinforcement learning further lifts our resulting model, PrivEsc-LLM, to 95.8%, nearly matching Claude Opus 4.6 at 97.5%. At the same time, the expected inference cost per successful escalation is reduced by over 100x. 

---
# FINER: MLLMs Hallucinate under Fine-grained Negative Queries 

**Authors**: Rui Xiao, Sanghwan Kim, Yongqin Xian, Zeynep Akata, Stephan Alaniz  

**Link**: [PDF](https://arxiv.org/pdf/2603.17662)  

**Abstract**: Multimodal large language models (MLLMs) struggle with hallucinations, particularly with fine-grained queries, a challenge underrepresented by existing benchmarks that focus on coarse image-related questions. We introduce FIne-grained NEgative queRies (FINER), alongside two benchmarks: FINER-CompreCap and FINER-DOCCI. Using FINER, we analyze hallucinations across four settings: multi-object, multi-attribute, multi-relation, and ``what'' questions. Our benchmarks reveal that MLLMs hallucinate when fine-grained mismatches co-occur with genuinely present elements in the image. To address this, we propose FINER-Tuning, leveraging Direct Preference Optimization (DPO) on FINER-inspired data. Finetuning four frontier MLLMs with FINER-Tuning yields up to 24.2\% gains (InternVL3.5-14B) on hallucinations from our benchmarks, while simultaneously improving performance on eight existing hallucination suites, and enhancing general multimodal capabilities across six benchmarks. Code, benchmark, and models are available at \href{this https URL}{this https URL}. 

---
# Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment 

**Authors**: Yaze Zhao, Yixiong Zou, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.17655)  

**Abstract**: Cross-Domain Few-Shot Learning (CDFSL) adapts models trained with large-scale general data (source domain) to downstream target domains with only scarce training data, where the research on vision-language models (e.g., CLIP) is still in the early stages. Typical downstream domains, such as medical diagnosis, require fine-grained visual cues for interpretable recognition, but we find that current fine-tuned CLIP models can hardly focus on these cues, albeit they can roughly focus on important regions in source domains. Although current works have demonstrated CLIP's shortcomings in capturing local subtle patterns, in this paper, we find that the domain gap and scarce training data further exacerbate such shortcomings, much more than that of holistic patterns, which we call the local misalignment problem in CLIP-based CDFSL. To address this problem, due to the lack of supervision in aligning local visual features and text semantics, we turn to self-supervision information. Inspired by the translation task, we propose the CC-CDFSL method with cycle consistency, which translates local visual features into text features and then translates them back into visual features (and vice versa), and constrains the original features close to the translated back features. To reduce the noise imported by richer information in the visual modality, we further propose a Semantic Anchor mechanism, which first augments visual features to provide a larger corpus for the text-to-image mapping, and then shrinks the image features to filter out irrelevant image-to-text mapping. Extensive experiments on various benchmarks, backbones, and fine-tuning methods show we can (1) effectively improve the local vision-language alignment, (2) enhance the interpretability of learned patterns and model decisions by visualizing patches, and (3) achieve state-of-the-art performance. 

---
# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening 

**Authors**: Tae Eun Choi, Sumin Shim, Junhyeok Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17651)  

**Abstract**: Generative inbetweening (GI) seeks to synthesize realistic intermediate frames between the first and last keyframes beyond mere interpolation. As sequences become sparser and motions larger, previous GI models struggle with inconsistent frames with unstable pacing and semantic misalignment. Since GI involves fixed endpoints and numerous plausible paths, this task requires additional guidance gained from the keyframes and text to specify the intended path. Thus, we give semantic and temporal guidance from the keyframes and text onto each intermediate frame through Keyframe-anchored Attention Bias. We also better enforce frame consistency with Rescaled Temporal RoPE, which allows self-attention to attend to keyframes more faithfully. TGI-Bench, the first benchmark specifically designed for text-conditioned GI evaluation, enables challenge-targeted evaluation to analyze GI models. Without additional training, our method achieves state-of-the-art frame consistency, semantic fidelity, and pace stability for both short and long sequences across diverse challenges. 

---
# Automated Grammar-based Algebraic Multigrid Design With Evolutionary Algorithms 

**Authors**: Dinesh Parthasarathy, Wayne Mitchell, Arjun Gambhir, Harald Köstler, Ulrich Rüde  

**Link**: [PDF](https://arxiv.org/pdf/2603.17641)  

**Abstract**: Although multigrid is asymptotically optimal for solving many important partial differential equations, its efficiency relies heavily on the careful selection of the individual algorithmic components. In contrast to recent approaches that can optimize certain multigrid components using deep learning techniques, we adopt a complementary strategy, employing evolutionary algorithms to construct efficient multigrid cycles from proven algorithmic building blocks. Here, we will present its application to generate efficient algebraic multigrid methods with so-called \emph{flexible cycling}, that is, level-specific smoothing sequences and non-recursive cycling patterns. The search space with such non-standard cycles is intractable to navigate manually, and is generated using genetic programming (GP) guided by context-free grammars. Numerical experiments with the linear algebra library, \emph{hypre}, demonstrate the potential of these non-standard GP cycles to improve multigrid performance both as a solver and a preconditioner. 

---
# Benchmarking Reinforcement Learning via Stochastic Converse Optimality: Generating Systems with Known Optimal Policies 

**Authors**: Sinan Ibrahim, Grégoire Ouerdane, Hadi Salloum, Henni Ouerdane, Stefan Streif, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.17631)  

**Abstract**: The objective comparison of Reinforcement Learning (RL) algorithms is notoriously complex as outcomes and benchmarking of performances of different RL approaches are critically sensitive to environmental design, reward structures, and stochasticity inherent in both algorithmic learning and environmental dynamics. To manage this complexity, we introduce a rigorous benchmarking framework by extending converse optimality to discrete-time, control-affine, nonlinear systems with noise. Our framework provides necessary and sufficient conditions, under which a prescribed value function and policy are optimal for constructed systems, enabling the systematic generation of benchmark families via homotopy variations and randomized parameters. We validate it by automatically constructing diverse environments, demonstrating our framework's capacity for a controlled and comprehensive evaluation across algorithms. By assessing standard methods against a ground-truth optimum, our work delivers a reproducible foundation for precise and rigorous RL benchmarking. 

---
# rSDNet: Unified Robust Neural Learning against Label Noise and Adversarial Attacks 

**Authors**: Suryasis Jana, Abhik Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2603.17628)  

**Abstract**: Neural networks are central to modern artificial intelligence, yet their training remains highly sensitive to data contamination. Standard neural classifiers are trained by minimizing the categorical cross-entropy loss, corresponding to maximum likelihood estimation under a multinomial model. While statistically efficient under ideal conditions, this approach is highly vulnerable to contaminated observations including label noises corrupting supervision in the output space, and adversarial perturbations inducing worst-case deviations in the input space. In this paper, we propose a unified and statistically grounded framework for robust neural classification that addresses both forms of contamination within a single learning objective. We formulate neural network training as a minimum-divergence estimation problem and introduce rSDNet, a robust learning algorithm based on the general class of $S$-divergences. The resulting training objective inherits robustness properties from classical statistical estimation, automatically down-weighting aberrant observations through model probabilities. We establish essential population-level properties of rSDNet, including Fisher consistency, classification calibration implying Bayes optimality, and robustness guarantees under uniform label noise and infinitesimal feature contamination. Experiments on three benchmark image classification datasets show that rSDNet improves robustness to label corruption and adversarial attacks while maintaining competitive accuracy on clean data, Our results highlight minimum-divergence learning as a principled and effective framework for robust neural classification under heterogeneous data contamination. 

---
# A Contextual Help Browser Extension to Assist Digital Illiterate Internet Users 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2603.17592)  

**Abstract**: This paper describes the design, implementation, and evaluation of a browser extension that provides contextual help to users who hover over technological acronyms and abbreviations on web pages. The extension combines a curated technical dictionary with OpenAI's large language model (LLM) to deliver on-demand definitions through lightweight tooltip overlays. A dual-layer artificial intelligence (AI) pipeline, comprising Google Cloud's Natural Language Processing (NLP) taxonomy API and OpenAI's ChatGPT, classifies each visited page as technology-related before activating the tooltip logic, thereby reducing false-positive detections. A mixed-methods study with 25 participants evaluated the tool's effect on reading comprehension and information-retrieval time among users with low to intermediate digital literacy. Results show that 92% of participants reported improved understanding of technical terms, 96% confirmed time savings over manual web searches, and all participants found the tooltips non-disruptive. Dictionary-based definitions were appended in an average of 2135 ms, compared to 16429 ms for AI-generated definitions and a mean manual search time of 17200 ms per acronym. The work demonstrates a practical, real-time approach to bridging the digital literacy gap and points toward extending contextual help to other domains such as medicine, law, and finance. 

---
# Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing 

**Authors**: Seongrae Noh, SeungWon Seo, Gyeong-Moon Park, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17583)  

**Abstract**: Editing a 3D indoor scene from natural language is conceptually straightforward but technically challenging. Existing open-vocabulary systems often regenerate large portions of a scene or rely on image-space edits that disrupt spatial structure, resulting in unintended global changes or physically inconsistent layouts. These limitations stem from treating editing primarily as a generative task. We take a different view. A user instruction defines a desired world state, and editing should be the minimal sequence of actions that makes this state true while preserving everything else. This perspective motivates Edit-As-Act, a framework that performs open-vocabulary scene editing as goal-regressive planning in 3D space. Given a source scene and free-form instruction, Edit-As-Act predicts symbolic goal predicates and plans in EditLang, a PDDL-inspired action language that we design with explicit preconditions and effects encoding support, contact, collision, and other geometric relations. A language-driven planner proposes actions, and a validator enforces goal-directedness, monotonicity, and physical feasibility, producing interpretable and physically coherent transformations. By separating reasoning from low-level generation, Edit-As-Act achieves instruction fidelity, semantic consistency, and physical plausibility - three criteria that existing paradigms cannot satisfy together. On E2A-Bench, our benchmark of 63 editing tasks across 9 indoor environments, Edit-As-Act significantly outperforms prior approaches across all edit types and scene categories. 

---
# Identifying Latent Actions and Dynamics from Offline Data via Demonstrator Diversity 

**Authors**: Felix Schur  

**Link**: [PDF](https://arxiv.org/pdf/2603.17577)  

**Abstract**: Can latent actions and environment dynamics be recovered from offline trajectories when actions are never observed? We study this question in a setting where trajectories are action-free but tagged with demonstrator identity. We assume that each demonstrator follows a distinct policy, while the environment dynamics are shared across demonstrators and identity affects the next observation only through the chosen action. Under these assumptions, the conditional next-observation distribution $p(o_{t+1}\mid o_t,e)$ is a mixture of latent action-conditioned transition kernels with demonstrator-specific mixing weights. We show that this induces, for each state, a column-stochastic nonnegative matrix factorization of the observable conditional distribution. Using sufficiently scattered policy diversity and rank conditions, we prove that the latent transitions and demonstrator policies are identifiable up to permutation of the latent action labels. We extend the result to continuous observation spaces via a Gram-determinant minimum-volume criterion, and show that continuity of the transition map over a connected state space upgrades local permutation ambiguities to a single global permutation. A small amount of labeled action data then suffices to fix this final ambiguity. These results establish demonstrator diversity as a principled source of identifiability for learning latent actions and dynamics from offline RL data. 

---
# Unsupervised Symbolic Anomaly Detection 

**Authors**: Md Maruf Hossain, Tim Katzke, Simon Klüttermann, Emmanuel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2603.17575)  

**Abstract**: We propose SYRAN, an unsupervised anomaly detection method based on symbolic regression. Instead of encoding normal patterns in an opaque, high-dimensional model, our method learns an ensemble of human-readable equations that describe symbolic invariants: functions that are approximately constant on normal data. Deviations from these invariants yield anomaly scores, so that the detection logic is interpretable by construction, rather than via post-hoc explanation. Experimental results demonstrate that SYRAN is highly interpretable, providing equations that correspond to known scientific or medical relationships, and maintains strong anomaly detection performance comparable to that of state-of-the-art methods. 

---
# FoMo X: Modular Explainability Signals for Outlier Detection Foundation Models 

**Authors**: Simon Klüttermann, Tim Katzke, Phuong Huong Nguyen, Emmanuel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2603.17570)  

**Abstract**: Tabular foundation models, specifically Prior-Data Fitted Networks (PFNs), have revolutionized outlier detection (OD) by enabling unsupervised zero-shot adaptation to new datasets without training. However, despite their predictive power, these models typically function as opaque black boxes, outputting scalar outlier scores that lack the operational context required for safety-critical decision-making. Existing post-hoc explanation methods are often computationally prohibitive for real-time deployment or fail to capture the epistemic uncertainty inherent in zero-shot inference. In this work, we introduce FoMo-X, a modular framework that equips OD foundation models with intrinsic, lightweight diagnostic capabilities. We leverage the insight that the frozen embeddings of a pretrained PFN backbone already encode rich, context-conditioned relational information. FoMo-X attaches auxiliary diagnostic heads to these embeddings, trained offline using the same generative simulator prior as the backbone. This allows us to distill computationally expensive properties, such as Monte Carlo dropout based epistemic uncertainty, into a deterministic, single-pass inference. We instantiate FoMo-X with two novel heads: a Severity Head that discretizes deviations into interpretable risk tiers, and an Uncertainty Head that provides calibrated confidence measures. Extensive evaluation on synthetic and real-world benchmarks (ADBench) demonstrates that FoMo-X recovers ground-truth diagnostic signals with high fidelity and negligible inference overhead. By bridging the gap between foundation model performance and operational explainability, FoMo-X offers a scalable path toward trustworthy, zero-shot outlier detection. 

---
# FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion 

**Authors**: Hugo Caselles-Dupré, Mathis Koroglu, Guillaume Jeanneret, Arnaud Dapogny, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2603.17555)  

**Abstract**: Diffusion-based image-to-video (I2V) models are increasingly effective, yet they struggle to scale to ultra-high-resolution inputs (e.g., 4K). Generating videos at the model's native resolution often loses fine-grained structure, whereas high-resolution tiled denoising preserves local detail but breaks global layout consistency. This failure mode is particularly severe in the fresco animation setting: monumental artworks containing many distinct characters, objects, and semantically different sub-scenes that must remain spatially coherent over time. We introduce FrescoDiffusion, a training-free method for coherent large-format I2V generation from a single complex image. The key idea is to augment tiled denoising with a precomputed latent prior: we first generate a low-resolution video at the underlying model resolution and upsample its latent trajectory to obtain a global reference that captures long-range temporal and spatial structure. For 4K generation, we compute per-tile noise predictions and fuse them with this reference at every diffusion timestep by minimizing a single weighted least-squares objective in model-output space. The objective combines a standard tile-merging criterion with our regularization term, yielding a closed-form fusion update that strengthens global coherence while retaining fine detail. We additionally provide a spatial regularization variable that enables region-level control over where motion is allowed. Experiments on the VBench-I2V dataset and our proposed fresco I2V dataset show improved global consistency and fidelity over tiled baselines, while being computationally efficient. Our regularization enables explicit controllability of the trade-off between creativity and consistency. 

---
# CLeAN: Continual Learning Adaptive Normalization in Dynamic Environments 

**Authors**: Isabella Marasco, Davide Evangelista, Elena Loli Piccolomini, Michele Colajanni  

**Link**: [PDF](https://arxiv.org/pdf/2603.17548)  

**Abstract**: Artificial intelligence systems predominantly rely on static data distributions, making them ineffective in dynamic real-world environments, such as cybersecurity, autonomous transportation, or finance, where data shifts frequently. Continual learning offers a potential solution by enabling models to learn from sequential data while retaining prior knowledge. However, a critical and underexplored issue in this domain is data normalization. Conventional normalization methods, such as min-max scaling, presuppose access to the entire dataset, which is incongruent with the sequential nature of continual learning. In this paper we introduce Continual Learning Adaptive Normalization (CLeAN), a novel adaptive normalization technique designed for continual learning in tabular data. CLeAN involves the estimation of global feature scales using learnable parameters that are updated via an Exponential Moving Average (EMA) module, enabling the model to adapt to evolving data distributions. Through comprehensive evaluations on two datasets and various continual learning strategies, including Resevoir Experience Replay, A-GEM, and EwC we demonstrate that CLeAN not only improves model performance on new data but also mitigates catastrophic forgetting. The findings underscore the importance of adaptive normalization in enhancing the stability and effectiveness of tabular data, offering a novel perspective on the use of normalization to preserve knowledge in dynamic learning environments. 

---
# Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant and Efficient Point Cloud Analysis 

**Authors**: Jaein Kim, Hee Bin Yoo, Dong-Sig Han, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17538)  

**Abstract**: A symmetry on rigid motion is one of the salient factors in efficient learning of 3D point cloud problems. Group convolution has been a representative method to extract equivariant features, but its realizations have struggled to retain both rigorous symmetry and scalability simultaneously. We advocate utilizing the intertwiner framework to resolve this trade-off, but previous works on it, which did not achieve complete SE(3) symmetry or scalability to large-scale problems, necessitate a more advanced kernel architecture. We present Equivariant Coordinate-based Kernel Convolution, or ECKConv. It acquires SE(3) equivariance from the kernel domain defined in a double coset space, and its explicit kernel design using coordinate-based networks enhances its learning capability and memory efficiency. The experiments on diverse point cloud tasks, e.g., classification, pose registration, part segmentation, and large-scale semantic segmentation, validate the rigid equivariance, memory scalability, and outstanding performance of ECKConv compared to state-of-the-art equivariant methods. 

---
# Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing 

**Authors**: Pengzhen Chen, Yanwei Liu, Xiaoyan Gu, Xiaojun Chen, Wu Liu, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17531)  

**Abstract**: Recent advancements in diffusion-based image editing pose a significant threat to the authenticity of digital visual content. Traditional embedding-based watermarking methods often introduce perceptible perturbations to maintain robustness, inevitably compromising visual fidelity. Meanwhile, existing zero-watermarking approaches, typically relying on global image features, struggle to withstand sophisticated manipulations. In this work, we uncover a key observation: while individual image patches undergo substantial alterations during AI-based editing, the relational distance between patch pairs remains relatively invariant. Leveraging this property, we propose Relational Zero-Watermarking (Rel-Zero), a novel framework that requires no modification to the original image but derives a unique zero-watermark from these editing-invariant patch relations. By grounding the watermark in intrinsic structural consistency rather than absolute appearance, Rel-Zero provides a non-invasive yet resilient mechanism for content authentication. Extensive experiments demonstrate that Rel-Zero achieves substantially improved robustness across diverse editing models and manipulations compared to prior zero-watermarking approaches. 

---
# AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection 

**Authors**: Manuel Barusco, Davide Dalle Pezze, Francesco Borsatti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2603.17530)  

**Abstract**: Visual Anomaly Detection (VAD) is crucial for industrial inspection, yet most existing methods are limited to single-category scenarios, failing to address the multi-class and continual learning demands of real-world environments. While Teacher-Student (TS) architectures are efficient, they remain unexplored for the Continual Setting. To bridge this gap, we propose AdapTS, a unified TS framework designed for multi-class and continual settings, optimized for edge deployment. AdapTS eliminates the need for two different architectures by utilizing a single shared frozen backbone and injecting lightweight trainable adapters into the student pathway. Training is enhanced via a segmentation-guided objective and synthetic Perlin noise, while a prototype-based task identification mechanism dynamically selects adapters at inference with 99\% accuracy.
Experiments on MVTec AD and VisA demonstrate that AdapTS matches the performance of existing TS methods across multi-class and continual learning scenarios, while drastically reducing memory overhead. Our lightest variant, AdapTS-S, requires only 8 MB of additional memory, 13x less than STFPM (95 MB), 48x less than RD4AD (360 MB), and 149x less than DeSTSeg (1120 MB), making it a highly scalable solution for edge deployment in complex industrial environments. 

---
# AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting 

**Authors**: Binqing Wu, Zongjiang Shang, Shiyu Liu, Jianlong Huang, Jiahui Xu, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17529)  

**Abstract**: Accurate air quality forecasting is essential for public health and environmental sustainability, but remains challenging due to the complex pollutant dynamics. Existing deep learning methods often model pollutant dynamics as an instantaneous process, overlooking the intrinsic delays in pollutant propagation. Thus, we propose AirDDE, the first neural delay differential equation framework in this task that integrates delay modeling into a continuous-time pollutant evolution under physical guidance. Specifically, two novel components are introduced: (1) a memory-augmented attention module that retrieves globally and locally historical features, which can adaptively capture delay effects modulated by multifactor data; and (2) a physics-guided delay evolving function, grounded in the diffusion-advection equation, that models diffusion, delayed advection, and source/sink terms, which can capture delay-aware pollutant accumulation patterns with physical plausibility. Extensive experiments on three real-world datasets demonstrate that AirDDE achieves the state-of-the-art forecasting performance with an average MAE reduction of 8.79\% over the best baselines. The code is available at this https URL. 

---
# KineVLA: Towards Kinematics-Aware Vision-Language-Action Models with Bi-Level Action Decomposition 

**Authors**: Gaoge Han, Zhengqing Gao, Ziwen Li, Jiaxin Huang, Shaoli Huang, Fakhri Karray, Mingming Gong, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17524)  

**Abstract**: In this paper, we introduce a novel kinematics-rich vision-language-action (VLA) task, in which language commands densely encode diverse kinematic attributes (such as direction, trajectory, orientation, and relative displacement) from initiation through completion, at key moments, unlike existing action instructions that capture kinematics only coarsely or partially, thereby supporting fine-grained and personalized manipulation. In this setting, where task goals remain invariant while execution trajectories must adapt to instruction-level kinematic specifications. To address this challenge, we propose KineVLA, a vision-language-action framework that explicitly decouples goal-level invariance from kinematics-level variability through a bi-level action representation and bi-level reasoning tokens to serve as explicit, supervised intermediate variables that align language and action. To support this task, we construct the kinematics-aware VLA datasets spanning both simulation and real-world robotic platforms, featuring instruction-level kinematic variations and bi-level annotations. Extensive experiments on LIBERO and a Realman-75 robot demonstrate that KineVLA consistently outperforms strong VLA baselines on kinematics-sensitive benchmarks, achieving more precise, controllable, and generalizable manipulation behaviors. 

---
# Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions 

**Authors**: Madhav S. Baidya, S. S. Baidya, Chirag Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17522)  

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for robust and generalizable detectors of machine-generated text. Existing benchmarks typically evaluate a single detector on a single dataset under ideal conditions, leaving open questions about cross-domain transfer, cross-LLM generalization, and adversarial robustness.
We present a comprehensive benchmark evaluating diverse detection approaches across two corpora: HC3 (23,363 human-ChatGPT pairs) and ELI5 (15,000 human-Mistral-7B pairs). Methods include classical classifiers, fine-tuned transformer encoders (BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa-v3), a CNN, an XGBoost stylometric model, perplexity-based detectors, and LLM-as-detector prompting.
Results show that transformer models achieve near-perfect in-distribution performance but degrade under domain shift. The XGBoost stylometric model matches performance while remaining interpretable. LLM-based detectors underperform and are affected by generator-detector identity bias. Perplexity-based methods exhibit polarity inversion, with modern LLM outputs showing lower perplexity than human text, but remain effective when corrected. No method generalizes robustly across domains and LLM sources. 

---
# QuantFL: Sustainable Federated Learning for Edge IoT via Pre-Trained Model Quantisation 

**Authors**: Charuka Herath, Yogachandran Rahulamathavan, Varuna De Silva, Sangarapillai Lambotharan  

**Link**: [PDF](https://arxiv.org/pdf/2603.17507)  

**Abstract**: Federated Learning (FL) enables privacy-preserving intelligence on Internet of Things (IoT) devices but incurs a significant carbon footprint due to the high energy cost of frequent uplink transmission. While pre-trained models are increasingly available on edge devices, their potential to reduce the energy overhead of fine-tuning remains underexplored. In this work, we propose QuantFL, a sustainable FL framework that leverages pre-trained initialisation to enable aggressive, computationally lightweight quantisation. We demonstrate that pre-training naturally concentrates update statistics, allowing us to use memory-efficient bucket quantisation without the energy-intensive overhead of complex error-feedback mechanisms. On MNIST and CIFAR-100, QuantFL reduces total communication by 40\% ($\simeq40\%$ total-bit reduction with full-precision downlink; $\geq80\%$ on uplink or when downlink is quantised) while matching or exceeding uncompressed baselines under strict bandwidth budgets; BU attains 89.00\% (MNIST) and 66.89\% (CIFAR-100) test accuracy with orders of magnitude fewer bits. We also account for uplink and downlink costs and provide ablations on quantisation levels and initialisation. QuantFL delivers a practical, "green" recipe for scalable training on battery-constrained IoT networks. 

---
# Auto-Unrolled Proximal Gradient Descent: An AutoML Approach to Interpretable Waveform Optimization 

**Authors**: Ahmet Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2603.17478)  

**Abstract**: This study explores the combination of automated machine learning (AutoML) with model-based deep unfolding (DU) for optimizing wireless beamforming and waveforms. We convert the iterative proximal gradient descent (PGD) algorithm into a deep neural network, wherein the parameters of each layer are learned instead of being predetermined. Additionally, we enhance the architecture by incorporating a hybrid layer that performs a learnable linear gradient transformation prior to the proximal projection. By utilizing AutoGluon with a tree-structured parzen estimator (TPE) for hyperparameter optimization (HPO) across an expanded search space, which includes network depth, step-size initialization, optimizer, learning rate scheduler, layer type, and post-gradient activation, the proposed auto-unrolled PGD (Auto-PGD) achieves 98.8% of the spectral efficiency of a traditional 200-iteration PGD solver using only five unrolled layers, while requiring only 100 training samples. We also address a gradient normalization issue to ensure consistent performance during training and evaluation, and we illustrate per-layer sum-rate logging as a tool for transparency. These contributions highlight a notable reduction in the amount of training data and inference cost required, while maintaining high interpretability compared to conventional black-box architectures. 

---
# UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models 

**Authors**: Segyu Lee, Boryeong Cho, Hojung Jung, Seokhyun An, Juhyeong Kim, Jaehyun Kwak, Yongjin Yang, Sangwon Jang, Youngrok Park, Wonjun Chang, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2603.17476)  

**Abstract**: Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities. To address this gap, we introduce UniSAFE, the first comprehensive benchmark for system-level safety evaluation of UMMs across 7 I/O modality combinations, spanning conventional tasks and novel multimodal-context image generation settings. UniSAFE is built with a shared-target design that projects common risk scenarios across task-specific I/O configurations, enabling controlled cross-task comparisons of safety failures. Comprising 6,802 curated instances, we use UniSAFE to evaluate 15 state-of-the-art UMMs, both proprietary and open-source. Our results reveal critical vulnerabilities across current UMMs, including elevated safety violations in multi-image composition and multi-turn settings, with image-output tasks consistently more vulnerable than text-output tasks. These findings highlight the need for stronger system-level safety alignment for UMMs. Our code and data are publicly available at this https URL 

---
# Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning 

**Authors**: Zelin Zang, Yehui Yang, Fei Wang, Liangyu Li, Baigui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.17474)  

**Abstract**: Unsupervised Domain Adaptation (UDA) seeks to transfer knowledge from a labeled source domain to an unlabeled target domain but often suffers from severe domain and scale gaps that degrade performance. Existing cross-attention-based transformers can align features across domains, yet they struggle to preserve content semantics under large appearance and scale variations. To explicitly address these challenges, we introduce the concept of beneficial noise, which regularizes cross-attention by injecting controlled perturbations, encouraging the model to ignore style distractions and focus on content. We propose the Domain-Adaptive Cross-Scale Matching (DACSM) framework, which consists of a Domain-Adaptive Transformer (DAT) for disentangling domain-shared content from domain-specific style, and a Cross-Scale Matching (CSM) module that adaptively aligns features across multiple resolutions. DAT incorporates beneficial noise into cross-attention, enabling progressive domain translation with enhanced robustness, yielding content-consistent and style-invariant representations. Meanwhile, CSM ensures semantic consistency under scale changes. Extensive experiments on VisDA-2017, Office-Home, and DomainNet demonstrate that DACSM achieves state-of-the-art performance, with up to +2.3% improvement over CDTrans on VisDA-2017. Notably, DACSM achieves a +5.9% gain on the challenging "truck" class of VisDA, evidencing the strength of beneficial noise in handling scale discrepancies. These results highlight the effectiveness of combining domain translation, beneficial-noise-enhanced attention, and scale-aware alignment for robust cross-domain representation learning. 

---
# VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection 

**Authors**: Chupeng Liu, Jiyong Rao, Shangquan Sun, Runkai Zhao, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.17470)  

**Abstract**: Monocular 3D object detection typically relies on pseudo-labeling techniques to reduce dependency on real-world annotations. Recent advances demonstrate that deterministic linguistic cues can serve as effective auxiliary weak supervision signals, providing complementary semantic context. However, hand-crafted textual descriptions struggle to capture the inherent visual diversity of individuals across scenes, limiting the model's ability to learn scene-aware representations. To address this challenge, we propose Visual-referred Probabilistic Prompt Learning (VirPro), an adaptive multi-modal pretraining paradigm that can be seamlessly integrated into diverse weakly supervised monocular 3D detection frameworks. Specifically, we generate a diverse set of learnable, instance-conditioned prompts across scenes and store them in an Adaptive Prompt Bank (APB). Subsequently, we introduce Multi-Gaussian Prompt Modeling (MGPM), which incorporates scene-based visual features into the corresponding textual embeddings, allowing the text prompts to express visual uncertainties. Then, from the fused vision-language embeddings, we decode a prompt-targeted Gaussian, from which we derive a unified object-level prompt embedding for each instance. RoI-level contrastive matching is employed to enforce modality alignment, bringing embeddings of co-occurring objects within the same scene closer in the latent space, thus enhancing semantic coherence. Extensive experiments on the KITTI benchmark demonstrate that integrating our pretraining paradigm consistently yields substantial performance gains, achieving up to a 4.8% average precision improvement than the baseline. 

---
# VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation 

**Authors**: Junyoung Kim, Woojoo Kim, Jaehyung Lim, Dongha Kim, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17450)  

**Abstract**: Sequential Recommendation (SR) in multimodal settings typically relies on small frozen pretrained encoders, which limits semantic capacity and prevents Collaborative Filtering (CF) signals from being fully integrated into item representations. Inspired by the recent success of Large Language Models (LLMs) as high-capacity embedders, we investigate the use of Vision-Language Models (VLMs) as CF-aware multimodal encoders for SR. However, we find that standard contrastive supervised fine-tuning (SFT), which adapts VLMs for embedding generation and injects CF signals, can amplify its inherent modality collapse. In this state, optimization is dominated by a single modality while the other degrades, ultimately undermining recommendation accuracy. To address this, we propose VLM2Rec, a VLM embedder-based framework for multimodal sequential recommendation designed to ensure balanced modality utilization. Specifically, we introduce Weak-modality Penalized Contrastive Learning to rectify gradient imbalance during optimization and Cross-Modal Relational Topology Regularization to preserve geometric consistency between modalities. Extensive experiments demonstrate that VLM2Rec consistently outperforms state-of-the-art baselines in both accuracy and robustness across diverse scenarios. 

---
# AdaZoom-GUI: Adaptive Zoom-based GUI Grounding with Instruction Refinement 

**Authors**: Siqi Pei, Liang Tang, Tiaonan Duan, Long Chen, Shuxian Li, Kaer Huang, Yanzhe Jing, Yiqiang Yan, Bo Zhang, Chenghao Jiang, Borui Zhang, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17441)  

**Abstract**: GUI grounding is a critical capability for vision-language models (VLMs) that enables automated interaction with graphical user interfaces by locating target elements from natural language instructions. However, grounding on GUI screenshots remains challenging due to high-resolution images, small UI elements, and ambiguous user instructions. In this work, we propose AdaZoom-GUI, an adaptive zoom-based GUI grounding framework that improves both localization accuracy and instruction understanding. Our approach introduces an instruction refinement module that rewrites natural language commands into explicit and detailed descriptions, allowing the grounding model to focus on precise element localization. In addition, we design a conditional zoom-in strategy that selectively performs a second-stage inference on predicted small elements, improving localization accuracy while avoiding unnecessary computation and context loss on simpler cases. To support this framework, we construct a high-quality GUI grounding dataset and train the grounding model using Group Relative Policy Optimization (GRPO), enabling the model to predict both click coordinates and element bounding boxes. Experiments on public benchmarks demonstrate that our method achieves state-of-the-art performance among models with comparable or even larger parameter sizes, highlighting its effectiveness for high-resolution GUI understanding and practical GUI agent deployment. 

---
# Baguan-TS: A Sequence-Native In-Context Learning Model for Time Series Forecasting with Covariates 

**Authors**: Linxiao Yang, Xue Jiang, Gezheng Xu, Tian Zhou, Min Yang, ZhaoYang Zhu, Linyuan Geng, Zhipeng Zeng, Qiming Chen, Xinyue Gu, Rong Jin, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.17439)  

**Abstract**: Transformers enable in-context learning (ICL) for rapid, gradient-free adaptation in time series forecasting, yet most ICL-style approaches rely on tabularized, hand-crafted features, while end-to-end sequence models lack inference-time adaptation. We bridge this gap with a unified framework, Baguan-TS, which integrates the raw-sequence representation learning with ICL, instantiated by a 3D Transformer that attends jointly over temporal, variable, and context axes. To make this high-capacity model practical, we tackle two key hurdles: (i) calibration and training stability, improved with a feature-agnostic, target-space retrieval-based local calibration; and (ii) output oversmoothing, mitigated via context-overfitting strategy. On public benchmark with covariates, Baguan-TS consistently outperforms established baselines, achieving the highest win rate and significant reductions in both point and probabilistic forecasting metrics. Further evaluations across diverse real-world energy datasets demonstrate its robustness, yielding substantial improvements. 

---
# TimeAPN: Adaptive Amplitude-Phase Non-Stationarity Normalization for Time Series Forecasting 

**Authors**: Yue Hu, Jialiang Tang, Siwei Yu, Baosheng Yu, Jing Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17436)  

**Abstract**: Non-stationarity is a fundamental challenge in multivariate long-term time series forecasting, often manifested as rapid changes in amplitude and phase. These variations lead to severe distribution shifts and consequently degrade predictive performance. Existing normalization-based methods primarily rely on first- and second-order statistics, implicitly assuming that distributions evolve smoothly and overlooking fine-grained temporal dynamics. To address these limitations, we propose TimeAPN, an Adaptive Amplitude-Phase Non-Stationarity Normalization framework that explicitly models and predicts non-stationary factors from both the time and frequency domains. Specifically, TimeAPN first models the mean sequence jointly in the time and frequency domains, and then forecasts its evolution over future horizons. Meanwhile, phase information is extracted in the frequency domain, and the phase discrepancy between the predicted and ground-truth future sequences is explicitly modeled to capture temporal misalignment. Furthermore, TimeAPN incorporates amplitude information into an adaptive normalization mechanism, enabling the model to effectively account for abrupt fluctuations in signal energy. The predicted non-stationary factors are subsequently integrated with the backbone forecasting outputs through a collaborative de-normalization process to reconstruct the final non-stationary time series. The proposed framework is model-agnostic and can be seamlessly integrated with various forecasting backbones. Extensive experiments on seven real-world multivariate datasets demonstrate that TimeAPN consistently improves long-term forecasting accuracy across multiple prediction horizons and outperforms state-of-the-art reversible normalization methods. 

---
# The Phasor Transformer: Resolving Attention Bottlenecks on the Unit Circle 

**Authors**: Dibakar Sigdel  

**Link**: [PDF](https://arxiv.org/pdf/2603.17433)  

**Abstract**: Transformer models have redefined sequence learning, yet dot-product self-attention introduces a quadratic token-mixing bottleneck for long-context time-series. We introduce the \textbf{Phasor Transformer} block, a phase-native alternative representing sequence states on the unit-circle manifold $S^1$. Each block combines lightweight trainable phase-shifts with parameter-free Discrete Fourier Transform (DFT) token coupling, achieving global $\mathcal{O}(N\log N)$ mixing without explicit attention maps. Stacking these blocks defines the \textbf{Large Phasor Model (LPM)}. We validate LPM on autoregressive time-series prediction over synthetic multi-frequency benchmarks. Operating with a highly compact parameter budget, LPM learns stable global dynamics and achieves competitive forecasting behavior compared to conventional self-attention baselines. Our results establish an explicit efficiency-performance frontier, demonstrating that large-model scaling for time-series can emerge from geometry-constrained phase computation with deterministic global coupling, offering a practical path toward scalable temporal modeling in oscillatory domains. 

---
# Caging the Agents: A Zero Trust Security Architecture for Autonomous AI in Healthcare 

**Authors**: Saikat Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2603.17419)  

**Abstract**: Autonomous AI agents powered by large language models are being deployed in production with capabilities including shell execution, file system access, database queries, and multi-party communication. Recent red teaming research demonstrates that these agents exhibit critical vulnerabilities in realistic settings: unauthorized compliance with non-owner instructions, sensitive information disclosure, identity spoofing, cross-agent propagation of unsafe practices, and indirect prompt injection through external resources [7]. In healthcare environments processing Protected Health Information, every such vulnerability becomes a potential HIPAA violation. This paper presents a security architecture deployed for nine autonomous AI agents in production at a healthcare technology company. We develop a six-domain threat model for agentic AI in healthcare covering credential exposure, execution capability abuse, network egress exfiltration, prompt integrity failures, database access risks, and fleet configuration drift. We implement four-layer defense in depth: (1) kernel level workload isolation using gVisor on Kubernetes, (2) credential proxy sidecars preventing agent containers from accessing raw secrets, (3) network egress policies restricting each agent to allowlisted destinations, and (4) a prompt integrity framework with structured metadata envelopes and untrusted content labeling. We report results from 90 days of deployment including four HIGH severity findings discovered and remediated by an automated security audit agent, progressive fleet hardening across three VM image generations, and defense coverage mapped to all eleven attack patterns from recent literature. All configurations, audit tooling, and the prompt integrity framework are released as open source. 

---
# Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression 

**Authors**: Xinning Chai, Zhengxue Cheng, Xin Li, Rong Xie, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.17408)  

**Abstract**: Recent diffusion-based extreme image compression methods have demonstrated remarkable performance at ultra-low bitrates. However, most approaches require training separate diffusion models for each target bitrate, resulting in substantial computational overhead and hindering practical deployment. Meanwhile, recent studies have shown that joint super-resolution can serve as an effective approach for enhancing low-bitrate reconstruction. However, when moving toward ultra-low bitrate regimes, these methods struggle due to severe information loss, and their reliance on fixed super-resolution scales prevents flexible adaptation across diverse bitrates.
To address these limitations, we propose ASSR-EIC, a novel image compression framework that leverages arbitrary-scale super-resolution (ASSR) to support variable-rate extreme image compression (EIC). An arbitrary-scale downsampling module is introduced at the encoder side to provide controllable rate reduction, while a diffusion-based, joint degradation-aware ASSR decoder enables rate-adaptive reconstruction within a single model. We exploit the compression- and rescaling-aware diffusion prior to guide the reconstruction, yielding high fidelity and high realism restoration across diverse compression and rescaling settings. Specifically, we design a global compression-rescaling adaptor that offers holistic guidance for rate adaptation, and a local compression-rescaling modulator that dynamically balances generative and fidelity-oriented behaviors to achieve fine-grained, bitrate-adaptive detail restoration. To further enhance reconstruction quality, we introduce a dual semantic-enhanced design.
Extensive experiments demonstrate that ASSR-EIC delivers state-of-the-art performance in extreme image compression while simultaneously supporting flexible bitrate control and adaptive rate-dependent reconstruction. 

---
# CRE-T1 Preview Technical Report: Beyond Contrastive Learning for Reasoning-Intensive Retrieval 

**Authors**: Guangzhi Wang, Yinghao Jiao, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17387)  

**Abstract**: The central challenge of reasoning-intensive retrieval lies in identifying implicitreasoning relationships between queries and documents, rather than superficial se-mantic or lexical similarity. The contrastive learning paradigm is fundamentallya static representation consolidation technique: during training, it encodes hier-archical relevance concepts into fixed geometric structures in the vector space,and at inference time it cannot dynamically adjust relevance judgments accord-ing to the specific reasoning demands of each query. Consequently, performancedegrades noticeably when vocabulary mismatch exists between queries and doc-uments or when implicit reasoning is required to establish relevance. This pa-per proposes Thought 1 (T1), a generative retrieval model that shifts relevancemodeling from static alignment to dynamic reasoning. On the query side, T1 dy-namically generates intermediate reasoning trajectories for each query to bridgeimplicit reasoning relationships and uses <embtoken> as a semantic aggregationpoint for the reasoning output. On the document side, it employs an instruction+ text + <embtoken> encoding format to support high-throughput indexing. Tointernalize dynamic reasoning capabilities into vector representations, we adopt athree-stage training curriculum and introduce GRPO in the third stage, enablingthe model to learn optimal derivation strategies for different queries through trial-and-error reinforcement learning. On the BRIGHT benchmark, T1-4B exhibitsstrong performance under the original query setting, outperforming larger modelstrained with contrastive learning overall, and achieving performance comparableto multi-stage retrieval pipelines. The results demonstrate that replacing static rep-resentation alignment with dynamic reasoning generation can effectively improvereasoning-intensive retrieval performance. 

---
# SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction 

**Authors**: Shuizhou Chen, Lang Yu, Kedu Jin, Songming Zhang, Hao Wu, Wenxuan Huang, Sheng Xu, Quan Qian, Qin Chen, Lei Bai, Siqi Sun, Zhangyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17380)  

**Abstract**: Virtual cell models aim to enable in silico experimentation by predicting how cells respond to genetic, chemical, or cytokine perturbations from single-cell measurements. In practice, however, large-scale perturbation prediction remains constrained by three coupled bottlenecks: inefficient training and inference pipelines, unstable modeling in high-dimensional sparse expression space, and evaluation protocols that overemphasize reconstruction-like accuracy while underestimating biological fidelity. In this work we present a specialized large-scale foundation model SCALE for virtual cell perturbation prediction that addresses the above limitations jointly. First, we build a BioNeMo-based training and inference framework that substantially improves data throughput, distributed scalability, and deployment efficiency, yielding 12.51* speedup on pretrain and 1.29* on inference over the prior SOTA pipeline under matched system settings. Second, we formulate perturbation prediction as conditional transport and implement it with a set-aware flow architecture that couples LLaMA-based cellular encoding with endpoint-oriented supervision. This design yields more stable training and stronger recovery of perturbation effects. Third, we evaluate the model on Tahoe-100M using a rigorous cell-level protocol centered on biologically meaningful metrics rather than reconstruction alone. On this benchmark, our model improves PDCorr by 12.02% and DE Overlap by 10.66% over STATE. Together, these results suggest that advancing virtual cells requires not only better generative objectives, but also the co-design of scalable infrastructure, stable transport modeling, and biologically faithful evaluation. 

---
# Efficient Exploration at Scale 

**Authors**: Seyed Mohammad Asghari, Chris Chute, Vikranth Dwaracherla, Xiuyuan Lu, Mehdi Jafarnia, Victor Minden, Zheng Wen, Benjamin Van Roy  

**Link**: [PDF](https://arxiv.org/pdf/2603.17378)  

**Abstract**: We develop an online learning algorithm that dramatically improves the data efficiency of reinforcement learning from human feedback (RLHF). Our algorithm incrementally updates reward and language models as choice data is received. The reward model is fit to the choice data, while the language model is updated by a variation of reinforce, with reinforcement signals provided by the reward model. Several features enable the efficiency gains: a small affirmative nudge added to each reinforcement signal, an epistemic neural network that models reward uncertainty, and information-directed exploration. With Gemma large language models (LLMs), our algorithm matches the performance of offline RLHF trained on 200K labels using fewer than 20K labels, representing more than a 10x gain in data efficiency. Extrapolating from our results, we expect our algorithm trained on 1M labels to match offline RLHF trained on 1B labels. This represents a 1,000x gain. To our knowledge, these are the first results to demonstrate that such large improvements are possible. 

---
# Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift 

**Authors**: Zhihua Wei, Qiang Li, Jian Ruan, Zhenxin Qin, Leilei Wen, Dongrui Liu, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17372)  

**Abstract**: Large vision-language models (VLMs) often exhibit weakened safety alignment with the integration of the visual modality. Even when text prompts contain explicit harmful intent, adding an image can substantially increase jailbreak success rates. In this paper, we observe that VLMs can clearly distinguish benign inputs from harmful ones in their representation space. Moreover, even among harmful inputs, jailbreak samples form a distinct internal state that is separable from refusal samples. These observations suggest that jailbreaks do not arise from a failure to recognize harmful intent. Instead, the visual modality shifts representations toward a specific jailbreak state, thereby leading to a failure to trigger refusal. To quantify this transition, we identify a jailbreak direction and define the jailbreak-related shift as the component of the image-induced representation shift along this direction. Our analysis shows that the jailbreak-related shift reliably characterizes jailbreak behavior, providing a unified explanation for diverse jailbreak scenarios. Finally, we propose a defense method that enhances VLM safety by removing the jailbreak-related shift (JRS-Rem) at inference time. Experiments show that JRS-Rem provides strong defense across multiple scenarios while preserving performance on benign tasks. 

---
# Public Profile Matters: A Scalable Integrated Approach to Recommend Citations in the Wild 

**Authors**: Karan Goyal, Dikshant Kukreja, Vikram Goyal, Mukesh Mohania  

**Link**: [PDF](https://arxiv.org/pdf/2603.17361)  

**Abstract**: Proper citation of relevant literature is essential for contextualising and validating scientific contributions. While current citation recommendation systems leverage local and global textual information, they often overlook the nuances of the human citation behaviour. Recent methods that incorporate such patterns improve performance but incur high computational costs and introduce systematic biases into downstream rerankers. To address this, we propose Profiler, a lightweight, non-learnable module that captures human citation patterns efficiently and without bias, significantly enhancing candidate retrieval. Furthermore, we identify a critical limitation in current evaluation protocol: the systems are assessed in a transductive setting, which fails to reflect real-world scenarios. We introduce a rigorous Inductive evaluation setting that enforces strict temporal constraints, simulating the recommendation of citations for newly authored papers in the wild. Finally, we present DAVINCI, a novel reranking model that integrates profiler-derived confidence priors with semantic information via an adaptive vector-gating mechanism. Our system achieves new state-of-the-art results across multiple benchmark datasets, demonstrating superior efficiency and generalisability. 

---
# WebPII: Benchmarking Visual PII Detection for Computer-Use Agents 

**Authors**: Nathan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17357)  

**Abstract**: Computer use agents create new privacy risks: training data collected from real websites inevitably contains sensitive information, and cloud-hosted inference exposes user screenshots. Detecting personally identifiable information in web screenshots is critical for privacy-preserving deployment, but no public benchmark exists for this task. We introduce WebPII, a fine-grained synthetic benchmark of 44,865 annotated e-commerce UI images designed with three key properties: extended PII taxonomy including transaction-level identifiers that enable reidentification, anticipatory detection for partially-filled forms where users are actively entering data, and scalable generation through VLM-based UI reproduction. Experiments validate that these design choices improve layout-invariant detection across diverse interfaces and generalization to held-out page types. We train WebRedact to demonstrate practical utility, more than doubling text-extraction baseline accuracy (0.753 vs 0.357 mAP@50) at real-time CPU latency (20ms). We release the dataset and model to support privacy-preserving computer use research. 

---
# Learning Permutation Distributions via Reflected Diffusion on Ranks 

**Authors**: Sizhuang He, Yangtian Zhang, Shiyang Zhang, David van Dijk  

**Link**: [PDF](https://arxiv.org/pdf/2603.17353)  

**Abstract**: The finite symmetric group S_n provides a natural domain for permutations, yet learning probability distributions on S_n is challenging due to its factorially growing size and discrete, non-Euclidean structure. Recent permutation diffusion methods define forward noising via shuffle-based random walks (e.g., riffle shuffles) and learn reverse transitions with Plackett-Luce (PL) variants, but the resulting trajectories can be abrupt and increasingly hard to denoise as n grows. We propose Soft-Rank Diffusion, a discrete diffusion framework that replaces shuffle-based corruption with a structured soft-rank forward process: we lift permutations to a continuous latent representation of order by relaxing discrete ranks into soft ranks, yielding smoother and more tractable trajectories. For the reverse process, we introduce contextualized generalized Plackett-Luce (cGPL) denoisers that generalize prior PL-style parameterizations and improve expressivity for sequential decision structures. Experiments on sorting and combinatorial optimization benchmarks show that Soft-Rank Diffusion consistently outperforms prior diffusion baselines, with particularly strong gains in long-sequence and intrinsically sequential settings. 

---
# Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress 

**Authors**: Yuelin Zhang, Sijie Cheng, Chen Li, Zongzhao Li, Yuxin Huang, Yang Liu, Wenbing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17312)  

**Abstract**: Accurately estimating task progress is critical for embodied agents to plan and execute long-horizon, multi-step tasks. Despite promising advances, existing Vision-Language Models (VLMs) based methods primarily leverage their video understanding capabilities, while neglecting their complex reasoning potential. Furthermore, processing long video trajectories with VLMs is computationally prohibitive for real-world deployment. To address these challenges, we propose the Recurrent Reasoning Vision-Language Model ($\text{R}^2$VLM). Our model features a recurrent reasoning framework that processes local video snippets iteratively, maintaining a global context through an evolving Chain of Thought (CoT). This CoT explicitly records task decomposition, key steps, and their completion status, enabling the model to reason about complex temporal dependencies. This design avoids the high cost of processing long videos while preserving essential reasoning capabilities. We train $\text{R}^2$VLM on large-scale, automatically generated datasets from ALFRED and Ego4D. Extensive experiments on progress estimation and downstream applications, including progress-enhanced policy learning, reward modeling for reinforcement learning, and proactive assistance, demonstrate that $\text{R}^2$VLM achieves strong performance and generalization, achieving a new state-of-the-art in long-horizon task progress estimation. The models and benchmarks are publicly available at \href{this https URL}{huggingface}. 

---
# ReLMXEL: Adaptive RL-Based Memory Controller with Explainable Energy and Latency Optimization 

**Authors**: Panuganti Chirag Sai, Gandholi Sarat, R. Raghunatha Sarma, Venkata Kalyan Tavva, Naveen M  

**Link**: [PDF](https://arxiv.org/pdf/2603.17309)  

**Abstract**: Reducing latency and energy consumption is critical to improving the efficiency of memory systems in modern computing. This work introduces ReLMXEL (Reinforcement Learning for Memory Controller with Explainable Energy and Latency Optimization), a explainable multi-agent online reinforcement learning framework that dynamically optimizes memory controller parameters using reward decomposition. ReLMXEL operates within the memory controller, leveraging detailed memory behavior metrics to guide decision-making. Experimental evaluations across diverse workloads demonstrate consistent performance gains over baseline configurations, with refinements driven by workload-specific memory access behaviour. By incorporating explainability into the learning process, ReLMXEL not only enhances performance but also increases the transparency of control decisions, paving the way for more accountable and adaptive memory system designs. 

---
# Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding 

**Authors**: Haiyang Yan, Hongyun Zhou, Peng Xu, Xiaoxue Feng, Mengyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17307)  

**Abstract**: Despite rapid developments and widespread applications of MLLM agents, they still struggle with long-form video understanding (LVU) tasks, which are characterized by high information density and extended temporal spans. Recent research on LVU agents demonstrates that simple task decomposition and collaboration mechanisms are insufficient for long-chain reasoning tasks. Moreover, directly reducing the time context through embedding-based retrieval may lose key information of complex problems. In this paper, we propose Symphony, a multi-agent system, to alleviate these limitations. By emulating human cognition patterns, Symphony decomposes LVU into fine-grained subtasks and incorporates a deep reasoning collaboration mechanism enhanced by reflection, effectively improving the reasoning capability. Additionally, Symphony provides a VLM-based grounding approach to analyze LVU tasks and assess the relevance of video segments, which significantly enhances the ability to locate complex problems with implicit intentions and large temporal spans. Experimental results show that Symphony achieves state-of-the-art performance on LVBench, LongVideoBench, VideoMME, and MLVU, with a 5.0% improvement over the prior state-of-the-art method on LVBench. Code is available at this https URL. 

---
# From Words to Worlds: Benchmarking Cross-Cultural Cultural Understanding in Machine Translation 

**Authors**: Bangju Han, Yingqi Wang, Huang Qing, Tiyuan Li, Fengyi Yang, Ahtamjan Ahmat, Abibulla Atawulla, Yating Yang, Xi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.17303)  

**Abstract**: Culture-expressions, such as idioms, slang, and culture-specific items (CSIs), are pervasive in natural language and encode meanings that go beyond literal linguistic form. Accurately translating such expressions remains challenging for machine translation systems. Despite this, existing benchmarks remain fragmented and do not provide a systematic framework for evaluating translation performance on culture-loaded expressions. To address this gap, we introduce CulT-Eval, a benchmark designed to evaluate how models handle different types of culturally grounded expressions. CulT-Eval comprises over 7,959 carefully curated instances spanning multiple types of culturally grounded expressions, with a comprehensive error taxonomy covering culturally grounded expressions. Through extensive evaluation of large language models and detailed analysis, we identify recurring and systematic failure modes that are not adequately captured by existing automatic metrics. Accordingly, we propose a complementary evaluation metric that targets culturally induced meaning deviations overlooked by standard MT metrics. The results indicate that current models struggle to preserve culturally grounded meaning and to capture the cultural and contextual nuances essential for accurate translation. Our benchmark and code are available at this https URL. 

---
# GUIDE: GenAI Units In Digital Design Education 

**Authors**: Weihua Xiao, Jason Blocklove, Matthew DeLorenzo, Johann Knechtel, Ozgur Sinanoglu, Kanad Basu, Jeyavijayan Rajendran, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17296)  

**Abstract**: GenAI Units In Digital Design Education (GUIDE) is an open courseware repository with runnable Google Colab labs and other materials. We describe the repository's architecture and educational approach based on standardized teaching units comprising slides, short videos, runnable labs, and related papers. This organization enables consistency for both the students' learning experience and the reuse and grading by instructors. We demonstrate GUIDE in practice with three representative units: VeriThoughts for reasoning and formal-verification-backed RTL generation, enhanced LLM-aided testbench generation, and LLMPirate for IP Piracy. We also provide details for four example course instances (GUIDE4ChipDesign, Build your ASIC, GUIDE4HardwareSecurity, and Hardware Design) that assemble GUIDE units into full semester offerings, learning outcomes, and capstone projects, all based on proven materials. For example, the GUIDE4HardwareSecurity course includes a project on LLM-aided hardware Trojan insertion that has been successfully deployed in the classroom and in Cybersecurity Games and Conference (CSAW), a student competition and academic conference for cybersecurity. We also organized an NYU Cognichip Hackathon, engaging students across 24 international teams in AI-assisted RTL design workflows. The GUIDE repository is open for contributions and available at: this https URL. 

---
# Directing the Narrative: A Finetuning Method for Controlling Coherence and Style in Story Generation 

**Authors**: Jianzhang Zhang, Yijing Tian, Jiwang Qu, Chuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17295)  

**Abstract**: Story visualization requires generating sequential imagery that aligns semantically with evolving narratives while maintaining rigorous consistency in character identity and visual style. However, existing methodologies often struggle with subject inconsistency and identity drift, particularly when depicting complex interactions or extended narrative arcs. To address these challenges, we propose a cohesive two-stage framework designed for robust and consistent story generation. First, we introduce Group-Shared Attention (GSA), a mechanism that fosters intrinsic consistency by enabling lossless cross-sample information flow within attention layers. This allows the model to structurally encode identity correspondence across frames without relying on external encoders. Second, we leverage Direct Preference Optimization (DPO) to align generated outputs with human aesthetic and narrative standards. Unlike conventional methods that rely on conflicting auxiliary losses, our approach simultaneously enhances visual fidelity and identity preservation by learning from holistic preference data. Extensive evaluations on the ViStoryBench benchmark demonstrate that our method establishes a new state-of-the-art, significantly outperforming strong baselines with gains of +10.0 in Character Identity (CIDS) and +18.7 in Style Consistency (CSD), all while preserving high-fidelity generation. 

---
# DANCE: Dynamic 3D CNN Pruning: Joint Frame, Channel, and Feature Adaptation for Energy Efficiency on the Edge 

**Authors**: Mohamed Mejri, Ashiqur Rasul, Abhijit Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.17275)  

**Abstract**: Modern convolutional neural networks (CNNs) are workhorses for video and image processing, but fail to adapt to the computational complexity of input samples in a dynamic manner to minimize energy consumption. In this research, we propose DANCE, a fine-grained, input-aware, dynamic pruning framework for 3D CNNs to maximize power efficiency with negligible to zero impact on performance. In the proposed two-step approach, the first step is called activation variability amplification (AVA), and the 3D CNN model is retrained to increase the variance of the magnitude of neuron activations across the network in this step, facilitating pruning decisions across diverse CNN input scenarios. In the second step, called adaptive activation pruning (AAP), a lightweight activation controller network is trained to dynamically prune frames, channels, and features of 3D convolutional layers of the network (different for each layer), based on statistics of the outputs of the first layer of the network. Our method achieves substantial savings in multiply-accumulate (MAC) operations and memory accesses by introducing sparsity within convolutional layers. Hardware validation on the NVIDIA Jetson Nano GPU and the Qualcomm Snapdragon 8 Gen 1 platform demonstrates respective speedups of 1.37X and 2.22X, achieving up to 1.47X higher energy efficiency compared to the state of the art. 

---
# Pathology-Aware Multi-View Contrastive Learning for Patient-Independent ECG Reconstruction 

**Authors**: Youssef Youssef, Jitin Singla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17248)  

**Abstract**: Reconstructing a 12-lead electrocardiogram (ECG) from a reduced lead set is an ill-posed inverse problem due to anatomical variability. Standard deep learning methods often ignore underlying cardiac pathology losing vital morphology in precordial leads. We propose Pathology-Aware Multi-View Contrastive Learning, a framework that regularizes the latent space through a pathological manifold. Our architecture integrates high-fidelity time-domain waveforms with pathology-aware embeddings learned via supervised contrastive alignment. By maximizing mutual information between latent representations and clinical labels, the framework learns to filter anatomical "nuisance" variables. On the PTB-XL dataset, our method achieves approx. 76\% reduction in RMSE compared to state-of-the-art model in patient-independent setting. Cross-dataset evaluation on the PTB Diagnostic Database confirms superior generalization, bridging the gap between hardware portability and diagnostic-grade reconstruction. 

---
# Deployment and Evaluation of an EHR-integrated, Large Language Model-Powered Tool to Triage Surgical Patients 

**Authors**: Jane Wang, Timothy Keyes, April S Liang, Stephen P Ma, Jason Shen, Jerry Liu, Nerissa Ambers, Abby Pandya, Rita Pandya, Jason Hom, Natasha Steele, Jonathan H Chen, Kevin Schulman  

**Link**: [PDF](https://arxiv.org/pdf/2603.17234)  

**Abstract**: Surgical co-management (SCM) is an evidence-based model in which hospitalists jointly manage medically complex perioperative patients alongside surgical teams. Despite its clinical and financial value, SCM is limited by the need to manually identify eligible patients. To determine whether SCM triage can be automated, we conducted a prospective, unblinded study at Stanford Health Care in which an LLM-based, electronic health record (EHR)-integrated triage tool (SCM Navigator) provided SCM recommendations followed by physician review.
Using pre-operative documentation, structured data, and clinical criteria for perioperative morbidity, SCM Navigator categorized patients as appropriate, not appropriate, or possibly appropriate for SCM. Faculty indicated their clinical judgment and provided free-text feedback when they disagreed. Sensitivity, specificity, positive predictive value, and negative predictive value were measured using physician determinations as a reference. Free-text reasons were thematically categorized, and manual chart review was conducted on all false-negative cases and 30 randomly selected cases from the largest false-positive category. Since deployment, 6,193 cases have been triaged, of which 1,582 (23%) were recommended for hospitalist consultation. SCM Navigator displayed high sensitivity (0.94, 95% CI 0.91-0.96) and moderate specificity (0.74, 95% CI 0.71-0.77). Post-hoc chart review suggested most discrepancies reflect modifiable gaps in clinical criteria, institutional workflow, or physician practice variability rather than LLM misclassification, which accounted for 2 of 19 (11%) false-negative cases. These findings demonstrate that an LLM-powered, EHR-integrated, human-in-the-loop AI system can accurately and safely triage surgical patients for SCM, and that AI-enabled screening tools can augment and potentially automate time-intensive clinical workflows. 

---
# KANtize: Exploring Low-bit Quantization of Kolmogorov-Arnold Networks for Efficient Inference 

**Authors**: Sohaib Errabii, Olivier Sentieys, Marcello Traiola  

**Link**: [PDF](https://arxiv.org/pdf/2603.17230)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have gained attention for their potential to outperform Multi-Layer Perceptrons (MLPs) in terms of parameter efficiency and interpretability. Unlike traditional MLPs, KANs use learnable non-linear activation functions, typically spline functions, expressed as linear combinations of basis splines (B-splines). B-spline coefficients serve as the model's learnable parameters. However, evaluating these spline functions increases computational complexity during inference. Conventional quantization reduces this complexity by lowering the numerical precision of parameters and activations. However, the impact of quantization on KANs, and especially its effectiveness in reducing computational complexity, is largely unexplored, particularly for quantization levels below 8 bits. The study investigates the impact of low-bit quantization on KANs and its impact on computational complexity and hardware efficiency. Results show that B-splines can be quantized to 2-3 bits with negligible loss in accuracy, significantly reducing computational complexity. Hence, we investigate the potential of using low-bit quantized precomputed tables as a replacement for the recursive B-spline algorithm. This approach aims to further reduce the computational complexity of KANs and enhance hardware efficiency while maintaining accuracy. For example, ResKAN18 achieves a 50x reduction in BitOps without loss of accuracy using low-bit-quantized B-spline tables. Additionally, precomputed 8-bit lookup tables improve GPU inference speedup by up to 2.9x, while on FPGA-based systolic-array accelerators, reducing B-spline table precision from 8 to 3 bits cuts resource usage by 36%, increases clock frequency by 50%, and enhances speedup by 1.24x. On a 28nm FD-SOI ASIC, reducing the B-spline bit-width from 16 to 3 bits achieves 72% area reduction and 50% higher maximum frequency. 

---
# From Drop-off to Recovery: A Mechanistic Analysis of Segmentation in MLLMs 

**Authors**: Boyong Wu, Sanghwan Kim, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2603.17228)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly applied to pixel-level vision tasks, yet their intrinsic capacity for spatial understanding remains poorly understood. We investigate segmentation capacity through a layerwise linear probing evaluation across the entire MLLM pipeline: vision encoder, adapter, and LLM. We further conduct an intervention based attention knockout analysis to test whether cross-token attention progressively refines visual representations, and an evaluation of bidirectional attention among image tokens on spatial consistency. Our analysis reveals that the adapter introduces a segmentation representation drop-off, but LLM layers progressively recover through attention-mediated refinement, where correctly classified tokens steer misclassified neighbors toward the correct label. At early image token positions, this recovery is bounded by causal attention, which bidirectional attention among image tokens alleviates. These findings provide a mechanistic account of how MLLMs process visual information for segmentation, informing the design of future segmentation-capable models. 

---
# TharuChat: Bootstrapping Large Language Models for a Low-Resource Language via Synthetic Data and Human Validation 

**Authors**: Prajwal Panth, Agniva Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2603.17220)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has created a profound digital divide, effectively excluding indigenous languages of the Global South from the AI revolution. The Tharu language, an Indo-Aryan vernacular spoken by approximately 1.7 million people across the Terai belt of Nepal and India, exemplifies this crisis. Despite a rich oral tradition, Tharu suffers from severe data scarcity and linguistic fragmentation, causing state-of-the-art multilingual models to routinely "hallucinate" or default to dominant high-resource neighbors like Hindi and Nepali due to contamination in pre-training corpora.
This paper presents Tharu-LLaMA (3B), a specialized instruction-following model designed to address this exclusion. We introduce TharuChat, a novel dataset constructed via a LLM-to-Human bootstrapping pipeline. We utilized prompt-engineered Gemini models, fed with Rana Tharu grammar and folklore, to synthesize training data. Unlike curated gold-standard corpora, TharuChat reflects the noisy, heterogeneous linguistic reality of the region: it is predominantly anchored in Rana Tharu (~70%) while integrating elements of Dangaura and Kochila dialects. We provide a transparent analysis of the dataset's limitations, including dialectal code-mixing and residual Awadhi/Hindi influence. Through a rigorous empirical ablation study, we demonstrate that despite these imperfections, small-scale synthetic data is highly effective, increasing the dataset volume from 25% to 100% results in a linear reduction in perplexity from 6.42 to 2.88. The resulting model serves as a proof-of-concept for the preservation of under-resourced Himalayan languages via generative AI, achievable on consumer-grade hardware. 

---
# SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization 

**Authors**: Ishrith Gowda, Chunwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17219)  

**Abstract**: Multi-site neuroimaging analysis is fundamentally confounded by scanner-induced covariate shifts, where the marginal distribution of voxel intensities $P(\mathbf{x})$ varies non-linearly across acquisition protocols while the conditional anatomy $P(\mathbf{y}|\mathbf{x})$ remains constant. This is particularly detrimental to radiomic reproducibility, where acquisition variance often exceeds biological pathology variance. Existing statistical harmonization methods (e.g., ComBat) operate in feature space, precluding spatial downstream tasks, while standard deep learning approaches are theoretically bounded by local effective receptive fields (ERF), failing to model the global intensity correlations characteristic of field-strength bias.
We propose SA-CycleGAN-2.5D, a domain adaptation framework motivated by the $H\Delta H$-divergence bound of Ben-David et al., integrating three architectural innovations: (1) A 2.5D tri-planar manifold injection preserving through-plane gradients $\nabla_z$ at $O(HW)$ complexity; (2) A U-ResNet generator with dense voxel-to-voxel self-attention, surpassing the $O(\sqrt{L})$ receptive field limit of CNNs to model global scanner field biases; and (3) A spectrally-normalized discriminator constraining the Lipschitz constant ($K_D \le 1$) for stable adversarial optimization. Evaluated on 654 glioma patients across two institutional domains (BraTS and UPenn-GBM), our method reduces Maximum Mean Discrepancy (MMD) by 99.1% ($1.729 \to 0.015$) and degrades domain classifier accuracy to near-chance (59.7%). Ablation confirms that global attention is statistically essential (Cohen's $d = 1.32$, $p < 0.001$) for the harder heterogeneous-to-homogeneous translation direction. By bridging 2D efficiency and 3D consistency, our framework yields voxel-level harmonized images that preserve tumor pathophysiology, enabling reproducible multi-center radiomic analysis. 

---
# Alignment Makes Language Models Normative, Not Descriptive 

**Authors**: Eilam Shapira, Moshe Tennenholtz, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2603.17218)  

**Abstract**: Post-training alignment optimizes language models to match human preference signals, but this objective is not equivalent to modeling observed human behavior. We compare 120 base-aligned model pairs on more than 10,000 real human decisions in multi-round strategic games - bargaining, persuasion, negotiation, and repeated matrix games. In these settings, base models outperform their aligned counterparts in predicting human choices by nearly 10:1, robustly across model families, prompt formulations, and game configurations. This pattern reverses, however, in settings where human behavior is more likely to follow normative predictions: aligned models dominate on one-shot textbook games across all 12 types tested and on non-strategic lottery choices - and even within the multi-round games themselves, at round one, before interaction history develops. This boundary-condition pattern suggests that alignment induces a normative bias: it improves prediction when human behavior is relatively well captured by normative solutions, but hurts prediction in multi-round strategic settings, where behavior is shaped by descriptive dynamics such as reciprocity, retaliation, and history-dependent adaptation. These results reveal a fundamental trade-off between optimizing models for human use and using them as proxies for human behavior. 

---
# Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text 

**Authors**: Federico Albanese, Pablo Ronco, Nicolás D'Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2603.17217)  

**Abstract**: Responsible use of AI demands that we protect sensitive information without undermining the usefulness of data, an imperative that has become acute in the age of large language models. We address this challenge with an on-premise, LLM-driven substitution pipeline that anonymizes text by replacing personally identifiable information (PII) with realistic, type-consistent surrogates. Executed entirely within organizational boundaries using local LLMs, the approach prevents data egress while preserving fluency and task-relevant semantics.
We conduct a systematic, multi-metric, cross-technique evaluation on the Action-Based Conversation Dataset, benchmarking against industry standards (Microsoft Presidio and Google DLP) and a state-of-the-art approach (ZSTS, in redaction-only and redaction-plus-substitution variants). Our protocol jointly measures privacy, semantic utility, and trainability under privacy via a lifecycle-ready criterion obtained by fine-tuning a compact encoder (BERT+LoRA) on sanitized text. In addition, we assess agentic Q&A performance by inserting an on-premise anonymization layer before the answering LLM and evaluating the quality of its responses. This intermediate, type-preserving substitution stage ensures that no sensitive content is exposed to third-party APIs, enabling responsible deployment of Q\&A agents without compromising confidentiality.
Our method attains state-of-the-art privacy, minimal topical drift, strong factual utility, and low trainability loss, outperforming rule-based approaches and named-entity recognition (NER) baselines and ZSTS variants on the combined privacy--utility--trainability frontier. These results show that local LLM substitution yields anonymized corpora that are both responsible to use and operationally valuable: safe for agentic pipelines and suitable for downstream fine-tuning with limited degradation. 

---
# Adaptive Contracts for Cost-Effective AI Delegation 

**Authors**: Eden Saig, Tamar Garbuz, Ariel D. Procaccia, Inbal Talgam-Cohen, Jamie Tucker-Foltz  

**Link**: [PDF](https://arxiv.org/pdf/2603.17212)  

**Abstract**: When organizations delegate text generation tasks to AI providers via pay-for-performance contracts, expected payments rise when evaluation is noisy. As evaluation methods become more elaborate, the economic benefits of decreased noise are often overshadowed by increased evaluation costs. In this work, we introduce adaptive contracts for AI delegation, which allow detailed evaluation to be performed selectively after observing an initial coarse signal in order to conserve resources. We make three sets of contributions: First, we provide efficient algorithms for computing optimal adaptive contracts under natural assumptions or when core problem dimensions are small, and prove hardness of approximation in the general unstructured case. We then formulate alternative models of randomized adaptive contracts and discuss their benefits and limitations. Finally, we empirically demonstrate the benefits of adaptivity over non-adaptive baselines using question-answering and code-generation datasets. 

---
# A scalable neural bundle map for multiphysics prediction in lithium-ion battery across varying configurations 

**Authors**: Zhiwei Zhao, Changqing Liu, Jie Lin, Fan Yang, Yifan Zhang, Yan Jin, Yingguang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.17209)  

**Abstract**: Efficient and accurate prediction of Multiphysics evolution across diverse cell geometries is fundamental to the design, management and safety of lithium-ion batteries. However, existing computational frameworks struggle to capture the coupled electrochemical, thermal, and mechanical dynamics across diverse cell geometries and varying operating conditions. Here, we present a Neural Bundle Map (NBM), a mathematically rigorous framework that reformulates multiphysics evolution as a bundle map over a geometric base manifold. This approach enables the complete decoupling of geometric complexity from underlying physical laws, ensuring strong operator continuity across varying domains. Our framework achieves high-fidelity spatiotemporal predictions with a normalized mean absolute error of less than 1% across varying configurations, while maintaining stability during long-horizon forecasting far beyond the training window and reducing computational costs by two orders of magnitude compared with conventional solvers. Leveraging this capability, we rapidly explored a vast configurational space to identify an optimal battery design that yields a 38% increase in energy density while adhering to thermal safety constraints. Furthermore, the NBM demonstrates remarkable scalability to multi-cell systems through few-shot transfer learning, providing a foundational paradigm for the intelligent design and real-time monitoring of complex energy storage infrastructures. 

---
# OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation 

**Authors**: Haoyang Fang, Shuai Zhang, Yifei Ma, Hengyi Wang, Cuixiong Hu, Katrin Kirchhoff, Bernie Wang, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2603.17205)  

**Abstract**: Domain-specific finetuning is essential for dense retrievers, yet not all training pairs contribute equally to the learning process. We introduce OPERA, a data pruning framework that exploits this heterogeneity to improve both the effectiveness and efficiency of retrieval model adaptation. We first investigate static pruning (SP), which retains only high-similarity query-document pairs, revealing an intrinsic quality-coverage tradeoff: ranking (NDCG) improves while retrieval (Recall) can degrade due to reduced query diversity. To resolve this tradeoff, we propose a two-stage dynamic pruning (DP) strategy that adaptively modulates sampling probabilities at both query and document levels throughout training, prioritizing high-quality examples while maintaining access to the full training set. Evaluations across eight datasets spanning six domains demonstrate the effectiveness of both approaches: SP improves ranking over standard finetuning (NDCG@10 +0.5\%), while DP achieves the strongest performance on both ranking (NDCG@10 +1.9\%) and retrieval (Recall@20 +0.7\%), with an average rank of 1.38 across all methods. These findings scale to Qwen3-Embedding, an LLM-based dense retriever, confirming architecture-agnostic benefits. Notably, DP reaches comparable performance in less than 50\% of the training time required by standard finetuning. 

---
# Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing 

**Authors**: Parsa Mirtaheri, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2603.17199)  

**Abstract**: Large language models (LLMs) can produce chains of thought (CoT) that do not accurately reflect the actual factors driving their answers. In multiple-choice settings with an injected hint favoring a particular option, models may shift their final answer toward the hinted option and produce a CoT that rationalizes the response without acknowledging the hint - an instance of motivated reasoning. We study this phenomenon across multiple LLM families and datasets demonstrating that motivated reasoning can be identified by probing internal activations even in cases when it cannot be easily determined from CoT. Using supervised probes trained on the model's residual stream, we show that (i) pre-generation probes, applied before any CoT tokens are generated, predict motivated reasoning as well as a LLM-based CoT monitor that accesses the full CoT trace, and (ii) post-generation probes, applied after CoT generation, outperform the same monitor. Together, these results show that motivated reasoning is detected more reliably from internal representations than from CoT monitoring. Moreover, pre-generation probing can flag motivated behavior early, potentially avoiding unnecessary generation. 

---
# Towards Unsupervised Adversarial Document Detection in Retrieval Augmented Generation Systems 

**Authors**: Patrick Levi  

**Link**: [PDF](https://arxiv.org/pdf/2603.17176)  

**Abstract**: Retrieval augmented generation systems have become an integral part of everyday life. Whether in internet search engines, email systems, or service chatbots, these systems are based on context retrieval and answer generation with large language models. With their spread, also the security vulnerabilities increase. Attackers become increasingly focused on these systems and various hacking approaches are developed. Manipulating the context documents is a way to persist attacks and make them affect all users. Therefore, detecting compromised, adversarial context documents early is crucial for security. While supervised approaches require a large amount of labeled adversarial contexts, we propose an unsupervised approach, being able to detect also zero day attacks. We conduct a preliminary study to show appropriate indicators for adversarial contexts. For that purpose generator activations, output embeddings, and an entropy-based uncertainty measure turn out as suitable, complementary quantities. With an elementary statistical outlier detection, we propose and compare their detection abilities. Furthermore, we show that the target prompt, which the attacker wants to manipulate, is not required for a successful detection. Moreover, our results indicate that a simple context summary generation might even be superior in finding manipulated contexts. 

---
# Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning 

**Authors**: Shenao Yan, Shimaa Ahmed, Shan Jin, Sunpreet S. Arora, Yiwei Cai, Yizhen Wang, Yuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2603.17174)  

**Abstract**: Code generation large language models (LLMs) are increasingly integrated into modern software development workflows. Recent work has shown that these models are vulnerable to backdoor and poisoning attacks that induce the generation of insecure code, yet effective defenses remain limited. Existing scanning approaches rely on token-level generation consistency to invert attack targets, which is ineffective for source code where identical semantics can appear in diverse syntactic forms. We present CodeScan, which, to the best of our knowledge, is the first poisoning-scanning framework tailored to code generation models. CodeScan identifies attack targets by analyzing structural similarities across multiple generations conditioned on different clean prompts. It combines iterative divergence analysis with abstract syntax tree (AST)-based normalization to abstract away surface-level variation and unify semantically equivalent code, isolating structures that recur consistently across generations. CodeScan then applies LLM-based vulnerability analysis to determine whether the extracted structures contain security vulnerabilities and flags the model as compromised when such a structure is found. We evaluate CodeScan against four representative attacks under both backdoor and poisoning settings across three real-world vulnerability classes. Experiments on 108 models spanning three architectures and multiple model sizes demonstrate 97%+ detection accuracy with substantially lower false positives than prior methods. 

---
# Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience 

**Authors**: Jacob Piland, Byron Dowling, Christopher Sweet, Adam Czajka  

**Link**: [PDF](https://arxiv.org/pdf/2603.17173)  

**Abstract**: Iris presentation attack detection (PAD) is critical for secure biometric deployments, yet developing specialized models faces significant practical barriers: collecting data representing future unknown attacks is impossible, and collecting diverse-enough data, yet still limited in terms of its predictive power, is expensive. Additionally, sharing biometric data raises privacy concerns. Due to rapid emergence of new attack vectors demanding adaptable solutions, we thus investigate in this paper whether general-purpose multimodal large language models (MLLMs) can perform iris PAD when augmented with human expert knowledge, operating under strict privacy constraints that prohibit sending biometric data to public cloud MLLM services. Through analysis of vision encoder embeddings applied to our dataset, we demonstrate that pre-trained vision transformers in MLLMs inherently cluster many iris attack types despite never being explicitly trained for this task. However, where clustering shows overlap between attack classes, we find that structured prompts incorporating human salience (verbal descriptions from subjects identifying attack indicators) enable these models to resolve ambiguities. Testing on an IRB-restricted dataset of 224 iris images spanning seven attack types, using only university-approved services (Gemini 2.5 Pro) or locally-hosted models (e.g., Llama 3.2-Vision), we show that Gemini with expert-informed prompts outperforms both a specialized convolutional neural networks (CNN)-based baseline and human examiners, while the locally-deployable Llama achieves near-human performance. Our results establish that MLLMs deployable within institutional privacy constraints offer a viable path for iris PAD. 

---
# PAuth - Precise Task-Scoped Authorization For Agents 

**Authors**: Reshabh K Sharma, Linxi Jiang, Zhiqiang Lin, Shuo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17170)  

**Abstract**: The emerging agentic web envisions AI agents that reliably fulfill users' natural-language (NL)-based tasks by interacting with existing web services. However, existing authorization models are misaligned with this vision. In particular, today's operator-scoped authorization, exemplified by OAuth, grants broad permissions tied to operators (e.g., the transfer operator) rather than to the specific operations (e.g., transfer $100 to Bob) implied by a user's task. This will inevitably result in overprivileged agents.
We introduce Precise Task-Scoped Implicit Authorization (PAuth), a fundamentally different model in which submitting an NL task implicitly authorizes only the concrete operations required for its faithful execution. To make this enforceable at servers, we propose NL slices: symbolic specifications of the calls each service expects, derived from the task and upstream results. Complementing this, we also propose envelopes: special data structure to bind each operand's concrete value to its symbolic provenance, enabling servers to verify that all operands arise from legitimate computations.
PAuth is prototyped in the agent-security evaluation framework AgentDojo. We evaluate it in both benign settings and attack scenarios where a spurious operation is injected into an otherwise normal task. In all benign tests, PAuth executes the tasks successfully without requiring any additional permissions. In all attack tests, PAuth correctly raises warnings about missing permissions. These results demonstrate that PAuth's reasoning about permissions is indeed precise. We further analyze the characteristics of these tasks and measure the associated token costs. 

---
# Intent Formalization: A Grand Challenge for Reliable Coding in the Age of AI Agents 

**Authors**: Shuvendu K. Lahiri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17150)  

**Abstract**: Agentic AI systems can now generate code with remarkable fluency, but a fundamental question remains: \emph{does the generated code actually do what the user intended?} The gap between informal natural language requirements and precise program behavior -- the \emph{intent gap} -- has always plagued software engineering, but AI-generated code amplifies it to an unprecedented scale. This article argues that \textbf{intent formalization} -- the translation of informal user intent into a set of checkable formal specifications -- is the key challenge that will determine whether AI makes software more reliable or merely more abundant. Intent formalization offers a tradeoff spectrum suitable to the reliability needs of different contexts: from lightweight tests that disambiguate likely misinterpretations, through full functional specifications for formal verification, to domain-specific languages from which correct code is synthesized automatically. The central bottleneck is \emph{validating specifications}: since there is no oracle for specification correctness other than the user, we need semi-automated metrics that can assess specification quality with or without code, through lightweight user interaction and proxy artifacts such as tests. We survey early research that demonstrates the \emph{potential} of this approach: interactive test-driven formalization that improves program correctness, AI-generated postconditions that catch real-world bugs missed by prior methods, and end-to-end verified pipelines that produce provably correct code from informal specifications. We outline the open research challenges -- scaling beyond benchmarks, achieving compositionality over changes, metrics for validating specifications, handling rich logics, designing human-AI specification interactions -- that define a research agenda spanning AI, programming languages, formal methods, and human-computer interaction. 

---
# REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge 

**Authors**: Yasi Zhang, Tianyu Chen, Mingyuan Zhou, Oscar Leong, Ying Nian Wu, Michal Lukasik  

**Link**: [PDF](https://arxiv.org/pdf/2603.17145)  

**Abstract**: Large language models (LLMs) are increasingly deployed as automated evaluators that assign numeric scores to model outputs, a paradigm known as LLM-as-a-Judge. However, standard Reinforcement Learning (RL) methods typically rely on binary rewards (e.g., 0-1 accuracy), thereby ignoring the ordinal structure inherent in regression tasks; for instance, they fail to recognize that predicting 4 is significantly better than predicting 1 when the ground truth is 5. Conversely, existing regression-aware approaches are often confined to Supervised Fine-Tuning (SFT), limiting their ability to explore optimal reasoning paths. To bridge this gap, we propose \textbf{REAL} (\underline{RE}gression-\underline{A}ware Reinforcement \underline{L}earning), a principled RL framework designed to optimize regression rewards, and also proven to be optimal for correlation metrics. A key technical challenge is that the regression objective is explicitly policy-dependent, thus invalidating standard policy gradient methods. To address this, we employ the generalized policy gradient estimator, which naturally decomposes optimization into two complementary components: (1) exploration over Chain-of-Thought (CoT) trajectory, and (2) regression-aware prediction refinement of the final score. Extensive experiments across model scales (8B to 32B) demonstrate that REAL consistently outperforms both regression-aware SFT baselines and standard RL methods, exhibiting significantly better generalization on out-of-domain benchmarks. On Qwen3-32B specifically, we achieve gains of +8.40 Pearson and +7.20 Spearman correlation over the SFT baseline, and +18.30/+11.20 over the base model. These findings highlight the critical value of integrating regression objectives into RL exploration for accurate LLM evaluation. 

---
# Security Assessment and Mitigation Strategies for Large Language Models: A Comprehensive Defensive Framework 

**Authors**: Taiwo Onitiju, Iman Vakilinia  

**Link**: [PDF](https://arxiv.org/pdf/2603.17123)  

**Abstract**: Large Language Models increasingly power critical infrastructure from healthcare to finance, yet their vulnerability to adversarial manipulation threatens system integrity and user safety. Despite growing deployment, no comprehensive comparative security assessment exists across major LLM architectures, leaving organizations unable to quantify risk or select appropriately secure LLMs for sensitive applications. This research addresses this gap by establishing a standardized vulnerability assessment framework and developing a multi-layered defensive system to protect against identified threats. We systematically evaluate five widely-deployed LLM families GPT-4, GPT-3.5 Turbo, Claude-3 Haiku, LLaMA-2-70B, and Gemini-2.5-pro against 10,000 adversarial prompts spanning six attack categories. Our assessment reveals critical security disparities, with vulnerability rates ranging from 11.9\% to 29.8\%, demonstrating that LLM capability does not correlate with security robustness. To mitigate these risks, we develop a production-ready defensive framework achieving 83\% average detection accuracy with only 5\% false positives. These results demonstrate that systematic security assessment combined with external defensive measures provides a viable path toward safer LLM deployment in production environments. 

---
# Hidden Clones: Exposing and Fixing Family Bias in Vision-Language Model Ensembles 

**Authors**: Zacharie Bugaud  

**Link**: [PDF](https://arxiv.org/pdf/2603.17111)  

**Abstract**: Ensembling Vision-Language Models (VLMs) from different providers maximizes benchmark accuracy, yet models from the same architectural family share correlated errors that standard voting ignores. We study this structure across 17 VLMs from 8 families on VQAv2, TextVQA, and GQA. Family-correlated errors reduce effective ensemble dimensionality to 2.5-3.6 independent voters and create a Misleading tier (1.5-6.5% of questions) where correlated majority errors destroy accuracy to 0% despite the best model being correct.
We propose three family-aware methods. Hierarchical Family Voting (HFV) aggregates within families before voting across them, recovering +18-26 pp on the Misleading tier. QualRCCV, a training-free method weighting models by calibration, family quality, and inverse family size, is the first to beat calibrated voting on all three benchmarks (p<0.05). Learned Candidate Scoring (LCS) trains a cross-validated classifier to re-rank candidate answers using support breadth, family diversity, and model quality, achieving the largest gains: +0.68% VQAv2, +0.61% TextVQA, +2.45% GQA -- all significant -- and is the only learned method that never degrades any benchmark. On VQAv2 test-standard (EvalAI), LCS reaches 87.83% with 12 models, confirming generalization. 

---
# When the Specification Emerges: Benchmarking Faithfulness Loss in Long-Horizon Coding Agents 

**Authors**: Lu Yan, Xuan Chen, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17104)  

**Abstract**: Current coding-agent benchmarks usually pro- vide the full task specification upfront. Real research coding often does not: the intended system is progressively disclosed through in- teraction, requiring the agent to track durable design commitments across a long session. We introduce a benchmark for this setting and study faithfulne Ss Loss U nder eM ergent s Pecification (SLUMP), defined as the reduc- tion in final implementation faithfulness un- der emergent specification relative to a single- shot specification control. The benchmark con- tains 20 recent ML papers (10 ICML 2025, 10 NeurIPS 2025), 371 atomic verifiable compo- nents, and interaction scripts of approximately 60 coding requests that progressively disclose the target design without revealing the paper itself. Final repositories are scored with a five-level component-faithfulness rubric and accompanied by an exposure audit to verify that scored components are recoverable from the visible interaction. Evaluated on Claude Code and Codex, the single-shot specification control achieves higher overall implementation fidelity on 16/20 and 14/20 papers, respectively. Structural integration degrades under emergent specification on both platforms, while seman- tic faithfulness loss is substantial on Claude Code and small on Codex. As a mitigation case study, we introduce ProjectGuard, an exter- nal project-state layer for specification tracking. On Claude Code, ProjectGuard recovers 90% of the faithfulness gap, increases fully faith- ful components from 118 to 181, and reduces severe failures from 72 to 49. These results identify specification tracking as a distinct eval- uation target for long-horizon coding agents. 

---
# Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17102)  

**Abstract**: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the necessity of this small number of experts for answering a knowledge question, we deactivate them and re-ask the question. We find that despite only deactivating about 20 out of 6000 experts, the model no longer answers correctly in over 40% of cases. Generally, this method provides a realistic and scalable knowledge localization approach to address increasingly complex LLMs. 

---
# CircuitBuilder: From Polynomials to Circuits via Reinforcement Learning 

**Authors**: Weikun K. Zhang, Rohan Pandey, Bhaumik Mehta, Kaijie Jin, Naomi Morato, Archit Ganapule, Michael Ruofan Zeng, Jarod Alper  

**Link**: [PDF](https://arxiv.org/pdf/2603.17075)  

**Abstract**: Motivated by auto-proof generation and Valiant's VP vs. VNP conjecture, we study the problem of discovering efficient arithmetic circuits to compute polynomials, using addition and multiplication gates. We formulate this problem as a single-player game, where an RL agent attempts to build the circuit within a fixed number of operations. We implement an AlphaZero-style training loop and compare two approaches: Proximal Policy Optimization with Monte Carlo Tree Search (PPO+MCTS) and Soft Actor-Critic (SAC). SAC achieves the highest success rates on two-variable targets, while PPO+MCTS scales to three variables and demonstrates steady improvement on harder instances. These results suggest that polynomial circuit synthesis is a compact, verifiable setting for studying self-improving search policies. 

---
# Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17070)  

**Abstract**: In this work, we analyze shortcomings in cross-lingual knowledge transfer in large, modern reasoning LLMs. We demonstrate that the perceived gap in knowledge transfer is primarily a script barrier. First, we conduct an observational data analysis on the performance of thinking models on two datasets with local knowledge from around the world, ECLeKTic and MultiLoKo. Our regression analysis shows that script match - not language or family - is the primary predictor of knowledge transfer failure once model capability and question difficulty are accounted for. We further this finding by providing the LLMs with the key entities of the questions in their source language and find that this disproportionately improves cross-script questions. We then posit that these LLMs could be reasoning better at test-time. To evaluate this, we develop a synthetic generation pipeline to design SFT samples to encourage the model to better reason about transliteration ambiguities when trying to fetch parametric knowledge at inference-time. We show that teaching two models to reason better reduces the cross-script transfer gap. As a result, we conclude that there is potential to improve cross-lingual parametric knowledge transfer during post-training. 

---
# Evaluating Ill-Defined Tasks in Large Language Models 

**Authors**: Yi Zhou, Basel Shbita  

**Link**: [PDF](https://arxiv.org/pdf/2603.17067)  

**Abstract**: Many evaluations of Large Language Models (LLMs) target tasks that are inherently ill-defined, with unclear input and output spaces and ambiguous success criteria. We analyze why existing evaluation benchmarks and metrics fail to provide reliable or diagnostic signals of model capability for such tasks. We examine two case studies: Complex Instruction Following (CIF), where we identify recurring issues including limited coverage of real-world instruction complexity, sensitivity to instruction phrasing, inconsistent and non-comparable metrics, and instability introduced by LLM-based judges; and Natural Language to Mermaid Sequence Diagrams (NL2Mermaid), where we show how multi-faceted evaluation criteria can yield actionable insights beyond aggregate scores. Together, these case studies show that current evaluations frequently conflate distinct failure modes, yielding scores that are unstable, non-diagnostic, and difficult to act upon. Our findings expose fundamental limitations in existing evaluation practices for ill-defined tasks and motivate more robust, interpretable evaluation designs. 

---
# Early Quantization Shrinks Codebook: A Simple Fix for Diversity-Preserving Tokenization 

**Authors**: Wenhao Zhao, Qiran Zou, Rushi Shah, Yudi Wu, Zhouhan Lin, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17052)  

**Abstract**: Vector quantization is a technique in machine learning that discretizes continuous representations into a set of discrete vectors. It is widely employed in tokenizing data representations for large language models, diffusion models, and other generative models. Despite its prevalence, the characteristics and behaviors of vector quantization in generative models remain largely underexplored. In this study, we systematically investigate the issue of collapses in vector quantization, where collapsed representations are observed across discrete codebook tokens and continuous latent embeddings. By leveraging both synthetic and real datasets, we identify the severity of each type of collapses and triggering conditions. Our analysis reveals that random initialization and limited encoder capacity result in tokens collapse and embeddings collapse. Building on these findings, we propose potential solutions aimed at mitigating each collapse. To the best of our knowledge, this is the first comprehensive study examining representation collapsing problems in vector quantization. 

---
# Do Understanding and Generation Fight? A Diagnostic Study of DPO for Unified Multimodal Models 

**Authors**: Abinav Rao, Sujan Rachuri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17044)  

**Abstract**: Unified multimodal models share a language model backbone for both understanding and generating images. Can DPO align both capabilities simultaneously? We present the first systematic study of this question, applying DPO to Janus-Pro at 1B and 7B parameters under seven training strategies and two post-hoc methods. The central finding is negative: generation quality resists DPO alignment across all tested conditions on this architecture. No method improves generation CLIPScore at 7B (|Delta| < 0.2, p > 0.5 at n=200 per seed, 3 seeds); at 1B, all methods degrade generation, and the result holds across preference data types (real-vs-generated and model-vs-model) and the data volumes tested (150-288 pairs). Gradient analysis reveals why: understanding and generation gradients are near-orthogonal (cos ~ 0) with ~11-14x magnitude imbalance driven by VQ token count asymmetry (576 generation tokens vs. ~30-100 text tokens). This imbalance is the dominant interference mechanism in multi-task DPO; magnitude-balancing yields directionally positive understanding deltas (+0.01-0.04 VQA, though individually not significant), but the generation gap persists regardless. We identify discrete VQ tokenization as a likely structural bottleneck -- supported by the generation DPO loss converging to ln(2) -- and provide practical guidance for practitioners working with VQ-based unified models. 

---
# Dependence Fidelity and Downstream Inference Stability in Generative Models 

**Authors**: Nazia Riasat  

**Link**: [PDF](https://arxiv.org/pdf/2603.17041)  

**Abstract**: Recent advances in generative AI have led to increasingly realistic synthetic data, yet evaluation criteria remain focused on marginal distribution matching. While these diagnostics assess local realism, they provide limited insight into whether a generative model preserves the multivariate dependence structures governing downstream inference. We introduce covariance-level dependence fidelity as a practical criterion for evaluating whether a generative distribution preserves joint structure beyond univariate marginals. We establish three core results. First, distributions can match all univariate marginals exactly while exhibiting substantially different dependence structures, demonstrating marginal fidelity alone is insufficient. Second, dependence divergence induces quantitative instability in downstream inference, including sign reversals in regression coefficients despite identical marginal behavior. Third, explicit control of covariance-level dependence divergence ensures stable behavior for dependence-sensitive tasks such as principal component analysis. Synthetic constructions illustrate how dependence preservation failures lead to incorrect conclusions despite identical marginal distributions. These results highlight dependence fidelity as a useful diagnostic for evaluating generative models in dependence-sensitive downstream tasks, with implications for diffusion models and variational autoencoders. These guarantees apply specifically to procedures governed by covariance structure; tasks requiring higher-order dependence such as tail-event estimation require richer criteria. 

---
# Shared Representation Learning for Reference-Guided Targeted Sound Detection 

**Authors**: Shubham Gupta, Adarsh Arigala, B. R. Dilleswari, Sri Rama Murty Kodukula  

**Link**: [PDF](https://arxiv.org/pdf/2603.17025)  

**Abstract**: Human listeners exhibit the remarkable ability to segregate a desired sound from complex acoustic scenes through selective auditory attention, motivating the study of Targeted Sound Detection (TSD). The task requires detecting and localizing a target sound in a mixture when a reference audio of that sound is provided. Prior approaches, rely on generating a sound-discriminative conditional embedding vector for the reference and pairing it with a mixture encoder, jointly optimized with a multi-task learning approach. In this work, we propose a unified encoder architecture that processes both the reference and mixture audio within a shared representation space, promoting stronger alignment while reducing architectural complexity. This design choice not only simplifies the overall framework but also enhances generalization to unseen classes. Following the multi-task training paradigm, our method achieves substantial improvements over prior approaches, surpassing existing methods and establishing a new state-of-the-art benchmark for targeted sound detection, with a segment-level F1 score of 83.15% and an overall accuracy of 95.17% on the URBAN-SED dataset. 

---
# HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning 

**Authors**: Shenzhi Wang, Shixuan Liu, Jing Zhou, Chang Gao, Xiong-Hui Chen, Binghai Wang, An Yang, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.17024)  

**Abstract**: VLMs show strong multimodal capabilities, but they still struggle with fine-grained vision-language reasoning. We find that long CoT reasoning exposes diverse failure modes, including perception, reasoning, knowledge, and hallucination errors, which can compound across intermediate steps. However, most existing vision-language data used for RLVR does not involve complex reasoning chains that rely on visual evidence throughout, leaving these weaknesses largely unexposed. We therefore propose HopChain, a scalable framework for synthesizing multi-hop vision-language reasoning data specifically for RLVR training of VLMs. Each synthesized multi-hop query forms a logically dependent chain of instance-grounded hops, where earlier hops establish the instances, sets, or conditions needed for later hops, while the final answer remains a specific, unambiguous number suitable for verifiable rewards. We add the multi-hop data synthesized by HopChain to the original RLVR data used to train Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, and compare against RLVR on the original RLVR data alone across 24 benchmarks spanning STEM and Puzzle, General VQA, Text Recognition and Document Understanding, and Video Understanding. Although this multi-hop data is not synthesized to target any specific benchmark, adding it improves 20 out of 24 benchmarks on both models, indicating broad and generalizable gains. To demonstrate that full chained queries are important, we replace them with half-multi-hop or single-hop variants, reducing the 24-benchmark average accuracy by 5.3 and 7.0 points, respectively. Multi-hop training also strengthens long-CoT vision-language reasoning, with gains peaking at more than 50 accuracy points in the ultra-long-CoT regime. These experiments establish HopChain as an effective, scalable framework for synthesizing multi-hop data that improves generalizable vision-language reasoning. 

---
# LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings 

**Authors**: Lifu Tu, Rongguang Wang, Tao Sheng, Sujjith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.17017)  

**Abstract**: Robustness evaluation for Natural Language to SQL (NL2SQL) systems is essential because real-world database environments are dynamic, noisy, and continuously evolving, whereas conventional benchmark evaluations typically assume static schemas and well-formed user inputs. In this work, we introduce a robustness evaluation benchmark containing approximately ten types of perturbations and conduct evaluations under both traditional and agentic settings. We assess multiple state-of-the-art large language models (LLMs), including Grok-4.1, Gemini-3-Pro, Claude-Opus-4.6, and GPT-5.2. Our results show that these models generally maintain strong performance under several perturbations; however, notable performance degradation is observed for surface-level noise (e.g., character-level corruption) and linguistic variation that preserves semantics while altering lexical or syntactic forms. Furthermore, we observe that surface-level noise causes larger performance drops in traditional pipelines, whereas linguistic variation presents greater challenges in agentic settings. These findings highlight the remaining challenges in achieving robust NL2SQL systems, particularly in handling linguistic variability. 

---
# Empirical Recipes for Efficient and Compact Vision-Language Models 

**Authors**: Jiabo Huang, Zhizhong Li, Sina Sajadmanesh, Weiming Zhuang, Lingjuan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16987)  

**Abstract**: Deploying vision-language models (VLMs) in resource-constrained settings demands low latency and high throughput, yet existing compact VLMs often fall short of the inference speedups their smaller parameter counts suggest. To explain this discrepancy, we conduct an empirical end-to-end efficiency analysis and systematically profile inference to identify the dominant bottlenecks. Based on these findings, we develop optimization recipes tailored to compact VLMs that substantially reduce latency while preserving accuracy. These techniques cut time to first token (TTFT) by 53% on InternVL3-2B and by 93% on SmolVLM-256M. Our recipes are broadly applicable across both VLM architectures and common serving frameworks, providing practical guidance for building efficient VLM systems. Beyond efficiency, we study how to extend compact VLMs with structured perception outputs and introduce the resulting model family, ArgusVLM. Across diverse benchmarks, ArgusVLM achieves strong performance while maintaining a compact and efficient design. 

---
# Implementation of tangent linear and adjoint models for neural networks based on a compiler library tool 

**Authors**: Sa Xiao, Hao Jing, Honglu Sun, Haoyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16976)  

**Abstract**: This paper presents TorchNWP, a compilation library tool for the efficient coupling of artificial intelligence components and traditional numerical models. It aims to address the issues of poor cross-language compatibility, insufficient coupling flexibility, and low data transfer efficiency between operational numerical models developed in Fortran and Python-based deep learning frameworks. Based on LibTorch, it optimizes and designs a unified application-layer calling interface, converts deep learning models under the PyTorch framework into a static binary format, and provides C/C++ interfaces. Then, using hybrid Fortran/C/C++ programming, it enables the deployment of deep learning models within numerical models. Integrating TorchNWP into a numerical model only requires compiling it into a callable link library and linking it during the compilation and linking phase to generate the executable. On this basis, tangent linear and adjoint model based on neural networks are implemented at the C/C++ level, which can shield the internal structure of neural network models and simplify the construction process of four-dimensional variational data assimilation systems. Meanwhile, it supports deployment on heterogeneous platforms, is compatible with mainstream neural network models, and enables mapping of different parallel granularities and efficient parallel execution. Using this tool requires minimal code modifications to the original numerical model, thus reducing coupling costs. It can be efficiently integrated into numerical weather prediction models such as CMA-GFS and MCV, and has been applied to the coupling of deep learning-based physical parameterization schemes (e.g., radiation, non-orographic gravity wave drag) and the development of their tangent linear and adjoint models, significantly improving the accuracy and efficiency of numerical weather prediction. 

---
# The State of Generative AI in Software Development: Insights from Literature and a Developer Survey 

**Authors**: Vincent Gurgul, Robin Gubela, Stefan Lessmann  

**Link**: [PDF](https://arxiv.org/pdf/2603.16975)  

**Abstract**: Generative Artificial Intelligence (GenAI) rapidly transforms software engineering, yet existing research remains fragmented across individual tasks in the Software Development Lifecycle. This study integrates a systematic literature review with a survey of 65 software developers. The results show that GenAI exerts its highest impact in design, implementation, testing, and documentation, where over 70 % of developers report at least halving the time for boilerplate and documentation tasks. 79 % of survey respondents use GenAI daily, preferring browser-based Large Language Models over alternatives integrated directly in their development environment. Governance is maturing, with two-thirds of organizations maintaining formal or informal guidelines. In contrast, early SDLC phases such as planning and requirements analysis show markedly lower reported benefits. In a nutshell, GenAI shifts value creation from routine coding toward specification quality, architectural reasoning, and oversight, while risks such as uncritical adoption, skill erosion, and technical debt require robust governance and human-in-the-loop mechanisms. 

---
# Are a Thousand Words Better Than a Single Picture? Beyond Images -- A Framework for Multi-Modal Knowledge Graph Dataset Enrichment 

**Authors**: Pengyu Zhang, Klim Zaporojets, Jie Liu, Jia-Hong Huang, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2603.16974)  

**Abstract**: Multi-Modal Knowledge Graphs (MMKGs) benefit from visual information, yet large-scale image collection is hard to curate and often excludes ambiguous but relevant visuals (e.g., logos, symbols, abstract scenes). We present Beyond Images, an automatic data-centric enrichment pipeline with optional human auditing. This pipeline operates in three stages: (1) large-scale retrieval of additional entity-related images, (2) conversion of all visual inputs into textual descriptions to ensure that ambiguous images contribute usable semantics rather than noise, and (3) fusion of multi-source descriptions using a large language model (LLM) to generate concise, entity-aligned summaries. These summaries replace or augment the text modality in standard MMKG models without changing their architectures or loss functions. Across three public MMKG datasets and multiple baseline models, we observe consistent gains (up to 7% Hits@1 overall). Furthermore, on a challenging subset of entities with visually ambiguous logos and symbols, converting images into text yields large improvements (201.35% MRR and 333.33% Hits@1). Additionally, we release a lightweight Text-Image Consistency Check Interface for optional targeted audits, improving description quality and dataset reliability. Our results show that scaling image coverage and converting ambiguous visuals into text is a practical path to stronger MMKG completion. Code, datasets, and supplementary materials are available at this https URL. 

---
# Continual Multimodal Egocentric Activity Recognition via Modality-Aware Novel Detection 

**Authors**: Wonseon Lim, Hyejeong Im, Dae-Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.16970)  

**Abstract**: Multimodal egocentric activity recognition integrates visual and inertial cues for robust first-person behavior understanding. However, deploying such systems in open-world environments requires detecting novel activities while continuously learning from non-stationary streams. Existing methods rely on the main logits for novelty scoring, without fully exploiting the complementary evidence available from individual modalities. Because these logits are often dominated by RGB, cues from other modalities, particularly IMU, remain underutilized, and this imbalance worsens over time under catastrophic forgetting. To address this, we propose MAND, a modality-aware framework for multimodal egocentric open-world continual learning. At inference, Modality-aware Adaptive Scoring (MoAS) estimates sample-wise modality reliability from energy scores and adaptively integrates modality logits to better exploit complementary modality cues for novelty detection. During training, Modality-wise Representation Stabilization Training (MoRST) preserves modality-specific discriminability across tasks via auxiliary heads and modality-wise logit distillation. Experiments on a public multimodal egocentric benchmark show that MAND improves novel activity detection AUC by up to 10\% and known-class classification accuracy by up to 2.8\% over state-of-the-art baselines. 

---
# DeepStage: Learning Autonomous Defense Policies Against Multi-Stage APT Campaigns 

**Authors**: Trung V. Phan, Tri Gia Nguyen, Thomas Bauschert  

**Link**: [PDF](https://arxiv.org/pdf/2603.16969)  

**Abstract**: This paper presents DeepStage, a deep reinforcement learning (DRL) framework for adaptive, stage-aware defense against Advanced Persistent Threats (APTs). The enterprise environment is modeled as a partially observable Markov decision process (POMDP), where host provenance and network telemetry are fused into unified provenance graphs. Building on our prior work, StageFinder, a graph neural encoder and an LSTM-based stage estimator infer probabilistic attacker stages aligned with the MITRE ATT&CK framework. These stage beliefs, combined with graph embeddings, guide a hierarchical Proximal Policy Optimization (PPO) agent that selects defense actions across monitoring, access control, containment, and remediation. Evaluated in a realistic enterprise testbed using CALDERA-driven APT playbooks, DeepStage achieves a stage-weighted F1-score of 0.89, outperforming a risk-aware DRL baseline by 21.9%. The results demonstrate effective stage-aware and cost-efficient autonomous cyber defense. 

---
# MSRAMIE: Multimodal Structured Reasoning Agent for Multi-instruction Image Editing 

**Authors**: Zhaoyuan Qiu, Ken Chen, Xiangwei Wang, Yu Xia, Sachith Seneviratne, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2603.16967)  

**Abstract**: Existing instruction-based image editing models perform well with simple, single-step instructions but degrade in realistic scenarios that involve multiple, lengthy, and interdependent directives. A main cause is the scarcity of training data with complex multi-instruction annotations. However, it is costly to collect such data and retrain these models. To address this challenge, we propose MSRAMIE, a training-free agent framework built on Multimodal Large Language Model (MLLM). MSRAMIE takes existing editing models as plug-in components and handle multi-instruction tasks via structured multimodal reasoning. It orchestrates iterative interactions between an MLLM-based Instructor and an image editing Actor, introducing a novel reasoning topology that comprises the proposed Tree-of-States and Graph-of-References. During inference, complex instructions are decomposed into multiple editing steps which enable state transitions, cross-step information aggregation, and original input recall, which enables systematic exploration of the image editing space and flexible progressive output refinement. The visualizable inference topology further provides interpretable and controllable decision pathways. Experiments show that as the instruction complexity increases, MSRAMIE can improve instruction following over 15% and increases the probability of finishing all modifications in a single run over 100%, while preserving perceptual quality and maintaining visual consistency. 

---
# CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization 

**Authors**: Liangbin Huang, Xiaohua Liao, Chaoqun Cui, Shijing Wang, Zhaolong Huang, Yanlong Du, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16966)  

**Abstract**: Traditional speaker diarization systems have primarily focused on constrained scenarios such as meetings and interviews, where the number of speakers is limited and acoustic conditions are relatively clean. To explore open-world speaker diarization, we extend this task to the visual media domain, encompassing complex audiovisual programs such as films and TV series. This new setting introduces several challenges, including long-form video understanding, a large number of speakers, cross-modal asynchrony between audio and visual cues, and uncontrolled in-the-wild variability. To address these challenges, we propose Cinematic Speaker Registration & Diarization (CineSRD), a unified multimodal framework that leverages visual, acoustic, and linguistic cues from video, speech, and subtitles for speaker annotation. CineSRD first performs visual anchor clustering to register initial speakers and then integrates an audio language model for speaker turn detection, refining annotations and supplementing unregistered off-screen speakers. Furthermore, we construct and release a dedicated speaker diarization benchmark for visual media that includes Chinese and English programs. Experimental results demonstrate that CineSRD achieves superior performance on the proposed benchmark and competitive results on conventional datasets, validating its robustness and generalizability in open-world visual media settings. 

---
# Adversarial attacks against Modern Vision-Language Models 

**Authors**: Alejandro Paredes La Torre  

**Link**: [PDF](https://arxiv.org/pdf/2603.16960)  

**Abstract**: We study adversarial robustness of open-source vision-language model (VLM) agents deployed in a self-contained e-commerce environment built to simulate realistic pre-deployment conditions. We evaluate two agents, LLaVA-v1.5-7B and Qwen2.5-VL-7B, under three gradient-based attacks: the Basic Iterative Method (BIM), Projected Gradient Descent (PGD), and a CLIP-based spectral attack. Against LLaVA, all three attacks achieve substantial attack success rates (52.6%, 53.8%, and 66.9% respectively), demonstrating that simple gradient-based methods pose a practical threat to open-source VLM agents. Qwen2.5-VL proves significantly more robust across all attacks (6.5%, 7.7%, and 15.5%), suggesting meaningful architectural differences in adversarial resilience between open-source VLM families. These findings have direct implications for the security evaluation of VLM agents prior to commercial deployment. 

---
# Machine intelligence supports the full chain of 2D dendrite synthesis 

**Authors**: Wenqiang Huang, Susu Fang, Xuhang Gu, Shen'ao Xue, Huanhuan Xing, Junjie Jiang, Junying Zhang, Shen Zhou, Zheng Luo, Jin Zhang, Fangping Ouyang, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16959)  

**Abstract**: Exemplified by the chemical vapor deposition growth of two-dimensional dendrites, which has potential applications in catalysis and presents a parameter-intensive, data-scarce and reaction process-complex model problem, we devise a machine intelligence-empowered framework for the full chain support of material synthesis, encompassing rapid process optimization, accurate customized synthesis, and comprehensive mechanism this http URL, active learning is integrated into the experimental workflow, identifying an optimal recipe for the growth of highly-branched, electrocatalytically-active ReSe2 dendrites through 60 experiments (4 iterations), which account for less than 1.3% of the numerous possible parameter this http URL, a prediction accuracy-guided data augmentation strategy is developed combined with a tree-based machine learning (ML) algorithm, unveiling a non-linear correlation between 5 process variables and fractal dimension (DF) of ReSe2 dendrites with only 9 experiment additions, which guides the synthesis of various user-defined DF. Finally, we construct a data-knowledge dual-driven mechanism model by integration of cross-scale characterizations, interpretable ML models, and domain knowledge in thermodynamics and kinetics, unraveling synergistic contributions of multiple process parameters to the product morphology. This work demonstrates the ML potential to transform the research paradigm and is adaptable to broader material synthesis. 

---
# PhysQuantAgent: An Inference Pipeline of Mass Estimation for Vision-Language Models 

**Authors**: Hisayuki Yokomizo, Taiki Miyanishi, Yan Gang, Shuhei Kurita, Nakamasa Inoue, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2603.16958)  

**Abstract**: Vision-Language Models (VLMs) are increasingly applied to robotic perception and manipulation, yet their ability to infer physical properties required for manipulation remains limited. In particular, estimating the mass of real-world objects is essential for determining appropriate grasp force and ensuring safe interaction. However, current VLMs lack reliable mass reasoning capabilities, and most existing benchmarks do not explicitly evaluate physical quantity estimation under realistic sensing conditions. In this work, we propose PhysQuantAgent, a framework for real-world object mass estimation using VLMs, together with VisPhysQuant, a new benchmark dataset for evaluation. VisPhysQuant consists of RGB-D videos of real objects captured from multiple viewpoints, annotated with precise mass measurements. To improve estimation accuracy, we introduce three visual prompting methods that enhance the input image with object detection, scale estimation, and cross-sectional image generation to help the model comprehend the size and internal structure of the target object. Experiments show that visual prompting significantly improves mass estimation accuracy on real-world data, suggesting the efficacy of integrating spatial reasoning with VLM knowledge for physical inference. 

---
# Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies 

**Authors**: Utkarsh Grover, Ravi Ranjan, Mingyang Mao, Trung Tien Dong, Satvik Praveen, Zhenqi Wu, J. Morris Chang, Tinoosh Mohsenin, Yi Sheng, Agoritsa Polyzou, Eiman Kanjo, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16952)  

**Abstract**: Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression. Real-time control must operate within strict size, weight, and power constraints, where memory traffic, compute latency, timing variability, and safety margins interact directly. The Deployment Gauntlet organizes these constraints into eight coupled barriers that determine whether embodied foundation models can run reliably in practice. Across representative edge workloads, autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost. Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning. 

---
# EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments 

**Authors**: Kun Luo, Xiaoguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.16947)  

**Abstract**: Zero-shot vision-and-language navigation in continuous environments (VLN-CE) remains challenging for modern vision-language models (VLMs). Although these models encode useful semantic priors, their open-ended reasoning does not directly translate into stable long-horizon embodied execution. We argue that the key bottleneck is not missing knowledge alone, but missing an execution structure for organizing instruction following, perceptual grounding, temporal progress, and stage verification. We propose EmergeNav, a zero-shot framework that formulates continuous VLN as structured embodied inference. EmergeNav combines a Plan--Solve--Transition hierarchy for stage-structured execution, GIPE for goal-conditioned perceptual extraction, contrastive dual-memory reasoning for progress grounding, and role-separated Dual-FOV sensing for time-aligned local control and boundary verification. On VLN-CE, EmergeNav achieves strong zero-shot performance using only open-source VLM backbones and no task-specific training, explicit maps, graph search, or waypoint predictors, reaching 30.00 SR with Qwen3-VL-8B and 37.00 SR with Qwen3-VL-32B. These results suggest that explicit execution structure is a key ingredient for turning VLM priors into stable embodied navigation behavior. 

---
# Automatic Termination Strategy of Inelastic Neutron-scattering Measurement Using Bayesian Optimization for Bin-width Selection 

**Authors**: Kensuke Muto, Hirotaka Sakamoto, Kenji Nagata, Taka-hisa Arima, Masato Okada  

**Link**: [PDF](https://arxiv.org/pdf/2603.16946)  

**Abstract**: Currently, an excessive amount of event data is being obtained in four-dimensional inelastic neutron-scattering experiments. A method for automatic bin-width optimization of multidimensional histograms has been developed and recently validated on real inelastic neutron-scattering data. However, measuring beyond the equipment resolution leads to inefficient use of valuable beam time. To improve experimental efficiency, an automatic termination strategy is essential. We propose a Bayesian-optimization-based method to compute stopping criteria and determine whether to continue or terminate the experiment in real time. In the proposed method, the bin-width optimization is performed using Bayesian optimization to efficiently compute the optimal bin widths. The experiment is terminated when the optimal bin widths become smaller than the target resolutions. In numerical experiments using real inelastic neutron-scattering data, the optimal bin widths decrease as the number of events increases. Even the optimal bin widths for data downsampled to 1/5 are comparable with the resolutions limited by the sample size, choppers, and so on. This implies excessive measurement of the inelastic neutron experiments for the moment. Moreover, we found that Bayesian optimization can reduce the search cost to approximately 10% of an exhaustive search in our numerical experiments. 

---
# Joint Optimization of Storage and Loading for High-Performance 3D Point Cloud Data Processing 

**Authors**: Ke Wang, Yanfei Cao, Xiangzhi Tao, Naijie Gu, Jun Yu, Zhengdong Wang, Shouyang Dong, Fan Yu, Cong Wang, Yang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16945)  

**Abstract**: With the rapid development of computer vision and deep learning, significant advancements have been made in 3D vision, partic- ularly in autonomous driving, robotic perception, and augmented reality. 3D point cloud data, as a crucial representation of 3D information, has gained widespread attention. However, the vast scale and complexity of point cloud data present significant chal- lenges for loading and processing and traditional algorithms struggle to handle large-scale this http URL diversity of storage formats for point cloud datasets (e.g., PLY, XYZ, BIN) adds complexity to data handling and results in inefficiencies in data preparation. Al- though binary formats like BIN and NPY have been used to speed up data access, they still do not fully address the time-consuming data loading and processing phase. To overcome these challenges, we propose the .PcRecord format, a unified data storage solution designed to reduce the storage occupation and accelerate the processing of point cloud data. We also introduce a high-performance data processing pipeline equipped with multiple modules. By leveraging a multi-stage parallel pipeline architecture, our system optimizes the use of computational resources, significantly improving processing speed and efficiency. This paper details the im- plementation of this system and demonstrates its effectiveness in addressing the challenges of handling large-scale point cloud this http URL average, our system achieves performance improvements of 6.61x (ModelNet40), 2.69x (S3DIS), 2.23x (ShapeNet), 3.09x (Kitti), 8.07x (SUN RGB-D), and 5.67x (ScanNet) with GPU and 6.9x, 1.88x, 1.29x, 2.28x, 25.4x, and 19.3x with Ascend. 

---
# Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models 

**Authors**: Yujia Yang, Yuanxiang Wang, Zhenyu Guan, Tiankun Yang, Chenxi Bao, Haopeng Jin, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Haijin Liang, Jin Ma, Xinming Wang, Ruiwen Tao, Hongzhu Yi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16944)  

**Abstract**: While Instruction-based Image Editing (IIE) has achieved significant progress, existing benchmarks pursue task breadth via mixed evaluations. This paradigm obscures a critical failure mode crucial in professional applications: the inconsistent performance of models across tasks of varying semantic scales. To address this gap, we introduce Omni IIE Bench, a high-quality, human-annotated benchmark specifically designed to diagnose the editing consistency of IIE models in practical application scenarios. Omni IIE Bench features an innovative dual-track diagnostic design: (1) Single-turn Consistency, comprising shared-context task pairs of attribute modification and entity replacement; and (2) Multi-turn Coordination, involving continuous dialogue tasks that traverse semantic scales. The benchmark is constructed via an exceptionally rigorous multi-stage human filtering process, incorporating a quality standard enforced by computer vision graduate students and an industry relevance review conducted by professional designers. We perform a comprehensive evaluation of 8 mainstream IIE models using Omni IIE Bench. Our analysis quantifies, for the first time, a prevalent performance gap: nearly all models exhibit a significant performance degradation when transitioning from low-semantic-scale to high-semantic-scale tasks. Omni IIE Bench provides critical diagnostic tools and insights for the development of next-generation, more reliable, and stable IIE models. 

---
# KGS-GCN: Enhancing Sparse Skeleton Sensing via Kinematics-Driven Gaussian Splatting and Probabilistic Topology for Action Recognition 

**Authors**: Yuhan Chen, Yicui Shi, Guofa Li, Liping Zhang, Jie Li, Jiaxin Gao, Wenbo Chu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16943)  

**Abstract**: Skeleton-based action recognition is widely utilized in sensor systems including human-computer interaction and intelligent surveillance. Nevertheless, current sensor devices typically generate sparse skeleton data as discrete coordinates, which inevitably discards fine-grained spatiotemporal details during highly dynamic movements. Moreover, the rigid constraints of predefined physical sensor topologies hinder the modeling of latent long-range dependencies. To overcome these limitations, we propose KGS-GCN, a graph convolutional network that integrates kinematics-driven Gaussian splatting with probabilistic topology. Our framework explicitly addresses the challenges of sensor data sparsity and topological rigidity by transforming discrete joints into continuous generative representations. Firstly, a kinematics-driven Gaussian splatting module is designed to dynamically construct anisotropic covariance matrices using instantaneous joint velocity vectors. This module enhances visual representation by rendering sparse skeleton sequences into multi-view continuous heatmaps rich in spatiotemporal semantics. Secondly, to transcend the limitations of fixed physical connections, a probabilistic topology construction method is proposed. This approach generates an adaptive prior adjacency matrix by quantifying statistical correlations via the Bhattacharyya distance between joint Gaussian distributions. Ultimately, the GCN backbone is adaptively modulated by the rendered visual features via a visual context gating mechanism. Empirical results demonstrate that KGS-GCN significantly enhances the modeling of complex spatiotemporal dynamics. By addressing the inherent limitations of sparse inputs, our framework offers a robust solution for processing low-fidelity sensor data. This approach establishes a practical pathway for improving perceptual reliability in real-world sensing applications. 

---
# UNICORN: Ultrasound Nakagami Imaging via Score Matching and Adaptation for Assessing Hepatic Steatosis 

**Authors**: Kwanyoung Kim, Jaa-Yeon Lee, Youngjun Ko, GunWoo Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.16942)  

**Abstract**: Ultrasound imaging is an essential first-line tool for assessing hepatic steatosis. While conventional B-mode ultrasound imaging has limitations in providing detailed tissue characterization, ultrasound Nakagami imaging holds promise for visualizing and quantifying tissue scattering in backscattered signals, with potential applications in fat fraction analysis. However, existing methods for Nakagami imaging struggle with optimal window size selection and suffer from estimator instability, leading to degraded image resolution. To address these challenges, we propose a novel method called UNICORN (Ultrasound Nakagami Imaging via Score Matching and Adaptation), which offers an accurate, closed-form estimator for Nakagami parameter estimation based on the score function of the ultrasound envelope signal. Unlike methods that visualize only specific regions of interest (ROI) and estimate parameters within fixed window sizes, our approach provides comprehensive parameter mapping by providing a pixel-by-pixel estimator, resulting in high-resolution imaging. We demonstrated that our proposed estimator effectively assesses hepatic steatosis and provides visual distinction in the backscattered statistics associated with this condition. Through extensive experiments using real envelope data from patient, we validated that UNICORN enables clinical detection of hepatic steatosis and exhibits robustness and generalizability. 

---
# On the Degrees of Freedom of Gridded Control Points in Learning-Based Medical Image Registration 

**Authors**: Wen Yan, Qianye Yang, Yipei Wang, Shonit Punwani, Mark Emberton, Vasilis Stavrinides, Yipeng Hu, Dean Barratt  

**Link**: [PDF](https://arxiv.org/pdf/2603.16940)  

**Abstract**: Many registration problems are ill-posed in homogeneous or noisy regions, and dense voxel-wise decoders can be unnecessarily high-dimensional. A sparse control-point parameterisation provides a compact, smooth deformation representation while reducing memory and improving stability. This work investigates the required control points for learning-based registration network development. We present GridReg, a learning-based registration framework that replaces dense voxel-wise decoding with displacement predictions at a sparse grid of control points. This design substantially cuts the parameter count and memory while retaining registration accuracy. Multiscale 3D encoder feature maps are flattened into a 1D token sequence with positional encoding to retain spatial context. The model then predicts a sparse gridded deformation field using a cross-attention module. We further introduce grid-adaptive training, enabling an adaptive model to operate at multiple grid sizes at inference without retraining. This work quantitatively demonstrates the benefits of using sparse grids. Using three data sets for registering prostate gland, pelvic organs and neurological structures, the results suggested a significant improvement with the usage of grid-controled displacement field. Alternatively, the superior registration performance was obtained using the proposed approach, with a similar or less computational cost, compared with existing algorithms that predict DDFs or displacements sampled on scattered key points. 

---
# Cryptographic Runtime Governance for Autonomous AI Systems: The Aegis Architecture for Verifiable Policy Enforcement 

**Authors**: Adam Massimo Mazzocchetti  

**Link**: [PDF](https://arxiv.org/pdf/2603.16938)  

**Abstract**: Contemporary AI governance frameworks rely heavily on post hoc oversight, policy guidance, and behavioral alignment techniques, yet these mechanisms become fragile as systems gain autonomy, speed, and operational opacity. This paper presents Aegis, a runtime governance architecture for autonomous AI systems that treats policy and legal constraints as execution conditions rather than advisory principles. Aegis binds each governed agent to a cryptographically sealed Immutable Ethics Policy Layer (IEPL) at system genesis and enforces external emissions through an Ethics Verification Agent (EVA), an Enforcement Kernel Module (EKM), and an Immutable Logging Kernel (ILK). Amendments to the governing policy layer require quorum approval and redeclaration of the system trust root; verified violations trigger autonomous shutdown and generation of auditable proof artifacts.
We evaluate the architecture within the Civitas runtime using three operational measures: proof verification latency under tamper conditions, publication overhead, and alignment retention performance relative to an ungoverned baseline. In controlled trials, Aegis demonstrates median proof verification latency of 238 ms, median publication overhead of approximately 9.4 ms, and higher alignment retention than the baseline condition across matched tasks. We argue that these results support a shift in AI governance from discretionary oversight toward verifiable runtime constraint. Rather than claiming to resolve machine ethics in the abstract, the proposed architecture seeks to show that policy violating behavior can be rendered operationally non executable within a controlled runtime governance framework. The paper concludes by discussing methodological limits, evidentiary implications, and the role of proof oriented governance in high assurance AI deployment. 

---
# TDMM-LM: Bridging Facial Understanding and Animation via Language Models 

**Authors**: Luchuan Song, Pinxin Liu, Haiyang Liu, Zhenchao Jin, Yolo Yunlong Tang, Zichong Xu, Susan Liang, Jing Bi, Jason J Corso, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16936)  

**Abstract**: Text-guided human body animation has advanced rapidly, yet facial animation lags due to the scarcity of well-annotated, text-paired facial corpora. To close this gap, we leverage foundation generative models to synthesize a large, balanced corpus of facial behavior. We design prompts suite covering emotions and head motions, generate about 80 hours of facial videos with multiple generators, and fit per-frame 3D facial parameters, yielding large-scale (prompt and parameter) pairs for training. Building on this dataset, we probe language models for bidirectional competence over facial motion via two complementary tasks: (1) Motion2Language: given a sequence of 3D facial parameters, the model produces natural-language descriptions capturing content, style, and dynamics; and (2) Language2Motion: given a prompt, the model synthesizes the corresponding sequence of 3D facial parameters via quantized motion tokens for downstream animation. Extensive experiments show that in this setting language models can both interpret and synthesize facial motion with strong generalization. To best of our knowledge, this is the first work to cast facial-parameter modeling as a language problem, establishing a unified path for text-conditioned facial animation and motion understanding. 

---
# GenLie: A Global-Enhanced Lie Detection Network under Sparsity and Semantic Interference 

**Authors**: Zongshun Zhang, Yao Liu, Qiao Liu, Xuefeng Peng, Peiyuan Jiang, Jiaye Yang, Daibing Yao, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16935)  

**Abstract**: Video-based lie detection aims to identify deceptive behaviors from visual cues. Despite recent progress, its core challenge lies in learning sparse yet discriminative representations. Deceptive signals are typically subtle and short-lived, easily overwhelmed by redundant information, while individual and contextual variations introduce strong identity-related noise. To address this issue, we propose GenLie, a Global-Enhanced Lie Detection Network that performs local feature modeling under global supervision. Specifically, sparse and subtle deceptive cues are captured at the local level, while global supervision and optimization ensure robust and discriminative representations by suppressing identity-related noise. Experiments on three public datasets, covering both high- and low-stakes scenarios, show that GenLie consistently outperforms state-of-the-art methods. Source code is available at this https URL. 

---
# AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding 

**Authors**: Abderrahmene Boudiaf, Irfan Hussain, Sajid Javed  

**Link**: [PDF](https://arxiv.org/pdf/2603.16934)  

**Abstract**: The deployment of Multimodal Large Language Models (MLLMs) in agriculture is currently stalled by a critical trade-off: the existing literature lacks the large-scale agricultural datasets required for robust model development and evaluation, while current state-of-the-art models lack the verified domain expertise necessary to reason across diverse taxonomies. To address these challenges, we propose the Vision-to-Verified-Knowledge (V2VK) pipeline, a novel generative AI-driven annotation framework that integrates visual captioning with web-augmented scientific retrieval to autonomously generate the AgriMM benchmark, effectively eliminating biological hallucinations by grounding training data in verified phytopathological literature. The AgriMM benchmark contains over 3,000 agricultural classes and more than 607k VQAs spanning multiple tasks, including fine-grained plant species identification, plant disease symptom recognition, crop counting, and ripeness assessment. Leveraging this verifiable data, we present AgriChat, a specialized MLLM that presents broad knowledge across thousands of agricultural classes and provides detailed agricultural assessments with extensive explanations. Extensive evaluation across diverse tasks, datasets, and evaluation conditions reveals both the capabilities and limitations of current agricultural MLLMs, while demonstrating AgriChat's superior performance over other open-source models, including internal and external benchmarks. The results validate that preserving visual detail combined with web-verified knowledge constitutes a reliable pathway toward robust and trustworthy agricultural AI. The code and dataset are publicly available at this https URL . 

---
# Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs 

**Authors**: Nimrod Shabtay, Moshe Kimhi, Artem Spector, Sivan Haray, Ehud Rivlin, Chaim Baskin, Raja Giryes, Eli Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2603.16932)  

**Abstract**: Vision-language models (VLMs) typically process images at a native high-resolution, forcing a trade-off between accuracy and computational efficiency: high-resolution inputs capture fine details but incur significant computational costs, while low-resolution inputs advocate for efficiency, they potentially miss critical visual information, like small text. We present AwaRes, a spatial-on-demand framework that resolves this accuracy-efficiency trade-off by operating on a low-resolution global view and using tool-calling to retrieve only high-resolution segments needed for a given query. We construct supervised data automatically: a judge compares low- vs.\ high-resolution answers to label whether cropping is needed, and an oracle grounding model localizes the evidence for the correct answer, which we map to a discrete crop set to form multi-turn tool-use trajectories. We train our framework with cold-start SFT followed by multi-turn GRPO with a composite reward that combines semantic answer correctness with explicit crop-cost penalties. Project page: this https URL 

---
# Script-to-Slide Grounding: Grounding Script Sentences to Slide Objects for Automatic Instructional Video Generation 

**Authors**: Rena Suzuki, Masato Kikuchi, Tadachika Ozono  

**Link**: [PDF](https://arxiv.org/pdf/2603.16931)  

**Abstract**: While slide-based videos augmented with visual effects are widely utilized in education and research presentations, the video editing process -- particularly applying visual effects to ground spoken content to slide objects -- remains highly labor-intensive. This study aims to develop a system that automatically generates such instructional videos from slides and corresponding scripts. As a foundational step, this paper proposes and formulates Script-to-Slide Grounding (S2SG), defined as the task of grounding script sentences to their corresponding slide objects. Furthermore, as an initial step, we propose ``Text-S2SG,'' a method that utilizes a large language model (LLM) to perform this grounding task for text objects. Our experiments demonstrate that the proposed method achieves high performance (F1-score: 0.924). The contribution of this work is the formalization of a previously implicit slide-based video editing process into a computable task, thereby paving the way for its automation. 

---
# Facial beauty prediction fusing transfer learning and broad learning system 

**Authors**: Junying Gan, Xiaoshan Xie, Yikui Zhai, Guohui He, Chaoyun Mai, Heng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16930)  

**Abstract**: Facial beauty prediction (FBP) is an important and challenging problem in the fields of computer vision and machine learning. Not only it is easily prone to overfitting due to the lack of large-scale and effective data, but also difficult to quickly build robust and effective facial beauty evaluation models because of the variability of facial appearance and the complexity of human perception. Transfer Learning can be able to reduce the dependence on large amounts of data as well as avoid overfitting problems. Broad learning system (BLS) can be capable of quickly completing models building and training. For this purpose, Transfer Learning was fused with BLS for FBP in this paper. Firstly, a feature extractor is constructed by way of CNNs models based on transfer learning for facial feature extraction, in which EfficientNets are used in this paper, and the fused features of facial beauty extracted are transferred to BLS for FBP, called E-BLS. Secondly, on the basis of E-BLS, a connection layer is designed to connect the feature extractor and BLS, called ER-BLS. Finally, experimental results show that, compared with the previous BLS and CNNs methods existed, the accuracy of FBP was improved by E-BLS and ER-BLS, demonstrating the effectiveness and superiority of the method presented, which can also be widely used in pattern recognition, object detection and image classification. 

---
# MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning 

**Authors**: Hongjun Wang, Wei Liu, Weibo Gu, Xing Sun, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.16929)  

**Abstract**: Regulating the importance ratio is critical for the training stability of Group Relative Policy Optimization (GRPO) based frameworks. However, prevailing ratio control methods, such as hard clipping, suffer from non-differentiable boundaries and vanishing gradient regions, failing to maintain gradient fidelity. Furthermore, these methods lack a hazard-aware mechanism to adaptively suppress extreme deviations, leaving the optimization process vulnerable to abrupt policy shifts. To address these challenges, we propose Modulated Hazard-aware Policy Optimization (MHPO), a novel framework designed for robust and stable reinforcement learning. The proposed MHPO introduces a Log-Fidelity Modulator (LFM) to map unbounded importance ratios into a bounded, differentiable domain. This mechanism effectively prevents high-variance outlier tokens from destabilizing the loss landscape while ensuring global gradient stability. Complementarily, a Decoupled Hazard Penalty (DHP) integrates cumulative hazard functions from survival analysis to independently regulate positive and negative policy shifts. By shaping the optimization landscape with hazard-aware penalties, the proposed MHPO achieves fine-grained regulation of asymmetric policy shifts simultaneously mitigating mode collapse from over-expansion and preventing policy erosion from catastrophic contraction within a stabilized trust region. Extensive evaluations on diverse reasoning benchmarks across both text-based and vision-language tasks demonstrate that MHPO consistently outperforms existing methods, achieving superior performance while significantly enhancing training stability. 

---
# Music Source Restoration with Ensemble Separation and Targeted Reconstruction 

**Authors**: Xinlong Deng, Yu Xia, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16926)  

**Abstract**: The Inaugural Music Source Restoration (MSR) Challenge targets the recovery of original, unprocessed stems from fully mixed and mastered music. Unlike conventional music source separation, MSR requires reversing complex production processes such as equalization, compression, reverberation, and other real-world degradations. To address MSR, we propose a two-stage system. First, an ensemble of pre-trained separation models produces preliminary source estimates. Then a set of pre-trained BSRNN-based restoration models performs targeted reconstruction to refine these estimates. On the official MSR benchmark, our system surpasses the baselines on all metrics, ranking second among all submissions. The code is available at this https URL 

---
# SimulU: Training-free Policy for Long-form Simultaneous Speech-to-Speech Translation 

**Authors**: Amirbek Djanibekov, Luisa Bentivogli, Matteo Negri, Sara Papi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16924)  

**Abstract**: Simultaneous speech-to-speech translation (SimulS2S) is essential for real-time multilingual communication, with increasing integration into meeting and streaming platforms. Despite this, SimulS2S remains underexplored in research, where current solutions often rely on resource-intensive training procedures and operate on short-form, pre-segmented utterances, failing to generalize to continuous speech. To bridge this gap, we propose SimulU, the first training-free policy for long-form SimulS2S. SimulU adopts history management and speech output selection strategies that exploit cross-attention in pre-trained end-to-end models to regulate both input history and output generation. Evaluations on MuST-C across 8 languages show that SimulU achieves a better or comparable quality-latency trade-off against strong cascaded models. By eliminating the need for ad-hoc training, SimulU offers a promising path to end-to-end SimulS2S in realistic, long-form scenarios. 

---
# Privacy and Safety Experiences and Concerns of U.S. Women Using Generative AI for Seeking Sexual and Reproductive Health Information 

**Authors**: Ina Kaleva, Xiao Zhan, Ruba Abu-Salma, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2603.16918)  

**Abstract**: The rapid adoption of generative AI (GenAI) chatbots has reshaped access to sexual and reproductive health (SRH) information, particularly following the overturning of Roe v. Wade, as individuals assigned female at birth increasingly turn to online sources. However, existing research remains largely model-centered, paying limited attention to user privacy and safety. We conducted semi-structured interviews with 18 U.S.-based participants from both restrictive and non-restrictive states who had used GenAI chatbots to seek SRH information. Adoption was influenced by perceived utility, usability, credibility, accessibility, and anthropomorphism, and many participants disclosed sensitive personal SRH details. Participants identified multiple privacy risks, including excessive data collection, government surveillance, profiling, model training, and data commodification. While most participants accepted these risks in exchange for perceived utility, abortion-related queries elicited heightened safety concerns. Few participants employed protective strategies beyond minimizing disclosures or deleting data. Based on these findings, we offer design and policy recommendations, such as health-specific features and stronger moderation practices, to enhance privacy and safety in GenAI-supported SRH information seeking. 

---
# Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection 

**Authors**: Jinyang Wu, Zihan Pan, Qiquan Zhang, Sailor Hardik Bhupendra, Soumik Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2603.16914)  

**Abstract**: Neural audio codecs discretize speech via residual vector quantization (RVQ), forming a coarse-to-fine hierarchy across quantizers. While codec models have been explored for representation learning, their discrete structure remains underutilized in speech deepfake detection. In particular, different quantization levels capture complementary acoustic cues, where early quantizers encode coarse structure and later quantizers refine residual details that reveal synthesis artifacts. Existing systems either rely on continuous encoder features or ignore this quantizer-level hierarchy. We propose a hierarchy-aware representation learning framework that models quantizer-level contributions through learnable global weighting, enabling structured codec representations aligned with forensic cues. Keeping the speech encoder backbone frozen and updating only 4.4% additional parameters, our method achieves relative EER reductions of 46.2% on ASVspoof 2019 and 13.9% on ASVspoof5 over strong baselines. 

---
# What on Earth is AlphaEarth? Hierarchical structure and functional interpretability for global land cover 

**Authors**: Ivan Felipe Benavides-Martinez, Justin Guthrie, Jhon Edwin Arias, Yeison Alberto Garces-Gomez, Angela Ines Guzman-Alvis, Cristiam Victoriano Portilla-Cabrera, Somnath Mondal, Andrew J. Allyn, Auroop R. Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2603.16911)  

**Abstract**: Geospatial foundation models generate high-dimensional embeddings that achieve strong predictive performance, yet their internal organization remains obscure, limiting their scientific use. Recent interpretability studies relate Google AlphaEarth Foundations (GAEF) embeddings to continuous environmental variables, but it is still unclear whether the embedding space exhibits a functional or hierarchical organization, in which some dimensions act as specialized representations while others encode shared or broader geospatial structure. In this work, we propose a functional interpretability framework that reverse-engineers the role of embedding dimensions by characterizing their contribution to land cover structure from observed classification behavior. The approach combines large-scale experimentation with a structural analysis of embedding-class relationships based on feature importance patterns and progressive ablation. Our results show that embedding dimensions exhibit consistent and non-uniform functional behavior, allowing them to be categorized along a hierarchical functional spectrum: specialist dimensions associated with specific land cover classes, low- and mid-generalist dimensions capturing shared characteristics between classes, and highgeneralist dimensions reflecting broader environmental gradients. Critically, we find that accurate land cover classification (98% of baseline performance) can be achieved using as few as 2 to 12 of the 64 available dimensions, depending on the class. This demonstrates substantial redundancy in the embedding space and offers a pathway toward significant reductions in computational cost. Together, these findings reveal that AlphaEarth embeddings are not only physically informative, but also functionally organized into a hierarchical structure, providing practical guidance for dimension selection in operational classification tasks. 

---
# TerraLingua: Emergence and Analysis of Open-endedness in LLM Ecologies 

**Authors**: Giuseppe Paolo, Jamieson Warner, Hormoz Shahrzad, Babak Hodjat, Risto Miikkulainen, Elliot Meyerson  

**Link**: [PDF](https://arxiv.org/pdf/2603.16910)  

**Abstract**: As autonomous agents increasingly operate in real-world digital ecosystems, understanding how they coordinate, form institutions, and accumulate shared culture becomes both a scientific and practical priority. This paper introduces TerraLingua, a persistent multi-agent ecology designed to study open-ended dynamics in such systems. Unlike prior large language model simulations with static or consequence-free environments, TerraLingua imposes resource constraints and limited lifespans for the agents. As a result, agents create artifacts that persist beyond individuals, shaping future interactions and selection pressures. To characterize the dynamics, an AI Anthropologist systematically analyzes agent behavior, group structure, and artifact evolution. Across experimental conditions, the results reveal the emergence of cooperative norms, division of labor, governance attempts, and branching artifact lineages consistent with cumulative cultural processes. Divergent outcomes across experimental runs can be traced back to specific innovations and organizational structures. TerraLingua thus provides a platform for characterizing the mechanisms of cumulative culture and social organization in artificial populations, and can serve as a foundation for guiding real-world agentic populations to socially beneficial outcomes. 

---
# Quantum-Assisted Optimal Rebalancing with Uncorrelated Asset Selection for Algorithmic Trading Walk-Forward QUBO Scheduling via QAOA 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.16904)  

**Abstract**: We present a hybrid classical-quantum framework for portfolio construction and rebalancing. Asset selection is performed using Ledoit-Wolf shrinkage covariance estimation combined with hierarchical correlation clustering to extract n = 10 decorrelated stocks from the S&P 500 universe without survivorship bias. Portfolio weights are optimised via an entropy-regularised Genetic Algorithm (GA) accelerated on GPU, alongside closed-form minimum-variance and equal-weight benchmarks.
Our primary contribution is the formulation of the portfolio rebalancing schedule as a Quadratic Unconstrained Binary Optimisation (QUBO) problem. The resulting combinatorial optimisation task is solved using the Quantum Approximate Optimisation Algorithm (QAOA) within a walk-forward framework designed to eliminate lookahead bias. This approach recasts dynamic rebalancing as a structured binary scheduling problem amenable to variational quantum methods.
Backtests on S&P 500 data (training: 2010-2024; out-of-sample test: 2025, n = 249 trading days) show that the GA + QAOA strategy attains a Sharpe ratio of 0.588 and total return of 10.1%, modestly outperforming the strongest classical baseline (GA with 10-day periodic rebalancing, Sharpe 0.575) while executing 8 rebalances versus 24, corresponding to a 44.5% reduction in transaction costs. Multi-restart QAOA (4096 measurement shots per run) exhibits concentrated probability mass on high-quality schedules, indicating stable convergence of the variational procedure.
These findings suggest that hybrid classical-quantum architectures can reduce turnover in portfolio rebalancing while preserving competitive risk-adjusted performance, providing a structured testbed for near-term quantum optimisation in financial applications. 

---
# From Language to Action in Arabic: Reliable Structured Tool Calling via Data-Centric Fine-Tuning 

**Authors**: Omer Nacar, Deema Alquffari, Saleh Alsharideh, Adeem AlOtaibi, Abdulaziz Alabdulkarim, Leen Alhazmi, Nada Alomar, Wareef Alzubaidi, Nada Alsultan, Ahmed Alrabghi, Demah Alhoshan, Rana Alsayyari, Hamed Alruwaili, Albaraa Jaafar, Khaled Alusmani, Abdulaziz Alsohimy, Munirah Alsubaie, Shahd Aldukhayil, Arwa Alali, Yazeed BinShihah, Razan Alsulaymi, Nourah Alhumaid, Razan Abdulsalam, Reem Alamoudi, Mohammed Alkhalifa  

**Link**: [PDF](https://arxiv.org/pdf/2603.16901)  

**Abstract**: Function-calling language models are essential for agentic AI systems that translate natural language into executable structured actions, yet existing models exhibit severe structural instability when applied to Arabic. We present AISA-AR-FunctionCall, a production-oriented Arabic function-calling framework built on a 270M-parameter FunctionGemma backbone and trained through systematic dataset auditing, schema repair, tool-aware prompt restructuring, and full-parameter supervised fine-tuning. On a held-out test set, fine-tuning reduces parse failures from 87\% to below 1\%, improves function name accuracy by more than eightfold, and substantially enhances argument alignment across dialects and domains. Error analysis reveals a transition from structural collapse to semantic misalignment, suggesting that serialization stability and decision-level reasoning are separable challenges. We further explore a reasoning-augmented LoRA variant that introduces explicit intermediate reasoning prior to tool invocation. All datasets and models are publicly released under the AISA framework. 

---
# Social physics in the age of artificial intelligence 

**Authors**: Anh Han, Joel Z. Leibo, Tom Lenaerts, Iyad Rahwan, Fernando Santos, Matjaž Perc, Valerio Capraro  

**Link**: [PDF](https://arxiv.org/pdf/2603.16900)  

**Abstract**: Artificial intelligence (AI) systems are rapidly becoming more capable, autonomous, and deeply embedded in social life. As humans increasingly interact, cooperate, and compete with AI, we move from purely human societies to hybrid human-AI societies whose collective dynamics cannot be captured by existing behavioural models alone. Drawing on evolutionary game theory, cultural evolution, and Large Language Models (LLMs) powered simulations, we argue that these developments open a new research agenda for social physics centred on the co-evolution of humans and machines. We outline six key research directions. First, modelling the evolutionary dynamics of social behaviours (e.g. cooperation, fairness, trust) in hybrid human-AI populations. Second, understanding machine culture: how AI systems generate, mediate, and select cultural traits. Third, analysing the co-evolution of language and behaviour when LLMs frame and participate in decisions. Fourth, studying the evolution of AI delegation: how responsibilities and control are negotiated between humans and machines. Fifth, formalising and comparing the distinct epistemic pipelines that generate human and AI behaviour. Sixth, modelling the co-evolution of AI development and regulation in a strategic ecosystem of firms, users, and institutions. Together, these directions define a programme for using social physics to anticipate and steer the societal impact of advanced AI. 

---
# A Novel end-to-end Digital Health System Using Deep Learning-based ECG Analysis 

**Authors**: Artemis Kontou, Natalia Miroshnikova, Costakis Matheou, Sophocles Sophocleous, Nicholas Tsekouras, Kleanthis Malialis, Panayiotis Kolios  

**Link**: [PDF](https://arxiv.org/pdf/2603.16891)  

**Abstract**: This study presents AI-HEART, a cloud-based information system for managing and analysing long-duration ambulatory electrocardiogram (ECG) recordings and supporting clinician decision-making. The platform operationalises an end-to-end pipeline that ingests multi-day three-lead ECGs, normalises inputs, performs signal preprocessing, and applies dedicated deep neural networks for wave delineation, noise/quality detection, and beat- and rhythm-level multi-class arrhythmia classification. To address class imbalance and real-world signal variability, model development combines large clinically annotated datasets with expert-in-the-loop curation and generative augmentation for under-represented rhythms. Empirical evaluation on three-lead ambulatory ECG data shows that delineation accuracy is sufficient for automated interval measurement, noise detection reliably flags poor-quality segments, and arrhythmia classification achieves high specificity with clinically useful macro-averaged performance across common and rarer rhythms. Beyond predictive accuracy, AI-HEART provides a scalable deployment approach for integrating AI into routine ECG services, enabling traceable outputs, audit-friendly storage of recordings and derived annotations, and clinician review/editing that captures feedback for controlled model improvement. The findings demonstrate the technical feasibility and operational value of a noise-aware AI-ECG platform as a digital health information system. 

---
# Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2603.16889)  

**Abstract**: Reliable and interpretable automated assessment of second-language (L2) speech remains a central challenge, as large speech-language models (SpeechLLMs) often struggle to align with the nuanced variability of human raters. To address this, we introduce a rubric-guided reasoning framework that explicitly encodes multi-aspect human assessment criteria: accuracy, fluency, and prosody, while calibrating model uncertainty to capture natural rating variability. We fine-tune the Qwen2-Audio-7B-Instruct model using multi-rater human judgments and develop an uncertainty-calibrated regression approach supported by conformal calibration for interpretable confidence intervals. Our Gaussian uncertainty modeling and conformal calibration approach achieves the strongest alignment with human ratings, outperforming regression and classification baselines. The model reliably assesses fluency and prosody while highlighting the inherent difficulty of assessing accuracy. Together, these results demonstrate that rubric-guided, uncertainty-calibrated reasoning offers a principled path toward trustworthy and explainable SpeechLLM-based speech assessment. 

---
# Multi-Agent Reinforcement Learning for Dynamic Pricing: Balancing Profitability,Stability and Fairness 

**Authors**: Krishna Kumar Neelakanta Pillai Santha Kumari Amma  

**Link**: [PDF](https://arxiv.org/pdf/2603.16888)  

**Abstract**: Dynamic pricing in competitive retail markets requires strategies that adapt to fluctuating demand and competitor behavior. In this work, we present a systematic empirical evaluation of multi-agent reinforcement learning (MARL) approaches-specifically MAPPO and MADDPG-for dynamic price optimization under competition. Using a simulated marketplace environment derived from real-world retail data, we benchmark these algorithms against an Independent DDPG (IDDPG) baseline, a widely used independent learner in MARL literature. We evaluate profit performance, stability across random seeds, fairness, and training efficiency. Our results show that MAPPO consistently achieves the highest average returns with low variance, offering a stable and reproducible approach for competitive price optimization, while MADDPG achieves slightly lower profit but the fairest profit distribution among agents. These findings demonstrate that MARL methods-particularly MAPPO-provide a scalable and stable alternative to independent learning approaches for dynamic retail pricing. 

---
# PowerModelsGAT-AI: Physics-Informed Graph Attention for Multi-System Power Flow with Continual Learning 

**Authors**: Chidozie Ezeakunne, Jose E. Tabarez, Reeju Pokharel, Anup Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2603.16879)  

**Abstract**: Solving the alternating current power flow equations in real time is essential for secure grid operation, yet classical Newton-Raphson solvers can be slow under stressed conditions. Existing graph neural networks for power flow are typically trained on a single system and often degrade on different systems. We present PowerModelsGAT-AI, a physics-informed graph attention network that predicts bus voltages and generator injections. The model uses bus-type-aware masking to handle different bus types and balances multiple loss terms, including a power-mismatch penalty, using learned weights. We evaluate the model on 14 benchmark systems (4 to 6,470 buses) and train a unified model on 13 of these under N-2 (two-branch outage) conditions, achieving an average normalized mean absolute error of 0.89% for voltage magnitudes and R^2 > 0.99 for voltage angles. We also show continual learning: when adapting a base model to a new 1,354-bus system, standard fine-tuning causes severe forgetting with error increases exceeding 1000% on base systems, while our experience replay and elastic weight consolidation strategy keeps error increases below 2% and in some cases improves base-system performance. Interpretability analysis shows that learned attention weights correlate with physical branch parameters (susceptance: r = 0.38; thermal limits: r = 0.22), and feature importance analysis supports that the model captures established power flow relationships. 

---
# A foundation model for electrodermal activity data 

**Authors**: Leonardo Alchieri, Matteo Garzon, Lidia Alecci, Francesco Bombassei De Bona, Martin Gjoreski, Giovanni De Felice, Silvia Santini  

**Link**: [PDF](https://arxiv.org/pdf/2603.16878)  

**Abstract**: Foundation models have recently extended beyond natural language and vision to timeseries domains, including physiological signals. However, progress in electrodermal activity (EDA) modeling is hindered by the absence of large-scale, curated, and openly accessible datasets. EDA reflects sympathetic nervous system activity and is widely used to infer cognitive load, stress, and engagement. Yet very few wearable devices provide continuous, unobtrusive sensing, and the only large-scale archive to date is proprietary. To address this gap, we compile EDAMAME, a collection of EDA traces from 24 public datasets, comprising more than 25,000 hours from 634 users. Using this resource, we train UME, the first dedicated foundation model for EDA. In eight out of ten scenarios, UME outperforms baselines and matches generalist timeseries foundation models while using 20x fewer computational resources. Our findings, however, also highlight the intrinsic challenges of EDA modeling, motivating further research to unlock its full potential. All datasets, model weights, and code are released to support further research. 

---
# Multi-Modal Multi-Agent Reinforcement Learning for Radiology Report Generation: Radiologist-Like Workflow with Clinically Verifiable Rewards 

**Authors**: Kaito Baba, Satoshi Kodera  

**Link**: [PDF](https://arxiv.org/pdf/2603.16876)  

**Abstract**: We propose MARL-Rad, a novel multi-modal multi-agent reinforcement learning framework for radiology report generation that coordinates region-specific agents and a global integrating agent, optimized via clinically verifiable rewards. Unlike prior single-model reinforcement learning or post-hoc agentization of independently trained models, our method jointly trains multiple agents and optimizes the entire agent system through reinforcement learning. Experiments on the MIMIC-CXR and IU X-ray datasets show that MARL-Rad consistently improves clinically efficacy (CE) metrics such as RadGraph, CheXbert, and GREEN scores, achieving state-of-the-art CE performance. Further analyses confirm that MARL-Rad enhances laterality consistency and produces more accurate, detail-informed reports. 

---
# Attention Guidance through Video Script: A Case Study of Object Focusing on 360° VR Video Tours 

**Authors**: Paulo Vitor Santana Silva, Arthur Ricardo Sousa Vitória, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2603.16875)  

**Abstract**: Within the expansive domain of virtual reality (VR), 360° VR videos immerse viewers in a spherical environment, allowing them to explore and interact with the virtual world from all angles. While this video representation offers unparalleled levels of immersion, it often lacks effective methods to guide viewers' attention toward specific elements within the virtual environment. This paper combines the models Grounding Dino and Segment Anything (SAM) to guide attention by object focusing based on video scripts. As a case study, this work conducts the experiments on a 360° video tour on the University of Reading. The experiment results show that video scripts can improve the user experience in 360° VR Videos Tour by helping in the task of directing the user's attention. 

---
# Disclosure By Design: Identity Transparency as a Behavioural Property of Conversational AI Models 

**Authors**: Anna Gausen, Sarenne Wallbridge, Hannah Rose Kirk, Jennifer Williams, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2603.16874)  

**Abstract**: As conversational AI systems become more realistic and widely deployed, users are increasingly uncertain about whether they are interacting with a human or an AI system. When AI identity is unclear, users may unwittingly share sensitive information, place unwarranted trust in AI-generated advice, or fall victim to AI-enabled fraud. More broadly, a persistent lack of transparency can erode trust in mediated communication. While regulations like the EU AI Act and California's BOT Act require AI systems to identify themselves, they provide limited guidance on reliable disclosure in real-time conversation. Existing transparency mechanisms also leave gaps: interface indicators can be omitted by deployers, and provenance tools require coordinated infrastructure and cannot provide reliable real-time verification.
We ask how conversational AI systems should maintain identity transparency as human-AI interactions become more ambiguous and diverse. We advocate for disclosure by design, where AI systems explicitly disclose their artificial identity when directly asked. Implemented as model behaviour, disclosure can persist across deployment contexts without relying on user interfaces, while preserving user agency to verify identity on demand without disrupting immersive uses like role-playing.
To assess current practice, we present the first multi-modal (text and voice) evaluation of disclosure behaviour in deployed systems across baseline, role-playing, and adversarial settings. We find that baseline disclosure rates are often high but drop substantially in role-play and can be suppressed under adversarial prompting. Importantly, disclosure rates vary significantly across providers and modalities, highlighting the fragility of current disclosure behaviour. We conclude with technical interventions to help developers embed disclosure as a fundamental property of conversational AI models. 

---
# Unsupervised learning for inverse problems in computed tomography 

**Authors**: Laura Hellwege, Johann Christopher Engster, Moritz Schaar, Thorsten M. Buzug, Maik Stille  

**Link**: [PDF](https://arxiv.org/pdf/2508.05321)  

**Abstract**: This study presents an unsupervised deep learning approach for computed tomography (CT) image reconstruction, leveraging the inherent similarities between deep neural network training and conventional iterative reconstruction methods. By incorporating forward and backward projection layers within the deep learning framework, we demonstrate the feasibility of reconstructing images from projection data without relying on ground-truth images. Our method is evaluated on the two-dimensional 2DeteCT dataset, showcasing superior performance in terms of mean squared error (MSE) and structural similarity index (SSIM) compared to traditional filtered backprojection (FBP) and maximum likelihood (ML) reconstruction techniques. Additionally, our approach significantly reduces reconstruction time, making it a promising alternative for real-time medical imaging applications. Future work will focus on extending this methodology to three-dimensional reconstructions and enhancing the adaptability of the projection geometry. 

---
