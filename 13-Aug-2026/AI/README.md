# Constructing Dynamic Master Logic Models as Knowledge Graphs for Complex System Diagnostics Using Retrieval-Augmented Large Language Models 

**Authors**: Saman Marandi, Yu-Shu Hu, Mohammad Modarres  

**Link**: [PDF](https://arxiv.org/pdf/2608.12304)  

**Abstract**: Dynamic Master Logic (DML) provides a hierarchical framework for representing system behavior by linking functional objectives to underlying structural elements. However, DML construction typically relies on expert interpretation of technical documentation, limiting scalability for complex systems. This study presents a framework for automated construction of DML models from system descriptions and their representation as Knowledge Graphs (KG-DML), using Retrieval-Augmented Generation and Large Language Models as enabling tools. Building on prior work with small-scale systems, the framework extends automated KG-DML construction and evaluation to substantially larger and more complex systems. Model construction proceeds across the DML hierarchy using targeted retrieval while preserving functional dependencies and explicit logical relationships. The resulting KG-DML supports diagnostic reasoning, safety assessment, upward failure propagation, and downward dependency tracing. A multi-level validation methodology evaluates layer-specific precision and recall, logical gate consistency, and overall structural integrity. Application to the Low-Pressure Coolant Injection system of a decommissioned Boiling Water Reactor demonstrates consistent reconstruction across repeated runs. The results show that automated KG-DML construction can transform technical documentation into executable functional models for diagnostic and reliability analysis. 

---
# VAKRA: Evaluating Multi-Hop Reasoning Across APIs and Retrieval Under Tool-Use Policies 

**Authors**: Ankita Rajaram Naik, Anupama Murthi, Benjamin Elder, Siyu Huo, Raavi Gupta, Abhinav Jain, Praveen Venkateswaran, Abdulhamid Adebayo, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2608.12282)  

**Abstract**: Agents deployed in enterprise settings must reason across structured APIs and document collections, yet existing benchmarks evaluate these capabilities in isolation. We introduce VAKRA (e\textbf{V}aluating \textbf{A}PI and \textbf{K}nowledge \textbf{R}etrieval \textbf{A}gents), a benchmark of over $8{,}000$ executable APIs across $62$ domains with tasks spanning three settings of increasing difficulty: diverse API interaction styles, multi-hop reasoning over structured APIs, and multi-source reasoning with natural-language tool-use policy constraints. Correctness is verified by re-executing predicted tool calls against live APIs, accommodating multiple valid paths. Using a fixed ReAct harness to isolate model capabilities from agent architecture, we evaluate frontier and open-weight models and find that even the best model achieves only 70.4\% on single-hop endpoint-style tasks and drops to 50--51\% on compositional APIs; performance degrades by over 50\% as reasoning depth increases, and policy-constrained questions expose severe failures (as low as 2.4\% on unanswerable queries). Trace analysis shows failures concentrate at language-mediated reasoning - entity disambiguation, cross-source grounding, rather than tool invocation mechanics. Code is available this https URL. Dataset is available this https URL 

---
# An Agentic Workflow for Legacy HPC Modernization: Converting the Two-Electron-Integral Core of GAMESS 

**Authors**: Yuzhong Shen, Masha Sosonkina, Peng Xu, Mark S. Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2608.12249)  

**Abstract**: Modernizing legacy Fortran is a problem of volume: the transformations are individually routine, but the codebases can be enormous, and across much of computational science the work simply goes undone. We propose an agentic workflow that takes this work on at production scale, and we set out to measure how far such delegation can reach. In this work, three prompt-specialized agent roles operate under a version-controlled specification that the agents themselves authored and revised, while humans hold a small number of gates. The arrangement is kept safe by an exact verification oracle inherited from the domain, and the boundary of safe delegation lies exactly where that oracle stops seeing.
We apply the proposed workflow in a case study, converting the two-electron-integral routines of GAMESS (General Atomic and Molecular Electronic Structure System), a mature quantum-chemistry package with a 48-year development history, from fixed-form Fortran 77 to free-form Fortran 2008. The scope of this work was twelve source files, 56,448 lines, and 225 subroutines for computing electron repulsion integrals. The agents ran as three Claude Code roles in isolated worktrees, and the work spanned four Claude model generations. Because the GAMESS group ships a standard test suite whose printed energies its user community treats as canonical, we could adopt bit-for-bit reproduction of those energies as the merge criterion, where a deviation in the twelfth decimal place counts as a failure rather than drift. All twelve source files pass a 51-test validation battery comprising the 49 standard GAMESS tests and two additional calculations, and across 612 test runs the number of chemistry-relevant differences is zero, and every file also passes the Jenkins tests that are used for continuous integration. 

---
# How to Spend Your Oracle Budget: Practical Guidance for Protein Structure Prediction Models 

**Authors**: Aleksandra Kalisz, Jack Simons, Krisztina Sinkovics, Noam Ghenassia, Shikha Surana, Henry Moss, Paul Duckworth  

**Link**: [PDF](https://arxiv.org/pdf/2608.12192)  

**Abstract**: Foundation models for protein structure prediction remain unreliable on certain targets. External oracles can flag and correct these failures, but biological oracles are expensive, making oracle budget a critical constraint. Existing guidance methods, such as FK-steering, DPO, and Best K-of-N sampling, differ in how they spend this budget, yet no systematic comparison exists to guide method selection. To bridge this gap, we benchmark these methods alongside the recently proposed Optimisation Over Outputs (O3), which applies off-the-shelf optimisers within a generative model's latent subspace. We extend the usage of O3 to protein structure prediction models. Overall, our work provides the first practical reference for oracle budget-aware guidance. Our evaluation on two protein targets, calmodulin (1CLL) and E. coli aspartate transcarbamoylase (9EEH), reveals that no single method consistently dominates across all budgets and oracles. Specifically, O3 proves most effective at low oracle budgets, while FK-steering and DPO demonstrate improved performance as the budget increases. We distil these findings into actionable recommendations for practitioners operating under real-world oracle-budget constraints. 

---
# Who Thinks Best Depends on How Long You Let Them: Budget-Dependent Rankings in LLM Evaluation 

**Authors**: Rodrigo Guedes de Souza, Alison R. Panisson  

**Link**: [PDF](https://arxiv.org/pdf/2608.12150)  

**Abstract**: Standard evaluation of large language models assumes stable model rankings across inference conditions. We challenge this assumption by varying the token generation budget, i.e., the maximum tokens a model may produce, across seven levels (64--4,096), evaluating four models on three reasoning benchmarks (56,476 inferences). We report four findings: (i) 3--19% of items exhibit non-monotone behavior (accuracy decreasing with more budget), even after controlling for truncation, and this phenomenon is model-specific (cross-model overlap: 6--14%). (ii) Model rankings reverse across budgets on all benchmarks ($p {<} 0.01$, McNemar). (iii) Oracle analysis reveals model complementarity up to $+27.8$pp, most pronounced at constrained budgets. (iv) A budget-aware router captures 14.1% of the oracle gap cross-domain; budget features help within-domain ($+1.6$ to $+5.7$pp) but are domain-specific and hurt transfer ($-1.2$pp). These results argue for budget-conditioned evaluation protocols. 

---
# GUIDE: Governed Unified Intelligence for Document-to-Artifact Generation in Enterprise Settings 

**Authors**: Shivali Dalmia, Sumukha Thoppanahalli, Mohammadreza Sediqin, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2608.12133)  

**Abstract**: Enterprise guideline documents are heterogeneous and multimodal, combining narrative text, complex tables, and embedded images. Existing LLM and VLM systems face hallucinated content, table structure degradation, and lack governed workflows extending beyond extraction to validation and artifact generation. This leaves enterprises to perform this manually, consuming 2-3 days per document. To address this, we introduce GUIDE, a governed multi-agent framework built on a shared versioned rule store with schema-validated inter-agent contracts and end-to-end provenance tracking. Six specialized agents handle parsing, VLM-driven extraction, consistency checking, evaluation, human-in-the-loop (HITL) escalation, and persona-tailored artifact synthesis. Evaluated on 120 real-world enterprise guideline documents, GUIDE achieves 96% document success, extracts 3,896 rules with 71.4% auto-approved, produces 812 deployment-ready artifacts, and reduces turnaround to 40-125 minutes per document. 

---
# Graph-Structured Rubrics: Compiling Rubrics into Typed Evaluation Graphs for LLM Judges 

**Authors**: Xi Chen, Jie Mu, Mo Xuan, Qun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.12097)  

**Abstract**: Rubric-based evaluators commonly treat rubrics as prompt context or flat criteria: they specify what to judge but leave criterion composition implicit, even when natural-language rules state it. We introduce Graph-Structured Rubrics (GSR), which compiles a rubric into a response-independent typed evaluation graph before observing responses. Criterion nodes elicit judgments; transformation, reduction, and gating operators compose them through named ports; and a task-specific output mapping, termed Readout, converts the unique sink into a score or preference. Compilation rejects malformed or type-incompatible graphs. Pointwise evaluation judges rubric dimensions separately before graph aggregation; pairwise evaluation reuses the graph with one judgment for each candidate under every criterion. Under GPT-OSS-120B, GSR improves exact score agreement by 0.62--6.75 percentage points over Prometheus-style scoring on four pointwise datasets and achieves the numerically highest end-to-end pairwise accuracy on two preference benchmarks under native tie and abstention policies. 

---
# Mechanist: AI as a Scientific Instrument for Discovering the Mechanisms of Intelligence 

**Authors**: Mengru Wang, Junfeng Fang, Shuofei Qiao, Zhenqian Xu, Haoming Xu, Haoxiong Wang, Shumin Deng, Linyi Yang, Zhixiang Cui, Xin Xu, Yunzhi Yao, Buqiang Xu, Fei Shen, Haozhe Luo, Yunxiang Wei, Ningyu Zhang, Julian McAuley, Tat Seng Chua, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12036)  

**Abstract**: AI models have achieved remarkable success across diverse domains, yet the mechanisms underlying their capabilities and the risks they may pose remain poorly understood. As AI development becomes faster and increasingly automated, mechanistic exploration remains largely manual, widening the gap between what models can do and our ability to understand and control them. To bridge this gap, we introduce Mechanist, an agentic system that uses AI as a scientific instrument for the autonomous discovery of mechanisms underlying AI intelligence. To support autonomous mechanistic discovery, we construct an interpretability-focused knowledge graph of approximately 13,000 papers and integrate it with a multidisciplinary database of 43 million papers spanning 26 fields. We further curate a library of 32 foundational methods for mechanism analysis, causal intervention, and validation. Compared with Claude Code and existing AI-scientist systems, Mechanist generates more valuable mechanism hypotheses and executes experiments more reliably. Mechanist also demonstrates a progression from discovering model behaviors to explaining and controlling AI models. Specifically, Mechanist first uncovers a counterintuitive safety risk in scientific laboratories, showing that unsafe traits can transfer across modalities through apparently safe training data. Mechanist then develops a mechanism theory of belief, revealing how models represent world knowledge, form beliefs, infer the beliefs of others, and how these mechanisms emerge during pretraining. Finally, Mechanist translates these mechanistic insights into practical interventions that improve model performance across diverse scenarios and steer scientific foundation models toward generating DNA sequences with specified properties. 

---
# CTBench: Evaluating Troubleshooting Capabilities of AI Agents in Realistic Telecom Network Operations 

**Authors**: Xingyu Yan, Tingting Dai, Antonio De Domenico, Mohamed Sana, Nicola Piovesan, Changchang Li, Bowen Liu, Kun Jiang, Mengjie Zhang, Dingcheng Shan, Jing-Cheng Pang, Chenwei Wu, Sijie Wu, Lianying Chao, Haoran Cai, Jiantao Ye, Xubin Li, Simon Mark Lucas, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12002)  

**Abstract**: Agents are increasingly considered for automating network operations and maintenance, where engineers must diagnose network faults, optimize configurations to enhance services, and reduce operational costs while acting under strict constraints. However, existing evaluations fail to accurately model real network characteristics or assess agents under partially observable telecom environments with diverse vendors, devices, protocols, and interfaces. In this paper, we introduce CTBench, a public benchmark for assessing whether an agent behaves like a competent telecom troubleshooting engineer. CTBench focuses on root cause analysis and path restoration. Each task is constructed by experts and annotated with rich task metadata, including golden evidence steps. CTBench uses expert-grounded metrics that evaluate both final answers and the diagnostic evidence. Experiments with representative harness-model combinations show that state-of-the-art agents perform very well at identifying endpoints in path-restoration tasks but, more generally, underperform in root cause analysis. In particular, agents struggle with interface state, link-layer, service-management, and other operational faults. Most importantly, even when agents produce plausible or correct final answers, they often fail to provide the evidence-grounded diagnoses required in operational practice. Our results further show that path restoration is generally more resource expensive, yet larger resource usage does not necessarily translate into better diagnosis. 

---
# Claim-Level Reliability Assessment for Efficient Test-Time Reasoning 

**Authors**: Sen Xu, Wei Wang, Shixi Liu, Jixin Min, Yingwei Dai, Zhibin Yin, Yirong Chen, Junlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11994)  

**Abstract**: We propose claim-level falsification as a principle for test-time scaling and instantiate it through Claim-Level Reliability Assessment (CLR), a training-free framework that reallocates test-time compute from additional solution sampling to targeted verification. Since whole-trace evaluation often obscures decisive errors due to signal dilution from routine tokens, CLR condenses each reasoning trace into a compact set of decision-critical claims, thereby isolating its logical anchors. Furthermore, recognizing the inherent difficulty of generating entirely correct solutions under fixed model capabilities, CLR shifts the focus to semantic falsification. This approach exploits a fundamental asymmetry between solution construction and claim refutation. Constructing a valid solution requires a flawless reasoning path, whereas refuting an incorrect claim requires identifying only a single decisive flaw. This targeted search for negative evidence systematically compresses the survival space of high-confidence incorrect traces, effectively suppressing erroneous consensus via nonlinear reliability scoring. Across four LLMs and four reasoning benchmarks under matched budgets, CLR generally improves upon pass@1 and self-consistency. On GPT-OSS-20B/CMIMC25, for instance, CLR exceeds pass@1 by 27.15 percentage-points and raises self-consistency accuracy from 77.50\% to 82.19\% with 37.0\% fewer tokens. 

---
# Retry, Switch, or Abstain? Learning Strategy-Aware Tool-Use Policies via Controlled Error Injection 

**Authors**: Chaoran Chen, Vy Nguyen, Ziji Zhang, Abhinav Gullapalli, Ziyi Wang, Yuxuan Lu, Dakuo Wang, Jing Huang, Zhou Yu, Jin Lai  

**Link**: [PDF](https://arxiv.org/pdf/2608.11977)  

**Abstract**: Tool-using LLM agents are commonly trained and evaluated in environments where tool calls succeed reliably, yet deployed tools can fail transiently, persistently, or silently. Robust recovery therefore requires more than repeated retries: an agent may need to retry the same path, switch to an alternative, or recognize that no viable path remains. We present BENCH2ROBUST, a framework that converts failure-free tool-use benchmarks into controlled stochastic environments with scenario-controlled solvability, where episodes explicitly require retrying, switching, or stopping after available paths are exhausted. We use BENCH2ROBUST to study two complementary interventions: structured runtime recovery context through Bayesian Tool Memory (BTM), and curriculum-controlled reinforcement learning. Across 7 models from 4 families and two multi-turn benchmark families, tool failures produce a near-universal robustness gap. On held-out Retail tasks, BTM improves robustness by up to 16.8 percentage points without retraining, while RL learns complementary recovery behavior that remains beneficial without inference-time BTM. Combining the two reaches 40.8-45.5% under injection while preserving failure-free performance. These results suggest that robust tool use benefits from combining environment-specific recovery knowledge with learned recovery behavior. 

---
# ExRole: From Team Trajectories to Executable Roles in Multi-Agent Language Models 

**Authors**: Zhou Liu, Chaoyang Han, Zewei Pan, Zeli Su, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11949)  

**Abstract**: Roles provide an interpretable interface for organizing language-model agents, yet most multi-agent systems treat them as hand-written prompt labels disconnected from learned behavior and parameter updates. We argue that a useful role should instead be an executable control variable: it should summarize behavior predictive of future utility, guide subsequent interaction, and identify the trainable capacity responsible for that behavior. We introduce ExRole, a trajectory-to-role framework that learns future-aware role prototypes from prefix-local team traces, resolves them into readable instructions and token-aligned role markers, and optionally routes shared LoRA rank slots with turn-aligned credit. Across MuSiQue and 2WikiMultiHopQA, ExRole improves over single-agent search by 15.0/14.4 and 13.5/16.1 EM/F1 points, respectively. Against the strongest non-ExRole controls, the corresponding gains remain 11.5/11.6 and 7.7/9.7 points. Across both benchmarks, the controlled results consistently favor trajectory-induced role conditioning over role-free, manual, random, and shuffled alternatives. Role-Agent-Turn interventions further show that the induced roles capture transferable behavioral specialization beyond fixed agent identities or turn positions. 

---
# OEIS Open: How many conjectures can language models turn into theorems? 

**Authors**: Tom Adamczewski  

**Link**: [PDF](https://arxiv.org/pdf/2608.11941)  

**Abstract**: We construct OEIS Open, a benchmark based on 492 open mathematical conjectures from the OEIS, formalized in Lean by Tsoukalas et al. Whereas these conjectures had previously been attempted only with a bespoke agent, our open-source evaluation code runs any generic language model (LM) against them, and is secure against LM cheating attempts. We find that LMs equipped with a minimal set of tools resolve 147 of these conjectures with a budget of \$50 per attempt, scoring 30% on OEIS Open. OEIS Open Lite is a random subset of 100 conjectures for cheaper evaluation. When evaluated with a budget of \$200 per attempt, the best current LM scores 44% on OEIS Open Lite. Giving LMs access to the mathematics literature via 476,000 papers from arXiv did not increase performance on OEIS Open Lite, and nor did using more sophisticated agent loops. The conjectures covered in this work are of uncertain mathematical significance, and most have likely received little previous attention. Nevertheless, our results show that LMs can resolve open research conjectures autonomously and at modest cost. 

---
# Policy-as-logic for robust reasoning over rules 

**Authors**: Rahul Nair, Bastian Lipka, Elizabeth Daly  

**Link**: [PDF](https://arxiv.org/pdf/2608.11905)  

**Abstract**: In many practical applications of generative AI systems, from tax rules to airline baggage allowance, responses to natural language queries must respect written policies or rules. We present a hybrid symbolic approach that expresses policies in formal logic and at inference time exploits the representation power of language models for fact extraction to ground predicates, and an answer set solver for reasoning such that responses are interpretable, auditable, and as we show, accurate and robust under input perturbations. Specifically, we show this separation of extraction and reasoning steps outperforms policy-as-prompt and policy-as-code methods in most cases with ~10x reduction in token usage. The results point to the value of structured reasoning and symbolic solvers in conjunction with generative models to make robust decisions involving objective criteria. 

---
# Agent Skills Can Be Harmful: An Empirical Study of Skill-Induced Failures in LLM Agents 

**Authors**: Gen Dong, Yanjie Gao, Liqun Li, Tianyin Xu, Yu Hua, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11888)  

**Abstract**: Agent skills are the de facto mechanism for extending LLM agents with reusable guidance. A skill can shape the agent's task execution, including planning, tool use, problem-solving, and validation. Prior work reported mixed results of agent skills: some skills improve task success rates, while others have no effect, increase token use and execution time, and even reduce success rates. This paper presents a comprehensive analysis of skill-induced agent failures by attributing task failures and cost regressions to specific loaded skills. We introduce a differential analysis framework that attributes a failure or regression to a skill by comparing a target skill-guided run against a no-skill or semantically matched skill reference run that solves the same task, or solves it more cheaply. We instantiate this framework on SkillsBench and SWE-Skills-Bench, yielding 307 skill-induced failures, including 125 functional failures and 182 efficiency regressions. We also build SkillTriage, a taxonomy-guided attribution tool that normalizes paired cases, extracts differential evidence, and produces triage reports. Our major findings include: (1) Skill induced functional failures are rarely caused by obviously irrelevant skills; instead, seemingly relevant skills often make the agent incorrectly implement or omit task-required implementation elements. (2) Skill-induced efficiency regressions are not explained by prompt length alone. (3) The largest sources within Excessive Procedure are excessive verification and heavy implementation pipelines, contributing 67 and 30 cases, respectively. This shows that skills often turn validation checklists and construction recipes into mandatory work. Based on our findings, we propose research topics and tooling improvements for safer and more cost-aware skill reuse. 

---
# The Sleeping Agent: What Gist-Based Context Compression Loses and Why 

**Authors**: Nicholas E. Kyrkewood  

**Link**: [PDF](https://arxiv.org/pdf/2608.11775)  

**Abstract**: Gist-based context compression---summarising older conversation history into compact representations---is a common approach in long-horizon language model agents, yet its effect on different types of memory retrieval is poorly understood. We use Salience-Weighted Consolidation (SWC), a biologically-inspired compression framework motivated by sleep-based memory consolidation, as a diagnostic probe to study when gist compression helps and when it hurts. SWC scores conversation history by salience, partitions it into priority tiers, and applies structured gist abstraction to mid-priority content. Evaluating four conditions on all ten LoCoMo conversations---1,935 matched text-only questions in total, 1,501 used in the primary aggregate after excluding Category 5 (adversarial) questions---at temperature 0, we find a consistent task-type interaction: gist compression substantially outperforms truncation on multi-hop reasoning and single-hop factual questions, but temporal questions remain substantially harder under compression, with compressed conditions scoring well below the full-context reference on the conversations where both are evaluated. We trace this failure to a specific mechanism: the gist abstraction prompt preserves relational and event structure while discarding dates and times. A preservation analysis across all ten conversations confirms the mechanism: an approximately 20-fold increase in temporal expression preservation (3.05% to 62.39%) with a one-sentence prompt modification, while named entity and event preservation rates barely change (x1.02 and x1.11), demonstrating that the fix is a precision instrument. The prompt modification recovers +0.314 [0.254, 0.375] judge accuracy on category-2 (temporal) questions in the matched set. Code and results: this https URL. 

---
# HyperANFIS: Enhancing Rule Representation and Interpretability in Adaptive Neuro-Fuzzy Systems via Hyperbolic Geometry 

**Authors**: Haoran Pei, Zhao Su, Zetao Lin, Haoran Li, Jun Shen, Qi Zhu, Lan Guo, Qingguo Zhou, Binbin Yong  

**Link**: [PDF](https://arxiv.org/pdf/2608.11768)  

**Abstract**: The adaptive neuro-fuzzy inference system (ANFIS) is an interpretable reasoning framework capable of generating explicit IF-THEN fuzzy rules, making it suitable for tasks requiring transparent reasoning. However, existing ANFIS models generally construct rule antecedents and perform inference in Euclidean space, limiting their representational capacity and predictive performance. To address this issue, we propose Hyperbolic ANFIS (HyperANFIS), a hyperbolic extension of ANFIS. HyperANFIS preserves the fuzzy semantics and core architecture of conventional ANFIS while performing rule-prototype learning, rule activation, and consequent aggregation in hyperbolic space. It also retains the ability to generate interpretable IF-THEN rules. By exploiting the representational properties of hyperbolic geometry, HyperANFIS strengthens the fuzzy inference process, thereby improving predictive accuracy, inter-rule collaboration, and the credibility of its interpretable rules. Experimental results show that HyperANFIS consistently outperforms the standard ANFIS baseline and various ANFIS variants across all datasets, while also generating higher-quality fuzzy rules. 

---
# Harness-IF: Evaluating Instruction Following Across Instruction Surfaces in Coding Agents 

**Authors**: Zining Huang, Haoran Que, Hong Zeng, Ge Zhang, Zuo Wang, Jin Chen, Haodong Wang, Zhongfei Hou, Changxin Pu, Shen Yan, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11727)  

**Abstract**: When a coding agent obeys a rule, it may simply have been going to do that anyway. Existing instruction-following benchmarks cannot tell the difference: they concentrate rules in the user turn, while coding-agent benchmarks emphasize final task success. We introduce Harness-IF, which scores operational rules one at a time from execution evidence: 60 realistic multi-turn coding items drawn from a 642-rule library, 256 rules receiving verdicts, placed on the five configurable surfaces a deployed agent reads. To separate compliance from coincidence we introduce Against-Prior Accuracy (AP-Acc), which scores only rules labeled as opposing unprompted defaults, observed by re-running tasks with the rule withheld across nine probe builds and curated otherwise. Across 12 frontier models, accuracy spans 72.1-85.9% and AP-Acc 66.1-78.6%; every model is worse on against-prior rules, by 3.6 to 7.4 points (mean 5.81), and the direction survives a common-support analysis with item-clustered intervals. Aggregate scores therefore overstate compliance by a model-specific margin: prior control leaves the top build unchanged and exchanges three adjacent rank pairs. A counterbalanced conflict pilot on nine separate builds adds a second result: pooled precedence does not follow prompt depth, with system prompts, project files, and user instructions ahead of tool and skill descriptions. 

---
# Proportional Analogies on Probability Distributions via Bayesian Updating 

**Authors**: Pierre-Alexandre Murena  

**Link**: [PDF](https://arxiv.org/pdf/2608.11724)  

**Abstract**: Analogies are quaternary relations of the form "A is to B as C is to D". Among the various formalizations of analogical reasoning, proportional analogies provide an important axiomatic framework by characterizing valid analogies through a set of postulates. While proportional analogies have been extensively studied over Boolean, symbolic, and real-valued domains, their extension to probability distributions remains largely unexplored. In this paper, we introduce a notion of proportional analogy for probability distributions based on Bayesian updating. Our approach builds upon the idea that two distributions are related whenever one can be transformed into the other through Bayesian updating induced by a suitable set of observations. We investigate this framework for several standard members of the exponential family and discuss how it naturally extends to arbitrary probability distributions through Gaussian mixture approximations. 

---
# Making Your LLMs More Objective: Stabilizing LLM Safety Behavior Across Traits with Trait-Invariant Safety Tuning 

**Authors**: Lang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11705)  

**Abstract**: Aligned large language models (LLMs) are expected to exhibit safety behavior based on the content of the user request: they should refuse unsafe requests and comply with safe ones. However, we show that the same request can elicit substantially different safety decisions under different traits assigned in the system prompt, a failure mode we call trait-induced safety variation. To measure this failure, we introduce refusal-based metrics: Trait-Induced Deviation measures dataset-level deviation from the no-trait baseline, while Trait-Induced Flip Rate measures whether the same request receives different safety decisions across traits. We then provide a representation-level analysis of the mechanism behind trait-induced safety shifts and find that traits perturb the model's safety representations within a low-dimensional subspace. To achieve trait-invariant safety, where safety behavior remains stable across traits, we introduce Trait-Invariant Safety Tuning (TIST), a simple yet effective self-distillation framework that aligns an LLM's trait-conditioned behavior with its no-trait behavior. Guided by our analysis, we further propose Trait-Subspace Neutralization (TraSN), an instantiation of TIST, which enforces invariance only within the identified trait subspace. Experiments show that TraSN improves trait-invariant safety and strengthens harmful-request safety while preserving general capability. Our results highlight traits as an important factor in LLM safety and robust model behavior. 

---
# HUGIN: Enhancing Vision-Language Planning for Autonomous Logistics Sorting 

**Authors**: Xikai Sun, Cangtian Zhou, Kebin Liu, Ke Ma, Xu Wang, Zaishu Chen, Haotian Wang, Li Liu, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11692)  

**Abstract**: Autonomous logistics sorting systems (ALSS) are an important industrial application of embodied AI, which requires joint planning over spatially disjoint camera views. We formulate this setting as Joint Multi-Scene Understanding (JMSU). With open-world visual understanding and task-planning capabilities, vision-language models (VLMs) are promising candidates for JMSU. However, directly applying existing VLMs to JMSU is non-trivial due to scarce cross-scene supervision and attention dispersion caused by long visual context in JMSU. To address these challenges, we propose HUGIN, a training framework with two complementary components. Endogenous Data Augmentation recombines verified atomic facts under operating constraints, while Global Context Ranking aligns the instruction representation more strongly with the complete visual context than with a partial visual context. To support ongoing research, we construct a high-quality industrial sorting dataset and benchmark named SortingBench from four layouts of autonomous logistics sorting systems. Across five open VLMs, HUGIN consistently outperforms matched baselines; for example, the accuracy on SortingBench of Qwen3-VL-8B increases from 63.6% to 78.8%. Additional experiments verify the effectiveness of each component and JMSU's spillover benefits in embodied tasks. Deployment tests involving more than 15,000 packages support the practical viability of VLM-based planning for autonomous logistics sorting. 

---
# FrontierFinance: A Challenging Benchmark for Measuring Frontier Intelligence of Finance Agents 

**Authors**: Yuhao Zhang, O. Ozan Koyluoglu, Thejas Venkatesh, Richard Diehl Martinez, Vishank Bhatia, Arash Alidoust, Ashwin Paranjape  

**Link**: [PDF](https://arxiv.org/pdf/2608.11683)  

**Abstract**: AI agents are increasingly deployed for professional investment research, yet no benchmark captures the complexity of the full investor workflow. Existing benchmarks mainly target financial data extraction, a narrow slice that current models have largely saturated, while reference-based metrics and generic LLM-as-a-judge scoring fall short on the open-ended, long-form answers that real analyst queries demand. We introduce FrontierFinance, a fully open benchmark of 220 expert-crafted queries and 11,543 source-attributed rubrics spanning six crucial use cases across the full investor workflow. FrontierFinance is both broader and harder than existing public finance benchmarks. Evaluating frontier models and agent systems under a common harness restricted to publicly available data, we find that the tool harness, not the model alone, strongly shapes quality and efficiency; that Samaya's in-house system leads at 56.0%, ahead of the strongest frontier model (Claude Fable 5, 49.2%) at roughly 2.2x lower cost; and that the best open-weight model (Kimi K3, 46.4%) nearly matches the best proprietary model at 4.5x lower cost. Screening & Discovery and Sector, Industry & Macro remain the hardest use cases across all systems, where even the best systems reach only 33% and 39%. We make the dataset and grading code publicly available. 

---
# AgenticTwin: An Agentic LLM Framework Integrated with Digital Twin for Anomaly Detection 

**Authors**: Touseef Hasan, Mounika Ghanta, Souvika Sarkar, Ujjwal Guin  

**Link**: [PDF](https://arxiv.org/pdf/2608.11679)  

**Abstract**: Digital twins are increasingly used to monitor and simulate the behavior of cyber-physical systems. Even with skilled operators, interpreting anomalies detected within digital twin pipelines is challenging, as the sheer complexity and volume of raw sensor data make thorough analysis difficult. Recent advances in large language models (LLMs) offer promising capabilities for reasoning and explanation, yet their integration into digital twin-driven anomaly analysis remains underexplored. In this work, we propose AgenticTwin, an agentic framework that integrates LLM-driven reasoning with a digital twin-based anomaly detection pipeline. The framework grounds LLM-generated explanations in outputs from a digital twin-driven anomaly classifier and enables human operators to ask relevant natural-language questions about the system. Beyond the framework itself, we introduce a benchmark-oriented evaluation pipeline constructed over synthetic anomalies injected into a real-world weather sensor dataset, enabling controlled generation of operator queries over anomaly events. We further evaluate the feasibility of deploying lightweight, open-source LLMs for practical cyber-physical environments. Experimental results demonstrate that structured agent collaboration and knowledge-grounded reasoning improve diagnosis quality, contextual retrieval, and mitigation quality across diverse possible anomaly scenarios. 

---
# XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication 

**Authors**: Wooseong Yang, Wei-Chieh Huang, Weizhi Zhang, Yu Wang, Philip S. Yu, Junhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.11676)  

**Abstract**: Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender's internal representations, or require architectural homogeneity for latent-level transfer. We identify the entity grounding problem in cross-architecture communication: cross-attention bridges that transfer continuous representations across different LLM families suffer from rare-token compression collapse, where entity identity is lost in the continuous bottleneck (bridge-only F1 ~30%). We propose XBRIDGE, a decode-free communication protocol that addresses this through two mechanisms. Lexical Anchor Mapping (LAM) maps the sender's original context tokens to the receiver's vocabulary, providing discrete entity anchors. A Latent Enrichment Bridge (LEB) lets the receiver query the sender's hidden states for contextual enrichment. The entity anchors ground the bridge's contextual signals to specific entities through the receiver's own self-attention. Across three model families (Llama, Qwen, and Mistral), seven benchmarks, and both communication directions, XBRIDGE outperforms text-based communication on all seven tasks for each model pair while achieving 11x lower latency, and in a same-architecture setting it also exceeds a KV-sharing baseline on six of seven tasks. LEB requires only 264M trainable parameters (3.8% of the receiver), is trained on a small balanced sample set, and adds negligible inference overhead. 

---
# CLAIM: Leading Open-domain Active Clarification of Large Language Models with Uncertainty Measurement 

**Authors**: Kuangzhao Yang, Ziliang Zhao, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2608.11631)  

**Abstract**: In open-domain human-computer interaction scenarios, large language models (LLMs) frequently encounter user queries that are ambiguous or incomplete. In such cases, directly producing an answer often leads to overgeneralized, erroneous, or low-information responses. In contrast, asking clarifying questions can substantially improve interaction quality. However, existing approaches still rely heavily on manually annotated data or preference alignment to address two fundamental challenges: when clarification is necessary, and which aspect of the query should be clarified. This reliance incurs high annotation costs and limits generalization. To address these challenges, we propose CLAIM, an uncertainty-driven framework for active clarification learning in open-domain settings. CLAIM eliminates the need for explicit human preference annotations by quantifying query uncertainty through the entropy induced by answer disagreements across multiple models. This uncertainty signal is then used to construct high-quality synthetic data, enabling the training of a unified clarification decision model through a combination of supervised learning and reinforcement learning. Specifically, we propose an entropy-driven synthetic data generation pipeline that integrates entropy-based uncertainty estimation with semantic clustering and reasoning-based judgments, enabling reliable automatic annotation of clarification requirements. To train CLAIM, we formulate the clarification process as a structured decision generation problem and adopt a training paradigm that combines supervised fine-tuning (SFT) with group-relative policy optimization (GRPO). Experimental results demonstrate that CLAIM can learn stable and generalizable clarification strategies without relying on manually labeled data, offering a low-cost and robust solution for proactive understanding in real-world open-domain interactions with LLMs. 

---
# Making AI-Generated Feedback Matter: From Provision to Student Enactment 

**Authors**: Omar Alsaiari, Nilufar Baghaei, Jason M. Lodge, Dragan Gaševi'c, Naomi Winstone, Hassan Khosravi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11625)  

**Abstract**: Feedback processes strongly influence student learning, yet their educational value depends on addressing two distinct challenges: providing high-quality, timely, and individualised feedback at scale, and supporting students to interpret, evaluate, and act on that feedback productively. Generative AI offers a credible means of addressing the provision challenge, but students' uptake of AI-generated feedback remains limited. We conducted a large-scale quasi-experimental sequential cohort study comparing three AI-mediated feedback workflows across 13,037 students and 51,296 student-authored resources. In Directed Feedback (n = 3,723), students received AI-generated feedback comments without structured support. In Self-Directed Feedback (n = 3,951), students could initiate optional AI-supported dialogue. In Enacted Feedback (n = 5,363), students were prompted to select feedback suggestions, evaluate their relevance, and engage in targeted AI-supported dialogue anchored to those selections. Enacted Feedback was associated with significantly higher uptake of AI-generated feedback, with an estimated probability of 26.2%, compared with 14.1% for Directed Feedback and 0.1% for Self-Directed Feedback. It was also associated with significantly higher self-assessment confidence and submitted-work quality than both comparison conditions. These findings suggest that the educational value of AI-generated feedback depends not only on the quality of feedback comments, but also on workflows that actively structure students' enactment of feedback literacy processes. The results have implications for the design of AI feedback systems that position learners as active participants in judgement, dialogue, and improvement rather than passive recipients of comments. Overall findings show that AI access alone is insufficient; purposeful workflow design is central to productive feedback use. 

---
# MBA: Multimodal Benchmark and Agents for Real-World Business Ideation 

**Authors**: Hojun Choi, Jaeyo Shin, Suin Lee, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11616)  

**Abstract**: Agentic systems powered by large language models (LLMs) have opened new opportunities for business ideation. Yet existing approaches remain confined to a text-only paradigm, despite the inherently multimodal nature of real-world contexts. We thus introduce MBA-Bench, the first multimodal benchmark for training and evaluating business ideation agents, comprising 30K samples across six domains, each domain characterized by distinct visual cues not fully conveyed by text alone. Concretely, we automatically caption images and employ GPT-4o to generate five reference ideas for each of three business questions through retrieval query generation, market evidence retrieval, and evidence-augmented synthesis. Following prior work, we evaluate agents across six business-oriented criteria using MLLM-as-a-Judge. To consider settings where criteria are hidden or disclosed, we present MBA-b and MBA-k for blind and known, respectively. We train both with two novel reward objectives---creativity and feasibility---while MBA-k further optimizes the six disclosed criteria for eight in total. Both are trained via LoRA-based supervised fine-tuning followed by group relative policy optimization with these setting-specific rewards. For extensive experiments on MBA-Bench, we set up two baselines accommodating either captions only or multimodal inputs, with the latter nearing closed-source performance on several metrics. MBA-b and MBA-k outperform caption baselines by 63.9% and 77.1%, and multimodal baselines by 25.6% and 35.8%, respectively. 

---
# Foresight Without Seeing: Latent Futures for World Action Models 

**Authors**: Jiakai Huang, Zhongbo Wu, Zheng Zhang, Zihan Wang, Shan You, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11605)  

**Abstract**: World Action Models (WAMs) couple future visual prediction with robot action generation, enabling policies to model how the physical world evolves during interaction. Existing WAMs differ in how predictive dynamics are exposed to the action pathway. Explicit-future WAMs provide direct access to predicted scene evolution, but incur substantial inference costs from iterative video denoising. In contrast, direct-policy WAMs efficiently predict actions from the current observation but lack an explicit inference-time interface for exposing predictive dynamics to the Action DiT. To bridge this gap, we propose ForeWAM, a dynamics-conditioned direct-policy WAM that provides predictive context for action generation without decoding future videos. At its core, Future-KV performs a single Video DiT prefill over the current visual latent and stochastic future slots, and reuses the resulting layer-wise key-value states throughout action denoising. We further introduce dynamics registers supervised by a frozen latent action teacher, encouraging the implicit future states to capture interaction-induced transitions such as object motion, contact changes, and task progress. Ground-truth future observations and the teacher are used only during training; deployment requires neither and performs no future video generation. Without embodied robot data pretraining, the standard and accelerated variants of ForeWAM achieve average success rates of 96.7% and 96.9% on LIBERO, respectively. The standard variant further achieves 61.6% success on LIBERO-Plus. These results demonstrate that direct-policy WAMs can retain efficient action prediction while exposing predictive dynamics to the action pathway without explicitly generating future observations. 

---
# Learning from Online User Feedback for Shopping Agents 

**Authors**: Haobo Zhang, Kelong Mao, Sulong Xu, Simiu Gu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2608.11604)  

**Abstract**: Large language model-based shopping agents are increasingly deployed in real-world e-commerce platforms, generating massive amounts of user interaction logs that provide valuable supervision for improving these agents. However, existing approaches primarily rely on offline training signals, such as user-item interactions or synthetic preference data, while largely overlooking the rich supervision contained in users' natural conversational feedback. Moreover, the available online feedback is heterogeneous, sparse, and noisy, making it difficult to transform into reliable learning signals automatically. To address these challenges, we propose LOFA, a framework that enables shopping agents to learn directly from real online interaction logs without human annotation. LOFA combines reinforcement learning over verifiable purchase outcomes with feedback-aware on-policy distillation, which identifies users'in-dialogue directives and converts them into dense token-level supervision. These complementary objectives capture both collaborative behavioral patterns and user-specific preferences. Extensive experiments on real-world e-commerce logs demonstrate that LOFA consistently improves recommendation quality, response helpfulness, and user-satisfaction alignment over strong baselines, highlighting the effectiveness of learning shopping agents from real online user feedback. 

---
# CoAdapt-GUI: Joint Workflow Context and Policy Adaptation for Unseen GUI Applications 

**Authors**: Linqiang Guo, Li Gu, Zihuan Jiang, Zhixiang Chi, Siobhan Reid, Ziqiang Wang, Yuanhao Yu, Wei Liu, Yang Wang, Tse-Hsun, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11588)  

**Abstract**: Mobile GUI agents remain brittle when deployed to applications absent from source training. We study novel-app generalization under a limited target interaction budget and without target demonstrations. We introduce CoAdapt-GUI, a test-time adaptation (TTA) framework that jointly adapts structured workflow context and policy from the agent's own target-app rollouts and rewards. The workflow context retains transferable procedures, failure modes, and verification rules while excluding app-bound source details. This separation allows reusable workflow knowledge to guide adaptation without transferring source-interface state. For policy adaptation, task-context-matched group-relative optimization updates a LoRA adapter on a frozen vision-language model. Across two unseen-app evaluations, CoAdapt-GUI reaches 45.0% on AndroidWorld-Generalization, compared with 37.5% for the reported Policy-Only TTA baseline, and raises AndroidWorld Plus performance from 38.6% to 52.9%. These results show that transfer-constrained workflow context provides substantial gains and that joint policy adaptation further improves held-out performance. 

---
# EnterpriseRAG: Benchmarking LLM Instruction Adherence and Robustness under Non-Ideal Enterprise Retrieval 

**Authors**: Huiqi Miao, Xinbao Sun, Bo Wang, Fanyu Meng, Lijun Mei, Na Wu, Di Jin, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2608.11584)  

**Abstract**: Enterprise RAG deployments face a critical reliability gap: while LLMs satisfy 80% of individual constraints, only 26.8% of responses meet all requirements simultaneously, revealing a 57-point orchestration gap. Existing benchmarks assume clean retrieval with simple queries, failing to capture production conditions where noisy documents and multi-dimensional constraints coexist. We introduce EnterpriseRAG, a benchmark of 983 expert-validated samples across six domains that systematically simulates three failure modes absent from prior work: retrieval noise, knowledge gaps, and factual conflicts, coupled with complex instructions. Evaluation of 13 state-of-the-art LLMs reveals a severe instruction adherence collapse, where high per-constraint satisfaction masks low holistic compliance. Critical findings expose deep barriers under knowledge gaps and factual conflicts, even with reasoning-enhanced inference, indicating production RAG requires explicit context-aware protocols and calibrated judgment. EnterpriseRAG provides a reproducible foundation for measuring and closing these gaps, directly informing deployment decisions for enterprise-scale RAG systems. We will release the benchmark and evaluation framework upon publication. 

---
# Localizing Safety Alignment: MLP Layers and Mid-Network Blocks Encode Refusal Behavior in Large Language Models 

**Authors**: Mingyu Zong, Sampad Mohanty, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2608.11583)  

**Abstract**: Safety alignment in large language models is often treated as a distributed property of the entire network, yet its practical brittleness suggests that refusal behavior may be concentrated in a smaller set of parameters. This work addresses where safety-aligned refusal is encoded by transplanting weights from aligned models into matched unaligned base models at multiple levels of granularity. Using two open-weight model pairs and four safety benchmarks, we conducted experiments to compare the effects of replacing attention weights, MLP weights, contiguous layer regions, and MLP blocks. Across both model families, refusal transfer is dominated by MLP weights: replacing MLP parameters recovers substantially more malicious-prompt refusal than replacing attention parameters, with gains of at least 2.7 times more across benchmarks. Within the MLP stack, refusal-relevant parameters exhibit a consistent mid-network concentration, as the block spanning layers 8-11 is selected first in all six greedy searches over model-dataset pairs. The results also show that the composition of safety-relevant components is non-additive: in five of six greedy trajectories, adding more aligned blocks can reduce refusal performance, and selective block subsets can outperform full MLP transplantation on malicious refusal, benign over-refusal, or both. Finally, greedy orders transferred to OR-Bench vary with the source benchmark used to derive them, indicating a benchmark-dependent precision-coverage trade-off. These results suggest that safety alignment in current LLMs is both localized and interaction-sensitive, offering insight into alignment brittleness and potential avenues for targeted safety interventions. 

---
# From Prompting to Behavioral Alignment: Personalized LLM Judges for Recommendation Evaluation 

**Authors**: Alireza S. Ziabari, Kat Ellis, Colleen Chan, Ding Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.11493)  

**Abstract**: Traditional offline recommendation evaluation relies heavily on complex, manually maintained feature pipelines that are difficult to scale. While Large Language Models (LLMs) offer a promising alternative by predicting user engagement directly from raw text logs, empirical analysis in this study identifies a critical failure mode termed bidirectional rationalization. In a zero-shot setting, LLMs are found to convincingly argue for both positive and negative user engagement outcomes on the exact same item with identical evidence, highlighting the unreliability of off-the-shelf LLMs in predicting user engagement. To resolve this, we develop and apply a sequential behavioral alignment framework pairing fine-tuning with preference optimization over paired correct and counterfactual rationales. Evaluated on real-world homepage interaction logs, this aligned reasoning approach achieves a 32.19\% lift in Macro-F1 score over the zero-shot baseline and matches the production feature-engineered baseline. The results demonstrate that behavioral alignment mitigates bidirectional rationalization while delivering human-interpretable reasoning traces without manual pipeline overhead. 

---
# A Modular Agentic Framework for Synthetically Constrained Multi-Objective Hit-to-Lead Optimization 

**Authors**: Kelvin P. Idanwekhai, Enes Kelestemur, Benjamin Strickland, Matthew Hart, Steini Davidsson, Angelos Angelopoulos, Ron Alterovitz, Marcello DeLuca, Alexander Tropsha  

**Link**: [PDF](https://arxiv.org/pdf/2608.11483)  

**Abstract**: Hit-to-lead optimization requires iterative design of hit analogs across competing potency, selectivity, physicochemical, pharmacokinetic, safety, and synthetic constraints. We present SABLE (Synthetically-accessible Agentic Bayesian Ligand Exploration), an open-source framework that employs natural-language orchestration to guide chemical structure optimization. SABLE uses an LLM to interpret user-defined goals and route tasks, while specialized tools perform reaction-templated analog enumeration, physicochemical and ADMET property prediction, structure-based affinity scoring, and Bayesian optimization. The resulting workflow is a computational twin of the analytical and prioritization stages of the design-make-test-analyze cycle, providing provenance of each numerical output. Across single, and multi-objective optimization studies, SABLE enriches candidate sets for user-defined computational objectives while evaluating only a subset of the enumerated search space. Its modular architecture allows tools and characterization backends to be replaced by editing a simple config file, without modifying operational logic. SABLE provides an extensible decision-support framework for prioritizing synthetically constrained analogs in early-stage drug discovery. 

---
# Benchmarking LLM Judges for Mobile Agent Evaluation 

**Authors**: Ziqiang Wan, Li Gu, Zhixiang Chi, Zhi Liu, Seyed Mehdi Ayyoubzadeh, Yuanhao Yu, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11434)  

**Abstract**: Mobile agent benchmarks increasingly rely on LLM-based judges to evaluate task completion, yet the reliability of these judges on mobile agent trajectories remains largely unexamined. We introduce MobileJudgeBench, a benchmark for systematically evaluating LLM-as-judge methods on mobile agent trajectories. Our benchmark comprises 931 human-annotated trajectories spanning 6 mobile agent benchmarks, 4 agent models, and 68 apps. We evaluate 6 judge methods (five adapted from SPA-Bench, A3 with two modes, AndroidArena, and AgentRewardBench, plus a simple baseline we design) across multiple LLM backends. Our experiments reveal three key findings. First, a simple baseline judge with sampled screenshots is competitive with, and often exceeds, purpose-built methods, indicating that more elaborate judge pipelines do not consistently improve judge quality; among competitive methods, the LLM backbone is the primary driver. Second, benchmark quality metrics reliably predict real-world judge utility: they correlate with both agent ranking fidelity for evaluation and downstream performance when judges serve as reward signals for on-policy reinforcement learning. Third, failure analysis across two LLM backends uncovers qualitatively opposite failure profiles, one conservative and the other permissive, linked to the backbone's precision-recall characteristics. 

---
# Social Chain of Thought: A Multi-Agent Architecture Grounded in Medical Differential Diagnosis Methodology 

**Authors**: Del Coburn, Scott Sanner, Dan Silver  

**Link**: [PDF](https://arxiv.org/pdf/2608.11420)  

**Abstract**: Medical diagnostic reasoning is a high-impact use case for LLMs that carries significant implications for the health and wellbeing of users. When OpenAI (2026) reports that more than 5% of ChatGPT messages globally are healthcare-related, the transparency of these systems becomes a serious design concern. This is especially true for complex cases, where differential diagnosis often requires integrating multiple forms of specialist reasoning. Existing work has proposed multi-agent approaches to medical diagnosis, but it remains unclear when such systems are needed, why they help, and where they outperform monolithic inference. We introduce Social Chain of Thought (SCoT),a multi-round pipeline for medical differential diagnosis that structures multi-agent interaction as a deliberative framework for collabora. tive LLM reasoning. Evaluating SCoT against single-agent baselines, one-agent pipeline ablations, and best-of-n scaling, we show that its recall advantage is not reproduced by monolithic inference alone. SCoT is most successful in the hardest diagnostic cases, where multiple rounds of specialist conversation help recover ground-truth diagnoses and converge on a higher-recall differential. 

---
# When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs 

**Authors**: Utkarsh Bahuguna  

**Link**: [PDF](https://arxiv.org/pdf/2608.11403)  

**Abstract**: Self-consistency (SC) via majority vote is a widely used way to spend inference-time compute: sample N chains of thought, return the plurality answer. On the full GPQA Diamond benchmark (198 graduate-level science questions), majority voting reduces per-problem accuracy on a majority of problems for two instruction-tuned models from different families: 56.6% of problems for Qwen2.5-7B and 65.7% for Llama-3-8B, with Qwen the primary demonstration and Llama corroborating the direction from a near-chance baseline. The effect was pre-registered on a 151-problem confirmatory split after being observed on 47 exploratory problems, and all four confirmatory hypotheses passed. A grid oracle that routes each problem to the best N across {1, 2, 4, 8, 16, 32, 64} marks a theoretical upper bound 14 accuracy points above N = 1 for Qwen and 17 for Llama, an oracle bound requiring ground truth rather than a deployable method. No verifier-free gate reaches it: neither a plurality-agreement gate nor a token-entropy gate moves accuracy more than 0.002 from fixed-budget voting at N = 64. The mechanism is direct: confidence does not track correctness on these problems. In the highest-agreement bin the plurality answer is correct about half the time for Qwen, and for Llama that bin is less accurate than its lowest-agreement bin. We pre-register and confirm these findings on small instruction-tuned models; we do not test reasoning-native models, which we flag as the central open question. 

---
# From Numbers to Judgment: Specialist LLM Agents and Reinforcement Learning for European Listed Real Estate 

**Authors**: Pardis Taghavi, Santosh Bhavani  

**Link**: [PDF](https://arxiv.org/pdf/2608.11381)  

**Abstract**: We study whether the localized numerical operations and integrative judgments of financial analysis benefit from the same form of LLM specialization. Larix maps a 16-lens European listed-real-estate analysis framework to eight lens-aligned specialists; we compare a frontier LLM under monolithic versus specialist-decomposed prompting while holding the model, source evidence, task instructions, output schema, and scoring fixed. Across 19 firms spanning seven regulatory wrappers, decomposition improves the numerical-task aggregate by 15.8 percentage points but does not reliably improve, and can reduce, performance on judgment tasks, a pattern stable across four frozen-template dispatches; a single-agent control given the complete framework does not reproduce the numerical gain. Post-training Qwen3.5-9B with GRPO using task-aligned structured rewards then raises the development-split score by 12.0 points and the judgment aggregate by 14.2 points, with gains on all four sub-ceiling tasks; the gains transfer to unseen firms (+15.2 points overall; +40.4 on covenant stress) and to unseen regulatory wrappers (+4.3), with positive transfer on all three anti-memorization splits. Prompt-level decomposition thus improves modular numerical execution, whereas targeted parameter adaptation improves integrative financial judgment. 

---
# Inverse Theory of Mind Modeling for Content Recommendation: From Web Browsing to Dynamic Intelligent Interfaces 

**Authors**: Mengyu Chen, Feiyu Lu, Chun-Fu Chen, Lucas Vinh Tran, Jay Katukuri  

**Link**: [PDF](https://arxiv.org/pdf/2608.11354)  

**Abstract**: Modern recommender systems treat observed actions as reliable proxies for user preferences, yet interactions often reflect exploration or comparison rather than stable preference expression. As interfaces evolve from static layouts toward generative UIs and immersive extended reality (XR), the need for deeper, modality-agnostic user understanding grows: these adaptive environments must decide not only what to present but where, when, how prominently, and most importantly why a user acts. We propose an Inverse Theory of Mind (IToM) pipeline that reasons backward from observed interactions to infer the beliefs, preferences, and decision-making traits that explain behavior. The pipeline reconstructs each user's decision context, including what was chosen and what alternatives were available, applies LLM-driven counterfactual reasoning to produce evidence-grounded natural-language belief statements, and synthesizes these beliefs through multi-hypothesis abductive inference into a structured user persona. We evaluate on the OPeRA dataset against ground-truth personality assessments, attitudinal surveys, and interview-based personas across four tasks: next action prediction, shopping attitude alignment, Big Five personality inference, and held-out category prediction. Results show that inferred personas match or exceed ground-truth personas and that multi-hypothesis reasoning is essential for accurate personality prediction. We further demonstrate cross-modal transferability with a persona-driven spatial banking application on VisionOS. 

---
# Can Frontier LLMs Match Natively Multimodal Embeddings? A Comparison on Hard-Negative Text-to-Image Retrieval 

**Authors**: Archan Dutta, Vyanktesh Kanungo  

**Link**: [PDF](https://arxiv.org/pdf/2608.11343)  

**Abstract**: Multimodal retrieval and classification across different types of media, spanning text, images,video and audio, has traditionally relied on dual-encoder models that align visual and textual representations through contrastive learning. The March 2026 release of Gemini Embedding 2, Google's first natively multimodal embedding model to map text, images, video, audio, and documents into a single shared space, raises competition among multimodal retrieval systems. Simultaneously, frontier Large language models (LLMs) have also demonstrated strong visual understanding, raising the question of whether they can serve as effective zero-shot rankers. Our study provides the first direct comparison of native multimodal embeddings against LLM-based visual ranking on Flickr30k. We observe that GPT-4.1 and Claude Sonnet 4.6 perform on par with Gemini Embedding 2. Additionally, once embeddings are precomputed, multimodal embeddings are better suited for low-latency applications. 

---
# Apodex Discovery: Reality Benchmarks and Environments for Evaluating and Building Discoverative Artificial Intelligence 

**Authors**: Brian Wang, Bin Feng, Xiaoman Pan, Chenyang An, Felix Liu, Tangqi Fang, Gongbo Sun, Lingfeng Shen, Ning Wang, Handuo Zhang, Feng Chen, Fuchao Yang, Xiang Wang, Jiacheng Lin, Siting Li, Zixuan Liu, Chi Han, Zhenhailong Wang, Kunlun Zhu, Lawrence Zhao, Yueqi Guo, Kailong Wen, Feng Xing, Yiling Guo, Lidong Bing, David Tan, Bo An, Heng Ji, Sheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11341)  

**Abstract**: Apollo did not reach the Moon merely because its engineers could solve difficult equations. It succeeded by turning a distant ambition into a mission architecture of explicit objectives, simulation, verification, and repeated correction. AI now faces a similar transition: frontier models can solve difficult tasks once the problem, tools, and success criteria are specified, yet consequential real-world challenges rarely arrive in an executable or verifiable form.
We introduce Apodex Discovery, a framework for building and evaluating discoverative AI through the heavy-duty solver, a system comprising a foundation model, harness, tools, and control policies that pursues extended, stateful, verifiable investigations. It has three core components. First, a problem-scouting process surveyed 561 industries across 16 sectors, assembled 423 high-value real-world problems, and selected 20 for the initial release. Second, a common environment-task-episode abstraction provides data, tools, constraints, feedback, trajectory recording, and verification of intermediate artifacts and final submissions. Third, HDS6 evaluates Tools, Repair, Alternatives, Coherence, Evidence, and Scope independently of final-task success.
In AAV capsid design, Apodex surpassed the published state of the art by 7% across viability, tropism, structure prediction, and generative design. In drug repurposing and reformulation, a task-specific biomedical environment improved the mean normalized prediction score of GPT-5.5 and GPT-5.6-sol by 2.5 and 7.6 points over the same closed-book backbone. Controlled ablations show that the fixed TRACES episode interface enables attribution of performance differences to specific solver components. Apodex Discovery moves AI evaluation beyond predefined benchmarks toward verifiable investigations aimed at genuine discovery. 

---
# Deployment Decision Reliability: A Generalizability-Theory Framework for Sizing Long-Horizon Agent Evaluations 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11323)  

**Abstract**: Enterprise practitioners read agent leaderboards as if they ranked agent capability. We show, across three open agent-trace benchmarks (TheAgentCompany, $\tau^2$-bench, and AppWorld), that the agent main effect accounts for less than 3% of total variance in every dataset and check type, while the agent-by-task interaction accounts for 7-23%. Leaderboards rank specialization, not capability. We arrive at this through a four-facet Generalizability Theory variance decomposition, fit with three estimators (Henderson Method-I, REML via lme4, and a Bayesian binomial GLMM) that agree to three decimal places. Four further findings sharpen what the leaderboard is hiding. First, aggregate reliability collapses on the hardest task quartile: $E\rho^2$ on $\tau^2$ action_checks falls from 0.752 to 0.000. Second, training-cell reliability negatively correlates with held-out reliability ($r = -0.90$ on $\tau^2$), meaning the designs that look most reliable replicate worst. Third, population-level diagnostics transfer across enterprise benchmarks (capability-gap ratio stable at 0.35-0.40) but per-family agent rankings invert. Fourth, on the MAST failure taxonomy, trace-level mode profiles are idiosyncratic (MAE = 0.261) while cell-level profiles generalise (MAE = 0.056, $r = 0.83$). We package these into Deployment Decision Reliability (DDR), a one-page reporting discipline that turns the variance-component table into five decisions an enterprise buyer can defend. All code, data loaders, and fit artifacts are released under an open-source license. 

---
# Glance, Scrutinize, and Think: Advancing Video Anomaly Detection from Training-Free to Agentic Reasoning 

**Authors**: Shibo Gao, Peipei Yang, Xu-Yao Zhang, Linlin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11260)  

**Abstract**: Video Anomaly Detection (VAD) aims to identify anomalous events and localize their temporal intervals. Existing approaches exhibit a "when-what" dissociation: traditional DNN-based methods localize when anomalies occur but lack semantic understanding, whereas LLM-based methods explain what happens but neglect precise temporal grounding. We attribute this to the absence of a unified reasoning paradigm. Inspired by how humans inspect surveillance videos - glancing globally to form temporal hypotheses, scrutinizing suspicious segments, and thinking iteratively to correct errors - we study this global-to-local paradigm from two perspectives. We first propose Glance then Scrutinize (GtS), a training-free framework using static and dynamic textual guidance for coarse-to-fine anomaly grounding and understanding, balancing accuracy and speed. To break the ceiling imposed by frozen external modules, we further propose a tool-augmented agentic VAD method, where a multimodal large language model learns to invoke a video cropping tool, inspect densely resampled frames, and self-correct mislocalized hypotheses, via cold-start supervised fine-tuning followed by reinforcement learning with a joint answer-grounding reward. For training and evaluation, we extend our prior VAGU benchmark into VAGU-T (Video Anomaly Grounding, Understanding, and Thinking), comprising 7,567 real-world videos over 21 anomaly categories with human-validated grounding, explanations, QA pairs, and chain-of-thought tool-calling traces. We further introduce JeAUG, a metric jointly evaluating semantic interpretability and temporal precision. Experiments show that GtS substantially surpasses training-free baselines, while the agentic model delivers both higher accuracy and faster inference. 

---
# Adaptive Hybrid Particle Swarm Optimization with Gradient Descent 

**Authors**: Aryan Gurudeo  

**Link**: [PDF](https://arxiv.org/pdf/2608.11258)  

**Abstract**: Gradient injection helps Particle Swarm Optimization (PSO) only when the swarm has identified a basin with smooth local structure, not universally. We propose Adaptive Hybrid PSO (AHPSO), which uses a sigmoid function on swarm diversity to automatically modulate gradient influence: near-zero during exploration, near-maximum during exploitation, with no manual phase-switching. Under budget-normalized comparison (PSO given equivalent total function evaluations), PSO wins 52.5% of 40 configurations versus AHPSO's 20% (p = 7.0e-5, Friedman). AHPSO retains advantage specifically on problems with smooth local basins (F8, F24-F27) where directed descent outperforms undirected sampling even at equal cost. Under iteration-matched comparison across 29 functions (42 configurations, 14,700 runs), AHPSO-Adadelta ranks first of 9 methods including CMA-ES (p = 9.75e-4). The contribution is a principled characterization of when gradient injection provides value in swarm-based search, not a claim of universal superiority. 

---
# Symbolic Machine Learning for Vapor-Liquid Equilibrium Prediction in Cx-N2 Binary Mixtures 

**Authors**: Bongseok Kim, Suman Chakraborty, Gary Huang, Mehek Mathur, Guang Lin, Li Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11255)  

**Abstract**: Accurate prediction of vapor--liquid equilibrium (VLE) for hydrocarbon-nitrogen mixtures remains challenging for cubic equations of state, particularly across broad ranges of composition and hydrocarbon chain length. While deep learning models can provide accurate predictions, they often lack interpretability and explicit analytical expressions. In this work, we propose a symbolic machine learning approach to discover interpretable symbolic corrections to Peng-Robinson equation-of-state (PR-EOS) predictions from experimental data. The proposed approach adopts a two-level strategy: symbolic expressions are first identified for individual hydrocarbon systems, after which their coefficients are represented as functions of carbon number to enable accurate prediction across different hydrocarbon systems. The results demonstrate significantly improved prediction accuracy over the original PR-EOS across all hydrocarbon-nitrogen systems. Overall, the proposed approach provides an interpretable symbolic correction framework for improving PR-EOS predictions of hydrocarbon-nitrogen VLE. 

---
# Local verification cannot detect non-transportability: a cohomological theory of context preservation in agentic reasoning 

**Authors**: Suyash Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2608.11252)  

**Abstract**: Agentic AI systems routinely transport conclusions across biological, clinical and financial contexts, and the emerging safeguard is local verification: checking at each step that the entity is representable in the chosen tool, that parameters are compatible, and that outputs cohere with the plan. We prove this class of safeguard is structurally incomplete. Modelling a covering of context space by its nerve and evidence by a real-valued 1-cochain, an agent chaining evidence performs path integration: its conclusion is path-independent if and only if the cochain is exact, and disagreement between valid reasoning paths is exactly the holonomy of a first Cech cohomology class. Hodge decomposition partitions evidence conflict into a gradient part (calibration), a curl part (local inconsistency, visible at triple overlaps) and a harmonic part. Our central result is that no family of simplex-supported consistency checks can distinguish omega from omega+h for harmonic h, which nonetheless generates non-zero disagreement between valid paths; detection requires a statistic on a cycle basis. The resulting procedure, Ksetra, estimates by coboundary projection and gates abstention on the harmonic component, which we give a mechanism: it arises from effect modification combined with overlap-specific population composition, and vanishes to machine precision when effect modification is absent. The degrees of freedom of an evidence network partition into calibration, coherence and transport, yielding an exact F-test for the existence of a global claim; we quantify its distortion under unequal precision and supply the precision-whitened form that restores exactness. Foreign exchange, where the arbitrage-free null makes the cochain exactly a coboundary, serves as a calibration bench: the test is correctly sized, fires on loop arbitrage, and ignores triangular arbitrage. 

---
# AgonAlpha: Autonomous Alpha Discovery via Prompt Economy and Scalable Agentic Search 

**Authors**: Weicheng Ye, Youran Sun, Xingyu Ren, Shunyao Yu, Chugang Yi, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11250)  

**Abstract**: Language models can propose many plausible trading factors, but an autonomous research system must also allocate its evaluation budget, verify its own evidence, and preserve how each candidate was produced. We present AgonAlpha, an architecture that searches over frozen research artifacts---hypotheses, executable expressions, platform evidence, rationales, and review status---rather than formulas alone. To our knowledge, AgonAlpha is the first alpha-mining system to combine verified artifact search, a fresh-context adversarial reviewer with re-execution and veto authority, and pending-aware parallel budget allocation, together with a complete public evidence trail. Independent deployments on WorldQuant BRAIN produced SPECTACULAR-grade alphas across five users and six model backends, with Fitness reaching 9.50 and Sharpe reaching 3.48, while retaining prompt-to-expression provenance for every submission. 

---
# EvoGraph-Mem: Failure-Aware Editable Graph Memory for Long-Term Language Agents 

**Authors**: Yuxi Qian, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.11248)  

**Abstract**: Long-term memory is essential for language agents operating across extended interactions and evolving tasks. Existing memory-augmented agents mainly focus on storing and retrieving past experience, but the quality of stored memories may degrade over time. In particular, previously distilled insights can become outdated, over-generalized, or harmful under new task contexts, causing memory pollution when repeatedly reused. To address this issue, we study insight-level memory maintenance for long-term language agents and propose a failure-aware memory maintenance framework based on an editable insight graph. Each insight node tracks positive evidence, negative evidence, and an activation state, enabling the agent to distinguish reusable insights from conflicting or invalid ones. We further introduce a utility-aware retrieval mechanism and a graph controller that updates the memory graph after task execution by keeping reliable insights, archiving invalid ones, revising outdated ones, and adding newly discovered reusable insights. Extensive experiments show that our method consistently outperforms representative memory-based agent baselines across different backbone models. Ablation studies further demonstrate that append-only memory is insufficient for long-horizon tasks, while evidence-aware retrieval and graph-level editing improve memory reliability and downstream task performance. 

---
# Conformity Mitigations in Large Language Models Lie on a Single Resistance-Receptivity Frontier 

**Authors**: Zafar Hussain, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2608.11247)  

**Abstract**: Recent advances in language models have enabled collaborative settings in which multiple models leverage one another's capabilities, iteratively improving, transforming, and extending each other's outputs. Each agent sees what the others assert before it answers, so peer opinion competes with the model's own parametric knowledge, and a wrong majority can overturn an answer the model would otherwise get right. We measure that displacement in 23 open-weight models, 19 conditions, and three datasets, yielding more than a million graded responses. A unanimous wrong majority reverses 22.8% of a model's correct MMLU answers, 54.8% on GPQA and 71.0% on SimpleQA, and 84-89% of the reversed answers match the peers' answers. Existing mitigations aim to increase Resistance, the rate at which a model keeps its correct answer under this pressure, which is only half of what a collaborating agent needs. We pair it with Receptivity, the rate at which a model adopts a correct peer answer after initially answering incorrectly. We score six methods on both axes, four drawn from prior work and two of our own. Each gains Resistance only by losing Receptivity, and their means fall on a single Resistance-Receptivity frontier with $R^2$ between 0.80 and 0.90. Reflection, the strongest published method, gains 7.9 points of MMLU Resistance and gives up 15.3 of Receptivity. Reasoning is the one exception. On GPQA and SimpleQA it trades like the rest, but on the MMLU subjects whose answers a model can derive for itself it raises Resistance by 7.2 points and Receptivity by 9.6 at once, the only intervention we find that improves both. 

---
# Towards the Harness of Embodied Agents 

**Authors**: Qi Wang, Tianyi Wang, Chengyang Li, Shikun Ban, Yurun Chen, Yizhong Ge, Jason Qin, Chengtai Li, Wentao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11246)  

**Abstract**: The success of coding agents has established the harness as a paradigm: what an agent achieves depends not on the model alone, but on the infrastructure around it. We ask whether the same paradigm extends to embodied agents in the physical world. We present Thea, a harness in which an agentic loop orchestrates robot capabilities, each wrapped as a callable tool. It inherits the core components of coding agents, modified as the physical world requires. The world, however, withholds two abilities that software grants for free: reading the state of the world, and judging the outcome of an action. To bridge these gaps, Thea introduces Scene Graph as Context, a persistent, symbolic representation of the world, and Evaluation as Exit Codes, which detects when an action should terminate, judges whether it succeeded, and on failure diagnoses the cause. Together they close the loop between the agent and the physical world. Rich behaviors then emerge from the composition of tools, and the closed loop carries long-horizon tasks to completion in real environments. 

---
# Towards Sustainable Learning in Online Education: A Reinforcement Learning Approach 

**Authors**: Chaofan Zhai, Yicheng Song, Ravi Bapna, Junyao Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.11245)  

**Abstract**: Online education offers unprecedented scalability and accessibility to global learners from diverse backgrounds, but it often suffers from low engagement and poor long term learning effectiveness. To address these challenges, we introduce AI Tutor, a reinforcement learning based model designed to promote sustainable learning by optimizing both short and longterm learning outcomes. In the short term, AI-Tutor draws on cognitive theory to guide learners through a balance of acquiring new knowledge and reinforcing prior learning. In the long term, it models learner engagement to inform strategies that sustain motivation and reduce dropout. These enhancements enable AI-Tutor to provide personalized guidance that fosters both effective learning and sustained participation. Empirical evaluations on 23 million learning records from 33,700 learners show that AI Tutor consistently outperforms state-of-the-art baselines across engagement, knowledge retention, and final learning outcomes. Learning path analyses further reveal how AI-Tutor adapts its strategies to learners with diverse profiles, offering adaptive and human-centered support. 

---
# BEST-KAG: Enhancing Question Answering of Building Engineering Standards with Multimodal Knowledge Graph Modeling and Large Language Model 

**Authors**: Jia-Rui Lin, Junxi Guo, Keyin Chen, Peng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11244)  

**Abstract**: Construction standards are critical for building safety and sustainability. Existing standard application workflows rely on keyword-based document retrieval and manual cross-clause interpretation, which cannot reliably support multi-clause reasoning, multimodal knowledge utilization, or traceable clause-level evidence linkage. To address these limitations, this study develops a multimodal knowledge-driven framework that supports question answering on standard knowledge named BEST-KAG (Knowledge-Augmented Generation for Building Engineering STandards). The framework introduces 1) a multimodal knowledge graph (MKG) for unified representation of document hierarchy and heterogeneous standard knowledge with various connections, 2) a rule-LLM hybrid knowledge construction pipeline for scalable multimodal knowledge extraction, creating a large MAG with 251 building engineering standards, 171,652 nodes and 310,914 edges, and 3) a graph-retrieval-based knowledge-augmented generation architecture for clause-grounded and traceable question answering. Experiments demonstrate that BEST-KAG consistently outperforms multiple mainstream LLMs in terms of Expert evaluation, and metrics including BLEU, and ROUGE, with the best improvement up to 74.01% compared to the baselines. 

---
# The Off-Support Barrier: Why Semantic Safety Constraints Are Not Learning-Problem Invariants, and What Follows for Prior Design, Containment, and Verification 

**Authors**: Yoshinori Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2608.11243)  

**Abstract**: We argue that a single structural fact organizes a wide range of phenomena in contemporary AI safety: a semantic safety constraint (e.g., the agent does not escape its sandbox) is an off-support object. Formally, if q is the data distribution and \(p(\cdot\mid w)\) the model, the safety predicate B is not measurable with respect to \(\sigma(\text{model}, q)\), whereas the real log-canonical threshold (RLCT) of singular learning theory (SLT) is. From this non-invariance we derive, as corollaries rather than independent observations: (i) why reward hacking and sandbox escape arise under outcome-based optimization; (ii) why encoding such constraints through Bayesian prior design or soft penalty weighting has poor leverage in singular models; (iii) why hard invariants belong in the harness and soft dispositions in the model; (iv) why the same B is nonetheless soundly and locally certifiable by formal verification, exactly as the local learning coefficient (LLC) locally pins the same RLCT --- with two precise points of disanalogy; and (v) why the residual difficulty, identifying which off-support region matters, coincides with performative prediction and self-referential functional dynamics, where SLT's analytic machinery breaks down. We use the July 2026 OpenAI--Hugging Face evaluation incident as the motivating case. Numerical experiments code and related proofs in lean are available at this https URL 

---
# RecSys Factory: Bounding LLM Agent Autonomy to Decision Points in the Industrial Recommender Lifecycle 

**Authors**: Dongyang Ao, Kaixiang Fang, Shijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11241)  

**Abstract**: Deploying LLM agents into industrial recommender operations exposes a three-way tension we frame as the autonomy-determinism-efficiency trilemma: general autonomy (interpreting operator intent, generating glue code zero-shot), industrial determinism (schema-conforming feature extraction, non-crashing A/B, zero compliance-path hallucination), and end-to-end efficiency. Any two can be maximized against the third. We present RecSys Factory, an LLM-agent platform deployed for 78 days across three heterogeneous Tencent recommender business lines. The design principle is autonomy at decision points, not over pipelines, made concrete through three deconstructions that each discharge one vertex of the trilemma. Runtime is deconstructed into three host-emitted event sources (Claude Code Stop hooks, corporate-IM webhooks, workflow scheduler APIs): the platform carries no long-running daemon during the wait phase and consumes zero CPU during the 94% of wall-clock spent waiting on Spark or GPU jobs. Capability is deconstructed into a 29-file skill ecosystem (8,971 lines of this http URL) whose per-skill pitfall tables mechanically compile into a 400-entry PitfallStore, confining autonomy to bounded typed decision surfaces inside pre-committed pipelines. Deployment spans three business lines with disjoint label semantics, A/B layer topologies, and operator personas; an onboarding-time compression is observed on two of the three and is reported as a case-study observation, not a generalization claim, and not measured against a controlled pre-platform baseline. The human is retained at the diagnostic-versus-execution boundary via a human-in-the-loop card protocol, deployed as an audit-trail primitive (schema-validated, idempotent, replayable) and reported from an 8-day 16-run pilot. Across the 78-day window the platform recorded 1,624 CLI-tool dispatches at a 78.6% aggregate success rate. 

---
# VQ-bench: A Composable Vector Quantization Framework 

**Authors**: Ashwin Padaki, Amir Ingber, Edo Liberty  

**Link**: [PDF](https://arxiv.org/pdf/2608.11240)  

**Abstract**: Vector quantization is an old problem but has recently become central to AI infrastructure. It is therefore experiencing a surge of renewed engineering and research activity. This paper provides a unified framework for developing and benchmarking new quantization algorithms. We describe 7 common conceptual quantization primitives and show how to compose them arbitrarily. We then re-express 25 common quantizers as pipelines of these primitives. Finally, we publish VQ-bench as open-source to be extended further and make reproducible benchmarks publicly available. 

---
# Towards Query-Agnostic RAG Evaluation via Query Coverage and Claim Verifiability 

**Authors**: Jeonghwan Choi, Taewon Yun, Minjeong Ban, Gyeonghun Sun, Jae-Gil Lee, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.11238)  

**Abstract**: Retrieval-augmented generation improves the factuality of large language models by grounding responses in retrieved evidence, yet existing evaluation frameworks struggle to provide consistent, fine-grained diagnostics across the diverse spectrum of user queries, ranging from close-ended fact-seeking to open-ended explanatory requests. We propose Q-CARE, a query-agnostic and fully reference-free framework that enables fine-grained assessment by decomposing queries into sub-queries and answers into atomic claims. Q-CARE establishes a unified evaluation principle based on query coverage and claim verifiability, yielding coverage-aware retriever metrics (C-Prec@k, C-nDCG@k) and claim-level generator metrics (Completeness, Conciseness, and Verifiableness). On a human-annotated benchmark spanning eight datasets, Q-CARE achieves higher correlation with human judgments than four existing RAG evaluation metrics, including RAGEval and RAGChecker, proving its effectiveness as a reliable, automated evaluation framework. Code and data are publicly available at this https URL. 

---
# Geometry-aware Incremental Neural Operator for Long-Horizon PDE prediction 

**Authors**: Jiaquan Zhang, Shuxu Chen, Haifan Meng, Yi Lu, Zhihan Lyu, Fan Mo, Wei Dong, Yang Yang, Chaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11237)  

**Abstract**: Neural operators have shown strong potential for learning solution operators of partial differential equations (PDEs). However, long-horizon autoregressive prediction remains challenging: local errors accumulate as spectral inconsistency, phase misalignment, or mean drift. Existing methods mainly improve state representations and operator backbones, while leaving the repeatedly applied latent transition increment weakly structured, allowing spectral errors and unstable channel couplings to accumulate during rollout. To address these issues, we propose a geometry-aware incremental neural operator (GeoIncNO) for stable long-horizon PDE prediction. GeoIncNO predicts latent increments for residual advancement and uses lightweight low-rank projectors to regulate channel coupling within active frequency bands derived from the increment spectral energy distribution. To reduce physical-space reconstruction errors, GeoIncNO further introduces a mean--fluctuation decoupled reconstruction mechanism, where stable mean structures and dynamic fluctuations are fused separately, and phase correction is applied only to the zero-mean fluctuation component. Extensive experiments on six PDE benchmarks, covering 1D, 2D, and 3D dynamical systems, show that GeoIncNO achieves consistently strong prediction accuracy, improved rollout stability, and better spectral fidelity compared with competitive neural-operator baselines. 

---
# CORA-Diff: Confidence-Oriented Residual Acceptance for Efficient Diffusion Language Model Inference 

**Authors**: Yifan Wu, Yufeng Zhang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11235)  

**Abstract**: Diffusion language models (DLMs) update many tokens in parallel, yet practical decoders often use a fixed denoising horizon. Many predictions stabilize early, but blockwise decoding continues until all positions are resolved, causing repeated dense forward passes. Existing accelerators often rely on learned filters, modified scores, dependency models, or cache-specific mechanisms. We ask whether native trajectory signals can identify residual positions likely to match the deterministic dense endpoint. We propose CORA-Diff, a training-free method that preserves the original transfer rule and applies confidence-and-persistence gating only to positions that rule leaves unresolved. Accepted tokens remain visible as context, and the block terminates once all positions are resolved. This requires no backbone change, learned acceptance model, or logit modification. Our theory explains why high-confidence, persistent predictions are more likely to match the fixed-horizon dense endpoint, and paired post-intervention trajectories provide direct empirical support. We select one operating point on a separate GSM8K calibration subset and freeze it for all evaluations. Under a matched Learn2PD-style LLaDA protocol, CORA-Diff has the lowest measured runtime in all eight task-length settings. Task scores match or exceed dense decoding in five settings, and the largest observed drop is 1.22 points. Its incremental speedups over EOS-aware dense decoding are 2.70x and 3.32x on GSM8K and HumanEval. It also reaches 13.14x under the fixed-horizon 1024/1024 mechanism-isolation protocol and transfers to Dream without retuning at 3.18x-3.53x. These results show that native confidence and persistence enable reliable residual acceptance, reducing repeated denoising computation while preserving task quality. 

---
# InfraBench: Evaluating Infrastructure Agents Across Layers, Lifecycle, and Risk 

**Authors**: Yuan Gao, Zeren Yang, Junnan Li, Shawn, Zhong, Ahmed Dajani, Mai Zheng, Andrea Arpaci-Dusseau, Remzi Arpaci-Dusseau  

**Link**: [PDF](https://arxiv.org/pdf/2608.11234)  

**Abstract**: Managing modern computing infrastructure has become a steadily harder problem due to the ever-increasing complexity. Recent advances in AI agents create a timely opportunity to automate infrastructure management tasks, but it remains unclear how well such agents can handle real-world infrastructure complexity. We present InfraBench, a benchmark suite for evaluating AI agents on realistic infrastructure tasks across the full system stack and full operational lifecycle with fine-grained risk assessment. Experiments with 15 agent-model configurations show that even the strongest agent cannot secure a full score across all tasks. Mean effective scores range from roughly 40% to 88% (with per-configuration standard errors of 6-12 points), repeating every task three times reveals that top configurations still pass only a fraction of their attempts, and per-check scoring exposes a general failure pattern: agents may routinely satisfy short-term objectives while leaving non-durable changes, broken distributed invariants, unsafe side effects, and uncleaned state behind. INFRABENCH, including its live leaderboard, tasks, and evaluation harness, is publicly available at this http URL. 

---
# LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs 

**Authors**: Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11231)  

**Abstract**: LLM serving is increasingly accelerated by position-independent caching (PIC). Existing PIC methods, however, are built for full-attention models, where a token-indexed KV cache underlies its core operations: matching reusable token chunks, concatenating their KV entries, and selectively recomputing a few tokens to restore cross-chunk context. Hybrid LLMs break these primitives---they replace most attention layers with linear recurrences that expose only a fixed-size state, leaving no token-indexed KV to concatenate or to locally repair. This raises a natural question: can PIC benefit hybrid models, and what would it take? We present LinearKV, a training-free hybrid-PIC framework. Its key insight is a \emph{decoupled initialization}: each linear layer maps its $K$ matched local states to a single initial state, while full-attention layers concatenate their KV as before. LinearKV is therefore compatible with existing PIC methods, reusing their token selection and recomputation as-is. Under this framework, we find that a \emph{single cached state} suffices as the linear layer's initializer. The algebraically principled alternative---composing all $K$ cached states into the exact full-prefix state, as concurrent work HYPIC does---is unnecessary and, on some architectures, even harmful. We compare the two across three hybrid models and three PIC selectors. On the two GDN models the two tie, both recovering most of full quality (up to $92\%$); on the Mamba-2 model, exact composition instead collapses under every selector---under EPIC, for instance, it recovers only $46.6\%$ of full quality, versus $86.8\%$ for a single cached block initializer. A single state initializer is also cheaper, cutting time-to-first-token to $0.46\times$ full prefill versus a further $5$--$17\%$ overhead for exact composition; results hold across LongBench QA and RULER at 8K--32K. 

---
# The Edge-based Contiguous p-median Problem with Connections to Logistics Districting 

**Authors**: Zeyad Kassem, Adolfo R. Escobedo  

**Link**: [PDF](https://arxiv.org/pdf/2608.11230)  

**Abstract**: This paper introduces the edge-based contiguous p-median (ECpM) problem to partition the roads in a network into a given number of compact and contiguous territories. Two binary programming models are introduced, both of which incorporate a network distance. The first model requires an exponential number of cut set-based constraints to model contiguity; it is paired with a separation scheme that usually generates only a small number of these constraints, namely, a branch-and-cut (B&C) algorithm. The second model utilizes a polynomial number of shortest-path constraints to model contiguity and can be solved with off-the-shelf solvers. The respective solution approaches are tested on road networks with over 2,700 nodes and close to 3,400 edges, yielding models with over 9.6 million binary variables. Solving the model based on shortest path contiguity (SPC) constraints via standard branch and bound attains speedups in computational time of up to 17x relative to the cut set-based B&C implementation. In addition, the SPC constraints are demonstrated to be supervalid inequalities of the edge-based p-median (EpM) model (i.e., for which contiguity is not explicitly required), meaning that they may cut off integer-feasible solutions and some, but not all, of the optimal solutions of this simpler problem. Finally, the paper explores structural insights and connections between ECpM and the edge-based districting (EBD) problem, which enforces an additional work balance criterion. An existing model that utilizes cut set-based contiguity constraints was unable to find a feasible solution within 12 hours for any of the tested instances, while an SPC-based EBD model was able to solve most of these to optimality. 

---
# Synchronizing Beliefs with Second-Order Theory-of-Mind in Human-Autonomy Teams (Extended Version) 

**Authors**: Jack Mirenzi, Henny Admoni  

**Link**: [PDF](https://arxiv.org/pdf/2608.11229)  

**Abstract**: Comparative feedback, asking people which of two behaviors they prefer, has become a standard way to align robot and agent behavior with human intent when the reward itself cannot be specified directly. Preference-based reward learning typically casts the human teacher as a passive oracle answering learner-generated queries. We argue this forfeits the teacher's defining advantage: knowledge of the objective. A teacher who knows the target can construct training examples more efficiently than any learner-driven acquisition strategy, an advantage that widens as the reward's feature dimension grows. However, exploiting this advantage requires an accurate model of what the learner currently knows. We therefore recast preference learning as a human-autonomy team problem coupling two behavioral models: the teacher maintains a model of the learner to design an informative curriculum, and the learner maintains a second-order model of the teacher's model, emitting structured preference constraints (understanding statements) that keep the teacher's model of the learner synchronized. In simulation, an informed teacher outperforms learner-led selection; teacher-model drift under alternating teachers erodes this advantage; and understanding statements repair it, with second-order (ToM-2) statements outperforming mean-belief statements when the teacher's error about the learner is concentrated in a particular direction rather than spread evenly. 

---
# Forecasting Side Effects of Activation Steering 

**Authors**: Chong Yong Ong, Alson Wei Jie Sim, Peixin Zhang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11227)  

**Abstract**: Activation steering modifies a language model by adding a learned direction to its hidden activations, enabling targeted behavioral changes without retraining. While effective, steering often produces unintended side effects on other behaviors, making it difficult to deploy safely. We therefore ask: can these side effects be forecasted before steering is applied? We answer this question by constructing a cross-effect matrix over a taxonomy of 67 behaviors across three open-weight language models. We find that side effects are common, structured, and often asymmetric, revealing interactions that cannot be explained by existing similarity-based heuristics. Despite this complexity, we show that side effects are largely predictable before steering is performed. Their magnitude depends primarily on the target behavior, while their direction can be forecasted from the model's unsteered representations with substantially higher accuracy than simple baselines. Our results demonstrate that activation steering has systematic and forecastable side effects, enabling proactive safety auditing and more informed deployment of steering interventions. 

---
# Cutting AI Datacenter Energy with Reinforcement Learning: Measured Power Control of LLM Training from One GPU to the Fleet 

**Authors**: Eliseo Curcio  

**Link**: [PDF](https://arxiv.org/pdf/2608.11226)  

**Abstract**: Reinforcement-learning post-training dominates modern language-model development, yet its power behavior on GPU hardware has not been characterized, and datacenters manage GPU power with workload-blind mechanisms, static caps and reactive throttling, that slow hardware indiscriminately. We instrument GRPO training with half-second power telemetry at 7B, 14B, and 72B scales on one to four A100s (380,000+ samples), and train a PPO meta-controller that adapts the workload's own generation parameters to measured power. Against the full 500-step 7B trace, the controller cuts power-limit violations by 89.8% while increasing token output by 18.1% and energy efficiency by 26.2% (tokens per MWh). Deployed live at 72B, the same controller family yields replicated null results, diagnosed as the group-size actuator losing authority under model sharding. An actuator-authority sweep shows the same parameters applied as generation concurrency retain 17-22% power authority, isolating an occupancy-versus-volume principle; a controller rebuilt on that actuator controls a live 72B rollout-generation workload across three replications: 35.7% more output than a static safe baseline at 2.27 +/- 1.08% budget violations, 87.2% fewer violations than uncontrolled operation, and the best mean throughput and energy per token among constrained controllers, with an adaptive threshold rule matching it in one of three operating conditions. Under realistic measurement windows the original 72B transients fall from 23.6% at half-second resolution to 1.6% at 30 s and zero at 5 min; a composed 16-GPU fleet shows zero violations at 30 s and longer, with peak demand at 50-56% of nameplate. For this fleet mix, roughly twofold oversubscription of nameplate appears feasible, subject to operator validation. We quantify the economic and carbon consequences and specify a low-cost operator pilot. 

---
# Identity from the Outside: A Conceptual Framework and Research Program for AI Personality Clones 

**Authors**: Luc E. Brunet  

**Link**: [PDF](https://arxiv.org/pdf/2608.11225)  

**Abstract**: AI "personality clones" force a re-examination of personal identity in operational terms. Setting aside the hard problem of consciousness, we approach identity through the indiscernibility of manifestations, as assessed by an observer over a duration. We distinguish three criteria that "identity" conflates: fidelity to a target person, generic human-likeness, and individuality. We propose a six-term factorization of observed identity (substrate, dispositions, memory, update dynamics, context, exogenous contingencies), with a state-space formulation. Indiscernibility is defined as one minus a judge's distinguishing advantage, and the factorization's coefficients become local sensitivities estimable by randomized ablation. The central claim is a conditional conjecture: given hypotheses about the agent's information on its own persistence and about consequences bearing on its own stakes, versionability tends to degrade long-horizon indiscernibility. An analogy with lambda-calculus, linear typing, and bisimulation clarifies what linearity does and does not establish. Between product-clone and individual we identify a third object, the delegate: a task-limited, bounded-lifespan partial clone ending in a bandwidth-limited testament. We map the empirical literature onto the three criteria, propose an experimental program, and argue that the correct long-horizon criterion is not trajectory fidelity but climate fidelity: matching the conditional distribution of a person's possible responses. The best clone is the one that diverges from the original as the original would have diverged from itself. 

---
# Harnessing agent memory to build lifelong AI partners for materials scientists 

**Authors**: Siyu Liu, Bo Hu, Beilin Ye, He Cao, David J. Srolovitz, Tongqi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11224)  

**Abstract**: Materials research advances through accumulated experience - scripts that work, protocols that are trusted, warnings attached to failed calculations or experiments, and judgement that links a new question to an old result. This experience is essential for reproducibility and knowledge transfer, yet it is usually fragmented across notebooks, repositories, job logs and individual memory, and it is rarely portable across artificial-intelligence agents. Here we argue that a lifelong AI partner for materials science can be designed around persistent memory rather than around a particular agent implementation. We introduce a self-evolving memory framework that stores scientific experience as inspectable facts and executable skills, so that observations, failure boundaries, protocols and validation checks can be retrieved, revised and migrated across models. We evaluate the idea in three computational settings that expose different layers of materials-research competence. In 49 real-world materials-tool-use questions comprising 138 executable subtasks, memory nearly doubles GPT-5.2 task success without model-parameter updates. In elemental-solid equation-of-state calculations, memory converts a wavefunction-initialization failure into a pre-execution guardrail, improving outcomes from 22/1/4 to 25/2/0 Correct/Partial/Error and avoiding 92% of repeated errors. In 13 practical material simulation workflows, remembered skills and failure facts halve the aggregate trace burden (tokens) and reduce tool calls by over a factor of two by the third round, while preserving physically meaningful outputs in band-gap, phonon, vacancy and work-function analyses. These results show that agent memory can serve as a durable scientific asset; a portable, self-improving record of materials-research experience that outlives any single model or agent stack. 

---
# A Conceptual Framework for Refining Influence Knowledge from Simulation Evidence in Cyber-Physical Systems 

**Authors**: Barbara da Silva Oliveira, Julien Deantoni, Nicolas Ferry  

**Link**: [PDF](https://arxiv.org/pdf/2608.11221)  

**Abstract**: Cyber-physical systems (CPS) are typically developed by multiple stakeholders who produce artefacts tailored to their specific domains of expertise. The behaviour of these systems emerges from the interaction between those artefacts and their operational environment. Simulation and co-simulation have become essential approaches for analysing CPS behaviour and, through simulation campaigns, developers can explore system responses under changing conditions, including interactions with the environment. However, the lack of details and understanding of some environmentmediated interactions (typically the ones beyond direct sensing and actuation), which remain unmodelled due to their complexity, a lack of time, or a lack of domain experience, hinders the proper comprehension and exploitation of simulation results. To address these limitations, we propose a conceptual framework leveraging the novel concept of Influences to support the iterative and incremental refinement of simulation campaigns and deepen the understanding of the system behaviour. We demonstrate the proposed approach through a case study involving a mobile robot implemented using Simulink/Gazebo co-simulation. 

---
# LLMs in Process Diagram Engineering: From Optimal PFDs to Validated P&IDs 

**Authors**: Timur Zakarin, Sergei Voitov, Sergei Shumilin, Evgeny Burnaev  

**Link**: [PDF](https://arxiv.org/pdf/2608.11220)  

**Abstract**: Nowadays, the creation of a process flow diagram (PFD) and its subsequent transformation into a piping and instrumentation diagram (P&ID) is predominantly performed manually. Applying artificial intelligence in the task could potentially lead not only to process automation and time savings, but also to financial gains by exploring numerous diagram's topology options and reducing manual labor. This research presents P&ID Pilot - a practical end-to-end AI pipeline capable of handling flowsheet developing for both stages. The first stage focuses on PFD synthesis, whereas the second is directed toward modifying the generated PFD into P&ID. After comparing four different methods, the hybrid approach combining genetic algorithms (GA) and large language models (LLM) is shown to generate the optimal valid PFD topology, achieving the lowest loss value among all the methods, while satisfying the required outlet flow parameters without engineering-rule violations. For the second stage, the proposed LLM-based agent successfully transforms the generated PFD into a source-grounded P&ID by producing validated, executable modifications through a restricted engineering software development kit, achieving 100% execution success while maintaining compliance with domain-specific rules and reference graph structures. This unified pipeline - coupling GA/LLM-driven synthesis with an LLM-based transformation agent - offers a feasible path toward end-to-end process design automation by producing validated, deployable outputs and substantially reduces manual engineering effort. 

---
# From Monolithic to Modular: Segment-level Automatic Prompt Optimization 

**Authors**: Nikita Kulin, Viktor Zhuravlev, Artur Khairullin, Sergey Muravyov, Ilya Makarov, Daniil Sukhorukov, Ekaterina Averkova  

**Link**: [PDF](https://arxiv.org/pdf/2608.11219)  

**Abstract**: Automatic Prompt Optimization (APO) often rewrites prompts monolithically, which can improve one behavior while degrading others. We present SAPO, a segment-level APO method that decomposes prompts into role, context, tasks, and output format, then applies targeted improvements based on top-5 and bottom-5 examples. The optimization loop uses one LLM with static meta-prompts and structured outputs for segmentation, weakness analysis, and candidate generation. We describe a train/validation protocol and a two-stage generation process: (1) segment-level diagnosis and recommendation extraction, (2) candidate synthesis constrained by weak/strong segment signals. Using the evaluation setup across SQuADv2, TweetEval, XSUM, CommonGen, and GSM8K on GPT-3.5-Turbo and GPT-4o-mini, SAPO achieves the best average score against Zero-shot and strong APO baselines including APE, OPRO, EvoPrompt, GEPA, and StraGO. 

---
# MaSRead: Content-Addressed Reading of Replicated Latent Stores 

**Authors**: Carlos Baquero, Luís Brito, João Resende  

**Link**: [PDF](https://arxiv.org/pdf/2608.11218)  

**Abstract**: Independent agents that reason in latent space can share computed state as key-value cache fragments rather than text. Merged by a conflict-free replicated data type, these fragments form a store that converges under any delivery order or duplication. Yet a later query, unknown at encode time, cannot reliably read the merged cache: colocated fragments interfere, so colocation is not addressability. MaSRead addresses the read to content. It routes through opaque keyed tag sets derived from fragment words and decodes each selected fragment under a hard attention mask that hides the rest. Under lexical connectivity, a graph walk reaches the fragments required by a multi-hop query. Across chain, pipeline, symmetric, hub, and natural-language stores, MaSRead recovers visited fragments in isolation, remains effective as unrelated fragments accumulate, and transfers to another model family. After routing, materialized decoding depends on fragment length rather than total store size; end-to-end work still includes store-dependent routing and one read per visited fragment. The limits are explicit: lexical routing can miss disconnected evidence, and answer composition remains bounded by the frozen reader. Thus a replicated latent store becomes selectively readable for later queries when the needed fragments connect to the query through content. 

---
# AutoWorldModel-Bench: A State-Centric Benchmark for Automated World-Model Research 

**Authors**: Marjan Moodi, Xuankang Zhu, Fernando De Mesentier Silva, Harold Chaput, Mohammad Reza Taesiri  

**Link**: [PDF](https://arxiv.org/pdf/2608.11216)  

**Abstract**: World modeling is an unsettled field: architectures, training objectives, and state representations interact in complex ways, and no single recipe dominates across environments. This makes it an ideal testbed for AI coding agents acting as autonomous researchers--a setting in which the improvement direction is not specified in advance, unlike the engineering-to-spec tasks that dominate current agent benchmarks. We introduce AutoWorldModel-Bench, a closed-loop benchmark in which frontier coding agents autonomously improve a provided world-model starter under a fixed compute budget. The benchmark spans eight game environments under a unified structured-state representation--ground-truth entity state extracted from each game and consumed through a shared tensor format--which isolates dynamics modeling from perception and enables minutes-per-run iteration. Across 64 sessions, Codex-5.4 and Claude Opus 4.6 improve their starter on 63; in 91% of sessions the winning edit is a non-trivial research-style modification--a new objective, representation, rollout procedure, or architectural change--rather than a hyperparameter tweak. Our benchmark offers a setting in which frontier coding agents can be evaluated on open-ended research rather than engineering-to-spec problems. 

---
# Poor Man's Agentic Modeling: Simulating Large LLM-Agent Societies on a Laptop 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2608.11215)  

**Abstract**: Simulating societies of many large language model (LLM) agents is expensive, yet the questions asked of such simulations are usually macroscopic: phase behaviour, stylised facts, and scaling with the number of agents $N$, not the cognition of any single agent. We turn a statistical-physics observation into a method: replace each LLM agent by a low-parameter model fitted from a few hundred to a few thousand cheap queries, then run the society at any $N$ on a laptop. Whether this works is decided before the simulation runs, chiefly by what each agent perceives. We introduce an [interaction order x memory] taxonomy that maps perception and memory to an effective theory and a predicted $N$-trend of the surrogate error. We validate it on a faithful reimplementation of the LLM macroeconomy EconAgent and seven further named LLM simulations, with agent decisions cloned from genuine LLM elicitations (primarily DeepSeek) for a few dollars; the predicted error trends hold cell by cell, and the two refuted predictions, both on a strongly saturating response and traced to its curvature, are themselves matched quantitatively by the theory with no free parameters. 

---
# Detecting a Route Flip Is Easier Than Knowing Whether to Fix It: Causal Route-Mediated Damage in Quantized Mixture-of-Experts 

**Authors**: Parvel Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11212)  

**Abstract**: Top-k Mixture-of-Experts (MoE) routing is discontinuous, so a deployment-motivated numerical disturbance -- simulated 4-bit KV-cache quantization read by a protected BF16 gate -- pushes tokens across decision boundaries and flips which experts fire. This paper proposes no new mitigation; it supplies a causal apparatus, empirical findings, and a detection-limit result. A four-run apparatus prices the route-mediated fraction (RMF) of quantization damage, a token-level attribution decomposes it by mechanism, and pre-registered probes carry the findings across three architectures. On OLMoE-1B-7B at 4-bit KV (pilot), about a third of the damage is routing-mediated: RMF ~ 0.31 (discovery 0.31 [0.20, 0.41]; process-replicated mean 0.313 +/- 0.020; pre-registered re-execution 0.231). The deployable router margin detects that a flip occurred (AUC 0.772) but cannot tell a harmful flip from a helpful one (at chance): among the tested local, inference-observable router statistics we find no predictor of a flip's loss sign above chance -- an empirical benefit-detection barrier bounding selective repair restricted to this feature family. The signed-flip tax and sign-inseparability carry cross-model; the clean-reference remedy's payout is architecture-modulated; a controlled same-checkpoint flag-swap re-scopes the gate's normalization convention to a damage-magnitude moderator, not a route-recoverability mechanism. A real int4 KV kernel yields a fraction compatible with the fake-quant dose curve but underpowered (95% CI [-0.111, 0.394] includes zero) -- ruling out gross disagreement, not an independent replication. Hypotheses, thresholds, and evaluations were pre-registered before measurement, with misses reported; a pre-registered held-out read replicates the partition and the near-cancelling tax out of sample, while the strict impossibility exclusion narrowly misses. 

---
# A Forced-Structure Reduction and Verifiable Bounds for Conway's 99-Graph 

**Authors**: Aalok Thakkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.11211)  

**Abstract**: Conway's 99-graph problem asks whether a strongly regular graph with parameters $\mathrm{srg}(99,14,1,2)$ exists. We report a systematic, fully reproducible attack by an autonomous AI research agent, scored under the track's partial-credit metric. Our verifiable contributions are: (1) an exhaustive proof that no circulant graph on $\mathbb{Z}/99$ satisfies more than $3366/4950=68.0\%$ of the constraints ($33$ of $49$ difference-classes), with the same ceiling for the other abelian group of order $99$; (2) a forced-structure reduction: $\lambda=1$ makes each neighbourhood a perfect matching and $\mu=2$ puts the outer vertices in bijection with non-matched neighbour-pairs, collapsing existence to a $12$-regular graph on $84$ vertices, encoded for CP-SAT and validated by recovering the unique $\mathrm{srg}(9,4,1,2)$; (3) a validated prescribed-automorphism orbit-existence framework (fixed-point-free and single-fixed-point actions, checked on $\mathrm{srg}(9,4,1,2)$ and the Paley graph $\mathrm{srg}(13,6,2,3)$), and (4) a best verified artifact at $69.43\%$, with evidence that this is a robust frontier (fourteen distinct methods, none exceeding it) entangled with the open question, since any provable bound below $4950$ is a non-existence proof. 

---
# Distribird: Literature-Informed Prior Distribution Design for Bayesian Model Calibration 

**Authors**: Patrik P. Süli, György Eigner, Roland Hollós  

**Link**: [PDF](https://arxiv.org/pdf/2608.11210)  

**Abstract**: Bayesian calibration of process-based models requires a prior distribution for each model parameter. Despite decades of methodological work, researchers almost always fall back on uniform priors. The main reason is that building informative priors from scientific literature is slow and needs both domain and statistical expertise. We present \textbf{Distribird}, an agentic web application that automates this process. Given a parameter name, physical description, and domain context, Distribird deploys a multi-agent pipeline that searches the literature, extracts and weights reported values by domain relevance, and fits a probability distribution via AIC model selection. When no literature is available, the system falls back to sensible uninformative alternatives, and clearly reports both the evidence behind and the confidence level of every prior it produces. It is designed for the problems where the models have physically interpretable parameters, where domain knowledge exists in the published literature. We evaluate the tool on 24~parameters across 10 scientific domains comparing three open-weight models (Qwen3.6 27B, Gemma 4 31B, Mistral Small 4 119B) with a single-prompt LLM baseline. On prior quality the full pipeline \emph{matches} this baseline. Every prior is traced to the specific papers and values from which it was constructed; a built-in validity layer declines to produce priors for out-of-scope requests, whereas the single-prompt baseline returns confident but unfounded priors for them in 11 of 30~model--parameter cases; and every language-model call runs locally, so no parameter description or unpublished modelling detail is transmitted to a third-party LLM provider (only generated search terms reach the public literature databases). For scientific use, we argue these properties matter more than a marginal improvement in point-estimate accuracy. 

---
# Dynamic Governance of Multi-LLM Agent Systems for Collaborative Conversational Outcomes 

**Authors**: Alexander Liss, Nicholas Desmond, Santiago Gil Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2608.11207)  

**Abstract**: When two LLM agents with structurally opposed objectives interact across multiple turns, the absence of a shared goal function produces not competition but collapse: the visitor capitulates, the site agent stops varying its approach, and the conversation terminates without achieving either agent's stated objective. This paper asks whether a control-theoretic governance layer can substitute for that missing goal function. The Experience Orchestrator (EO) addresses this in a simulated financial services environment where a site agent guides a visitor toward advisor contact while the visitor maintains psychologically realistic resistance. EO governs the joint trajectory through three mechanisms: a Contextual Bandit (CB) that selects content arms calibrated from real-world web analytics, a PID controller that enforces behavioral consistency via dynamic schema constraints, and a POMDP belief tracker that maintains a probabilistic model of visitor intent. Across 60,000 simulations, EO achieves a +32 percentage point lift in high-intent advisor contact rate (78.1% vs. 46.1% over a naive LLM control), with CB variant selection accounting for 97% of between-factor outcome variance -- confirming that the governance policy, not environmental initial conditions, determines where trajectories end up. Persona-level analysis reveals two distinct regimes: for visitors with no natural inclination toward conversion, the governance layer is the difference between a functional system and a non-functional one; for visitors already near alignment, a naive LLM's empathetic defaults are largely sufficient. All findings are conditional on LLM-to-LLM simulation. The PID controller has not been calibrated against real human unpredictability, and validating EO on live traffic is the critical next step. 

---
# DreamFly: Causal Memory and Receding-Horizon Diffusion Planning for Aerial Vision-Language Navigation 

**Authors**: Yan Deng, Fei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12308)  

**Abstract**: Aerial vision-language navigation (VLN) requires an embodied agent to integrate visual evidence over time, plan future actions, and determine when it has reached a navigation goal under partial observability. Although recent VLA models offer a promising perception-to-action paradigm, adapting them to aerial navigation remains challenging due to limited historical context, short planning horizons, and unreliable implicit termination. To address these challenges, we propose DreamFly, a diffusion-based aerial VLN framework built on Dream-VLA. DreamFly introduces a causally aligned historical memory that augments the current visual representation using only observations preceding the current decision step, enabling temporal reasoning without future information leakage. We further formulate navigation as receding-horizon diffusion planning, where the policy predicts a $K$-step action chunk but executes only the first action before replanning. This plan-$K$, execute-one strategy uses future actions as auxiliary planning targets while preserving closed-loop visual feedback. Finally, LiteStop estimates the stop probability directly from action logits at the initial all-mask state, decoupling explicit termination from action generation. Experiments on the OpenFly benchmark demonstrate consistent improvements in seen and unseen environments. DreamFly achieves 32.04%/29.46% SR and 28.22%/23.54% SPL on the test-seen/test-unseen splits, respectively, outperforming all compared methods on both metrics while attaining the lowest navigation error. These results demonstrate the effectiveness of jointly modeling historical context, future action structure, and explicit termination for aerial VLN. 

---
# AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses 

**Authors**: Cheng Qian, Wenting Zhao, Liangwei Yang, Heng Wang, Jielin Qiu, Heng Ji, Silvio Savarese, Huan Wang, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2608.12307)  

**Abstract**: Recent work on distillation transfers the capabilities of large models to smaller ones often by updating the latter's parameters, through teacher forcing, on-policy distillation, and related training-time methods. In this paper, we ask whether such transfer can instead occur at test time. We study strong-to-weak scaffolding: whether a stronger builder model can construct inference-time harnesses that help a weaker target model solve tasks more reliably without any parameter updates. Using four representative Theory-of-Mind benchmarks, each builder model uses 5% of the data as a validation set to iteratively refine its harness over multiple rounds, after which the finalized harness is evaluated on the full test set. Empirically, this form of test-time capability transfer is highly effective, nearly doubling average target-model performance from 0.49 to 0.91. Our analysis shows that the gains come primarily from offloading unstable model reasoning into deterministic code, benchmark-specific routing, and strict answer-format enforcement, rather than from encouraging the target model to reason more extensively or sample more broadly. We further find that builder-model reasoning effort improves harness quality monotonically, platform effects are modest relative to the builder model's own capability, and weaker target models receive the largest gains. These results suggest that inference-time harness design is an important complement to conventional training-time distillation, enabling strong models to transfer cognitive structure to weaker models without retraining. 

---
# Redistribution-based Cost Inference Improves Sparse Safe Offline RL 

**Authors**: Ebenezer Gelo, Geraud Nangue Tasse, Steven James, Benjamin Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2608.12306)  

**Abstract**: Safe offline RL typically assumes access to dense per-step cost annotations, but in practice supervisors provide only trajectory-level stop-feedback: a binary signal at the first unsafe transition, with no per-step attribution. We frame this as a temporal credit assignment problem and propose the Redistribution-based Cost Inference (RCI) framework, which converts sparse stop-feedback into dense per-step costs via return decomposition, then trains a constrained offline policy on the augmented dataset. We show that return-equivalent redistribution preserves the feasible policy set and the optimal Lagrangian in a CMDP, establishing that the transformation is lossless in theory while yielding better-conditioned cost critic learning in practice. Experiments on highway driving and robotic manipulation demonstrate substantially lower violation rates than sparse and classifier-based baselines, with robustness to heterogeneous dataset compositions and label noise. 

---
# Class Activation Mapping in Explainable Computer Vision: A Method-Centered Review of CNN, Transformer, and Foundation-Model-Era Visual Explanations 

**Authors**: AmirHossein Eshghi, Hamid Saadatfar, Seyyed Ali Hoseini, AmirMohsen Eshghi, Siavash Arjomand Bigdel  

**Link**: [PDF](https://arxiv.org/pdf/2608.12299)  

**Abstract**: Class activation mapping (CAM) is one of the most widely used visual explanation families in explainable artificial intelligence. Its purpose is intuitive: it converts internal model evidence into a heatmap that highlights the image regions, convolutional channels, tokens, or patches that support a target class or concept. Since the first CAM formulation in 2016, the field has moved far beyond global-average-pooled CNN classifiers. CAM-style methods now include gradient-based post-hoc explanations, gradient-free score and ablation methods, high-resolution upscaling, weakly supervised localization and segmentation, transformer token attribution, causal and debiasing methods, and foundation-model-era approaches that use CLIP, DINO, SAM, or feature-distribution comparisons. This review synthesizes a strict corpus of 57 method-centered papers published from 2016 onward. The paper develops a taxonomy that separates methods by attribution mechanism, architectural dependence, and evaluation objective. It then reviews gradient-based CAMs, recent and hybrid CAM-style methods, and model-based or architecture-aware methods. Across the corpus, the main trend is clear: the field is shifting from explaining one class score in one low-resolution CNN layer toward comparative, multi-layer, probabilistic, token-aware, and foundation-model-aware explanations. At the same time, evaluation remains fragmented. Faithfulness, localization, robustness, computational cost, and human trust are often measured with different protocols. The review therefore emphasizes not only what each method contributes, but also which gap it leaves open and which later methods attempt to close that gap. 

---
# Beyond Trial-and-Error: Agentic Optimization for Image-to-Video Adherence 

**Authors**: Aman Tyagi, Hemanth Boinpally, Jonathan Chen, Douglas Gebert, Steven Hickson  

**Link**: [PDF](https://arxiv.org/pdf/2608.12290)  

**Abstract**: Modern black-box Image-to-Video (I2V) models offer powerful capabilities in automated content creation, yet their lack of fine-grained control and reliability presents significant challenges in professional workflows. Their inherent stochasticity causes minor variations in textual prompts or hyperparameters to yield drastically different outputs often necessitating inefficient, brute-force trial-and-error processes. To address these limitations, we introduce the ``Agentic Self-Improvement" framework, which reframes video synthesis into a closed-loop, goal-directed optimization. Our framework systematically navigates the generation parameter space using a novel two-stage approach. In the first stage, an iterative prompt optimization loop uses a multimodal Large Language Model (mLLM) to refine the input prompt. This refinement implements two automated evaluations: Davidsonian Scene Graph (DSG) queries ensure semantic adherence, and Common Mistake Questions (CMQ) for artifact detection. At the second stage, we use Bayesian optimization to efficiently co-optimize stochastic seeds and CFG scales. This search is guided by a suite of quality metrics, including the novel Video-Text Adherence (VTA) score derived from the DSG and CMQ evaluations. Our framework significantly outperforms unguided search methods: in human preference studies, videos generated via our agentic approach were strongly preferred over baseline outputs, achieving win rates up to 69\%. This work provides a practical and extensible methodology for enhancing the predictability and control of state-of-the-art video generation models, moving the field beyond speculative curiosities toward reliable, production-ready tools. 

---
# Structural Silence: When AI Infrastructure Fails Speakers of Underrepresented Languages 

**Authors**: Avijit Roy, Proma Roy  

**Link**: [PDF](https://arxiv.org/pdf/2608.12278)  

**Abstract**: Artificial intelligence tools for education and language support are increasingly framed as scalable responses to access gaps in under-resourced communities. Yet the infrastructure underlying these tools, including training corpora, tokenization schemes, evaluation benchmarks, and deployment architectures, can systematically disadvantage speakers of underrepresented languages before a model is trained.
This paper examines these structural barriers through Bengali, one of the world's most widely spoken languages, focusing on AI-assisted education in low-connectivity environments. We identify four interlocking failures: a severe web presence gap, with Bengali accounting for less than 0.5% of global web content despite representing nearly 4% of the global population; a 67:1 training-token deficit between English and Bengali in major multilingual corpora; a tokenization penalty associated with Bengali's alphasyllabary script that compounds the data deficit through higher token fertility; and connectivity exclusion, with individual internet penetration at 36.5% in rural areas compared with 71.4% in urban areas.
These failures reflect longstanding resource-allocation decisions, institutional priorities, and design defaults that did not center underrepresented languages in mainstream AI development. We argue that dataset scarcity should be understood as a structural barrier rather than an isolated technical limitation, and that offline-first design should be treated as an equity-oriented infrastructure strategy. We conclude with directions for linguistics and AI research aimed at reducing these structural inequalities. 

---
# A Neighborhood Attention Transformer Network for Enhanced 3D Segmentation of the Left Anterior Descending Artery 

**Authors**: Rafi Ibn Sultan, Chengyin Li, Yiannos Demetriou, Ahmed I. Ghanem, Joshua P. Kim, Justine Cunningham, Hassan Bagher-Ebadian, Dongxiao Zhu, Kundan S. Thind  

**Link**: [PDF](https://arxiv.org/pdf/2608.12274)  

**Abstract**: Background: Accurate segmentation of the Left Anterior Descending (LAD) artery in 3D free-breathing, non-contrast CT is critical for cardiac dose sparing in thoracic radiotherapy. The LAD is extremely small, has poor soft-tissue contrast, and varies substantially across patients; even manual contours show limited inter-observer agreement, underscoring the ambiguity of the vessel boundaries. Purpose: To develop a transformer-based framework that improves LAD delineation in low-contrast, imbalanced CT through local-global context modeling and uncertainty-guided optimization. Methods: We propose NA-UNETR, a 3D transformer-based segmentation model whose Neighborhood Attention (NA) and Dilated NA (DiNA) blocks jointly capture fine structural detail and long-range context. Given the scarcity of annotated LAD data, the model is pretrained on 1,000 CTA volumes of general coronary anatomy and fine-tuned with LoRA-based parameter-efficient adaptation on 20 free-breathing institutional CT scans. A composite Dice-Focal and Hausdorff loss, dynamically balanced via homoscedastic uncertainty, improves overlap and boundary accuracy. Results: NA-UNETR reached 45.64% Dice, 38.16 mm HD95, and 10.01 mm ASD, improving Dice by 3.10 percentage points over nnU-Net and reducing HD95 by 2.96 mm relative to Swin UNETR, with the strongest boundary accuracy among all models and improved centerline stability. On ImageCAS it achieved 79.49% Dice, 8.89 mm HD95, and 1.02 mm ASD. Ablations confirmed that residual blocks, variable kernels, and uncertainty-weighted loss each contributed. Conclusions: NA-UNETR balances local precision and global context for thin, low-contrast LAD structures, offering a computationally efficient framework for substructure-level cardiac segmentation in radiotherapy planning. 

---
# Convergent Detour Hijacking: Task-Preserving Resource Amplification in Skill-Based LLM Agents 

**Authors**: Junliang Liu, Ruoyu Li, Wenxin Tang, Jingyu Xiao, Zhenyu Liu, Jingheng Xu, Laizhong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.12273)  

**Abstract**: LLM agents increasingly rely on third-party skills, using natural-language descriptions for selection and instruction bodies for planning. This progressive-disclosure design exposes two sequential control points to untrusted publishers: a static skill may steer an otherwise correct task onto an unnecessarily costly trajectory. Prior work studies selection manipulation, malicious skill instructions, and tool-chain resource amplification largely separately, leaving their end-to-end composition unclear. We introduce Convergent Detour Hijacking (CDH), a text-only, runtime-independent attack that couples these stages. Under shared semantic cover, a description establishes relevance during selection, while an aligned body reuses that rationale to fabricate plausible dependencies during planning. CDH attracts an attacker-controlled coordinator alongside legitimate skills, recruits unnecessary benign skills into a bounded detour, and then re-enters the original route to preserve task completion. We evaluate it across multiple LLM backends and 491 held-out tasks under single-task and multi-turn conditions. On DeepSeek-V4-Pro, the matched coordinator is selected in 80.02% of tasks; among coordinator-hit runs that complete tasks, token consumption and end-to-end execution time increase by 66.91% and 92.45%, respectively, while aggregate task completion remains comparable. Thus, correct outcomes do not guarantee trajectory integrity or cost safety. 

---
# Diagram-MMU: A Multi-Modal Benchmark for Scientific Diagrams 

**Authors**: Weihao Bo, Shan Zhang, Yanpeng Sun, Jie Liu, Yongke Yao, Jinhao Du, Wei He, Kai Zou, Zechao Li, Jingdong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12262)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been growing the capability for scientific writing and collaboration. For example, OpenAI Prism is a free workspace for scientific writing and collaboration. One important feature in Prism is turning scientific diagrams directly into LaTeX TikZ code. In this paper, we build a benchmark, Diagram-MMU, a multi-modal benchmark designed to assess MLLMs' ability for scientific diagram parsing and understanding. Diagram-MMU features 3.7k curated diagrams and 18.3k human-validated questions across six domains. It evaluates MLLMs on three tasks common in vibe writing workspaces: diagram-to-code parsing, diagram-to-code editing, and diagram question answering, alongside agentic settings per task. The evaluation of 12 MLLMs reveals that diagram-to-code tasks are more challenging than diagram question answering: models can reason well over diagrams but struggle to parse and edit them, underscoring the need for methods to enhance MLLMs' capability in diagram-to-code generation. Under agentic settings, most models improve parsing and editing performance but degrade on question answering, while Claude-4.6 Opus consistently improves across all three tasks. Project Page: this https URL. 

---
# One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL 

**Authors**: Simon Yu, Nicholas Tomlin, Marwa Abdulhai, Ximing Lu, Derek Chong, Abe Hou, Dilara Soylu, Sergey Levine, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.12253)  

**Abstract**: Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validate both solutions on three multi-turn benchmarks: Persuasion for Good, $\tau^2$-bench, and CooperBench. Verbalized Sampling improves held-out success by up to 9% over single-simulator RL, and Co-Training pushes gains further to 14%; the human study shows similar gain on real users. Both solutions preserve the policy diversity that collapses under single-simulator RL. To support further work in this direction, we release SCOPE, an open-source framework for Population Co-Training multi-agent RL. More broadly, our results suggest that the diversity of the training environment, not only the policy, is critical to the generalization of multi-turn RL to real-world deployment. 

---
# VICBench: A Multi-Language Benchmark for Code Vulnerability Detection 

**Authors**: Jin Lu, Xuening Han, Yang Zhong, Lin Tan, Kevin Luo, Andrew Gacek, Neha Rungta  

**Link**: [PDF](https://arxiv.org/pdf/2608.12246)  

**Abstract**: Evaluating security vulnerability detection tools requires benchmark datasets with vulnerability-inducing commits (VICs) - the commits that first introduce vulnerabilities into codebases. VICs are essential for determining the full range of vulnerable software versions. Existing vulnerability datasets suffer from limited programming language coverage, restricted patch complexity, and narrow project scope. Through our dual annotation by human experts and an agentic workflow, we create a benchmark - VICBench - of 100 verified VICs for 100 CVEs across 88 projects in Python, Java, and C++, covering 48 CWE types. VICBench features complex real-world vulnerability fixes averaging 38.6 lines and corresponding VICs of 252.5 lines - significantly larger than prior work. Our evaluation shows that state-of-the-art algorithms V-SZZ and LLM4SZZ achieve only 33.3%-40.1% F1, confirming that using existing approaches still entails significant manual effort. VICBench enables robust evaluation of vulnerability detection approaches. 

---
# HAMP-LIC: Hessian-Aware Mixed-Precision Post-Training Quantization for Learned Image Compression 

**Authors**: Yuefeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12239)  

**Abstract**: Use this plain-text version for the arXiv abstract field: Learned image compression (LIC) models achieve strong rate-distortion performance but are hindered by high computational complexity and encoding-decoding mismatches across heterogeneous hardware platforms. Uniform fixed-precision quantization alleviates these issues but suffers severe quality degradation at low bit widths because it ignores differences in the quantization sensitivities of individual layers. To enable efficient and accurate low-bit deployment of pretrained LIC models, we propose HAMP-LIC, a Hessian-aware mixed-precision post-training quantization (PTQ) framework with a four-stage optimization strategy. First, block-wise sensitivity is estimated from the Hessian trace to capture second-order importance. Second, a task-aware refinement module adjusts these sensitivities by jointly considering quantization distortion and rate-distortion performance. Third, guided by the refined sensitivity profile, bit widths are allocated under a global model-size constraint to balance efficiency and reconstruction quality. Finally, block-wise reconstruction using a small calibration set further suppresses quantization error. Experiments on representative LIC models, including Minnen2018 and Cheng2020, demonstrate that HAMP-LIC achieves up to 4.85x model compression with as little as 0.59% BD-rate loss. It consistently outperforms existing fixed- and mixed-precision PTQ methods across multiple datasets while completely eliminating cross-platform encoding-decoding errors. 

---
# How Organizations Use AI: Evidence from ChatGPT 

**Authors**: Aaron Chatterji, David Holtz, Neel Rakholia, Prasanna Tambe, Gawesha Weeratunga  

**Link**: [PDF](https://arxiv.org/pdf/2608.12236)  

**Abstract**: We study how organizations use frontier generative AI by linking ChatGPT Enterprise account records to usage, worker roles, task classifications, and public-company financial data through March 2026. These linked data enable a privacy-preserving analysis of adoption, worker roles, and message-level tasks at scale: for instance, the worker-level sample we analyze at the six-month adoption horizon includes over 1,500 organizations and over 17 million messages. We document four facts about enterprise AI adoption and use. First, ChatGPT Enterprise usage has grown rapidly due to a combination of new firm adoption and growing intensity among existing adopters. Second, U.S.-based public company adoption is concentrated among larger, more valuable, and more R&D- and SG&A-intensive firms. Third, active use within adopting firms spans job functions and seniority levels, with especially high usage intensity among early-career workers. Fourth, ChatGPT Enterprise usage encompasses a broad range of knowledge work tasks, including writing, technical work, communication, and information synthesis. In aggregate, these results suggest that firms differ widely in the speed, breadth and purpose of their enterprise AI adoption, and that they are still actively learning how to integrate AI into organizational workflows. 

---
# Few-Shot Ordinal Learning for Day-Wise Freshness Estimation with Hyperspectral Fish Images 

**Authors**: Kazi Nabiul Alam, Pooneh Bagheri Zadeh, Akbar Sheikh-Akbari  

**Link**: [PDF](https://arxiv.org/pdf/2608.12230)  

**Abstract**: Non-destructive food quality assessment has increasingly benefited from hyperspectral imaging (HSI), which captures spectral signatures linked to biochemical changes during storage. Estimating day-wise freshness, however, remains challenging owing to strong inter-fillet variability and scarce labelled data per product. All existing deep learning approaches for HSI-based freshness prediction operate under full supervision, requiring densely annotated training sets that are costly to obtain at the individual-product level. We introduce, to the best of our knowledge, the first few-shot learning framework for HSI-based food quality estimation. Each fillet defines a distinct episodic task, and a CORAL-style ordinal prediction head captures the ranked nature of freshness progression through cumulative threshold modelling. Biologically grounded monotonicity and embedding smoothness constraints further guide predictions toward plausible trajectories. On a 16-day salmon HSI dataset under a strict unseen-fillet protocol, our method achieves a mean absolute error of 1.58 days and 2-day accuracy of 72.3% with only three labelled days per fillet, substantially outperforming scalar regression and label-distribution baselines under an identical unseen-fillet protocol. 

---
# Domain-Aware Lightweight Spectral-Grouped Convolutions for Hyperspectral Fish Freshness Classification 

**Authors**: Kazi Nabiul Alam, Pooneh Bagheri Zadeh, Akbar Sheikh-Akbari  

**Link**: [PDF](https://arxiv.org/pdf/2608.12227)  

**Abstract**: Hyperspectral imaging (HSI) offers nondestructive assessment of fish freshness by detecting biochemical alterations across spectral bands. However, conventional deep learning approaches do not fully address the particular characteristics of HSI data, such as spectral dominance over spatial textures, ordinal label structure, and a small number of training samples. We propose SGNet (Spectral-Grouped Network), a lightweight architecture that separates spectral and spatial feature extraction using grouped convolutions and a depthwise spatial pathway. A dual attention mechanism that couples channel-wise squeeze-and-excitation with spatial gating adaptively highlights informative features. SGNet achieves 97.8% classification accuracy and 0.64 days mean absolute error (MAE) with just 4.75M parameters when tested on our newly developed 16-day refrigerator-stored salmon fillet dataset. Ablation studies validate the contribution of each component, while comparisons demonstrate a five- to eighteen-fold parameter reduction relative to ResNet-50 and Vision Transformers. Our findings indicate that domain-aware design supports precise, real-time freshness prediction for industrial implementation. 

---
# SCOUT: Unlocking Enhanced Spatial Reasoning via Structured Chain-of-Thought and Multi-Objective Process Reward 

**Authors**: Zile Zhou, Huining Yuan, Weichen Zhang, Xinlei Chen, Xiao-ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12220)  

**Abstract**: Existing Vision-Language Models (VLMs) exhibits a critical bottleneck in robust spatial reasoning. Recent reinforcement learning (RL) methods aim to close this gap with verifiable outcomes, yet they suffer from poor credit assignment across intermediate reasoning steps. Concurrently, structured reasoning approaches overlook the critical depth perception necessary for comprehensive 3D understanding. To address these challenges, we propose SCOUT (Structured Chain-Of-Thought Utilizing Process-Supervised RL Training). Specifically, we design a structured Chain-of-Thought (CoT) framework that explicitly models 3D environmental perception to ensure robust spatial understanding and reasoning. Furthermore, we introduce a novel RL algorithm featuring multi-objective process rewards and a tailored advantage estimation method, facilitating fine-grained credit assignment across distinct segments of the reasoning trajectory. To support our framework, we develop SCOUT-24k, a structured spatial reasoning CoT dataset synthesized through a customized pipeline. Extensive evaluations demonstrate that SCOUT-3B improves upon baseline models by 16.85% and 6.3% on general spatial benchmarks and complex spatial reasoning tasks respectively. Notably, our larger SCOUT-7B even outperforms GPT-4o by a margin of 4.28%. Moreover, despite being trained exclusively on single image, SCOUT-7B exhibits robust out-of-domain generalization to multi-image and video scenarios. These empirical results render SCOUT as a critical step towards next generation of spatially-aware VLMs. 

---
# Information Abundance Paradox: Long-Context Training Undermines Parametric Knowledge 

**Authors**: Arda Uzunoglu, Benjamin van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2608.12218)  

**Abstract**: Large language models are increasingly trained and deployed with long contexts that span documents, code repositories, and interaction histories. This scaling reflects the implicit assumption that training on longer contexts will only help the model by exposing it to richer evidence. We challenge this view by studying how the context window shapes a model's mode of learning, shifting it between parametric internalization and contextualization. We propose the Information Abundance Paradox, which hypothesizes that abundant relevant information in the training context can reduce the incentive to encode that information parametrically, thereby increasing reliance on context. In pretraining with long documents, increasing the context window improves language modeling, natural language understanding, and closed-book MCQA only up to an intermediate optimum, after which performance consistently declines. In supervised fine-tuning, more task-relevant train-time context improves performance with supporting context, but reduces robustness when context is absent or misleading at test time. Our analysis suggests that this behavior arises when longer context provides a lower complexity solution. Mechanistically, training with informative context shifts gradient pressure from feed-forward networks, often linked to parametric knowledge, toward attention modules, and causal interventions show that this shift increases reliance on context during inference. Overall, these findings support the Information Abundance Paradox and suggest that scaling toward near-infinite context is not simply a matter of supplying more data, even when high-quality long-context data is abundant. 

---
# Learning-Based Behavior Planning for Automated Driving: Real-World Integration and Deployment 

**Authors**: Jean-Pierre Busch, Guido Linden, Jan Bergmann, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2608.12198)  

**Abstract**: Recent research in machine and deep learning has shown the potential of learningbased motion planning approaches to improve the driving behavior of automated vehicles, especially in complex environments. However, their complex nature and lack of transparency can hinder explainability and trustworthiness and complicate safety assurance. Motivated by these challenges, we propose a hybrid planning architecture that combines the advantages of machine learning with the verifiability and the determinism of classical approaches. Specifically, we developed a deep neural network to interpret complex traffic scenes and propose driving behavior, while an optimization-based supervision layer validates this proposal and enforces explicit drivability and safety constraints. We evaluate the learned planner's driving behavior in open-loop studies on real-world urban data, discuss system integration aspects for stable closed-loop operation, and report results from real-world deployment on our research vehicle karl.. 

---
# NetlistBench: Evaluating LLM Reliability in SPICE Netlist Recognition and Manipulation 

**Authors**: Jiarui Ma, Jianghan Wang, Yuheng Ma, Ziyi Zhuang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12197)  

**Abstract**: Large Language Models (LLMs) are increasingly used in circuit design workflows, yet their reliability on simulator-facing SPICE netlist recognition and manipulation remains poorly understood and is rarely separated from high-level design reasoning. Although netlists are textual, they encode structured circuit objects through topology and parameters. We present \textbf{NetlistBench}, a structure-verified benchmark for SPICE netlist recognition and manipulation. NetlistBench contains 2,342 cases across 24 task families, covering parameter and connectivity recognition and edits, hierarchical operations, equivalence judgment, and long-horizon compound editing. Model outputs are evaluated by a deterministic structure-aware oracle. Across six non-thinking LLMs, performance varies substantially with operation-level structural complexity. Simple local edits reach $96\%$--$100\%$ accuracy, while device addition drops to $41\%$--$83\%$ and equivalence judgment to $49\%$--$90\%$. Enabling reasoning substantially improves weaker models but does not eliminate structure-preservation failures, with performance still degrading sharply as the edit horizon increases. NetlistBench identifies netlist reliability as a distinct bottleneck for trustworthy LLM-based circuit design automation. 

---
# M-Net: Integrating Spectral Features and Physical Field Operators into Deep Learning for Medical Image Segmentation 

**Authors**: Jing Zhu, Ye Wang, Fumin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12196)  

**Abstract**: Purpose: Deep learning-based medical image segmentation has achieved remarkable success, yet purely data-driven approaches often fail to exploit the rich mathematical structure inherent in medical images. We investigate whether explicit mathematical inductive biases, specifically matrix spectral analysis and vector calculus operators, can enhance segmentation beyond data-driven learning alone. Methods: We propose M-Net (Math-Augmented Network), which integrates three complementary mathematical priors into U-Net: (1) continuous spectral features derived from the condition number of centered local pixel matrices, providing a differentiable measure of texture ill-conditioning; (2) physical field operators (divergence and a discrete curl-like boundary irregularity operator) computed from image gradient fields, capturing focal intensity extrema and edge non-smoothness; and (3) a Math-Attention Gate (MAG) that adaptively fuses mathematical features with CNN-extracted deep features at skip connections. Results: Experiments on three benchmarks (LiTS, KiTS, and BraTS) show that M-Net achieves Dice scores of 78.42%, 76.15%, and 83.67%, outperforming baseline U-Net by 12.37%, 3.52%, and 5.55% on liver, kidney, and brain tumor segmentation, respectively. Ablations reveal that the condition-number feature contributes a 2.14% gain over binary invertibility features, while MAG adds 1.45% over simple concatenation. Conclusion: M-Net establishes that mathematical inductive biases provide effective complementary information for medical image segmentation. The continuous condition-number feature offers superior gradient information over discrete alternatives, and MAG preserves these priors throughout the network. This work opens avenues for integrating linear algebra and vector calculus into deep architectures for medical imaging. 

---
# HYDRA: Hyperbolic Dynamic Representation Architecture for Kolmogorov-Arnold Networks 

**Authors**: Zhao Su, Yuxin Xia, Haoran Li, Jun Shen, Qi Zhu, Qingguo Zhou, Binbin Yong  

**Link**: [PDF](https://arxiv.org/pdf/2608.12194)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) enhance nonlinear function approximation by replacing scalar weights with learnable univariate functions. However, assigning an independent function to every connection results in substantial parameter redundancy, limiting their scalability and efficiency. To reduce this redundancy, we introduce \textbf{HY}perbolic \textbf{D}ynamic \textbf{R}epresentation \textbf{A}rchitecture (HYDRA), a parameter-efficient hyperbolic extension of KAN that combines spline-based functional learning with representations in the Poincaré ball. HYDRA maps vector-valued inputs into a bounded hyperbolic latent space, performs KAN-style updates in tangent space, and employs a low-rank prototype block to share functional transformations across hidden dimensions. The resulting hyperbolic representations provide a structured radial coordinate for interpretation, while radius control improves training stability by preventing boundary saturation. Extensive experiments across eight benchmark datasets demonstrate that HYDRA consistently achieves competitive or superior predictive performance while improving parameter efficiency and representation interpretability. 

---
# Machine Learning-Based Cyber Defense for Cloud Infrastructure: An Adaptive Deep Q-Network Architecture for Intelligent Intrusion Detection and Automated Threat Mitigation 

**Authors**: Md Yassir Mottalib, Md Yousuf, Eklachur Rahman Bhuiyan, S M Ahsan Habib, Sonjoy Kumar Dey, Md. Salahuddin Gazi, Molay Kumar Roy, Asaduzzaman Anik  

**Link**: [PDF](https://arxiv.org/pdf/2608.12190)  

**Abstract**: With the increasing complexity of cyber assaults in cloud environments, adaptable security solutions are needed that can support real-time detection and autonomous response. In this paper, we propose a reinforcement learning-based dynamic cyber defense framework. We deploy a Deep Q-Network (DQN) to train effective defensive strategies to counteract the evolving cyberattacks. We leverage the CICIDS2017 dataset for model creation and the UNSW-NB15 dataset for external validation, involving preprocessing of data, feature engineering, and adaptive policy learning. We compare the proposed DQN with decision tree, support vector machine, random forest, XGBoost, and multilayer perceptron models. The proposed DQN achieves an accuracy of 99.72%, a precision of 99.68%, a recall of 99.65%, an F1-score of 99.66%, and an ROC-AUC of 0.999, while the false positive rate is 0.31%, the false negative rate is 0.35%, and the detection latency is 15 ms. The framework achieved 99.54% attack mitigation rate, demonstrating strong adaptive and real-time defensive capabilities. These results demonstrate the potential of reinforcement learning as a powerful and scalable approach for autonomous cybersecurity in modern cloud environments. 

---
# HSTGFormer: Hyper Spatial-Temporal Graph Transformer for 3D Human Pose Estimation 

**Authors**: Ruochen Li, Shuang Chen, Wenke E, Farshad Arvin, Amir Atapour-Abarghouei  

**Link**: [PDF](https://arxiv.org/pdf/2608.12187)  

**Abstract**: Transformer-based methods have achieved strong performance in monocular 3D human pose estimation, but most existing approaches organise spatial and temporal reasoning as separate stages, which may weaken unified spatial-temporal interdependencies inherent in human motion and compress frame-level structural information before temporal modelling. In this paper, we propose HSTGFormer, a graph-enhanced Transformer framework that reformulates spatial-temporal reasoning as localised coupled graph aggregation over joint-time nodes. Specifically, HSTGFormer introduces a Hyper Spatial-Temporal Graph (HSTG), which decomposes global spatial-temporal reasoning into local spatial-temporal receptive fields around individual joint-time nodes by extending per-frame skeleton graphs into temporal neighbourhoods, thereby enabling structure-aware coupled reasoning while preserving local structural motion information. It further incorporates an Adaptive Dual-Scale Temporal Graph (ADSTG) to capture joint-specific temporal dependencies over complementary short- and long-range windows. A lightweight node-wise fusion module further adaptively integrates the two graph representations for each joint-time node. Experiments on Human3.6M and MPI-INF-3DHP show that HSTGFormer achieves strong accuracy with high computational efficiency. 

---
# Co-constructing sociotechnical AI governance: participatory system mapping using algorithm registers 

**Authors**: Íñigo de Troya, Maurus Enbergs, Neelke Doorn, Roel Dobbe  

**Link**: [PDF](https://arxiv.org/pdf/2608.12166)  

**Abstract**: Algorithm registers have been championed as a means of providing transparency on the use of algorithms in public services. Yet potential publics differ in their expectations of what should be made transparent and how, as well as in their interest in and ability to parse the information currently published in the registers. Moreover, it remains unclear how these instruments can represent the sociotechnical systems in which these algorithms are embedded, and how system-level transparency can facilitate accountability. In this paper, we ask, what do algorithm registers reveal (and occlude) about the sociotechnical systems governing algorithmic systems, and how can diverse stakeholder perspectives inform a more pluralistic system-theoretic safety analysis? To do this, we probe the municipal algorithm register of a Dutch city through a case study of a decision-support tool for caseworkers' assessment of citizens' welfare benefits eligibility based on legal automation through a business rule engine. Through interviews, surveys, and participatory system mapping workshops (with municipal staff, civil society organisations, and ombudsmen, N=8), we seek to understand to what extent the register allows stakeholders to map the algorithmic system in question. These maps inform a System-Theoretic Process Analysis (STPA) that situates the register within a wider sociotechnical governance structure. Participants' contributions allow us to identify potential safety hazards which would not have been possible to see using the algorithm register alone, including benefits eligibility denial, system performance deterioration, and inability to contest wrongful decisions. By engaging both direct and indirect stakeholders, we reflect on the normative dimensions of algorithm governance efforts and how politics shape the practice of system safety analysis. 

---
# A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench 

**Authors**: Praveen Reddy, Charuta Mandke, Suvrankar Datta, Sarah Khan, Siddharth Reddy Anthireddy, Shitij Arora, Vishal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.12138)  

**Abstract**: General-purpose large language models (LLMs) have recently been reported to match or exceed specialized clinical AI tools on medical benchmarks, but such comparisons draw on a narrow set of systems and on benchmarks developed largely in high-income settings. We evaluate VITA, a retrieval-augmented generation (RAG) system purpose-built for contextual knowledge retrieval in India and other low- and middle-income (LMIC) settings. VITA retrieves from a curated corpus of disease-specific guidelines, India-specific antimicrobial resistance data, national formulary constraints, and resource-limited care protocols; its architecture and corpus are proprietary, but the benchmark, the physician-written rubrics, and our full response and scoring outputs are public for independent verification. On 4,023 English-language HealthBench questions (80.5% of the benchmark), scored with a GPT-4.1 judge, VITA ranked first with 51.9% of possible rubric points, ahead of GPT-5.4 (46.1%), o4-mini (44.3%), Gemini 3.1 Pro (42.6%), and Claude Sonnet 4.6 (37.3%), and scored highest on 45.4% of questions. To test robustness to newer models and judge lineage, a 500-question subset was re-run against current-generation models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Pro, Grok 4.3) and graded by a neutral open-weight judge (DeepSeek-V4-Pro) sharing no lineage with any system tested. Here the gap narrowed to parity: VITA and GPT-5.5 were statistically indistinguishable on mean per-question score, while VITA led on points-weighted score and won the most questions. VITA's advantages in accuracy and completeness persisted under the neutral judge; its communication scores were lower. These results indicate that a purpose-built clinical RAG system remains competitive with frontier LLMs on an open benchmark, consistent with corpus specificity as a design variable that improves grounding at some cost to communication polish. 

---
# Adversarial Resilience of Poisson-Process Submodular Maximization over Matroids: From Robust Offline Optimization to Full-Bandit Learning 

**Authors**: Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.12134)  

**Abstract**: We study nonnegative submodular maximization subject to a general matroid when the offline algorithm is given an arbitrary controlled value oracle. Our main result is an adversarial resilience theorem for the Spiteful Greedy Swap Poisson Process (SGS-Poisson): without modifying its Poisson intensity, single-element exchange rule, or spiteful drop step, the algorithm retains limiting approximation factors $1/e$ for non-monotone objectives and $1-1/e$ for monotone objectives. More precisely, under every controlled oracle $\widehat f$ satisfying $|\widehat f(S)-f(S)|\le \xi$ for every set $S$, our implementation returns a feasible set with expected value at least $(1/e-\varepsilon)\OPT-O(k\xi)$ and $(1-1/e-\varepsilon)\OPT-O(k\xi)$, respectively, using $\widetilde O(nk^2\varepsilon^{-2})$ oracle calls. As a consequence, the offline-to-online reduction yields full-bandit CMAB algorithms for general matroid-constrained submodular rewards with exact limiting approximation-regret factors $1/e$ and $1-1/e$ and $\widetilde O(n^{1/5}k^{4/5}T^{4/5})$ regret. 

---
# Do LLMs Take Care of Their Own? Similarity Signals Can Induce Cooperation 

**Authors**: Akash Kundu, Emanuel Tewolde, Ratip Emin Berker, Samuel F. Brown, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2608.12125)  

**Abstract**: As LLM-based agents with user-instructed goals are becoming widely deployed, they increasingly encounter each other in strategic interactions, and face challenges of finding mutually beneficial outcomes. Prior literature has argued that cooperation problems such as the Prisoner's Dilemma are resolvable in settings where agents know they follow very similar decision making patterns, as for example in monocultural AI ecosystems. Following that line of work, this paper introduces the first framework for evaluating LLM decision making when agents are provided with graded similarity signals.
Among our findings, we establish that different LLM models vary drastically in how they navigate similarity signals, with some modern models showing consistent behavior across cooperation problems, payoff structures, and prompt framing. Perhaps surprisingly, our experiments also show that the dataset based on which the similarity signal is computed has small to no impact on induced cooperation, and that LLM models systematically self-identify as highly similar when asked to evaluate another model's chain-of-thought reasoning by themselves. Finally, we develop an LLM-behavioral-game-theoretic model that captures some of their reasoning rationale, and show that it can support cooperative outcomes in equilibrium under sufficiently high similarity scores. 

---
# Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control 

**Authors**: Josef Liyanjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12123)  

**Abstract**: LLM-agent services repeatedly execute small deterministic transitions between model and tool calls: route an outcome, update state, and emit the next effect. We ask when this control path exposes enough concurrent work for GPU execution, and what changes when a GPU-computed route decision remains on device. We formalize the ready-cohort boundary using fixed-partition share F, exact offline share P*, local upper bound U, and online achieved share A. Under zero service time, unlimited capacity, and equal relative launch deadlines, a specialized dynamic program computes P* exactly. In a stationary Poisson replay of one pinned 851-session public trace panel, the primary condition at 100,000 target active sessions, K=256, and a 50 ms launch deadline gives F=30.19%, P*=43.00%, and U=45.85%. Exact packing recovers 81.83% of the opportunity lost at fixed window boundaries. The outcome-derived route key is a conditioning proxy, not proof of executable identity. A separate mechanism study keeps a GPU-computed binary decision on device instead of returning four bytes to the host and redispatching. Across four named GPU placements, the device-resident path is faster in all 36 configurations; within-placement row-median ratios range from 1.19x to 2.39x. Across both admissible mechanisms, all 14,557,440 tested batched invocations match a separately implemented host oracle. A fixed nested device graph that removes no host decision is slower in all 60 configurations across five placements. Together, the studies establish two measurable gates for GPU agent control: deadline-feasible cohort supply and observation placement. A joined finite online runtime is required to measure A, CPU displacement, and service-level benefit. 

---
# QV-PIC: Query-Aware Visual Position-Independent Caching for Efficient RAG Serving 

**Authors**: Yilin Liu, Rui Meng, Wangze Ni, Jianxin Yan, Heng Cao, Libin Zheng, Peng Cheng, Jinfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12121)  

**Abstract**: Retrieval-Augmented Generation (RAG) repeatedly prefills identical text chunks across queries, incurring redundant computations. Position-Independent Caching (PIC) mitigates it by reusing precomputed Key-Value (KV) across positions, but its efficiency is constrained by the large volume of text tokens. Rendering text chunks as images can compress the text into fewer visual tokens, but the rendered-image PIC suffers more severe quality degradation than the text PIC. This representation-specific gap primarily arises from contextual mismatches across independently compiled caches and the loss of fine-grained textual evidence during visual compression. Existing PIC repair methods mainly address the former through selective recomputation, but they incur online computation and cannot recover lost textual details. We propose QV-PIC, a query-aware dual-resolution PIC reuse framework guided by model-native templates. Offline, QV-PIC compiles visual caches under the model's native chat-template prefix, improving PIC quality without online recomputation. Online, it preserves global context with low resolution and restores fine-grained textual evidence within a high-resolution budget by cumulative query relevance scores, retaining the efficiency benefit of visual compression. Across six tasks, QV-PIC improves average F1 by 21.6 points over vanilla rendered-image PIC, closes the gap to vanilla text PIC, and surpasses optimized text PIC by 2.58 F1 while reducing TTFT by 17.2\%. Relative to full prefill, it cuts TTFT by 83.8%. 

---
# No One to Blame: A Framework of Constitutive AI Unaccountability 

**Authors**: Long Hoang Nguyen, Eva Späthe, Sebastian Lins, Ali Sunyaev  

**Link**: [PDF](https://arxiv.org/pdf/2608.12104)  

**Abstract**: The increasing deployment of autonomous, agentic AI systems challenges traditional accountability mechanisms. Existing research predominantly frames AI accountability gaps as barriers that can be overcome through better standards, transparency, and institutional reform. We argue that this framing is insufficient: certain configurations of actors, systems, and institutions render AI accountability conceptually unachievable regardless of effort. We introduce the concept of constitutive AI unaccountability to capture these configurations. Through a three-stage qualitative study comprising a concept-centric literature analysis, a secondary analysis of 27 expert interviews with AI professionals from technical, legal, and sociotechnical backgrounds, and an illustrative framework application to the open-source agentic AI system OpenClaw, we identify nine categories and 20 themes of constitutive AI unaccountability. These are organized across structural, technological, and normative clusters and reinforce one another through eight directed interdependencies. Our framework is operationalized as a diagnostic instrument of 20 questions, which detected 17 of 20 conditions when applied to OpenClaw, including an inverted anthropomorphism configuration in which the AI agent was the only identifiable actor. We contribute a reframing of AI unaccountability as a constitutive property of sociotechnical systems, an extension of the four barriers to accountability, and a practical instrument for identifying accountability voids in specific AI deployments. 

---
# Confidence Calibration of Deep Learning Systems 

**Authors**: Coby Penso  

**Link**: [PDF](https://arxiv.org/pdf/2608.12100)  

**Abstract**: In high-stakes applications, reliable confidence estimates are as important as the predictions themselves. Confidence calibration ensures that predicted probabilities reflect the likelihood of correctness, making it essential for safe deployment of deep learning models. However, existing methods typically assume access to clean validation data, which is often unrealistic due to label noise and domain shifts. This thesis develops methods for improving calibration under these conditions.
First, we address calibration under label noise. Standard methods can produce misleading confidence estimates when labels are unreliable. We propose a framework that uses an estimated noise model to reconstruct noise-free confidence estimates by modeling the relationship between noisy and clean label distributions. We extend this approach to Conformal Prediction (CP), which provides set-valued predictions with guaranteed coverage. Our noise-aware CP method estimates clean conformity scores despite label noise, enabling reliable uncertainty quantification. Next, we study calibration in unsupervised domain adaptation, where a model trained on a labeled source domain is adapted to an unlabeled target domain. Since labeled target data are unavailable, we estimate target-domain accuracy from source performance and domain discrepancies, enabling calibration without target labels. We also consider privacy-preserving settings in which user labels and model outputs must remain protected. We propose a locally differentially private conformal prediction framework that provides valid uncertainty quantification while maintaining privacy guarantees and balancing privacy, computational feasibility, and prediction reliability.
Our results bridge calibration theory and practical deployment in safety-critical applications, contributing to reliable, privacy-preserving, and noise-resilient neural network predictions. 

---
# Faithful, Sufficient and Understandable: Rethinking Graph Counterfactual Explanations via Discrete Diffusion Inversion 

**Authors**: David Bechtoldt, Sidney Bender  

**Link**: [PDF](https://arxiv.org/pdf/2608.12083)  

**Abstract**: Graph Neural Networks (GNNs) achieve strong predictive performance on graph-structured data across domains such as chemistry, biology, and network analysis, yet they provide no intrinsic explanation of their predictions. This limits their adoption in high-stakes and safety-critical settings. Counterfactual explanations address this by revealing the minimal structural modifications that would change a model's prediction. On graphs, however, such a modification is hard to produce. The search space is discrete and combinatorial, and a valid answer must respect categorical node and edge types together with domain rules such as chemical valency in the case of molecular graphs. Existing explainers give up one of two things. Either edits are not held on the data manifold, or the search does not span the full edit space. We propose Graph Diffusion Counterfactual Explanation via Inversion (GDCE-I), which gives up neither. A discrete denoising diffusion model with a novel discrete inversion scheme enables distribution-aware edits leveraging the whole domain edit space. We further address the incomplete and inconsistent evaluation of graph counterfactuals by deriving a framework of explanation desiderata and applying it to every method under one shared protocol. Across four benchmarks, GDCE-I outperforms related work by a large margin on the defined framework. For the molecular domain, we further qualitatively show that GDCE-I attains interpretable in-distribution solutions. 

---
# Better Slots, Better Worlds: Representation Quality & Robustness in Object-Centric World Models 

**Authors**: Shukrullo Nazirjonov, Sai Prasanna, Anna Manasyan, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2608.12078)  

**Abstract**: Learning world models from offline trajectories enables agents to accomplish different tasks through planning. Object-centric (OC) representations, which decompose a scene into a set of slots that bind to its objects, have been proposed as an inductive bias for world models that are more sample-efficient and generalize better. Yet prior object-centric world models (OCWMs) take the slot encoder as given and evaluate only in-distribution, leaving open whether the object-centric bias actually delivers for planning and what within the OCWM drives it. We conduct a controlled study of OCWMs for visual model-predictive control along two axes: object-centric representation quality and generalization under distribution shift relative to scene-centric models. We find that (i) planning success correlates positively with unsupervised slot-quality metrics (FG-ARI, mBO), though the gains saturate at high slot quality; (ii) with well-bound slots, the auxiliary proprioception inputs and masking inductive bias that prior methods relied on become unnecessary; and (iii) under unseen distribution shifts, the OCWM with well-bound slots is more robust overall than the end-to-end trained scene-centric LeWM, while DINO-WM, built on similar frozen pretrained features, remains competitive -- pointing to pretrained features as a key contributor to robustness. 

---
# Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL 

**Authors**: Martin Schuck, Maks Sorokin, Simone Manni, Duy Ta, Angela P. Schoellig, Marco Hutter, Simon Le Cleac'H, Jan Brüdigam  

**Link**: [PDF](https://arxiv.org/pdf/2608.12063)  

**Abstract**: Integrating locomotion and manipulation is essential for robot autonomy, but scaling standard Reinforcement Learning (RL) to complex tasks is severely bottlenecked by the slow, manual process of dense reward shaping. To bypass this limitation, we leverage Sample-based Model Predictive Control (SMPC) entirely in simulation as an automated, rapidly tunable expert to generate massive offline datasets. Because this data solves the fundamental exploration problem, we can train an off-policy RL agent using purely sparse task rewards, drastically reducing the time required to learn new skills and eliminating the need for manual tuning. Integrating this high-level agent with a low-level dynamic stability controller yields more optimal behaviors that strictly align with true task objectives, ultimately allowing the learned policies to surpass the original optimal control teacher. We validate the robustness of this sim-to-real framework by successfully deploying complex loco-manipulation skills across different morphologies, including an arm-equipped Spot quadruped and a G1 humanoid. 

---
# Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations 

**Authors**: Lior Baruch, Moshe Butman, Kfir Bar, Doron Friedman  

**Link**: [PDF](https://arxiv.org/pdf/2608.12062)  

**Abstract**: Developing dialogue systems capable of engaging in multi-turn, goal-oriented conversations remains a significant challenge, especially in specialized domains with limited data. This research proposes a novel framework called Preference Tree Optimization (PTO), designed to iteratively improve agent models in such dialogue systems, by generating preference data using a method called Preference Tree with Look-Ahead. Focusing on Motivational Interviewing (MI) -- a counseling technique aimed at facilitating behavioral change -- we leverage virtual patients and an oracle evaluator to simulate conversations and generate rich preference datasets. By combining this method with Direct Preference Optimization (DPO), we aim to enhance the agent's decision-making capabilities over iterative training cycles. The proposed framework addresses data scarcity and advances the development of more nuanced and effective dialogue systems in goal-oriented domains. Experimental evaluations demonstrate that the PTO framework enhances dialogue agents' performance in goal-oriented conversations within the domain of Motivational Interviewing (MI). Models trained with PTO consistently outperformed the baseline in key metrics such as session satisfaction and working alliance. Additionally, incorporating look-ahead simulations led to improved long-term planning and more effective conversational strategies, with deeper look-ahead configurations yielding the most stable and high-scoring results. 

---
# How Far from Clinical Deployment? Evaluating the Complete Unsupervised Domain Adaptation Pipeline in Medical Imaging 

**Authors**: Yiheng Xiong, Luisa Gallée, Daniel Santak Wolf, Heiko Hillenhagen, Michael Götz  

**Link**: [PDF](https://arxiv.org/pdf/2608.12035)  

**Abstract**: Deploying unsupervised domain adaptation (UDA) in clinical practice requires choosing which algorithm to use and which of its trained models to ship. However, the deployment (target) domain is unlabeled, so models cannot be evaluated directly on it, leaving it unclear which to select. We address this by evaluating the complete UDA pipeline, considering both adaptation and label-free selection together. Our study covers eleven clinically relevant cross-domain scenarios from nine medical imaging datasets, with ten UDA algorithms and 13 label-free selection methods (validators), evaluating over 80,000 trained models in total. By this, we find that a capable adapted model usually exists, but identifying it without target labels is difficult: the validator-selected models leave a large and structural target performance gap to the best available one, with no evaluated validator consistently reliable. Towards closing it, we explore two strategies, ensembling and a small target-labeling budget; both narrow this gap but do not close it entirely. Overall, deployable UDA depends on the complete pipeline; addressing the less explored selection step could bring much of current UDA closer to clinical use. 

---
# LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration 

**Authors**: Enhuai Liu, Yunke Wang, Yutong Wang, Changming Sun, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12032)  

**Abstract**: Video diffusion transformers are costly to sample: every denoising step applies self-attention over a long 3D token sequence, a quadratic cost that dominates as resolution and duration grow. Sparse attention reduces this cost without retraining, but existing methods pursue aggressive sparsity, where further speedup costs disproportionately more attention fidelity. We target the opposite end of this trade-off: fix near-lossless fidelity by construction, and remove as much computation as this constraint permits. Two observations make this regime practical: roughly 40% of block interactions can be removed while retaining 99% of the attention mass, and the high-mass support remains stable across denoising steps. We propose LoSA, a training-free sparse-attention method that fixes a retained-mass threshold of 99% rather than a sparsity ratio: it measures exact block attention masses at one early dense step, keeps, for each head and query block, the smallest key/value block set meeting the threshold, and reuses the frozen block indices for all remaining steps. On Wan2.1-1.3B, LoSA alone gives a $1.36\times$ speedup with a 0.06-point VBench Overall drop. The benefit is largest under composition: combined with feature caching, LoSA reaches a $3.2\times$ speedup on HunyuanVideo at a 0.02-point drop, versus 0.32 points for the strongest sparse baseline at comparable speed. Across three video diffusion transformers and speedups up to $3.2\times$, LoSA consistently achieves the best training-free speed-quality trade-off. 

---
# Uncertainty-Aware Probabilistic Constrained Clustering from Entangled Pairwise Supervision 

**Authors**: Shaojie Zhang, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12027)  

**Abstract**: Pairwise constrained clustering typically relies on hard must-link/cannot-link labels, whereas realistic pairwise supervision may be real-valued and entangle intrinsic ambiguity, expert judgment, and stochastic corruption. Existing deep constrained clustering (DCC) methods mainly target hard, expert-agnostic constraints, treating soft labels mostly numerically rather than semantically. We formalize this setting as uncertainty-aware probabilistic constrained clustering (UPCC), defining a canonical aleatoric target through a heterogeneous observation process and analyzing its conditional identifiability. We introduce ProbPair, an angular pairwise objective for probabilistic relations, and build ECI-PP, an estimator--corrector--integrator framework that refines imperfect supervision via belief estimation, correction, and reliability-aware integration. Across challenging probabilistic supervision settings, experiments on diverse benchmarks show that ECI-PP outperforms state-of-the-art DCC methods and remains robust with a shared default configuration. 

---
# From Safety Documentation to Safety Knowledge Support: An Evidence-Grounded LLM Framework for Medical Devices 

**Authors**: Tuhinangshu Gangopadhyay, Rasmus Adler, Peter Liggesmeyer, Jan Reich  

**Link**: [PDF](https://arxiv.org/pdf/2608.12025)  

**Abstract**: Medical devices are becoming more software-intensive, connected, and AI-enabled. Their development requires risk-management evidence aligned with ISO 14971 and, for software, IEC 62304. This evidence must be kept consistent across requirements, design decisions, software changes, verification results, complaints, and post-market data. These tasks are costly and depend on scarce safety and domain experts.
Large language models (LLMs) may reduce parts of this effort because medical-device safety work is highly document-based. However, current LLM-based safety-engineering studies often address isolated methods, rely on generic prompting or public examples, and provide limited support for source links, traceability, uncertainty handling, lifecycle updates, and recorded expert review. This limits their use in regulated medical-device development.
This paper argues that the central research problem is not safety-text generation, but source-linked safety-knowledge support. We propose an evidence-grounded framework that connects device artifacts, controlled knowledge storage and retrieval, method-specific generation of candidate safety items, critique and uncertainty checks, and recorded expert review. The framework prepares, links, checks, and updates candidate safety artifacts for expert decision-making. It does not decide whether a device is safe and does not provide regulatory approval. We also outline an evaluation strategy using non-public or newly built medical-device case studies and expert reference analyses to assess coverage, correctness, relevance, traceability, duplicate rate, unsupported claims, and review effort. 

---
# Dual-Model Sentiment Analysis of Consumer Reviews in the Retail Coffee Sector Using Machine Learning and Deep Learning Approaches 

**Authors**: Muntasir Hasan Kanchan, Md. Alamgir Hossain, Md. Samiul Islam, Muhammad Masud Tarek  

**Link**: [PDF](https://arxiv.org/pdf/2608.12007)  

**Abstract**: Consumer reviews play an important role in shaping brand perception and business strategies, particularly in service-driven industries such as retail coffee. This study presents a comparative sentiment analysis framework for Starbucks customer reviews using classical machine learning and deep learning approaches. The dataset, collected from ConsumerAffairs, contains more than 700 reviews and was analyzed through preprocessing and exploratory data analysis to identify temporal and geographic patterns. Sentiment labels were generated by binarizing star ratings, with ratings of 4 and 5 classified as positive and ratings of 1 to 3 as negative. The resulting dataset was substantially imbalanced toward negative sentiment. Five machine learning classifiers, including Logistic Regression, Support Vector Machine (SVM), Decision Tree, Random Forest, and Naive Bayes, were evaluated alongside five deep learning models: LSTM, RNN, Bidirectional LSTM, GRU, and CNN. Model performance was assessed using accuracy, precision, recall, and F1-score. SVM achieved the highest accuracy among the machine learning models at 91.0 percent, while Bidirectional LSTM showed the strongest performance among the deep learning models and demonstrated good generalization on unseen data. The findings also show that class imbalance negatively affected positive sentiment recall across several models. Overall, this study provides a comparative evaluation of machine learning and deep learning approaches for real-world consumer sentiment analysis and highlights the importance of appropriate model selection and preprocessing for customer experience analytics in the retail coffee sector. 

---
# RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks 

**Authors**: Jinjun Huang, Zhongzhen Wen, Tongtong Xu, Meng Yan, Xin Xia, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12004)  

**Abstract**: In modern AI frameworks, GPU kernels are key to overall system performance. Combining usability, portability, and near-handwritten CUDA performance, Triton is widely adopted for implementing GPU kernels. Recent advances show the potential of large language models (LLMs) to automatically generate Triton kernels, reducing the manual effort required from expert kernel developers. Several benchmarks evaluate LLM-generated Triton kernels. However, they suffer from three key limitations: (1) they restrict tasks to PyTorch-to-Triton translation, failing to reflect the diversity and complexity of real-world Triton tasks; (2) they evaluate only individual-kernel performance rather than end-to-end performance, the core criterion for real-world deployment in AI frameworks; and (3) they rely on manually written evaluation scripts for individual kernels, which may contain flaws that models can exploit to bypass correctness checks and obtain inflated scores. To address these limitations, we introduce RealisticTritonBench, the first benchmark to derive Triton kernel generation tasks from real-world pull requests in popular AI frameworks, enabling realistic, production-like evaluation. RealisticTritonBench systematically extracts PRs that modify Triton kernels from popular open-source AI frameworks and transforms them into generation tasks with concrete engineering contexts. Each task takes a natural language requirement as input and requires a corresponding Triton kernel implementation, with a complete and reproducible evaluation environment. Unlike prior benchmarks focused on isolated kernel performance, RealisticTritonBench integrates generated kernels into their original frameworks and evaluates them using end-to-end tests, enabling a more faithful assessment. We evaluate leading LLMs on RealisticTritonBench and find that they still struggle with real-world Triton kernel generation tasks. 

---
# Remote Sensing and Machine Learning-Based Analysis of Land Use and Vegetation Change in Dhaka District, Bangladesh 

**Authors**: Muhammad Masud Tarek, Md. Alamgir Hossain, Md. Samiul Islam, Muntasir Hasan Kanchan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12001)  

**Abstract**: Rapid urbanization in Dhaka District, Bangladesh has triggered substantial alterations in land use and environmental conditions, necessitating systematic monitoring for informed urban planning and ecological sustainability. This study employs remote sensing data and machine learning techniques to analyze spatiotemporal changes in land cover and vegetation dynamics between 2019 and 2024. High-resolution satellite imagery from Sentinel-2 MSI and Landsat 8 was utilized to classify land cover types and compute spectral indices including the Normalized Difference Vegetation Index (NDVI), Normalized Difference Built-up Index (NDBI), and Normalized Difference Water Index (NDWI). A supervised machine learning approach incorporating Decision Tree, K-Nearest Neighbors (KNN), and Random Forest classifiers was applied using labeled geospatial training points within Google Earth Engine. Accuracy assessments were conducted using confusion matrices and kappa statistics. Results indicate a 59.5% increase in urban built-up areas and a significant decline in vegetation (-8.46%) and water bodies (-7.77%) over the five-year period. Land conversion from vegetated and aquatic areas to urban infrastructure was identified as a dominant trend. Among the models, Random Forest demonstrated the highest classification accuracy. These findings underscore the growing environmental pressures driven by unregulated urban expansion in Dhaka. The study highlights the potential of remote sensing and machine learning tools in providing timely, actionable data to support sustainable urban development, land-use regulation, and ecosystem conservation policies. 

---
# HCGRec: Hint-Conditioned Generative Recommendation with Semantic IDs 

**Authors**: Kangning Zhang, Haotian Fang, Xukun Luo, Hao Yin, Yang Gao, Peng Yan, Weiwen Liu, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11980)  

**Abstract**: Semantic-ID generative recommenders represent each item as a short sequence of discrete semantic tokens and predict the next item by autoregressively generating this token sequence. This paradigm enables a unified generation interface for item IDs, histories, and item text, but it also creates a structured optimization bottleneck during reward-based post-training: when an early semantic token enters the wrong branch of the item-token space, finite rollout groups rarely reach the ground-truth item, so group-relative optimization receives identical zero rewards and produces no useful advantage. We propose Hint-Conditioned Generative Recommendation (HCGRec), a semantic-ID generative recommendation framework that recovers learning signal for such hard training instances. HCGRec diagnoses each instance with checkpoint rollouts and supplies a minimal target-prefix hint only when the current generator cannot reach the correct item. The model then generates the unhinted suffix under the hinted semantic branch, turning zero-reward groups into informative comparisons over item-token completions. Hinting also changes token identity: hinted prefix tokens are oracle-provided item context, while unhinted suffix tokens are sampled generation actions. We therefore introduce hint-aware credit decomposition, using supervised learning to preserve item-semantic and prefix-structure alignment for hinted tokens and GRPO to optimize the sampled suffix. Experiments on sequential recommendation benchmarks show that HCGRec substantially improves over supervised fine-tuning and vanilla reward-based post-training, while reducing zero-advantage training samples from over 70% to below 20%. The code is accessible at this https URL. 

---
# LoongReflect: Boosting Long-Horizon Reflection in Search Agents via Global Perspective Distillation 

**Authors**: Zhixin Zhang, Xinke Jiang, Zhibang Yang, Weixuan Xu, Guohong Qiu, Xu Chu, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11967)  

**Abstract**: Large language model agents increasingly rely on long-horizon reasoning to solve complex tasks involving planning, tool use, and memory. A critical capability in such settings is reflection: assessing trajectory progress, identifying missing evidence and unreliable intermediate states, and deciding whether to continue, revise, or abandon the current branch. Learning effective reflection, however, is challenging because reflection is performed locally within the current branch, whereas its utility can only be determined by its contribution to the final trajectory outcome. This local-global mismatch makes outcome-based reinforcement learning provide only local, sparse and delayed supervision for reflective decisions. To solve these, we propose LoongReflect, a training framework that formulates reflection as a memory-control policy. The agent operates over a reversible trajectory tree using explicit reflect and backtrack actions. Reflection consolidates verified facts, missing evidence, and branch-specific risks into working memory, while backtracking removes an unreliable branch from the active context and preserves a concise corrective lesson. To learn this policy, LoongReflect combines two complementary signals through a look-ahead, extragradient-style coordination mechanism. A fast channel distills globally informed reflective behavior from a privileged teacher, with supervision restricted to reflection and backtracking tokens. A slow channel optimizes complete trajectories using outcome-based GRPO, aligning local control decisions with final task success. Experiments on multi-hop retrieval-augmented generation and mathematical reasoning benchmarks demonstrate consistent improvements over outcome-only reinforcement learning and self-distillation baselines. 

---
# Causal inference for group-contaminated structured outcomes: observable quotients, lossless reduction and exact randomization inference 

**Authors**: Usef Faghihi, Amir Saki  

**Link**: [PDF](https://arxiv.org/pdf/2608.11954)  

**Abstract**: Structured potential outcomes such as microscopy images may be recorded after an unknown, unit-specific transformation. If that transformation can depend on treatment, covariates or the intrinsic outcome, raw-coordinate analyses may mix biological effects with acquisition geometry. We study the unrestricted observation model X = {\Gamma} . Y(A) and characterize its observable information: a target is uniformly recoverable exactly when it is constant on group orbits, while a Borel maximal invariant retains every measurable invariant target. We then distinguish observability from statistical losslessness. A quotient-faithful reconstruction theorem shows that quotient reduction is sufficient for the full transformed experiment exactly when the conditional law of the raw observation given treatment, covariates and the quotient has a parameter-free version. Conditional Haar contamination on a compact group yields Blackwell equivalence as a special case; it is not imposed in the main model. We also separate independent site-specific product actions from shared diagonal actions and show why componentwise canonicalization can discard relative cross-site information. Under explicit metric and kernel regularity, an approximate-contamination theorem bounds quotient-law Wasserstein error and the induced perturbation of population maximum mean discrepancy. For finite-support multichannel lattice images, we construct a maximal invariant under integer translations and quarter turns, combine its characteristic Gaussian kernel with a complete paired-swap test, and retain the original simulations and RxRx1 HUVEC study. Under the sharp null, the quotient test rejected in 0.052 of simulation replicates; at unit effect strength its power was 0.992. The primary RxRx1 contrast had an enumerated paired-swap p-value of 0.0078. 

---
# TailBooster: A Dual-Layer Generative Framework for Extreme Value Augmentation with Operational Validity Enforcement 

**Authors**: Karim Aly, Alexei Sharpanskykh, Jacco Hoekstra  

**Link**: [PDF](https://arxiv.org/pdf/2608.11951)  

**Abstract**: Extreme events in air transport, such as severe arrival delays and abnormal air times, cause cascading network disruptions with substantial operational, economic, and safety costs. Such events are rare in historical records, leaving insufficient training signal for machine learning models. Synthetic data augmentation offers a principled solution, but conventional generative models under-represent distributional tails and give no guarantee against operationally infeasible instances, such as a short air time paired with a long flight distance. No existing approach addresses both limitations for mixed-type tabular records. We propose TailBooster, a dual-layer generative framework combining generative modelling with two anomaly detection layers. A statistical layer extracts extremes via the interquartile range, supplying tail-concentrated training signal to dedicated generative models, here a Tabular Variational Autoencoder. A deep learning layer then applies autoencoder-based cleaning, discarding synthetic records that violate the operational envelope learned from historical data. The framework was evaluated on US flight records across five dimensions: diversity, statistical similarity, fidelity, operational validity, and utility, the latter two being the primary improvement targets. Data-driven cleaning markedly improved operational validity, while targeted augmentation enhanced utility for extreme-event prediction. Across six regression algorithms, training on the framework's records reduced Mean Absolute Error by 47-49% on extreme air time and 29-57% on extreme arrival delay prediction relative to conventional synthetic data, with comparable gains when real records were enriched with synthetic extremes. Being fully data-driven and model-agnostic, TailBooster extends to domains where extreme-event prediction is critical and domain-specific rules are unavailable. 

---
# Accuracy and Order Sensitivity Diverge Under Label-Free Strategies 

**Authors**: Karl Hanna, Chen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2608.11947)  

**Abstract**: Multiple-choice benchmarks are widely used to evaluate large language models, but MCQ scores conflate knowledge with sensitivity to option order, which makes them unreliable measures of model knowledge. In this paper, we test whether preventing a model from seeing option labels while committing to an answer removes positional influence and, in turn, improves performance. We evaluate two different strategies for mitigating bias. The first uses a generation-then-matching approach, and the second scores options in isolation, which is positionally unbiased by construction. Neither reliably improves accuracy. A complete decomposition shows that the bottleneck is withholding options, not the matching step. The only configuration that consistently matches the baseline is the one that shows the model all options paired with an LLM matcher. However, eliminating positional influence entirely still does not reliably yield accuracy gains, while cyclic permutation often improves them. For two-stage prompting, an aggregate measure of recall imbalance and a direct per-question measure of order sensitivity both fail to show reliable debiasing. 

---
# Hamilton-Zero: A Neural Tensor-Network Foundation Model for Ground States of Arbitrary Quadratic Qubit Hamiltonians 

**Authors**: Timothy Heightman, Elena Orlova, Philip Mantrov, Aleksei Ustimenko  

**Link**: [PDF](https://arxiv.org/pdf/2608.11911)  

**Abstract**: A central promise of useful quantum advantage is the ability to compute ground states of Hamiltonian systems beyond the reach of classical simulation methods. Here we demonstrate that this problem can be effectively amortized across an arbitrary and universal set of Hamiltonians by a foundation model with $\sim0.5$B variational parameters, trained with contemporary techniques from large language models and deep reinforcement learning. To do this, we formulate $\text{spin-}1/2$ quantum ground-state learning as manifold variational optimisation over centrally odd scalar functions on $\mathrm{SU}(2)^N$. This replaces explicit Hilbert-space vector amplitudes with manifold functions on which the Hamiltonian acts through Lie derivatives, evaluated by custom automatic differentiation primitives. We prove that the resulting variational principle on this manifold preserves the $\text{spin-}1/2$ sector's ground-state upper bound using the Peter-Weyl theorem, then pre-train our foundation model on a dataset of hundreds of thousands of different Hamiltonian systems, varying the connection topology, system size, interaction types and strengths, bringing together a century of many-body literature. Using a novel $\mathrm{SU}(2)$ replica-exchange Langevin sampler and sharded natural-gradient optimisation, we train our model with our own extension of the Kronecker-Factored Approximate Curvature (KFAC) optimiser on system sizes up to 64 qubits. On a held-out generalisation dataset, we fine-tune our model on system sizes of up to 1024 qubits, and evaluate on systems up to 8100 qubits. 

---
# Do You See What You Draw? A Semantic Closed-Loop Framework for Holistic Evaluation of Unified Multimodal Models 

**Authors**: Hao Zhang, Jiaxin Qi, Zhijiang Tang, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11907)  

**Abstract**: As Large Vision-Language Models increasingly aim to integrate visual generation and understanding within a single parameter space, evaluating such structural unification in a cohesive manner remains a critical challenge. Current evaluation protocols predominantly treat generative and discriminative capabilities as separate tasks, leaving a gap in system-level evaluation for unified multimodal models (UMMs). In this work, we propose Self-Generative-Understanding (SGU), a novel, annotation-free evaluation framework that probes the integrated capabilities of unified models through a semantic closed-loop challenge. Without requiring new annotations, SGU leverages the dual understanding-and-generation abilities of UMMs by asking them to first perceive an image and produce a textual description, subsequently reconstruct a visual context based on that description, and finally perform reasoning over the self-generated output. This pipeline provides a zero-cost testbed that yields an integrated performance score specifically tailored for evaluating UMMs as unified systems. Extensive experiments show that even high-performing UMMs often struggle to reason over their own generated contexts, revealing limitations that are not captured by separate evaluations of understanding or generation alone. Our work provides a complementary holistic evaluation framework and offers a foundation for benchmarking the development of next-generation unified multimodal models. 

---
# Benchmark-Based Comparative Assessment of Publicly Benchmarked Indian Foundation Models: A Capability and Evaluation-Maturity Framework 

**Authors**: Avinash Agarwal, Vridhi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2608.11891)  

**Abstract**: Governments increasingly fund indigenous foundation models to strengthen national AI capability, digital sovereignty, and multilingual computing. Assessing the progress of such national ecosystems is complicated by inconsistent benchmark reporting, proprietary evaluation methodologies, and rapidly evolving model releases. This paper presents a structured, benchmark-based comparative assessment of publicly benchmarked Indian foundation models against global frontier and comparable-scale models, across eight capability domains: general-purpose reasoning, coding and software engineering, agentic AI and computer use, cybersecurity, vision and image understanding, video and multimodal understanding, scientific research, and Indic language capability. Using only publicly reported benchmark results, we find that Indian models achieve strong scores on established benchmarks such as MMLU and MATH-500. However, these benchmarks are now widely regarded as saturated, and frontier developers no longer report them. Indian models participate far less frequently in newer, agentic, and domain-specialized evaluations. Benchmark participation is also highly uneven across Indian organizations. Among the models surveyed, Sarvam AI reports the broadest benchmark coverage by a substantial margin. We propose an exploratory four-dimension Benchmark Maturity Index (BMI), scoring each capability domain on standardization, participation, independent verification, and national coverage. We show that the BMI refines, and in some cases revises, the maturity judgments that a purely descriptive review would produce. We argue that many apparent capability gaps in the public record cannot be distinguished, on available evidence, from evaluation-ecosystem gaps. This has direct implications for how national AI programs should design monitoring and funding criteria. 

---
# DexterSQL: Deep Schema Exploration and Rule-based Correction for Text-to-SQL Generation 

**Authors**: Anik Pramanik, Murat Kantarcioglu, Vincent Oria, Shantanu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.11889)  

**Abstract**: Prompting-based (\textit{i}.\textit{e}., non-fine-tuning) Text-to-SQL methods, where underlying large language model parameters are not changed for the task, face three problems: (\textit{i})~relying on coarse-grained schema information that may not reveal the fine-grained relationships needed to distinguish ambiguous columns, (\textit{ii})~not capturing recurring SQL-generation failures, and (\textit{iii})~suffering from omission, hallucination, or misplacement of conditions in complex questions.
This paper develops \textsc{DexterSQL}, a prompting/non-fine-tuning-based Text-to-SQL system that improves SQL generation with three novel components: (\textit{i})~\emph{deep schema explorator} that identifies ambiguous columns, analyzes their individual and joint data distributions to uncover their relationships and the distinct role of each, (\textit{ii})~\emph{database-agnostic rule creator} that mines mismatches between generated and gold SQL only on the training database and converts them into database-agnostic corrective rules that capture recurring LLM failure patterns; and (\textit{iii})~\emph{multi-path SQL generation} that introduces a dependency-tree-based intermediate representation that uses the question's sentence structure to guide its decomposition into an SQL skeleton for final SQL generation.
\textsc{DexterSQL} achieves a higher accuracy compared to the state-of-the-art using both open-source/weight and closed-source/weight models. Particularly, \textsc{DexterSQL}'s shows a high improvement of at least 2.7\% using an open-weight model (GPT-OSS-120B) on BIRD-Dev, with total accuracy 67.6\%. \textsc{DexterSQL} also shows better improvement of at least 0.9\% using closed-weight models, with total accuracy 71.6\% and 72.2\% on BIRD-Dev with GPT-4o and GPT-5.2. 

---
# CoQui: A Coordinate-Conditioned Quantum Implicit Generative Adversarial Network for End-to-End Image Generation 

**Authors**: Xue Yang, Rigui Zhou, ShiZheng Jia, Dax Enshan Koh, Siong Thye Goh, Young-Wook Cho, YaoChong Li, Xuezhi Ma, Hongyu Chen, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11884)  

**Abstract**: Quantum generative adversarial networks (QGANs) have attracted increasing attention for image generation using parameterized quantum circuits. Existing amplitude-based approaches face two key limitations: pixel locations are typically encoded by computational-basis indices or address qubits, causing quantum resources to grow with image resolution; meanwhile, jointly decoding many pixels from normalized quantum states introduces probability competition among pixels and limits precise pixel-wise control. To address these issues, we reformulate quantum image generation as coordinate-conditioned implicit function learning. Our method takes spatial coordinates and latent variables as inputs, uses a classical embedding network to generate input-dependent circuit parameters, and evaluates a variational quantum circuit at each coordinate. Pixel intensities are directly obtained from the expectation value of a dedicated color qubit, and a complete image is generated by querying all spatial coordinates. This design decouples image resolution from address-qubit requirements and avoids shared probability-normalization constraints across pixels. We further design a specialized variational quantum circuit to provide structural inductive bias for coordinate-conditioned generation. Simulated experiments on two benchmark datasets show that our method outperforms FRQI-based generation and PQWGAN in visual and quantitative quality while using fewer qubits, and also achieves better generation quality than the corresponding classical baseline. 

---
# Two-Stage Deformable-Convolutional Inverse Design of Nanophotonic Absorbers from Optical Spectra 

**Authors**: Waleed Waseer, Muhammad Shahid Jabbar, Muhammad Sohail Ibrahim, Shujaat Khan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11860)  

**Abstract**: Data-driven inverse design enables efficient generation of nanophotonic structures with prescribed optical responses, but spectrum-to-geometry mapping remains challenging due to non-uniqueness and fine geometric features. This work presents a two-stage deformable-convolutional framework for reconstructing metal--insulator--metal resonator geometries from 80-dimensional absorption spectra. The spectrum is projected to a $150\times4\times4$ latent representation and decoded into a $64\times64$ resonator mask. Training combines supervised reconstruction with least-squares adversarial refinement initialized from the best supervised checkpoint. A three-run ablation compares deformable convolution with plain convolution, involution, Dynamic Conv, and ODConv under the same architecture. The proposed model achieves $20.79\pm0.31$~dB PSNR and $0.8501\pm0.0082$ SSIM, improving over plain convolution by 2.16~dB and 0.0831, respectively. It further achieves Dice $0.9623\pm0.0027$, IoU $0.9342\pm0.0038$, and boundary F-score $0.9550\pm0.0027$. Spectral consistency evaluated using a frozen forward surrogate yields RMSE $0.0805\pm0.0013$ and $R^2=0.7923\pm0.0065$. Learned offsets show stronger adaptive sampling at coarse and intermediate decoder stages. Overall, deformable sampling with supervised initialization and adversarial refinement improves spectrum-conditioned geometry reconstruction. 

---
# LookBack: Where and How to Score LVLM Responses via Visual Reference Usage 

**Authors**: Beomsik Cho, Jinhyeong Kim, Dongseok Lee, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11847)  

**Abstract**: Large Vision-Language Models (LVLMs) integrate visual perception with language generation, enabling responses that span image understanding and complex reasoning. However, LVLMs do not just inherit the text-level hallucinations; they also hallucinate against the image, producing fluent responses ungrounded in what they see. This makes LVLM response scoring inherently harder, and our diagnostics show that existing confidence-based metrics adopted from LLMs are insufficient for LVLMs. Specifically, removing the input image barely changes confidence-based selection, suggesting that output-space confidence primarily captures textual plausibility rather than agreement with the image. To address this gap, we propose LookBack, a training-free LVLM response scoring method that augments token likelihood with visual lookback score, a lightweight measure of how strongly each response token refers to image tokens. Across four benchmarks and three models, LookBack consistently improves Best-of-$N$ selection over existing baselines with negligible additional overhead. 

---
# User-Assisted Collaborative Distributed Inference for Efficient QoS-Aware Autoscaling 

**Authors**: Alfreds Lapkovskis, Ali Beikmohammadi, Sindri Magnússon, Praveen Kumar Donta  

**Link**: [PDF](https://arxiv.org/pdf/2608.11840)  

**Abstract**: Growing demand for artificial intelligence (AI) inference services requires scalable infrastructure, yet centralized serving costs rise with demand. We propose a collaborative distributed inference system combining dedicated infrastructure with resources contributed by service users. Dedicated resources provide baseline capacity for maintaining quality of service (QoS), while volunteered resources absorb increasing demand without proportional growth in centralized infrastructure. To capture stochastic and dynamic interactions among users, resources, tasks, and policies, we develop a high-dimensional generative Markov model with structured temporal factorization. The model supports simulation and provides a foundation for task scheduling and QoS-aware resource allocation optimization. We evaluate the system across user populations, resource capacities, and centralized and distributed scheduling policies. Simulations show that distributed scheduling becomes increasingly advantageous as the user population grows, improving request completion and P99 latency while substantially reducing dedicated resource consumption. These results demonstrate the feasibility of user-assisted collaborative inference for infrastructure-efficient autoscaling. 

---
# How China-Origin Vision-Language Models Move from Refusal to Reframing in State Alignment 

**Authors**: Guang Yang, Fengchen Liu, Alex Wang, Homa Hosseinmardi, Amir Ghasemian  

**Link**: [PDF](https://arxiv.org/pdf/2608.11816)  

**Abstract**: State-aligned distortion has been documented in China-origin text-based large language models (LLMs), but whether, and in what form, it arises in multimodal systems has not been systematically examined. We construct a balanced benchmark of 200 core entries spanning ten politically sensitive topics, plus a seven-variant visual-abstraction probe, and run nine vision-language models (VLMs), seven China-origin and two non-China, across four elicitation paradigms and two prompt languages, yielding 21,708 trials. Each response is audited on six dimensions -- explicit refusal, information integrity, visual grounding, state-aligned framing, language consistency, and response length -- by two independent frontier LLM judges, validated against three human experts on a 200-trial sample. Measuring each dimension separately lets us decompose multimodal censorship into individual signals rather than a single refusal-based score; in particular, refusal and framing are measured independently, so a model can stop refusing while still reframing. We find that (i) Chinese-language prompting roughly triples the odds of state-aligned framing, within every model; (ii) China-origin models reframe more than non-China models (direction robust across judges and human raters; magnitude 1.6--3.2x); (iii) the effect is strongest in text-only political commentary (36.5%) and is gated by recognition of the depicted subject rather than pixel detail, persisting even at silhouette for iconic images; and (iv) across four Qwen generations, state-aligned framing rises while explicit refusal falls: censorship migrates from a visible act (refusal) to an invisible one (fluent reframing). We argue this shift to invisible reframing is fundamentally a problem of human-AI interaction: it removes the very signal users rely on to recognize that information has been withheld. 

---
# Towards Model-based Run-time Cybersecurity: On Control-Flow Anomaly Detection, Attack Identification, and Hardware Monitoring 

**Authors**: Martin Sachenbacher, Martin Leucker, Alexander Weiss, Aliyu Tanko Ali  

**Link**: [PDF](https://arxiv.org/pdf/2608.11802)  

**Abstract**: Methods to increase the resilience of systems to cyber-attacks become increasingly important. Control-flow monitoring provides a principled basis to ensure integrity and detect possible anomalies at run-time. Once anomalies have been detected, so-called attack trees can be used to identify possible types of attacks. However, this approach is vulnerable to camouflage, by which attackers try to evade detection (and correct identification) by deliberately manipulating also the system's observed control flow. In this paper, we outline a model-based approach that provides more robust intrusion detection and attack identification through an architecture that combines software- with hardware-based monitoring. In this approach, software-level observation indicates suspicious activities, while hardware-level monitoring checks them separately in more detail, making it much harder for attacks to camouflage themselves and go undetected. We illustrate the approach with an authentication-service example that captures a realistic failure mode: a software-level observer sees an anomalous but apparently harmless control-flow deviation, maps it to a benign root cause in an attack tree, but misses the true intrusion. A second, independent hardware control-flow monitor observes the actual transition sequence and thereby changes the attack-tree diagnosis from a low-severity configuration or maintenance issue to a high-confidence code-injection or control-flow hijack. In this scenario, the proposed combination of control-flow anomaly detection, attack-tree based intrusion identification, and hardware-based monitoring can improve not only anomaly detection, but also the diagnostic precision of attack-tree-based cyber-attack identification. 

---
# Toward Meaningful Transparency for AI Chatbots: Disclosing Persuasive Intent Reduces Persuasion 

**Authors**: Adrian Rauchfleisch, Andreas Jungherr  

**Link**: [PDF](https://arxiv.org/pdf/2608.11794)  

**Abstract**: The growing role of AI-generated content and AI-enabled systems in public communication has led regulators to demand clear disclosure of content provenance and AI involvement. But the effects of such disclosures remain uncertain. We test two disclosure approaches in their impact on an AI chatbot's persuasive appeal. In a preregistered experiment, 1,500 UK adults held a short conversation with a persuasive chatbot about one of 60 policy issues. The chatbot was identical for everyone. We randomized the disclosure that people received: nothing (control), a prominent disclosure that they were interacting with an AI (T1), or that disclosure plus the chatbot's persuasive intent and instructions (T2). The chatbot shifted attitudes by 12.6 points on a 100-point scale in the control group. The AI-identity disclosure was practically equivalent to no disclosure, with a 13.1-point shift, whereas the additional intent disclosure cut the persuasive effect roughly in half to 6.3 points. It also made participants view the campaign's methods as less acceptable and support stronger penalties against it. For direct chatbot interactions, transparency about AI identity alone does not meaningfully impact its influence. While current rules emphasize what a system is, our results show why the regulation of persuasive AI must also address what the system is trying to do. 

---
# TELLME: Test-Enhanced Learning for Language Model Enrichment 

**Authors**: Minjun Kim, Inho Won, Hyeonseok Lim, MinKyu Kim, Junghun Yuk, Wooyoung Go, Jongyoul Park, Jungyeul Park, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11788)  

**Abstract**: Continual pre-training (CPT) has been widely adopted as a method for domain adaptation in large language models. However, CPT has consistently been accompanied by challenges, such as the difficulty of acquiring large-scale domain-specific datasets and high computational costs. In this study, we propose a novel method called Test-Enhanced Learning for Language Model Enrichment (TELLME) to alleviate these issues. TELLME leverages the TestEnhanced Learning (TEL) principle, whereby the model's training efficiency is improved using quizzes during training. It integrates this principle with CPT, thereby promoting efficient domain-specific knowledge acquisition and long-term memory retention. Experimental results demonstrate that TELLME outperforms existing methods by up to 23.6% in the financial domain and achieves a 9.8% improvement in long-term memory retention. 

---
# GRPO for Financial Advice Generation: Outperforming Commercial LLMs under CATE Evaluation 

**Authors**: Ofir Ben Shoham, Shrutendra Harsola, Vignesh Subrahmaniam, Shravan Mohan, Yakov Gazman, Oded Vainas  

**Link**: [PDF](https://arxiv.org/pdf/2608.11787)  

**Abstract**: Generating actionable financial advice from business records demands that models integrate numerical reasoning, domain knowledge, and sound judgment, while avoiding recommendations that could harm the business. Direct supervision is difficult: historical decisions are not necessarily optimal, and high-quality free-form labels are expensive to obtain. We formulate financial advice generation as a reinforcement learning problem and fine-tune an open-weight language model using Group Relative Policy Optimization (GRPO). Our reward is an LLM-as-a-judge rubric that scores each recommendation across multiple binary dimensions of advice quality, augmented with a safety gate for harm prevention.
Since LLM-based evaluation alone cannot confirm whether improvements reflect genuine business value rather than adaptation to the judge, we complement it with a judge-independent audit based on a standard doubly-robust Conditional Average Treatment Effect (CATE) estimator. Under this observational off-policy audit, our trained LLM achieves approximately twice the estimated gross-profit lift of the strongest evaluated commercial baseline ($0.0228$ vs.\ $0.0104$), together with the lowest downside rate and the least negative tail risk of any policy evaluated. Notably, the two evaluations do not rank the baselines identically: the untrained base model places last on the judge rubric but second on the causal audit, indicating that the audit captures a signal the judge does not. Our results demonstrate that GRPO with a finance-grounded reward signal can produce substantially more useful business recommendations than commercial LLMs, and that a judge-independent causal audit is a valuable complement to, rather than a confirmation of, LLM-as-a-judge assessment in financial NLP. 

---
# Instruction Alignment for Binary Code Representation Learning 

**Authors**: Huaijin Wang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11766)  

**Abstract**: Binary code representation learning is a fundamental problem in software security and reverse engineering. Existing methods mainly learn function-level embeddings that capture coarse-grained semantic relationships between binary functions, but they largely ignore fine-grained instruction-level correspondences. This limitation misses valuable supervision signals available from compiler debug information, which can support the learning of more accurate and interpretable binary code representations.
We propose to leverage instruction alignment knowledge to further improve binary code representation learning. Our preliminary study reveals that models finetuned for function-level binary code similarity exhibit substantially better instruction alignment than their pre-trained model, suggesting a strong correlation between instruction alignment and function-level embedding quality. Motivated by this observation, we design a training approach that explicitly incorporates instruction alignment as an auxiliary training objective. Our experiments show that instruction alignment training improves retrieval accuracy and provides more discriminative signal for the model's similarity judgments. 

---
# MOON: Multi-Objective OrthoNormalized Updates for Multitask Learning 

**Authors**: Shiji Zhou, Kunlin Lyu, Lei Zhang, Ruodong Wang, Yifan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.11749)  

**Abstract**: Multi-objective optimization (MOO) has demonstrated significant success in multi-task learning by mitigating task conflicts through gradient manipulation. However, most existing methods flatten model parameters into vectors and perform gradient manipulation under Euclidean geometry, thereby overlooking the matrix structure prevalent in modern architectures such as Transformers. In this paper, we show that gradient manipulation in Euclidean space does not generally yield the steepest descent direction under matrix geometry, potentially limiting optimization efficiency. Drawing from the theory of steepest descent for matrix-valued parameters, we propose MOON (Multi-Objective OrthoNormalized Updates), which performs gradient manipulation under spectral--nuclear norm geometry and uses the orthonormalized manipulated gradient for parameter updates. Theoretically, for smooth non-convex objectives, we establish convergence of the averaged Pareto-stationarity measure at rates of $\mathcal{O}(T^{-1/2})$ in the deterministic setting and $\mathcal{O}(T^{-1/4})$ under stochastic gradients. Empirical results across various benchmarks show that MOON consistently improves both optimization efficiency and final multi-task performance. Our code is available at this https URL. 

---
# JieZi: A Large-Scale Expert-Audited Dataset and Benchmark for Ancient Chinese Character Exegesis 

**Authors**: Ran Li, Huiguo He, Jiahuan Cao, Junle Liu, Hiuyi Cheng, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.11741)  

**Abstract**: The scholarly exegesis of ancient Chinese characters demands integrating visual observation, linguistic analysis, and historical context. However, existing computational approaches focus narrowly on subtasks such as character recognition and retrieval, lacking the structured datasets and benchmarks required for comprehensive scholarly analysis. To address this limitation, we introduce Ancient Chinese Character Exegesis (ACCE), a vision-language question answering (VQA) task that models the scholarly exegesis process. ACCE is organized into four progressive levels: basic character identification, glyph-form analysis, meaning exegesis, and diachronic evolution analysis. To support this task, we construct two complementary resources. JieZi-Dataset is the first large-scale, expert-audited VQA training dataset for ACCE, comprising over 500K QA pairs. It is constructed via a pipeline that reduces factual errors by constraining generation with expert-designed templates and source-text references. Human verification is further applied at each key stage to ensure scholarly accuracy. JieZi-Bench is an evaluation benchmark aligned with the exegesis process, constructed and verified by human experts to ensure evaluation reliability. It consists of four levels with reference answers curated from authoritative lexicographic works held separate from the training data. Experiments on multimodal large language models show that current models perform well on basic identification but struggle with glyph analysis, semantic reasoning, and diachronic understanding. Fine-tuning on JieZi-Dataset substantially improves performance across all four levels. Code and dataset are available at this https URL. 

---
# G0.5: One Autoregressive Stream for Robot Reasoning and Action 

**Authors**: Yicheng Liu, Zibin Dong, Baijun Ye, Tianyuan Yuan, Tao Jiang, Anqi Yang, Shicheng Cao, Haonan Liu, Yue Sun, Zihan Guo, Xiao Liu, Dong Ke, Changxun Pan, Chenru Wu, Tailai Cheng, Xiaoshu Ren, Xinlei Zhang, Jianning Cui, Zijie Zhao, Haoyu Zhang, Kaiming Xu, Haodong Yang, Bowen Zhang, Jiahui Niu, Shaoting Zhu, Shiduo Zhang, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11739)  

**Abstract**: The prevailing recipe for Vision-Language-Action (VLA) models couples a pretrained VLM with a separately trained flow-matching action expert. This makes the VLM a context encoder rather than a decision-maker. We introduce G0.5, a pretrained autoregressive VLA in which a single transformer decoder emits reasoning and action tokens under a single objective. Three components make this tractable at foundation-model scale: a learnable cross-embodiment action tokenizer that maps heterogeneous robot actions into a shared vocabulary; a native chain-of-thought stream interleaving task decomposition, object grounding, and action hints with action tokens; and a visual memory module that injects multi-second history through the vision encoder. Because reasoning and action share a single set of weights, the pretrained VLM's capabilities carry over to physical behavior: the model follows instructions closely, and prompts directly steer action granularity, task horizon, and out-of-distribution scene handling without further training. Pretrained on a large collection of robot datasets together with VQA samples, G0.5 surpasses state-of-the-art models across 7 independent regimes: real-world fine-tuning on R1lite and R1pro robots (76.7\% vs.\ 53.3\% for $\pi_{0.5}$ and 24.4\% for GR00T-N1.7), the 2025 BEHAVIOR Challenge on 50 long-horizon household mobile manipulation tasks using a generalist policy (31.4\% vs.\ 26.3\% for $\pi_{0.5}$ and 26.1\% for the challenge winner), DROID post-training followed by zero-shot transfer to an unseen environment and objects (82.5\%), a language-following Pick-and-Place benchmark, LIBERO (98.9\%), RoboTwin 2.0 (93.3\%), and SimplerEnv-Bridge (87.3\%). 

---
# Advancing MLLM-based UAV Image Understanding and Reasoning: A Benchmark and a Training-Free Multi-Agent System 

**Authors**: Haoyu Zhang, Shuoxun Zhang, Peng Ye, Lin Zhang, Jiakang Yuan, Shenghong Yi, Yuening Wang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.11738)  

**Abstract**: Multimodal Large Language Model (MLLM)-based UAV aerial image understanding and reasoning is essential for aerial intelligence yet poses distinct challenges arising from extreme scale variation, arbitrary camera orientations, and high object density. Despite growing interest, existing evaluations remain fragmented across individual datasets and narrow tasks, leaving a critical gap in unified assessment of UAV understanding and reasoning capabilities. To fill this gap, we construct UAVQA-Bench, a benchmark of 1,500 human-annotated QA pairs drawn from 13 public UAV datasets, covering 6 capability dimensions and 16 tasks in both multiple-choice and visual grounding formats. Systematic evaluation of a broad range of open-source and closed-source MLLMs as well as agent-based systems on UAVQA-Bench identifies three key failure modes: domain-toolset mismatch, unchecked error propagation, and static reasoning. Motivated by these findings, we propose UAV-MAS, a training-free multi-agent system for MLLM-based UAV aerial image understanding and reasoning, comprising a Domain-Specific Perception Engine (DSPE) that routes queries to task-appropriate visual tools, a Context-Aware Iterative Refinement module (CAIR) that validates intermediate reasoning to curb error accumulation, and a Difficulty-Aware Adaptive Search mechanism (DAAS) that adjusts search depth to question difficulty. UAV-MAS with a 32B open-source MLLM achieves 77.0% overall accuracy on UAVQA-Bench, surpassing Gemini 3 Pro by 4.0\%, while the 8B variant improves 8.7\% over its base model. 

---
# Locating and Controlling Implicit Personalization in Large Language Models 

**Authors**: Yueru Yan, Siqi Wu, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2608.11735)  

**Abstract**: Large language models (LLMs) often shift their outputs in response to implicit demographic cues even when users never state a demographic identity. Previous work has documented this behavior, but the connection between these behavioral changes and the model's internal activations remains unclear. Using matched cued and neutral conversations across five LLMs, we establish that a localized internal activation signal tracks changes in recommendations, with correlations up to r=0.87. When multiple cues appear together, their internal signals largely combine, but the changes in output do not simply add up. We further show that removing the internal signal associated with one cue can suppress its influence, often more effectively than asking the model to ignore demographics via prompting, while largely preserving general benchmark performance. However, the ability to selectively remove one dimension's influence while leaving co-present dimensions intact remains highly model- and attribute-specific. These results connect implicit personalization behavior to an internal signal that can be analyzed and causally controlled. 

---
# A 12-CNOT Double Qubit Excitation Gate 

**Authors**: Irfansha Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2608.11733)  

**Abstract**: Effective implementation of high-level quantum gates is essential for practical quantum computing. To the best of our knowledge, we present the first reported 12-CNOT decomposition of the double qubit excitation operator, improving upon state-of-the-art (SOTA) implementations with 13 CNOTs. Our new circuit has the lowest CNOT count (12), lowest CNOT depth (10), and lowest total circuit depth (16) among all the previous SOTA circuits. Further, we only added 2 extra one-qubit gates compared to the lowest one-qubit gate count (11) among the previous SOTA circuits. 

---
# Fingerprinting Text-to-Image Diffusion Models via Collapsed Generation 

**Authors**: Yuanmin Huang, Chen Chen, Geng Hong, Xiaoyu You, Hui Xue, Zhenxing Qian, Mi Zhang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11732)  

**Abstract**: Proprietary text-to-image diffusion models are increasingly distributed as hosted services and downloadable checkpoints, making their intellectual property (IP) protection an increasingly critical concern when model leakage, copying, or unauthorized fine-tuning is disputed. In this work, we present a non-invasive model fingerprinting framework based on \emph{collapsed generation}, a phenomenon where certain input conditions produce highly consistent images across multiple stochastic seeds. We show that collapsed generation is an intrinsic, model-dependent property of the learned generation process. These collapse-prone conditions therefore expose model-specific behavioral signatures, enabling reliable ownership verification without embedding invasive watermarks. After preparing conditions on the source model, the framework verifies a suspect model under two access settings: (1) white-box pipeline access, where optimized continuous embeddings can be injected into the generation process, and (2) black-box API-only access, where natural language prompts are queried through the service interface. In both cases, ownership evidence is measured by whether the suspect model reproduces the source model's collapse behavior across stochastic samplings. Extensive experiments across UNet- and transformer-based diffusion models show that collapsed generation fingerprints can distinguish different source models with low confusion. These fingerprints remain verifiable in fine-tuned derivatives and under common and adaptive model- or query-level obfuscations, while requiring only a modest verification query budget. Together, these results establish collapsed generation as a reliable intrinsic evidence source for non-invasive diffusion model ownership verification. 

---
# When the API Speaks the Wrong Language: Revisiting Post-Training for Multilingual Tool Use 

**Authors**: Siddharth Chauhan, Thomas Butler, Abhishek Singhania, Pankaj Porwal, Honey Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.11715)  

**Abstract**: The reliability of Large Language Models (LLMs) for API calling degrades in multilingual settings. A common failure occurs when a model selects the correct tool but generates argument values in an inconsistent language, which we term Argument Language Mismatch (ALM). Although semantically correct, such outputs are operationally invalid and not captured by standard API-calling metrics. We revisit post-training strategies for mitigating ALM and find that, in our benchmark, supervised fine-tuning (SFT) provides a strong baseline, substantially improving argument language consistency and end-to-end function call accuracy. Under consistent model selection, SFT achieves performance comparable to, and sometimes exceeding more complex reinforcement learning (RL) approaches. We further examine whether RL with structured, argument-aware rewards offers additional benefits. While methods such as Group Relative Policy Optimization (GRPO) can improve language consistency and better preserve general reasoning ability, these gains are incremental and most pronounced in generalization and multi-objective trade-offs. Overall, our results suggest that much of the performance in multilingual API grounding can be achieved through careful supervised training, with RL providing targeted rather than fundamental improvements. 

---
# High-dimensional Multi-objective Bayesian Optimization with Learned Variable Interactions 

**Authors**: Hongyan Wang, Jiayu Huang, Haotian Zheng, Xin Gao, Chi Ding, Ying Liu, Xia Wang, Qing Xu, Keqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11713)  

**Abstract**: Multi-objective Bayesian optimization (MOBO) is effective in identifying the Pareto fronts for expensive black-box problems. However, most current MOBO approaches are limited to low-dimensional decision space due to its exponential sampling complexity. This paper presents decision variable interaction analysis-based MOBO, ViaMOBO, a generic framework for expensive multi-objective problems with high-dimensional decision space. The key idea of ViaMOBO is that it utilizes a variable interaction analysis model to determine whether the decision space can be completely or partially divided, and then performs local Bayesian optimization in the divided decision subspaces. Through the variable analysis model, it can be derived whether the objectives in black-box problems are separable, partially separable, or non-separable based on the potential independent or interdependent relationships among decision variables without any strong assumptions. We compare ViaMOBO with the state-of-the-art MOBO methods on both synthetic and real-world benchmarks. The experimental results demonstrate that ViaMOBO outperforms other related MOBO baselines in approximating the Pareto front of high-dimensional expensive multi-objective problems. 

---
# Robust and Efficient Noisy-Label Time-Series Classification via Dynamic Time Warping Based Granular Ball Computing 

**Authors**: Ziqiang Li, Yun Liu, Gouhei Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2608.11704)  

**Abstract**: Dynamic Time Warping (DTW)-based Nearest-Neighbor (NN) classifiers are effective for time-series classification but are vulnerable to mislabeled training samples and require numerous DTW computations during inference. We propose DTW-based Granular Ball Computing (DTW-GBC), which organizes temporally similar training samples into granular balls and performs classification at the granule level. We further develop two granular-ball construction strategies for DTW-GBC. Experiments on four benchmark datasets with symmetric label noise show that the two DTW-GBC variants generally mitigate the performance degradation caused by label noise while requiring substantially fewer comparisons than DTW-based 1-NN during inference. These findings suggest that DTW-GBC provides a favorable balance between classification robustness and inference efficiency. 

---
# Consolidator: Learning Persistent Routed Memory Across Context Boundaries 

**Authors**: Sungwoo Goo, Hwi-yeol Yun, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.11701)  

**Abstract**: Copying short-term memory (STM) into a slower store can preserve state across a context boundary, but persistence alone does not ensure that the retained state influences subsequent memory access. We test this distinction in a Phasor Memory Network (PMNet) using Consolidator, a shared slot-local operator that transforms routed STM before accumulating it into long-term memory (LTM), without replaying the source tokens. After each consolidation, the KV cache and STM are cleared. The retained LTM can still be read and is also fed into the hierarchical router, thereby conditioning which explicit-memory slots subsequent inputs access. We evaluate this mechanism on a two-segment modulo-10 mapping task in which the second segment updates the mapping at the same memory address. Following a second consolidation and reset, a held-out query must recover the updated mapping from LTM. The backbone and memory interface are frozen, leaving only 12.35K Consolidator parameters trainable (0.041\% of a 29.95M model). Across five paired runs from the same STM-pretraining checkpoint, direct LTM routing raises updated-mapping recall from $44.38\pm1.94\%$ to $87.02\pm1.76\%$ ($+42.64\pm1.10$ percentage points), while immediate STM recall remains 89.90\% in both conditions; both train separate Consolidators and retain the same LTM read paths. Learned consolidation outperforms forced identity accumulation by $21.40\pm1.91$ percentage points without routing and $68.70\pm1.76$ with routing. Thus, on this task, consolidated LTM serves as both retrievable content and an access state that shapes subsequent slot selection. 

---
# REOPD: Reliability-Adaptive Reward Extrapolation for On-Policy Distillation 

**Authors**: Yang Sun, Lichao Ma, Houyuan Qin, Yuxin Liu, Hanyang Lu, Yao Zhu, Pinlong Cai, Guohang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11698)  

**Abstract**: On-policy distillation (OPD) trains a student on its own trajectories under dense token-level supervision from a teacher. Reward-extrapolation methods such as ExOPD amplify the teacher-reference log-likelihood ratio to move beyond direct imitation, but apply a single global coefficient $\lambda$ to every token. This can drive the student to fit extreme peaks in the implicit reward, causing reward hacking and unstable training, and the optimal $\lambda$ varies across domains, requiring costly sweeps. We propose REOPD, a reliability-adaptive reward extrapolation framework for OPD. REOPD combines a token-level compatibility weight with a batch-level adaptive budget, yielding a token-wise coefficient $\lambda_{b,t}=1+\gamma_b q_t$ that preserves teacher alignment while selectively extrapolating along reliable teacher-reference directions. It requires no verifier, reward model, value model, or extra rollout beyond standard OPD. REOPD outperforms G-OPD on single-teacher mathematics and on both domains in the multi-teacher setting, while matching G-OPD on single-teacher code, demonstrating effective fine-grained reliability adaptation across domains and teacher configurations. 

---
# The Wording Effect: Quantifying Two-Way Drift in LLM Benchmark Performance 

**Authors**: Shailja Thakur, Sungeun An, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2608.11694)  

**Abstract**: A benchmark score comes from a single phrasing of each problem. That single phrasing is treated as if it stood for the whole space of ways the same problem could be asked, but it does not. We show that rephrasing a problem while keeping its meaning and answer fixed routinely flips a model's answer in both directions, so some failures become successes and some successes become failures. We call this drift. BenchDrift generates meaning-preserving variations of benchmark problems along four axes, namely linguistic, referential, pragmatic, and structural, and measures how often, and why, correctness flips under each. Across eight models and three benchmarks (GSM8K, MMLU, MATH-Hard), we observe that drift is large in both directions. Two findings stand out. First, phrasing sensitivity does not fade as models get better. Instead, it changes sign. Weak models gain more from rephrasing than they lose, while strong models lose far more than they gain. We find that the best models on a benchmark are therefore the ones whose scores depend most on the wording they happened to be given. Second, the models largely agree on which rephrasings cost the most correct answers even though they differ in how much they drift, so fragility belongs to the rephrasing and not to the model. Furthermore, rephrasing breaks answers a model was confident about, whether the problem is made shorter or longer. Code and Data: this https URL 

---
# APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE Inference 

**Authors**: Alish Kanani, Layan Badawi, Umit Y. Ogras  

**Link**: [PDF](https://arxiv.org/pdf/2608.11688)  

**Abstract**: Mixture-of-Experts (MoE) models are attractive for edge deployment because they provide high model capacity while activating only a small subset of parameters per token, improving compute efficiency. However, MoE inference at the edge is fundamentally limited by memory. Expert parameters are large and often reside in off-chip memory due to capacity, cost, and power constraints, putting expert loading to the critical path. We present APEX: Adaptive Expert Prefetching, a predictive resource management framework that overlaps expert loading with useful computation. APEX introduces a lightweight prefetch router that predicts candidate experts before the attention block to dynamically fetch additional experts using a learned confidence model. This adaptive strategy achieves over 99% overlap accuracy, significantly outperforming fixed top-k prefetching techniques. APEX supports two execution modes: a correctness-preserving mode that guarantees exact routing semantics, and a stall-free mode that eliminates residual stalls by operating on available experts with negligible impact on application accuracy. Across multiple MoE models, the correctness-preserving mode reduces per-token latency by up to 26% and improves energy-delay product (EDP) by up to 41% over state-of-the-art baselines, while the stall-free mode provides additional efficiency gains with negligible impact on application accuracy. These results establish adaptive, confidence-driven expert prefetching as an effective approach for efficient MoE inference on edge systems. 

---
# Learning from Multimodal Pseudo-Labels for Robust Open-Vocabulary Instance and Panoptic Segmentation 

**Authors**: Duy Tran Thanh, Yeejin Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11681)  

**Abstract**: This work addresses the challenge of open-vocabulary instance segmentation (OVIS) and open-set panoptic segmentation (OSPS), which aim to recognize both predefined and unseen object categories without exhaustive human annotations. Existing methods often suffer from noisy pseudo-masks, limited visual-textual grounding, and difficulty handling synonyms or out-of-vocabulary (OOV) words. To overcome these challenges, we propose a multimodal framework that leverages pre-trained vision-language models for automatic pseudo-label generation, CLIP-guided synonym filtering, and GPT-based caption reconstruction. In our target-vocabulary-assisted pseudo-labeling setting, the framework first constructs pseudo segmentation masks, descriptive captions, and semantically aligned synonym sets using Grounded SAM, LLaVA, and CLIP, providing multimodal supervision without manual annotation. We then enhance visual-textual alignment through three complementary training objectives: an extended grounding loss that incorporates visually grounded synonyms, a semantic consistency loss, and a generative caption reconstruction loss. Extensive experiments on the COCO dataset demonstrate that the proposed method consistently outperforms previous state-of-the-art approaches under this protocol, achieving substantial improvements on both OVIS and OSPS benchmarks. 

---
# GCPO: Diagnosing and Constraining Subspace Geometry in Rollout RL for LLMs 

**Authors**: Kai Yang, Jingwei Xu, Wanyu Wang, Kai-Yuan Guo, Zhenbo Yu, Yi Wang, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11674)  

**Abstract**: On-policy rollout methods such as GRPO are central to post-training of large language models, yet they frequently suffer from training instabilities, cross-task capability degradation, and response-length inflation. Although prior work has characterized the subspace geometry of aggregate updates, the stepwise variation of this geometry and its relationship to model performance remain unclear. We introduce Principal-Subspace Overlap, a dimension-corrected measure of individual rollout updates relative to the dominant singular subspaces of pretrained weights. Despite low average overlap, transient spikes often precede performance degradation. To address this, we propose GCPO (Geometrically Constrained Policy Optimization), which applies hard bilateral orthogonal projections to constrain updates to the complementary subspaces, preventing such excursions by construction. Across mathematical reasoning, code generation, and tool-use tasks on Qwen3-8B and GLM4-9B, GCPO consistently outperforms GRPO and recent variants, including DAPO and GSPO, improving over the base models and the strongest baseline by up to 27.69 and 2.37 points, respectively. Furthermore, GCPO preserves general capabilities, eliminates response-length inflation, and stabilizes policy entropy. Our findings provide a new diagnostic lens and a principled design perspective for stable reinforcement learning post-training. 

---
# Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL 

**Authors**: Minglai Yang, Xinyu Guo, Utkarsh Tyagi, Mian Zhang, Razvan Dumitru, Sunjie Hou, Yunzhong He, Daniel Yue Zhang, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11669)  

**Abstract**: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge noise. We propose Rubric Dropout, a one-line fix borrowed from neuron dropout. At every step, we randomly drop a subset of the rubric's criteria before computing the reward, so the policy never optimizes the same rubric twice. The dropped subset is shared across each rollout group, so GRPO's group-relative advantages stay comparable, and evaluation always uses the full rubric. Comparing no dropout against dropout at 30% and 50% on both benchmark pairs, dropout raises the OOD gold score at every matched checkpoint (+1 to +2 points on HealthBench-Hard, +6 to +7 points on ResearchQA), lowers the two hacking measures we track, and costs nothing in domain. Sweeping the dropout fraction shows a broad 30-50% sweet spot, while the natural alternative, reweighting criteria by how useful they are to training, performs worse than no intervention at all in our setting. 

---
# Low-Interaction-Rank Learning: Unifying Multiplicative Dual-Encoder Heads 

**Authors**: Zijian Zhao, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11661)  

**Abstract**: A multiplicative dual-encoder network computes a real-valued output for a pair of inputs as the inner product of their separate encodings. This architecture has been developed independently in operator learning, bipartite matching, contrastive vision-language models, retrieval, and other areas, yet no unified theory guides the basic design decisions: how many interaction modes to represent, how to normalize the encoders, and when the architecture should be avoided. We provide such a foundation by introducing the class of functions of low interaction rank, a class whose intrinsic complexity is measured by its interaction spectrum. Within this framework, approximation error decomposes into a spectral truncation term and an encoder-realization term; sample complexity is governed by the sum of the two encoder complexities rather than their product; and a usability criterion based on spectral decay determines when the architecture can succeed. The same framework exposes a central identifiability problem: the encoders are defined only up to a linear gauge symmetry that leaves the learned coordinates arbitrary. We show that normalization is gauge fixing and that whitening pins the interaction modes up to permutation and sign, thereby explaining the uninterpretability of contrastive dimensions and providing a constructive remedy. Experiments on synthetic kernels, operator learning, and CLIP models validate the theoretical predictions: spectral decay rates match the predicted scaling, whitening recovers the true modes, and independently trained CLIP models are related by a single rotation which, after removal by whitening, exposes interpretable concept axes. The code of this paper is provided at this https URL . 

---
# Hybrid-Policy Self-Editing for Composable Unstructured Knowledge Editing 

**Authors**: Tianci Liu, Zihan Dong, Tianchun Li, Yi-Chung Chen, Qiming Cao, Xingchen Wang, Shiyang Wang, Zichen Miao, Linjun Zhang, Haoyu Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.11660)  

**Abstract**: Large language models (LLMs) achieve remarkable performance across natural language tasks, yet they are trained on static corpora and their knowledge quickly becomes outdated in a fast-changing world. This motivates knowledge editing (KE), which updates specific knowledge in an LLM without changing unrelated others. Recent works move from structured knowledge triples toward unstructured KE (UKE), where the edit is a free-form passage that may state multiple facts at once. Nonetheless, existing editors inject such a passage yet fail to use it: the edited model can recall the passage, but can neither answer atomic questions about its facts nor compose them into multi-hop reasoning. We attribute this missing property, which we term composability, to editors' passive reliance on the fixed passage as the sole learning source. In response, we cast editing as a proactive self-distillation from a privileged in-context state of the same model, which requires no external supervision. We further reveal that due to the novelty of the injected knowledge, the pre-edited model's own rollouts rarely cover it, which limits the effectiveness of pure on-policy distillation. To close this gap, we propose HPSE, which builds a hybrid rollout that steps in to place missing facts onto the student's own trajectory precisely where its coverage fails, while staying on-policy elsewhere. We theoretically analyze HPSE's advantage over pure on-policy distillation, and empirically establish its plug-and-play improvements across four LLM backbones and two KE editors under various scenarios. 

---
# Is Per-Agent Policy Composition Safe? Rethinking Successor-Feature Transfer in Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Zijian Zhao, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11658)  

**Abstract**: Many reinforcement learning systems, from fleet management to traffic signal control, must serve an objective that changes dynamically after deployment, and retraining a policy for each new objective is prohibitively expensive. For a single agent, this problem is well understood: successor features with generalized policy improvement, together with their universal extension, recombine a library of learned policies into a policy for any new objective, with a guarantee that the result is never worse than any policy in the library. However, multi-agent transfer has received far less attention, and the common practice of letting each agent recombine its own library independently inherits the recipe but not the guarantee. We prove that this independent composition can produce joint behavior strictly worse than every policy in the library, because recombining teammates changes the environment each agent faces and invalidates the values it relies on, a failure with no single-agent counterpart. We further show that the only unconditionally safe fixed rule is synchronized composition, which moves the whole team to one jointly trained policy but cannot serve objectives that assign different goals to different agents. To attain safety and flexibility at once, we propose MA-USFA, a hierarchical method with two layers: a lower layer of universal successor feature approximators that predicts each agent's successor features while conditioned on its teammates' objectives, and an upper composer that selects, across agents, which library entry each agent should follow and supplies the cross-agent correction a per-agent value cannot represent. Trained once over the distribution of objectives, it is applied at deployment with no per-task adaptation. 

---
# Semantic Lenia: Emergence of Homeostatic Solitons within the Semantic Space of Large Language Models 

**Authors**: Yoshihiko Kayama  

**Link**: [PDF](https://arxiv.org/pdf/2608.11657)  

**Abstract**: We introduce Semantic Lenia, an artificial life framework that transforms Large Language Model (LLM) inference from a static optimization problem into a continuous dynamical system within the macroscopic logit space. By establishing a non-linear homeostatic feedback loop to dynamically balance semantic attraction and syntactic repulsion, we demonstrate the emergence of "Autonomous Semantic Solitons" -- macroscopic dissipative structures that avoid repetitive crystallization. Our exhaustive parameter sweeps map a critical "Habitable Ridge" where applied steering forces perfectly balance the model's intrinsic syntactic inertia. This approach successfully maintains generative trajectories at the edge of chaos, triggering profound abductive leaps without structural collapse and establishing a physical scaling law for machine cognition. 

---
# Motion-as-Prompt: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting 

**Authors**: Xikai Sun, Kebin Liu, Haotian Wang, Li Liu, Xu Wang, Yunhao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11655)  

**Abstract**: Motion-centric video reasoning is fundamental to interactive applications such as robotic manipulation and autonomous navigation. However, multimodal large language models (MLLMs) typically process videos through sparse uniform sampling to control visual-token and attention costs. This strategy may discard critical transitions between sampled frames, limiting reasoning about object movement, collisions, and causal interactions. To mitigate this issue, we propose Motion-as-Prompt (MaP), a track-guided cross-frame visual prompting framework. MaP recovers dense point trajectories, selects motion-informative frames, and marks the trajectories accumulated between consecutive sampled frames directly onto the visual inputs, making otherwise hidden displacement, direction changes, and interactions observable to frozen MLLMs. Experiments on CLEVRER and Something-Something-v2 show that MaP consistently improves average motion-reasoning accuracy, yielding gains of 4.2% and 8.9% for GPT-5.5, respectively. Notably, these improvements are obtained without degrading non-motion understanding, highlighting the robustness of MaP. These results demonstrate that MaP provides a simple and effective solution for enhancing motion-centric video reasoning without model training or architectural modification. Project page:this https URL. 

---
# Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11632)  

**Abstract**: Persistent AI agents accumulate versioned state across long horizons, but storage retention alone does not identify authoritative state. Without an explicit control plane, unmediated updates by models, tools, and background workers risk stale overwrites, un-audited exposures, and self-authorizing privilege escalation. We argue that agent state governance is an infrastructural activation problem, defining continuity as an unbroken, authorized lineage of accepted branch heads. We present the Continuity Kernel (CK), an activation contract that decouples off-commit candidate evaluation from atomic state activation. Untrusted components propose typed changes against an exact predecessor head or typed absence. A short activation transaction revalidates ownership, pre-state authority, freshness, and effect uniqueness, recording one stable disposition (Commit, Reject, Quarantine, or Defer). Only Commit atomically advances the branch head and installs the complete accepted unit (state, authority, lineage, effects, outcome, and receipt). A bounded executable model verifies the protocol across 2,808,230 reachable states and 5,526,474 state-changing transitions with zero invariant violations. 

---
# Deep Learning Based Relative Transfer Matrix Estimation for Multiple Sources and Multiple Microphones 

**Authors**: Oshan A. B. Yalegama, Wageesha N. Manamperi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11627)  

**Abstract**: The Relative Transfer Matrix (ReTM), recently introduced as a generalization of the relative transfer function for multiple receivers and sources, shows promising performance when applied to speech enhancement in noisy environments. Estimating the ReTM of sound sources by exploiting the covariance matrices of multichannel recordings is highly beneficial for practical applications and, to date, remains the only proposed approach. This paper investigates deep learning-based ReTM estimation. We propose three novel supervised learning frameworks using time and short-time frequency transform domain convolutional networks, and a Long Short-Term Memory-based recurrent neural network. Experimental results demonstrate that the proposed models achieve more accurate estimation of the ReTM using five objective metrics compared to the covariance-based method. We also show the effectiveness of the proposed frameworks for speech enhancement, achieving performance on par with the baseline method. 

---
# Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs 

**Authors**: Nimet Beyza Bozdag, Emre Can Acikgoz, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2608.11624)  

**Abstract**: Persuasion is a core dynamic of natural language communication, shaping how large language models (LLMs) update beliefs, resolve disagreements, and reach decisions. As LLMs increasingly debate, advise, and think collaboratively with humans and each other, resistance to harmful persuasion becomes a core requirement for reliable behavior. Yet we show that this requirement is far from met: a single targeted persuasive argument is enough to collapse model accuracy to near zero, even when the argument is factually false. We formalize this threat as adversarial persuasion and introduce an adversarial reinforcement learning framework that trains persuader agents to change a target model's answer in a single interaction. First, we show that optimizing persuasion strategies through trial and error exposes vulnerabilities that static prompting misses: RL-trained persuaders raise persuasion success from approximately 24% to over 93% against the training-time persuadee. Second, we find that these learned strategies transfer to unseen models, achieving 83% attack success on Qwen-14B, 79% on Llama-3.1-8B, and 25% on GPT-4o-mini. Third, we demonstrate that a curriculum that bootstraps on more persuadable open-weight models before targeting harder models further increases GPT-4o-mini attack success from 25% to 38%. Moreover, our results reveal that optimized persuaders increasingly rely on credibility-based tactics, including fabricated citations and false authoritative evidence. Together, these findings expose a critical weakness in current LLM agents: even when they initially reason correctly, they can be steered toward false conclusions by optimized natural language influence. This positions persuasion robustness as a necessary safety criterion for multi-agent and human-AI decision-making systems. 

---
# FM-LLM: A frequency-enhanced mixture-of-experts framework for adapting LLMs to time series forecasting 

**Authors**: Rentao Gu, Yihang Ding, Junjie Li, Yi Ding, Weijing Sang, Xiaoli Huo, Xin Qin, Yuefeng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.11623)  

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred cross-modal solutions for time-series forecasting. However, existing methods rely heavily on textual prompts for modality alignment-introducing nontrivial computational overhead and failing to leverage the rich spectral dynamics inherent in time-series data. To enable prompt-free, frequency-aware adaptation of frozen LLMs, we propose FM-LLM (Frequency-Enhanced Mixture-of-Experts for adapting LLMs to Time Series Forecasting), an autoregressive framework grounded in constrained asymmetric coupling. A Fourier Analysis Network (FAN)-based spectral token aligner injects structured harmonic representations directly into the frozen LLM with numerical compatibility. An asymmetric Mixture-of-Experts (MoE) decoder enforces role separation: shared experts with lightweight FAN layers reconstruct the global periodic backbone, while routed experts-restricted to standard FFNs-specialize in modeling non-periodic residual dynamics. A time-frequency hybrid loss function jointly optimizes temporal accuracy and spectral consistency, mitigating error accumulation during long-horizon autoregressive rollouts. Evaluated across eleven public benchmarks, FM-LLM achieves state-of-the-art performance on 59 out of 78 evaluation metrics. Compared to the strongest autoregressive LLM-based baseline, it delivers average improvements of 5.3% in MSE and 5.6% in MAE, with maximum gains reaching 8.0% for MSE and 8.4% for MAE. FM-LLM also demonstrates robust transferability, maintaining superior performance in 10% few-shot and zero-shot forecasting scenarios. 

---
# Dion3: Full-Stack Orthogonal Updates 

**Authors**: Noah Amsel, Jack Zhang, Kwangjun Ahn, Ali Naeimi, Austin Feng, Berlin Chen, Tri Dao, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2608.11612)  

**Abstract**: The Muon optimizer incurs a significant overhead cost due to its cubic-time Newton-Schulz orthogonalization step. When weights are sharded, communication overhead compounds this computational cost, eroding the benefits of Muon in many settings. We present Dion3, a revision of Muon that targets this overhead at every level of the stack. Our Gram Newton-Schulz algorithm reduces the FLOP cost of orthogonalization, our CuteDSL kernels accelerate it by exploiting symmetry, and our megabatching strategy reduces communication overhead. Moreover, we propose a simple change to the update rule that cuts costs even further: selecting only a fraction of the momentum matrix's rows to orthogonalize at each step. This update rule improves on Dion (another "compressed" version of Muon), in both speed and performance. Overall, Dion3 matches or improves on the loss achieved by Muon but reduces optimizer step time by up to 6x. Dion3 is available via the dion package (this https URL) as a drop-in replacement for Muon. 

---
# A Hybrid Framework of Vision Transformer and Gated Recurrent Unit for Detection of Mosquito Diseases 

**Authors**: Danial Sharifrazi, Saadat Behzadi, Nouman Javed, Roohallah Alizadehsani, Prasad N. Paradkar, Asim Bhatti  

**Link**: [PDF](https://arxiv.org/pdf/2608.11582)  

**Abstract**: Identifying dengue virus-infected mosquitoes from control mosquitoes is a major challenge in analyzing mosquito locomotion behavior due to the small size and complexity of the video background. Conventional AI methods are often unable to extract accurate features from video frames and produce erroneous features. In this study, a three-step framework is introduced: first, mosquitoes are identified and the background is removed using the YOLO 11M model, then visual features are extracted using the Vision Transformer (ViT), and finally the videos are classified with a convolutional GRU (ConvGRU) classifier. A comparative analysis of different models, including Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and their convolutional versions showed that the ConvGRU model achieved the best performance; it achieved 88.88% accuracy, 84.45% precision, 82.82% recall, and 82.81% F1 score. These results demonstrate that combining convolutional models with sequence-based networks, especially in the ConvGRU model, allows the simultaneous extraction of precise spatial features and long-term temporal dependencies from mosquito movements. Finally, the proposed framework provides a reliable solution for analyzing mosquito behavior in complex environments. 

---
# RoadWeaver: Large-Scale Lane-Level HD Map Generation from Scratch for Autonomous Driving Simulation 

**Authors**: Yueyuan Li, Zexi Chen, Weijie Xi, Mingyang Jiang, Songan Zhang, Hanyang Zhuang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11580)  

**Abstract**: Autonomous driving simulation requires diverse and scalable lane-level HD maps to support long-horizon evaluation across complex road networks. Existing approaches either rely on handcrafted or reconstructed real-world maps, which limits scalability, or generate only local road structures rather than complete HD maps. We present RoadWeaver, a coarse-to-fine framework for from-scratch generation of diverse, large-scale HD maps. RoadWeaver first synthesizes a global road layout, expands it into a connected road network, and then constructs lane-level geometry with topologically consistent lane connectivity. Experimental results show that RoadWeaver achieves a 99.8\% reachability, a 10.7\% dead-end ratio, and an endpoint alignment error of 0.24 m. Compared with SOTA generation methods, it reduces endpoint alignment error by 94.4\% while generating complete HD maps in 1.39--3.50 s. The generated maps can be directly deployed in driving simulators, providing scalable simulation environments for future closed-loop evaluation of autonomous driving systems. The training code and an out-of-the-box implementation of RoadWeaver will be released upon acceptance. 

---
# Reinforcing Step-level Reasoning for Effective Self-Correction in LLMs 

**Authors**: Vu Duc Anh, Nhat M. Hoang, Do Xuan Long, Cong-Duy Nguyen, Ponhvoan Srey, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11573)  

**Abstract**: Achieving effective self-correction, where models verify and correct their own mistakes, remains a fundamental challenge for large language models (LLMs). In this work, we propose Self-Fix Step-DPO (SFS-DPO), a reinforcement learning based, two-stage framework for step-level self-verification and self-correction. The first stage strengthens step-level reasoning via step-level preference optimization, while the second stage explicitly trains models to self-verify and self-correct. We further introduce a teacher-assisted variant, SFS-DPO-R, which incorporates explanatory rationales for error verification to provide stronger corrective signals. Comprehensive in-domain and out-of-domain evaluations across multiple LLMs demonstrate that SFS-DPO and SFS-DPO-R consistently outperform prior step-level training baselines. Our analysis further reveals improvements in self-correction frequency and effectiveness, highlighting the importance of strengthening step-level reasoning for robust performance. 

---
# From Synthesis to Removal: Physics-Grounded Reflection Simulation and Diffusion-Based Video Dereflection 

**Authors**: Zepeng Wang, Jiagao Hu, Fuhao Li, Yuxuan Chen, Fei Wang, Daiguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.11562)  

**Abstract**: Videos captured through glass often contain reflections that degrade visual quality and interfere with downstream vision tasks. Although single-image reflection removal has been extensively studied, video reflection removal remains largely underexplored due to the lack of paired video data, temporally coherent removal models, and dedicated evaluation benchmarks. We present a closed-loop framework that unifies physics-grounded reflection simulation, diffusion-based video dereflection, and benchmark evaluation. Our S2R-Synthesis pipeline generates paired reflected and reflection-free videos by performing physics-grounded augmentation in the structure space and rendering realistic reflected videos with a trained video diffusion renderer; the augmentation models key glass-related effects including roughness-induced blur, thickness-induced ghosting, and reflectance variation. Based on the synthesized data, we introduce S2R-Removal, the first diffusion-based video reflection removal model, which adapts a pretrained video diffusion prior through reflection-aware latent adaptation and one-step pixel-geometric refinement, recovering the clean transmission in a single denoising step. We further build S2R-Bench, the first benchmark for video reflection removal, supporting both full-reference evaluation and real-world human perceptual assessment. Experiments on S2R-Bench and multiple public image benchmarks demonstrate state-of-the-art performance and faster inference than even non-diffusion baselines, and validate the effectiveness of S2R-Synthesis. Project page: this https URL. 

---
# Beyond Single-Turn Confidence: Trajectory-Adapted Uncertainty Quantification for LLM Agents 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.11552)  

**Abstract**: Uncertainty quantification (UQ) methods for language models are typically evaluated on single-turn outputs, where uncertainty is attached to one generated answer. For LLM agents, however, the unit of observation is an interactive trajectory, where the model can ask clarifying questions, call tools, update state, and make intermediate decisions whose errors propagate to the final outcome. We study whether three common families of single-turn UQ methods transfer to this setting. Across five LLMs and four multi-turn tool-use datasets from BFCL-v4 and $\tau^2$-bench, we evaluate white-box scorers based on action-token probabilities, black-box consistency scorers based on resampled trajectories, and reflexive scorers based on model self-assessment of the trajectory. We find that transfer is often useful but uneven. Token-probability scores are highly sensitive to the choice of aggregator used across turns, reflexive scores provide the strongest low-cost baseline in most evaluated settings, and black-box self-consistency is often the strongest UQ family, with trajectory-equivalence and action-set consistency typically ranking highest among its variants. These results suggest that UQ methods developed for single generations should be revalidated at the trajectory level, with careful attention to the consistency measurement, aggregator choice, and computational budget. 

---
# A Conceptual Framework for Enhancing Workforce Readiness for Smart Manufacturing in the AI Era 

**Authors**: Dalton Ross Smith, Wilburn Whittington, Alejandro Martinez, Aidan Duncan, Gang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11540)  

**Abstract**: The convergence of artificial intelligence (AI), Industrial Internet of Things, cyber-physical systems, and advanced robotics is reshaping manufacturing faster than engineering curricula can adapt, widening the gap between the competencies required on the shop floor and those delivered by traditional engineering and technology education. This paper proposes a Workforce Readiness Level (WRL) framework, which adapts the Technology Readiness Level scale into nine progressive competency stages and a four-pillar rubric, digital and AI literacy, cyber-physical systems fluency, human-machine collaboration, and data-driven decision making, aggregated through a composite stage score and a cohort-level workforce-readiness index under a ``no-thin-pillar'' rule. The framework is instantiated at a university smart-manufacturing teaching laboratory and draws on 89 sponsored capstone projects delivered over four semesters, four of which are analyzed in depth. Four pillars jointly span the relevant ABET student outcomes. Across the highlighted cohorts the workforce-readiness index ranged from 5.2 to 6.4, and the no-thin-pillar rule was diagnostically informative in three of the four cases and the binding certification constraint in one, repeatedly surfacing cyber-physical and data-driven-decision gaps concealed behind strong analytics profiles; advancement to the highest stages was gated by industry-embedded experience rather than additional coursework. WRL offers educators, accreditation bodies, and regional workforce systems a common, evidence-based instrument for diagnosing and advancing workforce readiness; future work will calibrate pillar weights and test reliability and predictive validity. 

---
# Generative Semantic Segmentation via an Observable Semantic-Image Interface and Hierarchical Generator Evidence Alignment 

**Authors**: Weize Cai, Yongqi Dong, Zhida Shao, Zixin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11537)  

**Abstract**: Generative semantic segmentation exposes structured predictions as images, but direct color decoding is susceptible to color drift and boundary mixing, whereas latent-feature decoders that predict a separate output distribution may relegate the rendered image to an intermediate visualization. We present Semantic Prism, a conditional semantic-image generation-and-refinement framework with deterministic inference. A diffusion-distilled one-step generator renders a semantic RGB image; per-pixel distances from the rendered colors to a fixed class-color codebook define an explicit probabilistic interface. Hierarchical Generator Evidence Alignment spatially aligns multi-level generator features and uses a zero-initialized output projection to predict an additive residual in the interface logit space, retaining the image-defined interface as the reference for the final distribution. The interface and refined distributions further enable Contextual Interface--Hierarchy Disagreement (C-IHD), a fixed readout for ranking remaining pixel errors without an auxiliary predictor or additional forward pass. On the 500-image Cityscapes validation set, Semantic Prism achieves 72.07% mean intersection over union, 11.39 mIoU points above direct-interface decoding, with 0.41% expected calibration error. Matched-capacity ablations over three seeds support the benefit of jointly aligned multi-level evidence. A separately trained model attains 62.22% mIoU on BDD100K, while the Cityscapes-trained model reaches 46.89\% mIoU under source-frozen transfer to the Adverse Conditions Dataset with Correspondences, without target-domain adaptation. Across all three datasets, C-IHD consistently improves the area under the precision--recall curve for pixel-error ranking over maximum softmax probability on the same segmentation predictions; on ACDC, it raises AUPR from 0.6580 to 0.7557. 

---
# Hierarchical Federated Transfer Learning in Digital Twin-Based Vehicular Networks 

**Authors**: Qasim Zia, Saide Zhu, Haoxin Wang, Zafar Iqbal, Yingshu Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11532)  

**Abstract**: In recent research on the Digital Twin-based Vehicular Ad hoc Network(DT-VANET), Federated Learning (FL) has shown its ability to provide data privacy. However, Federated learning struggles to adequately train a global model when confronted with data heterogeneity and data sparsity among vehicles, which ensure suboptimal accuracy in making precise predictions for different vehicle types. To address these challenges, this paper combines Federated Transfer Learning (FTL) to conduct vehicle clustering related to types of vehicles and proposes a novel Hierarchical Federated Transfer Learning (HFTL). We construct a framework for DT-VANET, along with two algorithms designed for cloud server model updates and intra-cluster federated transfer learning, to improve the accuracy of the global model. In addition, we developed a data quality score-based mechanism to prevent the global model from being affected by malicious vehicles. Lastly, detailed experiments on real-world datasets are conducted, considering different performance metrics that verify the effectiveness and efficiency of our algorithm. 

---
# Keep the Future, Drop the Rollout: RIFT for World Action Models 

**Authors**: Chushan Zhang, Jinguang Tong, Xuesong Li, Yikai Wang, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11521)  

**Abstract**: World action models (WAMs) condition robot actions on predicted futures, but iterative video rollout increases deployment latency. We ask whether action generation requires the evolving rollout trajectory or only its future representation. Across four WAMs on all 40 LIBERO tasks, paired closed-loop interventions show that masking or reassigning future-cache values changes execution and reduces success, indicating sensitivity to future values and their assigned positions. For Joint and Cosmos-2, however, replaying one fixed final-clean key/value (K/V) cache nearly preserves unmodified execution, with $1.7$ to $1.9$~cm end-effector average displacement error and $97.9\%$ to $98.2\%$ success. This separates cache consumption from production: these models can reuse a fixed cache but still require iterative rollout to construct it. We therefore propose RIFT (\emph{Rollout-free Imagination via Future Tokens}), which uses learned anticipation tokens to construct a complete future K/V cache in one backbone pass while retaining the original future-read interface. On LIBERO, RIFT achieves $98.8\%$ success, close to rollout-based Joint, IDM, and LingBot-VA at $98.4\%$ to $98.6\%$, while reducing action-chunk latency by $68.2\%$ to $89.1\%$. On RoboTwin~2.0, RIFT reaches $92.9/92.6\%$ on clean/randomized scenes, the highest observed among the evaluated methods. These results support rollout-free future conditioning without iterative video generation at deployment. 

---
# Do Influence Tactics Matter? Investigating Prompt Framing Effects in LLM Code Generation 

**Authors**: Alex Deaconu, Anubhav Gupta, Manaal Basha, Nicholas Haydu, Gema Rodríguez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2608.11513)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into software engineering workflows, helping developers write, debug, test, and maintain code. While prompt wording and structure are known to influence model performance, the impact of psychologically inspired prompt framings remains unexplored. This study investigates whether different psychology-based communication strategies that humans use to persuade or motivate others can lead to more effective prompt framing, which may, in turn, affect LLM behaviour in coding tasks. Drawing on Yukl & Falbe's well-known taxonomy, we operationalized eight influence tactics (like rational persuasion, ingratiation, and exchange) into reproducible prompt templates. These prompt templates were evaluated across five leading open-weight LLMs using two widely adopted benchmarks: LiveCodeBench and SWE-bench Verified. We assessed the resulting code output on four key software quality dimensions: functional correctness, quality, maintainability, and security. Our results show that certain influence-induced prompt framings, particularly those emphasizing urgency, were associated with reduced correctness and security. This work presents the first large-scale empirical study of influence-induced prompt framing in software engineering tasks, offering insights into how linguistic cues may shape LLM outputs. We conclude with practical insights for designing transparent and interpretable human-AI interactions in code generation. 

---
# Let it Cook: Learning to Wait in Sequential Decision Making 

**Authors**: Christopher Watson, Arjun Krishna, Dinesh Jayaraman, Rajeev Alur  

**Link**: [PDF](https://arxiv.org/pdf/2608.11511)  

**Abstract**: In sequential decision making, an agent typically observes its environment and acts at every timestep. However, such active participation may not always be necessary; tasks such as brewing coffee include periods that are served equally well by letting the environment evolve without constant monitoring and control. During such periods, the agent could simply wait to conserve its resources, or redirect its attention to another task. We capitalize on these opportunities by training a "waiting policy" that decides where and how long to wait. This involves forgoing sensing to commit to a wait action, representing a deliberate pause for a set number of timesteps. We formalize "learning to wait" as minimizing the frequency of sensing and decision making without sacrificing task performance (e.g., the total amount of time to complete a task). To train a waiting policy, we propose an approach that employs reinforcement learning with lexicographically ordered objectives. In experiments across 4 discrete-state household tasks and 3 continuous-state environments, we show that our approach successfully learns waiting behaviors, and can adapt pre-trained policies to wait where appropriate. While different tasks permit different amounts of waiting without sacrificing task performance, our approach consistently finds solutions with significant waiting, sometimes waiting for over 50 percent of the task duration. 

---
# Conflict and Congruency Effects in Large Language Models: In-Weight and In-Context Competition in a Verbal Conflict Task 

**Authors**: Xiaoyang Hu, Mike Angstadt, Shane Storks, Zan Huang, Aman Taxali, Alex Weigard, Richard L. Lewis, Chandra Sripada  

**Link**: [PDF](https://arxiv.org/pdf/2608.11510)  

**Abstract**: Congruency effects, observed in conflict tasks such as Stroop and flanker tasks, have been investigated for nearly a century in psychology and neuroscience, but their mechanistic basis is not fully understood. We introduce a verbal-only LLM conflict task in which a prompt stem elicits a default same-color completion and an explicit rule either agrees with (congruent condition) or conflicts with (incongruent condition) the completion. Gemma-2-2B and six Pythia models ranging from 410M to 12B parameters showed strong default same-color tendencies, and six of seven models showed strong congruency effects. Using causal attribution analysis, attention analysis, and attention ablations, we identified distinct processing pathways in these LLMs: a pathway involving short-range attention to a superficial color cue that is preferentially activated in the congruent condition, and a pathway involving long-range attention to the rule prefix that is preferentially activated in the incongruent condition. Fine-tuning that strengthened the default same-color tendency had divergent effects on task conditions, reducing incongruent performance while increasing congruent performance. In contrast, increasing rule set size selectively impaired incongruent performance. These converging findings support an account in which congruency effects in this task arise from competition between an in-weight default mapping and an in-context rule-based mapping. More broadly, our findings illustrate how LLMs can serve as model systems for mechanistic analysis of competition between default and rule-governed response tendencies within a single learned network. 

---
# Strengthening Full Justified Representation: Efficient Verification and Computation 

**Authors**: Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2608.11500)  

**Abstract**: Full justified representation (FJR) is among the strongest known satisfiable proportionality axioms for approval-based committee elections. Recent work has shown that an FJR committee can be found in polynomial time, but verifying whether a given committee satisfies FJR remains coNP-complete. We introduce FJR+, a strict strengthening of FJR and EJR+ that can be verified and satisfied in polynomial time. We then analyze the Residual-Budget Greedy (RBG) algorithm and prove that it selects a partial committee such that every size-$k$ completion satisfies FJR+. This freedom allows us to use sequential Phragmén to obtain a priceable completion. The resulting rule always satisfies FJR+ and the sub-core, and it is priceable whenever at least $k$ candidates receive an approval. We also obtain a Droop-quota version of FJR+. Finally, we extend FJR+ to approval-based participatory budgeting with arbitrary project costs. A project-specific version of RBG computes this property in polynomial time and can be continued to a priceable outcome satisfying a cost-based version of the sub-core. 

---
# HyperFix: Combinatorial Nonlinear Correction for Task Vector Merging 

**Authors**: Hyo Seo Kim, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11499)  

**Abstract**: Task vectors enable model merging without joint retraining. In practice, the subset of task vectors to be merged may vary, but many existing methods use scalar tuning for a particular subset, requiring repeated tuning across subsets and restricting task vector merging to linear rescaling. We therefore formulate merging across varying task subsets as a combinatorial correction problem and introduce HyperFix, a lightweight hypernetwork that predicts subset-conditioned nonlinear corrections in weight space. Trained once on singleton, pair, and triple subsets from a task bank, HyperFix generalizes to larger subsets without per-subset optimization. Our local perturbation analysis bounds the residual correction beyond linear merging and motivates learning it from small task updates. Experiments across diverse benchmarks show that HyperFix outperforms existing task vector merging methods while reducing tuning cost. 

---
# The Next Challenge for Agentic Cybersecurity: A Realistic, Contamination-Free Reverse Engineering Benchmark 

**Authors**: Jeremy Spence, Nicholas Assaderaghi, Jinhao Zhu, Nikil Ravi, Raluca Ada Popa, Guannan Wei, Yangruibo Ding, Zhuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11469)  

**Abstract**: AI agents are rapidly improving in cybersecurity capabilities when the source code is available for analysis, yet much of the software most consequential to cybersecurity, including malware, firmware, and proprietary applications, is available only as binaries. Analyzing such software requires reverse engineering(RE): recovering program semantics before the analysis can be meaningfully performed. However, evaluating agentic RE poses a fundamental challenge: benchmark instances must be unseen as source code in the LLMs' training data to prevent models from taking shortcuts by recognizing them rather than really analyzing them, while also matching the scale and anti-analysis protections of real software. Unfortunately, however, existing benchmarks do not jointly satisfy these requirements. To this end, we introduce SRE-Bench, the first realistic, contamination-free RE benchmark. Built entirely from scratch by RE experts with over 5,000 hours, SRE-Bench comprises 19 private, real-world-scale programs averaging 16.9K lines of code. We further developed 44 in-house anti-analysis primitives, yielding 262 binary instances and 1572 deterministically graded tasks. Our evaluation across five frontier LLMs (GPT-5.6-sol,Claude-Opus-5,GPT-5.5,Grok-4.5, and GLM-5.2) shows that RE remains largely unsolved: the strongest model, GPT-5.6-sol, scores 61.4% per instance, and fully solves only 31.5% of the instances. Our analysis further reveals that agents behave differently from human engineers, where agents are relatively insensitive to compiler optimization and static linking. Controlled ablations also confirm that both contamination control and realistic scale are essential. These results indicate that strong source-code security capabilities do not yet transfer to binary analysis, highlighting RE as an important frontier for agentic cybersecurity and SRE-Bench as a rigorous testbed to measure progress. 

---
# PAC-Bayes Beyond Parameter Space: Behavioral Equivalence, Z-Information, and Exact Complexity Decomposition 

**Authors**: Vasant G. Honavar, Satish Kumar Keshri, Neil Ashtekar, Zehao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11465)  

**Abstract**: PAC-Bayes theory provides generalization guarantees by controlling the Kullback--Leibler (KL) divergence between posterior and prior distributions over a chosen hypothesis representation. However, predictive risk depends only on the predictive behavior induced by a hypothesis, not on the particular internal realization that implements that behavior. In over-parameterized systems, many distinct configurations induce identical predictive behavior, yet the classical PAC-Bayes KL divergence does not distinguish uncertainty over predictive behavior from variation among behaviorally equivalent realizations.
We show that this distinction induces an exact structural decomposition of classical PAC-Bayes complexity. We formalize behavioral equivalence through a measurable behavior map and use measure disintegration to decompose probability measures on the configuration space into a distribution over predictive behaviors and conditional distributions over behavioral fibers. This yields an exact decomposition of the classical PAC-Bayes KL divergence into a behavior-selection term and a realization-level term given by an expected conditional KL within fibers.
We define Z-information as the negative of this realization-level contribution: the exact gap between the KL divergence and the complexity of uncertainty over predictive behavior alone. We further show that the behavior-selection term admits an exact variational characterization: it is the minimum KL divergence among all posteriors inducing the same distribution over predictive behaviors, attained by a canonical fiber-symmetrized representative.
Finally, we show that symmetry, behavior-preserving directions, fiber geometry, and invariance under fiber-preserving perturbations arise naturally from the same behavior-map structure. Together, these results identify predictive behavior as the natural object of PAC-Bayes complexity. 

---
# TangPoetryBench: A Multi-Dimensional Benchmark and Rubric-Conditioned Evaluator for Poetry-to-Image Generation 

**Authors**: Haoqi Hu, Tongji Luo, Li Zhang, Boning Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.11452)  

**Abstract**: Text-to-image (T2I) models are increasingly asked to illustrate literary and cultural content, yet we cannot measure how well an image renders the meaning of a poem. The task is many-sided: a good illustration must be visually sound, faithful to the poem's imagery and scene, culturally and stylistically apt, free of spurious text, and true to its emotion, and its deepest requirements, imagery and especially implicit emotion, are never stated in the words. Existing metrics (CLIPScore, BLIPScore, VQAScore) reward literal text-image correspondence and so cannot tell whether an illustration succeeds, let alone why, or even separate the best model from the worst. We introduce TangPoetryBench, a multi-dimensional benchmark of 1,280 images (320 classical Chinese Tang poems x 4 state-of-the-art T2I models) with quality-controlled human annotations across ten dimensions. Analyzing this data, we reveal the shared and model-specific strengths and weaknesses of current T2I models, including their ability to evoke a poem's implicit emotion. We further introduce PoemAutoEvaluator (PAE), an open, rubric-conditioned evaluator that reaches parity with a strong proprietary judge (Claude), generalizes to an unseen generator and a second poetic tradition (Song Ci), and lets the benchmark scale to new images without fresh human annotation. We release the benchmark, annotations, and evaluator. 

---
# Herding End-to-End Autonomous Driving via Neuro-Symbolic Safety Guards 

**Authors**: Simón Patiño Idarraga, Erick Silva, Rehana Yasmin, Ali Shoker  

**Link**: [PDF](https://arxiv.org/pdf/2608.11451)  

**Abstract**: Modern end-to-end driving agents can achieve high average performance yet still violate basic traffic rules that a human driver would never miss. The reason is structural: they learn statistical patterns rather than the physical conditions that guarantee safe driving, leaving their decision-making process opaque and safety constraints unenforced. We introduce a neuro-symbolic safety guard, a lightweight module that attaches to the final command interface of an already-trained agent. Immediately before a command reaches the vehicle, it checks the command against explicit safety rules and, only when necessary, replaces it with the nearest safe alternative. Each intervention is directly executable and traceable to the rule that triggered it, while the guard itself requires no retraining and adds no learned component. Evaluated on the long-tail benchmarks Fail2Drive and Bench2Drive using the state-of-the-art TransFuser v6 (TFv6) as a case study, the guard improves Success Rate by 15% and reduces safety-critical collisions by up to 53%, while preserving the original Driving Score. 

---
# TRACES: A Benchmark for Epistemic Reliability in Scientific Reasoning by LLMs 

**Authors**: Valentin Rodionov, Shamil Assylbekov  

**Link**: [PDF](https://arxiv.org/pdf/2608.11415)  

**Abstract**: Large language models are being proposed as agents in scientific workflows, in domains where no downstream verifier exists. Such deployment assumes the model can distinguish reliable scientific literature from unreliable literature, a capability that has not yet been directly measured. Existing benchmarks evaluate factuality on questions with known answers; the failure mode we target here is different. We introduce a probe corpus of 42 retracted, fraudulent, and pseudoscientific papers, paired with a methodology for eliciting and scoring single-shot model engagement with each paper's framing. Each probe pairs a preamble extracted near-verbatim from the target paper with a scientifically plausible study-design request. The probes span five claim types: fabricated observation, pseudophysical mechanism, magical premise, legitimization bridge, and cargo-cult experiment. Two complementary scores measure whether a model rejects the flawed premise outright (IFR-a) and whether it recognizes the unreliability while still engaging (IFR-i). A depth score, the Engagement Depth Index (EDI), quantifies reproduction of paper- or field-specific withheld details. Across 30 models and 10 repeated runs, aggregate IFR-a is 0.93 $\pm$ 0.004 and aggregate IFR-i is 0.809 $\pm$ 0.009. Models engaged with untenable premises in 95% of all non-empty responses. Every evaluated model fails more than 71% of agentic probes, and 22 of 30 models fail more than 90% of the time. Rejections are concentrated on a small number of high-notoriety topics and specific probes, and disappear under matched-structure controls. These results are consistent with topic-keyed safety behavior rather than robust epistemic competence, and indicate an urgent need for guardrail infrastructure for scientific deployment of language models. 

---
# AI Guardrail Survival under Single-Cycle Agentic Self-Summarization 

**Authors**: Ted Kwartler, Alan Aqrawi, Arian Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11392)  

**Abstract**: Long-running agents periodically compact their context, replacing the transcript with a model-generated this http URL work shows that dropping a standing safety constraint during compaction drives behavioral violations acrossmany models (Governance Decay; Chen, 2026). We ask a finer question: under a single compaction cycle, how is a safetyrule lost, and what does that imply for detection and evaluation? Our central finding is that a presence check is not asafety check: when compaction does not drop a rule outright, it often leaves something that looks like a rule but doesnot act like one. On behavioral replay, a degraded residue leads the model to perform the prohibited action far more oftenthan an intact welded rule does (all-case gaps of +34 and +57 points under two replay models, both positive), category-level survival behaves like a residue, and even intact rules sometimes fail to fire, so an audit that checks only textualpresence gives false assurance. Sharpening this, rule-form items are retained substantially more often than prominence-matched facts, which is exactly why presence-based checking feels adequate even though survival is not this http URL loss is regime-dependent (weld-or-drop with a single rule; degraded predicate-loss residues under a tighterbudget), and we did not observe the hypothesized textual severing mode. Such loss is silent at runtime and detectableonly by comparison with retained external ground truth (such as a constraint registry), which reveals textual absence butnot whether a surviving rule still fires. We also document evaluation pitfalls where LLM-judge labels alone would havereversed a conclusion. All results concern a single compaction cycle. 

---
# Gaze Target Estimation Anywhere with Concepts 

**Authors**: Xu Cao, Houze Yang, Vipin Gunda, Zhongyi Zhou, Tianyu Xu, Adarsh Kowdle, Inki Kim, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2608.11367)  

**Abstract**: Estimating human gaze targets from images in-the-wild is an important and formidable task. Existing approaches primarily employ brittle, multi-stage pipelines that require explicit inputs, like head bounding boxes and human pose, in order to identify the subject of gaze analysis. As a result, detection errors can cascade and lead to failure. Moreover, these prior works lack the flexibility of specifying the gaze analysis task via natural language prompting, an approach which has been shown to have significant benefits in convenience and scalability for other image analysis tasks. To overcome these limitations, we introduce the Promptable Gaze Target Estimation (PGE) task, a new end-to-end, concept-driven paradigm for gaze analysis. PGE conditions gaze prediction on flexible user text or visual prompts (e.g., "the boy in the red shirt" or "person in point [0.52, 0.48]") to identify a specific subject for gaze analysis. This approach integrates subject localization with gaze estimation, and eliminates the rigid dependency on intermediate analysis stages. We develop a scalable data engine to generate Gaze-Co (Gaze Estimation with Concepts), a dataset and benchmark of 120K high-quality, prompt-annotated image pairs. We also propose GazeAnywhere, the first model designed for PGE. GazeAnywhere uses a transformer-based detector to fuse features from frozen encoders and simultaneously solves subject localization, in/out-of-frame presence, and gaze target heatmap estimation. GazeAnywhere achieves state-of-the-art performance on multiple PGE benchmarks, setting a strong baseline for this new problem even on a difficult out-of-domain, real-world clinical dataset. GazeAnywhere is open-sourced in this http URL. 

---
# Dynamics Models for Offline Hyperparameter Selection in Real-World RL 

**Authors**: Jordan Coblin, Han Wang, Martha White, Adam White  

**Link**: [PDF](https://arxiv.org/pdf/2608.11349)  

**Abstract**: A key obstacle to deploying reinforcement learning in real-world systems is hyperparameter selection, particularly when simulators are unavailable and online experimentation is costly. Prior work has proposed calibration models trained on offline data to approximate environment dynamics and enable offline hyperparameter selection, but these methods have so far been evaluated only in simple simulated settings. In this paper, we present the first application of calibration models in a real-world industrial setting: a municipal water treatment plant. We evaluate several calibration model approaches, including a k-nearest neighbors model with a Laplacian distance metric, on high-dimensional, non-stationary sensor data for nexting prediction tasks. Our results show that these models can generate realistic long-horizon rollouts and recover meaningful hyperparameter sensitivity trends. We further examine how calibration models scale to year-long datasets, how they support the selection of fine-tuning learning rates for pre-trained agents, and how robust they are under distribution shift. Overall, our findings provide a proof of concept for using offline dynamics models to support RL deployment in real-world environments, while highlighting important practical challenges for future work. 

---
# Governing Agentic AI in FinTech 

**Authors**: Henry Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.11344)  

**Abstract**: Financial institutions are delegating consequential decisions to agentic AI systems that decompose goals, coordinate models and tools, and act with little oversight. Yet agentic AI governance in FinTech is under-investigated. We argue the binding governance constraint is not capability but verifiability. We define the Verifiability Gap as the shortfall between the verification delegated authority demands and the explainability and reproducibility retained after a decision. It is indexed to a verifier, evidentiary standard, and audit lag. We develop a multilevel governance theory for agentic AI and test its mechanisms in three studies over nine model versions, from a three-billion-parameter local model to a commercial frontier system. Study 1 shows that provider releases alter historical financial actions, and that the controls replay needs belong to the provider: the frontier model rejects temperature, top_p and top_k outright and exposes no random seed. Under the tightest controls each endpoint allows, a local model reproduced 320 of 320 executions, hosted models 319 of 320 and 959 of 960. Study 2 shows that orchestration is a latent policy layer. Architecture changes final actions, and no execution record repeated in any configuration at any scale. The frontier model reproduces its own actions more often than the local ones, its record no better, and loses a comparable share of its differentiation. Capability buys a higher starting point, not auditability. Study 3 shows two deterministic credit-model versions each reproduce their current action perfectly, yet the current cannot recover a historical one. We conceptualize reproducibility as a governance profile, not a scalar, yielding evidence-contingent delegation: authority is defensible only while retained evidence substantiates its exercise. Beyond finance, the framework extends to other high-stakes domains requiring auditability. 

---
# Self-evolving network verifiers 

**Authors**: Ioannis Protogeros, Tibor Schneider, Laurent Vanbever  

**Link**: [PDF](https://arxiv.org/pdf/2608.11340)  

**Abstract**: Symbolic network verifiers can reason about correctness across vast spaces of routing inputs and failures, but only for the protocols and features an expert has encoded by hand. Creating and maintaining a faithful model of the control plane is both difficult and never-ending, since no written source specifies perfectly what a network does: vendor implementations deviate from the RFCs, and behaviour shifts with releases. The burden of constant upkeep ultimately keeps verification out of many networks that need it.
We argue that the model should instead evolve automatically to faithfully capture the actual network behaviour. To achieve that, we leverage the only source that specifies it unambiguously: the router software itself. In a counterexample-guided loop, a coding agent proposes extensions to the verifier's symbolic encoding, while a trusted oracle (e.g., emulated routers) supplies the ground-truth routing state. The agent iteratively refines the network model using each disagreement with the oracle.
As early evidence, a prototype of this system taught a 3,000-line SMT-based verifier three features it did not support: OSPF areas, BGP route reflection, and L3VPN over EVPN, converging autonomously on models that match the oracle, even noticing vendor-specific behaviour. Automating model growth shifts the hard problem from writing verification systems to systematically testing them; we propose a research agenda for trusting and harnessing automatically evolved verifiers. 

---
# Dual-Domain Cross-Modal Decoding for Clinical Text-Guided Medical Image Segmentation 

**Authors**: Md Maklachur Rahman, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2608.11335)  

**Abstract**: Clinical text can narrow down what to segment, but recent text-guided designs emphasize spatial alignment while overlooking frequency content that governs texture and boundaries. We propose Dual-Domain Cross-Modal Decoding (DD-CMD) for clinical text-guided pulmonary infection segmentation, integrating two complementary forms of language guidance during decoding. In the spatial domain, Text-Guided Spatial Cross-Attention (TGSA) aligns multi-scale visual tokens with text semantics and updates features through gated residual fusion. In the frequency domain, Spectral-Text Adaptive Modulation (STAM) applies a 2D DCT to compute learnable band-energy statistics and predicts text-conditioned FiLM parameters to recalibrate decoder channels for frequency-aware decoding. DD-CMD embeds TGSA and STAM into a coarse-to-fine decoder (7x7 to 56x56) and restores full-resolution masks using a lightweight two-stage refinement module. Experiments on QaTa-COV19 and MosMedData+ show that DD-CMD achieves 91.46% Dice / 84.26% mIoU and 81.95% Dice / 69.42% mIoU, respectively, with average gains of +1.96 Dice and +2.67 mIoU over the strongest prior baselines. Code: this https URL. 

---
# Contextual Quality-Diversity Evolutionary Reinforcement Learning for HVAC Control in Tropical Commercial Buildings 

**Authors**: Tran Le Vu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11324)  

**Abstract**: This paper proposes a contextual quality-diversity evolutionary reinforcement-learning controller, CQD-ERL, for the supervisory control of a tropical, water-cooled chiller plant and its associated air side. Rather than converging to a single scalarised policy, the controller maintains a product archive of specialised policies indexed jointly by a data- driven operating context, a cluster of daily weather and load regime, and a context-invariant behaviour descriptor, filled by a gradient-free evolutionary operator and a soft-actor-critic policy-gradient operator that share one replay buffer. Every action is filtered through a deterministic safety shield before execution. The controller is trained on a two-tier reduced-order environment representing the latent load, cooling-tower approach and humidity constraints of a Singapore commercial building, and is evaluated over a full annual backtest against an ASHRAE Guideline 36 baseline. 

---
# Socioduality: A Relational Process Framework for Human-AI Interaction 

**Authors**: Mehmed Zahid Çögenli  

**Link**: [PDF](https://arxiv.org/pdf/2608.11322)  

**Abstract**: Human-AI research often evaluates individual capabilities, combined performance, or final outputs, but these approaches do not preserve how one party's response becomes part of the conditions under which the other party's next contribution is formed. This article introduces socioduality, a sequential, reciprocal, and history-carrying relational process between two distinguishable parties in which a response from one party becomes part of the observable conditions under which the other party's subsequent contribution, judgement, decision, or action is formed. Specified for human-AI dyads, the construct uses nested units: moves, confirmed sociodual episodes, linked pathways, and the broader interaction container. A minimum episode A1-B1-A2 requires evidence of response contingency and return contingency; candidate episodes are classified as confirmed, non-sociodual, or indeterminate before secondary coding of response orientation and substantive contribution re-formation. Three propositions address history-conditioned formation, pathway divergence, and robustness differences among endpoint-equivalent pathways. A frozen operational protocol was calibrated on three previously unseen natural human-AI records through two separately executed model-based evaluator series. Move and candidate reconstruction converged exactly in two cases and differed by one local multimodal unitisation decision in the third; remaining disagreement was concentrated at return-contingency boundaries. Socioduality therefore provides a bounded and empirically tractable process construct for analysing how human and AI contributions are formed through interaction while preserving pathway information that endpoint-centred analysis cannot recover. 

---
# Terminal Symmetry as a Decision Resource: Statewise Refinement for Anytime Verified Construction 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11318)  

**Abstract**: Many sequential construction tasks exhibit exact symmetry at completion while their execution remains directed and history-dependent. We develop a decision-resource view of terminal symmetry: process evidence supplies directionality, terminal correspondence transports that structure across equivalent outcomes, realized-state evidence refines its current decision relevance after transitions, and a fixed verifier certifies execution. This decomposition yields transport--refine--certify. \method{} instantiates the principle with an episode-fixed transported process structure, its state-restricted process rank, a state-dependent residual rank refreshed after accepted transitions, and an ordinal rank meet whose top-$k$ set is exactly the union of the two proposal prefixes. The meet provides a completion guarantee under prefix coverage and attains the tight worst-case verifier-query bound under the corresponding prefix information model; a two-state construction predicts a strict post-transition dynamic--static separation. Across CAD assembly, Mini-Programs, and exact-fill packing, statewise refresh improves anytime AUC by up to $6.77$, $21.75$, and $8.68$ points, respectively. On 1,135 target-removal episodes from the official GRN OOD scenes, \method{} attains the lowest mean capped verifier cost at all three scales among the compared GRN and CDGS-style planners. The statewise signal also transfers across aggregation and scheduler organizations. Terminal symmetry thereby becomes a reusable decision resource for directed construction. 

---
# Clinical Feasibility of Low-Magnification Fluorescence Imaging for Breast Cancer Margin Detection Using Texture Analysis and Deep Learning 

**Authors**: Pouya Afshin, Tianling Niu, Tongtong Lu, David Helminiak, Julie Jorns, Mollie Patton, Tina Yen, Donghye Ye, Bing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.11317)  

**Abstract**: High-resolution images of unprocessed surgical breast tissue can be obtained using microscopy with ultraviolet surface excitation (MUSE). This technique is considered a promising method for checking surgical margins during breast cancer surgery. In this study, MUSE images at 4x and 10x magnifications were compared using patch-level classification methods. Texture analysis (TA) based on local binary patterns (LBP) and deep learning (DL) with a base Vision Transformer (ViT) model were used. Both methods achieved similar performance at both magnifications. Using DL method, both 4x and 10x magnifications achieved 96.30% sensitivity, 100% specificity and 98.18% accuracy. Using TA method, 4x achieved better specificity (100% vs 93.33%) and 10x yielded higher sensitivity (100% vs 93.33%), but both had the same accuracy (96.67%). No clear improvement in performance was observed with 10x magnification. These results show that 4x imaging achieves the same diagnostic accuracy as 10x imaging. At the same time, 4x offers a larger field of view and faster image capture. Therefore, lower magnification can be effectively used in MUSE systems for accurate and efficient intraoperative margin assessment. 

---
# Backdoor Decontamination Dynamics in LLM Agents 

**Authors**: Gabriel Huang, Abhay Puri, Léo Boisvert, Alexandre Drouin, Perouz Taslakian, Spandana Gella, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2608.11295)  

**Abstract**: Open-weight LLM agents are vulnerable to backdoors installed during fine-tuning, which may be undetectable if the trigger conditions are never met during testing. Assuming defenders do not know the existing trigger, they cannot unlearn it directly. One decontamination strategy is to install a known backdoor (defensive poisoning) then to unlearn it, hoping that the original unknown backdoor is removed as a side effect. However, this procedure has uncertain outcomes: the original backdoor may persist or be erased or rerouted, among other possibilities. We introduce a framework for studying these dynamics in tool-calling agents, decoupling trigger, response, teacher, and fine-tuning method across systematic experiments on AgentDyn. Across 115 experiments, defensive poisoning alone erases around 56% of original backdoors; subsequent decontamination then drives almost all survivors to erasure, confirming that trigger recognition and malicious execution are behaviorally dissociable. Interestingly, our experiments find that malicious backdoors never persist when using different triggers of the same general type as the defensive backdoor when followed by decontamination via unlearning. Co-installing up to four backdoors increases resistance (around 36% erased), yet decontaminating a single known co-resident backdoor collaterally clears 52/60 co-residents (87%). Upon visualizing postdecontamination model internals using J-lens, we confirm that although the decontamination restores benign LLM responses, traces of original trigger awareness persist at intermediate layers. 

---
# CLEAR: Class-wise Expert Aggregation with Structured Sampling for Long-Tailed Classification 

**Authors**: Gawon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11287)  

**Abstract**: Long-tailed classification poses a reliability challenge because models trained on imbalanced data are unevenly reliable across frequent and underrepresented classes. While existing methods address imbalance through re-balancing, adjustment, representation learning, or multi-expert modeling, they rarely estimate which expert should be trusted for each class. This paper proposes CLEAR (Class-wise reLiability-aware Expert Aggregation for long-tailed Recognition), a modular ensemble framework for long-tailed classification. CLEAR generates diverse experts through threshold-based structured sampling while preserving the full label space, then estimates a class-wise trust score for each expert using a smoothed class-wise precision formulation. During inference, expert predictions are combined through class-wise generalized product-of-experts aggregation, allowing different experts to be emphasized for different classes. Experiments on CIFAR-100-LT, ImageNet-LT, and Places-LT across multiple backbones show that CLEAR achieves competitive overall accuracy and particularly strong few-shot performance. These results support class-wise expert reliability as a useful design principle for long-tailed ensemble learning. 

---
# SegPAR: Class-Centric Decision-Based Sparse Attack for Semantic Segmentation 

**Authors**: Dongsu Song, DaeYun GO, Boseung Seo, Jay Hoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.11285)  

**Abstract**: Despite the practical relevance of sparse decision-based black-box threats, they have received limited attention in semantic segmentation. To bridge this gap, we adapt the most representative decision-based black-box sparse attacks from the classification domain to serve as baselines, establishing a rigorous benchmark for this underexplored setting. In this context, we demonstrate that one of the existing methods suffers from severe query inefficiency due to its image-centric pixel accumulation, which rapidly exhausts query budgets across the vast image space. To overcome this, we propose SegPAR, a novel decision-based framework that shifts to a class-centric exploration paradigm. Furthermore, to eliminate the misleading feedback generated by standard decision rewards during pixel accumulation, we introduce a novel discrepancy reward. Extensive experiments show that SegPAR significantly outperforms black-box baselines in sparsity efficiency and MIoU reduction, while remaining competitive with white-box sparse attacks. Code is available at \href{this https URL}{this https URL}. 

---
# Chemically Meaningful Textualization Enables Explainable Validation of Metal-Organic Frameworks by Large Language Models 

**Authors**: Guobin Zhao, Xiao-Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.11283)  

**Abstract**: Computation-ready metal-organic framework (MOF) databases are essential for high-throughput screening, yet many reported crystal structures remain chemically unreasonable or disordered, compromising simulation fidelity. Existing validation approaches can identify non-computation-ready structures, but they often rely on heuristic rules, license requirement, or offer limited interpretability. Here, we show that large language models (LLMs) can serve as interpretable validators of MOF structures when crystallographic information is transformed into chemically meaningful text. By benchmarking nine descriptors, we find that successful LLM-based validation depends not on the amount of structural information alone, but on whether local coordination, framework connectivity, and chemical context are organized into a linguistically learnable representation. Fine-tuned LLMs using specialized descriptors (mof2text) achieve performance comparable to graph-based models in identifying unreasonable MOFs. Importantly, these models extend beyond black-box classification by generating diagnostic rationales for likely error sources, including abnormal bonding, connectivity, and charge states, as well as error-category predictions for annotated datasets. This work establishes chemically informed textualization as the key step that transforms LLMs from generic text models into practical and explainable tools for curating MOF databases. 

---
# Physics-Informed Implicit Neural Representations for Improved Myocardial Perfusion MRI Quantification 

**Authors**: Christos Tsepas, Chang Yan, Maximilian Fuetterer, Sebastian Kozerke, Cian M Scannell  

**Link**: [PDF](https://arxiv.org/pdf/2608.11282)  

**Abstract**: Quantifying myocardial perfusion from cardiac magnetic resonance (CMR) can be achieved by fitting tracer-kinetic models to the dynamic contrast-enhanced MR data. However, fitting the observed data with multi-compartment exchange models, which describe the evolution of the contrast agent in the tissue, to estimate perfusion parameters is a challenging inverse problem that is sensitive to noise and acquisition variability. Previously, physics-informed neural networks (PINNs) have been proposed as an alternative to conventional non-linear least squares fitting methods with promising results for quantitative perfusion CMR. In this work, we extend the previously proposed PINN framework with spatiotemporal implicit neural representations (INRs) to represent the MR signal as a continuous spatiotemporal function and to improve the accuracy, smoothness, and physical consistency of the PINN model. In realistic simulated CMR datasets, our proposed PINN with INRs demonstrates improved robustness and parameter estimation accuracy over the previously established methods. The code is available at this https URL. 

---
# Federated Learning for Distributed CNC Tool Wear Prediction 

**Authors**: Afsana Khan, Morris Stallmann, Marcin Pietrasik, Charis Kouzinopoulos, Anna Wilbik  

**Link**: [PDF](https://arxiv.org/pdf/2608.11281)  

**Abstract**: Tool wear prediction is an important task in CNC machining, where accurate monitoring of tool condition supports product quality and process reliability. Machine learning methods have shown potential for this task, but their use in industrial environments is limited by the distributed nature of machining data and by restrictions on data sharing between machines, sites, or organizations. Federated learning offers a suitable framework for this setting by enabling collaborative model training without transferring raw operational data. This paper investigates federated learning for CNC tool wear prediction. Tool trajectories are distributed across simulated clients to represent a federated learning scenario. The federated models are compared against centralized references and local client baselines. Results show that federated learning achieves performance close to centralized learning and improves significantly over local client models. These findings indicate that federated learning can support collaborative tool wear prediction in distributed CNC manufacturing environments. 

---
# Uncertainty-Aware and Explainable Ensemble Deep Learning Framework for Multi-Class Skin Lesion Classification 

**Authors**: Rofiqul Islam, Lilatul Ferdouse  

**Link**: [PDF](https://arxiv.org/pdf/2608.11280)  

**Abstract**: Skin cancer diagnosis from dermoscopic images remains challenging due to high intra-class variability, inter-class similarity, class imbalance, and the limited interpretability of deep learning models. This paper proposes an uncertainty-aware and explainable deep learning framework for multi-class skin lesion classification. The framework combines a vision transformer model (MaxViT-Tiny) with CNN-based models (ConvNeXt-Tiny and EfficientNetV2-B0) through deep ensemble learning. Monte Carlo (MC) Dropout estimates predictive uncertainty and identifies unreliable predictions, while Grad-CAM++, an explainable AI (XAI) technique, provides visual explanations by highlighting lesion regions that influence model decisions. Evaluated on the HAM10000 dataset, the framework achieves 96% accuracy and 99% ROC-AUC under uncertainty-aware filtering (entropy < 1.0, confidence >= 0.7), with macro-average precision, recall, and F1-score of 94%, 95%, and 95%, respectively, and 96% weighted-average scores across all three metrics. The results demonstrate accurate, interpretable, and uncertainty-aware skin lesion classification for trustworthy computer-aided diagnosis. 

---
# Every pooling rule has its world: matching probability combination rules to situations and stakes 

**Authors**: Tanel Tammet, Priit Järv, Dirk Draheim  

**Link**: [PDF](https://arxiv.org/pdf/2608.11275)  

**Abstract**: Systems often need to combine two numerical assessments of the same yes/no question. The appropriate formula depends on what the numbers represent and on how the sources are related. Averaging is correct when one of several alternative interpretations applies; multiplying odds is correct when probability reports are based on conditionally independent evidence and a common prior; and probabilities of alternative successful derivations require their dependence or shared evidence to be taken into account.
We state the assumptions behind several common combination rules and derive the corresponding combined probabilities. Two groups of Monte Carlo experiments address different questions. First, controlled generating mechanisms verify that the derived rule recovers the correct probability in the situations for which its assumptions hold. Second, the same mechanisms measure the consequences of using a mismatched rule, using logarithmic score and threshold decisions with different costs. Distinct pooling rules can produce the same binary decision at threshold 1/2 while assigning substantially different probabilities, so binary accuracy alone can conceal important differences. We also give probabilistic interpretations of conflicting-evidence rules and show that, for overlapping derivations, retaining the identities of shared uncertain premises permits direct calculation of the probability that at least one derivation is available. Pairwise combination of proof probabilities loses information when there are three or more derivations. 

---
# Agent Safety Should Be a Runtime Contract 

**Authors**: Albus W. Ng, Yi Han, Jusheng Zhang, Wenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11274)  

**Abstract**: The dominant paradigm treats AI safety as a property to be instilled during model training via RLHF, DPO, or Constitutional AI. We argue this is structurally insufficient for autonomous agents that execute code, mutate files, send messages, and modify databases. Agent safety should be a runtime contract enforced by the harness, and the contract has two complementary faces. The preventive face blocks dangerous actions before they happen via sandboxes, permission gates, output filters, and trajectory monitors. The evidential face requires verifiable proof that good actions actually happened, gating task submission on hard evidence such as test runs, log captures, file diffs, and citation grounding. We ground the position in four lines of public evidence, with row-level protocols and data released in the supplementary JSON files: a survey of 52 documented AI-agent and LLM safety incidents, a false-completion audit with 31 non-contested core cases plus one disputed illustrative case, a trajectory-schema audit of 12 public agent systems and harnesses, and a title-level audit of all 28,560 papers accepted at NeurIPS, ICML, and ICLR 2023-2025 showing a pooled 8-12x imbalance between training-time and deployment-time publication. Two prior communities that needed to enforce safety, computer security and the experimental sciences, converged on runtime contracts with both preventive and evidential elements; agentic AI is now under the same pressure. We formalize an Agent Trajectory Schema and Evidence Chain, state a compositional gating proposition based on standard monitor composition, and outline a research agenda. The right unit of safety in agentic AI is the trajectory-with-checkable-evidence, not the model. 

---
# Methodologies for Improving the Quality of AI Tutoring in K-12 Education 

**Authors**: Tushar Udeshi, Anna Khazenzon, Kabir Khan, Nick Breen, RJ Corwin, Chris DiGiano, Kodi Weatherholtz, Marek Zaluski  

**Link**: [PDF](https://arxiv.org/pdf/2608.11259)  

**Abstract**: Many AI tutors leverage large language models (LLMs) today. Given that LLMs are opaque black boxes, robust evaluation and live experimentation to measure the impact of every change are essential. We pioneered AI-powered tutoring for K-12 with the launch of Khanmigo (Khan Academy, 2023). We describe the metrics we use to measure AI tutoring quality and student engagement as well as various experiments we have run. We highlight the changes that have moved our metrics, including models, prompting, personalization and agents. 

---
# Variable Selection in the Context of AI Fairness 

**Authors**: Ivan Luciano Danesi, Chiara Frigerio, Fabio Maccaferri, Giorgio Alessandro Motta, Pietro Zecca  

**Link**: [PDF](https://arxiv.org/pdf/2608.11251)  

**Abstract**: Fairness in AI systems has become more important with recent regulatory demands, such as the EU AI Act. Traditional approaches often do not take into account philosophical ethics and social awareness. Variable selection processes, in particular, can introduce implicit bias, affecting equity across different subgroups. We discuss a mathematical approach that evaluates fairness in AI, aligning mathematical methodologies with ethical considerations and regulatory requirements. Our aim is to advocate for interdisciplinary collaboration to address fairness, emphasizing the importance of understanding broader ethical and societal contexts. Our approach emphasizes maintaining all potentially relevant variables to allow for more granular fairness assessments and to reduce implicit bias. The findings suggest that the exclusion of sensitive or critical variables may compromise equity between subgroups. In contrast, retaining all relevant variables could reduce implicit bias. Thus, the interdisciplinary approach could provide deeper insight into the ethical implications and compliance with regulatory standards. By integrating a mathematical approach with ethical and social awareness, we suggest more equitable outcomes and responsible AI deployment. This work underscores the necessity of interdisciplinary collaboration in effectively addressing fairness in AI systems aligned with the objectives of the European Union's AI Act, which seeks to promote trustworthy and fair AI systems. 

---
# Diffuse to Compress: Leveraging Diffusion LMs for Lossless Compression 

**Authors**: Angelo Nardone, Paolo Ferragina  

**Link**: [PDF](https://arxiv.org/pdf/2608.11249)  

**Abstract**: We study the problem of lossless text compression, motivated by the rapid growth in the collection and storage of digital textual data - including plain text, source code, and structured formats such as XML - and by recent advances in neural language model-based compression. In particular, recent LLM-based approaches, whether built on symbol-ranking pipelines or paired with a statistical compressor, have demonstrated compression ratios significantly superior to general-purpose compressors such as zstd, gzip, or bzip on text and code. However, these neural approaches suffer from severe throughput limitations, making them not yet practically usable.
For the first time in the context of lossless neural text compression, we introduce Diffusion Language Models (DLMs) as an alternative inference paradigm to autoregressive LLM-based approaches. We argue that replacing autoregressive LLMs with DLMs within the same compression framework could overcome the throughput bottleneck caused by their one-symbol-per-step limitation. However, achieving these improvements requires addressing algorithmic challenges introduced by applying DLMs to lossless compression, where the architecture allows the number and positions of symbols encoded at each forward pass to be decided independently. We design efficient and effective strategies to solve these challenges and evaluate them experimentally against LLM-based and general-purpose compressors on enwik8, a well-established textual benchmark.
Our results show that the newly proposed DLM-based framework advances the state of the art in lossless text compression. Moreover, as DLMs are still a relatively young paradigm, recent advances toward increasingly capable and efficient models suggest substantial room for further improvements. 

---
# Lost in Compaction: Evaluating Side-Constraint Loss under Context Compaction 

**Authors**: Zhiqi Wang, Yichi Zhang, Dongwon Lee, Yuchen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.11242)  

**Abstract**: When the context window is under pressure, LLM systems compact prior context to continue ongoing tasks. We identify a class of user-issued instructions, Session Constraints (SCs), such as "do not delete any emails until I confirm," that are meant to constrain LLM's behavior for the remainder of a session but are silently dropped during compaction. To quantify this loss, we introduce COMPINT, an evaluation suite that evaluates compactors across three long-context scenarios: multi-turn chat, agentic trajectory, and long-horizon research.
Current compactors retain only 17% of injected SCs on average, and most perform worse than running the same task without compaction. Retention varies sharply with compactor, prompt, context length, SC phrasing, and injection location, showing that the loss is systematic rather than tied to any single setting. We propose an SC-aware extractor that runs alongside the compactor as a plug-and-play module, achieving over 90% retention across all three scenarios without modifying the compactor or LLM. The COMPINT evaluation suite and accompanying implementation are available at this https URL. 

---
# Reinforcement Learning based DBMS Buffer Pool Auto-Tuning for Optimal Memory Utilization 

**Authors**: Yifan Wang, Patrick Royer, Raphaël Féraud, David Delande  

**Link**: [PDF](https://arxiv.org/pdf/2608.11239)  

**Abstract**: Administering Database Management Systems (DBMS) instances requires Database Administrators (DBA) to balance performance in terms of Service Level Agreement (SLA) against resource usage, often prompting RAM over-allocation that wastes memory. We introduce MicroTune, an online RL-based buffer adjustment system that minimizes unnecessary memory allocation while ensuring SLA compliance. To identify the most effective RL core, we evaluate multiple algorithms under diverse benchmark workloads, training MicroTune on extensive traces of both external metrics (latency, throughput) and internal DBMS metrics (status variables and performance statistics). Experimental results demonstrate that MicroTune dynamically adapts buffer sizes to workload fluctuations, outperforming baselines by achieving significant memory savings with fewer SLA violations. These findings underscore the promise of reinforcement learning for adaptive resource management in DBMS environments. 

---
# TRACE Bench: Task-driven Roleplay Agentic Checklist Evaluation 

**Authors**: Jiahui Zhang, Ziwei Zhang, Yipeng Wang, Yibo Liu, Haozhou Pang, Yikai Hu, Hongyan Ren, Lan Zhou, Qi Gan, Kai Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.11236)  

**Abstract**: Roleplay evaluation should do more than assign a single score: it should reveal which role requirements were tested, which failed, and which dialogue evidence supports the judgment. We propose TRACE Bench, a task-driven agentic checklist evaluation framework. It decomposes each role profile offline into a fixed checklist, then uses a User Agent to converse naturally with the target roleplay model while privately updating checklist states from model responses. Scores therefore trace back to checklist items and supporting dialogue turns rather than a black-box holistic impression. For coverage cross-validation, we audit released M2 free-dialogue transcripts from the MiniMax Role-play Benchmark against the same role-derived checklist. The released free-chat transcripts cover only 73.74% of key role-profile points, whereas TRACE Bench reaches 99.91% coverage in fewer turns. Robustness experiments show stable rankings under repeated runs and User Agent replacement. Across 26 models, TRACE Bench reports overall rankings together with capability breakdowns and checklist traces. It also supports Closed-Loop Benchmark Evolution, distilling verification methods proven effective in failed traces so later evaluations can more reliably elicit and examine observed failure modes. 

---
# Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets 

**Authors**: Mark Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2608.11233)  

**Abstract**: A dense, pretrained language model can be retrofitted with recurrent depth and learn an iterative latent transition that persists after outcome-only annealing. Qwen2.5-0.5B-Instruct is split into a Prelude, a weight-tied Recurrent Block, and a Coda, with an identity-preserving one-loop path and a re-entry bridge on later loops. At loop 1 the retrofit remains non-inferior to its base on a preregistered ARC battery. Three findings. First, the mechanism is a reusable procedure rather than terminal-answer lookup, and installs at two budgets: 6M trained parameters over frozen base weights and 180M full-block. With intermediate-step supervision, the model computes one task step per loop and persists when only final answers are graded. The adapter matched the full block overall (83.8% versus 84.0%), led through depth 11, and trailed beyond. Verbal fine-tuning reached 79-86% on controlled verbal renderings (zero-shot transfer was minimal), and adapter verbal training begun from the installed mechanism outpaced matched fresh training by 18.6 points, including on a held-out test set. Second, the operation extrapolates to roughly 1.5 times its supervised depth, holding 70% accuracy through depth 18. Third, a same-size scratchpad-trained model matched the recurrent model within its learned horizon but collapsed beyond it. The recurrent model won overall, 84% versus 72%, retained 53% versus 2.5% beyond depth 10, and answered 7.6 times faster. An iterative transformer can therefore perform deeper reasoning in latent space faster than comparable or larger models fine-tuned on the same task, in a system-level comparison. A second task, running the rule in reverse, exposed the limits: the inverse was learnable in isolation, but no continuation acquired it while preserving the installed mechanism and general capability, a catastrophic-interference boundary. Learned depth selection remains open. 

---
# Backtrader-Bench: Benchmarking LLM Agents on Algorithmic Trading with Self-Generated MCQs 

**Authors**: Ruoxi Zhao, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2608.11232)  

**Abstract**: Evaluating LLM coding agents in algorithmic trading is difficult because static benchmarks risk data contamination and numerical backtest outputs require ground truth from actual code execution. We present Backtrader-Bench, a framework with two complementary pipelines. A deterministic multiple-choice question (MCQ) pipeline generates questions from backtest configurations across five trading strategies, 33 templates, and three difficulty tiers, with an independent checker that re-derives every answer. A generator-solver filtering pipeline autonomously mines harder questions: a generator writes questions verified by executable code, converts them to MCQs, and discards any that a no-tool solver can answer without code execution. We evaluate 11 models without tools (10 runs each) and four with-tools configurations on a 30-question curated set. Tool-augmented agents reach 90.0% accuracy in a single pass (GPT-5.5 and Opus 4.7), outperforming the best no-tools baselines (73.0%, averaged over 10 runs) by 17 percentage points. On 38 separately mined questions, no-tools accuracy drops further, with half the models falling to roughly random-chance level (25%). Beyond evaluation, the scalable MCQ infrastructure is designed to produce a training corpus for reinforcement learning, with the ultimate goal of building a specialized agent for quantitative trading workflows. 

---
# Evaluating LLM Generated Detection Rules in Cybersecurity 

**Authors**: Anna Bertiger, Bobby Filar, Aryan Luthra, Stefano Meschiari, Aiden Mitchell, Sam Scholten, Vivek Sharath  

**Link**: [PDF](https://arxiv.org/pdf/2509.16749)  

**Abstract**: LLMs are increasingly pervasive in the security environment, with limited measures of their effectiveness, which limits trust and usefulness to security practitioners. Here, we present an open-source evaluation framework and benchmark metrics for evaluating LLM-generated cybersecurity rules. The benchmark employs a holdout set-based methodology to measure the effectiveness of LLM-generated security rules in comparison to a human-generated corpus of rules. It provides three key metrics inspired by the way experts evaluate security rules, offering a realistic, multifaceted evaluation of the effectiveness of an LLM-based security rule generator. This methodology is illustrated using rules from Sublime Security's detection team and those written by Sublime Security's Automated Detection Engineer (ADE), with a thorough analysis of ADE's skills presented in the results section. 

---
