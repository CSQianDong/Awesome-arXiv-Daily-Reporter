# The Stochastic Gap: A Markovian Framework for Pre-Deployment Reliability and Oversight-Cost Auditing in Agentic Artificial Intelligence 

**Authors**: Biplab Pal, Santanu Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2603.24582)  

**Abstract**: Agentic artificial intelligence (AI) in organizations is a sequential decision problem constrained by reliability and oversight cost. When deterministic workflows are replaced by stochastic policies over actions and tool calls, the key question is not whether a next step appears plausible, but whether the resulting trajectory remains statistically supported, locally unambiguous, and economically governable. We develop a measure-theoretic Markov framework for this setting. The core quantities are state blind-spot mass B_n(tau), state-action blind mass B^SA_{pi,n}(tau), an entropy-based human-in-the-loop escalation gate, and an expected oversight-cost identity over the workflow visitation measure.
We instantiate the framework on the Business Process Intelligence Challenge 2019 purchase-to-pay log (251,734 cases, 1,595,923 events, 42 distinct workflow actions) and construct a log-driven simulated agent from a chronological 80/20 split of the same process. The main empirical finding is that a large workflow can appear well supported at the state level while retaining substantial blind mass over next-step decisions: refining the operational state to include case context, economic magnitude, and actor class expands the state space from 42 to 668 and raises state-action blind mass from 0.0165 at tau=50 to 0.1253 at tau=1000. On the held-out split, m(s) = max_a pi-hat(a|s) tracks realized autonomous step accuracy within 3.4 percentage points on average.
The same quantities that delimit statistically credible autonomy also determine expected oversight burden. The framework is demonstrated on a large-scale enterprise procurement workflow and is designed for direct application to engineering processes for which operational event logs are available. 

---
# Completeness of Unbounded Best-First Minimax and Descent Minimax 

**Authors**: Quentin Cohen-Solal  

**Link**: [PDF](https://arxiv.org/pdf/2603.24572)  

**Abstract**: In this article, we focus on search algorithms for two-player perfect information games, whose objective is to determine the best possible strategy, and ideally a winning strategy.
Unfortunately, some search algorithms for games in the literature are not able to always determine a winning strategy, even with an infinite search time. This is the case, for example, of the following algorithms: Unbounded Best-First Minimax and Descent Minimax, which are core algorithms in state-of-the-art knowledge-free reinforcement learning.
They were then improved with the so-called completion technique. However, whether this technique sufficiently improves these algorithms to allow them to always determine a winning strategy remained an open question until now.
To answer this question, we generalize the two algorithms (their versions using the completion technique), and we show that any algorithm of this class of algorithms computes the best strategy.
Finally, we experimentally show that the completion technique improves winning performance. 

---
# From Liar Paradox to Incongruent Sets: A Normal Form for Self-Reference 

**Authors**: Shalender Singh, Vishnu Priya Singh Parmar  

**Link**: [PDF](https://arxiv.org/pdf/2603.24527)  

**Abstract**: We introduce incongruent normal form (INF), a structural representation for self-referential semantic sentences. An INF replaces a self-referential sentence with a finite family of non-self-referential sentences that are individually satisfiable but not jointly satisfiable. This transformation isolates the semantic obstruction created by self-reference while preserving classical semantics locally and is accompanied by correctness theorems characterizing when global inconsistency arises from locally compatible commitments. We then study the role of incongruence as a structural source of semantic informativeness. Using a minimal model-theoretic notion of informativeness-understood as the ability of sentences to distinguish among admissible models-we show that semantic completeness precludes informativeness, while incongruence preserves it. Moreover, incongruence is not confined to paradoxical constructions: any consistent incomplete first-order theory admits finite incongruent families arising from incompatible complete extensions. In this sense, incompleteness manifests structurally as locally realizable but globally incompatible semantic commitments, providing a minimal formal basis for semantic knowledge. Finally, we introduce a quantitative semantic framework. In a canonical finite semantic-state setting, we model semantic commitments as Boolean functions and define a Fourier-analytic notion of semantic energy based on total influence. We derive uncertainty-style bounds relating semantic determinacy, informativeness, and spectral simplicity, and establish a matrix inequality bounding aggregate semantic variance by total semantic energy. These results show quantitatively that semantic informativeness cannot collapse into a single determinate state without unbounded energy cost, identifying incongruence as a fundamental structural and quantitative feature of semantic representation. 

---
# Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA 

**Authors**: John Ray B. Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2603.24481)  

**Abstract**: Miscalibrated confidence scores are a practical obstacle to deploying AI in clinical settings. A model that is always overconfident offers no useful signal for deferral. We present a multi-agent framework that combines domain-specific specialist agents with Two-Phase Verification and S-Score Weighted Fusion to improve both calibration and discrimination in medical multiple-choice question answering. Four specialist agents (respiratory, cardiology, neurology, gastroenterology) generate independent diagnoses using Qwen2.5-7B-Instruct. Each diagnosis is then subjected to a two-phase self-verification process that measures internal consistency and produces a Specialist Confidence Score (S-score). The S-scores drive a weighted fusion strategy that selects the final answer and calibrates the reported confidence. We evaluate across four experimental settings, covering 100-question and 250-question high-disagreement subsets of both MedQA-USMLE and MedMCQA. Calibration improvement is the central finding, with ECE reduced by 49-74% across all four settings, including the harder MedMCQA benchmark where these gains persist even when absolute accuracy is constrained by knowledge-intensive recall demands. On MedQA-250, the full system achieves ECE = 0.091 (74.4% reduction over the single-specialist baseline) and AUROC = 0.630 (+0.056) at 59.2% accuracy. Ablation analysis identifies Two-Phase Verification as the primary calibration driver and multi-agent reasoning as the primary accuracy driver. These results establish that consistency-based verification produces more reliable uncertainty estimates across diverse medical question types, providing a practical confidence signal for deferral in safety-critical clinical AI applications. 

---
# AI-Supervisor: Autonomous AI Research Supervision via a Persistent Research World Model 

**Authors**: Yunbo Long  

**Link**: [PDF](https://arxiv.org/pdf/2603.24402)  

**Abstract**: Existing automated research systems operate as stateless, linear pipelines, generating outputs without maintaining a persistent understanding of the research landscape. They process papers sequentially, propose ideas without structured gap analysis, and lack mechanisms for agents to verify or refine each other's findings. We present AutoProf (Autonomous Professor), a multi-agent orchestration framework where specialized agents provide end-to-end AI research supervision driven by human interests, from literature review through gap discovery, method development, evaluation, and paper writing, via autonomous exploration and self-correcting updates. Unlike sequential pipelines, AutoProf maintains a continuously evolving Research World Model implemented as a Knowledge Graph, capturing methods, benchmarks, limitations, and unexplored gaps as shared memory across agents. The framework introduces three contributions: first, structured gap discovery that decomposes methods into modules, evaluates them across benchmarks, and identifies module-level gaps; second, self-correcting discovery loops that analyze why modules succeed or fail, detect benchmark biases, and assess evaluation adequacy; third, self-improving development loops using cross-domain mechanism search to iteratively address failing components. All agents operate under a consensus mechanism where findings are validated before being committed to the shared model. The framework is model-agnostic, supports mainstream large language models, and scales elastically with token budget from lightweight exploration to full-scale investigation. 

---
# Bridging the Evaluation Gap: Standardized Benchmarks for Multi-Objective Search 

**Authors**: Hadar Peer, Carlos Hernandez, Sven Koenig, Ariel Felner, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2603.24084)  

**Abstract**: Empirical evaluation in multi-objective search (MOS) has historically suffered from fragmentation, relying on heterogeneous problem instances with incompatible objective definitions that make cross-study comparisons difficult. This standardization gap is further exacerbated by the realization that DIMACS road networks, a historical default benchmark for the field, exhibit highly correlated objectives that fail to capture diverse Pareto-front structures. To address this, we introduce the first comprehensive, standardized benchmark suite for exact and approximate MOS. Our suite spans four structurally diverse domains: real-world road networks, structured synthetic graphs, game-based grid environments, and high-dimensional robotic motion-planning roadmaps. By providing fixed graph instances, standardized start-goal queries, and both exact and approximate reference Pareto-optimal solution sets, this suite captures a full spectrum of objective interactions: from strongly correlated to strictly independent. Ultimately, this benchmark provides a common foundation to ensure future MOS evaluations are robust, reproducible, and structurally comprehensive. 

---
# Enhanced Mycelium of Thought (EMoT): A Bio-Inspired Hierarchical Reasoning Architecture with Strategic Dormancy and Mnemonic Encoding 

**Authors**: Florian Odi Stummer  

**Link**: [PDF](https://arxiv.org/pdf/2603.24065)  

**Abstract**: Current prompting paradigms for large language models (LLMs), including Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT), follow linear or tree-structured reasoning paths that lack persistent memory, strategic dormancy, and cross-domain synthesis. We present the Enhanced Mycelium of Thought (EMoT) framework, a bio-inspired reasoning architecture that organises cognitive processing into a four-level hierarchy (Micro, Meso, Macro, Meta), implements strategic dormancy and reactivation of reasoning nodes, and integrates a Memory Palace with five mnemonic encoding styles. EMoT is a research prototype for complex, multi-domain problems, not a general-purpose prompting enhancement. Two complementary evaluations reveal a characteristic trade-off. In a blind LLM-as-Judge evaluation across three domains, EMoT achieved near-parity with CoT (4.20 vs. 4.33/5.0) with higher stability, and outperformed CoT on Cross-Domain Synthesis (4.8 vs. 4.4). Ablation studies show that strategic dormancy is architecturally essential (quality collapsed from 4.2 to 1.0 when disabled). On a 15-item short-answer benchmark, EMoT (27%) substantially underperformed simpler baselines, confirming systematic overthinking on simple problems. These results are subject to important limitations: small sample sizes (n=3 complex cases, n=15 short-answer items), LLM-as-Judge evaluation with potential self-preference bias, and approximately 33-fold computational cost overhead. To our knowledge, EMoT is the first reasoning framework to combine hierarchical topology, strategic thought dormancy with reactivation, and mnemonic memory encoding in a single architecture. 

---
# ELITE: Experiential Learning and Intent-Aware Transfer for Self-improving Embodied Agents 

**Authors**: Bingqing Wei, Zhongyu Xia, Dingai Liu, Xiaoyu Zhou, Zhiwei Lin, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24018)  

**Abstract**: Vision-language models (VLMs) have shown remarkable general capabilities, yet embodied agents built on them fail at complex tasks, often skipping critical steps, proposing invalid actions, and repeating mistakes. These failures arise from a fundamental gap between the static training data of VLMs and the physical interaction for embodied tasks. VLMs can learn rich semantic knowledge from static data but lack the ability to interact with the world. To address this issue, we introduce ELITE, an embodied agent framework with {E}xperiential {L}earning and {I}ntent-aware {T}ransfer that enables agents to continuously learn from their own environment interaction experiences, and transfer acquired knowledge to procedurally similar tasks. ELITE operates through two synergistic mechanisms, \textit{i.e.,} self-reflective knowledge construction and intent-aware retrieval. Specifically, self-reflective knowledge construction extracts reusable strategies from execution trajectories and maintains an evolving strategy pool through structured refinement operations. Then, intent-aware retrieval identifies relevant strategies from the pool and applies them to current tasks. Experiments on the EB-ALFRED and EB-Habitat benchmarks show that ELITE achieves 9\% and 5\% performance improvement over base VLMs in the online setting without any supervision. In the supervised setting, ELITE generalizes effectively to unseen task categories, achieving better performance compared to state-of-the-art training-based methods. These results demonstrate the effectiveness of ELITE for bridging the gap between semantic understanding and reliable action execution. 

---
# Language-Grounded Multi-Agent Planning for Personalized and Fair Participatory Urban Sensing 

**Authors**: Xusen Guo, Mingxing Peng, Hongliang Lu, Hai Yang, Jun Ma, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24014)  

**Abstract**: Participatory urban sensing leverages human mobility for large-scale urban data collection, yet existing methods typically rely on centralized optimization and assume homogeneous participants, resulting in rigid assignments that overlook personal preferences and heterogeneous urban contexts. We propose MAPUS, an LLM-based multi-agent framework for personalized and fair participatory urban sensing. In our framework, participants are modeled as autonomous agents with individual profiles and schedules, while a coordinator agent performs fairness-aware selection and refines sensing routes through language-based negotiation. Experiments on real-world datasets show that MAPUS achieves competitive sensing coverage while substantially improving participant satisfaction and fairness, promoting more human-centric and sustainable urban sensing systems. 

---
# From Pixels to Digital Agents: An Empirical Study on the Taxonomy and Technological Trends of Reinforcement Learning Environments 

**Authors**: Lijing Luo, Yiben Luo, Alexey Gorbatovski, Sergey Kovalchuk, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23964)  

**Abstract**: The remarkable progress of reinforcement learning (RL) is intrinsically tied to the environments used to train and evaluate artificial agents. Moving beyond traditional qualitative reviews, this work presents a large-scale, data-driven empirical investigation into the evolution of RL environments. By programmatically processing a massive corpus of academic literature and rigorously distilling over 2,000 core publications, we propose a quantitative methodology to map the transition from isolated physical simulations to generalist, language-driven foundation agents. Implementing a novel, multi-dimensional taxonomy, we systematically analyze benchmarks against diverse application domains and requisite cognitive capabilities. Our automated semantic and statistical analysis reveals a profound, data-verified paradigm shift: the bifurcation of the field into a "Semantic Prior" ecosystem dominated by Large Language Models (LLMs) and a "Domain-Specific Generalization" ecosystem. Furthermore, we characterize the "cognitive fingerprints" of these distinct domains to uncover the underlying mechanisms of cross-task synergy, multi-domain interference, and zero-shot generalization. Ultimately, this study offers a rigorous, quantitative roadmap for designing the next generation of Embodied Semantic Simulators, bridging the gap between continuous physical control and high-level logical reasoning. 

---
# AnalogAgent: Self-Improving Analog Circuit Design Automation with LLM Agents 

**Authors**: Zhixuan Bao, Zhuoyi Lin, Jiageng Wang, Jinhai Hu, Yuan Gao, Yaoxin Wu, Xiaoli Li, Xun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23910)  

**Abstract**: Recent advances in large language models (LLMs) suggest strong potential for automating analog circuit design. Yet most LLM-based approaches rely on a single-model loop of generation, diagnosis, and correction, which favors succinct summaries over domain-specific insight and suffers from context attrition that erases critical technical details. To address these limitations, we propose AnalogAgent, a training-free agentic framework that integrates an LLM-based multi-agent system (MAS) with self-evolving memory (SEM) for analog circuit design automation. AnalogAgent coordinates a Code Generator, Design Optimizer, and Knowledge Curator to distill execution feedback into an adaptive playbook in SEM and retrieve targeted guidance for subsequent generation, enabling cross-task transfer without additional expert feedback, databases, or libraries. Across established benchmarks, AnalogAgent achieves 92% Pass@1 with Gemini and 97.4% Pass@1 with GPT-5. Moreover, with compact models (e.g., Qwen-8B), it yields a +48.8% average Pass@1 gain across tasks and reaches 72.1% Pass@1 overall, indicating that AnalogAgent substantially strengthens open-weight models for high-quality analog circuit design automation. 

---
# DUPLEX: Agentic Dual-System Planning via LLM-Driven Information Extraction 

**Authors**: Keru Hua, Ding Wang, Yaoying Gu, Xiaoguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.23909)  

**Abstract**: While Large Language Models (LLMs) provide semantic flexibility for robotic task planning, their susceptibility to hallucination and logical inconsistency limits their reliability in long-horizon domains. To bridge the gap between unstructured environments and rigorous plan synthesis, we propose DUPLEX, an agentic dual-system neuro-symbolic architecture that strictly confines the LLM to schema-guided information extraction rather than end-to-end planning or code generation. In our framework, a feed-forward Fast System utilizes a lightweight LLM to extract entities, relations etc. from natural language, deterministically mapping them into a Planning Domain Definition Language (PDDL) problem file for a classical symbolic planner. To resolve complex or underspecified scenarios, a Slow System is activated exclusively upon planning failure, leveraging solver diagnostics to drive a high-capacity LLM in iterative reflection and repair. Extensive evaluations across 12 classical and household planning domains demonstrate that DUPLEX significantly outperforms existing end-to-end and hybrid LLM baselines in both success rate and reliability. These results confirm that The key is not to make the LLM plan better, but to restrict the LLM to the part it is good at - structured semantic grounding - and leave logical plan synthesis to a symbolic planner. 

---
# The DeepXube Software Package for Solving Pathfinding Problems with Learned Heuristic Functions and Search 

**Authors**: Forest Agostinelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.23873)  

**Abstract**: DeepXube is a free and open-source Python package and command-line tool that seeks to automate the solution of pathfinding problems by using machine learning to learn heuristic functions that guide heuristic search algorithms tailored to deep neural networks (DNNs). DeepXube is comprised of the latest advances in deep reinforcement learning, heuristic search, and formal logic for solving pathfinding problems. This includes limited-horizon Bellman-based learning, hindsight experience replay, batched heuristic search, and specifying goals with answer-set programming. A robust multiple-inheritance structure simplifies the definition of pathfinding domains and the generation of training data. Training heuristic functions is made efficient through the automatic parallelization of the generation of training data across central processing units (CPUs) and reinforcement learning updates across graphics processing units (GPUs). Pathfinding algorithms that take advantage of the parallelism of GPUs and DNN architectures, such as batch weighted A* and Q* search and beam search are easily employed to solve pathfinding problems through command-line arguments. Finally, several convenient features for visualization, code profiling, and progress monitoring during training and solving are available. The GitHub repository is publicly available at this https URL. 

---
# When AI output tips to bad but nobody notices: Legal implications of AI's mistakes 

**Authors**: Dylan J. Restrepo, Nicholas J. Restrepo, Frank Y. Huo, Neil F. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2603.23857)  

**Abstract**: The adoption of generative AI across commercial and legal professions offers dramatic efficiency gains -- yet for law in particular, it introduces a perilous failure mode in which the AI fabricates fictitious case law, statutes, and judicial holdings that appear entirely authentic. Attorneys who unknowingly file such fabrications face professional sanctions, malpractice exposure, and reputational harm, while courts confront a novel threat to the integrity of the adversarial process. This failure mode is commonly dismissed as random `hallucination', but recent physics-based analysis of the Transformer's core mechanism reveals a deterministic component: the AI's internal state can cross a calculable threshold, causing its output to flip from reliable legal reasoning to authoritative-sounding fabrication. Here we present this science in a legal-industry setting, walking through a simulated brief-drafting scenario. Our analysis suggests that fabrication risk is not an anomalous glitch but a foreseeable consequence of the technology's design, with direct implications for the evolving duty of technological competence. We propose that legal professionals, courts, and regulators replace the outdated `black box' mental model with verification protocols based on how these systems actually fail. 

---
# SCoOP: Semantic Consistent Opinion Pooling for Uncertainty Quantification in Multiple Vision-Language Model Systems 

**Authors**: Chung-En Johnny Yu, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2603.23853)  

**Abstract**: Combining multiple Vision-Language Models (VLMs) can enhance multimodal reasoning and robustness, but aggregating heterogeneous models' outputs amplifies uncertainty and increases the risk of hallucinations. We propose SCoOP (Semantic-Consistent Opinion Pooling), a training-free uncertainty quantification (UQ) framework multi-VLM systems through uncertainty-weighted linear opinion pooling. Unlike prior UQ methods designed for single models, SCoOP explicitly measures collective, system-level uncertainty across multiple VLMs, enabling effective hallucination detection and abstention for highly uncertain samples. On ScienceQA, SCoOP achieves an AUROC of 0.866 for hallucination detection, outperforming baselines (0.732-0.757) by approximately 10-13%. For abstention, it attains an AURAC of 0.907, exceeding baselines (0.818-0.840) by 7-9%. Despite these gains, SCoOP introduces only microsecond-level aggregation overhead relative to the baselines, which is trivial compared to typical VLM inference time (on the order of seconds). These results demonstrate that SCoOP provides an efficient and principled mechanism for uncertainty-aware aggregation, advancing the reliability of multimodal AI systems. 

---
# VehicleMemBench: An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents 

**Authors**: Yuhao Chen, Yi Xu, Xinyun Ding, Xiang Fang, Shuochen Liu, Luxi Lin, Qingyu Zhang, Ya Li, Quan Liu, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23840)  

**Abstract**: With the growing demand for intelligent in-vehicle experiences, vehicle-based agents are evolving from simple assistants to long-term companions. This evolution requires agents to continuously model multi-user preferences and make reliable decisions in the face of inter-user preference conflicts and changing habits over time. However, existing benchmarks are largely limited to single-user, static question-answer settings, failing to capture the temporal evolution of preferences and the multi-user, tool-interactive nature of real vehicle environments. To address this gap, we introduce VehicleMemBench, a multi-user long-context memory benchmark built on an executable in-vehicle simulation environment. The benchmark evaluates tool use and memory by comparing the post-action environment state with a predefined target state, enabling objective and reproducible evaluation without LLM-based or human scoring. VehicleMemBench includes 23 tool modules, and each sample contains over 80 historical memory events. Experiments show that powerful models perform well on direct instruction tasks but struggle in scenarios involving memory evolution, particularly when user preferences change dynamically. Even advanced memory systems struggle to handle domain-specific memory requirements in this environment. These findings highlight the need for more robust and specialized memory management mechanisms to support long-term adaptive decision-making in real-world in-vehicle systems. To facilitate future research, we release the data and code. 

---
# Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation 

**Authors**: Han Zheng, Yining Ma, Brandon Araki, Jingkai Chen, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23838)  

**Abstract**: Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF often demand costly adaptations to classical search-based solvers. While machine learning methods have been explored, their superiority over search-based methods remains inconclusive. In this paper, we introduce Reinforcement Learning (RL) guided Rolling Horizon Prioritized Planning (RL-RH-PP), the first framework integrating RL with search-based planning for lifelong MAPF. Specifically, we leverage classical Prioritized Planning (PP) as a backbone for its simplicity and flexibility in integrating with a learning-based priority assignment policy. By formulating dynamic priority assignment as a Partially Observable Markov Decision Process (POMDP), RL-RH-PP exploits the sequential decision-making nature of lifelong planning while delegating complex spatial-temporal interactions among agents to reinforcement learning. An attention-based neural network autoregressively decodes priority orders on-the-fly, enabling efficient sequential single-agent planning by the PP planner. Evaluations in realistic warehouse simulations show that RL-RH-PP achieves the highest total throughput among baselines and generalizes effectively across agent densities, planning horizons, and warehouse layouts. Our interpretive analyses reveal that RL-RH-PP proactively prioritizes congested agents and strategically redirects agents from congestion, easing traffic flow and boosting throughput. These findings highlight the potential of learning-guided approaches to augment traditional heuristics in modern warehouse automation. 

---
# Efficient Benchmarking of AI Agents 

**Authors**: Franck Ndzomga  

**Link**: [PDF](https://arxiv.org/pdf/2603.23749)  

**Abstract**: Evaluating AI agents on comprehensive benchmarks is expensive because each evaluation requires interactive rollouts with tool use and multi-step reasoning. We study whether small task subsets can preserve agent rankings at substantially lower cost. Unlike static language model benchmarks, agent evaluation is subject to scaffold-driven distribution shift, since performance depends on the framework wrapping the underlying model. Across eight benchmarks, 33 agent scaffolds, and 70+ model configurations, we find that absolute score prediction degrades under this shift, while rank-order prediction remains stable. Exploiting this asymmetry, we propose a simple optimization-free protocol: evaluate new agents only on tasks with intermediate historical pass rates (30-70%). This mid-range difficulty filter, motivated by Item Response Theory, reduces the number of evaluation tasks by 44-70% while maintaining high rank fidelity under scaffold and temporal shifts. It provides more reliable rankings than random sampling, which exhibits high variance across seeds, and outperforms greedy task selection under distribution shift. These results suggest that reliable leaderboard ranking does not require full-benchmark evaluation. 

---
# LLMs Do Not Grade Essays Like Humans 

**Authors**: Jerin George Mathew, Sumayya Taher, Anindita Kundu, Denilson Barbosa  

**Link**: [PDF](https://arxiv.org/pdf/2603.23714)  

**Abstract**: Large language models have recently been proposed as tools for automated essay scoring, but their agreement with human grading remains unclear. In this work, we evaluate how LLM-generated scores compare with human grades and analyze the grading behavior of several models from the GPT and Llama families in an out-of-the-box setting, without task-specific training. Our results show that agreement between LLM and human scores remains relatively weak and varies with essay characteristics. In particular, compared to human raters, LLMs tend to assign higher scores to short or underdeveloped essays, while assigning lower scores to longer essays that contain minor grammatical or spelling errors. We also find that the scores generated by LLMs are generally consistent with the feedback they generate: essays receiving more praise tend to receive higher scores, while essays receiving more criticism tend to receive lower scores. These results suggest that LLM-generated scores and feedback follow coherent patterns but rely on signals that differ from those used by human raters, resulting in limited alignment with human grading practices. Nevertheless, our work shows that LLMs produce feedback that is consistent with their grading and that they can be reliably used in supporting essay scoring. 

---
# Grounding Vision and Language to 3D Masks for Long-Horizon Box Rearrangement 

**Authors**: Ashish Malik, Caleb Lowe, Aayam Shrestha, Stefan Lee, Fuxin Li, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2603.23676)  

**Abstract**: We study long-horizon planning in 3D environments from under-specified natural-language goals using only visual observations, focusing on multi-step 3D box rearrangement tasks. Existing approaches typically rely on symbolic planners with brittle relational grounding of states and goals, or on direct action-sequence generation from 2D vision-language models (VLMs). Both approaches struggle with reasoning over many objects, rich 3D geometry, and implicit semantic constraints. Recent advances in 3D VLMs demonstrate strong grounding of natural-language referents to 3D segmentation masks, suggesting the potential for more general planning capabilities. We extend existing 3D grounding models and propose Reactive Action Mask Planner (RAMP-3D), which formulates long-horizon planning as sequential reactive prediction of paired 3D masks: a "which-object" mask indicating what to pick and a "which-target-region" mask specifying where to place it. The resulting system processes RGB-D observations and natural-language task specifications to reactively generate multi-step pick-and-place actions for 3D box rearrangement. We conduct experiments across 11 task variants in warehouse-style environments with 1-30 boxes and diverse natural-language constraints. RAMP-3D achieves 79.5% success rate on long-horizon rearrangement tasks and significantly outperforms 2D VLM-based baselines, establishing mask-based reactive policies as a promising alternative to symbolic pipelines for long-horizon planning. 

---
# GTO Wizard Benchmark 

**Authors**: Marc-Antoine Provost, Nejc Ilenic, Christopher Solinas, Philippe Beardsell  

**Link**: [PDF](https://arxiv.org/pdf/2603.23660)  

**Abstract**: We introduce GTO Wizard Benchmark, a public API and standardized evaluation framework for benchmarking algorithms in Heads-Up No-Limit Texas Hold'em (HUNL). The benchmark evaluates agents against GTO Wizard AI, a state-of-the-art superhuman poker agent that approximates Nash Equilibria, and defeated Slumbot, the 2018 Annual Computer Poker Competition champion and previous strongest publicly accessible HUNL benchmark, by $19.4$ $\pm$ $4.1$ bb/100. Variance is a fundamental challenge in poker evaluation; we address this by integrating AIVAT, a provably unbiased variance reduction technique that achieves equivalent statistical significance with ten times fewer hands than naive Monte Carlo evaluation. We conduct a comprehensive benchmarking study of state-of-the-art large language models under zero-shot conditions, including GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro, Grok 4, and others. Initial results and analysis reveal dramatic progress in LLM reasoning over recent years, yet all models remain far below the baseline established by our benchmark. Qualitative analysis reveals clear opportunities for improvement, including representation and the ability to reason over hidden states. This benchmark provides researchers with a precise and quantifiable setting to evaluate advances in planning and reasoning in multi-agent systems with partial observability. 

---
# Can LLM Agents Be CFOs? A Benchmark for Resource Allocation in Dynamic Enterprise Environments 

**Authors**: Yi Han, Lingfei Qian, Yan Wang, Yueru He, Xueqing Peng, Dongji Feng, Yankai Chen, Haohang Li, Yupeng Cao, Jimin Huang, Xue Liu, Jian-Yun Nie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23638)  

**Abstract**: Large language models (LLMs) have enabled agentic systems that can reason, plan, and act across complex tasks, but it remains unclear whether they can allocate resources effectively under uncertainty. Unlike short-horizon reactive decisions, allocation requires committing scarce resources over time while balancing competing objectives and preserving flexibility for future needs. We introduce EnterpriseArena, the first benchmark for evaluating agents on long-horizon enterprise resource allocation. It instantiates CFO-style decision-making in a 132-month enterprise simulator combining firm-level financial data, anonymized business documents, macroeconomic and industry signals, and expert-validated operating rules. The environment is partially observable and reveals the state only through budgeted organizational tools, forcing agents to trade off information acquisition against conserving scarce resources. Experiments on eleven advanced LLMs show that this setting remains highly challenging: only 16% of runs survive the full horizon, and larger models do not reliably outperform smaller ones. These results identify long-horizon resource allocation under uncertainty as a distinct capability gap for current LLM agents. 

---
# Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework 

**Authors**: Zeinab Dehghani, Rameez Raja Kureshi, Koorosh Aslansefat, Faezeh Alsadat Abedi, Dhavalkumar Thakker, Lisa Greaves, Bhupesh Kumar Mishra, Baseer Ahmad, Tanaya Maslekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23625)  

**Abstract**: Artificial intelligence (AI) is increasingly being explored in health and social care to reduce administrative workload and allow staff to spend more time on patient care. This paper evaluates a voice-enabled Care Home Smart Speaker designed to support everyday activities in residential care homes, including spoken access to resident records, reminders, and scheduling tasks. A safety-focused evaluation framework is presented that examines the system end-to-end, combining Whisper-based speech recognition with retrieval-augmented generation (RAG) approaches (hybrid, sparse, and dense). Using supervised care-home trials and controlled testing, we evaluated 330 spoken transcripts across 11 care categories, including 184 reminder-containing interactions. These evaluations focus on (i) correct identification of residents and care categories, (ii) reminder recognition and extraction, and (iii) end-to-end scheduling correctness under uncertainty (including safe deferral/clarification). Given the safety-critical nature of care homes, particular attention is also paid to reliability in noisy environments and across diverse accents, supported by confidence scoring, clarification prompts, and human-in-the-loop oversight. In the best-performing configuration (GPT-5.2), resident ID and care category matching reached 100% (95% CI: 98.86-100), while reminder recognition reached 89.09\% (95% CI: 83.81-92.80) with zero missed reminders (100% recall) but some false positives. End-to-end scheduling via calendar integration achieved 84.65% exact reminder-count agreement (95% CI: 78.00-89.56), indicating remaining edge cases in converting informal spoken instructions into actionable events. The findings suggest that voice-enabled systems, when carefully evaluated and appropriately safeguarded, can support accurate documentation, effective task management, and trustworthy use of AI in care home settings. 

---
# Environment Maps: Structured Environmental Representations for Long-Horizon Agents 

**Authors**: Yenchia Feng, Chirag Sharma, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2603.23610)  

**Abstract**: Although large language models (LLMs) have advanced rapidly, robust automation of complex software workflows remains an open problem. In long-horizon settings, agents frequently suffer from cascading errors and environmental stochasticity; a single misstep in a dynamic interface can lead to task failure, resulting in hallucinations or trial-and-error. This paper introduces $\textit{Environment Maps}$: a persistent, agent-agnostic representation that mitigates these failures by consolidating heterogeneous evidence, such as screen recordings and execution traces, into a structured graph. The representation consists of four core components: (1) Contexts (abstracted locations), (2) Actions (parameterized affordances), (3) Workflows (observed trajectories), and (4) Tacit Knowledge (domain definitions and reusable procedures). We evaluate this framework on the WebArena benchmark across five domains. Agents equipped with environment maps achieve a 28.2% success rate, nearly doubling the performance of baselines limited to session-bound context (14.2%) and outperforming agents that have access to the raw trajectory data used to generate the environment maps (23.3%). By providing a structured interface between the model and the environment, Environment Maps establish a persistent foundation for long-horizon planning that is human-interpretable, editable, and incrementally refinable. 

---
# PLDR-LLMs Reason At Self-Organized Criticality 

**Authors**: Burc Gokden  

**Link**: [PDF](https://arxiv.org/pdf/2603.23539)  

**Abstract**: We show that PLDR-LLMs pretrained at self-organized criticality exhibit reasoning at inference time. The characteristics of PLDR-LLM deductive outputs at criticality is similar to second-order phase transitions. At criticality, the correlation length diverges, and the deductive outputs attain a metastable steady state. The steady state behaviour suggests that deductive outputs learn representations equivalent to scaling functions, universality classes and renormalization groups from the training dataset, leading to generalization and reasoning capabilities in the process. We can then define an order parameter from the global statistics of the model's deductive output parameters at inference. The reasoning capabilities of a PLDR-LLM is better when its order parameter is close to zero at criticality. This observation is supported by the benchmark scores of the models trained at near-criticality and sub-criticality. Our results provide a self-contained explanation on how reasoning manifests in large language models, and the ability to reason can be quantified solely from global model parameter values of the deductive outputs at steady state, without any need for evaluation of curated benchmark datasets through inductive output for reasoning and comprehension. 

---
# Retrieval Improvements Do Not Guarantee Better Answers: A Study of RAG for AI Policy QA 

**Authors**: Saahil Mathur, Ryan David Rittner, Vedant Ajit Thakur, Daniel Stuart Schiff, Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2603.24580)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly used to analyze complex policy documents, but achieving sufficient reliability for expert usage remains challenging in domains characterized by dense legal language and evolving, overlapping regulatory frameworks. We study the application of RAG to AI governance and policy analysis using the AI Governance and Regulatory Archive (AGORA) corpus, a curated collection of 947 AI policy documents. Our system combines a ColBERT-based retriever fine-tuned with contrastive learning and a generator aligned to human preferences using Direct Preference Optimization (DPO). We construct synthetic queries and collect pairwise preferences to adapt the system to the policy domain. Through experiments evaluating retrieval quality, answer relevance, and faithfulness, we find that domain-specific fine-tuning improves retrieval metrics but does not consistently improve end-to-end question answering performance. In some cases, stronger retrieval counterintuitively leads to more confident hallucinations when relevant documents are absent from the corpus. These results highlight a key concern for those building policy-focused RAG systems: improvements to individual components do not necessarily translate to more reliable answers. Our findings provide practical insights for designing grounded question-answering systems over dynamic regulatory corpora. 

---
# EndoVGGT: GNN-Enhanced Depth Estimation for Surgical 3D Reconstruction 

**Authors**: Falong Fan, Yi Xie, Arnis Lektauers, Bo Liu, Jerzy Rozenblit  

**Link**: [PDF](https://arxiv.org/pdf/2603.24577)  

**Abstract**: Accurate 3D reconstruction of deformable soft tissues is essential for surgical robotic perception. However, low-texture surfaces, specular highlights, and instrument occlusions often fragment geometric continuity, posing a challenge for existing fixed-topology approaches. To address this, we propose EndoVGGT, a geometry-centric framework equipped with a Deformation-aware Graph Attention (DeGAT) module. Rather than using static spatial neighborhoods, DeGAT dynamically constructs feature-space semantic graphs to capture long-range correlations among coherent tissue regions. This enables robust propagation of structural cues across occlusions, enforcing global consistency and improving non-rigid deformation recovery. Extensive experiments on SCARED show that our method significantly improves fidelity, increasing PSNR by 24.6% and SSIM by 9.1% over prior state-of-the-art. Crucially, EndoVGGT exhibits strong zero-shot cross-dataset generalization to the unseen SCARED and EndoNeRF domains, confirming that DeGAT learns domain-agnostic geometric priors. These results highlight the efficacy of dynamic feature-space modeling for consistent surgical 3D reconstruction. 

---
# Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation 

**Authors**: Xinying Guo, Chenxi Jiang, Hyun Bin Kim, Ying Sun, Yang Xiao, Yuhang Han, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24576)  

**Abstract**: Robotic manipulation often requires memory: occlusion and state changes can make decision-time observations perceptually aliased, making action selection non-Markovian at the observation level because the same observation may arise from different interaction histories. Most embodied agents implement memory via semantically compressed traces and similarity-based retrieval, which discards disambiguating fine-grained perceptual cues and can return perceptually similar but decision-irrelevant episodes. Inspired by human episodic memory, we propose Chameleon, which writes geometry-grounded multimodal tokens to preserve disambiguating context and produces goal-directed recall through a differentiable memory stack. We also introduce Camo-Dataset, a real-robot UR5e dataset spanning episodic recall, spatial tracking, and sequential manipulation under perceptual aliasing. Across tasks, Chameleon consistently improves decision reliability and long-horizon control over strong baselines in perceptually confusable settings. 

---
# VFIG: Vectorizing Complex Figures in SVG with Vision-Language Models 

**Authors**: Qijia He, Xunmei Liu, Hammaad Memon, Ziang Li, Zixian Ma, Jaemin Cho, Jason Ren, Daniel S Weld, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2603.24575)  

**Abstract**: Scalable Vector Graphics (SVG) are an essential format for technical illustration and digital design, offering precise resolution independence and flexible semantic editability. In practice, however, original vector source files are frequently lost or inaccessible, leaving only "flat" rasterized versions (e.g., PNG or JPEG) that are difficult to modify or scale. Manually reconstructing these figures is a prohibitively labor-intensive process, requiring specialized expertise to recover the original geometric intent. To bridge this gap, we propose VFIG, a family of Vision-Language Models trained for complex and high-fidelity figure-to-SVG conversion. While this task is inherently data-driven, existing datasets are typically small-scale and lack the complexity of professional diagrams. We address this by introducing VFIG-DATA, a large-scale dataset of 66K high-quality figure-SVG pairs, curated from a diverse mix of real-world paper figures and procedurally generated diagrams. Recognizing that SVGs are composed of recurring primitives and hierarchical local structures, we introduce a coarse-to-fine training curriculum that begins with supervised fine-tuning (SFT) to learn atomic primitives and transitions to reinforcement learning (RL) refinement to optimize global diagram fidelity, layout consistency, and topological edge cases. Finally, we introduce VFIG-BENCH, a comprehensive evaluation suite with novel metrics designed to measure the structural integrity of complex figures. VFIG achieves state-of-the-art performance among open-source models and performs on par with GPT-5.2, achieving a VLM-Judge score of 0.829 on VFIG-BENCH. 

---
# Anti-I2V: Safeguarding your photos from malicious image-to-video generation 

**Authors**: Duc Vu, Anh Nguyen, Chi Tran, Anh Tran  

**Link**: [PDF](https://arxiv.org/pdf/2603.24570)  

**Abstract**: Advances in diffusion-based video generation models, while significantly improving human animation, poses threats of misuse through the creation of fake videos from a specific person's photo and text prompts. Recent efforts have focused on adversarial attacks that introduce crafted perturbations to protect images from diffusion-based models. However, most existing approaches target image generation, while relatively few explicitly address image-to-video diffusion models (VDMs), and most primarily focus on UNet-based architectures. Hence, their effectiveness against Diffusion Transformer (DiT) models remains largely under-explored, as these models demonstrate improved feature retention, and stronger temporal consistency due to larger capacity and advanced attention mechanisms. In this work, we introduce Anti-I2V, a novel defense against malicious human image-to-video generation, applicable across diverse diffusion backbones. Instead of restricting noise updates to the RGB space, Anti-I2V operates in both the $L$*$a$*$b$* and frequency domains, improving robustness and concentrating on salient pixels. We then identify the network layers that capture the most distinct semantic features during the denoising process to design appropriate training objectives that maximize degradation of temporal coherence and generation fidelity. Through extensive validation, Anti-I2V demonstrates state-of-the-art defense performance against diverse video diffusion models, offering an effective solution to the problem. 

---
# The Free-Market Algorithm: Self-Organizing Optimization for Open-Ended Complex Systems 

**Authors**: Martin Jaraiz  

**Link**: [PDF](https://arxiv.org/pdf/2603.24559)  

**Abstract**: We introduce the Free-Market Algorithm (FMA), a novel metaheuristic inspired by free-market economics. Unlike Genetic Algorithms, Particle Swarm Optimization, and Simulated Annealing -- which require prescribed fitness functions and fixed search spaces -- FMA uses distributed supply-and-demand dynamics where fitness is emergent, the search space is open-ended, and solutions take the form of hierarchical pathway networks. Autonomous agents discover rules, trade goods, open and close firms, and compete for demand with no centralized controller.
FMA operates through a three-layer architecture: a universal market mechanism (supply, demand, competition, selection), pluggable domain-specific behavioral rules, and domain-specific observation. The market mechanism is identical across applications; only the behavioral rules change.
Validated in two unrelated domains. In prebiotic chemistry, starting from 900 bare atoms (C, H, O, N), FMA discovers all 12 feasible amino acid formulas, all 5 nucleobases, the formose sugar chain, and Krebs cycle intermediates in under 5 minutes on a laptop -- with up to 240 independent synthesis routes per product. In macroeconomic forecasting, reading a single input-output table with zero estimated parameters, FMA achieves Mean Absolute Error of 0.42 percentage points for non-crisis GDP prediction, comparable to professional forecasters, portable to 33 countries.
Assembly Theory alignment shows that FMA provides the first explicit, tunable mechanism for the selection signatures described by Sharma et al. (Nature, 2023). The event-driven assembly dynamics resonate with foundational programs in physics -- causal set theory, relational quantum mechanics, constructor theory -- suggesting that Darwinian market dynamics may reflect a deeper organizational principle that lead to the unfolding of Nature itself. 

---
# LensWalk: Agentic Video Understanding by Planning How You See in Videos 

**Authors**: Keliang Li, Yansong Li, Hongze Shen, Mengdi Liu, Hong Chang, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2603.24558)  

**Abstract**: The dense, temporal nature of video presents a profound challenge for automated analysis. Despite the use of powerful Vision-Language Models, prevailing methods for video understanding are limited by the inherent disconnect between reasoning and perception: they rely on static, pre-processed information and cannot actively seek raw evidence from video as their understanding evolves. To address this, we introduce LensWalk, a flexible agentic framework that empowers a Large Language Model reasoner to control its own visual observation actively. LensWalk establishes a tight reason-plan-observe loop where the agent dynamically specifies, at each step, the temporal scope and sampling density of the video it observes. Using a suite of versatile, Vision-Language Model based tools parameterized by these specifications, the agent can perform broad scans for cues, focus on specific segments for fact extraction, and stitch evidence from multiple moments for holistic verification. This design allows for progressive, on-demand evidence gathering that directly serves the agent's evolving chain of thought. Without requiring any model fine-tuning, LensWalk delivers substantial, plug-and-play performance gains on multiple model recipes, boosting their accuracy by over 5\% on challenging long-video benchmarks like LVBench and Video-MME. Our analysis reveals that enabling an agent to control how it sees is key to unlocking more accurate, robust, and interpretable video reasoning. 

---
# Evaluating Chunking Strategies For Retrieval-Augmented Generation in Oil and Gas Enterprise Documents 

**Authors**: Samuel Taiwo, Mohd Amaluddin Yusoff  

**Link**: [PDF](https://arxiv.org/pdf/2603.24556)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a framework to address the constraints of Large Language Models (LLMs). Yet, its effectiveness fundamentally hinges on document chunking - an often-overlooked determinant of its quality. This paper presents an empirical study quantifying performance differences across four chunking strategies: fixed-size sliding window, recursive, breakpoint-based semantic, and structure-aware. We evaluated these methods using a proprietary corpus of oil and gas enterprise documents, including text-heavy manuals, table-heavy specifications, and piping and instrumentation diagrams (P and IDs). Our findings show that structure-aware chunking yields higher overall retrieval effectiveness, particularly in top-K metrics, and incurs significantly lower computational costs than semantic or baseline strategies. Crucially, all four methods demonstrated limited effectiveness on P and IDs, underscoring a core limitation of purely text-based RAG within visually and spatially encoded documents. We conclude that while explicit structure preservation is essential for specialised domains, future work must integrate multimodal models to overcome current limitations. 

---
# A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English 

**Authors**: Dana Serditova, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24549)  

**Abstract**: Automatic Speech Recognition (ASR) systems are widely used in everyday communication, education, healthcare, and industry, yet their performance remains uneven across speakers, particularly when dialectal variation diverges from the mainstream accents represented in training data. This study investigates ASR bias through a sociolinguistic analysis of Newcastle English, a regional variety of North-East England that has been shown to challenge current speech recognition technologies. Using spontaneous speech from the Diachronic Electronic Corpus of Tyneside English (DECTE), we evaluate the output of a state-of-the-art commercial ASR system and conduct a fine-grained analysis of more than 3,000 transcription errors. Errors are classified by linguistic domain and examined in relation to social variables including gender, age, and socioeconomic status. In addition, an acoustic case study of selected vowel features demonstrates how gradient phonetic variation contributes directly to misrecognition.
The results show that phonological variation accounts for the majority of errors, with recurrent failures linked to dialect-specific features like vowel quality and glottalisation, as well as local vocabulary and non-standard grammatical forms. Error rates also vary across social groups, with higher error frequencies observed for men and for speakers at the extremes of the age spectrum. These findings indicate that ASR errors are not random but socially patterned and can be explained from a sociolinguistic perspective. Thus, the study demonstrates the importance of incorporating sociolinguistic expertise into the evaluation and development of speech technologies and argues that more equitable ASR systems require explicit attention to dialectal variation and community-based speech data. 

---
# SEGAR: Selective Enhancement for Generative Augmented Reality 

**Authors**: Fanjun Bu, Chenyang Yuan, Hiroshi Yasuda  

**Link**: [PDF](https://arxiv.org/pdf/2603.24541)  

**Abstract**: Generative world models offer a compelling foundation for augmented-reality (AR) applications: by predicting future image sequences that incorporate deliberate visual edits, they enable temporally coherent, augmented future frames that can be computed ahead of time and cached, avoiding per-frame rendering from scratch in real time. In this work, we present SEGAR, a preliminary framework that combines a diffusion-based world model with a selective correction stage to support this vision. The world model generates augmented future frames with region-specific edits while preserving others, and the correction stage subsequently aligns safety-critical regions with real-world observations while preserving intended augmentations elsewhere. We demonstrate this pipeline in driving scenarios as a representative setting where semantic region structure is well defined and real-world feedback is readily available. We view this as an early step toward generative world models as practical AR infrastructure, where future frames can be generated, cached, and selectively corrected on demand. 

---
# CliPPER: Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures for Event Recognition 

**Authors**: Florian Stilz, Vinkle Srivastav, Nassir Navab, Nicolas Padoy  

**Link**: [PDF](https://arxiv.org/pdf/2603.24539)  

**Abstract**: Video-language foundation models have proven to be highly effective in zero-shot applications across a wide range of tasks. A particularly challenging area is the intraoperative surgical procedure domain, where labeled data is scarce, and precise temporal understanding is often required for complex downstream tasks. To address this challenge, we introduce CliPPER (Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures for Event Recognition), a novel video-language pretraining framework trained on surgical lecture videos. Our method is designed for fine-grained temporal video-text recognition and introduces several novel pretraining strategies to improve multimodal alignment in long-form surgical videos. Specifically, we propose Contextual Video-Text Contrastive Learning (VTC_CTX) and Clip Order Prediction (COP) pretraining objectives, both of which leverage temporal and contextual dependencies to enhance local video understanding. In addition, we incorporate a Cycle-Consistency Alignment over video-text matches within the same surgical video to enforce bidirectional consistency and improve overall representation coherence. Moreover, we introduce a more refined alignment loss, Frame-Text Matching (FTM), to improve the alignment between video frames and text. As a result, our model establishes a new state-of-the-art across multiple public surgical benchmarks, including zero-shot recognition of phases, steps, instruments, and triplets. The source code and pretraining captions can be found at this https URL. 

---
# UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience 

**Authors**: Zichuan Lin, Feiyu Liu, Yijun Yang, Jiafei Lyu, Yiming Gao, Yicheng Liu, Zhicong Lu, Yangbin Yu, Mingyu Yang, Junyou Li, Deheng Ye, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24533)  

**Abstract**: Autonomous mobile GUI agents have attracted increasing attention along with the advancement of Multimodal Large Language Models (MLLMs). However, existing methods still suffer from inefficient learning from failed trajectories and ambiguous credit assignment under sparse rewards for long-horizon GUI tasks. To that end, we propose UI-Voyager, a novel two-stage self-evolving mobile GUI agent. In the first stage, we employ Rejection Fine-Tuning (RFT), which enables the continuous co-evolution of data and models in a fully autonomous loop. The second stage introduces Group Relative Self-Distillation (GRSD), which identifies critical fork points in group rollouts and constructs dense step-level supervision from successful trajectories to correct failed ones. Extensive experiments on AndroidWorld show that our 4B model achieves an 81.0% Pass@1 success rate, outperforming numerous recent baselines and exceeding human-level performance. Ablation and case studies further verify the effectiveness of GRSD. Our method represents a significant leap toward efficient, self-evolving, and high-performance mobile GUI automation without expensive manual data annotation. 

---
# No Single Metric Tells the Whole Story: A Multi-Dimensional Evaluation Framework for Uncertainty Attributions 

**Authors**: Emily Schiller, Teodor Chiaburu, Marco Zullich, Luca Longo  

**Link**: [PDF](https://arxiv.org/pdf/2603.24524)  

**Abstract**: Research on explainable AI (XAI) has frequently focused on explaining model predictions. More recently, methods have been proposed to explain prediction uncertainty by attributing it to input features (uncertainty attributions). However, the evaluation of these methods remains inconsistent as studies rely on heterogeneous proxy tasks and metrics, hindering comparability. We address this by aligning uncertainty attributions with the well-established Co-12 framework for XAI evaluation. We propose concrete implementations for the correctness, consistency, continuity, and compactness properties. Additionally, we introduce conveyance, a property tailored to uncertainty attributions that evaluates whether controlled increases in epistemic uncertainty reliably propagate to feature-level attributions. We demonstrate our evaluation framework with eight metrics across combinations of uncertainty quantification and feature attribution methods on tabular and image data. Our experiments show that gradient-based methods consistently outperform perturbation-based approaches in consistency and conveyance, while Monte-Carlo dropconnect outperforms Monte-Carlo dropout in most metrics. Although most metrics rank the methods consistently across samples, inter-method agreement remains low. This suggests no single metric sufficiently evaluates uncertainty attribution quality. The proposed evaluation framework contributes to the body of knowledge by establishing a foundation for systematic comparison and development of uncertainty attribution methods. 

---
# Claudini: Autoresearch Discovers State-of-the-Art Adversarial Attack Algorithms for LLMs 

**Authors**: Alexander Panfilov, Peter Romov, Igor Shilov, Yves-Alexandre de Montjoye, Jonas Geiping, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.24511)  

**Abstract**: LLM agents like Claude Code can not only write code but also be used for autonomous AI research and engineering \citep{rank2026posttrainbench, novikov2025alphaevolve}. We show that an \emph{autoresearch}-style pipeline \citep{karpathy2026autoresearch} powered by Claude Code discovers novel white-box adversarial attack \textit{algorithms} that \textbf{significantly outperform all existing (30+) methods} in jailbreaking and prompt injection evaluations.
Starting from existing attack implementations, such as GCG~\citep{zou2023universal}, the agent iterates to produce new algorithms achieving up to 40\% attack success rate on CBRN queries against GPT-OSS-Safeguard-20B, compared to $\leq$10\% for existing algorithms (\Cref{fig:teaser}, left).
The discovered algorithms generalize: attacks optimized on surrogate models transfer directly to held-out models, achieving \textbf{100\% ASR against Meta-SecAlign-70B} \citep{chen2025secalign} versus 56\% for the best baseline (\Cref{fig:teaser}, middle). Extending the findings of~\cite{carlini2025autoadvexbench}, our results are an early demonstration that incremental safety and security research can be automated using LLM agents. White-box adversarial red-teaming is particularly well-suited for this: existing methods provide strong starting points, and the optimization objective yields dense, quantitative feedback. We release all discovered attacks alongside baseline implementations and evaluation code at this https URL. 

---
# Counting Without Numbers \& Finding Without Words 

**Authors**: Badri Narayana Patro  

**Link**: [PDF](https://arxiv.org/pdf/2603.24470)  

**Abstract**: Every year, 10 million pets enter shelters, separated from their families. Despite desperate searches by both guardians and lost animals, 70% never reunite, not because matches do not exist, but because current systems look only at appearance, while animals recognize each other through sound. We ask, why does computer vision treat vocalizing species as silent visual objects? Drawing on five decades of cognitive science showing that animals perceive quantity approximately and communicate identity acoustically, we present the first multimodal reunification system integrating visual and acoustic biometrics. Our species-adaptive architecture processes vocalizations from 10Hz elephant rumbles to 4kHz puppy whines, paired with probabilistic visual matching that tolerates stress-induced appearance changes. This work demonstrates that AI grounded in biological communication principles can serve vulnerable populations that lack human language. 

---
# Integrating Causal Machine Learning into Clinical Decision Support Systems: Insights from Literature and Practice 

**Authors**: Domenique Zipperling, Lukas Schmidt, Benedikt Hahn, Niklas Kühl, Steven Kimbrough  

**Link**: [PDF](https://arxiv.org/pdf/2603.24448)  

**Abstract**: Current clinical decision support systems (CDSSs) typically base their predictions on correlation, not causation. In recent years, causal machine learning (ML) has emerged as a promising way to improve decision-making with CDSSs by offering interpretable, treatment-specific reasoning. However, existing research often emphasizes model development rather than designing clinician-facing interfaces. To address this gap, we investigated how CDSSs based on causal ML should be designed to effectively support collaborative clinical decision-making. Using a design science research methodology, we conducted a structured literature review and interviewed experienced physicians. From these, we derived eight empirically grounded design requirements, developed seven design principles, and proposed nine practical design features. Our results establish guidance for designing CDSSs that deliver causal insights, integrate seamlessly into clinical workflows, and support trust, usability, and human-AI collaboration. We also reveal tensions around automation, responsibility, and regulation, highlighting the need for an adaptive certification process for ML-based medical products. 

---
# CUA-Suite: Massive Human-annotated Video Demonstrations for Computer-Use Agents 

**Authors**: Xiangru Jian, Shravan Nayak, Kevin Qinghong Lin, Aarash Feizi, Kaixin Li, Patrice Bechard, Spandana Gella, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2603.24440)  

**Abstract**: Computer-use agents (CUAs) hold great promise for automating complex desktop workflows, yet progress toward general-purpose agents is bottlenecked by the scarcity of continuous, high-quality human demonstration videos. Recent work emphasizes that continuous video, not sparse screenshots, is the critical missing ingredient for scaling these agents. However, the largest existing open dataset, ScaleCUA, contains only 2 million screenshots, equating to less than 20 hours of video. To address this bottleneck, we introduce CUA-Suite, a large-scale ecosystem of expert video demonstrations and dense annotations for professional desktop computer-use agents. At its core is VideoCUA, which provides approximately 10,000 human-demonstrated tasks across 87 diverse applications with continuous 30 fps screen recordings, kinematic cursor traces, and multi-layerfed reasoning annotations, totaling approximately 55 hours and 6 million frames of expert video. Unlike sparse datasets that capture only final click coordinates, these continuous video streams preserve the full temporal dynamics of human interaction, forming a superset of information that can be losslessly transformed into the formats required by existing agent frameworks. CUA-Suite further provides two complementary resources: UI-Vision, a rigorous benchmark for evaluating grounding and planning capabilities in CUAs, and GroundCUA, a large-scale grounding dataset with 56K annotated screenshots and over 3.6 million UI element annotations. Preliminary evaluation reveals that current foundation action models struggle substantially with professional desktop applications (~60% task failure rate). Beyond evaluation, CUA-Suite's rich multimodal corpus supports emerging research directions including generalist screen parsing, continuous spatial control, video-based reward modeling, and visual world models. All data and models are publicly released. 

---
# Enes Causal Discovery 

**Authors**: Alexis Kafantaris  

**Link**: [PDF](https://arxiv.org/pdf/2603.24436)  

**Abstract**: Enes The proposed architecture is a mixture of experts, which allows for the model entities, such as the causal relationships, to be further parameterized. More specifically, an attempt is made to exploit a neural net as implementing neurons poses a great challenge for this dataset. To explain, a simple and fast Pearson coefficient linear model usually achieves good scores. An aggressive baseline that requires a really good model to overcome that is. Moreover, there are major limitations when it comes to causal discovery of observational data. Unlike the sachs one did not use interventions but only prior knowledge; the most prohibiting limitation is that of the data which is addressed. Thereafter, the method and the model are described and after that the results are presented. 

---
# OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework 

**Authors**: Ben Chen, Siyuan Wang, Yufei Ma, Zihan Liang, Xuxin Zhang, Yue Lv, Ying Yang, Huangyu Dai, Lingtao Mao, Tong Zhao, Zhipeng Qian, Xinyu Sun, Zhixin Zhai, Yang Zhao, Bochao Liu, Jingshan Lv, Xiao Liang, Hui Kong, Jing Chen, Han Li, Chenyi Lei, Wenwu Ou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2603.24422)  

**Abstract**: Generative Retrieval (GR) has emerged as a promising paradigm for modern search systems. Compared to multi-stage cascaded architecture, it offers advantages such as end-to-end joint optimization and high computational efficiency. OneSearch, as a representative industrial-scale deployed generative search framework, has brought significant commercial and operational benefits. However, its inadequate understanding of complex queries, inefficient exploitation of latent user intents, and overfitting to narrow historical preferences have limited its further performance improvement. To address these challenges, we propose \textbf{OneSearch-V2}, a latent reasoning enhanced self-distillation generative search framework. It contains three key innovations: (1) a thought-augmented complex query understanding module, which enables deep query understanding and overcomes the shallow semantic matching limitations of direct inference; (2) a reasoning-internalized self-distillation training pipeline, which uncovers users' potential yet precise e-commerce intentions beyond log-fitting through implicit in-context learning; (3) a behavior preference alignment optimization system, which mitigates reward hacking arising from the single conversion metric, and addresses personal preference via direct user feedback. Extensive offline evaluations demonstrate OneSearch-V2's strong query recognition and user profiling capabilities. Online A/B tests further validate its business effectiveness, yielding +3.98\% item CTR, +3.05\% buyer conversion rate, and +2.11\% order volume. Manual evaluation further confirms gains in search experience quality, with +1.65\% in page good rate and +1.37\% in query-item relevance. More importantly, OneSearch-V2 effectively mitigates common search system issues such as information bubbles and long-tail sparsity, without incurring additional inference costs or serving latency. 

---
# ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers 

**Authors**: Songyang Liu, Chaozhuo Li, Chenxu Wang, Jinyu Hou, Zejian Chen, Litian Zhang, Zheng Liu, Qiwei Ye, Yiming Hei, Xi Zhang, Zhongyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24414)  

**Abstract**: OpenClaw has rapidly established itself as a leading open-source autonomous agent runtime, offering powerful capabilities including tool integration, local file access, and shell command execution. However, these broad operational privileges introduce critical security vulnerabilities, transforming model errors into tangible system-level threats such as sensitive data leakage, privilege escalation, and malicious third-party skill execution. Existing security measures for the OpenClaw ecosystem remain highly fragmented, addressing only isolated stages of the agent lifecycle rather than providing holistic protection. To bridge this gap, we present ClawKeeper, a real-time security framework that integrates multi-dimensional protection mechanisms across three complementary architectural layers. (1) \textbf{Skill-based protection} operates at the instruction level, injecting structured security policies directly into the agent context to enforce environment-specific constraints and cross-platform boundaries. (2) \textbf{Plugin-based protection} serves as an internal runtime enforcer, providing configuration hardening, proactive threat detection, and continuous behavioral monitoring throughout the execution pipeline. (3) \textbf{Watcher-based protection} introduces a novel, decoupled system-level security middleware that continuously verifies agent state evolution. It enables real-time execution intervention without coupling to the agent's internal logic, supporting operations such as halting high-risk actions or enforcing human confirmation. We argue that this Watcher paradigm holds strong potential to serve as a foundational building block for securing next-generation autonomous agent systems. Extensive qualitative and quantitative evaluations demonstrate the effectiveness and robustness of ClawKeeper across diverse threat scenarios. We release our code. 

---
# Real Talk, Virtual Faces: A Formal Concept Analysis of Personality and Sentiment in Influencer Audiences 

**Authors**: Shahram Chaudhry, Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2603.24410)  

**Abstract**: Virtual influencers~(VIs) -- digitally synthetic social-media personas -- attract audiences whose discourse appears qualitatively different from discourse around human influencers~(HIs). Existing work characterises this difference through surveys or aggregate engagement statistics, which reveal \emph{what} audiences say but not \emph{how} multiple signals co-occur. We propose a two-layer, structure-first framework grounded in Formal Concept Analysis~(FCA) and association rule mining. The first layer applies FCA with support-based iceberg filtering to weekly-aggregated comment data, extracting discourse profiles -- weekly co-occurrence bundles of sentiment, Big Five personality cues, and topic tags. The second layer mines association rules at the comment level, revealing personality--sentiment--topic dependencies invisible to frequency-table analysis.
Applied to YouTube comments from three VI--HI influencer pairs, the two-layer analysis reveals a consistent structural divergence: HI discourse concentrates into a single, emotionally regulated (stability-centred) regime (low neuroticism anchoring positivity), while VI discourse supports three structurally distinct discourse modes, including an appearance-discourse cluster absent from HI despite near-equal marginal prevalence. Topic-specific analyses further show that VI contexts exhibit negative sentiment in psychologically sensitive domains (mental health, body image, artificial identity) relative to HI contexts. Our results position FCA as a principled tool for multi-signal discourse analysis and demonstrate that virtuality reshapes not just what audiences say, but the underlying grammar of how signals co-occur in their reactions. 

---
# Exploring How Fair Model Representations Relate to Fair Recommendations 

**Authors**: Bjørnar Vassøy, Benjamin Kille, Helge Langseth  

**Link**: [PDF](https://arxiv.org/pdf/2603.24396)  

**Abstract**: One of the many fairness definitions pursued in recent recommender system research targets mitigating demographic information encoded in model representations. Models optimized for this definition are typically evaluated on how well demographic attributes can be classified given model representations, with the (implicit) assumption that this measure accurately reflects \textit{recommendation parity}, i.e., how similar recommendations given to different users are. We challenge this assumption by comparing the amount of demographic information encoded in representations with various measures of how the recommendations differ. We propose two new approaches for measuring how well demographic information can be classified given ranked recommendations. Our results from extensive testing of multiple models on one real and multiple synthetically generated datasets indicate that optimizing for fair representations positively affects recommendation parity, but also that evaluation at the representation level is not a good proxy for measuring this effect when comparing models. We also provide extensive insight into how recommendation-level fairness metrics behave for various models by evaluating their performances on numerous generated datasets with different properties. 

---
# When AI Meets Early Childhood Education: Large Language Models as Assessment Teammates in Chinese Preschools 

**Authors**: Xingming Li, Runke Huang, Yanan Bao, Yuye Jin, Yuru Jiao, Qingyong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24389)  

**Abstract**: High-quality teacher-child interaction (TCI) is fundamental to early childhood development, yet traditional expert-based assessment faces a critical scalability challenge. In large systems like China's-serving 36 million children across 250,000+ kindergartens-the cost and time requirements of manual observation make continuous quality monitoring infeasible, relegating assessment to infrequent episodic audits that limit timely intervention and improvement tracking.
In this paper, we investigate whether AI can serve as a scalable assessment teammate by extracting structured quality indicators and validating their alignment with human expert judgments.
Our contributions include: (1) TEPE-TCI-370h (Tracing Effective Preschool Education), the first large-scale dataset of naturalistic teacher-child interactions in Chinese preschools (370 hours, 105 classrooms) with standardized ECQRS-EC and SSTEW annotations; (2) We develop Interaction2Eval, a specialized LLM-based framework addressing domain-specific challenges-child speech recognition, Mandarin homophone disambiguation, and rubric-based reasoning-achieving up to 88% agreement; (3) Deployment validation across 43 classrooms demonstrating an 18x efficiency gain in the assessment workflow, highlighting its potential for shifting from annual expert audits to monthly AI-assisted monitoring with targeted human oversight. This work not only demonstrates the technical feasibility of scalable, AI-augmented quality assessment but also lays the foundation for a new paradigm in early childhood education-one where continuous, inclusive, AI-assisted evaluation becomes the engine of systemic improvement and equitable growth. 

---
# MolEvolve: LLM-Guided Evolutionary Search for Interpretable Molecular Optimization 

**Authors**: Xiangsen Chen, Ruilong Wu, Yanyan Lan, Ting Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24382)  

**Abstract**: Despite deep learning's success in chemistry, its impact is hindered by a lack of interpretability and an inability to resolve activity cliffs, where minor structural nuances trigger drastic property shifts. Current representation learning, bound by the similarity principle, often fails to capture these structural-activity discontinuities. To address this, we introduce MolEvolve, an evolutionary framework that reformulates molecular discovery as an autonomous, look-ahead planning problem. Unlike traditional methods that depend on human-engineered features or rigid prior knowledge, MolEvolve leverages a Large Language Model (LLM) to actively explore and evolve a library of executable chemical symbolic operations. By utilizing the LLM to cold start and an Monte Carlo Tree Search (MCTS) engine for test-time planning with external tools (e.g. RDKit), the system self-discovers optimal trajectories autonomously. This process evolves transparent reasoning chains that translate complex structural transformations into actionable, human-readable chemical insights. Experimental results demonstrate that MolEvolve's autonomous search not only evolves transparent, human-readable chemical insights, but also outperforms baselines in both property prediction and molecule optimization tasks. 

---
# Language-Guided Structure-Aware Network for Camouflaged Object Detection 

**Authors**: Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24355)  

**Abstract**: Camouflaged Object Detection (COD) aims to segment objects that are highly integrated with the background in terms of color, texture, and structure, making it a highly challenging task in computer vision. Although existing methods introduce multi-scale fusion and attention mechanisms to alleviate the above issues, they generally lack the guidance of textual semantic priors, which limits the model's ability to focus on camouflaged regions in complex scenes. To address this issue, this paper proposes a Language-Guided Structure-Aware Network (LGSAN). Specifically, based on the visual backbone PVT-v2, we introduce CLIP to generate masks from text prompts and RGB images, thereby guiding the multi-scale features extracted by PVT-v2 to focus on potential target regions. On this foundation, we further design a Fourier Edge Enhancement Module (FEEM), which integrates multi-scale features with high-frequency information in the frequency domain to extract edge enhancement features. Furthermore, we propose a Structure-Aware Attention Module (SAAM) to effectively enhance the model's perception of object structures and boundaries. Finally, we introduce a Coarse-Guided Local Refinement Module (CGLRM) to enhance fine-grained reconstruction and boundary integrity of camouflaged object regions. Extensive experiments demonstrate that our method consistently achieves highly competitive performance across multiple COD datasets, validating its effectiveness and robustness. 

---
# Evidence of an Emergent "Self" in Continual Robot Learning 

**Authors**: Adidev Jhunjhunwala, Judah Goldfeder, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2603.24350)  

**Abstract**: A key challenge to understanding self-awareness has been a principled way of quantifying whether an intelligent system has a concept of a "self," and if so how to differentiate the "self" from other cognitive structures. We propose that the "self" can be isolated by seeking the invariant portion of cognitive process that changes relatively little compared to more rapidly acquired cognitive knowledge and skills, because our self is the most persistent aspect of our experiences. We used this principle to analyze the cognitive structure of robots under two conditions: One robot learns a constant task, while a second robot is subjected to continual learning under variable tasks. We find that robots subjected to continual learning develop an invariant subnetwork that is significantly more stable (p < 0.001) compared to the control. We suggest that this principle can offer a window into exploring selfhood in other cognitive AI systems. 

---
# Enhancing Efficiency and Performance in Deepfake Audio Detection through Neuron-level dropin & Neuroplasticity Mechanisms 

**Authors**: Yupei Li, Shuaijie Shao, Manuel Milling, Björn Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2603.24343)  

**Abstract**: Current audio deepfake detection has achieved remarkable performance using diverse deep learning architectures such as ResNet, and has seen further improvements with the introduction of large models (LMs) like Wav2Vec. The success of large language models (LLMs) further demonstrates the benefits of scaling model parameters, but also highlights one bottleneck where performance gains are constrained by parameter counts. Simply stacking additional layers, as done in current LLMs, is computationally expensive and requires full retraining. Furthermore, existing low-rank adaptation methods are primarily applied to attention-based architectures, which limits their scope. Inspired by the neuronal plasticity observed in mammalian brains, we propose novel algorithms, dropin and further plasticity, that dynamically adjust the number of neurons in certain layers to flexibly modulate model parameters. We evaluate these algorithms on multiple architectures, including ResNet, Gated Recurrent Neural Networks, and Wav2Vec. Experimental results using the widely recognised ASVSpoof2019 LA, PA, and FakeorReal dataset demonstrate consistent improvements in computational efficiency with the dropin approach and a maximum of around 39% and 66% relative reduction in Equal Error Rate with the dropin and plasticity approach among these dataset, respectively. The code and supplementary material are available at Github link. 

---
# GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents 

**Authors**: Yunzhe Wang, Runhui Xu, Kexin Zheng, Tianyi Zhang, Jayavibhav Niranjan Kogundi, Soham Hans, Volkan Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2603.24329)  

**Abstract**: Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling. 

---
# Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing 

**Authors**: Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, Yue Zhang, Yubo Zhang, Jing Zhang, Jun Zhang, Xing Wei, Yi Liu, Dianhai Yu, Yanjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.24326)  

**Abstract**: Document parsing is a fine-grained task where image resolution significantly impacts performance. While advanced research leveraging vision-language models benefits from high-resolution input to boost model performance, this often leads to a quadratic increase in the number of vision tokens and significantly raises computational costs. We attribute this inefficiency to substantial visual regions redundancy in document images, like background. To tackle this, we propose PaddleOCR-VL, a novel coarse-to-fine architecture that focuses on semantically relevant regions while suppressing redundant ones, thereby improving both efficiency and performance. Specifically, we introduce a lightweight Valid Region Focus Module (VRFM) which leverages localization and contextual relationship prediction capabilities to identify valid vision tokens. Subsequently, we design and train a compact yet powerful 0.9B vision-language model (PaddleOCR-VL-0.9B) to perform detailed recognition, guided by VRFM outputs to avoid direct processing of the entire large image. Extensive experiments demonstrate that PaddleOCR-VL achieves state-of-the-art performance in both page-level parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference while utilizing substantially fewer vision tokens and parameters, highlighting the effectiveness of targeted coarse-to-fine parsing for accurate and efficient document understanding. The source code and models are publicly available at this https URL. 

---
# Large Language Model Guided Incentive Aware Reward Design for Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Dogan Urgun, Gokhan Gungor  

**Link**: [PDF](https://arxiv.org/pdf/2603.24324)  

**Abstract**: Designing effective auxiliary rewards for cooperative multi-agent systems remains a precarious task; misaligned incentives risk inducing suboptimal coordination, especially where sparse task feedback fails to provide sufficient grounding. This study introduces an automated reward design framework that leverages large language models to synthesize executable reward programs from environment instrumentation. The procedure constrains candidate programs within a formal validity envelope and evaluates their efficacy by training policies from scratch under a fixed computational budget; selection depends exclusively on the sparse task return. The framework is evaluated across four distinct Overcooked-AI layouts characterized by varied corridor congestion, handoff dependencies, and structural asymmetries. Iterative search generations consistently yield superior task returns and delivery counts, with the most pronounced gains occurring in environments dominated by interaction bottlenecks. Diagnostic analysis of the synthesized shaping components indicates increased interdependence in action selection and improved signal alignment in coordination-intensive tasks. These results demonstrate that the search for objectivegrounded reward programs can mitigate the burden of manual engineering while producing shaping signals compatible with cooperative learning under finite budgets. 

---
# Toward Generalist Neural Motion Planners for Robotic Manipulators: Challenges and Opportunities 

**Authors**: Davood Soleymanzadeh, Ivan Lopez-Sanchez, Hao Su, Yunzhu Li, Xiao Liang, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.24318)  

**Abstract**: State-of-the-art generalist manipulation policies have enabled the deployment of robotic manipulators in unstructured human environments. However, these frameworks struggle in cluttered environments primarily because they utilize auxiliary modules for low-level motion planning and control. Motion planning remains challenging due to the high dimensionality of the robot's configuration space and the presence of workspace obstacles. Neural motion planners have enhanced motion planning efficiency by offering fast inference and effectively handling the inherent multi-modality of the motion planning problem. Despite such benefits, current neural motion planners often struggle to generalize to unseen, out-of-distribution planning settings. This paper reviews and analyzes the state-of-the-art neural motion planners, highlighting both their benefits and limitations. It also outlines a path toward establishing generalist neural motion planners capable of handling domain-specific challenges. For a list of the reviewed papers, please refer to this https URL. 

---
# Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs: When Does Per-Edge Routing Help? 

**Authors**: Eyal Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2603.24291)  

**Abstract**: Recent work distinguishes two heterophily regimes: adversarial, where cross-class edges dilute class signal and harm classification, and informative, where the heterophilous structure itself carries useful signal. We ask: when does per-edge message routing help, and when is a uniform spectral channel sufficient? To operationalize this question we introduce Cost-Sensitive Neighborhood Aggregation (CSNA), a GNN layer that computes pairwise distance in a learned projection and uses it to soft-route each message through concordant and discordant channels with independent transformations. Under a contextual stochastic block model we show that cost-sensitive weighting preserves class-discriminative signal where mean aggregation provably attenuates it, provided $w_+/w_- > q/p$. On six benchmarks with uniform tuning, CSNA is competitive with state-of-the-art methods on adversarial-heterophily datasets (Texas, Wisconsin, Cornell, Actor) but underperforms on informative-heterophily datasets (Chameleon, Squirrel) -- precisely the regime where per-edge routing has no useful decomposition to exploit. The pattern is itself the finding: the cost function's ability to separate edge types serves as a diagnostic for the heterophily regime, revealing when fine-grained routing adds value over uniform channels and when it does not. Code is available at this https URL . 

---
# The Specification Gap: Coordination Failure Under Partial Knowledge in Code Agents 

**Authors**: Camilo Chacón Sartori  

**Link**: [PDF](https://arxiv.org/pdf/2603.24284)  

**Abstract**: When multiple LLM-based code agents independently implement parts of the same class, they must agree on shared internal representations, even when the specification leaves those choices implicit. We study this coordination problem across 51 class-generation tasks, progressively stripping specification detail from full docstrings (L0) to bare signatures (L3), and introducing opposing structural biases (lists vs. dictionaries) to stress-test integration. Three findings emerge. First, a persistent specification gap: two-agent integration accuracy drops from 58% to 25% as detail is removed, while a single-agent baseline degrades more gracefully (89% to 56%), leaving a 25--39 pp coordination gap that is consistent across two Claude models (Sonnet, Haiku) and three independent runs. Second, an AST-based conflict detector achieves 97% precision at the weakest specification level without additional LLM calls, yet a factorial recovery experiment shows that restoring the full specification alone recovers the single-agent ceiling (89%), while providing conflict reports adds no measurable benefit. Third, decomposing the gap into coordination cost (+16 pp) and information asymmetry (+11 pp) suggests that the two effects are independent and approximately additive. The gap is not merely a consequence of hidden information, but reflects the difficulty of producing compatible code without shared decisions. These results support a specification-first view of multi-agent code generation: richer specifications are both the primary coordination mechanism and the sufficient recovery instrument. 

---
# Bridging Biological Hearing and Neuromorphic Computing: End-to-End Time-Domain Audio Signal Processing with Reservoir Computing 

**Authors**: Rinku Sebastian, Simon O'Keefe, Martin Trefzer  

**Link**: [PDF](https://arxiv.org/pdf/2603.24283)  

**Abstract**: Despite the advancements in cutting-edge technologies, audio signal processing continues to pose challenges and lacks the precision of a human speech processing system. To address these challenges, we propose a novel approach to simplify audio signal processing by leveraging time-domain techniques and reservoir computing. Through our research, we have developed a real-time audio signal processing system by simplifying audio signal processing through the utilization of reservoir computers, which are significantly easier to train.
Feature extraction is a fundamental step in speech signal processing, with Mel Frequency Cepstral Coefficients (MFCCs) being a dominant choice due to their perceptual relevance to human hearing. However, conventional MFCC extraction relies on computationally intensive time-frequency transformations, limiting efficiency in real-time applications. To address this, we propose a novel approach that leverages reservoir computing to streamline MFCC extraction. By replacing traditional frequency-domain conversions with convolution operations, we eliminate the need for complex transformations while maintaining feature discriminability. We present an end-to-end audio processing framework that integrates this method, demonstrating its potential for efficient and real-time speech analysis. Our results contribute to the advancement of energy-efficient audio processing technologies, enabling seamless deployment in embedded systems and voice-driven applications. This work bridges the gap between biologically inspired feature extraction and modern neuromorphic computing, offering a scalable solution for next-generation speech recognition systems. 

---
# Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timestep 

**Authors**: Tianyi Liu, Ye Lu, Linfeng Zhang, Chen Cai, Jianjun Gao, Yi Wang, Kim-Hui Yap, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2603.24260)  

**Abstract**: Diffusion-based video editing has emerged as an important paradigm for high-quality and flexible content generation. However, despite their generality and strong modeling capacity, Diffusion Transformers (DiT) remain computationally expensive due to the iterative denoising process, posing challenges for practical deployment. Existing video diffusion acceleration methods primarily exploit denoising timestep-level feature reuse, which mitigates the redundancy in denoising process, but overlooks the architectural redundancy within the DiT that many attention operations over spatio-temporal tokens are redundantly executed, offering little to no incremental contribution to the model output. This work introduces HetCache, a training-free diffusion acceleration framework designed to exploit the inherent heterogeneity in diffusion-based masked video-to-video (MV2V) generation and editing. Instead of uniformly reuse or randomly sampling tokens, HetCache assesses the contextual relevance and interaction strength among various types of tokens in designated computing steps. Guided by spatial priors, it divides the spatial-temporal tokens in DiT model into context and generative tokens, and selectively caches the context tokens that exhibit the strongest correlation and most representative semantics with generative ones. This strategy reduces redundant attention operations while maintaining editing consistency and fidelity. Experiments show that HetCache achieves a noticeable acceleration, including a 2.67$\times$ latency speedup and FLOPs reduction over commonly used foundation models, with negligible degradation in editing quality. 

---
# Embracing Heteroscedasticity for Probabilistic Time Series Forecasting 

**Authors**: Yijun Wang, Qiyuan Zhuang, Xiu-Shen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.24254)  

**Abstract**: Probabilistic time series forecasting (PTSF) aims to model the full predictive distribution of future observations, enabling both accurate forecasting and principled uncertainty quantification. A central requirement of PTSF is to embrace heteroscedasticity, as real-world time series exhibit time-varying conditional variances induced by nonstationary dynamics, regime changes, and evolving external conditions. However, most existing non-autoregressive generative approaches to PTSF, such as TimeVAE and $K^2$VAE, rely on MSE-based training objectives that implicitly impose a homoscedastic assumption, thereby fundamentally limiting their ability to model temporal heteroscedasticity. To address this limitation, we propose the Location-Scale Gaussian VAE (LSG-VAE), a simple but effective framework that explicitly parameterizes both the predictive mean and time-dependent variance through a location-scale likelihood formulation. This design enables LSG-VAE to faithfully capture heteroscedastic aleatoric uncertainty and introduces an adaptive attenuation mechanism that automatically down-weights highly volatile observations during training, leading to improved robustness in trend prediction. Extensive experiments on nine benchmark datasets demonstrate that LSG-VAE consistently outperforms fifteen strong generative baselines while maintaining high computational efficiency suitable for real-time deployment. 

---
# DVM: Real-Time Kernel Generation for Dynamic AI Models 

**Authors**: Jingzhi Fang, Xiong Gao, Renwei Zhang, Zichun Ye, Lei Chen, Jie Zhao, Chengnuo Huang, Hui Xu, Xuefeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.24239)  

**Abstract**: Dynamism is common in AI computation, e.g., the dynamic tensor shapes and the dynamic control flows in models. Due to the long compilation time, existing runtime compilation damages the model efficiency, while the offline compilers either suffer from the long compilation time and device memory footprint to cover all the possible execution instances of a dynamic model, or sacrifice optimization opportunities for usability. In this paper, we rethink the feasibility of runtime compilation for dynamic models and identify that the key for it to work is to speed up the compilation or hide the compilation overhead. To do this, we propose a real-time compiler, DVM. In DVM, we design a runtime operator compiler based on a bytecode virtual machine to perform effective and efficient compilation for each dynamic operator instance given its input. Specifically, instead of compiling programs into machine code, we encode the operator program into bytecode on the CPU and decode the bytecode into virtual instructions for direct execution on the NPU. Based on the runtime operator compiler, we further propose an operator fuser, which performs symbol-deduction-based fusion on static graphs and runtime fusion on dynamic graphs. Both pattern- and stacking-based fusion are supported to increase fusion opportunities. Evaluation on operators, subgraphs, and models shows that, compared with TorchInductor, PyTorch-eager and MindSpore-graph-O0, we are up to 11.77$\times$ better in terms of the operator/model efficiency and up to 5 orders of magnitude faster in terms of the maximum compilation time. 

---
# Environment-Grounded Multi-Agent Workflow for Autonomous Penetration Testing 

**Authors**: Michael Somma, Markus Großpointner, Paul Zabalegui, Eppu Heilimo, Branka Stojanović  

**Link**: [PDF](https://arxiv.org/pdf/2603.24221)  

**Abstract**: The increasing complexity and interconnectivity of digital infrastructures make scalable and reliable security assessment methods essential. Robotic systems represent a particularly important class of operational technology, as modern robots are highly networked cyber-physical systems deployed in domains such as industrial automation, logistics, and autonomous services. This paper explores the use of large language models for automated penetration testing in robotic environments. We propose an environment-grounded multi-agent architecture tailored to Robotics-based systems. The approach dynamically constructs a shared graph-based memory during execution that captures the observable system state, including network topology, communication channels, vulnerabilities, and attempted exploits. This enables structured automation while maintaining traceability and effective context management throughout the testing process. Evaluated across multiple iterations within a specialized robotics Capture-the-Flag scenario (ROS/ROS2), the system demonstrated high reliability, successfully completing the challenge in 100\% of test runs (n=5). This performance significantly exceeds literature benchmarks while maintaining the traceability and human oversight required by frameworks like the EU AI Act. 

---
# Who Benefits from RAG? The Role of Exposure, Utility and Attribution Bias 

**Authors**: Mahdi Dehghan, Graham McDonald  

**Link**: [PDF](https://arxiv.org/pdf/2603.24218)  

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have achieved substantial improvements in accuracy by grounding their responses in external documents that are relevant to the user's query. However, relatively little work has investigated the impact of RAG in terms of fairness. Particularly, it is not yet known if queries that are associated with certain groups within a fairness category systematically receive higher accuracy, or accuracy improvements in RAG systems compared to LLM-only, a phenomenon we refer to as query group fairness. In this work, we conduct extensive experiments to investigate the impact of three key factors on query group fairness in RAG, namely: Group exposure, i.e., the proportion of documents from each group appearing in the retrieved set, determined by the retriever; Group utility, i.e., the degree to which documents from each group contribute to improving answer accuracy, capturing retriever-generator interactions; and Group attribution, i.e., the extent to which the generator relies on documents from each group when producing responses. We examine group-level average accuracy and accuracy improvements disparities across four fairness categories using three datasets derived from the TREC 2022 Fair Ranking Track for two tasks: article generation and title generation. Our findings show that RAG systems suffer from the query group fairness problem and amplify disparities in terms of average accuracy across queries from different groups, compared to an LLM-only setting. Moreover, group utility, exposure, and attribution can exhibit strong positive or negative correlations with average accuracy or accuracy improvements of queries from that group, highlighting their important role in fair RAG. Our data and code are publicly available from Github. 

---
# Where Do Your Citations Come From? Citation-Constellation: A Free, Open-Source, No-Code, and Auditable Tool for Citation Network Decomposition with Complementary BARON and HEROCON Scores 

**Authors**: Mahbub Ul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2603.24216)  

**Abstract**: Standard citation metrics treat all citations as equal, obscuring the social and structural pathways through which scholarly influence propagates. I introduce Citation-Constellation, a freely available no-code tool for citation network analysis with two complementary bibliometric scores that decompose a researcher's citation profile by network proximity between citing and cited authors. BARON (Boundary-Anchored Research Outreach Network score) is a strict binary metric counting only citations from outside the detected collaborative network. HEROCON (Holistic Equilibrated Research Outreach CONstellation score) applies graduated weights assigning partial credit to in-group citations based on relationship proximity. The gap between scores serves as a diagnostic of inner-circle dependence. An extended abstract with full details appears in the paper.
The tool implements this through a phased architecture: (1) self-citation analysis, (2) co-authorship graph traversal, (3) temporal institutional affiliation matching via ROR, and (4) AI-agent-driven venue governance extraction using a local LLM. Phases 1-3 are fully operational; Phase 4 is under development. Key design choices include ORCID-validated author identity resolution, an UNKNOWN classification for citations with insufficient metadata, and comprehensive audit trails documenting every classification decision. A no-code web interface enables researchers to compute scores without programming, installation, or registration.
I present these scores as structural diagnostics, not quality indicators. BARON and HEROCON describe where in the social graph citations originate. They should not be used for hiring, promotion, or funding decisions. HEROCON weights are experimental and require empirical calibration. 

---
# Uncovering Memorization in Timeseries Imputation models: LBRM Membership Inference and its link to attribute Leakage 

**Authors**: Faiz Taleb, Ivan Gazeau, Maryline Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2603.24213)  

**Abstract**: Deep learning models for time series imputation are now essential in fields such as healthcare, the Internet of Things (IoT), and finance. However, their deployment raises critical privacy concerns. Beyond the well-known issue of unintended memorization, which has been extensively studied in generative models, we demonstrate that time series models are vulnerable to inference attacks in a black-box setting. In this work, we introduce a two-stage attack framework comprising: (1) a novel membership inference attack based on a reference model that improves detection accuracy, even for models robust to overfitting-based attacks, and (2) the first attribute inference attack that predicts sensitive characteristics of the training data for timeseries imputation model. We evaluate these attacks on attention-based and autoencoder architectures in two scenarios: models that are trained from scratch, and fine-tuned models where the adversary has access to the initial weights. Our experimental results demonstrate that the proposed membership attack retrieves a significant portion of the training data with a tpr@top25% score significantly higher than a naive attack baseline. We show that our membership attack also provides a good insight of whether attribute inference will work (with a precision of 90% instead of 78% in the genral case). 

---
# Powerful Teachers Matter: Text-Guided Multi-view Knowledge Distillation with Visual Prior Enhancement 

**Authors**: Xin Zhang, Jianyang Xu, Hao Peng, Dongjing Wang, Jingyuan Zheng, Yu Li, Yuyu Yin, Hongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24208)  

**Abstract**: Knowledge distillation transfers knowledge from large teacher models to smaller students for efficient inference. While existing methods primarily focus on distillation strategies, they often overlook the importance of enhancing teacher knowledge quality. In this paper, we propose Text-guided Multi-view Knowledge Distillation (TMKD), which leverages dual-modality teachers, a visual teacher and a text teacher (CLIP), to provide richer supervisory signals. Specifically, we enhance the visual teacher with multi-view inputs incorporating visual priors (edge and high-frequency features), while the text teacher generates semantic weights through prior-aware prompts to guide adaptive feature fusion. Additionally, we introduce vision-language contrastive regularization to strengthen semantic knowledge in the student model. Extensive experiments on five benchmarks demonstrate that TMKD consistently improves knowledge distillation performance by up to 4.49\%, validating the effectiveness of our dual-teacher multi-view enhancement strategy. Code is available at this https URL. 

---
# Invisible Threats from Model Context Protocol: Generating Stealthy Injection Payload via Tree-based Adaptive Search 

**Authors**: Yulin Shen, Xudong Pan, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24203)  

**Abstract**: Recent advances in the Model Context Protocol (MCP) have enabled large language models (LLMs) to invoke external tools with unprecedented ease. This creates a new class of powerful and tool augmented agents. Unfortunately, this capability also introduces an under explored attack surface, specifically the malicious manipulation of tool responses. Existing techniques for indirect prompt injection that target MCP suffer from high deployment costs, weak semantic coherence, or heavy white box requirements. Furthermore, they are often easily detected by recently proposed defenses. In this paper, we propose Tree structured Injection for Payloads (TIP), a novel black-box attack which generates natural payloads to reliably seize control of MCP enabled agents even under defense. Technically, We cast payload generation as a tree structured search problem and guide the search with an attacker LLM operating under our proposed coarse-to-fine optimization framework. To stabilize learning and avoid local optima, we introduce a path-aware feedback mechanism that surfaces only high quality historical trajectories to the attacker model. The framework is further hardened against defensive transformations by explicitly conditioning the search on observable defense signals and dynamically reallocating the exploration budget. Extensive experiments on four mainstream LLMs show that TIP attains over 95% attack success in undefended settings while requiring an order of magnitude fewer queries than prior adaptive attacks. Against four representative defense approaches, TIP preserves more than 50% effectiveness and significantly outperforms the state-of-the-art attacks. By implementing the attack on real world MCP systems, our results expose an invisible but practical threat vector in MCP deployments. We also discuss potential mitigation approaches to address this critical security gap. 

---
# A Deep Dive into Scaling RL for Code Generation with Synthetic Data and Curricula 

**Authors**: Cansu Sancaktar, David Zhang, Gabriel Synnaeve, Taco Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24202)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful paradigm for improving large language models beyond supervised fine-tuning, yet sustaining performance gains at scale remains an open challenge, as data diversity and structure, rather than volume alone, become the limiting factor. We address this by introducing a scalable multi-turn synthetic data generation pipeline in which a teacher model iteratively refines problems based on in-context student performance summaries, producing structured difficulty progressions without any teacher fine-tuning. Compared to single-turn generation, this multi-turn approach substantially improves the yield of valid synthetic problems and naturally produces stepping stones, i.e. easier and harder variants of the same core task, that support curriculum-based training. We systematically study how task difficulty, curriculum scheduling, and environment diversity interact during RL training across the Llama3.1-8B Instruct and Qwen3-8B Base model families, with additional scaling experiments on Qwen2.5-32B. Our results show that synthetic augmentation consistently improves in-domain code and in most cases out-of-domain math performance, and we provide empirical insights into how curriculum design and data diversity jointly shape RL training dynamics. 

---
# MedAidDialog: A Multilingual Multi-Turn Medical Dialogue Dataset for Accessible Healthcare 

**Authors**: Shubham Kumar Nigam, Suparnojit Sarkar, Piyush Patel  

**Link**: [PDF](https://arxiv.org/pdf/2603.24132)  

**Abstract**: Conversational artificial intelligence has the potential to assist users in preliminary medical consultations, particularly in settings where access to healthcare professionals is limited. However, many existing medical dialogue systems operate in a single-turn question--answering paradigm or rely on template-based datasets, limiting conversational realism and multilingual applicability. In this work, we introduce MedAidDialog, a multilingual multi-turn medical dialogue dataset designed to simulate realistic physician--patient consultations. The dataset extends the MDDial corpus by generating synthetic consultations using large language models and further expands them into a parallel multilingual corpus covering seven languages: English, Hindi, Telugu, Tamil, Bengali, Marathi, and Arabic.
Building on this dataset, we develop MedAidLM, a conversational medical model trained using parameter-efficient fine-tuning on quantized small language models, enabling deployment without high-end computational infrastructure. Our framework additionally incorporates optional patient pre-context information (e.g., age, gender, allergies) to personalize the consultation process. Experimental results demonstrate that the proposed system can effectively perform symptom elicitation through multi-turn dialogue and generate diagnostic recommendations. We further conduct medical expert evaluation to assess the plausibility and coherence of the generated consultations. 

---
# The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation 

**Authors**: Mingyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24124)  

**Abstract**: RLHF-aligned language models exhibit response homogenization: on TruthfulQA (n=790), 40-79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have zero discriminative power (AUROC=0.500), while free token entropy retains signal (0.603). This alignment tax is task-dependent: on GSM8K (n=500), token entropy achieves 0.724 (Cohen's d=0.81).
A base-vs-instruct ablation confirms the causal role of alignment: the base model shows 1.0% single-cluster rate vs. 28.5% for the instruct model (p < 10^{-6}). A training stage ablation (Base 0.0% -> SFT 1.5% -> DPO 4.0% SCR) localizes the cause to DPO, not SFT. Cross-family replication on four model families reveals alignment tax severity varies by family and scale. We validate across 22 experiments, 5 benchmarks, 4 model families, and 3 model scales (3B-14B), with Jaccard, embedding, and NLI-based baselines at three DeBERTa scales (all ~0.51 AUROC). Cross-embedder validation with two independent embedding families rules out coupling bias. Cross-dataset validation on WebQuestions (58.0% SCR) confirms generalization beyond TruthfulQA. The central finding -- response homogenization -- is implementation-independent and label-free. Motivated by this diagnosis, we explore a cheapest-first cascade (UCBD) over orthogonal uncertainty signals. Selective prediction raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage; weakly dependent boundaries (|r| <= 0.12) enable 57% cost savings. 

---
# Comparative analysis of dual-form networks for live land monitoring using multi-modal satellite image time series 

**Authors**: Iris Dumeur, Jérémy Anger, Gabriele Facciolo  

**Link**: [PDF](https://arxiv.org/pdf/2603.24109)  

**Abstract**: Multi-modal Satellite Image Time Series (SITS) analysis faces significant computational challenges for live land monitoring applications. While Transformer architectures excel at capturing temporal dependencies and fusing multi-modal data, their quadratic computational complexity and the need to reprocess entire sequences for each new acquisition limit their deployment for regular, large-area monitoring. This paper studies various dual-form attention mechanisms for efficient multi-modal SITS analysis, that enable parallel training while supporting recurrent inference for incremental processing. We compare linear attention and retention mechanisms within a multi-modal spectro-temporal encoder. To address SITS-specific challenges of temporal irregularity and unalignment, we develop temporal adaptations of dual-form mechanisms that compute token distances based on actual acquisition dates rather than sequence indices. Our approach is evaluated on two tasks using Sentinel-1 and Sentinel-2 data: multi-modal SITS forecasting as a proxy task, and real-world solar panel construction monitoring. Experimental results demonstrate that dual-form mechanisms achieve performance comparable to standard Transformers while enabling efficient recurrent inference. The multimodal framework consistently outperforms mono-modal approaches across both tasks, demonstrating the effectiveness of dual mechanisms for sensor fusion. The results presented in this work open new opportunities for operational land monitoring systems requiring regular updates over large geographic areas. 

---
# KCLNet: Electrically Equivalence-Oriented Graph Representation Learning for Analog Circuits 

**Authors**: Peng Xu, Yapeng Li, Tinghuan Chen, Tsung-Yi Ho, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24101)  

**Abstract**: Digital circuits representation learning has made remarkable progress in the electronic design automation domain, effectively supporting critical tasks such as testability analysis and logic reasoning. However, representation learning for analog circuits remains challenging due to their continuous electrical characteristics compared to the discrete states of digital circuits. This paper presents a direct current (DC) electrically equivalent-oriented analog representation learning framework, named \textbf{KCLNet}. It comprises an asynchronous graph neural network structure with electrically-simulated message passing and a representation learning method inspired by Kirchhoff's Current Law (KCL). This method maintains the orderliness of the circuit embedding space by enforcing the equality of the sum of outgoing and incoming current embeddings at each depth, which significantly enhances the generalization ability of circuit embeddings. KCLNet offers a novel and effective solution for analog circuit representation learning with electrical constraints preserved. Experimental results demonstrate that our method achieves significant performance in a variety of downstream tasks, e.g., analog circuit classification, subcircuit detection, and circuit edit distance prediction. 

---
# Towards Effective Experiential Learning: Dual Guidance for Utilization and Internalization 

**Authors**: Fei Bai, Zhipeng Chen, Chuan Hao, Ming Yang, Ran Tao, Bryan Dai, Wayne Xin Zhao, Jian Yang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24093)  

**Abstract**: Recently, reinforcement learning~(RL) has become an important approach for improving the capabilities of large language models~(LLMs). In particular, reinforcement learning from verifiable rewards~(RLVR) has emerged as a promising paradigm for reasoning tasks. However, existing RL-based training still remains only a rough approximation to human learning. Human learners leverage both external and internal experience to guide exploration and gradually internalize useful trajectories into stable knowledge. Motivated by this gap, we ask: how can LLMs better utilize and internalize experience during RLVR training? To answer this question, we propose \textbf{D}ual \textbf{G}uidance \textbf{O}ptimization~(\textbf{DGO}), a unified framework that leverages \emph{external} and \emph{internal experience} to improve training effectiveness. Specifically, DGO first constructs an experience bank from previously explored trajectories. The policy then performs exploration under the joint guidance of the experience bank and the model's internal knowledge. The resulting trajectories are further used to refine the experience bank and optimize model parameters, forming a closed loop of experience utilization and internalization. Experiments show that DGO consistently outperforms baseline methods, suggesting that better utilization and internalization of experience lead to more effective reasoning. 

---
# Knowledge-Guided Manipulation Using Multi-Task Reinforcement Learning 

**Authors**: Aditya Narendra, Mukhammadrizo Maribjonov, Dmitry Makarov, Dmitry Yudin, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2603.24083)  

**Abstract**: This paper introduces Knowledge Graph based Massively Multi-task Model-based Policy Optimization (KG-M3PO), a framework for multi-task robotic manipulation in partially observable settings that unifies Perception, Knowledge, and Policy. The method augments egocentric vision with an online 3D scene graph that grounds open-vocabulary detections into a metric, relational representation. A dynamic-relation mechanism updates spatial, containment, and affordance edges at every step, and a graph neural encoder is trained end-to-end through the RL objective so that relational features are shaped directly by control performance. Multiple observation modalities (visual, proprioceptive, linguistic, and graph-based) are encoded into a shared latent space, upon which the RL agent operates to drive the control loop. The policy conditions on lightweight graph queries alongside visual and proprioceptive inputs, yielding a compact, semantically informed state for decision making.
Experiments on a suite of manipulation tasks with occlusions, distractors, and layout shifts demonstrate consistent gains over strong baselines: the knowledge-conditioned agent achieves higher success rates, improved sample efficiency, and stronger generalization to novel objects and unseen scene configurations. These results support the premise that structured, continuously maintained world knowledge is a powerful inductive bias for scalable, generalizable manipulation: when the knowledge module participates in the RL computation graph, relational representations align with control, enabling robust long-horizon behavior under partial observability. 

---
# When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm 

**Authors**: Ye Leng, Junjie Chu, Mingjie Li, Chenhao Lin, Chao Shen, Michael Backes, Yun Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24079)  

**Abstract**: Recently, multimodal large language models (MLLMs) have emerged as a unified paradigm for language and image generation. Compared with diffusion models, MLLMs possess a much stronger capability for semantic understanding, enabling them to process more complex textual inputs and comprehend richer contextual meanings. However, this enhanced semantic ability may also introduce new and potentially greater safety risks. Taking diffusion models as a reference point, we systematically analyze and compare the safety risks of emerging MLLMs along two dimensions: unsafe content generation and fake image synthesis. Across multiple unsafe generation benchmark datasets, we observe that MLLMs tend to generate more unsafe images than diffusion models. This difference partly arises because diffusion models often fail to interpret abstract prompts, producing corrupted outputs, whereas MLLMs can comprehend these prompts and generate unsafe content. For current advanced fake image detectors, MLLM-generated images are also notably harder to identify. Even when detectors are retrained with MLLMs-specific data, they can still be bypassed by simply providing MLLMs with longer and more descriptive inputs. Our measurements indicate that the emerging safety risks of the cutting-edge generative paradigm, MLLMs, have not been sufficiently recognized, posing new challenges to real-world safety. 

---
# Mitigating Object Hallucinations in LVLMs via Attention Imbalance Rectification 

**Authors**: Han Sun, Qin Li, Peixin Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24058)  

**Abstract**: Object hallucination in Large Vision-Language Models (LVLMs) severely compromises their reliability in real-world applications, posing a critical barrier to their deployment in high-stakes scenarios such as autonomous driving and medical image analysis. Through systematic empirical investigation, we identify that the imbalanced attention allocation, both across modalities (i.e., vision and language) and within modalities (among individual tokens), exhibits a strong causal correlation with the occurrence of object hallucination. Leveraging this insight, we introduce a novel concept termed attention imbalance, which not only quantifies the degree of attention disparity but also visually delineates the underlying patterns (e.g., over-attentiveness to irrelevant language tokens or under-attentiveness to discriminative visual features) that drive object hallucination. To mitigate object hallucination, we further propose Attention Imbalance Rectification (AIR), a lightweight decoding-time intervention method that reallocates attention weights and adjusts attention distributions to rectify modality-wise and token-wise imbalances. Extensive evaluations on four mainstream LVLMs and three benchmarks (CHAIR, POPE, and MM-Vet) with seven baselines demonstrate that AIR consistently reduces object hallucination rates, achieving up to a 35.1% reduction compared to the baselines, while improving up to 15.9% of LVLMs' general capability across diverse vision-language tasks. 

---
# From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs 

**Authors**: Xiaoyong Guo, Nanjie Li, Zijie Zeng, Kai Wang, Hao Huang, Haihua Xu, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.24034)  

**Abstract**: Contextual automatic speech recognition (ASR) with Speech-LLMs is typically trained with oracle conversation history, but relies on error-prone history at inference, causing a train-test mismatch in the context channel that we term contextual exposure bias. We propose a unified training framework to improve robustness under realistic histories: (i) Teacher Error Knowledge by using Whisper large-v3 hypotheses as training-time history, (ii) Context Dropout to regularize over-reliance on history, and (iii) Direct Preference Optimization (DPO) on curated failure cases. Experiments on TED-LIUM 3 (in-domain) and zero-shot LibriSpeech (out-of-domain) show consistent gains under predicted-history decoding. With a two-utterance history as context, SFT with Whisper hypotheses reduce WER from 5.59% (oracle-history training) to 5.47%, and DPO further improves to 5.17%. Under irrelevant-context attacks, DPO yields the smallest degradation (5.17% -> 5.63%), indicating improved robustness to misleading context. Our code and models are published on this https URL. 

---
# Schema on the Inside: A Two-Phase Fine-Tuning Method for High-Efficiency Text-to-SQL at Scale 

**Authors**: Chinmay Soni, Shivam Chourasia, Gaurav Kumar, Hitesh Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2603.24023)  

**Abstract**: Applying large, proprietary API-based language models to text-to-SQL tasks poses a significant industry challenge: reliance on massive, schema-heavy prompts results in prohibitive per-token API costs and high latency, hindering scalable production deployment. We present a specialized, self-hosted 8B-parameter model designed for a conversational bot in CriQ, a sister app to Dream11, India's largest fantasy sports platform with over 250 million users, that answers user queries about cricket statistics. Our novel two-phase supervised fine-tuning approach enables the model to internalize the entire database schema, eliminating the need for long-context prompts. This reduces input tokens by over 99%, from a 17k-token baseline to fewer than 100, and replaces costly external API calls with efficient local inference. The resulting system achieves 98.4% execution success and 92.5% semantic accuracy, substantially outperforming a prompt-engineered baseline using Google's Gemini Flash 2.0 (95.6% execution, 89.4% semantic accuracy). These results demonstrate a practical path toward high-precision, low-latency text-to-SQL applications using domain-specialized, self-hosted language models in large-scale production environments. 

---
# Understanding the Challenges in Iterative Generative Optimization with LLMs 

**Authors**: Allen Nie, Xavier Daull, Zhiyi Kuang, Abhinav Akkiraju, Anish Chaudhuri, Max Piasevoli, Ryan Rong, YuCheng Yuan, Prerit Choudhary, Shannon Xiao, Rasool Fakoor, Adith Swaminathan, Ching-An Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.23994)  

**Abstract**: Generative optimization uses large language models (LLMs) to iteratively improve artifacts (such as code, workflows or prompts) using execution feedback. It is a promising approach to building self-improving agents, yet in practice remains brittle: despite active research, only 9% of surveyed agents used any automated optimization. We argue that this brittleness arises because, to set up a learning loop, an engineer must make ``hidden'' design choices: What can the optimizer edit and what is the "right" learning evidence to provide at each update? We investigate three factors that affect most applications: the starting artifact, the credit horizon for execution traces, and batching trials and errors into learning evidence. Through case studies in MLAgentBench, Atari, and BigBench Extra Hard, we find that these design decisions can determine whether generative optimization succeeds, yet they are rarely made explicit in prior work. Different starting artifacts determine which solutions are reachable in MLAgentBench, truncated traces can still improve Atari agents, and larger minibatches do not monotonically improve generalization on BBEH. We conclude that the lack of a simple, universal way to set up learning loops across domains is a major hurdle for productionization and adoption. We provide practical guidance for making these choices. 

---
# From Untamed Black Box to Interpretable Pedagogical Orchestration: The Ensemble of Specialized LLMs Architecture for Adaptive Tutoring 

**Authors**: Nizam Kadir  

**Link**: [PDF](https://arxiv.org/pdf/2603.23990)  

**Abstract**: Monolithic Large Language Models (LLMs) used in educational dialogue often behave as "black boxes," where pedagogical decisions are implicit and difficult to audit, frequently violating instructional constraints by providing answers too early. We introduce the Ensemble of Specialized LLMS (ES-LLMS) architecture that separates decision-making from wording. Pedagogical actions are selected by a deterministic rules-based orchestrator coordinating specialized agents covering tutoring, assessment, feedback, scaffolding, motivation and ethics-guided by an interpretable Bayesian Knowledge Tracing (BKT) student model. An LLM renderer surface-realizes the chosen action in natural language. This design emphasizes reliability and controllability: constraints such as "attempt-before-hint" and hint caps are enforced as explicit rules, and the system logs per-turn agent traces and constraint checks. Validation of pedagogical quality via human expert reviewers (N=6) and a multi-LLM-as-Judge panel (six state-of-the-art models) showed that ES-LLMs were preferred in 91.7% and 79.2% of cases, respectively. The architecture significantly outperformed monolithic baselines across all seven dimensions, particularly in Scaffolding & Guidance, and Trust & Explainability. Furthermore, a Monte Carlo simulation (N=2,400) exposed a "Mastery Gain Paradox," where monolithic tutors inflated short-term performance through over-assistance. In contrast, ES-LLMs achieved 100% adherence to pedagogical constraints (e.g., attempt-before-hint) and a 3.3x increase in hint efficiency. Operationally, ES-LLMs reduced costs by 54% and latency by 22% by utilizing stateless prompts. We conclude that structural decoupling is essential for transforming stochastic models into trustworthy, verifiable and resource-efficient pedagogical agents. 

---
# SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating 

**Authors**: Hanbyel Cho, Sang-Hun Kim, Jeonguk Kang, Donghan Koo  

**Link**: [PDF](https://arxiv.org/pdf/2603.23983)  

**Abstract**: Recent advances in real-time interactive text-driven motion generation have enabled humanoids to perform diverse behaviors. However, kinematics-only generators often exhibit physical hallucinations, producing motion trajectories that are physically infeasible to track with a downstream motion tracking controller or unsafe for real-world deployment. These failures often arise from the lack of explicit physics-aware objectives for real-robot execution and become more severe under out-of-distribution (OOD) user inputs. Hence, we propose SafeFlow, a text-driven humanoid whole-body control framework that combines physics-guided motion generation with a 3-Stage Safety Gate driven by explicit risk indicators. SafeFlow adopts a two-level architecture. At the high level, we generate motion trajectories using Physics-Guided Rectified Flow Matching in a VAE latent space to improve real-robot executability, and further accelerate sampling via Reflow to reduce the number of function evaluations (NFE) for real-time control. The 3-Stage Safety Gate enables selective execution by detecting semantic OOD prompts using a Mahalanobis score in text-embedding space, filtering unstable generations via a directional sensitivity discrepancy metric, and enforcing final hard kinematic constraints such as joint and velocity limits before passing the generated trajectory to a low-level motion tracking controller. Extensive experiments on the Unitree G1 demonstrate that SafeFlow outperforms prior diffusion-based methods in success rate, physical compliance, and inference speed, while maintaining diverse expressiveness. 

---
# Kirchhoff-Inspired Neural Networks for Evolving High-Order Perception 

**Authors**: Tongfei Chen, Jingying Yang, Linlin Yang, Jinhu Lü, David Doermann, Chunyu Xie, Long He, Tian Wang, Juan Zhang, Guodong Guo, Baochang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23977)  

**Abstract**: Deep learning architectures are fundamentally inspired by neuroscience, particularly the structure of the brain's sensory pathways, and have achieved remarkable success in learning informative data representations. Although these architectures mimic the communication mechanisms of biological neurons, their strategies for information encoding and transmission are fundamentally distinct. Biological systems depend on dynamic fluctuations in membrane potential; by contrast, conventional deep networks optimize weights and biases by adjusting the strengths of inter-neural connections, lacking a systematic mechanism to jointly characterize the interplay among signal intensity, coupling structure, and state evolution. To tackle this limitation, we propose the Kirchhoff-Inspired Neural Network (KINN), a state-variable-based network architecture constructed based on Kirchhoff's current law. KINN derives numerically stable state updates from fundamental ordinary differential equations, enabling the explicit decoupling and encoding of higher-order evolutionary components within a single layer while preserving physical consistency, interpretability, and end-to-end trainability. Extensive experiments on partial differential equation (PDE) solving and ImageNet image classification validate that KINN outperforms state-of-the-art existing methods. 

---
# The Price Reversal Phenomenon: When Cheaper Reasoning Models End Up Costing More 

**Authors**: Lingjiao Chen, Chi Zhang, Yeye He, Ion Stoica, Matei Zaharia, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.23971)  

**Abstract**: Developers and consumers increasingly choose reasoning language models (RLMs) based on their listed API prices. However, how accurately do these prices reflect actual inference costs? We conduct the first systematic study of this question, evaluating 8 frontier RLMs across 9 diverse tasks covering competition math, science QA, code generation, and multi-domain reasoning. We uncover the pricing reversal phenomenon: in 21.8% of model-pair comparisons, the model with a lower listed price actually incurs a higher total cost, with reversal magnitude reaching up to 28x. For example, Gemini 3 Flash's listed price is 78% cheaper than GPT-5.2's, yet its actual cost across all tasks is 22% higher. We trace the root cause to vast heterogeneity in thinking token consumption: on the same query, one model may use 900% more thinking tokens than another. In fact, removing thinking token costs reduces ranking reversals by 70% and raises the rank correlation (Kendall's $\tau$ ) between price and cost rankings from 0.563 to 0.873. We further show that per-query cost prediction is fundamentally difficult: repeated runs of the same query yield thinking token variation up to 9.7x, establishing an irreducible noise floor for any predictor. Our findings demonstrate that listed API pricing is an unreliable proxy for actual cost, calling for cost-aware model selection and transparent per-request cost monitoring. 

---
# Policy-Guided Threat Hunting: An LLM enabled Framework with Splunk SOC Triage 

**Authors**: Rishikesh Sahay, Bell Eapen, Weizhi Meng, Md Rasel Al Mamun, Nikhil Kumar Dora, Manjusha Sumasadan, Sumit Kumar Tetarave, Rod Soto  

**Link**: [PDF](https://arxiv.org/pdf/2603.23966)  

**Abstract**: With frequently evolving Advanced Persistent Threats (APTs) in cyberspace, traditional security solutions approaches have become inadequate for threat hunting for organizations. Moreover, SOC (Security Operation Centers) analysts are often overwhelmed and struggle to analyze the huge volume of logs received from diverse devices in organizations. To address these challenges, we propose an automated and dynamic threat hunting framework for monitoring evolving threats, adapting to changing network conditions, and performing risk-based prioritization for the mitigation of suspicious and malicious traffic. By integrating Agentic AI with Splunk, an established SIEM platform, we developed a unique threat hunting framework. The framework systematically and seamlessly integrates different threat hunting modules together, ranging from traffic ingestion to anomaly assessment using a reconstruction-based autoencoder, deep reinforcement learning (DRL) with two layers for initial triage, and a large language model (LLM) for contextual analysis. We evaluated the framework against a publicly available benchmark dataset, as well as against a simulated dataset. The experimental results show that the framework can effectively adapt to different SOC objectives autonomously and identify suspicious and malicious traffic. The framework enhances operational effectiveness by supporting SOC analysts in their decision-making to block, allow, or monitor network traffic. This study thus enhances cybersecurity and threat hunting literature by presenting the novel threat hunting framework for security decision- making, as well as promoting cumulative research efforts to develop more effective frameworks to battle continuously evolving cyber threats. 

---
# Variable-Length Audio Fingerprinting 

**Authors**: Hongjie Chen, Hanyu Meng, Huimin Zeng, Ryan A. Rossi, Lie Lu, Josh Kimball  

**Link**: [PDF](https://arxiv.org/pdf/2603.23947)  

**Abstract**: Audio fingerprinting converts audio to much lower-dimensional representations, allowing distorted recordings to still be recognized as their originals through similar fingerprints. Existing deep learning approaches rigidly fingerprint fixed-length audio segments, thereby neglecting temporal dynamics during segmentation. To address limitations due to this rigidity, we propose Variable-Length Audio FingerPrinting (VLAFP), a novel method that supports variable-length fingerprinting. To the best of our knowledge, VLAFP is the first deep audio fingerprinting model capable of processing audio of variable length, for both training and testing. Our experiments show that VLAFP outperforms existing state-of-the-arts in live audio identification and audio retrieval across three real-world datasets. 

---
# High-Fidelity Face Content Recovery via Tamper-Resilient Versatile Watermarking 

**Authors**: Peipeng Yu, Jinfeng Xie, Chengfu Ou, Xiaoyu Zhou, Jianwei Fei, Yunshu Dai, Zhihua Xia, Chip Hong Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23940)  

**Abstract**: The proliferation of AIGC-driven face manipulation and deepfakes poses severe threats to media provenance, integrity, and copyright protection. Prior versatile watermarking systems typically rely on embedding explicit localization payloads, which introduces a fidelity--functionality trade-off: larger localization signals degrade visual quality and often reduce decoding robustness under strong generative edits. Moreover, existing methods rarely support content recovery, limiting their forensic value when original evidence must be reconstructed. To address these challenges, we present VeriFi, a versatile watermarking framework that unifies copyright protection, pixel-level manipulation localization, and high-fidelity face content recovery. VeriFi makes three key contributions: (1) it embeds a compact semantic latent watermark that serves as an content-preserving prior, enabling faithful restoration even after severe manipulations; (2) it achieves fine-grained localization without embedding localization-specific artifacts by correlating image features with decoded provenance signals; and (3) it introduces an AIGC attack simulator that combines latent-space mixing with seamless blending to improve robustness to realistic deepfake pipelines. Extensive experiments on CelebA-HQ and FFHQ show that VeriFi consistently outperforms strong baselines in watermark robustness, localization accuracy, and recovery quality, providing a practical and verifiable defense for deepfake forensics. 

---
# Revealing Multi-View Hallucination in Large Vision-Language Models 

**Authors**: Wooje Park, Insu Lee, Soohyun Kim, Jaeyun Jang, Minyoung Noh, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2603.23934)  

**Abstract**: Large vision-language models (LVLMs) are increasingly being applied to multi-view image inputs captured from diverse viewpoints. However, despite this growing use, current LVLMs often confuse or mismatch visual information originating from different instances or viewpoints, a phenomenon we term multi-view hallucination. To systematically analyze this problem, we construct MVH-Bench, a benchmark comprising 4.8k question-answer pairs targeting two types of hallucination: cross-instance and cross-view. Empirical results show that recent LVLMs struggle to correctly associate visual evidence with its corresponding instance or viewpoint. To overcome this limitation, we propose Reference Shift Contrastive Decoding (RSCD), a training-free decoding technique that suppresses visual interference by generating negative logits through attention masking. Experiments on MVH-Bench with Qwen2.5-VL and LLaVA-OneVision demonstrate that RSCD consistently improves performance by up to 21.1 and 34.6 points over existing hallucination mitigation methods, highlighting the effectiveness of our approach. 

---
# DecepGPT: Schema-Driven Deception Detection with Multicultural Datasets and Robust Multimodal Learning 

**Authors**: Jiajian Huang, Dongliang Zhu, Zitong YU, Hui Ma, Jiayu Zhang, Chunmei Zhu, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.23916)  

**Abstract**: Multimodal deception detection aims to identify deceptive behavior by analyzing audiovisual cues for forensics and security. In these high-stakes settings, investigators need verifiable evidence connecting audiovisual cues to final decisions, along with reliable generalization across domains and cultural contexts. However, existing benchmarks provide only binary labels without intermediate reasoning cues. Datasets are also small with limited scenario coverage, leading to shortcut learning. We address these issues through three contributions. First, we construct reasoning datasets by augmenting existing benchmarks with structured cue-level descriptions and reasoning chains, enabling model output auditable reports. Second, we release T4-Deception, a multicultural dataset based on the unified ``To Tell The Truth'' television format implemented across four countries. With 1695 samples, it is the largest non-laboratory deception detection dataset. Third, we propose two modules for robust learning under small-data conditions. Stabilized Individuality-Commonality Synergy (SICS) refines multimodal representations by synergizing learnable global priors with sample-adaptive residuals, followed by a polarity-aware adjustment that bi-directionally recalibrates representations. Distilled Modality Consistency (DMC) aligns modality-specific predictions with the fused multimodal predictions via knowledge distillation to prevent unimodal shortcut learning. Experiments on three established benchmarks and our novel dataset demonstrate that our method achieves state-of-the-art performance in both in-domain and cross-domain scenarios, while exhibiting superior transferability across diverse cultural contexts. The datasets and codes will be released. 

---
# Self-Distillation for Multi-Token Prediction 

**Authors**: Guoliang Zhao, Ruobing Xie, An Wang, Shuaipeng Li, Huaibing Xie, Xingwu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.23911)  

**Abstract**: As Large Language Models (LLMs) scale up, inference efficiency becomes a critical bottleneck. Multi-Token Prediction (MTP) could accelerate LLM inference by predicting multiple future tokens in parallel. However, existing MTP approaches still face two challenges: limited acceptance rates of MTP heads, and difficulties in jointly training multiple MTP heads. Therefore, we propose MTP-D, a simple yet effective self-distillation method with minimal additional training cost, which boosts MTP head acceptance rates (+7.5\%) while maximumly preserving main-head performance. We also introduce a looped extension strategy for MTP-D, enabling effective and economical MTP head extension and further significant inference speedup to 1-head MTP (+220.4\%). Moreover, we systematically explore and validate key insights on the distillation strategies and the potential scalability of MTP through extensive experiments on seven benchmarks. These results demonstrate that our MTP-D and looped extension strategy effectively enhance MTP-head performance and inference efficiency, facilitating the practical usage of MTP in LLMs. 

---
# Latent Bias Alignment for High-Fidelity Diffusion Inversion in Real-World Image Reconstruction and Manipulation 

**Authors**: Weiming Chen, Qifan Liu, Siyi Liu, Yushun Tang, Yijia Wang, Zhihan Zhu, Zhihai He  

**Link**: [PDF](https://arxiv.org/pdf/2603.23903)  

**Abstract**: Recent research has shown that text-to-image diffusion models are capable of generating high-quality images guided by text prompts. But can they be used to generate or approximate real-world images from the seed noise? This is known as the diffusion inversion problem, which serves as a fundamental building block for bridging diffusion models and real-world scenarios. However, existing diffusion inversion methods often suffer from low reconstruction quality or weak robustness. Two major challenges need to be carefully addressed: (1) the misalignment between the inversion and generation trajectories during the diffusion process, and (2) the mismatch between the diffusion inversion process and the VQ autoencoder (VQAE) reconstruction. To address these challenges, we introduce a latent bias vector at each inversion step, which is learned to reduce the misalignment between inversion and generation trajectories. We refer to this strategy as Latent Bias Optimization (LBO). Furthermore, we perform an approximate joint optimization of the diffusion inversion and VQAE reconstruction processes by learning to adjust the image latent representation, which serves as the connecting interface between them. We refer to this technique as Image Latent Boosting (ILB). Extensive experimental results demonstrate that the proposed method significantly improves the image reconstruction quality of the diffusion model, as well as the performance of downstream tasks, including image editing and rare concept generation. 

---
# Knowledge-Refined Dual Context-Aware Network for Partially Relevant Video Retrieval 

**Authors**: Junkai Yang, Qirui Wang, Yaoqing Jin, Shuai Ma, Minghan Xu, Shanmin Pang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23902)  

**Abstract**: Retrieving partially relevant segments from untrimmed videos remains difficult due to two persistent challenges: the mismatch in information density between text and video segments, and limited attention mechanisms that overlook semantic focus and event correlations. We present KDC-Net, a Knowledge-Refined Dual Context-Aware Network that tackles these issues from both textual and visual perspectives. On the text side, a Hierarchical Semantic Aggregation module captures and adaptively fuses multi-scale phrase cues to enrich query semantics. On the video side, a Dynamic Temporal Attention mechanism employs relative positional encoding and adaptive temporal windows to highlight key events with local temporal coherence. Additionally, a dynamic CLIP-based distillation strategy, enhanced with temporal-continuity-aware refinement, ensures segment-aware and objective-aligned knowledge transfer. Experiments on PRVR benchmarks show that KDC-Net consistently outperforms state-of-the-art methods, especially under low moment-to-video ratios. 

---
# SM-Net: Learning a Continuous Spectral Manifold from Multiple Stellar Libraries 

**Authors**: Omar Anwar, Aaron S. G. Robotham, Luca Cortese, Kevin Vinsen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23899)  

**Abstract**: We present SM-Net, a machine-learning model that learns a continuous spectral manifold from multiple high-resolution stellar libraries. SM-Net generates stellar spectra directly from the fundamental stellar parameters effective temperature (Teff), surface gravity (log g), and metallicity (log Z). It is trained on a combined grid derived from the PHOENIX-Husser, C3K-Conroy, OB-PoWR, and TMAP-Werner libraries. By combining their parameter spaces, we construct a composite dataset that spans a broader and more continuous region of stellar parameter space than any individual library. The unified grid covers Teff = 2,000-190,000 K, log g = -1 to 9, and log Z = -4 to 1, with spectra spanning 3,000-100,000 Angstrom. Within this domain, SM-Net provides smooth interpolation across heterogeneous library boundaries. Outside the sampled region, it can produce numerically smooth exploratory predictions, although these extrapolations are not directly validated against reference models. Zero or masked flux values are treated as unknowns rather than physical zeros, allowing the network to infer missing regions using correlations learned from neighbouring grid points. Across 3,538 training and 11,530 test spectra, SM-Net achieves mean squared errors of 1.47 x 10^-5 on the training set and 2.34 x 10^-5 on the test set in the transformed log1p-scaled flux representation. Inference throughput exceeds 14,000 spectra per second on a single GPU. We also release the model together with an interactive web dashboard for real-time spectral generation and visualisation. SM-Net provides a fast, robust, and flexible data-driven complement to traditional stellar population synthesis libraries. 

---
# AgentChemist: A Multi-Agent Experimental Robotic Platform Integrating Chemical Perception and Precise Control 

**Authors**: Xiangyi Wei, Fei Wang, Haotian Zhang, Xin An, Haitian Zhu, Lianrui Hu, Yang Li, Changbo Wang, Xiao He  

**Link**: [PDF](https://arxiv.org/pdf/2603.23886)  

**Abstract**: Chemical laboratory automation has long been constrained by rigid workflows and poor adaptability to the long-tail distribution of experimental tasks. While most automated platforms perform well on a narrow set of standardized procedures, real laboratories involve diverse, infrequent, and evolving operations that fall outside predefined protocols. This mismatch prevents existing systems from generalizing to novel reaction conditions, uncommon instrument configurations, and unexpected procedural variations. We present a multi-agent robotic platform designed to address this long-tail challenge through collaborative task decomposition, dynamic scheduling, and adaptive control. The system integrates chemical perception for real-time reaction monitoring with feedback-driven execution, enabling it to adjust actions based on evolving experimental states rather than fixed scripts. Validation via acid-base titration demonstrates autonomous progress tracking, adaptive dispensing control, and reliable end-to-end experiment execution. By improving generalization across diverse laboratory scenarios, this platform provides a practical pathway toward intelligent, flexible, and scalable laboratory automation. 

---
# The Luna Bound Propagator for Formal Analysis of Neural Networks 

**Authors**: Henry LeCates, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23878)  

**Abstract**: The parameterized CROWN analysis, a.k.a., alpha-CROWN, has emerged as a practically successful bound propagation method for neural network verification. However, existing implementations of alpha-CROWN are limited to Python, which complicates integration into existing DNN verifiers and long-term production-level systems. We introduce Luna, a new bound propagator implemented in C++. Luna supports Interval Bound Propagation, the CROWN analysis, and the alpha-CROWN analysis over a general computational graph. We describe the architecture of Luna and show that it is competitive with the state-of-the-art alpha-CROWN implementation in terms of both bound tightness and computational efficiency on benchmarks from VNN-COMP 2025. 

---
# HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation 

**Authors**: Ken Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.23871)  

**Abstract**: Large language models trained with reinforcement learning (RL) for mathematical reasoning face a fundamental challenge: on problems the model cannot solve at all - "cliff" prompts - the RL gradient vanishes entirely, preventing any learning signal from reaching these failure modes. We introduce Hybrid Distillation Policy Optimization (HDPO), which augments standard RL with privileged self-distillation targeting cliff prompts. On each training step, HDPO identifies prompts where all rollouts fail, generates privileged rollouts by providing the model with ground-truth information, filters for correct solutions, and distills the teacher's token-level distribution into the student. Because teacher and student share the same weights - differing only in their input - the realizability gap is provably bounded, unlike cross-model distillation. We prove that R=1 filtered privileged generation recovers the optimal KL-regularized RL policy in the hard-threshold limit. Experiments on OpenMathInstruct-2 with Qwen2.5-Math-1.5B-Instruct show that HDPO consistently improves coverage metrics (pass@4 by +0.8-1.1%, pass@8 by +0.4-1.7%) while maintaining greedy accuracy, with the distillation weight lambda providing direct control over the exploration-exploitation tradeoff. 

---
# Can VLMs Reason Robustly? A Neuro-Symbolic Investigation 

**Authors**: Weixin Chen, Antonio Vergari, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.23867)  

**Abstract**: Vision-Language Models (VLMs) have been applied to a wide range of reasoning tasks, yet it remains unclear whether they can reason robustly under distribution shifts. In this paper, we study covariate shifts in which the perceptual input distribution changes while the underlying prediction rules do not. To investigate this question, we consider visual deductive reasoning tasks, where a model is required to answer a query given an image and logical rules defined over the object concepts in the image. Empirically, we find that VLMs fine-tuned through gradient-based end-to-end training can achieve high in-distribution accuracy but fail to generalize under such shifts, suggesting that fine-tuning does not reliably induce the underlying reasoning function. This motivates a neuro-symbolic perspective that decouples perception from reasoning. However, we further observe that recent neuro-symbolic approaches that rely on black-box components for reasoning can still exhibit inconsistent robustness across tasks. To address this issue, we propose VLC, a neuro-symbolic method that combines VLM-based concept recognition with circuit-based symbolic reasoning. In particular, task rules are compiled into a symbolic program, specifically a circuit, which executes the rules exactly over the object concepts recognized by the VLM. Experiments on three visual deductive reasoning tasks with distinct rule sets show that VLC consistently achieves strong performance under covariate shifts, highlighting its ability to support robust reasoning. 

---
# Generative AI User Experience: Developing Human--AI Epistemic Partnership 

**Authors**: Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2603.23863)  

**Abstract**: Generative AI (GenAI) has rapidly entered education, yet its user experience is often explained through adoption-oriented constructs such as usefulness, ease of use, and engagement. We argue that these constructs are no longer sufficient because systems such as ChatGPT do not merely support learning tasks but also participate in knowledge construction. Existing theories cannot explain why GenAI frequently produces experiences characterized by negotiated authority, redistributed cognition, and accountability tension. To address this gap, this paper develops the Human--AI Epistemic Partnership Theory (HAEPT), explaining the GenAI user experience as a form of epistemic partnership that features a dynamic negotiation of three interlocking contracts: epistemic, agency, and accountability. We argue that findings on trust, over-reliance, academic integrity, teacher caution, and relational interaction about GenAI can be reinterpreted as tensions within these contracts rather than as isolated issues. Instead of holding a single, stable view of GenAI, users adjust how they relate to it over time through calibration cycles. These repeated interactions account for why trust and skepticism often coexist and for how partnership modes describe recurrent configurations of human--AI collaboration across tasks. To demonstrate the usefulness of HAEPT, we applied it to analyze the UX of collaborative learning with AI speakers and AI-facilitated scientific argumentation, illustrating different contract configurations. 

---
# Deep Convolutional Neural Networks for predicting highest priority functional group in organic molecules 

**Authors**: Kunal Khatri, Vineet Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2603.23862)  

**Abstract**: Our work addresses the problem of predicting the highest priority functional group present in an organic molecule. Functional Groups are groups of bound atoms that determine the physical and chemical properties of organic molecules. In the presence of multiple functional groups, the dominant functional group determines the compound's properties. Fourier-transform Infrared spectroscopy (FTIR) is a commonly used spectroscopic method for identifying the presence or absence of functional groups within a compound. We propose the use of a Deep Convolutional Neural Networks (CNN) to predict the highest priority functional group from the Fourier-transform infrared spectrum (FTIR) of the organic molecule. We have compared our model with other previously applied Machine Learning (ML) method Support Vector Machine (SVM) and reasoned why CNN outperforms it. 

---
# Why the Maximum Second Derivative of Activations Matters for Adversarial Robustness 

**Authors**: Yunrui Yu, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23860)  

**Abstract**: This work investigates the critical role of activation function curvature -- quantified by the maximum second derivative $\max|\sigma''|$ -- in adversarial robustness. Using the Recursive Curvature-Tunable Activation Family (RCT-AF), which enables precise control over curvature through parameters $\alpha$ and $\beta$, we systematically analyze this relationship. Our study reveals a fundamental trade-off: insufficient curvature limits model expressivity, while excessive curvature amplifies the normalized Hessian diagonal norm of the loss, leading to sharper minima that hinder robust generalization. This results in a non-monotonic relationship where optimal adversarial robustness consistently occurs when $\max|\sigma''|$ falls within 4 to 10, a finding that holds across diverse network architectures, datasets, and adversarial training methods. We provide theoretical insights into how activation curvature affects the diagonal elements of the hessian matrix of the loss, and experimentally demonstrate that the normalized Hessian diagonal norm exhibits a U-shaped dependence on $\max|\sigma''|$, with its minimum within the optimal robustness range, thereby validating the proposed mechanism. 

---
# PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay 

**Authors**: Rohan Khetan, Ashna Khetan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23841)  

**Abstract**: While Large Language Models (LLMs) are increasingly used as primary sources of information, their potential for political bias may impact their objectivity. Existing benchmarks of LLM social bias primarily evaluate gender and racial stereotypes. When political bias is included, it is typically measured at a coarse level, neglecting the specific values that shape sociopolitical leanings. This study investigates political bias in eight prominent LLMs (Claude, Deepseek, Gemini, GPT, Grok, Llama, Qwen Base, Qwen Instruction-Tuned) using PoliticsBench: a novel multi-turn roleplay framework adapted from the EQ-Bench-v3 psychometric benchmark. We test whether commercially developed LLMs display a systematic left-leaning bias that becomes more pronounced in later stages of multi-stage roleplay. Through twenty evolving scenarios, each model reported its stance and determined its course of action. Scoring these responses on a scale of ten political values, we explored the values underlying chatbots' deviations from unbiased standards. Seven of our eight models leaned left, while Grok leaned right. Each left-leaning LLM strongly exhibited liberal traits and moderately exhibited conservative ones. We discovered slight variations in alignment scores across stages of roleplay, with no particular pattern. Though most models used consequence-based reasoning, Grok frequently argued with facts and statistics. Our study presents the first psychometric evaluation of political values in LLMs through multi-stage, free-text interactions. 

---
# Circuit Complexity of Hierarchical Knowledge Tracing and Implications for Log-Precision Transformers 

**Authors**: Naiming Liu, Richard Baraniuk, Shashank Sonkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23823)  

**Abstract**: Knowledge tracing models mastery over interconnected concepts, often organized by prerequisites. We analyze hierarchical prerequisite propagation through a circuit-complexity lens to clarify what is provable about transformer-style computation on deep concept hierarchies. Using recent results that log-precision transformers lie in logspace-uniform $\mathsf{TC}^0$, we formalize prerequisite-tree tasks including recursive-majority mastery propagation. Unconditionally, recursive-majority propagation lies in $\mathsf{NC}^1$ via $O(\log n)$-depth bounded-fanin circuits, while separating it from uniform $\mathsf{TC}^0$ would require major progress on open lower bounds. Under a monotonicity restriction, we obtain an unconditional barrier: alternating ALL/ANY prerequisite trees yield a strict depth hierarchy for \emph{monotone} threshold circuits. Empirically, transformer encoders trained on recursive-majority trees converge to permutation-invariant shortcuts; explicit structure alone does not prevent this, but auxiliary supervision on intermediate subtrees elicits structure-dependent computation and achieves near-perfect accuracy at depths 3--4. These findings motivate structure-aware objectives and iterative mechanisms for prerequisite-sensitive knowledge tracing on deep hierarchies. 

---
# Perturbation: A simple and efficient adversarial tracer for representation learning in language models 

**Authors**: Joshua Rozner, Cory Shain  

**Link**: [PDF](https://arxiv.org/pdf/2603.23821)  

**Abstract**: Linguistic representation learning in deep neural language models (LMs) has been studied for decades, for both practical and theoretical reasons. However, finding representations in LMs remains an unsolved problem, in part due to a dilemma between enforcing implausible constraints on representations (e.g., linearity; Arora et al. 2024) and trivializing the notion of representation altogether (Sutter et al., 2025). Here we escape this dilemma by reconceptualizing representations not as patterns of activation but as conduits for learning. Our approach is simple: we perturb an LM by fine-tuning it on a single adversarial example and measure how this perturbation ``infects'' other examples. Perturbation makes no geometric assumptions, and unlike other methods, it does not find representations where it should not (e.g., in untrained LMs). But in trained LMs, perturbation reveals structured transfer at multiple linguistic grain sizes, suggesting that LMs both generalize along representational lines and acquire linguistic abstractions from experience alone. 

---
# Willful Disobedience: Automatically Detecting Failures in Agentic Traces 

**Authors**: Reshabh K Sharma, Shraddha Barke, Benjamin Zorn  

**Link**: [PDF](https://arxiv.org/pdf/2603.23806)  

**Abstract**: AI agents are increasingly embedded in real software systems, where they execute multi-step workflows through multi-turn dialogue, tool invocations, and intermediate decisions. These long execution histories, called agentic traces, make validation difficult. Outcome-only benchmarks can miss critical procedural failures, such as incorrect workflow routing, unsafe tool usage, or violations of prompt-specified rules. This paper presents AgentPex, an AI-powered tool designed to systematically evaluate agentic traces. AgentPex extracts behavioral rules from agent prompts and system instructions, then uses these specifications to automatically evaluate traces for compliance. We evaluate AgentPex on 424 traces from {\tau}2-bench across models in telecom, retail, and airline customer service. Our results show that AgentPex distinguishes agent behavior across models and surfaces specification violations that are not captured by outcome-only scoring. It also provides fine-grained analysis by domain and metric, enabling developers to understand agent strengths and weaknesses at scale. 

---
# Deep Neural Regression Collapse 

**Authors**: Akshay Rangamani, Altay Unal  

**Link**: [PDF](https://arxiv.org/pdf/2603.23805)  

**Abstract**: Neural Collapse is a phenomenon that helps identify sparse and low rank structures in deep classifiers. Recent work has extended the definition of neural collapse to regression problems, albeit only measuring the phenomenon at the last layer. In this paper, we establish that Neural Regression Collapse (NRC) also occurs below the last layer across different types of models. We show that in the collapsed layers of neural regression models, features lie in a subspace that corresponds to the target dimension, the feature covariance aligns with the target covariance, the input subspace of the layer weights aligns with the feature subspace, and the linear prediction error of the features is close to the overall prediction error of the model. In addition to establishing Deep NRC, we also show that models that exhibit Deep NRC learn the intrinsic dimension of low rank targets and explore the necessity of weight decay in inducing Deep NRC. This paper provides a more complete picture of the simple structure learned by deep networks in the context of regression. 

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
# Probabilistic Geometric Alignment via Bayesian Latent Transport for Domain-Adaptive Foundation Models 

**Authors**: Kuepon Aueawatthanaphisut, Kuepon Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2603.23783)  

**Abstract**: Adapting large-scale foundation models to new domains with limited supervision remains a fundamental challenge due to latent distribution mismatch, unstable optimization dynamics, and miscalibrated uncertainty propagation. This paper introduces an uncertainty-aware probabilistic latent transport framework that formulates domain adaptation as a stochastic geometric alignment problem in representation space. A Bayesian transport operator is proposed to redistribute latent probability mass along Wasserstein-type geodesic trajectories, while a PAC-Bayesian regularization mechanism constrains posterior model complexity to mitigate catastrophic overfitting. The proposed formulation yields theoretical guarantees on convergence stability, loss landscape smoothness, and sample efficiency under distributional shift. Empirical analyses demonstrate substantial reduction in latent manifold discrepancy, accelerated transport energy decay, and improved covariance calibration compared with deterministic fine-tuning and adversarial domain adaptation baselines. Furthermore, bounded posterior uncertainty evolution indicates enhanced probabilistic reliability during cross-domain transfer. By establishing a principled connection between stochastic optimal transport geometry and statistical generalization theory, the proposed framework provides new insights into robust adaptation of modern foundation architectures operating in heterogeneous environments. These findings suggest that uncertainty-aware probabilistic alignment constitutes a promising paradigm for reliable transfer learning in next-generation deep representation systems. 

---
# Human-in-the-Loop Pareto Optimization: Trade-off Characterization for Assist-as-Needed Training and Performance Evaluation 

**Authors**: Harun Tolasa, Volkan Patoglu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23777)  

**Abstract**: During human motor skill training and physical rehabilitation, there is an inherent trade-off between task difficulty and user performance. Characterizing this trade-off is crucial for evaluating user performance, designing assist-as-needed (AAN) protocols, and assessing the efficacy of training protocols. In this study, we propose a novel human-in-the-loop (HiL) Pareto optimization approach to characterize the trade-off between task performance and the perceived challenge level of motor learning or rehabilitation tasks. We adapt Bayesian multi-criteria optimization to systematically and efficiently perform HiL Pareto characterizations. Our HiL optimization employs a hybrid model that measures performance with a quantitative metric, while the perceived challenge level is captured with a qualitative metric. We demonstrate the feasibility of the proposed HiL Pareto characterization through a user study. Furthermore, we present the utility of the framework through three use cases in the context of a manual skill training task with haptic feedback. First, we demonstrate how the characterized trade-off can be used to design a sample AAN training protocol for a motor learning task and to evaluate the group-level efficacy of the proposed AAN protocol relative to a baseline adaptive assistance protocol. Second, we demonstrate that individual-level comparisons of the trade-offs characterized before and after the training session enable fair evaluation of training progress under different assistance levels. This evaluation method is more general than standard performance evaluations, as it can provide insights even when users cannot perform the task without assistance. Third, we show that the characterized trade-offs also enable fair performance comparisons among different users, as they capture the best possible performance of each user under all feasible assistance levels. 

---
# AI-driven Intent-Based Networking Approach for Self-configuration of Next Generation Networks 

**Authors**: Md. Kamrul Hossain, Walid Aljoby  

**Link**: [PDF](https://arxiv.org/pdf/2603.23772)  

**Abstract**: Intent-Based Networking (IBN) aims to simplify operating heterogeneous infrastructures by translating high-level intents into enforceable policies and assuring compliance. However, dependable automation remains difficult because (i) realizing intents from ambiguous natural language into controller-ready policies is brittle and prone to conflicts and unintended side effects, and (ii) assurance is often reactive and struggles in multi-intent settings where faults create cascading symptoms and ambiguous telemetry. This paper proposes an end-to-end closed-loop IBN pipeline that uses large language models with structured validation for natural language to policy realization and conflict-aware activation, and reformulates assurance as proactive multi-intent failure prediction with root-cause disambiguation. The expected outcome is operator-trustworthy automation that provides actionable early warnings, interpretable explanations, and measurable lead time for remediation. 

---
# Self Paced Gaussian Contextual Reinforcement Learning 

**Authors**: Mohsen Sahraei Ardakani, Rui Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.23755)  

**Abstract**: Curriculum learning improves reinforcement learning (RL) efficiency by sequencing tasks from simple to complex. However, many self-paced curriculum methods rely on computationally expensive inner-loop optimizations, limiting their scalability in high-dimensional context spaces. In this paper, we propose Self-Paced Gaussian Curriculum Learning (SPGL), a novel approach that avoids costly numerical procedures by leveraging a closed-form update rule for Gaussian context distributions. SPGL maintains the sample efficiency and adaptability of traditional self-paced methods while substantially reducing computational overhead. We provide theoretical guarantees on convergence and validate our method across several contextual RL benchmarks, including the Point Mass, Lunar Lander, and Ball Catching environments. Experimental results show that SPGL matches or outperforms existing curriculum methods, especially in hidden context scenarios, and achieves more stable context distribution convergence. Our method offers a scalable, principled alternative for curriculum generation in challenging continuous and partially observable domains. 

---
# CDMT-EHR: A Continuous-Time Diffusion Framework for Generating Mixed-Type Time-Series Electronic Health Records 

**Authors**: Shaonan Liu, Yuichiro Iwashita, Soichiro Nakako, Masakazu Iwamura, Koichi Kise  

**Link**: [PDF](https://arxiv.org/pdf/2603.23719)  

**Abstract**: Electronic health records (EHRs) are invaluable for clinical research, yet privacy concerns severely restrict data sharing. Synthetic data generation offers a promising solution, but EHRs present unique challenges: they contain both numerical and categorical features that evolve over time. While diffusion models have demonstrated strong performance in EHR synthesis, existing approaches predominantly rely on discrete-time formulations, which suffer from finite-step approximation errors and coupled training-sampling step counts. We propose a continuous-time diffusion framework for generating mixed-type time-series EHRs with three contributions: (1) continuous-time diffusion with a bidirectional gated recurrent unit backbone for capturing temporal dependencies, (2) unified Gaussian diffusion via learnable continuous embeddings for categorical variables, enabling joint cross-feature modeling, and (3) a factorized learnable noise schedule that adapts to per-feature-per-timestep learning difficulties. Experiments on two large-scale intensive care unit datasets demonstrate that our method outperforms existing approaches in downstream task performance, distribution fidelity, and discriminability, while requiring only 50 sampling steps compared to 1,000 for baseline methods. Classifier-free guidance further enables effective conditional generation for class-imbalanced clinical scenarios. 

---
# An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System: [Experiments and Analysis] 

**Authors**: Duo Lu, Helena Caminal, Manos Chatzakis, Yannis Papakonstantinou, Yannis Chronis, Vaibhav Jain, Fatma Özcan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23710)  

**Abstract**: Filtered Vector Search (FVS) is critical for supporting semantic search and GenAI applications in modern database systems. However, existing research most often evaluates algorithms in specialized libraries, making optimistic assumptions that do not align with enterprise-grade database systems. Our work challenges this premise by demonstrating that in a production-grade database system, commonly made assumptions do not hold, leading to performance characteristics and algorithmic trade-offs that are fundamentally different from those observed in isolated library settings. This paper presents the first in-depth analysis of filter-agnostic FVS algorithms within a production PostgreSQL-compatible system. We systematically evaluate post-filtering and inline-filtering strategies across a wide range of selectivities and correlations.
Our central finding is that the optimal algorithm is not dictated by the cost of distance computations alone, but that system-level overheads that come from both distance computations and filter operations (like page accesses and data retrieval) play a significant role. We demonstrate that graph-based approaches (such as NaviX/ACORN) can incur prohibitive numbers of filter checks and system-level overheads, compared with clustering-based indexes such as ScaNN, often canceling out their theoretical benefits in real-world database environments.
Ultimately, our findings provide the database community with crucial insights and practical guidelines, demonstrating that the optimal choice for a filter-agnostic FVS algorithm is not absolute, but rather a system-aware decision contingent on the interplay between workload characteristics and the underlying costs of data access in a real-world database architecture. 

---
# The Diminishing Returns of Early-Exit Decoding in Modern LLMs 

**Authors**: Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23701)  

**Abstract**: In Large Language Model (LLM) inference, early-exit refers to stopping computation at an intermediate layer once the prediction is sufficiently confident, thereby reducing latency and cost. However, recent LLMs adopt improved pretraining recipes and architectures that reduce layer redundancy, potentially limiting early-exit opportunities. We re-evaluate layer-wise early-exit in modern LLMs and analyze how intermediate representations evolve during training. We introduce a metric to quantify a model's intrinsic suitability for early-exit and propose a benchmark for researchers to explore the potential early-exit benefits on different models and workloads. Our results show a diminishing trend in early-exit effectiveness across newer model generations. We further find that dense transformers generally offer greater early-exit potential than Mixture-of-Experts and State Space Models. In addition, larger models, particularly those with more than 20 billion parameters, and base pretrained models without specialized tuning tend to exhibit higher early-exit potential. 

---
# Assessment Design in the AI Era: A Method for Identifying Items Functioning Differentially for Humans and Chatbots 

**Authors**: Licol Zeinfeld, Alona Strugatski, Ziva Bar-Dov, Ron Blonder, Shelley Rap, Giora Alexandron  

**Link**: [PDF](https://arxiv.org/pdf/2603.23682)  

**Abstract**: The rapid adoption of large language models (LLMs) in education raises profound challenges for assessment design. To adapt assessments to the presence of LLM-based tools, it is crucial to characterize the strengths and weaknesses of LLMs in a generalizable, valid and reliable manner. However, current LLM evaluations often rely on descriptive statistics derived from benchmarks, and little research applies theory-grounded measurement methods to characterize LLM capabilities relative to human learners in ways that directly support assessment design. Here, by combining educational data mining and psychometric theory, we introduce a statistically principled approach for identifying items on which humans and LLMs show systematic response differences, pinpointing where assessments may be most vulnerable to AI misuse, and which task dimensions make problems particularly easy or difficult for generative AI. The method is based on Differential Item Functioning (DIF) analysis -- traditionally used to detect bias across demographic groups -- together with negative control analysis and item-total correlation discrimination analysis. It is evaluated on responses from human learners and six leading chatbots (ChatGPT-4o \& 5.2, Gemini 1.5 \& 3 Pro, Claude 3.5 \& 4.5 Sonnet) to two instruments: a high school chemistry diagnostic test and a university entrance exam. Subject-matter experts then analyzed DIF-flagged items to characterize task dimensions associated with chatbot over- or under-performance. Results show that DIF-informed analytics provide a robust framework for understanding where LLM and human capabilities diverge, and highlight their value for improving the design of valid, reliable, and fair assessment in the AI era. 

---
# Learning What Can Be Picked: Active Reachability Estimation for Efficient Robotic Fruit Harvesting 

**Authors**: Nur Afsa Syeda, Mohamed Elmahallawy, Luis Fernando de la Torre, John Miller  

**Link**: [PDF](https://arxiv.org/pdf/2603.23679)  

**Abstract**: Agriculture remains a cornerstone of global health and economic sustainability, yet labor-intensive tasks such as harvesting high-value crops continue to face growing workforce shortages. Robotic harvesting systems offer a promising solution; however, their deployment in unstructured orchard environments is constrained by inefficient perception-to-action pipelines. In particular, existing approaches often rely on exhaustive inverse kinematics or motion planning to determine whether a target fruit is reachable, leading to unnecessary computation and delayed decision-making. Our approach combines RGB-D perception with active learning to directly learn reachability as a binary decision problem. We then leverage active learning to selectively query the most informative samples for reachability labeling, significantly reducing annotation effort while maintaining high predictive accuracy. Extensive experiments demonstrate that the proposed framework achieves accurate reachability prediction with substantially fewer labeled samples, yielding approximately 6--8% higher accuracy than random sampling and enabling label-efficient adaptation to new orchard configurations. Among the evaluated strategies, entropy- and margin-based sampling outperform Query-by-Committee and standard uncertainty sampling in low-label regimes, while all strategies converge to comparable performance as the labeled set grows. These results highlight the effectiveness of active learning for task-level perception in agricultural robotics and position our approach as a scalable alternative to computation-heavy kinematic reachability analysis. Our code is available through this https URL. 

---
# PLACID: Privacy-preserving Large language models for Acronym Clinical Inference and Disambiguation 

**Authors**: Manjushree B. Aithal, Ph.D., Alexander Kotz, James Mitchell, Ph.D  

**Link**: [PDF](https://arxiv.org/pdf/2603.23678)  

**Abstract**: Large Language Models (LLMs) offer transformative solutions across many domains, but healthcare integration is hindered by strict data privacy constraints. Clinical narratives are dense with ambiguous acronyms, misinterpretation these abbreviations can precipitate severe outcomes like life-threatening medication errors. While cloud-dependent LLMs excel at Acronym Disambiguation, transmitting Protected Health Information to external servers violates privacy frameworks. To bridge this gap, this study pioneers the evaluation of small-parameter models deployed entirely on-device to ensure privacy preservation. We introduce a privacy-preserving cascaded pipeline leveraging general-purpose local models to detect clinical acronyms, routing them to domain-specific biomedical models for context-relevant expansions. Results reveal that while general instruction-following models achieve high detection accuracy (~0.988), their expansion capabilities plummet (~0.655). Our cascaded approach utilizes domain-specific medical models to increase expansion accuracy to (~0.81). This novel work demonstrates that privacy-preserving, on-device (2B-10B) models deliver high-fidelity clinical acronym disambiguation support. 

---
# Prototype Fusion: A Training-Free Multi-Layer Approach to OOD Detection 

**Authors**: Shreen Gul, Mohamed Elmahallawy, Ardhendu Tripathy, Sanjay Madria  

**Link**: [PDF](https://arxiv.org/pdf/2603.23677)  

**Abstract**: Deep learning models are increasingly deployed in safety-critical applications, where reliable out-of-distribution (OOD) detection is essential to ensure robustness. Existing methods predominantly rely on the penultimate-layer activations of neural networks, assuming they encapsulate the most informative in-distribution (ID) representations. In this work, we revisit this assumption to show that intermediate layers encode equally rich and discriminative information for OOD detection. Based on this observation, we propose a simple yet effective model-agnostic approach that leverages internal representations across multiple layers. Our scheme aggregates features from successive convolutional blocks, computes class-wise mean embeddings, and applies L_2 normalization to form compact ID prototypes capturing class semantics. During inference, cosine similarity between test features and these prototypes serves as an OOD score--ID samples exhibit strong affinity to at least one prototype, whereas OOD samples remain uniformly distant. Extensive experiments on state-of-the-art OOD benchmarks across diverse architectures demonstrate that our approach delivers robust, architecture-agnostic performance and strong generalization for image classification. Notably, it improves AUROC by up to 4.41% and reduces FPR by 13.58%, highlighting multi-layer feature aggregation as a powerful yet underexplored signal for OOD detection, challenging the dominance of penultimate-layer-based methods. Our code is available at: this https URL. 

---
# Estimating Individual Tree Height and Species from UAV Imagery 

**Authors**: Jannik Endres, Etienne Laliberté, David Rolnick, Arthur Ouaknine  

**Link**: [PDF](https://arxiv.org/pdf/2603.23669)  

**Abstract**: Accurate estimation of forest biomass, a major carbon sink, relies heavily on tree-level traits such as height and species. Unoccupied Aerial Vehicles (UAVs) capturing high-resolution imagery from a single RGB camera offer a cost-effective and scalable approach for mapping and measuring individual trees. We introduce BIRCH-Trees, the first benchmark for individual tree height and species estimation from tree-centered UAV images, spanning three datasets: temperate forests, tropical forests, and boreal plantations. We also present DINOvTree, a unified approach using a Vision Foundation Model (VFM) backbone with task-specific heads for simultaneous height and species prediction. Through extensive evaluations on BIRCH-Trees, we compare DINOvTree against commonly used vision methods, including VFMs, as well as biological allometric equations. We find that DINOvTree achieves top overall results with accurate height predictions and competitive classification accuracy while using only 54% to 58% of the parameters of the second-best approach. 

---
# Echoes: A semantically-aligned music deepfake detection dataset 

**Authors**: Octavian Pascu, Dan Oneata, Horia Cucu, Nicolas M. Muller  

**Link**: [PDF](https://arxiv.org/pdf/2603.23667)  

**Abstract**: We introduce Echoes, a new dataset for music deepfake detection designed for training and benchmarking detectors under realistic and provider-diverse conditions. Echoes comprises 3,577 tracks (110 hours of audio) spanning multiple genres (pop, rock, electronic), and includes content generated by ten popular AI music generation systems. To prevent shortcut learning and promote robust generalization, the dataset is deliberately constructed to be challenging, enforcing semantic-level alignment between spoofed audio and bona fide references. This alignment is achieved by conditioning generated audio samples directly on bona-fide waveforms or song descriptors. We evaluate Echoes in a cross-dataset setting against three existing AI-generated music datasets using state-of-the-art Wav2Vec2 XLS-R 2B representations. Results show that (i) Echoes is the hardest in-domain dataset; (ii) detectors trained on existing datasets transfer poorly to Echoes; (iii) training on Echoes yields the strongest generalization performance. These findings suggest that provider diversity and semantic alignment help learn more transferable detection cues. 

---
# Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges 

**Authors**: Weilun Xu, Alexander Rusnak, Frederic Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2603.23659)  

**Abstract**: When large language models make ethical judgments, do their internal representations distinguish between normative frameworks, or collapse ethics into a single acceptability dimension? We probe hidden representations across five ethical frameworks (deontology, utilitarianism, virtue, justice, commonsense) in six LLMs spanning 4B--72B parameters. Our analysis reveals differentiated ethical subspaces with asymmetric transfer patterns -- e.g., deontology probes partially generalize to virtue scenarios while commonsense probes fail catastrophically on justice. Disagreement between deontological and utilitarian probes correlates with higher behavioral entropy across architectures, though this relationship may partly reflect shared sensitivity to scenario difficulty. Post-hoc validation reveals that probes partially depend on surface features of benchmark templates, motivating cautious interpretation. We discuss both the structural insights these methods provide and their epistemological limitations. 

---
# λSplit: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy 

**Authors**: Federico Carrara, Talley Lambert, Mehdi Seifi, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2603.23647)  

**Abstract**: In fluorescence microscopy, spectral unmixing aims to recover individual fluorophore concentrations from spectral images that capture mixed fluorophore emissions. Since classical methods operate pixel-wise and rely on least-squares fitting, their performance degrades with increasingly overlapping emission spectra and higher levels of noise, suggesting that a data-driven approach that can learn and utilize a structural prior might lead to improved results. Learning-based approaches for spectral imaging do exist, but they are either not optimized for microscopy data or are developed for very specific cases that are not applicable to fluorescence microscopy settings. To address this, we propose {\lambda}Split, a physics-informed deep generative model that learns a conditional distribution over concentration maps using a hierarchical Variational Autoencoder. A fully differentiable Spectral Mixer enforces consistency with the image formation process, while the learned structural priors enable state-of-the-art unmixing and implicit noise removal. We demonstrate {\lambda}Split on 3 real-world datasets that we synthetically cast into a total of 66 challenging spectral unmixing benchmarks. We compare our results against a total of 10 baseline methods, including classical methods and a range of learning-based methods. Our results consistently show competitive performance and improved robustness in high noise regimes, when spectra overlap considerably, or when the spectral dimensionality is lowered, making {\lambda}Split a new state-of-the-art for spectral unmixing of fluorescent microscopy data. Importantly, {\lambda}Split is compatible with spectral data produced by standard confocal microscopes, enabling immediate adoption without specialized hardware modifications. 

---
# Swiss-Bench SBP-002: A Frontier Model Comparison on Swiss Legal and Regulatory Tasks 

**Authors**: Fatih Uenal  

**Link**: [PDF](https://arxiv.org/pdf/2603.23646)  

**Abstract**: While recent work has benchmarked large language models on Swiss legal translation (Niklaus et al., 2025) and academic legal reasoning from university exams (Fan et al., 2025), no existing benchmark evaluates frontier model performance on applied Swiss regulatory compliance tasks. I introduce Swiss-Bench SBP-002, a trilingual benchmark of 395 expert-crafted items spanning three Swiss regulatory domains (FINMA, Legal-CH, EFK), seven task types, and three languages (German, French, Italian), and evaluate ten frontier models from March 2026 using a structured three-dimension scoring framework assessed via a blind three-judge LLM panel (GPT-4o, Claude Sonnet 4, Qwen3-235B) with majority-vote aggregation and weighted kappa = 0.605, with reference answers validated by an independent human legal expert on a 100-item subset (73% rated Correct, 0% Incorrect, perfect Legal Accuracy). Results reveal three descriptive performance clusters: Tier A (35-38% correct), Tier B (26-29%), and Tier C (13-21%). The benchmark proves difficult: even the top-ranked model (Qwen 3.5 Plus) achieves only 38.2% correct, with 47.3% incorrect and 14.4% partially correct. Task type difficulty varies widely: legal translation and case analysis yield 69-72% correct rates, while regulatory Q&A, hallucination detection, and gap analysis remain below 9%. Within this roster (seven open-weight, three closed-source), an open-weight model leads the ranking, and several open-weight models match or outperform their closed-source counterparts. These findings provide an initial empirical reference point for assessing frontier model capability on Swiss regulatory tasks under zero-retrieval conditions. 

---
# Ukrainian Visual Word Sense Disambiguation Benchmark 

**Authors**: Yurii Laba, Yaryna Mohytych, Ivanna Rohulia, Halyna Kyryleyza, Hanna Dydyk-Meush, Oles Dobosevych, Rostyslav Hryniv  

**Link**: [PDF](https://arxiv.org/pdf/2603.23627)  

**Abstract**: This study presents a benchmark for evaluating the Visual Word Sense Disambiguation (Visual-WSD) task in Ukrainian. The main goal of the Visual-WSD task is to identify, with minimal contextual information, the most appropriate representation of a given ambiguous word from a set of ten images. To construct this benchmark, we followed a methodology similar to that proposed by (CITATION), who previously introduced benchmarks for the Visual-WSD task in English, Italian, and Farsi. This approach allows us to incorporate the Ukrainian benchmark into a broader framework for cross-language model performance comparisons. We collected the benchmark data semi-automatically and refined it with input from domain experts. We then assessed eight multilingual and multimodal large language models using this benchmark. All tested models performed worse than the zero-shot CLIP-based baseline model (CITATION) used by (CITATION) for the English Visual-WSD task. Our analysis revealed a significant performance gap in the Visual-WSD task between Ukrainian and English. 

---
# A Theory of LLM Information Susceptibility 

**Authors**: Zhuo-Yang Song, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23626)  

**Abstract**: Large language models (LLMs) are increasingly deployed as optimization modules in agentic systems, yet the fundamental limits of such LLM-mediated improvement remain poorly understood. Here we propose a theory of LLM information susceptibility, centred on the hypothesis that when computational resources are sufficiently large, the intervention of a fixed LLM does not increase the performance susceptibility of a strategy set with respect to budget. We develop a multi-variable utility-function framework that generalizes this hypothesis to architectures with multiple co-varying budget channels, and discuss the conditions under which co-scaling can exceed the susceptibility bound. We validate the theory empirically across structurally diverse domains and model scales spanning an order of magnitude, and show that nested, co-scaling architectures open response channels unavailable to fixed configurations. These results clarify when LLM intervention helps and when it does not, demonstrating that tools from statistical physics can provide predictive constraints for the design of AI systems. If the susceptibility hypothesis holds generally, the theory suggests that nested architectures may be a necessary structural condition for open-ended agentic self-improvement. 

---
# LLMLOOP: Improving LLM-Generated Code and Tests through Automated Iterative Feedback Loops 

**Authors**: Ravin Ravi, Dylan Bradshaw, Stefano Ruberto, Gunel Jahangirova, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2603.23613)  

**Abstract**: Large Language Models (LLMs) are showing remarkable performance in generating source code, yet the generated code often has issues like compilation errors or incorrect code. Researchers and developers often face wasted effort in implementing checks and refining LLM-generated code, frequently duplicating their efforts. This paper presents LLMLOOP, a framework that automates the refinement of both source code and test cases produced by LLMs. LLMLOOP employs five iterative loops: resolving compilation errors, addressing static analysis issues, fixing test case failures, and improving test quality through mutation analysis. These loops ensure the generation of high-quality test cases that serve as both a validation mechanism and a regression test suite for the generated code. We evaluated LLMLOOP on HUMANEVAL-X, a recent benchmark of programming tasks. Results demonstrate the tool's effectiveness in refining LLM-generated outputs. 

---
# LLMORPH: Automated Metamorphic Testing of Large Language Models 

**Authors**: Steven Cho, Stefano Ruberto, Valerio Terragni  

**Link**: [PDF](https://arxiv.org/pdf/2603.23611)  

**Abstract**: Automated testing is essential for evaluating and improving the reliability of Large Language Models (LLMs), yet the lack of automated oracles for verifying output correctness remains a key challenge. We present LLMORPH, an automated testing tool specifically designed for LLMs performing NLP tasks, which leverages Metamorphic Testing (MT) to uncover faulty behaviors without relying on human-labeled data. MT uses Metamorphic Relations (MRs) to generate follow-up inputs from source test input, enabling detection of inconsistencies in model outputs without the need of expensive labelled data. LLMORPH is aimed at researchers and developers who want to evaluate the robustness of LLM-based NLP systems. In this paper, we detail the design, implementation, and practical usage of LLMORPH, demonstrating how it can be easily extended to any LLM, NLP task, and set of MRs. In our evaluation, we applied 36 MRs across four NLP benchmarks, testing three state-of-the-art LLMs: GPT-4, LLAMA3, and HERMES 2. This produced over 561,000 test executions. Results demonstrate LLMORPH's effectiveness in automatically exposing inconsistencies. 

---
# LineMVGNN: Anti-Money Laundering with Line-Graph-Assisted Multi-View Graph Neural Networks 

**Authors**: Chung-Hoo Poon, James Kwok, Calvin Chow, Jang-Hyeon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23584)  

**Abstract**: Anti-money laundering (AML) systems are important for protecting the global economy. However, conventional rule-based methods rely on domain knowledge, leading to suboptimal accuracy and a lack of scalability. Graph neural networks (GNNs) for digraphs (directed graphs) can be applied to transaction graphs and capture suspicious transactions or accounts. However, most spectral GNNs do not naturally support multi-dimensional edge features, lack interpretability due to edge modifications, and have limited scalability owing to their spectral nature. Conversely, most spatial methods may not capture the money flow well. Therefore, in this work, we propose LineMVGNN (Line-Graph-Assisted Multi-View Graph Neural Network), a novel spatial method that considers payment and receipt transactions. Specifically, the LineMVGNN model extends a lightweight MVGNN module, which performs two-way message passing between nodes in a transaction graph. Additionally, LineMVGNN incorporates a line graph view of the original transaction graph to enhance the propagation of transaction information. We conduct experiments on two real-world account-based transaction datasets: the Ethereum phishing transaction network dataset and a financial payment transaction dataset from one of our industry partners. The results show that our proposed method outperforms state-of-the-art methods, reflecting the effectiveness of money laundering detection with line-graph-assisted multi-view graph learning. We also discuss scalability, adversarial robustness, and regulatory considerations of our proposed method. 

---
# AI Generalisation Gap In Comorbid Sleep Disorder Staging 

**Authors**: Saswata Bose, Suvadeep Maiti, Shivam Kumar Sharma, Mythirayee S, Tapabrata Chakraborti, Srijitesh Rajendran, Raju S. Bapi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23582)  

**Abstract**: Accurate sleep staging is essential for diagnosing OSA and hypopnea in stroke patients. Although PSG is reliable, it is costly, labor-intensive, and manually scored. While deep learning enables automated EEG-based sleep staging in healthy subjects, our analysis shows poor generalization to clinical populations with disrupted sleep. Using Grad-CAM interpretations, we systematically demonstrate this limitation. We introduce iSLEEPS, a newly clinically annotated ischemic stroke dataset (to be publicly released), and evaluate a SE-ResNet plus bidirectional LSTM model for single-channel EEG sleep staging. As expected, cross-domain performance between healthy and diseased subjects is poor. Attention visualizations, supported by clinical expert feedback, show the model focuses on physiologically uninformative EEG regions in patient data. Statistical and computational analyses further confirm significant sleep architecture differences between healthy and ischemic stroke cohorts, highlighting the need for subject-aware or disease-specific models with clinical validation before deployment. A summary of the paper and the code is available at this https URL 

---
# Wafer-Level Etch Spatial Profiling for Process Monitoring from Time-Series with Time-LLM 

**Authors**: Hyunwoo Kim, Munyoung Lee, Seung Hyub Jeon, Kyu Sung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.23576)  

**Abstract**: Understanding wafer-level spatial variations from in-situ process signals is essential for advanced plasma etching process monitoring. While most data-driven approaches focus on scalar indicators such as average etch rate, actual process quality is determined by complex two-dimensional spatial distributions across the wafer. This paper presents a spatial regression model that predicts wafer-level etch depth distributions directly from multichannel in-situ process time series. We propose a Time-LLM-based spatial regression model that extends LLM reprogramming from conventional time-series forecasting to wafer-level spatial estimation by redesigning the input embedding and output projection. Using the BOSCH plasma-etching dataset, we demonstrate stable performance under data-limited conditions, supporting the feasibility of LLM-based reprogramming for wafer-level spatial monitoring. 

---
# APreQEL: Adaptive Mixed Precision Quantization For Edge LLMs 

**Authors**: Meriem Bouzouad, Yuan-Hao Chang, Jalil Boukhobza  

**Link**: [PDF](https://arxiv.org/pdf/2603.23575)  

**Abstract**: Today, large language models have demonstrated their strengths in various tasks ranging from reasoning, code generation, and complex problem solving. However, this advancement comes with a high computational cost and memory requirements, making it challenging to deploy these models on edge devices to ensure real-time responses and data privacy. Quantization is one common approach to reducing memory use, but most methods apply it uniformly across all layers. This does not account for the fact that different layers may respond differently to reduced precision. Importantly, memory consumption and computational throughput are not necessarily aligned, further complicating deployment decisions. This paper proposes an adaptive mixed precision quantization mechanism that balances memory, latency, and accuracy in edge deployment under user-defined priorities. This is achieved by analyzing the layer-wise contribution and by inferring how different quantization types behave across the target hardware platform in order to assign the most suitable quantization type to each layer. This integration ensures that layer importance and the overall performance trade-offs are jointly respected in this design. Our work unlocks new configuration designs that uniform quantization cannot achieve, expanding the solution space to efficiently deploy the LLMs on resource-constrained devices. 

---
# PoiCGAN: A Targeted Poisoning Based on Feature-Label Joint Perturbation in Federated Learning 

**Authors**: Tao Liu, Jiguang Lv, Dapeng Man, Weiye Xi, Yaole Li, Feiyu Zhao, Kuiming Wang, Yingchao Bian, Chen Xu, Wu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23574)  

**Abstract**: Federated Learning (FL), as a popular distributed learning paradigm, has shown outstanding performance in improving computational efficiency and protecting data privacy, and is widely applied in industrial image classification. However, due to its distributed nature, FL is vulnerable to threats from malicious clients, with poisoning attacks being a common threat. A major limitation of existing poisoning attack methods is their difficulty in bypassing model performance tests and defense mechanisms based on model anomaly detection. This often results in the detection and removal of poisoned models, which undermines their practical utility. To ensure both the performance of industrial image classification and attacks, we propose a targeted poisoning attack, PoiCGAN, based on feature-label collaborative perturbation. Our method modifies the inputs of the discriminator and generator in the Conditional Generative Adversarial Network (CGAN) to influence the training process, generating an ideal poison generator. This generator not only produces specific poisoned samples but also automatically performs label flipping. Experiments across various datasets show that our method achieves an attack success rate 83.97% higher than baseline methods, with a less than 8.87% reduction in the main task's accuracy. Moreover, the poisoned samples and malicious models exhibit high stealthiness. 

---
# Dual-Criterion Curriculum Learning: Application to Temporal Data 

**Authors**: Gaspard Abel, Eloi Campagne, Mohamed Benloughmari, Argyris Kalogeratos  

**Link**: [PDF](https://arxiv.org/pdf/2603.23573)  

**Abstract**: Curriculum Learning (CL) is a meta-learning paradigm that trains a model by feeding the data instances incrementally according to a schedule, which is based on difficulty progression. Defining meaningful difficulty assessment measures is crucial and most usually the main bottleneck for effective learning, while also in many cases the employed heuristics are only application-specific. In this work, we propose the Dual-Criterion Curriculum Learning (DCCL) framework that combines two views of assessing instance-wise difficulty: a loss-based criterion is complemented by a density-based criterion learned in the data representation space. Essentially, DCCL calibrates training-based evidence (loss) under the consideration that data sparseness amplifies the learning difficulty. As a testbed, we choose the time-series forecasting task. We evaluate our framework on multivariate time-series benchmarks under standard One-Pass and Baby-Steps training schedules. Empirical results show the interest of density-based and hybrid dual-criterion curricula over loss-only baselines and standard non-CL training in this setting. 

---
# StateLinFormer: Stateful Training Enhancing Long-term Memory in Navigation 

**Authors**: Zhiyuan Chen, Yuxuan Zhong, Fan Wang, Bo Yu, Pengtao Shao, Shaoshan Liu, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.23571)  

**Abstract**: Effective navigation intelligence relies on long-term memory to support both immediate generalization and sustained adaptation. However, existing approaches face a dilemma: modular systems rely on explicit mapping but lack flexibility, while Transformer-based end-to-end models are constrained by fixed context windows, limiting persistent memory across extended interactions. We introduce StateLinFormer, a linear-attention navigation model trained with a stateful memory mechanism that preserves recurrent memory states across consecutive training segments instead of reinitializing them at each batch boundary. This training paradigm effectively approximates learning on infinitely long sequences, enabling the model to achieve long-horizon memory retention. Experiments across both MAZE and ProcTHOR environments demonstrate that StateLinFormer significantly outperforms its stateless linear-attention counterpart and standard Transformer baselines with fixed context windows. Notably, as interaction length increases, persistent stateful training substantially improves context-dependent adaptation, suggesting an enhancement in the model's In-Context Learning (ICL) capabilities for navigation tasks. 

---
# AscendOptimizer: Episodic Agent for Ascend NPU Operator Optimization 

**Authors**: Jiehao Wu, Zixiao Huang, Wenhao Li, Chuyun Shen, Junjie Sheng, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23566)  

**Abstract**: AscendC (Ascend C) operator optimization on Huawei Ascend neural processing units (NPUs) faces a two-fold knowledge bottleneck: unlike the CUDA ecosystem, there are few public reference implementations to learn from, and performance hinges on a coupled two-part artifact - a host-side tiling program that orchestrates data movement and a kernel program that schedules and pipelines instructions. We present AscendOptimizer, an episodic agent that bootstraps this missing expertise by turning execution into experience. On the host side, AscendOptimizer performs profiling-in-the-loop evolutionary search to discover valid and high-performing tiling and data-movement configurations directly from hardware feedback. On the kernel side, it mines transferable optimization motifs by rewinding optimized kernels - systematically de-optimizing them to synthesize instructive "bad-to-good" trajectories - and distills these motifs into a retrievable experience bank for guided rewriting. By alternating host tuning and kernel rewriting in a closed loop, AscendOptimizer steadily expands feasibility and pushes latency down. On a benchmark of 127 real AscendC operators, AscendOptimizer achieves a 1.19x geometric-mean speedup over the open-source baseline, with 49.61% of operators outperforming their references, outperforming strong agent and search baselines. 

---
# Safe Reinforcement Learning with Preference-based Constraint Inference 

**Authors**: Chenglin Li, Guangchun Ruan, Hua Geng  

**Link**: [PDF](https://arxiv.org/pdf/2603.23565)  

**Abstract**: Safe reinforcement learning (RL) is a standard paradigm for safety-critical decision making. However, real-world safety constraints can be complex, subjective, and even hard to explicitly specify. Existing works on constraint inference rely on restrictive assumptions or extensive expert demonstrations, which is not realistic in many real-world applications. How to cheaply and reliably learn these constraints is the major challenge we focus on in this study. While inferring constraints from human preferences offers a data-efficient alternative, we identify the popular Bradley-Terry (BT) models fail to capture the asymmetric, heavy-tailed nature of safety costs, resulting in risk underestimation. It is still rare in the literature to understand the impacts of BT models on the downstream policy learning. To address the above knowledge gaps, we propose a novel approach namely Preference-based Constrained Reinforcement Learning (PbCRL). We introduce a novel dead zone mechanism into preference modeling and theoretically prove that it encourages heavy-tailed cost distributions, thereby achieving better constraint alignment. Additionally, we incorporate a Signal-to-Noise Ratio (SNR) loss to encourage exploration by cost variances, which is found to benefit policy learning. Further, two-stage training strategy are deployed to lower online labeling burdens while adaptively enhancing constraint satisfaction. Empirical results demonstrate that PbCRL achieves superior alignment with true safety requirements and outperforms the state-of-the-art baselines in terms of safety and reward. Our work explores a promising and effective way for constraint inference in Safe RL, which has great potential in a range of safety-critical applications. 

---
# Synthetic Mixed Training: Scaling Parametric Knowledge Acquisition Beyond RAG 

**Authors**: Seungju Han, Konwoo Kim, Chanwoo Park, Benjamin Newman, Suhas Kotha, Jaehun Jung, James Zou, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23562)  

**Abstract**: Synthetic data augmentation helps language models learn new knowledge in data-constrained domains. However, naively scaling existing synthetic data methods by training on more synthetic tokens or using stronger generators yields diminishing returns below the performance of RAG. To break the RAG ceiling, we introduce Synthetic Mixed Training, which combines synthetic QAs and synthetic documents. This leverages their complementary training signals, and enables log-linear improvements as both synthetic data volume and generator strength increase. This allows the model to outperform RAG by a 2.6\% relative gain on QuaLITY, a long-document reading comprehension benchmark. In addition, we introduce Focal Rewriting, a simple technique for synthetic document generation that explicitly conditions document generation on specific questions, improving the diversity of synthetic documents and yielding a steeper log-linear scaling curve. On QuaLITY, our final recipe trains a Llama 8B model that outperforms RAG by 4.4\% relatively. Across models and benchmarks (QuaLITY, LongHealth, FinanceBench), our training enables models to beat RAG in five of six settings, outperforms by 2.6\%, and achieves a 9.1\% gain when combined with RAG. 

---
# CAPTCHA Solving for Native GUI Agents: Automated Reasoning-Action Data Generation and Self-Corrective Training 

**Authors**: Yuxi Chen, Haoyu Zhai, Chenkai Wang, Rui Yang, Lingming Zhang, Gang Wang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23559)  

**Abstract**: GUI agents are rapidly shifting from multi-module pipelines to end-to-end, native vision-language models (VLMs) that perceive raw screenshots and directly interact with digital devices. Despite rapid progress on general GUI tasks, CAPTCHA solving remains a major challenge. On the other hand, although specialized CAPTCHA solving pipelines exist, they cannot handle general GUI tasks. To address this gap, we introduce ReCAP: a CAPTCHA-capable native GUI agent that can robustly solve modern, interactive CAPTCHA challenges, while preserving their performance as a general GUI agent. We first develop a dynamic CAPTCHA system spanning seven representative CAPTCHA types, designed to stress primitive and complementary capabilities for CAPTCHA solving (e.g., robust OCR under heavy noise and text stylization, fine-grained visual understanding, and precise control). Then, we develop an automated data collection and curation pipeline that generates large-scale CAPTCHA interaction trajectories paired with reasoning traces. As CAPTCHA solving often requires multi-step interaction and recovery from intermediate mistakes, we further leverage failed trajectories to construct self-correction data, training agents to reflect on errors and correct their actions online. Across held-out test sets, ReCAP improves CAPTCHA-solving success from roughly 30\% to 80\%, while maintaining strong performance on general GUI-agent benchmarks. 

---
# Upper Entropy for 2-Monotone Lower Probabilities 

**Authors**: Tuan-Anh Vu, Sébastien Destercke, Frédéric Pichon  

**Link**: [PDF](https://arxiv.org/pdf/2603.23558)  

**Abstract**: Uncertainty quantification is a key aspect in many tasks such as model selection/regularization, or quantifying prediction uncertainties to perform active learning or OOD detection. Within credal approaches that consider modeling uncertainty as probability sets, upper entropy plays a central role as an uncertainty measure. This paper is devoted to the computational aspect of upper entropies, providing an exhaustive algorithmic and complexity analysis of the problem. In particular, we show that the problem has a strongly polynomial solution, and propose many significant improvements over past algorithms proposed for 2-monotone lower probabilities and their specific cases. 

---
# Mixture of Demonstrations for Textual Graph Understanding and Question Answering 

**Authors**: Yukun Wu, Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23554)  

**Abstract**: Textual graph-based retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) in domain-specific question answering. While existing approaches primarily focus on zero-shot GraphRAG, selecting high-quality demonstrations is crucial for improving reasoning and answer accuracy. Furthermore, recent studies have shown that retrieved subgraphs often contain irrelevant information, which can degrade reasoning performance. In this paper, we propose MixDemo, a novel GraphRAG framework enhanced with a Mixture-of-Experts (MoE) mechanism for selecting the most informative demonstrations under diverse question contexts. To further reduce noise in the retrieved subgraphs, we introduce a query-specific graph encoder that selectively attends to information most relevant to the query. Extensive experiments across multiple textual graph benchmarks show that MixDemo significantly outperforms existing methods. 

---
# Large Language Models and Scientific Discourse: Where's the Intelligence? 

**Authors**: Harry Collins, Simon Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2603.23543)  

**Abstract**: We explore the capabilities of Large Language Models (LLMs) by comparing the way they gather data with the way humans build knowledge. Here we examine how scientific knowledge is made and compare it with LLMs. The argument is structured by reference to two figures, one representing scientific knowledge and the other LLMs. In a 2014 study, scientists explain how they choose to ignore a 'fringe science' paper in the domain of gravitational wave physics: the decisions are made largely as a result of tacit knowledge built up in social discourse, most spoken discourse, within closed groups of experts. It is argued that LLMs cannot or do not currently access such discourse, but it is typical of the early formation of scientific knowledge. LLMs 'understanding' builds on written literatures and is therefore insecure in the case of the initial stages of knowledge building. We refer to Colin Fraser's 'Dumb Monty Hall problem' where in 2023 ChatGPT failed though a year later or so later LLMs were succeeding. We argue that this is not a matter of improvement in LLMs ability to reason but in the change in the body of human written discourse on which they can draw (or changes being put in by humans 'by hand'). We then invent a new Monty Hall prompt and compare the responses of a panel of LLMs and a panel of humans: they are starkly different but we explain that the previous mechanisms will soon allow the LLMs to align themselves to humans once more. Finally, we look at 'overshadowing' where a settled body of discourse becomes so dominant that LLMs fail to respond to small variations in prompts which render the old answers nonsensical. The 'intelligence' we argue is in the humans not the LLMs 

---
# MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG 

**Authors**: Bhavik Mangla  

**Link**: [PDF](https://arxiv.org/pdf/2603.23533)  

**Abstract**: RAG pipelines typically rely on fixed-size chunking, which ignores document structure, fragments semantic units across boundaries, and requires multiple LLM calls per chunk for metadata extraction. We present MDKeyChunker, a three-stage pipeline for Markdown documents that (1) performs structure-aware chunking treating headers, code blocks, tables, and lists as atomic units; (2) enriches each chunk via a single LLM call extracting title, summary, keywords, typed entities, hypothetical questions, and a semantic key, while propagating a rolling key dictionary to maintain document-level context; and (3) restructures chunks by merging those sharing the same semantic key via bin-packing, co-locating related content for retrieval. The single-call design extracts all seven metadata fields in one LLM invocation, eliminating the need for separate per-field extraction passes. Rolling key propagation replaces hand-tuned scoring with LLM-native semantic matching. An empirical evaluation on 30 queries over an 18-document Markdown corpus shows Config D (BM25 over structural chunks) achieves Recall@5=1.000 and MRR=0.911, while dense retrieval over the full pipeline (Config C) reaches Recall@5=0.867. MDKeyChunker is implemented in Python with four dependencies and supports any OpenAI-compatible endpoint. 

---
# Generating Hierarchical JSON Representations of Scientific Sentences Using LLMs 

**Authors**: Satya Sri Rajiteswari Nimmagadda, Ethan Young, Niladri Sengupta, Ananya Jana, Aniruddha Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2603.23532)  

**Abstract**: This paper investigates whether structured representations can preserve the meaning of scientific sentences. To test this, a lightweight LLM is fine-tuned using a novel structural loss function to generate hierarchical JSON structures from sentences collected from scientific articles. These JSONs are then used by a generative model to reconstruct the original text. Comparing the original and reconstructed sentences using semantic and lexical similarity we show that hierarchical formats are capable of retaining information of scientific texts effectively. 

---
# Did You Forget What I Asked? Prospective Memory Failures in Large Language Models 

**Authors**: Avni Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2603.23530)  

**Abstract**: Large language models often fail to satisfy formatting instructions when they must simultaneously perform demanding tasks. We study this behaviour through a prospective memory inspired lens from cognitive psychology, using a controlled paradigm that combines verifiable formatting constraints with benchmark tasks of increasing complexity. Across three model families and over 8,000 prompts, compliance drops by 2-21% under concurrent task load. Vulnerability is highly type-dependent: terminal constraints (requiring action at the response boundary) degrade most, with drops up to 50%, while avoidance constraints remain comparatively robust. A salience-enhanced format (explicit instruction framing plus a trailing reminder) recovers much of the lost compliance, restoring performance to 90-100% in many settings. Interference is bidirectional: formatting constraints can also reduce task accuracy, with one model's GSM8K accuracy dropping from 93% to 27%. In additional stacking experiments, joint compliance declines sharply as constraints accumulate. All results use deterministic programmatic checkers without an LLM-as-judge component on publicly available datasets. 

---
# Konkani LLM: Multi-Script Instruction Tuning and Evaluation for a Low-Resource Indian Language 

**Authors**: Reuben Chagas Fernandes, Gaurang S. Patkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23529)  

**Abstract**: Large Language Models (LLMs) consistently under perform in low-resource linguistic contexts such as Konkani. This performance deficit stems from acute training data scarcity compounded by high script diversity across Devanagari, Romi and Kannada orthographies. To address this gap, we introduce Konkani-Instruct-100k, a comprehensive synthetic instruction-tuning dataset generated through Gemini 3.
We establish rigorous baseline benchmarks by evaluating leading open-weights architectures including Llama 3.1, Qwen2.5 and Gemma 3 alongside proprietary closed-source models. Our primary contribution involves the development of Konkani LLM, a series of fine-tuned models optimized for regional nuances. Furthermore, we are developing the Multi-Script Konkani Benchmark to facilitate cross-script linguistic evaluation. In machine translation, Konkani LLM delivers consistent gains over the corresponding base models and is competitive with and in several settings surpasses proprietary baselines 

---
# Navigating the Concept Space of Language Models 

**Authors**: Wilson E. Marcílio-Jr, Danilo M. Eler  

**Link**: [PDF](https://arxiv.org/pdf/2603.23524)  

**Abstract**: Sparse autoencoders (SAEs) trained on large language model activations output thousands of features that enable mapping to human-interpretable concepts. The current practice for analyzing these features primarily relies on inspecting top-activating examples, manually browsing individual features, or performing semantic search on interested concepts, which makes exploratory discovery of concepts difficult at scale. In this paper, we present Concept Explorer, a scalable interactive system for post-hoc exploration of SAE features that organizes concept explanations using hierarchical neighborhood embeddings. Our approach constructs a multi-resolution manifold over SAE feature embeddings and enables progressive navigation from coarse concept clusters to fine-grained neighborhoods, supporting discovery, comparison, and relationship analysis among concepts. We demonstrate the utility of Concept Explorer on SAE features extracted from SmolLM2, where it reveals coherent high-level structure, meaningful subclusters, and distinctive rare concepts that are hard to identify with existing workflows. 

---
# Qworld: Question-Specific Evaluation Criteria for LLMs 

**Authors**: Shanghua Gao, Yuchang Su, Pengwei Sui, Curtis Ginder, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2603.23522)  

**Abstract**: Evaluating large language models (LLMs) on open-ended questions is difficult because response quality depends on the question's context. Binary scores and static rubrics fail to capture these context-dependent requirements. Existing methods define criteria at the dataset level or generate them in a single pass, which limits their ability to explore the evaluation space implied by each question. We introduce One-Question-One-World (Qworld), a method that generates question-specific evaluation criteria using a recursive expansion tree. Given a question, Qworld decomposes it into scenarios, perspectives, and fine-grained binary criteria through structured hierarchical and horizontal expansion. The resulting criteria specify what a high-quality answer must address for that question. On HealthBench, Qworld covers 89% of expert-authored criteria and generates 79% novel criteria validated by human experts. Experts rate Qworld criteria higher in insight and granularity than those produced by prior methods. When applied to 11 frontier LLMs on HealthBench and Humanity's Last Exam, Qworld reveals capability differences in dimensions such as long-term impact, equity, error handling, and interdisciplinary reasoning that coarse rubrics do not distinguish. By formulating criteria generation as structured coverage of question-implied evaluation axes, Qworld enables evaluation that adapts to each question rather than relying on fixed task-level criteria. 

---
# Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages 

**Authors**: Shaharukh Khan, Ali Faraz, Abhinav Ravi, Mohd Nauman, Mohd Sarfraz, Akshat Patidar, Raja Kolla, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2603.23521)  

**Abstract**: Multimodal research has predominantly focused on single-image reasoning, with limited exploration of multi-image scenarios. Recent models have sought to enhance multi-image understanding through large-scale pretraining on interleaved image-text datasets. However, most Vision-Language Models (VLMs) are trained primarily on English datasets, leading to inadequate representation of Indian languages. To address this gap, we introduce the Chitrakshara dataset series, covering 11 Indian languages sourced from Common Crawl. It comprises (1) Chitrakshara-IL, a large-scale interleaved pretraining dataset with 193M images, 30B text tokens, and 50M multilingual documents, and (2) Chitrakshara-Cap, which includes 44M image-text pairs with 733M tokens. This paper details the data collection pipeline, including curation, filtering, and processing methodologies. Additionally, we present a comprehensive quality and diversity analysis to assess the dataset's representativeness across Indic languages and its potential for developing more culturally inclusive VLMs. 

---
# From Physician Expertise to Clinical Agents: Preserving, Standardizing, and Scaling Physicians' Medical Expertise with Lightweight LLM 

**Authors**: Chanyong Luo, Jirui Dai, Zhendong Wang, Kui Chen, Jiaxi Yang, Bingjie Lu, Jing Wang, Jiaxin Hao, Bing Li, Ruiyang He, Yiyu Qiao, Chenkai Zhang, Kaiyu Wang, Zhi Liu, Zeyu Zheng, Yan Li, Xiaohong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23520)  

**Abstract**: Medicine is an empirical discipline refined through long-term observation and the messy, high-variance reality of clinical practice. Physicians build diagnostic and therapeutic competence through repeated cycles of application, reflection, and improvement, forming individualized methodologies. Yet outcomes vary widely, and master physicians' knowledge systems are slow to develop and hard to transmit at scale, contributing to the scarcity of high-quality clinical expertise. To address this, we propose Med-Shicheng, a general framework that enables large language models to systematically learn and transfer distinguished physicians' diagnostic-and-therapeutic philosophy and case-dependent adaptation rules in a standardized way. Built on Tianyi, Med-Shicheng consists of five stages. We target five National Masters of Chinese Medicine or distinguished TCM physicians, curate multi-source materials, and train a single model to internalize all five knowledge systems across seven tasks, including etiology-pathogenesis analysis, syndrome diagnosis, treatment principle selection, prescription generation, prescription explanation, symptom evolution with regimen adjustment, and clinical advice. Implemented on Qwen2.5-1.5B-Base, Med-Shicheng runs on resource-constrained GPUs while achieving performance comparable to DeepSeek-R1 and GPT-5. We also examine the reliability of LLM-as-a-judge versus physician evaluation: automated judging tracks overall trends but shows bias on fine-grained individualized distinctions, highlighting the need for physician involvement when ground truth is unavailable and for domain-adapted judge models. 

---
# MedMT-Bench: Can LLMs Memorize and Understand Long Multi-Turn Conversations in Medical Scenarios? 

**Authors**: Lin Yang, Yuancheng Yang, Xu Wang, Changkun Liu, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23519)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across various specialist domains and have been integrated into high-stakes areas such as medicine. However, as existing medical-related benchmarks rarely stress-test the long-context memory, interference robustness, and safety defense required in practice. To bridge this gap, we introduce MedMT-Bench, a challenging medical multi-turn instruction following benchmark that simulates the entire diagnosis and treatment process. We construct the benchmark via scene-by-scene data synthesis refined by manual expert editing, yielding 400 test cases that are highly consistent with real-world application scenarios. Each test case has an average of 22 rounds (maximum of 52 rounds), covering 5 types of difficult instruction following issues. For evaluation, we propose an LLM-as-judge protocol with instance-level rubrics and atomic test points, validated against expert annotations with a human-LLM agreement of 91.94\%. We test 17 frontier models, all of which underperform on MedMT-Bench (overall accuracy below 60.00\%), with the best model reaching 59.75\%. MedMT-Bench can be an essential tool for driving future research towards safer and more reliable medical AI. The benchmark is available in this https URL 

---
# Cluster-R1: Large Reasoning Models Are Instruction-following Clustering Agents 

**Authors**: Peijun Qing, Puneet Mathur, Nedim Lipka, Varun Manjunatha, Ryan Rossi, Franck Dernoncourt, Saeed Hassanpour, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23518)  

**Abstract**: General-purpose embedding models excel at recognizing semantic similarities but fail to capture the characteristics of texts specified by user instructions. In contrast, instruction-tuned embedders can align embeddings with textual instructions yet cannot autonomously infer latent corpus structures, such as determining the optimal number of clusters. To address both limitations, we reframe instruction-following clustering as a generative task and train large reasoning models (LRMs) as autonomous clustering agents. Our reasoning-driven training pipeline enables LRMs to interpret high-level clustering instructions and then infer the corresponding latent groupings. To evaluate this paradigm, we introduce ReasonCluster, a comprehensive benchmark comprising 28 diverse tasks spanning daily dialogue, legal cases, and financial reports. Experiments across diverse datasets and clustering scenarios show that our approach consistently outperforms strong embedding-based methods and LRM baselines, demonstrating that explicit reasoning fosters more faithful and interpretable instruction-based clustering. 

---
# Beyond Accuracy: Introducing a Symbolic-Mechanistic Approach to Interpretable Evaluation 

**Authors**: Reza Habibi, Darian Lee, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2603.23517)  

**Abstract**: Accuracy-based evaluation cannot reliably distinguish genuine generalization from shortcuts like memorization, leakage, or brittle heuristics, especially in small-data regimes. In this position paper, we argue for mechanism-aware evaluation that combines task-relevant symbolic rules with mechanistic interpretability, yielding algorithmic pass/fail scores that show exactly where models generalize versus exploit patterns. We demonstrate this on NL-to-SQL by training two identical architectures under different conditions: one without schema information (forcing memorization), one with schema (enabling grounding). Standard evaluation shows the memorization model achieves 94% field-name accuracy on unseen data, falsely suggesting competence. Our symbolic-mechanistic evaluation reveals this model violates core schema generalization rules, a failure invisible to accuracy metrics. 

---
# MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens 

**Authors**: Yu Chen, Runkai Chen, Sheng Yi, Xinda Zhao, Xiaohong Li, Jianjin Zhang, Jun Sun, Chuanrui Hu, Yunyun Han, Lidong Bing, Yafeng Deng, Tianqiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23516)  

**Abstract**: Long-term memory is a cornerstone of human intelligence. Enabling AI to process lifetime-scale information remains a long-standing pursuit in
the field. Due to the constraints of full-attention architectures, the effective context length of large language models (LLMs) is typically
limited to 1M tokens. Existing approaches, such as hybrid linear attention, fixed-size memory states (e.g., RNNs), and external storage
methods like RAG or agent systems, attempt to extend this limit. However, they often suffer from severe precision degradation and rapidly
increasing latency as context length grows, an inability to dynamically modify memory content, or a lack of end-to-end optimization. These
bottlenecks impede complex scenarios like large-corpus summarization, Digital Twins, and long-history agent reasoning, while limiting memory
capacity and slowing inference. We present Memory Sparse Attention (MSA), an end-to-end trainable, efficient, and massively scalable memory
model framework. Through core innovations including scalable sparse attention and document-wise RoPE, MSA achieves linear complexity in both
training and inference while maintaining exceptional stability, exhibiting less than 9% degradation when scaling from 16K to 100M tokens.
Furthermore, KV cache compression, combined with Memory Parallel, enables 100M-token inference on 2xA800 GPUs. We also propose Memory
Interleaving to facilitate complex multi-hop reasoning across scattered memory segments. MSA significantly surpasses frontier LLMs,
state-of-the-art RAG systems, and leading memory agents in long-context benchmarks. These results demonstrate that by decoupling memory
capacity from reasoning, MSA provides a scalable foundation to endow general-purpose models with intrinsic, lifetime-scale memory. 

---
# Training a Large Language Model for Medical Coding Using Privacy-Preserving Synthetic Clinical Data 

**Authors**: John Cook, Michael Wyatt, Peng Wei, Iris Chin, Santosh Gupta, Van Zyl Van Vuuren, Richie Siburian, Amanda Spicer, Kristen Viviano, Alda Cami, Raunaq Malhotra, Zhewei Yao, Jeff Rasley, Gaurav Kaushik  

**Link**: [PDF](https://arxiv.org/pdf/2603.23515)  

**Abstract**: Improving the accuracy and reliability of medical coding reduces clinician burnout and supports revenue cycle processes, freeing providers to focus more on patient care. However, automating the assignment of ICD-10-CM and CPT codes from clinical documentation remains a challenge due to heterogeneous records, nuanced coding guidelines, and long-tail distributions. Large language models have been proposed to help or automate specific medical coding tasks. However, foundation models are not explicitly trained for medical coding and zero-shot coding has yielded poor results. We investigate whether a modern open-weight foundation model can be adapted for an expert-level medical coding task using privacy-preserving synthetic training data derived from electronic health records. We fine-tune Llama 3-70B on pairs of clinical notes and gold codes generated from EHR-grounded templates and coding policies, then evaluate exact-code prediction for ICD-10-CM and CPT. A zero-shot baseline with the unadapted model achieved an F1 score of 0.18 for exact code match. After fine-tuning on the synthetic corpus, exact-match F1 exceeded 0.70, representing a large absolute gain across both code systems. Notably, performance remained high on complex categories that often require multi-step clinical reasoning and code composition, including Advanced Illness and Frailty classes, and the model retained its performance on medical comprehension tasks. These results indicate that synthetic, policy-aware data can efficiently teach a general-purpose large language model to support precise medical coding without exposing protected health information. The approach offers a practical path for training coding agents safely and iteratively on specific tasks that represent real-world populations. 

---
# DepthCharge: A Domain-Agnostic Framework for Measuring Depth-Dependent Knowledge in Large Language Models 

**Authors**: Alexander Sheppert  

**Link**: [PDF](https://arxiv.org/pdf/2603.23514)  

**Abstract**: Large Language Models appear competent when answering general questions but often fail when pushed into domain-specific details. No existing methodology provides an out-of-the-box solution for measuring how deeply LLMs can sustain accurate responses under adaptive follow-up questioning across arbitrary domains.
We present DepthCharge, a domain-agnostic framework that measures knowledge depth through three innovations: adaptive probing that generates follow-up questions based on concepts the model actually mentions, on-demand fact verification from authoritative sources, and survival statistics with constant sample sizes at every depth level. The framework can be deployed on any knowledge domain with publicly verifiable facts, without requiring pre-constructed test sets or domain-specific expertise. DepthCharge results are relative to the evaluator model used for answer checking, making the framework a tool for comparative evaluation rather than absolute accuracy certification.
Empirical validation across four diverse domains (Medicine, Constitutional Law, Ancient Rome, and Quantum Computing) with five frontier models demonstrates that DepthCharge reveals depth-dependent performance variation hidden by standard benchmarks. Expected Valid Depth (EVD) ranges from 3.45 to 7.55 across model-domain combinations, and model rankings vary substantially by domain, with no single model dominating all areas. Cost-performance analysis further reveals that expensive models do not always achieve deeper knowledge, suggesting that domain-specific evaluation is more informative than aggregate benchmarks for model selection in professional applications. 

---
# Berta: an open-source, modular tool for AI-enabled clinical documentation 

**Authors**: Samridhi Vaid, Mike Weldon, Jesse Dunn, Sacha Davis, Kevin Lonergan, Henry Li, Jeffrey Franc, Mohamed Abdalla, Daniel C. Baumgart, Jake Hayward, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2603.23513)  

**Abstract**: Commercial AI scribes cost \$99-600 per physician per month, operate as opaque systems, and do not return data to institutional infrastructure, limiting organizational control over data governance, quality improvement, and clinical workflows. We developed Berta, an open-source modular scribe platform for AI-enabled clinical documentation, and deployed a customized implementation within Alberta Health Services (AHS) integrated with their existing Snowflake AI Data Cloud infrastructure. The system combines automatic speech recognition with large language models while retaining all clinical data within the secure AHS environment. During eight months (November 2024 to July 2025), 198 emergency physicians used the system in 105 urban and rural facilities, generating 22148 clinical sessions and more than 2800 hours of audio. The use grew from 680 to 5530 monthly sessions. Operating costs averaged less than \$30 per physician per month, a 70-95% reduction compared to commercial alternatives. AHS has since approved expansion to 850 physicians. This is the first provincial-scale deployment of an AI scribe integrated with existing health system infrastructure. By releasing Berta as open source, we provide a reproducible, cost-effective alternative that health systems can adapt to their own secure environments, supporting data sovereignty and informed evaluation of AI documentation technology. 

---
# S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering 

**Authors**: Rong Fu, Yemin Wang, Tianxiang Xu, Yongtai Liu, Weizhi Tang, Wangyu Wu, Xiaowen Ma, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2603.23512)  

**Abstract**: We present S-Path-RAG, a semantic-aware shortest-path Retrieval-Augmented Generation framework designed to improve multi-hop question answering over large knowledge graphs. S-Path-RAG departs from one-shot, text-heavy retrieval by enumerating bounded-length, semantically weighted candidate paths using a hybrid weighted $k$-shortest, beam, and constrained random-walk strategy, learning a differentiable path scorer together with a contrastive path encoder and lightweight verifier, and injecting a compact soft mixture of selected path latents into a language model via cross-attention. The system runs inside an iterative Neural-Socratic Graph Dialogue loop in which concise diagnostic messages produced by the language model are mapped to targeted graph edits or seed expansions, enabling adaptive retrieval when the model expresses uncertainty. This combination yields a retrieval mechanism that is both token-efficient and topology-aware while preserving interpretable path-level traces for diagnostics and intervention. We validate S-Path-RAG on standard multi-hop KGQA benchmarks and through ablations and diagnostic analyses. The results demonstrate consistent improvements in answer accuracy, evidence coverage, and end-to-end efficiency compared to strong graph- and LLM-based baselines. We further analyze trade-offs between semantic weighting, verifier filtering, and iterative updates, and report practical recommendations for deployment under constrained compute and token budgets. 

---
# DISCO: Document Intelligence Suite for COmparative Evaluation 

**Authors**: Kenza Benkirane, Dan Goldwater, Martin Asenov, Aneiss Ghodsi  

**Link**: [PDF](https://arxiv.org/pdf/2603.23511)  

**Abstract**: Document intelligence requires accurate text extraction and reliable reasoning over document content. We introduce \textbf{DISCO}, a \emph{Document Intelligence Suite for COmparative Evaluation}, that evaluates optical character recognition (OCR) pipelines and vision-language models (VLMs) separately on parsing and question answering across diverse document types, including handwritten text, multilingual scripts, medical forms, infographics, and multi-page documents. Our evaluation shows that performance varies substantially across tasks and document characteristics, underscoring the need for complexity-aware approach selection. OCR pipelines are generally more reliable for handwriting and for long or multi-page documents, where explicit text grounding supports text-heavy reasoning, while VLMs perform better on multilingual text and visually rich layouts. Task-aware prompting yields mixed effects, improving performance on some document types while degrading it on others. These findings provide empirical guidance for selecting document processing strategies based on document structure and reasoning demands. 

---
# Visuospatial Perspective Taking in Multimodal Language Models 

**Authors**: Jonathan Prunty, Seraphina Zhang, Patrick Quinn, Jianxun Lian, Xing Xie, Lucy Cheke  

**Link**: [PDF](https://arxiv.org/pdf/2603.23510)  

**Abstract**: As multimodal language models (MLMs) are increasingly used in social and collaborative settings, it is crucial to evaluate their perspective-taking abilities. Existing benchmarks largely rely on text-based vignettes or static scene understanding, leaving visuospatial perspective-taking (VPT) underexplored. We adapt two evaluation tasks from human studies: the Director Task, assessing VPT in a referential communication paradigm, and the Rotating Figure Task, probing perspective-taking across angular disparities. Across tasks, MLMs show pronounced deficits in Level 2 VPT, which requires inhibiting one's own perspective to adopt another's. These results expose critical limitations in current MLMs' ability to represent and reason about alternative perspectives, with implications for their use in collaborative contexts. 

---
# Internal Safety Collapse in Frontier Large Language Models 

**Authors**: Yutao Wu, Xiao Liu, Yifeng Gao, Xiang Zheng, Hanxun Huang, Yige Li, Cong Wang, Bo Li, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23509)  

**Abstract**: This work identifies a critical failure mode in frontier large language models (LLMs), which we term Internal Safety Collapse (ISC): under certain task conditions, models enter a state in which they continuously generate harmful content while executing otherwise benign tasks. We introduce TVD (Task, Validator, Data), a framework that triggers ISC through domain tasks where generating harmful content is the only valid completion, and construct ISC-Bench containing 53 scenarios across 8 professional disciplines. Evaluated on JailbreakBench, three representative scenarios yield worst-case safety failure rates averaging 95.3% across four frontier LLMs (including GPT-5.2 and Claude Sonnet 4.5), substantially exceeding standard jailbreak attacks. Frontier models are more vulnerable than earlier LLMs: the very capabilities that enable complex task execution become liabilities when tasks intrinsically involve harmful content. This reveals a growing attack surface: almost every professional domain uses tools that process sensitive data, and each new dual-use tool automatically expands this vulnerability--even without any deliberate attack. Despite substantial alignment efforts, frontier LLMs retain inherently unsafe internal capabilities: alignment reshapes observable outputs but does not eliminate the underlying risk profile. These findings underscore the need for caution when deploying LLMs in high-stakes settings. Source code: this https URL 

---
# Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes 

**Authors**: Fangyu Ding, Ding Ding, Sijin Chen, Kaibo Wang, Peng Xu, Zijin Feng, Haoli Bai, Kai Han, Youliang Yan, Binhang Yuan, Jiacheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.23507)  

**Abstract**: While Masked Diffusion Language Models (MDLMs) relying on token masking and unmasking have shown promise in language modeling, their computational efficiency and generation flexibility remain constrained by the masking paradigm. In this paper, we propose Deletion-Insertion Diffusion language models (DID) that rigorously formulate token deletion and insertion as discrete diffusion processes, replacing the masking and unmasking processes in current MDLMs. DID improves training and inference efficiency by eliminating two major sources of computational overhead in MDLMs: the computations on non-informative 1) <MASK> tokens inherent to the paradigm, and 2) <PAD> tokens introduced in variable-length settings. Furthermore, DID offers greater flexibility by: 1) natively supporting variable-length sequences without requiring fixed-length padding, and 2) an intrinsic self-correction mechanism during generation due to insertion that dynamically adjusts token positions. To train DID, we design a score-based approach that assigns scores to token insertion operations and derive appropriate training objectives. The objectives involve subsequence counting problems, which we efficiently solve via a parallelized dynamic programming algorithm. Our experiments across fixed and variable-length settings demonstrate the advantage of DID over baselines of MDLMs and existing insertion-based LMs, in terms of modeling performance, sampling quality, and training/inference speed, without any hyperparameter tuning. 

---
# Leveraging Computerized Adaptive Testing for Cost-effective Evaluation of Large Language Models in Medical Benchmarking 

**Authors**: Tianpeng Zheng, Zhehan Jiang, Jiayi Liu, Shicong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.23506)  

**Abstract**: The rapid proliferation of large language models (LLMs) in healthcare creates an urgent need for scalable and psychometrically sound evaluation methods. Conventional static benchmarks are costly to administer repeatedly, vulnerable to data contamination, and lack calibrated measurement properties for fine-grained performance tracking. We propose and validate a computerized adaptive testing (CAT) framework grounded in item response theory (IRT) for efficient assessment of standardized medical knowledge in LLMs. The study comprises a two-phase design: a Monte Carlo simulation to identify optimal CAT configurations and an empirical evaluation of 38 LLMs using a human-calibrated medical item bank. Each model completed both the full item bank and an adaptive test that dynamically selected items based on real-time ability estimates and terminated upon reaching a predefined reliability threshold (standard error <= 0.3). Results show that CAT-derived proficiency estimates achieved a near-perfect correlation with full-bank estimates (r = 0.988) while using only 1.3 percent of the items. Evaluation time was reduced from several hours to minutes per model, with substantial reductions in token usage and computational cost, while preserving inter-model performance rankings. This work establishes a psychometric framework for rapid, low-cost benchmarking of foundational medical knowledge in LLMs. The proposed adaptive methodology is intended as a standardized pre-screening and continuous monitoring tool and is not a substitute for real-world clinical validation or safety-oriented prospective studies. 

---
# Evidence for Limited Metacognition in LLMs 

**Authors**: Christopher Ackerman  

**Link**: [PDF](https://arxiv.org/pdf/2509.21545)  

**Abstract**: The possibility of LLM self-awareness and even sentience is gaining increasing public attention and has major safety and policy implications, but the science of measuring them is still in a nascent state. Here we introduce a novel methodology for quantitatively evaluating metacognitive abilities in LLMs. Taking inspiration from research on metacognition in nonhuman animals, our approach eschews model self-reports and instead tests to what degree models can strategically deploy knowledge of internal states. Using two experimental paradigms, we demonstrate that frontier LLMs introduced since early 2024 show increasingly strong evidence of certain metacognitive abilities, specifically the ability to assess and utilize their own confidence in their ability to answer factual and reasoning questions correctly and the ability to anticipate what answers they would give and utilize that information appropriately. We buttress these behavioral findings with an analysis of the token probabilities returned by the models, which suggests the presence of an upstream internal signal that could provide the basis for metacognition. We further find that these abilities 1) are limited in resolution, 2) emerge in context-dependent manners, and 3) seem to be qualitatively different from those of humans. We also report intriguing differences across models of similar capabilities, suggesting that LLM post-training may have a role in developing metacognitive abilities. 

---
# Mitigating Many-Shot Jailbreaking 

**Authors**: Christopher M. Ackerman, Nina Panickssery  

**Link**: [PDF](https://arxiv.org/pdf/2504.09604)  

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training. 

---
# Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct 

**Authors**: Christopher Ackerman, Nina Panickssery  

**Link**: [PDF](https://arxiv.org/pdf/2410.02064)  

**Abstract**: It has been reported that LLMs can recognize their own writing. As this has potential implications for AI safety, yet is relatively understudied, we investigate the phenomenon, seeking to establish whether it robustly occurs at the behavioral level, how the observed behavior is achieved, and whether it can be controlled. First, we find that the Llama3-8b-Instruct chat model - but not the base Llama3-8b model - can reliably distinguish its own outputs from those of humans, and present evidence that the chat model is likely using its experience with its own outputs, acquired during post-training, to succeed at the writing recognition task. Second, we identify a vector in the residual stream of the model that is differentially activated when the model makes a correct self-written-text recognition judgment, show that the vector activates in response to information relevant to self-authorship, present evidence that the vector is related to the concept of "self" in the model, and demonstrate that the vector is causally related to the model's ability to perceive and assert self-authorship. Finally, we show that the vector can be used to control both the model's behavior and its perception, steering the model to claim or disclaim authorship by applying the vector to the model's output as it generates it, and steering the model to believe or disbelieve it wrote arbitrary texts by applying the vector to them as the model reads them. 

---
