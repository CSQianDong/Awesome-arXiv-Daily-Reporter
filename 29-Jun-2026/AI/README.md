# Agent-Native Immune System: Architecture, Taxonomy, and Engineering 

**Authors**: Bo Shen, Lifeng Chang, Tianyuan Wei, Yunpeng Li, Feng Shi, Yichen Han, Peijie Gao, Shiyi Kuang, Xin Chang, Dehui Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28270)  

**Abstract**: The transition from static chat bots to autonomous agents--equipped with persistent memory, tool-use protocols, and multi-agent collaboration--has fundamentally expanded the AI threat landscape. Current defense mechanisms, such as perimeter security and training-time alignment, remain external to the agent's active reasoning loop. Consequently, they fall short: a fully aligned agent remains highly vulnerable to runtime hijacking via memory poisoning, tool-chain manipulation, or multi-agent protocol attacks. To address this critical gap, we introduce the Agent-Native Immune System (ANIS), the first biologically inspired, endogenous defense architecture embedded directly within the agent's cognitive loop. Our framework presents four primary contributions. First, we design a six-layer Immune Tower (L0-L5), distinctly incorporating Barrier Immunity (L1) as a non-cognitive, physical-and-logical isolation layer. Second, we establish a unified taxonomy of Agent Viruses and Agent Vaccines, formalizing the critical distinction between superficial non-parametric defenses and robust parametric vaccines. Third, we conceptualize the Harness Triad--Meta, Self, and Auto--a self-monitoring, meta-cognitive automation backbone that drives Continual Immune Learning (CIL), enabling vaccines to dynamically adapt to novel threats. Finally, we establish a rigorous theoretical demarcation between model alignment and agent immunity: while alignment provides a static "constitutional" value foundation during training, ANIS serves as the dynamic "law enforcement" mechanism during runtime. We conclude by framing open challenges for the field, including immune protocol standardization, novel evaluation metrics such as the Autoimmunity Rate (false-positive intervention rate), and the co-evolutionary dynamics between pathogens and vaccines within collective intelligence ecosystems. 

---
# Tandem Reinforcement Learning with Verifiable Rewards 

**Authors**: Difan Jiao, Raghav Singhal, Robert West, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2606.28166)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly improved the reasoning capability of large language models, reaching expert or even superhuman performance in domains such as competition math. However, whether weaker agents and humans can actually harness this capability is far less certain, with RLVR documented to drift reasoning toward idiosyncratic patterns such as poor readability and language mixing. Tandem training is a recently introduced paradigm that targets this compatibility problem: a trained, stronger senior co-generates each rollout with a frozen, weaker junior, and the two are rewarded as a team, so the senior is pushed to reason in ways the junior can follow. Yet this paradigm has so far been demonstrated only in proof-of-concept settings, leaving open whether it scales to the long chains of thought of the modern RLVR pipeline. In this work, we propose Tandem Reinforcement Learning (TRL), which carries the tandem training paradigm into RLVR. In TRL, the senior and a frozen junior alternate stochastically to co-generate the reasoning, the resulting generation is rewarded, and the standard GRPO loss is applied to the senior. Training Qwen3-4B-Instruct on competition math, we find that TRL matches vanilla GRPO on solo reasoning capability while three properties emerge together from the same rollout structure: stronger handoff robustness with the junior, reduced distributional drift from the junior, and a chain-of-thought more legible to the junior. Our results demonstrate a promising route for RLVR with practical payoffs in multi-model communication and human compatibility. 

---
# AI-Driven Synthesis for High-Tech System Design: Automating Innovation 

**Authors**: Luuk Oerlemans, Steven Westerhof, Theo Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2606.28126)  

**Abstract**: This article addresses the combinatorial complexity inherent in modern high-tech system design by presenting automation-in-design (AiD) as a transformative paradigm. We propose computational design synthesis (CDS), a framework utilising deep learning and generative AI to automate the creation of novel systems. Two case studies (e-drive system design and spatial dimensioning problem) serve as proof-points for this approach. The AI-driven methods used in the case studies represent a fundamental shift in engineering, advancing from simulation-based optimisation towards autonomous design with minimal human supervision. 

---
# Ontology-Guided Evidence Path Inference for Multi-hop Knowledge Graph Question Answering 

**Authors**: Yongxue Shan, Meihan Wu, Cundi Fang, Jie Peng, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28076)  

**Abstract**: Knowledge graph question answering (KGQA) aims to answer natural-language questions by reasoning over structured facts. Existing multi-hop KGQA methods mainly rely on topic-centered expansion, which faces two key challenges: the search space rapidly grows with noisy mixed-type paths, and retrieved paths may fail to satisfy the semantic constraints of complex questions. To address these challenges, we propose OPI, an ontology-guided evidence path inference framework for multi-hop KGQA. OPI introduces a relation-centric ontology graph to capture the head-tail type constraints of relations, providing a compact interface for answer-side constraints. Based on this ontology graph, OPI first introduces a bidirectional retrieval mechanism by mapping the predicted answer type to compatible final-hop relations and combining topic-side prefix expansion with answer-side final-hop matching, thereby suppressing noisy mixed-type expansion. OPI further adopts an iterative refinement strategy to reassess retrieved paths and candidate answers under the question context, filtering type-compatible but question-irrelevant evidence for more reliable answer prediction. Experiments on WebQSP, CWQ, and MetaQA show that OPI substantially reduces the search space, improves Hit@1/F1 by 4.6/5.0 points on WebQSP and 8.9/3.3 points on CWQ over the strongest prior results, and achieves near-saturated Hit@1 on MetaQA with the retrieval module alone. 

---
# JD Oxygen AI Item Center (Oxygen AIIC) V1: An Industrial-Scale LLM/VLM-Centric Solution for Item Understanding, Management, and Applications 

**Authors**: Oxygen AIIC, Chan Long, Chao Liu, Chaofan Chen, Chaohui Dong, Chunyuan Guo, Danping Liu, Debin Liu, Deping Xiang, Fulai Xu, Guangyue Liu, Hao Li, Huichun Hu, Jian Yang, Jianan Wang, Jianbo Zhao, Jiaoyang Li, Jiaxing Wang, Jinglong Li, Jinjin Guo, Jun Fang, Jun Liu, Kai Zhou, Li Wang, Lili Gao, Liying Chen, Luning Yang, Mengdi Zhou, Pengzhang Liu, Qi Lv, Qianyun Wang, Qixia Jiang, Ruyue Li, Shimu Liang, Shuxing Wang, Sijie Zhang, Siqi Li, Tianhao Gao, Wang Ke, Weihu Huang, Wencan Lai, Wenjie Zhang, Xiaohui Zhang, Xiaojing Dong, Ya Liu, Yifeng Zhang, Yixiang Wang, Yongtai Zhang, Yongyi Liao, Zhaoru Chen, Zhen Chen, Zhiyong Ma, Zhiyuan Liu, Zhongwei Liu, Ziyan Xing  

**Link**: [PDF](https://arxiv.org/pdf/2606.28070)  

**Abstract**: this http URL, one of the world's largest e-commerce platforms, serves over 700 million active users and millions of merchants, with a catalog of tens of billions of SKUs. At this scale, high-quality, structured item knowledge underpins a better consumer experience, lower management costs, and higher operational efficiency-yet producing and serving it poses three industrial-scale challenges: fast-emerging concepts, high-quality knowledge production for massive SKUs, and diverse downstream requirements. To address these challenges, we present the JD Oxygen AI Item Center (Oxygen AIIC), an industrial-scale platform built on LLMs/VLMs for item-knowledge production and service. Oxygen AIIC is built around four core pillars: (i) ontology engineering driven by efficient human-AI collaboration, which supports the dynamic evolution and agile expansion of an ontology with millions of entries; (ii) a "Semantic Search then Discrimination"(S2D) knowledge identification architecture that, combined with throughput improvement strategies, enables scalable, extensible, and high-throughput AI Item Library production for tens of billions of SKUs; (iii) self-evolving item-understanding LLMs/VLMs that improve in a stable and controllable manner, enabling knowledge production with 94.2% precision and 82.8% recall; and (iv) a unified item tunnel that serves as the data and service hub. Oxygen AIIC now covers tens of thousands of JD categories and processes hundreds of millions of item updates per day on Huawei Ascend NPUs. It has accumulated hundreds of billions of item-knowledge assets. Deployed across core business scenarios-including search, recommendation, operations, category planning-Oxygen AIIC has delivered measurable gains at scale. Search-traffic coverage reaches 80.4%, item-information quality issues drop by 37%, the automated fill rate of core attributes during item listing exceeds 80%. 

---
# Lifted Causal Inference 

**Authors**: Malte Luttermann, Tanya Braun, Ralf Möller, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2606.28024)  

**Abstract**: Lifted inference exploits indistinguishabilities in probabilistic graphical models by using a representative for indistinguishable objects, thereby speeding up query answering while maintaining exact answers. In this article, we show how lifting can be applied to efficiently compute causal effects in relational domains. More specifically, we introduce parametric causal factor graphs (PCFGs) to incorporate causal knowledge in lifted models and give a formal semantics of interventions therein. We further present the Lifted Causal Inference (LCI) algorithm to compute causal effects on a lifted level, thereby drastically speeding up causal inference compared to propositional inference, e.g., in causal Bayesian networks. In addition, we present partially directed parametric causal factor graphs (PD-PCFGs) as a generalisation of PCFGs to handle partial causal knowledge and extend LCI to perform lifted causal inference in a PD-PCFG, thereby extending the applicability of lifted causal inference to a broader range of models requiring less prior knowledge about causal relationships. 

---
# RelBall: Relation Ball with Quaternion Rotation for Knowledge Graph Completion 

**Authors**: Yike Liu, Peijia Xie, Chao He, Huiling Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27967)  

**Abstract**: Real-world knowledge graphs are often incomplete, lacking many valid facts. Knowledge Graph Completion (KGC) aims to predict missing links using known triples, thereby enhancing graph coverage. A key challenge is modeling diverse relational patterns such as symmetry, antisymmetry, inversion, composition and semantic hierarchy. Existing models such as RotatE can capture symmetric, antisymmetric, inverse, and commutative composition patterns, yet struggle with non-commutative composition. Rotate3D addresses this by introducing non-commutativity via three-dimensional rotations, but still fails to capture the semantic hierarchies prevalent in knowledge graphs. Moreover, both models cannot effectively model one-to-many relations. To overcome these limitations, we propose RelBall, which extends Rotate3D with two innovations. First, our model introduces modulus transformation to model hierarchies, driving abstract concepts toward smaller moduli and concrete instances toward larger ones. Second, it introduces a tail-centric relation ball to model one-to-one, one-to-many, many-to-one, and many-to-many relations. RelBall offers the following advantages: (1) coverage of all relational patterns, including the ones mentioned above; (2) an interpretable hierarchical representation where the modulus directly reflect semantic levels; (3) support for one-to-one, one-to-many, many-to-one, and many-to-many relations. Experiments on multiple datasets demonstrate RelBall's competitive link prediction performance against various baselines. 

---
# Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing 

**Authors**: Can Li, Ting Zhang, Junbo Zhao, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27926)  

**Abstract**: Geometry Problem Solving have increasingly adopt the neuro-symbolic paradigm, combining neural intuition with symbolic rigor. However, current frameworks suffer from severe bottlenecks in two core stages: autoformalization, which treats multimodal translation as a static task decoupled from downstream solver compatibility, and theorem prediction, where solvers frequently hit a deductive impasse due to fixed rule libraries. To address these, we propose SD-GPS, a solver-driven framework that treats the symbolic solver as an execution oracle throughout both formalization and deduction. First, Solver-Driven Autoformalization unifies supervised formal-language adaptation and solvability-guided reinforcement learning into a single module built on QwenVL3-2B, making executability the central training signal. Second, Verified Theorem Proposing introduces an impasse-aware agent that proposes local auxiliary lemmas from current proof states, ensuring soundness by filtering all proposals through symbolic verification. Empirical evaluations on Geometry3K and PGPS9K demonstrate that SD-GPS consistently outperforms existing MLLM, neural, and neuro-symbolic methods across standard completion, multiple-choice, and cross-modal reference regimes, proving that closing the loop between multimodal perception and symbolic execution significantly improves geometric reasoning, offering profound insights into how neural agents can be grounded by formal systems to achieve verifiable problem-solving capabilities. 

---
# NormAct: A Benchmark for Hidden Social Norm Compliance in Embodied Planning 

**Authors**: Shiyun Zhao, Xinwei Song, Tianyu Guo, Xiaomeng Gao, Mingyuan Liu, Xu Han, Yuanyuan Zhang, Zhenliang Zhang, Xue Feng, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.27826)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed as embodied planners in egocentric environments, where task success requires not only achieving instructed goals but also acting in socially appropriate ways. While explicit goals may render certain actions optimal, implicit social norms often impose hidden constraints. Existing evaluations typically focus on explicit goal achievement or direct norm knowledge, seldom assessing whether planners can infer and apply these hidden constraints within action sequences. We introduce NormAct, a benchmark for embodied social-norm interactions that evaluates plans on Goal Achievement, Norm Compliance, and overall Task Success. NormAct uniquely embeds hidden norms within ordinary tasks, testing whether models can realize them without explicit instruction. Experiments with state-of-the-art MLLMs (GPT-5.4, Claude Opus 4.7, Gemini 3 Pro) reveal a significant gap: models achieve explicit goals in 67.3\% of cases, but comply with hidden norms in only 26.4\%. Cue-condition experiments indicate that this gap stems not from a lack of general social knowledge, but from challenges in activating and grounding relevant norms in context. To address this, we propose NormPerceptor, a context-conditioned cue generator that infers scene-relevant norms prior to planning, increasing Task Success from 24.2\% to 46.7\%. Our results underscore the importance of enabling embodied agents to proactively detect hidden norms, ground them in visual evidence, and integrate them as action-planning constraints. Our benchmark is publicly available at this https URL. 

---
# ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents 

**Authors**: Qitai Tan, Zefang Zong, Yang Li, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27814)  

**Abstract**: Training small language-model agents for long-horizon interactive tasks requires both fast imitation and reward-driven improvement. On-policy distillation (OPD) provides dense teacher guidance and typically improves rapidly in the early stage, but its gains saturate once the student approaches the teacher, limiting the final performance ceiling. Reinforcement learning (RL) directly optimizes environment rewards and encourages exploratory improvement toward a higher reward-defined ceiling, but sparse and delayed feedback makes early-stage learning much less efficient than OPD. In this paper, we propose ATOD (Annealed Turn-aware On-policy Distillation), a hybrid online distillation algorithm that explicitly exploits this complementarity. (1) ATOD uses an annealed OPD-RL schedule: OPD dominates early training to approach teacher-level behavior, while RL is gradually strengthened to drive reward-based exploration. (2) ATOD introduces Turn-level Disagreement-Uncertainty Reweighting (T-DUR), which softly amplifies high-utility turns and improves dense supervision in long trajectories. Experiments on ALFWorld, WebShop, and Search-QA show that ATOD consistently outperforms competing post-training baselines: across the three student sizes, ATOD improves average success rate by 3.03 points over OPD and 23.62 points over GRPO, while surpassing the corresponding teacher models by 2.16 points. 

---
# Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents 

**Authors**: Xinyuan Song, Zekun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.27806)  

**Abstract**: World models for language agents come in two useful forms. An agent-based world model calls an LLM API and reasons flexibly in language, but its errors appear as hallucinated state changes that are hard to score with ordinary regression losses. A parameterized world model is a trained transition predictor; its errors are easier to measure with quantities such as NodeMSE, delta accuracy, and validity accuracy, but it is usually weaker as a standalone planner. We compare these two families on four graph-structured planning benchmarks and introduce operational hallucination metrics for the agent-based case. The comparison motivates \textbf{Grounded Iterative Language Planning} (GILP), which trains only a small parameterized backbone and combines it with API-based agent reasoning. The backbone supplies valid actions, predicted state deltas, risk, and value; the LLM drafts an action and imagined delta; and a consistency gate asks for revision when the two disagree. On real GPT-4o-mini calls, GILP reduces hallucinated-state rate from 0.176 to 0.035. In calibrated simulator ablations, it raises success from 0.668 to 0.838 while adding only ~22% extra LLM calls. 

---
# Understanding Rollout Error in Graph World Models 

**Authors**: Xinyuan Song, Zekun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.27780)  

**Abstract**: World models are often used for planning by rolling learned dynamics forward. Many planning environments, however, are not vectors or images; they are graphs of agents, tools, skills, routes, and dependencies. In these settings, a local prediction error may stay local or spread through the graph, and the failure mode changes again when edges are predicted rather than fixed. This paper studies long-horizon rollout error in Graph World Models (GWMs). We formulate a unified fixed-edge and dynamic-edge GWM framework with action nodes for node-, edge-, and graph-level decisions. We develop graph-valued rollout bounds that separate topology-induced amplification from model-induced amplification, and we introduce a joint node-edge operator for dynamic-edge rollouts. Guided by the analysis, we propose Error-Aware GWM, which combines spectral regularization, rollout consistency, and critical-node weighting. Across synthetic topologies and heterogeneous agent-graph testbeds, rollout error and planning regret grow with horizon, dynamic-edge training is needed when structure evolves, and Error-Aware GWM prevents long-horizon divergence while preserving prediction accuracy. Real-world graph benchmarks clarify the scope of GWMs: they are most useful for dynamic graph rollout and agent planning, while specialized graph models remain strong on static or sparse prediction tasks. 

---
# Towards Reliable and Robust LLM Planning: Symbolic Feedback-Driven Iterative Self-Refinement Framework 

**Authors**: Jiajing Zhang, Jiamei Jiang, Chenyang Zhang, Feifei Mo, Linjing Li, Daniel Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.27757)  

**Abstract**: Large language models (LLMs) have attracted widespread attention from academia and industry, yet their deployment raises critical security concerns regarding robustness and reliability. Planning, a core component of intelligent behavior, remains challenging for LLMs, which often produce infeasible or incorrect solutions in long-horizon decision-making tasks due to inherent complexity. In this paper, we propose a symbolic feedback-driven iterative self-refinement framework to enhance the robustness and reliability of LLMs in long-horizon planning. Specifically, a natural language prompting mechanism is introduced to map logical symbols into natural language descriptions, enabling LLMs to better capture task constraints and semantics. We further design a symbolic verifier that identifies errors and converts them into corrective instructions interpretable by the LLM, thereby guiding self-refinement. In addition, we leverage a plan recognizer to infer goal reachability, facilitating more effective guidance toward desired goals. Empirical results demonstrate that the proposed framework consistently improves both feasibility and correctness in long-horizon planning tasks. This highlights its effectiveness in enhancing the reliability of LLM-based planning and potential to enable more trustworthy AI systems. 

---
# ToE: A Hierarchical and Explainable Claim Verification Framework with Dynamic Multi-source Evidence Retrieval and Aggregation 

**Authors**: Zhaoqi Wang, Zijian Zhang, Kun Zheng, Zhen Li, Xin Li, Chunlei Li, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27736)  

**Abstract**: The rapid spread of fake news poses increasing threats to information ecosystems, especially as AI-generated misinformation under Generative Engine Optimization (GEO) poisoning allows adversarially crafted content to be systematically surfaced by retrieval systems, contaminating LLM reasoning. In this paper, we propose Tree of Evidence (ToE), a hierarchical evidence reasoning framework for automated fact-checking that models each claim as a dynamically expanding argument tree. ToE integrates a reinforcement learning-driven multi-source retrieval agent, an evidence evaluation agent, and an argument tree aggregation algorithm to iteratively decompose, retrieve, and verify claims through an explainable evidence chain. We further provide a theoretical analysis of the retrieval process, deriving a formal error bound that guarantees the learned policy converges to a neighborhood of the information-theoretically optimal policy. Experiments across multiple datasets and backbone LLMs demonstrate that ToE achieves improvements ranging from 4 to 24 percentage points over competitive baselines, with particularly pronounced gains on adversarially poisoned inputs. 

---
# MER-R1: Multimodal Emotion Reasoning via Slow-Fast Thinking Synergy 

**Authors**: Zhiyuan Han, Beier Zhu, Wenwen Tong, Chengwei Qin, Xinyi Wang, Jiayu Zhang, Jiangnan Chen, Hewei Guo, Dongchuan Ran, Lewei Lu, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27652)  

**Abstract**: We find that explicit reasoning does not necessarily translate into better multimodal emotion recognition (MER) accuracy, even though it makes predictions more interpretable. Specifically, for reasoning-based MLLMs, fast thinking by triggering direct answers often outperforms slow thinking after deliberative reasoning. Our empirical analyses show that fast thinking improves recall with broader and more confident predictions, whereas slow thinking favors precision through conservative filtering of incorrect categories. Building on these insights, we propose MER-R1, a reinforcement learning framework that turns slow-fast complementarity into explicit optimization. Dual-objective disentanglement separates recall and precision into two optimization signals, allowing them to be jointly optimized rather than traded off against each other. Slow-fast confidence calibration further aligns the final slow-thinking answer with fast-thinking intuition, strengthening correct emotions while suppressing incorrect ones. In this way, MER-R1 unifies the recall-oriented intuition of fast thinking with the precision-oriented selectivity of slow thinking. We further provide theoretical justification for this synergy, showing that it mitigates variance-induced interference during optimization. Extensive experiments on MER-UniBench and MME-Emotion show that MER-R1 achieves state-of-the-art performance and makes reasoning genuinely benefit emotion recognition. 

---
# DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums 

**Authors**: Dana Rezazadegan, Atie Kia, Phongpadid Nandavong, Dominique Carlon, Jeremy Nguyen, Abhik Banerjee, James Marshall, Anthony McCosker, Yong-Bin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27619)  

**Abstract**: Dyslexic learners increasingly use artificial intelligence (AI) tools to support reading, writing, organisation, and study-related tasks. However, their lived experiences with these tools remain largely underexamined. This paper proposes DysLexLens, a low-resource LLM framework, designed to analyse dyslexic learners experience with AI through online forum discussions. DysLexLens is designed as an end-to-end, evidence-traceable architecture which transforms noisy social media posts into a dictionary-driven corpora, provides knowledge-graph (KG)-based question reasoning, generates verifiable query responses, and enables response evaluation through quantitative and human-grounded assessment. DysLexLens has four key features. First, it employs a dictionary-driven filtering method to construct a more focused Reddit corpus on dyslexia and AI, filtering out noisy and weakly related posts to improve the relevance of data collected from low-resource forum contexts. Second, it integrates LLM-assisted semantic analysis with KG-based query reasoning to uncover meaningful patterns. Third, it has quantitative evaluation metrics (RAGAS and Query Robustness) to measure LLM-generated response performance. Fourth, it provides structured qualitative validation guidelines for assessing response quality, with a specific focus on hallucination and evidence alignment. We demonstrate the effectiveness of DysLexLens using dyslexia-related Reddit forum data and 30 questions. The results show its potential generalisability to other low-resource forum data contexts. DysLexLens, sample data, questions and evaluation results are available at Github to support reproducibility. 

---
# Odyssey: Constructing Verifiable Local Truth-Preserving Foundation Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2606.27593)  

**Abstract**: We introduce a categorical framework called ODYSSEY for constructing verifiable, local truth-preserving foundation models as compositions of foundries: building-block architectural components that specify a cover of local contexts, local representation families, restriction maps, gluing rules, obstruction policies, update obligations, and human-facing views. A foundry is an organized sheaf of knowledge that carries within it an argumentation component. Concrete foundries are built from generic foundries such as evidence/argument, operational decision, institutional/financial, market meaning, scientific challenge, research-program, assistant-build, and evaluation-harness foundries. Universal Foundry Learning (UFL) formalizes foundry construction as a composition of left and right Kan extensions, with left Kan extension rolling local artifacts into candidate foundries and right Kan extension enforcing the restriction, gluing, obstruction, and argumentation conditions required for promotion. Foundry SQL (FSQL) is a small typed query surface for slicing maintained foundry artifacts that uses TICKET (Topos Integration using Causal Kan Extension Transformers) certification for admitting external or pre-built models into durable ODYSSEY state. ODYSSEY is fully implemented and tested across a wide spectrum of concrete foundries, showing that the same categorical machinery supports domain construction, artifact replay, sheaf diagnostics, grounded Toulmin/local-LLM scrutiny, residual-obstruction ledgers, and optimized TICKET-compatible causal-claim extraction across heterogeneous sources. This paper is to be presented as a 2.5 hour tutorial at ICML 2026. The tutorial home page is at this https URL. 

---
# Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning 

**Authors**: Xuan Zhang, Zhijian Zhou, Lingfeng Qiao, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27483)  

**Abstract**: Large language model (LLM) agents have demonstrated strong capability in sequential decision-making, yet they remains fundamentally reactive in long-horizon tasks. Unlike humans who employ "what-if" reasoning to evaluate potential plans before commitment, standard agents lack an internal world model to simulate future outcomes. Therefore, we propose to internalize future-aware planning by training a single autoregressive model to verbalize both a prospective state rollout and a plan-conditioned success estimate-a textual analogue of the Q-value. Crucially, we identify a format-capability gap: simply fine-tuning agents on look-ahead traces during post-training leads to superficial mimicry of foresight without genuine predictive grounding. To bridge this gap, we introduce a three-stage training paradigm: (i) World Model Agentic Mid-Training (WM-AMT) to inject latent predictive capabilities into the policy; (ii) Format-Eliciting SFT (FE-SFT) to structure this injected capability; and (iii) Foresight-Conditioned Reinforcement Learning (FC-RL) to refine the calibration and utility of the generated simulations. Evaluated on search and mathematical reasoning tasks, our approach consistently outperforms other training baselines. Our results demonstrate that effective internal world modeling in LLM agents requires a capability-first training pipeline to achieve grounded and calibrated foresight. 

---
# When Does Personality Composition Matter for Multi-Agent LLM Teams? 

**Authors**: Aryan Keluskar, Amrita Bhattacharjee, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27443)  

**Abstract**: Personality prompting shapes how large language models communicate, yet whether these behavioral shifts affect objective task outcomes remains under-explored. Prior work shows that agents prompted with low agreeableness produce adversarial language, while those prompted with high agreeableness become cooperative, but the relationship between communication style and task performance has not been systematically examined across multiple domains. In this work, we investigate whether personality composition matters for multi-agent team performance by manipulating personality traits across frontier LLMs on three task domains: structured coding, open-ended research collaboration, and competitive bargaining. We find that personality effects depend critically on task structure. In coding tasks, low agreeableness leads to large communication shifts that have little effect on milestone completion. In open-ended collaboration and bargaining, the same manipulation substantially degrades performance. We discuss implications for multi-agent system design and the limits of personality manipulation. 

---
# AI-Model Network: Concept, Current State and Future 

**Authors**: Li Zhetao, Zeng Xiyu, Wang Jianhui, Xiao Yong, Liu Zhongren, Wu Junru, Lai Junjie, Huang Jijun, Long Saiqin  

**Link**: [PDF](https://arxiv.org/pdf/2606.27382)  

**Abstract**: While the primary function of computers lies in computation and processing, the core value of the Internet is rooted in sharing and collaboration. Computers create the Internet, and the Internet empowers the value of computers. The rapid development of the Internet, cloud computing, and big data is pushing artificial intelligence into the era of large models (LMs). However, the practical application of LMs is currently hindered by high training costs and deployment complexities, driving a shift toward lightweight, private, and domain-specific models. With the rapid proliferation and wide distribution of heterogeneous models, enabling effective interaction and collaboration among them has emerged as a critical bottleneck that urgently needs to be addressed in LM development. Drawing inspiration from the development of the Internet, this paper proposes the concept, vision, and system architecture of world wide AI-model network (AI-ModelNet). It is a novel paradigm that achieves interconnection, capability sharing, and collaborative reasoning by establishing pathways between models. We first briefly review the current state of single-model and multi-model research. Subsequently, the systemic vision and hierarchical architecture of AI-ModelNet are articulated, followed by validation of the framework's feasibility through a prototype system and diverse application cases. Finally, key directions for future research are discussed preliminarily. 

---
# DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation with a Single Hand 

**Authors**: Dihong Huang, Zhenyu Wei, Zhuxiu Xu, Yunchao Yao, Sikai Li, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.28323)  

**Abstract**: Dexterous manipulation policies can solve individual skills, but composing them to perform multiple tasks with a single hand remains challenging. Adding a new task on top of an existing manipulation skill often imposes conflicting demands on overlapping fingers and contact modes, causing destructive interference between preserving an existing manipulation outcome and executing a new one. We propose DexCompose, a role-aware residual composition framework that reuses pretrained dexterous policies for multi-task manipulation through explicit finger-level action ownership. Given two pretrained full-hand policies, DexCompose first collects successful post-task states from the first skill and performs release tests over candidate finger masks to identify which fingers are necessary for maintaining the established skill state. It then trains two asymmetric residual modules: a bounded residual stabilizer for task preservation, and a context-aware residual that adapts the frozen downstream policy only within the action subspace assigned to the new task. We evaluate the framework on 16 composite dexterous manipulation tasks spanning four object-retention skills and four downstream interactions. DexCompose achieves a 77.4% average composite success rate, demonstrating that structural action ownership with dual residuals offers a promising direction for composing dexterous skills beyond conventional policy chaining. 

---
# Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes 

**Authors**: Luis Leal  

**Link**: [PDF](https://arxiv.org/pdf/2606.28308)  

**Abstract**: Many two-player zero-sum games admit not a unique Nash equilibrium but a convex set of them: a polytope of profiles that all share the minimax value V* yet prescribe different behaviour. Standard solvers each converge to some equilibrium and are treated as interchangeable. We ask whether they instead select different members of the Nash set, systematically as a function of the algorithm rather than the seed. Using a tabular, exactly solvable testbed of six games with analytically known Nash sets -- including a two-dimensional Nash polytope and Kuhn poker -- we find that (i) selection is determined by the algorithm, not the seed, but families differ only on asymmetric Nash sets; (ii) regularized last-iterate methods (R-NaD, magnetic mirror descent) select the maximum-entropy member, the information projection of their uniform reference onto the Nash set -- exactly on the 2-D polytope and at 99.7% of maximum entropy in Kuhn -- while regret-averaging methods (CFR, CFR+, fictitious play) drift to a lower-entropy face; we confirm this on a randomized 180-game ensemble, where R-NaD attains the maximum-entropy member in 100% of converged games while CFR+ sits strictly below it in 94% (paired Wilcoxon p < 10^-27); (iii) the selected member has downstream consequences against sub-optimal opponents that scale with sequential/hidden-information structure but stay bounded -- in Kuhn the max-entropy member is a strictly better hedge, whereas on the matrix games the members differ without either dominating. We also report two negative results correcting common intuitions: removing CFR's positive-orthant (max(R,0)) projection does not eliminate boundary drift; and R-NaD's selection is anchor-following, not initialization-independent. We state the maximum-entropy / I-projection characterization as a strongly data-supported conjecture, checked throughout against analytic ground truth. 

---
# Agentic Hardware Design as Repository-Level Code Evolution 

**Authors**: Cunxi Yu, Chenhui Deng, Nathaniel Pinckney, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2606.28279)  

**Abstract**: We present HORIZON, a self-evolving agent framework that treats hardware design as repository-level code evolution. A Markdown harness is compiled into a project pack containing domain knowledge, an executable evaluator, an acceptance predicate, and a git/runtime policy; a hands-free agent loop then evolves an isolated git worktree, using repository operations for state management, tracing, and replay. This extends prior works of repository-scale self-evolution from EDA software systems, to hardware-design artifacts themselves. We evaluate our approach on ChipBench, RTLLM, Verilog-Eval, and nine CVDP categories, achieving 100\% benchmark completion across all suites with a fully hands-free agentic loop. However, we do not claim that agentic AI for hardware design is solved: these benchmarks are controlled proxies for a much broader engineering problem in chip design. Section~\ref{sec:discuss} examines the limitations of the current study and highlights open research challenges. 

---
# Towards Automating Scientific Review with Google's Paper Assistant Tool 

**Authors**: Rajesh Jayaram, Drew Tyler, David Woodruff, Corinna Cortes, Yossi Matias, Vahab Mirrokni, Vincent Cohen-Addad  

**Link**: [PDF](https://arxiv.org/pdf/2606.28277)  

**Abstract**: Artificial intelligence is driving a revolution in scientific discovery, accelerating everything from hypothesis generation to mathematical theorem proving. However, this rapid acceleration is creating a systemic challenge: traditional human peer review cannot scale to match the influx of AI-assisted science. Ultimately, to resolve this tension, we must also deploy AI to accelerate the verification and review process itself. To frame the discussion around this transition, we propose a taxonomy consisting of four progressive levels of AI-human collaboration in scientific evaluation, and discuss various trade-offs involved with each.
As a step toward this future, we introduce the Paper Assistant Tool (PAT), an agentic AI framework built for deep scientific review and verification. PAT ingests full scientific manuscripts and produces a comprehensive evaluation, checking theoretical results, validating experiments, suggesting improvements, and identifying potential flaws. By utilizing inference scaling techniques, PAT is able to identify deeper issues than a single model call alone, achieving a 34% improvement over zero-shot recall on mathematical errors in the SPOT benchmark. Pilot deployments of PAT as a pre-submission tool for authors at two major Computer Science conferences -- STOC and ICML -- demonstrate its ability to identify critical errors and suggest substantive improvements to research papers. By catching errors early, PAT eases the cognitive burden placed on referees, while preserving their control over the outcomes of the review process. 

---
# Parameter Efficient Hybrid Transformer (PEHT) for Network Traffic Prediction via Dynamic Urban Congestion Integration 

**Authors**: Abdolazim Rezaei, Mehdi Sookhak, Mahboobeh Haghparast  

**Link**: [PDF](https://arxiv.org/pdf/2606.28274)  

**Abstract**: Accurate network traffic prediction is a critical element for efficient resource allocation in dynamic urban cellular networks. However, prediction remains challenging because network demand is influenced by complex mobility patterns, congestion dynamics, and heterogeneous user behavior. This paper introduces the Parameter-Efficient Hybrid Transformer (PEHT), a network traffic prediction framework that integrates urban mobility and congestion information into a Transformer-based architecture. PEHT separates primary network communication features from secondary urban mobility features and incorporates Low-Rank Adaptation (LoRA) into the Transformer encoder to reduce the number of trainable parameters while maintaining high predictive accuracy. A multimodal fusion strategy then injects external mobility and congestion features into the decoder to improve traffic forecasting. Experiments on the Telecom Italia Milan dataset and multiple synthetic congestion scenarios show that PEHT outperforms state-of-the-art baselines in terms of RMSE, MAE, and $R^2$. The implementation is available in the GitHub repository. 

---
# Learning Topology-Aware Representations via Test-Time Adaptation for Anomaly Segmentation 

**Authors**: Ali Zia, Usman Ali, Abdul Rehman, Umer Ramzan, Kang Han, Muhammad Faheem, Shahnawaz Qureshi, Wei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28268)  

**Abstract**: Test-time adaptation (TTA) has emerged as a promising paradigm for mitigating distribution shifts in deep models. However, existing TTA approaches for anomaly segmentation remain limited by their reliance on pixel-level heuristics, such as confidence thresholding or entropy minimisation, which fail to preserve structural consistency under noise and texture variation. Moreover, they typically treat anomaly maps as flat intensity fields, ignoring the higher-order spatial relationships that characterise complex defect geometries. We introduce TopoTTA (Topological Test-Time Adaptation), a novel framework that integrates persistent homology, a tool from topological data analysis, into the TTA pipeline to enforce geometric and structural coherence during adaptation. By applying multi-level cubical complex filtration to anomaly score maps, TopoTTA derives robust topological pseudo-labels that guide a lightweight test-time classifier, enhancing segmentation quality without retraining the backbone model. The approach avoids reliance on method-specific raw-score thresholding for mask binarisation, preserves connectivity, and generalises across both 2D and 3D modalities. Extensive experiments across six standard benchmarks (MVTec AD, VisA, Real-IAD, MVTec 3D-AD, AnomalyShapeNet, and MVTec LOCO) demonstrate an average 15% F1 improvement over state-of-the-art unsupervised anomaly detection and segmentation methods, with the largest gains on anomalies exhibiting complex geometric or structural variations. These findings suggest that integrating topological reasoning into test-time adaptation provides a principled route to structure-aware generalisation, bridging the gap between geometric learning and robust adaptation. 

---
# How Width and Data Shape Generalization Scaling Laws in Quadratic Neural Networks 

**Authors**: Julius Girardin, Emanuele Troiani, Yizhou Xu, Vittorio Erba, Florent Krzakala, Lenka Zdeborová  

**Link**: [PDF](https://arxiv.org/pdf/2606.28242)  

**Abstract**: Understanding how performance scales jointly with model size and data is a central problem in modern machine learning. Existing theoretical works on scaling laws typically describe generalization as a function of data or compute, often in fixed-feature or infinite-width regimes and for online SGD. Here, we instead study how generalization scales with the number of trainable parameters and the number of samples in a feature-learning model. We analyze $\ell_2$-regularized empirical test error minimization in a quadratic two-layer network in a finite-sample setting with structured data. This setting allows for an explicit characterization of the generalization error as a function of the number of samples, model width, and regularization. Our results reveal a phase diagram with distinct scaling regimes as the number of parameters varies. In particular, the generalization error follows data-dependent power laws controlled by the spectral structure of the target. We further characterize the transitions between regimes, including the onset of interpolation, and their impact on generalization. 

---
# Govern the Repository, Not the Agent: Measuring Ecosystem-Level Risk in AI-Native Software 

**Authors**: Daniel Russo  

**Link**: [PDF](https://arxiv.org/pdf/2606.28235)  

**Abstract**: Autonomous coding agents now open and merge pull requests in shared repositories at scale, and the field evaluates them the way it has always evaluated components, one agent at a time, on isolated benchmark tasks. Yet agents that each pass their own tests still leave repositories that accumulate problems no single contribution accounts for. We ask whether this problem belongs to the individual agent or to the repository where it accumulates. We study integration friction, the cost of integrating a contribution into a codebase that other contributors are concurrently changing. Across more than 930,000 agent-authored pull requests, we measure how much of the variation in friction stays with the repository after the contribution, its author, its size, and its agent are accounted for. About half does, and it survives full controls. In the same repositories, agent-authored contributions concentrate this repository-level friction roughly twice as much as human ones (intraclass correlation 0.30 versus 0.16), a gap that holds after controlling for codebase size, age, task shape, process maturity, and merge path. The risk is a property of the ecosystem, not the agent. AI-native software is therefore better measured and governed at the ecosystem level than one agent at a time. 

---
# Exposure Bias Can Alleviate Itself via Directional and Frequency Rectification in Flow Matching 

**Authors**: Guanbo Huang, Jingjia Mao, Fanding Huang, Fengkai Liu, Xiangyang Luo, Yaoyuan Liang, Jiasheng Lu, Xiaoe Wang, Pei Liu, Ruiliu Fu, Ruqi Huang, Shao-Lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28226)  

**Abstract**: Flow Matching (FM) has achieved remarkable generative performance, yet it suffers from exposure bias due to discrepancies between training and inference. Existing mitigation strategies typically rely on static constraints or external heuristics. In this work, we propose that exposure bias itself inherently contains dynamic signals that can guide its own rectification. To leverage this, we introduce DEFAR (DirEctional-Frequency Adaptive Rectification). This framework simulates the single-step inference process during training to identify exposure bias. It utilizes directional and frequency-adaptive feedback signals from the bias itself to enhance the model's bias tolerance. It consists of two key components: (1) Anti-Drift Rectification (ADR). ADR treats inference-time drift as a signal to learn the direction to steer deviated states back toward the target. ADR endows the model with intrinsic active self-rectification capabilities; (2) Frequency Compensation (FC). Empirically, we observe that accumulated bias often stems from a lack of low-frequency components in high-noise stages, and exposure bias carries the missing frequency. FC leverages the bias itself as a self-feedback weighting factor to reinforce the missing frequency components. Experiments on CIFAR-10, CelebA-64, and ImageNet-256/512 show that DEFAR outperforms prior baselines and further demonstrates favorable scalability, compatibility, and inference robustness. 

---
# Towards Value-Constrained Credit Assignment in Fully Delegated AI Cooperatives 

**Authors**: Young Yoon, Jimin Kim, Soyeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.28217)  

**Abstract**: We propose a framework for reward allocation in fully delegated AI cooperatives where humans are represented by agents that contribute data and participate in model updates under heterogeneous value constraints. The key idea is to credit only those updates that remain admissible after screening them against each principal's value profile. We formulate value-conditioned gradient filtering, online marginal contribution signals, and cumulative revenue settlement within a traversal learning (TL) substrate. TL is especially attractive here because it performs decentralized backpropagation without the quality loss associated with aggregation-centric distributed learning and, we argue, offers a finer attribution substrate than FedAvg-style federated learning by preserving explicit traversal and gradient paths. The framework is positioned against data valuation, federated contribution estimation, personalized federated learning, and pluralistic alignment. 

---
# HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration 

**Authors**: Jiaxin Li, Yuxiang Wu, Zhenkai Zhang, Xinrui Shi, Haoyuan Wang, Yichen Zhao, Su Linxiang, Chenyang Yu, Mingyu Zhang, Yifan Ding, Boran Wen, Li Zhang, Ruiyang Liu, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28215)  

**Abstract**: Extracting dynamic 4D object interactions from massive, in-the-wild monocular videos offers a highly efficient data collection pathway for scaling Embodied AI and training VLAs. However, existing monocular 4D reconstruction methods primarily focus on isolated objects, often failing under the severe occlusions and complex dynamics inherent in multi-object interactions. To bridge this gap, we propose HAT-4D, the first agentic framework designed to reconstruct the 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single video. By integrating VLMs with a multi-level human-in-the-loop feedback mechanism, HAT-4D efficiently resolves depth ambiguities and interaction-induced occlusions during 3D generation and 4D propagation, yielding physically plausible assets without relying on expensive multicamera rigs. As a scalable data engine, HAT-4D facilitates the creation of MVOIK-4D, an open-world benchmark for monocular 4D interaction reconstruction, accompanied by a novel multi-dimensional evaluation protocol focused on physical plausibility and temporal consistency. Extensive experiments demonstrate that HAT-4D achieves SOTA performance on most evaluation metrics, while maintaining competitive semantic alignment. Ablation studies show that introducing a small amount of human feedback improves interaction reconstruction. Moreover, the data produced by HAT-4D effectively improves baseline performance when used for fine-tuning. Our data and code are available at this https URL 

---
# The Remittance Blueprint: Data-driven Intelligence for Sri Lanka 

**Authors**: Dhinanjaya Fernando, Dinura Ginige, Kalana Lakshan, Chanupa Gurusinghe, Lasana Pahanga, Subavarshana Arumugam, Sandeepa Weerasekara, Sandareka Wickramanayake, Nisansa de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2606.28190)  

**Abstract**: This study analyzes Sri Lankan migration and remittances over 32 years (1994-2025). Using a 384-month harmonized dataset, we apply exploratory data analysis, stationarity corrected time-series modeling (ADF, Johansen, VAR/VECM), and supervised learning. Results reveal remittance inflows are primarily driven by external macroeconomic variables, specifically exchange rate dynamics and global oil prices, rather than domestic indicators. Impulse response analysis confirms the asymmetric impact of currency depreciation and oil price shocks. Predictively, multivariate machine learning models outperform traditional univariate approaches; Ridge Regression achieves a 73.8% accuracy improvement over SARIMA (Annualized RMSE: USD 494.8 Mn). The optimized framework projects 2026 remittances at USD 9,001 million under stable conditions. These findings highlight the structural dependence of remittances on global economies, emphasizing the need for robust exchange rate policies, skilled migration, and formal financial channels to enhance long-term economic resilience. 

---
# Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction 

**Authors**: Chenguang Wang, Ming Li, Xinyue Zeng, Zhuochun Li, Hong Jiao, Tianyi Zhou, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28186)  

**Abstract**: Predicting human item difficulty is central to educational assessment, where reliable estimates support fairness and effective test construction. Existing methods often depend on costly human calibration or item-level textual representations, providing limited evidence about the cognitive processes that make items difficult. We argue that difficulty should be viewed not only as a property of item text, but also as an observable consequence of the problem-solving burden an item induces. Large Reasoning Models (LRMs) offer scalable process evidence through reasoning traces, but such evidence must be structured to support interpretable modeling. To this end, we introduce Epi2Diff (Episode to Difficulty), a framework that maps LRM reasoning traces into cognitively grounded episode sequences. These episodes group trace segments into functional problem-solving states, enabling difficulty to be modeled through reasoning scale, effort allocation, and state transitions. Epi2Diff extracts compact episode-dynamic features and combines them with semantic item representations for human difficulty prediction. Experiments on four real-world human difficulty datasets show that Epi2Diff consistently outperforms strong baselines, including fine-tuned small language models, LLM in-context learning, and supervised LLM adaptation. On SAT-derived classification benchmarks, Epi2Diff achieves an 8.1% average relative gain over supervised LLM fine-tuning baselines. Further analyses show that harder items induce more effortful, iterative, and implementation-centered episode dynamics, rather than merely longer responses. These results demonstrate that cognitive episodes in LRM reasoning traces provide a predictive and interpretable process representation for human item difficulty, offering a new lens for educational measurement with reasoning models. 

---
# LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior 

**Authors**: Qinhong Zhou, Chuang Gan, Anoop Cherian  

**Link**: [PDF](https://arxiv.org/pdf/2606.28182)  

**Abstract**: Embodied agents operating in decentralized and partially observable environments have attracted growing attention in recent years. However, existing large language model (LLM)-based agents often exhibit behaviors that are misaligned with their partners or inconsistent with the environment state, leading to inefficient cooperation and poor task success. To address this challenge, we propose a novel framework, Learning Laws of Cooperation (LLawCo), that enables embodied agents to autonomously align with both their partners and task objectives. Our framework allows agents to reflect on past failures to extract misaligned behavioral patterns, which are used to derive high-level behavioral laws, such as "Talk when necessary" and "Wait for partner." These laws are explicitly incorporated into the agents' chains of thought via supervised fine-tuning, aligning their reasoning with task requirements and the behavior of other agents. To evaluate our approach, we introduce PARTNR-Dialog, a large-scale multi-agent communicative and cooperative planning benchmark built on the PARTNR environment. Experiments on existing tasks and our new benchmark demonstrate significant improvements in cooperative efficiency and task success rates. Across four backbone LLMs, our method achieves average success rate improvements of 4.5% on the PARTNR-Dialog benchmark and 6.8% on the TDW-MAT benchmark over state-of-the-art open-source communicative agent frameworks. See the LLawCo project page for details: this https URL 

---
# CPAgents: Agentic Composite Phenotype Generation for Cardiac Disease Association 

**Authors**: Zuoou Li, Wenlong Zhao, Kelly Yu, Weitong Zhang, Paul M. Matthews, Wenjia Bai, Bernhard Kainz, Mengyun Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.28179)  

**Abstract**: Identifying robust associations between cardiac imaging phenotypes and clinical diseases is fundamental to population-scale cardiovascular research and reliable risk stratification. However, current phenome-wide association studies rely on pre-defined, single-variable phenotypes or expert-crafted features, which limits their ability to capture clinically meaningful non-linear effects and cross-phenotype interactions. To address this, we propose CPAgents, an iterative phenotype-Composition framework for cardiovascular Phenome-wide association study (PheWAS) that automatically constructs and validates interpretable composite phenotypes (e.g., polynomial, ratio, and interaction forms) from base imaging features. Specifically, our system coordinates three agents: (i) an Analyst that identifies statistical pathologies and nominates candidate transformations; (ii) a Proposer that generates constrained, medically and statistically motivated expressions under numerical safety rules; and (iii) a Verifier that evaluates candidates using multi-stage criteria and produces transparent evidence trails for accepted phenotypes. Evaluated on a population-scale cardiac imaging cohort, the discovered composite phenotypes markedly improve disease discrimination: across 72 classifier-disease-metric combinations, our variants achieve the top rank in 56 cases versus 18 for baselines, with gains observed across all nine clinical disease categories. Our framework yields compact, clinically interpretable phenotype formulas with transparent evidence trails, enabling scalable discovery of stronger phenotype-disease associations beyond expert-driven feature selection. 

---
# Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models 

**Authors**: Yanchen Yin, Dongqi Han, Linghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.28153)  

**Abstract**: Jailbreak attacks bypass LLM safety alignment, yet their mechanisms remain poorly understood. We provide evidence that attacks do not comprehensively eliminate safety features, but instead selectively suppress specific attention heads. We identify two functionally differentiated types: Adversarially Compromised Heads (ACHs) concentrated in early layers, which are suppressed under attacks, and Safety-Aligned Heads (SAHs) in mid-layers, which maintain robust activations even when attacks succeed. Ablation studies support the causal role of ACHs and the contribution of SAHs to robust activations: suppressing a small number of ACHs is sufficient to induce jailbreak-like behavior on normally refused inputs, while removing SAHs substantially weakens mid-layer safety activations. Token-level attribution further shows that ACH suppression is driven specifically by attack-template tokens, providing a mechanistic account of why attacks can bypass refusal decisions through ACH suppression while leaving internal safety signals sustained by SAHs -- a phenomenon we term Robust Harmful Features. To validate the practical significance of this robustness, we show that simply reading these persistent activations -- without any training -- yields competitive aggregate detection performance with strong adversarial robustness. 

---
# Toward Robust In-Context Segmentation via Concept Guidance 

**Authors**: Zhigang Chen, Xiawu Zheng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.28149)  

**Abstract**: In-context segmentation (ICS) requires a model to segment target regions in a query image using only a few reference images and their corresponding masks, without updating any parameters. Despite recent progress, prior ICS studies have largely overlooked a critical aspect: system robustness, ie, whether the model can produce stable segmentation results for the same query under different references. In this work, we revisit ICS from the robustness perspective and introduce a novel paradigm, Concept-Guided In-Context Segmentation (CG-ICS), which performs segmentation by extracting high-level semantic concepts from references rather than relying solely on low-level visual matching. Specifically, CG-ICS introduces a concept reasoning module that uses an MLLM to propose candidates and a SAM3-driven scoring function with tree-search refinement to select reliable textual concepts, together with a parallel visual exemplar route that provides query-side spatial grounding via a simple context construction. Both the textual concept and the visual exemplar are then used to activate the segmentation capability of a frozen SAM3 backbone. Extensive experiments on standard ICS benchmarks demonstrate that CG-ICS not only achieves state-of-the-art accuracy but also substantially improves robustness, yielding a more reliable ICS system with significantly reduced variance across diverse reference choices. 

---
# Beyond Sparse Supervision: Diffusion-Guided Learning for Few-Shot Graph Fraud Detection 

**Authors**: Liming Liu, Chao Hu, Mingfei Lu, Yiwei Ge, Xingle Li, Heyuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.28134)  

**Abstract**: Graph-based fraud detection is essential for safeguarding large-scale transaction systems, where undetected anomalies may lead to substantial financial losses and security risks. Real-world fraud graphs pose two coupled challenges: sparse and imbalanced supervision, where verified fraudulent labels are scarce and heavily skewed toward benign accounts, and representation dilution, where spatial message passing may oversmooth camouflaged anomalies while spectral filters may suppress fraud-relevant mid- and high-frequency irregularities. To address these challenges, we propose ADC-GNN, short for Attention-guided Diffusion-Contrastive Graph Neural Network, a unified framework that combines diffusion-guided feature augmentation, contrastive representation learning, and multi-hop spectral attention for few-shot graph fraud detection. The diffusion component is formulated as a feature-space denoising augmentation mechanism rather than a full topology-generative graph diffusion model: it constructs noise-perturbed node-feature views under a cosine schedule and uses contrastive learning to stabilize node representations across perturbations. The spectral attention module further adaptively emphasizes fraud-relevant hop-level and relation-level cues. We evaluate ADC-GNN primarily on three public benchmarks and additionally report a proprietary real-world telecom transaction dataset with approximately 60,000 records as a private case study. Under the 1% training setting, ADC-GNN achieves consistent improvements over original graph fraud baselines and four protocol-consistent recent graph anomaly/fraud baselines on the public benchmarks. Additional analyses on split stability, training ratios, oversampling alternatives, module-level ablations, diffusion schedules, and runtime and memory-consumption comparisons further characterize the effective operating regime of ADC-GNN. 

---
# PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation 

**Authors**: Peiwen Zhang, Yufan Deng, Shangkun Sun, Juncheng Ma, Duomin Wang, Jonas Du, Zilin Pan, Ye Huang, Hao Liang, Songyan Huang, Ruihua Zhang, Enze Xie, Ming-Yu Liu, Daquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28128)  

**Abstract**: Video generation models have emerged as a promising paradigm for embodied world simulation. However, both general-domain video generators and robot-specific data fine-tuned models can still produce physically implausible manipulations, including discontinuous motion trajectories and inconsistent robot-object interactions, which limits their reliability as world simulators. Through extensive experiments, we find that such physical instability mainly arises from two factors: deformation of moving objects and implausible spatio-temporal correlations among interacting entities, particularly during contact. Building on this observation, we propose PhysisForcing, a scalable training framework that strengthens physical consistency by focusing supervision on physics-informative regions through joint optimization of pixel-level and semantic-level features. The framework consists of a pixel-level trajectory alignment loss, which supervises DiT features using reference point trajectories, and a semantic-level relational alignment loss, which aligns DiT features with inter-region relations extracted from a frozen video understanding encoder. Extensive experiments on R-Bench, PAI-Bench, and EZS-Bench show that PhysisForcing consistently improves embodied video generation over strong baselines, improving the Wan2.2-I2V-A14B and Cosmos3-Nano base models on R-Bench by 22.3\% and 9.2\% (7.1\% and 3.7\% over vanilla finetuning), with the Cosmos3-Nano variant attaining the best overall score. Beyond generation, as a world model under the WorldArena action-planner protocol it raises the closed-loop success rate from 16.0\% to 24.0\% and further improves downstream policy success, indicating that physically aligned video models yield stronger representations for robotic manipulation. 

---
# From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond 

**Authors**: Paul Dubois  

**Link**: [PDF](https://arxiv.org/pdf/2606.28127)  

**Abstract**: The AI community has framed the relationship between large language models (LLMs) and world models as a dichotomy: LLMs predict tokens; world models simulate reality. Yann LeCun argues in 2022 that reaching general intelligence requires abandoning autoregressive token prediction in favour of latent-space architectures. This framing is unnecessarily binary. Two claims will be defended. First, LLMs are a degenerate special case of world models: the state space is the set of all token sequences, the only action is appending one token, and world models are therefore a strict generalisation of LLMs, not a replacement. Second, there is a natural continuous spectrum from NTP to JEPA, with multi-token prediction, future-summary prediction, and next-latent prediction as intermediate stations already populated by current research. Moving along this spectrum relaxes the LLM constraints one by one. It also progressively surrenders the two practical advantages that make LLMs trainable at scale: internet-scale self-supervised data, and a transformer architecture co-designed for discrete token prediction. Both are examined as open research questions: the data question (the cliff from self-supervised text to instrumented action-labelled environments) and the architecture question (whether the transformer generalises to continuous-state prediction, or whether a new primitive is needed). 

---
# Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs 

**Authors**: Alex Colagrande, Paul Caillon, Eva Feillet, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28122)  

**Abstract**: Neural operators provide deep neural networks for learning mappings between function spaces. Among them, the Fourier Neural Operator (FNO) is particularly effective: its spectral convolution relies on low-dimensional Fourier-domain representations and can handle inputs at different resolutions. This design aligns well with settings where the Fourier basis diagonalizes the underlying operator, such as linear, constant-coefficient PDEs on periodic domains, in which Fourier modes evolve independently. However, nonlinear PDEs may benefit from an additional inductive bias, as they exhibit structured interactions between modes, governed by polynomial nonlinearities. To capture this inductive bias, we introduce the Higher-Order Spectral Convolution, a spectral mixer that extends FNO from diagonal modulation to explicit n-linear mode mixing, aligned with the dynamics of nonlinear PDEs. Our experiments on standard benchmarks show that the proposed Higher-Order FNO (HO-FNO) retains the efficiency of FNO-based architectures and consistently improves over other spectral neural operators. HO-FNO also performs on par with or better than state-of-the-art transformers and state-space models on several datasets, with stronger gains in highly nonlinear regimes, such as the Poisson equation with polynomial forcing, where a single HO-FNO layer outperforms FNO models with up to 16 layers. We open-source our code for reproducibility at: this https URL. 

---
# BiDeMem: Bidirectional Degradation Memory for Explainable Image Restoration 

**Authors**: Xinrui Wu, Lichen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28112)  

**Abstract**: Degradation-aware prompts, conditions, and latent priors are increasingly used in image restoration, yet they are usually judged by a single endpoint: whether the restored image obtains higher PSNR. This is a weak test of semantics. A condition can help by adding capacity, acting as a global correction bias, or exploiting dataset shortcuts, without becoming an interpretable degradation prior. We propose BiDeMem, a bidirectional degradation memory for explainable image restoration. A query built from restoration features and input statistics retrieves a compact top-k subset of memory slots. The same selected slot identity supports the restoration path at inference time and a training-only forward-degradation explanation path. The study centers on verifiability in a controlled multi-degradation NAFNet setting. New controls separate the gain from a correction head alone, a dense query prior, and a static global prior: these variants are 0.2588, 0.2586, and 0.2839 dB below BiRank, respectively. Strong residual supervision and a wider degradation head also remain below the full bidirectional memory model. Intervention probes show that BiRank preserves restoration quality while increasing wrong-prior and native-prior sensitivity, framing degradation memory as both a restoration module and a falsifiable explanation mechanism. 

---
# OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal 

**Authors**: Qinming Zhou, Chenxi Sun, Deyang Kong, Junhao He, Xiangheng Tang, Peike Yu, Haotian Wu, Leilei Cao, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28094)  

**Abstract**: Real-world object removal is challenging due to two key difficulties: the target object's non-local effects, such as shadows and reflections, which are difficult to model, and the fact that user-provided masks are often inaccurate or incomplete. With billions of parameters and tens of denoising steps, diffusion-based models achieve strong removal performance at the expense of substantial computational cost, limiting their use in interactive applications and on edge devices. To address these challenges, we present OSOR (One-Step Object Removal), which simultaneously achieves efficient, effect-aware, and mask-robust object removal. Concretely, OSOR introduces: (1) an occupancy-guided discriminator for precise boundary supervision, enabling stable single-step diffusion training; (2) an alpha head that leverages knowledge from pretrained diffusion models to predict appropriate removal regions with minimal overhead, thereby handling imperfect masks; and (3) a semantic-anchored verification pipeline (SAVP) that filters noisy instruction-based triplets to produce effect-aware supervision at scale. Using SAVP, we curate CORNE, which contains 280K verified removal pairs, and further annotate AnimeEraseBench and TextEraseBench to evaluate performance on more complex removal tasks. Experiments show that OSOR surpasses strong multi-step diffusion baselines in perceptual quality while achieving $4\times$ to $30\times$ faster inference. 

---
# STAG: Spatio-temporal Evolving Structural Representation of Action Units for Micro-expression Recognition 

**Authors**: Nandani Sharma, Varun Sharma, Dinesh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28083)  

**Abstract**: Micro-expression recognition is challenging due to subtle and short-lived facial muscle movements. Existing methods rely heavily on apex-onset frames, overlook fine-grained inter-frame dynamics, and separately model spatial and temporal information, limiting generalization across datasets. To address these challenges, we propose STAG, a dynamic ROI-AU-coupled spatial-temporal network that jointly models motion flow and adaptive facial connectivity. The framework extracts optical flow from discriminative frames using magnitude-based selection and temporal attention. A dual-branch architecture combines an enhanced graph attention network for structured spatial reasoning with a transformer encoder for temporal modeling. A bidirectional cross-attention module enables mutual refinement of spatial and temporal features, while AU-guided dynamic connectivity adapts facial region interactions according to muscle activation patterns. The transformer captures subtle temporal dynamics beyond apex-based approaches, improving semantic consistency and interpretability for explainable micro-expression recognition. The fused representation is optimized using focal loss and evaluated on CASME II, 4DME, DFME, NaME, SAMM, and SMIC-HS. Extensive experiments demonstrate improved robustness, generalization, interpretability, and computational efficiency, confirming the effectiveness of adaptive relational reasoning, AU-guided dynamic connectivity, and deep spatial-temporal feature fusion for accurate cross-dataset micro-expression recognition. 

---
# OperatorSHAP: Fast and Accurate Shapley Value Estimation for Neural Operators 

**Authors**: Joshua Stiller, Santo M. A. R. Thies, Felix Czaja, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2606.28065)  

**Abstract**: Understanding model predictions is essential for physical applications, where outputs often inform safety-critical decisions, such as structural load assessment, weather warnings, and clinical diagnosis. Shapley values satisfy many desirable properties as an attribution method, but their computational cost during inference hinders their practical use. Current amortized explainers, such as FastSHAP, are limited to homogeneous inputs, which is problematic for physical applications where data often comes from irregular grids and geometries. We introduce OperatorSHAP, a grid-agnostic attribution method and training procedure that allows us to train FastSHAP-like explainers for neural operators. We establish a theoretical framework for attributions in function space, connecting to Aumann-Shapley values. We further show that OperatorSHAP's explanations are consistent with state-of-the-art discrete Shapley values across resolutions and transfer across grid sizes without retraining. 

---
# Single and Multi Truth Data Fusion using Large Language Models 

**Authors**: Hira Beril Kucuk, Norman W Paton, Jiaoyan Chen, Zhenyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28062)  

**Abstract**: Data fusion, also known as truth discovery, is a data integration problem that aims to determine the correct value or set of values for each attribute of an object when presented with potentially conflicting values from multiple sources. Data fusion tasks belong to two main categories: single-truth scenarios, where each attribute has only one correct value, and multi-truth scenarios, where multiple values can be valid simultaneously. This paper investigates the use of Large Language Models (LLMs) in data fusion tasks for tabular data. Various prompting strategies, encompassing both single-truth and multi-truth scenarios, are investigated empirically. Domain-dependent, domain-independent, zero-shot and one-shot prompts are evaluated on three different benchmark datasets. Experimental results demonstrate that LLM-based approaches outperform traditional unsupervised truth discovery methods, such as DART and LTM, across all datasets. The codebase of this study has been made publicly available on GitHub. 

---
# ToolPrivacyBench: Benchmarking Purpose-Bound Privacy in Tool-Using LLM Agents 

**Authors**: Shijing Hu, Liang Liu, Zhu Meng, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.28061)  

**Abstract**: Large language models (LLMs) have increasingly moved from standalone text generation systems to agents that invoke external tools, access environments, and execute multi-step tasks. However, conventional function-calling benchmarks mainly evaluate task completion and API correctness, while privacy evaluation benchmarks typically focus on final responses or privacy judgments. Neither perspective captures purpose-bound information flow across an executed multi-tool trajectory. Motivated by this limitation in current agent evaluation, ToolPrivacyBench audits whether task-private atoms are routed only to authorized tools and downstream sinks, thereby evaluating both task completion and privacy over-disclosure during tool use. The benchmark contains 2,150 cases, including 1,150 fully synthetic privacy-sensitive business workflows and 1,000 cases adapted from existing multi-tool and function-calling benchmarks. Each case is represented by a policy knowledge base. After an agent executes against mock business backends, the evaluator compares recorded tool arguments and backend audit logs with this policy knowledge base. The evaluation covers nine widely used agents to characterize purpose-bound privacy over-disclosure. The results show that successful tool execution does not imply appropriate privacy disclosure: an agent may complete a task while transmitting unnecessary private information through intermediate tool calls. ToolPrivacyBench therefore formalizes a need-to-know disclosure boundary, under which each tool should receive only the information necessary for its stated purpose, and uses trajectory-level auditing to identify privacy over-disclosure in multi-tool workflows. 

---
# MultiHashFormer: Hash-based Generative Language Models 

**Authors**: Huiyin Xue, Atsuki Yamaguchi, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2606.28057)  

**Abstract**: Language models (LMs) represent tokens using embedding matrices that scale linearly with the vocabulary size. To constrain the parameter footprint, prior work proposes hashing many tokens into a single vector within encoder-only models. While this offers parameter efficiency, many-to-one collisions prevent its use in causal LMs. In this paper, we propose MultiHashFormer, a new framework that allows hash-based autoregression. Each token is represented as a unique hash signature, a short sequence of discrete hash IDs, generated by multiple independent hash functions. A Hash Encoder compresses this signature into a single latent vector for processing by a Transformer decoder. Then, a Hash Decoder generates the hash signature of the next token, which is then mapped back to text. We evaluate our approach at the 100M, 1B and 3B parameter scales, demonstrating that MultiHashFormer consistently outperforms standard Transformer LMs across multiple benchmarks. Furthermore, we show that our model handles multilingual vocabulary expansion with a constant parameter footprint without any modifications. 

---
# Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA 

**Authors**: Sambaran Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2606.28050)  

**Abstract**: LLM-as-a-Judge and self-evaluation pipelines implicitly assume that evaluation is easier than generation. We test this in a controlled in-context QA setting where a context passage is the sole information source and each model judges the answer it generated, removing the parametric-knowledge confound of open-domain comparisons. Across four benchmarks (SQuAD 2.0, DROP, HotpotQA, MuSiQue) and two models, evaluation is not uniformly easier: generation accuracy exceeds self-evaluation on three of four, with multi-hop MuSiQue the exception. Attention analysis reveals why: evaluation attends to context 3--5x less than generation does and barely reads the candidate answer. LoRA fine-tuning confirms the asymmetry is not a training artifact: generation fine-tuning induces over-acceptance and evaluation fine-tuning degrades generation. These findings challenge core assumptions in self-evaluation pipelines. 

---
# DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions 

**Authors**: Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2606.28048)  

**Abstract**: Insurance fraud remains costly and operationally difficult, particularly in call-centre workflows where many customer interactions begin at FNOL. While recent fraud detection methods mainly rely on structured data, text, or images, repeated speaker identity across calls remains underused as an investigative signal. This paper presents DG^VoiC, a voice clustering framework for customer verification and cross-profile speaker linking on anonymised real call-centre audio. The approach combines sensitive information-aligned anonymisation, speech-focused preprocessing, sliding-window speaker embedding extraction, and cosine similarity based clustering to identify repeated speakers under real telephony conditions. The method was evaluated on 121 recordings, with a curated reference subset of 56 samples in 22 human-agreed speaker clusters. used for validation. The best configuration achieved 96% AMI, 95% ARI, 98% completeness, 100% homogeneity, and 99% V-measure. These results show that speaker clustering can provide a strong additional signal for fraud investigation by helping analysts verify speaker consistency and surface repeated voices across customers. 

---
# Mind the Gap: Quantifying the Domain Gap in Cross-Sensor Diffusion Super-Resolution 

**Authors**: Dawid Kopeć, Katarzyna Jabłońska, Wojciech Kozłowski, Maciej Zięba  

**Link**: [PDF](https://arxiv.org/pdf/2606.28039)  

**Abstract**: Demand for high-resolution satellite imagery has increased interest in super-resolution (SR) to bridge the spatial resolution gap between freely available missions such as Sentinel-2 and commercial systems like PlanetScope. Because no sensor provides true paired low- and high-resolution observations, SR models are usually trained on synthetically degraded data, creating a domain gap on real cross-sensor imagery. In this work, we provide the first systematic study of how this synthetic-to-real mismatch affects the performance of modern diffusion-based SR models. Using a large, geometrically and temporally aligned dataset of Sentinel-2 and PlanetScope imagery, we evaluate five state-of-the-art diffusion architectures under controlled experimental settings. We also introduce LPIPS-Sat, a domain-adapted perceptual metric based on Sentinel-2 self-supervised features. Our results show two persistent challenges: synthetically trained models degrade sharply on real pairs, while models trained on real cross-sensor data exhibit optimisation difficulties and struggle to adapt to the physical and radiometric diversity. These findings highlight a key limitation of current SR and motivate methods that disentangle super-resolution from domain adaptation. 

---
# MLVC: Multi-platform Learned Video Codec for Real-World Deployment 

**Authors**: Tanel Pärnamaa, Martin Lumiste, Ardi Loot, Evgenii Indenbom, Andrei Znobishchev, Ando Saabas  

**Link**: [PDF](https://arxiv.org/pdf/2606.28027)  

**Abstract**: Neural video codecs have surpassed classical codecs in coding efficiency but remain impractical for deployment due to cross-platform incompatibility and high computational cost. Existing quantization-based solutions fail to produce deterministic results across diverse hardware platforms, leading to catastrophic decoding failures. We introduce MLVC, a hardware-robust neural video codec designed for practical cross-platform inference. The key idea is to explicitly transmit scale parameters through the hyperprior, which guarantees entropy coding consistency across devices without requiring bit-exact arithmetic. While this increases bitrate overhead, we recover most of the coding efficiency through architectural improvements (gated memory, ReGLU activation), a long-term reference recovery mechanism, and domain-specific perceptual training. On the VCD video conferencing benchmark, MLVC achieves >70% BD-rate (MOS) improvement over hardware HEVC, the strongest deployable baseline, while reaching subjective quality competitive with DCVC-RT, which cannot operate across diverse platforms. Both the encoder and decoder run at 100 FPS on average on commodity NPUs from Apple, Intel, and Qualcomm. MLVC is the first neural video codec to combine competitive compression performance, real-time speed, and cross-platform robustness across diverse consumer devices, making it suitable for widespread deployment. Code will be released. 

---
# Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection 

**Authors**: Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2606.28002)  

**Abstract**: Insurance fraud imposes substantial financial losses and operational inefficiencies, raising premiums and impacting trust among legitimate policyholders. Early detection at FNOL remains a persistent challenge. Existing approaches rely largely on private, text-only datasets, limiting progress on multimodal methods that integrate linguistic, behavioural, and speaker-based indicators. We introduce a synthetic multimodal framework that replicates FNOL conditions. It generates agent-customer dialogue transcripts and two-speaker audios, performs ASR and diarisation. Downstream modules combine NER, regex-based feature extraction, LLM-RAG retrieval, and speaker embeddings in a rule-based risk score to flag narrative reuse, structural inconsistencies, and cross-case voice repetition while balancing sensitivity and false positives. Dataset validation and component-level evaluations show stability and transfer potential, offering a reproducible baseline beyond text-only fraud detection. 

---
# Parallel Rollout Approximation for Pixel-Space Autoregressive Image Generation 

**Authors**: Jiayi Xu, Di He, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2606.27978)  

**Abstract**: Pixel-space continuous-token autoregressive (AR) generation directly models images as sequences of raw pixel patches, avoiding discrete tokenization or a separately pretrained tokenizer. However, it faces coupled challenges: high-dimensional patch generation causes large single-step errors, and teacher-forced training creates a train--inference gap that makes these errors accumulate across AR steps. Existing fixes such as $x$-prediction and input noise injection only partially mitigate these issues. Exact rollout training better matches inference-time conditions, but is impractical due to prohibitively slow sequential sampling. We propose \emph{Parallel Rollout Approximation} (PRA), a scalable framework that addresses both challenges jointly. PRA generates low-dimensional intermediate states instead of high-dimensional pixel patches, then maps them back to pixel-space tokens with a pixel decoder, preserving a pixel-in, pixel-out AR interface. It also constructs inference-like pixel inputs through the same intermediate-state-to-pixel path used at inference, independently across positions, approximating the pixel-feedback interface encountered during inference-time rollout while retaining parallel teacher-forced training. On class-conditional ImageNet-1K generation at $256\times256$ resolution, PRA-S with 135M parameters achieves an FID of 2.58, surpassing the previous billion-scale pixel-space AR result of 3.60. Scaling to PRA-L with 511M parameters further improves FID to 1.94, establishing a new state of the art among pixel-space AR models. Beyond generation, PRA achieves higher ImageNet classification probing accuracy than other AR and diffusion baselines, suggesting its potential for unified pixel-space image generation and understanding. 

---
# SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieval 

**Authors**: Sergey Kurilenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.27976)  

**Abstract**: Dense embeddings underpin semantic search and RAG, yet a leaked vector store hands much of the underlying text back to whoever holds it. The attacks that make this possible (few-shot alignment, zero-shot inversion, unsupervised cross-space translation) share one weakness: the protected store is a single global geometry that can be aligned to a known one. A secret global rotation, the usual lightweight defence, is no exception: orthogonal Procrustes recovers it once the attacker has about the subspace dimension in known pairs.
We introduce Shard, a retrieval-preserving embedding transform that removes this weak axis. The centred embedding is split into a short public prefix (for stage-1 retrieval) and a private residual sharded into C cells under separate secret keys; the residual is reranked under CKKS, where the keys cancel and leave the inner product exact. A single parameter C runs the design from the global-linear baseline it replaces (C=1) to per-document micro-keys (C=N). Because the rerank is full-dimensional, Shard returns the raw-space nDCG@10 that half-SVD truncation gives up; and because the residual is keyed cell-locally, mapping it back to a common frame under a diffuse known-plaintext leak costs roughly C times more anchors (median 200 to 102,400 at C=256), for a few encrypted queries. The short public prefix leaks far less neighbour structure, and a micro-key limit drives the residual graph to zero with an unlinkable, renewable template. The barrier holds against learned, non-linear and unsupervised aligners, and where a matched-utility noise defence de-anonymises almost every probe, Shard de-anonymises none. We are plain about the limits: within a cell the keys cancel, a targeted attacker needs only about d_priv anchors, and an overlapping reference corpus still leaks through the prefix. Shard is an attack-aware geometric defence, not a cryptographic guarantee. 

---
# ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering 

**Authors**: ZhengXian Wu, Hangrui Xu, Kai Shi, Zhuohong Chen, Yunyao Yu, Chuanrui Zhang, Zirui Liao, Jun Yang, Zhenyu Yang, Haonan Lu, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27974)  

**Abstract**: Knowledge-based Visual Question Answering (KB-VQA) requires models to combine image understanding with external knowledge. Most prior methods use a fixed retrieve-then-generate pipeline with a pre-selected retriever and a static top-k setting, which is not adaptive during reasoning. We propose ProMSA, a progressive multimodal search agent for KB-VQA. Given an image-question pair, the agent iteratively chooses image search, text search, or stop, under explicit tool-call budgets and with deduplication to avoid redundant retrieval. For training, we first use rejection-sampling SFT to learn valid tool-use formats, then optimize the agent with TN-GSPO, a sequence-level RL objective that normalizes updates by both generation length and tool-interaction depth. Experiments on E-VQA and InfoSeek show consistent gains over strong RAG and agent baselines, and improved retrieval and end-to-end accuracy. The code is available at this https URL. 

---
# From Black-Box to Clinical Insight: A Multi-Stage Explainable Framework for Speech-Based Cognitive Impairment Detection 

**Authors**: Yasaman Haghbin, Sina Rashidi, Ali Zolnour, Fatemeh Taherinezhad, Ali Fartoot, Hossein Azadmaleki, James M Noble, Maryam Dadkhah, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2606.27973)  

**Abstract**: Speech-based cognitive impairment detection offers a noninvasive, accessible alternative to costly biomarker assays, yet transformer-based models remain clinically uninterpretable. We propose a multi-stage explainability framework that translates black-box transformer predictions into clinically grounded narratives by integrating SHapley Additive exPlanations (SHAP)-based token attribution, theory-informed linguistic features, and a four-stage LLM reasoning pipeline using LLaMA-3.1-70B-Instruct. Built on the SpeechCARE-Adaptive Gating Network multimodal screening model (F1 = 72.11% on the NIA PREPARE benchmark), the framework maps model outputs to four cognitive-linguistic dimensions, including lexical richness, syntactic complexity, and semantic coherence. Physician evaluation on 70 stratified English samples demonstrated strong alignment with patient-level cognitive profiles, and a System Usability Scale score of 82/100 indicated high potential for clinical workflow integration. 

---
# Reasoning Beyond Prediction: From Data-Driven to Causal Software Engineering 

**Authors**: Roberto Pietrantuono, Luca Giamattei, Stefano Russo  

**Link**: [PDF](https://arxiv.org/pdf/2606.27960)  

**Abstract**: Software engineering is an intellectually demanding, creative discipline that juggles a web of interdependent tasks to design, build, and assure the quality of increasingly complex systems. As our expectations from software soar - with demands spanning AI-driven products, pervasively distributed and cloud-native architectures, and deeply embedded cyber-physical environments - its complexity steadily increases. In response, a new wave of co-engineering methods and tools, fueled by deep learning, has emerged to augment the process, enhancing automation and decision support. Yet, these advances remain far from delivering the kind of intelligent support that modern software development demands. We call for a new paradigm of human-machine cooperation: one where machines don't just automate routine tasks or predict from learned patterns, but actively amplify engineers' reasoning through the lens of causation. As software becomes smarter, a smarter support is needed. 

---
# VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring 

**Authors**: Kairui Zhang, Ziwen Yu, Zahraa S. Abdallah, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2606.27941)  

**Abstract**: Sparse autoencoders (SAEs) provide useful decompositions of Transformer residual streams, but their learned features are usually named post hoc rather than directly connected to the Transformer's token vocabulary. We introduce Vocabulary-Aligned Sparse Autoencoder (VASAE), a method that trains SAE features under vocabulary-aligned anchoring and assigns each feature an intrinsic token name: the token string whose embedding is nearest to that feature. Without reducing reconstruction quality compared with a standard SAE, VASAE produces dictionaries with vocabulary-aligned features. Using a 0.8 cutoff on the nearest-token alignment score, dictionaries trained on GPT-2-small post-residual streams align about 90% of features in layers 0--10. In Llama-3.1-8B, representative shallow and middle-layer dictionaries contain strongly aligned features, including 92.8% in the shallow layer, while the representative final-layer dictionary shows limited alignment. After subtracting the sentence-level mean sparse code, case studies show that many remaining intrinsic token names are relevant to nearby input tokens. These results suggest that vocabulary-aligned anchoring can connect learned features to intrinsic token names during training, complementing post hoc interpretation of learned dictionaries. 

---
# Two-Stage Fine-Tuning for Protein Sequence Generation with Targeted Amino-Acid Composition 

**Authors**: Violeta Basten-Romero, Rubén Muñoz-Tafalla, Anna María Díaz-Rovira, Bertran Miquel-Oliver, Isaac Filella-Merce, Víctor Guallar  

**Link**: [PDF](https://arxiv.org/pdf/2606.27939)  

**Abstract**: Protein language models are standard priors for biological sequence generation, but steering them toward explicit distributional design targets remains largely unexplored. We study a constrained protein generation problem in which sequences must match a desired amino-acid (AA) composition profile while preserving plausible sequence statistics and diversity. The motivating application is synthetic feed protein design, where the AA composition of dietary proteins directly determines their nutritional value. We propose a two-stage pipeline in which domain-adaptive fine-tuning (FT) on an in-domain protein dataset is followed by iterative reward-weighted FT via reinforcement learning (RL) anchored against the FT model as a frozen reference. We evaluate the pipeline on two AA compositions and find that FT brings the average composition close to the target, while the subsequent RL enforces specific sequence constraints that FT alone cannot satisfy. We additionally evaluate the design choices of the proposed composition reward term against two baselines and an ablated variant, isolate the contribution of each training stage, and verify that AA composition alignment is achieved without degrading sequence quality. 

---
# Agentic AI-Powered Re-Identification: An Emerging, Scalable Threat to Mobility Microdata Privacy 

**Authors**: Oscar Thees, Roman Müller, Matthias Templ  

**Link**: [PDF](https://arxiv.org/pdf/2606.27936)  

**Abstract**: The widespread collection of fine-grained location data by commercial data brokers creates a re-identification risk that is not widely recognised by the public. While prior research has established that mobility traces are highly unique and that individuals can, in principle, be identified from a handful of spatio-temporal points, such attacks have historically required significant manual effort from skilled analysts, limiting their practical scale.
In this feasibility study, we demonstrate in a real world setting that agentic AI fundamentally changes this threat model. We present an end-to-end pipeline in which large language model agents autonomously search the open web, cross-reference public records and social media, and resolve raw coordinate sequences to candidate identities - without human intervention. We evaluate the pipeline on a spatio-temporal dataset containing simulated location points anchored at and around true home and work addresses, focusing on a high-risk disclosure scenario. Our results demonstrate that, from spatio-temporal data and public sources alone, our agentic AI successfully re-identified 18 of the 25 re-identifiable individuals (72%) and 18 of 43 cases overall (41.9%).
We discuss implications for Statistical Disclosure Control (SDC) practice and outline the near-future escalation that data custodians and regulators must anticipate. De facto anonymity - an implicit foundation of SDC practice - is shifting. Agentic AI strengthens the case that re-identification is reasonably likely by any means under the GDPR Recital-26 standard, at costs of minutes-and-dollars per target. 

---
# Home3D 1.0: A High-Fidelity Image-to-3D Asset Generation System for Interior Design 

**Authors**: Yiyun Fei, Guoqiu Li, Jin Song, Chuqiao Wu, Delong Wu, Hong Wu, Ziru Zeng, Haohui Chen, YinDong Kong, Jing Li, Qi Wu, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27923)  

**Abstract**: We present Home3D 1.0, a modular image-to-3D generation system that produces high-quality 3D assets from a single reference image, targeting interior design and e-commerce applications. Given a photograph of a furniture or decor item, the system outputs a mesh with physically-based rendering (PBR) materials, and the mesh can be decomposed into material-specific components. The pipeline is organized into four tightly coupled modules: Geometry reconstructs a watertight mesh through latent SDF modelling with a geometry VAE and a coarse-to-fine flow-matching DiT; Texture predicts multiview albedo observations, reprojects them onto the mesh, and completes unseen surface regions with a 3D texture field; Material uses MatWeaver to obtain component masks through video-based segmentation and UV-space voting, then retrieves and bakes PBR maps from a curated material library through hierarchical multi-modal matching; and Parts generates material-editable semantic part meshes with a PartVAE and PartDiT, decoding multi-head part-specific SDF fields in one pass. Each module is evaluated independently with dedicated metrics, highlighting both the current system capability and the remaining gaps toward broader deployment. 

---
# Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding 

**Authors**: Shuimu Chen, Yuteng Chen, Yuanshen Guan, Zebang Cheng, Zeyu Zhang, Shengqian Qin, Bin Xia, Jiaran Li, Wenming Yang, Fei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.27922)  

**Abstract**: Current multimodal reflection mechanisms for long video understanding predominantly rely on closed-loop self-reflection within internal parameters. Lacking objective external evidence, models are frequently trapped in blind confidence and often fail to correct errors. Furthermore, applying reinforcement learning to multi-stage reflection pipelines introduces severe policy coupling, which is exacerbated by a critical scarcity of dedicated training data. To address these limitations, this work proposes Reflect-R1, the first Evidence-Driven self-correction framework for long video understanding. The framework constructs a three-stage pipeline consisting of intuition, verification, and arbitration. By dynamically retrieving objective visual evidence to verify initial intuitions and autonomously executing multiple temporal searches to resolve conflicts, it completely breaks the hallucination loop. To overcome policy coupling, we design a stage-decoupled reinforcement learning algorithm named SD-GRPO that independently computes advantage functions across different reasoning stages. Concurrently, we construct a dataset of 120K samples to bridge the training data gap. Extensive experiments on benchmarks such as VideoMME and LongVideoBench demonstrate that Reflect-R1 achieves state-of-the-art performance. Our method significantly improves the genuine rectification rate and enables authentic self-correction strictly grounded in objective evidence. 

---
# Every Step of the Way: Video-based Parkinsonian Turning Step Counting 

**Authors**: Qiushuo Cheng, Jingjing Liu, Catherine Morgan, Alan Whone, Majid Mirmehdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27918)  

**Abstract**: As a prominent symptom of Parkinson's disease (PD), turning impairment is evaluated through parameters such as turning angle, duration, and particularly, the number of steps required to complete a turn, which directly reflects motor dysfunction. Accurate step counting is challenging due to variability in real-world turning movements and atypical shuffling patterns in parkinsonian gait. Existing methods are predominantly wearable-based, requiring users to wear and manage dedicated devices, which can be inconvenient for continuous daily use. To address this, we propose a passive, video-based framework that estimates step count in a coarse-to-fine manner using diverse motion representations. Specifically, an initial step count is estimated from foot movement signals derived from 3D human mesh recovery, providing high-level motion structures. To incorporate fine-grained motion details, a motion encoder learns complementary gait dynamics from mesh and optical flow to refine the initial estimate. In this process, coarse foot movement signals query the pixel-level motion cues via cross attention to capture subtle parkinsonian gait dynamics. To handle varying video lengths, we partition each video into clips and integrate clip-wise motion embeddings via multiple instance learning (MIL) for step count residual prediction. Extensive experiments show our method consistently outperforms existing step counting methods on real-world PD turning datasets. 

---
# Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs 

**Authors**: Avni Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2606.27909)  

**Abstract**: Theory-of-mind evaluations of large language models typically use dyadic social-deduction games, where every observable cue points to a single hidden side, so a model with strong language priors can score well without ever simulating opponents' incentives. We extend the Werewolf game with a Jester, a third faction whose utility on peer suspicion is inverted because it wins by being voted out, so optimal play requires reasoning across three opposing utility functions. Across 60 games on GPT-4.1, DeepSeek-V3.1, and Llama-3.3-70B with Jester self-learning on and off, the Jester wins 60-70% of games while Werewolves never exceed 20%, and GPT-4.1 wolves vote the Jester out on day 1 in 60-70% of games, a strictly self-defeating action. Self-learning helps DeepSeek and Llama but hurts GPT-4.1, with the cost landing on Villagers rather than Werewolves. Only DeepSeek learns the subtle strategy of looking suspicious without looking intentionally suspicious, and it gains the most from the loop. Triadic incentive structure exposes a layer of multi-agent reasoning that dyadic deduction games leave invisible. 

---
# SEADA: An efficient methodology for optimizing mixed-precision DNNs on multi-precision spatial architectures 

**Authors**: Leandro Fiorin, Marco Ronzani, Cristina Silvano  

**Link**: [PDF](https://arxiv.org/pdf/2606.27884)  

**Abstract**: Mixed-precision computation has been introduced in deep neural networks (DNNs) as an effective approach to reduce latency, energy consumption, and memory footprint. However, efficiently mapping mixed-precision networks onto multi-precision spatial architectures poses several challenges. These include determining the appropriate precision for each layer, balancing layer-wise accuracy sensitivity to quantization against architectural heterogeneity and system-level constraints, and accurately estimating the system-level cost of heterogeneous precision assignments. This work presents SEADA, an efficient methodology designed to address these challenges. SEADA comprises: (i) a configurable system-level analytical cost model of a multi-precision spatial accelerator architecture; (ii) a fast mapping tool that identifies near-optimal mappings of DNN workloads onto the target integer accelerator; (iii) analytical models for floating-point layers to estimate the overall benefits of mixed-precision execution; and (iv) a per-layer precision selection methodology based on bit-level entropy, enabling efficient assignment across multiple numerical precisions. SEADA's efficiency provides designers with a robust framework for the design-space exploration of multi-precision architectures. 

---
# A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts 

**Authors**: Emanuela Boros  

**Link**: [PDF](https://arxiv.org/pdf/2606.27881)  

**Abstract**: Temporal variation poses a unique challenge for named entity recognition (NER) in historical texts, where entities drift in surface form and salience across time. While language models (LMs) have made progress in various NLP tasks, their ability to reason about temporality, especially in diachronic contexts, remains limited or at least, questionable. In this paper, we systematically study how temporal metadata can be structurally embedded into NER models using a range of lightweight fusion strategies. We experiment with both absolute and relative temporal representations, injected into Transformer-based architectures via early or late fusion mechanisms such as cross-attention, adapters, and concatenation. Our evaluations on French and German historical datasets reveal that late fusion strategies yield more robust and temporally generalisable performance, particularly in early and noisy periods. 

---
# SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion 

**Authors**: Haoyu Zhang, Meng Liu, Qianlong Xiang, Kun Wang, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2606.27876)  

**Abstract**: Spatial intelligence is essential for low-altitude unmanned aerial vehicle (UAV) perception, collaboration, and navigation. However, existing UAV benchmarks often emphasize image-level recognition, single-view understanding, or narrow answer formats, leaving 3D spatial inference, multi-view collaboration, scene dynamics, and diverse task formulations insufficiently evaluated. To address these gaps, we introduce SpatialUAV, a real low-altitude UAV benchmark comprising 4,331 curated instances across 14 fine-grained task types, covering semantic discrimination, spatial relation, aerial--aerial collaboration, aerial--ground collaboration, and motion understanding. SpatialUAV organizes all samples into a unified visual-input--question--answer schema, while supporting seven input configurations and nine answer formats, including option labels, region identifiers, geometric values, cross-view correspondences, and free-form motion descriptions. To ensure reliable and grounded evaluation, our data construction pipeline integrates detector-assisted regions, depth supervision, metadata-derived rules, extensive manual annotation, blind filtering, and multi-turn human validation, together with task-specific metrics for heterogeneous outputs. Evaluating representative vision-language models across three categories, we show that current models remain far from human-level performance, with pronounced bottlenecks in cross-view association, structured grounding, geometric reasoning, and temporal viewpoint understanding. These results offer empirical guidance for advancing low-altitude UAV spatial intelligence. Code and data are available at this https URL. 

---
# S$^2$-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation 

**Authors**: Zhipeng Xie, Zongyi Han, Xiangyi Wei, Shiliang Sun, Yang Li, Jing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27872)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated strong capabilities in robotic manipulation, but their performance degrades significantly in long-horizon tasks due to cumulative error propagation. This limitation largely arises from static feature fusion mechanisms that rely on fixed weights to combine visual, language, and action representations, preventing the model from adapting to different phases of task execution. To address this limitation, we propose S$^2$-VLA, a framework that introduces a State-Space Guided Adaptive Attention (SSGAA) mechanism. SSGAA maintains a belief state that tracks task progression and generates dynamic gating weights to adaptively fuse information from three complementary sources visual features for spatial perception, task intents for high-level task planning, and temporal action sequences for execution consistency. This adaptive fusion allows the model to shift its focus throughout task execution, aligning with the evolving requirements of different task stages. Despite its compact 2B parameter size, S$^2$-VLA consistently outperforms larger 7B-scale models and achieves state-of-the-art performance on long-horizon manipulation benchmarks, including LIBERO and SimplerEnv. highlighting the importance of adaptive feature fusion for long-horizon robotic manipulation. 

---
# GNBAN: Graph Neural Basis Attention Networks for Long-Horizon Forecasting over Large Entity Sets 

**Authors**: Janak M. Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2606.27863)  

**Abstract**: Demand forecasting at the bottom of a retail hierarchy requires predicting tens of thousands of correlated long-horizon series across products, stores, and regions. Modern systems must scale across massive catalogs, capture shared demand dynamics, and remain interpretable enough to be trusted. Classical statistical methods need a separate model per series and are hard to manage at scale; deep autoregressive models struggle as the joint state grows to tens of thousands of dimensions; and recent graph-based forecasters, while capturing cross-entity dependencies, often produce opaque long-horizon forecasts. We propose GNBAN (Graph Neural Basis Attention Network), an end-to-end architecture combining heterogeneous graph representation learning with an interpretable basis-decomposition head. Retail data are represented directly as a heterogeneous graph derived from the relational schema, so a single model serves the entire catalog. Rather than predicting the horizon directly, GNBAN decomposes each forecast into trend, seasonal, and generic components. Its key innovation is a per-basis attention mechanism: each basis function keeps its own learnable query and retrieves information independently from the entity's historical neighborhood, letting different bases specialize to distinct temporal patterns while preserving interpretability. On two large-scale benchmarks, M5 Walmart and Favorita Grocery Sales, evaluated under matched protocols, GNBAN improves volume-weighted WRMSSE by roughly 4-5% over a matched graph baseline. Qualitative analysis shows the learned decomposition exposes trend, seasonal, and residual demand drivers without post-hoc explanation methods. These results demonstrate that scalable relational forecasting and interpretable forecast decomposition can be achieved together in a unified graph-based framework. 

---
# Applicability of memorization indicators for early spotting of overfitting while recalibrating sEMG-decoders on low sample sizes 

**Authors**: Stephan J. Lehmler, Tobias Glasmachers, Ioannis Iossifidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.27855)  

**Abstract**: Deep learning models for surface electromyography (sEMG) can benefit substantially from subject-specific (re-)calibration, since no sufficiently large and diverse datasets are available to train fully generic decoders. However, for user acceptance, the number of repetitions that can realistically be collected during calibration is severely limited, which increases the risk of overfitting and, in extreme cases, can even degrade performance compared to the uncalibrated model. Classical overfitting indicators such as validation performance and regularization with early stopping are difficult to apply in this low-sample regime, as they require additional held-out data that is rarely available in practical calibration scenarios. In this work, we investigate a recently proposed class of memorization indicators based solely on the activation statistics of rectified linear units (ReLU) in deep neural networks, which can be computed directly from training data without any extra validation set. We conduct a transferlearning experiment on a benchmark sEMG dataset, where a convolutional neural network is first pre-trained on multiple subjects and subsequently fine-tuned on individual users using only a small number of repetitions. During calibration, we monitor both decoding performance and the activation behaviour of the last hidden layer. Our results provide first evidence that decreases in test accuracy during fine-tuning are ac companied by characteristic changes in activation rates, indicating that activation-based memorization indicators are a promising tool for early spotting of unsuccessful learning in low-sample sEMG calibration settings. 

---
# WattLayer: Get Layers Right to Estimate Inference Energy of Neural Networks 

**Authors**: Adrien Sardi, Marie-Line Alberi Morel, Sara Alouf, Frédéric Giroire, Joanna Moulierac  

**Link**: [PDF](https://arxiv.org/pdf/2606.27841)  

**Abstract**: The widespread adoption of Artificial Intelligence (AI) has led to increasing concerns about energy consumption, yet there is a lack of standardized methodologies to accurately estimate AI inference energy consumption, particularly across various tasks and architectures. In this study, we propose a task independent, layer-wise energy estimation model for AI architectures. Our model is evaluated on a large dataset of more than 100,000 layers for 295 neural network architectures across 3 widely-used tasks and 3 distinct hardware platforms. Our approach achieves a median error of 19.6%, outperforming state-of-the-art methods. We further show that layer-wise decomposition generalize to new tasks without complete retraining, by leveraging shared layers across architectures. It offer tools, insights and a precise methodology to empower stakeholders in designing energy-efficient AI systems. 

---
# Hippocampus-DETR: An Explicit Memory Object Detection Framework Based on Hippocampus Modeling 

**Authors**: Zhaoning Shi, Bo Ma, Hao Xu, Zepeng Yang, Bo Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27831)  

**Abstract**: This paper addresses the lack of explicit memory mechanisms in current object detection models and proposes Hippocampus-DETR, a novel detection framework based on biological hippocampal memory modeling. This framework integrates a hippocampal memory network module, HipNet, into the DETR architecture and systematically simulates the anatomical structure and functional organization of hippocampal subregions, including the entorhinal cortex, dentate gyrus, CA3, CA1, and subiculum. Through this design, Hippocampus-DETR realizes pattern separation, pattern completion, importance filtering, and information integration of visual encoding features. During training, different memory submodules are optimized using a layer-wise training strategy, ultimately forming a memory system with memory retrieval and completion capabilities. Experimental results demonstrate that Hippocampus-DETR achieves higher detection accuracy than current mainstream models. More importantly, models equipped with this framework also exhibit excellent generalization ability and data efficiency in tasks such as few-shot image classification, multimodal feature construction, and image restoration. Subsequent experiments further validate the functional necessity and internal interpretability of each memory submodule. This study not only provides a novel object detection framework, but also offers a feasible technical pathway for integrating neurocognitive mechanisms with deep learning models, highlighting its significant value in improving model learning efficiency and task robustness. The project is available at this https URL. 

---
# Pepti-drift: Toxicity-Repulsive Drifting for Antigen-Conditioned Discrete Peptide Generation 

**Authors**: Takashi Fujiwara, Hikaru Shindo, Kaushalya Madhawa, Jun Jin Choong, Keisuke Ozawa  

**Link**: [PDF](https://arxiv.org/pdf/2606.27824)  

**Abstract**: Peptides are a promising therapeutic modality that combine the chemical tunability of small molecules with the target specificity of macromolecular therapeutics. However, designing antigen-specific binding peptides while avoiding toxicity remains a major challenge for therapeutic peptide discovery. Here, we present Pepti-drift, a toxicity-aware latent refinement framework that generates peptide candidates through a single antigen-conditioned drift step. In a peptide embedding space, Pepti-drift learns to attract generated peptide latents toward antigen-matched binding peptides while repelling them from toxicity-associated regions. This is challenging because binding-promoting physicochemical features often overlap with toxicity-associated features in peptide representation space. To address this, we introduce a warm-up strategy to stabilize this competing objective by first learning binding-oriented attraction and then increasing toxicity repulsion. 

---
# Parameter-Efficient Quantum-Inspired Fast Weight Programmers for Traffic-Matrix Forecasting 

**Authors**: Kuo-Chung Peng, Jiun-Cheng Jiang, Chun-Hua Lin, Tai-Yue Li, Nan-Yow Chen, Samuel Yen-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27821)  

**Abstract**: Traffic matrices (TMs) capture network-wide origin-destination demand and are central to traffic engineering, yet accurate whole-matrix forecasting remains challenging when prediction must be performed under the memory, update, and training-budget constraints of online network control. This paper investigates whether compact quantum-inspired recurrent models can provide effective TM forecasts without relying on dedicated graph, transformer, or diffusion modules. We adapt gated quantum-inspired Kolmogorov-Arnold network fast-weight programmers (QKAN-FWPs) to direct multi-step Abilene TM forecasting, where each model predicts the next 20 five-minute frames of a 144-channel origin-destination (OD) matrix from a two-hour history. We benchmark three QKAN placement variants against a matched-size long short-term memory (LSTM) network, a larger LSTM, and a classical gated fast-weight programmer under a shared fixed-budget training protocol. Among the evaluated recurrent models, G-QKANFWP achieves the best pooled root-mean-square error (RMSE), while using only 22.4% of the larger LSTM. It also outperforms both the matched-size LSTM and the classical G-FWP baseline, indicating that the gain is not due to gated fast-weight framework alone. Convergence and channel-wise analyses further show that the quantum-inspired variants obtain lower validation-loss area under the learning curve (AULC) than matched-size recurrent baselines, while G-QKANFWP and GQKAN-FWP achieve substantially more OD-channel wins. These results identify a classical slow programmer with a quantum-inspired fast programmer as a promising accuracy-efficiency design for resource-conscious network traffic-matrix forecasting. 

---
# Optimizing Teacher-Student Partitioning for Scalable Knowledge Distillation on HPC Systems 

**Authors**: Adrian P. Dieguez, Victor Conchello Vendrell, Alex Batlle, Vinnam Kim, Jordi Ros-Giralt, Harris Teague  

**Link**: [PDF](https://arxiv.org/pdf/2606.27797)  

**Abstract**: Knowledge Distillation (KD) enables training smaller student models under the guidance of larger teacher models, and the widely adopted TRL library implements it. Yet, TRL treats both models symmetrically, missing opportunities to exploit their pronounced asymmetry in memory footprint, and communication requirements. This paper presents an HPC-aware methodology for KD that decouples teacher and student partitioning efficiently. Our approach achieves up to 67% higher samples-per-second than TRL by avoiding unnecessary teacher-model data structures and selecting the best split strategy. We combine vertical and horizontal partitioning of models, deriving an analytical expression that identifies the existence of inflection points between splitting regimes. These results showed that exploiting teacher--student asymmetry through topology-aware parallelism notably accelerated GKD training on production HPC clusters at our company 

---
# Position Bias Correction is Insufficient for One-Pass Attention Sorting 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27793)  

**Abstract**: Long-context language models suffer from position bias, where information in middle positions is underutilized. Attention Sorting addresses this by iteratively reordering documents based on attention patterns, but its multiple sort-and-generate cycles increase deployment cost. We hypothesize that position bias is the primary bottleneck and propose Debiased One-Pass Attention Sorting, which estimates a per-prompt position-bias curve from the low-attention majority of documents and uses it to correct raw attention scores (via subtraction or division) to enable single-pass sorting. Our experiments on two models refute this hypothesis in the tested setting: on LLaMA-2-7B-32K-Instruct, debiasing produces identical results to uncalibrated single-pass sorting (94.83\% containment accuracy), while on YaRN-Llama-2-7b-64k, debiasing improves accuracy by 8.67 percentage points but remains 14.84pp behind iterative sorting, closing only 37\% of the gap. These results suggest that position-bias correction is insufficient to match iterative sorting, and that repeated reordering provides additional benefits beyond bias correction. 

---
# NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27791)  

**Abstract**: Hybrid attention models that mix full and sliding-window attention across layers offer a promising approach to efficient long-context inference, but the critical question of \emph{which layers} should retain full attention remains unsolved. Existing methods use either fixed periodic patterns or attention-based heuristics that may not capture what matters for downstream accuracy. We propose NLL-guided layer selection, a training-free method that directly measures each layer's importance by computing the negative log-likelihood degradation on answer tokens when that layer uses sliding-window instead of full attention. On LongMemEval with Qwen3-4B, our method achieves 64.6\% accuracy using only 1/4 full-attention layers, matching the 1/2-FA periodic baseline (65.0\%) while halving the computational budget. NLL-guided selection outperforms the SWAA-reported periodic 1/4-FA baseline by 10.4 percentage points and a matched LightTransfer-style baseline by 26.4 percentage points. De-confounding analysis shows the signal is consistent with long-range attention needs rather than generic layer sensitivity. The method requires only $\sim$15 minutes of one-time calibration, advancing the efficiency-accuracy Pareto frontier for long-context LLM deployment. 

---
# SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation 

**Authors**: Ruochang Li, Pengcheng Huang, Zhenghao Liu, Yukun Yan, Huiyuan Xie, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.27786)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLMs by incorporating external knowledge to support response generation. However, conflicts between retrieved context and parametric knowledge have emerged as a critical challenge in RAG systems. To mitigate such conflicts, numerous studies have attempted to identify and edit knowledge-related internal neurons, aiming to improve the ability of LLMs to rely on contextual evidence during generation. However, these neuron-level approaches may introduce unintended cascading effects that compromise the general capabilities of LLMs, as the modified neurons are often entangled with broader model behaviors and functionalities. In this paper, we introduce SHIFT, a novel framework that reformulates neuron-level modification as learnable gate modulation, allowing LLMs to adaptively regulate internal activations for knowledge conflict resolution. Technically, our SHIFT equips LLMs with a lightweight gate module and optimizes fewer than 0.01% trainable parameters while keeping the backbone model frozen. During generation, the gate module adjusts the model's internal representations to adaptively leverage contextual and parametric knowledge. Extensive experiments on six datasets validate the effectiveness of our SHIFT in comparison with various competing baselines. All datasets and code are available at this https URL. 

---
# Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27785)  

**Abstract**: Training-free compression methods for large language models (LLMs) often use calibration data to guide compression decisions. ROCKET, a recent method combining sparse-dictionary factorization with multi-choice knapsack problem (MCKP) allocation, derives its per-layer factorization from an output reconstruction objective but uses weight-space Frobenius error as the MCKP allocation cost. We investigate whether aligning the allocation cost with the output-space objective improves compressed model fidelity. On Qwen3-8B at 50\% compression, our ROCKET-ActCost achieves +0.8 percentage points higher average accuracy across 8 zero-shot benchmarks (53.1\% vs 52.3\%), but increases WikiText perplexity by 16\% (61.46 vs 52.98). This accuracy-perplexity tradeoff reveals that different allocation objectives favor different downstream metrics. The high correlation ($>$0.99) between weight-space and output-space errors limits allocation divergence, explaining the modest effect size. On Llama-3.2-1B at 20\% compression, the two methods produce near-identical results (53.3\% vs 53.5\% accuracy, 14.45 vs 14.66 PPL), suggesting that the effect of the cost function is minor at lower compression ratios. 

---
# Improving Adversarial Robustness via Activation Amplification and Attenuation 

**Authors**: Taïga Gonçalves, Yongsong Huang, Tomo Miyazaki, Shinichiro Omachi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27784)  

**Abstract**: The existence of adversarial attacks is often attributed to the presence of non-robust features in neural networks. While prior defenses reduce their impact via pruning, masking, or feature recalibration, we instead propose to jointly learn to amplify and attenuate these signals through a simple activation scaling mechanism. To this end, we introduce Activation Amplification and Attenuation (A3), a lightweight plug-in module that enhances adversarial robustness with minimal modifications of the activations. A3 dynamically rescales the activations using a learnable mask and a scaling factor derived from the original activation magnitudes. The influence of adversarial perturbations can be amplified or attenuated using the same learnable parameters by simply flipping the sign of the scaling operation. The amplified signals serve as negative references to construct novel contrastive and ranking loss functions. Experimental analysis shows that learning to degrade the predictions in amplification mode simultaneously improves adversarial robustness in attenuation mode. Moreover, A3 relies on only a small number of learnable parameters, with most of its behavior being determined by the scaling mechanism rather than additional network capacity. Extensive experiments demonstrate that integrating A3 into different backbones, datasets, and training methods consistently improves adversarial robustness while introducing negligible computational and memory overhead compared to existing plug-in modules. Code is available at: this https URL. 

---
# RS-Diffuser: Risk-Sensitive Diffusion Planning with Distributional Value Guidance 

**Authors**: Shiqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2606.27766)  

**Abstract**: Offline reinforcement learning enables policy learning from fixed datasets without additional environment interaction, making it appealing for safety-critical applications where online exploration is costly or unsafe. Diffusion-based decision-making methods have recently achieved strong performance in offline RL by modeling rich, multimodal trajectory distributions. However, existing diffusion planners are typically risk-neutral and therefore may overlook rare but catastrophic outcomes that are crucial in real-world deployment. In this work, we propose RS-Diffuser, a risk-sensitive offline diffusion planning framework that combines diffusion-based trajectory generation with distributional value critics. RS-Diffuser learns a diffusion planner over future state trajectories, a separate inverse dynamics model for action decoding, and a Monte Carlo distributional critic that estimates the full return distribution of candidate plans through quantile regression. At sampling time, we incorporate a risk-sensitive guidance signal into the denoising process, using gradients computed from tail-aware objectives such as Conditional Value at Risk to steer generation toward desired risk profiles. As a result, a single trained model can flexibly produce risk-averse, risk-neutral, or risk-seeking behaviors by changing only the inference-time risk parameter. Extensive experiments on risk-sensitive D4RL and risky robot navigation benchmarks demonstrate that RS-Diffuser achieves state-of-the-art performance, improving both overall return and worst-case robustness while reducing safety violations. 

---
# Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models? 

**Authors**: Guoheng Sun, Kaixi Feng, Shwai He, Xiaochuan Gong, Yexiao He, Ziyao Wang, Zheyu Shen, Wanghao Ye, Ramana Rao Kompella, Gaowen Liu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.27755)  

**Abstract**: Vision-Language-Action (VLA) models enable instruction-driven robotic manipulation, but they inherit oversized language backbones from pretrained VLMs whose capacity far exceeds what is needed for short robotic instructions. This raises a basic question: how much of a VLA model is actually necessary for closed-loop control? In this work, we study architectural redundancy in VLA models by using transformer block removal as a controlled intervention. We introduce \textbf{Drop-Then-Recovery (DTR)}, an analysis protocol that removes selected blocks from a pretrained VLA model and then fine-tunes the resulting model to measure whether the removed capacity was necessary for downstream control. To make this intervention reliable, we propose \textbf{GateProbe}, a one-shot virtual-gate sensitivity metric that ranks blocks by their contribution to the downstream action loss. Across multiple VLA architectures, manipulation benchmarks and even real-robot industrial scenarios, we find a strong asymmetry in post-removal recoverability: \ul{\textit{language backbones are highly redundant for standard robotic manipulation tasks, whereas vision and action pathways are substantially less tolerant to removal}}. On LIBERO, removing half of the LLM blocks even improves OpenVLA-OFT from 95.0% to 98.3% under the same downstream fine-tuning budget, and retaining only two language blocks still recovers baseline-level performance. These results suggest that current VLA benchmarks may exert limited pressure on deep language grounding and compositional instruction understanding, and that future VLA architectures should allocate capacity more deliberately across language, vision, and action components. The code is available at this https URL. 

---
# From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection 

**Authors**: Stefano Giacomelli, Stefano Damiano, Claudia Rinaldi, Fabio Graziosi, Toon van Waterschoot  

**Link**: [PDF](https://arxiv.org/pdf/2606.27751)  

**Abstract**: This report investigates the extension of pretrained General-Purpose Audio Tagging (GP-AT) models toward spatially grounded Sound Event Localization and Detection (SELD). The proposed AT2SELD framework couples a pretrained AT backbone with compact First-Order Ambisonics (FOA) spatial processing, track-wise SED and Cartesian DOA estimation, permutation aware supervision, and calibration. It characterizes how semantic audio priors support localization-aware scene analysis under data, computation, and deployment constraints. The framework is developed through informed multi-stage Neural Architecture Search (NAS). Stage 1 shows that spectral FOA descriptors, based on magnitude, phase, and Intensity Vectors (IVs), provide the most reliable interface for semantic-to-spatial transfer. Stage 2 identifies early residual spatial encoding as the main capacity-sensitive component, while late track-wise abstraction and recurrent smoothing act mainly as refinement stages. Stage 3 shows that late cross-stitch coupling improves semantic-spatial interaction, whereas early fusion is costlier and less effective. Diagnostic evaluation analyzes the selected architecture under class balancing, focal loss, activity-conditioned DOA supervision, threshold calibration, and transfer across STARSS23, TAU2019, TAU-NIGENS2020, and TAU-NIGENS2021. Focal loss improves the activity point, active-only DOA supervision mitigates inactive target dominance, and validation-selected thresholds recover calibration without replacing spatial learning. Cross-dataset and oracle-activity analyses indicate strong fixed source localization on TAU2019, transferable representations from TAU NIGENS2021, and meaningful but uncertain behavior on STARSS23. Overall, GP-AT priors appear promising for SELD design when embedded in spatial-aware architectures and optimized through integrated calibration and deployment oriented strategies. 

---
# Flexformer: Flexible Linear Transformer with Learnable Attention Kernel 

**Authors**: Haoran Zhang, Feng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.27748)  

**Abstract**: Transformer models rely on attention mechanism to capture long-range dependencies but suffer from quadratic complexity, limiting their scalability to long sequences. Kernel-based linear attention reduces this complexity but typically relies on fixed or weakly learnable kernels, restricting expressiveness and performance. In this work, we propose Flexformer, a flexible linear Transformer that learns attention kernels in a fully data-driven manner. Flexformer builds on random Fourier feature-based linear attention and treats spectral frequencies as trainable parameters, enabling the model to learn a broad family of attention kernels.
We develop both stationary and nonstationary variants, with the latter offering strictly greater expressiveness.
Extensive experiments on language modeling and sequence classification demonstrate that Flexformer consistently outperforms baselines. Moreover, Flexformer can be effectively distilled from pretrained Transformers to recover softmax attention and exhibits strong kernel transferability across domains, achieving both high efficiency and competitive performance on long-sequence tasks. 

---
# End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inference 

**Authors**: Yuhang Chen, Jinhao Duan, Ruichen Zhang, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Parish Aggarwal, Frank Shyu, Luke Simon, Sandeep Pandey, Tianlong Chen, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27743)  

**Abstract**: Large Language Models (LLMs) inference is typically deployed under a static resource assumption, where models execute a fixed computational graph regardless of the runtime environment. However, real-world cloud infrastructure is inherently dynamic, characterized by fluctuating availability (e.g., spot instance preemption) and tiered Quality-of-Service requirements. In such volatile settings, static models are inflexible: they either crash under resource constraints or waste compute on redundant operations. To bridge this gap, we propose Learning to Allocate (L2A), an end-to-end framework for resource-adaptive inference. Unlike prior methods that condition only on input difficulty, we formulate inference as a constrained allocation problem conditioned on both the input and the runtime resource budget itself. We introduce lightweight, budget-conditioned and input-aware gating networks integrated into the LLM. These gates are trained via a unified objective that jointly optimizes task performance, logical consistency, and resource costs along three axes matching how real-world dynamics manifest: layer skipping for memory and depth pressure, head pruning for throughput contention, and reasoning-token reduction for latency tightening. This lets the model learn a budget-aware policy beyond input difficulty alone: it adaptively configures its computational footprint with respect to real-time resource dynamics, maximizing reasoning depth when resources permit while enforcing strict frugality when budgets tighten. A single L2A model traces the entire compute-accuracy Pareto frontier on Llama-3-8B and Qwen-3-4B: at up to 34% realized layer sparsity, it stays within 0.6% of the dense baseline on GSM8K, with the same gap holding zero-shot on out-of-distribution tasks, while every static or heuristic baseline requires a separately tuned model and still drops by 5-10% at comparable inference time. 

---
# KG2Cypher: Data-Centric Pipeline for Building Enterprise Text-to-Cypher Systems 

**Authors**: Minjun Choi, Yerin Kim, Junghyuk Seo, Sujin Mo, Hyemin Lee, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2606.27742)  

**Abstract**: Enterprise Knowledge Graphs (KGs) are increasingly used for internal search, analytics, and question answering, but building natural-language interfaces for private enterprise graphs remains costly. We present KG2Cypher, a data-centric pipeline for building enterprise text-to-Cypher systems from existing KGs. KG2Cypher first constructs an executable Cypher query from observed graph facts and then uses LLMs to generate its associated natural-language question. The resulting Text-Cypher pairs are validated with an LLM judge and human validation, and are converted into candidate-aware SFT data. The trained generator is served with class-conditioned schema prompting, entity retrieval, and LoRA-based inference. We evaluate KG2Cypher in Korean enterprise settings, where short search-style queries and schema paraphrases make language grounding difficult. LoRA SFT improves execution-result F1 from 0.806 to 0.950 on broadcast-program queries and from 0.70 to 0.92 on company queries. In an 11-class setting, KG2Cypher achieves 95.2% exact match, 99.9% execution rate, and 0.964 execution-result F1. 

---
# Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation 

**Authors**: Yuhang Chen, Xianfeng Wu, Jinhao Duan, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Parish Aggarwal, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27732)  

**Abstract**: Discrete diffusion language models (dLLMs) recover masked tokens in parallel, offering significant speedups over autoregressive (AR) generation. However, such promising frameworks face a fundamental architectural design dilemma: \ding{182} Adopting bidirectional attention achieves strong generation quality by allowing each position to access the full context, but is inherently incompatible with KV caching, limiting inference throughput in batch-serving scenarios; \ding{183} Conversely, causal attention enables efficient cached inference but loses all right-side context, substantially degrading generation quality. This paper introduces Bifocal dLLMs, a new paradigm that resolves this dilemma through \emph{asymmetric bidirectional context}. Analogous to bifocal lenses, we instantiate the paradigm as \textbf{R2LM} (Right-to-Left Mamba), which combines two complementary mechanisms: $a$) standard causal attention providing precise left-context with full KV cache compatibility, while $b$) a lightweight reverse Mamba SSM sidecar supplying compressed right-side context without breaking cacheability. Comprehensive experiments on continued pretraining of Qwen3-1.7B with 60B tokens demonstrate that R2LM achieves $2.4\times$ to $12.9\times$ higher throughput than bidirectional dLLMs and $1.9\times$ to $2.9\times$ speedup over AR baselines in batch serving through parallel decoding with KV caching, while exceeding the causal baseline on most benchmarks and surpassing the bidirectional dLLM on average. 

---
# Enhancing Numerical Prediction in LLMs via Smooth MMD Alignment 

**Authors**: Zhuo Zuo, Li Yue, Wenhao Zheng, Chenpeng Wang, Xianggen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27731)  

**Abstract**: Despite their strong general capabilities, large language models (LLMs) often remain unreliable when outputs must be numerically precise. A key reason is the training objective: standard cross-entropy treats numeric tokens as unstructured categories and ignores the metric structure of their values. We address this mismatch with Smooth Maximum Mean Discrepancy (SMMD), which builds on the classic MMD by incorporating value-distance kernels over numeric tokens and graph-based smoothness. With this kernel defined over a numeric sub-vocabulary, SMMD aligns the predicted numeric distribution to the target via kernel matching and smooths the prediction-target residual over the induced kernel graph to encourage local consistency. We evaluate SMMD on four numeric-target tasks: mathematical reasoning, arithmetic calculation, clock-time recognition, and chart question answering, across multiple open-weight LLM and VLM backbones. SMMD consistently improves accuracy over both cross-entropy and recent numeric-target losses; analyses show complementary effects between MMD and smoothness and underscore the importance of distance-based kernel design. Code is available at this https URL. 

---
# Do Speech Emphasis Models Generalize across Languages and Emotions? 

**Authors**: Megan Wei, Deepali Aneja, Jiaqi Su, Yunyun Wang, Haonan Chen, Zeyu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.27717)  

**Abstract**: Prosodic emphasis varies across languages, emotions, and speaking styles, yet existing emphasis detection models are largely trained and evaluated on monolingual neutral read speech. We introduce MMEE (Multilingual Multi-Emotion Emphasis), a corpus of 10,000 professionally recorded expressive utterances (14.13 hours) across 7 languages and 34 emotion/style categories, with three-level perceptual labels (10 annotations per sample). We benchmark two state-of-the-art architectures under monolingual, cross-lingual, multilingual, cross-emotion, cross-dataset, and data-scale settings. Monolingual models show limited zero-shot transfer, degrading across typologically distant languages, while multilingual training substantially improves robustness. Models transfer robustly between high- and low-arousal emotions; bidirectional transfer between synthetic and perceptual benchmarks suggests shared prosodic structure; and performance stays robust even at smaller training scales. 

---
# The Simulacrum: Decision-Theoretic Pretraining for Near-Optimal Time-Series Forecasting and Inference 

**Authors**: Pablo Montero-Manso, Marcel Scharth  

**Link**: [PDF](https://arxiv.org/pdf/2606.27711)  

**Abstract**: We introduce a neural network-based framework for learning time series estimators through a process we term decision-theoretic pretraining. Analysts specify a generative world, a distribution over data-generating processes, and a target decision objective. A neural network trained on stratified simulations from this world approximates the corresponding optimal decision rule, yielding a neural estimator that provides forecasts, parameter estimates, predictive intervals, or model-selection for zero-shot inference on previously unseen time series.
The joint specification of the generative world and objective enables the estimators to directly approximate process-level, finite-sample properties: near-optimal risk, bias control, minimax performance, and uniform calibration. Our experiments demonstrate that these neural estimators can outperform traditional baselines such as maximum likelihood estimation and model selection via AICc, for the same model structural model classes. Furthermore, even when trained purely on simulations of structural models, they achieve competitive or state-of-the-art forecasting accuracy on major real-world benchmarks, compared with statistical, neural or large pre-trained models.
We illustrate the framework by addressing two longstanding challenges: finite-sample bias and miscalibration in AR(p) models, and the forecast combination puzzle. These applications highlight the approach's main advantage: its ability to approximate solutions to analytically intractable or computationally prohibitive time series problems, including complex structural equations or optimality criteria. Ultimately, by enabling explicit control over decision-theoretic trade-offs, the framework equips analysts with highly efficient estimation tools tailored to their specific analytical needs. 

---
# Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning 

**Authors**: Austin MY Cheung, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27709)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) for social warmth degrades factual reliability and increases sycophancy. We investigate a related but distinct failure mode: warmth fine-tuning also weakens adversarial safety, making models more susceptible to jailbreaks and harmful output generation. We examine whether this reflects an inherent consequence of empathetic adaptation or an artifact of data construction. To address this, we introduce a persona-driven rewriting pipeline that conditions user turns on low agreeableness and pairs this with warm, de-escalating assistant responses. Across three experiments on four models, our approach reduces jailbreak susceptibility and harmful output rates relative to generic warmth fine-tuning baselines, while preserving conversational warmth. Representational probing provides suggestive evidence that this conditioning reduces the geometric alignment between warmth and compliance directions in latent space. These results show that safer empathetic fine-tuning is achievable through data design alone, without safety labels, harm detectors, or changes to the training objective. 

---
# Room for Error: Large-Scale Simulation of Over-the-Air Acoustic Attacks 

**Authors**: Andrew C. Cullen, Neil Marchant, Jiani Xie, Paul Montague, Benjamin I.P. Rubinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.27701)  

**Abstract**: While voice control is rapidly becoming a ubiquitous vector of human-AI communication, the risks facing these systems remain poorly understood. This is, in part, a product of the difficulties in scaling strictly digital adversarial workflows to the physical world. These scale barriers have led the community to abstract away key acoustic factors relating to detectability and the influence of geometry on acoustics. These methodological and metrological shortcomings undermine our understanding of risk. We illuminate these issues through real-world testing, conceptual discussions, and a novel, high-throughput reality simulation framework. By testing over 8 million adversarial evaluations, we demonstrate that acoustic awareness yields relative Word Error Rate increases of up to 94.5\% under Whisper and wav2vec. We employ this framework to explore a formalize and operationalize a Dual-Form Signal to Noise Ratio to decouple source stealth from victim attack efficacy, resolving a crucial limitation in current works. This lays the groundwork for repeatable, verifiable research that embraces, rather than abstracts, the acoustic environment. 

---
# What Was That Again? Certified Robustness for Automatic Speech Recognition 

**Authors**: Andrew C. Cullen, Neil Marchant, Jiani Xie, Paul Montague, Benjamin I.P. Rubinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.27698)  

**Abstract**: Automatic Speech Recognition systems are notoriously both sensitive to adversarial and benign perturbations. While this has been repeatedly demonstrated using reference datasets, detecting such behaviors in deployed systems is incredibly challenging, due to the absence of oracle knowledge of the true transcription. We demonstrate that employing a certification-inspired mechanism can significantly decrease WER, increase recall, and decrease the Spearman correlation between confidence and WER. We achieve this through a dual-gate diagnostic pipeline: a Two-Sided Atomic Audit that accumulates statistical wealth to certify both token existence and adversarial exclusion, and a Rank-Based Tournament that selects the winning sequence. Our evaluations across four diverse architectures demonstrate up to a 55% relative reduction in Word Error Rate, while also providing granular word- and sentence-level certifications to enhance acoustic security. 

---
# Class-frequency Guided Noise Schedule for Diffusion Models 

**Authors**: Jiequan Cui, Beier Zhu, Qingshan Xu, Xiaojuan Qi, Bei Yu, Hanwang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27696)  

**Abstract**: In this paper, we are the first to examine the correlations between class frequency and the multi-scale noise schedule within diffusion models. For score-based generative models, low-density regions often lead to inaccurately estimated scores, thereby compromising the generation quality. Although the multi-scale noise schedule can alleviate this issue during the diffusion process, low-frequency classes still face the challenge of large low-density regions, resulting in more inaccurate estimated scores than high-frequency classes. Furthermore, high-frequency classes tend to dominate the score space, causing a convergence of most data points towards generating samples from these classes. Consequently, samples generated within low-frequency classes exhibit suboptimal quality and limited diversity. To address this challenge, we propose the \textit{Class-frequency Guided (CFRG)} noise schedule, leveraging the insight that low-frequency classes should be endowed with larger-scale noises. To illustrate the effectiveness of our method, we conduct experiments on various tasks, including image generation, image classification, and text-to-image generation, using imbalanced datasets, \textit{i.e.}, CIFAR-100-LT, and ImageNet-LT. By employing the CFRG noise schedule, we achieve substantial improvements over baselines, manifesting the crucial role of frequency statistics in noise schedule design. 

---
# Halt Fast! Early Stopping for Certified Robustness 

**Authors**: Andrew C. Cullen, Paul Montague, Benjamin I.P. Rubinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.27694)  

**Abstract**: Randomized Smoothing (RS) provides rigorous robustness guarantees for neural networks without architectural constraints, yet its adoption is limited by extreme computational costs. Standard RS requires tens of thousands of model evaluations per input and forces practitioners to commit to fixed sample sizes a priori. In this work, we present a novel meta-learning framework for anytime-valid certified robustness that adaptively deploys computational resources. By using a lightweight meta-learner to predict image-specific priors for a sequential E-process, we achieve a 20-fold reduction in sample complexity compared to traditional methods while maintaining rigorous statistical guarantees. Beyond raw efficiency, we demonstrate how anytime-validity enables adaptively allocating compute based upon application-specific risk thresholds, a form of resource triage impossible under classic certification frameworks. That this is achievable while also providing similar certification performance demonstrates that our approach provides a pathway for real-time, safety-critical certification deployments. 

---
# Deployment-Side Adaptiveness in Multi-Horizon Volatility Forecasting 

**Authors**: Riku Green, Zahraa S. Abdallah, Telmo M Silva Filho  

**Link**: [PDF](https://arxiv.org/pdf/2606.27688)  

**Abstract**: In financial forecasting, predictive performance depends not only on which model is trained, but also on how the trained model is deployed. We study this issue in multi-horizon volatility forecasting. Our starting point is that a trained multi-output (MIMO) forecaster does not define a single deployable predictor: by changing the inference-time rollout rule, the same trained model induces a family of forecasts with different accuracy and cost profiles. Across 20 stock-volatility series, three forecast horizons, and architectures ranging from linear models to PatchTST, we find that non-default rollout rules often improve over standard MIMO deployment. However, the best fixed rule varies substantially across architectures and horizons, making any single static replacement unreliable. We therefore evaluate validation-based deployment policies over the induced rule family. Under the primary MSE objective, validation-selected singletons provide a low-cost improvement over default MIMO, while small rule subsets recover much of the benefit of larger ensembles at substantially lower inference cost. We also find that policy rankings are metric-sensitive: MSE-selected policies do not transfer uniformly to QLIKE, a finance-standard volatility loss. These results show that inference-time deployment is a meaningful source of adaptiveness in financial forecasting, and that trained volatility forecasters should be evaluated not only by their architecture, but also by their deployment policy. 

---
# Mitigating LLM-based p-Hacking by Preregistering for the Next LLM 

**Authors**: Maria Thomas, Kristina Gligoric, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.27687)  

**Abstract**: Large language models (LLMs) are increasingly used to generate, classify, and annotate data whose outputs feed downstream hypothesis tests. However, LLM-based research is easy to p-hack: a researcher can tune the prompts, decoding parameters, or output format until a desired result is reached. We propose a protocol to mitigate p-hacking in LLM-based research: preregistering the experiment and eligible models, and then running it on the first eligible LLM that is released after the preregistration. The researcher finalizes the procedure on current models, preregisters the analysis plan together with a set of eligible future models, and runs the confirmatory analysis on the first eligible model released afterward. Because this model does not exist at commitment time, it cannot be hacked against; furthermore, configurations that hack one model frequently do not transfer to the next. We evaluate the protocol on two tasks whose true values are known. Across 20 models from four providers and 11 LLM-analysis configurations, the protocol would have blocked successful transfer of the p-hack in 73.9% and 72.7% of cases in the two tasks. Additional analyses reveal that mitigation remains substantial under several stress tests. Finally, putting money where our mouth is, we followed our own protocol and preregistered our experiment. The preregistered experiment confirmed the protocol's effectiveness: out of the 7 configurations that hacked the prior model, the hacking failed to carry over in 6 configurations on the first eligible model released afterward. 

---
# CBD: API-Only LLM Black-Box Unlearning through Controlled Behavioral Divergence 

**Authors**: Zhiqiang Xie, Yijing Lin, Zhipeng Gao, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.27683)  

**Abstract**: Edge devices increasingly invoke large language models (LLMs) through API services for context aware edge intelligence, while edge generated data may be collected to improve LLMs and may introduce sensitive, copyrighted, harmful, or outdated information into model behavior. Machine unlearning offers a practical way to remove the influence of undesired data without retraining LLMs. However, existing methods still face two gaps. The first is API only black box access, where target model parameters and internal logits are unavailable. The second is how to preserve retained utility when unlearning target data and retained data share highly similar prompt structures or semantic patterns. To address these challenges, we propose Controlled Behavioral Divergence (CBD), an API only black box unlearning framework. CBD uses two auxiliary models to create controlled behavioral divergence between retained inputs and unlearning target inputs, converts this divergence into an unlearning relevance score, and routes unlearning related prompts away from the target LLM. To improve discrimination accuracy under high similarity between target and retained data, CBD constructs a gradient statistics based discriminative basis by estimating empirical Fisher matrices and solving a regularized generalized eigenvalue problem, guiding the unlearning signal toward target specific information rather than shared prompt structures. Compared with eleven white box and gray box unlearning baselines, CBD achieves a better unlearning utility trade off and its performance varies little across settings. On ToFU forget10, CBD approaches the retrained reference on the forget set while raising model utility to 74.90, about 15% above the second best baseline. On WMDP, it lowers hazardous knowledge accuracy to 25.68, near random guessing, while preserving MMLU accuracy of 52.67. Code is at this https URL. 

---
# From Signals to Transfer: A Factorised Study of Probe-Based Uncertainty Estimation in Large Language Models 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Cong-Duy Nguyen, Quang Minh Nguyen, Duc Anh Vu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27679)  

**Abstract**: Probe-based uncertainty estimation (UE) has emerged as a prominent approach to detect hallucinations in Large Language Models (LLMs) by learning uncertainty from internal model signals. Yet, recent methods vary simultaneously across feature design, training data construction, and evaluation setting, obscuring what actually drives performance. To address this issue, we propose a factorised study of probe-based UE under matched conditions. Our results show that raw hidden states and attention features are difficult to outperform in-domain. However, under distribution shift, structured and compressed features are more robust, suggesting that in-domain performance alone is insufficient to measure progress. Furthermore, prompting and label construction significantly affect probe behaviour. Building on these best-practice findings, we train benchmark-based pretrained probes that transfer reasonably well to open-ended factual generation, providing a stable off-the-shelf baseline. Our work encourages more deployment-oriented evaluation of probe-based uncertainty estimators. The code repository is available at this https URL. 

---
# Explainable AI for Biodiversity Monitoring and Ecological Image Analysis 

**Authors**: Brinnae Bent, Holly R. Houliston, Jiayi Zhou, Günel Aghakishiyeva, David W. Johnston  

**Link**: [PDF](https://arxiv.org/pdf/2606.27667)  

**Abstract**: Artificial intelligence is transforming biodiversity monitoring by enabling automated analysis of ecological imagery collected from camera traps, drones, satellites, underwater platforms, and other sensing systems. These tools can expand the scale and speed of conservation assessments, yet many computer vision models remain difficult to inspect, making it challenging to determine whether predictions are based on ecologically meaningful signals or on spurious correlations, sampling biases, and other artifacts that may undermine conservation decisions. We argue that explainable artificial intelligence (XAI) should become a standard component of ecological model validation because conservation practitioners increasingly depend on understanding not only whether a model is accurate, but why it is accurate. We provide practical guidance for applying XAI to three common ecological computer vision tasks: image classification, object detection, and image segmentation. To illustrate how XAI can support ecological model auditing, refinement, and deployment, we present two case studies using aerial imagery: harbor seal detection and cetacean anatomical segmentation. These examples demonstrate how explanation methods can identify biologically meaningful cues, reveal false positives driven by background and shape confounds, uncover edge and occlusion effects, and guide data collection, augmentation, and retraining strategies. More broadly, they show how explainability can help assess whether model reasoning aligns with ecological understanding. We conclude by identifying key challenges and opportunities. By making model behavior more transparent and scientifically interrogable, XAI can help ensure that AI-supported ecological evidence is more reliable, understandable, and actionable for biodiversity conservation. 

---
# Reconstructing the Developmental Trajectory of Adipocytes in Human Adipose Tissue Using Single-Cell RNA Sequencing 

**Authors**: Weny S. M Sitinjak, Humasak Tommy Argo Simanjuntak  

**Link**: [PDF](https://arxiv.org/pdf/2606.27657)  

**Abstract**: Obesity is a global health crisis associated with metabolic disorders such as type 2 diabetes and cardiovascular disease. This study employed single-cell RNA sequencing to reconstruct the developmental trajectory of human adipocytes from adipose tissue samples. Our analysis identified 15 transcriptionally distinct cell clusters, including 7 transitional states, revealing the dynamic process of adipocyte differentiation. We detected 16 functionally active signaling pathways mediating cellular communication between adipocytes and their progenitors. Among these, insulin-like growth factor (IGF) and fibroblast growth factor (FGF) pathways emerged as the most prominent networks, showing consistent activity across differentiation stages (p<0.05). The study revealed depot-specific differences, with visceral adipocytes undergoing additional extracellular matrix remodeling absent in subcutaneous differentiation. Spatial analysis further showed that IGF signaling was particularly active in perivascular niches, while FGF activity dominated in mature adipocyte zones. These results provide the first comprehensive map of human adipocyte development, highlighting IGF and FGF pathways as potential therapeutic targets. The identified signaling networks offer new insights for developing interventions to promote healthy adipose expansion or inhibit pathological fat accumulation. This work advances our fundamental understanding of adipose tissue biology while providing clinically relevant data for metabolic disorder treatments. 

---
# Cross-Platform Chinese Offensive Comment Detection via Dual-Threshold Hard Example Mining 

**Authors**: Ruixing Ren, Junhui Zhao, Fangfang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27629)  

**Abstract**: Cross-platform deployment of offensive comment detection for Chinese social media suffers performance degradation. The paper proposes a dual-threshold hard mining method to address this. First, the clean-Chinese-base RoBERTa is finetuned on COLD to establish a binary baseline for fair comparison. Second, a three-class fine-labeled test set covering Weibo, Xiaohongshu, Tieba, and Zhihu is constructed, domain distances from the source are quantified using Jaccard and Proxy-A Distance, as well as the degradation bottleneck of the baseline under domain shift is systematically revealed. Herein, a dual threshold hard example mining strategy is proposed. High- and low-confidence error-prone samples are filtered from unlabeled corpora by prediction confidence. The model is secondarily finetuned under implicit contexts with merely a small set of manually labeled hard examples, realizing low-cost cross-platform domain adaptation. Experiments reveal significant performance gains of the optimized model across four platforms. 

---
# HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models 

**Authors**: Artem Ploujnikov, Francesco Verdini, Samir Sadok, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.27627)  

**Abstract**: Discrete audio representations have become increasingly popular for building multimodal text-audio systems and integrating audio capabilities into Large Language Models (LLMs). However, numerous studies report performance degradation on various downstream tasks due to information loss during discretization. To address this, we propose a novel approach combining temporally compressed discrete tokens with dimensionality-reduced continuous residuals. Our framework consists of a hybridized discrete-continuous focal modulation codec and a hybrid Transformer. This architecture performs autoregressive inference in the discrete domain, coupled with non-autoregressive prediction and continuous residual upsampling. Experimental results show that our approach significantly improves the retention of speaker characteristics compared to discrete-only methods, while simultaneously reducing the number of required autoregressive steps. 

---
# Global Explanations for Multivariate Time Series Forecasting Models via $K$-Order Markov Approximations 

**Authors**: Amadeo Tunyi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27599)  

**Abstract**: While many explainable AI (XAI) methods have been proposed, most are not designed for time-series forecasting models and often rely on the implicit assumption that timestamp features are independent. This assumption ignores the fundamental property of temporal dependence and can lead to explanations that violate the sequential and causal structure of the data. We introduce \textsc{KARMA}, a method for explaining time-series predictors by constructing a Markov surrogate model that captures the temporal dependencies learned by the predictor. Our approach revolves around three main aspects: identifying the minimal history length $K$ that is predictively sufficient for the model, estimating the best-fitting $K$-order Markov transition kernel from the discretized history space, and a five-level global explanation hierarchy that can be derived from the Markov transition kernel, which we illustrate using real-world weather data (Beijing PM 2.5). We also certify using complex synthetic data with known true causal edges that KARMA (i) recovers the data causal structure as learned by the model via a controlled experiment and (ii) identifies temporal dependencies better than established attribution methods such as TimeSHAP. 

---
# Narrative-UFET: Narrative Generation for Ultra-Fine Entity Typing 

**Authors**: Mreedul Gupta, Advait Deshmukh, Ashwin Umadi, Matt Pauk, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2606.27598)  

**Abstract**: Ultra-fine entity typing (UFET) assigns highly specific types to entity mentions, but current approaches struggle with types in the long tail. We hypothesize that a key limitation is the reliance on sentence-level context, since disambiguating evidence is often spread across multiple sentences. Testing this has been difficult because all existing UFET resources are sentence-level. We present Narrative-UFET, a controlled extension of UFET in which each entity mention is paired with an automatically generated short, coherent narrative. Synthesizing narratives lets us isolate the effect of specific discourse properties. We experiment with two paired variants: one in which the entity's type is held constant across the narrative (Maintain) and one in which it shifts (Change). We show that narrative context yields consistent improvements on long-tail types over sentence-level baselines, with the Change variant providing the stronger signal. A comparison against naturally occurring contexts shows that synthetic narratives yield stronger gains, indicating that controlled discourse construction can surface signals that real text leaves implicit. Substantial room for improvement remains, suggesting open directions in both discourse modeling and narrative construction. 

---
# Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding 

**Authors**: Liu Yu, Can Chen, Ping Kuang, Zhikun Feng, Fan Zhou, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2606.27596)  

**Abstract**: Large Vision-Language Models (LVLMs) exhibit sophisticated reasoning but remain susceptible to object hallucination. Deviating from the prevailing attention intensity assumption, we reveal a deeper dynamic structural misalignment: hallucination is triggered at decision-critical steps where specific attention heads, acting as risky mediators, decouple from visual evidence to lock onto language priors. This establishes a pathological shortcut that bypasses visual grounding. To dismantle this, we propose Fox (Faithfulness and Observational-flow via eXpression-rectification), a training-free inference-time framework. Fox diagnoses structural misalignment using a visual attention entropy probe to localize risky mediators unsupervisedly. We then execute a targeted causal intervention via numerical logit saturation to physically sever the shortcut path. Finally, a conflict-gated cooperative decoding strategy reconciles interventional faithfulness with observational fluency. Extensive experiments demonstrate that Fox achieves SOTA performance, outperforming SID by 29.1% while preserving linguistic richness. Code is available at this https URL. 

---
# CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidance 

**Authors**: Hana Kim, Minje Kim, Tae-Kyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.27584)  

**Abstract**: 3D scene inpainting is essential for reconstructing areas corrupted by occlusions or limited viewpoints. While recent methods leverage Gaussian Splatting (GS) for efficient 3D editing, they often depend on precise multi-view segmentation masks and are inherently constrained to object removal tasks. We propose CoIn, a novel framework that bridges 2D inpainting models and 3DGS through a multi-stage consistency pipeline. Our approach first generates initial inpainted images using a diffusion model, enabling the use of arbitrary-shaped masks and diverse tasks like object insertion. We then introduce Reference Adaptive GS with Feature Attention to reconstruct a coarse 3D scene by adaptively weighing towards a reference view (2D -> 3D). This 3D representation provides geometric guidance to the diffusion process via GS-based Reference Feature Warping, ensuring multi-view consistency (3D -> 2D). Finally, a Texture-Enhancing Discriminator refines the 3D scene to achieve high photometric realism (2D -> 3D). Experiments show that CoIn, effectively leveraging bidirectional information flow, achieves state-of-the-art performance and effectively handles both object removal and object insertion with flexible mask input. 

---
# SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction 

**Authors**: Sirui Chen, Shibo Zhao, Zhen Wu, Jiaman Li, Guanya Shi, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27581)  

**Abstract**: Current humanoid reinforcement-learning policies excel at free-space motions but struggle with contact-rich tasks, as pure kinematic tracking cannot resolve the physical ambiguities of interacting with objects and uneven terrain. To address this, we introduce SceneBot, a unified motion-tracking framework capable of handling freespace locomotion, terrain traversal, and whole-body manipulation. SceneBot conditions a single policy on both reference motions and per-link contact labels, explicitly defining expected environmental interactions. To overcome the lack of annotated interaction data, we propose a hindsight scene reconstruction approach that infers scene-interaction graphs from retargeted human motion. Trained on 7.5 hours of this reconstructed, contact-rich data, SceneBot successfully generalizes to unseen motions and environments. Our results demonstrate that SceneBot is the first general framework to seamlessly unify free-space and contact-rich behaviors executing complex, long-horizon tasks like carrying a box upstairs and establishing contact conditioning as a powerful interface for humanoid control. All code and data will be open-sourced. More demos and information are available at: this https URL 

---
# Retroactive Advantage Correction: Closed-Form V-Trace Bias Correction for Delay-Aware RLHF 

**Authors**: Arnav Raj  

**Link**: [PDF](https://arxiv.org/pdf/2606.27580)  

**Abstract**: Reinforcement learning from human feedback (RLHF) in production does not always have a synchronous reward signal. Code-execution verifiers, slow judge ensembles, and queued human review can return several gradient steps after the rollout that produced them, breaking the synchronous-reward assumption underlying standard PPO. We address this gap with Retroactive Advantage Correction (RAC): each pending slow completion is queued, aged through a non-negative kernel, and reinjected as a clipped residual into the next optimiser step's advantage. We prove that under an unbiased clipped importance ratio, the cumulative RAC correction is exactly unbiased when the effective delay kernel reinjects all of its mass, and carries a bias linear in the unreinjected fraction otherwise; at the no-delay identity kernel it reduces to V-trace. On a tabular Markov decision process (MDP) proof-of-concept, RAC reduces the closed-form policy bias by up to 47.9x at the two-slow-channel configuration, beating wait-for-slow at lower wall-clock cost. RAC integrates with PPO and GRPO through a two-line reward-manager patch. 

---
# Distribution-based deep multiple instance learning for tumor proportion scoring in NSCLC 

**Authors**: Krzysztof Pysz, Artur Bartczak, Jarosław Kwiecień, Piotr Krajewski, Witold Dyrka  

**Link**: [PDF](https://arxiv.org/pdf/2606.27579)  

**Abstract**: Accurate assessment of tumor proportion score (TPS) in non-small cell lung cancer (NSCLC) is critical for treatment planning and prognosis. Key challenges include the tedious manual work required to annotate each slide, combined with the limited number of experts certified for this task. Multiple instance learning (MIL) has proven to be an effective approach for predicting TPS scores at the slide level; however, existing methods struggle with non-expressive (zero class) images. Our approach involves two models: (1) an embedding-extraction and multiclass-classification network that captures the histopathological features of individual patches, and (2) a MIL model that aggregates these embeddings to predict zero-inflated beta (ZIBeta) parameters representing the overall TPS probability distribution for the entire slide. Using only slide-level TPS scores as labels, we demonstrate how this end-to-end framework can leverage a novel distribution-based architecture to improve prediction accuracy and explainability. ZIBeta modeling significantly outperforms baseline linear and ridge regression while capturing expected accuracy through distribution concentration. 

---
# PEBS: Per-rater Empirical-Bayes Shrinkage for RLHF Reward-Model Calibration 

**Authors**: Arnav Raj  

**Link**: [PDF](https://arxiv.org/pdf/2606.27578)  

**Abstract**: Reward models for Reinforcement Learning from Human Feedback (RLHF) pool preferences across thousands of annotators and fit one global affine calibrator, collapsing raters with systematically different rating-scale offsets and slopes into a single average-rater fit that does not match any individual annotator. PEBS is a per-rater empirical-Bayes shrinkage estimator: it fits per-rater affine calibrators on a held-out slice of each annotator's ratings and applies Morris-James-Stein empirical-Bayes shrinkage toward the population mean, in closed form and without retraining the reward model. On PRISM, PEBS reduces within-user held-out RMSE by 8.58% over the pooled population-slope baseline. The procedure replicates on PluriHarms harm ratings (Qwen-2.5 base, in-family) with a +9.66% RMSE reduction over the same population-slope baseline. PEBS is a closed-form post-hoc estimator for annotator-specific affine calibration in RLHF reward modeling; it leaves the reward base model unchanged and estimates only the rater-level map used at inference time for new ratings. 

---
# hia-gat: A Heterogeneous Interaction-Aware Graph Attention Network For Frame-Level Traffic Conflict Risk Prediction On Freeways 

**Authors**: Mahshid Malazizi, Seyedmehdi Khaleghian, Mina Sartipi, Toru Hirano, Yunfei Xu, Hoang H. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27577)  

**Abstract**: This paper formulates frame-level freeway risk assessment as a multi-agent scene graph-level binary classification problem, where each video or trajectory frame is labeled risky if any TTC- or PET-based conflict violates a specified severity threshold. We construct a relation-aware graph per frame with vehicles as nodes and two interaction types as edges: same-lane (longitudinal) and adjacent-lane (lateral), augmented with physics-informed edge features aligned to rear-end and lane-change conflict mechanisms. Building on a structured benchmarking suite of non-graph models and graph baselines, we propose HIA-GAT, a dual-stream heterogeneous graph attention network that processes longitudinal and lateral interactions through dedicated attention pathways and fuses them via a conflict-type-aware gating mechanism with event-level gate supervision derived from SSM conflict attribution. Experiments on the NGSIM I-80 and US-101 freeway datasets across nine TTC and PET threshold configurations show that HIA-GAT achieves the best average risk-ranking performance (AUC 0.835 on I-80 and 0.867 on US-101), with the largest gains on PET-only (lane-change) settings where relational structure is essential. Beyond accuracy, the learned gate provides interpretable per-vehicle attribution of dominant conflict type, supporting actionable, real-time freeway safety monitoring. We show that graph structure is critical for modeling lateral conflict risk, while longitudinal risk can often be captured by non-relational aggregation. 

---
# On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models 

**Authors**: Dewank Pant, Shruti Lohani, Avijit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.27567)  

**Abstract**: Prompt injection is the top security risk for LLM-integrated applications, yet every defense proposed so far has been broken. We prove this is not a coincidence: in shared-embedding architectures that lack enforced control-data separation, perfect prompt-injection prevention is mathematically impossible. We formalize prompted systems as Prompted Action Models whose outputs include control-authoritative actions: refusal decisions, tool authorization, policy routing, and memory writes. We define Semantic-Faithful Control (SFC), the property that such behavior depends only on the meaning of untrusted input, not on how it is encoded. We then prove SFC is unachievable within the shared pipeline, via three results: a provenance-recovery impossibility (shared representations make trusted and untrusted content statistically inseparable, bounded by total variation distance); control-path exposure (untrusted tokens enter control-relevant computation through the same attention value-aggregation that determines outputs); and a finite-coverage invariance gap (finite training cannot certify invariance over infinite semantic-equivalence classes). We ground each quantity in measurements on production tokenizers and models.
The result is structural, not a gap in current defenses. It mirrors the code-data confusion in Von Neumann machines that gives rise to buffer overflows, a vulnerability class that took decades of layered defenses (DEP, Write-XOR-Execute, ASLR, stack canaries, and ultimately memory-safe languages) to contain, because no single mechanism sufficed. The implication is the same: prompt injection cannot be eliminated by better in-pipeline classification or alignment alone. It requires architectural separation of instruction and data channels. We identify the root cause and the class of solution it demands. 

---
# Benchmarking Multi-Modal Graph-based Social Media Popularity Prediction 

**Authors**: Utkarsh Sahu, Zhisheng Qi, Li Zhu, Yizhao Yang, Jun Li, Ryan Rossi, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27539)  

**Abstract**: Social media popularity prediction aims to forecast the future reach or influence of online content from early-stage observations. Accurate prediction enables key downstream applications, such as advertising optimization and strategic content planning by users, creators, and platforms. Despite substantial progress, existing popularity prediction works often fail to jointly consider multimodal content and temporal social interaction signals. Moreover, the literature remains highly fragmented across datasets, modalities, observation windows, prediction targets, and evaluation protocols. This fragmentation prevents fair comparison and obscures a systematic understanding of how textual, visual, temporal, and interaction-based signals jointly shape popularity dynamics. To address these challenges, we introduce MMG-Pop, a Multi-modal Graph-based Popularity Prediction benchmark, which unifies datasets, modalities, temporal interaction signals, and representative baselines under a standardized evaluation protocol. Furthermore, we propose MMG-PopNet, a unified multi-modal graph-based network that jointly models the aforementioned multi-modal signals and graph-structured social interactions. Extensive experiments on MMG-Pop, comprising four datasets across Bluesky and Reddit platforms, demonstrate the superior performance of MMG-PopNet and yield new insights into cross-platform training generalization, multi-task prediction benefits, multi-modality contributions, and LLM prediction limitation. These findings establish a unified foundation for future research on social dynamics modeling and intervention under heterogeneous modalities and socially-aware agentic ecosystem paradigms. 

---
# The Context-Ready Transformer 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2606.27538)  

**Abstract**: We introduce the context-ready transformer, a new recurrent neural network architecture built from a D-layer transformer block that pre-contextualizes each token before it enters the block. During left-to-right generation, a correction network combines the previous position's block output -- a cached summary of past context -- with the current token embedding, so the tokenenters the block already contextualized rather than as a raw embedding. At sequential inference, the correction chain makes the architecture a recurrent neural network. For training, we unroll the correction process K times over the full sequence, processing all positions in parallel at each step. A pretrained transformer can also be converted to a context-ready model by adding a zero-initialized correction FFN and fine-tuning. We evaluate across widths, depths, block sizes, and two datasets, with all comparisons against standard transformers, variants, and ablations. A D=5 model beats a 12-layer transformer while generating 1.7x faster on an A100. With K=10, a single-layermodel (D=1) beats a 6-layer transformer with a 2.6x inference speedup, and sequential inference matches parallel K=10 to within 0.01 PPL. The architecture benefits most from wide representations and long contexts. On a pointer-chasing task, D=1 trained with BPTT solves all 10 composition levels, while standard transformers exhibit staircase-like depth dependence. 

---
# Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledge 

**Authors**: Thomas Shih-Chao Liang, Zhuoran Yu, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.27527)  

**Abstract**: Large Language Models (LLMs) possess broad conceptual knowledge acquired through large-scale text pretraining, yet their potential to supervise models in other modalities remains underexplored. In this work, we propose LaViD--Language-to-Visual Knowledge Distillation--a simple and effective framework for transferring high-level semantic knowledge from a language-only teacher to a vision-only student model. Instead of relying on paired multimodal data, LaViD elicits conceptual signals from an LLM by prompting it to generate multiple-choice questions (MCQs) that probe semantic distinctions between visual classes. Each class is mapped to a soft label distribution over these MCQs, forming a rich conceptual signature that guides the student through an auxiliary distillation loss. Notably, despite using a language-only teacher without access to image data, LaViD consistently outperforms recent methods like MaKD that distill from vision-language models across multiple fine-grained benchmarks. It also achieves competitive or superior performance compared to state-of-the-art visual distillation methods such as DKD and MLKD, with further gains when combined with logit standardization. On the Waterbirds dataset, LaViD substantially improves worst-group accuracy, demonstrating enhanced robustness to spurious correlations with distillation. Code is available at this https URL. 

---
# DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection 

**Authors**: Yujin Tang, Chenming Shang, Ruize Xu, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.27499)  

**Abstract**: Research on agent memory has matured rapidly, but almost entirely on the text side: few existing benchmarks ask, in an interactive environment, when an agent genuinely needs to remember what it saw rather than what it could write down. We introduce DMV-Bench (Code: this https URL), the first interactive benchmark for multimodal-agent visual memory. DMV-Bench is built on a controlled home-furnishing e-commerce catalogue of 1,000 product variants in which a text-leakage contract keeps the discriminative signal of each task in the pixels alone. Across a chain of autonomous shopping sessions, every visited product image carries a unique, pre-rendered incidental cue, and the agent is later asked to recall a particular cued product and navigate to its URL. Inspired by dual-coding theory, we propose DualMem, a memory architecture that maintains a visual and a verbal code in parallel. On DMV-Bench, DualMem outperforms a caption baseline and three recent multimodal agent-memory systems at every chain length J in {5, 10, 15, 50} on both Gemini 2.5 Flash and Qwen2.5-VL-7B, with the lead surviving controls for memory-bank size and encoding-position bias, and an asymmetric dual-coding regime in which vision carries the cue end-to-end while the verbal channel plays a smaller query-grounding role. 

---
# Speculative Refinement: A Hybrid Autoregressive Diffusion Decoding Strategy and Its Behavior Across Benchmarks 

**Authors**: Aditi Gupta, Neel Mishra, Kushagra Trivedi, Pawan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.27474)  

**Abstract**: How should we evaluate generation systems that combine autoregressive (AR) and diffusion decoding? We study this question through Speculative Refinement (SpecRef), a training-free hybrid method that warm-starts a masked diffusion language model from an AR draft using entropy-guided selective masking. Evaluating SpecRef across six benchmarks (HumanEval, MBPP, GSM8K, BBH, ARC-Challenge, HellaSwag) with three distinct evaluation protocols (execution-based pass@1, exact-match, log-likelihood scoring), we surface several findings relevant beyond our specific system: (1) code benchmarks conflate structural discovery with logical correctness: providing a syntactic scaffold lifts accuracy from near zero to over 20% without changing the model, indicating that much of the baseline failure is structural; (2) a refinement tension phenomenon where multi-stage correction degrades already-correct tokens, exposing benchmark saturation ceilings invisible to single-model evaluation; (3) log-likelihood and generative evaluation produce different model rankings for the same model pair, suggesting they measure different capabilities; (4) standard Python post-processing silently breaks code evaluation for non-AR generators. These observations apply to any multi-stage or non-autoregressive generation pipeline and point toward more diagnostic evaluation practices. 

---
# Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents 

**Authors**: Vedant Patel  

**Link**: [PDF](https://arxiv.org/pdf/2606.27472)  

**Abstract**: Large language model (LLM) agents operate over long, multi-session interactions in which facts change: a user moves, a price updates, a plan is revised. Acting correctly requires using the current value of a fact and discarding values that have been superseded. We isolate this ability on real conversational data and show that it is a distinct, unsolved failure. On the knowledge-update subset of LongMemEval, replacing an agent's full context with a bounded, self-maintained memory drops accuracy from 92% to 77% even on a frontier model (gpt-5.4), a gap that is statistically significant (paired McNemar p<0.005) and persists across model scale while full-context accuracy saturates near 92%. The bottleneck is therefore memory maintenance, not comprehension, and is not closed by a stronger model. We then ask whether this is merely an undersized memory, and find it is not: as the conversation grows 24x, accuracy falls further (from 68% to 28%), and granting the agent proportionally more memory yields no detectable recovery (28% to 28%, n=25). The failure scales with the length of the conversation, not the compression ratio. We release Supersede, an open reinforcement-learning environment (on the verifiers / prime-rl stack) that turns this measurement into a training signal: agents are rewarded for answering from the current value and penalized for stale ones. Finally, we close the loop and show the gap is trainable: GRPO fine-tuning a small open model (Qwen2.5-3B) on this environment nearly doubles its held-out supersession accuracy on real, unseen conversations (9.0% to 16.7%, a single run), along a monotonic checkpoint curve indicating the learned policy, not the harness, carries the gain. To our knowledge this is the first trainable environment whose reward targets temporal fact-currency, and the first evidence the supersession gap can be trained down, not only measured. 

---
# GRAFT: Biological Graph and Hypergraph Benchmarks for Linked Gene Expression and Phenotypic Trait Prediction in Arabidopsis thaliana 

**Authors**: Manuel Serna-Aguilera, Vanshika Jindal, Fiona L. Goggin, Jiamei Li, Aranyak Goswami, Alexander Bucksch, Suxing Liu, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27413)  

**Abstract**: Understanding which genes control which traits in an organism remains one of the central challenges in biology. Despite significant advances in data collection technology, our ability to map genes to traits is still limited. This genome-to-phenome (G2P) challenge spans several problem domains, including plant breeding, and requires methods capable of reasoning over high-dimensional, heterogeneous, and biologically structured data. Current datasets and data repositories, however, are not well-equipped for this task. Current studies do not link gene expression and trait data, and most focus on very specific traits, limiting the breadth of possible correlations. To address this gap, we present the novel Gene-Graph Regression for Arabidopsis Functional Traits (GRAFT) dataset, a curated multi-modal dataset linking gene expression profiles with phenotypic trait measurements in Arabidopsis thaliana, a model organism in plant biology. GRAFT supports tasks such as phenotype prediction and interpretable graph learning. In addition, we benchmark conventional regression and explanatory baselines, including a biologically-informed hypergraph baseline, to validate gene-trait associations. To the best of our knowledge, this is the first dataset to provide multimodal gene information and heterogeneous trait or phenotype data for the same Arabidopsis thaliana specimens. With GRAFT, we aim to foster research to accurately understand the relationship between genotypes and phenotypes using gene information, higher-order gene pairings, and trait data from multiple sources. 

---
# Not All Relations Rotate Alike: Transformation-Aware Decoupling for Viewpoint-Robust 3D Scene Graph Generation 

**Authors**: Jingjun Sun, Chaowei Wang, Zhirui Liu, Jiaxu Tian, Ming Yang, Yaoxing Wang, Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27412)  

**Abstract**: 3D Scene Graph Generation (3DSGG) represents 3D scenes as structured object-relation-object graphs, providing a compact relational abstraction for spatial understanding. In embodied intelligence settings, the same 3D scene may be observed by agents from viewpoints that differ by yaw rotations. However, current 3DSGG models often fail to produce relation predictions that follow the expected transformation behavior under such viewpoint shifts. This behavior reveals an empirical mismatch related to predicate-level transformation heterogeneity: directional predicates such as left, front, right, and behind should transform with the observation frame, whereas most contact, support, and semantic predicates such as standing on and attached to should remain stable. To reduce this mismatch, we propose Transformation-Aware Decoupling (TAD), a viewpoint-robust 3DSGG framework that decouples relation reasoning according to predicate transformation behavior and is supported by viewpoint-stable object representations. TAD decomposes relation reasoning into two parts: one learns cues that should stay stable across viewpoints, while the other learns directional cues that should change with the observation frame. The two parts are merged for standard multi-label predicate prediction. Transformation-specific descriptors and group-aware auxiliary supervision encourage the two branches to capture complementary relation cues. Extensive experiments on 3DSSG show that TAD achieves state-of-the-art robustness under yaw viewpoint changes without training-time rotation augmentation, while maintaining competitive performance under the standard benchmark. The project page is available at this https URL. 

---
# Compression-Driven Anomaly Detection in Brain MRI Using an Interpretable Quantum Autoencoder 

**Authors**: Santanu Ganguly, Xing Liang, Dimitrios Makris  

**Link**: [PDF](https://arxiv.org/pdf/2606.27411)  

**Abstract**: We study a quantum autoencoder (QAE) for compression-driven anomaly detection in brain MRI data. The approach leverages angle encoding to map image patches into quantum states, followed by a variational encoder-decoder architecture trained to discard information via auxiliary trash qubits. Anomaly scores reflect the degree to which inputs resist compression relative to normal data, with higher scores corresponding to deviations from the learned normal manifold. Evaluated on publicly available brain MRI DICOM datasets, the method achieves a slice-level ROC-AUC of approximately 0.95 and a patch-level ROC-AUC of approximately 0.813, outperforming classical autoencoder and PCA baselines. Analysis of the learned parameters reveals a pronounced encoder-decoder asymmetry, where effective anomaly detection arises from structured information compression within the encoder rather than increased parameter magnitude or decoder expressivity. This results in a controlled compression-reconstruction trade-off with a clear operating regime that supports principled threshold selection. Qualitative evaluation further shows that the QAE produces spatially localized anomaly heatmaps aligned with tumorous regions. The results, supported by promising baseline performances, demonstrate that quantum autoencoders provide an interpretable and controllable mechanism for anomaly detection based on incompressibility with respect to a learned latent representation. This work highlights the potential of quantum autoencoders as a principled tool for studying compression dynamics in quantum machine learning, with promising implications for decision support in medical imaging workflows. 

---
# Towards Evaluation of Implicit Software World Models in Coding LLMs 

**Authors**: Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2606.27406)  

**Abstract**: Software engineering, whether performed by humans or by AI agents, requires reasoning about how software behaves. We call the internal model that supports such reasoning the software world model, and view current code-execution benchmarks as covering one well-studied slice of it -- control flow. In this paper, we take a step toward a broader evaluation by shifting the observable axis to execution resources: alongside test outcome and exception class, we predict peak memory, wall-clock time, and ranked profiler outputs at method and line granularity. We use SWE-bench Verified as the source of data to hold the test close to real-world software engineering tasks. All tested models, frontier ones included, show modest performance and brittle behaviour, suggesting a notable lack of understanding of how software is executed, as opposed to how its source code is written. 

---
# Automated brain tumor detection in MRI images using CNN and ResNet architectures 

**Authors**: Annapurna V K, Asha N, K Paramesha, Shabana Sultana, Kirankumar Humse  

**Link**: [PDF](https://arxiv.org/pdf/2606.27405)  

**Abstract**: Deep learning has shown significant potential in medical image analysis, particularly for disease detection using MRI scans. Accurate and early diagnosis of brain tumors remains challenging due to the complexity of brain structures and reliance on manual interpretation. This work presents an automated deep learning-based approach for brain tumor detection from MRI images using Convolutional Neural Networks and Residual Networks. Transfer learning is applied with two pretrained architectures, ResNet18 and ResNet50, to classify MRI scans into tumor and non-tumor categories. Experiments are conducted on a dataset of 3,929 brain MRI images, evaluating the impact of model depth and fine-tuning strategies. The results show that ResNet18 achieves a higher accuracy of 97% compared to 96% for ResNet50, demonstrating better generalization on limited medical data. The proposed framework enables fast, accurate, and cost-effective brain tumor detection, supporting early diagnosis and clinical decision-making. 

---
# SidConArena: An Environment Evaluating Agents in Open-Ended,Positive-Sum Bargaining Game 

**Authors**: Yeqi Feng, Yuxin Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2606.27397)  

**Abstract**: Evaluating LLM agents requires dynamic environments that go beyond static reasoning and zero-sum games. Real-world economic interaction is often open-ended and mixed-motive: agents must negotiate, create positive-sum surplus, compete for scarce assets, and plan under delayed returns. We introduce SidConArena, a new benchmark framework for evaluating LLM agents in open-ended, positive-sum bargaining. SidConArena formalizes a multi-player economy as a finite-horizon partially observable stochastic game with three coupled phases: natural-language negotiation with binding trades, deterministic converter-based production, and sealed-bid auctions for long-term assets. The framework combines structured observations, phase-aware agent dispatching, a neural-symbolic action interface, and asynchronous execution, enabling free-form interaction while preserving rule-grounded evaluation. Across homogeneous and heterogeneous tournaments, stronger frontier models achieve higher economic outcomes, yet agents still misvalue resources, bargain passively, and remain limited in long-horizon investment planning. 

---
# Agentic Publication Protocol: An Attempt to Modernize Scientific Publication 

**Authors**: Sirui Lu, Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27386)  

**Abstract**: Scientific publication is still organized primarily around static manuscripts, even though much of scientific progress depends on tacit know-how: how to run code, reproduce figures, interpret edge cases, choose useful follow-up directions, and avoid failed paths. Large language model agents create an opportunity to publish not only knowledge, but also operational know-how in a form that future readers and researchers can directly use. This paper outlines the Agentic Publication Protocol (APP), a lightweight repository format for packaging a paper together with code, data, environment information, reproducibility instructions, and an agent-facing instruction file. APP treats a version-controlled repository as the publication object and uses \texttt{this http URL} and optional skills to define a paper agent that can explain the work, reproduce key results when possible, and support follow-up research. We describe the design principles and details of the protocol, as well as the agent skills useful for publishing papers under the protocol. We also describe development tools for evaluating and improving the protocol and associated agent skills. Finally, we provide a broader discussion of the future of scientific research in the agent era. 

---
# CalBrief: A Pilot Diagnostic Benchmark for Evidence-Calibrated Scientific Briefing with Large Language Models 

**Authors**: Yu Fu, Yongqi Kang, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27383)  

**Abstract**: Large language models (LLMs) are increasingly used as research assistants, yet it remains unclear whether they can calibrate research takeaways to the strength and scope of the supporting evidence. We study evidence-calibrated scientific briefing: given a bounded package of related papers, a system should generate package-level takeaways with evidence strength, scope boundaries, and missing-evidence caveats. We contribute a verified pilot benchmark of 16 heterogeneous scientific evidence packages and 96 human-verified takeaways, and we use CalBrief, an auditable role/gap/strength framework, as a diagnostic probe to locate where briefing breaks down. Under a fair-schema evaluation, structured organization improves role and gap reasoning, but an explicit strength-calibration policy is systematically over-conservative and falls below majority and direct-LLM baselines. To explain why, we run a controlled diagnostic across three closed-model backbones (GPT-4o, Claude Sonnet, Gemini Flash) that separates three potential causes of conservatism. Approximately 63% of the conservatism gap is attributable to expanding the label space from binary {moderate, weak} to four-way {moderate, weak, uncertain, insufficient_evidence} (p < 0.001 across all backbones); only 1% is attributable to gap/scope signal injection (not significant); the remaining 36% arises from the pipeline policy itself. We also find that 4-way predictions can be post-hoc collapsed back to binary and then match or exceed direct binary prompting, so the extra labels carry information that strict matching hides. Label-level strength judgment and auditable evidence organization are distinct abilities currently in tension, and should be evaluated separately for LLM research assistants. 

---
# OverFlowLight: Real-Time Gridlock Prevention and Traffic Signal Optimization for Urban Intersections 

**Authors**: Mingyuan Li, Boyang Huang, Tianqi Jiang, Chenpu Li, Chunyu Liu, Yang Li, Ruimin Li, Qiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27381)  

**Abstract**: Queue overflow, a severe consequence of urban traffic congestion, occurs when vehicle queues exceed intersection capacity, obstructing upstream traffic and triggering cascading gridlocks. Prevailing traffic signal control (TSC) algorithms, primarily optimized for throughput, often fail to address overflow during peak hours, exacerbating congestion and creating safety hazards. We propose OverFlowLight, a real-time framework designed to preemptively resolve overflow and enhance overall TSC performance. It first introduces a mechanism to accurately detect overflow in real-time by leveraging multi-modal sensing from cameras and radars. Upon detection, it dynamically generates and inserts dedicated overflow phases into the signal cycle to clear the blocking queues. This is orchestrated by a hybrid control design that combines rapid rule-based overflow intervention with controller back ends such as reinforcement learning (RL) for longer-horizon efficiency. We conducted extensive real-world deployments of OverFlowLight across 43 intersections in three major cities. The framework demonstrates seamless integration with existing RL-based TSC agents, highlighting its modularity and practical applicability. Empirical results show that OverFlowLight reduces overflow incidents by 60.4% and increases network throughput by 18.2% compared to deployed baselines. Furthermore, it substantially diminishes the need for manual intervention common with expert-tuned signal plans. This work presents the first practical, scalable, and data-driven framework for actively preventing traffic gridlock, offering a crucial component for building resilient and efficient urban transportation systems. Our demonstration videos, codes and datasets are available at the anonymous URL, this https URL. 

---
# Position: The Term "Machine Unlearning" Is Overused in LLMs 

**Authors**: Sangyeon Yoon, Yeachan Jun, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2606.27379)  

**Abstract**: Large language models increasingly face demands to "forget" training data, knowledge, or behaviors due to regulatory deletion obligations, copyright/licensing disputes, and safety or product-policy requirements. This position paper argues that machine unlearning is overused as a term in LLM research and should be reserved for dataset-defined deletion: removing the training influence of a precisely specified forget set such that the resulting model is approximately indistinguishable from retraining without that data. We contend that many tasks currently labeled "unlearning" (e.g., refusal for harmful requests, entity/knowledge removal, or targeted suppression) pursue different, often policy-dependent objectives and therefore require different terminology and baselines (e.g., alignment, suppression, editing, obfuscation). We further argue that this confusion is not cosmetic: because papers make different implicit guarantees under the same label, metrics and benchmarks are frequently reused outside their intended scope, rewarding surface-level non-disclosure (e.g., low ROUGE/forget accuracy) even when retraining-equivalence is not tested and derived capabilities remain. We conclude by calling for stricter terminology tied to explicit guarantees and reference models, and for evaluations that match the claimed objective. 

---
# DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers 

**Authors**: Avinash Maurya, M. Mustafa Rafique, Franck Cappello, Bogdan Nicolae  

**Link**: [PDF](https://arxiv.org/pdf/2601.16956)  

**Abstract**: The rapid growth of Large Transformer-based models, specifically Large Language Models (LLMs), now scaling to trillions of parameters, has necessitated training across thousands of GPUs using complex hybrid parallelism strategies (e.g., data, tensor, and pipeline parallelism). Checkpointing this massive, distributed state is critical for a wide range of use cases, such as resilience, suspend-resume, investigating undesirable training trajectories, and explaining model evolution. However, existing checkpointing solutions typically treat model state as opaque binary blobs, ignoring the ``3D heterogeneity'' of the underlying data structures--varying by memory location (GPU vs. Host), number of ``logical'' objects sharded and split across multiple files, data types (tensors vs. Python objects), and their serialization requirements. This results in significant runtime overheads due to blocking device-to-host transfers, data-oblivious serialization, and storage I/O contention. In this paper, we introduce DataStates-LLM, a novel checkpointing architecture that leverages State Providers to decouple state abstraction from data movement. DataStates-LLM exploits the immutability of model parameters during the forward and backward passes to perform ``lazy'', non-blocking asynchronous snapshots. By introducing State Providers, we efficiently coalesce fragmented, heterogeneous shards and overlap the serialization of metadata with bulk tensor I/O. We evaluate DataStates-LLM on models up to 70B parameters on 256 A100-40GB GPUs. Our results demonstrate that DataStates-LLM achieves up to 4$\times$ higher checkpointing throughput and reduces end-to-end training time by up to 2.2$\times$ compared to state-of-the-art solutions, effectively mitigating the serialization and heterogeneity bottlenecks in extreme-scale LLM training. 

---
