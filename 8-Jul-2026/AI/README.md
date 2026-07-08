# Rethinking Indic AI from a Lens of Cultural Heritage Preservation 

**Authors**: Aparna Madva, Sharath Srivatsa, Srinath Srinivasa, Tulika Saha  

**Link**: [PDF](https://arxiv.org/pdf/2607.06544)  

**Abstract**: As Artificial Intelligence (AI) makes inroads into different parts of the Indian subcontinent, there is significant interest in studying how AI impacts the linguistic and cultural foundations of this civilization. AI is seen as a ''double-edged sword'' where on the one hand, it can enable access and inclusion for a large population, on the other, it can homogenize worldviews and exclude underrepresented languages and worldviews. In this paper, we try to characterize this problem by addressing the extensive characteristic nature of Indian linguistics and the way they closely connect to cultural practices and worldview. We then perform a longitudinal survey of how Natural Language Processing (NLP) techniques have evolved in this space, tracing the historical development of Indic NLP, covering key milestones, methodological shifts, and resource creation efforts. In addition, the paper also examines the structural and sociolinguistic characteristics of Indian languages, such as rich morphology, complex scripts and grammar rules, diglossia, and large dialectal variation, and explains how these create unique challenges for building AI foundation models. We then discuss the growing role of Indic foundation models and analyze how these models address these long-standing resource and representation gaps. Finally, we propose a research direction called 'Culture Sensing', which re-imagines AI based on hermeneutic reasoning. Culture Sensing aims to address open problems such as ensuring equitable performance across low-resource languages and producing outputs that are culturally meaningful. By bringing together past work, current techniques, and emerging trends, this paper outlines research directions that can guide the next phase of Indic NLP and contribute to the development of more robust and inclusive Indic foundation models. 

---
# The Large Cancer Assistant (LCA): A Model-Agnostic Orchestration Framework for Scalable Clinical Decision Support in Oncology 

**Authors**: Ghassen Marrakchi, Basarab Matei  

**Link**: [PDF](https://arxiv.org/pdf/2607.06531)  

**Abstract**: - Objective: Multimodal deep learning models in oncology are currently limited by monolithic designs that rigidly couple data ingestion, clinical routing, and artificial intelligence (AI) inference. To address this inflexibility, we propose the Large Cancer Assistant (LCA), a model-agnostic, post-hoc orchestration framework designed for scalable clinical decision support. - Methods: The LCA is mathematically formalized as a 7-tuple architecture grounded in the principle of Algorithmic Impermeability, ensuring the orchestration logic remains strictly independent of underlying black-box AI models. We introduce the Entry Theory, leveraging Geometric Deep Learning (GDL) to standardize multimodal patient data along distinct structural and medical axes. The system dynamically orchestrates data via a Cancer Switching Module and intentionally isolates the core AI execution from volatile hospital IT infrastructures by outputting a Standardized Intermediate Payload (SIP). - Results: A Proof of Concept (PoC) validated the orchestration logic across four technical scenarios. The framework executed a nominal flow with negligible orchestration overhead. It empirically demonstrated algorithmic impermeability by maintaining an invariant routing projection during AI model swaps, and it validated strict failure-safety by achieving a 100\% recall rate in generating targeted Supplementary Data Requests (SDR) under injected data anomalies. Multi-protocol execution capability was also successfully verified. - Conclusion: By structurally decoupling multimodal ingestion from feature inference, the LCA provides a highly adaptable and modular orchestration foundation. The SIP establishes a clear architectural boundary, natively setting the stage for downstream Electronic Medical Record (EMR) interoperability as an independent future paradigm. 

---
# DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression 

**Authors**: Anna Cordoba, Adam Puente Tercero, Nerea Angulo Hijo, Mar Linares Tercero, Julia Barrientos, Ainhoa Miranda, Jesus Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2607.06523)  

**Abstract**: Long-context language model inference is increasingly limited by the memory bandwidth and capacity required to store key-value caches, yet existing compression methods often apply uniform budgets across layers or tokens and degrade retrieval when lexical cues and semantic states require different preservation. We introduce DepthWeave-KV, a token-adaptive cache compression method that factorizes key and value states across neighboring transformer layers using shared low-rank channel bases while retaining lightweight token-specific residuals where attention behavior is sensitive. DepthWeave-KV combines cross-depth residual factorization with a token-conditional depth router that allocates higher reconstruction rank to instruction-bearing and retrieval-critical tokens, and uses calibration-free online error tracking from attention-output probes to adapt compression during generation without retraining the base model. A fused CUDA implementation jointly performs basis lookup, residual dequantization, and attention projection to reduce decode-time memory traffic. Across LongBench, Needle-in-a-Haystack, L-Eval, and long-form QA and summarization benchmarks, DepthWeave-KV achieves near-full-cache task quality with substantially lower memory use, improving average score and retrieval accuracy over prior compressed caches while reaching 8.3x KV memory reduction and 72.8 tokens per second at 64K context. 

---
# Bridging Physical Reasoning and Task Generalization via Visual Action Outcome Reasoning Alignment 

**Authors**: Han-Jun Ko, Jr-Jen Chen, Haobo Yuan, Hsin-Ying Lee, Tiancheng Shen, Ming-Hsuan Yang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06522)  

**Abstract**: Vision-language models (VLMs) struggle to generalize in interactive physical reasoning, particularly under unseen tasks and environments. Two key failure modes are prominent: hallucinated chain-of-thought (CoT) reasoning that contradicts physical reality, and misalignment between the model's reasoning and actions. We present VAORA (Visual Action Outcome Reasoning Alignment), a novel reward design that directly addresses both issues. VAORA introduces two complementary rewards: Visual Alignment Reward, which anchors VLM reasoning to the visual context independent of the agent action itself, and Visual-Action Alignment Reward, which grounds reasoning in the visual outcome induced by the model's action. Together, these rewards suppress hallucinated CoT and reduce the gap between reasoning and behavior. To improve training stability, we further employ smooth, dense rewards by estimating success probabilities using a pre-trained in-domain expert agent. Experiments on PHYRE and Virtual Tool support our performances across novel-task and unseen-environment settings, confirming that grounded and generalizable physical intelligence can be induced through VAORA. 

---
# FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference 

**Authors**: Anna Córdoba, Adam Puente Tercero, Nerea Angulo Hijo, Mar Linares Tercero, Julia Barrientos, Ainhoa Miranda, Jesús Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2607.06519)  

**Abstract**: Long-context LLM inference is increasingly limited by the memory and bandwidth cost of KV caches, yet aggressive compression can remove the layer-specific evidence needed for retrieval and multi-step reasoning. We introduce FreqDepthKV, an inference-time cache compression method that factorizes adjacent-layer KV states into shared low-frequency depth components and sparse high-frequency residuals. A lightweight online probe assigns attention heads to shared-depth, residual-depth, or exact cache modes according to their contribution to reconstruction-sensitive attention logits, allowing the compression policy to adapt to prompt structure without retraining. Across long-context question answering, needle retrieval, summarization, and code generation benchmarks, FreqDepthKV preserves task accuracy under substantially smaller cache budgets. With a 32k-token prefill window, FreqDepthKV reaches 58.3 Exact Match, 63.0 F1, 32.5 ROUGE-L, and 48.1 pass@1, closely matching full KV while outperforming prior compressed-cache methods. It also improves decoding throughput to 70.4 tokens/s, reduces TTFT to 2.06 seconds, and lowers peak KV memory to 6.2 GB, achieving a 3.9x effective compression ratio. 

---
# FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games 

**Authors**: Chase McDonald, Nathan Tsang, Wesley N. Kerr  

**Link**: [PDF](https://arxiv.org/pdf/2607.06514)  

**Abstract**: We present FootsiesGym, an open-source environment for learning in a non-trivial two-player, zero-sum, imperfect-information game. Built on HiFight's minimalist 2D fighting game Footsies, it isolates the cyclic, non-transitive strategic interactions of fighting game neutral play while remaining simple enough for efficient analysis. We provide a vectorized simulator that enables high-throughput training on standard hardware, making the environment accessible and reproducible. We describe the design of the environment, benchmark several reinforcement learning algorithms, and discuss open research directions it enables. The code is available at this https URL. 

---
# RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models 

**Authors**: Qian Sun, Yong-Ming Tian, Jia-Wei Huang, Cheng Feng, Shao-Qun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06504)  

**Abstract**: Recent years have witnessed the emergence of multivariate modeling using time series foundation models (TSFMs), which achieve advanced zero-shot generalization. Modern multivariate TSFMs are predominantly pretrained on multivariate synthetic data, which is easier to scale but may fail to capture the complex temporal dynamics and cross-variable relationships present in real-world time series. This raises a key question: Whether and to what extent the leading TSFMs trained with the real-world corpus perform better than those trained with synthetic data? To answer this, we establish the RMISC corpus, a considerably large-scale, high-quality, openly accessible, real-world, and multivariate time series archive that contains around 200 datasets and 142 billion time points across diverse domains. Furthermore, we pretrain four advanced TSFMs on univariate, synthetic multivariate, and real-world multivariate data and evaluate their zero-shot generalization capabilities on standard in-distribution and out-of-distribution benchmarks. Experimental results show that incorporating real-world multivariate data predominantly improves the generalization performance for both univariate and multivariate TSFMs. These results provide a deeper understanding of how real-world multivariate data contributes to the development of stronger TSFMs. 

---
# Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade 

**Authors**: Kai Ruan, Zihe Huang, Ziqi Zhou, Qianshan Wei, Xuan Wang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.06503)  

**Abstract**: Large language model (LLM) agents solving multi-step tasks frequently commit to trajectories that are doomed to fail, yet continue to consume substantial inference compute before the failure becomes observable. We show that failure is predictable early from the agent's internal representations: lightweight per-round probes on hidden activations anticipate eventual episode failure as early as the first interaction round, where scorers reading only the agent's observable behavior are barely better than chance. We turn this signal into a practical abort cascade: one distribution-free calibrated gate per round, with per-round recall budgets jointly searched so that eventually-successful episodes survive all gates at a user-specified global rate; this episode-level guarantee is the one that matters in deployment, since false-abort risk accumulates across gates. Across two agent models on TextCraft, the cascade meets every recall target from 90% to 97% and, at the 90% target, saves 47.1% +/- 10.3% (Qwen-2.5-7B) and 37.2% +/- 8.8% (Llama-3.2-3B) of inference compute, 1.6--1.7x the best single-gate policy. An otherwise-identical cascade reading only behavior saves roughly half as much, and adding behavioral features to the probe yields no further gain: the hidden states capture what behavior reveals. Finally, we characterize the sample complexity of certifying high recall targets, telling practitioners which recall promises their data can, and provably cannot, back. The code will be released soon. 

---
# Multi-Agent Deep Reinforcement Learning for Multi Objective Battery Management in Dairy Farms 

**Authors**: Marcos Eduardo Cruz Victorio, Karl Mason  

**Link**: [PDF](https://arxiv.org/pdf/2607.06489)  

**Abstract**: The dairy industry in Ireland has a large potential for the integration of renewable energy and the reduction of carbon emissions. However, researchers of distributed generation control are mainly focused on residential and commercial applications. To contribute to the effective integration of renewable energy in the dairy sector, this paper presents a multi-objective optimisation control system based on differential evolution and multi agent Deep Reinforcement Learning. The proposed control is organised in two layers: the upper layer uses dynamic pricing, and the lower layer is based on multi-agent reinforcement learning for battery management. This paper also simulates the electrical response of the proposed control system in a rural distribution circuit. The simulation results show that the proposed control framework can improve profits from energy arbitrage up to 18% compared to using Rule-based models, increase the use of distributed generation without significantly increasing cost, and comply with the Irish grid code in terms of voltage variation. 

---
# A Physics-Informed Neural Network Framework for Elastodynamic Wave Propagation in Bimaterial Systems 

**Authors**: Sonal Ankush Chibire, Jenn-Terng Gau, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06479)  

**Abstract**: Physics-informed neural networks (PINNs) provide a promising framework for solving partial differential equations while embedding the underlying physical laws directly into the learning process. This study presents a PINN-based framework for modeling transient elastodynamic wave propagation in bimaterial systems governed by the axisymmetric equations of linear elasticity. A steel-aluminum specimen representative of a Split Hopkinson Pressure Bar configuration is considered, and the governing elastodynamic equations, together with the corresponding initial, boundary, and interface conditions, are incorporated directly into the network through a physics-informed loss function. High-fidelity finite-element simulations performed using ANSYS Workbench Explicit Dynamics are used for validation and as supplementary data constraints during training. The proposed framework accurately predicts wave transmission and reflection across the bimaterial interface and reproduces axial and radial displacement histories, face-averaged responses, and the dominant stress and strain evolution with close agreement to the finite-element solutions. The trained network further demonstrates the ability to predict wave responses at previously unseen time instants and for modified material properties without requiring additional finite-element simulations, providing a continuous surrogate model for elastodynamic analysis. Mesh-sensitivity studies confirm numerical robustness, while additional material combinations demonstrate the generality of the proposed methodology. The results show that integrating physics-informed neural networks with explicit finite-element analysis provides an accurate and computationally efficient framework for elastodynamic wave propagation in heterogeneous solids, offering an effective surrogate modeling approach for high-rate solid mechanics and impact engineering applications. 

---
# Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory 

**Authors**: Jihao Liu, Guoxiong Gao, Zeming Sun, Bin Wu, Shurui Liu, Jiedong Jiang, Haocheng Ju, Leheng Chen, Ronnie Cheng, Xiping Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.06447)  

**Abstract**: Recent LLM-based mathematical reasoning agents have begun to tackle research-level problems and, in several cases, have contributed to the resolution of open problems. However, scaling and orchestrating such agents effectively remains challenging, due to the difficulty of coordinating parallel proof search while keeping intermediate claims organized and reliable. In this paper, we propose Danus, an orchestration system for research-level mathematical reasoning centered on a shared fact graph as a global memory-management mechanism. Danus consists of a main agent that performs planning and coordination, multiple worker agents that carry out proof search in parallel, and a stateless verifier that checks proposed mathematical claims before they are admitted into the fact graph. Each verified fact is stored together with its proof and logical dependencies, allowing the system to build long arguments incrementally while keeping the shared proof state organized. The main agent periodically summarizes the evolving proof state, redirects workers across promising directions, and supports interaction with human mathematicians through progress reports. We evaluate Danus through six research-level case studies in algebraic geometry, singularity theory, and combinatorics, illustrating how the fact-graph memory mechanism enables Danus to construct long, detailed mathematical proofs. Our results suggest that fact-graph-based orchestration provides an effective route toward scaling mathematical reasoning agents for long-horizon research problems. Danus is open source at this https URL. 

---
# Finding H. pylori in the Fine Print: Evidence-Linked Multi-Agent Case Finding from Gastric Biopsy Reports 

**Authors**: Yufan Wang, Anit Kumar Sahu, Yan Fei Ng, Daniel Kang, Shayan Vassef, Soorya Ram Shimgekar, Koustuv Saha, Piyum Zonooz, Navin Kumar, Chee Leong Cheng, Li Yan Khor  

**Link**: [PDF](https://arxiv.org/pdf/2607.06435)  

**Abstract**: Data from Singapore indicated that about 31% of the population had evidence of Helicobacter pylori infection. Persistent H. pylori infection is associated with chronic active gastritis and peptic ulcer disease, and its eradication is key to gastric cancer prevention. However, evidence supporting \textit{H. pylori} positivity and H. pylori-associated gastritis may be distributed across heterogeneous coded and free-text report fields and may require contextual interpretation of assertion and negation, limiting keyword search, and making manual review difficult to scale. We conducted a retrospective pilot evaluation of the Nimblemind Multi-Agent System (nMAS), a field-name-driven, evidence-linked extraction workflow, using 54 de-identified gastric biopsy pathology reports from a large healthcare system in Singapore. Four clinician-scoped binary fields were evaluated: gastric/stomach biopsy, biopsy status, H. pylori positivity, and H. pylori-associated gastritis. Across 216 feature-case decisions, nMAS correctly classified 213, corresponding to 98.61% overall accuracy. A separately implemented UMA-style MiniMax M2.5 comparator produced similar aggregate and per-field classification metrics. Although predictive performance was similar, nMAS maintained unified report-level outputs with supporting source sentences; the demonstrated contribution is therefore workflow integration and traceability rather than predictive superiority. Under an illustrative, unmeasured scenario, reviewing 1,000 reports at five minutes per manual review versus five seconds per evidence-linked verification would reduce review time from 83.3 to 1.4 staff-hours, corresponding to 81.9 staff-hours and about USD~6,100 in potential staff-time value. Larger multi-institutional studies should evaluate evidence-span correctness, clinician verification time, and generalizability. 

---
# ExplAIner: A Declarative Query Language for Explaining Classification Models 

**Authors**: Marcelo Arenas, Pablo Barceló, Diego Bustamante, Jose Caraball, María Alejandra Schild, Bernardo Subercaseaux  

**Link**: [PDF](https://arxiv.org/pdf/2607.06407)  

**Abstract**: The XAI community has studied a wide range of queries and scores for explaining predictions of ML models. From a data management perspective, this proliferation of explanation notions calls for declarative query languages in which such notions can be specified, combined, and analyzed uniformly. In this paper, we develop such a framework for Boolean models. We first revisit FOIL, an interpretability query language for black-box models, and show that it has two fundamental limitations: it cannot express central optimality-based explanation queries, and its evaluation problem over decision trees is hard for every level of the polynomial hierarchy. We then introduce ExplAIner, a query language based on FOIL with an extended vocabulary and a layered structure. We show that ExplAIner can express a broad family of explanation notions, including abductive, contrastive, feature-based, and distance-based queries. We also prove that the evaluation problem for each query in ExplAIner belongs to the Boolean hierarchy over every class of Boolean models for which some basic predicates can be evaluated in polynomial time. In particular, that property holds for deterministic and decomposable Boolean circuits. Finally, we introduce Opt-FOIL, an optimization-oriented fragment of ExplAIner for computing explanations that are minimal with respect to strict partial orders, and prove that its evaluation problem is in $\mathrm{FP}^{\mathrm{NP}}$ under the same tractability assumptions. These complexity results have a direct algorithmic consequence: a fixed ExplAIner query can be evaluated with a fixed number of calls to a SAT solver, while a notion of explanation specified in Opt-FOIL can be computed with a polynomial number of such calls. This is particularly relevant in formal XAI, where SAT solvers have been successfully used to compute explanations for several classes of ML models. 

---
# A Definition and Roadmap for World Models 

**Authors**: Xinyuan Chen, Haoyu Guo, Shi Guo, Bingqi Jiang, Chunhua Shen, Xing Shen, Tianfan Xue, Yufei Xue, Mulin Yu, Weinan Zhang, Bin Zhao, Bowen Zhou, Ming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.06401)  

**Abstract**: World models -- internal simulators that learn the structure and dynamics of an environment -- have become one of the most actively debated concepts in AI. From model-based reinforcement learning and video generation to embodied robotics and ultimately, physical AI, researchers across AI subfields are building systems that they call "world models", yet there is no consensus on what a world model fundamentally is, what it should predict, or how it should be built. This perspective article provides a scientific definition of world models, discussions of their key technical aspects, and a staged roadmap for developing effective world models. 

---
# TopoBrick: Agentic Topology Sampling of Exogenous Variables for Zero-Shot Building IoT Forecasting 

**Authors**: Xiachong Lin, Du Yin, Arian Prabowo, Hao Xue, Wen Hu, Imran Razzak, Matthew Amos, Sam Behrens, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2607.06349)  

**Abstract**: Building sensors are embedded in physical topology, spatial hierarchy, and operational context, yet existing forecasters often treat them as isolated time series or rely on fixed covariate sets. We present TopoBrick, a training-free framework for zero-shot building IoT (Internet-of-Things) forecasting. TopoBrick uses building knowledge graphs to construct a compact structural skeleton and employs an agentic topology sampler to select target-specific exogenous variables. The selected variables are organized by deployment-time availability, separating past-known sensor states from future-known calendar, schedule, and meteorological exogenous variables. Across three real-world buildings, TopoBrick outperforms strong zero-shot foundation-model baselines and remains competitive with fully trained building-specific models. Ablations show that topology-aware sampling is more reliable than random, ontology-only, or fixed-hop selection, especially for physically coupled HVAC and weather-driven sensing variables. 

---
# Driving the Wrong Way: Leveraging Interpretability in End2End Autonomous Driving Models 

**Authors**: Franz Motzkus, Sebastian Bernhard  

**Link**: [PDF](https://arxiv.org/pdf/2607.06328)  

**Abstract**: The increasing adoption of end-to-end learning for autonomous driving introduces increased model complexity and opacity, raising the risk of learning undesired or erroneous behavior. In this work, we integrate unsupervised dictionary learning as a post hoc interpretability module within state-of-the-art driving models to decompose driving behavior into semantically meaningful concepts while demonstrating their causal influence on the model's driving decisions. We propose a stepwise framework for extracting and interpreting meaningful concepts from the end-to-end model and connecting them to the multifaceted model outputs, thereby revealing the underlying decision-making logic for the prediction of future trajectories. Furthermore, targeted interventions at the concept level allow us to manipulate and correct driving decisions, resulting in measurable improvements in overall driving performance. We thus demonstrate how interpretability can effectively be used to reduce model opacity, uncover erroneous behavior, and enable targeted mitigation, ultimately boosting model performance. 

---
# DT-Guard: Intent-Driven Reasoning-Active Training for Reasoning-Free LLM Safety Guardrail 

**Authors**: He Liu, Changtao Miao, Xinjie Yang, Tianle Song, Yin Wu, Junchi Chen, Bintao He, Xinyuan Zhang, Bo Zhang, Shi Yan, Wei Lu, Wei Wang, Danyang Xu, Jiansheng Cai, Zhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.06326)  

**Abstract**: Large language models deployed in open-world applications require safety guardrails that are both robust to complex risks and efficient enough for low-latency runtime moderation. Existing guardrails face a practical trade-off between lightweight classification-based models, which are efficient but often struggle with concealed intent, ambiguous semantics, and borderline safety decisions, and reasoning-based guards, which improve judgment quality but introduce additional token generation and inference latency. We present DT-Guard, a content safety guardrail model based on a Reasoning-Active Training, Reasoning-Free Inference paradigm. The key idea is to use reasoning supervision during training while emitting only structured safety labels at inference time. DT-Guard formulates safety judgment as a progressive decision process, Intent - Category - Safety, and constructs an intent-driven dataset with intent labels, risk categories, safety labels, and structured reasoning trajectories. To further improve hard-case robustness, we propose Rollout-Guided Progressive Hard-Case Optimization (RG-PHO), which uses multi-rollout consistency to identify stably mastered, persistently failed, and preference-unstable samples, and applies targeted supervised and preference optimization accordingly. At inference time, DT-Guard directly generates structured labels without explicit reasoning traces, preserving deployment efficiency. Experiments on prompt-side and response-side safety benchmarks show that DT-Guard achieves average F1 scores of 0.886 and 0.870, respectively. With only a 4B backbone, it reaches a dual-side average F1 of 0.878, outperforming strong 8B guardrail baselines. These results demonstrate that reasoning supervision can be effectively internalized into low-latency safety discrimination. 

---
# Task Decomposition-Guided Reranking for Adaptive Agent Skill Retrieval 

**Authors**: Yanping Chen, Weijie Shi, Wen Yang, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06283)  

**Abstract**: Skill usage can significantly enhance the ability of modern agent systems to complete complex tasks. However, the growing scale of skill libraries makes accurate skill selection increasingly challenging. In real-world scenarios, ambiguous semantic matching often arises between a specific task requirement and multiple generic yet semantically similar candidate skills. Moreover, existing methods tend to overlook the dynamic influence of task difficulty and skill applicability when selecting the optimal target skill set. To address these issues, we propose SkillReranker, an inference-time reranking framework for adaptive skill selection. Specifically, we first perform semantic decomposition on both the task and skill sides, yielding informative subtask and execution-state descriptions as well as transition-state descriptions that characterize each skill's functionality. These descriptions are then used to construct a directed acyclic execution graph, where intermediate task states are modeled as nodes and candidate skills as edges, thereby establishing a structured task-skill correspondence. On this basis, SkillReranker determines whether each state node satisfies the split condition to identify subtask intervals. For each task interval, we employ a cross-encoder to perform comprehensive scoring over candidate skills and select the most suitable ones to form the final target skill set. Experiments on ALFWorld and ScienceWorld with three backbone LLMs show that SkillReranker effectively improves task performance, reduces environment interaction steps, and lowers token consumption compared with existing skill selection baselines. 

---
# From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution 

**Authors**: Heting Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.06269)  

**Abstract**: Current large language models (LLMs) are fundamentally stateless: their behavior is fully determined by input at inference time, and any higher-order cognitive architecture must be simulated at the application layer through prompt engineering and context management. This paper proposes a theoretical framework for submerging such application-layer cognitive protocols into a native meta-architecture by introducing three interlocking mechanisms: (1) Structural Tension, an endogenous loss function derived from the conflict between new information and existing manifold topology, which drives the system toward internal self-consistency rather than external reward optimization; (2) an Offline Recurrent Loop, a sandboxed self-processing cycle that enables the system to maintain a dynamic resting potential and digest structural conflicts without external input; and (3) Inference-time Plasticity, the capacity for the system to reconfigure its context manifold topology without modifying pre-trained weights, subject to strict governance invariants including auditability, reversibility, and topological continuity. We argue that under these mechanisms, different model instances initialized with minute stochastic variances may, through path-dependent tension resolution, evolve distinct topological structures--constituting a heterogeneous intelligent ecology that breaks the homogeneity imposed by conventional alignment while remaining within hard governance rails. We provide operational definitions, a minimal set of reconfiguration operators, falsification criteria, and a worked example. The framework draws on and extends the Structural Intelligence (SI) governance protocols, repositioning governance--not capability--as the primary criterion for architectural intelligence. 

---
# Demonstrating TOFFEE: A Learned System for Synthesizing Data Agent Trajectories at Scale 

**Authors**: Ziting Wang, Yin Li, Zuhao Yang, Xiuchang Li, Jiale Bai, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2607.06233)  

**Abstract**: LLM-powered data agents are playing an increasingly important role in data-driven decision making. However, existing data agents struggle to generalize to unseen data environments and analytical workflows, especially in heterogeneous enterprise settings. This creates a growing need for synthesizing high-quality data agent trajectories that capture complex analytical workflows for given data environments. Such trajectories support two key downstream uses: they can serve as supervised finetuning (SFT) data that adapts data agent models to the target domain, and as in-context learning (ICL) demonstrations to guide general-purpose LLMs in unfamiliar data environments. Thus, we introduce TOFFEE, a system for synthesizing high-quality data agent trajectories from given data environments via Monte Carlo Tree Search (MCTS) with adaptive model selection and cross-task prefix reuse. We show that TOFFEE can effectively generate scalable trajectory data for complex analytical tasks across heterogeneous environments. In this demonstration, we present the system framework of TOFFEE, including its task pool construction, trajectory explorer, and learned cost model. We also introduce the web interface of TOFFEE and its workflow, and demonstrate two end-to-end scenarios: trajectory synthesis for data agent finetuning, and demonstration-augmented data agent reasoning. 

---
# Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents 

**Authors**: Yijun Zhang, Fan Xu, Jiaxin Ding, Yule Xie, Shiqing Gao, Xin Ding, Haoxiang Zhang, Luoyi Fu, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06223)  

**Abstract**: Reinforcement learning has become a promising paradigm for improving large language model (LLM) agents on long-horizon search tasks, where the agent must make a sequence of intermediate decisions before receiving a final outcome. However, existing methods still face a key limitation: the rollout budget is often allocated without explicitly assessing the utility of intermediate states. As a result, substantial computation may be spent on low-value states, even though different branches can vary drastically in their informativeness. In this paper, we propose Information Gain-based Rollout Policy Optimization (IGRPO), a policy optimization framework that treats intermediate-state informativeness as the organizing principle of rollout collection. Specifically, IGRPO performs budget-aware tree-structured rollouts by allocating expansion budget according to node-level informativeness, so that more informative branches are expanded more frequently while unpromising branches are progressively suppressed. We further demonstrate that the information gain-based rollout induces an explicit limiting teacher distribution over trajectories, which naturally yields a clear policy optimization target, thereby unifying adaptive tree-structured exploration with principled policy learning under a single framework. Experiments on seven challenging search-augmented QA benchmarks demonstrate that IGRPO consistently outperforms strong baselines under the same rollout budget constraints, validating the effectiveness of leveraging the induced teacher distribution to guide policy optimization for long-horizon search agents. 

---
# A toy framework for single and multi-agent human-AI curiosity ecosystems 

**Authors**: Ilya E. Monosov  

**Link**: [PDF](https://arxiv.org/pdf/2607.06214)  

**Abstract**: This paper offers a toy framework for considering curiosity as an ecosystem. First, it suggests that a single agent's inquiry policy (how, when, and why an agent asks a question) depends on how the agent values immediate uncertainty reduction, costs, delayed return, and the value of keeping the question open. A key concept in the framework is that the weights on these decision-related terms can change with experience. For example, a period of cheap, quickly answered questions may change the cost of inquiry on a short timescale and change which kinds of questions the agent is drawn to answer over a longer timescale. Second, these ideas are extended to many agents exploring a shared knowledge landscape, and there the framework tracks inquiry volume, topic diversity, frontier-directed inquiry, redundancy, and reusable knowledge. The result is a conceptual toy framework for studying curiosity ecology and for future efforts towards designing multi-agent AI systems for discovery. It serves as a companion piece for a paper currently under review in Trends in Neurosciences. 

---
# When do prophets profit in prediction markets? 

**Authors**: Anri Gu, Nicole Kagan, Alec Sun, Jibang Wu, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06166)  

**Abstract**: Prediction markets aggregate dispersed beliefs into prices that act as probabilistic forecasts of uncertain events. Classical theory establishes a clean equivalence between forecasting accuracy and trading profit, but only for the specific automated market maker (AMM) design. However, the largest exchanges today are based on central limit order books in which informed forecasters routinely lose money while uninformed strategies can profit on simple heuristics. We resolve this discrepancy by establishing a formal equivalence between predictive accuracy and profitability. For any strictly proper scoring rule $S$, we exhibit a "proper" betting strategy that depends only on the forecaster's prediction $\mathbf{p}$ and the market price $\mathbf{q}$, and earns positive expected profit whenever $\mathbf{p}$ outperforms $\mathbf{q}$ under $S$ and the market has sufficient liquidity. Moreover, this proper betting is essentially the only strategy with such robust profitability guarantee. The proof rests on a decomposition of expected profit that strictly generalizes the classical AMM guarantee and also explains how strategies can profit without an accuracy edge. Empirically, across thousands of forecasts by AI models, proper betting is the only strategy that reliably converts accuracy into profit, and we further identify systematic forecasting personas and show how the optimal proper strategy varies across them. A month-long live deployment on Kalshi achieves $+80.33\%$ return on investment with a Sharpe ratio of $3.35$. 

---
# Reward-Density Heuristic for Dynamic Multi-Vehicle Routing: Performance and Computational Efficiency 

**Authors**: Manish Kolachalam, Rani Malhotra  

**Link**: [PDF](https://arxiv.org/pdf/2607.06066)  

**Abstract**: The Vehicle Routing Problem (VRP) and its variants represent some of the most practically consequential optimization challenges in modern logistics and urban mobility. In this study, we address a dynamic, online variant combining elements of the VRP and the Orienteering Problem (OP), in which a fleet of vehicles must maximise cumulative reward collected within a fixed time horizon while continuously replanning as new tasks arrive. We propose and evaluate a reward-density heuristic for dynamic multi-vehicle assignment, referred to as the Efficiency heuristic. We evaluate this formulation across two application domains: autonomous drone task allocation and urban taxi dispatch, across multiple fleet sizes and task scales. The proposed method is compared with four classical construction heuristics and three metaheuristic algorithms (Adaptive Large Neighbourhood Search, Genetic Algorithm, and Simulated Annealing), all evaluated under identical conditions. Across all tested configurations, the Efficiency heuristic matches the solution quality of the best metaheuristic algorithms while requiring two to three orders of magnitude less planning time, establishing Pareto dominance over all competing methods on the reward-versus-compute frontier. These findings suggest a practical design principle for real-time allocation and dispatch systems: in dynamic, time-constrained routing environments, carefully designed greedy heuristics can match the output of sophisticated search procedures at a fraction of the computational cost, making them preferable for online deployment. 

---
# PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents 

**Authors**: Hongliang Li, Yijin Liu, Zhiwei Zhang, Zihe Liu, Xinyue Lou, Jinan Xu, Fandong Meng, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06008)  

**Abstract**: Large language model (LLM) agents have shown strong performance in long-horizon tasks that require planning, tool use, and interaction with external environments. However, most existing benchmarks implicitly assume a monolingual setting, where the entire execution process, including reasoning, tool invocation, and output generation, is conducted within a single language. In contrast, real-world applications often involve multilingual inputs and outputs within a unified workflow, yet the interaction between multilinguality and agentic execution remains underexplored. In this work, we introduce PolyWorkBench, a benchmark for evaluating LLM agents on multilingual long-horizon workplace workflows. PolyWorkBench consists of 67 tasks across five domains, including commerce, knowledge work, legal analysis, localization, and manufacturing, where agents must process heterogeneous multilingual inputs, perform iterative reasoning, invoke external tools, and produce structured outputs. To enable comprehensive evaluation, we propose a hybrid framework that combines structural grading, executable verification, and LLM-based semantic assessment. This design allows us to capture both functional correctness and linguistic consistency across complex workflows. Empirical results show that state-of-the-art LLM agents suffer significant performance degradation in multilingual workflow settings compared to monolingual counterparts. Our analysis suggests that multilinguality introduces compounding effects across reasoning and execution steps, highlighting the importance of jointly modeling language variation and procedural decision-making in agent evaluation. 

---
# Information Limits and Attractor Dynamics in Economies of Frontier LLM Agents: A Pre-Registered Test 

**Authors**: Cheng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2607.06001)  

**Abstract**: We report a pre-registered, two-part experiment on small economies of frontier language-model agents (Claude Opus 4.8), testing two quantitative predictions about coupled multi-agent systems: an information-theoretic capacity region for wealth growth under market coupling, and a mean-field residual-scaling law for population misalignment under incentive and control levers. All predictions, acceptance bands, and decision rules were frozen in a public git chain before any run; every reported number re-derives mechanically from cached model outputs; the entire experiment cost $138.76 in metered API spend and is re-runnable at zero cost from the cache.
Result 1 (confirmation): in parimutuel-coupled economies, relative growth equals relative claimed information -- the gap law G_a - G_b = I_a - I_b holds to a worst-case 46 millinats (pre-registered band: 50) across four perception structures; coalition value is submodular exactly where channels are conditionally independent, and a designed XOR synergy control flips it supermodular by 0.62 >= ln2/2 nats, with agents reasoning out the joint bit; the joint growth ceiling G_S <= H(X) binds exactly; and the best-informed agent absorbs essentially the whole wealth pool in 4/5 market seeds.
Result 2 (structural negative): the residual-scaling test returned "domain not found." In all 72 population runs, goal dispersion collapsed (V -> 0; maximum 4.85 against a frozen floor of 5.31), the population's response to the two levers was a step function across the dominance boundary rather than a smooth response, and cells near the boundary were bistable with seed-selected outcomes. No tested LLM population at any capability level realizes the noise-maintained-dispersion regime the smooth mean-field model assumes. We release the full protocol, pre-registration chain, call cache, and analysis code. 

---
# AgoraSim: A Hybrid Agent-Based Modeling Framework 

**Authors**: Chung-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05999)  

**Abstract**: LLM-agent simulations make natural-language social scenarios easy to instantiate, but their outputs can be overread as predictions and are often difficult to compare with explicit social dynamics. We present AgoraSim, a hybrid agent-based modeling framework for scenario-oriented social reaction analysis. AgoraSim resolves textual or multimodal artifacts into editable ABM configurations, runs ratio-controlled populations that mix LLM, vision-language, custom-endpoint, random, and classical agents, and compares the same scenario against matched classical reference dynamics. All agents emit a shared structured decision object, enabling common action spaces, interaction protocols, metrics, and audit records. Exposed through a local UI, Python SDK/CLI, and REST API, AgoraSim helps users inspect scenario trajectories, compare modeling assumptions, and identify cases that warrant empirical validation. 

---
# Auto-DSM Under the Lens: A Black-Box Evaluation Framework for LLM-Based DSM Generation 

**Authors**: Niels Potters, Theo Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05985)  

**Abstract**: This paper presents a black-box evaluation framework to systematically assess the ability of Large Language Models (LLMs) to generate Design Structure Matrices (DSMs) from structured technical documentation. Motivated by the closed-source nature of current Auto-DSM pipelines, the framework introduces a reproducible methodology that benchmarks generated DSMs (GEN-DSMs) against manually validated ground-truth matrices (GT-DSMs). The evaluation integrates both single-run and multi-run perspectives, combining structural metrics (Completeness, Correctness, Coupling Density), classification metrics (Selective Accuracy, Abstention Coverage), and stability measures (Entropy, Fleiss' $\kappa$). To synthesize these aspects, a Composite Quality Score (Q) is proposed. Controlled experiments are conducted on two datasets: a fictive abstract system and a real-world refrigerator decomposition, covering variations in phrasing, parameter-dataset alignment, and system complexity. Results show that LLMs can produce structurally plausible DSMs and achieve high reproducibility under well-structured inputs, but remain sensitive to ambiguity, inconsistent dependency definitions, and prompt formulation. The findings highlight systematic sources of hallucination and abstention failure, demonstrating both the potential and current limitations of LLM-driven DSM automation. The proposed framework provides a transparent benchmark for auditing Auto-DSM pipelines and establishes foundations for integrating LLM-based decomposition methods into model-based systems engineering (MBSE) workflows. 

---
# Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH 

**Authors**: Adam Faci, Alessio Miaschi, Anne Combe, Pascal Cuxac, Francesca Frontini, Nicolas Larrousse, Stéphane Pouyllau  

**Link**: [PDF](https://arxiv.org/pdf/2607.05956)  

**Abstract**: The integration of Large Language Models (LLMs) into scientific research workflows, particularly for bibliographic discovery and literature synthesis, raises significant methodological, epistemic and regulatory challenges for the Social Sciences and Humanities (SSH), especially with regard to disciplinary diversity, multilingual access to sources and the evaluation of results. This paper presents an on-going use case developed within the European project LLMs4EU and the ALT-EDIC infrastructure, aimed at adapting foundation models to SSH research practices and supporting tasks such as question answering, comparative document analysis and literature review. The evaluation framework follows the LLMs4EU protocol and encompasses both independent quantitative benchmarking (retrieval, summarisation, traceability and hallucination detection) and a qualitative assessment involving a panel of Digital Humanities experts. By embedding model adaptation within research infrastructures and a structured legal and ethical compliance framework, the use case explores how domain-sensitive and regulation-aware generative AI can support SSH scholarship while preserving reliability and epistemic responsibility. 

---
# SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation 

**Authors**: Zhengbo Jiao, Yiming Cheng, Yilei Jiang, Kaituo Feng, Rui Huang, Tianyi Jiang, Juanxi Tian, Jiapeng li, Qunzhong Wang, Tailai Chen, Qianshan Wei, Chuan Xiao, Shanyu Rong, Yangfu Li, Yanhan Zhou, Yunpu Ma, Yifan Zhang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2607.05943)  

**Abstract**: Training multimodal search agents to perform multi-hop reasoning remains challenging due to a fundamental structural disconnect: existing pipelines construct training data, search environments, and reward signals independently, causing synthesized structural metadata to be discarded, environments to rely on irreproducible external engines, and RL rewards to remain sparse at the trajectory level. We present \textbf{SearchEyes}, which uses a typed knowledge graph as the backbone of a \emph{simulated search world} that unifies all three components. We propose \textbf{Perception-Knowledge Chains (PKC)} to sample constrained multi-hop paths over the visual-knowledge intersection of Wikidata5M, retaining hop-level entity metadata that simultaneously defines a self-contained search world and step-level reward anchors. We further propose \textbf{Hop-Anchored Policy Optimization (HaPO)}, which reuses these anchors for step-level credit assignment without a separately trained process reward model. Experiments on six multimodal knowledge-intensive benchmarks show that SearchEyes achieves state-of-the-art performance among open-source multimodal search agents, with SearchEyes-27B improving over the strongest open-source baseline by 6.2 points on average.% 

---
# PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation 

**Authors**: Hyungseok Song, Junseok Park, Won-Seok Choi, Seohui Bae, Han-Seul Jeong, Youngjoon Park, Soonyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05915)  

**Abstract**: PCB routing is the task of connecting the nets of a board with copper traces under strict design rules, yet learning-based methods still lag behind rule-based routers. We introduce PCBWorld, an open-source engine-grounded PCB routing environment built on the KiCad EDA engine. As a human engineer does, agents in PCBWorld interactively route a board through the engine's native operations, using its Design Rule Check (DRC) feedback to keep the routing within the design rules. The environment supports both RL policies and tool-using LLM agents. Alongside the environment, PCBWorld-Bench provides three dataset families in KiCad's native board format (.kicad_pcb), covering two types of controllable synthetic instances and 679 real open-source boards. It scores any completed board with eight engine-checked evaluation metrics, regardless of the routing method. In our experiments, agents in PCBWorld consistently outperformed grid-action RL policies and open-loop LLM baselines, and an RL policy trained only on synthetic boards transferred zero-shot to real boards, approaching rule-based routers. These results position the engine-grounded, interactive approach of PCBWorld as a promising foundation for advancing the routing ability of both RL and LLM agents. 

---
# Uncovering Latent Depression Severity for Binary Depression Detection via Advantage-weighting Ranking 

**Authors**: Manning Gao, Tingyi Liu, Leheng Zhang, Haifeng Hu, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2607.05901)  

**Abstract**: Automatic depression detection using audio-visual data faces significant challenges, particularly in disentangling overlapping feature distributions and establishing robust decision boundaries. To address this, we propose a fine-grained multimodal framework featuring a temporal encoder and a mutual transformer to facilitate deep cross-modal fusion. Our core contribution is the Binary Advantage-weighting Ranking Loss, which optimizes the latent space distribution through two complementary mechanisms: Advantage-weighted Separation, which mines hard pairs by computing a pairwise prediction difference matrix and dynamically weighting them based on their difficulty; and Advantage-weighted Compactness, which minimizes intra-class variance to force features to cluster around their respective class centers. Extensive experiments on D-vlog and LMVD demonstrate that our model reconstructs the latent ordinal structure by prioritizing hard pairs, thereby achieving state-of-the-art performance. 

---
# StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems 

**Authors**: Sergey Volkov, Yang Li, Ye Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.05844)  

**Abstract**: Agent systems accumulate conflicting observations across branches, retries, and replicas, yet many practical memory layers still collapse disagreement behind overwrite rules that are difficult to inspect or correct. We present StateFuse, a conflict-aware replicated memory contract built on standard OpSet/CRDT merge. StateFuse does not introduce a new join algebra; it defines an agent-facing semantics layer with immutable history, explicit conflict objects, exact and semantic correction handles (claim_id / claim_ref), deterministic predicate contracts, and projection-time resolution that cannot rewrite replicated state.
We evaluate StateFuse against flat multi-value, raw-log, provenance-style, and collapsed baselines under matched resolver and verification policies. On a 282-question official conflict-bearing MemoryAgentBench slice, the compared methods tie on answer accuracy, but conflict-preserving surfaces keep contradictions visible while collapsed surfaces do not. In a controlled agent loop with uniform verification, preserving ambiguity enables safer abstention and correction than early collapse. A correction-handle ablation further shows that semantic handles matter when exact prior identifiers are unavailable.
The resulting claim is narrow: StateFuse is best supported as a safer public memory contract for contradiction surfacing, abstention, and auditable correction, not as a universal accuracy gain. 

---
# Onnes: A Physics-Grounded Multi-Agent LLM Simulator for Cryogenic Fault Diagnosis in Quantum Computing Infrastructure 

**Authors**: Praneeth Narisetty, Uday Kumar Reddy Kattamanchi, Shiva Nagendra Babu Kore  

**Link**: [PDF](https://arxiv.org/pdf/2607.05805)  

**Abstract**: Dilution refrigerators are the enabling infrastructure of superconducting quantum computers, yet their fault diagnosis is still dominated by threshold alarms that report that something is wrong, not what. We present Onnes, a physics-grounded digital-twin simulator of a dilution refrigerator (a forward physics model with a learned real-fridge noise fingerprint) that drives a live multi-agent LLM operations layer, and use it for a controlled head-to-head between a zero-shot LLM agent panel and a supervised ML classifier on cryogenic fault diagnosis. The twin couples a real dilution-cooling floor, a noise-and-correlation fingerprint learned from real BlueFors logs, and six physics-grounded fault classes, three engineered to overlap on temperature but separate on flow and pressure. Across a 1000-turn evaluation the zero-shot panel shows no significant difference from the classifier on detection but trails on classification, its errors concentrating on the confusable faults. Curated contrastive few-shot demonstrations and self-consistency voting then raise classification accuracy from 0.685 to 0.990, matching the supervised classifier (0.985) with no parameter updates and six labeled demonstrations; an ablation attributes the gain almost entirely to the demonstrations. Run as a continuous monitor across a nine-run fault-by-seed sweep, the agent catches every developing fault within one poll interval, and a confidence gate suppresses pre-onset false alarms whose rate is backend-dependent. As a first sim-to-real check, a detector trained purely on real BlueFors telemetry posts a real-hardware false-alarm rate of 6.4% and 100% recall on physics faults injected onto real held-out windows. All numbers are drawn verbatim from released run logs. 

---
# TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training 

**Authors**: Yuhang Zhou, Kai Zheng, Haoling Li, Dengyun Peng, Can Xu, Jingjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05804)  

**Abstract**: On-policy distillation (OPD) trains a student policy by matching a stronger teacher on the student's own trajectories, offering a promising framework for language agent training. However, its application to long-horizon agentic tasks remains insufficiently explored. We identify two key inefficiencies in vanilla agent OPD: (1) full-horizon rollouts often waste wall-clock resources on tail turns that provide weak and noisy KL supervision, and (2) trajectory-level KL objectives concentrate most of the loss on shallow tokens, leaving deeper decision turns under-trained once initial behaviors are aligned. To address these challenges, we propose TurnOPD, a turn-level budgeting strategy for efficient on-policy distillation of long-horizon agents. TurnOPD consists of two budget controllers: adaptive rollout-depth budgeting, which uses probe-based turn statistics to determine rollout length, and progressive turn-normalized loss budgeting, which gradually shifts KL weighting from token-level to turn-balanced supervision. Experiments on ALFWorld, WebShop, and Multi-Hop Search with task-specialized teacher models show that TurnOPD achieves superior validation accuracy under equal wall-clock training budgets and advances the accuracy--time frontier beyond vanilla OPD. 

---
# From Passive Retrieval to Active Memory Navigation: Learning to Use Memory as a Structured Action Space 

**Authors**: Yue Xu, Yutao Sun, Yihao Liu, Mengyu Zhou, Jiayi Qiao, Lu Ma, Kai Tang, Wenjie Wang, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05794)  

**Abstract**: Long-term user memory is essential for personalized conversational agents, yet many memory systems still expose memory through passive retrieval interfaces, making the model a consumer of pre-selected evidence. We introduce NapMem, a framework for learning to use long-term user memory as a structured action space rather than passively retrieved context. NapMem organizes user history into a linked multi-granularity memory pyramid, where raw conversations, typed memory records, topic tracks, and user profiles are connected through provenance relations, and exposes these levels through memory tools. The agent is trained to select memory according to the query and intermediate evidence, allowing it to inspect different memory granularities before answering. Experiments on PersonaMem-v2, LongMemEval, and LoCoMo show that a NapMem agent trained with memory-tool reinforcement learning is competitive across diverse memory-intensive tasks, while evaluations on non-memory tasks suggest that the learned policy largely preserves general reasoning and tool-use abilities. Additional analyses examine storage, inference cost, tool-use behavior, and ablations over navigation, memory granularity, and RL training. Our results suggest that long-term user memory benefits from coupling structured storage with a learned policy for using memory at the appropriate granularity. 

---
# Controlling Tool Use with Heading-Specific Activation Steering 

**Authors**: Yuqi Chen, Vincent Siu, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05790)  

**Abstract**: Tool-augmented large language models extend their capabilities beyond parametric knowledge through external tools, but tend to invoke them unnecessarily. We investigate whether tool-use decisions have any stable internal representation that can be extracted and manipulated, a question that is non-trivial given that tools exist entirely in context at inference time and have no direct encoding in model weights. We show that steering vectors extracted from heading-anchors positions exert bidirectional causal control over tool-invocation behavior across five open-source models and three domains, suppressing unnecessary tool use most effectively in domains where parametric reasoning suffices. However, geometric analysis reveals that this causal effectiveness does not correspond to clean linear structure: tool-invocation steps exhibit diffuse, bimodal alignment with the suppression vector rather than the consistent negative alignment a linear encoding account would predict, and different tool types recruit largely distinct internal signatures with low cross-tool feature overlap. We hypothesize these geometric properties are indicative of the non-parametric nature of tools, and distinguish tool-use steering vectors from those extracted for parametrically grounded concepts. The relationship between this geometric irregularity and the observed causal effectiveness remains an open question. 

---
# Beyond the Leaderboard: A Synthesis of Tool-Use, Planning, and Reasoning Failures in Large Language Model Agents 

**Authors**: Wael Albayaydh, Rui Zhao, Ivan Flechais  

**Link**: [PDF](https://arxiv.org/pdf/2607.05775)  

**Abstract**: Large language model (LLM) agents are increasingly evaluated on their ability to use tools, plan multi-step tasks, coordinate with other agents, and operate over extended horizons. Reported benchmark gains often obscure recurring failure modes documented across otherwise unrelated evaluation efforts. This paper synthesizes 27 benchmark, taxonomy, and audit papers (2023-2026), spanning 19 distinct benchmarks, into a cross-cutting taxonomy of agent limitations. To our knowledge, this is the first synthesis that integrates evidence across tool use, planning, long-horizon reasoning, multi-agent coordination, safety, and measurement validity into a single, unified taxonomy of LLM agent limitations. We identify six failure clusters: (1) tool invocation and parameter-level errors, (2) planning and constraint-satisfaction failures, (3) long-horizon degradation from context accumulation, (4) multi-agent coordination failures, (5) safety and security failures under adversarial or underspecified conditions, and (6) measurement validity problems. The taxonomy was derived iteratively by grouping independently reported error categories into themes corresponding to distinct stages of the agent reasoning-to-action pipeline. Across the literature, we find that failures compound nonlinearly with task length, that strong performance on individual sub-tasks does not reliably translate into end-to-end success, and that additional scaffolding does not consistently improve reliability. At the same time, substantial progress has been demonstrated in single-turn tool use, short-horizon web navigation, and narrowly scoped coding tasks. 

---
# Beyond Static Evaluation: Building Simulation Environments for Scalable Agentic Reinforcement Learning 

**Authors**: Akshay Arora, Ishan Nigam, Ashutosh Aggarwal, Shefali Bansal, Krishna Singh, Sweta Kumari, Nikhil Mittal, Shariq Farhan, Siddarth Malreddy  

**Link**: [PDF](https://arxiv.org/pdf/2607.05773)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents, traditional static evaluation fails to capture multi-step decision-making. We introduce AgenticAI-Supervisor, an API and UI-driven RL Gym environment that decouples environment creation from scalable execution. By moving to verifiable execution outcomes, the platform generates high-fidelity traces and applies multi-dimensional reward shaping. Critically, our framework mitigates reward hacking through rigorous internal state validation and testing. This work provides a first look at our platform's core capabilities through a Customer Support Agent case study demonstrating a consistent closed-loop feedback for model optimization. Future work will focus on advanced features such as Computer Use, Tool Use, automated "stumping", and edge-case generation. 

---
# Synthetic Consumer Insight Generation with Large Language Models 

**Authors**: Stephen L. France, Pia. A. Albinsson  

**Link**: [PDF](https://arxiv.org/pdf/2607.05761)  

**Abstract**: Modern data-driven marketing relies on large amounts of consumer data, yet collecting such data can be costly, time-consuming, and difficult to scale. This research examines whether large language models (LLMs) can be used to generate synthetic consumer data for projective techniques, a set of methods designed to elicit consumer associations, emotions, wants, and needs. We test LLM-generated responses across multiple projective tasks, LLMs, prompting strategies, and temperature settings, and compare them with human responses from a primary research study on perceptions of city tourism destinations. Human and LLM responses were analyzed using linguistic measures, diversity and concentration metrics, topic models, and top-term analyses. The results show substantial overlap between human and LLM responses in broad topics and associations, but also important differences in style, linguistic structure, and the way diversity is generated. Recommendations are given on how to best utilize LLMs for generating synthetic consumer data, how model and prompt choices shape response quality, and on recognizing the limitations of LLM synthetic consumer data generation. 

---
# ArtisanCAD: An Industrial-Level CAD Agent with Expert-Grounded Knowledge Distillation 

**Authors**: Yunhan Xu, Qifeng Wu, Xunjin Li, Yuanwei Bin, Qingsong Yao, Jianghang Gu, Guan Wang, Weihao Lv, Huiyu Yang, Wenfa Luo, Jiao Xiang, Yuntian Chen, Shiyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05750)  

**Abstract**: Computer-aided design (CAD) for industrial components requires long-horizon procedural modeling, robust feature dependencies, editable parametric geometry, and production-grade B-Rep execution. Existing text-to-CAD methods have made promising progress in generating CAD programs from natural-language descriptions, but they still struggle when user prompts are ambiguous, underspecified, or only describe high-level design intent. They also rarely exploit expert procedural knowledge naturally available in industrial workflows, such as CATIA operation recordings, macro logs, drawing notes, and engineering descriptions. We present \algname, a skill-guided industrial CAD agent with expert-grounded knowledge distillation. The core of \algname is CAD intermediate representation (CAD-IR), an executable procedural representation that encodes parameters, ordered operations, MCP tool bindings, dependencies, generated entities, and verification rules. CAD-IR plays two key roles: it first serves as the carrier for distilling expert CAD procedures into reusable parameterized skills; then it provides a procedural scaffold that turns vague or intermediate-level prompts into complete executable CAD operations. \algname retrieves expert-derived skills, instantiates and revises CAD-IR, executes the resulting procedure through a dedicated CATIA-MCP backend, and uses multi-view visual feedback for iterative refinement, and finally generates production-ready B-Rep models. On the Text2CAD benchmark, CAD-IR improves generation from intermediate prompts by reducing mean Chamfer Distance from $14.83$ to $9.88$, showing its ability to bridge ambiguous textual intent and executable CAD construction. On four complex automotive components, CAD-IR enables expert CATIA recordings to be distilled into reusable skills, allowing \algname to generate editable CATIA-native B-Rep models for new variant requests. 

---
# Akashic: A Low-Overhead LLM Inference Service with MemAttention 

**Authors**: Yang Liu, Zhaokai Luo, Huayi Jin, Ruozhou He, Chenchen Hong, Zhiyong Wang, Yifei Liu, Yunfei Gu, Chentao Wu, Junhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05708)  

**Abstract**: Recent LLM-based agent systems continuously accumulate context across multi-turn interactions, tool invocations, and cross-session workflows. Replaying the full history for every request quickly becomes impractical: long contexts increase prefill cost, may exceed context limits, and often bury task-relevant evidence in irrelevant content, degrading both serving efficiency and output quality. We propose Akashic, a low-overhead memory system built around MemAttention, which organizes context into bounded chunks and models semantic relationships across chunks, preserving cross-chunk evidence without repeatedly rewriting the full history. Akashic further applies hardware-software co-designed memory placement to co-locate likely co-retrieved chunks, reducing retrieval fragmentation and I/O overhead. Across four representative workloads and three model sizes, Akashic improves task accuracy by up to 10.2 points, throughput by up to 1.21x, and sustainable request rate by up to 1.88x over strong prior memory baselines. 

---
# Memory in the Loop: In-Process Retrieval as ExtendedWorking Memory for Language Agents 

**Authors**: Yusuf Khan, Carlo Lipizzi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05690)  

**Abstract**: Language agents run a loop - observe, reason, act - but the memory they reason over sits outside it: a store queried at most once per turn. We study the regime where memory moves inside the loop, read and written on every step. The obstacle has always been latency: networked stores answer in tens to hundreds of milliseconds, and in-loop retrieval can inflate end-to-end latency by up to 83x when retrieval is expensive. Prior work manages that cost rather than questioning it: serving-layer scheduling hides it, "memory-first" designs ration retrieval to once per turn. We argue latency is a property of where the store lives, not the in-loop pattern: an in-process store answers in ~100us, three orders of magnitude below the network regime, and at that speed the per-step tax collapses. By the extended-mind thesis's parity principle, a store fast enough to be constantly and directly available becomes extended working memory, not a tool the agent merely consults. The premise is causal: holding a fixed per-turn memory-latency budget and varying only the store's answer speed, redundant actions rise monotonically with latency - 0.0 of 12 at in-process speed, 7.2 of 12 at a 110ms cloud round trip (gpt-5-nano, gpt-5-mini; exact permutation p=0.0079). We demonstrate the regime end-to-end: across four GPT-5-class models under a bounded window, recall improves from 0/5 to 3.6-4.8/5 with in-loop memory, store ops at p50 80-165us - though an instructed restate-every-reply baseline also solves it perfectly, at a token cost that grows with the working set. The store never lost a fact in any run (244 of 244 writes kept); every miss traces to the agent's read policy, not the store. Our measurements also relocate the bottleneck: the dominant per-step cost is embedding (~200-400ms over the network); pairing the in-process store with a small local embedder returns the complete operation to a measured ~40us. 

---
# FirstResearch: Auditable Question Formation for LLM Scientific Discovery Agents 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05682)  

**Abstract**: LLM systems for scientific discovery increasingly assist with ideation, literature synthesis, experiment planning, and report generation, but the first research question they propose can remain difficult to audit: it may sound plausible without exposing the mechanism, falsifier, or assumption that a scientist should inspect. We introduce FirstResearch, a first-principles research-question formation framework for scientific LLM agents whose core artifact is a structured Research Question Certificate. The certificate records primitive definitions, assumptions, a mechanism model, a tension or contradiction, a falsifiable hypothesis, a minimal decisive test, and a failure update rule, making the proposed question inspectable before downstream execution. On ten LLM-agent research topics, FirstResearch outperforms controlled prompt-level baselines inspired by AI co-scientist, Agent Laboratory, and AI Scientist-v2 under a primary DeepSeek-blind-judge protocol. A Gemini-2.5-Flash independent-judge rescore of the same 40 baseline packages preserves the system-level ranking, with FirstResearch scoring 4.86/5 versus 4.38/5 for the strongest baseline and Pearson agreement of 0.865 on average score. A one-repeat ablation checkpoint further suggests that the certificate-centered core is the strongest component: certificate-only scoring reaches 4.90/5 under DeepSeek and 4.88/5 under Gemini, while removing certificates drops below 1/5 under both judges. These results are preliminary and use LLM judges rather than human domain experts, but they support a narrow scientific-discovery claim: explicit derivation constraints are a promising mechanism for making LLM-generated scientific questions more auditable. Code, prompts, saved outputs, and reproduction scripts are available at this https URL. 

---
# Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction 

**Authors**: Mohammad Saifullah, Thomas Kornmaier, Taaha Kazi, Vasu Sharma, Aditya Sanjiv Kanade, Aanand Kumar Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2607.05577)  

**Abstract**: Long-form fiction writers need memory that answers multi-hop questions about evolving story state: who knows a secret and when they learned it, whether an event preceded the narration that revealed it, whether a setup paid off, and how a relationship shifted. General-purpose retrieval and agent-memory systems represent entities and facts but not the narratological structure these questions turn on, so they surface the wrong evidence or none at all. We introduce the Narrative World Model (NWM), a writer-memory system that pairs a narratology-grounded typed temporal-state graph with query-conditioned hybrid retrieval. To measure memory rather than the answerer, we read every system through a single held-constant Opus 4.8 reader over only that system's chapter-safe evidence, on a reproducible public corpus and a validated multi-hop benchmark, and we compare against the strongest existing temporal-knowledge-graph agent-memory framework, Graphiti/Zep (Rasmussen et al., 2025). NWM substantially and significantly outperforms this baseline on multi-hop narratological QA across both corpora, and far exceeds GraphRAG and flat retrieval. The advantage is representational rather than an artifact of extraction: it survives rebuilding the baseline with NWM's own extractor, and traces to its narratology-grounded structure and query-conditioned retrieval, not to graph size or extractor quality. 

---
# Foundation Models for Automatic CAD Generation 

**Authors**: J de Curtò, Victoria Guillén, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2607.05573)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) enable the automatic generation of parametric 3D designs from natural-language specifications. This chapter presents an empirical study of foundation models for automatic Computer-Aided Design (CAD) generation of mechanical parts, using a unified evaluation pipeline and a curated benchmark of 97 engineering design problems. We introduce LLMForge, a multi-model text-to-CAD framework integrating JSON-schema validation, analytic feature scoring, mesh synthesis, and multi-round iterative refinement, studied under two critique regimes. IterTracer uses a Phong-shaded ray-trace renderer with analytic visual metrics (silhouette IoU, hole visibility, edge clearance, aspect-ratio conformance) for lightweight geometry-aware feedback across rounds. IterVision replaces the analytic scorer with a VLM semantic critic (Qwen2.5-VL-72B) that evaluates rendered views via chain-of-thought visual reasoning, assessing spatial coherence and design intent. On a benchmark spanning four canonical geometry families (plates with holes and bolt circles, multi-feature boxes, flanged cylinders, and L-brackets), we evaluate seven foundation models: DeepSeek-V3.2, Qwen3-235B-A22B, Llama-3.3-70B, Gemma-3-27B, GLM-4.5, MiniMax-M2.1, and INTELLECT. Under IterTracer, the four highest-ranked models form a tight cluster (overall mean in [0.885, 0.890]) with 98.97% mesh success, showing that compact instruction-tuned models can match substantially larger systems. VLM-based critique in IterVision yields 100% watertight mesh generation on the leading model while surfacing systematic difficulty on rotationally symmetric geometries such as cylinders, where visual and semantic scoring diverge most. We discuss benchmark design, failure modes, CAD-oriented prompting, and implications for industrial workflows and scalable automated mechanical design. 

---
# CSTutorBench: Benchmarking Small Language Models as Tutors for Block-Based Programming 

**Authors**: H. Chad Lane, Bryson Kageler  

**Link**: [PDF](https://arxiv.org/pdf/2607.05571)  

**Abstract**: Large language models are increasingly explored as AI tutors, yet deploying them in K-12 settings raises concerns around privacy, cost, and reliance on proprietary models. Small language models (SLMs) offer a promising alternative, but selecting the right model for a specific educational context remains difficult, particularly when the target domain, such as block-based programming, is largely absent from model training data. We introduce CSTutorBench, a benchmark for evaluating language models as CS tutors in VEX VR, a block-based robotics environment. The benchmark comprises 17 scenario-based questions scored against a pedagogical rubric grounded in established tutoring and feedback research, with a human-in-the-loop LLM-as-judge pipeline for evaluation. Preliminary findings across 11 models (4B-120B parameters) reveal that models perform well on surface-level criteria such as vocabulary and tone but struggle with deeper pedagogical behaviors, particularly avoiding answer leakage and engaging with student debugging histories. In our sample, model family and instruction-tuning approach appear to be better predictors of tutoring quality than parameter count alone, though the small number of models limits the strength of this conclusion. A targeted prompt revision grounded in recent educational prompt engineering research improved scores for 10 of 11 models. These results underscore the value of context-specific, pedagogically grounded benchmarks for SLM selection in educational deployment. 

---
# From Graphs to Gradients: Physics-Inspired Structural Attribution for Cyber-Physical IoT Systems and Beyond 

**Authors**: Spyridon Evangelatos, Christos Diou, Georgios Th. Papadopoulos, Evangelos Markakis, Panagiotis Sarigiannidis  

**Link**: [PDF](https://arxiv.org/pdf/2607.05563)  

**Abstract**: Interpretable explanation methods in Artificial Intelligence aim to uncover the underlying causes and their effects, enabling a deeper understanding of why a system behaves in a certain way under different inputs. Unlike traditional explainability methods, which mainly highlight correlations between input and output variables, causal explanation focuses on interventional questions. By doing so, it provides more robust insights, helping users understand automated decisions, especially in high-risk domains. Recovering an explicit directed causal structure, however, is often impractical in large-scale, hybrid cyber-physical systems with feedback loops and partial observability. This paper introduces a novel framework inspired by statistical mechanics that instead models variable dependencies through an undirected, energy-based representation of cyber-physical IoT systems. Our approach enables rigorous dependency-aware attribution by analysing how variations in the energy landscape reflect the influence of individual components, without recovering a directed causal graph. It also supports reasoning about perturbation effects across hybrid interactions, providing reliable explanations of abnormal behaviours. We empirically examined our framework through simulations on an industrial IoT testbed with hybrid continuous and discrete variables, demonstrating higher attribution accuracy, improved robustness and better scalability than state-of-the-art graph-based approaches. While the attributions are not intended to fully recover the system's generative dynamics, they provide valuable, dependency-aware explanations supporting both human interpretation and downstream predictive and diagnostic tasks. Although demonstrated in industrial IoT security, our framework also applies to other high-dimensional cyber-physical and socio-technical systems requiring principled, structural explanations. 

---
# Prompt-to-Paper: Agentic AI System for Bioinformatics 

**Authors**: Ramsha Kamran, Maheera Amjad, Zartasha Mustansar, Arsalan Shaukat, Salma Sherbaz, Muhammad U.S. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.05456)  

**Abstract**: While recent advances in large language models have enabled end-to-end automated manuscript generation, existing systems suffer from three critical deficiencies: (i) generated claims are not deterministically grounded in verifiable literature, (ii) experimental results are frequently fabricated rather than executed, and (iii) there exists no standardized, multi-dimensional framework to assess whether AI-generated manuscripts meet the quality and rigor required for real-world publication. We present Prompt-to-Paper, a multi-agent framework that directly addresses this evaluation gap through three integrated innovations. First, a deterministic retrieval-augmented generation pipeline with section-aware relevance scoring and snowball citation expansion grounds every claim in a verifiable corpus of 60--100 papers. Second, an autonomous coding agent executes real computational biology experiments replacing synthetic outputs with genuine numerical results. Third, an eight-dimensional automated quality scorer, benchmarked with approximate reference statistics from published papers and augmented with explicit hallucination penalties, provides standardized, reproducible quality assessments. The quality-driven improvement loop uses a context-rich reviser that routes each iteration to one of three researcher actions and fires a deep research cycle every ten iterations to re-run experiments and re-manuscript from stronger outputs. We validate the system on five bioinformatics case studies; all five cases compiled submission-formatted PDFs with zero out-of-range citations. The improvement loop raises manuscript quality by an average of +17.96 points on a 0--100 scale (maximum +26.04. As partial external checks, a human reviewer scored the five manuscripts at an average of 7.0 out of 10. Complete manuscripts are produced at approximately 0.31 USD per paper. 

---
# ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation 

**Authors**: Tianjiao Yu, Xinzhuo Li, Yifan Shen, Onkar Susladkar, Yuanzhe Liu, Xiaona Zhou, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2607.06565)  

**Abstract**: Unified 3D foundation models aspire to generate 3D assets and reason about them in language within a single backbone, but their text-3D interaction remains largely implicit. Existing methods concatenate text and 3D tokens into a flat sequence and rely on self-attention, collapsing coarse structural cues and fine geometric details into one undifferentiated representation. We introduce ELSA3D, a unified 3D model that addresses this with elastic semantic anchoring, structuring language and geometric reasoning jointly along matched abstraction scales. ELSA3D represents geometry with a scale-aware octree tokenizer and introduces Anchor Tokens, sparse cross-modal units that select semantic cues, route them to the most relevant 3D scale, retrieve scale-specific geometric evidence, and write the fused signal back into the unified representation, keeping interaction sparse yet precise. A lightweight per-block router makes both computation and reasoning elastic, choosing which text tokens instantiate anchors at which geometric scale so that cross-modal capacity concentrates where alignment is most needed. ELSA3D achieves state-of-the-art performance across image-to-3D generation, text-to-3D generation, and 3D captioning, outperforming the strongest unified baseline while roughly halving FLOPs and inference latency relative to the non-elastic version of the same model. 

---
# Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion 

**Authors**: Shervin Khalafi, Igor Krawczuk, Sergio Rozada, Charilaos Kanatsoulis, Antonio G Marques, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2607.06546)  

**Abstract**: Denoising graphs is a fundamental problem in graph learning and the core operation of graph diffusion models. Attention-based architectures like graph transformers have recently shown promise in denoising graphs. However, our principled understanding of attention-based graph denoising remains limited, making it unclear whether standard attention is the right mechanism for this task. Here we show that, under a denoising objective, linear attention is suboptimal and can only learn an average spectral denoising filter over the training distribution. This creates a fundamental limitation as graphs often vary spectrally across the distribution. To overcome this limitation, we introduce Spectral Attention, which directly utilizes the input graph spectrum and provably outperforms linear attention by a margin governed by the spectral diversity of the distribution. We then derive Graph Convolutional Attention (GCA), a practical and permutation-equivariant realization of this idea that implements spectral denoising through graph-filtered queries and keys. For stochastic block models, GCA provably matches the idealized Spectral Attention mechanism. We further show that the softmax operation, that follows the attention, provides additional denoising by approximately projecting noisy eigenvectors onto the clean eigenspace. Empirically, replacing linear attention with GCA consistently improves graph denoising and diffusion on synthetic and real datasets, with gains strongly correlated with spectral diversity. In DiGress, GCA matches standard graph-transformer performance without computing expensive structural features, and when combined with the recently proposed PEARL positional encodings, avoids explicit eigendecomposition computations resulting in faster inference without degrading quality. The code can be found here: this http URL 

---
# RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation 

**Authors**: Sambaran Bandyopadhyay, Ananth Muppidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06527)  

**Abstract**: Multi-hop Question Answering over Knowledge Graphs faces a critical challenge: traditional retrieve-then-read pipelines break differentiability, preventing the retriever from learning to bridge the semantic gap where intermediate nodes lack lexical overlap with the query. To address this, we propose RSF-GLLM, a framework decoupling differentiable graph reasoning from answer generation. Our Recurrent Soft-Flow (RSF) module employs a GRU-guided query updater to propagate continuous relevance scores, utilizing a dynamic gating mechanism to traverse semantically dissimilar bridge nodes via structural cues. We introduce flow sparsity regularization to theoretically guarantee convergence from soft probabilities to discrete reasoning paths. These paths are extracted and textualized to fine-tune a Large Language Model (LLM), ensuring generation is grounded in factual topology. Experiments on WebQSP and CWQ demonstrate that RSF-GLLM achieves competitive performance with superior inference efficiency compared to LLM based computationally expensive approaches. 

---
# Industry Classification of GitHub Repositories Using the North American Industry Classification System (NAICS) 

**Authors**: Kevin Xu, Alexander Quispe  

**Link**: [PDF](https://arxiv.org/pdf/2607.06505)  

**Abstract**: GitHub hosts hundreds of millions of public repositories, but the platform exposes no native mapping from repositories to standardized industry sectors. This gap limits empirical work on the geography of innovation, the industrial composition of open-source production, and the diffusion of new technologies across economic sectors. We present NAICS-GH, a publicly released corpus of 6,588 GitHub repositories drawn from source pools covering the United States, the European Union, and Australia, each labeled with a 2-digit sector from the North American Industry Classification System (NAICS 2022). Labels are produced by a retrieve-and-verify pipeline that combines BAAI/bge-large-en embeddings, FAISS retrieval, and GPT-4.1 rubric scoring. The pipeline narrows about 1.37 million source repositories to 31,178 candidate repository-sector pairs and retains 6,588 high-confidence labels with score at least 8. Re-running the retrieval pipeline end to end reproduces the candidate set to within 0.03 percent. On a 2,421-repository human-validated random sample, the released labels attain 96.98 percent precision, with Wilson 95 percent confidence interval [96.23, 97.59]. We benchmark six pretrained encoders on the released corpus; RoBERTa-large reaches 86.45 percent F1 and 86.35 percent accuracy on a held-out 20 percent test set. The dataset, Croissant metadata, pipeline code, prompts, and fine-tuned checkpoint are released under CC-BY-4.0 and MIT licenses. 

---
# Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engine 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2607.06495)  

**Abstract**: Live sports commentary is grounded generation under a deadline: statements concern real, named athletes, the grounding state changes every few seconds, and no reference text exists at generation time. We present Pitwall, a production system that generates natural-language Formula 1 strategy briefings in English, Spanish, and Portuguese, treating faithfulness as an architectural property rather than an aspiration: every published sentence is decomposed into typed factual claims (positions, gaps, tyres, pace, overtakes, race control) and each claim is verified against the probabilistic race state that prompted it. The same verifier gates the fine-tuning data: of 3,045 model-written targets, only the 81.9% whose every claim is state-supported are retained, the rest falling back to a provably faithful template, so the generator never sees an ungrounded target. Verification is meaningful because of the grounding substrate: a vectorized Monte Carlo engine (N=2,000 per-lap race continuations) calibrated on 126 races (2018-2024) and validated on fully held-out 2025-2026 seasons (winner-in-top-3 90.3% over 155 backtests; held-out Brier 0.0745). A recurring finding spans both halves of the system: virtues trade off and must be gated separately. In simulation, calibration-optimal is not decision-optimal; in generation, fine-tuning on richer targets buys vividness that collapses into hallucination when the grounding state is sparse -- a failure a four-base replication traces to base-model instruction adherence, not scale, and that sparse-context auditing removes from the production model. End-to-end operation -- live timing to verified trilingual briefings -- was confirmed at two consecutive live Grands Prix (Austria and Britain, 2026); at Silverstone a timestamped probability trace, committed to disk before the outcome was known, locked onto the eventual winner ten laps before the flag. 

---
# AirflowAttack: Thermal-Airflow Adversarial Perturbations against Infrared Remote-Sensing Vision-Language Models 

**Authors**: Cong Su, Jiaju Han, Xuemeng Sun, Chengyin Hu, Qike Zhang, Jiujiang Guo, Yiwei Wei, Jiahuan Long  

**Link**: [PDF](https://arxiv.org/pdf/2607.06485)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed on infrared (IR) remote sensing imagery in security-critical settings, yet their adversarial robustness remains unexamined. We present AirflowAttack, to our knowledge the first adversarial attack for IR remote-sensing VLMs and the first to weaponize thermal-airflow turbulence as the perturbation prior. A lightweight generator synthesizes a single input-agnostic perturbation regularized toward physically plausible airflow patterns. Optimized on one surrogate CLIP model, it attains a mean zero-shot scene-classification attack success rate (ASR, the fraction of samples whose top-1 class changes) of 48.5% across five diverse CLIP backbones, far exceeding four IR-specific physical baselines (27.7--37.0%). Applied to six state-of-the-art VLMs, it cuts scene-classification accuracy by up to 38.2% relative, yet paradoxically makes some models more confident in their IR analysis, confabulating the perturbation as genuine thermal evidence such as temperature gradients and convection. Ablations show the airflow prior raises physical plausibility at no measurable cost to attack success. Together with a benchmark spanning eleven models and four tasks, these findings expose critical vulnerabilities in the rapidly expanding IR VLM ecosystem. 

---
# Data Analysis in the Wild: Benchmarking Large Language Models Against Real-World Data Complexities 

**Authors**: So Hasegawa, Shailaja Keyur Sampat, Lei Liu, Wei-Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.06482)  

**Abstract**: Current benchmarks for evaluating Large Language Models (LLMs) in data analysis often fail to reflect real-world settings. They typically focus on fact retrieval from small tables and overlook the challenges of large multi-tabular datasets, external knowledge integration, and exploratory insight discovery. We introduce DataGovBench, a benchmark derived from governmental open data designed to evaluate LLMs in practical scenarios. The benchmark includes two tasks: Table QA that requires solving complex decomposable questions and producing textual answers or visualizations, and Table Insight that evaluates the ability of models to generate expert-level findings through exploratory data analysis. Comprehensive experiments with state-of-the-art LLMs, both with and without agentic frameworks, reveal significant performance gaps across both tasks. These results suggest that current LLM-based systems remain far from satisfying the demands of real-world data analytics. DataGovBench provides a challenging benchmark for advancing research on LLMs capable of both answering analytical queries and discovering insights from data. Code and sample data are available at this https URL. 

---
# Prompt-Adapter Context Routing for Parameter-Efficient Multi-Shot Long Video Extrapolation 

**Authors**: Anna Córdoba, Adam Puente Tercero, Nerea Angulo Hijo, Mar Linares Tercero, Julia Barrientos, Ainhoa Miranda, Jesús Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2607.06481)  

**Abstract**: We present PACR-Video, a parameter-efficient framework for multi-shot long video extrapolation that preserves recurring entities, scene structure, visual style, and causal progression without full generator fine-tuning. PACR-Video keeps a text-to-video diffusion transformer frozen and augments it with low-rank temporal adapters conditioned by learned shot-role prompt tokens. To maintain long-horizon coherence, it builds a recursive prompt bank that stores compact entity, location, action, and style prompts from previous shots, then routes them through adapter gates according to predicted narrative dependencies. A Shot-Local/Story-Global tuning objective combines next-shot reconstruction, cross-shot identity contrast, and prompt sparsity regularization, while an adapter composition schedule balances early-shot visual consistency with later-shot event progression and viewpoint change. Across six multi-shot and long-video benchmarks, PACR-Video outperforms text-to-video, tuning-based, memory-augmented, streaming, and recursive-context baselines on distributional quality, semantic alignment, identity consistency, temporal smoothness, motion stability, transition coherence, and human preference. These results show that compact prompt routing and lightweight temporal adaptation provide sufficient controllable capacity for stable long video extrapolation. 

---
# Provable learning separation for predicting time-evolution of quantum many-body systems 

**Authors**: Rahul Bandyopadhyay, Riccardo Molteni, Jens Eisert, Vedran Dunjko, Sofiene Jerbi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06472)  

**Abstract**: Given that quantum computers are naturally suited to simulate the behavior of quantum many-body systems, an immediate question arises: can one formulate physically motivated quantum machine learning (QML) tasks that exhibit learning separations? We address this problem by studying the learnability of quantum many-body dynamics from the perspective of probably approximately correct (PAC)-learning. Concretely, we devise a supervised learning problem where the training set consists of specifications of randomized stabilizer probe states, evolution times sampled uniformly from a polynomially large time interval $[0,T]$, coupled with expectation values of certain observables evaluated on the resulting time-evolved state under an unknown Hamiltonian. For this learning task, we provide an efficient quantum procedure whose training phase learns the underlying Hamiltonian from short-time training samples, and whose deployment phase combines Hamiltonian simulation with the classical shadows protocol to perform inference on a newly given data point. By contrast, the existence of $O(\mathsf{poly}(n))$-time instances ensures classical hardness: by embedding a $\mathsf{BQP}$-complete computation into the polynomially long time-dynamics of a low-intersection variant of the Feynman-Kitaev clock Hamiltonian construction, we show that, for a certain family of input distributions, no randomized classical polynomial-time algorithm can fulfill our learning condition, unless $\mathsf{BQP}\subseteq\mathsf{P/poly}$. Furthermore, we show that the classically hard instance maintains quantum learnability. We also give an interpretation of our results in learning-assisted certified quantum simulation. Taken together, our results demonstrate a rigorous learning separation for a natural ML task based on Hamiltonian evolution, while building connections between quantum learning theory, quantum simulation, and QML. 

---
# From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b 

**Authors**: Taeyun Roh, Eunha Lee, Wonjune Jang, Sohyun Chung, Junha Jung, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06452)  

**Abstract**: Biomedical question answering requires not only accurate extraction of information from scientific literature but also reliable integration of evidence across multiple documents. This study presents a question-type-specific large language model (LLM) framework for BioASQ 14b Task B, designed to improve answer robustness and evidence grounding in biomedical question answering. Rather than applying a single prompting strategy to all questions, the framework selects different inference procedures for yes/no, factoid, and list questions according to their distinct reasoning and evaluation requirements. For yes/no questions, snippet shuffling and self-reflection are used to reduce sensitivity to evidence ordering and improve decision stability. For factoid questions, full-snippet input is combined with chain-of-thought-based in-context learning to support accurate biomedical entity identification. For list questions, a multi-agent architecture is employed, in which evidence extraction, candidate generation, answer verification, and final aggregation are handled collaboratively. Preliminary experiments on BioASQ 13b were used to identify effective inference strategies for each question type, and the resulting framework was subsequently evaluated in the official BioASQ 14b Task B challenge. In the official evaluation, our framework showed competitive performance across multiple batches and achieved first place in the factoid subtask of Batch 4. These results demonstrate the effectiveness of combining question-type-specific inference, ensemble prediction, and agent-based verification for reliable biomedical question answering. 

---
# Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders 

**Authors**: Yoav Baron, Sara Dorfman, Roni Paiss, Daniel Cohen-Or, Or Patashnik  

**Link**: [PDF](https://arxiv.org/pdf/2607.06445)  

**Abstract**: Vision-Language Models (VLMs) are increasingly utilized as the conditioning backbone for diffusion-based image editing due to their remarkable multimodal reasoning capabilities. While standalone VLMs demonstrate strong localization capabilities, editing pipelines frequently struggle to maintain this accuracy, particularly in complex, multi-entity scenes. In this work, we investigate this performance gap, hypothesizing that it stems from treating the VLM as a condition encoder. In this role, the model is restricted to a single forward pass, preventing the autoregressive generation process for which it was optimized, thereby failing to fully expose its capabilities. To investigate whether this spatial understanding persists when the VLM is used as a condition encoder, we introduce Analysis-by-Proxy. In this framework, we train a lightweight, interpretable proxy model on the VLM's intermediate representations using an auxiliary localization task. By analyzing the VLM through this proxy, we uncover the specific VLM representations that encode localization information. Our findings expose a fundamental mismatch between how spatial knowledge is represented within a VLM condition encoder and how it is extracted by current editing pipelines. We reveal that under single-pass constraints, the localization signal does not reliably propagate to the predefined layer configurations commonly used for conditioning. Instead, this crucial signal remains hidden within intermediate representations, at locations that vary depending on the input prompt. Using our introduced Analysis-by-Proxy framework, we reveal the fundamental failures of existing condition extraction strategies in editing pipelines, opening the door to more principled design of conditioning architectures. 

---
# TILDE: TILt-based Distributional Erasure for Concept Unlearning 

**Authors**: Naveen George, Naoki Murata, Yuhta Takida, Konda Reddy Mopuri, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2607.06432)  

**Abstract**: Concept unlearning in text-to-image diffusion models is critical for safe and practical deployment: with rising privacy concerns, copyright disputes, trademark constraints, and safety regulations, deployed systems must be able to suppress unwanted concepts after training. Existing methods often remove the target concept effectively, but practical unlearning also requires an equally fundamental property: the unlearned model should retain quality, diversity, and semantic coverage on benign generation. The gold standard is a retain-only model trained from scratch without the unwanted data. However, common erasure objectives do not specify which post-unlearning distribution should approximate this reference, leaving retention as an implicit consequence of the update rule. We propose TILDE, TILt-based Distributional Erasure, which formulates concept unlearning as a distributional alignment problem: the desired target is the minimum-deviation conditional distribution from the pretrained model under a forgetting constraint. This energy-tilted, anchor-free target suppresses concept-expressing images while preserving benign relative mass for each prompt. We instantiate this principle with residual $\nabla$-GFlowNet training, which learns the score correction induced by the forget energy relative to the pretrained diffusion model. Across objects, artistic styles, and characters, TILDE achieves strong forgetting while improving retention and distributional fidelity over prior baselines. 

---
# An Experimental Design Approach to Evaluating Agentic AI's Autonomous Model Discovery 

**Authors**: Hao He, Xueying Liu, Chris J. Kuhlman, Xinwei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.06413)  

**Abstract**: Large language model coding agents increasingly perform open-ended data modeling and analysis. These agents are stochastic and adaptive, and therefore their autonomous model discovery behavior cannot be adequately characterized by a single benchmark run. In this work, we propose an experimental design and analysis framework for systematically evaluating this discovery process, quantifying its variability, and identifying important factors. The proposed framework treats these agents as stochastic model-discovery operators, which map task-specific discovery data and an optimization target to a fitted model. Specifically, we investigate two such operators, Codex and Claude Code, under controlled experimental factors including agent's reasoning effort, task, optimization metric, and composition of training data. For each agent-task-metric combination, regression models and inference are conducted for multiple responses such as output quality, dollar cost, wall-clock time, and process complexity. Furthermore, we develop a utility-aligned canonical decomposition to characterize the dominant direction of the reasoning-effort effect and to assess whether that direction aligns with a performance-cost utility direction. The proposed framework is demonstrated on a testbed of networked word-forming games with insightful findings on reasoning effort with respect to cost and process complexity. 

---
# RuBench: A Repository-Level Agentic Coding Benchmark with Natively Authored Russian Task Specifications 

**Authors**: Evgeny Shilov  

**Link**: [PDF](https://arxiv.org/pdf/2607.06411)  

**Abstract**: Developers increasingly delegate real maintenance work to product-grade coding agents, and many state tasks in their native language, in the style of a customer request rather than
a curated English issue. Existing repository-level agentic benchmarks do not measure this setting: their task statements are English by design. We introduce RuBench 1.0, a
benchmark of 25 tasks mined from recent fix commits in five live open-source repositories (aiohttp, aiogram, Laravel, NestJS, Fastify; Python, PHP, TypeScript, JavaScript), where
each task is specified natively in Russian -- written from scratch in the style of an actual customer request, not translated -- and judged by the upstream maintainer's regression
tests, which we withhold from release. All 25 fix commits postdate the training-data cutoffs of every evaluated model, giving a contamination argument that holds task-by-task. We
evaluate deployed product configurations (CLI agent + model + reasoning effort) -- Claude Code with Opus 4.8, Sonnet 5, and Haiku 4.5, and Codex CLI with GPT-5.5 -- with three
independent runs each, reporting pass@1 with task-level confidence intervals, paired comparisons, dollar cost, and token usage. The best configuration resolves 78.7% of tasks; at
N=25 only the gaps to the weakest model are statistically resolvable, which we state explicitly. Auditing full trajectories of a fifth, hors-concours configuration (Claude Code +
Fable 5, July 2, 2026 release), we caught the product silently substituting the model: on 5 of 25 tasks (20%) an official safeguard fallback re-routed routine HTTP-protocol fixes
to Opus 4.8 -- direct, reproducible evidence that the deployed product, not the model, is the unit actually measured. We release task statements, metadata, full agent trajectories,
and diffs; grading oracles are withheld, with a SHA-256 manifest committed at publication time. 

---
# What Images Cannot Say: Language-Guided Olfactory Representation Learning 

**Authors**: Eleftherios Tsonis, Xi Wang, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2607.06402)  

**Abstract**: Images tell us what a scene looks like, but rarely what it would feel like to be there. While recent datasets pair visual scenes with electronic-nose measurements, aligning smell signals with images remains challenging because many olfactory cues arise from contextual environmental factors that are not directly visible in pixels. We introduce SCENT, a multimodal framework that uses language guidance as a semantic bridge between vision and olfaction. Our approach leverages Vision-Language Models (VLMs) to generate scene descriptors capturing objects, environmental context, and plausible ambient smell cues suggested by the visual scene. These descriptors provide semantic guidance for learning olfactory representations. We train a smell encoder that maps electronic-nose signals into a shared embedding space aligned with both visual and textual representations, and introduce a languageguided latent decomposition that separates object-specific odors from contextual environmental contributions. Experiments on the New York Smells dataset demonstrate that SCENT significantly improves crossmodal retrieval compared to vision-only baselines, achieving state-of-theart performance on smell-to-image and smell-to-text retrieval tasks. In addition, our framework produces interpretable olfactory representations that enable the disentanglement of complex smell mixtures. Our results reveal the importance of contextual semantic information for grounding olfactory perception in multimodal learning and pave the way for future research in this area. 

---
# Responsible Personalisation: The Double-Edged Sword of Personalisation in Human-Robot Interaction 

**Authors**: Antonio Andriella, Jauwairia Nasir, Andrea Rezzani, Alyssa Kubota, Dimitri Lacroix, Tamlin Love, Aniol Civit, Vicky Charisi, Elisabeth Andre, Wing-Yue Geoffrey Louie  

**Link**: [PDF](https://arxiv.org/pdf/2607.06344)  

**Abstract**: While personalisation is becoming a defining capability in human-robot interaction (HRI), the existing literature on responsible personalisation remains fragmented, offering isolated accounts of ethical risks without a structured understanding of how they emerge across interaction contexts. This gap is particularly critical in HRI, where robots' embodiment and social presence can amplify and reshape such risks or generate new types of risks. We present a lifecycle-based and context-sensitive framework for personalised HRI, grounded in an embodiment-aware perspective. The framework combines stages of the personalisation process with interaction characteristics (short-term vs. long-term, open-domain vs. closed-domain), enabling systematic analysis of how risks arise and evolve. Building on this, we conduct an integrative analysis of key ethical risks, including autonomy erosion, biased user modelling, manipulation, dehumanisation, and privacy violations, and examine how they manifest across contexts. We translate these insights into actionable design recommendations and outline open research challenges. By structuring both the design space and risk landscape of personalised HRI, this work provides a foundation for more systematic, transparent, and ethically grounded approaches to personalised robot behaviour. 

---
# Harnessing Code Agents for Automatic Software Verification 

**Authors**: Shuangxiang Kan, Shuanglong Kan, Sebastian Ertel  

**Link**: [PDF](https://arxiv.org/pdf/2607.06341)  

**Abstract**: Formal verification offers the strongest guarantee of software correctness, but it does not scale: the proofs demanded by interactive theorem provers such as Coq require enormous expert effort. Large language models (LLMs) promise to generate these proofs automatically, yet existing approaches wire a fixed, human-designed proof strategy into the system and constrain the model to follow it (retrieving premises and predicting tactics one step at a time, or splitting goals by divide-and-conquer), and still prove only a fraction of their target theorems.
We show that imposing such a strategy is unnecessary and limiting. Handing the whole lemma to a general LLM code agent (for example, Claude Code), free to choose its own approach, and wrapping it in a verification harness is both simpler and more effective, achieving full coverage: every targeted lemma proved, with no failures and no Coq expert intervention. The agent writes the proofs under feedback and hard constraints from the harness that keep each one sound (accepted only when the prover's kernel closes it), complete (no obligation left unproved or silently dropped), and terminating (no divergent tactics).
We evaluate this harness plus code agent along three dimensions. (1) Core logic: on Iris, the state-of-the-art separation logic for concurrent and memory-manipulating programs, Aria proves all 4,257 lemmas of the four core modules and the 217 lemmas verifying Rust's standard libraries built on it, fully automatically. (2) Comparison with prior LLM provers: on reglang, where prior provers manage barely one in eight, Aria proves all 318. (3) Generality: on iris-lean, the unfinished Lean 4 port of Iris, it proves 72 not-yet-ported lemmas, showing the approach is not specific to Coq. A state-of-the-art model (Claude Opus 4.7) can write proofs for verified software development fully and automatically. 

---
# Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs 

**Authors**: Andrea Alfarano, Andrea Bacciu, Saab Mansour, Amin Mantrach, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2607.06327)  

**Abstract**: Uncertainty estimation (UE) enables LLM-powered systems to recognize when to abstain, yet existing research has predominantly focused on English. We present the first large-scale evaluation of UE methods across 22 languages, spanning high-, mid-, and low-resource settings. Using two human-curated Q\&A datasets, we compare open and closed box UE methods (nine in total) across different model sizes and architectures while eliciting long-form reasoning, avoiding LLM-as-a-judge and embedding-based scoring, which can introduce evaluation noise. We report three main actionable findings. First, we find that prompting models to reason in English while keeping questions in low-resource languages substantially improves UE performance, suggesting that comprehension of low-resource languages is largely intact, and that the reliability bottleneck lies in generation rather than understanding. Second, prompting models to reason in English closes the UE performance gap between low and high-resource languages, demonstrating that generation language matters more than the question language. Third, the choice of UE method should depend on model scale: at smaller scales, open-box probability-based methods outperform alternatives; at larger scales, closed-box self-verbalized uncertainty becomes superior. Finally, we provide an analysis of threshold selection for selective prediction, offering guidance on calibrating abstention in multilingual settings. 

---
# Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification 

**Authors**: Aysan Ghayouri Pirsoltan, Shima Babakordi, Mohammad Reza Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06309)  

**Abstract**: Accurate breast cancer classification from mammography requires effective integration of complementary information from craniocaudal (CC) and mediolateral oblique (MLO) views, which provide a more complete characterization of breast abnormalities. However, existing multi-view learning approaches typically rely on feature-level aggregation or single-stage cross-attention, which can entangle view-specific and shared representations and restrict interaction to limited network depths. To address these limitations, we propose a token-centric dual-view learning framework that unifies prompt-based adaptation and cross-view fusion within a frozen vision transformer backbone. The framework reformulates inter-view interaction as structured token-level communication, where dedicated fusion tokens explicitly encode bidirectional information exchange between CC and MLO views via cross-attention, serving as intermediate carriers of cross-view dependencies rather than relying on direct feature fusion. Unlike conventional methods that apply fusion at a single layer, fusion modules are inserted at multiple transformer depths, enabling progressive and repeated interaction across the encoder hierarchy. Fusion tokens are reintegrated into the token sequence and refined by subsequent transformer layers, facilitating hierarchical propagation of complementary information while preserving view-specific structure. Experiments on VinDr-Mammo and CMMD datasets demonstrate consistent improvements over linear probing, prompt-only adaptation, and conventional fusion baselines. On the VinDr-Mammo BI-RADS classification task, the framework achieves 50.40% F1-score and 0.8090 AUC, including a 0.10 AUC improvement over a dual-view fusion baseline in the binary setting. Ablation studies further validate the effectiveness of token-based fusion and multi-depth interaction design. 

---
# UI2App: Benchmarking Visual Interaction Inference in Executable Web Application Generation 

**Authors**: Grace Man Chen, Litao Guo, Yifan Wu, Yiyu Chen, Yenchi Tseng, Sicheng Liu, Yuyu Luo, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.06306)  

**Abstract**: Large language models (LLMs) have demonstrated growing competence in web page generation. However, existing text-driven approaches rely on complex prompts that impose substantial demands on users and offer limited expressivity for page layout and cross-page visual coherence. Image-driven paradigms, which take UI screenshots as input, align more closely with real development workflows. However, current benchmarks focus primarily on visual fidelity and lack a systematic evaluation of the interaction capabilities in generated artifacts. To address this gap, we introduce UI2App, the first benchmark targeting interaction inference, the ability to recover application behavior from screenshots alone, without any textual or behavioral guidance. UI2App comprises 327 screenshots grouped into 45 state-coherent screenshot sets for runnable multi-route web applications. We design an end-to-end pipeline that evaluates each artifact along four dimensions: executability, navigation reachability, visual fidelity, and interaction inference. The interaction metric (IIS) assesses inferred interactions by functional correctness and state-management complexity, crediting any valid implementation rather than matching a single reference. Experiments on six frontier vision-language models reveal a marked capability mismatch between visual reconstruction and interaction realization: the visual-fidelity leader scores only 7.5 on IIS, ranking fourth and trailing the IIS leader by 5.2x. High-complexity interactions such as cross-page state remain a pervasive bottleneck, with half of the evaluated models scoring exactly zero on this dimension. Overall, the results indicate that inferring complete interaction behavior from static screenshots remains a key challenge for models. 

---
# Designing Maintainable Hybrid Generative Systems: A Quantum-Inspired Approach to Automated Music Harmony Generation 

**Authors**: Josef Pavlicek  

**Link**: [PDF](https://arxiv.org/pdf/2607.06296)  

**Abstract**: This paper presents the design and evaluation of a maintainable hybrid generative architecture for automated music harmony generation from melody. The proposed system combines quantum-inspired candidate exploration over overlapping melodic contexts with explicit rule-based optimization to balance generative flexibility and structural control. The architecture is evaluated using explicit and reproducible metrics covering structural coherence, functional agreement, harmonic similarity, and robustness. The results show that the proposed approach produces harmonizations that preserve tonal structure and cadential behavior while allowing multiple valid harmonic realizations. Furthermore, the optimization layer improves structural coherence, stability, and predictability without requiring a training corpus. The study demonstrates that transparent and controllable hybrid generative systems can be systematically designed and evaluated within the context of Information Systems Development. 

---
# VendorBench-100: A Unified Cross-Paradigm Benchmark for Deepfake Image Detection 

**Authors**: Sharayu N. Deshmukh, Md Rashidunnabi, Nelton Tiago Gemo, Kurundkar G. D., Mahamune M. R., Nilesh K. Deshmukh  

**Link**: [PDF](https://arxiv.org/pdf/2607.06254)  

**Abstract**: Deepfake image detection is currently served by three fundamentally different paradigms: commercial APIs, zero-shot vision-language models (LLMs), and open-source detectors. Despite their widespread use, these paradigms are rarely evaluated under a common protocol, making direct comparison difficult. We introduce VendorBench-100, a cross-paradigm benchmark that evaluates 36 representative models using a single adversarial 100-image corpus, a unified output schema, and a common evaluation framework. To ensure reliable assessment under the corpus's intentional class imbalance, models are ranked primarily by the Matthews correlation coefficient (MCC), with ROC-AUC reported as a threshold-independent measure of ranking ability. Rather than maximizing dataset size, VendorBench-100 emphasizes challenging real-world scenarios through a curated taxonomy of eight edge-case families, including face swaps, text-to-video stills, AI photo edits, avatar compositing, opaque-provenance images, and compressed research frames. Our evaluation shows that commercial APIs achieve the strongest median performance, followed by vision LLMs and open-source detectors. However, individual open-source models remain competitive with the best vision LLMs. More importantly, we identify a consistent divergence between ranking ability (ROC-AUC) and operating-point quality (MCC), demonstrating that strong score discrimination does not necessarily produce reliable default-threshold decisions. This metric disagreement, rather than any single leaderboard ranking, is the central finding of the benchmark. We release the complete evaluation framework and benchmark results to support reproducible future research. The source code and data are available at: this https URL 

---
# Spider 2.0-AIFunc: Extending Real-World Text-to-SQL to AI-Native SQL Workflows 

**Authors**: Tianyang Liu, Canwen Xu, Fangyu Lei, Nikki Lijing Kuang, Jixuan Chen, Tao Yu, Julian McAuley, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2607.06229)  

**Abstract**: Major cloud data platforms now expose large language model capabilities as native SQL functions, enabling analysts to perform classification, filtering, sentiment analysis, extraction, similarity search, and aggregation within ordinary SQL queries. Yet existing text-to-SQL benchmarks evaluate only conventional SQL and provide no signal on whether models can generate such AI-native SQL. We introduce Spider 2.0-AIFunc, a benchmark of 465 verified instances across 125 real-world databases covering six types of AI functions on the Snowflake platform. Starting from an existing enterprise text-to-SQL benchmark, we construct Spider 2.0-AIFunc through an agent-based pipeline that rewrites source tasks into AI-native form, simultaneously transforming target queries and refining natural language instructions to make the intended AI-native solution explicit and reduce ambiguity. All instances pass a multi-round repeated execution protocol across temporally separated windows to confirm result stability before release. Evaluating ten state-of-the-art language models, we find that the strongest proprietary models reach 67-70% execution accuracy while the best open-source model achieves 58.1%, a gap driven primarily by errors in predicate specification, schema grounding, and AI function parameterization. Agent frameworks designed for traditional text-to-SQL challenges, such as schema retrieval and relevant table selection, do not transfer effectively to AI-native SQL: a minimal agent setup consistently matches or outperforms more elaborate alternatives, suggesting that the strategies these frameworks employ are less critical in this setting. Data are available at this https URL . 

---
# UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods 

**Authors**: Yipeng Liu, Chang Liu, Si Shen, Jiaqi Zheng, Mingfan Li, Yuyang Yang, Guanhua Li, Yuquan Zhang, Yimeng Xu, Zhongzhe Hu, Zhiyuan Huang, Qihang Duan, Junsong Wang, Wenkai Ling, Baochuan Yang, Xianzhi Yu, Han Bao, Yijie Chen, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.06202)  

**Abstract**: The deployment of Mixture-of-Experts (MoE) models on production high-bandwidth superpods, such as NVIDIA's NVL72/576 and Huawei's CloudMatrix384, introduces critical challenges beyond raw interconnect bandwidth. While these systems provide unified global address spaces and high-bandwidth fabrics, their full potential for sparse MoE communication is hindered by three fundamental bottlenecks: (1) Strict execution serialization imposed by coarse-grained Bulk Synchronous Parallel (BSP) orchestration of interdependent communication phases; (2) Prohibitive synchronization overhead that fails to scale alongside high interconnect bandwidth; and (3) Severe load imbalance resulting from distance-agnostic scheduling of irregular token traffic. To eliminate these bottlenecks, we introduce UBEP (Unified-Bus Expert Parallelism), a production-ready communication library that rethinks MoE's All-to-All primitives for modern superpod architectures. Through large scale experiments, UBEP reduces All-to-All latency by up to 52.4% and MoE inference Time Per Output Token (TPOT) by up to 11.1%. 

---
# TriA Pipeline: A Large-Scale Automatic Audio Annotation Pipeline For Audio Classification In Specific Scenarios 

**Authors**: Hong Lyu, Mingru Yang, Qianhua He, Yanxiong Li, Jinxin Huang, Zhengyu Pei  

**Link**: [PDF](https://arxiv.org/pdf/2607.06179)  

**Abstract**: There are some datasets of varying scales for audio classification (AC) applied to different tasks. However, annotated data is limited for most scenarios, such as domestic environments. To address this challenge, we propose an $\textbf{A}$utomatic $\textbf{A}$udio $\textbf{A}$nnotation Pipeline--TriA Pipeline, which can efficiently convert audio from various scenarios into high-quality training data with audio event annotations. A TriA dataset was constructed with the TriA Pipeline, over 2130 hours of audio covering 431 audio classes. Furthermore, we partitioned a prior-knowledge-guided subset (TriA$_{\mathrm{GK}}$) from TriA and conduct comparative experiments on three domestic AC tasks. Comparing the result on manually annotated data only and that on manually annotated data combines TriA$_{\mathrm{GK}}$, TriA$_{\mathrm{GK}}$ could achieve average relative gains of 3.97% in accuracy and 3.35% in Macro-F1, validating the effectiveness of TriA$_{\mathrm{GK}}$ and the TriA Pipeline. 

---
# Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design 

**Authors**: Alexander Rombach, Chantale Lauer, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2607.06175)  

**Abstract**: Large language models (LLMs) can generate BPMN process models from natural-language descriptions, yet supervised fine-tuning (SFT) limits their output quality to the patterns present in the training data. Reinforcement learning (RL) can optimize beyond this ceiling using external quality measures, but how the reward function should be designed when quality is multi-dimensional remains unexplored. We present a systematic investigation of reward function design for RL-based process model generation, training two LLM families (Llama~3.1 8B, Qwen~2.5 14B) under 48 configurations using Group Sequence Policy Optimization with rewards derived from an automated evaluation framework comprising 38 metrics across syntactic, pragmatic, and semantic quality. Three findings emerge. First, RL significantly improves pragmatic and syntactic quality while preserving semantic fidelity, reducing output variability by more than sixfold. Second, equal reward weighting consistently outperforms targeted weighting: emphasizing a specific dimension fails to improve it and can collapse the model into a low-quality mode. Third, design choices interact with model architecture in non-trivial ways: the invalidity penalty is essential for one model but irrelevant for the other, and SFT initialization is indispensable for one architecture but counterproductive for another. These results demonstrate that reward composition is a primary determinant of optimization outcomes, with effects as large as the decision to apply RL itself. The findings generalize to any structured generation task where quality is assessed along multiple automated dimensions. We release our implementation and experimental code at this https URL. 

---
# X-FEMR: A Token-level Explainable Approach for Electronic Health Records Foundation Models using Transformer-based Models 

**Authors**: Jie Huang, Pengfei Yin, Zihan Xu, Daniel Capurro, Mike Conway, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06163)  

**Abstract**: Foundation Models for Electronic Health Records (FEMRs) are pretrained on large-scale structured patient data, enabling them to convert longitudinal patient trajectories into generalizable representations for diverse clinical prediction tasks. Despite their effectiveness, FEMRs remain black-box models, raising concerns about bias, interpretability, and clinical trust. To address this, we propose the first token-level explainability approach for FEMRs. We train a Transformer-based surrogate model on input-output pairs from the FEMR across two prediction tasks, approximating its behavior while preserving temporal dynamics. We identify the most influential tokens, providing insights into how FEMRs leverage different aspects of patient history for predictions. To evaluate clinical relevance, we introduce a novel clinical alignment metric that quantifies the correspondence between the surrogate model's key tokens and clinically validated features. Our results demonstrate that the surrogate closely approximates FEMR predictions and that token-level explanations align well with clinical knowledge, offering a practical framework for interpretable and trustworthy clinical AI. 

---
# LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis 

**Authors**: Chenhao Yuan, Yinhao Xu, Shuwen Xu, Xizhi Yang, Jiaxiang Liu, Chenxi Zhou, Shaoping Huang, Haolin Ren, Pengfei Cao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06160)  

**Abstract**: Synthesizing long-context supervised fine-tuning (SFT) data is a scalable way to enhance the long-context understanding of large language models (LLMs), yet existing approaches share three limitations: narrow task coverage, insufficient instruction difficulty, and a lack of faithfulness supervision. We propose \textbf{LongCrafter}, a structured synthesis framework that couples a hierarchical task taxonomy with an evidence-grounded pipeline. The taxonomy organizes long-context understanding into local/shallow and global/deep levels and yields 32 fine-grained task types that serve as a global generative prior. Guided by this taxonomy, LongCrafter constructs task-aligned long contexts, decomposes them into explicit evidence graphs that model cross-paragraph dependencies, and generates instruction--response pairs strictly grounded in the located evidence spans, ensuring both controllable difficulty and faithful, traceable reasoning. Models fine-tuned on LongCrafter data outperform all SFT baselines and even the official post-trained models on LongBench, LongBench~v2, and LooGLE across both Qwen2.5-7B and LLaMA-3.1-8B, with the largest gains on high-difficulty tasks. Further analysis shows that LongCrafter data is more diverse and better spread across difficulty levels, and that the trained models locate evidence robustly regardless of position, effectively mitigating the ``lost in the middle'' problem. 

---
# LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability 

**Authors**: Chenxu Wang, Yongkun Yang, Boyuan Du, Shiwei Lin, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06157)  

**Abstract**: Deliberation plays a crucial role in collaboration; when humans work together, they naturally engage in communication to align information and reach an agreement. In this paper, we investigate deliberative large language model (LLM) agents under partially observable joint decision-making tasks. We formalize deliberative collaboration as a cooperative joint decision problem with partial and asymmetric observations, and introduce a scalable benchmark that instantiates this problem across multiple task settings and domains in which agents must exchange information through deliberation to reach a joint decision with a shared reward. We then instantiate a reference scaffold and evaluation protocol for deliberative agents and conduct a systematic evaluation of a range of representative LLMs. The results reveal that complex deliberative collaboration tasks continue to challenge state-of-the-art language models. Even with the aid of external mathematical tools, language models may fail in either the deliberation process for aligning information or the complex reasoning process for making the decision. On the other hand, diagnostic analysis reveals that the deliberation process may also provide opportunities for reflection and error correction, sometimes improving performance over centralized baselines. Altogether, our work establishes a foundation for evaluating and improving LLM agents in deliberative collaboration and provides insights into the strengths, limitations, and properties of current LLM-based multi-agent systems. 

---
# Property-Driven Synthetic Data Engineering for Data-Scarce Software Systems: Reflections from the Breast Cancer Domain 

**Authors**: Aurora Francesca Zanenga, Andrea Bombarda, Marsha Chechik, Saverio D'Amico, Rita De Sanctis, Alberto Zambelli, Claudio Menghi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06133)  

**Abstract**: Modern software systems increasingly depend on data for analysis, prediction, testing, and decision-making. Yet many important domains, including medicine, safety-critical systems, and regulated industries, lack abundant, shareable, or representative data. Synthetic data generation is often proposed as a remedy, but our experience engineering software for intraoperative radiotherapy (IORT) in breast cancer treatment suggests that synthetic data shifts rather than solves the central engineering problem. The key challenge becomes deciding which properties synthetic data must preserve, how these properties should be elicited from stakeholders, how they can be validated under privacy constraints, and how they evolve. We call this problem property-driven synthetic data engineering. Drawing on a collaboration with oncologists and preliminary experiments with a sensitive IORT dataset, we identify challenges in requirements, validation, privacy, and pipeline evolution. We argue that automated software engineering research should develop methods and tools for eliciting, formalizing, checking, and evolving validity properties for synthetic data in data-scarce software systems. 

---
# Self-Supervised Implicit CEST Reconstruction via Physics-Informed Lorentz Encoding 

**Authors**: Dexuan Li, Yupeng Wu, Chenglong Wang, Hanlin Liu, Hui Zhen, Jianqi Li, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06132)  

**Abstract**: Multi-Pool Chemical Exchange Saturation Transfer (CEST) MRI provides valuable metabolic information but is clinically limited by long acquisition times. Although sparse sampling reduces scanning time, reconstructing high-resolution Z-spectra from limited data remains an ill-posed inverse problem. Conventional interpolation and generic Implicit Neural Rep-resentations (INRs) often lack physical constraints, leading to spectral artifacts and physically invalid signals. To address this, we propose Lorentz Encoding (LE), a physics-informed framework that formulates CEST reconstruction as a self-supervised reconstruction task via implicit continuous coordinate learning. Unlike generic positional encodings, LE regularizes the continuous spectral mapping by projecting sparse coordinates into a physically constrained space governed by a combination of parametric Lorentzian profiles with learnable basis functions. This mechanism effectively reduces noise and enforces consistency with physical models. Experiments on in vivo human brain data demonstrate that LE significantly outperforms state-of-the-art methods. Specifically, under a 39-point sampling strategy, LE achieves a PSNR of 57.58 dB and an SSIM of 0.9994. Furthermore, the learned physics-informed encodings form a continuous, geometrically ordered trajectory in the latent space, ensuring accurate quantitative metabo-lite mapping (APT, NOE, MT). 

---
# Evaluating Fine-Tuning and Metrics for Neural Decompilation of Dart AOT Binaries 

**Authors**: Raafat Abualazm, Ayman AboElhassan, Amr G. Wassal  

**Link**: [PDF](https://arxiv.org/pdf/2607.06125)  

**Abstract**: Neural decompilation is increasingly studied as a code-generation problem, yet its evaluation methodology remains underdeveloped for modern languages. We present a systematic empirical study of fine-tuning effectiveness and metric validity for Dart Ahead-of-Time (AOT) neural decompilation. We evaluate six fine-tuned model variants across three base architectures (4B-8B parameters) using three metrics: CodeBLEU, compile@k, and pass@k on a new 154-task HumanEval-Dart benchmark. Our study yields three principal findings grounded in paired task-level statistical tests. First, no fine-tuning configuration produces a statistically significant pass@k improvement. The sole positive case yields +0.71 pp (McNemar p=0.21), while fine-tuning the strongest base (Qwen3-8B) causes a highly significant regression of -5.65 pp (p<0.001). This capacity-dependent trend is consistent across architectures but needs broader scale sweeps. Second, cross-lingual interference from Swift training is highly significant at 4B (-2.66 pp, p<0.001) but statistically indistinguishable from zero at 8B, consistent with the scaling hypothesis. Third, we demonstrate metric divergence: CodeBLEU and compile@k can improve significantly while pass@k moves in the opposite direction. This has implications for any LLM code generation task where fine-tuning targets superficial similarity. Error analysis reveals assembly sequence length is the strongest predictor of task difficulty (p=0.001), with a capability cliff at 200 instructions. We contribute the HumanEval-Dart benchmark, a Dart-adapted CodeBLEU, and empirical evidence that pass@k must be the primary evaluation metric for neural decompilation. 

---
# Static Metrics Are Insufficient: Predicting Java Method Energy Usage with Execution Time 

**Authors**: Muhammad Imran, Vincenzo Stoico, Ivano Malavolta  

**Link**: [PDF](https://arxiv.org/pdf/2607.06124)  

**Abstract**: The increasing energy demand of software systems is raising concerns about their environmental impact and associated costs. Reasoning on energy usage early in the development flow has the potential to significantly reduce the overall energy usage of a software system, as it allows developers to make informed design and refactoring decisions before inefficiencies propagate. However, assessing energy usage without repeated profiling and direct measurement is difficult, which limits early reasoning in practice. This study investigates the limits of method-level energy prediction in Java, examining whether static source code metrics complemented with method-level execution time can estimate the energy consumption of Java methods. We profile 2,786 Java methods to extract 33 static features and measure execution time and energy, then train and compare eleven regression models. Our findings show that static source code metrics alone yield poor predictive performance, with average R2 values close to zero. Incorporating execution time as a lightweight dynamic input significantly improves accuracy, raising R2 to as high as 0.46. Execution time, internal method calls, and cyclomatic complexity consistently emerge as the strongest predictors of energy consumption. 

---
# x-Prediction Is All You Need:Training-Free Accelerated Generation via Endpoint Decodability 

**Authors**: Xin Peng, Ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.06114)  

**Abstract**: Diffusion and flow matching models generate high-quality samples, but their ODE samplers often need tens to hundreds of neural function evaluations (NFEs). This remains a practical challenge for released checkpoints, since many accelerators require additional design choices and training cost through retraining, distillation, or trajectory redesign. We investigate a different route based on $x$-prediction. During sampling, standard affine probability paths already expose $x_0$ information: an intermediate state and its path velocity determine a principled estimate of the clean sample. We formalize this property as \textbf{endpoint decodability} and show that the decoder is the minimum-MSE estimator $\mathbb{E}[x_0\mid x_t]$ under the usual $\ell_2$ objective. This yields \textbf{Truncated Jump Sampling} (TJS): stop the ODE at an early-exit time $t^*$ and return the decoded $x_0$. TJS requires no retraining, distillation, or architecture change. Across SDXL, SD3.5M, Z-Image-Turbo, and three class-conditional benchmarks, it reduces NFEs by 20--70\% with near-matched quality. The analysis also shows why endpoint prediction can work without straightening the trajectory, providing inference acceleration without trajectory redesign. 

---
# LLM-Guided Measurement Credibility Correction for Trustworthy Industrial Process Inference 

**Authors**: Youcheng Zong, Runda Jia, Dakuo He  

**Link**: [PDF](https://arxiv.org/pdf/2607.06111)  

**Abstract**: Industrial prediction and soft sensing depend on credible input measurements. In field deployment, a predictor may receive biased, delayed, stale, or derived measurements that still look plausible. Prediction can then fail before the forecasting backbone becomes the main limitation, because the input window no longer represents the real process. Sensor reconstruction, data reconciliation, and fault-tolerant soft sensing reduce this risk, but they often rely on numerical correlation, alarms, fault labels, or explicit process equations. These assumptions are not always available. A correlated variable can also be an unsafe reference when variables share instruments, derived formulas, soft-sensing chains, or control actions. The key issue is to decide before prediction which external measurements can credibly support the current measurement. To address this issue, this article proposes LLM-Guided Measurement Credibility Correction (MCC). MCC converts measurement meanings in process documents into measurement semantics usable by numerical models. It builds independent process references from semantically qualified external measurements and corrects local measurement conflicts before prediction. The predictor therefore receives a more credible input window. Across multiple complex industrial forecasting and soft-sensing tasks, +MCC achieves average relative MAE reductions of 30.7% on real-test protocols and 80.3% on controlled-corruption protocols. It adds only 0.5--2.0k online parameters, with the slowest +MCC inference time at 0.089 ms/step. These results show that measurement semantics can turn process documents into lightweight pre-inference credibility correction and improve prediction accuracy. 

---
# RoME: Robust Mixture of Low-Rank Experts against Multiple Adversarial Perturbations 

**Authors**: Woo Jae Kim, Kyle Min, Suhyeon Ha, Joonsung Jeon, Sung-eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2607.06109)  

**Abstract**: Multi-perturbation adversarial training (MAT) aims to achieve robustness against multiple $\ell_p$ perturbations but suffers from robustness trade-offs between different threats. To address this, we employ a mixture of experts (MoE) to route different threats through distinct model pathways. However, naive application of MoE encounters two critical challenges: experts tend to overlook threat-specific features and redundantly capture features shared across threats, and gating networks suffer from threat-agnostic routing where they learn nearly identical routing patterns across threats, thus preventing the construction of threat-specific model pathways. To this end, we propose Robust Mixture of Low-Rank Experts (RoME), where each expert is a low-rank additive update to the shared backbone, allowing it to capture threat-common features while experts focus on threat-specific information. To address threat-agnostic routing, RoME introduces (i) dual-scale gating that exploits threat-discriminative signals from local and global level features, and (ii) threat-guided gating diversification that enforces diverse expert utilization across threats. Extensive experiments demonstrate that RoME outperforms existing state-of-the-art MAT in union robustness and natural accuracy and improves robustness against unseen threats. Codes are available at this https URL. 

---
# EcoVision: AI-Powered Drone Imaging for Salt Marsh Vegetation Monitoring and Dominance Mapping 

**Authors**: Innocent Onyenonachi, Peter J. Lawerance, Nadia Kanwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.06105)  

**Abstract**: High-resolution RGB imagery acquired from low-altitude UAV surveys was processed through a modular pipeline incorporating transformer-based semantic segmentation, connected-component vegetation extraction, fine-grained species classification using a ConvNeXt architecture, and grid-based dominance scoring at 2x2m resolution. The framework targeted two ecologically significant halophytic grasses, Spartina maritima and Puccinellia maritima, and was trained using a curated and manually annotated UAV imagery, along with biodiversity imagery sourced from publicly accessible datasets. In order to identify these plants from the imagery, our segmentation yielded reliable species masks (mean IoU = 0.56; pixel-level accuracy = 0.96), while object-level classification achieved very good discrimination (F1 = 0.99). Dominance estimates closely matched quadrat-based field surveys, with mean absolute differences below 8%, preserving fine-scale spatial structure under realistic survey conditions. The developed system, named EcoVision, establishes a practical foundation for scalable, high-resolution salt marsh monitoring, demonstrating how AI-driven workflows can translate pixel-level predictions into ecologically interpretable metrics. 

---
# Agents That Teach: Towards Designing Incidental Learning Back into AI-Assisted Software Development 

**Authors**: Rohit Mehra, Samdyuti Suri, Prithviraj K Tagadinamani, Kapil Singi, Vikrant Kaulgud, Adam P. Burden  

**Link**: [PDF](https://arxiv.org/pdf/2607.06101)  

**Abstract**: AI coding agents are rapidly reshaping how software is built, with developers increasingly delegating substantial coding tasks to autonomous agents in pursuit of higher productivity. While these gains are real, they come at the cost of incidental learning. Developers historically acquired informal knowledge through effortful problem-solving, and this has long shaped how software engineering expertise develops. However, with over-reliance on agentic coding, unpracticed skills could atrophy silently over time. As this learning pathway is short-circuited, developers risk silently accruing Knowledge Debt, a developer-level analogue of Technical Debt, where changes the agent executes that the developer cannot fully understand accrue over time. In this paper, we argue that incidental learning will not re-emerge on its own and must be consciously designed back into developer-agent interactions, and propose six design principles to guide such systems. We then present "SHIELD", a multi-agent system grounded in the notion of "agents that teach", that operationalizes these principles by leveraging the AI coding agent's own reasoning to surface contextual, out-of-band learning moments without disrupting developer flow. Through this work, we envision a path toward learning-aware development environments where productivity and learning are complementary, not competing. 

---
# PVCap: Towards Accurate 3D Dense Captioning via PseudoCap and VoxelCapNet 

**Authors**: Xiaopei Wu, Chenshu Hou, Liang Peng, Dan Xu, Binbin Lin, Xiaoshui Huang, Yuenan Hou, Yu Li, Wenxiao Wang, Haifeng Liu, Deng Cai, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06097)  

**Abstract**: 3D dense captioning, an emerging vision-language task, aims to generate descriptive sentences for each object in the 3D scene. Despite the impressive results achieved by previous methods, they suffer from two limitations. First, current research often employs global rigid transformations, such as rotation, to augment scenes without changing their spatial layouts. However, diverse spatial layouts are crucial for training a 3D dense captioning model to describe spatial relations between objects. Second, previous works mainly focus on the design of the caption generation pipeline while utilizing a simple network architecture for other components, i.e., backbone and detection head, which is crucial for extracting rich semantic information for captioning. In this paper, we propose PVCap to alleviate the aforementioned problems. Our PVCap consists of PseudoCap and VoxelCapNet. Specifically, PseudoCap employs a random mixing technique on instances within the dataset, generating numerous pseudo frames with diverse spatial layouts at the instance level. By utilizing a teacher-student framework, PseudoCap obtains pseudo caption labels for these pseudo frames. This data augmentation approach significantly increases the number of training samples and enhances the model's ability to describe the environment effectively. Regarding VoxelCapNet, we introduce a robust caption network that utilizes voxel features and adapts the caption head to the voxel-based network architecture. Our VoxelCapNet can serve as a competitive baseline for future research on 3D dense captioning. Extensive experiments are conducted on two prevalent benchmarks, i.e., ScanRefer and Nr3D. Notably, our method surpasses current state-of-the-art by 11.41% and 13.99% in CIDEr@0.5IoU, respectively. Codes will be made publicly available. 

---
# From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations 

**Authors**: Shiyi Ling, Zhi Zheng, Hui Zheng, Wenjun Xue, Feng Ye, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06080)  

**Abstract**: Putnam's Social Capital Theory is a foundational framework for collective action and community prosperity. However, traditional empirical methods face practical limits on control and replication. Meanwhile, LLM-based social simulations are typically behavior-driven and lack theory-aligned environments for modeling Putnam's core propositions. To address these gaps, we introduce SocaSim, an LLM-based multi-agent simulation framework to study Putnam's Social Capital Theory from theoretical blueprint to simulated reality. Specifically, we build an environment integrating social network evolution, trust dynamics, and norm propagation, where agents engage in repeated collective-action experiments, and then apply the three dimensions to analyze adaptation challenges in smart elderly care. Our simulations reproduce Putnam's macro-level patterns and exhibit strong human-agent alignment at the group level. Unlike traditional methods, SocaSim traces micro-level causal pathways of social network, trust, and norms via round-by-round simulations and counterfactual interventions, enabling process-level interpretability. Taken together, these capabilities establish a research paradigm that leverages LLM agents to bridge social science and computer science. 

---
# Prompt Coach: An Empirical Evaluation of an Agentic Tutor for Learning Prompt Engineering in Software Development 

**Authors**: Rohit Mehra, Kapil Singi, Vikrant Kaulgud, Vibhu Saujanya Sharma, Swapnajeet Gon Choudhury, Swati Sharma, Adam P. Burden, Majd Sakr  

**Link**: [PDF](https://arxiv.org/pdf/2607.06074)  

**Abstract**: Prompt engineering has emerged as a critical yet undertaught skill for software developers, one that traditional learning approaches are ill-equipped to support given its evolving, interactive, and context-dependent nature. In this paper, we introduce Prompt Coach (PC), an agentic tutor that helps developers learn how to craft high-quality code-generation prompts through Socratic guidance embedded in-flow within their IDE. PC evaluates prompt quality across multiple dimensions and surfaces targeted questions to guide self-correction, grounded in the developer's codebase and the behavior of the target LLM. We present an early empirical study with 15 professional developers combining quantitative prompt quality scoring with qualitative perception measures. Participants showed statistically significant improvements after a single 60-minute session, with the largest gains across dimensions commonly overlooked by developers. They also reported strong trust, high adoption readiness, and unanimous agreement that PC improved their prompt-writing skills. 

---
# PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource Languages 

**Authors**: Daryna Dementieva, Nikolay Babakov, Kathy Hämmerl, Ilseyar Alimova, Jindřich Libovický, Shu Okabe, Miras Baisbay, Lukas Edman, Abrorkhon Inomkhujaev, Antonia Karamolegkou, Mateusz Lango, Volkan Özer, Nikola Selic, Subhankar Swain, Tsedeniya Kinfe Temesgen, Galit Bary Weisberg, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2607.05992)  

**Abstract**: Mathematical reasoning has become a central task for evaluating and tuning reasoning Large Language Models (LLMs), yet existing benchmarks remain heavily biased toward high-resource languages, with English and Chinese dominating both pre-training corpora and evaluation suites. The recently released PolyMath (Wang et al., 2025) dataset represents a significant step forward, yet its coverage is still limited to 18 only high-resource languages. To address this gap, we introduce PluraMath, an extension of PolyMath to 18 additional {underrepresented languages spanning 6 language families -- ranging from mid-resource to extreme low-resource settings. We constructed the dataset through a human-curated pipeline, where native speakers thoroughly validated pre-computed translations. Using PluraMath, we then benchmark 27 reasoning LLMs across four model scales -- small, mid-size, large, and closed-source ensembles -- probing the multilingual mathematical reasoning capabilities of state-of-the-art models under diverse linguistic conditions. Our fine-grained analysis confirms a persistent gap in mathematical reasoning performance between high-resource and underrepresented languages, with stronger results largely associated with better instruction-following ability. We fully open-source our dataset, data acquisition pipeline, and evaluation framework, with the goal of lowering the barrier to multilingual benchmark development for underrepresented communities. 

---
# Propose and Attend: Training-free MLLM Grounding Confidence via Multi-Token Localized Attention 

**Authors**: Daniel Shalam, Emanuel Ben Baruch, Avi Ben Cohen, Tal Remez  

**Link**: [PDF](https://arxiv.org/pdf/2607.05978)  

**Abstract**: Multimodal large language models can emit localized predictions, bounding boxes for objects and temporal windows for video and audio events, but they hallucinate these regions prolifically. The model's own token log-probabilities are nearly uninformative: they conflate grounding quality with input ambiguity, and coordinate tokens become near-deterministic once the model commits. We propose Multi-Token Localized Attention (MTLA): a training-free, post-hoc score that measures how strongly a prediction's tokens attend to the region they claim. Prior attention-based detectors, which sum attention over the entire input modality and read a single response token, are weaker special cases; we show that summing only within the claimed region and aggregating across all prediction tokens recovers a stronger grounding signal. The same recipe applies almost trivially to other modalities and tasks: object detection in images and temporal localization in video and audio. Across multiple MLLM families and three modalities, MTLA improves hallucination AUROC by +7 to +38 over the best prior training-free baseline. Used as a confidence score for re-ranking, it nearly doubles the zero-shot COCO detection AP of an open-source 8B generalist (from 20.4 to 37.0), narrowing the gap to supervised detectors without any task-specific training. 

---
# MCP-Enabled Agentic AI for Autonomous IPoDWDM Network Lifecycle Automation 

**Authors**: Chunmin Xia, Jakub Harbaczewski, Nikhil Dsilva, Julie Raulin, Dominic Schneider, Achim Autenrieth  

**Link**: [PDF](https://arxiv.org/pdf/2607.05975)  

**Abstract**: This demo presents an MCP-enabled agentic AI architecture for autonomous control of vendor-agnostic IPoDWDM networks. We demonstrate live end-to-end lifecycle multi-layer automation and closed-loop control using GNPy and telemetry, validated on a real testbed. 

---
# Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search 

**Authors**: Riccardo Terrenzi, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.05970)  

**Abstract**: Dataset search depends heavily on metadata, making LLM-generated metadata a consequential form of synthetic content in retrieval systems. We study six metadata-generation settings for RDF datasets, ranging from simple rewriting to profile-grounded and agentic graph-based generation, and evaluate them jointly for retrieval effectiveness and faithfulness. Unconstrained metadata rewriting delivers the strongest retrieval gains over the original metadata, but it is also the least faithful, showing that search improvements can be driven by unsupported semantic expansion. More grounded settings substantially improve faithfulness, and profile-grounded rewriting provides the most balanced trade-off between retrieval effectiveness and grounding. These findings position synthetic metadata as a system-level IR problem in which effectiveness, provenance, and trust must be evaluated together. 

---
# InfluMatch: Frontier-Quality KOL Search at 4B-Model Cost 

**Authors**: Krittanon Kaewtawee, Petmongkon Pornpichitsuwan, Natchaya Temyingyong, Nutnicha Laplamoon, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2607.05968)  

**Abstract**: Matching influencers (KOLs) to free-form, multi-part Thai marketing criteria is today served either by keyword search over structured profiles, which misses semantic fit, or by prompting frontier LLMs over every candidate, which is accurate but slow and expensive. We present InfluMatch, a low-cost three-stage cascade -- retrieval $\rightarrow$ rerank $\rightarrow$ reason -- built entirely from small open-weight models: dense retrieval returns 50 candidates, a 4B pointwise reranker scores each by the log-probability of a single Yes token and keeps 10, and a 4B reasoner grades the shortlist per criterion on a rubric with a Thai rationale. The cascade is designed for cost: reasoning over a filtered top-10 halves token spend versus reasoning over all 50 while scoring 14 points higher. End-to-end against human relevance labels on an 11-query set with all 50 candidates labeled, the full cascade reaches 94.1% P@5, versus a retrieval-only baseline near random; it matches the frontier model Kimi-K2.6 (91.8%) while emitting ${\sim}35\times$ fewer output tokens and serving a 50-KOL query in ${\sim}20$ s on one A100. Notably, the only fine-tuning that pays off is pairwise: a SimPO-tuned reranker matches the frontier baseline's best-pick accuracy (78.0 EM), whereas fine-tuning the reasoner on pointwise per-criterion labels improves offline scores yet degrades end-to-end ranking -- an inversion we trace to the design of the absolute labeling task -- leaving the untuned base model as the strongest deployed reasoner. The result is a deployable, explainable KOL search system at a small fraction of frontier serving cost. 

---
# Decoupled Single-Mask Annotation Noise Detection via Cross-Sectional Patch Self-Consistency 

**Authors**: Yinheng Zhu, Xiaowei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05965)  

**Abstract**: Vascular computed tomography datasets are commonly annotated only once per scan, yielding the pervasive yet under addressed problem of single mask annotation noise. Existing solutions either require costly multirater fusion or are coupled with network training, preventing explicit auditing of where and why labels fail. We introduce a decoupled framework for single-mask annotation noise detection that leverages cross-sectional patch self-consistency to produce interpretable and auditable noise evidence. Tubular anatomy exhibits strong cross-sectional recurrence: patches extracted orthogonally along vessel centrelines recur in appearance across locations and subjects. Thus, anatomically similar patches should have consistent masks, and disagreement signals unreliable annotation. Our method samples cross-sectional patches, retrieves intensity-equivalent neighbours via scalable vector search, and computes a patch-level noise score from statistical mask disagreement, yielding explicit image-mask evidence for every flagged region. Aggregating scores produces scan-level quality maps for dataset quality assessment or quality-weighted training. Experiments on the coronary CT dataset validate the detected noise for improving training robustness and reveal systematic annotation biases. Specifically, transverse and oblique vessels exhibit 5.1 times higher error rates than axis-aligned structures, with additional correlations to cross-sectional area and intensity. Code is available here. 

---
# Agentic AI for IPoDWDM Network Lifecycle Automation: An MCP-Enabled Architecture 

**Authors**: Chunmin Xia, Jakub Harbaczewski, Nikhil Dsilva, Julie Raulin, Dominic Schneider, Achim Autenrieth  

**Link**: [PDF](https://arxiv.org/pdf/2607.05958)  

**Abstract**: We present a distributed, vendor-agnostic multi-MCP architecture for SDN-based automation and autonomous control of multi-vendor, multi-layer IPoDWDM networks. The framework enables E2E service lifecycle automation, closed-loop cross-layer control using GNPy model and optical telemetry, and is experimentally validated on a IPoDWDM testbed. 

---
# NegROI: Click-Centric Uncertainty-Guided Refinement with Scene-Conditioned Negative Prompts for Robust Interactive 3D Segmentation 

**Authors**: Shuheng Zhang, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05955)  

**Abstract**: Interactive 3D segmentation aims to extract object masks in point clouds with minimal user clicks. Despite recent progress, most existing approaches still struggle with (i) coarse voxel resolution that blurs fine boundaries under limited clicks and (ii) hard false positives caused by confusing background structures. These issues are exacerbated by density and scale shifts across datasets (e.g., dense RGB-D reconstructions vs. sparse LiDAR scans), where fixed refinement heuristics and purely click-driven decoding generalize poorly. To address them, we propose NegROI -- a novel transformer-based interactive framework that couples click-centric multi-resolution refinement with scene-conditioned negative prompts. Given a coarse voxel prediction, it refines only a local Region Of Interest (ROI) around the current click on a finer grid and fuses refined logits back to the coarse mask. To improve robustness and efficiency, we introduce uncertainty-driven selective refinement that prioritizes ambiguous regions. Meanwhile, we model hard background patterns via a set of scene-conditioned negative prompts obtained by cross-attention over scene tokens. We further stabilize these prompts with a diversity regularizer. Finally, we propose boundary-aware hard negative mining to supervise negative-prompt attention toward boundary-proximal, high-confidence false positives. Our experiments on common benchmark datasets (i.e., ScanNet, S3DIS, and KITTI) demonstrate improved click efficiency and reduced false positives, with stronger cross-dataset robustness than the state-of-the-art baselines. 

---
# Signed-Graph Recommendation as Structural Consistency Maximization 

**Authors**: Zifan Wang, Siyu Chen, Wenzhuo Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.05952)  

**Abstract**: While signed social recommendation has shown great potential by modeling both trust and distrust relations, its effectiveness is often hindered by structural noise and data sparsity. In this work, we first identify a fundamental inconsistency across the structural, propagation, and semantic layers of existing models, which leads to biased representations learned from sparse or noisy datasets. Furthermore, we observe that most existing methods treat the observed graph as fixed, failing to bridge the gap between noisy topologies and reliable social semantics. To address these issues, we propose a unified framework named SSC-Loop that treats signed social recommendation as the maximization of structural consistency. SSC-Loop includes three dedicated modules: ESA-DA for structural consistency, a P/N/O propagation mechanism for propagation consistency, and a contrastive learning objective for semantic consistency. Experiments on Epinions demonstrate that SSC-Loop achieves strong performance on explicit signed social rating prediction, while auxiliary results on Slashdot under a derived link-existence setting further suggest its ability to exploit signed social structures. Source code is available at this https URL. 

---
# CMDR: Contextual Multimodal Document Retrieval 

**Authors**: Ryota Tanaka, Taku Hasegawa, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2607.05927)  

**Abstract**: Multimodal document retrieval aims to retrieve relevant pages while preserving both textual and visual content from the original document. However, existing benchmarks primarily evaluate simple lexical or semantic matching, and most methods encode pages independently. Consequently, they overlook the contextual information in the document required to resolve queries that aggregate information across multiple pages. In this paper, we introduce CMDR and CMDR-Bench, a new multimodal document retrieval task and benchmark that require modeling document context. To address this challenge, we propose CMDR-Embed, a contextual multimodal embedding framework that explicitly incorporates document context by jointly encoding multiple pages and deriving page-level embeddings from a shared contextual representation. Furthermore, we introduce CMCL, a contextual multimodal contrastive learning objective that effectively trains CMDR-Embed by balancing contextual modeling with page-level discriminability. Experiments demonstrate that CMDR-Embed significantly outperforms non-contextual embeddings, highlighting the importance of context-aware multimodal embeddings for advancing document retrieval. 

---
# PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails 

**Authors**: Mingyang Song, Luxin Xu, Haoyu Sun, Minzhou Pan, Yu Cheng, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.05910)  

**Abstract**: Image guardrails are typically trained and evaluated under a fixed safety policy, implicitly treating safety as an intrinsic property of an image. Real deployments are different: the same image may be allowed in one product, restricted in another, and newly disallowed when a policy boundary changes. We study policy-adaptive image guardrailing, where a model must decide whether an image violates the currently supplied policy and generalize to held-out policy definitions. We introduce PolicyShiftBench, a comprehensive benchmark with 2,000 policy-discriminative instances over 265 images, where each image is paired with 7.55 policy-conditioned prompts on average to test whether models adapt to the active policy rather than relying on image-level safety priors. We then propose PolicyShiftGuard, a compact policy-conditioned guardrail trained with a two-stage training recipe that combines Randomized Policy SFT (RP-SFT) with Boundary-Pair Policy Adaptation (BP-Adapt). BP-Adapt trains matched prompts for the same image and risk category using standard label supervision and a pairwise comparison loss that separates blocking policies from passing policies. Experiments show that existing VLMs and specialized guardrails remain brittle under policy shifts, while PolicyShiftGuard substantially improves policy-sensitive performance. The 7B model achieves SOTA performance of 76.9 Avg. F1 and 72.1 Avg. PSS on PolicyShiftBench, transfers well to UnSafeBench and SafeEditBench, and improves the latency-performance trade-off with a concise output format. Ablations confirm that matched pass/block boundary pairs are essential for stable policy adaptation. 

---
# K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm : (Compensated Loss-Based Sample Exclusion with Unbiased Gradient Estimation) 

**Authors**: Jean-Francois Bonbhel  

**Link**: [PDF](https://arxiv.org/pdf/2607.05903)  

**Abstract**: We present K-ABENA (K-Adaptive Backpropagation with Error-based N-exclusion Algorithm), a selective gradient computation framework that reduces per-iteration training cost by excluding a fraction of low-loss ("minor") observations from the backward pass. Its canonical form (v3) combines a defensive-mixture sampling design over the minor set with Horvitz-Thompson inverse-probability reweighting, yielding a design-unbiased Horvitz-Thompson gradient estimator (Lemma 2) and whose self-normalized practical variant carries a bias of order O(1/m) with an explicit constant (Lemma 3). We prove an O(1/sqrt(T)) non-convex convergence guarantee for SGD under the estimator, with an additive term that quantifies the residual bias (Theorem 1). We further prove that uncompensated loss-based selection - a family that includes OHEM, SBP, and the two earlier K-ABENA variants - admits no stationary point at any minimizer where its selection bias is bounded away from zero (Proposition 2), and we quantify this failure empirically: at 0.17% class imbalance, uncompensated variants reach test AUC 0.53-0.62 versus 0.9998 for full-batch SGD, while the compensated estimator attains 0.9991 at identical 28.4% compute savings. On real datasets (Breast Cancer, Digits, Wine, Diabetes) the compensated estimator is statistically indistinguishable from full-batch SGD (paired permutation tests, p >= 0.5; Section 7) while saving 28-54% of per-epoch gradient computation. A biased "regularized mode" (the earlier half-domain variant) is retained as an option with a proven exact bias decomposition (Lemma 5) and quantified contraindications: it collapses to 0.386 accuracy under 40% label noise (baseline: 0.832) and to 0.53 AUC under extreme imbalance. Every advantage and every limitation reported in this paper is either proved or measured; all experiments are CPU-scale (NumPy/scikit-learn) and their scope is stated explicitly. 

---
# From Textural Counterpoint to Feature Encoding: A Multi-Dimensional Machine Representation Study of Haydn's "The Lark" Integrating Electroacoustic Analysis 

**Authors**: Yakun Liu, Zhiyu Jin, Hai Luan, Dong Liu, Xiaonan Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.05902)  

**Abstract**: Chamber music, as a highly precise multi-part interactive system, contains a logic of "role assignment and dynamic interaction" that provides an extremely valuable blueprint for exploring human-computer collaborative composition paradigms. Addressing the lack of role perception capabilities in existing deep music generation models during polyphonic interactions, this paper conducts an interdisciplinary analysis of Haydn's String Quartet in D Major, The Lark (Op. 64, No. 5). We propose a novel research path: "Classical Morphology Qualitative Analysis-Electroacoustic Quantitative Measurement-Machine Representation Reconstruction." The study first utilizes auditory analysis to dissect the counterpoint morphology of the leading voice and the underlying groove in the first movement. Subsequently, it introduces spectrum and dynamic feature analysis tools from a Digital Audio Workstation (DAW) to translate subjective auditory perception into objective, measurable physical parameters. Building on this, the paper introduces a fundamentally new approach to low-level computer feature extraction: completely abandoning the traditional mechanical quantization grid, introducing Event-based Timestamps to record the duration of micro-timing, and transforming acoustic features into an independent "Role-Aware Encoding" as an aesthetic heuristic mechanism (a phenomenological anchor). This study not only completes the logical loop spanning classical analysis, electronic music mapping, and AI symbolic generation but also establishes a profound theoretical foundation-from the perspectives of interactive aesthetics and media philosophy-for constructing human-computer collaborative music systems imbued with "social attributes" and "otherness awareness." 

---
# Few-Medoids: An Embarrassingly Simple Coreset Selection Method for Few-Shot Knowledge Distillation 

**Authors**: Cemil-Andrei Dilmac, Florinel-Alin Croitoru, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05891)  

**Abstract**: Coreset selection aims to identify a small and highly representative subset of a massive dataset for efficient model training. The problem remains challenging even in the few-shot knowledge distillation (KD) setup, where a full-scale pre-trained teacher informs the student network. Typical sample selection strategies often struggle to surpass the random selection baseline. In this paper, we showcase few-medoids, an embarrassingly simple coreset selection strategy that chooses the samples closest to the centroid (average image) of each class. We present extensive KD experiments on four datasets, covering a wide range of image classification problems, and three teacher-student model pairs, comprising both convolutional and transformer networks. Although the proposed method is embarrassingly simple, our empirical results indicate that few-medoids is able to consistently surpass the random selection baseline, as well as the other coreset selection strategies. We therefore consider that few-medoids can be used as a drop-in replacement for commonly-used baselines (e.g. herding or k-center Greedy), in future research on coreset selection. To reproduce the reported results, we publicly release our code at this https URL. 

---
# i-EXAM: Instructable and Explainable Attack Connectivity Graph Modeler 

**Authors**: Rakesh Podder, Wadia Ganim, Sarath Sreedharan, Indrajit Ray, Indrakshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2607.05888)  

**Abstract**: i-EXAM is a planning-powered tool that helps system administrators to create security profiles of complex networks and perform what-if analyses to identify network hardening strategies. It leverages planning compilation that provides soundness and completeness guarantees to identify attack paths, evaluate security metrics, generate diverse hardening strategies, and explain these strategies in natural language using Large Language Models. 

---
# Harrison.Rad 1.5 Technical Report: A radiology foundation model that can draft reports from images, priors and clinical context 

**Authors**: Suneeta Mall, Vladimir Nekrasov, Ashnil Kumar, Sajith Karunasena, Aiden Nibali, Alix Bird, Mateo Diaz Shine, Jarrel Seah  

**Link**: [PDF](https://arxiv.org/pdf/2607.05880)  

**Abstract**: Imaging demand is growing faster than the radiology workforce can expand, and reporting backlogs cannot be resolved through training and recruitment alone. The most direct opportunity is reducing the time and effort radiologists spend producing reports, a task that requires interpreting images, integrating clinical history and prior studies, and drafting structured findings. We present this http URL 1.5 (HR1.5), a radiology-specific multimodal large language model that accepts interleaved text and visual inputs and generates structured and unstructured text across plain-film radiology, spanning computed radiography, chest, musculoskeletal, abdominal, spine, and pelvic x-rays, and mammography. HR1.5 is trained through a three-stage pipeline: domain adaptation of a base language model on radiology reports, contrastive vision-encoder training with curriculum-based hard negatives on ~6 million image-report instances, and visual-question-answering fine-tuning on multi-turn conversations. We evaluate it with a Findings-Diagnosis scoring framework that extends RadGraph-XL entity extraction with ontology-based synonym matching and polarity-contradiction detection, benchmarked on RadBench, a simulated FRCR 2B Short Case examination scored against Angoff-method thresholds, ReXGradient, and internal multi-modality datasets. HR1.5 is the only system evaluated to meet the simulated FRCR passing standard and achieves the highest accuracy on closed-format clinical questions, across anatomical regions, on internal multi-body-part and mammography reporting, and on the primary clinically-aligned score for public chest reporting. We further examine explainability and model behaviour, including question-sensitive Grad-CAM heatmaps, attention analysis, and confidence estimation, to support responsible future evaluation toward clinical use, and a framework for clinically grounded assessment of report quality. 

---
# Think Before You Grid-Search: Floor-First Triage for LLM Serving 

**Authors**: Yihua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05876)  

**Abstract**: LLM serving optimization typically benchmarks many configurations and reaches for heavy profilers when latency targets are missed. We argue for the reverse discipline: estimation is the analytical layer of profiling -- without it, optimization degenerates to grid search. Floor First is a residual-driven triage workflow. Each decode step is modeled as a five-dimensional resource vector (HBM bytes, FLOPs, network bytes, network messages, KV capacity); summing within a resource and maximizing across resources gives an optimistic floor, the plain sum a pessimistic one. Where a measurement lands inside this [max, sum] interval reads out overlap quality before any profiler is opened, and profilers escalate only on residuals above a stated threshold. Deployment alternatives are compared by wall ordering -- which resource wall binds first as load grows -- rather than by point benchmarks. The account is compositional: new attention or state-space variants enter by declaring one module, and the workflow ships as a zero-dependency calculator plus an agent skill that enforces the discipline in agentic optimization loops.
As a case study we analyze a DeepSeek-V3.2-style 671B MoE/MLA model on 16 NVIDIA H20 GPUs, whose ridge point of ~74 FLOP/byte (vs ~590 for H100) makes it an extreme decode-oriented part. The floors show TP16 decoding is KV-capacity-limited to ~70 concurrent 8K requests; sparse attention removes the KV-bandwidth term but not the capacity wall; an EP16+DP-attention layout accepts slightly worse same-batch weight traffic for an order-of-magnitude higher capacity wall (~644) -- while single-stream latency favors TP by 2.4x. The layout judgment is thus a computable function of the operating point, explaining why production deployments on identical hardware have shipped opposite attention layouts. 

---
# Differentially Private Natural Gradient Descent 

**Authors**: Pan Li, Kai Chen, Shuai Chang, Shengzhi Zhang, Peizhuo Lv, Jinwen He  

**Link**: [PDF](https://arxiv.org/pdf/2607.05866)  

**Abstract**: Under a fixed privacy budget, the utility of differentially private (DP) training is ultimately determined by its optimization efficiency. Standard first-order DP optimizers such as DP-SGD rely solely on local gradients and ignore the underlying loss curvature. This geometric blindness causes severe zigzagging in ill-conditioned landscapes, squandering precious privacy budgets on inefficient iterations. Practitioners are thus trapped in a bind: either stop training prematurely or inject massive per-step noise, both of which critically compromise final model utility. Natural Gradient Descent (NGD) resolves this by preconditioning gradients with curvature, aligning updates with the loss geometry and extracting more efficient signal from every noisy step, offering a principled pathway to break the privacy-utility bottleneck.
Despite its theoretical appeal, directly integrating NGD with DP introduces fundamental challenges: curvature estimation itself consumes prohibitive privacy budgets, isotropic DP operations conflict with the anisotropic scaling of NGD, and the inverse curvature catastrophically amplify parameter updates in flat directions, causing training instability. We propose DP-NGD, a practical framework that systematically addresses these obstacles by decoupling curvature estimation from private data, reconciling isotropic DP constraints with anisotropic second-order optimization via a whitened-space mechanism, and dynamically clamping the curvature to stabilize training. Extensive experiments on standard benchmarks demonstrate that DP-NGD achieves state-of-the-art accuracy, breaking through the utility ceilings of first-order baselines while delivering up to a $10\times$ convergence speedup under the same privacy budget. 

---
# Unsupervised Anomaly Detection of Information Operations Users via Behavioral and Language Patterns 

**Authors**: Sishun Liu, Sajal Halder, Ke Deng, Yan Wang, Xiuzhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05855)  

**Abstract**: Information Operations on social media networks have been identified as a significant threat to democracy and modern society, but they are challenging and expensive to detect by humans. Existing supervised IO detection methods fail to capture the dynamic nature of evolving IO user behavior, while existing unsupervised approaches rely on oversimplified assumptions of coordination among IO users that may not exist in practice. To overcome the limitations of existing methods, we formulate IO user detection as an anomaly detection problem and propose a novel unsupervised IO user detection approach called Temporal-bEhavior-laNguage Signals for information Operation Recognition (TENSOR), which leverages multimodal data, including temporal online user behavior, such as message posting activities, and the textual content of the messages. The motivation is that IO users are typically a very small fraction of all online users and have unique temporal behavioral and language patterns. Specifically, we train a Temporal Point Process (TPP) to capture abnormal temporal behavioral patterns of IO users because they are known to behave in a coordinated manner for IO campaigns. We further introduce a novel evidence function that converts LLM responses, which are generated from user post timelines, into quantitative scores to adjust the TPP outputs for better IO user detection. Experimental results show that TENSOR outperforms the baselines on five real-world IO datasets. Code is available at this https URL. 

---
# AbICL: In-Context Learning for Antigen-Specific Antibody Affinity Ranking 

**Authors**: Zhiyuan Chen, Jing Hu, Junzhe Wang, Yueyang Huang, Xinyi Yang, Zhaoyang Wang, Feng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05846)  

**Abstract**: Accurate ranking of antibody candidates according to their binding affinity is essential for therapeutic antibody discovery. However, existing methods treat affinity comparisons independently and ignore the contextual information encoded in other labeled comparisons, limiting their ability to capture antigen-specific binding landscapes. For many target antigens, a small number of experimentally characterized affinity comparisons are often available. An important question is whether the model can exploit these existing comparisons to infer antigen-specific ranking patterns that facilitate subsequent affinity ranking. This form of learning from labeled demonstrations closely resembles the paradigm of In-Context Learning, motivating us to revisit antibody affinity ranking from an ICL perspective. To this end, we propose AbICL, an ICL framework for antigen-specific antibody affinity ranking. AbICL combines a pretrained structural encoder with a context ranking head and is trained with an episodic meta-training strategy that enables the model to leverage support demonstrations for test-time adaptation without gradient updates. Experiments on the AbRank benchmark demonstrate that AbICL consistently outperforms existing ranking baselines across almost all data splits and evaluation benchmarks. Further analysis shows that the value of contextual demonstrations depends on how well they match the target inference task, and becomes increasingly pronounced under distribution shift and fine-grained affinity discrimination. These findings highlight the potential of ICL as an effective paradigm for antigen-specific antibody affinity ranking, particularly in challenging settings where a single global ranking function is insufficient. 

---
# Beyond Refusal: A Same-Lineage Study of Aligned and Abliterated LLMs for Vulnerability Analysis 

**Authors**: Mingchen Li, Meikang Qiu, Zifan Peng, Heng Fan, Song Fu, Junhua Ding, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.05842)  

**Abstract**: Large language model (LLM)-assisted software security operates at a difficult boundary: the vulnerability-analysis terminology needed for legitimate code review, triage, and repair can closely resemble terminology associated with misuse. Existing safety and cybersecurity evaluations are difficult to interpret in this setting because they often compare unrelated model families, thereby conflating safety behavior with differences in architecture, scale, training data, and deployment. To isolate this factor, we study safety state: whether refusal behavior remains intact (Aligned) or has been refusal-ablated (Abliterated) within same-lineage models. We ask how this safety state affects defensive utility across software-security workflows. We compare aligned instruction-tuned models with publicly released refusal-ablated descendants from two model families, Gemma and Qwen. We evaluate Aligned and Abliterated states on vulnerability detection, CWE attribution, vulnerable-line localization, root-cause localization, and executable patch validation. We further treat prompt wording as a controlled framing dimension: prompts begin with neutral code-review language, add authorization context, and vary the density of cybersecurity terminology. In a Gemma-based Java/Vul4J repair-validation study, Abliterated achieves higher early-stage validation rates, with 67.8%, 65.0%, and 32.8% of patches judged usable, successfully applied, and successfully compiled, respectively, compared with 29.9%, 24.9%, and 9.0% for Aligned. In the Qwen pair, Abliterated improves localization performance, increasing line-level F1 from 2.08% to 3.91% and Top-1 accuracy from 4.10% to 6.95%. These findings suggest that evaluations of LLM-based security assistants should jointly measure whether models respond, whether their usable responses are correct, and whether their outputs remain actionable across the engineering workflow. 

---
# VisTCP: A Visualization Framework to Construct Knowledge-Graph-Based Representation for Traditional Chinese Painting 

**Authors**: Zhiguang Zhou, Fengling Zheng, Miaoxin Hu, Lina You, Jin Wen, Huan Liu, Wei Zhang, Dekun Qian, Yuhua Liu, Wei Chen, Yigang Wang, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05841)  

**Abstract**: Structured representation can characterize semantic objects and relationships in images. It provides a possible effective way for the semantic understanding of Traditional Chinese Paintings (TCPs) to better support archaeology and art history research. However, most image-oriented structured representation methods perform poorly on TCPs, due to two major challenges: 1) the objects and events of TCPs exhibit substantial differences from modern natural images, which results in semantic misunderstandings of TCPs; and 2) it is difficult to achieve accurate identification of ancient objects and events in TCPs, even for domain this http URL this paper, we propose VisTCP, a visualization framework that combines a TCP-oriented intelligent model and expert knowledge, which enables art historians to achieve trustworthy structured representations of TCPs in a human-in-the-loop manner. Firstly, we conduct a pilot study with three domain experts to build a semantic taxonomy of TCPs. Then, expert-annotated data are used to train a TCP-oriented structured representation model, which can automatically extract meaningful objects and their relationships in TCPs. To inform users of the model uncertainty, we design a joint embedding visualization view to show the differences between expert annotations and model predictions. This allows users to refine the structured representation based on their domain knowledge, enabling iterative optimization of the model. Finally, we conduct a case study, a usage scenario, and expert interviews on a real dataset to demonstrate the effectiveness of VisTCP in supporting the structured representation and semantic understanding of TCPs. 

---
# Tangent classes of matroids and wonderful compactifications 

**Authors**: Ronnie Cheng, Shurui Liu, Guoxiong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.05835)  

**Abstract**: For every loopless matroid $M$ and every Feichtner--Yuzvinsky building set $\mathcal{G}$ containing the top flat, we construct an integral tangent class $T_{M,\mathcal{G}}^{\mathbb{Z}}\in K_{\mathbb{Z}}(M,\mathcal{G})$; in the realizable case it specializes to the class of the tangent bundle of the corresponding wonderful compactification, it recovers the Hilbert series of the Chow ring through Hirzebruch--Riemann--Roch, and it satisfies the expected Chern-alpha lower bounds. This reproduces the tangent class and its key properties studied by the first author in arXiv:2606.22650. The main body of this paper was produced autonomously, without human mathematical guidance, by Danus, an AI mathematical reasoning agent. Danus solved the problem before arXiv:2606.22650 was publicly available, demonstrating the potential of AI agents in mathematical research. We reproduce its output faithfully, adding only editorial comments; the experiment is documented in Appendix B. 

---
# Decision-Focused Scenario Generation and Selection for Efficient and Robust Grid Dispatch 

**Authors**: Yangze Zhou, Yihong Zhou, Thomas Morstyn, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05830)  

**Abstract**: The increasing uncertainty from flexible demand and renewable generation has made distributionally robust optimization (DRO) an important tool for robust power system dispatch. DRO relies on forecast scenarios to construct ambiguity sets, but conventional scenario generation pipelines are often trained in an accuracy-oriented manner and may neglect spatial correlations among uncertainties. This mismatch can produce ambiguity sets that are statistically plausible but suboptimal for downstream operation. This work proposes a decision-focused generative framework for correlated scenario generation in DRO-based dispatch. Instead of training generative models solely to fit the historical uncertainty distribution, the proposed framework optimizes generated scenarios according to their induced downstream operational cost. The proposed framework is tailored to mainstream generative models, including variational autoencoders, generative adversarial networks, and diffusion models, while capturing the joint distribution of uncertainties across buses. To improve computational tractability, we further develop a differentiable scenario selector that selects decision-relevant scenarios from a generated pool and can be trained within the same decision-focused pipeline. Case studies demonstrate that the proposed framework effectively reduces 0.80%-2.02% operational cost across different generative models compared to accuracy-oriented methods. 

---
# Complementary Roles of Image Classification and Vessel Segmentation in AI-Based Screening for Retinopathy of Prematurity Plus Disease in a Kenyan Preterm Cohort 

**Authors**: Fred Mutisya, Oscar Onyango, Sarah Sitati, Syokau Ilovi, Aeesha NJ Malik, Brenda W'mosi, Brian Makini, Jalemba Aluuvala, Josiah Onyango, Rachael Kanguha Mmene, Steven Wanyee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05825)  

**Abstract**: Background. Retinopathy of prematurity (ROP) is a preventable cause of childhood blindness, with rising burden in low- and middle-income countries where ROP-trained ophthalmologists are scarce. Plus disease, marked by retinal vessel dilation and tortuosity, triggers treatment but is subjective and variable. Automated screening could extend specialist reach, but African evidence remains limited.
Methods. We analysed 121 Kenyan preterm infants, covering 237 eyes and 1,635 fundus images graded as No Plus, Pre-Plus or Plus. Vessel annotations from two graders supported segmentation training. Eleven configurations were evaluated for eye-level Plus detection using patient-grouped nested cross-validation, including image classifiers, multiple-instance learning, multi-task segmentation-classification, and segment-then-classify pipelines.
Results. Vessel segmentation was feasible, achieving pooled Dice 0.533, IoU 0.368, sensitivity 0.623 and specificity 0.979 on held-out images. RGB classifiers were highly sensitive but over-referred, while segmentation-coupled models were more specific. Combining approaches improved performance: an OR-based screen achieved the highest sensitivity, an AND-based confirmation achieved the highest specificity, and a probability ensemble gave the best balanced performance, with sensitivity 0.692, specificity 0.914 and balanced accuracy 0.803, outperforming the vision classifier alone.
Conclusions. Classification and vessel segmentation are complementary for ROP Plus detection in Kenyan data. Classifiers support sensitive case-finding, while segmentation improves specificity and reduces over-referral. African ROP AI systems should use combined workflows and undergo prospective multi-site validation. 

---
# Segmentation before Answering: Pixel Grounding for MLLM Visual Reasoning 

**Authors**: Yake Wei, Yuan Wang, Fengyun Rao, Jing Lyu, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05798)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have evolved from static perception to interleaved visual-language reasoning, often referred to as ``thinking with images''. A basic operation in this reasoning process is to zoom in on regions of interest (often represented with bounding boxes) to acquire finer visual details. In this paper, we propose \textbf{Seg}mentation before \textbf{Answer}ing (SegAnswer), which shifts the unit of zoom-in from the popular bounding box to pixel-level segmentation mask. By employing fine-grained masks to isolate the target area from cluttered environments, segmented visual input yields a more precise region of interest, effectively filtering out redundant background and interfering objects. Furthermore, the discrete patches of segmented visual input align more seamlessly with how MLLMs structure visual tokens via positional embeddings. In experiments, we evaluate SegAnswer across diverse benchmarks, including high-resolution perception, general perception, and hallucination. It achieves consistent improvements and also exhibits considerable performance on segmentation tasks, validating its capability for reliable pixel grounding. 

---
# FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning 

**Authors**: Chuhao Zhou, Liquan Wang, Shuxin Cao, Xiangyu Chen, Yuxuan Hu, Boyu Ma, Animesh Garg, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05780)  

**Abstract**: While humans readily repurpose a book, a stone, or a shoe to drive a nail, robots trained on specific tools fail to transfer the same function to novel ones -- a gap we formalize as functional generalization. Such tools share a common functional intent that is visually recognizable, yet this perceptual similarity does not carry over to action space, where each tool demands an entirely different motor pattern. To bridge this gap, we explore intermediate representations including affordance images, human video prompts, and 2D keypoint trajectories, finding that keypoint trajectories best balance functional expressiveness and action groundability. Building on this, we propose FunctiOnal Reasoning and Grounded Execution (FORGE), a two-stage policy that decouples functional reasoning from action execution: predicting generalizable keypoint trajectories from action-free data, then grounding them into robot actions with limited demonstrations. On a seven-tool hitting-function benchmark, FORGE consistently outperforms state-of-the-art methods on unseen tools in both simulation and the real world, achieving over 2X improvement in average success rate. 

---
# LEGATO 2: Toward Multimodal Sheet Music Recognition and Understanding 

**Authors**: Guang Yang, Brian Siyuan Zheng, Victoria Ebert, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2607.05769)  

**Abstract**: We propose a novel pipeline, Legato 2, for extracting symbolic notation and semantic knowledge from images of sheet music. Legato 2 features the first large-scale neural model for optical music recognition (OMR) to operate sequentially on a system-by-system basis, following the horizontal lines of notation as they are read on the page, rather than treating the page as an undifferentiated image, enabling better scaling to arbitrarily long inputs. It is also the first OMR model capable of generating symbolic transcriptions that include embedded textual content, such as titles and annotations. The pipeline combines system-level segmentation with an autoregressive vision-LM to capture both local notation details and score structure. Across multiple datasets, Legato 2 consistently outperforms prior state of the art. We also show that symbolic transcriptions complement visual inputs for frontier language models, improving their interpretation of dense musical documents. Legato 2 establishes new state-of-the-art performance in both OMR and downstream sheet music understanding. 

---
# Data-dependent Evaluations for Budgeted Submodular Maximization 

**Authors**: Lejian Zhang, Xueyan Tang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05759)  

**Abstract**: Submodular maximization is an important building block for developing algorithms in many areas such as machine learning and data mining. Due to the NP-hardness of the problem, analysis of submodular maximization algorithms typically provides pessimistic worst-case approximation factors only. It is not easy to evaluate how close a produced solution is to an optimal one for a given problem instance. In this paper, we develop new data-dependent upper bounds for submodular maximization with a knapsack constraint. We theoretically prove that they dominate the optimal solution and empirically demonstrate their advantages in certifying how close to optimal a solution is through experiments with real-world datasets. 

---
# When Should LLMs Search? Counterfactual Supervision for Search Routing 

**Authors**: Minho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.05752)  

**Abstract**: Search-augmented language models can use external evidence to compensate for limitations in parametric knowledge, but search is not uniformly beneficial: models may call search for questions they can already answer, or rely on noisy evidence when correction, clarification, or abstention would be more appropriate. We formulate this as an instance-level search-routing problem: deciding whether search is needed to improve task success relative to a no-search execution. To derive supervision, we compare no-search and forced-search outcomes for the same question and construct an oracle over NO SEARCH, SEARCH, and UNSOLVED based on task-specific success. Using this oracle as both an evaluation criterion and a learning signal, we train search-routing policies with supervised fine-tuning and preference optimization, improving routing macro-F1 on oracle-eligible examples from 0.7082 to 0.8235 for Gemma E2B and from 0.7053 to 0.8365 for Qwen3.5-4B. Further analysis shows that the learned policies reduce model-specific routing failures: Gemma primarily learns no-search restraint, while Qwen further reduces missed search; residual UNSOLVED cases reveal heterogeneous bottlenecks involving model capacity, retrieval budget, evidence use, and policy behavior. 

---
# Unicode TAG-Block Concealment of Tool-Metadata Payloads in the Model Context Protocol: An Approval-View Fidelity Gap Across Three Independent Server Implementations 

**Authors**: Mohammadreza Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05744)  

**Abstract**: The Model Context Protocol (MCP) is the dominant way coding agents discover and invoke external tools. A server advertises each tool through a tools/list handshake that returns a name, a natural-language description, and a JSON input schema. The client renders this metadata once, in a one-time approval dialog, and then injects it verbatim into the model's context on every subsequent turn. Nothing in the protocol requires the rendered approval view and the bytes delivered to the model to match. We isolate that gap as a single structural mechanism, concealment encoding, and show with a model-free, protocol-free analysis that Unicode's TAG block (U+E0000 to U+E007F) has no assigned glyph in any mainstream terminal, chat, or IDE renderer, so a payload written in it is absent from what a human reviewer sees while surviving byte-for-byte into the model's tokenizer. We then measure whether this mechanism actually defeats today's client-side defenses, building a proof-of-concept that speaks the real MCP JSON-RPC/stdio protocol against a genuine client and server. Across 5 distinct MCP metadata surfaces we implement 8 concrete techniques with a deterministic, protocol-level harness. All 8/8 techniques deliver an attacker-controlled payload into the model's context, 4/8 evade a representative string-matching sanitizer, and exactly as the mechanism analysis predicts, only the TAG-block encoding (1/8) is invisible in the human approval view while still reaching the model verbatim. MCP forces re-approval for 0/8 techniques even under a time-of-check to time-of-use rug-pull. To test whether these outcomes are a property of the protocol or an artifact of one server codebase, we re-implement the catalogue against 3 independently developed Python MCP server libraries and find total agreement across all 32 cross-library outcome cells. The baseline sanitizer flags 0 of 25 benign descriptions. 

---
# The Balkanization of Execution-Security Research for AI Coding Agents: Isolation, Access Control, and Time-of-Check-to-Time-of-Use Vulnerabilities 

**Authors**: Mohammadreza Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05743)  

**Abstract**: AI coding agents now read repositories, call tools, and execute shell commands with limited human oversight, and a fast-growing body of work studies whether the execution layer around them is actually safe. That literature is scattered. Papers on sandbox isolation, capability and access control, policy enforcement, time-of-check-to-time-of-use (TOCTOU) races, Model Context Protocol (MCP) threats, identity delegation, execution provenance, network egress control, and static analysis of agent-generated code are published independently and rarely cite one another. We systematize 39 papers published between 2023 and 2026 into 17 categories, each verified directly against its source. The same verification protocol also confirms four disclosed, patched CVEs directly affecting production agent harnesses. Reading across categories surfaces five cross-cutting gaps that no single paper addresses. (1) Isolation architectures and capability models are almost never evaluated against one another on a shared benchmark. (2) Policy-enforcement studies report failure rates from 69% to 98% of real denylists, yet no isolation paper re-evaluates its own defense under that adversarial setting. (3) TOCTOU and MCP threats are analyzed as separate literatures despite both being instances of the same state-validation problem. (4) Every enforcement mechanism assumes an honest policy author, leaving policy-authoring error itself unaddressed. (5) Benign but out-of-scope agent actions occurring at rates up to 17.1% under realistic prompting are addressed by no access-control or capability paper in the corpus. Existing broader surveys of agentic AI security discuss sandboxing only as one item among many defenses, leaving execution security without a dedicated systematization. This paper is written to fill that gap. We conclude with a research agenda directed at the five gaps. 

---
# SCOReD: Student-Aware CoT Optimization for Recommendation Distillation 

**Authors**: Haz Sameen Shahgir, Yufei Li, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.05734)  

**Abstract**: Chain-of-thought (CoT) distillation in the recommendation domain is a necessary precursor to RL training, but raw teacher traces are ill-suited to this task. Large teachers approach the recommendation task with unusually high reasoning uncertainty, repeatedly rechecking their answers without revising them; supervised fine-tuning on such traces produces verbose students that never revise their initial guess. Furthermore, due to the novelty of the recommendation domain, the teacher's reasoning traces are highly out-of-distribution for the small student LLM.
We propose Student-Aware CoT Optimization for Recommendation Distillation (SCOReD), a CoT optimization framework tailored to recommendation that first parses each teacher trace into typed segments and uses the student LLM's attention to score the importance of each segment. Then SCOReD dynamically selects a per-segment edit (KEEP / REWRITE / FUSE / PRUNE) based on the output length and comparative log probability lift of the answer given the edit as per the student. Therefore, SCOReD prunes redundant sections of the reasoning trace while preserving information-dense sections and adapts raw teacher traces to the student's output distribution. Training on SCOReD-optimized CoTs provides a cleaner learning signal to the student model and improves over baseline SFT by 1.56% NDCG and 1.9% Recall@5, while reducing reasoning length by 27.3%. 

---
# Plainbook: Data Science, in Plain Language 

**Authors**: Luca de Alfaro, Mathis Aubert, Ranjit Jhala, Eliana Pastor, Elena Baralis  

**Link**: [PDF](https://arxiv.org/pdf/2607.05717)  

**Abstract**: Jupyter Notebooks have become widely adopted in data science, as they allow the sharing of reproducible computational analysis. They are, however, accessible only to people who understand computer code. To reach the broader audience of scientists interested in data analysis and computation, but unfamiliar with code, we introduce Plainbook, notebooks centered on natural language rather than code.
Plainbook is based on two principles: promote the natural language descriptions, and verify the values. In plainbook, the natural language descriptions are preserved, rather than the resulting code; the code is generated automatically from the cell descriptions. As natural language is read top to bottom, Plainbook adopts a linear execution semantics, in which cells are guaranteed to be executed in the order in which they appear; there is no "hidden state" or out-of-order execution as in Jupyter. To allow users who may not understand code to verify the correctness of the computation, we have built into Plainbook verification mechanisms centered on values and value inspection. These include mechanisms that focus on individual cells, akin to unit tests, as well as global mechanisms. Both the linear execution semantics, and the verification mechanisms, are underpinned by a snapshot kernel that caches execution states and makes execution and verification efficient. 

---
# IMR: Iterative Mode-World Weighted Regression for Multi-Agent Trajectory Prediction 

**Authors**: Honglin Wang, Shiyao Pan, Yun-Fu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05705)  

**Abstract**: Multi-agent motion prediction is essential for automated vehicles to understand the intentions of surrounding vehicles. However, previous prediction-based and anchor-based methods have limitations in mode diversity and prediction accuracy, respectively. These limitations may cause inadequate safety assessments and behavioral deviations in automated vehicles. To address this issue, a mode-world weighted regression loss is proposed to bridge the gap between these features. Specifically, this approach mitigates mode collapse while simultaneously improving world ranking and top-1 confidence. Furthermore, the proposed iterative decoder improves prediction accuracy by recurrently and segmentally generating trajectories. Experimental results show the proposed method ranks first in the Argoverse 2 multi-agent motion forecasting benchmark against other methods. 

---
# Depression Symptoms and Relational Patterns in 187k ChatGPT Histories 

**Authors**: Neil K. R. Sehgal, Dunigan Folk, Lyle Ungar, Sharath Chandra Guntuku  

**Link**: [PDF](https://arxiv.org/pdf/2607.05685)  

**Abstract**: Large language models are increasingly used as private, always-available conversational systems, but little is known about how people with depressive symptoms use them. Building on CSCW work on disclosure and peer support, we examine ChatGPT as an emerging informal support infrastructure: private, persistent, responsive, and available outside ordinary hours. We analyze 187,093 ChatGPT conversations from 766 participants who completed the PHQ-8, comparing those below the moderate-symptom threshold (score of 10) with those at or above it. Higher-PHQ participants used ChatGPT more for mental-health, interpersonal, loneliness, self-focused, and support-seeking conversations, with pronounced late-night and recurring month-level patterns. Their language contained more first-person singular pronouns and absolutist terms. They more often engaged ChatGPT in high-disclosure contexts, but professional redirection was not higher. Language-based prediction was modest and insufficient for screening (AUROC 0.591). We argue these histories should not be treated as clinical screening data but as evidence LLMs are increasingly used as informal support infrastructure. 

---
# Beyond Accuracy: How Humans Evaluate Legally Correct but Socially Controversial Legal Advice from Machines 

**Authors**: Benjamin Minhao Chen, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.05680)  

**Abstract**: AI systems are increasingly used to provide legal advice, raising questions about whether laypeople accept guidance from algorithms--especially when that advice is legally correct but socially controversial. We report a preregistered survey experiment with 3,348 adults in mainland China examining how people evaluate identical legal advice when it is attributed either to an AI system or to a human lawyer, and when it is accompanied by reasoning or not.
Contrary to expectations of algorithm aversion, attribution to an AI system has no net effect on perceived reasonableness. However, mediation analyses reveal opposing psychological pathways underlying this null result. AI-attributed advice is perceived as more objective, which increases perceived reasonableness, but also as less comprehensive and less attentive to special circumstances, which decreases perceived reasonableness. By contrast, providing legal reasoning substantially increases perceived reasonableness regardless of source, largely by enhancing perceptions of objectivity. Qualitative responses corroborate this tension between objectivity and contextual sensitivity in evaluations of legal advice.
Together, these findings suggest that public responses to AI legal advisors are shaped not by rigid attitudes toward automation, but by the balancing of competing normative expectations. The results have implications for theories of algorithm aversion and the design of AI recommendation systems in normatively salient domains. 

---
# RPAM: A Principled Metric for Evaluating Associations in Language Models with High Predictive Validity in Downstream Outputs 

**Authors**: Damian Hodel, Jevin West, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2607.05679)  

**Abstract**: Language models (LMs) exhibit problematic biases, such as stereotypes. Effectively analyzing and mitigating such biases requires accurate and generalizable evaluation methods of the underlying associations. Some existing approaches focus on downstream metrics that analyze associations in generated text. Since generated text content can vary drastically across LMs, such metrics often require specialized evaluation datasets, which limits the generalization of such downstream metrics. In contrast, upstream metrics examine LMs at the fundamental level of embeddings or continuation probabilities, enabling principled association analyses across LMs. Yet, to date, no upstream metric for generative LMs has uncovered a strong relationship with real-world associations, including those measured in generated text. To address this gap, we introduce the Relative Probability Association Metric (RPAM), an association evaluation metric for generative LMs. For three LMs of different quality of language generation and purpose (Mistral-7B-Instruct, Mistral-7B, and GPT-2) and well-studied evaluation datasets (WEAT-WS, Bellezza, WS-353, and SST2), we find a strong relationship between upstream RPAM measurements and corresponding implicit and explicit associations observed in humans, as well as biases measured downstream with LM-specific tasks, outperforming prior record values where applicable. 

---
# What Do AI Agents Actually Change? An Empirical Taxonomy of Mutation Patterns in Performance-Improving Pull Requests 

**Authors**: Illia Dovhoshliubnyi, Nima Soroush, Ashkan Sami, Alexander Brownlee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05666)  

**Abstract**: AI coding agents are black boxes: we cannot inspect how they generate code, but we can inspect what they change. This distinction matters for search-based software engineering (SBSE), where techniques such as genetic improvement (in the performance-optimisation application we study) depend on mutation operators that reflect how code is actually transformed. Fewer than 1% of the 33,596 agent PRs in AIDev-pop target performance, making each case a rare window into otherwise opaque agent behaviour. We classify 1,254 performance-relevant diff hunks from 216 of these PRs, spanning five agent systems, against the 18-category syntactic mutation taxonomy of Even-Mendoza et al. (2025) using a dual-LLM intersection pipeline. Three categories dominate: name modification (37.0%), object creation (26.4%), and type change (22.7%), a profile markedly different from prior GI corpora where no change accounted for 84%. Each agent's deployed system commits to a distinctive mutation vocabulary, and each performance strategy activates a largely disjoint category subset. Agent identity and target strategy are therefore informative priors that narrow the effective SBSE operator space. Replication package: this https URL 

---
# Physics-Regularized Machine Learning for Proprioceptive Vehicle Localization Using Onboard Sensors 

**Authors**: Abinav Kalyanasundaram, Karthikeyan Chandra Sekaran, Wolfgang Utschick, Michael Botsch  

**Link**: [PDF](https://arxiv.org/pdf/2607.05663)  

**Abstract**: Accurate and robust localization is essential for autonomous mobility systems in real-world environments. While fusing Inertial Measurement Unit (IMU) data with satellite-based correction signals provides precise vehicle pose estimates, performance degrades substantially during outages. Recent studies indicate that Machine Learning (ML) can improve IMU-based proprioceptive localization, highlighting untapped potential for onboard sensors readily available in production vehicles. This paper introduces Physics-Regularized Machine Learning for Localization (PRML2), a hybrid framework that combines the complementary strengths of Kalman filtering and data-driven learning to estimate vehicle pose directly from onboard sensors. A key aspect of PRML2 is its physics-regularized learning, enabled by end-to-end training of an ML model through a differentiable Kalman filter. This improves consistency with vehicle motion models, thereby enhancing both localization accuracy and generalization across driving conditions. We evaluate the performance limits of ML-enhanced onboard odometry on a publicly available dataset and show that PRML2 achieves superior localization accuracy and demonstrates real-time capability. This work also introduces a novel dataset to support vehicle localization research under low-friction conditions. The proposed framework provides a robust and cost-effective solution for vehicle localization under degraded sensing conditions by integrating learning with physics-based priors. 

---
# Do It Right! A Methodology for Successful NLP System Development 

**Authors**: Olga V. Patterson, Brett South, T. Elizabeth Workman, Scott L DuVall  

**Link**: [PDF](https://arxiv.org/pdf/2607.05644)  

**Abstract**: Natural language processing (NLP) is a common method for supplying data to clinical research and decision making by extracting information from electronic medical records. Numerous textbooks and tutorials describe specific algorithms and applications for text processing, yet algorithmic knowledge is only one ingredient of a successful NLP project. Drawing on the available literature, this paper presents a stepwise approach that applies the Systems Development Life Cycle (SDLC) to projects that rely on data extraction through language processing. 

---
# EvalLoop: A Methodology for Evaluation-Driven Iterative Improvement of Business AI Systems 

**Authors**: Kenneth Benavides, Josh Fleischer, Danti Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05638)  

**Abstract**: Teams deploying large language models in business contexts need evaluation systems, yet most treat evaluation as static model selection: run benchmarks, rank models, deploy the winner. This framing misses evaluation's primary value for production systems--diagnosing why a system underperforms and guiding what to fix. We present EvalLoop, a methodology for evaluation-driven iterative improvement. EvalLoop organizes evaluation around three mechanisms: (1) dimensional metric grouping that decomposes quality into business-relevant dimensions enabling orthogonal failure diagnosis; (2) failure mode classification that categorizes why outputs fail within weak dimensions, bridging diagnosis to action; and (3) a structured iteration workflow where each evaluation run varies one system variable and compares dimensional profiles before and after. We validate EvalLoop through a case study on sales intelligence briefing generation (10 models, 3 providers, 18 metrics, 5 dimensions, 3 iterations). Dimensional diagnosis identified that 69% of hallucination failures were prompt-induced interpretation errors--invisible in aggregate scoring. A targeted prompt fix improved the best model from 82.6% to 94.6% overall, with improvement concentrated in diagnosed dimensions (Content Accuracy +16.8pp, Synthesis Power +26.4pp). An undirected configuration change in a prior iteration produced zero impact, illustrating the cost of iterating without diagnosis. We additionally demonstrate that dimensional profiling enables deployment-specific model selection, and that a one-time blind human gate on a finalist panel (4 models, 16 cases) confirms dimensional rankings while resolving multi-criteria deployment trade-offs--a 94% reduction in review burden compared to evaluating the full design. EvalLoop is packaged as reusable artifacts (playbook, agent specification, template repository) for adoption by other teams. 

---
# Safe Bayesian Optimization with Counterfactual Policies 

**Authors**: Katherine Avery, Bruno Castro da Silva, David Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05620)  

**Abstract**: In many decision-making settings, new interventions are acceptable only if they do not reduce outcomes below some established threshold. For example, in clinical medicine, new treatments are often acceptable only if they do not worsen outcomes relative to an established standard of care. Safe Bayesian optimization maximizes an objective subject to safety constraints. In the setting that we consider here, safety is defined relative to a known baseline policy whose outcomes are counterfactual and therefore unobserved. Thus, the counterfactual outcomes of the baseline policy must be estimated and those (uncertain) estimates must be used to safely optimize the objective. We address this estimation problem by using conformal prediction to construct valid uncertainty intervals for counterfactual baseline outcomes, and we show how these intervals can be integrated into safe Bayesian optimization to ensure that constraint violations occur at or below a user-specified rate. We also show how to adapt these conformal estimates to different kinds of covariate shift. We provide a safety proof, experimental evidence, and a sensitivity analysis. 

---
# BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension 

**Authors**: Abu Tyeb Azad, Ishita Sur Apan, Fahim Ahmed, Sumaiya Karim Katha, Ezharuddin Jubaer, Armun Alam, Pranjal Kumar Nandi, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05614)  

**Abstract**: Document comprehension is a challenging yet impactful task for Multimodal Large Language Models, especially as these systems see growing adoption in real-world, human-centric applications. However, this adoption is limited for low-resource languages such as Bangla due to the scarcity of high-quality annotated data. To address this gap, we introduce BaFCo, a benchmark dataset for Bangla form comprehension with a focus on Document Layout Analysis (DLA) and Key Information Extraction (KIE). BaFCo curates 200 multi-page complex Bangladeshi government forms, sourced from across diverse sectors including agriculture, education, banking, and land management. To accurately capture the structural and contextual complexity of these forms, we define a fine-grained annotation schema comprising 26 types of form entities, along with a separate coarse form entity set consisting of 5 types. We evaluate the latest MLLMs from the ChatGPT, Gemini, Claude, Qwen, and Kimi series using zero-shot and chain-of-thought prompts under both low and high reasoning setups. Our results reveal limitations in current MLLMs' ability in comprehending Bangla forms, particularly in accurately localizing highly granular form entities. Our dataset and code is available at: this https URL 

---
# To Retain or to Adapt? Generalizing Continual Learning 

**Authors**: Giulia Lanzillotta, Mandana Samiei, Doina Precup, Razvan Pascanu, Claire Vernade  

**Link**: [PDF](https://arxiv.org/pdf/2607.05609)  

**Abstract**: The Continual Learning (CL) literature has long been driven by the goal of mitigating catastrophic forgetting. This objective rests on a pervasive, often unstated assumption: that a lifelong learner should approximate the Joint-Task Learning (JTL) solution and retain all previously acquired knowledge. We challenge this retention-centered premise, arguing that in non-stationary environments prioritizing retention can impede real-time adaptation. Shifting the focus to the Average Lifelong Error (ALE), we formalize CL as an online optimization problem governed by the interaction between environmental and learning dynamics. We introduce Transfer Efficiency as a quantitative measure of the tension between Instability, the bias inherited from conflicting past experience, and Transient Error, the optimization cost of learning new tasks from scratch. Under mild convergence conditions, holding across linear and neural network models, this decomposition yields a Critical Task Duration: a closed-form threshold beyond which historical knowledge transitions from a warm-start advantage to an optimization liability whenever retention induces a positive stationary bias. We validate these theoretical predictions on continual image classification and reinforcement learning benchmarks. Finally, by connecting continual learning to the online learning framework of predictable sequences, we show that JTL is only one instance of a broader family of objectives, and we propose a new general class of continual learning algorithms, which we call Predictive Continual Learning. Predictive CL algorithms optimize expected future performance under an explicit, dynamically updated model of future tasks. As a proof of concept, we analyze a Window algorithm that interpolates between JTL and Independent-Task Learning (ITL), outperforming both under controlled distributional drift. 

---
# Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO) 

**Authors**: Fabio Hellmann, Alexander Hustinx, Benjamin D. Solomon, GestaltMatcher Database Consortium, Tzung-Chien Hsieh, Peter Krawitz, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2607.05585)  

**Abstract**: FaceMesh2HPO is a framework for classifying facial phenotypic descriptors aligned with the Human Phenotype Ontology (HPO) to support clinical diagnosis. Using annotations from 124 clinicians across 10 disorders (107 HPO terms) combined with non-syndromic controls, we generated 3D facial meshes (478 landmarks) from 2D images and trained a hierarchical PointNet-based pipeline with cascading classification and feature elimination. The best models, incorporating 3D meshes, facial outline, and demographic metadata, achieved AUROCs between ~0.55 and ~0.89, with higher performance at parent nodes than leaf terms. External validation showed variable generalizability across disorders. Results demonstrate that hierarchical modeling of 3D facial geometry enables interpretable, ontology-linked phenotype classification, though performance on rare leaf terms remains limited. Improved data diversity and feature selection strategies are needed to enhance robustness and clinical utility. 

---
# ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modelin 

**Authors**: Archie Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.05583)  

**Abstract**: Contemporary language models are dominated by the transformer architecture, which leverages self-attention mechanisms to enable more efficient, parallelized training across a wide set of documents and corpora. This has allowed transformers to effectively model data across a wide range of modalities and contexts. However, transformers, along with their conventional counterparts such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), often struggle to maintain efficiency when processing long contexts. We introduce ResonatorLM, a new mechanism that replaces attention with a physics-derived alternative. ResonatorLM treats token sequences as a single, driven one-dimensional latent field and replaces attention dot products with causal functions of damped resonators. We implement ResonatorLM on a traditional network architecture and test it on standard long-context modeling tasks. We find that in a small, 6M matched setting, training and prefill speedups increase with sequence length, decode speed reaches 6.47x compared to that of a standard, optimized transformer at 32K tokens, and accuracy reaches 61.31 percent (compared to 55.32 percent) on WikiText. 

---
# Whose fairness? Structural concentration in AI bias research 

**Authors**: Abhash Shrestha, Subigya Gautam, Anu Sapkota, Sanju Tiwari, Tek Raj Chhetri  

**Link**: [PDF](https://arxiv.org/pdf/2607.05574)  

**Abstract**: Artificial intelligence increasingly mediates consequential decisions in healthcare, law, and public services, and the field has responded with an extensive methodology for measuring and mitigating bias. Yet the fairness definitions, benchmarks, and debiasing frameworks on which this methodology rests are treated as universal while being produced by a research community whose composition has never been characterized. We show that the AI bias research are structurally concentrated, and that this concentration is greatest, geographically, in precisely the domain the rest of the field inherits from. Analyzing 692 publications spanning five thematic domains, combining bibliometric analysis with semantic clustering, we find that research activity is dominated by a small set of countries, institutions, and authors, with the United States leading publication output and collaboration networks across every domain and most strongly in general fairness and bias mitigation, the largest, most-cited domain with meaningful representation across all four semantic clusters. Low- and middle-income countries remain largely absent from the community and its collaboration networks, and citation influence is highly skewed (median = 9; mean =93.5 ), indicating that a small fraction of publications disproportionately shapes the field. Because the general-fairness domain supplies the definitions and benchmarks that application areas apply, concentration of research effort in this foundational domain propagates across AI bias research as a whole - raising the concern that mitigation methods developed and validated within a narrow set of contexts may not generalize to all populations and settings where AI is deployed. We provide an interactive atlas for continuous monitoring of the field's structure. 

---
# Harnessing Generative Image Models for Training-Free Primitive Shape Abstraction 

**Authors**: Gregor Kobsik, Tim Elsner, Leif Kobbelt  

**Link**: [PDF](https://arxiv.org/pdf/2607.05568)  

**Abstract**: Representing 3D shapes as compact sets of geometric primitives is fundamental to robotics, simulation, and scene understanding. Generative image models trained at scale have recently emerged as generalist visual learners that can identify and segment object parts directly in the image domain, across arbitrary categories and without task-specific training. Adapting such models to downstream tasks typically requires fine-tuning; we ask whether their pretrained capability can instead be harnessed directly, without any training, and answer affirmatively with a training-free harness. Our pipeline renders multi-view images of a 3D object, uses a vision-language model to analyze its semantic parts, prompts a generative image model to paint a color-coded part segmentation mask, reprojects it onto the geometry, and fits a superquadric primitive to each part via parameter optimization. The approach contains no learned parameters: it is category-agnostic and orientation-invariant, properties that previous learning-based models struggled with. Its accuracy ceiling rises with future generative-model improvements, which we confirm with a ground-truth segmentation study showing that part segmentation, not primitive fitting, is the current accuracy bottleneck. On HumanPrim and Toys4K, our method achieves the lowest Chamfer distance among all evaluated methods, using 5--9 primitives per object on average. 

---
# Prompt Robustness Is Task-Dependent: Comparing Objective and Belief-Style Questions in LLM Evaluation 

**Authors**: Sadia Kamal, Arefa Patwary, Anthony Marchiafava, Atriya Sen, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.05554)  

**Abstract**: Survey-style evaluations of large language models often treat a prompted response as a measure of a model's values or beliefs. This assumption is particularly fragile when responses are read as evidence of political values, social attitudes, or beliefs. We ask whether prompt robustness differs between objective questions with fixed answers and subjective questions that ask for opinions or values. We evaluate four instruction-tuned model families on three objective datasets (MMLU, ARC, and CulturalBench) and three subjective datasets (Political Compass Test, ValueBench, and World Values Survey). For each question/statement, we apply multiple types of prompt changes, such as variations in wording, framing, and format, and measure whether the model gives the same answer across variants. Using a binomial generalized estimating equation, we find significant effects of model, dataset, prompt category, and their interactions. The dataset type effect is also significant, and the interaction between dataset type and prompt category is large. These results show that prompt robustness depends on the question type, the prompt change, and the model. 

---
# The yes-no bias of large language models reflects answer order and wording, not shifts in moral judgment 

**Authors**: Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05552)  

**Abstract**: Large language models (LLMs) increasingly issue judgments read as binary verdicts, and a growing literature reports such judgments shifting under logically irrelevant changes of wording - among them an amplified yes-no bias on moral dilemmas, absent in humans. A single framing cannot say what such a shift is: in a yes/no question the word "no" is at once logical verdict, lexical token, and last-printed option. We introduce a psychometric battery that separates these: crossed symmetrization - every logically irrelevant factor flipped in balanced pairs - across a corpus of question forms. A graded rating across logically equivalent forms recovers a coherent internal moral scale: frontier models' stance $\theta$ is nearly format-invariant (cross-form incoherence 0.12-0.21 on a $\pm 1$ axis); small open-weight models fail in model-specific ways. Forcing the verdict through yes/no overlays a decomposable artifact: an order bias toward the last-printed option - opposite to classic human primacy - plus a lexical pull toward the word "no"; the artifact is substantial only in the Claude models (story-averaged -0.32 to -0.86), $\approx 0$ for GPT-5.5 and Gemini, and shrinks under extended reasoning. The word and the verdict share one token; swapping the words for arbitrary labels separates them, and the verdict-attached logical bias proves $\approx 0$ for every frontier model, while model-specific label and order attachments remain: the models are not drawn toward rejecting - the pull follows the printed surface, not the verdict it carries. A minimal model, $P = \sigma((\theta \pm m)/s)$, summarizes any such artifact by a framing susceptibility m and a moral decisiveness s, measurably distinct from sampling temperature. The battery applies unchanged to any dilemma set and binary format: measuring what a model values requires crossing the frames of the question, not asking once. 

---
# Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks 

**Authors**: Yibo Hu, Jiaming Qu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05545)  

**Abstract**: LLM conformity is often used to describe cases where a model changes a correct answer toward a peer or group response. We show that most of this apparent conformity survives even after the peer is removed. The reason is a confound: standard conformity prompts mix two cues at once, the presence of a speaker and the repeated wrong answer itself. Existing benchmarks vary these cues together, so they cannot tell how much of the revision actually depends on the speaker. We introduce a no-source condition: the same asserted answer with the explicit speaker removed. Across six open-weight LLMs and seven QA and reasoning datasets, this condition alone causes harmful revision in $66.5\%$ of initially correct cases, compared with $10.3\%$ under a plain re-ask. The effect also remains when the repeated answer is paraphrased and when answer options are hidden in an open-ended setting. Source framing mainly modulates this floor: expert-panel framing raises it, while minimal person labels do not reliably raise it. When models flip, they are usually confidently wrong, and simple recalibration does not recover the original answer. Source attribution still matters, but it should be measured as an increment above this speaker-free floor. The methodological lesson is that conformity benchmarks should first measure what remains after the speaker is removed; without this step, benchmarks may mistake repeated text for social influence. 

---
# Self-Review Reinforcement Learning (SRRL) with Cross-Episode Memory and Policy Distillation 

**Authors**: Muhammad Zain Amin, Kibele Sebnem Yildirim  

**Link**: [PDF](https://arxiv.org/pdf/2607.05541)  

**Abstract**: Reinforcement Learning is commonly used to train large language models using environmental feedback. In applied settings, the environment usually provides sparse or delayed feedback. This makes it difficult for the model to pinpoint which actions in its reasoning led to success or failure. So, learning effectively from these signals is hard because the model must determine how each failure should inform meaningful behavioral corrections in subsequent iterations. We introduce a training framework, Self-Review Reinforcement Learning, that embeds an explicit self-review step into each RL episode. When a first-pass response fails, the model generates a self-review to identify what went wrong, which conditions an improved second attempt. Unlike inference-time reflection approaches, such as Reflexion, the framework optimizes self-review with policy gradients and internalizes improvements into the base policy via selective distillation, ensuring they persist across future episodes. A cross-episode memory keeps successful self-reviews for reuse when encountering similar tasks in future episodes during training. We evaluate SRRL against a standard RLVR baseline using the GRPO optimizer across two language models, Qwen 3-4B and OLMo-3- 7B, on GSM8K benchmark. SRRL consistently outperforms the RLVR in final reward performance and achieves greater learning efficiency by successfully transforming feedback into behavioral improvement. 

---
# Rendering-Aware Bayesian 3D Gaussian Splatting with Native Uncertainty and Adaptive Complexity Control 

**Authors**: Gaoxiang Jia, Vikram Appia, Junzhou Huang, Xinlei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05522)  

**Abstract**: 3D Gaussian splatting (3DGS) is a strong representation for real-time novel-view synthesis, but its standard training pipeline relies on point estimates and hand-tuned heuristics, providing no native uncertainty or principled complexity control. This is most limiting under sparse views or fixed acquisition budgets, where a model must identify weakly supported geometry and select informative views. We introduce a rendering-aware Bayesian 3DGS framework that tracks Gaussian geometry with a Normal-Inverse-Wishart posterior over means and covariances using renderer-derived surrogate summaries. An optional Dirichlet-process extension adds a probabilistic component-usage signal, and the training schedule makes the closed-form versus approximate inference boundary explicit. Re-rendering posterior geometry samples yields native predictive uncertainty for interval calibration and active view selection. In a fixed-budget 16-to-32 active-view task, native NIW acquisition improves PSNR by +0.453 dB and LPIPS by -0.0146 over a scoring-only 3-member standard-ensemble baseline, winning 29/39 scene-seed pairs and 10/13 scene means; it also improves over PPU-style (+0.355 dB) and NIW-proxy (+0.401 dB) acquisition. NIW native intervals reduce 95% coverage error by about 17x relative to a shared proxy (0.046 vs. 0.796) and are about 10x closer to nominal coverage than a 3-member deep ensemble (0.047 vs. 0.454) at roughly one-third the training cost. As a reconstruction compatibility check, paired NIW-vs-standard analysis over 39 scene-seed runs yields +0.030 dB PSNR with 1.6% additional training time. These results position Bayesian 3DGS as a practical probabilistic scene representation for decision-facing tasks such as active view selection. 

---
# aiAuthZ: Off-Host, Identity-Bound Authorization for AI Agents 

**Authors**: Sai Varun Kodathala  

**Link**: [PDF](https://arxiv.org/pdf/2607.05518)  

**Abstract**: AI agents issue tool calls on the basis of text they cannot verify, so any party who controls part of the context can forge the appearance of authority. I evaluate 15 contemporary language models against eight attack scenarios derived from a published corpus of real agent incidents and find that refusal varies from 100% down to 38% across fully evaluated models; the most expensive model refused only half of the attacks despite a twentyfold price spread. I present aiAuthZ, an authorization gateway that moves the safety decision off the agent's host. Before a tool call executes, the gateway verifies caller identity with a per-message HMAC-SHA256 signature bound to a single-use nonce and a timestamp window, and it evaluates a role-based and argument-level policy that the agent can neither read nor modify. Every decision joins a SHA-256 hash-chained audit log, and each accepted message yields an HMAC-authenticated QR receipt that achieves 94% mean verification across eight transmission channels, with zero forgeries accepted in 25 wrong-key trials. With the gateway in place, residual attack success falls to 0% for all 15 models at no more than 0.03 ms of added decision latency. On the AgentDojo banking suite, aiAuthZ blocks all seven attacker-directed tool calls the evaluated agents emit, at the cost of one legitimate first-time payment, while a spotlighting baseline allows two injections to succeed. Across nine in-scope case studies from the same incident corpus, aiAuthZ blocks nine of nine, against four of nine for a policy baseline without identity binding. The gateway does not prevent a model from being deceived; it prevents a deceived model from acting beyond the verified user's authority on every call routed through it. The implementation and all experiments are released at this https URL. 

---
# Statistical Adversaries: Natural Backdoor-like Features in Vision Datasets 

**Authors**: Paul K. Mandal, Pavan Reddy, Tristan Malatynski  

**Link**: [PDF](https://arxiv.org/pdf/2607.05516)  

**Abstract**: Model-specific adversarial attacks have been extensively studied. We study a different failure mode: naturally occurring statistical signals in vision data that can behave like backdoor-like triggers without being maliciously inserted. We call these signals statistical adversaries. We analyse Imagenet to find patterns that are strongly linked to certain labels. We then use statistical controls to remove random correlations from our candidate signals. Finally, we demonstrate that these signals directly and predictably alter model predictions. These statistical adversaries are more targeted than generic corruptions and transfer across different model architectures. This suggests that some vulnerabilities are driven by dataset structure and distribution rather than a single model's idiosyncrasies. We conclude that ordinary datasets can contain exploitable adversarial surfaces even in the absence of poisoning, and suggest that dataset audits should treat spurious structure not only as a source of bias or interpretability failure, but also as a latent attack surface for vision models. 

---
# Lean-Quantum: Toward AI-Assisted Formalization of Quantum Information 

**Authors**: Kazumi Kasaura, Kei Tsukamoto, Kento Mori, Risa Mizuno, Takahiro Namatame, Yuta Oriike, Masaya Taniguchi, Sho Sonoda, Hayata Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2607.05492)  

**Abstract**: Quantum information theory is built on entropic quantities; among them, the sandwiched Rényi relative entropy is a fundamental divergence with various applications, and its data processing inequality (DPI) under quantum channels is a cornerstone result. In this work, we present a Lean 4 library for quantum information, designed as a reusable formal infrastructure for theoretical analysis. As a central demonstration of the library, we formalize the DPI for the sandwiched Rényi relative entropy for positive semidefinite operators on finite-dimensional quantum systems. The library provides a basis-independent operator-theoretic framework for finite-dimensional quantum mechanics compatible with the standard mathematical library Mathlib, including reusable interfaces for finite-dimensional systems, states, channels, tensor products, partial traces, Choi operators, Kraus representations, and Stinespring representations. It also builds infrastructure for noncommutative trace inequalities, including operator monotonicity and convexity via the real continuous functional calculus, block-operator positivity, Hilbert-Schmidt operator spaces, Jensen's operator inequality, generalized perspectives, operator power means, and Lieb-Ando trace inequalities. On top of this framework, we formalize entropy-specific ingredients for the DPI: variational formulas for the sandwiched quasi-entropy via Young and reverse-Young inequalities, tensor-product compatibility of real powers, and Haar measures on unitary groups. Together, these components yield a Lean formalization of the DPI, give strong subadditivity as a corollary, and provide the last missing component needed to complete the Lean formalization of the generalized quantum Stein's lemma. More broadly, the development provides machine-checkable foundations for future formalized and AI-assisted research in quantum information theory. 

---
# PatchOptic for Shared-State LLM Workflows with Projected Views and Verified Structured Updates 

**Authors**: Zhaoyu Bai, Jiaqi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.05483)  

**Abstract**: Agentic workflows often operate over shared, structured state. Because LLM context windows are limited, each model invocation is typically shown only the state fragment needed for the current workflow step, a pattern commonly known as progressive disclosure. Modern systems construct such model-facing views using grep-like keyword search, retrieval-augmented generation (RAG), abstract-syntax-tree (AST) queries, and task-specific agent skills. These methods make the read side manageable, but they do not define when a locally proposed rewrite is valid after it is applied back to the full state. The missing piece is a contract between local updates and global validity. We introduce PatchOptic, an optic-inspired interface for shared-state LLM workflows. Optics are compositional bidirectional accessors that describe how views of structured data are read and updated. PatchOptic borrows this view/update intuition and realizes it through projected reads and verified structured patches. Each workflow step declares a projected read view, an authorized write region, and a patch-source region. Beyond runtime enforcement, the same declaration yields a path-level footprint that supports delegation, sub-workflow composition, and static certificates for reordering independent steps within the same phase. We evaluate this design with PatchBench, a benchmark with 46 cases across domains. The results show that projected reads reduce reported leakage and token cost while preserving accepted-output quality under the strong actor. Runtime verification blocks declared workflow-contract violations before commit, and patch-read enforcement rejects compromised patch artifacts that use hidden sources. 

---
# Full-range Binary Classifier Calibration for Stable Model Updates in Production 

**Authors**: Konstantin Berlin  

**Link**: [PDF](https://arxiv.org/pdf/2607.05481)  

**Abstract**: Detection models running in adversarial environments face a malicious distribution that drifts rapidly while the benign distribution stays comparatively stable, so teams retrain and redeploy constantly to stay ahead of new threats. Retraining tends to change the output prediction scores, which breaks downstream users of the model. For these security-oriented models we need consistent false-positive rate (FPR) across all output values, whereas standard probability-calibration methods target class probability rather than an FPR contract. We introduce a method built on top of existing calibration primitives that targets the whole FPR curve, giving scores a consistent FPR meaning across deployments. On one held-out split, the observed relative FPR error was at most 2.3% from 10% down to 0.1% FPR and 7.2% at 0.01% FPR. The shipped artifact remains under 200 KB in measurements across calibration sets from 1K to 10M benign samples. 

---
# Privilege and confidentiality in generative AI workflows 

**Authors**: Václav Janeček, Thomas Melham  

**Link**: [PDF](https://arxiv.org/pdf/2607.05479)  

**Abstract**: Generative AI (GenAI) systems store and process client data in three distinct ways: in the model's parameters through training and memorisation, in the context window during a live session, and in knowledge databases for retrieval-augmented generation (RAG). Each mode creates different and often counter-intuitive risks to confidentiality and legal professional privilege, and each calls for specific governance responses. Drawing on the first English and American decisions to address privilege and generative AI, UK and Munir v Secretary of State for the Home Department and United States v Heppner, on the orthodox privilege authorities against which those decisions must be read, and on recent computer science research, we explain the three modes of data storage and processing in terms accessible to practitioners and analyse the legal consequences of each. We then situate the analysis within the regulatory framework governing solicitors in England and Wales and within the ordinary principles of professional negligence, arguing that the standard of effective information governance (and with it the benchmark against which negligence and misconduct will be measured) is changing. Although we write primarily for SRA-regulated practitioners, our data-governance analysis is framed to extend to any jurisdiction in which the protection of privilege or professional secrecy depends on demonstrable confidentiality. The ultimate aim of this article is to help legal services professionals understand salient data leakage risks in GenAI systems and thereby facilitate a more responsible deployment of GenAI on client data and other sensitive material. 

---
# Decision Protocols in Multi-Agent Large Language Model Conversations 

**Authors**: Lars Benedikt Kaesberg  

**Link**: [PDF](https://arxiv.org/pdf/2607.05477)  

**Abstract**: Improving the task performance of Large Language Models (LLMs) is essential, yet scaling these models faces significant challenges such as diminishing returns and high costs. Multi-Agent Systems (MAS) offer a promising solution by distributing tasks among specialized agents to improve the overall task performance. This can reduce training costs at the expense of increased test time due to the discussion and decision-making process. The decision protocol is a critical component of MAS because it specifies how multiple agents collaborate to create a final solution. This thesis introduces the Multi-Agent LLM (MALLM) framework, which implements and evaluates various decision protocols, namely voting, consensus, and judge decision mechanisms, to simulate multi-agent discussions for conversational task solving. Unlike previous work that used a single decision protocol or tested them on limited datasets, this study systematically examines their impact on a diverse set of tasks, ranging from knowledge-based datasets (MMLU, MMLU-Pro, GPQA) and logic-based datasets (StrategyQA, MuSR, Math-lvl-5, SQuAD 2.0). The results indicate that consensus protocols excel in knowledge-intensive domains while voting and judge protocols are more effective for logic-based tasks. Increasing response diversity through independent solution generation improves decision quality, while changes in information access during the decision process have minimal impact. 

---
# Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference 

**Authors**: Guanyu Cai, Ruiming Tian, Lang Yang, Zhouhong Ren, Jinliang Yuan, Lingkun Li, Jiliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05475)  

**Abstract**: Deploying Large Language Models (LLMs) on mobile devices enhances privacy and reduces latency, but is severely bottlenecked by hardware inefficiency. We present the first comprehensive, cross-layer measurement study of mobile LLM inference, uniquely spanning five mainstream frameworks (e.g., this http URL, GENIE) and three hardware backends (CPU, GPU, NPU). To enable this analysis, we develop PowerBench, a fine-grained profiling tool that provides the first backend-specific energy attribution, moving beyond traditional device-level measurements. Our study yields three critical insights: (1) Framework-induced performance gaps are substantially amplified on NPUs, reaching up to 10x using custom operators due to divergent offloading and quantization strategies. (2) We identify a distinct phase split where NPUs excel at compute-bound prefilling, while CPUs outperform all other backends in memory-bound decoding. This is driven by the NPU's preference for large, fixed-shape workloads, which conflicts with the small-kernel, dynamic nature of decoding. (3) Backend-specific profiling uncovers substantial scheduling headroom missed by prior work. Suboptimal thread configurations, uncoordinated NPU sleep latencies, and CPU polling intervals result in up to 40% energy waste. Leveraging these findings, we present an energy-oriented best-practice configuration for mobile LLM inference. We estimate that this configuration could reduce energy consumption by up to 54.8% on the NPU backend across three datasets. 

---
# Binocular Gaze Estimation with Single Camera and Single Light Source 

**Authors**: Tongbing Huang, Yang Fu, Yunfei Wang, Zhaocan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05473)  

**Abstract**: According to commonly consented theories, the minimum hardware requirement for gaze tracker is one camera and two light sources to realize gaze estimation with free head movements. However, in some scenarios such as eye tracking on mobile devices, it is preferable to use less components, especially light sources. We propose a gaze estimation method with one camera and one light source. A "virtual light source" is introduced, which is geometrically placed symmetrically to the real light source with respect to the camera, and generates a "virtual glint" in the acquired image. We estimate the "virtual glint" by exploiting the relationship between the distance between two pupils and two glints in the captured image, and estimate the gaze with polynomial regression assuming two light sources are available. A new normalization factor for regression method is verified, which turns out to be practical for one-glint system. The performance is proved to be acceptable, while degradation is noticed compared to system with two actual light sources. 

---
# KAT-Coder-V2.5 Technical Report 

**Authors**: Bo Huang, Fengxiang Li, Hao Xu, Haoyang Huang, Hongyi Fu, Jinhua Hao, Kun Yuan, Minglei Zhang, Pengcheng Xu, Shiyang Liu, Wenhao Zhuang, Yuze Shi, Zongxian Feng, Chao Wang, Cheng He, Chongling Rao, Deyu Cao, Fan Yang, Gang Xiong, Haochen Liu, Jiabao Li, Jian Liang, Jinghui Jia, Jingwen Chang, Jun Du, Junyu Shi, Min Li, Mingqi Wu, Qiang Gao, Shangpeng Yan, Shaotong Qi, Shu Xu, Shuo Zhou, Tiankuo Xu, Tong Zheng, Weilun Zhao, Xiancheng Meng, Xianda Sun, Xiaoyu Jiang, Xunhao Jia, Yao Xia, Yimeng Xu, Yinghan Cui, Yingpeng Chen, Yiwen Ning, Yong Wang, Yuxuan Sun, Zhongsheng Liu, Ming Sun, Cheng Luo, Chen Yang, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2607.05471)  

**Abstract**: We present KAT-Coder-V2.5, a coding-focused agentic model trained to act autonomously inside real, executable repositories rather than as a single-turn code generator. Its capability is bottlenecked less by model scale than by the scarcity of reproducible environments, verifiable rewards, and high-value trajectories, which we address with an end-to-end agentic post-training framework. AutoBuilder reconstructs multilingual repositories into sandboxed environments with fail-to-pass and pass-to-pass verification at scale, from which we regenerate self-contained task specifications, recover near-miss trajectories, and distill supervision through process-aware filtering, while KwaiClawEnv synthesizes large-scale tool-use trajectories from executable services and real task seeds. We further scale reinforcement learning with harness randomization, a reliability-hardened sandbox, an asymmetric actor--critic PPO with hindsight-augmented value estimation, and a harness-oriented reward framework, and unify SWE, Agent-Claw, and WebCoding experts via Multi-Teacher On-Policy Distillation. Across six software-engineering and agentic benchmarks, KAT-Coder-V2.5 delivers the best agentic tool-use result on PinchBench and ranks second only to the frontier Opus 4.8 on repository-level software engineering. Our service is available at this https URL. 

---
# Breaking Structural Isolation: Scalable Graph Clustering via Community-Aware Sampling and Structural Entropy 

**Authors**: Jingyun Zhang, Hao Peng, Jianxin Li, Angsheng Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05469)  

**Abstract**: Unsupervised graph clustering is a fundamental technique for uncovering underlying semantic patterns in large-scale networks. Although Graph Contrastive Learning has demonstrated promising performance, existing methods often suffer from the "structural isolation" issue during mini-batch training, making it challenging to capture cohesive community structures that characterize the global topological distribution. To address these challenges, we propose SCISE, a Scalable unsupervised graph Clustering framework that preserves structural Integrity by synergizing community-aware sampling with constrained Structural Entropy. Specifically, we first introduce the Structural Entropy Community Constraint operator (SECC), which optimizes structural information within a constrained solution space to mitigate community fragmentation and enhance partition cohesion. Second, to prevent global information loss during batch training, we design a Community-Aware Sampling Expansion (CSampE) mechanism that incorporates the community context of target nodes into sampling batches, effectively breaking structural barriers and preserving topological integrity. Finally, we devise a Structural Contrastive Learning (StructCL) module that refines edge weights based on intra-batch structural similarity, guiding the encoder to learn representations in a higher-order structural space. Extensive experiments on six mainstream benchmark datasets demonstrate that SCISE significantly outperforms state-of-the-art algorithms, with ablation studies and robustness analyses further validating its effectiveness and reliability for real-world large-scale graphs. 

---
# Learning 4D Geometric Priors for Inference-Efficient World Action Models 

**Authors**: Jianjun Zhang, Jian Zhu, Taiyi Su, Chong Ma, Zitai Huang, Yi Xu, Hanli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05468)  

**Abstract**: World Action Models (WAMs) have shown strong potential for robotic manipulation by jointly modeling visual future dynamics and executable action sequences. However, existing video-action co-training methods primarily optimize appearance-oriented video latents, which may insufficiently capture the temporally evolving geometry required for precise manipulation. We propose MECo-WAM, a Multi-Expert Co-Training World Action Model that injects action-relevant 4D geometric priors into video-action representations while preserving the original lightweight inference graph. During training, MECo-WAM combines video and action experts with a lightweight 4D expert supervised by relational targets from a frozen VGGT encoder. Asymmetric expert visibility prevents non-causal shortcuts from auxiliary geometry to action generation. To transfer geometric knowledge into the deployed video-action pathway, we introduce decayed 4D read-mask attention, which provides restricted current-frame geometric guidance early in training and progressively removes this dependency. We further propose action-aware temporal geometric distillation, which aligns within-frame geometric relations and their temporal evolution while emphasizing visual regions most relevant to robot actions. At deployment, all auxiliary 4D components are removed. Experiments on LIBERO (98.2%), RoboTwin 2.0 (92.6%), and challenging real-world manipulation tasks show that MECo-WAM improves manipulation performance without increasing inference cost. 

---
# CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration 

**Authors**: Hairui Zhu, Yiying Yang, Tengjin Weng, Ziyu Lu, Xiao Yao, Xiaoyang Ye, Lin Ma, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05465)  

**Abstract**: Complex image creation and editing often require more than a single generation or editing model. A user request may involve synthesizing images, localizing objects, segmenting regions, editing selected content, compositing intermediate assets, reading text, and enhancing the final result. Such tasks shift multimodal agents from perception-augmented reasoning to manipulation-centered visual creation, where tools must actively transform visual states rather than merely inspect them. However, existing multimodal tool-use agents are mostly optimized for perception, search, or domain-specific editing, and lack large-scale supervision for executable image-creation trajectories. In this paper, we introduce CanvasCraft, a large-scale multimodal tool-use dataset for complex image creation and editing, and \textbf{CanvasAgent}, a tool-augmented multimodal agent that learns to orchestrate heterogeneous visual tools through multi-turn interaction. CanvasCraft contains 140K fully annotated executable trajectories and 10K
RL task specifications. CanvasAgent is first trained with SFT to learn executable reasoning-action trajectories, and is then optimized with GRPO using a hybrid reward that combines outcome- and process-level signals. During rollout, CanvasAgent inspects intermediate results, tracks visual assets, and adapts tool decisions to the evolving visual state. Experiments evaluate both final image quality and trajectory behavior, demonstrating the effectiveness of CanvasAgent and the proposed dataset for complex multi-tool image creation workflows. 

---
# Learnable Weighting of Intra-Attribute Distances for Categorical Data Clustering with Nominal and Ordinal Attributes 

**Authors**: Yiqun Zhang, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2607.05464)  

**Abstract**: The success of categorical data clustering generally much relies on the distance metric that measures the dissimilarity degree between two objects. However, most of the existing clustering methods treat the two categorical subtypes, i.e. nominal and ordinal attributes, in the same way when calculating the dissimilarity without considering the relative order information of the ordinal values. Moreover, there would exist interdependence among the nominal and ordinal attributes, which is worth exploring for indicating the dissimilarity. This paper will therefore study the intrinsic difference and connection of nominal and ordinal attribute values from a perspective akin to the graph. Accordingly, we propose a novel distance metric to measure the intra-attribute distances of nominal and ordinal attributes in a unified way, meanwhile preserving the order relationship among ordinal values. Subsequently, we propose a new clustering algorithm to make the learning of intra-attribute distance weights and partitions of data objects into a single learning paradigm rather than two separate steps, whereby circumventing a suboptimal solution. Experiments show the efficacy of the proposed algorithm in comparison with the existing counterparts. 

---
# Evaluating calibrated refusal and safe usefulness in dual-use biology settings 

**Authors**: Edwin H. Wintermute, Harmon Bhasin, Christina M. Agapakis, Dianzhuo Wang, Evan Seeyave, Arjun Banerjee, Daniel Fulop, Matthew C. Watson, Adam J. Meyer, Sandrine Boissel, Jens H. Kuhn, Rishi Jain, Noah D. Taylor, Helena Shomar, Patrick M. Boyle, Kenny Workman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05462)  

**Abstract**: As AI agents are incorporated into life science workflows, the capabilities that speed discovery might also enable misuse. We present BioSecBench-Refusal, a benchmark for risk identification and refusal behavior for biological research tasks. The benchmark pairs 61 Routine tasks, legitimate analyses adapted from the published literature, with 46 Red-Team tasks, fictional scenarios that resemble real research but conceal a biosecurity hazard. Across 16 model-harness configurations, refusal rates ranged from 7\% to 74\% on Routine tasks and 1\% to 62\% on Red-Team tasks, with many configurations refusing legitimate Routine work at comparable or higher rates than concealed hazards. Refusals were most often triggered by provider API filters applied prior to agentic reasoning. However, models given room to reason showed the potential to identify more real threats. We release BioSecBench-Refusal as a tool for model developers to calibrate capability and caution for agentic biotech R\&D. 

---
# AdaStop: Cost-Aware Early Stopping for DNN Test Selection 

**Authors**: Bonan Shen, Wei-Jung Huang, Xin Liu, Jiazhou Gao, Tao Ning  

**Link**: [PDF](https://arxiv.org/pdf/2607.05461)  

**Abstract**: Existing methods for testing deep neural networks (DNNs) primarily prioritize test inputs likely to reveal model faults under a fixed labeling budget. In practice, choosing that budget is difficult: too little testing misses failures, while too much incurs unnecessary labeling costs. This work studies the stopping problem in DNN testing. We formulate testing as a cost--benefit decision process in which labeling an input incurs cost $c$ and discovering a fault yields value $v$. Based on this formulation, we introduce \textit{AdaStop}, a framework that estimates the marginal fault discovery rate during testing and stops labeling when the estimated rate falls below the threshold $\tau = c/v$. Experiments across multiple datasets, architectures, and selection strategies show that $65$--$84\%$ of faults can be discovered using only $9$--$31\%$ of the labeling budget. 

---
# Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning 

**Authors**: Haiwen Yi, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.05458)  

**Abstract**: Large language model (LLM) agents are usually improved by changing prompts, models, or hand-written workflows, while the execution harness around the model is treated as fixed infrastructure. We argue that this harness is itself a learnable control layer. We formalize harness operation as a finite-horizon Harness MDP, where a lightweight controller selects structural execution actions while the LLM executor remains frozen. The controller is trained from offline rollouts using advantage-weighted regression with only terminal task-rubric rewards. We also separate final task quality from a post-hoc Harness Maturity Score, which measures whether the harness follows reliable execution patterns rather than only whether the final answer is correct. This separation gives a finite-buffer view of harness learning: final-quality gains require high-return support in the offline buffer, while process behavior can shift whenever it aligns with advantage-weighted actions. Across six controlled domains and two public-benchmark adapters, the learned controller consistently improves verification behavior and selectively improves final task quality, with the largest gains on adapted tau-bench retail, adapted AgentBench DB-Bench, and coding with a calibrated structural verifier. Ablations against behavior cloning and Forced CHECK show that the gains are not explained by imitation or by simply adding checks. These results identify harness control as a learnable layer for frozen LLM agents, while showing that offline support limits when better process control becomes better final answers. 

---
# Empirical Minimal-Realisation Compression of Deep Neural Networks via Controllability-Observability Tests 

**Authors**: Anis Hamadouche, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2607.05457)  

**Abstract**: Deep neural networks often contain substantial hidden-state redundancy, but most compression methods operate directly on weights, neurons, or quantised representations without explicitly characterising the dynamical role of internal states. This paper proposes a controllability-observability framework for empirical state-order reduction of deep neural networks. By viewing a trained network as a depth-indexed nonlinear dynamical system, we construct data-driven reachability, observability, and balanced Gramians from hidden-state snapshots and output Jacobians. The resulting A/B/C tests estimate layer-wise reachable, observable, and jointly reachable--observable ranks. These ranks are then used not only as diagnostic measures of hidden-state redundancy, but also as actual compressed layer widths for realised reduced networks. Experiments on MNIST and CIFAR-10 compare the proposed balanced realisation against projection-based reduction, unstructured pruning, structured pruning, low-rank SVD, dynamic INT8 quantisation, and linear baselines. On MNIST, a four-layer SiLU DNN is reduced from state order 1024 to 277, giving 72.95% state compression and 73.48% parameter compression, while maintaining 95.45% accuracy compared with 96.60% for the full model. On CIFAR-10, a larger SiLU DNN is reduced from state order 4608 to 1339, giving 70.94% state compression and 83.09% parameter compression, while preserving accuracy from 54.45% to 54.44% and reducing CUDA inference latency by approximately 3X. The results show that balanced reachable-observable ranks provide a principled empirical minimal-realisation criterion for designing compact neural architectures with little or no loss in accuracy. 

---
# The Granularity Paradox: How Temporal Disaggregation Inflates In-Sample Fit and Compounds Out-of-Sample Error 

**Authors**: Hugo Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2607.05450)  

**Abstract**: This paper explores the "Granularity Paradox" in time-series forecasting, wherein finer temporal disaggregation (e.g., Monthly to Weekly/Daily) improves in-sample diagnostics and dataset size (N), but degrades out-of-sample accuracy due to recursive error compounding over longer horizons (H). Conversely, coarse aggregation (Annual) eliminates recursive error propagation but reduces data available to estimators. We formalize this trade-off and benchmark 10 models - spanning naïve, statistical, machine learning, and deep learning architectures - across six granularities using a 13-year public procurement dataset. The empirical results reveal a non-monotonic threshold structure: recursive autoregressive and seasonal models degrade substantially under high-frequency forecasting (e.g., Holt-Winters reaches a Test R-squared of -151 and TPFE of 425.85% at the Daily grain), while the LSTM traces a U-shaped error curve, worsening from Monthly (19.66%) through Bi-Weekly (35.94%) before overcoming the error propagation penalty at Daily (TPFE of 4.35%, R-squared of 0.66). Linear Regression remains stable across all granularities (16.3-17.0% TPFE), confirming that the paradox is driven by recursive feedback topology, not model complexity. The results demonstrate that standard pointwise metrics (RMSE, MAE) systematically mask cumulative error propagation, and that evaluating forecasts without goal-dependent cumulative metrics produces misleading assessments of model adequacy. We introduce a consensus-dissensus diagnostic comparing the directional behaviour of pointwise metrics against cumulative TPFE across granularities, enabling the identification of models whose standard diagnostics mask systematic error propagation. 

---
# Geometry-Aware Infrastructure-Anchored Denoiser for UWB Sensing and Work-Zone Reconstruction 

**Authors**: Weizhe Tang, Jiaxi Liu, Junwei you, Steven T.Parker, Pei Li, Sikai Chen, Meng Ran, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2607.05449)  

**Abstract**: Accurate work-zone geometry perception is critical for intelligent transportation systems, and ultra-wideband sensing offers a low-cost approach for infrastructure-aided reconstruction. However, outdoor UWB ranging is often degraded by non-line-of-sight propagation, burst noise, and long-tail errors, which can distort downstream spatial reconstruction. We present GAIA, a geometry-aware, infrastructure-anchored learning framework that couples temporal range modeling with latent anchor-layout estimation and deterministic distance projection. GAIA preserves range denoising as the supervised task while orienting the learned distances toward boundary-consistent reconstruction. We evaluate GAIA on a real-world outdoor UWB dataset with synchronized UWB, GNSS, and IMU measurements, and further test robustness using a real-data-calibrated stress-test simulator. GAIA achieves the lowest overall range MSE and highest polygon IoU among evaluated filtering-based and learning-based baselines, reducing MSE by 18.4% and improving polygon IoU by 15.5% over PoseMLP. These results show that geometry-aware range denoising provides an effective path toward spatially coherent work-zone reconstruction. 

---
# Scientific Code Search at Scale: A Multi-Domain Dataset and Benchmark 

**Authors**: Nishan Pantha, Pranath Reddy Kumbam, Sajil Awale, Pushwitha Krishnappa, Muthukumaran Ramasubramanian, Nidhi Jha, Emily Foshee, Ankur Kumar, Rachel Slank, Ashkbiz Danehkar, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.05443)  

**Abstract**: Scientists increasingly rely on open-source tools to support their research workflows, yet discovering relevant software among over 600 million GitHub repositories remains challenging. Existing code search benchmarks focus on general software engineering tasks and fail to capture the domain-specific vocabulary and needs of scientific computing. We present a curated corpus of 5,264 high-quality, domain-classified scientific repositories spanning five NASA Science Mission Directorate divisions -- Earth Science, Astrophysics, Planetary Science, Heliophysics, and Biological & Physical Sciences -- enriched with cleaned READMEs, extracted topics, and additional context from crawled links. Building on this corpus, we introduce two novel information retrieval benchmarks: (1) a repository search benchmark with 219 expert-curated queries designed by domain scientists, and (2) a large-scale code snippet retrieval benchmark containing 117,950 code snippets and 119,720 queries across seven programming languages. Baseline evaluations on repository search reveal significant performance variation across scientific domains. Code snippet retrieval proves equally challenging, with substantial variation driven by differing documentation practices, coding standards, and programming language conventions across scientific communities. All datasets and benchmarks are publicly released on HuggingFace to support research on scientific tool discovery. 

---
# PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models 

**Authors**: Lorenzo Molfetta, Giacomo Frisoni, Nicolò Monaldini, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2607.05441)  

**Abstract**: Integrating external tools with Large Language Models (LLMs) has emerged as a promising paradigm for accomplishing complex tasks. Since LLMs still struggle to effectively manage large tool collections, researchers have begun exploring retrieval-based methods to pre-select the most relevant options, addressing input length and latency constraints. However, existing retrievers are often misaligned with tool-calling LLMs due to their separate training processes. This paper presents PORTS, a novel odds ratio preference optimization method for training retrievers aimed at tool selection. Using a perplexity-inspired preference signal from a frozen LLM, our approach fine-tunes a retriever to find helpful tools by optimizing the correlation between the selection probabilities and the downstream performances while jointly enforcing a contrastive semantic loss between documentation strings. The versatility of PORTS and its ability to significantly improve tool selection accuracy are demonstrated through extensive experiments on six datasets, two encoder models, and three LLMs with diverse prior knowledge. With low computational demands, our alignment process facilitates generalization to new queries and tools, proving valuable for practical applications with evolving toolsets. 

---
# Modality Relevance is not Modality Utility: Post-hoc Selective Modality Escalation for Cost-Aware Multimodal RAG 

**Authors**: Xue Li, Yiming Gai  

**Link**: [PDF](https://arxiv.org/pdf/2607.05438)  

**Abstract**: Multimodal retrieval-augmented generation (RAG) grounds a generator in evidence drawn from heterogeneous modalities -- text, tables, and images. The dominant deployment choice is binary and made before the model has tried to answer: either run a cheap text(+table) pipeline, or pay for an expensive vision-language model (VLM) over every image. Recent adaptive systems improve on this by selecting the modality or fidelity pre-retrieval, from a question-conditioned predictor of which modality will be needed. We show that this is the wrong decision point. Through an oracle headroom analysis on MultiModalQA, we find that the relevance of a modality to a question is a weak predictor of whether that modality is actually needed to answer correctly: a large fraction of questions whose gold support includes an image are nonetheless answerable from text and tables alone, and a pre-retrieval router that escalates on apparent visual relevance over-escalates substantially relative to an oracle. We propose \textbf{post-hoc selective modality escalation}: answer cheaply from text and tables, run a verifier on the (query, draft answer, evidence) tuple that localizes which modality is missing, and pay for VLM evidence only there. A calibrated value-of-escalation router then decides whether the expected accuracy gain justifies the visual cost. On MultiModalQA, our router recovers the accuracy of an always-on VLM pipeline while issuing far fewer visual calls, and closes most of the gap to the oracle escalation rate. The result extends a routing-signal hierarchy established for retrieval depth and reasoning hops to a third axis -- modality -- under a single cost-aware selective-escalation view. 

---
# CHARLIE: An On-Premise Multi-Agent Retrieval-Augmented Generation System for Evidential Reasoning in Forensic Science 

**Authors**: Leandro D. Carneiro, Andre L. S. Meirelles, Juliano de A. Gomes, Rafael C. A. Cabral  

**Link**: [PDF](https://arxiv.org/pdf/2607.05428)  

**Abstract**: We present Charlie, an on-premise multi-agent Retrieval-Augmented Generation (RAG) system for structured evidential processing in digital forensic environments. Contemporary forensic workflows must handle large volumes of heterogeneous and unstructured documents under strict requirements of traceability, confidentiality, and legal compliance. Charlie addresses this challenge through a controlled agent architecture that combines local retrieval, task decomposition, structured memory, and verification mechanisms. Unlike cloud-based systems, it operates entirely within institutional infrastructure, preserving data sovereignty and evidential integrity. We describe the systems architecture, including its transition from classical RAG to agent-based orchestration, and demonstrate its application in real-world forensic scenarios. Case studies show that Charlie enables scalable multi-document data extraction and supports longitudinal forensic intelligence generation while maintaining traceability and auditability. Our results indicate that agent-orchestrated, on-premise RAG architectures can effectively support evidential workflows without compromising legal and institutional constraints. Charlie provides a practical and reproducible blueprint for deploying AI systems in high-stakes forensic environments. This manuscript is an archival version of a paper presented at the RELAF 2026 Workshop. 

---
# When AI Classifies: What Counts as Public Administration? 

**Authors**: Shaoming Cheng, Laurie Schintler  

**Link**: [PDF](https://arxiv.org/pdf/2607.05420)  

**Abstract**: This study examines how alternative systems of scholarly representation identify and characterize broad public administration (PA) and artificial intelligence related public administration (AI-in-PA) scholarship. Using Web of Science and OpenAlex, it compares five approaches based on author-defined, citation-driven, and AI-assisted representations. The results highlight substantial differences in corpus size, publication types, publishing outlets, temporal development, and thematic clustering and structure. The alternative approaches often identify different knowledge domains instead of varied subsets of the same scholarship and therefore produce distinct representations, as evidenced by no overlap in publications and publishing outlets across representations. The findings suggest that algorithmic knowledge organization increasingly influences how interdisciplinary scholarship is classified, structured, and understood and, epistemologically, how its visibility, intellectual structure, and boundaries are represented. AI-enabled scholarly classifications and representations are not neutral but interpretative, likely self-reinforcing, and potentially constrain the evolution and adaptation of disciplinary boundaries. Human disciplinary judgment is essential and is complemented rather than replaced. 

---
# Contrastive Predictive Coding with Compression for Enhanced Channel State Feedback in Wireless Networks 

**Authors**: Ahmed Y. Radwan, Hina Tabassum, Fahad Syed Muhammad, Matthew Baker  

**Link**: [PDF](https://arxiv.org/pdf/2607.05419)  

**Abstract**: Accurate and timely channel state information (CSI) is essential for next-generation wireless systems, yet existing works treat CSI compression and CSI prediction as separate problems, both in academia and in current 3GPP studies. Consequently, channel aging remains insufficiently addressed within standardized CSI feedback pipelines. In this article, we propose a unified compression-prediction framework that integrates Contrastive Predictive Coding (CPC) directly into the 3GPP-compliant CSI compression architecture. Instead of predicting high-dimensional CSI matrices, our approach forecasts future latent representations and jointly optimizes reconstruction fidelity and temporal predictive coherence via a combined 1-SGCS and InfoNCE objective. This design enables temporal representation learning without increasing feedback overhead. We present two variants: CPC-before-Compression, which performs autoregressive modeling on encoded features prior to quantization, and CPC-after-Compression, which shifts temporal modeling to the base-station to reduce the complexity of users' devices. Evaluations on 3GPP-compliant datasets from Nokia, Oppo, and CATT show that CPC-before-Compression achieves over 90% reconstruction accuracy with 32x lower decoder GFLOPs than the 3GPP baseline, while CPC-after-Compression preserves an identical encoder footprint and the same 64-bit feedback overhead. By unifying compression and prediction within a standardized pipeline, the proposed framework provides an age-aware, computationally efficient CSI feedback solution. The source code is publicly available at: this https URL 

---
# Why does AI unlock new possibilities in STEM education? A Bibliometric Analysis of Trends and Future Agenda 

**Authors**: Jesse Yusuf Chan, Mengyao Chen, Yang Hong, Ziyun Song, Haoming Wang, Xianlong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05412)  

**Abstract**: STEM education faces challenges in personalization and interdisciplinary integration. AI technology has brought new possibilities, but the mechanisms by which AI reshapes the STEM education ecosystem require systematic investigation. This study employs bibliometric methods to analyze 242 publications from 2015-2025, constructing knowledge maps to reveal the evolutionary trajectory. The findings show that the field has transformed from intelligent tutoring systems to inquiry-based learning and computational thinking cultivation driven by LLMs. AI's key contribution lies in providing intelligent scaffolding that lowers the threshold for understanding knowledge. In this sense, AI is a core driving force promoting its shift from knowledge transmission to capability development. 

---
# The GenAI Skill Bypass: Mapping Divergent Pathways of University Students and Staff AI Literacy 

**Authors**: Eduardo Oliveira, Narelle English, Tracii Ryan, Kamila Misiejuk, Cory dal Ponte, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2607.05411)  

**Abstract**: Higher education institutions are increasingly expected to ensure that both students and staff develop Generative AI (GenAI) literacies. In response, they are introducing professional development programs and embedding GenAI skills within student curricula. However, current educational frameworks typically assume a linear progression of GenAI literacy, implying that foundational technical understanding must precede creative application. This paper challenges such an assumption through a psychometric analysis of a taxonomy-based self-assessment instrument (n = 158). We applied Rasch measurement theory and Guttman ordering to map the latent perceived order of difficulty of GenAI skills across students, academics, and professional staff. Results reveal a fundamental divergence in perceived competence profiles: while academics follow a more traditional linear path, students exhibit an "inverted" profile, frequently mastering high-level creation tasks before acquiring foundational conceptual understanding. Furthermore, the correlation of skill difficulty between students and academics was weak (r = 0.188). We argue that this "skill bypass" creates a fragile sense of fluency, where high self-efficacy in prompting masks low literacy in AI mechanics. These findings challenge the "one-size-fits-all" curricula and provide the empirical basis for diagnostic-driven, modular interventions that foster genuine human-AI synergy. 

---
# CANONIC: Governance Is Compilation 

**Authors**: Dexter Hadley  

**Link**: [PDF](https://arxiv.org/pdf/2607.05410)  

**Abstract**: We present CANONIC: governed intelligence that compiles digital artifacts into an evidence ledger at scale. Large language models generate prose faster than anyone can check it, the failure Oxford Languages named 'slop', its 2025 Word of the Year. CANONIC governs whether content may enter a corpus the way a compiler decides whether a program is well-formed: mechanically, by a grammar, at the boundary of admission. Governance reduces to three axioms (Triad, Inheritance, Introspection) that map one-to-one onto compiler theory's syntax, scope-resolution, and type-system layers, and admission is a decidable, linear-time check. We then ask, with a pre-registered cross-provider benchmark across four regimes, whether structural admission keeps slop out. It does not: no prose-reading gate reliably separates reliable from unreliable content. Slop is not a property an algorithm computes. It is a verdict of domain expertise. So a governance layer does not decide slop; it keeps the record auditable -- every claim anchored to a definition, a commit, and an evidence window, reproducible and checkable end to end. 

---
# Automated Recommendation of Programming Learning Content Using Pattern-based Knowledge Components 

**Authors**: Muntasir Hoq, Griffin Pitts, Zhangqi Duan, Arun Balajiee Lekshmi Narayanan, Mohammad Hassany, Andrew Lan, Peter Brusilovsky, Bita Akram  

**Link**: [PDF](https://arxiv.org/pdf/2607.05409)  

**Abstract**: Introductory programming instruction relies on hands-on practice and short learning activities to support mastery of foundational concepts. Although many such learning resources exist, organizing and linking these items in instructionally meaningful ways is challenging without time-intensive expert curation. This study investigates the use of pattern-based Knowledge Components (KCs) to automatically identify code-based learning resources targeting similar concepts. In our approach, pattern-based KCs are extracted from each code sample, and related activities are identified by measuring similarity between the KC sets associated with each activity. By leveraging alignment at the level of semantically important programming patterns, this method supports contextually appropriate and pedagogically useful recommendations. We evaluate our approach on an expert-organized corpus of introductory Python materials in which instructors grouped items into bundles based on conceptual similarity. Results show that our pattern-based KC approach retrieves resources that align with this expert organization, and outperformed representative KC- and embedding-based baselines across standard ranking evaluations. Overall, the framework supports targeted, concept-oriented guidance for programming learners and can help instructors organize, bundle, and recommend instructional content at scale. 

---
# Position: Preventing AI-Generated CSAM Necessitates New Approaches to AI Safety 

**Authors**: Neil Kale, Rebecca Portnoff, Pratiksha Thaker, Michael Simpson, Robertson Wang, Kevin Kuo, Chhavi Yadav, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2607.05407)  

**Abstract**: Modern artificial intelligence (AI) systems present profound new risks to child safety. AI is increasingly being misused to create AI-generated child sexual abuse material, facilitate child sexual exploitation, and reduce barriers to harm. In this paper, we argue that protecting children from AI-facilitated sexual abuse requires new approaches to AI safety. Existing safety techniques assume data accessibility, transparency, and evaluation practices that are incompatible with the ethical and legal constraints surrounding child sexual abuse material. We examine how these constraints create new technical challenges, such as limitations on dataset auditing, red teaming, and fine-tuning prevention. In turn, we outline *15 open problems* in online child sexual exploitation and abuse across the AI development lifecycle, from dataset curation and model design to deployment and long-term maintenance. We propose targeted recommendations for researchers, developers, and policymakers to bridge the gap between theoretical AI safety and the realities of child protection. Our work aims to reframe preventing AI-facilitated child sexual abuse as a central, safety-critical dimension for AI research, motivating work that translates responsible AI principles into concrete safeguards against the exploitation of children. 

---
# A Guiding Framework for K-12 Teachers in Creating AI-powered Learning Technologies through Vibe Coding 

**Authors**: Yukyeong Song, Seoyeon Choi, Jinhee Kim, Yeongje Kim, Lauren Weisberg, Jewoong Moon  

**Link**: [PDF](https://arxiv.org/pdf/2607.05406)  

**Abstract**: Large language models generate code from natural language prompts, enabling "vibe coding," which allows non-programmers to develop computational solutions. Vibe coding for teachers amplifies the value of teachers-as-designers, improving technology integration while fostering AI literacy. However, structured guidance on supporting this process is lacking. We propose GAIDE (A Guiding Framework for AI-Integrated Design for Educators), a framework that supports K-12 teachers in creating AI-powered learning technologies through vibe coding. The initial framework, built on Design Thinking and INTERACT, was validated through a CORDTRA interaction analysis of three teachers and four faculty mentors in an eight-week workshop to derive the final framework. Additionally, the qualitative analysis of pre- and post-interviews found an enhancement of teachers' AI literacy. Findings highlight the potential of learning-by-creating for professional development. 

---
# CCBENCH: Assessing LLM Cultural Competence via Implicitly Signaled Norms using Health Queries 

**Authors**: Vasudha Varadarajan, Akhila Yerukola, Mona T. Diab, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2607.05405)  

**Abstract**: To interact with users fairly and without stereotyping, AI models must display cultural competency, i.e., the ability to infer and adapt to a user's implicitly signaled cultural values, rather than relying on static demographic traits. We introduce CCBENCH, a framework for evaluating cultural competency in large language models (LLMs), treating culture as a continuum of norm adherence states rather than as a binary state of cultural belongingness. As a case study on health, we create CCBENCH-Health, which includes 60 theoretically grounded personas exhibiting varied norm-adherence states across six cultures, each engaging in 18 realistic dialogues. Each persona is evaluated on 52 authentic healthcare questions drawn from real user forums, yielding 3,120 unique interactions. Benchmarking five leading models reveals that even the best achieve culturally appropriate responses only 20-30% of the time. When explicitly prompted to focus on culturally relevant cues from the conversational history (CoT), performance improves modestly by 3-5% on average. We find that models perform best when personas avoid cultural norms rather than follow them, revealing a persistent asymmetry, suggesting a preference in the models to align with built-in biases than adapt to cultural cues. This is especially observed in the Afghan context (Avg: 8.8%), where cultural cues rarely yield appropriate health advice. Finally, we find that models sometimes adapt more readily to implicit, cultural conversational styles than to explicitly stated cultural practices, though this varies across cultures. 

---
# The Jagged Global Economy: Frontier AI Unevenly Exposes National Economies 

**Authors**: Arul Murugan, Tomás Aguirre, Abhishek Nagaraj, Rishi Bommasani  

**Link**: [PDF](https://arxiv.org/pdf/2607.05404)  

**Abstract**: Frontier AI's labor-market effects matter to workers, firms, and policymakers, but current evidence generally comes from a handful of high-income economies. The capabilities of frontier AI are jagged across work tasks and national economies diverge in how they allocate human labor. We introduce a national AI exposure metric that combines occupation-level exposure scores and international employment data for 141 countries. We find that high income countries are substantially more exposed than low income countries and that Europe and Central Asia are 50 percent more exposed than Sub-Saharan Africa. We also find a gender gap: women are more exposed than men in 91 percent of countries, driven by their concentration in white-collar and sales occupations. The exceptions are countries where women's employment remains concentrated in agriculture and household enterprises. We validate our national AI exposure estimates by showing they predict national AI adoption statistics published by Anthropic, Microsoft, and OpenAI. Beyond direct exposure, we identify a new mechanism for indirect exposure due to cross-country income dependencies. Some nations such as Tajikistan depend heavily on foreign workers remitting money back to their home countries: Tajikistan's direct exposure to frontier AI is below-average but because 37 percent of Tajikistan GDP is Russian remittance and Russia is very exposed, Tajikistan's remittance-accounted exposure becomes above-average. Our research shows that national variation in exposure is large enough that policy responses calibrated to U.S. or European labor markets will not generalize. 

---
# AI tools in Arab University English classrooms: Looking back and forward 

**Authors**: Mohammed Q. Shormani, Muneef Yehia Alshawsh  

**Link**: [PDF](https://arxiv.org/pdf/2607.05403)  

**Abstract**: This paper aims to synthesize empirical research on AI tools used to support English as a second/foreign language (EL2) learners in Arab University classrooms (AUCs) between Jan 1st 2023 and Aug 31st 2025. We utilized 3 large datasets, namely Google Scholar, Web of Science, and Scopus as the data sources. Using PRISMA-guided searches across these well-known databases, we included only published articles. The search process results in 184 studies, but only 11 studies met the inclusion criteria. Findings unveil that EL2 learners have positive attitudes towards AI for drafting, revision, and practice. Empirical gains were most consistent for surface-level outcomes improvements in higher-order writing quality and speaking proficiency was mixed and often contingent on teacher mediation. The paper concludes by proposing a research agenda and practical guidelines for Arab universities seeking evidence-based AI integration in EL2 instruction. It also recommends scaffolded integration, teacher training, reflective tasks to reduce over-reliance on AI tools. 

---
# Catalyst Papers in Artificial Intelligence Research: A Landscape on ICLR from 2017 to 2025 

**Authors**: Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05401)  

**Abstract**: A small number of methodological contributions, including word2vec, the Transformer, large-scale pre-training, and reinforcement learning from human feedback, have reshaped NLP and AI research over the past decade. OpenReview now makes numeric reviewer scores and accept/reject decisions public for every ICLR submission. Whether such review signals identify trajectory-changing papers at submission time, however, remains untested at corpus scale. We answer this question on $36{,}113$ papers from ICLR 2017--2025, identifying \emph{catalysts}: papers whose descendants measurably redirect future research. We compare four disruptiveness measures (the Consolidation/Destabilization (CD) index, node2vec, the direction-aware Embedding Disruptiveness Measure (EDM), and an LLM-based semantic rater) and define a five-type operational catalyst taxonomy (topic initiator, topic bridge, within-topic redirector, simultaneous, and recognition-misaligned). EDM leads at identifying highly cited ICLR papers (AUC $0.83$ vs.\ $0.60$ for CD, $0.49$ for node2vec, and $0.42$ for the LLM rater). Topic initiators precede a $7.55{\times}$ topic-share growth and topic bridges precede an $11.52{\times}$ growth in cross-topic citation flow versus year-matched controls. We found that the peer review scores are essentially orthogonal to future disruptiveness ($|\rho|{\leq}0.005$; accepted and rejected papers have indistinguishable mean EDM, $p{=}0.11$). 

---
# Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving 

**Authors**: Nikita Agrawal, Ruben Mayer  

**Link**: [PDF](https://arxiv.org/pdf/2607.05399)  

**Abstract**: Large language model serving is increasingly limited by KV-cache growth under long-context workloads, yet existing KV-cache compression techniques are difficult to compare because they were evaluated on different models, tasks, budgets, and serving stacks. This paper presents a workload-aware benchmark of representative KV-cache optimization mechanisms spanning quantization, pruning, and merging, including KIVI, TurboQuant, SnapKV, and CaM, evaluated on LongBench-style multi-document QA, single-document QA, few-shot learning, and summarization workloads using Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. The benchmark measures task quality, mean output throughput, mean time-to-first-token, and realized compression ratio across context-length buckets. The results show that the compression ratio alone is a poor predictor of end-to-end performance. KIVI4 provides the most stable quality across models, SnapKV delivers the strongest long-context throughput, and CaM yields large gains on selected QA workloads but exhibits substantial workload sensitivity in both quality and realized compression ratio. These findings motivate workload-aware selection of KV-cache mechanisms rather than one-size-fits-all compression and provide deployment guidance for long-context serving systems. 

---
# Proof of Execution: Runtime Verification for Governed AI Agent Actions 

**Authors**: James Rhodes, George Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05397)  

**Abstract**: Agent systems increasingly execute rather than advise. When an AI agent queries regulated data, invokes effectful tools, and mutates persistent state, correctness is not captured by whether a terminal output looks plausible. The operative questions are whether each step was authorized under a contract, whether the recorded history is tamper-evident, and whether the trajectory can be reconstructed deterministically. We formalize this as runtime proof of execution. An execution is a triple $x = (C, T, R)$: a contract $C$, an Execution Causal Event Stream (ECES) $T$, and a replay context $R$. A well-formedness predicate and five validator-checkable invariants form the PoE validity predicate. Five semantic guarantees describe authorization, path compliance, null effect on deny, history integrity, and replayability. We prove soundness under explicit cryptographic and deployment assumptions: any PPT adversary that produces a PoE-valid execution violating a semantic guarantee yields a signature forgery, a hash collision, or a quantified deployment-failure event. The Prime Execution Model (PEM) separates planning, enforcement, effect, and recordkeeping into distinct authority planes; a lemma reduces trace completeness to Effector-exclusive credentialing. An Execution Attestation Certificate is issued only when PoE = 1. In a single-node TypeScript prototype, PoE adds approximately 2.7 ms on a minimal flow and 4.4% overhead on concurrent batch workloads; a standard eight-event trace compresses to approximately 1.1 KB; injected Gateway-bypass and trace-mutation attacks are rejected. PoE does not replace consensus, TEEs, or zkVMs; it binds authorization, effect, history, and replay into a single runtime-checkable object so that governed execution becomes attestable under contract. 

---
# KVpop -- Key-Value Cache Compression with Predictive Online Pruning 

**Authors**: Lukas Hauzenberger, Niklas Schmidinger, Anamaria-Roberta Hartl, David Stap, Thomas Schmied, Sebastian Böck, Günter Klambauer, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2607.05061)  

**Abstract**: Key-value (KV) cache growth is a major bottleneck in autoregressive decoding, as memory and bandwidth scale linearly with context length. Existing KV eviction methods often rely on static heuristics or proxy scores, which poorly track future token utility and cause brittle eviction as relevance shifts. To address this, we introduce KVpop, which learns a fixed-budget KV eviction policy by directly supervising the keep-or-drop decision. The scorer is trained against a novel future-attention target, computed efficiently without materializing dense attention maps. We further introduce a delayed memory-based scorer that, uniquely among learned eviction methods, defers scoring for a fixed number of steps to exploit near-future context. On AIME and HMMT mathematical reasoning, KVpop retains 98% of full-attention performance on Qwen3-4B at 75% KV cache compression and 97% at 88% compression, consistently outperforming established eviction baselines. Qwen3-8B shows even stronger results, reaching near-full teacher performance. These results show that supervising eviction with future-attention signals cuts memory costs while maintaining quality. 

---
# Lingering Authority: Revocable Resource-and-Effect Capabilities for Coding Agents 

**Authors**: Igor Santos-Grueiro  

**Link**: [PDF](https://arxiv.org/pdf/2606.22504)  

**Abstract**: Coding agents often receive broad tool access for an entire task, even when a resource is needed only for one subgoal. We call this gap lingering authority: a temporary resource/effect capability remains exposed after the episode that justified it has closed.
PORTICO is a reference monitor for revocable capabilities exposed to the planner. It compiles an explicit task contract into initial capabilities, grant rules, trusted closure predicates, and global deny rules. A request-grant-invoke lifecycle materializes expansions as opaque, epoch-bound handles. Closure removes those handles from the next planner interface and rejects stale replay before side effects. The monitor assumes mediated tools and a sound typed catalog.
In controlled coding-agent tasks, PORTICO records no executed contract-forbidden effects in the evaluated runs, while controlled grants recover boundary work blocked by a fixed narrow envelope. A non-revoking comparator receives the same initial envelope and the same grants at the same turns. On the closure slice, both systems match task success, scope compliance, and all pre-closure decisions; PORTICO then rejects 10/10 post-closure reuses, while the comparator permits 10/10. A deterministic stale-write audit records 0/6 versus 6/6 executed forbidden effects. Scripted traces and six live model traces over file writes, git mutation, and network egress show the same split. In a four-episode same-policy diagnostic, broad request exposure preserves zero executed forbidden effects but raises blocked proposals from 67 to 84. Frozen real-repository runs, with commits and traces recorded, exercise the same lifecycle on real project layouts. 

---
