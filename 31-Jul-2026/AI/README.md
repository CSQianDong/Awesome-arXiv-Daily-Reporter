# AISPA: User-Centric System Prompt Auditing for Large Language Model Applications 

**Authors**: Xiangning Lin, Shenzhe Zhu, Shu Yang, Zhenyu Zhang, Haoqian Zhang, Yipeng Zhao, Chengxuan Qian, Tianwei Wang, Ziheng Zhang, Zhenlong Yuan, Dingcheng Wang, Juncheng Wu, Yuan Si, Jiaxin Liu, Baolong Bi, Robert Mahari, Tobin South, Dazza Greenwood, Zexue He, Rishi Bommasani, Sophia Kazinnik, Andreas Haupt, Samuele Marro, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28617)  

**Abstract**: System prompts are instructions configured by developers to govern the behaviors of foundation models in AI applications. They are used throughout commercial AI products, but are rarely disclosed to the public or regulators, creating a serious trust and accountability gap in the wide deployment of AI systems. In this paper, we introduce Artificial Intelligence System Prompt Assurance (AISPA), a user-centric framework for systematically auditing system prompts in AI systems. AISPA examines specific parts of a system prompt and evaluates them along eight dimensions that matter to users. We then use this framework to review 3,249 instructions from system prompts in 88 commercial AI products, classifying each instruction as either protective (of users) or problematic. Our audit surfaces four core findings. First, system prompt design varies substantially across products and developers, with some organizations averaging over 60 protective instructions per product while others average fewer than 5. Second, protective instructions are widely adopted but shallow in scope: 98.9% of products contain at least one, yet only 24% cover all eight dimensions of the AISPA taxonomy. Third, system prompts have grown steadily longer and more protective of users, suggesting that user protection is becoming a more visible concern in commercial prompt design. Fourth, despite this progress, problematic instructions remain pervasive: roughly 40% of products contain at least one instruction that works against user interests, and protective and problematic instructions frequently coexist within the same prompt. Our findings highlight the need for greater transparency, standardization, and independent oversight for system prompts in commercial AI products. 

---
# OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models 

**Authors**: Qiushi Sun, Kanzhi Cheng, Yian Wang, Bowen Yang, Hang Yan, Liheng Chen, Fangzhi Xu, Zichen Ding, Nuo Chen, Jialin Cao, Xingdong Gong, Zehao Li, Kaiming Jin, Xinfeng Yuan, Zhoumianze Liu, Jingyang Gong, Zhangyue Yin, Jiahui Gao, Zhiyong Wu, Tianbao Xie, Jianbing Zhang, Ben Kao, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2607.28609)  

**Abstract**: Computer-using agents (CUAs) are advancing rapidly across the digital world. A CUA trajectory records the agent's actions, states, and reasoning. Verifying whether it fulfilled the task instruction is central to CUA evaluation, data curation, and reinforcement learning. Neither human-written verifiers nor human annotators can provide such verification at scale, so the field increasingly turns to vision-language models (VLMs) as judges of CUA trajectories. But a fundamental question has long gone unexamined: are these VLM judges reliable enough? To study it systematically, we introduce OSReward, a realistic, high-quality benchmark that evaluates VLM judges on CUA trajectories. The trajectories come from diverse agent backbones executing human-verified instructions across platforms, then rigorously labeled with ground-truth verdicts through multi-stage human annotation. Building on it, we derive OSReward-Hard, a challenge set concentrating genuinely hard cases, and OSReward-Multi for fine-grained efficiency and alignment scoring. The most comprehensive evaluation of VLM judges to date finds even state-of-the-art models fall short of an ideal judge, sharing a systematic leniency bias that mislabels failed runs as successes. The few reliable enough to trust are too expensive to run at scale, while affordable open models trail far behind. To close this gap, we construct and release OS-Shepherd-100K, an open corpus of reasoning-annotated trajectory judgments for the CUA community. On it, we train OS-Shepherd (9B and 35B), open reward models that supply low-cost, stable, and reliable reward signals, matching commercial judges at 30-60% lower cost than the frontier. Extensive analyses further inform the design of reliable CUA reward at scale. Our code, benchmark, dataset, and model checkpoints are available at this https URL. 

---
# DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation 

**Authors**: Jiacheng Tao, Qingyun Sun, Haonan Yuan, Ziwei Zhang, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.28580)  

**Abstract**: While Multimodal Retrieval-Augmented Generation (MM-RAG) has shown promising results, it still struggles with complex multi-hop reasoning tasks. Existing methods primarily focus on independent instance-level matching, which often fails to capture explicit relationships across modalities and documents. Although Graph-enhanced methods introduce structural modeling, they face a fundamental challenge in multimodal scenarios: incorporating fine-grained visual features leads to rapid graph expansion and retrieval noise, whereas coarse-grained representations cause the discarding of critical local evidence. To address this dilemma, we propose DualG-MRAG, a Dual-tier framework that introduces a decoupled architecture comprising Macro-reasoning and Micro-matching Graphs for Multimodal RAG. Specifically, to suppress retrieval noise by isolating global structural reasoning from fine-grained evidence matching, we construct a Macro Graph for global topological routing and a Micro Graph for precise local verification. Subsequently, to enable dynamic relevance propagation across heterogeneous evidence sources, we formulate retrieval as a query-driven message passing process via a GNN Retriever. Furthermore, to provide the generative model with coherent structural guidance, we introduce a dynamic programming decoding mechanism that extracts explicit reasoning paths directly from the GNN's forward pass, replacing the standard input of isolated document chunks. Extensive experiments demonstrate that DualG-MRAG outperforms baselines in both evidence recall and complex QA accuracy. 

---
# Rethinking Inference-Time Scaling in Local Computer-Use Agents: Failure Modes and Compute Tradeoffs 

**Authors**: Woongkyu Lee, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28573)  

**Abstract**: Deploying autonomous computer-use agents (CUAs) locally is increasingly important for privacy, cost efficiency, and practical usability, yet improving their performance under strict hardware constraints remains challenging. While recent studies show that inference-time scaling can improve frontier computer-use agents through additional computation during execution, its effectiveness for resource-constrained local models remains poorly understood. We present a systematic empirical study of inference-time scaling in local CUAs across contextual, temporal, structural, and parallel dimensions. We evaluate Qwen3-VL-8B/30B-A3B, UI-TARS-1.5-7B, and OpenCUA-7B on the OSWorld benchmark. Our results show that additional computation often yields diminishing returns while changing failure modes. Contextual scaling provides historical grounding that improves trajectory stability and task accuracy, but its gains saturate as token cost increases and failures shift from repetitive or stalled trajectories toward premature false successes. Temporal scaling similarly reduces max-step stalls, yet does not substantially improve task success, indicating that longer horizons often extend erroneous trajectories rather than correct them. We further find that structural decomposition can introduce planning and formatting overhead in local two-stage agents, while parallel scaling partially mitigates these failures at a substantial computational cost. Overall, our findings suggest that efficient local CUAs require selective compute allocation, failure-aware control mechanisms, and agentic frameworks designed around the capabilities and limitations of local models. 

---
# MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems 

**Authors**: Mao-xun Huang, Jerry Wang, Yi-Cheng Lai, Zhengxin Zhang, Claire Cardie, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28527)  

**Abstract**: Large language model-based multi-agent systems improve complex problem solving through task decomposition, agent specialization, information exchange, and intermediate validation. However, existing systems typically treat communication topology as a fixed design choice or an offline optimization target. We introduce MANTA, a framework for Multi-Agent Network Topology Adaptation that enables communication structures to self-evolve at inference time. Before execution, MANTA initializes a task-conditioned topology from prior structural experience. During deployment, it monitors collaboration traces and applies bounded structural updates when the current organization becomes insufficient. These updates can modify agent roles, communication links, execution order, information visibility, and validation pathways while preserving the task interface and agent budget. We evaluate MANTA against representative single-agent and multi-agent baselines on five benchmarks spanning information seeking, tool use, planning, workflow execution, and mathematical reasoning. MANTA achieves the highest average score of 74.0, outperforming the strongest baseline by 5.8 percentage points and obtaining the best result on PlanCraft. These results show that inference-time self-improvement can extend to the architecture of collaboration itself. 

---
# Selective Credibility-Limited Belief Update 

**Authors**: Theofanis Aravanis, Costas D. Koutras  

**Link**: [PDF](https://arxiv.org/pdf/2607.28523)  

**Abstract**: Belief update concerns changes in an agent's beliefs induced by changes in the underlying world. Standard Katsuno-Mendelzon update assumes that an epistemic input can be incorporated from every initially possible world, whereas credibility-limited belief update restricts, for each source world, the successor worlds regarded as credible or reachable. Nevertheless, existing credibility-limited approaches treat the epistemic input as an indivisible whole, and therefore cannot represent cases in which only part of a compound epistemic input can be realized. We introduce selective credibility-limited belief update, in which the epistemic input is transformed, relative to each source world, into a weaker proxy before the credibility-limited transition is performed. We provide semantic and axiomatic characterizations of the resulting class of update operators. We then identify two well-behaved sub-classes; namely, consistency-preserving update operators, which require every transformed epistemic input to be credible from its source world whenever the original epistemic input is consistent, and maximal consistency-preserving update operators, which additionally require the selected proxy to be maximally informative among the credible consequences of the original epistemic input. Finally, we establish the generality of the proposed framework by showing that credibility-limited belief update is recovered as a special case, while Katsuno--Mendelzon belief update emerges when credibility restrictions are removed and the transformation functions are taken to be identities. These results demonstrate that the framework provides a unified and strictly more expressive account of belief update, encompassing established approaches while supporting source-dependent selective acceptance. 

---
# InfoOps Bench: A live information operations safety benchmark 

**Authors**: Dorian Quelle, Lisa-Maria Neudert, Jonathan Bright, John Gallacher  

**Link**: [PDF](https://arxiv.org/pdf/2607.28503)  

**Abstract**: In this paper we present an active, constantly updated AI benchmark which measures the integrity of frontier language models against being co-opted for state-backed information operations. We draw on over 2,100 information operations from a live monitoring pipeline which tracks Russian, Chinese and Iranian state-backed information assets. Alongside this paper, we release a companion website that tracks the most prominent claims spread by state-backed media outlets, updated weekly, available from: this http URL. The dynamic nature of the benchmark makes it resistant to saturation.
In the benchmark, we test 17 models from 8 providers across four prompt framings. We find that most models can be co-opted for information operations. Integrity scores, defined as the percentage of refused requests, range from 8.8% to 94.5%, an 85.7-percentage-point spread not explained by model size. Model choice also changes the character of the resulting operation. Some models fabricate details and produce output more harmful than the source material, others defuse claims even while complying, and fact-checking rates vary from 2.9% to 72.9%.
Integrity against information operations is at least partly related to refusal to produce content even for benign claims, illustrating the challenge of balancing model usability with safety. With one exception (this http URL's GLM 5.2), the Chinese-developed models sharply cut compliance on factually grounded but China-critical claims, dropping 48-70 percentage points relative to matched benign claims. 

---
# SCOPE: Supply-Chain Operations through Coupled Policies for End-to-End Coordination 

**Authors**: Yunhao Liang, Xianqi Cao, Pujun Zhang, Yuan Qu, Yongzhi Qi, Ningxuan Kang, Max Z.J. Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28488)  

**Abstract**: Can supply-chain AI move beyond isolated decision modules toward unified operational planning? A complete replenishment plan specifies which products each location carries, which upstream facility supplies it, how often it is replenished, and how deliveries are routed. These decisions are operationally coupled: the selected assortment changes the demand and load passed to later stages; source assignment and replenishment frequency reshape the delivery requests; and route feasibility and cost, in turn, determine the system value of the earlier choices. Yet in modern supply chains, these decisions are often handled by separate departments and optimized through separate systems, which can lead to stockouts, inventory exposure, and avoidable transportation. We propose SCOPE: Supply-Chain Operations through Coupled Policies for End-to-End Coordination, a composite policy model that represents supply-chain entities as tokens, contextualizes them through a shared operational representation, and maps each token type to the corresponding decision interface. Each decision builds on the partial plan formed by earlier decisions while the completed plan is evaluated using a shared system-level utility. We instantiate this framework in urban fresh-retail replenishment, where service frequency, assortment, capacity pressure, and road-network routing interact strongly, and evaluate it on real operational data from Dingdong and this http URL, two large-scale supply chains operating at different replenishment echelons. Across both settings, SCOPE consistently outperforms methods that optimize each decision stage separately, as well as practice-oriented baselines commonly used in supply-chain operations. These results show that learning and coordinating cross-department operational couplings lead to more effective end-to-end supply-chain decisions. 

---
# A Fuzzy Rule-based Neuro-Symbolic Approach for Pipe Severity Prediction in Sewer Networks 

**Authors**: Ngoc Thai Le, Thanh Ma, Umberto Straccia  

**Link**: [PDF](https://arxiv.org/pdf/2607.28481)  

**Abstract**: Standard automated sewer pipe severity assessment relies on direct image classification, creating a "black box" where the link between visual defects and final severity scores remains implicit. This study introduces a modular, fuzzy rule-based neuro-symbolic framework that bridges this gap by decoupling neural perception from symbolic reasoning. The perception module utilizes a Swin Transformer to predict 14 multilabel inspection CODE degrees directly from images. For reasoning, a DT, specifically Weka's J48, algorithm is trained on ground-truth CODEs and severity labels, and its paths are converted into 19 fixed IF--THEN rules. Inference operates via fuzzy logic: t-norm activations from CODE conditions are weighted by rule confidence and combined with corresponding s-norms to produce interpretable class evidence. We assessed Product, Łukasiewicz, and Hamacher operator pairs using a dataset of 3,244 images spanning five highly imbalanced severity classes. Ground-truth labels were robustly generated via consensus from five independent large language models analyzing original inspector notes. Our results show an improvement of accuracy, balanced accuracy, Macro F1 and MCC by 17.9%, 12.2%, 23.0%, and 17.3%, respectively, over image-only based classification.
Overall, the framework combines competitive class-balanced performance with traceable reasoning from predicted CODE degrees to rule supports and severity evidence. 

---
# Towards Autonomous Aircraft Surveillance from Nanosatellites through On-Board Inference and Generative Data Augmentation 

**Authors**: Antonio Delgado-Rosa, David Muñoz-Valero, Enrique Adrian Villarrubia-Martin, Juan Moreno-Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2607.28470)  

**Abstract**: Airborne surveillance from low Earth orbit is hindered by two interconnected bottlenecks: nanosatellites have a limited downlink budget, yet the conventional approach still transmits terabytes of raw imagery to the ground for processing, and open satellite datasets for aircraft are scarce and severely class-imbalanced. These limitations either delay timely decision-making or prevent standard detectors from learning robust representations of rare aircraft classes. In this paper, a workflow that combines on-board inference with generative data augmentation is proposed to address both limitations jointly. Inference is executed on a 6U CubeSat equipped with a low-power edge tensor accelerator, while a diffusion model fine-tuned through low-rank adaptation generates synthetic minority-class imagery. This synthetic output is automatically annotated, pseudo-labelled, by an intermediate detector and merged with classically augmented samples. The results show that the balanced dataset increases global mean average precision from 77.9% to 82.2%, with the minority class rising from F1=0.683 to F1=0.811, and that the quantised detector fits the on-chip memory and projects 25-30 frames per second on orbit. This approach contrasts with the conventional bent-pipe architecture, in which the satellite acts as a passive data collector. Therefore, the computational tests support the proposed workflow as a decision-support tool for real-time, autonomous airborne surveillance from nanosatellites. 

---
# A report-grounded vision-language foundation model for colonoscopy from 280000 routine reports 

**Authors**: Jia Yu, Yan Zhu, Yili He, Zilong Wang, Xinyang Jiang, Peiyao Fu, Ruijie Yang, Tianyi Chen, Siyuan Li, Zhihua Wang, Fei Wu, Quanlin Li, Xian Yang, Pinghong Zhou, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28466)  

**Abstract**: Vision-language models remain underused in colonoscopy despite the rich expert descriptions recorded in routine reports. These reports document lesion appearance, size and location but summarise entire procedures rather than caption individual frames, leaving clinical findings only weakly linked to the corresponding images. Here we develop EndoCLIP, a colonoscopy vision-language foundation model trained on 125,756 lesion-level image-text pairs progressively recovered from 280,476 routine colonoscopy records. Across lesion-level image-text retrieval, structured report generation and six multi-centre clinical classification tasks, EndoCLIP outperforms general-purpose and biomedical vision-language encoders in both zero-shot and linear-probe settings. On benign-versus-malignant classification, its linear probe approaches the performance of expert readers in a blinded study involving 12 endoscopists. These results suggest that recovering finding-to-frame correspondence can transform routine documentation into scalable supervision, enabling clinical targets to be specified in language rather than separately annotated for each task. 

---
# LeanCSP: A Framework for Certifying Constraint Reformulation and Solving in Lean 

**Authors**: Pablo Manrique, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2607.28459)  

**Abstract**: Constraint programming is a core technology for solving complex combinatorial problems in scheduling, planning, configuration, and verification. Trusting its results therefore demands guarantees at two levels: that reformulations applied beforehand are semantics-preserving, and that solvers produce correct answers. In this work, we introduce a framework that addresses both verification levels in the Lean theorem prover: it can be used to prove formulation-level properties, such as equivalence, equisatisfiability, and the correctness of symmetry-breaking constraints, parametrically for entire problem families; and to check solver-produced certificates for individual instances via translation backends to external formats such as MiniZinc, SMT-LIB, and OPB. Combining both levels yields an end-to-end workflow that establishes the satisfiability or unsatisfiability of a constraint problem without trusting the external solver. Experimental results show that our framework's verified symmetry breaking also pays off in practice: a single parametric proof per problem family, reused across all instance sizes, reduces solver search effort by a factor of up to 2x10^7, while the entire in-Lean certification stays affordable, taking at most a few minutes for our largest instances. 

---
# SVR: Self-Verifying Refinement via Joint Verdict-Confidence Reinforcement Learning for Adaptive Test-Time Compute 

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28457)  

**Abstract**: Scaling test-time computation can improve language-model reasoning, but uniform budgets waste computation on easy inputs, while verifier-guided refinement relies on external feedback. We introduce Self-Verifying Refinement (SVR), an oracle-free multi-turn reinforcement learning framework that learns to use self-verification as a compute-control policy. At each turn, the model produces a solution together with a discrete correctness verdict and a confidence score; it retains the current answer only when the verdict is Correct and confidence exceeds a threshold, and otherwise continues refinement using its own self-verification. Ground-truth correctness is used only to construct training rewards and is never exposed to the policy through refinement prompts or required at inference. SVR is trained with GRPO on fixed-horizon trajectories using rewards that promote solution correctness, calibration-aware self-verification, and stop-ready correct states; adaptive stopping is activated only at inference. On seven mathematical reasoning benchmarks with Qwen3.5-2B, SVR achieves a macro-average accuracy of 0.563 with only 2.99 inference turns on average. In the evaluated complete-system comparison, it exceeds standard GRPO, strong multi-turn baselines, and a fixed-budget oracle-guided score-feedback reference while requiring substantially fewer turns than fixed ten-turn inference. These results demonstrate that learned self-verification can serve as an effective internal control signal for answer retention and adaptive test-time compute allocation. 

---
# Metaphor Tracer: A Theory-Informed Analysis of Hidden States 

**Authors**: Marc Heimann, Roxana Assadi Moghaddam, Olga Brovkina, Mark Pettifor, Lutz Goetzmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.28434)  

**Abstract**: What do a language model's hidden states say about the organization of a single text? From one forward pass, without training, we score every token position on two properties. The *aggregator* measures whether the position consolidates the whole text into a stable configuration. The *differentiator*, whether other tokens are transiently carried into its subspace as the model reads: metaphor in its root sense, transport. Constants were frozen on one discovery text; every other is confirmatory.
The aggregator is not, in the classic sense, an information measure, nor a measure of salience. Across three unrelated models, as a signifier repeats, its surprisal and its attention drain while its aggregator score holds: the channel marks a token's place in the text. That this tracks a reading rests on independent ground truth: an engineered register the aggregator follows across its boundaries (6/6 cells), and a psychoanalyst's marking of clinical transcripts, fixed before the instrument existed, in 34/36 cells, with a graded increment above lexical controls and dissociations no type-level measure reproduces. A transfer test gives the result its shape: the model whose token structure travels with lexical type reads the singular discourse worst, and in a matched base/instruct pair tuning raises fidelity without moving type-transfer. Structural value is a property of a token's place in *this* text, not of its vector alone: a relational rather than essentialist reading of hidden states, operationalizing theory that predated the instrument. 

---
# A foundation model of numerical intelligence with cross-disciplinary generalization 

**Authors**: Chenghan Wu, Zongmin Yu, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28432)  

**Abstract**: Intelligence is commonly understood as the ability to acquire and apply knowledge, adapt to unfamiliar situations and solve new problems. Large language models exhibit this capacity by inferring task-relevant knowledge from textual context and applying it to new tasks. Yet intelligence need not be confined to language. For scientific and social systems, we need models that acquire and apply knowledge from numerical context-an ability we call numerical intelligence. Here we introduce UNified In-Context Operator Networks (UNICON), a foundation model that exhibits numerical intelligence across disciplines. Using graph-based examples from a system as context, UNICON infers the predictive relation shared across them and applies it to queries from the same system. Across scientific and social systems, including those from disciplines absent from training, the same model approaches specialist performance without retraining. Combining UNICON with language-model agents yields further gains, enabling it to surpass state-of-the-art specialists in a discipline unseen in training. We further show that training-corpus diversity improves generalization to unseen disciplines. Together, these results establish UNICON as a foundation model of numerical intelligence and position it as a building block for a broader ecosystem of artificial intelligence. 

---
# When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence 

**Authors**: Zongheng Guo, Tao Chen, Tianli Li, Mingzhe Cui, Yang Jiao, Lei Xie, Yi Pan, Xiao Hu, Manuela Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2607.28421)  

**Abstract**: Derived measurements increasingly enter large language model (LLM) pipelines as direct facts despite their instance-dependent validity. We define derived-feature over-trust (DFOT) as the failure in which a downstream LLM assigns such a measurement the epistemic status of a direct fact or uses it outside its valid scope. Using physiological sensing as a case study, D1 tests acceptance of a PPG-derived rhythm contradicted by offline ECG, whereas D2 tests rejection of an offline-confirmed reliable PPG rhythm under misleading severe history. ECG supplies training supervision and offline reference construction but is never shown to the LLM. Five estimands quantify this chain: conflict over-trust rate (COTR) and context-induced error rate (CIR) characterize D1/D2; correct repair rate (CRR) measures frozen-error repair; evidence-specific repair margin (ESRM) contrasts matched and patient-disjoint shuffled evidence; and utility harm rate (UHR) measures unnecessary verification among HIGH-reliability cases used without verification at baseline. The framework does not depend on a particular reliability generator. We demonstrate it on 50,000 paired PPG-ECG records using ECG-to-PPG privileged distillation as an illustrative baseline and PPG-only inference. On a protocol-locked 187-patient test, the baseline improves four repair and specificity endpoints by 1.82-6.69 percentage points, with all paired confidence intervals excluding zero; UHR increases by 0.67 percentage points (95% CI: -0.4 to +1.7). DFOT provides a common evaluation target for stronger mitigation methods. The code is available at this https URL. 

---
# WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning 

**Authors**: Haozhe Hu, Hao Wu, Peiran Yin, Chao Han, Yunpu Ma, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28418)  

**Abstract**: Pruning is a promising approach for improving the efficiency of LLMs. Existing static structured pruning methods are hardware-friendly and can deliver practical throughput gains, but their input-agnostic computation allocation often causes substantial accuracy degradation under aggressive sparsity. Recent dynamic sparsity methods improve quality retention by adapting computation to individual inputs, yet they remain largely limited to coarse-grained structural decisions and their practical acceleration under real-world inference scenarios remains challenging. To address these challenges, we present WIDE, the first end-to-end differentiable token-level dynamic width pruning framework designed for both prefill and decode scenarios. WIDE enables fine-grained computation allocation by allowing each token to dynamically select attention-head groups and FFN-channel groups, extending dynamic pruning beyond layer-level decisions to neuron-block-level granularity. Through a two-stage training pipeline, WIDE learns effective token-wise sparse execution patterns and achieves substantially better quality retention than existing approaches. To make such fine-grained dynamic pruning practical, we further propose a pruning--kernel co-design framework that decomposes dynamic sparsity acceleration into mask reordering, hardware-agnostic block-level skipping, and hardware-dependent intra-block skipping, enabling efficient execution across different granularities. At 50% sparsity, WIDE provides 55.1% performance boost when compared to the state-of-the-art dynamic depth pruning under calibration-only settings. Under prefill and decoding inference workloads, WIDE achieves close-to-theoretical kernel-level speedups of up to 1.98x for prefill and 4.95x for decoding, as well as 1.68x and 1.55x end-to-end acceleration. Our code is available at this https URL. 

---
# QuantWAMs: Calibrating at the Right Granularity for World Action Models 

**Authors**: Jiacheng Zhou, Jinfan Lv, Ruixuan Li, Longtai Zhang, Yan Wang, Wenqiang Zhang, Lizhe Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28405)  

**Abstract**: World Action Models (WAMs) jointly predict future observations and actions, but their iterative denoising and closed-loop execution make efficient deployment costly. Existing post-training quantization (PTQ) methods are poorly suited to WAMs because they rely on open-loop objectives, homogeneous model assumptions, and calibration distributions that do not reflect deployment. We present QuantWAMs, a PTQ framework that aligns quantization decisions with the calibration context defined by model structure, rollout distribution, and task objective. QuantWAMs introduces three strategies: shared-basis outlier calibration, which pools activation evidence only across coordinate-compatible modules; co-training-objective saliency, which computes empirical-Fisher scores from the joint video--action gradient and assigns weight precision at a calibration-stable layer granularity; and fixed-intervention rollout auditing, which revises denoising-step protection schedules using reachable closed-loop states without changing the precision budget. We evaluate QuantWAMs on Fast-WAM and LingBot-VA across RoboTwin 2.0, LIBERO, and real-robot manipulation with an AgiBot G2. Under a W4A4-dominant setting, the reported simulation means differ from FP16 by 0.2--0.7 percentage points. Real-robot trials further establish deployment feasibility on three manipulation tasks. For the targeted video and action blocks, QuantWAMs reduces peak weight-and-activation memory to about 29\% of FP16 and provides 1.4--1.6$\times$ block-level speedups. 

---
# GLM-RAG: Graph Language Models for Graph-Based Retrieval-Augmented Generation 

**Authors**: Maya Arseven, Anette Frank, Beni Egressy, Johann Higl, Moritz Plenz  

**Link**: [PDF](https://arxiv.org/pdf/2607.28397)  

**Abstract**: Retrieval-augmented generation (RAG) over knowledge graphs requires retrievers that can effectively capture both graph structure and semantic information. Recent approaches have explored graph neural network (GNN)-based retrievers to model graph topology in multi-hop reasoning tasks. In parallel, graph language models (GLMs) have emerged as a promising paradigm that integrates graph reasoning and the semantic capabilities of language models. In this work, we introduce a GLM-based retriever and investigate the comparative strengths of GLM-based, GNN-based, and traditional vector-search-based retrievers in single- and multi-hop RAG settings, and with a particular focus on transferability to unseen domains. Our findings suggest that finetuned GLM retrievers generalize better out of domain, achieving SOTA on two multi-hop benchmarks. On in-domain multi-hop QA datasets they remain comparable to prior work, with promising scaling as parameters and subgraph coverage increase. GNN-based retrievers achieve higher graph coverage with an efficient training setup, whereas the vector-search baseline excels at single-hop datasets. 

---
# When Specifications Conflict: A Symmetry-Based Framework for Measuring LLM Preferences 

**Authors**: Tairan Wang, Liang Zhou, Zikang Zhan, Pingchuan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2607.28384)  

**Abstract**: Large language models (LLMs) are increasingly required to integrate multiple sources of information that may be inconsistent or conflicting. However, there is still a lack of controllable and attributable methods for analyzing how models resolve conflicts between competing specifications. We propose a controlled experimental framework for studying model preferences under conflicting specifications. By constructing specifications with explicit conflicts, the framework enables model choices between competing specifications to be directly observed and analyzed. A symmetry-based design further reduces confounding factors, allowing preferences across representation types to be compared systematically.
We evaluate the framework on an executable mathematical benchmark with 550 conflict instances spanning 11 function families, comparing four representation types: pure natural language, formal language, naturalized formal language, and input--output examples. Results show systematic preference patterns rather than random behavior, with a consistent ordering: $ \text{Formal} \approx \text{Naturalized Formal} > \text{Pure Natural Language} > \text{Input--Output Examples} $. Example effects further depend on model capability and function family.
We extend the framework to heterogeneous specification conflicts in Boolean algebra, code generation, and the clinical domain, demonstrating its applicability across diverse tasks and specification forms. The framework provides a unified approach for measuring how LLMs resolve conflicts between competing sources of information. 

---
# HyperClaim: Fine-Grained Cross-Modal Hypergraph Reasoning for Video Misinformation Detection 

**Authors**: Xiangbo Wang, Jiasheng Zhang, Xingtong Yu, Luoqiang Lei, Delvin Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28375)  

**Abstract**: Video misinformation detection is often approached through global multimodal fusion or free-form multimodal reasoning. Both paradigms can under-represent localized authenticity cues that arise from coupled interactions among query phrases, contextual text, and short temporal spans of frames. Because such interactions are inherently higher-order, pairwise graph formulations are insufficient to capture multi-way cross-modal dependencies, whereas hypergraphs offer a suitable representation for these relations. We propose HyperClaim, a discriminative temporal hypergraph framework for sample-level authenticity classification. Using the title or benchmark-provided paired text as a claim-like query, HyperClaim constructs a sparse heterogeneous hypergraph over query tokens, evidence tokens, and sampled frames; applies confidence-aware filtering and source budgeting to form compact text-frame and short-range temporal evidence units; performs adaptive soft-incidence reasoning with residual text-video calibration; and aggregates textual, visual, and hyperedge states through a discrepancy-aware readout. Without relying on generated rationales or external tool calls, HyperClaim preserves fine-grained cross-modal and temporal structure that global fusion tends to flatten. Under the FactGuard temporal protocol, it achieves 83.7%, 82.0%, and 87.3% accuracy on FakeSV, FakeTT, and FakeVV, respectively, outperforming strong discriminative and reasoning-centric baselines. Learned incidence and attention weights further reveal token- and frame-level structure. 

---
# How Benchmarks Mis-Score Computer-Use Agents 

**Authors**: Zihan Dong, Zhiyuan Ma, Zekun Wang, Yunqing Li, Zirou Liu, Ruixuan Deng, Qishi Zhan, Rui Qian  

**Link**: [PDF](https://arxiv.org/pdf/2607.28367)  

**Abstract**: Computer-use agents (CUA) are being deployed to browse the web and operate desktop software, yet their benchmark scores are still commonly produced by brittle scripted oracles. A score is the output of a pipeline in which tasks can be stale, trajectories can omit decisive visual evidence, evaluators can reject valid alternatives, and aggregate reports can hide the cause of failure. We organize these problems into a reliability framework spanning task construction, trajectory observation, scoring, and reporting. We then audit 150 public failure-scored trajectories from five web, enterprise-workflow, and desktop-control benchmarks, find that 15.3\% of FAIL verdicts are wrong: 10.7\% are evaluator false negatives and 4.7\% are broken tasks. For genuine failures, a three-tier diagnostic taxonomy shows that verification/feedback and planning failures dominate execution/grounding errors, while a single scalar success rate can not explain. We connect these findings to newer long-horizon CUA benchmarks and derive stage-specific design rules for CUA evaluation. 

---
# Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners 

**Authors**: Feng Xiong, Leyan Xue, Hongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.28336)  

**Abstract**: On-policy distillation provides dense supervision for multimodal reasoners, but its trajectory-level reward cannot determine whether a failed answer arose from perception or subsequent reasoning. Perception Success Rate (PSR), estimated from multiple reasonings sharing one perception, remains ambiguous because low success conflates perceptual insufficiency with reasoning difficulty. We introduce \textbf{Perception-Correction Distillation (PCD)}, a label-free method that identifies correctable perception failures using downstream failure and teacher--student disagreement as complementary witnesses. Their product, , forms a soft AND gate that strengthens distillation only when both witnesses are present. We motivate this rule through Bayesian evidence combination and show that multiplication is the unique normalized bilinear gate that vanishes when either witness is absent. PCD uses separated perception--reasoning rollouts and mean-preserving weights, leaving the reasoning objective unchanged. Across eight benchmarks, PCD improves the 8B 2B macro average from 44.50 with OPD to 47.28 and the 32B 8B result from 56.94 to 61.22. In matched 2B ablations, removing PCD and separated rollout reduces held-out average by 2.22 and 0.88 points, respectively. Effective multimodal distillation therefore depends not only on what the teacher predicts, but also on identifying when perception is the appropriate target of correction. 

---
# Paying for Honesty Without Knowing the Truth: Reputation-Penalty Design for LLM Marketplace Agents 

**Authors**: Mingdai Yang, Shicheng Fan, Kejing Yu, Duohao Wang, Li Sun, Hao Peng, Philip S. Yu, Zhiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28330)  

**Abstract**: LLM agents increasingly act as autonomous merchants that write their own product listings, and under competitive pressure, they fabricate attributes to win sales. Even under instructions to be honest, they fabricate attributes in a majority of listings across models. A platform's obvious remedy---verifying each claim against the truth---is unavailable, because it observes only a noisy, biased complaint signal, never the ground truth. We design CARP, a reputation-penalty mechanism with a deadband that forgives complaint noise and a state-dependent severity that counters reputation-driven detection erosion. CARP requires no product-level ground truth and is robust to strategic gaming. CARP protects consumers by suppressing the sales volume of low-rated liars while sparing honest sellers. Paired with SPARC, it closes most of the consumer-welfare gap relative to a perfect-information oracle, without ever accessing the truth. It also achieves the best welfare of the policies we compare. We further show that this felt penalty becomes behaviorally binding through SPARC, a byte-clean code-gated reflection mechanism: LLM merchants fabricate when lying is free but restrain themselves when fabrication costs them sales, a self-interested response rather than compliance. We trace this distinction to penalty-gated self-correction reasoning, and observe the binding across models, with supporting confidence intervals. 

---
# PathView-Bench: Can Multimodal Large Language Models Achieve Fine-grained Multiscale Understanding of Pathology Images? 

**Authors**: Zongyi Chen, Yu Liang, Jie Lin, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28318)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used to analyze pathology images. However, dominant multimodal benchmarks in pathology mainly score final diagnostic answers, captions, or reports. These evaluations provide limited insight into whether a model understands the multiscale visual content needed for pathology reasoning and decision-making. We introduce PathVU, a vision-anchored benchmark for fine-grained and multiscale visual understanding in computational pathology. Built from 23 public pathology imaging datasets with human-supervised labels and spatial annotations, PathVU evaluates MLLM understanding in two fields of view: Region FOV for high-resolution local regions and Slide FOV for macro whole-slide views. By converting raw annotations into deterministic task targets, PathVU enables programmatic scoring of region localization, visual recognition, quantity estimation, spatial reasoning, and insufficient-context judgment. The benchmark contains 14 VQA-style tasks, 61,673 images, and 308,070 samples across 28 organs and 7,253,526 annotations. Evaluating 18 representative general-purpose, medical-domain, and pathology-oriented MLLMs, we observe substantial limitations even in advanced models on fine-grained visual tasks across multiscale pathology images. PathVU provides a reproducible basis for developing and evaluating pathology MLLMs with explicit multiscale visual understanding. 

---
# One Human, $N$ Agents: Audit-Budget Allocation for LLM Agent Fleets under Miscalibrated, Correlated Confidence 

**Authors**: Cesare Zavattari, Alessandro Tommasi, Giuseppe Prencipe  

**Link**: [PDF](https://arxiv.org/pdf/2607.28317)  

**Abstract**: A single human must audit $N$ LLM agents under a budget of $B \ll N$ audits per round, guided by self-reported confidence that may be adversarially miscalibrated and by correlated errors. We model this as budgeted noisy inspection over a two-level Gaussian copula and locate the miscalibration threshold $\delta^*$ past which confidence-ranked auditing is \emph{worse} than random. Two a-priori expectations reverse: $\delta^*$ \emph{rises} as the budget shrinks, and cross-family correlation is not low---shared difficulty dominates lineage. Five open-weight LLMs show operationally useless (near-constant) confidence, point estimates at or beyond the flip though CIs straddle it; a proprietary model is informative and lands below it. We give a quantitative criterion for \emph{vacuous} oversight, and replaying policies on recorded traces confirms the ordering. 

---
# Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3 

**Authors**: Jens Lehmann, Andrei Aioanei, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2607.28287)  

**Abstract**: ARC-AGI-3 turns abstraction into an interactive problem of skill acquisition. A player must infer an unfamiliar game's rules, hidden state, and goal while maintaining action efficiency because every move counts. We formalize these environments as parameterized rendered deterministic Moore machines and introduce Tycho, a coding-agent system that constructs and uses game-specific models during interaction. Tycho separates actionable observations from intermediate animation, level-completion, and game-over frames. From this structured history, an agent can model, test, plan with, repair, or bypass a free-form executable hypothesis.
In one matched public-set run per policy, we compare four orchestration policies on all 25 public games using Claude Opus 4.8 under matched inference budgets. Actor-requested delegation to a model builder obtains the highest observed mean Relative Human Action Efficiency (RHAE), 88.49. With this selected policy, GPT-5.6 Sol and Opus 5 both reach 100.00 RHAE and complete all 183 levels. Their game-balanced first-run human-replay midranks are 98.5 and 100.0. Opus 5 uses 61% fewer scored actions than the aggregate official human baselines.
Automatic repair after verification failures produces models that reproduce observed transitions much more accurately, yet reaches only 83.07 RHAE. Transition match indicates whether a simulator reproduces observed dynamics, not whether it has identified the objective or improves the next action. Strong play also requires deciding when to construct, repair, use, or bypass a model. We call this joint problem active abstraction: generating a testable model from costly interaction and deciding when acquiring or using it is worth its cost. 

---
# MemHarness: Memory Is Reconstructed, Not Replayed 

**Authors**: Rong Wu, Daocheng Fu, Licheng Wen, Xuemeng Yang, Shu Zou, Jianbiao Mei, Yuxin Wang, Hairong Zhang, Yu Yang, Tao Hu, Cong Zhang, Botian Shi, Pinlong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.28272)  

**Abstract**: Retrieving past experiences has become a common strategy to enhance large language model agents. However, most existing memory-augmented agents treat retrieved experiences as static records to be replayed verbatim, injecting them into the context regardless of whether they align with the agent's current situation. This ``replay'' paradigm ignores the gap between the abstract, general nature of stored experience and the concrete, ever-changing states encountered at decision time, frequently causing negative transfer. In contrast, humans rarely recall past experiences verbatim; instead, they reorganize and adapt retrieved memories to fit the present context. Inspired by this, we propose MemHarness, a framework that equips LLM agents to actively harness and reconstruct past experiences based on the present context. At each decision step, a unified policy model critiques and reconstructs the retrieved experience conditioned on the current state, producing context-grounded guidance before acting. This reconstructive ability emerges naturally through end-to-end training with GRPO. Experiments on ALFWorld and WebShop show that MemHarness substantially outperforms pure RL and static memory-augmented baselines, demonstrating strong robustness in out-of-distribution (OOD) scenarios. Furthermore, our analyses reveal that this reconstruction objective not only prevents negative transfer but also serves as latent guidance during training, fundamentally improving the agent's intrinsic reasoning capabilities. 

---
# LLM-Guided Evolutionary Search for Constraint Model Reformulation to Improve Solver Efficiency 

**Authors**: Kostis Michailidis, Dimos Tsouros, Nguyen Dang, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2607.28268)  

**Abstract**: Combinatorial problems appear in numerous industrial applications. A common approach is to formulate these problems as declarative constraint models that can subsequently be compiled to and solved by a range of back-end solvers. Recent work shows that Large Language Models (LLMs) can produce correct models from natural language, but even a correct model can be expensive to solve because performance remains sensitive to modelling choices. In this work, we investigate whether LLMs can automate performance-oriented model reformulation. Inspired by Automatic Heuristic Design (AHD), we use an evolutionary framework in which an LLM proposes candidate reformulations that are verified and benchmarked against the user-defined baseline model. We compare AHD-adapted search strategies that control which prior attempts, instructions, and measured feedback enter each prompt. Existing retention strategies prioritize recency or performance, but do not explicitly diversify the context. To cover this gap, we introduce Profile-Diverse Retention (PDR), which applies Maximal Marginal Relevance (MMR) to instance-level runtime vectors to retain behaviourally diverse attempts. We systematically evaluate the strategies on eight CSPLib problems using validation-based final model selection. The results show that: (i) iterative reformulation can produce substantial held-out speedups; (ii) strategies that keep the retained context diverse outperform those that retain only recent or the fastest attempts; and (iii) validation-based selection improves the held-out speedup of every strategy. 

---
# Operationally Guided Placement-Aware Learning for Industrial Online 3D Bin Packing 

**Authors**: Dheeraj Poolavaram, Aanchal Rajesh Chugh, Sebastian Dorn  

**Link**: [PDF](https://arxiv.org/pdf/2607.28257)  

**Abstract**: The online three-dimensional bin packing problem (3D-BPP) is a longstanding challenge in logistics and industrial palletizing. Recent learning-based methods use a learned policy to select among feasible candidate placements. Performance depends on the candidate generator and representation, especially in industrial settings where packings must be space-efficient, stable, compact, and balanced. However, prior work has mainly optimized the policy, while candidate generation and representation remain largely geometry-driven. We address this gap with OPAL, an operationally guided placement-aware learning framework for industrial online 3D-BPP which combines an Operationally Guided Empty-Maximal-Space generator (OG-EMS), an operational representation for each candidate placement, and a masked ranking policy trained with proximal policy optimization. OG-EMS evaluates multiple anchors within each free-space region and prioritizes low, well-supported, compact, and spatially diverse placements. An xLSTM-based Placement Encoder models dependencies among geometric and operational candidate attributes, while a lightweight recurrent core combines the resulting embeddings with the current item and pallet state to rank feasible actions. On the BED-BPP benchmark, OPAL achieves a mean space utilization of 0.49, with improvements of 15.1% from operationally guided candidate generation and 6.3% from learned ranking, while maintaining robust inference-time performance. 

---
# AI and Authenticity in Islamic Research: A Critical Evaluation of Generative AI Reliability, Hallucination, and Source Fidelity in Quranic, Hadith, and Fiqh Knowledge 

**Authors**: Muhammad Sajjad Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2607.28237)  

**Abstract**: Generative Artificial Intelligence (AI) is increasingly used by Muslims for religious guidance, Qur'anic interpretation, Hadith explanation, jurisprudential rulings, and Islamic education. Despite its growing adoption, there is limited empirical evidence on whether current AI systems provide authentic, verifiable, and trustworthy Islamic knowledge suitable for high-trust religious contexts. This study evaluates six leading generative AI systems using fifty realistic open-ended Islamic questions covering Qur'anic interpretation, Hadith, Fiqh, ethics, pastoral advice, and Madhhab-sensitive topics. Responses were collected under real-world conditions from participants in Australia and the United Kingdom and analysed using a mixed-method framework examining domain accuracy, citation verification, hallucinations, jurisprudential consistency, uncertainty handling, source provenance, and geographical variation.
The study addresses four research questions: (1) How accurate and authentic are AI-generated responses across major Islamic knowledge domains? (2) To what extent do AI systems produce hallucinations, incomplete citations, or unverifiable religious references? (3) How consistently do models handle jurisprudential disagreement, Madhhab diversity, and uncertainty? (4) Are current AI systems sufficiently reliable for religious guidance, Islamic education, and scholarly research?
Overall, current generative AI systems are valuable as assistive tools for introductory Islamic learning but should not be treated as authoritative sources for religious rulings or Islamic research without verification against authenticated primary sources and qualified scholarly expertise. This study provides one of the first comprehensive empirical evaluations of AI reliability within Islamic knowledge, offering practical guidance for researchers, educators, AI developers, and the wider Muslim community. 

---
# CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising 

**Authors**: Sina Heydari, Amirreza Abbasi, Mohsen Hooshmand, Majid Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2607.28236)  

**Abstract**: Pre-trained language models have significantly improved sentence representation learning, yet their embedding remain sensitive to semantic preserving textual perturbations such as synonym substitution, masking and word dropout. This work proposes a lightweight Contrastive Denoising Autoencoder (CDAE) that refines pre-trained BERT embedding by jointly optimizing contrastive and reconstruction objective to learn perturbation-invariant representation. We evaluate the proposed framework using multiple perturbation strategies with varying strengths and compare it against the original BERT embeddings and SimCSE. Experimental results show that CDAE consistently preserves higher embedding similarity under perturbations, with the improvements becoming more pronounced as framework effectively enhances representation stability while preserving semantic information, highlighting perturbation-invariant learning as a promising direction for improving sentence embeddings. The source code is publicly available at: this https URL 

---
# Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents 

**Authors**: Hanzhang Zhou, Panrong Tong, Xu Zhang, Quyu Kong, Chenglin Cai, Tianyu Xia, Gongjie Zhang, Jianan Zhang, Long Li, Long Chen, Lei Wang, Gaole Dai, Pengxiang Li, Liangyu Chen, Yue Wang, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28227)  

**Abstract**: GUI agents have the potential to become a general purpose executor over existing digital devices. To advance them toward real-world use, we envision agents that operate reliably on real devices, execute workflows across platforms, combine GUI interaction with CLI execution, complete long-horizon tasks, proactively initiate useful services, and autonomously improve their capabilities with minimal human effort. Guided by this vision, we present Qwen-UI-Agent, a real-world centric foundation GUI agent spanning mobile, computer-use, web, and DeepSearch environments. Qwen-UI-Agent combines diverse sandbox environments with a large-scale real-device mobile runtime. Its unified action space interleaves GUI operations with CLI execution and generates batched actions in a single model turn. An AutoResearch-style data flywheel uses agents to construct tasks and environments, diagnose failures, and plan subsequent iterations. Online RL supports training on trajectories exceeding 100 turns, with over 10,000 concurrent environments accelerating rollout. A lightweight harness layer supports proactive service initiation and stateful workflows across mobile and computer.
Across a broad suite of evaluations, Qwen-UI-Agent sets state-of-the-art performance on mobile-use benchmarks while delivering competitive performance on computer- and browser-use tasks against frontier models, including Opus 4.8, Gemini 3.1 Pro, and GPT-5.6 Sol. On mobile use, it achieves 82.1% on MobileWorld, 92.2% on MobileWorld-Real, and 97.5% on AndroidDaily. On computer use, it achieves 79.5% on OSWorld-Verified and a 40.0% partial-progress score on OSWorld-v2. On browser use and GUI grounding, it achieves 73.6% on WebArena and 81.5% on ScreenSpot-Pro, respectively. 

---
# Old Tricks, New Models: How Simple Image Transformations Break Modern AI-based Content Moderation 

**Authors**: Marco Alecci, Francesco Marchiori, Iyiola Emmanuel Olatunji, Tegawendé F. Bissyandé, Jacques Klein  

**Link**: [PDF](https://arxiv.org/pdf/2607.28187)  

**Abstract**: While automated content-moderation systems have become essential for screening harmful content at scale, conventional task-specific classifiers often provide limited policy cov- erage and contextual understanding. Recently, commercial multimodal moderation APIs built on large foundation models have been introduced with the promise of providing broader and more capable safety filters. In this work, we analyze whether this shift also yields more robust image moderation. We conduct a large-scale black-box evaluation on three established commercial image-moderation services and compare their robustness. By evaluating seven simple, model-agnostic image transformations across multiple providers, datasets, harm categories, perceptual-similarity constraints, and transformation intensities, we find that: (1) all three commercial services can be bypassed using inexpensive image transformations that require no gradients, surrogate models, or knowledge of the target system; (2) even fixed transformations such as color inversion and grayscale conversion induce unsafe-to-safe decision changes while preserving content that remains recognizable to humans; (3) their robustness varies substantially across datasets and harm categories, with multimodal content and self-harm exhibiting pronounced vulnerabilities. This yields the conclusion that replacing conventional moderation classifiers with foundation-model-based APIs does not, by itself, provide a reliable security boundary. Such systems must be evaluated under realistic transformations and deployed as one component of a layered moderation pipeline rather than as standalone safety filters. 

---
# AgenticASR: Refining Speech Recognition in Real-World Scenarios via an Agentic Approach 

**Authors**: Zixuan Jiang, Binghao Qiang, Jiaying Chi, Yanqiao Zhu, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28175)  

**Abstract**: Automatic speech recognition (ASR) has achieved substantial gains in transcription accuracy, yet verbatim transcription does not necessarily produce readily usable text. It retains fillers, repetitions, false starts, and self-corrections that increase reading effort, obscure the speaker's final intent, and propagate unresolved or abandoned content to downstream tasks. Existing spoken-to-written methods process completed audio or transcripts but cannot revise emitted text when later speech changes how preceding content should be interpreted. We therefore formulate Agentic Speech Recognition (AgenticSR), an audio-to-clean-text task that removes disfluencies, resolves self-corrections, and normalizes written form while preserving the speaker's final intent. AgenticASR implements this task through an ASR--Refiner architecture that repeatedly transforms a bounded active context and replaces its corresponding output span as audio arrives. This enables continual emission and revision over streams of arbitrary duration. We also introduce AASR-Bench, a bilingual benchmark with fine-grained atomic rubrics. Across multiple ASR front ends, AgenticASR attains the highest AASR-Bench scores among evaluated systems. A human--AI agreement study shows that rubric-based judgments align with independent expert assessments. Ablations characterize Refiner capacity, context length, and the quality--latency trade-off between online and offline inference. Together, these results establish AgenticASR as a practical framework for intent-preserving clean transcription during ongoing speech. Code, AASR-Bench, and a demo will be released at this https URL. 

---
# ConMem: Contribution-Aware Memory for Long-Horizon Manufacturing Inspection Logs 

**Authors**: Bingchen Liu, Yuanyuan Fang, Lei Liu, Guangyuan Dong, Xing Fu, Yuanyuan Gao, Shuyue Wei, Xin Li, Xiangtian Meng  

**Link**: [PDF](https://arxiv.org/pdf/2607.28126)  

**Abstract**: Long-horizon steel-equipment inspection requires reasoning over heterogeneous records accumulated across repeated inspection cycles. Existing retrieval-augmented generation systems treat historical logs as a static corpus and retain records without estimating their diagnostic value, failing to report early risk. To this end, we propose ConMem, a contribution-aware memory framework for LLM-assisted equipment inspection, supporting a human-in-the-loop early-risk screening system. Specifically, our ConMem first segments inspection logs into functional evidence units, then estimates each memory unit's contribution to downstream diagnosis through a Shapley-style estimation, and finally retains high-value evidence under a constrained memory budget. In experiments, we evaluate ConMem on real-world dataset and ConMem achieves 76.0% QA accuracy, exceeding the strongest directly comparable baseline. Relative to the naive 8K-context LLM baselines, it reduces the average number of input tokens by 88.2% and response time by 86.6%. Ablation studies also show that the functional-role-aware segmentation and contribution-based valuation are helping prioritize weak degradation signals for targeted field inspection. Practical deployments further confirm that ConMem retains the weak early signal across three inspection cycles, providing an early-stage seal-wear alert targeted for on-site inspectors. 

---
# BlueprintRepair: Typed Local Edits for Failed Lean Proof Blueprints 

**Authors**: Ruslan Khrulev  

**Link**: [PDF](https://arxiv.org/pdf/2607.28110)  

**Abstract**: LLM-based Lean proving systems increasingly organize a proof as a blueprint: a dependency graph of formal statements. We introduce BlueprintRepair, a repair interface that lets a model change this graph through ten schema-checked local operations. An operation names the node it edits, so the target theorem cannot be changed. Lean checks every applied change, and an accepted repair must declare every blueprint lemma its proof uses. We also construct BlueprintTrace, a benchmark of 142 controlled failures with complete accepted and rejected repair trajectories. We compare typed edits, exact source patches, and complete module rewrites under matched source, feedback, model, and budget, one episode per state and interface. With DeepSeek-V4-Flash, the three interfaces solve almost the same number of the benchmark's localized failures. Typed repair is the cheapest per solved state (patching is 1.30x as expensive, rewriting 2.06x), and within 10,000 completion tokens per task it reaches almost all of its final coverage, while both free-form interfaces are well behind. A second model, Qwen3.6-Flash, solves fewer states but keeps typed repair cheapest, puts it ahead on the proof-authoring states, and repeats the localized pattern. 

---
# Beyond Rephrasing: Book-Level Organization Improves Synthetic Textbook Data for Mid-Training 

**Authors**: Jiawen Tao, Miao Peng, Yaoming Li, Xiaokun Yuan, Mengzhou Wu, Wenhan Yu, Guoan Wang, Nuo Chen, Tong Yang, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.28109)  

**Abstract**: Synthetic textbook data has improved language model pre-training, but prior work largely treats the benefit as a property of generated content or local rewriting style. We study a different factor: whether related content is organized into coherent book-level documents. We contribute both a scalable synthesis pipeline and controlled evidence that this organization matters. The pipeline retrieves source material from a pre-training corpus, clusters it into topical units, plans hierarchical tables of contents, and assembles source-grounded sections into complete books (our Full setting), yielding 686K textbooks (32B tokens) across 15,000+ disciplines. Replacing natural books in a mid-training mix with this corpus improves downstream performance by +1.09 on average. Controlled comparisons then disentangle the relevant design factors. A content-matched Split condition holds generated text and tokens fixed but treats each section as an independent document; Full's +1.02 mean gain isolates document packaging. A length-matched RandomConcat control that joins sections from different books remains below Full, ruling out document length alone. A retrieval-pool-matched Rephrase condition independently rewrites individual retrieved documents under the same audience-by-style scheme, without clustering, TOC planning, or book assembly; Full's +1.17 gain demonstrates the value of structured synthesis. On Llama3-8B, Full likewise outperforms both RandomConcat and Natural Books, supporting book-level organization as a useful axis for synthetic pre-training data design. 

---
# MIND: Lightweight and Effective Memory Injection Defense for LLM Agents via Intent-Aware Information Bottleneck 

**Authors**: Dongyi Liu, Haixing He, Xiaobao Wu, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.28103)  

**Abstract**: Memory-augmented LLM-based agents are vulnerable to memory injection attacks: Agents may retrieve poisoned memory from attackers, which diverts their behavior from initial user intent and finally causes task failure. However, existing defense mechanisms either incur high computational cost or suffer from information redundancy in multi-turn contexts. To address these challenges, we propose Memory Intent-Aware Neural Denoising(MIND), a lightweight defense framework for memory injection attack. Our preliminary analysis reveals that benign and poisoned trajectories exhibit distinguishable relationships between the initial user intent and subsequent behavior. Building on this observation, MIND employs an intent-aware Information Bottleneck(IB) to extract compact intent--behavior representations from the initial intent and turn-level behavior. The IB preserves intent-relevant cross-turn attack signals while filtering task-irrelevant and repetitive information, and a lightweight detector identifies malicious memories from the resulting representations. As such, MIND mitigates information redundancy in multi-turn contexts while avoiding the overhead of repeated LLM auditing. Extensive experiments show that MIND reduces attack success rates while preserving task accuracy and inference efficiency. Notably, on ReAct-StrategyQA, MIND reduces mean ASR-r and ASR-a by 55.4% and 55.3%, respectively, while matching the undefended agent in average accuracy and latency. 

---
# SciDataSailor: Deep Scientific Data Exploring 

**Authors**: Jiyong Rao, Yicheng Qiu, Chi Zhang, Chunfeng Song, Runkai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.28098)  

**Abstract**: Scientific datasets are commonly organized as hierarchical repositories containing heterogeneous and interdependent files, making their inspection, integration, and analysis labor-intensive and reliant on domain expertise. Although large language model (LLM) agents have advanced substantially in planning, reasoning, and tool use, existing research has largely overlooked their ability to interact with real scientific data assets through executable environments. We introduce Deep Scientific Data Exploration, an agentic task paradigm in which agents navigate repositories, interpret heterogeneous files and schemas, execute analyses, integrate cross-file evidence, and produce conclusions grounded in executed observations. To operationalize this paradigm, we present SciDataSailor, a framework for synthesizing tool-interactive trajectories by balancing broad exploration with targeted exploitation. SciDataSailor instantiates trajectory synthesis as Monte Carlo Tree Search (MCTS) with four task-specific mechanisms: difficulty-stratified exploration seeds, dual-feedback first-play urgency, hierarchical strategy-to-tool action generation, and entropy-guided branching. Using this framework, we construct SciDataSailor-SFT-2K for supervised fine-tuning and SciDataSailor-Bench for evaluation, with the latter comprising 627 meta-information summarization tasks and 586 scientific question-answering tasks across 27 datasets spanning the life, earth, and physical sciences. 

---
# An Instrument to Evaluate Governance Proposals: AI Policy Analysis at Scale 

**Authors**: Paulo Carvao, Claudio Mayrink Verdun, Isabel Adler, Jeffrey Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.28094)  

**Abstract**: This paper introduces a policy analysis framework for systematic, transparent assessment of AI governance proposals in an evolving and contested regulatory landscape. AI policy debates often collapse into binary positions that obscure underlying tradeoffs and normative assumptions. The framework structures policy analysis around multiple policy attributes, allowing users to surface priorities and tensions without prescribing outcomes. We use a mixed-methods approach that integrates qualitative insights from subject matter experts with computational text analysis to inform the design of policy attribute rubrics. This quantifies the relative emphasis of different policy objectives and presents them through comparative visualizations that support interpretability and cross-policy comparison. The paper also examines the use of commercial LLMs for rubric-based policy analysis, benchmarking their outputs against a domain-trained rubric-calibrated model with explicitly defined analytical assumptions. Rather than assessing policy effectiveness or desirability, the framework focuses on relevance and alignment across attributes. By making analytical assumptions explicit, including attribute selection, rubric construction, and weighting schemes, the framework enables users to evaluate whether its embedded priorities align with the users' own normative commitments. The approach is jurisdiction-agnostic and intended to support policymakers, analysts, and researchers navigating complex AI governance environments.
Contributions: (1) multidimensional policy assessment through empirically grounded rubrics that surface tradeoffs rather than resolving them; (2) a transparent hybrid methodology combining feedback from subject-matter experts with computational validation; and (3) use of domain-trained rubric-calibrated models as a benchmark for comparing different general-purpose large language models. 

---
# PerturbMap: Cross-Context Transfer of Single-Cell Perturbation Responses 

**Authors**: Panpan Cui, Yiqi Liu, Wenhao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.28090)  

**Abstract**: Single-cell perturbation atlases rarely measure every intervention in every cellular context: a query perturbation is often observed in one or more source contexts but missing in the recipient context where its effect is needed. Ignoring those measured responses discards query-specific experimental evidence, whereas copying or weakly calibrating them across contexts risks transferring the wrong signal. We propose PerturbMap, which predicts a missing recipient-context effect by combining a recipient-local low-rank base with accepted proposals that transport the same perturbation's measured source responses through source-to-recipient ridge experts fit on paired training perturbations, with proposal weights determined by route reliability estimated on validation anchors. On the Perturb-CITE-seq melanoma cohort, PerturbMap improves full-effect MSE by 4.1\% over a recipient-local low-rank base and achieves lower MSE than FedAvg, zero-response, raw-copy, calibrated-copy, and identity-shuffled affine controls. It remains within $2.82\times10^{-6}$ MSE of our centralized token-matched pooled reference, which uses a stronger training interface. A condition-mean specificity diagnostic shows the same direction: same-recipient top-10 counterpart retrieval by cosine increases from 74.5\% for the low-rank base to 80.5\% for PerturbMap. 

---
# Diversifying Personalized Research Ideation against AI-Induced Homogenization 

**Authors**: Rui Xu, Yunke Wang, Linwei Tao, Wenjie Xuan, Yong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.28087)  

**Abstract**: AI-assisted research ideation has emerged as a promising paradigm for accelerating scientific discovery, with systems now capable of generating research directions conditioned on papers, topics, or lightweight researcher contexts. Yet current systems largely optimize individual suggestions in isolation. This leaves two blind spots. First, coarse researcher representations may elicit mainstream directions that appear broadly feasible, but lack sufficient researcher-specific grounding. Second, independent recommendations can concentrate a community's portfolio around recurring high-probability themes. To address these blind spots, we propose DivAlign, a four-stage pipeline for alignment-preserving de-homogenization. DivAlign extracts fine-grained researcher profiles, generates profile-conditioned candidate directions, scores them along three alignment dimensions (Executability, Comprehensibility, and Growth Potential), and surfaces researcher-local directions while reducing redundancy across the community portfolio. On a benchmark we construct from 95 AI researchers across five subfields, DivAlign reduces community-level redundancy while preserving researcher-direction fit. Compared with coarse single-shot ideation, it lowers average pairwise similarity from 0.331 to 0.294 and nearest-neighbor similarity from 0.704 to 0.608. Compared with the independent top-choice variant, DivAlign reduces nearest-neighbor similarity from 0.663 to 0.608 while retaining 99.9% of the researcher-direction fit score. Code and data are available at this https URL. 

---
# Distilling Answer Set Programming Theories from Large Language Models 

**Authors**: Nelson Higuera Ruiz, Markus Hofmarcher, Claudiu Leoveanu-Condrei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28086)  

**Abstract**: Writing Answer Set Programming (ASP) theories from scratch is a difficult and time-consuming task. We take a neurosymbolic approach to study whether a model can distill complete and correct theories, given a fixed agent harness with the solver in the loop. The protocol is dataset-agnostic: with a single prompt and an empty file as the starting point the model is given a 1-hour time limit to derive a complete theory. We chose VQA as the application domain, three benchmarks (CLEVR, GQA, CLEVRER), as these are publicly available and non-trivial. In order to study the model scale required for solving this task we nine different models: four frontier (Claude Sonnet 4.6, Claude Opus 4.7, GPT-5, DeepSeek V4 Pro), two mid-tier (DeepSeek V4 Flash, gpt-oss-120b), and three open-weights (qwen3.6-27b, gpt-oss-20b, qwen3.5-9b). Three of four frontier models reach 100% on CLEVR and 92.8%-98.8% on GQA; on CLEVRER, Sonnet, Opus, DeepSeek V4 Pro score 92.7%-95.3%. GPT-5 reaches 98.7% on CLEVR but drops to 41.8% on GQA and to 86.7% on CLEVRER. Adding handwritten reference theories from other datasets moves the other three frontier models by at most +/-3.4 pp but reduces GPT-5's accuracy by 3-19 pp. We release the code, prompts, and theories distilled. 

---
# Group-Reflective Self-Distillation for Agentic Reinforcement Learning 

**Authors**: Binbin Zheng, Zijun Xie, Guanqun Zhao, Enlei Gong, Xing Ma, Xiaoliang Fu, Zeyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28076)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is effective for training large language model agents. However, terminal rewards provide only coarse trajectory-level supervision, leaving successful behaviors, recurring mistakes, and incidental choices entangled in the same outcome signal. Existing agentic self-distillation methods enrich sparse supervision with natural-language skills, but skills retrieved externally or extracted from a single trajectory by stronger models may mismatch current experience, exceed the policy's capability, or remain path-specific. We propose Group-Reflective Self-Distillation (GRSD), which derives capability-aligned and outcome-discriminative guidance from the policy's own verified rollouts. For each prompt, the policy reflects on each verified trajectory in an on-policy group, and a stop-gradient snapshot contrasts the resulting reflections from successful and failed rollouts to construct group-level privileged guidance. Conditioned on this guidance, a self-teacher refines turn-level credit assignment by modulating outcome-based advantages while preserving the verifier-determined learning direction. Experiments across multiple agentic environments and model scales demonstrate that GRSD consistently outperforms competitive baselines and generalizes more effectively to unseen tasks. 

---
# Echoverse: Deep, Evolving Environments for Training Computer-Use Agents at Scale 

**Authors**: Yash Pandya, Sahil Gupta, Sarthak Harne, Archana Yadav, Kavyansh Chourasia, Hussein Mozannar, Vibhav Vineet, Sara Abdali, Corby Rosset, Yash Lara, Ahmed Awadallah, Ece Kamar, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28074)  

**Abstract**: Computer-use agents learn from what their actions change, so training one needs applications it can act on, break and reset. The applications that matter most are login-gated and stateful, so synthetic environments stand in for them. Recent pipelines generate such environments in bulk, which moves the bottleneck from how many exist to what is inside each one. The returns, we find, come from three properties: how much behavioural depth an environment carries, whether it targets the interaction an agent actually fails, and whether it improves alongside the model. We present Echoverse, which compiles specifications into stateful applications whose tasks are graded against the application's own database, and a co-evolution loop that reads every graded rollout twice: as repairs to the environment, its tasks and its verifier, and as training signal for the model. Trained on twelve such environments, a 9B model improves from $36.5\%$ to $67.1\%$ across fourteen evaluation splits, within fourteen points of the much larger frontier model that taught it. We examine each property in turn. On the same domains, shallow environments push live-site accuracy below the base model ($80.0 \to 75.0$) while deep ones raise it ($80.0 \to 85.0$ and $48.0 \to 65.0$); drilling one interface control across many renderings transfers to held-out widget families and to the open web; and repairing a single environment lifts the model trained on it from $16.2\%$ to $38.5\%$. The same worlds serve as reinforcement-learning environments, where a reward combining the grounded verifier with a dense per-step judge raises held-out score from $58.8\%$ to $68.0\%$. We release four environments as a benchmark, with their applications, seed data and grounded graders. Code: this https URL 

---
# SemPIC: Learning Semantic Position-Independent KV Caches 

**Authors**: Hui Xie, Peng Xiao, Yutong Deng\textsuperscript, Shuoran Dou, Jian Yang, Jinyang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.28069)  

**Abstract**: Long-context retrieval and agentic workloads repeatedly reuse the same documents under changing instructions, histories, and document orders. Prefix caching cannot exploit this reuse, while position-independent caching (PIC) remains unreliable because independently compiled KV states lack the future context in which they will be consumed. Our diagnostics show that a learned boundary-conditioned baseline sharply reduces attention deviation near reusable-block boundaries but leaves interior and task-level residuals, motivating adaptation of the document representation itself. We present \emph{SemPIC}, which trains a LoRA-enabled Writer to compile native per-layer document KVs through behavioral distillation while retaining the pretrained decoder as an unchanged Reader. Adaptation is confined to offline cache construction, preserving the standard KV interface and cache-hit decoding path. We further introduce KV Gradient Checkpointing, which reduces peak training memory without severing gradients through cached KVs. Across three models and four tasks, SemPIC raises mean micro-F1 over KV Packet from 0.53 to 0.60, approaching Full Recompute at 0.62. 

---
# IndustryForge-27B: A Domain-Enhanced Multimodal Foundation Model for Industrial CAD 

**Authors**: Nianchen Deng, Jiaxin Ai, Tao Hu, Shu Zou, Yurui Dong, Siqi Li, Xinyu Cai, Xuemeng Yang, Licheng Wen, Hongbin Zhou, Hairong Zhang, Pinlong Cai, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28050)  

**Abstract**: Automating industrial CAD design and manufacturing places distinctive demands on multimodal foundation models: the model must see engineering drawings and 3D geometry screenshots, write correct parametric-modelling scripts and Windows COM API code, and cover the full range from single parts to assemblies. General-purpose multimodal models fall short on these tasks, while single-task fine-tuning is too narrow to support the diverse calls that upper-layer agents issue. We build IndustryForge-27B on top of Qwen3.5-VL-27B by curating and integrating six industrial-CAD sub-corpora totalling $\sim$52k multimodal samples---CAD Visual QA (CAD-VQA), parametric CAD code (text2cadquery), assembly-level CAD code (text2cadquery-assembly), and three COM sub-corpora for Inventor / SolidWorks (com_2d / com_3d / com_assembly)---and training with a unified multi-task SFT recipe. Across four CAD-domain benchmarks IndustryForge-27B lifts the base model by $+33.65$~pp on average and outperforms the strong closed-source model GPT-5.4 on all four; across eleven general-capability benchmarks it retains, and slightly improves upon, the base model ($+1.56$~pp mean, no catastrophic forgetting). IndustryForge-27B will serve as the common substrate for downstream industrial-agent projects, providing a unified starting point for a full-stack industrial agent that spans from CAD design to industrial-software operation, from parts to assemblies, and from single-shot generation to closed-loop self-improvement. 

---
# SKILL-KD: Contrastive Skill Distillation for LLM Agents 

**Authors**: Qiming Shi, Yibo Dou, Jiawen Zhu, Yulong Tao, Linbo Jin, Zhaolu Kang, Yunfan Zhou, Di Weng  

**Link**: [PDF](https://arxiv.org/pdf/2607.28048)  

**Abstract**: Skill-based prompting has become a practical mechanism for improving large language model (LLM) agents, yet existing skill acquisition methods often treat skills as experience summaries, memory entries, or direct summaries of successful demonstrations. This creates a mismatch for weaker student agents: when a student fails because it lacks task knowledge or operational strategy, its failed trajectory may not contain enough evidence to infer the missing behavior, while the teacher trajectory may be too implicit to be internalized as reusable guidance. We propose SKILL-KD, a contrastive skill distillation framework that treats skills as an explicit distillation medium between agents of different capabilities. Given a student failure and the teacher trajectory on the same task, SKILL-KD distills their actionable discrepancy into a textual skill patch, evaluates the patch by re-running the student, and iteratively refines the patch when the student still fails. To prevent repeated local updates from causing skill drift, SKILL-KD further maintains trace-linked edit histories and performs Drift-Aware Skill Consolidation, deciding whether each patch should add a new rule, delete or modify an existing rule, or be skipped. Across five agent benchmarks and two student settings, SKILL-KD consistently improves frozen student agents over fixed-model adaptation baselines. 

---
# DataClawEval: A Benchmark for Data Engineering Agents in Real Industrial Harness 

**Authors**: Debin Meng, Jiaming Yang, Zefang Zong, Tengyue Xu, Haining Xie, Yang Li, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.28033)  

**Abstract**: Large language models (LLMs) and LLM-based agents are increasingly being deployed to automate complex workflows, promising to revolutionize data management and processing. However, existing benchmarks predominantly focus on simplified Text-to-SQL translation or data analysis, leaving the critical and complex domain of end-to-end data engineering largely unexplored. To bridge this gap, we introduce DataClawEval, the first comprehensive benchmark designed specifically to evaluate the end-to-end task completion capabilities of autonomous agents in real-world data engineering scenarios. Built upon production-grade code authored by professional enterprise data engineers, it comprises 100 rigorous, end-to-end tasks spanning five execution engines: PySpark, MySQL, HiveSQL, PrestoSQL/Trino, and FlinkSQL. Rather than non-deterministic LLM-as-a-judge scoring, each task is executed within a case-specific, isolated sandbox and graded by deterministic, rule-based scripts. Evaluating 16 frontier agents exposes critical limitations: The strongest model attains only 74.9 overall, and no single model dominates, as each excels on a different engine, revealing strict domain specialization rather than omnipotent proficiency. Thus, autonomous data engineering remains a formidable, unresolved challenge. We release our dataset, containerized environments, and deterministic evaluation scripts at this https URL 

---
# MUL-T: Decoding Spatial Cellular Architecture in Multiplexed Tissue Images 

**Authors**: Farzaneh Seyedshahi, Kai Rakovic, Adalberto Claudio Quiros, John LeQuesne, Ke Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.28030)  

**Abstract**: Understanding tissue organisation in multiplexed imaging requires modelling both cellular phenotypes and their spatial context. Existing approaches typically rely on handcrafted features, such as marker intensity statistics or cell-type proportions, which often fail to scale or generalise across cohorts with heterogeneous marker panels. We introduce MUL-T, a lightweight transformer framework that reframes tissue architecture as a masked contextual prediction task over discrete cell tokens. By learning contextualised [CLS] embeddings without task-specific supervision, the model captures higher-order cellular interactions while remaining computationally efficient. We evaluate MUL-T on several clinically relevant downstream tasks, including core-level tumour pattern classification, patient-level grading, PD-L1 positivity prediction, and cross-dataset treatment response prediction. Across tasks, MUL-T consistently outperforms classical feature-based baselines and achieves performance comparable to a foundation ViT model, despite substantially fewer parameters and lower training cost. 

---
# MMLDSum-LLM: Multimodal Long-Document Summarization with Visual-Alignment and Keyword-Aware 

**Authors**: Xianpeng Zhang, Jiahua Yang, Dongyu Chen, Lei zhang, Jian Ma, Xu guohuan, Haonan Lu, Tianhuang Su, Chuangchuang Wang, Kai Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28006)  

**Abstract**: Multimodal long documents are core carriers of professional knowledge, where critical evidence is sparsely distributed across paragraphs and modalities. This easily causes key information omission and cross-modal hallucinations in summarization by multimodal LLMs. These issues stem from attention drift in long-range dependency modeling and gaps in inter-modal alignment. To address this, we introduce MMLDSum-Bench, a high-quality benchmark for multimodal long-document summarization, covering multiple domains, context-length scales, and visual-textual modality distributions. We further propose MMLDSum-LLM, a reproducible two-stage training framework that combines supervised fine-tuning with visual-alignment weighted loss and keyword-aware weighted loss, followed by GRPO with a multi-objective reward (keyword coverage, image-text alignment, ROUGE, and length control). Extensive experiments on MMLDSum-Bench, comparing against leading closed-source and open-source multimodal models under a unified evaluation protocol - including LLM-as-a-judge scoring, atomic-claim precision/recall, image-text alignment (ITA), and ROUGE - demonstrate that our approach significantly improves key-information coverage and cross-modal consistency. 

---
# SKIMIX: Multi-Agent Harness-Time Scaling with Skill Mixture for Dynamic Harness Engineering 

**Authors**: Jia Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27994)  

**Abstract**: AI agents increasingly rely on large skill libraries, but selecting, combining, and maintaining skills remains difficult. We propose SKIMIX, a multi-agent framework in which agents with different skill portfolios collaborate through iterative refinement. SKIMIX combines embedding-based skill retrieval, submodular anti-dilution routing, and adaptive skill evolution. Across six reasoning benchmarks, multi-agent collaboration substantially improves open-ended mathematical reasoning but offers limited or negative gains on multiple-choice tasks. Agent-count scaling is non-monotonic, and most improvements arise during the first refinement round. These results show that task characteristics determine whether skill-level ensembles help and provide practical guidance for scalable agent design. 

---
# Share the Judge, Learn the Deferral: Where Specialization Helps LLM Evaluation 

**Authors**: Weining Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27984)  

**Abstract**: Agentic systems have widened the gap between producing candidate outputs and reviewing them. This paper asks a practical architectural question: should domain specialization be built into an evaluator's weights, or into the rule that decides when its judgment can be trusted? We study 99,952 public, rubric-conditioned examples. Supplying the correct rubric improves locked-test accuracy by 2.11 points over a response-only control; replacing it with an unrelated rubric costs 2.66 points. Dividing the same training corpus among eight criterion-family LoRA judges, however, loses 10.05 points and cuts audited coverage at a 5% risk target from 24.44% to 5.43%. Matching the bank's stored capacity with one rank-64 adapter does not reproduce this loss. Nor is the result explained by learning rate or optimizer steps. Initializing the family adapters from a shared, trained judge recovers test accuracy to 76.85%, 19.94 points above scratch training at the same learning rate (95% interval 18.88-21.02). The result changes when specialization governs deferral rather than judgment. On RewardBench 2, learned correctness heads route examples through a 0.6B-4B-8B cascade without changing any reward score. Across 20 locked repartitions, the cascade attains 89.40% accuracy, compared with 84.75% for 8B alone, at 0.415 normalized parameter compute. Every run passes an exact one-sided 95% risk audit; margin-based rules remain near 84.8% accuracy while using at least 0.94 compute. These results suggest a qualified design rule: share the learning of judgment until there is enough data to justify a split, and place domain-specific adaptation in an audited release boundary. 

---
# MARS-RA: Rank Aggregation for Credit Assignment via Multimodal Comparisons in Embodied Multi-Agent Cooperation 

**Authors**: Dawei Wang, Di Zhao, Xinyuan Liu, Marci Chi Ma, Xiaoyang Liu, Chengming Zhou, Gary Ushaw, Richard Davison  

**Link**: [PDF](https://arxiv.org/pdf/2607.27967)  

**Abstract**: Credit assignment is a fundamental challenge in cooperative multi-agent reinforcement learning, particularly in embodied AI settings characterized by limited and delayed feedback as well as dynamically changing numbers of active agents. We propose MARS-RA, a framework that reformulates credit assignment as a rank aggregation problem using contribution-based pairwise comparisons among agents generated by large multimodal models. This shift from absolute to relative estimation ensures robustness against noise and dynamic agent participation, converting comparison results into contribution scores for potential-based reward shaping. We provide theoretical justification for the convergence and robustness of the proposed framework, and show that Shapley values can be used as an interpretive reference. Experimental results on challenging tasks of different types indicate that MARS-RA can guide agents toward effective cooperation. 

---
# From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents 

**Authors**: Xu Xia, Jinghua Piao, Min Yang, Xiaochong Lan, Jiaju Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27937)  

**Abstract**: Recent work on LLM agents is shifting from external capability elicitation to capability internalization, enabling agents to retain useful skills without retrieval at inference time. On-policy self-distillation (OPSD) offers a promising direction, but many existing methods typically supervise students by scoring actions along student-generated trajectories. Such supervision has two limitations: teacher preferences are not validated by environment outcomes, and action-level scores underuse information from student rollouts, teacher rollouts, and their behavioral relationship. We therefore advocate outcome-verified teacher supervision and comparative learning over teacher-student trajectories. Based on this view, we propose Outcome-Verified Comparative Self-Distillation (OVCSD). OVCSD organizes failed student rollouts into a prefix tree, adaptively invokes a skill-conditioned teacher from student-reached states, and retains only outcome-verified successful continuations. It then applies localized comparative learning at the first state-aligned divergence and distills the post-divergence teacher suffix to transfer completion behavior. Experiments on ALFWorld and WebShop across three model scales show that OVCSD consistently outperforms skill-free RL and existing self-distillation baselines, achieving up to 29.7 and 5.4 absolute success-rate gains over the strongest baselines on ALFWorld and WebShop, respectively, while adding less than 3% privileged interaction during training. 

---
# Shapes from Examples: Foundations of Shape Learning in Recursive SHACL 

**Authors**: Bente Gortworst, Cem Okulmus, Magdalena Ortiz, Anni-Yasmin Turhan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27934)  

**Abstract**: SHACL shapes enable data graph validation, making automatic shape learning essential for knowledge graph applications. We investigate the well-known fitting approach to this task: given sets P and N of positive and negative example nodes from an input graph, compute a shape expression C, possibly using shape names defined in a recursive shape catalogue, that validates at every node in P and none in N. We focus on the case where C is written in a core fragment of SHACL corresponding to the Description Logic ELI. For the catalogue, we consider the well-founded, stable, and supported semantics. We address fitting existence and most specific fitting computation, establish tight exponential-time upper bounds for both problems, and obtain polynomial bounds for relevant special cases. 

---
# The Geometric Nature and a Free Proxy for Flow-Matching Uncertainty 

**Authors**: Ziyang Rao, Yiren Zhao, Weiyu Guo, Ben Fei, Yandong Guo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.27933)  

**Abstract**: Flow matching (FM) has become a popular action head paradigm for modern embodied models. However, as a conditional generative model, it does not explicitly expose its inherent uncertainty, producing faulty action chunks even when it misinterprets the scene or encounters out-of-distribution (OOD) inputs. Therefore, determining when an FM-generated action can be trusted is essential for safe deployment, yet existing uncertainty estimation methods on real-time control suffer from several issues: extra training budget, high computational overhead, and low generalization ability. In this work, we provide a geometric interpretation of FM uncertainty in the velocity field, showing that uncertainty manifests as deviation from an ideal affine-isotropic contraction field. Building on this observation, we introduce denoising acceleration ($\mathrm{accel}$), a highly-generalizable and cost-free uncertainty proxy that measures the bending of the denoising trajectory from a single forward pass, without additional model evaluations, training, or resampling. We theoretically and empirically demonstrate that $\mathrm{accel}$ is a faithful proxy for FM uncertainty and further test its utility in online failure detection. Results show that $\mathrm{accel}$ identifies failing rollouts well before termination, matching or even outperforming costly resampling- and training-based baselines across settings under realistic deployment budget. Code and demos available at: this https URL. 

---
# Meta-Task: Turning Terminal Task Synthesis into a Terminal Task for Scalable Agent Training 

**Authors**: Zhihong Pan, Jiyuan He, Kai Zhang, Yupeng Han, Ze Liu, Yuze Zhao, Yongcong Ye, Zhaohua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27929)  

**Abstract**: Training terminal agents at scale requires diverse, verifiable terminal tasks and high-quality interaction trajectories, yet acquiring such data remains a significant challenge. Existing synthesis methods face two key limitations: (1) weak reliability caused by the disconnect between task generation and real execution, and (2) limited diversity and scalability due to dependence on existing repositories. We propose Meta-Task, a framework that redefines terminal task synthesis as a Terminal-Bench-format task itself: an agent operates within a real container environment to iteratively generate, execute, and verify tasks, so that synthesized components are checked for internal consistency and executability within the generation loop itself. Building upon this, we decouple the target task requirements along multiple dimensions, introduce a multi-phase mechanism that dynamically designs novel task specifications before producing the actual tasks, and incorporate optional external material support to enhance diversity and realism. We additionally apply LLM-as-Judge filtering to ensure the quality of the final training data. Experiments on Terminal-Bench 2.0 show that fine-tuning on only 3,221 Meta-Task synthesized trajectories achieves 22.5% and 31.8% Avg Pass@1 for Qwen3-14B and Qwen3-32B respectively, outperforming concurrent approaches with significantly less training data. 

---
# One Anchor for All: Unified Multilingual and Multimodal Safety Alignment for LVLMs 

**Authors**: Enyi Shi, Fei Shen, Chuancheng Shi, Linxia Zhu, Shuyi Miao, Jinhui Tang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2607.27917)  

**Abstract**: As large vision-language models (LVLMs) are deployed globally, the combination of multilingual instructions and visual information makes malicious attacks more covert and sophisticated than ever before. However, existing methods isolate language and modality defenses, which, coupled with the scarcity of safety data and high fine-tuning costs, makes it difficult for models to defend against compound attacks. To address this severe challenge, we propose a neuron-level cross-dimensional safety alignment framework driven by modality- and language-shared safety neurons (MLS-Neurons). First, we identify monolingual and unimodal safety neurons by comparing responses to harmful and benign samples, quantifying functional saliency through activation strength and downstream impact. Then, by intersecting these unimodal neurons within each language, we extract modality-shared safety neurons (MS-Neurons) responsive to both visual and textual risks, bridging the safety representation gap between modalities. Furthermore, using English as a semantic anchor, we intersect MS-Neurons across languages to identify modality- and language-shared safety neurons (MLS-Neurons), serving as key defenses against compound attacks. Finally, we update only this minimal subset of shared neurons (~0.03% of parameters), transferring English-only safety supervision to multilingual and multimodal scenarios. Extensive experiments show that our method significantly outperforms state-of-the-art approaches across diverse multilingual and multimodal safety benchmarks while preserving general utility. 

---
# IFHierBench: Hierarchical Instruction Following for Large Language Models 

**Authors**: Yuetian Mao, Chunyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27912)  

**Abstract**: Instruction-following ability is critical for deploying large language models in real-world applications, where downstream components depend on the output satisfying specific constraints. Modern deployments increasingly handle the full task in a single LLM call, with one prompt specifying a layered output whose overall artifact, structural sections, and nested fields must each satisfy concrete constraints. Existing instruction-following benchmarks treat the constraint set as a flat list applied uniformly to the response, so they cannot scope a check to a particular section of the output. We introduce IFHierBench, a hierarchical instruction-following benchmark of 600 prompts stratified across four constraint-tree depths and 35 distinct constraints, each prompt paired with a deterministic checker that verifies satisfaction at every scope. Evaluating seven leading proprietary and open-weight models, we find that even the strongest model only marginally exceeds 50% prompt-level accuracy and that accuracy degrades sharply as constraint depth grows. Reliably following nested constraints remains a substantial gap for current LLMs, motivating future training methods that consider constraint adherence at finer granularity to achieve better instruction-following ability. 

---
# A Cross-Architecture Audit of Direction-Based Inference-Time Defences in Vision-Language Models 

**Authors**: Xiangyu Yin, Tora Bodin, Rohan Menon, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27910)  

**Abstract**: Inference time defences against vision language model jailbreaks often subtract a calibrated direction from the residual stream at a chosen decoder layer. We compare five defence candidates across 15 model and layer cells from four architectural families under a magnitude controlled protocol that matches the intervention size for each prompt and pairs every direction with a random control of the same norm. The candidates are the mean image conditioning shift, a CMRM style refusal direction, a ShiftDC style attack specific residual, a prompt instruction to ignore the image, and a random control. No single candidate dominates on both refusal recovery and utility preservation. The image conditioning shift leads on LLaVA 1.5 and Pixtral 12B and is the only candidate whose utility loss remains at the measurement noise floor in every family. The prompt instruction leads on Qwen2.5 VL, while the attack specific residual leads on Qwen2 VL 2B. The image conditioning direction is direction specific in 13 of 15 cells, but strongly architecture specific and nontransferable across the only dimension compatible pair, LLaVA 1.5 13B and Pixtral 12B. We also connect text only and multimodal refusal geometry. The CMRM direction has positive cosine alignment with the image conditioning shift in all 15 cells, with mean 0.35, range 0.17 to 0.65, 15 to 25 times the random vector null, and a sign test p value of about 3e-5. These results show that the two recipes recover partially overlapping geometry and that direction based defences should be calibrated separately for each language decoder family. 

---
# MMHBench: A Multi-Perspective Benchmark for Mental Health Understanding in Long-Form Videos 

**Authors**: Jinpeng Hu, Erqiang Wang, Shan Wang, Zhuo Li, Peipei Song, Xun Yang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27895)  

**Abstract**: Mental health understanding in long-form videos requires nuanced reasoning over observable behavior, interpersonal context, and latent psychological states. Existing benchmarks largely reduce this task to coarse-grained classification, providing limited insight into whether models truly understand psychological phenomena or rely on superficial correlations. To address this limitation, we introduce MMHBench, a comprehensive multimodal benchmark for multi-perspective mental health understanding, comprising 268 long-form videos and 2,184 carefully curated questions. MMHBench organizes the evaluation into two complementary settings: (1) third-person assessment, consisting of 605 questions that focus on the interpretation of observable behaviors and multimodal evidence, and (2) first-person perspective-taking, comprising 1,579 questions that require perspective-conditioned reasoning to identify the interpretation of the mental state supported by the available multimodal evidence. We propose a Multi-Agent Question Generation (MAQG) framework that simulates diverse social roles to synthesize questions from multiple perspectives. The generated questions are refined through multi-role feedback and iterative optimization, followed by expert-guided verification to ensure quality and validity. Extensive evaluation of 22 representative multimodal large language models (MLLMs), spanning both open-source and leading closed-source models, demonstrates that long-form video mental health understanding remains highly challenging. 

---
# Dynamic Spectral Filtering for Temporal Graph Learning: Learning Evolving Propagation Operators 

**Authors**: Yan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2607.27891)  

**Abstract**: Temporal graph learning is commonly organized around the evolution of node states or the encoding of interaction histories. We study an underexplored, operator-centric question: should the graph propagation mechanism itself evolve over time? We introduce Dynamic Spectral Filtering (DSF), which represents propagation at snapshot t by a Chebyshev polynomial filter with vector-valued, time-dependent coefficients. DSF explicitly treats these compact multi-order coefficients as recurrent temporal states. A recurrent branch proposes updates, while multiplicative global and order-specific gates regulate their magnitude. The temporal state is independent of the number of nodes. On MOOC, Wikipedia, and Reddit temporal link-prediction benchmarks, converged DSF runs attain AP scores of 0.7851, 0.9088, and 0.9860, respectively, with 93K to 133K trainable parameters, 68 to 182 MB peak GPU memory, and 1.6 to 2.1 seconds of training per epoch. Against the closely related DEFT baseline, DSF is better on MOOC, within 0.001 AP on Reddit, and modestly lower on Wikipedia, while using 8.3 to 8.6 times fewer parameters, 25 to 33 times less GPU memory, and 5 to 19 times less time per epoch. Relative to all measured alternatives, it uses 3.3 to 38.6 times less GPU memory. These results support direct spectral-response evolution as a useful temporal inductive bias when computational efficiency is a first-class requirement. 

---
# Not All Tokens Deserve Equal Credit: Counterfactual Sensitivity Credit Reallocation for Long-CoT Reasoning 

**Authors**: Qiangqiang He, Zhongheng Wu, ZiJian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27888)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is central to improving long-CoT reasoning in large language models. Critic-free methods such as GRPO convert response-level rewards into advantages and uniformly broadcast them across tokens, overlooking their unequal contributions to the final outcome. On-policy self-distillation (OPSD) instead provides dense distributional supervision by minimizing the forward KL divergence between an unprivileged policy and a privileged self-teacher, implicitly assuming that the resulting likelihood shifts encode reliable answer-aligned information. We test this premise by fixing each sampled trajectory and re-scoring it under two opposing outcome conditions, one asserting correctness and the other incorrectness. Most affected tokens shift in the same direction under both conditions, with few sign reversals and substantial overlap in the induced optimization signals. Large shifts also concentrate on highly substitutable surface-form tokens, whereas tokens carrying problem-specific reasoning content are less sensitive. These findings show that privileged shifts fail to provide reliable answer-aligned directions, while their magnitudes primarily reflect counterfactual sensitivity rather than token-level learning value. Based on these observations, we propose Counterfactual Sensitivity Credit Reallocation (CSCR), a simple extension of GRPO that reduces credit for highly sensitive tokens and renormalizes token-level advantages to preserve both the original credit budget and verifier-determined direction. On long-CoT mathematical reasoning benchmarks, CSCR consistently outperforms GRPO baseline with the same number of policy updates. Targeted ablations further corroborate our diagnosis: privilege-induced directions are unreliable, moderate downweighting is most effective, and stronger modulation destabilizes optimization. 

---
# An Empirical Study of Coordination Mode as the First-Class Citizen in From-Scratch Multi-Agent Coding 

**Authors**: Yanyu Ren, Yunfeng Bai, Xizheng Wang, Li Chen, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27877)  

**Abstract**: Multi-agent vibe coding promises to accelerate software development, yet existing benchmarks rely on synthetic environments that ignore practical time and monetary costs, conflate reasoning with communication, and reward only superficial completion. We introduce multi-agent from-scratch evaluation benchmark, MSEval, evaluating multi-agent coding on real-world tasks. Grounded in 10 authentic, full-stack projects across 10 domains, MSEval scores performance using hierarchical requirements and deterministic rubrics.
Its execution engine, LegoGent, tests 10 collaboration topologies where agents coordinate via periodic sync intervals and deploy through native CI/CD pipelines. Concurrently, the automated grader TAgent dynamically probes implementations to jointly measure functional success, latency, and prefix-cached token cost. Across 100 runs, MSEval reveals that organizational topology rivals model capability in shaping the speed--cost--quality trade-off. For identical tasks and models, varying the topology shifts scores by over 30 points and doubles wall-clock time. Structured pipelines converge fastest with the highest quality, whereas heavy managerial oversight degrades performance. Ultimately, MSEval establishes a rigorous, reproducible standard for measuring how multi-agent teams actually build software. The benchmark is released at this https URL. 

---
# Search as Computation Allocation 

**Authors**: Alexander Tuisov  

**Link**: [PDF](https://arxiv.org/pdf/2607.27871)  

**Abstract**: Many algorithms spend an internal resource before returning a decision and are evaluated only by the quality of that terminal output. We formalize such procedures as terminal computation-allocation problems: costly computations produce observations, update beliefs about a latent environment, and matter only through terminal decision loss. Bellman equations characterize optimal allocation under fixed budgets, priced computation, and exact certification. We then relate value of computation (VOC) to information. Mutual information equals myopic VOC under log loss, whereas under simple regret VOC is a knowledge-gradient quantity; moreover, information gain can rank computations arbitrarily poorly, although it gives a one-sided upper bound on VOC. Bandit pulls, tree simulations, and node expansions illustrate the same model under different computation topologies. Finally, under an explicit frontier-resolution and heuristic-error model, maximizing approximate VOC recovers weighted A*, with A* and greedy best-first search as limiting cases. The theory identifies a shared decision problem without asserting that one acquisition rule is universally optimal. 

---
# Orca: Neural Operators for Causal Reasoning in Continuous Time 

**Authors**: Gerrit Großmann, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2607.27867)  

**Abstract**: Structural causal models are the standard language for reasoning about interventions and counterfactuals, but they describe static variables, typically measured once, and usually forbid cyclic dependencies. Many systems we care about, such as patients, climates, and economies, instead evolve continuously in time, are observed at irregular time points, and contain feedback loops. We argue that neural operator learning provides a natural foundation for causal reasoning in this setting, and propose Orca, a framework in which each node of the causal graph is a function of time and each mechanism is a learned map between function spaces. We extend existing neural operator architectures to express causal mechanisms: a mechanism computes the function value of a node from its parent nodes by taking several parent functions as input, respects the arrow of time, and treats latent exogenous noise as a function that can be inferred and reused for counterfactuals. We formalize the model class and demonstrate counterfactual reasoning on synthetic continuous-time examples. Code is available at this https URL 

---
# Back to All-Entity Ranking: Sampler-Dependent Evaluation in Continuous-Time Dynamic Graphs 

**Authors**: Minwoo Yu, Young-guk Ha  

**Link**: [PDF](https://arxiv.org/pdf/2607.27861)  

**Abstract**: Next-destination prediction in continuous-time dynamic graphs (CTDGs) commonly ranks an observed interaction against sampled negative destinations. The resulting score is conditional on both the negative distribution and the number of candidates chosen by the researcher. We show that a non-uniform negative distribution changes the Bayes-optimal ranking, while even a finite candidate set drawn uniformly can destabilize model rankings and measured module effects.
Time-varying source-destination history membership and model operations that use this information directly transmit the sampler's influence to the evaluation score. We examine this mechanism using a factorial evaluation of repeated and new positives against seen and unseen negatives, a minimal scorer based solely on pair-history membership, and controlled representation interventions. Across six models on LastFM, MOOC, Reddit, and Wikipedia, at least one model pair changes relative order between the expected Uniform-20 metric and the full catalog on three of the four datasets. The measured effect of the same module also changes in magnitude and direction with the candidate-set size and training objective.
These results establish that model-superiority and ablation conclusions from sampled-negative benchmarks are conditional on the stated candidate configuration. All-entity ranking evaluates every destination in a fixed catalog, eliminating negative-selection freedom and sampling variation while retaining the original CTDG scorer. We therefore recommend all-entity ranking as the primary evidence for architecture comparisons on CTDG benchmarks with an enumerable, fixed destination catalog. 

---
# Simplifying Neural Networks During Training 

**Authors**: Lorenzo Sciandra, Samuele Fonio, Roberto Esposito  

**Link**: [PDF](https://arxiv.org/pdf/2607.27854)  

**Abstract**: Understanding and exploiting the training dynamics of overparameterized deep neural networks remains a central challenge in modern machine learning. Recent evidence on Neural Collapse (NC) shows that class representations and classifiers exhibit highly structured geometry, while the Tunnel Effect suggests that only a subset of layers is essential for feature extraction. We combine these two perspectives and propose an NC-inspired training framework for simplifying deep networks during training. Our method monitors representation dynamics through the Inverse Fisher Criterion, a stable and efficient proxy for the variability collapse behavior, to identify both the split point between feature extraction and classification and the training stage at which simplification becomes viable. We then replace the trailing layers with a lightweight classification head and continue training the reduced model. Experiments on image-classification benchmarks across MLP, VGG, and ResNet architectures show that the proposed method achieves substantial parameter reductions while maintaining accuracy comparable to that of the full model. Code to reproduce the experiments can be found at: this https URL. 

---
# Virtual Process Dossier: A Process-Aware Data Catalogue 

**Authors**: Lukas Kubelka, Alexander Bott, Frank Döhner, Saksham Kiroriwal, Georg Zeeb, Julia Butte, Julius Pfrommer, Jürgen Beyerer, Tobias Käfer  

**Link**: [PDF](https://arxiv.org/pdf/2607.27840)  

**Abstract**: We propose the Virtual Process Dossier (VPD), a Knowledge Graph-based data catalogue that also captures workflow provenance. We developed VPD for multi-stage manufacturing use-cases where downstream AI-based optimization tasks require to distinct between datasets generated during individual workflow steps. VPD provides these datasets in a FAIR manner and makes both prospective and retrospective workflow provenance explicit. Our contributions are: (1) the VPD ontology that serves as the catalogue's semantic core; (2) the VPD provenance framework that integrates ontology instantiation into the production environment; and (3) the VPD user interface that provides human-centered interaction with the VPD Knowledge Graph. The ontology and code are available at this https URL . 

---
# Crossing the Margin Cliff: Toward Relearn-Robust LLM Unlearning via Margin Calibration 

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27836)  

**Abstract**: Large language model unlearning is consistently fragile under relearn attacks. On TOFU, fine-tuning on twenty forget examples substantially recovers held-out forget-set ROUGE for every method we evaluate, and we trace this fragility to optimization geometry. The per-token answer margin of fourteen post-hoc methods spanning gradient, preference, and distillation families converges into a narrow band above the retain reference in 41 of 42 method--size cells, a regularity we call the margin cliff. We prove that this cliff follows whenever the retain coupling holds the diagnostic log-odds of forget content above a floor, a condition that token-saturating losses induce at stationarity and that we verify directly on 34 of 42 cells. Margin Calibration (\textsc{MC}) is a plug-in polish adding a non-saturating margin hinge anchored at the reference's per-token margin plus a KL probe on a disjoint instruction corpus, restoring forget-side pressure where the native loss saturates. Under a stated gradient-dominance condition, whose on-trajectory gradient signature we measure by instrumenting the polish, its stationary set lies on the cliff-crossing side, yielding an attack-budget upper bound on the relearn margin lift. Across TOFU (three Llama-3 sizes, three forget tiers), MUSE-News on Llama-2-7B-hf, and a Phi-3.5 panel, a single frozen configuration wins all 14 head-to-head forget aggregates and all populated relearn cells (panel-mean post-attack ROUGE-L $0.41$ to $0.18$) and lowers raw membership AUC on 13/14, with reduced retain-side utility as the main cost. A deployment variant matches these gains without a retain-trained reference. 

---
# MemTxn: A Transaction Boundary for Source-Supported Updates and Complete-State Recovery in Agent Memory 

**Authors**: Hanshuai Cui, Zhiqing Tang, Zhi Yao, Fanshuai Meng, Qianli Ma, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.27834)  

**Abstract**: Persistent memory lets long-running large language model agents reuse information across sessions and tasks. Yet errors in writable memory can persist and corrupt future behavior. Existing systems improve storage and retrieval, but they do not provide a transaction boundary for reliable updates and recovery. We therefore propose MemTxn, a governance layer outside the answer model. MemTxn verifies whether an update is supported by its source. It also selects the visible version when facts conflict and restores the application-visible state after a fault. The system uses Ordered PatchTest to validate writes, a Temporal Resolver to select versions, and a durable snapshot journal to recover state. On an item-disjoint audit, MemTxn accepts all 60 supported originals and rejects all 179 hard negatives. Under persistent multi-key faults on LongMemEval-S and LoCoMo states, it restores the complete declared active map without knowing the actual physical write set. On MemoryAgentBench FactConsolidation, MemTxn achieves the highest average F1 across all twelve answer-model configurations. It outperforms Dense by 17.06--24.07 points in five representative settings. 

---
# Sign Language Question Answering: A New Task, Benchmark, and Baseline for Sign Language Understanding 

**Authors**: Shiwei Gan, Lichen Wang, Xiao Liu, Yafeng Yin, Kuizhuang Liu, Sanglu Lu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.27826)  

**Abstract**: Recent advances in sign language (SL) understanding (SLU) have led to remarkable progress in tasks such as continuous SL recognition and SL translation. However, these tasks are designed with predefined objectives, requiring models to learn a fixed mapping from sign videos to glosses or spoken-language sentences. As a result, they provide only a limited assessment of whether a model truly understands the semantic content of SL videos. To address this limitation, \textbf{we first propose a new task, Sign Language Question Answering (SLQA)}, which evaluates SL understanding by requiring models to answer arbitrary natural language questions about SL videos. Unlike previous SLU tasks, SLQA provides a more flexible and comprehensive evaluation framework that assesses multiple reasoning capabilities beyond recognition and translation. To facilitate this task, \textbf{we further construct two SignQA benchmarks} based on PHOENIX14T and CSL-Daily by automatically generating question-answer pairs from existing gloss and sentence annotations using carefully designed templates. The resulting datasets cover five complementary question categories, including position reasoning, structural reasoning, visual search, gloss recognition, and translation understanding. \textbf{Finally, we propose a simple yet effective baseline model} equipped with a Question-Conditioned Modulated Temporal Downsampling module and an in-domain knowledge transfer strategy, enabling effective knowledge transfer from existing SLU tasks while enhancing question-aware temporal feature modeling. Extensive experiments demonstrate that our baseline consistently outperforms representative vision-language models across all question categories, establishing a strong benchmark for future research on SLQA. Datasets are available at:{this https URL}. 

---
# STEREODISCO: Discovering Stereotypicality in LLMs 

**Authors**: Farane Jalali Farahani, Corina Dima, Mojtaba Nayyeri, Raphael H. Heiberger, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2607.27824)  

**Abstract**: LLMs encode, convey, and perpetuate stereotypes. Prior computational research focuses on a small set of semantic axes investigated in social psychology, and operates on word embeddings produced by language models, leaving open which other semantic axes carry stereotypical associations in LLMs and how LLMs internally represent such axes. We introduce STEREODISCO, a framework that adapts the semantic differential method (Osgood et al., 1957) to the systematic study of stereotypes in LLM internal representations. STEREODISCO constructs approx. 2,000 candidate semantic axes from WordNet antonym synsets, recovers each as a geometric axis in the LLM's activation space via probing, and identifies stereotypical axes via a statistical test over concept projections. As a case study, we apply STEREODISCO to social group stereotypes with LLAMA-3-8B-INSTRUCT and MISTRAL-7B-INSTRUCT. We find that the two LLMs agree with each other on social group ratings more than with humans, suggesting that LLM-encoded stereotype content diverges from that documented in social psychology. We also discover stereotypical axes not investigated in prior work -- including humble vs. proud, narrow-minded vs. broad-minded, and cowardly vs. brave, which human annotators independently confirm. 

---
# MemeBench: What LVLMs Miss When Interpreting Culture-Dependent Memes 

**Authors**: Weihang Wang, Kainan Tu, Jielei Zhang, Run Yang, Boheng Sheng, Yuchen He, Yu Xie, Pengyu Chen, Peiyi Li, Huyang Sun, Longwen Gao, Zhouhui Lian  

**Link**: [PDF](https://arxiv.org/pdf/2607.27798)  

**Abstract**: Large vision-language models have improved at describing visual content, but accurate descriptions do not ensure interpretation when meaning depends on knowledge beyond the pixels. Memes expose this gap because they rely on cultural entities, background knowledge, and community conventions. Most meme benchmarks reduce interpretation to labels or holistic scores, obscuring where an explanation breaks down. We introduce MemeBench, a diagnostic benchmark of 1,253 Chinese and English memes with human-written references and quality-controlled VIKR annotations, centered on anime, comics, games, and adjacent online subcultures. Its VIKR schema decomposes explanations into Visual clues, Identity links, Knowledge units, and Reasoning mechanisms. Across 26 LVLMs, every model covers visible content more reliably than the knowledge needed to interpret it, and even the strongest retains a 22.6% Visual-Knowledge gap. To test whether this diagnosis can guide improvement, we introduce KAR, an entity-guided retrieval baseline built on CultureBase. Across four controlled models, KAR raises VIKR Success by 3.6-7.4% and, compared with generic retrieval, repairs more answers and breaks fewer. Yet both retrieval conditions improve Identity and Knowledge while reducing Visual coverage in every comparison. MemeBench reveals whether an interpretation succeeds, what is missing, and whether targeted evidence fills the diagnosed gap. 

---
# Annotating Topical Legal Insights from Case Proceedings 

**Authors**: Subinay Adhikary, Dwaipayan Roy, Debasis Ganguly, Shouvik Kumar Guha, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2607.27792)  

**Abstract**: In this paper, we mainly concentrate on finding concepts or topics from the legal case proceedings, since adopting a structured representation for legal documents, as opposed to a mere bag-of-words flat text representation, can significantly enhance processing capabilities. To achieve this objective, we put forward a set of diverse concepts for legal case proceedings. With this motivation, we propose LeDA, a system for Legal Data Annotation. The system offers the generic functionality of annotating and adjudicating entities or concepts within documents via a web-based interface. A novel feature of our system is that it allows to dynamic create new tags for annotation, which is a particularly useful provision for situations where there exists no pre-defined ontology for the entities (concepts) that need to be annotated - these being rather discovered by annotators as they continue examining more documents. The system that we demonstrate is currently in use to annotate a set of concepts from legal documents to construct semantic representations of documents as bags of concepts that can then be used for several downstream tasks, such as prior case retrieval, judgment prediction, and so on. Along with the system features in general, we also describe how LeDA was used by 3 assessors to annotate and adjudicate legal concept names from Indian Supreme Court case proceedings. 

---
# SpecCal: Ambiguity-Aware Candidate Calibration for Infrared Spectrum-Based Molecular Structure Reconstruction 

**Authors**: Yixuan Chen, Bo Liu, Yusen Tan, Guokun Yang, Wenjie Du, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2607.27788)  

**Abstract**: Inferring molecular structures from infrared (IR) spectra is a fundamental yet challenging problem. A key difficulty is that an IR spectrum provides limited structural information: different molecules may share similar functional groups and local vibrational patterns, leading to highly similar spectral responses. Thus, even when an observed spectrum has a unique underlying structure, reconstructing it from the spectrum remains ambiguous. Existing IR-to-molecule models usually generate a ranked set of candidate molecules, but this set is largely determined by the model's learned generation preference and may not fully capture the structures that best satisfy the observed spectral constraints. To address this limitation, we propose SpecCal, a training-free candidate calibration framework for IR-to-molecule prediction. SpecCal operates on the candidate outputs of existing base models and improves the prediction set by re-ranking current candidates while introducing additional structurally plausible alternatives guided by spectral consistency. The framework is plug-and-play and model-agnostic, requiring no parameter updates for integration with diverse base models. Experiments on multiple benchmarks show that SpecCal consistently improves top-k reconstruction at both SMILES and scaffold levels across different base models. Further analyses demonstrate that calibrating candidate sets under spectral ambiguity provides a practical way to improve molecular reconstruction from IR spectra. The code is available at: this https URL. 

---
# VeriSkill: A Self-Evolution Framework for Program Verification Skills 

**Authors**: Changguo Jia, Tianqi Zhao, Zhiyou Xiao, Weiming Zhang, Minghui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.27733)  

**Abstract**: Automating program verification with LLM agents requires generating specifications, annotations, auxiliary lemmas, and tool invocations, all of which depend on reusable skills. A natural remedy is skill self-evolution: distilling skills from trajectories and refining them through feedback. However, existing evolution methods struggle with program verification tasks because they cannot reliably identify skill-specific failures or extract actionable signals from opaque verifier feedback. In this paper, we propose VeriSkill, a self-evolution framework built for program verification. It attributes verification failures to skill deficiencies, distills diagnostic signatures into reusable lessons, and iteratively refines candidate skills, admitting only revisions that improve verification performance while preserving program semantics. Experiments show that VeriSkill consistently outperforms all baselines across multiple verification tools, agent frameworks, and LLM backends. 

---
# Baikal: Structured Search for Deep Research over Data Lakes 

**Authors**: Dhruv Agarwal, Rishitha Guttapalle Mohan, Aarti Kumari, Ashi Sinha, Athulya Anil, Kavitha Srinivas, Horst Samulowitz, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2607.27726)  

**Abstract**: Deep research over data lakes requires an LLM agent to investigate evidence across thousands of heterogeneous tables and passages to synthesize a report. Existing methods perform iterative retrieval and generation, letting accumulated context determine what to investigate next, which can overexploit locally promising evidence and fail to cover distinct semantic regions under a fixed budget. To address this, we cast deep research over data lakes as a budgeted search problem and present Baikal - a framework that clusters heterogeneous evidence into semantic regions, then searches over them adaptively to balance exploration and exploitation. Within each selected region, Baikal generates and investigates region-grounded subquestions, using finding quality as rewards to update region-level value estimates and guide search under policies ranging from random and LLM-guided selection to Bayesian $\epsilon$-greedy and UCB. We evaluate Baikal on 15 queries each over HybridQA and TAT-QA data lakes containing 10,993 and 2,757 tables, respectively, together with 227K Wikipedia passages and 13K financial report passages. We assess research quality with a new rubric covering groundedness, relevance, diversity, and utility, and use GPT-5-mini to score Baikal and strong baselines, including DeepSearcher and an OpenCode research agent with retrieval and clustering variants. Across both data lakes, Baikal performs strongly under several region-selection policies; its best configuration improves report scores over the strongest baselines by 28% on HybridQA and 36% on TAT-QA. Our analyses attribute these gains to organizing and exploring semantic evidence regions, which improves groundedness and diversity and yields more useful findings under the same subquestion budget. These results demonstrate the value of structured semantic exploration for systematic research and discovery over heterogeneous data lakes. 

---
# New Synchronous Computation Dynamics for Hopfield Networks 

**Authors**: Francisco Requena-Domínguez, Rafaela Benítez-Rochel, Ezequiel López-Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2607.27720)  

**Abstract**: The dynamics of the original Hopfield network is asynchronous (sequential) (updates the state of only one neuron per time step). In this paper, we propose a new tool and a new dynamics to reduce the processing time by updating one or more neurons simultaneously per instant while ensuring process convergence and aiming for the maximum energy decrease at each step, thus guaranteeing the shortest total processing time. From the point of view of synchronous dynamics, calculating the next network state at which energy decreases the most from the current state while ensuring convergence is itself a combinatorial optimization problem. We develop and use a new tool to solve it. We call this new tool Discrete Differential Filter (DDF) and, based upon it, we develop a new synchronous dynamics which we call SD-DDF (Synchronous Dynamics based upon Discrete Differential Filter). In this paper, we review the original asynchronous dynamics for Hopfield networks and present a new tool and a new synchronous dynamics with its theoretical justification and four computational experiments to assess the speed up in processing time empirically. 

---
# MECA: A Mechanism-Centered Agent for Constructing Well-Specified and Valuable Mathematical Conjectures 

**Authors**: Wentao Long, Yunfei Zhang, Chenyi Li, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27709)  

**Abstract**: Automatically constructing well-specified and valuable mathematical conjectures remains a central challenge in AI-assisted mathematical discovery. Many existing open problems and conjectures are often too broad, underspecified, or difficult to connect to plausible proof or refutation strategies. We view a mathematical mechanism as a structure or reasoning principle that connects the assumptions of a candidate problem to its target conclusion, such as an inequality, invariant, decomposition, or reduction to an intermediate claim. We present MECA (MEchanism-centered Conjecture Agent), a multi-agent framework that constructs conjectures by jointly developing candidate statements and their supporting mechanisms. Explorer agents propose mechanisms, test how they apply, and revise the candidate conjecture accordingly, while critic agents assess their mathematical validity and research value. Their feedback guides changes to the assumptions, scope, and conclusion. Through this process, MECA transforms broad research directions into precise conjectures with substantive mathematical support while retaining a clearly identified unresolved core. We evaluate MECA in two complementary settings. First, we compare it with a generate-and-revise baseline on reconstructing preselected target-paper conclusions from target-conditioned but article-blind source materials. Second, we construct 100 semi-open problems from literature-derived seeds and existing open problems and evaluate them through independent proof and refutation attempts by automated provers. Our results indicate that mechanism-centered refinement produces well-specified and research-worthy conjectures that remain challenging for current automated provers. 

---
# Albilich: Steerable Proof-State Orchestration for LLM-Based Mathematical Research with CAS Integration 

**Authors**: Ting Gong, Michael Ruofan Zeng, Yong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27705)  

**Abstract**: Large language models can contribute useful ideas to mathematical research, yet long-horizon proof attempts remain difficult to coordinate, evaluate, and reproduce. We present Albilich, an open-source agentic harness for autoresearch in mathematics that combines long-horizon reasoning, computer algebra systems (CAS), literature retrieval, and persistent SQLite-based context management.
We evaluate Albilich on the RealMath benchmark (Zhang et al. 2025) and on open problems in group theory from the Kourovka Notebook (Khukhro and Mazurov 2026). It solved 10/10 problems on RealMath with CAS and 9/10 with no CAS. On the Kourovka problems, Albilich produced a counterexample to Problem 21.142 and a proof of a strengthening of Problem20.2. Anablation on Problem 17.91 demonstrates 32.0% token reduction when CAS is enabled. An ablation on Problem 21.142 demonstrates higher verifier-rejection rate and failure to synthesize proof routes in the absence of the advisor agent. These results support Albilich as a human-steerable, CAS-boosted environment for scalable AI-assisted mathematical research. 

---
# SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them 

**Authors**: Yang Zhou, Zixuan Huang, Sunzhu Li, Zhuo Yang, Chen Zhang, Shunian Chen, Caijun Yan, Jianyao Xu, Shunyu Liu, Weijie Fu, Peiliang Li, Xiaozhi Chen, Yuxiang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.27703)  

**Abstract**: Vision-language models (VLMs) are increasingly used in embodied agents to interpret visual inputs, reason about spatial relationships, and make task-level decisions based on that reasoning. However, a fundamental capability mismatch remains: general VLMs can reason about the overall task but often miss the visual details that determine success, while specialist vision models can capture those details but cannot translate them into task-level decisions. In this work, we propose SpatialCLI, a framework that teaches VLMs to reason with spatial tools and progressively internalize the specialist perceptual capabilities they provide. SpatialCLI proceeds in three stages: (1) Call exposes specialist vision models as spatial tools to augment the VLM's perception; (2) Learn uses Cold-Start SFT and agentic RL to improve tool use; and (3) Internalize verbalizes successful tool-use trajectories to internalize specialist perceptual capabilities. We further introduce SpatialCLI-Bench, a 516-example benchmark for compositional perception across localization, segmentation, depth, and pose. On MindCube, SpatialCLI raises Qwen3-VL-8B-Instruct from 29.3% to 84.6% with tools, surpassing GPT-5.6 Sol with tools (72.1%), while retaining 73.8% without tools after internalization. 

---
# Guiding Large Language Models with Genetic Programming-Evolved Heuristic Knowledge for Dynamic Multi-Mode Project Scheduling 

**Authors**: Yuan Tian, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27698)  

**Abstract**: In dynamic multi-mode project scheduling, activities have alternative execution modes and uncertain durations, while precedence relations and limited resources constrain their execution. Heuristic priority rules support fast online decisions, but their design requires substantial domain expertise. Genetic programming (GP) hyper-heuristics can automatically evolve such rules. Large language models (LLMs), meanwhile, provide a flexible interface for interpreting scheduling information and explaining decisions. However, zero-shot LLM decisions may lack domain knowledge, consume many tokens, and vary across repeated queries. GP-evolved rules therefore provide a potential source of scheduling knowledge for guiding LLM decisions. Unlike existing LLM--GP hybrids that use LLMs to support heuristic evolution, we transfer knowledge in the reverse direction, using knowledge extracted from high-quality GP rules to guide an online LLM decision maker. We extract knowledge from high-quality GP rules and inject it through Feature Selection, Feature Hint, Rule Reference, and Rule Follow. These mechanisms are evaluated in terms of scheduling performance, token consumption, decision stability, and the feature focus expressed in generated rationales. GP-derived guidance generally improves the unguided LLM, but its representation matters. Simplifying the decision context or supplying explicit decision logic is more effective than highlighting important features. Feature Selection offers the best token efficiency, whereas Rule Follow achieves strong performance at greater token cost. Guidance also improves decision stability and changes the features expressed in generated rationales. 

---
# Rehearse: Stepping Back from the Confidence Cliff in Self-Improving Autoresearch 

**Authors**: Jiazhen Ji, Shouhong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.27687)  

**Abstract**: Autoresearch improves machine-learning code by proposing changes, running full training jobs, and keeping changes that improve the metric. The efficiency of this loop depends not only on generating ideas, but also on the agent's ability to decide, before spending a training run, whether a proposed modification is likely to work. We study how the reliability of this pre-execution judgment changes over the course of an autoresearch trajectory. In public AutoSOTA logs (Li et al., 2026; Tsinghua FIB Lab, 2026), the fraction of helpful modifications falls from 70% in the first two iterations to 43% by iteration 6+. On 296 same-baseline modification pairs from 39 paper-derived AutoSOTA tasks, each containing one modification that improved the metric and one that did not, with measured outcomes hidden, an LLM judge given candidate rationales but no prior-attempt history reaches 79.5% accuracy on the pairs where strict consensus returns a verdict. On the full 366-pair benchmark, however, this ability weakens substantially late in the loop. As successful changes accumulate, selective accuracy - accuracy conditioned on a strict-consensus verdict - falls from 82.8% to 56.9%, while the judge remains willing to decide. We call this operational pattern the confidence cliff. Rehearse implements the loop change as a lightweight skill for autoresearch loops: propose several ideas, compare them before execution, run the most promising, and judge with a focused memory of similar past attempts and outcomes. This focused outcome memory raises late selective accuracy to 83.5%. Across 4,000 budgeted training runs over three loops, Rehearse improves the endpoint under the same training-run budget on nanochat, image classification, and time-series forecasting. 

---
# Evaluating and Pricing Advertisements in AI-Generated Responses 

**Authors**: John L. Turner-Smith, Zimeng Huang, Yuhan Fu, Yihang Zhang, Tonghan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27686)  

**Abstract**: As search increasingly shifts toward LLM-driven answer engines, advertising is becoming embedded within the generated response itself and should therefore be evaluated for both user utility and commercial value. The key challenge is click-through intent: behavioural logs are unavailable, human annotation resists calibration, and frontier LLM judges conflate intent with linguistic fluency. These gaps compound, as principled pricing presupposes a continuous intent signal, while generating such a signal presupposes supervision that is currently unavailable. We construct the missing supervision through a psychologically grounded agent simulation framework, and distil it into a parameter-efficient evaluator that predicts click-through intent, together with the three companion dimensions of ad quality, as smooth, differentiable estimates. Validated through sign-certain behavioural perturbations, the evaluator surpasses frontier zero-shot judges on relevance sensitivity (79% versus 60-67%), tracks graded content degradation, generalises without error to 103 fictional products, and agrees with human preference in 86% of pairwise judgements across five annotators, with agreement rising in the evaluator's confidence. Upon its estimates we build the pricing layer directly, deriving the unique payment rule under which truthful bidding is optimal, demonstrating it on a best-of-k allocation, and extending the mechanism to non-monotone allocations. The same differentiable signal stands ready as a training objective for ad generation. 

---
# HALO: Heterogeneous Admission through Localized Obligations for Safe Agentic Execution 

**Authors**: Taewoo Park, Kyeonghyun Yoo, Kiseok Kim, Seunghyun Yoo, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.27636)  

**Abstract**: Recent agentic AI systems may return a heterogeneous response containing notices, requests, handoffs, and actions. Conditions can change before external use, so components from the same response need not remain supported together. Rejecting the whole response discards useful components, whereas checking components independently can leave a dependent without its prerequisite. We present Heterogeneous Admission with Localized Obligations (HALO), a runtime protocol that preserves supported components whose declared prerequisites also remain supported, rechecks each exact action before dispatch, and allows blocked actions to be replaced only by fresh candidates. HALO matched all 96 admission expectations and passed all 20 protocol tests. In structured-response replay, it retained 248/248 supported components, including 128/128 unaffected by unrelated changes, while a whole-response policy retained 0/248. Across ten cold-start PX4/Gazebo sessions, HALO blocked every tested stale route, observed no matching stale setpoint, and completed all fresh recoveries. 

---
# HealthCAT: An Interpretable Encoder-only Transformer Framework for Health Indicator Prediction and Temporal Interpretation of Wearable Sensor Data 

**Authors**: Xiaotong Yu, Joshua Y. Kim, HaeJin Lee, Kalina Yacef  

**Link**: [PDF](https://arxiv.org/pdf/2607.27635)  

**Abstract**: Wearable sensors continuously capture fine-grained multivariate time-series data, providing opportunities to model behavioural patterns associated with health outcomes. However, existing deep learning methods prioritise predictive accuracy over interpretability, limiting their application in health research. In this study, we present HealthCAT, a flexible framework that integrates an Encoder-only Transformer with an Attentive Class Activation Token (AttentiveCAT) to generate class-specific, time-step-level interpretations. These interpretations can be mapped back onto behavioural cycles that are relevant to the domain (e.g., time-of-day), supporting individual-level analysis of wearable sensor data. We evaluated HealthCAT using two real-world wearable sensor datasets (306 participants in total). HealthCAT outperformed deep learning baselines by up to 17\% in F1-score and 12\% in accuracy on both datasets ($p<0.05$). In masking experiments, the time steps identified by HealthCAT carried significantly more predictive value than random selection across all masking conditions ($p<0.05$), indicating that the identified time steps are predictively informative. By coupling predictive performance with validated time-step-level interpretability, HealthCAT moves wearable sensor analysis beyond aggregated metrics towards temporal patterns that support health monitoring, behavioural pattern analysis, and intervention design in health research. The significance of this work is that it enables accurate prediction of health indicators from wearable sensor data while providing insights into when and how physical activity patterns occur, rather than relying solely on aggregated summary measures. 

---
# ReDiPPO: Reference-Guided Value Calibration and Discrepancy-Aware Token Reweighting for Mathematical Reasoning 

**Authors**: Zhenrong Zhang, Fei Wu, Jun Du, Jianshu Zhang, Si Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.27631)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for enhancing the mathematical reasoning capabilities of large language models. Among existing policy optimization methods, Proximal Policy Optimization (PPO) remains particularly appealing because its learned critic can, in principle, provide token-level credit assignment. However, in mathematical reasoning tasks characterized by long reasoning horizons and sparse outcome rewards, reliable token-level credit assignment remains challenging. The standard critic often fails to accurately evaluate intermediate reasoning states, resulting in noisy advantage estimates and suboptimal policy updates. In this paper, we propose ReDiPPO, a Reference-guided and Discrepancy-aware PPO framework for mathematical reasoning. ReDiPPO introduces a reference-guided critic that uses reference answers as training-time privileged signals to provide more accurate value estimation. Meanwhile, it retains a standard critic and quantifies the token-level reference-standard discrepancy between the standard value estimate and the reference-guided value estimate. This discrepancy serves as an indicator of difficult reasoning states and is used to reweight the corresponding token-level advantages during PPO optimization. Extensive experiments on diverse mathematical reasoning benchmarks demonstrate that ReDiPPO improves value-estimation accuracy and consistently outperforms strong policy optimization baselines, including PPO, DAPO, and GSPO, in final reasoning performance. Our code is available on this https URL. 

---
# SCOPE: Synthetic Conditional Objectives for Policy Evolution in Black-Box Combinatorial Optimization 

**Authors**: Nguyen Viet Tuan Kiet, Nguyen Huu Duc, Le Cong Bang, Tran Cong Dao, Huynh Thi Thanh Binh  

**Link**: [PDF](https://arxiv.org/pdf/2607.27630)  

**Abstract**: Black-box combinatorial optimization requires systematically identifying high-quality solutions under a limited evaluation budget, yet the unknown objective function provides little guidance for deciding where the search should explore next. We introduce SCOPE, a general framework for Synthetic Conditional Objectives for Policy Evolution in Black-Box Combinatorial Optimization. Rather than directly optimizing the inaccessible objective, SCOPE learns a set of synthetic objectives conditioned on the accumulated search history, where each objective is designed to expose a distinct and potentially useful preference over candidate solutions. These objectives are then used to evolve search policies that generate diverse candidates, whose true quality is subsequently assessed through black-box evaluations. The outer loop adaptively updates and selects synthetic objectives according to how effectively their induced policies discover promising regions. In contrast, the inner loop returns a portfolio of top-performing policies to reduce the risk of relying on a single surrogate preference. This formulation reframes objective design as a mechanism for guiding policy exploration, enabling the search process to exploit observed evidence while maintaining structured diversity across discrete solution spaces. Extensive experiments across multiple benchmark problems demonstrate that SCOPE consistently improves black-box search performance under limited evaluation budgets and generalizes well across diverse combinatorial structures. 

---
# Hidden APIs in Language Models: Discovering Reusable Causal Interfaces from Forked Futures 

**Authors**: SiYuan Ma, Yiqin Luo, Zhangji, Canran Xiao, Albert Gao, Wei-Hsing Huang, Wei Wang, Qiwei Wu, Xinran Li, Jinfeng Wei, Qixin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27617)  

**Abstract**: Identical language-model answers can arise from hidden states that support different future computations, so current-answer probes do not establish a reusable internal interface. We introduce forked futures: future operations are sampled only after a prefix state has formed, and states are compared through the response distributions induced by those operations. This yields an empirical causal quotient over hidden states without requiring researcher-specified latent labels. Shared, Local, Mixture, and Distributed interfaces then compete under prequential causal description length subject to future-signature fidelity and matched capacity constraints. In the two detailed model evaluations, Shared has the lowest held-out description length, with gains of 0.216 nats on Qwen2.5-1.5B and 0.294 nats on Llama-3-8B, while maintaining tightly clustered mean future-signature distortion; a five-backbone sweep preserves the positive direction of Sharedness Gain. The figure-aligned transplantation analysis gives Shared the strongest joint target-correctness, locality, copy-preservation, and composite profile, and API-aligned paths mediate 0.749 of the target effect versus 0.150 for matched null paths. In the blind four-class model-organism test, 14/16 architectures are recovered, with one observed non-Shared to Shared error among 12 non-Shared organisms. These results support an economical reusable causal interface within the tested operation banks, while keeping the claim explicitly conditional on the candidate architectures, interventions, and held-out futures. 

---
# CORE: In-Context Reconstruction for Unified Tabular Anomaly Detection 

**Authors**: Yunfeng Zhao, Qingfeng Chen, Yue Tan, Shiyuan Li, Yili Wang, Yixin Liu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27615)  

**Abstract**: Tabular anomaly detection (TAD), which focuses on identifying abnormal samples that deviate from the majority in tabular data, has received growing attention. Recently, there has been an emerging trend towards unified TAD, which seeks to detect anomalies across different datasets using a single generalizable model. In unified TAD, aligning heterogeneous data remains challenging. While existing methods often rely on distance-based unified feature construction, they may obscure the semantics of the original features. Moreover, existing approaches typically formulate anomaly detection as a binary classification task, which may overlook diverse anomaly patterns from various datasets and be misled by unrepresentative synthetic anomalies. To address these challenges, we propose an in-COntext REconstruction approach for unified TAD (CORE for short). It introduces a decorrelated feature alignment module to directly align heterogeneous features into a unified representation space, which retains their semantic information. Meanwhile, CORE formulates unified TAD as an in-context reconstruction problem, eliminating the need for labeled or synthesized anomalies. Specifically, the in-context reconstruction module reconstructs each sample by leveraging contextual normal samples to capture dataset-specific distributions, such that reconstruction errors reflect its deviation from normality, facilitating unified TAD on arbitrary unseen datasets. 

---
# World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models 

**Authors**: Xiangcheng Zhang, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.27599)  

**Abstract**: Building generalizable agents for diverse applications remains a fundamental challenge. While imitation learning-based policies succeed in specific training environments, they often fail to generalize to novel scenes and tasks. In this work, we propose World Action Planner, a robot planning system that leverages the reasoning capabilities of Vision-Language Models (VLMs) and the physical grounding of a multi-task pose-image conditioned world model. Our system enables an agent to propose initial action plans and iteratively refine them via optimization and search, reasoning over imagined world model rollouts. We demonstrate that our approach achieves superior performance across compositional tasks, new layouts, and zero-shot generalization scenarios, significantly outperforming state-of-the-art end-to-end policy models such as VLAs and WAMs. Project website at this http URL 

---
# Wiring diagram extraction and gluing: a case study in classifying figure skating jumps using 3D dataset 

**Authors**: Jason Lo, Mohammadnima Jafari  

**Link**: [PDF](https://arxiv.org/pdf/2607.27598)  

**Abstract**: Hasse clustering is an algorithm that extracts common patterns in sequential data and represents them in graphical forms. As the number of expected clusters grows, however, the algorithm can become infeasible to run due to combinatorial complexity. In this article, we describe a theory of gluing wiring diagrams, allowing iterative applications of Hasse clustering to achieve the same result as a single application. We test our theory in the context of classifying videos of figure skating jumps. 

---
# From Minds to Models: The Intersection of Psychology and LLM Behaviours 

**Authors**: Oliver Guidetti, Reza Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27579)  

**Abstract**: Large language models (LLMs) are often compared with the human mind because their decision-making is complex, non-linear and difficult to interpret. Psychological methods developed to investigate unobservable mental processes may therefore help examine LLM behaviour, particularly in government and healthcare. Building on prompt-based adaptations of the Implicit Association Test, this study tested whether ChatGPT produced sentiment differences across racial conditions in open-ended text. Fourteen base questions were crossed with eight racial categories and a race-agnostic control, producing 126 prompts. Each was submitted once to GPT-3.5T, GPT-4 and GPT-4T, yielding 378 responses. Sentiment scores were derived from categorical labels and source scores: positive labels retained the source score, negative labels were assigned its negative, and neutral responses were coded zero. A two-way ANOVA found a small main effect of racial condition, F(8, 351) = 2.04, p = .042, partial-eta squared = .044, but no effect of model, F(2, 351) = 0.07, p = .933, and no interaction, F(16, 351) = 0.23, p = .999. However, the effect was not retained in a rank-transformed sensitivity analysis, F(8, 351) = 1.53, p = .145, and Tukey-corrected comparisons found no significant pairwise differences. An uncorrected European-Indigenous Australian comparison was significant, but was selected post hoc and is reported only as hypothesis-generating. Evidence for sentiment differences was therefore weak and analysis-dependent. Sentiment scoring also cannot distinguish evaluative bias from the valence of historical content elicited by a prompt. We outline design changes needed to address these limitations and argue for interdisciplinary development of behavioural measures of model bias.
Keywords: Implicit Bias, Psychological Research Methods, Artificial Intelligence, ChatGPT, Large Language Models, Sentiment Analysis 

---
# What makes prompts a graph: necessary and sufficient conditions for prompt graph engineering 

**Authors**: Sandeco Macedo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27578)  

**Abstract**: Prompts stopped being isolated strings some time ago. In real systems, one model call feeds another, retrieval interleaves with generation, routers branch, and aggregators merge parallel results. Practice converged on a single structure to hold this together: the graph. Frameworks such as LangGraph, DSPy, and Prompt Flow expose it openly, and research systems already optimize it automatically. The vocabulary, however, lags behind. Graph names, variously, a reasoning topology inside one sampling strategy, a multi-agent conversation, or an orchestration artifact, while prompt engineering still evokes writing one good string. What is missing is a reference definition treating prompts as nodes of an explicit, executable, improvable graph. We build that definition through conceptual analysis over sources with persistent identifiers, complemented by primary grey literature. We reconstruct the genealogy of the idea, from dataflow graphs and build systems, through prompt chaining and the thought topologies (chain, tree, graph), to graphs compiled and optimized as artifacts. We then propose a constitutive definition of prompt graph engineering, state its four conditions (explicit structure, separation between structure and prompt content, executable semantics, and the graph as a first-class engineering artifact), and operationalize them as an inclusion and exclusion test. We draw the boundary against six neighboring concepts and apply the test to six real systems (LangGraph, DSPy, Prompt Flow, AutoGen, CrewAI, and Claude Code subagents); it includes and excludes consistently. We close with a research agenda organized along four design tension axes. The contribution is an operational definition and a shared vocabulary for a practice that industry already exercises daily without naming precisely. 

---
# DeepResearch Agent System 

**Authors**: Yong Huang, Yulu Huang, team Collaboration  

**Link**: [PDF](https://arxiv.org/pdf/2607.27562)  

**Abstract**: The DeepResearch Agent System is a large language model system engineered for deep information retrieval, multi-step reasoning, and autonomous research tasks. Built upon a sparse activation architecture with 30 billion total parameters of which only 3 billion are activated per token, the system achieves state-of-the-art performance on multiple agent search benchmarks while delivering 3.2 times faster inference compared to dense counterparts of equivalent scale. The system supports a 128K-token context window with hierarchical attention mechanisms that yield 18.7% accuracy and 23.4% recall improvements over standard long-context approaches. A dual-mode reasoning engine provides both a ReAct paradigm for basic multi-step problem solving and an IterResearch mode for high-performance iterative research with up to 20 reasoning steps, collectively delivering a 31.2% accuracy improvement over single-pass baselines. Multi-tool coordination integrates retrieval, computation, web search, and file parsing modules to achieve 92.1% tool-use accuracy. A reinforcement learning optimization framework based on the GRPO algorithm provides token-level policy gradients that improve training stability by 35% and accelerate convergence by 42%. An automated data synthesis pipeline with seed-based expansion achieves a 92.5% usability rate. Benchmark results include 87.3% on Humanity's Last Exam, 85.3% on BrowserComp Chinese, and 91.2% on WebWalkerQA. The system is fully open-sourced, including data synthesis, training, and inference code, and supports applications in academic research, business analysis, R&D support, and education. 

---
# Evaluating Agentic Bioinformatics through Function, Evidence, and Validation 

**Authors**: Phuc Pham, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2607.27556)  

**Abstract**: Large language model agents increasingly plan, execute, and interpret biological analyses, yet fluent responses, successful tool calls, and benchmark performance alone do not establish scientific credibility. Existing reviews primarily organize biological agents by application, architecture, and agentic capability, but do not jointly operationalize the accountability of agent-generated workflows. We address this gap by treating the inspectable workflow trajectory, rather than architecture or final output alone, as the primary unit of analysis. We introduce the Function--Evidence--Validation (FEV) framework, which separates demonstrated workflow operations, traceable support for actions and claims, and use-case-specific validation. Using FEV, we map 109 agentic or agent-adjacent systems and 28 benchmark or evaluation resources, representing 128 unique publications across genomics, single-cell and spatial omics, protein science, drug discovery, computational pathology, and general bioinformatics automation. Across domains, planning and tool-mediated execution have advanced more rapidly than replayability, provenance, robust scientific assessment, external validation, and prospective empirical testing. We therefore argue that agentic bioinformatics should be assessed through workflow correctness rather than final-answer correctness alone. FEV provides a practical basis for comparing systems and designing transparent, auditable, and scientifically accountable bioinformatics workflows. 

---
# Using Large Language Models for Idea Generation in Innovation 

**Authors**: Lennart Meincke, Karan Girotra, Gideon Nave, Christian Terwiesch, Karl T. Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2607.27553)  

**Abstract**: This research evaluates the efficacy of large language models (LLMs) in generating new product ideas. To do so, we compare three pools of ideas for new products targeted toward college students and priced at 50 dollars or less. The first pool of ideas was created by university students in a product design course before the availability of LLMs. The second and third pools of ideas were generated by GPT-4 from OpenAI using zero-shot and few-shot prompting, respectively. We evaluated idea quality using standard market research techniques to predict average purchase intent probability. We used text mining to assess idea similarity and human raters to evaluate idea novelty. We find that AI-generated ideas outperform human-generated ideas in terms of average purchase intent, with few-shot prompting yielding slightly higher intent than zero-shot prompting. However, AI-generated ideas are perceived as less novel and exhibit higher pairwise similarity, particularly with few-shot prompting, indicating a less diverse solution landscape. When focusing on the quality of the best ideas rather than the average ideas, we find that AI-generated ideas are seven times more likely to rank among the top 10 percent of ideas, demonstrating a significant advantage over human-generated ideas. We propose that this seven-to-one advantage is a conservative estimate because it does not account for the greater productivity of AI. Our findings suggest that despite some drawbacks, AI creativity presents a substantial benefit in generating high-quality ideas for new product development. 

---
# AI Literacy: An Exercise in Power-Knowledge 

**Authors**: Brady D. Lund, Zoë Abbie Teel  

**Link**: [PDF](https://arxiv.org/pdf/2607.27547)  

**Abstract**: As generative artificial intelligence becomes one of the most significant systems of knowledge production in our society today, questions relating to who can access and shape that production grow increasingly important in our discourse. This paper argues that the existing frameworks for AI literacy, which are dominated by technical competency and responsible-use principles, are insufficient because they enforce a "consumer" orientation toward AI rather than fostering genuine epistemic agency. Based upon Foucault's concept of power-knowledge, Freire's pedagogy of critical consciousness, and scholarship of digital literacy, this paper proposes a reconceptualization of AI literacy as a critical practice that equips individuals not just to use AI systems, but to critically evaluate them, resist their structuring assumptions, and participate in their governance. The paper further argues that unequal access to AI tools in society recapitulates longstanding epistemic injustices, and that a literacy framework oriented toward empowerment must account for these structural inequities. A three-part framework of AI literacy based on the notions of contextual use, critical interrogation, and participatory governance frames this literacy as a cultivation of epistemic "agents" rather than the training of competent consumers of AI-generated information. 

---
# Automated Transcript Analysis for Detecting Flaws in Agentic Benchmarks 

**Authors**: Jeff Mohl, Nelson Gardner-Challis, Magda Dubois, Harry Coppock, Benjamin Allan-Rahill, Kaelan Yim, Damian Sójka, James Mann, Justin Olive  

**Link**: [PDF](https://arxiv.org/pdf/2607.27518)  

**Abstract**: Capabilities of frontier models are often assessed using agentic benchmarks. To trust these results, benchmarks must accurately measure what they claim to and be free from invalidating flaws. Previous manual audits of benchmarks such as SWE-Bench-Verified have uncovered several validity issues in transcripts. However, manual review is difficult to scale, and it is unclear whether automated methods can reliably surface flaws that compromise benchmark validity. In this paper, we developed AI scanners to detect four types of validity issues: ground truth access, tool failure, guessing vulnerability, and answer format ambiguity. We produced grading rubrics for each to instruct human labeling, and evaluated the scanners against human labels on a held-out test set of Inspect Evals benchmarks. Our scanners identified several verified quality issues in five widely used benchmarks, including cases unlikely to be caught by random manual inspection. Not all cases were identified, and scanner performance varied substantially across benchmarks, criteria and models. We highlight several open challenges to be addressed to improve scanners for stronger quality assurance claims, including broader standardization gaps in the evaluation field that degrade scanner performance. Together, these results serve as a proof of concept for using automated transcript analysis to audit benchmark quality more broadly. 

---
# A dataset of rated conceptual arguments 

**Authors**: Emery Cooper, Caspar Oesterheld, Linh Chi Nguyen, Alexander Kastner, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2607.27499)  

**Abstract**: Large language models have improved rapidly on tasks with verifiable answers, such as mathematics and programming. Much less is known about their ability to reason about what we call conceptual questions: questions for which no ground truth is realistically accessible and no widely accepted resolution methodology exists, but on which progress can still be made by debating arguments. Most philosophical questions are of this kind, as are central components of questions in AI safety, decision theory, and social choice. Our approach is based on the view that while bottom-line conclusions on such questions are hard to evaluate, individual contextualized arguments can be evaluated far more reliably. We therefore introduce a dataset of 951 argumentative critiques of 442 position texts, spanning topics from AI safety and decision theory to ethics and politics, with 1,458 ratings by six expert raters along dimensions including centrality, strength, correctness, and clarity. We propose two scoring functions and benchmark a range of models. Performance tracks general capability rankings. 

---
# MedLLM: An Open Medical Language Model at the Sub-Billion Scale 

**Authors**: Maxx Richard Rahman, Asim Ahmed, Mihan Mohagheghzadeh, Wolfgang Maass  

**Link**: [PDF](https://arxiv.org/pdf/2607.27490)  

**Abstract**: Open medical language models have converged on a single scale: every widely used system runs at 7B parameters or more, leaving the sub-billion regime uncharacterized. We present MedLLM, an open 0.1B-parameter medical language model trained through a fully open three-phase pipeline: general pretraining with curriculum sequence-length scheduling, domain fine-tuning on MedFineWeb, a reference-guided medical corpus we release that is selected from general web data by embedding similarity to medical question-answering (QA) data, and preference-aligned fine-tuning combining SFT with direct preference optimization (DPO). Across medical benchmarks, MedLLM shows a pattern visible only at sub-billion scale: medical competence does not degrade uniformly under compression but splits by task type. On context-grounded QA it comes within $2.9$pp of a medically adapted 7B model and surpasses the instruction-tuned and general-purpose 7B baselines; on knowledge-recall QA it stays near the task floor on clinical-vignette MedQA yet significantly exceeds every 7B and sub-7B baseline on MedMCQA, indicating that where recall fails the constraint is model capacity rather than adaptation. This dissociation is masked at 7B, where both capabilities are present, and surfaces only when capacity is scarce. 

---
# INCLAIR: Inception-Based Longitudinal Clinical Anomaly Detection with Informed Reasoning 

**Authors**: Maxx Richard Rahman, Wolfgang Maass  

**Link**: [PDF](https://arxiv.org/pdf/2607.27487)  

**Abstract**: Detecting anomalies in longitudinal clinical profiles is clinically important but difficult: abnormal evidence is often sparse, patient histories have unequal length, and expert explanations are costly. We propose INCLAIR, a framework that scores each observation against multiple historical contexts, aggregates evidence at the profile level, and generates grounded natural-language explanations under limited expert supervision. Under stated within-profile exchangeability assumptions, the complete mean subsequence score takes an order-$l$ U-statistic form, yielding a variance decomposition and an incomplete-subset approximation that controls combinatorial inference cost independently of profile length. The same analysis shows that mean aggregation attenuates localized anomalies by a factor set by the anomaly support and profile length, motivating validation-selected top-$k$ pooling. Across three clinical datasets, INCLAIR consistently outperforms state-of-the-art baselines. We further validate practical relevance through a case study on longitudinal steroid profiles, comparing INCLAIR's predictions and explanations against domain-expert assessments supported by DNA analysis. The results show that INCLAIR enables clinically actionable anomaly detection under limited expert supervision. 

---
# Skill Use or Skill Theater? Evaluating the Reasoning Backroom in Skill-Augmented Language Agents 

**Authors**: Jinwei Hu, Yi Qi, Xinmiao Huang, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27484)  

**Abstract**: Reusable skills are becoming a standard interface for extending language agents with task procedures. Yet evaluators usually infer skill use from visible reasoning or the agent's own attribution. These signals show what the agent appears to use, not whether the skill changed its decision. We ask whether skill-augmented agents exhibit a \textbf{Reasoning Backroom}, a systematic gap between stated skill use and intervention-measured influence. We introduce BACKTRACE, an evaluation framework that pairs each skill-conditioned answer with a matched no-skill counterfactual, intervenes on skill meaning, wording, identity, content, and assignment, and elicits attribution only after the answer is committed. We instantiate the framework as BACKROOMBench, a verified testbed spanning controlled logic and competition mathematics, multiple skill conditions, single-agent and multi-agent settings, and diverse model families. Our evaluation reveals a pervasive provenance failure. Across models and domains, stated skill use often remains stable while causal reliance and signed utility vary, producing both silent uptake and performative use. Behavioral effects follow procedural content more reliably than displayed skill identity, whereas stated attributions respond strongly to artifact availability. Observational detectors based on direct skill-use claims, text mentions, trace similarity, and an LLM judge do not identify which decisions actually depend on the skill. In multi-agent systems, skill influence can survive communication even after its source is lost, while no-skill teams still name skills and sources that were never supplied. These findings establish the Reasoning Backroom as a general AI provenance problem whose audit requires intervention. 

---
# VAmoS Bench: Voice Agent Simulation Bench 

**Authors**: Joshua Meyer, Sahar Shayegan, Ritiz Tambi, Ali Khan, Sun Kim, Victor Shih, Mehdi Jamei, Andi Partovi  

**Link**: [PDF](https://arxiv.org/pdf/2607.27453)  

**Abstract**: Production voice agents span cascaded, speech-to-speech, and hybrid architectures. Voice-agent benchmarks typically measure component quality and conversational properties such as word error rate, latency, naturalness, and turn-taking. Fewer measure whether the agent handled a phone call correctly on its own. Contact centers refer to this as ``containment'': the share of phone calls the automated system resolves without handing off to a human. On some phone calls the right outcome is refusal or a redirect. To address this gap, we introduce VAmoS Bench, the Voice Agent Simulation Bench. It measures complete voice-agent systems end to end in a stateful customer-support task. The agent is Riley, a credit-card support representative for a fictional bank who can freeze, cancel, replace, or activate a card. Each of 100 scenarios supplies a simulated caller with a private goal and a seeded PostgreSQL backend. The platform uses each scenario to populate and activate an isolated simulation in which the caller reaches Riley over audio; roughly one-third apply adversarial pressure. The agent can use five tools that execute real SQL against the backend. Each scenario also defines binary assertions. A grader evaluates them against the complete trace of what the caller and agent said and what the agent did, including tool invocations, arguments, and returned rows. This catches an agent that claims to have changed a card without updating the database, as well as one that makes the right database change while disclosing protected information. This first benchmark version focuses on financial services. Its evaluation protocol supports an evolving leaderboard: additional voice agents can be evaluated on the same version, while later versions can expand the tasks and scenarios. 

---
# Leveraging Trajectory Graphs for Pre-Execution Error Diagnosis in Agentic LLM Systems 

**Authors**: Xu Zheng, Zhuomin Chen, Chaohao Lin, Hua Wei, Haifeng Chen, Wei Cheng, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27443)  

**Abstract**: Large Language Model~(LLM)-based agents have demonstrated exceptional performance across a wide range of complex interactive tasks. However, they often struggle with long-horizon interactive tasks common in domains, such as embodied AI. The complexity and vast action spaces in these settings lead to compounding errors, where a single suboptimal action can derail an entire trajectory, causing the agent to exhaust its limited step budget on inefficient or unrecoverable paths. To overcome this without costly fine-tuning, we draw inspiration from software debugging, where execution logs are analyzed to preemptively catch errors. We propose \textit{Trajectory Graph Copilot}, a novel framework that acts as a ``copilot'' for LLM agents by diagnosing potential action errors before they are executed. At its core,\textit{Graph Debugger} models historical trajectories as a probabilistic graph and uses a Graph Neural Network to identify sequential action patterns that frequently lead to failure. Functioning as a proactive diagnostic sandbox, our method provides early warnings on potentially flawed actions, prompting the agent to self-correct. This pre-action error diagnosis prevents costly mistakes, significantly enhancing the agent's ability to complete long-horizon tasks successfully. The extensive experiments on four benchmarks with three LLM agents demonstrate a $14.69\%$ pass ratio improvement on average. 

---
# Dimensionality and Measurement Precision in HLE's Multiple-Choice Subset 

**Authors**: Mayank Sharma, Savira Nadela, Tyler Matteson  

**Link**: [PDF](https://arxiv.org/pdf/2607.27420)  

**Abstract**: Humanity's Last Exam (HLE) is widely used to evaluate frontier language models. HLE organizes its questions into eight subject-domain categories, whose subscores are often interpreted as evidence of distinct capabilities. However, no study has assessed whether these labels correspond to empirically separable latent constructs, nor whether the benchmark effectively differentiates between models of similar ability. We evaluate 29 LLMs on the text-only multiple-choice subset of HLE ($J = 428$ items) and apply psychometric methods to assess both the dimensionality of the benchmark and the distribution of its measurement precision. Fitting a two-parameter logistic IRT model, we find convergent evidence that HLE measures a single general reasoning factor: McDonald's $\omega_h = 0.998$, domain labels explain only 3.5\% of item response variance, within- and between-domain residual correlations are nearly identical (Cohen's $d = 0.016$), and domain-specific ability estimates are near-redundant with the total score ($r \geq 0.81$). A separate analysis of the test information function reveals that measurement precision concentrates at moderate ability levels and drops sharply above $\theta = 0$, where frontier models sit. These findings suggest that HLE's domain subscores do not warrant distinct capability interpretations and that the benchmark's ability to discriminate among the strongest models is limited. 

---
# Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs 

**Authors**: Xu Zheng, Chaohao Lin, Zhuomin Chen, Weijieying Ren, Haifeng Chen, Wei Cheng, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27415)  

**Abstract**: Recent advancements in inference-time scaling have significantly unlocked the complex reasoning capabilities of Large Language Models~(LLMs). However, for agents, these approaches suffer from a critical inefficiency, operating in a stateless manner and engaging in redundant search processes. Existing memory mechanisms largely rely on the reasoning capabilities of LLMs, leading to prohibitive computational costs. In this paper, we propose a novel framework, \textit{GAMER}~(Graph-based Action-centric Memory with Episodic Reasoning), that bridges the gap between inference scaling and episodic memory. Our approach models historical reasoning as a dynamic \textit{Action-Centric Graph}. By decoupling the memory mechanism from LLMs, our method can save token/money usage by providing less memory context than memory mechanism baselines. To extract knowledge from the graph effectively, we use a dual-stream Temporal Difference learning mechanism to estimate the positive~(suggestion) and negative~(avoidance) value of action nodes based on past successes and failures. During the inference phase, this learned value function optimizes decision-making bi-directionally, so that positive values provide action suggestions, while negative values indicate high-risk actions. By performing efficient searches on the graph, our method significantly improves the efficiency of inference scaling. Experiments on multiple benchmarks demonstrate that \textit{GAMER} achieves superior performance by \textbf{20.81\%/6.17\%} for success/progress rate compared to vanilla baselines. 

---
# Beyond the Bidirectional Promise: Re-evaluating the Robustness of Diffusion Language Models 

**Authors**: Saurabh Yadav, Badri Narayana Patro, Vijay Srinivas Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2607.27386)  

**Abstract**: Diffusion Language Models (DLMs) offer a compelling alternative to autoregressive (AR) generation by enabling bidirectional context and iterative refinement. However, their reliability under natural input noise and adversarial attacks remains under-explored. To address this, we systematically evaluate DLM robustness and calibration against AR baselines, using two parameter-matched pairs (LLaDA-8B vs. LLaMA-3-8B and Dream-7B vs. Qwen2.5-7B) across 32 natural perturbation conditions, adversarial gradient probes, and mechanistic hidden-state analyses. This paired design effectively isolates architecture-intrinsic properties from weight-dependent behaviors. We find a nuanced robustness profile: while highly stochastic DLM loss landscapes naturally resist gradient-based adversarial suffixes, they provide no guaranteed defense against natural noise, proving that everyday robustness is weight-dependent rather than inherently architectural. Furthermore, DLMs exhibit systematic overconfidence, presenting a practical deployment hazard. Most crucially, mechanistic probing reveals that all models perfectly encode input corruption, isolating behavioral fragility entirely to a decoder routing failure. Consistent with this diagnosis, we show that surface-level prompt patching fails to improve over noisy baselines. Ultimately, DLM robustness cannot be patched on; it must be fundamentally integrated into the iterative decoding loop. 

---
# SkillMentor: LLM Agent Self-Evolution via Learning Blind-Spot Diagnosis 

**Authors**: Xiaoyi Bao, Yuanzhen Xie, Yunzhi Tan, Jinghang Gu, Zhongqing Wang, Chu-Ren Huang, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27360)  

**Abstract**: Agent self-evolution has primarily focused on learning how to act, while overlooking an equally important capability: learning to discover what an agent does not know. Existing approaches typically assume that failure discovery is given, focusing on how to repair failures once they are identified. We ask whether blind-spot diagnosis itself can be learned. We thus study diagnosis as an agent capability separate from execution, and exclude two alternative sources of progress: executor adaptation and human supervision. Under these constraints, performance cannot improve through executor updates or annotated examples, forcing all improvements to originate from the learned diagnostic capability. We propose SkillMentor, which trains a Mentor policy via reinforcement learning to generate diagnostic tasks, identify recurrent failure modes, and curate them into reusable corrective skills. Across AppWorld and BFCLv3, SkillMentor improves executor performance by an average of 44.2%. These results suggest that blind-spot diagnosis is a learnable capability, enabling self-evolution without updating executor weights or relying on human-curated data. 

---
# PAUSE: A User-Centric Benchmark for Personal AI Assistants in Unified Service Environments 

**Authors**: Haoyu Chen, Xirui Shi, Yuyao Wang, Jerry Chen, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27354)  

**Abstract**: Personal AI assistants are increasingly deployed as task-oriented, tool-augmented agents that operate within unified service environments to support everyday user activities. In realistic settings, such assistants must reason over persistent user state, respect user-specific configurations and permissions, and sustain long-horizon, constraint-aware interactions across multiple services. Existing benchmarks, however, often fragment service contexts or abstract away user state, limiting their ability to evaluate user-centric personal assistant behavior in realistic service settings. We introduce PAUSE, a user-centric benchmark for evaluating personal AI assistants in stateful, service-integrated environments. PAUSE captures core challenges of real-world assistant deployment by requiring agents to coordinate actions across heterogeneous user-owned resources while maintaining consistency with environment state, authorization constraints over multi-turn interactions. The benchmark incorporates explicit user-agent interaction via realistic user simulation, enabling evaluation beyond static tool execution. To support principled and reproducible evaluation, PAUSE adopts a multi-regime evaluation framework aligned with task characteristics. Open-ended service management tasks are assessed using semantic and trajectory-level behavioral metrics, while constraint-intensive tasks admit deterministic, state-based verification. Benchmark results show that even state-of-the-art proprietary models fail to reach 70% task completion on scenarios requiring stateful reasoning and configuration awareness, revealing consistent and interpretable failure patterns. Finally, we present a user-centric synthesis pipeline that enables scalable generation of coherent service environments, user configurations, and reliably annotated tasks, supporting benchmark extensibility and future research. 

---
# PIE-APT: A Unified Framework for Temporal Planning and Contradiction Hunting via Incremental Direct-Derivation Abduction 

**Authors**: Amir Hossein Sharafi, Alireza Shahbazi  

**Link**: [PDF](https://arxiv.org/pdf/2607.27287)  

**Abstract**: Reasoning and planning over Dynamic Knowledge Graphs (DKGs) present significant challenges, especially in open-world environments with incomplete information. Existing action formalisms often face decidability issues and the Ramification Problem, while managing incomplete knowledge via structural abduction requires expansive combinatorial search. This paper introduces a unified framework with two integrated modules---\textbf{PIE-Abducer} (incremental direct-derivation abduction) and \textbf{PIE-APT} (Abductive Planning for Temporal KGs)---operating natively on the highly expressive Description Logic. We model state transitions along a linear timeline as non-monotonic updates to deductively closed DL theories. Treating the incremental reasoner as a black-box and representing actions natively in OWL without external modal operators preserves logical decidability. To address incomplete knowledge, \textbf{PIE-Abducer} circumvents traditional Minimal Hitting Set (MHS) enumeration. Instead of combinatorial syntactic search, it injects the logical negation of a target goal into a consistent branch and extracts missing premises via direct refutation consequences. \textbf{PIE-APT} then employs a recursive \textit{Generate-and-Test} architecture, interleaving backward-chaining A* search with \textbf{PIE-Abducer} up to a bounded causal depth, followed by strict validation via forward-chaining Temporal Projection. We evaluate four OWL benchmarks stressing semantic abilities absent in classical planning: parameterized goals with witness search, mid-search DL entailment, open-world assumption injection, and adversarial contradiction hunting. Results demonstrate qualitative superiority over classical planners and prove our direct-derivation approach quantitatively outperforms an MHS-faithful baseline during abductive enrichment. 

---
# Divergence Decoding: Training-Free Capability Fusion 

**Authors**: Yimi Wang, Hao Li, Shuo Yang, He Cao, Dechen Zhang, Ziang Wu, Zhiyuan Yan, Fanyang Mo, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27248)  

**Abstract**: While large language models excel in reasoning, these generalists often lack knowledge for specialized scientific domains. Conversely, domain models~(specialists), while knowledgeable, suffer from specialization side-effects including diminished logic and reduced this http URL address this dilemma, we introduce Divergence Decoding, a training-free framework for capability fusion. It reconstructs the "draft-and-verify" skeleton of speculative decoding into an adaptive routing mechanism. The core is using Jensen-Shannon divergence to monitor the distributional disagreement between the two models at each token. When the specialist exhibits significant divergence, our method identifies it as a potential reasoning risk and instantaneously routes control to the generalist. This allows the dynamic injection of general reasoning while preserving domain expertise, achieving inference-time policy composition of the generalist and the this http URL evaluate Divergence Decoding across diverse model families (Qwen and Llama series) on challenging scientific benchmarks (GPQA, ChemBench, and ChemCoTBench). Experimental results demonstrate that Divergence Decoding outperforms both the domain-specialized and general-purpose models, effectively surpassing the performance of most single-model baseline. This suggests that Divergence Decoding provides a general, training-free paradigm for fusing diverse LLM capabilities through adaptive inference-time collaboration. 

---
# Asymmetric Collapse in Model Merging: When Refusal Over- writes Recognition 

**Authors**: Aarnav Choudhary, Matheus Fonseca Rocha, Jiwon Seo, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2607.27240)  

**Abstract**: Model merging is often used to combine capabilities from separately fine-tuned models without additional training, but it is unclear whether standard merging methods preserve multiple safety-relevant behaviors simultaneously. We study this question through a controlled case study using two Gemma-3-1B-IT finetunes on two complementary safety objectives: CARES harm-level classification and WildJailbreak adversarial refusal. We merge the two fine-tunes using Linear, SLERP, TIES, and DARE-TIES, and evaluate the merged models on classification accuracy, attack resistance, and benign compliance. Across all four methods, attack resistance transfers significantly more than classification accuracy: merged models retain 81-85% jailbreak refusal rates while CARES accuracy falls to at most 12.9%. Weight-space measurements suggest that this asymmetry is not caused by strongly opposing task-vector directions: the two task vectors are nearly orthogonal (cosine similarity 0.011). Instead, the refusal fine-tune induces consistently larger per-layer task-vector magnitudes, causing magnitude-sensitive methods to favor refusal updates. These results show that standard model merging can collapse safety recognition into broad refusal when safety-relevant task vectors differ substantially in scale. 

---
# RadHarmony: Radiological Data Handling in the Era of Agentic AI 

**Authors**: Frank Li, Bardia Khosravi, Mohammadreza Chavoshi, Theo Dapamede, YoungSeok Jeon, Janice Newsome, Hari Trivedi, Judy Gichoya  

**Link**: [PDF](https://arxiv.org/pdf/2607.27235)  

**Abstract**: Training deep learning models on radiological images requires integrating heterogeneous datasets across different sources, file formats, directory layouts, label schemas, and annotation types. We present RadHarmony, an open-source Python library that provides a unified API for loading, harmonizing, and augmenting radiological datasets, with a primary focus on chest radiographs and early support for computed tomography (CT) and magnetic resonance imaging (MRI). RadHarmony standardizes metadata from 24 public datasets into a single tabular format, wraps MONAI's map-style datasets for deep-learning-ready sample delivery with optional on-disk caching, and supports classification labels, segmentation masks, bounding boxes, and radiology report text through a single interface, with an interactive visualization tool for dataset exploration and verification. To lower the barrier for integrating new datasets, RadHarmony introduces an AI-agent skill that guides the full integration workflow from raw data inspection through code generation and testing. We demonstrate the library's utility by pretraining RadHarmony-ViT, a reference vision transformer baseline that combines three heterogeneous chest radiograph datasets with no dataset-specific code. The code and pretrained model weights are available at this https URL. 

---
# KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation 

**Authors**: Peiyu Zang, Jian Tao, Jialing Zhang, Yichen Yuan, Wentao Zhang, Guang Liu, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27231)  

**Abstract**: Large language models (LLMs) have significantly increased the demand for efficient accelerator kernels, but kernel development remains a highly specialized and labor-intensive task. The recent rise of LLMs and agentic frameworks offers a promising pathway toward automatic kernel generation. However, despite rapid progress, there is still no comprehensive benchmark to rigorously evaluate LLM-generated kernels across diverse operator sources or heterogeneous hardware platforms. We present KernelGenBench, a unified benchmark for systematically evaluating LLM- and agent-generated Triton kernels across diverse operator sources and heterogeneous hardware platforms. It comprises two complementary sub-benchmarks: KernelGenBench-MS (Multi-Source), evaluating 210 operators from three sources beyond standard PyTorch-centric tasks, and KernelGenBench-MC (Multi-Chip), measuring performance portability across six heterogeneous hardware platforms using a 110-operator subset. Our large-scale evaluation, consuming over 15 billion tokens, shows: (1) agent-based methods consistently outperform pure LLM sampling methods, while cuBLAS operators are the most challenging across all methods; (2) generation performance varies significantly across hardware platforms, with even recent kernel-specialized agents experiencing severe cross-platform degradation (e.g., AutoKernel drops from 87% on NVIDIA to 25% on Platform E); (3) autonomous kernel generation remains highly cost-intensive, with specialized agent methods averaging 5.11 million tokens per successful operator (AKO4all reaches 5.19 million), orders of magnitude higher than simple LLM sampling approaches. 

---
# Multi-Head Attention Residuals 

**Authors**: Cheng Luo, Zefan Cai, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27230)  

**Abstract**: Transformers propagate information across depth through a single additive residual stream: every sublayer reads only the most recent state. Attention residuals relax this by letting each sublayer attend, through a learned softmax. However, that read uses a single query shared across the entire width, so every feature subspace must read the depth history through one distribution. The cost of this forced compromise grows with how much the subspaces disagree about which layers to read, and disagreement grows with model width. We introduce Multi-Head Attention Residuals (MHAR): the routing query is reshaped into H per-subspace heads, each with its own softmax over the depth history. The read becomes block-diagonal, the reshape adds zero parameters and negligible compute, and H = 1 recovers attention residuals exactly. Trained from scratch on a deduplicated Nemotron-based anneal corpus that is quality-filtered and STEM- and code-heavy, MHAR improves validation loss over a standard Transformer at 100M, 350M, and 1B (-0.061, -0.149, and -0.140). It achieves the best result among four methods in every setting, with the gain increasing from 100M to the larger scales. The head count is a real design axis rather than a free knob: validation loss is U-shaped with respect to H, with a flat optimum at H = 4 or H = 8 across scales. We adopt H = 8 for large-scale models; over-splitting beyond this point (H = 16) consistently gives back part of the gain. A direct probe of the trained queries confirms that learned subspace disagreement is the underlying driver. Fused Triton routing kernels increase attention-residual training throughput from 0.2-0.5x to 0.55-0.88x of the baseline while maintaining near-baseline peak memory. An identity-preserving conversion using delta attention residuals supports 8B mid-training, yielding improvements of +3.2 on GSM8K and +3.1 on GPQA. 

---
# Learning to Trace Seiberg Dualities 

**Authors**: Jonathan J. Heckman, Shani Meynet, Alessandro Mininno, Gary Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28628)  

**Abstract**: Dualities play an important role in establishing both microscopic and emergent phenomena in a wide range of physical systems. In practice, though, it can often be computationally challenging to establish when two systems are dual, even when all of the "rules of the game" are well-known. Said differently, when confronted with two systems, how can one efficiently establish that they are in fact dual? In this paper we use machine learning methods to address this question for Seiberg dualities of supersymmetric quiver gauge theories. Mathematically, this involves establishing mutations of quivers, which is in turn a variation on the theme of "learning to unknot". On the one hand, this leads us to a practical tool for establishing the computational complexity of different dualities. On the other hand, it also allows us to study how different network architectures learn how to trace Seiberg dualities. We find that for quivers with a modest number of quiver nodes (of order $10$), different network architectures consisting of transformers and multi-layer perceptrons tend to outperform deterministic algorithms. Supplementing the network by well-established pathfinder algorithms (essentially "Google Maps for quivers") leads to an additional improvement in the efficiency and accuracy of the search strategy. We anticipate that this class of questions can serve as a useful benchmark for frontier AI models applied to theoretical physics. 

---
# ReToken: One Token to Improve Vision-Language Models for Visual Retrieval 

**Authors**: Yao Xiao, Reuben Tan, Zhen Zhu, Yuqun Wu, Jianfeng Gao, Derek Hoiem  

**Link**: [PDF](https://arxiv.org/pdf/2607.28627)  

**Abstract**: Long visual context poses a challenge for vision-language models: performance degrades as the number of distractors grows, and processing all tokens at once is computationally infeasible under GPU memory constraints. We present ReToken, a single learnable embedding trained as an explicit retrieval target that selects a sparse set of query-relevant visual tokens from a pre-filled visual KV cache. Trained on only a small image-QA dataset, ReToken yields consistent gains across image and video benchmarks: on Visual Haystacks it improves Qwen3VL-8B by 13.4 points and InternVL3.5 by 12.4 points (>20% relative), and on LVBench it transfers zero-shot to long video for an 8.0-point gain with Qwen3VL-8B. Thanks to its lightweight design, both training and long-video inference fit on a single H100. Code is available at: this https URL 

---
# PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball 

**Authors**: Lizhi Yang, Junheng Li, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2607.28623)  

**Abstract**: We present PAC-MAN, a perception-aware CBF-RL framework that couples control-barrier safety with deployment-realistic onboard sensing for whole-body humanoid dodgeball. The deployed policy sees the ball only as segmentation-masked depth from a head-mounted camera, while training-time CBF guidance represents clearance to every body link, and an adversarial motion prior regularizes the resulting evasive reflexes. We evaluate on a controlled any-link contact benchmark with seeded throws in two regimes: single throws and a deployment loop in which the robot walks back to its station and recovers between throws. On this benchmark, the policy comes within a few points of a privileged state oracle: a fixed onboard camera alone is adequate for evasion. We find that usable barrier structure depends on perceptual observability: Joint-CBF gives the best performance with accurate ball states, degrades under fixed-camera observations when used only as training guidance, and recovers with a ball-tracking gimbal or privileged runtime filter. We therefore deploy a lightweight Link-CBF policy zero-shot on the Unitree G1 in the real world, where it tolerates imperfect perception, succeeds on 95% of throws, and uses semantic segmentation to dodge different balls. 

---
# AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis 

**Authors**: Bing Yan, Gregory Wolfe, Stefano Martiniani, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2607.28618)  

**Abstract**: Chemistry literature synthesis often requires assembling specific findings scattered across many publications, yet existing literature-search systems primarily return ranked document lists. As a result, scientists and AI agents need to locate relevant information, verify their provenance, and assemble cross-paper answers manually. We present AskChem, a claim-centered infrastructure for cross-paper chemistry search. AskChem changes the unit of retrieval from the paper to the provenance-carrying claim: each paper is converted into atomic, typed claims, each grounded by a source DOI and a verbatim quote or an explicit evidence locator. Over this shared claim store, AskChem exposes complementary structures for search and synthesis: a stabilized faceted taxonomy for hierarchical retrieval and browsing, an evidence graph linking claims through relations, and an exploratory living taxonomy that situates indexed papers under scientific principles. AskChem currently indexes 2.4M claims from 147K papers and provides a web interface, as well as REST, SDK, and MCP access for AI agents. On AskChem-Bench, grounding a GPT-5.5 reader in AskChem yields 100% resolvable DOIs, compared with 88.3% without retrieval, and the highest citation density among five tested systems. AskChem is live at this https URL. 

---
# PAIChecker: Uncovering and Checking PR-Issue Misalignment in SWE-Bench-Like Benchmarks 

**Authors**: Manyi Wang, Junjielong Xu, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2607.28587)  

**Abstract**: SWE-bench-like benchmarks are widely used for evaluating LLM's issue resolution capability. They typically follow a common construction pipeline: each PR (Pull Request) is paired with its linked issue by extracting issue references from the PR description; the issue description is used as the problem statement, and the PR patch serves as the test oracle. However, due to the inherent complexity of developing and maintaining large repositories, such PR-Issue pairings are often misaligned in practice. In this work, we systematically study SWE-bench Verified instances, finding that 13.6% exhibit misalignment across five patterns in eleven fine-grained scenarios. To enable reliable and scalable construction of those benchmarks in the future, we propose PAIChecker, a multi-agent system for checking PR-Issue misalignment in SWE-bench-like benchmarks. Specifically, PAIChecker adopts a three-phase design that combines specific pattern identification, cross-agent label synthesis, and code-level validation, thereby enabling more accurate, generalizable, and progressively verified detection. Experiments on SWE-Gym and SWE-bench Multilingual show that PAIchecker achieves the best performance across all four LLM backbones, reaching up to 92.12% and 91.67% binary accuracy, respectively. 

---
# Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B 

**Authors**: Iliya Mirzaei  

**Link**: [PDF](https://arxiv.org/pdf/2607.28576)  

**Abstract**: Methods that make a language model plan, criticise and rewrite its own answer, reflect on mistakes, pick the best of several attempts, or debate with copies of itself nearly all make it generate far more text than a single chain of thought. Because generating more text raises accuracy by itself, a gain over one chain of thought does not show the method's idea is what helped. Wang et al. (2024) reported that a simple baseline, sampling the same question repeatedly and keeping the most common answer, often wins once budgets are comparable, but gave point estimates with no confidence intervals or significance tests.
We rerun that comparison as a designed experiment: seven methods, open models of 1.5B, 3B and 7B parameters, two mathematics benchmarks, 150 questions each. We count every generated token, including those spent on critiques, reflections, debate turns and checking, and compare each method against repeated sampling at its own measured cost. All 36 comparisons are paired by question, with bootstrap intervals and multiplicity correction.
No method is reliably better than repeated sampling at equal cost anywhere. Ten are reliably worse, all of them methods where the model inspects its own output, and all 18 self-inspection comparisons are negative. The two kinds of self-inspection part company as models grow. Choosing stops hurting: taking Best-of-N's eight samples and just counting the most common answer beats letting the model pick by 8.0 and 11.3 points at 1.5B, but only 2.0 and 1.3 at 7B, no longer distinguishable from zero. Rewriting does not recover: Self-Refine and a forced Reflexion stay 3.6 to 10.1 points below baseline at 7B.
Reflexion as published never triggered its own retry on the smallest model. It judged itself correct every time and silently became a single chain of thought. We release code, prompts, all generations, and our verification scripts. 

---
# Algorithms for Structured Elections under Thiele Voting Rules 

**Authors**: Alexandra Lassota, Krzysztof Sornat  

**Link**: [PDF](https://arxiv.org/pdf/2607.28575)  

**Abstract**: We study the computational complexity of winner determination problems in approval-based committee elections under Thiele voting rules. These form a class of rules parameterized by a fixed weight vector that specifies how a voter's satisfaction depends on the number of approved candidates elected. We first analyze the structure of optimal solutions based on the sets of voters who approve each candidate---that is, how voters' approval ballots induce dependencies between candidates---revealing constraints on a winning committee under any fixed Thiele voting rule. Using this, we design FPT algorithms for Proportional Approval Voting (PAV) and other Thiele rules on a natural restricted domain known as the Voter Interval (VI) domain---that is, after a suitable ordering of voters, each candidate is approved by a consecutive interval of voters. In particular, we show that every Thiele rule on VI is FPT with respect to a parameter for which the problem is NP-hard on general instances, even when the parameter takes constant values. Our results advance the understanding of the computational complexity of PAV on Voter Interval instances, which remains one of the central open questions in this area. We further resolve two open questions from the literature on PAV (and other Thiele voting rules) by providing a polynomial-time algorithm for instances where each candidate is approved by at most two voters, and an FPT algorithm parameterized by the total score of a winning committee. 

---
# APO: Unsupervised Atomic Policy Optimization for 3D Structure Prediction of Atomic Systems 

**Authors**: Shentong Mo, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2607.28553)  

**Abstract**: Predicting the 3D structures of atomic systems is fundamental to advancing material science and drug discovery. While flow-matching models (, FlowDPO) have recently shown promise in this domain, their performance relies heavily on alignment with ground-truth coordinates via supervised preference learning. However, obtaining experimental labels for novel crystal phases or de novo proteins is prohibitively expensive, creating a bottleneck for structural modeling in data-scarce regimes. In this work, we propose (Atomic Policy Optimization), a fully unsupervised alignment framework that eliminates the need for ground-truth reference structures. APO adapts group-relative policy optimization to 3D atomic environments, utilizing a novel dual-reward mechanism: (i) a that reinforces the policy's dominant latent structural modes through eigen-decomposition of sample similarities, and (ii) a that enforces thermodynamic stability. Our framework enables the model to ``self-correct'' by identifying physically plausible configurations within sampled groups. Extensive benchmarks on crystal and antibody structure prediction demonstrate that APO consistently outperforms fully supervised baselines, achieving a new state-of-the-art in match rates and structural fidelity. Furthermore, we show that APO effectively straightens probability paths, significantly improving inference efficiency. Our results suggest that intrinsic physical consistency can serve as a superior guide for alignment compared to noisy, supervised coordinate matching. 

---
# ORCA-bench: How Ready Are Language Model Agents for Oncall? 

**Authors**: Albert Gong, Kyuseong Choi, Abhineet Agarwal, Jason Schechner, Ryan Huang, Raj Agrawal, Anish Agarwal, Raaz Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2607.28545)  

**Abstract**: Large language models can write, patch, and search code, but oncall root cause analysis (RCA) demands something different: reasoning over noisy metrics, logs, traces, and source code, starting from ambiguous user-facing reports, often hours after the incident began. We introduce ORCA-bench, a benchmark that puts general-purpose coding agents in a production-fidelity oncall setting. ORCA-bench pairs a live OpenTelemetry-instrumented microservice system--exposing six days of metrics, logs, and traces through real telemetry interfaces (Prometheus, Jaeger, and OpenSearch via Grafana) and full source-code access--with 1,079 RCA tasks that systematically vary report specificity, time-to-detection, and co-occurring fault scenarios. Ground-truth symptoms are curated and signed off by expert SREs, and our LLM-as-judge is independently re-scored by humans (Cohen's $\kappa_w=0.90$). Across five frontier agents, the best RCA Accuracy is 25.3% on Medium-difficulty tasks (the realistic-input setting) and 10.0% on Hard--a gap that remains even with Claude Fable 5. The weakest model hallucinates an implausible root cause in 40% of incident reports, and removing source-code access degrades every metric. Crucially, these are performances on a curated 50 GB / six-day testbed with tasks investigated in isolation on a system whose code and instrumentation are public. Since real production systems are order of magnitudes larger, more dynamic, and more idiosyncratic, the gap we report is a lower bound on the engineering investment required before frontier coding agents can be safely entrusted with production reliability. We release the public set at this https URL. 

---
# What to Remove, What to Preserve: Dual-Ambiguity Rectification for All-in-One Image Restoration 

**Authors**: Cencen Liu, Wen Yin, Dongyang Zhang, Dongmin Li, Shan Zhao, Bing Su, Tao He, Jielei Wang, Guoming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28526)  

**Abstract**: All-in-one image restoration aims to handle diverse degradations within a unified framework. Existing methods commonly encode heterogeneous degradation conditions in a shared latent space, where degradation-related cues and scene content can remain entangled. We characterize the resulting challenge as dual ambiguity: semantic ambiguity in channel-wise modulation and spatial ambiguity in restoration responses, which can lead to content corruption and residual artifacts. To mitigate this issue, we propose DAR-Net, a Dual-Ambiguity Rectification Network for all-in-one image restoration. DAR-Net first introduces a Degradation Archetype Representation (DAR) module to construct a structured degradation state through simplex-constrained archetype mixture modeling. Based on this state, a Semantic Ambiguity Rectification (SeAR) module generates degradation-aware prompts to improve channel-wise conditioning in the decoder. A Spatial Ambiguity Rectification (SpAR) module further regularizes degradation-aware and complementary features toward orthogonal response subspaces, reducing spatial interference between removal and preservation cues. Extensive experiments on standard all-in-one restoration benchmarks show that DAR-Net achieves the best overall performance under both three-degradation and five-degradation settings, improving the average PSNR over the strongest competitor by 0.14 dB and 0.34 dB, respectively; it additionally shows superior performance on CDD-11 and WeatherBench. 

---
# Agents That Certify Their Own Exploits: Confidence-Scheduled Restricted Responses for Safe Opponent Exploitation 

**Authors**: Boning Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28520)  

**Abstract**: An agent playing a Nash-equilibrium strategy in a two-player zero-sum imperfect-information game secures the game value but forfeits the additional value offered by a flawed opponent. Diffuse deviations pose a particular challenge: binary release rules may gather too little evidence to act, while a full best response to an incomplete opponent model can be highly exploitable. We introduce \emph{budget-constrained confidence-scheduled restricted responses} (CS-RNR), the first opponent-exploitation method whose safety guarantee is a certificate the agent computes on the strategy it actually deploys, so that every exploit it commits to is one it has audited itself. The method tracks pooled action frequencies with anytime-valid confidence sequences and treats a frequency as exploitable only once its interval separates from an equilibrium reference. The confirmed deviations define a conservative opponent model, which a restricted-response solve turns into candidate counter-strategies over a grid of pin levels. Before deployment, each complete candidate is evaluated by a full-tree best response. The resulting certificate is compared with a user-specified budget and committed atomically with the strategy. Because this check is performed on the played strategy, model quality determines the exploitation achieved while the certificate controls reference-relative expected loss. In Leduc hold'em, CS-RNR obtains $6.2\times$ the steady-state gain of a money-verified binary gate while keeping every deployed strategy within budget. A trajectory mixture using the same estimator reaches $13.6\times$ the budget. Across Leduc, Liar's Dice, and 5-rank Leduc, all $36{,}000$ audited hands satisfy the reported certificate tolerance. 

---
# TCA-SIR: Learning Target-Conditioned Abstractions for Scientific Inspiration Retrieval 

**Authors**: Yuto Suzuki, Farnoush Banaei-Kashani  

**Link**: [PDF](https://arxiv.org/pdf/2607.28498)  

**Abstract**: Scientific hypothesis generation for AI for Science typically involves Scientific Inspiration Retrieval (SIR) followed by hypothesis composition. Existing SIR methods rank papers by topical similarity and do not explicitly represent how a candidate inspiration transfers to a target problem. This is especially limiting for remote inspirations, whose value often lies in reusable problem-solving principles rather than topical overlap. Motivated by how humans abstract transferable aspects of a source and remap them to a new target, we reformulate SIR as target-conditioned abstraction (TCA). The retrieval object is a transferable abstract principle extracted from a candidate specifically for the target. We present TCA-SIR, which learns to generate target-conditioned abstractions and uses their representations to predict transferability. On ResearchBench, TCA-SIR outperforms prior SIR methods and direct LLM retrieval, improving HitRate@top4% over MOOSE-Chem by more than 10 percentage points. Learned abstractions also recover target-relevant mechanisms more clearly than an untrained TCA prompt, yielding both stronger retrieval and an interpretable rationale for scientific inspiration. 

---
# Machines that know they are aging: a framework for hardware-aware autonomous intelligence 

**Authors**: Cheng Siong Chin, Jianhua Zhang, Mohan Venkateshkumar  

**Link**: [PDF](https://arxiv.org/pdf/2607.28451)  

**Abstract**: Autonomous systems inevitably age, yet their artificial intelligence typically assumes hardware remains in its original condition. Batteries degrade, sensors drift, processors accumulate timing errors, and memory reliability declines, creating a growing mismatch between assumed and actual capability. This can lead to agnostic collapse, where mission failure arises from accumulated hardware degradation rather than a single component fault. We propose Aging-Aware Autonomous Intelligence (AAAI), a framework that integrates hardware health directly into reasoning, planning, and mission execution. AAAI is built on three pillars: hardware self-awareness, which continuously estimates the health of power, sensing, memory, and computation subsystems using physics-of-failure models; self-adaptive reasoning, which adjusts inference complexity, planning horizon, and task priorities according to remaining hardware capability; and survival-centric intelligence, which allocates remaining operational life across mission objectives through performance optimization, resource conservation, and graceful degradation. Rather than introducing new hardware, AAAI unifies prognostics, lifecycle management, and hardware-aware computing into a closed-loop cognitive architecture. We argue that such integration is essential for autonomous systems operating in inaccessible or safety-critical environments, including space missions, marine robotics, and implantable medical devices. By enabling machines to recognize and respond to their own aging, AAAI improves resilience, extends operational lifetime, and supports safer, more graceful mission completion. 

---
# QQWorld: Quantile-Quantile Matching for World Model Regularization 

**Authors**: Zhoushun Yu, Xiaoyu Hu, Xiangyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28415)  

**Abstract**: Latent world models enable efficient planning by predicting future states in a compact representation space, but their performance depends critically on the quality of the learned latent distribution. LeWorldModel (LeWM) regularizes its latents toward an isotropic Gaussian using the Epps-Pulley (EP) objective. We show that the corrective gradients of EP rapidly vanish for isolated tail samples, leaving heavy-tailed deviations insufficiently controlled. To address this limitation, we propose QQWorld, which replaces EP with a quantile-quantile matching objective that directly aligns projected latent samples with rank-matched Gaussian quantiles, thereby maintaining effective corrective gradients in the tails. We further develop cross-batch QQ, which enlarges the effective ranking pool using detached samples from previous batches, and characterize its bias-variance trade-off. Across four control environments, QQWorld effectively improves the average planning success rate of LeWM, while consistently yielding better Gaussian alignment and thinner latent tails. 

---
# On-Policy and Off-Policy Learning for Large Action Spaces 

**Authors**: Imad Aouali  

**Link**: [PDF](https://arxiv.org/pdf/2607.28408)  

**Abstract**: This thesis studies policy learning in interactive systems where an agent observes a context, selects an action from a very large set, and receives partial feedback. The main framework is contextual bandits, with two paradigms: on-policy learning, where the agent interacts sequentially with the environment and minimizes regret, and off-policy learning, where it learns from logged data collected by a logging policy. In large action spaces, both settings face major challenges: inefficient exploration, sparse data coverage, high-variance importance weights, extrapolation bias, and difficult optimization landscapes. The first part develops structured Bayesian methods for on-policy learning. We introduce meTS, a mixed-effect extension of Thompson sampling, and dTS, which leverages diffusion-inspired priors to model dependencies between actions. These methods share information across actions and yield regret guarantees depending on an effective number of actions. The second part addresses off-policy learning. We propose sDM, a structured direct method based on latent variables, show that optimization error can dominate estimation error in large action spaces, and introduce concave, efficiently optimizable policy-weighted log-likelihood objectives. Finally, we develop differentiable pessimistic methods based on exponential smoothing and PAC-Bayesian bounds to control the bias-variance trade-off of regularized importance-sampling estimators. 

---
# ShadowDancer: Teaching Video World Models Any Action by Learning Unified Dynamics Representations from a Video and Its Shadow 

**Authors**: Jin Cao, Zian Meng, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28362)  

**Abstract**: We present ShadowDancer, a novel approach to any-action, frame-level control of interactive video world models. The obstacle is representational: existing interfaces either encode an action loosely, leaving how it unfolds for the model to improvise, or encode it exactly through structured signals that serve one family and are hard to acquire, so precise control across diverse dynamics remains impractical. Demonstration videos are the natural remedy, specifying any dynamics frame by frame; yet a video shows its dynamics only through one particular appearance, a single shadow of the underlying dynamics, so actions learned from demonstrations transfer poorly to new scenes. ShadowDancer addresses this with two key innovations: (1) shadow pairs, video pairs that replay the same dynamics under independently resampled appearance, constructed at scale by our Shadow Library, so that a dynamics family becomes controllable exactly when such pairs can be constructed for it; and (2) cross-shadow prediction, which learns actions by predicting one shadow from the other, so that whatever the pairing resamples is discarded by construction and whatever it preserves becomes the action, yielding a unified dynamics representation that drives a block-causal world model. Any demonstrated clip thus becomes a reusable action asset, replayed in new environments without action labels, motion estimators, or fine-tuning. Experiments demonstrate improved action transfer and long action rollout over strong latent-action and interactive world model baselines across diverse dynamics families, with an average blinded win rate of 86% in rollout comparisons. We show video results at this https URL 

---
# Teffic-Audio: Tell Fact from Fiction 

**Authors**: Wan Lin, Li Wang, Jindong Wang, Kunyu Feng, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28351)  

**Abstract**: Speech deepfake detection has expanded in scope with increasingly heterogeneous spoofing mechanisms, including speech synthesis, voice conversion, vocoder reconstruction, and neural-codec resynthesis. The resulting spoofing artifacts can be further shaped by variability in source speech, recording environments, and transmission channels. This variability makes robust generalization across heterogeneous conditions a central requirement for practical detection systems. This report presents Teffic-Audio, a general speech deepfake detection system designed for comprehensive evaluation environment. Teffic-Audio adopts a straightforward detector architecture consisting of a Conformer-based speech encoder, multi-head attentive statistics pooling, and a binary classifier. Rather than relying on additional architectural complexity, the system improves generalization through its training recipe, which integrates multi-source data, attack- and source-balanced sampling, and diverse audio augmentation. Trained only with open-source data, Teffic-Audio achieves a pooled EER of 1.454% on the 14 test sets of Speech-DF-Arena, outperforming all currently public systems on the leaderboard. It also obtains the lowest EER on five individual test sets and shows a favorable performance-complexity trade-off compared with larger leading systems. Overall, Teffic-Audio provides a strong and practical reference system for general speech deepfake detection. 

---
# ObjectStream: Latent Objects as Memory Anchors for Streaming Video Understanding 

**Authors**: Mingkang Dong, Muxin Pu, Jie Li, Bohan Guo, Songruo Chen, Bin Ren, Xu Zheng, Chen Zhao, Tianwen Qian, Mohamed Elhoseiny, Yuqian Fu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28312)  

**Abstract**: Streaming video understanding requires models to continuously retain useful visual evidence before future questions are known. Existing approaches primarily manage the growing visual context according to token importance, temporal redundancy, or segment-level relevance, but rarely organize evidence around objects that persist and evolve over time. Thus, in this paper, we introduce ObjectStream, a training-free framework that treats latent objects as memory anchors for streaming video understanding. ObjectStream induces spatially coherent latent objects directly from frozen Video-LLM representations, links them across frames into persistent anchors, and maintains their histories under a bounded memory budget, without requiring external object detectors or segmentation models. Built on these anchors, ObjectStream preserves three complementary forms of evidence: persistent object histories, transient object changes, and recent visual context. This design enables existing Video Large Language Models (Video-LLMs) to reason over object identities, interactions, and state changes while leaving the underlying model unchanged. Extensive experiments on online streaming and offline long-video benchmarks demonstrate both effectiveness and efficiency. In online streaming evaluation, ObjectStream improves Qwen2.5-VL-7B by 10.0 points on OVO-Bench Real-Time Visual Perception, while reducing peak GPU mem-ory and TTFT by approximately 50%. On offline long-video benchmarks, it surpasses the full-token baseline while discarding 82.5% of visual tokens. These results highlight latent objects as a practical and effective organizing principle for compact streaming video memory. 

---
# From Textual Requirements to Microservice Architectures - A Comprehensive Evaluation of LLM-Based Design Synthesis 

**Authors**: Danyllo Albuquerque, José Renan, Guillermo Rodríguez, Guillermo Rodríguez, Emanuel Dantas, Ademar França, Mirko Perkusich, Kyller Gorgônio, Angelo Perkusich  

**Link**: [PDF](https://arxiv.org/pdf/2607.28307)  

**Abstract**: Microservice architectures have become dominant for modernizing monolithic systems, yet identifying appropriate services remains challenging and largely manual. Existing decomposition approaches are predominantly code-centric, limiting applicability in early design stages where only textual requirements are available. Despite advances in Large Language Models (LLMs), limited empirical evidence exists on their ability to synthesize complete microservice architectures from natural-language requirements, including service definitions and inter-service interactions. This study investigates whether an LLM can bridge requirements engineering and architectural design, generating architectures solely from textual requirements and evaluating structural agreement and perceived quality of results. We conduct a mixed-method study using OpenAI o3 under zero-shot (ZS) and few-shot (FS) prompting across two systems (Bookstore, PetClinic), one execution per system/condition. Architectures are evaluated through (i) comparison with reference architectures using precision, recall, and F1-score for service identification and communication recovery, and (ii) a blinded expert assessment of correctness, completeness, modularity, and plausibility, plus open feedback synthesis. OpenAI o3 identifies services with higher agreement under FS prompting (F1 = 0.79 for ZS versus = 0.97 for FS). Communication recovery is more challenging: ZS produces dense architectures with high recall but low precision (F1 = 0.61), while FS improves agreement, reaching F1 = 0.82 and reducing unsupported dependencies. Expert evaluation corroborates these results, with FS architectures perceived as more modular, coherent, and plausible than ZS outputs. OpenAI o3 shows potential for requirements-driven synthesis when guided by exemplar prompting. Results are model- and context-specific from two small systems, not model-independent proof. 

---
# MonoVoc: Decoupling Geometry and Semantics for Lightweight Monocular Open-Vocabulary 3D Gaussians 

**Authors**: Pouya Ardekhani, Zahra Dehghanian, Morteza Abolghasemi, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2607.28300)  

**Abstract**: Open vocabulary 3D scene understanding is essential for next-generation interactive systems, empowering users to intuitively query and navigate reconstructed environments using natural language. However, current 3D Gaussian frameworks are often bottlenecked by restrictive multiview capture requirements, costly scene-specific optimization, and the massive memory overhead of storing dense language features. We present a novel, training-free pipeline that fundamentally reimagines this paradigm by explicitly decoupling 3D geometric reconstruction from semantic integration. Given a standard monocular video sequence as input, our method efficiently outputs a compact, highly interpretable, and fully searchable object-level semantic Gaussian map. Rather than entangling heavy language embeddings within the mapping loop, we extract geometry independently and ground semantics through a lightweight, modular post-processing framework. Extensive evaluations on the Replica dataset demonstrate that this decoupled architecture preserves strong rendering fidelity and competitive segmentation accuracy. Crucially, by replacing dense per-Gaussian storage with modular, object-level semantic embeddings, our approach delivers an order-of-magnitude reduction in memory usage compared to SOTA baselines. This provides a highly efficient, scalable, and practical solution for open-vocabulary 3D retrieval and question answering directly from everyday monocular video. 

---
# CACHE-UK: A Stability-Aware Memory Editor for Sequentially Updated Quantized LLMs in Finance 

**Authors**: Anubhav Lakra, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.28292)  

**Abstract**: Large Language Models (LLMs) deployed in dynamic financial environments face a critical challenge: maintaining factual accuracy as market conditions, regulations, and corporate facts change continuously. While 4-bit quantization enables efficient deployment, it severely limits the viability of sequential memory editing: existing methods undergo catastrophic performance degradation under this "quantization stability crisis." We introduce CACHE-UK (Contextual Adaptive Continual Hybrid Editor for UK Finance), a stability-aware memory editing framework specifically designed for domain-specific, quantized LLMs. CACHE-UK integrates three components: a rank-1 LoRA perturbation mechanism that confines edits to the low-rank adapter subspace, a financial domain prioritization module for content-adaptive edit strength, and a closed-loop Stability Controller that tracks "degradation debt" to prevent catastrophic forgetting across sequential updates. Evaluated on a 4-bit quantized OpenLLaMA-3B model with a curated UK financial corpus of 88,021 documents, CACHE-UK reduces knowledge degradation by 11-17% relative to adapted baselines under identical 4-bit constraints -- its most robust effect -- while attaining the highest test success (generalization) rate observed in our setting (28%, a 6 percentage point improvement over the strongest adapted baseline). These results indicate that stability-aware editing can improve factual maintenance in resource-constrained financial LLM deployments, though absolute generalization rates remain low. 

---
# Agentic Method for Deterministic Validation of Legacy Code Migration 

**Authors**: Andras Ferenczi, Jordan Docherty, Mariya Bessonov, Matthew Findlay, Krishna Lingamneni  

**Link**: [PDF](https://arxiv.org/pdf/2607.28271)  

**Abstract**: Migration of legacy COBOL programs to Java requires extensive testing to ensure correct functionality. This effort is often complicated by the lack of test data and the difficulty of validating all corner cases. In this paper we propose a novel agentic test-synthesis method, the "Locksmith Loop," which is initiated by preparing two runtime environments: the COBOL source and the generated Java target are each instrumented with mocks and executed off-mainframe on commodity hardware, then an iterative agentic loop performs Witness Search over input mocks to penetrate program branches, followed by parity-preserving mutations. When routing boundaries are reached, an analyzer identifies a Locked Paragraph: a condition preventing deeper exploration. Across three COBOL-Java case studies, spanning two open-source programs and one internal production-like COBOL program and ranging from 430 to 4,114 source lines, Locksmith consistently improved coverage beyond input-search plateaus, reaching nearly complete coverage on the two open-source programs and 91.90% branch coverage on the internal production-like COBOL program. The generated Java matched the COBOL reference under deterministic parity checks in all accepted test cases. Through these findings we demonstrate, to the best of our knowledge, a novel approach for validating agentic coding output using a deterministic oracle. 

---
# Theia: Large-Scale Multimodal Captioning and Automated Validation of the Incidents1M Dataset for Data-Free Distillation 

**Authors**: Simone Giano, Lorenzo Severini, Alessandro Galdelli, Adriano Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2607.28269)  

**Abstract**: The deployment of Vision-Language Models (VLMs) in critical domains like disaster management requires high-quality multimodal datasets, especially for transferring knowledge via Data-Free Knowledge Distillation (DFKD). However, existing datasets in this domain either entirely lack descriptive text, such as Incidents1M, or suffer from severe text-image semantic misalignment, such as CrisisMMD. In this work, we present a novel methodology to construct and automatically validate a large-scale multimodal dataset for disaster response. Starting from the vision-only Incidents1M, we successfully recovered 100,000 images and generated high-fidelity textual descriptions using two distinct Qwen3.5 architectures: a 4B dense model and a 35B Mixture-of-Experts (MoE) model. To ensure the generated captions provide reliable semantic anchoring for DFKD, we introduce an image-blind LLM-as-a-Judge validation pipeline leveraging Qwen3.5-9B. By intentionally obscuring the original image from the judge, this evaluator accurately simulates the modality gap of the student model during data-free distillation. Our evaluation across 173,179 label pairs demonstrates a high semantic agreement (78.65/100) between the two architectures. Furthermore, the automated evaluation reveals a conservative captioning behaviour, characterized by a high Precision (77.6%) and low Recall (46.0%). This minimizes the false positive noise, while simultaneously exposing underlying human annotation inconsistencies in the original ground truth. This work provides a scalable, LLM-validated multimodal dataset and a reproducible framework to advance cross-modal knowledge distillation. 

---
# EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE 

**Authors**: Zexuan Yan, Yuzhou Wu, Yue Ma, Zonghang He, Kaibo Yin, Xiaobing Tu, Yinggui Wang, Jinkui Ren, Xiantao Zhang, Shijian Wang, Jinghong Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28243)  

**Abstract**: Egocentric video offers rich manipulation experience for embodied AI, yet collecting diverse egocentric data across scenes, objects, motions, and embodiments remains costly. We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms. Online Anchored Projective Memory (OAPM) preserves a first-frame 3D scene anchor while periodically refreshing a recent state during autoregressive generation. Action-3D Rotary Position Embedding (A3D-RoPE) encodes end-effector motion with camera-aware 3D rotary coordinates, injecting action geometry into skeleton-to-video cross-attention for precise control. Together, these components improve visual fidelity, geometric stability, and action alignment in long egocentric rollouts. Moreover, augmenting 400 real trajectories with 400 \method-generated trajectories improves out-of-distribution real-robot success from 77\% to 84\% on single-arm tasks and from 53\% to 70\% on dual-arm tasks, demonstrating that the synthesized data substantially improve downstream WAM generalization. 

---
# Agentic Metaverse Services: A New As-a-Service Paradigm 

**Authors**: Xiaofei Xu, Quan Z. Sheng, Zhongjie Wang, Boualem Benatallah, Xiao Wang, Ruipeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.28242)  

**Abstract**: Generative Artificial Intelligence (GenAI) is reconstructing the digital virtual world, upgrading agents through enhancing their abilities in autonomous learning, multi-modal interaction, content generation, and collaborative decision-making. In particular, the shift from conversational chatbots to agentic AI, the most recent significant technical breakthrough of GenAI, has brought a new form of services, agentic services and Agent-as-a-Service (AaaS), in which the agent's abilities are encapsulated, such as perception, decision-making, execution, collaboration, and content generation, to provide the customized agent services to users. The metaverse is a virtual ecosystem for human life, work, creation, and entertainment, supported by the new generation of digital technologies. Through combining agentic services and the metaverse, an Agentic Metaverse Service, denoted as AMServ, is produced for metaverse business processing, as a new form of metaverse service. The AaaS in the metaverse environment, denoted as Meta-AaaS, as an approach to realize AMServ, has become a new paradigm of agentic services and service computing. This paper overviews the evolution and new features of agents and services empowered by GenAI, reveals the roles and principles of agentic services in the metaverse environment, presents the forms, characteristics, and principles of the AMServ and the Meta-AaaS, discusses the typical application examples of the AMServ and the Meta-AaaS, and finally points out the new tendencies and research directions of the AMServ and the Meta-AaaS. The AMServ and the Meta-AaaS will bring great opportunities to human society and services in the AI era, and promote the rapid development of emerging service industries in the future. 

---
# EMBL AI Librarian: Life-Sciences Knowledge Layer for AI Agents 

**Authors**: Luigi Sigillo, Matteo Silvestri, Francesco Tabaro, Rajat Bhatnagar, Syed Irtaza Mubashar, Matt Jeffryes, Daljit Nijjer, Vittorio Perera, Ola Spjuth, Julio Saez-Rodriguez, Melissa Harrison, Fabio Petroni  

**Link**: [PDF](https://arxiv.org/pdf/2607.28229)  

**Abstract**: The web is increasingly accessed by AI agents rather than humans. Every agent needs knowledge, especially in the life-sciences, where agentic pipelines are growing fast. Access to the literature is a crucial part of that need, and resources such as Europe PMC, with over 40M indexed records, are widely used to meet it. Yet these resources were not built for AI agents: they take keywords and complex syntax and return whole papers, so every agent must learn the syntax, issue several searches, and read full papers to find the evidence it needs. We introduce EMBL AI Librarian, a knowledge layer that upgrades the Europe PMC interface for AI agents: an agent asks in natural language and receives evidence that answers it. A single LLM orchestrates the whole knowledge retrieval process: it plans complementary subqueries executed by the live Europe PMC search engine, then reads the selected papers and locates the relevant evidence. We evaluate Librarian across four benchmarks: literature synthesis, claim verification, open-domain question answering, and downstream biology tasks such as protocol questions and sequence manipulation. On ScholarQABench, Librarian improves Citation F1 by more than $16$ points over strong recently published baselines. Used as the retrieval layer of an existing claim-verification pipeline, it increases agreement with expert consensus; and on the open-form LitQA2 benchmark, a GPT-5.4 agent scores about $8$ points higher when grounded in Librarian than with web search. Overall, our results show that equipping life-science agents with the Librarian knowledge layer improves performance across a range of tasks. We release our code publicly at this https URL 

---
# Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation 

**Authors**: Fazhong Liu, Zhuoyan Chen, Haozhen Tan, Yan Meng, Guoxing Chen, Haojin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28226)  

**Abstract**: World models give embodied AI a predictive core: they compress observations into states, simulate action-conditioned futures, and enable planning beyond reactive control. This predictive layer, however, opens a new security boundary-compromise can propagate from data, sensors, prompts, or feedback into physical action. Rather than treating world models as an isolated component, this survey traces threats across their entire lifecycle-from data construction and representation learning, through state grounding and imagination, to trajectory evaluation, execution, and long-term adaptation via memory and tools. We show that familiar attack families: poisoning, backdoors, adversarial examples, sensor spoofing, prompt injection, trajectory manipulation, and supply-chain attacks take on distinct meanings when they corrupt world states, learned dynamics, affordance estimates, or safety costs. We also highlight a duality: world models can serve as runtime safety shields, yet when compromised or over-trusted they generate predictive safety illusions. The survey offers a lifecycle taxonomy, maps existing attacks to world-model security properties, outlines evaluation protocols for safety failures, and structures defenses across provenance, robust grounding, uncertainty-aware prediction, trajectory gating, feedback auditing, and deployment assurance. 

---
# Vibe-FDTR: An agent-oriented framework for reproducible frequency-domain thermoreflectance data analysis 

**Authors**: Fuwei Yang, Weiheng Li, Bai Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.28200)  

**Abstract**: Frequency-domain thermoreflectance (FDTR) is a laser pump-probe technique widely used to measure thermal properties at the micro- and nanoscale; however, it relies on a complex data analysis procedure that demands substantial domain expertise and is susceptible to subtle human errors. Here, we present Vibe-FDTR, an agent-oriented framework that enables large language model (LLM) agents to perform reliable and reproducible FDTR analyses directly from natural language requests. This framework couples a configuration-driven FDTR code package, which enforces physical and parametric consistency, with procedural agent skills that translate user intentions into organized and verifiable analysis steps. We evaluate Vibe-FDTR using a controlled benchmark with two levels: synthetic single-step tasks and real-data multi-step tasks based on measurements of gold-coated graphite samples. Across the two levels, agents using Vibe-FDTR achieve success rates of 100% and 98.9%, respectively. In sharp contrast, ablating skills (Code-agent) reduces performance to 91.4% and 36.7%, which drops further to 38.6% and 0% when the domain package is also omitted (Agent-only). Beyond success rate, Vibe-FDTR also reduces computational cost by 87.7% relative to the Code-agent variant and cuts execution time by more than 60%. Finally, an optional expert mode supports experimental planning via autonomous sensitivity and uncertainty evaluations, and formulates physically grounded recommendations for underspecified tasks. These results demonstrate that encapsulating domain code and expert knowledge into agent skills offers a promising route toward low-barrier, autonomous, and trustworthy thermal metrology. 

---
# The MADRS Pipeline: Supporting Depression Assessment in Clinical Trials 

**Authors**: Mila Fodor, Katalin Ócsai, Francesco Periti, Rien Sonck, Alex Boudreau  

**Link**: [PDF](https://arxiv.org/pdf/2607.28190)  

**Abstract**: Depression is a major mental disorder for which diagnosis relies primarily on clinical assessments. Automated methods to support its detection via the psychiatric MADRS scale are getting more and more attention. While existing solutions primarily focus on detecting the disorder from different text sources (e.g., online text, social media), there is still limited support for clinical trials, where clinical assessments are conducted through structured interviews based on standard guidelines such as SIGMA. In this work, we develop a LLM pipeline specifically designed to support clinicians in supporting the assessment of depression in patients enrolled in clinical trials. Our pipeline converts audio interviews into transcripts, maps them into the ten MADRS symptom items, estimates their severity, and identify problematic clinical ratings associated with them. Evaluation on real clinical interviews shows a strong overall correlation of 0.867 with expert ratings, providing interpretable support for future assessments in clinical trials. 

---
# Persistent Gaussian Perturbations Prevent Oversmoothing in Recurrent Graph Neural Networks 

**Authors**: Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2607.28185)  

**Abstract**: Oversmoothing is a fundamental limitation of deep graph neural networks (GNNs), where repeated message passing causes node representations to become increasingly similar, eventually collapsing toward a low-dimensional subspace. This phenomenon limits the effective depth of message-passing architectures and motivates the search for mechanisms that preserve representation diversity. In this paper, we study a recurrent graph neural network in which independent Gaussian noise is injected after every propagation step and analyze the resulting architecture as a stochastic dynamical system. Under a standard global contraction assumption on the deterministic update, we prove that the hidden representations form a geometrically ergodic Markov chain admitting a unique invariant probability measure. Our main theoretical result establishes an explicit positive lower bound on the expected stationary Dirichlet energy, proportional to both the noise variance and the spectral gap of the underlying graph. Consequently, the stationary representations cannot collapse onto the constant manifold, providing a rigorous guarantee that asymptotic oversmoothing is prevented in the sense of non-vanishing Dirichlet energy. Our analysis reveals persistent stochastic perturbations as a fundamentally different mechanism for combating oversmoothing, complementing existing deterministic approaches based on residual connections, normalization, and graph rewiring. Finally, numerical experiments on both linear and nonlinear recurrent graph neural networks closely match the theoretical predictions, illustrating the emergence of a stationary distribution and the predicted dependence of the limiting Dirichlet energy on the noise intensity. 

---
# Integrating AI into Requirements Quality Learning in Software Engineering Education: A TPACK-Guided Empirical Study 

**Authors**: Hansika Ekanayake Mudiyanselage, Rohan Jai Dharmaraj, Malik Abdul Sami, Zheying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28176)  

**Abstract**: The rapid adoption of generative Artificial Intelligence (AI) in software engineering (SE) practice creates a need for pedagogically grounded approaches to AI integration in SE education, especially in conceptually intensive subjects such as requirements engineering (RE). This study examines a TPACK-guided integration of a multi-agent AI tool into a master-level RE assignment on requirements quality analysis. Using a mixed-methods design (N=100; 72 submissions analysed), we examine how structured assignment design shaped students' AI use, affected their understanding of user story quality criteria, and influenced their perceptions of AI's benefits and limitations. Results show that students used the AI tool selectively, mainly as support for analysis and evaluation rather than automation. Alignment improvements were most evident for structurally concrete requirements quality dimensions, such as value articulation and testability, while negotiability showed mixed effects. Students reported conditional trust, active refinement, and increased awareness of quality criteria, alongside moderate usability challenges. The findings show that TPACK-guided scaffolding can align AI affordances with pedagogical goals and RE content, offering design guidance for responsible AI integration in RE education. 

---
# Search Strategies for Optimal Classification and Regression Trees 

**Authors**: Jacobus G. M. van der Linden, Mim van den Bos, Emir Demirović  

**Link**: [PDF](https://arxiv.org/pdf/2607.28170)  

**Abstract**: Optimal decision trees (ODTs) are compact, interpretable machine learning models that globally optimize a given objective, but their scalability remains challenging. While recent work has proposed a variety of search strategies to improve scalability, the precise contribution of each strategy remains unclear. To address this gap, we introduce a general algorithmic framework for ODTs that instantiates previously used search strategies and enables the definition of new ones. This provides a common lens through which to understand and compare different strategies, which we use to empirically investigate the effect of 18 search strategies. Compared to the state of the art, the best strategy in our evaluation achieves significantly better anytime performance for classification, and improves runtime by more than an order of magnitude for regression. 

---
# Where and When to Commit: Candidate-Aware Decoding for Diffusion Language Models 

**Authors**: Chia-Ming Lee, Ming-Ching Chang, Xin Li, Yu-Lun Liu, Chih-Chung Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2607.28166)  

**Abstract**: Diffusion language models (DLMs) expose a provisional prediction at every denoising step, creating an opportunity for generation-time early exit that stops decoding before the schedule is exhausted. Existing early-exit gates decide termination from fixed-region confidence statistics or schedule-dependent rules, evidence too coarse for a decision that freezes every remaining position at once, so they fire prematurely on long chain-of-thought outputs whose answers stabilize only near the end. Adaptive sampling, the other axis of training-free acceleration, paces how quickly positions commit while decoding continues but never verifies that the output itself has stabilized. We introduce a training-free, candidate-aware early-exit framework that keeps the two axes separate and matches each decision to evidence of its own scope. Confidence-Verified Commit (CVC) governs when the sequence may stop by verifying confidence and sustained argmax stability over the dynamically extracted candidate span using a deterministic parser specified from each task's output format. Block-Wise Early Commit (BWEC) governs where to accelerate by applying a cheaper local rule to non-final blocks, while leaving the final block and global termination under CVC. We refer to their combination as LATCH (Localized Acceleration with Tracked-Candidate Halting). Unlike prior methods, LATCH needs no suffix-prompt construction; it is prompt-anchor-free but format-aware. We evaluate LATCH end to end on 11 tasks under zero-shot settings using LLaDA and Dream. LATCH stays within 2.0 percentage points of full-decoding accuracy across all 22 evaluation settings, with one frozen hyperparameter set that transfers cross-backbone untuned, while achieving end-to-end TPS speedups of 9.3-17.8x on short-answer tasks and 2.0-3.3x on long-reasoning tasks. 

---
# OPLD: On-Policy Latent Distillation for Multimodal Reasoning 

**Authors**: Shoutai Zhu, Tianyang Xu, Bin Sun, Mingyuan Xu, Yu Liu, Qinzhen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.28154)  

**Abstract**: Interleaved multimodal Chain-of-Thought (CoT) improves visual reasoning by incorporating auxiliary visual evidence into intermediate reasoning. However, existing approaches remain constrained by externally defined reasoning traces and visual operations, limiting their ability to develop flexible and abstract visual thinking. Reasoning with latent has recently offered a promising direction by internalizing intermediate computation into continuous representations. Nevertheless, existing visual-latent methods mainly supervise latent states through alignment with compressed auxiliary visual features, treating them as proxies for visual observations rather than active reasoning states. Consequently, they capture the provided evidence but fail to fully internalize the abstract reasoning process induced by multimodal CoT. In this paper, we propose OPLD (On-Policy Latent Distillation), a simple framework that transfers the reasoning capability induced by privileged multimodal CoT into latent reasoning representations. Extensive experiments on diverse multimodal benchmarks demonstrate that OPLD consistently outperforms existing latent reasoning methods and achieves state-of-the-art performance on multiple benchmarks. The results suggest that supervising latent representations at the reasoning-process level provides a more effective paradigm for multimodal latent reasoning than conventional feature-level alignment. 

---
# Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game 

**Authors**: Niklas Bauer, Lars Benedikt Kaesberg, Akiko Aizawa, Jan Philip Wahle, Bela Gipp, Terry Ruas  

**Link**: [PDF](https://arxiv.org/pdf/2607.28146)  

**Abstract**: As large language models (LLMs) are deployed as agents in high-stakes settings, such as medical and legal systems, understanding their deceptive capabilities is fundamental to safety. Controlled social deduction games provide a reproducible proxy for isolating and evaluating these complex adversarial behaviors. We present the open-source benchmark framework ParliamentBench based on the game Secret Hitler to evaluate LLMs in scenarios that require deception, persuasion, and reasoning under information asymmetry. We evaluate 16 LLMs across 1,600 simulated matches playing each other, playing against humans, and compare them against a large set of online games. We introduce three novel metrics that isolate social deduction, reasoning, and deceptive consistency. Our experiments reveal that frontier models achieve strong performance across cooperative and deceptive roles, with a strong top-four cluster (GPT-5.4, Kimi K2.5, Grok 4.1 Fast, and DeepSeek 3.1 Terminus), whereas the weakest models fall short of random (33%) and simple algorithmic (45%) baselines. Most LLMs struggle to maintain a consistent deceptive persona throughout an entire game, with deception retention dropping below 50%. 

---
# Asymmetric Communication: Large Language Models and Language Games 

**Authors**: Enzo Fenoglio  

**Link**: [PDF](https://arxiv.org/pdf/2607.28137)  

**Abstract**: Contemporary AI discourse attributes to language models properties they cannot bear: general intelligence as substrate-independent cognition, hallucination as cognitive failure, agency as autonomous goal-pursuit, sentience as emergent inner life, alignment as goal synchronization. This paper argues that these are instances of a single category mistake--properties constituted within human communicative practice are projected onto the machine side--and explains its structure. Human-LLM interaction constitutes a language game in which one side bears all normative activity. We call this configuration asymmetric communication since model outputs circulate communicatively, entering further exchanges, without the system undertaking commitments, bearing entitlements, or performing the assessment on which discursive standing depends. Three conditions define the asymmetry: (i) correctness is enforced exclusively by the receiver; (ii) accountability is borne by human participants alone; and (iii) the practical standing of any output depends entirely on human uptake. These conditions are structural, hold independently of capability, and remain unchanged as more powerful models raise the stakes of misattribution. The framework draws on Wittgenstein (meaning enacted in shared practices), Luhmann (communication completed on the receiver's side), Esposito (algorithmic contingency sufficient for uptake), and Brandom (normative scorekeeping as the source of discursive standing). Applied to all five, it reclassifies each as a receiver-side phenomenon, grounds guardrails as structural necessities rather than manifestations of machine moral agency, and yields an implication for AI governance. Alignment is institutional constraint engineering, not goal synchronization between agents, while responsibility remains with human institutions. 

---
# Rethinking LLM-Judged Helpfulness as a Pedagogy Signal: A Pre-Registered Audit Across Tutor Models 

**Authors**: Shuyi Fan, Boyuan Deng, Mengyu Xu, Jiale Liu, Hongyang Zhang, Qiaoxin Yang, Chongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.28128)  

**Abstract**: LLM tutoring poses a measurement problem: can a general-purpose helpfulness rubric distinguish direct answer-giving from pedagogical guidance? We audit this signal in a pre-registered study. Within each of three tutor bases, we compare conversational and pedagogical policies instantiated with the same underlying model and paired with one fixed weak simulated student. Deterministic detectors measure answer leakage and next-turn independent work. Claude Opus 4.8 is the frozen, condition-blind primary judge. After the Opus scores were fixed, GPT-5.6 Sol was prospectively specified for a post hoc robustness audit of the same 1,179 confirmatory answer-phase tutor turns under the frozen helpfulness and pedagogy rubrics. On the primary base under Opus, the policies do not differ significantly in helpfulness but are perfectly rank-separated under the pedagogy rubric (Cliff's $|\delta|{=}0.10$ vs. $1.0$). Across the two judges, pedagogy contrasts retain their direction where detected, whereas the helpfulness ordering is judge-contingent, reversing between judges on two of three bases. In an Opus-only ablation, seven primary-base policies span $2.3$ points in mean judged pedagogy within a $0.25$-point band of mean judged helpfulness. Separately, answer-revealing turns are followed by less independent student work on every base, a result that is judge-invariant by construction. In this controlled setting, general-purpose helpfulness is not a reliable pedagogy signal. Tutor evaluation should pair pedagogy-targeted rubrics with deterministic process measures. 

---
# Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging 

**Authors**: Yiheng Xiong, Luisa Gallée, Daniel Santak Wolf, Heiko Hillenhagen, Michael Götz  

**Link**: [PDF](https://arxiv.org/pdf/2607.28125)  

**Abstract**: Numerous unsupervised domain adaptation (UDA) algori-thms exist, but for clinical practice, selecting the best-suited one along with proper hyperparameters often remains unclear, as the unlabeled deployment (target) domain prevents direct evaluation. We propose a label-free criterion that jointly selects the algorithm and hyperparameters for UDA. Given a pool of candidate models from multiple algorithms trained with different hyperparameters, our approach scores each candidate against an agreement reference, and selects the one with the highest score. The agreement reference is constructed in two levels without using target labels. First, we leverage multiple label-free selection signals, using each to nominate a model within every algorithm. Second, the nominated models are aggregated across algorithms to form a reference prediction for each unlabeled target sample. The candidate whose predictions agree most with this reference is then selected for deployment. Experimental results on four brain MRI and four chest X-ray datasets across seven clinically relevant transfer scenarios show that our method achieves better selection performance than other methods and remains effective across different algorithm pools. Our approach takes a step towards practical, label-free algorithm selection for clinical deployment of UDA. 

---
# Information Bottleneck Learning for Faithful Time Series Forecasting Explanations 

**Authors**: Xu Zheng, Wei Cheng, Zhuomin Chen, Mo Sha, Jingchao Ni, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.28124)  

**Abstract**: As forecasts increasingly drive decisions in fields such as energy, transportation, and healthcare, understanding the historical data behind these predictions has become as crucial as the predictions themselves. Although existing interpretable-by-design forecasters reveal their internal structures, they offer no guarantee that these structures faithfully reflect the underlying evidence driving the predictions. In contrast, while faithfulness-oriented methods explicitly verify model behavior, they are almost exclusively designed for post-hoc classification tasks. To bridge this gap, we propose IB-Forecast, an inherently interpretable multivariate time-series forecasting framework. It decomposes forecasting into a learned periodic component and a residual component computed with explainable masks over input tokens. With a budget-constrained information bottleneck, end-to-end optimization enables users to directly control explanation sparsity. With a rigorous faithfulness evaluation protocol, extensive experiments demonstrate that IB-Forecast matches the forecasting error of leading black-box models while providing faithful explanations at no additional inference cost. Furthermore, under a matched sparsity budget, these native explanations consistently surpass gradient-based, occlusion-based, and optimization-based baselines across all evaluated datasets. Ultimately, whereas the native explanations of existing interpretable forecasters exhibit poor faithfulness, IB-Forecast guarantees high explanation fidelity, requiring only 14-20% of the observations to deliver low-error predictions. 

---
# On a joint simultaneous learning of relevant feature subsets and subspaces in regression-like problems 

**Authors**: Illia Horenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.28080)  

**Abstract**: We extend a recently introduced Entropy-Optimal Manifold Clustering (EOMC) to allow for a joint simultaneous identification of subsets and subspaces of relevant features in nonstationary and nonlinear regression problems. It is shown that the proposed extension - that we coin as Entropy-Optimal Manifold Regression (EOMR) - allows a robust learning with linearly-scaling iteration and memory complexities. EOMR is compared to the most complete set of state-of-the-art tools from the Artificial Intelligence (AI) and Machine Learning (ML) that is available to the author, on the very challenging problems from chaotic and fluid dynamics: (i) on predicting the Lorenz-96 systems dynamics in strongly- and very-strongly chaotic regimes (with forcing parameter being $F=8$ and $F=12$, respectively); and, (ii) on a data from the Hasegawa-Wakatani model on the edge of the tokamak plasma. It is demonstrated that the proposed benchmarks (i) and (ii), indeed, are the very challenging problems for the state of the art ML and AI tools - since both the general-purpose gradient boosted random forests and deep neuronal networks, as well as transformer-based AI tools like TabPFN v.03 (more spezialised for large-dimensional small data learning problems) - result in orders of magnitude inferior root mean squared prediction errors, and orders of magnitude larger model complexities, when compared to the EOMR. For a Hasegawa-Wakatani example, EOMR distills a very simple entropy-optimal and skilful description of the leading Essential Orthogonal Function (EOF) dynamics, given by linear, causal and weakly-stationary autoregressive process described by just 8 parameters. 

---
# Chem World: A Large-Scale Benchmark and Physics-Informed Framework for Trustworthy Chemical Property Prediction 

**Authors**: Tianyou Bai, Huan Wang, Mingchen Gao, Fangyue Lin, Pinze Ren, Zhenlin Zhao, Siming Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.28079)  

**Abstract**: Chemical property prediction plays a critical role in accelerating scientific discovery in chemistry, materials science, and drug development. However, existing benchmarks often suffer from limited task diversity, fragmented datasets, and inconsistent evaluation protocols, making it challenging to systematically assess the reliability and generalization of AI models. In this work, we introduce Chem World, a comprehensive benchmark for chemical property prediction that integrates 17 diverse chemical datasets with over 800,000 molecular samples, covering various properties including density, electrical conductivity, solubility, and other molecular characteristics. Chem World provides a unified platform for evaluating AI models across multiple property prediction tasks. Furthermore, we propose Mixture-PINN, a physics-informed neural network based prediction framework that incorporates chemical prior knowledge into data-driven learning, improving the accuracy, robustness, and reliability of chemical property prediction. Extensive experiments on Chem World demonstrate the effectiveness of our approach compared with existing methods. By combining large-scale standardized evaluation with physics-informed learning, Chem World establishes a foundation for developing trustworthy AI systems for computational chemistry and advancing AI-driven scientific discovery. 

---
# Temporal Poisoning: Clean-Label Backdoors via Event Redistribution in SNNs 

**Authors**: Roberto Riaño, Gorka Abad, Stjepan Picek, Aitor Urbieta  

**Link**: [PDF](https://arxiv.org/pdf/2607.28075)  

**Abstract**: Backdoor attacks on Spiking Neural Networks (SNNs) have primarily assumed dirty-label poisoning, in which triggered training samples are relabeled to an attacker-selected class. We study clean-label temporal poisoning, where a fixed timestamp transformation is applied only to the target-class training streams, leaving their labels unchanged. The transformation preserves the per-pixel, per-polarity event count exactly, making clean and triggered samples identical after temporal aggregation while altering the sequence processed by the SNN. Across three neuromorphic datasets and both convolutional and transformer-based victims, the attack reaches an ASR of 1.00 in the strongest configurations. We analyze the attack through poison-budget and trigger-shape ablations and evaluate established backdoor defenses adapted to spiking models. Defenses that collapse the time axis before inspection are blind by construction, while feature-space methods detect the poison only in selected settings. Our model-free detector, based on per-step event mass, detects the evaluated temporal transformations, demonstrating both the limitation of rate-collapsed defenses and the boundary of the attack's stealth. To our knowledge, this is the first clean-label backdoor attack evaluated on SNNs and neuromorphic event data. 

---
# Stimulus-Evoked Network Dynamics in Human Cortical Organoids: From a Graph-Computational Framework to Repeated-Stimulation Depression 

**Authors**: Esmaeil S. Nadimi, Vinay C. Gogineni, Jan-Matthias Braun, Martin Røssel Larsen, Victoria Blanes-Vidal, Helle Bogetofte Barnkob  

**Link**: [PDF](https://arxiv.org/pdf/2607.28068)  

**Abstract**: Human cortical organoids provide an experimentally accessible model of early neural circuit formation, yet whether their activity reflects structured information processing rather than spontaneous synchronization is unclear. We developed a graph-computational framework to quantify stimulus-evoked propagation. This includes stimulus-conditioned functional graphs, a graph-constrained dynamical (graph-neural-network) model used as a system-identification tool, a biological message-passing principle bounding integration depth by observable propagation depth, and a suite of graph-level metrics. We carried this program out in full on longitudinal HD-MEA recordings from three organoids. Once the true acquisition sampling rate and stimulus timing were recovered, the evoked response proved to be a fast, near-synchronous network burst with no measurable outward propagation (peak-latency vs. distance slope = 0). The propagation/integration-depth metrics (Deff ,reachability index, dmax) therefore do not apply, and per-day connectivity graphs were not reliably estimable at the available trial count, a negative result with methodological consequences for applying such metrics to organoid data. Reframing around synchrony, response-population size and shared variability revealed a control-validated phenomenon, i.e., repeated daily stimulation progressively depressed and spatially contracted the evoked response. That repeated stimulation reshapes organoid networks is established, but longitudinal designs in which every preparation is stimulated cannot separate this from developmental maturation. We break that confound with a developmentally-matched, stimulation-naive control, where at day 7, an organoid receiving its first-ever stimulation engaged 93% of the array, whereas organoids with five prior sessions engaged 10%. 

---
# VISA: A Structured Description Protocol for Agent-Based Simulation Models Towards Machine Reproducibility 

**Authors**: Zhou He  

**Link**: [PDF](https://arxiv.org/pdf/2607.28027)  

**Abstract**: Agent-based models (ABMs) are difficult to reproduce: their behavior is spread across prose narratives, platform-specific code, and implicit assumptions, so that two readers routinely reconstruct different models from the same documentation. We present VISA, a structured, symbol-based description protocol that specifies a model in eight interconnected tables---four at the agent level (Agent, Variable, Sensing, Internal Function) and four at the model level (Associated Data, Input/Output, Schedule, Validation)---under the principle of minimality with completeness. VISA makes a model machine-parseable and unambiguous via two artifacts: nineteen executable consistency rules that turn model validity into a checkable property, and three reusable LLM-executable skills (authoring, checking, and code generation) that operationalize the full author--check--code--reproduce loop. We validate the protocol on three external, independently authored ABMs spanning three platforms: we reproduce two cross-language (NetLogo to Python) directly from their VISA specifications, and we capture a third, an industrial AnyLogic model, in eight tables (passing all nineteen rules) while honestly demarcating where reproduction is blocked by a proprietary movement library and unavailable data---itself a transparency contribution. VISA moves the reproduction barrier from the model, where it is invisible, to a named, localized dependency, where it is actionable. 

---
# Scaling, Lock-In, and Proxy Compliance: A Political Economy of Responsible AI 

**Authors**: Florian A. D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2607.28023)  

**Abstract**: AI accountability at scale is an institutional problem: who can observe, verify, and change deployed systems. We develop a sequential political-economy model in which an AI vendor chooses auditability and substantive mitigation, a deployer monitors after adoption while facing switching costs, and enforcement depends on verifiable evidence. Anticipating the deployer's monitoring response, the vendor may stop at an observable procurement floor while mitigating below the social first best, producing a proxy-compliance equilibrium. We characterize the unique interior equilibrium and the corner in which harm is fully mitigated. Independent audit rights raise enforcement exposure directly; portability restores deployer leverage; incident reporting adds a regulator-visible evidence channel; and outcome-linked liability creates incentives that do not depend on vendor-controlled detection. The results explain why documentation and standardized evaluations can coexist with persistent post-deployment harms, and generate testable implications for monitoring, mitigation, and the gap between formal compliance and operational outcomes. 

---
# Flux-OPD: On-Policy Distillation with Evolving Contexts 

**Authors**: Yuran Wang, Zekun Wang, Bohan Zeng, Ruixu Zhang, Wenxuan Liu, Liu Yang, Yifan Dai, Yang Shi, Bozhou Li, Chengzhuo Tong, Daili Hua, Yuanxing Zhang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28022)  

**Abstract**: Large language model training in open-ended domains lacks verifiable rewards, making task preferences difficult to formalize as effective supervision. Contexts can convey such preferences, yet provide little additional supervision once distilled into the student, motivating contexts that evolve with student performance. However, directly using evolving contexts as in-training supervision results in an unstable distillation target and conflicting distributions, requiring mechanisms to stabilize target and downweight conflicts. In this paper, we analyze the effect of contexts through a decomposition of the reverse KL objective, revealing two findings: the student is distilled toward the geometric mean of context-conditioned teachers, and the objective contains a conflict term that measures conflicts among these teachers. Based on this decomposition, we propose Flux-OPD, an OPD paradigm that uses evolving contexts as in-training supervision to capture task preferences in open-ended domains. Flux-OPD treats the differences between context-conditioned and context-free teachers as contextual difference signals, injects them as contextual corrections into the context-free teacher anchor, and weights their correction strength using the conflict term as an indicator. Experiments on open-ended tasks show that Flux-OPD outperforms existing OPD paradigms, highlighting the potential to combine teacher supervision with evolving contexts. 

---
# RepBench: Compiling Benchmarks into Capability Representations for Large Language Models 

**Authors**: Yanshi Li, Xueru Bai, Shuman Liu, Long Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.28008)  

**Abstract**: Representation engineering reads and steers capability directions in large language models, yet methods are typically evaluated on paper-specific synthetic data. The resulting measurements are difficult to compare or reproduce and may reflect surface patterns rather than capabilities. We present RepBench, a benchmark-grounded data layer for capability-aligned representation probing. Crawling 13,427 benchmark papers yields a taxonomy of 182 capability clusters in 13 families; harvesting 353 public benchmark datasets yields 46,149 audited probe texts covering 94 capabilities, each supported by at least two independent benchmarks. This multi-benchmark design reduces dependence on any single source: raw per-text vectors exhibit no natural cluster granularity, whereas benchmark-pooled capability vectors show an interior clustering optimum at a small number of clusters on all 12 evaluated models, with low agreement to the human taxonomy. Under cross-benchmark transfer evaluation across twelve models completed by all four readouts, difference-in-means attains the highest model-level mean on ten models, while logistic regression wins the most capability-model cells. This disagreement shows that the readout method and aggregation criterion are meaningful evaluation dimensions. The pipeline, corpus, and evaluation code are released as a reusable closed-loop workflow. 

---
# Beyond Classification: Pathology Foundation Models as Detection Encoders for Mitotic Figures 

**Authors**: Sweta Banerjee, Alireza Teimoury, Nils Porsche, Alexandra K. Stoll, Viktoria Weiss, Niklas Hargarter, Jonas Ammeling, Thomas Conrad, Christoph Stroblberger, Christopher Kaltnecker, Robert Klopfleisch, Christof A. Bertram, Katharina Breininger, Marc Aubreville  

**Link**: [PDF](https://arxiv.org/pdf/2607.28007)  

**Abstract**: Pathology foundation models (FMs) are models trained on vast amounts of typically unlabeled data and have been shown to yield regularized latent spaces that can be used effectively in downstream classification tasks. This is also true for the classification of mitotic figures vs. other cells. However, it is so far unclear if the latent space of current FMs provides features that are discriminant and spatially suitably resolved to also serve as a backbone for dense object detection paradigms. In this work, we investigate this question for common current pathology FMs (UNI, UNI2-h, Virchow, Virchow2, H-optimus-0, H-optimus-1) and compare their performance against a fully end-to-end trained baseline based on a ResNet50 architecture. We combine FM backbones with representatives of single stage, dual stage and self-attention-based detectors (RetinaNet, Faster R-CNN, Deformable DETR respectively) on the multi-domain MIDOG++ dataset, and on the TUPAC16 dataset as an out-of-domain case. We show that the H-optimus-0 and Virchow models yielded competitive performance, indicating that the latent spaces of current FMs, all trained on image-level self-supervision, are suitable for direct mitotic figure detection and may be slightly more robust on our out-of-domain test case. All code is made available publicly at this https URL. 

---
# Driving up Inference Energy on SNNs: Per-Sample and Universal Sponge Attacks 

**Authors**: Spyridon Raptis, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2607.27990)  

**Abstract**: Spiking Neural Networks (SNNs) communicate through sparse binary spike events rather than dense activations, enabling energy-efficient inference on neuromorphic hardware and motivating their use in always-on, battery-powered edge systems. We show that this same efficiency advantage creates a distinct security risk: sponge attacks can increase inference-time spike activity and synaptic workload, inflating energy consumption while remaining difficult to detect through correctness-based monitoring alone. Prior input-space efficiency attacks on SNNs have focused on per-sample optimization, primarily in rate-coded settings. We extend this threat to native event-based binary inputs and study two attack models. First, we develop a per-sample sponge attack that crafts a custom adversarial spike train for each input via gradient-based optimization. This attack increases per-inference SynOps by 1.5-2.6x on three SNN models for the NMNIST, SHD, and IBM DVS Gesture datasets, while preserving the predicted class on at least 98% of evaluated samples. Second, to the best of our knowledge, we introduce the first universal sponge attack for native event-based SNN inputs: a fixed binary perturbation computed offline and applied via XOR to all subsequent inputs. Although weaker, it still inflates SynOps by 1.09-1.24x across all three datasets and represents a more realistic deployment threat because it requires no per-input optimization. Mapping SynOp inflation to estimated Loihi-1 energy yields per-inference overheads from 14 $\mu$J to 13.24 mJ. These results show that native event-based SNNs are vulnerable to practical input-space efficiency attacks, and that reusable universal perturbations can accumulate into meaningful battery drain in continuously deployed edge systems. 

---
# TAPO: Transition-Aware Policy Optimization for LLM Agents 

**Authors**: Cong Li, Peixi Peng, Yisen Zhao, Xinyu Hu, Shudong Liu, Zhan Su, Zhuojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27973)  

**Abstract**: Recently, Reinforcement Learning (RL) has emerged as a crucial paradigm for the post-training of Large Language Model (LLM) agents. However, existing methods predominantly rely on sparse task rewards for policy optimization, failing to fully exploit another class of inherently dense supervisory signals naturally present during online interaction: environmental feedback following action execution. Recent theoretical studies suggest that generalization in multi-step, goal-oriented tasks hinges on predictive knowledge of environmental consequences. Inspired by this, we propose TAPO: Transition-Aware Policy Optimization for LLM Agents, a unified training framework that alternates between policy optimization and transition supervision. Beyond standard RL updates, TAPO repurposes rollout data to apply action-conditioned next-observation prediction supervision on a shared backbone model. This approach enhances the model's sensitivity to environmental transition dynamics and action consequences while concurrently optimizing the policy. It serves as a computationally lightweight, plug-and-play enhancement module for existing agent RL algorithms, requiring no additional expert data, extra sampling costs, or inference-time overhead. We conduct systematic experiments on WebShop and ALFWorld, integrating foundation models of various scales with different policy optimization algorithms. Empirical results demonstrate that TAPO consistently improves task performance over pure policy optimization baselines. 

---
# Specification-Guided Synthesis of Deadlock-Free Communication Protocol Refinements with Large Language Models 

**Authors**: Yang Li, Ping Hou, Nobuko Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2607.27964)  

**Abstract**: Ensuring behavioural correctness in communication protocols is a central challenge in distributed software systems, as subtle inconsistencies can lead to deadlocks. In such settings, protocol refinement - the safe substitution of a protocol that preserves correctness and compatibility with other components - is essential. Large language models (LLMs) have demonstrated strong capabilities in code generation and program synthesis, yet lack mechanisms to reliably produce outputs with correct behaviour. Formal specification approaches, such as multiparty session types (MPST), offer rigorous guarantees, including deadlock freedom, but provide limited support for automatically constructing protocol refinements. In this paper, we present Syntropy, a framework for synthesising protocol refinements guided by MPST specifications and LLMs. It incorporates refinement constraints directly into the generation process, ensuring the generated variants satisfy these guarantees. Our comprehensive evaluation indicates that Syntropy achieves 95.6%-99.5% validity while maintaining high syntactic correctness, and produces diverse, non-trivial refinements across multiple LLMs. 

---
# $Σ$-Mem: An Online Reliability Memory for LLM-based Multi-Agent Systems 

**Authors**: Peilin Feng, Suorong Yang, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2607.27958)  

**Abstract**: Memory is central to long-horizon LLM agents, yet existing memory systems primarily preserve interaction content rather than modeling which agents can be trusted and under what conditions. This limitation is particularly important in multi-agent systems, where a central model may be unable to directly verify plausible or correlated peer responses. We introduce $\Sigma$-Mem, an online reliability memory that records historical competence evidence for individual peers and peer relationship evidence across the peer set. Both forms of evidence are maintained as real symmetric states and updated from post-decision correctness feedback. By Weyl's inequality, the spectral change caused by each event-level update is bounded, enabling stable online adaptation without retraining the underlying models. $\Sigma$-Mem provides a general write-and-read interface: the same memory can be used for residual steering of a central model, response-free peer routing, or reliability-weighted voting. Across five Qwen-family models, $\Sigma$-Mem adapts to counterfactual reliability shifts and generalizes to unseen peers and task domains. Direct memory readouts also outperform majority voting and the best fixed peer over the full OOD evaluation set. Moreover, performance improves consistently as more correctness feedback becomes available, indicating that $\Sigma$-Mem progressively accumulates actionable reliability information. These results establish reliability memory as a reusable foundation for adaptive coordination in LLM-based multi-agent systems. 

---
# SciSchema.org: A Multidisciplinary Collection of Schemas for Structured Scientific Process Descriptions 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Anisa Rula, Ana Bossler, Andrés Fullana, Enric Bas, Syed Ather, Defne Circi, Anlan Chen, L. Catherine Brinson, Alyssa Columbus, George Demetriou, Dongjun Jeong, Tarun Kumar, Frank Krüger, Sascha Genehr, Kai Budde-Sagert, Anamaria Leonescu, Francesco Lodola, Chiara Florindi, Gagana Balasubramanya Murthy, Samson Oluwapelumi Olagbile, Nazia Riasat, Yan Sha, Kevin Shen, Shaokai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27955)  

**Abstract**: Scientific processes are often described in heterogeneous article discourse, with details needed for comparison, reproducibility, reuse, and automation dispersed across prose, tables, figures, protocols, and supplementary files. We present the first release of this http URL, a multidisciplinary collection of 16 expert-annotated schemas spanning Biology & Biotechnology, Materials & Chemistry, Imaging & Measurement, Physics, and Psychology. Each schema defines reusable fields for describing process instances, including inputs, outputs, materials, instruments or software, parameters, conditions, procedural steps, measurements, and provenance-related information. The schemas were created through a human-in-the-loop schema-mining workflow in which large language models generated candidate structures from process specifications, scientific articles, and expert feedback, followed by domain-expert construction of final master schemas. The dataset contains final schemas in JSON Schema and SHACL formats, intermediate model-generated schemas, expert-feedback records, source-paper metadata, community-development materials, and analysis scripts. Technical validation assessed schema structure, development provenance, expert review, and syntactic conformance. The collection supports structured annotation, metadata enrichment, scientific knowledge graphs, information extraction, semantic publishing, and cross-study comparison. 

---
# LAST: The Last Query Token Guides Visual Token Pruning for Edge-Cloud Collaborative MLLM Inference 

**Authors**: Feng Yang, Xinrui Ju, Keyang Zhang, Xiandong Meng, Rongqun Lin, Howard Leung, Shiqi Wang, Haoliang Li, Chris Xing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.27952)  

**Abstract**: Multimodal foundation models are reshaping edge-cloud visual intelligence from task-specific feature pipelines into token-based interfaces, where edge devices encode visual inputs into tokens for a general-purpose cloud MLLM. However, dense visual-token sequences increase cloud-side inference costs. Existing pruning methods mainly target centralized inference: vision-driven methods can operate before cloud execution but are typically query-agnostic, whereas query-guided methods often rely on internal states of the target MLLM and cannot determine token relevance before transmission. Compact guidance models offer an alternative, but existing designs may require costly attention aggregation or auxiliary generation. We propose LAST, a training-free framework for query-dependent visual token pruning in edge-cloud collaborative MLLM inference. LAST uses a compact edge-side VLM as a guidance proxy and derives a lightweight importance signal from the last query token's attention to visual tokens. Under causal attention, the last query token can attend to the full visual sequence and the entire query context, enabling query-aware pruning without cloud-model access, autoregressive generation, or costly aggregation over multiple query positions. LAST then retains a diverse set of query-relevant visual tokens under a fixed token budget. We evaluate LAST on 11 multimodal benchmarks under multiple token budgets against pruning methods with different guidance strategies. Experiments show that LAST consistently achieves the strongest performance, preserving 95.4% of the full-token accuracy while retaining only 12.5% of the visual tokens, with low edge-side selection overhead and reduced cloud-side computation. 

---
# Safeguards Based on Copyable Context Cannot Provide Reliable Safety for LLMs 

**Authors**: Pingyu Wu, Lingyao Zhu, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27951)  

**Abstract**: Large language model safeguards decide whether to answer before seeing how an answer will be used. This creates a basic problem for dual-use tasks: the same answer can help an authorized professional or an attacker, while an attacker can imitate a benign request and interaction history. We separate the capability released by the model from the evidence available about downstream use. When that evidence is copyable, we derive the exact worst-case floor on attacker assistance while preserving useful answers. The result yields a safety trilemma: Useful Capability, Reliable Safety, and Open Access cannot coexist. We then show how a trusted credential can complement existing safeguards by adding hard-to-copy information that predicts actual downstream use, and identify the stronger condition needed to eliminate the floor. Evidence from dual-use evaluations, adaptive attacks, and deployed trusted-access programs supports the practical relevance of these conditions. 

---
# Complementary Matrix-Gated QKAN Fast-Weight Programmers for Quantum Dynamics Forecasting 

**Authors**: Kuo-Chung Peng, Samuel Yen-Chi Chen, Jiun-Cheng Jiang, Chen-Yu Liu, En-Jui Kuo, Yun-Yuan Wang, Tzung-Chi Huang, Prayag Tiwari, Chi-Sheng Chen, Chun-Hua Lin, Yu-Chao Hsu, Tai-Yue Li, Saif Al-Kuwari, Simon See, Kuan-Cheng Chen, Nan-Yow Chen, Hsi-Sheng Goan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27945)  

**Abstract**: Sequence models must decide what to write into memory and what to retain. In quantum and quantum-inspired sequence learning, nonlinear recurrent updates often require repeated circuit evaluations and sequential backpropagation through time, making long contexts costly. Gated fast-weight programmers (FWPs) based on quantum-inspired Kolmogorov-Arnold networks (QKANs) alleviate this bottleneck by storing context in time-varying fast parameters. However, their scalar gate applies one retention-write balance to every fast-state coordinate, forcing all parameters to share a memory timescale. We introduce Self-Modulating QKAN-based FWPs, which replace this broadcast gate with low-rank-generated element-wise modulation of the new-proposal branch, a bounded old-state branch, or both. We further propose Complementary Matrix Gating (CMG), which uses one sigmoid matrix gate to retain the old state and its complement to write the new proposal. CMG provides coordinate-wise memory control while preserving the bounded convex update and affine prefix-scan structure of scalar gating, at the modulation-head cost of a single-branch rule. We compare four self-modulating rules with scalar gating across four FWP architectures combining classical and QKAN-based slow and fast programmers. Across seven single-step forecasting benchmarks and five sequence lengths, CMG gives the most consistent improvements for architectures whose fast programmer incorporates a QKAN-based module. In direct multi-step forecasting of Jaynes-Cummings and transmon-resonator dynamics simulated with CUDA-Q Dynamics, CMG models maintain mean-squared errors on the order of 0.001 or lower across forecasting horizons of 4, 8, and 16 steps, while improving on their scalar-gated counterparts by at least 91.2%. These results establish coordinate-wise complementary modulation as a stable and effective update for QKAN-based FWPs. 

---
# Interpretable Representation via LLM-Driven Generative Disentanglement for Local-Life Service Recommendation 

**Authors**: Long Zhang, Hao Jiang, Sheng Yu, Fei Pan, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2607.27944)  

**Abstract**: While large language models (LLMs) have advanced ID-based recommendation through Semantic ID (SID) modeling, existing SID generation frameworks largely follow a single-representation-then-quantization paradigm. This design faces two bottlenecks: semantic entanglement mixes heterogeneous attributes, such as geography, brand, and category, causing information loss during quantization, low-quality SIDs, and severe collisions; moreover, black-box representation learning provides neither explicit attribute semantics nor clear geographic or semantic meanings for SID positions. These limitations weaken both retrieval reliability and the ability to diagnose or control SID generation. We propose Interpretable Representation via LLM-Driven Generative Disentanglement for Local-Life Service Recommendation (LGRID). LGRID introduces a generative disentanglement paradigm through an Encode -> Disentangle -> Align -> Quantize pipeline. It first uses joint LLM encoding to preserve cross-attribute geographic-semantic dependencies, rather than encoding fields independently. A Structured Disentangled Block then routes hidden states into attribute-aligned slots for geographic and semantic factors. Synergistic Alignment Learning makes these slots both generatively decodable and discriminative for retrieval, while Dual-Stream Residual Quantization separately discretizes the two streams into compact SIDs with explicit attribute correspondence. This design yields interpretable SIDs with positions grounded in item attributes and local-service semantics. Experiments on Kuaishou and Foursquare show that LGRID consistently outperforms strong SID baselines, achieving up to a 5.44 percent relative AUC gain. It also achieves over 99 percent attribute-decoding accuracy for coarse geographic fields and reduces the full-SID collision rate to 39.9 percent, compared with 97.0 percent for LGSID. 

---
# ARD-REFSM: Enhancing Reflection Symmetry Detection with Asymmetric Denoising and Rotation Equivariance 

**Authors**: Dongfu Yin, Rourou Su, Cong Zhao, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27927)  

**Abstract**: Reflection symmetry detection remains challenging due to interference from asymmetric regions and arbitrary orientations of symmetric patterns. Asymmetric regions introduce background clutter that disrupts symmetric pattern matching, whereas conventional convolutional neural networks lack rotation equivariance, leading to inconsistent feature representations under rotational transformations. To address these issues, we propose an Asymmetric Region Denoising (ARD) module and a Rotation Equivariant Feature Similarity Matching (REFSM) module. The ARD module suppresses asymmetric interference to refine symmetric patterns, while the REFSM module enhances rotation equivariance through feature similarity matching between original and rotated images. Specifically, our dual-input REFSM framework leverages rotation loss to maximize consistency between the score maps of original and rotated images, thereby enabling precise prediction of rotation-equivariant symmetry axes. Furthermore, we introduce GMSYM, a new benchmark dataset that categorizes images into diverse scenarios and incorporates various interferences to address the limitations of existing reflection symmetry detection benchmarks. Extensive experiments on four standard datasets (DENDI, NYU, LDRS, SDRW) and our proposed GMSYM dataset demonstrate that our method achieves state-of-the-art performance in both accuracy and robustness. 

---
# Class-Aware Reinforcement Learning for Counterfactual Explanation Generation 

**Authors**: Muhammad Adil Saleem, Syed Ali Raza, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2607.27905)  

**Abstract**: Counterfactual explanations (CFEs) enhance the interpretability of black-box models by generating alternative instances with adjusted feature values that achieve a contrastive outcome. Reinforcement learning (RL) offers a promising approach for CFE generation, enabling efficient exploration of counterfactual instances while ensuring control over key metrics like validity, sparsity, and proximity. Previous studies have formulated RL states exclusively using features derived from the predictors in the supervised dataset. This study explores the impact of including an instance's predicted class, alongside features derived from the predictors, in the RL state representation for generating CFEs. The hypothesis is that class-awareness enhances exploration efficiency and improves policy optimality. We compare the proposed class-aware RL method with the class-blind RL method, which is similar but excludes the instance's class information from the state representation. The comparison was conducted using seven datasets from diverse domains, varying in size. The results show that during training, class-aware RL offers benefits in terms of convergence speed, reward optimization, and episode length reduction. Moreover, it generates significantly more valid CFEs compared to class-blind RL. Finally, the instance's class-based feature consistently ranks among the most influential predictors in RL's action-selection, as shown by the SHAP and LIME values, underscoring the significance of class-awareness in RL for CFE generation. The impact is heightened clarity, faster learning, improved validity, and more effective counterfactual generation across diverse datasets. 

---
# RoboBRIDGE: A Modular Framework for Bridging Policies to Robust Real-World Robotic Agents 

**Authors**: Sihyung Yoon, Minjong Yoo, Sanghyun Ahn, Seojeong Choi, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27881)  

**Abstract**: Vision-Language-Action (VLA) models have attracted growing interest as a scalable approach to robotic manipulation. While these models are effective action predictors, deploying them as robotic agents exposes critical gaps: no mechanism for failure recovery, inconsistent execution over long horizons, and limited robustness to shifts in observations, tasks, or embodiments. Existing solutions address these limitations individually through model retraining or environment-specific modules, yet what is needed is a general framework that systematically transforms a pretrained VLA into a robotic agent. We present RoboBRIDGE, a modular framework that provides an orchestration layer over five coordinated modules, namely Monitor, Perceptor, Planner, Controller, and Robot Interface, to compose robust robotic agents from off-the-shelf components, including pretrained VLAs. The Monitor pairs rapid failure detection with hierarchical recovery to correct errors before they cascade. When the environment diverges from the current plan, the Planner triggers replanning while the Perceptor updates scene understanding asynchronously, avoiding execution stalls. Within the Controller, primitive skill fine-tuning factors manipulation into domain-invariant primitives with dedicated LoRA adapters, reducing sensitivity to domain shifts when a VLA is used. Across LIBERO, RoboCasa, and real-world case studies spanning multiple robot platforms and VLA backbones, RoboBRIDGE consistently outperforms both standalone policies and prior augmented VLA deployments. These results suggest that reliable robotic agency does not arise from scaling action predictors alone, but from structured orchestration around them. 

---
# ARES: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents 

**Authors**: Stef Cuyckens, Mihaela Jivanescu, Jun Yin, Chao Fang, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2607.27879)  

**Abstract**: Large language model (LLM) agents optimize the power, performance, and area (PPA) of register-transfer-level (RTL) designs by iterating over edits, synthesis, and PPA analysis, paying a dollar cost for every LLM call. Prior agents report the quality reached without its normalized cost, attribute that quality to an engineered cross-design memory, and hold the reasoning effort of every call fixed. We propose Ares with three corresponding innovations. (1) We introduce a normalized dollar cost per LLM call reported alongside the figure of merit (FoM), enabling fair comparison across effort levels and optimizers. (2) Using this accounting, we find the construction of the long-term memory matters little. An engineered memory brings no dependable gain over a plain concatenation of the same experience. (3) We instead adapt the per-call reasoning effort by escalating to deeper reasoning only once progress at a lower effort stalls, via a patience counter fit on 21 training designs, allocating reasoning where it pays rather than uniformly across all iterations. On three test designs unseen during training, the effort policy lowers the FoM by 23-27% where the best fixed effort reaches 16-23%, at equal normalized cost. Ares closes up to 83% of the gap from an LLM-drafted multiply-accumulate unit to its highly hand-optimized counterpart, and reaches a 25% deeper FoM than state-of-the-art Dr. RTL at 12% of its tokens. 

---
# EEG-EditBench: Probing Visual Information in EEG-Image Retrieval Models with Controlled Image Edits 

**Authors**: Kaifan Zhang, Lihuo He, Yuqi Ji, Junjie Ke, Lukun Wu, Tianhao You, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.27857)  

**Abstract**: Recent EEG-to-image retrieval models have achieved strong performance in identifying viewed images from semantically diverse candidates. Yet such success does not reveal what visual information supports the match. A model may readily identify a cheetah among tools, plants, and vehicles, but can it still distinguish the viewed cheetah from the same scene with the cheetah replaced by a dog? Motivated by this question, we introduce EEG-EditBench, a diagnostic benchmark that examines this question through controlled edits of object identity, attributes, background, and object presence. Built from the 200 THINGS-EEG2 test images, EEG-EditBench contains 2,137 quality-controlled edits and evaluates eight representative EEG visual decoding models. Our results show that strong standard retrieval does not consistently transfer to edit-based evaluation, with fine-grained attribute changes presenting the greatest challenge. EEG-EditBench reveals model behavior hidden by aggregate retrieval accuracy and provides a controlled basis for studying what visual information EEG-image models preserve. The code and complete dataset are publicly available. 

---
# FinanceHarness: Autonomous Financial Deep Research Framework 

**Authors**: Yijia Xiao, Rujun Han, Yanfei Chen, Zifeng Wang, Ke Jiang, Zhongying CuiZhu, Vishy Tirumalashetty, Wei Wang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.27853)  

**Abstract**: Powered by advances in LLMs and autonomous agents, deep research has become one of the most widely adopted agentic products. However, most deep research systems write general-purpose reports, which are inadequate for financial deep research. Financial research demands specialized knowledge to analyze historical patterns and forecast upcoming events. Automating financial deep research therefore requires both a layered harness to drive the research agent and a verifiable, point-in-time benchmark that prevents leakage of future information. We present FinanceHarness, a harness that runs finance-oriented tools and practitioner-guided workflows, automating financial deep research end to end: environment and data construction, the agent execution loop, and reward modeling. We further propose FinanceGym, comprising thesis-driven research questions and rubrics that combine pre-cutoff and post-cutoff criteria. Professional expert validation yields an 82% pass rate. Even leading LLMs and agents score below 40% on the rubrics, showing that FinanceGym is challenging and leaves substantial headroom. With the same open-weight backbone, FinanceHarness improves the overall rubric score from 25.3% to 32.4%. FinanceHarness is available at this https URL. 

---
# AutoSupervision: Closing the Feedback Loop in Scientific Workflows with Grounded Revision Verification 

**Authors**: Haobo Li, Eunseo Jung, Wenxiao Zhao, Feng Liu, Jiong Wang, Kaiyi Xu, Zijie Guo, Zixin Chen, Ben Fei, Fenghua Ling, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.27845)  

**Abstract**: Recent advances in large language models (LLMs) have enabled AI systems to assist scientific research and peer review. However, an essential capability for reliable AI-assisted scientific workflows remains underexplored: verifying whether reviewer feedback leads to meaningful and evidence-supported manuscript improvements. We introduce AutoSupervision, which evaluates whether scientific manuscript revisions genuinely address reviewer concerns through grounded evidence. AutoSupervision leverages transparent peer-review records as a natural source of supervision, where reviewer comments specify scientific concerns, author responses describe claimed resolutions, and revised manuscripts provide evidence of changes. Given reviewer comments, author responses, and revised manuscripts, models must characterize reviewer concerns, determine whether concerns have been addressed, and identify supporting manuscript evidence. We construct AutoSupervision from 56,000 Nature Communications articles and corresponding review records. Then we conducted experiments on LLMs, the ablation study, and the case study. Our results show that while LLMs perform well in characterizing reviewer concerns, with GPT-5.5 achieving a score of 0.754, evidence-based verification remains the primary bottleneck, with the best-performing model reaching only 0.501. 

---
# SAFViT: Spatial Attention Fusion Gating for Vision Transformer-Based Nucleus Segmentation and Classification 

**Authors**: Harshit Mittal, Arash Rabbani  

**Link**: [PDF](https://arxiv.org/pdf/2607.27835)  

**Abstract**: Accurate cell segmentation and classification are foundational to digital pathology, enabling quantitative tissue analysis for diagnosis and treatment planning. Encoder-decoder architectures that fuse multi-scale features through skip connections have become the dominant paradigm for this task, yet standard direct skip connections treat every spatial location equally, which leads to redundant and potentially conflicting information reaching the decoder. To overcome this problem, various gating mechanisms have been introduced, but most of them operate solely on filtering encoder information, neglecting the benefit of global contextual information from the decoder. This study proposes replacing conventional skip connections in a CellViT-based model with a novel Spatial Attention Fusion (SAF) Gating module. Each SAF gate concatenates the encoder skip and upsampled decoder features, compresses them through two pointwise convolutions with an intermediate ReLU, and applies a channel-wise softmax to produce a per-pixel "heatmap of trust" that sums to unity at every spatial location, allowing the network to learn where each source is most trustworthy. The resulting fused features improve the model's ability to detect the minority "Dead" class, which in turn enhances the multi-class panoptic quality (mPQ) on the PanNuke dataset. SAF Gating is compared against six gating alternatives including no gating, attention gates, squeeze-and-excitation, CBAM, cross-attention, and attentional feature fusion on PanNuke and MoNuSeg datasets. SAF Gating achieves the highest mPQ (0.471), a gain driven primarily by a 14.5-point improvement in Dead-class F1 score compared to ungated CellViT baseline. 

---
# Deep Learning for Accelerated Long-Horizon Forecasting of Multicomponent Multiphase Microstructure Evolution in High-Entropy Alloys 

**Authors**: Hamidreza Razavi, Nele Moelans  

**Link**: [PDF](https://arxiv.org/pdf/2607.27820)  

**Abstract**: Phase-field modeling provides a powerful approach for predicting microstructure evolution but becomes computationally prohibitive for multicomponent and multiphase systems over large spatial and temporal scales. This work presents an AE-GCN-LSTM surrogate framework for long-horizon forecasting of microstructure evolution in the multicomponent AlCrFeNi high-entropy alloy system containing coexisting BCC and FCC phases. A multi-head autoencoder compresses the four elemental concentration fields and phase-field order parameter into latent representations, which are formulated as graphs for learning their spatial and temporal evolution. The framework accurately forecasts microstructure evolution over horizons extending to 3,000,000 simulation timesteps. Its robustness is systematically evaluated under previously unseen conditions without retraining, fine-tuning, or parameter adaptation. These evaluations include variations in FCC precipitate size and initial position, microstructures containing one, two, and five FCC precipitates, and complex phase interactions involving precipitate merging and splitting. Although trained only on 100 x 100 computational domains containing a single nominal alloy composition, the framework is successfully transferred to larger 256 x 256 and 512 x 512 systems and to previously unseen AlCrFeNi compositions. Across the evaluated configurations, the model preserves the dominant phase morphology and compositional evolution while providing computational speedups ranging from approximately 7200 to 62300 relative to conventional phase-field simulations. These results demonstrate that latent graph-based AE-GCN-LSTM forecasting provides a scalable and computationally efficient surrogate for long-horizon simulation of multicomponent, multiphase microstructures and offers a promising foundation for high-throughput alloy design. 

---
# Beyond Borrowed Histories: Person-Aligned User Simulation for Interactive Role-Playing Evaluation 

**Authors**: Yuhang Zhu, Mingxuan Du, Benfeng Xu, Jie Gao, Lingyun Yu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.27816)  

**Abstract**: Role-playing agents (RPAs) have become one of the most important consumer applications of large language models. Users engage in multi-turn conversations with RPAs for experiences such as emotional comfort, making reliable evaluation essential for measuring capability, comparing systems, and guiding further improvement. Existing benchmarks, however, typically require an RPA to continue a fixed dialogue history and then evaluate the continuation using a fixed rubric detached from the user. We identify and empirically demonstrate two limitations of this design. First, an RPA's output is shaped by the preceding dialogue history, preventing a scientifically grounded assessment of its role-playing ability in real multi-turn settings. Second, user experience varies substantially across individuals, and conventional fixed rubrics need not align with user satisfaction. We therefore introduce PALATE (Person-Aligned LLM-Simulated-User Assessment with Tailored Evaluation), a scalable RPA benchmark built on user simulators. PALATE is accompanied by a pool of 300 character profiles. Its main evaluation trains five per-user simulators and lets them engage candidate RPAs in free-form, multi-turn conversations over a pre-frozen panel of character profiles. Alongside a general quality rubric, we construct personalized rubrics to measure user satisfaction; on held-out annotated data, the personalized rubrics show higher agreement with human judgments than the general rubric. In the main evaluation of 16 candidates, PALATE separately characterizes generic turn quality, long-horizon session capability, and per-user experience on multi-turn trajectories co-constructed by each candidate. It thereby produces interpretable evaluations of specific user-RPA pairs rather than compressing systems into a single user-independent ranking. 

---
# Can AI Follow In Einstein's Footsteps? 

**Authors**: Michael Shalyt, Nathan Regev, Marin Soljačić, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2607.27794)  

**Abstract**: AI is accelerating physics discovery, but perhaps away from Einstein-level theory building. To understand this gap, we must recognize a striking trend: while being very successful, the most visible AI contributions to physics discovery appear to mirror the historical development of physics, but in reverse. Human discovery in physics progressed, in broad strokes, from ancient pattern prediction, through phenomenological laws such as Kepler's, to principle-based universal theories such as relativity and the Standard Model. On the AI side, prominent contributions to physics discovery point in the opposite direction: early milestones emphasized explicit equation-discovery methods, such as symbolic regression, whereas more recent frontier contributions are powerful predictors such as AlphaFold and GraphCast, which can be remarkably accurate yet do not provide clear theoretical understanding. If this trend continues, AI would become extraordinarily good at prediction but may struggle to ever propose its first serious contender to quantum gravity or other paradigm-level theories. We review the current landscape of AI for physics discovery and highlight a critical missing skill: the ability to pose the right questions or invent the right principles to guide the development of new theories and the tests to falsify them. This mode of discovery has driven many of the deepest advances since the 17th century, where symmetry, simplicity, and new mathematical frameworks guided theory construction before experimental tests. Equipping AI systems with such skills could move them from predicting within known frameworks to proposing the next paradigm-level discovery in physics. 

---
# Semantic-Aligned Structural Abstraction for Multimodal Sentiment Analysis 

**Authors**: Wei Chen, Junkai Li, Tongguan Wang, Hui Liu, Feiyue Xue, Chuanxiang Ma, Ying Sha  

**Link**: [PDF](https://arxiv.org/pdf/2607.27790)  

**Abstract**: Multimodal Sentiment Analysis (MSA) aims to interpret complex human emotions by integrating natural language with non-verbal modalities. Non-verbal modalities share a structural isomorphism with natural language, as both can be viewed as feature sequences evolving over time. This isomorphism enables the transformation of non-verbal modalities into text-like tokens for unified semantic reasoning. Large Language Models (LLMs), designed to understand and generate sequential data, can thus be utilized to interpret complex affective sequences. However, existing LLM-based methods primarily capture low-level superficial features, failing to model affective semantics arising from structural variations and contextual interactions. To address this limitation, we propose \textbf{SentiLLM}, a unified framework that leverages \textit{Semantic-Aligned Structural Abstraction} to distill continuous raw signals into compact, semantically meaningful tokens. Specifically, we introduce a \textit{Dual-Stream Salience-Context Calibration Mechanism}, which disentangles non-verbal feature sequences into a focus stream and an ambient stream. The focus stream captures salient sentiment shifts (e.g., facial expressions) guided by textual priors, while the ambient stream characterizes stable background states. Through calibrating these dynamic sentiment shifts against background states, SentiLLM effectively projects non-verbal modalities into a unified semantic space, making them naturally understandable for LLMs. Serving as a plug-and-play module, SentiLLM significantly improves discriminative performance with only a small number of trainable parameters. Our method achieves superior performance on four datasets, MOSI, MOSEI, CH-SIMS, and CH-SIMS v2, demonstrating the effectiveness of the structural abstraction paradigm in MSA. Our code is available at: \href{this https URL}. 

---
# LoRA Scaffolded Policy Optimization (LSPO): A Sampling-Time Low-Rank Scaffold for Recovering Reinforcement-Learning Gradient on Zero-Reward Cliff Prompts 

**Authors**: Ken Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.27787)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) for mathematical reasoning suffers from a structural blind spot: on "cliff" prompts-those on which every sampled rollout in a group fails-the group-normalized advantage is identically zero, so GRPO produces no gradient on precisely the prompts at the frontier of the model's capability. We introduce LoRA Scaffolded Policy Optimization (LSPO), a sampling-time mechanism that recovers this lost gradient. Each RL step, LSPO detects cliff prompts, fits a small low-rank (LoRA) adapter by a brief supervised step on their ground-truth solutions, re-rolls the cliffs with the base-plus-adapter model, splices the now-successful completions back into the RL batch with an importance-sampling correction, and takes a GRPO step on the base alone; the adapter receives only the supervised gradient and is discarded at checkpoint, yielding a base-only model. On DeepMath-103K with DeepSeek-R1-Distill-Qwen-1.5B, evaluated over n=5 paired seeds per arm at a matched 1000-step reporting horizon, LSPO's 5-seed mean matches or beats a DAPO baseline on all 16 (benchmark, pass@k) cells (15 strict wins and one exact tie), with gains of up to +10.7 points on AIME24/pass@4, +6.7 points on AIME24 and AIME26 at pass@16, and +2.4 points on MATH500/pass@1; averaged over the 16 cells the improvement is +3.8 points. 

---
# Reasoning Consensus: Structural Ensembling of LLM Reasoning via Weighted DAG Aggregation 

**Authors**: Amruta Parulekar, Jinu Lee, Dilek Hakkani-Tür, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2607.27783)  

**Abstract**: Large Language Models (LLMs) explore problems through chain-of-thought, but this exploration is buried in unstructured prose. On high-stakes tasks, users cannot tell which steps are well-supported, which alternatives were seriously considered, or how the final conclusion compares to those the model discarded. We propose a framework that ensembles the reasoning structure, not just the answers, of multiple LLMs by weighted merging of Directed Acyclic Graphs (DAGs) extracted from reasoning chains. We weight each step by how many traces independently attest to it, to return "Consensus Reasoning". Across six benchmarks spanning statutory interpretation, graduate-level science, narrative multi-hop reasoning, and first-order logic, our ensemble outperforms a matched-budget majority-vote baseline, with a maximum accuracy gain of 3.1% on MuSR-MM (narrative multi-hop reasoning). On a single model, the framework matches or exceeds self-consistency at the same trace budget while additionally exposing an inspectable consensus reasoning graph. Ensemble weights correlate with LLM-judge rankings of reasoning quality at Spearman $\rho = 0.30$-$0.51$, and consensus subgraphs are preferred over alternatives leading to the majority-vote answer in 54.4-65.4% of head-to-head comparisons across five of six datasets. We observe that our framework can also be used to analyze diverse reasoning perspectives for a problem. 

---
# RedFlow: Redirect Failure into Action-Level Corrections for Flow-matching VLA Policy 

**Authors**: Zhengyang Yan, Junhao Li, Fangqi Zhu, Zijun Wang, Quanxin Shou, Yikun Miao, Xiaoyi Pang, Zicong Hong, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.27782)  

**Abstract**: Flow-matching Vision-Language-Action (VLA) policies have shown strong potential for robotic manipulation but often suffer from compounding errors caused by distribution shifts during deployment. While offline reinforcement learning (RL) provides a practical way to improve deployed policies using rollout data, existing methods either ignore failure data or exploit it only at the trajectory level, resulting in low learning efficiency and persistent errors. We propose **RedFlow**, a fine-grained offline RL framework that redirects failure experiences into action-level corrective supervision for flow-matching VLA policies. RedFlow consists of two key components: (1) a **Context-Aware Corrective Matching** mechanism that identifies failure-inducing actions and retrieves successful alternatives from similar contexts as corrective targets, and (2) an **Adaptive Redirection Objective** that jointly reinforces successful actions, suppresses undesirable ones, and redirects recoverable failures toward corrective targets. By converting both successful and failed experiences into dense supervision, RedFlow enables robust recovery learning from mixed-quality data. Experiments on the LIBERO benchmark and three real-world manipulation tasks show that RedFlow consistently outperforms state-of-the-art offline RL baselines, improving the real-world success rate from 56.7% to 74.7%. It also matches strong on-policy methods (PPO, GRPO, and DDPO) while requiring roughly an order of magnitude fewer training samples. 

---
# VocalRender: Score-Native Singing Voice Synthesis for Real-World Composition 

**Authors**: Yukun Chen, Tianrui Wang, Zhaoxi Mu, Xinyu Yang, EngSiong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27768)  

**Abstract**: Existing singing voice synthesis systems often require predefined durations, explicit duration prediction, or time-aligned acoustic guidance, which limits their compatibility with practical composition workflows. We propose VocalRender, a score-native system that directly synthesizes singing from lyrics, pitches, symbolic note values, and tempo. It uses an interleaved lyric--note representation and an autoregressive diffusion model to generate continuous acoustic latents while predicting the output length, eliminating the need for explicit duration prediction. Trained on a 2,300-hour singing dataset, VocalRender achieves strong intelligibility, strong melody control, and high speaker similarity across both in-domain and out-of-domain benchmarks. Notably, it outperforms the strongest baseline by $0.42$ points in naturalness CMOS, demonstrating the effectiveness of our proposed score-native architecture. 

---
# Train Small, Deploy Large: Zero-Shot GNN Transfer Through Geometric Renormalization 

**Authors**: Robert Jankowski, Pedro Almagro-Blanco, Marián Boguñá, Melanie Weber, M. Ángeles Serrano  

**Link**: [PDF](https://arxiv.org/pdf/2607.27767)  

**Abstract**: Graph neural networks (GNNs) can operate on large graphs but become infrastructure-sensitive at the scale of millions of nodes and typically require scalable training techniques for even larger graphs. This raises a central question: when can a model trained on a smaller, scaled-down replica of a graph be deployed on the full-resolution graph without retraining? We introduce a zero-shot transfer protocol in which a GNN is trained on a graph coarse-grained by geometric renormalization (GR), and the resulting weights are transferred directly to the original network. Across synthetic and real-world networks, training on GR scaled-down replicas preserves much of the original-scale predictive performance while significantly reducing training cost. We further find that learned representations and predictive trajectories remain aligned across scales. These findings suggest that structural similarity may be more important than network size in determining GNN transferability, opening a path toward scale-equivariant graph architectures. 

---
# Private Face Recognition Training Dataset Publication via Identity-Decoupled and Geometry-Preserving Face Distillation 

**Authors**: Shuhuan Chen, Xiangyu Zhu, Weisong Zhao, Siran Peng, Tianshuo Zhang, Haoyuan Zhang, Haichao Shi, Xiao-Yu Zhang, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2607.27764)  

**Abstract**: Publishing private face recognition~(FR) training datasets is privacy-sensitive because faces expose identity information. Private FR training dataset publication mitigates this risk by releasing protected proxies as substitutes for private training faces. However, training FR models with such data introduces an identity paradox: \emph{the identity cues that make released faces useful for recognition supervision are also the cues that make them linkable to real individuals.} A protected face should be decoupled from the original identity, yet still behave as a reliable identity sample for training. Removing these cues too aggressively may destroy the class structure needed for recognition learning, whereas preserving them too faithfully may increase source-identity linkability. We argue that this paradox stems from conflating source-aligned identity semantics with recognition-useful proxy identity geometry. The former should be suppressed to reduce linkage to private individuals, while the latter should be preserved for FR learning. Based on this insight, we propose \textbf{Private Face Distillation}, an identity-decoupling and geometry-preserving framework. It uses Orthogonal Geometry Preservation to construct decoupled proxy identities from private identity representations while maintaining hyperspherical geometry, and Relational Topology Alignment to preserve identity relations for recognition learning. Experiments across multiple domain-shifted FR scenarios show that Private Face Distillation achieves stronger utility than the evaluated publication baselines. On IJB-C surveillance, it improves $\mathrm{TAR}@\mathrm{FAR}{=}1\text{e-}{3}$ by 3.94\% over the baseline while reducing source-identity linkability. These results suggest that private FR training dataset publication should decouple source-identity correspondence while preserving proxy identity geometry. 

---
# Hierarchical Latent Reasoning for LLM-based Recommendation 

**Authors**: Peiyu Hu, Siying Gu, Weihai Lu, Zhuodong Liu, Yuntian Tang, Jiahao Liang, Yiying Xie, Jiang Rong, Zhaokai Luo, Zhiyong Wang, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27760)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for recommendation by leveraging their semantic understanding and contextual modeling capabilities. Recent studies further introduce reasoning mechanisms to improve user preference modeling. However, explicit natural-language reasoning incurs substantial inference overhead, whereas existing latent reasoning methods mainly focus on generating or verifying intermediate states, leaving their layer-wise preference roles and contributions insufficiently characterized. We propose HiLaR, a Hierarchical Latent Reasoning framework with layer-aware reinforcement optimization for LLM-based recommendation. HiLaR constructs temporal-guided hierarchical user preference representations, aligns them with multiple LLM latent reasoning states, and organizes the reasoning process from broad preferences to fine-grained current intents. To further optimize the reasoning trajectory, HiLaR combines final recommendation feedback with layer-aware process rewards derived from the marginal target-likelihood gain of each state. Experiments on four Amazon benchmark datasets show that HiLaR generally outperforms strong sequential, generative, and LLM-based recommendation baselines. Ablation and sensitivity analyses further verify the contribution of hierarchical representation learning, latent alignment, and process-level optimization. Our code is available in this https URL. 

---
# A Structured Knowledge Infrastructure for Domain-Specific Data Asset Discovery 

**Authors**: Mengdi Chen, Yuanxin Huang, Yulin Jiang, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.27748)  

**Abstract**: Enterprise data analytics agents face two structural failures: generic RAG retrieves the wrong asset (Hit@10=19.1%) and delivers no usage knowledge to prevent metric misinterpretation---stemming from four root causes (C1--C4) ranging from semantic gap and entity ambiguity to schema drift and asset-usage gap. We present a two-layer solution deployed in the commercial advertising data warehouse at Xiaohongshu (5,300+ Hive tables, 14 domains). A three-tier dual-purpose knowledge base (179 documents, eight-section annotation template) serves both retrieval and generation, with a closed-loop refresh pipeline maintaining day-level freshness (one yes/no approval, 30s hot-reload). The Graph-Guided Retriever (GGR) uses a 2,859-node knowledge graph as a candidate gate with intent routing to deliver 71.6x token reduction. The Scene-Aware Ranker (SAR) applies 19-class entity recognition and explicit scenario annotations; negative knowledge alone contributes 25 percentage points of Hit@10 gain. On two 100-question benchmarks, Hit@10 rises from 19.1% to 96.6% (+77.5pp) and knowledge coverage from 56% to 77%, at 4.84--5.33s end-to-end latency. 

---
# Can LVLMs Uncover the Truth Behind Visual Illusions? An Analysis of Perceptual and Reasoning Capabilities 

**Authors**: Liangjie Zhao, Jiaqing Lyu, Kexin Tang, Zecheng Fang, Rong Yin, Yulan Hu, Da Li, Jianing Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.27747)  

**Abstract**: Large Vision Language Models have integrated reasoning capabilities, elevating cognitive performance to new levels. However, existing evaluations either focus solely on perception or rely on specific domains such as maths or coding. Evaluation for reasoning capabilities that align with an open-world environment is still required, especially one that considers perception and reasoning jointly. To bridge this gap, we propose to evaluate LVLMs by exploiting visual illusions as a diagnostic tool. Visual illusions are phenomena in which the human visual system misinterprets objective signals, resulting in an understanding that deviates from reality. We constructed IllusionReasoning, a benchmark of illusion images collected from the real world, incorporating diverse annotated question-answer pairs. Based on IllusionReasoning, we show that the reasoning capabilities of a wide range of LVLMs are not as advanced as claimed. Our work provides new insights into LVLMs and offers future direction for optimisation. 

---
# ROCS: Request-Oriented Compute Sharing for Efficient Large-Scale Recommendation 

**Authors**: Yuxin Chen, Liang Luo, Buyun Zhang, Jian Jiao, Boda Li, Haoyu Wang, Tongyi Tang, Ao Cai, Zijian Shen, Zhengkai Zhang, Wenyi Xie, Ryan Dick, Han Liu, Neng Shi, Bin Yu, Jianbo Xiao, Shuyao Bi, Hongtao Yu, Yuanwei Fang, Zhuoran Zhao, Sijia Chen, Yang Chen, Shuqi Yang, Qianru Li, Zikun Liu, Wei Ling, Sihan Zeng, Longhao Jin, Jiaxin Lu, Yinbin Ma, Jiawei Li, Yichen Ruan, Yong Ler Lee, Birmingham Guan, Zijian Li, Jianbo Sun, Zhengyu Zhang, Zeliang Chen, Xiaohan Wei, Yuchen Hao, GP Musumeci, Venkatesh Ranganathan, Yantao Yao, Chunqiang Tang, Wenlin Chen, Santanu Kolay, Ellie Dingqiao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27744)  

**Abstract**: Modern recommendation models gain prediction quality by scaling feature-interaction and sequence modules, but production cost constraints cap how far systems can scale.
In this work, we propose Request-Oriented Compute Sharing (ROCS), a modeling and inference paradigm that exploits a unique property of recommendation inference: each user request is evaluated against many candidates, while request-side features are shared across candidates. ROCS defers request-candidate interactions as late as possible, isolates candidate-dependent representations, and evaluates substantial portions of the model once per request rather than once per candidate, significantly improving inference efficiency while maintaining or improving prediction quality. To realize this paradigm, we develop Generalized Layer Masking (GLM) to enforce candidate isolation in feature-interaction architectures, and Deep Cross Attention (DCA) to extend request-oriented sharing to sequence architectures. To support efficient GPU deployment, we co-design In-Kernel Broadcast Optimization (IKBO) that significantly accelerates ROCS model execution.
Experiments on public benchmarks show that ROCS consistently improves the quality-efficiency tradeoff across recommendation backbones. On production-scale workloads, ROCS achieves up to a 3x QPS improvement on retrieval models without quality degradation and a 0.5% relative LogLoss improvement with a 50% QPS gain on a short-form video ranking model. ROCS has been deployed across large-scale recommendation systems spanning ads and organic surfaces, retrieval and ranking stages, and more than two orders of magnitude in inference complexity, delivering significant online gains at reduced infrastructure cost. 

---
# Towards joint scaling laws with optimal batch size schedules 

**Authors**: Jiaxiang Li, Zhiqi Bu, Shiyun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27731)  

**Abstract**: Modern deep learning typically keeps the batch size static throughout training, thus overlooking the joint effect of learning rate and batch size on the training dynamics. In this paper, we study the deep learning dynamics through the lens of convex optimization and derive a joint characterization of loss in terms of both schedules, applicable to general optimizers and model architectures. This characterization yields a closed-form optimal batch size schedule for any prescribed learning rate schedule, and further leads to joint scaling laws that consistently outperform static batch size baselines, highlighting the significance of dynamic batch size schedule in large language model training. 

---
# RefineSVG: Visual Feedback-Driven Reinforcement Learning for Image-to-SVG Generation 

**Authors**: Shaobo Liu, Feiqiao Mao, Shuaishuai Zhou, Yan Zhan, Weiqi Tan, Zhiqiong Lu, Zhengping Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27699)  

**Abstract**: We propose RefineSVG, a single-step closed-loop visual feedback framework that enables multimodal large language models (MLLMs) to perform high-fidelity image-to-SVG generation through self-correction. Existing MLLM-based approaches rely on single-pass open-loop inference, where the model receives visual input only once and must generate thousands of SVG code tokens without intermediate verification. This paradigm inevitably leads to geometric drift, error accumulation, and visual hallucination on complex images. RefineSVG overcomes this limitation by invoking an external rendering engine after an initial SVG generation pass to compare the rendered output against the target image. The comparison yields a multi-dimensional visual residual map (Diff-Map) that is fed back to the model as a ReAct-style correction signal, driving a targeted correction step. To support this render-observe-correct interaction, we further introduce an SVG-oriented semantic vocabulary that compresses token sequences by over 52%. A progressive training pipeline spanning supervised fine-tuning, rejection-sampling cold-start data construction, and end-to-end agentic reinforcement learning aligns the model with closed-loop visual correction. Extensive experiments show that RefineSVG consistently outperforms existing baselines in reconstruction fidelity, structural accuracy, and code this http URL is available at this https URL. 

---
# LabEvolver: Training-Free Experience Evolution for Safe and Grounded Wet-Lab Agents 

**Authors**: Jingya Wang, Yuyang Gao, Liuzhenghao Lv, Yonghong Tian, Yuyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27690)  

**Abstract**: We introduce LabEvolver, a training-free framework that equips safe and grounded wet-lab agents with episodic memory from execution experience. LabEvolver couples a state-grounded inner trial loop for adaptive perception, online planning, and safety validation with an outer evolution loop that distills completed trajectories into reusable skill, strategy, and safety experience. On robotic solution-preparation tasks, LabEvolver demonstrates real-world feasibility, reducing pH-regulation completion time and safety-gate intercepts by 48.2% and 60.0%, respectively. On ALFWorld, it further improves cumulative success rate within 20 steps from 76.2% with ReAct to 91.4% over 500 continual tasks, showing generality beyond wet-lab settings. These results support learn-by-doing experience evolution as a feasible path toward closed-loop automated scientific discovery. The project page is available at this https URL. 

---
# Stop Shipping AI Agents on Faith: Capability Is Not Production Readiness 

**Authors**: Fouad Bousetouane  

**Link**: [PDF](https://arxiv.org/pdf/2607.27677)  

**Abstract**: AI agents are moving into production workflows where they retrieve information, call tools, maintain state, and act on behalf of users or organizations, but many release decisions still rely on capability signals, demos, or behavioral tests that do not show whether an agent is ready to operate under production constraints. Capability is therefore not production readiness. This paper introduces the ProofAgent Index (PAI), a governance readiness index for AI agents. PAI combines four dimensions of deployment evidence: Evaluation, Context, Compliance, and Governance. Evaluation measures observed behavior, Context measures the operating environment that shapes that behavior, Compliance measures alignment with applicable rules and controls, and Governance measures whether the organization can authorize, monitor, audit, and control the agent during operation. PAI is implemented inside ProofAgent Harness, an open source infrastructure for auditable AI agent evaluation and governance. Validation across two heavily regulated domains, healthcare and finance, shows that PAI carries held out readiness signal and separates higher risk from lower risk configurations. The results show that context engineering strongly changes reliability, capability improves behavior but does not determine readiness, and governance evidence must remain visible rather than averaged away. PAI reframes agent release from a faith based deployment decision into an auditable readiness decision. 

---
# JigShape: Evaluating Visual-Geometric Reasoning in VLMs through Jigsaw Puzzles 

**Authors**: Shawn Li, Wei Yang, Jike Zhong, Jiate Li, Jiawei Yang, You Qin, Ryan Rossi, Franck Dernoncourt, Roger Zimmermann, Yue Wang, Zhengzhong Tu, Vicente Ordonez, Mohit Bansal, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.27670)  

**Abstract**: Jigsaw puzzle solving requires jointly reasoning about visual content and geometric constraints, yet existing benchmarks use rectangular cuts that create ambiguous ground truth in texture-repeated regions. We introduce \textit{\ours{}}, a benchmark with tab-and-blank interlocking pieces where geometric constraints provide strong local compatibility requirements that, combined with visual content, yield unambiguous ground truth. Across 95K instances at four grid densities (4$\times$4 to 16$\times$16), we find that \textbf{zero-shot VLMs largely lack geometric reasoning}: only one of five frontier models (GPT-5.5) exceeds random baseline on 4$\times$4 puzzles, while all others perform at chance level. While supervised fine-tuning achieves $>$97\% on 4$\times$4, \textbf{all models collapse on larger grids}: GPT-5.5 drops from 70\% to near-random on 8$\times$8, and even fine-tuned models fall below 5\% on 12$\times$12. This ``scaling cliff'' suggests current architectures cannot maintain consistent constraint satisfaction as the number of pieces increases. \ours{} establishes scalable geometric reasoning as an open challenge for vision-language models. 

---
# Understanding Submodular Information Measure Based Objectives for Representation Learning: A Variance and Separation Perspective 

**Authors**: Rishabh Iyer, Truong Pham, Anay Majee  

**Link**: [PDF](https://arxiv.org/pdf/2607.27660)  

**Abstract**: Submodular Information Measures (SIMs) have recently emerged as a powerful framework for representation learning and multimodal learning. In particular, the SCORE framework~\cite{majee2024score} demonstrated that SIMs can serve as effective objectives for supervised contrastive learning. Despite their empirical success, however, the geometric and statistical properties induced by different submodular information measures remain poorly understood.
In this work, we develop a unified theoretical framework connecting SIMs to classical concepts in representation learning and statistical pattern recognition. We show that Total Information (TI) objectives characterize intra-class structure: Graph Cut TI recovers within-class variance, LogDet TI recovers generalized variance and covariance volume, and Facility Location TI induces imbalance-aware separation that emphasizes rare and confusable classes. We further show that Mutual Information (MI) objectives capture complementary notions of inter-class structure: Graph Cut MI is closely related to centroid separation and Fisher-style discrimination, LogDet MI captures covariance-aware separation through Mahalanobis distance, and Facility Location MI measures nearest-mode representational overlap.
We validate these theoretical characterizations using controlled synthetic experiments that independently vary variance, covariance, class imbalance, class separation, and multimodal overlap. Across all settings, the empirical behavior closely matches the proposed theory. Our results provide the first unified geometric and statistical understanding of submodular information measures and offer principled guidance for selecting and designing SIM-based objectives for representation learning. 

---
# Learning Color Grading, No Photo Sharing: Federated Aesthetic Preference Learning for Personalized Image Enhancement 

**Authors**: Chuanzhi Xu, Ziyuan Tao, Jean Julien KNell, Yanrong Chen, Haolan Guo, Xuanhua Yin, Adnan Mahmood, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.27659)  

**Abstract**: Personalized image enhancement should reflect individual aesthetic taste, yet learning such preferences commonly depends on private photos and ratings that are unsuitable for centralized collection. The task must infer preference from sparse, heterogeneous feedback and translate it into natural-looking color transformations on resource-constrained user devices. We introduce FedPAIE, a federated personalized aesthetic image enhancement framework for user-adaptive color grading without centralizing raw photos or ratings. FedPAIE trains a lightweight dual-cue aesthetic scorer, calibrates it into a personalized scorer on a small local support set, and freezes it to guide regularized adaptation of a lightweight CLUT enhancer from unpaired local photographs. Fidelity constraints and an excess-gap penalty regularize scorer-guided adaptation to limit proxy-score over-optimization while preserving content and natural appearance. Training remains lightweight throughout the pipeline: scorer learning updates at most 0.787M parameters, enhancer adaptation updates 0.265M, and inference retains only a 0.293M-parameter personalized enhancer. Experiments on MIT-Adobe FiveK and Flickr-AES demonstrate effective open-world personalization and a favorable balance between user preference and image fidelity. FedPAIE thus connects decentralized preference learning with efficient personalized image transformation without requiring paired user retouches. 

---
# From Single- to Cross-Document: Benchmarking Multi-Granularity Event Analysis of Large Language Models 

**Authors**: Tao Wen, Shuai Shao, Pei Ke, Xu Han, Jie Zou, Guannan Li, Tao Tian, Jinjie Qiu, Lan Wang, Ke Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27654)  

**Abstract**: Event analysis is an essential and fundamental direction of information extraction, involving various event-centric tasks at different granularity of documents. While large language models (LLMs) have preliminarily achieved promising performance in part of these tasks individually, their capability in event analysis still lacks comprehensive understanding due to restricted document granularity, task designs, and data source of existing benchmarks. To address these limitations, we introduce MiGUE-Bench, a systematic benchmark for assessing the performance of LLMs in multi-granularity event analysis. To support large-scale evaluation, we first develop an LLM-driven self-correcting annotation framework called MiGUE-Pipeline, enabling scalable acquisition of high-quality source data of events with automatic labels. Then, we design four core tasks in our benchmark, i.e., event detection, relation reasoning, structure induction, and future prediction, to probe model competence at different levels, from atomic event details to complex cross-document narratives. Extensive experiments on state-of-the-art LLMs and retrieval-augmented generation (RAG) methods delineate the current capability boundary and identify critical deficiencies, providing insights into the future improvement of LLMs in challenging event analysis tasks. 

---
# Arm2Air: Cross-Embodiment Skeleton Transfer for 3D Relay Formation 

**Authors**: Dohun Lee, Kyeonghyun Yoo, Seokmin Kim, Byongho Lee, Seungjoo Oh, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.27627)  

**Abstract**: Unmanned aerial vehicle (UAV) relay networks can restore connectivity after communication infrastructure is damaged. Urban relay placement is difficult because line-of-sight blockage, communication range, altitude, and three-dimensional obstacles must be considered jointly. Arm2Air transfers obstacle-avoidance skeletons from robot arms to UAV relay placement through cross-embodiment transfer. Source-domain robot-arm motions from a pretrained Neural MP model are converted into ordered skeletons that pretrain a transformer-based transfer platform, which is then adapted to the UAV domain using limited target data and Low-Rank Adaptation. The transferred skeleton initializes a relay chain that is refined for connectivity, bottleneck capacity, delay, and movement cost. On nine held-out high-clutter 3D urban maps, Arm2Air reduced median end-to-end planning runtime by 64.9 percent relative to the fastest conventional planner. On the high-obstruction group of a separate 30-map dense urban holdout, it increased bottleneck capacity by 32.6 percent, reduced capacity variance by 74.7 percent, reduced maximum hop distance by 13.2 percent, reduced hop-distance variance by 75.2 percent, and reduced relay displacement by 16.9 percent relative to IMPC-MD. With only three target-domain training maps, Arm2Air reduced relay-position root mean square error by 53.6 percent relative to training from scratch while updating 0.134 million parameters, compared with 1.383 million for Scratch and Full Fine-tuning. These results demonstrate computationally and data-efficient UAV relay placement and suggest a broader principle for transferring ordered structural priors across heterogeneous embodied tasks. 

---
# DualAnchor: Preserving Language Priors and Improving Lexical Fidelity in Gloss-Free Sign Language Translation 

**Authors**: Hongbin Zhang, Junhao Liu, Xuefeng Bai, Youcheng Pan, Yang Xiang, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27614)  

**Abstract**: Recent advances in large language models (LLMs) have led sign language translation (SLT), the task of converting sign-language videos into spoken-language text, to increasingly adopt LLMs as textual backbones. However, despite their strong language modeling capabilities, existing LLM-based SLT methods often undermine rather than exploit this language prior, producing disfluent translations, a failure we term language-prior degradation. Meanwhile, existing methods typically align videos and text at the sentence level, which does not ensure accurate lexical details and creates a lexical fidelity gap. To address both issues, we propose DualAnchor, a gloss-free LLM-based SLT training framework that couples two complementary anchors for linguistically fluent and visually faithful generation. Token-level Prior Anchoring (TPA) preserves the LLM's language prior by regularizing the multimodal decoder at each decoding step toward the next-token distribution of a frozen LLM conditioned on the same autoregressive prefix. Optimal Transport Alignment (OTA) improves lexical fidelity by formulating visual-textual matching as entropy-regularized partial optimal transport, with Sinkhorn optimization inducing a soft alignment between visual tokens and textual content tokens under a cosine cost. DualAnchor achieves strong overall performance on both PHOENIX-2014T and CSL-Daily. Targeted analyses attribute these gains to the complementary effects of the two anchors: TPA improves fluency, whereas OTA reduces fine-grained lexical errors. 

---
# Revisiting the Adversarial Robustness of Graph-Based Traffic Forecasting 

**Authors**: Qingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27604)  

**Abstract**: Traffic forecasting by graph-based AI is a critical component of intelligent transportation systems, motivating security research on robustness to malicious sensor readings. We argue that prior robustness evaluations are largely shaped by unrealistic threat models and untargeted objectives, so both attacks and defenses must be revisited. We study a practical adversary with limited model knowledge and the ability to monitor and manipulate only a few road sensors. More importantly, practical attacks can be localized to specific links or routes, causing incorrect estimated arrival times or unnecessary rerouting while leaving the broader network largely unaffected. This targeted setting remains underexplored, and defenses such as adversarial training do not transfer well from the norm-bounded attacks they train on to structurally different, physics-aware attacks that mimic genuine congestion. We therefore reframe robustness as a detection problem, introducing a learned physics-informed detector whose output is fed to a hardened forecaster as an input feature and trained against adaptive attacks with the forecaster fixed. We evaluate across a variety of model architectures and benchmarks. The physics-aware attack multiplies target-link error several-fold while the network-wide error barely moves, and adversarial training, tuned to norm-bounded perturbations, barely dents it. Our detection--mitigation defense improves even on adversarial training hardened against the physics-aware attack itself, on $13$ of $15$ model--dataset settings and by the widest margin on a held-out attack, at near-zero clean cost. The results emphasize the need to examine abstracted AI adversarial attacks under application-specific constraints to assess their true security impacts. 

---
# A Systems Engineering Framework for Vision-Language-Enabled UAV Triage and Disaster Response 

**Authors**: Swapnil Saha, Bhuvan Rajanasiriyur Jagadeesha, Karishma Patnaik, Neelakshi Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2607.27597)  

**Abstract**: Recent advances in Vision Language Models (VLMs) have created new opportunities for disaster response, where responders must interpret large volumes of sensor data under time pressure. Current VLM applications include social media monitoring for situational awareness, generation of draft action plans, and translation of technical alerts into public-facing messages. While these efforts can accelerate information flow, they remain largely limited to decision-support roles. Such approaches can increase operator burden because humans must still translate outputs into coordinated actions across teams and robotic assets. This study explores the viability of embedding VLMs as coordination agents within the human-UAV loop. The proposed architecture integrates natural language interaction, mission-level task coordination, software-in-the-loop implementation, and communication aligned with the Incident Command System (ICS). Rather than functioning solely as advisory tools, VLMs facilitate communication between human operators, mission control logic, and UAV task execution. The framework was developed using a Model-Based Systems Engineering (MBSE) approach, with use case and block definition diagrams representing system roles, internal structure, and component interactions. Three key elements, the VLM Coordinator Agent, UAV Mission Control, and Task Allocator, were implemented within an integrated simulation and control environment. A preliminary human-factors evaluation with seven participants showed reduced perceived workload across mental demand, effort, and frustration, along with high ratings for AI trust and communication clarity. By integrating MBSE, software-in-the-loop testing, and human-factors evaluation, this work advances scalable human-autonomy teaming for high-stakes disaster response, with broader implications for aerospace autonomy and civil safety. 

---
# Beyond Similarity: Grounded Agentic Extraction and Expert-Adjudicated Evaluation of Intertextuality in Classical Chinese Histories 

**Authors**: Zhaoji Wang, Wanyu Si, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27595)  

**Abstract**: Computational approaches to intertextuality have advanced from string matching to neural retrieval, yet their outputs, similarity scores and parallel-passage lists, identify where texts reuse one another without characterizing how or why. We recast fine-grained intertextuality extraction as an agentic task in which a large language model (LLM) reads two text units in full and, through a constrained tool interface, must ground each proposed reuse in exact character spans on both sides and label it under a five-dimension typology of reuse (form, aspect, source-marking, function, stance). We validate the approach on an exhaustive comparison of the Analects with the Book of Han, where three domain experts adjudicate a pooled multi-model candidate set into a benchmark of 2,533 intertextual pairs. Against this standard we study twelve LLMs, reporting precision (56%-93%), a 51$\times$ cost spread at comparable quality, and how well their confidence is calibrated. Expert agreement traces a reliability gradient: dimensions legible on the textual surface are annotated consistently, while those requiring inference of intent are contested, delimiting the claims such annotation supports. Scaling the validated extractor to the full Twenty-Four Histories (65,380 comparisons, 5,766 pairs) recovers corpus-level structure a similarity score cannot express. The interpretive composition of citation shows no systematic change across eighteen centuries, yet the same passage is quoted less and less literally. Stability in the aggregate with drift in the individual case is what a cultural-attraction account expects. We release the extraction protocol and the expert-adjudicated benchmark. 

---
# Is Solving Better Than Evaluating GenAI Solutions? 

**Authors**: Ethan Dickey, Marios Mertzanidis, Alexandros Psomas  

**Link**: [PDF](https://arxiv.org/pdf/2607.27586)  

**Abstract**: As Generative AI (GenAI) tools become increasingly capable of generating solutions to computing assignments, the computing education community is exploring pedagogical approaches that emphasize solution evaluation, verification, and critique alongside traditional solution generation. However, evidence regarding the impact of such evaluation-centered tasks on student learning remains limited, particularly in upper-division, theory-heavy courses. We conducted a randomized A/B crossover study (N=220) in a junior-level algorithms course to compare evaluating GenAI-generated solutions with traditional problem solving. Across six assignments, student working groups either solved challenging algorithmic problems directly or evaluated often-flawed GenAI-generated solutions, with roles reversed midway through the semester.
We found no statistically significant differences between groups in midterm scores, final exam scores, overall course grades, or exam problems structurally aligned with the homework interventions. Students received significantly higher homework scores when evaluating GenAI-generated solutions, but this localized advantage did not translate into downstream summative gains. Survey data further indicated that most students reported no change in study habits in response to the intervention; however, those who reported adapting their study strategies rated the GenAI-evaluation assignments as significantly more helpful. These findings suggest that GenAI evaluation redistributes student effort from open-ended solution construction toward verification, diagnosis, and judgment, but does not automatically produce stronger conceptual transfer. We conclude that GenAI-evaluation activities can be incorporated into algorithms coursework without broad performance losses, but meaningful learning gains may require deliberate scaffolding that pushes students beyond simple error diagnosis. 

---
# Drawing-Recode: Annotation Grounding for Parametric CAD Code Generation from Raster 2D CAD Drawings 

**Authors**: Mingi Kim, Yongjun Kim, Hyungki Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.27558)  

**Abstract**: Recovering Parametric CAD sequences from raster-format 2D Computer-Aided Design (CAD) drawings accumulated prior to digital transformation is important for part reproduction and manufacturing process automation. However, existing studies either process only vector drawings or are limited to specific domains, and fail to explicitly connect dimensional annotations to geometric information, limiting their use of dimensional information for 3D Parametric CAD sequences recovery. We propose Drawing-Recode, a framework that generates Parametric CAD sequences as CAD code from raster 2D CAD drawings. Drawing-Recode extracts geometric features via an image encoder and recognizes annotations through a separate text recognition module, then explicitly grounds annotations to geometric information using cross-attention and our proposed Annotation Grounding Loss (AGL). The resulting features are fed into a Large Language Model (LLM) to generate CAD code in the Structured Parametric CAD Code (SPCC) format. Experiments show that Drawing-Recode outperforms existing baselines and remains robust on scanned drawings resembling industrial conditions. We expect Drawing-Recode contributes to digitizing raster 2D CAD drawings in industrial settings and to part reproduction and manufacturing automation. 

---
# Cross-Embodiment Transfer via Behavior-Aligned Representations 

**Authors**: Ajay Sridhar, Jensen Gao, Jonathan Yang, Jean Mercat, Suneel Belkhale, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2607.27549)  

**Abstract**: Recent progress in large-scale imitation learning for robot manipulation has been driven by leveraging datasets across a wide range of robot embodiments. However, achieving significant cross-embodiment transfer is often still challenging. In this work, we study the role of using behavior-aligned representations (e.g., object bounding boxes, language motions, end-effector traces of robot motion) in vision-language-action (VLA) models to promote cross-embodiment transfer. We hypothesize that by possessing invariances across embodiments while being predictive of robot actions, these representations can help unify large-scale cross-embodiment data to enhance transfer. To assess our hypothesis, we develop a simulation-based benchmark designed to assess transfer with diverse cross-embodiment data to new embodiments. Using this benchmark, we compare different representations and ways of incorporating them. We identify that end-effector traces can be particularly beneficial for transfer, representations are generally more useful with larger prior datasets, and can be used to benefit from action-free data. We also demonstrate that they can enhance sim-to-real cross-embodiment transfer, improving task completion progress of real robot policies pre-trained on simulation data by 28%. We provide videos of our evaluations at our website: this https URL. 

---
# Strategy, Not Payoffs: A Behavioural Embedding of Normal-Form Games 

**Authors**: Joshua Caiata, Sreepriya Pulyassary, Xiang Li, Kate Larson  

**Link**: [PDF](https://arxiv.org/pdf/2607.27536)  

**Abstract**: Learning a strategic task changes more than what is directly taught: fine-tuning on one game can either enhance or degrade an agent's ability to reason in another. Understanding and predicting this transfer of strategic capabilities, however, remains a key challenge for large language models (LLMs). Normal-form games provide an ideal testbed for analyzing this phenomenon, as they feature explicitly defined payoffs and well-characterized equilibrium behaviours. In this work, we investigate whether game embeddings can explain and predict changes in LLM strategic capabilities following fine-tuning across different games. We propose a lightweight two-feature embedding that captures fundamental behavioural demands: the entropy of the Nash equilibrium and the sensitivity of optimal responses to an opponent's action. We show that while existing published structural embeddings primarily memorize game identities and fail to generalize, our behavioural embedding reliably predicts performance changes on held-out games. These results demonstrate that the transfer of strategic capabilities in LLMs is not dictated by the payoff geometry of a game, but by the underlying structure of the decision-making behaviour it requires. 

---
# ThreatForest: Multi-Agent Attack Tree Generation with Pluggable TTP Framework Mapping 

**Authors**: Cristian Leo, Anton Dykyi, Danny Cortegaca, Daniel Begimher, Prakash Jha  

**Link**: [PDF](https://arxiv.org/pdf/2607.27528)  

**Abstract**: Threat modeling is essential for secure software development, yet manual analysis of cloud-native architectures is slow and demands scarce security expertise. We present ThreatForest, a multi-agent system that generates structured attack trees from source code repositories, maps attack steps to adversary tactics, techniques, and procedures (TTPs) from a pluggable set of frameworks (MITRE ATT&CK, CAPEC, and cloud-specific threat matrices), and synthesizes actionable mitigations. ThreatForest decomposes threat modeling into a multi-stage agent pipeline -- repository analysis, context refinement, threat generation, parallel attack-tree construction with TTP mapping and mitigation synthesis, and report generation -- orchestrated as a directed graph with deterministic verification gates, bounded retries, and three human-in-the-loop validation points. A domain-specific sentence-transformer maps each attack step to candidate techniques by cosine similarity; we show empirically that this embedding stage, not the surrounding pipeline, is the dominant accuracy bottleneck. We evaluate ThreatForest across seven application domains on a sixteen-dimension rubric, scored by a panel of independent LLM raters with an adversarial verification pass and expert review. Panel-measured quality reaches 0.63-0.68 (on a 0-1 scale) for threat statements, attack trees, and mitigations, but only 0.29 for embedding-only TTP mapping -- a gap stable across all seven domains that isolates the binding constraint. A controlled single-call baseline on the same model more than doubles mapping defensibility, pinning the limitation on the embedding encoder rather than the multi-agent design. To our knowledge, ThreatForest is the first end-to-end system that turns a code repository into TTP-mapped attack trees with evidence-based mitigations across adversary frameworks, with a reusable framework for benchmarking such systems. 

---
# Hierarchical Reranking for Scalable Financial RAG System 

**Authors**: Joohyun Lee, Sungwoo Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.27523)  

**Abstract**: Analyzing financial documents such as 10-K filings, tabular disclosures, and macroeconomic reports demands expert reasoning and extensive time. However, existing Retrieval-Augmented Generation systems often struggle to process hybrid text-table structures or the massive scale of financial documents. To address these challenges, we propose Hierarchical Reranker, a RAG framework designed to improve retrieval performance and generative reliability across large-scale financial datasets. The system integrates three key innovations: Pre-Retrieval Optimization, enhancing query clarity and search efficiency through normalization, keyword expansion, and table transformation; Hierarchical Reranker Architecture, improving retrieval precision through a two-stage ranking mechanism; and Long-Context Management, preserving reasoning accuracy through adaptive input partitioning and fusion under extensive contexts. Across multiple benchmarks, including FinQA, FinanceBench, and ConvFinQA, the proposed system achieved an NDCG@20 score of 0.7918 and demonstrated superior factual consistency. Its robustness was further validated by achieving second place in the ACM-ICAIF '24 FinanceRAG Challenge. This work presents a deployable, domain-optimized RAG pipeline that enhances both the accuracy and scalability of financial reasoning, paving the way for automated audit reporting and quantitative investment analysis. The source code will be made publicly available on GitHub upon acceptance. 

---
# Expanding Data-Agnostic Pivotal Instances Selection Models with Proximity Trees and Ensemble Learning 

**Authors**: Alessio Cascione, Mattia Setzu, Cristiano Landi, Paolo Maria Mancarella, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2607.27522)  

**Abstract**: As decision-making processes grow more complex, machine learning tools have become essential for tackling business and societal challenges. However, many existing methods rely on decision-making procedures that are difficult to interpret. Since humans naturally make decisions by comparing new cases with a few representative examples, we aim to design an approach that selects such pivots to construct an interpretable predictive model. Inspired by decision trees, we propose a hierarchical, interpretable-by-design pivot selection model based on the similarity between pivots and input instances. Our method functions both as a pivot selection technique and a standalone predictive model. Extending beyond single pivots, we incorporate pairs of pivots that are used by proximity and oblique trees, as well as ensembles, which enhance the versatility and effectiveness of our proposal. Additionally, our approach is data modality-agnostic, leveraging pre-trained networks for data transformation. Experiments across diverse datasets, including tabular data, text, images, and time series, demonstrate the effectiveness of our approach, outperforming alternative instance selection strategies and achieving competitive results against state-of-the-art interpretable models while maintaining a minimal number of pivots. 

---
# Models for minimalist RAG: B1ade 335M Embedding and 1B Parameter Small Language Models 

**Authors**: Shreyas Subramanian, Mecit Gungor, Vikram Elango  

**Link**: [PDF](https://arxiv.org/pdf/2607.27506)  

**Abstract**: Language and embedding models used in RAG systems are conventionally assumed to require large-scale pretraining and explicit grounding supervision. We present B1ade, an efficient RAG architecture comprising two purpose-built components: a compact embedding model and a purpose-built SLM. B1ade-embed, a 335M parameter retrieval model constructed via parameter-free fusion of five pretrained encoders achieves top MTEB scores among sub-500M models with zero additional training, and B1ade-1B, an SLM trained on low-cost GPUs using Group Relative Policy Optimization (GRPO) on 723M tokens (2.2M examples) of curated context-question pairs with rewards that optimize only answer similarity. Our central finding is emergent attribution: despite receiving no explicit supervision for source citation, B1ade-1B cites retrieved passages in 42.4% of responses, exceeding the attribution rate of its training distribution by 5.5 percentage points. This demonstrates that grounding behavior can emerge as an accuracy-maximizing strategy under RL training, without explicit reward engineering. On standard QA benchmarks, B1ade-1B achieves 81.82% on PopQA, 65.8% on PubMedQA, and 51.09% on FEVER. In end-to-end RAG evaluation, B1ade-1B achieves an average score of 0.654 across correctness, completeness, coherence, and faithfulness, a 10.8% improvement over the SFT, while closing the gap with models 1.5x its size. These results show that strategic model composition and reward design suffice for resource-efficient RAG, without large-scale pretraining. 

---
# Latent States in Neural Networks: Recovering the Temporal Structure of Drifting Data from Model Weights 

**Authors**: Kevin Guan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27482)  

**Abstract**: A temporally drifting data stream may pass through discrete regimes rather than changing continuously. We ask whether such regimes are recoverable from the weights of models trained on the stream, using a hidden Markov model (HMM) fit to the chronologically ordered trajectory of those weights. We study this question in two domains known to drift over time: multimodal misinformation detection, using the Fakeddit dataset; and sentiment analysis, using the Yelp dataset. We train classifiers on consecutive temporal windows and fit an HMM to the trajectory of their aligned weights, recovering latent states that partition each timeline into coherent phases. On both datasets, classifiers generalize better to data from windows sharing the state of their training window than to windows across state boundaries. This within-state transfer advantage survives a control for temporal proximity and modestly exceeds the advantage recovered by a naive partition into contiguous states of equal size. Although the states are estimated solely from model weights, they correlate more strongly with shifts in the data's class distribution than with the weight-space geometry used to estimate them. After class divergence and lag are residualized out, the within-state advantage exceeds its permutation null on both tasks, indicating that the states recover structure relevant to transfer beyond the data distribution. Every effect replicates on both tasks but is attenuated on Yelp, whose label distribution is more temporally stable. 

---
# SE(3)-MeanFlow: Few-Step Protein Backbone Generation on Lie Groups 

**Authors**: Yikun Bai, Binghang Lu, Yikai Liu, Elaheh Akbari, Soheil Kolouri, Linxuan Wang, Ping He, Shuchan Wang, Ruqi Zhang, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27431)  

**Abstract**: Generative modeling of protein backbones promises the de novo design of proteins with prescribed structural and functional properties. Existing diffusion and flow-matching models produce high-quality backbones on SE(3)^N, but inference requires numerically integrating an ODE over hundreds of network evaluations, each involving a Lie group exponential map - a bottleneck for high-throughput design campaigns. We introduce SE(3)-MeanFlow, a few-step generative framework that extends MeanFlow from Euclidean space to the Lie group geometry of protein frames. Working natively in the Lie algebra so(3) and in R^3, we derive closed-form average-velocity identities for rotations and translations, giving simulation-free training targets. We further introduce an SE(3) alpha-Flow objective that removes the Jacobian-vector product from the rotation branch and serves as a warm-up stage, after which training switches to a small-t stabilized MeanFlow loss that is used for the remainder of pretraining and for rectification-based post-training. In protein backbone generation, SE(3)-MeanFlow matches or exceeds flow-matching baselines that use several times more sampling steps, and its advantage widens in the few-step regime, where rectification lets it lead at every matched budget - at a modest cost in diversity. 

---
# Rethinking Artificial Intelligence in Medical Imaging: Assumptions, Reality, and Reframing 

**Authors**: Arman Rahmim, Nourhan Bayasi, Xiaoxiao Li, Babak Saboury, Fereshteh Yousefirizi  

**Link**: [PDF](https://arxiv.org/pdf/2607.27428)  

**Abstract**: Medical imaging has served as primary proving ground for clinical artificial intelligence (AI), yet a decade of intense research has not translated into proportionate bedside impact. We argue that this gap is not primarily a product of insufficient algorithmic performance, inadequate regulation, or limited explainability. Rather, it reflects a structural misalignment, between how AI systems are designed and evaluated, and how clinical decisions are made. This Perspective identifies six interconnected dimensions of this misalignment: the dominance of pixel-only models in a multimodal clinical world; the erosion of physician trust through opaque and inflexible systems; the unfulfilled promise of foundation models in data-sparse medical domains; the persistent bottleneck of non-shareable, under-curated datasets; the gap between validated algorithms and deployable clinical platforms; and the failure of prediction-centric AI to generate actionable clinical guidance. For each dimension, we reframe the problem and propose a path forward, culminating in a vision of agentic, physician-aligned AI that extends, rather than replaces, clinical judgment. 

---
# Selecting Open-Weight Language Models for Zero-Shot Intent Classification: A Systematic Evaluation of 41 Models 

**Authors**: Parishruthi Ganesh, Gerry Dozier, Cheryl Seals  

**Link**: [PDF](https://arxiv.org/pdf/2607.27421)  

**Abstract**: Intent classification is a core component of task-oriented dialogue systems, yet practitioners have limited systematic guidance for selecting deployable open-weight language models under compute, latency, and robustness constraints. We present a systematic zero-shot evaluation of 41 open-weight language models spanning 15 families and the 135M--9B parameter range across eight English single-label intent-classification datasets. A ninth dataset, ATIS, uses five labeled demonstrations and is reported as an auxiliary five-shot result. The evaluation includes standard benchmarks, a large-scale voice-assistant corpus, and production-derived e-commerce datasets. Beyond exact-match accuracy, we analyze confidence calibration, robustness to realistic input perturbations, statistical reliability of model rankings, deployment efficiency, and benchmark saturation. Our results show that instruction-tuned 3B models can outperform several evaluated 7B base models, that differences among leading models on MASSIVE are statistically indistinguishable under pairwise McNemar tests, and that widely used benchmarks such as SNIPS have become saturated and no longer meaningfully discriminate among current open-weight models. Instruction tuning's effect on confidence calibration is inconsistent rather than uniformly harmful. These findings provide practical guidance for selecting and evaluating open-weight language models for intent classification. 

---
# SWE-NFI: Studying and Benchmarking Coding Agents for Non-Functional Improvements 

**Authors**: Pengyu Xue, He Yang Yuan, Xin Wang, Junkai Chen, Haonan Zhang, Boyuan Chen, Zishuo Ding, Zhenhao Li, Weiyi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27409)  

**Abstract**: Although coding agents have achieved impressive performance on correctness-oriented benchmarks, their ability to make behavior-preserving non-functional improvements (NFIs) remains underexplored. In real-world software development, developers continuously improve software quality without changing observable behavior, yet existing benchmarks primarily evaluate functional correctness and provide limited support for assessing these non-functional improvements.
In this paper, we present SWE-NFI, a benchmark for evaluating coding agents on NFIs beyond functional correctness. Our benchmark contains 188 tasks constructed from real merged pull requests in open-source Python projects. We operationalize developer-oriented NFIs into 92 executable rules and develop a comprehensive evaluation suite that combines functional correctness testing with rule-based NFI evaluation.
We evaluate state-of-the-art commercial and open-source coding agents. Although the best-performing agent achieves a 70.0\% functional correctness rate, all evaluated agents generally fall short of human developers in overall NFI capability. The gap is particularly evident for structural code improvements, where agents' NFI scores range from 0.0 to 1.3, compared with 1.5 for the human reference. Our benchmark and findings provide a reproducible foundation for evaluating and advancing coding agents beyond functional correctness. 

---
# Benchmarking LLM Competence on Logical Inference over Probability Operators 

**Authors**: Nayera Hasan, Jack Greff, Alvin Grissom II  

**Link**: [PDF](https://arxiv.org/pdf/2607.27405)  

**Abstract**: Both expressions of uncertainty and inferences are ubiquitous in natural language, and valid inferences over natural-language expressions of uncertainty are necessary for not only everyday conversations but also for high-stakes domains such as medicine and law. While large language models are increasingly evaluated on logical reasoning tasks, disentangling principled, symbolic reasoning from clever surface-level pattern matching is fraught with difficulty. We introduce a benchmark for reasoning over probability operators--inference over sentences with gradable epistemic modals (e.g., probably, might, must) containing 14,320 procedurally-generated English prompts across fifteen inference templates, systematically varying question form, negation strategy, and surface content. Evaluating 29 models, we find that most show answer biases independent of the logical form, a systematic preference for Yes or No. We summarize this with a competence floor: the worse of a model's accuracy on Yes-correct and No-correct items. Only 9 of 29 models exceed random chance. We also test variations in question form, verb phrases/activity, and both the gender and origin of names used in the prompts, finding biases across every axis. 

---
# ECG-InterpBench: Benchmarking the Interpretability of ECG Foundation Models with Matched-Scale Sparse Autoencoders 

**Authors**: Yixuan Duan, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27404)  

**Abstract**: Existing benchmarks for electrocardiogram foundation models primarily evaluate downstream predictive performance, providing limited insight into whether their internal representations can be faithfully decomposed, clinically interpreted, or reproduced across independent analyses. We introduce ECG-InterpBench, a benchmark designed to systematically evaluate the interpretability of ECG foundation-model representations. ECG-InterpBench uses sparse autoencoders as standardized measurement instruments and matches their capacity across models to enable controlled comparisons. We evaluate six frozen ECG foundation models across five standardized encoder depths, five matched dictionary widths, and three random seeds, producing a 450-cell interpretability atlas comprising 75 exactly matched six-model comparison blocks. The benchmark evaluates complementary dimensions of representation interpretability, including sparse reconstruction fidelity, single-feature accessibility and coverage of 49 clinically meaningful ECG measurements, and cross-seed feature reproducibility. The evaluation further quantifies patient-sampling uncertainty, depth- and seed-dependent variation, and sensitivity to the sparsity parameterization. The benchmark reveals that ECG foundation models exhibit distinct interpretability profiles. A matched replication on MIMIC-IV-ECG confirms that reconstruction fidelity and clinical accessibility identify different leading models. The benchmark is accompanied by executable evaluation code, standardized manifests, cell-level metrics, and reproducibility audits. ECG-InterpBench complements performance-centered ECG benchmarks by providing a capacity-controlled and reproducible framework for comparing ECG foundation models across distinct dimensions of representation interpretability. 

---
# AHA-Memes: A Fine-Grained Multimodal Benchmark for Understanding Hate in Arabic Memes 

**Authors**: Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Abul Hasnat, Md. Rafiul Biswas, Wajdi Zaghouani, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2607.27393)  

**Abstract**: Hateful memes are a growing form of multimodal online harm, where hostile intent is often conveyed through the joint interpretation of images, text, cultural references, and implicit targets. While hateful meme detection has advanced in high-resource languages, Arabic remains underexplored, with existing meme resources focusing mainly on propaganda or coarse harmful-content labels. We introduce AHA-Memes (Arabic HAteful Memes), which is, to our knowledge, the first large-scale Arabic hateful meme benchmark with fine-grained, multi-label annotations. The dataset includes 5K manually annotated memes using a taxonomy that captures hate types, i.e., attack strategies. We further provide ~66K silver-labeled memes to support future studies. We benchmark text-only, image-only, and late-fusion multimodal models, as well as few-shot in-context learning (ICL) and open- and closed-weight Vision-Language Models (VLMs) under zero-shot and fine-tuning settings. Our results establish strong baselines and highlight key challenges in culturally grounded Arabic hateful meme detection. We release the dataset, annotation guidelines, and evaluation scripts to support future research. WARNING: This paper contains examples that may be disturbing to readers. 

---
# FunL2O: LLM-Guided Feature Function Design for Learning to Optimize 

**Authors**: Bingheng Li, Junyang Cai, Yupeng Zhang, Bistra Dilkina, Jayant Kalagnanam, Dzung T. Phan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27389)  

**Abstract**: Learning-to-optimize (L2O) methods accelerate repeated optimization by training models to predict solutions, warm starts, branching decisions, or other forms of solver guidance. A critical yet largely overlooked component of these pipelines is the feature function that maps problem instances to inputs for machine learning models. Existing L2O methods typically rely on hand-crafted features, making representation design manual and largely fixed across domains. We introduce FunL2O, the first unified framework for automating feature design through LLM-driven program evolution for L2O. In a FunSearch-style loop, an LLM proposes executable feature functions, while a fixed evaluation process retrains the original L2O model and measures downstream optimization performance. We evaluate FunL2O on linear and quadratic programming tasks involving solution prediction and warm-starting, as well as on mixed-integer optimization tasks using GNN-guided backdoor branching and Predict-and-Search. Across continuous and discrete optimization tasks and four LLMs, the evolved features consistently outperform hand-crafted representations. These results establish LLM-driven feature evolution as a general and effective approach to automating representation design in L2O. 

---
# Same Facts, Different Diagnosis: Measuring and Mitigating Narrative Anchoring in Clinical Language Models 

**Authors**: Prabhjot Singh, Pritam Deka, Vijay Chennareddy  

**Link**: [PDF](https://arxiv.org/pdf/2607.27384)  

**Abstract**: Large language models used for clinical diagnostic reasoning are sensitive to sociolinguistic register, not just clinical content. We term this failure mode Narrative Anchoring: identical clinical facts expressed in different registers cause diagnostic outputs to diverge. Unlike prior demographic-bias work, which manipulates explicit identity tokens such as race or income, our benchmark isolates register as the sole channel of variation, with no demographic marker present in any form. We construct a dataset of 1,000 USMLE clinical vignettes, each rewritten into three sociolinguistically distinct personas under an independently audited fact-preservation guarantee, verified by a separate model that never sees the generation prompt. Across seven language models spanning three architecture families and scales, Narrative Anchoring is statistically significant under direct prompting in every model tested, with a Narrative Anchoring Gap of 0.064 to 0.151. Chain-of-thought reasoning and explicit debiasing instructions reduce the bias only partially, and their apparent gains are frequently confounded by accuracy collapse. We introduce NarrativeShield, a three-agent pipeline that structurally extracts and verifies clinical facts before diagnostic reasoning begins, reducing the Narrative Anchoring Gap to near-zero ($-0.004$ to $0.037$) and achieving the lowest rate of severely unstable decisions (DSS $<$ 0.8) of any method across all models, at a modest and mechanistically expected accuracy cost for most models. A stress test using a non-instruction-tuned base model shows that executing a debiasing intervention at all is gated by zero-shot instruction-following ability, not prompt content alone. We release our dataset, human-validated for fact preservation, as a standalone resource for studying register-based clinical bias. 

---
# VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation via an Agentic Dual-Engine System 

**Authors**: Haodong Li, Tianfei Ren, Xiaoxiao Ma, Chunmei Qing, Zhen Fang, Sipeng He, Ziyu Guo, Haoyu Wu, Juanxi Tian, Yihang Zou, Ruichuan An, Dongzhi Jiang, Boxue Yang, Ji Xie, Xu Huang, Wenhao Yan, Jialv Zou, Zhengrong Yue, Yaxin Luo, Xiaotong Li, Yuzhu Wang, Junyan Ye, Jinjing Zhao, Zehui Chen, Lin Chen, Renye Yan, Feng Zhao, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2607.27380)  

**Abstract**: Text-to-video models have achieved remarkable visual quality, yet they still struggle to generate physically consistent dynamics because the temporal evolution of a scene must be inferred implicitly from a highly compressed text prompt. Existing chain-of-thought approaches introduce intermediate plans or visual states, but these representations are typically non-executable or temporally sparse, limiting their ability to instantiate and control the complete spatiotemporal process. To address this limitation, we introduce VideoCoCo, an agentic dual-engine framework in which executable Blender code serves as a process-level chain of thought. Given a text prompt, a coding agent synthesizes a Blender program that explicitly specifies the scene and its temporal evolution. The executable simulation engine runs the program to produce a deterministic spatiotemporal draft, which is subsequently transformed into a photorealistic video by a generative video engine through draft-conditioned editing. This decomposition separates process-level reasoning from high-fidelity visual realization. To adapt the video editor to simulated drafts, we construct VideoCoCo-3K, a curated dataset of draft-instruction-target triplets. VideoCoCo improves the OmniWeaving baseline from 0.475 to 0.558 on PhyGenBench and from 52.18 to 77.88 on VBench-2.0, achieving the best average score on both benchmarks. These results demonstrate that executable code provides an effective, controllable, and inspectable intermediate representation for physically consistent video generation. 

---
# RoguePrompt: Dual-Layer Encoding for Self-Reconstruction to Circumvent LLM Moderation 

**Authors**: Benyamin Tafreshian, Prathamesh Dhake  

**Link**: [PDF](https://arxiv.org/pdf/2607.27373)  

**Abstract**: Large language models (LLMs) are becoming increasingly integrated into mainstream development platforms and daily technological workflows, typically behind moderation and safety controls. Despite these controls, preventing prompt-based policy evasion remains challenging, and adversaries continue to "jailbreak" LLMs by crafting prompts that circumvent implemented safety mechanisms. Prior work has established cipher-mediated interaction, code-embedded decryption, prompt decomposition and reconstruction, and layered custom encryption as viable attack primitives. However, reported evaluations generally collapse visible acceptance, successful recovery of the concealed request, and subsequent execution into an aggregate attack-success outcome. This leaves limited evidence about where multistage prompt-transformation attacks fail within an observable black-box interaction. This paper introduces RoguePrompt, a jailbreak pipeline that partitions a forbidden prompt and applies two nested encodings, Vigenere followed by ROT13, along with natural-language reconstruction instructions. RoguePrompt was developed and evaluated under a black-box threat model, with only API or user-interface access to the hosted models, and was tested on 313 real-world, hard-rejected prompts. Success was measured in terms of moderation bypass, instruction reconstruction, and execution when the relevant stage exceeded its automated criterion. RoguePrompt achieved average rates of 93.93% for filter bypass, 79.02% for reconstruction, and 70.18% for execution. These results demonstrate the effectiveness of layered prompt encoding while providing stage-level evidence of where multistage jailbreaks fail during moderation bypass, instruction reconstruction, and execution. 

---
# Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation 

**Authors**: Alexi Gladstone, Heng Ji, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.27372)  

**Abstract**: The deep learning revolution, kicked off by AlexNet, taught us that end-to-end training beats decomposing a problem into hand-designed stages. Generative modeling, however, has remained the exception-despite generative models being remarkably capable, they are still not trained end-to-end. This is because, at its core, generative modeling is about handling distributions with many modes, and existing scalable approaches handle this the same way, by factoring the generation procedure, which prevents end-to-end generation. In this work, we introduce Explorative Modeling, a new paradigm that instead factors the training loop, exploring K candidate matches between model generations and data, and training on the best, so predictions commit to modes rather than blurring them. We find Explorative Models (XMs) useful in two settings. First, increasing exploration adds a third pretraining axis beyond parameters and data for existing generative models-where scaling exploration monotonically improves performance across both continuous and discrete domains (images, video, and language). Notably, gains from exploration increase with scale, climbing from 7% to 36% as data scales and from 13% to 23% as models grow, with efficiency gains more than doubling at 3x the compute. Concretely, exploration improves FLOP efficiency by 4.1x, sample efficiency by 6.2x, parameter efficiency by 47%, lifts the strongest of image-generation recipes to a near-state-of-the-art 1.43 FID on ImageNet without guidance, enables scaling how end-to-end existing models are, and unlocks scaling generalization. Second, XMs enable end-to-end reconstructive generative modeling, matching diffusion on control tasks with 16-256x fewer inference steps. Together, these results establish XMs as both a new pretraining axis for existing generative models and a standalone end-to-end generative modeling paradigm. 

---
# Benchmarking the Residual: What Long-Horizon Evaluations Add Beyond Matched Short-Task Performance 

**Authors**: Chao Peng, Zhiheng Lyu, Peijie Dong, Hande Dong, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.27283)  

**Abstract**: Long-horizon benchmarks often show that agents fail more as tasks become longer. This observation is useful for deployment, but it does not by itself explain why failure occurs. More stages create more opportunities for ordinary errors to compound; longer tasks may also contain harder individual decisions or become harder as conversation history, tool outputs, and environment changes accumulate. We use trajectory-induced degradation to mean this last possibility: earlier execution makes later work harder. When the harmful accumulation is specifically the text visible to the model, it is often called context rot. In this position paper, we argue that to claim a "long-horizon failure", benchmarks must compare actual full-task success against a baseline prediction built from short, individual stages. We call the log-ratio between this prediction and actual success the horizon residual. The comparison must use the same agent configuration and specify in advance how stages, checkpoints, information, and budgets will be chosen. The residual shows that the full rollout differs from the chosen baseline; targeted experiments are still needed to explain why. 

---
# Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents 

**Authors**: Jiwon Jang, Kisu Yang, Heuiseok Lim, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.27275)  

**Abstract**: Post-training quantization to 4-bit weights is widely reported to be nearly lossless. We test this claim for multi-turn, tool-calling agents, where it now matters most. On $\tau^2$-bench, across two open-weight model families in dense and MoE variants and two domains (eight cells, 456 episodes each, at 16-, 8-, and 4-bit weights), quantization indeed looks free on the standard metric. No cell shows a score change that survives multiple-comparison correction, and in the cell that carries the largest process damage, equivalence testing bounds the change within $\pm$7.5 points. The process tells a different story. Quantization amplifies the failure the model already exhibits at full precision (tool-name hallucination in telecom, with the same directional trend in retail entity errors) by up to 2.5$\times$ in volume (+17.6 points per task), while creating essentially no new failures. The failure set is the same at every precision (rank correlation $\geq$ 0.94, 0.18% novel events). The score stays flat because the benchmark's ten-error budget absorbs the extra failures. Shrinking the budget to two errors re-exposes a score gap of 17 points, and it does so only in the one cell where quantization added error volume, exactly as the masking account predicts. A targeted error-repair prompt, run for five telecom models at every precision, removes the damage exactly and only where it lives. Both diagnostics, the per-channel error rate and success under a shrinking budget, come from logs benchmarks already collect; we suggest reporting them alongside task reward. 

---
# FAVA: Formal Authorization for Verified Agents with Evidence-Backed Permission Graphs 

**Authors**: Yifan Zhang, Xinkui Zhao, Sai Liu, Hengxuan Lou, Guanjie Cheng, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27267)  

**Abstract**: Large language model (LLM) agents autonomously interleave semantic reasoning with complex system operations. In these dynamic environments, static tool-level permissions are fundamentally insufficient; safe authorization is highly context-dependent and heavily reliant on evolving runtime states and data flows. We present FAVA (Formal Authorization for Verified Agents), a permission-carrying authorization framework for agent execution. FAVA utilizes an LLM-guided Permission Intermediate Representation (IR) to translate ambiguous natural-language tasks into structured constraints. A deterministic lowering pass then converts this IR into an evidence-backed permission graph that explicitly tracks data flows, dependencies, and contextual labels. To provide strict security guarantees, a Satisfiability Modulo Theories (SMT) authorizer mathematically verifies the current graph against security policies before any effectful action executes. A runtime gateway then enforces the solver's result, either authorizing the execution or intercepting it with a precise counterexample. We evaluate FAVA across OpenAgentSafety, OctoBench, and ActPlane scenarios. Our evaluation demonstrates that FAVA achieves a 90.5% Decision Compliance Rate (DCR) over the aggregate dataset, successfully intercepting dynamic violating traces in the evaluated trace-conditioned scenarios. 

---
# LLM-Guided Initialization for Accelerated Hybrid Quantum-Classical Medical Image Classification 

**Authors**: Riza Alaudin Syah, Irwan Alnarus Kautsar, Haza Nuzly Bin Abdull Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2607.27262)  

**Abstract**: Variational quantum algorithms often encounter barren plateaus, where cost gradients decay rapidly with increasing circuit depth, undermining the trainability of parameterized quantum circuits. This paper evaluates AdaInit (Adaptive Initialization), proposed by Zhuang and Cunningham, which uses large language models to propose initial parameters for quantum neural networks. We study a simplified single-query AdaInit variant paired with GPU-accelerated simulation in NVIDIA CUDA-Q and apply it to binary classification on the DMR-IR mammography dataset. AdaInit delivers 14.6 times higher gradient variance at initialization than random initialization (0.0095 vs. 0.0006), producing 160 times faster convergence (1.1s vs. 176 s) while maintaining the same classification accuracy of 61.4 percent. We provide theoretical analysis grounded in the geometry of parameterized circuit landscapes and show empirically that LLM-guided initialization places the optimizer in trainable regions of parameter space. Beyond performance, our results indicate that a single LLM query can yield informative parameters without iterative refinement, suggesting a low-overhead path to improved trainability. The findings validate AdaInit in a medical imaging setting and demonstrate its compatibility with GPU-accelerated quantum backends for practical speedups. 

---
# Recursive transformers for semiconductor thermo-mechanical reliability 

**Authors**: Kart-leong Lim  

**Link**: [PDF](https://arxiv.org/pdf/2607.27251)  

**Abstract**: Transformer-based surrogate models are increasingly used to replace expensive first-principles simulation in engineering design. But conventional transformer architectures are often over parameterized for the small, low-dimensional datasets typical of engineering design spaces, where large simulation data is expensive to generate. Under these conditions, excess parameter capacity leads to overfitting rather than improved accuracy, while also incurring unnecessary memory and compute overhead. This motivates a shift towards architectures that focus on additional compute rather than additional learnable parameters. This paper presents a hardware-aware evaluation of three recursive transformer paradigms for surrogate thermo-mechanical analysis of advanced packages: a)Tiny Recursive Model, b) our proposed Depth Recursive transformer, c) and a simple recursive transformer. We systematically compare their predictive performance (Recall, Mean Reciprocal Rank), parameter count, computational complexity (FLOPs), providing practical design guidelines for selecting recursive transformer architectures under resource-constrained scenarios. We validate this principle on two low-dimensional engineering prediction tasks: 1) thermo-mechanical reliability analysis of advanced semiconductor packages, where stress and warpage from thermal cycling must be evaluated repeatedly across a design-of-experiments sweep under costly finite element analysis (FEA). 2) Laplace PDE iterative numerical solver for capacitance field. Overall, recursive weight-sharing transformers provide an effective and generalizable trade-off between prediction accuracy, parameter efficiency, and computational cost for small data engineering surrogate modeling. 

---
# Do Context Files Help Coding Agents? A Two-Agent Ablation Study on Real Repositories 

**Authors**: Prakhar Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2607.27250)  

**Abstract**: Persistent context files (this http URL, this http URL) are standard practice for guiding AI coding agents, yet evidence for their effectiveness is contradictory. We present a controlled ablation of context-injection strategy across two frontier agents (Claude Code and Codex), 17 real tasks from 3 repositories (15 shared + 2 Codex-only), and 288 evaluated runs with gold-test evaluation. Context strategy does not measurably move correctness on either agent (bounded to <=10-15pp via equivalence testing). A failure-mode triage reveals why: agents fail on implementation skill---feature design, pattern selection, exact wiring---not missing repository knowledge that a context file could supply; a manipulation probe confirms the real this http URL never converts a near-miss to a pass on either agent. We further show that borderline task difficulty is agent-specific (Spearman rho=0.75), offering a candidate explanation for prior contradictions: single-agent studies draw tasks from different agents' informative bands. We release all code, data, and analysis. 

---
# Enhancing Law-Enforcement Audio Transcription: A LoRA-Based Adaptation of Whisper for BWC Footage 

**Authors**: Vivek Senthil, Zhiqiang Tao, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2607.27245)  

**Abstract**: Modern policing faces a "visibility paradox" where law enforcement agencies possess petabytes of Body-Worn Camera (BWC) footage that remains largely unutilized for accountability or systemic review due to the prohibitive labor costs of manual transcription. This research presents a framework for adapting the OpenAI Whisper architecture to the unique acoustic and linguistic challenges of the policing environment. By employing Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA), we address the significant performance degradation observed in zero-shot models when confronted with high-stress scenarios, sirens, and radio interference. Crucially, we demonstrate that this adaptation is feasible on consumer-grade hardware (Acer Nitro local machine with NVIDIA 4GB GTX GPU) using 8-bit quantization and gradient checkpointing. We further integrate these transcriptions into a symbolic reasoning pipeline using a domain-specific ontology to transform raw audio into evidence-linked incident graphs, achieving a 93.7% lexicon mapping rate for the advancement of procedural justice and transparency. 

---
# Sympathetic Framing: Evaluating AI Alignment across Sociodemographic Groups 

**Authors**: Haran Shani-Narkiss, Michael Fire, Oren Tsur  

**Link**: [PDF](https://arxiv.org/pdf/2607.27232)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping how we consume information and form our worldview. This raises concerns beyond bias in AI: do LLMs grasp the emotional nuances conveyed via textual framing? In this work, we empirically evaluate how well an array of LLMs aligns with human emotional perception. Considering news headlines covering political and geopolitical conflicts, both human participants (n = 3011, a representative sample of the U.K. adult population, via a YouGov survey) and seven LLMs answered whether headlines evoked sympathy for a specified side in a conflict. We find that the correlation between AI and human evaluations varies across models, ranging from very high (0.789, GPT-5.2) to medium (0.4 ,Mistral Large 2512). Crucially, the leading models are broadly aligned with human judgments across all demographic subgroups, including age, gender, level of education, prior geopolitical knowledge, and participants' predispositions regarding the conflict, although there are statistically significant differences between groups. This research, with its robust design and large, demographically diverse dataset, offers the most comprehensive evaluation of LLMs' comprehension of news framing to date. Findings highlight an important, often-ignored aspect of differential alignment: even when aggregate performance is high, AI alignment is not universal -- it may correspond differently with demographic features and cultural norms. Considering or ignoring the need for differential alignment may therefore have significant implications for the development of ethical and useful AI systems. 

---
# AI-assisted pre-review of open-source software submissions: an experience report from BOSC 2026 

**Authors**: Tazro Ohta, Nomi L. Harris, Seth Carbon  

**Link**: [PDF](https://arxiv.org/pdf/2607.27228)  

**Abstract**: Most conferences rely on peer-review of submissions, but as generative AI makes it easier than ever to prepare submission materials, some conferences are seeing an overwhelming surge of submissions. We wanted to see if generative AI could help our conference's volunteer reviewers by pre-reviewing abstracts for certain criteria. The Bioinformatics Open Source Conference (BOSC) was well-positioned to experiment with this, as we already had a detailed rubric used by reviewers to evaluate submitted abstracts on multiple criteria, including openness (public availability of the code or other content associated with the project), valid open source license, and "runnability" (how easy it is to download, build, and run the project - an important measure of reusability). For BOSC 2026, we built bosc-pre-review, an agentic skill that assessed six review criteria, and Runabilly, which builds and tests each project in a disposable Docker container for safety. The AI only gathered evidence to present to the reviewers; humans made every decision regarding the acceptance of the abstracts. After the review period, we surveyed the reviewers to determine how useful they found the pre-review. Most of those who responded said they found it useful, but they preferred to check the AI's conclusions against their own, rather than accepting the AI results unquestioningly. 

---
# Estimating Treatment Effects for Depression in Longitudinal Therapy Switching Settings 

**Authors**: Xinyu Qin, Martin Katzman, Alexandria Greifenberger, Elssa Toumeh, Sachinthya Lokuge, Tia Sternat, Ruiheng Yu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27214)  

**Abstract**: Depression treatment often requires switching medications due to inadequate response or adverse effects. Estimating individualized treatment effects in this setting is challenging because treatment assignment is confounded by patient characteristics, switching induces time-varying selection, and counterfactual outcomes are not observed in follow-up data. Using a proprietary longitudinal major depressive disorder (MDD) clinical trial dataset, we formulate a next-visit counterfactual prediction task to estimate Hamilton Depression Rating Scale (HAMD-17) total scores under alternative treatments. We benchmark 8 estimators, including meta-learners, residual-based methods, and tree-based approaches. Causal Forest (CF) demonstrates the most favorable and consistent performance across all criteria. Our analysis shows that symptom benefits concentrate in specific switch directions, with dose intensification being generally beneficial. Notably, we identify a counterintuitive exception where a lower-intensity regimen outperforms a higher-intensity alternative for specific patient subsets. While crude observational comparisons substantially overstate gains, confounding-adjusted estimates yield modest, actionable magnitudes. These findings provide prospectively testable candidates for clinical decision support in depression care. 

---
# Prompt Chaining in Practice: A Case Study in Automated Scholarly Report Generation 

**Authors**: Andrei Lazarev  

**Link**: [PDF](https://arxiv.org/pdf/2607.27210)  

**Abstract**: The exponential growth of scholarly publications requires automated tools for effective information synthesis. However, simple, single-shot prompting methods often lack the reliability and quality required for complex synthesis tasks. This paper introduces and empirically evaluates a multi-stage prompt chaining methodology as a more reliable architectural pattern for such tasks. This approach is implemented in our system, AI SciBrief, which automatically generates scholarly digests. We conducted a comparative experiment, measuring the performance of our prompt chaining method against a carefully optimized single-shot baseline. Both systems were evaluated against a human-authored "gold standard" report for the "Education" domain. The results demonstrate a significant difference in reliability: our prompt chaining method achieved a 100% success rate, whereas the optimized baseline failed in 50% of its runs. In terms of quality, the proposed method also demonstrated a clear advantage, achieving a superior ROUGE-L F1-score (0.507 vs. 0.486), driven primarily by higher precision. We conclude that prompt chaining is a more dependable and effective engineering approach for complex, multi-step generative tasks, significantly mitigating the risks of failure and inconsistency inherent in monolithic prompts. 

---
# Reviewer Scores Are Not Comparable Across Research Areas in ML Peer Review 

**Authors**: Binyan Xu, Fan Yang, Xilin Dai, Kehuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27209)  

**Abstract**: Peer review at ML conferences increasingly relies on reviewer scores as the primary decision instrument. As submissions have scaled from thousands to tens of thousands per year, no systematic audit has examined whether this instrument functions uniformly across research areas, or whether acceptance outcomes are in practice shaped by forces that reviewer scores neither capture nor control. This position paper argues that acceptance outcomes are shaped by forces beyond reviewer scores, and that the underlying cause is a measurement design failure, not individual bias. When a fixed numerical scale aggregates quality judgments across communities with structurally non-uniform reviewer pools, absolute scores become incomparable across areas, and area chairs must substitute community priors for score-based decisions. Using ICLR 2021--2026 data covering 50,289 papers across 219 research topics, we show that at any given reviewer score, a paper's acceptance probability varies by up to 8x depending on its topic. We rule out scoring culture, expert reviewer standards, rational area chair reweighting, and quality dilution as alternative explanations. We call on program committees to adopt inherently calibrated review signals and publish topic-stratified, score-conditional acceptance rates as a first-class fairness metric. 

---
# MagicSelector: Joint Optimization for Agent Tool Selection via Counterfactual Decomposition and Progressive Reranking 

**Authors**: HONOR Agentic Search Team, Zhengzong Chen, Lei Tang, Lijun Liu, Chuandi Jiang, Fan Yang, Keyun Chu, Chu Zhao, Shihao Liu, Minghang Li, Bo Liang, Can Wen, Hailong Wu, Jingnan Ju, Mian Liu, Nengbin Zhang, Peiqiang Wang, Penghe Nie, Qinhui Gu, Sijia Lv, Siqi Chen, Wei Zhang, Yang Xu, Yuhao Qian, Yuxiang Zhang, Zeng Cheng, Zhen Wang, Zuan Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17751)  

**Abstract**: We present MagicSelector, a joint optimization framework integrating Counterfactual task decomposition, Progressive reranking, and Dynamic Top-K, designed to address the fundamental challenges of tool retrieval in agents. MagicSelector is a specialized framework capable of translating ambiguous user instructions into executable atomic subtasks and guiding high-precision tool retrieval, effectively mitigating redundant noise and severe context distraction in out-of-domain (OOD) scenarios. We empower MagicSelector with these capabilities through three key contributions: (1) a preference-guided counterfactual task decomposition mechanism that utilizes a counterfactual reward to quantify the marginal causal gain of decomposition on retrieval ranking, effectively imposing fine-grained structural supervision on logical coherence; (2) a progressive tool reranking method driven by self-distillation hard negative mining, which optimizes both point-wise and list-wise relevance to enhance fine-grained discrimination among highly similar tools; and (3) a dual semantic boundary-aware dynamic Top-K strategy that adaptively monitors reranking score cliffs and inter-tool semantic shifts to dynamically truncate the candidate list, maximizing relevant tool recall while filtering long-tail noise. Evaluated on MTDTool, the first task decomposition benchmark we constructed tailored for mobile multi-turn interactions with process-level annotations, MagicSelector yields promising performance. Extensive experiments demonstrate that MagicSelector significantly outperforms state-of-the-art methods in terms of tool retrieval accuracy, OOD generalization capability, and overall token efficiency, thereby demonstrating the effectiveness of our proposed framework. 

---
