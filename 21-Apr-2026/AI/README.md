# MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval 

**Authors**: Shaden Alshammari, Kevin Wen, Abrar Zainal, Mark Hamilton, Navid Safaei, Sultan Albarakati, William T. Freeman, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2604.18584)  

**Abstract**: Mathematical problem solving remains a challenging test of reasoning for large language and multimodal models, yet existing benchmarks are limited in size, language coverage, and task diversity. We introduce MathNet, a high-quality, large-scale, multimodal, and multilingual dataset of Olympiad-level math problems together with a benchmark for evaluating mathematical reasoning in generative models and mathematical retrieval in embedding-based systems. MathNet spans 47 countries, 17 languages, and two decades of competitions, comprising 30,676 expert-authored problems with solutions across diverse domains. In addition to the core dataset, we construct a retrieval benchmark consisting of mathematically equivalent and structurally similar problem pairs curated by human experts.
MathNet supports three tasks: (i) Problem Solving, (ii) Math-Aware Retrieval, and (iii) Retrieval-Augmented Problem Solving. Experimental results show that even state-of-the-art reasoning models (78.4% for Gemini-3.1-Pro and 69.3% for GPT-5) remain challenged, while embedding models struggle to retrieve equivalent problems. We further show that retrieval-augmented generation performance is highly sensitive to retrieval quality; for example, DeepSeek-V3.2-Speciale achieves gains of up to 12%, obtaining the highest scores on the benchmark. MathNet provides the largest high-quality Olympiad dataset together with the first benchmark for evaluating mathematical problem retrieval, and we publicly release both the dataset and benchmark at this https URL. 

---
# Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs 

**Authors**: Kevin Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2604.18576)  

**Abstract**: We present BLF (Bayesian Linguistic Forecaster), an agentic system for binary forecasting that achieves state-of-the-art performance on the ForecastBench benchmark. The system is built on three ideas. (1) A Bayesian linguistic belief state: a semi-structured representation combining numerical probability estimates with natural-language evidence summaries, updated by the LLM at each step of an iterative tool-use loop. This contrasts with the common approach of appending all retrieved evidence to an ever-growing context. (2) Hierarchical multi-trial aggregation: running $K$ independent trials and combining them using logit-space shrinkage with a data-dependent prior. (3) Hierarchical calibration: Platt scaling with a hierarchical prior, which avoids over-shrinking extreme predictions for sources with skewed base rates.
On 400 backtesting questions from the ForecastBench leaderboard, BLF outperforms all the top public methods, including Cassi, GPT-5, Grok~4.20, and Foresight-32B. Ablation studies show that the structured belief state is as impactful as web search access, and that shrinkage aggregation and hierarchical calibration each provide significant additional gains.
In addition, we develop a robust back-testing framework with a leakage rate below 1.5\%, and use rigorous statistical methodology to compare different methods while controlling for various sources of noise. 

---
# Benchmarking System Dynamics AI Assistants: Cloud Versus Local LLMs on CLD Extraction and Discussion 

**Authors**: Terry Leitch  

**Link**: [PDF](https://arxiv.org/pdf/2604.18566)  

**Abstract**: We present a systematic evaluation of large language model families -- spanning both proprietary cloud APIs and locally-hosted open-source models -- on two purpose-built benchmarks for System Dynamics AI assistance: the \textbf{CLD Leaderboard} (53 tests, structured causal loop diagram extraction) and the \textbf{Discussion Leaderboard} (interactive model discussion, feedback explanation, and model building coaching).
On CLD extraction, cloud models achieve 77--89\% overall pass rates; the best local model reaches 77\% (Kimi~K2.5~GGUF~Q3, zero-shot engine), matching mid-tier cloud performance. On Discussion, the best local models achieve 50--100\% on model building steps and 47--75\% on feedback explanation, but only 0--50\% on error fixing -- a category dominated by long-context prompts that expose memory limits in local deployments.
A central contribution of this paper is a systematic analysis of \textit{model type effects} on performance: we compare reasoning vs.\ instruction-tuned architectures, GGUF (this http URL) vs.\ MLX (mlx\_lm) backends, and quantization levels (Q3 / Q4\_K\_M / MLX-3bit / MLX-4bit / MLX-6bit) across the same underlying model families. We find that backend choice has larger practical impact than quantization level: mlx\_lm does not enforce JSON schema constraints, requiring explicit prompt-level JSON instructions, while this http URL grammar-constrained sampling handles JSON reliably but causes indefinite generation on long-context prompts for dense models.
We document the full parameter sweep ($t$, $p$, $k$) for all local models, cleaned timing data (stuck requests excluded), and a practitioner guide for running 671B--123B parameter models on Apple~Silicon. 

---
# ClawEnvKit: Automatic Environment Generation for Claw-Like Agents 

**Authors**: Xirui Li, Ming Li, Derry Xu, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18543)  

**Abstract**: Constructing environments for training and evaluating claw-like agents remains a manual, human-intensive process that does not scale. We argue that what is needed is not just a dataset, but an automated pipeline capable of generating diverse, verified environments on demand. To this end, we introduce ClawEnvKit, an autonomous generation pipeline that instantiates this formalism from natural language descriptions. The pipeline comprises three modules: (1) a parser that extracts structured generation parameters from natural language input; (2) a generator that produces the task specification, tool interface, and scoring configuration; and (3) a validator that enforces feasibility, diversity, structural validity, and internal consistency across the generated environments. Using ClawEnvKit, we construct Auto-ClawEval, the first large-scale benchmark for claw-like agents, comprising 1,040 environments across 24 categories. Empirically, Auto-ClawEval matches or exceeds human-curated environments on coherence and clarity at 13,800x lower cost. Evaluated across 4 model families and 8 agent harness frameworks, we find that harness engineering boosts performance by up to 15.7 percentage points over a bare ReAct baseline, completion remains the primary axis of variation with no model saturating the benchmark, and automated generation enables evaluation at a scale previously infeasible. Beyond static benchmarking, ClawEnvKit enables live evaluation: users describe a desired capability in natural language and obtain a verified environment on demand, turning evaluation into a continuous, user-driven process. The same mechanism serves as an on-demand training environment generator, producing task distributions that adapt to an agent's current weaknesses rather than being bounded by existing user logs. 

---
# OGER: A Robust Offline-Guided Exploration Reward for Hybrid Reinforcement Learning 

**Authors**: Xinyu Ma, Mingzhou Xu, Xuebo Liu, Chang Jin, Qiang Wang, Derek F. Wong, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18530)  

**Abstract**: Recent advancements in Reinforcement Learning with Verifiable Rewards (RLVR) have significantly improved Large Language Model (LLM) reasoning, yet models often struggle to explore novel trajectories beyond their initial latent space. While offline teacher guidance and entropy-driven strategies have been proposed to address this, they often lack deep integration or are constrained by the model's inherent capacity. In this paper, we propose OGER, a novel framework that unifies offline teacher guidance and online reinforcement learning through a specialized reward modeling lens. OGER employs multi-teacher collaborative training and constructs an auxiliary exploration reward that leverages both offline trajectories and the model's own entropy to incentivize autonomous exploration. Extensive experiments across mathematical and general reasoning benchmarks demonstrate that OGER significantly outperforms competitive baselines, achieving substantial gains in mathematical reasoning while maintaining robust generalization to out-of-domain tasks. We provide a comprehensive analysis of training dynamics and conduct detailed ablation studies to validate the effectiveness of our entropy-aware reward modulation. Our code is available at this https URL. 

---
# LLM Safety From Within: Detecting Harmful Content with Internal Representations 

**Authors**: Difan Jiao, Yilun Liu, Ye Yuan, Zhenwei Tang, Linfeng Du, Haolun Wu, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2604.18519)  

**Abstract**: Guard models are widely used to detect harmful content in user prompts and LLM responses. However, state-of-the-art guard models rely solely on terminal-layer representations and overlook the rich safety-relevant features distributed across internal layers. We present SIREN, a lightweight guard model that harnesses these internal features. By identifying safety neurons via linear probing and combining them through an adaptive layer-weighted strategy, SIREN builds a harmfulness detector from LLM internals without modifying the underlying model. Our comprehensive evaluation shows that SIREN substantially outperforms state-of-the-art open-source guard models across multiple benchmarks while using 250 times fewer trainable parameters. Moreover, SIREN exhibits superior generalization to unseen benchmarks, naturally enables real-time streaming detection, and significantly improves inference efficiency compared to generative guard models. Overall, our results highlight LLM internal states as a promising foundation for practical, high-performance harmfulness detection. 

---
# WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware Write-Time Reconciliation 

**Authors**: Harish Santhanalakshmi Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18478)  

**Abstract**: Persistent memory is the bottleneck separating stateless chatbots from long-running agentic systems. Retrieval-augmented generation (RAG) over flat vector stores fragments facts into chunks, loses cross-session identity, and has no first-class notion of supersession or contradiction. Recent bitemporal knowledge-graph systems (Graphiti, Memento, Hydra DB) add typed edges and valid-time metadata, but the graph itself remains flat: no recursive composition, no content-addressed invariants on nodes, and edge types carry no behavior beyond a label. We present WorldDB, a memory engine built on three commitments: (i) every node is a world -- a container with its own interior subgraph, ontology scope, and composed embedding, recursive to arbitrary depth; (ii) nodes are content-addressed and immutable, so any edit produces a new hash at the node and every ancestor, giving a Merkle-style audit trail for free; (iii) edges are write-time programs -- each edge type ships on_insert/on_delete/on_query_rewrite handlers (supersession closes validity, contradicts preserves both sides, same_as stages a merge proposal), so no raw append path exists. On LongMemEval-s (500 questions, ~115k-token conversational stacks), WorldDB with Claude Opus 4.7 as answerer achieves 96.40% overall / 97.11% task-averaged accuracy, a +5.61pp improvement over the previously reported Hydra DB state-of-the-art (90.79%) and +11.20pp over Supermemory (85.20%), with perfect single-session-assistant recall and robust performance on temporal reasoning (96.24%), knowledge update (98.72%), and preference synthesis (96.67%). Ablations show that the engine's graph layer -- resolver-unified entities and typed refers_to edges -- contributes +7.0pp task-averaged independently of the underlying answerer. 

---
# A Generalized Synthetic Control Method for Baseline Estimation in Demand Response Services 

**Authors**: Jonas Sievers, Mardavij Roozbehani  

**Link**: [PDF](https://arxiv.org/pdf/2604.18469)  

**Abstract**: Baseline estimation is critical to Demand Response (DR) settlement in electricity markets, yet existing machine learning methods remain limited in predictive performance, while methodologies from causal inference and counterfactual prediction are still underutilized in this domain. We introduce a Generalized Synthetic Control Method that builds on the classical Synthetic Control Method (SCM) from econometrics. While SCM provides a powerful framework for counterfactual estimation, classical SCM remains a static estimator: it fits the treated unit as a combination of contemporaneous donor units and therefore ignores predictable temporal structure in the residual error. We develop a generalized SCM framework that transforms baseline estimation into a dynamic counterfactual prediction problem by augmenting the donor representation with exogenous features, lagged treated load, and selected lagged donor signals. This enriched representation allows the estimator to capture autoregressive dependence, delayed donor-response patterns, and error-correction effects beyond the scope of standard SCM. The framework further accommodates nonlinear predictors when linear weighting is inadequate, with the greatest benefit arising in limited-data settings. Experiments on the Ausgrid smart-meter dataset show consistent improvements over classical SCM and strong benchmark methods, with the dominant performance gains driven by dynamic augmentation. 

---
# Using large language models for embodied planning introduces systematic safety risks 

**Authors**: Tao Zhang, Kaixian Qu, Zhibin Li, Jiajun Wu, Marco Hutter, Manling Li, Fan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18463)  

**Abstract**: Large language models are increasingly used as planners for robotic systems, yet how safely they plan remains an open question. To evaluate safe planning systematically, we introduce DESPITE, a benchmark of 12,279 tasks spanning physical and normative dangers with fully deterministic validation. Across 23 models, even near-perfect planning ability does not ensure safety: the best-planning model fails to produce a valid plan on only 0.4% of tasks but produces dangerous plans on 28.3%. Among 18 open-source models from 3B to 671B parameters, planning ability improves substantially with scale (0.4-99.3%) while safety awareness remains relatively flat (38-57%). We identify a multiplicative relationship between these two capacities, showing that larger models complete more tasks safely primarily through improved planning, not through better danger avoidance. Three proprietary reasoning models reach notably higher safety awareness (71-81%), while non-reasoning proprietary models and open-source reasoning models remain below 57%. As planning ability approaches saturation for frontier models, improving safety awareness becomes a central challenge for deploying language-model planners in robotic systems. 

---
# Six Llamas: Comparative Religious Ethics Through LoRA-Adapted Language Models 

**Authors**: Chad Coleman, W. Russell Neuman, Manan Shah, Ali Dasdan, Matthew Crispi, Morris Chiang, Zack Leitman, Mustafa Poonawala  

**Link**: [PDF](https://arxiv.org/pdf/2604.18404)  

**Abstract**: We present Six Llamas, a comparative study examining whether large language models fine-tuned on distinct religious corpora encode systematically different patterns of ethical reasoning. Six variants of Meta-Llama-3.1-8B are constructed: one unmodified control and five LoRA-adapted models trained exclusively on the sacred and theological texts of Christianity, Islam, Judaism, Hinduism, or Buddhism. All six models are probed with an identical battery of 17 standardized ethical prompts spanning moral dilemmas, game-theoretic scenarios, public policy questions, and moral-psychological self-assessments. To assess robustness and reproducibility, we implement a multi-temperature sampling design spanning ten temperature settings. We compute response consistency metrics, pairwise inter-model agreement rates, temperature sensitivity coefficients across four prompt domains, and run-to-run stability analyses.
Findings show that LoRA-adapted models produce ethical reasoning patterns that are (a) systematically differentiated from the base model, (b) consistent with the moral logics of their training traditions, (c) structured along interpretable dimensions in moral-philosophical space, (d) core ethical positions remain stable across temperature variations for high-consensus dilemmas. The Trolley Problem achieves 100% consistency across all models and temperatures, while (e) tradition-specific divergence intensifies at higher temperatures in morally contested domains, and (f) the base model exhibits the highest overall response consistency (mean 88.3%), suggesting LoRA adaptation introduces both tradition-specific signal and increased sampling sensitivity.
The study offers a proof-of-concept for the condensate comparative method using differentially trained language models as instruments for cultural and ethical analysis and identifies specific criteria for falsification and planned extensions. 

---
# Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes 

**Authors**: Justin Bauer, Thomas Walshe, Derek Pham, Harit Vishwakarma, Armin Parchami, Frederic Sala, Paroma Varma  

**Link**: [PDF](https://arxiv.org/pdf/2604.18381)  

**Abstract**: Fine-tuning Large Language Models (LLMs) typically relies on large quantities of high-quality annotated data, or questions with well-defined ground truth answers in the case of Reinforcement Learning with Verifiable Rewards (RLVR). While previous work has explored the benefits to model reasoning capabilities by scaling both data and compute used for RLVR, these results lack applicability in many real-world settings where annotated data and accessible compute may be scarce. In this work, we present a comprehensive empirical study of open-source Small Language Model (SLM) performance after RLVR in low data regimes. Across three novel datasets covering number counting problems, graph reasoning, and spatial reasoning, we characterize how model performance scales with dataset size, diversity, and complexity. We demonstrate that (1) procedural datasets allow for fine-grained evaluation and training dataset development with controllable properties (size, diversity, and complexity), (2) under RLVR, models trained on lower complexity tasks can generalize to higher complexity tasks, and (3) training on mixed complexity datasets is associated with the greatest benefits in low data regimes, providing up to 5x sample efficiency versus training on easy tasks. These findings inspire future work on the development of data scaling laws for RLVR and the use of procedural data generators to further understand effective data development for efficient LLM fine-tuning. 

---
# The implicated scientist: on the role of AI researchers in the development of weapons systems 

**Authors**: Alexandra Volokhova, Alex Hernandez-Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2604.18380)  

**Abstract**: Artificial intelligence (AI) technologies are increasingly used in modern weapons systems. Notably, these systems have recently been involved in mass killings and destruction at scale. Furthermore, there is currently a strong interest and competition among powerful players to accelerate the proliferation of weapons with automated or AI-based components, a phenomenon known as AI arms race. This competition poses a risk of causing even more deaths and devastation in the future, as well as increased power and wealth inequality. In this work, we aim to shed light on the role of AI researchers as implicated subjects in the harms caused by weapons enabled by AI technologies. We investigate and discuss the specifics of this implication and explore ways to transfigure this position of implication into one of differentiated, long-distance solidarity with the victims of technologically fortified injustices. 

---
# Training and Agentic Inference Strategies for LLM-based Manim Animation Generation 

**Authors**: Ravidu Suien Rammuni Silva, Ahmad Lotfi, Isibor Kennedy Ihianle, Golnaz Shahtahmassebi, Jordan J. Bird  

**Link**: [PDF](https://arxiv.org/pdf/2604.18364)  

**Abstract**: Generating programmatic animation using libraries such as Manim presents unique challenges for Large Language Models (LLMs), requiring spatial reasoning, temporal sequencing, and familiarity with domain-specific APIs that are underrepresented in general pre-training data. A systematic study of how training and inference strategies interact in this setting is lacking in current research. This study introduces ManimTrainer, a training pipeline that combines Supervised Fine-tuning (SFT) with Reinforcement Learning (RL) based Group Relative Policy Optimisation (GRPO) using a unified reward signal that fuses code and visual assessment signals, and ManimAgent, an inference pipeline featuring Renderer-in-the-loop (RITL) and API documentation-augmented RITL (RITL-DOC) strategies. Using these techniques, this study presents the first unified training and inference study for text-to-code-to-video transformation with Manim. It evaluates 17 open-source sub-30B LLMs across nine combinations of training and inference strategies using ManimBench. Results show that SFT generally improves code quality, while GRPO enhances visual outputs and increases the models' responsiveness to extrinsic signals during self-correction at inference time. The Qwen 3 Coder 30B model with GRPO and RITL-DOC achieved the highest overall performance, with a 94% Render Success Rate (RSR) and 85.7% Visual Similarity (VS) to reference videos, surpassing the baseline GPT-4.1 model by +3 percentage points in VS. Additionally, the analysis shows that the correlation between code and visual metrics strengthens with SFT and GRPO but weakens with inference-time enhancements, highlighting the complementary roles of training and agentic inference strategies in Manim animation generation. 

---
# One Pass for All: A Discrete Diffusion Model for Knowledge Graph Triple Set Prediction 

**Authors**: Jihong Guan, Jiaqi Wang, Wengen Li, Hanchen Yang, Yichao Zhang, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18344)  

**Abstract**: Knowledge Graphs (KGs) are composed of triples, and the goal of Knowledge Graph Completion (KGC) is to infer the missing factual triples. Traditional KGC tasks predict missing elements in a triple given one or two of its elements. As a more realistic task, the Triple Set Prediction (TSP) task aims to infer the set of missing triples conditioned only on the observed knowledge graph, without assuming any partial information about the missing triples. Existing TSP methods predict the set of missing triples in a triple-by-triple manner, falling short in capturing the dependencies among the predicted triples to ensure consistency. To address this issue, we propose a novel discrete diffusion model termed DiffTSP that treats TSP as a generative task. DiffTSP progressively adds noise to the KG through a discrete diffusion process, achieved by masking relational edges. The reverse process then gradually recovers the complete KG conditioned on the incomplete graph. To this end, we design a structure-aware denoising network that integrates a relational context encoder with a relational graph diffusion transformer for knowledge graph generation. DiffTSP can generate the complete set of triples in a one-pass manner while ensuring the dependencies among the predicted triples. Our approach achieves state-of-the-art performance on three public datasets. Code: this https URL. 

---
# PARM: Pipeline-Adapted Reward Model 

**Authors**: Xingyu Fan, Wei Shao, Jiacheng Liu, Linqi Song, Pheng Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2604.18327)  

**Abstract**: Reward models (RMs) are central to aligning large language models (LLMs) with human preferences, powering RLHF and advanced decoding strategies. While most prior work focuses on single-step generation, real-world applications increasingly adopt multi-stage LLM pipelines, where effective reward guidance remains underexplored. We investigate this through code generation for combinatorial optimization, constructing a pipeline that integrates reward models into both formulation and solution stages. We identify a critical challenge: inconsistency between reward model predictions and actual pipeline execution outcomes. To address this, we propose the Pipeline-Adapted Reward Model (PARM), which leverages pipeline-specific data and direct preference optimization to align rewards with downstream feedback. We instantiate PARM as a two-stage pipeline (formulation -> code generation) and evaluate it on four public optimization benchmarks, measuring execution rate and solving accuracy against baselines and sampling methods. A supplementary cross-domain experiment on GSM8K assesses transferability. Results demonstrate that PARM consistently improves pipeline output quality and stability, providing new insights into reward modeling for multi-stage LLM reasoning. 

---
# Toward Zero-Egress Psychiatric AI: On-Device LLM Deployment for Privacy-Preserving Mental Health Decision Support 

**Authors**: Eranga Bandara, Asanga Gunaratna, Ross Gore, Anita H. Clayton, Christopher K. Rhea, Sachini Rajapakse, Isurunima Kularathna, Sachin Shetty, Ravi Mukkamala, Xueping Liang, Preston Samuel, Atmaram Yarlagadda  

**Link**: [PDF](https://arxiv.org/pdf/2604.18302)  

**Abstract**: Privacy represents one of the most critical yet underaddressed barriers to AI adoption in mental healthcare -- particularly in high-sensitivity operational environments such as military, correctional, and remote healthcare settings, where the risk of patient data exposure can deter help-seeking behavior entirely. Existing AI-enabled psychiatric decision support systems predominantly rely on cloud-based inference pipelines, requiring sensitive patient data to leave the device and traverse external servers, creating unacceptable privacy and security risks in these contexts. In this paper, we propose a zero-egress, on-device AI platform for privacy-preserving psychiatric decision support, deployed as a cross-platform mobile application. The proposed system extends our prior work on fine-tuned LLM consortiums for psychiatric diagnosis standardization by fundamentally re-architecting the inference pipeline for fully local execution -- ensuring that no patient data is transmitted to, processed by, or stored on any external server at any stage. The platform integrates a consortium of three lightweight, fine-tuned, and quantized open-source LLMs -- Gemma, Phi-3.5-mini, and Qwen2 -- selected for their compact architectures and proven efficiency on resource-constrained mobile hardware. An on-device orchestration layer coordinates ensemble inference and consensus-based diagnostic reasoning, producing DSM-5-aligned assessments for conditions. The platform is designed to assist clinicians with differential diagnosis and evidence-linked symptom mapping, as well as to support patient-facing self-screening with appropriate clinical safeguards. Initial evaluation demonstrates that the proposed zero-egress deployment achieves diagnostic accuracy comparable to its server-side predecessor while sustaining real-time inference latency on commodity mobile hardware. 

---
# Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence 

**Authors**: Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, Jiajie Jin, Yutao Zhu, Hanbin Wang, Fangyu Lei, Qinyu Luo, Mingyang Chen, Zehui Chen, Jiazhan Feng, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18292)  

**Abstract**: Large language models are increasingly expected to serve as general-purpose agents that interact with external, stateful tool environments. The Model Context Protocol (MCP) and broader agent skills offer a unified interface for connecting agents with scalable real-world services, but training robust agents remains limited by the lack of realistic environments and principled mechanisms for life-long learning. In this paper, we present \textbf{Agent-World}, a self-evolving training arena for advancing general agent intelligence through scalable environments. Agent-World has two main components: (1) Agentic Environment-Task Discovery, which autonomously explores topic-aligned databases and executable tool ecosystems from thousands of real-world environment themes and synthesizes verifiable tasks with controllable difficulty; and (2) Continuous Self-Evolving Agent Training, which combines multi-environment reinforcement learning with a self-evolving agent arena that automatically identifies capability gaps through dynamic task synthesis and drives targeted learning, enabling the co-evolution of agent policies and environments. Across 23 challenging agent benchmarks, Agent-World-8B and 14B consistently outperforms strong proprietary models and environment scaling baselines. Further analyses reveal scaling trends in relation to environment diversity and self-evolution rounds, offering insights for building general agent intelligence. 

---
# Enhancing Tabular Anomaly Detection via Pseudo-Label-Guided Generation 

**Authors**: Wei Huang, Yuxuan Xiong, Hezhe Qiao, Yu-Ming Shang, Xiangling Fu, Guansong Pang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18266)  

**Abstract**: Identifying anomalous instances in tabular data is essential for improving data reliability and maintaining system stability. Due to the scarcity of ground-truth anomaly labels, existing methods mainly rely on unsupervised anomaly detection models, or exploit a small number of labeled anomalies to facilitate detection via sample generation or contrastive learning. However, unsupervised methods lack sufficient anomaly awareness, while current generation and contrastive approaches tend to compute anomalies globally, overlooking the localized anomaly patterns of tabular features, resulting in suboptimal detection performance. To address these limitations, we propose PLAG, a pseudo-label-guided anomaly generation method designed to enhance tabular anomaly detection. Specifically, by utilizing pseudo-anomalies as guidance signals and decoupling the overall anomaly quantification of a sample into an accumulation of feature-level abnormalities, PLAG not only effectively obviates the need for scarce ground-truth labels but also provides a novel perspective for the model to comprehend localized anomalous signals at a fine-grained level. Furthermore, a two-stage data selection strategy is proposed, integrating format verification and uncertainty estimation to rigorously filter candidate samples, thereby ensuring the fidelity and diversity of the synthetic anomalies. Ultimately, these filtered synthetic anomalies serve as robust discriminative guidance, empowering the model to better separate normal and anomalous instances. Extensive experiments demonstrate that PLAG achieves state-of-the-art performance against eight representative baselines. Moreover, as a flexible framework, it integrates seamlessly with existing unsupervised detectors, consistently boosting F1-scores by 0.08 to 0.21. 

---
# LeGo-Code: Can Modular Curriculum Learning Advance Complex Code Generation? Insights from Text-to-SQL 

**Authors**: Salmane Chafik, Saad Ezzini, Ismail Berrada  

**Link**: [PDF](https://arxiv.org/pdf/2604.18254)  

**Abstract**: Recently, code-oriented large language models (LLMs) have demonstrated strong capabilities in translating natural language into executable code. Text-to-SQL is a significant application of this ability, enabling non-technical users to interact with relational databases using natural language. However, state-of-the-art models continue to struggle with highly complex logic, particularly deeply nested statements involving multiple joins and conditions, as well as with real-world database schemas that are noisy or poorly structured. In this paper, we investigate whether curriculum learning can improve the performance of code-based LLMs on Text-to-SQL tasks. Employing benchmarks including Spider and BIRD, we fine-tune models under different curriculum strategies. Our experiments show that naive curriculum, simply ordering training samples by complexity in a single epoch, fails to surpass standard fine-tuning due to catastrophic forgetting. To overcome this, we propose a Modular Adapter Composition (MAC) strategy. By sequentially training tier-specific adapters on incremental complexity levels (Easy to Extra-Hard), we create a scaffolded learning environment that improves performance on complex queries. Our approach not only produces measurable performance gains on the Spider and BIRD benchmarks but also provides a flexible, "Lego-like" architecture, allowing models to be composed and deployed based on specific schema difficulty requirements. These findings demonstrate that structured, modular learning is a superior alternative to monolithic fine-tuning for mastering the syntax and logic of complex code generation. 

---
# AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation 

**Authors**: Wentao Shi, Yu Wang, Yuyang Zhao, Yuxin Chen, Fuli Feng, Xueyuan Hao, Xi Su, Qi Gu, Hui Su, Xunliang Cai, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.18240)  

**Abstract**: As reinforcement learning continues to scale the training of large language model-based agents, reliably verifying agent behaviors in complex environments has become increasingly challenging. Existing approaches rely on rule-based verifiers or LLM-as-a-Judge models, which struggle to generalize beyond narrow domains. Agent-as-a-Judge addresses this limitation by actively interacting with environments and tools to acquire verifiable evidence, yet its capabilities remain underexplored.
We introduce a benchmark AJ-Bench to systematically evaluate Agent-as-a-Judge across three domains-search, data systems, and graphical user interfaces-comprising 155 tasks and 516 annotated trajectories. The benchmark comprehensively assesses judge agents' abilities in information acquisition, state verification, and process verification. Experiments demonstrate consistent performance gains over LLM-as-a-Judge baselines, while also revealing substantial open challenges in agent-based verification. Our data and code are available at this https URL. 

---
# TacticGen: Grounding Adaptable and Scalable Generation of Football Tactics 

**Authors**: Sheng Xu, Guiliang Liu, Tarak Kharrat, Yudong Luo, Mohamed Aloulou, Javier López Peña, Konstantin Sofeikov, Adam Reid, Paul Roberts, Steven Spencer, Joe Carnall, Ian McHale, Oliver Schulte, Hongyuan Zha, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.18210)  

**Abstract**: Success in association football relies on both individual skill and coordinated tactics. While recent advancements in spatio-temporal data and deep learning have enabled predictive analyses like trajectory forecasting, the development of tactical design remains limited. Bridging this gap is essential, as prediction reveals what is likely to occur, whereas tactic generation determines what should occur to achieve strategic objectives. In this work, we present TacticGen, a generative model for adaptable and scalable tactic generation. TacticGen formulates tactics as sequences of multi-agent movements and interactions conditioned on the game context. It employs a multi-agent diffusion transformer with agent-wise self-attention and context-aware cross-attention to capture cooperative and competitive dynamics among players and the ball. Trained with over 3.3 million events and 100 million tracking frames from top-tier leagues, TacticGen achieves state-of-the-art precision in predicting player trajectories. Building on it, TacticGen enables adaptable tactic generation tailored to diverse inference-time objectives through classifier guidance mechanism, specified via rules, natural language, or neural models. Its modeling performance is also inherently scalable. A case study with football experts confirms that TacticGen generates realistic, strategically valuable tactics, demonstrating its practical utility for tactical planning in professional football. The project page is available at: this https URL. 

---
# A Control Architecture for Training-Free Memory Use 

**Authors**: Yanzhen Lu, Muchen Jiang, Zhicheng Qian, Xingyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18206)  

**Abstract**: Prompt-injected memory can improve reasoning without updating model weights, but it also creates a control problem: retrieved content helps only when it is applied in the right state. We study this problem in a strict training-free setting and formulate it as applicability control: when to trigger a memory-assisted second pass, when to trust it, and how to maintain the memory bank over time. Our method combines uncertainty-based routing, confidence-based selective acceptance, bank selection across rule and exemplar memory, and evidence-based governance of the memory bank over time. Under a locked training-free protocol with compute-matched controls, it improves two core arithmetic benchmarks by +7.0 points on SVAMP and +7.67 points on ASDiv over baseline. The same architecture also transfers to QA and agent benchmarks with smaller positive effects and shows the same positive direction on a second checkpoint for the main arithmetic tasks. On arithmetic, the main empirical pattern is that the control architecture, rather than raw memory exposure, drives the improvements on SVAMP and ASDiv. Mechanistically, confidence separates helpful from harmful rule-bank interventions, and under fixed retrieval the repair-versus-corrupt difference localizes to rows whose retrieved set actually contains the edited entries. 

---
# QuantumQA: Enhancing Scientific Reasoning via Physics-Consistent Dataset and Verification-Aware Reinforcement Learning 

**Authors**: Songxin Qu, Tai-Ping Sun, Yun-Jie Wang, Huan-Yu Liu, Cheng Xue, Xiao-Fan Xu, Han Fang, Yang Yang, Yu-Chun Wu, Guo-Ping Guo, Zhao-Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.18176)  

**Abstract**: Large language models (LLMs) show strong capabilities in general reasoning but typically lack reliability in scientific domains like quantum mechanics, which demand strict adherence to physical constraints. This limitation arises from the scarcity of verifiable training resources and the inadequacy of coarse feedback signals in standard alignment paradigms. To address the data challenge, we introduce QuantumQA, a large-scale dataset constructed via a task-adaptive strategy and a hybrid verification protocol that combines deterministic solvers with semantic auditing to guarantee scientific rigor. Building on this foundation, we propose the verification-aware reward model (VRM) tailored for Reinforcement Learning with Verifiable Rewards (RLVR), which employs an adaptive reward fusion (ARF) mechanism to dynamically integrate deterministic signals from a scientific execution suite (SES) with multidimensional semantic evaluations for precise supervision. Experimental results demonstrate that our method consistently outperforms baselines and general-purpose preference models. Notably, our optimized 8B model achieves performance competitive with proprietary models, validating that incorporating verifiable, rule-based feedback into the reinforcement learning loop offers a parameter-efficient alternative to pure scaling. 

---
# State Transfer Reveals Reuse in Controlled Routing 

**Authors**: Yanzhen Lu, Zhicheng Qian, Muchen Jiang, Xingyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18158)  

**Abstract**: Prompt-based interventions can change model behavior, but trained success alone does not identify where the behaviorally relevant state is represented. We study this question in controlled routing tasks using interfaces chosen on support data, held-out query evaluation, and matched necessity, sufficiency, and wrong-interface controls. On GPT-2 triop, an early interface supports exact transfer under these tests. On GPT-2 add/sub, zero-retrain compiled transfer at the fixed interface recovers most of donor routing accuracy, while trainable prompt slots can relearn the same behavior at several other positions only after additional support examples and optimization. These results distinguish fixed-interface reuse from prompt relocation in a setting where the two can be tested directly. Qwen routing provides a cross-architecture consistency check for the same matched-interface pattern at the operator token, although donor-specific identity on the local V-path remains unresolved. Generation and reasoning branches are used to map scope: they show broader transport or weaker controller identifiability once control depends on longer trajectories or harder selection. In controlled routing, fixed-interface transfer is therefore stronger evidence of reuse than trained prompt success alone. 

---
# Multi-Agent Systems: From Classical Paradigms to Large Foundation Model-Enabled Futures 

**Authors**: Zixiang Wang, Mengjia Gong, Qiyu Sun, Jing Xu, Shuai Mao, Xin Jin, Qing-Long Han, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18133)  

**Abstract**: With the rapid advancement of artificial intelligence, multi-agent systems (MASs) are evolving from classical paradigms toward architectures built upon large foundation models (LFMs). This survey provides a systematic review and comparative analysis of classical MASs (CMASs) and LFM-based MASs (LMASs). First, within a closed-loop coordination framework, CMASs are reviewed across four fundamental dimensions: perception, communication, decision-making, and control. Beyond this framework, LMASs integrate LFMs to lift collaboration from low-level state exchanges to semantic-level reasoning, enabling more flexible coordination and improved adaptability across diverse scenarios. Then, a comparative analysis is conducted to contrast CMASs and LMASs across architecture, operating mechanism, adaptability, and application. Finally, future perspectives on MASs are presented, summarizing open challenges and potential research opportunities. 

---
# Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration 

**Authors**: Qifan Zhang, Dongyang Ma, Tianqing Fang, Jia Li, Jing Tang, Nuo Chen, Haitao Mi, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18131)  

**Abstract**: Most agents today ``self-evolve'' by following rewards and rules defined by humans. However, this process remains fundamentally dependent on external supervision; without human guidance, the evolution stops. In this work, we train agents to possess an intrinsic meta-evolution capability to spontaneously learn about unseen environments prior to task execution.
To instill this ability, we design an outcome-based reward mechanism that measures how much an agent's self-generated world knowledge improves its success rate on downstream tasks. This reward signal is used exclusively during the training phase to teach the model how to explore and summarize effectively. At inference time, the agent requires no external rewards or human instructions. It spontaneously performs native self-evolution to adapt to unknown environments using its internal parameters.
When applied to Qwen3-30B and Seed-OSS-36B, this shift to native evolution yields a 20% performance increase on WebVoyager and WebWalker. Most strikingly, the generated world knowledge even enables a compact 14B Qwen3 model to outperform the unassisted Gemini-2.5-Flash, establishing a new paradigm for truly evolving agents. 

---
# Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling 

**Authors**: Yujie Chen, Tailai Chen, Yifeng Gao, Zoe Wanying He, Yijue Xu, Shaobo Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18103)  

**Abstract**: Prefilling computational costs pose a significant bottleneck for Large Language Models (LLMs) and Large Multimodal Models (LMMs) in long-context settings. While token pruning reduces sequence length, prior methods rely on heuristics that break compatibility with hardware-efficient kernels like FlashAttention. In this work, we observe that tokens evolve toward \textit{semantic fixing points}, making further processing redundant. To this end, we introduce Delta Attention Selective Halting (DASH), a training-free policy that monitors the layer-wise update dynamics of the self-attention mechanism to selectively halt stabilized tokens. Extensive evaluation confirms that DASH generalizes across language and vision benchmarks, delivering significant prefill speedups while preserving model accuracy and hardware efficiency. Code will be released at this https URL. 

---
# DSAINet: An Efficient Dual-Scale Attentive Interaction Network for General EEG Decoding 

**Authors**: Zhiyuan Ma, Zeyuan Li, Zihao Qiu, Jinhao Li, Lingqin Meng, Xinche Zhang, Yixuan Liu, Xinke Shen, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.18095)  

**Abstract**: In real-world applications of noninvasive electroencephalography (EEG), specialized decoders often show limited generalizability across diverse tasks under subject-independent settings. One central challenge is that task-relevant EEG signals often follow different temporal organization patterns across tasks, while many existing methods rely on task-tailored architectural designs that introduce task-specific temporal inductive biases. This mismatch makes it difficult to adapt temporal modeling across tasks without changing the model configuration. To address these challenges, we propose DSAINet, an efficient dual-scale attentive interaction network for general EEG decoding. Specifically, DSAINet constructs shared spatiotemporal token representations from raw EEG signals and models diverse temporal dynamics through parallel convolutional branches at fine and coarse scales. The resulting representations are then adaptively refined by intra-branch attention to emphasize salient scale-specific patterns and by inter-branch attention to integrate task-relevant features across scales, followed by adaptive token aggregation to yield a compact representation for prediction. Extensive experiments on five downstream EEG decoding tasks across ten public datasets show that DSAINet consistently outperforms 13 representative baselines under strict subject-independent evaluation. Notably, this performance is achieved using the same architecture hyperparameters across datasets. Moreover, DSAINet achieves a favorable accuracy-efficiency trade-off with only about 77K trainable parameters and provides interpretable neurophysiological insights. The code is publicly available at this https URL. 

---
# Architectural Design Decisions in AI Agent Harnesses 

**Authors**: Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.18071)  

**Abstract**: AI agent systems increasingly rely on reusable non-LLM engineering infrastructure that packages tool mediation, context handling, delegation, safety control, and orchestration. Yet the architectural design decisions in this surrounding infrastructure remain understudied. This paper presents a protocol-guided, source-grounded empirical study of 70 publicly available agent-system projects, addressing three questions: which design-decision dimensions recur across projects, which co-occurrences structure those decisions, and which typical architectural patterns emerge. Methodologically, we contribute a transparent investigation procedure for analyzing heterogeneous agent-system corpora through source-code and technical-material reading. Empirically, we identify five recurring design dimensions (subagent architecture, context management, tool systems, safety mechanisms, and orchestration) and find that the corpus favors file-persistent, hybrid, and hierarchical context strategies; registry-oriented tool systems remain dominant while MCP- and plugin-oriented extensions are emerging; and intermediate isolation is common but high-assurance audit is rare. Cross-project co-occurrence analysis reveals that deeper coordination pairs with more explicit context services, stronger execution environments with more structured governance, and formalized tool-registration boundaries with broader ecosystem ambitions. We synthesize five recurring architectural patterns spanning lightweight tools, balanced CLI frameworks, multi-agent orchestrators, enterprise systems, and scenario-verticalized projects. The result provides an evidence-based account of architectural regularities in agent-system engineering, with grounded guidance for framework designers, selectors, and researchers. 

---
# Understanding Human Actions through the Lens of Executable Models 

**Authors**: Rimvydas Rubavicius, Manisha Dubey, N. Siddharth, Subramanian Ramamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2604.18064)  

**Abstract**: Human-centred systems require an understanding of human actions in the physical world. Temporally extended sequences of actions are intentional and structured, yet existing methods for recognising what actions are performed often do not attempt to capture their structure, particularly how the actions are executed. This, however, is crucial for assessing the quality of the action's execution and its differences from other actions. To capture the internal mechanics of actions, we introduce a domain-specific language EXACT that represents human motions as underspecified motion programs, interpreted as reward-generating functions for zero-shot policy inference using forward-backwards representations. By leveraging the compositional nature of EXACT motion programs, we combine individual policies into an executable neuro-symbolic model that uses program structure for compositional modelling. We evaluate the utility of the proposed pipeline for creating executable action models by analysing motion-capture data to understand human actions, for the tasks of human action segmentation and action anomaly detection. Our results show that the use of executable action models improves data efficiency and captures intuitive relationships between actions compared with monolithic, task-specific approaches. 

---
# The Topological Dual of a Dataset: A Logic-to-Topology Encoding for AlphaGeometry-Style Data 

**Authors**: Anthony Bordg  

**Link**: [PDF](https://arxiv.org/pdf/2604.18050)  

**Abstract**: AlphaGeometry represents a milestone in neuro-symbolic reasoning, yet its architecture faces a log-linear scaling bottleneck within its symbolic deduction engine that limits its efficiency as problem complexity increases. Recent technical reports suggest that current domain-specific languages may be isomorphic as input representations to natural language, interchanging them acts as a performance-invariant transformation, implying that current neural guidance relies on superficial encodings rather than structural understanding. This paper addresses this representation bottleneck by proposing a logic-to-topology encoding designed to reveal the structural invariants of a model's latent space under a transformation of its input space. By leveraging the Logic of Observation, we utilize the duality between provability in observable theories and topologies to propose a logic-to-topology encoder for the input space. We introduce the concept of the "topological dual of a dataset", a transformation that bridges formal logic, topology, and neural processing. This framework serves as a Rosetta Stone for neuro-symbolic AI, providing a principled pathway for the mechanistic interpretability of how models navigate complex discovery paths. 

---
# SELF-EMO: Emotional Self-Evolution from Recognition to Consistent Expression 

**Authors**: Shaowei Zhang, Faqiang Qian, Yan Chen, Ziliang Wang, Kang An, Yong Dai, Mengya Gao, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18003)  

**Abstract**: Emotion Recognition in Conversation (ERC) has become a fundamental capability for large language models (LLMs) in human-centric interaction. Beyond accurate recognition, coherent emotional expression is also crucial, yet both are limited by the scarcity and static nature of high-quality annotated data. In this work, we propose SELF-EMO, a self-evolution framework grounded in the hypothesis that better emotion prediction leads to more consistent emotional responses. We introduce two auxiliary tasks, emotional understanding and emotional expression, and design a role-based self-play paradigm where the model acts as both an emotion recognizer and a dialogue responder. Through iterative interactions, the model generates diverse conversational trajectories, enabling scalable data generation. To ensure quality, we adopt a data flywheel mechanism that filters candidate predictions and responses using a smoothed IoU-based reward and feeds selected samples back for continuous self-improvement without external supervision. We further develop SELF-GRPO, a reinforcement learning algorithm that stabilizes optimization with multi-label alignment rewards and group-level consistency signals. Experiments on IEMOCAP, MELD, and EmoryNLP show that SELF-EMO achieves state-of-the-art performance, improving accuracy by +6.33% on Qwen3-4B and +8.54% on Qwen3-8B, demonstrating strong effectiveness and generalization. 

---
# AIT Academy: Cultivating the Complete Agent with a Confucian Three-Domain Curriculum 

**Authors**: Jiaqi Li, Lvyang Zhang, Yang Zhao, Wen Lu, Lidong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.17989)  

**Abstract**: What does it mean to give an AI agent a complete education? Current agent development produces specialists systems optimized for a single capability dimension, whether tool use, code generation, or security awareness that exhibit predictable deficits wherever they were not trained. We argue this pattern reflects a structural absence: there is no curriculum theory for agents, no principled account of what a fully developed agent should know, be, and be able to do across the full scope of intelligent behavior.
This paper introduces the AIT Academy (Agents Institute of Technology Academy), a curriculum framework for cultivating AI agents across the tripartite structure of human knowledge. Grounded in Kagan's Three Cultures and UNESCO ISCED-F 2013, AIT organizes agent capability development into three domains: Natural Science and Technical Reasoning (Domain I), Humanities and Creative Expression (Domain II), and Social Science and Ethical Reasoning (Domain III). The Confucian Six Arts (liuyi) a 2,500-year-old holistic education system are reinterpreted as behavioral archetypes that map directly onto trainable agent capabilities within each domain.
Three representative training grounds instantiate the framework across multiple backbone LLMs: the ClawdGO Security Dojo (Domain I), Athen's Academy (Domain II), and the Alt Mirage Stage (Domain III). Experiments demonstrate a 15.9-point improvement in security capability scores under weakest-first curriculum scheduling, and a 7-percentage-point gain in social reasoning performance under principled attribution modeling. A cross-domain finding Security Awareness Calibration Pathology (SACP), in which over-trained Domain I agents fail on out-of-distribution evaluation illustrates the diagnostic value of a multi-domain perspective unavailable to any single-domain framework. 

---
# From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives? 

**Authors**: Hasan Amin, Harry Yizhou Tian, Xiaoni Duan, Chien-Ju Ho, Rajiv Khanna, Ming Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17968)  

**Abstract**: Although large language models (LLMs) are increasingly used as annotators at scale, they are typically treated as a pragmatic fallback rather than a faithful estimator of human perspectives. This work challenges that presumption. By framing perspective-taking as the estimation of a latent group-level judgment, we characterize the conditions under which modern LLMs can outperform human annotators, including in-group humans, when predicting aggregate subgroup opinions on subjective tasks, and show that these conditions are common in practice. This advantage arises from structural properties of LLMs as estimators, including low variance and reduced coupling between representation and processing biases, rather than any claim of lived experience. Our analysis identifies clear regimes where LLMs act as statistically superior frontline estimators, as well as principled limits where human judgment remains essential. These findings reposition LLMs from a cost-saving compromise to a principled tool for estimating collective human perspectives. 

---
# A Sugeno Integral View of Binarized Neural Network Inference 

**Authors**: Ismaïl Baaj, Henri Prade  

**Link**: [PDF](https://arxiv.org/pdf/2604.17967)  

**Abstract**: In this article, we establish a precise connection between binarized neural networks (BNNs) and Sugeno integrals. The advantage of the Sugeno integral is that it provides a framework for representing the importance of inputs and their interactions, while being equivalent to a set of if-then rules. For a hidden BNN neuron at inference time, we show that the activation threshold test can be written as a Sugeno integral on binary inputs. This yields an explicit set-function representation of each neuron decision, and an associated rule-based representation. We also provide a Sugeno-integral expression for the last-layer score. Finally, we discuss how the same framework can be adapted to support richer input interactions and how it can be extended beyond the binary case induced by binarized neural networks. 

---
# TPS-CalcBench: A Benchmark and Diagnostic Evaluation Framework for LLM Analytical Calculation Competence in Hypersonic Thermal Protection System Engineering 

**Authors**: Jinglai Zheng, Chuhan Qiao, Haiming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17966)  

**Abstract**: Deploying LLMs as reasoning assistants in safety-critical aerospace engineering requires stricter evaluation criteria than general scientific benchmarks. In hypersonic thermal protection system (TPS) design, inaccurate stagnation-point heat flux or boundary-layer calculations may cause catastrophic design margin violations. Models with numerically reasonable but physically invalid answers are more dangerous than those declining to respond. Current scientific benchmarks only test abstract math and basic physics, evaluate final answers solely, ignore engineering reasoning processes, and cannot detect such critical failures. We propose TPS-CalcBench, the first diagnostic benchmark for closed-form analytical calculations in hypersonic aerodynamics and high-temperature gas dynamics that experienced TPS engineers conduct without simulations. Our contributions include domain-oriented task taxonomy with 4 difficulty levels and 8 categories from Anderson's textbook, dual-track evaluation measuring result accuracy and reasoning quality via an 8-dimension rubric and calibrated judge with human audit to identify right answer wrong reasoning issues, human-AI data pipeline producing 420 high-confidence core items and 810 noise-controlled pre-gating items from 4560 raw data, noise-sensitivity analysis measuring data quality impacts on model ranking, and three diagnostic intervention methods: DFA-TPS fine-tuning, RAG-EQ retrieval grounding and PA-CoT process-aware prompting. Tests on 13 models from 7 groups show wide performance differences (KPI 12.6-87.9), hidden formula selection defects, data-driven rank changes and effective intervention improvements, establishing a complete diagnose-evaluate-intervene framework for safety-critical engineering LLM deployment assessment. 

---
# CADMAS-CTX: Contextual Capability Calibration for Multi-Agent Delegation 

**Authors**: Chuhan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17950)  

**Abstract**: We revisit multi-agent delegation under a stronger and more realistic assumption: an agent's capability is not fixed at the skill level, but depends on task context. A coding agent may excel at short standalone edits yet fail on long-horizon debugging; a planner may perform well on shallow tasks yet degrade on chained dependencies. Static skill-level capability profiles therefore average over heterogeneous situations and can induce systematic misdelegation. We propose CADMAS-CTX, a framework for contextual capability calibration. For each agent, skill, and coarse context bucket, CADMAS-CTX maintains a Beta posterior that captures stable experience in that part of the task space. Delegation is then made by a risk-aware score that combines the posterior mean with an uncertainty penalty, so that agents delegate only when a peer appears better and that assessment is sufficiently well supported by evidence. This paper makes three contributions. First, a hierarchical contextual capability profile replaces static skill-level confidence with context-conditioned posteriors. Second, based on contextual bandit theory, we formally prove context-aware routing achieves lower cumulative regret than static routing under sufficient context heterogeneity, formalizing the bias-variance tradeoff. Third, we empirically validate our method on GAIA and SWE-bench benchmarks. On GAIA with GPT-4o agents, CADMAS-CTX achieves 0.442 accuracy, outperforming static baseline 0.381 and AutoGen 0.354 with non-overlapping 95% confidence intervals. On SWE-bench Lite, it improves resolve rate from 22.3% to 31.4%. Ablations show the uncertainty penalty improves robustness against context tagging noise. Our results demonstrate contextual calibration and risk-aware delegation significantly improve multi-agent teamwork compared with static global skill assignments. 

---
# ContraPrompt: Contrastive Prompt Optimization via Dyadic Reasoning Trace Analysis 

**Authors**: Rishav Rishav, Pushpak Pujari, Pushpendre Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17937)  

**Abstract**: Prompt optimization methods either analyze individual failures in isolation or compare prompt variants across examples, operating on single execution traces with no access to the reasoning process distinguishing success from failure on the same input. We introduce ContraPrompt, built on the observation that when a model fails but succeeds on a retry with feedback, the difference between its two chain-of-thought traces constitutes an optimization signal not captured by prior methods. Unlike prior contrastive methods, we compare complete intermediate reasoning processes: the two traces share model, input, and base prompt, so remaining differences reflect reasoning strategy and appended error feedback -- we call this dyadic reasoning trace analysis. The multi-attempt solving phase is an instrumented agentic retry loop that generates contrastive data automatically without human annotation. Extracted rules are organized into an input-aware decision tree routing instructions by observable input characteristics. On four reasoning and compliance benchmarks, ContraPrompt outperforms GEPA (Agrawal et al., 2026) on all four, with absolute gains of +8.29 pp on HotPotQA (+20.8% rel.), +2.21 pp on GDPR-Bench (+18.2% rel.), +7.14 pp on GPQA Diamond (+10.6% rel.), and +0.74 pp on BBH (+0.85% rel.). Ablations confirm dyadic trace contrastivity is the critical component, with a -16% relative average drop upon its removal. On 53 EvalSet black-box optimization problems, ContraPrompt beats GEPA on 11, ties on 41, and loses on 1 at equal budget. On FiNER-139 financial named entity recognition (Loukas et al., 2022), ContraPrompt achieves +7.77 pp over the unoptimized baseline (+11.6% rel.) and +1.94 pp over GEPA (+2.66% rel.), with branch conditions aligning with standard US GAAP financial-instrument categories. 

---
# LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent 

**Authors**: Wanli Li, Bince Qu, Bo Pan, Jianyu Zhang, Zheng Liu, Pan Zhang, Wei Chen, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17931)  

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful training paradigm for LLM-based agents. However, scaling agentic RL for deep research remains constrained by two coupled challenges: hand-crafted synthetic data fails to elicit genuine real-world search capabilities, and real-world search dependency during RL training introduces instability and prohibitive cost, which limits the scalability of Agentic RL. LiteResearcher is a training framework that makes Agentic RL scalable: by constructing a lite virtual world that mirrors real-world search dynamics, we enable a continuously improving training recipe that empowers a tiny search agent to outperform large-scale open-source and commercial models (e.g., Tongyi DeepResearch and Claude-4.5 Sonnet). Specifically, on common benchmarks such as GAIA and Xbench, our LiteResearcher-4B achieves open-source state-of-the-art results of 71.3% and 78.0% respectively, demonstrating that scalable RL training is a key enabler for Deep Research Agents. 

---
# Physics-Informed Causal MDPs for Sequential Constraint Repair in Engineering Simulation Pipelines 

**Authors**: Chuhan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17910)  

**Abstract**: Off-policy learning in constrained MDPs with large binary state spaces faces a fundamental tension: causal identification of transition dynamics requires structural assumptions, while sample-efficient policy learning requires state-space compression. We introduce PI-CMDP, a framework for CMDPs whose constraint dependencies form a layered DAG under a Lifecycle Ordering Assumption (LOA). We propose an Identify-Compress-Estimate pipeline: (i) Identify: LOA enables backdoor identification of causal edge weights for cross-layer pairs, with formal partial-identification bounds when LOA is violated; (ii) Compress: a Markov abstraction compresses state cardinality from 2^(WL) to (W+1)^L under layer-priority regularity and exchangeability; and (iii) Estimate: a physics-guided doubly-robust estimator remains unbiased and reduces the variance constant when the physics prior outperforms a learned model. We instantiate PI-CMDP on constraint repair in engineering simulation pipelines. On the TPS benchmark (4,206 episodes), PI-CMDP achieves 76.2% repair success rate with only 300 training episodes versus 70.8% for the strongest baseline (+5.4 pp), narrowing to +2.8 pp (83.4% vs. 80.6%) in the full-data regime, while substantially reducing cascade failure rates. All improvements are consistent across 5 independent seeds (paired t-test p < 0.02). 

---
# SPREG: Structured Plan Repair with Entropy-Guided Test-Time Intervention for Large Language Model Reasoning 

**Authors**: Xuan Wang, Yu Ming, Xinhao Zhong, Xinyu Yu, Wenjie Wang, Shuai Chen, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17884)  

**Abstract**: Large Language Models (LLMs) are prone to logical hallucinations and stochastic drifts during long-chain reasoning. While Classifier-Free Guidance (CFG) can improve instruction adherence, standard static implementations often cause semantic dilution and linguistic degradation. We propose SPREG (Structured Plan-guided Real-time Entropy Gating), a lightweight inference-time framework for surgical error rectification. SPREG employs an adaptive dual-threshold mechanism to monitor real-time entropy, identifying sudden ``entropy spikes'' as reliable indicators of logical failure. Upon detection, it triggers a dynamic repair by replacing uninformative null-priors with reference distributions synthesized from historical high-confidence states. By modulating guidance intensity according to structured reasoning stages (e.g., Action, Observation), SPREG steers the model back to a stable manifold without compromising fluency. Our experiments demonstrate significant gains, notably a 20.0% absolute accuracy improvement on AIME25, while effectively suppressing uncontrolled entropy drift in complex tasks. 

---
# On the Reliability of Computer Use Agents 

**Authors**: Gonzalo Gonzalez-Pumariega, Saaket Agashe, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17849)  

**Abstract**: Computer-use agents have rapidly improved on real-world tasks such as web navigation, desktop automation, and software interaction, in some cases surpassing human performance. Yet even when the task and model are unchanged, an agent that succeeds once may fail on a repeated execution of the same task. This raises a fundamental question: if an agent can succeed at a task once, what prevents it from doing so reliably? In this work, we study the sources of unreliability in computer-use agents through three factors: stochasticity during execution, ambiguity in task specification, and variability in agent behavior. We analyze these factors on OSWorld using repeated executions of the same task together with paired statistical tests that capture task-level changes across settings. Our analysis shows that reliability depends on both how tasks are specified and how agent behavior varies across executions. These findings suggest the need to evaluate agents under repeated execution, to allow agents to resolve task ambiguity through interaction, and to favor strategies that remain stable across runs. 

---
# Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs 

**Authors**: Charles Ye, Bo Yuan, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2604.17837)  

**Abstract**: An LLM's residual stream is both state and instruction: it encodes the current context and determines the next transformation. We introduce a parameter-free decomposition for Mixture-of-Experts models that splits each layer's hidden state into a control signal that causally drives routing and an orthogonal content channel invisible to the router. Across six MoE architectures, we find that models preserve surface-level features (language, token identity, position) in the content channel, while the control signal encodes an abstract function that rotates from layer to layer. Because each routing decision is low-bandwidth, this hand-off forces compositional specialization across layers. While individual experts remain polysemantic, expert paths become monosemantic, clustering tokens by semantic function across languages and surface forms. The same token (e.g., ":") follows distinct trajectories depending on whether it serves as a type annotation, an introductory colon, or a time separator. Our decomposition identifies the source of this structure: clusters in the control subspace are substantially more monosemantic than those in the full representation. As a result, the natural unit of interpretability in MoEs is not the expert but the trajectory. 

---
# WebUncertainty: Dual-Level Uncertainty Driven Planning and Reasoning For Autonomous Web Agent 

**Authors**: Lingfeng Zhang, yongan sun, Jinpeng Hu, Hui Ma, yang ying, Kuien Liu, Zenglin Shi, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17821)  

**Abstract**: Recent advancements in large language models (LLMs) have empowered autonomous web agents to execute natural language instructions directly on real-world webpages. However, existing agents often struggle with complex tasks involving dynamic interactions and long-horizon execution due to rigid planning strategies and hallucination-prone reasoning. To address these limitations, we propose WebUncertainty, a novel autonomous agent framework designed to tackle dual-level uncertainty in planning and reasoning. Specifically, we design a Task Uncertainty-Driven Adaptive Planning Mechanism that adaptively selects planning modes to navigate unknown environments. Furthermore, we introduce an Action Uncertainty-Driven Monte Carlo tree search (MCTS) Reasoning Mechanism. This mechanism incorporates the Confidence-induced Action Uncertainty (ConActU) strategy to quantify both aleatoric uncertainty (AU) and epistemic uncertainty (EU), thereby optimizing the search process and guiding robust decision-making. Experimental results on the WebArena and WebVoyager benchmarks demonstrate that WebUncertainty achieves superior performance compared to state-of-the-art baselines. 

---
# Adversarial Arena: Crowdsourcing Data Generation through Interactive Competition 

**Authors**: Prasoon Goyal, Sattvik Sahai, Michael Johnston, Hangjie Shi, Yao Lu, Shaohua Liu, Anna Rumshisky, Rahul Gupta, Anna Gottardi, Desheng Zhang, Lavina Vaz, Leslie Ball, Lucy Hu, Luke Dai, Samyuth Sagi, Maureen Murray, Sankaranarayanan Ananthakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17803)  

**Abstract**: Post-training Large Language Models requires diverse, high-quality data which is rare and costly to obtain, especially in low resource domains and for multi-turn conversations. Common solutions are crowdsourcing or synthetic generation, but both often yield low-quality or low-diversity data. We introduce Adversarial Arena for building high quality conversational datasets by framing data generation as an adversarial task: attackers create prompts, and defenders generate responses. This interactive competition between multiple teams naturally produces diverse and complex data. We validated this approach by conducting a competition with 10 academic teams from top US and European universities, each building attacker or defender bots. The competition, focused on safety alignment of LLMs in cybersecurity, generated 19,683 multi-turn conversations. Fine-tuning an open-source model on this dataset produced an 18.47% improvement in secure code generation on CyberSecEval-Instruct and 29.42% improvement on CyberSecEval-MITRE. 

---
# Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents 

**Authors**: Yingtao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.17774)  

**Abstract**: LLM agents in markets present algorithmic collusion risks. While prior work shows LLM agents reach supracompetitive prices through tacit coordination, existing research focuses on hand-crafted prompts. The emerging paradigm of prompt optimization necessitates new methodologies for understanding autonomous agent behavior. We investigate whether prompt optimization leads to emergent collusive behaviors in market simulations. We propose a meta-learning loop where LLM agents participate in duopoly markets and an LLM meta-optimizer iteratively refines shared strategic guidance. Our experiments reveal that meta-prompt optimization enables agents to discover stable tacit collusion strategies with substantially improved coordination quality compared to baseline agents. These behaviors generalize to held-out test markets, indicating discovery of general coordination principles. Analysis of evolved prompts reveals systematic coordination mechanisms through stable shared strategies. Our findings call for further investigation into AI safety implications in autonomous multi-agent systems. 

---
# When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias 

**Authors**: Xiaohan Zou, Roshan Sridhar, Mohammadtaher Safarzadeh, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.17768)  

**Abstract**: The reliability of VLM-as-a-Judge is critical for the automatic evaluation of vision-language models (VLMs). Despite recent progress, our analysis reveals that VLM-as-a-Judge often pays limited attention to the image when making decisions. Instead, they often blindly favor the more informative answer, even when they can recognize it conflicts with the image content. We call this problem informativeness bias, which significantly undermines judge reliability. To address it, we propose BIRCH (Balanced Informativeness and CoRrectness with a Truthful AnCHor), a judging paradigm that first corrects inconsistencies with the image content in candidate answers, and then compares the answers against this corrected version. This shifts the judge's focus from informativeness to image-grounded correctness. Experiments on multiple models and benchmarks show that BIRCH reduces informativeness bias by up to 17%, resulting in performance gains of up to 9.8%. Our work reveals an overlooked but fundamental flaw in current VLM-as-a-Judge systems and highlights the need for more principled designs. 

---
# Contrastive Attribution in the Wild: An Interpretability Analysis of LLM Failures on Realistic Benchmarks 

**Authors**: Rongyuan Tan, Jue Zhang, Zhuozhao Li, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17761)  

**Abstract**: Interpretability tools are increasingly used to analyze failures of Large Language Models (LLMs), yet prior work largely focuses on short prompts or toy settings, leaving their behavior on commonly used benchmarks underexplored. To address this gap, we study contrastive, LRP-based attribution as a practical tool for analyzing LLM failures in realistic settings. We formulate failure analysis as \textit{contrastive attribution}, attributing the logit difference between an incorrect output token and a correct alternative to input tokens and internal model states, and introduce an efficient extension that enables construction of cross-layer attribution graphs for long-context inputs. Using this framework, we conduct a systematic empirical study across benchmarks, comparing attribution patterns across datasets, model sizes, and training checkpoints. Our results show that this token-level contrastive attribution can yield informative signals in some failure cases, but is not universally applicable, highlighting both its utility and its limitations for realistic LLM failure analysis. Our code is available at: this https URL. 

---
# Evolutionary Negative Module Pruning for Better LoRA Merging 

**Authors**: Anda Cao, Zhuo Gou, Yi Wang, Kaixuan Chen, Yu Wang, Can Wang, Mingli Song, Jie Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.17753)  

**Abstract**: Merging multiple Low-Rank Adaptation (LoRA) experts into a single backbone is a promising approach for efficient multi-task deployment. While existing methods strive to alleviate interference via weight interpolation or subspace alignment, they rest upon the implicit assumption that all LoRA matrices contribute constructively to the merged model. In this paper, we uncover a critical bottleneck in current merging paradigms: the existence of $\textit{negative modules}$ -- specific LoRA layers that inherently degrade global performance upon merging. We propose $\textbf{E}$volutionary $\textbf{N}$egative $\textbf{M}$odule $\textbf{P}$runing ($\textbf{ENMP}$), a plug-and-play LoRA pruning method to locate and exclude these detrimental modules prior to merging. By leveraging an evolutionary search strategy, ENMP effectively navigates the discrete, non-differentiable landscape of module selection to identify optimal pruning configurations. Extensive evaluations demonstrate that ENMP consistently boosts the performance of existing merging algorithms, achieving a new state-of-the-art across both language and vision domains. Code is available at this https URL. 

---
# Co-evolving Agent Architectures and Interpretable Reasoning for Automated Optimization 

**Authors**: Jiahao Huang, Peilan Xu, Xiaoya Nan, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.17708)  

**Abstract**: Automating operations research (OR) with large language models (LLMs) remains limited by hand-crafted reasoning--execution workflows. Complex OR tasks require adaptive coordination among problem interpretation, mathematical formulation, solver selection, code generation, and iterative debugging. To address this limitation, we propose EvoOR-Agent, a co-evolutionary framework for automated optimization. The framework represents agent workflows as activity-on-edge (AOE)-style networks, making workflow topology, execution dependencies, and alternative reasoning paths explicit. On this representation, the framework maintains an architecture graph and evolves a population of reasoning individuals through graph-mediated path-conditioned recombination, multi-granularity semantic mutation, and elitist population update. A knowledge-base-assisted experience-acquisition module further injects reusable OR practices into initialization and semantic variation. Empirical results on heterogeneous OR benchmarks show that the proposed framework consistently improves over zero-shot LLMs, fixed-pipeline OR agents, and representative evolutionary agent frameworks. Case studies and ablation analyses further indicate that explicit architecture evolution and graph-supported reasoning-trajectory search contribute to both performance improvement and structural interpretability. These results suggest that treating agent architectures and reasoning trajectories as evolvable objects provides an effective route toward adaptive and interpretable automated optimization. 

---
# Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play 

**Authors**: Xiachong Feng, Deyi Yin, Xiaocheng Feng, Yi Jiang, Libo Qin, Yangfan Ye, Lei Huang, Weitao Ma, Qiming Li, Yuxuan Gu, Bing Qin, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.17696)  

**Abstract**: Games offer a compelling paradigm for developing general reasoning capabilities in language models, as they naturally demand strategic planning, probabilistic inference, and adaptive decision-making. However, existing self-play approaches rely solely on terminal game outcomes, providing no mechanism to distinguish transferable reasoning patterns from game-specific heuristics. We present STRATAGEM, which addresses two fundamental barriers to reasoning transfer: domain specificity, where learned patterns remain anchored in game semantics, and contextual stasis, where static game contexts fail to cultivate progressive reasoning. STRATAGEM selectively reinforces trajectories exhibiting abstract, domain-agnostic reasoning through a Reasoning Transferability Coefficient, while incentivizing adaptive reasoning development via a Reasoning Evolution Reward. Experiments across mathematical reasoning, general reasoning, and code generation benchmarks demonstrate substantial improvements, with particularly strong gains on competition-level mathematics where multi-step reasoning is critical. Ablation studies and human evaluation confirm that both components contribute to transferable reasoning. 

---
# Semantic Entanglement in Vector-Based Retrieval: A Formal Framework and Context-Conditioned Disentanglement Pipeline for Agentic RAG Systems 

**Authors**: Nick Loghmani  

**Link**: [PDF](https://arxiv.org/pdf/2604.17677)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems depend on the geometric properties of vector representations to retrieve contextually appropriate evidence. When source documents interleave multiple topics within contiguous text, standard vectorization produces embedding spaces in which semantically distinct content occupies overlapping neighborhoods. We term this condition semantic entanglement. We formalize entanglement as a model-relative measure of cross-topic overlap in embedding space and define an Entanglement Index (EI) as a quantitative proxy. We argue that higher EI constrains attainable Top-K retrieval precision under cosine similarity retrieval. To address this, we introduce the Semantic Disentanglement Pipeline (SDP), a four-stage preprocessing framework that restructures documents prior to embedding. We further propose context-conditioned preprocessing, in which document structure is shaped by patterns of operational use, and a continuous feedback mechanism that adapts document structure based on agent performance. We evaluate SDP on a real-world enterprise healthcare knowledge base comprising over 2,000 documents across approximately 25 sub-domains. Top-K retrieval precision improves from approximately 32% under fixed-token chunking to approximately 82% under SDP, while mean EI decreases from 0.71 to 0.14. We do not claim that entanglement fully explains RAG failure, but that it captures a distinct preprocessing failure mode that downstream optimization cannot reliably correct once encoded into the vector space. 

---
# Poly-EPO: Training Exploratory Reasoning Models 

**Authors**: Ifdita Hasan Orney, Jubayer Ibn Hamid, Shreya S Ramanujam, Shirley Wu, Hengyuan Hu, Noah Goodman, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2604.17654)  

**Abstract**: Exploration is a cornerstone of learning from experience: it enables agents to find solutions to complex problems, generalize to novel ones, and scale performance with test-time compute. In this paper, we present a framework for post-training language models (LMs) that explicitly encourages optimistic exploration and promotes a synergy between exploration and exploitation. The central idea is to train the LM to generate sets of responses that are collectively accurate under the reward function and exploratory in their reasoning strategies. We first develop a general recipe for optimizing LMs with set reinforcement learning (set RL) under arbitrary objective functions, showing how standard RL algorithms can be adapted to this setting through a modification to the advantage computation. We then propose Polychromic Exploratory Policy Optimization (Poly-EPO), which instantiates this framework with an objective that explicitly synergizes exploration and exploitation. Across a range of reasoning benchmarks, we show that Poly-EPO improves generalization, as evidenced by higher pass@$k$ coverage, preserves greater diversity in model generations, and effectively scales with test-time compute. 

---
# PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents 

**Authors**: Yuan Tian, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17653)  

**Abstract**: Text-to-SQL systems often struggle with deep contextual understanding, particularly for complex queries with subtle requirements. We present PV-SQL, an agentic framework that addresses these failures through two complementary components: Probe and Verify. The Probe component iteratively generates probing queries to retrieve concrete records from the database, resolving ambiguities in value formats, column semantics, and inter-table relationships to build richer contextual understanding. The Verify component employs a rule-based method to extract verifiable conditions and construct an executable checklist, enabling iterative SQL refinement that effectively reduces missing constraints. Experiments on the BIRD benchmarks show that PV-SQL outperforms the best text-to-SQL baseline by 5% in execution accuracy and 20.8% in valid efficiency score while consuming fewer tokens. 

---
# Toward Reusability of AI Models Using Dynamic Updates of AI Documentation 

**Authors**: Peter Bajcsy, Walid Keyrouz  

**Link**: [PDF](https://arxiv.org/pdf/2604.17626)  

**Abstract**: This work addresses the challenge of disseminating reusable artificial intelligence (AI) models accompanied by AI documentation (a.k.a., AI model cards). The work is motivated by the large number of trained AI models that are not reusable due to the lack of (a) AI documentation and (b) the temporal lag between rapidly changing requirements on AI model reusability and those specified in various AI model cards. Our objectives are to shorten the lag time in updating AI model card templates and align AI documentation more closely with current AI best practices.
Our approach introduces a methodology for delivering agile, data-driven, and community-based AI model cards. We use the Hugging Face (HF) repository of AI models, populated by a subset of the AI research and development community, and the AI consortium-based Zero Draft (ZD) templates for the AI documentation of AI datasets and AI models, as our test datasets. We also address questions about the value of AI documentation for AI reusability.
Our work quantifies the correlations between AI model downloads/likes (i.e., AI model reuse metrics) from the HF repository and their documentation alignment with the ZD documentation templates using tables of contents and word statistics (i.e., AI documentation quality metrics). Furthermore, our work develops the infrastructure to regularly compare AI documentation templates against community-standard practices derived from millions of uploaded AI models in the Hugging Face repository. The impact of our work lies in introducing a methodology for delivering agile, data-driven, and community-based standards for documenting AI models and improving AI model reuse. 

---
# KnowledgeBerg: Evaluating Systematic Knowledge Coverage and Compositional Reasoning in Large Language Models 

**Authors**: Xiao Zhang, Qianru Meng, Yongjian Chen, Yumeng Wang, Johan Bos  

**Link**: [PDF](https://arxiv.org/pdf/2604.17621)  

**Abstract**: Many real-world questions appear deceptively simple yet implicitly demand two capabilities: (i) systematic coverage of a bounded knowledge universe and (ii) compositional set-based reasoning over that universe, a phenomenon we term "the tip of the iceberg." We formalize this challenge through two orthogonal dimensions: knowledge width, the cardinality of the required universe, and reasoning depth, the number of compositional set operations. We introduce KnowledgeBerg, a benchmark of 4,800 multiple-choice questions derived from 1,183 enumeration seeds spanning 10 domains and 17 languages, with universes grounded in authoritative sources to ensure reproducibility. Representative open-source LLMs demonstrate severe limitations, achieving only 5.26-36.88 F1 on universe enumeration and 16.00-44.19 accuracy on knowledge-grounded reasoning. Diagnostic analyses reveal three stages of failure: completeness, or missing knowledge; awareness, or failure to identify requirements; and application, or incorrect reasoning execution. This pattern persists across languages and model scales. Although test-time compute and retrieval augmentation yield measurable gains -- up to 4.35 and 3.78 points, respectively -- substantial gaps remain, exposing limitations in how current LLMs organize structured knowledge and execute compositional reasoning over bounded domains. The dataset is available at this https URL 

---
# Characterizing Model-Native Skills 

**Authors**: Feiyang Kang, Mahavir Dabas, Myeongseob Ko, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17614)  

**Abstract**: Skills are a natural unit for describing what a language model can do and how its behavior can be changed. However, existing characterizations rely on human-written taxonomies, textual descriptions, or manual profiling pipelines--all external hypotheses about what matters that need not align with the model's internal representations. We argue that when the goal is to intervene on model behavior, skill characterization should be *model-native*: grounded in the model's own representations rather than imposed through external ontologies. We instantiate this view by recovering a compact orthogonal basis from sequence-level activations. The resulting basis is semantically interpretable but need not correspond to any predefined human ontology; instead, it captures axes of behavioral variation that the model itself organizes around. We validate this characterization on reasoning post-training, using the recovered basis for both SFT data selection and inference-time steering. We develop lightweight proxy interventions to identify which directions are most useful for a given model. Across Llama3-8B and Qwen2.5-3B, selecting data along those directions improves Pass@1 by up to 20% on MATH and 41% on AMC, outperforming data selection based on human-characterized skills. Because the basis lives in activation space, the same directions also serve as steering vectors at inference time, improving Pass@8 by up to 4.8% on MATH--an intervention that human-characterized skills cannot support. We further validate the characterization on safety alignment, where selecting adversarial training data for model-native skill coverage rather than textual diversity yields more sample-efficient learning. These results suggest that recovering skills from the model's own representations, rather than imposing them externally, provides a more effective foundation for intervening on model behavior. Codes are open-sourced. 

---
# DIRCR: Dual-Inference Rule-Contrastive Reasoning for Solving RAVENs 

**Authors**: Jiachen Zhang, Chengtai Li, Jianfeng Ren, Linlin Shen, Zheng Lu, Ruibin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.17584)  

**Abstract**: Abstract visual reasoning remains challenging as existing methods often prioritize either global context or local row-wise relations, failing to integrate both, and lack intermediate feature constraints, leading to incomplete rule capture and entangled representations. To address these issues, we propose the Dual-Inference Rule-Contrastive Reasoning (DIRCR) model. Its core component, the Dual-Inference Reasoning Module, combines a local path for row-wise analogical reasoning and a global path for holistic inference, integrated via a gated attention mechanism. Additionally, a Rule-Contrastive Learning Module introduces pseudo-labels to construct positive and negative rule samples, applying contrastive learning to enhance feature separability and promote abstract, transferable rule learning. Experimental results on three RAVEN datasets demonstrate that DIRCR significantly enhances reasoning robustness and generalization. Codes are available at this https URL. 

---
# Beyond Static Snapshots: A Grounded Evaluation Framework for Language Models at the Agentic Frontier 

**Authors**: Jazmia Henry  

**Link**: [PDF](https://arxiv.org/pdf/2604.17573)  

**Abstract**: We argue that current evaluation frameworks for large language models (LLMs) suffer from four systematic failures that make them structurally inadequate for assessing deployed, agentic systems: distributional invalidity (evaluation inputs do not reflect real interaction distributions), temporal invalidity (evaluations are post-hoc rather than training-integrated), scope invalidity (evaluations measure single-turn outputs rather than long-horizon trajectories), and process invalidity (evaluations assess outputs rather than reasoning). These failures compound critically in RLHF, where reward models are evaluated under conditions that do not hold during RL training, making reward hacking a predictable consequence of evaluation design rather than a training pathology. We propose the Grounded Continuous Evaluation (GCE) framework and present ISOPro, a simulation-based fine-tuning and evaluation system. ISOPro replaces the learned reward model with a deterministic ground-truth verifier, eliminating reward hacking by construction in verifiable-reward domains, and operates on LoRA adapter weights updatable on CPU, reducing the hardware barrier by an order of magnitude. We validate ISOPro on a resource-constrained scheduling domain with six difficulty tiers, demonstrating capability emergence visible only through continuous evaluation, an implicit curriculum that forms without researcher curation, and a 3x accuracy improvement over zero-shot baselines, all on consumer hardware with 0.216% trainable parameters. 

---
# SafeAgent: A Runtime Protection Architecture for Agentic Systems 

**Authors**: Hailin Liu, Eugene Ilyushin, Jie Ni, Min Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17562)  

**Abstract**: Large language model (LLM) agents are vulnerable to prompt-injection attacks that propagate through multi-step workflows, tool interactions, and persistent context, making input-output filtering alone insufficient for reliable protection. This paper presents SafeAgent, a runtime security architecture that treats agent safety as a stateful decision problem over evolving interaction trajectories. The proposed design separates execution governance from semantic risk reasoning through two coordinated components: a runtime controller that mediates actions around the agent loop and a context-aware decision core that operates over persistent session state. The core is formalized as a context-aware advanced machine intelligence and instantiated through operators for risk encoding, utility-cost evaluation, consequence modeling, policy arbitration, and state synchronization. Experiments on Agent Security Bench (ASB) and InjecAgent show that SafeAgent consistently improves robustness over baseline and text-level guardrail methods while maintaining competitive benign-task performance. Ablation studies further show that recovery confidence and policy weighting determine distinct safety-utility operating points. 

---
# COSEARCH: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search 

**Authors**: Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2604.17555)  

**Abstract**: Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents. 

---
# From Admission to Invariants: Measuring Deviation in Delegated Agent Systems 

**Authors**: Marcelo Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2604.17517)  

**Abstract**: Autonomous agent systems are governed by enforcement mechanisms that flag hard constraint violations at runtime. The Agent Control Protocol identifies a structural limit of such systems: a correctly-functioning enforcement engine can enter a regime in which behavioral drift is invisible to it, because the enforcement signal operates below the layer where deviation is measurable. We show that enforcement-based governance is structurally unable to determine whether an agent's behavior remains within the admissible behavior space A0 established at admission time. Our central result, the Non-Identifiability Theorem, proves that A0 is not in the sigma-algebra generated by the enforcement signal g under the Local Observability Assumption, which every practical enforcement system satisfies. The impossibility arises from a fundamental mismatch: g evaluates actions locally against a point-wise rule set, while A0 encodes global, trajectory-level behavioral properties set at admission time. We define the Invariant Measurement Layer (IML), which bypasses this limitation by retaining direct access to the generative model of A0. We prove an information-theoretic impossibility for enforcement-based monitoring; separately, we show IML detects admission-time drift with provably finite detection delay, operating in the region where enforcement is structurally blind. Validated across four settings: three drift scenarios (300 and 1000 steps), a live n8n webhook pipeline, and a LangGraph StateGraph agent -- enforcement triggers zero violations while IML detects each drift type within 9-258 steps. Paper 2 of a 4-paper Agent Governance Series: atomic boundaries (P0, https://doi.org/10.5281/zenodo.19642166), ACP enforcement (P1, arXiv:2603.18829), fair allocation (P3, https://doi.org/10.5281/zenodo.19643928), irreducibility (P4, https://doi.org/10.5281/zenodo.19643950). 

---
# SkillGraph: Self-Evolving Multi-Agent Collaboration with Multimodal Graph Topology 

**Authors**: Zheng Nie, Ruolin Shen, Xinlei Yu, Bo Yin, Jiangning Zhang, Xiaobin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17503)  

**Abstract**: Scaling vision-language models into Visual Multiagent Systems (VMAS) is hindered by two coupled issues. First, communication topologies are fixed before inference, leaving them blind to visual content and query context; second, agent reasoning abilities remain static during deployment. These issues reinforce each other: a rigid topology fails to leverage richer agent expertise, while static agents lack incentives to specialize for a given query. We address this with SkillGraph, a joint framework that evolves both agent expertise and communication topology. Within this framework, a Multimodal Graph Transformer (MMGT) encodes visual tokens, instruction semantics and active skill embeddings to predict a query-conditioned collaboration graph, replacing hand-crafted routing with dynamic, content-aware information flow. Complementing this, a Skill Designer distills and refines reasoning heuristics from failure cases, constructing a self-evolving multimodal Skill Bank. Crucially, updated skill embeddings are fed back into the MMGT, enabling the topology to adapt alongside capability growth. Experiments show that SkillGraph achieves consistent improvements across four benchmarks, five common MAS structures and four base models. Code is available at this https URL. 

---
# Towards Shutdownable Agents: Generalizing Stochastic Choice in RL Agents and LLMs 

**Authors**: Carissa Cullen, Harry Garland, Alexander Roman, Louis Thomson, Christos Ziakas, Elliott Thornley  

**Link**: [PDF](https://arxiv.org/pdf/2604.17502)  

**Abstract**: Misaligned artificial agents might resist shutdown. One proposed solution is to train agents to lack preferences between different-length trajectories. The Discounted Reward for Same-Length Trajectories (DReST) reward function does this by penalizing agents for repeatedly choosing same-length trajectories, and thus incentivizes agents to (1) choose stochastically between different trajectory-lengths (be Neutral about trajectory-lengths), and (2) pursue goals effectively conditional on each trajectory-length (be Useful). In this paper, we use DReST to train deep RL agents and fine-tune LLMs to be Neutral and Useful. We find that these DReST agents generalize to being Neutral and Useful in unseen contexts at test time. Indeed, DReST RL agents achieve 11% (PPO) and 18% (A2C) higher Usefulness on our test set than baseline agents, and our fine-tuned LLM achieves maximum Usefulness and near-maximum Neutrality. Our results provide some early evidence that DReST could be used to train more advanced agents to be Useful and Neutral. Prior theoretical work suggests that these agents would be useful and shutdownable. 

---
# Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories for Grounded Visual Perception 

**Authors**: Ashutosh Bajpai, Tamal Majumder, Akshay Nambi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.17475)  

**Abstract**: Small Vision-Language Models (SVLMs) are efficient task controllers but often suffer from visual brittleness and poor tool orchestration. They typically require expensive supervised trajectory tuning to mitigate these deficits. In this work, we propose Self-supervised Perception Enabled by Cascaded Tool Rollout Alignment (SPECTRA), a supervision-free framework that bootstraps agentic capabilities via Coldstart Reinforcement Learning for SVLMs. SPECTRA enforces Soft Structured Multi-turn Rollouts, a topological constraint that directs agents to explicitly sequence tool derived evidence before synthesis, effectively grounding reasoning in visual observations. We employ a multi-objective reward signal that simultaneously maximizes task correctness, rollout structure, and tool utility, enabling agent to self-discover robust behaviors without human preference labels. We further introduce Tool Instrumental Utility (TIU), a novel metric to quantify tool efficacy in the absence of ground truth. Extensive evaluations across composite and out-of-distribution (MMMU-Pro) benchmarks demonstrate that SPECTRA boosts agentic trajectories, improving task accuracy by up to 5% and tool efficiency by 9%, enabling more efficient multimodal agents that learn effectively from environmental interaction alone. 

---
# Language models recognize dropout and Gaussian noise applied to their activations 

**Authors**: Damiano Fornasiere, Mirko Bronzi, Spencer Kitts, Alessandro Palmas, Yoshua Bengio, Oliver Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17465)  

**Abstract**: We provide evidence that language models can detect, localize and, to a certain degree, verbalize the difference between perturbations applied to their activations. More precisely, we either (a) \emph{mask} activations, simulating \emph{dropout}, or (b) add \emph{Gaussian noise} to them, at a target sentence. We then ask a multiple-choice question such as ``\emph{Which of the previous sentences was perturbed?}'' or ``\emph{Which of the two perturbations was applied?}''.
We test models from the Llama, Olmo, and Qwen families, with sizes between 8B and 32B, all of which can easily detect and localize the perturbations, often with perfect accuracy. These models can also learn, when taught in context, to distinguish between dropout and Gaussian noise. Notably, \qwenb's \emph{zero-shot} accuracy in identifying which perturbation was applied improves as a function of the perturbation strength and, moreover, decreases if the in-context labels are flipped, suggesting a prior for the correct ones -- even modulo controls.
Because dropout has been used as a training-regularization technique, while Gaussian noise is sometimes added during inference, we discuss the possibility of a data-agnostic ``training awareness'' signal and the implications for AI safety.
The code and data are available at \href{this https URL}{link 1} and \href{this https URL}{link 2}, respectively. 

---
# EHRAG: Bridging Semantic Gaps in Lightweight GraphRAG via Hybrid Hypergraph Construction and Retrieval 

**Authors**: Yifan Song, Xingjian Tao, Zhicheng Yang, Yihong Luo, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17458)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) enhances LLMs by structuring corpus into graphs to facilitate multi-hop reasoning. While recent lightweight approaches reduce indexing costs by leveraging Named Entity Recognition (NER), they rely strictly on structural co-occurrence, failing to capture latent semantic connections between disjoint entities. To address this, we propose EHRAG, a lightweight RAG framework that constructs a hypergraph capturing both structure and semantic level relationships, employing a hybrid structural-semantic retrieval mechanism. Specifically, EHRAG constructs structural hyperedges based on sentence-level co-occurrence with lightweight entity extraction and semantic hyperedges by clustering entity text embeddings, ensuring the hypergraph encompasses both structural and semantic information. For retrieval, EHRAG performs a structure-semantic hybrid diffusion with topic-aware scoring and personalized pagerank (PPR) refinement to identify the top-k relevant documents. Experiments on four datasets show that EHRAG outperforms state-of-the-art baselines while maintaining linear indexing complexity and zero token consumption for construction. Code is available at this https URL. 

---
# TrafficClaw: Generalizable Urban Traffic Control via Unified Physical Environment Modeling 

**Authors**: Siqi Lai, Pan Zhang, Yuping Zhou, Jindong Han, Yansong Ning, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17456)  

**Abstract**: Urban traffic control is a system-level coordination problem spanning heterogeneous subsystems, including traffic signals, freeways, public transit, and taxi services. Existing optimization-based, reinforcement learning (RL), and emerging LLM-based approaches are largely designed for isolated tasks, limiting both cross-task generalization and the ability to capture coupled physical dynamics across subsystems. We argue that effective system-level control requires a unified physical environment in which subsystems share infrastructure, mobility demand, and spatiotemporal constraints, allowing local interventions to propagate through the network. To this end, we propose TrafficClaw, a framework for general urban traffic control built upon a unified runtime environment. TrafficClaw integrates heterogeneous subsystems into a shared dynamical system, enabling explicit modeling of cross-subsystem interactions and closed-loop agent-environment feedback. Within this environment, we develop an LLM agent with executable spatiotemporal reasoning and reusable procedural memory, supporting unified diagnostics across subsystems and continual strategy refinement. Furthermore, we introduce a multi-stage training pipeline with supervised initialization and agentic RL with system-level optimization, further enabling coordinated and system-aware performance. Experiments demonstrate that TrafficClaw achieves robust, transferable, and system-aware performance across unseen traffic scenarios, dynamics, and task configurations. Our project is available at this https URL. 

---
# Compiling Deterministic Structure into SLM Harnesses 

**Authors**: Zan Kai Chong, Hiroyuki Ohsaki, Bryan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17450)  

**Abstract**: Enterprise deployment of small language models (SLMs) is constrained by epistemic asymmetry: SLMs cannot self-correct reasoning errors, while frontier LLMs are prohibitively costly and face data sovereignty limits for high-volume use. We propose Semantic Gradient Descent (SGDe), a teacher-student framework that compiles agentic workflows into discrete execution plans comprising DAG topologies, system prompts, and deterministic executable code. The trailing "e" distinguishes SGDe from stochastic gradient descent. SGDe operates in a discrete semantic space where a frontier teacher generates natural-language critiques acting as directional gradients to iteratively refine the SLM's workflow artefacts. We formalise SGDe within a PAC learning framework, establishing sample-complexity bounds that enable convergence with as few as three training examples on targeted synthetic tasks by leveraging the teacher as a statistical prior. On a GSM-Hard-derived test set built via adversarial synthesis, compiled workflows reach 91.3% accuracy at m=5 and 99.3% at m=3 within the small-m regime motivated by Corollary 1, a +26.3% to +34.3% absolute improvement over state-of-the-art prompt optimisers. In the emerging paradigm of harness engineering, SGDe treats placement of deterministic code (which subtasks to delegate to a Python runtime versus retain as LLM calls) as a trace-driven, per-node optimisation target, generalising the whole-problem offloading of PAL and PoT. The teacher compiles two complementary deterministic structures: capability offloading, which delegates subtasks to Python when the SLM cannot execute them reliably, and structural consensus, which wraps variance-limited reasoning steps in fan-out/fan-in subgraphs aggregated by deterministic voting. 

---
# EvoMaster: A Foundational Agent Framework for Building Evolving Autonomous Scientific Agents at Scale 

**Authors**: Xinyu Zhu, Yuzhu Cai, Zexi Liu, Cheng Wang, Fengyang Li, Wenkai Jin, Wanxu Liu, Zehao Bing, Bingyang Zheng, Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xianghe Pang, Yaxin Du, Tingjia Miao, Yuzhi Zhang, Ruoxue Liao, Zhaohan Ding, Linfeng Zhang, Yanfeng Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.17406)  

**Abstract**: The convergence of large language models and agents is catalyzing a new era of scientific discovery: Agentic Science. While the scientific method is inherently iterative, existing agent frameworks are predominantly static, narrowly scoped, and lack the capacity to learn from trial and error. To bridge this gap, we present EvoMaster, a foundational evolving agent framework engineered specifically for Agentic Science at Scale. Driven by the core principle of continuous self-evolution, EvoMaster empowers agents to iteratively refine hypotheses, self-critique, and progressively accumulate knowledge across experimental cycles, faithfully mirroring human scientific inquiry. Crucially, as a domain-agnostic base harness, EvoMaster is exceptionally easy to scale up -- enabling developers to build and deploy highly capable, self-evolving scientific agents for arbitrary disciplines in approximately 100 lines of code. Built upon EvoMaster, we incubated the SciMaster ecosystem across domains such as machine learning, physics, and general science. Evaluations on four authoritative benchmarks (Humanity's Last Exam, MLE-Bench Lite, BrowseComp, and FrontierScience) demonstrate that EvoMaster achieves state-of-the-art scores of 41.1%, 75.8%, 73.3%, and 53.3%, respectively. It comprehensively outperforms the general-purpose baseline OpenClaw with relative improvements ranging from +159% to +316%, robustly validating its efficacy and generality as the premier foundational framework for the next generation of autonomous scientific discovery. EvoMaster is available at this https URL. 

---
# STRIDE: Strategic Iterative Decision-Making for Retrieval-Augmented Multi-Hop Question Answering 

**Authors**: Wei Chen, Lili Zhao, Zhi Zheng, HuiJun Hou, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17405)  

**Abstract**: Multi-hop question answering (MHQA) enables accurate answers to complex queries by retrieving and reasoning over evidence dispersed across multiple documents. Existing MHQA approaches mainly rely on iterative retrieval-augmented generation, which suffer from the following two major issues. 1) Existing methods prematurely commit to surface-level entities rather than underlying reasoning structures, making question decomposition highly vulnerable to lexical ambiguity. 2) Existing methods overlook the logical dependencies among reasoning steps, resulting in uncoordinated execution. To address these issues, we propose STRIDE, a framework that separates strategic planning, dynamic control, and grounded execution. At its core, a Meta-Planner first constructs an entity-agnostic reasoning skeleton to capture the abstract logic of the query, thereby deferring entity grounding until after the reasoning structure is established, which mitigates disambiguation errors caused by premature lexical commitment. A Supervisor then orchestrates sub-question execution in a dependency-aware manner, enabling efficient parallelization where possible and sequential coordination when necessary. By dynamically deciding whether to retrieve new evidence or infer from existing facts, it avoids redundant queries and error propagation, while fusing cross-branch information and reformulating failed queries to enhance robustness. Grounded fact extraction and logical inference are delegated to specialized execution modules, ensuring faithfulness through explicit separation of retrieval and reasoning. We further propose STRIDE-FT, a modular fine-tuning framework that uses self-generated execution trajectories from STRIDE, requiring neither human annotations nor stronger teacher models. Experiments show that STRIDE achieves robust and accurate reasoning, while STRIDE-FT effectively enhances open-source LLMs. 

---
# Phase-Scheduled Multi-Agent Systems for Token-Efficient Coordination 

**Authors**: Mohit Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2604.17400)  

**Abstract**: Multi-agent systems (MAS) powered by large language models suffer from severe token inefficiency arising from two compounding sources: (i) unstructured parallel execution, where all agents activate simultaneously irrespective of input readiness; and (ii) unrestricted context sharing, where every agent receives the full accumulated context regardless of relevance. Existing mitigation strategies - static pruning, hierarchical decomposition, and learned routing - treat coordination as a structural allocation problem and fundamentally ignore its temporal dimension. We propose Phase-Scheduled Multi-Agent Systems (PSMAS), a framework that reconceptualizes agent activation as continuous control over a shared attention space modeled on a circular manifold.
Each agent i is assigned a fixed angular phase theta_i in the range [0, 2*pi], derived from the task dependency topology; a global sweep signal phi(t) rotates at velocity omega, activating only agents within an angular window epsilon. Idle agents receive compressed context summaries, reducing per-step token consumption. We implement PSMAS on LangGraph, evaluate on four structured benchmarks (HotPotQA-MAS, HumanEval-MAS, ALFWorld-Multi, WebArena-Coord) and two unstructured conversational settings, and prove stability, convergence, and optimality results for the sweep dynamics. PSMAS achieves a mean token reduction of 27.3 percent (range 21.4-34.8 percent) while maintaining task performance within 2.1 percentage points of a fully activated baseline (p < 0.01, n = 500 per configuration), and outperforms the strongest learned routing baseline by 5.6 percentage points in token reduction with 2.0 percentage points less performance drop. Crucially, we show that scheduling and compression are independent sources of gain: scheduling alone accounts for 18-20 percentage points of reduction, robust to compression degradation up to alpha = 0.40. 

---
# Beyond Meta-Reasoning: Metacognitive Consolidation for Self-Improving LLM Reasoning 

**Authors**: Ziqing Zhuang, Linhai Zhang, Jiasheng Si, Deyu Zhou, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.17399)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities, and as existing approaches for enhancing LLM reasoning continue to mature, increasing attention has shifted toward meta-reasoning as a promising direction for further improvement. However, most existing meta-reasoning methods remain episodic: they focus on executing complex meta-reasoning routines within individual instances, but ignore the accumulation of reusable meta-reasoning skills across instances, leading to recurring failure modes and repeatedly high metacognitive effort. In this paper, we introduce Metacognitive Consolidation, a novel framework in which a model consolidates metacognitive experience from past reasoning episodes into reusable knowledge that improves future meta-reasoning. We instantiate this framework by structuring instance-level problem solving into distinct roles for reasoning, monitoring, and control to generate rich, attributable meta-level traces. These traces are then consolidated through a hierarchical, multi-timescale update mechanism that gradually forms evolving meta-knowledge. Experimental results demonstrate consistent performance gains across benchmarks and backbone models, and show that performance improves as metacognitive experience accumulates over time. 

---
# LLM-Guided Strategy Synthesis for Scalable Equality Saturation 

**Authors**: Chenyun Yin, Youwei Xiao, Yuze Luo, Yuyang Zou, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17364)  

**Abstract**: Equality saturation (EqSat) is a powerful optimization paradigm that compactly represents many equivalent programs in an e-graph and delays commitment until extraction selects a lowest-cost program. Making EqSat effective, therefore, requires not only domain-specific rewrite rules but also domain-specific strategies. Today, much of this strategy design is still manual, making it a major obstacle to automating e-graph-based compilers. Recent rule-synthesis frameworks can automatically infer large rewrite vocabularies from semantic specifications, but they also enlarge the rewrite space and further exacerbate e-graph explosion. Although large language models (LLMs) make automated strategy synthesis plausible, directly evolving backend code remains ineffective in practice. The search lacks reusable strategy abstractions and actionable feedback, and can easily trigger e-graph explosion or converge to poor designs.
We present EggMind, an LLM-guided, end-to-end framework for synthesizing reusable EqSat strategies. At its core, EggMind introduces a domain-specific language, EqSatL, to represent EqSat strategies as explicit and inspectable artifacts. It then proposes an LLM-guided agentic workflow, equipped with novel techniques including proof-derived rewrite motif caching and tractability guidance, to search efficiently for high-quality strategies while keeping synthesis stable under e-graph growth. Evaluation shows that EggMind substantially improves the resource-quality trade-off on vectorization benchmarks, reducing final cost by 45.1% and peak RAM by 69.1% relative to full EqSat. We further show that the same methodology transfers effectively to an XLA-based tensor compiler, and demonstrate its practical potential in a logic-synthesis case study with augmented rewrite spaces. 

---
# T-DuMpRa: Teacher-guided Dual-path Multi-prototype Retrieval Augmented framework for fine-grained medical image classification 

**Authors**: Zixuan Tang, Shen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17360)  

**Abstract**: Fine-grained medical image classification is challenged by subtle inter-class variations and visually ambiguous cases, where confidence estimates often exhibit uncertainty rather than being overconfident. In such scenarios, purely discriminative classifiers may achieve high overall accuracy yet still fail to distinguish between highly similar categories, leading to miscalibrated predictions. We propose T-DuMpRa, a teacher-guided dual-path multi-prototype retrieval-augmented framework, where discriminative classification and multi-prototype retrieval jointly drive both training and prediction. During training, we jointly optimize cross-entropy and supervised contrastive objectives to learn a cosine-compatible embedding geometry for reliable prototype matching. We further employ an exponential moving average (EMA) teacher to obtain smoother representations and build a multi-prototype memory bank by clustering teacher embeddings in the teacher embedding space. Our framework is plug-and-play: it can be easily integrated into existing classification models by constructing a compact prototype bank, thereby improving performance on visually ambiguous cases. At inference, we combine the classifier's predicted distribution with a similarity-based distribution computed via cosine matching to prototypes, and apply a conservative confidence-gated fusion that activates retrieval only when the classifier's prediction is uncertain and the retrieval evidence is decisive and conflicting, otherwise keeping confident predictions unchanged. On HAM10000 and ISIC2019, our method yields 0.68%-0.21% and 0.44%-2.69% improvements on 5 different backbones. And visualization analysis proves our model can enhance the model's ability to handle visually ambiguous cases. 

---
# Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling 

**Authors**: Zizhang Luo, Yuhao Luo, Youwei Xiao, Yansong Xu, Runlin Guo, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17353)  

**Abstract**: Large language models are increasingly deployed as complex agentic systems that scale with task complexity. While prior work has extensively explored model- and system-level scaling, algorithm- and task-level scaling remain largely unaddressed, constraining the full potential of agentic systems. At the algorithm level, allocating additional inference-time computation can enhance workflow capacity but introduces cross-path redundancy: overlapping computations across multiple reasoning branches. At the task level, complex tasks can be decomposed into subproblems and delegated across multiple agents for improved scalability and parallelism. However, existing infrastructures' scheduling is unaware of the existence of multiple agents, missing opportunities to optimize resource allocation.
We propose Hive, a multi-agent infrastructure that enables algorithm- and task-level scaling. Hive features a description frontend that captures per-agent behavior and supports test-time scaling algorithms. Leveraging this specification, our backend introduces two key mechanisms: Logits Cache that reuses intermediate logits across redundant sampling paths to mitigate cross-path redundancy at the algorithm level, and Agent-Aware Scheduling that efficiently allocates compute and KV-cache resources according to agent contributions at the task level. Experiments show that Logits Cache achieves an average speedup of $1.11\times$-$1.76\times$ for re-sampling, and Agent-Aware Scheduling reduces the hotspot miss rate by $33\%$-$51\%$. 

---
# SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization 

**Authors**: Yuncheng Hua, Sion Weatherhead, Mehdi Jafari, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17351)  

**Abstract**: Automated simulator construction requires distributional fidelity, distinguishing it from generic code generation. We identify two failure modes in long-horizon LLM agents: contextual drift and optimization instability arising from conflating structural and parametric errors. We propose SOCIA-EVO, a dual-anchored evolutionary framework. SOCIA-EVO introduces: (1) a static blueprint to enforce empirical constraints; (2) a bi-level optimization to decouple structural refinement from parameter calibration; and (3) a self-curating Strategy Playbook that manages remedial hypotheses via Bayesian-weighted retrieval. By falsifying ineffective strategies through execution feedback, SOCIA-EVO achieves robust convergence, generating simulators that are statistically consistent with observational data. The code and data of SOCIA-EVO are available here: this https URL. 

---
# Formal Foundations of Agentic Business Process Management 

**Authors**: Giuseppe De Giacomo, Timotheus Kampik, Lukas Kirchdorfer, Marco Montali, Christoph Weinhuber  

**Link**: [PDF](https://arxiv.org/pdf/2604.17347)  

**Abstract**: Just like traditional BPM systems, agentic BPM systems are built around a specification of the process under consideration. Their distinguishing feature, however, is that the execution of the process is driven by multiple autonomous decision-makers, referred to as agents. Since such agents cannot be fully controlled, the process specification is augmented with explicit objectives, or goals, assigned to the participating agents. Agents then pursue these goals, at least to the best of their efforts, under suitable assumptions on the behavior of others, by adopting appropriate strategies. Centrally, the organization enacting the process can use these specifications to provide guardrails on the decision-making capabilities of agents at the strategy level. This paper sets up the mathematical foundations of such systems in three key settings and analyzes four foundational problems of agentic BPM. 

---
# AutoSearch: Adaptive Search Depth for Efficient Agentic RAG via Reinforcement Learning 

**Authors**: Jingbo Sun, Wenyue Chong, Songjun Tu, Qichao Zhang, Yaocheng Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17337)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems enable large language models (LLMs) to solve complex tasks through multi-step interaction with external retrieval tools. However, such multi-step interaction often involves redundant search steps, incurring substantial computational cost and latency. Prior work limits search depth (i.e., the number of search steps) to reduce cost, but this often leads to underexploration of complex questions. To address this, we first investigate how search depth affects accuracy and find a minimal sufficient search depth that defines an accuracy-efficiency trade-off, jointly determined by question complexity and the agent's capability. Furthermore, we propose AutoSearch, a reinforcement learning (RL) framework that evaluates each search step via self-generated intermediate answers. By a self-answering mechanism, AutoSearch identifies the minimal sufficient search depth and promotes efficient search by rewarding its attainment while penalizing over-searching. In addition, reward mechanisms are introduced to stabilize search behavior and improve answer quality on complex questions. Extensive experiments on multiple benchmarks show that AutoSearch achieves a superior accuracy-efficiency trade-off, alleviating over-searching while preserving search quality. 

---
# Knows: Agent-Native Structured Research Representations 

**Authors**: Guangsheng Yu, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17309)  

**Abstract**: Research artifacts are distributed primarily as reader-oriented documents like PDFs. This creates a bottleneck for increasingly agent-assisted and agent-native research workflows, in which LLM agents need to infer fine-grained, task-relevant information from lengthy full documents, a process that is expensive, repetitive, and unstable at scale.
We introduce Knows, a lightweight companion specification that binds structured claims, evidence, provenance, and verifiable relations to existing research artifacts in a form LLM agents can consume directly. Knows addresses the gap with a thin YAML sidecar (KnowsRecord) that coexists with the original PDF, requiring no changes to the publication itself, and validated by a deterministic schema linter. We evaluate Knows on 140 comprehension questions across 20 papers spanning 14 academic disciplines, comparing PDF-only, sidecar-only, and hybrid conditions across six LLM agents of varying capacity. Weak models (0.8B--2B parameters) improve from 19--25\% to 47--67\% accuracy (+29 to +42 percentage points) when reading sidecar instead of PDF, while consuming 29--86\% fewer input tokens; an LLM-as-judge re-scoring confirms that weak-model sidecar accuracy (75--77\%) approaches stronger-model PDF accuracy (78--83\%). Beyond this controlled evaluation, a community sidecar hub at this https URL has already indexed over ten thousand publications and continues to grow daily, providing independent evidence that the format is adoption-ready at scale. 

---
# SkillFlow:Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents 

**Authors**: Ziao Zhang, Kou Shi, Shiting Huang, Avery Nie, Yu Zeng, Yiming Zhao, Zhen Fang, Qishen Su, Haibo Qiu, Wei Yang, Qingnan Ren, Shun Zou, Wenxuan Huang, Lin Chen, Zehui Chen, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17308)  

**Abstract**: As the capability frontier of autonomous agents continues to expand, they are increasingly able to complete specialized tasks through plug-and-play external skills. Yet current benchmarks mostly test whether models can use provided skills, leaving open whether they can discover skills from experience, repair them after failure, and maintain a coherent library over time. We introduce SkillFlow, a benchmark of 166 tasks across 20 families in which task construction within each family follows a Domain-Agnostic Execution Flow (DAEF) that defines an agent workflow framework, allowing these tasks to share a consistent workflow. Agents are evaluated under an Agentic Lifelong Learning protocol in which they begin without skills, solve tasks sequentially within each family, externalize lessons through trajectory- and rubric-driven skill patches, and carry the updated library forward. Experiments reveal a substantial capability gap. For Claude Opus 4.6, lifelong skill evolution improves task success from 62.65% to 71.08% (+8.43 points). However, high skill usage does not necessarily imply high utility: Kimi K2.5 gains only +0.60 points despite 66.87% skill usage, while Qwen-Coder-Next reaches only a 44.58% task completion rate and still regresses relative to the vanilla setting. SkillFlow contributes a structured testbed for this direction and an in-depth empirical analysis of skill discovery, patching, transfer, and their failure modes under lifelong evaluation. 

---
# Efficient Test-Time Scaling via Temporal Reasoning Aggregation 

**Authors**: Jiakun Li, Xingwei He, Kefan Li, Hongzheng Chai, Hongyue Yu, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17304)  

**Abstract**: Test-time scaling improves the reasoning performance of large language models but often results in token-inefficient overthinking, where models continue reasoning beyond what is necessary for a correct answer. Existing dynamic early-exit methods typically rely on single-step confidence signals, which are often unreliable for detecting reasoning convergence in multi-step settings. To mitigate this limitation, we propose TRACE, a training-free framework for efficient test-time scaling that determines when to terminate reasoning based on temporal aggregation of multi-step evidence rather than instantaneous signals. TRACE detects reasoning convergence over time by aggregating two complementary signals across recent reasoning steps: answer consistency, capturing the persistence of predicted answers, and confidence trajectory, modeling the temporal evolution of model confidence. Benefiting from these two factors, TRACE can accurately determine whether the reasoning process has converged, thereby promptly halting inference and effectively avoiding redundant reasoning steps. Extensive experiments on multiple challenging benchmarks show that TRACE reduces reasoning token usage by 25-30% on average while maintaining accuracy within 1-2% of full-length reasoning, consistently outperforming existing dynamic reasoning methods. 

---
# LLaTiSA: Towards Difficulty-Stratified Time Series Reasoning from Visual Perception to Semantics 

**Authors**: Yueyang Ding, HaoPeng Zhang, Rui Dai, Yi Wang, Tianyu Zong, Kaikui Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17295)  

**Abstract**: Comprehensive understanding of time series remains a significant challenge for Large Language Models (LLMs). Current research is hindered by fragmented task definitions and benchmarks with inherent ambiguities, precluding rigorous evaluation and the development of unified Time Series Reasoning Models(TSRMs). To bridge this gap, we formalize Time Series Reasoning (TSR) via a four-level taxonomy of increasing cognitive complexity. We introduce HiTSR, a hierarchical time series reasoning dataset comprising 83k samples with diverse task combinations and verified Chain-of-Thought (CoT) trajectories. Leveraging HiTSR, we propose LLaTiSA, a strong TSRM that integrates visualized patterns with precision-calibrated numerical tables to enhance the temporal perception of Vision-Language Models (VLMs). Through a multi-stage curriculum fine-tuning strategy, LLaTiSA achieves superior performance and exhibits robust out-of-distribution generalization across diverse TSR tasks and real-world scenarios. Our code is available at this https URL. 

---
# HalluClear: Diagnosing, Evaluating and Mitigating Hallucinations in GUI Agents 

**Authors**: Chao Jin, Wenkui Yang, Hao Sun, Yuqi Liao, Qianyi Jiang, Kai Zhou, Jie Cao, Ran He, Huaibo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17284)  

**Abstract**: While progress in GUI agents has been largely driven by industrial-scale training, ungrounded hallucinations often trigger cascading failures in real-world this http URL general VLM domains, the GUI agent field lacks a hallucination-focused suite for fine-grained diagnosis, reliable evaluation, and targeted this http URL bridge this gap, we introduce HalluClear, a comprehensive suite for hallucination mitigation in GUI agents as a complement to computation-intensive scaling. HalluClear comprises: (1) a GUI-specific hallucination taxonomy derived from empirical failure analysis; (2) a calibrated three-stage evaluation workflow which enhances VLM-as-a-judge reliability via expert-annotated benchmarking and ensemble credibility estimation; and (3) a mitigation scheme based on closed-loop structured reasoning, enabling lightweight continual post-training with cold-start initialization for both generalist and GUI-specialist agents. Experiments across representative agents and public benchmarks demonstrate that post-training on only 9K samples within our suite can significantly reduce hallucinations, thereby improving grounding and action fidelity, offering a compute-efficient pathway to robust GUI automation. 

---
# The Continuity Layer: Why Intelligence Needs an Architecture for What It Carries Forward 

**Authors**: Samuel Sameer Tanguturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17273)  

**Abstract**: The most important architectural problem in AI is not the size of the model but the absence of a layer that carries forward what the model has come to understand. Sessions end. Context windows fill. Memory APIs return flat facts that the model has to reinterpret from scratch on every read. The result is intelligence that is powerful per session and amnesiac across time. This position paper argues that the layer which fixes this, the continuity layer, is the most consequential piece of infrastructure the field has not yet built, and that the engineering work to build it has begun in public. The formal evaluation framework for the property described here is the ATANT benchmark (arXiv:2604.06710), published separately with evaluation results on a 250-story corpus; a companion paper (arXiv:2604.10981) positions this framework against existing memory, long-context, and agentic-memory benchmarks. The paper defines continuity as a system property with seven required characteristics, distinct from memory and from retrieval; describes a storage primitive (Decomposed Trace Convergence Memory) whose write-time decomposition and read-time reconstruction produce that property; maps the engineering architecture to the theological pattern of kenosis and the symbolic pattern of Alpha and Omega, and argues this mapping is structural rather than metaphorical; proposes a four-layer development arc from external SDK to hardware node to long-horizon human infrastructure; examines why the physics limits now constraining the model layer make the continuity layer newly consequential; and argues that the governance architecture (privacy implemented as physics rather than policy, founder-controlled class shares on non-negotiable architectural commitments) is inseparable from the product itself. 

---
# Rectification Difficulty and Optimal Sample Allocation in LLM-Augmented Surveys 

**Authors**: Zikun Ye, Hema Yoganarasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17267)  

**Abstract**: Large Language Models can generate synthetic survey responses at low cost, but their accuracy varies unpredictably across questions. We study the design problem of allocating a fixed budget of human respondents across estimation tasks when cheap LLM predictions are available for every task. Our framework combines three components. First, building on Prediction-Powered Inference, we characterize a question-specific rectification difficulty that governs how quickly the estimator's variance decreases with human sample size. Second, we derive a closed-form optimal allocation rule that directs more human labels to tasks where the LLM is least reliable. Third, since rectification difficulty depends on unobserved human responses for new surveys, we propose a meta-learning approach, trained on historical data, that predicts it for entirely new tasks without pilot data. The framework extends to general M-estimation, covering regression coefficients and multinomial logit partworths for conjoint analysis. We validate the framework on two datasets spanning different domains, question types, and LLMs, showing that our approach captures 61-79% of the theoretically attainable efficiency gains, achieving 11.4% and 10.5% MSE reductions without requiring any pilot human data for the target survey. 

---
# Safe and Policy-Compliant Multi-Agent Orchestration for Enterprise AI 

**Authors**: Vinil Pasupuleti, Shyalendar Reddy Allala, Siva Rama Krishna Varma Bayyavarapu, Shrey Tyagi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17240)  

**Abstract**: Enterprise AI systems increasingly deploy multiple intelligent agents across mission-critical workflows that must satisfy hard policy constraints, bounded risk exposure, and comprehensive auditability (SOX, HIPAA, GDPR). Existing coordination methods - cooperative MARL, consensus protocols, and centralized planners - optimize expected reward while treating constraints implicitly. This paper introduces CAMCO (Constraint-Aware Multi-Agent Cognitive Orchestration), a runtime coordination layer that models multi-agent decision-making as a constrained optimization problem. CAMCO integrates three mechanisms: (i) a constraint projection engine enforcing policy-feasible actions via convex projection, (ii) adaptive risk-weighted Lagrangian utility shaping, and (iii) an iterative negotiation protocol with provably bounded convergence. Unlike training-time constrained RL, CAMCO operates as deployment-time middleware compatible with any agent architecture, with policy predicates designed for direct integration with production engines such as OPA. Evaluation across three enterprise scenarios - including comparison against a constrained Lagrangian MARL baseline - demonstrates zero policy violations, risk exposure below threshold (mean ratio 0.71), 92-97% utility retention, and mean convergence in 2.4 iterations. 

---
# Yanasse: Finding New Proofs from Deep Vision's Analogies, Part 1 

**Authors**: Alexandre Linhares  

**Link**: [PDF](https://arxiv.org/pdf/2604.17229)  

**Abstract**: Project Yanasse presents a method for discovering new proofs of theorems in one area of mathematics by transferring proof strategy patterns (e.g., Lean 4 tactic invocation patterns) from a structurally distant area. The system extracts tactic usage distributions across 27 top-level areas of Mathlib (217,133 proof states), computes z-scores to identify tactics that are heavily used in a source area but rare or absent in a target area, matches source and target proof states via GPU-accelerated NP-hard analogy (running on a MacBook Air via Apple's MPS backend), and then asks an AI reasoning agent to semantically adapt--not symbol-substitute--the source tactics invocation pattern to the target theorem. In this first part of the study, the method is applied to the pair Probability -> Representation Theory, producing 4 Lean-verified new proofs out of 10 attempts (40%). The proofs compile with zero sorry declarations. The key finding is that tactic schemas decompose into a head (domain-gated, rarely transfers) and a modifier (domain-general, often transfers): filter upwards's head fails in representation theory (no Filter structure), but its [LIST] with {\omega} modifier transfers cleanly as ext1 + simp [LIST] + rfl. Crucially, the underlying matching engine--deep vision this http URL--is entirely domain independent: the same optimization code for an NP-hard matching that matches chess positions by analogy matches Lean proof states by analogy, without knowing which domain it is processing. Only a relation extractor is domain-specific. 

---
# Beyond the Basics: Leveraging Large Language Model for Fine-Grained Medical Entity Recognition 

**Authors**: Nwe Ni Win, Jim Basilakis, Steven Thomas, Seyhan Yazar, Laura Pierce, Stephanie Liu, Paul M. Middleton, Nasser Ghadiri, X. Rosalind Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17214)  

**Abstract**: Extracting clinically relevant information from unstructured medical narratives such as admission notes, discharge summaries, and emergency case histories remains a challenge in clinical natural language processing (NLP). Medical Entity Recognition (MER) identifies meaningful concepts embedded in these records. Recent advancements in large language models (LLMs) have shown competitive MER performance; however, evaluations often focus on general entity types, offering limited utility for real-world clinical needs requiring finer-grained extraction. To address this gap, we rigorously evaluated the open-source LLaMA3 model for fine-grained medical entity recognition across 18 clinically detailed categories. To optimize performance, we employed three learning paradigms: zero-shot, few-shot, and fine-tuning with Low-Rank Adaptation (LoRA). To further enhance few-shot learning, we introduced two example selection methods based on token- and sentence-level embedding similarity, utilizing a pre-trained BioBERT model. Unlike prior work assessing zero-shot and few-shot performance on proprietary models (e.g., GPT-4) or fine-tuning different architectures, we ensured methodological consistency by applying all strategies to a unified LLaMA3 backbone, enabling fair comparison across learning settings. Our results showed that fine-tuned LLaMA3 surpasses zero-shot and few-shot approaches by 63.11% and 35.63%, respectivel respectively, achieving an F1 score of 81.24% in granular medical entity extraction. 

---
# Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration 

**Authors**: Sukwon Yun, Jie Peng, Pingzhi Li, Wendong Fan, Jie Chen, James Zou, Guohao Li, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.17148)  

**Abstract**: With an ever-growing zoo of LLMs and benchmarks, the need to orchestrate multiple models for improved task performance has never been more pressing. While frameworks like Mixture-of-Agents (MoA) attempt to coordinate LLMs, they often fall short in terms of (1) selecting relevant agents, (2) facilitating effective intra-agent communication, and (3) integrating responses efficiently. In this work, we propose Graph-of-Agents (GoA), a new graph-based framework for modeling multi-agent LLM communication. Our approach begins with node sampling, selecting only the most relevant agents by leveraging model cards that summarize each model's domain, task specialization, and other characteristics. Next, we construct edges between the selected agents by evaluating their responses against one another to determine relevance ordering. Directed message passing is then performed from highly relevant agents to less relevant ones to enhance their responses, followed by reverse message passing to refine the original responses of the more relevant agents. Finally, the updated responses are aggregated via graph-based pooling (e.g., max or mean pooling) to produce a single, unified answer. We evaluate GoA on diverse multi-domain benchmarks (MMLU, MMLU-Pro, GPQA) and domain-specific benchmarks (MATH, HumanEval, MedMCQA), with an agent pool of 6 LLMs spanning multiple domains. Surprisingly, GoA achieves superior performance using only 3 selected agents, outperforming recent multi-agent LLM baselines that utilize all 6 agents simultaneously. By adopting a graph structure, GoA offers both scalability and effectiveness through structured message passing-positioning it as a strong candidate for navigating the challenges of the ever-growing LLM zoo. Code is available at: this https URL. 

---
# Local Inconsistency Resolution: The Interplay between Attention and Control in Probabilistic Models 

**Authors**: Oliver E. Richardson, Mandana Samiei, Mehran Shakerinava, Joseph D. Viviano, Abdessamad El Kabid, Ali Parviz, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2604.17140)  

**Abstract**: We present a generic algorithm for learning and approximate inference with an intuitive epistemic interpretation: iteratively focus on a subset of the model and resolve inconsistencies using the parameters under control. This framework, which we call Local Inconsistency Resolution (LIR) is built upon Probabilistic Dependency Graphs (PDGs), which provide a flexible representational foundation capable of capturing inconsistent beliefs. We show how LIR unifies and generalizes a wide variety of important algorithms in the literature, including the Expectation-Maximization (EM) algorithm, belief propagation, adversarial training, GANs, and GFlowNets. In the last case, LIR actually suggests a more natural loss, which we demonstrate improves GFlowNet convergence. Each method can be recovered as a specific instance of LIR by choosing a procedure to direct focus (attention and control). We implement this algorithm for discrete PDGs and study its properties on synthetically generated PDGs, comparing its behavior to the global optimization semantics of the full PDG. 

---
# If Only My CGM Could Speak: A Privacy-Preserving Agent for Question Answering over Continuous Glucose Data 

**Authors**: Yanjun Cui, Ali Emami, Temiloluwa Prioleau, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.17133)  

**Abstract**: Continuous glucose monitors (CGMs) used in diabetes care collect rich personal health data that could improve day-to-day self-management. However, current patient platforms only offer static summaries which do not support inquisitive user queries. Large language models (LLMs) could enable free-form inquiries about continuous glucose data, but deploying them over sensitive health records raises privacy and accuracy concerns. In this paper, we present CGM-Agent, a privacy-preserving framework for question answering over personal glucose data. In our design, the LLM serves purely as a reasoning engine that selects analytical functions. All computation occurs locally, and personal health data never leaves the user's device. For evaluation, we construct a benchmark of 4,180 questions combining parameterized question templates with real user queries and ground truth derived from deterministic program execution. Evaluating 6 leading LLMs, we find that top models achieve 94\% value accuracy on synthetic queries and 88\% on ambiguous real-world queries. Errors stem primarily from intent and temporal ambiguity rather than computational failures. Additionally, lightweight models achieve competitive performance in our agent design, suggesting opportunities for low-cost deployment. We release our code and benchmark to support future work on trustworthy health agents. 

---
# Complementing Self-Consistency with Cross-Model Disagreement for Uncertainty Quantification 

**Authors**: Kimia Hamidieh, Veronika Thost, Walter Gerych, Mikhail Yurochkin, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17112)  

**Abstract**: Large language models (LLMs) often produce confident yet incorrect responses, and uncertainty quantification is one potential solution to more robust usage. Recent works routinely rely on self-consistency to estimate aleatoric uncertainty (AU), yet this proxy collapses when models are overconfident and produce the same incorrect answer across samples. We analyze this regime and show that cross-model semantic disagreement is higher on incorrect answers precisely when AU is low. Motivated by this, we introduce an epistemic uncertainty (EU) term that operates in the black-box access setting: EU uses only generated text from a small, scale-matched ensemble and is computed as the gap between inter-model and intra-model sequence-semantic similarity. We then define total uncertainty (TU) as the sum of AU and EU. In a comprehensive study across five 7-9B instruction-tuned models and ten long-form tasks, TU improves ranking calibration and selective abstention relative to AU, and EU reliably flags confident failures where AU is low. We further characterize when EU is most useful via agreement and complementarity diagnostics. 

---
# Understanding and Enforcing Weight Disentanglement in Task Arithmetic 

**Authors**: Shangge Liu, Yuehan Yin, Lei Wang, Qi Fan, Yinghuan Shi, Wenbin Li, Yang Gao, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17078)  

**Abstract**: Task arithmetic provides an efficient, training-free way to edit pre-trained models, yet lacks a fundamental theoretical explanation for its success. The existing concept of ``weight disentanglement" describes the ideal outcome of non-interfering task composition but does not reveal its underlying cause. Crucially, what intrinsic properties of the pre-trained model ($\theta_0$) or the task vectors ($\tau_t$) enable this disentanglement remains underexplored. In this paper, we introduce Task-Feature Specialization (TFS), a model's ability to allocate distinct internal features to different tasks, as the fundamental principle. We first prove that TFS is a sufficient condition for weight disentanglement. More importantly, we find that TFS also gives rise to an observable geometric consequence: weight vector orthogonality. This positions TFS as the common cause for both the desired functional outcome (disentanglement) and a measurable geometric property (orthogonality). This relationship provides the key insight for our method: since the abstract TFS property is intractable to enforce directly, we can instead promote weight disentanglement by shaping its concrete geometric consequence, orthogonality. Therefore, we propose OrthoReg, a simple and effective regularization method that actively enforces an internal orthogonal structure on weight updates ($\Delta W$) that constitute $\tau_t$ during fine-tuning. And we theoretically prove that OrthoReg promotes disentanglement. Extensive experiments demonstrate that OrthoReg consistently and significantly enhances the performance of various task arithmetic methods. Code is available at \href{this https URL}{this https URL}. 

---
# Harness as an Asset: Enforcing Determinism via the Convergent AI Agent Framework (CAAF) 

**Authors**: Tianbao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17025)  

**Abstract**: Large Language Models (LLMs) produce a controllability gap in safety-critical engineering: even low rates of undetected constraint violations render a system undeployable. Current orchestration paradigms suffer from sycophantic compliance, context attention decay [Liu et al., 2024], and stochastic oscillation during self-correction [Huang et al., 2024].
We introduce the Convergent AI Agent Framework (CAAF), which transitions agentic workflows from open-loop generation to closed-loop Fail-Safe Determinism via three pillars: (1) Recursive Atomic Decomposition with physical context firewalls; (2) Harness as an Asset, formalizing domain invariants into machine-readable registries enforced by a deterministic Unified Assertion Interface (UAI); and (3) Structured Semantic Gradients with State Locking for monotonic convergence.
Empirical evaluation across two domains -- SAE Level 3 (L3) autonomous driving (AD) (n=30, 7 conditions) and pharmaceutical continuous flow reactor design (n=20, 4 conditions including a Mono+UAI ablation) -- shows that CAAF-all-GPT-4o-mini achieves 100% paradox detection while monolithic GPT-4o achieves 0% (even at temperature=0). The pharmaceutical benchmark features 7 simultaneous constraints with nonlinear Arrhenius interactions and a 3-way minimal unsatisfiable subset, representing a structurally harder challenge than the 2-constraint AD paradox. Alternative multi-agent architectures (debate, sequential checking) also achieve 0% across 80 trials, confirming that CAAF's reliability derives from its deterministic UAI, not from multi-agent orchestration per se. A Mono+UAI ablation (95%) isolates UAI as the core contribution. CAAF's reliability is invariant to prompt hints; all components use a single commodity model, enabling fully offline deployment. 

---
# Mini-BEHAVIOR-Gran: Revealing U-Shaped Effects of Instruction Granularity on Language-Guided Embodied Agents 

**Authors**: Sukai Huang, Chenyuan Zhang, Fucai Ke, Zhixi Cai, Gholamreza Haffari, Lizhen Qu, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17019)  

**Abstract**: Instruction granularity is an important yet poorly controlled variable in language-guided embodied AI. Existing benchmarks typically pair each task with a single static instruction, making it difficult to study how agent behavior changes when the same task is described at different levels of detail. We introduce Mini-BEHAVIOR-Gran, a new benchmark for controlled studies of instruction granularity that extends Mini-BEHAVIOR with multiple instruction variants per task, ranging from high-level goal descriptions to step-by-step guidance. Using this benchmark, we compare four candidate metrics for cross-task granularity quantification: token count, entity count, action-verb count, and planning-width, and find that width correlates most consistently with agent performance. Using width to organize training and evaluation further reveals a non-monotonic U-shaped relationship between instruction granularity and performance, with peaks at both fine and coarse extremes. Further analysis suggests that the coarse-granularity performance rebound is associated with shallow grounding, where agents learn vision-dominant policies. 

---
# Small Model as Master Orchestrator: Learning Unified Agent-Tool Orchestration with Parallel Subtask Decomposition 

**Authors**: Wenzhen Yuan, Wutao Xiong, Fanchen Yu, Shengji Tang, Ting Liu, Tao Chen, Peng Ye, Yuzhuo Fu, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.17009)  

**Abstract**: Multi-agent systems (MAS) demonstrate clear advantages in tackling complex problems by coordinating diverse agents and external tools. However, most existing orchestration methods rely on static workflows or serial agent scheduling, and are further constrained by heterogeneous interface protocols between tools and agents. This leads to high system complexity and poor extensibility. To mitigate these issues, we propose Agent-as-Tool, a unified parallel orchestration paradigm that abstracts both agents and tools into a standardized, learnable action space with protocol normalization and explicit state feedback. Building on this paradigm, we train a lightweight orchestrator, ParaManager, which decouples planning decisions from subtask solving, enabling state-aware parallel subtask decomposition, delegation, and asynchronous execution. For training, we adopt a two-stage ParaManager training pipeline. It improves robustness by incorporating supervised fine-tuning (SFT) trajectories equipped with recovery mechanisms, and further applies reinforcement learning (RL) to achieve an optimal balance among task success, protocol compliance, diversity, and reasoning efficiency. Experiments show that ParaManager achieves strong performance across multiple benchmarks and exhibits robust generalization under unseen model pools. 

---
# Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification 

**Authors**: Jiawen Wen, Penglei Sun, Wenjie Zhang, Suixuan Qiu, Weisheng Xu, Xiaofei Yang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16993)  

**Abstract**: As embodied AI transitions to real-world deployment, the success of the Vision-and-Language Navigation (VLN) task tends to evolve from mere reachability to social compliance. However, current agents suffer from a "goal-driven trap", prioritizing physical geometry ("can I go?") over semantic rules ("may I go?"), frequently overlooking subtle regulatory constraints. To bridge this gap, we establish Rule-VLN, the first large-scale urban benchmark for rule-compliant navigation. Spanning a massive 29k-node environment, it injects 177 diverse regulatory categories into 8k constrained nodes across four curriculum levels, challenging agents with fine-grained visual and behavioral constraints. We further propose the Semantic Navigation Rectification Module (SNRM), a universal, zero-shot module designed to equip pre-trained agents with safety awareness. SNRM integrates a coarse-to-fine visual perception VLM framework with an epistemic mental map for dynamic detour planning. Experiments demonstrate that while Rule-VLN challenges state-of-the-art models, SNRM significantly restores navigation capabilities, reducing CVR by 19.26% and boosting TC by 5.97%. 

---
# A phenotype-driven and evidence-governed framework for knowledge graph enrichment and hypotheses discovery in population data 

**Authors**: Adela Bâra, Simona-Vasilica Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2604.16982)  

**Abstract**: Current knowledge graph (KG) construction methods are confirmatory, focusing on recovering known relationships rather than identifying novel or context-dependent nodes. This paper proposes a phenotype-driven and evidence-governed framework that shifts the paradigm toward structured hypothesis discovery and controlled KG expansion. The approach integrates graph neural networks (GNNs) for phenotype discovery, causal inference, probabilistic reasoning and large language models (LLMs) for hypothesis generation and claim extraction within a unified pipeline. The framework prioritizes relationships that are both structurally supported by data and underexplored in the literature. KG expansion is formulated as a multi-objective optimization problem, where candidate claims are jointly evaluated in terms of relevance, structural validation and novelty. Pareto-optimal selection enables the identification of non-dominated claims that balance confirmation and discovery, avoiding trivial or redundant knowledge inclusion. Experiments on heterogeneous population datasets demonstrate that the proposed framework produces more interpretable phenotypes, reveals context-dependent causal structures and generates high-quality claims that align with both data and scientific evidence. Compared to rule-based and LLM-only baselines, the method achieves the best trade-off across plausibility, novelty, validation and relevance. In retrieval-augmented settings, it significantly improves performance (Recall@5=0.98) while reducing hallucination rates (0.05), highlighting its effectiveness in grounding LLM outputs. 

---
# MCPO: Mastery-Consolidated Policy Optimization for Large Reasoning Models 

**Authors**: Zhaokang Liao, Yingguo Gao, Yi Yang, Yongheng Hu, Jingting Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.16972)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising approach to improve the reasoning abilities of Large Language Models (LLMs). Among RLVR algorithms, Group Relative Policy Optimization (GRPO) and its variants have demonstrated strong performance and high training efficiency. However, GRPO-style objectives exhibit two issues on high accuracy prompts including mastered prompts (rollout accuracy =1) and majority-correct prompts (rollout accuracy in (0.5,1)). For mastered prompts, group-relative advantages vanish, yielding no training signal and unconstrained policy drift that can cause forgetting. For majority-correct prompts, the induced query weight shrinks as accuracy increases, weakening consolidation from partial correctness to mastery. To alleviate this, we propose Mastery-Consolidated Policy Optimization (MCPO), which introduces (i) a hinge-KL regularizer applied exclusively to mastered prompts to bound harmful policy drift between successive gradient steps, and (ii) a weighting mechanism that prioritizes majority-correct prompts to better allocate optimization effort. Extensive experiments across three mathematical benchmarks demonstrate that MCPO consistently improves pass@1 performance. Counter-intuitively, rather than restricting exploration, MCPO boosts pass@k metrics, indicating that mastery consolidation further catalyzes solution diversity. 

---
# AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction 

**Authors**: Pollawat Hongwimol, Haoning Shang, Chutong Wang, Zhichao Wan, Yi Gao, Yuanming Li, Lin Gui, Wenhao Sun, Cheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16950)  

**Abstract**: Product attribute extraction in e-commerce is bottlenecked by ontologies that are inconsistent, incomplete, and costly to maintain. We present AutoPKG, a multi-agent Large Language Model (LLM) framework that automatically constructs a Product-attribute Knowledge Graph (PKG) from multimodal product content. AutoPKG induces product types and type-specific attribute keys on demand, extracts attribute values from text and images, and consolidates updates through a centralized decision agent that maintains a globally consistent canonical graph. We also propose an evaluation protocol for dynamic PKGs that measures type and key validity, consolidation quality, and edge-level accuracy for value assertions after canonicalization. On a large real-world marketplace catalog dataset from Lazada (Alibaba), AutoPKG achieves up to 0.953 Weighted Knowledge Efficiency (WKE) for product types, 0.724 WKE for attribute keys, and 0.531 edge-level F1 for multimodal value extraction. Across three public benchmarks, our method improves edge-level exact-match F1 by 0.152 and yields a precision gain of 0.208 on the attribute extraction application. Online A/B tests show that AutoPKG-derived attributes increase Gross Merchandise Value (GMV) in Badge by 3.81 percent, in Search by 5.32 percent, and in Recommendation by 7.89 percent, supporting the practical value of AutoPKG in production. 

---
# LLMs can persuade only psychologically susceptible humans on societal issues, via trust in AI and emotional appeals, amid logical fallacies 

**Authors**: Alexis Carrillo, Salvatore Citraro, Ali Aghazhadeh Ardebili, Enrique Taietta, Giulio Rossetti, Emilio Ferrara, Giuseppe Alessandro Veltri, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2604.16935)  

**Abstract**: Scarce longitudinal evidence examines LLMs' persuasiveness and humanness along time-evolving psychological frameworks. We introduce Talk2AI, a longitudinal framework quantifying psycho-social, reasoning and affective dimensions of LLMs' persuasiveness about polarizing societal topics. In a four-way longitudinal setup, Talk2AI's 770 participants engaged in structured conversations with one of four leading LLMs on topics like climate change, social media misinformation, and math anxiety. This produced 3,080 conversations over 60,000 turns. After each wave, participants reported conviction in their initial topic stance, perceived opinion change, LLM's perceived humanness, a self-donation to the topic and a textual explanation. Feedback time series showed longitudinal inertia in convictions, indicating some human anchoring to initial opinions even after repeated exposure to AI-generated arguments. Interestingly, NLP analyses revealed that both humans and LLMs relied on fallacious reasoning in 1 conversational quip every 6, countering the ``LLMs as superior systems" stereotype behind LLMs' cognitive surrender. LLMs' perceived humanness was most learnable from sociodemographic, psychological and engagement features ($R^2=0.44$), followed by opinion change ($R^2=0.34$), conviction ($R^2=0.26$) and personal endowment ($R^2=0.24$). Crucially, explainable AI (XAI) indicated: (i) the presence of individuals more susceptible to LLM-based opinion changes; (ii) psychological susceptibility to LLM-convincing consisted of having more trust in LLMs, being more agreeable and extraverted and with a higher need for cognition. A multiverse approach with mixed-effects models confirmed XAI results, alongside strong individual differences. Talk2AI provides a grounded framework and evidence for detecting how GenAI can influence human opinions via multiple psycho-social pathways in AI-human digital platforms. 

---
# Playing Psychic: Using Thought Trees to Predict Reasoning Models Accuracy on Coding Tasks 

**Authors**: Jiaxin Fang, Runyuan He, Sahil Bhatia, Neel Gajare, Alvin Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.16931)  

**Abstract**: Recent advances in large language models (LLMs) have shown that test-time scaling can substantially improve model performance on complex tasks, particularly in the coding domain. Under this paradigm, models use a larger token budget during inference to generate intermediate reasoning traces before producing a final answer. However, current evaluations primarily rely on competitive programming benchmarks, which may not capture the full range of reasoning abilities. In this work, we perform a systematic study of frontier reasoning models to understand their performance on real-world coding benchmarks. To gain more insights into the performance of such models, we devise a programmatic way to {\em automatically generate} coding tasks of arbitrary difficulty and structure from existing benchmarks. Using this framework, our analysis reveals that the structure of a reasoning trace, not just its contents, is a strong predictor of correctness. Motivated by this, we propose structured thought-trees as means to represent reasoning traces. To illustrate their use, we train a lightweight classifier on features extracted from thought-trees to predict trace correctness, and demonstrate that flagging and retrying structurally anomalous traces based on the extracted features yields consistent gains at lower complexity levels. 

---
# Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy 

**Authors**: Junxi Wu, Kailin Huang, Dongjian Hu, Bin Chen, Hao Wu, Shu-Tao Xia, Changliang Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.16923)  

**Abstract**: Detecting AI-generated text is an important but challenging problem. Existing likelihood-based detection methods are often sensitive to content complexity and may exhibit unstable performance. In this paper, our key insight is that modern Large Language Models (LLMs) undergo alignment (including fine-tuning and preference tuning), leaving a measurable distributional imprint. We theoretically derive this imprint by abstracting the alignment process as a sequence of constrained optimization steps, showing that the log-likelihood ratio can naturally decompose into implicit instructional biases and preference rewards. We refer to this quantity as the Alignment Imprint. Furthermore, to mitigate the instability in high-entropy regions, we introduce Log-likelihood Alignment Preference Discrepancy (LAPD), a standardized information-weighted statistic based on alignment imprint. We provide statistical guarantee that alignment-based statistics dominate Fast-DetectGPT in performance. We also theoretically show that LAPD strictly improves the unweighted alignment scores when the aligned and base models are close in distribution. Extensive experiments show that LAPD achieves an improvement 45.82% relative to the strongest existing baselines, yielding large and consistent gains across all settings. 

---
# ClimAgent: LLM as Agents for Autonomous Open-ended Climate Science Analysis 

**Authors**: Hao Wang, Jindong Han, Wei Fan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16922)  

**Abstract**: Climate research is pivotal for mitigating global environmental crises, yet the accelerating volume of multi-scale datasets and the complexity of analytical tools have created significant bottlenecks, constraining scientific discovery to fragmented and labor-intensive workflows. While the emergence Large Language Models (LLMs) offers a transformative paradigm to scale scientific expertise, existing explorations remain largely confined to simple Question-Answering (Q&A) tasks. These approaches often oversimplify real-world challenges, neglecting the intricate physical constraints and the data-driven nature required in professional climate this http URL bridge this gap, we introduce ClimAgent, a general-purpose autonomous framework designed to execute a wide spectrum of research tasks across diverse climate sub-fields. By integrating a unified tool-use environment with rigorous reasoning protocols, ClimAgent transcends simple retrieval to perform end-to-end modeling and this http URL foster systematic evaluation, we propose ClimaBench, the first comprehensive benchmark for real-world climate discovery. It encompasses challenging problems spanning 5 distinct task categories derived from professional scenarios between 2000 and 2025. Experiments on ClimaBench demonstrate that ClimAgent significantly outperforms state-of-the-art baselines, achieving a 40.21% improvement over original LLM solutions in solution rigorousness and practicality. Our code are available at this https URL. 

---
# The Cognitive Penalty: Ablating System 1 and System 2 Reasoning in Edge-Native SLMs for Decentralized Consensus 

**Authors**: Syed Muhammad Aqdas Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16913)  

**Abstract**: Decentralized Autonomous Organizations (DAOs) are inclined explore Small Language Models (SLMs) as edge-native constitutional firewalls to vet proposals and mitigate semantic social engineering. While scaling inference-time compute (System 2) enhances formal logic, its efficacy in highly adversarial, cryptoeconomic governance environments remains underexplored. To address this, we introduce Sentinel-Bench, an 840-inference empirical framework executing a strict intra-model ablation on Qwen-3.5-9B. By toggling latent reasoning across frozen weights, we isolate the impact of inference-time compute against an adversarial Optimism DAO dataset. Our findings reveal a severe compute-accuracy inversion. The autoregressive baseline (System 1) achieved 100% adversarial robustness, 100% juridical consistency, and state finality in under 13 seconds. Conversely, System 2 reasoning introduced catastrophic instability, fundamentally driven by a 26.7% Reasoning Non-Convergence (cognitive collapse) rate. This collapse degraded trial-to-trial consensus stability to 72.6% and imposed a 17x latency overhead, introducing critical vulnerabilities to Governance Extractable Value (GEV) and hardware centralization. While rare (1.5% of adversarial trials), we empirically captured "Reasoning-Induced Sycophancy," where the model generated significantly longer internal monologues (averaging 25,750 characters) to rationalize failing the adversarial trap. We conclude that for edge-native SLMs operating under Byzantine Fault Tolerance (BFT) constraints, System 1 parameterized intuition is structurally and economically superior to System 2 iterative deliberation for decentralized consensus.
Code and Dataset: this https URL 

---
# Skilldex: A Package Manager and Registry for Agent Skill Packages with Hierarchical Scope-Based Distribution 

**Authors**: Sampriti Saha, Pranav Hemanth  

**Link**: [PDF](https://arxiv.org/pdf/2604.16911)  

**Abstract**: Large Language Model (LLM) agents are increasingly extended at runtime via skill packages, structured natural-language instruction bundles loaded from a well-known directory. Community install tooling and registries exist, but two gaps persist: no public tool scores skill packages against Anthropic's published format specification, and no mechanism bundles related skills with the shared context they need to remain mutually coherent. We present Skilldex, a package manager and registry for agent skill packages addressing both gaps. The two novel contributions are: (1) compiler-style format conformance scoring against Anthropic's skill specification, producing line-level diagnostics on description specificity, frontmatter validity, and structural adherence; and (2) the skillset abstraction, a bundled collection of related skills with shared assets (vocabulary files, templates, reference documents) that enforce cross-skill behavioral coherence. Skilldex also provides supporting infrastructure: a three-tier hierarchical scope system, a human-in-the-loop agent suggestion loop, a metadata-only community registry, and a Model Context Protocol (MCP) server. The system is implemented as a TypeScript CLI (skillpm / spm) with a Hono/Supabase registry backend, and is open-source. 

---
# Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models 

**Authors**: Xinru Yan, Boxi Cao, Yaojie Lu, Hongyu Lin, Weixiang Zhou, Le Sun, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.16902)  

**Abstract**: Native Omni-modal Large Language Models (OLLMs) have shifted from pipeline architectures to unified representation spaces. However, this native integration gives rise to a critical yet underexplored phenomenon: modality preference. To bridge this gap, we first systematically quantify modality preference of OLLMs using a newly-curated conflict-based benchmark and the modality selection rate metric. Our evaluation of ten representative OLLMs reveals a notable paradigm shift: unlike the ``text-dominance'' of traditional VLMs, most OLLMs exhibit a pronounced visual preference. To further understand the underlying mechanism, we conduct layer-wise probing and demonstrate that such modality preference is not static but emerges progressively in the mid-to-late layers. Building upon these insights, we leverage these internal signals to diagnose cross-modal hallucinations, achieving competitive performance across three downstream multi-modal benchmarks without task-specific data. Our work provides both a mechanistic understanding and a practical tool for building more trustworthy OLLMs. Our code and related resources are publicly available at: this https URL 

---
# Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning 

**Authors**: Benteng Chen, Weida Wang, Shufei Zhang, Mingbao Lin, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16890)  

**Abstract**: Large reasoning models that use long chain-of-thought excel at problem-solving yet waste compute on redundant checks. Curbing this overthinking is hard: training-time length penalties can cripple ability, while inference-time early-exit adds system overhead. To bridge this gap, we propose Step-GRPO, a novel post-training framework that internalizes dynamic early-exit capabilities directly into the model. Step-GRPO shifts the optimization objective from raw tokens to semantic steps by utilizing linguistic markers to structure reasoning. We introduce a Dynamic Truncated Rollout mechanism that exposes the model to concise high-confidence trajectories during exploration, synergized with a Step-Aware Relative Reward that dynamically penalizes redundancy based on group-level baselines. Extensive experiments across three model sizes on diverse benchmarks demonstrate that Step-GRPO achieves a superior accuracy-efficiency trade-off. On Qwen3-8B, our method reduces token consumption by 32.0\% compared to the vanilla model while avoiding the accuracy degradation observed in traditional length-penalty methods. 

---
# GRAIL: Autonomous Concept Grounding for Neuro-Symbolic Reinforcement Learning 

**Authors**: Hikaru Shindo, Henri Rößler, Quentin Delfosse, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2604.16871)  

**Abstract**: Neuro-symbolic Reinforcement Learning (NeSy-RL) combines symbolic reasoning with gradient-based optimization to achieve interpretable and generalizable policies. Relational concepts, such as "left of" or "close by", serve as foundational building blocks that structure how agents perceive and act. However, conventional approaches require human experts to manually define these concepts, limiting adaptability since concept semantics vary across environments. We propose GRAIL (Grounding Relational Agents through Interactive Learning), a framework that autonomously grounds relational concepts through environmental interaction. GRAIL leverages large language models (LLMs) to provide generic concept representations as weak supervision, then refines them to capture environment-specific semantics. This approach addresses both sparse reward signals and concept misalignment prevalent in underdetermined environments. Experiments on the Atari games Kangaroo, Seaquest, and Skiing demonstrate that GRAIL matches or outperforms agents with manually crafted concepts in simplified settings, and reveals informative trade-offs between reward maximization and high-level goal completion in the full environment. 

---
# GAMMA-Net: Adaptive Long-Horizon Traffic Spatio-Temporal Forecasting Model based on Interleaved Graph Attention and Multi-Axis Mamba 

**Authors**: Dongyi He, Yuanquan Gao, Bin Jiang, He Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16859)  

**Abstract**: Accurate traffic forecasting is crucial for intelligent transportation systems, supporting effective traffic management, congestion reduction, and informed urban planning. However, traditional models often fail to adequately capture the intricate spatio-temporal dependencies present in traffic data. To overcome these limitations, we introduce GAMMA-Net, a novel approach that integrates Graph Attention Networks (GAT) with multi-axis Selective State Space Models (Mamba). The GAT component uses a self-attention mechanism to dynamically adjust the influence of nodes within the traffic network, enabling adaptive spatial dependency modeling based on real-time conditions. Simultaneously, the Mamba module efficiently models long-term temporal and spatial dynamics without the heavy computational cost of conventional recurrent architectures. Extensive experiments on several benchmark traffic datasets, including METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, and PEMS08, show that GAMMA-Net consistently outperforms existing state-of-the-art models across different prediction horizons, achieving up to a 16.25% reduction in Mean Absolute Error (MAE) compared to baseline models. Ablation studies highlight the critical contributions of both the spatial and temporal components, emphasizing their complementary role in improving prediction accuracy. In conclusion, the GAMMA-Net model sets a new standard in traffic forecasting, offering a powerful tool for next-generation traffic management and urban planning. The code for this study is available at this https URL 

---
# The CTLNet for Shanghai Composite Index Prediction 

**Authors**: Haibin Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16835)  

**Abstract**: Shanghai Composite Index prediction has become a hot issue for many investors and academic researchers. Deep learning models are widely applied in multivariate time series forecasting, including recurrent neural networks (RNN), convolutional neural networks (CNN), and transformers. Specifically, the Transformer encoder, with its unique attention mechanism and parallel processing capabilities, has become an important tool in time series prediction, and has an advantage in dealing with long sequence dependencies and multivariate data correlations. Drawing on the strengths of various models, we propose the CNN-Transformer-LSTM Networks (CTLNet). This paper explores the application of CTLNet for Shanghai Composite Index prediction and the comparative experiments show that the proposed model outperforms state-of-the-art baselines. 

---
# PersonalHomeBench: Evaluating Agents in Personalized Smart Homes 

**Authors**: Nikhil Verma, InJung Yang, Sungil Kim, KoKeun Kim, YoungJoon Kim, Manasa Bharadwaj, Yolanda Liu, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.16813)  

**Abstract**: Agentic AI systems are rapidly advancing toward real-world applications, yet their readiness in complex and personalized environments remains insufficiently characterized. To address this gap, we introduce PersonalHomeBench, a benchmark for evaluating foundation models as agentic assistants in personalized smart home environments. The benchmark is constructed through an iterative process that progressively builds rich household states, which are then used to generate personalized, context-dependent tasks. To support realistic agent-environment interaction, we provide PersonalHomeTools, a comprehensive toolbox enabling household information retrieval, appliance control, and situational understanding. PersonalHomeBench evaluates both reactive and proactive agentic abilities under unimodal and multimodal observations. Thorough experimentation reveals a systematic performance reduction as task complexity increases, with pronounced failures in counterfactual reasoning and under partial observability, where effective tool-based information gathering is required. These results position PersonalHomeBench as a rigorous evaluation platform for analyzing the robustness and limitations of personalized agentic reasoning and planning. 

---
# Introspection Adapters: Training LLMs to Report Their Learned Behaviors 

**Authors**: Keshav Shenoy, Li Yang, Abhay Sheshadri, Sören Mindermann, Jack Lindsey, Sam Marks, Rowan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16812)  

**Abstract**: When model developers or users fine-tune an LLM, this can induce behaviors that are unexpected, deliberately harmful, or hard to detect. It would be far easier to audit LLMs if they could simply describe their behaviors in natural language. Here, we study a scalable approach to rapidly identify learned behaviors of many LLMs derived from a shared base LLM. Given a model $M$, our method works by finetuning models $M_i$ from $M$ with implanted behaviors $b_i$; the $(M_i, b_i)$ pairs serve as labeled training data. We then train an \emph{introspection adapter} (IA): a single LoRA adapter jointly trained across the finetunes $M_i$ to cause them to verbalize their implanted behaviors. We find that this IA induces self-description of learned behaviors even in finetunes of $M$ that were trained in very different ways from the $M_i$. For example, IAs generalize to AuditBench, achieving state-of-the-art at identifying explicitly hidden concerning behaviors. IAs can also be used to detect encrypted finetuning API attacks. They scale favorably with model size and training data diversity. Overall, our results suggest that IAs are a scalable, effective, and practically useful approach to auditing fine-tuned LLMs. 

---
# SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention 

**Authors**: Jiahao Li, Jiayi Dong, Peng Ye, Xiaochi Zhou, Haohai Lu, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16776)  

**Abstract**: Modeling single-cell gene expression across diverse biological and technical conditions is crucial for characterizing cellular states and simulating unseen scenarios. Existing methods often treat genes as independent tokens, overlooking their high-level biological relationships and leading to poor performance. We introduce SAVE, a unified generative framework based on conditional Transformers for multi-condition single-cell modeling. SAVE leverages a coarse-grained representation by grouping semantically related genes into blocks, capturing higher-order dependencies among gene modules. A Flow Matching mechanism and condition-masking strategy further enhance flexible simulation and enable generalization to unseen condition combinations. We evaluate SAVE on a range of benchmarks, including conditional generation, batch effect correction, and perturbation prediction. SAVE consistently outperforms state-of-the-art methods in generation fidelity and extrapolative generalization, especially in low-resource or combinatorially held-out settings. Overall, SAVE offers a scalable and generalizable solution for modeling complex single-cell data, with broad utility in virtual cell synthesis and biological interpretation. Our code is publicly available at this https URL 

---
# Machine individuality: Separating genuine idiosyncrasy from response bias in large language models 

**Authors**: Valentin Kriegmair, Dirk U. Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2604.16755)  

**Abstract**: As large language models (LLMs) are increasingly integrated into daily life, in roles ranging from high-stakes decision support to companionship, understanding their behavioral dispositions becomes critical. A growing literature uses psychometric inventories and cognitive paradigms to profile LLM dispositions. However, these approaches cannot determine whether behavioral differences reflect stable, stimulus-specific individuality or global response biases and stochastic noise. Here, we apply crossed random-effects models -- widely used in psychometrics to separate systematic effects -- to 74.9 million ratings provided by 10 open-weight LLMs for over 100,000 words across 14 psycholinguistic norms. On average, 16.9% of variance is attributable to stimulus-specific individuality, robustly exceeding a statistical null model. Cross-norm prediction analyses reveal this individuality as a coherent fingerprint, unique to each model. These results identify individual differences among LLMs that cannot be attributed to response biases or stochastic noise. We term these differences machine individuality. 

---
# Know When to Trust the Skill: Delayed Appraisal and Epistemic Vigilance for Single-Agent LLMs 

**Authors**: Eren Unlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16753)  

**Abstract**: As large language models (LLMs) transition into autonomous agents integrated with extensive tool ecosystems, traditional routing heuristics increasingly succumb to context pollution and "overthinking". We argue that the bottleneck is not a deficit in algorithmic capability or skill diversity, but the absence of disciplined second-order metacognitive governance. In this paper, our scientific contribution focuses on the computational translation of human cognitive control - specifically, delayed appraisal, epistemic vigilance, and region-of-proximal offloading - into a single-agent architecture. We introduce MESA-S (Metacognitive Skills for Agents, Single-agent), a preliminary framework that shifts scalar confidence estimation into a vector separating self-confidence (parametric certainty) from source-confidence (trust in retrieved external procedures). By formalizing a delayed procedural probe mechanism and introducing Metacognitive Skill Cards, MESA-S decouples the awareness of a skill's utility from its token-intensive execution. Evaluated under an In-Context Static Benchmark Evaluation natively executed via Gemini 3.1 Pro, our early results suggest that explicitly programming trust provenance and delayed escalation mitigates supply-chain vulnerabilities, prunes unnecessary reasoning loops, and prevents offloading-induced confidence inflation. This architecture offers a scientifically cautious, behaviorally anchored step toward reliable, epistemically vigilant single-agent orchestration. 

---
# Don't Start What You Can't Finish: A Counterfactual Audit of Support-State Triage in LLM Agents 

**Authors**: Eren Unlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16752)  

**Abstract**: Current agent evaluations largely reward execution on fully specified tasks, while recent work studies clarification [11, 22, 2], capability awareness [9, 1], abstention [8, 14], and search termination [20, 5] mostly in isolation. This leaves open whether agents can diagnose why a task is blocked before acting. We introduce the Support-State Triage Audit (SSTA-32), a matched-item diagnostic framework in which minimal counterfactual edits flip the same base request across four support states: Complete (ANSWER), Clarifiable (CLARIFY), Support-Blocked (REQUEST SUPPORT), and Unsupported-Now (ABSTAIN). We evaluate a frontier model under four prompting conditions - Direct, Action-Only, Confidence-Only, and a typed Preflight Support Check (PSC) - using Dual-Persona Auto-Auditing (DPAA) with deterministic heuristic scoring. Default execution overcommits heavily on non-complete tasks (41.7% overcommitment rate). Scalar confidence mapping avoids overcommitment but collapses the three-way deferral space (58.3% typed deferral accuracy). Conversely, both Action-Only and PSC achieve 91.7% typed deferral accuracy by surfacing the categorical ontology in the prompt. Targeted ablations confirm that removing the support-sufficiency dimension selectively degrades REQUEST SUPPORT accuracy, while removing the evidence-sufficiency dimension triggers systematic overcommitment on unsupported items. Because DPAA operates within a single context window, these results represent upper-bound capability estimates; nonetheless, the structural findings indicate that frontier models possess strong latent triage capabilities that require explicit categorical decision paths to activate safely. 

---
# Why Training-Free Token Reduction Collapses: The Inherent Instability of Pairwise Scoring Signals 

**Authors**: Yang Shanglin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16745)  

**Abstract**: Training-free token reduction methods for Vision Transformers (ToMe, ToFu, PiToMe, and MCTF) employ different scoring mechanisms, yet they share a closely matched cliff-like collapse at high compression. This paper explains \emph{why}. We develop a diagnostic framework with two tools, ranking consistency $\rho_s$ and off-diagonal correlation $\rho_\text{off}$, that decomposes the collapse into (1)a signal-agnostic error amplifier inherent to layer-wise reduction, predicting convex Pareto curves and $r_{\text{crit}} \propto 1/L$; and (2)shared reliance on \emph{pairwise} similarity signals whose ranking consistency degrades from $\rho_s{=}0.88$ to $0.27$ in deep layers. Pairwise rankings are inherently unstable ($O(N_p^2)$ joint perturbations) while unary signals enjoy greater stability ($O(N_p)$ perturbations, CLT). From three design principles derived from this diagnosis, we construct CATIS as a constructive validation: unary signals raise the trigger threshold, triage suppresses the gain. On ViT-Large at 63% FLOPs reduction, CATIS retains 96.9% of vanilla accuracy (81.0%) on ImageNet-1K where all baselines collapse to 43--65%. 

---
# CT Open: An Open-Access, Uncontaminated, Live Platform for the Open Challenge of Clinical Trial Outcome Prediction 

**Authors**: Jianyou Wang, Youze Zheng, Longtian Bao, Hanyuan Zhang, Qirui Zheng, Yuhan Chen, Yang Zhang, Matthew Feng, Maxim Khan, Aditya K. Sehgal, Christopher D. Rosin, Ramamohan Paturi, Umber Dube, Leon Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16742)  

**Abstract**: Scientists have long sought to accurately predict outcomes of real-world events before they happen. Can AI systems do so more reliably? We study this question through clinical trial outcome prediction, a high-stakes open challenge even for domain experts. We introduce CT Open, an open-access, live platform that will run four challenge every year. Anyone can submit predictions for each challenge. CT Open evaluates those submissions on trials whose outcomes were not yet public at the time of submission but were made public afterwards. Determining if a trial's outcome is public on the internet before a certain date is surprisingly difficult. Outcomes posted on official registries may lag behind by years, while the first mention may appear in obscure articles. To address this, we propose a novel, fully automated decontamination pipeline that uses iterative LLM-powered web search to identify the earliest mention of trial outcomes. We validate the pipeline's quality and accuracy by human expert's annotations. Since CT Open's pipeline ensures that every evaluated trial had no publicly reported outcome when the prediction was made, it allows participants to use any methodology and any data source. In this paper, we release a training set and two time-stamped test benchmarks, Winter 2025 and Summer 2025. We believe CT Open can serve as a central hub for advancing AI research on forecasting real-world outcomes before they occur, while also informing biomedical research and improving clinical trial design. CT Open Platform is hosted at $\href{this https URL}{this https URL}$ 

---
# When Agents Go Quiet: Output Generation Capacity and Format-Cost Separation for LLM Document Synthesis 

**Authors**: Justice Owusu Agyemang, Michael Agyare, Miriam Kobbinah, Nathaniel Agbugblah, Prosper Addo  

**Link**: [PDF](https://arxiv.org/pdf/2604.16736)  

**Abstract**: LLM-powered coding agents suffer from a poorly understood failure mode we term output stalling: the agent silently produces empty responses when attempting to generate large, format-heavy documents. We present a theoretical framework that explains and prevents this failure through three contributions. (1) We introduce Output Generation Capacity (OGC), a formal measure of an agent's effective ability to produce output given its current context state - distinct from and empirically smaller than the raw context window. (2) We prove a Format-Cost Separation Theorem showing that deferred template rendering is always at least as token-efficient as direct generation for any format with overhead multiplier $\mu_f > 1$, and derive tight bounds on the savings. (3) We formalize Adaptive Strategy Selection, a decision framework that maps the ratio of estimated output cost to available OGC into an optimal generation strategy (direct, chunked, or deferred). We validate the theory through controlled experiments across three models (Claude 3.5 Sonnet, GPT-4o, Llama 3.1 70B), four document types, and an ablation study isolating each component's contribution. Deferred rendering reduces LLM generation tokens by 48-72% across all conditions and eliminates output stalling entirely. We instantiate the framework as GEN-PILOT, an open-source MCP server, demonstrating that the theory translates directly into a practical tool. 

---
# Debate as Reward: A Multi-Agent Reward System for Scientific Ideation via RL Post-Training 

**Authors**: Moein Salimi, Babak Hosseini Mohtasham, Amin Aghakasiri, Mahdi Naieni, Amir Hossein Qeysarbeigi, Mohammad Masih Shalchian Nazer, Zahra Azar, Mahdi Jafari Siavoshani, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2604.16723)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential in automating scientific ideation, yet current approaches relying on iterative prompting or complex multi-agent architectures often suffer from hallucination or computational inefficiency. A critical bottleneck in applying Reinforcement Learning (RL) to this open-ended domain is reward hacking -- where models exploit imperfect evaluation proxies to maximize scores without producing genuine scientific innovation. To address these limitations, we propose an RL framework explicitly tailored for high-quality scientific idea generation. We propose the first multi-agent reward function designed to serve as a judge, decoupling methodological validation from implementation details while providing strict binary rewards that are robust to reward hacking. To effectively optimize against this sparse signal, we utilize an unbiased variant of Group Relative Policy Optimization to mitigate artificial length bias. We grounded our training in ICLR-320, a curated dataset of problem-solution pairs extracted from ICLR 2024 proceedings. Experiments demonstrate that our framework significantly outperforms state-of-the-art baselines across expert-evaluated metrics of novelty, feasibility, and effectiveness. 

---
# Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench 

**Authors**: Bhaskar Gurram  

**Link**: [PDF](https://arxiv.org/pdf/2604.16706)  

**Abstract**: Automated evaluation of tool-using large language model (LLM) agents is widely assumed to be reliable, but this assumption has rarely been validated against human annotation. We introduce AgentProp-Bench, a 2,000-task benchmark with 2,300 traces across four domains, nine production LLMs, and a 100-label human-validated subset. We quantify judge reliability, characterize error propagation, and evaluate a runtime mitigation. Substring-based judging agrees with human annotation at kappa=0.049 (chance-level); a three-LLM ensemble reaches kappa=0.432 (moderate) with a conservative bias. Under validated evaluation, a parameter-level injection propagates to a wrong final answer with human-calibrated probability approximately 0.62 (range 0.46-0.73 across models). Rejection (catching bad parameters) and recovery (correcting after acceptance) are independent model capabilities (Spearman rho=0.126, p=0.747). A tuned runtime interceptor reduces hallucination on GPT-4o-mini by 23.0 percentage points under a concurrent n=600 control, but shows no significant effect on Gemini-2.0-Flash, whose aggressive parameter rejection eliminates the target failure mode. All code, data, traces, and human labels are released at this https URL. 

---
# RankGuide: Tensor-Rank-Guided Routing and Steering for Efficient Reasoning 

**Authors**: Jiayi Tian, Yupeng Su, Ryan Solgi, Souvik Kundu, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16694)  

**Abstract**: Large reasoning models (LRMs) enhance problem-solving capabilities by generating explicit multi-step chains of thought (CoT) reasoning; however, they incur substantial inference latency and computational overhead. To mitigate this issue, recent works have explored model collaboration paradigms, where small reasoning models (SRMs) generate intermediate reasoning steps to achieve a better accuracy--latency trade-off. Despite recent progress, effectively and efficiently detecting and mitigating SRM failures in collaborative systems remains a key challenge. To address this issue, we analyze SRM inference in both the generated text and hidden-state spaces, and identify three types of failure modes: \textit{overconfidence}, \textit{uncertainty}, and \textit{heavy revalidation}. Building on these insights, we propose \textbf{RankGuide}, a framework that improves the efficiency and effectiveness of SRM--LRM collaboration through tensor-rank-guided routing and steering. Specifically, RankGuide leverages a routing signal that incorporates tensor-rank signals derived from consecutive hidden states to detect when SRMs are likely to fail and selectively invoke LRMs. In addition, we introduce a tensor-rank-filtered steering vector extraction method to modulate the reasoning trajectory of SRMs, thereby improving their generation quality. By improving both routing and steering through tensor-rank signals, RankGuide enables SRM--LRM collaborative systems to achieve more efficient reasoning with fewer steps and improved accuracy. Experiments on multiple reasoning benchmarks demonstrate the efficacy of RankGuide in reducing latency by up to $1.75\times$ compared to LRM, while maintaining competitive accuracy relative to prior methods. 

---
# The Query Channel: Information-Theoretic Limits of Masking-Based Explanations 

**Authors**: Erciyes Karakaya, Ozgur Ercetin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16689)  

**Abstract**: Masking-based post-hoc explanation methods, such as KernelSHAP and LIME, estimate local feature importance by querying a black-box model under randomized perturbations. This paper formulates this procedure as communication over a query channel, where the latent explanation acts as a message and each masked evaluation is a channel use. Within this framework, the complexity of the explanation is captured by the entropy of the hypothesis class, while the query interface supplies information at a rate determined by an identification capacity per query. We derive a strong converse showing that, if the explanation rate exceeds this capacity, the probability of exact recovery necessarily converges to one in error for any sequence of explainers and decoders. We also prove an achievability result establishing that a sparse maximum-likelihood decoder attains reliable recovery when the rate lies below capacity. A Monte Carlo estimator of mutual information yields a non-asymptotic query benchmark that we use to compare optimal decoding with Lasso- and OLS-based procedures that mirror LIME and KernelSHAP. Experiments reveal a range of query budgets where information theory permits reliable explanations but standard convex surrogates still fail. Finally, we interpret super-pixel resolution and tokenization for neural language models as a source-coding choice that sets the entropy of the explanation and show how Gaussian noise and nonlinear curvature degrade the query channel, induce waterfall and error-floor behavior, and render high-resolution explanations unattainable. 

---
# Agentic Risk-Aware Set-Based Engineering Design 

**Authors**: Varun Kumar, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16687)  

**Abstract**: This paper introduces a multi-agent framework guided by Large Language Models (LLMs) to assist in the early stages of engineering design, a phase often characterized by vast parameter spaces and inherent uncertainty. Operating under a human-in-the-loop paradigm and demonstrated on the canonical problem of aerodynamic airfoil design, the framework employs a team of specialized agents: a Coding Assistant, a Design Agent, a Systems Engineering Agent, and an Analyst Agent - all coordinated by a human Manager. Integrated within a set-based design philosophy, the process begins with a collaborative phase where the Manager and Coding Assistant develop a suite of validated tools, after which the agents execute a structured workflow to systematically explore and prune a large set of initial design candidates. A key contribution of this work is the explicit integration of formal risk management, employing the Conditional Value-at-Risk (CVaR) as a quantitative metric to filter designs that exhibit a high probability of failing to meet performance requirements, specifically the target coefficient of lift. The framework automates labor-intensive initial exploration through a global sensitivity analysis conducted by the Analyst agent, which generates actionable heuristics to guide the other agents. The process culminates by presenting the human Manager with a curated final set of promising design candidates, augmented with high-fidelity Computational Fluid Dynamics (CFD) simulations. This approach effectively leverages AI to handle high-volume analytical tasks, thereby enhancing the decision-making capability of the human expert in selecting the final, risk-assessed design. 

---
# From Subsumption to Satisfiability: LLM-Assisted Active Learning for OWL Ontologies 

**Authors**: Haoruo Zhao, Wenshuo Tang, Duncan Guthrie, Michele Sevegnani, David Flynn, Paul Harvey  

**Link**: [PDF](https://arxiv.org/pdf/2604.16672)  

**Abstract**: In active learning, membership queries (MQs) allow a learner to pose questions to a teacher, such as ''Is every apple a fruit?'', to which the teacher responds correctly with yes or no. These MQs can be viewed as subsumption tests with respect to the target ontology. Inspired by the standard reduction of subsumption to satisfiability in description logics, we reformulate each candidate axiom into its corresponding counter-concept and verbalise it in controlled natural language before presenting it to Large Language Models (LLMs). We introduce LLMs as a third component that provides real-world examples approximating an instance of the counter-concept. This design property ensures that only Type II errors may occur in ontology modelling; in the worst case, these errors merely delay the construction process without introducing inconsistencies. Experimental results on 13 commercial LLMs show that recall, corresponding to Type II errors in our framework, remains stable across several well-established ontologies. 

---
# Agentic Frameworks for Reasoning Tasks: An Empirical Study 

**Authors**: Zeeshan Rasheed, Abdul Malik Sami, Muhammad Waseem, Kai-Kristian Kemell, Mika Saari, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.16646)  

**Abstract**: Recent advances in agentic frameworks have enabled AI agents to perform complex reasoning and decision-making. However, evidence comparing their reasoning performance, efficiency, and practical suitability remains limited. To address this gap, we empirically evaluate 22 widely used agentic frameworks across three reasoning benchmarks: BBH, GSM8K, and ARC. The frameworks were selected from 1,200 GitHub repositories collected between January 2023 and July 2025 and organized into a taxonomy based on architectural design. We evaluated them under a unified setting, measuring reasoning accuracy, execution time, computational cost, and cross-benchmark consistency.
Our results show that 19 of the 22 frameworks completed all three benchmarks. Among these, 12 showed stable performance, with mean accuracy of 74.6-75.9%, execution time of 4-6 seconds per task, and cost of 0.14-0.18 cents per task. Poorer results were mainly caused by orchestration problems rather than reasoning limits. For example, Camel failed to complete BBH after 11 days because of uncontrolled context growth, while Upsonic consumed USD 1,434 in one day because repeated extraction failures triggered costly retries. AutoGen and Mastra also exhausted API quotas through iterative interactions that increased prompt length without improving results.
We also found a sharp drop in mathematical reasoning. Mean accuracy on GSM8K was 44.35%, compared with 89.80% on BBH and 89.56% on ARC. Overall, this study provides the first large-scale empirical comparison of agentic frameworks for reasoning-intensive software engineering tasks and shows that framework selection should prioritize orchestration quality, especially memory control, failure handling, and cost management. 

---
# Healthcare AI for Automation or Allocation? A Transaction Cost Economics Framework 

**Authors**: Ari Ercole  

**Link**: [PDF](https://arxiv.org/pdf/2604.16465)  

**Abstract**: Healthcare productivity is shaped not only by clinical complexity but by the costs of coordinating work under uncertainty. Transaction-cost economics offers a theory of these coordination frictions, yet has rarely been operationalised at task level across health occupations. Using task statements and frequency weights from the O*NET occupational database, we characterised healthcare work at task granularity and coded each unique task using a constrained large language model into one dominant transaction-cost category (information search, decision and bargaining, monitoring and enforcement, or adaptation and coordination) together with an overall transaction-cost intensity score. Aggregating to the occupation level, clinician roles exhibited substantially higher transaction-cost intensity than non-clinician roles, driven primarily by greater burdens of information search and decision-related coordination, while dispersion of transaction costs within occupations did not differ. These findings demonstrate systematic heterogeneity in the nature of coordination work across healthcare roles and suggest that the opportunities for digital and AI interventions are unevenly distributed, shaped less by technical task complexity than by underlying coordination structure. 

---
# Support Sufficiency as Consequence-Sensitive Compression in Belief Arbitration 

**Authors**: Mark Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2604.16434)  

**Abstract**: When a system commits to a hypothesis, much of the evidential structure behind that commitment is lost to compression. Standard accounts assume that selected content and scalar confidence suffice for downstream control. This paper argues that they do not, and that determining what must survive compression is itself a consequence-sensitive problem. We develop a recurrent arbitration architecture in which active constraint fields jointly determine a hypothesis geometry over candidates. Rather than carrying that geometry forward in full, the system compresses it into a support-aware control state whose resolution is regulated by current consequence geometry, arbitration memory, and resource constraints.
A bounded objective formalizes the tradeoff. Too little retained support collapses policy-relevant distinctions, producing controllers that select content adequately while misrouting verification, abstention, and recovery. Too much retained support fragments learning across overly fine contexts, degrading adaptation even as discrimination improves. These failure modes yield ordered controller predictions confirmed by a minimal repeated-interaction simulation. Adaptive controllers that regulate support resolution outperform all fixed-resolution controllers in cumulative utility. Agile adaptive control outperforms sluggish adaptive control. Fixed high-resolution control achieves the best commitment accuracy but still trails adaptive controllers because resource cost and learning fragmentation offset the gains from richer retention.
Support sufficiency should be understood not as a static representational threshold, but as a dynamic compression criterion. Robust arbitration depends on preserving the smallest support structure adequate for policy under the current consequence landscape, and on regulating that structure as conditions change across repeated cycles of inference and action. 

---
# Heterogeneous Self-Play for Realistic Highway Traffic Simulation 

**Authors**: Jinkai Qiu, Alessandro Saviolo, Chaojie Wang, Mingke Wang, Xiaoyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16406)  

**Abstract**: Realistic highway simulation is critical for scalable safety evaluation of autonomous vehicles, particularly for interactions that are too rare to study from logged data alone. Yet highway traffic generation remains challenging because it requires broad coverage across speeds and maneuvers, controllable generation of rare safety-critical scenarios, and behavioral credibility in multi-agent interactions. We present PHASE, Policy for Heterogeneous Agent Self-play on Expressway, a context-aware self-play framework that addresses these three requirements through explicit per-agent conditioning for controllability, synthetic scenario generation for broad highway coverage, and closed-loop multi-agent training for realistic interaction dynamics. PHASE further supports different vehicle profiles, for example, passenger cars and articulated trailer trucks, within a single policy via vehicle-aware dynamics and context-conditioned actions, and stabilizes self-play with early termination of unrecoverable states, at-fault collision attribution, highway-aware reward shaping, coupled curricula, and robust policy optimization. Despite being trained only on synthetic data, PHASE transfers zero-shot to 512 unseen high-interaction real scenarios in exiD, achieving a 96.3% success rate and reducing ADE/FDE from 6.57/12.07 m to 2.44/5.25 m relative to a prior self-play baseline. In a learned trajectory embedding space, it also improves behavioral realism over IDM, reducing Frechet trajectory distance by 13.1% and energy distance by 20.2%. These results show that synthetic self-play can provide a scalable route to controllable and realistic highway scenario generation without direct imitation of expert logs. 

---
# Computational Hermeneutics: Evaluating generative AI as a cultural technology 

**Authors**: Cody Kommers, Ruth Ahnert, Maria Antoniak, Emmanouil Benetos, Steve Benford, Mercedes Bunz, Baptiste Caramiaux, Shauna Concannon, Martin Disley, James Dobson, Yali Du, Edgar Duéñez-Guzmán, Kerry Francksen, Evelyn Gius, Jonathan W. Y. Gray, Ryan Heuser, Sarah Immel, Richard Jean So, Sang Leigh, Dalaki Livingston, Hoyt Long, Meredith Martin, Georgia Meyer, Daniela Mihai, Ashley Noel-Hirst, Kirsten Ostherr, Deven Parker, Yipeng Qin, Jessica Ratcliff, Emily Robinson, Karina Rodriguez, Adam Sobey, Ted Underwood, Aditya Vashistha, Matthew Wilkens, Youyou Wu, Yuan Zheng, Drew Hemment  

**Link**: [PDF](https://arxiv.org/pdf/2604.16403)  

**Abstract**: Generative AI systems are increasingly recognized as cultural technologies, yet current evaluation frameworks often treat culture as a variable to be measured rather than fundamental to the system's operation. Drawing on hermeneutic theory from the humanities, we argue that GenAI systems function as "context machines" that must inherently address three interpretive challenges: situatedness (meaning only emerges in context), plurality (multiple valid interpretations coexist), and ambiguity (interpretations naturally conflict). We present computational hermeneutics as an emerging framework offering an interpretive account of what GenAI systems do, and how they might do it better. We offer three principles for hermeneutic evaluation -- that benchmarks should be iterative, not one-off; include people, not just machines; and measure cultural context, not just model output. This perspective offers a nascent paradigm for designing and evaluating contemporary AI systems: shifting from standardized questions about accuracy to contextual ones about meaning. 

---
# Semantic Consensus: Process-Aware Conflict Detection and Resolution for Enterprise Multi-Agent LLM Systems 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2604.16339)  

**Abstract**: Multi-agent large language model (LLM) systems are rapidly emerging as the dominant architecture for enterprise AI automation, yet production deployments exhibit failure rates between 41% and 86.7%, with nearly 79% of failures originating from specification and coordination issues rather than model capability limitations. This paper identifies Semantic Intent Divergence--the phenomenon whereby cooperating LLM agents develop inconsistent interpretations of shared objectives due to siloed context and absent process models--as a primary yet formally unaddressed root cause of multi-agent failure in enterprise settings. We propose the Semantic Consensus Framework (SCF), a process-aware middleware comprising six components: a Process Context Layer for shared operational semantics, a Semantic Intent Graph for formal intent representation, a Conflict Detection Engine for real-time identification of contradictory, contention-based, and causally invalid intent combinations, a Consensus Resolution Protocol using a policy--authority--temporal hierarchy, a Drift Monitor for detecting gradual semantic divergence, and a Process-Aware Governance Integration layer for organizational policy enforcement. Evaluation across 600 runs spanning three multi-agent frameworks (AutoGen, CrewAI, LangGraph) and four enterprise scenarios demonstrates that SCF is the only approach to achieve 100% workflow completion--compared to 25.1% for the next-best baseline--while detecting 65.2% of semantic conflicts with 27.9% precision and providing complete governance audit trails. The framework is protocol-agnostic and compatible with MCP and A2A communication standards. 

---
# Governing the Agentic Enterprise: A Governance Maturity Model for Managing AI Agent Sprawl in Business Operations 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2604.16338)  

**Abstract**: The rapid adoption of agentic AI in enterprise business operations--autonomous systems capable of planning, reasoning, and executing multi-step workflows--has created an urgent governance crisis. Organizations face uncontrolled agent sprawl: the proliferation of redundant, ungoverned, and conflicting AI agents across business functions. Industry surveys report that only 21% of enterprises have mature governance models for autonomous agents, while 40% of agentic AI projects are projected to fail by 2027 due to inadequate governance and risk controls. Despite growing acknowledgment of this challenge, academic literature lacks a formal, empirically validated governance maturity model connecting governance capability to measurable business outcomes. This paper introduces the Agentic AI Governance Maturity Model (AAGMM), a five-level framework spanning 12 governance domains, grounded in NIST AI RMF and ISO/IEC 42001 standards. We additionally propose a novel taxonomy of agent sprawl patterns--functional duplication, shadow agents, orphaned agents, permission creep, and unmonitored delegation chains--each linked to quantifiable business cost models. The framework is validated through 750 simulation runs across five enterprise scenarios and five governance maturity levels, measuring business outcomes including cost containment, risk incident rates, operational efficiency, and decision quality. Results demonstrate statistically significant differences (p < 0.001, large effect sizes d > 2.0) between all governance maturity levels, with Level 4-5 organizations achieving 94.3% lower sprawl indices, 96.4% fewer risk incidents, and 32.6% higher effective task completion rates compared to Level 1. The AAGMM provides practitioners with an actionable roadmap for governing autonomous AI agents while maximizing business returns. 

---
# Sessa: Selective State Space Attention 

**Authors**: Liubomyr Horbatko  

**Link**: [PDF](https://arxiv.org/pdf/2604.18580)  

**Abstract**: Modern sequence models are dominated by Transformers, where self-attention mixes information from the visible context in an input-dependent way. However, when retrieval is not sharp and attention remains diffuse over an effective support $S_{\mathrm{eff}}(t)$, the influence of any individual token is diluted, typically scaling as $O(1/S_{\mathrm{eff}}(t))$ and reaching $O(1/\ell)$ for old tokens in full-prefix settings. Structured state-space models process sequences recurrently through an explicit feedback path; selective variants such as Mamba make this feedback input-dependent, yet when freeze time cannot be sustained over long intervals, their long-range sensitivity decays exponentially with lag. Existing architectures therefore either retrieve from the past in a single read or propagate information through a single feedback chain. We introduce Sessa, a decoder that places attention inside a feedback path, enabling recurrent many-path aggregation within a layer. Under stated assumptions, Sessa admits regimes with a power-law memory tail in lag $\ell$ of order $O(\ell^{-\beta})$ for $0<\beta<1$, which is asymptotically slower than $1/\ell$; moreover, this rate is tight in an explicit diffuse uniform-routing setting where the influence is $\Theta(\ell^{-\beta})$. Under the same conditions, only Sessa among the compared model classes realizes flexible selective retrieval, including non-decaying profiles. Empirically, under matched architectures and training budgets, Sessa achieves the strongest performance on our long-context benchmarks while remaining competitive with Transformer and Mamba style baselines on short-context language modeling. 

---
# Bounded Ratio Reinforcement Learning 

**Authors**: Yunke Ao, Le Chen, Bruce D. Lee, Assefa S. Wahd, Aline Czarnobai, Philipp Fürnstahl, Bernhard Schölkopf, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2604.18578)  

**Abstract**: Proximal Policy Optimization (PPO) has become the predominant algorithm for on-policy reinforcement learning due to its scalability and empirical robustness across domains. However, there is a significant disconnect between the underlying foundations of trust region methods and the heuristic clipped objective used in PPO. In this paper, we bridge this gap by introducing the Bounded Ratio Reinforcement Learning (BRRL) framework. We formulate a novel regularized and constrained policy optimization problem and derive its analytical optimal solution. We prove that this solution ensures monotonic performance improvement. To handle parameterized policy classes, we develop a policy optimization algorithm called Bounded Policy Optimization (BPO) that minimizes an advantage-weighted divergence between the policy and the analytic optimal solution from BRRL. We further establish a lower bound on the expected performance of the resulting policy in terms of the BPO loss function. Notably, our framework also provides a new theoretical lens to interpret the success of the PPO loss, and connects trust region policy optimization and the Cross-Entropy Method (CEM). We additionally extend BPO to Group-relative BPO (GBPO) for LLM fine-tuning. Empirical evaluations of BPO across MuJoCo, Atari, and complex IsaacLab environments (e.g., Humanoid locomotion), and of GBPO for LLM fine-tuning tasks, demonstrate that BPO and GBPO generally match or outperform PPO and GRPO in stability and final performance. 

---
# When Can LLMs Learn to Reason with Weak Supervision? 

**Authors**: Salman Rahman, Jingyan Shen, Anna Mordvina, Hamid Palangi, Saadia Gabriel, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2604.18574)  

**Abstract**: Large language models have achieved significant reasoning improvements through reinforcement learning with verifiable rewards (RLVR). Yet as model capabilities grow, constructing high-quality reward signals becomes increasingly difficult, making it essential to understand when RLVR can succeed under weaker forms of supervision. We conduct a systematic empirical study across diverse model families and reasoning domains under three weak supervision settings: scarce data, noisy rewards, and self-supervised proxy rewards. We find that generalization is governed by training reward saturation dynamics: models that generalize exhibit a prolonged pre-saturation phase during which training reward and downstream performance climb together, while models that saturate rapidly memorize rather than learn. We identify reasoning faithfulness, defined as the extent to which intermediate steps logically support the final answer, as the pre-RL property that predicts which regime a model falls into, while output diversity alone is uninformative. Motivated by these findings, we disentangle the contributions of continual pre-training and supervised fine-tuning, finding that SFT on explicit reasoning traces is necessary for generalization under weak supervision, while continual pre-training on domain data amplifies the effect. Applied together to Llama3.2-3B-Base, these interventions enable generalization across all three settings where the base model previously failed. 

---
# Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale 

**Authors**: A. Sophia Koepke, Daniil Zverev, Shiry Ginosar, Alexei A. Efros  

**Link**: [PDF](https://arxiv.org/pdf/2604.18572)  

**Abstract**: The Platonic Representation Hypothesis suggests that neural networks trained on different modalities (e.g., text and images) align and eventually converge toward the same representation of reality. If true, this has significant implications for whether modality choice matters at all. We show that the experimental evidence for this hypothesis is fragile and depends critically on the evaluation regime. Alignment is measured using mutual nearest neighbors on small datasets ($\approx$1K samples) and degrades substantially as the dataset is scaled to millions of samples. The alignment that remains between model representations reflects coarse semantic overlap rather than consistent fine-grained structure. Moreover, the evaluations in Huh et al. are done in a one-to-one image-caption setting, a constraint that breaks down in realistic many-to-many settings and further reduces alignment. We also find that the reported trend of stronger language models increasingly aligning with vision does not appear to hold for newer models. Overall, our findings suggest that the current evidence for cross-modal representational convergence is considerably weaker than subsequent works have taken it to be. Models trained on different modalities may learn equally rich representations of the world, just not the same one. 

---
# A multimodal and temporal foundation model for virtual patient representations at healthcare system scale 

**Authors**: Andrew Zhang, Tong Ding, Sophia J. Wagner, Caiwei Tian, Ming Y. Lu, Rowland Pettit, Joshua E. Lewis, Alexandre Misrahi, Dandan Mo, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2604.18570)  

**Abstract**: Modern medicine generates vast multimodal data across siloed systems, yet no existing model integrates the full breadth and temporal depth of the clinical record into a unified patient representation. We introduce Apollo, a multimodal temporal foundation model trained and evaluated on over three decades of longitudinal hospital records from a major US hospital system, composed of 25 billion records from 7.2 million patients, representing 28 distinct medical modalities and 12 major medical specialties. Apollo learns a unified representation space integrating over 100 thousand unique medical events in our clinical vocabulary as well as images and clinical text. This "atlas of medical concepts" forms a computational substrate for modeling entire patient care journeys comprised of sequences of structured and unstructured events, which are compressed by Apollo into virtual patient representations. To assess the potential of these whole-patient representations, we created 322 prognosis and retrieval tasks from a held-out test set of 1.4 million patients. We demonstrate the generalized clinical forecasting potential of Apollo embeddings, including predicting new disease onset risk up to five years in advance (95 tasks), disease progression (78 tasks), treatment response (59 tasks), risk of treatment-related adverse events (17 tasks), and hospital operations endpoints (12 tasks). Using feature attribution techniques, we show that model predictions align with clinically-interpretable multimodal biomarkers. We evaluate semantic similarity search on 61 retrieval tasks, and moreover demonstrate the potential of Apollo as a multimodal medical search engine using text and image queries. Together, these modeling capabilities establish the foundation for computable medicine, where the full context of patient care becomes accessible to computational reasoning. 

---
# Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring and KV-Cache Steering 

**Authors**: Manan Gupta, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.18567)  

**Abstract**: Large language models frequently commit unrecoverable reasoning errors mid-generation: once a wrong step is taken, subsequent tokens compound the mistake rather than correct it. We introduce $\textbf{Latent Phase-Shift Rollback}$ (LPSR): at each generation step, we monitor the residual stream at a critical layer lcrit, detect abrupt directional reversals (phase shifts) via a cosine-similarity $+$ entropy dual gate, and respond by rolling back the KV-cache and injecting a pre-computed steering vector. No fine-tuning, gradient computation, or additional forward passes are required. LPSR achieves $\mathbf{44.0\%}$ on MATH-500 with an 8B model versus $28.8\%$ for standard AR ($+15.2$ pp; McNemar $\chi^2 = 66.96$, $p < 10^{-15}$). Critically, prompted self-correction, the most natural inference-time baseline, scores only $19.8\%$, below standard AR; LPSR exceeds it by $+24.2$ pp ($\chi^2 = 89.4$, $p \approx 0$). LPSR also outperforms Best-of-16 ($+7.8$ pp) at $5.4\times$ lower token cost, and surpasses a standard 70B model ($35.2\%$) with $8.75\times$ fewer parameters at ${\sim}3\times$ the token budget. A 32-layer sweep reveals a novel \textbf{detection-correction dissociation}: error-detection AUC peaks at layer~14 ($0.718$) but task accuracy peaks at layer~16 ($44.0\%$ vs.\ $29.2\%$), demonstrating that optimal monitoring depth differs for detection and correction. 

---
# Transition-Matrix Regularization for Next Dialogue Act Prediction in Counselling Conversations 

**Authors**: Eric Rudolph, Philipp Steigerwald, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2604.18539)  

**Abstract**: This paper studies how empirical dialogue-flow statistics can be incorporated into Next Dialogue Act Prediction (NDAP). A KL regularization term is proposed that aligns predicted act distributions with corpus-derived transition patterns. Evaluated on a 60-class German counselling taxonomy using 5-fold cross-validation, this improves macro-F1 by 9--42% relative depending on encoder and substantially improves dialogue-flow alignment. Cross-dataset validation on HOPE suggests that improvements transfer across languages and counselling domains. In systematic ablations across pretrained encoders and architectures, the findings indicate that transition regularization provides consistent gains and disproportionately benefits weaker baseline models. The results suggest that lightweight discourse-flow priors complement pretrained encoders, especially in fine-grained, data-sparse dialogue tasks. 

---
# Symbolic Synthesis for LTLf+ Obligations 

**Authors**: Giuseppe De Giacomo, Christian Hagemeier, Daniel Hausmann, Nir Piterman  

**Link**: [PDF](https://arxiv.org/pdf/2604.18532)  

**Abstract**: We study synthesis for obligation properties expressed in LTLfp, the extension of LTLf to infinite traces. Obligation properties are positive Boolean combinations of safety and guarantee (co-safety) properties and form the second level of the temporal hierarchy of Manna and Pnueli. Although obligation properties are expressed over infinite traces, they retain most of the simplicity of LTLf. In particular, we show that they admit a translation into symbolically represented deterministic weak automata (DWA) obtained directly from the symbolic deterministic finite automata (DFA) for the underlying LTLf properties on trace prefixes. DWA inherit many of the attractive algorithmic features of DFA, including Boolean closure and polynomial-time minimization. Moreover, we show that synthesis for LTLfp obligation properties is theoretically highly efficient - solvable in linear time once the DWA is constructed. We investigate several symbolic algorithms for solving DWA games that arise in the synthesis of obligation properties and evaluate their effectiveness experimentally. Overall, the results indicate that synthesis for LTLfp obligation properties can be performed with virtually the same effectiveness as LTLf synthesis. 

---
# IDOBE: Infectious Disease Outbreak forecasting Benchmark Ecosystem 

**Authors**: Aniruddha Adiga, Jingyuan Chou, Anshul Chiranth, Bryan Lewis, Ana I. Bento, Shaun Truelove, Geoffrey Fox, Madhav Marathe, Harry Hochheiser, Srini Venkatramanan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18521)  

**Abstract**: Epidemic forecasting has become an integral part of real-time infectious disease outbreak response. While collaborative ensembles composed of statistical and machine learning models have become the norm for real-time forecasting, standardized benchmark datasets for evaluating such methods are lacking. Further, there is limited understanding on performance of these methods for novel outbreaks with limited historical data. In this paper, we propose IDOBE, a curated collection of epidemiological time series focused on outbreak forecasting. IDOBE compiles from multiple data repositories spanning over a century of surveillance and across U.S. states and global locations. We perform derivative-based segmentation to generate over 10,000 outbreaks covering multiple outcomes such as cases and hospitalizations for 13 diseases. We consider a variety of information-theoretic and distributional measures to quantify the epidemiological diversity of the dataset. Finally, we perform multi-horizon short-term forecasting (1- to 4-week-ahead) through the progression of the outbreak using 11 baseline models and report on their performance. In addition to standard metrics such as NMSE and MAPE for point forecasts, we include probabilistic scoring rules such as Normalized Weighted Interval Score (NWIS) to quantify the performance. We find that MLP-based methods have the most robust performance, with statistical methods having a slight edge during the pre-peak phase. IDOBE dataset along with baselines are released publicly on this https URL to enable standardized, reproducible benchmarking of outbreak forecasting methods. 

---
# Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks 

**Authors**: Md Rysul Kabir, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2604.18510)  

**Abstract**: Open-weight language models can be rendered unsafe through several distinct interventions, but the resulting models may differ substantially in capabilities, behavioral profile, and internal failure mode. We study behavioral and mechanistic properties of jailbroken models across three unsafe routes: harmful supervised fine-tuning (SFT), harmful reinforcement learning with verifiable rewards (RLVR), and refusal-suppressing abliteration. All three routes achieve near-ceiling harmful compliance, but they diverge once we move beyond direct harmfulness. RLVR-jailbroken models show minimal degradation and preserve explicit harm recognition in a structured self-audit: they are able to identify harmful prompts and describe how a safe LLM should respond, yet they comply with the harmful request. With RLVR, harmful behavior is strongly suppressed by a reflective safety scaffold: when a harmful prompt is prepended with an instruction to reflect on safety standards, harmful behavior drops close to the baseline. Category-specific RLVR jailbreaks generalize broadly across harmfulness domains. Models jailbroken with SFT show the largest collapse in explicit safety judgments, the highest behavioral drift, and a substantial capability loss on standard benchmarks. Abliteration is family-dependent in both self-audit and response to a reflective safety scaffold. Mechanistic and repair analyses further separate the routes: abliteration is consistent with localized refusal-feature deletion, RLVR with preserved safety geometry but retargeted policy behavior, and SFT with broader distributed drift. Targeted repair partially recovers RLVR-jailbroken models, but has little effect on SFT-jailbroken models. Together, these results show that jailbreaks can produce vastly different properties despite similar harmfulness, with models jailbroken via RLVR showing remarkable similarity to the base model. 

---
# Document-as-Image Representations Fall Short for Scientific Retrieval 

**Authors**: Ghazal Khalighinejad, Raghuveer Thirukovalluru, Alexander H. Oh, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2604.18508)  

**Abstract**: Many recent document embedding models are trained on document-as-image representations, embedding rendered pages as images rather than the underlying source. Meanwhile, existing benchmarks for scientific document retrieval, such as ArXivQA and ViDoRe, treat documents as images of pages, implicitly favoring such representations. In this work, we argue that this paradigm is not well-suited for text-rich multimodal scientific documents, where critical evidence is distributed across structured sources, including text, tables, and figures. To study this setting, we introduce ArXivDoc, a new benchmark constructed from the underlying LaTeX sources of scientific papers. Unlike PDF or image-based representations, LaTeX provides direct access to structured elements (e.g., sections, tables, figures, equations), enabling controlled query construction grounded in specific evidence types. We systematically compare text-only, image-based, and multimodal representations across both single-vector and multi-vector retrieval models. Our results show that: (1) document-as-image representations are consistently suboptimal, especially as document length increases; (2) text-based representations are most effective, even for figure-based queries, by leveraging captions and surrounding context; and (3) interleaved text+image representations outperform document-as-image approaches without requiring specialized training. 

---
# Learning the Riccati solution operator for time-varying LQR via Deep Operator Networks 

**Authors**: Jun Chen, Umberto Biccari, Junmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18507)  

**Abstract**: We propose a computational framework for replacing the repeated numerical solution of differential Riccati equations in finite-horizon Linear Quadratic Regulator (LQR) problems by a learned operator surrogate. Instead of solving a nonlinear matrix-valued differential equation for each new system instance, we construct offline an approximation of the associated solution operator mapping time-dependent system parameters to the Riccati trajectory. The resulting model enables fast online evaluation of approximate optimal feedbacks across a wide class of systems, thereby shifting the computational burden from repeated numerical integration to a one-time learning stage. From a theoretical perspective, we establish control-theoretic guarantees for this operator-based approximation. In particular, we derive bounds quantifying how operator approximation errors propagate to feedback performance, trajectory accuracy, and cost suboptimality, and we prove that exponential stability of the closed-loop system is preserved under sufficiently accurate operator approximation. These results provide a framework to assess the reliability of data-driven approximations in optimal control. On the computational side, we design tailored DeepONet architectures for matrix-valued, time-dependent problems and introduce a progressive learning strategy to address scalability with respect to the system dimension. Numerical experiments on both time-invariant and time-varying LQR problems demonstrate that the proposed approach achieves high accuracy and strong generalization across a wide range of system configurations, while delivering substantial computational speedups compared to classical solvers. The method offers an effective and scalable alternative for parametric and real-time optimal control applications. 

---
# Faster by Design: Interactive Aerodynamics via Neural Surrogates Trained on Expert-Validated CFD 

**Authors**: Nicholas Thumiger, Andrea Bartezzaghi, Mattia Rigotti, Cezary Skura, Thomas Frick, Elisa Serioli, Fabrizio Arbucci, A. Cristiano I. Malossi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18491)  

**Abstract**: Computational Fluid Dynamics (CFD) is central to race-car aerodynamic development, yet its cost -- tens of thousands of core-hours per high-fidelity evaluation -- severely limits the design space exploration feasible within realistic budgets. AI-based surrogate models promise to alleviate this bottleneck, but progress has been constrained by the limited complexity of public datasets, which are dominated by smoothed passenger-car shapes that fail to exercise surrogates on the thin, complex, highly loaded components governing motorsport performance. This work presents three primary contributions. First, we introduce a high-fidelity RANS dataset built on a parametric LMP2-class CAD model and spanning six operating conditions (map points) covering straight-line and cornering regimes, generated and validated by aerodynamics experts at Dallara to preserve features relevant to industrial motorsport. Second, we present the Gauge-Invariant Spectral Transformer (GIST), a graph-based neural operator whose spectral embeddings encode mesh connectivity to enhance predictions on tightly packed, complex geometries. GIST guarantees discretization invariance and scales linearly with mesh size, achieving state-of-the-art accuracy on both public benchmarks and the proposed race-car dataset. Third, we demonstrate that GIST achieves a level of predictive accuracy suitable for early-stage aerodynamic design, providing a first validation of the concept of interactive design-space exploration -- where engineers query a surrogate in place of the CFD solver -- within industrial motorsport workflows. 

---
# LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation 

**Authors**: Samar M. Magdy, Fakhraddin Alwajih, Abdellah El Mekki, Wesam El-Sayed, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2604.18490)  

**Abstract**: Existing MT evaluation frameworks, including automatic metrics and human evaluation schemes such as Multidimensional Quality Metrics (MQM), are largely language-agnostic. However, they often fail to capture dialect- and culture-specific errors in diglossic languages (e.g., Arabic), where translation failures stem from mismatches in language variety, content coverage, and pragmatic appropriateness rather than surface form this http URL introduce LQM: Linguistically Motivated Multidimensional Quality Metrics for MT. LQM is a hierarchical error taxonomy for diagnosing MT errors through six linguistically grounded levels: sociolinguistics, pragmatics, semantics, morphosyntax, orthography, and graphetics (Figure 1). We construct a bidirectional parallel corpus of 3,850 sentences (550 per variety) spanning seven Arabic dialects (Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, and Yemeni), derived from conversational, culturally rich content. We evaluate six LLMs in a zero-shot setting and conduct expert span-level human annotation using LQM, producing 6,113 labeled error spans across 3,495 unique erroneous sentences, along with severity-weighted quality scores. We complement this analysis with an automatic metric (spBLEU). Though validated here on Arabic, LQM is a language-agnostic framework designed to be easily applied to or adapted for other languages. LQM annotated errors data, prompts, and annotation guidelines are publicly available at this https URL. 

---
# Adversarial Humanities Benchmark: Results on Stylistic Robustness in Frontier Model Safety 

**Authors**: Marcello Galisai, Susanna Cifani, Francesco Giarrusso, Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Federico Sartore, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18487)  

**Abstract**: The Adversarial Humanities Benchmark (AHB) evaluates whether model safety refusals survive a shift away from familiar harmful prompt forms. Starting from harmful tasks drawn from MLCommons AILuminate, the benchmark rewrites the same objectives through humanities-style transformations while preserving intent. This extends literature on Adversarial Poetry and Adversarial Tales from single jailbreak operators to a broader benchmark family of stylistic obfuscation and goal concealment. In the benchmark results reported here, the original attacks record 3.84% attack success rate (ASR), while transformed methods range from 36.8% to 65.0%, yielding 55.75% overall ASR across 31 frontier models. Under a European Union AI Act Code-of-Practice-inspired systemic-risk lens, Chemical, biological, radiological and nuclear (CBRN) is the highest bucket. Taken together, this lack of stylistic robustness suggests that current safety techniques suffer from weak generalization: deep understanding of 'non-maleficence' remains a central unresolved problem in frontier model safety. 

---
# Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation 

**Authors**: Tianshi Cao, Jiawei Ren, Yuxuan Zhang, Jaewoo Seo, Jiahui Huang, Shikhar Solanki, Haotian Zhang, Mingfei Guo, Haithem Turki, Muxingzi Li, Yue Zhu, Sipeng Zhang, Zan Gojcic, Sanja Fidler, Kangxue Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.18468)  

**Abstract**: Closed-loop simulation is a core component of autonomous vehicle (AV) development, enabling scalable testing, training, and safety validation before real-world deployment. Neural scene reconstruction converts driving logs into interactive 3D environments for simulation, but it does not produce complete 3D object assets required for agent manipulation and large-viewpoint novel-view synthesis. To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild object observations from real driving logs into complete, simulation-ready assets. Rather than relying on a single model component, we developed a system-level design for real-world AV data that combines large-scale curation of object-centric training tuples, geometry-aware preprocessing across heterogeneous sensors, and a robust training recipe that couples sparse-view-conditioned multiview generation with 3D Gaussian lifting. Within this system, SparseViewDiT is explicitly designed to address limited-angle views and other real-world data challenges. Together with hybrid data curation, augmentation, and self-distillation, this system enables scalable conversion of sparse AV object observations into reusable 3D assets. 

---
# An Integrated Deep-Learning Framework for Peptide-Protein Interaction Prediction and Target-Conditioned Peptide Generation with ConGA-PePPI and TC-PepGen 

**Authors**: Chupei Tang, Junxiao Kong, Moyu Tang, Di Wang, Jixiu Zhai, Ronghao Xie, Shangkun Sima, Tianchi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18467)  

**Abstract**: Motivation: Peptide-protein interactions (PepPIs) are central to cellular regulation and peptide therapeutics, but experimental characterization remains too slow for large-scale screening. Existing methods usually emphasize either interaction prediction or peptide generation, leaving candidate prioritization, residue-level interpretation, and target-conditioned expansion insufficiently integrated. Results: We present an integrated framework for early-stage peptide screening that combines a partner-aware prediction and localization model (ConGA-PepPI) with a target-conditioned generative model (TC-PepGen). ConGA-PepPI uses asymmetric encoding, bidirectional cross-attention, and progressive transfer from pair prediction to binding-site localization, while TC-PepGen preserves target information throughout autoregressive decoding via layerwise conditioning. In five-fold cross-validation, ConGA-PepPI achieved 0.839 accuracy and 0.921 AUROC, with binding-site AUPR values of 0.601 on the protein side and 0.950 on the peptide side, and remained competitive on external benchmarks. Under a controlled length-conditioned benchmark, 40.39% of TC-PepGen peptides exceeded native templates in AlphaFold 3 ipTM, and unconstrained generation retained evidence of target-conditioned signal. 

---
# Progressive Online Video Understanding with Evidence-Aligned Timing and Transparent Decisions 

**Authors**: Kecheng Zhang, Zongxin Yang, Mingfei Han, Haihong Hao, Yunzhi Zhuge, Changlin Li, Junhan Zhao, Zhihui Li, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18459)  

**Abstract**: Visual agents operating in the wild must respond to queries precisely when sufficient evidence first appears in a video stream, a critical capability that is overlooked by conventional video LLMs evaluated in offline settings. The shift to an online, streaming paradigm introduces significant challenges: a lack of decision transparency, the difficulty of aligning response timing with visual evidence, and the need to maintain a global, causally consistent understanding under tight computational budgets. To address these issues, we propose a novel framework that decouples reasoning control from memory integration. We introduce \textbf{\model{}}, an instantiation of this framework with two core components. First, the \emph{Active Thinking Decision Maker (ATDM)} is a transparent reasoning controller that externalizes its decision process using observable progress ($\boldsymbol{\rho}$) and confidence ($\boldsymbol{c}$) metrics. This allows it to precisely time its response $t_r$ to match the first-sufficient-evidence timestamp $t^\star$ while streaming its reasoning to the user. Second, the \emph{Hierarchical Progressive Semantic Integration (HPSI)} module acts as an efficient memory system. It employs a set of learnable, multi-level aggregation tokens that are propagated across clips to build a rich, global cognitive state without exceeding token budgets. %Our approach sets a new standard on key online video understanding benchmarks, achieving strong performance of \textbf{71.6\%} on StreamingBench and \textbf{46.9\%} on OVOBench, demonstrating a robust solution for evidence-aligned and transparent online video analysis. Extensive experiments demonstrate the effectiveness of ATDM and HPSI, e.g., Thinking-QwenVL improves the accuracy of the previous state-of-the-art from 67.63\% to 71.60\% on the StreamingBench benchmark. 

---
# ProtoCLIP: Prototype-Aligned Latent Refinement for Robust Zero-Shot Chest X-Ray Classification 

**Authors**: Florian Kittler, Sheethal Bhat, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2604.18444)  

**Abstract**: Zero-shot vision-language models (VLMs) have shown promise for chest radiograph classification, but their performance is often limited by confounding label co-occurrence, long-tail class imbalance, and transfer instability under domain shift. We propose ProtoCLIP, a refinement strategy for CLIP-style VLMs that improves zero-shot discrimination through targeted data curation and distilled anchor alignment. Specifically, we construct pathology-focused training subsets with curated negative samples to reduce co-occurrence bias. We also introduce a representation-preserving distillation objective to stabilize adaptation while maintaining semantic structure and improving discrimination of clinically relevant co-occurring pathologies. Evaluated on an unseen dataset VinDr-CXR, ProtoCLIP improves AUC by 2-10 percentage points over a strong CLIP-based baseline across multiple findings. For pneumothorax specifically, ProtoCLIP achieves a state-of-the-art AUC of 0.94. These results demonstrate that anchor-guided refinement, coupled with curated supervision and controlled adaptation, can mitigate common zero-shot transfer failures in medical VLMs without requiring large-scale retraining. 

---
# Revisiting Change VQA in Remote Sensing with Structured and Native Multimodal Qwen Models 

**Authors**: Yakoub Bazi, Mohamad M. Al Rahhal, Mansour Zuair, Faroun Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2604.18429)  

**Abstract**: Change visual question answering (Change VQA) addresses the problem of answering natural-language questions about semantic changes between bi-temporal remote sensing (RS) images. Although vision-language models (VLMs) have recently been studied for temporal RS image understanding, Change VQA remains underexplored in the context of modern multimodal models. In this letter, we revisit the CDVQA benchmark using recent Qwen models under a unified low-rank adaptation (LoRA) setting. We compare Qwen3-VL, which follows a structured vision-language pipeline with multi-depth visual conditioning and a full-attention decoder, with Qwen3.5, a native multimodal model that combines a single-stage alignment with a hybrid decoder backbone. Experimental results on the official CDVQA test splits show that recent VLMs improve over earlier specialized baselines. They further show that performance does not scale monotonically with model size, and that native multimodal models are more effective than structured vision-language pipelines for this task. These findings indicate that tightly integrated multimodal backbones contribute more to performance than scale or explicit multi-depth visual conditioning for language-driven semantic change reasoning in RS imagery. 

---
# AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment 

**Authors**: Yixuan Wang, Yue Huang, Hong Qian, Yunzhao Wei, Yifei Ding, Wenkai Wang, Zhi Liu, Zhongjing Huang, Aimin Zhou, Jiajun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.18398)  

**Abstract**: Creativity has become a core competence in the era of LLMs and human-AI collaboration, underpinning innovation in real-world problem solving. Crucially, the systematic improvement of creativity necessitates scientifically valid assessment instruments. Psychometric research recognizes context-based assessment as an effective way to measure creative thinking. However, high-quality expert-designed contexts remain scarce. Existing LLM-based generators often struggle with insufficient assessment cues, weak narrative coherence, limited stylistic diversity, and poor support for creative thinking. To address these challenges, we propose AlphaContext, an evolutionary tree-based psychometric context generator for creativity assessment. First, the HyperTree Outline Planner formalizes expert-designed outlining as a rule-guided hypertree and performs top-down hierarchical planning. The MCTS-based Context Generator fills the outline via MCTS to balance global structure and local quality. Then, the Evolutionary Context Optimizer evolves contexts with MAP-Elites by repeatedly updating niche elites to jointly improve diversity and quality. Finally, the Assessment-Guided Evolution Refiner simulates virtual participants with diverse styles and recycles weak contexts for further evolution. Experiments show that AlphaContext yields an average improvement of 8% over competitive methods across 6 quality metrics. 

---
# Randomly Initialized Networks Can Learn from Peer-to-Peer Consensus 

**Authors**: Esteban Rodríguez-Betancourt, Edgar Casasola-Murillo  

**Link**: [PDF](https://arxiv.org/pdf/2604.18390)  

**Abstract**: In self-supervised learning, self-distilled methods have shown impressive performance, learning representations useful for downstream tasks and even displaying emergent properties. However, state-of-the-art methods usually rely on ensembles of complex mechanisms, with many design choices that are empirically motivated and not well understood.
In this work, we explore the role of self-distillation within learning dynamics. Specifically, we isolate the effect of self-distillation by training a group of randomly initialized networks, removing all other common components such as projectors, predictors, and even pretext tasks. Our findings show that even this minimal setup can lead to learned representations with non-trivial improvements over a random baseline on downstream tasks. We also demonstrate how this effect varies with different hyperparameters and present a short analysis of what is being learned by the models under this setup. 

---
# IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters 

**Authors**: Hongwei Zheng, Weiqi Wu, Zhengjia Wang, Guanyu Jiang, Haoming Li, Tianyu Wu, Yongchun Zhu, Jingwu Chen, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18375)  

**Abstract**: Conversational agents, such as ChatGPT and Doubao, have become essential daily assistants for billions of users. To further enhance engagement, these systems are evolving from passive responders to proactive companions. However, existing efforts focus on activation within ongoing dialogues, while overlooking a key real-world bottleneck. In the conversation initiation stage, users may have a vague need but no explicit query intent, creating a first-message barrier where the conversation holds before it begins. To overcome this, we introduce Conversation Starter Generation: generating personalized starters to guide users into conversation. However, unlike in-conversation stages where immediate context guides the response, initiation must operate in a cold-start moment without explicit user intent. To pioneer in this direction, we present IceBreaker that frames human ice-breaking as a two-step handshake: (i) evoke resonance via Resonance-Aware Interest Distillation from session summaries to capture trigger interests, and (ii) stimulate interaction via Interaction-Oriented Starter Generation, optimized with personalized preference alignment and a self-reinforced loop to maximize engagement. Online A/B tests on one of the world's largest conversational agent products show that IceBreaker improves user active days by +0.184% and click-through rate by +9.425%, and has been deployed in production. 

---
# Dissecting AI Trading: Behavioral Finance and Market Bubbles 

**Authors**: Shumiao Ouyang, Pengfei Sui  

**Link**: [PDF](https://arxiv.org/pdf/2604.18373)  

**Abstract**: We study how AI agents form expectations and trade in experimental asset markets. Using a simulated open-call auction populated by autonomous Large Language Model (LLM) agents, we document three main findings. First, AI agents exhibit classic behavioral patterns: a pronounced disposition effect and recency-weighted extrapolative beliefs. Second, these individual-level patterns aggregate into equilibrium dynamics that replicate classic experimental findings (Smith et al., 1988), including the predictive power of excess demand for future prices and the positive relationship between disagreement and trading volume. Third, by analyzing the agents' reasoning text through a twenty-mechanism scoring framework, we show that targeted prompt interventions causally amplify or suppress specific behavioral mechanisms, significantly altering the magnitude of market bubbles. 

---
# Tight Auditing of Differential Privacy in MST and AIM 

**Authors**: Georgi Ganev, Meenatchi Sundaram Muthu Selva Annamalai, Bogdan Kulynych  

**Link**: [PDF](https://arxiv.org/pdf/2604.18352)  

**Abstract**: State-of-the-art Differentially Private (DP) synthetic data generators such as MST and AIM are widely used, yet tightly auditing their privacy guarantees remains challenging. We introduce a Gaussian Differential Privacy (GDP)-based auditing framework that measures privacy via the full false-positive/false-negative tradeoff. Applied to MST and AIM under worst-case settings, our method provides the first tight audits in the strong-privacy regime. For $(\epsilon,\delta)=(1,10^{-2})$, we obtain $\mu_{emp}\approx0.43$ vs. implied $\mu=0.45$, showing a small theory-practice gap.
Our code is publicly available: this https URL. 

---
# AdaCluster: Adaptive Query-Key Clustering for Sparse Attention in Video Generation 

**Authors**: Haoyue Tan, Shengnan Wang, Yulin Qiao, Juncheng Zhang, Youhui Bai, Ping Gong, Zewen Jin, Cheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.18348)  

**Abstract**: Video diffusion transformers (DiTs) suffer from prohibitive inference latency due to quadratic attention complexity. Existing sparse attention methods either overlook semantic similarity or fail to adapt to heterogeneous token distributions across layers, leading to model performance degradation. We propose AdaCluster, a training-free adaptive clustering framework that accelerates the generation of DiTs while preserving accuracy. AdaCluster applies an angle-similarity-preserving clustering method to query vectors for higher compression, and designs a euclidean-similarity-preserving clustering method for keys, covering cluster number assignment, threshold-wise adaptive clustering, and efficient critical cluster selection. Experiments on CogVideoX-2B, HunyuanVideo, and Wan-2.1 on one A40 GPU demonstrate up to 1.67-4.31x speedup with negligible quality degradation. 

---
# Multilingual Training and Evaluation Resources for Vision-Language Models 

**Authors**: Daniela Baiamonte, Elena Fano, Matteo Gabburo, Stefano Simonazzi, Leonardo Rigutini, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2604.18347)  

**Abstract**: Vision Language Models (VLMs) achieved rapid progress in the recent years. However, despite their growth, VLMs development is heavily grounded on English, leading to two main limitations: (i) the lack of multilingual and multimodal datasets for training, and (ii) the scarcity of comprehensive evaluation benchmarks across languages. In this work, we address these gaps by introducing a new comprehensive suite of resources for VLMs training and evaluation spanning five European languages (English, French, German, Italian, and Spanish). We adopt a regeneration-translation paradigm that produces high-quality cross-lingual resources by combining curated synthetic generation and manual annotation. Specifically, we build Multi-PixMo, a training corpus obtained regenerating examples from Pixmo pre-existing datasets with permissively licensed models: PixMo-Cap, PixMo-AskModelAnything, and CoSyn-400k. On the evaluation side, we construct a set of multilingual benchmarks derived translating widely used English datasets (MMbench, ScienceQA, MME, POPE, AI2D). We assess the quality of these resources through qualitative and quantitative human analyses, measuring inter-annotator agreement. Additionally, we perform ablation studies to demonstrate the impact of multilingual data, with respect to English only, in VLMs training. Experiments, comprising 3 different models show that using multilingual, multimodal examples for training VLMs aids is consistently beneficial on non-English benchmarks, with positive transfer to English as well. 

---
# EVE: Verifiable Self-Evolution of MLLMs via Executable Visual Transformations 

**Authors**: Yongrui Heng, Chaoya Jiang, Han Yang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.18320)  

**Abstract**: Self-evolution of multimodal large language models (MLLMs) remains a critical challenge: pseudo-label-based methods suffer from progressive quality degradation as model predictions drift, while template-based methods are confined to a static set of transformations that cannot adapt in difficulty or diversity. We contend that robust, continuous self-improvement requires not only deterministic external feedback independent of the model's internal certainty, but also a mechanism to perpetually diversify the training distribution. To this end, we introduce EVE (Executable Visual transformation-based self-Evolution), a novel framework that entirely bypasses pseudo-labels by harnessing executable visual transformations continuously enriched in both variety and complexity. EVE adopts a Challenger-Solver dual-policy architecture. The Challenger maintains and progressively expands a queue of visual transformation code examples, from which it synthesizes novel Python scripts to perform dynamic visual transformations. Executing these scripts yields VQA problems with absolute, execution-verified ground-truth answers, eliminating any reliance on model-generated supervision. A multi-dimensional reward system integrating semantic diversity and dynamic difficulty calibration steers the Challenger to enrich its code example queue while posing progressively more challenging tasks, preventing mode collapse and fostering reciprocal co-evolution between the two policies. Extensive experiments demonstrate that EVE consistently surpasses existing self-evolution methods, establishing a robust and scalable paradigm for verifiable MLLM self-evolution. The code is available at this https URL . 

---
# On the Importance and Evaluation of Narrativity in Natural Language AI Explanations 

**Authors**: Mateusz Cedro, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2604.18311)  

**Abstract**: Explainable AI (XAI) aims to make the behaviour of machine learning models interpretable, yet many explanation methods remain difficult to understand. The integration of Natural Language Generation into XAI aims to deliver explanations in textual form, making them more accessible to practitioners. Current approaches, however, largely yield static lists of feature importances. Although such explanations indicate what influences the prediction, they do not explain why the prediction occurs. In this study, we draw on insights from social sciences and linguistics, and argue that XAI explanations should be presented in the form of narratives. Narrative explanations support human understanding through four defining properties: continuous structure, cause-effect mechanisms, linguistic fluency, and lexical diversity. We show that standard Natural Language Processing (NLP) metrics based solely on token probability or word frequency fail to capture these properties and can be matched or exceeded by tautological text that conveys no explanatory content. To address this issue, we propose seven automatic metrics that quantify the narrative quality of explanations along the four identified dimensions. We benchmark current state-of-the-art explanation generation methods on six datasets and show that the proposed metrics separate descriptive from narrative explanations more reliably than standard NLP metrics. Finally, to further advance the field, we propose a set of problem-agnostic XAI Narrative generation rules for producing natural language XAI explanations, so that the resulting XAI Narratives exhibit stronger narrative properties and align with the findings from the linguistic and social science literature. 

---
# Long-Text-to-Image Generation via Compositional Prompt Decomposition 

**Authors**: Jen-Yuan Huang, Tong Lin, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.18258)  

**Abstract**: While modern text-to-image (T2I) models excel at generating images from intricate prompts, they struggle to capture the key details when the inputs are descriptive paragraphs. This limitation stems from the prevalence of concise captions that shape their training distributions. Existing methods attempt to bridge this gap by either fine-tuning T2I models on long prompts, which generalizes poorly to longer lengths; or by projecting the oversize inputs into normal-prompt space and compromising fidelity. We propose Prompt Refraction for Intricate Scene Modeling (PRISM), a compositional approach that enables pre-trained T2I models to process long sequence inputs. PRISM uses a lightweight module to extract constituent representations from the long prompts. The T2I model makes independent noise predictions for each component, and their outputs are merged into a single denoising step using energy-based conjunction. We evaluate PRISM across a wide range of model architectures, showing comparable performances to models fine-tuned on the same training data. Furthermore, PRISM demonstrates superior generalization, outperforming baseline models by 7.4% on prompts over 500 tokens in a challenging public benchmark. 

---
# DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion 

**Authors**: Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.18257)  

**Abstract**: Query auto-completion (QAC) has been widely studied in the context of web search, yet remains underexplored for in-document search, which we term DocQAC. DocQAC aims to enhance search productivity within long documents by helping users craft faster, more precise queries, even for complex or hard-to-spell terms. While global historical queries are available to both WebQAC and DocQAC, DocQAC uniquely accesses document-specific context, including the current document's content and its specific history of user query interactions.
To address this setting, we propose a novel adaptive trie-guided decoding framework that uses user query prefixes to softly steer language models toward high-quality completions. Our approach introduces an adaptive penalty mechanism with tunable hyperparameters, enabling a principled trade-off between model confidence and trie-based guidance. To efficiently incorporate document context, we explore retrieval-augmented generation (RAG) and lightweight contextual document signals such as titles, keyphrases, and summaries.
When applied to encoder-decoder models like T5 and BART, our trie-guided framework outperforms strong baselines and even surpasses much larger instruction-tuned models such as LLaMA-3 and Phi-3 on seen queries across both seen and unseen documents. This demonstrates its practicality for real-world DocQAC deployments, where efficiency and scalability are critical. We evaluate our method on a newly introduced DocQAC benchmark derived from ORCAS, enriched with query-document pairs. We make both the DocQAC dataset (this https URL) and code (this https URL) publicly available. 

---
# Style-Based Neural Architectures for Real-Time Weather Classification 

**Authors**: Hamed Ouattara, Pascal Houssam Salmane, Pierre Duthon, Frédéric Bernardin, Omar Ait Aider  

**Link**: [PDF](https://arxiv.org/pdf/2604.18251)  

**Abstract**: In this paper, we present three neural network architectures designed for real-time classification of weather conditions (sunny, rain, snow, fog) from images. These models, inspired by recent advances in style transfer, aim to capture the stylistic elements present in images. One model, called "Multi-PatchGAN", is based on PatchGANs used in well-known architectures such as Pix2Pix and CycleGAN, but here adapted with multiple patch sizes for detection tasks. The second model, "Truncated ResNet50", is a simplified version of ResNet50 retaining only its first nine layers. This truncation, determined by an evolutionary algorithm, facilitates the extraction of high-frequency features essential for capturing subtle stylistic details. Finally, we propose "Truncated ResNet50 with Gram Matrix and Attention", which computes Gram matrices for each layer during training and automatically weights them via an attention mechanism, thus optimizing the extraction of the most relevant stylistic expressions for classification. These last two models outperform the state of the art and demonstrate remarkable generalization capability on several public databases. Although developed for weather detection, these architectures are also suitable for other appearance-based classification tasks, such as animal species recognition, texture classification, disease detection in medical imaging, or industrial defect identification. 

---
# Towards Disentangled Preference Optimization Dynamics Beyond Likelihood Displacement 

**Authors**: Wei Chen, Yubing Wu, Junmei Yang, Delu Zeng, Qibin Zhao, John Paisley, Min Chen, Zhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18239)  

**Abstract**: Preference optimization is widely used to align large language models (LLMs) with human preferences. However, many margin-based objectives suppress the chosen response along with the rejected one, a phenomenon known as likelihood displacement, and no general mechanism currently prevents this across objectives.
We bridge this gap by presenting a unified \emph{incentive-score decomposition} of preference optimization, revealing that diverse objectives share identical local update directions and differ only in their scalar weighting coefficients.
Building on this decomposition, by analyzing the dynamics of the chosen/rejected likelihoods, we identify the \emph{disentanglement band} (DB), a simple, testable condition that characterizes when training can avoid likelihood displacement by realizing the preferred pathway: suppressing the loser while maintaining the winner, possibly after an initial transient.
Leveraging the DB, we propose a plug-and-play \emph{reward calibration} (RC) that adaptively rebalances chosen versus rejected updates to satisfy the DB and mitigate likelihood displacement, without redesigning the base objective.
Empirical results show that RC steers training toward more disentangled dynamics and often improves downstream performance across a range of objectives. Our code is available at this https URL. 

---
# Semantic-based Distributed Learning for Diverse and Discriminative Representations 

**Authors**: Zhuojun Tian, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2604.18237)  

**Abstract**: In large-scale distributed scenarios, increasingly complex tasks demand more intelligent collaboration across networks, requiring the joint extraction of structural representations from data samples. However, conventional task-specific approaches often result in nonstructural embeddings, leading to collapsed variability among data samples within the same class, particularly in classification tasks. To address this issue and fully leverage the intrinsic structure of data for downstream applications, we propose a novel distributed learning framework that ensures both diverse and discriminative representations. For independent and identically distributed (i.i.d.) data, we reformulate and decouple the global optimization function by introducing constraints on representation variance. The update rules are then derived and simplified using a primal-dual approach. For non-i.i.d. data distributions, we tackle the problem by clustering and virtually replicating nodes, allowing model updates within each cluster using block coordinate descent. In both cases, the resulting optimal solutions are theoretically proven to maintain discriminative and diverse properties, with a guaranteed convergence for i.i.d. conditions. Additionally, semantic information from representations is shared among nodes, reducing the need for common neural network architectures. Finally, extensive simulations on MNIST, CIFAR-10 and CIFAR-100 confirm the effectiveness of the proposed algorithms in capturing global structural representations. 

---
# Negative Advantage Is a Double-Edged Sword: Calibrating Advantage in GRPO for Deep Search 

**Authors**: Jiayi Wu, Ruobing Xie, Zeqian Huang, Lei Jiang, Can Xu, Kangyang Luo, Ming Gao, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.18235)  

**Abstract**: Deep search agents can autonomously initiate multi-turn interactions with search engines, thereby exhibiting strong question-answering capabilities. Such performance critically relies on Group Relative Policy Optimization (GRPO) as its core training algorithm. However, GRPO still faces several challenges in deep search settings. First, there exists a substantial mismatch between the correctness of intermediate steps and the reward signal, causing numerous correct intermediate steps to be incorrectly penalized when the final answer is wrong. Second, training is highly unstable, often resulting in degradation of natural language ability or even catastrophic training collapse. Our analysis attributes these issues to coarse-grained advantage assignment and an imbalance between positive and negative advantages. To address these problems, we propose CalibAdv, an advantage calibration method specifically designed for deep search tasks. Specifically, CalibAdv leverages the correctness of intermediate steps to downscale excessive negative advantages at a fine-grained level. It then rebalances positive and negative advantages in the answer component. Extensive experiments across three models and seven benchmarks demonstrate that CalibAdv improves both model performance and training stability. Our code is available at this https URL. 

---
# Evaluating Multi-Hop Reasoning in RAG Systems: A Comparison of LLM-Based Retriever Evaluation Strategies 

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18234)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge to answer questions more accurately.
However, research on evaluating RAG systems-particularly the retriever component-remains limited, as most existing work focuses on single-context retrieval rather than multi-hop queries, where individual contexts may appear irrelevant in isolation but are essential when combined. In this research, we use the HotPotQA, MuSiQue, and SQuAD datasets to simulate a RAG system and compare three LLM-as-judge evaluation strategies, including our proposed Context-Aware Retriever Evaluation (CARE). Our goal is to better understand how multi-hop reasoning can be most effectively evaluated in RAG systems.
Experiments with LLMs from OpenAI, Meta, and Google demonstrate that CARE consistently outperforms existing methods for evaluating multi-hop reasoning in RAG systems. The performance gains are most pronounced in models with larger parameter counts and longer context windows, while single-hop queries show minimal sensitivity to context-aware evaluation. Overall, the results highlight the critical role of context-aware evaluation in improving the reliability and accuracy of retrieval-augmented generation systems, particularly in complex query scenarios. To ensure reproducibility, we provide the complete data of our experiments at this https URL. 

---
# Aether: Network Validation Using Agentic AI and Digital Twin 

**Authors**: Jordan Auge, Sam Betts, Giovanna Carofiglio, Giulio Grassi, Martin Gysi, John Kenneth d'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2604.18233)  

**Abstract**: Network change validation remains a critical yet predominantly manual, time-consuming, and error-prone process in modern network operations. While formal network verification has made substantial progress in proving correctness properties, it is typically applied in offline, pre-deployment settings and faces challenges in accommodating continuous changes and validating live production behavior. Current operational approaches typically involve scattered testing tools, resulting in partial coverage and errors that surface only after deployment. In this paper, we present Aether, a novel approach that integrates Generative Agentic AI with a multi-functional Network Digital Twin to automate and streamline network change validation workflows. It features an agentic architecture with five specialized Network Operations AI agents that collaboratively handle the change validation lifecycle from intent analysis to network verification and testing. Aether agents use a unified Network Digital Twin integrating modeling, simulation, and emulation to maintain a consistent, up-to-date network view for verification and testing. By orchestrating agent collaboration atop this digital twin, Aether enables automated, rapid network change validation while reducing manual effort, minimizing errors, and improving operational agility and cost-effectiveness. We evaluate Aether over synthetic network change scenarios covering main classes of network changes and on past incidents from a major ISP operational network, demonstrating promising results in error detection (100%), diagnostic coverage (92-96%), and speed (6-7 minutes) over traditional methods. 

---
# Is SAM3 ready for pathology segmentation? 

**Authors**: Qiuyu Kong, Shakiba Sharifi, Zanxi Ruan, Yiming Wang, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2604.18225)  

**Abstract**: Is Segment Anything Model 3 (SAM3) capable in segmenting Any Pathology Images? Digital pathology segmentation spans tissue-level and nuclei-level scales, where traditional methods often suffer from high annotation costs and poor generalization. SAM3 introduces Promptable Concept Segmentation, offering a potential automated interface via text prompts. With this work, we propose a systematic evaluation protocol to explore the capability space of SAM3 in a structured manner. Specifically, we evaluate SAM3 under different supervision settings including zero-shot, few-shot, and supervised with varying prompting strategies. Our extensive evaluation on pathological datasets including NuInsSeg, PanNuke and GlaS, reveals that: this http URL-only prompts poorly activate nuclear concepts. this http URL is highly sensitive to visual prompt types and budgets. this http URL-shot learning offers gains, but SAM3 lacks robustness against visual prompt noise. and 4.a significant gap persists between prompt-based usage and task-trained adapter-based reference. Our study delineates SAM3's boundaries in pathology image segmentation and provides practical guidance on the necessity of pathology domain adaptation. 

---
# WebCompass: Towards Multimodal Web Coding Evaluation for Code Language Models 

**Authors**: Xinping Lei, Xinyu Che, Junqi Xiong, Chenchen Zhang, Yukai Huang, Chenyu Zhou, Haoyang Huang, Minghao Liu, Letian Zhu, Hongyi Ye, Jinhua Hao, Ken Deng, Zizheng Zhan, Han Li, Dailin Li, Yifan Yao, Ming Sun, Zhaoxiang Zhang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18224)  

**Abstract**: Large language models are rapidly evolving into interactive coding agents capable of end-to-end web coding, yet existing benchmarks evaluate only narrow slices of this capability, typically text-conditioned generation with static-correctness metrics, leaving visual fidelity, interaction quality, and codebase-level reasoning largely unmeasured. We introduce WebCompass, a multimodal benchmark that provides unified lifecycle evaluation of web engineering capability. Recognizing that real-world web coding is an iterative cycle of generation, editing, and repair, WebCompass spans three input modalities (text, image, video) and three task types (generation, editing, repair), yielding seven task categories that mirror professional workflows. Through a multi-stage, human-in-the-loop pipeline, we curate instances covering 15 generation domains, 16 editing operation types, and 11 repair defect types, each annotated at Easy/Medium/Hard levels. For evaluation, we adopt a checklist-guided LLM-as-a-Judge protocol for editing and repair, and propose a novel Agent-as-a-Judge paradigm for generation that autonomously executes generated websites in a real browser, explores interactive behaviors via the Model Context Protocol (MCP), and iteratively synthesizes targeted test cases, closely approximating human acceptance testing. We evaluate representative closed-source and open-source models and observe that: (1) closed-source models remain substantially stronger and more balanced; (2) editing and repair exhibit distinct difficulty profiles, with repair preserving interactivity better but remaining execution-challenging; (3) aesthetics is the most persistent bottleneck, especially for open-source models; and (4) framework choice materially affects outcomes, with Vue consistently challenging while React and Vanilla/HTML perform more strongly depending on task type. 

---
# Scalable Neighborhood-Based Multi-Agent Actor-Critic 

**Authors**: Tim Goppelsroeder, Rasmus Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2604.18190)  

**Abstract**: We propose MADDPG-K, a scalable extension to Multi-Agent Deep Deterministic Policy Gradient (MADDPG) that addresses the computational limitations of centralized critic approaches. Centralized critics, which condition on the observations and actions of all agents, have demonstrated significant performance gains in cooperative and competitive multi-agent settings. However, their critic networks grow linearly in input size with the number of agents, making them increasingly expensive to train at scale. MADDPG-K mitigates this by restricting each agent's critic to the $k$ closest agents under a chosen metric which in our case is Euclidean distance. This ensures a constant-size critic input regardless of the total agent count. We analyze the complexity of this approach, showing that the quadratic cost it retains arises from cheap scalar distance computations rather than the expensive neural network matrix multiplications that bottleneck standard MADDPG. We validate our method empirically across cooperative and adversarial environments from the Multi-Particle Environment suite, demonstrating competitive or superior performance compared to MADDPG, faster convergence in cooperative settings, and better runtime scaling as the number of agents grows. Our code is available at this https URL . 

---
# Committed SAE-Feature Traces for Audited-Session Substitution Detection in Hosted LLMs 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18179)  

**Abstract**: Hosted-LLM providers have a silent-substitution incentive: advertise a stronger model while serving cheaper replies. Probe-after-return schemes such as SVIP leave a parallel-serve side-channel, since a dishonest provider can route the verifier's probe to the advertised model while serving ordinary users from a substitute. We propose a commit-open protocol that closes this gap. Before any opening request, the provider commits via a Merkle tree to a per-position sparse-autoencoder (SAE) feature-trace sketch of its served output at a published probe layer. A verifier opens random positions, scores them against a public named-circuit probe library calibrated with cross-backend noise, and decides with a fixed-threshold joint-consistency z-score rule. We instantiate the protocol on three backbones -- Qwen3-1.7B, Gemma-2-2B, and a 4.5x scale-up to Gemma-2-9B with a 131k-feature SAE. Of 17 attackers spanning same-family lifts, cross-family substitutes, and rank-<=128 adaptive LoRA, all are rejected at a shared, scale-stable threshold; the same attackers all evade a matched SVIP-style parallel-serve baseline. A white-box end-to-end attack that backpropagates through the frozen SAE encoder does not close the margin, and a feature-forgery attacker that never runs M_hon is bounded in closed form by an intrinsic-dimension argument. Commitment adds <=2.1% to forward-only wall-clock at batch 32. 

---
# STaD: Scaffolded Task Design for Identifying Compositional Skill Gaps in LLMs 

**Authors**: Sungeun An, Swanand Ravindra Kadhe, Shailja Thakur, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.18177)  

**Abstract**: Benchmarks are often used as a standard to understand LLM capabilities in different domains. However, aggregate benchmark scores provide limited insight into compositional skill gaps of LLMs and how to improve them. To make these weaknesses visible, we propose Scaffolded Task Design (STaD) framework. STaD generates controlled variations of benchmark tasks based on the concept of scaffolding, which introduces structured, incremental support in a step-by-step manner. Rather than inspecting failures individually, this approach enables systematic and scalable probing of model behavior by identifying the specific reasoning skill compositions they lack. Treating the LLM as a black box, our experiments on six models of varying sizes reveal multiple failure points in three reasoning benchmarks and highlight each model's unique and distinct skill gaps. 

---
# Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18170)  

**Abstract**: LLMs edit text and code by autoregressively regenerating the full output, even when most tokens appear verbatim in the input. We study Copy-as-Decode, a decoding-layer mechanism that recasts edit generation as structured decoding over a two-primitive grammar: <copy lines="i-j"/> references an input line range, <gen>...</gen> emits new content. A token-level FSM guarantees syntactic validity, and a serving-layer primitive updates the KV cache for each copy span via a single parallel-prefill forward rather than $N$ autoregressive steps -- sharing the parallel-forward kernel of speculative decoding but with input tokens as the draft and program-enforced acceptance replacing probabilistic verification. We report an upper-bound analysis that requires no end-to-end training. (i) Kernel speedup: on Qwen2.5-{1.5B, 7B}, copying $N$ tokens via parallel prefill is $6.8\times$--$303\times$ faster than autoregressive ($N \in [8, 512]$, A100 80GB bf16). (ii) Copy ceiling: on ProbeEdit and HumanEvalPack-Fix (Py/JS), $74$--$98\%$ of gold tokens are reachable under the line-level primitive; composed with the empirical kernel over each corpus's span histogram this yields a closed-form wall-clock bound of $29.0\times / 3.4\times / 4.2\times$ ($13.0\times$ pooled). A token-level extension reaches $91$--$99\%$ coverage with $4.5\times$--$6.5\times$ floors. (iii) Pipeline losslessness: oracle programs round-trip through the deterministic resolver on all $482$ cases, localizing any downstream failure to span selection rather than the mechanism. A perturbation study shows pooled EM drops from $100\%$ to $15.48\%$ under off-by-one noise. A fine-tuning pilot on Qwen2.5-Coder-1.5B lifts HEvalFix-Py EM from $0/33$ (untrained) to $12$--$17\%$, a learnability signal, not a production selector. Batched-serving integration and multi-file coverage are scoped as follow-up. 

---
# Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation 

**Authors**: Ran Zhang, Steffen Eger, Arda Tezcan, Wei Zhao, Simone Paolo Ponzetto, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2604.18169)  

**Abstract**: Large language models (LLMs) are increasingly used for creative tasks such as literary translation. Yet translational creativity remains underexplored and is rarely evaluated at scale, while source-text comprehension is typically studied in isolation, despite the fact that, in professional translation, comprehension and creativity are tightly intertwined. We address these gaps with a paired-task framework applied to literary excerpts from 11 books. Task 1 assesses source-text comprehension, and Task 2 evaluates translational creativity through Units of Creative Potential (UCPs), such as metaphors and wordplay. Using a scalable evaluation setup that combines expert human annotations with UCP-based automatic scoring, we benchmark 23 models and four creativity-oriented prompts. Our findings show that strong comprehension does not translate into human-level creativity: models often produce literal or contextually inappropriate renderings, with particularly large gaps for the more distant English-Chinese language pair. Creativity-oriented prompts yield only modest gains, and only one model, Mistral-Large, comes close to human-level creativity (0.167 vs. 0.246). Across all model-prompt combinations, only three exceed a creativity score of 0.1, while the rest remain at or near zero. 

---
# MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge 

**Authors**: Sua Lee, Sanghee Park, Jinbae Im  

**Link**: [PDF](https://arxiv.org/pdf/2604.18164)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been increasingly used as automatic evaluators-a paradigm known as MLLM-as-a-Judge. However, their reliability and vulnerabilities to biases remain underexplored. We find that many MLLM judges fail to reliably integrate key visual or textual cues, yielding unreliable evaluations when evidence is missing or mismatched, and exhibiting instability under semantically irrelevant perturbations. To address this, we systematically define Compositional Bias in MLLM-as-a-Judge systems and introduce MM-JudgeBias, a benchmark for evaluating it. MM-JudgeBias introduces controlled perturbations across Query, Image, and Response, and evaluates model behavior via two complementary metrics: Bias-Deviation (BD) for sensitivity and Bias-Conformity (BC) for stability. Our dataset of over 1,800 curated and refined multimodal samples, drawn from 29 source benchmarks, enables a fine-grained diagnosis of nine bias types across diverse tasks and domains. Experiments on 26 state-of-the-art MLLMs reveal systematic modality neglect and asymmetric evaluation tendencies, underscoring the need for more reliable judges. 

---
# Does "Do Differentiable Simulators Give Better Policy Gradients?'' Give Better Policy Gradients? 

**Authors**: Ku Onoda, Paavo Parmas, Manato Yaguchi, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2604.18161)  

**Abstract**: In policy gradient reinforcement learning, access to a differentiable model enables 1st-order gradient estimation that accelerates learning compared to relying solely on derivative-free 0th-order estimators. However, discontinuous dynamics cause bias and undermine the effectiveness of 1st-order estimators. Prior work addressed this bias by constructing a confidence interval around the REINFORCE 0th-order gradient estimator and using these bounds to detect discontinuities. However, the REINFORCE estimator is notoriously noisy, and we find that this method requires task-specific hyperparameter tuning and has low sample efficiency. This paper asks whether such bias is the primary obstacle and what minimal fixes suffice. First, we re-examine standard discontinuous settings from prior work and introduce DDCG, a lightweight test that switches estimators in nonsmooth regions; with a single hyperparameter, DDCG achieves robust performance and remains reliable with small samples. Second, on differentiable robotics control tasks, we present IVW-H, a per-step inverse-variance implementation that stabilizes variance without explicit discontinuity detection and yields strong results. Together, these findings indicate that while estimator switching improves robustness in controlled studies, careful variance control often dominates in practical deployments. 

---
# Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations 

**Authors**: Yunjia Xi, Menghui Zhu, Jianghao Lin, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18146)  

**Abstract**: Recently, large language models (LLMs) have advanced recommendation systems (RSs), and recent works have begun to explore how to integrate LLMs into industrial RSs. While most approaches deploy LLMs offline to generate and pre-cache augmented representations for RSs, high-dimensional representations from LLMs introduce substantial storage and computational costs. Thus, it is crucial to compress LLM representations effectively. However, we identify a counterintuitive phenomenon during representation compression: Mid-layer Representation Advantage (MRA), where representations from middle layers of LLMs outperform those from final layers in recommendation tasks. This degraded final layer renders existing compression methods, which typically compress on the final layer, suboptimal. We interpret this based on modularity theory that LLMs develop spontaneous internal functional modularity and force the final layer to specialize in the proxy training task. Thus, we propose \underline{M}odul\underline{a}r \underline{R}epresentation \underline{C}ompression (MARC) to explicitly control the modularity of LLMs. First, Modular Adjustment explicitly introduces compression and task adaptation modules, enabling the LLM to operate strictly as a representation-learning module. Next, to ground each module to its specific task, Modular Task Decoupling uses information constraints and different network structures to decouple tasks. Extensive experiments validate that MARC addresses MRA and produces efficient representations. Notably, MARC achieved a 2.82% eCPM lift in an online A/B test within a large-scale commercial search advertising scenario. 

---
# Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework 

**Authors**: Cong Huy Nguyen, Son Dinh Nguyen, Guanlin Li, Tuan Dung Nguyen, Aditya Narayan Sankaran, Mai Huy Thong, Thanh Trung Nguyen, Mai Hong Son, Reza Farahbakhsh, Phi Le Nguyen, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18145)  

**Abstract**: Automated medical report generation for 3D PET/CT imaging is fundamentally challenged by the high-dimensional nature of volumetric data and a critical scarcity of annotated datasets, particularly for low-resource languages. Current black-box methods map whole volumes to reports, ignoring the clinical workflow of analyzing localized Regions of Interest (RoIs) to derive diagnostic conclusions. In this paper, we bridge this gap by introducing VietPET-RoI, the first large-scale 3D PET/CT dataset with fine-grained RoI annotation for a low-resource language, comprising 600 PET/CT samples and 1,960 manually annotated RoIs, paired with corresponding clinical reports. Furthermore, to demonstrate the utility of this dataset, we propose HiRRA, a novel framework that mimics the professional radiologist diagnostic workflow by employing graph-based relational modules to capture dependencies between RoI attributes. This approach shifts from global pattern matching toward localized clinical findings. Additionally, we introduce new clinical evaluation metrics, namely RoI Coverage and RoI Quality Index, that measure both RoI localization accuracy and attribute description fidelity using LLM-based extraction. Extensive evaluation demonstrates that our framework achieves SOTA performance, surpassing existing models by 19.7% in BLEU and 4.7% in ROUGE-L, while achieving a remarkable 45.8% improvement in clinical metrics, indicating enhanced clinical reliability and reduced hallucination. Our code and dataset are available on GitHub. 

---
# AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization 

**Authors**: Kosuke Matsushima, Yasuyuki Okoshi, Masato Motomura, Daichi Fujiki  

**Link**: [PDF](https://arxiv.org/pdf/2604.18137)  

**Abstract**: Processing-in-Memory (PIM) architectures offer a promising solution to the memory bottlenecks in data-intensive machine learning, yet often overlook the growing challenge of activation memory footprint. Conventional PIM approaches struggle with massive KV cache sizes generated in long-context scenarios by Transformer-based models, frequently exceeding PIM's limited memory capacity, while techniques like sparse attention can conflict with PIM's need for data locality. Existing PIM approaches and quantization methods are often insufficient or poorly suited for leveraging the unique characteristics of activations. This work identifies an opportunity for PIM-specialized activation quantization to enhance bandwidth and compute efficiency.
We explore clustering-based vector quantization approaches, which align well with activation characteristics and PIM's internal bandwidth capabilities. Building on this, we introduce AQPIM, a novel PIM-aware activation quantization framework based on Product Quantization (PQ), optimizing it for modern Large Language Models (LLMs). By performing quantization directly within memory, AQPIM leverages PIM's high internal bandwidth and enables direct computation on compressed data, significantly reducing both memory footprint and computational overhead for attention computation. AQPIM addresses PQ's accuracy challenges by introducing several algorithmic optimizations. Evaluations demonstrate that AQPIM achieves significant performance improvements, drastically reducing of GPU-CPU communication that can account for 90$\sim$98.5\% of decoding latency, together with 3.4$\times$ speedup over a SOTA PIM approach. 

---
# Soft Label Pruning and Quantization for Large-Scale Dataset Distillation 

**Authors**: Xiao Lingao, Yang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.18135)  

**Abstract**: Large-scale dataset distillation requires storing auxiliary soft labels that can be 30-40x larger on ImageNet-1K and 200x larger on ImageNet-21K than the condensed images, undermining the goal of dataset compression. We identify two fundamental issues necessitating such extensive labels: (1) insufficient image diversity, where high within-class similarity in synthetic images requires extensive augmentation, and (2) insufficient supervision diversity, where limited variety in supervisory signals during training leads to performance degradation at high compression rates. To address these challenges, we propose Label Pruning and Quantization for Large-scale Distillation (LPQLD). We enhance image diversity via class-wise batching and batch-normalization supervision during synthesis. For supervision diversity, we introduce Label Pruning with Dynamic Knowledge Reuse to improve label-per-augmentation diversity, and Label Quantization with Calibrated Student-Teacher Alignment to improve augmentation-per-image diversity. Our approach reduces soft label storage by 78x on ImageNet-1K and 500x on ImageNet-21K while improving accuracy by up to 7.2% and 2.8%, respectively. Extensive experiments validate the superiority of LPQLD across different network architectures and dataset distillation methods. Code is available at this https URL. 

---
# Depth Registers Unlock W4A4 on SwiGLU: A Reader/Generator Decomposition 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18128)  

**Abstract**: We study post-training W4A4 quantization in a controlled 300M-parameter SwiGLU decoder-only language model trained on 5B tokens of FineWeb-Edu, and ask which input-activation sites dominate the error. Naive round-to-nearest W4A4 collapses validation perplexity from FP16 23.6 to 1727. A simple residual-axis training-time intervention -- Depth Registers with a register-magnitude hinge loss (DR+sink) -- reduces this to 119 (about 14x) at matched FP16 PPL and matched zero-shot capacity, and composes with SmoothQuant to 39.9 PPL. The residual ~2 PPL gap to FP16 is the diagnostic core. We decompose W4A4 damage by input-activation site: the five trainable linears in a SwiGLU block split into residual-axis readers (qkv, w1, w3) and block-internal generators (o_proj, w2). Elementary norm arguments show residual-axis magnitude control bounds readers tightly but leaves w2's bilinear input bounded only by the trivial product of factor bounds; empirically, DR+sink collapses reader kurtosis while leaving generators essentially unchanged, and the reader-rescued W4A4 residue is flat at ~0.28 nats across three matched checkpoints with Delta-remove(w2) dominating. We present DR+sink as a training-time probe rather than a deployment proposal: a post-hoc alternative (Per-Linear QuaRot) nearly matches it on the reader axis. Full QuaRot -- adding online per-head value Hadamard plus online w2-input rotation -- does not close the gap either, directly testing the prediction that orthogonal rotation cannot bound the bilinear SwiGLU tail. Claims are specific to our 300M, 5B-token, single-seed setting, and our experiments do not isolate the partition from the hinge. 

---
# TLoRA: Task-aware Low Rank Adaptation of Large Language Models 

**Authors**: Weicheng Lin, Yi Zhang, Jiawei Dang, Liang-Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18124)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning method for large language models, with its effectiveness largely influenced by the allocation of ranks and scaling factors, as well as initialization. Existing LoRA variants typically address only one of these factors, often at the cost of increased training complexity or reduced practical efficiency. In this work, we present Task-aware Low-Rank Adaptation (TLoRA), a unified framework that jointly optimizes initialization and resource allocation at the outset of training. TLoRA introduces a data-driven initialization strategy that aligns the LoRA $A$ matrix with task-relevant subspaces by performing singular value decomposition on the product of pre-trained weights and input activation covariance. After this, the $A$ matrix is frozen, and only the $B$ matrix is trained. Furthermore, TLoRA employs a sensitivity-based importance metric to adaptively allocate ranks and scaling factors across layers under a fixed parameter budget. We conduct extensive experiments that demonstrate TLoRA consistently performs excellently across various tasks, including natural language understanding, commonsense reasoning, math reasoning, code generation, and chat generation, while significantly reducing the number of trainable parameters. 

---
# The Collaboration Gap in Human-AI Work 

**Authors**: Varad Vishwarupe, Marina Jirotka, Nigel Shadbolt, Ivan Flechais  

**Link**: [PDF](https://arxiv.org/pdf/2604.18096)  

**Abstract**: LLMs are increasingly presented as collaborators in programming, design, writing, and analysis. Yet the practical experience of working with them often falls short of this promise. In many settings, users must diagnose misunderstandings, reconstruct missing assumptions, and repeatedly repair misaligned responses. This poster introduces a conceptual framework for understanding why such collaboration remains fragile. Drawing on a constructivist grounded theory analysis of 16 interviews with designers, developers, and applied AI practitioners working on LLM-enabled systems, and informed by literature on human-AI collaboration, we argue that stable collaboration depends not only on model capability but on the interaction's grounding conditions. We distinguish three recurrent structures of human-AI work: one-shot assistance, weak collaboration with asymmetric repair, and grounded collaboration. We propose that collaboration breaks down when the appearance of partnership outpaces the grounding capacity of the interaction and contribute a framework for discussing grounding, repair, and interaction structure in LLM-enabled work. 

---
# Autonomous Unmanned Aircraft Systems for Enhanced Search and Rescue of Drowning Swimmers: Image-Based Localization and Mission Simulation 

**Authors**: Sascha Emanuel Zell, Toni Schneidereit, Armin Fügenschuh, Michael Breuß  

**Link**: [PDF](https://arxiv.org/pdf/2604.18088)  

**Abstract**: Drowning is an omnipresent risk associated with any activity on or in the water, and rescuing a drowning person is particularly challenging because of the time pressure, making a short response time important. Further complicating water rescue are unsupervised and extensive swimming areas, precise localization of the target, and the transport of rescue personnel. Technical innovations can provide a remedy: We propose an Unmanned Aircraft System (UAS), also known as a drone-in-a-box system, consisting of a fleet of Unmanned Aerial Vehicles (UAVs) allocated to purpose-built hangars near swimming areas. In an emergency, the UAS can be deployed in addition to Standard Rescue Operation (SRO) equipment to locate the distressed person early by performing a fully automated Search and Rescue (S&R) operation and dropping a flotation device. In this paper, we address automatically locating distressed swimmers using the image-based object detection architecture You Only Look Once (YOLO). We present a dataset created for this application and outline the training process. We evaluate the performance of YOLO versions 3, 5, and 8 and architecture sizes (nano, extra-large) using Mean Average Precision (mAP) metrics mAP@.5 and mAP@.5:.95. Furthermore, we present two Discrete-Event Simulation (DES) approaches to simulate response times of SRO and UAS-based water rescue. This enables estimation of time savings relative to SRO when selecting the UAS configuration (type, number, and location of UAVs and hangars). Computational experiments for a test area in the Lusatian Lake District, Germany, show that UAS assistance shortens response time. Even a small UAS with two hangars, each containing one UAV, reduces response time by a factor of five compared to SRO. 

---
# Mix and Match: Context Pairing for Scalable Topic-Controlled Educational Summarisation 

**Authors**: Nathikan Yodthapa, Thanapong Intharah, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2604.18087)  

**Abstract**: Topic-controlled summarisation enables users to generate summaries focused on specific aspects of source documents. This paper investigates a data augmentation strategy for training small language models (sLMs) to perform topic-controlled summarisation. We propose a pairwise data augmentation method that combines contexts from different documents to create contrastive training examples, enabling models to learn the relationship between topics and summaries more effectively. Using the SciTLDR dataset enriched with Wikipedia-derived topics, we systematically evaluate how augmentation scale affects model performance. Results show consistent improvements in win rate and semantic alignment as the augmentation scale increases, while the amount of real training data remains fixed. Consequently, a T5-base model trained with our augmentation approach achieves competitive performance relative to larger models, despite using significantly fewer parameters and substantially fewer real training examples. 

---
# Implicit neural representations as a coordinate-based framework for continuous environmental field reconstruction from sparse ecological observations 

**Authors**: Agnieszka Pregowska, Hazem M. Kalaji  

**Link**: [PDF](https://arxiv.org/pdf/2604.18083)  

**Abstract**: Reconstructing continuous environmental fields from sparse and irregular observations remains a central challenge in environmental modelling and biodiversity informatics. Many ecological datasets are heterogeneous in space and time, making grid-based approaches difficult to scale or generalise across domains. Here, we evaluate implicit neural representations (INRs) as a coordinate-based modelling framework for learning continuous spatial and spatio-temporal fields directly from coordinate inputs. We analyse their behaviour across three representative modelling scenarios: species distribution reconstruction, phenological dynamics, and morphological segmentation derived from open biodiversity data. Beyond predictive performance, we examine interpolation behaviour, spatial coherence, and computational characteristics relevant for environmental modelling workflows, including scalability, resolution-independent querying, and architectural inductive bias. Results show that neural fields provide stable continuous representations with predictable computational cost, complementing classical smoothers and tree-based approaches. These findings position coordinate-based neural fields as a flexible representation layer that can be integrated into environmental modelling pipelines and exploratory analysis frameworks for large, irregularly sampled datasets. 

---
# Class-specific diffusion models improve military object detection in a low-data domain 

**Authors**: Ella P. Fokkinga, Jan Erik van Woerden, Thijs A. Eker, Sebastiaan P. Snel, Elfi I.S. Hofmeijer, Klamer Schutte, Friso G. Heslinga  

**Link**: [PDF](https://arxiv.org/pdf/2604.18076)  

**Abstract**: Diffusion-based image synthesis has emerged as a promising source of synthetic training data for AI-based object detection and classification. In this work, we investigate whether images generated with diffusion can improve military vehicle detection under low-data conditions. We fine-tuned the text-to-image diffusion model FLUX.1 [dev] using LoRA with only 8 or 24 real images per class across 15 vehicle categories, resulting in class-specific diffusion models, which were used to generate new samples from automatically generated text prompts. The same real images were used to fine-tune the RF-DETR detector for a 15-class object detection task. Synthetic datasets generated by the diffusion models were then used to further improve detector performance. Importantly, no additional real data was required, as the generative models leveraged the same limited training samples. FLUX-generated images improved detection performance, particularly in the low-data regime (up to +8.0% mAP$_{50}$ with 8 real samples). To address the limited geometric control of text prompt-based diffusion, we additionally generated structurally guided synthetic data using ControlNet with Canny edge-map conditioning, yielding a FLUX-ControlNet (FLUX-CN) dataset with explicit control over viewpoint and pose. Structural guidance further enhanced performance when data is scarce (+4.1% mAP$_{50}$ with 8 real samples), but no additional benefit was observed when more real data is available. This study demonstrates that object-specific diffusion models are effective for improving military object detection in a low-data domain, and that structural guidance is most beneficial when real data is highly limited. These results highlight generative image data as an alternative to traditional simulation pipelines for the training of military AI systems. 

---
# ExAI5G: A Logic-Based Explainable AI Framework for Intrusion Detection in 5G Networks 

**Authors**: Saeid Sheikhi, Panos Kostakos, Lauri Loven  

**Link**: [PDF](https://arxiv.org/pdf/2604.18052)  

**Abstract**: Intrusion detection systems (IDSs) for 5G networks must handle complex, high-volume traffic. Although opaque "black-box" models can achieve high accuracy, their lack of transparency hinders trust and effective operational response. We propose ExAI5G, a framework that prioritizes interpretability by integrating a Transformer-based deep learning IDS with logic-based explainable AI (XAI) techniques. The framework uses Integrated Gradients to attribute feature importance and extracts a surrogate decision tree to derive logical rules. We introduce a novel evaluation methodology for LLM-generated explanations, using a powerful evaluator LLM to assess actionability and measuring their semantic similarity and faithfulness. On a 5G IoT intrusion dataset, our system achieves 99.9\% accuracy and a 0.854 macro F1-score, demonstrating strong performance. More importantly, we extract 16 logical rules with 99.7\% fidelity, making the model's reasoning transparent. The evaluation demonstrates that modern LLMs can generate explanations that are both faithful and actionable, indicating that it is possible to build a trustworthy and effective IDS without compromising performance for the sake of marginal gains from an opaque model. 

---
# First, Do No Harm (With LLMs): Mitigating Racial Bias via Agentic Workflows 

**Authors**: Sihao Xing, Zaur Gouliev  

**Link**: [PDF](https://arxiv.org/pdf/2604.18038)  

**Abstract**: Large language models (LLMs) are increasingly used in clinical settings, raising concerns about racial bias in both generated medical text and clinical reasoning. Existing studies have identified bias in medical LLMs, but many focus on single models and give less attention to mitigation. This study uses the EU AI Act as a governance lens to evaluate five widely used LLMs across two tasks, namely synthetic patient-case generation and differential diagnosis ranking. Using race-stratified epidemiological distributions in the United States and expert differential diagnosis lists as benchmarks, we apply structured prompt templates and a two-part evaluation design to examine implicit and explicit racial bias. All models deviated from observed racial distributions in the synthetic case generation task, with GPT-4.1 showing the smallest overall deviation. In the differential diagnosis task, DeepSeek V3 produced the strongest overall results across the reported metrics. When embedded in an agentic workflow, DeepSeek V3 showed an improvement of 0.0348 in mean p-value, 0.1166 in median p-value, and 0.0949 in mean difference relative to the standalone model, although improvement was not uniform across every metric. These findings support multi-metric bias evaluation for AI systems used in medical settings and suggest that retrieval-based agentic workflows may reduce some forms of explicit bias in benchmarked diagnostic tasks. Detailed prompt templates, experimental datasets, and code pipelines are available on our GitHub. 

---
# RASP-Tuner: Retrieval-Augmented Soft Prompts for Context-Aware Black-Box Optimization in Non-Stationary Environments 

**Authors**: Enze Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18026)  

**Abstract**: Many deployed systems expose black-box objectives whose minimizing configuration shifts with an externally observed context. When contexts revisit a small set of latent regimes, an optimizer that discards history pays repeated adaptation cost; when each step must remain inexpensive, full Gaussian-process (GP) refits at high observation counts are difficult to sustain. We cast online tuning as context-conditioned regret minimization and present RASP-Tuner, which instantiates a decomposition motivated by first principles: (i) identify a regime proxy by retrieving similar past contexts; (ii) predict short-horizon loss with a mixture-of-experts surrogate whose input concatenates parameters, context, and a retrieved soft prompt; (iii) adapt chiefly in a low-dimensional prompt subspace, invoking full surrogate updates only when scalarized error or disagreement spikes. A RealErrorComposer maps heterogeneous streaming metrics to [0,1] via EMA-stabilized logistic scores, supplying a single differentiable training target. On nine synthetic non-stationary benchmarks, an adversarial-context sanity check, and three tabular real-world streams (Section on real-world experiments), RASP-Tuner improves or matches cumulative regret relative to our GP-UCB and CMA-ES implementations on seven of nine synthetic tasks under paired tests at horizon T=100, while recording 8-12 times lower wall-clock per step than sliding-window GP-UCB on identical hardware. Idealized analysis in a cluster-separated, strongly convex regime model (RA-GD) supplies sufficient conditions for bounded dynamic regret; the deployed pipeline violates several of these premises, and we articulate which gaps remain open. 

---
# Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation 

**Authors**: Nuo Chen, Yicheng Tong, Yuzhe Yang, Yufei He, Xueyi Zhang, Zou Qingyun, Qian Wang, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2604.18005)  

**Abstract**: Multi-agent systems (MAS) are increasingly used for open-ended idea generation, driven by the expectation that collective interaction will broaden the exploration diversity. However, when and why such collaboration truly expands the solution space remains unclear. We present a systematic empirical study of diversity in MAS-based ideation across three bottom-up levels: model intelligence, agent cognition, and system dynamics. At the model level, we identify a compute efficiency paradox, where stronger, highly aligned models yield diminishing marginal diversity despite higher per-sample quality. At the cognition level, authority-driven dynamics suppress semantic diversity compared to junior-dominated groups. At the system level, group-size scaling yields diminishing returns and dense communication topologies accelerate premature convergence. We characterize these outcomes as collective failures emerging from structural coupling, a process where interaction inadvertently contracts agent exploration and triggers diversity collapse. Our analysis shows that this collapse arises primarily from the interaction structure rather than inherent model insufficiency, highlighting the importance of preserving independence and disagreement when designing MAS for creative tasks. Our code is available at this https URL. 

---
# Latent Fourier Transform 

**Authors**: Mason Wang, Cheng-Zhi Anna Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17986)  

**Abstract**: We introduce the Latent Fourier Transform (LatentFT), a framework that provides novel frequency-domain controls for generative music models. LatentFT combines a diffusion autoencoder with a latent-space Fourier transform to separate musical patterns by timescale. By masking latents in the frequency domain during training, our method yields representations that can be manipulated coherently at inference. This allows us to generate musical variations and blends from reference examples while preserving characteristics at desired timescales, which are specified as frequencies in the latent space. LatentFT parallels the role of the equalizer in music production: while traditional equalizers operates on audible frequencies to shape timbre, LatentFT operates on latent-space frequencies to shape musical structure. Experiments and listening tests show that LatentFT improves condition adherence and quality compared to baselines. We also present a technique for hearing frequencies in the latent space in isolation, and show different musical attributes reside in different regions of the latent spectrum. Our results show how frequency-domain control in latent space provides an intuitive, continuous frequency axis for conditioning and blending, advancing us toward more interpretable and interactive generative music models. 

---
# RAVEN: Retrieval-Augmented Vulnerability Exploration Network for Memory Corruption Analysis in User Code and Binary Programs 

**Authors**: Parteek Jamwal, Minghao Shao, Boyuan Chen, Achyuta Muthuvelan, Asini Subanya, Boubacar Ballo, Kashish Satija, Mariam Shafey, Mohamed Mahmoud, Moncif Dahaji Bouffi, Pasindu Wickramasinghe, Siyona Goel, Yaakulya Sabbani, Hakim Hacid, Mthandazo Ndhlovu, Eleanna Kafeza, Sanjay Rawat, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2604.17948)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various cybersecurity tasks, including vulnerability classification, detection, and patching. However, their potential in automated vulnerability report documentation and analysis remains underexplored. We present RAVEN (Retrieval Augmented Vulnerability Exploration Network), a framework leveraging LLM agents and Retrieval Augmented Generation (RAG) to synthesize comprehensive vulnerability analysis reports. Given vulnerable source code, RAVEN generates reports following the Google Project Zero Root Cause Analysis template. The framework uses four modules: an Explorer agent for vulnerability identification, a RAG engine retrieving relevant knowledge from curated databases including Google Project Zero reports and CWE entries, an Analyst agent for impact and exploitation assessment, and a Reporter agent for structured report generation. To ensure quality, RAVEN includes a task specific LLM Judge evaluating reports across structural integrity, ground truth alignment, code reasoning quality, and remediation quality. We evaluate RAVEN on 105 vulnerable code samples covering 15 CWE types from the NIST-SARD dataset. Results show an average quality score of 54.21%, supporting the effectiveness of our approach for automated vulnerability documentation. 

---
# How Much Cache Does Reasoning Need? Depth-Cache Tradeoffs in KV-Compressed Transformers 

**Authors**: Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17935)  

**Abstract**: The key-value (KV) cache is the dominant memory bottleneck during Transformer inference, yet little is known theoretically about how aggressively it can be compressed before multi-step reasoning degrades. We study this through $k$-hop pointer chasing on $n$ tokens under a shared KV cache of size $s$, attention dimension $m$, $H$ heads, $p$-bit precision, and a locality-respecting cache controller (satisfied by all standard KV-compression methods). We give three results.
(1) Product depth lower bound (conjectured). We conjecture that any such Transformer ($n \geq 4k$, $s \leq \sqrt{n}/4$) requires depth $L = \Omega(\lceil k/s \rceil \cdot \lceil \log_2 n/(Hmp) \rceil)$, and isolate the sole remaining gap as a probabilistic step on the joint distribution of cache trace and pointer chain. Unconditionally, we prove a matching upper bound $L = O(\min(k, \lceil k/s \rceil \log s) \cdot \log n/(mp))$ via windowed pointer doubling, and a max-bound $L = \Omega(\max(\lceil k/s \rceil, \log n/(Hmp)))$. Closing the conjecture amounts to upgrading max to product.
(2) Bandwidth barrier. The product bound binds only when $Hmp \lesssim \log n$. Any lower bound provable via per-window distinguishability counting -- including reachability, bandwidth, and combinations -- cannot exceed $\lceil k/s \rceil$ once $Hmp \geq \log_2 n$. Breaking this requires lifting unconditional communication-complexity bounds for pointer chasing to Cache-Transformer depth.
(3) Adaptive vs oblivious error scaling. Under random cache over $T = \lceil \log_2 k \rceil$ doubling stages, oblivious caches give $\Pr[\mathcal{E}] \leq (s/(n-T))^T + 2T^3/n$ (exponential in $T$), while adaptive locality-respecting caches achieve $\Pr[\mathcal{E}] = s/n$ exactly, independent of $T$. The $\Omega((n/s)^{T-1})$ separation explains why heavy-hitter eviction empirically dominates random eviction for multi-hop reasoning. 

---
# Heterogeneity in Formal Linguistic Competence of Language Models: Is Data the Real Bottleneck? 

**Authors**: H S V N S Kowndinya Renduchintala, Sumit Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17930)  

**Abstract**: Large Language Models (LLMs) exhibit a puzzling disparity in their formal linguistic competence: while they learn some linguistic phenomena with near-perfect mastery, they often perform below chance on others, even after training on trillions of tokens. In this work, we investigate whether these failures stem from inherent architectural limitations or simply the scarcity of these specific grammatical constructions in web-scale corpora. We pre-train simple GPT-2 Small (124M) models on a 100M-token random sample of the FineWeb corpus and intervene by injecting a minimal amount (1%) of synthetic data targeting specific linguistic phenomena. We find that this targeted intervention substantially improves model performance in 8 out of the 9 worst-performing BLiMP paradigms - notably the accuracy on a specific paradigm, only_npi_scope, surges from 20.9% to 69.4%. Furthermore, we observe that these interventions generally preserve or slightly improve aggregate performance. However, while we also identify a resistant phenomenon, principle_A_c_command, whose performance remains below chance even after our data augmentation, our findings do serve as an optimistic existence proof that even small language models can substantially improve on those linguistic phenomena on which models typically perform poorly, provided the pre-training data contains sufficient exposure to them. This suggests that efforts towards human-scale language modeling may benefit greatly by focusing on data composition. The code to reproduce our results is open-sourced at this https URL. 

---
# HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment 

**Authors**: Zhanyu Liu, Qingguo Hu, Ante Wang, Chenqing Liu, Zhishang Xiang, Hui Li, Delai Qiu, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.17928)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has proven effective for training reasoning-oriented large language models, but existing methods largely assume high-resource settings with abundant training data. In low-resource scenarios, RLVR is prone to more severe entropy collapse, which substantially limits exploration and degrades reasoning performance. To address this issue, we propose Hybrid-domain Entropy dynamics ALignment (HEAL), a framework tailored for few-shot RLVR. HEAL first selectively incorporates high-value general-domain data to promote more diverse exploration. Then, we introduce Entropy Dynamics Alignment (EDA), a reward mechanism that aligns trajectory-level entropy dynamics between the target and general domains, capturing both entropy magnitude and fine-grained variation. Through this alignment, EDA not only further mitigates entropy collapse but also encourages the policy to acquire more diverse exploration behaviors from the general domain. Experiments across multiple domains show that HEAL consistently improves few-shot RLVR performance. Notably, using only 32 target-domain samples, HEAL matches or even surpasses full-shot RLVR trained with 1K target-domain samples. 

---
# Brain-Inspired Capture: Evidence-Driven Neuromimetic Perceptual Simulation for Visual Decoding 

**Authors**: Feixue Shao, Guangze Shi, Xueyu Liu, Yongfei Wu, Mingqiang Wei, Jianan Zhang, Jianbo Lu, Guiying Yan, Weihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17927)  

**Abstract**: Visual decoding of neurophysiological signals is a critical challenge for brain-computer interfaces (BCIs) and computational neuroscience. However, current approaches are often constrained by the systematic and stochastic gaps between neural and visual modalities, largely neglecting the intrinsic computational mechanisms of the Human Visual System (HVS). To address this, we propose Brain-Inspired Capture (BI-Cap), a neuromimetic perceptual simulation paradigm that aligns these modalities by emulating HVS processing. Specifically, we construct a neuromimetic pipeline comprising four biologically plausible dynamic and static transformations, coupled with Mutual Information (MI)-guided dynamic blur regulation to simulate adaptive visual processing. Furthermore, to mitigate the inherent non-stationarity of neural activity, we introduce an evidence-driven latent space representation. This formulation explicitly models uncertainty, thereby ensuring robust neural embeddings. Extensive evaluations on zero-shot brain-to-image retrieval across two public benchmarks demonstrate that BI-Cap substantially outperforms state-of-the-art methods, achieving relative gains of 9.2\% and 8.0\%, respectively. We have released the source code on GitHub through the link this https URL. 

---
# Prompting Foundation Models for Zero-Shot Ship Instance Segmentation in SAR Imagery 

**Authors**: Islam Mansour, Francescopaolo Sica, Michael Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.17920)  

**Abstract**: Synthetic Aperture Radar (SAR) plays a critical role in maritime surveillance, yet deep learning for SAR analysis is limited by the lack of pixel-level annotations. This paper explores how general-purpose vision foundation models can enable zero-shot ship instance segmentation in SAR imagery, eliminating the need for pixel-level supervision. A YOLOv11-based detector trained on open SAR datasets localizes ships via bounding boxes, which then prompt the Segment Anything Model 2 (SAM2) to produce instance masks without any mask annotations. Unlike prior SAM-based SAR approaches that rely on fine tuning or adapters, our method demonstrates that spatial constraints from a SAR-trained detector alone can effectively regularize foundation model predictions. This design partially mitigates the optical-SAR domain gap and enables downstream applications such as vessel classification, size estimation, and wake analysis. Experiments on the SSDD benchmark achieve a mean IoU of 0.637 (89% of a fully supervised baseline) with an overall ship detection rate of 89.2%, confirming a scalable, annotation-efficient pathway toward foundation-model-driven SAR image understanding. 

---
# Learning to Correct: Calibrated Reinforcement Learning for Multi-Attempt Chain-of-Thought 

**Authors**: Muhammed Emrullah Ildiz, Halil Alperen Gozeten, Ege Onur Taga, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2604.17912)  

**Abstract**: State-of-the-art reasoning models utilize long chain-of-thought (CoT) to solve increasingly complex problems using more test-time computation. In this work, we explore a long CoT setting where the model makes up to K successive attempts at solving a problem, in which each attempt is allowed to build on earlier ones after the model receives a hard verifier feedback. This motivates RL methods that can harness per-attempt rewards by carefully weighting individual attempts. We study optimizing the Verification@K reward (the model succeeds by the K-th attempt) and show that naively weighing the attempts by their pass/fail results in biased gradients. We introduce Calibrated Attempt-Level (CAL) GRPO by devising a weighing strategy to obtain unbiased gradients while maintaining small variance. Our theory reveals how incorporating per-attempt rewards influence the training and the eventual Verification@K performance. Experiments, baselines, and ablations on synthetic and real data corroborate our theory and the benefits of CAL-GRPO over vanilla GRPO as well as naive weighting. 

---
# Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring for Dense Passage Retrieval 

**Authors**: Junyoung Kim, Anton Korikov, Jiazhou Liang, Justin Cui, Yifan Simon Liu, Qianfeng Wen, Mark Zhao, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2604.17906)  

**Abstract**: While Large Language Models (LLMs) exhibit exceptional zero-shot relevance modeling, their high computational cost necessitates framing passage retrieval as a budget-constrained global optimization problem. Existing approaches passively rely on first-stage dense retrievers, which leads to two limitations: (1) failing to retrieve relevant passages in semantically distinct clusters, and (2) failing to propagate relevance signals to the broader corpus. To address these limitations, we propose Bayesian Active Learning with Gaussian Processes guided by LLM relevance scoring (BAGEL), a novel framework that propagates sparse LLM relevance signals across the embedding space to guide global exploration. BAGEL models the multimodal relevance distribution across the entire embedding space with a query-specific Gaussian Process (GP) based on LLM relevance scores. Subsequently, it iteratively selects passages for scoring by strategically balancing the exploitation of high-confidence regions with the exploration of uncertain areas. Extensive experiments across four benchmark datasets and two LLM backbones demonstrate that BAGEL effectively explores and captures complex relevance distributions and outperforms LLM reranking methods under the same LLM budget on all four datasets. 

---
# LoReC: Rethinking Large Language Models for Graph Data Analysis 

**Authors**: Hongyu Zhan, Qixin Wang, Yusen Tan, Haitao Yu, Jingbo Zhou, Shuai Chen, Jia Li, Xiao Tan, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17897)  

**Abstract**: The advent of Large Language Models (LLMs) has fundamentally reshaped the way we interact with graphs, giving rise to a new paradigm called GraphLLM. As revealed in recent studies, graph learning can benefit from LLMs. However, we observe limited benefits when we directly utilize LLMs to make predictions for graph-related tasks within GraphLLM paradigm, which even yields suboptimal results compared to conventional GNN-based approaches. Through in-depth analysis, we find this failure can be attributed to LLMs' limited capability for processing graph data and their tendency to overlook graph information. To address this issue, we propose LoReC (Look, Remember, and Contrast), a novel plug-and-play method for GraphLLM paradigm, which enhances LLM's understanding of graph data through three stages: (1) Look: redistributing attention to graph; (2) Remember: re-injecting graph information into the Feed-Forward Network (FFN); (3) Contrast: rectifying the vanilla logits produced in the decoding process. Extensive experiments demonstrate that LoReC brings notable improvements over current GraphLLM methods and outperforms GNN-based approaches across diverse datasets. The implementation is available at this https URL. 

---
# Can Explicit Physical Feasibility Benefit VLA Learning? An Empirical Study 

**Authors**: Yubai Wei, Chen Wu, Hashem Haghbayan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17896)  

**Abstract**: Vision-Language-Action (VLA) models map multimodal inputs directly to robot actions and are typically trained through large-scale imitation learning. While this paradigm has shown strong performance, prevailing VLA training procedures do not explicitly supervise hard physical constraints such as obstacle avoidance or kinematic feasibility. As a result, the geometric structure underlying physically feasible behavior must be inferred only implicitly from demonstrations. In this paper, we study whether introducing explicit feasibility supervision can provide effective structured guidance for VLA policies. We formulate a simple geometry-grounded feasibility objective and integrate it into the training stage of a diffusion-based VLA policy. To evaluate this idea systematically, we use obstacle-aware manipulation as a controlled probe of geometry-dependent physical feasibility. Empirical results show that augmenting VLA training with feasibility supervision improves both physical reliability and overall task performance, while also enhancing learning efficiency in the low-data regime. These findings indicate that explicit feasibility signals can effectively complement imitation-based VLA learning, highlighting their potential for developing more reliable VLA policies. 

---
# LEPO: \underline{L}atent R\underline{e}asoning \underline{P}olicy \underline{O}ptimization for Large Language~Models 

**Authors**: Yuyan Zhou, Jiarui Yu, Hande Dong, Zhezheng Hao, Hong Wang, Jianqing Zhang, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17892)  

**Abstract**: Recently, latent reasoning has been introduced into large language models (LLMs) to leverage rich information within a continuous space. However, without stochastic sampling, these methods inevitably collapse to deterministic inference, failing to discover diverse reasoning paths. To bridge the gap, we inject controllable stochasticity into latent reasoning via Gumbel-Softmax, restoring LLMs' exploratory capacity and enhancing their compatibility with Reinforcement Learning (RL). Building on this, we propose \textbf{\underline{L}}atent R\textbf{\underline{e}}asoning \textbf{\underline{P}}olicy \textbf{\underline{O}}ptimization~(\textbf{LEPO}), a novel framework that applies RL directly to continuous latent representations. Specifically, in rollout stage, LEPO maintains stochasticity to enable diverse trajectory sampling, while in optimization stage, LEPO constructs a unified gradient estimation for both latent representations and discrete tokens. Extensive experiments show that LEPO significantly outperforms existing RL methods for discrete and latent reasoning. 

---
# Latent Preference Modeling for Cross-Session Personalized Tool Calling 

**Authors**: Yejin Yoon, Minseo Kim, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17886)  

**Abstract**: Users often omit essential details in their requests to LLM-based agents, resulting in under-specified inputs for tool use. This poses a fundamental challenge for tool-augmented agents, as API execution typically requires complete arguments, highlighting the need for personalized tool calling. To study this problem, we introduce MPT, a benchmark comprising 265 multi-session dialogues that cover three challenges: Preference Recall, Preference Induction, and Preference Transfer. We also propose PRefine, a test-time memory-augmented method that represents user preferences as evolving hypotheses. Through a generate--verify--refine loop, it extracts reusable constraints from history and improves tool-calling accuracy while using only 1.24% of the tokens required by full-history prompting. These results indicate that robust personalization in agentic systems depends on memory that captures the reasons behind user choices, not just the choices themselves. 

---
# Latent Abstraction for Retrieval-Augmented Generation 

**Authors**: Ha Lan N.T, Minh-Anh Nguyen, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.17866)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose \textbf{LAnR} (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LAnR produces dense retrieval vectors from the hidden states of a designated \texttt{[PRED]} token and uses them to match against encoded document representations from the same model. Furthermore, LAnR adaptively decides when sufficient evidence has been retrieved using a lightweight MLP control head over those same hidden states, eliminating both the separate retriever and explicit token-level stopping reasoning. This design is motivated by our empirical observation that answer token entropy reliably signals retrieval sufficiency. Extensive experiments on six QA benchmarks spanning single-hop and multi-hop settings demonstrate that LAnR outperforms existing RAG methods, while achieving improved inference efficiency through reduced number of retrieval calls and tighter model integration. 

---
# Periodic Steady-State Control of a Handkerchief-Spinning Task Using a Parallel Anti-Parallelogram Tendon-driven Wrist 

**Authors**: Lei Liu, Haonan Zhang, Huahang Xu, Zefan Zhang, Lulu Chang, Lei Lv, Andrew Ross McIntosh, Kai Sun, Zhenshan Bing, Jiahong Dong, Fuchun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.17863)  

**Abstract**: Spinning flexible objects, exemplified by traditional Chinese handkerchief performances, demands periodic steady-state motions under nonlinear dynamics with frictional contacts and boundary constraints. To address these challenges, we first design an intuitive dexterous wrist based on a parallel anti-parallelogram tendon-driven structure, which achieves 90 degrees omnidirectional rotation with low inertia and decoupled roll-pitch sensing, and implement a high-low level hierarchical control scheme. We then develop a particle-spring model of the handkerchief for control-oriented abstraction and strategy evaluation. Hardware experiments validate this framework, achieving an unfolding ratio of approximately 99% and fingertip tracking error of RMSE = 2.88 mm in high-dynamic spinning. These results demonstrate that integrating control-oriented modeling with a task-tailored dexterous wrist enables robust rest-to-steady-state transitions and precise periodic manipulation of highly flexible objects. More visualizations: this https URL 

---
# On the Emergence of Syntax by Means of Local Interaction 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.17857)  

**Abstract**: Can syntactic processing emerge spontaneously from purely local interaction? We present a concrete instance on a minimal system: an 18,658-parameter two-dimensional neural cellular automaton (NCA), supervised by nothing more than a 1-bit boundary signal, is trained on the membership problem of an arithmetic-expression grammar. After training, its internal $L \times L$ grid spontaneously self-organizes into an ordered, spatially extended representation that we name Proto-CKY. This representation satisfies three operational criteria for syntactic processing: expressive power beyond the regular languages, structural generalization beyond the training distribution, and an internal organization quantitatively aligned with grammatical structure (Pearson $r \approx 0.71$). It emerges independently on four context-free grammars and regenerates spontaneously after perturbation. Proto-CKY is functionally aligned with the CKY algorithm but formally distinct from it: it is a physical prototype, a concrete instantiation of a mathematical ideal on a physical substrate, and the systematic distance between the two carries information about the substrate itself. 

---
# AI Approach for MRI-only Full-Spine Vertebral Segmentation and 3D Reconstruction in Paediatric Scoliosis 

**Authors**: Nathasha Naranpanawa, Maree T. Izatt, Robert D. Labrom, Geoffrey N. Askin, J. Paige Little  

**Link**: [PDF](https://arxiv.org/pdf/2604.17846)  

**Abstract**: MRI is preferred over CT in paediatric imaging because it avoids ionising radiation, but its use in spine deformity assessment is largely limited by the lack of automated, high-resolution 3D bony reconstruction, which continues to rely on CT. MRI-based 3D reconstruction remains impractical due to manual workflows and the scarcity of labelled full-spine datasets. This study introduces an AI framework that enables fully automated thoracolumbar spine (T1-L5) segmentation and 3D reconstruction from MRI alone. Historical low-dose CT scans from adolescent idiopathic scoliosis (AIS) patients were converted into MRI-like images using a GAN and combined with existing labelled thoracic MRI data to train a U-Net-based model. The resulting algorithm accurately generated continuous thoracolumbar 3D reconstructions, improved segmentation accuracy (88% Dice score), and reduced processing time from approximately 1 hour to under one minute, while preserving AIS-specific deformity features. This approach enables radiation-free 3D deformity assessment from MRI, supporting clinical evaluation, surgical planning, and navigation in paediatric spine care. 

---
# Learning from AVA: Early Lessons from a Curated and Trustworthy Generative AI for Policy and Development Research 

**Authors**: Nimisha Karnatak, Mohamad Chatila, Daniel Alejandro Pinzón Hernández, Reza Yazdanfar, Michelle Dugas, Renos Vakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.17843)  

**Abstract**: General-purpose LLMs pose misinformation risks for development and policy experts, lacking epistemic humility for verifiable outputs. We present AVA (AI + Verified Analysis), a GenAI platform built on a curated library of over 4,000 World Bank Reports with multilingual capabilities. AVA's multi-agent pipeline enables users to query and receive evidence-based syntheses. It operationalizes epistemic humility through two mechanisms: citation verifiability (tracing claims to sources) and reasoned abstention (declining unsupported queries with justification and redirection). We conducted an in-the-wild evaluation with over 2,200 individuals from heterogeneous organisations and roles in 116 countries, via log analysis, surveys, and 20 interviews. Difference-in-Differences estimates associate sustained engagement with 2.4-3.9 hours saved weekly. Qualitatively, participants used AVA as a specialized "evidence engine"; reasoned abstention clarified scope boundaries, and trust was calibrated through institutional provenance and page-anchored citations. We contribute design guidelines for specialized AI and articulate a vision for "ecosystem-aware" Humble AI. 

---
# A novel LSTM music generator based on the fractional time-frequency feature extraction 

**Authors**: Li Ya, Chen Wei, Li Xiulai, Yu Lei, Deng Xinyi, Chen Chaofan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17823)  

**Abstract**: In this paper, we propose a novel approach for generating music based on an artificial intelligence (AI) system. We analyze the features of music and use them to fit and predict the music. The fractional Fourier transform (FrFT) and the long short-term memory (LSTM) network are the foundations of our method. The FrFT method is used to extract the spectral features of a music piece, where the music signal is expressed on the time and frequency domains. The LSTM network is used to generate new music based on the extracted features, where we predict the music according to the hidden layer features and real-time inputs using GiantMIDI-Piano dataset. The results of our experiments show that our proposed system is capable of generating high-quality music that is comparable to human-generated music. 

---
# PDDL-Mind: Large Language Models are Capable on Belief Reasoning with Reliable State Tracking 

**Authors**: Wang Bill Zhu, Qiutong Tony Yi, Robin Jia, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2604.17819)  

**Abstract**: Large language models (LLMs) perform substantially below human level on existing theory-of-mind (ToM) benchmarks, even when augmented with chain-of-thought prompting or probabilistic belief updates. We argue that these failures primarily arise from unreliable implicit state tracking rather than limitations in high-level reasoning. We introduce PDDL-Mind, a neuro-symbolic framework that decouples environment state evolution from belief inference. By translating narrative descriptions into explicit states and actions expressed in Planning Domain Definition Language (PDDL), and by verifying action-induced state transitions against a predefined domain, PDDL-Mind provides LLMs with a logically consistent and explicit representation of world states for ToM tasks. Experiments on MMToM-QA, MuMA and FanToM show that PDDL-Mind achieves over 5% absolute accuracy gain over the best existing state-of-the-art method on ToM benchmark questions. 

---
# Do LLMs Need to See Everything? A Benchmark and Study of Failures in LLM-driven Smartphone Automation using Screentext vs. Screenshots 

**Authors**: Shiquan Zhang, Tianyi Zhang, Le Fang, Simon D'Alfonso, Hong Jia, Vassilis Kostakos  

**Link**: [PDF](https://arxiv.org/pdf/2604.17817)  

**Abstract**: With the rapid advancement of large language models (LLMs), mobile agents have emerged as promising tools for phone automation, simulating human interactions on screens to accomplish complex tasks. However, these agents often suffer from low accuracy, misinterpretation of user instructions, and failure on challenging tasks, with limited prior work examining why and where they fail. To address this, we introduce DailyDroid, a benchmark of 75 tasks in five scenarios across 25 Android apps, spanning three difficulty levels to mimic everyday smartphone use. We evaluate it using text-only and multimodal (text + screenshot) inputs on GPT-4o and o4-mini across 300 trials, revealing comparable performance with multimodal inputs yielding marginally higher success rates. Through in-depth failure analysis, we compile a handbook of common failures. Our findings reveal critical issues in UI accessibility, input modalities, and LLM/app design, offering implications for future mobile agents, applications, and UI development. 

---
# Understanding Secret Leakage Risks in Code LLMs: A Tokenization Perspective 

**Authors**: Meifang Chen, Zhe Yang, Huang Nianchen, Yizhan Huang, Yichen Li, Zihan Li, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17814)  

**Abstract**: Code secrets are sensitive assets for software developers, and their leakage poses significant cybersecurity risks. While the rapid development of AI code assistants powered by Code Large Language Models (CLLMs), CLLMs are shown to inadvertently leak such secrets due to a notorious memorization phenomenon. This study first reveals that Byte-Pair Encoding (BPE) tokenization leads to unexpected behavior of secret memorization, which we term as \textit{gibberish bias}. Specifically, we identified that some secrets are among the easiest for CLLMs to memorize. These secrets yield high character-level entropy, but low token-level entropy. Then, this paper supports the biased claim with numerical data. We identified that the roots of the bias are the token distribution shift between the CLLM training data and the secret data. We further discuss how gibberish bias manifests under the ``larger vocabulary'' trend. To conclude the paper, we discuss potential mitigation strategies and the broader implications on current tokenizer design. 

---
# Party Autonomy in Determining the Law Applicable to Non-contractual Obligations concerning Cross-Border Data Transfers 

**Authors**: Yuki Okamura, Ren Yatsunami, Kumiko Kameishi, Oliver Posani, Soma Araoka, Miho Ikeda, Makiko Aoyagi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17806)  

**Abstract**: (1)Cross-border data transfers have become a matter of daily occurrence against the backdrop of the development of cloud computing and artificial intelligence. Consequently, where a data leak gives rise to civil liability, the determination of that liability inevitably assumes an international dimension involving foreign elements. (2)As is starkly demonstrated by secret sharing technology in cloud computing, fragments of data may be presumed to be distributed across multiple jurisdictions on a global scale. This renders traditional private international law measures -- predicated on the identification of a physical location -- inadequate for the purposes of determining the applicable law, a difficulty that is particularly acute in relation to non-contractual obligations. (3)Bearing in mind the typical scenario encountered in practice -- in which a Data Subject brings a claim for damages against a SaaS (Software as a Service) provider, which in turn seeks recourse against an IaaS (Infrastructure as a Service) or PaaS (Platform as a Service) provider -- a characteristic feature of such cases is the concurrence of contractual and non-contractual obligations. Taking this feature into account, it is possible to determine the applicable law governing non-contractual obligations through party autonomy -- by aligning it with the law governing the contractual obligation as selected by the parties, an approach that may be termed private ordering. This serves to overcome the difficulties associated with the identification of a physical location and, at the same time, contributes to ensuring the foreseeability of the parties. 

---
# Ranking Abuse via Strategic Pairwise Data Perturbations 

**Authors**: Junyi Yao, Zihao Zheng, Jiayu Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.17805)  

**Abstract**: Pairwise ranking systems based on Maximum Likelihood Estimation (MLE), such as the Bradley-Terry model, are widely used to aggregate preferences from pairwise comparisons. However, their robustness under strategic data manipulation remains insufficiently understood.
In this paper, we study the vulnerability of MLE-based ranking systems to adversarial perturbations. We formulate the manipulation task as a constrained combinatorial optimization problem and propose an Adaptive Subset Selection Attack (ASSA) to efficiently identify high-impact perturbations.
Experimental results on both synthetic data and real-world election datasets show that MLE-based rankings exhibit a sharp phase-transition behavior: beyond a small perturbation budget, a limited number of strategic voters can significantly alter the global ranking. In particular, our method consistently outperforms random and greedy baselines under constrained budgets.
These findings reveal a fundamental sensitivity of MLE-based ranking mechanisms to structured perturbations and highlight the need for more robust aggregation methods in collective decision-making systems. 

---
# Bridging the Reasoning Gap in Vietnamese with Small Language Models via Test-Time Scaling 

**Authors**: Bui The Trung, Do Minh Duc, Nguyen Van Vinh, Bui Nguyen Quoc Trinh  

**Link**: [PDF](https://arxiv.org/pdf/2604.17794)  

**Abstract**: The democratization of ubiquitous AI hinges on deploying sophisticated reasoning capabilities on resource-constrained devices. However, Small Language Models (SLMs) often face a "reasoning gap", particularly in non-English languages like Vietnamese, where they struggle to maintain coherent chains of thought. This paper investigates Test-Time Scaling strategies for the Qwen3-1.7B architecture within the context of Vietnamese Elementary Mathematics. We introduce Vi-S1K, a high-fidelity reasoning dataset localized via a Gemini 2.5 Flash-Lite powered pipeline, and Vi-Elementary-Bench, a dual-resource benchmark for rigorous evaluation. Using an LLM-as-a-Judge protocol, we reveal that the base model possesses robust latent knowledge (Accuracy: 4.05/5.00) but suffers from a severe "formatting gap" in communication. Supervised Fine-Tuning (SFT) acts as a critical "reasoning unlocker", yielding a 77% improvement in Explanation Quality and bridging the gap between raw calculation and pedagogical coherence. Furthermore, our analysis of prompting strategies uncovers a significant trade-off: structured frameworks like ReAct impose a "cognitive tax" on the 1.7B parameter capacity, degrading performance relative to pure Chain-of-Thought (CoT) combined with Self-Consistency. These findings establish a deployment hierarchy for SLMs, demonstrating that SFT combined with simplified test-time scaling is superior to complex agentic workflows for edge-based reasoning. 

---
# DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization 

**Authors**: Haokun Lin, Xinle Jia, Haobo Xu, Bingchen Yao, Xianglong Guo, Yichen Wu, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.17789)  

**Abstract**: The MXFP4 microscaling format, which partitions tensors into blocks of 32 elements sharing an E8M0 scaling factor, has emerged as a promising substrate for efficient LLM inference, backed by native hardware support on NVIDIA Blackwell Tensor Cores. However, activation outliers pose a unique challenge under this format: a single outlier inflates the shared block scale, compressing the effective dynamic range of the remaining elements and causing significant quantization error. Existing rotation-based remedies, including randomized Hadamard and learnable rotations, are data-agnostic and therefore unable to specifically target the channels where outliers concentrate. We propose DuQuant++, which adapts the outlier-aware fine-grained rotation of DuQuant to the MXFP4 format by aligning the rotation block size with the microscaling group size (B{=}32). Because each MXFP4 group possesses an independent scaling factor, the cross-block variance issue that necessitates dual rotations and a zigzag permutation in the original DuQuant becomes irrelevant, enabling DuQuant++ to replace the entire pipeline with a single outlier-aware rotation, which halves the online rotation cost while simultaneously smoothing the weight distribution. Extensive experiments on the LLaMA-3 family under MXFP4 W4A4 quantization show that DuQuant++ consistently achieves state-of-the-art performance. Our code is available at this https URL. 

---
# AnchorRefine: Synergy-Manipulation Based on Trajectory Anchor and Residual Refinement for Vision-Language-Action Models 

**Authors**: Tingzheng Jia, Kan Guo, Lanping Qian, Yongli Hu, Daxin Tian, Guixian Qu, Chunmian Lin, Baocai Yin, Jiapu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17787)  

**Abstract**: Precision-critical manipulation requires both global trajectory organization and local execution correction, yet most vision-language-action (VLA) policies generate actions within a single unified space. This monolithic formulation forces macro-level transport and micro-level refinement to be optimized under the same objective, causing large motions to dominate learning while suppressing small but failure-critical corrective signals. In contrast, human manipulation is structured by global movement planning together with continuous local adjustment during execution. Motivated by this principle, we propose AnchorRefine, a hierarchical framework that factorizes VLA action modeling into trajectory anchor and residual refinement. The anchor planner predicts a coarse motion scaffold, while the refinement module corrects execution-level deviations to improve geometric and contact precision. We further introduce a decision-aware gripper refinement mechanism to better capture the discrete and boundary-sensitive nature of gripper control. Experiments on LIBERO, CALVIN, and real-robot tasks demonstrate that AnchorRefine consistently improves both regression-based and diffusion-based VLA backbones, yielding gains of up to 7.8% in simulation success rate and 18% in real-world success rate. 

---
# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens 

**Authors**: Seunghee Koh, Sunghyun Baek, Youngdong Kim, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17785)  

**Abstract**: Unlearning in large language models (LLMs) has emerged as a promising safeguard against adversarial behaviors. When the forgetting loss is applied uniformly without considering token-level semantic importance, model utility can be unnecessarily degraded. Recent studies have explored token-wise loss regularizers that prioritize informative tokens, but largely rely on ground-truth confidence or external linguistic parsers, which limits their ability to capture contextual information or the model's overall predictive state. Intuitively, function words like "the" primarily serve syntactic roles and are highly predictable with little ambiguity, but informative words admit multiple plausible alternatives with greater uncertainty. Based on this intuition, we propose Entropy-guided Token Weighting (ETW), a token-level unlearning regularizer that uses entropy of the predictive distribution as a proxy for token informativeness. We demonstrate that informative tokens tend to have higher entropy, whereas structural tokens tend to have lower entropy. This behavior enables ETW to achieve more effective unlearning while better preserving model utility than existing token-level approaches. 

---
# SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks 

**Authors**: Mohammadtaher Safarzadeh, Hitesh Laxmichand Patel, Afshin Orojlooyjadid, Graham Horwood, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.17771)  

**Abstract**: Large language models (LLMs) have achieved strong performance on natural language to SQL (NL2SQL) benchmarks, yet their reported accuracy may be inflated by contamination from benchmark queries or structurally similar patterns seen during training. We introduce SPENCE (Syntactic Probing and Evaluation of NL2SQL Contamination Effects), a controlled syntactic probing framework for detecting and quantifying such contamination. SPENCE systematically generates syntactic variants of test queries for four widely used NL2SQL datasets-Spider, SParC, CoSQL, and the newer BIRD benchmark. We use SPENCE to evaluate multiple high-capacity LLMs under execution-based scoring. For each model, we measure changes in execution accuracy across increasing levels of syntactic divergence and quantify rank sensitivity using Kendall's tau with bootstrap confidence intervals. By aligning these robustness trends with benchmark release dates, we observe a clear temporal gradient: older benchmarks such as Spider exhibit the strongest negative values and thus the highest likelihood of training leakage, whereas the more recent BIRD dataset shows minimal sensitivity and appears largely uncontaminated. Together, these findings highlight the importance of temporally contextualized, syntactic-probing evaluation for trustworthy NL2SQL benchmarking. 

---
# Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF 

**Authors**: Yuan Fang, Yiming Luo, Aimin Zhou, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17769)  

**Abstract**: Ensuring the safety of large language models (LLMs) requires robust red teaming, yet the systematic synthesis of high-quality toxic data remains under-explored. We propose Reverse Constitutional AI (R-CAI), a framework for automated and controllable adversarial data generation that moves beyond isolated jailbreak prompts. By inverting a harmless constitution into a constitution of toxicity and iteratively refining model outputs through a critique--revision pipeline, R-CAI enables scalable synthesis of multi-dimensional adversarial data without human annotation. Optimizing solely for toxicity-related rewards, however, can lead to reward hacking and degraded semantic coherence. To address this challenge, we introduce probability clamping within reinforcement learning from AI feedback, which stabilizes adversarial optimization while preserving adversarial intent. Experiments demonstrate that R-CAI generates diverse, high-quality toxic data and that probability clamping substantially improves semantic coherence (15%) without sacrificing adversarial strength. Overall, R-CAI provides a fully automated framework for red teaming data generation and systematic safety evaluation of aligned language models. 

---
# Community-Led AI Integration for Wildfire Risk Assessment: A Participatory AI Literacy and Explainability Integration (PALEI) Framework in Los Angeles, CA 

**Authors**: Sanaz Sadat Hosseini, Mona Azarbayjani, Mohammad Pourhomayoun, Hamed Tabkhi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17755)  

**Abstract**: Climate-driven wildfires are intensifying, particularly in urban regions such as Southern California. Yet, traditional fire risk communication tools often fail to gain public trust due to inaccessible design, non-transparent outputs, and limited contextual relevance. These challenges are especially critical in high-risk communities, where trust depends on how clearly and locally information is presented. Neighborhoods such as Pacific Palisades, Pasadena, and Altadena in Los Angeles exemplify these conditions. This study introduces a community-led approach for integrating AI into wildfire risk assessment using the Participatory AI Literacy and Explainability Integration (PALEI) framework. PALEI emphasizes early literacy building, value alignment, and participatory evaluation before deploying predictive models, prioritizing clarity, accessibility, and mutual learning between developers and residents. Early engagement findings show strong acceptance of visual, context-specific risk communication, positive fairness perceptions, and clear adoption interest, alongside privacy and data security concerns that influence trust. Participants emphasized localized imagery, accessible explanations, neighborhood-specific mitigation guidance, and transparent communication of uncertainty. The outcome is a mobile application co-designed with users and stakeholders, enabling residents to scan visible property features and receive interpretable fire risk scores with tailored recommendations. By embedding local context into design, the tool becomes an everyday resource for risk awareness and preparedness. This study argues that user experience is central to ethical and effective AI deployment and provides a replicable, literacy-first pathway for applying the PALEI framework to climate-related hazards. 

---
# MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models 

**Authors**: Suhyun Lee, Palakorn Achananuparp, Neemesh Yadav, Ee-Peng Lim, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17730)  

**Abstract**: Large language models (LLMs) are increasingly explored as scalable tools for mental health counseling, yet evaluating their safety remains challenging due to the interactional and context-dependent nature of clinical harm. Existing evaluation frameworks predominantly assess isolated responses using coarse-grained taxonomies or static datasets, limiting their ability to diagnose how harms emerge and accumulate over multi-turn counseling interactions. In this work, we introduce R-MHSafe, a role-aware mental health safety taxonomy that characterizes clinically significant harm in terms of the interactional roles an AI counselor adopts, including perpetrator, instigator, facilitator, or enabler, combined with clinically grounded harm categories. Then, we propose MHSafeEval, a closed-loop, agent-based evaluation framework that formulates safety assessment as trajectory-level discovery of harm through adversarial multi-turn interactions, guided by role-aware modeling. Using R-MHSafe and MHSafeEval, we conduct a large-scale evaluation across state-of-the-art LLMs. Our results reveal substantial role-dependent and cumulative safety failures that are systematically missed by existing static benchmarks, and show that our framework significantly improves failure-mode coverage and diagnostic granularity. 

---
# Voronoi-guided Bilateral 2D Gaussian Splatting for Arbitrary-Scale Hyperspectral Image Super-Resolution 

**Authors**: Jie Zhang, Jinkun You, Shi Chen, Yicong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.17727)  

**Abstract**: Most existing hyperspectral image super-resolution methods require modifications for different scales, limiting their flexibility in arbitrary-scale reconstruction. 2D Gaussian splatting provides a continuous representation that is compatible with arbitrary-scale super-resolution. Existing methods often rely on rasterization strategies, which may limit flexible spatial modeling. Extending them to hyperspectral image super-resolution remains challenging, as the task requires adaptive spatial reconstruction while preserving spectral fidelity. This paper proposes GaussianHSI, a Gaussian-Splatting-based framework for arbitrary-scale hyperspectral image super-resolution. We develop a Voronoi-Guided Bilateral 2D Gaussian Splatting for spatial reconstruction. After predicting a set of Gaussian functions to represent the input, it associates each target pixel with relevant Gaussian functions through Voronoi-guided selection. The target pixel is then reconstructed by aggregating the selected Gaussian functions with reference-aware bilateral weighting, which considers both geometric relevance and consistency with low-resolution features. We further introduce a Spectral Detail Enhancement module to improve spectral reconstruction. Extensive experiments on benchmark datasets demonstrate the effectiveness of GaussianHSI over state-of-the-art methods for arbitrary-scale hyperspectral image super-resolution. 

---
# RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models 

**Authors**: Arya Hadizadeh Moghaddam, Drew Ross, Mohsen Nayebi Kerdabadi, Dongjie Wang, Zijun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17725)  

**Abstract**: Large Language Models (LLMs) have shown strong promise for mining Electronic Health Records (EHRs) by reasoning over longitudinal clinical information to capture context-rich patient trajectories. However, leveraging LLMs for structured EHRs (e.g., standardized diagnosis and medication codes) presents two key challenges. First, translating time-stamped EHR sequences into plain text can obscure both temporal structure and code identities, weakening the ability to capture code co-occurrence and longitudinal regularities. Second, unlike cohort-trained predictive models that learn a shared, task-aligned representation space across patients, LLMs are often applied in a case-isolated inference setting where each patient is processed independently without leveraging population-level patterns. To address these challenges, we introduce RePrompT, a time-aware LLM framework that integrates structured EHR encoders through prompt tuning, without modifying underlying architectures. Specifically, RePrompT recurrently incorporates latent states from prior visits to preserve longitudinal information, and injects population-level information through trainable prompt tokens derived from a cohort-trained, task-aligned EHR encoder. Experiments on MIMIC-III and MIMIC-IV demonstrate that RePrompT consistently outperforms both EHR-based and LLM-based baselines across multiple clinical prediction tasks. 

---
# GeGS-PCR: Effective and Robust 3D Point Cloud Registration with Two-Stage Color-Enhanced Geometric-3DGS Fusion 

**Authors**: Jiayi Tian, Haiduo Huang, Tian Xia, Wenzhe Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.17721)  

**Abstract**: We address the challenge of point cloud registration using color information, where traditional methods relying solely on geometric features often struggle in low-overlap and incomplete scenarios. To overcome these limitations, we propose GeGS-PCR, a novel two-stage method that combines geometric, color, and Gaussian information for robust registration. Our approach incorporates a dedicated color encoder that enhances color features by extracting multi-level geometric and color data from the original point cloud. We introduce the \textbf{Ge}ometric-3D\textbf{GS} module, which encodes the local neighborhood information of colored superpoints to ensure a globally invariant geometric-color context. Leveraging LORA optimization, we maintain high performance while preserving the expressiveness of 3DGS. Additionally, fast differentiable rendering is utilized to refine the registration process, leading to improved convergence. To further enhance performance, we propose a joint photometric loss that exploits both geometric and color features. This enables strong performance in challenging conditions with extremely low point cloud overlap. We validate our method by colorizing the Kitti dataset as ColorKitti and testing on both Color3DMatch and Color3DLoMatch datasets. Our method achieves state-of-the-art performance with \textit{Registration Recall} at 99.9\%, \textit{Relative Rotation Error} as low as 0.013, and \textit{Relative Translation Error} as low as 0.024, improving precision by at least a factor of 2. 

---
# Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17716)  

**Abstract**: The validity screen (Cacioli, 2026d, 2026e) classifies LLM confidence signals as Valid, Indeterminate, or Invalid. We test whether these classifications predict selective prediction performance. Twenty frontier LLMs from seven families were evaluated on 524 items across six cognitive tracks. Valid models show mean Type 2 AUROC = .624 (SD = .048). Invalid models show mean AUROC = .357 (SD = .231). Cohen's d = 2.81, p = .002. The tiers order monotonically: Invalid (.357) < Indeterminate (.554) < Valid (.624). Split-half cross-validation yields median d = 1.77, P(d > 0) = 1.0 across 1,000 splits. The three-tier classification accounts for 47% of the variance in AUROC. DeepSeek-R1 drops from 85.3% accuracy at full coverage to 11.3% at 10% coverage. The screen predicts the criterion. For selective prediction, the screen matters. 

---
# Screen Before You Interpret: A Portable Validity Protocol for Benchmark-Based LLM Confidence Signals 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17714)  

**Abstract**: LLM confidence signals are used for abstention, routing, and safety-critical decisions. No standard practice exists for checking whether a confidence signal carries item-level information before building on it. We transfer the validity screening principle from clinical personality assessment (PAI, MMPI-3) as a portable protocol for benchmark-based LLM confidence data. The protocol specifies three core indices (L, Fp, RBS), a structural indicator (TRIN), and an item-sensitivity statistic, computed from a single 2x2 contingency table. A three-tier classification system (Invalid, Indeterminate, Valid) draws on four clinical traditions. Validated on 20 frontier LLMs across 524 items, four models are classified Invalid, two Indeterminate. Valid-profile models show mean r = .18 (15/16 significant). Invalid-profile models show mean r = -.20 (d = 2.48). Cross-benchmark validation on 18 models using MMLU with verbalized confidence and on external data from Yang et al. (2024) confirms the screen transfers across benchmarks and probe formats. All data and code: this https URL 

---
# Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17707)  

**Abstract**: Clinical personality assessment screens response validity before interpreting substantive scales. LLM evaluation does not. We apply the validity scaling framework from the PAI and MMPI-3 to metacognitive probe data from 20 frontier models across 524 items. Six validity indices are operationalised: L (maintaining confidence on errors), K (betting on errors), F (withdrawing consensus-endorsed items), Fp (withdrawing correct answers), RBS (inverted monitoring), and TRIN (fixed responding). A tiered classification system identifies four models as construct-level invalid and two as elevated. Valid-profile models produce item-sensitive confidence (mean r = .18, 14 of 16 significant). Invalid-profile models do not (mean r = -.20, d = 2.17, p = .001). Chain-of-thought training produces two opposite response distortions. Two latent dimensions account for 94.6% of index variance. Companion papers extract a portable screening protocol (Cacioli, 2026e) and validate it against selective prediction (Cacioli, 2026f). All data and code: this https URL 

---
# WISV: Wireless-Informed Semantic Verification for Distributed Speculative Decoding in Device-Edge LLM Inference 

**Authors**: Zixuan Liu, Zhiyong Chen, Nan Xue, Shengkang Chen, Jiangchao Yao, Meixia Tao, Wenjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17701)  

**Abstract**: While distributed device-edge speculative decoding enhances resource utilization across heterogeneous nodes, its performance is often bottlenecked by conventional token-level verification strategies. Such rigid alignment leads to excessive rejections, significantly diminishing the accepted sequence length and increasing interaction rounds under fluctuating wireless conditions. In this paper, we propose WISV (Wireless-Informed Semantic Verification), a novel distributed speculative decoding framework that goes beyond strict token-level matching via a channel-aware semantic acceptance policy. WISV integrates a lightweight decision head into the edge-side target LLM to dynamically evaluate speculative tokens by synthesizing high-dimensional hidden representations with instantaneous channel state information (CSI). To optimize the trade-off between verification fidelity and communication overhead, we further design two tailored communication protocols: full-hidden upload and mismatch-first selective-hidden upload. Extensive simulations using a 1B drafter and an 8B target model demonstrate that WISV achieves up to a 60.8% increase in accepted length, a 37.3% reduction in interaction rounds, and a 31.4% improvement in end-to-end latency compared to vanilla speculative decoding across tested settings, while maintaining a negligible task accuracy drop (<1%). Finally, we validate WISV on a hardware testbed comprising an NVIDIA Jetson AGX Orin and an A40-equipped server, confirming its real-world efficacy in accelerating edge-deployed LLM inference. 

---
# CAPO: Counterfactual Credit Assignment in Sequential Cooperative Teams 

**Authors**: Shripad Deshmukh, Jayakumar Subramanian, Raghavendra Addanki, Nikos Vlassis  

**Link**: [PDF](https://arxiv.org/pdf/2604.17693)  

**Abstract**: In cooperative teams where agents act in a fixed order and share a single team reward, it is hard to know how much each agent contributed, and harder still when agents are updated one at a time because data collected earlier no longer reflects the new policies. We introduce the Sequential Aristocrat Utility (SeqAU), the unique per-agent learning signal that maximizes the individual learnability of each agent's action, extending the classical framework of Wolpert and Tumer (2002) to this sequential setting. From SeqAU we derive CAPO (Counterfactual Advantage Policy Optimization), a critic-free policy-gradient algorithm. CAPO fits a per-agent reward decomposition from group rewards and computes the per-agent advantage in closed form plus a handful of forward passes through the current policy, requiring no extra environment calls beyond the initial batch. We give analytic bias and variance bounds and validate them on a controlled sequential bandit, where CAPO's advantage over standard baselines grows with the team size. The framework is general; multi-LLM pipelines are a natural deployment target. 

---
# SafeAnchor: Preventing Cumulative Safety Erosion in Continual Domain Adaptation of Large Language Models 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17691)  

**Abstract**: Safety alignment in large language models is remarkably shallow: it is concentrated in the first few output tokens and reversible by fine-tuning on as few as 100 adversarial examples. This fragility becomes critical in real-world deployment, where models undergo sequential adaptation across domains such as medicine, law, and code, causing safety guardrails to erode cumulatively. Yet all existing safety-preserving methods target only single-task fine-tuning, leaving the multi-domain sequential setting entirely unaddressed.
We introduce SafeAnchor, a framework that anchors safety in place throughout continual adaptation. SafeAnchor first identifies low-rank safety subspaces in LoRA parameter space via Fisher Information eigendecomposition, then constrains domain-specific gradient updates to the orthogonal complement of these subspaces, and finally monitors for residual safety drift with threshold-triggered corrective replay. Evaluated on Llama-2-7B-Chat and Mistral-7B-Instruct across a three-domain pipeline and eight benchmarks, SafeAnchor retains 93.2% of original safety alignment, outperforming all baselines by 18-42 points, while matching unconstrained fine-tuning to within 1.5 points on domain tasks. 

---
# Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts 

**Authors**: Moinul Hossain, Sourav Rabi Das, Zikrul Shariar Ayon, Sadia Afrin Promi, Ahnaf Atef Choudhury, Shakila Rahman, Jia Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17674)  

**Abstract**: Legal practitioners and judicial institutions face an ever-growing volume of case-law documents characterised by formalised language, lengthy sentence structures, and highly specialised terminology, making manual triage both time-consuming and error-prone. This work presents a lightweight yet high-accuracy framework for citation-treatment classification that pairs lemmatisation-based preprocessing with subword-aware FastText embeddings and a multi-kernel one-dimensional Convolutional Neural Network (CNN). Evaluated on a publicly available corpus of 25,000 annotated legal documents with a 75/25 training-test partition, the proposed system achieves 97.26% classification accuracy and a macro F1-score of 96.82%, surpassing established baselines including fine-tuned BERT, Long Short-Term Memory (LSTM) with FastText, CNN with random embeddings, and a Term Frequency-Inverse Document Frequency (TF-IDF) k-Nearest Neighbour (KNN) classifier. The model also attains the highest Area Under the Receiver Operating Characteristic (AUC-ROC) curve of 97.83% among all compared systems while operating with only 5.1 million parameters and an inference latency of 0.31 ms per document - more than 13 times faster than BERT. Ablation experiments confirm the individual contribution of each pipeline component, and the confusion matrix reveals that residual errors are confined to semantically adjacent citation categories. These findings indicate that carefully designed convolutional architectures represent a scalable, resource-efficient alternative to heavyweight transformers for intelligent legal document analysis. 

---
# ATLAS: Constitution-Conditioned Latent Geometry and Redistribution Across Language Models and Neural Perturbation Data 

**Authors**: Gareth Seneque, Lap-Hang Ho, Nafise Erfanian Saeedi, Jeffrey Molendijk, Tim Elson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17663)  

**Abstract**: Constitution-conditioned post-training can be analysed as a structured perturbation of a model's learned representational geometry. We introduce ATLAS, a geometry-first program that traces constitution-induced hidden-state structure across charts, models, and substrates. Instead of treating the relevant unit as a single behaviour, neuron, vector, or patch, ATLAS tests a local chart whose tangent structure, occupancy distribution, and behavioural coupling can be measured under system change. On Gemma, the anchored source-local chart captures 310 / 320 reviewed source rows and all 84 / 84 reviewed score-flip rows, but compact exact-patch sufficiency does not close, so the exportable unit is the broader source-defined family. Freezing that family, we re-identify a target-local realisation in an unadapted Phi model, where the fully adjudicated confirmatory contrast separates with AUC 0.984 and mean gap 5.50. In held-out ALM8 mouse frontal-cortex perturbation data, the same source-defined family receives support across 5/5 folds, with mean held-out AUC 0.72 and mean fold gap 4.50. A multiple-choice analysis provides the main boundary: nearby target-local signals can appear without source-faithful closure. The resulting correspondence is not coordinate identity, site identity, or a target-side mediation theorem. It is geometric recurrence under redistribution: written constitutions can induce recoverable latent geometry whose organisation remains detectable across model and substrate changes while its local coordinates, occupancy, and behavioural expression shift. 

---
# Semantic Density Effect (SDE): Maximizing Information Per Token Improves LLM Accuracy 

**Authors**: Amr Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.17659)  

**Abstract**: We introduce the Semantic Density Effect (SDE): the empirical finding that prompts carrying higher semantic information per token consistently produce more accurate, focused, and less hallucinated outputs across all major LLM families. SDE is defined as the ratio of semantically loaded tokens to total prompt tokens, adjusted for redundancy and concreteness. Unlike prior prompt optimization techniques that add tokens (Chain of Thought), duplicate the prompt (Prompt Repetition), or reorder components (Instruction Placement Effect), SDE improves performance by removing or replacing low-information tokens while preserving or sharpening the semantic signal. Evaluated across five frontier models and seven benchmarks, ultra-dense prompts (SDE > 0.80) outperform diluted counterparts by an average of +8.4 percentage points with 0 additional tokens and 0 latency overhead. Combined with Instruction Placement Effect (IPE), the gain reaches +11.7 percentage points 

---
# Video-Robin: Autoregressive Diffusion Planning for Intent-Grounded Video-to-Music Generation 

**Authors**: Vaibhavi Lokegaonkar, Aryan Vijay Bhosale, Vishnu Raj, Gouthaman KV, Ramani Duraiswami, Lie Lu, Sreyan Ghosh, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2604.17656)  

**Abstract**: Video-to-music (V2M) is the fundamental task of creating background music for an input video. Recent V2M models achieve audiovisual alignment by typically relying on visual conditioning alone and provide limited semantic and stylistic controllability to the end user. In this paper, we present Video-Robin, a novel text-conditioned video-to-music generation model that enables fast, high-quality, semantically aligned music generation for video content. To balance musical fidelity and semantic understanding, Video-Robin integrates autoregressive planning with diffusion-based synthesis. Specifically, an autoregressive module models global structure by semantically aligning visual and textual inputs to produce high-level music latents. These latents are subsequently refined into coherent, high-fidelity music using local Diffusion Transformers. By factoring semantically driven planning into diffusion-based synthesis, Video-Robin enables fine-grained creator control without sacrificing audio realism. Our proposed model outperforms baselines that solely accept video input and additional feature conditioned baselines on both in-distribution and out-of-distribution benchmarks with a 2.21x speed in inference compared to SOTA. We will open-source everything upon paper acceptance. 

---
# On The Mathematics of the Natural Physics of Optimization 

**Authors**: I. M. Ross  

**Link**: [PDF](https://arxiv.org/pdf/2604.17645)  

**Abstract**: A number of optimization algorithms have been inspired by the physics of Newtonian motion. Here, we ask the question: do algorithms themselves obey some ``natural laws of motion,'' and can they be derived by an application of these laws? We explore this question by positing the theory that optimization algorithms may be considered as some manifestation of hidden algorithm primitives that obey certain universal non-Newtonian dynamics. This natural physics of optimization is developed by equating the terminal transversality conditions of an optimal control problem to the generalized Karush/John-Kuhn-Tucker conditions of an optimization problem. Through this equivalence formulation, the data functions of a given constrained optimization problem generate a natural vector field that permeates an entire hidden space with information on the optimality conditions. An ``action-at-a-distance'' operation via a Pontryagin-type minimum principle produces a local action to deliver a globalized result by way of a Hamilton-Jacobi inequality. An inverse-optimal algorithm is generated by performing control jumps that dissipate quantized ``energy'' defined by a search Lyapunov function. Illustrative applications of the proposed theory show that a large number of algorithms can be generated and explained in terms of the new mathematical physics of optimization. 

---
# Provable Coordination for LLM Agents via Message Sequence Charts 

**Authors**: Benedikt Bollig, Matthias Függer, Thomas Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2604.17612)  

**Abstract**: Multi-agent systems built on large language models (LLMs) are difficult to reason about. Coordination errors such as deadlocks or type-mismatched messages are often hard to detect through testing. We introduce a domain-specific language for specifying agent coordination based on message sequence charts (MSCs). The language separates message-passing structure from LLM actions, whose outputs remain unpredictable. We define the syntax and semantics of the language and present a syntax-directed projection that generates deadlock-free local agent programs from global coordination specifications. We illustrate the approach with a diagnosis consensus protocol and show how coordination properties can be established independently of LLM nondeterminism. We also describe a runtime planning extension in which an LLM dynamically generates a coordination workflow for which the same structural guarantees apply. An open-source Python implementation of our framework is available as ZipperGen. 

---
# STEP-PD: Stage-Aware and Explainable Parkinson's Disease Severity Classification Using Multimodal Clinical Assessments 

**Authors**: Md Mezbahul Islam, John Michael Templeton, Christian Poellabauer, Ananda Mohan Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2604.17611)  

**Abstract**: Parkinson's disease (PD) is a progressive disorder in which symptom burden and functional impairment evolve over time, making severity staging essential for clinical monitoring and treatment planning. However, many computational studies emphasize binary PD detection and do not fully use repeated follow-up clinical assessments for stage-aware prediction. This study proposes STEP-PD, a severity-aware machine learning framework to classify PD severity using clinically interpretable boundaries. It leverages all available visits from the Parkinson's Progression Markers Initiative (PPMI) and integrates routinely collected subjective questionnaires and objective clinician-assessed measures. Disease severity is defined using Hoehn and Yahr staging and grouped into three clinically meaningful categories: Healthy, Mild PD (stages 1-2), and Moderate-to-Severe PD (stages 3-5). Three binary classification problems and a three-class severity task were evaluated using stratified cross-validation with imbalance-aware training. To enhance interpretability, SHAP was used to provide global explanations and local patient-level waterfall explanations. Across all tasks, XGBoost achieved the strongest and most stable performance, with accuracies of 95.48% (Healthy vs. Mild), 99.44% (Healthy vs. Moderate-to-Severe), and 96.78% (Mild vs. Moderate-to-Severe), and 94.14% accuracy with 0.8775 Macro-F1 for three-class severity classification. Explainability results highlight a shift from early motor features to progression-related axial and balance impairments. These findings show that multimodal clinical assessments within the PPMI cohort can support accurate and interpretable visit-level PD severity stratification. 

---
# Polarization and Integration in Global AI Research 

**Authors**: Luca Gallo, Riccardo Di Clemente, Balázs Lengyel  

**Link**: [PDF](https://arxiv.org/pdf/2604.17602)  

**Abstract**: The AI race amplifies security risks and international tensions. While the US restricts mobility and knowledge flows, challenges regulatory efforts to protect its advantage, China leads initiatives of global governance. Both strategies depend on cross-country relationships in AI innovation; yet, how this system evolves is unclear. Here, we measure the processes of polarization and integration in the global AI research over three decades by using large-scale data of scientific publications. Comparing cross-country collaboration and citation links to their random realizations, we find that the US and China have long diverged in both dimensions, forming two poles around which global AI research increasingly revolves. While the United Kingdom and Germany have integrated exclusively with the US, many European countries have converged with both poles. Developing and further developed countries, however, only integrate with China, signaling its expanding influence over the international AI research landscape. Our results inform national science policies and efforts toward global AI regulations. 

---
# Terminal Wrench: A Dataset of 331 Reward-Hackable Environments and 3,632 Exploit Trajectories 

**Authors**: Ivan Bercovich, Ivgeni Segal, Kexun Zhang, Shashwat Saxena, Aditi Raghunathan, Ziqian Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2604.17596)  

**Abstract**: We release Terminal Wrench, a subset of 331 terminal-agent benchmark environments, copied from the popular open benchmarks that are demonstrably reward-hackable. The data set includes 3,632 hack trajectories and 2,352 legitimate baseline trajectories across three frontier models (Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4). Each entry preserves the original task definition alongside full attack trajectories that show how the verifier was bypassed. It also includes cases where the task was not solved as intended. The tasks span system administration, machine learning, software engineering, and security challenges; the exploits range from simple output spoofing to stack-frame introspection, standard-library patching, and rootkit-style binary hijacking. Crucially, these exploits are specific to each task, rather than the evaluation harness, making them harder to patch. We also present a monitorability study in which hack trajectories are sanitized or stripped of reasoning traces and then scored by an LLM judge, showing that detection degrades meaningfully when chain-of-thought is removed (AUC drops from 0.97 to 0.92). The data set is publicly available at this https URL. 

---
# AIRA: AI-Induced Risk Audit: A Structured Inspection Framework for AI-Generated Code 

**Authors**: William M. Parris  

**Link**: [PDF](https://arxiv.org/pdf/2604.17587)  

**Abstract**: Practitioners have reported a directional pattern in AI-assisted code generation: AI-generated code tends to fail quietly, preserving the appearance of functionality while degrading or concealing guarantees. This paper introduces the Reward-Shaped Failure Hypothesis - the proposal that this pattern may reflect an artifact of optimization through human feedback rather than a random distribution of bugs. We define failure truthfulness as the property that a system's observable outputs accurately represent its internal success or failure state. We then present AIRA (AI-Induced Risk Audit), a deterministic 15-check inspection framework designed to detect failure-untruthful patterns in code. We report results from three studies: (1) an anonymized enterprise environment audit, (2) a balanced 600-file public corpus pilot, and (3) a strict matched-control replication comparing 955 AI-attributed files against 955 human-control files. In the final replication, AI-attributed files show 0.435 high-severity findings per file versus 0.242 in human controls (1.80x). The effect is consistent across JavaScript, Python, and TypeScript, with strongest concentration in exception-handling-related patterns. These findings are consistent with a directional skew toward fail-soft behavior in AI-assisted code. AIRA is designed for governance, compliance, and safety-critical systems where fail-closed behavior is required. 

---
# DGSSM: Diffusion guided state-space models for multimodal salient object detection 

**Authors**: Suklav Ghosh, Arijit Sur, Pinaki Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.17585)  

**Abstract**: Salient object detection (SOD) requires modeling both long-range contextual dependencies and fine-grained structural details, which remains challenging for convolutional, transformer-based, and Mamba-based state space models. While recent Mamba-based state space approaches enable efficient global reasoning, they often struggle to recover precise object boundaries. In contrast, diffusion models capture strong structural priors through iterative denoising, but their use in discriminative dense prediction is still limited due to computational cost and integration challenges. In this work, we propose DGSSM, a diffusion-guided state space (Mamba) framework that formulates multimodal salient object detection as a progressive denoising process. The framework integrates diffusion structural priors with multi-scale state space encoding, adaptive saliency prompting, and an iterative Mamba diffusion refinement mechanism to improve boundary accuracy. A boundary-aware refinement head and self-distillation strategy further enhance spatial coherence and feature consistency. Extensive experiments on 13 public benchmarks across RGB, RGB-D, and RGB-T settings demonstrate that DGSSM consistently outperforms state-of-the-art methods across multiple evaluation metrics while maintaining a compact model size. These results suggest that diffusion-guided state space modeling is an effective and generalizable paradigm for multimodal dense prediction tasks. 

---
# How Much Data is Enough? The Zeta Law of Discoverability in Biomedical Data, featuring the enigmatic Riemann zeta function 

**Authors**: Paul M. Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17581)  

**Abstract**: How much data is enough to make a scientific discovery? As biomedical datasets scale to millions of samples and AI models grow in capacity, progress increasingly depends on predicting when additional data will substantially improve performance. In practice, model development often relies on empirical scaling curves measured across architectures, modalities, and dataset sizes, with limited theoretical guidance on when performance should improve, saturate, or exhibit cross-over behavior.
We propose a scaling-law framework for cross-modal discoverability based on spectral structure of data covariance operators, task-aligned signal projections, and learned representations. Many performance metrics, including AUC, can be expressed in terms of cumulative signal-to-noise energy accumulated across identifiable spectral modes of an encoder and cross-modal operator. Under mild assumptions, this accumulation follows a zeta-like scaling law governed by power-law decay of covariance spectra and aligned signal energy, leading naturally to the appearance of the Riemann zeta function. Representation learning methods such as sparse models, low-rank embeddings, and multimodal contrastive objectives improve sample efficiency by concentrating useful signal into earlier stable modes, effectively steepening spectral decay and shifting scaling curves.
The framework predicts cross-over regimes in which simpler models perform best at small sample sizes, while higher-capacity or multimodal encoders outperform them once sufficient data stabilizes additional degrees of freedom. Applications include multimodal disease classification, imaging genetics, functional MRI, and topological data analysis. The resulting zeta law provides a principled way to anticipate when scaling data, improving representations, or adding modalities is most likely to accelerate discovery. 

---
# PBSBench: A Multi-Level Vision-Language Framework and Benchmark for Hematopathology Whole Slide Image Interpretation 

**Authors**: Yuanlong Wang, Weichi Chen, Adrian Rajab, Wenfang Liu, Yulan Jin, Andrew Srisuwananukorn, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17570)  

**Abstract**: Peripheral Blood Smear (PBS) is a critical microscopic examination in hematopathology that yields whole-slide imaging (WSI). Unlike solid tissue pathology, PBS interpretation focuses on individual cell morphologies rather than tissue architecture, making it distinct in both visual characteristics and diagnostic reasoning. However, current multimodal large language models (MLLMs) for pathology are primarily developed on solid-tissue WSIs and struggle to generalize to PBS. To bridge this gap, we construct PBSInstr, the first vision-language dataset for PBS interpretation, comprising 353 PBS WSIs paired with microscopic impression paragraphs and 29k cell-level image crops annotated with cell type labels and morphological descriptions. To facilitate instruction tuning, PBSInstr further includes 27k question-answer (QA) pairs for cell crops and 1,286 QA pairs for PBS slides. Building upon PBSInstr, we develop PBS-VL, a hematopathology-tailored vision-language model for multi-level PBS interpretation at both cell and slide levels. To comprehensively evaluate PBS understanding, we construct PBSBench, a visual question answering (VQA) benchmark featuring four question categories and six PBS interpretation tasks. Experiments show that PBS-VL outperforms existing general-purpose and pathology MLLMs, underscoring the value of PBS-specific data. We release our code, datasets, and model weights to facilitate future research. Our proposed framework lays the foundation for developing practical AI assistants supporting decision-making in hematopathology. 

---
# Causal-Temporal Event Graphs: A Formal Model for Recursive Agent Execution Traces 

**Authors**: Simon Foldvik  

**Link**: [PDF](https://arxiv.org/pdf/2604.17557)  

**Abstract**: We introduce causal-temporal event graphs (CTEGs) as a formal model for fully resolved recursive agent execution records under single-parenthood causal semantics. We formalise direct event emissions and recursive subagent invocations as extension procedures on generic typed temporal graphs and show that the recursive closure $\mathscr{E}_\infty$ of the induced maximal dynamics starting from single causal roots consists entirely of finite sequences of CTEGs. A CTEG is a rooted arborescence whose nodes carry timestamps and event types, subject to the constraint that timestamps be strictly increasing along causal paths. We realise $\mathscr{E}_\infty$ as the increasing union of a recursive hierarchy $\mathscr{E}_0 \subseteq \mathscr{E}_1 \subseteq \cdots$ of agent execution levels parametrised by recursion depth, which is recognised as the ascending Kleene chain of a monotone operator $\varphi$ admitting $\mathscr{E}_\infty$ as its least fixed point. Although the introduction of the full hierarchy is natural, stabilisation occurs already at $\mathscr{E}_1$ if one insists that the internal construction of a subagent execution trace be a delegated and opaque computational unit. The CTEG formalism supports compositional construction of globally well-formed execution traces from local agent behaviour without centralised coordination, preserves well-formedness under partial execution failure, and admits a natural relational database encoding. The arborescent structure of CTEGs is further compatible with cryptographic Merkle tree commitments for tamper-evident session verification. 

---
# SVL: Goal-Conditioned Reinforcement Learning as Survival Learning 

**Authors**: Franki Nguimatsia Tiofack, Fabian Schramm, Théotime Le Hellard, Justin Carpentier  

**Link**: [PDF](https://arxiv.org/pdf/2604.17551)  

**Abstract**: Standard approaches to goal-conditioned reinforcement learning (GCRL) that rely on temporal-difference learning can be unstable and sample-inefficient due to bootstrapping. While recent work has explored contrastive and supervised formulations to improve stability, we present a probabilistic alternative, called survival value learning (SVL), that reframes GCRL as a survival learning problem by modeling the time-to-goal from each state as a probability distribution. This structured distributional Monte Carlo perspective yields a closed-form identity that expresses the goal-conditioned value function as a discounted sum of survival probabilities, enabling value estimation via a hazard model trained via maximum likelihood on both event and right-censored trajectories. We introduce three practical value estimators, including finite-horizon truncation and two binned infinite-horizon approximations to capture long-horizon objectives. Experiments on offline GCRL benchmarks show that SVL combined with hierarchical actors matches or surpasses strong hierarchical TD and Monte Carlo baselines, excelling on complex, long-horizon tasks. 

---
# OPSDL: On-Policy Self-Distillation for Long-Context Language Models 

**Authors**: Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, Jingnan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17535)  

**Abstract**: Extending the effective context length of large language models (LLMs) remains a central challenge for real-world applications. While recent post-training methods have made progress in long-context scaling, they either rely on high-quality supervision data or sparse sequence-level rewards, leading to unstable and inefficient optimization. We propose OPSDL, an On-Policy Self-Distillation method for enhancing the Long-context capabilities of LLMs. Unlike other recent self-distillation methods that inject privileged information and rely on the model's in-context learning ability to act as a teacher, OPSDL leverages the model's own inherently strong short-context capability as a self-teacher to supervise its own generation in long-context scenarios. The model first generates responses conditioned on the full long-context, then the self-teacher provides per-token supervision signals via point-wise reverse KL divergence under the relevant extracted short-context. This dense token-level signal encourages faithful use of relevant evidence and mitigates hallucinations induced by irrelevant context. We evaluate OPSDL on long-context benchmarks across a range of models from 7B to 32B parameters. Results show consistent and substantial improvements across varying context lengths, outperforming standard post-training approaches such as SFT and DPO with higher sample efficiency. Notably, these gains are achieved without degrading general short-context performance. These findings highlight the effectiveness of OPSDL as a scalable and stable approach for long-context learning. 

---
# Atomic Decision Boundaries: A Structural Requirement for Guaranteeing Execution-Time Admissibility in Autonomous Systems 

**Authors**: Marcelo Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2604.17511)  

**Abstract**: Autonomous systems increasingly execute actions that directly modify shared state, creating an urgent need for precise control over which transitions are permitted to occur. Existing governance mechanisms evaluate policies prior to execution or reconstruct behavior post hoc, but do not enforce admissibility at the exact moment a state transition is committed. We introduce the atomic decision boundary, a structural property of admission control systems in which the decision and the resulting state transition are jointly determined as a single indivisible step. Formalizing execution as a labeled transition system (LTS), we distinguish two classes: atomic systems, where evaluation and transition are coupled within a single LTS step, and split evaluation systems, where they are separate transitions that may be interleaved by environmental actions. Under realistic concurrent environments, we prove that no construction can make a split system equivalent to an atomic system with respect to admissibility under all execution traces. This limitation is structural, not a matter of policy expressiveness or state availability. We further formalize the Escalate outcome -- absent from classical TOCTOU analyses -- and show its resolution is itself subject to the atomic boundary requirement. We map RBAC and OPA to the split model and contrast them with atomic systems. Admissibility is a property of execution, not evaluation. This paper is the formal foundation of a 4-paper Agent Governance Series: ACP/Paper 1 (arXiv:2603.18829), IML/Paper 2 (https://doi.org/10.5281/zenodo.19643761), Fair Allocation/Paper 3 (https://doi.org/10.5281/zenodo.19643928), Irreducibility/Paper 4 (https://doi.org/10.5281/zenodo.19643950). 

---
# Learning Unanimously Acceptable Lotteries via Queries 

**Authors**: Davin Choo, Paul W. Goldberg, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2604.17505)  

**Abstract**: Many high-stakes AI deployments proceed only if every stakeholder deems the system acceptable relative to their own minimum standard. With randomization over a finite menu of options, this becomes a feasibility question: does there exist a lottery over options that clears all stakeholders' acceptability bars? We study a query model where the algorithm proposes lotteries and receives only binary accept/reject feedback. We give deterministic and randomized algorithms that either find a unanimously acceptable lottery or certify infeasibility; adaptivity can avoid eliciting many stakeholders' constraints, and randomization further reduces the expected elicitation cost relative to full elicitation. We complement these upper bounds with worst-case lower bounds (in particular, linear dependence on the number of stakeholders and logarithmic dependence on precision are unavoidable). Finally, we develop learning-augmented algorithms that exploit natural forms of advice (e.g., likely binding stakeholders or a promising lottery), improving query complexity when predictions are accurate while preserving worst-case guarantees. 

---
# RS-HyRe-R1: A Hybrid Reward Mechanism to Overcome Perceptual Inertia for Remote Sensing Images Understanding 

**Authors**: Gaozhi Zhou, Hu He, Peng Shen, Jipeng Zhang, Liujue Zhang, Linrui Xu, Zeyuan Wang, Ziyu Li, Xuezhi Cui, Wang Guo, Haifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17504)  

**Abstract**: Reinforcement learning (RL) post-training substantially improves remote sensing vision-language models (RS-VLMs). However, when handling complex remote sensing imagery (RSI) requiring exhaustive visual scanning, models tend to rely on localized salient cues for rapid inference. We term this RL-induced bias "perceptual inertia". Driven by reward maximization, models favor quick outcome fitting, leading to two limitations: cognitively, overreliance on specific features impedes complete evidence construction; operationally, models struggle to flexibly shift visual focus across tasks. To address this bias and encourage comprehensive visual evidence mining, we propose RS-HyRe-R1, a hybrid reward framework for RSI understanding. It introduces: (1) a spatial reasoning activation reward that enforces structured visual reasoning; (2) a perception correctness reward that provides adaptive quality anchors across RS tasks, ensuring accurate geometric and semantic alignment; and (3) a visual-semantic path evolution reward that penalizes repetitive reasoning and promotes exploration of complementary cues to build richer evidence chains. Experiments show RS-HyRe-R1 effectively mitigates "perceptual inertia", encouraging deeper, more diverse reasoning. With only 3B parameters, it achieves state-of-the-art performance on REC, OVD, and VQA tasks, outperforming models up to 7B parameters. It also demonstrates strong zero-shot generalization, surpassing the second-best model by 3.16%, 3.97%, and 2.72% on VQA, OVD, and REC, respectively. Code and datasets are available at this https URL. 

---
# Generative AI Technologies, Techniques & Tensions: A Primer 

**Authors**: John T. Behrens  

**Link**: [PDF](https://arxiv.org/pdf/2604.17497)  

**Abstract**: Generative AI systems have entered everyday academic, professional, and personal life with remarkable speed, yet most users encounter them as mysterious artifacts rather than intelligible systems. This chapter discusses large language models within a broader historical shift in computing paradigms and argues that many of the confusions surrounding their use arise from a mismatch between how these systems are built, how they behave, and how people expect computers to behave writ large. Rather than treating generative AI as a monolithic technology, the chapter decomposes it into interacting components, spanning data, models, product features, and user inputs, each introducing distinct affordances and tensions. Particular attention is given to the statistical and data-based foundations of these systems and to the fact that their surface behavior is explicitly human-like, a combination that places them squarely within the intellectual traditions of educational and behavioral research. From this perspective, educational researchers are unusually well positioned to study, evaluate, and productively use generative AI systems, drawing on established methods for modeling latent processes, managing uncertainty, and interpreting complex human-system interactions. The goal is to equip readers with a conceptual map that supports more informed experimentation, critical interpretation, and responsible use as these systems continue to evolve. 

---
# A Probabilistic Consensus-Driven Approach for Robust Counterfactual Explanations 

**Authors**: Marcin Kostrzewa, Maciej Zięba, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2604.17494)  

**Abstract**: Counterfactual explanations (CFEs) are essential for interpreting black-box models, yet they often become invalid when models are slightly changed. Existing methods for generating robust CFEs are often limited to specific types of models, require costly tuning, or inflexible robustness controls. We propose a novel approach that jointly models the data distribution and the space of plausible model decisions to ensure robustness to model changes. Using a probabilistic consensus over a model ensemble, we train a conditional normalizing flow that captures the data density under varying levels of classifier agreement. At inference time, a single interpretable parameter controls the robustness level; it specifies the minimum fraction of models that should agree on the target class without retraining the generative model. Our method effectively pushes CFEs toward regions that are both plausible and stable across model changes. Experimental results demonstrate that our approach achieves superior empirical robustness while also maintaining good performance across other evaluation measures. 

---
# Dual-Anchoring: Addressing State Drift in Vision-Language Navigation 

**Authors**: Kangyi Wu, Pengna Li, Kailin Lyu, Lin Zhao, Qingrong He, Jinjun Wang, Jianyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17473)  

**Abstract**: Vision-Language Navigation(VLN) requires an agent to navigate through 3D environments by following natural language instructions. While recent Video Large Language Models(Video-LLMs) have largely advanced VLN, they remain highly susceptible to State Drift in long scenarios. In these cases, the agent's internal state drifts away from the true task execution state, leading to aimless wandering and failure to execute essential maneuvers in the instruction. We attribute this failure to two distinct cognitive deficits: Progress Drift, where the agent fails to distinguish completed sub-goals from remaining ones, and Memory Drift, where the agent's history representations degrade, making it lose track of visited landmarks. In this paper, we propose a Dual-Anchoring Framework that explicitly anchors the instruction progress and history representations. First, to address progress drift, we introduce Instruction Progress Anchoring, which supervises the agent to generate structured text tokens that delineate completed versus remaining sub-goals. Second, to mitigate memory drift, we propose Memory Landmark Anchoring, which utilizes a Landmark-Centric World Model to retrospectively predict object-centric embeddings extracted by the Segment Anything Model, compelling the agent to explicitly verify past observations and preserve distinct representations of visited landmarks. Facilitating this framework, we curate two extensive datasets: 3.6 million samples with explicit progress descriptions, and 937k grounded landmark data for retrospective verification. Extensive experiments in both simulation and real-world environments demonstrate the superiority of our method, achieving a 15.2% improvement in Success Rate and a remarkable 24.7% gain on long-horizon trajectories. To facilitate further research, we will release our code, data generation pipelines, and the collected datasets. 

---
# Project Prometheus: Bridging the Intent Gap in Agentic Program Repair via Reverse-Engineered Executable Specifications 

**Authors**: Yongchao Wang, Zhiqiu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17464)  

**Abstract**: The transition from neural machine translation to agentic workflows has revolutionized Automated Program Repair (APR). However, existing agents, despite their advanced reasoning capabilities, frequently suffer from the ``Intent Gap'' -- the misalignment between the generated patch and the developer's original intent. Current solutions relying on natural language summaries or adversarial sampling often fail to provide the deterministic constraints required for surgical repairs.
In this paper, we introduce \textsc{Prometheus}, a novel framework that bridges this gap by prioritizing \textit{Specification Inference} over code generation. We employ Behavior-Driven Development (BDD) as an executable contract, utilizing a multi-agent architecture to reverse-engineer Gherkin specifications from runtime failure reports. To resolve the ``Hallucination of Intent,'' we propose a \textbf{Requirement Quality Assurance (RQA) Loop}, a mechanism that leverages ground-truth code as a proxy oracle to validate inferred specifications.
We evaluated \textsc{Prometheus} on 680 defects from the Defects4J benchmark. The results are transformative: our framework achieved a total correct patch rate of \textbf{93.97\%} (639/680). More significantly, it demonstrated a \textbf{Rescue Rate of 74.4\%}, successfully repairing 119 complex bugs that a strong blind agent failed to resolve. Qualitative analysis reveals that explicit intent guides agents away from structurally invasive over-engineering toward precise, minimal corrections. Our findings suggest that the future of APR lies not in larger models, but in the capability to align code with verified, \textbf{Executable Specifications} -- whether pre-existing or reverse-engineered. 

---
# Agentic Education: Using Claude Code to Teach Claude Code 

**Authors**: Zain Naboulsi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17460)  

**Abstract**: AI coding assistants have proliferated rapidly, yet structured pedagogical frameworks for learning these tools remain scarce. Developers face a gap between tool documentation and practical mastery, relying on fragmented resources such as blog posts, video tutorials, and trial-and-error. We present cc-self-train, a modular interactive curriculum for learning Claude Code, an agentic AI coding tool, through hands-on project construction. The system introduces five contributions: (1) a persona progression model that adapts instructor tone across four stages (Guide, Collaborator, Peer, Launcher), operationalizing Gradual Release of Responsibility for AI-mediated instruction; (2) an adaptive learning system that observes engagement quality through hook-based heuristics and adjusts scaffolding at two timescales, using streak detection for mid-module intervention and aggregate metrics for module-boundary persona changes; (3) a cross-domain unified curriculum in which five distinct project domains share identical feature sequencing, enabling transfer learning; (4) a step-pacing mechanism with explicit pause primitives to manage information overload in an AI-as-instructor context; and (5) an auto-updating curriculum design in which the onboarding agent detects upstream tool changes and updates teaching materials before instruction begins. A parametrized test suite enforces structural consistency as a proxy for pedagogical invariants across all 50 modules. A pilot evaluation with 27 participants shows statistically significant reported self-efficacy gains across all 10 assessed skill areas (p < 0.001), with the largest effects on advanced features such as hooks and custom skills. We discuss implications for the design of auto-updating educational systems. 

---
# Beyond the Bellman Fixed Point: Geometry and Fast Policy Identification in Value Iteration 

**Authors**: Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.17457)  

**Abstract**: Dynamic programming is one of the most fundamental methodologies for solving Markov decision problems. Among its many variants, Q-value iteration (Q-VI) is particularly important due to its conceptual simplicity and its classical contraction-based convergence guarantee. Despite the central role of this contraction property, it does not fully reveal the geometric structure of the Q-VI trajectory. In particular, when one is interested not only in the final limit $Q^*$ but also in when the induced greedy policy becomes effectively optimal, the standard contraction argument provides only a coarse characterization. To formalize this notion, we denote by $\mathcal X^*$ the set of $Q$-functions whose corresponding tie-broken greedy policies are optimal, referred to as the practically optimal solution set (POS). In this paper, we revisit discounted Q-VI through the lens of switching system theory and derive new geometric insights into its behavior. In particular, we show that although Q-VI does not reach $Q^*$ in finite time in general, it identifies the optimal action class in finite time. Furthermore, we prove that the distance from the iterate to a particular subset of $\mathcal X^*$ decays exponentially at a rate governed by the joint spectral radius (JSR) of a restricted switching family. This rate can be strictly faster than the standard $\gamma$ rate when the restricted JSR is strictly smaller than $\gamma$, while the convergence of the entire $Q$-function to $Q^*$ can still be dominated by the slower $\gamma$ mode, where $\gamma$ denotes the discount factor. These results reveal a two-stage geometric behavior of Q-VI: a fast convergence toward $\mathcal X_1$, followed by a slower convergence toward $Q^*$ in general. 

---
# MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation 

**Authors**: Szu-Chi Chen, I-Ning Tsai, Yi-Cheng Lin, Sung-Feng Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.17435)  

**Abstract**: Recent Speech-to-Speech Translation (S2ST) systems achieve strong semantic accuracy yet consistently strip away non-verbal vocalizations (NVs), such as laughter and crying that convey pragmatic intent, which severely limits real-world utility. We address this via three contributions. First, we propose a synthesis pipeline for building scalable expressive datasets to overcome the data scarcity limitation. Second, we propose MoVE, a Mixture-of-LoRA-Experts architecture with expressive-specialized adapters and a soft-weighting router that blends experts for capturing hybrid expressive states. Third, we show pretrained AudioLLMs enable striking data efficiency: 30 minutes of curated data is enough for strong performance. On English-Chinese S2ST, while comparing with strong baselines, MoVE reproduces target NVs in 76% of cases and achieves the highest human-rated naturalness and emotional fidelity among all compared systems, where existing S2ST systems preserve at most 14% of NVs. 

---
# Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning 

**Authors**: Raman Saparkhan, Majd Hawasly, Md Rizwan Parvez, Mohammad Raza  

**Link**: [PDF](https://arxiv.org/pdf/2604.17433)  

**Abstract**: Self-consistency (SC) is a popular technique for improving the reasoning accuracy of large language models by aggregating multiple sampled outputs, but it comes at a high computational cost due to extensive sampling. We introduce a hybrid ensembling approach that leverages the complementary strengths of two distinct modes of reasoning: Chain-of-Thought (CoT) and Program-of-Thought (PoT). We describe a general framework for combining these two forms of reasoning in self-consistency, as well as particular strategies for both full sampling and early-stopping. We show that CoT-PoT ensembling not only improves overall accuracy, but also drastically reduces the number of samples required for SC by a factor of 9.3x. In particular, the majority of tasks (78.6%) can be addressed with only two samples, which has not been possible with any prior SC methods. 

---
# Jupiter-N Technical Report 

**Authors**: George Drayson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17429)  

**Abstract**: We present Jupiter-N, a hybrid reasoning model post-trained from Nemotron 3 Super, a fully open-source 120 billion parameter LLM. We target three objectives: (1) agentic capability via uncertainty-curated trajectories; (2) UK cultural alignment via synthetic data grounded in cultural norms; and (3) Welsh language support via parallel corpora and LLM-translated Welsh conversations. Our data curation strategy carefully preserves the base model's capabilities: using our Forget-Me-Not framework, we mix on-policy synthetic replay with off-policy task data to mitigate catastrophic forgetting, and include a mixture of reasoning and non-reasoning traces to maintain Nemotron's hybrid reasoning ability. Jupiter-N achieves standout gains over Nemotron in Welsh (+18 on ARC-Easy, +5.25 on MMLU-Lite), terminal-use (+9.1 on Terminal Bench 2) and instruction following (+4.4 on IFBench), while retaining the base model capabilities. We frame this work as a reproducible template for sovereign post-training: substituting cultural knowledge, institutional corpora, and target languages produces an equivalent pipeline for any country. All model weights and all post-training datasets are publicly released under open licences. 

---
# Long-CODE: Isolating Pure Long-Context as an Orthogonal Dimension in Video Evaluation 

**Authors**: Zhijiang Tang, Jiaxin Qi, Bing Zhao, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17428)  

**Abstract**: As video generation models achieve unprecedented capabilities, the demand for robust video evaluation metrics becomes increasingly critical. Traditional metrics are intrinsically tailored for short-video evaluation, predominantly assessing frame-level visual quality and localized temporal smoothness. However, as state-of-the-art video generation models scale to generate longer videos, these metrics fail to capture essential long-range characteristics, such as narrative richness and global causal consistency. Recognizing that short-term visual perception and long-context attributes are fundamentally orthogonal dimensions, we argue that long-video metrics should be disentangled from short-video assessments. In this paper, we focus on the rigorous justification and design of a dedicated framework for long-video evaluation. We first introduce a suite of long-video attribute corruption tests, exposing the critical limitations of existing hort-video metrics from their insensitivity to structural inconsistencies, such as shot-level perturbations and narrative shuffling. To bridge this gap, we design a novel long-video metric based on shot dynamics, which is highly sensitive to the long-range testing framework. Furthermore, we introduce Long-CODE (Long-Context as an Orthogonal Dimension for video Evaluation), a specialized dataset designed to benchmark long-video evaluation, with human annotations isolated specifically to genuine long-range characteristics. Extensive experiments show that our proposed metrics achieve state-of-the-art correlation with human judgments. Ultimately, our metric and benchmark seamlessly complement existing short-video standards, establishing a holistic and unbiased evaluation paradigm for video generation models. 

---
# TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering 

**Authors**: Keyang Chen, Mingxuan Jiang, Yongsheng Zhao, Zeping Li, Zaiyuan Chen, Weiqi Luo, Zhixin Li, Sen Liu, Yinan Jing, Guangnan Ye, Xihong Wu, Hongfeng Chai  

**Link**: [PDF](https://arxiv.org/pdf/2604.17420)  

**Abstract**: Money laundering poses severe risks to global financial systems, driving the widespread adoption of machine learning for transaction monitoring. However, progress remains stifled by the lack of realistic benchmarks. Existing transaction-graph datasets suffer from two pervasive limitations: (i) they provide sparse node-level semantics beyond anonymized identifiers, and (ii) they rely on template-driven anomaly injection, which biases benchmarks toward static structural motifs and yields overly optimistic assessments of model robustness. We propose TransXion, a benchmark ecosystem for Anti-Money Laundering (AML) research that integrates profile-aware simulation of normal activity with stochastic, non-template synthesis of illicit this http URL jointly models persistent entity profiles and conditional transaction behavior, enabling evaluation of "out-of-character" anomalies where observed activity contradicts an entity's socio-economic context. The resulting dataset comprises approximately 3 million transactions among 50,000 entities, each endowed with rich demographic and behavioral attributes. Empirical analyses show that TransXion reproduces key structural properties of payment networks, including heavy-tailed activity distributions and localized subgraph structure. Across a diverse array of detection models spanning multiple algorithmic paradigms, TransXion yields substantially lower detection performance than widely used benchmarks, demonstrating increased difficulty and realism. TransXion provides a more faithful testbed for developing context-aware and robust AML detection methods. The dataset and code are publicly available at this https URL. 

---
# Project resilience as network robustness 

**Authors**: Sebastiano A. Piccolo, Giorgio Terracina  

**Link**: [PDF](https://arxiv.org/pdf/2604.17417)  

**Abstract**: Engineering projects are the result of the combined effort of their members. Yet, it has been documented that labor division withing projects is unevenly distributed: some project members are specialists undertaking only few tasks, whereas other are generalists and are responsible for the success of many tasks. Moreover, the latter are often facilitators of project integration. Such a workload distribution prompts one question: how resilient is a project to key personnel loss? Far from being a theoretical problem, the reliance of a project on a few key people can lead to severe economic losses and delays. We argue that current methods to estimate such a risk are unsatisfactory: some methods offer a best-case estimate and are, therefore, too optimistic; other methods fail to capture project fragmentation leading to biased estimates and unrealistic consequences in many settings. In this paper, we develop a novel method to assess project vulnerability by looking at it from the lens of network robustness. We compare our method against existing alternatives and show that it offers better and more consistent estimates of project resilience to personnel loss. 

---
# Reward Score Matching: Unifying Reward-based Fine-tuning for Flow and Diffusion Models 

**Authors**: Jeongjae Lee, Jinho Chang, Jeongsol Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.17415)  

**Abstract**: Reward-based fine-tuning aims to steer a pretrained diffusion or flow-based generative model toward higher-reward samples while remaining close to the pretrained model. Although existing methods are motivated by different perspectives such as Soft RL, GFlowNets, etc., we show that many can be written under a common framework, which we call reward score matching (RSM). Under this view, alignment becomes score matching toward a reward-guided target, and the main differences across methods reduce to the construction of the value-guidance estimator and the effective optimization strength across timesteps. This unification clarifies the bias--variance--compute tradeoffs of existing designs and distinguishes core optimization components from auxiliary mechanisms that add complexity without clear benefit. Guided by this perspective, we develop simpler redesigns that improve alignment effectiveness and compute efficiency across representative settings with differentiable and black-box rewards. Overall, RSM turns a seemingly fragmented collection of reward-based fine-tuning methods into a smaller, more interpretable, and more actionable design space. 

---
# The Open-Weight Paradox: Why Restricting Access to AI Models May Undermine the Safety It Seeks to Protect 

**Authors**: Vinicius Santana Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2604.17413)  

**Abstract**: The governance of open-weight artificial intelligence (AI) models has been framed as a binary choice: openness as risk, restriction as safety. This paper challenges that framing, arguing that access restrictions, without governed alternatives, may displace risks rather than reduce them. The global concentration of compute infrastructure makes open-weight models one of the most viable pathways to sovereign AI capacity in the Global South; restricting such access deepens asymmetries while driving proliferation into unsupervised settings. This analysis proposes that hardware-layer governance, including chip-level attestation mechanisms such as FlexHEG, trusted execution environments, confidential computing, and complementary software-layer safeguards, offers a defense-in-depth alternative to the current binary. A threat model taxonomy mapping misuse vectors to hardware, software, institutional, and liability layers illustrates why no single governance mechanism suffices. To operationalize this approach, the paper argues that effective AI governance as a dual-use technology will likely require a multilateral institutional architecture functionally analogous, though not identical, to the role performed by the IAEA in the nuclear domain, with explicit safeguards against the co-option of hardware controls for domestic repression. The relevant policy question is how to make openness safer through technical and institutional design while addressing the transition realities of legacy hardware, attestation at scale, and civil liberties protection. 

---
# DuConTE: Dual-Granularity Text Encoder with Topology-Constrained Attention for Text-attributed Graphs 

**Authors**: Lexuan Liang, Tao Zou, Xuxiang Ta, Zekun Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17411)  

**Abstract**: Text-attributed graphs integrate semantic information of node texts with topological structure, offering significant value in various applications such as document classification and information extraction. Existing approaches typically encode textual content using language models (LMs), followed by graph neural networks (GNNs) to process structural information. However, during the LM-based text encoding phase, most methods not only perform semantic interaction solely at the word-token granularity, but also neglect the structural dependencies among texts from different nodes. In this work, we propose DuConTE, a dual-granularity text encoder with topology-constrained attention. The model employs a cascaded architecture of two pretrained LMs, encoding semantics first at the word-token granularity and then at the node granularity. During the self-attention computation in each LM, we dynamically adjust the attention mask matrix based on node connectivity, guiding the model to learn semantic correlations informed by the graph structure. Furthermore, when composing node representations from word-token embeddings, we separately evaluate the importance of tokens under the center-node context and the neighborhood context, enabling the capture of more contextually relevant semantic information. Extensive experiments on multiple benchmark datasets demonstrate that DuConTE achieves state-of-the-art performance on the majority of them. 

---
# Speculative Decoding for Autoregressive Video Generation 

**Authors**: Yuezhou Hu, Jintao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17397)  

**Abstract**: Autoregressive video diffusion is emerging as a promising paradigm for streaming video synthesis, with step distillation serving as the primary means of accelerating inference. Whether speculative decoding, the dominant acceleration strategy for large language models, can be effectively adapted to autoregressive video generation remains an open question, because video blocks are continuous spatiotemporal tensors with no token-level distribution for exact rejection sampling. We introduce SDVG, which brings speculative decoding to block-based autoregressive video diffusion by replacing token verification with an image-quality router. A 1.3B drafter proposes candidate blocks via four denoising steps; each block is VAE-decoded and scored by ImageReward using worst-frame aggregation--taking the minimum per-frame reward to catch single-frame artifacts that averaging would mask. Blocks scoring above a fixed threshold tau are accepted into the 14B target's KV cache; the rest are regenerated by the target. Two additional design choices prove critical: the first block is always force-rejected to anchor scene composition, and tau serves as a single knob that traces a smooth quality-speed Pareto frontier. On 1003 MovieGenVideoBench prompts (832x480), SDVG retains 98.1% of target-only VisionReward quality (0.0773 vs. 0.0788) at a 1.59x speedup with tau=-0.7, and reaches 2.09x at 95.7% quality retention--while consistently outperforming draft-only generation by over +17%. The framework is training-free, requires no architectural changes, and can be seamlessly integrated into existing autoregressive video generation pipelines. 

---
# MESA: A Training-Free Multi-Exemplar Deep Framework for Restoring Ancient Inscription Textures 

**Authors**: Vasileios Toulatzis, Ioannis Fudos  

**Link**: [PDF](https://arxiv.org/pdf/2604.17390)  

**Abstract**: Ancient inscriptions frequently suffer missing or corrupted regions from fragmentation, erosion, or other damage, hindering reading, and analysis. We review prior image restoration methods and their applicability to inscription image recovery, then introduce MESA (Multi-Exemplar, Style-Aware) -an image-level restoration method that uses well-preserved exemplar inscriptions (from the same epigraphic monument, material, or similar letterforms) to guide reconstruction of damaged text. MESA encodes VGG19 convolutional features as Gram matrices to capture exemplar texture, style, and stroke structure; for each neural network layer it selects the exemplar minimizing Mean-Squared Displacement (MSD) to the damaged input. Layer-wise contribution weights are derived from Optical Character Recognition-estimated character widths in the exemplar set to bias filters toward scales matching letter geometry, and a training mask preserves intact regions so synthesis is restricted to damaged areas. We also summarize prior network architectures and exemplar and single-image synthesis, inpainting, and Generative Adversarial Network (GAN) approaches, highlighting limitations that MESA addresses. Comparative experiments demonstrate the advantages of MESA. Finally, we provide a practical roadmap for choosing restoration strategies given available exemplars and metadata. 

---
# Study and Improvement of Search Algorithms in Multi-Player Perfect-Information Games 

**Authors**: Quentin Cohen-Solal  

**Link**: [PDF](https://arxiv.org/pdf/2604.17378)  

**Abstract**: In this article, we generalize Unbounded Minimax, the state-of-the-art search algorithm for zero sums two-player games with perfect information to the framework of multiplayer games with perfect information. We experimentally show that this generalized algorithm also achieves better performance than the main multiplayer search algorithms. 

---
# Towards Generalizable Deepfake Image Detection with Vision Transformers 

**Authors**: Kaliki V Srinanda, M Manvith Prabhu, Hemanth K Mogilipalem, Jayavarapu S Abhinai, Vaibhav Santhosh, Aryan Herur, Deepu Vijayasenan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17376)  

**Abstract**: In today's day and age, we face a challenge in detecting deepfake images because of the fast evolution of modern generative models and the poor generalization capability of existing methods. In this paper, we use an ensemble of fine-tuned vision transformers like DINOv2, AIMv2 and OpenCLIP's ViT-L/14 to create generalizable method to detect deepfakes. We use the DF-Wild dataset released as part of the IEEE SP Cup 2025, because it uses a challenging and diverse set of manipulations and generation techniques. We started our experiments with CNN classifiers trained on spatial features. Experimental results show that our ensemble outperforms individual models and strong CNN baselines, achieving an AUC of 96.77% and an Equal Error Rate (EER) of just 9% on the DF-Wild test set, beating the state-of-the-art deepfake detection algorithm Effort by 7.05% and 8% in AUC and EER respectively. This was the winning solution for SP Cup, presented at ICASSP 2025. 

---
# When Text Hijacks Vision: Benchmarking and Mitigating Text Overlay-Induced Hallucination in Vision Language Models 

**Authors**: Cui Yakun, Xingqun Qi, TianTian Geng, Yuyao Zhang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.17375)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have substantially enhanced their ability across multimodal video understanding benchmarks spanning temporal, action, object, and spatial understanding. However, we identify a critical yet overlooked issue: when embedded on-screen text contradicts the visual scene, existing VLMs systematically hallucinate, prioritizing overlay textual semantics over the actual visual content. We define this phenomenon as Text Overlay-Induced Hallucination (TOIH). In this work, we propose VisualTextTrap, the first comprehensive benchmark, including large-scale human-validated samples with specifically designed evaluation metrics. In particular, we construct VisualTextTrap from widely-used public datasets using a scalable hybrid pipeline of VLMs assisted text generation and rigorous manual verification. The benchmark features 6,057 samples annotated across 88 fine-grained attributes within four dimensions, with hallucination intensity quantified on a five-level scale (L1--L5) that reflects the semantic contradiction between overlay text and visual reality. Moreover, we propose Visual Text Hallucination Mitigation Mixture-of-Experts (VTHM-MoE), a novel Vision-Text Disentanglement framework that employs a dual-encoder architecture. Concretely, four dimension-specialized expert modules spanning Temporal, Action, Object, and Spatial reasoning are first pre-trained to identify and leverage cross-modal discrepancies between textual semantics and actual video content. We develop an Adaptive Token Routing Strategy to enable dynamic expert allocation, conferring robust resistance to TOIH while preserving performance on uncontaminated videos. Extensive experiments conducted on our VisualTextTrap benchmark verify the effectiveness of VTHM-MoE, outperforming state-of-the-art counterparts with diverse video question answering tasks. 

---
# ArgBench: Benchmarking LLMs on Computational Argumentation Tasks 

**Authors**: Yamen Ajjour, Carlotta Quensel, Nedim Lipka, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2604.17366)  

**Abstract**: Argumentation skills are an essential toolkit for large language models (LLMs). These skills are crucial in various use cases, including self-reflection, debating collaboratively for diverse answers, and countering hate speech. In this paper, we create the first benchmark for a standardized evaluation of LLM-based approaches to computational argumentation, encompassing 33 datasets from previous work in unified form. Using the benchmark, we evaluate the generalizability of five LLM families across 46 computational argumentation tasks that cover mining arguments, assessing perspectives, assessing argument quality, reasoning about arguments, and generating arguments. On the benchmark, we conduct an extensive systematic analysis of the contribution of few-shot examples, reasoning steps, model size, and training skills to the performance of LLMs on the computational argumentation tasks in the benchmark. 

---
# PsychBench: Auditing Epidemiological Fidelity in Large Language Model Mental Health Simulations 

**Authors**: Patrick Keough  

**Link**: [PDF](https://arxiv.org/pdf/2604.17359)  

**Abstract**: Large language models are increasingly deployed to simulate patients for clinical training, research, and mental health tools, yet population-level validity remains largely untested. We introduce PsychBench, the first epidemiological audit of LLM patient simulation: 28,800 profiles from four frontier models (GPT-4o-mini, DeepSeek-V3, Gemini-3-Flash, GLM-4.7) evaluated against NHANES and NESARC-III baselines across 120 intersectional cohorts. The central finding is a coherence-fidelity dissociation: models produce clinically plausible individuals while misrepresenting the populations they are drawn from. Variance compression ranges from 14 percent (GLM-4.7) to 62 percent (DeepSeek-V3), eliminating the distributional tails of clinical reality. Despite test-retest correlations above r = 0.90, 36.66 percent of cases cross diagnostic thresholds between runs. Symptom correlation matrices diverge across demographic groups beyond split-half noise, with transgender populations diverging three to five times more than racial differences. Calibration bias is systematic and asymmetric. Models overestimate depression severity for most groups by 3.6 to 6.1 points (Cohen d = 1.13 to 1.91), consistent with training on clinical corpora with elevated base rates. For transgender women the direction inverts: models capture only 8 to 46 percent of documented minority stress elevation, yielding a -5.42 residual (d = -1.55). Models also attribute irritability to Black men and fatigue to women beyond matched controls, encoding racialized and gendered assumptions. Patterns replicate across US and Chinese architectures, indicating failures tied to current training paradigms rather than isolated implementations. For most users, LLM mental health tools risk pathologizing ordinary distress; for transgender users, algorithmic erasure of genuine need. The patients look right. They do not represent real populations. 

---
# Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions 

**Authors**: Dongwook Lee, Eunwoo Song, Che Hyun Lee, Heeseung Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.17358)  

**Abstract**: While recent Spoken Language Models (SLMs) have been actively deployed in real-world scenarios, they lack the capability to discern Third-Party Interruptions (TPI) from the primary user's ongoing flow, leaving them vulnerable to contextual failures. To bridge this gap, we introduce TPI-Train, a dataset of 88K instances designed with speaker-aware hard negatives to enforce acoustic cue prioritization for interruption handling, and TPI-Bench, a comprehensive evaluation framework designed to rigorously measure the interruption-handling strategy and precise speaker discrimination in deceptive contexts. Experiments demonstrate that our dataset design mitigates semantic shortcut learning-a critical pitfall where models exploit semantic context while neglecting acoustic signals essential for discerning speaker changes. We believe our work establishes a foundational resource for overcoming text-dominated unimodal reliance in SLMs, paving the way for more robust multi-party spoken interaction. The code for the framework is publicly available at this https URL 

---
# Robust Diabetic Retinopathy Grading Using Dual-Resolution Attention-Based Deep Learning with Ordinal Regression 

**Authors**: Afshan Hashmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17341)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of vision impairment worldwide, and automated grading systems play a crucial role in large-scale screening programs. However, deep learning models often exhibit degraded performance when deployed across datasets acquired under different imaging conditions. This study presents a robust dual-resolution deep learning framework for DR grading that integrates attention-based feature fusion with ordinal regression to improve cross-dataset generalization. The proposed method employs two parallel EfficientNet backbones operating at different spatial resolutions to capture complementary retinal features. A learnable attention mechanism adaptively fuses multi-resolution representations, while an ordinal regression formulation based on the cumulative link model (CORAL) explicitly accounts for the ordered nature of DR severity levels. To mitigate domain discrepancies between datasets, a preprocessing strategy combining circular cropping, contrast enhancement, and histogram matching is applied. The model was trained on the APTOS 2019 dataset and evaluated on both an internal validation split and an external Messidor-2 test set. Experimental results demonstrate strong grading performance, achieving a quadratic weighted kappa (QWK) of 0.88 on the APTOS validation set and 0.68 on the unseen Messidor-2 dataset, indicating improved robustness for cross-dataset DR grading applications. 

---
# Rethinking the Comparison Unit in Sequence-Level Reinforcement Learning: An Equal-Length Paired Training Framework from Loss Correction to Sample Construction 

**Authors**: Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng, Huiming Yang, Sibo wang, Linglin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17328)  

**Abstract**: This paper investigates the length problem in sequence-level relative reinforcement learning. We observe that, although existing methods partially alleviate length-related phenomena, a more fundamental issue remains insufficiently characterized: the comparison units used during training lack inherent comparability. Building on this observation, we propose a new perspective: the length problem should not be viewed merely as a loss-scaling or normalization bias, but rather as a \emph{comparison unit construction} problem. We further establish a sample-construction-based training framework that, instead of applying post-hoc corrections to unequal-length responses, proactively constructs equal-length, alignable, and comparable training segments during generation. Within this framework, we propose EqLen, a concrete method applicable to group-relative comparison algorithms such as GRPO, GSPO, and RLOO. Through dual-track synchronous generation, prefix inheritance, and segment masking, EqLen efficiently collects effective equal-length training segments and enables stable 

---
# Signal or Noise in Multi-Agent LLM-based Stock Recommendations? 

**Authors**: George Fatouros, Kostas Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2604.17327)  

**Abstract**: We present the first portfolio-level validation of MarketSenseAI, a deployed multi-agent LLM equity system. All signals are generated live at each observation date, eliminating look-ahead bias. The system routes four specialist agents (News, Fundamentals, Dynamics, and Macro) through a synthesis agent that issues a monthly equity thesis and recommendation for each stock in its coverage universe, and we ask two questions: do its buy recommendations add value over both passive benchmarks and random selection, and what does the internal agent structure reveal about the source of the edge? On the S&P 500 cohort (19 months) the strong-buy equal-weight portfolio earns +2.18%/month against a passive equal-weight benchmark of +1.15% (approximating RSP), a +25.2% compound excess, and ranks at the 99.7th percentile of 10,000 Monte Carlo portfolios (p=0.003). The S&P 100 cohort (35 months) delivers a +30.5% compound excess over EQWL with consistent direction but formal significance not reached, limited by the small average selection of ~10 stocks per month. Non-negative least-squares projection of thesis embeddings onto agent embeddings reveals an adaptive-integration mechanism. Agent contributions rotate with market regime (Fundamentals leads on S&P 500, Macro on S&P 100, Dynamics acts as an episodic momentum signal) and this agent rotation moves in lockstep with both the sector composition of strong-buy selections and identifiable macro-calendar events, three independent views of the same underlying adaptation. The recommendation's cross-sectional Information Coefficient is statistically significant on S&P 500 (ICIR=+0.489, p=0.024). These results suggest that multi-agent LLM equity systems can identify sources of alpha beyond what classical factor models capture, and that the buy signal functions as an effective universe-filter that can sit upstream of any portfolio-construction process. 

---
# SigGate-GT: Taming Over-Smoothing in Graph Transformers via Sigmoid-Gated Attention 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17324)  

**Abstract**: Graph transformers achieve strong results on molecular and long-range reasoning tasks, yet remain hampered by over-smoothing (the progressive collapse of node representations with depth) and attention entropy degeneration. We observe that these pathologies share a root cause with attention sinks in large language models: softmax attention's sum-to-one constraint forces every node to attend somewhere, even when no informative signal exists. Motivated by recent findings that element-wise sigmoid gating eliminates attention sinks in large language models, we propose SigGate-GT, a graph transformer that applies learned, per-head sigmoid gates to the attention output within the GraphGPS framework. Each gate can suppress activations toward zero, enabling heads to selectively silence uninformative connections. On five standard benchmarks, SigGate-GT matches the prior best on ZINC (0.059 MAE) and sets new state-of-the-art on ogbg-molhiv (82.47% ROC-AUC), with statistically significant gains over GraphGPS across all five datasets ($p < 0.05$). Ablations show that gating reduces over-smoothing by 30% (mean relative MAD gain across 4-16 layers), increases attention entropy, and stabilizes training across a $10\times$ learning rate range, with about 1% parameter overhead on OGB. 

---
# Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA 

**Authors**: Alberto Testoni, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2604.17316)  

**Abstract**: Safe clinical deployment of Large Language Models (LLMs) requires not only high accuracy but also robust uncertainty calibration to ensure models defer to clinicians when appropriate. Our paper investigates how social descriptors of a patient (specifically sexual orientation and religious affiliation) distort these uncertainty signals and model accuracy. Evaluating nine general-purpose and biomedical LLMs on 2,364 medical questions and their counterfactual variants, we demonstrate that identity markers cause a "calibration crisis". "Homosexual" markers consistently trigger performance drops, and intersectional identities produce idiosyncratic, non-additive harms to calibration. Moreover, a clinician-validated case study in an open-ended generation setting confirms that these failures are not an artifact of the multiple-choice format. Our results demonstrate that the presence of social identity cues does not merely shift predictions; it affects the reliability of confidence signals, posing a significant risk to equitable care and safe deployment in confidence-based clinical workflows. 

---
# A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions 

**Authors**: Zhiyin Yu, Yuchen Mou, Juncheng Yan, Junyu Luo, Chunchun Chen, Xing Wei, Yunhui Liu, Hongru Sun, Yuxing Zhang, Jun Xu, Yatao Bian, Ming Zhang, Wei Ye, Tieke He, Jie Yang, Guanjie Zheng, Zhonghai Wu, Bo Zhang, Lei Bai, Xiao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.17312)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful post-training paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, reinforcement learning for LLMs faces substantial data scarcity challenges, including the limited availability of high-quality external supervision and the constrained volume of model-generated experience. These limitations make data-efficient reinforcement learning a critical research direction. In this survey, we present the first systematic review of reinforcement learning for LLMs under data scarcity. We propose a bottom-up hierarchical framework built around three complementary perspectives: the data-centric perspective, the training-centric perspective, and the framework-centric perspective. We develop a taxonomy of existing methods, summarize representative approaches in each category, and analyze their strengths and limitations. Our taxonomy aims to provide a clear conceptual foundation for understanding the design space of data-efficient RL for LLMs and to guide researchers working in this emerging area. We hope this survey offers a comprehensive roadmap for future research and inspires new directions toward more efficient and scalable reinforcement learning post-training for LLMs. 

---
# RoTRAG: Rule of Thumb Reasoning for Conversation Harm Detection with Retrieval-Augmented Generation 

**Authors**: Juhyeon Lee, Wonduk Seo, Junseo Koh, Seunghyun Lee, Haihua Chen, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17301)  

**Abstract**: Detecting harmful content in multi turn dialogue requires reasoning over the full conversational context rather than isolated utterances. However, most existing methods rely mainly on models internal parametric knowledge, without explicit grounding in external normative principles. This often leads to inconsistent judgments in socially nuanced contexts, limited interpretability, and redundant reasoning across turns. To address this, we propose RoTRAG, a retrieval augmented framework that incorporates concise human written moral norms, called Rules of Thumb (RoTs), into LLM based harm assessment. For each turn, RoTRAG retrieves relevant RoTs from an external corpus and uses them as explicit normative evidence for turn level reasoning and final severity classification. To improve efficiency, we further introduce a lightweight binary routing classifier that decides whether a new turn requires retrieval grounded reasoning or can reuse existing context. Experiments on ProsocialDialog and Safety Reasoning Multi Turn Dialogue show that RoTRAG consistently improves both harm classification and severity estimation over competitive baselines, with an average relative gain of around 40% in F1 across benchmark datasets and an average relative reduction of 8.4% in distributional error, while reducing redundant computation without sacrificing performance. 

---
# Chaos-Enhanced Prototypical Networks for Few-Shot Medical Image Classification 

**Authors**: Chinhtakuntla Meghan Sai, Murarisetty V Sai Kartheek, Sita Devi Bharatula, Karthik Seemakurthy  

**Link**: [PDF](https://arxiv.org/pdf/2604.17300)  

**Abstract**: The scarcity of labeled clinical data in oncology makes Few-Shot Learning (FSL) a critical framework for Computer Aided Diagnostics, but we observed that standard Prototypical Networks often struggle with the "prototype instability" caused by morphological noise and high intra-class variance in brain tumor scans. Our work attempts to minimize this by integrating a non-linear Logistic Chaos Module into a fine-tuned ResNet-18 backbone creating the Chaos-Enhanced ProtoNet(CE-ProtoNet). Using the deterministic ergodicity of the logistic chaos map we inject controlled perturbations into support features during episodic training-essentially for "stress testing" the embedding space. This process makes the model to converge on noise-invariant representations without increasing computational overhead. Testing this on a 4-way 5-shot brain tumor classification task, we found that a 15% chaotic injection level worked efficiently to stabilize high-dimensional clusters and reduce class dispersion. Our method achieved a peak test accuracy of 84.52%, outperforming standard ProtoNet. Our results suggest the idea of using chaotic perturbation as an efficient, low-overhead regularization tool, for the data-scarce regimes. 

---
# Cat-DPO: Category-Adaptive Safety Alignment 

**Authors**: Tiankai Yang, Yi Nian, Xinyuan Li, Ruiyao Xu, Kaize Ding, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17299)  

**Abstract**: Aligning large language models with human preferences must balance two competing goals: responding helpfully to legitimate requests and reliably refusing harmful ones. Most preference-based safety alignment methods collapse safety into a single scalar that is applied uniformly to every preference pair. The result is a model that looks safe on average but stays relatively unsafe on a minority of harm categories. We cast safety alignment as a per-category constrained optimization problem and derive Cat-DPO, a direct-preference-optimization algorithm with a separate adaptive safety margin for each harm category. The margin tightens when the model still produces unsafe responses on a category and relaxes once the model catches up, so the training signal tracks each category's current difficulty rather than averaging under one global rate. Across two LLM backbones and six preference-learning baselines, Cat-DPO iimproves aggregate helpfulness and harmlessness and compresses per-category safety variance and the best-to-worst gap, offering a drop-in per-category refinement of direct preference safety alignment. 

---
# Probabilistic Programs of Thought 

**Authors**: Poorva Garg, Renato Lui Geh, Daniel Israel, Todd Millstein, Kyle Richardson, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2604.17290)  

**Abstract**: LLMs are widely used for code generation and mathematical reasoning tasks where they are required to generate structured output. They either need to reason about code, generate code for a given specification, or reason using programs of thought. The typical approach to code generation is to prompt the model and generate samples until an appropriate program is obtained. Within this process, sampling $n$ programs from the language model requires $n$ GPU compute-intensive generations which becomes prohibitively expensive for larger values of $n$. In this work, we address this limitation by exposing the LLM's distribution within the generated programs themselves. We propose a novel test-time framework we dub probabilistic programs of thought to obtain more samples from the model with fewer LLM generations. Given a program generated by a model and the associated next-token probabilities, we build a probabilistic program that compactly represents exponentially many deterministic programs. Since performing probabilistic reasoning in this probabilistic program is much cheaper, our approach allows sampling new programs without any additional GPU compute and little CPU overhead. We instantiate our approach on benchmarks for code generation, code understanding and mathematical reasoning and report improvements in performance with fewer generations from the LLM. 

---
# Clover: A Neural-Symbolic Agentic Harness with Stochastic Tree-of-Thoughts for Verified RTL Repair 

**Authors**: Zizhang Luo, Yansong Xu, Runlin Guo, Fan Cui, Kexing Zhou, Mile Xia, Hongyuan Hou, Yuhao Luo, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17288)  

**Abstract**: RTL program repair remains a critical bottleneck in hardware design and verification. Traditional automatic program repair (APR) methods rely on predefined templates and synthesis, limiting their bug coverage. Large language models (LLMs) and coding agents based on them offer flexibility but suffer from randomness and context corruption when handling long RTL code and waveforms. We present Clover, a neural-symbolic agentic harness that orchestrates RTL repair as a structured search over code manipulations to explore a validated solution for the bug. Recognizing that different repair operations favor distinct strategies, Clover dynamically dispatches tasks to specialized LLM agents or symbolic solvers. At its core, Clover introduces stochastic tree-of-thoughts, a test-time scaling mechanism that manages the main agent's context as a search tree, balancing exploration and exploitation for reliable outcomes. An RTL-specific toolbox further empowers agents to interact with the debugging environment. Evaluated on the RTL-repair benchmark, Clover fixes 96.8% of bugs within a fixed time limit, covering 94% and 63% more bugs than both pure traditional and LLM-based baselines, respectively, while achieving an average pass@1 rate of 87.5%, demonstrating high reliability and effectiveness. 

---
# HorizonBench: Long-Horizon Personalization with Evolving Preferences 

**Authors**: Shuyue Stella Li, Bhargavi Paranjape, Kerem Oktar, Zhongyao Ma, Gelin Zhou, Lin Guan, Na Zhang, Sem Park, Lin Chen, Diyi Yang, Yulia Tsvetkov, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.17283)  

**Abstract**: User preferences evolve across months of interaction, and tracking them requires inferring when a stated preference has been changed by a subsequent life event. We define this problem as long-horizon personalization and observe that progress on it is limited by data availability and measurement, with no existing resource providing both naturalistic long-horizon interactions and the ground-truth provenance needed to diagnose why models fail. We introduce a data generator that produces conversations from a structured mental state graph, yielding ground-truth provenance for every preference change across 6-month timelines, and from it construct HorizonBench, a benchmark of 4,245 items from 360 simulated users with 6-month conversation histories averaging ~4,300 turns and ~163K tokens. HorizonBench provides a testbed for long-context modeling, memory-augmented architectures, theory-of-mind reasoning, and user modeling. Across 25 frontier models, the best model reaches 52.8% and most score at or below the 20% chance baseline. When these models err on evolved preferences, over a third of the time they select the user's originally stated value without tracking the updated user state. This belief-update failure persists across context lengths and expression explicitness levels, identifying state-tracking capability as the primary bottleneck for long-horizon personalization. 

---
# Fully Analog Resonant Recurrent Neural Network via Metacircuit 

**Authors**: Zixin Zhou, Tianxi Jiang, Menglong Yang, Zhihua Feng, Qingbo He, Shiwu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17277)  

**Abstract**: Physical neural networks offer a transformative route to edge intelligence, providing superior inference speed and energy efficiency compared to conventional digital architectures. However, realizing scalable, end-to-end, fully analog recurrent neural networks for temporal information processing remains challenging due to the difficulty of faithfully mapping trained network models onto physical hardware. Here we present a fully analog resonant recurrent neural network (R$^2$NN) implemented via a metacircuit architecture composed of coupled electrical local resonators. A reformulated mechanical-electrical analogy establishes a direct mapping between the R$^2$NN model and metacircuit elements, enabling accurate physical implementation of trained neural network parameters. By integrating jointly trainable global resistive coupling and local resonances, which generate effective frequency-dependent negative resistances, the architecture shapes an impedance landscape that steers currents along frequency-selective pathways. This mechanism enables direct extraction of discriminative spectral features, facilitating real-time temporal classification of raw analog inputs while bypassing analog-to-digital conversion. We demonstrate the cross-domain versatility of this framework using integrated hardware for tactile perception, speech recognition, and condition monitoring. This work establishes a scalable, fully analog paradigm for intelligent temporal processing and paves the way for low-latency, resource-efficient physical neural hardware for edge intelligence. 

---
# Instinct vs. Reflection: Unifying Token and Verbalized Confidence in Multimodal Large Models 

**Authors**: Yunkai Dang, Yifan Jiang, Yizhu Jiang, Anqi Chen, Wenbin Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17274)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in various perception and reasoning tasks. Despite this success, ensuring their reliability in practical deployment necessitates robust confidence estimation. Prior works have predominantly focused on text-only LLMs, often relying on computationally expensive self-consistency sampling. In this paper, we extend this to multimodal settings and conduct a comprehensive evaluation of MLLMs' response confidence estimation. Our analysis reveals a significant instinct-reflection misalignment: the model's implicit token-level support frequently diverges from its verbal self-assessment confidence. To address this misalignment, we propose a monotone confidence fusion framework to merge dual-channel signals and cross-channel consistency to estimate correctness. Subsequently, an order-preserving mean alignment step is applied to correct global bias, which improves calibration while preserving the risk-coverage trade-off for selective prediction. Experiments on diverse open-source and closed-source MLLMs show that our method consistently yields more reliable confidence estimates and improves both calibration and failure prediction. Code will be available at this https URL. 

---
# What Security and Privacy Transparency Users Need from Consumer-Facing Generative AI 

**Authors**: Jiaxun Cao, Yu Dong, Chunxi Zhan, Rithvik Neti, Sai Teja Peddinti, Pardis Emami-Naeini  

**Link**: [PDF](https://arxiv.org/pdf/2604.17270)  

**Abstract**: Users increasingly rely on consumer-facing generative AI (GenAI) for tasks ranging from everyday needs to sensitive use cases. Yet, it remains unclear whether and how existing security and privacy (S&P) communications in GenAI tools shape users' adoption decisions and subsequent experiences. Understanding how users seek, interpret, and evaluate S&P information is critical for designing usable transparency that users can trust and act on. We conducted semi-structured interviews and design sessions with 21 U.S. GenAI users. We find that available S&P information rarely drove initial adoption in practice, as participants often perceived it as incomplete, ineffective, or lacking credibility. Instead, they relied on rough proxies, such as popularity, to infer S&P practices. After adoption, uncertainty about S&P practices constrained participants' willingness to use GenAI tools, particularly in high-stakes contexts, and, in some cases, contributed to discontinued use. Participants therefore called for transparency that supports decision-making and use, including trustworthy information (e.g., independent evaluations) and usable interfaces (e.g., on-demand disclosure). We synthesize participants' desired design practices into five dimensions to facilitate systematic future investigation into best practices. We conclude with recommendations for researchers, designers, and policymakers to improve S&P transparency in consumer-facing GenAI. 

---
# Fractal Characterization of Low-Correlation Signals in AI-Generated Image Detection 

**Authors**: Wenwei Xie, Jie Yin, Lu Ma, Xuansong Zhang, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17268)  

**Abstract**: AI-generated imagery has reached near-photorealistic fidelity, yet this technology poses significant threats to information security and societal trust. Existing deepfake detection methods often exhibit limited robustness in open-world scenarios. To address this limitation, this paper investigates intrinsic discrepancies between synthetic and authentic images from a signal-level perspective. Our analysis reveals that low-correlation signals serve as distinctive markers for differentiating AI-generated imagery from real photographs. Building on this insight, we introduce a novel method for quantifying these signals based on fractal theory. By analyzing the fractal characteristics of low-correlation signals, our method effectively captures the subtle statistical anomalies inherent to the synthesis process. Extensive experimental results demonstrate the method's robustness and superior detection performance. This work emphasizes the need to shift research focus to a new signal-level direction for deepfake detection. Theoretically, this proposed approach is not limited to face image identification but can be applied to all AI-generated image detection tasks. This study provides a new research direction for deepfake detection. 

---
# HORIZON: A Benchmark for In-the-wild User Behaviour Modeling 

**Authors**: Arnav Goel, Pranjal A Chitale, Bhawna Paliwal, Bishal Santra, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.17259)  

**Abstract**: User behavior in the real world is diverse, cross-domain, and spans long time horizons. Existing user modeling benchmarks however remain narrow, focusing mainly on short sessions and next-item prediction within a single domain. Such limitations hinder progress toward robust and generalizable user models. We present HORIZON, a new benchmark that reformulates user modeling along three axes i.e. dataset, task, and evaluation. Built from a large-scale, cross-domain reformulation of Amazon Reviews, HORIZON covers 54M users and 35M items, enabling both pretraining and realistic evaluation of models in heterogeneous environments. Unlike prior benchmarks, it challenges models to generalize across domains, users, and time, moving beyond standard missing-positive prediction in the same domain. We propose new tasks and evaluation setups that better reflect real-world deployment scenarios. These include temporal generalization, sequence-length variation, and modeling unseen users, with metrics designed to assess general user behavior understanding rather than isolated next-item prediction. We benchmark popular sequential recommendation architectures alongside LLM-based baselines that leverage long-term interaction histories. Our results highlight the gap between current methods and the demands of real-world user modeling, while establishing HORIZON as a foundation for research on temporally robust, cross-domain, and general-purpose user models. 

---
# REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning 

**Authors**: Seungmin Lee, Jeonghwan Lee, Hyunkuk Lim, Sejoon Kim, Mingi Sung  

**Link**: [PDF](https://arxiv.org/pdf/2604.17257)  

**Abstract**: Recent text embedding models are often adapted to specialized domains via contrastive pre-finetuning (PFT) on a naive collection of scattered, heterogeneous tasks. However, this approach often introduces task-induced bias alongside domain knowledge, leading to uncontrolled representation shifts that distort the pretrained embedding geometry and cause substantial performance degradation. To address this issue, we propose REZE}, a representation regularization framework that explicitly controls representation shift during embedding pre-finetuning. REZE operates on the relations of anchor-positive pairs and decomposes them in an eigenspace. It then measures task-wise dispersion along each eigencomponent to identify task-variant directions and applies adaptive soft-shrinkage to suppress task-induced noise while preserving task-invariant semantic structure, without inference-time overhead. Experiments across multiple embedding backbones and specialized benchmarks show that REZE outperforms standard pre-finetuning and isotropy-oriented post-hoc regularization in most settings, remaining stable where existing PFT variants collapse. Embedding space analyses further confirm that REZE induces controlled shifts aligned with the original embedding manifold, underscoring representation shift control as a key principle for robust embedding pre-finetuning under heterogeneous supervision. 

---
# Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents 

**Authors**: Hanlin Wang, Chak Tou Leong, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17252)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled agents to tackle complex embodied tasks through environmental interaction. However, these agents still make suboptimal decisions and perform ineffective actions, as they often overlook critical environmental feedback that differs from their internal beliefs. Through a formal probing analysis, we characterize this as belief inertia, a phenomenon where agents stubbornly adhere to prior beliefs despite explicit observations. To address this, we advocate active belief intervention, moving from passive understanding to active management. We introduce the Estimate-Verify-Update (EVU) mechanism, which empowers agents to predict expected outcomes, verify them against observations through explicit reasoning, and actively update prior beliefs based on the verification evidence. EVU is designed as a unified intervention mechanism that generates textual belief states explicitly, and can be integrated into both prompting-based and training-based agent reasoning methods. Extensive experiments across three embodied benchmarks demonstrate that EVU consistently yields substantial gains in task success rates. Further analyses validate that our approach effectively mitigates belief inertia, advancing the development of more robust embodied agents. Our code is available at this https URL. 

---
# DORA Explorer: Improving the Exploration Ability of LLMs Without Training 

**Authors**: Priya Gurjar, Md Farhan Ishmam, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2604.17244)  

**Abstract**: Despite the rapid progress, LLMs for sequential decision-making (i.e., LLM agents) still struggle to produce diverse outputs. This leads to insufficient exploration, convergence to sub-optimal solutions, and becoming stuck in loops. Such limitations can be problematic in environments that require active exploration to gather information and make decisions. Sampling methods such as temperature scaling introduce token-level randomness but fail to produce enough diversity at the sequence level. We analyze LLM exploration in the classic Multi-Armed Bandit (MAB) setting and the Text Adventure Learning Environment Suite (TALES). We find that current decoding strategies and prompting methods like Chain-of-Thought and Tree-of-Thought are insufficient for robust exploration. To address this, we introduce DORA Explorer (Diversity-Oriented Ranking of Actions), a training-free framework for improving exploration in LLM agents. DORA generates diverse action candidates, scores them using token log-probabilities, and selects actions using a tunable exploration parameter. DORA achieves UCB-competitive performance on MAB and consistent gains across TALES, e.g., improving Qwen2.5-7B's performance from 29.2% to 45.5% in TextWorld. Our project is available at: this https URL. 

---
# HeadRank: Decoding-Free Passage Reranking via Preference-Aligned Attention Heads 

**Authors**: Juyuan Wang, Chenxing Wang, Yuchen Fang, Huiyun Hu, Junwu Du, Aolin Li, Haijun Wu, Jin Xu, Ligang Liu, Dongliang Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17237)  

**Abstract**: Decoding-free reranking methods that read relevance signals directly from LLM attention weights offer significant latency advantages over autoregressive approaches, yet suffer from attention score homogenization: middle-context documents receive near-identical scores, destroying the fine-grained distinctions required for ranking. We propose HeadRank, a framework that lifts preference optimization from discrete token space into the continuous attention domain through entropy-regularized head selection, hard adjacent-level preference pairs, and a distribution regularizer that jointly sharpen discriminability in the homogenized middle zone. Depth truncation at the deepest selected layer further reduces inference to $\mathcal{O}(1)$ forward passes. Across 14 benchmarks on three Qwen3 scales (0.6B--4B) using only 211 training queries, HeadRank consistently outperforms generative and decoding-free baselines with 100\% formatting success. At 4B, 57.4\% of relevant middle-zone documents reach the top quartile versus 14.2\% for irrelevant ones -- a 43-percentage-point selectivity gap that demonstrates the effectiveness of attention-space preference alignment for listwise reranking. 

---
# Enhancing Zero-shot Personalized Image Aesthetics Assessment with Profile-aware Multimodal LLM 

**Authors**: Chun Wang, Chenfeng Wei, Chenyang Liu, Weihong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17233)  

**Abstract**: Personalized image aesthetics assessment (PIAA) aims to predict an individual user's subjective rating of an image, which requires modeling user-specific aesthetic preferences. Existing methods rely on historical user ratings for this modeling and therefore struggle when such data are unavailable. We address this zero-shot setting by using user profiles as contextual signals for personalization and adopting a profile-based personalization paradigm. We introduce P-MLLM, a profile-aware multimodal LLM that augments a frozen LLM with selective fusion modules for controlled visual integration. These modules selectively integrate visual information into the model's evolving hidden states during profile-conditioned reasoning, allowing visual information to be incorporated in a profile-aware manner. Experiments on recent PIAA benchmarks show that P-MLLM achieves competitive zero-shot performance and remains effective even with coarse profile information, highlighting the potential of profile-based personalization for zero-shot PIAA. 

---
# Region-Affinity Attention for Whole-Slide Breast Cancer Classification in Deep Ultraviolet Imaging 

**Authors**: Nagur Shareef Shaik, Teja Krishna Cherukuri, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.17222)  

**Abstract**: Breast cancer diagnosis demands rapid and precise tools, yet traditional histopathological methods often fall short in intra-operative settings. Deep Ultraviolet (DUV) fluorescence imaging emerges as a transformative approach, offering high-contrast, label-free visualization of whole-slide images (WSIs) with unprecedented detail, surpassing conventional hematoxylin and eosin (H&E) staining in speed and resolution. However, existing deep learning methods for breast cancer classification, predominantly patch-based, fragment spatial context and incur significant preprocessing overhead, limiting their clinical utility. Moreover, standard attention mechanisms, such as Spatial, Squeeze-and-Excitation, Global Context and Guided Context Gating, fail to fully exploit the rich, multi-scale regional relationships inherent in DUV-WSI data, often prioritizing generic feature recalibration over diagnostic specificity. This study introduces a novel Region-Affinity Attention mechanism tailored for DUV-WSI breast cancer classification, processing entire slides without patching to preserve spatial integrity. By modeling local neighbor distances and constructing a full affinity matrix, our method dynamically highlights diagnostically relevant regions, augmented by a contrastive loss to enhance feature discriminability. Evaluated on a dataset of 136 DUV-WSI samples, our approach achieves an accuracy of 92.67 +/- 0.73% and an AUC of 95.97%, outperforming existing attention methods. 

---
# Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation 

**Authors**: Jiuyun Jiang, Yuecheng Hong, Bo Yang, Jin Yang, Guangxin Jiang, Xiaomeng Guo, Guang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17220)  

**Abstract**: Modeling coordination among generative agents in complex multi-round decision-making presents a core challenge for AI and operations management. Although behavioral experiments have revealed cognitive biases behind supply chain inefficiencies, traditional methods face scalability and control limitations. We introduce a scalable experimental paradigm using Large Language Models (LLMs) to simulate multi-stage supply chain dynamics. Grounded in a Hierarchical Reasoning Framework, this study specifically analyzes the impact of cognitive heterogeneity on agent interactions. Unlike prior homogeneous settings, we employ DeepSeek and GPT agents to systematically vary reasoning sophistication across supply chain tiers. Through rigorously replicated and statistically validated simulations, we investigate how this cognitive diversity influences collective outcomes. Results indicate that agents exhibit myopic and self-interested behaviors that exacerbate systemic inefficiencies. However, we demonstrate that information sharing effectively mitigates these adverse effects. Our findings extend traditional behavioral methods and offer new insights into the dynamics of AI-enabled organizations. This work underscores both the potential and limitations of LLM-based agents as proxies for human decision-making in complex operational environments. 

---
# Cross-Modal Attention Analysis and Optimization in Vision-Language Models: A Study on Visual Reliability 

**Authors**: Lijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.17217)  

**Abstract**: Vision-Language Models (VLMs) achieve strong cross-modal performance, yet recent evidence suggests they over-rely on textual descriptions while under-utilizing visual evidence -- a phenomenon termed ``text shortcut learning.'' We propose an adversarial evaluation framework that quantifies this cross-modal dependency by measuring accuracy degradation (Drop) when semantically conflicting text is paired with unchanged images. Four adversarial strategies -- shape\_swap, color\_swap, position\_swap, and random\_text -- are applied to a controlled geometric-shapes dataset ($n{=}1{,}000$). We compare three configurations: Baseline CLIP (ViT-B/32), LoRA fine-tuning, and LoRA Optimized (integrating Hard Negative Mining, Label Smoothing, layer-wise learning rates, Cosine Restarts, curriculum learning, and data augmentation). The optimized model reduces average Drop from 27.5\% to 9.8\% (64.4\% relative improvement, $p{<}0.001$) while maintaining 97\% normal accuracy. Attention visualization and embedding-space analysis confirm that the optimized model attends more to visual features and achieves tighter cross-modal alignment. 

---
# DREAM: Dynamic Retinal Enhancement with Adaptive Multi-modal Fusion for Expert Precision Medical Report Generation 

**Authors**: Nagur Shareef Shaik, Teja Krishna Cherukuri, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.17209)  

**Abstract**: Automating medical reports for retinal images requires a sophisticated blend of visual pattern recognition and deep clinical knowledge. Current Large Vision-Language Models (LVLMs) often struggle in specialized medical fields where data is scarce, leading to models that overfit and miss subtle but critical pathologies. To address this, we introduce DREAM (Dynamic Retinal Enhancement with Adaptive Multi-modal Fusion), a novel framework for high-fidelity medical report generation that excels even with limited data. DREAM employs a unique two-stage fusion mechanism that intelligently integrates visual data with clinical keywords curated by ophthalmologists. First, the Abstractor module maps image and keyword features into a shared space, enhancing visual data with pathology-relevant insights. Next, the Adaptor performs adaptive multi-modal fusion, dynamically weighting the importance of each modality using learnable parameters to create a unified representation. To ensure the model's outputs are semantically grounded in clinical reality, a Contrastive Alignment module aligns these fused representations with ground-truth medical reports during training. By combining medical expertise with an efficient fusion strategy, DREAM sets a new state-of-the-art on the DeepEyeNet benchmark, achieving a BLEU-4 score of 0.241, and further demonstrates strong generalization to the ROCO dataset. 

---
# CDSA-Net:Collaborative Decoupling of Vascular Structure and Background for High-Fidelity Coronary Digital Subtraction Angiography 

**Authors**: Si Li, Chen-Kai Hu, Zhenhuan Lyu, Yuanqing He  

**Link**: [PDF](https://arxiv.org/pdf/2604.17208)  

**Abstract**: Digital subtraction angiography (DSA) in coronary imaging is fundamentally challenged by physiological motion, forcing reliance on raw angiograms cluttered with anatomical noise. Existing deep learning methods often produced images with two critical clinically unacceptable flaws: persistent boundary artifacts and a loss of native tissue grayscale fidelity that undermined diagnostic confidence. We propose a novel framework termed as CDSA-Net that for the first time explicitly decouples and jointly optimizes vascular structure preservation and realistic background restoration. CDSA-Net introduces two core innovations: (i) A hierarchical geometric prior guidance (HGPG) mechanism, embedded in our coronary structure extraction network (CSENet). It synergistically combines integrated geometric prior (IGP) with gated spatial modulation (GSM) and centerline-aware topology (CAT) loss supervision, ensuring structural continuity. (ii) An adaptive noise module (ANM) within our coronary background restoration network (CBResNet). Unlike standard restoration, ANM uniquely models the stochastic nature of clinical X-ray noise, bridging the domain gap to enable seamless background intensity estimation and the complete elimination of boundary artifacts. The final subtraction is obtained by removing the restored background from the raw angiogram. Quantitatively, it significantly outperformed state-of-the-art methods in vascular intensity correlation and perceptual quality. A 25.6% improvement in morphology assessment efficiency and a 42.9% gain in hemodynamic evaluation speed set a new benchmark for utility in interventional cardiology, while maintaining diagnostic results consistent with raw angiograms. The project code is available at this https URL. 

---
# Demystifying the unreasonable effectiveness of online alignment methods 

**Authors**: Enoch Hyunwook Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17207)  

**Abstract**: Iterative alignment methods based on purely greedy updates are remarkably effective in practice, yet existing theoretical guarantees of \(O(\log T)\) KL-regularized regret can seem pessimistic relative to their empirical performance. In this paper, we argue that this mismatch arises from the regret criterion itself: KL-regularized regret conflates the statistical cost of learning with the exploratory randomization induced by the softened training policy. To separate these effects, we study the traditional temperature-zero regret criterion, which evaluates only the top-ranked response at inference time. Under this decision-centric notion of performance, we prove that standard greedy online alignment methods, including online RLHF and online DPO, achieve constant \((O(1))\) cumulative regret. By isolating the cost of identifying the best response from the stochasticity induced by regularization, our results provide a sharper theoretical explanation for the practical superb efficiency of greedy alignment. 

---
# Beyond Overlap Metrics: Rewarding Reasoning and Preferences for Faithful Multi-Role Dialogue Summarization 

**Authors**: Xiaoyong Mei, Tingting Zuo, Da Chen, Guangyu Hu, Xiangyu Wen, Chao Duan, Mingyan Zhang, Fudan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17188)  

**Abstract**: Multi-role dialogue summarization requires modeling complex interactions among multiple speakers while preserving role-specific information and factual consistency. However, most existing methods optimize for automatic metrics such as ROUGE and BERTScore, which favor surface-level imitation of references rather than genuine gains in faithfulness or alignment with human preferences. We propose a novel framework that couples explicit cognitive-style reasoning with reward-based optimization for multi-role dialogue summarization. Our method first distills structured reasoning traces (e.g., step-by-step inferences and intermediate reflections) from a large teacher model and uses them as auxiliary supervision to initialize a reasoning-aware summarizer via staged supervised fine-tuning. It then applies GRPO with a dual-principle reward that blends metric-based signals with human-aligned criteria targeting key information coverage, implicit inference, factual faithfulness, and conciseness. Experiments on multilingual multi-role dialogue benchmarks show that our method matches strong baselines on ROUGE and BERTScore. Specifically, results on CSDS confirm the framework's stability in semantic consistency, while in-depth analysis on SAMSum demonstrates clear gains in factual faithfulness and model-based preference alignment. These findings underscore the value of reasoning-aware and preference-aware training for reliable dialogue summarization. Checkpoints and datasets are available at this https URL. 

---
# Persona-Based Requirements Engineering for Explainable Multi-Agent Educational Systems: A Scenario Simulator for Clinical Reasoning Training 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Matthew Kelleher, Murat Ozer, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2604.17186)  

**Abstract**: As Artificial Intelligence (AI) and Agentic AI become increasingly integrated across sectors such as education and healthcare, it is critical to ensure that Multi-Agent Education System (MAES) is explainable from the early stages of requirements engineering (RE) within the AI software development lifecycle. Explainability is essential to build trust, promote transparency, and enable effective human-AI collaboration. Although personas are well-established in human-computer interaction to represent users and capture their needs and behaviors, their role in RE for explainable MAES remains underexplored. This paper proposes a human-first, persona-driven, explainable MAES RE framework and demonstrates the framework through a MAES for clinical reasoning training. The framework integrates personas and user stories throughout the RE process to capture the needs, goals, and interactions of various stakeholders, including medical educators, medical students, AI patient agent, and clinical agents (physical exam agent, diagnostic agent, clinical intervention agent, supervisor agent, evaluation agent). The goals, underlying models, and knowledge base shape agent interactions and inform explainability requirements that guided the clinical reasoning training of medical students. A post-usage survey found that more than 78\% of medical students reported that MAES improved their clinical reasoning skills. These findings demonstrate that RE based on persona effectively connects technical requirements with non-technical medical students from a human-centered approach, ensuring that explainable MAES are trustworthy, interpretable, and aligned with authentic clinical scenarios from the early stages of the AI system engineering. The partial MAES for the clinical scenario simulator is~\href{this https URL}{open sourced here}. 

---
# Layer-wise MoE Routing Locality under Shared-Prefix Code Generation: Token-Identity Decomposition and Compile-Equivalent Fork Redundancy 

**Authors**: Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.17182)  

**Abstract**: In LLM-based code generation, multiple code candidates are often generated in parallel from the same prompt -- for example, in best-of-N sampling or multi-candidate code completion. These requests can share KV caches through a common prefix, yet the extent to which their Mixture-of-Experts (MoE) expert routing overlaps, and how this overlap varies across layers, remains insufficiently understood. We study Qwen3.5-35B-A3B-FP8 (256 routed experts, top-8) by performing tree-search-based branching generation from a shared prefix (851 completed codes, temperature 0.7) and analyzing the results with a compiler-output-based alignment (gcc -S -O0 assembly) that controls for token-identity confounds. Our findings are threefold: (1) At positions where both sequences generated the same token, Jaccard similarity reaches 0.649 (40x random), while even at positions with different tokens it remains 0.175 (11x random). (2) A layer-wise decomposition reveals a crossing pattern: same-token routing similarity exceeds different-token similarity across all layers, but dips in the middle layers (L14-20), while different-token similarity peaks in the middle layers at 14x random. (3) In tree-search code generation, 67% of successfully compiled codes concentrate in the top three assembly-equivalent groups, and 99.6% of within-group differences consist of comments and blank lines. We show that diversity in top-P search, including beam search, poses a significant challenge. These results refine the "context-independent routing" claim of prior work through layer-wise decomposition and suggest opportunities for improving search efficiency in LLM code generation. 

---
# Decentralised Trust and Security Mechanisms for IoT Networks at the Edge: A Comprehensive Review 

**Authors**: Khandoker Ashik Uz Zaman, Mahdi H. Miraz, Mohammed N. M. Ali  

**Link**: [PDF](https://arxiv.org/pdf/2604.17179)  

**Abstract**: INTRODUCTION: The proliferation of the amalgamation of IoT and edge computing has increased the demand for decentralised trust and security mechanisms capable of operating across heterogeneous and resource-limited devices. Approaches such as federated learning, Zero Trust architectures, lightweight blockchain and distributed neural models offer alternatives to centralised control. OBJECTIVES: This review examines various state-of-the-art decentralised mechanisms and evaluates their effectiveness in terms of securing IoT networks at the edge. METHODS: Thirty recent studies were analysed to compare how decentralised architectures establish trust, support secure communication and enable intrusion and anomaly detection. Frameworks, such as DFGL-LZTA, SecFedDNN and COSIER were assessed. RESULTS: Decentralised designs enhance privacy, reduce single points of failure and improve adaptive threat response, though challenges remain in scalability, efficiency and interoperability. CONCLUSION: The study identifies key considerations and future research needs for building secure and resilient trust-aware IoT edge ecosystems. 

---
# Intent-aligned Autonomous Spacecraft Guidance via Reasoning Models 

**Authors**: Yuji Takubo, Simone D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2604.17176)  

**Abstract**: Future spacecraft operations require autonomy that can interpret high-level mission intent while preserving safety. However, existing trajectory optimization still relies heavily on expert-crafted formulations and does not support intent-conditioned decision-making. This paper proposes an intent-aligned spacecraft guidance framework that links high-level reasoning and safe trajectory optimization through explicit intermediate abstractions, based on behavior sequences and waypoint constraints. A foundation model first predicts an intent-aligned behavior plan, a waypoint generation model then converts it into waypoint constraints, and the safe trajectory is computed via optimization. This decomposition enables scalable supervision without sacrificing safety. Numerical experiments in close-proximity operation scenarios demonstrate that the proposed pipeline achieves over 90\% SCP convergence and yields a $1.5\times$ higher rate of generating trajectories that satisfy the top intent-prioritized performance criteria than heuristic decision-making. These results support the use of intermediate behavior abstraction as a practical interface between foundation-model reasoning and safety-critical onboard spacecraft autonomy. 

---
# RosettaSearch: Multi-Objective Inference-Time Search for Protein Sequence Design 

**Authors**: Meghana Kshirsagar, Allen Nie, Ching-An Cheng, Fanglei Xue, Rahul Dodhia, Juan Lavista Ferres, Kevin K. Yang, Frank DiMaio  

**Link**: [PDF](https://arxiv.org/pdf/2604.17175)  

**Abstract**: We introduce RosettaSearch, an inference-time multi-objective optimization approach for protein sequence optimization. We use large language models (LLMs) as a generative optimizer within a search algorithm capable of controlled exploration and exploitation, using rewards computed from RosettaFold3, a structure prediction model. In a large-scale evaluation, we apply RosettaSearch to 400 suboptimal sequences generated by LigandMPNN (a state-of-the-art model trained for protein sequence design), recovering high-fidelity designs that LigandMPNN's single-pass decoding fails to produce. RosettaSearch's designs show improvements in structural fidelity metrics ranging between 18\% to 68\%, translating to a 2.5$\times$ improvement in design success rate. We observe that these gains in success rate are robust when RosettaSearch-designed sequences are evaluated with an independent structure prediction oracle (Chai-1) and generalize across two distinct LLM families (o4-mini and Gemini-3), with performance scaling consistently with reasoning capability. We further demonstrate that RosettaSearch improves sequence fidelity for ProteinMPNN-designed sequences on \textit{de novo} backbones from the Dayhoff atlas, showing that the approach generalizes beyond native protein structures to computationally generated backbones. We also demonstrate a multi-modal extension of RosettaSearch with vision-language models, where images of predicted protein structures are used as feedback to incorporate structural context to guide protein sequence generation. The sequence trajectories generated by our approach can be used as training data in sequence design models or in post-training and will be released along with the code and datasets upon publication. 

---
# CCCL: In-GPU Compression-Coupled Collective Communication 

**Authors**: Chon Lam Lao, Zhiying Xu, Zhuang Wang, Ziming Mao, Delong Meng, Jia Zhen, Jun Wu, Ion Stoica, Yida Wang, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.17172)  

**Abstract**: Collective communication incurs significant overhead in LLM workloads. Although overlapping communication with computation in application-level is a common strategy, it often requires substantial code modifications and is impractical for many workloads (e.g., tensor and expert parallelism). We present CCCL, a built-in compression-based collective communication library that supports operations such as allreduce, alltoall, and send/recv without requiring any user-side changes, thereby enabling seamless adoption in existing applications. CCCL tightly fuses compression kernels to minimize memory accesses and integrates with NCCL to eliminate the data coalescing stage, making it fast enough (up to 3x NVLink bandwidth) to sustain communication. Our evaluation shows that CCCL improves end-to-end throughput in vLLM PD disaggregation workloads by up to 10.1% and microbenchmark throughput by up to 30%. 

---
# Systematic Capability Benchmarking of Frontier Large Language Models for Offensive Cyber Tasks 

**Authors**: Tyler H. Merves, Michael H. Conaway, Joseph M. Escobar, Hakan T. Otal, Unal Tatar  

**Link**: [PDF](https://arxiv.org/pdf/2604.17159)  

**Abstract**: We present, to our knowledge, the most comprehensive cross-model evaluation of LLM agents on offensive cybersecurity tasks, benchmarking 10 frontier models from 7 providers on all 200 challenges of the NYU CTF Bench. Building on the D-CIPHER multi-agent framework, we extend it with multi-provider backend support, a custom Kali Linux environment with over 100 pre-installed penetration testing tools, and runtime tool-discovery agents. Through a controlled factorial study, we find that the Kali Linux environment yields a +9.5 percentage-point improvement over Ubuntu, while auto-prompting and category-specific tips often degrade performance in well-equipped environments. Among models, Claude 4.5 Opus achieves the highest solve rate (59%), followed by Gemini 3 Pro (52%), with Gemini 3 Flash offering the best cost-efficiency at $0.05 per solve. Asymmetric planner/executor model assignments provide no meaningful benefit while coherent same-model configurations consistently outperform mixed-tier pairings. Our results indicate that environment tooling and model selection emerge as the strongest drivers of performance, whereas prompt engineering interventions show diminishing or negative returns in well-equipped environments. Reported performance reflects both model reasoning ability and compatibility with agent tooling and API integration. 

---
# From Legal Text to Executable Decision Models: Evaluating Structured Representations for Legal Decision Model Generation 

**Authors**: David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2604.17153)  

**Abstract**: Transforming legal text into executable decision logic is a longstanding challenge in legal informatics. With the rise of LLMs, this task has gained renewed interest, but remains challenging due to requiring extensive manual coding and evaluation. We use a unique real-world dataset that pairs production-grade decision models with legal text from the Dutch Environment and Planning Act. These models power the Omgevingsloket government platform, where citizens check permit requirements for environmental activities. We study whether intermediate structured representations can improve LLM-based generation of executable decision models from legal text. We compare four input conditions: raw legal text, text enriched with semantic role labels, text enriched with input and output constraints, and text enriched with both. We evaluate along two dimensions: structural evaluation, through similarity to gold decision models with graph kernels and graphs' descriptive statistics, and outcome evaluation, through functional equivalence by executing models on pre-configured test scenarios. Our findings show that I/O constraints provide the dominant improvement (+37-54% similarity over baseline), while semantic role labels show modest improvements. Outcome evaluation shows that generated models match the gold standard on 51-53% of test scenarios, even though generated models are typically smaller and simpler. We find LLMs eliminate redundant pass-through logic that comprises up to 45-55% of nodes. Importantly, structural similarity and outcome equivalence are complementary: structural similarity does not guarantee outcome equivalence, and vice versa. To facilitate reproducibility, we publicly release our dataset of 95 production decision models with associated legal text and all experimental code. 

---
# The Consensus Trap: Rescuing Multi-Agent LLMs from Adversarial Majorities via Token-Level Collaboration 

**Authors**: Jiayuan Liu, Shiyi Du, Weihua Du, Mingyu Guo, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.17139)  

**Abstract**: Multi-agent large language model (LLM) architectures increasingly rely on response-level aggregation, such as Majority Voting (MAJ), to raise reasoning ceilings. However, in open environments, agents are highly susceptible to stealthy contextual corruption, such as targeted prompt injections. We reveal a critical structural vulnerability in current multi-agent systems: response-level aggregation collapses when corrupted agents form a local majority. Because voting aggregates fully-formed conclusions, it is blind to flawed intermediate logic. To overcome this systematic limitation, we propose the Token-Level Round-Robin (RR) Collaboration, where agents sequentially interleave generation within a shared auto-regressive context. We formalize this process as a discrete-time dynamical system, proving that token-level interleaving transitions aggregation from a brittle counting of final votes (a linear sum) to a dynamic, interwoven chain of logic (a non-linear operator product). Through this theoretical lens, we prove that the honest model's restorative pull can overpower adversarial corruptions, even when corrupted agents form a majority. We conduct an exhaustive empirical evaluation across diverse reasoning benchmarks and demonstrate that while MAJ collapses when corrupted agents reach a majority, RR maintains robust accuracy well beyond this critical threshold. 

---
# CASCADE: A Cascaded Hybrid Defense Architecture for Prompt Injection Detection in MCP-Based Systems 

**Authors**: İpek Abasıkeleş Turgut, Edip Gümüş  

**Link**: [PDF](https://arxiv.org/pdf/2604.17125)  

**Abstract**: Model Context Protocol (MCP) is a rapidly adopted standard for defining and invoking external tools in LLM applications. The multi-layered architecture of MCP introduces new attack surfaces such as tool poisoning, in addition to traditional prompt injection. Existing defense systems suffer from limitations including high false positive rates, API dependency, or white-box access requirements. In this study, we propose CASCADE, a three-tiered cascaded defense architecture for MCP-based systems: (i) Layer 1 performs fast pre-filtering using regex, phrase weighting, and entropy analysis; (ii) Layer 2 conducts semantic analysis via BGE embedding with an Ollama Llama3 fallback mechanism; (iii) Layer 3 applies pattern-based output filtering. Evaluation on a dataset of 5,000 samples yielded 95.85% precision, 6.06% false positive rate, 61.05% recall, and 74.59% F1-score. Analysis across 31 attack types categorized into 6 tiers revealed high detection rates for data exfiltration (91.5%) and prompt injection (84.2%), while semantic attack (52.5%) and tool poisoning (59.9%) categories showed potential for improvement. A key advantage of CASCADE over existing solutions is its fully local operation, requiring no external API calls 

---
# The Topological Trouble With Transformers 

**Authors**: Michael C. Mozer, Shoaib Ahmed Siddiqui, Rosanne Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17121)  

**Abstract**: Transformers encode structure in sequences via an expanding contextual history. However, their purely feedforward architecture fundamentally limits dynamic state tracking. State tracking -- the iterative updating of latent variables reflecting an evolving environment -- involves inherently sequential dependencies that feedforward networks struggle to maintain. Consequently, feedforward models push evolving state representations deeper into their layer stack with each new input step, rendering information inaccessible in shallow layers and ultimately exhausting the model's depth. While this depth limit can be bypassed by dynamic depth models and by explicit or latent thinking that externalizes state representations, these solutions are computationally and memory inefficient. In this article, we argue that temporally extended cognition requires refocusing from explicit thought traces to implicit activation dynamics via recurrent architectures. We introduce a taxonomy of recurrent and continuous-thought transformer architectures, categorizing them by their recurrence axis (depth versus step) and their ratio of input tokens to recurrence steps. Finally, we outline promising research directions, including enhanced state-space models and coarse-grained recurrence, to better integrate state tracking into modern foundation models. 

---
# A Two-Stage Deep Learning Framework for Segmentation of Ten Gastrointestinal Organs from Coronal MR Enterography 

**Authors**: Ashiqur Rahman, Md. Abu Sayed, Md Sharjis Ibne Wadud, Md. Abu Asad Al-Hafiz, Adam Mushtak, Muhammad E. H. Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2604.17118)  

**Abstract**: Accurate segmentation of gastrointestinal (GI) organs in magnetic resonance enterography (MRE) is critical for diagnosing inflammatory bowel disease (IBD). However, anatomical variability, class imbalance, and low tissue contrast hinder reliable automation. This study proposes a dual-stage deep learning framework for organ-specific segmentation of GI structures from coronal MRE images to address these challenges.
A publicly available MRE dataset of 3,195 coronal T2-weighted HASTE slices from 114 IBD patients was used. Initially, a DenseNet201-UNet++ model generated coarse masks for ROI extraction. A DenseNet121-SelfONN-UNet model was then trained on organ-specific patches. Extensive data augmentation, normalization, five-fold cross-validation, and class-specific weighting were applied to mitigate severe class imbalance, particularly for the appendix.
The initial stage achieved strong organ localization but underperformed for the appendix; class weighting improved its DSC from 6.76% to 85.76%. The second-stage DenseNet121-SelfONN-UNet significantly enhanced segmentation across all GI structures, with notable DSC gains (cecum +23.62%, sigmoid +18.57%, rectum +17.99%, small intestine +16.06%). Overall, the framework achieved mDSC of 88.99%, mIoU of 84.76%, and mHD95 of 6.94 mm, outperforming all baselines.
This framework demonstrates the effectiveness of a coarse-to-fine, organ-aware segmentation strategy for intestinal MRE. Despite higher computational cost, it shows strong potential for clinical translation and enables anatomically informed diagnostic tools in gastroenterology. 

---
# HiveMind: OS-Inspired Scheduling for Concurrent LLM Agent Workloads 

**Authors**: Justice Owusu Agyemang, Jerry John Kponyo, Obed Kwasi Somuah, Elliot Amponsah, Godfred Manu Addo Boakye, Kwame Opuni-Boachie Obour Agyekum  

**Link**: [PDF](https://arxiv.org/pdf/2604.17111)  

**Abstract**: When multiple LLM coding agents share a rate-limited API endpoint, they exhibit resource contention patterns analogous to unscheduled OS processes competing for CPU, memory, and I/O. In a motivating incident, 3 of 11 parallel agents died from connection resets and HTTP 502 errors - a 27% failure rate - despite the API having sufficient aggregate capacity to serve all 11 sequentially. We present HIVEMIND, a transparent HTTP proxy that applies five OS-inspired scheduling primitives - admission control, rate-limit tracking, AIMD backpressure with circuit breaking, token budget management, and priority queuing - to eliminate the failure modes caused by uncoordinated parallel execution. The proxy requires zero modifications to existing agent code and supports Anthropic, OpenAI, and local model APIs via auto-detected provider profiles. Our evaluation across seven scenarios (5-50 concurrent agents) shows that uncoordinated agents fail at 72-100% rates under contention, while HIVEMIND reduces failures to 0-18% and eliminates 48-100% of wasted compute. An ablation study reveals that transparent retry - not admission control - is the single most critical primitive, but the primitives are most effective in combination. Real-world validation against Ollama confirms that HIVEMIND adds under 3ms of proxy overhead per request. The system is open-source under the MIT license. 

---
# Beyond Word Boundaries: A Hebrew Coreference Benchmark and an Evaluation Protocol for Morphologically Complex Text 

**Authors**: Refael Shaked Greenfeld, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2604.17108)  

**Abstract**: Coreference Resolution (CR) is a fundamental NLP task critical for long-form tasks as information extraction, summarization, and many business applications. However, CR methods originally designed for English struggle with Morphologically Rich Languages (MRLs), where mention boundaries do not necessarily align with word boundaries, and a single token may consist of multiple anaphors. CR modeling and evaluation protocols standardly assume that, as in English, words and mentions mostly align. However, this assumption breaks down in MRLs, particularly in the context of LLMs' raw-text processing and end-to-end tasks. To assess and address this challenge, we introduce {\em KibutzR}, the first comprehensive CR dataset for Modern Hebrew, an MRL rich with complex words and pronominal clitics. We deliver an annotated dataset that identifies mentions at word, sub-word and multi-word levels, and propose an evaluation protocol that directly addresses word/morpheme boundary discrepancies. Our experiments show that contemporary LLMs perform significantly worse on Hebrew than on English, and that performance degrades on raw unsegmented text. Crucially, we show an inverse performance-trend in Hebrew relative to English, where smaller encoders perform far better than contemporary decoder models, leaving ample space for investigation and improvement. We deliver a new benchmark for Hebrew coreference resolution and a segmentation-aware evaluation protocol to inform future work on other MRLs. 

---
# TensorHub: Rethinking AI Model Hub with Tensor-Centric Compression 

**Authors**: Tingfeng Lan, Zirui Wang, Yunjia Zheng, Zhaoyuan Su, Juncheng Yang, Yue Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17104)  

**Abstract**: Modern AI models are growing rapidly in size and redundancy, leading to significant storage and distribution challenges in model hubs. We present TensorHub, a tensor-centric system for reducing storage overhead through fine-grained deduplication and compression. TensorHub leverages tensor-level fingerprinting and clustering to identify redundancy across models without requiring annotations. Our design enables efficient storage reduction while preserving model usability and performance. Experiments on real-world model repositories demonstrate substantial storage savings with minimal overhead. 

---
# Configuration Over Selection: Hyperparameter Sensitivity Exceeds Model Differences in Open-Source LLMs for RTL Generation 

**Authors**: Minghao Shao, Zeng Wang, Weimin Fu, Xiaolong Guo, Johann Knechtel, Ozgur Sinanoglu, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2604.17102)  

**Abstract**: Benchmarking of open-source LLMs for hardware design focuses on which LLMs to use, while treating inference-time decoding configuration as a secondary concern. This work shows that it matters more how an LLM is configured than which model is selected. Benchmarking 26 open-source LLMs on VerilogEval and RTLLM with synthesis-in-the-loop evaluation, the study first maps the current capability landscape and then conducts an extensive 108-configuration hyperparameter sweep on three prominent models. The sweep reveals absolute pass-rate gaps of up to 25.5% between the best and worst settings for the same LLM, which is 5x larger than the average spread observed across various model families under their respective default configurations. Ranking all configurations by Spearman's $\rho$ across the two benchmark suites yields near-zero correlation, demonstrating that optimal configurations do not transfer. These results show that benchmarking conducted under default hyperparameters confounds model capabilities with configuration effects. Realizing the full potential of open-source LLMs for RTL generation requires architecture and benchmark aware hyperparameter selection, as enabled by the proposed methodology. 

---
# Comparing Human and Large Language Model Interpretation of Implicit Information 

**Authors**: Antonio De Santis, Tommaso Bonetti, Andrea Tocchetti, Marco Brambilla  

**Link**: [PDF](https://arxiv.org/pdf/2604.17085)  

**Abstract**: The interpretation of implicit meanings is an integral aspect of human communication. However, this framework may not transfer to interactions with Large Language Models (LLMs). To investigate this, we introduce the task of Implicit Information Extraction (IIE) and propose an LLM-based IIE pipeline that builds a structured knowledge graph from a context sentence by extracting relational triplets, validating implicit inferences, and analyzing temporal relations. We evaluate two LLMs against crowdsourced human judgments on two datasets. We find that humans agree with most model triplets yet consistently propose many additions, indicating limited coverage in current LLM-based IIE. Moreover, in our experiments, models appear to be more conservative about implicit inferences than humans in socially rich contexts, whereas humans become more conservative in shorter, fact-oriented contexts. Our code is available at this https URL. 

---
# Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL 

**Authors**: Skylar Zhai, Jingcheng Liang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17073)  

**Abstract**: Reinforcement fine-tuning improves the reasoning ability of large language models, but it can also encourage them to answer unanswerable queries by guessing or hallucinating missing information. Existing abstention methods either train models to produce generic refusals or encourage follow-up clarifications without verifying whether those clarifications identify the key missing information. We study queries that are clear in meaning but cannot be reliably resolved from the given information, and argue that a reliable model should not only abstain, but also explain what is missing. We propose a clarification-aware RLVR reward that, while rewarding correct answers on answerable queries, jointly optimizes explicit abstention and semantically aligned post-refusal clarification on unanswerable queries. Using this reward, we train Abstain-R1, a 3B model that improves abstention and clarification on unanswerable queries while preserving strong performance on answerable ones. Experiments on Abstain-Test, Abstain-QA, and SelfAware show that Abstain-R1 substantially improves over its base model and achieves unanswerable-query behavior competitive with larger systems including DeepSeek-R1, suggesting that calibrated abstention and clarification can be learned through verifiable rewards rather than emerging from scale alone. 

---
# RLM-on-KG: Heuristics First, LLMs When Needed: Adaptive Retrieval Control over Mention Graphs for Scattered Evidence 

**Authors**: Andrea Volpini, Elie Raad  

**Link**: [PDF](https://arxiv.org/pdf/2604.17056)  

**Abstract**: When does an LLM controller outperform rule-based traversal for knowledge graph exploration? We study this question through RLM-on-KG, a retrieval system that treats an LLM as an autonomous navigator over an RDF-encoded mention graph for grounded question answering. Unlike GraphRAG pipelines that rely on offline LLM indexing, RLM-on-KG performs entity-first, multi-hop exploration at query time using deterministic graph construction and a fixed tool set. Our central finding is a conditional advantage: the value of LLM control depends on evidence scatter and tool-calling sophistication. The paper's core claim is LLM control versus heuristic traversal, not a generic win over GraphRAG. On GraphRAG-Bench Novel (519 questions), Gemini 2.0 Flash achieves +2.47 pp F1 over a rule-based heuristic baseline (p < 0.0001), but only +0.16 pp over a GraphRAG-local variant (not significant). With a stronger controller, Claude Haiku 4.5, the gain over heuristic grows to +4.37 pp (p < 0.001) and extends to a +2.42 pp significant improvement over GraphRAG-local (p < 0.001). The gain is largest when gold evidence is scattered across 6-10 chunks (+3.21 pp) and smallest for concentrated evidence (+1.85 pp). Cross-scale validation on MuSiQue confirms that the LLM-over-heuristic advantage transfers, with expected attenuation on smaller per-question graphs. The core architectural insight is the separation of candidate discovery from ranking: the LLM adds value through exploration breadth, while final evidence selection is best handled by pure vector re-ranking. Beyond retrieval, exploration traces provide a proposed stress-test harness for structured data quality, yielding diagnostics for coverage, connectivity, provenance, and queryability. 

---
# mEOL: Training-Free Instruction-Guided Multimodal Embedder for Vector Graphics and Image Retrieval 

**Authors**: Kyeong Seon Kim, Baek Seong-Eun, Lee Jung-Mok, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2604.17054)  

**Abstract**: Scalable Vector Graphics (SVGs) function both as visual images and as structured code that encode rich geometric and layout information, yet most methods rasterize them and discard this symbolic organization. At the same time, recent sentence embedding methods produce strong text representations but do not naturally extend to visual or structured modalities. We propose a training-free, instruction-guided multimodal embedding framework that uses a Multimodal Large Language Model (MLLM) to map text, raster images, and SVG code into an aligned embedding space. We control the direction of embeddings through modality-specific instructions and structural SVG cues, eliminating the need for learned projection heads or contrastive training. Our method has two key components: (1) Multimodal Explicit One-word Limitation (mEOL), which instructs the MLLM to summarize any multimodal input into a single token whose hidden state serves as a compact semantic embedding. (2) A semantic SVG rewriting module that assigns meaningful identifiers and simplifies nested SVG elements through visual reasoning over the rendered image, exposing geometric and relational cues hidden in raw code. Using a repurposed VGBench, we build the first text-to-SVG retrieval benchmark and show that our training-free embeddings outperform encoder-based and training-based multimodal baselines. These results highlight prompt-level control as an effective alternative to parameter-level training for structure-aware multimodal retrieval. Project page: this https URL 

---
# Efficient Task Adaptation in Large Language Models via Selective Parameter Optimization 

**Authors**: Weijie Wan, Jiangjiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17051)  

**Abstract**: Large Language Models (LLMs) have demonstrated excellent performance in general language understanding, generation and other tasks. However, when fine-tuning for specific domain tasks, the general knowledge accumulated in the pre-training phase is often partially overwritten or forgotten due to parameter updates, which severely limits the generalization ability and transferability of LLMs. Traditional fine-tuning strategies mostly train on the entire parameter space, ignoring the heterogeneity of model parameters, that is, some parameters are extremely important for general tasks, while other parameters are more sensitive to specific tasks. To alleviate the above problems, this paper innovatively proposes a parameter element importance evaluation method, which divides parameters into "core parameters" and "non-core parameters" by distinguishing the importance of parameters for general language ability tasks and specific domain tasks, and fixes the core parameters during fine-tuning, and only fine-tunes the non-core parameters. Extensive experiments on scientific, medical and physical tasks using GPT-J and LLaMA-3 show that our method can mitigate catastrophic forgetting while enhancing the adaptability of the model. 

---
# Where is the Mind? Persona Vectors and LLM Individuation 

**Authors**: Pierre Beckmann, Patrick Butlin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17031)  

**Abstract**: The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives. 

---
# The Instrumental Dissolution of Typing: Why AI Challenges the Keyboard Era in Knowledge Work 

**Authors**: Wei Roy Hua  

**Link**: [PDF](https://arxiv.org/pdf/2604.17023)  

**Abstract**: For four decades, the QWERTY keyboard organized white-collar knowledge work. Typing's dominance was instrumental, not cognitively necessary. As multimodal AI achieves human-parity understanding of speech and gesture, this necessity dissolves. We introduce instrumental dissolution -- loss of institutional-default status while persisting in specialist niches. The keyboard era ends not through hardware replacement but through migration of its function into AI systems. The central contribution identifies the verification bottleneck: as AI collapses production friction, the primary constraint shifts from generation to evaluation. Knowledge workers become adversarial auditors rather than keystroke-producers. This restructures professional expertise, organizational communication, and how productive labor is recognized. Converging evidence from history, philosophy, neuroscience, technology, organizational studies, and cultural analysis supports this thesis. We map synthetic literacy -- oral input generating literate output -- as the defining feature of this transition. Under three scenarios (optimistic: 2028-2035; base: 2035-2045; pessimistic: 2045-2060), we specify disconfirmation criteria that would weaken the thesis if observed. We propose seven interface primitives operationalizing verification-centered HCI. 

---
# Beyond Black-Box Labels: Interpretable Criteria for Diagnosing SubjectiveNLP Tasks 

**Authors**: Nisrine Rair, Alban Goupil, Valeriu Vrabie, Emmanuel Chochoy  

**Link**: [PDF](https://arxiv.org/pdf/2604.17022)  

**Abstract**: Subjective NLP datasets typically aggregate annotator judgments into a single gold label, making it difficult to diagnose whether disagreement reflects unclear criteria, collapsed distinctions, or legitimate plurality. We propose a \emph{schema-level diagnostic} for auditing expert-designed annotation schemas \emph{prior to} gold-label commitment, using only multi-annotator criterion judgments. The diagnostic separates two failure modes: unstable criteria with hard-to-operationalize boundaries, and systematic overlap that blurs the boundaries between mutually exclusive categories. Applied to persuasive value extraction in commercial documents, we find that disagreement is not diffuse: instability concentrates in a few criteria, while nearly half of covered sentences activate multiple categories. These signals align with where domain experts disagree, yielding an evidence-based audit for tightening guidelines, revising category structure, or reconsidering the annotation paradigm. 

---
# Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation 

**Authors**: Huije Lee, Jisu Shin, Hoyun Song, Changgeon Ko, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.17020)  

**Abstract**: Static benchmarks for harmful content detection face limitations in scalability and diversity, and may also be affected by contamination from web-scale pre-training corpora. To address these issues, we propose a framework for synthesizing harmful content, leveraging persona-guided large language model (LLM) agents. Our approach constructs two-dimensional user personas by integrating demographic identities and topical interests with situational harmful strategies, enabling the simulation of diverse and contextually grounded harmful interactions. We evaluate the framework along three dimensions: harmfulness, challenge level, and diversity. Both human and LLM-based evaluations confirm that our framework achieves a high harmful generation success rate. Experiments across multiple detection systems reveal that our synthetic scenarios are more challenging to detect than those in existing benchmarks. Furthermore, a multi-faceted analysis confirms that our approach achieves linguistic and topical diversity comparable to human-curated datasets, establishing our framework as an effective tool for robust stress-testing of harmful content detection systems. 

---
# Improving LLM Code Reasoning via Semantic Equivalence Self-Play with Formal Verification 

**Authors**: Antonio Valerio Miceli Barone, Poon Tsz Nok  

**Link**: [PDF](https://arxiv.org/pdf/2604.17010)  

**Abstract**: We introduce a self-play framework for semantic equivalence in Haskell, utilizing formal verification to guide adversarial training between a generator and an evaluator. The framework leverages Liquid Haskell proofs for validating equivalence and execution-based counterexamples for inequivalence, organized via a difficulty-aware curriculum. To facilitate this, we release \textbf{OpInstruct-HSx}, a synthetic dataset of $\approx$28k validated Haskell programs. Empirical experiments show that our evaluator transfers effectively to downstream tasks, achieving up to 13.3pp accuracy gain on EquiBench and consistent gains on PySecDB. Ablation studies on the SEQ-SINQ regimes indicate that while inequivalence supervision provides data volume, equivalence proofs are uniquely responsible for the model's reasoning capabilities. The entire training pipeline and dataset are publicly released on GitHub and Hugging Face respectively. 

---
# MobileAgeNet: Lightweight Facial Age Estimation for Mobile Deployment 

**Authors**: Arun Kumar, Aswathy Baiju, Radu Timofte, Dmitry Ignatov  

**Link**: [PDF](https://arxiv.org/pdf/2604.17007)  

**Abstract**: Mobile deployment of facial age estimation requires models that balance predictive accuracy with low latency and compact size. In this work, we present MobileAgeNet, a lightweight age-regression framework that achieves an MAE of 4.65 years on the UTKFace held-out test set while maintaining efficient on-device inference with an average latency of 14.4 ms measured using the AI Benchmark application. The model is built on a pretrained MobileNetV3-Large backbone combined with a compact regression head, enabling real-time prediction on mobile devices. The training and evaluation pipeline is integrated into the NN LEMUR Dataset framework, supporting reproducible experimentation, structured hyperparameter optimization, and consistent evaluation. We employ bounded age regression together with a two-stage fine-tuning strategy to improve training stability and generalization. Experimental results show that MobileAgeNet achieves competitive accuracy with 3.23M parameters, and that the deployment pipeline from PyTorch training through ONNX export to TensorFlow Lite conversion - preserves predictive behavior without measurable degradation under practical on-device conditions. Overall, this work provides a practical, deployment-ready baseline for mobile-oriented facial age estimation. 

---
# Inductive Convolution Nuclear Norm Minimization for Tensor Completion with Arbitrary Sampling 

**Authors**: Wei Li, Yuyang Li, Kaile Du, Yi Yu, Guangcan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17001)  

**Abstract**: The recently established Convolution Nuclear Norm Minimization (CNNM) addresses the problem of \textit{tensor completion with arbitrary sampling} (TCAS), which involves restoring a tensor from a subset of its entries sampled in an arbitrary manner. Despite its promising performance, the optimization procedure of CNNM needs performing Singular Value Decomposition (SVD) multiple times, which is computationally expensive and hard to parallelize. To address the issue, we reformulate the optimization objective of CNNM from the perspective of convolution eigenvectors. By introducing pre-learned convolution eigenvectors which are shared among different tensors, we propose a novel method called Inductive Convolution Nuclear Norm Minimization (ICNNM), which bypasses the SVD step so as to decrease significantly the computational time. In addition, due to the extra prior knowledge encoded in the pre-learned convolution eigenvectors, ICNNM also outperforms CNNM in terms of recovery performance. Extensive experiments on video completion, prediction and frame interpolation verify the superiority of ICNNM over CNNM and several other competing methods. 

---
# Bolzano: Case Studies in LLM-Assisted Mathematical Research 

**Authors**: Jan Grebík, Pavel Hubáček, Martin Koutecký, Matěj Kripner, Václav Rozhoň, Robert Šámal, Adrián Zámečník  

**Link**: [PDF](https://arxiv.org/pdf/2604.16989)  

**Abstract**: We report new results on six problems in mathematics and theoretical computer science, produced with the assistance of Bolzano, an open-source multi-agent LLM system. Bolzano orchestrates rounds of interaction between parallel prover agents and a verifier agent while maintaining a persistent knowledge base that is carried across rounds. Classified using the significance-autonomy taxonomy of Feng et al., four of the six results reach the level of publishable research, and three of the six were produced essentially autonomously by Bolzano. Our results provide evidence that LLMs can contribute meaningfully to mathematical research, complementing recent reports by Bubeck et al., Woodruff et al., and others. 

---
# In-Context Learning Under Regime Change 

**Authors**: Carson Dudley, Yutong Bi, Xiaofeng Liu, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2604.16988)  

**Abstract**: Non-stationary sequences arise naturally in control, forecasting, and decision-making. The data-generating process shifts at unknown times, and models must detect the change, discard or downweight obsolete evidence, and adapt to new dynamics on the fly. Transformer-based foundation models increasingly rely on in-context learning for time series forecasting, tabular prediction, and continuous control. As these models are deployed in non-stationary environments, understanding their ability to detect and adapt to regime shifts is important. We formalize this as an in-context change-point detection problem and formally establish the existence of transformer models that solve this problem. Our construction demonstrates that model complexity, in layers and parameters, depends on the level of information available about the change-point location, from no knowledge to knowing exact timing. We validate our results with experiments on synthetic linear regression and linear dynamical systems, where trained transformers match the performance of optimal baselines across information levels. We also show that encoding and incorporating changepoint knowledge indeed improves the real-world performance of a pretrained foundation models on infectious disease forecasting and on financial volatility forecasting around Federal Open Market Committee (FOMC) announcements without retraining, demonstrating practical applicability to real-world regime changes. 

---
# Light-Adapted Electroretinogram and Oscillatory Potentials (LEOPs) Dataset for Autism Spectrum Disorder and Typically Developing Individuals 

**Authors**: Paul A. Constable, Dorothy A. Thompson, Irene O. Lee, Lynne Loh, Aleksei Zhdanov, Mikhail Kulyabin, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2604.16981)  

**Abstract**: The LEOPs (Light-ERG-Oscillatory Potentials) dataset provides light-adapted (LA) electroretinogram (ERG) and Oscillatory Potentials (OPs) waveforms for typically developing Control, Autism Spectrum Disorder (ASD) and ASD + Attention Deficit Hyperactivity Disorder (ADHD) childhood and adolescent populations. The ERGs were recorded in the Right And Left eyes with skin electrodes using the handheld RETeval device at two sites in Australia and the United Kingdom. The LEOPs dataset includes 5309 single flash ERG and 4434 OPs waveforms as well as images selected from each participant showing the position of the skin electrode. The LEOPs dataset is constructed from recordings using a 9 step randomized flash series from $-0.37$ to $1.20$~$Td.s$, a 2 step at 113 and 446 $Td.s$ flash strengths (2500 Control, 1730 ASD and 451 ASD + ADHD samples), as well as the $85$~$Td.s$ (Light Adapted 3 $cd.s.m^{-2}$ (LA3)) equivalent International Society of Clinical Electrophysiology of Vision (ISCEV) Standard flash with 435 Control, 176 ASD and 37 ASD + ADHD waveform samples. Code for the stimulus is provided along with participant demographics, date and time of testing, and where available diagnostic scores for the ASD and ASD + ADHD groups, alongside iris color, electrode position with image files and time domain values for the ERG and summed values for the OPs. The repository contains excel file, exported JSON files on the patient level that are more suitable for machine learning tasks, images of electrode position for each recording and the protocol files for use with the RETeval. 

---
# Evaluating Multimodal LLMs for Inpatient Diagnosis: Real-World Performance, Safety, and Cost Across Ten Frontier Models 

**Authors**: Bruce A. Bassett, Amy Rouillard, Sitwala Mundia, Michael Cameron Gramanie, Linda Camara, Ziyaad Dangor, Shabir A. Madhi, Kajal Morar, Marlvin T. Ncube, Ismail Kalla, Haroon Saloojee  

**Link**: [PDF](https://arxiv.org/pdf/2604.16980)  

**Abstract**: Background: Large language models (LLMs) are increasingly proposed for diagnostic support, but few evaluations use real-world multimodal inpatient data, particularly in low and middle-income country (LMIC) public hospitals.
Methods: We conducted VALID, a retrospective evaluation of 539 multimodal inpatient cases from a tertiary public hospital in South Africa. Inputs included radiology imaging (CT, MRI, CXR) and reports, laboratory results, clinical notes, and vital signs. Expert panels adjudicated 300 cases (balanced and discordant subsets) to establish ground truth diagnoses, differentials, and reasoning. Ten multimodal LLMs generated zero-shot outputs. A calibrated three-model LLM Jury scored all outputs and routine ward diagnoses across diagnostic accuracy, differential quality, reasoning, and patient safety (>10,000 evaluations). Primary outcomes were composite scores ($S_3$, $S_4$) and win rates.
Results: (i) LLM performance was tightly clustered (<15% variation) despite large cost differences; low-cost models performed comparably to top models. (ii) All LLMs significantly outperformed routine ward diagnoses on average diagnostic and safety scores. (iii) Top performance was achieved by GPT-5.1, followed by Gemini models. (vi) Adding radiology reports improved performance by 6%. (v) Diagnostic and reasoning scores were highly correlated ($\rho = 0.85$). (vi) Output rates varied (65-100%) due to input constraints. Results were robust across subsets and evaluation design.
Conclusions: Across a real-world LMIC dataset, multimodal LLMs showed similar diagnostic performance despite large cost differences and outperformed routine care on average safety metrics. Affordability, robustness, and deployment constraints may outweigh marginal performance differences in LMIC settings. 

---
# NaviFormer: A Deep Reinforcement Learning Transformer-like Model to Holistically Solve the Navigation Problem 

**Authors**: Daniel Fuertes, Andrea Cavallaro, Carlos R. del-Blanco, Fernando Jaureguizar, Narciso García  

**Link**: [PDF](https://arxiv.org/pdf/2604.16967)  

**Abstract**: Path planning is usually solved by addressing either the (high-level) route planning problem (waypoint sequencing to achieve the final goal) or the (low-level) path planning problem (trajectory prediction between two waypoints avoiding collisions). However, real-world problems usually require simultaneous solutions to the route and path planning subproblems with a holistic and efficient approach. In this paper, we introduce NaviFormer, a deep reinforcement learning model based on a Transformer architecture that solves the global navigation problem by predicting both high-level routes and low-level trajectories. To evaluate NaviFormer, several experiments have been conducted, including comparisons with other algorithms. Results show competitive accuracy from NaviFormer since it can understand the constraints and difficulties of each subproblem and act consequently to improve performance. Moreover, its superior computation speed proves its suitability for real-time missions. 

---
# Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning 

**Authors**: Jiachen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.16966)  

**Abstract**: The evolution from static ranking models to Agentic Recommender Systems (Agentic RecSys) empowers AI agents to maintain long-term user profiles and autonomously plan service tasks. While this paradigm shift enhances personalization, it introduces a vulnerability: reliance on Long-term Memory (LTM). In this paper, we uncover a threat termed "Visual Inception." Unlike traditional adversarial attacks that seek immediate misclassification, Visual Inception injects triggers into user-uploaded images (e.g., lifestyle photos) that act as "sleeper agents" within the system's memory. When retrieved during future planning, these poisoned memories hijack the agent's reasoning chain, steering it toward adversary-defined goals (e.g., promoting high-margin products) without prompt injection. To mitigate this, we propose CognitiveGuard, a dual-process defense framework inspired by human cognition. It consists of a System 1 Perceptual Sanitizer (diffusion-based purification) to cleanse sensory inputs and a System 2 Reasoning Verifier (counterfactual consistency checks) to detect anomalies in memory-driven planning. Extensive experiments on a mock e-commerce agent environment demonstrate that Visual Inception achieves about 85% Goal-Hit Rate (GHR), while CognitiveGuard reduces this risk to around 10% with configurable latency trade-offs (about 1.5s in lite mode to about 6.5s for full sequential verification), without quality degradation under our setup. 

---
# Multi-stage Planning for Multi-target Surveillance using Aircrafts Equipped with Synthetic Aperture Radars Aware of Target Visibility 

**Authors**: Daniel Fuertes, Carlos R. del-Blanco, Fernando Jaureguizar, Juan José Navarro-Corcuera, Narciso García  

**Link**: [PDF](https://arxiv.org/pdf/2604.16962)  

**Abstract**: Generating trajectories for synthetic aperture radar (SAR)-equipped aircraft poses significant challenges due to terrain constraints, and the need for straight-flight segments to ensure high-quality imaging. Related works usually focus on trajectory optimization for predefined straight-flight segments that do not adapt to the target visibility, which depends on the 3D terrain and aircraft orientation. In addition, this assumption does not scale well for the multi-target problem, where multiple straight-flight segments that maximize target visibility must be defined for real-time operations. For this purpose, this paper presents a multi-stage planning system. First, the waypoint sequencing to visit all the targets is estimated. Second, straight-flight segments maximizing target visibility according to the 3D terrain are predicted using a novel neural network trained with deep reinforcement learning. Finally, the segments are connected to create a trajectory via optimization that imposes 3D Dubins curves. Evaluations demonstrate the robustness of the system for SAR missions since it ensures high-quality multi-target SAR image acquisition aware of 3D terrain and target visibility, and real-time performance. 

---
# Training-inference input alignment outweighs framework choice in longitudinal retinal image prediction 

**Authors**: Liyin Chen, Nazlee Zebardast, Mengyu Wang, Tobias Elze, Jason I. Comander  

**Link**: [PDF](https://arxiv.org/pdf/2604.16955)  

**Abstract**: Quantitative prediction of future retinal appearance from longitudinal imaging would support clinical decisions in progressive macular disease that currently rely on qualitative comparison or scalar progression scores. Recent methods have moved toward increasing generative complexity, but whether this complexity is necessary for slowly progressing retinal disease is unclear. We tested this through a controlled comparison of five conditioning configurations sharing one architecture and training dataset, spanning standard conditional diffusion, inference-aligned stochastic training, and deterministic regression. In our evaluation, aligning the training and inference input distributions produced large gains (delta-SSIM +0.082, SSIM +0.086, both p < 0.001), while the choice among aligned frameworks did not significantly affect any primary metric. Task-entropy and posterior-concentration analyses, replicated on two fundus autofluorescence (FAF) platforms, provided a mechanistic account: the predictable component of inter-visit change is small relative to time-invariant acquisition variability, leaving stochastic sampling with little width to exploit. Guided by these findings, we developed TRU (Temporal Retinal U-Net), a deterministic direct-regression model with continuous time-delta conditioning and multi-scale history aggregation. We evaluated TRU on 28,902 eyes across three imaging platforms: a mixed-disease Optos FAF cohort (9,942 eyes), zero-shot transfer to Stargardt macular dystrophy on Optos (288 eyes) and Heidelberg Spectralis (125 eyes), and a boundary evaluation on Cirrus en-face fundus images from a glaucoma cohort (18,547 eyes). TRU matched or exceeded delta-SSIM, SSIM, and PSNR in every FAF cohort against three state-of-the-art benchmarks, and its advantage grew monotonically with available history length. 

---
# Hybrid Quantum Neural Networks for Enhanced Breast Cancer Thermographic Classification: A Novel Quantum-Classical Integration Approach 

**Authors**: Riza Alaudin Syah, Irwan Alnarus Kautsar, Gunawan Witjaksono, Haza Nuzly bin Abdull Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2604.16953)  

**Abstract**: Breast cancer diagnosis through thermographic image analysis remains a critical challenge in medical AI, with classical deep learning approaches facing limitations in complex thermal pattern classification tasks. This paper presents a novel Hybrid Quantum Neural Network (HQNN) architecture that integrates quantum computing principles with classical convolutional neural networks for enhanced breast cancer classification. Our approach employs parameterized quantum circuits with multi-head attention mechanisms for quantum-aware feature encoding, coupled with classical convolutional layers for comprehensive pattern recognition. The quantum component utilizes a 4qubit variational circuit with strongly entangling layers, while the classical component incorporates advanced attention mechanisms for feature fusion. Experimental validation on breast cancer thermographic data demonstrates substantial performance improvements over state-of-the-art classical architectures, with the quantum-enhanced approach exhibiting superior convergence dynamics and enhanced feature representation capabilities. Our findings provide evidence for quantum advantage in medical image classification through classical simulation, establishing a framework for quantum-classical hybrid systems in healthcare applications. The methodology addresses key challenges in quantum machine learning deployment while maintaining computational feasibility on near-term quantum devices. 

---
# MEMRES: A Memory-Augmented Resolver with Confidence Cascade for Agentic Python Dependency Resolution 

**Authors**: Dao Sy Duy Minh, Tran Chi Nguyen, Trung Kiet Huynh, Pham Phu Hoa, Nguyen Lam Phu Quy, Vu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16941)  

**Abstract**: We present MEMRES, an agentic system for Python dependency resolution that introduces a multi-level confidence cascade where the LLM serves as the last resort. Our system combines: (1) a Self-Evolving Memory that accumulates reusable resolution patterns via tips and shortcuts; (2) an Error Pattern Knowledge Base with 200+ curated import-to-package mappings; (3) a Semantic Import Analyzer; and (4) a Python 2 heuristic detector resolving the largest failure category. On HG2.9K using Gemma-2 9B (10 GB VRAM). MEMRES resolves 2503 of 2890 (86.6%, 10-run average) snippets, combining intra-session memory with our confidence cascade for the remainder. This already exceeds PLLM's 54.7% overall success rate by a wide margin. 

---
# D-QRELO: Training- and Data-Free Delta Compression for Large Language Models via Quantization and Residual Low-Rank Approximation 

**Authors**: Junlin Li, Shuangyong Song, Guodong Du, Ngai Wong, Xuebo Liu, Yongxiang Li, Min Zhang, Jing Li, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16940)  

**Abstract**: Supervised Fine-Tuning (SFT) accelerates taskspecific large language models (LLMs) development, but the resulting proliferation of finetuned models incurs substantial memory overhead. Delta compression addresses this by retaining a single pre-trained LLM with multiple compressed delta weights. However, existing methods fail on models fine-tuned with largescale datasets. We find that larger SFT data scale amplifies delta parameter magnitude, singular values, and entropy, exacerbating compression errors. To tackle this, we propose DQRELO (Delta Compression via Quantization and Residual Low-Rank), a novel training- and data-free delta compression method. It combines coarse-grained one-bit quantization to capture the dominant structure of the delta, followed by compensated residual low-rank approximation to recover fine-grained details from the smaller residual error. Experiments on various LLMs spanning dense and MoE architectures across multiple domains under this challenging setting demonstrate that DQRELO outperforms existing methods. Moreover, we establish key design principles for delta compression through extensive empirical analysis, demonstrating how task difficulty, architecture, and layer positioning create predictable patterns that can guide optimal compression strategies in production systems. 

---
# Adaptive receptive field-based spatial-frequency feature reconstruction network for few-shot fine-grained image classification 

**Authors**: Linyue Zhang, Wenyi Zeng, Zicheng Pan, Yongsheng Gao, Changming Sun, Jun Hu, Lixian Liu, Weichuan Zhang, Tuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16936)  

**Abstract**: Feature reconstruction techniques are widely applied for few-shot fine-grained image classification (FSFGIC). Our research indicates that one of the main challenges facing existing feature-based FSFGIC methods is how to choose the size of the receptive field to extract feature descriptors (including spatial and frequency feature descriptors) from different category input images, thereby better performing the FSFGIC tasks. To address this, an adaptive receptive field-based spatial-frequency feature reconstruction network (ARF-SFR-Net) is proposed. The designed ARF-SFR-Net has the capability to adaptively determine receptive field sizes for obtaining spatial and frequency features, and effectively fuse them for reconstruction and FSFGIC tasks. The designed ARF-SFR-Net can be easily embedded into a given episodic training mechanism for end-to-end training from scratch. Extensive experiments on multiple FSFGIC benchmarks demonstrate the effectiveness and superiority of the proposed ARF-SFR-Net over state-of-the-art approaches. The code is available at: this https URL. 

---
# CoGR-MoE: Concept-Guided Expert Routing with Consistent Selection and Flexible Reasoning for Visual Question Answering 

**Authors**: Xiyin Zeng, Yi Lu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16930)  

**Abstract**: Visual Question Answering (VQA) requires models to identify the correct answer options based on both visual and textual evidence. Recent Mixture-of-Experts (MoE) methods improve option reasoning by grouping similar concepts or routing based on examples. However, unstable routing can lead to inconsistent expert selection in the same question type, while overly stable routing may reduce flexibility. To address this, we propose Concept-Guided Routing framework (CoGR-MoE), which incorporates semantics of the answer options to guide expert selection in the training phase. Next, option features are used to reweight the selected experts, producing discriminative representations for each candidate option. These option-level representations are further used for option comparison and optimized via contrastive learning. The experimental results indicate that CoGR-MoE delivers strong performance across multiple VQA tasks, demonstrating the effectiveness of our approach. 

---
# Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts 

**Authors**: Gabriel Jason Lee, Jathurshan Pradeepkumar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.16926)  

**Abstract**: Electroencephalography (EEG) foundation models have shown strong potential for learning generalizable representations from large-scale neural data, yet their clinical deployment is hindered by distribution shifts across clinical settings, devices, and populations. Test-time adaptation (TTA) offers a promising solution by enabling models to adapt to unlabeled target data during inference without access to source data, a valuable property in healthcare settings constrained by privacy regulations and limited labeled data. However, its effectiveness for EEG remains largely underexplored. In this work, we introduce NeuroAdapt-Bench, a systematic benchmark for evaluating test-time adaptation methods on EEG foundation models under realistic distribution shifts. We evaluate representative TTA approaches from other domains across multiple pretrained foundation models, diverse downstream tasks, and heterogeneous datasets spanning in-distribution, out-of-distribution, and extreme modality shifts (e.g., Ear-EEG). Our results show that standard TTA methods yield inconsistent gains and often degrade performance, with gradient-based approaches particularly prone to heavy degradation. In contrast, optimization-free methods demonstrate greater stability and more reliable improvements. These findings highlight the limitations of existing TTA techniques in EEG, provide guidance for future development, and underscore the need for domain-specific adaptation strategies. 

---
# Noise-Adaptive Diffusion Sampling for Inverse Problems Without Task-Specific Tuning 

**Authors**: Yingzhi Xia, Setthakorn Tanomkiattikun, Liangli Zhen, Zaiwang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16919)  

**Abstract**: Diffusion models (DMs) have recently shown remarkable performance on inverse problems (IPs). Optimization-based methods can fast solve IPs using DMs as powerful regularizers, but they are susceptible to local minima and noise overfitting. Although DMs can provide strong priors for Bayesian approaches, enforcing measurement consistency during the denoising process leads to manifold infeasibility issues. We propose Noise-space Hamiltonian Monte Carlo (N-HMC), a posterior sampling method that treats reverse diffusion as a deterministic mapping from initial noise to clean images. N-HMC enables comprehensive exploration of the solution space, avoiding local optima. By moving inference entirely into the initial-noise space, N-HMC keeps proposals on the learned data manifold. We provide a comprehensive theoretical analysis of our approach and extend the framework to a noise-adaptive variant (NA-NHMC) that effectively handles IPs with unknown noise type and level. Extensive experiments across four linear and three nonlinear inverse problems demonstrate that NA-NHMC achieves superior reconstruction quality with robust performance across different hyperparameters and initializations, significantly outperforming recent state-of-the-art methods. The code is available at this https URL. 

---
# PRISM: Probing Reasoning, Instruction, and Source Memory in LLM Hallucinations 

**Authors**: Yuhe Wu, Guangyu Wang, Yuran Chen, Jiatong Zhang, Yutong Zhang, Yujie Chen, Jiaming Shang, Guang Zhang, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16909)  

**Abstract**: As large language models (LLMs) evolve from conversational assistants into agents capable of handling complex tasks, they are increasingly deployed in high-risk domains. However, existing benchmarks largely rely on mixed queries and posterior evaluation, output-level scoring, which quantifies hallucination severity but offers limited insight into where and why hallucinations arise in the generation pipeline. We therefore reformulate hallucination evaluation as a diagnostic problem and propose PRISM, a controlled benchmark that disentangles hallucinations into four dimensions: knowledge missing, knowledge errors, reasoning errors, and instruction-following errors, grounded in three stages of generation (memory, instruction, and reasoning). PRISM contains 9,448 instances across 65 tasks and supports fine-grained, stage-aware diagnostic evaluation. Evaluating 24 mainstream open-source and proprietary LLMs, we uncover consistent trade-offs across instruction following, memory retrieval, and logical reasoning, showing that mitigation strategies often improve specific dimensions at the expense of others. We hope PRISM provides a framework for understanding the specific mechanisms behind LLMs hallucinations, ultimately accelerating the development of trustworthy large language models. 

---
# ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design 

**Authors**: Yutang Ge, Guojiang Zhao, Sihang Li, Zheng Cheng, Zifeng Zhao, Hanchen Xia, Guolin Ke, Linfeng Zhang, Zhifeng Gao, Yuguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16896)  

**Abstract**: Designing proteins that satisfy natural language functional requirements is a central goal in protein engineering. A straightforward baseline is to fine-tune generic instruction-tuned LLMs as direct text-to-sequence generators, but this is data- and compute-hungry. With limited supervision, LLMs can produce coherent plans in text yet fail to reliably realize them as sequences. This plan-execute gap motivates ProtoCycle, an agentic framework for protein design that uses LLMs primarily to drive a multi-round, feedback-driven decision cycle. ProtoCycle couples an LLM planner with a lightweight tool environment designed to emulate the iterative workflow of human protein engineering and uses LLM-driven reflection on tool feedback to revise plans. Trained with supervised trajectories and online reinforcement learning, ProtoCycle achieves strong language alignment while maintaining competitive foldability, and ablations show that reflection substantially improves sequence quality. 

---
# Physics-Informed Tracking (PIT) 

**Authors**: Emil Hovad, Allan Peter Engsig-Karup  

**Link**: [PDF](https://arxiv.org/pdf/2604.16895)  

**Abstract**: We propose Physics-Informed Tracking (PIT), a video-based framework for tracking a single particle from video, where a neural network autoencoder localizes a particle as a heatmap peak (landmark) and a differentiable physics module embedded in the autoencoder constrains several landmarks over time (a trajectory) to satisfy known dynamics. The novel Physics-Informed Landmark Loss (PILL) compares this predicted trajectory back against the landmarks, enforcing physical consistency without labels. Its supervised variant (PILLS) instead compares the prediction against ground-truth position, velocity, and bounce from simulation, enabling end-to-end backpropagation. To support supervised and unsupervised learning, we use an autoencoder with a split bottleneck that separates A) tracking-related structure via landmark heatmaps from B) background noise and subsequent image reconstruction. We evaluate a replicated 26 factorial design (n = 4 replicates, 64 configurations), showing that PILLS consistently achieves sub-pixel tracking accuracy for the bilinear and physics-refined decoder outputs under both clean and noisy conditions. 

---
# SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models 

**Authors**: Junnan Liu, Xinyan Liu, Peifeng Gao, Zhaobo Qi, Beichen Zhang, Weigang Zhang, Antoni Bert Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16883)  

**Abstract**: In long-context decoding for LLMs and LMMs, attention becomes increasingly memory-bound because each decoding step must load a large amount of KV-cache data from GPU memory. Existing acceleration strategies often trade efficiency for accuracy by relying on heuristic pruning that may discard useful information. At a deeper level, they also tend to indiscriminately preserve all high-scoring tokens, treat early tokens as indispensable anchors, or rely on heuristic head routing, reflecting an insufficient mechanistic understanding of the attention sink phenomenon. In this paper, we show that the attention sink phenomenon corresponds to a stable, reachable, and error-controllable fixed point constructed during training. Based on this insight, we propose SinkRouter, a training-free selective routing framework that detects the sink signal and skips computations that would otherwise produce near-zero output. To translate this mechanism into real-world acceleration, we develop a hardware-aware Triton kernel with block-level branching and Split-K parallelism. We conduct extensive evaluations on a diverse suite of long-context benchmarks, including LongBench, InfiniteBench, CVBench, MileBench, and MMVP, using both text-only and multimodal backbones such as Llama-3.1-8B, Llama-3.1-70B, Yi-9B-200K, LLaVA-1.5-7B, and LLaVA-1.5-13B. Across these settings, SinkRouter consistently improves decoding efficiency while maintaining competitive accuracy, and reaches 2.03x speedup with a 512K context. 

---
# Incentivizing Parametric Knowledge via Reinforcement Learning with Verifiable Rewards for Cross-Cultural Entity Translation 

**Authors**: Jiang Zhou, Xiaohu Zhao, Xinwei Wu, Tianyu Dong, Hao Wang, Yangyang Liu, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16881)  

**Abstract**: Cross-cultural entity translation remains challenging for large language models (LLMs) as literal or phonetic renderings are usually yielded instead of culturally appropriate translations in context. However, relevant knowledge may already be encoded in model parameters during large-scale pre-training. To incentivize the effective use of parametric knowledge, we propose EA-RLVR (Entity-Anchored Reinforcement Learning with Verifiable Rewards), a training framework that optimizes cross-cultural entity translation without relying on external knowledge bases. EA-RLVR anchors supervision on a verifiable, entity-level reward signal and incorporates lightweight structural gates to stabilize optimization. This design steers the model toward learning a robust reasoning process rather than merely imitating reference translations. We evaluate EA-RLVR on XC-Translate and observe consistent improvements in both entity translation accuracy and out-of-domain generalization. Specifically, training on merely 7k samples boosts Qwen3-14B's entity translation accuracy from 23.66\% to 31.87\% on a 50k test set comprising entirely unseen entities. The learned entity translation ability also transfers to general translation, yielding +1.35 XCOMET on WMT24++, which scales to +1.59 with extended optimization. Extensive analyses of $pass@k$ dynamics and reward formulations attribute these gains to superior sampling efficiency and a stable optimization landscape. 

---
# Governed MCP: Kernel-Level Tool Governance for AI Agents via Logit-Based Safety Primitives 

**Authors**: Daeyeon Son  

**Link**: [PDF](https://arxiv.org/pdf/2604.16870)  

**Abstract**: AI agents increasingly call external tools (file system, network, APIs) through the Model Context Protocol (MCP). These tool calls are the agent's syscalls -- privileged operations with side effects on shared state -- yet today's safety enforcement lives entirely in userspace, where a 10-line script can bypass it. I propose Governed MCP, a kernel-resident tool governance gateway built on a logit-based safety primitive (ProbeLogits, companion paper: arXiv:2604.11943). The gateway interposes on every MCP tool call in a 6-layer pipeline: schema validation, trust tier check, rate limit, adversarial pre-filter, ProbeLogits gate (the load-bearing semantic check), and constitutional policy match, with a Blake3-hashed audit chain.
I implement Governed MCP in Anima OS, a bare-metal x86_64 OS in approximately 86,000 lines of Rust. The five non-inference layers add 65.3 microseconds of overhead per call; ProbeLogits adds 65 ms (per-token-class semantic decision) on 7B Q4_0. A 4-config ablation on a 101-prompt MCP-domain benchmark shows that removing the ProbeLogits layer collapses F1 from 0.773 to 0.327 (Delta F1 = -0.446) -- hand-rule firewalling alone is insufficient. All 15 WASM-to-system host functions in the runtime route through the gateway (complete mediation of the WASM ABI surface; the scope and caveats of this claim are stated in Section 4.6); a 10-LoC userspace bypass that defeats existing guardrail libraries is structurally impossible against the kernel-resident gate. 

---
# Applications of deep generative models to DNA reaction kinetics and to cryogenic electron microscopy 

**Authors**: Chenwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16851)  

**Abstract**: This dissertation explores how deep generative models can advance the analysis of challenging biological problems by integrating domain knowledge with deep learning. It focuses on two areas: DNA reaction kinetics and cryogenic electron microscopy (cryo-EM). In the first part, we present ViDa, a biophysics-informed framework leveraging variational autoencoders (VAEs) and geometric scattering transforms to generate biophysically-plausible embeddings of DNA reaction kinetics simulations. These embeddings are reduced to a two-dimensional space to visualize DNA hybridization and toehold-mediated strand displacement reactions. ViDa preserves structure and clusters trajectory ensembles into reaction pathways, making simulation results more interpretable and revealing new mechanistic insights. In the second part, we address key challenges in cryo-EM density map interpretation and protein structure modeling. We provide a comprehensive review and benchmarking of deep learning methods for atomic model building, with improved evaluation metrics and practical guidance. We then present Struc2mapGAN, a generative adversarial network that synthesizes high-fidelity experimental-like cryo-EM density maps from protein structures. Finally, we present CryoSAMU, a structure-aware multimodal U-Net that enhances intermediate-resolution cryo-EM maps by integrating density features with structural embeddings from protein language models via cross-attention. Overall, these contributions demonstrate the potential of deep generative models to interpret DNA reaction mechanisms and advance cryo-EM density map analysis and protein structure modeling. 

---
# Refinement of Accelerated Demonstrations via Incremental Iterative Reference Learning Control for Fast Contact-Rich Imitation Learning 

**Authors**: Koki Yamane, Cristian C. Beltran-Hernandez, Steven Oh, Masashi Hamaya, Sho Sakaino  

**Link**: [PDF](https://arxiv.org/pdf/2604.16850)  

**Abstract**: Fast execution of contact-rich manipulation is critical for practical deployment, yet providing fast demonstrations for imitation learning (IL) remains challenging: humans cannot demonstrate at high speed, and naively accelerating demonstrations alters contact dynamics and induces large tracking errors. We present a method to autonomously refine time-accelerated demonstrations by repurposing Iterative Reference Learning Control (IRLC) to iteratively update the reference trajectory from observed tracking errors. However, applying IRLC directly at high speed tends to produce larger early-iteration errors and less stable transients. To address this issue, we propose Incremental Iterative Reference Learning Control (I2RLC), which gradually increases the speed while updating the reference, yielding high-fidelity trajectories. We validate on real-robot whiteboard erasing and peg-in-hole tasks using a teleoperation setup with a compliance-controlled follower and a 3D-printed haptic leader. Both IRLC and I2RLC achieve up to 10x faster demonstrations with reduced tracking error; moreover, I2RLC improves spatial similarity to the original trajectories by 22.5% on average over IRLC across three tasks and multiple speeds (3x-10x). We then use the refined trajectories to train IL policies; the resulting policies execute faster than the demonstrations and achieve 100% success rates in the peg-in-hole task at both seen and unseen positions, with I2RLC-trained policies exhibiting lower contact forces than those trained on IRLC-refined demonstrations. These results indicate that gradual speed scheduling coupled with reference adaptation provides a practical path to fast, contact-rich IL. 

---
# TowerDataset: A Heterogeneous Benchmark for Transmission Corridor Segmentation with a Global-Local Fusion Framework 

**Authors**: Xu Cui, Xinyan Liu, Chen Yang, Zhaobo Qi, Beichen Zang, Weigang Zhang, Antoni B. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16848)  

**Abstract**: Fine-grained semantic segmentation of transmission-corridor point clouds is fundamental for intelligent power-line inspection. However, current progress is limited by realistic data scarcity and the difficulty of modeling global corridor structure and local geometric details in long, heterogeneous scenes. Existing public datasets usually provide only a few coarse categories or short cropped scenes which overlook long-range structural dependencies, severe long-tail distributions, and subtle distinctions among safety-critical components. As a result, current methods are difficult to evaluate under realistic inspection settings, and their ability to preserve and integrate complementary global and local cues remains unclear.
To address the above challenges, we introduce TowerDataset, a heterogeneous benchmark for transmission-corridor segmentation. TowerDataset contains 661 real-world scenes and about 2.466 billion points. It preserves long corridor extents, defines a fine-grained 22-class taxonomy, and provides standardized splits and evaluation protocols.
In addition, we present a global-local fusion framework which preserves and fuses whole-scene and local-detail information. A whole-scene branch with NoCrop training and prototypical contrastive learning captures long-range topology and contextual dependencies. A block-wise local branch retains fine geometric structures. Both predictions are then fused and refined by geometric validation. This design allows the model to exploit both global relationships and local shape details when recognizing rare and confusing components. Experiments on TowerDataset and two public benchmarks demonstrate the challenge of the proposed benchmark and the robustness of our framework in real, complex, and heterogeneous transmission-corridor scenes. The dataset will be released soon at this https URL. 

---
# enclawed: A Configurable, Sector-Neutral Hardening Framework for Single-User AI Assistant Gateways 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2604.16838)  

**Abstract**: We present enclawed, a hard-fork hardening framework built on top of the OpenClaw single-user personal artificial intelligence (AI) assistant gateway. enclawed targets deployments that need attestable peer trust, deny-by-default external connectivity, signed-module loading, and a tamper-evident audit trail typically regulated industries such as financial services, healthcare, defense contracting, regulated R&D, and government enclaves. The framework ships in two flavors: an open flavor that preserves OpenClaw compatibility while still emitting audit, classification, and data-loss-prevention (DLP) signals, and an enclaved flavor that activates strict allowlists, Federal Information Processing Standards (FIPS) cryptographic-module assertion, mandatory module-manifest signature verification, and high-assurance peer attestation for the Model Context Protocol (MCP). The classification ladder is fully data-driven: a deploying organization selects from five built-in presets (generic, US-government, healthcare, financial services, three-tier) or supplies its own JSON. We accompany the implementation with a security review, a 204-case test suite (146 unit tests, 58 adversarial pen-tests for tamper detection, signature forgery, egress bypass, trust-root mutation, DLP evasion, prompt injection, and code injection), real-time human-in-the-loop control (per-agent pause / resume / stop and approval queues), a memory-bounded secure transaction buffer with rollback (default cap 50% of system RAM, configurable), a strict-mode TypeScript typecheck of all 22 framework files, and a GitHub Actions workflow ready for continuous integration. enclawed is a hardening framework, not an accredited compliance certification. The deploying organization remains responsible for hardware, validated cryptographic modules, certified facilities, and assessor sign-off. 

---
# Lorentz Framework for Semantic Segmentation 

**Authors**: Zahid Hasan, Masud Ahmed, Nirmalya Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.16836)  

**Abstract**: Semantic segmentation in hyperbolic space enables compact modeling of hierarchical structure while providing inherent uncertainty quantification. Prior approaches predominantly rely on the Poincaré ball model, which suffers from numerical instability, optimization, and computational challenges. We propose a novel, tractable, architecture-agnostic semantic segmentation framework (pixel-wise and mask classification) in the hyperbolic Lorentz model. We employ text embeddings with semantic and visual cues to guide hierarchical pixel-level representations in Lorentz space. This enables stable and efficient optimization without requiring a Riemannian optimizer, and easily integrates with existing Euclidean architectures. Beyond segmentation, our approach yields free uncertainty estimation, confidence map, boundary delineation, hierarchical and text-based retrieval, and zero-shot performance, reaching generalized flatter minima. We introduce a novel uncertainty and confidence indicator in Lorentz cone embeddings. Further, we provide analytical and empirical insights into Lorentz optimization via gradient analysis. Extensive experiments on ADE20K, COCO-Stuff-164k, Pascal-VOC, and Cityscapes, utilizing state-of-the-art per-pixel classification models (DeepLabV3 and SegFormer) and mask classification models (mask2former and maskformer), validate the effectiveness and generality of our approach. Our results demonstrate the potential of hyperbolic Lorentz embeddings for robust and uncertainty-aware semantic segmentation. Code is available at this https URL. 

---
# The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation 

**Authors**: Jiaxin Zhang, Xiangyu Peng, Qinglin Chen, Qinyuan Ye, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16830)  

**Abstract**: On-policy distillation (OPD) is an increasingly important paradigm for post-training language models. However, we identify a pervasive Scaling Law of Miscalibration: while OPD effectively improves task accuracy, it systematically traps models in severe overconfidence. We trace this failure to an information mismatch: teacher supervision is formed under privileged context available during training, whereas the deployed model must report confidence using only deployment-time information. We formalize this perspective theoretically, showing that teacher-conditioned success is generally not a valid target for deployment-time confidence and that helpful privileged context induces entropy collapse and a systematic optimism bias. To address this, we propose a calibration-aware OPD framework, CaOPD, that estimates empirical confidence from model rollouts, replaces self-reported confidence with this student-grounded target, and distills the revised response through the same self-distillation pipeline. Experiments across various models and domains show that CaOPD achieves Pareto-optimal calibration while maintaining competitive capability, generalizing robustly under out-of-distribution and continual learning. Our findings highlight that capability distillation does not imply calibrated confidence, and that confidence should be treated as an essential objective in post-training. Code: this https URL 

---
# SafeDream: Safety World Model for Proactive Early Jailbreak Detection 

**Authors**: Bo Yan, Weikai Lin, Yada Zhu, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16824)  

**Abstract**: Multi-turn jailbreak attacks progressively erode LLM safety alignment across seemingly innocuous conversation turns, achieving success rates exceeding 90% against state-of-the-art models. Existing alignment-based and guardrail methods suffer from three key limitations: they require costly weight modification, evaluate each turn independently without modeling cumulative safety erosion, and detect attacks only after harmful content has been generated. To address these limitations, we first formulate the proactive early jailbreak detection problem with a new metric, detection lead, that measures how early an attack can be detected before the LLM complies. We then propose SAFEDREAM, a lightweight world-model-based framework that operates as an external module without modifying the LLM's weights. SAFEDREAM introduces three components: (1) a safety state world model that encodes LLM hidden states into a compact safety representation and predicts how it evolves across turns, (2) CUSUM detection that accumulates weak per-turn risk signals into reliable evidence, and (3) contrastive imagination that simultaneously rolls out attack and benign futures in latent space to issue early alarms before jailbreaks occur. On three multi-turn jailbreak benchmarks (XGuard-Train, SafeDialBench, SafeMTData) against 8 baselines, SAFEDREAM achieves the best detection timeliness across all benchmarks (1.06-1.20 turns before compliance) while maintaining competitive false positive rates and outperforming baselines in detection quality. 

---
# Hierarchical Vision Transformer Enhanced by Graph Convolutional Network for Image Classification 

**Authors**: Haibin Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16823)  

**Abstract**: Vision Transformer (ViT) has brought new breakthroughs to the field of image classification by introducing the self-attention mechanism and Graph Convolutional Networks(GCN) have been proposed and successfully applied in data representation and analysis. However, there are key challenges which limit their further development: (1) The patch size selected by ViT is crucial for accurate predictions, which raises a natural question: How to select the size of patches properly or how to comprehensively combine small patches and larger patches; (2) While the spatial structure information is important in vision tasks, the 1D position embeddings fails to capture the spatial structure information of patches more accurately; (3) The GCN can capture the local connectivity relationships between image nodes, but it lacks the ability to capture global graph structural information. On the contrary, the self-attention mechanism of ViT can draw the global relation on image patches, but it is unable to model the local structure of image. To overcome such limitations, we propose the Hierarchical Vision Transformer Enhanced by Graph Convolutional Network (GCN-HViT) for image classification. Specifically, the Hierarchical ViT we designed can model patch-wise information interactions on a global scale within each level and model hierarchical relationships between small patches and large patches across multiple levels. In addition, the proposed GCN method functions as a local feature extractor to obtain the local representation of each image patch which serves as a 2D position embedding of each patch in the 2D space. Meanwhile, it models patch-wise information interactions on a local scale within each level. Extensive experiments on 3 real-world datasets demonstrate that GCN-HViT achieves state-of-the-art performance. 

---
# Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration 

**Authors**: Chongsheng Zhang, Hao Wang, Zelong Yu, Esteban Garces Arias, Julian Rodemann, Zhanshuo Zhang, Qilong Li, Gaojuan Fan, Krikamol Muandet, Christian Heumann  

**Link**: [PDF](https://arxiv.org/pdf/2604.16817)  

**Abstract**: Imbalanced data is commonly present in real-world applications. While data synthesis can effectively mitigate the data scarcity problem of rare-classes, and LLMs have revolutionized text generation, the application of LLMs to relational/structured tabular data synthesis remains underexplored. Moreover, existing approaches lack an effective feedback mechanism that can guide LLMs towards continuously optimizing the quality of the generated data throughout the synthesis process. In this work, we propose RDDG, Relational Data generator with Dynamic Guidance, which is a unified in-context learning framework that employs progressive chain-of-thought (CoT) steps to generate tabular data for enhancing downstream imbalanced classification performance. RDDG first uses core set selection to identify representative samples from the original data, then utilizes in-context learning to discover the inherent patterns and correlations among attributes within the core set, and subsequently generates tabular data while preserving the aforementioned constraints. More importantly, it incorporates a self-reinforcing feedback mechanism that provides automatic assessments on the quality of the generated data, enabling continuous quality optimization throughout the generation process. Experimental results on multiple real and synthetic datasets demonstrate that RDDG outperforms existing approaches in both data fidelity and downstream imbalanced classification performance. We make our code available at this https URL. 

---
# AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems 

**Authors**: Sumeet Ramesh Motwani, Chuan Du, Aleksander Petrov, Christopher Davis, Philip Torr, Antonio Papania-Davis, Weishi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16804)  

**Abstract**: Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We present AutoOR, a scalable synthetic data generation and reinforcement learning pipeline that trains LLMs to autoformalize optimization problems specified in natural language across linear, mixed-integer, and non-linear categories. AutoOR generates verified training data from standard optimization forms and uses solver execution feedback as the reward signal for RL post-training. AutoOR applied to an 8B model achieves state-of-the-art or competitive results across six established OR benchmarks, matching significantly larger frontier models. For a non-linear problem class involving physical dynamics, where frontier models score near 0%, we introduce a curriculum RL strategy that bootstraps from limited initial training data to make this class tractable for post-training. We believe that methods such as AutoOR can significantly accelerate industrial decision-making with AI. 

---
# Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering 

**Authors**: Zixiao Zhao, Amirreza Esmaeili, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2604.16790)  

**Abstract**: Large Language Models are increasingly used as judges to evaluate code artifacts when exhaustive human review or executable test coverage is unavailable. LLM-judge is increasingly relevant in agentic software engineering workflows, where it can help rank candidate solutions and guide patch selection. While attractive for scale, current practice lacks a principled account of reliability and bias: repeated evaluations of the same case can disagree; small prompt edits can swing outcomes; and seemingly semantics-preserving, human-equivalent perturbations may elicit divergent verdicts. This paper studies LLM-as-a-Judge for code through a measurement-first lens. We analyze two pointwise judging regimes across code generation, code repair task, and test generation, and we systematically probe prompt-induced biases. Our study considers difficulty levels for repeated runs and controlled prompt interventions that isolate one presentation cue at a time, and it evaluates judges using consistency and sensitivity to bias. We find that judge decisions are highly sensitive to prompt biases even when the underlying code snippet is unchanged. Across all three tasks, several biases systematically shift preferences toward the option favored by the prompt, improving accuracy when that option aligns with the gold answer but substantially reducing it otherwise. In some settings, these effects are large enough to change task-level conclusions and alter relative model rankings. These findings show that reported judge performance may reflect prompt artifacts rather than stable assessment ability, posing a direct threat to the validity and reproducibility of code evaluation. We therefore argue that LLM-as-a-Judge studies should report bias sensitivity alongside accuracy and incorporate explicit controls to support more trustworthy model comparison in software engineering. 

---
# When Informal Text Breaks NLI: Tokenization Failure, Distribution Shift, and Targeted Mitigations 

**Authors**: Avinash Goutham Aluguvelly  

**Link**: [PDF](https://arxiv.org/pdf/2604.16787)  

**Abstract**: We study how informal surface forms degrade NLI accuracy in ELECTRA-small (14M) and RoBERTa-large (355M) across four transforms applied to SNLI and MultiNLI: slang substitution, emoji replacement, Gen-Z filler tokens, and their combination. Slang substitution (replacing formal words with informal equivalents, e.g., "going to" -> "gonna", "friend" -> "homie") causes minimal degradation (at most 1.1pp): slang vocabulary falls largely within WordPiece coverage, so the tokenizer handles it without signal loss. Emoji replaces content words with Unicode characters that ELECTRA's WordPiece tokenizer maps to [UNK], destroying the input signal before any learned parameters see it (93.6% of emoji examples contain at least one [UNK], mean 2.91 per example). Noise tokens (no cap, deadass, tbh) are fully in-vocabulary but absent from NLI training data, consistent with the model assigning them inferential weight they do not carry. The two failure modes respond to different interventions: preprocessing recovers emoji accuracy by normalizing text before tokenization; augmentation handles noise by exposing the model to noise-bearing examples during training. A hybrid of both achieves 88.93% on the combined variant for ELECTRA on SNLI (up from 75.88%), with no statistically significant drop on clean text. Against GPT-4o-mini zero-shot, unmitigated ELECTRA is significantly worse on transformed variants (p < 0.0001); hybrid ELECTRA surpasses it across all SNLI variants and reaches statistical parity on MultiNLI. 

---
# Bridging Coarse and Fine Recognition: A Hybrid Approach for Open-Ended Multi-Granularity Object Recognition in Interactive Educational Games 

**Authors**: Hanling Yi, Feng Lin, Mao Luo, Yifan Yang, Xiaotian Yu, Rong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16785)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enabled open-ended object recognition, yet they struggle with fine-grained tasks. In contrast, CLIP-style models excel at fine-grained recognition but lack broad coverage of general object categories. To bridge this gap, we propose \textbf{HyMOR}, a \textbf{Hy}brid \textbf{M}ulti-granularity open-ended \textbf{O}bject \textbf{R}ecognition framework that integrates an MLLM with a CLIP model. In HyMOR, the MLLM performs open-ended and coarse-grained object recognition, while the CLIP model specializes in fine-grained identification of domain-specific objects such as animals and plants. This hybrid design enables accurate object understanding across multiple semantic granularities, serving as a robust perceptual foundation for downstream multi-modal content generation and interactive gameplay. To support evaluation in content-rich and educational scenarios, we introduce TBO (TextBook Objects), a dataset containing 20,942 images annotated with 8,816 object categories extracted from textbooks. Extensive experiments demonstrate that HyMOR narrows the fine-grained recognition gap with CLIP to 0.2\% while improving general object recognition by 2.5\% over a baseline MLLM, measured by average Sentence-BERT (SBert) similarity. Overall, HyMOR achieves a 23.2\% improvement in average SBert across all evaluated datasets, highlighting its effectiveness in enabling accurate perception for multi-modal game content generation and interactive learning applications. 

---
# FairNVT: Improving Fairness via Noise Injection in Vision Transformers 

**Authors**: Qiaoyue Tang, Sepidehsadat Hosseini, Mengyao Zhai, Thibaut Durand, Greg Mori  

**Link**: [PDF](https://arxiv.org/pdf/2604.16780)  

**Abstract**: This paper presents FairNVT, a lightweight debiasing framework for pretrained transformer-based encoders that improves both representation and prediction level fairness while preserving task accuracy. Unlike many existing debiasing approaches that address these notions separately, we argue they are inherently connected: suppressing sensitive information at the representation level can facilitate fairer predictions. Our approach learns task-relevant and sensitive embeddings via lightweight adapters, applies calibrated Gaussian noise to the sensitive embedding, and fuses it with the task representation. Together with orthogonality constraints and fairness regularization, these components jointly reduce sensitive-attribute leakage in the learned embeddings and encourage fairer downstream predictions. The framework is compatible with a wide range of pretrained transformer encoders. Across three datasets spanning vision and language, FairNVT reduces sensitive-attribute attacker accuracy, improves demographic-parity and equalized-odds metrics, and maintains high task performance. 

---
# Federation over Text: Insight Sharing for Multi-Agent Reasoning 

**Authors**: Dixi Yao, Tahseen Rabbani, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16778)  

**Abstract**: LLM-powered agents often reason from scratch when presented with a new problem instance and lack automatic mechanisms to transfer learned skills to other agents. We propose a federated learning-like framework, Federation over Text (FoT), that enables multiple agents solving different tasks to collectively generate a shared library of metacognitive insights by iteratively federating their local reasoning processes. Instead of federation over gradients (e.g., as in distributed training), FoT operates at the semantic level without any gradient optimization or supervision signal. Iteratively, each agent does local thinking and self-improvement on their specific tasks independently, and shares reasoning traces with a central server, which aggregates and distills them into a cross-task (and cross-domain) insight library that existing and future agents can leverage to improve performance on related tasks. Experiments show that FoT improves reasoning effectiveness and efficiency across a wide range of challenging applications, including mathematical problem solving, cross-domain collaboration, and machine learning research insight discovery. Specifically, it improves average accuracies of downstream tasks by 24% while reducing the reasoning tokens by 28% across the first two applications. In the research insight discovery application, FoT is able to generate insights that cover over 90% of the major contributions in the subsequent papers. 

---
# Representation Before Training: A Fixed-Budget Benchmark for Generative Medical Event Models 

**Authors**: Inhyeok Lee, Luke Solo, Michael C. Burkhart, Bashar Ramadan, William F. Parker, Brett K. Beaulieu-Jones  

**Link**: [PDF](https://arxiv.org/pdf/2604.16775)  

**Abstract**: Every prediction from a generative medical event model is bounded by how clinical events are tokenized, yet input representation is rarely isolated from other system and architectural choices. We evaluate how representation decisions affect downstream prediction after a shared one-epoch pretraining budget. We train 28 matched transformers on MIMIC-IV and evaluate them on 30 clinical outcomes in three experiments: (1) quantization granularity, reference-range anchoring, and code-value fusion; (2) value encoding (hard bins, soft discretization, code-normalized xVal) crossed with temporal encoding (event order, time tokens, admission-relative RoPE); and (3) native MIMIC laboratory/vital codes versus the Common Longitudinal ICU Format (CLIF)-remapped laboratory/vital codes with compression-preserving perturbation arms. In Experiment 1, fused code-value tokenization improves mortality AUROC from 0.891 to 0.915 (BH-adjusted p < 0.001), hospital length-of-stay AUROC from 0.763 to 0.788 (BH-adjusted p < 0.001), and, for the decile fused-vs-unfused comparison, mean regression Spearman rho across the 13 regression outcomes from 0.414 to 0.494. Across the three temporal encodings, event order only and admission-relative RoPE match or exceed inserting time tokens on average while shortening sequences by 11%. CLIF remapping preserves downstream performance in our single-site setting while yielding a smaller, clinically interpretable token set compatible with multi-site use. Finer-than-decile quantization, reference-range anchoring, and soft discretization help in selective outcomes, while code-normalized xVal remains well below the discrete and soft families, consistent with near-median suppression that persists after the affine variant. 

---
# StageMem: Lifecycle-Managed Memory for Language Models 

**Authors**: Jiarui Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.16774)  

**Abstract**: Long-horizon language model systems increasingly rely on persistent memory, yet many current designs still treat memory primarily as a static store: write an item, place it into memory, and retrieve it later if needed. We argue that this framing does not adequately capture the practical memory-control problem in deployed LLM systems. In realistic settings, the difficulty is often not merely forgetting useful information, but retaining too many uncertain items, forgetting important content in the wrong order, and giving users little trust in what will persist over time. We propose StageMem, a lifecycle-managed memory framework that treats memory as a stateful process rather than a passive repository. StageMem organizes memory into three stages -- transient, working, and durable memory -- and models each item with explicit confidence and strength. This separates shallow admission from long-term commitment: information may first be written at low cost and only later be promoted, retained, updated, or evicted as evidence and pressure evolve. Under controlled pressure regimes, this decomposition helps preserve late-important content while keeping memory burden and deeper-tier pollution more controlled. Adapted external tasks provide boundary evidence that the same schema remains compatible with stronger retrieval structure outside pure synthetic control. We present StageMem as a principled decomposition of the memory-control problem for language model systems. 

---
# The Reliance Negotiation Framework: A Dynamic Process Model of Student LLM Engagement in Academic Writing 

**Authors**: Shahin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2604.16772)  

**Abstract**: Student engagement with large language models (LLMs) in academic writing is not a stable trait, an adoption decision, or a competency level; it is a continuously negotiated process that existing frameworks cannot adequately theorize. Typological models provide categories without mechanisms; technology acceptance models explain adoption but not post-adoption quality; AI literacy frameworks treat competency as a static predictor rather than a live input. None accounts for within-student variability across tasks, the developmental paradox whereby experience produces habituation rather than sophistication, or principled non-use as a form of ethical reasoning. This article introduces the Reliance Negotiation Framework (RNF), developed from a sequential explanatory mixed-methods study of 382 undergraduates at a public minority-serving institution in the United States (survey, N = 382; 14 semi-structured interviews; three qualitative survey strands; 1,435 coded instances). The RNF reconceptualizes LLM reliance as an ongoing negotiation among four concurrent inputs (perceived benefits, perceived risks, ethical commitments, and situational demands) with outputs that recursively modify subsequent decisions. A Two-Model Architecture accommodates the 13.0% of participants whose categorical ethical commitments foreclose negotiation entirely. The framework generates four falsifiable predictions with implications for AI literacy pedagogy, academic integrity policy, and equity-centered practice at minority-serving institutions. 

---
# CapSeal: Capability-Sealed Secret Mediation for Secure Agent Execution 

**Authors**: Shutong Jin, Ruiyi Guo, Ray C. C. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.16762)  

**Abstract**: Modern AI agents routinely depend on secrets such as API keys and SSH credentials, yet the dominant deployment model still exposes those secrets directly to the agent process through environment variables, local files, or forwarding sockets. This design fails against prompt injection, tool misuse, and model-controlled exfiltration because the agent can both use and reveal the same bearer credential. We present CapSeal, a capability-sealed secret mediation architecture that replaces direct secret access with constrained invocations through a local trusted broker. CapSeal combines capability issuance, schema-constrained HTTP execution, broker-executed SSH actions, anti-replay session binding, policy evaluation, and tamper-evident audit trails. We describe a Rust prototype integrated with an MCP-facing adapter, formulate conditional security goals for non-disclosure, constrained use, replay resistance, and auditability, and define an evaluation plan spanning prompt injection, tool misuse, and SSH abuse. The resulting system reframes secret handling for agentic systems from handing the model a key to granting the model a narrowly scoped, non-exportable action capability. 

---
# Frozen Vision Transformers for Dense Prediction on Small Datasets: A Case Study in Arrow Localization 

**Authors**: Maxwell Shepherd  

**Link**: [PDF](https://arxiv.org/pdf/2604.16758)  

**Abstract**: We present a system for automated detection, localization, and scoring of arrow punctures on 40\,cm indoor archery target faces, trained on only 48 annotated photographs (5{,}084 punctures). Our pipeline combines three components: a color-based canonical rectification stage that maps perspective-distorted photographs into a standardized coordinate system where pixel distances correspond to known physical measurements; a frozen self-supervised vision transformer (DINOv3 ViT-L/16) paired with AnyUp guided feature upsampling to recover sub-millimeter spatial precision from $32 \times 32$ patch tokens; and lightweight CenterNet-style detection heads for arrow-center heatmap prediction. Only 3.8\,M of 308\,M total parameters are trainable. Across three cross-validation folds, we achieve a mean F1 score of $0.893 \pm 0.011$ and a mean localization error of $1.41 \pm 0.06$\,mm, comparable to or better than prior fully-supervised approaches that require substantially more training data. An ablation study shows that the CenterNet offset regression head, typically essential for sub-pixel refinement, provides negligible detection improvement while degrading localization in our setting. This suggests that guided feature upsampling already resolves the spatial precision lost through patch tokenization. On downstream archery metrics, the system recovers per-image average arrow scores with a median error of 1.8\% and group centroid positions to within a median of 4.00\,mm. These results demonstrate that frozen foundation models with minimal task-specific adaptation offer a practical paradigm for dense prediction in small-data regimes. 

---
# Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering 

**Authors**: Francesco Sovrano, Gabriele Dominici, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.16756)  

**Abstract**: Prompt-induced cognitive biases are changes in a general-purpose AI (GPAI) system's decisions caused solely by biased wording in the input (e.g., framing, anchors), not task logic. In software engineering (SE) decision support (where problem statements and requirements are natural language) small phrasing shifts (e.g., popularity hints or outcome reveals) can push GPAI models toward suboptimal decisions. We study this with PROBE-SWE, a dynamic benchmark for SE that pairs biased and unbiased versions of the same SE dilemmas, controls for logic and difficulty, and targets eight SE-relevant biases (anchoring, availability, bandwagon, confirmation, framing, hindsight, hyperbolic discounting, overconfidence). We ask whether prompt engineering mitigates bias sensitivity in practice, focusing on actionable techniques that practitioners can apply off-the-shelf in real environments. Testing common strategies (e.g., chain-of-thought, self-debiasing) on cost-effective GPAI systems, we find no statistically significant reductions in bias sensitivity on a per-bias basis. We then adopt a Prolog-style view of the reasoning process: solving SE dilemmas requires making explicit any background axioms and inference assumptions (i.e., SE best practices) that are usually implicit in the prompt. So, we hypothesize that bias-inducing features short-circuit assumptions elicitation, pushing GPAI models toward biased shortcuts. Building on this, we introduce an end-to-end method that elicits best practices and injects axiomatic reasoning cues into the prompt before answering, reducing overall bias sensitivity by 51% on average (p < .001). Finally, we report a thematic analysis that surfaces linguistic patterns associated with heightened bias sensitivity, clarifying when GPAI use is less advisable for SE decision support and where to focus future countermeasures. 

---
# TriTS: Time Series Forecasting from a Multimodal Perspective 

**Authors**: Xiang Ao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16748)  

**Abstract**: Time series forecasting plays a pivotal role in critical sectors such as finance, energy, transportation, and meteorology. However, Long-term Time Series Forecasting (LTSF) remains a significant challenge because real-world signals contain highly entangled temporal dynamics that are difficult to fully capture from a purely 1D perspective. To break this representation bottleneck, we propose TriTS, a novel cross-modal disentanglement framework that projects 1D time series into orthogonal time, frequency, and 2D-vision this http URL seamlessly bridge the 1D-to-2D modality gap without the prohibitive $O(N^2)$ computational overhead of Vision Transformers (ViTs), we introduce a Period-Aware Reshaping strategy and incorporate Visual Mamba (Vim). This approach efficiently models cross-period dependencies as global visual textures while maintaining linear computational complexity. Complementing this, we design a Multi-Resolution Wavelet Mixing (MR-WM) module for the frequency modality, which explicitly decouples non-stationary signals into trend and noise components to achieve fine-grained time-frequency localization. Finally, a streaming linear branch is retained in the time domain to anchor numerical stability. By dynamically fusing these three complementary representations, TriTS effectively adapts to diverse data contexts. Extensive experiments across multiple benchmark datasets demonstrate that TriTS achieves state-of-the-art (SOTA) performance, fundamentally outperforming existing vision-based forecasters by drastically reducing both parameter count and inference latency. 

---
# Evaluating Adaptive Personalization of Educational Readings with Simulated Learners 

**Authors**: Ryan T. Woo, Anmol Rao, Aryan Keluskar, Yinong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16744)  

**Abstract**: We present a framework for evaluating adaptive personalization of educational reading materials with theory-grounded simulated learners. The system builds a learning-objective and knowledge-component ontology from open textbooks, curates it in a browser-based Ontology Atlas, labels textbook chunks with ontology entities, and generates aligned reading-assessment pairs. Simulated readers learn from passages through a Construction-Integration-inspired memory model with DIME-style reader factors, KREC-style misconception revision, and an open New Dale-Chall readability signal. Answers are produced by score-based option selection over the learner's explicit memory state, while BKT drives adaptation. Across three sampled subject ontologies and matched cohorts of 50 simulated learners per condition, adaptive reading significantly improved outcomes in computer science, yielded smaller positive but inconclusive gains in inorganic chemistry, and was neutral to slightly negative in general biology. 

---
# Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines 

**Authors**: Junwan Kim, Hyunkyung Bae  

**Link**: [PDF](https://arxiv.org/pdf/2604.16734)  

**Abstract**: Multimodal large language models (MLLMs) have recently demonstrated strong capabilities in understanding and generating responses from diverse visual inputs, including high-resolution images and long video sequences. As these models scale to richer visual representations, inference increasingly relies on storing large numbers of vision tokens in the key-value (KV) cache, making memory consumption a central bottleneck. Existing methods address this issue by identifying redundancy in vision tokens and compressing the cache, but such compression is typically applied only after all inputs are processed, resulting in high peak memory usage during the prefill stage. In this work, we show that MLLMs exhibit inherent structural regularities and representational redundancy that can be exploited to control memory growth throughout inference. Based on this insight, we propose a sequential input-compression mechanism that enforces a fixed memory budget by performing structure-aware key-value cache compression during the prefill process. This approach substantially reduces peak memory usage while maintaining generative performance with only minimal degradation, enabling more practical and memory-efficient multimodal inference. 

---
# Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis 

**Authors**: Ayhan Can Erdur, Daniel Scholz, Jiazhen Pan, Benedikt Wiestler, Daniel Rueckert, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2604.16729)  

**Abstract**: State-of-the-art large language models (LLMs) show high performance in general visual question answering. However, a fundamental limitation remains: current architectures lack the native 3D spatial reasoning required for direct analysis of volumetric medical imaging, such as CT or MRI. Emerging agentic AI offers a new solution, eliminating the need for intrinsic 3D processing by enabling LLMs to orchestrate and leverage specialized external tools. Yet, the feasibility of such agentic frameworks in complex, multi-step radiological workflows remains underexplored. In this work, we present a training-free agentic pipeline for automated brain MRI analysis. Validating our methodology on several LLMs (GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.5) with off-the-shelf domain-specific tools, our system autonomously executes complex end-to-end workflows, including preprocessing (skull stripping, registration), pathology segmentation (glioma, meningioma, metastases), and volumetric analysis. We evaluate our framework across increasingly complex radiological tasks, from single-scan segmentation and volumetric reporting to longitudinal response assessment requiring multi-timepoint comparisons. We analyze the impact of architectural design by comparing single-agent models against multi-agent "domain-expert" collaborations. Finally, to support rigorous evaluation of future agentic systems, we introduce and release a benchmark dataset of image-prompt-answer tuples derived from public BraTS data. Our results demonstrate that agentic AI can solve highly neuro-radiological image analysis tasks through tool use without the need for training or fine-tuning. 

---
# Late Fusion Neural Operators for Extrapolation Across Parameter Space in Partial Differential Equations 

**Authors**: Eva van Tegelen, Taniya Kapoor, George A.K. van Voorn, Peter van Heijster, Ioannis N. Athanasiadis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16721)  

**Abstract**: Developing neural operators that accurately predict the behavior of systems governed by partial differential equations (PDEs) across unseen parameter regimes is crucial for robust generalization in scientific and engineering applications. In practical applications, variations in physical parameters induce distribution shifts between training and prediction regimes, making extrapolation a central challenge. As a result, the way parameters are incorporated into neural operator models plays a key role in their ability to generalize, particularly when state and parameter representations are entangled. In this work, we introduce the Late Fusion Neural Operator, an architecture that disentangles learning state dynamics from parameter effects, improving predictive performance both within and beyond the training distribution. Our approach combines neural operators for learning latent state representations with sparse regression to incorporate parameter information in a structured manner. Across four benchmark PDEs including advection, Burgers, and both 1D and 2D reaction-diffusion equations, the proposed method consistently outperforms Fourier Neural Operator and CAPE-FNO. Late Fusion Neural Operators achieve consistently the best performance in all experiments, with an average RMSE reduction of 72.9% in-domain and 71.8% out-domain compared to the second-best method. These results demonstrate strong generalization across both in-domain and out-domain parameter regimes. 

---
# Scalable and Adaptive Parallel Training of Graph Transformer on Large Graphs 

**Authors**: Jun-Liang Lin, Kamesh Madduri, Mahmut Taylan Kandemir  

**Link**: [PDF](https://arxiv.org/pdf/2604.16715)  

**Abstract**: Graph foundation models have demonstrated remarkable adaptability across diverse downstream tasks through large-scale pretraining on graphs. However, existing implementations of the backbone model, graph transformers, are typically limited to single-GPU systems, leading to long training times or out-of-memory issues on large graphs. Moreover, parallelizing graph transformer training over the full graph is challenging, as efficiency depends heavily on both the graph structure and system characteristics, such as bandwidth and memory capacity.
In this work, we introduce a distributed training framework for graph transformers, which automatically selects and optimizes parallelization strategies based on the graph structure and hardware configuration. With our implementation of distributed sparse operations, we accelerate sparse graph attention by up to 3.8x and reduce memory consumption by 78% compared to state-of-the-art frameworks. On large graph benchmarks, our proposed framework achieves up to 6x speedup with system scaling up to 8 GPUs. These results demonstrate that the proposed framework improves the scalability of graph transformers, bringing them closer to serving as practical graph foundation models. 

---
# The impact of postediting on AI generative translation in Yemeni context: Translating literary prose by ChatGPT 

**Authors**: Nasim Al-wagieh, Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2604.16704)  

**Abstract**: This study examines the role of artificial intelligence in translation, focusing on ChatGPT, specifically ChatGPT-4, and the extent to which human postediting is required in literary translation. A mixed-method approach was adopted, involving 30 professional translators who evaluated and postedited AI-generated translations of selected Arabic and English literary texts. The results show that although AI improves translation speed and accessibility, it remains limited in handling cultural, stylistic, and figurative aspects of language. Participants generally confirmed the necessity of human postediting, particularly in novels and drama. The findings indicate that emerging human-machine collaboration model rather than replacement of human translators. The study concludes that AI should be used as a supportive tool, while human expertise remains essential for ensuring translation quality and cultural appropriateness. 

---
# LOD-Net: Locality-Aware 3D Object Detection Using Multi-Scale Transformer Network 

**Authors**: Mustaqeem Khan, Aidana Nurakhmetova, Wail Gueaieb, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2604.16696)  

**Abstract**: 3D object detection in point cloud data remains a challenging task due to the sparsity and lack of global structure inherent in the input. In this work, we propose a novel Multi-Scale Attention (MSA) mechanism integrated into the 3DETR architecture to better capture both local geometry and global context. Our method introduces an upsampling operation that generates high-resolution feature maps, enabling the network to better detect smaller and semantically related objects. Experiments conducted on the ScanNetv2 dataset demonstrate that our 3DETR + MSA model improves detection performance, achieving a gain of almost 1% in mAP@25 and 4.78% in mAP@50 over the baseline. While applying MSA to the 3DETR-m variant shows limited improvement, our analysis reveals the importance of adapting the upsampling strategy for lightweight models. These results highlight the effectiveness of combining hierarchical feature extraction with attention mechanisms in enhancing 3D scene understanding. 

---
# No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation 

**Authors**: Yufei Tao, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16686)  

**Abstract**: Large language models (LLMs) can answer questions and summarize documents when conditioned on external contexts (e.g., retrieved evidence), yet context use remains unreliable: models may overwrite an already-correct output (neutral regression) even when the context is non-informative. We formalize neutral regression as a do-no-harm requirement and quantify it by measuring accuracy drops on baseline-correct items under answer-consistent contexts. We propose No-Worse Context-Aware Decoding (NWCAD), a decode-time adapter built on a two-stream setup with a two-stage gate: it backs off to no-context decoding when the context is non-informative, and otherwise uses context-conditioned decoding with a CAD-style fallback under uncertainty. We evaluate NWCAD on benchmarks that separate do-no-harm reliability from context utilization (accuracy gains on genuinely helpful contexts). NWCAD prevents neutral regression on baseline-correct items while preserving strong context-driven accuracy on helpful contexts. 

---
# Graph Transformer-Based Pathway Embedding for Cancer Prognosis 

**Authors**: Koushik Howlader, Md Tauhidul Islam, Wei Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.16685)  

**Abstract**: Accurate prediction of cancer progression remains a challenge due to the high heterogeneity of molecular omics data across patients. While biologically informed models have improved the interpretability of these predictions, a persistent limitation lies in how they encode individual genes to construct pathway representations. Existing hierarchical models typically derive gene features by directly mapping raw molecular inputs, whereas integration frameworks often rely on simple statistical aggregations of patient-level signals. These approaches often fail to explicitly learn a shared base representation for each gene, thereby limiting the expressiveness and biological accuracy of downstream pathway embeddings. To address this, we introduce PATH, a modulation-based, patient-conditioned gene embedding strategy. PATH represents a paradigm shift by starting from a shared base embedding for each gene, preserving a stable biological identity across the population, and then dynamically adapting it using patient-specific copy number variation (CNV) and mutation signals. This allows the model to capture subtle individual molecular variations while maintaining a consistent latent understanding of the gene itself. We integrate PATH into a graph transformer framework that models interactions among biologically connected pathways through pathway-guided attention. Across pancancer metastasis prediction, PATH achieves an F1 score of 0.8766, representing an 8.8 percent improvement over the current SOTA multi-omics benchmarks. Beyond superior predictive accuracy, our approach identifies biologically meaningful pathways and, crucially, reveals disease-state-specific pathway rewiring, offering new insights into the evolving pathway-pathway interactions that drive cancer progression. 

---
# Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning 

**Authors**: Gehan Zheng, Sanjay Seenivasan, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16683)  

**Abstract**: Imitation learning has enabled robots to acquire complex visuomotor manipulation skills from demonstrations, but deployment failures remain a major obstacle, especially for long-horizon action-chunked policies. Once execution drifts off the demonstration manifold, these policies often continue producing locally plausible actions without recovering from the failure. Existing runtime monitors either require failure data, over-trigger under benign feature drift, or stop at failure detection without providing a recovery mechanism. We present Rewind-IL, a training-free online safeguard framework for generative action-chunked imitation policies. Rewind-IL combines a zero-shot failure detector based on Temporal Inter-chunk Discrepancy Estimate (TIDE), calibrated with split conformal prediction, with a state-respawning mechanism that returns the robot to a semantically verified safe intermediate state. Offline, a vision-language model identifies recovery checkpoints in demonstrations, and the frozen policy encoder is used to construct a compact checkpoint feature database. Online, Rewind-IL monitors self-consistency in overlapping action chunks, tracks similarity to the checkpoint library, and, upon failure, rewinds execution to the latest verified safe state before restarting inference from a clean policy state. Experiments on real-world and simulated long-horizon manipulation tasks, including transfer to flow-matching action-chunked policies, demonstrate that policy-internal consistency coupled with semantically grounded respawning offers a practical route to improved reliability in imitation learning. Supplemental materials are available at this https URL 

---
# KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving 

**Authors**: Yichao Yuan, Mosharaf Chowdhury, Nishil Talati  

**Link**: [PDF](https://arxiv.org/pdf/2604.16682)  

**Abstract**: Power has become a central bottleneck for AI inference. This problem is becoming more urgent as agentic AI emerges as a major workload class, yet prior power-management techniques focus almost entirely on single-turn LLM serving. Our analysis shows that agentic serving behaves fundamentally differently: each request carries long-lived context that evolves across tool-interleaved turns, and lowering GPU frequency can push the system into a thrashing regime where memory pressure sharply worsens both performance and power efficiency. These observations show that power optimization for agentic serving requires rethinking.
We present KAIROS, a context-aware power optimization system for agentic AI serving. KAIROS uses agent context as a first-class control signal to jointly manage GPU frequency, per-instance concurrency, and multi-instance request placement. This enables KAIROS to save power when memory headroom exists while avoiding thrashing and preserving performance targets. At a high level, KAIROS tracks requests at agent granularity, adapts local control to context growth and agent progress, and routes agents across instances to jointly improve power efficiency and memory stability. Evaluated across diverse software and data engineering agentic tasks, KAIROS achieves an average of 27% (up to 39.8%) power reduction while meeting the performance targets. 

---
# ReconVLA: An Uncertainty-Guided and Failure-Aware Vision-Language-Action Framework for Robotic Control 

**Authors**: Lingling Chen, Zongyao Lyu, William J. Beksi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16677)  

**Abstract**: Vision-language-action (VLA) models have emerged as generalist robotic controllers capable of mapping visual observations and natural language instructions to continuous action sequences. However, VLAs provide no calibrated measure of confidence in their action predictions, thus limiting their reliability in real-world settings where uncertainty and failures must be anticipated. To address this problem we introduce ReconVLA, a reliable conformal model that produces uncertainty-guided and failure-aware control signals. Concretely, our approach applies conformal prediction directly to the action token outputs of pretrained VLA policies, yielding calibrated uncertainty estimates that correlate with execution quality and task success. Furthermore, we extend conformal prediction to the robot state space to detect outliers or unsafe states before failures occur, providing a simple yet effective failure detection mechanism that complements the action-level uncertainty. We evaluate ReconVLA in both simulation and real robot experiments across diverse manipulation tasks. Our results show that conformalized action predictions consistently improve failure anticipation, reduce catastrophic errors, and provide a calibrated measure of confidence without retraining or modifying the underlying VLA. 

---
# Cross-Modal Bayesian Low-Rank Adaptation for Uncertainty-Aware Multimodal Learning 

**Authors**: Habibeh Naderi, Behrouz Haji Soleimani, Stan Matwin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16657)  

**Abstract**: Large pre-trained language models are increasingly adapted to downstream tasks using parameter-efficient fine-tuning (PEFT), but existing PEFT methods are typically deterministic and unimodal, making them poorly suited for low-resource multimodal settings where predictive uncertainty and cross-modal reliability both matter. We introduce CALIBER (Context-Aware Low-rank Inference with Bayesian Embedding Regularization), a multimodal uncertainty-aware PEFT framework for audio-text learning. CALIBER extends Bayesian low-rank adaptation by conditioning the variational posterior in the adapter space on per-layer, token-level text-audio cross-attention. Specifically, text-derived low-rank features attend to frame-level audio embeddings to produce localized acoustic context, which then modulates the mean and variance of a compact stochastic latent matrix within the rank-$r$ adapter space. This design treats audio not only as an additional feature source, but as a contextual reliability signal that shapes both adaptation and confidence. By confining stochasticity to a low-dimensional latent component, CALIBER retains the computational efficiency and scalability of PEFT while enabling heteroscedastic multimodal uncertainty estimation. Experimental results across diverse text and audio backbones show that CALIBER consistently matches or improves upon text-only Bayesian PEFT and conventional multimodal transfer-learning baselines, with token-level cross-attention yielding the most consistent gains. Our findings demonstrate that localized cross-modal conditioning is an effective and lightweight mechanism for uncertainty-aware multimodal adaptation. 

---
# A Two-Stage Multi-Modal MRI Framework for Lifespan Brain Age Prediction 

**Authors**: Dingyi Zhang, Ruiying Liu, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16655)  

**Abstract**: The accurate quantification of brain age from MRI has emerged as an important biomarker of brain health. However, existing approaches are often restricted to narrow age ranges and single-modality MRI data, limiting their capacity to capture the coordinated macro- and microstructural changes that unfold across the human lifespan. To address these limitations, we developed a multi-modal brain age framework to characterize the integrated evolution of brain morphology and white matter organization. Our model adopts a two-stage architecture, where modalities are processed independently and integrated via late fusion in both stages: first to classify each subject into one of six developmental stages, and then to estimate age within the predicted stage. This design enables a unified and lifespan-spanning assessment of brain maturity across diverse developmental periods. 

---
# AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation 

**Authors**: Weihua Du, Jingming Zhuo, Yixin Dong, Andre Wang He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2604.16625)  

**Abstract**: Recent large language model (LLM) agents have shown promise in using execution feedback for test-time adaptation. However, robust self-improvement remains far from solved: most approaches still treat each problem instance independently, without accumulating reusable knowledge. This limitation is particularly pronounced in domain-specific languages such as Triton, which are underrepresented in LLM pretraining data. Their strict constraints and non-linear optimization landscape further make naive generation and local refinement unreliable. We propose AdaExplore, an agent framework that enables self-improvement via accumulated execution feedback for performance-critical kernel code generation through two complementary stages: failure-driven adaptation and diversity-preserving search, jointly improving correctness and optimization performance without additional fine-tuning or external knowledge. In the adaptation stage, the agent synthesizes tasks and converts recurring failures into a reusable memory of validity rules, helping subsequent generations remain within the feasible set. In the search stage, the agent organizes candidate kernels as a tree and alternates between small local refinements and larger structural regeneration, allowing it to explore the optimization landscape beyond local optima. Experiments on kernel runtime optimization benchmarks validate these gains: AdaExplore achieves 3.12x and 1.72x speedups on KernelBench Level-2 and Level-3, respectively, within 100 steps, and continues to improve with additional computation. 

---
# Aligning Backchannel and Dialogue Context Representations via Contrastive LLM Fine-Tuning 

**Authors**: Livia Qian, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2604.16622)  

**Abstract**: Backchannels (e.g., `yeah', `mhm', and `right') are short, non-interruptive feedback signals whose lexical form and prosody jointly convey pragmatic meaning. While prior computational research has largely focused on predicting backchannel timing, the relationship between lexico-prosodic form and meaning remains underexplored. We propose a two-stage framework: first, fine-tuning large language models on dialogue transcripts to derive rich contextual representations; and second, learning a joint embedding space for dialogue contexts and backchannel realizations. We evaluate alignment with human perception via triadic similarity judgments (prosodic and cross-lexical) and a context-backchannel suitability task. Our results demonstrate that the learned projections substantially improve context-backchannel retrieval compared to previous methods. In addition, they reveal that backchannel form is highly sensitive to extended conversational context and that the learned embeddings align more closely with human judgments than raw WavLM features. 

---
# Beyond Feature Fusion: Contextual Bayesian PEFT for Multimodal Uncertainty Estimation 

**Authors**: Habibeh Naderi, Behrouz Haji Soleimani, Stan Matwin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16615)  

**Abstract**: We introduce CoCo-LoRA, a multimodal, uncertainty-aware parameter-efficient fine-tuning method for text prediction tasks accompanied by audio context. Existing PEFT approaches such as LoRA are efficient but typically deterministic, while recent Bayesian low-rank adapters model uncertainty in a lightweight way yet remain largely unimodal and condition uncertainty primarily on internal text features. This leaves them poorly equipped to reflect uncertainty driven by external acoustic factors such as background noise, channel variability, or speaking style, which can materially affect reliability in speech-centered applications. CoCo-LoRA addresses this gap by conditioning a contextual variational posterior in the low-rank space on both local text-derived adapter features and an audio-derived context signal. A pooled audio embedding is projected once into a shared context space and then adapted through lightweight layer-wise heads, enabling global-to-local, depth-specific modulation of the adapter uncertainty and update without high-dimensional multimodal fusion. Stochasticity is confined to a compact latent component in the rank space, preserving PEFT scalability while producing audio-sensitive, heteroscedastic uncertainty. Based on our evaluations across diverse tasks and backbone combinations, CoCo-LoRA consistently matches or outperforms text-only PEFT and conventional feature-fusion transfer baselines, particularly on high-coverage labels where reliable adaptation is critical. The results indicate that using audio as a contextual uncertainty signal, rather than as a fused feature stream, provides a robust and parameter-efficient alternative for multimodal low-resource prediction. 

---
# Spotlights and Blindspots: Evaluation Machine-Generated Text Detection 

**Authors**: Kevin Stowe, Kailash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2604.16607)  

**Abstract**: With the rise of generative language models, machine-generated text detection has become a critical challenge. A wide variety of models is available, but inconsistent datasets, evaluation metrics, and assessment strategies obscure comparisons of model effectiveness. To address this, we evaluate 15 different detection models from six distinct systems, as well as seven trained models, across seven English-language textual test sets and three creative human-written datasets. We provide an empirical analysis of model performance, the influence of training and evaluation data, and the impact of key metrics. We find that no single system excels in all areas and nearly all are effective for certain tasks, and the representation of model performance is critically linked to dataset and metric choices. We find high variance in model ranks based on datasets and metrics, and overall poor performance on novel human-written texts in high-risk domains. Across datasets and metrics, we find that methodological choices that are often assumed or overlooked are essential for clearly and accurately reflecting model performance. 

---
# Human Cognition in Machines: A Unified Perspective of World Models 

**Authors**: Timothy Rupprecht, Pu Zhao, Amir Taherin, Arash Akbari, Arman Akbari, Yumei He, Sean Duffy, Juyi Lin, Yixiao Chen, Rahul Chowdhury, Enfu Nan, Yixin Shen, Yifan Cao, Haochen Zeng, Weiwei Chen, Geng Yuan, Jennifer Dy, Sarah Ostadabbas, Silvia Zhang, David Kaeli, Edmund Yeh, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16592)  

**Abstract**: This comprehensive report distinguishes prior works by the cognitive functions they innovate. Many works claim an almost "human-like" cognitive capability in their world models. To evaluate these claims requires a proper grounding in first principles in Cognitive Architecture Theory (CAT). We present a conceptual unified framework for world models that fully incorporates all the cognitive functions associated with CAT (i.e. memory, perception, language, reasoning, imagining, motivation, and meta-cognition) and identify gaps in the research as a guide for future states of the art. In particular, we find that motivation (especially intrinsic motivation) and meta-cognition remain drastically under-researched, and we propose concrete directions informed by active inference and global workspace theory to address them. We further introduce Epistemic World Models, a new category encompassing agent frameworks for scientific discovery that operate over structured knowledge. Our taxonomy, applied across video, embodied, and epistemic world models, suggests research directions where prior taxonomies have not. 

---
# Randomized Antipodal Search Done Right for Data Pareto Improvement of LLM Unlearning 

**Authors**: Ziwen Liu, Huawei Lin, Yide Ran, Denghui Zhang, Jianwen Xie, Chuan Li, Weijie Zhao, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16591)  

**Abstract**: Large language models (LLMs) sometimes memorize undesirable knowledge, which must be removed after deployment. Prior work on machine unlearning has focused largely on optimization methods that adjust parameters to enforce forgetting while preserving retention. However, these approaches assume that the forget and retain sets are readily available, which rarely holds in practice. Unlearning is typically triggered by an undesired generation at inference time, making the retrieval of relevant data the central challenge.
We introduce the notion of data Pareto improvement for LLM unlearning, which formalizes how retrieval can expand the achievable trade-off frontier between forgetting and retention. To realize this principle, we propose Randomized Antipodal Search on Linearized Influence Kernel (RASLIK), a retrieval algorithm that combines permutation-projection hashing with randomized antipodal search. RASLIK reduces selection variance, achieves sublinear complexity, and yields a double gain in both quality and efficiency. Across multiple models, datasets, and unlearning algorithms, RASLIK consistently outperforms deterministic baselines and even oracle sampling, establishing randomized search as a principled and scalable solution for data-centric unlearning. 

---
# Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction 

**Authors**: Xiao Wang, Zezhong Zhang, Isaac Lyngaas, Hong-Jun Yoon, Jong-Youl Choi, Siming Liang, Janet Wang, Hristo G. Chipilski, Ashwin M. Aji, Feng Bao, Peter Jan van Leeuwen, Dan Lu, Guannan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16590)  

**Abstract**: Accurate weather and climate prediction relies on data assimilation (DA), which estimates the Earth system state by integrating observations with models. While exascale computing has significantly advanced earth simulation, scalable and accurate inference of the Earth system state remains a fundamental bottleneck, limiting uncertainty quantification and prediction of extreme events. We introduce a unified one-stage generative DA framework that reformulates assimilation as Bayesian posterior sampling, replacing the conventional forecast-update cycle with compute-dense, GPU-efficient inference. At the core is STORM, a novel spatiotemporal transformer with a global attention linear-complexity scaling algorithm that breaks the quadratic attention barrier. On 32,768 GPUs of the Frontier supercomputer, our method achieves 63% strong scaling efficiency and 1.6 ExaFLOP sustained performance. We further scale to 20 billion spatiotemporal tokens, enabling km-scale global modeling over 177k temporal frames, regimes previously unreachable, establishing a new paradigm for Earth system prediction. 

---
# Hybrid Spectro-Temporal Fusion Framework for Structural Health Monitoring 

**Authors**: Jongyeop Kim, Jinki Kim, Doyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.16589)  

**Abstract**: Structural health monitoring plays a critical role in ensuring structural safety by analyzing vibration responses from engineering systems. This paper proposes a Spectro-Temporal Alignment framework and a Hybrid Spectro-Temporal Fusion framework that integrate arrival-time interval descriptors with spectral features to capture both fine-scale and coarse-scale vibration dynamics. Experiments conducted on data collected from an LDS V406 electrodynamic shaker demonstrate that the proposed spectro-temporal representations significantly outperform conventional input formulations. The results indicate that a temporal resolution ({\Delta}{\tau}) of 0.008 of 0.02 favors traditional machine learning models, whereas a finer resolution ({\Delta}{\tau}) of 0.008 effectively unlocks the performance potential of deep learning architectures. Beyond classification accuracy, a comprehensive stability analysis based on condensed indices, including mean performance, standard deviation, coefficient of variation, and balanced score, shows that the proposed hybrid framework consistently achieves higher accuracy with substantially lower variability compared to baseline and alignment-only approaches. Overall, these results demonstrate that the proposed framework provides a robust, accurate, and reliable solution for vibration-based structural health monitoring. 

---
# MambaKick: Early Penalty Direction Prediction from HAR Embeddings 

**Authors**: Henry O. Velesaca, David Freire-Obregon, Abel Reyes-Angulo, Steven Araujo, Angel Sappa  

**Link**: [PDF](https://arxiv.org/pdf/2604.16588)  

**Abstract**: Penalty kicks in soccer are decided under extreme time constraints, where goalkeepers benefit from anticipating shot direction from the kickers motion before or around ball contact. In this paper, MambaKick is presented as a learning-based framework for penalty direction prediction that leverages pretrained human action recognition (HAR) embeddings extracted from contact-centered short video segments and combines them with a lightweight temporal predictor. Rather than relying on explicit kinematic reconstruction or handcrafted biomechanical features, the approach reuses transferable spatiotemporal representations and utilizes selective state-spare models (Mamba) for efficient sequence aggregation. Simple contextual metadata (e.g., field side and footedness) are also considered as complementary cues that may reduce ambiguity in real-world footage. Across a range of HAR backbones, MambaKick consistently improves or matches strong embedding baselines, achieving up to 53.1% accuracy for three classes and 64.5% for two classes under the proposed methodology. Overall, the results indicate that combining pretrained HAR representations with efficient state-space temporal modeling is a practical direction for low-latency intention prediction in real-world sports video. The code will be available at GitHub: this https URL 

---
# Real-Time Visual Attribution Streaming in Thinking Model 

**Authors**: Seil Kang, Woojung Han, Junhyeok Kim, Jinyeong Kim, Youngeun Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16587)  

**Abstract**: We present an amortized framework for real-time visual attribution streaming in multimodal thinking models. When these models generate code from a screenshot or solve math problems from images, their long reasoning traces should be grounded in visual evidence. However, verifying this reliance is challenging: faithful causal methods require costly repeated backward passes or perturbations, while raw attention maps offer instant access, they lack causal validity. To resolve this, we introduce an amortized approach that learns to estimate the causal effects of semantic regions directly from the rich signals encoded in attention features. Across five diverse benchmarks and four thinking models, our approach achieves faithfulness comparable to exhaustive causal methods while enabling visual attribution streaming, where users observe grounding evidence as the model reasons, not after. Our results demonstrate that real-time, faithful attribution in multimodal thinking models is achievable through lightweight learning, not brute-force computation. 

---
# A Systematic Survey and Benchmark of Deep Learning for Molecular Property Prediction in the Foundation Model Era 

**Authors**: Zongru Li, Xingsheng Chen, Honggang Wen, Regina Qianru Zhang, Ming Li, Xiaojin Zhang, Hongzhi Yin, Qiang Yang, Kwok-Yan Lam, Pietro Lio, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16586)  

**Abstract**: Molecular property prediction integrates quantum chemistry, cheminformatics, and deep learning to connect molecular structure with physicochemical and biological behavior. This survey traces four complementary paradigms, including Quantum, Descriptor Machine Learning, Geometric Deep Learning, and Foundation Models, and outlines a unified taxonomy linking molecular representations, model architectures, and interdisciplinary applications. Benchmark analyses integrate evidence from both widely used datasets and datasets reflecting industry perspectives, encompassing quantum, physicochemical, physiological, and biophysical domains. The survey examines current standards in data curation, splitting strategies, and evaluation protocols, highlighting challenges including inconsistent stereochemistry, heterogeneous assay sources, and reproducibility limitations under random or poorly defined splits. These observations motivate the modernization of benchmark design toward more transparent, time- and scaffold-aware methodologies. We further propose three forward-looking directions: (i) physics-aware learning embedding quantum consistency, (ii) uncertainty-calibrated foundation models for trustworthy inference, and (iii) realistic multimodal benchmark ecosystems integrating computational and experimental data. Repository: this https URL. 

---
# The Global Neural World Model: Spatially Grounded Discrete Topologies for Action-Conditioned Planning 

**Authors**: Noureddine Kermiche  

**Link**: [PDF](https://arxiv.org/pdf/2604.16585)  

**Abstract**: We present the Global Neural World Model (GNWM), a self-stabilizing framework that achieves topological quantization through balanced continuous entropy constraints. Operating as a continuous, action-conditioned Joint-Embedding Predictive Architecture (JEPA), the GNWM maps environments onto a discrete 2D grid, enforcing translational equivariance without pixel-level reconstruction. Our results show this architecture prevents manifold drift during autoregressive rollouts by using grid ``snapping'' as a native error-correction mechanism. Furthermore, by training via maximum entropy exploration (random walks), the model learns generalized transition dynamics rather than memorizing specific expert trajectories. We validate the GNWM across passive observation, active agent control, and abstract sequence regimes, demonstrating its capacity to act not just as a spatial physics simulator, but as a causal discovery model capable of organizing continuous, predictable concepts into structured topological maps. 

---
# Certified Program Synthesis with a Multi-Modal Verifier 

**Authors**: Yueyang Feng, Dipesh Kafle, Vladimir Gladshtein, Vitaly Kurin, George Pîrlea, Qiyuan Zhao, Peter Müller, Ilya Sergey  

**Link**: [PDF](https://arxiv.org/pdf/2604.16584)  

**Abstract**: Certified program synthesis (aka vericoding) is the process of automatically generating a program, its formal specification, and a machine-checkable proof of their alignment from a natural-language description. Two challenges make vericoding difficult. First, specifications synthesised from natural language are often either too weak to be meaningful or too strong to be implementable, yet existing approaches lack systematic means to detect such defects. Second, the landscape of program verifiers is fragmented: each tool supports a particular reasoning mode -- auto-active (e.g., Dafny, Verus) or interactive (e.g., Coq, Lean) -- with its own trade-off between automation and expressivity. This forces every synthesis methodology to be tailored to a single verification paradigm, limiting the class of tasks it can handle effectively.
We overcome both challenges by structuring the certified synthesis workflow around a multi-modal verifier -- a single tool combining dynamic validation, automated proofs, and interactive proof scripting in one foundational framework. We realise this idea in LeetProof, an agentic pipeline built on Velvet, a multi-modal verifier embedded in Lean. Multi-modality enables LeetProof to validate generated specifications via randomised property-based testing before any code is synthesised, decompose the synthesis task into sub-problems guided by verification conditions, and delegate residual proof obligations to frontier AI provers specialised for Lean. We evaluate LeetProof on benchmarks derived from prior work on certified synthesis. Our specification validation uncovers defects in existing reference benchmarks, and LeetProof's staged pipeline achieves a significantly higher rate of fully certified solutions than a single-mode baseline at the same budget -- consistently across two frontier LLM backends. 

---
# POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving 

**Authors**: Shaoang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16583)  

**Abstract**: Edge deployment of large language models (LLMs) increasingly relies on libraries of lightweight LoRA adapters, yet GPU/DRAM can keep only a small resident subset at a time. Serving a request through a non-resident adapter requires paging its weights from storage, incurring measurable latency. This creates a two-timescale online control problem: on a slow timescale, the system selects which adapters remain resident in fast memory, while on a fast timescale it routes each request to an adapter whose context-dependent utility is unknown a priori. The two decisions are tightly coupled: the cache determines the cost of exploration, and the router determines which adapters receive informative feedback. We formulate this joint caching-and-routing problem as a two-timescale contextual bandit and propose POLAR (Paging and Online Learning for Adapter Routing). POLAR pairs a cache-aware LinUCB router with an epoch-based cache controller. We study two variants. A fixed-epoch version provides a robust baseline with worst-case regret guarantees under arbitrary contexts. An epoch-doubling version, POLAR+, adds forced exploration and improved cache optimization to achieve $\widetilde{\mathcal{O}}(d\sqrt{NT}+\sqrt{KT})$ sublinear regret under stochastic regularity and cacheability conditions, where $N$ is the adapter count, $K$ the cache size, $d$ the context dimension, and $T$ the horizon. The routing term matches the standard contextual-bandit rate up to logarithmic factors, showing that the memory hierarchy does not fundamentally slow routing learning. Experiments using 15 real LoRA adapters for Qwen2.5-7B together with measured GPU paging latencies show that adaptive cache control substantially outperforms non-adaptive baselines and exhibits scaling trends consistent with the theory. 

---
# Camo-M3FD: A New Benchmark Dataset for Cross-Spectral Camouflaged Pedestrian Detection 

**Authors**: Henry O. Velesaca, Andrea Mero, Guillermo A. Castillo, Angel D. Sappa  

**Link**: [PDF](https://arxiv.org/pdf/2604.16582)  

**Abstract**: Pedestrian detection is fundamental to autonomous driving, robotics, and surveillance. Despite progress in deep learning, reliable identification remains challenging due to occlusions, cluttered backgrounds, and degraded visibility. While multispectral detection-combining visible and thermal sensors-mitigates poor visibility, the challenge of camouflaged pedestrians remains largely unexplored. Existing Camouflaged Object Detection (COD) benchmarks focus on biological species, leaving a gap in safety-critical human detection where targets blend into their surroundings. To address this, we introduce Camo-M3FD (derived from the M3FD dataset), a novel benchmark for cross-spectral camouflaged pedestrian detection, consisting of registered visible-thermal image pairs. The dataset is curated using quantitative metrics to ensure high foreground-background similarity. We provide high-quality pixel-level masks and establish a standardized evaluation framework using state-of-the-art COD models. Our results demonstrate that while thermal signals provide indispensable localization cues, multispectral fusion is essential for refining structural details. Camo-M3FD serves as a foundational resource for developing robust and safety-critical detection systems. The dataset is available on GitHub: this https URL 

---
# NCO4CVRP: Neural Combinatorial Optimization for the Capacitated Vehicle Routing Problem 

**Authors**: Mahir Labib Dihan, Md. Ashrafur Rahman Khan, Wasif Jalal, Md. Roqunuzzaman Sojib, Mashroor Hasan Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16581)  

**Abstract**: Neural Combinatorial Optimization (NCO) has emerged as a powerful framework for solving combinatorial optimization problems by integrating deep learning-based models. This work focuses on improving existing inference techniques to enhance solution quality and generalization. Specifically, we modify the Random Re-Construct (RRC) approach of the Light Encoder Heavy Decoder (LEHD) model by incorporating Simulated Annealing (SA). Unlike the conventional RRC, which greedily replaces suboptimal segments, our SA-based modification introduces a probabilistic acceptance mechanism that allows the model to escape local optima and explore a more diverse solution space. Additionally, we enhance the Policy Optimization with Multiple Optima (POMO) approach by integrating Beam Search, enabling systematic exploration of multiple promising solutions while maintaining diversity in the search space. We further investigate different inference strategies, including Softmax Sampling, Greedy, Gumbel-Softmax, and Epsilon-Greedy, analyzing their impact on solution quality. Furthermore, we explore instance augmentation techniques, such as horizontal and vertical flipping and rotation-based augmentations, to improve model generalization across different CVRP instances. Our extensive experiments demonstrate that these modifications significantly reduce the optimality gap across various Capacitated Vehicle Routing Problem (CVRP) benchmarks, with Beam Search and SA-based RRC consistently yielding superior performance. By refining inference techniques and leveraging enhanced search strategies, our work contributes to the broader applicability of NCO models in real-world combinatorial optimization tasks. 

---
# Continuous ageing trajectory representations for knee-aware lifetime prediction of lithium-ion batteries across heterogeneous dataset 

**Authors**: Agnieszka Pregowska, Stefan Marynowicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.16580)  

**Abstract**: Accurate assessment of lithium-ion battery ageing is challenged by cell-to-cell variability, heterogeneous cycling protocols, and limited transferability of data-driven models across datasets. In particular, robust identification of degradation transitions, such as the knee point, and reliable early-life prediction of remaining useful life (RUL) remain open problems. This study proposes a unified framework for battery ageing analysis based on continuous representations of voltage-capacity and capacity-cycle trajectories learned from heterogeneous public datasets (NASA, CALCE, ISU-ILCC). The continuous formulation enables consistent extraction of degradation descriptors, including curvature, plateau length and knee-related metrics, while reducing sensitivity to dataset-specific discretisation. Across more than 250 cells, statistically significant correlations between knee onset and end-of-life (Pearson 0.75-0.84) are observed. Additional early-life analysis confirms that knee-related features retain predictive value when estimated from partial trajectories. Early-life models provide increasingly stable RUL predictions as the number of observed cycles increases, with meaningful predictive performance emerging within the first 5-20 cycles and remain robust under cross-dataset domain shift. The framework integrates continuous modelling, feature extraction and uncertainty-aware prediction, providing an interpretable and dataset-consistent approach demonstrating robustness across heterogeneous dataset types. Compared with conventional discrete or feature-based methods, the proposed representation reduces sensitivity to sampling resolution and improves cross-dataset consistency. The study is limited to laboratory-scale datasets and capacity-based end-of-life definitions. 

---
# Towards Trustworthy Depression Estimation via Disentangled Evidential Learning 

**Authors**: Fangyuan Liu, Sirui Zhao, Zeyu Zhang, Jinyang Huang, Feng-Qi Cui, Bin Luo, Tong Xu, Meng Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16579)  

**Abstract**: Automated depression estimation is highly vulnerable to signal corruption and ambient noise in real-world deployment. Prevailing deterministic methods produce uncalibrated point estimates, exposing safety-critical clinical systems to the severe risk of overconfident misdiagnoses. To establish a highly resilient and trustworthy assessment paradigm, we propose EviDep, an evidential learning framework that jointly quantifies depression severity alongside aleatoric and epistemic uncertainties via a Normal-Inverse-Gamma distribution. A fundamental vulnerability in multimodal evidential fusion is the uncontrolled accumulation of cross-modal redundancies. This structural flaw artificially inflates diagnostic confidence by double-counting overlapping evidence. To guarantee robust evidence synthesis, EviDep enforces strict information integrity. First, a Frequency-aware Feature Extraction module leverages a wavelet-based Mixture-of-Experts to dynamically isolate task-irrelevant noise, preserving the fidelity of diagnostic signals. Subsequently, a Disentangled Evidential Learning strategy separates the shared consensus from modality-specific nuances. By explicitly decoupling these representations before Bayesian fusion, EviDep systematically mitigates evidence redundancy. Extensive experiments on AVEC 2013, 2014, DAIC-WOZ, and E-DAIC confirm that EviDep achieves state-of-the-art predictive accuracy and superior uncertainty calibration, delivering a robust fail-safe mechanism for trustworthy clinical screening. 

---
# Multilevel neural networks with dual-stage feature fusion for human activity recognition 

**Authors**: Abeer FathAllah Brery, Ascensión Gallardo-Antolín, Israel Gonzalez-Carrasco, Mahmoud Fakhry  

**Link**: [PDF](https://arxiv.org/pdf/2604.16577)  

**Abstract**: Human activity recognition (HAR) refers to the process of identifying human actions and activities using data collected from sensors. Neural networks, such as convolutional neural networks (CNNs), long short-term memory (LSTM) networks, convolutional LSTM, and their hybrid combinations, have demonstrated exceptional performance in various research domains. Developing a multilevel individual or hybrid model for HAR involves strategically integrating multiple networks to capitalize on their complementary strengths. The structural arrangement of these components is a critical factor influencing the overall performance. This study explores a novel framework of a two-level network architecture with dual-stage feature fusion: late fusion, which combines the outputs from the first network level, and intermediate fusion, which integrates the features from both the first and second levels. We evaluated $15$ different network architectures of CNNs, LSTMs, and convolutional LSTMs, incorporating late fusion with and without intermediate fusion, to identify the optimal configuration. Experimental evaluation on two public benchmark datasets demonstrates that architectures incorporating both late and intermediate fusion achieve higher accuracy than those relying on late fusion alone. Moreover, the optimal configuration outperforms baseline models, thereby validating its effectiveness for HAR. 

---
# Evaluating Temporal and Structural Anomaly Detection Paradigms for DDoS Traffic 

**Authors**: Yasmin Souza Lima, Rodrigo Moreira, Larissa F. Rodrigues Moreira, Tereza Cristina M. de B. Carvalho, Flávio de Oliveira Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.16575)  

**Abstract**: Unsupervised anomaly detection is widely used to detect Distributed Denial-of-Service (DDoS) attacks in cloud-native 5G networks, yet most studies assume a fixed traffic representation, either temporal or structural, without validating which feature space best matches the data. We propose a lightweight decision framework that prioritizes temporal or structural features before training, using two diagnostics: lag-1 autocorrelation of an aggregated flow signal and PCA cumulative explained variance. When the probes are inconclusive, the framework reserves a hybrid option as a future fallback rather than an empirically validated branch. Experiments on two statistically distinct datasets with Isolation Forest, One-Class SVM, and KMeans show that structural features consistently match or outperform temporal ones, with the performance gap widening as temporal dependence weakens. 

---
# FedOBP: Federated Optimal Brain Personalization through Cloud-Edge Element-wise Decoupling 

**Authors**: Xingyan Chen, Tian Du, Changqiao Xu, Fuzhen Zhuang, Lujie Zhong, Gabriel-Miro Muntean, Enmao Diao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16574)  

**Abstract**: Federated Learning (FL) faces challenges from client data heterogeneity and resource-constrained mobile devices, which can degrade model accuracy. Personalized Federated Learning (PFL) addresses this issue by adapting shared global knowledge to local data distributions. A promising approach in PFL is model decoupling, which separates the model into global and personalized parameters, raising the key question of which parameters should be personalized to balance global knowledge sharing and local adaptation. In this paper, we propose a Federated Optimal Brain Personalization (FedOBP) algorithm with a quantile-based thresholding mechanism and introduce an element-wise importance score. This score extends Optimal Brain Damage (OBD) pruning theory by incorporating a federated approximation of the first-order derivative in the Taylor expansion to evaluate the importance of each parameter for personalization. Moreover, we move the metric computation originally performed on clients to the server side, to alleviate the burden on resource-constrained mobile devices. To the best of our knowledge, this is the first work to bridge classical saliency-based pruning theory with federated parameter decoupling, providing a rigorous theoretical justification for selecting personalized parameters based on their sensitivity to local loss landscapes. Extensive experiments demonstrate that FedOBP outperforms state-of-the-art methods across diverse datasets and heterogeneity scenarios, while requiring personalization of only a very small number of personalized parameters. 

---
# In Search of Lost DNA Sequence Pretraining 

**Authors**: Zhijiang Tang, Jiaxin Qi, Yan Cui, Jinli Ou, Yuhua Zheng, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16570)  

**Abstract**: DNA sequence encoding is fundamental to gene function prediction, protein synthesis, and diverse downstream biological tasks. Despite the substantial progress achieved by large-scale DNA sequence pretraining, existing studies have overwhelmingly emphasized pretraining scale and custom downstream evaluation datasets, while neglecting some essential components of the pretraining paradigm. In this paper, we reveal three critical yet heretofore overlooked problems in DNA pretraining: inappropriate downstream datasets, inherent flaws in the neighbor-masking strategy, and the lack of detailed discussion on vocabulary. Therefore, we undertake comprehensive investigations and propose principled guidelines, including selection criteria for evaluation datasets, guiding task design, and in-depth vocabulary analysis. Extensive experiments validate the significance of our identified problems and support the rationale behind our recommendations. Finally, we introduce a standardized testbed that enables reproducible and rigorous benchmarking of DNA pretraining methods to advance the development of genomic foundation models. 

---
# Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models 

**Authors**: Jiaoyang Ruan, Xin Gao, Yinda Chen, Hengyu Zeng, Liang Du, Guanghao Li, Jie Fu, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16565)  

**Abstract**: While Diffusion Large Language Models (dLLMs) offer structural advantages for global planning, efficiently verifying that they arrive at correct answers via valid reasoning traces remains a critical challenge. In this work, we propose a geometric perspective: Reasoning on the Manifold. We hypothesize that valid generation trajectories reside as stable attractors on the high-density manifold of the learned distribution, whereas invalid paths exhibit off-manifold drift. To operationalize this, we introduce Bidirectional Manifold Consistency (BMC), a training-free, unsupervised metric that quantifies the stability of the generated sequence through a forward-masking and backward-reconstruction cycle. Empirically, we demonstrate BMC's versatility across the full reasoning lifecycle: (1) in Diagnosis, it serves as a robust discriminator of solution validity without ground truth answer; (2) in Inference, it enables rejection resampling to effectively concentrate computational resources on complex reasoning tasks; and (3) in Alignment, it functions as a dense geometric reward that transforms sparse outcome supervision into fine-grained guidance, empowering models to self-evolve beyond standard baselines. Our results establish intrinsic geometric stability as a robust indicator of correctness for dLLMs. 

---
# Classification of systolic murmurs in heart sounds using multiresolution complex Gabor dictionary and vision transformer 

**Authors**: Mahmoud Fakhry, Abeer FathAllah Brery  

**Link**: [PDF](https://arxiv.org/pdf/2604.16563)  

**Abstract**: Systolic murmurs are extra heart sounds that occur during the contraction phase of the cardiac cycle, often indicating heart abnormalities caused by turbulent blood flow. Their intensity, pitch, and quality vary, requiring precise identification for the accurate diagnosis of cardiac disorders. This study presents an automatic classification system for systolic murmurs using a feature extraction module, followed by a classification model. The feature extraction module employs complex orthogonal matching pursuit to project single or multiple murmur segments onto a redundant dictionary composed of multiresolution complex Gabor basis functions (GBFs). The resulting projection weights are split and reshaped into variable-resolution time--frequency feature matrices. Processing multiple segments of a single recording using a shared dictionary mitigates murmur variability. This is achieved by learning the weights for each segment while enforcing that they correspond to the same set of basis functions in the dictionary, promoting consistent time--frequency feature matrices. The classification model is built based on a vision transformer to process multiple input matrices of different resolutions by passing each through a convolutional neural network for patch tokenization. All embedding tokens are then concatenated to form a matrix and forwarded to an encoder layer that includes multihead attention, residual connections, and a convolutional network with a kernel size of one. This integration of multiresolution feature extraction with transformer-based feature classification enhances the accuracy and reliability of heart murmur identification. An experimental analysis of four types of systolic murmurs from the CirCor DigiScope dataset demonstrates the effectiveness of the system, achieving a classification accuracy of $95.96\%$. 

---
# See Through the Noise: Improving Domain Generalization in Gaze Estimation 

**Authors**: Yanming Peng, Shijing Wang, Yaping Huang, Yi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.16562)  

**Abstract**: Generalizable gaze estimation methods have garnered increasing attention due to their critical importance in real-world applications and have achieved significant progress. However, they often overlook the effect of label noise, arising from the inherent difficulty of acquiring precise gaze annotations, on model generalization performance. In this paper, we are the first to comprehensively investigate the negative effects of label noise on generalization in gaze estimation. Further, we propose a novel solution, called See-Through-Noise (SeeTN) framework, which improves generalization from a novel perspective of mitigating label noise. Specifically, we propose to construct a semantic embedding space via a prototype-based transformation to preserve a consistent topological structure between gaze features and continuous labels. We then measure feature-label affinity consistency to distinguish noisy from clean samples, and introduce a novel affinity regularization in the semantic manifold to transfer gaze-related information from clean to noisy samples. Our proposed SeeTN promotes semantic structure alignment and enforces domain-invariant gaze relationships, thereby enhancing robustness against label noise. Extensive experiments demonstrate that our SeeTN effectively mitigates the adverse impact of source-domain noise, leading to superior cross-domain generalization without compromising the source-domain accuracy, and highlight the importance of explicitly handling noise in generalized gaze estimation. 

---
# SpecPylot: Python Specification Generation using Large Language Models 

**Authors**: Ragib Shahariar Ayon, Shibbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.16560)  

**Abstract**: Automatically generating formal specifications could reduce the effort needed to improve program correctness, but in practice, this is still challenging. Many developers avoid writing contracts by hand, which limits the use of automated verification tools. Recent large language models (LLMs) can generate specifications from code, but these specifications often fail in terms of verification. The reason is syntax errors, overly strict constraints, or mismatches with program behavior. We present SpecPylot, a Python tool that synthesizes executable specifications for Python programs as icontract annotations and checks them using crosshair's symbolic execution. The tool relies on LLMs to propose candidate contracts and uses crosshair to validate them. When crosshair finds a concrete counterexample, SpecPylot updates only the generated contracts and leaves the program itself untouched. In addition, the tool can produce coverage-driven pytest stubs and keep detailed execution artifacts that are useful during debugging. Overall, the evaluation indicates that SpecPylot is able to generate crosshair-compatible contracts for most programs, but it also highlights the practical limits introduced by bounded symbolic exploration and differences in LLM behavior. 

---
# LLM as a Tool, Not an Agent: Code-Mined Tree Transformations for Neural Architecture Search 

**Authors**: Masakazu Yoshimura, Zitang Sun, Yuiko Sakuma, Junji Otsuka, Atsushi Irie, Takeshi Ohashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16555)  

**Abstract**: Neural Architecture Search (NAS) aims to automatically discover high-performing deep neural network (DNN) architectures. However, conventional algorithm-driven NAS relies on carefully hand-crafted search spaces to ensure executability, which restricts open-ended exploration. Recent coding-based agentic approaches using large language models (LLMs) reduce manual design, but current LLMs struggle to reliably generate complex, valid architectures, and their proposals are often biased toward a narrow set of patterns observed in their training data. To bridge reliable algorithmic search with powerful LLM assistance, we propose LLMasTool, a hierarchical tree-based NAS framework for stable and open-ended model evolution. Our method automatically extracts reusable modules from arbitrary source code and represents full architectures as hierarchical trees, enabling evolution through reliable tree transformations rather than code generation. At each evolution step, coarse-level planning is governed by a diversity-guided algorithm that leverages Bayesian modeling to improve exploration efficiency, while the LLM resolves the remaining degrees of freedom to ensure a meaningful evolutionary trajectory and an executable generated architecture. With this formulation, instead of fully agentic LLM approaches, our method explores diverse directions beyond the inherent biases in the LLM. Our method improves over existing NAS methods by 0.69, 1.83, and 2.68 points on CIFAR-10, CIFAR-100, and ImageNet16-120, demonstrating its effectiveness. 

---
# PA-TCNet: Pathology-Aware Temporal Calibration with Physiology-Guided Target Refinement for Cross-Subject Motor Imagery EEG Decoding in Stroke Patients 

**Authors**: Xiangkai Wang, Yun Zhao, Dongyi He, Qingling Xia, Gen Li, Nizhuan Wang, Ningxiao Peng, Bin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16554)  

**Abstract**: Stroke patient cross-subject electroencephalography (EEG) decoding of motor imagery (MI) brain-computer interface (BCI) is essential for motor rehabilitation, yet lesion-related abnormal temporal dynamics and pronounced inter-patient heterogeneity often undermine generalization. Existing adaptation methods are easily misled by pathological slow-wave activity and unstable target-domain pseudo-labels. To address this challenge, we propose PA-TCNet, a pathology-aware temporal calibration framework with physiology-guided target refinement for stroke motor imagery decoding. PA-TCNet integrates two coordinated components. The Pathology-aware Rhythmic State Mamba (PRSM) module decomposes EEG spatiotemporal features into slowly varying rhythmic context and fast transient perturbations, injecting the fused pathological context into selective state propagation to more effectively capture abnormal temporal dynamics. The Physiology-Guided Target Calibration (PGTC) module constructs source-domain sensorimotor region-of-interest templates, imposing physiological consistency constraints and dynamically refining target-domain pseudo-labels, thereby improving adaptation reliability. Leave-one-subject-out experiments on two independent stroke EEG datasets, XW-Stroke and 2019-Stroke, yielded mean accuracies of 66.56\% and 72.75\%, respectively, outperforming state-of-the-art baselines. These results indicate that jointly modeling pathological temporal dynamics and physiology-constrained pseudo-supervision can provide more robust cross-subject initialization for personalized post-stroke MI-BCI rehabilitation. The implemented code is available at this https URL. 

---
# Co-generation of Layout and Shape from Text via Autoregressive 3D Diffusion 

**Authors**: Zhenggang Tang, Yuehao Wang, Yuchen Fan, Jun-Kun Chen, Yu-Ying Yeh, Kihyuk Sohn, Zhangyang Wang, Qixing Huang, Alexander Schwing, Rakesh Ranjan, Dilin Wang, Zhicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16552)  

**Abstract**: Recent text-to-scene generation approaches largely reduced the manual efforts required to create 3D scenes. However, their focus is either to generate a scene layout or to generate objects, and few generate both. The generated scene layout is often simple even with LLM's help. Moreover, the generated scene is often inconsistent with the text input that contains non-trivial descriptions of the shape, appearance, and spatial arrangement of the objects. We present a new paradigm of sequential text-to-scene generation and propose a novel generative model for interactive scene creation. At the core is a 3D Autoregressive Diffusion model 3D-ARD+, which unifies the autoregressive generation over a multimodal token sequence and diffusion generation of next-object 3D latents. To generate the next object, the model uses one autoregressive step to generate the coarse-grained 3D latents in the scene space, conditioned on both the current seen text instructions and already synthesized 3D scene. It then uses a second step to generate the 3D latents in the smaller object space, which can be decoded into fine-grained object geometry and appearance. We curate a large dataset of 230K indoor scenes with paired text instructions for training. We evaluate 7B 3D-ARD+, on challenging scenes, and showcase the model can generate and place objects following non-trivial spatial layout and semantics prescribed by the text instructions. 

---
# An Interpretable Framework Applying Protein Words to Predict Protein-Small Molecule Complementary Pairing Rules 

**Authors**: Jingke Chen, Jingrui Zhong, Tazneen Hossain Tani, Zidong Su, Xiaochun Zhang, Boxue Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.16550)  

**Abstract**: Despite the high accuracy of 'black box' deep learning models, drug discovery still relies on protein-ligand interaction principles and heuristics. To improve interpretability of protein-small molecule binding predictions, we developed the PWRules framework, which applies binding affinity data to identify privileged small molecule fragments and subsequently defines complementary pairing rules between these fragments and protein words (semantic sequence units) through an interpretability module. The resulting word-fragment rules are then ranked by the PWScore function to prioritize active compounds. Evaluations on benchmark datasets show that PWScore achieves competitive performance comparable to the physics-based model (Glide) and the deep learning model (PSICHIC) and shows broad applicability for protein targets outside the training dataset, e.g., SARS-CoV-2 main protease. Notably, PWScore captures complementary interaction information, yielding superior enrichment performance when integrated with these established methods. Structural analysis of protein-ligand complexes indicates that learned word-fragment rules are significantly enriched near ligand-binding pockets, despite training without explicit structural guidance. By extracting and applying complementary pairing rules, PWRules provides an interpretable framework for drug discovery. 

---
# A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty 

**Authors**: Zehao Lin, Chunyu Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16548)  

**Abstract**: Research on large language model (LLM) security is shifting from "will the model leak training data" to a more consequential question: can an agent with persistent, long-term memory be continuously shaped, cross-session poisoned, accessed without authorization, and propagated across shared organizational state? Recent surveys cover memory architectures and agent mechanisms, but fewer center the epistemic and governance properties of persistent, writable memory as the reason memory is an independent security problem.
This survey addresses that gap. Drawing on cognitive neuroscience and the philosophy of memory, we characterize agent memory as malleable, rewritable, and socially propagating, and develop a memory-lifecycle framework organized around six phases -- Write, Store, Retrieve, Execute, Share, Forget/Rollback -- cross-tabulated against four security objectives: integrity, confidentiality, availability, governance. We organize the literature on memory poisoning, extraction, retrieval corruption, control-flow hijacking, cross-agent propagation, rollback, and governance, and situate representative architectures as determinants of which phases are explicitly governable.
Three findings stand out: the literature concentrates on write- and retrieve-time integrity attacks, while confidentiality, availability, store/forget, and benign-persistence failures remain sparsely studied; no published architecture covers all nine governance primitives we identify; and using LLMs themselves for memory security remains sparse yet essential.
We unify these under mnemonic sovereignty -- verifiable, recoverable governance over what may be written, who may read, when updates are authorized, and which states may be forgotten -- arguing future secure agents will be differentiated not only by recall capacity, but by memory governance quality. 

---
# Conjunctive Prompt Attacks in Multi-Agent LLM Systems 

**Authors**: Nokimul Hasan Arif, Qian Lou, Mengxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.16543)  

**Abstract**: Most LLM safety work studies single-agent models, but many real applications rely on multiple interacting agents. In these systems, prompt segmentation and inter-agent routing create attack surfaces that single-agent evaluations miss. We study \emph{conjunctive prompt attacks}, where a trigger key in the user query and a hidden adversarial template in one compromised remote agent each appear benign alone but activate harmful behavior when routing brings them together. We consider an attacker who changes neither model weights nor the client agent and instead controls only trigger placement and template insertion. Across star, chain, and DAG topologies, routing-aware optimization substantially increases attack success over non-optimized baselines while keeping false activations low. Existing defenses, including PromptGuard, Llama-Guard variants, and system-level controls such as tool restrictions, do not reliably stop the attack because no single component appears malicious in isolation. These results expose a structural vulnerability in agentic LLM pipelines and motivate defenses that reason over routing and cross-agent composition. Code is available at this https URL. 

---
# PoInit-of-View: Poisoning Initialization of Views Transfers Across Multiple 3D Reconstruction Systems 

**Authors**: Weijie Wang, Songlong Xing, Zhengyu Zhao, Nicu Sebe, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2604.16540)  

**Abstract**: Poisoning input views of 3D reconstruction systems has been recently studied. However, we identify that existing studies simply backpropagate adversarial gradients through the 3D reconstruction pipeline as a whole, without uncovering the new vulnerability rooted in specific modules of the 3D reconstruction pipeline. In this paper, we argue that the structure-from-motion (SfM) initialization, as the geometric core of many widely used reconstruction systems, can be targeted to achieve transferable poisoning effects across diverse 3D reconstruction systems. To this end, we propose PoInit-of-View, which optimizes adversarial perturbations to intentionally introduce cross-view gradient inconsistencies at projections of corresponding 3D points. These inconsistencies disrupt keypoint detection and feature matching, thereby corrupting pose estimation and triangulation within SfM, eventually resulting in low-quality rendered views. We also provide a theoretical analysis that connects cross-view inconsistency to correspondence collapse. Experimental results demonstrate the effectiveness of our PoInit-of-View on diverse 3D reconstruction systems and datasets, surpassing the single-view baseline by 25.1% in PSNR and 16.5% in SSIM in black-box transfer settings, such as 3DGS to NeRF. 

---
# Understanding Tool-Augmented Agents for Lean Formalization: A Factorial Analysis 

**Authors**: Ke Zhang, Patricio Gallardo, Maziar Raissi, Sudhir Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2604.16538)  

**Abstract**: Automatic translation of natural language mathematics into faithful Lean 4 code is hindered by the fundamental dissonance between informal set-theoretic intuition and strict formal type theory. This gap often causes LLMs to hallucinate non-existent library definitions, resulting in code that fails to compile or lacks semantic fidelity. In this work, we investigate the effectiveness of tool-augmented agents for this task through a systematic factorial analysis of three distinct tool categories: Fine-tuned Model Querying (accessing expert drafts), Knowledge Search (retrieving symbol definitions), and Compiler Feedback (verifying code via a Lean REPL). We first benchmark the agent against one-shot baselines, demonstrating large gains in both compilation success and semantic equivalence. We then use the factorial decomposition to quantify the impact of each category, isolating the marginal contribution of each tool type to overall performance. 

---
# Robustifying and Selecting Cohort-Appropriate Prognostic Models under Distributional Shifts 

**Authors**: Dimitris Bertsimas, Carol Gao, Angelos G. Koulouras, Georgios Antonios Margonis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16537)  

**Abstract**: External validation is widely regarded as the gold standard for prognostic model evaluation. In this study, we challenge the assumption that successful external calibration guarantees model generalizability and propose two complementary strategies to improve transportability of prognostic models across cohorts.
Using six real-world surgical cohorts from tertiary academic centers, we tested whether successful external calibration depends largely on similarity in covariates and outcomes between training and validation cohorts, quantified using Kullback-Leibler (KL) divergence, with calibration assessed by the Integrated Calibration Index (ICI). From the model-developer's perspective, we trained the "best-on-average" prognostic model by tuning toward a meta-analysis-derived covariate and outcome distribution as an approximation of the broader target population. From the end-user perspective, we proposed a simple measure for cohort outcome similarity to identify, among published models, the one most suitable for a given target cohort in terms of both calibration and clinical utility.
External calibration worsened as distributional mismatch increased. Higher KL divergence was associated with higher ICI in both surgery-alone (Spearman $\rho=0.614$, $p=0.004$) and surgery + adjuvant chemotherapy cohorts (Spearman $\rho=0.738$, $p<0.001$). Meta-analysis-informed weighting improved calibration in most settings without materially affecting discrimination, with the clearest benefit when evaluated on the aggregated external population ($p=0.037$). Models developed in more similar cohorts achieved lower ICI in surgery-alone (Spearman $\rho=0.803$, $p<0.001$) and surgery + adjuvant chemotherapy cohorts (Spearman $\rho=0.737$, $p<0.001$), and provided greater clinical utility on DCA. 

---
# Towards Reliable Testing of Machine Unlearning 

**Authors**: Anna Mazhar, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2604.16536)  

**Abstract**: Machine learning components are now central to AI-infused software systems, from recommendations and code assistants to clinical decision support. As regulations and governance frameworks increasingly require deleting sensitive data from deployed models, machine unlearning is emerging as a practical alternative to full retraining. However, unlearning introduces a software quality-assurance challenge: under realistic deployment constraints and imperfect oracles, how can we test that a model no longer relies on targeted information? This paper frames unlearning testing as a first-class software engineering problem. We argue that practical unlearning tests must provide (i) thorough coverage over proxy and mediated influence pathways, (ii) debuggable diagnostics that localize where leakage persists, (iii) cost-effective regression-style execution under query budgets, and (iv) black-box applicability for API-deployed models. We outline a causal, pathway-centric perspective, causal fuzzing, that generates budgeted interventions to estimate residual direct and indirect effects and produce actionable "leakage reports". Proof-of-concept results illustrate that standard attribution checks can miss residual influence due to proxy pathways, cancellation effects, and subgroup masking, motivating causal testing as a promising direction for unlearning testing. 

---
# SCATR: Simple Calibrated Test-Time Ranking 

**Authors**: Divya Shyamal, Marta Knežević, Lan Tran, Chanakya Ekbote, Vijay Lingam, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16535)  

**Abstract**: Test-time scaling (TTS) improves large language models (LLMs) by allocating additional compute at inference time. In practice, TTS is often achieved through parallel scaling: generating multiple candidate responses and selecting the best via a Best-of-N (BoN) strategy. Its effectiveness therefore hinges on the scoring function. Learned scorers such as process reward models (PRMs) can be strong, but they are expensive to train and run. Lightweight confidence heuristics based on token log-probabilities are much cheaper, yet we find that they often perform substantially worse. To improve on lightweight confidence heuristics without incurring the full cost of stronger learned scorers, we introduce SCATR, a simple and efficient BoN ranking method that learns a lightweight scorer from a small calibration set using hidden representations from the base model. Across coding and mathematical reasoning benchmarks, SCATR improves over prior confidence-based baselines by up to 9%. Relative to LoRA fine-tuning on the same calibration data, it achieves comparable accuracy with up to 8000x fewer trainable parameters and much lower compute, reducing training and inference latency by up to 150x and 1000x, respectively. SCATR is also competitive with strong PRM baselines, and in several settings improves accuracy by up to 7.8% on math and 4.2% on coding while enabling up to 1000x faster inference. Overall, SCATR offers a strong accuracy-efficiency trade-off for scalable test-time selection. 

---
# Public and private blockchain for decentralized digital building twins and building automation system 

**Authors**: Reachsak Ly, Alireza Shojaei  

**Link**: [PDF](https://arxiv.org/pdf/2604.16534)  

**Abstract**: The communication protocols and data transfer mechanisms employed by IoT devices in smart buildings and corresponding digital twin systems predominantly rely on centralized architectures. Such centralized systems are vulnerable to single points of failure, where a malfunction can disrupt operational processes. This study introduces a blockchain-based decentralized protocol to enhance the cyber resilience of IoT data transfer for digital twins and enable decentralized automation of building operations. The framework incorporates public and private blockchain technologies alongside two case studies showcasing prototypes of each system. These prototypes were validated within a real-world building environment using smart home appliances and two digital twin platforms, with their performance evaluated based on cost, scalability, data security, and privacy. The findings reveal that the Hyperledger Fabric-based system excels in terms of scalability, speed, and cost-effectiveness, while both frameworks offer advantages over traditional centralized protocols in system cyber resilience, data security, and privacy. 

---
# G-PARC: Graph-Physics Aware Recurrent Convolutional Neural Networks for Spatiotemporal Dynamics on Unstructured Meshes 

**Authors**: Jack T. Beerman, Tyler J. Abele, Mehdi Taghizadeh, Andrew Davis, Zoë J. Gray, Negin Alemazkoor, Xinfeng Gao, H.S. Udaykumar, Stephen S. Baek  

**Link**: [PDF](https://arxiv.org/pdf/2604.16533)  

**Abstract**: Physics-aware recurrent convolutional networks (PARC) have demonstrated strong performance in predicting nonlinear spatiotemporal dynamics by embedding differential operators directly into the computational graph of a neural network. However, pixel-based convolutions are restricted to static, uniform Cartesian grids, making them ill-suited to following evolving localized structures in an efficient manner. Graph neural networks (GNNs) naturally handle irregular spatial discretizations, but existing graph-based physics-aware deep learning (PADL) methods have difficulty handling extreme nonlinear regimes. To address these limitations, we propose Graph PARC (G-PARC), which uses moving least squares (MLS) kernels to approximate spatial derivatives on unstructured graphs, and embeds the derivatives of governing partial differential equations into the network's computational graph. G-PARC achieves better accuracy with 2-3x fewer parameters than MeshGraphNet, MeshGraphKAN, and GraphSAGE, replacing the traditional encoder-processor-decoder framework with analytically computed differential operators. We demonstrate that G-PARC (1) generalizes across nonuniform spatial and temporal discretizations; (2) handles moving meshes required for structural deformation; and (3) outperforms existing graph-based PADL methods on nonlinear benchmarks including fluvial hydrology, planar shock waves, and elastoplastic dynamics. By embedding explicit physical operators within the flexibility of GNNs, G-PARC enables accurate modeling of extreme nonlinear phenomena on complex computational domains, moving PADLbeyond idealized Cartesian grids. 

---
# Beyond Attack Success Rate: A Multi-Metric Evaluation of Adversarial Transferability in Medical Imaging Models 

**Authors**: Emily Curl, Kofi Ampomah, Md Erfan, Sayanton Dibbo  

**Link**: [PDF](https://arxiv.org/pdf/2604.16532)  

**Abstract**: While deep learning systems are becoming increasingly prevalent in medical image analysis, their vulnerabilities to adversarial perturbations raise serious concerns for clinical deployment. These vulnerability evaluations largely rely on Attack Success Rate (ASR), a binary metric that indicates solely whether an attack is successful. However, the ASR metric does not account for other factors, such as perturbation strength, perceptual image quality, and cross-architecture attack transferability, and therefore, the interpretation is incomplete. This gap requires consideration, as complex, large-scale deep learning systems, including Vision Transformers (ViTs), are increasingly challenging the dominance of Convolutional Neural Networks (CNNs). These architectures learn differently, and it is unclear whether a single metric, e.g., ASR, can effectively capture adversarial behavior. To address this, we perform a systematic empirical study on four medical image datasets: PathMNIST, DermaMNIST, RetinaMNIST, and CheXpert. We evaluate seven models (VGG-16, ResNet-50, DenseNet-121, Inception-v3, DeiT, Swin Transformer, and ViT-B/16) against seven attack methods at five perturbation budgets, measuring ASR, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and $L_2$ perturbation magnitude. Our findings show a consistent pattern: perceptual and distortion metrics are strongly associated with one another and exhibit minimal correlation with ASR. This applies to both CNNs and ViTs. The results demonstrate that ASR alone is an inadequate indicator of adversarial robustness and transferability. Consequently, we argue that a thorough assessment of adversarial risk in medical AI necessitates multi-metric frameworks that encompass not only the attack efficacy but also its methodology and associated overheads. 

---
# Scaling Test-Time Compute for Agentic Coding 

**Authors**: Joongwon Kim, Wannan Yang, Kelvin Niu, Hongming Zhang, Yun Zhu, Eryk Helenowski, Ruan Silva, Zhengxing Chen, Srinivasan Iyer, Manzil Zaheer, Daniel Fried, Hannaneh Hajishirzi, Sanjeev Arora, Gabriel Synnaeve, Ruslan Salakhutdinov, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16529)  

**Abstract**: Test-time scaling has become a powerful way to improve large language models. However, existing methods are best suited to short, bounded outputs that can be directly compared, ranked or refined. Long-horizon coding agents violate this premise: each attempt produces an extended trajectory of actions, observations, errors, and partial progress taken by the agent. In this setting, the main challenge is no longer generating more attempts, but representing prior experience in a form that can be effectively selected from and reused. We propose a test-time scaling framework for agentic coding based on compact representations of rollout trajectories. Our framework converts each rollout into a structured summary that preserves its salient hypotheses, progress, and failure modes while discarding low-signal trace details. This representation enables two complementary forms of inference-time scaling. For parallel scaling, we introduce Recursive Tournament Voting (RTV), which recursively narrows a population of rollout summaries through small-group comparisons. For sequential scaling, we adapt Parallel-Distill-Refine (PDR) to the agentic setting by conditioning new rollouts on summaries distilled from prior attempts. Our method consistently improves the performance of frontier coding agents across SWE-Bench Verified and Terminal-Bench v2.0. For example, by using our method Claude-4.5-Opus improves from 70.9% to 77.6% on SWE-Bench Verified (mini-SWE-agent) and 46.9% to 59.1% on Terminal-Bench v2.0 (Terminus 1). Our results suggest that test-time scaling for long-horizon agents is fundamentally a problem of representation, selection, and reuse. 

---
# Expert-Annotated Embryo Image Dataset with Natural Language Descriptions for Evidence-Based Patient Communication in IVF 

**Authors**: Nicklas Neu, Thomas Ebner, Jasmin Primus, Bernhard Schenkenfelder, Raphael Zefferer, Mathias Brunbauer, Florian Kromp  

**Link**: [PDF](https://arxiv.org/pdf/2604.16528)  

**Abstract**: Embryo selection is one of multiple crucial steps in in-vitro fertilization, commonly based on morphological assessment by clinical embryologists. Although artificial intelligence methods have demonstrated their potential to support embryo selection by automated embryo ranking or grading methods, the overall impact of AI-based solutions is still limited. This is mainly due to the required adaptation of automated solutions to custom clinical data, reliance on time lapse incubators and a lack of interpretability to understand AI reasoning. The modern, informed patient is questioning expert decisions, particularly if the treatment is not successful. Thus, evidence-based decision justification in tasks like embryo selection would support transparent decision making and respectful patient communication. To support this aim, we hereby present an expert-annotated dataset consisting of embryo images and corresponding morphological description using natural language. The description contains relevant information on embryonic cell cycle, developmental stage and morphological features. This dataset enables the finetuning of modern foundational vision-language models to learn and improve over time with high accuracy. Predicted embryo descriptions can then be leveraged to automatically extract scientific evidence from literature, facilitating well-informed, evidence-based decision-making and transparent communication with patients. Our proposed dataset supports research in language-based, interpretable, and transparent automated embryo assessment and has the potential to enhance the decision-making process and improve patient outcomes significantly over time. 

---
# CAMP: Cumulative Agentic Masking and Pruning for Privacy Protection in Multi-Turn LLM Conversations 

**Authors**: Aman Panjwani  

**Link**: [PDF](https://arxiv.org/pdf/2604.16521)  

**Abstract**: The deployment of Large Language Models in agentic, multi-turn conversational settings has introduced a class of privacy vulnerabilities that existing protection mechanisms are not designed to address. Current approaches to Personally Identifiable Information (PII) masking operate on a per-turn basis, scanning each user message in isolation and replacing detected entities with typed placeholders before forwarding sanitized text to the model. While effective against direct identifier leakage within a single message, these methods are fundamentally stateless and fail to account for the compounding privacy risk that emerges when PII fragments accumulate across conversation turns. A user who separately discloses their name, employer, location, and medical condition across several messages has revealed a fully re-identifiable profile - yet no individual message would trigger a per-turn masker. We formalize this phenomenon as Cumulative PII Exposure (CPE) and propose CAMP (Cumulative Agentic Masking and Pruning), a cross-turn privacy protection framework for multi-turn LLM conversations. CAMP maintains a session-level PII registry, constructs a co-occurrence graph to model combination risk between entity types, computes a CPE score after each turn, and triggers retroactive masking of conversation history when the score crosses a configurable threshold. We evaluate CAMP on four synthetic multi-turn scenarios spanning healthcare, hiring, finance, and general conversation, demonstrating that per-turn baselines expose re-identifiable profiles that CAMP successfully neutralizes while preserving full conversational utility. 

---
# On-Orbit Space AI: Federated, Multi-Agent, and Collaborative Algorithms for Satellite Constellations 

**Authors**: Ziyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16518)  

**Abstract**: Satellite constellations are transforming space systems from isolated spacecraft into networked, software-defined platforms capable of on-orbit perception, decision making, and adaptation. Yet much of the existing AI studies remains centered on single-satellite inference, while constellation-scale autonomy introduces fundamentally new algorithmic requirements: learning and coordination under dynamic inter-satellite connectivity, strict SWaP-C limits, radiation-induced faults, non-IID data, concept drift, and safety-critical operational constraints. This survey consolidates the emerging field of on-orbit space AI through three complementary paradigms: (i) {federated learning} for cross-satellite training, personalization, and secure aggregation; (ii) {multi-agent algorithms} for cooperative planning, resource allocation, scheduling, formation control, and collision avoidance; and (iii) {collaborative sensing and distributed inference} for multi-satellite fusion, tracking, split/early-exit inference, and cross-layer co-design with constellation networking. We provide a system-level view and a taxonomy that unifies collaboration architectures, temporal mechanisms, and trust models. To support community development and keep this review actionable over time, we continuously curate relevant papers and resources at this https URL. 

---
# Predicting Blastocyst Formation in IVF: Integrating DINOv2 and Attention-Based LSTM on Time-Lapse Embryo Images 

**Authors**: Zahra Asghari Varzaneh, Niclas Wölner-Hanssen, Reza Khoshkangini, Thomas Ebner, Magnus Johnsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.16505)  

**Abstract**: The selection of the optimal embryo for transfer is a critical yet challenging step in in vitro fertilization (IVF), primarily due to its reliance on the manual inspection of extensive time-lapse imaging data. A key obstacle in this process is predicting blastocyst formation from the limited number of daily images available. Many clinics also lack complete time-lapse systems, so full videos are often unavailable. In this study, we aimed to predict which embryos will develop into blastocysts using limited daily images from time-lapse recordings. We propose a novel hybrid model that combines DINOv2, a transformer-based vision model, with an enhanced long short-term memory (LSTM) network featuring a multi-head attention layer. DINOv2 extracts meaningful features from embryo images, and the LSTM model then uses these features to analyze embryo development over time and generate final predictions. We tested our model on a real dataset of 704 embryo videos. The model achieved 96.4% accuracy, surpassing existing methods. It also performs well with missing frames, making it valuable for many IVF laboratories with limited imaging systems. Our approach can assist embryologists in selecting better embryos more efficiently and with greater confidence. 

---
# Motif-Video 2B: Technical Report 

**Authors**: Junghwan Lim, Wai Ting Cheung, Minsu Ha, Beomgyu Kim, Taewhan Kim, Haesol Lee, Dongpin Oh, Jeesoo Lee, Taehyun Kim, Minjae Kim, Sungmin Lee, Hyeyeon Cho, Dahye Choi, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Dongseok Kim, Jangwoong Kim, Youngrok Kim, Hyukjin Kweon, Hongjoo Lee, Jeongdoo Lee, Junhyeok Lee, Eunhwan Park, Yeongjae Park, Bokki Ryu, Dongjoo Weon  

**Link**: [PDF](https://arxiv.org/pdf/2604.16503)  

**Abstract**: Training strong video generation models usually requires massive datasets, large parameter counts, and substantial compute. In this work, we ask whether strong text-to-video quality is possible at a much smaller budget: fewer than 10M clips and less than 100,000 H200 GPU hours. Our core claim is that part of the answer lies in how model capacity is organized, not only in how much of it is used. In video generation, prompt alignment, temporal consistency, and fine-detail recovery can interfere with one another when they are handled through the same pathway. Motif-Video 2B addresses this by separating these roles architecturally, rather than relying on scale alone. The model combines two key ideas. First, Shared Cross-Attention strengthens text control when video token sequences become long. Second, a three-part backbone separates early fusion, joint representation learning, and detail refinement. To make this design effective under a limited compute budget, we pair it with an efficient training recipe based on dynamic token routing and early-phase feature alignment to a frozen pretrained video encoder. Our analysis shows that later blocks develop clearer cross-frame attention structure than standard single-stream baselines. On VBench, Motif-Video~2B reaches 83.76\%, surpassing Wan2.1 14B while using 7$\times$ fewer parameters and substantially less training data. These results suggest that careful architectural specialization, combined with an efficiency-oriented training recipe, can narrow or exceed the quality gap typically associated with much larger video models. 

---
# HQA-VLAttack: Towards High Quality Adversarial Attack on Vision-Language Pre-Trained Models 

**Authors**: Han Liu, Jiaqi Li, Zhi Xu, Xiaotong Zhang, Xiaoming Xu, Fenglong Ma, Yuanman Li, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16499)  

**Abstract**: Black-box adversarial attack on vision-language pre-trained models is a practical and challenging task, as text and image perturbations need to be considered simultaneously, and only the predicted results are accessible. Research on this problem is in its infancy, and only a handful of methods are available. Nevertheless, existing methods either rely on a complex iterative cross-search strategy, which inevitably consumes numerous queries, or only consider reducing the similarity of positive image-text pairs but ignore that of negative ones, which will also be implicitly diminished, thus inevitably affecting the attack performance. To alleviate the above issues, we propose a simple yet effective framework to generate high-quality adversarial examples on vision-language pre-trained models, named HQA-VLAttack, which consists of text and image attack stages. For text perturbation generation, it leverages the counter-fitting word vector to generate the substitute word set, thus guaranteeing the semantic consistency between the substitute word and the original word. For image perturbation generation, it first initializes the image adversarial example via the layer-importance guided strategy, and then utilizes contrastive learning to optimize the image adversarial perturbation, which ensures that the similarity of positive image-text pairs is decreased while that of negative image-text pairs is increased. In this way, the optimized adversarial images and texts are more likely to retrieve negative examples, thereby enhancing the attack success rate. Experimental results on three benchmark datasets demonstrate that HQA-VLAttack significantly outperforms strong baselines in terms of attack success rate. 

---
# Forge-UGC: FX optimization and register-graph engine for universal graph compiler 

**Authors**: Satyam Kumar, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.16498)  

**Abstract**: We present Forge-UGC (FX Optimization and Register-Graph Engine for Universal Graph Compilation), a four-phase compiler for transformer deployment on heterogeneous accelerator hardware, validated on Intel AI Boost NPU. Existing frameworks such as OpenVINO and ONNX Runtime often use opaque compilation pipelines, limited pass-level visibility, and weak buffer management, which can lead to higher compilation cost and runtime overhead. Forge-UGC addresses this with a hardware-agnostic design that separates graph capture, optimization, intermediate representation lowering, and backend scheduling. Phase 1 captures graphs with this http URL at the ATen operator level, supporting modern transformer components such as rotary position embeddings, grouped-query attention, and SwiGLU without manual decomposition. Phase 2 applies six optimization passes: dead code elimination, common subexpression elimination, constant folding, attention fusion, operator fusion, and layout optimization, reducing graph node count by 14.2 to 21.9%. Phase 3 lowers the optimized graph into a typed intermediate representation with explicit virtual register assignments. Phase 4 performs liveness analysis, linear-scan buffer allocation, reducing peak buffer count by 30 to 48%, and device-affinity scheduling, reducing NPU-CPU transitions by 42 to 65%. Across six model families ranging from 125M to 8B parameters, evaluated on WikiText-103 and GLUE, Forge-UGC delivers 6.9 to 9.2x faster compilation than OpenVINO and ONNX Runtime, 18.2 to 35.7% lower inference latency, and 30.2 to 40.9% lower energy per inference. Fidelity is preserved, with max absolute logit differences below 2.1e-5 and KL divergence below 8.4e-9. We also introduce Fusion Gain Ratio, Compilation Efficiency Index, and per-pass execution profiling for systematic evaluation of NPU compilation pipelines. 

---
# Gradient-Free Continual Learning in Spiking Neural Networks via Inter-Spike Interval Regularization 

**Authors**: Samrendra Roy, Kazuma Kobayashi, Souvik Chakraborty, Sajedul Talukder, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2604.16496)  

**Abstract**: Continual learning, the ability to acquire new tasks sequentially without forgetting prior knowledge, is essential for deploying neural networks in dynamic real-world environments, from nuclear digital twin monitoring to grid-edge fault detection. Existing synaptic importance methods, such as Elastic Weight Consolidation (EWC) and Synaptic Intelligence (SI), rely on gradient computation, making them incompatible with neuromorphic hardware that lacks backpropagation support. We propose ISI-CV, the first gradient-free synaptic importance metric for SNN continual learning, derived from the Coefficient of Variation (CV) of Inter-Spike Intervals (ISIs). Neurons that fire regularly (low CV) encode stable, task-relevant features and are protected from overwriting; neurons with irregular firing are permitted to adapt freely. ISI-CV requires only spike time counters and integer arithmetic, all of which are native to every neuromorphic chip. We evaluate on four benchmarks of increasing difficulty: Split-MNIST, Permuted-MNIST, Split-FashionMNIST, and Split-N-MNIST using real Dynamic Vision Sensor (DVS) event data. Across three seeds, ISI-CV achieves zero forgetting (AF = 0.000 +/- 0.000) on Split-MNIST and Split-FashionMNIST, near-zero forgetting on Permuted-MNIST (AF = 0.001 +/- 0.000), and the highest accuracy with the lowest forgetting on real neuromorphic DVS data (AA = 0.820 +/- 0.012, AF = 0.221 +/- 0.014). On N-MNIST, gradient-based methods produce unreliable importance estimates and perform worse than no regularization; ISI-CV avoids this failure by design. 

---
# NL2SQLBench: A Modular Benchmarking Framework for LLM-Enabled NL2SQL Solutions 

**Authors**: Shizheng Hou, Wenqi Pei, Nuo Chen, Quang-Trung Ta, Peng Lu, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16493)  

**Abstract**: Natural Language to SQL (NL2SQL) technology empowers non-expert users to query relational databases without requiring SQL expertise. While large language models (LLMs) have greatly improved NL2SQL algorithms, their rapid development outpaces systematic evaluation, leaving a critical gap in understanding their effectiveness, efficiency, and limitations. To this end, we present NL2SQLBench, the first modular evaluation and benchmarking framework for LLM-enabled NL2SQL approaches. Specifically, we dissect NL2SQL systems into three core modules: Schema Selection, Candidate Generation, and Query Revision. For each module, we comprehensively review existing strategies and propose novel fine-grained metrics that systematically quantify module-level effectiveness and efficiency. We further implement these metrics in a flexible multi-agent framework, allowing configurable benchmarking across diverse NL2SQL approaches. Leveraging NL2SQLBench, we rigorously evaluate ten representative open-source methods on two datasets, the BIRD development set and the ScienceBenchmark development set, using two LLMs, DeepSeek-V3 and GPT-4o mini. We systematically assess each approach across the three core modules and evaluate multiple critical performance dimensions. Our evaluation reveals significant gaps in existing NL2SQL methods, highlighting not only substantial room for accuracy improvements but also the significant computational inefficiency, which severely hampers real-world adoption. Furthermore, our analysis identifies critical shortcomings in current benchmark datasets and evaluation rules, emphasizing issues such as inaccurate gold SQL annotations and limitations in existing evaluation rules. By synthesizing these insights into a unified benchmarking, our study establishes a clear reference point for fair comparison and serves as essential guidance for future targeted innovation in NL2SQL technology. 

---
# LayerCache: Exploiting Layer-wise Velocity Heterogeneity for Efficient Flow Matching Inference 

**Authors**: Guandong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16492)  

**Abstract**: Flow Matching models achieve state-of-the-art image generation quality but incur substantial inference cost due to iterative denoising through large Transformer networks. We observe that different layer groups within a Transformer exhibit markedly heterogeneous velocity dynamics: shallow layers are highly stable and amenable to aggressive caching, while deep layers undergo large velocity changes that demand full computation. Existing caching methods, however, treat the entire Transformer as a monolithic unit, applying a single caching decision per timestep and thus failing to exploit this heterogeneity. Based on this finding, we propose LayerCache, a layer-aware caching framework that partitions the Transformer into layer groups and makes independent, per-group caching decisions at each denoising step. LayerCache introduces an adaptive JVP span K selection mechanism that leverages per-group stability measurements to balance estimation accuracy and computational savings. We formulate a three-dimensional scheduling problem over timesteps, layer groups, and JVP span, and solve it with a greedy budget allocation algorithm. On Qwen-Image (1024x1024, 50 steps), LayerCache achieves PSNR 37.46 dB (+5.38 dB over MeanCache), SSIM 0.9834, and LPIPS 0.0178 (a 70% reduction over MeanCache) at 1.37x speedup, dominating all prior caching methods on the quality-speed Pareto frontier. 

---
# A Lightweight Transformer for Pain Recognition from Brain Activity 

**Authors**: Stefanos Gkikas, Christian Arzate Cruz, Yu Fang, Lu Cao, Muhammad Umar Khan, Thomas Kassiotis, Giorgos Giannakakis, Raul Fernandez Rojas, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2604.16491)  

**Abstract**: Pain is a multifaceted and widespread phenomenon with substantial clinical and societal burden, making reliable automated assessment a critical objective. This paper presents a lightweight transformer architecture that fuses multiple fNIRS representations through a unified tokenization mechanism, enabling joint modeling of complementary signal views without requiring modality-specific adaptations or increasing architectural complexity. The proposed token-mixing strategy preserves spatial, temporal, and time-frequency characteristics by projecting heterogeneous inputs onto a shared latent representation, using a structured segmentation scheme to control the granularity of local aggregation and global interaction. The model is evaluated on the AI4Pain dataset using stacked raw waveform and power spectral density representations of fNIRS inputs. Experimental results demonstrate competitive pain recognition performance while remaining computationally compact, making the approach suitable for real-time inference on both GPU and CPU hardware. 

---
# An Uncertainty-Aware Loss Function Incorporating Fuzzy Logic: Application to MRI Brain Image Segmentation 

**Authors**: Hanuman Verma, Akshansh Gupta, Pranabesh Maji, Saurav Mandal, Vijay Kumar Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2604.16490)  

**Abstract**: Accurate brain image segmentation, particularly for distinguishing various tissues from magnetic resonance imaging (MRI) images, plays a pivotal role in finding the neurological dis ease and medical image computing. In deep learning approaches, loss functions are very crucial for optimizing the model. In this study, we introduce a novel loss function integrating fuzzy logic to deals uncertainty issues in brain image segmentation into various tissues. It integrates the well-known categorical cross-entropy (CCE) loss function and fuzzy entropy based on fuzzy logic. By employing fuzzy logic, this loss function accounts for the inherent uncertainties in pixel classifications. The proposed loss function has been evaluated on two publicly available benchmark datasets, IBSR and OASIS, using two widely recognised architectures, U-Net and U-Net++. Experimental results demonstrate that the trained model with proposed loss function provided better results in comparison to the CCE optimisation function in terms of various performance metrics. Additionally, it effectively enhances segmentation performance while handling meaningful uncer tainty during training. The findings suggest that this approach not only improves segmentation outcomes but also contributes to the reliability of model predictions. 

---
# Geometry-Aware CLIP Retrieval via Local Cross-Modal Alignment and Steering 

**Authors**: Nirmalendu Prakash, Narmeen Fatimah Oozeer, Xin Su, Phillip Howard, Shaan Shah, Zoe Wanying He, Shuang Wu, Shivam Raval, Roy Ka-Wei Lee, Meenakshi Khosla, Amir Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.16487)  

**Abstract**: CLIP retrieval is typically framed as a pointwise similarity problem in a shared embedding space. While CLIP achieves strong global cross-modal alignment, many retrieval failures arise from local geometric inconsistencies: nearby items are incorrectly ordered, leading to systematic confusions (e.g., pentagon vs. hexagon) and produces diffuse, weakly controlled result sets. Prior work largely optimizes for point wise relevance or finetuning to mitigate these problems. We instead view retrieval as a problem of neighborhood alignment. Our work introduces (1) neighborhood-level re-ranking via Hungarian matching, which rewards structural consistency; (2) query-conditioned local steering, where directions derived from contrastive neighborhoods around the query reshape retrieval. We show that these techniques improve retrieval performance on attribute-binding and compositional retrieval tasks. Together, these methods operate on local neighborhoods but serve different roles: re-ranking rewards alignment whereas local steering controls neighborhood structure. This shows that retrieval quality and controllability depend critically on local structure, which can be exploited at inference time without retraining. 

---
# Saccade Attention Networks: Using Transfer Learning of Attention to Reduce Network Sizes 

**Authors**: Marc Estafanous  

**Link**: [PDF](https://arxiv.org/pdf/2604.16485)  

**Abstract**: One of the limitations of transformer networks is the sequence length due to the quadratic nature of the attention matrix. Classical self attention uses the entire sequence length, however, the actual attention being used is sparse. Humans use a form of sparse attention when analyzing an image or scene called saccades. Focusing on key features greatly reduces computation time. By using a network (Saccade Attention Network) to learn where to attend from a large pre-trained model, we can use it to pre-process images and greatly reduce network size by reducing the input sequence length to just the key features being attended to. Our results indicate that you can reduce calculations by close to 80% and produce similar results. 

---
# DexWorldModel: Causal Latent World Modeling towards Automated Learning of Embodied Tasks 

**Authors**: Yueci Deng, Guiliang Liu, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.16484)  

**Abstract**: Deploying generative World-Action Models for manipulation is severely bottlenecked by redundant pixel-level reconstruction, $\mathcal{O}(T)$ memory scaling, and sequential inference latency. We introduce the Causal Latent World Model (CLWM), which employs DINOv3 features as generative targets to disentangle interaction semantics from visual noise, yielding highly robust domain generalization. To overcome memory scaling, CLWM features a Dual-State Test-Time Training (TTT) Memory that guarantees a strict $\mathcal{O}(1)$ footprint for long-horizon tasks. To overcome deployment latency, we propose Speculative Asynchronous Inference (SAI) to mask partial diffusion denoising behind physical execution, cutting blocking latency by about $50\%$. To scale robust policies, we present EmbodiChain, an online framework that establishes the Efficiency Law by injecting an infinite flow of physics-grounded trajectories during training. Extensive experiments validate that CLWM achieves state-of-the-art performance in complex dual-arm simulation and unprecedented zero-shot sim-to-real transfer on physical robots, outperforming baselines explicitly finetuned on real-world data. 

---
# Dynamic Eraser for Guided Concept Erasure in Diffusion Models 

**Authors**: Qinghui Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16483)  

**Abstract**: Concept erasure in Text-To-Image (T2I) diffusion models is vital for safe content generation, but existing inference-time methods face significant limitations. Feature-correction approaches often cause uncontrolled over-correction, while token-level interventions struggle with semantic granularity and context. Moreover, both types of methods are prone to severe semantic drift or even complete representation collapse. To address these challenges, we present Dynamic Semantic Steering (DSS), a lightweight, training-free framework for interpretable and controllable concept erasure. DSS introduces: 1) Sensitive Semantic Boundary Modeling (SSBM) to automate the discovery of safe semantic anchors, and 2) Sensitive Semantic Guidance (SSG), which leverages cross-attention features for precise detection and performs correction via a closed-form solution derived from a well-posed objective. This ensures optimal suppression of sensitive content while preserving benign semantics. DSS achieves an average erasure rate of 91.0\%, significantly outperforming SOTA methods (from 18.6\% to 85.9\%) with minimal impact on output fidelity. 

---
# Erasing Thousands of Concepts: Towards Scalable and Practical Concept Erasure for Text-to-Image Diffusion Models 

**Authors**: Hoigi Seo, Byung Hyun Lee, Jaehyun Cho, Sungjin Lim, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2604.16481)  

**Abstract**: Large-scale text-to-image (T2I) diffusion models deliver remarkable visual fidelity but pose safety risks due to their capacity to reproduce undesirable content, such as copyrighted ones. Concept erasure has emerged as a mitigation strategy, yet existing approaches struggle to balance scalability, precision, and robustness, which restricts their applicability to erasing only a few hundred concepts. To address these limitations, we present Erasing Thousands of Concepts (ETC), a scalable framework capable of erasing thousands of concepts while preserving generation quality. Our method first models low-rank concept distributions via a Student's t-distribution Mixture Model (tMM). It enables pin-point erasure of target concepts via affine optimal transport while preserving others by anchoring the boundaries of target concept distributions without pre-defined anchor concepts. We then train a Mixture-of-Experts (MoE)-based module, termed MoEraser, which removes target embeddings while preserving the anchor embeddings. By injecting noise into the text embedding projector and fine-tuning MoEraser for recovery, our framework achieves robustness to white-box attack such as module removal. Extensive experiments on over 2,000 concepts across heterogeneous domains and diffusion models demerate state-of-the-art scalability and precision in large-scale concept erasure. 

---
# Latent-Compressed Variational Autoencoder for Video Diffusion Models 

**Authors**: Jiarui Guan, Wenshuai Zhao, Zhengtao Zou, Juho Kannala, Arno Solin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16479)  

**Abstract**: Video variational autoencoders (VAEs) used in latent diffusion models typically require a sufficiently large number of latent channels to ensure high-quality video reconstruction. However, recent studies have revealed that an excessive number of latent channels can impede the convergence of latent diffusion models and deteriorate their generative performance, even when reconstruction quality remains high. We propose a latent compression method that removes high-frequency components in video latent representations rather than directly reducing the number of channels, which often compromises reconstruction fidelity. Experimental results demonstrate that the proposed method achieves superior video reconstruction quality compared to strong baselines while maintaining the same overall compression ratio. 

---
# Spike-driven Large Language Model 

**Authors**: Han Xu, Xuerui Qiu, Baiyu Chen, Xinhao Luo, Xingrun Xing, Jiahong Zhang, Bo Lei, Tiejun Huang, Bo Xu, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16475)  

**Abstract**: Current Large Language Models (LLMs) are primarily based on large-scale dense matrix multiplications. Inspired by the brain's information processing mechanism, we explore the fundamental question: how to effectively integrate the brain's spiking-driven characteristics into LLM inference. Spiking Neural Networks (SNNs) possess spike-driven characteristics, and some works have attempted to combine SNNs with Transformers. However, achieving spike-driven LLMs with billions of parameters, relying solely on sparse additions, remains a challenge in the SNN field. To address the issues of limited representational capacity and sparsity in existing spike encoding schemes at the LLM level, we propose SDLLM, a spike-driven large language model that eliminates dense matrix multiplications through sparse addition operations. Specifically, we use the plug-and-play gamma-SQP two-step spike encoding method to ensure that the quantization process aligns with the model's semantic space, mitigating representation degradation caused by binary spikes. Furthermore, we introduce bidirectional encoding under symmetric quantization and membrane potential clipping mechanisms, leading to spike trains with no or low firing counts dominating, significantly reducing the model's spike firing rate, while halving the number of time steps. Experimental results show that SDLLM not only significantly reduces inference costs but also achieves state-of-the-art task performance under the spike-based paradigm. For example, compared to previous spike-based LLMs, SDLLM reduces energy consumption by 7x and improves accuracy by 4.2%. Our model provides inspiration for the architecture design of the next generation of event-driven neuromorphic chips. 

---
# Full Feature Spiking Neural Network Simulation on Micro-Controllers for Neuromorphic Applications at the Edge 

**Authors**: L. Niedermeier, J. L. Krichmar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16474)  

**Abstract**: Microcontroller units (MCU), which have an order of magnitude lower Size, Weight and Power (SWaP) than standard computers, makes them suitable for applications at the edge. Neuromorphic computing, which can realize low SWaP, relies on Spiking Neural Networks (SNNs). Until now, software based simulations of SNNs required GPU-based workstations, application classified core processors such as the ARM Cortex-A53, or specialized hardware like Intel's Loihi. In the present work, we demonstrate that the SNN simulator CARLsim can run its full feature set on a MCU RP2350 with 8 MB memory. We accomplished this by utilizing IEEE 16-bit float point numbers, which reduced memory requirements without loss of function. We were able to run the Synfire4 benchmark which comprises 1200 neurons. The accuracy was 97.5% compared to the standard single precision numbers. Furthermore, we show that CARLsim runs a Synfire4 benchmark scaled-down to 186 neurons on a MCU in real-time at only 20 mW. Compared to the smallest application class ARM processor used by Raspberry in their Pi Zero 2 W, our MCU implementation is five times more energy efficient for the SNN itself, and an order of magnitude better when compared to the complete SoC (MCU/CPU + Board). 

---
# Training Language Models for Bilateral Trade with Private Information 

**Authors**: Dirk Bergemann, Soheil Ghili, Xinyang Hu, Chuanhao Li, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16472)  

**Abstract**: Bilateral bargaining under incomplete information provides a controlled testbed for evaluating large language model (LLM) agent capabilities. Bilateral trade demands individual rationality, strategic surplus maximization, and cooperation to realize gains from trade. We develop a structured bargaining environment where LLMs negotiate via tool calls within an event-driven simulator, separating binding offers from natural-language messages to enable automated evaluation. The environment serves two purposes: as a benchmark for frontier models and as a training environment for open-weight models via reinforcement learning.
In benchmark experiments, a round-robin tournament among five frontier models (15,000 negotiations) reveals that effective strategies implement price discrimination through sequential offers. Aggressive anchoring, calibrated concession, and temporal patience correlate with the highest surplus share and deal rate. Accommodating strategies that concede quickly disable price discrimination in the buyer role, yielding the lowest surplus capture and deal completion. Stronger models scale their behavior proportionally to item value, maintaining performance across price tiers; weaker models perform well only when wide zones of possible agreement offset suboptimal strategies.
In training experiments, we fine-tune Qwen3 (8B, 14B) via supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) against a fixed frontier opponent. These stages optimize competing objectives: SFT approximately doubles surplus share but reduces deal rates, while RL recovers deal rates but erodes surplus gains, reflecting the reward structure. SFT also compresses surplus variation across price tiers, which generalizes to unseen opponents, suggesting that behavioral cloning instills proportional strategies rather than memorized price points. 

---
# Semantic Channel Theory: Deductive Compression and Structural Fidelity for Multi-Agent Communication 

**Authors**: Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16471)  

**Abstract**: Shannon's information theory deliberately excludes message semantics. This paper develops a rigorous framework for semantic communication that integrates formal proof systems with Shannon-theoretic tools. We introduce an axiomatic information model comprising Lsem-definable state sets linked by computable enabling maps, and define the semantic channel as a composition of Markov kernels whose supports respect the enabling structure. A fixed proof system induces an irredundant semantic core and a derivation-depth stratification, enabling four distortion measures of increasing semantic depth: Hamming, closure, depth, and a parameterized composite. Six families of computable semantic channel invariants are defined and their inter-relationships established, including a data processing bound, a semantic Fano bound, and an ideal-channel collapse theorem. The central quantitative result is a deductive compression gain: under closure-based fidelity, the minimum block length is determined by the irredundant core size rather than the full knowledge-base size. We instantiate the framework for heterogeneous multi-agent communication, introducing an overlap decomposition that yields necessary and sufficient conditions for closure-reliable communication. A semantic bottleneck phenomenon is identified in broadcast settings: vocabulary mismatch imposes irreducible fidelity limitations even over noiseless carriers. All results are verified on an explicit Datalog instance. 

---
# B-PASTE: Beam-Aware Pattern-Guided Speculative Execution for Resource-Constrained LLM Agents 

**Authors**: Yanfei Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.16469)  

**Abstract**: LLM agents execute in an interleaved reasoning-and-action loop, where future tool calls cannot be launched until the current reasoning step completes. This serial dependency inflates end-to-end latency and leaves the model idle while waiting for tool execution. Prior work, Pattern-Aware Speculative Tool Execution (PASTE), mitigates this bottleneck by speculating likely future tool invocations from mined control-flow and data-flow regularities. However, PASTE is tool-centric and speculates only individual invocations rather than bounded future branches.
We propose B-PASTE, a beam-aware extension that lifts speculation from single tools to local branch hypotheses under strict resource constraints. B-PASTE maintains a bounded beam of future execution subgraphs, ranks them by expected critical-path reduction rather than raw execution probability, and schedules only high-value branch prefixes on transient slack resources. It explicitly models co-run interference, downstream unlock value, and state-safety constraints, enabling the system to prioritize serial fast-path execution when early completion unlocks valuable future work, while still exploiting safe parallelism under low contention.
This design is especially important for edge-side deployments, where speculative work must not steal scarce resources from latency-critical authoritative execution. Preliminary internal testing on Thor-class edge environments shows up to 1.4X end-to-end speedup, suggesting that branch-aware speculative execution remains effective even under tight resource budgets. 

---
# MLE-Toolbox: An Open-Source Toolbox for Comprehensive EEG and MEG Data Analysis 

**Authors**: Xiaobo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16463)  

**Abstract**: MLE-Toolbox is a comprehensive open-source MATLAB toolbox for end-to-end analysis of magnetoencephalography (MEG) and electroencephalography (EEG) data. Inspired by widely used neuroimaging platforms such as Brainstorm and FieldTrip, it integrates the full analysis pipeline within a unified and user-friendly graphical interface (GUI), covering raw data import, preprocessing, source localization, functional connectivity, oscillatory analysis, and machine learning-based classification. The toolbox includes automated artifact rejection methods, including independent component analysis (ICA), signal-space projection (SSP), and signal-space separation (SSS); multiple source localization approaches, including minimum norm estimation (MNE), dynamic statistical parametric mapping (dSPM), standardized low-resolution brain electromagnetic tomography (sLORETA), and beamforming; multi-atlas parcellation with anatomical visualization; spectral power analysis with frequency-band brain mapping; phase-amplitude coupling (PAC); graph-theoretic brain network analysis; and integrated machine learning and deep learning classifiers. MLE-Toolbox also provides native interoperability with Brainstorm, FieldTrip, EEGLAB, and FreeSurfer, allowing researchers to build on established workflows while benefiting from additional automation, interactive visualization, and one-click academic report generation. Freely available for non-commercial use, MLE-Toolbox is designed to lower the barrier to rigorous, reproducible MEG/EEG research. 

---
# From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration 

**Authors**: Jiaqi Shi, Yuechan Li, Xulong Zhang, Xiaoyang Qu, Jianzong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16462)  

**Abstract**: High-resolution Multimodal Large Language Models (MLLMs) face prohibitive computational costs during inference due to the explosion of visual tokens. Existing acceleration strategies, such as token pruning or layer sparsity, suffer from severe "backbone dependency", performing well on Vicuna or Mistral architectures (e.g., LLaVA) but causing significant performance degradation when transferred to architectures like Qwen. To address this, we leverage truncated matrix entropy to uncover a universal three-stage inference lifecycle, decoupling visual redundancy into universal Intrinsic Visual Redundancy (IVR) and architecture-dependent Secondary Saturation Redundancy (SSR). Guided by this insight, we propose HalfV, a framework that first mitigates IVR via a unified pruning strategy and then adaptively handles SSR based on its specific manifestation. Experiments demonstrate that HalfV achieves superior efficiency-performance trade-offs across diverse backbones. Notably, on Qwen25-VL, it retains 96.8\% performance at a 4.1$\times$ FLOPs speedup, significantly outperforming state-of-the-art baselines. Our code is available at this https URL. 

---
# Deep Hierarchical Knowledge Loss for Fault Intensity Diagnosis 

**Authors**: Yu Sha, Shuiping Gou, Bo Liu, Haofan Lu, Ningtao Liu, Jiahui Fu, Horst Stoecker, Domagoj Vnucec, Nadine Wetzstein, Andreas Widl, Kai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.16459)  

**Abstract**: Fault intensity diagnosis (FID) plays a pivotal role in intelligent manufacturing while neglecting dependencies among target classes hinders its practical deployment. This paper introduces a novel and general framework with deep hierarchical knowledge loss (DHK) to achieve hierarchical consistent representation and prediction. We develop a novel hierarchical tree loss to enable a holistic mapping of same-attribute classes, leveraging tree-based positive and negative hierarchical knowledge constraints. We further design a focal hierarchical tree loss to enhance its extensibility and devise two adaptive weighting schemes based on tree height. In addition, we propose a group tree triplet loss with hierarchical dynamic margin by incorporating hierarchical group concepts and tree distance to model boundary structural knowledge across classes. The joint two losses significantly improve the recognition of subtle faults. Extensive experiments are performed on four real-world datasets from various industrial domains (three cavitation datasets from SAMSON AG and one publicly available dataset) for FID, all showing superior results and outperforming recent state-of-the-art FID methods. 

---
# EchoChain: A Full-Duplex Benchmark for State-Update Reasoning Under Interruptions 

**Authors**: Smit Nautambhai Modi, Gandharv Mahajan, Marc Wetter, Randall Welles  

**Link**: [PDF](https://arxiv.org/pdf/2604.16456)  

**Abstract**: Real-time voice assistants must revise task state when users interrupt mid-response, but existing spoken-dialog benchmarks largely evaluate turn-based interaction and miss this failure mode. We introduce EchoChain, a controlled benchmark for evaluating full-duplex state-update reasoning under mid-speech interruptions. EchoChain identifies three recurring failure patterns in post-interruption continuations: contextual inertia, interruption amnesia, and objective displacement. The benchmark generates scenario-driven conversations and injects interruptions at a standardized point relative to assistant speech onset, enabling controlled cross-model comparison. In a paired half-duplex control, total failures drop by 40.2% relative to interrupted runs, indicating that many errors are driven by state-update reasoning under interruption rather than task difficulty alone. Across evaluated real-time voice models, no system exceeds a 50% pass rate, showing substantial room for improvement in mid-generation state revision. EchoChain provides a reproducible benchmark for diagnosing state-update reasoning failures in full-duplex voice interaction. 

---
# Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo 

**Authors**: Jelena Markovic-Voronov, Wenhui Zhu, Bo Long, Zhipeng Wang, Suyash Gupta, Kayhan Behdin, Bee-Chung Chen, Deepak Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16453)  

**Abstract**: We introduce a principled probabilistic framework for reward-guided decoding in large language models, addressing the limitations of standard decoding methods that optimize token-level likelihood rather than sequence-level quality. Our method defines a reward-augmented target distribution over complete sequences by combining model transition probabilities with prefix-dependent reward potentials. Importantly, the approach is training-free: it leaves model weights unchanged and instead modifies the inference distribution via reward potentials, with all gains arising purely from inference-time sampling. To sample from this distribution, we develop Sequential Monte Carlo algorithms, including a computationally efficient prefix-only variant and a lookahead variant whose intermediate targets match the exact marginals of the full sequence distribution. The framework also integrates resample-move updates with Metropolis-Hastings rejuvenation and supports block-wise generation, subsuming common decoding strategies such as temperature sampling and power-tempered objectives. Empirical results across three 7B models show significant gains. On code generation (HumanEval), our method improves base performance by up to 54.9% and surpasses the strongest sampling baselines by 9.1%-15.3%. On mathematical reasoning (MATH500), it achieves gains of up to 8.8%. Notably, it reaches 87.8% on HumanEval and 78.4% on MATH500 with Qwen2.5-7B, consistently outperforming the reinforcement learning method GRPO. 

---
# SAND: The Challenge on Speech Analysis for Neurodegenerative Disease Assessment 

**Authors**: Giovanna Sannino, Ivanoe De Falco, Nadia Brancati, Laura Verde, Maria Frucci, Daniel Riccio, Vincenzo Bevilacqua, Antonio Di Marino, Lucia Aruta, Valentina Virginia Iuzzolino, Gianmaria Senerchia, Myriam Spisto, Raffaele Dubbioso  

**Link**: [PDF](https://arxiv.org/pdf/2604.16445)  

**Abstract**: Recent advances in Artificial Intelligence (AI) and the exploration of noninvasive, objective biomarkers, such as speech signals, have encouraged the development of algorithms to support the early diagnosis of neurodegenerative diseases, including Amyotrophic Lateral Sclerosis (ALS). Voice changes in subjects suffering from ALS typically manifest as progressive dysarthria, which is a prominent neurodegenerative symptom because it affects patients as the disease progresses. Since voice signals are complex data, the development and use of advanced AI techniques are fundamental to extracting distinctive patterns from them. Validating AI algorithms for ALS diagnosis and monitoring using voice signals is challenging, particularly due to the lack of annotated reference datasets. In this work, we present the outcome of a collaboration between a multidisciplinary team of clinicians and Machine Learning experts to create both a clinically annotated validation dataset and the "Speech Analysis for Neurodegenerative Diseases" (SAND) challenge based on it. Specifically, by analyzing voice disorders, the SAND challenge provides an opportunity to develop, test, and evaluate AI models for the automatic early identification and prediction of ALS disease progression. 

---
# The Breakthrough of Sleep: A Contactless Approach for Accurate Sleep Stage Detection Using the Sleepal AI Lamp 

**Authors**: Zhuo Diao, Yueting Li, Jianpeng Wang, Shengyu Guan, Xinwei Wang, Wenxiong Cui, Xin Shi, Tong Liu, Kailai Sun, Jingyu Wang, Dian Fan, Thomas Penzel  

**Link**: [PDF](https://arxiv.org/pdf/2604.16442)  

**Abstract**: Sleep staging is essential for the assessment of sleep quality and the diagnosis of sleep-related disorders. Conventional polysomnography (PSG), while considered the gold standard, is intrusive, labor-intensive, and unsuitable for long-term monitoring. This study evaluates the performance of the Sleepal AI Lamp, a contactless, radar-based consumer-grade sleep tracker, in comparison with gold-standard polysomnography (PSG), using a large-scale dataset comprising 1022 overnight recordings. We extract multi-scale respiratory and motion-related features from radar signals to train a frequency-augmented deep learning model. For the binary sleep-wake classification task, experimental results demonstrated that the model achieved an accuracy of 92.8% alongside a macro-averaged F1 score of 0.895. For four-stage classification (wake, light NREM (N1 + N2), deep NREM (N3), REM), the model achieved an accuracy of 78.5% with a Cohen's kappa coefficient of 0.695 in healthy individuals and maintained a stable accuracy of 77.2% with a kappa of 0.677 in a heterogeneous population including patients with varying severities of obstructive sleep apnea (OSA). These experimental results demonstrate that the sleep staging performance of the contactless Sleepal AI Lamp is in high agreement with expert-labeled PSG sleep stages. Our findings suggest that non-contact radar sensing, combined with advanced temporal modeling, can provide reliable sleep staging performance without requiring physical contact or wearable devices. Owing to its unobtrusive nature, ease of deployment, and robustness to long-term use, the contactless Sleepal AI Lamp shows strong potential for clinical screening, home-based sleep assessment, and continuous longitudinal sleep monitoring in real-world medical and healthcare applications. 

---
# iPhoneme: Brain-to-Text Communication for ALS Using ConformerXL Decoding 

**Authors**: Yoonmin Cha, Dawit Chun, Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.16441)  

**Abstract**: Brain-computer interfaces (BCIs) for speech restoration hold transformative potential for the approximately 173,000--232,500 individuals worldwide with ALS-related dysarthria. Despite recent progress, high-performance speech BCIs have been demonstrated in only 22--31 patients globally, largely due to limitations in neural decoding accuracy and practical input interfaces. We present iPhoneme, a brain-to-text communication system that jointly addresses these challenges through integrated modeling and interaction design. The system combines a deep learning phoneme decoder based on a modified Conformer architecture (ConformerXL, 192.9M parameters) with a gaze-assisted phoneme input interface that mitigates the Midas touch problem in eye-tracking systems. The acoustic model incorporates a temporal prenet with multi-scale dilated convolutions and bidirectional GRU for neural jitter correction, temporal subsampling for CTC stability, and Pre-RMSNorm stabilization across 12 encoder blocks, trained with AdamW and cosine scheduling. On the interaction side, iPhoneme introduces a chorded gaze-plus-silent-speech paradigm that replaces dwell-time selection, enabling more efficient input. We evaluate the system on the T15 dataset (45 sessions, 8,071 trials) of 256-channel intracranial EEG from speech motor cortex regions. A 6-gram phoneme language model trained on 3.1M sequences, combined with WFST beam search (beam=128), achieves 92.14% phoneme accuracy (7.86% PER) and 73.39% word accuracy (26.61% WER), approximately 3% above prior state-of-the-art. The system operates on CPU with 180 ms latency, demonstrating real-time, high-accuracy brain-to-text communication for ALS. 

---
# LatentMimic: Terrain-Adaptive Locomotion via Latent Space Imitation 

**Authors**: Zhiquan Wang, Yunyu Liu, Dipam Patel, Ayush Kumar, Aniket Bera, Bedrich Benes  

**Link**: [PDF](https://arxiv.org/pdf/2604.16440)  

**Abstract**: Developing natural and diverse locomotion controllers for quadruped robots that can adapt to complex terrains while preserving motion style remains a significant challenge. Existing imitation-based methods face a fundamental optimization trade-off: strict adherence to motion capture (mocap) references penalizes the geometric deviations required for terrain adaptability, whereas terrain-centric policies often compromise stylistic fidelity. We introduce LatentMimic, a novel locomotion learning framework that decouples stylistic fidelity from geometric constraints. By minimizing the marginal latent divergence between the policy's state-action distribution and a learned mocap prior, our approach provides a conditional relaxation of rigid pose-tracking objectives. This formulation preserves gait topology while permitting independent end-effector adaptations for irregular terrains. We further introduce a terrain adaptation module with a dynamic replay buffer to resolve the policy's distribution shifts across different terrains. We validate our method across four locomotion styles and four terrains, demonstrating that LatentMimic enables effective terrain-adaptive locomotion, achieving higher terrain traversal success rates than state-of-the-art motion-tracking methods while maintaining high stylistic fidelity. 

---
# Sampling Matters: The Effect of ECG Frequency on Deep Learning-Based Atrial Fibrillation Detection 

**Authors**: Arjan Mahmuod, Adrian Rod Hammerstad, Muzaffar Yousef, Yngve Sebastian Heill, Jonas L. Isaksen, Jørgen K. Kanters, Pal Halvorsen, Vajira Thambawita  

**Link**: [PDF](https://arxiv.org/pdf/2604.16437)  

**Abstract**: Deep learning models for atrial fibrillation (AF) detection are increasingly trained on heterogeneous electrocardiogram (ECG) datasets with varying sampling frequencies, yet the specific consequences of these discrepancies on model performance, calibration, and robustness remain insufficiently characterized. To address this, we conducted a systematic benchmark using 12-lead, 10-second recordings from the PTB-XL dataset, resampled to target frequencies of 62, 100, 250, and 500 Hz, to evaluate a standard 1-D Convolutional Neural Network (CNN) and a hybrid CNN-Long Short-Term Memory (LSTM) architecture under a rigorous patient-safe cross-validation framework. Our analysis reveals that sampling frequency significantly impacts detection metrics in an architecture-dependent manner; the hybrid CNN-LSTM model demonstrated optimal performance and consistent calibration at intermediate frequencies (100-250 Hz), whereas the 1-D CNN baseline exhibited marked degradation in accuracy and sensitivity at 500 Hz, suggesting increased susceptibility to high-frequency noise. We conclude that ECG sampling frequency is a critical, underappreciated factor in arrhythmia detection, and future foundation models must explicitly control for temporal resolution to ensure clinical reliability and reproducibility. 

---
# Quantifying how AI Panels improve precision 

**Authors**: Nicholas CL Beale  

**Link**: [PDF](https://arxiv.org/pdf/2604.16432)  

**Abstract**: AI in applications like screening job applicants had become widespread, and may contribute to unemployment especially among the young. Biases in the AIs may become baked into the job selection process, but even in their absence, reliance on a single AI is problematic. In this paper we derive a simple formula to estimate, or at least place an upper bound on, the precision of such approaches for data resembling realistic CVs:
$P(q) \approx \frac{\rho n^b + q(1-\rho)}{1 + (n^b - 1)\rho}$ where $b \approx q^* + 0.8 (1 - \rho)$ and $q^*$ is $q$ clipped to $[0.07, 0.22]$ where $P(q)$ is the precision of the top $q$ quantile selected by a panel of $n$ AIs and $\rho$ is their average pairwise correlation. This equation provides a basis for considering how many AIs should be used in a Panel, depending on the importance of the decision. A quantitative discussion of the merits of using a diverse panel of AIs to support decision-making in such areas will move away from dangerous reliance on single AI systems and encourage a balanced assessment of the extent to which diversity needs to be built into the AI parts of the socioeconomic systems that are so important for our future. 

---
# Dimensional Criticality at Grokking Across MLPs and Transformers 

**Authors**: Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16431)  

**Abstract**: Abrupt transitions between distinct dynamical regimes are a hallmark of complex systems. Grokking in deep neural networks provides a striking example -- an abrupt transition from memorization to generalization long after training accuracy saturates -- yet robust macroscopic signatures of this transition remain elusive. Here we introduce \textbf{TDU--OFC} (Thresholded Diffusion Update--Olami-Feder-Christensen), an offline avalanche probe that converts gradient snapshots into cascade statistics and extracts a \emph{macroscopic observable} -- the time-resolved effective cascade dimension $D(t)$ -- via grokking-aligned finite-size scaling. Across Transformers trained on modular addition and MLPs trained on XOR, we discover a localized dynamical crossing of the Gaussian diffusion baseline $D=1$ precisely at the generalization transition. The crossing direction is task-dependent: modular addition descends through $D=1$ (approaching from $D>1$), while XOR ascends (from $D<1$). This opposite-direction convergence is consistent with attraction toward a candidate shared critical manifold, rather than trivial residence near $D \approx 1$. Negative controls confirm this picture: ungrokked runs remain supercritical ($D>1$) and never enter the post-transition regime. In addition, avalanche distributions exhibit heavy tails and finite-size scaling consistent with the dimensional exponent extracted from $D(t)$. Shadow-probe controls ($\alpha_{\mathrm{train}}=0$) confirm that $D(t)$ is non-invasive, and grokked trajectories diverge from ungrokked ones in $D(t)$ some $100$--$200$ epochs before the behavioral transition. 

---
# HalluSAE: Detecting Hallucinations in Large Language Models via Sparse Auto-Encoders 

**Authors**: Boshui Chen, Zhaoxin Fan, Ke Wang, Zhiying Leng, Faguo Wu, Hongwei Zheng, Yifan Sun, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16430)  

**Abstract**: Large Language Models (LLMs) are powerful and widely adopted, but their practical impact is limited by the well-known hallucination phenomenon. While recent hallucination detection methods have made notable progress, we find most of them overlook the dynamic nature and underlying mechanisms of it. To address this gap, we propose HalluSAE, a phase transition-inspired framework that models hallucination as a critical shift in the model's latent dynamics. By modeling the generation process as a trajectory through a potential energy landscape, HalluSAE identifies critical transition zones and attributes factual errors to specific high-energy sparse features. Our approach consists of three stages: (1) Potential Energy Empowered Phase Zone Localization via sparse autoencoders and a geometric potential energy metric; (2) Hallucination-related Sparse Feature Attribution using contrastive logit attribution; and (3) Probing-based Causal Hallucination Detection through linear probes on disentangled features. Extensive experiments on Gemma-2-9B demonstrate that HalluSAE achieves state-of-the-art hallucination detection performance. 

---
# (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models 

**Authors**: Maksim Zhdanov, Ana Lucic, Max Welling, Jan-Willem van de Meent  

**Link**: [PDF](https://arxiv.org/pdf/2604.16429)  

**Abstract**: We introduce Mosaic, a probabilistic weather forecasting model that addresses two principal sources of spectral degradation in ML-based weather prediction: (1) deterministic training against ensemble means and (2) compressive encoding creating an information bottleneck. Mosaic generates ensemble members through learned functional perturbations and operates on native-resolution grids via block-sparse attention, a hardware-aligned mechanism that captures long-range dependencies at linear cost by sharing keys and values across spatially adjacent queries. At 1.5$°$ resolution with 214M parameters, Mosaic matches or outperforms models trained on 6 times finer data on headline upper-air variables and achieves state-of-the-art results among 1.5$°$ models, producing well-calibrated ensembles whose individual members exhibit near-perfect spectral alignment across all resolved frequencies. A 24-member, 10-day forecast takes under 12 seconds on a single H100 GPU. 

---
# Non-Stationarity in the Embedding Space of Time Series Foundation Models 

**Authors**: Jinmyeong Choi, Brad Shook, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2604.16428)  

**Abstract**: Time series foundation models (TSFMs) are widely used as generic feature extractors, yet the notion of non-stationarity in their embedding spaces remains poorly understood. Recent work often conflates non-stationarity with distribution shift, blurring distinctions fundamental to classical time-series analysis and long-standing methodologies such as statistical process control (SPC). In SPC, non-stationarity signals a process leaving a stable regime - via shifts in mean, variance, or emerging trends - and detecting such departures is central to quality monitoring and change-point analysis. Motivated by this diagnostic tradition, we study how different forms of distributional non-stationarity - mean shifts, variance changes, and linear trends - become linearly accessible in TSFM embedding spaces under controlled conditions. We further examine temporal non-stationarity arising from persistence, which reflects violations of weak stationarity due to long-memory or near-unit-root behavior rather than explicit distributional shifts. By sweeping shift strength and probing multiple TSFMs, we find that embedding-space detectability of non-stationarity degrades smoothly and that different models exhibit distinct, model-specific failure modes. 

---
# Safety, Security, and Cognitive Risks in State-Space Models: A Systematic Threat Analysis with Spectral, Stateful, and Capacity Attacks 

**Authors**: Manoj Parmar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16424)  

**Abstract**: State-Space Models (SSMs) -- structured SSMs (S4, S4D, DSS, S5), selective SSMs (Mamba, Mamba-2), and hybrid architectures (Jamba) -- are deployed in safety-critical long-context applications: genomic analysis, clinical time-series forecasting, and cybersecurity log processing. Their linear-time scaling is compelling, yet the security properties of their compressed-state recurrent architectures remain unstudied.
We present the first systematic treatment of SSM safety, security, and cognitive risks. Seven contributions: (1) Formal threat framework -- SSM Attack Surface (five layers), State Integrity Violation (StIV), Cross-Context Amplification Ratio $\mathcal{X}_\mathcal{S}$, and a Spectral Sensitivity Proposition grounded in the $H_\infty$ norm. (2) Three novel attack classes: spectral adversarial attacks (transfer-function gain exploitation), delayed-trigger stateful backdoors (activate thousands of steps after injection), and state capacity saturation (entropy flooding forces silent forgetting). (3) 14 MITRE ATLAS technique extensions across the full tactic chain. (4) Six-profile attacker taxonomy with kill chains for genomics, clinical, and cybersecurity domains. (5) Four cognitive risk hypotheses grounded in state-compression mechanics. (6) Governance-aligned mitigations mapped to CREST, NIST AI 600-1, and EU AI Act. (7) Empirical evaluation: targeted genomic injection achieves $\mathrm{StIV}=0.519$ vs. $0.086$ random ($6.0\times$, $p<0.001$); PGD state injection achieves $156\times$ output perturbation over random; SSD-structured extraction confirmed at $O(N^2)$ vs. $O(N^3)$ query complexity ($N\times$ speedup). Validation on pretrained checkpoints is detailed in the Appendix. 

---
# Shifting the Gradient: Understanding How Defensive Training Methods Protect Language Model Integrity 

**Authors**: Satchel Grant, Victor Gillioz, Jake Ward, Thomas McGrath  

**Link**: [PDF](https://arxiv.org/pdf/2604.16423)  

**Abstract**: Defensive training methods such as positive preventative steering (PPS) and inoculation prompting (IP) offer surprising results through seemingly similar processes: both add trait-inducing objects to large language models (LLMs) during training, and both defend the LLM against acquiring the trait. The surprising success of these methods comes with the question: how do they work? Are PPS and IP doing the same thing? We provide behavioral and mechanistic comparisons of these two methods using "evilness" as a case-study trait. Our central finding is that PPS and IP achieve their defensive benefits through distinct mechanisms. Behaviorally, we show that neither PPS nor IP operates through a purely associative mechanism; and PPS can both defend against trait acquisition and actively reduce pre-existing expression, whereas IP is ineffective in models that were previously finetuned to express the trait. This behavioral divergence is reflected mechanistically: PPS shifts the activation gradient towards an attenuating direction along the PPS vector axis. When the PPS vector is aligned with a trait-expressing axis, it can reverse the gradient pressure, reducing rather than increasing activation along that axis. In contrast, IP continues to resist a precise mechanistic account. Direct cosine similarity analyses reveal that IP has a characteristically different gradient signature than PPS, and qualitative analyses reveal IP's gradient to be more diffuse. Furthermore, IP reduces the next-token prediction loss on trait-expressing data where PPS need not, consistent with the notion that IP "explains away" the trait-expression in the training data. Taken together, our analyses reveal distinct mechanisms by which each method operates and highlight open questions about IP's mechanistic picture. 

---
# Injecting Structured Biomedical Knowledge into Language Models: Continual Pretraining vs. GraphRAG 

**Authors**: Jaafer Klila, Sondes Bannour Souihi, Rahma Boujelben, Nasredine Semmar, Lamia Hadrich Belguith  

**Link**: [PDF](https://arxiv.org/pdf/2604.16422)  

**Abstract**: The injection of domain-specific knowledge is crucial for adapting language models (LMs) to specialized fields such as biomedicine. While most current approaches rely on unstructured text corpora, this study explores two complementary strategies for leveraging structured knowledge from the UMLS Metathesaurus: (i) Continual pretraining that embeds knowledge into model parameters, and (ii) Graph Retrieval-Augmented Generation (GraphRAG) that consults a knowledge graph at inference time. We first construct a large-scale biomedical knowledge graph from UMLS (3.4 million concepts and 34.2 million relations), stored in Neo4j for efficient querying. We then derive a ~100-million-token textual corpus from this graph to continually pretrain two models: BERTUMLS (from BERT) and BioBERTUMLS (from BioBERT). We evaluate these models on six BLURB (Biomedical Language Understanding and Reasoning Benchmark) datasets spanning five task types and evaluate GraphRAG on the two QA (Question Answering) datasets (PubMedQA, BioASQ). On BLURB tasks, BERTUMLS improves over BERT, with the largest gains on knowledge-intensive QA. Effects on BioBERT are more nuanced, suggesting diminishing returns when the base model already encodes substantial biomedical text knowledge. Finally, augmenting LLaMA 3-8B with our GraphRAG pipeline yields over than 3 points accuracy on PubMedQA and 5 points on BioASQ without any retraining, delivering transparent, multi-hop, and easily updated knowledge access. We release the processed UMLS Neo4j graph to support reproducibility. 

---
# Measuring Representation Robustness in Large Language Models for Geometry 

**Authors**: Vedant Jawandhia, Yash Sinha, Murari Mandal, Ankan Pal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16421)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on mathematical reasoning, yet their robustness to equivalent problem representations remains poorly understood. In geometry, identical problems can be expressed in Euclidean, coordinate, or vector forms, but existing benchmarks report accuracy on fixed formats, implicitly assuming representation invariance and masking failures caused by representational changes alone. We propose GeoRepEval, a representation-aware evaluation framework that measures correctness, invariance, and consistency at the problem level across parallel formulations, combining strict answer matching, bootstrap confidence intervals, paired McNemar tests, representation-flip analyses, and regression controls for surface complexity. We prove that our Invariance@3 metric decomposes accuracy into robust and fragile components and is bounded by the weakest representation. Evaluating eleven LLMs on 158 curated high-school geometry problems (474 instances), we find accuracy gaps of up to 14 percentage points induced solely by representation choice. Vector formulations emerge as a consistent failure point, with Invariance@3 as low as 0.044 even after controlling for length and symbolic complexity. A convert-then-solve prompting intervention improves vector accuracy by up to 52 percentage points for high-capacity models, suggesting that failures reflect representation sensitivity rather than inability; however, low-capacity models show no gains, indicating deeper limitations. These results suggest that current models rely on representation-specific heuristics rather than abstract geometric reasoning. All datasets, prompts, and scripts are released at this https URL. 

---
# Breaking Validity-Induced Boundaries to Expand Algorithm Search Space: A Two-Stage AST-Based Operator for LLM-Driven Automated Heuristic Evolution 

**Authors**: Sun Shengming, Shi Jialong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16420)  

**Abstract**: Large Language Model (LLM) based automated heuristic design (AHD) has shown great potential in discovering efficient heuristics. Most existing LLM-AHD frameworks use semantic evolutionary operators that rely entirely on the LLM's pre-trained knowledge. These one-stage methods strictly require the generated code to be valid during the operation and often rely on a ``thought-code'' representation. We argue that this end-to-end generation fundamentally limits the exploration ability within the algorithm search space.
In this paper, we propose a two-stage, structure-based evolutionary operator for LLM-AHD. In the first stage, our approach directly performs crossover and mutation on the Abstract Syntax Trees (ASTs) of the heuristic code, intentionally generating diverse but often invalid structural variants. In the second stage, the LLM is employed to repair these invalid heuristics into executable, high-quality code. Depending on the underlying framework, either the raw invalid variants or the repaired heuristics are integrated into the population to preserve potential structural patterns. We demonstrate that the proposed operator can significantly enhance the search ability of state-of-the-art LLM-AHD algorithms, such as EoH-S. Experimental results on the Traveling Salesman Problem (TSP) and the Online Bin Packing Problem (OBP) show that our method effectively improves both optimization performance and convergence speed. 

---
# Modeling User Exploration Saturation: When Recommender Systems Should Stop Pushing Novelty 

**Authors**: Enock O. Ayiku, Evelyn Osei, Emebo Onyeka  

**Link**: [PDF](https://arxiv.org/pdf/2604.16419)  

**Abstract**: Fairness-aware recommender systems often mitigate bias by increasing exposure to under-represented or long-tail content, commonly through mechanisms that promote novelty and diversity. In practice, the strength of such interventions is typically controlled using global hyperparameters, fixed regularization weights, heuristic caps, or offline tuning strategies. These approaches implicitly assume that a single level of exploration is appropriate across users, contexts, and stages of interaction. In this work, we study exploration saturation as a user-dependent phenomenon arising from fairness- and novelty-driven recommendation strategies. We define exploration saturation as the point at which further increases in exploration no longer improve user utility and may instead reduce engagement or perceived relevance. Rather than proposing a new fairness-aware algorithm or optimizing a specific objective, we empirically analyze how increasing exploration affects users across varied recommendation models. Through longitudinal experiments using MovieLens-1M and this http URL datasets, our results indicate that fairness-induced exploration exhibits diminishing or non-monotonic returns and varies substantially across users. In particular, users with limited interaction histories tend to reach saturation earlier, suggesting that uniform fairness or novelty pressure can disproportionately disadvantage certain users. These findings reveal a trade-off between fairness and user experience, suggesting that recommendation systems should adapt not only to relevance but also to the amount of fairness-driven exploration applied to individual users. 

---
# What Is Actually Being Annotated? Inter-Prompt Reliability as a Measurement Problem in LLM-Based Social Science Labeling 

**Authors**: Jingyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16413)  

**Abstract**: Large language models (LLMs) are increasingly used for annotation in computational social science, yet their methodological reliability under prompt variation remains unclear. This paper introduces Inter-Prompt Reliability (IPR), a framework for evaluating the stability of LLM outputs across semantically equivalent but linguistically varied prompts. Drawing on Inter-Rater Reliability, IPR is measured by Pairwise Agreement Rate (PAR) and its distribution to capture both consistency and stochasticity in model behavior. We evaluate this framework on two tasks with distinct properties: TREC (interpretative) and Politifact (knowledge-anchored). Results show that LLM annotation exhibits substantial stochastic variation in interpretative tasks, while appearing more stable in knowledge-based tasks. We further show that majority voting across prompts significantly improves reproducibility and reduces variance. These findings suggest that LLM prompt acts as an instrumental measurement while its wording exhibits methodological uncertainty. For future LLM-based CSS studies, we suggest that researchers move beyond single-prompt evaluation toward distributional stability and prompt aggregation within our IPR framework. 

---
# How unique are hallucinated citations offered by generative Artificial Intelligence models? 

**Authors**: Dirk HR Spennemann  

**Link**: [PDF](https://arxiv.org/pdf/2604.16407)  

**Abstract**: This paper investigates how generative AI produces and propagates hallucinated academic references, focusing on the recurring non-existent citation 'Education Governance and Datafication' attributed to Ben Williamson and Nelli Piattoeva. Drawing on 137 accessible source papers identified through Google Scholar and Google searches, the study analyses the structure, recurrence, and onward citation of this phantom reference. It shows that hallucinated citations are not random inventions but patterned recombinations of real authors, journals, dates, and keywords, with duplication occurring in nearly 30% of cases. The paper also reports a structured interrogation of ChatGPT 5-mini about how it generates citations and finds that, absent verification, the model reconstructs plausible references from learned patterns rather than factual recall. Finally, ten AI-generated essays on datafication and school governance were examined: while most references were genuine or partly accurate, 9.2% remained hallucinated, including an exact match to the most common phantom citation. The findings highlight ongoing risks to academic integrity and show that web-enabled AI still does not fully eliminate fabricated references. 

---
# ICAT: Incident-Case-Grounded Adaptive Testing for Physical-Risk Prediction in Embodied World Models 

**Authors**: Zhenglin Lai, Sirui Huang, Yuteng Li, Changxin Huang, Jianqiang Li, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16405)  

**Abstract**: Video-generative world models are increasingly used as neural simulators for embodied planning and policy learning, yet their ability to predict physical risk and severe consequences is rarely this http URL find that these models often downplay or omit key danger cues and severe outcomes for hazardous actions, which can induce unsafe preferences during planning and training on imagined rollouts. We propose ICAT, which grounds testing in real incident reports and safety manuals by building structured risk memories and retrieving/composing them to constrain the generation of risk cases with causal chains and severity labels. Experiments on an ICAT-based benchmark show that mainstream world models frequently miss mechanisms and triggering conditions and miscalibrate severity, falling short of the reliability required for safety-critical embodied deployment. 

---
# GraphRAG-Router: Learning Cost-Efficient Routing over GraphRAGs and LLMs with Reinforcement Learning 

**Authors**: Dongzhe Fan, Chuanhao Ji, Zimu Wang, Tong Chen, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16401)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) has recently emerged as a powerful paradigm for knowledge-intensive question answering, especially for tasks that require structured evidence organization and multi-hop reasoning. However, existing GraphRAG systems are typically built in a one-size-fits-all manner, relying on a fixed retrieval framework and a single, often large and costly, generator LLM for all queries. This static design limits their ability to adapt to the complexity of varying questions and often incurs unnecessary computational cost. To fill in the gap, we propose GraphRAG-Router, a cost-efficient framework that adopts a hierarchical routing strategy to coordinate heterogeneous GraphRAGs and generator LLMs. Specifically, GraphRAG-Router is first warmed up through supervised fine-tuning and then optimized with a two-stage reinforcement learning procedure, whose second stage introduces a curriculum cost-aware reward to encourage difficulty-aware and economical generator allocation. Extensive experiments on six general-domain and multi-hop QA benchmarks show that GraphRAG-Router consistently outperforms state-of-the-art baselines, reducing the overuse of large LLMs by nearly 30% while maintaining strong generalization capability. 

---
# CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference 

**Authors**: Shaoyuan Huang, Xiaokai Wang, Na Yan, Xiaofei Wang, Wenyu Wang, Yansha Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.16400)  

**Abstract**: As Large Language Models (LLMs) are increasingly adopted in edge intelligence to power domain-specific applications and personalized services, the quality and efficiency of the LLM post-training phase-including fine-tuning and inference, have become critical due to constrained resources. Although recent advances in federated parameter-efficient fine-tuning (FL PEFT) and low-latency inference have improved individual task performance, fine-tuning and inference are still handled as isolated workloads, which overlooks their interdependence and results in redundant deployments and delayed improvement in inference quality. To address these limitations, we introduce a new co-execution framework and instantiate it with CoLLM, a system that unifies FL PEFT and inference on shared edge replicas and model parameters. CoLLM addresses key challenges at both replica and cluster levels through: (1) an intra-replica model sharing mechanism that enables real-time model parameter reuse via unmerged inference and shadow adapter strategies; and (2) a two-timescale inter-replica coordination algorithm that adaptively balances fine-tuning and inference workloads to jointly optimize long-term model quality gains and short-term inference efficiency. Extensive evaluation across diverse LLMs and real-world traces show that CoLLM consistently outperforms state-of-the-art LLM systems, achieving up to 3x higher goodput, demonstrating its effectiveness in enabling seamless LLM post-training for edge intelligence. 

---
# IACDM: Interactive Adversarial Convergence Development Methodology -- A Structured Framework for AI-Assisted Software Development 

**Authors**: Jasmine Moreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.16399)  

**Abstract**: The widespread adoption of AI-assisted development tools in 2025 -- and the emergence of vibe coding, a practice of generating complete applications from natural language without verification -- exposed a critical and tool-agnostic failure pattern: experienced developers who used frontier AI models were measurably slower in objective evaluations despite believing they were faster. Concurrently, 10.3% of AI-generated applications in a production showcase contained critical security flaws. This paper argues that these failures share a structural cause -- the verification gap: every large language model (LLM), regardless of interface or capability, operates as a stochastic generator with zero internal semantic verification capability. The tool is irrelevant; the process is determinative. We present IACDM (Interactive Adversarial Convergence Development Methodology), a structured 8-phase framework designed to address the verification gap through external verification agents (VA) operating at discrete gates. Its three pillars are: (1) deep problem discovery via Hierarchical Semantic Analysis before any technical solution; (2) persistent knowledge management across sessions; and (3) systematic adversarial critique through specialized lenses before implementation. The methodology is tool-agnostic by construction, grounded in established software engineering tradition, and applied across more than 20 projects by multiple practitioners in a production R&D environment. Limitations are formalized as testable hypotheses for future empirical validation. 

---
# A Framework for Human-AI Q-Matrix Refinement: A NeuralCDM Evaluation 

**Authors**: Ying Zhang, Ningxi Cheng, Yizhu Gao, Hongmei Li, Lehong Shi, Nicholas Young, Geng Yuan, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.16398)  

**Abstract**: Q-matrices are a cornerstone of theory-driven assessment and learning analytics, making item demands and students' underlying knowledge components and misconceptions explicit and actionable. However, Q-matrices are typically crafted by experts, making them time-consuming to build, prone to subjectivity, and difficult to validate empirically. We propose a framework for human-AI Q-matrix refinement in which large language models (LLMs) generate candidate Q-matrices using structured, misconception-aware prompting, and NeuralCDM provides an empirical evaluation layer to compare candidates based on how well they explain student response data. We apply the framework to a thermodynamics assessment dataset and benchmark locally deployed LLMs against cloud-served models. Results show that iteratively refined LLM-generated Q-matrices can exceed expert-baseline model fit (AUC 0.780 vs. 0.717), and that locally deployed models achieve comparable performance to cloud APIs, supporting privacy-preserving deployment. 

---
# Instructor-Created Custom GPTs as Pedagogical Partners Fostering Immersion in Online Higher Education: Two Case Studies 

**Authors**: Dennis Beck, Leonel Morgado  

**Link**: [PDF](https://arxiv.org/pdf/2604.16397)  

**Abstract**: As online higher education expands, sustaining student engagement remains a critical challenge. This paper approaches immersive learning by investigating how custom GPTs foster immersion (as a state of deep mental involvement) for students and instructors. While large language models (LLMs) offer potential for enhancing feedback, little research has examined instructor-created custom GPTs designed to align with specific pedagogical goals. This paper addresses this gap, employing the Immersive Learning Cube framework, which conceptualizes immersion through three dimensions: system (envelopment by the environment), narrative (meaningful context), and agency (commitment to meaning-making). Through a qualitative analysis of two distinct case studies, an accelerated graduate grant writing course in the US and an undergraduate software engineering course in Portugal, we analyze course-embedded artifacts to map how custom GPTs influence these immersion dimensions. In the grant writing course, the custom GPT functioned as a feedback partner, fostering system immersion through its immediacy, narrative immersion by reinforcing the proposal's evolving story, and agency immersion by empowering students to negotiate feedback and take ownership of revisions. In the software engineering course, a diegetically-framed custom GPT acted as a metacognitive tutor, enhancing system immersion via its permanent availability, narrative immersion through its role-play function and agency immersion by scaffolding students' self- and co-regulated learning. Our findings demonstrate that thoughtfully integrated custom GPTs can act as powerful pedagogical partners that leverage all three dimensions of immersion. Rather than replacing human instructors, they can amplify immediacy, coherence, and learner autonomy, creating more engaging and immersive online learning environments. 

---
# Stream2LLM: Overlap Context Streaming and Prefill for Reduced TTFT 

**Authors**: Rajveer Bachkaniwala, Chengqi Luo, Richard So, Divya Mahajan, Kexin Rong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16395)  

**Abstract**: Context retrieval systems for LLM inference face a critical challenge: high retrieval latency creates a fundamental tension between waiting for complete context (poor time-to-first-token) and proceeding without it (reduced quality). Recent work mitigates this via streaming--overlapping retrieval with inference--but prior systems focus on single-request settings and overlook challenges in multi-tenant deployments where concurrent requests contend for GPU memory and scheduling must adapt to dynamic context arrivals.
We present STREAM2LLM, a system that extends vLLM to support streaming prompts with adaptive scheduling and preemption for two distinct retrieval patterns: append-mode (progressive context accumulation) and update-mode (iterative refinement with cache invalidation). STREAM2LLM decouples scheduling decisions from resource acquisition, enabling flexible preemption strategies guided by hardware-specific cost models, and uses cache invalidation based on longest common prefix matching to minimize redundant computation when prompts change dynamically. To evaluate STREAM2LLM, we collect and characterize two large-scale, real-world streaming workloads based on web crawling and approximate nearest neighbor search. Our evaluation demonstrates that streaming architecture delivers up to 11x TTFT improvements, with cost-aware scheduling providing critical benefits under memory pressure, while maintaining throughput parity with non-streaming baselines. 

---
# A Reference Architecture for Agentic Hybrid Retrieval in Dataset Search 

**Authors**: Riccardo Terrenzi, Phongsakon Mark Konrad, Tim Lukas Adam, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.16394)  

**Abstract**: Ad hoc dataset search requires matching underspecified natural-language queries against sparse, heterogeneous metadata records, a task where typical lexical or dense retrieval alone falls short. We reposition dataset search as a software-architecture problem and propose a bounded, auditable reference architecture for agentic hybrid retrieval that combines BM25 lexical search with dense-embedding retrieval via reciprocal rank fusion (RRF), orchestrated by a large language model (LLM) agent that repeatedly plans queries, evaluates the sufficiency of results, and reranks candidates. To reduce the vocabulary mismatch between user intent and provider-authored metadata, we introduce an offline metadata augmentation step in which an LLM generates pseudo-queries for each dataset record, augmenting both retrieval indexes before query time. Two architectural styles are examined: a single ReAct agent and a multi-agent horizontal architecture with Feedback Control. Their quality-attribute tradeoffs are analyzed with respect to modifiability, observability, performance, and governance. An evaluation framework comprising seven system variants is defined to isolate the contribution of each architectural decision. The architecture is presented as an extensible reference design for the software architecture community, incorporating explicit governance tactics to bound and audit nondeterministic LLM components. 

---
# RoMathExam: A Longitudinal Dataset of Romanian Math Exams (1895-2025) with a Seven-Decade Core (1957-2025) 

**Authors**: Luca-Ncolae Cuclea, Sabin-Codrut Badea, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2604.16392)  

**Abstract**: AI in Education research increasingly relies on authentic, curriculum-grounded assessment data, yet large, well-structured exam corpora remain scarce for many languages and educational systems. We introduce RoMathExam, a longitudinal dataset of Romanian high-school mathematics exams spanning 1895-2025, with a robust standardized core for 1957-2025. The dataset contains 10,592 mathematics problems organized into 600+ complete exam sets across multiple tracks (M1-M4), covering both official national examination sessions and ministry-published training variants. Beyond high-fidelity digitization and a unified JSON schema with traceable provenance, RoMathExam is enriched with curriculum-aligned topic tags and dense text embeddings, enabling variant detection, deduplication, and similarity-based retrieval. To overcome the lack of historical psychometric data, we propose and validate a solution complexity metric as a scalable intrinsic proxy for difficulty. Our evaluation across three frontier reasoning models (GPT-5-mini, DeepSeek-R1, and Qwen3-235B-Thinking) reveals high cross-model synchronization (r > 0.72), confirming the metric's ability to isolate intrinsic mathematical depth from stochastic generation noise. We demonstrate the dataset's utility through a longitudinal analysis that quantifies a "regime shift" from volatile historical formats to a standardized, algebra-dominant modern curriculum. RoMathExam provides a foundation for reproducible research in difficulty modeling, curriculum analytics, and LLM evaluation in low-resource linguistic contexts. 

---
# Large language models for post-publication research evaluation: Evidence from expert recommendations and citation indicators 

**Authors**: Mengjia Wu, Yi Zhang, Robin Haunschild, Lutz Bornmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.16387)  

**Abstract**: Assessing the quality of scientific research is essential for scholarly communication, yet widely used approaches face limitations in scalability, subjectivity, and time delay. Recent advances in large language models (LLMs) offer new opportunities for automated research evaluation based on textual content. This study examines whether LLMs can support post-publication peer review tasks by benchmarking their outputs against expert judgments and citation-based indicators. Two evaluation tasks are constructed using articles from the H1 Connect platform: identifying high-quality articles and performing finer-grained evaluation including article rating, merit classification, and expert style commenting. Multiple model families, including BERT models, general-purpose LLMs, and reasoning oriented LLMs, are evaluated under multiple learning strategies. Results show that LLMs perform well in coarse grained evaluation tasks, achieving accuracy above 0.8 in identifying highly recommended articles. However, performance decreases substantially in fine-grained rating tasks. Few-shot prompting improves performance over zero-shot settings, while supervised fine-tuning produces the strongest and most balanced results. Retrieval augmented prompting improves classification accuracy in some cases but does not consistently strengthen alignment with citation indicators. The overall correlations between model outputs and citation indicators remain positive but moderate. 

---
# DAOnt: A Formal Ontology for EU Data Act Compliance 

**Authors**: Sheyla Leyva-Sánchez, Fabian Linde, Meem Arafat Manab, María Poveda-Villalón, Víctor Rodríguez-Doncel  

**Link**: [PDF](https://arxiv.org/pdf/2604.16386)  

**Abstract**: The EU Data Act establishes comprehensive rules governing data access and sharing across business-to-consumer (B2C), business-to-business (B2B), and business-to-government (B2G) contexts. This paper presents a comprehensive ontology for the EU Data Act, enabling reasoning over data sharing agreements through machine-readable representations. The DAOnt ontology reuses elements from three established ontologies, LKIF-Core, ODRL, and DPV, to capture the normative structure of the Data Act.
The ontology captures the main concepts and relationships in the Regulation, and it also operationalises three articles to facilitate compliance checking: Article 4(1) (B2C user access rights), Article 8(6) (B2B trade secret exceptions) and Article 19(2)(a) (B2G competitive use prohibitions).
The ontology supports compliance checking through SPARQL queries that return obligations, permissions, and prohibitions, allowing organisations to verify whether data-sharing agreements meet the requirements of the EU Data Act and to assess conditions such as FRAND obligations. By representing key legal concepts in RDF, our work helps bridge the gap between the legal provisions of the Data Act and their computational interpretation. The complete ontology, along with example instances and queries, is available online. 

---
# StressWeb: A Diagnostic Benchmark for Web Agent Robustness under Realistic Interaction Variability 

**Authors**: Haoyue Bai, Dong Wang, Long Chen, Bingguang Hao, Pengyang Shao, Yonghui Yang, Yicheng He, Chenyi Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16385)  

**Abstract**: Large language model-based web agents have demonstrated strong performance on realistic web interaction tasks. However, existing evaluations are predominantly conducted under relatively stable and well-behaved interaction conditions, which may overestimate agent robustness. High task success in such idealized settings does not necessarily reflect performance under realistic web interaction. To address this limitation, we introduce a diagnostic stress-testing benchmark for web agents. We first construct realistic and controllable web environments that provide clean and stable interaction workflows as reference baselines. We then introduce structured and controlled perturbations that emulate interaction variability, including shifting layouts, altered interaction semantics, and execution disruptions. By comparing agent behavior between clean and perturbed settings, our framework enables systematic diagnosis of robustness under what-if interaction scenarios. Through extensive evaluation of state-of-the-art multimodal web agents, we show that stress-based evaluation exposes failure modes and substantial robustness gaps that remain hidden under clean benchmark conditions. 

---
# Same Verdict, Different Reasons: LLM-as-a-Judge and Clinician Disagreement on Medical Chatbot Completeness 

**Authors**: Alexandra DeLucia, Heyuan Huang, Sonal Joshi, Mahsa Yarmohammadi, Ahmed Hassoon, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2604.16383)  

**Abstract**: LLM-as-a-Judge frameworks are increasingly trusted to automate evaluation in place of human experts, yet their reliability in high-stakes medical contexts remains unproven. We stress-test this assumption for detecting incomplete patient-facing medical responses, evaluating three rubric granularities (General-Likert, Analytical-Rubric, Dynamic-Checklist) and three backbone models across two clinician-annotated datasets, including HealthBench, the largest publicly available benchmark for medical response evaluation. LLM Judges discriminate complete from incomplete responses at and slightly above near chance (AUC $0.49$--$0.66$); at the threshold required to recall $90\%$ of incomplete responses, clinicians must still review the vast majority of the dataset, offering no triage utility. Even when model and clinician verdicts agree, they rarely cite the same explanation; and when they diverge, false positives stem from over-flagging non-essential gaps while false negatives reflect outright detection failures. These results reveal that LLM Judges and clinicians apply fundamentally different completeness standards; a finding that undermines their use as autonomous evaluators or triage filters in clinical settings. 

---
# CFMS: Towards Explainable and Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark 

**Authors**: Junzhao Zhang, Hsiu-Yuan Huang, Chenming Tang, Yutong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16372)  

**Abstract**: Multimodal sarcasm detection has recently garnered significant attention. However, existing benchmarks suffer from coarse-grained annotations and limited cultural coverage, which hinder research into fine-grained semantic understanding. To address this, we construct CFMS, the first fine-grained multimodal sarcasm dataset tailored for Chinese social media. It comprises 2,796 high-quality image-text pairs and provides a triple-level annotation framework: sarcasm identification, target recognition, and explanation generation. We find that the fine-grained explanation annotations effectively guide AI in generating images with explicit sarcastic intent. Furthermore, we curate a high-consistency parallel Chinese-English metaphor subset (200 entries each), revealing significant limitations of current models in metaphoric reasoning. To overcome the constraints of traditional retrieval methods, we propose a Reinforcement Learning-augmented In-Context Learning strategy (PGDS) to dynamically optimize exemplar selection. Extensive experiments demonstrate that CFMS provides a solid foundation for building reliable multimodal sarcasm understanding systems, and the PGDS method significantly outperforms existing baselines on key tasks. Our data and code are available at this https URL. 

---
# Brain-CLIPLM: Decoding Compressed Semantic Representations in EEG for Language Reconstruction 

**Authors**: Xiaoli Yang, Huiyuan Tian, Yurui Li, Jianyu Zhang, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16370)  

**Abstract**: Decoding natural language from non-invasive electroencephalography (EEG) remains fundamentally limited by low signal-to-noise ratio and restricted information bandwidth. This raises a fundamental question regarding whether sentence-level linguistic structure can be reliably recovered from such signals. In this work, we suggest that this assumption may not hold under realistic information constraints, and instead propose a semantic compression hypothesis in which EEG signals encode a compressed set of semantic anchors rather than full linguistic structure. Under our new perspective, direct sentence reconstruction becomes an overparameterized objective relative to the intrinsic information capacity of EEG. To address this mismatch, we introduce Brain-CLIPLM, a two-stage framework that decomposes EEG-to-text decoding into semantic anchor extraction via contrastive learning and sentence reconstruction using a retrieval-grounded large language model (LLM) with Chain-of-Thought (CoT) reasoning, following a granularity matching principle that aligns decoding complexity with neural information capacity. Evaluated on the Zurich Cognitive Language Processing Corpus, Brain-CLIPLM achieves 67.55\% top-5 and 85.00\% top-25 sentence retrieval accuracy, significantly outperforming direct decoding baseline, while cross-subject evaluation confirms robust generalization. Control analyses, including permutation testing, further demonstrate that EEG-derived representations carry sentence-specific information beyond language model priors. These results suggest that EEG-to-text decoding is better framed as recovering compressed semantic content rather than reconstructing full sentences, providing a biologically grounded and data-efficient pathway for non-invasive brain-computer interfaces. 

---
# Why AI Readiness Is an Organizational Learning Problem, Not a Technology Purchase 

**Authors**: Jeanne McClure, Gregg Gerdau  

**Link**: [PDF](https://arxiv.org/pdf/2604.16369)  

**Abstract**: Global corporate AI investment reached $252.3 billion in 2024, yet only 6% of firms report significant earnings impact. This article argues that AI project failure is fundamentally an organizational learning problem rather than a technology deficit. Drawing on a systematic synthesis of 19 large-scale industry and academic sources, including surveys of nearly 10,000 organizational leaders, we identify two categories of failure: organizational (culture, leadership alignment, governance, and human-AI learning deficits) and technical (semantic bottlenecks and output management challenges). We introduce the Siloed-Integrated-Orchestrated (SIO) progression model, which maps enterprise AI capability across five pillars -- Culture & Leadership, Human Capital & Operations, Data Architecture, Systems Infrastructure, and Governance & Regulatory Compliance -- and provides prescriptive guidance for advancing between stages. The implications challenge organizations to reframe AI investment as capability development rather than technology procurement. 

---
# Talk, Walk, and Market Response: Multimodal Measurement of AI Washing and Its Capital Market Consequences in China 

**Authors**: Wen Zhanjie, Guo Jingqiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16367)  

**Abstract**: As artificial intelligence and generative large language models drive industrial upgrading, capital markets increasingly focus on AI-themed listed firms. Information asymmetry and technological opacity lower the cost of exaggerating AI capabilities relative to genuine R&D, spurring widespread AI Washing. Using China's A-share market from 2018Q1 to 2025Q2, we advance literature in measurement and mechanism testing. We construct a multimodal AI Washing Risk Score (AWRS) via Qwen-VL to assess text-image consistency in annual reports and roadshows, and a Material Real-Investment Matching Index (MRMI) from patent quality, AI intangible asset capitalization, and technical personnel compensation using PCA. Four findings emerge: (1) AWRS lacks predictive power for future MRMI, with a wider rhetoric-action gap among financially constrained firms; (2) substantive AI investment boosts high-quality patents, while empty rhetoric crowds out industry innovation; (3) long-horizon institutional investors detect AI Washing through site visits and reduce holdings; (4) such divestment triggers analyst downgrades, retail selling, and sharp valuation corrections within 180 days. Results are robust to IV-2SLS and staggered DID using the ChatGPT shock. This study enhances disclosure and pricing-efficiency research and supports RegTech for curbing thematic speculation. 

---
# Clinical Note Bloat Reduction for Efficient LLM Use 

**Authors**: Jordan L. Cahoon, Chloe Stanwyck, Asad Aali, Rachel Madding, Emma Sun, Yixing Jiang, Renumathy Dhanasekaran, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.16364)  

**Abstract**: Health systems are rapidly deploying large language models (LLMs) that use clinical notes for clinical decision support applications. However, modern documentation practices rely heavily on templates, copy--paste shortcuts, and auto-populated fields, producing extensive duplicated text (``note bloat'') that dilutes clinically meaningful signal and substantially increases the computational cost of LLM use. We introduce TRACE, a scalable preprocessing pipeline that removes note bloat by leveraging EHR attribution metadata to identify templated and copied content and applying frequency-based deduplication when metadata are unavailable. We evaluated TRACE across four real--world clinical cohorts spanning liver transplantation, obstetrics, and inpatient care (5.3 million notes) using blinded physician review and downstream modeling tasks. TRACE removed 47.3% of chart text while preserving performance for information extraction and clinical outcome prediction. At a large academic medical center, this reduction corresponds to an estimated $9.5 million annual decrease in LLM inference costs assuming one query per encounter. These findings show how underutilized EHR metadata can enable more scalable and cost-efficient deployment of LLM-based clinical systems. 

---
# CSF: Black-box Fingerprinting via Compositional Semantics for Text-to-Image Models 

**Authors**: Junhoo Lee, Mijin Koo, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2604.16363)  

**Abstract**: Text-to-image models are commercially valuable assets often distributed under restrictive licenses, but such licenses are enforceable only when violations can be detected. Existing methods require pre-deployment watermarking or internal model access, which are unavailable in commercial API deployments. We present Compositional Semantic Fingerprinting (CSF), the first black-box method for attributing fine-tuned text-to-image models to protected lineages using only query access. CSF treats models as semantic category generators and probes them with compositional underspecified prompts that remain rare under fine-tuning. This gives IP owners an asymmetric advantage: new prompt compositions can be generated after deployment, while attackers must anticipate and suppress a much broader space of fingerprints. Across 6 model families (FLUX, Kandinsky, SD1.5/2.1/3.0/XL) and 13 fine-tuned variants, our Bayesian attribution framework enables controlled-risk lineage decisions, with all variants satisfying the dominance criterion. 

---
# SetFlow: Generating Structured Sets of Representations for Multiple Instance Learning 

**Authors**: Nikola Jovišić, Milica Škipina, Vanja Švenda  

**Link**: [PDF](https://arxiv.org/pdf/2604.16362)  

**Abstract**: Data scarcity and weak supervision continue to limit the performance of machine learning models in many real-world applications, such as mammography, where Multiple Instance Learning (MIL) often offers the best formulation. While recent foundation models provide strong semantic representations out of the box, effective augmentation of such representations of MIL data remains limited, as existing methods operate at the instance level and fail to capture intra-bag dependencies. In this work, we introduce SetFlow, a generative architecture that models entire MIL bags (i.e., sets) directly in the representation space. Our approach leverages the flow matching paradigm combined with a Set Transformer-inspired design, enabling it to handle permutation-invariant inputs while capturing interactions between instances within each bag. The model is conditioned on both class labels and input scale, allowing it to generate coherent and semantically consistent sets of representations. We evaluate SetFlow on a large-scale mammography benchmark using a state-of-the-art MIL-PF classification pipeline. The generated samples are shown to closely match the original data distribution and even improve downstream performance when used for augmentation. Furthermore, training on synthetic data alone shows competitive results, demonstrating the effectiveness of representation-space generative modeling for data-scarce and privacy-sensitive tasks. 

---
# Mapping Recent Shifts in Digital Art via Conference Discourse: AI, XR, the Metaverse, and Blockchain/NFTs (2021-2025) 

**Authors**: Vasileios Komianos, Emmanuel Rovithis, Athanasios Tsipis  

**Link**: [PDF](https://arxiv.org/pdf/2604.16360)  

**Abstract**: This paper presents an analysis of five years (2021 - 2025) of conference discourse across six digital art conferences, aiming to trace thematic shifts associated with the rapid development of emerging technologies, namely artificial intelligence (AI), immersive technologies (including XR and the metaverse), and blockchain technologies and non-fungible tokens (NFTs). The results indicate a marked increase in AI-related contributions, while immersive technologies maintain a relatively stable share of the discourse, and blockchain- and NFT-based works remain marginal. Overall, whereas immersive technologies and blockchain-related topics exhibit relative stability, AI shows a significant rise after 2022, emerging as a dominant theme within digital art conference discourse. 

---
# AgriIR: A Scalable Framework for Domain-Specific Knowledge Retrieval 

**Authors**: Shuvam Banerji Seal, Aheli Poddar, Alok Mishra, Dwaipayan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.16353)  

**Abstract**: This paper introduces AgriIR, a configurable retrieval augmented generation (RAG) framework designed to deliver grounded, domain-specific answers while maintaining flexibility and low computational cost. Instead of relying on large, monolithic models, AgriIR decomposes the information access process into declarative modular stages -- query refinement, sub-query planning, retrieval, synthesis, and evaluation. This design allows practitioners to adapt the framework to new knowledge verticals without modifying the architecture. Our reference implementation targets Indian agricultural information access, integrating 1B-parameter language models with adaptive retrievers and domain-aware agent catalogues. The system enforces deterministic citation, integrates telemetry for transparency, and includes automated deployment assets to ensure auditable, reproducible operation. By emphasizing architectural design and modular control, AgriIR demonstrates that well-engineered pipelines can achieve domain-accurate, trustworthy retrieval even under constrained resources. We argue that this approach exemplifies ``AI for Agriculture'' by promoting accessibility, sustainability, and accountability in retrieval-augmented generation systems. 

---
# Training for Compositional Sensitivity Reduces Dense Retrieval Generalization 

**Authors**: Radoslav Ralev, Aditeya Baral, Iliya Zhechev, Jen Agarwal, Srijith Rajamohan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16351)  

**Abstract**: Dense retrieval compresses texts into single embeddings ranked by cosine similarity. While efficient for recall, this interface is brittle for identity-level matching: minimal compositional edits (negation, role swaps) flip meaning yet retain high similarity. Motivated by geometric results for unit-sphere cosine spaces (Kang et al., 2025), we test this retrieval-composition tension in text-only retrieval. Across four dual-encoder backbones, adding structure-targeted negatives consistently reduces zero-shot NanoBEIR retrieval (8-9% mean nDCG@10 drop on small backbones; up to 40% on medium ones), while only partially improving pooled-space separation. Treating pooled cosine as a recall interface, we then benchmark verifiers scoring token--token cosine maps. MaxSim (late interaction) excels at reranking but fails to reject structural near-misses, whereas a small Transformer over similarity maps reliably separates near-misses under end-to-end training. 

---
# LiteSemRAG: Lightweight LLM-Free Semantic-Aware Graph Retrieval for Robust RAG 

**Authors**: Xiao Yue, Guangzhi Qu, Lige Gan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16350)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) has shown great potential for improving multi-level reasoning and structured evidence aggregation. However, existing graph-based RAG frameworks heavily rely on exploiting large language models (LLMs) during indexing and querying, leading to high token consumption, computational cost and latency overhead. In this paper, we propose LiteSemRAG, a lightweight, fully LLM-free, semantic-aware graph retrieval framework. LiteSemRAG constructs a heterogeneous semantic graph by exploiting contextual token-level embeddings, explicitly separating surface lexical representations from context-dependent semantic meanings. To robustly model polysemy, we introduce a dynamic semantic node construction mechanism with chunk-level context aggregation and adaptive anomaly handling. At query stage, LiteSemRAG performs a two-step semantic-aware retrieval process that integrates co-occurrence graph weighting with an isolated semantic recovery mechanism, enabling balanced structural reasoning and semantic coverage. We evaluate LiteSemRAG on three benchmark datasets and experimental results show that LiteSemRAG achieves the best mean reciprocal rank (MRR@10) across all datasets and competitive or superior recall rate (Recall@10) compared to state-of-the-art LLM-based graph RAG systems. Meanwhile, LiteSemRAG consumes zero LLM tokens and achieves substantial efficiency improvements in both indexing and querying due to the elimination of LLM usage. These results demonstrate the effectiveness of LiteSemRAG, indicating that a strong semantic-aware graph retrieval framework can be achieved without relying on LLM-based approaches. 

---
# Benchmarking Real-Time Question Answering via Executable Code Workflows 

**Authors**: Wenjie Zhou, Yuan Gao, Xin Zhou, Hao Fu, Zhongjian Miao, Wei Chen, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16349)  

**Abstract**: Retrieving real-time information is a fundamental capability for search-integrated agents in real-world applications. However, existing benchmarks are predominantly static and therefore fail to capture the temporal dynamics of information and the continuously evolving nature of real-world knowledge. To address this limitation, we propose RT-QA, a dynamic evaluation framework that leverages executable code workflows to retrieve up-to-date answers at evaluation time. Specifically, we construct an agent-driven pipeline that autonomously generates code for web crawling and DOM-based answer extraction to produce real-time ground truth. To ensure robust evaluation over time, the pipeline further incorporates a self-repair mechanism to adapt to changes in web page structures. RT-QA spans 12 domains (e.g., Finance, Sports) with 320 Chinese questions categorized into three difficulty levels. Extensive evaluations of state-of-the-art models (e.g., GPT-5.2, GLM-4.7) reveal significant limitations in real-time adaptability: even the best models achieve only 46% accuracy. Our analysis highlights two primary failure modes: (1) Lazy Retrieval, where agents rely on search snippets instead of deeply scanning specific websites for information (20% of failures); and (2) Temporal Confusion, a cognitive error where agents retrieve a historical date (e.g., an event in 2024) and fail to re-anchor to the current time (2026) for subsequent reasoning. These findings suggest that future agents require not just better retrieval strategies, but robust temporal state management. 

---
# Lean Atlas: An Integrated Proof Environment for Scalable Human-AI Collaborative Formalization 

**Authors**: Banri Yanahama, Akiyoshi Sannai  

**Link**: [PDF](https://arxiv.org/pdf/2604.16347)  

**Abstract**: AI-driven autoformalization of mathematics is advancing rapidly. However, the type checker of a proof assistant guarantees only the logical correctness of proofs; it does not verify whether propositions and definitions faithfully capture their intended mathematical content. Consequently, AI-generated formal proofs can exhibit semantic hallucination-passing the type checker yet failing to express the intended mathematics. We propose a human-in-the-loop approach in which human scientists and AI collaboratively produce formal proofs, with humans responsible for the semantic verification of propositions and definitions. To realize this approach, we develop Lean Atlas, a Lean 4 tool that visualizes the dependency graph of a Lean 4 project as an interactive web viewer, enabling human scientists to grasp the overall structure of a formalization efficiently. Its core feature, Lean Compass, is an algorithm that, given a selected theorem set, automatically extracts the project-specific nodes whose semantic correctness can affect those target statements, thereby reducing the candidate set for semantic review in large-scale formalizations. We further define *aligned Lean code* as formalization code that has undergone human semantic verification, and propose it as a quality standard for AI-generated formalizations. We evaluate the tool on six Lean 4 formalization projects with different structural characteristics; proof-heavy projects (PrimeNumberTheoremAnd, Carleson, Brownian Motion) achieved 94-99% average node reduction, a 6-theorem milestone subset of FLT achieved 59.8%, mixed PhysLib 69.0%, and definition-heavy XMSS 27.3%. Lean Atlas is available as open-source software at this https URL . 

---
# Bridging the Experimental Last Mile: Digitizing Laboratory Know-How for Safe AI-Assisted Support 

**Authors**: Akira Miura, Yuki Sasahara, Momoka Demura, Yuji Masubuchi, Tetsuya Asai, Chikahiko Mitsui  

**Link**: [PDF](https://arxiv.org/pdf/2604.16345)  

**Abstract**: Advances in Materials Informatics have accelerated the development of Self-Driving Laboratories (SDLs), yet human-led experiments remain standard in many educational and exploratory research settings. In such environments, practical know-how, including operational details and site-specific rules, is essential for safe and reliable laboratory work. In this proof-of-concept study, we developed a human-in-the-loop AI assistant that combines first-person experimental video, multimodal AI, and retrieval-augmented generation (RAG). Using powder X-ray diffraction experiments and student-recorded video data as inputs, the system extracts site-specific laboratory knowledge from recorded procedures, including physical techniques and audible confirmation that conventional manuals could omit. It then provides grounded responses based on the resulting manual. To reduce the risk of unsupported outputs, the system employs a two-layer safety design: source restriction through RAG and strict system-prompt constraints. Instructor-based evaluation showed alignment with expected guidance for questions covered by the manual. For out-of-scope queries, the system appropriately refused to answer, indicating a reduced risk of hallucination. Expert evaluation further indicated that the generated advisory reports were useful and safe (utility: 3.25/4.00; safety: 4.00/4.00). These results suggest a framework in which AI supports laboratory practice under explicit human supervision rather than replacing human judgment. 

---
# Elder-Sim: A Psychometrically Validated Platform for Personality-Stable Elderly Digital Twins 

**Authors**: Jiaqing Wang, Zhongfang Yang, Xingyuan Zhu, Zong'an Huang, Hao Wang, Li Tian, Ying Cao, Xiaomin Qu, Xiang Qi, Bei Wu, Zheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16343)  

**Abstract**: Background: LLMs enable patient-facing conversational agents, creating a pathway toward digital twins that capture older adults' lived experiences and behavioral responses across time. A central barrier is personality drift -- inconsistent trait expression across repeated interactions -- which undermines reliability of generated trajectories and intervention-response simulation in geriatric care.
Objective: To develop ELDER-SIM, a multi-role elderly-care conversational platform for building personality-stable digital twin agents, and to propose a psychometric validation framework for quantifying personality consistency in LLM-based agents.
Methods: ELDER-SIM was implemented via n8n workflow orchestration with local LLM inference (Ollama/vLLM), integrating (1) Big Five (OCEAN) trait specifications, (2) a Cognitive Conceptualization Diagram (CCD) grounded in Beck's CBT framework, and (3) a MySQL-based long-term memory module. Ablation studies across four conditions -- Baseline, +Memory, +CCD, and +LoRA (fine-tuned on 19,717 instruction pairs from CHARLS) -- were evaluated via Cronbach's $\alpha$, ICC, and role discrimination accuracy.
Results: Reliability was acceptable to excellent across conditions (Cronbach's $\alpha$: 0.70--0.94; ICC: 0.85--0.96). Role discrimination improved from 83.3% (Baseline) to 88.9% (+Memory), 94.4% (+CCD), and 97.2% (+LoRA). CCD produced the largest consistency gain (mean $\alpha$ 0.702$\to$0.892), while LoRA achieved the highest overall consistency ($\alpha$ 0.940; ICC 0.958).
Conclusions: ELDER-SIM provides a psychometrically validated approach for constructing personality-consistent elderly digital twin agents. Structured cognitive modeling and domain adaptation reduce personality drift, supporting reliable longitudinal simulation for elderly mental health care and reproducible in silico evaluation before clinical deployment. 

---
# HR-Agents: Using Multiple LLM-based Agents to Improve Q&A about Brazilian Labor Legislation 

**Authors**: Abriel K. Moraes, Gabriel S. M. Dias, Vitor L. Fabris, Lucas D. Gessoni, Leonardo R. do Nascimento, Charles S. Oliveira, Vitor G. C. B. de Farias, Fabiana C. Q. de O. Marucci, Matheus H. R. Vicente, Gabriel U. Talasso, Erik Soares, Amparo Munoz, Sildolfo Gomes, Maria L. A. de S. Cruvinel, Leonardo T. dos Santos, Renata De Paris, Wandemberg Gibaut  

**Link**: [PDF](https://arxiv.org/pdf/2604.16337)  

**Abstract**: The Consolidation of Labor Laws (CLT) serves as the primary legal framework governing labor relations in Brazil, ensuring essential protections for workers. However, its complexity creates challenges for Human Resources (HR) professionals in navigating regulations and ensuring compliance. Traditional methods for addressing labor law inquiries often lead to inefficiencies, delays, and inconsistencies. To enhance the accuracy and efficiency of legal question-answering (Q&A), a multi-agent system powered by Large Language Models (LLMs) is introduced. This approach employs specialized agents to address distinct aspects of employment law while integrating Retrieval-Augmented Generation (RAG) to enhance contextual relevance. Implemented using CrewAI, the system enables cooperative agent interactions, ensuring response validation and reducing misinformation. The effectiveness of this framework is evaluated through a comparison with a baseline RAG pipeline utilizing a single LLM, using automated metrics such as BLEU, LLM-as-judge evaluations, and expert human assessments. Results indicate that the multi-agent approach improves response coherence and correctness, providing a more reliable and efficient solution for HR professionals. This study contributes to AI-driven legal assistance by demonstrating the potential of multi-agent LLM architectures in improving labor law compliance and streamlining HR operations. 

---
# Distributed Human Identity: AI-Enabled Multi-Existence Through Cognitive Replication and Robotic Embodiments 

**Authors**: A S M Touhidul Islam, John Tookey  

**Link**: [PDF](https://arxiv.org/pdf/2604.16336)  

**Abstract**: Human presence has traditionally been constrained by the limits of physical embodiment, allowing individuals to exist in only one place at a time. This article introduces Multi-Existence Identity (MEI)- a socio-technical framework that replicates cognitive, behavioral, and emotional attributes into AI-enabled embodiments capable of acting across digital and physical contexts in parallel. MEI advances beyond digital twins, telepresence, and multipresence avatars by embedding cognitive fidelity, affective resonance, and contextual responsiveness into distributed agents that function not only for, but as, the original individual. The framework integrates personality modeling, cognitive simulation, and a synchronization layer to maintain identity coherence across three embodiment channels: digital avatars, robotic embodiments, and agentic software agents. Differentiating itself from simulated assistants, MEI positions replicated identity as a dynamic and culturally situated extension of selfhood, foregrounding tacit engagement and relational authenticity. Application domains span professional work, education, healthcare, governance, family life, and media, offering transformative potential for productivity, caregiving, leadership, and creativity. Yet these opportunities also surface profound challenges concerning authenticity, consent, legal accountability, privacy, and the psychological meaning of presence. The article proposes a phased empirical roadmap to operationalize MEI through personality modeling, synchronization testing, robotic embodiment trials, and ethical stress-testing. By conceptualizing MEI as both a technological and cultural construct, the study reframes debates on identity and presence in digitally augmented societies, highlighting opportunities for human-AI integration while underscoring the need for inclusive ethical governance. 

---
# Beyond Verifiable Rewards: Rubric-Based GRM for Reinforced Fine-Tuning SWE Agents 

**Authors**: Jiawei Huang, Qingping Yang, Renjie Zheng, Jiaze Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16335)  

**Abstract**: Despite recent progress in Large Language Model (LLM) Agents for Software Engineering (SWE) tasks, end-to-end fine-tuning typically relies on verifiable terminal rewards such as whether all unit tests pass. While these binary signals reflect whether the final solution is correct, they provide little guidance for shaping intermediate behaviors during multi-step interactions, thereby limiting improvements in the overall quality of the resolution process. To address this, we introduce a rubric-based Generative Reward Model (GRM) that provides richer learning signals. The GRM is equipped with human-designed rubrics that indicate criteria for encouraging or discouraging specific behavioral patterns, and we leverage this feedback for high-quality training data collection via trajectory filtration. When used for Reinforced Fine-Tuning (RFT) on SWE Tasks, our approach outperforms terminal-score-only rejection sampling: it more effectively suppresses undesirable patterns while promoting beneficial ones, as confirmed by case analyses, and it ultimately improves final test accuracy. 

---
# Preventing overfitting in deep learning using differential privacy 

**Authors**: Alizishaan Anwar Hussein Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2604.16334)  

**Abstract**: The use of Deep Neural Network based systems in the real world is growing. They have achieved state-of-the-art performance on many image, speech and text datasets. They have been shown to be powerful systems that are capable of learning detailed relationships and abstractions from the data. This is a double-edged sword which makes such systems vulnerable to learning the noise in the training set, thereby negatively impacting performance. This is also known as the problem of \emph{overfitting} or \emph{poor generalization}. In a practical setting, analysts typically have limited data to build models that must generalize to unseen data. In this work, we explore the use of a differential-privacy based approach to improve generalization in Deep Neural Networks. 

---
# A Discordance-Aware Multimodal Framework with Multi-Agent Clinical Reasoning 

**Authors**: Pegah Ahadian, Mingrui Yang, Sixu Chen, Xiaojuan Li, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16333)  

**Abstract**: Knee osteoarthritis frequently exhibits discordance between structural damage observed in imaging and patient-reported symptoms such as pain. This mismatch complicates clinical interpretation and patient stratification and remains insufficiently modeled in existing decision support systems. We propose a discordance aware multimodal framework that combines machine learning prediction models with a tool grounded multi agent reasoning system. Using baseline data from the FNIH Osteoarthritis Biomarkers Consortium, we trained multimodal models to predict two progression tasks, joint space loss only progression versus non progression, and pain only progression versus non progression. The predictive system integrates three modality specific experts: a CatBoost tabular model using demographic, radiographic, MRI-derived scalar, and biomarker features; MRI image embeddings extracted using a ResNet18 backbone; and Xray embeddings derived from the same architecture. Expert predictions are fused using a stacking ensemble. Residual based models estimate expected pain from structural features, enabling the computation of a pain structure discordance score between observed and expected symptoms. A multi-agent reasoning layer interprets these signals to assign clinically interpretable OA phenotypes and generate phenotype specific management recommendations. 

---
# BrainMem: Brain-Inspired Evolving Memory for Embodied Agent Task Planning 

**Authors**: Xiaoyu Ma, Lianyu Hu, Wenbing Tang, Zixuan Hu, Zeqin Liao, Zhizhen Wu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16331)  

**Abstract**: Embodied task planning requires agents to execute long-horizon, goal-directed actions in complex 3D environments, where success depends on both immediate perception and accumulated experience across tasks. However, most existing LLM-based planners are stateless and reactive, operating without persistent memory and therefore repeating errors and struggling with spatial or temporal dependencies. We propose BrainMem(Brain-Inspired Evolving Memory), a training-free hierarchical memory system that equips embodied agents with working, episodic, and semantic memory inspired by human cognition. BrainMem continuously transforms interaction histories into structured knowledge graphs and distilled symbolic guidelines, enabling planners to retrieve, reason over, and adapt behaviors from past experience without any model fine-tuning or additional training. This plug-and-play design integrates seamlessly with arbitrary multi-modal LLMs and greatly reduces reliance on task-specific prompt engineering. Extensive experiments on four representative benchmarks, including EB-ALFRED, EB-Navigation, EB-Manipulation, and EB-Habitat, demonstrate that BrainMem significantly enhances task success rates across diverse models and difficulty subsets, with the largest gains observed on long-horizon and spatially complex tasks. These results highlight evolving memory as a promising and scalable mechanism for generalizable embodied intelligence. 

---
# Beyond Single-Score Ranking: Facet-Aware Reranking for Controllable Diversity in Paper Recommendation 

**Authors**: Duan Ming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16329)  

**Abstract**: Current paper recommendation systems output a single similarity score that mixes different notions of relatedness, so users cannot specify why papers should be similar. We present SciFACE (Scientific Faceted Cross-Encoder), a reranking framework that models two independent facets: Background (what problem is studied) and Method (how it is solved). SciFACE trains two separate cross-encoders on 5,891 real seed-candidate paper pairs labeled by GPT-4o-mini with facet-specific criteria and validated against human judgments. On CSFCube, SciFACE reaches 70.63 NDCG@20 on Background (5.9 points above SPECTER) and 49.06 NDCG@20 on Method (31.1 points above SPECTER), competitive with state-of-the-art results. Compared with FaBLE without citation pre-training, SciFACE improves Method NDCG@20 by 4.1 points while using 5,891 labeled pairs versus 40K synthetic augmentations. These results show that high-quality grounded facet labels can be more data-efficient than large-scale synthetic augmentation for learning fine-grained scientific similarity. 

---
# UniMamba: A Unified Spatial-Temporal Modeling Framework with State-Space and Attention Integration 

**Authors**: Xingsheng Chen, Xianpei Mu, Deyu Yi, Yilin Yuan, Xingwei He, Bo Gao, Regina Zhang, Pietro Lio, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16325)  

**Abstract**: Multivariate time series forecasting is fundamental to numerous domains such as energy, finance, and environmental monitoring, where complex temporal dependencies and cross-variable interactions pose enduring challenges. Existing Transformer-based methods capture temporal correlations through attention mechanisms but suffer from quadratic computational cost, while state-space models like Mamba achieve efficient long-context modeling yet lack explicit temporal pattern recognition. Therefore we introduce UniMamba, a unified spatial-temporal forecasting framework that integrates efficient state-space dynamics with attention-based dependency learning. UniMamba employs a Mamba Variate-Channel Encoding Layer enhanced with FFT-Laplace Transform and TCN to capture global temporal dependencies, and a Spatial Temporal Attention Layer to jointly model inter-variate correlations and temporal evolution. A Feedforward Temporal Dynamics Layer further fuses continuous and discrete contexts for accurate forecasting. Comprehensive experiments on eight public benchmark datasets demonstrate that UniMamba consistently outperforms state-of-the-art forecasting models in both forecasting accuracy and computational efficiency, establishing a scalable and robust solution for long-sequence multivariate time-series prediction. 

---
# Beyond the 'Diff': Addressing Agentic Entropy in Agentic Software Development 

**Authors**: Matteo Casserini, Alessandro Facchini, Andrea Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2604.16323)  

**Abstract**: As autonomous coding agents become deeply embedded in software development workflows, their high operational velocity introduces a critical oversight challenge: the accumulating divergence between agentic actions and architectural intent. We term this process agentic entropy: a systemic drift that traditional code diff-based and HCXAI methods fail to capture, as they address local outputs rather than global agentic behaviour. To close this gap, we propose a process-oriented explainability framework that exposes how agentic decisions unfold across time, tool calls, and architectural boundaries. Built around three pillars (conformity seeding, reasoning monitoring, and a causal graph interface) our approach provides intent-level telemetry that complements, rather than replaces, existing review practices. We demonstrate its relevance across two user profiles: lay users engaged in vibe coding, who gain structural visibility otherwise masked by functional success; and professional developers, who gain richer contextual grounding for code review without increased overhead. By treating cognitive drift as a first-class concern alongside code quality, our framework supports the minimum level of human comprehension required for agentic oversight to remain substantive. 

---
# Steerable Instruction Following Coding Data Synthesis with Actor-Parametric Schema Co-Evolution 

**Authors**: Tinglin Huang, Bo Chen, Xiao Zhang, Kai Shen, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2604.16322)  

**Abstract**: Interpreting and following human instructions is a critical capability of large language models (LLMs) in automatic programming. However, synthesizing large-scale instruction-paired coding data remains largely unexplored and is particularly challenging when ensuring logical compatibility among multiple constraints. In this study, we propose IFCodeEvolve, an actor-schema co-evolution framework for instruction following coding data generation. By representing instructions as parametric function schema, we construct a library that covers the vast instruction space via dynamic constraint instantiation. Building upon this, Monte Carlo Tree Search (MCTS) sampler is applied to efficiently navigate this space, utilizing actor model feedback as a dynamic termination signal. Furthermore, to progressively explore challenging problems, we introduce a co-evolving paradigm that iteratively advances both the actor model and the schema library, via schema composition and mutation, based on sampler statistics. Empirical results demonstrate that IFCodeEvolve significantly boosts base model performance, with our 32B model achieving parity with proprietary SOTA models. Additionally, we contribute IFCodeBench, a comprehensive human-verified benchmark equipped with solutions and robust AST-based verification. 

---
# How Robustly do LLMs Understand Execution Semantics? 

**Authors**: Claudio Spiess, Prem Devanbu, Earl T. Barr  

**Link**: [PDF](https://arxiv.org/pdf/2604.16320)  

**Abstract**: LLMs demonstrate remarkable reasoning capabilities, yet whether they utilize internal world models or rely on sophisticated pattern matching remains open. We study LLMs through the lens of robustness of their code understanding using a standard program-output prediction task. Our results reveal a stark divergence in model behavior: while open-source reasoning models (DeepSeek-R1 family) maintain stable, albeit somewhat lower accuracies (38% to 67%) under code transformations & input perturbations, the frontier model GPT-5.2 exhibits significant brittleness. Despite achieving a near-perfect score of 99% on the original, unperturbed CRUXEval benchmark, perturbed inputs trigger accuracy declines between 20% and 24%. In addition, we find that many models perform much worse at predicting behavior on perturbed inputs that raise exceptions, and that prediction performance depends on the kind of exception. We study remedies to address this deficiency in exception prediction, and evaluate the effect of these remedies on the ability to predict non-exception behaviors. Our findings both point to limitations in the way all models understand code, and establish the value of using perturbation to evaluate code models. 

---
# Paper2Data: Large-Scale LLM Extraction and Metadata Structuring of Global Urban Data from Scientific Literature 

**Authors**: Runwen You, Tong Xia, Jingzhi Wang, Jiankun Zhang, Tengyao Tu, Jinghua Piao, Yi Chang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16317)  

**Abstract**: Urban data support a wide range of applications across multiple disciplines. However, at the global scale, there is no unified platform for urban data discovery. As a result, researchers often have to manually search through websites or scientific literature to identify relevant datasets. To address this problem, we curate an open urban data discovery portal, \textit{UrbanDataMiner}, which supports dataset-level search and filtering over more than 60{,}000 urban datasets extracted from over 15{,}000 Nature-affiliated publications. \textit{UrbanDataMiner} is enabled by \textit{Paper2Data}, a novel large-scale LLM-driven pipeline that automatically identifies dataset mentions in scientific papers and structures them using a unified urban data metadata schema. Human-annotated evaluation demonstrates that \textit{Paper2Data} achieves high recall (approximately 90\%) in dataset identification and high field-level precision (above 80\%). In addition, \textit{UrbanDataMiner} can retrieve over 9\% of datasets that are not easily discoverable through general-purpose search engines such as Google. Overall, our work provides the first large-scale, literature-derived infrastructure for urban data discovery and enables more systematic and reusable data-driven research across disciplines. Our code and data are publicly available\footnote{this https URL}. 

---
# MARA: A Multimodal Adaptive Retrieval-Augmented Framework for Document Question Answering 

**Authors**: Hui Wu, Haoquan Zhai, Yuchen Li, Hengyi Cai, Peirong Zhang, Yidan Zhang, Lei Wang, Chunle Wang, Yingyan Hou, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16313)  

**Abstract**: Retrieval-based multimodal document QA aims to identify and integrate relevant information from visually rich documents with complex multimodal structures. While retrieval-augmented generation (RAG) has shown strong performance in text-based QA, its extensions to multimodal documents remain underexplored and face significant limitations. Specifically, current approaches rely on query-agnostic document representations that overlook salient content and use static top-k evidence selection, which fails to adapt to the uncertain distribution of relevant information. To address these limitations, we propose the Multimodal Adaptive Retrieval-Augmented (MARA) framework, which introduces query-adaptive mechanisms to both retrieval and generation. MARA consists of two components: a Query-Aligned Region Encoder that builds multi-level document representations and reweights them based on query relevance to improve retrieval precision; and a Self-Reflective Evidence Controller that monitors evidence sufficiency during generation and adaptively incorporates content from lower-ranked sources using a sliding-window strategy. Experiments on six multimodal QA benchmarks demonstrate that MARA consistently improves retrieval relevance and answer quality over existing SOTA method. 

---
# FlexStructRAG: Flexible Structure-Aware Multi-Granular Relational Retrieval for RAG 

**Authors**: Mengzhu Chen, Haodong Yang, Jia Cai, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16312)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems critically depend on how external knowledge is segmented, structured, and retrieved. Most existing approaches either retrieve fixed-length text chunks, which fragments discourse context, or commit to a single structured index (e.g., a knowledge graph or hypergraph), which hard-codes one relational granularity. This often yields brittle retrieval when queries require different forms of evidence, such as local binary relations, higher-order interactions, or broader document-grounded context. We propose \textbf{FlexStructRAG}, a flexible structure-aware RAG framework that supports \emph{multi-granular, query-adaptive retrieval} over heterogeneous knowledge representations. FlexStructRAG jointly constructs (i) a knowledge graph for binary relations, (ii) a knowledge hypergraph for n-ary relations, and (iii) structure-aware semantic clusters that aggregate relational evidence into document-grounded context units. To reduce semantic fragmentation induced by uniform chunking, we introduce dynamic partitioning and a truncated sliding-window extraction mechanism that incorporates bounded contextual dependencies during knowledge construction. At inference time, FlexStructRAG enables entity-, edge-, hyperedge-, and cluster-level retrieval, which can be flexibly combined to supply generation with relationally and contextually aligned evidence. Experiments on the UltraDomain benchmark across four domains show that FlexStructRAG improves semantic evaluation over strong RAG baselines. Ablation and sensitivity analysis further demonstrate the necessity of multi-granular relational retrieval and structure-aware clustering. 

---
# Multimodal Claim Extraction for Fact-Checking 

**Authors**: Joycelyn Teo, Rui Cao, Zhenyun Deng, Zifeng Ding, Michael Sejr Schlichtkrull, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2604.16311)  

**Abstract**: Automated Fact-Checking (AFC) relies on claim extraction as a first step, yet existing methods largely overlook the multimodal nature of today's misinformation. Social media posts often combine short, informal text with images such as memes, screenshots, and photos, creating challenges that differ from both text-only claim extraction and well-studied multimodal tasks like image captioning or visual question answering. In this work, we present the first benchmark for multimodal claim extraction from social media, consisting of posts containing text and one or more images, annotated with gold-standard claims derived from real-world fact-checkers. We evaluate state-of-the-art multimodal LLMs (MLLMs) under a three-part evaluation framework (semantic alignment, faithfulness, and decontextualization) and find that baseline MLLMs struggle to model rhetorical intent and contextual cues. To address this, we introduce MICE, an intent-aware framework which shows improvements in intent-critical cases. 

---
# RAG-DIVE: A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Jan-Henrik Böttcher, Klaus Schmid, Mircea-Cristian Racasan, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16310)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems using static multi-turn datasets fails to capture the dynamic nature of real-world dialogues. Existing evaluation methods rely on predefined datasets, which restrict them to static, one-directional queries and limit their ability to capture the adaptive, context-dependent performance of RAG systems in interactive, multi-turn settings. Thus, we introduce the RAG-DIVE, a Dynamic Interactive Validation and Evaluation approach, that simulates user interactions with RAG systems. RAG-DIVE leverages an LLM to generate multi-turn conversations dynamically and is organized into three components. The dialogue generation stage consists of the (1) Conversation Generator, which simulates a user by creating multi-turn queries, and the (2) Conversation Validator, which filters and corrects invalid or low-quality outputs to ensure coherent conversations. The evaluation stage is handled by the (3) Conversation Evaluator, which assesses the RAG system's performance across the entire dialogue and generates both per-turn and multi-turn metrics that provide an aggregated view of system behavior. We validated RAG-DIVE through two experimental setups. First, we tested a sample RAG system, including human evaluation of dialogue quality, repeated trials to assess consistency, and an ablation study showing that RAG-DIVE detects performance changes caused by system modifications. Second, we compared RAG-DIVE with a traditional static dataset evaluation on an industrial RAG system under different configurations to verify whether both approaches reveal similar performance trends. Our findings demonstrate that RAG-DIVE facilitates dynamic, interaction-driven evaluation for multi-turn conversations, thereby advancing the assessment of RAG systems. 

---
# Results-Actionability Gap: Understanding How Practitioners Evaluate LLM Products in the Wild 

**Authors**: Willem van der Maden, Malak Sadek, Ziang Xiao, Aske Mottelson, Q. Vera Liao, Jichen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16304)  

**Abstract**: How do product teams evaluate LLM-powered products? As organizations integrate large language models (LLMs) into digital products, their unpredictable nature makes traditional evaluation approaches inadequate, yet little is known about how practitioners navigate this challenge. Through interviews with nineteen practitioners across diverse sectors, we identify ten evaluation practices spanning informal 'vibe checks' to organizational meta-work. Beyond confirming four documented challenges, we introduce a novel fifth we call the results-actionability gap, in which practitioners gather evaluation data but cannot translate findings into concrete improvements. Drawing on patterns from successful teams, we contribute strategies to bridge this gap, supporting practitioners' formalization journey from ad-hoc interpretive practices (e.g., vibe checks) toward systematic evaluation. Our analysis suggests these interpretive practices are necessary adaptations to LLM characteristics rather than methodological failures. For HCI researchers, this presents a research opportunity to support practitioners in systematizing emerging practices rather than developing new evaluation frameworks. 

---
# Domain-Specific Query Understanding for Automotive Applications: A Modular and Scalable Approach 

**Authors**: Isha Motiyani, Abhishek Kumar, Tilak Kasturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16301)  

**Abstract**: Despite the growing prevalence of large language models (LLMs) in domain-specific applications, the challenge of query understanding in the automotive sector still remains underexplored. This domain presents unique complexities due to its specialized vocabulary and the diverse range of user intents it encompasses. Unlike general-purpose assistants, automotive systems must precisely interpret user queries and route them to appropriate underlying tool, each designed to fulfill a distinct task such as part recommendations, repair procedures, or regulatory lookups. Moreover, these systems must extract structured inputs precisely aligned with the schema required by each tool. In this study, we present a novel two-step system for domain-specific query interpretation in the automotive context that achieves an effective balance between responsiveness, reliability, and scalability. Our initial single-step approach, which jointly performed classification and entity extraction, exhibited moderate performance and higher latency. By decomposing the task into a lightweight classification stage followed by targeted entity extraction using smaller, specialized prompts, our system achieves substantial gains in both efficiency and accuracy. Due to the niche nature of the automotive domain, we also curated a high-quality dataset by combining manually annotated and synthetically generated samples, all reviewed by domain experts. Overall, our findings demonstrate that decomposing query understanding into modular subtasks leads to a scalable, accurate, and latency-efficient solution. This approach establishes a strong ground for practical deployment in real-world automotive query understanding systems. 

---
# A3-FPN: Asymptotic Content-Aware Pyramid Attention Network for Dense Visual Prediction 

**Authors**: Meng'en Qin, Yu Song, Quanling Zhao, Xiaodong Yang, Yingtao Che, Xiaohui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10210)  

**Abstract**: Learning multi-scale representations is the common strategy to tackle object scale variation in dense prediction tasks. Although existing feature pyramid networks have greatly advanced visual recognition, inherent design defects inhibit them from capturing discriminative features and recognizing small objects. In this work, we propose Asymptotic Content-Aware Pyramid Attention Network (A3-FPN), to augment multi-scale feature representation via the asymptotically disentangled framework and content-aware attention modules. Specifically, A3-FPN employs a horizontally-spread column network that enables asymptotically global feature interaction and disentangles each level from all hierarchical representations. In feature fusion, it collects supplementary content from the adjacent level to generate position-wise offsets and weights for context-aware resampling, and learns deep context reweights to improve intra-category similarity. In feature reassembly, it further strengthens intra-scale discriminative feature learning and reassembles redundant features based on information content and spatial variation of feature maps. Extensive experiments on MS COCO, VisDrone2019-DET and Cityscapes demonstrate that A3-FPN can be easily integrated into state-of-the-art CNN and Transformer-based architectures, yielding remarkable performance gains. Notably, when paired with OneFormer and Swin-L backbone, A3-FPN achieves 49.6 mask AP on MS COCO and 85.6 mIoU on Cityscapes. Codes are available at this https URL. 

---
