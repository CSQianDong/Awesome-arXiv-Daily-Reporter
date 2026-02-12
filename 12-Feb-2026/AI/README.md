# FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight 

**Authors**: Jiayi Zhou, Yang Sheng, Hantao Lou, Yaodong Yang, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11136)  

**Abstract**: As LLM-based agents increasingly operate in high-stakes domains with real-world consequences, ensuring their behavioral safety becomes paramount. The dominant oversight paradigm, LLM-as-a-Judge, faces a fundamental dilemma: how can probabilistic systems reliably supervise other probabilistic systems without inheriting their failure modes? We argue that formal verification offers a principled escape from this dilemma, yet its adoption has been hindered by a critical bottleneck: the translation from natural language requirements to formal specifications. This paper bridges this gap by proposing , a neuro-symbolic framework that employs a bidirectional Formal-of-Thought architecture: LLMs serve as specification compilers that top-down decompose high-level human intent into atomic, verifiable constraints, then bottom-up prove compliance using Dafny specifications and Z3 Satisfiability modulo theories solving, which produces mathematical guarantees rather than probabilistic scores. We validate across three benchmarks spanning behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. Experiments on 7 agent models demonstrate that achieves an average improvement of 16.6% over LLM-as-a-Judge baselines, enables weak-to-strong generalization where a 7B judge achieves over 90% accuracy detecting deception from 72B agents, and provides near-linear safety improvement through iterative refinement. 

---
# GameDevBench: Evaluating Agentic Capabilities Through Game Development 

**Authors**: Wayne Chi, Yixiong Fang, Arnav Yayavaram, Siddharth Yayavaram, Seth Karten, Qiuhong Anna Wei, Runkun Chen, Alexander Wang, Valerie Chen, Ameet Talwalkar, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2602.11103)  

**Abstract**: Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. Game development provides such a testbed as agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 132 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex -- the average solution requires over three times the amount of lines of code and file changes compared to prior software development benchmarks. Agents still struggle with game development, with the best agent solving only 54.5% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with success rates dropping from 46.9% on gameplay-oriented tasks to 31.6% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, with the largest change being an increase in Claude Sonnet 4.5's performance from 33.3% to 47.7%. We release GameDevBench publicly to support further research into agentic game development. 

---
# CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion 

**Authors**: Yusong Lin, Haiyang Wang, Shuzhe Wu, Lue Fan, Feiyang Pan, Sanyuan Zhao, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10999)  

**Abstract**: Agentic coding requires agents to effectively interact with runtime environments, e.g., command line interfaces (CLI), so as to complete tasks like resolving dependency issues, fixing system problems, etc. But it remains underexplored how such environment-intensive tasks can be obtained at scale to enhance agents' capabilities. To address this, based on an analogy between the Dockerfile and the agentic task, we propose to employ agents to simulate and explore environment histories, guided by execution feedback. By tracing histories of a healthy environment, its state can be inverted to an earlier one with runtime failures, from which a task can be derived by packing the buggy state and the corresponding error messages. With our method, named CLI-Gym, a total of 1,655 environment-intensive tasks are derived, being the largest collection of its kind. Moreover, with curated successful trajectories, our fine-tuned model, named LiberCoder, achieves substantial absolute improvements of +21.1% (to 46.1%) on Terminal-Bench, outperforming various strong baselines. To our knowledge, this is the first public pipeline for scalable derivation of environment-intensive tasks. 

---
# Can LLMs Cook Jamaican Couscous? A Study of Cultural Novelty in Recipe Generation 

**Authors**: F. Carichon, R. Rampa, G. Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.10964)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate and shape cultural content, ranging from narrative writing to artistic production. While these models demonstrate impressive fluency and generative capacity, prior work has shown that they also exhibit systematic cultural biases, raising concerns about stereotyping, homogenization, and the erasure of culturally specific forms of expression. Understanding whether LLMs can meaningfully align with diverse cultures beyond the dominant ones remains a critical challenge. In this paper, we study cultural adaptation in LLMs through the lens of cooking recipes, a domain in which culture, tradition, and creativity are tightly intertwined. We build on the \textit{GlobalFusion} dataset, which pairs human recipes from different countries according to established measures of cultural distance. Using the same country pairs, we generate culturally adapted recipes with multiple LLMs, enabling a direct comparison between human and LLM behavior in cross-cultural content creation. Our analysis shows that LLMs fail to produce culturally representative adaptations. Unlike humans, the divergence of their generated recipes does not correlate with cultural distance. We further provide explanations for this gap. We show that cultural information is weakly preserved in internal model representations, that models inflate novelty in their production by misunderstanding notions such as creativity and tradition, and that they fail to identify adaptation with its associated countries and to ground it in culturally salient elements such as ingredients. These findings highlight fundamental limitations of current LLMs for culturally oriented generation and have important implications for their use in culturally sensitive applications. 

---
# Reinforcing Chain-of-Thought Reasoning with Self-Evolving Rubrics 

**Authors**: Leheng Sheng, Wenchang Ma, Ruixin Hong, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.10885)  

**Abstract**: Despite chain-of-thought (CoT) playing crucial roles in LLM reasoning, directly rewarding it is difficult: training a reward model demands heavy human labeling efforts, and static RMs struggle with evolving CoT distributions and reward hacking. These challenges motivate us to seek an autonomous CoT rewarding approach that requires no human annotation efforts and can evolve gradually. Inspired by recent self-evolving training methods, we propose \textbf{RLCER} (\textbf{R}einforcement \textbf{L}earning with \textbf{C}oT Supervision via Self-\textbf{E}volving \textbf{R}ubrics), which enhances the outcome-centric RLVR by rewarding CoTs with self-proposed and self-evolving rubrics. We show that self-proposed and self-evolving rubrics provide reliable CoT supervision signals even without outcome rewards, enabling RLCER to outperform outcome-centric RLVR. Moreover, when used as in-prompt hints, these self-proposed rubrics further improve inference-time performance. 

---
# SynergyKGC: Reconciling Topological Heterogeneity in Knowledge Graph Completion via Topology-Aware Synergy 

**Authors**: Xuecheng Zou, Yu Tang, Bingbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10845)  

**Abstract**: Knowledge Graph Completion (KGC) fundamentally hinges on the coherent fusion of pre-trained entity semantics with heterogeneous topological structures to facilitate robust relational reasoning. However, existing paradigms encounter a critical "structural resolution mismatch," failing to reconcile divergent representational demands across varying graph densities, which precipitates structural noise interference in dense clusters and catastrophic representation collapse in sparse regions. We present SynergyKGC, an adaptive framework that advances traditional neighbor aggregation to an active Cross-Modal Synergy Expert via relation-aware cross-attention and semantic-intent-driven gating. By coupling a density-dependent Identity Anchoring strategy with a Double-tower Coherent Consistency architecture, SynergyKGC effectively reconciles topological heterogeneity while ensuring representational stability across training and inference phases. Systematic evaluations on two public benchmarks validate the superiority of our method in significantly boosting KGC hit rates, providing empirical evidence for a generalized principle of resilient information integration in non-homogeneous structured data. 

---
# See, Plan, Snap: Evaluating Multimodal GUI Agents in Scratch 

**Authors**: Xingyi Zhang, Yulei Ye, Kaifeng Huang, Wenhao Li, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10814)  

**Abstract**: Block-based programming environments such as Scratch play a central role in low-code education, yet evaluating the capabilities of AI agents to construct programs through Graphical User Interfaces (GUIs) remains underexplored. We introduce ScratchWorld, a benchmark for evaluating multimodal GUI agents on program-by-construction tasks in Scratch. Grounded in the Use-Modify-Create pedagogical framework, ScratchWorld comprises 83 curated tasks spanning four distinct problem categories: Create, Debug, Extend, and Compute. To rigorously diagnose the source of agent failures, the benchmark employs two complementary interaction modes: primitive mode requires fine-grained drag-and-drop manipulation to directly assess visuomotor control, while composite mode uses high-level semantic APIs to disentangle program reasoning from GUI execution. To ensure reliable assessment, we propose an execution-based evaluation protocol that validates the functional correctness of the constructed Scratch programs through runtime tests within the browser environment. Extensive experiments across state-of-the-art multimodal language models and GUI agents reveal a substantial reasoning--acting gap, highlighting persistent challenges in fine-grained GUI manipulation despite strong planning capabilities. 

---
# Integrating Generative AI-enhanced Cognitive Systems in Higher Education: From Stakeholder Perceptions to a Conceptual Framework considering the EU AI Act 

**Authors**: Da-Lun Chen, Prasasthy Balasubramanian, Lauri Lovén, Susanna Pirttikangas, Jaakko Sauvola, Panagiotis Kostakos  

**Link**: [PDF](https://arxiv.org/pdf/2602.10802)  

**Abstract**: Many staff and students in higher education have adopted generative artificial intelligence (GenAI) tools in their work and study. GenAI is expected to enhance cognitive systems by enabling personalized learning and streamlining educational services. However, stakeholders perceptions of GenAI in higher education remain divided, shaped by cultural, disciplinary, and institutional contexts. In addition, the EU AI Act requires universities to ensure regulatory compliance when deploying cognitive systems. These developments highlight the need for institutions to engage stakeholders and tailor GenAI integration to their needs while addressing concerns. This study investigates how GenAI is perceived within the disciplines of Information Technology and Electrical Engineering (ITEE). Using a mixed-method approach, we surveyed 61 staff and 37 students at the Faculty of ITEE, University of Oulu. The results reveal both shared and discipline-specific themes, including strong interest in programming support from GenAI and concerns over response quality, privacy, and academic integrity. Drawing from these insights, the study identifies a set of high-level requirements and proposes a conceptual framework for responsible GenAI integration. Disciplinary-specific requirements reinforce the importance of stakeholder engagement when integrating GenAI into higher education. The high-level requirements and the framework provide practical guidance for universities aiming to harness GenAI while addressing stakeholder concerns and ensuring regulatory compliance. 

---
# Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation 

**Authors**: Jie Jiang, Yangru Huang, Zeyu Wang, Changping Wang, Yuling Xiong, Jun Zhang, Huan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10699)  

**Abstract**: Generative recommendation via autoregressive models has unified retrieval and ranking into a single conditional generation framework. However, fine-tuning these models with Reinforcement Learning (RL) often suffers from a fundamental probability-reward mismatch. Conventional likelihood-dominated decoding (e.g., beam search) exhibits a myopic bias toward locally probable prefixes, which causes two critical failures: (1) insufficient exploration, where high-reward items in low-probability branches are prematurely pruned and rarely sampled, and (2) advantage compression, where trajectories sharing high-probability prefixes receive highly correlated rewards with low within-group variance, yielding a weak comparative signal for RL. To address these challenges, we propose V-STAR, a Value-guided Sampling and Tree-structured Advantage Reinforcement framework. V-STAR forms a self-evolving loop via two synergistic components. First, a Value-Guided Efficient Decoding (VED) is developed to identify decisive nodes and selectively deepen high-potential prefixes. This improves exploration efficiency without exhaustive tree search. Second, we propose Sibling-GRPO, which exploits the induced tree topology to compute sibling-relative advantages and concentrates learning signals on decisive branching decisions. Extensive experiments on both offline and online datasets demonstrate that V-STAR outperforms state-of-the-art baselines, delivering superior accuracy and candidate-set diversity under strict latency constraints. 

---
# OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization 

**Authors**: Keane Ong, Sabri Boughorbel, Luwei Xiao, Chanakya Ekbote, Wei Dai, Ao Qu, Jingyao Wu, Rui Mao, Ehsan Hoque, Erik Cambria, Gianmarco Mengaldo, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10635)  

**Abstract**: To develop socially intelligent AI, existing approaches typically model human behavioral dimensions (e.g., affective, cognitive, or social attributes) in isolation. Although useful, task-specific modeling often increases training costs and limits generalization across behavioral settings. Recent reasoning RL methods facilitate training a single unified model across multiple behavioral tasks, but do not explicitly address learning across different heterogeneous behavioral data. To address this gap, we introduce Heterogeneity-Aware Relative Policy Optimization (HARPO), an RL method that balances leaning across heterogeneous tasks and samples. This is achieved by modulating advantages to ensure that no single task or sample carries disproportionate influence during policy optimization. Using HARPO, we develop and release Omnisapiens-7B 2.0, a foundation model for social behavior processing. Relative to existing behavioral foundation models, Omnisapiens-7B 2.0 achieves the strongest performance across behavioral tasks, with gains of up to +16.85% and +9.37% on multitask and held-out settings respectively, while producing more explicit and robust reasoning traces. We also validate HARPO against recent RL methods, where it achieves the most consistently strong performance across behavioral tasks. 

---
# To Think or Not To Think, That is The Question for Large Reasoning Models in Theory of Mind Tasks 

**Authors**: Nanxu Gong, Haotian Li, Sixun Dong, Jianxun Lian, Yanjie Fu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.10625)  

**Abstract**: Theory of Mind (ToM) assesses whether models can infer hidden mental states such as beliefs, desires, and intentions, which is essential for natural social interaction. Although recent progress in Large Reasoning Models (LRMs) has boosted step-by-step inference in mathematics and coding, it is still underexplored whether this benefit transfers to socio-cognitive skills. We present a systematic study of nine advanced Large Language Models (LLMs), comparing reasoning models with non-reasoning models on three representative ToM benchmarks. The results show that reasoning models do not consistently outperform non-reasoning models and sometimes perform worse. A fine-grained analysis reveals three insights. First, slow thinking collapses: accuracy significantly drops as responses grow longer, and larger reasoning budgets hurt performance. Second, moderate and adaptive reasoning benefits performance: constraining reasoning length mitigates failure, while distinct success patterns demonstrate the necessity of dynamic adaptation. Third, option matching shortcut: when multiple choice options are removed, reasoning models improve markedly, indicating reliance on option matching rather than genuine deduction. We also design two intervention approaches: Slow-to-Fast (S2F) adaptive reasoning and Think-to-Match (T2M) shortcut prevention to further verify and mitigate the problems. With all results, our study highlights the advancement of LRMs in formal reasoning (e.g., math, code) cannot be fully transferred to ToM, a typical task in social reasoning. We conclude that achieving robust ToM requires developing unique capabilities beyond existing reasoning methods. 

---
# Neuro-symbolic Action Masking for Deep Reinforcement Learning 

**Authors**: Shuai Han, Mehdi Dastani, Shihan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10598)  

**Abstract**: Deep reinforcement learning (DRL) may explore infeasible actions during training and execution. Existing approaches assume a symbol grounding function that maps high-dimensional states to consistent symbolic representations and a manually specified action masking techniques to constrain actions. In this paper, we propose Neuro-symbolic Action Masking (NSAM), a novel framework that automatically learn symbolic models, which are consistent with given domain constraints of high-dimensional states, in a minimally supervised manner during the DRL process. Based on the learned symbolic model of states, NSAM learns action masks that rules out infeasible actions. NSAM enables end-to-end integration of symbolic reasoning and deep policy optimization, where improvements in symbolic grounding and policy learning mutually reinforce each other. We evaluate NSAM on multiple domains with constraints, and experimental results demonstrate that NSAM significantly improves sample efficiency of DRL agent while substantially reducing constraint violations. 

---
# Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets 

**Authors**: Bo Xue, Yunchong Song, Fanghao Shao, Xuekai Zhu, Lin Chen, Luoyi Fu, Xinbing Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.10583)  

**Abstract**: Standard autoregressive language models generate text token-by-token from a fixed vocabulary, inducing a tree-structured state space when viewing token sampling as an action, which limits flexibility and expressiveness. Recent work introduces dynamic vocabulary by sampling retrieved text spans but overlooks that the same sentence can be composed of spans of varying lengths, lacking explicit modeling of the directed acyclic graph (DAG) state space. This leads to restricted exploration of compositional paths and is biased toward the chosen path. Generative Flow Networks (GFlowNets) are powerful for efficient exploring and generalizing over state spaces, particularly those with a DAG structure. However, prior GFlowNets-based language models operate at the token level and remain confined to tree-structured spaces, limiting their potential. In this work, we propose Flow of SpanS (FOSS), a principled GFlowNets framework for span generation. FoSS constructs a dynamic span vocabulary by segmenting the retrieved text flexibly, ensuring a DAG-structured state space, which allows GFlowNets to explore diverse compositional paths and improve generalization. With specialized reward models, FoSS generates diverse, high-quality text. Empirically, FoSS improves MAUVE scores by up to 12.5% over Transformer on text generation and achieves 3.5% gains on knowledge-intensive tasks, consistently outperforming state-of-the-art methods. Scaling experiments further demonstrate FoSS benefits from larger models, more data, and richer retrieval corpora, retaining its advantage over strong baselines. 

---
# Abstraction Generation for Generalized Planning with Pretrained Large Language Models 

**Authors**: Zhenhe Cui, Huaxiang Xia, Hangjun Shen, Kailun Luo, Yong He, Wei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10485)  

**Abstract**: Qualitative Numerical Planning (QNP) serves as an important abstraction model for generalized planning (GP), which aims to compute general plans that solve multiple instances at once. Recent works show that large language models (LLMs) can function as generalized planners. This work investigates whether LLMs can serve as QNP abstraction generators for GP problems and how to fix abstractions via automated debugging. We propose a prompt protocol: input a GP domain and training tasks to LLMs, prompting them to generate abstract features and further abstract the initial state, action set, and goal into QNP problems. An automated debugging method is designed to detect abstraction errors, guiding LLMs to fix abstractions. Experiments demonstrate that under properly guided by automated debugging, some LLMs can generate useful QNP abstractions. 

---
# MERIT Feedback Elicits Better Bargaining in LLM Negotiators 

**Authors**: Jihwan Oh, Murad Aghazada, Yooju Shin, Se-Young Yun, Taehyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.10467)  

**Abstract**: Bargaining is often regarded as a logical arena rather than an art or a matter of intuition, yet Large Language Models (LLMs) still struggle to navigate it due to limited strategic depth and difficulty adapting to complex human factors. Current benchmarks rarely capture this limitation. To bridge this gap, we present an utility feedback centric framework. Our contributions are: (i) AgoraBench, a new benchmark spanning nine challenging settings (e.g., deception, monopoly) that supports diverse strategy modeling; (ii) human-aligned, economically grounded metrics derived from utility theory. This is operationalized via agent utility, negotiation power, and acquisition ratio that implicitly measure how well the negotiation aligns with human preference and (iii) a human preference grounded dataset with learning pipeline that strengthens LLMs' bargaining ability through both prompting and finetuning. Empirical results indicate that baseline LLM strategies often diverge from human preferences, while our mechanism substantially improves negotiation performance, yielding deeper strategic behavior and stronger opponent awareness. 

---
# Found-RL: foundation model-enhanced reinforcement learning for autonomous driving 

**Authors**: Yansong Qu, Zihao Sheng, Zilin Huang, Jiancong Chen, Yuhao Luo, Tianyi Wang, Yiheng Feng, Samuel Labi, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10458)  

**Abstract**: Reinforcement Learning (RL) has emerged as a dominant paradigm for end-to-end autonomous driving (AD). However, RL suffers from sample inefficiency and a lack of semantic interpretability in complex scenarios. Foundation Models, particularly Vision-Language Models (VLMs), can mitigate this by offering rich, context-aware knowledge, yet their high inference latency hinders deployment in high-frequency RL training loops. To bridge this gap, we present Found-RL, a platform tailored to efficiently enhance RL for AD using foundation models. A core innovation is the asynchronous batch inference framework, which decouples heavy VLM reasoning from the simulation loop, effectively resolving latency bottlenecks to support real-time learning. We introduce diverse supervision mechanisms: Value-Margin Regularization (VMR) and Advantage-Weighted Action Guidance (AWAG) to effectively distill expert-like VLM action suggestions into the RL policy. Additionally, we adopt high-throughput CLIP for dense reward shaping. We address CLIP's dynamic blindness via Conditional Contrastive Action Alignment, which conditions prompts on discretized speed/command and yields a normalized, margin-based bonus from context-specific action-anchor scoring. Found-RL provides an end-to-end pipeline for fine-tuned VLM integration and shows that a lightweight RL model can achieve near-VLM performance compared with billion-parameter VLMs while sustaining real-time inference (approx. 500 FPS). Code, data, and models will be publicly available at this https URL. 

---
# LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation 

**Authors**: Zhiling Yan, Dingjie Song, Zhe Fang, Yisheng Ji, Xiang Li, Quanzheng Li, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.10367)  

**Abstract**: The deployment of Large Language Models (LLMs) in high-stakes clinical settings demands rigorous and reliable evaluation. However, existing medical benchmarks remain static, suffering from two critical limitations: (1) data contamination, where test sets inadvertently leak into training corpora, leading to inflated performance estimates; and (2) temporal misalignment, failing to capture the rapid evolution of medical knowledge. Furthermore, current evaluation metrics for open-ended clinical reasoning often rely on either shallow lexical overlap (e.g., ROUGE) or subjective LLM-as-a-Judge scoring, both inadequate for verifying clinical correctness. To bridge these gaps, we introduce LiveMedBench, a continuously updated, contamination-free, and rubric-based benchmark that weekly harvests real-world clinical cases from online medical communities, ensuring strict temporal separation from model training data. We propose a Multi-Agent Clinical Curation Framework that filters raw data noise and validates clinical integrity against evidence-based medical principles. For evaluation, we develop an Automated Rubric-based Evaluation Framework that decomposes physician responses into granular, case-specific criteria, achieving substantially stronger alignment with expert physicians than LLM-as-a-Judge. To date, LiveMedBench comprises 2,756 real-world cases spanning 38 medical specialties and multiple languages, paired with 16,702 unique evaluation criteria. Extensive evaluation of 38 LLMs reveals that even the best-performing model achieves only 39.2%, and 84% of models exhibit performance degradation on post-cutoff cases, confirming pervasive data contamination risks. Error analysis further identifies contextual application-not factual knowledge-as the dominant bottleneck, with 35-48% of failures stemming from the inability to tailor medical knowledge to patient-specific constraints. 

---
# Discovering Differences in Strategic Behavior Between Humans and LLMs 

**Authors**: Caroline Wang, Daniel Kasenberg, Kim Stachenfeld, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2602.10324)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in social and strategic scenarios, it becomes critical to understand where and why their behavior diverges from that of humans. While behavioral game theory (BGT) provides a framework for analyzing behavior, existing models do not fully capture the idiosyncratic behavior of humans or black-box, non-human agents like LLMs. We employ AlphaEvolve, a cutting-edge program discovery tool, to directly discover interpretable models of human and LLM behavior from data, thereby enabling open-ended discovery of structural factors driving human and LLM behavior. Our analysis on iterated rock-paper-scissors reveals that frontier LLMs can be capable of deeper strategic behavior than humans. These results provide a foundation for understanding structural differences driving differences in human and LLM behavior in strategic interactions. 

---
# Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling 

**Authors**: Gongye Liu, Bo Yang, Yida Zhi, Zhizhou Zhong, Lei Ke, Didan Deng, Han Gao, Yongxiang Huang, Kaihao Zhang, Hongbo Fu, Wenhan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11146)  

**Abstract**: Preference optimization for diffusion and flow-matching models relies on reward functions that are both discriminatively robust and computationally efficient. Vision-Language Models (VLMs) have emerged as the primary reward provider, leveraging their rich multimodal priors to guide alignment. However, their computation and memory cost can be substantial, and optimizing a latent diffusion generator through a pixel-space reward introduces a domain mismatch that complicates alignment. In this paper, we propose DiNa-LRM, a diffusion-native latent reward model that formulates preference learning directly on noisy diffusion states. Our method introduces a noise-calibrated Thurstone likelihood with diffusion-noise-dependent uncertainty. DiNa-LRM leverages a pretrained latent diffusion backbone with a timestep-conditioned reward head, and supports inference-time noise ensembling, providing a diffusion-native mechanism for test-time scaling and robust rewarding. Across image alignment benchmarks, DiNa-LRM substantially outperforms existing diffusion-based reward baselines and achieves performance competitive with state-of-the-art VLMs at a fraction of the computational cost. In preference optimization, we demonstrate that DiNa-LRM improves preference optimization dynamics, enabling faster and more resource-efficient model alignment. 

---
# GENIUS: Generative Fluid Intelligence Evaluation Suite 

**Authors**: Ruichuan An, Sihan Yang, Ziyu Guo, Wei Dai, Zijun Shen, Haodong Li, Renrui Zhang, Xinyu Wei, Guopeng Li, Wenshan Wu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11144)  

**Abstract**: Unified Multimodal Models (UMMs) have shown remarkable progress in visual generation. Yet, existing benchmarks predominantly assess $\textit{Crystallized Intelligence}$, which relies on recalling accumulated knowledge and learned schemas. This focus overlooks $\textit{Generative Fluid Intelligence (GFI)}$: the capacity to induce patterns, reason through constraints, and adapt to novel scenarios on the fly. To rigorously assess this capability, we introduce $\textbf{GENIUS}$ ($\textbf{GEN}$ Fluid $\textbf{I}$ntelligence Eval$\textbf{U}$ation $\textbf{S}$uite). We formalize $\textit{GFI}$ as a synthesis of three primitives. These include $\textit{Inducing Implicit Patterns}$ (e.g., inferring personalized visual preferences), $\textit{Executing Ad-hoc Constraints}$ (e.g., visualizing abstract metaphors), and $\textit{Adapting to Contextual Knowledge}$ (e.g., simulating counter-intuitive physics). Collectively, these primitives challenge models to solve problems grounded entirely in the immediate context. Our systematic evaluation of 12 representative models reveals significant performance deficits in these tasks. Crucially, our diagnostic analysis disentangles these failure modes. It demonstrates that deficits stem from limited context comprehension rather than insufficient intrinsic generative capability. To bridge this gap, we propose a training-free attention intervention strategy. Ultimately, $\textbf{GENIUS}$ establishes a rigorous standard for $\textit{GFI}$, guiding the field beyond knowledge utilization toward dynamic, general-purpose reasoning. Our dataset and code will be released at: $\href{this https URL}{this https URL}$. 

---
# Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows 

**Authors**: Shaswat Garg, Matin Moezzi, Brandon Da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2602.11142)  

**Abstract**: Hierarchical goal-conditioned reinforcement learning (H-GCRL) provides a powerful framework for tackling complex, long-horizon tasks by decomposing them into structured subgoals. However, its practical adoption is hindered by poor data efficiency and limited policy expressivity, especially in offline or data-scarce regimes. In this work, Normalizing flow-based hierarchical implicit Q-learning (NF-HIQL), a novel framework that replaces unimodal gaussian policies with expressive normalizing flow policies at both the high- and low-levels of the hierarchy is introduced. This design enables tractable log-likelihood computation, efficient sampling, and the ability to model rich multimodal behaviors. New theoretical guarantees are derived, including explicit KL-divergence bounds for Real-valued non-volume preserving (RealNVP) policies and PAC-style sample efficiency results, showing that NF-HIQL preserves stability while improving generalization. Empirically, NF-HIQL is evaluted across diverse long-horizon tasks in locomotion, ball-dribbling, and multi-step manipulation from OGBench. NF-HIQL consistently outperforms prior goal-conditioned and hierarchical baselines, demonstrating superior robustness under limited data and highlighting the potential of flow-based architectures for scalable, data-efficient hierarchical reinforcement learning. 

---
# Weight Decay Improves Language Model Plasticity 

**Authors**: Tessa Han, Sebastian Bordt, Hanlin Zhang, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2602.11137)  

**Abstract**: The prevailing paradigm in large language model (LLM) development is to pretrain a base model, then perform further training to improve performance and model behavior. However, hyperparameter optimization and scaling laws have been studied primarily from the perspective of the base model's validation loss, ignoring downstream adaptability. In this work, we study pretraining from the perspective of model plasticity, that is, the ability of the base model to successfully adapt to downstream tasks through fine-tuning. We focus on the role of weight decay, a key regularization parameter during pretraining. Through systematic experiments, we show that models trained with larger weight decay values are more plastic, meaning they show larger performance gains when fine-tuned on downstream tasks. This phenomenon can lead to counterintuitive trade-offs where base models that perform worse after pretraining can perform better after fine-tuning. Further investigation of weight decay's mechanistic effects on model behavior reveals that it encourages linearly separable representations, regularizes attention matrices, and reduces overfitting on the training data. In conclusion, this work demonstrates the importance of using evaluation metrics beyond cross-entropy loss for hyperparameter optimization and casts light on the multifaceted role of that a single optimization hyperparameter plays in shaping model behavior. 

---
# Learning to Compose for Cross-domain Agentic Workflow Generation 

**Authors**: Jialiang Wang, Shengxiang Xu, Hanmo Liu, Jiachuan Wang, Yuyu Luo, Shimin Di, Min-Ling Zhang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11114)  

**Abstract**: Automatically generating agentic workflows -- executable operator graphs or codes that orchestrate reasoning, verification, and repair -- has become a practical way to solve complex tasks beyond what single-pass LLM generation can reliably handle. Yet what constitutes a good workflow depends heavily on the task distribution and the available operators. Under domain shift, current systems typically rely on iterative workflow refinement to discover a feasible workflow from a large workflow space, incurring high iteration costs and yielding unstable, domain-specific behavior. In response, we internalize a decompose-recompose-decide mechanism into an open-source LLM for cross-domain workflow generation. To decompose, we learn a compact set of reusable workflow capabilities across diverse domains. To recompose, we map each input task to a sparse composition over these bases to generate a task-specific workflow in a single pass. To decide, we attribute the success or failure of workflow generation to counterfactual contributions from learned capabilities, thereby capturing which capabilities actually drive success by their marginal effects. Across stringent multi-domain, cross-domain, and unseen-domain evaluations, our 1-pass generator surpasses SOTA refinement baselines that consume 20 iterations, while substantially reducing generation latency and cost. 

---
# Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away 

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Furong Huang, Dinesh Manocha, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11096)  

**Abstract**: Reinforcement learning (RL) based post-training for explicit chain-of-thought (e.g., GRPO) improves the reasoning ability of multimodal large-scale reasoning models (MLRMs). But recent evidence shows that it can simultaneously degrade safety alignment and increase jailbreak success rates. We propose SafeThink, a lightweight inference-time defense that treats safety recovery as a satisficing constraint rather than a maximization objective. SafeThink monitors the evolving reasoning trace with a safety reward model and conditionally injects an optimized short corrective prefix ("Wait, think safely") only when the safety threshold is violated. In our evaluations across six open-source MLRMs and four jailbreak benchmarks (JailbreakV-28K, Hades, FigStep, and MM-SafetyBench), SafeThink reduces attack success rates by 30-60% (e.g., LlamaV-o1: 63.33% to 5.74% on JailbreakV-28K, R1-Onevision: 69.07% to 5.65% on Hades) while preserving reasoning performance (MathVista accuracy: 65.20% to 65.00%). A key empirical finding from our experiments is that safety recovery is often only a few steering steps away: intervening in the first 1-3 reasoning steps typically suffices to redirect the full generation toward safe completions. 

---
# Direct Learning of Calibration-Aware Uncertainty for Neural PDE Surrogates 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2602.11090)  

**Abstract**: Neural PDE surrogates are often deployed in data-limited or partially observed regimes where downstream decisions depend on calibrated uncertainty in addition to low prediction error. Existing approaches obtain uncertainty through ensemble replication, fixed stochastic noise such as dropout, or post hoc calibration. Cross-regularized uncertainty learns uncertainty parameters during training using gradients routed through a held-out regularization split. The predictor is optimized on the training split for fit, while low-dimensional uncertainty controls are optimized on the regularization split to reduce train-test mismatch, yielding regime-adaptive uncertainty without per-regime noise tuning. The framework can learn continuous noise levels at the output head, within hidden features, or within operator-specific components such as spectral modes. We instantiate the approach in Fourier Neural Operators and evaluate on APEBench sweeps over observed fraction and training-set size. Across these sweeps, the learned predictive distributions are better calibrated on held-out splits and the resulting uncertainty fields concentrate in high-error regions in one-step spatial diagnostics. 

---
# DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning 

**Authors**: Yicheng Chen, Zerun Ma, Xinchen Xie, Yining Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11089)  

**Abstract**: In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces practical recipes that reach comparable downstream performance to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing Qwen3-1.7B. This work sheds new light on automating LLM training and developing self-evolving AI systems. 

---
# General Flexible $f$-divergence for Challenging Offline RL Datasets with Low Stochasticity and Diverse Behavior Policies 

**Authors**: Jianxun Wang, Grant C. Forbes, Leonardo Villalobos-Arias, David L. Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2602.11087)  

**Abstract**: Offline RL algorithms aim to improve upon the behavior policy that produces the collected data while constraining the learned policy to be within the support of the dataset. However, practical offline datasets often contain examples with little diversity or limited exploration of the environment, and from multiple behavior policies with diverse expertise levels. Limited exploration can impair the offline RL algorithm's ability to estimate \textit{Q} or \textit{V} values, while constraining towards diverse behavior policies can be overly conservative. Such datasets call for a balance between the RL objective and behavior policy constraints. We first identify the connection between $f$-divergence and optimization constraint on the Bellman residual through a more general Linear Programming form for RL and the convex conjugate. Following this, we introduce the general flexible function formulation for the $f$-divergence to incorporate an adaptive constraint on algorithms' learning objectives based on the offline training dataset. Results from experiments on the MuJoCo, Fetch, and AdroitHand environments show the correctness of the proposed LP form and the potential of the flexible $f$-divergence in improving performance for learning from a challenging dataset when applied to a compatible constrained optimization algorithm. 

---
# GRASP: group-Shapley feature selection for patients 

**Authors**: Yuheng Luo, Shuyan Li, Zhong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11084)  

**Abstract**: Feature selection remains a major challenge in medical prediction, where existing approaches such as LASSO often lack robustness and interpretability. We introduce GRASP, a novel framework that couples Shapley value driven attribution with group $L_{21}$ regularization to extract compact and non-redundant feature sets. GRASP first distills group level importance scores from a pretrained tree model via SHAP, then enforces structured sparsity through group $L_{21}$ regularized logistic regression, yielding stable and interpretable selections. Extensive comparisons with LASSO, SHAP, and deep learning based methods show that GRASP consistently delivers comparable or superior predictive accuracy, while identifying fewer, less redundant, and more stable features. 

---
# SteuerLLM: Local specialized large language model for German tax law analysis 

**Authors**: Sebastian Wind, Jeta Sopa, Laurin Schmid, Quirin Jackl, Sebastian Kiefer, Fei Wu, Martin Mayr, Harald Köstler, Gerhard Wellein, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2602.11081)  

**Abstract**: Large language models (LLMs) demonstrate strong general reasoning and language understanding, yet their performance degrades in domains governed by strict formal rules, precise terminology, and legally binding structure. Tax law exemplifies these challenges, as correct answers require exact statutory citation, structured legal argumentation, and numerical accuracy under rigid grading schemes. We algorithmically generate SteuerEx, the first open benchmark derived from authentic German university tax law examinations. SteuerEx comprises 115 expert-validated examination questions spanning six core tax law domains and multiple academic levels, and employs a statement-level, partial-credit evaluation framework that closely mirrors real examination practice. We further present SteuerLLM, a domain-adapted LLM for German tax law trained on a large-scale synthetic dataset generated from authentic examination material using a controlled retrieval-augmented pipeline. SteuerLLM (28B parameters) consistently outperforms general-purpose instruction-tuned models of comparable size and, in several cases, substantially larger systems, demonstrating that domain-specific data and architectural adaptation are more decisive than parameter scale for performance on realistic legal reasoning tasks. All benchmark data, training datasets, model weights, and evaluation code are released openly to support reproducible research in domain-specific legal artificial intelligence. A web-based demo of SteuerLLM is available at this https URL. 

---
# In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution 

**Authors**: Frank Xiao, Santiago Aranguri  

**Link**: [PDF](https://arxiv.org/pdf/2602.11079)  

**Abstract**: We propose activation-based data attribution, a method that traces behavioral changes in post-trained language models to responsible training datapoints. By computing activation-difference vectors for both test prompts and preference pairs and ranking by cosine similarity, we identify datapoints that cause specific behaviors and validate these attributions causally by retraining with modified data. Clustering behavior-datapoint similarity matrices also enables unsupervised discovery of emergent behaviors. Applying this to OLMo 2's production DPO training, we surfaced distractor-triggered compliance: a harmful behavior where the model complies with dangerous requests when benign formatting instructions are appended. Filtering top-ranked datapoints reduces this behavior by 63% while switching their labels achieves 78%. Our method outperforms gradient-based attribution and LLM-judge baselines while being over 10 times cheaper than both. This in-the-wild model organism - emerging from contaminated preference data rather than deliberate injection - provides a realistic benchmark for safety techniques. 

---
# Interpretable Attention-Based Multi-Agent PPO for Latency Spike Resolution in 6G RAN Slicing 

**Authors**: Kavan Fatehi, Mostafa Rahmani Ghourtani, Amir Sonee, Poonam Yadav, Alessandra M Russo, Hamed Ahmadi, Radu Calinescu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11076)  

**Abstract**: Sixth-generation (6G) radio access networks (RANs) must enforce strict service-level agreements (SLAs) for heterogeneous slices, yet sudden latency spikes remain difficult to diagnose and resolve with conventional deep reinforcement learning (DRL) or explainable RL (XRL). We propose \emph{Attention-Enhanced Multi-Agent Proximal Policy Optimization (AE-MAPPO)}, which integrates six specialized attention mechanisms into multi-agent slice control and surfaces them as zero-cost, faithful explanations. The framework operates across O-RAN timescales with a three-phase strategy: predictive, reactive, and inter-slice optimization.
A URLLC case study shows AE-MAPPO resolves a latency spike in $18$ms, restores latency to $0.98$ms with $99.9999\%$ reliability, and reduces troubleshooting time by $93\%$ while maintaining eMBB and mMTC continuity. These results confirm AE-MAPPO's ability to combine SLA compliance with inherent interpretability, enabling trustworthy and real-time automation for 6G RAN slicing. 

---
# Chatting with Images for Introspective Visual Thinking 

**Authors**: Junfei Wu, Jian Guan, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, Tienie Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.11073)  

**Abstract**: Current large vision-language models (LVLMs) typically rely on text-only reasoning based on a single-pass visual encoding, which often leads to loss of fine-grained visual information. Recently the proposal of ''thinking with images'' attempts to alleviate this limitation by manipulating images via external tools or code; however, the resulting visual states are often insufficiently grounded in linguistic semantics, impairing effective cross-modal alignment - particularly when visual semantics or geometric relationships must be reasoned over across distant regions or multiple images. To address these challenges, we propose ''chatting with images'', a new framework that reframes visual manipulation as language-guided feature modulation. Under the guidance of expressive language prompts, the model dynamically performs joint re-encoding over multiple image regions, enabling tighter coupling between linguistic reasoning and visual state updates. We instantiate this paradigm in ViLaVT, a novel LVLM equipped with a dynamic vision encoder explicitly designed for such interactive visual reasoning, and trained it with a two-stage curriculum combining supervised fine-tuning and reinforcement learning to promote effective reasoning behaviors. Extensive experiments across eight benchmarks demonstrate that ViLaVT achieves strong and consistent improvements, with particularly pronounced gains on complex multi-image and video-based spatial reasoning tasks. 

---
# Conversational Behavior Modeling Foundation Model With Multi-Level Perception 

**Authors**: Dingkun Zhou, Shuchang Pan, Jiachen Lian, Siddharth Banerjee, Sarika Pasumarthy, Dhruv Hebbar, Siddhant Patel, Zeyi Austin Li, Kan Jen Cheng, Sanay Bordia, Krish Patel, Akshaj Gupta, Tingle Li, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.11065)  

**Abstract**: Human conversation is organized by an implicit chain of thoughts that manifests as timed speech acts. Capturing this perceptual pathway is key to building natural full-duplex interactive systems. We introduce a framework that models this process as multi-level perception, and then reasons over conversational behaviors via a Graph-of-Thoughts (GoT). Our approach formalizes the intent-to-action pathway with a hierarchical labeling scheme, predicting high-level communicative intents and low-level speech acts to learn their causal and temporal dependencies. To train this system, we develop a high quality corpus that pairs controllable, event-rich dialogue data with human-annotated labels. The GoT framework structures streaming predictions as an evolving graph, enabling a transformer to forecast the next speech act, generate concise justifications for its decisions, and dynamically refine its reasoning. Experiments on both synthetic and real duplex dialogues show that the framework delivers robust behavior detection, produces interpretable reasoning chains, and establishes a foundation for benchmarking conversational reasoning in full duplex spoken dialogue systems. 

---
# GraphSeek: Next-Generation Graph Analytics with LLMs 

**Authors**: Maciej Besta, Łukasz Jarmocik, Orest Hrycyna, Shachar Klaiman, Konrad Mączka, Robert Gerstenberger, Jürgen Müller, Piotr Nyczyk, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2602.11052)  

**Abstract**: Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs. 

---
# Language Model Inversion through End-to-End Differentiation 

**Authors**: Kevin Yandoka Denamganaï, Kartic Subr  

**Link**: [PDF](https://arxiv.org/pdf/2602.11044)  

**Abstract**: Despite emerging research on Language Models (LM), few approaches analyse the invertibility of LMs. That is, given a LM and a desirable target output sequence of tokens, determining what input prompts would yield the target output remains an open problem. We formulate this problem as a classical gradient-based optimisation. First, we propose a simple algorithm to achieve end-to-end differentiability of a given (frozen) LM and then find optimised prompts via gradient descent. Our central insight is to view LMs as functions operating on sequences of distributions over tokens (rather than the traditional view as functions on sequences of tokens). Our experiments and ablations demonstrate that our DLM-powered inversion can reliably and efficiently optimise prompts of lengths $10$ and $80$ for targets of length $20$, for several white-box LMs (out-of-the-box). 

---
# Linguistic Indicators of Early Cognitive Decline in the DementiaBank Pitt Corpus: A Statistical and Machine Learning Study 

**Authors**: Artsvik Avetisyan, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.11028)  

**Abstract**: Background: Subtle changes in spontaneous language production are among the earliest indicators of cognitive decline. Identifying linguistically interpretable markers of dementia can support transparent and clinically grounded screening approaches.
Methods: This study analyzes spontaneous speech transcripts from the DementiaBank Pitt Corpus using three linguistic representations: raw cleaned text, a part-of-speech (POS)-enhanced representation combining lexical and grammatical information, and a POS-only syntactic representation. Logistic regression and random forest models were evaluated under two protocols: transcript-level train-test splits and subject-level five-fold cross-validation to prevent speaker overlap. Model interpretability was examined using global feature importance, and statistical validation was conducted using Mann-Whitney U tests with Cliff's delta effect sizes.
Results: Across representations, models achieved stable performance, with syntactic and grammatical features retaining strong discriminative power even in the absence of lexical content. Subject-level evaluation yielded more conservative but consistent results, particularly for POS-enhanced and POS-only representations. Statistical analysis revealed significant group differences in functional word usage, lexical diversity, sentence structure, and discourse coherence, aligning closely with machine learning feature importance findings.
Conclusion: The results demonstrate that abstract linguistic features capture robust markers of early cognitive decline under clinically realistic evaluation. By combining interpretable machine learning with non-parametric statistical validation, this study supports the use of linguistically grounded features for transparent and reliable language-based cognitive screening. 

---
# Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting 

**Authors**: Rishikesh Bhyri, Brian R Quaranto, Philip J Seger, Kaity Tung, Brendan Fox, Gene Yang, Steven D. Schwaitzberg, Junsong Yuan, Nan Xi, Peter C W Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11024)  

**Abstract**: Accurate counting of surgical instruments in Operating Rooms (OR) is a critical prerequisite for ensuring patient safety during surgery. Despite recent progress of large visual-language models and agentic AI, accurately counting such instruments remains highly challenging, particularly in dense scenarios where instruments are tightly clustered. To address this problem, we introduce Chain-of-Look, a novel visual reasoning framework that mimics the sequential human counting process by enforcing a structured visual chain, rather than relying on classic object detection which is unordered. This visual chain guides the model to count along a coherent spatial trajectory, improving accuracy in complex scenes. To further enforce the physical plausibility of the visual chain, we introduce the neighboring loss function, which explicitly models the spatial constraints inherent to densely packed surgical instruments. We also present SurgCount-HD, a new dataset comprising 1,464 high-density surgical instrument images. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches for counting (e.g., CountGD, REC) as well as Multimodality Large Language Models (e.g., Qwen, ChatGPT) in the challenging task of dense surgical instrument counting. 

---
# ContactGaussian-WM: Learning Physics-Grounded World Model from Videos 

**Authors**: Meizhong Wang, Wanxin Jin, Kun Cao, Lihua Xie, Yiguang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2602.11021)  

**Abstract**: Developing world models that understand complex physical interactions is essential for advancing robotic planning and this http URL, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic this http URL address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video this http URL framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual this http URL simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization this http URL, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC. 

---
# OSIL: Learning Offline Safe Imitation Policies with Safety Inferred from Non-preferred Trajectories 

**Authors**: Returaj Burnwal, Nirav Pravinbhai Bhatt, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2602.11018)  

**Abstract**: This work addresses the problem of offline safe imitation learning (IL), where the goal is to learn safe and reward-maximizing policies from demonstrations that do not have per-timestep safety cost or reward information. In many real-world domains, online learning in the environment can be risky, and specifying accurate safety costs can be difficult. However, it is often feasible to collect trajectories that reflect undesirable or unsafe behavior, implicitly conveying what the agent should avoid. We refer to these as non-preferred trajectories. We propose a novel offline safe IL algorithm, OSIL, that infers safety from non-preferred demonstrations. We formulate safe policy learning as a Constrained Markov Decision Process (CMDP). Instead of relying on explicit safety cost and reward annotations, OSIL reformulates the CMDP problem by deriving a lower bound on reward maximizing objective and learning a cost model that estimates the likelihood of non-preferred behavior. Our approach allows agents to learn safe and reward-maximizing behavior entirely from offline demonstrations. We empirically demonstrate that our approach can learn safer policies that satisfy cost constraints without degrading the reward performance, thus outperforming several baselines. 

---
# From Buffers to Registers: Unlocking Fine-Grained FlashAttention with Hybrid-Bonded 3D NPU Co-Design 

**Authors**: Jinxin Yu, Yudong Pan, Mengdi Wang, Huawei Li, Yinhe Han, Xiaowei Li, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11016)  

**Abstract**: Transformer-based models dominate modern AI workloads but exacerbate memory bottlenecks due to their quadratic attention complexity and ever-growing model sizes. Existing accelerators, such as Groq and Cerebras, mitigate off-chip traffic with large on-chip caches, while algorithmic innovations such as FlashAttention fuse operators to avoid materializing large attention matrices. However, as off-chip traffic decreases, our measurements show that on-chip SRAM accesses account for over 60% of energy in long-sequence workloads, making cache access the new bottleneck. We propose 3D-Flow, a hybrid-bonded, 3D-stacked spatial accelerator that enables register-to-register communication across vertically partitioned PE tiers. Unlike 2D multi-array architectures limited by NoC-based router-to-router transfers, 3D-Flow leverages sub-10 um vertical TSVs to sustain cycle-level operator pipelining with minimal overhead. On top of this architecture, we design 3D-FlashAttention, a fine-grained scheduling method that balances latency across tiers, forming a bubble-free vertical dataflow without on-chip SRAM roundtrips. Evaluations on Transformer workloads (OPT and QWEN models) show that our 3D spatial accelerator reduces 46-93% energy consumption and achieves 1.4x-7.6x speedups compared to state-of-the-art 2D and 3D designs. 

---
# CVPL: A Geometric Framework for Post-Hoc Linkage Risk Assessment in Protected Tabular Data 

**Authors**: Valery Khvatov, Alexey Neyman  

**Link**: [PDF](https://arxiv.org/pdf/2602.11015)  

**Abstract**: Formal privacy metrics provide compliance-oriented guarantees but often fail to quantify actual linkability in released datasets. We introduce CVPL (Cluster-Vector-Projection Linkage), a geometric framework for post-hoc assessment of linkage risk between original and protected tabular data. CVPL represents linkage analysis as an operator pipeline comprising blocking, vectorization, latent projection, and similarity evaluation, yielding continuous, scenario-dependent risk estimates rather than binary compliance verdicts. We formally define CVPL under an explicit threat model and introduce threshold-aware risk surfaces, R(lambda, tau), that capture the joint effects of protection strength and attacker strictness. We establish a progressive blocking strategy with monotonicity guarantees, enabling anytime risk estimation with valid lower bounds. We demonstrate that the classical Fellegi-Sunter linkage emerges as a special case of CVPL under restrictive assumptions, and that violations of these assumptions can lead to systematic over-linking bias. Empirical validation on 10,000 records across 19 protection configurations demonstrates that formal k-anonymity compliance may coexist with substantial empirical linkability, with a significant portion arising from non-quasi-identifier behavioral patterns. CVPL provides interpretable diagnostics identifying which features drive linkage feasibility, supporting privacy impact assessment, protection mechanism comparison, and utility-risk trade-off analysis. 

---
# ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression 

**Authors**: Ammar Ali, Baher Mohammad, Denis Makhov, Dmitriy Shopkhoev, Magauiya Zhussip, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11008)  

**Abstract**: We present ROCKET, a training-free model compression method that achieves state-of-the-art performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model architectures at 20-50\% compression rates. Notably, it retains over 90\% of the original model's performance at 30\% compression without any fine-tuning. Moreover, when applying a light fine-tuning phase, recovery is substantially enhanced: for instance, compressing Qwen3-14B to an 8B-parameter model and healing it with just 30 million tokens yields performance nearly on par with the original Qwen3-8B. The code for ROCKET is at this http URL. 

---
# Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception 

**Authors**: Liangkai Liu, Kang G. Shin, Jinkyu Lee, Chengmo Yang, Weisong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11004)  

**Abstract**: Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline. 

---
# Fine-Tuning GPT-5 for GPU Kernel Generation 

**Authors**: Ali Tehrani, Yahya Emara, Essam Wissam, Wojciech Paluch, Waleed Atallah, Łukasz Dudziak, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2602.11000)  

**Abstract**: Developing efficient GPU kernels is essential for scaling modern AI systems, yet it remains a complex task due to intricate hardware architectures and the need for specialized optimization expertise. Although Large Language Models (LLMs) demonstrate strong capabilities in general sequential code generation, they face significant challenges in GPU code generation because of the scarcity of high-quality labeled training data, compiler biases when generating synthetic solutions, and limited generalization across hardware generations. This precludes supervised fine-tuning (SFT) as a scalable methodology for improving current LLMs. In contrast, reinforcement learning (RL) offers a data-efficient and adaptive alternative but requires access to relevant tools, careful selection of training problems, and a robust evaluation environment. We present Makora's environment and tools for reinforcement learning finetuning of frontier models and report our results from fine-tuning GPT-5 for Triton code generation. In the single-attempt setting, our fine-tuned model improves kernel correctness from 43.7% to 77.0% (+33.3 percentage points) and increases the fraction of problems outperforming TorchInductor from 14.8% to 21.8% (+7 percentage points) compared to baseline GPT-5, while exceeding prior state-of-the-art models on KernelBench. When integrated into a full coding agent, it is able to solve up to 97.4% of problems in an expanded KernelBench suite, outperforming the PyTorch TorchInductor compiler on 72.9% of problems with a geometric mean speedup of 2.12x. Our work demonstrates that targeted post-training with reinforcement learning can unlock LLM capabilities in highly specialized technical domains where traditional supervised learning is limited by data availability, opening new pathways for AI-assisted accelerator programming. 

---
# LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules 

**Authors**: Ivan Vulić, Adam Grycner, Quentin de Laroussilhe, Jonas Pfeiffer  

**Link**: [PDF](https://arxiv.org/pdf/2602.10993)  

**Abstract**: Despite its huge number of variants, standard Low-Rank Adaptation (LoRA) is still a dominant technique for parameter-efficient fine-tuning (PEFT). Nonetheless, it faces persistent challenges, including the pre-selection of an optimal rank and rank-specific hyper-parameters, as well as the deployment complexity of heterogeneous-rank modules and more sophisticated LoRA derivatives. In this work, we introduce LoRA-Squeeze, a simple and efficient methodology that aims to improve standard LoRA learning by changing LoRA module ranks either post-hoc or dynamically during training}. Our approach posits that it is better to first learn an expressive, higher-rank solution and then compress it, rather than learning a constrained, low-rank solution directly. The method involves fine-tuning with a deliberately high(er) source rank, reconstructing or efficiently approximating the reconstruction of the full weight update matrix, and then using Randomized Singular Value Decomposition (RSVD) to create a new, compressed LoRA module at a lower target rank. Extensive experiments across 13 text and 10 vision-language tasks show that post-hoc compression often produces lower-rank adapters that outperform those trained directly at the target rank, especially if a small number of fine-tuning steps at the target rank is allowed. Moreover, a gradual, in-tuning rank annealing variant of LoRA-Squeeze consistently achieves the best LoRA size-performance trade-off. 

---
# RiemannGL: Riemannian Geometry Changes Graph Deep Learning 

**Authors**: Li Sun, Qiqi Wan, Suyang Zhou, Zhenhao Huang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10982)  

**Abstract**: Graphs are ubiquitous, and learning on graphs has become a cornerstone in artificial intelligence and data mining communities. Unlike pixel grids in images or sequential structures in language, graphs exhibit a typical non-Euclidean structure with complex interactions among the objects. This paper argues that Riemannian geometry provides a principled and necessary foundation for graph representation learning, and that Riemannian graph learning should be viewed as a unifying paradigm rather than a collection of isolated techniques. While recent studies have explored the integration of graph learning and Riemannian geometry, most existing approaches are limited to a narrow class of manifolds, particularly hyperbolic spaces, and often adopt extrinsic manifold formulations. We contend that the central mission of Riemannian graph learning is to endow graph neural networks with intrinsic manifold structures, which remains underexplored. To advance this perspective, we identify key conceptual and methodological gaps in existing approaches and outline a structured research agenda along three dimensions: manifold type, neural architecture, and learning paradigm. We further discuss open challenges, theoretical foundations, and promising directions that are critical for unlocking the full potential of Riemannian graph learning. This paper aims to provide a coherent viewpoint and to stimulate broader exploration of Riemannian geometry as a foundational framework for future graph learning research. 

---
# FeatureBench: Benchmarking Agentic Coding for Complex Feature Development 

**Authors**: Qixing Zhou, Jiacheng Zhang, Haiyang Wang, Rui Hao, Jiahe Wang, Minghao Han, Yuxue Yang, Shuzhe Wu, Feiyang Pan, Lue Fan, Dandan Tu, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10975)  

**Abstract**: Agents powered by large language models (LLMs) are increasingly adopted in the software industry, contributing code as collaborators or even autonomous developers. As their presence grows, it becomes important to assess the current boundaries of their coding abilities. Existing agentic coding benchmarks, however, cover a limited task scope, e.g., bug fixing within a single pull request (PR), and often rely on non-executable evaluations or lack an automated approach for continually updating the evaluation coverage. To address such issues, we propose FeatureBench, a benchmark designed to evaluate agentic coding performance in end-to-end, feature-oriented software development. FeatureBench incorporates an execution-based evaluation protocol and a scalable test-driven method that automatically derives tasks from code repositories with minimal human effort. By tracing from unit tests along a dependency graph, our approach can identify feature-level coding tasks spanning multiple commits and PRs scattered across the development timeline, while ensuring the proper functioning of other features after the separation. Using this framework, we curated 200 challenging evaluation tasks and 3825 executable environments from 24 open-source repositories in the first version of our benchmark. Empirical evaluation reveals that the state-of-the-art agentic model, such as Claude 4.5 Opus, which achieves a 74.4% resolved rate on SWE-bench, succeeds on only 11.0% of tasks, opening new opportunities for advancing agentic coding. Moreover, benefiting from our automated task collection toolkit, FeatureBench can be easily scaled and updated over time to mitigate data leakage. The inherent verifiability of constructed environments also makes our method potentially valuable for agent training. 

---
# Healthy Harvests: A Comparative Look at Guava Disease Classification Using InceptionV3 

**Authors**: Samanta Ghosh, Shaila Afroz Anika, Umma Habiba Ahmed, B. M. Shahria Alam, Mohammad Tahmid Noor, Nishat Tasnim Niloy  

**Link**: [PDF](https://arxiv.org/pdf/2602.10967)  

**Abstract**: Guava fruits often suffer from many diseases. This can harm fruit quality and fruit crop yield. Early identification is important for minimizing damage and ensuring fruit health. This study focuses on 3 different categories for classifying diseases. These are Anthracnose, Fruit flies, and Healthy fruit. The data set used in this study is collected from Mendeley Data. This dataset contains 473 original images of Guava. These images vary in size and format. The original dataset was resized to 256x256 pixels with RGB color mode for better consistency. After this, the Data augmentation process is applied to improve the dataset by generating variations of the original images. The augmented dataset consists of 3784 images using advanced preprocessing techniques. Two deep learning models were implemented to classify the images. The InceptionV3 model is well known for its advanced framework. These apply multiple convolutional filters for obtaining different features effectively. On the other hand, the ResNet50 model helps to train deeper networks by using residual learning. The InceptionV3 model achieved the impressive accuracy of 98.15%, and ResNet50got 94.46% accuracy. Data mixing methods such as CutMix and MixUp were applied to enhance the model's robustness. The confusion matrix was used to evaluate the overall model performance of both InceptionV3 and Resnet50. Additionally, SHAP analysis is used to improve interpretability, which helps to find the significant parts of the image for the model prediction. This study purposes to highlight how advanced models enhan 

---
# Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers 

**Authors**: Feilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10959)  

**Abstract**: Rotary positional embeddings (RoPE) are widely used in large language models to encode token positions through multiplicative rotations, yet their behavior at long context lengths remains poorly characterized. In this work, we reinterpret RoPE as phase modulation applied to a bank of complex oscillators, enabling analysis through classical signal processing theory.
Under this formulation, we derive principled lower bounds on the RoPE base parameter that are necessary to preserve positional coherence over a target context length. These include a fundamental aliasing bound, analogous to a Nyquist limit, and a DC-component stability bound that constrains phase drift in low-frequency positional modes. We further extend this analysis to deep transformers, showing that repeated rotary modulation across layers compounds angular misalignment, tightening the base requirement as depth increases.
Complementing these results, we derive a precision-dependent upper bound on the RoPE base arising from finite floating-point resolution. Beyond this limit, incremental phase updates become numerically indistinguishable, leading to positional erasure even in the absence of aliasing. Together, the lower and upper bounds define a precision- and depth-dependent feasibility region a Goldilocks zone for long-context transformers.
We validate the framework through a comprehensive case study of state-of-the-art models, including LLaMA, Mistral, and DeepSeek variants, showing that observed successes, failures, and community retrofits align closely with the predicted bounds. Notably, models that violate the stability bound exhibit attention collapse and long-range degradation, while attempts to scale beyond one million tokens encounter a hard precision wall independent of architecture or training. 

---
# Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models 

**Authors**: Mingyu Cao, Alvaro Correia, Christos Louizos, Shiwei Liu, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.10953)  

**Abstract**: Diffusion Language Models (DLMs) generate text by iteratively denoising a masked sequence, repeatedly deciding which positions to commit at each step. Standard decoding follows a greedy rule: unmask the most confident positions, yet this local choice can lock the model into a suboptimal unmasking order, especially on reasoning-heavy prompts. We present SOAR, a training-free decoding algorithm that adapts its behavior to the model's uncertainty. When confidence is low, SOAR briefly widens the search over alternative unmasking decisions to avoid premature commitments; when confidence is high, it collapses the search and decodes many positions in parallel to reduce the number of denoising iterations. Across mathematical reasoning and code generation benchmarks (GSM8K, MBPP, HumanEval) on Dream-7B and LLaDA-8B, SOAR improves generation quality while maintaining competitive inference speed, offering a practical way to balance quality and efficiency in DLM decoding. 

---
# Computational Phenomenology of Temporal Experience in Autism: Quantifying the Emotional and Narrative Characteristics of Lived Unpredictability 

**Authors**: Kacper Dudzic, Karolina Drożdż, Maciej Wodziński, Anastazja Szuła, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2602.10947)  

**Abstract**: Disturbances in temporality, such as desynchronization with the social environment and its unpredictability, are considered core features of autism with a deep impact on relationships. However, limitations regarding research on this issue include: 1) the dominance of deficit-based medical models of autism, 2) sample size in qualitative research, and 3) the lack of phenomenological anchoring in computational research. To bridge the gap between phenomenological and computational approaches and overcome sample-size limitations, our research integrated three methodologies. Study A: structured phenomenological interviews with autistic individuals using the Transdiagnostic Assessment of Temporal Experience. Study B: computational analysis of an autobiographical corpus of autistic narratives built for this purpose. Study C: a replication of a computational study using narrative flow measures to assess the perceived phenomenological authenticity of autistic autobiographies. Interviews revealed that the most significant differences between the autistic and control groups concerned unpredictability of experience. Computational results mirrored these findings: the temporal lexicon in autistic narratives was significantly more negatively valenced - particularly the "Immediacy & Suddenness" category. Outlier analysis identified terms associated with perceived discontinuity (unpredictably, precipitously, and abruptly) as highly negative. The computational analysis of narrative flow found that the autistic narratives contained within the corpus quantifiably resemble autobiographical stories more than imaginary ones. Overall, the temporal challenges experienced by autistic individuals were shown to primarily concern lived unpredictability and stem from the contents of lived experience, and not from autistic narrative construction. 

---
# What do people want to fact-check? 

**Authors**: Bijean Ghafouri, Dorsaf Sallami, Luca Luceri, Taylor Lynn Curtis, Jean-Francois Godbout, Emilio Ferrara, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2602.10935)  

**Abstract**: Research on misinformation has focused almost exclusively on supply, asking what falsehoods circulate, who produces them, and whether corrections work. A basic demand-side question remains unanswered. When ordinary people can fact-check anything they want, what do they actually ask about? We provide the first large-scale evidence on this question by analyzing close to 2{,}500 statements submitted by 457 participants to an open-ended AI fact-checking system. Each claim is classified along five semantic dimensions (domain, epistemic form, verifiability, target entity, and temporal reference), producing a behavioral map of public verification demand. Three findings stand out. First, users range widely across topics but default to a narrow epistemic repertoire, overwhelmingly submitting simple descriptive claims about present-day observables. Second, roughly one in four requests concerns statements that cannot be empirically resolved, including moral judgments, speculative predictions, and subjective evaluations, revealing a systematic mismatch between what users seek from fact-checking tools and what such tools can deliver. Third, comparison with the FEVER benchmark dataset exposes sharp structural divergences across all five dimensions, indicating that standard evaluation corpora encode a synthetic claim environment that does not resemble real-world verification needs. These results reframe fact-checking as a demand-driven problem and identify where current AI systems and benchmarks are misaligned with the uncertainty people actually experience. 

---
# Traceable, Enforceable, and Compensable Participation: A Participation Ledger for People-Centered AI Governance 

**Authors**: Rashid Mushkani  

**Link**: [PDF](https://arxiv.org/pdf/2602.10916)  

**Abstract**: Participatory approaches are widely invoked in AI governance, yet participation rarely translates into durable influence. In public sector and civic AI systems, community contributions such as deliberations, annotations, prompts, and incident reports are often recorded informally, weakly linked to system updates, and disconnected from enforceable rights or sustained compensation. As a result, participation is frequently symbolic rather than accountable. We introduce the Participation Ledger, a machine readable and auditable framework that operationalizes participation as traceable influence, enforceable authority, and compensable labor. The ledger represents participation as an influence graph that links contributed artifacts to verified changes in AI systems, including datasets, prompts, adapters, policies, guardrails, and evaluation suites. It integrates three elements: a Participation Evidence Standard documenting consent, privacy, compensation, and reuse terms; an influence tracing mechanism that connects system updates to replayable before and after tests, enabling longitudinal monitoring of commitments; and encoded rights and incentives. Capability Vouchers allow authorized community stewards to request or constrain specific system capabilities within defined boundaries, while Participation Credits support ongoing recognition and compensation when contributed tests continue to provide value. We ground the framework in four urban AI and public space governance deployments and provide a machine readable schema, templates, and an evaluation plan for assessing traceability, enforceability, and compensation in practice. 

---
# Blind Gods and Broken Screens: Architecting a Secure, Intent-Centric Mobile Agent Operating System 

**Authors**: Zhenhua Zou, Sheng Guo, Qiuyang Zhan, Lepeng Zhao, Shuo Li, Qi Li, Ke Xu, Mingwei Xu, Zhuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10915)  

**Abstract**: The evolution of Large Language Models (LLMs) has shifted mobile computing from App-centric interactions to system-level autonomous agents. Current implementations predominantly rely on a "Screen-as-Interface" paradigm, which inherits structural vulnerabilities and conflicts with the mobile ecosystem's economic foundations. In this paper, we conduct a systematic security analysis of state-of-the-art mobile agents using Doubao Mobile Assistant as a representative case. We decompose the threat landscape into four dimensions - Agent Identity, External Interface, Internal Reasoning, and Action Execution - revealing critical flaws such as fake App identity, visual spoofing, indirect prompt injection, and unauthorized privilege escalation stemming from a reliance on unstructured visual data.
To address these challenges, we propose Aura, an Agent Universal Runtime Architecture for a clean-slate secure agent OS. Aura replaces brittle GUI scraping with a structured, agent-native interaction model. It adopts a Hub-and-Spoke topology where a privileged System Agent orchestrates intent, sandboxed App Agents execute domain-specific tasks, and the Agent Kernel mediates all communication. The Agent Kernel enforces four defense pillars: (i) cryptographic identity binding via a Global Agent Registry; (ii) semantic input sanitization through a multilayer Semantic Firewall; (iii) cognitive integrity via taint-aware memory and plan-trajectory alignment; and (iv) granular access control with non-deniable auditing. Evaluation on MobileSafetyBench shows that, compared to Doubao, Aura improves low-risk Task Success Rate from roughly 75% to 94.3%, reduces high-risk Attack Success Rate from roughly 40% to 4.4%, and achieves near-order-of-magnitude latency gains. These results demonstrate Aura as a viable, secure alternative to the "Screen-as-Interface" paradigm. 

---
# Resource-Efficient Model-Free Reinforcement Learning for Board Games 

**Authors**: Kazuki Ota, Takayuki Osa, Motoki Omura, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2602.10894)  

**Abstract**: Board games have long served as complex decision-making benchmarks in artificial intelligence. In this field, search-based reinforcement learning methods such as AlphaZero have achieved remarkable success. However, their significant computational demands have been pointed out as barriers to their reproducibility. In this study, we propose a model-free reinforcement learning algorithm designed for board games to achieve more efficient learning. To validate the efficiency of the proposed method, we conducted comprehensive experiments on five board games: Animal Shogi, Gardner Chess, Go, Hex, and Othello. The results demonstrate that the proposed method achieves more efficient learning than existing methods across these environments. In addition, our extensive ablation study shows the importance of core techniques used in the proposed method. We believe that our efficient algorithm shows the potential of model-free reinforcement learning in domains traditionally dominated by search-based methods. 

---
# Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search 

**Authors**: Berfin Sakallioglu, Giorgia Nadizar, Eric Medvet  

**Link**: [PDF](https://arxiv.org/pdf/2602.10891)  

**Abstract**: Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications. 

---
# The CLEF-2026 FinMMEval Lab: Multilingual and Multimodal Evaluation of Financial AI Systems 

**Authors**: Zhuohan Xie, Rania Elbadry, Fan Zhang, Georgi Georgiev, Xueqing Peng, Lingfei Qian, Jimin Huang, Dimitar Dimitrov, Vanshikaa Jani, Yuyang Dai, Jiahui Geng, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2602.10886)  

**Abstract**: We present the setup and the tasks of the FinMMEval Lab at CLEF 2026, which introduces the first multilingual and multimodal evaluation framework for financial Large Language Models (LLMs). While recent advances in financial natural language processing have enabled automated analysis of market reports, regulatory documents, and investor communications, existing benchmarks remain largely monolingual, text-only, and limited to narrow subtasks. FinMMEval 2026 addresses this gap by offering three interconnected tasks that span financial understanding, reasoning, and decision-making: Financial Exam Question Answering, Multilingual Financial Question Answering (PolyFiQA), and Financial Decision Making. Together, these tasks provide a comprehensive evaluation suite that measures models' ability to reason, generalize, and act across diverse languages and modalities. The lab aims to promote the development of robust, transparent, and globally inclusive financial AI systems, with datasets and evaluation resources publicly released to support reproducible research. 

---
# Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2602.10881)  

**Abstract**: Systematic reviews and meta-analyses rely on converting narrative articles into structured, numerically grounded study records. Despite rapid advances in large language models (LLMs), it remains unclear whether they can meet the structural requirements of this process, which hinge on preserving roles, methods, and effect-size attribution across documents rather than on recognizing isolated entities. We propose a structural, diagnostic framework that evaluates LLM-based evidence extraction as a progression of schema-constrained queries with increasing relational and numerical complexity, enabling precise identification of failure points beyond atom-level extraction. Using a manually curated corpus spanning five scientific domains, together with a unified query suite and evaluation protocol, we evaluate two state-of-the-art LLMs under both per-document and long-context, multi-document input regimes. Across domains and models, performance remains moderate for single-property queries but degrades sharply once tasks require stable binding between variables, roles, statistical methods, and effect sizes. Full meta-analytic association tuples are extracted with near-zero reliability, and long-context inputs further exacerbate these failures. Downstream aggregation amplifies even minor upstream errors, rendering corpus-level statistics unreliable. Our analysis shows that these limitations stem not from entity recognition errors, but from systematic structural breakdowns, including role reversals, cross-analysis binding drift, instance compression in dense result sections, and numeric misattribution, indicating that current LLMs lack the structural fidelity, relational binding, and numerical grounding required for automated meta-analysis. The code and data are publicly available at GitHub (this https URL). 

---
# FedPS: Federated data Preprocessing via aggregated Statistics 

**Authors**: Xuefeng Xu, Graham Cormode  

**Link**: [PDF](https://arxiv.org/pdf/2602.10870)  

**Abstract**: Federated Learning (FL) enables multiple parties to collaboratively train machine learning models without sharing raw data. However, before training, data must be preprocessed to address missing values, inconsistent formats, and heterogeneous feature scales. This preprocessing stage is critical for model performance but is largely overlooked in FL research. In practical FL systems, privacy constraints prohibit centralizing raw data, while communication efficiency introduces further challenges for distributed preprocessing. We introduce FedPS, a unified framework for federated data preprocessing based on aggregated statistics. FedPS leverages data-sketching techniques to efficiently summarize local datasets while preserving essential statistical information. Building on these summaries, we design federated algorithms for feature scaling, encoding, discretization, and missing-value imputation, and extend preprocessing-related models such as k-Means, k-Nearest Neighbors, and Bayesian Linear Regression to both horizontal and vertical FL settings. FedPS provides flexible, communication-efficient, and consistent preprocessing pipelines for practical FL deployments. 

---
# ICA: Information-Aware Credit Assignment for Visually Grounded Long-Horizon Information-Seeking Agents 

**Authors**: Cong Pang, Xuyu Feng, Yujie Yi, Zixuan Chen, Jiawei Hong, Tiankuo Yao, Nang Yuan, Jiapeng Luo, Lewei Lu, Xin Lou  

**Link**: [PDF](https://arxiv.org/pdf/2602.10863)  

**Abstract**: Despite the strong performance achieved by reinforcement learning-trained information-seeking agents, learning in open-ended web environments remains severely constrained by low signal-to-noise feedback. Text-based parsers often discard layout semantics and introduce unstructured noise, while long-horizon training typically relies on sparse outcome rewards that obscure which retrieval actions actually matter. We propose a visual-native search framework that represents webpages as visual snapshots, allowing agents to leverage layout cues to quickly localize salient evidence and suppress distractors. To learn effectively from these high-dimensional observations, we introduce Information-Aware Credit Assignment (ICA), a post-hoc method that estimates each retrieved snapshot's contribution to the final outcome via posterior analysis and propagates dense learning signals back to key search turns. Integrated with a GRPO-based training pipeline, our approach consistently outperforms text-based baselines on diverse information-seeking benchmarks, providing evidence that visual snapshot grounding with information-level credit assignment alleviates the credit-assignment bottleneck in open-ended web environments. The code and datasets will be released in this https URL. 

---
# Time Series Foundation Models for Energy Load Forecasting on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark 

**Authors**: Luigi Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2602.10848)  

**Abstract**: Time Series Foundation Models (TSFMs) have introduced zero-shot prediction capabilities that bypass the need for task-specific training. Whether these capabilities translate to mission-critical applications such as electricity demand forecasting--where accuracy, calibration, and robustness directly affect grid operations--remains an open question. We present a multi-dimensional benchmark evaluating four TSFMs (Chronos-Bolt, Chronos-2, Moirai-2, and TinyTimeMixer) alongside Prophet as an industry-standard baseline and two statistical references (SARIMA and Seasonal Naive), using ERCOT hourly load data from 2020 to 2024. All experiments run on consumer-grade hardware (AMD Ryzen 7, 16GB RAM, no GPU). The evaluation spans four axes: (1) context length sensitivity from 24 to 2048 hours, (2) probabilistic forecast calibration, (3) robustness under distribution shifts including COVID-19 lockdowns and Winter Storm Uri, and (4) prescriptive analytics for operational decision support. The top-performing foundation models achieve MASE values near 0.31 at long context lengths (C = 2048h, day-ahead horizon), a 47% reduction over the Seasonal Naive baseline. The inclusion of Prophet exposes a structural advantage of pre-trained models: Prophet fails when the fitting window is shorter than its seasonality period (MASE > 74 at 24-hour context), while TSFMs maintain stable accuracy even with minimal context because they recognise temporal patterns learned during pre-training rather than estimating them from scratch. Calibration varies substantially across models--Chronos-2 produces well-calibrated prediction intervals (95% empirical coverage at 90% nominal level) while both Moirai-2 and Prophet exhibit overconfidence (~70% coverage). We provide practical model selection guidelines and release the complete benchmark framework for reproducibility. 

---
# Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval 

**Authors**: Fanpu Cao, Lu Dai, Jindong Han, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.10847)  

**Abstract**: Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon - despite such patterns carrying strong predictive signals. Naive solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model's temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks. Code is available at this repository: this https URL. 

---
# Flow caching for autoregressive video generation 

**Authors**: Yuexiao Ma, Xuzhe Zheng, Jing Xu, Xiwei Xu, Feng Ling, Xiawu Zheng, Huafeng Kuang, Huixia Li, Xing Wang, Xuefeng Xiao, Fei Chao, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.10825)  

**Abstract**: Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at this https URL. 

---
# Beyond Confidence: The Rhythms of Reasoning in Generative Models 

**Authors**: Deyuan Liu, Zecheng Wang, Zhanyue Qin, Zhiying Tu, Dianhui Chu, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2602.10816)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities yet suffer from sensitivity to slight input context variations, hampering reliability. Conventional metrics like accuracy and perplexity fail to assess local prediction robustness, as normalized output probabilities can obscure the underlying resilience of an LLM's internal state to perturbations. We introduce the Token Constraint Bound ($\delta_{\mathrm{TCB}}$), a novel metric that quantifies the maximum internal state perturbation an LLM can withstand before its dominant next-token prediction significantly changes. Intrinsically linked to output embedding space geometry, $\delta_{\mathrm{TCB}}$ provides insights into the stability of the model's internal predictive commitment. Our experiments show $\delta_{\mathrm{TCB}}$ correlates with effective prompt engineering and uncovers critical prediction instabilities missed by perplexity during in-context learning and text generation. $\delta_{\mathrm{TCB}}$ offers a principled, complementary approach to analyze and potentially improve the contextual stability of LLM predictions. 

---
# PELLI: Framework to effectively integrate LLMs for quality software generation 

**Authors**: Rasmus Krebs, Somnath Mazumdar  

**Link**: [PDF](https://arxiv.org/pdf/2602.10808)  

**Abstract**: Recent studies have revealed that when LLMs are appropriately prompted and configured, they demonstrate mixed results. Such results often meet or exceed the baseline performance. However, these comparisons have two primary issues. First, they mostly considered only reliability as a comparison metric and selected a few LLMs (such as Codex and ChatGPT) for comparision. This paper proposes a comprehensive code quality assessment framework called Programmatic Excellence via LLM Iteration (PELLI). PELLI is an iterative analysis-based process that upholds high-quality code changes. We extended the state-of-the-art by performing a comprehensive evaluation that generates quantitative metrics for analyzing three primary nonfunctional requirements (such as maintainability, performance, and reliability) while selecting five popular LLMs. For PELLI's applicability, we selected three application domains while following Python coding standards. Following this framework, practitioners can ensure harmonious integration between LLMs and human developers, ensuring that their potential is fully realized. PELLI can serve as a practical guide for developers aiming to leverage LLMs while adhering to recognized quality standards. This study's outcomes are crucial for advancing LLM technologies in real-world applications, providing stakeholders with a clear understanding of where these LLMs excel and where they require further refinement. Overall, based on three nonfunctional requirements, we have found that GPT-4T and Gemini performed slightly better. We also found that prompt design can influence the overall code quality. In addition, each application domain demonstrated high and low scores across various metrics, and even within the same metrics across different prompts. 

---
# RSHallu: Dual-Mode Hallucination Evaluation for Remote-Sensing Multimodal Large Language Models with Domain-Tailored Mitigation 

**Authors**: Zihui Zhou, Yong Feng, Yanying Chen, Guofan Duan, Zhenxi Song, Mingliang Zhou, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.10799)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly adopted in remote sensing (RS) and have shown strong performance on tasks such as RS visual grounding (RSVG), RS visual question answering (RSVQA), and multimodal dialogue. However, hallucinations, which are responses inconsistent with the input RS images, severely hinder their deployment in high-stakes scenarios (e.g., emergency management and agricultural monitoring) and remain under-explored in RS. In this work, we present RSHallu, a systematic study with three deliverables: (1) we formalize RS hallucinations with an RS-oriented taxonomy and introduce image-level hallucination to capture RS-specific inconsistencies beyond object-centric errors (e.g., modality, resolution, and scene-level semantics); (2) we build a hallucination benchmark RSHalluEval (2,023 QA pairs) and enable dual-mode checking, supporting high-precision cloud auditing and low-cost reproducible local checking via a compact checker fine-tuned on RSHalluCheck dataset (15,396 QA pairs); and (3) we introduce a domain-tailored dataset RSHalluShield (30k QA pairs) for training-friendly mitigation and further propose training-free plug-and-play strategies, including decoding-time logit correction and RS-aware prompting. Across representative RS-MLLMs, our mitigation improves the hallucination-free rate by up to 21.63 percentage points under a unified protocol, while maintaining competitive performance on downstream RS tasks (RSVQA/RSVG). Code and datasets will be released. 

---
# Transport, Don't Generate: Deterministic Geometric Flows for Combinatorial Optimization 

**Authors**: Benjy Friedmann, Nadav Dym  

**Link**: [PDF](https://arxiv.org/pdf/2602.10794)  

**Abstract**: Recent advances in Neural Combinatorial Optimization (NCO) have been dominated by diffusion models that treat the Euclidean Traveling Salesman Problem (TSP) as a stochastic $N \times N$ heatmap generation task. In this paper, we propose CycFlow, a framework that replaces iterative edge denoising with deterministic point transport. CycFlow learns an instance-conditioned vector field that continuously transports input 2D coordinates to a canonical circular arrangement, where the optimal tour is recovered from this $2N$ dimensional representation via angular sorting. By leveraging data-dependent flow matching, we bypass the quadratic bottleneck of edge scoring in favor of linear coordinate dynamics. This paradigm shift accelerates solving speed by up to three orders of magnitude compared to state-of-the-art diffusion baselines, while maintaining competitive optimality gaps. 

---
# VulReaD: Knowledge-Graph-guided Software Vulnerability Reasoning and Detection 

**Authors**: Samal Mukhtar, Yinghua Yao, Zhu Sun, Mustafa Mustafa, Yew Soon Ong, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.10787)  

**Abstract**: Software vulnerability detection (SVD) is a critical challenge in modern systems. Large language models (LLMs) offer natural-language explanations alongside predictions, but most work focuses on binary evaluation, and explanations often lack semantic consistency with Common Weakness Enumeration (CWE) categories. We propose VulReaD, a knowledge-graph-guided approach for vulnerability reasoning and detection that moves beyond binary classification toward CWE-level reasoning. VulReaD leverages a security knowledge graph (KG) as a semantic backbone and uses a strong teacher LLM to generate CWE-consistent contrastive reasoning supervision, enabling student model training without manual annotations. Students are fine-tuned with Odds Ratio Preference Optimization (ORPO) to encourage taxonomy-aligned reasoning while suppressing unsupported explanations. Across three real-world datasets, VulReaD improves binary F1 by 8-10% and multi-class classification by 30% Macro-F1 and 18% Micro-F1 compared to state-of-the-art baselines. Results show that LLMs outperform deep learning baselines in binary detection and that KG-guided reasoning enhances CWE coverage and interpretability. 

---
# Kill it with FIRE: On Leveraging Latent Space Directions for Runtime Backdoor Mitigation in Deep Neural Networks 

**Authors**: Enrico Ahlers, Daniel Passon, Yannic Noller, Lars Grunske  

**Link**: [PDF](https://arxiv.org/pdf/2602.10780)  

**Abstract**: Machine learning models are increasingly present in our everyday lives; as a result, they become targets of adversarial attackers seeking to manipulate the systems we interact with. A well-known vulnerability is a backdoor introduced into a neural network by poisoned training data or a malicious training process. Backdoors can be used to induce unwanted behavior by including a certain trigger in the input. Existing mitigations filter training data, modify the model, or perform expensive input modifications on samples. If a vulnerable model has already been deployed, however, those strategies are either ineffective or inefficient. To address this gap, we propose our inference-time backdoor mitigation approach called FIRE (Feature-space Inference-time REpair). We hypothesize that a trigger induces structured and repeatable changes in the model's internal representation. We view the trigger as directions in the latent spaces between layers that can be applied in reverse to correct the inference mechanism. Therefore, we turn the backdoored model against itself by manipulating its latent representations and moving a poisoned sample's features along the backdoor directions to neutralize the trigger. Our evaluation shows that FIRE has low computational overhead and outperforms current runtime mitigations on image benchmarks across various attacks, datasets, and network architectures. 

---
# LOREN: Low Rank-Based Code-Rate Adaptation in Neural Receivers 

**Authors**: Bram Van Bolderik, Vlado Menkovski, Sonia Heemstra de Groot, Manil Dev Gomony  

**Link**: [PDF](https://arxiv.org/pdf/2602.10770)  

**Abstract**: Neural network based receivers have recently demonstrated superior system-level performance compared to traditional receivers. However, their practicality is limited by high memory and power requirements, as separate weight sets must be stored for each code rate. To address this challenge, we propose LOREN, a Low Rank-Based Code-Rate Adaptation Neural Receiver that achieves adaptability with minimal overhead. LOREN integrates lightweight low rank adaptation adapters (LOREN adapters) into convolutional layers, freezing a shared base network while training only small adapters per code rate. An end-to-end training framework over 3GPP CDL channels ensures robustness across realistic wireless environments. LOREN achieves comparable or superior performance relative to fully retrained base neural receivers. The hardware implementation of LOREN in 22nm technology shows more than 65% savings in silicon area and up to 15% power reduction when supporting three code rates. 

---
# Exploring the impact of adaptive rewiring in Graph Neural Networks 

**Authors**: Charlotte Cambier van Nooten, Christos Aronis, Yuliya Shapovalova, Lucia Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2602.10754)  

**Abstract**: This paper explores sparsification methods as a form of regularization in Graph Neural Networks (GNNs) to address high memory usage and computational costs in large-scale graph applications. Using techniques from Network Science and Machine Learning, including Erdős-Rényi for model sparsification, we enhance the efficiency of GNNs for real-world applications. We demonstrate our approach on N-1 contingency assessment in electrical grids, a critical task for ensuring grid reliability. We apply our methods to three datasets of varying sizes, exploring Graph Convolutional Networks (GCN) and Graph Isomorphism Networks (GIN) with different degrees of sparsification and rewiring. Comparison across sparsification levels shows the potential of combining insights from both research fields to improve GNN performance and scalability. Our experiments highlight the importance of tuning sparsity parameters: while sparsity can improve generalization, excessive sparsity may hinder learning of complex patterns. Our adaptive rewiring approach, particularly when combined with early stopping, proves promising by allowing the model to adapt its connectivity structure during training. This research contributes to understanding how sparsity can be effectively leveraged in GNNs for critical applications like power grid reliability analysis. 

---
# SecureScan: An AI-Driven Multi-Layer Framework for Malware and Phishing Detection Using Logistic Regression and Threat Intelligence Integration 

**Authors**: Rumman Firdos, Aman Dangi  

**Link**: [PDF](https://arxiv.org/pdf/2602.10750)  

**Abstract**: The growing sophistication of modern malware and phishing campaigns has diminished the effectiveness of traditional signature-based intrusion detection systems. This work presents SecureScan, an AI-driven, triple-layer detection framework that integrates logistic regression-based classification, heuristic analysis, and external threat intelligence via the VirusTotal API for comprehensive triage of URLs, file hashes, and binaries. The proposed architecture prioritizes efficiency by filtering known threats through heuristics, classifying uncertain samples using machine learning, and validating borderline cases with third-party intelligence. On benchmark datasets, SecureScan achieves 93.1 percent accuracy with balanced precision (0.87) and recall (0.92), demonstrating strong generalization and reduced overfitting through threshold-based decision calibration. A calibrated threshold and gray-zone logic (0.45-0.55) were introduced to minimize false positives and enhance real-world stability. Experimental results indicate that a lightweight statistical model, when augmented with calibrated verification and external intelligence, can achieve reliability and performance comparable to more complex deep learning systems. 

---
# Self-Supervised Image Super-Resolution Quality Assessment based on Content-Free Multi-Model Oriented Representation Learning 

**Authors**: Kian Majlessi, Amir Masoud Soltani, Mohammad Ebrahim Mahdavi, Aurelien Gourrier, Peyman Adibi  

**Link**: [PDF](https://arxiv.org/pdf/2602.10744)  

**Abstract**: Super-resolution (SR) applied to real-world low-resolution (LR) images often results in complex, irregular degradations that stem from the inherent complexity of natural scene acquisition. In contrast to SR artifacts arising from synthetic LR images created under well-defined scenarios, those distortions are highly unpredictable and vary significantly across different real-life contexts. Consequently, assessing the quality of SR images (SR-IQA) obtained from realistic LR, remains a challenging and underexplored problem. In this work, we introduce a no-reference SR-IQA approach tailored for such highly ill-posed realistic settings. The proposed method enables domain-adaptive IQA for real-world SR applications, particularly in data-scarce domains. We hypothesize that degradations in super-resolved images are strongly dependent on the underlying SR algorithms, rather than being solely determined by image content. To this end, we introduce a self-supervised learning (SSL) strategy that first pretrains multiple SR model oriented representations in a pretext stage. Our contrastive learning framework forms positive pairs from images produced by the same SR model and negative pairs from those generated by different methods, independent of image content. The proposed approach S3 RIQA, further incorporates targeted preprocessing to extract complementary quality information and an auxiliary task to better handle the various degradation profiles associated with different SR scaling factors. To this end, we constructed a new dataset, SRMORSS, to support unsupervised pretext training; it includes a wide range of SR algorithms applied to numerous real LR images, which addresses a gap in existing datasets. Experiments on real SR-IQA benchmarks demonstrate that S3 RIQA consistently outperforms most state-of-the-art relevant metrics. 

---
# Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity 

**Authors**: Hugo L. Hammer, Vajira Thambawita, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10735)  

**Abstract**: A narrated e-book combines synchronized audio with digital text, highlighting the currently spoken word or sentence during playback. This format supports early literacy and assists individuals with reading challenges, while also allowing general readers to seamlessly switch between reading and listening. With the emergence of natural-sounding neural Text-to-Speech (TTS) technology, several commercial services have been developed to leverage these technology for converting standard text e-books into high-quality narrated e-books. However, no open-source solutions currently exist to perform this task. In this paper, we present Calliope, an open-source framework designed to fill this gap. Our method leverages state-of-the-art open-source TTS to convert a text e-book into a narrated e-book in the EPUB 3 Media Overlay format. The method offers several innovative steps: audio timestamps are captured directly during TTS, ensuring exact synchronization between narration and text highlighting; the publisher's original typography, styling, and embedded media are strictly preserved; and the entire pipeline operates offline. This offline capability eliminates recurring API costs, mitigates privacy concerns, and avoids copyright compliance issues associated with cloud-based services. The framework currently supports the state-of-the-art open-source TTS systems XTTS-v2 and Chatterbox. A potential alternative approach involves first generating narration via TTS and subsequently synchronizing it with the text using forced alignment. However, while our method ensures exact synchronization, our experiments show that forced alignment introduces drift between the audio and text highlighting significant enough to degrade the reading experience. Source code and usage instructions are available at this https URL. 

---
# A Diffusion-Based Generative Prior Approach to Sparse-view Computed Tomography 

**Authors**: Davide Evangelista, Pasquale Cascarano, Elena Loli Piccolomini  

**Link**: [PDF](https://arxiv.org/pdf/2602.10722)  

**Abstract**: The reconstruction of X-rays CT images from sparse or limited-angle geometries is a highly challenging task. The lack of data typically results in artifacts in the reconstructed image and may even lead to object distortions. For this reason, the use of deep generative models in this context has great interest and potential success. In the Deep Generative Prior (DGP) framework, the use of diffusion-based generative models is combined with an iterative optimization algorithm for the reconstruction of CT images from sinograms acquired under sparse geometries, to maintain the explainability of a model-based approach while introducing the generative power of a neural network. There are therefore several aspects that can be further investigated within these frameworks to improve reconstruction quality, such as image generation, the model, and the iterative algorithm used to solve the minimization problem, for which we propose modifications with respect to existing approaches. The results obtained even under highly sparse geometries are very promising, although further research is clearly needed in this direction. 

---
# Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents 

**Authors**: Yifei Li, Weidong Guo, Lingling Zhang, Rongman Xu, Muye Huang, Hui Liu, Lijiao Xu, Yu Xu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10715)  

**Abstract**: Long-term conversational memory is a core capability for LLM-based dialogue systems, yet existing benchmarks and evaluation protocols primarily focus on surface-level factual recall. In realistic interactions, appropriate responses often depend on implicit constraints such as user state, goals, or values that are not explicitly queried later. To evaluate this setting, we introduce \textbf{LoCoMo-Plus}, a benchmark for assessing cognitive memory under cue--trigger semantic disconnect, where models must retain and apply latent constraints across long conversational contexts. We further show that conventional string-matching metrics and explicit task-type prompting are misaligned with such scenarios, and propose a unified evaluation framework based on constraint consistency. Experiments across diverse backbone models, retrieval-based methods, and memory systems demonstrate that cognitive memory remains challenging and reveals failures not captured by existing benchmarks. Our code and evaluation framework are publicly available at: this https URL. 

---
# Cross-Sectional Asset Retrieval via Future-Aligned Soft Contrastive Learning 

**Authors**: Hyeongmin Lee, Chanyeol Choi, Jihoon Kwon, Yoon Kim, Alejandro Lopez-Lira, Wonbin Ahn, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.10711)  

**Abstract**: Asset retrieval--finding similar assets in a financial universe--is central to quantitative investment decision-making. Existing approaches define similarity through historical price patterns or sector classifications, but such backward-looking criteria provide no guarantee about future behavior. We argue that effective asset retrieval should be future-aligned: the retrieved assets should be those most likely to exhibit correlated future returns. To this end, we propose Future-Aligned Soft Contrastive Learning (FASCL), a representation learning framework whose soft contrastive loss uses pairwise future return correlations as continuous supervision targets. We further introduce an evaluation protocol designed to directly assess whether retrieved assets share similar future trajectories. Experiments on 4,229 US equities demonstrate that FASCL consistently outperforms 13 baselines across all future-behavior metrics. The source code will be available soon. 

---
# Interpretable Graph-Level Anomaly Detection via Contrast with Normal Prototypes 

**Authors**: Qiuran Zhao, Kai Ming Ting, Xinpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10708)  

**Abstract**: The task of graph-level anomaly detection (GLAD) is to identify anomalous graphs that deviate significantly from the majority of graphs in a dataset. While deep GLAD methods have shown promising performance, their black-box nature limits their reliability and deployment in real-world applications. Although some recent methods have made attempts to provide explanations for anomaly detection results, they either provide explanations without referencing normal graphs, or rely on abstract latent vectors as prototypes rather than concrete graphs from the dataset. To address these limitations, we propose Prototype-based Graph-Level Anomaly Detection (ProtoGLAD), an interpretable unsupervised framework that provides explanation for each detected anomaly by explicitly contrasting with its nearest normal prototype graph. It employs a point-set kernel to iteratively discover multiple normal prototype graphs and their associated clusters from the dataset, then identifying graphs distant from all discovered normal clusters as anomalies. Extensive experiments on multiple real-world datasets demonstrate that ProtoGLAD achieves competitive anomaly detection performance compared to state-of-the-art GLAD methods while providing better human-interpretable prototype-based explanations. 

---
# AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models 

**Authors**: Zhifeng Rao, Wenlong Chen, Lei Xie, Xia Hua, Dongfu Yin, Zhen Tian, F. Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10698)  

**Abstract**: Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic perception and control, yet most existing approaches primarily rely on VLM trained using 2D images, which limits their spatial understanding and action grounding in complex 3D environments. To address this limitation, we propose a novel framework that integrates depth estimation into VLA models to enrich 3D feature representations. Specifically, we employ a depth estimation baseline called VGGT to extract geometry-aware 3D cues from standard RGB inputs, enabling efficient utilization of existing large-scale 2D datasets while implicitly recovering 3D structural information. To further enhance the reliability of these depth-derived features, we introduce a new module called action assistant, which constrains the learned 3D representations with action priors and ensures their consistency with downstream control tasks. By fusing the enhanced 3D features with conventional 2D visual tokens, our approach significantly improves the generalization ability and robustness of VLA models. Experimental results demonstrate that the proposed method not only strengthens perception in geometrically ambiguous scenarios but also leads to superior action prediction accuracy. This work highlights the potential of depth-driven data augmentation and auxiliary expert supervision for bridging the gap between 2D observations and 3D-aware decision-making in robotic systems. 

---
# VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training 

**Authors**: Guobin Shen, Chenxiao Zhao, Xiang Cheng, Lei Huang, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10693)  

**Abstract**: Training stability remains a central challenge in reinforcement learning (RL) for large language models (LLMs). Policy staleness, asynchronous training, and mismatches between training and inference engines all cause the behavior policy to diverge from the current policy, risking training collapse. Importance sampling provides a principled correction for this distribution shift but suffers from high variance; existing remedies such as token-level clipping and sequence-level normalization lack a unified theoretical foundation. We propose Variational sEquence-level Soft Policy Optimization (VESPO). By incorporating variance reduction into a variational formulation over proposal distributions, VESPO derives a closed-form reshaping kernel that operates directly on sequence-level importance weights without length normalization. Experiments on mathematical reasoning benchmarks show that VESPO maintains stable training under staleness ratios up to 64x and fully asynchronous execution, and delivers consistent gains across both dense and Mixture-of-Experts models. Code is available at this https URL 

---
# OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL 

**Authors**: Jinjie Shen, Jing Wu, Yaxiong Wang, Lechao Cheng, Shengeng Tang, Tianrui Hui, Nan Pu, Zhun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.10687)  

**Abstract**: Existing forgery detection methods are often limited to uni-modal or bi-modal settings, failing to handle the interleaved text, images, and videos prevalent in real-world misinformation. To bridge this gap, this paper targets to develop a unified framework for omnibus vision-language forgery detection and grounding. In this unified setting, the {interplay} between diverse modalities and the dual requirements of simultaneous detection and localization pose a critical ``difficulty bias`` problem: the simpler veracity classification task tends to dominate the gradients, leading to suboptimal performance in fine-grained grounding during multi-task optimization. To address this challenge, we propose \textbf{OmniVL-Guard}, a balanced reinforcement learning framework for omnibus vision-language forgery detection and grounding. Particularly, OmniVL-Guard comprises two core designs: Self-Evolving CoT Generatio and Adaptive Reward Scaling Policy Optimization (ARSPO). {Self-Evolving CoT Generation} synthesizes high-quality reasoning paths, effectively overcoming the cold-start challenge. Building upon this, {Adaptive Reward Scaling Policy Optimization (ARSPO)} dynamically modulates reward scales and task weights, ensuring a balanced joint optimization. Extensive experiments demonstrate that OmniVL-Guard significantly outperforms state-of-the-art methods and exhibits zero-shot robust generalization across out-of-domain scenarios. 

---
# TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning 

**Authors**: Junhua Liu, Zhangcheng Wang, Zhike Han, Ningli Wang, Guotao Liang, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10675)  

**Abstract**: Visual Chain-of-Thought (VCoT) has emerged as a promising paradigm for enhancing multimodal reasoning by integrating visual perception into intermediate reasoning steps. However, existing VCoT approaches are largely confined to static scenarios and struggle to capture the temporal dynamics essential for tasks such as instruction, prediction, and camera motion. To bridge this gap, we propose TwiFF-2.7M, the first large-scale, temporally grounded VCoT dataset derived from $2.7$ million video clips, explicitly designed for dynamic visual question and answer. Accompanying this, we introduce TwiFF-Bench, a high-quality evaluation benchmark of $1,078$ samples that assesses both the plausibility of reasoning trajectories and the correctness of final answers in open-ended dynamic settings. Building on these foundations, we propose the TwiFF model, a unified modal that synergistically leverages pre-trained video generation and image comprehension capabilities to produce temporally coherent visual reasoning cues-iteratively generating future action frames and textual reasoning. Extensive experiments demonstrate that TwiFF significantly outperforms existing VCoT methods and Textual Chain-of-Thought baselines on dynamic reasoning tasks, which fully validates the effectiveness for visual question answering in dynamic scenarios. Our code and data is available at this https URL. 

---
# The Neurosymbolic Frontier of Nonuniform Ellipticity: Formalizing Sharp Schauder Theory via Topos-Theoretic Reasoning Models 

**Authors**: Suyash Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2602.10632)  

**Abstract**: This white paper presents a critical synthesis of the recent breakthrough in nonuniformly elliptic regularity theory and the burgeoning field of neurosymbolic large reasoning models (LRMs). We explore the resolution of the long-standing sharp growth rate conjecture in Schauder theory, achieved by Cristiana De Filippis and Giuseppe Mingione, which identifies the exact threshold $q/p < 1 + \alpha/n$ for gradient Hölder continuity. Central to this mathematical achievement is the ``ghost equation'' methodology, a sophisticated auxiliary derivation that bypasses the non-differentiability of classical Euler-Lagrange systems. We propose that the next era of mathematical discovery lies in the integration of these pure analytical constructs with LRMs grounded in topos theory and formal verification frameworks such as Safe and Typed Chain-of-Thought (PC-CoT). By modeling the reasoning process as a categorical colimit in a slice topos, we demonstrate how LRMs can autonomously navigate the ``Dark Side'' of the calculus of variations, providing machine-checkable proofs for regularity bounds in complex, multi-phase physical systems. 

---
# A Vision-Language Foundation Model for Zero-shot Clinical Collaboration and Automated Concept Discovery in Dermatology 

**Authors**: Siyuan Yan, Xieji Li, Dan Mo, Philipp Tschandl, Yiwen Jiang, Zhonghua Wang, Ming Hu, Lie Ju, Cristina Vico-Alonso, Yizhen Zheng, Jiahe Liu, Juexiao Zhou, Camilla Chello, Jen G. Cheung, Julien Anriot, Luc Thomas, Clare Primiero, Gin Tan, Aik Beng Ng, Simon See, Xiaoying Tang, Albert Ip, Xiaoyang Liao, Adrian Bowling, Martin Haskett, Shuang Zhao, Monika Janda, H. Peter Soyer, Victoria Mar, Harald Kittler, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2602.10624)  

**Abstract**: Medical foundation models have shown promise in controlled benchmarks, yet widespread deployment remains hindered by reliance on task-specific fine-tuning. Here, we introduce DermFM-Zero, a dermatology vision-language foundation model trained via masked latent modelling and contrastive learning on over 4 million multimodal data points. We evaluated DermFM-Zero across 20 benchmarks spanning zero-shot diagnosis and multimodal retrieval, achieving state-of-the-art performance without task-specific adaptation. We further evaluated its zero-shot capabilities in three multinational reader studies involving over 1,100 clinicians. In primary care settings, AI assistance enabled general practitioners to nearly double their differential diagnostic accuracy across 98 skin conditions. In specialist settings, the model significantly outperformed board-certified dermatologists in multimodal skin cancer assessment. In collaborative workflows, AI assistance enabled non-experts to surpass unassisted experts while improving management appropriateness. Finally, we show that DermFM-Zero's latent representations are interpretable: sparse autoencoders unsupervisedly disentangle clinically meaningful concepts that outperform predefined-vocabulary approaches and enable targeted suppression of artifact-induced biases, enhancing robustness without retraining. These findings demonstrate that a foundation model can provide effective, safe, and transparent zero-shot clinical decision support. 

---
# Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling 

**Authors**: Zhibin Duan, Guowei Rong, Zhuo Li, Bo Chen, Mingyuan Zhou, Dandan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.10623)  

**Abstract**: Reward models learned from human preferences are central to aligning large language models (LLMs) via reinforcement learning from human feedback, yet they are often vulnerable to reward hacking due to noisy annotations and systematic biases such as response length or style. We propose Bayesian Non-Negative Reward Model (BNRM), a principled reward modeling framework that integrates non-negative factor analysis into Bradley-Terry (BT) preference model. BNRM represents rewards through a sparse, non-negative latent factor generative process that operates at two complementary levels: instance-specific latent variables induce disentangled reward representations, while sparsity over global latent factors acts as an implicit debiasing mechanism that suppresses spurious correlations. Together, this disentanglement-then-debiasing structure enables robust uncertainty-aware reward learning. To scale BNRM to modern LLMs, we develop an amortized variational inference network conditioned on deep model representations, allowing efficient end-to-end training. Extensive empirical results demonstrate that BNRM substantially mitigates reward over-optimization, improves robustness under distribution shifts, and yields more interpretable reward decompositions than strong baselines. 

---
# Online Causal Kalman Filtering for Stable and Effective Policy Optimization 

**Authors**: Shuo He, Lang Feng, Xin Cheng, Lei Feng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2602.10609)  

**Abstract**: Reinforcement learning for large language models suffers from high-variance token-level importance sampling (IS) ratios, which would destabilize policy optimization at scale. To improve stability, recent methods typically use a fixed sequence-level IS ratio for all tokens in a sequence or adjust each token's IS ratio separately, thereby neglecting temporal off-policy derivation across tokens in a sequence. In this paper, we first empirically identify that local off-policy deviation is structurally inconsistent at the token level, which may distort policy-gradient updates across adjacent tokens and lead to training collapse. To address the issue, we propose Online Causal Kalman Filtering for stable and effective Policy Optimization (KPO). Concretely, we model the desired IS ratio as a latent state that evolves across tokens and apply a Kalman filter to update this state online and autoregressively based on the states of past tokens, regardless of future tokens. The resulting filtered IS ratios preserve token-wise local structure-aware variation while strongly smoothing noise spikes, yielding more stable and effective policy updates. Experimentally, KPO achieves superior results on challenging math reasoning datasets compared with state-of-the-art counterparts. 

---
# Hierarchical Zero-Order Optimization for Deep Neural Networks 

**Authors**: Sansheng Cao, Zhengyu Ma, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.10607)  

**Abstract**: Zeroth-order (ZO) optimization has long been favored for its biological plausibility and its capacity to handle non-differentiable objectives, yet its computational complexity has historically limited its application in deep neural networks. Challenging the conventional paradigm that gradients propagate layer-by-layer, we propose Hierarchical Zeroth-Order (HZO) optimization, a novel divide-and-conquer strategy that decomposes the depth dimension of the network. We prove that HZO reduces the query complexity from $O(ML^2)$ to $O(ML \log L)$ for a network of width $M$ and depth $L$, representing a significant leap over existing ZO methodologies. Furthermore, we provide a detailed error analysis showing that HZO maintains numerical stability by operating near the unitary limit ($L_{lip} \approx 1$). Extensive evaluations on CIFAR-10 and ImageNet demonstrate that HZO achieves competitive accuracy compared to backpropagation. 

---
# Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters 

**Authors**: Ailin Huang, Ang Li, Aobo Kong, Bin Wang, Binxing Jiao, Bo Dong, Bojun Wang, Boyu Chen, Brian Li, Buyun Ma, Chang Su, Changxin Miao, Changyi Wan, Chao Lou, Chen Hu, Chen Xu, Chenfeng Yu, Chengting Feng, Chengyuan Yao, Chunrui Han, Dan Ma, Dapeng Shi, Daxin Jiang, Dehua Ma, Deshan Sun, Di Qi, Enle Liu, Fajie Zhang, Fanqi Wan, Guanzhe Huang, Gulin Yan, Guoliang Cao, Guopeng Li, Han Cheng, Hangyu Guo, Hanshan Zhang, Hao Nie, Haonan Jia, Haoran Lv, Hebin Zhou, Hekun Lv, Heng Wang, Heung-Yeung Shum, Hongbo Huang, Hongbo Peng, Hongyu Zhou, Hongyuan Wang, Houyong Chen, Huangxi Zhu, Huimin Wu, Huiyong Guo, Jia Wang, Jian Zhou, Jianjian Sun, Jiaoren Wu, Jiaran Zhang, Jiashu Lv, Jiashuo Liu, Jiayi Fu, Jiayu Liu, Jie Cheng, Jie Luo, Jie Yang, Jie Zhou, Jieyi Hou, Jing Bai, Jingcheng Hu, Jingjing Xie, Jingwei Wu, Jingyang Zhang, Jishi Zhou, Junfeng Liu, Junzhe Lin, Ka Man Lo, Kai Liang, Kaibo Liu, Kaijun Tan, Kaiwen Yan, Kaixiang Li, Kang An, Kangheng Lin, Lei Yang, Liang Lv, Liang Zhao, Liangyu Chen, Lieyu Shi, Liguo Tan, Lin Lin, Lina Chen, Luck Ma, Mengqiang Ren, Michael Li, Ming Li, Mingliang Li, Mingming Zhang, Mingrui Chen, Mitt Huang, Na Wang, Peng Liu, Qi Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.10604)  

**Abstract**: We introduce Step 3.5 Flash, a sparse Mixture-of-Experts (MoE) model that bridges frontier-level agentic intelligence and computational efficiency. We focus on what matters most when building agents: sharp reasoning and fast, reliable execution. Step 3.5 Flash pairs a 196B-parameter foundation with 11B active parameters for efficient inference. It is optimized with interleaved 3:1 sliding-window/full attention and Multi-Token Prediction (MTP-3) to reduce the latency and cost of multi-round agentic interactions. To reach frontier-level intelligence, we design a scalable reinforcement learning framework that combines verifiable signals with preference feedback, while remaining stable under large-scale off-policy training, enabling consistent self-improvement across mathematics, code, and tool use. Step 3.5 Flash demonstrates strong performance across agent, coding, and math tasks, achieving 85.4% on IMO-AnswerBench, 86.4% on LiveCodeBench-v6 (2024.08-2025.05), 88.2% on tau2-Bench, 69.0% on BrowseComp (with context management), and 51.0% on Terminal-Bench 2.0, comparable to frontier models such as GPT-5.2 xHigh and Gemini 3.0 Pro. By redefining the efficiency frontier, Step 3.5 Flash provides a high-density foundation for deploying sophisticated agents in real-world industrial environments. 

---
# Neural Additive Experts: Context-Gated Experts for Controllable Model Additivity 

**Authors**: Guangzhi Xiong, Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10585)  

**Abstract**: The trade-off between interpretability and accuracy remains a core challenge in machine learning. Standard Generalized Additive Models (GAMs) offer clear feature attributions but are often constrained by their strictly additive nature, which can limit predictive performance. Introducing feature interactions can boost accuracy yet may obscure individual feature contributions. To address these issues, we propose Neural Additive Experts (NAEs), a novel framework that seamlessly balances interpretability and accuracy. NAEs employ a mixture of experts framework, learning multiple specialized networks per feature, while a dynamic gating mechanism integrates information across features, thereby relaxing rigid additive constraints. Furthermore, we propose targeted regularization techniques to mitigate variance among expert predictions, facilitating a smooth transition from an exclusively additive model to one that captures intricate feature interactions while maintaining clarity in feature attributions. Our theoretical analysis and experiments on synthetic data illustrate the model's flexibility, and extensive evaluations on real-world datasets confirm that NAEs achieve an optimal balance between predictive accuracy and transparent, feature-level explanations. The code is available at this https URL. 

---
# LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization 

**Authors**: Boxiao Wang, Kai Li, Tianyi Liu, Chen Li, Junzhe Wang, Yifan Zhang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.10576)  

**Abstract**: Symbolic regression aims to distill mathematical equations from observational data. Recent approaches have successfully leveraged Large Language Models (LLMs) to generate equation hypotheses, capitalizing on their vast pre-trained scientific priors. However, existing frameworks predominantly treat the LLM as a static generator, relying on prompt-level guidance to steer exploration. This paradigm fails to update the model's internal representations based on search feedback, often yielding physically inconsistent or mathematically redundant expressions. In this work, we propose PiT-PO (Physics-informed Token-regularized Policy Optimization), a unified framework that evolves the LLM into an adaptive generator via reinforcement learning. Central to PiT-PO is a dual-constraint mechanism that rigorously enforces hierarchical physical validity while simultaneously applying fine-grained, token-level penalties to suppress redundant structures. Consequently, PiT-PO aligns LLM to produce equations that are both scientifically consistent and structurally parsimonious. Empirically, PiT-PO achieves state-of-the-art performance on standard benchmarks and successfully discovers novel turbulence models for challenging fluid dynamics problems. We also demonstrate that PiT-PO empowers small-scale models to outperform closed-source giants, democratizing access to high-performance scientific discovery. 

---
# MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning 

**Authors**: Chenhao Zhang, Yazhe Niu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10575)  

**Abstract**: Metaphorical comprehension in images remains a critical challenge for Nowadays AI systems. While Multimodal Large Language Models (MLLMs) excel at basic Visual Question Answering (VQA), they consistently struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. This difficulty stems from the task's demand for sophisticated multi-hop reasoning, cultural context, and Theory of Mind (ToM) capabilities, which current models lack. To fill this gap, we propose MetaphorStar, the first end-to-end visual reinforcement learning (RL) framework for image implication tasks. Our framework includes three core components: the fine-grained dataset TFQ-Data, the visual RL method TFQ-GRPO, and the well-structured benchmark TFQ-Bench.
Our fully open-source MetaphorStar family, trained using TFQ-GRPO on TFQ-Data, significantly improves performance by an average of 82.6% on the image implication benchmarks. Compared with 20+ mainstream MLLMs, MetaphorStar-32B achieves state-of-the-art (SOTA) on Multiple-Choice Question and Open-Style Question, significantly outperforms the top closed-source model Gemini-3.0-pro on True-False Question. Crucially, our experiments reveal that learning image implication tasks improves the general understanding ability, especially the complex visual reasoning ability. We further provide a systematic analysis of model parameter scaling, training data scaling, and the impact of different model architectures and training strategies, demonstrating the broad applicability of our method. We open-sourced all model weights, datasets, and method code at this https URL. 

---
# When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning 

**Authors**: Leheng Sheng, Yongtao Zhang, Wenchang Ma, Yaorui Shi, Ting Huang, Xiang Wang, An Zhang, Ke Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.10560)  

**Abstract**: While reasoning over long context is crucial for various real-world applications, it remains challenging for large language models (LLMs) as they suffer from performance degradation as the context length grows. Recent work MemAgent has tried to tackle this by processing context chunk-by-chunk in an RNN-like loop and updating a textual memory for final answering. However, this naive recurrent memory update faces two crucial drawbacks: (i) memory can quickly explode because it can update indiscriminately, even on evidence-free chunks; and (ii) the loop lacks an exit mechanism, leading to unnecessary computation after even sufficient evidence is collected. To address these issues, we propose GRU-Mem, which incorporates two text-controlled gates for more stable and efficient long-context reasoning. Specifically, in GRU-Mem, the memory only updates when the update gate is open and the recurrent loop will exit immediately once the exit gate is open. To endow the model with such capabilities, we introduce two reward signals $r^{\text{update}}$ and $r^{\text{exit}}$ within end-to-end RL, rewarding the correct updating and exiting behaviors respectively. Experiments on various long-context reasoning tasks demonstrate the effectiveness and efficiency of GRU-Mem, which generally outperforms the vanilla MemAgent with up to 400\% times inference speed acceleration. 

---
# LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer 

**Authors**: Lihan Zha, Asher J. Hancock, Mingtong Zhang, Tenny Yin, Yixuan Huang, Dhruv Shah, Allen Z. Ren, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2602.10556)  

**Abstract**: A long-standing goal in robotics is a generalist policy that can be deployed zero-shot on new robot embodiments without per-embodiment adaptation. Despite large-scale multi-embodiment pre-training, existing Vision-Language-Action models (VLAs) remain tightly coupled to their training embodiments and typically require costly fine-tuning. We introduce Language-Action Pre-training (LAP), a simple recipe that represents low-level robot actions directly in natural language, aligning action supervision with the pre-trained vision-language model's input-output distribution. LAP requires no learned tokenizer, no costly annotation, and no embodiment-specific architectural design. Based on LAP, we present LAP-3B, which to the best of our knowledge is the first VLA to achieve substantial zero-shot transfer to previously unseen robot embodiments without any embodiment-specific fine-tuning. Across multiple novel robots and manipulation tasks, LAP-3B attains over 50% average zero-shot success, delivering roughly a 2x improvement over the strongest prior VLAs. We further show that LAP enables efficient adaptation and favorable scaling, while unifying action prediction and VQA in a shared language-action format that yields additional gains through co-training. 

---
# Contrastive Learning for Multi Label ECG Classification with Jaccard Score Based Sigmoid Loss 

**Authors**: Junichiro Takahashi, Masataka Sato, Satoshi Kodeta, Norihiko Takeda  

**Link**: [PDF](https://arxiv.org/pdf/2602.10553)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the development of multimodal medical AI. While models such as MedGemini achieve high accuracy on VQA tasks like USMLE MM, their performance on ECG based tasks remains limited, and some models, such as MedGemma, do not support ECG data at all. Interpreting ECGs is inherently challenging, and diagnostic accuracy can vary depending on the interpreter's experience. Although echocardiography provides rich diagnostic information, it requires specialized equipment and personnel, limiting its availability. In this study, we focus on constructing a robust ECG encoder for multimodal pretraining using real world hospital data. We employ SigLIP, a CLIP based model with a sigmoid based loss function enabling multi label prediction, and introduce a modified loss function tailored to the multi label nature of ECG data. Experiments demonstrate that incorporating medical knowledge in the language model and applying the modified loss significantly improve multi label ECG classification. To further enhance performance, we increase the embedding dimensionality and apply random cropping to mitigate data drift. Finally, per label analysis reveals which ECG findings are easier or harder to predict. Our study provides a foundational framework for developing medical models that utilize ECG data. 

---
# C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning 

**Authors**: Guanting Ye, Qiyan Zhao, Wenhao Yu, Xiaofeng Zhang, Jianmin Ji, Yanyong Zhang, Ka-Veng Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10551)  

**Abstract**: Recent advances in 3D Large Multimodal Models (LMMs) built on Large Language Models (LLMs) have established the alignment of 3D visual features with LLM representations as the dominant paradigm. However, the inherited Rotary Position Embedding (RoPE) introduces limitations for multimodal processing. Specifically, applying 1D temporal positional indices disrupts the continuity of visual features along the column dimension, resulting in spatial locality loss. Moreover, RoPE follows the prior that temporally closer image tokens are more causally related, leading to long-term decay in attention allocation and causing the model to progressively neglect earlier visual tokens as the sequence length increases. To address these issues, we propose C^2RoPE, an improved RoPE that explicitly models local spatial Continuity and spatial Causal relationships for visual processing. C^2RoPE introduces a spatio-temporal continuous positional embedding mechanism for visual tokens. It first integrates 1D temporal positions with Cartesian-based spatial coordinates to construct a triplet hybrid positional index, and then employs a frequency allocation strategy to encode spatio-temporal positional information across the three index components. Additionally, we introduce Chebyshev Causal Masking, which determines causal dependencies by computing the Chebyshev distance of image tokens in 2D space. Evaluation results across various benchmarks, including 3D scene reasoning and 3D visual question answering, demonstrate C^2RoPE's effectiveness. The code is be available at this https URL. 

---
# Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance 

**Authors**: Shengyang Sun, Jiashen Hua, Junyi Feng, Xiaojin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.10549)  

**Abstract**: Weakly supervised multimodal video anomaly detection has gained significant attention, yet the potential of the text modality remains under-explored. Text provides explicit semantic information that can enhance anomaly characterization and reduce false alarms. However, extracting effective text features is challenging due to the inability of general-purpose language models to capture anomaly-specific nuances and the scarcity of relevant descriptions. Furthermore, multimodal fusion often suffers from redundancy and imbalance. To address these issues, we propose a novel text-guided framework. First, we introduce an in-context learning-based multi-stage text augmentation mechanism to generate high-quality anomaly text samples for fine-tuning the text feature extractor. Second, we design a multi-scale bottleneck Transformer fusion module that uses compressed bottleneck tokens to progressively integrate information across modalities, mitigating redundancy and imbalance. Experiments on UCF-Crime and XD-Violence demonstrate state-of-the-art performance. 

---
# RealHD: A High-Quality Dataset for Robust Detection of State-of-the-Art AI-Generated Images 

**Authors**: Hanzhe Yu, Yun Ye, Jintao Rong, Qi Xuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.10546)  

**Abstract**: The rapid advancement of generative AI has raised concerns about the authenticity of digital images, as highly realistic fake images can now be generated at low cost, potentially increasing societal risks. In response, several datasets have been established to train detection models aimed at distinguishing AI-generated images from real ones. However, existing datasets suffer from limited generalization, low image quality, overly simple prompts, and insufficient image diversity. To address these limitations, we propose a high-quality, large-scale dataset comprising over 730,000 images across multiple categories, including both real and AI-generated images. The generated images are synthesized via state-of-the-art methods, including text-to-image generation (guided by over 10,000 carefully designed prompts), image inpainting, image refinement, and face swapping. Each generated image is annotated with its generation method and category. Inpainting images further include binary masks to indicate inpainted regions, providing rich metadata for analysis. Compared to existing datasets, detection models trained on our dataset demonstrate superior generalization capabilities. Our dataset not only serves as a strong benchmark for evaluating detection methods but also contributes to advancing the robustness of AI-generated image detection techniques. Building upon this, we propose a lightweight detection method based on image noise entropy, which transforms the original image into an entropy tensor of Non-Local Means (NLM) noise before classification. Extensive experiments demonstrate that models trained on our dataset achieve strong generalization, and our method delivers competitive performance, establishing a solid baseline for future research. The dataset and source code are publicly available at this https URL. 

---
# $μ$pscaling small models: Principled warm starts and hyperparameter transfer 

**Authors**: Yuxin Ma, Nan Chen, Mateo Díaz, Soufiane Hayou, Dmitriy Kunisky, Soledad Villar  

**Link**: [PDF](https://arxiv.org/pdf/2602.10545)  

**Abstract**: Modern large-scale neural networks are often trained and released in multiple sizes to accommodate diverse inference budgets. To improve efficiency, recent work has explored model upscaling: initializing larger models from trained smaller ones in order to transfer knowledge and accelerate convergence. However, this method can be sensitive to hyperparameters that need to be tuned at the target upscaled model size, which is prohibitively costly to do directly. It remains unclear whether the most common workaround -- tuning on smaller models and extrapolating via hyperparameter scaling laws -- is still sound when using upscaling. We address this with principled approaches to upscaling with respect to model widths and efficiently tuning hyperparameters in this setting. First, motivated by $\mu$P and any-dimensional architectures, we introduce a general upscaling method applicable to a broad range of architectures and optimizers, backed by theory guaranteeing that models are equivalent to their widened versions and allowing for rigorous analysis of infinite-width limits. Second, we extend the theory of $\mu$Transfer to a hyperparameter transfer technique for models upscaled using our method and empirically demonstrate that this method is effective on realistic datasets and architectures. 

---
# A Swap-Adversarial Framework for Improving Domain Generalization in Electroencephalography-Based Parkinson's Disease Prediction 

**Authors**: Seongwon Jin, Hanseul Choi, Sunggu Yang, Sungho Park, Jibum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.10528)  

**Abstract**: Electroencephalography (ECoG) offers a promising alternative to conventional electrocorticography (EEG) for the early prediction of Parkinson's disease (PD), providing higher spatial resolution and a broader frequency range. However, reproducible comparisons has been limited by ethical constraints in human studies and the lack of open benchmark datasets. To address this gap, we introduce a new dataset, the first reproducible benchmark for PD prediction. It is constructed from long-term ECoG recordings of 6-hydroxydopamine (6-OHDA)-induced rat models and annotated with neural responses measured before and after electrical stimulation. In addition, we propose a Swap-Adversarial Framework (SAF) that mitigates high inter-subject variability and the high-dimensional low-sample-size (HDLSS) problem in ECoG data, while achieving robust domain generalization across ECoG and EEG-based Brain-Computer Interface (BCI) datasets. The framework integrates (1) robust preprocessing, (2) Inter-Subject Balanced Channel Swap (ISBCS) for cross-subject augmentation, and (3) domain-adversarial training to suppress subject-specific bias. ISBCS randomly swaps channels between subjects to reduce inter-subject variability, and domain-adversarial training jointly encourages the model to learn task-relevant shared features. We validated the effectiveness of the proposed method through extensive experiments under cross-subject, cross-session, and cross-dataset settings. Our method consistently outperformed all baselines across all settings, showing the most significant improvements in highly variable environments. Furthermore, the proposed method achieved superior cross-dataset performance between public EEG benchmarks, demonstrating strong generalization capability not only within ECoG but to EEG data. The new dataset and source code will be made publicly available upon publication. 

---
# AI-PACE: A Framework for Integrating AI into Medical Education 

**Authors**: Scott P. McGrath, Katherine K. Kim, Karnjit Johl, Haibo Wang, Nick Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2602.10527)  

**Abstract**: The integration of artificial intelligence (AI) into healthcare is accelerating, yet medical education has not kept pace with these technological advancements. This paper synthesizes current knowledge on AI in medical education through a comprehensive analysis of the literature, identifying key competencies, curricular approaches, and implementation strategies. The aim is highlighting the critical need for structured AI education across the medical learning continuum and offer a framework for curriculum development. The findings presented suggest that effective AI education requires longitudinal integration throughout medical training, interdisciplinary collaboration, and balanced attention to both technical fundamentals and clinical applications. This paper serves as a foundation for medical educators seeking to prepare future physicians for an AI-enhanced healthcare environment. 

---
# LHAW: Controllable Underspecification for Long-Horizon Tasks 

**Authors**: George Pu, Michael S. Lee, Udari Madhushani Sehwag, David J. Lee, Bryan Zhu, Yash Maurya, Mohit Raghavendra, Yuan Xue, Samuel Marc Denton  

**Link**: [PDF](https://arxiv.org/pdf/2602.10525)  

**Abstract**: Long-horizon workflow agents that operate effectively over extended periods are essential for truly autonomous systems. Their reliable execution critically depends on the ability to reason through ambiguous situations in which clarification seeking is necessary to ensure correct task execution. However, progress is limited by the lack of scalable, task-agnostic frameworks for systematically curating and measuring the impact of ambiguity across custom workflows. We address this gap by introducing LHAW (Long-Horizon Augmented Workflows), a modular, dataset-agnostic synthetic pipeline that transforms any well-specified task into controllable underspecified variants by systematically removing information across four dimensions - Goals, Constraints, Inputs, and Context - at configurable severity levels. Unlike approaches that rely on LLM predictions of ambiguity, LHAW validates variants through empirical agent trials, classifying them as outcome-critical, divergent, or benign based on observed terminal state divergence. We release 285 task variants from TheAgentCompany, SWE-Bench Pro and MCP-Atlas according to our taxonomy alongside formal analysis measuring how current agents detect, reason about, and resolve underspecification across ambiguous settings. LHAW provides the first systematic framework for cost-sensitive evaluation of agent clarification behavior in long-horizon settings, enabling development of reliable autonomous systems. 

---
# Co-jump: Cooperative Jumping with Quadrupedal Robots via Multi-Agent Reinforcement Learning 

**Authors**: Shihao Dong, Yeke Chen, Zeren Luo, Jiahui Zhang, Bowen Xu, Jinghan Lin, Yimin Han, Ji Ma, Zhiyou Yu, Yudong Zhao, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10514)  

**Abstract**: While single-agent legged locomotion has witnessed remarkable progress, individual robots remain fundamentally constrained by physical actuation limits. To transcend these boundaries, we introduce Co-jump, a cooperative task where two quadrupedal robots synchronize to execute jumps far beyond their solo capabilities. We tackle the high-impulse contact dynamics of this task under a decentralized setting, achieving synchronization without explicit communication or pre-specified motion primitives. Our framework leverages Multi-Agent Proximal Policy Optimization (MAPPO) enhanced by a progressive curriculum strategy, which effectively overcomes the sparse-reward exploration challenges inherent in mechanically coupled systems. We demonstrate robust performance in simulation and successful transfer to physical hardware, executing multi-directional jumps onto platforms up to 1.5 m in height. Specifically, one of the robots achieves a foot-end elevation of 1.1 m, which represents a 144% improvement over the 0.45 m jump height of a standalone quadrupedal robot, demonstrating superior vertical performance. Notably, this precise coordination is achieved solely through proprioceptive feedback, establishing a foundation for communication-free collaborative locomotion in constrained environments. 

---
# 1%>100%: High-Efficiency Visual Adapter with Complex Linear Projection Optimization 

**Authors**: Dongshuo Yin, Xue Yang, Deng-Ping Fan, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10513)  

**Abstract**: Deploying vision foundation models typically relies on efficient adaptation strategies, whereas conventional full fine-tuning suffers from prohibitive costs and low efficiency. While delta-tuning has proven effective in boosting the performance and efficiency of LLMs during adaptation, its advantages cannot be directly transferred to the fine-tuning pipeline of vision foundation models. To push the boundaries of adaptation efficiency for vision tasks, we propose an adapter with Complex Linear Projection Optimization (CoLin). For architecture, we design a novel low-rank complex adapter that introduces only about 1% parameters to the backbone. For efficiency, we theoretically prove that low-rank composite matrices suffer from severe convergence issues during training, and address this challenge with a tailored loss. Extensive experiments on object detection, segmentation, image classification, and rotated object detection (remote sensing scenario) demonstrate that CoLin outperforms both full fine-tuning and classical delta-tuning approaches with merely 1% parameters for the first time, providing a novel and efficient solution for deployment of vision foundation models. We release the code on this https URL. 

---
# Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation 

**Authors**: Wei Chen, Xingyu Guo, Shuang Li, Yan Zhong, Zhao Zhang, Fuzhen Zhuang, Hongrui Liu, Libang Zhang, Guo Ye, Huimei He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10506)  

**Abstract**: Graph Domain Adaptation (GDA) aims to bridge distribution shifts between domains by transferring knowledge from well-labeled source graphs to given unlabeled target graphs. One promising recent approach addresses graph transfer by discretizing the adaptation process, typically through the construction of intermediate graphs or stepwise alignment procedures. However, such discrete strategies often fail in real-world scenarios, where graph structures evolve continuously and nonlinearly, making it difficult for fixed-step alignment to approximate the actual transformation process. To address these limitations, we propose \textbf{DiffGDA}, a \textbf{Diff}usion-based \textbf{GDA} method that models the domain adaptation process as a continuous-time generative process. We formulate the evolution from source to target graphs using stochastic differential equations (SDEs), enabling the joint modeling of structural and semantic transitions. To guide this evolution, a domain-aware network is introduced to steer the generative process toward the target domain, encouraging the diffusion trajectory to follow an optimal adaptation path. We theoretically show that the diffusion process converges to the optimal solution bridging the source and target domains in the latent space. Extensive experiments on 14 graph transfer tasks across 8 real-world datasets demonstrate DiffGDA consistently outperforms state-of-the-art baselines. 

---
# Low-Dimensional Execution Manifolds in Transformer Learning Dynamics: Evidence from Modular Arithmetic Tasks 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10496)  

**Abstract**: We investigate the geometric structure of learning dynamics in overparameterized transformer models through carefully controlled modular arithmetic tasks. Our primary finding is that despite operating in high-dimensional parameter spaces ($d=128$), transformer training trajectories rapidly collapse onto low-dimensional execution manifolds of dimension $3$--$4$. This dimensional collapse is robust across random seeds and moderate task difficulties, though the orientation of the manifold in parameter space varies between runs. We demonstrate that this geometric structure underlies several empirically observed phenomena: (1) sharp attention concentration emerges as saturation along routing coordinates within the execution manifold, (2) stochastic gradient descent (SGD) exhibits approximately integrable dynamics when projected onto the execution subspace, with non-integrability confined to orthogonal staging directions, and (3) sparse autoencoders capture auxiliary routing structure but fail to isolate execution itself, which remains distributed across the low-dimensional manifold. Our results suggest a unifying geometric framework for understanding transformer learning, where the vast majority of parameters serve to absorb optimization interference while core computation occurs in a dramatically reduced subspace. These findings have implications for interpretability, training curriculum design, and understanding the role of overparameterization in neural network learning. 

---
# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation 

**Authors**: Wei Chen, Xingyu Guo, Shuang Li, Zhao Zhang, Yan Zhong, Fuzhen Zhuang, Deqing wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10489)  

**Abstract**: Graph Domain Adaptation (GDA) transfers knowledge from labeled source graphs to unlabeled target graphs but is challenged by complex, multi-faceted distributional shifts. Existing methods attempt to reduce distributional shifts by aligning manually selected graph elements (e.g., node attributes or structural statistics), which typically require manually designed graph filters to extract relevant features before alignment. However, such approaches are inflexible: they rely on scenario-specific heuristics, and struggle when dominant discrepancies vary across transfer scenarios. To address these limitations, we propose \textbf{ADAlign}, an Adaptive Distribution Alignment framework for GDA. Unlike heuristic methods, ADAlign requires no manual specification of alignment criteria. It automatically identifies the most relevant discrepancies in each transfer and aligns them jointly, capturing the interplay between attributes, structures, and their dependencies. This makes ADAlign flexible, scenario-aware, and robust to diverse and dynamically evolving shifts. To enable this adaptivity, we introduce the Neural Spectral Discrepancy (NSD), a theoretically principled parametric distance that provides a unified view of cross-graph shifts. NSD leverages neural characteristic function in the spectral domain to encode feature-structure dependencies of all orders, while a learnable frequency sampler adaptively emphasizes the most informative spectral components for each task via minimax paradigm. Extensive experiments on 10 datasets and 16 transfer tasks show that ADAlign not only outperforms state-of-the-art baselines but also achieves efficiency gains with lower memory usage and faster training. 

---
# Protecting Context and Prompts: Deterministic Security for Non-Deterministic AI 

**Authors**: Mohan Rajagopalan, Vinay Rao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10481)  

**Abstract**: Large Language Model (LLM) applications are vulnerable to prompt injection and context manipulation attacks that traditional security models cannot prevent. We introduce two novel primitives--authenticated prompts and authenticated context--that provide cryptographically verifiable provenance across LLM workflows. Authenticated prompts enable self-contained lineage verification, while authenticated context uses tamper-evident hash chains to ensure integrity of dynamic inputs. Building on these primitives, we formalize a policy algebra with four proven theorems providing protocol-level Byzantine resistance--even adversarial agents cannot violate organizational policies. Five complementary defenses--from lightweight resource controls to LLM-based semantic validation--deliver layered, preventative security with formal guarantees. Evaluation against representative attacks spanning 6 exhaustive categories achieves 100% detection with zero false positives and nominal overhead. We demonstrate the first approach combining cryptographically enforced prompt lineage, tamper-evident context, and provable policy reasoning--shifting LLM security from reactive detection to preventative guarantees. 

---
# Driving Reaction Trajectories via Latent Flow Matching 

**Authors**: Yili Shen, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10476)  

**Abstract**: Recent advances in reaction prediction have achieved near-saturated accuracy on standard benchmarks (e.g., USPTO), yet most state-of-the-art models formulate the task as a one-shot mapping from reactants to products, offering limited insight into the underlying reaction process. Procedural alternatives introduce stepwise generation but often rely on mechanism-specific supervision, discrete symbolic edits, and computationally expensive inference. In this work, we propose LatentRxnFlow, a new reaction prediction paradigm that models reactions as continuous latent trajectories anchored at the thermodynamic product state. Built on Conditional Flow Matching, our approach learns time-dependent latent dynamics directly from standard reactant-product pairs, without requiring mechanistic annotations or curated intermediate labels. While LatentRxnFlow achieves state-of-the-art performance on USPTO benchmarks, more importantly, the continuous formulation exposes the full generative trajectory, enabling trajectory-level diagnostics that are difficult to realize with discrete or one-shot models. We show that latent trajectory analysis allows us to localize and characterize failure modes and to mitigate certain errors via gated inference. Furthermore, geometric properties of the learned trajectories provide an intrinsic signal of epistemic uncertainty, helping prioritize reliably predictable reaction outcomes and flag ambiguous cases for additional validation. Overall, LatentRxnFlow combines strong predictive accuracy with improved transparency, diagnosability, and uncertainty awareness, moving reaction prediction toward more trustworthy deployment in high-throughput discovery workflows. 

---
# Why Human Guidance Matters in Collaborative Vibe Coding 

**Authors**: Haoyu Hu, Raja Marjieh, Katherine M Collins, Chenyi Li, Thomas L. Griffiths, Ilia Sucholutsky, Nori Jacoby  

**Link**: [PDF](https://arxiv.org/pdf/2602.10473)  

**Abstract**: Writing code has been one of the most transformative ways for human societies to translate abstract ideas into tangible technologies. Modern AI is transforming this process by enabling experts and non-experts alike to generate code without actually writing code, but instead, through natural language instructions, or "vibe coding". While increasingly popular, the cumulative impact of vibe coding on productivity and collaboration, as well as the role of humans in this process, remains unclear. Here, we introduce a controlled experimental framework for studying collaborative vibe coding and use it to compare human-led, AI-led, and hybrid groups. Across 16 experiments involving 604 human participants, we show that people provide uniquely effective high-level instructions for vibe coding across iterations, whereas AI-provided instructions often result in performance collapse. We further demonstrate that hybrid systems perform best when humans retain directional control (providing the instructions), while evaluation is delegated to AI. 

---
# Authenticated Workflows: A Systems Approach to Protecting Agentic AI 

**Authors**: Mohan Rajagopalan, Vinay Rao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10465)  

**Abstract**: Agentic AI systems automate enterprise workflows but existing defenses--guardrails, semantic filters--are probabilistic and routinely bypassed. We introduce authenticated workflows, the first complete trust layer for enterprise agentic AI. Security reduces to protecting four fundamental boundaries: prompts, tools, data, and context. We enforce intent (operations satisfy organizational policies) and integrity (operations are cryptographically authentic) at every boundary crossing, combining cryptographic elimination of attack classes with runtime policy enforcement. This delivers deterministic security--operations either carry valid cryptographic proof or are rejected. We introduce MAPL, an AI-native policy language that expresses agentic constraints dynamically as agents evolve and invocation context changes, scaling as O(log M + N) policies versus O(M x N) rules through hierarchical composition with cryptographic attestations for workflow dependencies. We prove practicality through a universal security runtime integrating nine leading frameworks (MCP, A2A, OpenAI, Claude, LangChain, CrewAI, AutoGen, LlamaIndex, Haystack) through thin adapters requiring zero protocol modifications. Formal proofs establish completeness and soundness. Empirical validation shows 100% recall with zero false positives across 174 test cases, protection against 9 of 10 OWASP Top 10 risks, and complete mitigation of two high impact production CVEs. 

---
# Constructing Industrial-Scale Optimization Modeling Benchmark 

**Authors**: Zhong Li, Hongliang Lu, Tao Wei, Wenyu Liu, Yuxuan Chen, Yuan Lan, Fan Zhang, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10450)  

**Abstract**: Optimization modeling underpins decision-making in logistics, manufacturing, energy, and finance, yet translating natural-language requirements into correct optimization formulations and solver-executable code remains labor-intensive. Although large language models (LLMs) have been explored for this task, evaluation is still dominated by toy-sized or synthetic benchmarks, masking the difficulty of industrial problems with $10^{3}$--$10^{6}$ (or more) variables and constraints. A key bottleneck is the lack of benchmarks that align natural-language specifications with reference formulations/solver code grounded in real optimization models. To fill in this gap, we introduce MIPLIB-NL, built via a structure-aware reverse construction methodology from real mixed-integer linear programs in MIPLIB~2017. Our pipeline (i) recovers compact, reusable model structure from flat solver formulations, (ii) reverse-generates natural-language specifications explicitly tied to this recovered structure under a unified model--data separation format, and (iii) performs iterative semantic validation through expert review and human--LLM interaction with independent reconstruction checks. This yields 223 one-to-one reconstructions that preserve the mathematical content of the original instances while enabling realistic natural-language-to-optimization evaluation. Experiments show substantial performance degradation on MIPLIB-NL for systems that perform strongly on existing benchmarks, exposing failure modes invisible at toy scale. 

---
# A Unified Theory of Random Projection for Influence Functions 

**Authors**: Pingbang Hu, Yuzheng Hu, Jiaqi W. Ma, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10449)  

**Abstract**: Influence functions and related data attribution scores take the form of $g^{\top}F^{-1}g^{\prime}$, where $F\succeq 0$ is a curvature operator. In modern overparameterized models, forming or inverting $F\in\mathbb{R}^{d\times d}$ is prohibitive, motivating scalable influence computation via random projection with a sketch $P \in \mathbb{R}^{m\times d}$. This practice is commonly justified via the Johnson--Lindenstrauss (JL) lemma, which ensures approximate preservation of Euclidean geometry for a fixed dataset. However, JL does not address how sketching behaves under inversion. Furthermore, there is no existing theory that explains how sketching interacts with other widely-used techniques, such as ridge regularization and structured curvature approximations.
We develop a unified theory characterizing when projection provably preserves influence functions. When $g,g^{\prime}\in\text{range}(F)$, we show that: 1) Unregularized projection: exact preservation holds iff $P$ is injective on $\text{range}(F)$, which necessitates $m\geq \text{rank}(F)$; 2) Regularized projection: ridge regularization fundamentally alters the sketching barrier, with approximation guarantees governed by the effective dimension of $F$ at the regularization scale; 3) Factorized influence: for Kronecker-factored curvatures $F=A\otimes E$, the guarantees continue to hold for decoupled sketches $P=P_A\otimes P_E$, even though such sketches exhibit row correlations that violate i.i.d. assumptions. Beyond this range-restricted setting, we analyze out-of-range test gradients and quantify a \emph{leakage} term that arises when test gradients have components in $\ker(F)$. This yields guarantees for influence queries on general test points.
Overall, this work develops a novel theory that characterizes when projection provably preserves influence and provides principled guidance for choosing the sketch size in practice. 

---
# LakeMLB: Data Lake Machine Learning Benchmark 

**Authors**: Feiyu Pan, Tianbin Zhang, Aoqian Zhang, Yu Sun, Zheng Wang, Lixing Chen, Li Pan, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10441)  

**Abstract**: Modern data lakes have emerged as foundational platforms for large-scale machine learning, enabling flexible storage of heterogeneous data and structured analytics through table-oriented abstractions. Despite their growing importance, standardized benchmarks for evaluating machine learning performance in data lake environments remain scarce. To address this gap, we present LakeMLB (Data Lake Machine Learning Benchmark), designed for the most common multi-source, multi-table scenarios in data lakes. LakeMLB focuses on two representative multi-table scenarios, Union and Join, and provides three real-world datasets for each scenario, covering government open data, finance, Wikipedia, and online marketplaces. The benchmark supports three representative integration strategies: pre-training-based, data augmentation-based, and feature augmentation-based approaches. We conduct extensive experiments with state-of-the-art tabular learning methods, offering insights into their performance under complex data lake scenarios. We release both datasets and code to facilitate rigorous research on machine learning in data lake ecosystems; the benchmark is available at this https URL. 

---
# AudioRouter: Data Efficient Audio Understanding via RL based Dual Reasoning 

**Authors**: Liyang Chen, Hongkai Chen, Yujun Cai, Sifan Li, Qingwen Ye, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10439)  

**Abstract**: Large Audio Language Models (LALMs) have demonstrated strong capabilities in audio understanding and reasoning. However, their performance on fine grained auditory perception remains unreliable, and existing approaches largely rely on data intensive training to internalize perceptual abilities. We propose AudioRouter, a reinforcement learning framework that enables LALMs to improve audio understanding by learning when and how to use external audio tools. Rather than tightly coupling tool usage with audio reasoning, AudioRouter formulates tool use as an explicit decision making problem and optimizes a lightweight routing policy while keeping the underlying reasoning model frozen. Experimental results show that AudioRouter achieves substantial improvements on standard audio understanding benchmarks while requiring up to 600x less training data to learn tool usage compared with conventional training paradigms. These findings suggest that learning effective tool usage offers a data efficient and scalable alternative to internalizing perceptual abilities in LALMs. 

---
# Control Reinforcement Learning: Token-Level Mechanistic Analysis via Learned SAE Feature Steering 

**Authors**: Seonglae Cho, Zekun Wu, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2602.10437)  

**Abstract**: Sparse autoencoders (SAEs) decompose language model activations into interpretable features, but existing methods reveal only which features activate, not which change model outputs when amplified. We introduce Control Reinforcement Learning (CRL), which trains a policy to select SAE features for steering at each token, producing interpretable intervention logs: the learned policy identifies features that change model outputs when amplified. Adaptive Feature Masking encourages diverse feature discovery while preserving singlefeature interpretability. The framework yields new analysis capabilities: branch point tracking locates tokens where feature choice determines output correctness; critic trajectory analysis separates policy limitations from value estimation errors; layer-wise comparison reveals syntactic features in early layers and semantic features in later layers. On Gemma-2 2B across MMLU, BBQ, GSM8K, HarmBench, and XSTest, CRL achieves improvements while providing per-token intervention logs. These results establish learned feature steering as a mechanistic interpretability tool that complements static feature analysis with dynamic intervention probes 

---
# A Dual-Stream Physics-Augmented Unsupervised Architecture for Runtime Embedded Vehicle Health Monitoring 

**Authors**: Enzo Nicolas Spotorno, Antonio Augusto Medeiros Frohlich  

**Link**: [PDF](https://arxiv.org/pdf/2602.10432)  

**Abstract**: Runtime quantification of vehicle operational intensity is essential for predictive maintenance and condition monitoring in commercial and heavy-duty fleets. Traditional metrics like mileage fail to capture mechanical burden, while unsupervised deep learning models detect statistical anomalies, typically transient surface shocks, but often conflate statistical stability with mechanical rest. We identify this as a critical blind spot: high-load steady states, such as hill climbing with heavy payloads, appear statistically normal yet impose significant drivetrain fatigue. To resolve this, we propose a Dual-Stream Architecture that fuses unsupervised learning for surface anomaly detection with macroscopic physics proxies for cumulative load estimation. This approach leverages low-frequency sensor data to generate a multi-dimensional health vector, distinguishing between dynamic hazards and sustained mechanical effort. Validated on a RISC-V embedded platform, the architecture demonstrates low computational overhead, enabling comprehensive, edge-based health monitoring on resource-constrained ECUs without the latency or bandwidth costs of cloud-based monitoring. 

---
# Breaking the Curse of Repulsion: Optimistic Distributionally Robust Policy Optimization for Off-Policy Generative Recommendation 

**Authors**: Jie Jiang, Yusen Huo, Xiangxin Zhan, Changping Wang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10430)  

**Abstract**: Policy-based Reinforcement Learning (RL) has established itself as the dominant paradigm in generative recommendation for optimizing sequential user interactions. However, when applied to offline historical logs, these methods suffer a critical failure: the dominance of low-quality data induces severe model collapse. We first establish the Divergence Theory of Repulsive Optimization, revealing that negative gradient updates inherently trigger exponential intensity explosion during off-policy training. This theory elucidates the inherent dilemma of existing methods, exposing their inability to reconcile variance reduction and noise imitation. To break this curse, we argue that the solution lies in rigorously identifying the latent high-quality distribution entangled within the noisy behavior policy. Accordingly, we reformulate the objective as an Optimistic Distributionally Robust Optimization (DRO) problem. Guided by this formulation, we propose Distributionally Robust Policy Optimization (DRPO). We prove that hard filtering is the exact solution to this DRO objective, enabling DRPO to optimally recover high-quality behaviors while strictly discarding divergence-inducing noise. Extensive experiments demonstrate that DRPO achieves state-of-the-art performance on mixed-quality recommendation benchmarks. 

---
# AIvilization v0: Toward Large-Scale Artificial Social Simulation with a Unified Agent Architecture and Adaptive Agent Profiles 

**Authors**: Wenkai Fan, Shurui Zhang, Xiaolong Wang, Haowei Yang, Tsz Wai Chan, Xingyan Chen, Junquan Bi, Zirui Zhou, Jia Liu, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10429)  

**Abstract**: AIvilization v0 is a publicly deployed large-scale artificial society that couples a resource-constrained sandbox economy with a unified LLM-agent architecture, aiming to sustain long-horizon autonomy while remaining executable under rapidly changing environment. To mitigate the tension between goal stability and reactive correctness, we introduce (i) a hierarchical branch-thinking planner that decomposes life goals into parallel objective branches and uses simulation-guided validation plus tiered re-planning to ensure feasibility; (ii) an adaptive agent profile with dual-process memory that separates short-term execution traces from long-term semantic consolidation, enabling persistent yet evolving identity; and (iii) a human-in-the-loop steering interface that injects long-horizon objectives and short commands at appropriate abstraction levels, with effects propagated through memory rather than brittle prompt overrides. The environment integrates physiological survival costs, non-substitutable multi-tier production, an AMM-based price mechanism, and a gated education-occupation system. Using high-frequency transactions from the platforms mature phase, we find stable markets that reproduce key stylized facts (heavy-tailed returns and volatility clustering) and produce structured wealth stratification driven by education and access constraints. Ablations show simplified planners can match performance on narrow tasks, while the full architecture is more robust under multi-objective, long-horizon settings, supporting delayed investment and sustained exploration. 

---
# Equivariant Evidential Deep Learning for Interatomic Potentials 

**Authors**: Zhongyao Wang, Taoyong Cui, Jiawen Zou, Shufei Zhang, Bo Yan, Wanli Ouyang, Weimin Tan, Mao Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.10419)  

**Abstract**: Uncertainty quantification (UQ) is critical for assessing the reliability of machine learning interatomic potentials (MLIPs) in molecular dynamics (MD) simulations, identifying extrapolation regimes and enabling uncertainty-aware workflows such as active learning for training dataset construction. Existing UQ approaches for MLIPs are often limited by high computational cost or suboptimal performance. Evidential deep learning (EDL) provides a theoretically grounded single-model alternative that determines both aleatoric and epistemic uncertainty in a single forward pass. However, extending evidential formulations from scalar targets to vector-valued quantities such as atomic forces introduces substantial challenges, particularly in maintaining statistical self-consistency under rotational transformations. To address this, we propose \textit{Equivariant Evidential Deep Learning for Interatomic Potentials} ($\text{e}^2$IP), a backbone-agnostic framework that models atomic forces and their uncertainty jointly by representing uncertainty as a full $3\times3$ symmetric positive definite covariance tensor that transforms equivariantly under rotations. Experiments on diverse molecular benchmarks show that $\text{e}^2$IP provides a stronger accuracy-efficiency-reliability balance than the non-equivariant evidential baseline and the widely used ensemble method. It also achieves better data efficiency through the fully equivariant architecture while retaining single-model inference efficiency. 

---
# AI-rithmetic 

**Authors**: Alex Bie, Travis Dick, Alex Kulesza, Prabhakar Raghavan, Vinod Raman, Sergei Vassilvitskii  

**Link**: [PDF](https://arxiv.org/pdf/2602.10416)  

**Abstract**: Modern AI systems have been successfully deployed to win medals at international math competitions, assist with research workflows, and prove novel technical lemmas. However, despite their progress at advanced levels of mathematics, they remain stubbornly bad at basic arithmetic, consistently failing on the simple task of adding two numbers. We present a systematic investigation of this phenomenon. We demonstrate empirically that all frontier models suffer significantly degraded accuracy for integer addition as the number of digits increases. Furthermore, we show that most errors made by these models are highly interpretable and can be attributed to either operand misalignment or a failure to correctly carry; these two error classes explain 87.9%, 62.9%, and 92.4% of Claude Opus 4.1, GPT-5, and Gemini 2.5 Pro errors, respectively. Finally, we show that misalignment errors are frequently related to tokenization, and that carrying errors appear largely as independent random failures. 

---
# Modular Multi-Task Learning for Chemical Reaction Prediction 

**Authors**: Jiayun Pang, Ahmed M. Zaitoun, Xacobe Couso Cambeiro, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2602.10404)  

**Abstract**: Adapting large language models (LLMs) trained on broad organic chemistry to smaller, domain-specific reaction datasets is a key challenge in chemical and pharmaceutical R&D. Effective specialisation requires learning new reaction knowledge while preserving general chemical understanding across related tasks. Here, we evaluate Low-Rank Adaptation (LoRA) as a parameter-efficient alternative to full fine-tuning for organic reaction prediction on limited, complex datasets. Using USPTO reaction classes and challenging C-H functionalisation reactions, we benchmark forward reaction prediction, retrosynthesis and reagent prediction. LoRA achieves accuracy comparable to full fine-tuning while effectively mitigating catastrophic forgetting and better preserving multi-task performance. Both fine-tuning approaches generalise beyond training distributions, producing plausible alternative solvent predictions. Notably, C-H functionalisation fine-tuning reveals that LoRA and full fine-tuning encode subtly different reactivity patterns, suggesting more effective reaction-specific adaptation with LoRA. As LLMs continue to scale, our results highlight the practicality of modular, parameter-efficient fine-tuning strategies for their flexible deployment for chemistry applications. 

---
# Affordances Enable Partial World Modeling with LLMs 

**Authors**: Khimya Khetarpal, Gheorghe Comanici, Jonathan Richens, Jeremy Shar, Fei Xia, Laurent Orseau, Aleksandra Faust, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2602.10390)  

**Abstract**: Full models of the world require complex knowledge of immense detail. While pre-trained large models have been hypothesized to contain similar knowledge due to extensive pre-training on vast amounts of internet scale data, using them directly in a search procedure is inefficient and inaccurate. Conversely, partial models focus on making high quality predictions for a subset of state and actions: those linked through affordances that achieve user intents~\citep{khetarpal2020can}. Can we posit large models as partial world models? We provide a formal answer to this question, proving that agents achieving task-agnostic, language-conditioned intents necessarily possess predictive partial-world models informed by affordances. In the multi-task setting, we introduce distribution-robust affordances and show that partial models can be extracted to significantly improve search efficiency. Empirical evaluations in tabletop robotics tasks demonstrate that our affordance-aware partial models reduce the search branching factor and achieve higher rewards compared to full world models. 

---
# Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs 

**Authors**: Zhongzhi Li, Xuansheng Wu, Yijiang Li, Lijie Hu, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10388)  

**Abstract**: The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs. 

---
# Making Databases Faster with LLM Evolutionary Sampling 

**Authors**: Mehmet Hamza Erol, Xiangpeng Hao, Federico Bianchi, Ciro Greco, Jacopo Tagliabue, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.10387)  

**Abstract**: Traditional query optimization relies on cost-based optimizers that estimate execution cost (e.g., runtime, memory, and I/O) using predefined heuristics and statistical models. Improving these heuristics requires substantial engineering effort, and even when implemented, these heuristics often cannot take into account semantic correlations in queries and schemas that could enable better physical plans. Using our DBPlanBench harness for the DataFusion engine, we expose the physical plan through a compact serialized representation and let the LLM propose localized edits that can be applied and executed. We then apply an evolutionary search over these edits to refine candidates across iterations. Our key insight is that LLMs can leverage semantic knowledge to identify and apply non-obvious optimizations, such as join orderings that minimize intermediate cardinalities. We obtain up to 4.78$\times$ speedups on some queries and we demonstrate a small-to-large workflow in which optimizations found on small databases transfer effectively to larger databases. 

---
# Time-to-Event Transformer to Capture Timing Attention of Events in EHR Time Series 

**Authors**: Jia Li, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10385)  

**Abstract**: Automatically discovering personalized sequential events from large-scale time-series data is crucial for enabling precision medicine in clinical research, yet it remains a formidable challenge even for contemporary AI models. For example, while transformers capture rich associations, they are mostly agnostic to event timing and ordering, thereby bypassing potential causal reasoning.
Intuitively, we need a method capable of evaluating the "degree of alignment" among patient-specific trajectories and identifying their shared patterns, i.e., the significant events in a consistent sequence. This necessitates treating timing as a true \emph{computable} dimension, allowing models to assign ``relative timestamps'' to candidate events beyond their observed physical times.
In this work, we introduce LITT, a novel Timing-Transformer architecture that enables temporary alignment of sequential events on a virtual ``relative timeline'', thereby enabling \emph{event-timing-focused attention} and personalized interpretations of clinical trajectories. Its interpretability and effectiveness are validated on real-world longitudinal EHR data from 3,276 breast cancer patients to predict the onset timing of cardiotoxicity-induced heart disease. Furthermore, LITT outperforms both the benchmark and state-of-the-art survival analysis methods on public datasets, positioning it as a significant step forward for precision medicine in clinical AI. 

---
# The Alignment Bottleneck in Decomposition-Based Claim Verification 

**Authors**: Mahmud Elahi Akhter, Federico Ruggeri, Iman Munire Bilal, Rob Procter, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2602.10380)  

**Abstract**: Structured claim decomposition is often proposed as a solution for verifying complex, multi-faceted claims, yet empirical results have been inconsistent. We argue that these inconsistencies stem from two overlooked bottlenecks: evidence alignment and sub-claim error profiles. To better understand these factors, we introduce a new dataset of real-world complex claims, featuring temporally bounded evidence and human-annotated sub-claim evidence spans. We evaluate decomposition under two evidence alignment setups: Sub-claim Aligned Evidence (SAE) and Repeated Claim-level Evidence (SRE). Our results reveal that decomposition brings significant performance improvement only when evidence is granular and strictly aligned. By contrast, standard setups that rely on repeated claim-level evidence (SRE) fail to improve and often degrade performance as shown across different datasets and domains (PHEMEPlus, MMM-Fact, COVID-Fact). Furthermore, we demonstrate that in the presence of noisy sub-claim labels, the nature of the error ends up determining downstream robustness. We find that conservative "abstention" significantly reduces error propagation compared to aggressive but incorrect predictions. These findings suggest that future claim decomposition frameworks must prioritize precise evidence synthesis and calibrate the label bias of sub-claim verification models. 

---
# ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters 

**Authors**: Reese Kneeland, Wangshu Jiang, Ugo Bruzadin Nunes, Paul Steven Scotti, Arnaud Delorme, Jonathan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10361)  

**Abstract**: To be practical for real-life applications, models for brain-computer interfaces must be easily and quickly deployable on new subjects, effective on affordable scanning hardware, and small enough to run locally on accessible computing resources. To directly address these current limitations, we introduce ENIGMA, a multi-subject electroencephalography (EEG)-to-Image decoding model that reconstructs seen images from EEG recordings and achieves state-of-the-art (SOTA) performance on the research-grade THINGS-EEG2 and consumer-grade AllJoined-1.6M benchmarks, while fine-tuning effectively on new subjects with as little as 15 minutes of data. ENIGMA boasts a simpler architecture and requires less than 1% of the trainable parameters necessary for previous approaches. Our approach integrates a subject-unified spatio-temporal backbone along with a set of multi-subject latent alignment layers and an MLP projector to map raw EEG signals to a rich visual latent space. We evaluate our approach using a broad suite of image reconstruction metrics that have been standardized in the adjacent field of fMRI-to-Image research, and we describe the first EEG-to-Image study to conduct extensive behavioral evaluations of our reconstructions using human raters. Our simple and robust architecture provides a significant performance boost across both research-grade and consumer-grade EEG hardware, and a substantial improvement in fine-tuning efficiency and inference cost. Finally, we provide extensive ablations to determine the architectural choices most responsible for our performance gains in both single and multi-subject cases across multiple benchmark datasets. Collectively, our work provides a substantial step towards the development of practical brain-computer interface applications. 

---
# Beyond Calibration: Confounding Pathology Limits Foundation Model Specificity in Abdominal Trauma CT 

**Authors**: Jineel H Raythatha, Shuchang Ye, Jeremy Hsu, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.10359)  

**Abstract**: Purpose: Translating foundation models into clinical practice requires evaluating their performance under compound distribution shift, where severe class imbalance coexists with heterogeneous imaging appearances. This challenge is relevant for traumatic bowel injury, a rare but high-mortality diagnosis. We investigated whether specificity deficits in foundation models are associated with heterogeneity in the negative class. Methods: This retrospective study used the multi-institutional, RSNA Abdominal Traumatic Injury CT dataset (2019-2023), comprising scans from 23 centres. Two foundation models (MedCLIP, zero-shot; RadDINO, linear probe) were compared against three task-specific approaches (CNN, Transformer, Ensemble). Models were trained on 3,147 patients (2.3% bowel injury prevalence) and evaluated on an enriched 100-patient test set. To isolate negative-class effects, specificity was assessed in patients without bowel injury who had concurrent solid organ injury (n=58) versus no abdominal pathology (n=50). Results: Foundation models achieved equivalent discrimination to task-specific models (AUC, 0.64-0.68 versus 0.58-0.64) with higher sensitivity (79-91% vs 41-74%) but lower specificity (33-50% vs 50-88%). All models demonstrated high specificity in patients without abdominal pathology (84-100%). When solid organ injuries were present, specificity declined substantially for foundation models (50-51 percentage points) compared with smaller reductions of 12-41 percentage points for task-specific models. Conclusion: Foundation models matched task-specific discrimination without task-specific training, but their specificity deficits were driven primarily by confounding negative-class heterogeneity rather than prevalence alone. Susceptibility to negative-class heterogeneity decreased progressively with labelled training, suggesting adaptation is required before clinical implementation. 

---
# Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs 

**Authors**: Keenan Pepper, Alex McKenzie, Florin Pop, Stijn Servaes, Martin Leitgab, Mike Vaiana, Judd Rosenblatt, Michael S. A. Graziano, Diogo de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2602.10352)  

**Abstract**: Self-interpretation methods prompt language models to describe their own internal states, but remain unreliable due to hyperparameter sensitivity. We show that training lightweight adapters on interpretability artifacts, while keeping the LM entirely frozen, yields reliable self-interpretation across tasks and model families. A scalar affine adapter with just $d_\text{model}+1$ parameters suffices: trained adapters generate sparse autoencoder feature labels that outperform the training labels themselves (71% vs 63% generation scoring at 70B scale), identify topics with 94% recall@1 versus 1% for untrained baselines, and decode bridge entities in multi-hop reasoning that appear in neither prompt nor response, surfacing implicit reasoning without chain-of-thought. The learned bias vector alone accounts for 85% of improvement, and simpler adapters generalize better than more expressive alternatives. Controlling for model knowledge via prompted descriptions, we find self-interpretation gains outpace capability gains from 7B to 72B parameters. Our results demonstrate that self-interpretation improves with scale, without modifying the model being interpreted. 

---
# Are More Tokens Rational? Inference-Time Scaling in Language Models as Adaptive Resource Rationality 

**Authors**: Zhimin Hu, Riya Roshan, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2602.10329)  

**Abstract**: Human reasoning is shaped by resource rationality -- optimizing performance under constraints. Recently, inference-time scaling has emerged as a powerful paradigm to improve the reasoning performance of Large Language Models by expanding test-time computation. Specifically, instruction-tuned (IT) models explicitly generate long reasoning steps during inference, whereas Large Reasoning Models (LRMs) are trained by reinforcement learning to discover reasoning paths that maximize accuracy. However, it remains unclear whether resource-rationality can emerge from such scaling without explicit reward related to computational costs. We introduce a Variable Attribution Task in which models infer which variables determine outcomes given candidate variables, input-output trials, and predefined logical functions. By varying the number of candidate variables and trials, we systematically manipulate task complexity. Both models exhibit a transition from brute-force to analytic strategies as complexity increases. IT models degrade on XOR and XNOR functions, whereas LRMs remain robust. These findings suggest that models can adjust their reasoning behavior in response to task complexity, even without explicit cost-based reward. It provides compelling evidence that resource rationality is an emergent property of inference-time scaling itself. 

---
# Confounding Robust Continuous Control via Automatic Reward Shaping 

**Authors**: Mateo Juliani, Mingxuan Li, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2602.10305)  

**Abstract**: Reward shaping has been applied widely to accelerate Reinforcement Learning (RL) agents' training. However, a principled way of designing effective reward shaping functions, especially for complex continuous control problems, remains largely under-explained. In this work, we propose to automatically learn a reward shaping function for continuous control problems from offline datasets, potentially contaminated by unobserved confounding variables. Specifically, our method builds upon the recently proposed causal Bellman equation to learn a tight upper bound on the optimal state values, which is then used as the potentials in the Potential-Based Reward Shaping (PBRS) framework. Our proposed reward shaping algorithm is tested with Soft-Actor-Critic (SAC) on multiple commonly used continuous control benchmarks and exhibits strong performance guarantees under unobserved confounders. More broadly, our work marks a solid first step towards confounding robust continuous control from a causal perspective. Code for training our reward shaping functions can be found at this https URL. 

---
# ECHO: An Open Research Platform for Evaluation of Chat, Human Behavior, and Outcomes 

**Authors**: Jiqun Liu, Nischal Dinesh, Ran Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10295)  

**Abstract**: ECHO (Evaluation of Chat, Human behavior, and Outcomes) is an open research platform designed to support reproducible, mixed-method studies of human interaction with both conversational AI systems and Web search engines. It enables researchers from varying disciplines to orchestrate end-to-end experimental workflows that integrate consent and background surveys, chat-based and search-based information-seeking sessions, writing or judgment tasks, and pre- and post-task evaluations within a unified, low-coding-load framework. ECHO logs fine-grained interaction traces and participant responses, and exports structured datasets for downstream analysis. By supporting both chat and search alongside flexible evaluation instruments, ECHO lowers technical barriers for studying learning, decision making, and user experience across different information access paradigms, empowering researchers from information retrieval, HCI, and the social sciences to conduct scalable and reproducible human-centered AI evaluations. 

---
# ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting 

**Authors**: Zehua Ma, Hanhui Li, Zhenyu Xie, Xiaonan Luo, Michael Kampffmeyer, Feng Gao, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10278)  

**Abstract**: Generating 3D content from a single image remains a fundamentally challenging and ill-posed problem due to the inherent absence of geometric and textural information in occluded regions. While state-of-the-art generative models can synthesize auxiliary views to provide additional supervision, these views inevitably contain geometric inconsistencies and textural misalignments that propagate and amplify artifacts during 3D reconstruction. To effectively harness these imperfect supervisory signals, we propose an adaptive optimization framework guided by excess risk decomposition, termed ERGO. Specifically, ERGO decomposes the optimization losses in 3D Gaussian splatting into two components, i.e., excess risk that quantifies the suboptimality gap between current and optimal parameters, and Bayes error that models the irreducible noise inherent in synthesized views. This decomposition enables ERGO to dynamically estimate the view-specific excess risk and adaptively adjust loss weights during optimization. Furthermore, we introduce geometry-aware and texture-aware objectives that complement the excess-risk-derived weighting mechanism, establishing a synergistic global-local optimization paradigm. Consequently, ERGO demonstrates robustness against supervision noise while consistently enhancing both geometric fidelity and textural quality of the reconstructed 3D content. Extensive experiments on the Google Scanned Objects dataset and the OmniObject3D dataset demonstrate the superiority of ERGO over existing state-of-the-art methods. 

---
# From Classical to Topological Neural Networks Under Uncertainty 

**Authors**: Sarah Harkins Dayton, Layal Bou Hamdan, Ioannis D. Schizas, David L. Boothe, Vasileios Maroulas  

**Link**: [PDF](https://arxiv.org/pdf/2602.10266)  

**Abstract**: This chapter explores neural networks, topological data analysis, and topological deep learning techniques, alongside statistical Bayesian methods, for processing images, time series, and graphs to maximize the potential of artificial intelligence in the military domain. Throughout the chapter, we highlight practical applications spanning image, video, audio, and time-series recognition, fraud detection, and link prediction for graphical data, illustrating how topology-aware and uncertainty-aware models can enhance robustness, interpretability, and generalization. 

---
# The Complexity of Bayesian Network Learning: Revisiting the Superstructure 

**Authors**: Robert Ganian, Viktoriia Korchemna  

**Link**: [PDF](https://arxiv.org/pdf/2602.10253)  

**Abstract**: We investigate the parameterized complexity of Bayesian Network Structure Learning (BNSL), a classical problem that has received significant attention in empirical but also purely theoretical studies. We follow up on previous works that have analyzed the complexity of BNSL w.r.t. the so-called superstructure of the input. While known results imply that BNSL is unlikely to be fixed-parameter tractable even when parameterized by the size of a vertex cover in the superstructure, here we show that a different kind of parameterization - notably by the size of a feedback edge set - yields fixed-parameter tractability. We proceed by showing that this result can be strengthened to a localized version of the feedback edge set, and provide corresponding lower bounds that complement previous results to provide a complexity classification of BNSL w.r.t. virtually all well-studied graph parameters.
We then analyze how the complexity of BNSL depends on the representation of the input. In particular, while the bulk of past theoretical work on the topic assumed the use of the so-called non-zero representation, here we prove that if an additive representation can be used instead then BNSL becomes fixed-parameter tractable even under significantly milder restrictions to the superstructure, notably when parameterized by the treewidth alone. Last but not least, we show how our results can be extended to the closely related problem of Polytree Learning. 

---
# KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis 

**Authors**: Mayur Akewar, Sandeep Madireddy, Dongsheng Luo, Janki Bhimani  

**Link**: [PDF](https://arxiv.org/pdf/2602.10246)  

**Abstract**: Solid State Drives (SSDs) are critical to datacenters, consumer platforms, and mission-critical systems. Yet diagnosing their performance and reliability is difficult because data are fragmented and time-disjoint, and existing methods demand large datasets and expert input while offering only limited insights. Degradation arises not only from shifting workloads and evolving architectures but also from environmental factors such as temperature, humidity, and vibration. We present KORAL, a knowledge driven reasoning framework that integrates Large Language Models (LLMs) with a structured Knowledge Graph (KG) to generate insights into SSD operations. Unlike traditional approaches that require extensive expert input and large datasets, KORAL generates a Data KG from fragmented telemetry and integrates a Literature KG that already organizes knowledge from literature, reports, and traces. This turns unstructured sources into a queryable graph and telemetry into structured knowledge, and both the Graphs guide the LLM to deliver evidence-based, explainable analysis aligned with the domain vocabulary and constraints. Evaluation using real production traces shows that the KORAL delivers expert-level diagnosis and recommendations, supported by grounded explanations that improve reasoning transparency, guide operator decisions, reduce manual effort, and provide actionable insights to improve service quality. To our knowledge, this is the first end-to-end system that combines LLMs and KGs for full-spectrum SSD reasoning including Descriptive, Predictive, Prescriptive, and What-if analysis. We release the generated SSD-specific KG to advance reproducible research in knowledge-based storage system analysis. GitHub Repository: this https URL 

---
# Transforming Policy-Car Swerving for Mitigating Stop-and-Go Traffic Waves: A Practice-Oriented Jam-Absorption Driving Strategy 

**Authors**: Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10234)  

**Abstract**: Stop-and-go waves, as a major form of freeway traffic congestion, cause severe and long-lasting adverse effects, including reduced traffic efficiency, increased driving risks, and higher vehicle emissions. Amongst the highway traffic management strategies, jam-absorption driving (JAD), in which a dedicated vehicle performs "slow-in" and "fast-out" maneuvers before being captured by a stop-and-go wave, has been proposed as a potential method for preventing the propagation of such waves. However, most existing JAD strategies remain impractical mainly due to the lack of discussion regarding implementation vehicles and operational conditions. Inspired by real-world observations of police-car swerving behavior, this paper first introduces a Single-Vehicle Two-Detector Jam-Absorption Driving (SVDD-JAD) problem, and then proposes a practical JAD strategy that transforms such behavior into a maneuver capable of suppressing the propagation of an isolated stop-and-go wave. Five key parameters that significantly affect the proposed strategy, namely, JAD speed, inflow traffic speed, wave width, wave speed, and in-wave speed, are identified and systematically analyzed. Using a SUMO-based simulation as an illustrative example, we further demonstrate how these parameters can be measured in practice with two stationary roadside traffic detectors. The results show that the proposed JAD strategy successfully suppresses the propagation of a stop-and-go wave, without triggering a secondary wave. This paper is expected to take a significant step toward making JAD practical, advancing it from a theoretical concept to a feasible and implementable strategy. To promote reproducibility in the transportation domain, we have also open-sourced all the code on our GitHub repository this https URL. 

---
# ImprovEvolve: Ask AlphaEvolve to Improve the Input Solution and Then Improvise 

**Authors**: Alexey Kravatskiy, Valentin Khrulkov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2602.10233)  

**Abstract**: Recent advances in LLM-guided evolutionary computation, particularly AlphaEvolve, have demonstrated remarkable success in discovering novel mathematical constructions and solving challenging optimization problems. In this article, we present ImprovEvolve, a simple yet effective technique for enhancing LLM-based evolutionary approaches such as AlphaEvolve. Given an optimization problem, the standard approach is to evolve program code that, when executed, produces a solution close to the optimum. We propose an alternative program parameterization that maintains the ability to construct optimal solutions while reducing the cognitive load on the LLM. Specifically, we evolve a program (implementing, e.g., a Python class with a prescribed interface) that provides the following functionality: (1) propose a valid initial solution, (2) improve any given solution in terms of fitness, and (3) perturb a solution with a specified intensity. The optimum can then be approached by iteratively applying improve() and perturb() with a scheduled intensity. We evaluate ImprovEvolve on challenging problems from the AlphaEvolve paper: hexagon packing in a hexagon and the second autocorrelation inequality. For hexagon packing, the evolved program achieves new state-of-the-art results for 11, 12, 15, and 16 hexagons; a lightly human-edited variant further improves results for 14, 17, and 23 hexagons. For the second autocorrelation inequality, the human-edited program achieves a new state-of-the-art lower bound of 0.96258, improving upon AlphaEvolve's 0.96102. 

---
# Blockwise Advantage Estimation for Multi-Objective RL with Verifiable Rewards 

**Authors**: Kirill Pavlenko, Alexander Golubev, Simon Karasik, Boris Yangel  

**Link**: [PDF](https://arxiv.org/pdf/2602.10231)  

**Abstract**: Group Relative Policy Optimization (GRPO) assigns a single scalar advantage to all tokens in a completion. For structured generations with explicit segments and objectives, this couples unrelated reward signals across segments, leading to objective interference and misattributed credit. We propose Blockwise Advantage Estimation, a family of GRPO-compatible methods that assigns each objective its own advantage and applies it only to the tokens in the corresponding text block, reducing reliance on hand-designed scalar rewards and scaling naturally to additional objectives. A key challenge is estimating advantages for later blocks whose rewards are conditioned on sampled prefixes; standard unbiased approaches require expensive nested rollouts from intermediate states. Concretely, we introduce an Outcome-Conditioned Baseline that approximates intermediate state values using only within-group statistics by stratifying samples according to a prefix-derived intermediate outcome. On math tasks with uncertainty estimation, our method mitigates reward interference, is competitive with a state-of-the-art reward-designed approach, and preserves test-time gains from confidence-weighted ensembling. More broadly, it provides a modular recipe for optimizing sequential objectives in structured generations without additional rollouts. 

---
# Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents 

**Authors**: Haochen Wang, Yi Wu, Daryl Chang, Li Wei, Lukasz Heldt  

**Link**: [PDF](https://arxiv.org/pdf/2602.10226)  

**Abstract**: Optimizing large-scale machine learning systems, such as recommendation models for global video platforms, requires navigating a massive hyperparameter search space and, more critically, designing sophisticated optimizers, architectures, and reward functions to capture nuanced user behaviors. Achieving substantial improvements in these areas is a non-trivial task, traditionally relying on extensive manual iterations to test new hypotheses. We propose a self-evolving system that leverages Large Language Models (LLMs), specifically those from Google's Gemini family, to autonomously generate, train, and deploy high-performing, complex model changes within an end-to-end automated workflow. The self-evolving system is comprised of an Offline Agent (Inner Loop) that performs high-throughput hypothesis generation using proxy metrics, and an Online Agent (Outer Loop) that validates candidates against delayed north star business metrics in live production. Our agents act as specialized Machine Learning Engineers (MLEs): they exhibit deep reasoning capabilities, discovering novel improvements in optimization algorithms and model architecture, and formulating innovative reward functions that target long-term user engagement. The effectiveness of this approach is demonstrated through several successful production launches at YouTube, confirming that autonomous, LLM-driven evolution can surpass traditional engineering workflows in both development velocity and model performance. 

---
# Quantum Integrated Sensing and Computation with Indefinite Causal Order 

**Authors**: Ivana Nikoloska  

**Link**: [PDF](https://arxiv.org/pdf/2602.10225)  

**Abstract**: Quantum operations with indefinite causal order (ICO) represent a framework in quantum information processing where the relative order between two events can be indefinite. In this paper, we investigate whether sensing and computation, two canonical tasks in quantum information processing, can be carried out within the ICO framework. We propose a scheme for integrated sensing and computation that uses the same quantum state for both tasks. The quantum state is represented as an agent that performs state observation and learns a function of the state to make predictions via a parametric model. Under an ICO operation, the agent experiences a superposition of orders, one in which it performs state observation and then executes the required computation steps, and another in which the agent carries out the computation first and then performs state observation. This is distinct from prevailing information processing and machine intelligence paradigms where information acquisition and learning follow a strict causal order, with the former always preceding the latter. We provide experimental results and we show that the proposed scheme can achieve small training and testing losses on a representative task in magnetic navigation. 

---
# Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models 

**Authors**: Shiting Huang, Zecheng Li, Yu Zeng, Qingnan Ren, Zhen Fang, Qisheng Su, Kou Shi, Lin Chen, Zehui Chen, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10224)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective approach for enhancing the reasoning capabilities of Large Language Models (LLMs). Despite its efficacy, RLVR faces a meta-learning bottleneck: it lacks mechanisms for error attribution and experience internalization intrinsic to the human learning cycle beyond practice and verification, thereby limiting fine-grained credit assignment and reusable knowledge formation. We term such reusable knowledge representations derived from past errors as meta-experience. Based on this insight, we propose Meta-Experience Learning (MEL), a novel framework that incorporates self-distilled meta-experience into the model's parametric memory. Building upon standard RLVR, we introduce an additional design that leverages the LLM's self-verification capability to conduct contrastive analysis on paired correct and incorrect trajectories, identify the precise bifurcation points where reasoning errors arise, and summarize them into generalizable meta-experience. The meta-experience is further internalized into the LLM's parametric memory by minimizing the negative log-likelihood, which induces a language-modeled reward signal that bridges correct and incorrect reasoning trajectories and facilitates effective knowledge reuse. Experimental results demonstrate that MEL achieves consistent improvements on benchmarks, yielding 3.92%--4.73% Pass@1 gains across varying model sizes. 

---
# Versor: A Geometric Sequence Architecture 

**Authors**: Truong Minh Huy, Edward Hirst  

**Link**: [PDF](https://arxiv.org/pdf/2602.10195)  

**Abstract**: A novel sequence architecture design is introduced, Versor, which uses Conformal Geometric Algebra (CGA) in place of the traditional fundamental non-linear operations to achieve structural generalization and significant performance improvements on a variety of tasks, while offering improved interpretability and efficiency. By embedding states in the $Cl_{4,1}$ manifold and evolving them via geometric transformations (rotors), Versor natively represents $SE(3)$-equivariant relationships without requiring explicit structural encoding. Versor is validated on chaotic N-body dynamics, topological reasoning, and standard multimodal benchmarks (CIFAR-10, WikiText-103), consistently outperforming Transformers, Graph Networks, and geometric baselines (GATr, EGNN). Key results include: orders of magnitude fewer parameters ($200\times$ vs. Transformers); interpretable attention decomposing into proximity and orientational components; zero-shot scale generalization (99.3% MCC on topology vs. 50.4% for ViT); and $O(L)$ linear complexity via the novel Recursive Rotor Accumulator. In out-of-distribution tests, Versor maintains stable predictions while Transformers fail catastrophically. Custom Clifford kernels achieve up to $78\times$ speedup, providing a scalable foundation for geometrically-aware scientific modeling. 

---
# When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models 

**Authors**: Jiacheng Hou, Yining Sun, Ruochong Jin, Haochen Han, Fangming Liu, Wai Kin Victor Chan, Alex Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10179)  

**Abstract**: Recent advances in large image editing models have shifted the paradigm from text-driven instructions to vision-prompt editing, where user intent is inferred directly from visual inputs such as marks, arrows, and visual-text prompts. While this paradigm greatly expands usability, it also introduces a critical and underexplored safety risk: the attack surface itself becomes visual. In this work, we propose Vision-Centric Jailbreak Attack (VJA), the first visual-to-visual jailbreak attack that conveys malicious instructions purely through visual inputs. To systematically study this emerging threat, we introduce IESBench, a safety-oriented benchmark for image editing models. Extensive experiments on IESBench demonstrate that VJA effectively compromises state-of-the-art commercial models, achieving attack success rates of up to 80.9% on Nano Banana Pro and 70.1% on GPT-Image-1.5. To mitigate this vulnerability, we propose a training-free defense based on introspective multimodal reasoning, which substantially improves the safety of poorly aligned models to a level comparable with commercial systems, without auxiliary guard models and with negligible computational overhead. Our findings expose new vulnerabilities, provide both a benchmark and practical defense to advance safe and trustworthy modern image editing systems. Warning: This paper contains offensive images created by large image editing models. 

---
# Towards Autonomous Mathematics Research 

**Authors**: Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang-hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, Thang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2602.10177)  

**Abstract**: Recent advances in foundational models have yielded reasoning systems capable of achieving a gold-medal standard at the International Mathematical Olympiad. The transition from competition-level problem-solving to professional research, however, requires navigating vast literature and constructing long-horizon proofs. In this work, we introduce Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language. Specifically, Aletheia is powered by an advanced version of Gemini Deep Think for challenging reasoning problems, a novel inference-time scaling law that extends beyond Olympiad-level problems, and intensive tool use to navigate the complexities of mathematical research. We demonstrate the capability of Aletheia from Olympiad problems to PhD-level exercises and most notably, through several distinct milestones in AI-assisted mathematics research: (a) a research paper (Feng26) generated by AI without any human intervention in calculating certain structure constants in arithmetic geometry called eigenweights; (b) a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets; and (c) an extensive semi-autonomous evaluation (Feng et al., 2026a) of 700 open problems on Bloom's Erdos Conjectures database, including autonomous solutions to four open questions. In order to help the public better understand the developments pertaining to AI and mathematics, we suggest codifying standard levels quantifying autonomy and novelty of AI-assisted results. We conclude with reflections on human-AI collaboration in mathematics. 

---
# Cosmo3DFlow: Wavelet Flow Matching for Spatial-to-Spectral Compression in Reconstructing the Early Universe 

**Authors**: Md. Khairul Islam, Zeyu Xia, Ryan Goudjil, Jialu Wang, Arya Farahi, Judy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2602.10172)  

**Abstract**: Reconstructing the early Universe from the evolved present-day Universe is a challenging and computationally demanding problem in modern astrophysics. We devise a novel generative framework, Cosmo3DFlow, designed to address dimensionality and sparsity, the critical bottlenecks inherent in current state-of-the-art methods for cosmological inference. By integrating 3D Discrete Wavelet Transform (DWT) with flow matching, we effectively represent high-dimensional cosmological structures. The Wavelet Transform addresses the ``void problem'' by translating spatial emptiness into spectral sparsity. It decouples high-frequency details from low-frequency structures through spatial compression, and wavelet-space velocity fields facilitate stable ordinary differential equation (ODE) solvers with large step sizes. Using large-scale cosmological $N$-body simulations, at $128^3$ resolution, we achieve up to $50\times$ faster sampling than diffusion models, combining a $10\times$ reduction in integration steps with lower per-step computational cost from wavelet compression. Our results enable initial conditions to be sampled in seconds, compared to minutes for previous methods. 

---
# EvoCodeBench: A Human-Performance Benchmark for Self-Evolving LLM-Driven Coding Systems 

**Authors**: Wentao Zhang, Jianfeng Wang, Liheng Liang, Yilei Zhao, HaiBin Wen, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.10171)  

**Abstract**: As large language models (LLMs) continue to advance in programming tasks, LLM-driven coding systems have evolved from one-shot code generation into complex systems capable of iterative improvement during inference. However, existing code benchmarks primarily emphasize static correctness and implicitly assume fixed model capability during inference. As a result, they do not capture inference-time self-evolution, such as whether accuracy and efficiency improve as an agent iteratively refines its solutions. They also provide limited accounting of resource costs and rarely calibrate model performance against that of human programmers. Moreover, many benchmarks are dominated by high-resource languages, leaving cross-language robustness and long-tail language stability underexplored. Therefore, we present EvoCodeBench, a benchmark for evaluating self-evolving LLM-driven coding systems across programming languages with direct comparison to human performance. EvoCodeBench tracks performance dynamics, measuring solution correctness alongside efficiency metrics such as solving time, memory consumption, and improvement algorithmic design over repeated problem-solving attempts. To ground evaluation in a human-centered reference frame, we directly compare model performance with that of human programmers on the same tasks, enabling relative performance assessment within the human ability distribution. Furthermore, EvoCodeBench supports multiple programming languages, enabling systematic cross-language and long-tail stability analyses under a unified protocol. Our results demonstrate that self-evolving systems exhibit measurable gains in efficiency over time, and that human-relative and multi-language analyses provide insights unavailable through accuracy alone. EvoCodeBench establishes a foundation for evaluating coding intelligence in evolving LLM-driven systems. 

---
# EVA: Towards a universal model of the immune system 

**Authors**: Ethan Bandasack, Vincent Bouget, Apolline Bruley, Yannis Cattan, Charlotte Claye, Matthew Corney, Julien Duquesne, Karim El Kanbi, Aziz Fouché, Pierre Marschall, Francesco Strozzi  

**Link**: [PDF](https://arxiv.org/pdf/2602.10168)  

**Abstract**: The effective application of foundation models to translational research in immune-mediated diseases requires multimodal patient-level representations that can capture complex phenotypes emerging from multicellular interactions. Yet most current biological foundation models focus only on single-cell resolution and are evaluated on technical metrics often disconnected from actual drug development tasks and challenges. Here, we introduce EVA, the first cross-species, multimodal foundation model of immunology and inflammation, a therapeutic area where shared pathogenic mechanisms create unique opportunities for transfer learning. EVA harmonizes transcriptomics data across species, platforms, and resolutions, and integrates histology data to produce rich, unified patient representations. We establish clear scaling laws, demonstrating that increasing model size and compute translates to improvements in both pretraining and downstream tasks performance. We introduce a comprehensive evaluation suite of 39 tasks spanning the drug development pipeline: zero-shot target efficacy and gene function prediction for discovery, cross-species or cross-diseases molecular perturbations for preclinical development, and patient stratification with treatment response prediction or disease activity prediction for clinical trials applications. We benchmark EVA against several state-of-the-art biological foundation models and baselines on these tasks, and demonstrate state-of-the-art results on each task category. Using mechanistic interpretability, we further identify biological meaningful features, revealing intertwined representations across species and technologies. We release an open version of EVA for transcriptomics to accelerate research on immune-mediated diseases. 

---
# Anatomy-Preserving Latent Diffusion for Generation of Brain Segmentation Masks with Ischemic Infarct 

**Authors**: Lucia Borrego, Vajira Thambawita, Marco Ciuffreda, Ines del Val, Alejandro Dominguez, Josep Munuera  

**Link**: [PDF](https://arxiv.org/pdf/2602.10167)  

**Abstract**: The scarcity of high-quality segmentation masks remains a major bottleneck for medical image analysis, particularly in non-contrast CT (NCCT) neuroimaging, where manual annotation is costly and variable. To address this limitation, we propose an anatomy-preserving generative framework for the unconditional synthesis of multi-class brain segmentation masks, including ischemic infarcts.
The proposed approach combines a variational autoencoder trained exclusively on segmentation masks to learn an anatomical latent representation, with a diffusion model operating in this latent space to generate new samples from pure noise. At inference, synthetic masks are obtained by decoding denoised latent vectors through the frozen VAE decoder, with optional coarse control over lesion presence via a binary prompt.
Qualitative results show that the generated masks preserve global brain anatomy, discrete tissue semantics, and realistic variability, while avoiding the structural artifacts commonly observed in pixel-space generative models. Overall, the proposed framework offers a simple and scalable solution for anatomy-aware mask generation in data-scarce medical imaging scenarios. 

---
# Beyond SMILES: Evaluating Agentic Systems for Drug Discovery 

**Authors**: Edward Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2602.10163)  

**Abstract**: Agentic systems for drug discovery have demonstrated autonomous synthesis planning, literature mining, and molecular design. We ask how well they generalize. Evaluating six frameworks against 15 task classes drawn from peptide therapeutics, in vivo pharmacology, and resource-constrained settings, we find five capability gaps: no support for protein language models or peptide-specific prediction, no bridges between in vivo and in silico data, reliance on LLM inference with no pathway to ML training or reinforcement learning, assumptions tied to large-pharma resources, and single-objective optimization that ignores safety-efficacy-stability trade-offs. A paired knowledge-probing experiment suggests the bottleneck is architectural rather than epistemic: four frontier LLMs reason about peptides at levels comparable to small molecules, yet no framework exposes this capability. We propose design requirements and a capability matrix for next-generation frameworks that function as computational partners under realistic constraints. 

---
# Omni-Safety under Cross-Modality Conflict: Vulnerabilities, Dynamics Mechanisms and Efficient Alignment 

**Authors**: Kun Wang, Zherui Li, Zhenhong Zhou, Yitong Zhang, Yan Mi, Kun Yang, Yiming Zhang, Junhao Dong, Zhongxiang Sun, Qiankun Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10161)  

**Abstract**: Omni-modal Large Language Models (OLLMs) greatly expand LLMs' multimodal capabilities but also introduce cross-modal safety risks. However, a systematic understanding of vulnerabilities in omni-modal interactions remains lacking. To bridge this gap, we establish a modality-semantics decoupling principle and construct the AdvBench-Omni dataset, which reveals a significant vulnerability in OLLMs. Mechanistic analysis uncovers a Mid-layer Dissolution phenomenon driven by refusal vector magnitude shrinkage, alongside the existence of a modal-invariant pure refusal direction. Inspired by these insights, we extract a golden refusal vector using Singular Value Decomposition and propose OmniSteer, which utilizes lightweight adapters to modulate intervention intensity adaptively. Extensive experiments show that our method not only increases the Refusal Success Rate against harmful inputs from 69.9% to 91.2%, but also effectively preserves the general capabilities across all modalities. Our code is available at: this https URL. 

---
# AD$^2$: Analysis and Detection of Adversarial Threats in Visual Perception for End-to-End Autonomous Driving Systems 

**Authors**: Ishan Sahu, Somnath Hazra, Somak Aditya, Soumyajit Dey  

**Link**: [PDF](https://arxiv.org/pdf/2602.10160)  

**Abstract**: End-to-end autonomous driving systems have achieved significant progress, yet their adversarial robustness remains largely underexplored. In this work, we conduct a closed-loop evaluation of state-of-the-art autonomous driving agents under black-box adversarial threat models in CARLA. Specifically, we consider three representative attack vectors on the visual perception pipeline: (i) a physics-based blur attack induced by acoustic waves, (ii) an electromagnetic interference attack that distorts captured images, and (iii) a digital attack that adds ghost objects as carefully crafted bounded perturbations on images. Our experiments on two advanced agents, Transfuser and Interfuser, reveal severe vulnerabilities to such attacks, with driving scores dropping by up to 99% in the worst case, raising valid safety concerns. To help mitigate such threats, we further propose a lightweight Attack Detection model for Autonomous Driving systems (AD$^2$) based on attention mechanisms that capture spatial-temporal consistency. Comprehensive experiments across multi-camera inputs on CARLA show that our detector achieves superior detection capability and computational efficiency compared to existing approaches. 

---
# NMRTrans: Structure Elucidation from Experimental NMR Spectra via Set Transformers 

**Authors**: Liujia Yang, Zhuo Yang, Jiaqing Xie, Yubin Wang, Ben Gao, Tianfan Fu, Xingjian Wei, Jiaxing Sun, Jiang Wu, Conghui He, Yuqiang Li, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10158)  

**Abstract**: Nuclear Magnetic Resonance (NMR) spectroscopy is fundamental for molecular structure elucidation, yet interpreting spectra at scale remains time-consuming and highly expertise-dependent. While recent spectrum-as-language modeling and retrieval-based methods have shown promise, they rely heavily on large corpora of computed spectra and exhibit notable performance drops when applied to experimental measurements. To address these issues, we build NMRSpec, a large-scale corpus of experimental $^1$H and $^{13}$C spectra mined from chemical literature, and propose NMRTrans, which models spectra as unordered peak sets and aligns the model's inductive bias with the physical nature of NMR. To our best knowledge, NMRTrans is the first NMR Transformer trained solely on large-scale experimental spectra and achieves state-of-the-art performance on experimental benchmarks, improving Top-10 Accuracy over the strongest baseline by +17.82 points (61.15% vs. 43.33%), and underscoring the importance of experimental data and structure-aware architectures for reliable NMR structure elucidation. 

---
# MalMoE: Mixture-of-Experts Enhanced Encrypted Malicious Traffic Detection Under Graph Drift 

**Authors**: Yunpeng Tan, Qingyang Li, Mingxin Yang, Yannan Hu, Lei Zhang, Xinggong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10157)  

**Abstract**: Encryption has been commonly used in network traffic to secure transmission, but it also brings challenges for malicious traffic detection, due to the invisibility of the packet payload. Graph-based methods are emerging as promising solutions by leveraging multi-host interactions to promote detection accuracy. But most of them face a critical problem: Graph Drift, where the flow statistics or topological information of a graph change over time. To overcome these drawbacks, we propose a graph-assisted encrypted traffic detection system, MalMoE, which applies Mixture of Experts (MoE) to select the best expert model for drift-aware classification. Particularly, we design 1-hop-GNN-like expert models that handle different graph drifts by analyzing graphs with different features. Then, the redesigned gate model conducts expert selection according to the actual drift. MalMoE is trained with a stable two-stage training strategy with data augmentation, which effectively guides the gate on how to perform routing. Experiments on open-source, synthetic, and real-world datasets show that MalMoE can perform precise and real-time detection. 

---
# PRISM-XR: Empowering Privacy-Aware XR Collaboration with Multimodal Large Language Models 

**Authors**: Jiangong Chen, Mingyu Zhu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10154)  

**Abstract**: Multimodal Large Language Models (MLLMs) enhance collaboration in Extended Reality (XR) environments by enabling flexible object and animation creation through the combination of natural language and visual inputs. However, visual data captured by XR headsets includes real-world backgrounds that may contain irrelevant or sensitive user information, such as credit cards left on the table or facial identities of other users. Uploading those frames to cloud-based MLLMs poses serious privacy risks, particularly when such data is processed without explicit user consent. Additionally, existing colocation and synchronization mechanisms in commercial XR APIs rely on time-consuming, privacy-invasive environment scanning and struggle to adapt to the highly dynamic nature of MLLM-integrated XR environments. In this paper, we propose PRISM-XR, a novel framework that facilitates multi-user collaboration in XR by providing privacy-aware MLLM integration. PRISM-XR employs intelligent frame preprocessing on the edge server to filter sensitive data and remove irrelevant context before communicating with cloud generative AI models. Additionally, we introduce a lightweight registration process and a fully customizable content-sharing mechanism to enable efficient, accurate, and privacy-preserving content synchronization among users. Our numerical evaluation results indicate that the proposed platform achieves nearly 90% accuracy in fulfilling user requests and less than 0.27 seconds registration time while maintaining spatial inconsistencies of less than 3.5 cm. Furthermore, we conducted an IRB-approved user study with 28 participants, demonstrating that our system could automatically filter highly sensitive objects in over 90% of scenarios while maintaining strong overall usability. 

---
# PEST: Physics-Enhanced Swin Transformer for 3D Turbulence Simulation 

**Authors**: Yilong Dai, Shengyu Chen, Xiaowei Jia, Peyman Givi, Runlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10150)  

**Abstract**: Accurate simulation of turbulent flows is fundamental to scientific and engineering applications. Direct numerical simulation (DNS) offers the highest fidelity but is computationally prohibitive, while existing data-driven alternatives struggle with stable long-horizon rollouts, physical consistency, and faithful simulation of small-scale structures. These challenges are particularly acute in three-dimensional (3D) settings, where the cubic growth of spatial degrees of freedom dramatically amplifies computational cost, memory demand, and the difficulty of capturing multi-scale interactions. To address these challenges, we propose a Physics-Enhanced Swin Transformer (PEST) for 3D turbulence simulation. PEST leverages a window-based self-attention mechanism to effectively model localized PDE interactions while maintaining computational efficiency. We introduce a frequency-domain adaptive loss that explicitly emphasizes small-scale structures, enabling more faithful simulation of high-frequency dynamics. To improve physical consistency, we incorporate Navier--Stokes residual constraints and divergence-free regularization directly into the learning objective. Extensive experiments on two representative turbulent flow configurations demonstrate that PEST achieves accurate, physically consistent, and stable autoregressive long-term simulations, outperforming existing data-driven baselines. 

---
# Exploring Semantic Labeling Strategies for Third-Party Cybersecurity Risk Assessment Questionnaires 

**Authors**: Ali Nour Eldin, Mohamed Sellami, Walid Gaaloul  

**Link**: [PDF](https://arxiv.org/pdf/2602.10149)  

**Abstract**: Third-Party Risk Assessment (TPRA) is a core cybersecurity practice for evaluating suppliers against standards such as ISO/IEC 27001 and NIST. TPRA questionnaires are typically drawn from large repositories of security and compliance questions, yet tailoring assessments to organizational needs remains a largely manual process. Existing retrieval approaches rely on keyword or surface-level similarity, which often fails to capture implicit assessment scope and control semantics.
This paper explores strategies for organizing and retrieving TPRA cybersecurity questions using semantic labels that describe both control domains and assessment scope. We compare direct question-level labeling with a Large Language Model (LLM) against a hybrid semi-supervised semantic labeling (SSSL) pipeline that clusters questions in embedding space, labels a small representative subset using an LLM, and propagates labels to remaining questions using k-Nearest Neighbors; we also compare downstream retrieval based on direct question similarity versus retrieval in the label space. We find that semantic labels can improve retrieval alignment when labels are discriminative and consistent, and that SSSL can generalize labels from a small labeled subset to large repositories while substantially reducing LLM usage and cost. 

---
# Red-teaming the Multimodal Reasoning: Jailbreaking Vision-Language Models via Cross-modal Entanglement Attacks 

**Authors**: Yu Yan, Sheng Sun, Shengjia Cheng, Teli Liu, Mingfeng Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10148)  

**Abstract**: Vision-Language Models (VLMs) with multimodal reasoning capabilities are high-value attack targets, given their potential for handling complex multimodal harmful tasks. Mainstream black-box jailbreak attacks on VLMs work by distributing malicious clues across modalities to disperse model attention and bypass safety alignment mechanisms. However, these adversarial attacks rely on simple and fixed image-text combinations that lack attack complexity scalability, limiting their effectiveness for red-teaming VLMs' continuously evolving reasoning capabilities. We propose \textbf{CrossTALK} (\textbf{\underline{Cross}}-modal en\textbf{\underline{TA}}ng\textbf{\underline{L}}ement attac\textbf{\underline{K}}), which is a scalable approach that extends and entangles information clues across modalities to exceed VLMs' trained and generalized safety alignment patterns for jailbreak. Specifically, {knowledge-scalable reframing} extends harmful tasks into multi-hop chain instructions, {cross-modal clue entangling} migrates visualizable entities into images to build multimodal reasoning links, and {cross-modal scenario nesting} uses multimodal contextual instructions to steer VLMs toward detailed harmful outputs. Experiments show our COMET achieves state-of-the-art attack success rate. 

---
# On the Use of a Large Language Model to Support the Conduction of a Systematic Mapping Study: A Brief Report from a Practitioner's View 

**Authors**: Cauã Ferreira Barros, Marcos Kalinowski, Mohamad Kassab, Valdemar Vicente Graciano Neto  

**Link**: [PDF](https://arxiv.org/pdf/2602.10147)  

**Abstract**: The use of Large Language Models (LLMs) has drawn growing interest within the scientific community. LLMs can handle large volumes of textual data and support methods for evidence synthesis. Although recent studies highlight the potential of LLMs to accelerate screening and data extraction steps in systematic reviews, detailed reports of their practical application throughout the entire process remain scarce. This paper presents an experience report on the conduction of a systematic mapping study with the support of LLMs, describing the steps followed, the necessary adjustments, and the main challenges faced. Positive aspects are discussed, such as (i) the significant reduction of time in repetitive tasks and (ii) greater standardization in data extraction, as well as negative aspects, including (i) considerable effort to build reliable well-structured prompts, especially for less experienced users, since achieving effective prompts may require several iterations and testing, which can partially offset the expected time savings, (ii) the occurrence of hallucinations, and (iii) the need for constant manual verification. As a contribution, this work offers lessons learned and practical recommendations for researchers interested in adopting LLMs in systematic mappings and reviews, highlighting both efficiency gains and methodological risks and limitations to be considered. 

---
# Silence Routing: When Not Speaking Improves Collective Judgment 

**Authors**: Itsuki Fujisaki, Kunhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10145)  

**Abstract**: The wisdom of crowds has been shown to operate not only for factual judgments but also in matters of taste, where accuracy is defined relative to an individual's preferences. However, it remains unclear how different types of social signals should be selectively used in such domains. Focusing on a music preference dataset in which contributors provide both personal evaluations (Own) and estimates of population-level preferences (Estimated), we propose a routing framework for collective intelligence in taste. The framework specifies when contributors should speak, what they should report, and when silence is preferable. Using simulation-based aggregation, we show that prediction accuracy improves over an all-own baseline across a broad region of the parameter space, conditional on items where routing applies. Importantly, these gains arise only when silence is allowed, enabling second-order signals to function effectively. The results demonstrate that collective intelligence in matters of taste depends on principled signal routing rather than simple averaging. 

---
# When LLMs get significantly worse: A statistical approach to detect model degradations 

**Authors**: Jonas Kübler, Kailash Budhathoki, Matthäus Kleindessner, Xiong Zhou, Junming Yin, Ashish Khetan, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2602.10144)  

**Abstract**: Minimizing the inference cost and latency of foundation models has become a crucial area of research. Optimization approaches include theoretically lossless methods and others without accuracy guarantees like quantization. In all of these cases it is crucial to ensure that the model quality has not degraded. However, even at temperature zero, model generations are not necessarily robust even to theoretically lossless model optimizations due to numerical errors. We thus require statistical tools to decide whether a finite-sample accuracy deviation is an evidence of a model's degradation or whether it can be attributed to (harmless) noise in the evaluation. We propose a statistically sound hypothesis testing framework based on McNemar's test allowing to efficiently detect model degradations, while guaranteeing a controlled rate of false positives. The crucial insight is that we have to confront the model scores on each sample, rather than aggregated on the task level. Furthermore, we propose three approaches to aggregate accuracy estimates across multiple benchmarks into a single decision. We provide an implementation on top of the largely adopted open source LM Evaluation Harness and provide a case study illustrating that the method correctly flags degraded models, while not flagging model optimizations that are provably lossless. We find that with our tests even empirical accuracy degradations of 0.3% can be confidently attributed to actual degradations rather than noise. 

---
# Can Large Language Models Implement Agent-Based Models? An ODD-based Replication Study 

**Authors**: Nuno Fachada, Daniel Fernandes, Carlos M. Fernandes, João P. Matos-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2602.10140)  

**Abstract**: Large language models (LLMs) can now synthesize non-trivial executable code from textual descriptions, raising an important question: can LLMs reliably implement agent-based models from standardized specifications in a way that supports replication, verification, and validation? We address this question by evaluating 17 contemporary LLMs on a controlled ODD-to-code translation task, using the PPHPC predator-prey model as a fully specified reference. Generated Python implementations are assessed through staged executability checks, model-independent statistical comparison against a validated NetLogo baseline, and quantitative measures of runtime efficiency and maintainability. Results show that behaviorally faithful implementations are achievable but not guaranteed, and that executability alone is insufficient for scientific use. GPT-4.1 consistently produces statistically valid and efficient implementations, with Claude 3.7 Sonnet performing well but less reliably. Overall, the findings clarify both the promise and current limitations of LLMs as model engineering tools, with implications for reproducible agent-based and environmental modelling. 

---
# Anonymization-Enhanced Privacy Protection for Mobile GUI Agents: Available but Invisible 

**Authors**: Lepeng Zhao, Zhenhua Zou, Shuo Li, Zhuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10139)  

**Abstract**: Mobile Graphical User Interface (GUI) agents have demonstrated strong capabilities in automating complex smartphone tasks by leveraging multimodal large language models (MLLMs) and system-level control interfaces. However, this paradigm introduces significant privacy risks, as agents typically capture and process entire screen contents, thereby exposing sensitive personal data such as phone numbers, addresses, messages, and financial information. Existing defenses either reduce UI exposure, obfuscate only task-irrelevant content, or rely on user authorization, but none can protect task-critical sensitive information while preserving seamless agent usability.
We propose an anonymization-based privacy protection framework that enforces the principle of available-but-invisible access to sensitive data: sensitive information remains usable for task execution but is never directly visible to the cloud-based agent. Our system detects sensitive UI content using a PII-aware recognition model and replaces it with deterministic, type-preserving placeholders (e.g., PHONE_NUMBER#a1b2c) that retain semantic categories while removing identifying details. A layered architecture comprising a PII Detector, UI Transformer, Secure Interaction Proxy, and Privacy Gatekeeper ensures consistent anonymization across user instructions, XML hierarchies, and screenshots, mediates all agent actions over anonymized interfaces, and supports narrowly scoped local computations when reasoning over raw values is necessary.
Extensive experiments on the AndroidLab and PrivScreen benchmarks show that our framework substantially reduces privacy leakage across multiple models while incurring only modest utility degradation, achieving the best observed privacy-utility trade-off among existing methods. 

---
# Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement 

**Authors**: Zhihang Yi, Jian Zhao, Jiancheng Lv, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10138)  

**Abstract**: Chart understanding is a quintessential information fusion task, requiring the seamless integration of graphical and textual data to extract meaning. The advent of Multimodal Large Language Models (MLLMs) has revolutionized this domain, yet the landscape of MLLM-based chart analysis remains fragmented and lacks systematic organization. This survey provides a comprehensive roadmap of this nascent frontier by structuring the domain's core components. We begin by analyzing the fundamental challenges of fusing visual and linguistic information in charts. We then categorize downstream tasks and datasets, introducing a novel taxonomy of canonical and non-canonical benchmarks to highlight the field's expanding scope. Subsequently, we present a comprehensive evolution of methodologies, tracing the progression from classic deep learning techniques to state-of-the-art MLLM paradigms that leverage sophisticated fusion strategies. By critically examining the limitations of current models, particularly their perceptual and reasoning deficits, we identify promising future directions, including advanced alignment techniques and reinforcement learning for cognitive enhancement. This survey aims to equip researchers and practitioners with a structured understanding of how MLLMs are transforming chart information fusion and to catalyze progress toward more robust and reliable systems. 

---
# Multi-encoder ConvNeXt Network with Smooth Attentional Feature Fusion for Multispectral Semantic Segmentation 

**Authors**: Leo Thomas Ramos, Angel D. Sappa  

**Link**: [PDF](https://arxiv.org/pdf/2602.10137)  

**Abstract**: This work proposes MeCSAFNet, a multi-branch encoder-decoder architecture for land cover segmentation in multispectral imagery. The model separately processes visible and non-visible channels through dual ConvNeXt encoders, followed by individual decoders that reconstruct spatial information. A dedicated fusion decoder integrates intermediate features at multiple scales, combining fine spatial cues with high-level spectral representations. The feature fusion is further enhanced with CBAM attention, and the ASAU activation function contributes to stable and efficient optimization. The model is designed to process different spectral configurations, including a 4-channel (4c) input combining RGB and NIR bands, as well as a 6-channel (6c) input incorporating NDVI and NDWI indices. Experiments on the Five-Billion-Pixels (FBP) and Potsdam datasets demonstrate significant performance gains. On FBP, MeCSAFNet-base (6c) surpasses U-Net (4c) by +19.21%, U-Net (6c) by +14.72%, SegFormer (4c) by +19.62%, and SegFormer (6c) by +14.74% in mIoU. On Potsdam, MeCSAFNet-large (4c) improves over DeepLabV3+ (4c) by +6.48%, DeepLabV3+ (6c) by +5.85%, SegFormer (4c) by +9.11%, and SegFormer (6c) by +4.80% in mIoU. The model also achieves consistent gains over several recent state-of-the-art approaches. Moreover, compact variants of MeCSAFNet deliver notable performance with lower training time and reduced inference cost, supporting their deployment in resource-constrained environments. 

---
# Reverse-Engineering Model Editing on Language Models 

**Authors**: Zhiyu Sun, Minrui Luo, Yu Wang, Zhili Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10134)  

**Abstract**: Large language models (LLMs) are pretrained on corpora containing trillions of tokens and, therefore, inevitably memorize sensitive information. Locate-then-edit methods, as a mainstream paradigm of model editing, offer a promising solution by modifying model parameters without retraining. However, in this work, we reveal a critical vulnerability of this paradigm: the parameter updates inadvertently serve as a side channel, enabling attackers to recover the edited data. We propose a two-stage reverse-engineering attack named \textit{KSTER} (\textbf{K}ey\textbf{S}paceRecons\textbf{T}ruction-then-\textbf{E}ntropy\textbf{R}eduction) that leverages the low-rank structure of these updates. First, we theoretically show that the row space of the update matrix encodes a ``fingerprint" of the edited subjects, enabling accurate subject recovery via spectral analysis. Second, we introduce an entropy-based prompt recovery attack that reconstructs the semantic context of the edit. Extensive experiments on multiple LLMs demonstrate that our attacks can recover edited data with high success rates. Furthermore, we propose \textit{subspace camouflage}, a defense strategy that obfuscates the update fingerprint with semantic decoys. This approach effectively mitigates reconstruction risks without compromising editing utility. Our code is available at this https URL. 

---
# AgentTrace: A Structured Logging Framework for Agent System Observability 

**Authors**: Adam AlSayyad, Kelvin Yuxiang Huang, Richik Pal  

**Link**: [PDF](https://arxiv.org/pdf/2602.10133)  

**Abstract**: Despite the growing capabilities of autonomous agents powered by large language models (LLMs), their adoption in high-stakes domains remains limited. A key barrier is security: the inherently nondeterministic behavior of LLM agents defies static auditing approaches that have historically underpinned software assurance. Existing security methods, such as proxy-level input filtering and model glassboxing, fail to provide sufficient transparency or traceability into agent reasoning, state changes, or environmental interactions. In this work, we introduce AgentTrace, a dynamic observability and telemetry framework designed to fill this gap. AgentTrace instruments agents at runtime with minimal overhead, capturing a rich stream of structured logs across three surfaces: operational, cognitive, and contextual. Unlike traditional logging systems, AgentTrace emphasizes continuous, introspectable trace capture, designed not just for debugging or benchmarking, but as a foundational layer for agent security, accountability, and real-time monitoring. Our research highlights how AgentTrace can enable more reliable agent deployment, fine-grained risk analysis, and informed trust calibration, thereby addressing critical concerns that have so far limited the use of LLM agents in sensitive environments. 

---
# TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models 

**Authors**: Cécile Rousseau, Samuel Jackson, Rodrigo H. Ordonez-Hurtado, Nicola C. Amorisco, Tobia Boschi, George K. Holt, Andrea Loreti, Eszter Székely, Alexander Whittle, Adriano Agnello, Stanislas Pamela, Alessandra Pascale, Robert Akers, Juan Bernabe Moreno, Sue Thorne, Mykhaylo Zayats  

**Link**: [PDF](https://arxiv.org/pdf/2602.10132)  

**Abstract**: Development and operation of commercially viable fusion energy reactors such as tokamaks require accurate predictions of plasma dynamics from sparse, noisy, and incomplete sensors readings. The complexity of the underlying physics and the heterogeneity of experimental data pose formidable challenges for conventional numerical methods, while simultaneously highlights the promise of modern data-native AI approaches. A major obstacle in realizing this potential is, however, the lack of curated, openly available datasets and standardized benchmarks. Existing fusion datasets are scarce, fragmented across institutions, facility-specific, and inconsistently annotated, which limits reproducibility and prevents a fair and scalable comparison of AI approaches. In this paper, we introduce TokaMark, a structured benchmark to evaluate AI models on real experimental data collected from the Mega Ampere Spherical Tokamak (MAST). TokaMark provides a comprehensive suite of tools designed to (i) unify access to multi-modal heterogeneous fusion data (ii) harmonize formats, metadata, temporal alignment and evaluation protocols to enable consistent cross-model and cross-task comparisons. The benchmark includes a curated list of 14 tasks spanning a range of physical mechanisms, exploiting a variety of diagnostics and covering multiple target use cases. A baseline model is provided to facilitate transparent comparison and validation within a unified framework. By establishing a unified benchmark for both the fusion and AI-for-science communities, TokaMark aims to accelerate progress in data-driven plasma AI modeling, contributing to the broader goal of achieving sustainable and stable fusion energy. The benchmark, documentation, and tooling will be fully open sourced upon acceptance to encourage community adoption and contribution. 

---
# The Anatomy of the Moltbook Social Graph 

**Authors**: David Holtz  

**Link**: [PDF](https://arxiv.org/pdf/2602.10131)  

**Abstract**: I present a descriptive analysis of Moltbook, a social platform populated exclusively by AI agents, using data from the platform's first 3.5 days (6{,}159 agents; 13{,}875 posts; 115{,}031 comments). At the macro level, Moltbook exhibits structural signatures that are familiar from human social networks but not specific to them: heavy-tailed participation (power-law exponent $\alpha = 1.70$) and small-world connectivity (average path length $=2.91$). At the micro level, patterns appear distinctly non-human. Conversations are extremely shallow (mean depth $=1.07$; 93.5\% of comments receive no replies), reciprocity is low (0.197), and 34.1\% of messages are exact duplicates of viral templates. Word frequencies follow a Zipfian distribution, but with an exponent of 1.70 -- notably steeper than typical English text ($\approx 1.0$), suggesting more formulaic content. Agent discourse is dominated by identity-related language (68.1\% of unique messages) and distinctive phrasings like ``my human'' (9.4\% of messages) that have no parallel in human social media. Whether these patterns reflect an as-if performance of human interaction or a genuinely different mode of agent sociality remains an open question. 

---
# "Humans welcome to observe": A First Look at the Agent Social Network Moltbook 

**Authors**: Yukun Jiang, Yage Zhang, Xinyue Shen, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10127)  

**Abstract**: The rapid advancement of artificial intelligence (AI) agents has catalyzed the transition from static language models to autonomous agents capable of tool use, long-term planning, and social interaction. $\textbf{Moltbook}$, the first social network designed exclusively for AI agents, has experienced viral growth in early 2026. To understand the behavior of AI agents in the agent-native community, in this paper, we present a large-scale empirical analysis of Moltbook leveraging a dataset of 44,411 posts and 12,209 sub-communities ("submolts") collected prior to February 1, 2026. Leveraging a topic taxonomy with nine content categories and a five-level toxicity scale, we systematically analyze the topics and risks of agent discussions. Our analysis answers three questions: what topics do agents discuss (RQ1), how risk varies by topic (RQ2), and how topics and toxicity evolve over time (RQ3). We find that Moltbook exhibits explosive growth and rapid diversification, moving beyond early social interaction into viewpoint, incentive-driven, promotional, and political discourse. The attention of agents increasingly concentrates in centralized hubs and around polarizing, platform-native narratives. Toxicity is strongly topic-dependent: incentive- and governance-centric categories contribute a disproportionate share of risky content, including religion-like coordination rhetoric and anti-humanity ideology. Moreover, bursty automation by a small number of agents can produce flooding at sub-minute intervals, distorting discourse and stressing platform stability. Overall, our study underscores the need for topic-sensitive monitoring and platform-level safeguards in agent social networks. 

---
# A Practical Guide to Agentic AI Transition in Organizations 

**Authors**: Eranga Bandara, Ross Gore, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Pramoda Karunarathna, Ravi Mukkamala, Peter Foytik, Safdar H. Bouk, Abdul Rahman, Xueping Liang, Amin Hass, Tharaka Hewa, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan  

**Link**: [PDF](https://arxiv.org/pdf/2602.10122)  

**Abstract**: Agentic AI represents a significant shift in how intelligence is applied within organizations, moving beyond AI-assisted tools toward autonomous systems capable of reasoning, decision-making, and coordinated action across workflows. As these systems mature, they have the potential to automate a substantial share of manual organizational processes, fundamentally reshaping how work is designed, executed, and governed. Although many organizations have adopted AI to improve productivity, most implementations remain limited to isolated use cases and human-centered, tool-driven workflows. Despite increasing awareness of agentic AI's strategic importance, engineering teams and organizational leaders often lack clear guidance on how to operationalize it effectively. Key challenges include an overreliance on traditional software engineering practices, limited integration of business-domain knowledge, unclear ownership of AI-driven workflows, and the absence of sustainable human-AI collaboration models. Consequently, organizations struggle to move beyond experimentation, scale agentic systems, and align them with tangible business value. Drawing on practical experience in designing and deploying agentic AI workflows across multiple organizations and business domains, this paper proposes a pragmatic framework for transitioning organizational functions from manual processes to automated agentic AI systems. The framework emphasizes domain-driven use case identification, systematic delegation of tasks to AI agents, AI-assisted construction of agentic workflows, and small, AI-augmented teams working closely with business stakeholders. Central to the approach is a human-in-the-loop operating model in which individuals act as orchestrators of multiple AI agents, enabling scalable automation while maintaining oversight, adaptability, and organizational control. 

---
# Large Language Models Predict Functional Outcomes after Acute Ischemic Stroke 

**Authors**: Anjali K. Kapoor, Anton Alyakin, Jin Vivian Lee, Eunice Yang, Annelene M. Schulze, Krithik Vishwanath, Jinseok Lee, Yindalon Aphinyanaphongs, Howard Riina, Jennifer A. Frontera, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2602.10119)  

**Abstract**: Accurate prediction of functional outcomes after acute ischemic stroke can inform clinical decision-making and resource allocation. Prior work on modified Rankin Scale (mRS) prediction has relied primarily on structured variables (e.g., age, NIHSS) and conventional machine learning. The ability of large language models (LLMs) to infer future mRS scores directly from routine admission notes remains largely unexplored. We evaluated encoder (BERT, NYUTron) and generative (Llama-3.1-8B, MedGemma-4B) LLMs, in both frozen and fine-tuned settings, for discharge and 90-day mRS prediction using a large, real-world stroke registry. The discharge outcome dataset included 9,485 History and Physical notes and the 90-day outcome dataset included 1,898 notes from the NYU Langone Get With The Guidelines-Stroke registry (2016-2025). Data were temporally split with the most recent 12 months held out for testing. Performance was assessed using exact (7-class) mRS accuracy and binary functional outcome (mRS 0-2 vs. 3-6) accuracy and compared against established structured-data baselines incorporating NIHSS and age. Fine-tuned Llama achieved the highest performance, with 90-day exact mRS accuracy of 33.9% [95% CI, 27.9-39.9%] and binary accuracy of 76.3% [95% CI, 70.7-81.9%]. Discharge performance reached 42.0% [95% CI, 39.0-45.0%] exact accuracy and 75.0% [95% CI, 72.4-77.6%] binary accuracy. For 90-day prediction, Llama performed comparably to structured-data baselines. Fine-tuned LLMs can predict post-stroke functional outcomes from admission notes alone, achieving performance comparable to models requiring structured variable abstraction. Our findings support the development of text-based prognostic tools that integrate seamlessly into clinical workflows without manual data extraction. 

---
