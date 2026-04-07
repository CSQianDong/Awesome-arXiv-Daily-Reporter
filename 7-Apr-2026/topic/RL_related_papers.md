# Retrieval Augmented Conversational Recommendation with Reinforcement Learning 

**Authors**: Zhenrui Yue, Honglei Zhuang, Zhen Qin, Zhankui He, Huimin Zeng, Julian McAuley, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04457)  

**Abstract**: Large language models (LLMs) exhibit enhanced capabilities in language understanding and generation. By utilizing their embedded knowledge, LLMs are increasingly used as conversational recommender systems (CRS), achieving improved performance across diverse scenarios. However, existing LLM-based methods rely on pretrained knowledge without external retrieval mechanisms for novel items. Additionally, the lack of a unified corpus poses challenges for integrating retrieval augmentation into CRS. Motivated by these challenges, we present RAR, a novel two-stage retrieval augmented conversational recommendation framework that aligns retrieval and generation to enhance both performance and factuality. To support this framework and provide a unified corpus, we construct a large-scale movie corpus, comprising over 300k movies with rich metadata, such as titles, casts and plot summaries. Leveraging this data, our primary contribution is RAR, the first framework to departs from standard two-stage CRS by dynamically bridging retrieval and generation. First, a retriever model generates candidate items based on user history; in the subsequent stage, an LLM refines the recommendations by incorporating conversational context with retrieved results. In addition, we introduce a novel reinforcement learning (RL) method that leverages LLM feedback to iteratively update the retriever. By creating a collaborative feedback loop that reinforces sampled candidate sets with higher ranking metrics, RAR effectively mitigates the misalignment between the retrieval and generation stages. Furthermore, grounding the LLM in factual metadata allows our RL-driven approach to capture subtle user intentions and generate context-aware recommendations with reduced hallucinations. We validate our approach through extensive experiments on multiple benchmarks, where RAR consistently outperforms state-of-the-art baseline methods. 

---
# User Simulator-Guided Multi-Turn Preference Optimization for Reasoning LLM-based Conversational Recommendation 

**Authors**: Xingyuan Xiang, Xiangchen Pan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03671)  

**Abstract**: Conversational Recommender Systems (CRSs) leverage natural language interactions for personalized recommendation, yet information-scarce dialogue histories and single-turn recommendation paradigms may severely hinder accurate modeling of complex user preferences. To alleviate this issue, recent studies have introduced LLM-based user simulators, which generate natural language feedback and perform simulated multi-turn interactions to assist recommendation. Nevertheless, since simulators cannot access true user preference labels during inference, their feedback may deviate from actual user interests, causing errors to accumulate over multiple interactions and severely affecting the generalization of the recommender. Inspired by the multi-step reasoning capabilities of LLMs and the effectiveness of reinforcement learning in policy optimization, we propose SMTPO, a user simulator-guided multi-turn preference optimization conversational recommendation framework. To align simulator-generated feedback with true user preferences in the absence of explicit labels, we enhance feedback quality via multi-task supervised fine-tuning (SFT), enabling the simulator to better reflect users' complex and diverse needs. To address the challenge of biased feedback destabilizing multi-turn optimization, we first allow the reasoning LLM-based recommender to learn preference reasoning and recommendation patterns through SFT and then employ reinforcement learning with fine-grained reward design to progressively align with true user preferences, improving recommendation performance. Extensive experiments on public datasets demonstrate the effectiveness and transferability of our method. 

---
# Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation 

**Authors**: Hengrui Gu, Xiaotian Han, Yujing Bian, Kaixiong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04894)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning capabilities of large language models (LLMs). However, it faces a fundamental limitation termed \textit{restricted exploration}, where the policy rapidly converges to a narrow set of solutions. While entropy regularization is a popular approach used to sustain exploration, it often proves unreliable for LLMs, suffering from high hyperparameter sensitivity and yielding only marginal performance gains. Motivated by these inefficiencies, we propose to rethink the relationship between policy entropy and exploration. By deriving a parametric formulation of group-relative advantage estimation and analyzing entropy dynamics, we conceptually decompose policy entropy into \textit{informative entropy}, which preserves diverse solution paths, and \textit{spurious entropy}, which erodes reasoning patterns. Our analysis reveals that, in contrast to blind maximization, effective exploration requires \textit{entropy refinement}-a mechanism implicitly embedded in group-relative advantage estimation that sustains informative entropy on positive rollouts while suppressing spurious entropy on negative ones. Guided by this insight, we propose \textbf{AsymGRPO}, an exploratory framework that explicitly decouples the modulation of positive and negative rollouts. This allows for independent control over the preservation of informative entropy and the suppression of spurious noise. Extensive experiments demonstrate that AsymGRPO achieves superior performance compared to strong baselines and exhibits the potential to synergize with existing entropy regularization methods. 

---
# MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation 

**Authors**: Zhixiang Lu, Chong Zhang, Chenyu Xue, Angelos Stefanidis, Chong Li, Jionglong Su, Zhengyong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04839)  

**Abstract**: Neural machine translation (NMT) from Chinese to low-resource Southeast Asian languages remains severely constrained by the extreme scarcity of clean parallel corpora and the pervasive noise in existing mined data. This chronic shortage not only impedes effective model training but also sustains a large performance gap with high-resource directions, leaving millions of speakers of languages such as Lao, Burmese, and Tagalog with persistently low-quality translation systems despite recent advances in large multilingual models. We introduce \textbf{M}ultilingual \textbf{E}xpert-\textbf{R}eward \textbf{I}nformed \textbf{T}uning (\textbf{MERIT}), a unified translation framework that transforms the traditional English-centric ALT benchmark into a Chinese-centric evaluation suite for five Southeast Asian low-resource languages (LRLs). Our framework combines language-specific token prefixing (LTP) with supervised fine-tuning (SFT) and a novel group relative policy optimization (GRPO) guided by the semantic alignment reward (SAR). These results confirm that, in LRL{\textrightarrow}Chinese translation, targeted data curation and reward-guided optimization dramatically outperform mere model scaling. 

---
# DeonticBench: A Benchmark for Reasoning over Rules 

**Authors**: Guangyao Dou, Luis Brena, Akhil Deo, William Jurayj, Jingyu Zhang, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2604.04443)  

**Abstract**: Reasoning with complex, context-specific rules remains challenging for large language models (LLMs). In legal and policy settings, this manifests as deontic reasoning: reasoning about obligations, permissions, and prohibitions under explicit rules. While many recent benchmarks emphasize short-context mathematical reasoning, fewer focus on long-context, high-stakes deontic reasoning. To address this gap, we introduce DEONTICBENCH, a benchmark of 6,232 tasks across U.S. federal taxes, airline baggage policies, U.S. immigration administration, and U.S. state housing law. These tasks can be approached in multiple ways, including direct reasoning in language or with the aid of symbolic computation. Besides free-form chain-of-thought reasoning, DEONTICBENCH enables an optional solver-based workflow in which models translate statutes and case facts into executable Prolog, leading to formal problem interpretations and an explicit program trace. We release reference Prolog programs for all instances. Across frontier LLMs and coding models, best hard-subset performance reaches only 44.4% on SARA Numeric and 46.6 macro-F1 on Housing. We further study training with supervised fine-tuning and reinforcement learning for symbolic program generation. Although training improves Prolog generation quality, current RL methods still fail to solve these tasks reliably. Overall, DEONTICBENCH provides a benchmark for studying context-grounded rule reasoning in real-world domains under both symbolic and non-symbolic settings. 

---
# How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models 

**Authors**: Gregory N. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2604.04385)  

**Abstract**: We identify a recurring sparse routing mechanism in alignment-trained language models: a gate attention head reads detected content and triggers downstream amplifier heads that boost the signal toward refusal. Using political censorship and safety refusal as natural experiments, we trace this mechanism across 9 models from 6 labs, all validated on corpora of 120 prompt pairs. The gate head passes necessity and sufficiency interchange tests (p < 0.001, permutation null), and core amplifier heads are stable under bootstrap resampling (Jaccard 0.92-1.0). Three same-generation scaling pairs show that routing distributes at scale (ablation up to 17x weaker) while remaining detectable by interchange. By modulating the detection-layer signal, we continuously control policy strength from hard refusal through steering to factual compliance, with routing thresholds that vary by topic. The circuit also reveals a structural separation between intent recognition and policy routing: under cipher encoding, the gate head's routing contribution collapses (78% in Phi-4 at n=120) while the model responds with puzzle-solving rather than refusal. The routing mechanism never fires, even though probe scores at deeper layers indicate the model begins to represent the harmful content. This asymmetry is consistent with different robustness properties of pretraining and post-training: broad semantic understanding versus narrower policy binding that generalizes less well under input transformation. 

---
# Structured Causal Video Reasoning via Multi-Objective Alignment 

**Authors**: Zinuo Li, Yongxin Guo, Jun Liu, Jiawei Zhan, Xi Jiang, Chengjie Wang, Mohammed Bennamoun, Farid Boussaid, Feng Zheng, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2604.04415)  

**Abstract**: Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical visual evidence is embedded in verbose textual descriptions and temporal causality is often weakly modeled. This leads to inefficient processes and fragile causal inference. To bridge this cognitive gap, we propose constructing a compact representation of salient events and their causal relationships, which we name Structured Event Facts, prior to the reasoning stage. This structured prior serves as an explicit constraint to promote concise and causally grounded reasoning, while also making intermediate evidence easier to verify. To effectively train models on such structured facts, we introduce CausalFact-60K and a four-stage training pipeline comprising facts alignment, format warm-start, thinking warm-start, and reinforcement learning-based post-training. During RL stage, we find that this framework introduces competing objectives, as structural completeness and causal fidelity must be balanced against reasoning length, making it difficult to optimize. We address this challenge by formulating the optimization as a Multi-Objective Reinforcement Learning (MORL) problem and explicitly optimizing toward the Pareto-Frontier to balance these trade-offs. As a result, we introduce Factum-4B, which yields more reliable reasoning and delivers stronger performance on challenging video understanding tasks requiring fine-grained temporal inference. 

---
# DARE: Diffusion Large Language Models Alignment and Reinforcement Executor 

**Authors**: Jingyi Yang, Yuxian Jiang, Xuhao Hu, Shuang Cheng, Biqing Qi, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04215)  

**Abstract**: Diffusion large language models (dLLMs) are emerging as a compelling alternative to dominant autoregressive models, replacing strictly sequential token generation with iterative denoising and parallel generation dynamics. However, their open-source ecosystem remains fragmented across model families and, in particular, across post-training pipelines, where reinforcement learning objectives, rollout implementations and evaluation scripts are often released as paper-specific codebases. This fragmentation slows research iteration, raises the engineering burden of reproduction, and makes fair comparison across algorithms difficult. We present \textbf{DARE} (\textbf{d}LLMs \textbf{A}lignment and \textbf{R}einforcement \textbf{E}xecutor), an open framework for post-training and evaluating dLLMs. Built on top of verl~\cite{sheng2024hybridflow} and OpenCompass~\cite{2023opencompass}, DARE unifies supervised fine-tuning, parameter-efficient fine-tuning, preference optimization, and dLLM-specific reinforcement learning under a shared execution stack for both masked and block diffusion language models. Across representative model families including LLaDA, Dream, SDAR, and LLaDA2.x, DARE provides broad algorithmic coverage, reproducible benchmark evaluation, and practical acceleration. Extensive empirical results position that DARE serves as a reusable research substrate for developing, comparing, and deploying post-training methods for current and emerging dLLMs. 

---
# Many Preferences, Few Policies: Towards Scalable Language Model Personalization 

**Authors**: Cheol Woo Kum, Jai Moondra, Roozbeh Nahavandi, Andrew Perrault, Milind Tambe, Swati Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.04144)  

**Abstract**: The holy grail of LLM personalization is a single LLM for each user, perfectly aligned with that user's preferences. However, maintaining a separate LLM per user is impractical due to constraints on compute, memory, and system complexity. We address this challenge by developing a principled method for selecting a small portfolio of LLMs that captures representative behaviors across heterogeneous users. We model user preferences across multiple traits (e.g., safety, humor, brevity) through a multi-dimensional weight vector. Given reward functions across these dimensions, our algorithm PALM (Portfolio of Aligned LLMs) generates a small portfolio of LLMs such that, for any weight vector, the portfolio contains a near-optimal LLM for the corresponding scalarized objective. To the best of our knowledge, this is the first result that provides theoretical guarantees on both the size and approximation quality of LLM portfolios for personalization. It characterizes the trade-off between system cost and personalization, as well as the diversity of LLMs required to cover the landscape of user preferences. We provide empirical results that validate these guarantees and demonstrate greater output diversity over common baselines. 

---
# Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression 

**Authors**: Lingjie Zeng, Xiaofan Chen, Yanbo Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04120)  

**Abstract**: Long chain-of-thought (Long-CoT) reasoning models have motivated a growing body of work on compressing reasoning traces to reduce inference cost, yet existing evaluations focus almost exclusively on task accuracy and token savings. Trustworthiness properties, whether acquired or reinforced through post-training, are encoded in the same parameter space that compression modifies. This means preserving accuracy does not, a priori, guarantee preserving trustworthiness. We conduct the first systematic empirical study of how CoT compression affects model trustworthiness, evaluating multiple models of different scales along three dimensions: safety, hallucination resistance, and multilingual robustness. Under controlled comparisons, we find that CoT compression frequently introduces trustworthiness regressions and that different methods exhibit markedly different degradation profiles across dimensions. To enable fair comparison across bases, we propose a normalized efficiency score for each dimension that reveals how naïve scalar metrics can obscure trustworthiness trade-offs. As an existence proof, we further introduce an alignment-aware DPO variant that reduces CoT length by 19.3\% on reasoning benchmarks with substantially smaller trustworthiness loss. Our findings suggest that CoT compression should be optimized not only for efficiency but also for trustworthiness, treating both as equally important design constraints. 

---
# Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution 

**Authors**: Jacob Dineen, Aswin RRV, Zhikun Xu, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03472)  

**Abstract**: Co-evolutionary self-play, where one language model generates problems and another solves them, promises autonomous curriculum learning without human supervision. In practice, the proposer quickly converges to a narrow distribution of problems that satisfy the reward function. This diversity collapse renders the curriculum uninformative for the solver, stalling the co-evolutionary loop. We introduce vocabulary dropout, a random mask applied to the proposer's output logits during both policy training and curriculum generation, as a lightweight mechanism to sustain diversity. The mask is hard and non-stationary, preventing the proposer from locking into fixed token sequences. Training Qwen3-4B and Qwen3-8B on mathematical reasoning via R-Zero, we find that vocabulary dropout sustains proposer diversity across lexical, semantic, and functional metrics throughout training, and yields solver improvements averaging +4.4 points at 8B, with the largest gains on competition-level benchmarks. Our findings suggest that explicit action-space constraints, analogous to the structural role that game rules play in classical self-play, can help sustain productive co-evolution in language. Vocabulary dropout is one simple instantiation of this principle. 

---
# Self-Execution Simulation Improves Coding Models 

**Authors**: Gallil Maimon, Ori Yoran, Felix Kreuk, Michael Hassid, Gal Cohen, Pierre Chambon, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03253)  

**Abstract**: A promising research direction in enabling LLMs to generate consistently correct code involves addressing their inability to properly estimate program execution, particularly for code they generate. In this work, we demonstrate that Code LLMs can be trained to simulate program execution in a step-by-step manner and that this capability can be leveraged to improve competitive programming performance. Our approach combines supervised fine-tuning on natural language execution traces, textual explanations grounded in true execution, with reinforcement learning using verifiable rewards. We introduce two complementary objectives: output prediction given code and inputs, and solving competitive programming tasks with either ground-truth or self-predicted execution feedback. These objectives enable models to perform self-verification over multiple candidate solutions, and iterative self-fixing by simulating test execution. Across multiple competitive programming benchmarks, our method yields consistent improvements over standard reasoning approaches. We further present ablations and analysis to elucidate the role of execution simulation and its limitations. 

---
# Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems 

**Authors**: Justin Chih-Yao Chen, Archiki Prasad, Zaid Khan, Joykirat Singh, Runchu Tian, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.04767)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has improved the reasoning abilities of LLMs, yet a fundamental limitation remains: models cannot learn from problems that are too difficult to solve under their current policy, as these yield no meaningful reward signal. We propose a simple yet effective solution based on task reformulation. We transform challenging open-ended problems into cognitively simpler variants -- such as multiple-choice and cloze formats -- that preserve the original answer while reducing the effective search space and providing denser learning signals. These reformulations span a spectrum from discriminative to generative tasks, which we exploit to bootstrap learning: models first learn from structured, easier formats, and this knowledge transfers back to improve performance on the original open-ended problems. Building on this insight, we introduce Cog-DRIFT, a framework that constructs reformulated variants and organizes them into an adaptive curriculum based on difficulty. Training progresses from easier to harder formats, enabling the model to learn from problems that previously yielded zero signal under standard RL post-training. Cog-DRIFT not only improves on the originally unsolvable hard problems (absolute +10.11% for Qwen and +8.64% for Llama) but also generalizes well to other held-out datasets. Across 2 models and 6 reasoning benchmarks, our method consistently outperforms standard GRPO and strong guided-exploration baselines. On average, Cog-DRIFT shows +4.72% (Qwen) and +3.23% (Llama) improvements over the second-best baseline. We further show that Cog-DRIFT improves pass@k at test time, and the curriculum improves sample efficiency. Overall, our results highlight task reformulation and curriculum learning as an effective paradigm for overcoming the exploration barrier in LLM post-training. 

---
# QED-Nano: Teaching a Tiny Model to Prove Hard Theorems 

**Authors**: LM-Provers, Yuxiao Qu, Amrith Setlur, Jasper Dekoninck, Edward Beeching, Jia Li, Ian Wu, Lewis Tunstall, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04898)  

**Abstract**: Proprietary AI systems have recently demonstrated impressive capabilities on complex proof-based problems, with gold-level performance reported at the 2025 International Mathematical Olympiad (IMO). However, the training pipelines behind these systems remain largely undisclosed, and their reliance on large "internal" models and scaffolds makes them expensive to run, difficult to reproduce, and hard to study or improve upon. This raises a central question: can small, open models also be trained to achieve competitive reasoning performance on difficult Olympiad-level math? In this paper, we answer this question by building QED-Nano, a 4B model post-trained for Olympiad-level proofs. Our training recipe has three stages: (1) supervised fine-tuning to imbue good proof-writing styles by distilling from DeepSeek-Math-V2, (2) reinforcement learning (RL) with rubric-based rewards, and (3) expanding RL with a reasoning cache, which decomposes long proofs into iterative summarize-and-refine cycles and enables stronger test-time reasoning. QED-Nano surpasses the proof-generation performance of much larger open models, including Nomos-1 and GPT-OSS-120B, and approaches the performance of proprietary models like Gemini 3 Pro, at a fraction of the inference cost. To support further research on open mathematical reasoning, we release the full QED-Nano pipeline, including the QED-Nano and QED-Nano-SFT models, the FineProofs-SFT and FineProofs-RL datasets, and the training and evaluation code. 

---
# One Model for All: Multi-Objective Controllable Language Models 

**Authors**: Qiang He, Yucheng Yang, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04497)  

**Abstract**: Aligning large language models (LLMs) with human preferences is critical for enhancing LLMs' safety, helpfulness, humor, faithfulness, etc. Current reinforcement learning from human feedback (RLHF) mainly focuses on a fixed reward learned from average human ratings, which may weaken the adaptability and controllability of varying preferences. However, creating personalized LLMs requires aligning LLMs with individual human preferences, which is non-trivial due to the scarce data per user and the diversity of user preferences in multi-objective trade-offs, varying from emphasizing empathy in certain contexts to demanding efficiency and precision in others. Can we train one LLM to produce personalized outputs across different user preferences on the Pareto front? In this paper, we introduce Multi-Objective Control (MOC), which trains a single LLM to directly generate responses in the preference-defined regions of the Pareto front. Our approach introduces multi-objective optimization (MOO) principles into RLHF to train an LLM as a preference-conditioned policy network. We improve the computational efficiency of MOC by applying MOO at the policy level, enabling us to fine-tune a 7B-parameter model on a single A6000 GPU. Extensive experiments demonstrate the advantages of MOC over baselines in three aspects: (i) controllability of LLM outputs w.r.t. user preferences on the trade-off among multiple rewards; (ii) quality and diversity of LLM outputs, measured by the hyper-volume of multiple solutions achieved; and (iii) generalization to unseen preferences. These results highlight MOC's potential for real-world applications requiring scalable and customizable LLMs. 

---
# Memory Intelligence Agent 

**Authors**: Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.04503)  

**Abstract**: Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while suffering from key limitations of ineffective memory evolution and increasing storage and retrieval costs. To address these problems, we propose a novel Memory Intelligence Agent (MIA) framework, consisting of a Manager-Planner-Executor architecture. Memory Manager is a non-parametric memory system that can store compressed historical search trajectories. Planner is a parametric memory agent that can produce search plans for questions. Executor is another agent that can search and analyze information guided by the search plan. To build the MIA framework, we first adopt an alternating reinforcement learning paradigm to enhance cooperation between the Planner and the Executor. Furthermore, we enable the Planner to continuously evolve during test-time learning, with updates performed on-the-fly alongside inference without interrupting the reasoning process. Additionally, we establish a bidirectional conversion loop between parametric and non-parametric memories to achieve efficient memory evolution. Finally, we incorporate a reflection and an unsupervised judgment mechanisms to boost reasoning and self-evolution in the open world. Extensive experiments across eleven benchmarks demonstrate the superiority of MIA. 

---
# PSY-STEP: Structuring Therapeutic Targets and Action Sequences for Proactive Counseling Dialogue Systems 

**Authors**: Jihyun Lee, Yejin Min, Yejin Jeon, SungJun Yang, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04448)  

**Abstract**: Cognitive Behavioral Therapy (CBT) aims to identify and restructure automatic negative thoughts pertaining to involuntary interpretations of events, yet existing counseling agents struggle to identify and address them in dialogue settings. To bridge this gap, we introduce STEP, a dataset that models CBT counseling by explicitly reflecting automatic thoughts alongside dynamic, action-level counseling sequences. Using this dataset, we train STEPPER, a counseling agent that proactively elicits automatic thoughts and executes cognitively grounded interventions. To further enhance both decision accuracy and empathic responsiveness, we refine STEPPER through preference learning based on simulated, synthesized counseling sessions. Extensive CBT-aligned evaluations show that STEPPER delivers more clinically grounded, coherent, and personalized counseling compared to other strong baseline models, and achieves higher counselor competence without inducing emotional disruption. 

---
# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training 

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03675)  

**Abstract**: In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts are under-utilized during training, and supervision is typically available only at the final answer, resulting in severe reward sparsity. We present Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards (PRAISE), a framework for improving both data efficiency and credit assignment in agentic search training. Given a complete search trajectory, PRAISE extracts prefix states at different search turns, elicits intermediate answers from them, and uses these prefixes both to construct additional training trajectories and to derive step-level rewards from performance differences across prefixes. Our method uses a single shared model for both search policy learning and prefix answer evaluation, enabling joint optimization without extra human annotations or a separate reward model. Experiments on multi-hop QA benchmarks show that PRAISE consistently improves performance over strong baselines. 

---
# BioAlchemy: Distilling Biological Literature into Reasoning-Ready Reinforcement Learning Training Data 

**Authors**: Brian Hsu, Ozan Gökdemir, Carlo Siebenschuh, Bruce Parrello, Neil Getty, Thomas S. Brettin, Rick L. Stevens, Ian T. Foster, Nicholas Chia, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03506)  

**Abstract**: Despite the large corpus of biology training text, the impact of reasoning models on biological research generally lags behind math and coding. In this work, we show that biology questions from current large-scale reasoning datasets do not align well with modern research topic distributions in biology, and that this topic imbalance may negatively affect performance. In addition, we find that methods for extracting challenging and verifiable research problems from biology research text are a critical yet underdeveloped ingredient in applying reinforcement learning for better performance on biology research tasks. We introduce BioAlchemy, a pipeline for sourcing a diverse set of verifiable question-and-answer pairs from a scientific corpus of biology research text. We curate BioAlchemy-345K, a training dataset containing over 345K scientific reasoning problems in biology. Then, we demonstrate how aligning our dataset to the topic distribution of modern scientific biology can be used with reinforcement learning to improve reasoning performance. Finally, we present BioAlchemist-8B, which improves over its base reasoning model by 9.12% on biology benchmarks. These results demonstrate the efficacy of our approach for developing stronger scientific reasoning capabilities in biology. The BioAlchemist-8B model is available at: this https URL. 

---
# Vero: An Open RL Recipe for General Visual Reasoning 

**Authors**: Gabriel Sarch, Linrong Cai, Qunzhong Wang, Haoyang Wu, Danqi Chen, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04917)  

**Abstract**: What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.7-5.5 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released. 

---
# APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs 

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.04261)  

**Abstract**: Aligning large language models (LLMs) with diverse human preferences requires pluralistic alignment, where a single model must respect the values of multiple distinct groups simultaneously. In federated reinforcement learning from human feedback (FedRLHF), these groups align a shared policy without centralizing preference data, which makes fair reward aggregation essential. Existing aggregation methods exhibit clear trade offs: average based aggregation systematically under aligns worst performing groups, while min aggregation prioritizes worst group performance at the cost of overall alignment. We propose APPA, an Adaptive Preference Pluralistic Alignment framework that dynamically reweights group level rewards based on historical alignment rewards. Our approach prioritizes under aligned groups without degrading well aligned ones, while requiring no access to raw preference data. Integrated into a proximal policy optimization (PPO) based FedRLHF pipeline and evaluated on GLOBALQA and OQA across three model families (Gemma 2 2B, Llama 3.2 3B, Qwen3 0.6B), APPA achieves strong fairness alignment trade offs, improving worst group alignment by up to 28% over average aggregation while maintaining higher overall alignment than min aggregation across most configurations. 

---
# Can LLMs Learn to Reason Robustly under Noisy Supervision? 

**Authors**: Shenzhi Yang, Guangcheng Zhu, Bowen Song, Sharon Li, Haobo Wang, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03993)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains reasoning models that rely on abundant perfect labels, but its vulnerability to unavoidable noisy labels due to expert scarcity remains critically underexplored. In this work, we take the first step toward a systematic analysis of noisy label mechanisms in RLVR. In contrast to supervised classification, most RLVR algorithms incorporate a rollout-based condition: a label's influence on training is contingent on whether the current policy can generate rollouts that realize it, a property that naturally extends to noisy labels. Based on this observation, we distinguish two types of noise: inactive noisy labels, which reduce data efficiency, and active noisy labels, which are reinforced and risk skewing the model toward incorrect distributions. From experiments on training with noisy samples, we identify an Early Correctness Coherence phenomenon: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. Motivated by this dynamic, we propose Online Label Refinement (OLR), which progressively corrects potentially noisy labels with majority-voted answers when two conditions hold: a positive slope in the majority answer's rollout pass rate and stable historical consistency across updates, enabling gradual self-correction as the policy improves. We evaluate OLR on six in-distribution mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). Across noise ratios from 0.1 to 0.9, OLR consistently improves robustness under both inactive and active noisy-label settings, achieving average gains of 3.6% to 3.9% on in-distribution benchmarks and 3.3% to 4.6% on out-of-distribution evaluations. 

---
# Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling 

**Authors**: Yunyao Yu, Zhengxian Wu, Zhuohong Chen, Hangrui Xu, Zirui Liao, Xiangwen Deng, Zhifang Liu, Senyuan Shi, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03647)  

**Abstract**: In the unsupervised self-evolution of Multimodal Large Language Models, the quality of feedback signals during post-training is pivotal for stable and effective learning. However, existing self-evolution methods predominantly rely on majority voting to select the most frequent output as the pseudo-golden answer, which may stem from the model's intrinsic biases rather than guaranteeing the objective correctness of the reasoning paths. To counteract the degradation, we propose \textbf{C}ontinuous \textbf{S}oftened \textbf{R}etracing re\textbf{S}ampling (\textbf{CSRS}) in MLLM self-evolution. Specifically, we introduce a Retracing Re-inference Mechanism (\textbf{RRM}) that the model re-inferences from anchor points to expand the exploration of long-tail reasoning paths. Simultaneously, we propose Softened Frequency Reward (\textbf{SFR}), which replaces binary rewards with continuous signals, calibrating reward based on the answers' frequency across sampled reasoning sets. Furthermore, incorporated with Visual Semantic Perturbation (\textbf{VSP}), CSRS ensures the model prioritizes mathematical logic over visual superficiality. Experimental results demonstrate that CSRS significantly enhances the reasoning performance of Qwen2.5-VL-7B on benchmarks such as MathVision. We achieve state-of-the-art (SOTA) results in unsupervised self-evolution on geometric tasks. Our code is avaible at this https URL. 

---
