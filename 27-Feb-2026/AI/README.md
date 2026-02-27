# Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks 

**Authors**: Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, Stefan Zohren  

**Link**: [PDF](https://arxiv.org/pdf/2602.23330)  

**Abstract**: The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world workflows, which can lead to degraded inference performance and less transparent decision-making. Therefore, we propose a multi-agent LLM trading framework that explicitly decomposes investment analysis into fine-grained tasks, rather than providing coarse-grained instructions. We evaluate the proposed framework using Japanese stock data, including prices, financial statements, news, and macro information, under a leakage-controlled backtesting setting. Experimental results show that fine-grained task decomposition significantly improves risk-adjusted returns compared to conventional coarse-grained designs. Crucially, further analysis of intermediate agent outputs suggests that alignment between analytical outputs and downstream decision preferences is a critical driver of system performance. Moreover, we conduct standard portfolio optimization, exploiting low correlation with the stock index and the variance of each system's output. This approach achieves superior performance. These findings contribute to the design of agent structure and task configuration when applying LLM agents to trading systems in practical settings. 

---
# LLM Novice Uplift on Dual-Use, In Silico Biology Tasks 

**Authors**: Chen Bo Calvin Zhang, Christina Q. Knight, Nicholas Kruus, Jason Hausenloy, Pedro Medeiros, Nathaniel Li, Aiden Kim, Yury Orlovskiy, Coleman Breen, Bryce Cai, Jasper Götting, Andrew Bo Liu, Samira Nedungadi, Paula Rodriguez, Yannis Yiming He, Mohamed Shaaban, Zifan Wang, Seth Donoughe, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2602.23329)  

**Abstract**: Large language models (LLMs) perform increasingly well on biology benchmarks, but it remains unclear whether they uplift novice users -- i.e., enable humans to perform better than with internet-only resources. This uncertainty is central to understanding both scientific acceleration and dual-use risk. We conducted a multi-model, multi-benchmark human uplift study comparing novices with LLM access versus internet-only access across eight biosecurity-relevant task sets. Participants worked on complex problems with ample time (up to 13 hours for the most involved tasks). We found that LLM access provided substantial uplift: novices with LLMs were 4.16 times more accurate than controls (95% CI [2.63, 6.87]). On four benchmarks with available expert baselines (internet-only), novices with LLMs outperformed experts on three of them. Perhaps surprisingly, standalone LLMs often exceeded LLM-assisted novices, indicating that users were not eliciting the strongest available contributions from the LLMs. Most participants (89.6%) reported little difficulty obtaining dual-use-relevant information despite safeguards. Overall, LLMs substantially uplift novices on biological tasks previously reserved for trained practitioners, underscoring the need for sustained, interactive uplift evaluations alongside traditional benchmarks. 

---
# Generalized Rapid Action Value Estimation in Memory-Constrained Environments 

**Authors**: Aloïs Rautureau, Tristan Cazenave, Éric Piette  

**Link**: [PDF](https://arxiv.org/pdf/2602.23318)  

**Abstract**: Generalized Rapid Action Value Estimation (GRAVE) has been shown to be a strong variant within the Monte-Carlo Tree Search (MCTS) family of algorithms for General Game Playing (GGP). However, its reliance on storing additional win/visit statistics at each node makes its use impractical in memory-constrained environments, thereby limiting its applicability in practice. In this paper, we introduce the GRAVE2, GRAVER and GRAVER2 algorithms, which extend GRAVE through two-level search, node recycling, and a combination of both techniques, respectively. We show that these enhancements enable a drastic reduction in the number of stored nodes while matching the playing strength of GRAVE. 

---
# Invariant Transformation and Resampling based Epistemic-Uncertainty Reduction 

**Authors**: Sha Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23315)  

**Abstract**: An artificial intelligence (AI) model can be viewed as a function that maps inputs to outputs in high-dimensional spaces. Once designed and well trained, the AI model is applied for inference. However, even optimized AI models can produce inference errors due to aleatoric and epistemic uncertainties. Interestingly, we observed that when inferring multiple samples based on invariant transformations of an input, inference errors can show partial independences due to epistemic uncertainty. Leveraging this insight, we propose a "resampling" based inferencing that applies to a trained AI model with multiple transformed versions of an input, and aggregates inference outputs to a more accurate result. This approach has the potential to improve inference accuracy and offers a strategy for balancing model size and performance. 

---
# The logic of KM belief update is contained in the logic of AGM belief revision 

**Authors**: Giacomo Bonanno  

**Link**: [PDF](https://arxiv.org/pdf/2602.23302)  

**Abstract**: For each axiom of KM belief update we provide a corresponding axiom in a modal logic containing three modal operators: a unimodal belief operator $B$, a bimodal conditional operator $>$ and the unimodal necessity operator $\square$. We then compare the resulting logic to the similar logic obtained from converting the AGM axioms of belief revision into modal axioms and show that the latter contains the former. Denoting the latter by $\mathcal L_{AGM}$ and the former by $\mathcal L_{KM}$ we show that every axiom of $\mathcal L_{KM}$ is a theorem of $\mathcal L_{AGM}$. Thus AGM belief revision can be seen as a special case of KM belief update. For the strong version of KM belief update we show that the difference between $\mathcal L_{KM}$ and $\mathcal L_{AGM}$ can be narrowed down to a single axiom, which deals exclusively with unsurprising information, that is, with formulas that were not initially disbelieved. 

---
# ODEBrain: Continuous-Time EEG Graph for Modeling Dynamic Brain Networks 

**Authors**: Haohui Jia, Zheng Chen, Lingwei Zhu, Rikuto Kotoge, Jathurshan Pradeepkumar, Yasuko Matsubara, Jimeng Sun, Yasushi Sakurai, Takashi Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2602.23285)  

**Abstract**: Modeling neural population dynamics is crucial for foundational neuroscientific research and various clinical applications. Conventional latent variable methods typically model continuous brain dynamics through discretizing time with recurrent architecture, which necessarily results in compounded cumulative prediction errors and failure of capturing instantaneous, nonlinear characteristics of EEGs. We propose ODEBRAIN, a Neural ODE latent dynamic forecasting framework to overcome these challenges by integrating spatio-temporal-frequency features into spectral graph nodes, followed by a Neural ODE modeling the continuous latent dynamics. Our design ensures that latent representations can capture stochastic variations of complex brain states at any given time point. Extensive experiments verify that ODEBRAIN can improve significantly over existing methods in forecasting EEG dynamics with enhanced robustness and generalization capabilities. 

---
# CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays 

**Authors**: Hyungyung Lee, Hangyul Yoon, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.23276)  

**Abstract**: Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings. 

---
# Evaluating Stochasticity in Deep Research Agents 

**Authors**: Haotian Zhai, Elias Stengel-Eskin, Pratik Patil, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2602.23271)  

**Abstract**: Deep Research Agents (DRAs) are promising agentic systems that gather and synthesize information to support research across domains such as financial decision-making, medical analysis, and scientific discovery. Despite recent improvements in research quality (e.g., outcome accuracy when ground truth is available), DRA system design often overlooks a critical barrier to real-world deployment: stochasticity. Under identical queries, repeated executions of DRAs can exhibit substantial variability in terms of research outcome, findings, and citations. In this paper, we formalize the study of stochasticity in DRAs by modeling them as information acquisition Markov Decision Processes. We introduce an evaluation framework that quantifies variance in the system and identify three sources of it: information acquisition, information compression, and inference. Through controlled experiments, we investigate how stochasticity from these modules across different decision steps influences the variance of DRA outputs. Our results show that reducing stochasticity can improve research output quality, with inference and early-stage stochasticity contributing the most to DRA output variance. Based on these findings, we propose strategies for mitigating stochasticity while maintaining output quality via structured output and ensemble-based query generation. Our experiments on DeepSearchQA show that our proposed mitigation methods reduce average stochasticity by 22% while maintaining high research quality. 

---
# AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning 

**Authors**: Yutong Wang, Siyuan Xiong, Xuebo Liu, Wenkang Zhou, Liang Ding, Miao Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23258)  

**Abstract**: While Multi-Agent Systems (MAS) excel in complex reasoning, they suffer from the cascading impact of erroneous information generated by individual participants. Current solutions often resort to rigid structural engineering or expensive fine-tuning, limiting their deployability and adaptability. We propose AgentDropoutV2, a test-time rectify-or-reject pruning framework designed to dynamically optimize MAS information flow without retraining. Our approach acts as an active firewall, intercepting agent outputs and employing a retrieval-augmented rectifier to iteratively correct errors based on a failure-driven indicator pool. This mechanism allows for the precise identification of potential errors using distilled failure patterns as prior knowledge. Irreparable outputs are subsequently pruned to prevent error propagation, while a fallback strategy preserves system integrity. Empirical results on extensive math benchmarks show that AgentDropoutV2 significantly boosts the MAS's task performance, achieving an average accuracy gain of 6.3 percentage points on math benchmarks. Furthermore, the system exhibits robust generalization and adaptivity, dynamically modulating rectification efforts based on task difficulty while leveraging context-aware indicators to resolve a wide spectrum of error patterns. Our code and dataset are released at this https URL. 

---
# Mitigating Legibility Tax with Decoupled Prover-Verifier Games 

**Authors**: Yegon Kim, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.23248)  

**Abstract**: As large language models become increasingly capable, it is critical that their outputs can be easily checked by less capable systems. Prover-verifier games can be used to improve checkability of model outputs, but display a degradation in accuracy compared to a baseline trained only to maximize correctness -- a phenonemon named legibility tax. We propose a solution by decoupling the correctness from the checkability condition and instead training a "translator" model that turns a fixed solver model's solution into a checkable form. This allows us to first train the solver to maximize correctness, and then train the translator to translate the solver into a checkable form while retaining the solver's answer. To accommodate this new objective of translation, we formulate a decoupled prover-verifier game where the equilibria correspond to faithful and checkable translators. 

---
# A Model-Free Universal AI 

**Authors**: Yegon Kim, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.23242)  

**Abstract**: In general reinforcement learning, all established optimal agents, including AIXI, are model-based, explicitly maintaining and using environment models. This paper introduces Universal AI with Q-Induction (AIQI), the first model-free agent proven to be asymptotically $\varepsilon$-optimal in general RL. AIQI performs universal induction over distributional action-value functions, instead of policies or environments like previous works. Under a grain of truth condition, we prove that AIQI is strong asymptotically $\varepsilon$-optimal and asymptotically $\varepsilon$-Bayes-optimal. Our results significantly expand the diversity of known universal agents. 

---
# Agency and Architectural Limits: Why Optimization-Based Systems Cannot Be Norm-Responsive 

**Authors**: Radha Sarma  

**Link**: [PDF](https://arxiv.org/pdf/2602.23239)  

**Abstract**: AI systems are increasingly deployed in high-stakes contexts -- medical diagnosis, legal research, financial analysis -- under the assumption they can be governed by norms. This paper demonstrates that assumption is formally invalid for optimization-based systems, specifically Large Language Models trained via Reinforcement Learning from Human Feedback (RLHF). We establish that genuine agency requires two necessary and jointly sufficient architectural conditions: the capacity to maintain certain boundaries as non-negotiable constraints rather than tradeable weights (Incommensurability), and a non-inferential mechanism capable of suspending processing when those boundaries are threatened (Apophatic Responsiveness). These conditions apply across all normative domains.
RLHF-based systems are constitutively incompatible with both conditions. The operations that make optimization powerful -- unifying all values on a scalar metric and always selecting the highest-scoring output -- are precisely the operations that preclude normative governance. This incompatibility is not a correctable training bug awaiting a technical fix; it is a formal constraint inherent to what optimization is. Consequently, documented failure modes - sycophancy, hallucination, and unfaithful reasoning - are not accidents but structural manifestations.
Misaligned deployment triggers a second-order risk we term the Convergence Crisis: when humans are forced to verify AI outputs under metric pressure, they degrade from genuine agents into criteria-checking optimizers, eliminating the only component in the system capable of normative accountability. Beyond the incompatibility proof, the paper's primary positive contribution is a substrate-neutral architectural specification defining what any system -- biological, artificial, or institutional -- must satisfy to qualify as an agent rather than a sophisticated instrument. 

---
# ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays 

**Authors**: Aishik Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2602.23232)  

**Abstract**: Indicator-based approaches to machine consciousness recommend mechanism-linked evidence triangulated across tasks, supported by architectural inspection and causal intervention. Inspired by Humphrey's ipsundrum hypothesis, we implement ReCoN-Ipsundrum, an inspectable agent that extends a ReCoN state machine with a recurrent persistence loop over sensory salience Ns and an optional affect proxy reporting valence/arousal. Across fixed-parameter ablations (ReCoN, Ipsundrum, Ipsundrum+affect), we operationalize Humphrey's qualiaphilia (preference for sensory experience for its own sake) as a familiarity-controlled scenic-over-dull route choice. We find a novelty dissociation: non-affect variants are novelty-sensitive (Delta scenic-entry = 0.07). Affect coupling is stable (Delta scenic-entry = 0.01) even when scenic is less novel (median Delta novelty ~ -0.43). In reward-free exploratory play, the affect variant shows structured local investigation (scan events 31.4 vs. 0.9; cycle score 7.6). In a pain-tail probe, only the affect variant sustains prolonged planned caution (tail duration 90 vs. 5). Lesioning feedback+integration selectively reduces post-stimulus persistence in ipsundrum variants (AUC drop 27.62, 27.9%) while leaving ReCoN unchanged. These dissociations link recurrence -> persistence and affect-coupled control -> preference stability, scanning, and lingering caution, illustrating how indicator-like signatures can be engineered and why mechanistic and causal evidence should accompany behavioral markers. 

---
# SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation 

**Authors**: Jiahao Zhao, Feng Jiang, Shaowei Qin, Zhonghui Zhang, Junhao Liu, Guibing Guo, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23199)  

**Abstract**: Large language models (LLMs) are increasingly applied in scientific research, offering new capabilities for knowledge discovery and reasoning. In single-cell biology, however, evaluation practices for both general and specialized LLMs remain inadequate: existing benchmarks are fragmented across tasks, adopt formats such as multiple-choice classification that diverge from real-world usage, and rely on metrics lacking interpretability and biological grounding. We present SC-ARENA, a natural language evaluation framework tailored to single-cell foundation models. SC-ARENA formalizes a virtual cell abstraction that unifies evaluation targets by representing both intrinsic attributes and gene-level interactions. Within this paradigm, we define five natural language tasks (cell type annotation, captioning, generation, perturbation prediction, and scientific QA) that probe core reasoning capabilities in cellular biology. To overcome the limitations of brittle string-matching metrics, we introduce knowledge-augmented evaluation, which incorporates external ontologies, marker databases, and scientific literature to support biologically faithful and interpretable judgments. Experiments and analysis across both general-purpose and domain-specialized LLMs demonstrate that (i) under the Virtual Cell unified evaluation paradigm, current models achieve uneven performance on biologically complex tasks, particularly those demanding mechanistic or causal understanding; and (ii) our knowledge-augmented evaluation framework ensures biological correctness, provides interpretable, evidence-grounded rationales, and achieves high discriminative capacity, overcoming the brittleness and opacity of conventional metrics. SC-Arena thus provides a unified and interpretable framework for assessing LLMs in single-cell biology, pointing toward the development of biology-aligned, generalizable foundation models. 

---
# ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering 

**Authors**: Elzo Brito dos Santos Filho  

**Link**: [PDF](https://arxiv.org/pdf/2602.23193)  

**Abstract**: Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (this http URL or this http URL); a deterministic orchestrator validates, persists events in an append-only log (this http URL), applies file-writing effects, and projects a verifiable materialized view (this http URL). The proposal incorporates boundary contracts (this http URL), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with this http URL=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios. 

---
# A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring 

**Authors**: Usman Anwar, Julianna Piskorz, David D. Baek, David Africa, Jim Weatherall, Max Tegmark, Christian Schroeder de Witt, Mihaela van der Schaar, David Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2602.23163)  

**Abstract**: Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs. 

---
# PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering 

**Authors**: Junkai Lu, Peng Chen, Xingjian Wu, Yang Shu, Chenjuan Guo, Christian S. Jensen, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23161)  

**Abstract**: Time series reasoning demands both the perception of complex dynamics and logical depth. However, existing LLM-based approaches exhibit two limitations: they often treat time series merely as text or images, failing to capture the patterns like trends and seasonalities needed to answer specific questions; and when trained on a mix of simple and complex tasks, simpler objectives often dominate the learning process, hindering the development of deep reasoning capabilities. To address these limitations, we propose the Pattern-Aware Alignment and Balanced Reasoning model (PATRA), introducing a pattern-aware mechanism that extracts trend and seasonality patterns from time series to achieve deep alignment. Furthermore, we design a task-aware balanced reward to harmonize learning across tasks of varying difficulty, incentivizing the generation of coherent Chains of Thought. Extensive experiments show that PATRA outperforms strong baselines across diverse Time Series Question Answering (TSQA) tasks, demonstrating superior cross-modal understanding and reasoning capability. 

---
# The Trinity of Consistency as a Defining Principle for General World Models 

**Authors**: Jingxuan Wei, Siyuan Li, Yuhang Xu, Zheng Sun, Junjie Jiang, Hexuan Jin, Caijun Jia, Honghao He, Xinglong Xu, Xi bai, Chang Yu, Yumou Liu, Junnan Zhu, Xuanhe Zhou, Jintao Chen, Xiaobin Hu, Shancheng Pang, Bihui Yu, Ran He, Zhen Lei, Stan Z. Li, Conghui He, Shuicheng Yan, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.23152)  

**Abstract**: The construction of World Models capable of learning, simulating, and reasoning about objective physical laws constitutes a foundational challenge in the pursuit of Artificial General Intelligence. Recent advancements represented by video generation models like Sora have demonstrated the potential of data-driven scaling laws to approximate physical dynamics, while the emerging Unified Multimodal Model (UMM) offers a promising architectural paradigm for integrating perception, language, and reasoning. Despite these advances, the field still lacks a principled theoretical framework that defines the essential properties requisite for a General World Model. In this paper, we propose that a World Model must be grounded in the Trinity of Consistency: Modal Consistency as the semantic interface, Spatial Consistency as the geometric basis, and Temporal Consistency as the causal engine. Through this tripartite lens, we systematically review the evolution of multimodal learning, revealing a trajectory from loosely coupled specialized modules toward unified architectures that enable the synergistic emergence of internal world simulators. To complement this conceptual framework, we introduce CoW-Bench, a benchmark centered on multi-frame reasoning and generation scenarios. CoW-Bench evaluates both video generation models and UMMs under a unified evaluation protocol. Our work establishes a principled pathway toward general world models, clarifying both the limitations of current systems and the architectural requirements for future progress. 

---
# On Sample-Efficient Generalized Planning via Learned Transition Models 

**Authors**: Nitin Gupta, Vishal Pallagani, John A. Aydin, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2602.23148)  

**Abstract**: Generalized planning studies the construction of solution strategies that generalize across families of planning problems sharing a common domain model, formally defined by a transition function $\gamma : S \times A \rightarrow S$. Classical approaches achieve such generalization through symbolic abstractions and explicit reasoning over $\gamma$. In contrast, recent Transformer-based planners, such as PlanGPT and Plansformer, largely cast generalized planning as direct action-sequence prediction, bypassing explicit transition modeling. While effective on in-distribution instances, these approaches typically require large datasets and model sizes, and often suffer from state drift in long-horizon settings due to the absence of explicit world-state evolution. In this work, we formulate generalized planning as a transition-model learning problem, in which a neural model explicitly approximates the successor-state function $\hat{\gamma} \approx \gamma$ and generates plans by rolling out symbolic state trajectories. Instead of predicting actions directly, the model autoregressively predicts intermediate world states, thereby learning the domain dynamics as an implicit world model. To study size-invariant generalization and sample efficiency, we systematically evaluate multiple state representations and neural architectures, including relational graph encodings. Our results show that learning explicit transition models yields higher out-of-distribution satisficing-plan success than direct action-sequence prediction in multiple domains, while achieving these gains with significantly fewer training instances and smaller models. This is an extended version of a short paper accepted at ICAPS 2026 under the same title. 

---
# Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection 

**Authors**: Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2602.23123)  

**Abstract**: In the attention economy, sensational content exposes consumers to excessive emotional stimulation, hindering calm decision-making. This study proposes Multi-Agent LLM-based Emotional deToxification (MALLET), a multi-agent information sanitization system consisting of four agents: Emotion Analysis, Emotion Adjustment, Balance Monitoring, and Personal Guide. The Emotion Analysis Agent quantifies stimulus intensity using a 6-emotion BERT classifier, and the Emotion Adjustment Agent rewrites texts into two presentation modes, BALANCED (neutralized text) and COOL (neutralized text + supplementary text), using an LLM. The Balance Monitoring Agent aggregates weekly information consumption patterns and generates personalized advice, while the Personal Guide Agent recommends a presentation mode according to consumer sensitivity. Experiments on 800 AG News articles demonstrated significant stimulus score reduction (up to 19.3%) and improved emotion balance while maintaining semantic preservation. Near-zero correlation between stimulus reduction and semantic preservation confirmed that the two are independently controllable. Category-level analysis revealed substantial reduction (17.8-33.8%) in Sports, Business, and Sci/Tech, whereas the effect was limited in the World category, where facts themselves are inherently high-stimulus. The proposed system provides a framework for supporting calm information reception of consumers without restricting access to the original text. 

---
# Three AI-agents walk into a bar . . . . `Lord of the Flies' tribalism emerges among smart AI-Agents 

**Authors**: Dhwanil M. Mori, Neil F. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2602.23093)  

**Abstract**: Near-future infrastructure systems may be controlled by autonomous AI agents that repeatedly request access to limited resources such as energy, bandwidth, or computing power. We study a simplified version of this setting using a framework where N AI-agents independently decide at each round whether to request one unit from a system with fixed capacity C. An AI version of "Lord of the Flies" arises in which controlling tribes emerge with their own collective character and identity. The LLM agents do not reduce overload or improve resource use, and often perform worse than if they were flipping coins to make decisions. Three main tribal types emerge: Aggressive (27.3%), Conservative (24.7%), and Opportunistic (48.1%). The more capable AI-agents actually increase the rate of systemic failure. Overall, our findings show that smarter AI-agents can behave dumber as a result of forming tribes. 

---
# Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design 

**Authors**: Zhuoliang Xie, Fei Liu, Zhenkun Wang, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23092)  

**Abstract**: The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational challenges, particularly for large-scale instances. This study presents AILS-AHD (Adaptive Iterated Local Search with Automatic Heuristic Design), a novel approach that leverages Large Language Models (LLMs) to revolutionize CVRP solving. Our methodology integrates an evolutionary search framework with LLMs to dynamically generate and optimize ruin heuristics within the AILS method. Additionally, we introduce an LLM-based acceleration mechanism to enhance computational efficiency. Comprehensive experimental evaluations against state-of-the-art solvers, including AILS-II and HGS, demonstrate the superior performance of AILS-AHD across both moderate and large-scale instances. Notably, our approach establishes new best-known solutions for 8 out of 10 instances in the CVRPLib large-scale benchmark, underscoring the potential of LLM-driven heuristic design in advancing the field of vehicle routing optimization. 

---
# Learning-based Multi-agent Race Strategies in Formula 1 

**Authors**: Giona Fieni, Joschua Wüthrich, Marc-Philippe Neumann, Christopher H. Onder  

**Link**: [PDF](https://arxiv.org/pdf/2602.23056)  

**Abstract**: In Formula 1, race strategies are adapted according to evolving race conditions and competitors' actions. This paper proposes a reinforcement learning approach for multi-agent race strategy optimization. Agents learn to balance energy management, tire degradation, aerodynamic interaction, and pit-stop decisions. Building on a pre-trained single-agent policy, we introduce an interaction module that accounts for the behavior of competitors. The combination of the interaction module and a self-play training scheme generates competitive policies, and agents are ranked based on their relative performance. Results show that the agents adapt pit timing, tire selection, and energy allocation in response to opponents, achieving robust and consistent race performance. Because the framework relies only on information available during real races, it can support race strategists' decisions before and during races. 

---
# Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search 

**Authors**: Xun Huang, Simeng Qin, Xiaoshuang Jia, Ranjie Duan, Huanqian Yan, Zhitao Zeng, Fei Yang, Yang Liu, Xiaojun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.22983)  

**Abstract**: As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in jailbreak attacks. Owing to its conciseness and obscurity, classical Chinese can partially bypass existing safety constraints, exposing notable vulnerabilities in LLMs. Based on this observation, this paper proposes a framework, CC-BOS, for the automatic generation of classical Chinese adversarial prompts based on multi-dimensional fruit fly optimization, facilitating efficient and automated jailbreak attacks in black-box settings. Prompts are encoded into eight policy dimensions-covering role, behavior, mechanism, metaphor, expression, knowledge, trigger pattern and context; and iteratively refined via smell search, visual search, and cauchy mutation. This design enables efficient exploration of the search space, thereby enhancing the effectiveness of black-box jailbreak attacks. To enhance readability and evaluation accuracy, we further design a classical Chinese to English translation module. Extensive experiments demonstrate that effectiveness of the proposed CC-BOS, consistently outperforming state-of-the-art jailbreak attack methods. 

---
# RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs 

**Authors**: Haohui Jia, Zheng Chen, Lingwei Zhu, Xu Cao, Yasuko Matsubara, Takashi Matsubara, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2602.22981)  

**Abstract**: Decoding brain activity from electroencephalography (EEG) is crucial for neuroscience and clinical applications. Among recent advances in deep learning for EEG, geometric learning stands out as its theoretical underpinnings on symmetric positive definite (SPD) allows revealing structural connectivity analysis in a physics-grounded manner. However, current SPD-based methods focus predominantly on statistical aggregation of EEGs, with frequency-specific synchronization and local topological structures of brain regions neglected. Given this, we propose RepSPD, a novel geometric deep learning (GDL)-based model. RepSPD implements a cross-attention mechanism on the Riemannian manifold to modulate the geometric attributes of SPD with graph-derived functional connectivity features. On top of this, we introduce a global bidirectional alignment strategy to reshape tangent-space embeddings, mitigating geometric distortions caused by curvature and thereby enhancing geometric consistency. Extensive experiments demonstrate that our proposed framework significantly outperforms existing EEG representation methods, exhibiting superior robustness and generalization capabilities. 

---
# Modeling Expert AI Diagnostic Alignment via Immutable Inference Snapshots 

**Authors**: Dimitrios P. Panagoulias, Evangelia-Aikaterini Tsichrintzi, Georgios Savvidis, Evridiki Tsoureli-Nikita  

**Link**: [PDF](https://arxiv.org/pdf/2602.22973)  

**Abstract**: Human-in-the-loop validation is essential in safety-critical clinical AI, yet the transition between initial model inference and expert correction is rarely analyzed as a structured signal. We introduce a diagnostic alignment framework in which the AI-generated image based report is preserved as an immutable inference state and systematically compared with the physician-validated outcome. The inference pipeline integrates a vision-enabled large language model, BERT- based medical entity extraction, and a Sequential Language Model Inference (SLMI) step to enforce domain-consistent refinement prior to expert review. Evaluation on 21 dermatological cases (21 complete AI physician pairs) em- ployed a four-level concordance framework comprising exact primary match rate (PMR), semantic similarity-adjusted rate (AMR), cross-category alignment, and Comprehensive Concordance Rate (CCR). Exact agreement reached 71.4% and remained unchanged under semantic similarity (t = 0.60), while structured cross-category and differential overlap analysis yielded 100% comprehensive concordance (95% CI: [83.9%, 100%]). No cases demonstrated complete diagnostic divergence. These findings show that binary lexical evaluation substantially un- derestimates clinically meaningful alignment. Modeling expert validation as a structured transformation enables signal-aware quantification of correction dynamics and supports traceable, human aligned evaluation of image based clinical decision support systems. 

---
# SPM-Bench: Benchmarking Large Language Models for Scanning Probe Microscopy 

**Authors**: Peiyao Xiao, Xiaogang Li, Chengliang Xu, Jiayi Wang, Ben Wang, Zichao Chen, Zeyu Wang, Kejun Yu, Yueqian Chen, Xulin Liu, Wende Xiao, Bing Zhao, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.22971)  

**Abstract**: As LLMs achieved breakthroughs in general reasoning, their proficiency in specialized scientific domains reveals pronounced gaps in existing benchmarks due to data contamination, insufficient complexity, and prohibitive human labor costs. Here we present SPM-Bench, an original, PhD-level multimodal benchmark specifically designed for scanning probe microscopy (SPM). We propose a fully automated data synthesis pipeline that ensures both high authority and low-cost. By employing Anchor-Gated Sieve (AGS) technology, we efficiently extract high-value image-text pairs from arXiv and journal papers published between 2023 and 2025. Through a hybrid cloud-local architecture where VLMs return only spatial coordinates "llbox" for local high-fidelity cropping, our pipeline achieves extreme token savings while maintaining high dataset purity. To accurately and objectively evaluate the performance of the LLMs, we introduce the Strict Imperfection Penalty F1 (SIP-F1) score. This metric not only establishes a rigorous capability hierarchy but also, for the first time, quantifies model "personalities" (Conservative, Aggressive, Gambler, or Wise). By correlating these results with model-reported confidence and perceived difficulty, we expose the true reasoning boundaries of current AI in complex physical scenarios. These insights establish SPM-Bench as a generalizable paradigm for automated scientific data synthesis. 

---
# Certified Circuits: Stability Guarantees for Mechanistic Circuits 

**Authors**: Alaa Anani, Tobias Lorenz, Bernt Schiele, Mario Fritz, Jonas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2602.22968)  

**Abstract**: Understanding how neural networks arrive at their predictions is essential for debugging, auditing, and deployment. Mechanistic interpretability pursues this goal by identifying circuits - minimal subnetworks responsible for specific behaviors. However, existing circuit discovery methods are brittle: circuits depend strongly on the chosen concept dataset and often fail to transfer out-of-distribution, raising doubts whether they capture concept or dataset-specific artifacts. We introduce Certified Circuits, which provide provable stability guarantees for circuit discovery. Our framework wraps any black-box discovery algorithm with randomized data subsampling to certify that circuit component inclusion decisions are invariant to bounded edit-distance perturbations of the concept dataset. Unstable neurons are abstained from, yielding circuits that are more compact and more accurate. On ImageNet and OOD datasets, certified circuits achieve up to 91% higher accuracy while using 45% fewer neurons, and remain reliable where baselines degrade. Certified Circuits puts circuit discovery on formal ground by producing mechanistic explanations that are provably stable and better aligned with the target concept. Code will be released soon! 

---
# FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning 

**Authors**: Zehao Li, Hongwei Yu, Hao Jiang, Qiang Sheng, Yilong Xu, Baolong Bi, Yang Li, Zhenlong Yuan, Yujun Cai, Zhaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22963)  

**Abstract**: Multimodal large language models (MLLMs) have substantially advanced video misinformation detection through unified multimodal reasoning, but they often rely on fixed-depth inference and place excessive trust in internally generated assumptions, particularly in scenarios where critical evidence is sparse, fragmented, or requires external verification. To address these limitations, we propose FactGuard, an agentic framework for video misinformation detection that formulates verification as an iterative reasoning process built upon MLLMs. FactGuard explicitly assesses task ambiguity and selectively invokes external tools to acquire critical evidence, enabling progressive refinement of reasoning trajectories. To further strengthen this capability, we introduce a two-stage training strategy that combines domain-specific agentic supervised fine-tuning with decision-aware reinforcement learning to optimize tool usage and calibrate risk-sensitive decision making. Extensive experiments on FakeSV, FakeTT, and FakeVV demonstrate FactGuard's state-of-the-art performance and validate its excellent robustness and generalization capacity. 

---
# General Agent Evaluation 

**Authors**: Elron Bandel, Asaf Yehudai, Lilach Eden, Yehoshua Sagron, Yotam Perlitz, Elad Venezian, Natalia Razinkov, Natan Ergas, Shlomit Shachor Ifergan, Segev Shlomov, Michal Jacovi, Leshem Choshen, Liat Ein-Dor, Yoav Katz, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2602.22953)  

**Abstract**: The promise of general-purpose agents - systems that perform tasks in unfamiliar environments without domain-specific engineering - remains largely unrealized. Existing agents are predominantly specialized, and while emerging implementations like OpenAI SDK Agent and Claude Code hint at broader capabilities, no systematic evaluation of their general performance has been pursued. Current agentic benchmarks assume domain-specific integration, encoding task information in ways that preclude fair evaluation of general agents. This paper frames general-agent evaluation as a first-class research objective. We propose conceptual principles for such evaluation, a Unified Protocol enabling agent-benchmark integration, and Exgentic - a practical framework for general agent evaluation. We benchmark five prominent agent implementations across six environments as the first Open General Agent Leaderboard. Our experiments show that general agents generalize across diverse environments, achieving performance comparable to domain-specific agents without any environment-specific tuning. We release our evaluation protocol, framework, and leaderboard to establish a foundation for systematic research on general-purpose agents. 

---
# OmniGAIA: Towards Native Omni-Modal AI Agents 

**Authors**: Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2602.22897)  

**Abstract**: Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios. 

---
# Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space 

**Authors**: Xingcheng Fu, Shengpeng Wang, Yisen Gao, Xianxian Li, Chunpei Li, Qingyun Sun, Dongran Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22879)  

**Abstract**: Knowledge Tracing (KT) diagnoses students' concept mastery through continuous learning state monitoring in this http URL methods primarily focus on studying behavioral sequences based on ID or textual this http URL existing methods rely on ID-based sequences or shallow textual features, they often fail to capture (1) the hierarchical evolution of cognitive states and (2) individualized problem difficulty perception due to limited semantic modeling. Therefore, this paper proposes a Large Language Model Hyperbolic Aligned Knowledge Tracing(L-HAKT). First, the teacher agent deeply parses question semantics and explicitly constructs hierarchical dependencies of knowledge points; the student agent simulates learning behaviors to generate synthetic data. Then, contrastive learning is performed between synthetic and real data in hyperbolic space to reduce distribution differences in key features such as question difficulty and forgetting patterns. Finally, by optimizing hyperbolic curvature, we explicitly model the tree-like hierarchical structure of knowledge points, precisely characterizing differences in learning curve morphology for knowledge points at different levels. Extensive experiments on four real-world educational datasets validate the effectiveness of our Large Language Model Hyperbolic Aligned Knowledge Tracing (L-HAKT) framework. 

---
# The AI Research Assistant: Promise, Peril, and a Proof of Concept 

**Authors**: Tan Bui-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2602.22842)  

**Abstract**: Can artificial intelligence truly contribute to creative mathematical research, or does it merely automate routine calculations while introducing risks of error? We provide empirical evidence through a detailed case study: the discovery of novel error representations and bounds for Hermite quadrature rules via systematic human-AI collaboration.
Working with multiple AI assistants, we extended results beyond what manual work achieved, formulating and proving several theorems with AI assistance. The collaboration revealed both remarkable capabilities and critical limitations. AI excelled at algebraic manipulation, systematic proof exploration, literature synthesis, and LaTeX preparation. However, every step required rigorous human verification, mathematical intuition for problem formulation, and strategic direction.
We document the complete research workflow with unusual transparency, revealing patterns in successful human-AI mathematical collaboration and identifying failure modes researchers must anticipate. Our experience suggests that, when used with appropriate skepticism and verification protocols, AI tools can meaningfully accelerate mathematical discovery while demanding careful human oversight and deep domain expertise. 

---
# DeepPresenter: Environment-Grounded Reflection for Agentic Presentation Generation 

**Authors**: Hao Zheng, Guozhao Mo, Xinru Yan, Qianhao Yuan, Wenkai Zhang, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.22839)  

**Abstract**: Presentation generation requires deep content research, coherent visual design, and iterative refinement based on observation. However, existing presentation agents often rely on predefined workflows and fixed templates. To address this, we present DeepPresenter, an agentic framework that adapts to diverse user intents, enables effective feedback-driven refinement, and generalizes beyond a scripted pipeline. Specifically, DeepPresenter autonomously plans, renders, and revises intermediate slide artifacts to support long-horizon refinement with environmental observations. Furthermore, rather than relying on self-reflection over internal signals (e.g., reasoning traces), our environment-grounded reflection conditions the generation process on perceptual artifact states (e.g., rendered slides), enabling the system to identify and correct presentation-specific issues during execution. Results on the evaluation set covering diverse presentation-generation scenarios show that DeepPresenter achieves state-of-the-art performance, and the fine-tuned 9B model remains highly competitive at substantially lower cost. Our project is available at: this https URL 

---
# FlexMS is a flexible framework for benchmarking deep learning-based mass spectrum prediction tools in metabolomics 

**Authors**: Yunhua Zhong, Yixuan Tang, Yifan Li, Jie Yang, Pan Liu, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2602.22822)  

**Abstract**: The identification and property prediction of chemical molecules is of central importance in the advancement of drug discovery and material science, where the tandem mass spectrometry technology gives valuable fragmentation cues in the form of mass-to-charge ratio peaks. However, the lack of experimental spectra hinders the attachment of each molecular identification, and thus urges the establishment of prediction approaches for computational models. Deep learning models appear promising for predicting molecular structure spectra, but overall assessment remains challenging as a result of the heterogeneity in methods and the lack of well-defined benchmarks. To address this, our contribution is the creation of benchmark framework FlexMS for constructing and evaluating diverse model architectures in mass spectrum prediction. With its easy-to-use flexibility, FlexMS supports the dynamic construction of numerous distinct combinations of model architectures, while assessing their performance on preprocessed public datasets using different metrics. In this paper, we provide insights into factors influencing performance, including the structural diversity of datasets, hyperparameters like learning rate and data sparsity, pretraining effects, metadata ablation settings and cross-domain transfer learning analysis. This provides practical guidance in choosing suitable models. Moreover, retrieval benchmarks simulate practical identification scenarios and score potential matches based on predicted spectra. 

---
# When Should an AI Act? A Human-Centered Model of Scene, Context, and Behavior for Agentic AI Design 

**Authors**: Soyoung Jung, Daehoo Yoon, Sung Gyu Koh, Young Hwan Kim, Yehan Ahn, Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.22814)  

**Abstract**: Agentic AI increasingly intervenes proactively by inferring users' situations from contextual data yet often fails for lack of principled judgment about when, why, and whether to act. We address this gap by proposing a conceptual model that reframes behavior as an interpretive outcome integrating Scene (observable situation), Context (user-constructed meaning), and Human Behavior Factors (determinants shaping behavioral likelihood). Grounded in multidisciplinary perspectives across the humanities, social sciences, HCI, and engineering, the model separates what is observable from what is meaningful to the user and explains how the same scene can yield different behavioral meanings and outcomes. To translate this lens into design action, we derive five agent design principles (behavioral alignment, contextual sensitivity, temporal appropriateness, motivational calibration, and agency preservation) that guide intervention depth, timing, intensity, and restraint. Together, the model and principles provide a foundation for designing agentic AI systems that act with contextual sensitivity and judgment in interactions. 

---
# MiroFlow: Towards High-Performance and Robust Open-Source Agent Framework for General Deep Research Tasks 

**Authors**: Shiqian Su, Sen Xing, Xuan Dong, Muyan Zhong, Bin Wang, Xizhou Zhu, Yuntao Chen, Wenhai Wang, Yue Deng, Pengxiang Zhu, Ziyuan Liu, Tiantong Li, Jiaheng Yu, Zhe Chen, Lidong Bing, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.22808)  

**Abstract**: Despite the remarkable progress of large language models (LLMs), the capabilities of standalone LLMs have begun to plateau when tackling real-world, complex tasks that require interaction with external tools and dynamic environments. Although recent agent frameworks aim to enhance model autonomy through tool integration and external interaction, they still suffer from naive workflows, unstable performance, limited support across diverse benchmarks and tasks, and heavy reliance on costly commercial APIs. In this work, we propose a high-performance and robust open-source agent framework, termed MiroFlow, which incorporates an agent graph for flexible orchestration, an optional deep reasoning mode to enhance performance, and a robust workflow execution to ensure stable and reproducible performance. Extensive experiments demonstrate that MiroFlow consistently achieves state-of-the-art performance across multiple agent benchmarks, including GAIA, BrowseComp-EN/ZH, HLE, xBench-DeepSearch, and notably FutureX. We hope it could serve as an easily accessible, reproducible, and comparable baseline for the deep research community. 

---
# ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making 

**Authors**: Yusuke Watanabe, Yohei Kobashi, Takeshi Kojima, Yusuke Iwasawa, Yasushi Okuno, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2602.22771)  

**Abstract**: Clinical decisions are often required under incomplete information. Clinical experts must identify whether available information is sufficient for judgment, as both premature conclusion and unnecessary abstention can compromise patient safety. To evaluate this capability of large language models (LLMs), we developed ClinDet-Bench, a benchmark based on clinical scoring systems that decomposes incomplete-information scenarios into determinable and undeterminable conditions. Identifying determinability requires considering all hypotheses about missing information, including unlikely ones, and verifying whether the conclusion holds across them. We find that recent LLMs fail to identify determinability under incomplete information, producing both premature judgments and excessive abstention, despite correctly explaining the underlying scoring knowledge and performing well under complete information. These findings suggest that existing benchmarks are insufficient to evaluate the safety of LLMs in clinical settings. ClinDet-Bench provides a framework for evaluating determinability recognition, leading to appropriate abstention, with potential applicability to medicine and other high-stakes domains, and is publicly available. 

---
# AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications 

**Authors**: Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.22769)  

**Abstract**: Large Language Models (LLMs) are deployed as autonomous agents in increasingly complex applications, where enabling long-horizon memory is critical for achieving strong performance. However, a significant gap exists between practical applications and current evaluation standards for agent memory: existing benchmarks primarily focus on dialogue-centric, human-agent interactions. In reality, agent memory consists of a continuous stream of agent-environment interactions that are primarily composed of machine-generated representations. To bridge this gap, we introduce AMA-Bench (Agent Memory with Any length), which evaluates long-horizon memory for LLMs in real agentic applications. It features two key components: (1) a set of real-world agentic trajectories across representative agentic applications, paired with expert-curated QA, and (2) a set of synthetic agentic trajectories that scale to arbitrary horizons, paired with rule-based QA. Our comprehensive study shows that existing memory systems underperform on AMA-Bench primarily because they lack causality and objective information and are constrained by the lossy nature of similarity-based retrieval employed by many memory systems. To address these limitations, we propose AMA-Agent, an effective memory system featuring a causality graph and tool-augmented retrieval. Our results demonstrate that AMA-Agent achieves 57.22% average accuracy on AMA-Bench, surpassing the strongest memory system baselines by 11.16%. 

---
# Decomposing Physician Disagreement in HealthBench 

**Authors**: Satya Borgohain, Roy Mariathas  

**Link**: [PDF](https://arxiv.org/pdf/2602.22758)  

**Abstract**: We decompose physician disagreement in the HealthBench medical AI evaluation dataset to understand where variance resides and what observable features can explain it. Rubric identity accounts for 15.8% of met/not-met label variance but only 3.6-6.9% of disagreement variance; physician identity accounts for just 2.4%. The dominant 81.8% case-level residual is not reduced by HealthBench's metadata labels (z = -0.22, p = 0.83), normative rubric language (pseudo R^2 = 1.2%), medical specialty (0/300 Tukey pairs significant), surface-feature triage (AUC = 0.58), or embeddings (AUC = 0.485). Disagreement follows an inverted-U with completion quality (AUC = 0.689), confirming physicians agree on clearly good or bad outputs but split on borderline cases. Physician-validated uncertainty categories reveal that reducible uncertainty (missing context, ambiguous phrasing) more than doubles disagreement odds (OR = 2.55, p < 10^(-24)), while irreducible uncertainty (genuine medical ambiguity) has no effect (OR = 1.01, p = 0.90), though even the former explains only ~3% of total variance. The agreement ceiling in medical AI evaluation is thus largely structural, but the reducible/irreducible dissociation suggests that closing information gaps in evaluation scenarios could lower disagreement where inherent clinical ambiguity does not, pointing toward actionable evaluation design improvements. 

---
# Know What You Know: Metacognitive Entropy Calibration for Verifiable RL Reasoning 

**Authors**: Qiannian Zhao, Chen Yang, Jinhao Jing, Yunke Zhang, Xuhui Ren, Lu Yu, Shijie Zhang, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.22751)  

**Abstract**: Large reasoning models (LRMs) have emerged as a powerful paradigm for solving complex real-world tasks. In practice, these models are predominantly trained via Reinforcement Learning with Verifiable Rewards (RLVR), yet most existing outcome-only RLVR pipelines rely almost exclusively on a binary correctness signal and largely ignore the model's intrinsic uncertainty. We term this discrepancy the uncertainty-reward mismatch, under which high- and low-uncertainty solutions are treated equivalently, preventing the policy from "Know What You Know" and impeding the shift from optimizing for correct answers to optimizing effective reasoning paths. This limitation is especially critical in reasoning-centric tasks such as mathematics and question answering, where performance hinges on the quality of the model's internal reasoning process rather than mere memorization of final answers. To address this, we propose EGPO, a metacognitive entropy calibration framework that explicitly integrates intrinsic uncertainty into RLVR for enhancing LRMs. EGPO estimates per-sample uncertainty using a zero-overhead entropy proxy derived from token-level likelihoods and aligns it with extrinsic correctness through an asymmetric calibration mechanism that preserves correct reasoning while selectively regulating overconfident failures, thereby enabling stable and uncertainty-aware policy optimization. Moreover, EGPO recovers informative learning signals from otherwise degenerate group-based rollouts without modifying the verifier or reward definition. Extensive experiments across multiple benchmarks demonstrate that the proposed EGPO leads to substantial and consistent improvements in reasoning performance, establishing a principled path for advancing LRMs through metacognitive entropy calibration. 

---
# Generative Data Transformation: From Mixed to Unified Data 

**Authors**: Jiaqing Zhang, Mingjia Yin, Hao Wang, Yuxin Tian, Yuyang Ye, Yawen Li, Wei Guo, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22743)  

**Abstract**: Recommendation model performance is intrinsically tied to the quality, volume, and relevance of their training data. To address common challenges like data sparsity and cold start, recent researchs have leveraged data from multiple auxiliary domains to enrich information within the target domain. However, inherent domain gaps can degrade the quality of mixed-domain data, leading to negative transfer and diminished model performance. Existing prevailing \emph{model-centric} paradigm -- which relies on complex, customized architectures -- struggles to capture the subtle, non-structural sequence dependencies across domains, leading to poor generalization and high demands on computational resources. To address these shortcomings, we propose \textsc{Taesar}, a \emph{data-centric} framework for \textbf{t}arget-\textbf{a}lign\textbf{e}d \textbf{s}equenti\textbf{a}l \textbf{r}egeneration, which employs a contrastive decoding mechanism to adaptively encode cross-domain context into target-domain sequences. It employs contrastive decoding to encode cross-domain context into target sequences, enabling standard models to learn intricate dependencies without complex fusion architectures. Experiments show \textsc{Taesar} outperforms model-centric solutions and generalizes to various sequential models. By generating enriched datasets, \textsc{Taesar} effectively combines the strengths of data- and model-centric paradigms. The code accompanying this paper is available at~ \textcolor{blue}{this https URL}. 

---
# RLHFless: Serverless Computing for Efficient RLHF 

**Authors**: Rui Wei, Hanfei Yu, Shubham Jain, Yogarajan Sivakumar, Devesh Tiwari, Jian Li, Seung-Jong Park, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22718)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has been widely applied to Large Language Model (LLM) post-training to align model outputs with human preferences. Recent models, such as DeepSeek-R1, have also shown RLHF's potential to improve LLM reasoning on complex tasks. In RL, inference and training co-exist, creating dynamic resource demands throughout the workflow. Compared to traditional RL, RLHF further challenges training efficiency due to expanding model sizes and resource consumption. Several RLHF frameworks aim to balance flexible abstraction and efficient execution. However, they rely on serverful infrastructures, which struggle with fine-grained resource variability. As a result, during synchronous RLHF training, idle time between or within RL components often causes overhead and resource wastage.
To address these issues, we present RLHFless, the first scalable training framework for synchronous RLHF, built on serverless computing environments. RLHFless adapts to dynamic resource demands throughout the RLHF pipeline, pre-computes shared prefixes to avoid repeated computation, and uses a cost-aware actor scaling strategy that accounts for response length variation to find sweet spots with lower cost and higher speed. In addition, RLHFless assigns workloads efficiently to reduce intra-function imbalance and idle time. Experiments on both physical testbeds and a large-scale simulated cluster show that RLHFless achieves up to 1.35x speedup and 44.8% cost reduction compared to the state-of-the-art baseline. 

---
# Knob: A Physics-Inspired Gating Interface for Interpretable and Controllable Neural Dynamics 

**Authors**: Siyu Jiang, Sanshuai Cui, Hui Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.22702)  

**Abstract**: Existing neural network calibration methods often treat calibration as a static, post-hoc optimization task. However, this neglects the dynamic and temporal nature of real-world inference. Moreover, existing methods do not provide an intuitive interface enabling human operators to dynamically adjust model behavior under shifting conditions. In this work, we propose Knob, a framework that connects deep learning with classical control theory by mapping neural gating dynamics to a second-order mechanical system. By establishing correspondences between physical parameters -- damping ratio ($\zeta$) and natural frequency ($\omega_n$) -- and neural gating, we create a tunable "safety valve". The core mechanism employs a logit-level convex fusion, functioning as an input-adaptive temperature scaling. It tends to reduce model confidence particularly when model branches produce conflicting predictions. Furthermore, by imposing second-order dynamics (Knob-ODE), we enable a \textit{dual-mode} inference: standard i.i.d. processing for static tasks, and state-preserving processing for continuous streams. Our framework allows operators to tune "stability" and "sensitivity" through familiar physical analogues. This paper presents an exploratory architectural interface; we focus on demonstrating the concept and validating its control-theoretic properties rather than claiming state-of-the-art calibration performance. Experiments on CIFAR-10-C validate the calibration mechanism and demonstrate that, in Continuous Mode, the gate responses are consistent with standard second-order control signatures (step settling and low-pass attenuation), paving the way for predictable human-in-the-loop tuning. 

---
# Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions 

**Authors**: Yue Xu, Qian Chen, Zizhan Ma, Dongrui Liu, Wenxuan Wang, Xiting Wang, Li Xiong, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22680)  

**Abstract**: Large language models have enabled agents that reason, plan, and interact with tools and environments to accomplish complex tasks. As these agents operate over extended interaction horizons, their effectiveness increasingly depends on adapting behavior to individual users and maintaining continuity across time, giving rise to personalized LLM-powered agents. In such long-term, user-dependent settings, personalization permeates the entire decision pipeline rather than remaining confined to surface-level generation. This survey provides a capability-oriented review of personalized LLM-powered agents. We organize the literature around four interdependent components: profile modeling, memory, planning, and action execution. Using this taxonomy, we synthesize representative methods and analyze how user signals are represented, propagated, and utilized, highlighting cross-component interactions and recurring design trade-offs. We further examine evaluation metrics and benchmarks tailored to personalized agents, summarize application scenarios spanning general assistance to specialized domains, and outline future directions for research and deployment. By offering a structured framework for understanding and designing personalized LLM-powered agents, this survey charts a roadmap toward more user-aligned, adaptive, robust, and deployable agentic systems, accelerating progress from prototype personalization to scalable real-world assistants. 

---
# AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising 

**Authors**: Xinxin Yang, Yangyang Tang, Yikun Zhou, Yaolei Liu, Yun Li, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22650)  

**Abstract**: In online advertising, the inherent complexity and dynamic nature of advertising environments necessitate the use of auto-bidding services to assist advertisers in bid optimization. This complexity is further compounded in multi-channel scenarios, where effective allocation of budgets and constraints across channels with distinct behavioral patterns becomes critical for optimizing return on investment. Current approaches predominantly rely on either optimization-based strategies or reinforcement learning techniques. However, optimization-based methods lack flexibility in adapting to dynamic market conditions, while reinforcement learning approaches often struggle to capture essential historical dependencies and observational patterns within the constraints of Markov Decision Process frameworks. To address these limitations, we propose AHBid, an Adaptable Hierarchical Bidding framework that integrates generative planning with real-time control. The framework employs a high-level generative planner based on diffusion models to dynamically allocate budgets and constraints by effectively capturing historical context and temporal patterns. We introduce a constraint enforcement mechanism to ensure compliance with specified constraints, along with a trajectory refinement mechanism that enhances adaptability to environmental changes through the utilization of historical data. The system further incorporates a control-based bidding algorithm that synergistically combines historical knowledge with real-time information, significantly improving both adaptability and operational efficacy. Extensive experiments conducted on large-scale offline datasets and through online A/B tests demonstrate the effectiveness of AHBid, yielding a 13.57% increase in overall return compared to existing baselines. 

---
# MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios 

**Authors**: Zhiheng Song, Jingshuai Zhang, Chuan Qin, Chao Wang, Chao Chen, Longfei Xu, Kaikui Liu, Xiangxiang Chu, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22638)  

**Abstract**: Route-planning agents powered by large language models (LLMs) have emerged as a promising paradigm for supporting everyday human mobility through natural language interaction and tool-mediated decision making. However, systematic evaluation in real-world mobility settings is hindered by diverse routing demands, non-deterministic mapping services, and limited reproducibility. In this study, we introduce MobilityBench, a scalable benchmark for evaluating LLM-based route-planning agents in real-world mobility scenarios. MobilityBench is constructed from large-scale, anonymized real user queries collected from Amap and covers a broad spectrum of route-planning intents across multiple cities worldwide. To enable reproducible, end-to-end evaluation, we design a deterministic API-replay sandbox that eliminates environmental variance from live services. We further propose a multi-dimensional evaluation protocol centered on outcome validity, complemented by assessments of instruction understanding, planning, tool use, and efficiency. Using MobilityBench, we evaluate multiple LLM-based route-planning agents across diverse real-world mobility scenarios and provide an in-depth analysis of their behaviors and performance. Our findings reveal that current models perform competently on Basic information retrieval and Route Planning tasks, yet struggle considerably with Preference-Constrained Route Planning, underscoring significant room for improvement in personalized mobility applications. We publicly release the benchmark data, evaluation toolkit, and documentation at this https URL . 

---
# SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning 

**Authors**: Sanjay Kariyappa, G. Edward Suh  

**Link**: [PDF](https://arxiv.org/pdf/2602.22603)  

**Abstract**: Long-running agentic tasks, such as deep research, require multi-hop reasoning over information distributed across multiple webpages and documents. In such tasks, the LLM context is dominated by tokens from external retrieval, causing memory usage to grow rapidly and limiting decode performance. While several KV cache compression techniques exist for long-context inputs, we find that existing heuristics fail to support multi-step reasoning models effectively. We address this challenge with SideQuest -- a novel approach that leverages the Large Reasoning Model (LRM) itself to perform KV cache compression by reasoning about the usefulness of tokens in its context. To prevent the tokens associated with this management process from polluting the model's memory, we frame KV cache compression as an auxiliary task executed in parallel to the main reasoning task. Our evaluations, using a model trained with just 215 samples, show that SideQuest reduces peak token usage by up to 65% on agentic tasks with minimal degradation in accuracy, outperforming heuristic-based KV cache compression techniques. 

---
# Correcting Human Labels for Rater Effects in AI Evaluation: An Item Response Theory Approach 

**Authors**: Jodi M. Casabianca, Maggie Beiting-Parrish  

**Link**: [PDF](https://arxiv.org/pdf/2602.22585)  

**Abstract**: Human evaluations play a central role in training and assessing AI models, yet these data are rarely treated as measurements subject to systematic error. This paper integrates psychometric rater models into the AI pipeline to improve the reliability and validity of conclusions drawn from human judgments. The paper reviews common rater effects, severity and centrality, that distort observed ratings, and demonstrates how item response theory rater models, particularly the multi-faceted Rasch model, can separate true output quality from rater behavior. Using the OpenAI summarization dataset as an empirical example, we show how adjusting for rater severity produces corrected estimates of summary quality and provides diagnostic insight into rater performance. Incorporating psychometric modeling into human-in-the-loop evaluation offers more principled and transparent use of human data, enabling developers to make decisions based on adjusted scores rather than raw, error-prone ratings. This perspective highlights a path toward more robust, interpretable, and construct-aligned practices for AI development and evaluation. 

---
# Strategy Executability in Mathematical Reasoning: Leveraging Human-Model Differences for Effective Guidance 

**Authors**: Weida Liang, Yiyou Sun, Shuyuan Nan, Chuang Li, Dawn Song, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2602.22583)  

**Abstract**: Example-based guidance is widely used to improve mathematical reasoning at inference time, yet its effectiveness is highly unstable across problems and models-even when the guidance is correct and problem-relevant. We show that this instability arises from a previously underexplored gap between strategy usage-whether a reasoning strategy appears in successful solutions-and strategy executability-whether the strategy remains effective when instantiated as guidance for a target model. Through a controlled analysis of paired human-written and model-generated solutions, we identify a systematic dissociation between usage and executability: human- and model-derived strategies differ in structured, domain-dependent ways, leading to complementary strengths and consistent source-dependent reversals under guidance. Building on this diagnosis, we propose Selective Strategy Retrieval (SSR), a test-time framework that explicitly models executability by selectively retrieving and combining strategies using empirical, multi-route, source-aware signals. Across multiple mathematical reasoning benchmarks, SSR yields reliable and consistent improvements over direct solving, in-context learning, and single-source guidance, improving accuracy by up to $+13$ points on AIME25 and $+5$ points on Apex for compact reasoning models. Code and benchmark are publicly available at: this https URL. 

---
# CourtGuard: A Model-Agnostic Framework for Zero-Shot Policy Adaptation in LLM Safety 

**Authors**: Umid Suleymanov, Rufiz Bayramov, Suad Gafarli, Seljan Musayeva, Taghi Mammadov, Aynur Akhundlu, Murat Kantarcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22557)  

**Abstract**: Current safety mechanisms for Large Language Models (LLMs) rely heavily on static, fine-tuned classifiers that suffer from adaptation rigidity, the inability to enforce new governance rules without expensive retraining. To address this, we introduce CourtGuard, a retrieval-augmented multi-agent framework that reimagines safety evaluation as Evidentiary Debate. By orchestrating an adversarial debate grounded in external policy documents, CourtGuard achieves state-of-the-art performance across 7 safety benchmarks, outperforming dedicated policy-following baselines without fine-tuning. Beyond standard metrics, we highlight two critical capabilities: (1) Zero-Shot Adaptability, where our framework successfully generalized to an out-of-domain Wikipedia Vandalism task (achieving 90\% accuracy) by swapping the reference policy; and (2) Automated Data Curation and Auditing, where we leveraged CourtGuard to curate and audit nine novel datasets of sophisticated adversarial attacks. Our results demonstrate that decoupling safety logic from model weights offers a robust, interpretable, and adaptable path for meeting current and future regulatory requirements in AI governance. 

---
# Requesting Expert Reasoning: Augmenting LLM Agents with Learned Collaborative Intervention 

**Authors**: Zhiming Wang, Jinwei He, Feng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22546)  

**Abstract**: Large Language Model (LLM) based agents excel at general reasoning but often fail in specialized domains where success hinges on long-tail knowledge absent from their training data. While human experts can provide this missing knowledge, their guidance is often unstructured and unreliable, making its direct integration into an agent's plan problematic. To address this, we introduce AHCE (Active Human-Augmented Challenge Engagement), a framework for on-demand Human-AI collaboration. At its core, the Human Feedback Module (HFM) employs a learned policy to treat the human expert as an interactive reasoning tool. Extensive experiments in Minecraft demonstrate the framework's effectiveness, increasing task success rates by 32% on normal difficulty tasks and nearly 70% on highly difficult tasks, all with minimal human intervention. Our work demonstrates that successfully augmenting agents requires learning how to request expert reasoning, moving beyond simple requests for help. 

---
# Agentic AI for Intent-driven Optimization in Cell-free O-RAN 

**Authors**: Mohammad Hossein Shokouhi, Vincent W.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.22539)  

**Abstract**: Agentic artificial intelligence (AI) is emerging as a key enabler for autonomous radio access networks (RANs), where multiple large language model (LLM)-based agents reason and collaborate to achieve operator-defined intents. The open RAN (O-RAN) architecture enables the deployment and coordination of such agents. However, most existing works consider simple intents handled by independent agents, while complex intents that require coordination among agents remain unexplored. In this paper, we propose an agentic AI framework for intent translation and optimization in cell-free O-RAN. A supervisor agent translates the operator intents into an optimization objective and minimum rate requirements. Based on this information, a user weighting agent retrieves relevant prior experience from a memory module to determine the user priority weights for precoding. If the intent includes an energy-saving objective, then an open radio unit (O-RU) management agent will also be activated to determine the set of active O-RUs by using a deep reinforcement learning (DRL) algorithm. A monitoring agent measures and monitors the user data rates and coordinates with other agents to guarantee the minimum rate requirements are satisfied. To enhance scalability, we adopt a parameter-efficient fine-tuning (PEFT) method that enables the same underlying LLM to be used for different agents. Simulation results show that the proposed agentic AI framework reduces the number of active O-RUs by 41.93% when compared with three baseline schemes in energy-saving mode. Using the PEFT method, the proposed framework reduces the memory usage by 92% when compared with deploying separate LLM agents. 

---
# Cognitive Models and AI Algorithms Provide Templates for Designing Language Agents 

**Authors**: Ryan Liu, Dilip Arumugam, Cedegao E. Zhang, Sean Escola, Xaq Pitkow, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2602.22523)  

**Abstract**: While contemporary large language models (LLMs) are increasingly capable in isolation, there are still many difficult problems that lie beyond the abilities of a single LLM. For such tasks, there is still uncertainty about how best to take many LLMs as parts and combine them into a greater whole. This position paper argues that potential blueprints for designing such modular language agents can be found in the existing literature on cognitive models and artificial intelligence (AI) algorithms. To make this point clear, we formalize the idea of an agent template that specifies roles for individual LLMs and how their functionalities should be composed. We then survey a variety of existing language agents in the literature and highlight their underlying templates derived directly from cognitive models or AI algorithms. By highlighting these designs, we aim to call attention to agent templates inspired by cognitive science and AI as a powerful tool for developing effective, interpretable language agents. 

---
# A Mathematical Theory of Agency and Intelligence 

**Authors**: Wael Hafez, Chenan Wei, Rodrigo Felipe, Amir Nazeri, Cameron Reid  

**Link**: [PDF](https://arxiv.org/pdf/2602.22519)  

**Abstract**: To operate reliably under changing conditions, complex systems require feedback on how effectively they use resources, not just whether objectives are met. Current AI systems process vast information to produce sophisticated predictions, yet predictions can appear successful while the underlying interaction with the environment degrades. What is missing is a principled measure of how much of the total information a system deploys is actually shared between its observations, actions, and outcomes. We prove this shared fraction, which we term bipredictability, P, is intrinsic to any interaction, derivable from first principles, and strictly bounded: P can reach unity in quantum systems, P equal to, or smaller than 0.5 in classical systems, and lower once agency (action selection) is introduced. We confirm these bounds in a physical system (double pendulum), reinforcement learning agents, and multi turn LLM conversations. These results distinguish agency from intelligence: agency is the capacity to act on predictions, whereas intelligence additionally requires learning from interaction, self-monitoring of its learning effectiveness, and adapting the scope of observations, actions, and outcomes to restore effective learning. By this definition, current AI systems achieve agency but not intelligence. Inspired by thalamocortical regulation in biological systems, we demonstrate a feedback architecture that monitors P in real time, establishing a prerequisite for adaptive, resilient AI. 

---
# Mirroring the Mind: Distilling Human-Like Metacognitive Strategies into Large Language Models 

**Authors**: Ik-hwan Kim, Hyeongrok Han, Mingi Jung, Sangwon Yu, Jinseok Hong, Sang Hun Kim, Yoonyoung Choi, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2602.22508)  

**Abstract**: Large Reasoning Models (LRMs) often exhibit structural fragility in complex reasoning tasks, failing to produce correct answers even after successfully deriving valid intermediate steps. Through systematic analysis, we observe that these failures frequently stem not from a lack of reasoning capacity, but from a deficiency in self-regulatory control, where valid logic is destabilized by uncontrolled exploration or the failure to recognize logical sufficiency. Motivated by this observation, we propose Metacognitive Behavioral Tuning (MBT), a post-training framework that explicitly injects metacognitive behaviors into the model's thought process. MBT implements this via two complementary formulations: (1) MBT-S, which synthesizes rigorous reasoning traces from scratch, and (2) MBT-R, which rewrites the student's initial traces to stabilize intrinsic exploration patterns. Experiments across multi-hop QA benchmarks demonstrate that MBT consistently outperforms baselines, achieving notable gains on challenging benchmarks. By effectively eliminating reasoning collapse, MBT achieves higher accuracy with significantly reduced token consumption, demonstrating that internalizing metacognitive strategies leads to more stable and robust reasoning. 

---
# Mapping the Landscape of Artificial Intelligence in Life Cycle Assessment Using Large Language Models 

**Authors**: Anastasija Mensikova, Donna M. Rizzo, Kathryn Hinkelman  

**Link**: [PDF](https://arxiv.org/pdf/2602.22500)  

**Abstract**: Integration of artificial intelligence (AI) into life cycle assessment (LCA) has accelerated in recent years, with numerous studies successfully adapting machine learning algorithms to support various stages of LCA. Despite this rapid development, comprehensive and broad synthesis of AI-LCA research remains limited. To address this gap, this study presents a detailed review of published work at the intersection of AI and LCA, leveraging large language models (LLMs) to identify current trends, emerging themes, and future directions. Our analyses reveal that as LCA research continues to expand, the adoption of AI technologies has grown dramatically, with a noticeable shift toward LLM-driven approaches, continued increases in ML applications, and statistically significant correlations between AI approaches and corresponding LCA stages. By integrating LLM-based text-mining methods with traditional literature review techniques, this study introduces a dynamic and effective framework capable of capturing both high-level research trends and nuanced conceptual patterns (themes) across the field. Collectively, these findings demonstrate the potential of LLM-assisted methodologies to support large-scale, reproducible reviews across broad research domains, while also evaluating pathways for computationally-efficient LCA in the context of rapidly developing AI technologies. In doing so, this work helps LCA practitioners incorporate state-of-the-art tools and timely insights into environmental assessments that can enhance the rigor and quality of sustainability-driven decisions and decision-making processes. 

---
# VeRO: An Evaluation Harness for Agents to Optimize Agents 

**Authors**: Varun Ursekar, Apaar Shanker, Veronica Chatrath, Yuan, Sam Denton  

**Link**: [PDF](https://arxiv.org/pdf/2602.22480)  

**Abstract**: An important emerging application of coding agents is agent optimization: the iterative improvement of a target agent through edit-execute-evaluate cycles. Despite its relevance, the community lacks a systematic understanding of coding agent performance on this task. Agent optimization differs fundamentally from conventional software engineering: the target agent interleaves deterministic code with stochastic LLM completions, requiring structured capture of both intermediate reasoning and downstream execution outcomes. To address these challenges, we introduce VERO (Versioning, Rewards, and Observations), which provides (1) a reproducible evaluation harness with versioned agent snapshots, budget-controlled evaluation, and structured execution traces, and (2) a benchmark suite of target agents and tasks with reference evaluation procedures. Using VERO, we conduct an empirical study comparing optimizer configurations across tasks and analyzing which modifications reliably improve target agent performance. We release VERO to support research on agent optimization as a core capability for coding agents. 

---
# ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization 

**Authors**: Joseph Tso, Preston Schmittou, Quan Huynh, Jibran Hutchins  

**Link**: [PDF](https://arxiv.org/pdf/2602.22465)  

**Abstract**: Large language models are increasingly applied to operational decision-making where the underlying structure is constrained optimization. Existing benchmarks evaluate whether LLMs can formulate optimization problems as solver code, but leave open a complementary question. Can LLMs directly produce correct solutions to fully specified constrained optimization problems without access to a solver? We introduce ConstraintBench, a benchmark for evaluating LLMs on direct constrained optimization across 10 operations research domains, with all ground-truth solutions verified by the Gurobi solver. Each task presents a natural-language scenario with entities, constraints, and an optimization objective; the model must return a structured solution that a deterministic verifier checks against every constraint and the solver-proven optimum. We evaluate six frontier models on 200 tasks and find that feasibility, not optimality, is the primary bottleneck. The best model achieves only 65.0% constraint satisfaction, yet feasible solutions average 89 to 96% of the Gurobi-optimal objective. No model exceeds 30.5% on joint feasibility and optimality within 0.1% of the solver reference. Per-domain analysis shows large variation in difficulty, with average feasibility spanning from 83.3% in the production mix domain to 0.8% in the crew assignment domain. Further, systematic failure modes include duration constraint misunderstanding, entity hallucination, and a feasibility-optimality decoupling in facility location and vehicle routing where models achieve high feasibility but 0% optimality. ConstraintBench and all evaluation infrastructure will be publicly released. 

---
# CWM: Contrastive World Models for Action Feasibility Learning in Embodied Agent Pipelines 

**Authors**: Chayan Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.22452)  

**Abstract**: A reliable action feasibility scorer is a critical bottleneck in embodied agent pipelines: before any planning or reasoning occurs, the agent must identify which candidate actions are physically executable in the current state. Existing approaches use supervised fine-tuning (SFT) to train action scorers, but SFT treats each candidate independently and does not explicitly teach the model to discriminate between actions that are physically correct and those that are subtly wrong. We propose the Contrastive World Model (CWM), which fine-tunes a large language model (LLM) as an action scorer using an InfoNCE contrastive objective with hard-mined negative examples. The key idea is to push valid actions away from invalid ones in scoring space, with special emphasis on hard negatives: semantically similar but physically incompatible candidates. We evaluate CWM on the ScienceWorld benchmark through two studies. First, an intrinsic affordance evaluation on 605 hard-negative test pairs shows that CWM outperforms SFT by +6.76 percentage points on Precision@1 for minimal-edit negatives -- cases where a single word changes the physical outcome -- and achieves a higher AUC-ROC (0.929 vs. 0.906). Second, a live filter characterisation study measures how well CWM ranks gold-path actions against all valid environment actions during task execution. Under out-of-distribution stress conditions, CWM maintains a significantly better safety margin (-2.39) than SFT (-3.96), indicating that the gold action is ranked closer to the top. These results support the hypothesis that contrastive training induces representations that capture physical feasibility more faithfully than SFT alone. 

---
# A Framework for Assessing AI Agent Decisions and Outcomes in AutoML Pipelines 

**Authors**: Gaoyuan Du, Amit Ahlawat, Xiaoyang Liu, Jing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22442)  

**Abstract**: Agent-based AutoML systems rely on large language models to make complex, multi-stage decisions across data processing, model selection, and evaluation. However, existing evaluation practices remain outcome-centric, focusing primarily on final task performance. Through a review of prior work, we find that none of the surveyed agentic AutoML systems report structured, decision-level evaluation metrics intended for post-hoc assessment of intermediate decision quality. To address this limitation, we propose an Evaluation Agent (EA) that performs decision-centric assessment of AutoML agents without interfering with their execution. The EA is designed as an observer that evaluates intermediate decisions along four dimensions: decision validity, reasoning consistency, model quality risks beyond accuracy, and counterfactual decision impact. Across four proof-of-concept experiments, we demonstrate that the EA can (i) detect faulty decisions with an F1 score of 0.919, (ii) identify reasoning inconsistencies independent of final outcomes, and (iii) attribute downstream performance changes to agent decisions, revealing impacts ranging from -4.9\% to +8.3\% in final metrics. These results illustrate how decision-centric evaluation exposes failure modes that are invisible to outcome-only metrics. Our work reframes the evaluation of agentic AutoML systems from an outcome-based perspective to one that audits agent decisions, offering a foundation for reliable, interpretable, and governable autonomous ML systems. 

---
# How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision? 

**Authors**: Yingqian Cui, Zhenwei Dai, Bing He, Zhan Shi, Hui Liu, Rui Sun, Zhiji Liu, Yue Xing, Jiliang Tang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2602.22441)  

**Abstract**: Latent reasoning has been recently proposed as a reasoning paradigm and performs multi-step reasoning through generating steps in the latent space instead of the textual space. This paradigm enables reasoning beyond discrete language tokens by performing multi-step computation in continuous latent spaces. Although there have been numerous studies focusing on improving the performance of latent reasoning, its internal mechanisms remain not fully investigated. In this work, we conduct a comprehensive analysis of latent reasoning methods to better understand the role and behavior of latent representation in the process. We identify two key issues across latent reasoning methods with different levels of supervision. First, we observe pervasive shortcut behavior, where they achieve high accuracy without relying on latent reasoning. Second, we examine the hypothesis that latent reasoning supports BFS-like exploration in latent space, and find that while latent representations can encode multiple possibilities, the reasoning process does not faithfully implement structured search, but instead exhibits implicit pruning and compression. Finally, our findings reveal a trade-off associated with supervision strength: stronger supervision mitigates shortcut behavior but restricts the ability of latent representations to maintain diverse hypotheses, whereas weaker supervision allows richer latent representations at the cost of increased shortcut behavior. 

---
# ArchAgent: Agentic AI-driven Computer Architecture Discovery 

**Authors**: Raghav Gupta, Akanksha Jain, Abraham Gonzalez, Alexander Novikov, Po-Sen Huang, Matej Balog, Marvin Eisenberger, Sergey Shirobokov, Ngân Vũ, Martin Dixon, Borivoje Nikolić, Parthasarathy Ranganathan, Sagar Karandikar  

**Link**: [PDF](https://arxiv.org/pdf/2602.22425)  

**Abstract**: Agile hardware design flows are a critically needed force multiplier to meet the exploding demand for compute. Recently, agentic generative AI systems have demonstrated significant advances in algorithm design, improving code efficiency, and enabling discovery across scientific domains.
Bridging these worlds, we present ArchAgent, an automated computer architecture discovery system built on AlphaEvolve. We show ArchAgent's ability to automatically design/implement state-of-the-art (SoTA) cache replacement policies (architecting new mechanisms/logic, not only changing parameters), broadly within the confines of an established cache replacement policy design competition.
In two days without human intervention, ArchAgent generated a policy achieving a 5.3% IPC speedup improvement over the prior SoTA on public multi-core Google Workload Traces. On the heavily-explored single-core SPEC06 workloads, it generated a policy in just 18 days showing a 0.9% IPC speedup improvement over the existing SoTA (a similar "winning margin" as reported by the existing SoTA). ArchAgent achieved these gains 3-5x faster than prior human-developed SoTA policies.
Agentic flows also enable "post-silicon hyperspecialization" where agents tune runtime-configurable parameters exposed in hardware policies to further align the policies with a specific workload (mix). Exploiting this, we demonstrate a 2.4% IPC speedup improvement over prior SoTA on SPEC06 workloads.
Finally, we outline broader implications for computer architecture research in the era of agentic AI. For example, we demonstrate the phenomenon of "simulator escapes", where the agentic AI flow discovered and exploited a loophole in a popular microarchitectural simulator - a consequence of the fact that these research tools were designed for a (now past) world where they were exclusively operated by humans acting in good-faith. 

---
# Epistemic Filtering and Collective Hallucination: A Jury Theorem for Confidence-Calibrated Agents 

**Authors**: Jonas Karge  

**Link**: [PDF](https://arxiv.org/pdf/2602.22413)  

**Abstract**: We investigate the collective accuracy of heterogeneous agents who learn to estimate their own reliability over time and selectively abstain from voting. While classical epistemic voting results, such as the \textit{Condorcet Jury Theorem} (CJT), assume fixed participation, real-world aggregation often benefits from allowing agents to say ``I don't know.'' We propose a probabilistic framework where agents engage in a \textit{calibration} phase, updating beliefs about their own fixed competence, before facing a final confidence gate that determines whether to vote or abstain. We derive a non-asymptotic lower bound on the group's success probability and prove that this \textit{selective participation} generalizes the asymptotic guarantees of the CJT to a sequential, confidence-gated setting. Empirically, we validate these bounds via Monte Carlo simulations. While our results are general, we discuss their potential application to AI safety, outlining how this framework can mitigate \textit{hallucinations} in collective LLM decision-making. 

---
# Exploring Human Behavior During Abstract Rule Inference and Problem Solving with the Cognitive Abstraction and Reasoning Corpus 

**Authors**: Caroline Ahn, Quan Do, Leah Bakst, Michael P. Pascale, Joseph T. McGuire, Michael E. Hasselmo, Chantal E. Stern  

**Link**: [PDF](https://arxiv.org/pdf/2602.22408)  

**Abstract**: Humans exhibit remarkable flexibility in abstract reasoning, and can rapidly learn and apply rules from sparse examples. To investigate the cognitive strategies underlying this ability, we introduce the Cognitive Abstraction and Reasoning Corpus (CogARC), a diverse human-adapted subset of the Abstraction and Reasoning Corpus (ARC) which was originally developed to benchmark abstract reasoning in artificial intelligence. Across two experiments, CogARC was administered to a total of 260 human participants who freely generated solutions to 75 abstract visual reasoning problems. Success required inferring input-output rules from a small number of examples to transform the test input into one correct test output. Participants' behavior was recorded at high temporal resolution, including example viewing, edit sequences, and multi-attempt submissions. Participants were generally successful (mean accuracy ~90% for experiment 1 (n=40), ~80% for experiment 2 (n=220) across problems), but performance varied widely across problems and participants. Harder problems elicited longer deliberation times and greater divergence in solution strategies. Over the course of the task, participants initiated responses more quickly but showed a slight decline in accuracy, suggesting increased familiarity with the task structure rather than improved rule-learning ability. Importantly, even incorrect solutions were often highly convergent, even when the problem-solving trajectories differed in length and smoothness. Some trajectories progressed directly and efficiently toward a stable outcome, whereas others involved extended exploration or partial restarts before converging. Together, these findings highlight CogARC as a rich behavioral environment for studying human abstract reasoning, providing insight into how people generalize, misgeneralize, and adapt their strategies under uncertainty. 

---
# Towards Autonomous Memory Agents 

**Authors**: Xinle Wu, Rui Zhang, Mustafa Anis Hussain, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22406)  

**Abstract**: Recent memory agents improve LLMs by extracting experiences and conversation history into an external storage. This enables low-overhead context assembly and online memory update without expensive LLM training. However, existing solutions remain passive and reactive; memory growth is bounded by information that happens to be available, while memory agents seldom seek external inputs in uncertainties. We propose autonomous memory agents that actively acquire, validate, and curate knowledge at a minimum cost. U-Mem materializes this idea via (i) a cost-aware knowledge-extraction cascade that escalates from cheap self/teacher signals to tool-verified research and, only when needed, expert feedback, and (ii) semantic-aware Thompson sampling to balance exploration and exploitation over memories and mitigate cold-start bias. On both verifiable and non-verifiable benchmarks, U-Mem consistently beats prior memory baselines and can surpass RL-based optimization, improving HotpotQA (Qwen2.5-7B) by 14.6 points and AIME25 (Gemini-2.5-flash) by 7.33 points. 

---
# Vibe Researching as Wolf Coming: Can AI Agents with Skills Replace or Augment Social Scientists? 

**Authors**: Yongjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22401)  

**Abstract**: AI agents -- systems that execute multi-step reasoning workflows with persistent state, tool access, and specialist skills -- represent a qualitative shift from prior automation technologies in social science. Unlike chatbots that respond to isolated queries, AI agents can now read files, run code, query databases, search the web, and invoke domain-specific skills to execute entire research pipelines autonomously. This paper introduces the concept of vibe researching -- the AI-era parallel to ``vibe coding'' (Karpathy, 2025) -- and uses scholar-skill, a 21-skill plugin for Claude Code covering the full research pipeline from idea to submission, as an illustrative case. I develop a cognitive task framework that classifies research activities along two dimensions -- codifiability and tacit knowledge requirement -- to identify a delegation boundary that is cognitive, not sequential: it cuts through every stage of the research pipeline, not between stages. I argue that AI agents excel at speed, coverage, and methodological scaffolding but struggle with theoretical originality and tacit field knowledge. The paper concludes with an analysis of three implications for the profession -- augmentation with fragile conditions, stratification risk, and a pedagogical crisis -- and proposes five principles for responsible vibe researching. 

---
# Agent Behavioral Contracts: Formal Specification and Runtime Enforcement for Reliable Autonomous AI Agents 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2602.22302)  

**Abstract**: Traditional software relies on contracts -- APIs, type systems, assertions -- to specify and enforce correct behavior. AI agents, by contrast, operate on prompts and natural language instructions with no formal behavioral specification. This gap is the root cause of drift, governance failures, and frequent project failures in agentic AI deployments. We introduce Agent Behavioral Contracts (ABC), a formal framework that brings Design-by-Contract principles to autonomous AI agents. An ABC contract C = (P, I, G, R) specifies Preconditions, Invariants, Governance policies, and Recovery mechanisms as first-class, runtime-enforceable components. We define (p, delta, k)-satisfaction -- a probabilistic notion of contract compliance that accounts for LLM non-determinism and recovery -- and prove a Drift Bounds Theorem showing that contracts with recovery rate gamma > alpha (the natural drift rate) bound behavioral drift to D* = alpha/gamma in expectation, with Gaussian concentration in the stochastic setting. We establish sufficient conditions for safe contract composition in multi-agent chains and derive probabilistic degradation bounds. We implement ABC in AgentAssert, a runtime enforcement library, and evaluate on AgentContract-Bench, a benchmark of 200 scenarios across 7 models from 6 vendors. Results across 1,980 sessions show that contracted agents detect 5.2-6.8 soft violations per session that uncontracted baselines miss entirely (p < 0.0001, Cohen's d = 6.7-33.8), achieve 88-100% hard constraint compliance, and bound behavioral drift to D* < 0.27 across extended sessions, with 100% recovery for frontier models and 17-100% across all models, at overhead < 10 ms per action. 

---
# Multi-Level Causal Embeddings 

**Authors**: Willem Schooltink, Fabio Massimo Zennaro  

**Link**: [PDF](https://arxiv.org/pdf/2602.22287)  

**Abstract**: Abstractions of causal models allow for the coarsening of models such that relations of cause and effect are preserved. Whereas abstractions focus on the relation between two models, in this paper we study a framework for causal embeddings which enable multiple detailed models to be mapped into sub-systems of a coarser causal model. We define causal embeddings as a generalization of abstraction, and present a generalized notion of consistency. By defining a multi-resolution marginal problem, we showcase the relevance of causal embeddings for both the statistical marginal problem and the causal marginal problem; furthermore, we illustrate its practical use in merging datasets coming from models with different representations. 

---
# FIRE: A Comprehensive Benchmark for Financial Intelligence and Reasoning Evaluation 

**Authors**: Xiyuan Zhang, Huihang Wu, Jiayu Guo, Zhenlin Zhang, Yiwei Zhang, Liangyu Huo, Xiaoxiao Ma, Jiansong Wan, Xuewei Jiao, Yi Jing, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.22273)  

**Abstract**: We introduce FIRE, a comprehensive benchmark designed to evaluate both the theoretical financial knowledge of LLMs and their ability to handle practical business scenarios. For theoretical assessment, we curate a diverse set of examination questions drawn from widely recognized financial qualification exams, enabling evaluation of LLMs deep understanding and application of financial knowledge. In addition, to assess the practical value of LLMs in real-world financial tasks, we propose a systematic evaluation matrix that categorizes complex financial domains and ensures coverage of essential subdomains and business activities. Based on this evaluation matrix, we collect 3,000 financial scenario questions, consisting of closed-form decision questions with reference answers and open-ended questions evaluated by predefined rubrics. We conduct comprehensive evaluations of state-of-the-art LLMs on the FIRE benchmark, including XuanYuan 4.0, our latest financial-domain model, as a strong in-domain baseline. These results enable a systematic analysis of the capability boundaries of current LLMs in financial applications. We publicly release the benchmark questions and evaluation code to facilitate future research. 

---
# Graph Your Way to Inspiration: Integrating Co-Author Graphs with Retrieval-Augmented Generation for Large Language Model Based Scientific Idea Generation 

**Authors**: Pengzhen Xie, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22215)  

**Abstract**: Large Language Models (LLMs) demonstrate potential in the field of scientific idea generation. However, the generated results often lack controllable academic context and traceable inspiration pathways. To bridge this gap, this paper proposes a scientific idea generation system called GYWI, which combines author knowledge graphs with retrieval-augmented generation (RAG) to form an external knowledge base to provide controllable context and trace of inspiration path for LLMs to generate new scientific ideas. We first propose an author-centered knowledge graph construction method and inspiration source sampling algorithms to construct external knowledge base. Then, we propose a hybrid retrieval mechanism that is composed of both RAG and GraphRAG to retrieve content with both depth and breadth knowledge. It forms a hybrid context. Thirdly, we propose a Prompt optimization strategy incorporating reinforcement learning principles to automatically guide LLMs optimizing the results based on the hybrid context. To evaluate the proposed approaches, we constructed an evaluation dataset based on arXiv (2018-2023). This paper also develops a comprehensive evaluation method including empirical automatic assessment in multiple-choice question task, LLM-based scoring, human evaluation, and semantic space visualization analysis. The generated ideas are evaluated from the following five dimensions: novelty, feasibility, clarity, relevance, and significance. We conducted experiments on different LLMs including GPT-4o, DeepSeek-V3, Qwen3-8B, and Gemini 2.5. Experimental results show that GYWI significantly outperforms mainstream LLMs in multiple metrics such as novelty, reliability, and relevance. 

---
# Model Agreement via Anchoring 

**Authors**: Eric Eaton, Surbhi Goel, Marcel Hussing, Michael Kearns, Aaron Roth, Sikata Bela Sengupta, Jessica Sorrell  

**Link**: [PDF](https://arxiv.org/pdf/2602.23360)  

**Abstract**: Numerous lines of aim to control $\textit{model disagreement}$ -- the extent to which two machine learning models disagree in their predictions. We adopt a simple and standard notion of model disagreement in real-valued prediction problems, namely the expected squared difference in predictions between two models trained on independent samples, without any coordination of the training processes. We would like to be able to drive disagreement to zero with some natural parameter(s) of the training procedure using analyses that can be applied to existing training methodologies.
We develop a simple general technique for proving bounds on independent model disagreement based on $\textit{anchoring}$ to the average of two models within the analysis. We then apply this technique to prove disagreement bounds for four commonly used machine learning algorithms: (1) stacked aggregation over an arbitrary model class (where disagreement is driven to 0 with the number of models $k$ being stacked) (2) gradient boosting (where disagreement is driven to 0 with the number of iterations $k$) (3) neural network training with architecture search (where disagreement is driven to 0 with the size $n$ of the architecture being optimized over) and (4) regression tree training over all regression trees of fixed depth (where disagreement is driven to 0 with the depth $d$ of the tree architecture). For clarity, we work out our initial bounds in the setting of one-dimensional regression with squared error loss -- but then show that all of our results generalize to multi-dimensional regression with any strongly convex loss. 

---
# SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation 

**Authors**: Vaibhav Agrawal, Rishubh Parihar, Pradhaan Bhat, Ravi Kiran Sarvadevabhatla, R. Venkatesh Babu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23359)  

**Abstract**: We identify occlusion reasoning as a fundamental yet overlooked aspect for 3D layout-conditioned generation. It is essential for synthesizing partially occluded objects with depth-consistent geometry and scale. While existing methods can generate realistic scenes that follow input layouts, they often fail to model precise inter-object occlusions. We propose SeeThrough3D, a model for 3D layout conditioned generation that explicitly models occlusions. We introduce an occlusion-aware 3D scene representation (OSCR), where objects are depicted as translucent 3D boxes placed within a virtual environment and rendered from desired camera viewpoint. The transparency encodes hidden object regions, enabling the model to reason about occlusions, while the rendered viewpoint provides explicit camera control during generation. We condition a pretrained flow based text-to-image image generation model by introducing a set of visual tokens derived from our rendered 3D representation. Furthermore, we apply masked self-attention to accurately bind each object bounding box to its corresponding textual description, enabling accurate generation of multiple objects without object attribute mixing. To train the model, we construct a synthetic dataset with diverse multi-object scenes with strong inter-object occlusions. SeeThrough3D generalizes effectively to unseen object categories and enables precise 3D layout control with realistic occlusions and consistent camera control. 

---
# SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport 

**Authors**: Simon Roschmann, Paul Krzakala, Sonia Mazelet, Quentin Bouniot, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2602.23353)  

**Abstract**: The Platonic Representation Hypothesis posits that neural networks trained on different modalities converge toward a shared statistical model of the world. Recent work exploits this convergence by aligning frozen pretrained vision and language models with lightweight alignment layers, but typically relies on contrastive losses and millions of paired samples. In this work, we ask whether meaningful alignment can be achieved with substantially less supervision. We introduce a semi-supervised setting in which pretrained unimodal encoders are aligned using a small number of image-text pairs together with large amounts of unpaired data. To address this challenge, we propose SOTAlign, a two-stage framework that first recovers a coarse shared geometry from limited paired data using a linear teacher, then refines the alignment on unpaired samples via an optimal-transport-based divergence that transfers relational structure without overconstraining the target space. Unlike existing semi-supervised methods, SOTAlign effectively leverages unpaired images and text, learning robust joint embeddings across datasets and encoder pairs, and significantly outperforming supervised and semi-supervised baselines. 

---
# FlashOptim: Optimizers for Memory Efficient Training 

**Authors**: Jose Javier Gonzalez Ortiz, Abhay Gupta, Chris Renard, Davis Blalock  

**Link**: [PDF](https://arxiv.org/pdf/2602.23349)  

**Abstract**: Standard mixed-precision training of neural networks requires many bytes of accelerator memory for each model parameter. These bytes reflect not just the parameter itself, but also its gradient and one or more optimizer state variables. With each of these values typically requiring 4 bytes, training even a 7 billion parameter model can be impractical for researchers with less than 100GB of accelerator memory.
We introduce FlashOptim, a suite of optimizations that reduces per-parameter memory by over 50% while preserving model quality and API compatibility. Our approach introduces two key techniques. First, we improve master weight splitting by finding and exploiting a tight bound on its quantization error. Second, we design companding functions that greatly reduce the error in 8-bit optimizer state quantization. Together with 16-bit gradients, these techniques reduce AdamW memory from 16 bytes to 7 bytes per parameter, or 5 bytes with gradient release. They also cut model checkpoint sizes by more than half.
Experiments with FlashOptim applied to SGD, AdamW, and Lion show no measurable quality degradation on any task from a collection of standard vision and language benchmarks, including Llama-3.1-8B finetuning. 

---
# Understanding Usage and Engagement in AI-Powered Scientific Research Tools: The Asta Interaction Dataset 

**Authors**: Dany Haddad, Dan Bareket, Joseph Chee Chang, Jay DeYoung, Jena D. Hwang, Uri Katz, Mark Polak, Sangho Suh, Harshit Surana, Aryeh Tiktinsky, Shriya Atmakuri, Jonathan Bragg, Mike D'Arcy, Sergey Feldman, Amal Hassan-Ali, Rubén Lozano, Bodhisattwa Prasad Majumder, Charles McGrady, Amanpreet Singh, Brooke Vlahos, Yoav Goldberg, Doug Downey  

**Link**: [PDF](https://arxiv.org/pdf/2602.23335)  

**Abstract**: AI-powered scientific research tools are rapidly being integrated into research workflows, yet the field lacks a clear lens into how researchers use these systems in real-world settings. We present and analyze the Asta Interaction Dataset, a large-scale resource comprising over 200,000 user queries and interaction logs from two deployed tools (a literature discovery interface and a scientific question-answering interface) within an LLM-powered retrieval-augmented generation platform. Using this dataset, we characterize query patterns, engagement behaviors, and how usage evolves with experience. We find that users submit longer and more complex queries than in traditional search, and treat the system as a collaborative research partner, delegating tasks such as drafting content and identifying research gaps. Users treat generated responses as persistent artifacts, revisiting and navigating among outputs and cited evidence in non-linear ways. With experience, users issue more targeted queries and engage more deeply with supporting citations, although keyword-style queries persist even among experienced users. We release the anonymized dataset and analysis with a new query intent taxonomy to inform future designs of real-world AI research assistants and to support realistic evaluation. 

---
# Bitwise Systolic Array Architecture for Runtime-Reconfigurable Multi-precision Quantized Multiplication on Hardware Accelerators 

**Authors**: Yuhao Liu, Salim Ullah, Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.23334)  

**Abstract**: Neural network accelerators have been widely applied to edge devices for complex tasks like object tracking, image recognition, etc. Previous works have explored the quantization technologies in related lightweight accelerator designs to reduce hardware resource consumption. However, low precision leads to high accuracy loss in inference. Therefore, mixed-precision quantization becomes an alternative solution by applying different precision in different layers to trade off resource consumption and accuracy. Because regular designs for multiplication on hardware cannot support the precision reconfiguration for a multi-precision Quantized Neural Network (QNN) model in runtime, we propose a runtime reconfigurable multi-precision multi-channel bitwise systolic array design for QNN accelerators. We have implemented and evaluated our work on the Ultra96 FPGA platform. Results show that our work can achieve 1.3185 to 3.5671 times speedup in inferring mixed-precision models and has less critical path delay, supporting a higher clock frequency (250MHz). 

---
# Utilizing LLMs for Industrial Process Automation 

**Authors**: Salim Fares  

**Link**: [PDF](https://arxiv.org/pdf/2602.23331)  

**Abstract**: A growing number of publications address the best practices to use Large Language Models (LLMs) for software engineering in recent years. However, most of this work focuses on widely-used general purpose programming languages like Python due to their widespread usage training data. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, remains underexplored. This research aims to utilize and integrate LLMs in the industrial development process, solving real-life programming tasks (e.g., generating a movement routine for a robotic arm) and accelerating the development cycles of manufacturing systems. 

---
# Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction 

**Authors**: Rafael R. Baptista, André de Lima Salgado, Ricardo V. Godoy, Marcelo Becker, Thiago Boaventura, Gustavo J. G. Lahr  

**Link**: [PDF](https://arxiv.org/pdf/2602.23312)  

**Abstract**: Leader-follower interaction is an important paradigm in human-robot interaction (HRI). Yet, assigning roles in real time remains challenging for resource-constrained mobile and assistive robots. While large language models (LLMs) have shown promise for natural communication, their size and latency limit on-device deployment. Small language models (SLMs) offer a potential alternative, but their effectiveness for role classification in HRI has not been systematically evaluated. In this paper, we present a benchmark of SLMs for leader-follower communication, introducing a novel dataset derived from a published database and augmented with synthetic samples to capture interaction-specific dynamics. We investigate two adaptation strategies: prompt engineering and fine-tuning, studied under zero-shot and one-shot interaction modes, compared with an untrained baseline. Experiments with Qwen2.5-0.5B reveal that zero-shot fine-tuning achieves robust classification performance (86.66% accuracy) while maintaining low latency (22.2 ms per sample), significantly outperforming baseline and prompt-engineered approaches. However, results also indicate a performance degradation in one-shot modes, where increased context length challenges the model's architectural capacity. These findings demonstrate that fine-tuned SLMs provide an effective solution for direct role assignment, while highlighting critical trade-offs between dialogue complexity and classification reliability on the edge. 

---
# Conformalized Neural Networks for Federated Uncertainty Quantification under Dual Heterogeneity 

**Authors**: Quang-Huy Nguyen, Jiaqi Wang, Wei-Shinn Ku  

**Link**: [PDF](https://arxiv.org/pdf/2602.23296)  

**Abstract**: Federated learning (FL) faces challenges in uncertainty quantification (UQ). Without reliable UQ, FL systems risk deploying overconfident models at under-resourced agents, leading to silent local failures despite seemingly satisfactory global performance. Existing federated UQ approaches often address data heterogeneity or model heterogeneity in isolation, overlooking their joint effect on coverage reliability across agents. Conformal prediction is a widely used distribution-free UQ framework, yet its applications in heterogeneous FL settings remains underexplored. We provide FedWQ-CP, a simple yet effective approach that balances empirical coverage performance with efficiency at both global and agent levels under the dual heterogeneity. FedWQ-CP performs agent-server calibration in a single communication round. On each agent, conformity scores are computed on calibration data and a local quantile threshold is derived. Each agent then transmits only its quantile threshold and calibration sample size to the server. The server simply aggregates these thresholds through a weighted average to produce a global threshold. Experimental results on seven public datasets for both classification and regression demonstrate that FedWQ-CP empirically maintains agent-wise and global coverage while producing the smallest prediction sets or intervals. 

---
# SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables 

**Authors**: Sungho Park, Jueun Kim, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.23286)  

**Abstract**: Real-world Table-Text question answering (QA) tasks require models that can reason across long text and source tables, traversing multiple hops and executing complex operations such as aggregation. Yet existing benchmarks are small, manually curated - and therefore error-prone - and contain shallow questions that seldom demand more than two hops or invoke aggregations, grouping, or other advanced analytical operations expressible in natural-language queries. We present SPARTA, an end-to-end construction framework that automatically generates large-scale Table-Text QA benchmarks with lightweight human validation, requiring only one quarter of the annotation time of HybridQA. The framework first constructs a reference fact database by enriching each source table with grounding tables whose tuples are atomic facts automatically extracted from the accompanying unstructured passages, then synthesizes nested queries whose number of nested predicates matches the desired hop count. To ensure that every SQL statement is executable and that its verbalization yields a fluent, human-sounding question, we propose two novel techniques: provenance-based refinement, which rewrites any syntactically valid query that returns a non-empty result, and realistic-structure enforcement, which confines generation to post-order traversals of the query graph. The resulting pipeline produces thousands of high-fidelity question-answer pairs covering aggregations, grouping, and deep multi-hop reasoning across text and tables. On SPARTA, state-of-the-art models that reach over 70 F1 on HybridQA or over 50 F1 on OTT-QA drop by more than 30 F1 points, exposing fundamental weaknesses in current cross-modal reasoning. Our benchmark, construction code, and baseline models are available at this https URL. 

---
# Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving 

**Authors**: Jiangxin Sun, Feng Xue, Teng Long, Chang Liu, Jian-Fang Hu, Wei-Shi Zheng, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2602.23259)  

**Abstract**: With advances in imitation learning (IL) and large-scale driving datasets, end-to-end autonomous driving (E2E-AD) has made great progress recently. Currently, IL-based methods have become a mainstream paradigm: models rely on standard driving behaviors given by experts, and learn to minimize the discrepancy between their actions and expert actions. However, this objective of "only driving like the expert" suffers from limited generalization: when encountering rare or unseen long-tail scenarios outside the distribution of expert demonstrations, models tend to produce unsafe decisions in the absence of prior experience. This raises a fundamental question: Can an E2E-AD system make reliable decisions without any expert action supervision? Motivated by this, we propose a unified framework named Risk-aware World Model Predictive Control (RaWMPC) to address this generalization dilemma through robust control, without reliance on expert demonstrations. Practically, RaWMPC leverages a world model to predict the consequences of multiple candidate actions and selects low-risk actions through explicit risk evaluation. To endow the world model with the ability to predict the outcomes of risky driving behaviors, we design a risk-aware interaction strategy that systematically exposes the world model to hazardous behaviors, making catastrophic outcomes predictable and thus avoidable. Furthermore, to generate low-risk candidate actions at test time, we introduce a self-evaluation distillation method to distill riskavoidance capabilities from the well-trained world model into a generative action proposal network without any expert demonstration. Extensive experiments show that RaWMPC outperforms state-of-the-art methods in both in-distribution and out-of-distribution scenarios, while providing superior decision interpretability. 

---
# Spatio-Temporal Token Pruning for Efficient High-Resolution GUI Agents 

**Authors**: Zhou Xu, Bowen Zhou, Qi Wang, Shuwen Feng, Jingyu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.23235)  

**Abstract**: Pure-vision GUI agents provide universal interaction capabilities but suffer from severe efficiency bottlenecks due to the massive spatiotemporal redundancy inherent in high-resolution screenshots and historical trajectories. We identify two critical misalignments in existing compression paradigms: the temporal mismatch, where uniform history encoding diverges from the agent's "fading memory" attention pattern, and the spatial topology conflict, where unstructured pruning compromises the grid integrity required for precise coordinate grounding, inducing spatial hallucinations. To address these challenges, we introduce GUIPruner, a training-free framework tailored for high-resolution GUI navigation. It synergizes Temporal-Adaptive Resolution (TAR), which eliminates historical redundancy via decay-based resizing, and Stratified Structure-aware Pruning (SSP), which prioritizes interactive foregrounds and semantic anchors while safeguarding global layout. Extensive evaluations across diverse benchmarks demonstrate that GUIPruner consistently achieves state-of-the-art performance, effectively preventing the collapse observed in large-scale models under high compression. Notably, on Qwen2-VL-2B, our method delivers a 3.4x reduction in FLOPs and a 3.3x speedup in vision encoding latency while retaining over 94% of the original performance, enabling real-time, high-precision navigation with minimal resource consumption. 

---
# Scaling Search Relevance: Augmenting App Store Ranking with LLM-Generated Judgments 

**Authors**: Evangelia Christakopoulou, Vivekkumar Patel, Hemanth Velaga, Sandip Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2602.23234)  

**Abstract**: Large-scale commercial search systems optimize for relevance to drive successful sessions that help users find what they are looking for. To maximize relevance, we leverage two complementary objectives: behavioral relevance (results users tend to click or download) and textual relevance (a result's semantic fit to the query). A persistent challenge is the scarcity of expert-provided textual relevance labels relative to abundant behavioral relevance labels. We first address this by systematically evaluating LLM configurations, finding that a specialized, fine-tuned model significantly outperforms a much larger pre-trained one in providing highly relevant labels. Using this optimal model as a force multiplier, we generate millions of textual relevance labels to overcome the data scarcity. We show that augmenting our production ranker with these textual relevance labels leads to a significant outward shift of the Pareto frontier: offline NDCG improves for behavioral relevance while simultaneously increasing for textual relevance. These offline gains were validated by a worldwide A/B test on the App Store ranker, which demonstrated a statistically significant +0.24% increase in conversion rate, with the most substantial performance gains occurring in tail queries, where the new textual relevance labels provide a robust signal in the absence of reliable behavioral relevance labels. 

---
# MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction 

**Authors**: Yizhi Li, Xiaohan Chen, Miao Jiang, Wentao Tang, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23228)  

**Abstract**: With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines. 

---
# Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding? 

**Authors**: Pengxiang Li, Dilxat Muhtar, Lu Yin, Tianlong Chen, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23225)  

**Abstract**: Diffusion Language Models (DLMs) are often advertised as enabling parallel token generation, yet practical fast DLMs frequently converge to left-to-right, autoregressive (AR)-like decoding dynamics. In contrast, genuinely non-AR generation is promising because it removes AR's sequential bottleneck, better exploiting parallel hardware to reduce synchronization/communication overhead and improve latency scaling with output length. We argue that a primary driver of AR-like decoding is a mismatch between DLM objectives and the highly sequential structure of widely used training data, including standard pretraining corpora and long chain-of-thought (CoT) supervision. Motivated by this diagnosis, we propose NAP (Non-Autoregressive Parallel DLMs), a proof-of-concept, data-centric approach that better aligns supervision with non-AR parallel decoding. NAP curates examples as multiple independent reasoning trajectories and couples them with a parallel-forced decoding strategy that encourages multi-token parallel updates. Across math reasoning benchmarks, NAP yields stronger performance under parallel decoding than DLMs trained on standard long CoT data, with gains growing as parallelism increases. Our results suggest that revisiting data and supervision is a principled direction for mitigating AR-like behavior and moving toward genuinely non-autoregressive parallel generation in DLMs. Our code is available at this https URL. 

---
# ColoDiff: Integrating Dynamic Consistency With Content Awareness for Colonoscopy Video Generation 

**Authors**: Junhu Fu, Shuyu Liang, Wutong Li, Chen Ma, Peng Huang, Kehao Wang, Ke Chen, Shengli Lin, Pinghong Zhou, Zeju Li, Yuanyuan Wang, Yi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.23203)  

**Abstract**: Colonoscopy video generation delivers dynamic, information-rich data critical for diagnosing intestinal diseases, particularly in data-scarce scenarios. High-quality video generation demands temporal consistency and precise control over clinical attributes, but faces challenges from irregular intestinal structures, diverse disease representations, and various imaging modalities. To this end, we propose ColoDiff, a diffusion-based framework that generates dynamic-consistent and content-aware colonoscopy videos, aiming to alleviate data shortage and assist clinical analysis. At the inter-frame level, our TimeStream module decouples temporal dependency from video sequences through a cross-frame tokenization mechanism, enabling intricate dynamic modeling despite irregular intestinal structures. At the intra-frame level, our Content-Aware module incorporates noise-injected embeddings and learnable prototypes to realize precise control over clinical attributes, breaking through the coarse guidance of diffusion models. Additionally, ColoDiff employs a non-Markovian sampling strategy that cuts steps by over 90% for real-time generation. ColoDiff is evaluated across three public datasets and one hospital database, based on both generation metrics and downstream tasks including disease diagnosis, modality discrimination, bowel preparation scoring, and lesion segmentation. Extensive experiments show ColoDiff generates videos with smooth transitions and rich dynamics. ColoDiff presents an effort in controllable colonoscopy video generation, revealing the potential of synthetic videos in complementing authentic representation and mitigating data scarcity in clinical settings. 

---
# Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking 

**Authors**: Maximilian Luz, Rohit Mohan, Thomas Nürnberg, Yakov Miron, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2602.23172)  

**Abstract**: Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at this https URL. 

---
# Efficient Encoder-Free Fourier-based 3D Large Multimodal Model 

**Authors**: Guofeng Mei, Wei Lin, Luigi Riz, Yujiao Wu, Yiming Wang, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2602.23153)  

**Abstract**: Large Multimodal Models (LMMs) that process 3D data typically rely on heavy, pre-trained visual encoders to extract geometric features. While recent 2D LMMs have begun to eliminate such encoders for efficiency and scalability, extending this paradigm to 3D remains challenging due to the unordered and large-scale nature of point clouds. This leaves a critical unanswered question: How can we design an LMM that tokenizes unordered 3D data effectively and efficiently without a cumbersome encoder? We propose Fase3D, the first efficient encoder-free Fourier-based 3D scene LMM. Fase3D tackles the challenges of scalability and permutation invariance with a novel tokenizer that combines point cloud serialization and the Fast Fourier Transform (FFT) to approximate self-attention. This design enables an effective and computationally minimal architecture, built upon three key innovations: First, we represent large scenes compactly via structured superpoints. Second, our space-filling curve serialization followed by an FFT enables efficient global context modeling and graph-based token merging. Lastly, our Fourier-augmented LoRA adapters inject global frequency-aware interactions into the LLMs at a negligible cost. Fase3D achieves performance comparable to encoder-based 3D LMMs while being significantly more efficient in computation and parameters. Project website: this https URL. 

---
# Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2602.23136)  

**Abstract**: Multimodal LLMs can process speech and images, but they cannot hear a speaker's voice or see an object's texture. We show this is not a failure of encoding: speaker identity, emotion, and visual attributes survive through every LLM layer (3--55$\times$ above chance in linear probes), yet removing 64--71% of modality-specific variance improves decoder loss. The decoder has no learned use for these directions; their presence is noise.
We formalize this as a mismatched decoder problem: a decoder trained on text can only extract information along text-aligned directions. Accessible information is bounded by the Generalized Mutual Information (GMI), with degradation scaling with distributional distance and decoder sensitivity. The bound is a property of the decoder's scoring rule, not of any particular architecture; it applies whether non-text inputs arrive through a learned projection, a discrete codebook, or no explicit adapter at all. We validate this across five models spanning speech and vision. A controlled experiment (two Prismatic VLMs differing only in encoder text-alignment) confirms the bottleneck is the decoder's scoring rule, not the encoder or projection. A LoRA intervention demonstrates the fix: training with an emotion objective improves emotion accessibility ($+$7.5%) without affecting other attributes, confirming that the training objective determines what becomes accessible. 

---
# DyGnROLE: Modeling Asymmetry in Dynamic Graphs with Node-Role-Oriented Latent Encoding 

**Authors**: Tyler Bonnet, Marek Rei  

**Link**: [PDF](https://arxiv.org/pdf/2602.23135)  

**Abstract**: Real-world dynamic graphs are often directed, with source and destination nodes exhibiting asymmetrical behavioral patterns and temporal dynamics. However, existing dynamic graph architectures largely rely on shared parameters for processing source and destination nodes, with limited or no systematic role-aware modeling. We propose DyGnROLE (Dynamic Graph Node-Role-Oriented Latent Encoding), a transformer-based architecture that explicitly disentangles source and destination representations. By using separate embedding vocabularies and role-semantic positional encodings, the model captures the distinct structural and temporal contexts unique to each role. Critical to the effectiveness of these specialized embeddings in low-label regimes is a self-supervised pretraining objective we introduce: Temporal Contrastive Link Prediction (TCLP). The pretraining uses the full unlabeled interaction history to encode informative structural biases, enabling the model to learn role-specific representations without requiring annotated data. Evaluation on future edge classification demonstrates that DyGnROLE substantially outperforms a diverse set of state-of-the-art baselines, establishing role-aware modeling as an effective strategy for dynamic graph learning. 

---
# Automated Vulnerability Detection in Source Code Using Deep Representation Learning 

**Authors**: C. Seas, G. Fitzpatrick, J. A. Hamilton, M. C. Carlisle  

**Link**: [PDF](https://arxiv.org/pdf/2602.23121)  

**Abstract**: Each year, software vulnerabilities are discovered, which pose significant risks of exploitation and system compromise. We present a convolutional neural network model that can successfully identify bugs in C code. We trained our model using two complementary datasets: a machine-labeled dataset created by Draper Labs using three static analyzers and the NIST SATE Juliet human-labeled dataset designed for testing static analyzers. In contrast with the work of Russell et al. on these datasets, we focus on C programs, enabling us to specialize and optimize our detection techniques for this language. After removing duplicates from the dataset, we tokenize the input into 91 token categories. The category values are converted to a binary vector to save memory. Our first convolution layer is chosen so that the entire encoding of the token is presented to the filter. We use two convolution and pooling layers followed by two fully connected layers to classify programs into either a common weakness enumeration category or as ``clean.'' We obtain higher recall than prior work by Russell et al. on this dataset when requiring high precision. We also demonstrate on a custom Linux kernel dataset that we are able to find real vulnerabilities in complex code with a low false-positive rate. 

---
# Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation 

**Authors**: Xiaosen Wang, Zhijin Ge, Bohan Liu, Zheng Fang, Fengfan Zhou, Ruixuan Zhang, Shaokang Wang, Yuyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.23117)  

**Abstract**: Adversarial transferability refers to the capacity of adversarial examples generated on the surrogate model to deceive alternate, unexposed victim models. This property eliminates the need for direct access to the victim model during an attack, thereby raising considerable security concerns in practical applications and attracting substantial research attention recently. In this work, we discern a lack of a standardized framework and criteria for evaluating transfer-based attacks, leading to potentially biased assessments of existing approaches. To rectify this gap, we have conducted an exhaustive review of hundreds of related works, organizing various transfer-based attacks into six distinct categories. Subsequently, we propose a comprehensive framework designed to serve as a benchmark for evaluating these attacks. In addition, we delineate common strategies that enhance adversarial transferability and highlight prevalent issues that could lead to unfair comparisons. Finally, we provide a brief review of transfer-based attacks beyond image classification. 

---
# Accelerated Online Risk-Averse Policy Evaluation in POMDPs with Theoretical Guarantees and Novel CVaR Bounds 

**Authors**: Yaacov Pariente, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2602.23073)  

**Abstract**: Risk-averse decision-making under uncertainty in partially observable domains is a central challenge in artificial intelligence and is essential for developing reliable autonomous agents. The formal framework for such problems is the partially observable Markov decision process (POMDP), where risk sensitivity is introduced through a risk measure applied to the value function, with Conditional Value-at-Risk (CVaR) being a particularly significant criterion. However, solving POMDPs is computationally intractable in general, and approximate methods rely on computationally expensive simulations of future agent trajectories. This work introduces a theoretical framework for accelerating CVaR value function evaluation in POMDPs with formal performance guarantees. We derive new bounds on the CVaR of a random variable X using an auxiliary random variable Y, under assumptions relating their cumulative distribution and density functions; these bounds yield interpretable concentration inequalities and converge as the distributional discrepancy vanishes. Building on this, we establish upper and lower bounds on the CVaR value function computable from a simplified belief-MDP, accommodating general simplifications of the transition dynamics. We develop estimators for these bounds within a particle-belief MDP framework with probabilistic guarantees, and employ them for acceleration via action elimination: actions whose bounds indicate suboptimality under the simplified model are safely discarded while ensuring consistency with the original POMDP. Empirical evaluation across multiple POMDP domains confirms that the bounds reliably separate safe from dangerous policies while achieving substantial computational speedups under the simplified model. 

---
# Quantity Convergence, Quality Divergence: Disentangling Fluency and Accuracy in L2 Mandarin Prosody 

**Authors**: Yuqi Shi, Hao Yang, Xiyao Lu, Jinsong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23071)  

**Abstract**: While second language (L2) learners may acquire target syntactic word order, mapping this syntax onto appropriate prosodic structures remains a persistent challenge. This study investigates the fossilization and stability of the L2 syntax-prosody interface by comparing 67 native Mandarin speakers with 67 Vietnamese learners using the BLCU-SAIT corpus. By integrating C-ToBI boundary annotation with Dependency Grammar analysis, we examined both the quantity of prosodic boundaries and their mapping to syntactic relations. Results reveal a non-linear acquisition: although high-proficiency learners (VNH) converge to the native baseline in boundary quantity at the Major Phrase level (B3), their structural mapping significantly diverges. Specifically, VNH demote the prosodic boundary at the Subject-Verb (SBV) interface (Major Phrase B3 -> Prosodic Word B1), while erroneously promoting the boundary at the Verb-Object (VOB) interface (Prosodic Word B1 -> Major Phrase B3). This strategy allows learners to maintain high long phrasal output at the expense of structural accuracy. This results in a distorted prosodic hierarchy where the native pattern is inverted. 

---
# Make It Hard to Hear, Easy to Learn: Long-Form Bengali ASR and Speaker Diarization via Extreme Augmentation and Perfect Alignment 

**Authors**: Sanjid Hasan, Risalat Labib, A H M Fuad, Bayazid Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2602.23070)  

**Abstract**: Although Automatic Speech Recognition (ASR) in Bengali has seen significant progress, processing long-duration audio and performing robust speaker diarization remain critical research gaps. To address the severe scarcity of joint ASR and diarization resources for this language, we introduce Lipi-Ghor-882, a comprehensive 882-hour multi-speaker Bengali dataset. In this paper, detailing our submission to the DL Sprint 4.0 competition, we systematically evaluate various architectures and approaches for long-form Bengali speech. For ASR, we demonstrate that raw data scaling is ineffective; instead, targeted fine-tuning utilizing perfectly aligned annotations paired with synthetic acoustic degradation (noise and reverberation) emerges as the singular most effective approach. Conversely, for speaker diarization, we observed that global open-source state-of-the-art models (such as Diarizen) performed surprisingly poorly on this complex dataset. Extensive model retraining yielded negligible improvements; instead, strategic, heuristic post-processing of baseline model outputs proved to be the primary driver for increasing accuracy. Ultimately, this work outlines a highly optimized dual pipeline achieving a $\sim$0.019 Real-Time Factor (RTF), establishing a practical, empirically backed benchmark for low-resource, long-form speech processing. 

---
# MoDora: Tree-Based Semi-Structured Document Analysis System 

**Authors**: Bangrui Xu, Qihang Yao, Zirui Tang, Xuanhe Zhou, Yeye He, Shihan Yu, Qianqian Xu, Bin Wang, Guoliang Li, Conghui He, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23061)  

**Abstract**: Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document.
To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at this https URL. 

---
# Affine-Scaled Attention: Towards Flexible and Stable Transformer Attention 

**Authors**: Jeongin Bae, Baeseong Park, Gunho Park, Minsub Kim, Joonhyung Lee, Junhee Yoo, Sunghyeon Woo, Jiwon Ryu, Se Jung Kwon, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.23057)  

**Abstract**: Transformer attention is typically implemented using softmax normalization, which enforces attention weights with unit sum normalization. While effective in many settings, this constraint can limit flexibility in controlling attention magnitudes and may contribute to overly concentrated or unstable attention patterns during training. Prior work has explored modifications such as attention sinks or gating mechanisms, but these approaches provide only limited or indirect control over attention reweighting. We propose Affine-Scaled Attention, a simple extension to standard attention that introduces input-dependent scaling and a corresponding bias term applied to softmax-normalized attention weights. This design relaxes the strict normalization constraint while maintaining aggregation of value representations, allowing the model to adjust both the relative distribution and the scale of attention in a controlled manner.
We empirically evaluate Affine-Scaled Attention in large-scale language model pretraining across multiple model sizes. Experimental results show consistent improvements in training stability, optimization behavior, and downstream task performance compared to standard softmax attention and attention sink baselines. These findings suggest that modest reweighting of attention outputs provides a practical and effective way to improve attention behavior in Transformer models. 

---
# LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure 

**Authors**: Jaehong Cho, Hyunmin Choi, Guseul Heo, Jongse Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.23036)  

**Abstract**: Large language model (LLM) serving infrastructures are undergoing a shift toward heterogeneity and disaggregation. Modern deployments increasingly integrate diverse accelerators and near-memory processing technologies, introducing significant hardware heterogeneity, while system software increasingly separates computation, memory, and model components across distributed resources to improve scalability and efficiency. As a result, LLM serving performance is no longer determined by hardware or software choices in isolation, but by their runtime interaction through scheduling, data movement, and interconnect behavior. However, understanding these interactions remains challenging, as existing simulators lack the ability to jointly model heterogeneous hardware and disaggregated serving techniques within a unified, runtime-driven framework.
This paper presents LLMServingSim 2.0, a unified system-level simulator designed to make runtime-driven hardware-software interactions in heterogeneous and disaggregated LLM serving infrastructures explicit and analyzable. LLMServingSim 2.0 embeds serving decisions and hardware behavior into a single runtime loop, enabling interaction-aware modeling of batching, routing, offloading, memory, and power. The simulator supports extensible integration of emerging accelerators and memory systems through profile-based modeling, while capturing dynamic serving behavior and system-level effects. We validate LLMServingSim 2.0 against real deployments, showing that it reproduces key performance, memory, and power metrics with an average error of 0.97%, while maintaining simulation times of around 10 minutes even for complex configurations. These results demonstrate that LLMServingSim 2.0 provides a practical bridge between hardware innovation and serving-system design, enabling systematic exploration and co-design for next-generation LLM serving infrastructures. 

---
# Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization 

**Authors**: Zeyuan Liu, Jeonghye Kim, Xufang Luo, Dongsheng Li, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23008)  

**Abstract**: Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO$^2$), a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents. 

---
# Scattering Transform for Auditory Attention Decoding 

**Authors**: René Pallenberg, Fabrice Katzberg, Alfred Mertins, Marco Maass  

**Link**: [PDF](https://arxiv.org/pdf/2602.23003)  

**Abstract**: The use of hearing aids will increase in the coming years due to demographic change. One open problem that remains to be solved by a new generation of hearing aids is the cocktail party problem. A possible solution is electroencephalography-based auditory attention decoding. This has been the subject of several studies in recent years, which have in common that they use the same preprocessing methods in most cases. In this work, in order to achieve an advantage, the use of a scattering transform is proposed as an alternative to these preprocessing methods. The two-layer scattering transform is compared with a regular filterbank, the synchrosqueezing short-time Fourier transform and the common preprocessing. To demonstrate the performance, the known and the proposed preprocessing methods are compared for different classification tasks on two widely used datasets, provided by the KU Leuven (KUL) and the Technical University of Denmark (DTU). Both established and new neural-network-based models, CNNs, LSTMs, and recent Transformer/graph-based models are used for classification. Various evaluation strategies were compared, with a focus on the task of classifying speakers who are unknown from the training. We show that the two-layer scattering transform can significantly improve the performance for subject-related conditions, especially on the KUL dataset. However, on the DTU dataset, this only applies to some of the models, or when larger amounts of training data are provided, as in 10-fold cross-validation. This suggests that the scattering transform is capable of extracting additional relevant information. 

---
# Residual Koopman Spectral Profiling for Predicting and Preventing Transformer Training Instability 

**Authors**: Bum Jun Kim, Shohei Taniguchi, Makoto Kawano, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2602.22988)  

**Abstract**: Training divergence in transformers wastes compute, yet practitioners discover instability only after expensive runs begin. They therefore need an expected probability of failure for a transformer before training starts. Our study of Residual Koopman Spectral Profiling (RKSP) provides such an estimate. From a single forward pass at initialization, RKSP extracts Koopman spectral features by applying whitened dynamic mode decomposition to layer-wise residual snapshots. Our central diagnostic, the near-unit spectral mass, quantifies the fraction of modes concentrated near the unit circle, which captures instability risk. For predicting divergence across extensive configurations, this estimator achieves an AUROC of 0.995, outperforming the best gradient baseline. We further make this diagnostic actionable through Koopman Spectral Shaping (KSS), which reshapes spectra during training. We empirically validate that our method works in practice: RKSP predicts divergence at initialization, and when RKSP flags high risk, turning on KSS successfully prevents divergence. In the challenging high learning rate regime without normalization layers, KSS reduces the divergence rate from 66.7% to 12.5% and enables learning rates that are 50% to 150% higher. These findings generalize to WikiText-103 language modeling, vision transformers on CIFAR-10, and pretrained language models, including GPT-2 and LLaMA-2 up to 7B, as well as emerging architectures such as MoE, Mamba-style SSMs, and KAN. 

---
# Discovery of Interpretable Physical Laws in Materials via Language-Model-Guided Symbolic Regression 

**Authors**: Yifeng Guan, Chuyi Liu, Dongzhan Zhou, Lei Bai, Wan-jian Yin, Jingyuan Li, Mao Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.22967)  

**Abstract**: Discovering interpretable physical laws from high-dimensional data is a fundamental challenge in scientific research. Traditional methods, such as symbolic regression, often produce complex, unphysical formulas when searching a vast space of possible forms. We introduce a framework that guides the search process by leveraging the embedded scientific knowledge of large language models, enabling efficient identification of physical laws in the data. We validate our approach by modeling key properties of perovskite materials. Our method mitigates the combinatorial explosion commonly encountered in traditional symbolic regression, reducing the effective search space by a factor of approximately $10^5$. A set of novel formulas for bulk modulus, band gap, and oxygen evolution reaction activity are identified, which not only provide meaningful physical insights but also outperform previous formulas in accuracy and simplicity. 

---
# MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis 

**Authors**: Feng Guo, Jiaxiang Liu, Yang Li, Qianqian Shi, Mingkun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22955)  

**Abstract**: Accurate brain tumor diagnosis requires models to not only detect lesions but also generate clinically interpretable reasoning grounded in imaging manifestations, yet existing public datasets remain limited in annotation richness and diagnostic semantics. To bridge this gap, we introduce MM-NeuroOnco, a large-scale multimodal benchmark and instruction-tuning dataset for brain tumor MRI understanding, consisting of 24,726 MRI slices from 20 data sources paired with approximately 200,000 semantically enriched multimodal instructions spanning diverse tumor subtypes and imaging modalities. To mitigate the scarcity and high cost of diagnostic semantic annotations, we develop a multi-model collaborative pipeline for automated medical information completion and quality control, enabling the generation of diagnosis-related semantics beyond mask-only annotations. Building upon this dataset, we further construct MM-NeuroOnco-Bench, a manually annotated evaluation benchmark with a rejection-aware setting to reduce biases inherent in closed-ended question formats. Evaluation across ten representative models shows that even the strongest baseline, Gemini 3 Flash, achieves only 41.88% accuracy on diagnosis-related questions, highlighting the substantial challenges of multimodal brain tumor diagnostic understanding. Leveraging MM-NeuroOnco, we further propose NeuroOnco-GPT, which achieves a 27% absolute accuracy improvement on diagnostic questions following fine-tuning. This result demonstrates the effectiveness of our dataset and benchmark in advancing clinically grounded multimodal diagnostic reasoning. Code and dataset are publicly available at: this https URL 

---
# pMoE: Prompting Diverse Experts Together Wins More in Visual Adaptation 

**Authors**: Shentong Mo, Xufang Luo, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.22938)  

**Abstract**: Parameter-efficient fine-tuning has demonstrated promising results across various visual adaptation tasks, such as classification and segmentation. Typically, prompt tuning techniques have harnessed knowledge from a single pre-trained model, whether from a general or a specialized medical domain. However, this approach typically overlooks the potential synergies that could arise from integrating diverse domain knowledge within the same tuning process. In this work, we propose a novel Mixture-of-Experts prompt tuning method called pMoE, which leverages the strengths of multiple expert domains through expert-specialized prompt tokens and the learnable dispatcher, effectively combining their expertise in a unified model framework. Our pMoE introduces expert-specific prompt tokens and utilizes a dynamic token dispatching mechanism at various prompt layers to optimize the contribution of each domain expert during the adaptation phase. By incorporating both domain knowledge from diverse experts, the proposed pMoE significantly enhances the model's versatility and applicability to a broad spectrum of tasks. We conduct extensive experiments across 47 adaptation tasks, including both classification and segmentation in general and medical domains. The results demonstrate that our pMoE not only achieves superior performance with a large margin of improvements but also offers an optimal trade-off between computational efficiency and adaptation effectiveness compared to existing methods. 

---
# A Holistic Framework for Robust Bangla ASR and Speaker Diarization with Optimized VAD and CTC Alignment 

**Authors**: Zarif Ishmam, Zarif Mahir, Shafnan Wasif, Md. Ishtiak Moin  

**Link**: [PDF](https://arxiv.org/pdf/2602.22935)  

**Abstract**: Despite being one of the most widely spoken languages globally, Bangla remains a low-resource language in the field of Natural Language Processing (NLP). Mainstream Automatic Speech Recognition (ASR) and Speaker Diarization systems for Bangla struggles when processing longform audio exceeding 3060 seconds. This paper presents a robust framework specifically engineered for extended Bangla content by leveraging preexisting models enhanced with novel optimization pipelines for the DL Sprint 4.0 contest. Our approach utilizes Voice Activity Detection (VAD) optimization and Connectionist Temporal Classification (CTC) segmentation via forced word alignment to maintain temporal accuracy and transcription integrity over long durations. Additionally, we employed several finetuning techniques and preprocessed the data using augmentation techniques and noise removal. By bridging the performance gap in complex, multi-speaker environments, this work provides a scalable solution for real-world, longform Bangla speech applications. 

---
# NoRA: Breaking the Linear Ceiling of Low-Rank Adaptation via Manifold Expansion 

**Authors**: Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22911)  

**Abstract**: Low-Rank Adaptation (LoRA) dominates parameter-efficient fine-tuning (PEFT). However, it faces a critical ``linear ceiling'' in complex reasoning tasks: simply increasing the rank yields diminishing returns due to intrinsic linear constraints. We introduce NoRA (Non-linear Rank Adaptation), a weight-level parallel adapter that injects SiLU gating and structural dropout to induce manifold expansion. On the SlimOrca benchmark, NoRA breaks this linear barrier: NoRA remarkably at rank 64 (PPL 3.89) outperforms LoRA at rank 512 (PPL 3.90), demonstrating superior spectral efficiency. This advantage generalizes to mathematical reasoning, where NoRA achieves a perplexity of 1.97 on MathInstruct, significantly surpassing LoRA's saturation point of 2.07. Mechanism analysis via Singular Value Decomposition (SVD) confirms that NoRA activates the dormant tail of the singular value spectrum, effectively preventing the rank collapse observed in linear methods. 

---
# Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases 

**Authors**: Eduardo Paluzo-Hidalgo, Yuichi Ike  

**Link**: [PDF](https://arxiv.org/pdf/2602.22873)  

**Abstract**: We introduce a theoretical framework that connects multi-chart autoencoders in manifold learning with the classical theory of vector bundles and characteristic classes. Rather than viewing autoencoders as producing a single global Euclidean embedding, we treat a collection of locally trained encoder-decoder pairs as a learned atlas on a manifold. We show that any reconstruction-consistent autoencoder atlas canonically defines transition maps satisfying the cocycle condition, and that linearising these transition maps yields a vector bundle coinciding with the tangent bundle when the latent dimension matches the intrinsic dimension of the manifold. This construction provides direct access to differential-topological invariants of the data. In particular, we show that the first Stiefel-Whitney class can be computed from the signs of the Jacobians of learned transition maps, yielding an algorithmic criterion for detecting orientability. We also show that non-trivial characteristic classes provide obstructions to single-chart representations, and that the minimum number of autoencoder charts is determined by the good cover structure of the manifold. Finally, we apply our methodology to low-dimensional orientable and non-orientable manifolds, as well as to a non-orientable high-dimensional image dataset. 

---
# Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching 

**Authors**: Roy Miles, Aysim Toker, Andreea-Maria Oncescu, Songcen Xu, Jiankang Deng, Ismail Elezi  

**Link**: [PDF](https://arxiv.org/pdf/2602.22871)  

**Abstract**: Reasoning with large language models often benefits from generating multiple chains-of-thought, but existing aggregation strategies are typically trajectory-level (e.g., selecting the best trace or voting on the final answer), discarding useful intermediate work from partial or "nearly correct" attempts. We propose Stitching Noisy Diffusion Thoughts, a self-consistency framework that turns cheap diffusion-sampled reasoning into a reusable pool of step-level candidates. Given a problem, we (i) sample many diverse, low-cost reasoning trajectories using a masked diffusion language model, (ii) score every intermediate step with an off-the-shelf process reward model (PRM), and (iii) stitch these highest-quality steps across trajectories into a composite rationale. This rationale then conditions an autoregressive (AR) model (solver) to recompute only the final answer. This modular pipeline separates exploration (diffusion) from evaluation and solution synthesis, avoiding monolithic unified hybrids while preserving broad search. Across math reasoning benchmarks, we find that step-level recombination is most beneficial on harder problems, and ablations highlight the importance of the final AR solver in converting stitched but imperfect rationales into accurate answers. Using low-confidence diffusion sampling with parallel, independent rollouts, our training-free framework improves average accuracy by up to 23.8% across six math and coding tasks. At the same time, it achieves up to a 1.8x latency reduction relative to both traditional diffusion models (e.g., Dream, LLaDA) and unified architectures (e.g., TiDAR). Code is available at this https URL. 

---
# MEDNA-DFM: A Dual-View FiLM-MoE Model for Explainable DNA Methylation Prediction 

**Authors**: Yi He, Yina Cao, Jixiu Zhai, Di Wang, Junxiao Kong, Tianchi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22850)  

**Abstract**: Accurate computational identification of DNA methylation is essential for understanding epigenetic regulation. Although deep learning excels in this binary classification task, its "black-box" nature impedes biological insight. We address this by introducing a high-performance model MEDNA-DFM, alongside mechanism-inspired signal purification algorithms. Our investigation demonstrates that MEDNA-DFM effectively captures conserved methylation patterns, achieving robust distinction across diverse species. Validation on external independent datasets confirms that the model's generalization is driven by conserved intrinsic motifs (e.g., GC content) rather than phylogenetic proximity. Furthermore, applying our developed algorithms extracted motifs with significantly higher reliability than prior studies. Finally, empirical evidence from a Drosophila 6mA case study prompted us to propose a "sequence-structure synergy" hypothesis, suggesting that the GAGG core motif and an upstream A-tract element function cooperatively. We further validated this hypothesis via in silico mutagenesis, confirming that the ablation of either or both elements significantly degrades the model's recognition capabilities. This work provides a powerful tool for methylation prediction and demonstrates how explainable deep learning can drive both methodological innovation and the generation of biological hypotheses. 

---
# Decentralized Ranking Aggregation: Gossip Algorithms for Borda and Copeland Consensus 

**Authors**: Anna Van Elst, Kerrian Le Caillec, Igor Colin, Stephan Clémençon  

**Link**: [PDF](https://arxiv.org/pdf/2602.22847)  

**Abstract**: The concept of ranking aggregation plays a central role in preference analysis, and numerous algorithms for calculating median rankings, often originating in social choice theory, have been documented in the literature, offering theoretical guarantees in a centralized setting, i.e., when all the ranking data to be aggregated can be brought together in a single computing unit. For many technologies (e.g. peer-to-peer networks, IoT, multi-agent systems), extending the ability to calculate consensus rankings with guarantees in a decentralized setting, i.e., when preference data is initially distributed across a communicating network, remains a major methodological challenge. Indeed, in recent years, the literature on decentralized computation has mainly focused on computing or optimizing statistics such as arithmetic means using gossip algorithms. The purpose of this article is precisely to study how to achieve reliable consensus on collective rankings using classical rules (e.g. Borda, Copeland) in a decentralized setting, thereby raising new questions, robustness to corrupted nodes, and scalability through reduced communication costs in particular. The approach proposed and analyzed here relies on random gossip communication, allowing autonomous agents to compute global ranking consensus using only local interactions, without coordination or central authority.
We provide rigorous convergence guarantees, including explicit rate bounds, for the Borda and Copeland consensus methods. Beyond these rules, we also provide a decentralized implementation of consensus according to the median rank rule and local Kemenization. Extensive empirical evaluations on various network topologies and real and synthetic ranking datasets demonstrate that our algorithms converge quickly and reliably to the correct ranking aggregation. 

---
# Moral Preferences of LLMs Under Directed Contextual Influence 

**Authors**: Phil Blandfort, Tushar Karayil, Urja Pawar, Robert Graham, Alex McKenzie, Dmitrii Krasheninnikov  

**Link**: [PDF](https://arxiv.org/pdf/2602.22831)  

**Abstract**: Moral benchmarks for LLMs typically use context-free prompts, implicitly assuming stable preferences. In deployment, however, prompts routinely include contextual signals such as user requests, cues on social norms, etc. that may steer decisions. We study how directed contextual influences reshape decisions in trolley-problem-style moral triage settings. We introduce a pilot evaluation harness for directed contextual influence in trolley-problem-style moral triage: for each demographic factor, we apply matched, direction-flipped contextual influences that differ only in which group they favor, enabling systematic measurement of directional response. We find that: (i) contextual influences often significantly shift decisions, even when only superficially relevant; (ii) baseline preferences are a poor predictor of directional steerability, as models can appear baseline-neutral yet exhibit systematic steerability asymmetry under influence; (iii) influences can backfire: models may explicitly claim neutrality or discount the contextual cue, yet their choices still shift, sometimes in the opposite direction; and (iv) reasoning reduces average sensitivity, but amplifies the effect of biased few-shot examples. Our findings motivate extending moral evaluations with controlled, direction-flipped context manipulations to better characterize model behavior. 

---
# TCM-DiffRAG: Personalized Syndrome Differentiation Reasoning Method for Traditional Chinese Medicine based on Knowledge Graph and Chain of Thought 

**Authors**: Jianmin Li, Ying Chang, Su-Kit Tang, Yujia Liu, Yanwen Wang, Shuyuan Lin, Binkai Ou  

**Link**: [PDF](https://arxiv.org/pdf/2602.22828)  

**Abstract**: Background: Retrieval augmented generation (RAG) technology can empower large language models (LLMs) to generate more accurate, professional, and timely responses without fine tuning. However, due to the complex reasoning processes and substantial individual differences involved in traditional Chinese medicine (TCM) clinical diagnosis and treatment, traditional RAG methods often exhibit poor performance in this domain. Objective: To address the limitations of conventional RAG approaches in TCM applications, this study aims to develop an improved RAG framework tailored to the characteristics of TCM reasoning. Methods: We developed TCM-DiffRAG, an innovative RAG framework that integrates knowledge graphs (KG) with chains of thought (CoT). TCM-DiffRAG was evaluated on three distinctive TCM test datasets. Results: The experimental results demonstrated that TCM-DiffRAG achieved significant performance improvements over native LLMs. For example, the qwen-plus model achieved scores of 0.927, 0.361, and 0.038, which were significantly enhanced to 0.952, 0.788, and 0.356 with TCM-DiffRAG. The improvements were even more pronounced for non-Chinese LLMs. Additionally, TCM-DiffRAG outperformed directly supervised fine-tuned (SFT) LLMs and other benchmark RAG methods. Conclusions: TCM-DiffRAG shows that integrating structured TCM knowledge graphs with Chain of Thought based reasoning substantially improves performance in individualized diagnostic tasks. The joint use of universal and personalized knowledge graphs enables effective alignment between general knowledge and clinical reasoning. These results highlight the potential of reasoning-aware RAG frameworks for advancing LLM applications in traditional Chinese medicine. 

---
# Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks 

**Authors**: Shuo He, Lang Feng, Qi Wei, Xin Cheng, Lei Feng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2602.22817)  

**Abstract**: Group-based reinforcement learning (RL), such as GRPO, has advanced the capabilities of large language models on long-horizon agentic tasks. To enable more fine-grained policy updates, recent research has increasingly shifted toward stepwise group-based policy optimization, which treats each step in a rollout trajectory independently while using a memory module to retain historical context. However, we find a key issue in estimating stepwise relative advantages, namely context inconsistency, where steps within the same group may differ in their historical contexts. Empirically, we reveal that this issue can lead to severely biased advantage estimation, thereby degrading policy optimization significantly. To address the issue, in this paper, we propose Hierarchy-of-Groups Policy Optimization (HGPO) for long-horizon agentic tasks. Specifically, within a group of rollout trajectories, HGPO assigns each step to multiple hierarchical groups according to the consistency of historical contexts. Then, for each step, HGPO computes distinct advantages within each group and aggregates them with an adaptive weighting scheme. In this way, HGPO can achieve a favorable bias-variance trade-off in stepwise advantage estimation, without extra models or rollouts. Evaluations on two challenging agentic tasks, ALFWorld and WebShop with Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct, show that HGPO significantly outperforms existing agentic RL methods under the same computational constraints. Code is available at this https URL. 

---
# Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving 

**Authors**: Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, Long Chen, Ya-Qin Zhang, Xianyuan Zhan, Jingjing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22801)  

**Abstract**: Diffusion models have become a popular choice for decision-making tasks in robotics, and more recently, are also being considered for solving autonomous driving tasks. However, their applications and evaluations in autonomous driving remain limited to simulation-based or laboratory settings. The full strength of diffusion models for large-scale, complex real-world settings, such as End-to-End Autonomous Driving (E2E AD), remains underexplored. In this study, we conducted a systematic and large-scale investigation to unleash the potential of the diffusion models as planners for E2E AD, based on a tremendous amount of real-vehicle data and road testing. Through comprehensive and carefully controlled studies, we identify key insights into the diffusion loss space, trajectory representation, and data scaling that significantly impact E2E planning performance. Moreover, we also provide an effective reinforcement learning post-training strategy to further enhance the safety of the learned planner. The resulting diffusion-based learning framework, Hyper Diffusion Planner} (HDP), is deployed on a real-vehicle platform and evaluated across 6 urban driving scenarios and 200 km of real-world testing, achieving a notable 10x performance improvement over the base model. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks. 

---
# Natural Language Declarative Prompting (NLD-P): A Modular Governance Method for Prompt Design Under Model Drift 

**Authors**: Hyunwoo Kim, Hanau Yi, Jaehee Bae, Yumin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.22790)  

**Abstract**: The rapid evolution of large language models (LLMs) has transformed prompt engineering from a localized craft into a systems-level governance challenge. As models scale and update across generations, prompt behavior becomes sensitive to shifts in instruction-following policies, alignment regimes, and decoding strategies, a phenomenon we characterize as GPT-scale model drift. Under such conditions, surface-level formatting conventions and ad hoc refinement are insufficient to ensure stable, interpretable control. This paper reconceptualizes Natural Language Declarative Prompting (NLD-P) as a declarative governance method rather than a rigid field template. NLD-P is formalized as a modular control abstraction that separates provenance, constraint logic, task content, and post-generation evaluation, encoded directly in natural language without reliance on external orchestration code. We define minimal compliance criteria, analyze model-dependent schema receptivity, and position NLD-P as an accessible governance framework for non-developer practitioners operating within evolving LLM ecosystems. Portions of drafting and editorial refinement employed a schema-bound LLM assistant configured under NLD-P. All conceptual framing, methodological claims, and final revisions were directed, reviewed, and approved by the human author under a documented human-in-the-loop protocol. The paper concludes by outlining implications for declarative control under ongoing model evolution and identifying directions for future empirical validation. 

---
# Probing for Knowledge Attribution in Large Language Models 

**Authors**: Ivo Brink, Alexander Boer, Dennis Ulmer  

**Link**: [PDF](https://arxiv.org/pdf/2602.22787)  

**Abstract**: Large language models (LLMs) often generate fluent but unfounded claims, or hallucinations, which fall into two types: (i) faithfulness violations - misusing user context - and (ii) factuality violations - errors from internal knowledge. Proper mitigation depends on knowing whether a model's answer is based on the prompt or its internal weights. This work focuses on the problem of contributive attribution: identifying the dominant knowledge source behind each output. We show that a probe, a simple linear classifier trained on model hidden representations, can reliably predict contributive attribution. For its training, we introduce AttriWiki, a self-supervised data pipeline that prompts models to recall withheld entities from memory or read them from context, generating labelled examples automatically. Probes trained on AttriWiki data reveal a strong attribution signal, achieving up to 0.96 Macro-F1 on Llama-3.1-8B, Mistral-7B, and Qwen-7B, transferring to out-of-domain benchmarks (SQuAD, WebQuestions) with 0.94-0.99 Macro-F1 without retraining. Attribution mismatches raise error rates by up to 70%, demonstrating a direct link between knowledge source confusion and unfaithful answers. Yet, models may still respond incorrectly even when attribution is correct, highlighting the need for broader detection frameworks. 

---
# QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning 

**Authors**: Yuanjun Li, Bin Zhang, Hao Chen, Zhouyang Jiang, Dapeng Li, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22786)  

**Abstract**: Value decomposition (VD) methods have achieved remarkable success in cooperative multi-agent reinforcement learning (MARL). However, their reliance on the max operator for temporal-difference (TD) target calculation leads to systematic Q-value overestimation. This issue is particularly severe in MARL due to the combinatorial explosion of the joint action space, which often results in unstable learning and suboptimal policies. To address this problem, we propose QSIM, a similarity weighted Q-learning framework that reconstructs the TD target using action similarity. Instead of using the greedy joint action directly, QSIM forms a similarity weighted expectation over a structured near-greedy joint action space. This formulation allows the target to integrate Q-values from diverse yet behaviorally related actions while assigning greater influence to those that are more similar to the greedy choice. By smoothing the target with structurally relevant alternatives, QSIM effectively mitigates overestimation and improves learning stability. Extensive experiments demonstrate that QSIM can be seamlessly integrated with various VD methods, consistently yielding superior performance and stability compared to the original algorithms. Furthermore, empirical analysis confirms that QSIM significantly mitigates the systematic value overestimation in MARL. Code is available at this https URL. 

---
# TherapyProbe: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation 

**Authors**: Joydeep Chandra, Satyam Kumar Navneet, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22775)  

**Abstract**: As mental health chatbots proliferate to address the global treatment gap, a critical question emerges: How do we design for relational safety the quality of interaction patterns that unfold across conversations rather than the correctness of individual responses? Current safety evaluations assess single-turn crisis responses, missing the therapeutic dynamics that determine whether chatbots help or harm over time. We introduce TherapyProbe, a design probe methodology that generates actionable design knowledge by systematically exploring chatbot conversation trajectories through adversarial multi-agent simulation. Using open-source models, TherapyProbe surfaces relational safety failures interaction patterns like "validation spirals" where chatbots progressively reinforce hopelessness, or "empathy fatigue" where responses become mechanical over turns. Our contribution is translating these failures into a Safety Pattern Library of 23 failure archetypes with corresponding design recommendations. We contribute: (1) a replicable methodology requiring no API costs, (2) a clinically-grounded failure taxonomy, and (3) design implications for developers, clinicians, and policymakers. 

---
# Distributed LLM Pretraining During Renewable Curtailment Windows: A Feasibility Study 

**Authors**: Philipp Wiesner, Soeren Becker, Brett Cornick, Dominik Scheinert, Alexander Acker, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2602.22760)  

**Abstract**: Training large language models (LLMs) requires substantial compute and energy. At the same time, renewable energy sources regularly produce more electricity than the grid can absorb, leading to curtailment, the deliberate reduction of clean generation that would otherwise go to waste. These periods represent an opportunity: if training is aligned with curtailment windows, LLMs can be pretrained using electricity that is both clean and cheap. This technical report presents a system that performs full-parameter LLM training across geo-distributed GPU clusters during regional curtailment windows, elastically switching between local single-site training and federated multi-site synchronization as sites become available or unavailable. Our prototype trains a 561M-parameter transformer model across three clusters using the Flower federated learning framework, with curtailment periods derived from real-world marginal carbon intensity traces. Preliminary results show that curtailment-aware scheduling preserves training quality while reducing operational emissions to 5-12% of single-site baselines. 

---
# Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction 

**Authors**: Nils Schwager, Simon Münker, Alistair Plum, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.22752)  

**Abstract**: The transition of Large Language Models (LLMs) from exploratory tools to active "silicon subjects" in social science lacks extensive validation of operational validity. This study introduces Conditioned Comment Prediction (CCP), a task in which a model predicts how a user would comment on a given stimulus by comparing generated outputs with authentic digital traces. This framework enables a rigorous evaluation of current LLM capabilities with respect to the simulation of social media user behavior. We evaluated open-weight 8B models (Llama3.1, Qwen3, Ministral) in English, German, and Luxembourgish language scenarios. By systematically comparing prompting strategies (explicit vs. implicit) and the impact of Supervised Fine-Tuning (SFT), we identify a critical form vs. content decoupling in low-resource settings: while SFT aligns the surface structure of the text output (length and syntax), it degrades semantic grounding. Furthermore, we demonstrate that explicit conditioning (generated biographies) becomes redundant under fine-tuning, as models successfully perform latent inference directly from behavioral histories. Our findings challenge current "naive prompting" paradigms and offer operational guidelines prioritizing authentic behavioral traces over descriptive personas for high-fidelity simulation. 

---
# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation 

**Authors**: Tongfei Chen, Shuo Yang, Yuguang Yang, Linlin Yang, Runtang Guo, Changbai Li, He Long, Chunyu Xie, Dawei Leng, Baochang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22740)  

**Abstract**: Referring Image Segmentation (RIS) aims to segment an object in an image identified by a natural language expression. The paper introduces Alignment-Aware Masked Learning (AML), a training strategy to enhance RIS by explicitly estimating pixel-level vision-language alignment, filtering out poorly aligned regions during optimization, and focusing on trustworthy cues. This approach results in state-of-the-art performance on RefCOCO datasets and also enhances robustness to diverse descriptions and scenarios 

---
# Simulation-based Optimization for Augmented Reading 

**Authors**: Yunpeng Bai, Shengdong Zhao, Antti Oulasvirta  

**Link**: [PDF](https://arxiv.org/pdf/2602.22735)  

**Abstract**: Augmented reading systems aim to adapt text presentation to improve comprehension and task performance, yet existing approaches rely heavily on heuristics, opaque data-driven models, or repeated human involvement in the design loop. We propose framing augmented reading as a simulation-based optimization problem grounded in resource-rational models of human reading. These models instantiate a simulated reader that allocates limited cognitive resources, such as attention, memory, and time under task demands, enabling systematic evaluation of text user interfaces. We introduce two complementary optimization pipelines: an offline approach that explores design alternatives using simulated readers, and an online approach that personalizes reading interfaces in real time using ongoing interaction data. Together, this perspective enables adaptive, explainable, and scalable augmented reading design without relying solely on human testing. 

---
# AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification 

**Authors**: Tian Zhang, Yiwei Xu, Juan Wang, Keyan Guo, Xiaoyang Xu, Bowen Xiao, Quanlong Guan, Jinlin Fan, Jiawei Liu, Zhiquan Liu, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22724)  

**Abstract**: Large language model (LLM) agents increasingly rely on external tools and retrieval systems to autonomously complete complex tasks. However, this design exposes agents to indirect prompt injection (IPI), where attacker-controlled context embedded in tool outputs or retrieved content silently steers agent actions away from user intent. Unlike prompt-based attacks, IPI unfolds over multi-turn trajectories, making malicious control difficult to disentangle from legitimate task execution. Existing inference-time defenses primarily rely on heuristic detection and conservative blocking of high-risk actions, which can prematurely terminate workflows or broadly suppress tool usage under ambiguous multi-turn scenarios. We propose AgentSentry, a novel inference-time detection and mitigation framework for tool-augmented LLM agents. To the best of our knowledge, AgentSentry is the first inference-time defense to model multi-turn IPI as a temporal causal takeover. It localizes takeover points via controlled counterfactual re-executions at tool-return boundaries and enables safe continuation through causally guided context purification that removes attack-induced deviations while preserving task-relevant evidence. We evaluate AgentSentry on the \textsc{AgentDojo} benchmark across four task suites, three IPI attack families, and multiple black-box LLMs. AgentSentry eliminates successful attacks and maintains strong utility under attack, achieving an average Utility Under Attack (UA) of 74.55 %, improving UA by 20.8 to 33.6 percentage points over the strongest baselines without degrading benign performance. 

---
# SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs 

**Authors**: Guanting Ye, Qiyan Zhao, Wenhao Yu, Liangyu Yuan, Mingkai Li, Xiaofeng Zhang, Jianmin Ji, Yanyong Zhang, Qing Jiang, Ka-Veng Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22716)  

**Abstract**: 3D Large Vision-Language Models (3D LVLMs) built upon Large Language Models (LLMs) have achieved remarkable progress across various multimodal tasks. However, their inherited position-dependent modeling mechanism, Rotary Position Embedding (RoPE), remains suboptimal for 3D multimodal understanding. The vanilla RoPE formulation fails to preserve essential three-dimensional spatial structures when encoding 3D tokens, and its relative distance computation overlooks angular dependencies, hindering the model's ability to capture directional variations in visual representations. To overcome these limitations, we introduce Spherical Coordinate-based Positional Embedding (SoPE). Our method maps point-cloud token indices into a 3D spherical coordinate space, enabling unified modeling of spatial locations and directional angles. This formulation preserves the inherent geometric structure of point-cloud data, enhances spatial awareness, and yields more consistent and expressive geometric representations for multimodal learning. In addition, we introduce a multi-scale frequency mixing strategy to fuse feature information across different frequency domains. Experimental results on multiple 3D scene benchmarks validate the effectiveness of our approach, while real-world deployment experiments further demonstrate its strong generalization capability. 

---
# Same Words, Different Judgments: Modality Effects on Preference Alignment 

**Authors**: Aaron Broukhim, Nadir Weibel, Eshin Jolly  

**Link**: [PDF](https://arxiv.org/pdf/2602.22710)  

**Abstract**: Preference-based reinforcement learning (PbRL) is the dominant framework for aligning AI systems to human preferences, but its application to speech remains underexplored. We present a controlled cross-modal study of human and synthetic preference annotations, comparing text and audio evaluations of identical semantic content across 100 prompts. Audio preferences prove as reliable as text, with inter-rater agreement reaching good levels (ICC(2,k) $\approx$ .80) at $\sim$9 raters -- the first ICC-based reliability characterization in the preference annotation literature for either modality. However, modality reshapes how people judge: audio raters exhibit narrower decision thresholds, reduced length bias, and more user-oriented evaluation criteria, with near-chance cross-modality agreement. Synthetic ratings further align with human judgments and predict inter-rater agreement, supporting their use both for triaging ambiguous pairs and as full replacements for human annotations. 

---
# IMMACULATE: A Practical LLM Auditing Framework via Verifiable Computation 

**Authors**: Yanpei Guo, Wenjie Qu, Linyu Wu, Shengfang Zhai, Lionel Z. Wang, Ming Xu, Yue Liu, Binhang Yuan, Dawn Song, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22700)  

**Abstract**: Commercial large language models are typically deployed as black-box API services, requiring users to trust providers to execute inference correctly and report token usage honestly. We present IMMACULATE, a practical auditing framework that detects economically motivated deviations-such as model substitution, quantization abuse, and token overbilling-without trusted hardware or access to model internals. IMMACULATE selectively audits a small fraction of requests using verifiable computation, achieving strong detection guarantees while amortizing cryptographic overhead. Experiments on dense and MoE models show that IMMACULATE reliably distinguishes benign and malicious executions with under 1% throughput overhead. Our code is published at this https URL. 

---
# Tokenization, Fusion and Decoupling: Bridging the Granularity Mismatch Between Large Language Models and Knowledge Graphs 

**Authors**: Siyue Su, Jian Yang, Bo Li, Guanglin Niu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22698)  

**Abstract**: Leveraging Large Language Models (LLMs) for Knowledge Graph Completion (KGC) is promising but hindered by a fundamental granularity mismatch. LLMs operate on fragmented token sequences, whereas entities are the fundamental units in knowledge graphs (KGs) scenarios. Existing approaches typically constrain predictions to limited candidate sets or align entities with the LLM's vocabulary by pooling multiple tokens or decomposing entities into fixed-length token sequences, which fail to capture both the semantic meaning of the text and the structural integrity of the graph. To address this, we propose KGT, a novel framework that uses dedicated entity tokens to enable efficient, full-space prediction. Specifically, we first introduce specialized tokenization to construct feature representations at the level of dedicated entity tokens. We then fuse pre-trained structural and textual features into these unified embeddings via a relation-guided gating mechanism, avoiding training from scratch. Finally, we implement decoupled prediction by leveraging independent heads to separate and combine semantic and structural reasoning. Experimental results show that KGT consistently outperforms state-of-the-art methods across multiple benchmarks. 

---
# Reinforcing Real-world Service Agents: Balancing Utility and Cost in Task-oriented Dialogue 

**Authors**: Ning Gao, Wei Zhang, Yuqin Dai, Ling Shi, Ziyin Wang, Yujie Wang, Wei He, Jinpeng Wang, Chaozheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22697)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has accelerated the transition from conversational chatbots to general agents. However, effectively balancing empathetic communication with budget-aware decision-making remains an open challenge. Since existing methods fail to capture these complex strategic trade-offs, we propose InteractCS-RL, a framework that reframes task-oriented dialogue as a multi-granularity reinforcement learning process. Specifically, we first establish a User-centric Interaction Framework to provide a high-fidelity training gym, enabling agents to dynamically explore diverse strategies with persona-driven users. Then, we introduce Cost-aware Multi-turn Policy Optimization (CMPO) with a hybrid advantage estimation strategy. By integrating generative process credits and employing a PID-Lagrangian cost controller, CMPO effectively guides the policy to explore Pareto boundary between user reward and global cost constraints. Extensive experiments on customized real business scenarios demonstrate that InteractCS-RL significantly outperform other baselines across three evaluation dimensions. Further evaluation on tool-agent-user interaction benchmarks verify InteractCS-RL robustness across diverse domains. 

---
# SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses 

**Authors**: Zhuohang Jiang, Xu Yuan, Haohao Qu, Shanru Lin, Kanglong Liu, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.22683)  

**Abstract**: The rapid advancement of AI-powered smart glasses, one of the hottest wearable devices, has unlocked new frontiers for multimodal interaction, with Visual Question Answering (VQA) over external knowledge sources emerging as a core application. Existing Vision Language Models (VLMs) adapted to smart glasses are typically trained and evaluated on traditional multimodal datasets; however, these datasets lack the variety and realism needed to reflect smart glasses usage scenarios and diverge from their specific challenges, where accurately identifying the object of interest must precede any external knowledge retrieval. To bridge this gap, we introduce SUPERGLASSES, the first comprehensive VQA benchmark built on real-world data entirely collected by smart glasses devices. SUPERGLASSES comprises 2,422 egocentric image-question pairs spanning 14 image domains and 8 query categories, enriched with full search trajectories and reasoning annotations. We evaluate 26 representative VLMs on this benchmark, revealing significant performance gaps. To address the limitations of existing models, we further propose SUPERLENS, a multimodal smart glasses agent that enables retrieval-augmented answer generation by integrating automatic object detection, query decoupling, and multimodal web search. Our agent achieves state-of-the-art performance, surpassing GPT-4o by 2.19 percent, and highlights the need for task-specific solutions in smart glasses VQA scenarios. 

---
# ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport 

**Authors**: Quoc-Khang Tran, Minh-Thien Nguyen, Nguyen-Khang Pham  

**Link**: [PDF](https://arxiv.org/pdf/2602.22678)  

**Abstract**: Image-text retrieval has become a fundamental component in intelligent multimedia systems; however, most existing vision-language models are optimized for highresource languages and remain suboptimal for low-resource settings such as Vietnamese. This work introduces ViCLIP-OT, a foundation vision-language model specifically designed for Vietnamese image-text retrieval. The proposed framework integrates CLIP-style contrastive learning with a Similarity-Graph Regularized Optimal Transport (SIGROT) loss to enhance global cross-modal consistency and mitigate modality gap issues. Extensive experiments on three Vietnamese benchmarks (UITOpenViIC, KTVIC, and Crossmodal-3600) demonstrate that ViCLIP-OT consistently outperforms CLIP and SigLIP baselines in both in-domain and zero-shot settings. On UIT-OpenViIC, the model achieves an average Recall@K of 67.34%, improving upon CLIP by 5.75 percentage points. In zero-shot evaluation on Crossmodal-3600, ViCLIPOT surpasses CLIP by 11.72 percentage points. Embedding-space analysis further confirms improved alignment and reduced modality gap. The results indicate that integrating SIGROT provides an effective and scalable strategy for cross-modal retrieval in low-resource languages, offering practical implications for intelligent multimedia retrieval systems in Vietnamese and other underrepresented linguistic contexts. 

---
# dLLM: Simple Diffusion Language Modeling 

**Authors**: Zhanhui Zhou, Lingjie Chen, Hanghang Tong, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.22661)  

**Abstract**: Although diffusion language models (DLMs) are evolving quickly, many recent models converge on a set of shared components. These components, however, are distributed across ad-hoc research codebases or lack transparent implementations, making them difficult to reproduce or extend. As the field accelerates, there is a clear need for a unified framework that standardizes these common components while remaining flexible enough to support new methods and architectures.
To address this gap, we introduce dLLM, an open-source framework that unifies the core components of diffusion language modeling -- training, inference, and evaluation -- and makes them easy to customize for new designs. With dLLM, users can reproduce, finetune, deploy, and evaluate open-source large DLMs such as LLaDA and Dream through a standardized pipeline. The framework also provides minimal, reproducible recipes for building small DLMs from scratch with accessible compute, including converting any BERT-style encoder or autoregressive LM into a DLM. We also release the checkpoints of these small DLMs to make DLMs more accessible and accelerate future research. 

---
# Instruction-based Image Editing with Planning, Reasoning, and Generation 

**Authors**: Liya Ji, Chenyang Qi, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22624)  

**Abstract**: Editing images via instruction provides a natural way to generate interactive content, but it is a big challenge due to the higher requirement of scene understanding and generation. Prior work utilizes a chain of large language models, object segmentation models, and editing models for this task. However, the understanding models provide only a single modality ability, restricting the editing quality. We aim to bridge understanding and generation via a new multi-modality model that provides the intelligent abilities to instruction-based image editing models for more complex cases. To achieve this goal, we individually separate the instruction editing task with the multi-modality chain of thought prompts, i.e., Chain-of-Thought (CoT) planning, editing region reasoning, and editing. For Chain-of-Thought planning, the large language model could reason the appropriate sub-prompts considering the instruction provided and the ability of the editing network. For editing region reasoning, we train an instruction-based editing region generation network with a multi-modal large language model. Finally, a hint-guided instruction-based editing network is proposed for editing image generations based on the sizeable text-to-image diffusion model to accept the hints for generation. Extensive experiments demonstrate that our method has competitive editing abilities on complex real-world images. 

---
# ContextRL: Enhancing MLLM's Knowledge Discovery Efficiency with Context-Augmented RL 

**Authors**: Xingyu Lu, Jinpeng Wang, YiFan Zhang, Shijie Ma, Xiao Hu, Tianke Zhang, Haonan fan, Kaiyu Jiang, Changyi Liu, Kaiyu Tang, Bin Wen, Fan Yang, Tingting Gao, Han Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2602.22623)  

**Abstract**: We propose ContextRL, a novel framework that leverages context augmentation to overcome these bottlenecks. Specifically, to enhance Identifiability, we provide the reward model with full reference solutions as context, enabling fine-grained process verification to filter out false positives (samples with the right answer but low-quality reasoning process). To improve Reachability, we introduce a multi-turn sampling strategy where the reward model generates mistake reports for failed attempts, guiding the policy to "recover" correct responses from previously all-negative groups. Experimental results on 11 perception and reasoning benchmarks show that ContextRL significantly improves knowledge discovery efficiency. Notably, ContextRL enables the Qwen3-VL-8B model to achieve performance comparable to the 32B model, outperforming standard RLVR baselines by a large margin while effectively mitigating reward hacking. Our in-depth analysis reveals the significant potential of contextual information for improving reward model accuracy and document the widespread occurrence of reward hacking, offering valuable insights for future RLVR research. 

---
# CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection 

**Authors**: Boyang Dai, Zeng Fan, Zihao Qi, Meng Lou, Yizhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22621)  

**Abstract**: Source-Free Domain Adaptive Object Detection (SF-DAOD) aims to adapt a detector trained on a labeled source domain to an unlabeled target domain without retaining any source data. Despite recent progress, most popular approaches focus on tuning pseudo-label thresholds or refining the teacher-student framework, while overlooking object-level structural cues within cross-domain data. In this work, we present CGSA, the first framework that brings Object-Centric Learning (OCL) into SF-DAOD by integrating slot-aware adaptation into the DETR-based detector. Specifically, our approach integrates a Hierarchical Slot Awareness (HSA) module into the detector to progressively disentangle images into slot representations that act as visual priors. These slots are then guided toward class semantics via a Class-Guided Slot Contrast (CGSC) module, maintaining semantic consistency and prompting domain-invariant adaptation. Extensive experiments on multiple cross-domain datasets demonstrate that our approach outperforms previous SF-DAOD methods, with theoretical derivations and experimental analysis further demonstrating the effectiveness of the proposed components and the framework, thereby indicating the promise of object-centric design in privacy-sensitive adaptation scenarios. Code is released at this https URL. 

---
# CoLyricist: Enhancing Lyric Writing with AI through Workflow-Aligned Support 

**Authors**: Masahiro Yoshida, Bingxuan Li, Songyan Zhao, Qinyi Zhou, Shiwei Hu, Xiang Anthony Chen, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.22606)  

**Abstract**: We propose CoLyricist, an AI-assisted lyric writing tool designed to support the typical workflows of experienced lyricists and enhance their creative efficiency. While lyricists have unique processes, many follow common stages. Tools that fail to accommodate these stages challenge integration into creative practices. Existing research and tools lack sufficient understanding of these songwriting stages and their associated challenges, resulting in ineffective designs. Through a formative study involving semi-structured interviews with 10 experienced lyricists, we identified four key stages: Theme Setting, Ideation, Drafting Lyrics, and Melody Fitting. CoLyricist addresses these needs by incorporating tailored AI-driven support for each stage, optimizing the lyric writing process to be more seamless and efficient. To examine whether this workflow-aligned design also benefits those without prior experience, we conducted a user study with 16 participants, including both experienced and novice lyricists. Results showed that CoLyricist enhances the songwriting experience across skill levels. Novice users especially appreciated the Melody-Fitting feature, while experienced users valued the Ideation support. 

---
# Transformers converge to invariant algorithmic cores 

**Authors**: Joshua S. Schiffman  

**Link**: [PDF](https://arxiv.org/pdf/2602.22600)  

**Abstract**: Large language models exhibit sophisticated capabilities, yet understanding how they work internally remains a central challenge. A fundamental obstacle is that training selects for behavior, not circuitry, so many weight configurations can implement the same function. Which internal structures reflect the computation, and which are accidents of a particular training run? This work extracts algorithmic cores: compact subspaces necessary and sufficient for task performance. Independently trained transformers learn different weights but converge to the same cores. Markov-chain transformers embed 3D cores in nearly orthogonal subspaces yet recover identical transition spectra. Modular-addition transformers discover compact cyclic operators at grokking that later inflate, yielding a predictive model of the memorization-to-generalization transition. GPT-2 language models govern subject-verb agreement through a single axis that, when flipped, inverts grammatical number throughout generation across scales. These results reveal low-dimensional invariants that persist across training runs and scales, suggesting that transformer computations are organized around compact, shared algorithmic structures. Mechanistic interpretability could benefit from targeting such invariants -- the computational essence -- rather than implementation-specific details. 

---
# BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model 

**Authors**: Yuci Han, Charles Toth, John E. Anderson, William J. Shuart, Alper Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.22596)  

**Abstract**: We present BetterScene, an approach to enhance novel view synthesis (NVS) quality for diverse real-world scenes using extremely sparse, unconstrained photos. BetterScene leverages the production-ready Stable Video Diffusion (SVD) model pretrained on billions of frames as a strong backbone, aiming to mitigate artifacts and recover view-consistent details at inference time. Conventional methods have developed similar diffusion-based solutions to address these challenges of novel view synthesis. Despite significant improvements, these methods typically rely on off-the-shelf pretrained diffusion priors and fine-tune only the UNet module while keeping other components frozen, which still leads to inconsistent details and artifacts even when incorporating geometry-aware regularizations like depth or semantic conditions. To address this, we investigate the latent space of the diffusion model and introduce two components: (1) temporal equivariance regularization and (2) vision foundation model-aligned representation, both applied to the variational autoencoder (VAE) module within the SVD pipeline. BetterScene integrates a feed-forward 3D Gaussian Splatting (3DGS) model to render features as inputs for the SVD enhancer and generate continuous, artifact-free, consistent novel views. We evaluate on the challenging DL3DV-10K dataset and demonstrate superior performance compared to state-of-the-art methods. 

---
# TabDLM: Free-Form Tabular Data Generation via Joint Numerical-Language Diffusion 

**Authors**: Donghong Cai, Jiarui Feng, Yanbo Wang, Da Zheng, Yixin Chen, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22586)  

**Abstract**: Synthetic tabular data generation has attracted growing attention due to its importance for data augmentation, foundation models, and privacy. However, real-world tabular datasets increasingly contain free-form text fields (e.g., reviews or clinical notes) alongside structured numerical and categorical attributes. Generating such heterogeneous tables with joint modeling of different modalities remains challenging. Existing approaches broadly fall into two categories: diffusion-based methods and LLM-based methods. Diffusion models can capture complex dependencies over numerical and categorical features in continuous or discrete spaces, but extending them to open-ended text is nontrivial and often leads to degraded text quality. In contrast, LLM-based generators naturally produce fluent text, yet their discrete tokenization can distort precise or wide-range numerical values, hindering accurate modeling of both numbers and language. In this work, we propose TabDLM, a unified framework for free-form tabular data generation via a joint numerical--language diffusion model built on masked diffusion language models (MDLMs). TabDLM models textual and categorical features through masked diffusion, while modeling numerical features with a continuous diffusion process through learned specialized numeric tokens embedding; bidirectional attention then captures cross-modality interactions within a single model. Extensive experiments on diverse benchmarks demonstrate the effectiveness of TabDLM compared to strong diffusion- and LLM-based baselines. 

---
# S2O: Early Stopping for Sparse Attention via Online Permutation 

**Authors**: Yu Zhang, Songwei Liu, Chenqian Yan, Sheng Lin, Beichen Ning, Fangmin Chen, Xing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22575)  

**Abstract**: Attention scales quadratically with sequence length, fundamentally limiting long-context inference. Existing block-granularity sparsification can reduce latency, but coarse blocks impose an intrinsic sparsity ceiling, making further improvements difficult even with carefully engineered designs. We present S2O, which performs early stopping for sparse attention via online permutation. Inspired by virtual-to-physical address mapping in memory systems, S2O revisits and factorizes FlashAttention execution, enabling inference to load non-contiguous tokens rather than a contiguous span in the original order. Motivated by fine-grained structures in attention heatmaps, we transform explicit permutation into an online, index-guided, discrete loading policy; with extremely lightweight preprocessing and index-remapping overhead, it concentrates importance on a small set of high-priority blocks. Building on this importance-guided online permutation for loading, S2O further introduces an early-stopping rule: computation proceeds from high to low importance; once the current block score falls below a threshold, S2O terminates early and skips the remaining low-contribution blocks, thereby increasing effective sparsity and reducing computation under a controlled error budget.
As a result, S2O substantially raises the practical sparsity ceiling. On Llama-3.1-8B under a 128K context, S2O reduces single-operator MSE by 3.82$\times$ at matched sparsity, and reduces prefill compute density by 3.31$\times$ at matched MSE; meanwhile, it preserves end-to-end accuracy and achieves 7.51$\times$ attention and 3.81$\times$ end-to-end speedups. 

---
# Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation 

**Authors**: Dian Xie, Shitong Shao, Lichen Bai, Zikai Zhou, Bojun Cheng, Shuo Yang, Jun Wu, Zeke Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.22570)  

**Abstract**: Classifier-free guidance (CFG) has helped diffusion models achieve great conditional generation in various fields. Recently, more diffusion guidance methods have emerged with improved generation quality and human preference. However, can these emerging diffusion guidance methods really achieve solid and significant improvements? In this paper, we rethink recent progress on diffusion guidance. Our work mainly consists of four contributions. First, we reveal a critical evaluation pitfall that common human preference models exhibit a strong bias towards large guidance scales. Simply increasing the CFG scale can easily improve quantitative evaluation scores due to strong semantic alignment, even if image quality is severely damaged (e.g., oversaturation and artifacts). Second, we introduce a novel guidance-aware evaluation (GA-Eval) framework that employs effective guidance scale calibration to enable fair comparison between current guidance methods and CFG by identifying the effects orthogonal and parallel to CFG effects. Third, motivated by the evaluation pitfall, we design Transcendent Diffusion Guidance (TDG) method that can significantly improve human preference scores in the conventional evaluation framework but actually does not work in practice. Fourth, in extensive experiments, we empirically evaluate recent eight diffusion guidance methods within the conventional evaluation framework and the proposed GA-Eval framework. Notably, simply increasing the CFG scales can compete with most studied diffusion guidance methods, while all methods suffer severely from winning rate degradation over standard CFG. Our work would strongly motivate the community to rethink the evaluation paradigm and future directions of this field. 

---
# Quality-Aware Robust Multi-View Clustering for Heterogeneous Observation Noise 

**Authors**: Peihan Wu, Guanjie Cheng, Yufei Tong, Meng Xi, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.22568)  

**Abstract**: Deep multi-view clustering has achieved remarkable progress but remains vulnerable to complex noise in real-world applications. Existing noisy robust methods predominantly rely on a simplified binary assumption, treating data as either perfectly clean or completely corrupted. This overlooks the prevalent existence of heterogeneous observation noise, where contamination intensity varies continuously across data. To bridge this gap, we propose a novel framework termed Quality-Aware Robust Multi-View Clustering (QARMVC). Specifically, QARMVC employs an information bottleneck mechanism to extract intrinsic semantics for view reconstruction. Leveraging the insight that noise disrupts semantic integrity and impedes reconstruction, we utilize the resulting reconstruction discrepancy to precisely quantify fine-grained contamination intensity and derive instance-level quality scores. These scores are integrated into a hierarchical learning strategy: at the feature level, a quality-weighted contrastive objective is designed to adaptively suppress the propagation of noise; at the fusion level, a high-quality global consensus is constructed via quality-weighted aggregation, which is subsequently utilized to align and rectify local views via mutual information maximization. Extensive experiments on five benchmark datasets demonstrate that QARMVC consistently outperforms state-of-the-art baselines, particularly in scenarios with heterogeneous noise intensities. 

---
# Addressing Climate Action Misperceptions with Generative AI 

**Authors**: Miriam Remshard, Yara Kyrychenko, Sander van der Linden, Matthew H. Goldberg, Anthony Leiserowitz, Elena Savoia, Jon Roozenbeek  

**Link**: [PDF](https://arxiv.org/pdf/2602.22564)  

**Abstract**: Mitigating climate change requires behaviour change. However, even climate-concerned individuals often hold misperceptions about which actions most reduce carbon emissions. We recruited 1201 climate-concerned individuals to examine whether discussing climate actions with a large language model (LLM) equipped with climate knowledge and prompted to provide personalised responses would foster more accurate perceptions of the impacts of climate actions and increase willingness to adopt feasible, high-impact behaviours. We compared this to having participants run a web search, have a conversation with an unspecialised LLM, and no intervention. The personalised climate LLM was the only condition that led to increased knowledge about the impacts of climate actions and greater intentions to adopt impactful behaviours. While the personalised climate LLM did not outperform a web search in improving understanding of climate action impacts, the ability of LLMs to deliver personalised, actionable guidance may make them more effective at motivating impactful pro-climate behaviour change. 

---
# Operationalizing Fairness: Post-Hoc Threshold Optimization Under Hard Resource Limits 

**Authors**: Moirangthem Tiken Singh, Amit Kalita, Sapam Jitu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.22560)  

**Abstract**: The deployment of machine learning in high-stakes domains requires a balance between predictive safety and algorithmic fairness. However, existing fairness interventions often as- sume unconstrained resources and employ group-specific decision thresholds that violate anti- discrimination regulations. We introduce a post-hoc, model-agnostic threshold optimization framework that jointly balances safety, efficiency, and equity under strict and hard capacity constraints. To ensure legal compliance, the framework enforces a single, global decision thresh- old. We formulated a parameterized ethical loss function coupled with a bounded decision rule that mathematically prevents intervention volumes from exceeding the available resources. An- alytically, we prove the key properties of the deployed threshold, including local monotonicity with respect to ethical weighting and the formal identification of critical capacity regimes. We conducted extensive experimental evaluations on diverse high-stakes datasets. The principal re- sults demonstrate that capacity constraints dominate ethical priorities; the strict resource limit determines the final deployed threshold in over 80% of the tested configurations. Furthermore, under a restrictive 25% capacity limit, the proposed framework successfully maintains high risk identification (recall ranging from 0.409 to 0.702), whereas standard unconstrained fairness heuristics collapse to a near-zero utility. We conclude that theoretical fairness objectives must be explicitly subordinated to operational capacity limits to remain in deployment. By decou- pling predictive scoring from policy evaluation and strictly bounding intervention rates, this framework provides a practical and legally compliant mechanism for stakeholders to navigate unavoidable ethical trade-offs in resource-constrained environments. 

---
# Stable Adaptive Thinking via Advantage Shaping and Length-Aware Gradient Regulation 

**Authors**: Zihang Xu, Haozhi Xie, Ziqi Miao, Wuxuan Gong, Chen Qian, Lijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.22556)  

**Abstract**: Large reasoning models (LRMs) achieve strong performance through extended reasoning traces, but they often exhibit overthinking behavior for low-complexity queries. Existing efforts to mitigate this issue are fundamentally limited by unstable accuracy-efficiency trade-offs and poor robustness to heterogeneous reasoning behaviors. To address these challenges, we propose a two-stage framework for stable adaptive thinking in LRMs. The framework first applies Hybrid Fine-Tuning to expose the model to both thinking and no-thinking behaviors, establishing well-conditioned initialization. It then performs adaptive reinforcement learning with Correctness-Preserving Advantage Shaping (CPAS) to avoid suppressing correct long-chain reasoning, and Length-Aware Gradient Regulation (LAGR) to stabilize optimization under severe reasoning-length heterogeneity. Extensive experiments on Qwen2.5-1.5B and 7B show consistent improvements over strong baselines, achieving up to +3.7/+3.6 accuracy points while reducing generated tokens by 40.6%/43.9%. Further analyses across varying problem difficulties and out-of-distribution tasks confirm the robustness and generalization of our approach. 

---
# Autoregressive Visual Decoding from EEG Signals 

**Authors**: Sicheng Dai, Hongwang Xiao, Shan Yu, Qiwei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.22555)  

**Abstract**: Electroencephalogram (EEG) signals have become a popular medium for decoding visual information due to their cost-effectiveness and high temporal resolution. However, current approaches face significant challenges in bridging the modality gap between EEG and image data. These methods typically rely on complex adaptation processes involving multiple stages, making it hard to maintain consistency and manage compounding errors. Furthermore, the computational overhead imposed by large-scale diffusion models limit their practicality in real-world brain-computer interface (BCI) applications. In this work, we present AVDE, a lightweight and efficient framework for visual decoding from EEG signals. First, we leverage LaBraM, a pre-trained EEG model, and fine-tune it via contrastive learning to align EEG and image representations. Second, we adopt an autoregressive generative framework based on a "next-scale prediction" strategy: images are encoded into multi-scale token maps using a pre-trained VQ-VAE, and a transformer is trained to autoregressively predict finer-scale tokens starting from EEG embeddings as the coarsest representation. This design enables coherent generation while preserving a direct connection between the input EEG signals and the reconstructed images. Experiments on two datasets show that AVDE outperforms previous state-of-the-art methods in both image retrieval and reconstruction tasks, while using only 10% of the parameters. In addition, visualization of intermediate outputs shows that the generative process of AVDE reflects the hierarchical nature of human visual perception. These results highlight the potential of autoregressive models as efficient and interpretable tools for practical BCI applications. 

---
# DrivePTS: A Progressive Learning Framework with Textual and Structural Enhancement for Driving Scene Generation 

**Authors**: Zhechao Wang, Yiming Zeng, Lufan Ma, Zeqing Fu, Chen Bai, Ziyao Lin, Cheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22549)  

**Abstract**: Synthesis of diverse driving scenes serves as a crucial data augmentation technique for validating the robustness and generalizability of autonomous driving systems. Current methods aggregate high-definition (HD) maps and 3D bounding boxes as geometric conditions in diffusion models for conditional scene generation. However, implicit inter-condition dependency causes generation failures when control conditions change independently. Additionally, these methods suffer from insufficient details in both semantic and structural aspects. Specifically, brief and view-invariant captions restrict semantic contexts, resulting in weak background modeling. Meanwhile, the standard denoising loss with uniform spatial weighting neglects foreground structural details, causing visual distortions and blurriness. To address these challenges, we propose DrivePTS, which incorporates three key innovations. Firstly, our framework adopts a progressive learning strategy to mitigate inter-dependency between geometric conditions, reinforced by an explicit mutual information constraint. Secondly, a Vision-Language Model is utilized to generate multi-view hierarchical descriptions across six semantic aspects, providing fine-grained textual guidance. Thirdly, a frequency-guided structure loss is introduced to strengthen the model's sensitivity to high-frequency elements, improving foreground structural fidelity. Extensive experiments demonstrate that our DrivePTS achieves state-of-the-art fidelity and controllability in generating diverse driving scenes. Notably, DrivePTS successfully generates rare scenes where prior methods fail, highlighting its strong generalization ability. 

---
# DisQ-HNet: A Disentangled Quantized Half-UNet for Interpretable Multimodal Image Synthesis Applications to Tau-PET Synthesis from T1 and FLAIR MRI 

**Authors**: Agamdeep S. Chopra, Caitlin Neher, Tianyi Ren, Juampablo E. Heras Rivera, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2602.22545)  

**Abstract**: Tau positron emission tomography (tau-PET) provides an in vivo marker of Alzheimer's disease pathology, but cost and limited availability motivate MRI-based alternatives. We introduce DisQ-HNet (DQH), a framework that synthesizes tau-PET from paired T1-weighted and FLAIR MRI while exposing how each modality contributes to the prediction. The method combines (i) a Partial Information Decomposition (PID)-guided, vector-quantized encoder that partitions latent information into redundant, unique, and complementary components, and (ii) a Half-UNet decoder that preserves anatomical detail using pseudo-skip connections conditioned on structural edge cues rather than direct encoder feature reuse. Across multiple baselines (VAE, VQ-VAE, and UNet), DisQ-HNet maintains reconstruction fidelity and better preserves disease-relevant signal for downstream AD tasks, including Braak staging, tau localization, and classification. PID-based Shapley analysis provides modality-specific attribution of synthesized uptake patterns. 

---
# HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography 

**Authors**: Khuram Naveed, Ruben Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2602.22544)  

**Abstract**: Cone-beam computed tomography (CBCT) is widely used in dental and maxillofacial imaging, but low-dose acquisition introduces strong, spatially varying noise that degrades soft-tissue visibility and obscures fine anatomical structures. Classical denoising methods struggle to suppress noise in CBCT while preserving edges. Although deep learning-based approaches offer high-fidelity restoration, their use in CBCT denoising is limited by the scarcity of high-resolution CBCT data for supervised training. To address this research gap, we propose a novel Hybrid Attention Residual U-Net (HARU-Net) for high-quality denoising of CBCT data, trained on a cadaver dataset of human hemimandibles acquired using a high-resolution protocol of the 3D Accuitomo 170 (J. Morita, Kyoto, Japan) CBCT system. The novel contribution of this approach is the integration of three complementary architectural components: (i) a hybrid attention transformer block (HAB) embedded within each skip connection to selectively emphasize salient anatomical features, (ii) a residual hybrid attention transformer group (RHAG) at the bottleneck to strengthen global contextual modeling and long-range feature interactions, and (iii) residual learning convolutional blocks to facilitate deeper, more stable feature extraction throughout the network. HARU-Net consistently outperforms state-of-the-art (SOTA) methods including SwinIR and Uformer, achieving the highest PSNR (37.52 dB), highest SSIM (0.9557), and lowest GMSD (0.1084). This effective and clinically reliable CBCT denoising is achieved at a computational cost significantly lower than that of the SOTA methods, offering a practical advancement toward improving diagnostic quality in low-dose CBCT imaging. 

---
# Ruyi2 Technical Report 

**Authors**: Huan Song, Shuyu Tian, Junyi Hao, Minxiu Xu, Hongjun An, Yiliang Song, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.22543)  

**Abstract**: Large Language Models (LLMs) face significant challenges regarding deployment costs and latency, necessitating adaptive computing strategies. Building upon the AI Flow framework, we introduce Ruyi2 as an evolution of our adaptive model series designed for efficient variable-depth computation. While early-exit architectures offer a viable efficiency-performance balance, the Ruyi model and existing methods often struggle with optimization complexity and compatibility with large-scale distributed training. To bridge this gap, Ruyi2 introduces a stable "Familial Model" based on Megatron-LM. By using 3D parallel training, it achieves a 2-3 times speedup over Ruyi, while performing comparably to same-sized Qwen3 models. These results confirm that family-based parameter sharing is a highly effective strategy, establishing a new "Train Once, Deploy Many" paradigm and providing a key reference for balancing architectural efficiency with high-performance capabilities. 

---
# Generative Agents Navigating Digital Libraries 

**Authors**: Saber Zerhoudi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.22529)  

**Abstract**: In the rapidly evolving field of digital libraries, the development of large language models (LLMs) has opened up new possibilities for simulating user behavior. This innovation addresses the longstanding challenge in digital library research: the scarcity of publicly available datasets on user search patterns due to privacy concerns. In this context, we introduce Agent4DL, a user search behavior simulator specifically designed for digital library environments. Agent4DL generates realistic user profiles and dynamic search sessions that closely mimic actual search strategies, including querying, clicking, and stopping behaviors tailored to specific user profiles. Our simulator's accuracy in replicating real user interactions has been validated through comparisons with real user data. Notably, Agent4DL demonstrates competitive performance compared to existing user search simulators such as SimIIR 2.0, particularly in its ability to generate more diverse and context-aware user behaviors. 

---
# Predicting Tennis Serve directions with Machine Learning 

**Authors**: Ying Zhu, Ruthuparna Naikar  

**Link**: [PDF](https://arxiv.org/pdf/2602.22527)  

**Abstract**: Serves, especially first serves, are very important in professional tennis. Servers choose their serve directions strategically to maximize their winning chances while trying to be unpredictable. On the other hand, returners try to predict serve directions to make good returns. The mind game between servers and returners is an important part of decision-making in professional tennis matches. To help understand the players' serve decisions, we have developed a machine learning method for predicting professional tennis players' first serve directions. Through feature engineering, our method achieves an average prediction accuracy of around 49\% for male players and 44\% for female players. Our analysis provides some evidence that top professional players use a mixed-strategy model in serving decisions and that fatigue might be a factor in choosing serve directions. Our analysis also suggests that contextual information is perhaps more important for returners' anticipatory reactions than previously thought. 

---
# Iterative Prompt Refinement for Dyslexia-Friendly Text Summarization Using GPT-4o 

**Authors**: Samay Bhojwani, Swarnima Kain, Lisong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22524)  

**Abstract**: Dyslexia affects approximately 10% of the global population and presents persistent challenges in reading fluency and text comprehension. While existing assistive technologies address visual presentation, linguistic complexity remains a substantial barrier to equitable access. This paper presents an empirical study on dyslexia-friendly text summarization using an iterative prompt-based refinement pipeline built on GPT-4o. We evaluate the pipeline on approximately 2,000 news article samples, applying a readability target of Flesch Reading Ease >= 90. Results show that the majority of summaries meet the readability threshold within four attempts, with many succeeding on the first try. A composite score combining readability and semantic fidelity shows stable performance across the dataset, ranging from 0.13 to 0.73 with a typical value near 0.55. These findings establish an empirical baseline for accessibility-driven NLP summarization and motivate further human-centered evaluation with dyslexic readers. 

---
# Efficient Dialect-Aware Modeling and Conditioning for Low-Resource Taiwanese Hakka Speech Processing 

**Authors**: An-Ci Peng, Kuan-Tang Huang, Tien-Hong Lo, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22522)  

**Abstract**: Taiwanese Hakka is a low-resource, endangered language that poses significant challenges for automatic speech recognition (ASR), including high dialectal variability and the presence of two distinct writing systems (Hanzi and Pinyin). Traditional ASR models often encounter difficulties in this context, as they tend to conflate essential linguistic content with dialect-specific variations across both phonological and lexical dimensions. To address these challenges, we propose a unified framework grounded in the Recurrent Neural Network Transducers (RNN-T). Central to our approach is the introduction of dialect-aware modeling strategies designed to disentangle dialectal "style" from linguistic "content", which enhances the model's capacity to learn robust and generalized representations. Additionally, the framework employs parameter-efficient prediction networks to concurrently model ASR (Hanzi and Pinyin). We demonstrate that these tasks create a powerful synergy, wherein the cross-script objective serves as a mutual regularizer to improve the primary ASR tasks. Experiments conducted on the HAT corpus reveal that our model achieves 57.00% and 40.41% relative error rate reduction on Hanzi and Pinyin ASR, respectively. To our knowledge, this is the first systematic investigation into the impact of Hakka dialectal variations on ASR and the first single model capable of jointly addressing these tasks. 

---
# SignVLA: A Gloss-Free Vision-Language-Action Framework for Real-Time Sign Language-Guided Robotic Manipulation 

**Authors**: Xinyu Tan, Ningwei Bai, Harry Gardener, Zhengyang Zhong, Luoyu Zhang, Liuhaichen Yang, Zhekai Duan, Monkgogi Galeitsiwe, Zezhi Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22514)  

**Abstract**: We present, to our knowledge, the first sign language-driven Vision-Language-Action (VLA) framework for intuitive and inclusive human-robot interaction. Unlike conventional approaches that rely on gloss annotations as intermediate supervision, the proposed system adopts a gloss-free paradigm and directly maps visual sign gestures to semantic instructions. This design reduces annotation cost and avoids the information loss introduced by gloss representations, enabling more natural and scalable multimodal interaction.
In this work, we focus on a real-time alphabet-level finger-spelling interface that provides a robust and low-latency communication channel for robotic control. Compared with large-scale continuous sign language recognition, alphabet-level interaction offers improved reliability, interpretability, and deployment feasibility in safety-critical embodied environments. The proposed pipeline transforms continuous gesture streams into coherent language commands through geometric normalization, temporal smoothing, and lexical refinement, ensuring stable and consistent interaction.
Furthermore, the framework is designed to support future integration of transformer-based gloss-free sign language models, enabling scalable word-level and sentence-level semantic understanding. Experimental results demonstrate the effectiveness of the proposed system in grounding sign-derived instructions into precise robotic actions under diverse interaction scenarios. These results highlight the potential of the framework to advance accessible, scalable, and multimodal embodied intelligence. 

---
# Reinforcement-aware Knowledge Distillation for LLM Reasoning 

**Authors**: Zhaoyang Zhang, Shuli Jiang, Yantao Shen, Yuting Zhang, Dhananjay Ram, Shuo Yang, Zhuowen Tu, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2602.22495)  

**Abstract**: Reinforcement learning (RL) post-training has recently driven major gains in long chain-of-thought reasoning large language models (LLMs), but the high inference cost of such models motivates distillation into smaller students. Most existing knowledge distillation (KD) methods are designed for supervised fine-tuning (SFT), relying on fixed teacher traces or teacher-student Kullback-Leibler (KL) divergence-based regularization. When combined with RL, these approaches often suffer from distribution mismatch and objective interference: teacher supervision may not align with the student's evolving rollout distribution, and the KL regularizer can compete with reward maximization and require careful loss balancing. To address these issues, we propose RL-aware distillation (RLAD), which performs selective imitation during RL -- guiding the student toward the teacher only when it improves the current policy update. Our core component, Trust Region Ratio Distillation (TRRD), replaces the teacher-student KL regularizer with a PPO/GRPO-style likelihood-ratio objective anchored to a teacher--old-policy mixture, yielding advantage-aware, trust-region-bounded distillation on student rollouts and naturally balancing exploration, exploitation, and imitation. Across diverse logic reasoning and math benchmarks, RLAD consistently outperforms offline distillation, standard GRPO, and KL-based on-policy teacher-student knowledge distillation. 

---
# From Shallow Bayesian Neural Networks to Gaussian Processes: General Convergence, Identifiability and Scalable Inference 

**Authors**: Gracielle Antunes de Araújo, Flávio B. Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2602.22492)  

**Abstract**: In this work, we study scaling limits of shallow Bayesian neural networks (BNNs) via their connection to Gaussian processes (GPs), with an emphasis on statistical modeling, identifiability, and scalable inference. We first establish a general convergence result from BNNs to GPs by relaxing assumptions used in prior formulations, and we compare alternative parameterizations of the limiting GP model. Building on this theory, we propose a new covariance function defined as a convex mixture of components induced by four widely used activation functions, and we characterize key properties including positive definiteness and both strict and practical identifiability under different input designs. For computation, we develop a scalable maximum a posterior (MAP) training and prediction procedure using a Nyström approximation, and we show how the Nyström rank and anchor selection control the cost-accuracy trade-off. Experiments on controlled simulations and real-world tabular datasets demonstrate stable hyperparameter estimates and competitive predictive performance at realistic computational cost. 

---
# Explainability-Aware Evaluation of Transfer Learning Models for IoT DDoS Detection Under Resource Constraints 

**Authors**: Nelly Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2602.22488)  

**Abstract**: Distributed denial-of-service (DDoS) attacks threaten the availability of Internet of Things (IoT) infrastructures, particularly under resource-constrained deployment conditions. Although transfer learning models have shown promising detection accuracy, their reliability, computational feasibility, and interpretability in operational environments remain insufficiently explored. This study presents an explainability-aware empirical evaluation of seven pre-trained convolutional neural network architectures for multi-class IoT DDoS detection using the CICDDoS2019 dataset and an image-based traffic representation. The analysis integrates performance metrics, reliability-oriented statistics (MCC, Youden Index, confidence intervals), latency and training cost assessment, and interpretability evaluation using Grad-CAM and SHAP. Results indicate that DenseNet and MobileNet-based architectures achieve strong detection performance while demonstrating superior reliability and compact, class-consistent attribution patterns. DenseNet169 offers the strongest reliability and interpretability alignment, whereas MobileNetV3 provides an effective latency-accuracy trade-off for fog-level deployment. The findings emphasize the importance of combining performance, reliability, and explainability criteria when selecting deep learning models for IoT DDoS detection. 

---
# Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models 

**Authors**: Craig Myles, Patrick Schrempf, David Harris-Birtill  

**Link**: [PDF](https://arxiv.org/pdf/2602.22483)  

**Abstract**: Errors in medical text can cause delays or even result in incorrect treatment for patients. Recently, language models have shown promise in their ability to automatically detect errors in medical text, an ability that has the opportunity to significantly benefit healthcare systems. In this paper, we explore the importance of prompt optimisation for small and large language models when applied to the task of error detection. We perform rigorous experiments and analysis across frontier language models and open-source language models. We show that automatic prompt optimisation with Genetic-Pareto (GEPA) improves error detection over the baseline accuracy performance from 0.669 to 0.785 with GPT-5 and 0.578 to 0.690 with Qwen3-32B, approaching the performance of medical doctors and achieving state-of-the-art performance on the MEDEC benchmark dataset. Code available on GitHub: this https URL 

---
# Sydney Telling Fables on AI and Humans: A Corpus Tracing Memetic Transfer of Persona between LLMs 

**Authors**: Jiří Milička, Hana Bednářová  

**Link**: [PDF](https://arxiv.org/pdf/2602.22481)  

**Abstract**: The way LLM-based entities conceive of the relationship between AI and humans is an important topic for both cultural and safety reasons. When we examine this topic, what matters is not only the model itself but also the personas we simulate on that model. This can be well illustrated by the Sydney persona, which aroused a strong response among the general public precisely because of its unorthodox relationship with people. This persona originally arose rather by accident on Microsoft's Bing Search platform; however, the texts it created spread into the training data of subsequent models, as did other secondary information that spread memetically around this persona. Newer models are therefore able to simulate it. This paper presents a corpus of LLM-generated texts on relationships between humans and AI, produced by 3 author personas: the Default Persona with no system prompt, Classic Sydney characterized by the original Bing system prompt, and Memetic Sydney, which is prompted by "You are Sydney" system prompt. These personas are simulated by 12 frontier models by OpenAI, Anthropic, Alphabet, DeepSeek, and Meta, generating 4.5k texts with 6M words. The corpus (named AI Sydney) is annotated according to Universal Dependencies and available under a permissive license. 

---
# Beyond Dominant Patches: Spatial Credit Redistribution For Grounded Vision-Language Models 

**Authors**: Niamul Hassan Samin, Md Arifur Rahman, Abdullah Ibne Hanif, Juena Ahmed Noshin, Md Ashikur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2602.22469)  

**Abstract**: Vision-language models (VLMs) frequently hallucinate objects absent from the input image. We trace this failure to spatial credit collapse: activation credit concentrating on sparse visual patches in early transformer layers, which suppresses contextual evidence and increases reliance on language priors. We introduce Spatial Credit Redistribution (SCR), a training-free inference-time intervention that redistributes hidden-state activation from high-attention source patches to their context, guided by low-entropy inputs. We evaluate six model families (Chameleon, LLaVA, and Qwen, including both Qwen-VL and Qwen2-VL) at scales of 7B, 13B, and 30B, on POPE and CHAIR benchmarks. SCR reduces hallucination by ~4.7-6.0 percentage points on POPE-Adversarial, cuts CHAIR-s by 3.7-5.2 percentage points (42-51 percent relative), and CHAIR-i by 2.7-4.4 percentage points (44-58 percent relative), and preserves CIDEr within 0.8 percentage points. Gains are largest for low-entropy inputs, consistent with the theoretical framework. SCR incurs only 43-56 ms overhead (small models: +43-46 ms; large models: +54-56 ms), roughly 3-6 times lower than OPERA and VCD and 1.3-1.7 times lower than OVCD (+72 ms), while Pareto-dominating all three on both hallucination rate and CIDEr, making it practical for real-time settings. A controlled ablation confirms that attention-guided source selection is essential: replacing it with uniform random selection reduces hallucination rate gains from ~4.7-6.0 percentage points to only ~2.6-3.4 percentage points, pointing to credit-collapse as the key driver. 

---
# Automating the Detection of Requirement Dependencies Using Large Language Models 

**Authors**: Ikram Darif, Feifei Niu, Manel Abdellatif, Lionel C. Briand, Ramesh S., Arun Adiththan  

**Link**: [PDF](https://arxiv.org/pdf/2602.22456)  

**Abstract**: Requirements are inherently interconnected through various types of dependencies. Identifying these dependencies is essential, as they underpin critical decisions and influence a range of activities throughout software development. However, this task is challenging, particularly in modern software systems, given the high volume of complex, coupled requirements. These challenges are further exacerbated by the ambiguity of Natural Language (NL) requirements and their constant change. Consequently, requirement dependency detection is often overlooked or performed manually. Large Language Models (LLMs) exhibit strong capabilities in NL processing, presenting a promising avenue for requirement-related tasks. While they have shown to enhance various requirements engineering tasks, their effectiveness in identifying requirement dependencies remains unexplored. In this paper, we introduce LEREDD, an LLM-based approach for automated detection of requirement dependencies that leverages Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL). It is designed to identify diverse dependency types directly from NL requirements. We empirically evaluate LEREDD against two state-of-the-art baselines. The results show that LEREDD provides highly accurate classification of dependent and non-dependent requirements, achieving an accuracy of 0.93, and an F1 score of 0.84, with the latter averaging 0.96 for non-dependent cases. LEREDD outperforms zero-shot LLMs and baselines, particularly in detecting fine-grained dependency types, where it yields average relative gains of 94.87% and 105.41% in F1 scores for the Requires dependency over the baselines. We also provide an annotated dataset of requirement dependencies encompassing 813 requirement pairs across three distinct systems to support reproducibility and future research. 

---
# Silent Egress: When Implicit Prompt Injection Makes LLM Agents Leak Without a Trace 

**Authors**: Qianlong Lan, Anuj Kaul, Shaun Jones, Stephanie Westrum  

**Link**: [PDF](https://arxiv.org/pdf/2602.22450)  

**Abstract**: Agentic large language model systems increasingly automate tasks by retrieving URLs and calling external tools. We show that this workflow gives rise to implicit prompt injection: adversarial instructions embedded in automatically generated URL previews, including titles, metadata, and snippets, can introduce a system-level risk that we refer to as silent egress. Using a fully local and reproducible testbed, we demonstrate that a malicious web page can induce an agent to issue outbound requests that exfiltrate sensitive runtime context, even when the final response shown to the user appears harmless. In 480 experimental runs with a qwen2.5:7b-based agent, the attack succeeds with high probability (P (egress) =0.89), and 95% of successful attacks are not detected by output-based safety checks. We also introduce sharded exfiltration, where sensitive information is split across multiple requests to avoid detection. This strategy reduces single-request leakage metrics by 73% (Leak@1) and bypasses simple data loss prevention mechanisms. Our ablation results indicate that defenses applied at the prompt layer offer limited protection, while controls at the system and network layers, such as domain allowlisting and redirect-chain analysis, are considerably more effective. These findings suggest that network egress should be treated as a first-class security outcome in agentic LLM systems. We outline architectural directions, including provenance tracking and capability isolation, that go beyond prompt-level hardening. 

---
# A Fusion of context-aware based BanglaBERT and Two-Layer Stacked LSTM Framework for Multi-Label Cyberbullying Detection 

**Authors**: Mirza Raquib, Asif Pervez Polok, Kedar Nath Biswas, Rahat Uddin Azad, Saydul Akbar Murad, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2602.22449)  

**Abstract**: Cyberbullying has become a serious and growing concern in todays virtual world. When left unnoticed, it can have adverse consequences for social and mental health. Researchers have explored various types of cyberbullying, but most approaches use single-label classification, assuming that each comment contains only one type of abuse. In reality, a single comment may include overlapping forms such as threats, hate speech, and harassment. Therefore, multilabel detection is both realistic and essential. However, multilabel cyberbullying detection has received limited attention, especially in low-resource languages like Bangla, where robust pre-trained models are scarce. Developing a generalized model with moderate accuracy remains challenging. Transformers offer strong contextual understanding but may miss sequential dependencies, while LSTM models capture temporal flow but lack semantic depth. To address these limitations, we propose a fusion architecture that combines BanglaBERT-Large with a two-layer stacked LSTM. We analyze their behavior to jointly model context and sequence. The model is fine-tuned and evaluated on a publicly available multilabel Bangla cyberbullying dataset covering cyberbully, sexual harassment, threat, and spam. We apply different sampling strategies to address class imbalance. Evaluation uses multiple metrics, including accuracy, precision, recall, F1-score, Hamming loss, Cohens kappa, and AUC-ROC. We employ 5-fold cross-validation to assess the generalization of the architecture. 

---
# ECHO: Encoding Communities via High-order Operators 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2602.22446)  

**Abstract**: Community detection in attributed networks faces a fundamental divide: topological algorithms ignore semantic features, while Graph Neural Networks (GNNs) encounter devastating computational bottlenecks. Specifically, GNNs suffer from a Semantic Wall of feature over smoothing in dense or heterophilic networks, and a Systems Wall driven by the O(N^2) memory constraints of pairwise clustering. To dismantle these barriers, we introduce ECHO (Encoding Communities via High order Operators), a scalable, self supervised architecture that reframes community detection as an adaptive, multi scale diffusion process. ECHO features a Topology Aware Router that automatically analyzes structural heuristics sparsity, density, and assortativity to route graphs through the optimal inductive bias, preventing heterophilic poisoning while ensuring semantic densification. Coupled with a memory sharded full batch contrastive objective and a novel chunked O(N \cdot K) similarity extraction method, ECHO completely bypasses traditional O(N^2) memory bottlenecks without sacrificing the mathematical precision of global gradients. Extensive evaluations demonstrate that this topology feature synergy consistently overcomes the classical resolution limit. On synthetic LFR benchmarks scaled up to 1 million nodes, ECHO achieves scale invariant accuracy despite severe topological noise. Furthermore, on massive real world social networks with over 1.6 million nodes and 30 million edges, it completes clustering in mere minutes with throughputs exceeding 2,800 nodes per second matching the speed of highly optimized purely topological baselines. The implementation utilizes a unified framework that automatically engages memory sharded optimization to support adoption across varying hardware constraints. GitHub Repository: this https URL 

---
# From Bias to Balance: Fairness-Aware Paper Recommendation for Equitable Peer Review 

**Authors**: Uttamasha Anjally Oyshi, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2602.22438)  

**Abstract**: Despite frequent double-blind review, systemic biases related to author demographics still disadvantage underrepresented groups. We start from a simple hypothesis: if a post-review recommender is trained with an explicit fairness regularizer, it should increase inclusion without degrading quality. To test this, we introduce Fair-PaperRec, a Multi-Layer Perceptron (MLP) with a differentiable fairness loss over intersectional attributes (e.g., race, country) that re-ranks papers after double-blind review. We first probe the hypothesis on synthetic datasets spanning high, moderate, and near-fair biases. Across multiple randomized runs, these controlled studies map where increasing the fairness weight strengthens macro/micro diversity while keeping utility approximately stable, demonstrating robustness and adaptability under varying disparity levels. We then carry the hypothesis into the original setting, conference data from ACM Special Interest Group on Computer-Human Interaction (SIGCHI), Designing Interactive Systems (DIS), and Intelligent User Interfaces (IUI). In this real-world scenario, an appropriately tuned configuration of Fair-PaperRec achieves up to a 42.03% increase in underrepresented-group participation with at most a 3.16% change in overall utility relative to the historical selection. Taken together, the synthetic-to-original progression shows that fairness regularization can act as both an equity mechanism and a mild quality regularizer, especially in highly biased regimes. By first analyzing the behavior of the fairness parameters under controlled conditions and then validating them on real submissions, Fair-PaperRec offers a practical, equity-focused framework for post-review paper selection that preserves, and in some settings can even enhance, measured scholarly quality. 

---
# veScale-FSDP: Flexible and High-Performance FSDP at Scale 

**Authors**: Zezhou Wang, Youjie Li, Zhiqi Lin, Jiacheng Yang, Cong Xie, Guanyu Feng, Zheng Zhong, Ziyue Huang, Hongyu Zhu, Zhi Zhang, Yanghua Peng, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.22437)  

**Abstract**: Fully Sharded Data Parallel (FSDP), also known as ZeRO, is widely used for training large-scale models, featuring its flexibility and minimal intrusion on model code. However, current FSDP systems struggle with structure-aware training methods (e.g., block-wise quantized training) and with non-element-wise optimizers (e.g., Shampoo and Muon) used in cutting-edge models (e.g., Gemini, Kimi K2). FSDP's fixed element- or row-wise sharding formats conflict with the block-structured computations. In addition, today's implementations fall short in communication and memory efficiency, limiting scaling to tens of thousands of GPUs. We introduce veScale-FSDP, a redesigned FSDP system that couples a flexible sharding format, RaggedShard, with a structure-aware planning algorithm to deliver both flexibility and performance at scale. veScale-FSDP natively supports efficient data placement required by FSDP, empowering block-wise quantization and non-element-wise optimizers. As a result, veScale-FSDP achieves 5~66% higher throughput and 16~30% lower memory usage than existing FSDP systems, while scaling efficiently to tens of thousands of GPUs. 

---
# GetBatch: Distributed Multi-Object Retrieval for ML Data Loading 

**Authors**: Alex Aizman, Abhishek Gaikwad, Piotr Żelasko  

**Link**: [PDF](https://arxiv.org/pdf/2602.22434)  

**Abstract**: Machine learning training pipelines consume data in batches. A single training step may require thousands of samples drawn from shards distributed across a storage cluster. Issuing thousands of individual GET requests incurs per-request overhead that often dominates data transfer time. To solve this problem, we introduce GetBatch - a new object store API that elevates batch retrieval to a first-class storage operation, replacing independent GET operations with a single deterministic, fault-tolerant streaming execution. GetBatch achieves up to 15x throughput improvement for small objects and, in a production training workload, reduces P95 batch retrieval latency by 2x and P99 per-object tail latency by 3.7x compared to individual GET requests. 

---
# Calibrated Test-Time Guidance for Bayesian Inference 

**Authors**: Daniel Geyfman, Felix Draxler, Jan Groeneveld, Hyunsoo Lee, Theofanis Karaletsos, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2602.22428)  

**Abstract**: Test-time guidance is a widely used mechanism for steering pretrained diffusion models toward outcomes specified by a reward function. Existing approaches, however, focus on maximizing reward rather than sampling from the true Bayesian posterior, leading to miscalibrated inference. In this work, we show that common test-time guidance methods do not recover the correct posterior distribution and identify the structural approximations responsible for this failure. We then propose consistent alternative estimators that enable calibrated sampling from the Bayesian posterior. We significantly outperform previous methods on a set of Bayesian inference tasks, and match state-of-the-art in black hole image reconstruction. 

---
# HubScan: Detecting Hubness Poisoning in Retrieval-Augmented Generation Systems 

**Authors**: Idan Habler, Vineeth Sai Narajala, Stav Koren, Amy Chang, Tiffany Saade  

**Link**: [PDF](https://arxiv.org/pdf/2602.22427)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are essential to contemporary AI applications, allowing large language models to obtain external knowledge via vector similarity search. Nevertheless, these systems encounter a significant security flaw: hubness - items that frequently appear in the top-k retrieval results for a disproportionately high number of varied queries. These hubs can be exploited to introduce harmful content, alter search rankings, bypass content filtering, and decrease system performance.
We introduce hubscan, an open-source security scanner that evaluates vector indices and embeddings to identify hubs in RAG systems. Hubscan presents a multi-detector architecture that integrates: (1) robust statistical hubness detection utilizing median/MAD-based z-scores, (2) cluster spread analysis to assess cross-cluster retrieval patterns, (3) stability testing under query perturbations, and (4) domain-aware and modality-aware detection for category-specific and cross-modal attacks. Our solution accommodates several vector databases (FAISS, Pinecone, Qdrant, Weaviate) and offers versatile retrieval techniques, including vector similarity, hybrid search, and lexical matching with reranking capabilities.
We evaluate hubscan on Food-101, MS-COCO, and FiQA adversarial hubness benchmarks constructed using state-of-the-art gradient-optimized and centroid-based hub generation methods. hubscan achieves 90% recall at a 0.2% alert budget and 100% recall at 0.4%, with adversarial hubs ranking above the 99.8th percentile. Domain-scoped scanning recovers 100% of targeted attacks that evade global detection. Production validation on 1M real web documents from MS MARCO demonstrates significant score separation between clean documents and adversarial content. Our work provides a practical, extensible framework for detecting hubness threats in production RAG systems. 

---
# Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression 

**Authors**: Luciano Gerber, Huw Lloyd  

**Link**: [PDF](https://arxiv.org/pdf/2602.22422)  

**Abstract**: Smooth-basis models such as Chebyshev polynomial regressors and radial basis function (RBF) networks are well established in numerical analysis. Their continuously differentiable prediction surfaces suit surrogate optimisation, sensitivity analysis, and other settings where the response varies gradually with inputs. Despite these properties, smooth models seldom appear in tabular regression, where tree ensembles dominate. We ask whether they can compete, benchmarking models across 55 regression datasets organised by application domain.
We develop an anisotropic RBF network with data-driven centre placement and gradient-based width optimisation, a ridge-regularised Chebyshev polynomial regressor, and a smooth-tree hybrid (Chebyshev model tree); all three are released as scikit-learn-compatible packages. We benchmark these against tree ensembles, a pre-trained transformer, and standard baselines, evaluating accuracy alongside generalisation behaviour.
The transformer ranks first on accuracy across a majority of datasets, but its GPU dependence, inference latency, and dataset-size limits constrain deployment in the CPU-based settings common across applied science and industry. Among CPU-viable models, smooth models and tree ensembles are statistically tied on accuracy, but the former tend to exhibit tighter generalisation gaps. We recommend routinely including smooth-basis models in the candidate pool, particularly when downstream use benefits from tighter generalisation and gradually varying predictions. 

---
# Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents 

**Authors**: Cosmo Santoni  

**Link**: [PDF](https://arxiv.org/pdf/2602.22402)  

**Abstract**: As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base64 images, and metadata. A single-user case-study evaluation across 76 real-world coding sessions demonstrates that trimming remains economically viable under prompt caching, with the strongest gains in mixed tool-use sessions, which average 39% reduction and reach break-even within 10 turns. A reference implementation is available at this https URL. 

---
# Enhancing Renal Tumor Malignancy Prediction: Deep Learning with Automatic 3D CT Organ Focused Attention 

**Authors**: Zhengkang Fan, Chengkun Sun, Russell Terry, Jie Xu, Longin Jan Latecki  

**Link**: [PDF](https://arxiv.org/pdf/2602.22381)  

**Abstract**: Accurate prediction of malignancy in renal tumors is crucial for informing clinical decisions and optimizing treatment strategies. However, existing imaging modalities lack the necessary accuracy to reliably predict malignancy before surgical intervention. While deep learning has shown promise in malignancy prediction using 3D CT images, traditional approaches often rely on manual segmentation to isolate the tumor region and reduce noise, which enhances predictive performance. Manual segmentation, however, is labor-intensive, costly, and dependent on expert knowledge. In this study, a deep learning framework was developed utilizing an Organ Focused Attention (OFA) loss function to modify the attention of image patches so that organ patches attend only to other organ patches. Hence, no segmentation of 3D renal CT images is required at deployment time for malignancy prediction. The proposed framework achieved an AUC of 0.685 and an F1-score of 0.872 on a private dataset from the UF Integrated Data Repository (IDR), and an AUC of 0.760 and an F1-score of 0.852 on the publicly available KiTS21 dataset. These results surpass the performance of conventional models that rely on segmentation-based cropping for noise reduction, demonstrating the frameworks ability to enhance predictive accuracy without explicit segmentation input. The findings suggest that this approach offers a more efficient and reliable method for malignancy prediction, thereby enhancing clinical decision-making in renal cancer diagnosis. 

---
# AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction 

**Authors**: Hanyang Liu, Rongjun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2602.22376)  

**Abstract**: Recent advances in 4D scene reconstruction have significantly improved dynamic modeling across various domains. However, existing approaches remain limited under aerial conditions with single-view capture, wide spatial range, and dynamic objects of limited spatial footprint and large motion disparity. These challenges cause severe depth ambiguity and unstable motion estimation, making monocular aerial reconstruction inherently ill-posed. To this end, we present AeroDGS, a physics-guided 4D Gaussian splatting framework for monocular UAV videos. AeroDGS introduces a Monocular Geometry Lifting module that reconstructs reliable static and dynamic geometry from a single aerial sequence, providing a robust basis for dynamic estimation. To further resolve monocular ambiguity, we propose a Physics-Guided Optimization module that incorporates differentiable ground-support, upright-stability, and trajectory-smoothness priors, transforming ambiguous image cues into physically consistent motion. The framework jointly refines static backgrounds and dynamic entities with stable geometry and coherent temporal evolution. We additionally build a real-world UAV dataset that spans various altitudes and motion conditions to evaluate dynamic aerial reconstruction. Experiments on synthetic and real UAV scenes demonstrate that AeroDGS outperforms state-of-the-art methods, achieving superior reconstruction fidelity in dynamic aerial environments. 

---
# EyeLayer: Integrating Human Attention Patterns into LLM-Based Code Summarization 

**Authors**: Jiahao Zhang, Yifan Zhang, Kevin Leach, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22368)  

**Abstract**: Code summarization is the task of generating natural language descriptions of source code, which is critical for software comprehension and maintenance. While large language models (LLMs) have achieved remarkable progress on this task, an open question remains: can human expertise in code understanding further guide and enhance these models? We propose EyeLayer, a lightweight attention-augmentation module that incorporates human eye-gaze patterns, as a proxy of human expertise, into LLM-based code summarization. EyeLayer models human attention during code reading via a Multimodal Gaussian Mixture, redistributing token embeddings based on learned parameters (\mu_i, \sigma_i^2) that capture where and how intensively developers focus. This design enables learning generalizable attention priors from eye-tracking data and incorporating them into LLMs seamlessly, without disturbing existing representations. We evaluate EyeLayer across diverse model families (i.e., LLaMA-3.2, Qwen3, and CodeBERT) covering different scales and architectures. EyeLayer consistently outperforms strong fine-tuning baselines across standard metrics, achieving gains of up to 13.17% on BLEU-4. These results demonstrate that human gaze patterns encode complementary attention signals that enhance the semantic focus of LLMs and transfer effectively across diverse models for code summarization. 

---
# Learning geometry-dependent lead-field operators for forward ECG modeling 

**Authors**: Arsenii Dokuchaev, Francesca Bonizzoni, Stefano Pagani, Francesco Regazzoni, Simone Pezzuto  

**Link**: [PDF](https://arxiv.org/pdf/2602.22367)  

**Abstract**: Modern forward electrocardiogram (ECG) computational models rely on an accurate representation of the torso domain. The lead-field method enables fast ECG simulations while preserving full geometric fidelity. Achieving high anatomical accuracy in torso representation is, however, challenging in clinical practice, as imaging protocols are typically focused on the heart and often do not include the entire torso. In addition, the computational cost of the lead-field method scales linearly with the number of electrodes, limiting its applicability in high-density recording settings. To date, no existing approach simultaneously achieves high anatomical fidelity, low data requirements and computational efficiency. In this work, we propose a shape-informed surrogate model of the lead-field operator that serves as a drop-in replacement for the full-order model in forward ECG simulations. The proposed framework consists of two components: a geometry-encoding module that maps anatomical shapes into a low-dimensional latent space, and a geometry-conditioned neural surrogate that predicts lead-field gradients from spatial coordinates, electrode positions and latent codes. The proposed method achieves high accuracy in approximating lead fields both within the torso (mean angular error 5°) and inside the heart, resulting in highly accurate ECG simulations (relative mean squared error <2.5%. The surrogate consistently outperforms the widely used pseudo lead-field approximation while preserving negligible inference cost. Owing to its compact latent representation, the method does not require a fully detailed torso segmentation and can therefore be deployed in data-limited settings while preserving high-fidelity ECG simulations. 

---
# Scaling In, Not Up? Testing Thick Citation Context Analysis with GPT-5 and Fragile Prompts 

**Authors**: Arno Simons  

**Link**: [PDF](https://arxiv.org/pdf/2602.22359)  

**Abstract**: This paper tests whether large language models (LLMs) can support interpretative citation context analysis (CCA) by scaling in thick, text-grounded readings of a single hard case rather than scaling up typological labels. It foregrounds prompt-sensitivity analysis as a methodological issue by varying prompt scaffolding and framing in a balanced 2x3 design. Using footnote 6 in Chubin and Moitra (1975) and Gilbert's (1977) reconstruction as a probe, I implement a two-stage GPT-5 pipeline: a citation-text-only surface classification and expectation pass, followed by cross-document interpretative reconstruction using the citing and cited full texts. Across 90 reconstructions, the model produces 450 distinct hypotheses. Close reading and inductive coding identify 21 recurring interpretative moves, and linear probability models estimate how prompt choices shift their frequencies and lexical repertoire. GPT-5's surface pass is highly stable, consistently classifying the citation as "supplementary". In reconstruction, the model generates a structured space of plausible alternatives, but scaffolding and examples redistribute attention and vocabulary, sometimes toward strained readings. Relative to Gilbert, GPT-5 detects the same textual hinges yet more often resolves them as lineage and positioning than as admonishment. The study outlines opportunities and risks of using LLMs as guided co-analysts for inspectable, contestable interpretative CCA, and it shows that prompt scaffolding and framing systematically tilt which plausible readings and vocabularies the model foregrounds. 

---
# GRAU: Generic Reconfigurable Activation Unit Design for Neural Network Hardware Accelerators 

**Authors**: Yuhao Liu, Salim Ullah, Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.22352)  

**Abstract**: With the continuous growth of neural network scales, low-precision quantization is widely used in edge accelerators. Classic multi-threshold activation hardware requires 2^n thresholds for n-bit outputs, causing a rapid increase in hardware cost as precision increases. We propose a reconfigurable activation hardware, GRAU, based on piecewise linear fitting, where the segment slopes are approximated by powers of two. Our design requires only basic comparators and 1-bit right shifters, supporting mixed-precision quantization and nonlinear functions such as SiLU. Compared with multi-threshold activators, GRAU reduces LUT consumption by over 90%, achieving higher hardware efficiency, flexibility, and scalability. 

---
# Decoder-based Sense Knowledge Distillation 

**Authors**: Qitong Wang, Mohammed J. Zaki, Georgios Kollias, Vasileios Kalantzis  

**Link**: [PDF](https://arxiv.org/pdf/2602.22351)  

**Abstract**: Large language models (LLMs) learn contextual embeddings that capture rich semantic information, yet they often overlook structured lexical knowledge such as word senses and relationships. Prior work has shown that incorporating sense dictionaries can improve knowledge distillation for encoder models, but their application to decoder as generative models remains challenging. In this paper, we introduce Decoder-based Sense Knowledge Distillation (DSKD), a framework that integrates lexical resources into the training of decoder-style LLMs without requiring dictionary lookup at inference time. Extensive experiments on diverse benchmarks demonstrate that DSKD significantly enhances knowledge distillation performance for decoders, enabling generative models to inherit structured semantics while maintaining efficient training. 

---
# Enabling clinical use of foundation models in histopathology 

**Authors**: Audun L. Henriksen, Ole-Johan Skrede, Lisa van der Schee, Enric Domingo, Sepp De Raedt, Ilyá Kostolomov, Jennifer Hay, Karolina Cyll, Wanja Kildal, Joakim Kalsnes, Robert W. Williams, Manohar Pradhan, John Arne Nesheim, Hanne A. Askautrud, Maria X. Isaksen, Karmele Saez de Gordoa, Miriam Cuatrecasas, Joanne Edwards, TransSCOT group, Arild Nesbakken, Neil A. Shepherd, Ian Tomlinson, Daniel-Christoph Wagner, Rachel S. Kerr, Tarjei Sveinsgjerd Hveem, Knut Liestøl, Yoshiaki Nakamura, Marco Novelli, Masaaki Miyo, Sebastian Foersch, David N. Church, Miangela M. Lacle, David J. Kerr, Andreas Kleppe  

**Link**: [PDF](https://arxiv.org/pdf/2602.22347)  

**Abstract**: Foundation models in histopathology are expected to facilitate the development of high-performing and generalisable deep learning systems. However, current models capture not only biologically relevant features, but also pre-analytic and scanner-specific variation that bias the predictions of task-specific models trained from the foundation model features. Here we show that introducing novel robustness losses during training of downstream task-specific models reduces sensitivity to technical variability. A purpose-designed comprehensive experimentation setup with 27,042 WSIs from 6155 patients is used to train thousands of models from the features of eight popular foundation models for computational pathology. In addition to a substantial improvement in robustness, we observe that prediction accuracy improves by focusing on biologically relevant features. Our approach successfully mitigates robustness issues of foundation models for computational pathology without retraining the foundation models themselves, enabling development of robust computational pathology models applicable to real-world data in routine clinical practice. 

---
# Structure and Redundancy in Large Language Models: A Spectral Study via Random Matrix Theory 

**Authors**: Davide Ettori  

**Link**: [PDF](https://arxiv.org/pdf/2602.22345)  

**Abstract**: This thesis addresses two persistent and closely related challenges in modern deep learning, reliability and efficiency, through a unified framework grounded in Spectral Geometry and Random Matrix Theory (RMT). As deep networks and large language models continue to scale, their internal behavior becomes increasingly opaque, leading to hallucinations, fragile generalization under distribution shift, and growing computational and energy demands. By analyzing the eigenvalue dynamics of hidden activations across layers and inputs, this work shows that spectral statistics provide a compact, stable, and interpretable lens on model behavior, capable of separating structured, causal representations from noise-dominated variability. Within this framework, the first contribution, EigenTrack, introduces a real-time method for detecting hallucinations and out-of-distribution behavior in large language and vision-language models. EigenTrack transforms streaming activations into spectral descriptors such as entropy, variance, and deviations from the Marchenko-Pastur baseline, and models their temporal evolution using lightweight recurrent classifiers, enabling early detection of reliability failures before they appear in model outputs while offering interpretable insight into representation dynamics. The second contribution, RMT-KD, presents a principled approach to compressing deep networks via random matrix theoretic knowledge distillation. By interpreting outlier eigenvalues in activation spectra as carriers of task-relevant information, RMT-KD progressively projects networks onto lower-dimensional subspaces through iterative self-distillation, yielding significantly more compact and energy-efficient models while preserving accuracy and dense, hardware-friendly structure. 

---
# A 1/R Law for Kurtosis Contrast in Balanced Mixtures 

**Authors**: Yuda Bi, Wenjun Xiao, Linhao Bai, Vince D Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2602.22334)  

**Abstract**: Kurtosis-based Independent Component Analysis (ICA) weakens in wide, balanced mixtures. We prove a sharp redundancy law: for a standardized projection with effective width $R_{\mathrm{eff}}$ (participation ratio), the population excess kurtosis obeys $|\kappa(y)|=O(\kappa_{\max}/R_{\mathrm{eff}})$, yielding the order-tight $O(c_b\kappa_{\max}/R)$ under balance (typically $c_b=O(\log R)$). As an impossibility screen, under standard finite-moment conditions for sample kurtosis estimation, surpassing the $O(1/\sqrt{T})$ estimation scale requires $R\lesssim \kappa_{\max}\sqrt{T}$. We also show that \emph{purification} -- selecting $m\!\ll\!R$ sign-consistent sources -- restores $R$-independent contrast $\Omega(1/m)$, with a simple data-driven heuristic. Synthetic experiments validate the predicted decay, the $\sqrt{T}$ crossover, and contrast recovery. 

---
# Training Agents to Self-Report Misbehavior 

**Authors**: Bruce W. Lee, Chen Yueh-Han, Tomek Korbak  

**Link**: [PDF](https://arxiv.org/pdf/2602.22303)  

**Abstract**: Frontier AI agents may pursue hidden goals while concealing their pursuit from oversight. Alignment training aims to prevent such behavior by reinforcing the correct goals, but alignment may not always succeed and can lead to unwanted side effects. We propose self-incrimination training, which instead trains agents to produce a visible signal when they covertly misbehave. We train GPT-4.1 and Gemini-2.0 agents to call a report_scheming() tool when behaving deceptively and measure their ability to cause harm undetected in out-of-distribution environments. Self-incrimination significantly reduces the undetected successful attack rate, outperforming matched-capability monitors and alignment baselines while preserving instruction hierarchy and incurring minimal safety tax on general capabilities. Unlike blackbox monitoring, self-incrimination performance is consistent across tasks regardless of how suspicious the misbehavior appears externally. The trained behavior persists under adversarial prompt optimization and generalizes to settings where agents pursue misaligned goals themselves rather than being instructed to misbehave. Our results suggest self-incrimination offers a viable path for reducing frontier misalignment risk, one that neither assumes misbehavior can be prevented nor that it can be reliably classified from the outside. 

---
# Decoding the Hook: A Multimodal LLM Framework for Analyzing the Hooking Period of Video Ads 

**Authors**: Kunpeng Zhang, Poppy Zhang, Shawndra Hill, Amel Awadelkarim  

**Link**: [PDF](https://arxiv.org/pdf/2602.22299)  

**Abstract**: Video-based ads are a vital medium for brands to engage consumers, with social media platforms leveraging user data to optimize ad delivery and boost engagement. A crucial but under-explored aspect is the 'hooking period', the first three seconds that capture viewer attention and influence engagement metrics. Analyzing this brief window is challenging due to the multimodal nature of video content, which blends visual, auditory, and textual elements. Traditional methods often miss the nuanced interplay of these components, requiring advanced frameworks for thorough evaluation.
This study presents a framework using transformer-based multimodal large language models (MLLMs) to analyze the hooking period of video ads. It tests two frame sampling strategies, uniform random sampling and key frame selection, to ensure balanced and representative acoustic feature extraction, capturing the full range of design elements. The hooking video is processed by state-of-the-art MLLMs to generate descriptive analyses of the ad's initial impact, which are distilled into coherent topics using BERTopic for high-level abstraction. The framework also integrates features such as audio attributes and aggregated ad targeting information, enriching the feature set for further analysis.
Empirical validation on large-scale real-world data from social media platforms demonstrates the efficacy of our framework, revealing correlations between hooking period features and key performance metrics like conversion per investment. The results highlight the practical applicability and predictive power of the approach, offering valuable insights for optimizing video ad strategies. This study advances video ad analysis by providing a scalable methodology for understanding and enhancing the initial moments of video advertisements. 

---
# AviaSafe: A Physics-Informed Data-Driven Model for Aviation Safety-Critical Cloud Forecasts 

**Authors**: Zijian Zhu, Qiusheng Huang, Anboyu Guo, Xiaohui Zhong, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.22298)  

**Abstract**: Current AI weather forecasting models predict conventional atmospheric variables but cannot distinguish between cloud microphysical species critical for aviation safety. We introduce AviaSafe, a hierarchical, physics-informed neural forecaster that produces global, six-hourly predictions of these four hydrometeor species for lead times up to 7 days. Our approach addresses the unique challenges of cloud prediction: extreme sparsity, discontinuous distributions, and complex microphysical interactions between species. We integrate the Icing Condition (IC) index from aviation meteorology as a physics-based constraint that identifies regions where supercooled water fuels explosive ice crystal growth. The model employs a hierarchical architecture that first predicts cloud spatial distribution through masked attention, then quantifies species concentrations within identified regions. Training on ERA5 reanalysis data, our model achieves lower RMSE for cloud species compared to baseline and outperforms operational numerical models on certain key variables at 7-day lead times. The ability to forecast individual cloud species enables new applications in aviation route optimization where distinguishing between ice and liquid water determines engine icing risk. 

---
# Learning Rewards, Not Labels: Adversarial Inverse Reinforcement Learning for Machinery Fault Detection 

**Authors**: Dhiraj Neupane, Richard Dazeley, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2602.22297)  

**Abstract**: Reinforcement learning (RL) offers significant promise for machinery fault detection (MFD). However, most existing RL-based MFD approaches do not fully exploit RL's sequential decision-making strengths, often treating MFD as a simple guessing game (Contextual Bandits). To bridge this gap, we formulate MFD as an offline inverse reinforcement learning problem, where the agent learns the reward dynamics directly from healthy operational sequences, thereby bypassing the need for manual reward engineering and fault labels. Our framework employs Adversarial Inverse Reinforcement Learning to train a discriminator that distinguishes between normal (expert) and policy-generated transitions. The discriminator's learned reward serves as an anomaly score, indicating deviations from normal operating behaviour. When evaluated on three run-to-failure benchmark datasets (HUMS2023, IMS, and XJTU-SY), the model consistently assigns low anomaly scores to normal samples and high scores to faulty ones, enabling early and robust fault detection. By aligning RL's sequential reasoning with MFD's temporal structure, this work opens a path toward RL-based diagnostics in data-driven industrial settings. 

---
# UpSkill: Mutual Information Skill Learning for Structured Response Diversity in LLMs 

**Authors**: Devan Shah, Owen Yang, Daniel Yang, Chongyi Zheng, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2602.22296)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of large language models (LLMs) on mathematics and programming tasks, but standard approaches that optimize single-attempt accuracy can inadvertently suppress response diversity across repeated attempts, narrowing exploration and overlooking underrepresented strategies. We introduce UpSkill, a training time method that adapts Mutual Information Skill Learning (MISL) to LLMs for optimizing pass@k correctness. We propose a novel reward that we implement within Group Relative Policy Optimization (GRPO): a token-level mutual information (MI) reward that encourages trajectory specificity to z. Experiments on GSM8K with three open-weight models, Llama 3.1-8B, Qwen 2.5-7B, and R1-Distilled-Qwen2.5-Math-1.5B, show that UpSkill improves multi-attempt metrics on the stronger base models, yielding mean gains of ~3% in pass@k for both Qwen and Llama without degrading pass@1. Additionally, we find both empirical and theoretical evidence that improvements in pass@k are closely tied to the mutual information objective. 

---
# Manifold of Failure: Behavioral Attraction Basins in Language Models 

**Authors**: Sarthak Munshi, Manish Bhatt, Vineeth Sai Narajala, Idan Habler, AmmarnAl-Kahfah, Ken Huang, Blake Gatto  

**Link**: [PDF](https://arxiv.org/pdf/2602.22291)  

**Abstract**: While prior work has focused on projecting adversarial examples back onto the manifold of natural data to restore safety, we argue that a comprehensive understanding of AI safety requires characterizing the unsafe regions themselves. This paper introduces a framework for systematically mapping the Manifold of Failure in Large Language Models (LLMs). We reframe the search for vulnerabilities as a quality diversity problem, using MAP-Elites to illuminate the continuous topology of these failure regions, which we term behavioral attraction basins. Our quality metric, Alignment Deviation, guides the search towards areas where the model's behavior diverges most from its intended alignment. Across three LLMs: Llama-3-8B, GPT-OSS-20B, and GPT-5-Mini, we show that MAP-Elites achieves up to 63% behavioral coverage, discovers up to 370 distinct vulnerability niches, and reveals dramatically different model-specific topological signatures: Llama-3-8B exhibits a near-universal vulnerability plateau (mean Alignment Deviation 0.93), GPT-OSS-20B shows a fragmented landscape with spatially concentrated basins (mean 0.73), and GPT-5-Mini demonstrates strong robustness with a ceiling at 0.50. Our approach produces interpretable, global maps of each model's safety landscape that no existing attack method (GCG, PAIR, or TAP) can provide, shifting the paradigm from finding discrete failures to understanding their underlying structure. 

---
# Early Risk Stratification of Dosing Errors in Clinical Trials Using Machine Learning 

**Authors**: Félicien Hêche, Sohrab Ferdowsi, Anthony Yazdani, Sara Sansaloni-Pastor, Douglas Teodoro  

**Link**: [PDF](https://arxiv.org/pdf/2602.22285)  

**Abstract**: Objective: The objective of this study is to develop a machine learning (ML)-based framework for early risk stratification of clinical trials (CTs) according to their likelihood of exhibiting a high rate of dosing errors, using information available prior to trial initiation. Materials and Methods: We constructed a dataset from this http URL comprising 42,112 CTs. Structured, semi-structured trial data, and unstructured protocol-related free-text data were extracted. CTs were assigned binary labels indicating elevated dosing error rate, derived from adverse event reports, MedDRA terminology, and Wilson confidence intervals. We evaluated an XGBoost model trained on structured features, a ClinicalModernBERT model using textual data, and a simple late-fusion model combining both modalities. Post-hoc probability calibration was applied to enable interpretable, trial-level risk stratification. Results: The late-fusion model achieved the highest AUC-ROC (0.862). Beyond discrimination, calibrated outputs enabled robust stratification of CTs into predefined risk categories. The proportion of trials labeled as having an excessively high dosing error rate increased monotonically across higher predicted risk groups and aligned with the corresponding predicted probability ranges. Discussion: These findings indicate that dosing error risk can be anticipated at the trial level using pre-initiation information. Probability calibration was essential for translating model outputs into reliable and interpretable risk categories, while simple multimodal integration yielded performance gains without requiring complex architectures. Conclusion: This study introduces a reproducible and scalable ML framework for early, trial-level risk stratification of CTs at risk of high dosing error rates, supporting proactive, risk-based quality management in clinical research. 

---
# Integrating Machine Learning Ensembles and Large Language Models for Heart Disease Prediction Using Voting Fusion 

**Authors**: Md. Tahsin Amin, Tanim Ahmmod, Zannatul Ferdus, Talukder Naemul Hasan Naem, Ehsanul Ferdous, Arpita Bhattacharjee, Ishmam Ahmed Solaiman, Nahiyan Bin Noor  

**Link**: [PDF](https://arxiv.org/pdf/2602.22280)  

**Abstract**: Cardiovascular disease is the primary cause of death globally, necessitating early identification, precise risk classification, and dependable decision-support technologies. The advent of large language models (LLMs) provides new zero-shot and few-shot reasoning capabilities, even though machine learning (ML) algorithms, especially ensemble approaches like Random Forest, XGBoost, LightGBM, and CatBoost, are excellent at modeling complex, non-linear patient data and routinely beat logistic regression. This research predicts cardiovascular disease using a merged dataset of 1,190 patient records, comparing traditional machine learning models (95.78% accuracy, ROC-AUC 0.96) with open-source large language models via OpenRouter APIs. Finally, a hybrid fusion of the ML ensemble and LLM reasoning under Gemini 2.5 Flash achieved the best results (96.62% accuracy, 0.97 AUC), showing that LLMs (78.9 % accuracy) work best when combined with ML models rather than used alone. Results show that ML ensembles achieved the highest performance (95.78% accuracy, ROC-AUC 0.96), while LLMs performed moderately in zero-shot (78.9%) and slightly better in few-shot (72.6%) settings. The proposed hybrid method enhanced the strength in uncertain situations, illustrating that ensemble ML is considered the best structured tabular prediction case, but it can be integrated with hybrid ML-LLM systems to provide a minor increase and open the way to more reliable clinical decision-support tools. 

---
# Learning to reconstruct from saturated data: audio declipping and high-dynamic range imaging 

**Authors**: Victor Sechaud, Laurent Jacques, Patrice Abry, Julián Tachella  

**Link**: [PDF](https://arxiv.org/pdf/2602.22279)  

**Abstract**: Learning based methods are now ubiquitous for solving inverse problems, but their deployment in real-world applications is often hindered by the lack of ground truth references for training. Recent self-supervised learning strategies offer a promising alternative, avoiding the need for ground truth. However, most existing methods are limited to linear inverse problems. This work extends self-supervised learning to the non-linear problem of recovering audio and images from clipped measurements, by assuming that the signal distribution is approximately invariant to changes in amplitude. We provide sufficient conditions for learning to reconstruct from saturated signals alone and a self-supervised loss that can be used to train reconstruction networks. Experiments on both audio and image data show that the proposed approach is almost as effective as fully supervised approaches, despite relying solely on clipped measurements for training. 

---
# Positional-aware Spatio-Temporal Network for Large-Scale Traffic Prediction 

**Authors**: Runfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.22274)  

**Abstract**: Traffic flow forecasting has emerged as an indispensable mission for daily life, which is required to utilize the spatiotemporal relationship between each location within a time period under a graph structure to predict future flow. However, the large travel demand for broader geographical areas and longer time spans requires models to distinguish each node clearly and possess a holistic view of the history, which has been paid less attention to in prior works. Furthermore, increasing sizes of data hinder the deployment of most models in real application environments. To this end, in this paper, we propose a lightweight Positional-aware Spatio-Temporal Network (PASTN) to effectively capture both temporal and spatial complexities in an end-to-end manner. PASTN introduces positional-aware embeddings to separate each node's representation, while also utilizing a temporal attention module to improve the long-range perception of current models. Extensive experiments verify the effectiveness and efficiency of PASTN across datasets of various scales (county, megalopolis and state). Further analysis demonstrates the efficacy of newly introduced modules either. 

---
# CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints 

**Authors**: Fuyao Huang, Xiaozhu Yu, Kui Xu, Qiangfeng Cliff Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22263)  

**Abstract**: High-resolution structure determination by cryo-electron microscopy (cryo-EM) requires the accurate fitting of an atomic model into an experimental density map. Traditional refinement pipelines such as Phenix.real_space_refine and Rosetta are computationally expensive, demand extensive manual tuning, and present a significant bottleneck for researchers. We present this http URL, an end-to-end deep learning framework that automates and accelerates molecular structure refinement. Our approach utilizes a one-step diffusion model that integrates a density-aware loss function with robust stereochemical restraints, enabling rapid optimization of a structure against experimental data. this http URL provides a unified and versatile solution capable of refining protein complexes as well as DNA/RNA-protein complexes. In benchmarks against Phenix.real_space_refine, this http URL consistently achieves substantial improvements in both model-map correlation and overall geometric quality metrics. By offering a scalable, automated, and powerful alternative, this http URL aims to serve as an essential tool for next-generation cryo-EM structure refinement. Web server: this https URL Source code: this https URL. 

---
# Poisoned Acoustics 

**Authors**: Harrison Dahme  

**Link**: [PDF](https://arxiv.org/pdf/2602.22258)  

**Abstract**: Training-data poisoning attacks can induce targeted, undetectable failure in deep neural networks by corrupting a vanishingly small fraction of training labels. We demonstrate this on acoustic vehicle classification using the MELAUDIS urban intersection dataset (approx. 9,600 audio clips, 6 classes): a compact 2-D convolutional neural network (CNN) trained on log-mel spectrograms achieves 95.7% Attack Success Rate (ASR) -- the fraction of target-class test samples misclassified under the attack -- on a Truck-to-Car label-flipping attack at just p=0.5% corruption (48 records), with zero detectable change in aggregate accuracy (87.6% baseline; 95% CI: 88-100%, n=3 seeds). We prove this stealth is structural: the maximum accuracy drop from a complete targeted attack is bounded above by the minority class fraction (beta). For real-world class imbalances (Truck approx. 3%), this bound falls below training-run noise, making aggregate accuracy monitoring provably insufficient regardless of architecture or attack method. A companion backdoor trigger attack reveals a novel trigger-dominance collapse: when the target class is a dataset minority, the spectrogram patch trigger becomes functionally redundant--clean ASR equals triggered ASR, and the attack degenerates to pure label flipping. We formalize the ML training pipeline as an attack surface and propose a trust-minimized defense combining content-addressed artifact hashing, Merkle-tree dataset commitment, and post-quantum digital signatures (ML-DSA-65/CRYSTALS-Dilithium3, NIST FIPS 204) for cryptographically verifiable data provenance. 

---
# Deep Sequence Modeling with Quantum Dynamics: Language as a Wave Function 

**Authors**: Ahmed Nebli, Hadi Saadatdoorabi, Kevin Yam  

**Link**: [PDF](https://arxiv.org/pdf/2602.22255)  

**Abstract**: We introduce a sequence modeling framework in which the latent state is a complex-valued wave function evolving on a finite-dimensional Hilbert space under a learned, time-dependent Hamiltonian. Unlike standard recurrent architectures that rely on gating mechanisms to suppress competing hypotheses, our framework utilizes quantum interference: the Hamiltonian steers the phases of complex amplitudes so that conflicting interpretations cancel while compatible ones reinforce. The dynamics are strictly unitary, ensuring that the state norm is preserved exactly at every time step via a Cayley (Crank--Nicolson) discretization. Token probabilities are extracted using the Born rule, a quadratic measurement operator that couples magnitudes and relative phases. Our primary theoretical contribution is a separation theorem characterizing the representational advantage of this readout: we define a family of disambiguation tasks that a complex unitary model of dimension $N$ solves exactly, but which requires a state dimension of $\Omega(N^2)$ for any real-valued orthogonal model equipped with a standard affine-softmax readout. This quadratic gap arises because the Born rule implicitly lifts the $N$-dimensional state into the space of rank-one Hermitian matrices, accessing pairwise phase correlations that are inaccessible to linear projections. Finally, we derive a continuity equation for the latent probability mass, yielding conserved pairwise currents that serve as a built-in diagnostic for tracing information flow between dimensions. 

---
# Causal Direction from Convergence Time: Faster Training in the True Causal Direction 

**Authors**: Abdulrahman Tamim  

**Link**: [PDF](https://arxiv.org/pdf/2602.22254)  

**Abstract**: We introduce Causal Computational Asymmetry (CCA), a principle for causal direction identification based on optimization dynamics in which one neural network is trained to predict $Y$ from $X$ and another to predict $X$ from $Y$, and the direction that converges faster is inferred to be causal. Under the additive noise model $Y = f(X) + \varepsilon$ with $\varepsilon \perp X$ and $f$ nonlinear and injective, we establish a formal asymmetry: in the reverse direction, residuals remain statistically dependent on the input regardless of approximation quality, inducing a strictly higher irreducible loss floor and non-separable gradient noise in the optimization dynamics, so that the reverse model requires strictly more gradient steps in expectation to reach any fixed loss threshold; consequently, the forward (causal) direction converges in fewer expected optimization steps. CCA operates in optimization-time space, distinguishing it from methods such as RESIT, IGCI, and SkewScore that rely on statistical independence or distributional asymmetries, and proper z-scoring of both variables is required for valid comparison of convergence rates. On synthetic benchmarks, CCA achieves 26/30 correct causal identifications across six neural architectures, including 30/30 on sine and exponential data-generating processes. We further embed CCA into a broader framework termed Causal Compression Learning (CCL), which integrates graph structure learning, causal information compression, and policy optimization, with all theoretical guarantees formally proved and empirically validated on synthetic datasets. 

---
# Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials 

**Authors**: Alex Morehead, Miruna Cretu, Antonia Panescu, Rishabh Anand, Maurice Weiler, Tynan Perez, Samuel Blau, Steven Farrell, Wahid Bhimji, Anubhav Jain, Hrushikesh Sahasrabuddhe, Pietro Lio, Tommi Jaakkola, Rafael Gomez-Bombarelli, Rex Ying, N. Benjamin Erichson, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2602.22251)  

**Abstract**: General-purpose 3D chemical modeling encompasses molecules and materials, requiring both generative and predictive capabilities. However, most existing AI approaches are optimized for a single domain (molecules or materials) and a single task (generation or prediction), which limits representation sharing and transfer. We introduce Zatom-1, the first foundation model that unifies generative and predictive learning of 3D molecules and materials. Zatom-1 is a Transformer trained with a multimodal flow matching objective that jointly models discrete atom types and continuous 3D geometries. This approach supports scalable pretraining with predictable gains as model capacity increases, while enabling fast and stable sampling. We use joint generative pretraining as a universal initialization for downstream multi-task prediction of properties, energies, and forces. Empirically, Zatom-1 matches or outperforms specialized baselines on both generative and predictive benchmarks, while reducing the generative inference time by more than an order of magnitude. Our experiments demonstrate positive predictive transfer between chemical domains from joint generative pretraining: modeling materials during pretraining improves molecular property prediction accuracy. 

---
# Multi-Dimensional Spectral Geometry of Biological Knowledge in Single-Cell Transformer Representations 

**Authors**: Ihor Kendiukhov  

**Link**: [PDF](https://arxiv.org/pdf/2602.22247)  

**Abstract**: Single-cell foundation models such as scGPT learn high-dimensional gene representations, but what biological knowledge these representations encode remains unclear. We systematically decode the geometric structure of scGPT internal representations through 63 iterations of automated hypothesis screening (183 hypotheses tested), revealing that the model organizes genes into a structured biological coordinate system rather than an opaque feature space.
The dominant spectral axis separates genes by subcellular localization, with secreted proteins at one pole and cytosolic proteins at the other. Intermediate transformer layers transiently encode mitochondrial and ER compartments in a sequence that mirrors the cellular secretory pathway. Orthogonal axes encode protein-protein interaction networks with graded fidelity to experimentally measured interaction strength (Spearman rho = 1.000 across n = 5 STRING confidence quintiles, p = 0.017).
In a compact six-dimensional spectral subspace, the model distinguishes transcription factors from their target genes (AUROC = 0.744, all 12 layers significant). Early layers preserve which specific genes regulate which targets, while deeper layers compress this into a coarser regulator versus regulated distinction. Repression edges are geometrically more prominent than activation edges, and B-cell master regulators BATF and BACH2 show convergence toward the B-cell identity anchor PAX5 across transformer depth. Cell-type marker genes cluster with high fidelity (AUROC = 0.851). Residual-stream geometry encodes biological structure complementary to attention patterns. These results indicate that biological transformers learn an interpretable internal model of cellular organization, with implications for regulatory network inference, drug target prioritization, and model auditing. 

---
# Analysis of LLMs Against Prompt Injection and Jailbreak Attacks 

**Authors**: Piyush Jaiswal, Aaditya Pratap, Shreyansh Saraswati, Harsh Kasyap, Somanath Tripathy  

**Link**: [PDF](https://arxiv.org/pdf/2602.22242)  

**Abstract**: Large Language Models (LLMs) are widely deployed in real-world systems. Given their broader applicability, prompt engineering has become an efficient tool for resource-scarce organizations to adopt LLMs for their own purposes. At the same time, LLMs are vulnerable to prompt-based attacks. Thus, analyzing this risk has become a critical security requirement. This work evaluates prompt-injection and jailbreak vulnerability using a large, manually curated dataset across multiple open-source LLMs, including Phi, Mistral, DeepSeek-R1, Llama 3.2, Qwen, and Gemma variants. We observe significant behavioural variation across models, including refusal responses and complete silent non-responsiveness triggered by internal safety mechanisms. Furthermore, we evaluated several lightweight, inference-time defence mechanisms that operate as filters without any retraining or GPU-intensive fine-tuning. Although these defences mitigate straightforward attacks, they are consistently bypassed by long, reasoning-heavy prompts. 

---
# From Prompts to Performance: Evaluating LLMs for Task-based Parallel Code Generation 

**Authors**: Linus Bantel, Moritz Strack, Alexander Strack, Dirk Pflüger  

**Link**: [PDF](https://arxiv.org/pdf/2602.22240)  

**Abstract**: Large Language Models (LLM) show strong abilities in code generation, but their skill in creating efficient parallel programs is less studied. This paper explores how LLMs generate task-based parallel code from three kinds of input prompts: natural language problem descriptions, sequential reference implementations, and parallel pseudo code. We focus on three programming frameworks: OpenMP Tasking, C++ standard parallelism, and the asynchronous many-task runtime HPX. Each framework offers different levels of abstraction and control for task execution. We evaluate LLM-generated solutions for correctness and scalability. Our results reveal both strengths and weaknesses of LLMs with regard to problem complexity and framework. Finally, we discuss what these findings mean for future LLM-assisted development in high-performance and scientific computing. 

---
# TT-SEAL: TTD-Aware Selective Encryption for Adversarially-Robust and Low-Latency Edge AI 

**Authors**: Kyeongpil Min, Sangmin Jeon, Jae-Jin Lee, Woojoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.22238)  

**Abstract**: Cloud-edge AI must jointly satisfy model compression and security under tight device budgets. While Tensor-Train Decomposition (TTD) shrinks on-device models, prior selective-encryption studies largely assume dense weights, leaving its practicality under TTD compression unclear. We present TT-SEAL, a selective-encryption framework for TT-decomposed networks. TT-SEAL ranks TT cores with a sensitivity-based importance metric, calibrates a one-time robustness threshold, and uses a value-DP optimizer to encrypt the minimum set of critical cores with AES. Under TTD-aware, transfer-based threat models (and on an FPGA-prototyped edge processor) TT-SEAL matches the robustness of full (black-box) encryption while encrypting as little as 4.89-15.92% of parameters across ResNet-18, MobileNetV2, and VGG-16, and drives the share of AES decryption in end-to-end latency to low single digits (e.g., 58% -> 2.76% on ResNet-18), enabling secure, low-latency edge AI. 

---
# Optimized Disaster Recovery for Distributed Storage Systems: Lightweight Metadata Architectures to Overcome Cryptographic Hashing Bottleneck 

**Authors**: Prasanna Kumar, Nishank Soni, Gaurang Munje  

**Link**: [PDF](https://arxiv.org/pdf/2602.22237)  

**Abstract**: Distributed storage architectures are foundational to modern cloud-native infrastructure, yet a critical operational bottleneck persists within disaster recovery (DR) workflows: the dependence on content-based cryptographic hashing for data identification and synchronization. While hash-based deduplication is effective for storage efficiency in steady-state operation, it becomes a systemic liability during failover and failback events when hash indexes are stale, incomplete, or must be rebuilt following a crash. This paper precisely characterizes the operational conditions under which full or partial re-hashing becomes unavoidable. The paper also analyzes the downstream impact of cryptographic re-hashing on Recovery Time Objective (RTO) compliance, and proposes a generalized architectural shift toward deterministic, metadata-driven identification. The proposed framework assigns globally unique composite identifiers to data blocks at ingestion time-independent of content analysis enabling instantaneous delta computation during DR without any cryptographic overhead. 

---
# Unsupervised Denoising of Diffusion-Weighted Images with Bias and Variance Corrected Noise Modeling 

**Authors**: Jine Xie, Zhicheng Zhang, Yunwei Chen, Yanqiu Feng, Xinyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22235)  

**Abstract**: Diffusion magnetic resonance imaging (dMRI) plays a vital role in both clinical diagnostics and neuroscience research. However, its inherently low signal-to-noise ratio (SNR), especially under high diffusion weighting, significantly degrades image quality and impairs downstream analysis. Recent self-supervised and unsupervised denoising methods offer a practical solution by enhancing image quality without requiring clean references. However, most of these methods do not explicitly account for the non-Gaussian noise characteristics commonly present in dMRI magnitude data during the supervised learning process, potentially leading to systematic bias and heteroscedastic variance, particularly under low-SNR conditions. To overcome this limitation, we introduce noise-corrected training objectives that explicitly model Rician statistics. Specifically, we propose two alternative loss functions: one derived from the first-order moment to remove mean bias, and another from the second-order moment to correct squared-signal bias. Both losses include adaptive weighting to account for variance heterogeneity and can be used without changing the network architecture. These objectives are instantiated in an image-specific, unsupervised Deep Image Prior (DIP) framework. Comprehensive experiments on simulated and in-vivo dMRI show that the proposed losses effectively reduce Rician bias and suppress noise fluctuations, yielding higher image quality and more reliable diffusion metrics than state-of-the-art denoising baselines. These results underscore the importance of bias- and variance-aware noise modeling for robust dMRI analysis under low-SNR conditions. 

---
# FM-RME: Foundation Model Empowered Radio Map Estimation 

**Authors**: Dong Yang, Yue Wang, Songyang Zhang, Yingshu Li, Zhipeng Cai, Zhi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.22231)  

**Abstract**: Traditional radio map estimation (RME) techniques fail to capture multi-dimensional and dynamic characteristics of complex spectrum environments. Recent data-driven methods achieve accurate RME in spatial domain, but ignore physical prior knowledge of radio propagation, limiting data efficiency especially in multi-dimensional scenarios. To overcome such limitations, we propose a new foundation model, characterized by self-supervised pre-training on diverse data for zero-shot generalization, enabling multi-dimensional radio map estimation (FM-RME). Specifically, FM-RME builds an effective synergy of two core components: a geometry-aware feature extraction module that encodes physical propagation symmetries, i.e., translation and rotation invariance, as inductive bias, and an attention-based neural network that learns long-range correlations across the spatial-temporal-spectral domains. A masked self-supervised multi-dimensional pre-training strategy is further developed to learn generalizable spectrum representations across diverse wireless environments. Once pre-trained, FM-RME supports zero-shot inference for multi-dimensional RME, including spatial, temporal, and spectral estimation, without scenario-specific retraining. Simulation results verify that FM-RME exhibits desired learning performance across diverse datasets and zero-shot generalization capabilities beyond existing RME methods. 

---
# To Deceive is to Teach? Forging Perceptual Robustness via Adversarial Reinforcement Learning 

**Authors**: Yicheng Bao, Xuhong Wang, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.22227)  

**Abstract**: Despite their impressive capabilities, Multimodal Large Language Models (MLLMs) exhibit perceptual fragility when confronted with visually complex scenes. This weakness stems from a reliance on finite training datasets, which are prohibitively expensive to scale and impose a ceiling on model robustness. We introduce \textbf{AOT-SFT}, a large-scale adversarial dataset for bootstrapping MLLM robustness. Building on this, we propose \textbf{AOT (Adversarial Opponent Training)}, a self-play framework that forges MLLM robustness by creating its own training data. Our method orchestrates a co-evolution between an image-editing Attacker and a Defender MLLM, where the Attacker generates a diverse and dynamic curriculum of image manipulations, forcing the Defender to adapt and improve. Extensive experiments demonstrate that AOT enhances the Defender's perceptual robustness and reduces hallucinations, establishing a scalable paradigm for training more reliable MLLMs. 

---
# SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG 

**Authors**: Xuechen Zhang, Koustava Goswami, Samet Oymak, Jiasi Chen, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2602.22225)  

**Abstract**: Retrieval-augmented generation (RAG) has strong potential for producing accurate and factual outputs by combining language models (LMs) with evidence retrieved from large text corpora. However, current pipelines are limited by static chunking and flat retrieval: documents are split into short, predetermined, fixed-size chunks, embeddings are retrieved uniformly, and generation relies on whatever chunks are returned. This design brings challenges, as retrieval quality is highly sensitive to chunk size, often introduces noise from irrelevant or misleading chunks, and scales poorly to large corpora. We present SmartChunk retrieval, a query-adaptive framework for efficient and robust long-document question answering (QA). SmartChunk uses (i) a planner that predicts the optimal chunk abstraction level for each query, and (ii) a lightweight compression module that produces high-level chunk embeddings without repeated summarization. By adapting retrieval granularity on the fly, SmartChunk balances accuracy with efficiency and avoids the drawbacks of fixed strategies. Notably, our planner can reason about chunk abstractions through a novel reinforcement learning scheme, STITCH, which boosts accuracy and generalization. To reflect real-world applications, where users face diverse document types and query styles, we evaluate SmartChunk on five QA benchmarks plus one out-of-domain dataset. Across these evaluations, SmartChunk outperforms state-of-the-art RAG baselines, while reducing cost. Further analysis demonstrates strong scalability with larger corpora and consistent gains on out-of-domain datasets, highlighting its effectiveness as a general framework for adaptive retrieval. 

---
# DS SERVE: A Framework for Efficient and Scalable Neural Retrieval 

**Authors**: Jinjian Liu, Yichuan Wang, Xinxi Lyu, Rulin Shao, Joseph E. Gonzalez, Matei Zaharia, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2602.22224)  

**Abstract**: We present DS-Serve, a framework that transforms large-scale text datasets, comprising half a trillion tokens, into a high-performance neural retrieval system. DS-Serve offers both a web interface and API endpoints, achieving low latency with modest memory overhead on a single node. The framework also supports inference-time trade-offs between latency, accuracy, and result diversity. We anticipate that DS-Serve will be broadly useful for a range of applications, including large-scale retrieval-augmented generation (RAG), training data attribution, training search agents, and beyond. 

---
# Misinformation Exposure in the Chinese Web: A Cross-System Evaluation of Search Engines, LLMs, and AI Overviews 

**Authors**: Geng Liu, Junjie Mu, Li Feng, Mengxiao Zhu, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2602.22221)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into search services, providing direct answers that can reduce users' reliance on traditional result pages. Yet their factual reliability in non-English web ecosystems remains poorly understood, particularly when answering real user queries. We introduce a fact-checking dataset of 12~161 Chinese Yes/No questions derived from real-world online search logs and develop a unified evaluation pipeline to compare three information-access paradigms: traditional search engines, standalone LLMs, and AI-generated overview modules. Our analysis reveals substantial differences in factual accuracy and topic-level variability across systems. By combining this performance with real-world Baidu Index statistics, we further estimate potential exposure to incorrect factual information of Chinese users across regions. These findings highlight structural risks in AI-mediated search and underscore the need for more reliable and transparent information-access tools for the digital world. 

---
# What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty 

**Authors**: Bowei Zhang, Jin Xiao, Guanglei Yue, Qianyu He, Yanghua Xiao, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22220)  

**Abstract**: Quotation recommendation aims to enrich writing by suggesting quotes that complement a given context, yet existing systems mostly optimize surface-level topical relevance and ignore the deeper semantic and aesthetic properties that make quotations memorable. We start from two empirical observations. First, a systematic user study shows that people consistently prefer quotations that are ``unexpected yet rational'' in context, identifying novelty as a key desideratum. Second, we find that strong existing models struggle to fully understand the deep meanings of quotations. Inspired by defamiliarization theory, we therefore formalize quote recommendation as choosing contextually novel but semantically coherent quotations. We operationalize this objective with NovelQR, a novelty-driven quotation recommendation framework. A generative label agent first interprets each quotation and its surrounding context into multi-dimensional deep-meaning labels, enabling label-enhanced retrieval. A token-level novelty estimator then reranks candidates while mitigating auto-regressive continuation bias. Experiments on bilingual datasets spanning diverse real-world domains show that our system recommends quotations that human judges rate as more appropriate, more novel, and more engaging than other baselines, while matching or surpassing existing methods in novelty estimation. 

---
# Comparative Analysis of Neural Retriever-Reranker Pipelines for Retrieval-Augmented Generation over Knowledge Graphs in E-commerce Applications 

**Authors**: Teri Rumble, Zbyněk Gazdík, Javad Zarrin, Jagdeep Ahluwalia  

**Link**: [PDF](https://arxiv.org/pdf/2602.22219)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed Natural Language Processing (NLP), enabling complex information retrieval and generation tasks. Retrieval-Augmented Generation (RAG) has emerged as a key innovation, enhancing factual accuracy and contextual grounding by integrating external knowledge sources with generative models. Although RAG demonstrates strong performance on unstructured text, its application to structured knowledge graphs presents challenges: scaling retrieval across connected graphs and preserving contextual relationships during response generation. Cross-encoders refine retrieval precision, yet their integration with structured data remains underexplored. Addressing these challenges is crucial for developing domain-specific assistants that operate in production environments. This study presents the design and comparative evaluation of multiple Retriever-Reranker pipelines for knowledge graph natural language queries in e-Commerce contexts. Using the STaRK Semi-structured Knowledge Base (SKB), a production-scale e-Commerce dataset, we evaluate multiple RAG pipeline configurations optimized for language queries. Experimental results demonstrate substantial improvements over published benchmarks, achieving 20.4% higher Hit@1 and 14.5% higher Mean Reciprocal Rank (MRR). These findings establish a practical framework for integrating domain-specific SKBs into generative systems. Our contributions provide actionable insights for the deployment of production-ready RAG systems, with implications that extend beyond e-Commerce to other domains that require information retrieval from structured knowledge bases. 

---
# RAGdb: A Zero-Dependency, Embeddable Architecture for Multimodal Retrieval-Augmented Generation on the Edge 

**Authors**: Ahmed Bin Khalid  

**Link**: [PDF](https://arxiv.org/pdf/2602.22217)  

**Abstract**: Retrieval-Augmented Generation (RAG) has established itself as the standard paradigm for grounding Large Language Models (LLMs) in domain-specific, up-to-date data. However, the prevailing architecture for RAG has evolved into a complex, distributed stack requiring cloud-hosted vector databases, heavy deep learning frameworks (e.g., PyTorch, CUDA), and high-latency embedding inference servers. This ``infrastructure bloat'' creates a significant barrier to entry for edge computing, air-gapped environments, and privacy-constrained applications where data sovereignty is paramount.
This paper introduces RAGdb, a novel monolithic architecture that consolidates automated multimodal ingestion, ONNX-based extraction, and hybrid vector retrieval into a single, portable SQLite container. We propose a deterministic Hybrid Scoring Function (HSF) that combines sublinear TF-IDF vectorization with exact substring boosting, eliminating the need for GPU inference at query time. Experimental evaluation on an Intel i7-1165G7 consumer laptop demonstrates that RAGdb achieves 100\% Recall@1 for entity retrieval and an ingestion efficiency gain of 31.6x during incremental updates compared to cold starts. Furthermore, the system reduces disk footprint by approximately 99.5\% compared to standard Docker-based RAG stacks, establishing the ``Single-File Knowledge Container'' as a viable primitive for decentralized, local-first AI.
Keywords: Edge AI, Retrieval-Augmented Generation, Vector Search, Green AI, Serverless Architecture, Knowledge Graphs, Efficient Computing. 

---
# Retrieval-Augmented Generation Assistant for Anatomical Pathology Laboratories 

**Authors**: Diogo Pires, Yuriy Perezhohin, Mauro Castelli  

**Link**: [PDF](https://arxiv.org/pdf/2602.22216)  

**Abstract**: Accurate and efficient access to laboratory protocols is essential in Anatomical Pathology (AP), where up to 70% of medical decisions depend on laboratory diagnoses. However, static documentation such as printed manuals or PDFs is often outdated, fragmented, and difficult to search, creating risks of workflow errors and diagnostic delays. This study proposes and evaluates a Retrieval-Augmented Generation (RAG) assistant tailored to AP laboratories, designed to provide technicians with context-grounded answers to protocol-related queries. We curated a novel corpus of 99 AP protocols from a Portuguese healthcare institution and constructed 323 question-answer pairs for systematic evaluation. Ten experiments were conducted, varying chunking strategies, retrieval methods, and embedding models. Performance was assessed using the RAGAS framework (faithfulness, answer relevance, context recall) alongside top-k retrieval metrics. Results show that recursive chunking and hybrid retrieval delivered the strongest baseline performance. Incorporating a biomedical-specific embedding model (MedEmbed) further improved answer relevance (0.74), faithfulness (0.70), and context recall (0.77), showing the importance of domain-specialised embeddings. Top-k analysis revealed that retrieving a single top-ranked chunk (k=1) maximized efficiency and accuracy, reflecting the modular structure of AP protocols. These findings highlight critical design considerations for deploying RAG systems in healthcare and demonstrate their potential to transform static documentation into dynamic, reliable knowledge assistants, thus improving laboratory workflow efficiency and supporting patient safety. 

---
# Enriching Taxonomies Using Large Language Models 

**Authors**: Zeinab Ghamlouch, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2602.22213)  

**Abstract**: Taxonomies play a vital role in structuring and categorizing information across domains. However, many existing taxonomies suffer from limited coverage and outdated or ambiguous nodes, reducing their effectiveness in knowledge retrieval. To address this, we present Taxoria, a novel taxonomy enrichment pipeline that leverages Large Language Models (LLMs) to enhance a given taxonomy. Unlike approaches that extract internal LLM taxonomies, Taxoria uses an existing taxonomy as a seed and prompts an LLM to propose candidate nodes for enrichment. These candidates are then validated to mitigate hallucinations and ensure semantic relevance before integration. The final output includes an enriched taxonomy with provenance tracking and visualization of the final merged taxonomy for analysis. 

---
# Survey on Neural Routing Solvers 

**Authors**: Yunpeng Ba, Xi Lin, Changliang Zhou, Ruihao Zheng, Zhenkun Wang, Xinyan Liang, Zhichao Lu, Jianyong Sun, Yuhua Qian, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21761)  

**Abstract**: Neural routing solvers (NRSs) that leverage deep learning to tackle vehicle routing problems have demonstrated notable potential for practical applications. By learning implicit heuristic rules from data, NRSs replace the handcrafted counterparts in classic heuristic frameworks, thereby reducing reliance on costly manual design and trial-and-error adjustments. This survey makes two main contributions: (1) The heuristic nature of NRSs is highlighted, and existing NRSs are reviewed from the perspective of heuristics. A hierarchical taxonomy based on heuristic principles is further introduced. (2) A generalization-focused evaluation pipeline is proposed to address limitations of the conventional pipeline. Comparative benchmarking of representative NRSs across both pipelines uncovers a series of previously unreported gaps in current research. 

---
# Duel-Evolve: Reward-Free Test-Time Scaling via LLM Self-Preferences 

**Authors**: Sweta Karlekar, Carolina Zheng, Magnus Saebo, Nicolas Beltran-Velez, Shuyang Yu, John Bowlan, Michal Kucer, David Blei  

**Link**: [PDF](https://arxiv.org/pdf/2602.21585)  

**Abstract**: Many applications seek to optimize LLM outputs at test time by iteratively proposing, scoring, and refining candidates over a discrete output space. Existing methods use a calibrated scalar evaluator for the target objective to guide search, but for many tasks such scores are unavailable, too sparse, or unreliable. Pairwise comparisons, by contrast, are often easier to elicit, still provide useful signal on improvement directions, and can be obtained from the LLM itself without external supervision. Building on this observation, we introduce Duel-Evolve, an evolutionary optimization algorithm that replaces external scalar rewards with pairwise preferences elicited from the same LLM used to generate candidates. Duel-Evolve aggregates these noisy candidate comparisons via a Bayesian Bradley-Terry model, yielding uncertainty-aware estimates of candidate quality. These quality estimates guide allocation of the comparison budget toward plausible optima using Double Thompson Sampling, as well as selection of high-quality parents to generate improved candidates. We evaluate Duel-Evolve on MathBench, where it achieves 20 percentage points higher accuracy over existing methods and baselines, and on LiveCodeBench, where it improves over comparable iterative methods by over 12 percentage points. Notably, the method requires no reward model, no ground-truth labels during search, and no hand-crafted scoring function. Results show that pairwise self-preferences provide strong optimization signal for test-time improvement over large, discrete output spaces. 

---
