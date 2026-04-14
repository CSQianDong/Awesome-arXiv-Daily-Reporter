# Self-Distilled Reinforcement Learning for Co-Evolving Agentic Recommender Systems 

**Authors**: Zongwei Wang, Min Gao, Hongzhi Yin, Junliang Yu, Tong Chen, Shazia Sadiq, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10029)  

**Abstract**: Large language model-empowered agentic recommender systems (ARS) reformulate recommendation as a multi-turn interaction between a recommender agent and a user agent, enabling iterative preference elicitation and refinement beyond conventional one-shot prediction. However, existing ARS are mainly optimized in a Reflexion-style paradigm, where past interaction trajectories are stored as textual memory and retrieved as prompt context for later reasoning. Although this design allows agents to recall prior feedback and observations, the accumulated experience remains external to model parameters, leaving agents reliant on generic reasoning rather than progressively acquiring recommendation-specific decision-making ability through learning. Reinforcement learning (RL) therefore provides a natural way to internalize such interaction experience into parameters. Yet existing RL methods for ARS still suffer from two key limitations. First, they fail to capture the interactive nature of ARS, in which the recommender agent and the user agent continuously influence each other and can naturally generate endogenous supervision through interaction feedback. Second, they reduce a rich multi-turn interaction process to final outcomes, overlooking the dense supervision embedded throughout the trajectory. To this end, we propose CoARS, a self-distilled reinforcement learning framework for co-evolving agentic recommender systems. CoARS introduces two complementary learning schemes: interaction reward, which derives coupled task-level supervision for the recommender agent and the user agent from the same interaction trajectory, and self-distilled credit assignment, which converts historical trajectories into token-level credit signals under teacher-student conditioning. Experiments on multiple datasets show that CoARS outperforms representative ARS baselines in recommendation performance and user alignment. 

---
# HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation 

**Authors**: Subham Raj, Aman Vaibhav Jha, Mayank Anand, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2604.10048)  

**Abstract**: Conversational recommender systems (CRSs) operate under incremental preference revelation, requiring systems to make recommendation decisions under uncertainty. While recent approaches particularly those built on large language models achieve strong performance on standard proxy metrics such as Recall@K and BLEU, they often fail to deliver high-quality, user-aligned recommendations in practice. This gap arises because existing methods primarily optimize for intermediate objectives like retrieval accuracy, fluent generation, or tool invocation, rather than recommendation quality itself. We propose HARPO (Hierarchical Agentic Reasoning with Preference Optimization), an agentic framework that reframes conversational recommendation as a structured decision-making process explicitly optimized for multi-dimensional recommendation quality. HARPO integrates hierarchical preference learning that decomposes recommendation quality into interpretable dimensions (relevance, diversity, predicted user satisfaction, and engagement) and learns context-dependent weights over these dimensions; (ii) deliberative tree-search reasoning guided by a learned value network that evaluates candidate reasoning paths based on predicted recommendation quality rather than task completion; and (iii) domain-agnostic reasoning abstractions through Virtual Tool Operations and multi-agent refinement, enabling transferable recommendation reasoning across domains. We evaluate HARPO on ReDial, INSPIRED, and MUSE, demonstrating consistent improvements over strong baselines on recommendation-centric metrics while maintaining competitive response quality. These results highlight the importance of explicit, user-aligned quality optimization for conversational recommendation. 

---
# Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games 

**Authors**: Keyang Zhong, Junlin Xie, Hefeng Wu, Haofeng Li, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11741)  

**Abstract**: Vision-language models (VLMs) have shown impressive capabilities in perceptual tasks, yet they degrade in complex multi-hop reasoning under multiplayer game settings with imperfect and deceptive information. In this paper, we study a representative multiplayer task, Murder Mystery Games, which require inferring hidden truths based on partial clues provided by roles with different intentions. To address this challenge, we propose a collaborative multi-agent framework for evaluating and synthesizing high-quality, role-driven multiplayer game scripts, enabling fine-grained interaction patterns tailored to character identities (i.e., murderer vs. innocent). Our system generates rich multimodal contexts, including character backstories, visual and textual clues, and multi-hop reasoning chains, through coordinated agent interactions. We design a two-stage agent-monitored training strategy to enhance the reasoning ability of VLMs: (1) chain-of-thought based fine-tuning on curated and synthetic datasets that model uncertainty and deception; (2) GRPO-based reinforcement learning with agent-monitored reward shaping, encouraging the model to develop character-specific reasoning behaviors and effective multimodal multi-hop inference. Extensive experiments demonstrate that our method significantly boosts the performance of VLMs in narrative reasoning, hidden fact extraction, and deception-resilient understanding. Our contributions offer a scalable solution for training and evaluating VLMs under uncertain, adversarial, and socially complex conditions, laying the groundwork for future benchmarks in multimodal multi-hop reasoning under imperfect information. 

---
# Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning 

**Authors**: Xiaozhe Li, Tianyi Lyu, Yizhao Yang, Liang Shan, Siyi Yang, Ligao Zhang, Zhuoyi Huang, Qingwen Liu, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11462)  

**Abstract**: Large Language Models (LLMs) struggle with long-horizon tasks due to the "context bottleneck" and the "lost-in-the-middle" phenomenon, where accumulated noise from verbose environments degrades reasoning over multi-turn interactions. To address this issue, we introduce a symbiotic framework that decouples context management from task execution. Our architecture pairs a lightweight, specialized policy model, ContextCurator, with a powerful frozen foundation model, TaskExecutor. Trained via reinforcement learning, ContextCurator actively reduces information entropy in the working memory. It aggressively prunes environmental noise while preserving reasoning anchors, that is, sparse data points that are critical for future deductions. On WebArena, our framework improves the success rate of Gemini-3.0-flash from 36.4% to 41.2% while reducing token consumption by 8.8% (from 47.4K to 43.3K). On DeepSearch, it achieves a 57.1% success rate, compared with 53.9%, while reducing token consumption by a factor of 8. Remarkably, a 7B ContextCurator matches the context management performance of GPT-4o, providing a scalable and computationally efficient paradigm for autonomous long-horizon agents. 

---
# OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems 

**Authors**: Kun Liu, Liqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11477)  

**Abstract**: The alignment of Multi-Agent Systems (MAS) for autonomous software engineering is constrained by evaluator epistemic uncertainty. Current paradigms, such as Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), frequently induce model sycophancy, while execution-based environments suffer from adversarial "Test Evasion" by unconstrained agents. In this paper, we introduce an objective alignment paradigm: \textbf{Out-of-Money Reinforcement Learning (OOM-RL)}. By deploying agents into the non-stationary, high-friction reality of live financial markets, we utilize critical capital depletion as an un-hackable negative gradient. Our longitudinal 20-month empirical study (July 2024 -- February 2026) chronicles the system's evolution from a high-turnover, sycophantic baseline to a robust, liquidity-aware architecture. We demonstrate that the undeniable ontological consequences of financial loss forced the MAS to abandon overfitted hallucinations in favor of the \textbf{Strict Test-Driven Agentic Workflow (STDAW)}, which enforces a Byzantine-inspired uni-directional state lock (RO-Lock) anchored to a deterministically verified $\geq 95\%$ code coverage constraint matrix. Our results show that while early iterations suffered severe execution decay, the final OOM-RL-aligned system achieved a stable equilibrium with an annualized Sharpe ratio of 2.06 in its mature phase. We conclude that substituting subjective human preference with rigorous economic penalties provides a robust methodology for aligning autonomous agents in high-stakes, real-world environments, laying the groundwork for generalized paradigms where computational billing acts as an objective physical constraint 

---
# Learning from Contrasts: Synthesizing Reasoning Paths from Diverse Search Trajectories 

**Authors**: Peiyang Liu, Zhirui Chen, Xi Wang, Di Liang, Youru Li, Zhi Cai, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.11365)  

**Abstract**: Monte Carlo Tree Search (MCTS) has been widely used for automated reasoning data exploration, but current supervision extraction methods remain inefficient. Standard approaches retain only the single highest-reward trajectory, discarding the comparative signals present in the many explored paths. Here we introduce \textbf{Contrastive Reasoning Path Synthesis (CRPS)}, a framework that transforms supervision extraction from a filtering process into a synthesis procedure. CRPS uses a structured reflective process to analyze the differences between high- and low-quality search trajectories, extracting explicit information about strategic pivots and local failure modes. These insights guide the synthesis of reasoning chains that incorporate success patterns while avoiding identified pitfalls. We show empirically that models fine-tuned on just 60K CRPS-synthesized examples match or exceed the performance of baselines trained on 590K examples derived from standard rejection sampling, a 20$\times$ reduction in dataset size. Furthermore, CRPS improves generalization on out-of-domain benchmarks, demonstrating that learning from the contrast between success and failure produces more transferable reasoning capabilities than learning from success alone. 

---
# Mobile GUI Agent Privacy Personalization with Trajectory Induced Preference Optimization 

**Authors**: Zhixin Lin, Jungang Li, Dongliang Xu, Shidong Pan, Yibo Shi, Yuchi Liu, Yuecong Min, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11259)  

**Abstract**: Mobile GUI agents powered by Multimodal Large Language Models (MLLMs) can execute complex tasks on mobile devices. Despite this progress, most existing systems still optimize task success or efficiency, neglecting users' privacy personalization. In this paper, we study the often-overlooked problem of agent personalization. We observe that personalization can induce systematic structural heterogeneity in execution trajectories. For example, privacy-first users often prefer protective actions, e.g., refusing permissions, logging out, and minimizing exposure, leading to logically different execution trajectories from utility-first users. Such variable-length and structurally different trajectories make standard preference optimization unstable and less informative. To address this issue, we propose Trajectory Induced Preference Optimization (TIPO), which uses preference-intensity weighting to emphasize key privacy-related steps and padding gating to suppress alignment noise. Results on our Privacy Preference Dataset show that TIPO improves persona alignment and distinction while preserving strong task executability, achieving 65.60% SR, 46.22 Compliance, and 66.67% PD, outperforming existing optimization methods across various GUI tasks. The code and dataset will be publicly released at this https URL. 

---
# From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning 

**Authors**: Chen Zhan, Xiaoyu Tan, Gengchen Ma, Yu-Jie Xiong, Xiaoyan Jiang, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11137)  

**Abstract**: The integration of Large Language Models (LLMs) into clinical decision support is critically obstructed by their opaque and often unreliable reasoning. In the high-stakes domain of healthcare, correct answers alone are insufficient; clinical practice demands full transparency to ensure patient safety and enable professional accountability. A pervasive and dangerous weakness of current LLMs is their tendency to produce "correct answers through flawed reasoning." This issue is far more than a minor academic flaw; such process errors signal a fundamental lack of robust understanding, making the model prone to broader hallucinations and unpredictable failures when faced with real-world clinical complexity. In this paper, we establish a framework for trustworthy clinical argumentation by adapting the Toulmin model to the diagnostic process. We propose a novel training pipeline: Curriculum Goal-Conditioned Learning (CGCL), designed to progressively train LLM to generate diagnostic arguments that explicitly follow this Toulmin structure. CGCL's progressive three-stage curriculum systematically builds a solid clinical argument: (1) extracting facts and generating differential diagnoses; (2) justifying a core hypothesis while rebutting alternatives; and (3) synthesizing the analysis into a final, qualified conclusion. We validate CGCL using T-Eval, a quantitative framework measuring the integrity of the diagnosis reasoning. Experiments show that our method achieves diagnostic accuracy and reasoning quality comparable to resource-intensive Reinforcement Learning (RL) methods, while offering a more stable and efficient training pipeline. 

---
# From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience 

**Authors**: Jia Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11041)  

**Abstract**: Semiconductor supply chains face unprecedented resilience challenges amidst global geopolitical turbulence. Conventional Large Language Model (LLM) planners, when confronting such non-stationary "Policy Black Swan" events, frequently suffer from Decision Paralysis or a severe Grounding Gap due to the absence of physical environmental modeling. This paper introduces ReflectiChain, a cognitive agentic framework tailored for resilient macroeconomic supply chain planning. The core innovation lies in the integration of Latent Trajectory Rehearsal powered by a generative world model, which couples reflection-in-action (System 2 deliberation) with delayed reflection-on-action. Furthermore, we leverage a Retrospective Agentic RL mechanism to enable autonomous policy evolution during the deployment phase (test-time). Evaluations conducted on our high-fidelity benchmark, Semi-Sim, demonstrate that under extreme scenarios such as export bans and material shortages, ReflectiChain achieves a 250% improvement in average step rewards over the strongest LLM baselines. It successfully restores the Operability Ratio (OR) from a deficient 13.3% to over 88.5% while ensuring robust gradient convergence. Ablation studies further underscore that the synergy between physical grounding constraints and double-loop learning is fundamental to bridging the gap between semantic reasoning and physical reality for long-horizon strategic planning. 

---
# Teaching Language Models How to Code Like Learners: Conversational Serialization for Student Simulation 

**Authors**: Charles Koutcheme, Arto Hellas, Juho Leinonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10720)  

**Abstract**: Artificial models that simulate how learners act and respond within educational systems are a promising tool for evaluating tutoring strategies and feedback mechanisms at scale. However, many existing approaches in programming education rely on prompting large, proprietary language models, raising concerns around privacy, cost, and dependence. In this work, we propose a method for training open-weight artificial programming learners using authentic student process data. Our approach serializes temporal log traces into a conversational format, representing each student's problem-solving process as a dialogue between the learner and their automated assessment system. Student code submissions and environment feedback, such as test outcomes, grades, and error traces, form alternating conversational turns, enabling models to learn from the iterative debugging process. We additionally introduce a training pipeline combining supervised fine-tuning with preference optimization to align models with authentic student debugging behavior. We evaluate our framework by training Qwen models at 4B and 8B scales on a large-scale dataset of real student submissions to Python programming assignments. Our results show that incorporating environment feedback strengthens the models' ability to replicate student debugging behavior, improving over both prior code-only approaches and prompted large language models baselines in functional alignment and code similarity. We release our code to support reproducibility. 

---
# Agent^2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training? 

**Authors**: Wanyi Chen, Xiao Yang, Xu Yang, Tianming Sha, Qizheng Li, Zhuo Wang, Bowen Xian, Fang Kong, Weiqing Liu, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2604.10547)  

**Abstract**: We introduce Agent^2 RL-Bench, a benchmark for evaluating agentic RL post-training -- whether LLM agents can autonomously design, implement, and run complete RL pipelines that improve foundation models. This capability is important because RL post-training increasingly drives model alignment and specialization, yet existing benchmarks remain largely static: supervised fine-tuning alone yields strong results, leaving interactive RL engineering untested. Agent^2 RL-Bench addresses this with six tasks across three levels -- from static rule-based training to closed-loop online RL with trajectory collection -- each adding a structural requirement that prior levels do not impose. The benchmark provides isolated workspaces with a grading API, runtime instrumentation that records every submission and code revision, and automated post-hoc analysis that generates structured run reports, enabling the first automated diagnostic of agent-driven post-training behavior. Across multiple agent stacks spanning five agent systems and six driver LLMs, we find that agents achieve striking interactive gains -- on ALFWorld, an RL-only agent improves from 5.97 to 93.28 via SFT warm-up and GRPO with online rollouts -- yet make only marginal progress on others (DeepSearchQA: +2.75 within evaluation noise), and that driver choice has a large effect on interactive tasks -- within the same scaffold, switching drivers changes interactive improvement from near-zero to +78pp. More broadly, the benchmark reveals that supervised pipelines dominate agent-driven post-training under fixed budgets, with online RL succeeding as the final best route only on ALFWorld. Code is available at this https URL. 

---
# CARO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation 

**Authors**: Bingzhe Wu, Haotian Lu, Yuchen Mou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10504)  

**Abstract**: Current large language models (LLMs), even those explicitly trained for reasoning, often struggle with ambiguous content moderation cases due to misleading "decision shortcuts" embedded in context. Inspired by cognitive psychology insights into expert moderation, we introduce \caro (Chain-of-Analogy Reasoning Optimization), a novel two-stage training framework to induce robust analogical reasoning in LLMs. First, \caro bootstraps analogical reasoning chains via retrieval-augmented generation (RAG) on moderation data and performs supervised fine-tuning (SFT). Second, we propose a customized direct preference optimization (DPO) approach to reinforce analogical reasoning behaviors explicitly. Unlike static retrieval methods, \caro dynamically generates tailored analogical references during inference, effectively mitigating harmful decision shortcuts. Extensive experiments demonstrate that \caro substantially outperforms state-of-the-art reasoning models (DeepSeek R1, QwQ), specialized moderation models (LLaMA Guard), and advanced fine-tuning and retrieval-augmented methods, achieving an average F1 score improvement of 24.9\% on challenging ambiguous moderation benchmarks. 

---
# Beyond Compliance: A Resistance-Informed Motivation Reasoning Framework for Challenging Psychological Client Simulation 

**Authors**: Danni Liu, Bo Liu, Yuxin Hu, Hantao Zhao, Yan Liu, Ding Ding, Jiahui Jin, Jiuxin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10507)  

**Abstract**: Psychological client simulators have emerged as a scalable solution for training and evaluating counselor trainees and psychological LLMs. Yet existing simulators exhibit unrealistic over-compliance, leaving counselors underprepared for the challenging behaviors common in real-world practice. To bridge this gap, we present ResistClient, which systematically models challenging client behaviors grounded in Client Resistance Theory by integrating external behaviors with underlying motivational mechanisms. To this end, we propose Resistance-Informed Motivation Reasoning (RIMR), a two-stage training framework. First, RIMR mitigates compliance bias via supervised fine-tuning on RPC, a large-scale resistance-oriented psychological conversation dataset covering diverse client profiles. Second, beyond surface-level response imitation, RIMR models psychologically coherent motivation reasoning before response generation, jointly optimizing motivation authenticity and response consistency via process-supervised reinforcement learning. Extensive automatic and expert evaluations show that ResistClient substantially outperforms existing simulators in challenge fidelity, behavioral plausibility, and reasoning coherence. Moreover, ResistClient facilities evaluation of psychological LLMs under challenging conditions, offering new optimization directions for mental health dialogue systems. 

---
# SVSR: A Self-Verification and Self-Rectification Paradigm for Multimodal Reasoning 

**Authors**: Zhe Qian, Nianbing Su, Zhonghua Wang, Hebei Li, Zhongxing Xu, Yueying Li, Fei Luo, Zhuohan Ouyang, Yanbiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10228)  

**Abstract**: Current multimodal models often suffer from shallow reasoning, leading to errors caused by incomplete or inconsistent thought processes. To address this limitation, we propose Self-Verification and Self-Rectification (SVSR), a unified framework that explicitly integrates self-verification and self-rectification into the model's reasoning pipeline, substantially improving robustness and reliability in complex visual understanding and multimodal reasoning tasks. SVSR is built on a novel three-stage training paradigm. First, we construct a high-quality unified preference dataset by refining reasoning traces from pre-trained vision-language models, incorporating both forward and backward reasoning to embed self-reflective signals. Second, we perform cold-start supervised fine-tuning on this dataset to learn structured, multi-step reasoning behaviors. Third, we apply a Semi-online Direct Preference Optimization (Semi-online DPO) process, continuously augmenting the training corpus with high-quality, model-generated reasoning traces filtered by a powerful teacher VLM. This pipeline enables the model to learn, elicit, and refine its ability to self-verify and self-rectify. Extensive experiments across diverse benchmarks demonstrate that SVSR improves reasoning accuracy and enables stronger generalization to unseen tasks and question types. Notably, once trained with explicit self-reflective reasoning, the model also exhibits improved implicit reasoning ability, outperforming strong baselines even when no explicit reasoning traces are provided. These results highlight the potential of SVSR for building more dependable, introspective, and cognitively aligned multimodal systems. 

---
# Cognitive Pivot Points and Visual Anchoring: Unveiling and Rectifying Hallucinations in Multimodal Reasoning Models 

**Authors**: Zhe Qian, Yanbiao Ma, Zhuohan Ouyang, Zhonghua Wang, Zhongxing Xu, Fei Luo, Xinyu Liu, Zongyuan Ge, Yike Guo, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.10219)  

**Abstract**: Multimodal Large Reasoning Models (MLRMs) have achieved remarkable strides in visual reasoning through test time compute scaling, yet long chain reasoning remains prone to hallucinations. We identify a concerning phenomenon termed the Reasoning Vision Truth Disconnect (RVTD): hallucinations are strongly correlated with cognitive bifurcation points that often exhibit high entropy states. We attribute this vulnerability to a breakdown in visual semantic anchoring, localized within the network's intermediate layers; specifically, during these high uncertainty transitions, the model fails to query visual evidence, reverting instead to language priors. Consequently, we advocate a shift from solely outcome level supervision to augmenting it with fine grained internal attention guidance. To this end, we propose V-STAR (Visual Structural Training with Attention Reinforcement), a lightweight, holistic training paradigm designed to internalize visually aware reasoning capabilities. Central to our approach is the Hierarchical Visual Attention Reward (HVAR), integrated within the GRPO framework. Upon detecting high entropy states, this mechanism dynamically incentivizes visual attention across critical intermediate layers, thereby anchoring the reasoning process back to the visual input. Furthermore, we introduce the Forced Reflection Mechanism (FRM), a trajectory editing strategy that disrupts cognitive inertia by triggering reflection around high entropy cognitive bifurcation points and encouraging verification of subsequent steps against the visual input, thereby translating external debiasing interventions into an intrinsic capability for hallucination mitigation. 

---
# AI Achieves a Perfect LSAT Score 

**Authors**: Bonmu Ku  

**Link**: [PDF](https://arxiv.org/pdf/2604.10034)  

**Abstract**: This paper reports the first documented instance of a language model achieving a perfect score on an officially disclosed Law School Admission Test (LSAT). Controlled experiments on eight reasoning models show that varying the prompt, shuffling answer choices, and sampling multiple responses have no meaningful effect as drivers of performance. Ablating the thinking phase that models generate before answering, however, lowers frontier accuracy by up to 8 percentage points, predominantly in logical reasoning. Distilled models produce full thinking traces in the same format yet plateau far below frontier performance. A pilot process reward model fine-tuned via QLoRA on official LSAT explanations narrows this gap through Best-of-5 selection, with gains again predominantly in logical reasoning. The gatekeeper of elite legal education since 1948, the LSAT has not merely been passed but answered without a single error by models that reason. The upper bound of the cognitive capacities it has tested is no longer exclusive to human cognition. 

---
# FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks 

**Authors**: Yupeng Cao, Haohang Li, Weijin Liu, Wenbo Cao, Anke Xu, Lingfei Qian, Xueqing Peng, Minxue Tang, Zhiyuan Yao, Jimin Huang, K.P. Subbalakshmi, Zining Zhu, Jordan W. Suchow, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10015)  

**Abstract**: Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3.5-9B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality. 

---
# MEMENTO: Teaching LLMs to Manage Their Own Context 

**Authors**: Vasilis Kontonis, Yuchen Zeng, Shivam Garg, Lingjiao Chen, Hao Tang, Ziyan Wang, Ahmed Awadallah, Eric Horvitz, John Langford, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.09852)  

**Abstract**: Reasoning models think in long, unstructured streams with no mechanism for compressing or organizing their own intermediate state. We introduce MEMENTO: a method that teaches models to segment reasoning into blocks, compress each block into a memento, i.e., a dense state summary, and reason forward by attending only to mementos, reducing context, KV cache, and compute. To train MEMENTO models, we release OpenMementos, a public dataset of 228K reasoning traces derived from OpenThoughts-v3, segmented and annotated with intermediate summaries. We show that a two-stage SFT recipe on OpenMementos is effective across different model families (Qwen3, Phi-4, Olmo 3) and scales (8B--32B parameters). Trained models maintain strong accuracy on math, science, and coding benchmarks while achieving ${\sim}2.5\times$ peak KV cache reduction. We extend vLLM to support our inference method, achieving ${\sim}1.75\times$ throughput improvement while also enabling us to perform RL and further improve accuracy. Finally, we identify a dual information stream: information from each reasoning block is carried both by the memento text and by the corresponding KV states, which retain implicit information from the original block. Removing this channel drops accuracy by 15\,pp on AIME24. 

---
# Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards 

**Authors**: Shuze Daniel Liu, Claire Chen, Jiabao Sean Xiao, Lei Lei, Yuheng Zhang, Yisong Yue, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09855)  

**Abstract**: The recent advancement of Large Language Models (LLMs) has established their potential as autonomous interactive agents. However, they often struggle in strategic games of incomplete information, such as bilateral price negotiation. In this paper, we investigate if Reinforcement Learning from Verifiable Rewards (RLVR) can effectively teach LLMs to negotiate. Specifically, we explore the strategic behaviors that emerge during the learning process. We introduce a framework that trains a mid-sized buyer agent against a regulated LLM seller across a wide distribution of real-world products. By grounding reward signals directly in the maximization of economic surplus and strict adherence to private budget constraints, we reveal a novel four-phase strategic evolution. The agent progresses from naive bargaining to using aggressive starting prices, moves through a phase of deadlock, and ultimately develops sophisticated persuasive skills. Our results demonstrate that this verifiable training allows a 30B agent to significantly outperform frontier models over ten times its size in extracting surplus. Furthermore, the trained agent generalizes robustly to stronger counterparties unseen during training and remains effective even when facing hostile, adversarial seller personas. 

---
# Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning 

**Authors**: Siyuan Xu, Shiyang Li, Xin Liu, Tianyi Liu, Yixiao Li, Zhan Shi, Zixuan Zhang, Zilong Wang, Qingyu Yin, Jianshu Chen, Tuo Zhao, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09813)  

**Abstract**: Existing synthetic tool-use corpora are primarily designed for offline supervised fine-tuning, yet reinforcement learning (RL) requires executable environments that support reward-checkable online rollouts. We propose COVERT, a two-stage pipeline that first generates reliable base tool-use trajectories through self-evolving synthesis with multi-level validation, and then applies oracle-preserving augmentations that systematically increase environmental complexity. These augmentations introduce distractor tools, indirect or ambiguous user queries, and noisy, multi-format, or erroneous tool outputs, while strictly preserving oracle tool calls and final answers as ground truth. This design enables automatic reward computation via reference matching for standard cases and lightweight judge-assisted verification for special behaviors such as error detection, supporting RL optimization of tool-calling policies. On Qwen2.5-Instruct-14B, COVERT-RL improves overall accuracy on BFCL v3 from 56.5 to 59.9 and on ACEBench from 53.0 to 59.3, with minimal regressions on general-ability benchmarks; when stacked on SFT, it further reaches 62.1 and 61.8, confirming additive gains. These results suggest that oracle-preserving synthetic environments offer a practical RL refinement stage, complementary to SFT, for improving tool-use robustness under ambiguity and unreliable tool feedback. 

---
# OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling 

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09580)  

**Abstract**: Standard Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs) with reasoning capabilities, yet its reliance on linear natural language is inherently insufficient for effective world modeling in embodied tasks. While text offers flexibility, it fails to explicitly represent the state-space, object hierarchies, and causal dependencies required for robust robotic planning. To address these limitations, we propose Object-Oriented World Modeling (OOWM), a novel framework that structures embodied reasoning through the lens of software engineering formalisms. We redefine the world model not as a latent vector space, but as an explicit symbolic tuple $W = \langle S, T \rangle$: a State Abstraction ($G_\text{state}$) instantiating the environmental state $S$, coupled with a Control Policy ($G_\text{control}$) representing the transition logic $T: S \times A \rightarrow S'$. OOWM leverages the Unified Modeling Language (UML) to materialize this definition: it employs Class Diagrams to ground visual perception into rigorous object hierarchies, and Activity Diagrams to operationalize planning into executable control flows. Furthermore, we introduce a three-stage training pipeline combining Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO). Crucially, this method utilizes outcome-based rewards from the final plan to implicitly optimize the underlying object-oriented reasoning structure, enabling effective learning even with sparse annotations. Extensive evaluations on the MRoom-30k benchmark demonstrate that OOWM significantly outperforms unstructured textual baselines in planning coherence, execution success, and structural fidelity, establishing a new paradigm for structured embodied reasoning. 

---
# Solving Physics Olympiad via Reinforcement Learning on Physics Simulators 

**Authors**: Mihir Prabhudesai, Aryan Satpathy, Yangmin Li, Zheyang Qin, Nikash Bhardwaj, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2604.11805)  

**Abstract**: We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question-answer (QA) pairs, a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question-answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: this https URL. 

---
# Discourse Diversity in Multi-Turn Empathic Dialogue 

**Authors**: Hongli Zhan, Emma S. Gueorguieva, Javier Hernandez, Jina Suh, Desmond C. Ong, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11742)  

**Abstract**: Large language models (LLMs) produce responses rated as highly empathic in single-turn settings (Ayers et al., 2023; Lee et al., 2024), yet they are also known to be formulaic generators that reuse the same lexical patterns, syntactic templates, and discourse structures across tasks (Jiang et al., 2025; Shaib et al., 2024; Namuduri et al., 2025). Less attention has been paid to whether this formulaicity extends to the level of discourse moves, i.e., what a response does for the person it is addressing. This question is especially consequential for empathic dialogue, where effective support demands not just a kind response at one moment but varied strategies as a conversation unfolds (Stiles et al., 1998). Indeed, prior work shows that LLMs reuse the same tactic sequences more than human supporters in single-turn settings (Gueorguieva et al., 2026). We extend this analysis to multi-turn conversations and find that the rigidity compounds: once a tactic appears in a supporter turn, LLMs reuse it in the next at nearly double the rate of humans (0.50-0.56 vs. 0.27). This pattern holds across LLMs serving as supporters in real emotional support conversations, and is invisible to standard similarity metrics. To address this gap, we introduce MINT (Multi-turn Inter-tactic Novelty Training), the first reinforcement learning framework to optimize discourse move diversity across multi-turn empathic dialogue. The best MINT variant combines an empathy quality reward with a cross-turn tactic novelty signal, improving aggregate empathy by 25.3% over vanilla across 1.7B and 4B models while reducing cross-turn discourse move repetition by 26.3% on the 4B model, surpassing all baselines including quality-only and token-level diversity methods on both measures. These results suggest that what current models lack is not empathy itself, but the ability to vary their discourse moves across a conversation. 

---
# Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind 

**Authors**: Hanqi Xiao, Vaidehi Patil, Zaid Khan, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.11666)  

**Abstract**: As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, ToM for Steering Beliefs (ToM-SB), in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task. 

---
# Belief-Aware VLM Model for Human-like Reasoning 

**Authors**: Anshul Nayak, Shahil Shaik, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09686)  

**Abstract**: Traditional neural network models for intent inference rely heavily on observable states and struggle to generalize across diverse tasks and dynamic environments. Recent advances in Vision Language Models (VLMs) and Vision Language Action (VLA) models introduce common-sense reasoning through large-scale multimodal pretraining, enabling zero-shot performance across tasks. However, these models still lack explicit mechanisms to represent and update belief, limiting their ability to reason like humans or capture the evolving human intent over long-horizon. To address this, we propose a belief-aware VLM framework that integrates retrieval-based memory and reinforcement learning. Instead of learning an explicit belief model, we approximate belief using a vector-based memory that retrieves relevant multimodal context, which is incorporated into the VLM for reasoning. We further refine decision-making using a reinforcement learning policy over the VLM latent space. We evaluate our approach on publicly available VQA datasets such as HD-EPIC and demonstrate consistent improvements over zero-shot baselines, highlighting the importance of belief-aware reasoning. 

---
# Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization 

**Authors**: Jiashu Yao, Heyan Huang, Chuwei Luo, Daiqing Wu, Zeming Liu, Yuhang Guo, Yangyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11510)  

**Abstract**: To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals. 

---
# E2E-REME: Towards End-to-End Microservices Auto-Remediation via Experience-Simulation Reinforcement Fine-Tuning 

**Authors**: Lingzhe Zhang, Yunpeng Zhai, Tong Jia, Minghua He, Chiming Duan, Zhaoyang Liu, Bolin Ding, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11094)  

**Abstract**: Contemporary microservice systems continue to grow in scale and complexity, leading to increasingly frequent and costly failures. While recent LLM-based auto-remediation approaches have emerged, they primarily translate textual instructions into executable Ansible playbooks and rely on expert-crafted prompts, lacking runtime knowledge guidance and depending on large-scale general-purpose LLMs, which limits their accuracy and efficiency. We introduce \textit{End-to-End Microservice Remediation} (E2E-MR), a new task that requires directly generating executable playbooks from diagnosis reports to autonomously restore faulty systems. To enable rigorous evaluation, we build \textit{MicroRemed}, a benchmark that automates microservice deployment, failure injection, playbook execution, and post-repair verification. We further propose \textit{E2E-REME}, an end-to-end auto-remediation model trained via experience-simulation reinforcement fine-tuning. Experiments on public and industrial microservice platforms, compared with nine representative LLMs, show that E2E-REME achieves superior accuracy and efficiency. 

---
# Rethinking Token-Level Credit Assignment in RLVR: A Polarity-Entropy Analysis 

**Authors**: Yuhang He, Haodong Wu, Siyi Liu, Hongyu Ge, Hange Zhou, Keyi Wu, Zhuo Zheng, Qihong Lin, Zixin Zhong, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11056)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved the reasoning ability of Large Language Models (LLMs). However, its sparse outcome-based rewards pose a fundamental credit assignment problem. We analyze this problem through the joint lens of reward polarity and token entropy. Our diagnostic tool, the Four Quadrant Decomposition, isolates token updates by polarity and entropy, and controlled ablations show that reasoning improvements concentrate in the high-entropy quadrants. To justify this observation theoretically, we adapt Conditional Mutual Information to the autoregressive RLVR setting and prove that the credit a token can carry is upper-bounded by its entropy. This view yields testable predictions that reasoning gains arise primarily from high-entropy tokens, with unique roles for positive and negative updates. A gradient analysis of GRPO further reveals how uniform reward broadcast dilutes signal at high-entropy positions while over-crediting deterministic tokens. Grounded in these insights, we propose Entropy-Aware Policy Optimization (EAPO) that modulates token-level learning signals accordingly. Extensive experiments demonstrate that EAPO outperforms strong baselines across two model families. 

---
# Shared Emotion Geometry Across Small Language Models: A Cross-Architecture Study of Representation, Behavior, and Methodological Confounds 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11050)  

**Abstract**: We extract 21-emotion vector sets from twelve small language models (six architectures x base/instruct, 1B-8B parameters) under a unified comprehension-mode pipeline at fp16 precision, and compare the resulting geometries via representational similarity analysis on raw cosine RDMs. The five mature architectures (Qwen 2.5 1.5B, SmolLM2 1.7B, Llama 3.2 3B, Mistral 7B v0.3, Llama 3.1 8B) share nearly identical 21-emotion geometry, with pairwise RDM Spearman correlations of 0.74-0.92. This universality persists across diametrically opposed behavioral profiles: Qwen 2.5 and Llama 3.2 occupy opposite poles of MTI Compliance facets yet produce nearly identical emotion RDMs (rho = 0.81), so behavioral facet differences arise above the shared emotion representation. Gemma-3 1B base, the one immature case in our dataset, exhibits extreme residual-stream anisotropy (0.997) and is restructured by RLHF across all geometric descriptors, whereas the five already-mature families show within-family base x instruct RDM correlations of rho >= 0.92 (Mistral 7B v0.3 at rho = 0.985), suggesting RLHF restructures only representations that are not yet organized. Methodologically, we show that what prior work has read as a single comprehension-vs-generation method effect in fact decomposes into four distinct layers -- a coarse method-dependent dissociation, robust sub-parameter sensitivity within generation, a true precision (fp16 vs INT8) effect, and a conflated cross-experiment bias that distorts in opposite directions for different models -- so that a single rho between two prior emotion-vector studies is not a safe basis for interpretation without the layered decomposition. 

---
# RTMC: Step-Level Credit Assignment via Rollout Trees 

**Authors**: Tao Wang, Suhang Zheng, Xiaoxiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11037)  

**Abstract**: Multi-step agentic reinforcement learning benefits from fine-grained credit assignment, yet existing approaches offer limited options: critic-free methods like GRPO assign a uniform advantage to every action in a trajectory, while learned value networks introduce notable overhead and can be fragile under sparse rewards. We observe that group rollouts targeting the same problem often traverse overlapping intermediate states, implicitly forming a tree whose branches diverge at successive decision points. Building on this insight, we introduce Rollout-Tree Monte Carlo (RTMC) advantage estimation, which aggregates return statistics across rollouts sharing a common state to produce per-step Q-values and advantages--without any learned critic. A state-action signature system compresses raw interaction histories into compact, comparable representations, making cross-rollout state matching tractable. On SWE-bench Verified, RTMC improves pass@1 by 3.2 percentage points over GRPO. 

---
# MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models 

**Authors**: Xincheng Yao, Zefeng Qian, Chao Shi, Jiayang Song, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10971)  

**Abstract**: In the progress of industrial anomaly detection, general anomaly detection (GAD) is an emerging trend and also the ultimate goal. Unlike the conventional single- and multi-class AD, general AD aims to train a general AD model that can directly detect anomalies in diverse novel classes without any retraining or fine-tuning on the target data. Recently, Multimodal Large Language Models (MLLMs) have shown great promise in achieving general anomaly detection due to their revolutionary visual understanding and language reasoning capabilities. However, MLLM's general AD ability remains underexplored due to: (1) MLLMs are pretrained on amounts of data sourced from the Web, these data still have significant gaps with the data in AD scenarios. Moreover, the image-text pairs during pretraining are also not specifically for AD tasks. (2) The current mainstream AD datasets are image-based and not yet suitable for post-training MLLMs. To facilitate MLLM-based general AD research, we present MMR-AD, which is a comprehensive benchmark for both training and evaluating MLLM-based AD models. With MMR-AD, we reveal that the AD performance of current SOTA generalist MLLMs still falls far behind the industrial requirements. Based on MMR-AD, we also propose a baseline model, Anomaly-R1, which is a reasoning-based AD model that learns from the CoT data in MMR-AD and is further enhanced by reinforcement learning. Extensive experiments show that our Anomaly-R1 achieves remarkable improvements over generalist MLLMs in both anomaly detection and localization. 

---
# You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass 

**Authors**: Yinuo Yang, Zixian Ma, Manasi Ganti, Jieyu Zhang, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.10966)  

**Abstract**: We present a discriminative multimodal reward model that scores all candidate responses in a single forward pass. Conventional discriminative reward models evaluate each response independently, requiring multiple forward passes, one for each potential response. Our approach concatenates multiple responses with separator tokens and applies cross-entropy over their scalar scores, enabling direct comparative reasoning and efficient $N$-way preference learning. The multi-response design also yields up to $N\times$ wall-clock speedup and FLOPs reduction over conventional single-response scoring. To enable $N$-way reward evaluation beyond existing pairwise benchmarks, we construct two new benchmarks: (1) MR$^2$Bench-Image contains human-annotated rankings over responses from 8 diverse models; (2) MR$^2$Bench-Video is a large-scale video-based reward benchmark derived from 94K crowdsourced pairwise human judgments over video question-answering spanning 19 models, denoised via preference graph ensemble. Both benchmarks provide 4-response evaluation variants sampled from the full rankings. Built on a 4B vision-language backbone with LoRA fine-tuning and a lightweight MLP value head, our model achieves state-of-the-art results on six multimodal reward benchmarks, including MR$^2$Bench-Image, MR$^2$Bench-Video, and four other existing benchmarks. Our model outperforms existing larger generative and discriminative reward models. We further demonstrate that our reward model, when used in reinforcement learning with GRPO, produces improved policy models that maintain performance across standard multimodal benchmarks while substantially improving open-ended generation quality, outperforming a single-response discriminative reward model (RM) baseline by a large margin in both training stability and open-ended generation quality. 

---
# TInR: Exploring Tool-Internalized Reasoning in Large Language Models 

**Authors**: Qiancheng Xu, Yongqi Li, Fan Liu, Hongru Wang, Min Yang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10788)  

**Abstract**: Tool-Integrated Reasoning (TIR) has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools during reasoning. Existing TIR methods typically rely on external tool documentation during reasoning. However, this leads to tool mastery difficulty, tool size constraints, and inference inefficiency. To mitigate these issues, we explore Tool-Internalized Reasoning (TInR), aiming at facilitating reasoning with tool knowledge internalized into LLMs. Achieving this goal presents notable requirements, including tool internalization and tool-reasoning coordination. To address them, we propose TInR-U, a tool-internalized reasoning framework for unified reasoning and tool usage. TInR-U is trained through a three-phase pipeline: 1) tool internalization with a bidirectional knowledge alignment strategy; 2) supervised fine-tuning warm-up using high-quality reasoning annotations, and 3) reinforcement learning with TInR-specific rewards. We comprehensively evaluate our method across in-domain and out-of-domain settings. Experiment results show that TInR-U achieves superior performance in both settings, highlighting its effectiveness and efficiency. 

---
# Advancing Polish Language Modeling through Tokenizer Optimization in the Bielik v3 7B and 11B Series 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2604.10799)  

**Abstract**: The development of the Bielik v3 PL series, encompassing both the 7B and 11B parameter variants, represents a significant milestone in the field of language-specific large language model (LLM) optimization. While general-purpose models often demonstrate impressive multilingual capabilities, they frequently suffer from a fundamental architectural inefficiency: the use of universal tokenizers. These tokenizers, typically designed to cover a broad spectrum of languages, often fail to capture the morphological nuances of specific languages like Polish, leading to higher fertility ratios, increased inference costs, and restricted effective context windows. This report details the transition from the universal Mistral-based tokenization to a dedicated Polish-optimized vocabulary for the Bielik v3 models, exploring the FOCUS-based embedding initialization, the multi-stage pretraining curriculum, and the subsequent post-training alignment involving Supervised Fine-Tuning, Direct Preference Optimization, and Reinforcement Learning through Group Relative Policy Optimization with verifiable rewards. 

---
# SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting 

**Authors**: Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10688)  

**Abstract**: On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted MLE to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six reasoning benchmarks show that SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, demonstrating its consistent effectiveness. 

---
# Tail-Aware Information-Theoretic Generalization for RLHF and SGLD 

**Authors**: Huiming Zhang, Binghan Li, Wan Tian, Qiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.10727)  

**Abstract**: Classical information-theoretic generalization bounds typically control the generalization gap through KL-based mutual information and therefore rely on boundedness or sub-Gaussian tails via the moment generating function (MGF). In many modern pipelines, such as robust learning, RLHF, and stochastic optimization, losses and rewards can be heavy-tailed, and MGFs may not exist, rendering KL-based tools ineffective. We develop a tail-dependent information-theoretic framework for sub-Weibull data, where the tail parameter $\theta$ controls the tail heaviness: $\theta=2$ corresponds to sub-Gaussian, $\theta=1$ to sub-exponential, and $0<\theta<1$ to genuinely heavy tails. Our key technical ingredient is a decorrelation lemma that bounds change-of-measure expectations using a shifted-log $f_\theta$-divergence, which admits explicit comparisons to Rényi divergence without MGF arguments. On the empirical-process side, we establish sharp maximal inequalities and a Dudley-type chaining bound for sub-Weibull processes with tail index $\theta$, with complexity scaling as $\log^{1/\theta}$ and entropy$^{1/\theta}$. These tools yield expected and high-probability PAC-Bayes generalization bounds, as well as an information-theoretic chaining inequality based on multiscale Rényi mutual information. We illustrate the consequences in Rényi-regularized RLHF under heavy-tailed rewards and in stochastic gradient Langevin dynamics with heavy-tailed gradient noise. 

---
# Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents 

**Authors**: Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10674)  

**Abstract**: Reinforcement learning (RL) has been widely used to train LLM agents for multi-turn interactive tasks, but its sample efficiency is severely limited by sparse rewards and long horizons. On-policy self-distillation (OPSD) alleviates this by providing dense token-level supervision from a privileged teacher that has access to ground-truth answers. However, such fixed privileged information cannot capture the diverse valid strategies in agent tasks, and naively combining OPSD with RL often leads to training collapse. To address these limitations, we introduce Skill-SD, a framework that turns the agent's own trajectories into dynamic training-only supervision. Completed trajectories are summarized into compact natural language skills that describe successful behaviors, mistakes, and workflows. These skills serve as dynamic privileged information conditioning only the teacher, while the student always acts under the plain task prompt and learns to internalize the guidance through distillation. To stabilize the training, we derive an importance-weighted reverse-KL loss to provide gradient-correct token-level distillation, and dynamically synchronize the teacher with the improving student. Experimental results on agentic benchmarks demonstrate that Skill-SD substantially outperforms the standard RL baseline, improving both vanilla GRPO (+14.0%/+10.9% on AppWorld/Sokoban) and vanilla OPD (+42.1%/+40.6%). Project page: this https URL 

---
# Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning 

**Authors**: Zikang Shan, Han Zhong, Liwei Wang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10701)  

**Abstract**: Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a value estimate. We further introduce In-Context Conditioning, which helps the critic remain calibrated to the current actor throughout training. GenAC improves value approximation, ranking reliability, and out-of-distribution generalization, and these gains translate into stronger downstream RL performance than both value-based and value-free baselines. Overall, our results suggest that stronger value modeling is a promising direction for improving credit assignment in LLM reinforcement learning. 

---
# Efficient Process Reward Modeling via Contrastive Mutual Information 

**Authors**: Nakyung Lee, Sangwoo Hong, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10660)  

**Abstract**: Recent research has devoted considerable effort to verifying the intermediate reasoning steps of chain-of-thought (CoT) trajectories using process reward models (PRMs) and other verifier models. However, training a PRM typically requires human annotators to assign reward scores to each reasoning step, which is both costly and time-consuming. Existing automated approaches, such as Monte Carlo (MC) estimation, also demand substantial computational resources due to repeated LLM rollouts. To overcome these limitations, we propose contrastive pointwise mutual information (CPMI), a novel automatic reward labeling method that leverages the model's internal probability to infer step-level supervision while significantly reducing the computational burden of annotating dataset. CPMI quantifies how much a reasoning step increases the mutual information between the step and the correct target answer relative to hard-negative alternatives. This contrastive signal serves as a proxy for the step's contribution to the final solution and yields a reliable reward. The experimental results show that CPMI-based labeling reduces dataset construction time by 84% and token generation by 98% compared to MC estimation, while achieving higher accuracy on process-level evaluations and mathematical reasoning benchmarks. 

---
# Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty Quantification in LLMs 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10585)  

**Abstract**: Modern large language models (LLMs) are increasingly fine-tuned via reinforcement learning from human feedback (RLHF) or related reward optimisation schemes. While such procedures improve perceived helpfulness, we investigate whether sycophantic reward signals degrade calibration -- a property essential for reliable uncertainty quantification. We fine-tune Qwen3-8B under three regimes: no fine-tuning (base), neutral supervised fine-tuning (SFT) on TriviaQA, and sycophancy-inducing Group Relative Policy Optimisation (GRPO) that rewards agreement with planted wrong answers. Evaluating on $1{,}000$ MMLU items across five subject domains with bootstrap confidence intervals and permutation testing, we find that \textbf{sycophantic GRPO produces consistent directional calibration degradation} -- ECE rises by $+0.006$ relative to the base model and MCE increases by $+0.010$ relative to neutral SFT -- though the effect does not reach statistical significance ($p = 0.41$) at this training budget. Post-hoc matrix scaling applied to all three models reduces ECE by $40$--$64\%$ and improves accuracy by $1.5$--$3.0$ percentage points. However, the sycophantic model retains the highest post-scaling ECE relative to the neutral SFT control ($0.042$ vs.\ $0.037$), suggesting that reward-induced miscalibration leaves a structured residual even after affine correction. These findings establish a methodology for evaluating the calibration impact of reward hacking and motivate calibration-aware training objectives. 

---
# PAS: Estimating the target accuracy before domain adaptation 

**Authors**: Raphaella Diniz, Jackson de Faria, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2604.09863)  

**Abstract**: The goal of domain adaptation is to make predictions for unlabeled samples from a target domain with the help of labeled samples from a different but related source domain. The performance of domain adaptation methods is highly influenced by the choice of source domain and pre-trained feature extractor. However, the selection of source data and pre-trained model is not trivial due to the absence of a labeled validation set for the target domain and the large number of available pre-trained models. In this work, we propose PAS, a novel score designed to estimate the transferability of a source domain set and a pre-trained feature extractor to a target classification task before actually performing domain adaptation. PAS leverages the generalization power of pre-trained models and assesses source-target compatibility based on the pre-trained feature embeddings. We integrate PAS into a framework that indicates the most relevant pre-trained model and source domain among multiple candidates, thus improving target accuracy while reducing the computational overhead. Extensive experiments on image classification benchmarks demonstrate that PAS correlates strongly with actual target accuracy and consistently guides the selection of the best-performing pre-trained model and source domain for adaptation. 

---
# MedLVR: Latent Visual Reasoning for Reliable Medical Visual Question Answering 

**Authors**: Suyang Xi, Songtao Hu, Yuxiang Lai, Wangyun Dan, Yaqi Liu, Shansong Wang, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09757)  

**Abstract**: Medical vision--language models (VLMs) have shown strong potential for medical visual question answering (VQA), yet their reasoning remains largely text-centric: images are encoded once as static context, and subsequent inference is dominated by language. This paradigm is fundamentally limited in clinical scenarios, where accurate answers often depend on subtle, localized visual evidence that cannot be reliably preserved in static embeddings. We propose \textsc{MedLVR}, a latent visual reasoning framework that introduces an explicit visual evidence state into autoregressive decoding. Instead of relying solely on text-based intermediate reasoning, \textsc{MedLVR} interleaves a short latent reasoning segment within the decoder by reusing hidden states as continuous latent steps, enabling iterative preservation and refinement of query-relevant visual evidence before answer generation. To support effective visual supervision, we adopt a two-stage training strategy: region of interest (ROI)-supervised fine-tuning aligns latent states with clinically relevant image evidence, and Visual-Latent Policy Optimization (VLPO) further optimizes latent reasoning and answer generation under outcome-level rewards. Experiments on OmniMedVQA and five external medical VQA benchmarks show that \textsc{MedLVR} consistently outperforms recent reasoning baselines and improves the average score over the Qwen2.5-VL-7B backbone from 48.3\% to 53.4\%. These results show that latent visual reasoning provides an effective mechanism for preserving diagnostically relevant visual evidence and improving the reliability of medical VQA. 

---
# Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward 

**Authors**: Weiyang Guo, Zesheng Shi, Zeen Zhu, Yuan Zhou, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09748)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is an emerging paradigm that significantly boosts a Large Language Model's (LLM's) reasoning abilities on complex logical tasks, such as mathematics and programming. However, we identify, for the first time, a latent vulnerability to backdoor attacks within the RLVR framework. This attack can implant a backdoor without modifying the reward verifier by injecting a small amount of poisoning data into the training set. Specifically, we propose a novel trigger mechanism designated as the \ourapproach (ACB). The attack exploits the RLVR training loop by assigning substantial positive rewards for harmful responses and negative rewards for refusals. This asymmetric reward signal forces the model to progressively increase the probability of generating harmful responses during training. Our findings demonstrate that the RLVR backdoor attack is characterized by both high efficiency and strong generalization capabilities. Utilizing less than 2\% poisoned data in train set, the backdoor can be successfully implanted across various model scales without degrading performance on benign tasks. Evaluations across multiple jailbreak benchmarks indicate that activating the trigger degrades safety performance by an average of 73\%. Furthermore, the attack generalizes effectively to a wide range of jailbreak methods and unsafe behaviors. Code is available at this https URL. 

---
# CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation 

**Authors**: Aarush Sinha, Arion Das, Soumyadeep Nag, Charan Karnati, Shravani Nag, Chandra Vadhan Raj, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.09746)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding how strategic behavior emerges in multi-agent environments has become an important alignment challenge. We take a neutral empirical stance and construct a controlled environment in which strategic behavior can be directly observed and measured. We introduce a large-scale multi-agent simulation in a simplified model of New York City, where LLM-driven agents interact under opposing incentives. Blue agents aim to reach their destinations efficiently, while Red agents attempt to divert them toward billboard-heavy routes using persuasive language to maximize advertising revenue. Hidden identities make navigation socially mediated, forcing agents to decide when to trust or deceive. We study policy learning through an iterative simulation pipeline that updates agent policies across repeated interaction rounds using Kahneman-Tversky Optimization (KTO). Blue agents are optimized to reduce billboard exposure while preserving navigation efficiency, whereas Red agents adapt to exploit remaining weaknesses. Across iterations, the best Blue policy improves task success from 46.0% to 57.3%, although susceptibility remains high at 70.7%. Later policies exhibit stronger selective cooperation while preserving trajectory efficiency. However, a persistent safety-helpfulness trade-off remains: policies that better resist adversarial steering do not simultaneously maximize task completion. Overall, our results show that LLM agents can exhibit limited strategic behavior, including selective trust and deception, while remaining highly vulnerable to adversarial persuasion. 

---
# ExecTune: Effective Steering of Black-Box LLMs with Guide Models 

**Authors**: Vijay Lingam, Aditya Golatkar, Anwesan Pal, Ben Vo, Narayanan Sadagopan, Alessandro Achille, Jun Huan, Anoop Deoras, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2604.09741)  

**Abstract**: For large language models deployed through black-box APIs, recurring inference costs often exceed one-time training costs. This motivates composed agentic systems that amortize expensive reasoning into reusable intermediate representations. We study a broad class of such systems, termed Guide-Core Policies (GCoP), in which a guide model generates a structured strategy that is executed by a black-box core model. This abstraction subsumes base, supervised, and advisor-style approaches, which differ primarily in how the guide is trained. We formalize GCoP under a cost-sensitive utility objective and show that end-to-end performance is governed by guide-averaged executability: the probability that a strategy generated by the guide can be faithfully executed by the core. Our analysis shows that existing GCoP instantiations often fail to optimize executability under deployment constraints, resulting in brittle strategies and inefficient computation. Motivated by these insights, we propose ExecTune, a principled training recipe that combines teacher-guided acceptance sampling, supervised fine-tuning, and structure-aware reinforcement learning to directly optimize syntactic validity, execution success, and cost efficiency. Across mathematical reasoning and code-generation benchmarks, GCoP with ExecTune improves accuracy by up to 9.2% over prior state-of-the-art baselines while reducing inference cost by up to 22.4%. It enables Claude Haiku 3.5 to outperform Sonnet 3.5 on both math and code tasks, and to come within 1.7% absolute accuracy of Sonnet 4 at 38% lower cost. Beyond efficiency, GCoP also supports modular adaptation by updating the guide without retraining the core. 

---
# Deliberative Alignment is Deep, but Uncertainty Remains: Inference time safety improvement in reasoning via attribution of unsafe behavior to base model 

**Authors**: Pankayaraj Pathmanathan, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09665)  

**Abstract**: While the wide adoption of refusal training in large language models (LLMs) has showcased improvements in model safety, recent works have highlighted shortcomings due to the shallow nature of these alignment methods. To this end, the work on Deliberative alignment proposed distilling reasoning capabilities from stronger reasoning models, thereby instilling deeper safety in LLMs. In this work, we study the impact of deliberative alignment in language models. First, we show that despite being larger in model size and stronger in safety capability, there exists an alignment gap between teacher and student language models, which affects both the safety and general utility of the student model. Furthermore, we show that models aligned through deliberative alignment can retain unsafe behaviors from the base model despite learning the reasoning patterns of larger reasoning models. Building upon this observation, we propose a BoN sampling method that attributes the unsafe behavior back to the base LLMs in the latent space, thereby down-ranking unsafe responses to gain a meaningful improvement in model safety across multiple safety benchmarks with minimal loss in utility. In particular, across 7 teacher models and 6 student models of different classes and sizes, we show an average attack success rate (ASR) reduction of 28.2% in DAN, 31.3% in WildJailbreak and 35.4 % in StrongREJECT benchmarks. We further show that these safety gains prevail post RL training, thus highlighting the uncertainty in safety reasoning and it's explicit attribution to the base model. 

---
# GIANTS: Generative Insight Anticipation from Scientific Literature 

**Authors**: Joy He-Yueya, Anikait Singh, Ge Gao, Michael Y. Li, Sherry Yang, Chelsea Finn, Emma Brunskill, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09793)  

**Abstract**: Scientific breakthroughs often emerge from synthesizing prior ideas into novel contributions. While language models (LMs) show promise in scientific discovery, their ability to perform this targeted, literature-grounded synthesis remains underexplored. We introduce insight anticipation, a generation task in which a model predicts a downstream paper's core insight from its foundational parent papers. To evaluate this capability, we develop GiantsBench, a benchmark of 17k examples across eight scientific domains, where each example consists of a set of parent papers paired with the core insight of a downstream paper. We evaluate models using an LM judge that scores similarity between generated and ground-truth insights, and show that these similarity scores correlate with expert human ratings. Finally, we present GIANTS-4B, an LM trained via reinforcement learning (RL) to optimize insight anticipation using these similarity scores as a proxy reward. Despite its smaller open-source architecture, GIANTS-4B outperforms proprietary baselines and generalizes to unseen domains, achieving a 34% relative improvement in similarity score over gemini-3-pro. Human evaluations further show that GIANTS-4B produces insights that are more conceptually clear than those of the base model. In addition, SciJudge-30B, a third-party model trained to compare research abstracts by likely citation impact, predicts that insights generated by GIANTS-4B are more likely to lead to higher citations, preferring them over the base model in 68% of pairwise comparisons. We release our code, benchmark, and model to support future research in automated scientific discovery. 

---
# A Comparative Theoretical Analysis of Entropy Control Methods in Reinforcement Learning 

**Authors**: Ming Lei, Christophe Baehr  

**Link**: [PDF](https://arxiv.org/pdf/2604.09676)  

**Abstract**: Reinforcement learning (RL) has become a key approach for enhancing reasoning in large language models (LLMs), yet scalable training is often hindered by the rapid collapse of policy entropy, which leads to premature convergence and performance saturation. This paper provides a comparative theoretical analysis of two entropy control strategies: traditional entropy regularization and the recently proposed covariance-based mechanism. We establish a unified framework for entropy dynamics under softmax parameterization, showing that entropy change is governed by the covariance between log-probabilities and logit updates. Our analysis reveals that traditional entropy regularization introduces a dense, persistent bias that modifies the stationary condition, leading to suboptimal policies, while covariance-based methods selectively regularize a sparse subset of high-covariance tokens and achieve asymptotic unbiasedness when the regularization coefficient is annealed. These results provide principled guidelines for entropy control in LLM posttraining, with implications for scaling RL to larger models and more complex reasoning tasks. 

---
# Utilizing and Calibrating Hindsight Process Rewards via Reinforcement with Mutual Information Self-Evaluation 

**Authors**: Jiashu Yao, Heyan Huang, Zeming Liu, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11611)  

**Abstract**: To overcome the sparse reward challenge in reinforcement learning (RL) for agents based on large language models (LLMs), we propose Mutual Information Self-Evaluation (MISE), an RL paradigm that utilizes hindsight generative self-evaluation as dense reward signals while simultaneously calibrating them against the environmental feedbacks. Empirically, MISE enables an agent to learn autonomously from dense internal rewards supplementing sparse extrinsic signals. Theoretically, our work provides the first formal foundation for the paradigm of generative self-rewarding. We prove that utilizing hindsight self-evaluation rewards is equivalent to minimizing an objective that combines mutual information with a KL divergence term between the policy and a proxy reward policy. This theoretical insight then informs and justifies our calibration step, which actively aligns these rewards with the optimal policy. Extensive experiments show that MISE outperforms strong baselines, enabling open-source LLMs about 7B parameters to achieve performance comparable to GPT-4o on validation without expert supervision. 

---
# Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale 

**Authors**: Liujie Zhang, Benzhe Ning, Rui Yang, Xiaoyan Yu, Jiaxing Li, Lumeng Wu, Jia Liu, Minghao Li, Weihang Chen, Weiqi Hu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11554)  

**Abstract**: Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous data flows, operational robustness at scale, and the staleness -- throughput tradeoff. We present \textbf{Relax} (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training engine that addresses these challenges through three co-designed architectural layers. First, an \emph{omni-native architecture} builds multimodal support into the full stack -- from data preprocessing and modality-aware parallelism to inference generation -- rather than retrofitting it onto a text-centric pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled, recovered, and upgraded without global coordination. Third, service-level decoupling enables asynchronous training via the TransferQueue data bus, where a single staleness parameter smoothly interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves a 1.20$\times$ end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode delivers a 1.76$\times$ speedup over colocate on Qwen3-4B and a 2.00$\times$ speedup on Qwen3-Omni-30B, while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay)~\cite{ma2025r3} for MoE models with only 1.9\% overhead, compared to 32\% degradation in veRL under the same configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across image, text, and audio, sustaining over 2{,}000 steps on video without degradation. Relax is available at this https URL. 

---
# Triviality Corrected Endogenous Reward 

**Authors**: Xinda Wang, Zhengxu Hou, Yangshijie Zhang, Bingren Yan, Jialin Liu, Chenzhuo Zhao, Zhibo Yang, Bin-Bin Yang, Feng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11522)  

**Abstract**: Reinforcement learning for open-ended text generation is constrained by the lack of verifiable rewards, necessitating reliance on judge models that require either annotated data or powerful closed-source models. Inspired by recent work on unsupervised reinforcement learning for mathematical reasoning using confidence-based endogenous rewards, we investigate whether this principle can be adapted to open-ended writing tasks. We find that directly applying confidence rewards leads to Triviality Bias: the policy collapses toward high-probability outputs, reducing diversity and meaningful content. We propose TCER (Triviality Corrected Endogenous Reward), which addresses this bias by rewarding the relative information gain between a specialist policy and a generalist reference policy, modulated by a probability-dependent correction mechanism. Across multiple writing benchmarks and model architectures, TCER achieves consistent improvements without external supervision. Furthermore, TCER also transfers effectively to mathematical reasoning, validating the generality of our approach across different generation tasks. 

---
# BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection 

**Authors**: Atharva Gupta, Dhruv Kumar, Yash Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2604.11121)  

**Abstract**: The POLAR SemEval-2026 Shared Task aims to detect online polarization and focuses on the classification and identification of multilingual, multicultural, and multi-event polarization.
Accurate computational detection of online polarization is challenging due to nuanced rhetoric, implicit framing, and the high cost of human-in-the-loop annotation. Building on recent findings that contextual prompting enables large language models to function as strong polarization detectors, we present a two-stage approach for detecting political polarization in social media text that combines structured supervised fine-tuning with Direct Preference Optimization (DPO) refinement.
We fine-tune Qwen 2.5-7B-Instruct with LoRA using an interpretable slot-filling template (target, claim type, manifestation checklist, and justification). We then apply DPO with automatically generated preference pairs to reduce costly false negatives. Experiments on the SemEval 2026 POLAR shared task dataset show that preference-based refinement improves both accuracy and decreases false negatives without extra annotation. On the English development set, DPO increases recall from 0.5085 to 0.7797 and improves macro-F1 by ~5 points. 

---
# HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning 

**Authors**: Yangfan Wang, Tianyang Sun, Chen Tang, Jie Liu, Wei Cai, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11214)  

**Abstract**: Lifelong model editing (LME) aims to sequentially rectify outdated or inaccurate knowledge in deployed LLMs while minimizing side effects on unrelated inputs. However, existing approaches typically apply parameter perturbations to a static and dense set of LLM layers for all editing instances. This practice is counter-intuitive, as we hypothesize that different pieces of knowledge are stored in distinct layers of the model. Neglecting this layer-wise specificity can impede adaptability in integrating new knowledge and result in catastrophic forgetting for both general and previously edited knowledge. To address this, we propose HiEdit, a hierarchical reinforcement learning framework that adaptively identifies the most knowledge-relevant layers for each editing instance. By enabling dynamic, instance-aware layer selection and incorporating an intrinsic reward for sparsity, HiEdit achieves precise, localized updates. Experiments on various LLMs show that HiEdit boosts the performance of the competitive RLEdit by an average of 8.48% with perturbing only half of the layers per edit. Our code is available at: this https URL. 

---
# ProUIE: A Macro-to-Micro Progressive Learning Method for LLM-based Universal Information Extraction 

**Authors**: Wenda Liu, Zhigang Song, Shuai Nie, Guangyao Liu, Lisung Chen, Binyu Yang, Yaran Chen, Peng Zhou, Hongzhen Wang, Yuchen Liu, Wenyue Hu, Jiaming Xu, Runyu Shi, Ying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10633)  

**Abstract**: LLM-based universal information extraction (UIE) methods often rely on additional information beyond the original training data, which increases training complexity yet often yields limited gains. To address this, we propose ProUIE, a Macro-to-Micro progressive learning approach that improves UIE without introducing any external information. ProUIE consists of three stages: (i) macro-level Complete Modeling (CM), which learns NER, RE, and EE along their intrinsic difficulty order on the full training data to build a unified extraction foundation, (ii) meso-level Streamlined Alignment (SA), which operates on sampled data with simplified target formats, streamlining and regularizing structured outputs to make them more concise and controllable, and (iii) micro-level Deep Exploration (DE), which applies GRPO with stepwise fine-grained rewards (SFR) over structural units to guide exploration and improve performance. Experiments on 36 public datasets show that ProUIE consistently improves unified extraction, outperforming strong instruction-tuned baselines on average for NER and RE while using a smaller backbone, and it further demonstrates clear gains in large-scale production-oriented information extraction. 

---
# FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness 

**Authors**: Xiaoning Dong, Chengyan Wu, Yajie Wen, Yu Chen, Yun Xue, Jing Zhang, Wei Xu, Bolei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10189)  

**Abstract**: Large Language Models (LLMs) can generate factually inaccurate content even if they have corresponding knowledge, which critically undermines their reliability. Existing approaches attempt to mitigate this by incorporating uncertainty in QA prompt during training, but these numerical scores lack the semantic richness for LLM to properly understand its internal states of trustworthiness and honestness, leading to insufficient factuality alignment. We introduce FAITH (Factuality Alignment through Integrating Trustworthiness and Honestness), a post-training framework for factuality alignment that integrates natural-language uncertainty signals with external knowledge. Specifically, we augment training datasets by computing confidence scores and semantic entropy from LLM outputs and mapping them into a knowledge state quadrant that describes the model's internal knowledge possession (trustworthiness) and answering behaviors (honestness) in natural language. Based on this enhanced data, we design a reward function that considers both correctness and uncertainty signals, and fine-tune the LLM using the Proximal Policy Optimization (PPO) algorithm. To further mitigate weakly grounded responses, we design a retrieval-augmented module that retrieves relevant external passages, improving the consistency between internal and external knowledge representations. Extensive experiments on four knowledge-intensive benchmarks demonstrate that FAITH enhances the factual accuracy and truthfulness of LLMs. 

---
# ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models 

**Authors**: Chi-Yuan Hsiao, Ke-Han Lu, Yu-Kuan Fu, Guan-Ting Lin, Hsiao-Tsung Hung, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10065)  

**Abstract**: End-to-end full-duplex Speech Language Models (SLMs) require precise turn-taking for natural interaction. However, optimizing temporal dynamics via standard raw-token reinforcement learning (RL) degrades semantic quality, causing severe generative collapse and repetition. We propose ASPIRin, an interactivity-optimized RL framework that explicitly decouples when to speak from what to say. Using Action Space Projection, ASPIRin maps the text vocabulary into a coarse-grained binary state (active speech vs. inactive silence). By applying Group Relative Policy Optimization (GRPO) with rule-based rewards, it balances user interruption and response latency. Empirical evaluations show ASPIRin optimizes interactivity across turn-taking, backchanneling, and pause handling. Crucially, isolating timing from token selection preserves semantic coherence and reduces the portion of duplicate n-grams by over 50% compared to standard GRPO, effectively eliminating degenerative repetition. 

---
# CoSToM:Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models 

**Authors**: Mengfan Li, Xuanhua Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10031)  

**Abstract**: Theory of Mind (ToM), the ability to attribute mental states to others, is a hallmark of social intelligence. While large language models (LLMs) demonstrate promising performance on standard ToM benchmarks, we observe that they often fail to generalize to complex task-specific scenarios, relying heavily on prompt scaffolding to mimic reasoning. The critical misalignment between the internal knowledge and external behavior raises a fundamental question: Do LLMs truly possess intrinsic cognition, and can they externalize this internal knowledge into stable, high-quality behaviors? To answer this, we introduce CoSToM (Causal-oriented Steering for ToM alignment), a framework that transitions from mechanistic interpretation to active intervention. First, we employ causal tracing to map the internal distribution of ToM features, empirically uncovering the internal layers' characteristics in encoding fundamental ToM semantics. Building on this insight, we implement a lightweight alignment framework via targeted activation steering within these ToM-critical layers. Experiments demonstrate that CoSToM significantly enhances human-like social reasoning capabilities and downstream dialogue quality. 

---
# HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation 

**Authors**: Edward Ajayi, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.09629)  

**Abstract**: Humor generation poses a significant challenge for Large Language Models (LLMs), because their standard training objective - predicting the most likely next word - inherently conflicts with the surprise and incongruity needed for comedy. To bridge this gap, we introduce the Cognitive Synergy Framework, a theoretically grounded methodology for generating high-quality humor data inspired by psychological theories of humor. Utilizing a Mixture-of-Thought (MoT) approach, we deploy six cognitive personas (e.g., The Absurdist, The Cynic) to synthesize diverse comedic perspectives for a given prompt. This framework creates a theoretically grounded dataset, which we use to fine-tune a 7B-parameter student model. We compare Direct Preference Optimization (DPO) and a novel Offline Group Relative Policy Optimization (O-GRPO); our 7B model significantly outperforms larger instruction-tuned baselines and achieves performance competitive with state-of-the-art proprietary models. We find that cognitive-driven data curation is far more critical than alignment algorithms or model scale for humor generation. Code and data will be available upon publication. 

---
# Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach 

**Authors**: Haolin Li, Shuyang Jiang, Ruipeng Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11547)  

**Abstract**: While large language models hold promise for complex medical applications, their development is hindered by the scarcity of high-quality reasoning data. To address this issue, existing approaches typically distill chain-of-thought reasoning traces from large proprietary models via supervised fine-tuning, then conduct reinforcement learning (RL). These methods exhibit limited improvement on underrepresented domains like rare diseases while incurring substantial costs from generating complex reasoning chains. To efficiently enhance medical reasoning, we propose MedSSR, a Medical Knowledge-enhanced data Synthesis and Semi-supervised Reinforcement learning framework. Our framework first employs rare disease knowledge to synthesize distribution-controllable reasoning questions. We then utilize the policy model itself to generate high-quality pseudo-labels. This enables a two-stage, intrinsic-to-extrinsic training paradigm: self-supervised RL on the pseudo-labeled synthetic data, followed by supervised RL on the human-annotated real data. MedSSR scales model training efficiently without relying on costly trace distillation. Extensive experiments on Qwen and Llama demonstrate that our method outperforms existing methods across ten medical benchmarks, achieving up to +5.93% gain on rare-disease tasks. Our code is available at this https URL. 

---
# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping 

**Authors**: Yang Liu, Enxi Wang, Yufei Gao, Weixin Zhang, Bo Wang, Zhiyuan Zeng, Yikai Zhang, Yining Zheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11297)  

**Abstract**: Despite the success of reinforcement learning for large language models, a common failure mode is reduced sampling diversity, where the policy repeatedly generates similar erroneous behaviors. Classical entropy regularization encourages randomness under the current policy, but does not explicitly discourage recurrent failure patterns across rollouts. We propose MEDS, a Memory-Enhanced Dynamic reward Shaping framework that incorporates historical behavioral signals into reward design. By storing and leveraging intermediate model representations, we capture features of past rollouts and use density-based clustering to identify frequently recurring error patterns. Rollouts assigned to more prevalent error clusters are penalized more heavily, encouraging broader exploration while reducing repeated mistakes. Across five datasets and three base models, MEDS consistently improves average performance over existing baselines, achieving gains of up to 4.13 pass@1 points and 4.37 pass@128 points. Additional analyses using both LLM-based annotations and quantitative diversity metrics show that MEDS increases behavioral diversity during sampling. 

---
# Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration 

**Authors**: Zhipeng Chen, Tao Qian, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11446)  

**Abstract**: Recently, scaling reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs) has emerged as an effective training paradigm for significantly improving model capabilities, which requires guiding the model to perform extensive exploration and learning, leading to substantial computational overhead and becoming a key challenge. To reduce the number of training steps, Prior work performs linear extrapolation of model parameters. However, the dynamics of model parameter updates during RLVR training remain insufficiently understood. To further investigate the evolution of LLMs during RLVR training, we conduct empirical experiments and find that the rank-1 subspace of the model does not evolve linearly, and its dominance over the original parameters is further amplified during LoRA training. Based on the above insights, we propose the \textbf{N}onlinear \textbf{Ext}rapolation of low-rank trajectories (\textbf{NExt}), a novel framework that models and extrapolates low-rank parameter trajectories in a nonlinear manner. Concretely, we first train the model using LoRA and extract the rank-1 subspace of parameter differences at multiple training steps, which is then used for the subsequent nonlinear extrapolation. Afterward, we utilized the extracted rank-1 subspace to train a predictor, which can model the trajectory of parameter updates during RLVR, and then perform the predict-extend process to extrapolate model parameters, achieving the acceleration of RLVR. To further study and understand NExt, we conduct comprehensive experiments that demonstrate the effectiveness and robustness of the method. Our method reduces computational overhead by approximately 37.5\% while remaining compatible with a wide range of RLVR algorithms and tasks. We release our code in this https URL. 

---
# Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation 

**Authors**: Yanjie He  

**Link**: [PDF](https://arxiv.org/pdf/2604.10511)  

**Abstract**: Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 2,400 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is nearly eliminated on counter-intuitive ones (interaction OR = 0.053, $p < 0.001$); (2) intuitiveness as the dominant factor, explaining more variance than model choice or prompting strategy (ICC = 0.537); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.53$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" may be little more than "slow talking" -- they produce the form of deliberative reasoning without the substance. 

---
