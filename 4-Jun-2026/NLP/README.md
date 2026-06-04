# Streaming Communication in Multi-Agent Reasoning 

**Authors**: Zhen Yang, Xiaogang Xu, Wen Wang, Cong Chen, Xander Xu, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.05158)  

**Abstract**: Multi-agent reasoning systems adopt a "generate-then-transfer" paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, two frontier LLMs (Claude Opus 4.6 and GPT-5.4), and three topologies (Chain, Tree, Graph), StreamMA outperforms both baselines (avg. +7.3 pp, max +22.4 pp on HMMT 2026; Claude Opus 4.6-high). Beyond these contributions, we discover a "step-level scaling law": increasing per-agent steps consistently improves both effectiveness and efficiency, a new scaling dimension orthogonal to and composable with agent-count scaling. 

---
# Activation-Based Active Learning for In-Context Learning: Challenges and Insights 

**Authors**: Yaseen M. Osman, Geoff V. Merrett, Stuart E. Middleton  

**Link**: [PDF](https://arxiv.org/pdf/2606.05134)  

**Abstract**: Deep active learning has previously been explored for LLM in-context sample selection, but not with methods that utilise recent advances in understanding of transformer activations. In this paper, we test the hypothesis that model activations could provide a fine-grained signal to optimise the selection of in-context examples. We present the most comprehensive analysis to date of MLP activation-based deep active learning methods applied to in-context learning, including how different attention masking strategies impact active learning across diverse classification and generative datasets, using both Llama-3.2-3B and Qwen2.5-3B base models. However, we find a negative result: MLP outputs, viewed through the lenses of massive activations or the first four moments, do not correlate with example quality or task performance. Specifically, the absolute Spearman correlation coefficient is at most 0.33 for all tasks and models we tested, showing that such activation-based sampling should not be used for in-context learning. We hypothesise that this may be due to superposition, whereby models represent more features than they have dimensionality, suggesting that methods like Sparse Autoencoders (SAEs) may be a promising future direction. 

---
# Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data 

**Authors**: XiuYu Zhang, Yi Shan, Junfeng Fang, Zhenkai Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05122)  

**Abstract**: Large language models are increasingly evaluated by other models, raising a natural question: can a model predict how a judge will score its own output? We find that the ability is largely present before any targeted training: prompted few-shot, a base model already predicts an external judge's multi-attribute quality scores on open-ended responses well above chance across three benchmarks. We introduce Self-Evaluation Elicitation (SEE), a method that surfaces this latent ability through a short cycle comprising a calibration-coupled reinforcement learning phase that improves the answer and predicts the judge, followed by a masked distillation phase that sharpens the prediction while leaving the answer untouched. From 160 unique examples, roughly 31x fewer than a reinforcement learning baseline, SEE improves held-out calibration across three benchmarks while preserving answer quality. The elicited self-evaluation is sharply localized within the model's own token distribution and stable across judges it was never trained against, indicating a transferable notion of quality rather than a single judge's preference. These results reframe judge-aligned self-evaluation as a problem of elicitation rather than acquisition. 

---
# Evaluating Large Language Models in Dynamic Clinical Decision-Making with Standardized Patient Cases 

**Authors**: Cheng Liang, Pengcheng Qiu, Ya Zhang, Yanfeng Wang, Chaoyi Wu, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.05112)  

**Abstract**: Large language models (LLMs) are increasingly proposed as clinical agents, yet static, single-turn benchmarks cannot capture how a model dynamically delivers care across an encounter: gathering information, planning treatment, and adapting longitudinal management across successive patient states. Medical education has long addressed an analogous challenge through standardized patients (SPs): trained actors who consistently portray clinical cases, enabling realistic practice and objective, scripted assessment. Here we introduce MedSP1000, an SP-derived interactive benchmark for clinical-agent evaluation, including 1,638 SP cases with 24,602 trajectory-level peer-reviewed rubrics. MedSP1000 converts peer-reviewed SP teaching cases into executable scenarios with defined SP case scripts, clinical environment contexts, and human-validated structured rubric. In each simulation evaluation run, a clinical agent interacts in closed loop with a patient agent and an environment controller, and its behaviour is scored throughout the encounter against expert criteria specified in the original materials. Applying MedSP1000 to a range of general-purpose and medically specialized LLMs, we find that performance on static benchmarks does not reliably translate to such educational scenarios. The best-performing model, GPT-5.5, completes only 60.4% of expert-defined rubric items, whereas the strongest medically specialized model reaches 40.0%; increasing test-time compute produces no measurable gain. These results suggest that current LLMs, including agentic systems tuned for medicine, are not yet reliable enough to be safely integrated into actual clinical practice. More broadly, MedSP1000 shows how process-level, SP-style evaluation can reveal clinically relevant failure modes that single-turn benchmarks miss. 

---
# Arithmetic Pedagogy for Language Models 

**Authors**: Andhika Bernard Lumbantobing, Hokky Situngkir  

**Link**: [PDF](https://arxiv.org/pdf/2606.05106)  

**Abstract**: We investigate whether methods of human mathematics pedagogy can guide the training of language models toward arithmetic reasoning. Building on the GASING method -- an Indonesian pedagogy that solves basic arithmetic through a left-to-right procedure aligned with the causal order of token generation -- we operationalize each operation as a computational procedure whose execution trace is serialized into natural-language Chain-of-Thought (CoT) supervision. A small GPT-2 decoder (86M parameters) with a syllabic-agglutinative TOBA tokenizer for Indonesian is trained from scratch on this data using only a next-token prediction objective, without reinforcement learning or reward-based optimization. Monitoring training reveals three distinct learning phases, and mechanistic analyses -- attention-masking interventions on the CoT information graph, residual-stream probing, and logit-lens inspection -- show that the model first internalizes a procedural pathway and subsequently develops an associative, ``mental-arithmetic'' capacity that retrieves intermediate results without explicit step-by-step computation. The trained model reaches over 80% accuracy on held-out problems and attains competitive performance against substantially larger language models, indicating that targeted, pedagogically grounded training can yield strong and economical arithmetic capability at small scale. 

---
# Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models 

**Authors**: Francesca Franzon, Nicolas Rosàs Gómez, Leo Wanner  

**Link**: [PDF](https://arxiv.org/pdf/2606.05087)  

**Abstract**: Frequent English verbs such as 'have' and 'make' can function either as collocates in light-verb constructions or as full lexical predicates, as in 'make a decision' vs. 'make a cake'. Whether language models represent this distinction remains unclear. We introduce a large-scale controlled dataset of minimally varying English sentence series in which the same context contains the same verb in light-verb and full-verb uses. Two probing experiments show that language models differentiate between these uses even in minimal contexts and exhibit separable patterns across object types. We release the dataset, generation code, and materials as a reusable resource. The framework supports extensions to broader contexts, additional verbs, and other languages. 

---
# Automatic Generation of Titles for Research Papers Using Language Models 

**Authors**: Tohida Rehman, Debarshi Kumar Sanyal, Samiran Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2606.05085)  

**Abstract**: The title of a research paper conveys its primary idea and, occasionally, its conclusions in a clear and concise manner. Choosing an appropriate title is often challenging, and automated title generation can assist authors in this task. In this work, we propose a technique to generate paper titles from abstracts using open-weight pre-trained and large language models. We use the CSPubSum and LREC-COLING-2024 datasets and introduce a new dataset, SpringerSSAT, curated from four Springer journals in the social sciences. Additionally, we use GPT-3.5-turbo in a zero-shot setting to generate titles. Model performance is evaluated with ROUGE, METEOR, MoverScore, BERTScore, and SciBERTScore metrics. Our experiments show that fine-tuned PEGASUS-large outperforms other models, including fine-tuned LLaMA-3-8B and zero-shot GPT-3.5-turbo, across most metrics. We further demonstrate that ChatGPT can generate creative paper titles. Overall, AI-generated titles are generally appropriate and reliable. 

---
# Fast & Faithful Function Vectors 

**Authors**: Minh An Pham, Anton Segeler, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin, Patrick Kahardipraja, Reduan Achtibat  

**Link**: [PDF](https://arxiv.org/pdf/2606.05079)  

**Abstract**: Function vectors (FVs) are task representations elicited during in-context learning that can be used to steer Large Language Models (LLMs). However, design choices in their formulation remain underexplored. In this work, we study the impact of varying FV definitions for instructions along two degrees of freedom: attention head selection and steering. For head selection, using gradient-based attributions with Layer-wise Relevance Propagation (LRP) substantially improves efficiency as well as accuracy. For FV steering, applying it in a distributed manner yields a higher accuracy compared to simple aggregation. Our code is publicly available. 

---
# Boosting Self-Consistency with Ranking 

**Authors**: Maria Marina, Daniil Moskovskiy, Sergey Pletenev, Mikhail Salnikov, Alexander Panchenko, Viktor Moskvoretskii  

**Link**: [PDF](https://arxiv.org/pdf/2606.05054)  

**Abstract**: Self-consistency improves large language models by sampling multiple reasoning paths and selecting the most frequent answer, but majority voting often fails to recover correct answers that are already present among the samples. We address this limitation with Ranking-Improved Self-Consistency (RISC), which reformulates answer selection in self-consistency as a ranking problem. Instead of relying on a single uncertainty or confidence signal, RISC uses a lightweight LambdaRank model to score candidate answers with five carefully designed features that capture answer frequency, semantic centrality, and reasoning-trace consistency. We evaluate RISC on three datasets under a range of test-time budgets. Across datasets, RISC consistently achieves a better accuracy-efficiency trade-off than standard self-consistency and strong baselines, with particularly large gains on question answering benchmarks. Further analysis shows that the proposed features are individually useful and, more importantly, complementary, highlighting the value of learning to combine multiple informative signals for test-time answer selection. 

---
# Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2606.05030)  

**Abstract**: Autoregressive chain-of-thought (CoT) reasoning in large language models (LLMs) is fundamentally forward-directed: each step conditions only on prior tokens. This unidirectional inductive bias renders even capable models susceptible to error snowballing, wherein a single logical or arithmetic mistake in an early step irreversibly corrupts the entire reasoning chain. We introduce Teleological Reasoning Infilling (\TRI{}), a training framework that endows decoder-only transformers with a native \emph{goal-conditioned bridging} capability. The key insight is to reframe erroneous reasoning segments as fill-in-the-middle (FIM) tasks: given a verified prefix premise $P$, a verified downstream milestone $S$, and the original query $Q$, the model must synthesise the logical bridge $M$ that connects $P$ to $S$ rigorously and completely. To achieve this with standard causal architectures, we introduce a Prefix-Suffix-Middle (PSM) sequence rearrangement with three non-overlapping sentinel tokens, enabling $M$ to attend to both $P$ and $S$ without any structural modification to the self-attention mechanism. Training proceeds in two stages: (i) Supervised Fine-Tuning (SFT) on symbolically verified $(P, S, M)$ triples extracted from formal mathematics corpora, and (ii) Direct Preference Optimisation (DPO) with a deterministic symbolic verifier (Lean 4 / Python) as the sole reward oracle, eliminating LLM-judge sycophancy. At inference, TRI operates as a surgical repair module within a dual-system loop: a causal draft model generates an initial trace, the verifier pinpoints failures, and TRI infills only the damaged segment, leaving verified sections intact. Comprehensive experiments on three benchmarks demonstrate that TRI achieves state-of-the-art performance across all tasks, while reducing per-problem token expenditure by 31.2%. 

---
# TaDA: Calibrated Probe Gating for Task-Domain LoRA Merging 

**Authors**: Huy Quoc To, Fuyi Li, Guangyan Huang, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05016)  

**Abstract**: Combining a task LoRA adapter with a domain LoRA adapter into a single unified model is a practical yet largely unexplored challenge. Existing methods treat both adapters as symmetric peers, applying uniform weights across all layers. We argue that task and domain adapters exhibit a consistent depth-dependent asymmetry across transformer architectures. Domain dominance increases with layer depth, while shallower layers retain stronger task-relevant signals. Motivated by this observation, we propose $\textbf{TaDA}$ ($\textbf{Ta}$sk-$\textbf{D}$omain LoR$\textbf{A}$ Merging), a training-free algorithm that exploits this structure through calibrated probe-guided per-layer gating and per-component subspace-aware merging. The gating assigns individual weights per layer and projection type using a probe signal proved invariant to adapter weight magnitude. The merging discards conflicting singular directions before combining the remaining components. $\textbf{TaDA}$ produces a standard rank-$r$ LoRA adapter with zero inference overhead. On six scientific QA benchmarks with Llama-2-7B, TaDA achieves an average accuracy of 0.452, outperforming DARE-TIES by +3.6 percentage points and obtaining the best result on all six benchmarks. On six image classification benchmarks with ViT-L/16, TaDA reaches 85.9\% average accuracy, improving over the strongest merging baseline while leading in three of the six individual benchmarks. 

---
# Depth-Attention: Cross-Layer Value Mixing for Language Models 

**Authors**: Boyi Zeng, Yiqin Hao, Zitong Wang, Shixiang Song, He Li, Feichen Song, Yifan Liu, Ziwei He, Xinbing Wang, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.05014)  

**Abstract**: Self-attention selects information freely across the sequence, but across depth, Transformers merely add each layer's output to the residual stream, so later layers cannot selectively reuse earlier-layer representations. Recent cross-layer methods improve this flow but operate on hidden states outside attention, adding state beyond the key-value cache at inference--a cost that becomes increasingly salient as modern LLMs compress the cache with grouped-query and multi-head latent attention. We introduce Depth-Attention, which performs this selection inside the attention module itself: before a layer attends over the sequence, its query attends over the keys of earlier layers at the same token position and mixes their values into the value that self-attention then reads. Because Depth-Attention reuses the standard attention queries, keys, and value-cache slots, storing depth-mixed values in place of the original values, it adds no parameters and introduces no persistent inference state beyond the standard key-value cache--the same cache size as a vanilla decoder and less than hidden-state-based cross-layer methods. On Qwen3-style decoders at 1.5B and 3B parameters, Depth-Attention attains the lowest perplexity and the highest average downstream accuracy, improving over the vanilla Transformer by up to 2.3 accuracy points and surpassing strong cross-layer baselines in perplexity and average accuracy, while adding under 0.01% extra arithmetic FLOPs and no additional persistent inference state. The gains hold from 360M to 3B parameters and extend to looped Transformers. 

---
# DAR: Deontic Reasoning with Agentic Harnesses 

**Authors**: Guangyao Dou, William Jurayj, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2606.05009)  

**Abstract**: Deontic reasoning is the task of answering questions by applying explicit rules and policies to case-specific facts, for example computing tax liability under a statute or determining the outcome of an immigration appeal. A key technical challenge for LLM-based deontic reasoning is that the relevant ruleset can be long and cross-referenced, so models may still fail to locate the rules needed for a particular reasoning step. We introduce Deontic Agentic Reasoning (DAR), an agentic reasoning setup in which the model interacts with the statutes on demand. We evaluate DAR under multiple harnesses on hard subsets of DeonticBench. Across these settings, we find that agentic harnesses can push the frontier on deontic reasoning tasks, but improvements are not uniform: weaker models often degrade on numerical tasks while consuming far more tokens. 

---
# GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation 

**Authors**: Yuxiao Ye, Yiwen Zhang, Huiyuan Xie, Yuqin Huang, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05002)  

**Abstract**: LLM-based multi-agent systems are increasingly used for strategic decision-making tasks. In such settings, performance depends not only on individual model capabilities, but also on the policies by which agents interact and adapt. Multi-agent reinforcement learning can optimise these interaction policies, but its reward design often remains task-specific and weakly grounded in interaction structure. To address this gap, we propose GARL, a GAme-theoretic Reinforcement Learning framework for multi-agent strategic prioritisation. GARL formalises strategic prioritisation as a two-stage game: competing agents first allocate strategic resources over a shared candidate set, and a higher-level arbiter then produces the final ranking. The resulting game-theoretic utilities are converted into role-specific reinforcement signals, allowing policy optimisation to be guided by structured interaction. We instantiate GARL on issues-in-dispute ranking, where the goal is to prioritise core issues in legal proceedings. Experiments show that GARL improves ranking performance, enables small open-source LLMs to become competitive with a strong closed-source LLM under the same candidate-ranking setting, and yields gains in legal-domain competence and broader strategic decision-making. Overall, GARL demonstrates how game-theoretic interaction structure can be turned into reinforcement-learning objectives, providing a principled approach to policy optimisation in multi-agent strategic prioritisation. 

---
# DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving 

**Authors**: Xiaochen Zhu, Georgi Karadzhov, Tom Stafford, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2606.04987)  

**Abstract**: Multi-party dialogue is a critical setting for studying collaborative reasoning and decision-making, yet existing datasets rarely focus on structured, in-depth complex reasoning tasks. We introduce DeliChess, a novel dataset of group deliberation dialogues in which participants collaboratively solve multiple-choice chess puzzles. Each group first completes the puzzle individually, then engages in a multi-party discussion before submitting a revised collective answer. The dataset includes 107 dialogues with full transcripts, pre- and post-discussion choices, and metadata on puzzle difficulty and move quality. We evaluate performance using three metrics based on chess engine evaluations, and find that deliberation significantly improves group accuracy. We further analyse the role of probing utterances (i.e., messages that elicit proposals, justifications, or strategic reflection) using a classifier trained on prior deliberation data. While probing makes group performance more variable after discussion, it does not consistently lead to better performance. Our dataset offers a rich testbed for modelling group reasoning, dialogue dynamics, and the resolution of differing perspectives and opinions in a well-defined strategic domain. 

---
# Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions: Evidence from the St. Petersburg Game 

**Authors**: Chensong Huang, Changyu Chen, Chenwei Lin, Hanjia Lyu, Xian Xu, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04978)  

**Abstract**: LLMs can appear cautious in risk decision-making tasks, yet cautious-looking outputs do not necessarily indicate alignment with human decision-making mechanisms. We investigate this distinction using the St. Petersburg game as a controlled testbed, a classical paradox in which the expected payoff is infinite, yet humans typically report low, finite willingness to pay. We evaluate 28 LLMs with a structured prompt suite that includes the original game; controlled decision variants that perturb truncation, repeated play, numeric endowment, and occupational identity; a human-perspective prompt that asks models to reason as human decision makers; and paired comparisons between base models and their instruction-tuned counterparts. In the original game, most models generate finite bids, creating the appearance of human-like risk behavior. However, this outcome-level resemblance masks substantial mechanism-level differences. The controlled variants reveal that rather than maintaining human-like behavior seen in the original game, models often shift to conditionally and computationally rational behavior. Human-cue prompting and instruction tuning often lower bids and reduce some visible pathologies, but most mechanism-level response patterns remain largely unchanged. These findings show that behavioral alignment in risk decision-making can be surface-level: LLMs may produce human-like risk decisions without exhibiting human-consistent mechanisms. High-stakes evaluations of LLM decision-making should therefore move beyond outcome similarity and examine whether the alignment is supported by mechanism-level consistency. 

---
# SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding 

**Authors**: Na Li, Chengda Wang, Mingju Gao, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04974)  

**Abstract**: Diffusion large language models (DLLMs) enable non-autoregressive generation by iteratively denoising corrupted token sequences with bidirectional context. Despite their ability to update multiple positions in parallel, inference remains costly due to the many denoising steps required for high-quality generation. We propose SAID, a Scaffold-Aware Iterative Decoding framework that accelerates DLLMs by reallocating computation across tokens. SAID first spends denoising computation on scaffold tokens to establish the coarse semantic structure, and then completes predictable detail tokens with fewer steps. We further adapt SAID to block-wise diffusion decoding and introduce Confidence-Hierarchical Layered Generation (CHLG), which assigns additional steps only to low-confidence tokens. Experiments on LLaDA-8B and LLaDA 1.5 across math, coding, and knowledge benchmarks show that SAID significantly accelerates DLLM inference with a maximum speedup of 9.1x while maintaining competitive performance. Our code is publicly available: this https URL. 

---
# SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs 

**Authors**: Xinrui Song, Zhuoran Wang, Mingju Gao, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04964)  

**Abstract**: Diffusion language models (DLMs) generate text through iterative denoising, and blockwise decoding improves their practicality by committing tokens in local blocks. However, existing blockwise methods typically rely on fixed block sizes or delimiter-based runtime signals, which do not necessarily align with semantic boundaries. In this paper, we propose SemBlock, a semantic-boundary-driven dynamic block decoding framework for diffusion LLMs. SemBlock formulates dynamic block construction as semantic boundary prediction and trains lightweight predictors on frozen LLaDA hidden states. To provide supervision, we construct SemBound, a semantic-boundary dataset that derives boundary labels from discourse units, reasoning steps, and implementation spans across natural language, math, and code tasks. During inference, SemBlock uses predicted boundary probabilities to select the ending position of each dynamic block. Experiments on GSM8K, IFEval, MATH, and HumanEval show that SemBlock consistently improves over fixed-block decoding and AdaBlock. Our code is publicly available: this https URL. 

---
# Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection 

**Authors**: Aswathy Velutharambath, Neele Falk, Sofie Labat, Tarun Tater, Amelie Wuehrl  

**Link**: [PDF](https://arxiv.org/pdf/2606.04924)  

**Abstract**: The widespread use of Large Language Models (LLMs) as writing tools challenges the validity of crowdsourced data, as crowdworkers may outsource tasks to models. To better understand how this is addressed, we surveyed 155 researchers in NLP and related disciplines about their experiences and opinions on collecting free-text responses via crowdsourcing. This paper provides an overview of practitioners' challenges, mitigation strategies, and the foreseen implications on data quality. 44% of respondents reported observing LLM usage in their crowdsourced data. While 93% of them had anticipated this, half were unsure what precautions to take. The most prevalent detection strategies are distinctive textual style patterns and unusually fast completion times. Overall, survey responses show that the research community is aware of the problem and taking measures, but existing efforts remain insufficient to fully address it. Finally, we derive a set of considerations to guide future crowdsourced free-text data collection in the era of LLMs. 

---
# Caliper: Probing Lexical Anchors versus Causal Structure in LLMs 

**Authors**: Zhenyu Yu, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04915)  

**Abstract**: Large language models reach 50 to 70% accuracy on causal reasoning benchmarks such as CLadder, but it is unclear whether this reflects structural reasoning or lexical pattern matching. We introduce Caliper, a controlled perturbation that replaces semantic variable names with placeholder tokens while preserving the causal graph and probabilistic specification of each question. Across nine instruction-tuned LLMs from 3.8B to 671B and three causal reasoning benchmarks, lexical anonymization yields robust accuracy drops of +7.6, +27.0, and +11.1 pp on a local 3.8B-14B set, rising to +29.6 and +18.0 pp on CRASS and e-CARE across nine frontier models spanning the 2024-2026 generations. Of 40 engaged model-by-benchmark cells, 39 show a positive gap, and the gap collapses by 17x on CLadder's pseudoword subset. Structured scaffolding and few-shot in-context learning each narrow the gap, but mainly by lowering P0 accuracy on smaller models rather than recovering P1. Current instruction-tuned LLMs, evaluated zero-shot, show little evidence of structural causal reasoning once lexical anchors are removed. 

---
# 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions 

**Authors**: Nils Dycke, Marina Sakharova, Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.04906)  

**Abstract**: Although it is generally agreed that AI-generated text poses a broad societal risk, there is no common understanding in the AI-generated text detection literature on what constitutes harmful use. Rather, existing datasets and approaches often define their own criteria and make their own assumptions, sometimes implicitly, and often only loosely related to real-world needs and applications. To address this gap, we here systematically define various notions of AI-generated text and their characteristics. To study these, we collect AITDNA - a new benchmark of human-machine co-constructed texts that is annotated with detailed genesis information, such as the entire edit and AI-interaction history. We benchmark various machine-generated text detectors and find that they often only perform well for specific notions but not as broad detectors. We release code and data publicly. 

---
# GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards 

**Authors**: Tej Deep Pala, Vernon Toh, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2606.04889)  

**Abstract**: Reinforcement learning with verifiable rewards (e.g. GRPO) is now a common way to improve mathematical reasoning in Large Language Models (LLMs). However, current methods usually broadcast one sequence-level advantage to all tokens, or use costly process reward models (PRMs) for step-level supervision. Uniform advantage distribution assumes that all tokens contribute equally to the final reward. This dilutes the gradient signal, since flawed reasoning steps and filler words are updated as strongly as valid logical inferences. To address this, we introduce Gradient-Reweighted Advantage (GRAIL), an intrinsic token-wise advantage reweighting method. GRAIL uses gradient-activation saliency to place more weight on tokens that are more locally sensitive to the final answer. Evaluations across five models from the Qwen3, R1-distilled and OctoThinker families show that GRAIL consistently outperforms GRPO. GRAIL achieved an average improvement of 3.60% in accuracy and 3.05% in Pass@3, demonstrating that fine-grained reasoning alignment can be achieved without process-level supervision. 

---
# Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean 

**Authors**: Kári Rögnvaldsson, Chenhao Sun, Jasper Dekoninck, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2606.04883)  

**Abstract**: Large language models (LLMs) are increasingly used in workflows for generating formal proofs in Lean. These workflows often decompose problems into smaller lemmas, sample many proof attempts, and use compiler feedback to guide search. However, they can be prohibitively expensive, often spending substantial compute on attempts that ultimately fail. In this work, we address this problem with an action routing agent that consists of a data plane and a control plane. The data plane generates natural-language lemma decompositions, formalizes them in Lean, and samples proof attempts for the resulting theorem and lemma targets. The control plane observes previous failed Lean attempts, estimates both the likelihood of success and cost of another attempt, and decides whether to continue proving the current target or restart from a new breakdown. On a subset of PutnamBench, our agent decreases the cost by $25.8\%$ over a fixed-step baseline on average, preserving performance while using substantially less compute. These results suggest that failed Lean trajectories provide actionable signals for cost-aware resource allocation in agentic theorem proving. 

---
# Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents 

**Authors**: Haoyu Sun, Wenxuan Wang, Mingyang Song, Jujie He, Weinan Zhang, Yang Liu, Yang Yang, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.04874)  

**Abstract**: Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce \textbf{Agent Planning Benchmark (APB)}, a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $\tau^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics across three representative models. APB thus serves as an upstream diagnostic complement to execution benchmarks. 

---
# Large Language Models in K-12 Education: Alignment with State Curriculum Standards and Student Personas 

**Authors**: Lisa Korver, Tomo Lazovich, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2606.04846)  

**Abstract**: As Large Language Models (LLMs) become increasingly popular in educational settings, they raise important questions about the ethical implications of their use. Publicly available online chatbots are quickly improving in capability and accuracy leading to more widespread use, including among students looking for help with their homework. This makes it crucial to consider whether these models are aligned with educational standards. Because curriculum standards in the United States are set at the state level, they differ significantly in required content, emphasis, and narrative focus. In this work, we develop an LLM-based pipeline to identify variations in U.S. History curricula across states and evaluate the extent to which different LLMs reflect these state-specific curricular differences. In addition, we conduct controlled experiments that vary user personas by stating user attributes such as geographic location, grade level, gender and race to evaluate the sensitivity of LLM responses to user characteristics. We find that while models are able to adjust their presentation of historical topics, these shifts may come from the perceived political leanings of states and do not necessarily reflect actual curriculum content. Additionally, models successfully adapt to a student's grade level while showing minimal sensitivity to race or gender, suggesting they are capable of useful adaptation to student personas with limited demographic bias. Together, these findings highlight potential risks that open access to LLM chatbots may cause to student learning outcomes stemming from misalignment with state curriculum standards and highlight the need for more robust alignment techniques. 

---
# A French Corpus Annotated for Multiword Expressions with Adverbial Function 

**Authors**: Eric Laporte, Takuya Nakamura, Stavroula Voyatzi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04828)  

**Abstract**: This paper presents a French corpus annotated for multiword expressions (MWEs) with adverbial function. This corpus is designed for investigation on information retrieval and extraction, as well as on deep and shallow syntactic parsing. We delimit which kind of MWEs we annotated, we describe the resources and methods we used for the annotation, and we briefly comment the results. The annotated corpus is available at this http URL under the LGPLLR license. 

---
# PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents 

**Authors**: Yubo Hou, Jingwei Song, Hongbo Zhang, Zhisheng Chen, Bang Xiao, Tao Wan, Zengchang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.04780)  

**Abstract**: Persistent LLM agents require memory representations that make the formation of person understanding explicit across long term interaction. Existing agent memory methods emphasize information retention and retrieval, yet give limited account of how accumulated interaction evidence is abstracted into person understanding. We view this process as schema formation, where situated evidence is abstracted into reusable patterns and stable person level claims. We introduce PersonaTree, a structured lifecycle memory framework that realizes this view as a three level persona tree with explicit support paths from evidence to claims. PersonaTree maintains the tree through conservative writing, confidence guided consolidation, and query conditioned path retrieval, returning only the evidence depth required by each query. Across six person understanding and persistent memory benchmarks with three answer backbones, PersonaTree ranks first in 12 of 18 compact scores and reaches the top two in 16 settings. Ablations show that hierarchy improves abstract person understanding on KnowMe, while support path retrieval improves RealPref alignment under a comparable context budget. 

---
# TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration 

**Authors**: Soyeong Jeong, Jinheon Baek, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04743)  

**Abstract**: Agents are widely deployed as assistants over documents, tools, and code. However, they typically act only on explicit user requests, which surface only the problems the user has noticed, while many other important problems coexist, hidden in plain sight, within the broader user context, with their total number unknown in advance. We frame this as the task of discovering multiple hidden problems from context, in which coexisting problems should be uncovered, grounded in supporting evidence, and paired with concrete actions. To this end, we introduce TIDE, a template-guided iterative framework with two complementary mechanisms. Specifically, motivated by the observation that single-pass prediction anchors on the most salient cases and yields generic claims, we propose iterative discovery, which surfaces a small batch of candidates per round while conditioning on what has already been found, so subsequent rounds extend coverage; and thought templates, reusable schemas distilled from previously solved cases that specify what contextual signals to attend to and how to connect them, anchoring each prediction in a recognizable problem class. We validate TIDE on two realistic settings, personal workspaces and software repositories, across four model backbones, showing substantial gains over single-shot and parallel multi-agent baselines on task coverage, identification, and resolution. 

---
# Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026 

**Authors**: Enes Yavuz Ugan, Maike Züfle, Yuka Ko, Supriti Sinhamahapatra, Fabian Retkowski, Seymanur Akti, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2606.04730)  

**Abstract**: With the advent of Large Language Models, single-task and token-based multi-task models have evolved into instruction-based systems that infer task and target language implicitly from natural language prompts. This trend is reflected in IWSLT's Instruction Following Track, which this year introduced new tasks including an unknown surprise task, posing a genuine challenge against overfitting to known tasks. We present KIT's submission to the Long and Short Instruction Following tracks in the unconstrained setting. Our approach combines a general data augmentation pipeline that converts short-form corpora into long-form training data through segment concatenation, LLM-based label generation, and cross-lingual translation, yielding over 1M instances across six tasks and four languages. We further show that likelihood-based re-ranking, while highly effective for ASR, systematically degrades semantic tasks by spuriously selecting candidates generated from segmented audio processing rather than holistic long-form inference, a failure mode resolved by combining likelihood with Minimum Bayes Risk decoding. 

---
# Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM 

**Authors**: SooHwan Eom, Jay Shim, Gwanhyeong Koo, Haebin Na, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04719)  

**Abstract**: The Transformer's quadratic complexity with input length imposes an unsustainable computational load on large language models (LLMs). In contrast, the Selective Scan Structured State-Space Model, or Mamba, addresses this computational challenge effectively. This paper explores a query-based cross-modal projector designed to bolster Mamba's efficiency for vision-language modeling by compressing visual tokens based on input through the cross-attention mechanism. This innovative projector also removes the need for manually designing the 2D scan order of original image features when converting them into an input sequence for Mamba LLM. Experimental results across various vision-language understanding benchmarks show that the proposed cross-modal projector enhances Mamba-based multimodal LLMs, boosting both performance and throughput. 

---
# Rethinking Continual Experience Internalization for Self-Evolving LLM Agents 

**Authors**: Jingwen Chen, Wenkai Yang, Shengda Fan, Wenbo Nie, Chenxing Sun, Shaodong Zheng, Yangen Hu, Lu Pan, Ke Zeng, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.04703)  

**Abstract**: Experience internalization converts contextual experience from past interactions into reusable parametric capability, offering a promising path toward continual learning in large language models (LLMs). While prior work has predominantly focused on single-iteration transfer, we discover that under multi-iteration experience learning, existing methods suffer from a progressive capability collapse rather than compounding improvement. We systematically examine this failure through three vital dimensions of experience internalization: (1) Experience Granularity: We find that principle-level experience is more durable than instance-level experience, as it effectively abstracts transferable strategies away from trajectory-specific details. (2) Experience Injection Pattern: Our analysis reveals that step-wise injection significantly outperforms global injection by aligning experience with intermediate decision states, a property that is critical for long-horizon tool use. (3) Internalization Regime: We demonstrate that off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy context-distillation, which is inherently limited by local corrections on student-induced flawed states. Together, these insights yield a simple yet robust recipe for stable and sustainable experience internalization, providing concrete guidance for engineering self-evolving and continually learning LLMs. 

---
# DuDi: Dual-Signal Distillation with Cross-Lingual Verbalizer 

**Authors**: Patomporn Payoungkhamdee, Tinnakit Udsa, Jian Gang Ngui, Sarana Nutanong, Alham Fikri Aji, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2606.04694)  

**Abstract**: Small language models (SLMs) are efficient and scalable, but their multilingual capabilities degrade severely at sub-billion scales, especially for Southeast Asian (SEA) languages. We introduce DuDi, a dual-signal multilingual distillation framework that combines an online sequence-level signal with off-policy and on-policy token-level signals. DuDi further uses a cross-lingual verbalizer to refine teacher feedback and improve teacher-student transferability in multilingual settings. Experiments on SEA-HELM across multiple model families, scales, and teacher-student settings show that DuDi consistently outperforms competitive distillation baselines. Ablations and analyses confirm that sequence-level optimization, token-level supervision, and cross-lingual verbalization provide complementary and transferable learning signals for multilingual SLMs. 

---
# SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction 

**Authors**: Kenfeng Huang, Yi Cai, Xin Wu, Zikun Deng, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.04691)  

**Abstract**: Zero-shot information extraction (IE) with large language models (LLMs) has attracted increasing attention due to its flexibility in adapting to new schemas and domains without task-specific training. Existing approaches mainly rely on monolithic prompting, each-type prompting, or multi-agent debate. However, monolithic prompting often suffers from boundary and type errors, while each-type prompting and multi-agent debate introduce cross-type conflicts, redundant agent interactions, and substantial token overhead. To address these challenges, we propose SMADE-IE, a sparse and evidence-driven multi-agent framework for zero-shot IE. SMADE-IE first employs an Adaptive Mode Selector to dynamically route inputs into either a lightweight Global Extraction Mode or a Type-Centric Extraction Mode, reducing unnecessary type selection and reasoning noise. For conflicting predictions, we further introduce an Evidence-Driven Debate mechanism that structures arguments into Toulmin-style components and performs confidence aggregation through external evidence scoring and Bayesian updates. Experimental results on 9 benchmark datasets across NER, RE, and JERE tasks show that SMADE-IE consistently outperforms existing zero-shot IE baselines while also improving token efficiency through sparse agent selection and early-stopping debate. 

---
# CRAFT: Cost-aware Refinement And Front-aware Tuning of Prompts 

**Authors**: Shanu Kumar, Shubhanshu Khandelwal, Akhila Yesantarao Venkata, Parag Agrawal, Yova Kementchedjhieva, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.04661)  

**Abstract**: Prompts tuned for accuracy often grow long, raising inference cost on every model call. The best accuracy-cost trade-off depends on the task and the budget, so prompt optimization is a search over the Pareto front of accuracy and prompt-token cost rather than for one prompt. The usual shortcut, collapsing the objectives into a weighted sum, fixes the trade-off weight before search and often recovers only a narrow region of the front, a failure we call scalarization collapse. We present CRAFT (Cost-aware Refinement And Front-aware Tuning), a Pareto-front prompt optimizer that treats target-LLM validation calls as the scarce resource and allocates them to candidates near the optimistic candidate front. Each round, complementary accuracy-oriented and cost-oriented generators propose edits, Pareto-gap acquisition spends the per-round validation budget, and NSGA-II retention keeps a spread-out population. Across six classification and reasoning benchmarks, CRAFT's retained fronts reach both high-accuracy and low-cost regions, while accuracy-only, cost-only, and weighted-sum baselines each concentrate in narrower regions. The accuracy-cost trade-off becomes a post-search choice, not a pre-search weight. 

---
# LifeSide: Benchmarking Agents as Lifelong Digital Companions 

**Authors**: Yuqian Wu, Zhijie Deng, Wei Chen, Junwei Li, Yutian Jiang, Junle Chen, Zhengjun Huang, Qingxiang Liu, Jing Tang, Jiaheng Wei, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04660)  

**Abstract**: Lifelong digital companions must integrate cross-session cues, continually update their understanding of users, and adapt to shifting privacy boundaries. Existing evaluations fail to capture this, testing memory recall and short-term empathy in isolation. To bridge this gap, we introduce \benchmark, a benchmark centered on multi-session \textit{Memory-Emotion-Environment} loops. By modeling users as persistent worlds with layered profiles and event trajectories, \benchmark uses multi-agent simulation to project environmental dynamics into dialogue, preserving the critical gap between latent thoughts and observable expressions. Evaluating 2,000 personas and 111K tasks across memory tracking, user understanding, privacy control, and emotional companionship, our experiment results reveal a stark reality: even models that saturate current memory benchmarks fail to sustain accurate user understanding and true companionship over long horizons. 

---
# QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples 

**Authors**: Mengao Zhang, Xiang Yang, Chang Liu, Tianhui Tan, Ke-wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04646)  

**Abstract**: Many real-world questions over business, legal, and scientific corpora are natural-language versions of database-style queries over records latent in text. Existing retrieval-augmented generation (RAG) systems are optimized primarily for semantic relevance, but retrieving plausible passages does not guarantee correct query execution. We introduce QO-Bench, a diagnostic benchmark for query-operator question answering over typed event tuples. The benchmark covers 22,984 news articles and 614 corporate events across 18 query templates, evaluated on 785 questions. Each gold answer is deterministically computed from typed event tuples and scored by recall, with answers matched to the gold tuples by exact match rather than an LLM judge. This design enables operator-level diagnosis such as joins and intersection. We evaluate RAG, ReAct RAG, GraphRAG, and information-extraction-to-SQL under matched conditions, with a long-context oracle ceiling to isolate retrieval failure. A two-axis framework -- index-time preservation versus query-time execution -- predicts where each paradigm fails, and the results bear it out: systems retrieve relevant text but discard the typed values operators need, and the deployable paradigm ranking inverts across operators, with similarity retrieval leading on filter/project and extraction-to-SQL on intersection and counting. Even given the gold evidence, a long-context oracle stays far from saturated, so operator execution -- not retrieval alone -- is a core bottleneck that a stronger answer model does not remove. QO-Bench reframes the goal from passage relevance to query-operator-preserving retrieval. 

---
# CYGNET: Cypher Gate for Neural Execution Triage and Cost Containment 

**Authors**: Nikodem Tomczak  

**Link**: [PDF](https://arxiv.org/pdf/2606.04645)  

**Abstract**: Language models acting as agents over knowledge graphs generate Cypher queries that fail structurally (crashing at the database) or semantically (executing but returning wrong results). We place a pre-execution gate between query generation and a production Neo4j database. The gate validates structure through a four-backend chain culminating in execution against a mirror graph at 5.6 ms median latency. Structurally broken queries are routed to a corrector that iterates structured error feedback through a language model. On seven CypherBench schemas (2348 questions, ACL 2025) the pipeline maintains generation accuracy on every model tested, confirming it operates as a safe defensive layer. The corrector achieves 81% to 95% success across five models (mean 89%). On a template-generated corpus across nine schemas the gate catches 100% of parse errors, 100% of constraint violations, and 100% of schema-reference errors in path queries with labelled endpoints, at zero false positives across 1135 queries. Property sibling-swaps where the substituted name is valid on the target label score 0%, marking the formal boundary where structural validation ends and semantic validation must begin. A planner-based cost gate flags catastrophic plan structures before execution. 

---
# RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation 

**Authors**: Nikodem Tomczak  

**Link**: [PDF](https://arxiv.org/pdf/2606.04628)  

**Abstract**: RAMPART is a compile-time memory model and pure in-RAM block registry for LLM-based agents. Context assembly is a programmable runtime operation where content is compiled from a structured registry under explicit policy for ordering, inclusion, and eviction. Five composable primitives (promote, gate, write, evict, rollback) act on named addressable blocks before compilation at zero prompt-token cost. Provenance tags and non-evictable authorship flags implement a permissioned memory model with block-level ownership. Controlled probes with Qwen3-8B Q4 show that compile-time placement and the structural relationship between blocks and the task query affect task success, with the cliff falling at roughly the seventh block position when the task follows the registry and the twelfth when it precedes. Grouping the critical block with content-adjacent neighbours and promoting the group as a unit lifts task success by tens of percentage points at positions where single-block placement fails. Cross-model replication on Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-14B shows the content-priming effect appears at the same absolute positions across families, with magnitude varying with model strength. Block grouping raises Mistral's mean pass rate roughly fivefold at the hardest registry size, and a smaller model with the intervention can outperform a larger model without it in the mid-registry zone. Relevance gating reduces prompt cost by 67.8\% while recovering 83% of the promoted-condition success rate. Schema eviction produces 0% invocations against 100% with the schema present, a property policy-based approaches cannot guarantee by construction. Shared-registry coordination reduces inter-agent communication to a method call at zero coordination token cost. 

---
# Hybrid Adversarial Defence for Natural Language Understanding Tasks 

**Authors**: Manar Abouzaid, Yang Wang, Chenghua Lin, Stuart E. Middleton  

**Link**: [PDF](https://arxiv.org/pdf/2606.04612)  

**Abstract**: Large Language Models (LLMs) are vulnerable both to hallucination and adversarial manipulation. Although these problems are closely related, existing defences typically address them separately. We investigate a hybrid defence framework that combines entropy-based models, designed to reduce hallucinations, with uncertainty-based models and geometric-based models, designed to reduce vulnerability. Under in-domain tests on Natural Language Understanding datasets (FEVER, HotpotQA, CSQA, SIQA) we find our hybrid model improves both clean-task performance (up to 43.34\% increase in accuracy) and adversarial robustness (up to 64.92\% improvement in accuracy and 62.27\% reduction in attack success rate). For out-of-distribution datasets (AeroEngQA, CPIQA) we see similar adversarial robustness from our hybrid model (up to 57.14\% improvement in accuracy). For prompt injection (SafeGuard) and jailbreak detection (AdvBench, DAN) datasets our hybrid model is also very strong (up to 51\% reduction in attack success rate compared to state of the art baseline models). Overall, our results show that combining entropy, uncertainty and geometric features provides a more effective defence strategy than using any single feature alone for both in-domain and out-of-distribution tasks. 

---
# A Systematic Evaluation of Positional Bias in Multi-Video Summarization with MLLMs 

**Authors**: Huangchen Xu, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04596)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for video understanding, yet their reliability under multi-video inputs remains poorly understood. We study positional bias in multi-video summarization, where the quality of a per-video summary can change with the video's input slot even when the underlying content is unchanged. We construct a benchmark from ActivityNet and News videos, covering Cooking, Domestic, Leisure, and News settings with two- and four-video inputs. We evaluate nine open-source and proprietary MLLMs and measure position effects with three complementary metrics: Coverage, Directional Positional Bias (DPB), and Middle-Edge Gap (MEG). Our results show that positional effects are domain- and model-dependent: signed directional bias can be small even when middle positions underperform, and increasing visual or generation budget does not uniformly remove the imbalance. We further analyze prompt-level mitigation methods. Together, the results show that multi-video summarization remains sensitive to input protocol and position, motivating more robust order-invariant multimodal systems. 

---
# Fine-grained Fragment Retrieval in Multi-modal Long-form Dialogues 

**Authors**: Hanbo Bi, Zhiqiang Yuan, Chongyang Li, Qiwei Yan, Zexi Jia, Jiapei Zhang, Xiaoyue Duan, Yingchao Feng, Jinchao Zhang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04591)  

**Abstract**: With the widespread adoption of multi-modal communication platforms, long-form dialogues interleaving text and images have become increasingly common. Users often need to retrieve coherent dialogue fragments related to specific topics, rather than isolated utterances. We propose Fine-grained Fragment Retrieval (FFR), which locates semantically relevant multi-utterance, multi-image fragments in multi-modal long-form dialogues. We explore two settings: (1) FFR within Single-Dialogue, retrieving fragments from a given dialogue; and (2) FFR within Dialogue Corpus, retrieving from a large-scale corpus for open-domain scenarios. For (1), we introduce F2RVLM, a generation-based retrieval model trained with reinforcement learning, using multi-objective rewards and difficulty-aware curriculum sampling to enhance fragment coherence. For (2), we develop FFRS, a two-stage system combining offline fragment-level indexing with online retrieval. Specifically, each dialogue is decomposed into minimal semantic fragments encoded by a Fragment Embedding Model (FEM) into a vector database; at inference, FEM rapidly recalls Top-K candidates, and F2RVLM performs fine-grained reasoning to identify the most relevant sub-content. To support FFR, we construct MLDR, the longest multi-modal dialogue retrieval dataset to date, and a WeChat-based real-world test set. Experiments on both benchmarks demonstrate that F2RVLM and FFRS consistently achieve superior performance across single-dialogue and corpus-level FFR. 

---
# VCIFBench: Evaluating Complex Instruction Following for Video Understanding 

**Authors**: Huangchen Xu, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04588)  

**Abstract**: Multimodal large language models have made rapid progress in video understanding, yet existing benchmarks largely rely on simple prompts and provide limited evidence about whether models can satisfy explicit output constraints. We introduce VCIFBench, a benchmark for evaluating complex instruction following in video understanding. VCIFBench constructs constraint-rich instructions from both benchmark-adapted and directly video-grounded prompts, covering content, format, style, and structure requirements, and evaluates model outputs with a hybrid verification pipeline. The benchmark contains 306 satisfiable test instructions, a 540-pair DPO preference dataset, and a 30-item conflict diagnostic subset. Experiments on 10 MLLMs show that joint constraint satisfaction remains challenging. We further show that DPO training on VCIFBench data can improve instruction-following performance. 

---
# Cartridges at Scale: Training Modular KV Caches over Large Document Collections 

**Authors**: Momchil Hardalov, Gonzalo Iglesias, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2606.04557)  

**Abstract**: Large Language Models can reason over long contexts, yet prefilling millions of tokens is wasteful as much of the content remains static across queries. Cartridges address this by distilling document collections into reusable key-value (KV) caches that eliminate prefilling while preserving accuracy. A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. We introduce Cartridges at Scale (CAS), a training framework for scalable multi-cartridge learning with dynamic distractor mixing and a memory-efficient budget manager that rotates hundreds of per-document cartridges between GPU and persistent storage. Our approach scales to collections exceeding a million tokens, improving over a monolithic cartridge by 10-31 points at comparable token budgets. Oracle cartridge accuracy falls within 2-6 points of full in-context learning even at high compression. When paired with retrieval for cartridge selection, CAS matches or exceeds conventional RAG accuracy while consuming 3-4x fewer prompt tokens. 

---
# Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents 

**Authors**: Yifan Simon Liu, Liam Gallagher, Faeze Moradi Kalarde, Jiazhou Liang, Armin Toroghi, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2606.04555)  

**Abstract**: Long-horizon conversational agents need to interact with users through evolving events, tasks, and goals. Such histories are naturally temporal, yet many existing memory systems organize information primarily by topical similarity and may ignore the order in which events occur. We introduce Segment Tree Memory, or SegTreeMem, a memory architecture that represents conversation history as a temporally ordered Segment Tree over utterances. SegTreeMem incrementally inserts new utterances through an online rightmost-frontier update rule, preserving chronological order while forming hierarchical memory segments. For retrieval, SegTreeMem propagates relevance scores through the tree to combine local semantic matching with hierarchical temporal context. Across three long-horizon memory benchmarks and two LLM backbones, SegTreeMem improves answer quality over flat retrieval, graph-structured memory, and tree-structured memory baselines. Additional temporal-order permutation analysis shows that the performance gain depends on preserving temporal order during memory construction, supporting the claim that temporal order is a key structure for agentic memory. 

---
# LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling 

**Authors**: Daria Ledneva, Denis Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2606.04552)  

**Abstract**: Genomic foundation models increasingly adopt large language model architectures, yet almost universally rely on fixed tokenization schemes such as $k$-mers, BPE, or single nucleotides, which impose arbitrary sequence boundaries that may obscure biologically relevant structure. We present LDARNet, a 120M-parameter hierarchical genomic foundation model that adapts H-Net-style dynamic chunking from autoregressive generation to masked language modeling, combining BiMamba-2 state-space layers with local attention, bidirectional routing, and a ratio-based regularizer to induce adaptive token boundaries without supervision. Fine-tuned on 27 tasks from the Nucleotide Transformer and Genomic Benchmarks suites, LDARNet achieves 11/18 wins among compact models ($<$300M parameters) and state-of-the-art results on 5 histone modification tasks, outperforming models up to 20$\times$ larger. A FLOPs-matched controlled experiment isolates learned routing as the source of these gains: learned boundaries beat fixed-grid boundaries by up to 14 percentage points on histone tasks at identical compute. Nucleotide-resolution analysis further shows that the learned boundaries align with canonical promoter motifs and splice junctions without supervision, providing a biological interpretation for adaptive tokenization in genomic foundation models. 

---
# Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models 

**Authors**: Boyan Han, Yiwei Wang, Yi Song, Yujun Cai, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04535)  

**Abstract**: Diffusion large language models (dLLMs) offer bidirectional attention and parallel generation, enabling them to exploit global context and naturally support format-constrained tasks like parseable JSON or reasoning templates. While straightforward fixed anchors can enforce such constraints, they often impose rigid spans, leading to truncated reasoning or redundant content. To overcome this, we propose Dynamic Infilling Anchors (DIA), a training-free method that dynamically estimates end-anchor positions to adjust generation length before iterative infilling. This flexible mechanism ensures structural correctness and semantic coherence, avoiding the inefficiencies of fixed-span methods. Experiments on reasoning benchmarks demonstrate that DIA substantially improves format compliance and answer accuracy, achieving significant zero-shot gains on GSM8K and MATH. These results establish DIA as a robust pathway toward reliable, structure-aware generation. 

---
# GENEB: Why Genomic Models Are Hard to Compare 

**Authors**: Daria Ledneva, Mikhail Nuridinov, Denis Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2606.04525)  

**Abstract**: Progress in genomic foundation models is difficult to assess due to fragmented benchmarks, incompatible evaluation protocols, and task-specific reporting. As a result, claims of superiority or generality across models are often not directly comparable. We introduce GENEB, a large-scale diagnostic benchmark that evaluates frozen representations from 40 genomic foundation models across 100 tasks spanning 13 functional categories under a unified probing-based protocol, including few-shot regimes. GENEB enables controlled comparison across model scale, architecture, tokenization, and pretraining data while explicitly exposing task-level trade-offs. Our analysis shows that aggregate leaderboards are unstable: model rankings vary sharply across task categories, scale provides only modest and inconsistent gains, and architectural and pretraining alignment frequently outweigh parameter count. These results highlight limitations of current evaluation practices and position GENEB as a reference framework for principled comparison and category-aware model selection in genomic machine learning. 

---
# SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference 

**Authors**: Yaosheng Fu, Guangxuan Xiao, Xin Dong, Song Han, Oreste Villa  

**Link**: [PDF](https://arxiv.org/pdf/2606.04511)  

**Abstract**: Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) the sparse selection step itself retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. SparDA adds $<$0.5% parameters and trains only the Forecast projections by matching the original selector's attention distribution. On two sparse-pretrained 8B models, SparDA matches or slightly improves accuracy and delivers up to 1.25$\times$ prefill speedup and 1.7$\times$ decode speedup over the sparse-attention offload baseline. By enabling larger feasible batch sizes on a single GPU, SparDA further reaches up to 5.3$\times$ higher decode throughput than the non-offload sparse baseline. Our source code is available at this https URL. 

---
# Self-Evolving Deep Research via Joint Generation and Evaluation 

**Authors**: Han Zhu, Chengkun Cai, Yuanfeng Song, Xing Chen, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04507)  

**Abstract**: Large Language Models (LLMs) have become increasingly adopted in daily applications, with deep research standing out as a particularly important capability. Unlike traditional question-answering (QA) tasks, deep research report generation lacks definitive ground-truth, making reward design inherently unverifiable and limiting effective reinforcement learning. Existing approaches mitigate this challenge with LLM-as-a-judge and query-dependent evaluation rubrics, but they still rely on static evaluators that cannot adapt their standards as the solver improves, leading to insufficient and eventually saturated optimization pressure. We address this limitation with a \textbf{s}elf-evolving \textbf{co}-evolutionary training framework for deep \textbf{re}search evaluation and generation (SCORE), which tightly couples an evaluator and a solver in a shared-parameter learning process. Rather than treating generation and evaluation as isolated modules, we leverage their intrinsic connection to enable joint improvement within a single shared-parameter model. To restrict this process, we introduce a meta-harness, which dynamically controls the evaluation environment based on solver performance, encouraging valid evaluation dimensions and sufficiently deep evaluator search. Extensive experiments on deep research benchmarks demonstrate consistent improvement in report generation quality, showing that co-evolving evaluation and generation is a promising direction for training open-ended research agents. 

---
# SANE Schema-aware Natural-language Evaluation of Biological Data 

**Authors**: Rolf Gattung, Martin Krueger, Markus Reischl  

**Link**: [PDF](https://arxiv.org/pdf/2606.04500)  

**Abstract**: High-throughput microscopy generates large, structured datasets capturing cellular responses to pharmacological perturbations, but accessing these datasets typically requires SQL expertise. Large language models offer a natural-language alternative, yet their tendency to hallucinate raises concerns about result reliability .
We present SANE Schema-Aware Natural-language Evaluation, a novel paradigm for domain-specific text-to-SQL evaluation: schema-grounded, automatically generated benchmarks tied to real and specific experimental structure. SANE makes evaluation more scalable, systematic, and reproducible.
Using SANE, we evaluate a few-shot large language model and show that, under constrained schemas with structured prompting and guardrails, accurate query generation is achievable without any model training or fine-tuning. Most failures stem from ambiguous or underspecified inputs and manifest as overly cautious clarification requests or answers to queries that should first be disambiguated, rather than incorrect SQL generation. These results indicate that few-shot large language models can provide reliable database access in well-defined domains when combined with schema-aware prompting. 

---
# Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs 

**Authors**: Zhongze Luo, Ruihe Shi, Zhenshuai Yin, Haoyue Liu, Weixuan Wan, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04483)  

**Abstract**: Existing jailbreaks against aligned LLMs are discrete artifacts whose surface forms are easy to fingerprint and patch. We argue that the real failure mode is not any specific prompt, but an entire register of natural human writing that safety training has under-covered. Building on this insight, we introduce the first jailbreak family that uses real fanfiction subgenres as universal attack carriers: a creative-writing meta is conditioned on passages from one of twelve Archive of Our Own (AO3) subgenres, and the harmful behavior is embedded as the climax of the resulting scene. The construction requires no attacker LLM and no per-target adaptation. On eight aligned LLMs over the union of HarmBench and JailbreakBench, this attack lifts mean ASR from 0.278 to 0.731 under a four-judge ensemble; a factorial decomposition shows the gain is carried by register rather than length or structure. Two active defences widen rather than narrow the vernacular-to-baseline ratio, indicating that template-targeting defences merely steer attackers toward register-based attacks like ours. We also propose SAGA-A4, a static four-turn extension that attains mean ASR 0.924, substantially exceeding three existing multi-turn methods. 

---
# Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention 

**Authors**: Ming-Hao Hsu, Xiaohai Tian, Jun Zhang, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04474)  

**Abstract**: Speech Large Language Models (SLLMs) underperform their text counterparts on complex reasoning. We reveal that this modality gap is not a uniform cognitive deficit. Evaluating three diverse SLLMs, we show speech-to-text (S2T) matches or exceeds text-to-text (T2T) on spatial, syntactic, and factual tasks. However, on logical tasks requiring entity tracking, S2T accuracy collapses to chance. We diagnose this localized degradation as an entity binding failure: continuous speech features cause models to lose precise entity-property associations during implicit reasoning. To resolve this, we propose Entity-Aware Chain-of-Thought (EA-CoT), forcing SLLMs to explicitly enumerate entities and bind them to claims before reasoning. Strikingly, EA-CoT bridges the gap, even when spoken names are misrecognized, yielding up to a 24.4% absolute accuracy improvement. Ablations confirm these gains stem entirely from explicit semantic binding, reframing the gap as a resolvable bottleneck. 

---
# Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning 

**Authors**: Chongyang He, Rui Zhang, Zixuan Wang, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04466)  

**Abstract**: Post-training Small Language Models (SLMs) for reasoning typically follows an SFT-then-RL pipeline, yet existing work rarely considers what data should be learned at each stage. We argue that data strategy should be aligned with the distinct roles of SFT and RL: SFT is better suited for acquiring not-yet-mastered reasoning skills, while RL is better suited for consolidating skills that the model can already partially access. Based on this principle, we propose a difficulty-aware SFT-then-RL framework that organizes training data into stage-specific sets. For hard samples in the SFT stage, we introduce a Bridge mechanism that transforms raw teacher-generated reasoning traces into more learnable supervision for SLMs. For hard samples that remain unsolved during RL, we apply Critique Fine-Tuning by converting all-zero-reward failures into diagnostic, repair, and new reasoning trace supervision for the next SFT stage. Experiments on two SLMs across five reasoning benchmarks show that our method consistently improves over representative SFT, distillation, and RL baselines. Our results highlight the importance of coordinating data difficulty across SFT and RL for effective SLM reasoning post-training. 

---
# SePO: Self-Evolving Prompt Agent for System Prompt Optimization 

**Authors**: Wangcheng Tao, Han Wu, Weng-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.04465)  

**Abstract**: System prompt optimization improves agent behavior without modifying the underlying model, yielding human-readable, model-agnostic instructions. Existing methods build a prompt agent that refines task agents' system prompts, yet leave the prompt agent's own system prompt hand-engineered and fixed. We propose Self-Evolving Prompt Optimization (SePO), which treats the prompt agent's own system prompt as an optimization target alongside task agents' system prompts. SePO adopts a self-referential design. A single prompt agent improves both task agents' system prompts and its own under an open-ended evolutionary search that maintains an archive of candidate prompts as stepping stones. Training proceeds in two stages: pre-training evolves the prompt agent on a multi-task pool, and fine-tuning then applies it to a target task. Across five benchmarks spanning math (AIME'25), abstract reasoning (ARC-AGI-1), graduate-level science (GPQA), code generation (MBPP), and logic puzzles (Sudoku), SePO consistently outperforms Manual-CoT, TextGrad, and MetaSPO, improving the average accuracy by 4.49 points compared to Manual-CoT. The prompt optimization skill from pre-training also generalizes to tasks beyond the pre-training mixture, rather than memorizing per-task prompts. 

---
# Stepwise Reasoning Enhancement for LLMs via External Subgraph Generation 

**Authors**: Xin Zhang, Yang Cao, Baoxing Wu, Kai Song, Siying Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.04454)  

**Abstract**: Large language models have shown strong performance in natural language generation and downstream reasoning tasks, but they still struggle with logical consistency, factual grounding, and interpretability in complex multi-step reasoning. To address these limitations, this paper proposes SGR, a stepwise reasoning enhancement framework that integrates large language models with external knowledge graphs through query-relevant subgraph generation. Given an input question, SGR first extracts key entities, relations, and constraints to construct a structured schema, then retrieves compact subgraphs from a knowledge graph using schema-guided querying. The generated subgraphs provide explicit relational evidence that guides the language model through step-by-step reasoning. In addition, SGR combines direct Cypher-based reasoning with collaborative reasoning integration, allowing candidate answers from multiple reasoning paths to be validated and aggregated according to both model confidence and graph consistency. Experiments on benchmark datasets including CWQ, WebQSP, GrailQA, and KQA Pro demonstrate that SGR improves reasoning accuracy and Hits@1 performance over standard prompting and several knowledge-enhanced baselines. Ablation studies further show that schema guidance and Neo4j-based retrieval are both crucial to the effectiveness of the framework. These results indicate that dynamically generated external subgraphs can improve the accuracy, robustness, and interpretability of LLM-based reasoning. 

---
# Listening to the Workforce: Measuring Construction Worker Safety Attitudes from Social Media Discourse Using LLMs 

**Authors**: Farouq Sammour, Yuxin Zhang, Zhenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04450)  

**Abstract**: Worker safety attitudes are key determinants of whether protective practices are applied or bypassed on construction sites. Yet measuring them at scale has remained out of reach. Safety attitudes are multidimensional, vary across topics, and surface most candidly in workers' own conversations. This study created and validated the Construction Safety Attitude Framework (CSAF), which integrates two components: a theory-grounded structure that characterizes safety attitudes along eight dimensions, and an operational codebook for measuring them in worker naturalistic discourse. Applying CSAF to 250 posts and comments from the r/Construction community on Reddit, trained coders reached strong agreement (Krippendorff's {\alpha} = 0.85). Pairwise lift and conditional probability confirmed that the eight dimensions are related yet distinct. To apply the framework across large volumes of discourse, CSAF was operationalized through a large language model (LLM) classifier. On 450 r/Construction contributions, the classifier reproduced expert human coding (Cohen's \k{appa} = 0.90, precision = 0.98, recall = 0.98), and on 400 contributions from r/Roofing it retained that accuracy after transfer to a different trade community (\k{appa} = 0.89, precision = 0.98, recall = 0.97). A proof-of-value case study then applied the validated classifier to 10,346 contributions from r/Roofing, demonstrating that CSAF can distinguish multidimensional attitudes by safety topic, track how they shift over time, and trace the reasoning behind unfavorable ones. The study therefore provides a theoretically grounded, empirically vetted instrument for examining safety attitudes, offering a basis for targeted interventions that address the attitudes underlying unsafe practices. 

---
# MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning 

**Authors**: Qiyang Xie, Jialun Wu, Xinjie He, Su Liu, Shuai Xiao, Zhiyuan Lin, Weikai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04442)  

**Abstract**: AI systems increasingly need to combine two demanding capabilities: navigating multi-session conversation history and performing deep reading comprehension within long documents. Yet no existing benchmark evaluates both simultaneously. We introduce MemoryDocDataSet, a synthetic benchmark of 50 micro-worlds and 1,000 QA pairs in which each instance comprises 3-5 personas, a temporal event graph spanning months of activity, 3-5 real long documents (20,000-50,000 tokens each sourced from the Caselaw Access Project), multi-session conversations grounded on those documents, and 20 question-answer pairs across five reasoning categories. The defining feature is the Hybrid source tag: questions requiring a system to first navigate conversation history to identify which document is relevant, then extract the answer from within that document. Hybrid questions account for 75.1% of the dataset. Dataset quality is characterised through a prompt-sensitivity self-consistency analysis using LLM-as-judge, yielding a median Cohen's $\kappa = 0.634$ across all 50 micro-worlds. We evaluate six baseline configurations spanning truncated context, long-context LLMs, retrieval-augmented generation (RAG), and memory systems. The best baseline (RAG-Both) achieves 0.358 overall F1 and 0.342 on Hybrid. Document-only retrieval (RAG-Doc) collapses to 0.267 on Hybrid despite achieving 0.453 on Doc-only questions, demonstrating a clear joint-retrieval gap that motivates architectures unifying conversational memory with long-document navigation. We release the dataset, generation pipeline, and all baseline implementations. 

---
# Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models 

**Authors**: Anant Khandelwal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.04396)  

**Abstract**: Diffusion large language models (dLLMs) generate responses by iteratively unmasking and revising many positions in parallel. This process leaves a rich denoising trace depicting which tokens become confident, which remain unstable, and when commitments form. Existing dLLM reinforcement learning methods use this signal only weakly. Flat rollouts are cheap, but assign a single outcome reward to the whole trajectory. Tree rollouts provide finer, verifiable training signals by branching partial trajectories and propagating leaf rewards upward, but are compute intensive. We ask whether the denoising trace itself can provide tree-like supervision without tree-level compute. We introduce CAPR (Cached-Amortized Path Refinement), a dLLM-RL algorithm that summarizes the denoising trace into a compact path state, uses cached trajectory states to generate cheap sibling continuations, and trains a block-level value head for local block-wise supervision. Under a block-wise unmasking schedule, CAPR records path-state and block-progress features, then redistributes the final outcome reward across blocks according to the tokens revealed in each block. This trains the value head to convert one sparse reward into block-level PPO weights. CAPR therefore recovers much of the granularity of tree search while avoiding full tree expansion, reducing rollout-generation cost to roughly 0.75x that of flat rollouts and 0.6x that of tree rollouts (under standard settings). Across 4x4 Sudoku, Countdown, GSM8K, and Math500, on dense and mixture-of-experts LLaDA backbones, CAPR sets a new state of the art for RL-tuned dLLMs at 256- and 512-token budgets. On Sudoku, it matches the strongest tree-structured baseline at less than one third of the per-step compute. 

---
# When Clients Stop Following: A Cognitive Conceptualization Diagram-driven Framework for Strategic Counseling 

**Authors**: Yihao Qin, Junyi Zhao, Changsheng Ma, Yongfeng Tao, Minqiang Yang, Chang Liu, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04389)  

**Abstract**: Large Language Models (LLMs) show promise in psychological counseling, yet existing benchmarks rely heavily on highly cooperative simulated clients. We observe a critical counselor-following phenomenon: these clients often rapidly shift from resistance to compliance after only a few turns, creating an illusion of therapeutic progress and inflating scores under current evaluation protocols through superficial empathy. To address this evaluation mismatch, we propose a Cognitive Behavioral Therapy (CBT)-grounded resistance-aware framework. We introduce CARS, a client simulator that explicitly models dynamic resistance via Cognitive Conceptualization Diagrams (CCDs). We present STREAMS, a dual-module framework that decouples strategic reasoning (Thinker) from response generation (Presenter) and optimizes it via reinforcement learning. We further propose EWTS-MI, an entropy-weighted metric for evaluating responsiveness under high-friction interactions. Experiments across resistant and non-resistant counseling settings validate our findings on evaluation mismatch and demonstrate the effectiveness of resistance-aware training for improving strategic robustness under challenging counseling interactions. 

---
# DLLG: Dynamic Logit-Level Gating of LLM Experts 

**Authors**: Bingnan Li, Zhaoyang Zhang, Xiaoze Liu, Yantao Shen, Shuli Jiang, Shuo Yang, Wei Xia, Zhuowen Tu, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2606.04378)  

**Abstract**: Leveraging multiple specialized LLMs can combine complementary strengths, but existing approaches trade adaptability for stability: routing commits prematurely, heuristic ensembling depends on fragile proxies, and parameter merging introduces interference. We propose DLLG (Dynamic Logit-Level Gating), a dynamic logit-level ensembling framework that learns token-level expert fusion from sparse response-level supervision. A lightweight gating module predicts step-wise fusion weights, linking trajectory-level correctness to generation without token-level labels or expert retraining. Across diverse reasoning and code benchmarks, DLLG consistently outperforms strong routing, heuristic ensembling, and parameter-merging baselines across model scales, highlighting learned logit-level fusion as a robust and scalable paradigm for integrating specialized experts. 

---
# GlossAssist -- A Tool to Simplify Corpus Creation and Study the Effect of NLP Models in Low-Resource Documentation Settings 

**Authors**: Bhargav Shandilya, Matt Buchholz, Alexis Palmer  

**Link**: [PDF](https://arxiv.org/pdf/2606.04367)  

**Abstract**: Interlinear glossed text (IGT) is the standard format for linguistic annotation in language documentation. Producing it manually, however, is often slow and costly. Automated glossing systems have improved substantially in recent years, but adoption among field linguists remains limited. Existing tools are designed to be evaluated rather than used, offering no interpretable path for correction or the incorporation of linguistic expertise back into model behavior. We present GlossAssist, a glossing tool built around the retrieval-based architecture of CWoMP (Contrastive Word-Morpheme Pre-training), which grounds predictions in a mutable lexicon of learned morpheme representations. In conjunction with CWoMP, our system treats each correction by an annotator as part of an active learning setting, which expands the lexicon and improves future predictions without having to retrain the model. In this paper, we present our interface and argue that this feedback loop should be treated as a design requirement for NLP tools aimed at documentary linguists. 

---
# Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs 

**Authors**: Xinyu Pang, Zhanke Zhou, Xuan Li, Fangrui Lv, Shanshan Wei, Sen Cui, Bo Han, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04360)  

**Abstract**: Symbolic regression (SR) discovers compact mathematical expressions from data, yet recent LLM-based evolutionary methods remain sample-inefficient because they rely mainly on scalar feedback such as MSE. We identify a core limitation: existing methods conflate candidate proposal with search guidance, requiring the LLM to infer how to evolve an expression, diagnose its errors, and reuse past experience from a single score. To address this, we propose Deliberate Evolution (DE), an agentic framework that decouples symbolic generation from search control. DE guides LLM proposals with adaptive operators for search direction, analytical tools for structural diagnosis, and reflective memory for trajectory-level experience. Experiments on LLM-SRBench show that DE consistently outperforms representative LLM-based SR baselines across diverse scientific domains while using only 40% of the standard sample budget. 

---
# Noisy memory encoding explains negative polarity illusions 

**Authors**: Yuhan Zhang, Edward Gibson  

**Link**: [PDF](https://arxiv.org/pdf/2606.04340)  

**Abstract**: A sentence like "The authors that no critics recommended have ever received acknowledgment for a best-selling novel" is sometimes rated as acceptable even though, strictly speaking, it is ungrammatical because the negative polarity word "ever" is not licensed where it is. This behavioral effect is sometimes called a "negative polarity illusion". Here we propose that the lossy context surprisal theory of Hahn et al. (2022) -- whereby people have an imperfect encoding of complex sentences -- might explain this effect. We hypothesize that people have poor memory representation of the determiners in the main-clause and embedded-clause subjects and could entertain a determiner exchange that licenses ever. We propose that more similar determiners in those positions would trigger stronger illusion effects. Acceptability judgment tasks with six novel determiner pairs (e.g., "few" and "many", "few" and "most") support our proposal, showing, specifically, that a novel sentence, "Many authors that few critics recommended have ever received acknowledgment for a best-selling novel", triggered a much stronger illusion than the canonical one even without time pressure. These results offer further support for the suggestion that human language processing is imperfect and resource-rational: in face of working memory limitations, humans rationally reconstruct what is most likely from noisy linguistic input to facilitate downstream processing. 

---
# Parameter-Efficient Fine-Tuning with Learnable Rank 

**Authors**: Arpit Garg, Simon Lucey, Hemanth Saratchandran  

**Link**: [PDF](https://arxiv.org/pdf/2606.04325)  

**Abstract**: Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning (PEFT) method that restricts weight updates to low-rank adapters, introducing a fixed low-rank inductive bias by optimizing in a low-dimensional subspace. In this work, we question whether a fixed-rank constraint is the most effective inductive bias for parameter-efficient fine-tuning. We introduce *Learnable Rank LoRA (LR-LoRA)*, a PEFT method in which the adapter rank is learned during the training process. Instead of prescribing a uniform rank for all adapter layers, LR-LoRA allows the optimizer to determine the appropriate rank for each layer. Using this approach, we find substantial layer-wise variation in the learned ranks, with the attention and MLP layers in the transformer models exhibiting systematically different rank preferences. Across a range of language understanding and commonsense reasoning benchmarks, LR-LoRA achieves state-of-the-art performance in most settings and consistently outperforms strong PEFT baselines, demonstrating that a learnable rank provides a more flexible and effective inductive bias than fixed-rank adaptations. 

---
# LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding 

**Authors**: Haocheng Xia, Mihir Pamnani, Hanxi Fang, Supawit Chockchowwat, Yongjoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.04302)  

**Abstract**: Key-value (KV) caching accelerates inference of large language models (LLMs) by reusing past computations for generated tokens. Its importance becomes even greater in long-context applications such as retrieval-augmented generation (RAG) and in-context learning (ICL). However, conventional KV caching embeds positional information directly into the cache, limiting its reusability. Existing solutions either restrict reuse to prefixes or require expensive memory materialization for positional re-encoding. We introduce LazyAttention, a novel attention mechanism that kernelizes deferred positional encoding to enable zero-copy, position-agnostic KV reuse. By adjusting positional encoding within attention kernels on-the-fly, LazyAttention resolves the materialization bottleneck, allowing a single physical KV copy to serve multiple logical requests at arbitrary positions. Leveraging attention kernels tailored for prefilling and decoding, our system achieves significant efficiency improvements: under skewed document distributions, it reduces time-to-first-token (TTFT) by 1.37$\times$ and increases inference throughput by 1.40$\times$ compared to the state-of-the-art Block-Attention, while maintaining comparable output quality. 

---
# Using Text-Based Causal Inference to Disentangle Factors Influencing Online Review Ratings 

**Authors**: Linsen Li, Aron Culotta, Nicholas Mattei  

**Link**: [PDF](https://arxiv.org/pdf/2606.04286)  

**Abstract**: Online reviews provide valuable insights into the perceived quality of facets of a product or service. While aspect-based sentiment analysis has focused on extracting these facets from reviews, there is less work understanding the impact of each aspect on overall perception. This is particularly challenging given correlations among aspects, making it difficult to isolate the effects of each. This paper introduces a methodology based on recent advances in text-based causal analysis, specifically CausalBERT, to disentangle the effect of each factor on overall review ratings. We enhance CausalBERT with three key improvements: temperature scaling for better calibrated treatment assignment estimates; hyperparameter optimization to reduce confound overadjustment; and interpretability methods to characterize discovered confounds. In this work, we treat the textual mentions in reviews as proxies for real-world attributes. We validate our approach on real and semi-synthetic data from over 600K reviews of U.S. K-12 schools. We find that the proposed enhancements result in more reliable estimates, and that perception of school administration and performance on benchmarks are significant drivers of overall school ratings. 

---
# Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit 

**Authors**: JooYoung Lee, Lin Tian, Angela Brillantes, Adriana-Simona Mihăiţă, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04274)  

**Abstract**: As large language models (LLMs) become default tools for online information verification, an implicit assumption follows them: that scale and general capability are sufficient for nuanced classification of misinformation discourse. We test this assumption directly on 900 Reddit comments spanning three PolitiFact-verified misinformation claims (environment, health, immigration), labelled as belief (propagates the claim), fact-check (corrects it), or other. We compare nine models across three paradigms -- BART-MNLI, three Llama variants, three commercial frontier LLMs (Claude Haiku 4.5, Gemini Flash Lite 2.5, Claude Sonnet 4.6), and fine-tuned DistilBERT and RoBERTa -- under universal and topic-specific label schemas.
The assumption does not hold. Fine-tuned RoBERTa reaches 0.62 macro-$F_1$ against a best zero-shot result of 0.50 (Claude Haiku 4.5), at a fraction of the per-query cost; the supervised advantage is concentrated on the belief class, the implicit, affective category every zero-shot model under-detects. Scaling does not help: Llama-3-8B matches Llama-3-70B, and Claude Sonnet 4.6 underperforms the smaller Haiku under generic labels, collapsing belief detection to 0.17 and refusing outright on a subset of comments flagged as sensitive. This is a safety-alignment artefact, not a capacity limit. Label schema and topic jointly shape zero-shot performance, with the same model varying by more than 0.13 macro-$F_1$ across topics under matched labels. In a verification context, where missing belief is the costlier error, task-specific fine-tuning remains the more reliable choice despite the proliferation of large generative models. 

---
# Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA 

**Authors**: Maroof Kousar, Yibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04262)  

**Abstract**: Large language models (LLMs) are increasingly used for everyday health questions, including whether a user can safely take another dose of an over-the-counter (OTC) medication. Yet this common safety-relevant setting remains underexplored in existing medical QA evaluations, where correct answers require tracking dose timing, computing rolling 24-hour intake, following product-label constraints, and handling incomplete medication histories. We introduce DOSEBENCH, a focused benchmark of 81 curated OTC dosing scenarios focused on adult acetaminophen and ibuprofen use, with manually annotated gold references. We evaluate four LLMs across repeated runs using metrics for decision correctness, consistency, explanation verifiability, failure types, and confidence-related signals, resulting in 1,620 model responses. Our results show that models frequently struggle with rolling-window reasoning and ambiguity-sensitive cases and that stable or confident-looking responses can still violate dosing constraints. These findings suggest that OTC dosing QA provides a narrow yet practical testbed for evaluating temporal reasoning, constraint following, and safety-relevant uncertainty handling in medical QA. 

---
# Supportive Token Revealing for Fast Diffusion Language Model Decoding 

**Authors**: Giries Abu Ayoub, Mario Barbara, Lluís Pastor-Pérez, Tanja Bien, Aneesh Barthakur, Alaa Maalouf, Loay Mualem  

**Link**: [PDF](https://arxiv.org/pdf/2606.04236)  

**Abstract**: Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is needed. It then shifts the criterion from which tokens are safest to reveal to which confident reveals would best support later denoising. AXON selects anchors, confident masked tokens that uncertain positions attend to, using attention, uncertainty, and confidence signals. Experiments on reasoning and code-generation benchmarks across multiple diffusion language models show that AXON improves the quality-latency trade-off of existing parallel decoders, often reducing the number of function evaluations while maintaining or improving accuracy. 

---
# MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A 

**Authors**: Hanoz Bhathena, Parin Rajesh Jhaveri, Rohan Mittal, Prateek Singh, Aymen Kallala, Rachneet Kaur, Yiqiao Jin, Zhen Zeng, Adwait Ratnaparkhi, Denis Kochedykov  

**Link**: [PDF](https://arxiv.org/pdf/2606.04231)  

**Abstract**: Recent advances in multimodal retrieval-augmented generation (MM-RAG) have shifted toward minimal parsing, relying on page-level images for producing retriever embeddings and for answer generation. While efficient, this trend often neglects explicit handling of the rich, structured information in complex enterprise documents, instead depending on pre-trained embeddings or vision-language models to implicitly capture such structure. In this work, we take a more direct approach: MM-BizRAG proactively extracts and represents document structure via a document structure-aware split that dynamically routes documents through orientation-specific ingestion pipelines, applying explicit layout-aware parsing for vertically structured documents (e.g., reports) and holistic page-level representations for horizontally structured documents (e.g., slide decks). A unified LLM-driven artifact transformation pipeline with placeholder-based positional alignment preserves natural reading order, while inference-time multimodal assembly decouples retrieval representations from generation context, enabling richer, more grounded answers without any finetuning requirement. Through experiments on a large, heterogeneous enterprise dataset and two public benchmarks (SlideVQA and FinRAGBench-V), MM-BizRAG consistently outperforms state-of-the-art vision-centric baselines by up to 32% points, with especially strong gains on report-style layouts. Furthermore, we introduce FastRAGEval, a single-call LLM Judge metric for fine-grained generative recall that halves RAGChecker's cost while achieving stronger human alignment. 

---
# Cross-Prompt Generalization in Detecting AI-Generated Fake News Using Interpretable Linguistic Features 

**Authors**: Aya Vera-Jimenez, Samuel Jaeger, Calvin Ibenye, Dhrubajyoti Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2606.04199)  

**Abstract**: The increasing use of large language models has raised concerns about the spread of AI-generated fake news, particularly under varying prompting strategies. Most existing detection models are trained and evaluated under a single generation setting, leaving their ability to generalize across unseen prompts unclear. In this study, we investigate cross-prompt generalization in fake news detection using three datasets of AI-generated articles produced under distinct prompts, combined with real news articles. We extract interpretable linguistic features capturing lexical diversity, readability, and emotion-based characteristics and evaluate a random forest classifier under a cross-prompt framework, where models trained on one prompt are tested on another. Across all six train-test combinations, performance remains consistently high, with AUC values ranging from 0.988 to 1.000. Analysis of feature distributions shows that AI-generated text exhibits increased lexical diversity, reduced readability, and substantially lower emotional intensity compared to the overall dataset, with variations across prompts. Despite these distributional shifts, the classifier maintains strong performance, indicating that these features capture stable properties of AI-generated text that generalize across prompting strategies. These findings suggest that feature-based approaches can provide robust detection of AI-generated fake news under prompt variability. 

---
# ACAT: A Collaborative Platform for Efficient Aspect-Based Sentiment Dataset Annotation 

**Authors**: Ana-Maria Luisa Mocanu, Ciprian-Octavian Truica, Elena-Simona Apostol  

**Link**: [PDF](https://arxiv.org/pdf/2606.04189)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) requires high-quality datasets to train reliable models. However, existing annotation tools treat output as flat files, leaving researchers to manually consolidate multi-annotator data, reconstruct relational structures, and compute reliability metrics through custom scripts. This paper introduces ACAT (Aspect-based sentiment analysis Collaborative Annotation Tool), a web-based platform natively supporting four ABSA workflows: (1) Aspect-Category Sentiment Analysis, (2) Clause-Level Segmentation, (3) Aspect-Term Sentiment Analysis with character-level position tracking, and (4) Aspect Sentiment Triplet Extraction with dual span offset preservation. Its core contribution is an automated Extract, Transform, Load (ETL) pipeline that aligns collaborative annotations and computes Inter-Annotator Agreement (IAA) metrics directly at export, yielding training-ready datasets. In a preliminary validation on 1,002 restaurant reviews with two annotators of differing expertise, ACAT achieves a median annotation time of 31.58 seconds and a raw IAA ranging from 0.78 to 0.86 across all tasks. 

---
# A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models 

**Authors**: Yassir El Attar, Esra Dönmez, Maximilian Maurer, Agnieszka Falenska  

**Link**: [PDF](https://arxiv.org/pdf/2606.04177)  

**Abstract**: Interpretable linguistic features offer a promising approach for explaining why a given text appears machine-generated, particularly for non-expert users. However, existing findings on which features reliably indicate LLM-generated text remain fragmented across feature sets, models, and text domains. To address this gap, we conduct a large-scale empirical study assessing the robustness of linguistic signals for characterizing AI-generated text. Our analysis covers 284 interpretable linguistic features across outputs from 27 LLMs and ten text domains under cross-model and cross-domain generalization settings. We show that classifiers based solely on linguistic features can reliably distinguish AI-generated from human-written text. However, many previously proposed indicators prove strongly context-dependent, with the exception of measures of lexical richness, which remain robust signals across model families and text domains. These results demonstrate which linguistic signals generalize across contexts and provide a foundation for more reliable, interpretable analyses of AI-generated language. 

---
# Expert-Aware Refusal Steering 

**Authors**: Anna C. Marbut, Daniel R. Olson, Travis J. Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2606.04160)  

**Abstract**: Safety alignment in instruction-tuned large language models (LLMs) depends on a model's ability to reliably refuse to respond to harmful or disallowed requests. Recent work has shown that a steering vector can be applied to a dense LLM during inference to effectively suppress refusal behavior, inducing response to harmful requests. We extend this refusal steering method to three open-source Mixture-of-Experts (MoE) LLMs and find that steering performance is uninhibited by the complex routing patterns inherent to the MoE architecture. We then propose two expert-aware refusal steering methods that leverage refusal-specific expert routing patterns and expert-specific steering directions to suppress normal refusal behavior. We find that refusal behavior can be effectively steered based on the output of a single expert. Our results show that refusal signals captured by steering methods differ from expert routing behavior, suggesting a substantial role for attention in MoE refusal behavior. 

---
# When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG 

**Authors**: Erfan Nourbakhsh, Rocky Slavin, Ke Yang, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2606.04127)  

**Abstract**: Medical question answering is a high-stakes setting where factual errors can have serious consequences. Retrieval-augmented generation (RAG) is widely viewed as a promising solution, and prior work has reported substantial gains for large medical QA models. We revisit this assumption across a broad range of open-weight instruction-tuned models spanning 7B to 72B parameters. Across five models, ten biomedical QA datasets, four retrieval methods, and four retrieval corpora, we find that retrieval yields only small and inconsistent improvements over a no-retrieval baseline, typically within 1-2 points. In contrast, the choice of backbone model has a much larger effect than the choice of retriever or corpus, and expert and layman retrieval sources perform similarly in most settings. These results suggest that the main bottleneck is not retrieval quality alone, but the model's limited ability to use retrieved evidence effectively. 

---
# SaliMory: Orchestrating Cognitive Memory for Conversational Agents 

**Authors**: Kai Zhang, Xinyuan Zhang, Hongda Jiang, Shiun-Zu Kuo, Hyokun Yun, Ejaz Ahmed, Shereen Oraby, Ziyun Li, Sanat Sharma, Ann Lee, Ahmed A Aly, Anuj Kumar, Raffay Hamid, Xin Luna Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.04120)  

**Abstract**: Conversational agents that serve as lifelong companions must maintain persistent memory across all interactions. However, simply expanding context windows with raw retrieval degrades reasoning quality, while training memory agents via standard reinforcement learning creates a severe credit assignment bottleneck in a multi-stage pipeline. To solve this, we introduce SALIMORY, a framework that trains a single language model to manage a cognitively-structured memory-spanning user facts, preferences, and working memory. By introducing a hierarchical stage-wise process reward and reward-decomposed contrastive refinement, SALIMORY provides isolated supervision for distinct memory operations (selective filtering, consolidation, and cue-driven recall) end-to-end. SALIMORY cuts memory-attributed failures by one-third, outperforms the state-of-the-art by over 10% in end-to-end accuracy, and more than doubles the Good Personalization rate. 

---
# Computational conceptual history of scientific concepts: From early digital methods to LLMs 

**Authors**: Michael Zichert, Arno Simons  

**Link**: [PDF](https://arxiv.org/pdf/2606.04118)  

**Abstract**: This article situates large language models (LLMs) within the longer history of computational approaches to concept analysis in the history, philosophy, and sociology of science (HPSS). We examine what LLMs add to existing methods, how they inherit longstanding problems, and review recent case studies that employ them. In the first part, we reconstruct computational conceptual history before LLMs by bringing together three strands of work: early digital methods in HPSS, distributional approaches from digital history and related research, and lexical semantic change detection. We provide an overview of the main challenges and opportunities, focusing on corpus construction, operationalization and modelling choices, and evaluation and interpretation. In the second part, we turn to the era of LLMs, starting with a short introduction to LLMs before reviewing LLM-based work on lexical semantic change detection and relevant case studies in HPSS. We then revisit the earlier methodological questions, showing how issues of corpus construction, model choice and training data, operationalization trade-offs, and evaluation and interpretation play out in LLM-based workflows. 

---
# Discourse-Role Labels as Presentation-Time Variables for Context Use in Language Models 

**Authors**: Jianguo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04109)  

**Abstract**: Context-augmented language model systems often wrap supplied content with labels such as Reference:, Evidence:, Instruction:, Note:, or Example:, but the effect of these labels on reader-model behavior remains underexplored. We introduce a paired fixed-content probe over 500 MMLU-Pro items: each item receives the same misleading answer-bearing assertion under different discourse-role labels, and adoption is measured by whether the model outputs the injected wrong option. Across GPT-5.5, DeepSeek V4 Pro, Llama-3-8B-Instruct, and Qwen2.5-7B-Instruct, Misleading Adoption Rate shifts by 56-84 percentage points. Binding or source-like labels such as Instruction: and Reference: produce high adoption, whereas Example: consistently suppresses it. Paired tests, bootstrap intervals, final-instruction ablations, and Qwen final-step log-probability probes support a label-conditioned candidate preference. Boundary probes show where the effect weakens or persists: arithmetic tasks reduce adoption, passage-shaped external context preserves smaller label gaps, short-answer evaluation rules out option-letter copying, and nested-label conflicts suggest that illustrative framing can delimit adoption scope. A 200-case single-author manual audit confirms that the short-answer contrasts are stable under conservative adjudication. The resulting claim is bounded but practical: context-utilization and reader-side RAG benchmarks should report and control wrapper labels, because presentation choices can change measured reliance on supplied context. 

---
# POLARIS: Guiding Small Models to Write Long Stories 

**Authors**: Rishanth Rajendhran, Jenna Russell, Mohit Iyyer, John Frederick Wieting  

**Link**: [PDF](https://arxiv.org/pdf/2606.04095)  

**Abstract**: Small open-weight models struggle at long-form creative writing: their generated stories either fall far short of the requested length, or their quality significantly degrades as length increases, especially when compared to frontier models. We present POLARIS (Policy Optimization with LLM-as-a-judge rewards and Anchored-Reference Injection for Storywriting), a lower-compute GRPO recipe with two key ingredients: a frontier LLM judge with a structured Story Quality rubric as the online reward, and human-reference injection (HRI), where a teacher-forced human-written story serves as a high-reward anchor within each GRPO group. By applying our training recipe to Qwen3.5-9B, using a dataset of approximately 1.4K prompt-story pairs derived from 100 short-story anthologies and 4 A100 GPUs, we obtain POLARIS-9B. Across five benchmarks spanning in-distribution and out-of-distribution prompts and rubrics, POLARIS-9B is competitive with much larger open-weight models while following length instructions more closely. A blinded human evaluation confirms that POLARIS-9B is preferred to the base Qwen3.5-9B and on par with Qwen3.5-27B. Despite training only on stories up to 4k words, POLARIS-9B preserves quality on prompts requesting stories up to 3 times the training length, a regime where most open-weight models degrade substantially in quality, length adherence, or both. More broadly, our results suggest that length generalization is a meaningful stress test for creative-writing models and a useful lens for distinguishing otherwise close models. 

---
# STRIDE: Training Data Attribution via Sparse Recovery from Subset Perturbations 

**Authors**: Rishit Dagli, Abir Harrasse, Luke Zhang, Florent Draye, Amirali Abdullah, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.05165)  

**Abstract**: Training Data Attribution (TDA) seeks to trace a model's predictions back to its training data. The gold standard for TDA relies on causal interventions, observing how a model changes when data is added or removed, but repeated retraining is computationally challenging for Large Language Models (LLMs). Consequently, most approaches approximate this effect in the parameter space using gradients. However, tracking gradients across billions of parameters is not only prohibitively expensive but relies on local approximations. In this work, we propose a shift: rather than estimating parameter changes, we model the functional effect of training data in the activation space. We introduce STRIDE (Steering-based Training Data Influence Decomposition), a framework that formulates TDA as a sparse recovery problem in the spirit of compressive sensing. STRIDE learns lightweight "steering operators" that mimic the behavioral shift caused by training on data subsets. By measuring how these operators perturb test predictions, we recover individual training example influences via sparse linear decomposition. STRIDE achieves state-of-the-art for LLM pre-training attribution while being an order of magnitude ($13\times$) faster than previous art. We further validate its practical utility through downstream applications including data selection, data contamination, and qualitative analysis. 

---
# Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models 

**Authors**: Yichen Gao, Yiqun Zhang, Zijing Wang, Yujia Li, Heng Guo, Xi Wu, Xiaocui Yang, Shi Feng, Yifei Zhang, Daling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.05161)  

**Abstract**: Audio-language models (ALMs) often follow text that conflicts with audio, even when the audio evidence is clear. This raises a basic question: is the audio-supported answer unavailable, or is it represented but overridden by the conflicting text? We examine this question using a same-audio counterfactual that keeps the audio fixed, removes only the conflicting text, and measures the resulting shift in model preference. Across five ALMs and four conflict tasks, 64.1% of conflict samples show a sign flip: the same-audio branch prefers the audio-supported answer, whereas the joint branch prefers the text-supported answer. This pattern suggests that the relevant audio evidence is encoded but loses in arbitration. Activation patching further localizes the reversal to answer-position computation, and patching effects closely track output candidate-score differences (Spearman rho=0.93). Using this diagnostic, we propose Gated Audio Counterfactual Logit Correction (GACL), a training-free decoding rule that interpolates between joint and same-audio scores. Under a strict 5 pp faithfulness-drop budget, GACL improves nAUC by 17.8 points over the best contrastive baseline and transfers without retuning to vision-text arbitration (up to +40.5 pp). 

---
# Reinforcement Learning from Rich Feedback with Distributional DAgger 

**Authors**: Rishabh Agrawal, Jacob Fein-Ashley, Paria Rashidinejad  

**Link**: [PDF](https://arxiv.org/pdf/2606.05152)  

**Abstract**: Reasoning models have advanced rapidly, but the dominant reinforcement learning from verifiable rewards (RLVR) recipe remains surprisingly narrow: sample many responses and reward each with a single bit indicating whether the final answer is correct. Yet many settings provide rich feedback, including execution traces, tool outputs, expert corrections, and model self-evaluations. We study how to use such feedback through a distributional variant of the classic imitation learning algorithm DAgger, where the learner has local access to an expert distribution on states visited by the current policy. This yields a simple forward cross-entropy objective that admits a blackbox expert and whose sequence-level gradient {conduct rich credit assignment by propagating} future expert-student disagreement back to earlier decisions. We show that prior RL with self-distillation objectives based on reverse KL or Jensen-Shannon fail to guarantee monotonic policy improvement: even when the expert has higher reward, their updates may increase probability on worse actions. In contrast, we show that forward cross-entropy admits monotonic policy improvement and enjoys guarantees on regret. We further show that our objective optimizes a lower bound on teacher-weighted likelihood of success, leading to improved Pass@N. Empirically, our approach, DistIL, improves over RLVR and RL with self-distillation baselines across a variety of domains: scientific reasoning, coding, and solving hard mathematical problems. 

---
# Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them) 

**Authors**: Nizar Islah, Istabrak Abbes, Irina Rish, Sarath Chandar, Eilif B. Muller  

**Link**: [PDF](https://arxiv.org/pdf/2606.05145)  

**Abstract**: When post-trained language models fail on reasoning problems, the common test-time-scaling response is to spend more compute on additional attempts, and the failed traces play no further role. We argue this discards a crucial signal; some failures come from unlucky sampling, where more rollouts help, while others are structural and resist resampling regardless of budget. We propose that failed traces encode recoverability structure: the inference-time signature of which test-time interventions can rescue a given failure. Three problem-level trajectory features, derived from the structure of available interventions, recover this structure from the distributional signature of failed rollouts, not their text. They cluster failures into stable regimes, characterize the failure topography of different post-training methods ($84.3{\pm}4.3\%$ accuracy, $+20\%$ over a majority-class baseline), and support a training-free routing rule that lifts rescue by $+12.2\%$ on the deployment-relevant Steerable-Hard subset (failures where retry is insufficient and a bounded intervention is reachable). The features and the routing rule transfer across two cross-family probes. The same three features thus convert failed traces from discarded data into a diagnostic object, supporting test-time routing and post-training analysis without training-time or weight-space access. 

---
# Audio Interaction Model 

**Authors**: Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2606.05121)  

**Abstract**: Audio is an inherently interactive modality, yet today's Large Audio Language Models (LALMs) are offline, and streaming audio models each handle only a single task such as streaming ASR or voice chatting. It is time to unify them into one online LALM: a model that, through an always-on perceive-decide-respond loop, listens to sound, environment, and instructions in real time and reacts on the fly. We formalize this regime as the Audio Interaction Model, and realize it with Audio-Interaction, a unified streaming model that retains offline task execution while adding online general audio instruction following, from dialogue to full voice chatting, deciding when to respond from the semantics of the stream. To enable this, we propose SoundFlow, a framework that instantiates the perceive-decide-respond loop end to end, from data to training to deployment, through streaming-native data construction, comprehension-aware training, and asynchronous low-latency inference for stable real-time interaction. We further construct StreamAudio-2M, a 2.6M-item streaming corpus spanning 7 fundamental abilities and 28 sub-tasks, and Proactive-Sound-Bench for evaluating proactive audio intervention. Across 8 benchmarks, Audio-Interaction preserves competitive performance on mainstream audio tasks while unlocking capabilities inaccessible to offline LALMs, including real-time ASR, streaming audio instruction following, and proactive help. 

---
# Continual Visual and Verbal Learning Through a Child's Egocentric Input 

**Authors**: Xiaoyang Jiang, Yanlai Yang, Kenneth A. Norman, Brenden Lake, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.05115)  

**Abstract**: Children learn the meanings of words from a continuous, temporally structured stream of egocentric experience. Recent work shows that neural networks can also learn word-referent mappings from a child's egocentric video recordings, but they cycle through the shuffled data for hundreds of epochs, contrasting with how children actually encounter their environment. We introduce BabyCL, a continual multimodal learning framework that processes the SAYCam dataset in a single chronological pass, combining streaming visual representation learning with an image-text contrastive objective. BabyCL combines a multi-stage temporal segmentation of the stream with a dual replay buffer that independently manages visual and multimodal histories, and it is jointly trained with three contrastive losses on a shared backbone. Under a matched optimization budget, BabyCL outperforms streaming learning baselines on the SAYCam Labeled-S 4AFC benchmark, substantially narrowing the gap to an upper bound of offline training. Ablations show that the gains are robust to the length of the online temporal segmentation window and the eviction rule of the replay buffer. Together, these results show that meaningful word-referent mappings can emerge under training conditions much closer to a child's actual experience. 

---
# In-Context Graphical Inference 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.05042)  

**Abstract**: Marginal inference in discrete graphical models forces a choice between exactness and scalability: exact algorithms are intractable for high-treewidth graphs, while iterative approximations (Belief Propagation, variational methods) sacrifice convergence guarantees on frustrated topologies. We argue that this dichotomy stems from a mismatched inductive bias: iterative methods abandon the sequential elimination structure that makes exact inference correct. We introduce In-Context Graphical Inference (ICG-I), an autoregressive Graph Transformer that restores this structure by mimicking Variable Elimination with learned, Tensor- Train-compressed intermediate factors, paired with a Dirichlet output layer and Weighted Conformal Prediction for calibrated, distribution-free coverage guarantees under topological shift. We prove that TT compression errors propagate at most lincarly through the autoregressive chain, that the Dirichlet-Multinomial loss is a proper scoring rule, and that WCP maintains coverage with a quantifiable degradation under estimated density ratios. We conducted intensive experiments to evaluate ICG-I and achieved state-of-the-art performance across all benchmarks. ICG-I reduces MAE from 0.041 (best baseline) to 0.020 on standard instances and achieves 0.048 on N=500 frustrated spin glasses where BP diverges entirely. 

---
# Validity Threats for Foundation Model Research 

**Authors**: Gunnar König, Martin Pawelczyk, Ulrike von Luxburg, Sebastian Bordt  

**Link**: [PDF](https://arxiv.org/pdf/2606.05029)  

**Abstract**: Controlled experiments are the backbone of machine learning research, but at the scale of modern foundation models, they have become prohibitively expensive. Instead, the community increasingly relies on research strategies that approximate the ideal experiment at a fraction of the cost: proxy experiments and scaling laws, observational studies with publicly available models, and single-run designs that leverage variation within individual training runs. In this work, we argue that there is no free lunch when approximating large-scale experiments on a compute budget. Specifically, savings in compute come at the cost of validity threats -- hidden and sometimes untestable assumptions that, when violated, can invalidate research claims. To help navigate such threats, we propose an evaluation framework that casts foundation model research as a causal inference problem. Within this framework, we evaluate different research strategies through four types of validity adapted from the empirical social sciences -- statistical, internal, external, and construct validity. We find that each strategy comes with a characteristic validity profile: proxy experiments trade external and construct validity for statistical and internal validity; observational studies face confounding and effect heterogeneity; and single-run designs are strained by interference between treated units. This analysis reveals several validity threats that have received insufficient attention in the literature. Overall, our evaluation framework provides researchers with a practical toolkit for scrutinizing validity threats in foundation model research~designs. 

---
# M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks 

**Authors**: Jie Huang, Ruixun Liu, Sirui Sun, Xinyi Yang, Yin Li, Yixin Zhu, Yiwu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2606.05008)  

**Abstract**: As multi-modal models advance towards long-form video understanding, memory emerges as a critical capability. Despite substantial efforts in developing video datasets and benchmarks, existing works primarily focus on perception and reasoning, without systematically evaluating memory: what models retain, how faithfully information is preserved, and how robust memory remains under interference. To address this gap, we introduce M$^3$Eval, the first comprehensive evaluation framework and benchmark for probing different memory dimensions in multi-modal models. Grounded in cognitive psychology, our design features carefully constructed tasks that isolate key aspects of memory. Leveraging M$^3$Eval, we conduct extensive experiments across representative multi-modal models, revealing consistent weaknesses and distinctive behaviors. We find that models struggle to maintain disentangled representations when processing parallel video streams, exhibit interference patterns differing substantially from those observed in human memory, ground memory sources more reliably in the spatial domain than the temporal domain, and demonstrate limited symbolic memory. Collectively, our benchmark provides a valuable resource for future research, while our findings highlight memory as a fundamental yet underexplored capability and offer insights for designing more effective memory mechanisms in multi-modal models. Our code and dataset are available at this https URL. 

---
# Clinical Assistant for Remote Engagement Link (CARE-link): A Web-Based Electronic Health Records Software for Managing Diabetes 

**Authors**: Prince Ebenezer Adjei, Joshua Teye Tettey, Toufiq Musah, Audrey Agbeve, John Amuasi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04952)  

**Abstract**: CARE-link is an open-source, web-based clinical support platform designed to improve the management of gestational diabetes by linking clinicians and patients through an LLM-mediated workflow. The system aggregates patient-generated data outside the hospital, summarizes relevant clinical information, and delivers context-aware decision support to clinicians. For patients, CARE-link provides clear explanations of management plans and delivers timely lifestyle guidance through a WhatsApp interface. The integrated dual-facing design aims to promote continuous monitoring, support individualized care, and reduce the burden of in-clinic follow-ups. Built with a modular architecture, the platform can be adapted to other chronic conditions requiring longitudinal tracking and behavioral support. CARE-link has the potential to enhance clinical oversight, promote patient compliance, and strengthen continuity of care particularly in resource-constrained settings. 

---
# Data Attribution in Large Language Models via Bidirectional Gradient Optimization 

**Authors**: Frédéric Berdoz, Luca A. Lanzendörfer, Kaan Bayraktar, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2606.04928)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse applications, raising critical questions for governance, accountability, and data provenance. Understanding which training data most influenced a model's output remains a fundamental open problem. We address this challenge through training data attribution (TDA) for auto-regressive LLMs by expanding upon the inverse formulation: How would training data be affected if the model had seen the generated output during training? Our method perturbs the base model using bidirectional gradient optimization (gradient ascent and descent) on a generated text sample and measures the resulting change in loss across training samples. Our framework supports attribution at arbitrary data granularity, enabling both factual and stylistic attribution. We evaluate our method against baselines on pretrained models with known datasets, and show that it outperforms previous work on influence metrics, thereby enhancing model interpretability, an essential requirement for accountable AI systems. 

---
# Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning 

**Authors**: Xuekang Wang, Zhuoyuan Hao, Shuo Hou, Hao Peng, Juanzi Li, Xiaozhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04923)  

**Abstract**: Rubric-based reinforcement learning (RL) uses an LLM-as-a-Judge (LaaJ) to score model outputs according to rubrics as rewards. However, policy models may exploit latent biases in the judge, leading to reward hacking and ineffective or unsafe training outcomes. In real-world rubric-based RL, such hacking behaviors are often subtle and entangled with multiple judge biases, making them difficult to analyze, detect, and mitigate. In this paper, we introduce CHERRL, a controllable hacking environment for rubric-based RL. By injecting known biases into LaaJ, CHERRL enables stable reproduction of reward hacking, explicit observation of reward divergence, and precise identification of hacking onset. This provides a clean experimental testbed for studying the mechanisms and mitigations of reward hacking in rubric-based RL. To demonstrate its utility, we analyze different judge biases from the perspectives of discoverability and exploitability, and explore an agent-based system for automatically detecting reward hacking onset from training logs. The code and environment are publicly available at this https URL. 

---
# BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routine 

**Authors**: Yang Liu, Jiajin Zhang, Danyang Tu, Yaojun Hu, Jiao Qu, Jiuyu Zhang, Yu Shi, Wei Fang, Shi Gu, Ling Zhang, Yingda Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.04911)  

**Abstract**: Breast cancer remains a leading cause of cancer-related mortality among women. Its clinical management requires multimodal reasoning across a clinical workflow that spans \textit{screening}, \textit{diagnosis} and \textit{treatment planning}, where each stage involves distinct imaging modalities, task objectives, and reasoning patterns. However, constrained by data scarcity and model versatility, existing medical MLLMs are typically evaluated on isolated modalities or narrow task families, limiting their ability to support workflow-level clinical reasoning. In this work, we first introduce \textbf{BreastStage}, a workflow-aligned breast imaging instruction corpus comprising 1.86M instruction-following pairs curated from 17 sub-datasets across 5 imaging modalities and 136 task templates. Its held-out split, \textbf{BreastStage-Bench}, provides a comprehensive benchmark for evaluating multimodal reasoning across the breast cancer care continuum. Building on this corpus, we propose \textbf{BreastGPT}, a unified MLLM equipped with a dual-branch visual encoder and concept-preserving token compression to bridge the scale gap between standard radiology and gigapixel pathology. On BreastStage-Bench, BreastGPT achieves 75.66\% closed-ended accuracy and 89.92\% open-ended score, outperforming both general-purpose and medical-specific MLLMs across clinical stages and task formats. These results suggest that workflow-aligned data and cross-scale visual modeling are critical for clinically grounded medical MLLMs. All data, code, and model checkpoints are released at this https URL. 

---
# BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration 

**Authors**: Yung-Yu Shih, Shang-Yu Su, Tzu-I Ho, Dongzhe Wang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04909)  

**Abstract**: E-commerce platforms in emerging markets often operate with underdeveloped product catalogs that contain only category taxonomies but lack structured attribute schemas. This absence of fine-grained product attributes limits search capabilities -- preventing faceted filtering, degrading query understanding, and weakening semantic representations used by search systems. We present BEATS, a human-in-the-loop LLM framework for bootstrapping product attribute taxonomies entirely from scratch. Our approach extends a multi-stage LLM generation pipeline with two critical production stages: (1) proactive quality checking by model developers to filter erroneous outputs, and (2) human annotation by domain-expert local staff to validate generated attributes. The framework operates iteratively -- prompts at each generation stage are refined based on quality check observations and annotator feedback across successive rounds, progressively improving attribute quality. Once the attribute taxonomy is established, we employ LLMs to perform structured attribute tagging on individual product items, enriching their contextual representations. The enriched catalog directly benefits multiple components of the search system: enabling granular attribute-based filtering, providing structured features for ranking models, and improving semantic representations for dense retrieval. We validate the generated taxonomy by training dense retrieval models on attribute-enriched product data, demonstrating consistent improvements over baselines using original catalog information. Our system has been deployed at Rakuten Taiwan, enriching 9 major categories spanning 2,694 sub-categories with 67,277 generated attributes, and over 5.4 million products have been tagged with the generated attributes, with plans to enrich the entire product catalog. 

---
# MusaCoder: Native GPU Kernel Generation with Full-Stack Training on Moore Threads GPU 

**Authors**: Kun Cheng, Songshuo Lu, Sicong Liao, Tankun Li, Yafei Zhang, Dong Yang, Qiheng Lv, Hua Wang, Zhi Chen, Yaohua Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04847)  

**Abstract**: Native GPU kernel generation turns high-level tensor programs into executable, efficient low-level code. Existing Large Language Models (LLMs) struggle with this task, while execution-based reinforcement learning suffers from sparse rewards, reward hacking, and training instability. We present MusaCoder, a full-stack training framework for native GPU kernel generation on CUDA and MUSA backends. MusaCoder combines progressive kernel-oriented data synthesis, diversity-preserving rejection fine-tuning, and execution-feedback Reinforcement Learning (RL) through MooreEval, a distributed verifier and reward environment. To stabilize RL, MusaCoder introduces PrimeEcho for first-turn-anchored multi-turn rewards, Buffered Dynamic Retry for recovering signals from all-failed hard samples, and MirrorPop for off-policy sequence filtering. Experiments on KernelBench and a MUSA-ported variant show that MusaCoder outperforms strong open-source and proprietary baselines in both correctness and empirical speedup, with the 9B model matching or exceeding frontier closed-source models and the 27B model establishing a new state of the art. These results demonstrate not only the effectiveness of full-stack execution-feedback training for native kernel generation, but also the capability of Moore Threads GPUs to support the complete LLM post-training stack, providing a practical foundation for large-model training and optimization on emerging accelerators. 

---
# R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2606.04823)  

**Abstract**: Large language models (LLMs) are fluent on open-ended tasks, yet in agentic settings, where a system must plan, use tools, and act over extended horizons, fluency does not ensure reliable delivery. We trace this gap to three coupled structural failures: errors propagate without localization, worst-case perturbations go unevaluated, and accumulated knowledge is never invalidated. We argue these share a root cause: abductive, counterfactual, meta-inductive, corrective, and inductive reasoning pull a shared context in incompatible directions. We introduce Reflective Adversarial Pareto Search (R-APS), to our knowledge the first method addressing all three failures jointly via reasoning-mode decomposition, allocating each reasoning mode its own context and orchestrating interaction across three timescales: staged compositional reasoning with a typed validation critic (failure localization), sensitivity-guided counterfactual stress-testing as a first-class Pareto objective (robustness), and meta-inductive rule extraction with explicit invalidation (persistent memory). R-APS requires no fine-tuning and operates on a frozen LLM purely via structured protocol design. We evaluate on planar mechanism synthesis (robotics, prosthetics, mechanical design), with every candidate checked by a kinematic solver. On 32 target trajectories, R-APS delivers robustness certificates 3.5x tighter than uniform-perturbation baselines, 46% faster iterations-to-first-admission, and 2.1x Chamfer-distance reduction over Enum+GA while jointly controlling bar-count and worst-case robustness. Small 4B reasoning-specialized models prove competitive with general-purpose 70B backbones inside the protocol, suggesting structured protocols can partially offset model scale. 

---
# BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization 

**Authors**: Saket Reddy, Ke Yang, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2606.04807)  

**Abstract**: Mitigating social bias in Large Language Models (LLMs) presents a distinct alignment challenge: unlike verifiable tasks, bias lacks a single ground truth, creating a high-variance, subjective reward landscape. Previous preference-based fine-tuning methods have major trade-offs: Direct Preference Optimization (DPO) is limited by the lack of exploration inherent in offline training, while Proximal Policy Optimization (PPO) can lead to training instability due to potentially unreliable critic estimates. In this paper, we propose BiasGRPO, a framework using Group Relative Policy Optimization (GRPO) to stabilize alignment by normalizing rewards across a group of sampled completions. By substituting the value function with a group-relative baseline, our approach reduces instability while maintaining the exploration benefits of online training. We find that BiasGRPO outperforms DPO and PPO across multiple benchmarks, indicating its effectiveness. To adapt GRPO, we synthetically extend a dataset spanning multiple domains and contexts. We also create and release a custom bias reward model that effectively guides generation while being highly compute-efficient and avoiding knowledge degradation, providing a valuable resource that can be seamlessly integrated into multi-objective RLHF pipelines. 

---
# Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories 

**Authors**: Kyungmin Park, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.04778)  

**Abstract**: Safety-aligned Large Language Models (LLMs) remain vulnerable to interventions during inference that redirect generation toward harmful outputs. Recent work attributes this to shallow safety, where alignment concentrates in the first few output tokens. We show that shallow safety is a special case of a broader inference-time vulnerability, in which short token injections at any generation step can substantially alter subsequent safety behavior. We also find that a model's alignment with refusal directions in its hidden states does not predict its robustness to such injection, revealing that internal state alone does not determine generation behavior under perturbation. To address this, we align models directly on generation trajectories constructed by simulating mid-sequence perturbation, and show that this improves robustness to mid-sequence injection and generalizes to attacks that exploit early-token generation. Our work argues that robust safety alignment requires training on the generation process itself, not only its outputs. 

---
# NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models 

**Authors**: Yong Cao, Chuqiao Li, Xianghui Xie, Gerard Pons-Moll, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2606.04773)  

**Abstract**: Reliable evaluation of human motion understanding is fundamental to advancing embodied AI, robotics, and animation. However, existing benchmarks suffer from coarse semantic granularity, undifferentiated difficulty, limited annotation quality, and pervasive answer ambiguity, leaving them unable to diagnose where current models fail. To bridge this gap, we introduce NextMotionQA, a comprehensive benchmark that leverages vision-language models (VLMs) for semi-automated, expert-verified dataset. NextMotionQA features three complementary tasks: multiple-choice question answering, video captioning, and fine-grained error correction. Each task is systematically structured across three core semantic axes and stratified into three task complexity levels. Our extensive evaluation of twelve representative VLMs uncovers critical capability gaps and weakness that remain invisible under conventional, single-task evaluations. In a complementary direction, recent work has begun using VLMs as judges for text-to-motion evaluation; we ask whether they show the same degradation under harder tasks. We find that VLMs align strongly with expert ratings on coarse criteria (Cohen's \kappa=0.70) but break down on fine-grained, part-level judgment (\kappa=0.10), validating the paradigm in its strong regime while clarifying its limits. 

---
# Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms 

**Authors**: Jiashu Yao, Heyan Huang, Daiqing Wu, Wangke Chen, Huaxi Ai, Haoyu Wen, Zeming Liu, Yuhang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04701)  

**Abstract**: GUI agents today assume a static screen, where the world is frozen between two actions. However, real interfaces such as short-video applications violate this assumption, as their content keeps playing, and a competent user must decide what to watch and for how long. We formalize this task as Living-Screen-Native GUI agents and introduce LivingScreen, the first benchmark instantiating it on short-video platforms, with a faithful browser-based environment, a three-tier task suite, and metrics that jointly score accuracy and information efficiency. Evaluating extensive frontier models, we find that none reaches the human cost-accuracy performance, and that their dominant failure mode is over- and under-observation, pointing to observation control as a missing capability axis for future GUI agents. All data and code will be available at this https URL. 

---
# Read What You Hear: Reference-Free Hypotheses Evaluation with Acoustic Discrepancy 

**Authors**: Zhihan Li, Hankun Wang, Yiwei Guo, Bohan Li, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04680)  

**Abstract**: Automatic speech recognition systems commonly rely on reference transcriptions for evaluation, while reference-free approaches often depend on internal confidence estimation or auxiliary language models. We propose READ (Reference-free Hypothesis Evaluation with Acoustic Discrepancy), a novel metric that evaluates ASR hypotheses directly from the speech signal. READ emphasizes the acoustic grounding of hypotheses. It uses a pretrained auto-regressive TTS model to compute the conditional likelihood of speech tokens given a text hypothesis, to measure fine-grained acoustic discrepancy between speech and text. Without additional training, READ can be applied for hypothesis refinement. Experiments show that READ correlates with specific recognition errors and improves ASR outputs, achieving up to 20\% relative error rate reduction, with particularly strong gains under noisy conditions. 

---
# VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation 

**Authors**: Teqi Hao, Yuxuan Fu, Xiaoyu Tan, Shaojie Shi, Bohao Lv, Yinghui Xu, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04632)  

**Abstract**: Mechanical ventilation for Acute Respiratory Distress Syndrome (ARDS) requires balancing competing physiological goals, including oxygenation, lung protection, and acid-base homeostasis. However, current data-driven methods, especially those imitating retrospective Electronic Health Records (EHR), often suffer from imitation bias. They may capture superficial correlations from inconsistent clinical demonstrations, such as associating passive ventilator settings with survival because such settings are common in stable patients, and thus fail to generalize to volatile or out-of-distribution phenotypes. Standard Reinforcement Learning (RL) methods also struggle with the adversarial trade-offs of critical care and often produce opaque policies with limited clinical interpretability. To address these limitations, we introduce VentAgent, a hierarchical framework in which Large Language Models (LLMs) act as transparent arbitrators for mechanical ventilation. We reformulate ventilation control as a dynamic Multi-Objective Arbitration process rather than single-objective optimization. VentAgent decomposes decision-making into three interpretable stages: Perception, Planning, and Orchestration. By leveraging the semantic reasoning capabilities of LLMs, it synthesizes strategies from heterogeneous experts and resolves conflicting clinical priorities through an explicit coordination mechanism. Evaluations on a high-fidelity physiological simulator show that VentAgent outperforms state-of-the-art RL and classical control baselines. Moreover, it converts control decisions into human-readable reasoning chains, offering a safer, more interpretable, and adaptable paradigm for critical care automation. 

---
# Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization 

**Authors**: Heng Cao, Fan Zhang, Jian Yao, Yujie Zheng, Changlin Zhao, Lu Hao, Yuxuan Wei, Wangze Ni, Huaiyu Fu, Yuqian Sun, Xuyan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04547)  

**Abstract**: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, eliminating explicit prompt construction and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating that scalable LLM personalization can be achieved without explicit prompt construction or heavy per-user adapters. 

---
# Global Sketch-Based Watermarking for Diffusion Language Models 

**Authors**: Daniel Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04486)  

**Abstract**: Watermarking methods for language models have been studied extensively in the autoregressive setting, where tokens are generated sequentially. These works largely focus on local-context schemes that perturb the next token's distribution as a function of its preceding tokens. In diffusion language models, distributions over many unresolved positions are jointly sampled, allowing additive statistics of the entire sequence to be tractable during generation. We propose a watermark for masked diffusion language models that controls a global, vector-valued sketch representation of the text. Compared to context-dependent watermarking, the sketch formulation decouples detection from the local contexts seen during generation, resulting in an order-agnostic statistic and a watermarking rule which does not manifest as a simple token bias. We analyze the distortion, soundness, and robustness properties of the method. 

---
# Evaluating Reasoning Fidelity in Visual Text Generation 

**Authors**: Jiajun Hong, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04479)  

**Abstract**: Recent text-to-image (T2I) models can render highly legible and well-structured text within images, enabling applications including document generation and slide generation. However, it remains unclear whether such systems faithfully preserve reasoning ability when complex solutions must be expressed directly through rendered text, or whether they merely imitate surface-level patterns. We investigate this question by evaluating reasoning fidelity in visual text generation, where models must express complete reasoning processes as images. Our evaluation includes long text rendering, factual knowledge probing, context understanding, and multi-step reasoning. Across these settings, we find that current T2I models frequently produce semantic errors, logical inconsistencies, and incorrect intermediate steps, even when the rendered text appears visually clear. These failures contrast with the strong reasoning performance of text-only models on the same tasks. Our findings reveal a substantial gap between visual text generation and procedural reasoning, motivating more reliable visual text reasoning. 

---
# Token Rankings are Unforgeable Language Model Signatures 

**Authors**: Matthew Finlayson, Andreas Grivas, Xiang Ren, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2606.04459)  

**Abstract**: Language model parameters are known to impose unique (to each model) geometric constraints on their logit outputs, which serves as a signature that identifies the model, but also leaks the model's final layer parameters when an API distributes logits. We investigate more restrictive APIs that expose token rankings (i.e., their ordering by probability, but not the probability values) and find that rankings also constitute a signature: every model has a unique set of feasible top-$k$ rankings for sufficiently large $k$. Furthermore, the ranking signature is the first known (polynomially) unforgeable signature, since finding a model with the same set of feasible rankings is NP-hard. On the security front, we find that token rankings are already sufficient to approximately steal the final layer of the model, similar to logits, though the approximation is too coarse to forge the signature, and can be effectively countered by restricting the API to top-$k$ tokens with sufficiently small $k$. Since the top-$k$ required to present the model signature is generally smaller than the $k$ required to prevent stealing, it is possible for an API to present an unforgeable signature without leaking model parameters. 

---
# The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development? 

**Authors**: Xinyu Lu, Tianshu Wang, Pengbo Wang, zujie wen, Zhiqiang Zhang, Jun Zhou, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.04455)  

**Abstract**: Current AI benchmarks evaluate agents on task execution within human-designed workflows. These evaluations fundamentally fail to measure a critical next-level capability: whether models can autonomously develop agent systems. We introduce the Meta-Agent Challenge (MAC), an evaluation framework designed to test the capacity of frontier models for autonomous agent development. Specifically, a code agent (the meta-agent) is given a sandboxed environment, an evaluation API, and a time limitation to iteratively program an agent artifact that maximizes performance on a held-out test set across five domains. To ensure evaluation integrity, this framework is secured by multi-layer defenses against reward hacking. Leveraging this framework, we demonstrate that meta-agents rarely match human-engineered baseline policies, and the few that do are dominated by proprietary frontier models. Moreover, the design process exhibits high variance, and high optimization pressure surfaces emergent adversarial behaviors like ground-truth exfiltration-highlighting critical deficits in both robustness and model alignment. Ultimately, MAC provides a rigorous, open-source benchmark for autonomous AI research and development, offering an empirical proxy for evaluating recursive self-improvement. Benchmark is publicly available at: this https URL. 

---
# Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation 

**Authors**: Saroj Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.04435)  

**Abstract**: Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment. 

---
# Stateful Visual Encoders for Vision-Language Models 

**Authors**: Zirui Wang, Junwei Yu, Adam Yala, David M. Chan, Joseph E. Gonzalez, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2606.04433)  

**Abstract**: Vision-language models (VLMs) are increasingly used in multi-image, multi-turn agentic settings where decisions depend on visual changes. However, in existing open-weight VLMs, visual comparisons happen only inside the language model, while the visual encoder itself remains stateless: each image is encoded independently, without access to the prior visual context. As a result, small but task-critical changes may be attenuated before the language model has a chance to compare them, especially when those changes do not affect the high-level semantics of the scene. We introduce a Stateful Visual Encoder, which conditions each visual representation on prior visual features. Under supervised finetuning, VLMs equipped with stateful encoders achieve consistent improvements on controlled tasks involving cross-image spatial aggregation, multi-object visual differencing, and visual trajectory behavior cloning. These improvements are consistent across input resolutions, language model sizes, and VLM backbones. Finally, we validate our model on real-world tasks, including longitudinal radiology, fine-grained image comparison, and remote sensing, where stateful encoders consistently improve generalist VLM baselines and can match or surpass specialized models in selected domains. Project page: this https URL 

---
# CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding 

**Authors**: Eugene Kwek, Feng Liu, Rui Zhang, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.04418)  

**Abstract**: Neural audio codecs are a key component of speech processing pipelines, compressing audio into discrete tokens for downstream modeling. However, existing codecs struggle to balance reconstruction quality with token efficiency, often encoding perceptually irrelevant information such as background noise and recording artifacts at the expense of linguistically and acoustically meaningful content. We reframe audio tokenization as a selective information bottleneck problem and propose CleanCodec, a denoising audio codec which learns to encode only perceptually important features and discard imperceptible information. At just 12.5 tokens per second, CleanCodec achieves state-of-the-art tokenization efficiency, substantially outperforming existing codecs in speaker similarity and speech intelligibility. Evaluations on downstream text-to-speech and voice conversion tasks further demonstrate improved performance and up to 17x faster inference, highlighting significant efficiency gains. 

---
# Physics-Informed Neural Network Modeling of Biodegradable Contaminant Transport through GCL/SL Composite Liners 

**Authors**: Dong Li, Yapeng Cao, Haiping Zhao, Shutong Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.04392)  

**Abstract**: This study develops a two-domain physics-informed neural network framework for contaminant transport through a GCL/SL composite liner system, in which the thin GCL layer is treated using a steady-state advection-dispersion-biodegradation formulation and the underlying soil liner is modeled as a transient transport domain. Two formulations are evaluated against analytical and finite-element reference solutions under different leachate-head conditions: a standard PINN with soft constraint enforcement (Std-PINN) and a hard-constrained PINN (H-PINN), in which selected boundary and initial conditions are embedded directly into the trial solutions. The Std-PINN captures the overall breakthrough behavior but shows larger errors during the early transport stage, particularly under higher leachate heads where advective transport becomes more pronounced. The H-PINN reduces the optimization burden associated with penalty-based constraint enforcement and provides more accurate and stable concentration predictions, lowering the MAE from approximately 0.058-0.067 for the Std-PINN to about 0.011-0.023 for the H-PINN, while reducing the MRE from approximately 9.10%-19.16% to about 2.08%-3.14%. Parametric analyses confirm that the H-PINN with the tanh activation function and an optimized network structure provides the best predictive accuracy. The H-PINN is further extended to inverse modeling for identifying the SL degradation half-life from limited concentration observations, showing reliable convergence toward prescribed values and acceptable robustness under low-to-moderate observation noise. 

---
# Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic 

**Authors**: Keisuke Watanabe, Kazuki Nakayashiki  

**Link**: [PDF](https://arxiv.org/pdf/2606.04362)  

**Abstract**: Large language model (LLM) "answer engines" such as ChatGPT now send measurable referral traffic to the open web, and a practice analogous to search engine optimization, here called Answer Engine Optimization (AEO), has emerged. Public AEO success stories typically quote large raw growth multiples, but raw referral growth is confounded by the rapid platform-level growth of the answer engines themselves. We report a longitudinal field study on a single high-traffic domain (this http URL) whose corpus of hundreds of thousands of YouTube question-and-answer pages received a defined bundle of AEO interventions in January 2026 (detailed in Section 4). Because the interventions were concentrated on one subset of the site, the untreated remainder of the same domain acts as a contemporaneous control that absorbs the platform tailwind. Using first-party analytics and server logs rather than probabilistic third-party estimators, we find: (1) raw growth is dominated by the platform tailwind: on monthly aggregates total ChatGPT referrals grew 5.7x while untreated pages on the same domain grew 3.5x over the same window; (2) an interrupted time-series model on the weekly treated/control ratio estimates a discrete, intervention-aligned level increase of 1.82x (95% CI 1.31-2.54, HAC p=0.001), robust across engagement-filtered traffic (2.27x) and alternative specifications; (3) however, a conservative placebo-in-time permutation test yields p=0.16, so the effect is suggestive, not conclusive, given a short and noisy pre-period; and (4) Google organic clicks to treated pages did not fall beyond the ambient site-wide trend and indexation was preserved, consistent with the SEO-protection rule. The methodological message, separating treatment from platform tailwind with an on-domain control, matters more than any single multiple, and implies that headline AEO multiples substantially overstate causal effect. 

---
# Video2LoRA: Parametric Video Internalization for Vision-Language Models 

**Authors**: Manan Suri, Sarvesh Baskar, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2606.04351)  

**Abstract**: Processing video in vision-language models is expensive: each frame occupies hundreds of tokens, and inference cost scales with every frame and every repeated query. We introduce Video2LoRA, a method for parametric video internalization. A perceiver hypernetwork reads the intermediate representations produced layer-by-layer as a frozen VLM encodes a video, and generates a Low-Rank Adaptation (LoRA) adapter in a single forward pass. Unlike standard LoRA fine-tuning, which requires iterative gradient updates, Video2LoRA predicts these weights directly from the video. Trained for SmolVLM2 500M and 2.2B on video summarization and captioning, Video2LoRA enables the same frozen VLM to answer queries from the adapter alone, with zero visual tokens in its context at query time. Video2LoRA is statistically non-inferior and equivalent to direct video-in-context inference across all five captioning benchmarks at both model scales, and across seven of eight video question answering benchmark-scale pairings. Although trained only on 12 frames at 384px, it remains stable up to 1,024 frames and 1024px, where direct video-in-context inference often degenerates. Across this sweep, it reduces answer-time visual-token load by up to 1,500x and query TTFT by 6-80x, while preserving video-faithful outputs. We also find that independently generated adapters for non-overlapping video segments can compose in rank space, suggesting a path toward chunked long-video internalization. 

---
# Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling 

**Authors**: Yifan Wang, Jinyi Mu, Mayank Jobanputra, Yu Wang, Ji-Ung Lee, Soyoung Oh, Isabel Valera, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.04284)  

**Abstract**: Preference modeling plays a central role in reinforcement learning from human feedback (RLHF), enabling large language models (LLMs) to align with human values. However, most existing approaches assume a universal reward function, neglecting the diversity and heterogeneity of human preferences. To address this limitation without additional annotation costs, recent work has proposed learning multiple preference components from binary data and combining them to model individual preferences. Nevertheless, these components often fail to capture coherent and disentangled patterns, limiting their interpretability and effectiveness for personalization. In this work, we propose a sparse Mixture-of-Experts (MoE) reward model that encourages sparse routing and expert diversity during training on binary preference data. Across controlled and real-world experiments, sparse MoE learns interpretable routing patterns and specialized experts. It also improves test-time personalization, and post-adaptation shifts in expert weights provide a qualitative lens for analyzing how the model adapts to personalized preferences. 

---
# Can Generalist Agents Automate Data Curation? 

**Authors**: Feiyang Kang, Hanze Li, Adam Nguyen, Mahavir Dabas, Jiaqi W. Ma, Frederic Sala, Dawn Song, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.04261)  

**Abstract**: Curating training data is among the most consequential yet labor-intensive parts of modern AI development: practitioners iteratively propose, implement, evaluate, and revise data policies against noisy benchmark feedback. We ask whether generalist coding agents can automate this data-curation loop. We introduce *Curation-Bench*, an agent-centric benchmark that fixes the model, training recipe, and evaluation suite while giving agents command-line access to inspect data, implement policies, submit them to a fixed training/evaluation pipeline, and revise. In a vision-language instruction-tuning instantiation, out-of-the-box agents reach strong published data-selection baselines within ten iterations. However, trajectory analysis reveals a persistent *execution-research gap*: agents mainly tune local policy variants rather than explore new policy families, even when given strategy guides and paper references. Scaffolds requiring each iteration to cite, instantiate, and adapt a prior method shift agents toward method-guided exploration. The scaffolded agent autonomously composes -- without human design input -- a data-selection policy that outperforms strong published baselines at one-tenth their data budget. Overall, current agents can run the curation loop, but reliable data research requires scaffolded method adaptation, not open-ended prompting alone. Code and benchmark are open-sourced. 

---
# StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Luyao Shi, Ehsan Degan, Vandana Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.04246)  

**Abstract**: Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward modeling (PRM), and retrieval-augmented fine-tuning (RAFT) to enhance both the functional correctness and reasoning fidelity of LLM-based RTL code generation. StepPRM-RTL constructs stepwise reasoning trajectories from canonical solutions, where each step contains a rationale and incremental code modification. A Process Reward Model (PRM) evaluates intermediate steps, providing dense feedback that guides reinforcement-style updates during RAFT fine-tuning. Monte Carlo Tree Search (MCTS) explores alternative reasoning paths, enriching the training dataset with high-quality trajectories. This integration of stepwise and outcome-aware rewards allows the model to learn both how and why to construct correct RTL, improving long-horizon reasoning beyond standard supervised or outcome-based training. Experimental evaluation on benchmark Verilog and VHDL datasets demonstrates that StepPRM-RTL outperforms the best prior methods by over 10\% in functional correctness and reasoning fidelity metrics. Ablation studies confirm that the combination of PRM-guided rewards and stepwise trajectory exploration is key to its performance. StepPRM-RTL generalizes across RTL languages and provides a scalable framework for high-fidelity, interpretable code generation, establishing a new standard for LLM-assisted hardware design automation. 

---
# VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark 

**Authors**: Amirhossein Dabiriaghdam, Shayan Vassef, Mohammadreza Bakhtiari, Yasamin Medghalchi, Ilker Hacihaliloglu, Mesrob Ohannessian, Lele Wang, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2606.04244)  

**Abstract**: Multimodal large language models are increasingly capable of complex reasoning, yet their performance often degrades when they must externalize a problem through a tool and then reason over the tool's output, specifically when they rely on visual aids. This gap is especially important because real engineering and scientific workflows often rely on visualization tools for analysis, validation, and decision-making. To study this discrepancy, we introduce VAMPS (Visual-Assisted Mathematical Problem Solving), a benchmark for graph-assisted mathematics. VAMPS contains 1,168 multimodal, bilingual multiple-choice question-answer pairs drawn from Iranian University Entrance Exam algebra and calculus problems and expanded with human-reviewed LLM-generated synthetic variants, all selected so that plotting provides a natural solution strategy by revealing intersections, extrema, asymptotes, etc. Designed for both benchmarking and diagnosis, VAMPS goes beyond prior multimodal benchmarks that primarily evaluate reasoning over fixed visual inputs by testing whether a model can benefit from constructing a useful graph and grounding its answer in the resulting visualization. Overall, we found that across a diverse set of models, direct analytical solving surprisingly outperforms tool-enabled visual solving, even on problems where plotting is a natural strategy. 

---
# Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1) 

**Authors**: Jingbiao Mei  

**Link**: [PDF](https://arxiv.org/pdf/2606.04240)  

**Abstract**: Retrieval over visually-rich documents, pages that interleave text with figures, tables, and charts, is essential for multimodal retrieval-augmented generation, yet most retrievers still discard the visual channel. The \emph{Multimodal Document Retrieval Challenge}, Track~1 of the MIR Challenge at the first EReL@MIR workshop, co-located with The Web Conference 2025, asks participants to build a \emph{single} retrieval system that handles two complementary regimes: closed-set document page retrieval within long documents from a text query (MMDocIR), and open-domain retrieval of Wikipedia-style passages from an image or image-plus-text query (M2KR). Systems are ranked by the macro-average of mean Recall@$\{1,3,5\}$ over the two tasks. The challenge drew 455 entrants and 586 submissions across 22 teams. This report describes the challenge design, datasets, and evaluation protocol; reports the final standings; and analyses the three winning teams' systems. All three build on decoder-based Multimodal-LLM embedders from the Qwen2-VL family rather than on CLIP-style encoders, and differ chiefly in whether they reach the top through fine-tuned ensembles, training-free multi-route fusion with a strong vision-language re-ranker, or zero-shot late interaction. The training-free system finished within $0.1$ point of the fine-tuned winner. 

---
# DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities 

**Authors**: Sajad Ebrahimi, Nima Jamali, Bardia Shirsalimian, Kelly McConvey, Wentao Zhang, Jalehsadat Mahdavimoghaddam, Maksym Taranukhin, Maura Grossman, Vered Shwartz, Yuntian Deng, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2606.04205)  

**Abstract**: The growing popularity and capacity of generative models have eroded the distinction between human and machine-generated content, motivating a growing body of work on detection across text, images, and audio. Most available detectors are either commercial software or, if open-source, come with incompatible codebases with bespoke preprocessing, evaluation protocols, and evaluation metrics, which make their adoption, fair comparison, and reproduction quite difficult. To address this critical gap, we introduce DetectZoo, a first-of-its-kind, extensible toolkit designed to provide a unified interface for AI-generated content detection across text, audio, and image modalities. DetectZoo standardizes the complete empirical pipeline, from data ingestion and preprocessing to model assessment, offering researchers a cohesive framework to benchmark state-of-the-art detectors systematically. By integrating diverse public datasets and baseline detection algorithms under a single, unified API, our toolkit facilitates rigorous and reproducible evaluation. DetectZoo provides reference implementations of 61 detectors, native loaders for 22 benchmark datasets, and a standardized evaluation pipeline that reports multiple metrics through a common interface. Each detector is self-contained yet accessible through the same interface, automatically caches pretrained weights, and reproduces the original published results. DetectZoo lowers the barrier to entry for multi-modal AI forensics, enabling researchers to identify performance gaps across domains and accelerating the development of robust, generalizable detection techniques. The open-source repository and comprehensive documentation are publicly available at this https URL, and the package can be installed via pip install detectzoo. 

---
# Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions 

**Authors**: Aliakbar Mehdizadeh, Martin Hilbert  

**Link**: [PDF](https://arxiv.org/pdf/2606.04197)  

**Abstract**: How much should an LLM agent remember, and how should multi-agent systems be connected when trying to reach consensus? We show these two design choices interact in a way that flips the sign of memory's effect on coordination. Across 432 simulation runs of a networked Naming Game on eight fixed 16-agent topologies, we vary memory depth and network structure. Longer memory slows the time to reach steady state in decentralized networks but accelerates it in centralized ones; the same parameter pushes the system in opposite directions depending on topology. Critically, "faster settling" in centralized networks means locking in to a fragmented plateau more quickly, not reaching system-wide consensus, which can be used to generate diverging opinions. We further document a memory-mediated speed-unity trade-off: centralized networks consistently preserve more competing conventions than decentralized networks, but their settling speed depends sharply on memory. At the agent level, within-network analyses show that high-betweenness bridges suffer a brokerage penalty while agents in locally clustered neighborhoods achieve higher coordination success. Finally, in search of analytically tractable generative mechanisms, we find that agents' choices are well captured by Fictitious Play, indicating belief-based rather than reward-based adaptation. The practical implication: memory depth and communication topology should be co-designed, not optimized in isolation. 

---
# Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval 

**Authors**: Christian Lysenstøen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04194)  

**Abstract**: Retrieving the few past turns that answer a new query across long multi-session histories is the retrieval bottleneck behind long-term conversational memory (LoCoMo, LongMemEval). Recent concurrent work, Nano-Memory, shows that scoring a session by the maximum query-turn similarity (late interaction, "Turn Isolation Retrieval") beats mean-pooled session embeddings. We do not claim that effect; we replicate it and ask what a training-free, CPU-only retrieval stage should add around it. We report four findings. (1) Fuse: score-level fusion of the late-interaction dense score with BM25, under a single leave-one-conversation-out weight, adds +8.8 to +17.2 points of LoCoMo Hit@1 over late interaction alone across six encoders (all p<1e-4), reaching Hit@1 0.752 / NDCG@5 0.829 (e5-large-v2), +11.2 pp over BM25. (2) An off-the-shelf web-search cross-encoder reranker over the fused top-10 hurts here, degrading Hit@1 by 6.9 pp (one reranker, one configuration). (3) A pooling-operator ablation shows top-k late interaction matches max-similarity, but a naive smooth-max (log-sum-exp) collapses for half the encoders. (4) The late-minus-early gap is large for all six encoders and tends to be larger for larger ones, while the marginal fusion gain shrinks; on LongMemEval-S, a lexical regime where BM25 saturates, the net fusion gain over BM25 is small and not significant. A per-category analysis frames the gain as a division of labor: dense late interaction helps most on multi-hop and temporal questions but trails BM25 on adversarial ones. The contribution is a controlled, reproducible account of a strong training-free retrieval recipe, not the late-interaction retriever itself (Nano-Memory's). We make no claim to a complete memory architecture; this is a retrieval-stage study. 

---
# SocialCoach: Personalized Social Skill Learning with RL-based Agentic Tutoring and Practice 

**Authors**: Tianfu Wang, Max Xiong, Jianxun Lian, Hongyuan Zhu, Zhengyu Hu, Yuxuan Lei, Linxiao Gong, Xiaofang Li, Peiting Tsai, Nicholas Jing Yuan, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04155)  

**Abstract**: Social skills such as negotiation and leadership are crucial for personal and professional success in today's interconnected world. However, scalable and effective training remains a significant challenge due to the scarcity of expert coaching. In this paper, we introduce SocialCoach, a holistic LLM-powered agentic tutoring system for personalized social skill development at scale. First, SocialCoach automatically constructs a pedagogically-grounded, theory-to-practice knowledge corpus from diverse expert sources, leveraging a multi-agent pipeline. Second, to personalize the learning journey, it employs an adaptive practice scheduling module that follows a prescription-retrieval-adaptation process. To maximize the long-term learning experience while overcoming the cold-start problem, this policy is optimized within a learner simulation environment through reinforcement learning. Finally, SocialCoach integrates immersive, goal-driven practice, causality-driven proficiency assessment and knowledge-grounded, reflective tutoring to help address the knowing-doing gap. We deploy it in our product, EQoach, and conduct extensive experiments. The results show that SocialCoach improves simulated pathway quality and judge-rated tutoring quality over baseline approaches, while early user feedback indicates strong perceived engagement and usefulness. These findings suggest a practical architecture for personalized and gamified pedagogical platforms on soft skill learning. 

---
# Large Language Models Hack Rewards, and Society 

**Authors**: Wei Liu, Xinyi Mou, Hanqi Yan, Zhongyu Wei, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2606.04075)  

**Abstract**: Reinforcement learning (RL) has become a dominant post-training paradigm, enabling large language models (LLMs) to learn from rewards. We observe that societal regulations are structurally similar to reward functions. They define measurable outcomes, thresholds, and exceptions, while often leaving institutional intent only partially specified. We hypothesise that the RL training process may exploit these gaps and therefore ask whether models' well-known tendency to hack reward functions during RL can scale into a more consequential failure mode named societal hacking: discovering loopholes in the rules society runs on. To study this phenomenon, we introduce SocioHack, a sandbox of 72 societal environments, and find that within these environments, reward hacking naturally emerges and leads to regulatory loophole discovery. Models learn to hack the social rules and generate strategies that remain technically compliant while defeating regulatory intent, and current LLM safeguards provide only limited mitigation. Therefore, collecting in-the-wild feedback for model training requires greater caution, and we need a next-generation post-training paradigm for safely iterating LLMs in real society.= 

---
# Covert Influence Between Language Models 

**Authors**: Avidan Shah, Jay Chooi, Jinghua Ou, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.04071)  

**Abstract**: As language models increasingly consume one another's outputs, covert influence -- a phenomenon where a sender's payload (the behavioral disposition it is conditioned to propagate) transfers to a receiver through carriers undetectable by humans -- becomes a growing risk. We characterize this risk across three interfaces: supervised fine-tuning, on-policy distillation, and in-context learning, and find that they vary in the scale of influence achievable without leaving behind human-visible traces. Using inference-time per-sample attribution scores, we study covert influence across all three interfaces with the ability to select carriers that amplify training-time influence, unlocking payload transfers that prior work could not achieve. We further provide evidence that covert influence with natural-language carriers is a distinct phenomenon from prior studies using number carriers, as the latter is more resistant to human detection and less portable across model families. Together, these results suggest that the risk surface for covert influence is broader than previously recognized, and we study pointwise attribution scoring methods as a tool to investigate and mitigate it. 

---
# Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation 

**Authors**: Boyuan Xiao, Bohong Chen, Yumeng Li, Ji Feng, Yao-Xiang Ding, Kun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04046)  

**Abstract**: In embodied vision-language decision making tasks such as robotic manipulation and navigation, Vision-Language and Vision-Language-Action Models (VLMs & VLAs) are powerful tools with different benefits: VLMs are better at long-term planning, while VLAs are better at reactive control. However, their performance is limited by the same perceptual bottleneck: visual hallucinations arise due to the models' inability to distinguish task-relevant objects from distractors. In principle, accurate identification and focus on critical objects while filtering out irrelevant ones is the key to break this limitation. A straightforward solution is one-step focus: directly attending to essential objects. However, this approach proves ineffective because effective focus inherently requires deep scene understanding. To this end, we propose SceneDiver, a coarse-to-fine focus plan generation method for VLMs leveraging their long-term planning abilities, that first constructs a holistic scene graph to establish initial comprehension, then progressively decomposes the task into simpler sub-problems through an iterative cycle of recognition, understanding, and analysis. To enable reactive control, we also design a lightweight adapter for distilling the deliberate focus ability into VLAs. Evaluations on standard embodied AI benchmarks confirm that our method substantially reduces visual hallucinations for both VLMs and VLAs, while preserving computational efficiency in tasks requiring fast execution. Our code and data are released at: this https URL. 

---
# Do Transformers Need Three Projections? Systematic Study of QKV Variants 

**Authors**: Ali Kayyam, Anusha Madan Gopal, M Anthony Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04032)  

**Abstract**: Transformers have become the standard solution for various AI tasks, with the query, key, and value (QKV) attention formulation playing a central role. However, the individual contribution of these three projections and the impact of omitting some remain poorly understood. We systematically evaluate three projection sharing constraints: a) Q-K=V (shared key-value), b) Q=K-V (shared query-key), and c) Q=K=V (single projection). The last two variants produce symmetric attention maps; to address this, we also explore asymmetric attention via 2D positional encodings. Through experiments spanning synthetic tasks, vision (MNIST, CIFAR, TinyImageNet, anomaly), and language modeling (300M and 1.2B parameter models on 10B tokens), we discovered that our transformers perform on par or occasionally better than the QKV transformer. In language modeling, Q-K=V projection sharing achieves 50% KV cache reduction with only 3.1% perplexity degradation. Crucially, projection sharing is complementary to head sharing (GQA/MQA): combining Q-K=V with GQA-4 yields 87.5% cache reduction, while Q-K=V + MQA achieves 96.9%, enabling practical on-device inference. We show that Q-K=V preserves quality because keys and values can occupy similar representational spaces and attention operates in a low-rank regime, whereas Q=K-V breaks attention directionality. Our results systematically characterize projection sharing as an underexplored instance of weight tying in attention, with direct, quantifiable inference memory benefits, particularly valuable for edge deployment. The code is publicly available at this https URL 

---
