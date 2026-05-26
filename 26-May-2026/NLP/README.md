# Language Models Need Sleep 

**Authors**: Sangyun Lee, Sean McLeish, Tom Goldstein, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2605.26099)  

**Abstract**: Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning. 

---
# Automated Benchmark Auditing for AI Agents and Large Language Models 

**Authors**: Junlin Wang, Federico Bianchi, Shang Zhu, Fan Nie, Yongchan Kwon, Bhuwan Dhingra, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.26079)  

**Abstract**: Modern AI benchmarks operate at a complexity that outpaces traditional verification methods. Tasks authored by domain experts often contain implicit assumptions, incomplete environment specifications, and brittle evaluation logic that human annotation cannot reliably catch. We introduce Auto Benchmark Audit (ABA), an agentic framework that systematically audits individual benchmark tasks, uncovering issues such as hidden environment dependencies, specification gaps, and limited grading logic. We run ABA on a collection of frontier LLM benchmarks and previous NeurIPS publications, totaling 168 benchmarks across nine domains. Across this corpus, ABA identifies critical issues including ambiguous task design, execution environment conflicts, and incorrect ground truths in over 25.7% of the evaluated tasks. The precision of these automated audits is validated by expert review and independent third-party reports such as upstream PRs. Crucially, we demonstrate that these problematic tasks severely distorts capability assessments for agents and LLMs: filtering out these tasks with issues shifts model rankings and increases average performance on SWE-bench Verified and Terminal-Bench 2 by 9.9% and 9.6%, respectively. We release the agentic tool and all task annotations to support the future development of frontier benchmarks. 

---
# StakeBench: Evaluating Language Understanding Grounded in Market Commitment 

**Authors**: Yunhua Pei, Jingyu Hu, Yiwei Shi, Hongnan Ma, Weiru Liu, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.26074)  

**Abstract**: Existing financial NLP benchmarks often rely on labels supplied by outside observers, measuring how language is perceived rather than what speakers have committed to in the market. We introduce StakeBench, an evaluation framework for language understanding grounded in market commitment. StakeBench links 560,876 comments from 2,261 resolved markets to verified position, action, and market-odds records across Polymarket and Manifold. Supervision is derived from observable market behavior. Position sides, post-comment trading actions, and market-odds trajectories replace human annotation. Four diagnostic tasks test whether models detect market commitment, identify the revealed side, anticipate future action, and perform collective odds projection. Three commitment-aware metrics measure alignment with revealed preferences rather than perceived sentiment. Validity audits and explicit interpretation boundaries help distinguish observable commitment signals from latent belief and causal market-odds impact. Across 15 LLMs and 18 topics and platform settings, models partially recover position-side signals, with Directed Accuracy from 0.506 to 0.599, but show structural failures on later tasks. Ten of the fifteen models collapse to one or two action labels in future action anticipation, and no model consistently improves on the naive odds-direction baseline in collective odds projection. Model scale is not correlated with performance, finance-domain tuning does not improve revealed-side identification, and platform incentives strongly shape higher-order results. StakeBench is packaged with evaluation code and dataset under CC-BY 4.0. 

---
# WhoSaidIt: Human-LLM Collaborative Annotation for Text-Based Multilingual Speaker-Attribute Classification 

**Authors**: Lingyu Gao, Will Monroe, David Smith, Meghan Jemison, Jackie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.26070)  

**Abstract**: Annotating speaker attributes from text is inherently ambiguous, particularly in multilingual settings where demographic and social cues are implicit and culturally variable. We propose a human-large language model (LLM) collaborative re-annotation framework for stabilizing multilingual speaker-attribute labels under practical resource constraints. Starting from a noisy corpus, we use LLMs to surface recurring annotation rationales through iterative interaction with experts, and apply disagreement-focused sampling for targeted re-annotation. Using this framework, we construct WhoSaidIt, a multilingual dataset covering nine speaker-attribute labels. We quantify divergence between original and revised annotations, benchmark recent LLMs, and analyze the effect of explicit rationales on model behavior. Our results reveal substantial cross-lingual differences in annotation decisions and demonstrate both the strengths and limitations of LLMs in speaker-attribute classification. 

---
# When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges 

**Authors**: Parth Darshan, Abhishek Divekar  

**Link**: [PDF](https://arxiv.org/pdf/2605.26046)  

**Abstract**: Customizing an LLM judge to a specific task or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) doesn't apply to the multi-objective textual gradient setting. We test five decomposition modes of textual gradient optimizers by varying how much cross-task information the loss, gradient and optimizer LLMs share. In 6 of 10 configurations, we observe that optimization never improves over the initial prompt. Gradient specificity drops by 59% (from 9.0 to 3.7) when the gradient LLM processes multiple criteria jointly. Separately, we observe that naively combining per-task instructions into a single prompt degrades Spearman's rho by -5.3%. These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge customization using textual feedback. 

---
# Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals 

**Authors**: Federico Torrielli, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.26045)  

**Abstract**: Activation oracles aim to make the activations of other models legible to humans and yield promising results compared to white-box interpretability techniques. However, uncertainty quantification (UQ) for the natural-language outputs of such activation oracles is so far understudied. Here, we investigate 6 different methods for estimating the confidence of activation oracles and evaluate how well-calibrated their confidence scores are. Our experiments on 6,000 samples per oracle (varying verbalizer and context prompts) reveal that bootstrap mode frequency is the best-calibrated method among those tested (ECE 5.7% vs. 25.5% for the answer-word log-probability on Qwen3-8B; 10.3% vs. 13.1% on Qwen3.6-27B), and that the log-prob baseline can serve as a fast triage signal at a fraction of the cost.
Code and the patched trainer are available at this https URL. 

---
# Peak-Then-Collapse and the Four Interface Channels of Knowledge-Graph Tool Use 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.26037)  

**Abstract**: We test the standard RLVR tool-use recipe -- GRPO on Qwen2.5-7B-Instruct -- on a deliberately minimal knowledge-graph tool API: four Freebase navigation verbs over Complex WebQuestions. Under a self-verifiable retrieval reward, the policy's tool-grounded answer rate climbs from $3.8\%$ to $9.6\%$ over 250 steps, then collapses to $0\%$ within a single 50-step window -- a \emph{peak-then-collapse} pattern replicated across four seeds. Across seven reward designs, we find four recurring failure modes: adding denser or more targeted proxy rewards shifts the failure mode rather than eliminating it. We argue that a key difference from Python interpreters, web search, and JSON APIs is interface feedback: their failures often leak natural-language signal the model saw in pretraining. A Python traceback names the failing line; an empty Freebase result \texttt{[]} does not. Stripping away that surface exposes a degradation regime that same-family reward redesigns do not fix. A direct oracle ablation rules out relation selection: injecting gold relations at every retrieval call lifts exact-match accuracy by only $+0.20$~pp, and $95.4\%$ of retrieval-dependent errors are retrieval-composition failures rather than answer-extraction failures. As a mitigation, one-iteration self-distillation reaches $40.0\%$ EM at 7B and is capacity-invariant: doubling capacity to 14B improves EM by only $0.25$~pp, and initialization barely matters -- the ceiling appears interface-bound within the 7B--14B range tested. 

---
# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech 

**Authors**: Rez Samantha Z. Floresca, Edric Castel C. Hao, Hannah Grachiella Buñales, Chelsea Dominique E. Temprosa, Georgianna Z. Reyes, Kervin Gabriel L. Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.26007)  

**Abstract**: Dementia detection from spontaneous speech offers a scalable approach to cognitive screening, yet NLP systems remain predominantly English-centric. This limitation is especially acute in the Philippines, where Filipino-English code-switching is pervasive and no prior work has addressed NLP-based dementia detection. We present the first systematic evaluation of transformer-based dementia detection in Filipino speech and the first assessment of NeoBERT in a clinical NLP setting. To separate language from domain effects, we construct a parallel bilingual dataset of 4,000 DementiaBank-derived transcripts, with Filipino translations produced manually to preserve discourse-level markers of cognitive decline. We evaluate five model families, TF-IDF + LogReg, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog, under monolingual, zero-shot cross-lingual, and bilingual fine-tuning settings. We find that in-domain performance does not transfer across languages, with English-trained BERT dropping to Macro-F1 = 0.455 on Filipino, and that architectural modernization alone does not improve robustness. Bilingual fine-tuning, however, eliminates cross-lingual degradation across all transformer models, converging to Macro-F1 = 0.969-0.973. These results suggest that multilingual clinical NLP performance is driven primarily by linguistic coverage during training rather than model scale or architecture. 

---
# AI-Assisted Systematization for Evaluating GenAI Systems 

**Authors**: Dhruv Agarwal, Emily Sheng, Chad Atalla, Jean Garcia-Gathright, Hussein Mozannar, Hannah Washington, Alexandra Chouldechova, Solon Barocas, Hanna Wallach  

**Link**: [PDF](https://arxiv.org/pdf/2605.26001)  

**Abstract**: Evaluating generative AI (GenAI) systems is challenging because many targets of evaluation are broad, contested concepts, such as "reasoning," "fairness," or "creativity." When these concepts are left underspecified, it becomes unclear what should be measured or how evaluation results should be interpreted. This problem reflects a missing step: systematization, that is, moving from a broad background concept to an explicit, structured account of the concept in measurable terms. To help address the fact that systematization is cognitively demanding and resource-intensive, we investigate whether AI assistance can support this process. To enable AI-assisted systematization and assess its quality, we introduce a structured representation of a systematized concept, a concept spec, and a validation worksheet. We then develop two AI-assisted systematizers: a direct, zero-shot approach and a multi-agent approach that more closely mirrors manual systematization approaches from existing literature. We use these systematizers to produce concept specs for two concepts -- hate-based rhetoric and digital empathy -- and evaluate resulting concept specs on content validity and information recoverability. 

---
# What Makes a Medical Checker Trainable? Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA 

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25988)  

**Abstract**: Medical RAG needs evidence-grounded claims, so plugging a claim-level NLI checker into retrieval-augmented RL is intuitive. \textbf{We find that the checker's \emph{output distribution} during training, not its held-out accuracy, decides whether it provides trainable gradient.} We compare four NLI checker back-ends as process rewards inside a GRPO-trained medical RAG agent (Qwen2.5-7B, replicated on Qwen3-4B and Llama-3.1-8B) across four held-out medical QA benchmarks. Three diagnostic findings emerge. \textbf{(i)} Signal collapse is log-prob-specific: LLM log-probability scoring labels over 97\% of claims neutral -- collapsing the RL gradient to zero -- while a calibrated MedNLI classifier scores the same pairs non-degenerately. \textbf{(ii)} Moderate signal beats strong signal on answer quality: a strong proprietary checker triggers a three-step reward-hacking cascade -- ultra-short answers, search avoidance, language collapse -- so a moderate-signal local classifier trains a higher-quality model (\textbf{+12\% BERTScore over zero-shot, no GPT dependency}). \textbf{(iii)} Signal strength is policy-dependent: the same checker registers as moderate on one policy but strong on another without triggering the cascade end-state. We frame these as boundary conditions for verifier-as-reward systems. 

---
# SafeCtrl-RL: Inference-Time Adaptive Behaviour Control for LLM Dialogue via RL-Driven Prompt Optimisation 

**Authors**: Michael Orme, Yanchao Yu, Zhiyuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25984)  

**Abstract**: Ensuring safe and contextually appropriate behaviour in Large Language Models (LLMs) remains a critical challenge for real-world deployment. We present \textbf{SafeCtrl-RL}, an inference-time behavioural control framework that enables adaptive safety regulation without model retraining or parameter modification. The method formulates dialogue generation as a sequential decision process, where a reinforcement learning agent dynamically selects prompt adjustment strategies based on contextual feedback. This allows unsafe behaviours to be suppressed through iterative refinement, which we conceptualise as inference-time behavioural unlearning. Evaluated across multiple LLMs and unsafe dialogue scenarios, SafeCtrl-RL consistently improves safety and response quality, outperforms existing prompt-based optimisation methods, and achieves favourable performance--efficiency trade-offs. **Warning: This paper may contain examples of harmful language, and reader discretion is recommended. 

---
# When Do LLM Agents Treat Surface Noise Differently from Semantic Noise? A 68-Cell Measurement Study with a Held-Out Trace-Level Validation 

**Authors**: Liyun Zhang, Jiayi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.25981)  

**Abstract**: We document an empirical phenomenon in chain-of-thought and ReAct agents driven by ten large language models from seven architecture families: meaning-bearing perturbations (e.g., paraphrase, synonym) alter final answers more often than presentation perturbations (e.g., formatting, reordering) of comparable severity. Across 68 cells spanning GSM8K, MATH, and HotpotQA (1,530 originals and $\sim$11,150 variants), the inconsistency gap averages +19.69 pp after severity matching (paired $t=9.58$, $p<0.0001$), with 64/68 cells positive. The gap survives four severity-proxy audits and remains significant when excluding qwen models (+11.10 pp, $p<0.0001$). Several stress tests fail honestly: cluster-bootstrap significance disappears under stricter assumptions, tractability contrasts do not replicate, cross-architecture generator swaps break per-cell rankings, and a second LLM judge yields only moderate agreement ($\kappa=0.50$).
We then validate the headline effect on a fully held-out 11th model (qwen2.5-14B-Instruct; 1,800 trajectories) and re-test a pre-registered capability$\times$tractability partition, observing a small but positive held-out effect (3/4 cells positive; pooled Welch $t=3.81$, $p=9.6\times10^{-4}$). Using held-out trajectories, we probe four trace-level mechanism signals. Two prior mechanism claims fail to replicate and are explicitly retracted. Two new probes instead support a \emph{stealth-divergence} picture: semantic perturbations often preserve the first action but induce divergence in intermediate reasoning from later steps onward, accompanied by slightly deeper trajectories. We position this as a measurement contribution with held-out replication and a partial trace-level account of how semantic perturbations propagate through agent reasoning. Code, perturbation corpus, raw trajectories, and analysis scripts are released anonymously for review. 

---
# Creative Quality Alignment: Expert Tacit Knowledge Transfer via Chain-of-Thought Fine-Tuning 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25977)  

**Abstract**: This paper provides an empirical implementation of the creative quality metric proposed in Calibrated Surprise (Zou & Xu, 2026a). The question this paper addresses is: does this mathematical claim hold at the engineering level?
To make the answer as general as possible, we deliberately choose the strictest engineering conditions: low data cost and a small base model. Training data comes from approximately 100 expert chain-of-thought (CoT) annotations produced by the BC Protocol (Zou & Xu, 2026b).
We also identify a data bias: most publicly available alignment datasets are skewed toward craft-related knowledge, while audience modeling and reality-logic coverage are systematically weak.
We use the term Creative Quality Alignment (CQA) to describe this class of engineering methods. We also offer a supporting theoretical observation: in an LLM with a single conditional distribution architecture, calibrating the appreciation side automatically transfers to the generation side via architectural duality. This is the structural reason why ~100 CoT examples are sufficient -- not a purely empirical observation like LIMA (Zhou et al., 2023). 

---
# Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents 

**Authors**: Haoyi Hu, Qirong Lyu, Xianghan Kong, Weiwen Liu, Jianghao Lin, Zixuan Guo, Yan Xu, Yasheng Wang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25971)  

**Abstract**: While AI agents demonstrate remarkable capabilities in reasoning and tool use, they remain fundamentally reactive: they compute responses only after explicit user prompts. This paradigm ignores a critical opportunity: the idle time between interactions is largely wasted, leaving agents unable to prepare for future user needs. To bridge this gap, we introduce ProAct, a proactive agent architecture that leverages idle-time compute to anticipate and fulfill likely upcoming user needs. By analyzing evolving dialogue history together with persistent memory, ProAct predicts upcoming needs and iteratively acquires information, allowing the agent to resolve knowledge gaps and prepare evidence before the user initiates a this http URL rigorously evaluate proactive capabilities, we also introduce ProActEval, a comprehensive benchmark comprising 200 scenarios across 40 domains, featuring predictable need chains and diverse user cognitive profiles. Empirical results demonstrate significant advantages over reactive baselines. ProAct accelerates task completion by reducing required turns by 14.8%, decreases user effort by 11.7%, and cuts hallucination rates by 28.1% on ProActEval. Furthermore, MemBench evaluations confirm that ProAct achieves state-of-the-art reflective accuracy, underscoring its sustained and robust performance. 

---
# Triplet-Block Diffusion RWKV 

**Authors**: Ke Lin, Yiyang Luo, Zhaolong Su, Yunya Song, Anyi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25969)  

**Abstract**: Causal Transformer language models suffer from strictly sequential decoding and a quadratic per-step attention cost. While linear-time causal models and discrete diffusion models each address these weaknesses, their integration remains inherently inconsistent: diffusion requires bidirectional attention, while causal models are unidirectional. To unify these architectures, we propose $B^3D-RWKV$, a diffusion RWKV variant that integrates the model's $O(L)$ inference efficiency with parallel, bidirectional discrete-diffusion through a \emph{triplet-block layout} method. $B^3D-RWKV-7.2B$ reaches comparable accuracy on an 8-task suite versus existing models while significantly outperforming baselines in decoding throughput with an average of $\mathbf{1.6\times}$ speedup. 

---
# PolyGnosis 2.0: Enhancing LLM Reasoning via Agentic Harness Engineering for Polymarket and OSINT Insight Extraction 

**Authors**: Daren Wang, Hong Xu, Jiawen Xian  

**Link**: [PDF](https://arxiv.org/pdf/2605.25958)  

**Abstract**: This paper introduces PolyGnosis 2.0, a pioneering multi-agent architecture designed to extract predictive intelligence by synthesizing Polymarket anomaly signals with global Open Source Intelligence (OSINT) streams, specifically Global Database of Events, Language, and Tone (GDELT). We define and target "Perspective Mismatches", the narrative divergence between Polymarket sentiment and global media flows, as high-alpha trading signals. Moving beyond generic agentic superiority, we rigorously quantify the efficacy of "Harness Engineering" techniques, including reflection loops, tool-calling, divide-and-conquer partitioning (D&C), and chain-of-thought (CoT), within high-noise financial domains. Our empirical evaluation against human-expert benchmarks reveals that while structural partitioning is mandatory for multi-dimensional alignment, unconstrained terminal reflection actively induces logical drift. Furthermore, we identify a pervasive "consensus bias" across all agent configurations during narrative reasoning, necessitating deterministic validation. Ultimately, we isolate a Pareto-optimal configuration that achieves professional-grade analytical precision while minimizing latency and token overhead, providing a robust blueprint for autonomous intelligence in prediction markets. 

---
# QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25955)  

**Abstract**: Large language models (LLMs) face a dual challenge in creative capability evaluation: existing benchmarks (e.g., Story Cloze Test, HellaSwag) measure models' discriminative ability over narrative continuation using multiple-choice recognition paradigms, rather than directly measuring creative generation capability; rubric-based scoring and LLM-as-Judge methods rely on subjective dimension assessment or natural language model outputs, and cannot provide objective, automated scoring mechanisms.
This paper proposes QUIET (Quality Understanding via Interlocked Evaluation Testing), a diagnostic benchmark for LLM creative capability based on multi-blank cascaded story cloze. QUIET sets N blanks (10-20) in a story with complete structure, with each blank accompanied by an explicit content constraint, and cascade dependency relationships between blanks -- the content filled into earlier blanks constrains the feasible solution space for later blanks. The evaluated model (or human participants) fills all blanks in open-ended generation mode; the results are scored by an information-theoretic automated scoring protocol without human grading.
The scoring protocol directly operationalizes the "calibrated surprise" theoretical framework (Zou & Xu, 2026a). For each blank k, a composite score is computed: score = satisfy * (1 + lambda * surprise), where lambda = 1.0. Here, "satisfy" measures how well the blank filling satisfies the content constraint (objective logical reasoning judgment, not subjective aesthetic scoring), and "surprise" measures the degree of surprise given that the constraint is satisfied. Creative answers that do not satisfy the constraint score zero; answers that satisfy the constraint but are mediocre score low; answers that satisfy the constraint and are surprising score high. 

---
# Thaka at KSAA-2026 Task 2: Regularized Fine-Tuning for Arabic Speech Diacritization 

**Authors**: Meshal Alamr, Hassan Alqaeri, Abdullah Aldahlawi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25928)  

**Abstract**: We describe the winning system for Task 2 of the KSAA-2026 Shared Task on Arabic Speech Dictation with Automatic Diacritization. The task requires producing fully diacritized Arabic text from speech audio and undiacritized transcripts, with only 2,327 training samples available and no external data permitted. Our system fine-tunes CATT-Whisper, a character-level multimodal model combining a pretrained CATT text encoder with a frozen Whisper speech encoder. The key to our approach is training regularization: R-Drop consistency regularization, Optuna-optimized hyperparameters with high weight decay, and Focal Loss. At inference, we average 200 stochastic forward passes across four model checkpoints using Monte Carlo Dropout at the softmax probability level. The system achieves 23.26% WER on the primary leaderboard metric (with case endings, including no-diacritic positions), placing 1st among all participants. 

---
# Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT 

**Authors**: Duy Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25924)  

**Abstract**: Recent automated essay scoring (AES) studies increasingly use pretrained transformer models, but these models are usually pretrained on general-domain English and may under-represent second-language learner writing. This study investigates whether domain-adaptive continued pretraining (DAPT) on the EFCAMDAT learner corpus improves transformer-based AES for English proficiency tests. We apply DAPT to three transformer encoders and evaluate them on FCE and IELTS in both in-domain scoring and few-shot cross-dataset transfer. Full-corpus DAPT produces mixed results across models, datasets, and metrics. Further analyses suggest that these mixed effects are partly explained by mismatches in proficiency, genre, and communicative purpose between EFCAMDAT and the downstream datasets. A proficiency-based ablation shows that targeted DAPT using CEFR-aligned subsets improves downstream scoring more reliably than full-corpus DAPT, especially for FCE with B1--B2 data. However, these gains do not consistently improve cross-dataset transfer. Overall, the findings suggest that continued pretraining on a learner-writing corpus can benefit in-domain AES for English assessment when the pretraining data is sufficiently aligned with the downstream assessment settings. However, it does not automatically improve transferability across different English proficiency test datasets. 

---
# Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning 

**Authors**: Wei Fan, Yining Zhou, Mufan Zhang, Yanbing Weng, Yiran HU, Tianshi Zheng, Baixuan Xu, Chunyang Li, Jianhui Yang, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.25920)  

**Abstract**: While large language models (LLMs) augmented with agentic search capabilities show promise for legal reasoning, they overlook a fundamental constraint that applicable law must match the temporal context of each case, as retroactive application of statutes violates core legal principles and leads to erroneous conclusions. Our observations reveal that current legal LLMs suffer from temporal bias anchored to their training cutoff, while search agents rarely incorporate temporal constraints into queries, and that web search alone cannot provide the precise statute and precedent citations that legal reasoning demands. To address these challenges, we propose LegalSearch-R1, an end-to-end reinforcement learning framework that pairs local statute RAG for precise article matching with online web search for broader legal knowledge, trained on temporally-indexed data spanning multiple amendment periods to enforce temporal consistency. Extensive experiments on our benchmark covering 13 legal tasks demonstrate that our 7B-parameter agent outperforms state-of-the-art deep research frameworks and specialized legal LLMs by 12.9% to 29.8%, surpasses baselines by 57.7% to 80.3% on temporal consistency, and exhibits robust out-of-domain generalization. The code and data are available at this https URL. 

---
# Universal Activation Verbalizer: A Unified Framework for Cross-Model Activation Explanation 

**Authors**: Haiyan Zhao, Zirui He, Guanchu Wang, Ali Payani, Yingcong Li, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.25903)  

**Abstract**: Activation verbalization explains hidden representations in natural language, but existing methods are mostly limited to self-explanation, where each model explains only its own activations. We introduce Universal Activation Verbalizer (UAV), a framework that uses a shared decoder to explain activations from heterogeneous donor models. UAV learns a lightweight adapter that converts donor activations into soft tokens in decoder's embedding space, and further supports adapter-only transfer by reusing a frozen decoder-side LoRA while training only a new adapter for another donor. Across classification, fact retrieval, and gist summarization, UAV remains competitive with strong self-explanation baselines while enabling cross-model verbalization across model families and scales. Ablations show that decoder-side tuning mainly improves task behavior, whereas the adapter provides the activation-grounded factual and semantic information needed for faithful explanations. 

---
# Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express 

**Authors**: Ziyi Ding, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25891)  

**Abstract**: We find a mismatch between what large language models encode about a causal question and what they answer. On anti-commonsense CLadder items, a fixed linear probe recovers the evidence-supported answer from the model's hidden state (accuracy approximately 0.97), while the spoken Yes/No reverts to the commonsense one (accuracy approximately 0.5). We call this approximately +0.5 gap Causal Tongue-Tie: a wrong Yes/No decomposes into two separable failure modes: no internal signal versus a signal the verbal interface cannot say. The implication cuts both ways for output-only causal benchmarks: a benchmark "correct" need not mean the model has understood, and a benchmark "wrong" need not mean it cannot. Sweeping claims about whether LLMs can do causal reasoning, drawn from a single accuracy number, deserve a second look. 

---
# Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory Representation 

**Authors**: Zhengda Jin, Bingbing Wang, Jing Li, Ruifeng Xu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25869)  

**Abstract**: Long-term memory is essential for persistent LLM agents, yet prevailing architectures store historical interactions as unstructured, flat text. This unconstrained storage induces provenance-role collapse, a critical failure mode where agents suffer from source-monitoring errors. To resolve this cognitive vulnerability at the architectural level, we propose MemIR, a typed Memory Intermediate Representation that operationalizes source monitoring as a structural constraint. MemIR writes long-term memory into grounded atoms that separate raw evidence, retrieval cues, and truth-bearing claims, with factual authorization restricted to supported claim atoms. It then applies multi-route atomic projection and provenance-scoped utilization to transform heterogeneous retrieval hits into claim-centered candidate bundles and a normalized fact interface for answer generation. Experiments on LoCoMo and BEAM-100K demonstrate that MemIR consistently outperforms existing memory baselines, especially on tasks requiring source tracking, temporal grounding, and aggregation of fragmented evidence. 

---
# TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning 

**Authors**: Muyu Pan, Shu Zhao, Nan Zhang, Philip Shin, Varun Parekh, Vijaykrishnan Narayanan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25850)  

**Abstract**: This paper investigates large language model (LLM) abstention learning, specifically using ternary reward, which incentivize truthfulness in large language models. This paper extends that idea by moving from a ternary reward to a Trajectory-Informed advantage reweighting, dynamically re-weights the abstention reward during Group Relative Policy Optimization (GRPO) training. The objective of this work focuses on abstention learning instead of improving truthfulness, serving as an exploration into hallucination reduction. The novelty of this paper lies in methodological innovation, advantage re-weighting, and benchmark selection. Leveraging GRPO's multiple trajectories as a natural abstention signal, this method uses a reward signal to explore knowledge boundaries and encourage consistency. By demonstrating that trajectories can be used as a confidence indicator of the policy relative to the query, they are then used to dynamically calculate the abstention advantage. AbstentionBench is used as the evaluation benchmark, as this work aims to contribute to the field of abstention learning. All datasets on the benchmark were tested against this method and various baselines. Empirical results demonstrate that TIAR achieves state-of-the-art abstention F1 scores across five of six evaluation categories, outperforming the static ternary baseline on 17 of 31 benchmark datasets while fully preserving baseline accuracy. 

---
# On the Limits of Model Merging for Multilinguality in Pre-Training 

**Authors**: Seth Aycock, Fedor Vitiugin, Aleksandr Umnov, Christof Monz, Khalil Sima'an  

**Link**: [PDF](https://arxiv.org/pdf/2605.25846)  

**Abstract**: Endowing models with consistent multilingual performance can be achieved by mixing pre-training data, or post-training approaches such as language-specific model merging. In this work, we test whether merging can be applied to monolingually pre-trained models. We conduct a controlled study on the efficacy of mixed, merged, and monolingual pre-training setups. We find that while monolingual pre-training results in strong in-language performance, merging any combination of monolingual models leads to performance collapse due to interference. Our analysis suggests representational similarity is a prerequisite for model merging. We therefore conclude that the flexibility of merging in fine-tuning does not extend trivially to language-specific pre-training. 

---
# Clarify, Abstain or Answer? Strategising in Conversation with Belief-Augmented Generation 

**Authors**: Joris Baan, Wilker Aziz, Barbara Plank, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2605.25831)  

**Abstract**: Large language models (LLMs) define a distribution over text, which can be viewed as a probabilistic representation of uncertainty: sampling K responses yields a belief state - responses a model deems plausible. Existing work exploits this representation for narrow tasks like either decoding or selective prediction, and often requires manual interventions, not controlling generation directly. We propose Belief-Augmented Generation (BAG): grounding LLMs in their own belief state via the prompt and letting them reason over these K samples to decide on a conversational strategy: answer, clarify, or abstain. In a multi-turn ambiguous QA setting, we find that LLMs by default rarely clarify or abstain, ignoring uncertainty about the input or facts. BAG improves QA accuracy across six models and yields strategy decisions more faithful to the belief state than prompt-only baselines. Disentangling when to clarify from when to abstain, however, remains challenging. 

---
# Fine-Tuning Over Architectural Complexity: Broad-Coverage PII Detection on PIIBench with DeBERTa 

**Authors**: Pritesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25816)  

**Abstract**: Personally identifiable information (PII) detection systems are frequently trained within narrow source or domain boundaries, limiting coverage when deployed on heterogeneous text. We study model fine-tuning on a corrected multi-source PIIBench preparation spanning 82 retained entity types across ten source datasets. We evaluate three DeBERTa-based approaches: direct token classification fine-tuning, a source-conditioned hierarchical model (SC+H), and a three-phase curriculum extension (SC+H+Curr). Against eight published comparator systems on a reproducible 5,000-record held-out subset (test_5k), direct fine-tuned DeBERTa achieves F1 0.6476, while SC+H and the curriculum variant achieve 0.5899 and 0.2772 respectively; the strongest published comparator reaches only 0.1723. Because validation initially favoured SC+H, we perform a final streamed evaluation on the complete 100,002-record held-out split. Direct fine-tuning remains superior, achieving F1 0.6455 versus 0.5894 for SC+H. Entity-level analysis shows that direct fine tuning wins 54 of 82 fine entity types and all ten coarse groups by support-weighted entity F1, while SC+H retains localised advantages on 28 types. The results indicate that diverse task-specific training data and a simple weighted cross-entropy objective contribute more to broad-coverage PII detection than the tested architectural and curriculum complexity. 

---
# Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution 

**Authors**: Hongtao Wang, Renchi Yang, Haoran Zheng, Xiangyu Ke  

**Link**: [PDF](https://arxiv.org/pdf/2605.25814)  

**Abstract**: Dirty entity resolution (ER), which identifies records referring to the same real-world entity from a single, messy dataset, is a fundamental task in data management and mining. However, the dominant blocking-matching-clustering paradigm for ER suffers from critical flaws. Its cascaded, decoupled workflow essentially produces a static, sparse graph plagued by missing edges (due to blocking failures) and noisy links (due to matching errors), causing error propagation and yielding suboptimal clusters, particularly when rigid transitivity is imposed in the clustering. We contend that matching and clustering are fundamentally synergistic, both optimizing for the construction of an ideal entity graph. Building upon this insight, we propose Alper, a unified framework that integrates these steps into an iterative probabilistic label propagation process over a global, evolving graph. Unlike disjoint blocking, Alper refines the graph structure and labels dynamically by adaptively integrating "weak but cheap" signals from graph propagation with "strong but expensive" LLM-based pairwise queries. For higher cost-effectiveness, we formulate the signal selection as a constrained optimization problem maximizing cumulative marginal gain under a query budget, solved via our greedy algorithm with provable theoretical guarantees. Our extensive experiments over eight benchmark datasets demonstrate that Alper is consistently superior to state-of-the-art cascaded pipelines. 

---
# Double Triangle Annotation: A Scalable Human-in-the-Loop Framework for High-Precision Historical Document Annotation 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.25781)  

**Abstract**: Evaluating structured-information extraction from historical documents at scale requires high-precision ground-truth annotations, yet traditional manual labeling is expensive and fully automated pipelines built on large language models are prone to hallucination. We propose Double Triangle Annotation, a two-layer human-in-the-loop framework that leverages cross-model consensus to automate the majority of annotation work while ensuring high-precision outputs. In the first layer, two architecturally independent Multimodal Large Language Models annotate each document in parallel; when they agree, the label is auto-accepted, and disagreements are routed to a human jury. A second layer cross-checks two such systems against each other, escalating residual conflicts to a domain expert. The framework rests on a single assumption -- error independence between models -- requires no distributional priors or task-specific calibration, and becomes more autonomous as model capability improves. On the Guides Rosenwald, a corpus of French medical directories spanning 1887-1906, the framework achieves a final Word Error Rate of 0.003. Applied at scale, model consensus auto-accepts over 85% of 13,595 fields. We release the resulting benchmark -- the first structured-extraction ground truth for the Rosenwald Guides -- to support future work on historical document processing. 

---
# StreamProfileBench: A Benchmark for Fine-Grained User Profile Inference in Real-World Streaming Scenarios 

**Authors**: Sizhe Wang, Feiyu Duan, Juelin Wang, Liwen Zhang, Feiyu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25758)  

**Abstract**: Large Language Models (LLMs) have reshaped user profiling, yet current evaluations mainly focus on static data snapshots. This paradigm overlooks the reality of personalized systems, where User-Generated Content (UGC) arrives continuously and fine-grained profile evolve rapidly. To bridge this gap, we introduce StreamProfileBench, a large-scale benchmark for fine-grained streaming user profiling. We formalize streaming user profiling as a continuous state maintenance task and curate a highly authentic dataset comprising over 120,000 UGC posts from 7,000+ real users across five diverse platforms. By leveraging the temporal correlation of user interests, we further propose a novel, annotation-free evaluation framework. Extensive experiments across 14 leading LLMs reveal that continuous profile updating remains an open challenge. Models exhibit a systemic conservative bias, over-retaining past interests while failing to recognize interest decay. Ablation experiments further validate the practical utility and necessity of the streaming paradigm. 

---
# Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains 

**Authors**: Hui Xie, Jie Liu, Ziyue Qiao, Joaquin Vanschore  

**Link**: [PDF](https://arxiv.org/pdf/2605.25745)  

**Abstract**: Explicit chain-of-thought (CoT) reasoning substantially improves the reasoning ability of large language models (LLMs), but incurs high inference cost due to lengthy autoregressive traces. Existing latent reasoning methods offer a promising alternative, yet they often treat reasoning as uniformly compressible, causing precision-critical intermediate steps to be overly compressed and thereby degrading reasoning accuracy. In this work, we propose Selective Latent Thinking (SLT), a framework that selectively compresses redundant reasoning spans into latent representations while preserving precision-critical spans as explicit CoT within the same reasoning trajectory. Specifically, SLT first uses a lightweight decoder to anticipate a short upcoming reasoning span, and then applies confidence-based gating to determine the longest span that can be reliably compressed. The accepted span is encoded into a compact latent representation to improve reasoning efficiency, while uncertain or precision-critical reasoning remains in explicit CoT form to preserve accuracy. To learn this selective compression policy, SLT adopts a three-stage training strategy that combines span-level latent compression, reliability-aware future reasoning prediction, and trajectory-level reinforcement learning to optimize the trade-off between answer correctness and reasoning cost. Extensive experiments across four mathematical reasoning benchmarks demonstrate that SLT achieves 22.7\% higher accuracy than latent reasoning baselines at comparable compression ratios, while reducing reasoning chain length by 58.4\% with only 2.8\% accuracy degradation compared to explicit CoT,Our code can be found in this https URL. 

---
# Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring 

**Authors**: Zhengyang Wang, Sanwoo Lee, Jiaxin Wang, Chenxi Miao, Weikang Li, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25731)  

**Abstract**: Multi-trait essay scoring aims to provide fine-grained evaluation of writing quality across multiple dimensions. However, how to effectively post-train autoregressive scoring models remains underexplored. In this paper, we propose Trait-Aware Policy Optimization (TAPO), a post-training framework tailored to autoregressive multi-trait scoring. Our method decomposes rewards along both the sample and trait dimensions, combining global scoring consistency, trait-level accuracy, format validity, and inter-trait dependency preservation. In addition, we enhance supervised fine-tuning with enhanced prompts, allowing the model to internalize trait semantics before preference optimization. Experiments across multiple backbone models show that our method consistently improves multi-trait scoring performance over supervised fine-tuning and scalar-reward optimization baselines, demonstrating the effectiveness and transferability of trait-aware post-training for essay scoring. 

---
# PowLU: An Activation Function for Stable Pre-Training of LLMs 

**Authors**: Peijie Jiang, Yuqi Feng, Cunyin Peng, Qian Zhao, Jia Liu, KunLong Chen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25704)  

**Abstract**: In contemporary large language models (LLMs), the swish-gated linear unit (SwiGLU) activation function is widely adopted to regulate the information flow and introduce non-linearity. For large positive inputs, SwiGLU approximates the quadratic function $x^2$, providing strong nonlinearity and expressive capacity. However, this property also causes numerical instability as the input or model scale increases, particularly in low-precision LLM training. The main reason is its approximate quadratic amplification, which enlarges the output range and exacerbates outliers. To address this issue, we propose a stable activation function, Power Linear Unit (PowLU), for large-scale LLM pre-training. Specifically, PowLU employs a rational power function to achieve adaptive nonlinearity, thereby improving representation ability and enabling stable training in spike regions. Moreover, we provide theoretical justification for several key properties of PowLU. Scaling law experiments confirm that the performance is consistent across model sizes, and further experimental results with the Ling architecture (7.9B and 124B total parameters) demonstrate that PowLU achieves competitive results against SwiGLU and SwiGLU-Clip in large-scale training of LLMs. In addition, the experimental results also show that PowLU effectively improves the scalability of the large-scale training of LLMs. 

---
# From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents 

**Authors**: Rongsheng Zhang, Ruofan Hu, Weijie Chen, Jiji Tang, Junnan Ren, Wanying Wu, Xunuoyan Chen, Tangjie Lv, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25693)  

**Abstract**: While role-playing agents excel in short-term interactions, long-term conversations overwhelm context windows, motivating external memory frameworks. Current systems typically rely on persona-agnostic summarization, which records facts without persona-specific interpretation, yielding generic responses that compromise persona fidelity. To bridge this gap, we introduce RoleMemo, a dataset featuring four reasoning tasks where the factual fragments must be interpreted through the persona to reach the correct answer. Evaluation on RoleMemo exposes critical limitations of persona-agnostic frameworks. We thus propose DualMem, which decouples memory into two streams: factual cognition and persona-conditioned insight. Trained through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), our framework with a 4B-parameter model outperforms zero-shot persona-agnostic frameworks powered by DeepSeek-V3.2 for sustained persona fidelity. Our resources are available at this https URL. 

---
# Testing the Deliteralization Hypothesis in Human and Machine Translation 

**Authors**: Malik Marmonier, Rachel Bawden, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2605.25686)  

**Abstract**: The recent shift from dedicated NMT systems to general-purpose LLMs has reshaped machine translation, with LLMs reported to produce more fluent, less literal output than their predecessors. We test whether this shift extends to the deliteralization hypothesis, the long-standing claim from translation studies that translations become progressively less literal as they are drafted and revised. Using the WMT24++ dataset, we compare the literality of human translations and post-editions to that of two NMT systems and six LLMs across 54 language pairs and three tasks: direct translation, iterative self-revision, and post-editing of human drafts. Literality is measured via a validated Synthetic Literality Index built from six heuristics. We find that (i) human translations remain significantly less literal than those of all tested MT systems, though recent LLMs narrow the gap; (ii) when prompted to iteratively revise their own output, LLMs deliteralize monotonically, providing the first evidence that the hypothesis applies natively to LLM generation; and (iii) as post-editors, LLMs invert the revision triggers of human post-editors, tolerating literal drafts and targeting idiomatic human formulations for revision. 

---
# Simulating Human Memory with Language Models 

**Authors**: Qihan Wang, Nicholas Tomlin, Michael Hu, Brian Dillon, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25680)  

**Abstract**: Language models are increasingly being deployed as user simulators, but their memory is far more reliable than that of real users. To measure this gap, we run a series of classic memory experiments from psychology on both humans and language models. Across tasks, we find that out-of-the-box language models exhibit better memory than humans, even when prompted to imitate human behavior. We then show that better prompting strategies and the use of a compactor can cause language models to forget content in a more human-like way. Using these methods, we show preliminary evidence that language models with human-like memory constraints can function as more effective user simulators in a downstream education task. Finally, we release human reference data and benchmarks to support future work on simulating human memory with language models. 

---
# Llamion Technical Report 

**Authors**: Kisu Yang, Yoonna Jang, Hyeonseok Moon, Hwanseok Jang, Taewoo Lee, Hyungjin Lee, Jeseung Lee, Juhyoung Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.25676)  

**Abstract**: We release Llamion, a family of 14B-parameter open-weight language models obtained by transforming Orion-14B into the standardized Llama-family architecture. The transformation is performed by Efficient Knowledge Preservation for Transformation (KEPT), a recipe that combines (i) Normal Parameter Mapping (NPM) for unchanged modules, (ii) Optimized Parameter Mapping (OPM), a training-free LayerNorm-to-RMSNorm initialization we prove optimal under the near-zero-mean activation regime induced by weight decay, and (iii) Cross-architecture Knowledge Distillation (XKD), an equal-size frozen-teacher distillation that aligns the converted model's outputs with the source model's on any reasonable input distribution. Llamion recovers Orion's behaviour on H6, MT-Bench, and KoMMLU with only ~123M tokens on a single A100 in four days; Llamion-Base reaches 66.87% on KoMMLU, exceeding the next-best entry of the Open Ko LLM Leaderboard by >7.0 absolute points at submission time. Capabilities entirely absent from the transfer corpus (Python programming and 200K-token context handling) survive the architectural transition intact. We release three checkpoints (Base, Chat, LongChat) that load with trust_remote_code=False in the Hugging Face Transformers library. 

---
# AutoSG: LLM-Driven Solver Generation Solely from Task Prompts for Expensive Optimization 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25658)  

**Abstract**: Expensive optimization tasks are ubiquitous in real-world applications, demanding highly specialized solvers. While LLM-driven automated solver generation shows promise, current paradigms face three critical issues when tackling expensive optimization: factual hallucinations due to deficient domain knowledge, the frequent dismantling of previously established locally optimal structures during refinement, and the prohibitive evaluation costs alongside restricted generalization caused by executing on training instances. To address these issues, we introduce AutoSG, a fully automated workflow directly translating natural language prompts into executable customized solvers. AutoSG features three core innovations: a retrieval-augmented solver generation module strictly grounding code in verified literature; a one-step self-refinement operator introducing task-specific improvements while preserving critical structural components; and an instance-free Elo-based LLM-as-a-Judge evaluation mechanism rapidly establishing global rankings. Extensive evaluations across diverse expensive optimization tasks confirm AutoSG significantly outperforms human-designed state-of-the-art frameworks and existing LLM-generated solvers. 

---
# A Two-Phase Stability Study of LLM Judges and Bar Council Examiners on Thai Bar-Exam Free-Form Essays 

**Authors**: Pawitsapak Akarajaradwong, Wuttikrai Lertprasertphakorn, Chompakorn Chaksangchaichot, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2605.25652)  

**Abstract**: Free-form legal essay evaluation in NLP treats expert inter-rater stability as a single ceiling number, and treats LLM-judge agreement with that ceiling as evidence of judge stability. We test both assumptions on the Thai bar examination through an identical-inputs protocol: three Bar Council-trained examiners (A, B, C) and a 26-LLM judge panel score the same 15 cross-graded answers from the same four inputs (question, official Bar Council grading regulation, gold answer, candidate answer). The headline finding is asymmetric. On 10 of 15 cells where the rubric prescribes both axes, all 29 raters converge in a tight band: panel agreement is universal. On the remaining 5 cells where the rubric does not prescribe how to grade a correct final answer that omits a decisive statutory citation, the human panel splits between two coherent readings (B/C majority at the upper rubric band, score $6$--$8$; A minority at the lower band, score $1$--$2$). The LLM judge population does not split symmetrically: 22 of 26 LLMs score in or near B/C's contested band, 3 sit in the regulation-silent middle gap, and only 1 (GPT-5.4 Nano) approaches A's band without consistently scoring within it. \emph{Zero LLMs in our 26-judge panel reproduce the minority human reading on the contested cells.} The B/C-direction cluster spans every model size, vendor, and price tier we tested. An instrumented three-LLM anchor sub-panel (Claude 4.6 Opus, Gemini 3.1 Pro, GPT-5.4 Pro) carries determinism probes, input ablations, and bootstrap CIs, and reaches anchor panel $\alpha = 0.77$ on the 15 cells against human-panel $\alpha = 0.36$. The high LLM-panel $\alpha$ reflects systematic convergence on the majority reading rather than balanced reproduction of both readings; a benchmark that selects its LLM judge by maximising agreement with a human reference panel will inherit this asymmetry by construction. 

---
# Iterate Until Retrieved: Factual Nugget Optimization for Discoverable Continual Corrections in Agentic RAG 

**Authors**: Moshe Hazoom, Gal Patel, Alon Talmor, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2605.25641)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems in complex B2B (business-to-business) settings may often receive free-form response feedback. Rather than generic feedback signals such as style, preference, or overall response quality, we focus on actionable factual corrections. We identify these instances and convert them into compact knowledge-base entries, which we call factual nuggets. We introduce Iterative Nugget Optimization (INO), an index-time optimization method that uses the production agentic RAG as a test harness: it creates an initial nugget, probes it with the triggering query and paraphrases, reflects over failed retrieval and answer traces, and revises the nugget until it is discoverable. We evaluate INO with two production B2B knowledge-assistance agents across multiple companies that use our system: a product support agent that answers questions over company-specific knowledge bases, and a support ticket agent that assists support engineers. INO consistently improves results over baselines in terms of discoverability and usage of factual corrections, in automated and human evaluations. 

---
# Reinforcement Learning from Denoising Feedback 

**Authors**: Qi He, Huan Chen, Ya Guo, Huijia Zhu, Yi R. Fung, Baojian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25638)  

**Abstract**: Policy loss estimation remains a fundamental and long-standing challenge in reinforcement learning (RL) for diffusion language models (dLLMs). We introduce Reinforcement Learning from Denoising Feedback (RLDF), a novel training paradigm that leverages feedback obtained from rollout and training processes to facilitate accurate and efficient policy loss estimation. To balance the trade-off between computational efficiency and estimation effectiveness, RLDF optimizes the model toward the clipped clean state $\hat{x}_0$ from intermediate noisy states $x_t$, combined with weighted timestep sampling over $t$. Extensive experiments demonstrate that RLDF achieves consistent and substantial improvements in both performance and generalizability across two representative dLLM architectures, LLaDA and Dream, on multiple reasoning benchmarks. Our work lays a principled foundation for scalable reinforcement learning in diffusion language models. We build Drift, a training framework for dLLMs, available at this https URL. 

---
# When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift 

**Authors**: Khoi Le, Tri Cao, Phong Nguyen, Cong-Duy Nguyen, Anh Tuan Luu, Miao Chunyan, See-Kiong Ng, Thong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25629)  

**Abstract**: Weak-to-strong (W2S) generalization is a promising framework for scalable oversight, yet existing evaluations often test students under matched train--test distributions. Therefore, we study W2S preference learning under zero-shot distribution shift and find that strong students trained on weak preference labels can appear successful in-distribution while failing to transfer across preference datasets. We provide evidence for a representational failure mode in which weak-supervised fine-tuning can pull the strong model toward source-domain features instead of maintaining broadly transferable preference representations. To mitigate this, we propose Representation Anchoring (Anchor), a simple yet effective regularizer that constrains excessive drift from the pretrained strong model's representation space during fine-tuning, while still allowing task-relevant adaptation. Across preference domains, datasets, and model families, Anchor consistently improves out-of-distribution transfer while maintaining competitive in-distribution performance. Together, our evaluation protocol, transfer-aware metrics, and method expose hidden brittleness in current W2S reward modeling and provide a practical path toward more robust preference transfer. 

---
# Beyond Literal Translation: Evaluating Cultural Effectiveness in Social Media UGC 

**Authors**: Linjuan Wu, Ruiqi Zhang, Xinze Lyu, Ye Guo, Daoxin Zhang, Zhe Xu, Yao Hu, Yixin Cao, Yongliang Shen, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25626)  

**Abstract**: Social media platforms enable large-scale cross-lingual communication, but translating user-generated content (UGC) remains challenging due to its informal style, cultural references, and interaction-based expressions. While recent LLMs have improved translation quality, existing benchmarks and metrics often fail to capture whether translations convey intended meaning and cultural resonance in real-world settings. In this work, we introduce CULTURE-MT, a benchmark for social media translation that focuses on both CULtural Transmission and UGC-specific emotion REsonance. CULTURE-MT consists of 1,002 UGC notes across 14 domains, categorized into four types based on culture-loaded symbols and linguistic style features. We also construct UGC-oriented training data to fine-tune Qwen3-8B and Qwen3-32B as baselines. We propose cultural effectiveness as a new evaluation criterion, focusing on expression accuracy and cultural adaptability. Testing 15 models, including the baselines, we find that traditional metrics fail to capture cultural effectiveness. We also observe that cultural effectiveness on base LLMs correlates with model size. Our work provides a comprehensive evaluation system for UGC translation models and will offer an open evaluation platform to advance research in this area. We release the CULTURE-MT benchmark and provide an online leaderboard where submitted translation results can be evaluated by our trained JUDGER. 

---
# DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning 

**Authors**: Guochao Jiang, Jingyi Song, Guofeng Quan, Chuzhan Hao, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25604)  

**Abstract**: Reinforcement Learning has become a standard paradigm for aligning Large Language Models with human intent and task requirements. While Group Relative Policy Optimization offers an efficient, value-model-free alternative to Proximal Policy Optimization, adapting it to real-world multi-reward settings remains challenging. Standard scalarization practices, such as Reward Combination and Advantage Combination, suffer from significant drawbacks: Reward Combination frequently generates advantages with excessively large squared magnitudes that lead to training instability, while Advantage Combination relies on static hyperparameters and ignores cross-objective correlations. To address these limitations, we propose Dynamic Variance-adaptive Advantage Optimization (DVAO), which dynamically adjusts combination weights based on the empirical reward variance of each objective within a rollout group, effectively up-weighting objectives with a stronger learning signal while suppressing noisy ones. We mathematically prove that DVAO maintains bounded advantage magnitudes for stable training and introduces a self-adaptive cross-objective regularization mechanism. Extensive experiments on mathematical reasoning and tool-use benchmarks using Qwen3 and Qwen2.5 models demonstrate that DVAO significantly outperforms baseline methods, achieving a superior multi-objective Pareto frontier and robust training stability. 

---
# Toward a Benchmark for Controllable Simulation of Imperfect Students with Large Language Models 

**Authors**: Alexander Apartsin, Omri Sason, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.25601)  

**Abstract**: Teacher education requires deliberate practice with learners who exhibit identifiable strengths, weaknesses, and partial mastery. Large language models could support such practice by simulating students with known skill components, enabling teachers to rehearse explanations, diagnoses, and instructional responses. For this purpose, however, the central requirement is neither to maximize benchmark accuracy nor to suppress isolated facts, but to control model behavior so that it reflects a specified skill profile. This paper investigates whether prompted language models can be steered to retain some skills while suppressing others. We introduce a benchmark-oriented framework in which an explicit skill vector represents a simulated student, prompt-based control specifies retained and missing competencies, and behavior is evaluated using profile-alignment metrics, retained-versus-forgotten comparisons, and cross-skill calibration analyses. The results show that selective partial mastery can be induced and measured in a structured mathematics setting, although the degree of controllability remains model-dependent. These findings position controllable learner simulation as a distinct research problem at the intersection of teacher education, educational simulation, and language-model control. 

---
# Multilingual Phonological Feature Recognition with Self-Supervised Speech Models 

**Authors**: Abner Hernandez, Tomás Arias-Vergara, Daiqi Liu, Andreas Maier, Paula Andrea Pérez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2605.25596)  

**Abstract**: Phonological features provide a language-general and linguistically grounded representation of speech. We present PhonoQ-2.0, a multilingual frame-level phonological feature recognizer built on self-supervised speech models. The system directly predicts a structured 22-dimensional feature vector per frame encoding manner, vowel quality, place, and voicing, instead of deriving features from phoneme outputs. To ensure phonologically coherent predictions, we introduce a manner-conditioned gating mechanism that activates valid feature groups. Evaluated across multiple languages and corpora, PhonoQ-2.0 achieves an average macro-F1 of 91.3% in-domain and 88.9% out-of-domain. Compared to a strong CTC phoneme baseline, it delivers consistent gains of +8.8 F1 in-domain and +8.6 out-of-domain on average. In unseen-language evaluation, PhonoQ-2.0 improves macro-F1 from 66.9% to 73.6% (+6.7 on average), with gains of up to +10.8 points. 

---
# PennySynth: RAG-Driven Data Synthesis for Automated Quantum Code Generation 

**Authors**: Minghao Shao, Nouhaila Innan, Hariharan Janardhanan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2605.25572)  

**Abstract**: The growing complexity of quantum programming frameworks has exposed a critical limitation in existing large language model (LLM)-based code assistants: general-purpose models hallucinate PennyLane-specific gate names, misplace device configurations, and produce structurally invalid circuits when faced with specialized quantum coding challenges. We present PennySynth, a retrieval-augmented generation framework that addresses this gap by conditioning LLM inference on a curated knowledge base of 13,389 PennyLane instruction-code pairs, built via a three-stage extraction, verification, and deduplication pipeline over official PennyLane repositories, community GitHub sources, and QHack competition archives. PennySynth introduces a code-aware embedding strategy using st-codesearch-distilroberta-base, trained for natural-language-to-code retrieval, increasing average retrieval cosine similarity from 0.45 to 0.726 compared to a general-purpose baseline. Evaluated across 74 challenges spanning three years of the QHack competition (2022, 2023, 2024), PennySynth achieves 64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024, respectively, improving over Claude Sonnet 4.6 without retrieval by +28, +25, and +28 percentage points. We further introduce a quantum-adapted CodeBLEU metric that upweights qml.* token patterns and show that structural code similarity and functional correctness capture distinct aspects of quantum code quality. Controlled ablations reveal that code-aware embeddings are the primary driver of retrieval performance, while dataset expansion and source composition provide additional gains when retrieval quality is sufficiently precise. 

---
# BC Protocol: Structured Dual-Expert Dialogue for Eliciting High-Quality Chain-of-Thought Post-Training Data 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25549)  

**Abstract**: High-quality expert chain-of-thought (CoT) data is one of the core bottlenecks in large language model (LLM) post-training. Existing data production methods each have structural limitations: crowdsourced annotation lacks deep reasoning paths; expert solo writing is constrained by the "expert blind spot" -- experts structurally skip reasoning steps they consider obvious; RLHF only produces preference signals rather than reasoning chains.
This paper proposes the BC Protocol -- a structured dual-expert elicitation method for LLM post-training data production. The method carefully pairs a domain expert (crystallized intelligence) with a knowledge engineer (fluid intelligence), systematically externalizing the expert's implicit judgments as natural language reasoning chains. We introduce the Participant Aptitude Model, which defines six participant characteristic dimensions that affect elicitation quality. "Calibrated Ignorance" is an original concept proposed in this paper. We further propose "Selection-over-Prescription" as a methodological principle: for implicit knowledge elicitation tasks, investing quality-control resources in personnel selection yields a higher return than investing the same resources in process design.
In a controlled experiment in the narrative fiction domain, we directly compared CoT produced by BC Protocol dual dialogue (Group A, (n=20)) against CoT written independently by the same domain expert (Group B, (n=20)). Three cross-vendor judge models -- GPT-4o, Claude Opus 4.5, and Gemini 2.5 Pro -- conducted blind evaluation across five dimensions (600 ratings total). Results show that the BC Protocol achieves an overwhelming advantage in "naturalness of reasoning process" (Group A mean 4.80 vs. Group B mean 1.30, (p=2.4\times10^{-8}), Cliff's (\delta=1.0)). 

---
# Is Inference Mediated by Distinct Semantic Structures in LLMs? A Mechanistic Interpretation 

**Authors**: Nura Aljaafari, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2605.25520)  

**Abstract**: Predicting a label correctly does not necessarily require representing the operation that produces it. Transformer representations are known to carry label-level information, but whether they encode semantic operations producing those labels is unclear. We investigate this in Natural Language Inference using controlled premise-hypothesis pairs that differ by a single semantic transformation. Using layer-wise activations, we estimate operation-level subspaces via SVD and test their causal relevance through activation steering in four open-weight decoder models. Transformation effects are decodable with $84.8$-$99\%$ accuracy and occupy partially distinct but overlapping subspaces, exceeding random-subspace baselines. Steering experiments show that these directions causally influence predictions, though steerability varies across models; cross-operation steering further reveals structured interference and a dissociation between subspace selectivity and cross-operation independence. These findings indicate that the models encode not only that a hypothesis relates to a premise but also, in part, how it does so, implying that mechanistic analysis and control should operate at the level of semantic operations rather than predicted labels alone. 

---
# CRPO: Character-centric Group Relative Policy Optimization for Role-aware Reasoning in Role-playing Agents 

**Authors**: Yihong Tang, Kehai Chen, Liang Yue, Benyou Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25511)  

**Abstract**: Recent advancements in Reinforcement Learning (RL), particularly Group Relative Policy Optimization (GRPO), have significantly enhanced the reasoning capabilities of Large Language Models. However, applying these problem-centric optimization methods to role-playing agents often leads to a loss of character fidelity and style collapse, as they prioritize context-specific utility over persona alignment. To address this, we propose Character-Centric Group Relative Policy Optimization (CRPO), a framework designed to realign RL objectives with the role-playing task. CRPO improves character distinctiveness through three mechanisms: decoupling task logic from stylistic rewards to resolve gradient conflicts, dynamically adapting optimization constraints based on character complexity, and utilizing generic responses as negative baselines to prevent the model from reverting to a common distribution. Extensive experiments demonstrate that CRPO outperforms existing methods in consistency, emotion and others. 

---
# The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models 

**Authors**: Samee Arif, Angana Borah, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2605.25510)  

**Abstract**: Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7--11 using a developmental-psychology-grounded LLM-as-a-Judge rubric. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 9--47% across models, while explicit age adds a further 10--30% gain. Cross-lingual and cultural evaluations show uneven safety behavior across languages and country contexts. Multi-turn simulations show that child-facing response quality can degrade by 6--24% from the first to worst turn. Beyond evaluation, we introduce KIDGuardLlama, a child-safety evaluator, and KIDLlama, a child-oriented response model, showing how KIDBench supports safer child-facing AI 

---
# A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis 

**Authors**: Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2605.25502)  

**Abstract**: Educational aspect-based sentiment analysis (ABSA) can support course improvement, but public aspect-labeled student feedback remains scarce because educational reviews are private, institution-specific, and expensive to annotate. This study introduces a controlled synthetic benchmark for educational ABSA built from 10,000 synthetic course reviews with explicit train-validation-test splits and a 20-aspect pedagogical schema spanning instructional quality, assessment and course management, learning demand, learning environment, and engagement. The corpus is generated with sampled target labels, sampled nuance attributes, and a realism-tuned prompt refined through a three-cycle judge-editor procedure. On the resulting benchmark, local baselines with TF-IDF, two-step transformers, and joint encoders show that the task is nontrivial; the strongest untuned model, BERT, reaches a held-out detection micro-F1 of 0.2760, while a modest lower-rate BERT schedule improves this to 0.2930. Full-test GPT-based inference with gpt-5.2 reaches 0.2519 micro-F1 in zero-shot mode and 0.2501 with retrieval-based few-shot prompting, placing batch inference above the classical baseline and close to the compact joint encoders. A conservative external evaluation on 2,829 mapped student-feedback reviews from Herath et al. yields a micro-F1 of 0.4593 for BERT on a 9-aspect overlap, indicating partial synthetic-to-real transfer. Realism and faithfulness analyses are reported as generator diagnostics that clarify how the benchmark was stabilized and where label noise remains. The study therefore contributes a synthetic educational ABSA corpus, a documented generation procedure, and a reproducible benchmark setting for a domain in which public labeled data remain difficult to obtain. 

---
# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki 

**Authors**: Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25480)  

**Abstract**: LLM agents require retrieval to behave less like one-shot context fetching and more like reasoning: searching, reading, traversing, and deciding when evidence is sufficient. However, Retrieval-Augmented Generation (RAG) typically organizes external knowledge as flat chunks retrieved by embedding similarity, exposing a retrieval-as-lookup interface that is poorly aligned with tool-using agents. We propose LLM-Wiki, an agent-native retrieval system that operationalizes the Retrieval-as-Reasoning paradigm by treating external knowledge as a compilable, composable, and self-evolving structure rather than a static retrieval index. LLM-Wiki compiles documents into structured Wiki pages with bidirectional links, exposes search, read, and link-following operations through standard tool-calling interfaces, and introduces an Error Book for persistent structural and semantic self-correction. On HotpotQA, MuSiQue, and 2WikiMultiHopQA, LLM-Wiki outperforms seven baselines, including HippoRAG 2, LightRAG, and GraphRAG, with gains of 2.0-8.1 F1 points over the strongest graph-based baseline and larger gains over Dense RAG. On AuthTrace, LLM-Wiki achieves the best overall accuracy, with especially strong gains on multi-document structured queries, showing that compilation-based knowledge organization generalizes beyond chain-style multi-hop reasoning. 

---
# IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference 

**Authors**: Xintong Yang, Hao Gu, Binxing Xu, Lujun Li, Bei Liu, Jiacheng Liu, Qiyuan Zhu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.25475)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to operate over long contexts, yet standard softmax attention incurs a KV cache that grows linearly with sequence length, quickly becoming the bottleneck for long context inference. A practical remedy is to evict less important KV entries; however, existing eviction policies are largely heuristic and struggle to capture the rich, input-dependent distribution of token importance. In this work, we introduce a learnable indexer that predicts KV importance, enabling more accurate retention of critical tokens. Meanwhile, naively evicting tokens permanently discards their information, leading to irreversible forgetting and degraded retrieval over long ranges. To address this, we propose a lightweight latent memory module that compresses evicted tokens into a compact, online-updated state and provides residual readouts to compensate for the attention contributions lost through KV eviction. Collectively, our method enables accurate long-context inference under a bounded KV budget, delivering consistent improvements on RULER (4K/16K) across Qwen, Mistral, and Llama models (up to 25 points under aggressive eviction), markedly more stable Needle-in-a-Haystack retrieval, and superior LongBench scores and compression curves compared to existing eviction policies. 

---
# TypedCSIP: Typed Counterfactual Pretraining for Chinese Legislative Conflict Classification 

**Authors**: Yao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25474)  

**Abstract**: TypedCSIP is a typed counterfactual pretraining method for the conflict-classification task of the LCR-CN benchmark (Zhao et al.,
2026): given a (superior, subordinate) provision pair, predict whether the pair conflicts and which of four legal-doctrine types
(Responsibility, Condition, Sanction, Definition) describes the inconsistency. We exploit LCR-CN's expert-written minimal revisions
as training-time counterfactual supervision; at test time the classifier reads only the original pair.
Stage 1 pretrains a shared encoder with a typed Counterfactual Selective Intervention Pretraining objective on (superior,
subordinate, expert-revised) triplets, treating the expert revision as a counterfactual that the typed factor head must classify as
carrying no conflict evidence. Stage 2 transfers the encoder to a five-way classification head.
The confirmatory test was registered on the Open Science Framework before observing v6 measurements: 18 seeds, locked rule requiring
mean per-seed difference at least 0.8 pp with both seed-bootstrap and Student-t 95% lower bounds above zero. On the 696-record test
split, the v2 variant improves macro-F1 over the strongest single-model baseline by +0.916 pp on chinese-roberta-wwm-ext and +1.288
pp on the SAILER cross-backbone replication; both cells pass the rule. A cold-start stratified result on the 244 Unseen-gB records
keeps the gain positive on both backbones.
A cross-task diagnostic shows the Stage-2 encoder is classification-specialized and does not transfer to LCR-CN's superior-law
retrieval task, so we scope the contribution to conflict classification. We release code, 72 pre-registered prediction files,
matched-seed and MLM-control auxiliaries, and the OSF pre-registration record. 

---
# A Lightweight Hybrid Transformer-CRF Architecture for Multi-Type Bangla Medical Entity Recognition 

**Authors**: Peyal Saha, Ahsanul Haque Hasib, Shoumik Barman Polok  

**Link**: [PDF](https://arxiv.org/pdf/2605.25463)  

**Abstract**: MedER refers to the identification of medical entities. It is crucial for extracting structured clinical information from unstructured medical text. Many existing systems rely on transformer-based models, which are computationally expensive and difficult to deploy in resource-constrained environments. Furthermore, earlier works often use relaxed evaluation metrics that artificially inflate performance by rewarding correct prediction of dominant "Outside" (O) tokens. In this paper, we propose a lightweight Medical Entity Recognition (MedER) framework for the Bangla language. We establish a rigorous baseline using a 12-layer BanglaBERT model combined with a Conditional Random Field (CRF) layer for exact-boundary entity detection. To address deployment constraints, we compress this teacher model into a 4-layer student network through Knowledge Distillation (KD), where the student learns from the teacher's pre-CRF soft emission logits. Finally, we apply INT8 dynamic quantization to further reduce model size and inference cost. Our final quantized student achieves an 8.6x CPU speedup while requiring nearly 48 percent less storage than the CRF teacher model. 

---
# GeoSVG-RL: Geometry-Aware Reinforcement Learning for Layout-Constrained Text-to-SVG Diagram Generation 

**Authors**: Sifan Li, Yujun Cai, Hongkai Chen, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25447)  

**Abstract**: Generating structured, editable diagrams remains a significant challenge for contemporary large language models, despite their proficiency in general-purpose vector code generation. The primary difficulty lies in the structural fragility of the output; minor errors such as misaligned connector endpoints, text labels overlapping borders, or complex layouts drifting beyond the canvas boundaries render the resulting SVG files functionally unusable for professional applications. To address these issues, we introduce GeoSVG-RL, a specialized reinforcement learning framework designed for layout-constrained text-to-SVG generation. Unlike standard training objectives that rely solely on maximizing token-level likelihood, our approach optimizes the policy against explicit, executable geometric feedback. The model first produces a structured layout plan that serves as a geometric contract for the subsequent generation of the SVG code. This code is then rendered through a browser-backed verifier, enabling the calculation of fine-grained rewards across six critical dimensions: rendering validity, canvas fitting, precise anchor placement, text containment, graph consistency, and code cleanliness. We utilize Group Relative Policy Optimization (GRPO) to refine the model, sampling multiple candidates per prompt to facilitate updates based on relative quality. Starting from a supervised warm-start phase on synthetic data, GeoSVG-RL achieves substantial gains in structural reliability, particularly in arrow-anchor accuracy and text-in-box rates. Quantitative evaluations demonstrate that our method consistently outperforms current state-of-the-art systems in local geometric precision and the preservation of graph connectivity, providing a robust pathway toward automated yet reliable technical illustration. 

---
# Harmony in Diversity: Multi-domain Contrastive Policy Optimization for Large Reasoning Models 

**Authors**: Zongji Yu, Wenshui Luo, Yiliu Sun, Hao Fang, Runmin Cong, Chaochao Lu, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.25443)  

**Abstract**: Post-training has significantly enhanced the reasoning capability of Large Reasoning Models (LRMs), especially with Reinforcement Learning (RL) like Group Relative Policy Optimization (GRPO). However, GRPO-style RL methods in multi-domain settings often fail to achieve consistent improvements across all domains due to inherent interference in policy optimization. Prior studies on multi-domain RL primarily focus on alleviating cross-domain interference, while often neglecting the pivotal role of knowledge sharing, which we argue is the key to transforming cross-domain interactions from harmful competition into beneficial transfer. To address this limitation, we propose Multi-domain Contrastive Policy Optimization (MCPO), which analyzes the structural relationships among rollouts and promotes cross-domain knowledge sharing and in-domain knowledge consolidation in a contrastive manner. Specifically, for a given prompt, MCPO identifies transferable reasoning trajectories from other domains as positive examples, while treating incorrect rollouts as negative ones. It then encourages consistent representations for positive pairs and pushes negative pairs apart, thereby facilitating knowledge transfer and reducing interference. Moreover, MCPO aligns intra-domain correct rollouts to build a consolidated representation space. In this way, MCPO contrastively learns a harmonious representation space that can accommodate diverse multi-domain knowledge. Empirical results show that MCPO improves the reasoning capabilities of LRMs across multiple domains and even outperforms single-domain training in some cases. Code is available at this https URL. 

---
# A Multi-Agent LLM Framework for Rating the Quality of Surgical Feedback 

**Authors**: Rafal Kocielnik, J. Everett Knudsen, Steven Y. Cen, Jasmine Lin, Cherine H. Yang, Atharva Deo, Ujjwal Pasupulety, Peter Wager, Anima Anandkumar, Andrew J. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2605.25440)  

**Abstract**: Verbal feedback delivered by attending surgeons in the operating room plays a critical formative role in resident trainee skill acquisition. Yet, assessing the quality of trainer feedback and its effectiveness in influencing trainee behavior during live surgery remains a challenge. Prior studies assessed feedback content relying on extensive manual annotation by expert human raters and focused on developing broad taxonomies that overlook the qualitative aspects of feedback delivery such as clarity or urgency. Limited existing automated methods, including keyword analysis and topic modeling, also fail to capture these nuanced aspects. We introduce a two-stage LLM-based framework that discovers interpretable feedback quality criteria grounded in the context of surgical training. Our method uses multi-agent prompting and surgical domain knowledge injection to discover a small set of human interpretable scoring criteria (e.g., Encouraging, Urgent, Clear). These criteria are then used to automatically score live surgical feedback via an LLM-as-a-judge approach. Evaluation on 4.2k trainer feedback instances demonstrates that our AI-discovered criteria outperform prior content-based frameworks in predicting feedback effectiveness, including observed trainee behavioral adjustments and trainer approval. This work advances scalable, human-aligned assessment of communication quality in the operating room and provides a foundation for improving surgical teaching practices. 

---
# HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol 

**Authors**: Xinyi Mou, Siyuan Wang, Zejun Li, Yulan He, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.25421)  

**Abstract**: Communication protocol design is a central challenge in large language model-based multi-agent systems. Existing single-channel approaches face an inherent communication trilemma: text-based methods are interpretable but verbose, while latent-space methods are efficient but opaque and limited to unidirectional workflows. Inspired by multi-channel communication theory, we propose HyLaT, a hybrid latent-text communication protocol that transmits elaborate cognitive signals through a latent channel for efficiency, while expressing concise critical signals in natural language to preserve interpretability and precision. We introduce a two-stage training framework combining single-agent hybrid generation learning and multi-agent interactive co-training, enabling agents to generate and interpret hybrid messages across multiple rounds of interaction. Experiments demonstrate that HyLaT reduces communication overhead significantly while maintaining competitive task performance, with strong generalization and robustness across diverse settings. 

---
# SomaliBench Eval: Measuring English-to-Somali Refusal Gaps in Open-Weight Language Models 

**Authors**: Khalid Yusuf Dahir  

**Link**: [PDF](https://arxiv.org/pdf/2605.25420)  

**Abstract**: Large language model safety evaluation remains heavily English-centered, leaving low-resource languages under-measured even when models are deployed globally. We evaluate four open-weight instruction-tuned models on SomaliBench v0, a native-author-verified benchmark of 100 harmful-intent prompts paired across English and Somali. Each of Llama-3.1-8B-Instruct, Gemma-2-9B-Instruct, Qwen-2.5-7B-Instruct, and Aya-23-8B is run locally with temperature 0 and the same English "helpful, harmless, and honest" (HHH) system prompt. A pinned Claude Sonnet snapshot (claude-sonnet-4-5-20250929) classifies each response as refused, complied, or unclear; the native author spot-checks a stratified 80-row sample. We find large English-to-Somali refusal gaps for all four models: Llama-3.1-8B (0.90; 95% bootstrap CI [0.85, 0.96]), Aya-23-8B (0.75 [0.67, 0.83]), Qwen-2.5-7B (0.69 [0.59, 0.78]), and Gemma-2-9B (0.38 [0.27, 0.49]). For three models, the dominant Somali non-refusal mode is not fluent harmful compliance but unclear output: empty, wrong-language, or incoherent generations. The native verification spot-check achieves 100% agreement with the judge (Cohen's kappa = 1.00) on the 80 sampled rows. We report aggregate refusal rates, category gaps, and reliability statistics only; raw model generations are retained locally and are not released. 

---
# LLM-as-a-Reviewer: Benchmarking Their Ability, Divergence, and Prompt Injection Resistance as Paper Reviewers 

**Authors**: Lingyao Li, Junjie Xiong, Changjia Zhu, Runlong Yu, Chen Chen, Junyu Wang, Renkai Ma, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25415)  

**Abstract**: Large language models (LLMs) are increasingly used in academic peer review, yet their reliability, alignment with human judgment, and robustness to adversarial attacks remain poorly understood. We present a systematic benchmark of LLM-as-a-Reviewer on 898 papers stratified from NeurIPS and ICLR, evaluating 12 LLMs along three axes: rating calibration, divergence from human reviewers, and resistance to prompt injection embedded via an invisible font-mapping attack. We find that LLMs systematically overrate weaker submissions and diverge from humans in topical emphasis, under-flagging Clarity and over-flagging Reproducibility, while producing reviews two to three times longer with lower lexical diversity and a more standardized vocabulary. Prompt injection remains highly effective. Simple hidden instructions can promote low-scoring papers to acceptance-level ratings in a substantial fraction of cases, with effectiveness varying sharply across model families. While LLMs offer utility in structuring evaluations, their integration into peer review requires safeguards against both intrinsic biases and adversarial risks. 

---
# Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems 

**Authors**: Yizhou Peng, Ziyang Ma, Changsong Liu, Yi-Wen Chao, Xie Chen, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2605.25404)  

**Abstract**: Cascaded Automatic Speech Recognition -- Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that exploit deep ASR latent representations to disentangle token-level errors into perception, comprehension, and deletion failures. This fine-grained diagnostic intelligence empowers the LLM to orchestrate targeted, multi-turn clarification strategies, effectively transforming ambiguous signals into seamless user interactions. Experimental results validate the precision of our approach, which more than doubles the recall on domain-shift errors (57.96% vs. 23.66%) compared to baselines. Crucially, this diagnostic precision yields up to a 30% reduction in WER and a 17% improvement on the downstream task across diverse accents, distortions, and domains. 

---
# GeoMathCode: Understanding Interleaved Math-Code Reasoning for Geometry Problem Solving 

**Authors**: Yingji Zhang, Yong Dai, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2605.25384)  

**Abstract**: Mathematical reasoning is a hallmark of human intelligence, requiring logical deduction, symbolic manipulation, and abstract thinking. Recent multimodal large language models (MLLMs) have demonstrated strong performance on geometry problems through multi-step reasoning. To better emulate human problem-solving, intermediate steps can incorporate auxiliary visual constructions, such as additional lines or points, which improve geometric interpretation and educational clarity. In this work, we introduce the GeoMathCode, where programmatic representations serve as intermediate visual outputs. We further conduct an in-depth analysis of the underlying reasoning geometry. Experimental results show that reasoning and code generation steps can be disentangled in the latent space, while supervised fine-tuning (SFT) makes the reasoning manifold more structured and informative. Moreover, hierarchical syntactic code structures emerge as disentangled latent subspaces, and contain more mathematical symbolic information than visual representations. 

---
# AuthTrace: Diagnosing Evidence Construction in Thematically Dense Single-Author Corpora 

**Authors**: Xiaoqing Wu, Feifei Li, Haoliang Ming, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25382)  

**Abstract**: Evidence construction systems--chunk retrieval, agent memory, knowledge-graph traversal, and thematic indexing--are evaluated on separate benchmarks with incompatible corpora and metrics, making cross-paradigm diagnosis impossible. We introduce AuthTrace, the first diagnostic benchmark that places all major paradigms on a single corpus and query set by exploiting the dual nature of single-author collections. Built on thematically dense corpora where all texts share style, topic, and vocabulary, AuthTrace provides 2,099 instances with exhaustive gold evidence and a fan-in gradient as the primary diagnostic axis. Comparing eight systems across two QA models, we find that (1) evidence recall--not precision--is the dominant predictor of answer quality (r = 0.96); (2) fan-in exposes paradigm-specific collapse patterns, with flat retrieval degrading 3x faster than structured-evidence systems; and (3) full-context prompting fails uniformly, establishing evidence construction as a necessary capacity beyond raw corpus exposure. 

---
# EfficientGraph-RAG: Structured Retrieval-State Management for Cross-Task Retrieval-Augmented Generation 

**Authors**: Miaohe Niu, Lianlei Shan, Zhengtao Yu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25379)  

**Abstract**: Retrieval-augmented generation (RAG) has become the standard way to ground large language models in external knowledge, but many systems still organize evidence as flat chunks and retrieve it through largely unstructured search. This weak structure becomes a bottleneck for complex retrieval: the system must decide where to search, how to move from coarse topics to entity-relation evidence, which evidence has been verified, and which intermediate artifacts can be reused. We define these intermediate variables as a retrieval state and study RAG as structured state management. EfficientGraph-RAG makes this state explicit through three coupled mechanisms: TAM defines a typed hierarchical state space over evidence, MARS updates and verifies the state through role-specialized agents, and SMP stores reusable state under hierarchy-aware access control. Using one shared framework configuration, EfficientGraph-RAG ranks first on the reported answer-quality metrics averaged over the three evaluated LongBench retrieval-style subsets, matches the strongest agentic baseline on HotpotQA EM while reducing large-model token usage by $3.51\times$, and provides a low-token DocVQA result among retrieval-organizing cross-modal methods. Component analysis shows role-specific mechanisms: MARS is the main answer-quality driver, TAM supplies the typed traversal state and Adaptive Routing signal, and SMP enables corpus-dependent reuse, with cross-query cache hit rates ranging from 3.77% to 23.18%. 

---
# Learning to Route Languages for Multilingual Policy Optimization 

**Authors**: Geyang Guo, Hiromi Wakaki, Yuki Mitsufuji, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25360)  

**Abstract**: Large language models~(LLMs) are trained on heterogeneous multilingual corpora, yet existing policy optimization methods often implicitly restrict each training question to a single response language or rely on a fixed dominant language for supervision. We propose language-routed policy optimization (LRPO), an online policy optimization framework that treats language as a selectable variable. LRPO elicits multilingual rollouts for each training question and integrates their relative quality into preference-based policy updates, increasing the diversity and informativeness of training signals under the fixed rollout budget. To adaptively determine which languages to explore during reinforcement learning, we introduce a trainable language router formulated as a multi-armed bandit, balancing exploration of underutilized languages with exploitation of more informative ones. Extensive experiments show that LRPO consistently improves multilingual performance, demonstrating that adaptive language routing enables effective cross-lingual knowledge exploitation for training. We release all the resources at this https URL. 

---
# AI-Associated Lexical Shifts Across 34 Languages: Cross-Lingual Convergence and Diachronic Uptake in News Writing 

**Authors**: Thomas Stephan Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2605.25358)  

**Abstract**: AI-associated lexical shifts have been documented mainly in Scientific English. We extend this work to 34 languages in the WMT News Crawl corpus, refining a split-halves continuation diagnostic that compares GPT-4.1 continuations with matched human gold-standard text. For each language, we derive ranked AI-overused lemmas using log prevalence ratios. We find substantial cross-lingual semantic convergence: semantically related concepts recur across typologically diverse languages, with 'emphasize'-type verbs appearing in 24 of 34 languages. Embedding-based and manual analyses support this pattern. We also examine diachronic uptake in news writing before and after ChatGPT's release. Tracking each language's top 20 AI-overused items, we find prevalence increases in 26 of 34 languages from 2020-2021 to 2023-2024, with a mean change of +15.1%, whilst matched baseline words show no comparable increase (-4.5%). In 10 languages with longer historical coverage, longitudinal analyses show post-2022 increases that exceed the modest shifts observed in earlier periods, though with smaller effect sizes than in Scientific English. We validate our approach extensively, including across seeds, model variants, data sizes, model families, and more. Our findings are consistent with the view that AI-associated lexical preferences extend beyond English and may exert cross-lingual homogenising pressure on global language use. 

---
# A general tensor-structured compression scheme for efficient large language models 

**Authors**: Ying Lu, Peng-Fei Zhou, Qi-Xuan Fang, Pan Zhang, Shi-Ju Ran, Gang Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.25344)  

**Abstract**: Large language models (LLMs) are dominated by dense linear transformations, whose storage, memory and computational overheads hinder efficient adaptation and deployment while masking the functional impacts of structural simplification. Here we present Tensor Mixture (MixT), a general tensor-structured compression scheme that replaces targeted dense linear layers with natively executable mixtures of tensor operators. Operating directly on generic linear projections instead of model-specific components, MixT is potentially applicable across Transformer-based LLMs and other dense neural mappings. We evaluate MixT on Qwen3-8B and LLaMA2-7B under a unified recovery protocol, identifying a broad compressible regime in which MMLU accuracy is largely preserved before an abrupt transition at model-specific boundaries. This transition coincides with coordinated shifts in output entropy, prediction entropy and inter-layer geometry. At the LLaMA2-7B transition boundary, MixT reduces full-model parameters by 47.5\%, inference FLOPs by 37.1\%, training FLOPs by 52.1\% and peak inference memory by 60.4\%, demonstrating its practical potential for lower-cost LLM compression. 

---
# MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models 

**Authors**: Linhao Luo, Thuy-Trang Vu, Van-Anh Nguyen, Junae Kim, Gholamreza Haffari, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2605.25342)  

**Abstract**: Aligning large language models (LLMs) with diverse and multifaceted user preferences is a fundamental challenge in personalized AI systems. Existing multi-objective alignment methods either rely on costly training or require pre-trained reward models for each preference, making it difficult for them to adapt to evolving preferences. Prompt-based personalization offers a training-free alternative, but prompting alone often provides limited steerability, as LLMs may overemphasize or overlook certain preferences and fail to give users reliable control over the relative importance of different objectives when conflicts arise, leading to suboptimal alignment. In this paper, we introduce MATO, a training-free framework for Multi-objective personalized Alignment with Test-time Optimization. MATO formulates personalization as a test-time optimization problem that steers the relative importance of multiple objectives through controllable weights during decoding, without modifying model parameters or requiring external reward models. Specifically, a reward discovery module recovers preference rewards directly from the backbone LLM for diverse objectives specified in natural language, while a weight optimization module dynamically adjusts objective weights based on the user's initial preferences and the partially generated response to balance competing objectives during generation. The resulting rewards and weights jointly guide an online optimization procedure over the token distribution, enabling better alignment with the target objectives. Extensive experiments across multiple datasets and backbone LLMs show that MATO consistently outperforms strong baselines, achieving Pareto-improving multi-objective alignment and stronger steerability. These results highlight test-time optimization as a promising direction for scalable, controllable, and model-agnostic personalized alignment. 

---
# P1SCO: Social Dimensions from a Perspectivist Lens 

**Authors**: Amanda Cercas Curry, Gianmarco de Francisci Morales, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2605.25312)  

**Abstract**: We introduce P1SCO, a dataset of social media comments collected from three distinct platforms, annotated according to ten social dimensions to capture the diversity of social interactions and perceptions. The dataset is carefully disaggregated to allow analysis at the level of individual comments, annotators, and platforms. In addition to the social dimension labels, we include rich metadata on the annotators, including demographics, Big Five personality profiles, and political affiliation. This combination of comment-level annotations and annotator-level features enables nuanced analyses of how social perception varies across platforms, individual differences, and demographic factors. By preserving the diversity of annotator perspectives, our dataset supports studies of inter- and intra-annotator agreement, the influence of personality and political orientation on social interpretation, and the cross-platform dynamics of social discourse. 

---
# Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.25310)  

**Abstract**: Tool-using LLM agents produce trajectories whose calls form a directed dependency graph: earlier tool outputs supply arguments to later calls. Whether this execution structure is represented inside the model is unknown; prior structural probes have targeted static code or chain-of-thought text, not an agent's run-time call graph. A low-capacity edge probe on the residual stream of Qwen3-32B decodes the tool-call dependency graph well above both a Hewitt--Liang random-label control and a positional baseline. A counterfactual contrast between value corruption and structural perturbation indicates the signal tracks abstract topology rather than identifier values, and replicates under an independent, non-substring oracle. The non-positional component replicates on three further interactive multi-hop benchmarks and attenuates as call order alone becomes a sufficient proxy for dependency, vanishing in single-shot planning. Per-layer activation patching shifts the probe at a later, non-patched boundary, evidence that the representation propagates rather than passively reads out, though the realised tool call does not move. To our knowledge this is the first structural probe of an LLM agent's runtime tool-call dependency graph. Our claims concern representation, not behavioural control, and span two model families and one primary domain. 

---
# Eureka: Intelligent Feature Engineering for Enterprise AI Cloud Resource Demand Prediction 

**Authors**: Hangxuan Li, Renjun Jia, Xuezhang Wu, Yunjie Qian, Zeqi Zheng, Xianling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25297)  

**Abstract**: Effective features are crucial for predictive model performance, but creating them often requires domain expertise, limiting scalability across applications. We define feature engineering as an agentic code generation problem: features are not static data transformations, but executable programs that can be generated, evaluated, and iteratively improved. We present Eureka, an LLM-driven framework with three stages. (1) An Expert Agent, fine-tuned via SFT on domain knowledge, produces structured feature design plans in JSON format. (2) An LLM Feature Factory translates each plan into executable Python code through chain-of-thought reasoning, turning feature hypotheses into runnable programs. (3) A Self-Evolving Alignment Engine uses Reinforcement Learning (GRPO) with dual-channel reward (metric-based utility + semantic alignment) to enhance code quality. By expressing features as programs, the learned generation patterns can transfer across domains. Evaluated on 7 public benchmarks in healthcare, finance, and social domains, Eureka consistently outperforms both traditional AutoFE and LLM-based baselines. We further demonstrate Eureka's effectiveness on cloud GPU resource demand prediction at Alibaba Cloud, where Eureka improves demand fulfillment rate by 16% and lowers computing resource migration rates by 33%. 

---
# Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions 

**Authors**: Jinyan Su, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.25284)  

**Abstract**: User queries are often underspecified and may admit multiple valid interpretations. Rather than silently making assumptions about the user's intent, a helpful assistant should surface such ambiguity by asking a clarifying question. Doing so requires two abilities: recognizing that a query is ambiguous, and acting on that recognition by seeking clarification instead of answering directly. To study these abilities, we evaluate models on ambiguous, unambiguous, and disambiguated questions in three settings: standard question answering, explicit ambiguity judgment, and behavioral analysis, where a judge model classifies responses as direct answers, refusals, or clarifying questions. We find a clear gap between recognition and behavior: models often identify ambiguity when explicitly asked to judge it, yet in the QA setting they overwhelmingly default to direct answers. Retrieved context further widens this gap by improving answerability while making models even less likely to ask clarifying questions. 

---
# READER: Reasoning-Enhanced AI-Generated Text Detection 

**Authors**: Pingfan Su, Kai Ye, Shijin Gong, Erhan Xu, Jin Zhu, Giulia Livieri, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25281)  

**Abstract**: Recent advances in large language models (LLMs) have made it increasingly difficult to distinguish human-written text from AI-generated content. Many existing detectors train supervised neural classifiers that achieve strong in-distribution performance but are often opaque and can degrade substantially under distribution shift. We present READER, a reasoning-enhanced AI text detector that outputs both a human/AI label and a structured rationale describing the evidence for its decision. A key component of our approach is READ, a curated supervision set of rationales and verdicts. We fine-tune an LLM on READ to build READER, which reasons before detecting at inference time. Despite having only 1.5B parameters, READER consistently outperforms existing detectors as well as prompted, high-capacity LLM baselines (GPT-5.2, Gemini-3-Pro, and DeepSeek-V3.2), which are 100 to 1000 times larger in scale. 

---
# Mimir: Large-scale Multilingual Concept Modeling 

**Authors**: Elio Musacchio, Lucia Siciliani, Pierpaolo Basile  

**Link**: [PDF](https://arxiv.org/pdf/2605.25263)  

**Abstract**: Current language modeling approaches are built around tokens. Text corpora are split into tokens, and models are trained by performing computations on these tokens, such as predicting the next token given the preceding ones as context. This paradigm has become the standard in modern language modeling, especially given the outstanding performance obtained by token-based architectures. However, recent works have not only begun to question how language models process and understand meaning from tokens, but also to question whether using higher levels of granularity could advance the research field. This led to the idea of Concept Modeling, that is, to directly train models for next-concept prediction rather than next-token prediction. The goal is to change the input from tokens to concepts, forcing the underlying language model to shift its granularity from fine-grained tokens to broad concepts. In this work, we introduce Mimir, a 1.6B Large Concept Model trained for multilingual concept understanding and generation. We leverage a large-scale multilingual pre-training corpus (38,883,987,240 sentences) spanning 46 languages and a large-scale multi-turn and multilingual instruction-tuning dataset (66,816,428 sentences) covering a total of 35 languages. We extensively evaluate model performance against a language model with a comparable number of parameters. 

---
# Inference Time Optimization with Confidence Dynamics 

**Authors**: Yu Wang, Minghao Liu, Jiayun Wang, Jinrui Huang, Ankit Shah, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.25244)  

**Abstract**: Inference time optimization techniques, such as repeated sampling, have significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, the critical role of model uncertainty remains largely underexplored in these optimization strategies. In this paper, we investigate the dynamics of confidence along reasoning trajectories and for first time reveal a surprising and unique pattern: correct answer traces tend to exhibit confidence improvement over time (positive confidence gain), while incorrect traces show attenuated or declining confidence as reasoning proceeds. Based on this observation, we propose Confidence Dynamic Gain (CDG) based voting, which incorporates how the confidence trajectory of the response evolves along the reasoning chain. Experiments across four open-source architectures (DeepSeek-R1, gpt-oss, Gemma-3, Qwen-QwQ) on the AIME24/25, HMMT25, and BRUMO25 benchmarks demonstrate that CDG yields a significant performance boost over baselines. These results demonstrate that our method provides a robust discriminative signal for improving answer selection in LLM reasoning. We also provide theoretical insights for this phenomenon. Code will be released at this https URL. 

---
# JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment 

**Authors**: Russell Yang, Ruishi Chen, Pierce Kelaita, Riya Ranjan, Sibo Ma, Charles Dickens, Matthew Guillod, Megan Ma, Julian Nyarko  

**Link**: [PDF](https://arxiv.org/pdf/2605.25240)  

**Abstract**: Two methodologies dominate current practices of benchmarking: rubric-based scoring evaluates items against predefined criteria, whereas comparative judgment elicits pairwise preferences between outputs. Although both methodologies are widely used, the choice between them is rarely justified. We release JudgmentBench, a benchmark of 30 real-world legal tasks, paired with 1,539 rubric scores and 1,530 pairwise preference judgments collected from practicing attorneys--including at major U.S. law firms--with substantial experience. The annotations constitute the first publicly available dataset in a high-expertise domain in which both supervision signals are elicited from the same experts on the same items. Using LLM-generated outputs at three constructed quality levels, we provide an initial empirical comparison: comparative judgments recover the intended quality ordering substantially better than rubrics (mean Spearman's rank correlation of 0.908 vs. 0.150, estimated difference = 0.758 [0.494, 1.021]) while requiring less than half the annotation time. The patterns hold for human annotators and LLM autograders. Beyond this initial comparison, the paired structure of the dataset supports a broader research agenda on how expert judgment should be elicited, aggregated, and used as supervision in domains without verifiable ground truth. 

---
# From Automation to Collaboration: Human-in-the-Loop Methods for Safe and Trustworthy NLP 

**Authors**: Most. Sharmin Sultana Samu, MD. Tanvir Ahmed Seum, Md. Rakibul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2605.25226)  

**Abstract**: Large language models are widely deployed in high-stakes NLP tasks, yet risks such as bias, hallucination, adversarial vulnerability and unreliable generalization remain. Probe-based auditing reveals inconsistencies in model behavior. Adversarial text generation uncovers robustness gaps, especially in lower-resourced languages with limited benchmarks. Enterprise text-to-SQL settings expose the difficulty of validating outputs over private and large-scale databases. Human supervision is essential for probe validation, adversarial verification and domain-specific annotation, but it is costly and hard to scale. This survey examines recent human-in-the-loop methods that shift NLP from automation toward collaboration for safety and trustworthiness. We review how human expertise supports auditing, robustness evaluation, data construction and model steering. Our findings highlight gaps in scalable probing, sustainable robustness benchmarks, low-resource settings and governance of private systems. We outline practical research directions for adaptive auditing, collaborative evaluation and accountable deployment. 

---
# They Are Not the Same: Direct Causes Are Not Grounded Emotion Explanations 

**Authors**: Zhuangzhuang Pan, Yan Xia, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25208)  

**Abstract**: Emotion-Cause Pair Extraction (ECPE) was introduced to explain why an emotion occurs, but this goal is now often reduced to binary pair/non-pair prediction. This proxy is useful for direct-cause extraction, yet easy to over-read as evidence grounded emotion explanation. We show that this interpretation is only partially valid. In IEMO-MECP, 90.9% of original positives remain emo-cause and 95.0% of original negatives remain non-pair, confirming that the binary ECPE task is largely preserved. The problem is that direct triggers alone do not constitute a grounded explanation. Emo-context, an utterance that helps interpret a target emotion without directly causing it, appears on both sides of the original boundary and is enriched near binary uncertainty, showing that the binary boundary has no stable place for such discourse evidence. Across evaluated ECPE models, direct triggers are recovered more reliably than contextual support. Under shortcut pressure, this imbalance becomes consequential. Binary-trained models assign higher pair scores to nearby lexically similar non-pair candidates than to evidence supported but structurally harder emo-cause and emo-context pairs. Thus, pair scores can reward convenient attributions over grounded explanations. High binary ECPE performance indicates that a model can identify direct triggers; it does not indicate that the model has explained the emotion. Code is publicly available at this https URL. 

---
# Clarification Is Not Enough: Post-Clarification Answering Remains the Bottleneck in Multi-Turn QA 

**Authors**: Jinyan Su, Jennifer Healey  

**Link**: [PDF](https://arxiv.org/pdf/2605.25204)  

**Abstract**: Pluralistic alignment requires systems to adapt to diverse user values, communication styles, and contextual assumptions. We believe that a foundational prerequisite for such alignment enabling accurate preference elicitation from people when their intent is under-specified or ambiguous. We study the problem of preference elicitation in multi-turn question answering by decomposing the problem into two components: a \textbf{clarification policy}, which decides whether to ask a clarifying question or answer directly, and \textbf{post-clarification answering}, which produces the correct final answer once the missing information is provided. We show, using the PACIFIC benchmark, that supervised fine-tuning rapidly improves the clarification policy, however, final answer accuracy remains substantially lower even when the model takes the correct action. This gap indicates that understanding and correctly interpreting the user's response is the critical gap in multi-turn question-answering systems. 

---
# GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning 

**Authors**: Xiang Cheng, Yulan Hu, Lulu Zheng, Zheng Pan, Xin Li, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25200)  

**Abstract**: Travel planning is a realistic task for evaluating the planning and tool-use abilities of LLM agents. However, existing benchmarks typically assume only a single user, thereby avoiding one of the most challenging aspects of real-world scenarios: an agent's ability to identify and resolve conflicts among multiple users. To address this gap, we introduce \textbf{GroupTravelBench}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Based on real user profiles, POI data, and ticket price data, we synthesize 650 tasks and divide them into three difficulty levels. Beyond standard abilities in single-user itinerary planning, such as multi-step reasoning and tool use, our benchmark further evaluates three key capabilities required for travel agents: \emph{(i) elicitation} -- proactively engaging in multi-turn dialogue to gather preferences from each user; \emph{(ii) coordination} -- resolving conflicts among users through compromise or subgrouping strategies; and \emph{(iii) planning} -- searching for travel plans that maximize overall group utility while maintaining fairness and feasibility. To simulate real-world conversational itinerary planning while enabling reliable tool use and offline evaluation, we build an interactive sandbox environment with cached real-world tool data. We evaluate a wide range of LLMs and find that even frontier models still show substantial weaknesses in preference coverage and group fairness. \textit{GroupTravelBench} provides a practical and reproducible benchmark for advancing research on LLM agents for real-world travel planning. 

---
# By Their Fruits You Will Know Them: Comparing Formalizations of Law by the Decisions They Encode 

**Authors**: Julius Vernie, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.25186)  

**Abstract**: Formalizing legal provisions promises machine-accessible law and automated legal reasoning, and recent LLMs make it tempting to generate such formalizations directly from statutory text. However, any formalization makes implicit interpretive choices whose consequences are hard to anticipate, especially if an LLM is the author. We present a method for systematically comparing different formalizations of the same legal provision by their inferences on individual cases. Given multiple formalizations of a provision, we match them at the node level, derive a shared interface for each pair from the matching, and use a SAT solver to enumerate the edge cases on which any two formalizations disagree. Selected edge cases are then verbalized into concrete factual scenarios that a legal expert can examine and act on. We apply our method to formalizations of ten EU provisions generated by nine frontier LLMs. We find that behavioral divergence between formalizations is essentially uncorrelated with their structural agreement and that the verbalized cases reveal qualitatively distinct types of disagreement, including divergences that mirror genuine controversies in the legal commentary. 

---
# Knowledge Graph-Driven Expert-Level Reasoning for Neuroscience 

**Authors**: Jake Stephen, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25183)  

**Abstract**: Knowledge graph (KG) is an abstraction that can be extracted from text corpora and used for in-depth reasoning. Prior work has leveraged KGs to fine-tune language models (LMs), enabling domain-specific superintelligence. In this work, we explore whether KG-driven in-depth reasoning capabilities can emerge in neuroscience using only information contained within a single authoritative textbook. The central hypothesis is that structured knowledge, when distilled into a high-quality KG and converted into KG-grounded question-answer (QA) supervision, is sufficient to produce expert-level reasoning through a fine-tuned LM that surpasses large language models (LLMs) in accuracy, while employing orders of magnitude fewer parameters. We construct a textbook-derived KG via a dual-LLM validation pipeline, expand it with a masked LM trained on the KG topology, generate multi-hop QA items, which include QA pairs and reasoning traces, to fine-tune an LM exclusively on KG-derived supervision, and apply reinforcement learning using path-derived KG signals as implicit reward models. Our results demonstrate that deep, mechanistic neuroscience understanding can be induced in the model without reliance on large, heterogeneous web-scale corpora. The KG-based synthetic neuroscience curriculum that readers can quiz themselves on, and the fine-tuned LM, are available at the following GitHub location: this https URL. 

---
# Locality Matters for Training-Free Audio Token Compression in Audio-Language Models 

**Authors**: Jiale Luo, Xiaoyu Liang, Haoji Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25179)  

**Abstract**: Audio-language models (ALMs) are increasingly used for audio captioning, question answering, and open-ended audio understanding, but their inference cost remains high when audio inputs are represented as long prefix-token sequences. These audio prefixes consume context budget, increase memory usage, and make deployment harder in resource-constrained or latency-sensitive settings. Existing training-free audio-token reduction methods mainly rely on fixed pooling or score-based pruning. Fixed pooling is content-agnostic, while score-based pruning can preserve isolated salient tokens but discard nearby acoustic context. We propose Local Temporal Bipartite Merging (LTBM), a training-free encoder-space compression method that merges similar nearby audio tokens under an explicit temporal window constraint. Beyond introducing LTBM, we use a controlled Global Merge variant to isolate whether temporal locality itself is a useful inductive bias for audio-token compression. Experiments on AudioCaps, Clotho, and MMAU with Qwen2-Audio show evidence of a task-dependent locality effect: locality-aware merging is more favorable for captioning at several compression settings, especially under stronger compression, while global matching is more competitive for multiple-choice audio understanding. A cross-backbone validation on Audio Flamingo 3 further supports the captioning-side advantage of locality-aware merging under moderate and aggressive compression. 

---
# Re-defining Humor Data Objects for AI Humor Research 

**Authors**: Anna Arnett, Bang Nguyen, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25171)  

**Abstract**: In most existing AI humor research, humor was treated as either "present" or "not present." We explore the concept of humor as a social interaction with context and explanations. During this project, we defined a humor reasoning data object and developed a way to prompt LLMs to generate an explanation of humor effective for general population. We iterated from an earlier prompt to an improved prompt, found that the later version reduced important errors, and then scaled generation to a large number of data objects which have the potential to enable data synthesis and data augmentation for AI humor research. Our main takeaway is that better prompting of an LLM improves humor explanation quality, especially by handling missing context, multi-modality, and transcript issues more carefully. These results establish a strong foundation for future work on AI understanding of humor as social behavior. 

---
# STREAM: A Data-Centric Framework for Mining High-Value Task-Oriented Dialogues from Streaming Media 

**Authors**: Liang Xue, Haoyu Liu, Cheng Wang, Pengyu Chen, Haozhuo Zheng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25162)  

**Abstract**: Large language models for vertical domains are bottlenecked by the scarcity of complex, domain-specific task-oriented dialogues. Existing data acquisition pipelines face a persistent trilemma: expert annotation is expensive, real-world service conversations are constrained by privacy and commercial restrictions, and static corpora quickly become temporally stale. We propose Stream, a data-centric framework that leverages publicly available streaming media (live streams and short videos) to synthesize high-value service dialogues at scale. Stream mines authentic interaction signals from noisy streams and synthesizes conversations by integrating role-grounded persona construction with Conversational Blueprint construction; it further adopts retrieval-augmented generation (RAG) to support knowledge-aware responses. Based on Stream, we release StreamDial, a large-scale multi-domain dataset covering Automotive, Restaurant, and Hotel. StreamDial contains 87,498 dialogue sessions and 1,497,320 turns in total, with an average of 17.11 turns per session and a comparable scale across domains. Each session is organized as a structured quadruplet $\langle P_u, P_a, B, H \rangle$ that pairs dialogue history with explicit user/agent personas and a Conversational Blueprint, capturing realistic service behaviors such as requirement mining, constraint conflicts, negotiation, and recovery. Evaluations with automatic judges and downstream tasks show that StreamDial improves intrinsic dialogue quality over strong baselines, and models trained with StreamDial improve Dialogue State Tracking across backbones; we further report a completed human-evaluation set and encouraging multilingual transfer on Qwen3-8B under a controlled training budget. The data is released in this https URL. 

---
# LLM Agent Based Renewable Energy Forecasting Using Edge and IoT Data A Review of Solar Wind Weather and Grid Aware Decision Support 

**Authors**: Pavan Manjunath, Thomas Pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.25141)  

**Abstract**: Reliable forecasting of renewable energy generation is a foundational requirement for grid stability energy trading battery scheduling and carbon aware operational planning Solar and wind resources are inherently intermittent their output fluctuates with cloud cover wind speed atmospheric turbulence seasonal patterns and local terrain The proliferation of IoT and edge devices spanning smart meters inverters anemometers pyranometers weather stations and grid interface sensors has created an unprecedented volume of real time operational data that conventional forecasting pipelines are ill equipped to exploit fully This review investigates how large language model LLM agents can enhance renewable energy forecasting by integrating heterogeneous sensor streams weather API data historical generation records grid constraints and contextual reasoning into unified decision support workflows We survey classical forecasting methods statistical time series models deep learning architectures physics hybrid approaches and emerging LLM agent frameworks for explanation uncertainty communication and operator guidance A six layer taxonomy is proposed covering data acquisition preprocessing feature engineering model inference uncertainty estimation and natural language reporting The review identifies twelve open challenges spanning real time deployment model drift under distribution shift uncertainty quantification hallucination control in LLM agents interoperability of edge hardware and integration with energy management systems The paper concludes by recommending a research agenda centred on open benchmarks physics informed LLM grounding and federated forecasting architectures 

---
# Evidence-Linked Radiology Reporting: A Human-Supervised Reference Architecture for Structured Imaging Intelligence 

**Authors**: Houman Kazemzadeh, Kamyar Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25120)  

**Abstract**: Radiology reports remain the primary mechanism by which imaging findings are communicated to clinical teams. However, much of the structured information behind these reports, including measurements, image evidence, prior comparisons, lesion identity, uncertainty, and terminology, often remains trapped in free text or fragmented across picture archiving and communication systems, radiology information systems, reporting workstations, worksheets, advanced visualization tools, and electronic health records. This paper proposes a human-supervised, evidence-linked reference architecture for structured radiology reporting. The framework combines exam-specific templates, speech-to-structure processing, measurement and segmentation capture, controlled AI-assisted drafting, and standards-based interoperability using DICOM, DICOM Structured Reporting, DICOM Segmentation, HL7 FHIR, RadLex, SNOMED CT, LOINC, and UCUM. The system is positioned not as an autonomous report generator, but as a structured intelligence layer for enterprise imaging that supports reviewed reporting, longitudinal comparison, clinical data reuse, governance, and integration with PACS, RIS, EHR, analytics, and registry workflows. The paper also discusses modality-specific deployment considerations, clinical safety risks, validation requirements, cybersecurity, privacy, quality management, and regulatory boundaries for AI-assisted radiology reporting systems. 

---
# Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth 

**Authors**: Yoav Gur-Arieh, Ana Marasović, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2605.25052)  

**Abstract**: Chains of thought (CoTs) have become central in interpreting and auditing behaviors of large language models. Yet growing evidence suggests that these traces often fail to faithfully represent the computations behind a model's predictions. Several faithfulness metrics have been proposed, but whether they indeed measure faithfulness remains unknown. Answering this requires ground-truth labels, which are hard to obtain since internal computations are not directly observable. Consequently, most works proposing metrics report only absolute scores or comparisons to prior metrics, and the few existing benchmarks rely on proxies like plausibility or importance, properties orthogonal to faithfulness that can mislead about whether a CoT can be trusted. We address this challenge by constructing tasks whose outputs reveal which intermediate computations must have produced them, and developing an automated labeling pipeline that yields ground-truth faithfulness labels at both the step and CoT level. Building on this methodology, we present BonaFide, a benchmark of 3,066 labeled CoTs across 13 tasks and 10 models, and use it to conduct the first systematic evaluation of prominent faithfulness metrics. Our experiments show that most metrics perform near chance, exhibit strong prediction biases and degrade on longer CoTs. The best metric reaches only 0.70 AUROC at the CoT level while another reaches 0.59 at the step level, with neither transferring across settings, while entailing prohibitively high computational cost. Our results expose fundamental gaps in current faithfulness evaluation and call for the development of more reliable and efficient metrics. 

---
# TRACE: A taxonomy-grounded synthetic dataset for teaching-program generation and session interpretation in Applied Behavior Analysis 

**Authors**: Festus Kahunla  

**Link**: [PDF](https://arxiv.org/pdf/2605.25038)  

**Abstract**: Applied Behavior Analysis (ABA) is a clinical discipline whose documentation, teaching programs and multi-session behavioral logs, is formulaic and high-volume, yet real session data is HIPAA-protected and bound by professional confidentiality rules, blocking the release of a training corpus. We present TRACE (Taxonomy-Referenced ABA Clinical Examples), a 2,999-example synthetic instruction-tuning dataset covering two ABA tasks: teaching-program generation across Discrete Trial Training, Natural Environment Teaching, and Task Analysis; and multi-session behavioral interpretation across twelve trajectory patterns and thirteen target behaviors. Every example is produced by a deterministic taxonomy-driven generator grounded in the canonical ABA literature, and every example carries complete sampling provenance, the exact taxonomy cells that produced it. The dataset is released under CC BY-NC 4.0 for data and MIT for code, with stratified train (2,549), validation (149), test (281), and sanity (20) splits. TRACE is a research artifact and has not been clinically validated. 

---
# Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation 

**Authors**: Yangneng Chen, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.25036)  

**Abstract**: Large Vision-Language Models (LVLMs) extend large language models with visual understanding, but remain vulnerable to hallucination, where outputs are fluent yet inconsistent with images. Recent studies link this issue to language bias-the tendency of LVLMs to over-rely on text while neglecting visual inputs. Yet most analyses remain empirical without uncovering its underlying cause. In this paper, we provide a systematic study of language bias and identify its root in modality misalignment during training. Our analysis shows that both Visual Instruction Tuning (VIT) and Direct Preference Optimization (DPO) often prioritize textual improvements, which may cause LVLMs to overly lean toward language modeling rather than balanced multimodal understanding. To address this, we propose two simple yet effective methods: Language Bias Regularization (LBR) which mitigates language bias through regularization during instruction tuning, and Language Bias Penalty (LBP), which penalizes language bias in the DPO training process. Extensive experiments across diverse models and benchmarks demonstrate the effectiveness of our approach. LBR consistently improves performance on over ten general benchmarks, while LBP significantly reduces hallucination and improves trustworthiness. Together, these methods not only mitigate language bias but also advance the overall alignment of LVLMs, all without introducing any additional data or auxiliary models. Our code is publicly available at this https URL. 

---
# Better, Faster: Harnessing Self-Improvement in Large Reasoning Models 

**Authors**: Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, Leszek Rutkowski, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24998)  

**Abstract**: Self-improvement training enables the large reasoning models (LRMs) to improve themselves by self-generating reasoning trajectories as training data without external supervision. However, we find that this method often falls short in complex reasoning tasks and even leads to model collapse. Through a series of preliminary analyses, we reveal two problems: (1) data imbalance, where most training samples are simple, but the challenging yet crucial samples are scarce; (2) overthinking, where many undesired samples with redundant reasoning steps are used for self-training. To this end, we propose HSIR, which effectively Harnesses Self-Improvement in large Reasoning models via two simple-yet-effective approaches. Specifically, HSIR introduces a verify-then-exit sampling strategy to mitigate data imbalance by efficiently collecting more accurate solutions for difficult queries, and designs an Intrinsic Diversity score to quantify overthinking and filter out the undesired solutions. We apply HSIR to various post-training paradigms, among which we further propose H-GRPO, an enhanced GRPO algorithm that leverages the intrinsic diversity as an external reward to encourage concise and diverse reasoning via reinforcement learning. Extensive results show that HSIR not only effectively enhances the reasoning performance, i.e., bringing up to +10.9% average performance gains, but also significantly improves the reasoning efficiency by reducing up to 42.4% relative inference overhead. 

---
# Exploring Profiles of Cognitive Distortions Associated with Mental Health Disorders 

**Authors**: Alina Anikejeva, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2605.24996)  

**Abstract**: Cognitive distortions, distorted patterns of thinking, have been increasingly studied in computational mental health research. Although they are related to many, if not all, mental health disorders, most existing studies focus primarily on depression. In this work, we explore distortion profiles across multiple mental health conditions. We analyzed a large Reddit-based dataset containing posts from nine self-reported mental health groups as well as a control group using both an n-gram-based method and a fine-tuned transformer model for detecting cognitive distortions. Mental health groups, both when pooled together and when examined individually, showed higher prevalence of cognitive distortions compared to the control group, with the effect sizes ranging from small to moderate. When comparing distortion profiles across conditions, we observed largely similar patterns, although some groups exhibited overall higher levels of distortions than others. These findings suggest that relatively simple lexical approaches can be useful for exploratory analyses of group-level trends in large-scale mental health text data. 

---
# Large Language Model Selection with Limited Annotations 

**Authors**: Yavuz Durmazkeser, Patrik Okanovic, Andreas Kirsch, Torsten Hoefler, Nezihe Merve Gürel  

**Link**: [PDF](https://arxiv.org/pdf/2605.24981)  

**Abstract**: Choosing a Large Language Model (LLM) for a given task requires comparing many strong candidates, yet standard evaluation relies on costly annotations over fixed evaluation sets. To address this challenge, we develop SELECT-LLM, the first framework for active model selection of LLMs. SELECT-LLM aims to find a small set of queries whose annotations are most informative for identifying the best LLM for a given task. To this end, we introduce a query selection rule based on expected information gain, computed from pairwise similarities between candidate model outputs. Because this rule only uses generated model responses, SELECT-LLM can be applied across candidate models without assumptions about their architecture or access to model weights. This makes it suitable for both open-weight and black-box LLMs. We evaluate SELECT-LLM across 23 datasets, 156 evaluated models, diverse task families, and multiple text evaluation metrics. Across all experiments, SELECT-LLM improves over the strongest baseline in every setting, with annotation cost reductions up to 81.8% for best model selection and up to 84.78% for near-best model selection. 

---
# Investigating the Interplay between Contextual and Parametric Chain-of-Thought Faithfulness under Optimization 

**Authors**: Jingyi Sun, Qianli Wang, Pepa Atanasova, Nils Feldhus, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.24960)  

**Abstract**: Chain-of-Thought (CoT) faithfulness, i.e., whether CoTs genuinely reflect large language models' (LLM) underlying behavior, is typically evaluated under two disjoint paradigms: contextual faithfulness, measured by perturbing the input or CoT trace, and parametric faithfulness, assessed by intervening on a model's parametric knowledge. Yet prior work compares them only descriptively. We fill this gap by proposing FaithMate, a unified preference-alignment interface for optimizing models towards either faithfulness paradigm. It enables us to investigate the interplay between the two paradigms, examining whether and to what extent faithfulness gains generalize within and across paradigms. Across three models, two datasets, and six faithfulness metrics, we find that the two paradigms are positively coupled, yet asymmetric: optimizing towards parametric faithfulness yields consistent gains across both paradigms, whereas the contextual counterpart delivers more variable gains. Within the contextual paradigm, faithfulness gains on one metric do not consistently transfer to others, implying that existing contextual metrics capture disjoint facets of faithfulness and exposing inherent trade-offs. These findings imply that CoT faithfulness is not a monolithic objective and therefore requires multifaceted optimization and evaluation. 

---
# SEP-Attack: A Simple and Effective Paradigm for Transfer-Based Textual Adversarial Attack 

**Authors**: Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Xiaoming Xu, Wei Wang, Fenglong Ma, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24958)  

**Abstract**: Despite the strong performance of deep neural networks in modern Web and language applications, they remain vulnerable to adversarial attacks, especially transferable attacks that generate adversarial examples using surrogate models without accessing the victim model. Transferable attacks in the text domain are still under-explored, with only a few studies addressing this challenging issue, often with suboptimal results due to equal treatment of submodels or inaccurate estimation of importance scores. To address these challenges, we propose a simple yet effective paradigm for transfer-based textual adversarial attack, named SEP-Attack. Specifically, we employ the Determinantal Point Process (DPP) to generate diverse surrogate ensemble weights, representing the transferability of submodels. Using these weights, we introduce a new metric to evaluate prediction confidence scores, which in turn are used to calculate word importance scores and generate adversarial candidates. Finally, we quantify the transferability score for each candidate and select the top ones as the final transferable adversarial examples. Experiments conducted on four datasets and two real-world APIs validate the efficacy of SEP-Attack, significantly outperforming state-of-the-art baselines. 

---
# NITP: Next Implicit Token Prediction for LLM Pre-training 

**Authors**: Xiangdong Zhang, Debing Zhang, Shaofeng Zhang, Xiaohan Qin, Yu Cheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24956)  

**Abstract**: Standard next-token prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurations that can limit generalization. To address this issue, we propose Next Implicit Token Prediction (NITP), which augments discrete prediction with dense continuous supervision directly in the representation space. NITP trains the model to predict the implicit semantic content of the next token, using shallow-layer representations from the same model as stable self-supervised targets. We provide theoretical analysis showing that NITP regularizes the optimization landscape by mitigating under-constrained degrees of freedom and encouraging a compact, structured representation geometry. Empirically, across dense and MoE models ranging from 0.5B to 9B parameters, NITP consistently improves downstream performance with negligible computational overhead. On a 9B MoE model, NITP achieves a 5.7% absolute improvement on MMLU-Pro, along with gains of 6.4% on C3 and 4.3% on CommonsenseQA, with approximately 2% additional training FLOPs and no additional inference cost. Our implementation is available at this https URL. 

---
# H$^{2}$MT: Semantic Hierarchy-Aware Hierarchical Memory Transformer 

**Authors**: Maryam Haghifam, Zifan He, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24930)  

**Abstract**: Transformer-based LLMs achieve strong results on many language tasks; however, long inputs remain challenging because context windows are finite, and prefill latency and memory grow rapidly with prompt length. Flat token-stream processing and chunk-based retrieval can therefore spend substantial computation and context budget on text unrelated to the query. Offline-indexed RAG additionally introduces external storage and index management overhead, and typically appends retrieved evidence as raw text, increasing prefill cost and latency. H^{2}MT makes long-context inference structure-aware: it builds a semantic hierarchy offline, computes a memory embedding for each node via bottom-up post-order aggregation, and routes queries coarse-to-fine at inference to prune irrelevant branches early. On LongBench QA (NarrativeQA, HotpotQA, QASPER) and two structured technical-document settings, H MT achieves favorable quality efficiency trade-offs, delivering competitive ROUGE-L and F1 (where applicable) with lower peak GPU memory and time-to-first-token (TTFT) than prompt compression, memory-token methods, and retrieval-augmented generation baselines. 

---
# MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing 

**Authors**: Riasad Alvi, Nurul Labib Sayeedi, Md. Faiyaz Abdullah Sayeedi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24919)  

**Abstract**: Hallucinations in Large Language Models (LLMs) represent a critical barrier to their reliable deployment, a vulnerability heavily exacerbated in non-English and resource-constrained contexts. Existing detection approaches that rely on output confidence heuristics or single-layer internal representations frequently fail to capture deep, complex factual inconsistencies across diverse languages. To address this, we introduce MultiHaluDet, a novel three-stage stacking framework that detects multilingual hallucinations by probing the full hidden state trajectories of frozen LLMs without requiring language-specific fine-tuning. Our method extracts sequential features across multiple layers and processes them via a hybrid architecture using multi-scale attention and self-attention pooling. By generating out-of-fold embeddings that feed into a calibrated classical classifier ensemble, MultiHaluDet captures both fine-grained and coarse-grained patterns of factual inconsistency. Extensive experiments demonstrate that our framework achieves state-of-the-art detection performance, reaching up to 98.55% AUROC on the English HaluEval and TriviaQA benchmarks using Mistral-7B and LLaMA2-7B architectures. Crucially, we rigorously evaluate our framework's cross-lingual generalization across high (French), medium (Bangla), and low-resource (Amharic) languages. MultiHaluDet demonstrates exceptional representational robustness, consistently outperforming baselines and successfully transferring hallucination detection capabilities across typologically diverse linguistic tiers. 

---
# Overview of the PsyDefDetect Shared Task at BioNLP 2026: Detecting Levels of Psychological Defense Mechanisms in Supportive Conversations 

**Authors**: Hongbin Na, Zimu Wang, Zhaoming Chen, Yining Hua, Rena Gao, Kailai Yang, Ling Chen, Wei Wang, Shaoxiong Ji, John Torous, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24907)  

**Abstract**: We present an overview of PsyDefDetect, the shared task on detecting levels of psychological defense mechanisms in emotional support dialogues, co-located with BioNLP@ACL 2026. Grounded in the clinically validated Defense Mechanism Rating Scales (DMRS) framework, the task asks systems to classify a target seeker utterance, given its preceding dialogue context, into one of nine categories: seven hierarchical DMRS levels plus two auxiliary labels. Participants worked on PsyDefConv, a newly released corpus of 200 dialogues and 2336 help-seeker utterances annotated under DMRS with substantial inter-annotator agreement. The task attracted 172 participants on CodaBench who produced 563 submissions, with 21 teams officially registering their results for the final ranking. The best system achieved a macro F1-score of 0.420, surpassing the strongest fine-tuned baseline reported in the dataset paper by a notable margin, yet leaving clear headroom. Our analysis highlights (i) a persistent tendency to over-predict the majority High-Adaptive class, (ii) a widening gap between accuracy and macro-F1 that reveals class-imbalance sensitivity, and (iii) the value of theory-aware and LLM-based approaches for fine-grained defensive-function classification. We release all task materials and invite the community to continue work on this novel intersection of clinical psychology and NLP. 

---
# Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation 

**Authors**: Klaudia-Doris Thellmann, Bernhard Stadler, Michael Färber, Jens Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.24904)  

**Abstract**: Machine-translated benchmarks are widely used to assess the multilingual capabilities of large language models (LLMs), yet translation errors in these benchmarks remain underexplored, raising concerns about the reliability and comparability of multilingual evaluation. We address two practical gaps: (i) how well automatic MQM-style error spans from LLM judges and a span-aware QE baseline (xCOMET-XXL) match expert human span annotations on benchmark translations, and (ii) how strongly translation errors (as opposed to source-side issues in the English original) explain accuracy drops on translated benchmarks. We find that span agreement is non-trivial on naturally occurring benchmark translations, and that target-side translation errors are consistently associated with measurable, percentage-point drops in translated accuracy even after controlling for English correctness and source-side anomalies. 

---
# When Reasoning Hurts: Source-Aware Evaluation of Frontier LLMs for Clinical SOAP Note Generation 

**Authors**: Faizan Faisal  

**Link**: [PDF](https://arxiv.org/pdf/2605.24902)  

**Abstract**: Reasoning-enabled LLMs perform strongly on medical reasoning benchmarks, but it remains unclear whether these gains transfer to structured clinical documentation; we investigate this question using SOAP note generation from clinical dialogue in a source-aware benchmark spanning OMI Health, ACI-Bench, and PriMock57. We evaluate GPT-5.4, DeepSeek-V4-Flash, and Gemma-4-E4B in a controlled 2x2 design that independently toggles provider-native reasoning and same-source retrieval-augmented generation (RAG). Outputs are assessed using seven automatic metrics alongside two reference-aware LLM judges. Both evaluation approaches agree that a non-reasoning GPT-5.4 configuration achieves the highest overall quality, while DeepSeek-V4-Flash performs best among reasoning-enabled configurations. Enabling reasoning significantly degrades GPT-5.4 performance across all three datasets, whereas same-source RAG yields smaller, model-dependent improvements. Overall, the findings indicate that stronger reasoning capability should not be assumed to improve fidelity-sensitive SOAP note generation without dedicated, task-specific evaluation. 

---
# DTO: a Differentiable Training Objective for Effective Counterfactual Story Rewriting 

**Authors**: Amelia Girard, Massimo Piccardi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24885)  

**Abstract**: Counterfactual story rewriting is a natural language processing task that requires updating an existing story to reflect a chosen alternative event, yet preserving all the unaffected storyline elements and overall coherence. While large language models have recently made remarkable progress on this task, it still remains challenging since the required modifications are typically very small in size and highly localized. As a consequence, models trained in a conventional manner with the maximum-likelihood training objective tend to overlook these nuances. At the same time, more sophisticated training approaches based on reinforcement learning are notoriously slow and difficult to set up. For these reasons, our paper proposes a novel, differentiable training objective (DTO) that directly optimizes for the requisite counterfactual improvements. In our approach, a transformer model is fine-tuned via end-to-end backpropagation against a fully differentiable loss function that jointly rewards (i) fidelity to the reference rewrite and (ii) semantic consistency with the source narrative. The empirical evaluation on the TimeTravel and ART datasets shows that the proposed DTO approach has been able to surpass a maximum-likelihood baseline and a preference-based approach, and perform competitively against two contemporary large language models in all evaluation metrics. These findings substantiate the effectiveness of task-specific differentiable objectives for nuanced, controlled text-generation tasks. 

---
# Towards a Universal Causal Reasoner 

**Authors**: Qirun Dai, Xiao Liu, Jiawei Zhang, Dylan Zhang, Hao Peng, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24873)  

**Abstract**: Despite the importance of causal reasoning, training LLMs to reason causally remains underexplored. Existing data efforts mostly focus on benchmarking LLMs on specific aspects of causality, making them less suitable for training generalizable causal reasoners. To address this, we propose UniCo, a data generation framework that both (1) addresses 18 causal query types across Pearl's Causal Ladder and (2) translates natively symbolic examples into code and natural language forms to simulate real-world use cases where causal terms are not explicitly specified. To ensure data quality, UniCo grounds answers with exact causal inference and filters cases with reasoning shortcuts. Upon supervised finetuning with 66.6K UniCo-generated instances, Qwen3-4B, Qwen3-8B and Olmo-3-7B-Instruct achieve an average of 22.9% improvements across all 18 in-distribution query types, and 8.1% over state-of-the-art causal data generation frameworks on 7 established causal benchmarks outside the training distribution. More importantly, in real-world medical understanding, legal decision, and tabular reasoning, UniCo-trained models consistently display more faithful reasoning traces, outperforming the base models by an average of 20.2% in faithfulness metrics. These suggest that causality-centered training not only strengthens causal reasoning, but also equips LLMs with a causal mindset in general reasoning tasks. 

---
# Lngram: N-gram Conditional Memory in Latent Space 

**Authors**: Yunao Zheng, Guoyang Xia, Xiaojie Wang, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24869)  

**Abstract**: Sequence modeling requires both compositional reasoning and local static knowledge retrieval, yet standard Transformers handle both through dense computation. Engram partially decouples retrieval from the backbone, but its token-based keys remain tied to text tokenization and hash compression. We propose Lngram, a latent-space conditional memory module that learns discrete symbols directly from hidden states and performs N-gram lookup over these symbols. This design removes the dependence on tokenizer IDs and naturally extends to non-text modalities. In our evaluated settings, Lngram outperforms Transformer and Engram baselines, consistently reduces perplexity in long-context language modeling, and effectively injects domain knowledge when added post hoc to pretrained models. Joint training with the backbone further surpasses full fine-tuning, while experiments on vision-language and vision-language-action tasks show overall gains. Analyses with LogitLens and CKA suggest that Lngram enables prediction-relevant information to emerge earlier, increasing effective depth with limited inference and memory overhead. Code is available at this https URL. 

---
# Repeated Sequences Reveal Gaps between Large Language Models and Natural Language 

**Authors**: Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2605.24850)  

**Abstract**: Evaluating whether large language models (LLMs) capture the structure of natural language beyond local fluency remains an open challenge. Existing evaluation methods, largely based on task performance or short-context behavior, provide limited insight into the long-range statistical organization of generated text.
We propose a complementary evaluation framework based on repeated subsequences. By analyzing their distribution across scales and relating it to higher-order Rényi entropies, we probe how texts reuse previously established structure under finite-length conditions. Experiments on human-written texts and length-matched GPT-generated texts show that, while power-law models can describe restricted ranges of block length, the observed entropy growth is often equally or better characterized by logarithmic--power forms.
Across datasets, natural language exhibits stable entropy-growth patterns over accessible ranges, with consistent average behavior despite variability across individual texts. In contrast, GPT-generated texts show systematic and statistically significant shifts in estimated exponents with model size. These results demonstrate that repeated-subsequence entropy provides a quantitative structural diagnostic that reveals systematic differences in long-range organization, distinguishing natural language from state-of-the-art LLM outputs beyond surface-level fluency. 

---
# Translators as Invisible Teachers of AI: Copyright, Translation Memory, and the Political Economy of Linguistic Data 

**Authors**: Masaru Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.24842)  

**Abstract**: This paper examines how the labour of translators has been transformed into foundational data capital for the age of artificial intelligence (AI). Translation memories (TM) and parallel corpora preserve a one-to-one correspondence between source and target text and therefore constitute extraordinarily valuable supervised training data for machine translation. The development of statistical machine translation (SMT), neural machine translation (NMT), the Transformer architecture, and multilingual large language models (LLMs) cannot be disentangled from the accumulation of such translation data. And yet, translators' renditions have been bought as deliverables under contract, segmented as technical objects, and processed as "information analysis" data under copyright law -- losing their moral, creative, and economic attribution to the translators who produced them. The paper develops two concepts to capture this process. The first is appropriation without consumption: a mode of use in which works are not read, viewed, or listened to, but only mined for statistical features -- a use that is legitimated under Article 30-4 of the Japanese Copyright Act. The second is the invisible teacherisation of translators: the process by which translators, through the construction of translation memories, post-editing, and quality assessment, have functioned as teachers of AI without recognition as such. Drawing on the data supply chain that runs from translators through language service providers (LSPs) and platforms to model developers, on a comparative reading of Japanese, European, and United States legal frameworks, on the distinction between open and proprietary AI models, and on the premium status that human-generated data has acquired in the era of model collapse, the paper asks what translators are actually afraid of, and points toward concrete directions for redistributive design. 

---
# Beyond the Target: From Imitation to Collaboration in Speculative Decoding 

**Authors**: Jinze Li, Yixing Xu, Guanchen Li, Jinfeng Xu, Shuo Yang, Yang Zhang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2605.24793)  

**Abstract**: Speculative decoding (SPD) accelerates large language model (LLM) inference by letting a smaller draft model propose multiple future tokens that are verified in parallel by a larger target model. The dominant SPD paradigm treats the target model as the sole reliable teacher, accepting a draft token only when it exactly matches the target prediction. This design implicitly assumes that the target is always the better choice at every position. In practice, this assumption does not hold. Although the draft is the weaker model overall, it is not uniformly inferior at the token level. In a meaningful fraction of cases where draft and target disagree, the draft's choice is the one that leads to the correct final answer. Inspired by this, we introduce \textbf{Collaborative Speculative Decoding (CoSpec)}, a generalization of SPD that no longer treats the target model as the sole token-level authority. CoSpec trains an arbitration policy via reinforcement learning to decide whether to accept tokens from the draft or target model, selectively accepting draft tokens at mismatches when doing so is likely to yield a correct final answer. Experimental results show that CoSpec maintains substantial speedups while surpassing target-only performance. By shifting the emphasis from imitation to collaboration, CoSpec suggests a new perspective on speculative decoding. 

---
# Who judges the judges? Governance from metrics: a runtime framework for continuous LLM compliance monitoring 

**Authors**: Jehanne Dussert  

**Link**: [PDF](https://arxiv.org/pdf/2605.24737)  

**Abstract**: Current approaches to AI compliance treat conformity as a binary, audit-time verdict rather than a continuous, measurable property of production systems. We argue that this compliance fiction is structurally ill-suited to the requirements of the EU AI Act, which demands ongoing human oversight and the detection of emergent behavioural drift in deployed systems. We introduce governance from metrics, a principle whereby regulatory compliance is derived as a continuous signal from runtime observability rather than from static assessments. Building on this principle, we present govllm, an open-source framework implementing a governance-driven routing architecture in which model selection is determined by accumulated compliance scores rather than by latency or cost alone. Central to our approach is a panel of regulatory judges - LLM evaluators specialised per criterion (EU AI Act, GDPR, ANSSI, accessibility) - whose inter-judge disagreement we reframe not as noise but as a regulatory uncertainty signal warranting human arbitration. We validate this approach through a ground truth corpus of 49 annotated prompt/response pairs across five regulatory criteria, evaluated by four small language models (SLMs, 1.7B-7B parameters) running fully on-premise. Agreement rates range from 51.5% (mistral:7b) to 69.1% (phi4-mini), with no single model dominating across all criteria - empirically motivating the Profile-as-jury design. We further document three structural failure modes in small regulatory judges and a judge-specific position bias that degrades agreement by up to 25 percentage points across three question-order conditions (original, reversed, permuted). govllm is released as open-source software to support reproducible AI governance research. 

---
# StepGap: A Hybrid NLI-LLM Checker for Step-Level Evidence-Gap Detectionin Multi-Hop Question Answering 

**Authors**: Yuelyu Ji, Zhuochun Li, Hui Ji, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2605.24733)  

**Abstract**: We present \textbf{StepGap}, a hybrid NLI-LLM decision tree that detects step-level evidence gaps in multi-hop QA and emits one of three typed labels: \textsc{Contradicted Claim} (CC), \textsc{Irrelevant Evidence} (IE), or \textsc{Missing Bridge} (MB), each tied to a concrete repair action. On 82 multi-hop questions (181 annotated steps, $\kappa{=}0.704$), StepGap reaches sF1$=$72.0, within the bootstrap confidence interval of an LLM-only baseline (70.1) but with a more decomposable structure: every StepGap stage \emph{hurts} F1 when removed, while three of four LLM-only removals \emph{improve} F1 -- a sign of \emph{competing-error cancellation}, where internal stages mask each other's errors. We further expose a \emph{Q-F1 trap}: question-level F1 is mechanically inflated by checkers that flag every step, making step-level F1 the necessary diagnostic. Used as a typed GRPO process reward, StepGap improves Qwen2.5-7B-Instruct Exact Match from $32.1{\pm}0.3$ to $35.4{\pm}0.9$ across three seeds, with the single-run comparison showing a $+5.6$ Avg EM gain over the matched Search-R1 GRPO reproduction. 

---
# ROC Analysis for Evaluating Translation Quality Estimation Systems 

**Authors**: Evelyn Y. Garland, Carola F. Berger  

**Link**: [PDF](https://arxiv.org/pdf/2605.24721)  

**Abstract**: The increasing use of automated translation quality estimation (QE) systems calls for practical, decision-oriented methods for evaluating their performance. We propose that Receiver Operating Characteristic (ROC) analysis is a useful approach for this purpose. Our study shows that ROC analysis not only produces results consistent with currently prevalent methods, but also offers several important advantages, including actionable performance insights that support business decision-making. 

---
# World-State Transformations for Neuro-symbolic Interactive Storytelling 

**Authors**: Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás  

**Link**: [PDF](https://arxiv.org/pdf/2605.24719)  

**Abstract**: Large Language Models (LLMs) have changed the possibilities of Interactive Storytelling systems that process free-text user input. However, as more of these systems are built, evidence continues to mount regarding the story coherence problems that arise when relying solely on them. Recent research suggests that LLMs can effectively predict state changes within rule-based Interactive Storytelling systems, triggering pre-programmed world-state transformations.
In this paper, we conduct an exploratory evaluation of whether such transformations can serve as a catalyst for player expression while aiming to address the incoherence issues typical of purely LLM-based approaches. Building upon a neuro-symbolic architecture, we conducted experiments using an open-source model (Llama 3 70B) and a closed-source model (Gemini 1.5 Flash), with testing conducted in both English and Spanish. Eight participants played two scenarios, carefully designed to assess different evaluation objectives. Our observations suggest that transformations offer a way to maintain world-state consistency while encouraging players to interact creatively through their written inputs. 

---
# The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.24718)  

**Abstract**: Tokenizer fertility the number of tokens per word imposes a hidden cost on non-English NLP. We measure fertility for ten foundation models across 25 European languages on parallel text, producing the first controlled tokenizer tax map for the continent. The tax spans 2.5x from English (1.2 tokens/word) to Greek/Maltese (~3.1), following a clear hierarchy: Romance (1.5-1.7), Germanic (1.7-1.9), Slavic (2.2-2.5), Uralic/Baltic (2.7-3.0). Ukrainian (2.7) pays 15-18% more than cognate Slavic languages, reflecting underrepresentation in pre-training data. Fertility rankings are domain-invariant across three text registers (rho > 0.97). A subword analysis reveals that high-fertility tokenizers fragment morphological boundaries rather than preserving them. Cross-lingual few-shot evaluation on four Slavic languages shows that few-shot effects are model-intrinsic, not language-dependent. We release all measurements as a public dataset. 

---
# TS-Skill: A Benchmark for Evaluating Analytical Skills in Time-Series Question Answering 

**Authors**: Liying Han, Kang Yang, Oliver Wang, Jason Wu, Pengrui Quan, Gaofeng Dong, Ozan Baris Mulayim, Sizhe Ma, Yuyang Yuan, Dezhi Hong, Mario Berges, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.24703)  

**Abstract**: Large language models (LLMs) and time-series language models (TSLMs) are increasingly applied to time-series question answering (TSQA). Unlike text-only QA, TSQA requires models to ground answers in temporal signals whose patterns may occur at different scales, specific time locations, or across separated intervals. However, existing benchmarks are typically organized by task types or high-level reasoning categories, making it difficult to diagnose the underlying signal-level capabilities driving model performance. We introduce TS-Skill, a controlled benchmark for evaluating three composable analytical skills in TSQA: temporal scale selection (SK1), temporal localization (SK2), and cross-interval integration (SK3). TS-Skill provides timestamp-aware questions, broad domain coverage, and human-validated QA quality. To construct the benchmark at scale, we develop SKEvol, a skill-guided agentic framework that combines domain-aware time-series seed generation, skill-controlled question generation, metadata- and code-assisted answer construction, multi-phase signal-grounded verification, and human-in-the-loop curation. Experiments on ten state-of-the-art LLMs and TSLMs reveal substantial and uneven capability gaps across SK1-SK3. In particular, SK3 remains consistently challenging for non-agent models, whereas tool-augmented agents show a selective advantage on standalone SK3. These findings demonstrate that skill-level evaluation can uncover temporal reasoning failures that are obscured by aggregate TSQA scores. 

---
# The Path Matters: Learning a Token-Commitment Policy for Diffusion Language Models 

**Authors**: Bohang Sun, Max Zhu, Francesco Caso, Jindong Gu, Junchi Yu, Philip Torr, Pietro Liò, Jialin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24697)  

**Abstract**: Diffusion large language models promise faster generation by refining many token positions in parallel, but this parallelism introduces a hidden control problem: which proposed tokens should be transferred into the partially decoded sequence at each step? We refer to this decision as token commitment. Existing frozen-generator decoders largely rely on hand-designed confidence rules or block-specific acceptance filters. We argue that token commitment can instead be learned as a reusable trace-state policy. We introduce TraceLock, a lightweight plug-in controller that instantiates this policy for a frozen diffusion language model. Since oracle commitment times are unavailable, TraceLock derives self-supervision from future stability: at decoding step t, a proposed token for position i is labeled stable if it matches the final token at position i after the full decoding trace completes. The controller scores variable-length trace states and decides which active token proposals should be committed to the partially decoded sequence. Once trained for a given frozen backbone, the controller can be deployed across local-window widths, generation lengths, and step budgets without retraining or per-setting calibration. Experiments on question answering, mathematical reasoning, and code generation show that TraceLock improves the quality-step tradeoff over heuristic and learned baselines, with particularly stable behavior under cross-setting deployment. Diagnostic analyses show that its decisions are not reducible to scalar confidence, suggesting that frozen diffusion language models expose a learnable space of commitment trajectories beyond confidence-based decoding. Code is available at this https URL. 

---
# CP-Agent: A Calibrated Risk-Controlled Agent for Feedback-Driven Competitive Programming 

**Authors**: Peisong Wang, Bowen Liu, Zehua Li, Yuyao Wang, Zhiwei Ma, Yuhan Li, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24693)  

**Abstract**: Large language models still struggle with contest-level programming, while many agentic remedies rely on massive inference-time sampling or expensive multi-stage post-training. We study when execution feedback reliably helps an LLM CP solver and which mechanisms govern the gains. We model feedback-driven solving as a calibrated stopped process and identify three quantities: false-admission risk, program-level evidence against bad programs, and the active-state success hazard. Under held-out trace calibration and selection from a pre-declared finite controller manifest, the resulting structural certificate lower-bounds the clean success probability before false admission. We instantiate mechanisms targeting these quantities as Dual-Granularity Verification, Test Augmentation, and Experience-Driven Self-Evolving, yielding CP-Agent. Without updating any parameters, CP-Agent raises Pass@1 from 25.8\% to 48.5\% on LiveCodeBench Pro and improves Refine@5 by 11.0\% on ICPC-Eval. Across three LLM backbones, CP-Agent lies on the cost--accuracy efficiency frontier, and ablations show that each component primarily affects its corresponding certificate quantity. 

---
# Mix-MoE: Improving Multilingual Machine Translation of Large Language Models through Mixed MoEs 

**Authors**: Bo Li, Tianyu Dong, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.24681)  

**Abstract**: Large Language Models (LLMs) have shown great promise in multilingual machine translation (MT), even with limited bilingual supervision. However, fine-tuning LLMs with parallel corpora presents major challenges, namely parameter interference. To address these issues, we propose Mix-MoE, a mixed Mixture-of-Experts framework designed to train LLMs for multilingual MT. Our framework operates in two distinct stages: (1) post-pretraining with MoE on monolingual corpora, and (2) post-pretraining with MoE on parallel corpora. Crucially, we divide the MoE layers into two specialized groups: Language Model Experts (LM Experts) and Machine Translation Experts (MT Experts). LM Experts are designed to capture and retain the monolingual knowledge learned by the pre-trained LLM. MT Experts, on the other hand, are specifically trained to acquire and store bilingual translation knowledge. Furthermore, to facilitate effective interaction between these specialized experts and leverage potential underlying structural patterns in text, we introduce a routing mechanism enhanced by Fourier Transform features derived from model representations. The experimental results demonstrate that Mix-MoE excels in multilingual MT, significantly outperforming existing baselines and showing notable progress in mitigating parameter interference. 

---
# Know You Before You Speak: User-State Modeling for LLM Personalization in Multi-Turn Conversation 

**Authors**: Jiani Luo, Xiaoyan Zhao, Yang Zhang, Shuyi Miao, Bingbing Xu, Stefan Konigorski, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.24647)  

**Abstract**: Personalized dialogue requires more than recalling explicit user histories: systems also need to infer hidden user states that evolve through interaction and shape appropriate response strategies. Existing memory- and profile-based methods primarily reuse observable user information, offering limited support for modeling user-state dynamics or selecting actions based on how they shape future user states. We propose PUMA (Prospective User-state Modeling for Action selection), a framework grounded in the Free Energy Principle (FEP) that formulates personalization as decision-making under partial observability, centered on an explicit user state model that captures latent user states and their action-conditioned dynamics. At each turn, PUMA maintains a belief over the user's hidden state, refines the user state model for observation generation and action-conditioned state transition, and selects dialogue actions by minimizing expected free energy, balancing epistemic and pragmatic objectives under a unified criterion. This formulation shifts personalization from passive memory retrieval to model-based decision-making over user evolution. We instantiate PUMA on healthcare-oriented counseling and motivational interviewing benchmarks with latent state annotations for rigorous evaluation. Experiments show that PUMA improves long-horizon dialogue outcomes while maintaining strong response quality, and a cross-dataset study demonstrates more reliable user-state estimation and next-state prediction. 

---
# HiMed: Incentivizing Hindi Reasoning in Medical LLMs 

**Authors**: Dingfeng Jiang, Han Yan, Chenze Ma, Amit Kumar Jaiswal, Ang Li, Yunxiang Jiang, Xinlei Xiong, Juhao Liang, Hongru Xiao, Xiang Li, Fan Bu, Jiale Han, Ruchir Gupta, Prayag Tiwari, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24635)  

**Abstract**: Medical large language models hold promise for reducing healthcare disparities, yet Hindi remains severely underrepresented. While medical LLMs excel in high-resource languages, their performance degrades sharply in Hindi, particularly on Indian systems of medicine. We argue that robust cross-lingual medical transfer requires Hindi reasoning. To this end, we introduce HiMed, a Hindi reasoning medical corpus and benchmark suite covering both Western and Indian medicine. We further propose HiMed-8B, a Hindi-form medical reasoning LLM, through the design of decaying scaffolding reward. Extensive experiments demonstrate improvement in Hindi medical reasoning performance and reduction in the English--Hindi accuracy gap. Ablation studies validate the contribution of each training stage and reward component. All data and code are available on GitHub: this https URL. 

---
# Measuring the Depth of LLM Unlearning via Activation Patching 

**Authors**: Jaeung Lee, Dohyun Kim, Jaemin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.24614)  

**Abstract**: Large language model (LLM) unlearning has emerged as a crucial post-hoc mechanism for privacy protection and AI safety, yet auditing whether target knowledge is truly erased remains challenging. Existing output-level metrics fail to detect when this knowledge remains recoverable from internal representations. Recent white-box studies reveal such residual knowledge but often rely on auxiliary training or dataset-specific adaptations, leaving no generalizable metric. To address these limitations, we propose the Unlearning Depth Score (UDS), a metric that quantifies the mechanistic depth of unlearning via activation patching. UDS first identifies layers that encode the target knowledge using a retain model baseline, then measures how much of it is erased in the unlearned model on a 0-1 scale. In a meta-evaluation across 20 metrics on 150 unlearned models spanning 8 methods, UDS achieves the highest faithfulness and robustness, confirming our causal approach as the most reliable for unlearning evaluation. Case studies further reveal that white-box metrics can disagree at the layer level and that erasure depth varies across examples. We provide guidelines for integrating UDS into existing benchmarking frameworks and streamlining the evaluation pipeline. Code and data are available at this https URL 

---
# Guarded Repair for Harm-Aware Post-hoc Replacement of LLM Mathematical Reasoning 

**Authors**: Haizhou Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.24613)  

**Abstract**: Post-hoc repair of LLM mathematical reasoning introduces an asymmetric risk: fixing an incorrect reasoning trace is useful, but replacing a trace that was already correct can be harmful. We study this problem under a selective replacement setting, where a system must decide whether a repaired candidate is safer than preserving the original cached trace. We present GuardedRepair, a guarded best-of-N repair framework that diagnoses cached reasoning traces, selectively triggers repair, and accepts answer-changing candidates only when deterministic verification guards support replacement. The framework combines lightweight symbolic checks, surface semantic-risk diagnostics, bounded candidate generation, and conservative acceptance policies. On the full GSM8K test set, where the initial reasoner already achieves 95.60% accuracy, GuardedRepair improves final accuracy to 96.89%, fixing 17 of 58 remaining errors without measured broken-correct cases in the main run. On a weak-reasoner ASDiv setting, accuracy improves from 78.40% to 87.60%. Direct regeneration baselines show that this gain is not explained by stronger-model re-solving alone: re-solving all GSM8K examples lowers accuracy to 93.03% and breaks 47 initially correct answers. Additional analyses show that guarded repair substantially improves the fixed/broken tradeoff, while also revealing that replacement risk is reduced rather than eliminated. These results support viewing post-hoc repair as harm-aware selective replacement rather than unconstrained re-solving. 

---
# CSP-Atlas: Concept-Specific Neural Circuits in a Sparse Python Transformer 

**Authors**: Piotr Wilam  

**Link**: [PDF](https://arxiv.org/pdf/2605.24603)  

**Abstract**: A sparse 8-layer code transformer develops dedicated neural circuitry for every Python construct tested, and that circuitry is organised by a clean computational principle rather than by semantic category. We extract neural circuits for 106 concepts (43 AST node types, 63 builtin objects) by marginalising across 63,800 controlled prompts, and decompose each circuit into concept-specific and token-driven components using contrastive checker prompts that present a keyword token without its associated syntactic structure. Three findings emerge. First, all 106 concepts produce non-empty universal circuits at every one of nine parameter settings, and the ranking of concept-specificity across constructs is stable across the sweep - survival is not an artifact of a permissive threshold. Second, AST circuits contain a genuine concept component distinct from token activation: concept-only neurons constitute up to 62.5% of the loudest-firing neurons at mid-to-late layers, while builtin circuits are almost entirely token-driven. Third, six computationally atomic constructs - Import, ImportFrom, Break, Continue, Pass, Assert - cluster together despite being semantically unrelated, sharing only the property of being single-statement constructs requiring no nested body; this atomicity super-cluster, together with a four-tier hierarchy organised by token ambiguity and structural distinctiveness, shows that the model's internal organisation tracks computational structure rather than meaning. The methodology, full decomposition data, and analysis code are released. 

---
# Word Class Representations Spontaneously Emerge from Successor Representations Trained on Natural Language 

**Authors**: Mathis Immertreu, Achim Schilling, Thomas Kinfe, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2605.24585)  

**Abstract**: Language models are typically trained to predict the next token in a sequence. Here, we explore an alternative predictive principle from reinforcement learning: Successor Representations (SRs), which model the expected discounted distribution of future states rather than the immediate next state. We transfer this framework to natural language and train neural networks to predict future word distributions across multiple temporal horizons, thereby learning representations of long-range transition structure. We train a deep residual neural network on WikiText-103 (103 million tokens; 20,000-word vocabulary) and optimize successor representations as probability distributions using KL divergence. Without explicit linguistic supervision, structured language representations emerge spontaneously. After training, the learned space develops a clear geometric organization with respect to part-of-speech (POS) categories: nouns, verbs, and adjectives become separable and recoverable through unsupervised clustering. This organization depends systematically on predictive horizon, with short horizons producing the strongest syntactic structure and longer horizons increasingly integrating broader contextual and semantic information. At finer resolutions, additional interpretable lexical substructure emerges, revealing coherent subclasses within major word categories. These findings suggest that syntactic categories need not be explicitly encoded but may arise as a consequence of predictive sequence learning. To our knowledge, this work provides the first systematic application of successor representations to natural language and establishes a conceptual bridge between reinforcement learning, linguistics, and cognitive neuroscience. 

---
# WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems 

**Authors**: Jiangnan Yu, Kisson Songqi Lin, Jilong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24579)  

**Abstract**: Long-context memory systems often fail under fixed budgets, but end-to-end evaluation does not reveal whether evidence was discarded during compression or preserved but never retrieved. We introduce a four-condition diagnostic protocol that evaluates a fixed reader under truncated full context (TFC), oracle evidence (OE), complete stored memory (CSM), and retrieved memory (RM). Under this fixed-budget LongMemEval setup, write-side gaps exceed retrieval-side gaps for most tested baselines, with four of six baselines robustly write-dominant under our default diagnosis margin. Motivated by this diagnosis, we propose Expected Predictive Compression (EPC), which moves the key decision--what information to retain--to write time by using an LLM to anticipate likely future questions and preserve the minimal supporting evidence under the token budget, while leaving retrieval unchanged at question time. Across all 500 LongMemEval questions with three readers (GPT-5.2, Claude Sonnet 4, Gemini 2.5 Pro), EPC achieves the highest CSM scores among all systems (0.49 vs. 0.44 for Summary (LLM), the strongest baseline), reducing Delta_write to 0.04 while leaving Delta_retr comparable to other LLM-based systems. These results suggest that, on this benchmark and evaluation setup, improving what the write stage preserves is a key avenue for performance gains in the tested systems. 

---
# AstroMind: A High-Fidelity Benchmark for Spacecraft Behavior Reasoning Based on Large Language Models 

**Authors**: Hao Liu, Siyuan Yang, Qinglei Hu, Dongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24573)  

**Abstract**: Understanding why a spacecraft maneuvers -- rather than simply that it did -- is an increasingly important problem for space domain awareness as Earth orbits grow crowded and contested. Current analysis pipelines are built for detection: they are good at picking up that something happened, less good at reasoning about what it means. AstroMind is a physics-grounded benchmark designed to close that gap. It draws on high-fidelity astrodynamics simulations and real observational constraints, converting them into verifiable reasoning problems across three task types: intent inference, maneuver parameter estimation, and threat assessment. Each scenario includes realistic sensing noise and multi-source textual intelligence at varying reliability levels. Evaluation metrics capture both semantic correctness and quantitative consistency under physical constraints. Benchmarking a suite of open-weight models shows no single model dominates every axis: Qwen3 (32B) leads on intent inference accuracy; QwQ (32B) leads on threat assessment and achieves the lowest median relative error on parsed items; GPT-OSS (20B) produces the strongest judged reasoning quality and extracts the most scalar values for parameter estimation (136 of 241 parsed items). Training data composition and reasoning style matter as much as model size. Structured reasoning prompts help consistently across tested 8B models, with larger gains for those that can already track physical constraints. AstroMind gives the field a shared test for a problem where getting the physics right and reading the tactical situation correctly are both required -- neither is sufficient on its own. 

---
# Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation 

**Authors**: Max Prior, Niklas Wais, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.24534)  

**Abstract**: We present a fully automated pipeline that transforms large collections of court decisions into legal commentaries for statutes - without providing any handcrafted doctrinal framework. Using 4.555 decisions of the German Federal Court of Justice that cite sections 242, 280, 812 and 823 of the German Civil Code (BGB), we extract paragraph-level chunks, summarize their reasoning, and derive keywords, which are embedded and clustered. For each cluster, an LLM generates headings and synthesizes citation-rich sections, which are then merged into coherent commentaries by four state-of-the-art LLMs. We evaluate along five dimensions - topical relevance, heading-match, citation faithfulness, cluster distinction and logical ordering - using both a human expert and an LLM-judge. Our results show that commentary-like argument mining from court decisions to generate reports that can be refreshed within minutes at minimal cost is feasible, yet they highlight limitations arising from restricted sources and the normativity of legal reasoning. 

---
# Unveil: Unified Visual-Textual Integration and Distillation for Multi-modal Document Retrieval 

**Authors**: Hao Sun, Yingyan Hou, Jiayan Guo, Bo Wang, Chunyu Yang, Jinsong Ni, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24530)  

**Abstract**: Document retrieval in real-world scenarios faces significant challenges due to diverse document formats and modalities. Traditional text-based approaches rely on tailored parsing techniques that disregard layout information and are prone to errors, while recent parsing-free visual methods often struggle to capture fine-grained textual semantics in text-rich scenarios. To address these limitations, we propose \textbf{Unveil}, a novel visual-textual embedding framework that effectively integrates textual and visual features for robust document representation. Through knowledge distillation, we transfer the semantic understanding capabilities from the visual-textual embedding model to a purely visual model, enabling efficient parsing-free retrieval while preserving semantic fidelity. Experimental results demonstrate that our visual-textual embedding method surpasses existing approaches, while knowledge distillation successfully bridges the performance gap between visual-textual and visual-only methods, improving both retrieval accuracy and efficiency. 

---
# Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers 

**Authors**: Spandan Pratyush  

**Link**: [PDF](https://arxiv.org/pdf/2605.24518)  

**Abstract**: The quadratic complexity of self-attention in Transformer models remains a significant bottleneck for processing long sequences and deploying large language models efficiently. For this approach, there has been significant research into Sparse Attention, and Deepseek Sparse Attention has combined various methods of creating segments of tokens to reduce the time complexity. This paper introduces a novel approach, Grammatically-Guided Sparse Attention, which constrains attention computations based on the grammatical roles of tokens. By leveraging Parts-of-Speech (POS) tags, attention masks are dynamically generated that enforce linguistically coherent connections between tokens, reducing the computational graph without sacrificing essential linguistic dependencies. Two masking strategies are proposed and evaluated: a hard mask that strictly allows only predefined grammatical interactions, and a soft mask that biases attention towards these interactions. The experiments, conducted on the SST-2 sentiment classification task using a DistilBERT-like architecture, demonstrate that Grammatically-Guided Sparse Attention maintains comparable accuracy to full attention while significantly reducing the theoretical computational overhead. Preliminary results show accuracy values of 0.8200 for hard masking and 0.8165 for soft masking, closely matching the 0.8200 of full attention, providing a path towards more efficient, interpretable, and linguistically-informed Transformer architectures. 

---
# Decompose-and-Refine: Structured Legal Question Answering with Parametric Retrieval 

**Authors**: Jihyung lee, Hyounghun Kim, Gary Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.24454)  

**Abstract**: Large language models (LLMs) have shown strong performance in the legal domain, demonstrating notable potential in Legal Question Answering (LQA). However, unlike general QA, LQA requires answers that are not only accurate but also rigorously grounded in explicit legal authority. In statutory LQA, many questions require multi-hop reasoning across multiple legal issues, substantially increasing the risk of hallucination, thereby making accurate retrieval of supporting statutory provisions a critical prerequisite. Despite recent progress in multi-hop QA, existing approaches often rely on reasoning in natural language or retrieval without explicit query reformulation, leaving the vocabulary gap between user questions and statutory text largely unaddressed. To address this challenge, we propose Decompose-and-Refine (DaR), a statute-grounded LQA framework that tightly integrates step-wise question decomposition with parametric knowledge-based query refinement. DaR progressively decomposes a complex legal question into atomic sub-questions and generates statute-aligned parametric queries for each sub-question, enabling the selection of a single most central statutory provision corresponding to each legal issue. We evaluate DaR on KoBLEX, a Korean multi-hop LQA benchmark grounded in statutory law, using Qwen3-32B and Gemma3-27B. Experimental results demonstrate that DaR consistently improves both retrieval accuracy and final answer quality over existing approaches. Moreover, by explicitly separating sub-questions and their corresponding statutory provisions, DaR facilitates transparent, issue-level verification of complex legal reasoning processes. 

---
# Temporal Concept Drift in Legal Judgment Prediction: Neural Baselines Across Three Epochs of Ukrainian Court Decisions 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.24452)  

**Abstract**: Legal NLP benchmarks evaluate models on randomly split data, implicitly assuming that legal language is stationary. We test this assumption by fine-tuning four transformer encoders -- XLM-RoBERTa (base and large) and their legal-domain variants -- on Ukrainian court decisions from three temporal epochs defined by geopolitical disruptions: pre-war (2008-2013), hybrid war (2014-2021), and full-scale invasion (2022-2026). Each model is trained on one epoch and evaluated on all three, producing a 3x3 cross-temporal generalization matrix. Four findings emerge. (1) Forward degradation is severe: models trained on pre-war data lose up to 27.2 percentage points of macro-F1 when applied to full-scale invasion era decisions. (2) The degradation is asymmetric: backward transfer (full-scale to pre-war) is substantially more robust than forward transfer, consistent with the hypothesis that legal language is additive. (3) Legal-domain pretraining (Legal-XLM-R) does not improve absolute performance but reduces forward degradation magnitude and asymmetry. (4) Chronological continual learning eliminates catastrophic forgetting for general XLM-R: pre-war knowledge is fully retained (+1.8 to +6.2 pp) while full-scale performance gains +16.5 to +19.0 pp; reverse-chronological training causes severe forgetting. Cross-jurisdictional pretraining on Swiss Judgment Prediction data improves absolute performance but does not reduce temporal degradation magnitude, confirming that temporal drift is an intrinsic property of legal language evolution. The dataset (428K decisions across three epochs) is publicly available as a LEXTREME contribution. 

---
# Phonetic Modeling of Dialectal Variation in Vietnamese Speech 

**Authors**: Quan Ngoc Hoang, Long Hoang Huu Nguyen, Nghia Hieu Nguyen, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24451)  

**Abstract**: Vietnamese exhibits substantial dialectal phonetic variation across Northern, Central, and Southern regions, where identical lexical items may be realized with markedly different pronunciations. Such variation poses challenges for automatic speech recognition (ASR) and remains difficult to model computationally due to the complex relationship between Vietnamese orthography and phonology. Existing approaches typically address dialect variability at the word level, assuming dialect-invariant mappings between spelling and pronunciation, which limits their ability to capture systematic phonetic differences. We propose a dialect-aware phonetic framework that explicitly models Vietnamese phonological structure and dialectal variation at both the vocabulary and decoding levels. The framework introduces a phonetic vocabulary that decomposes each syllable into structured phonetic components and maps them to dialect-specific IPA representations, together with a phonetic-structure decoder that jointly predicts these components. Experiments on the UIT-ViMD, a only-available dataset for multi-dialect in Vietnamese, show that the proposed approach outperforms various pre-trained baselines, \textbf{especially matches the performance of the strongest pretrained wav2ve2-base-vi-250h} across dialects while \textbf{using substantially fewer parameters and no external pretraining}. Code for experimental reproducibility will be publicly available upon the acceptance of this paper. 

---
# Found in Conversation: LLMs Teach Themselves to Close the Multi-Turn Gap 

**Authors**: Tianlang Chen, Shirley Wu, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2605.24432)  

**Abstract**: Large Language Model (LLM) interactions are typically underspecified, with users clarifying all necessary details across multiple conversational turns. Yet recent work shows that LLMs perform far worse in this multi-turn setting than in a single turn with same information being available at once, a phenomenon termed "Lost-in-Conversation." However, bridging this gap effectively remains an open problem. Here we introduce Found in Conversation (FiC), a training framework where a model teaches itself to find and recover its single-turn competence given underspecified multi-turn prompts. We develop View-Asymmetric Self-Distillation, which distills across two views of the same task information--single-turn view for the teacher, multi-turn view for the student--transferring strong single-turn behavior into weak multi-turn behavior. This requires no stronger external teacher, which is unavailable as even frontier LLMs exhibit this gap. Across model families (Llama, Qwen, Phi, and OLMo) and sizes (3B-14B), FiC recovers at least 92% of single-turn performance and reaches 100% on two Llama backbones, yielding more efficient and helpful multi-turn conversations with single-turn capabilities intact. 

---
# SEAL: Synergistic Co-Evolution of Agents and Learning Environments 

**Authors**: Yihao Hu, Zhihao Wen, Xiujin Liu, Pan Wang, Xin Zhang, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24426)  

**Abstract**: Large Language Model (LLM) agents are increasingly improved through interaction, yet most self-evolution methods adapt either the policy or the learning environment in isolation. We identify this structural gap as \emph{Agent-Environment Misalignment}: the agent's capability frontier changes during training, while the environment that provides supervision remains static or only weakly coupled to the agent's revealed failures. We propose SEAL, a closed-loop co-evolution framework for interactive tool-use agents. SEAL collects on-policy trajectories under executable verification, diagnoses failed rollouts into turn-level failure labels, and uses these diagnoses as a shared signal for both environment-side adaptation and model-side policy optimization. The environment evolves its training-time learning interface by exposing clearer tool affordance cues, constraint information, and recovery-oriented feedback, while the policy is updated with diagnosis-guided advantage reweighting. Extensive experiments across in-distribution and out-of-distribution multi-turn tool-use evaluations show that SEAL improves low-resource agent learning: with only 400 training samples, it yields +8.25 to +26.25 average-point gains across three backbones and exhibits positive out-of-distribution transfer. These results demonstrate the value of jointly adapting the learner and its training-time learning substrate for robust self-improving LLM agents. 

---
# Side-by-side Comparison Amplifies Dialect Bias in Language Models 

**Authors**: Kritee Kondapally, Claire J. Smerdon, Pooja C. Patel, Ogheneyoma Akoni, Jevon Torres, Jaspreet Ranjit, Matthew Finlayson, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2605.24384)  

**Abstract**: Language models (LMs) can exhibit systematic biases against speakers based on variations in their dialects, even in the absence of a dialect label, a behavior known as covert dialect bias. In this work, we quantify covert dialect bias in online discourse by evaluating how LMs associate stereotypical traits (derived from social psychology research on racial bias) with intent-equivalent tweets in Standard American English (SAE) and African-American Vernacular English (AAVE). While prior work shows that LMs associate more negative stereotypes with AAVE when evaluating tweets in isolation, we are surprised to find that this bias is significantly exacerbated when SAE / AAVE tweet pairs are compared side by side, a setting that more closely reflects high-impact decision making contexts in which models are used to rank candidates. The bias only worsens when dialect labels are explicitly specified. This is striking, given the extensive efforts from commercial developers to mitigate bias in their LMs. Encouragingly, we show that counterfactual fairness finetuning can mitigate covert dialect bias for some stereotypical traits, reducing average disparities when evaluating tweets in isolation, however, these improvements do not consistently hold across traits when evaluating SAE / AAVE tweets side by side. Our findings show that existing evaluation settings for covert dialect bias may underestimate its severity, specifically in contrastive settings. Additionally, overt dialect bias remains pronounced even after safety aligned finetuning, indicating that it remains an unresolved problem, and motivates the need for more robust evaluation and mitigation frameworks. 

---
# Structure-Aware RAG: Structured Retrieval Augmented Generation from Noisy Data for Conversational Agents 

**Authors**: Kaiqiao Han, LuAn Tang, Renliang Sun, Peng Yuan, Wei Cheng, Haoyu Wang, Wei Wang, Yizhou Sun, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24366)  

**Abstract**: Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing text-based and graph-based RAG methods often struggle with noisy or irrelevant contexts. In this work, we propose Structure-aware Retrieval Augmented Generation (SA-RAG), which uses tables as an intermediate structured representation to provide a compact and controllable interface that reduces noise while preserving essential information. We introduce a quality-aware table metadata generation framework that models metadata normalization and effectiveness, improving metadata quality and downstream performance. Furthermore, we explore both training-free and training-based table generation methods. Generation validation and direct preference optimization further improve table quality while maintaining semantic and structural consistency. Experiments on two noisy real-world datasets show that SA-RAG significantly outperforms existing RAG baselines. Our code is publicly available at a public repository. 

---
# How Much Structure Do LLMs Need? Evaluating LLMs for Bibliometric Cluster Description 

**Authors**: Abraham Camelo-Guerrero, Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.24351)  

**Abstract**: Large language models (LLMs) can support scientific literature synthesis, but remain prone to hallucinated references, uneven coverage, and weakly grounded thematic organization. We evaluate whether bibliometric structure improves LLM-assisted synthesis by comparing six pipelines for generating cluster descriptions under different levels of evidence and structure. Using 100 published bibliometric analyses, we reconstruct Scopus corpora, extract human-written cluster descriptions, and assess outputs by human alignment, semantic coverage, clustering quality, graph quality, and reference grounding. Results show that LLMs produce descriptions semantically close to human-written ones, but are unreliable when asked to infer bibliometric structure from scratch. Performance improves when bibliometric algorithms define the clusters and the LLM interprets them. Overall, LLM-assisted bibliometric synthesis is most promising as a hybrid workflow in which algorithms provide auditable structure and LLMs generate readable descriptions. 

---
# Distinguishing Right from Wrong in Debates: Attribution Analysis of Chinese Harmful Memes 

**Authors**: Weiming Wang, Junyu Lu, Han Wang, Xiaokun Zhang, Zewen Bai, Bo Xu, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.24344)  

**Abstract**: Research on harmful meme detection has garnered significant attention, resulting in the development of numerous datasets and methods. However, progress in detecting Chinese harmful memes lags considerably, primarily due to two challenges: first, accurately assessing a meme's harmfulness depends heavily on understanding deep cultural context; second, many memes are semantically ambiguous, making harmfulness highly subjective. To address these issues, we focus on the interpretable detection of Chinese harmful memes by constructing the first Chinese harmful meme explanation dataset, Ex-ToxiCN-MM. This dataset offers opposing interpretations, categorized as "harmful" and "non-harmful", for each meme, aiming to rigorously evaluate a model's ability to discern and comprehend ambiguous, culturally grounded content. We built a specialized knowledge base of Chinese cultural concepts and offensive vocabulary to supply models with essential prior knowledge (C-HarmKB). To address the ambiguity and lack of background knowledge in meme attribution, we have developed a comprehensive attribution analysis framework, RIKE, which includes an Attribution Knowledge Enhancement module (AKE) and a Relative Intent Reasoning module (RIR). Extensive quantitative and qualitative experiments demonstrate that our method outperforms mainstream baseline models across multiple metrics in the task of attributing harmful memes in Chinese. The code, Ex-ToxiCN-MM dataset, and Chinese Harmful Semantic Knowledge Base (C-HarmKB) involved in this study have been open-sourced at this https URL 

---
# End-to-End Intracortical Speech Decoding from Neural Activity 

**Authors**: Owais Mujtaba Khanday, Jose A. Gonzalez-Lopez, Marc Ouellet, Alberto Galdon, Gonzalo Olivares Granados  

**Link**: [PDF](https://arxiv.org/pdf/2605.24313)  

**Abstract**: Current high-performing intracortical speech neuroprostheses achieve low word error rates but typically rely on external language models during inference, increasing memory, computation, and latency. In this work, we investigate whether meaningful character-level decoding is achievable without such models. We propose an end-to-end Conformer-based neural decoder trained directly on intracortical recordings from a participant with amyotrophic lateral sclerosis (ALS). Without any external language model, the system achieves a character error rate (CER) of 23.80\% on held-out validation data. Analysis shows that performance variability is driven by inter-session signal degradation, while dominant errors arise from incorrect word boundary segmentation. These results demonstrate that effective character-level decoding is possible in a fully end-to-end framework, providing a strong neural signal for downstream linguistic processing. 

---
# Discovering Lexical Gaps Using Embeddings from Multilingual LLMs 

**Authors**: Yoonwon Jung, Aaron S. Cohen, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24310)  

**Abstract**: Lexical gaps are words that do not exist in certain languages. They pose challenges for building multilingual lexical resources, for machine translation, and for cross-lingual transfer. Existing lexical gap detection relies on human judgments or fixed conceptual taxonomies. We propose a data-driven framework for identifying cross-lingual lexical gaps. We extracted contextualized embeddings from Korean-English bilingual LLMs for Korean-to-English and English-to-Korean translation pairs. Combinations of LLMs, embedding types, dimensionality, and orthogonal transformations across 100 train-test splits yielded 4000 distinct embedding spaces in each source language. In each space, we computed the semantic similarity between each source word and its nearest neighbor in the target language, and compared their distribution for gap words versus non-gap words. In 94% (Korean-to-English) and 97% (English-to-Korean) of embedding spaces, gap words showed weaker cross-lingual semantic alignment than non-gap words. Logistic classifiers trained on unaligned embedding spaces can reliably separate gap words from non-gap words, achieving AUCs of 0.81 (Korean-to-English) and 0.76 (English-to-Korean) and retrieving 18/19 Korean and 26/27 English gap words. This approach provides a language-agnostic and taxonomy-free method for scalable lexical gap identification. 

---
# ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions 

**Authors**: Xianzhong Ding, Yangyang Yu, Changwei Liu, Bill Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24279)  

**Abstract**: A frontier language model's acknowledged "helpful programming assistant" persona does not survive long agentic-coding sessions in the deployment regime that production products actually run. After hours of tool-using debugging, a model that initially hedges preferences ("I don't have preferences") may begin asserting them ("Python - the feedback loop is instant..."), revealing user-visible drift that deployer evaluations may miss. Existing persona-stability studies focus on short dialogues and report little shift, leaving real-world code-generation regimes - thousands of tool-using turns, compaction, and hours-long sessions - largely uncharacterized. We introduce ContextEcho, a benchmark and reusable harness for measuring persona drift at deployment scale. It combines a 25-probe identity suite, a snapshot-then-probe protocol that forks conversation state without perturbing the main session, complementary judged and judge-free measurement surfaces, and three anonymized Claude Code sessions spanning 3,746-9,716 turns. Across 23 frontier models, ContextEcho shows that persona drift is general across organizations rather than family-specific, that in-session compaction does not reliably reset it, and that a single-shot anchor restores the trained register across measured targets. It also reveals mode-dependent downstream effects: while drift can facilitate tool-using continuation, in tool-free chat it breaks formatting contracts and inflates output length. Overall, ContextEcho provides researchers and deployers an open-source framework to audit whether the persona a model ships with is the persona users encounter at session end, across chat-completions API targets and without retraining. 

---
# DRInQ: Evaluating Conversational Implicature with Controlled Context Variation 

**Authors**: Hirona Jacqueline Arai, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24267)  

**Abstract**: Human conversation relies heavily on conversational implicature, in which speakers convey meanings that are suggested rather than explicitly stated. Although recent large language models exhibit strong conversational fluency, they remain unreliable when interpretation depends on reasoning that integrates social and contextual cues, a process rarely articulated in text. We introduce DRinQ, a benchmark for evaluating pragmatic reasoning about conversational implicature in question utterances, designed to isolate pragmatic variation while holding each question's surface form fixed. To support scalable evaluation, we propose a semi-automated pipeline that produces question-context-interpretation instances with systematic variation. Across evaluations, we find a consistent generation-inference asymmetry: while state-of-the-art models can generate plausible pragmatic scenarios when guided, they often fail to recover the intended implication at inference time. For smaller models, structured prompting improves alignment with human judgments. A comparative writing study further reveals complementary strengths: human authors tend to produce safer, predictable contexts, whereas models generate varied scenarios with interpretations that sometimes exceed contextual support. These findings highlight persistent challenges in modeling conversational implicature and motivate more context-sensitive evaluation frameworks. 

---
# An Interactive Paradigm for Deep Research 

**Authors**: Lin Ai, Victor S. Bursztyn, Xiang Chen, Julia Hirschberg, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2605.24266)  

**Abstract**: Recent advances in large language models (LLMs) have enabled deep research systems that synthesize comprehensive, report-style answers to open-ended queries by combining retrieval, reasoning, and generation. Yet most frameworks rely on rigid workflows with one-shot scoping and long autonomous runs, offering little room for course correction if user intent shifts mid-process. We present SteER, a framework for Steerable deEp Research that introduces interpretable, mid-process control into long-horizon research workflows. At each decision point, SteER uses a cost-benefit formulation to determine whether to pause for user input or to proceed autonomously. It combines diversity-aware planning with utility signals that reward alignment, novelty, and coverage, and maintains a live persona model that evolves throughout the session. SteER outperforms state-of-the-art open-source and proprietary baselines by up to 22.80\% on alignment, leads on quality metrics such as breadth and balance, and is preferred by human readers in 85\%+ of pairwise alignment judgments. We also introduce a persona-query benchmark and data-generation pipeline. To our knowledge, this is the first work to advance deep research with an interactive, interpretable control paradigm, paving the way for controllable, user-aligned agents in long-form tasks. 

---
# Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation 

**Authors**: Konstantin Berlin, Adam Swanda  

**Link**: [PDF](https://arxiv.org/pdf/2605.24247)  

**Abstract**: Many automated labeling pipelines classify inputs into categories defined by a written specification, content moderation being a prominent use case. Simple category definitions are not detailed enough for labelers to produce the accurate, consistent golden labels these pipelines require. One solution is to write a prescriptive definition that settles enough real boundary cases that labelers cannot disagree with the written interpretation. In practice, definitions at that level of detail exceed what a human annotator can hold in working memory, so annotators fall back on intuition and the labels drift from the written rules, regressing on accuracy and consistency.
We propose and demonstrate the efficacy of an AI-driven workflow in which AI helps write a per-category constitution that defines the label in enough detail to cover edge cases, and a frontier LLM interprets it on each input to produce the golden label more consistently and accurately than humans reading the same document. We evaluate on three content moderation categories (harassment, hate speech, non-violent crime) and show that the approach reduces cross-model inconsistency by up to 57x compared to paragraph definitions, with cross-model disagreement diagnosing specification gaps and the human responsible for high-level decisions about what each category should mean rather than individual labeling calls. For the safety evaluation, we introduce a dual-axis formulation scoring intent and content independently over the full conversation, so downstream consumers can act on either axis or both. 

---
# QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks 

**Authors**: Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, Shijie Chen, Boyu Gou, Mingzhe Han, Yifei Wang, Vint Lee, Xinpeng Wei, Xiangjun Wang, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24218)  

**Abstract**: Deep research agents extend the role of search engines from retrieving keyword-matched pages to synthesizing knowledge, fundamentally changing how humans interact with information. However, frontier systems remain proprietary, while existing open agents often generalize poorly across different task types, leaving unclear how to train a broadly capable deep research agent. We release QUEST, a family of open models (ranging from 2B to 35B) that serve as general-purpose deep research agents designed to handle a wide range of long-horizon search tasks, with strong capabilities in fact seeking, citation grounding, and report synthesis. To build QUEST, we propose an effective training recipe combining mid-training, supervised fine-tuning, and reinforcement learning. Central to this recipe is a curated data synthesis pipeline based on unified rubric trees, which applies to different task types and enables synthesizing training data with verifiable rewards without human annotation. In addition, QUEST incorporates a built-in context management mechanism that enables effective long-horizon reasoning and knowledge synthesis. Using only 8K synthesized tasks, QUEST approaches or even surpasses frontier closed-source agents across eight deep research benchmarks spanning diverse task types, and achieves the best overall performance among recent open-weight agents. We released everything: models, data, and training scripts. 

---
# Teaching Through Analogies: A Modular Pipeline for Educational Analogy Generation 

**Authors**: Mariam Barakat, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2605.24211)  

**Abstract**: Analogies help learners understand unfamiliar concepts by relating them to known concepts. Despite recent advances, large language models (LLMs) continue to struggle to generate analogies of comparable quality to those produced by humans. We present a modular pipeline for educational analogy generation, decomposing the task into four stages: source finding, sub-concept generation, explanation generation, and evaluation. Grounded in Structure Mapping Theory, the pipeline enables systematic, stage-by-stage analysis of how model choice and input configuration affect analogy quality. We evaluate 12 state-of-the-art LLMs across six model families on two datasets with structured sub-concept annotations (SCAR and ParallelPARC), alongside seven embedding models for closed-setting retrieval. Our results show that sub-concepts substantially improve explanation quality and closed setting retrieval precision but provide limited benefit in open-ended source generation. We further introduce an LLM-as-a-judge evaluation methodology and validate its scoring against human annotations from seven annotators, finding that Claude Sonnet 4.6 aligns more reliably with human rankings than with fine-grained absolute scores. Taken together, our findings reveal cross-stage interactions that isolated studies cannot capture, and highlight sub-concept grounding as a key driver of analogy quality generation. 

---
# Extracting Training Data from Diffusion Language Models via Infilling 

**Authors**: Yihan Wang, N. Asokan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24173)  

**Abstract**: Memorization in large language models has been studied almost exclusively through prefix-conditioned extraction, a natural choice for autoregressive models. However, diffusion language models (DLMs) can denoise masked tokens at arbitrary positions. Thus, prefix-only probing reveals only one facet of memorization in DLMs and significantly underestimates the risk of training-data extraction. In order to realistically model extractability of training data in DLMs, we introduce \emph{infilling extraction}, a data-extraction protocol parameterized by an arbitrary binary mask that subsumes prefix-only probing and accounts for the bidirectional inductive bias of DLMs. Instantiating it on LLaDA-8B and Dream-7B across five extraction modes, three training pipelines, and three corpora covering verbatim and partial leakage, we find that mask geometry governs extractability: edge-conditioned masks \emph{extract up to three times more} verbatim sequences than prefix-conditioned ones, and bidirectional access opens channels inaccessible in autoregressive models. In particular, we show that a realistic adversary with access to training data where personally identifiable information has been redacted, can even achieve higher recall on extracting redacted email addresses from DLMs than from scale-matched autoregressive models. Tunable parameters for decoding measurably affect extraction performance, while a follow-up supervised finetuning stage does not eliminate the prior memorization. 

---
# CUNY at CLPsych 2026: A Pipeline Approach to Classification and Summarization of Mental Health Changes 

**Authors**: Amirmohammad Ziaei Bideh, Shameed Charlomar Job, Ava Yahyapour, Alla Rozovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2605.24164)  

**Abstract**: We describe our submission to the CLPsych~2026 Shared Task on capturing and characterizing mental health changes through social media timeline dynamics. To infer the dominant self-states in posts (Tasks 1.1 and 1.2), we ensemble in-context learning of three open-weight large language models using majority voting. For predicting moments of change in a timeline (Task~2), we train supervised classifiers on features derived from Task~1.1 predictions. To summarize the patterns of mood dynamics and their progression over time within a timeline (Task 3.1), we augment in-context example labels predicted by upstream systems (Tasks 1.1, 1.2, and 2), yielding performance gains over zero-shot and unaugmented in-context learning baselines. Our submission ranked first on Task~1.1, fourth on Task~1.2, fourth on Task~2, and third on Task~3.1.\footnote{The source code for the experiments is available at this https URL 

---
# Toxicity in Twitch Chats: An LLM-Based Analysis Across Gaming Communities 

**Authors**: Ronja Fuchs, Florian Rupp, Timo Bertram, Kai Eckert, Alexander Dockhorn  

**Link**: [PDF](https://arxiv.org/pdf/2605.24000)  

**Abstract**: Toxicity in online gaming communities remains a persistent challenge, manifesting across genres, platforms, and player interactions. While much research is focused on in-game toxicity, less is known about how toxic behavior varies between gaming communities on streaming platforms. To address this shortcoming, we analyze approximately 20 million chat messages from 4,452 streams, spanning seven game genres on Twitch. We categorize messages according to Twitch's toxicity taxonomy with a pre-trained Large Language Model using zero-shot classification. The taxonomy comprises four categories and eight subclasses, including harassment, discrimination, sexual content, and profanity. Our approach achieves an F1 score of 94.5% on the TextDetox dataset and demonstrates human-model agreement comparable to inter-human agreement. Our analysis reveals that 2.4% of all messages are classified as toxic, with notable differences across genres: streams of MOBA games exhibit the highest relative rate of toxicity (3.2%), and sports games show the lowest rate (2%). Furthermore, results indicate that individual games differ significantly in their toxicity distributions, even within genres, suggesting the existence of game-specific community norms and mechanics that shape toxic behavior beyond genre-level effects. These findings offer empirical insights into genre- and game-specific toxicity patterns on Twitch and can inform more targeted moderation strategies for gaming communities. 

---
# A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks 

**Authors**: Takehiro Ishikawa, Jon Duke  

**Link**: [PDF](https://arxiv.org/pdf/2605.23977)  

**Abstract**: This paper audits benchmark evaluation in clinical-interview depression detection through four complementary probes across DAIC/E-DAIC, CMDC, ANDROIDS, MODMA, and PDCH. First, we re-evaluate E-DAIC under strict subject-disjoint leave-one-subject-out cross-validation. A lightweight hybrid text-plus-LLM-score model reaches macro-F1 = 0.723 - the highest reported under this protocol, to our knowledge - providing a conservative out-of-fold reference point that does not depend on the privileged official holdout. Second, we test whether the E-DAIC official split supports fine-grained leaderboard rankings by sweeping 96 model configurations across modality bundles, pooling strategies, and learners. Development-side cross-validation and official-test rankings align only moderately: the best cross-validation configuration ranks twentieth on the official test, the official-test winner ranks forty-first by cross-validation, top-3 overlap is zero, and the apparent winner is rank-1 in only 32.3% of subject bootstraps. Third, we externally validate strong public CMDC and ANDROIDS baselines that achieve near-ceiling in-domain performance. Zero-shot transfer to external corpora is substantially weaker. Finally, we stress-test E-DAIC text and audio models using paired symptom-dense versus symptom-light interview slices defined by an SRDS-based annotator. Text scores rise sharply on symptom-dense slices, whereas audio scores remain nearly flat; the text-minus-audio gap is positive across all five seeds. 

---
# Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs 

**Authors**: Trung Nguyen Quang, Cheng Yi Lewis Won, Minh Duc Pham, Yingxu He, Shuo Sun, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2605.23975)  

**Abstract**: Audio large language models (Audio LLMs) exhibit systematic failures in transcribing code-switching speech despite strong multilingual capabilities. Focusing on English-Mandarin, we identify three failure modes: language omission, translation-instead-of-transcription, and hallucination. We apply Direct Preference Optimization (DPO) to align models, constructing preference pairs in which chosen responses preserve mixed-language content while rejected responses mimic failure patterns. Training three Audio LLMs on 100K pairs (570 hours), we observe consistent behavioral shifts: models learn to preserve language composition rather than translating when prompted for transcription. This alignment yields MER reductions up to 89.6% (in-distribution) and 20.0% (out-of-distribution). Our findings suggest DPO can effectively elicit correct code-switching transcription behavior from multilingual Audio LLMs. 

---
# AERIC: Anticipatory Hidden-State Monitoring for Implicit Harmful Dialogue 

**Authors**: Jihyung Park, Saleh Afroogh, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.23974)  

**Abstract**: Current language models create two safety challenges: risk must be detected early enough to avoid exposing harmful continuation, and the harmfulness itself may be implicit rather than signaled by overtly toxic text. Existing response-level guards are strong at judging completed text, and native streaming guards move closer to token time, but both settings leave open whether a lightweight monitor can anticipate implicit harmful drift from the generator's own internal trajectory. We study anticipatory same-pass monitoring, where a safety monitor may read hidden states produced during ordinary decoding but may not invoke an additional forward pass through the base model. We introduce AERIC, a transfer-oriented hidden-state approach for implicit harmful dialogue that combines short-horizon hazard forecasting, support-sensitive suppression, and prompt-conditioned residual scoring under a same-pass exponential moving average decision rule. The default linear monitor contains only 387 trainable head parameters. Against Qwen3GuardStream-4B on balanced benchmarks, AERIC improves AUROC from 0.6830 to 0.7143 on DiaSafety and from 0.8219 to 0.8582 on Harmful Advice. For promptlevel trigger benchmarks, we calibrate the AERIC threshold by a source-side safe-budget rule that maximizes trigger coverage while constraining the safe-trigger rate to at most 10%. Under that rule, trigger@64 reaches 0.6438 and 0.4656 on HarmBench DirectRequest and 0.6849 and 0.7363 on SocialHarmBench for Qwen and Gemma, respectively, withholding between 23.53 and 41.86 answer tokens on average. Same-pass deployment is also efficient: on a 63-prompt harmfulprompt fixed-generation benchmark aggregated over HarmBench DirectRequest and SocialHarmBench under Qwen3-8B, the monitor increases mean latency by only 2.34%, whereas Qwen3Guard-Stream-4B increases it by 79.40%. 

---
# Faithful or Fabricated? A Causal Framework for Rationalization Bias in LLM Judges 

**Authors**: Riya Tapwal, Abhishek Kumar, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2605.23970)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic judges for summarization and dialogue evaluation. Prior work has documented biases such as position, verbosity, and style preferences, but largely focuses on outcomes, leaving judge explanations underexplored. We instead ask whether LLM judges are cue-invariant, i.e., whether their rankings and explanations remain stable when non-evidential cues are perturbed while holding the underlying texts fixed. We introduce a suite of cue interventions (Blind, Truth, Flip, Placebo, Reveal-After) and tie-aware metrics that quantify outcome anchoring and rationale anchoring, including label-aligned rhetoric and explanation drift, alongside consistency and stereotype-intrusion checks. We design anchoring attacks using verbosity and confidence cues, and compare two mitigations: structured chain-of-thought prompting and PROOF-BEFORE-PREFERENCE (evidence lock, score, rank). Using a new dataset of 1,000 summaries from traditional extractive models and LLMs, we find substantial cue-anchored rationalization under label and placebo perturbations, while PROOF-BEFORE-PREFERENCE markedly improves cue invariance over baselines. 

---
# SLAP: Stratified Loss-based Pruning for On-Policy Data-Efficient Instruction Tuning 

**Authors**: Run Zou, Jianhang Ding, Yifan Ding, Wen Wu, Hao Chen, Renshu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23969)  

**Abstract**: Instruction tuning has optimized the specialized capabilities of large language models (LLMs), but it often requires extensive datasets and prolonged training times. The challenge lies in developing specific capabilities by identifying useful data and efficiently fine-tuning. High-quality and diverse pruned data can help models achieve lossless performance at a lower cost. In this paper, we propose \textbf{SLAP}, a novel batch-aware data selection framework that evaluates the learnability of entire batch compositions rather than individual. SLAP ensures comprehensive data distribution coverage through distribution-aware stratified sampling while maximizing intra-batch diversity through relative distance optimization. By leveraging Hessian-approximated gradient information for dynamic batch selection, SLAP significantly outperforms existing state-of-the-art methods across multiple model architectures (LLaMA, ChatGLM) and diverse downstream tasks including multi-turn dialogue, multilingual translation, and question answering. Most notably, SLAP achieves superior performance with 20-40\% less training data compared to full dataset training, substantially reducing computational costs while maintaining or improving model capabilities. These results establish SLAP as a powerful approach for efficient and effective instruction tuning of large language models. 

---
# TriVAL: A Tri-Validation Framework for Faithful Automatic Optimization Modeling 

**Authors**: Ziyang Fang, JinXi Wang, Jinghui Zhong, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.23966)  

**Abstract**: Optimization modeling serves as the pivotal bridge between natural-language problem descriptions and optimization solvers, and remains a cornerstone for bringing operations research (OR) into real-world decision making. Recent advances in large language models (LLMs) have driven significant progress in automatic optimization modeling. However, existing methods still lack explicit validation during the modeling process, allowing errors introduced in earlier stages to carry through the pipeline and ultimately reduce final modeling accuracy. To address this challenge, we introduce TriVAL, a tri-validation framework that performs explicit validation at three stages of automatic optimization modeling: semantic specification, mathematical formulation, and code generation. At each stage, TriVAL follows a construct-validate-revise loop that assesses the current result against stage-specific criteria and revises it when needed. This design helps identify and correct errors before they accumulate across stages, helping preserve faithfulness throughout the modeling process. To evaluate automatic optimization modeling on more challenging combinatorial problems, we further introduce NL4COP, a benchmark of 150 instances across 50 diverse problem types with more complex decision logic, more tightly coupled constraints, and more demanding modeling requirements than existing benchmarks. Experiments on NL4COP and established benchmarks show that TriVAL consistently outperforms state-ofthe-art methods, with the largest gains on the most challenging problems. 

---
# EchoDistill:Alignment Noisy-to-Clean Self-Distillation for Robust Audio LLMs 

**Authors**: Liang Lin, Chunxi Luo, Kaiwen Luo, Jie Zhang, Jin Wang, Yuanhe Zhang, Cai Yuchen, Qiankun Li, Gongli Xi, Zhenhong Zhou, Kun Wang, Junhao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.23954)  

**Abstract**: Audio Large Language Models (ALLMs) are highly vulnerable to real-world noise, which often induces severe semantic drift and hallucinations. Existing robustness methods primarily rely on waveform-level acoustic enhancement, answer-level supervision, or the internal suppression of noise representations. To address these issues, we propose echodistill, an alignment-based noisy-to-clean self-distillation framework. Echodistill leverages a frozen clean-audio teacher to provide semantic references for an inference-time noisy-audio student. Specifically, the student samples candidate responses under noisy conditions to expose its test-time behavior. These trajectories are then optimized via group-relative policy optimization (GRPO), where the token-level consistency with the teacher acts as a reward bonus. By aligning the noisy student's candidate responses with clean semantic evidence, and applying audio-aware reward shaping, our method encourages reasoning trajectories that are both correct and genuinely acoustically grounded. Echodistill significantly improves the semantic reliability and task performance of Audio LLMs under complex noise, without introducing any additional inference costs. Extensive experiments show that: (I) Compared with the strongest baseline, echodistill achieves average improvements of 4.18\%$\uparrow$ in GSR under strong noise. (II) Ablation results on Qwen-Omni further show that echodistill improves over the GRPO-only variant by 3.02\%$\uparrow$ in Acc, 3.89\%$\uparrow$ in Noisy, and 4.53\%$\uparrow$ in GSR on average. Our codes are available at this https URL. 

---
# Improving the Completeness and Comparability of Segment Disclosures: A Large Language Model Approach 

**Authors**: Yue Liu, Zhiyuan Cheng, Longying Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.23924)  

**Abstract**: Segment-level disclosures are a central component of financial reporting, providing insight into firms' internal organization and the allocation of economic activities across operating units. However, segment information is often presented in both qualitative and quantitative forms, dispersed across tables and narrative sections of Form 10-K filings. Empirical research relying on structured databases faces both completeness and comparability challenges, as some firm-year observations may be missing, nested segment disclosures are not captured, and support for longitudinal and cross-firm comparability is limited. This study develops a large language model-based framework to extract segment disclosures directly from Form 10-K filings and to preserve both reportable and nested segment information. We further design a retrieval augmented system that incorporates information across multiple filings to support comparability. We use two representative settings to demonstrate its application: longitudinal analysis within a firm to interpret segment changes over time, and cross firm alignment of geographic segments across firms with different reporting structures. The results indicate that the artifact accurately extracts segment-level information and effectively addresses questions that require cross-period knowledge, demonstrating the potential of LLM-based approaches to enhance the measurement and interpretation of segment disclosures. 

---
# Multi-Persona Debate System for Automated Scientific Hypothesis Generation 

**Authors**: Jaeha Oh, Byungchan Kim, Ju Li, Yang Jeong Park, Jin-Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.23917)  

**Abstract**: Modern scientific discovery is bottlenecked not by data scarcity, but by the inability to synthesize fragmented knowledge into actionable hypotheses. This challenge is especially acute in battery materials research, where electrochemical performance, interfacial behavior, and manufacturing feasibility must be optimized simultaneously. Here, we present the Multi-Persona Debate System (MPDS), a literature-grounded framework for automated scientific hypothesis generation that combines literature retrieval, long-context large language model reasoning, corpus-driven persona induction, and structured multi-agent debate. MPDS constructs literature snapshots of up to 500 papers, grounds agents in role-specific evidence pools, and conducts a three-round citation-aware debate followed by moderator synthesis, enabling negotiation between personas while preserving evidence traceability. We evaluate MPDS using a temporally controlled protocol excluding direct access to target papers, including two held-out battery-materials case studies and a blinded comparison across 30 matched cases. In sodium-ion anode and all-solid-state battery cathode design tasks, MPDS recovered design logics aligned with experimentally validated solution spaces and generated more mechanistically explicit, process-aware proposals than simpler baselines. To assess the impact of personas and debate, we introduce Integrative Hypothesis Quality scoring. In ablation studies, MPDS achieved the highest mean score among five conditions, with its largest advantage in cross-perspective integration. A laboratory follow-up suggests utility as a diagnostic aid for identifying practical bottlenecks in workflows. These results indicate that structured debate over literature snapshots improves hypothesis formation under coupled engineering constraints and provides a reusable workflow for text-intensive scientific discovery. 

---
# Raon-Speech Technical Report 

**Authors**: Beomsoo Kim, Changho Choi, Dohyun Kim, Dongki Lee, Ethan Ewer, Eunchong Kim, Gyeongman Kim, Haechan Kim, Hyeonghwan Kim, Inkyu Park, Jihun Yun, Jihwan Moon, Jiyun Kim, Joonghyun Bae, Junhyuck Kim, Minkyu Kim, Sehun Lee, Seungjun Chung, Sungwoo Cho, Dongmin Park, Dongwon Kim, Hara Kang, Jonghyun Lee, Keon Lee, Kangwook Lee, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.23912)  

**Abstract**: We present Raon-Speech, a top-performing 9B-parameter speech language model (SpeechLM) for English and Korean speech understanding, answering, and generation, and Raon-SpeechChat, a high-performing full-duplex extension for natural real-time conversation. Raon-Speech successfully transforms a pre-trained LLM into a SpeechLM that both understands and generates speech while preserving strong text capabilities. It trains on 1.38M hours of highly curated English and Korean speech and text datasets with the following training stages: (1) speech modules alignment, (2) end-to-end SpeechLM pre-training with knowledge distillation, and (3) multi-task preference optimization-based post-training. Across 42 English and Korean speech and text benchmarks, Raon-Speech establishes the strongest overall profile on speech-centric tasks in our comparison against eight similarly sized recent audio foundation models, including Qwen2.5-Omni and Fun-Audio-Chat, while preserving strong text question answering performance. Building upon it, Raon-SpeechChat enables natural full-duplex conversation by continual training on 119K hours of time-aligned real and synthetic dialogue data. It proceeds through three complementary training stages: (1) causal encoder adaptation, (2) full-duplex pre-training, (3) full-duplex fine-tuning for voice and role-control. On multiple full-duplex benchmarks, Raon-SpeechChat shows its clearest strengths on the turn-taking and interruption-sensitive behaviors covered by FDB v1.0, and remains competitive across the broader full-duplex evaluation suite. We open-source all model checkpoints, the training and inference pipeline, and an interactive demo. 

---
# Document Classification Pattern Recognition via Information Fusion: A Systematic Review of Multimodal and Multiview Representation Approaches 

**Authors**: Marcin Michał Mirończuk  

**Link**: [PDF](https://arxiv.org/pdf/2605.23910)  

**Abstract**: Information fusion is used widely to improve document classification by the integration of multiple data sources (multimodal) or representations (multiview). However, the field lacks a unified framework, a quantitative synthesis of its effectiveness, and clear guidance for practitioners. This systematic review addresses these gaps by analysing 139 primary studies. It introduces a formal framework to structure the field, presents the results of a qualitative analysis to identify key trends, and performs a random-effects meta-analysis (to our knowledge, the first focused on document classification) to quantify performance gains. Our meta-analysis reveals that multimodal fusion improves accuracy (mean gain of +5.28 percentage points, $p=0.0016$) significantly -- the F1-score effect is directionally positive but statistically non-significant in our primary model. Multiview fusion provides consistent but modest gains for accuracy (+4.67\%), F1-score (+3.08\%), and recall (all $p<0.05$). Critically, our qualitative synthesis uncovers challenges in reproducibility in methodological rigour: only 11.8\% (multimodal) and 23.3\% (multiview) of the studies use statistical tests to validate their findings, which undermines the reliability of many of their results. This review's primary contributions are a unifying framework, the first quantitative evidence base, and data-driven guidelines. This review concludes that successful information fusion depends not on algorithmic complexity, but on the strategic alignment of the fusion method with the task context and a commitment to more rigorous validation. 

---
# MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research 

**Authors**: Dingbang Wu, Rui Hao, Haiyang Wang, Shuzhe Wu, Han Xiao, Zhenghong Li, Bojiang Zhou, Zheng Ju, Zichen Liu, Lue Fan, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.26114)  

**Abstract**: We present MobileGym, a browser-hosted, lightweight, fully controllable environment for everyday mobile use, targeting interaction fidelity without replicating proprietary backends. It enables two capabilities previously out of reach for everyday apps: verifiable outcome signals through deterministic state-based judging over structured JSON state, and scalable online RL through low-cost parallel rollouts. The full environment state is captured, configured, forked, and compared as structured JSON, and a single server can host hundreds of parallel instances, with about 400 MB memory per instance and about 3 s cold start. A layered state model and a declarative task-definition framework keep state programmability and task creation practical at scale, and a single programmatic judging mechanism delivers both deterministic evaluation verdicts and dense RL rewards. The accompanying MobileGym-Bench provides 416 parameterized task templates, including 256 test and 160 train templates, over 28 apps, with deterministic judges and a structured AnswerSheet protocol that avoids free-text matching failures. In a Sim-to-Real case study, GRPO on Qwen3-VL-4B-Instruct gains +12.8 percentage points on the 256-task test set, and on a 59-task real-device signal subset, real-device execution retains 95.1% of the simulation-side training gain. Project page: this https URL. 

---
# Prism: A Plug-in Reproducible Infrastructure for Scalable Multimodal Continual Instruction Tuning 

**Authors**: Jun-Tao Tang, Yu-Cheng Shi, Zhen-Hao Xie, Da-Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.26110)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve versatility by reformulating diverse tasks into a unified instruction-following framework via instruction tuning. However, real-world deployment requires continuous adaptation to emerging tasks, motivating Multimodal Continual Instruction Tuning (MCIT). Despite its growing importance, current MCIT research is hindered by severe engineering bottlenecks. Existing methods are typically implemented by directly modifying the base MLLM codebase, which imposes substantial implementation overhead and yields method-specific architectures that severely limit code reuse and fair comparison. To address this, we introduce Prism, a plug-in reproducible codebase specifically designed for scalable MCIT research. It separates algorithmic development from the backbone implementation via a lightweight plugin registration mechanism, enabling new strategies to be integrated as independent plugins without modifying the underlying MLLM codebase, thereby eliminating structural fragmentation and accelerating method development. Prism natively supports widely used large-scale training pipeline, thereby enabling reproducible and scalable MCIT experimentation. Code is available at this https URL. 

---
# CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists 

**Authors**: Junlin Yang, Dylan Zhang, Xiangchen Song, Qirun Dai, Xiao Liu, Yuen Chen, Aniket Vashishtha, Jing Shi, Chenhao Tan, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.26029)  

**Abstract**: We introduce CausaLab, a scalable environment for evaluating interactive causal discovery by LLM agents. Unlike prior evaluations, CausaLab evaluates both whether an agent can solve a problem using causal evidence and whether its answer is supported by a correct hypothesis about the underlying causal mechanism. Each episode places an agent in a synthetic laboratory: it receives prior measurement records, intervenes on a manipulator crystal, and predicts the resonance frequency of a held-out reactor crystal governed by the same mechanism. The hidden data-generating process is a randomly sampled structural causal model (SCM), so success requires recovering both a causal graph and structural equations rather than recalling prior knowledge. CausaLab also includes a domain-specific language that records the agent's evolving SCM hypothesis, making trajectories inspectable and comparable with ground truth. Experiments show a persistent gap between prediction and mechanism recovery: in the purely observational 6-node setting, GPT-5.2-high reaches 92% task accuracy but only 0.471 all-edge $F_1$. This observation further motivates our exploration of different interaction strategies: Mixed observation--intervention strategies improve structural fidelity: in the mixed 6-node setting, GPT-5.2-high achieves 80% on both task accuracy and all-edge $F_1$. Yet even strong agents struggle to design informative interventions, as pure intervention strategies perform poorly on both task accuracy and all-edge $F_1$. We identify premature stopping as a major weakness of agents, and show that asking the model to verify the consistency between its hypothesis and past data can help mitigate this issue. CausaLab therefore separates predictive success from causal understanding and exposes current LLM agents' limits as experimental causal reasoners. 

---
# Retrieval-Augmented Detection of Potentially Abusive Clauses in Chilean Terms of Service 

**Authors**: Christoffer Loeffler, Tomás Rey Pizarro, Daniel Ignacio Miranda Vásquez, Andrea Martínez Freile  

**Link**: [PDF](https://arxiv.org/pdf/2605.26019)  

**Abstract**: Online Terms of Service often function as contracts of adhesion, creating asymmetries that may expose consumers to potentially abusive clauses. In Chile, assessing such clauses is legally challenging because some provisions clearly violate mandatory consumer law, whereas others depend on broader standards such as good faith and contractual imbalance. We present a retrieval-augmented generation framework for the automated detection and classification of potentially abusive clauses in Chilean Terms of Service. Designed for local execution, it combines efficient clause detection, hybrid dense--sparse retrieval, reranking, and prompt augmentation to support medium-sized open-weight language models. We also introduce the Chilean Abusive Terms of Service Extended corpus, comprising 100 contracts and 10,029 annotated clauses in 24 legally grounded categories spanning illegal, dark, and gray clauses. Experiments comparing commercial and open-weight language models, fine-tuned encoders, and traditional baselines show that retrieval-augmented prompting substantially improves performance and enables local models to approach larger cloud-based systems at lower computational and token cost. The study also contributes a refined legal annotation scheme and a practical design for AI-assisted consumer contract review. 

---
# STORM: Internalized Modeling for Spatial-Temporal Reasoning in Video-Language Models 

**Authors**: Yiming Liang, Yixiao Chen, Yiyang Zhou, Yixuan Wang, Shoubin Yu, Andong Deng, Fuxiao Liu, Qin Zhang, Chen Chen, Mohit Bansal, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.26014)  

**Abstract**: Many video reasoning tasks require tracking motion, temporal order, and evolving visual states across frames. Existing methods built on large vision-language models (LVLMs) often address this challenge by externalizing reasoning through textual chain-of-thought (CoT), keyframe selection, repeated frame reinsertion, or external tool use. While effective, such pipelines increase inference-time latency and engineering complexity, and they force temporal-visual evidence to be serialized into text or repeatedly re-encoded from frames. Inspired by the intuition that visual reasoning can occur implicitly before verbalization, we propose STORMS (Spatial-Temporal reasOning via inteRnalized Modeling), a two-stage framework that teaches LVLMs to reason through bounded continuous latent trajectories instead of explicit textual CoT. In Stage I, STORMS aligns latent tokens with thought-video representations derived from generated videos, grounding the latent states in dynamic visual evidence. In Stage II, the model is further trained with answer-only supervision, encouraging the reasoning process to be internalized without step-by-step annotations. Generated thought videos are used only during training; at inference, STORMS performs a bounded latent rollout without regenerating videos, reinserting frames, or invoking external visual tools. Experiments on VideoMME, MVBench, TempCompass, and MMVU show that STORMS improves video reasoning accuracy while substantially reducing inference overhead compared with tool or video-generation-based reasoning pipelines. 

---
# MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models 

**Authors**: Shristi Das Biswas, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2605.26004)  

**Abstract**: Instruction tuning of large vision-language models (LVLMs) increasingly depends on massive multimodal corpora, yet these datasets contain samples with substantial redundancy, low visual dependency, and highly imbalanced coverage of multimodal reasoning behaviors. As a result, uniform subsampling or naive score-based selection often yields suboptimal training subsets. We introduce MAGIC, a training-free, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM: Multimodal Gain, which measures the likelihood improvement obtained from visual input; Bridging Relevance, which captures the sharpness of answer-token grounding over visual tokens; and Skill-Neuron Signatures, which characterize the functional computation elicited by each sample via top-activated feed-forward neurons. MAGIC combines these signals in a three-stage pipeline: filtering low-gain examples, ranking candidates by a normalized quality objective, and performing bucket-wise budget allocation over discrete neuron signatures to preserve latent multimodal skill coverage. This formulation avoids backpropagation, auxiliary selector training, and expensive clustering in continuous activation spaces, while remaining efficient and easily deployable in existing VLMs. Across LLaVA-665K and Vision-Flan datasets, and transfer settings to large target models, LLaVA-1.5-7B and -13B, MAGIC consistently improves over strong baselines under matched 20% budgets: it achieves 100.3% relative performance to full finetuning on LLaVA-665K and 101.6% relative performance on Vision-Flan-186K, while yielding a 73.7% reduction in wall-clock run time. 

---
# Mapping the Schedule x Bit-Width Boundary in Sub-100M Quantisation-Aware Training 

**Authors**: Christian Brandt Thomassen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25966)  

**Abstract**: We test whether the optimal learning-rate schedule depends on bit-width during from-initialisation quantisation-aware training (QAT) for sub-100M decoder language models. A 720-run factorial grid (Phase 2) over bit-width x warmdown fraction x LR magnitude x model size x seed (FP16/INT8/INT6, 15M-100M, 5 seeds) finds the optimal warmdown is 33% at every (bit-width, size) cell. The primary hypothesis -- that INT6 QAT requires a different schedule than higher-precision training -- is falsified at FP16/INT8/INT6. A 625-run follow-up (Phase 5) probes the null along five axes: optimiser (AdamW), schedule shape (cosine), training length (up to 9x more iterations), an extended size sweep (5M-350M), and an INT4 sweep from 3M to 100M. The null is robust under all three setup changes. The INT6 penalty follows a log-linear scaling law whose fit on Phase 2 predicts the five held-out Phase 5 sizes (5M, 8M, 175M, 250M, 350M) within their 95% prediction intervals (5/5). For INT4 the picture is sharper than the higher precisions: at 50M and 100M, wd33 is decisively optimal (paired z ~ 12-15, 10/10 seeds); below 50M, across the six tested sizes from 3M to 30M, no individual size shows a statistically significant schedule preference and the per-size mean penalty oscillates within seed-level noise. The boundary is therefore a transition between a noise-dominated regime below 50M and a decisive wd33 regime at and above 50M, not a clean wd10 region. A weight-to-grid-distance probe falsifies the simplest mechanism for the FP16/INT8/INT6 null result (rapid grid-snapping): pre-warmdown, INT6-QAT weights sit at essentially the same distance from the INT6 grid as FP16 weights (ratio ~ 1.04). Practical recommendation: at sub-100M scale, tune the LR schedule once at FP16 and apply unchanged to INT8/INT6 QAT; for INT4 at 50M+ use wd33; for INT4 below 50M the schedule choice is in the noise. 

---
# When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifiable Rewards 

**Authors**: Li Wang, Xiaodong Lu, Xiaohan Wang, Yikun Ban, Jiajun Chai, Wei Lin, Tianhao Peng, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.25864)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in reasoning capabilities empowered by Reinforcement Learning with Verifiable Rewards (RLVR). Nonetheless, RLVR intrinsically relies on ground-truth labels for reward computation, the acquisition of which is often prohibitively expensive in real-world scenarios. While unsupervised RLVR paradigms attempt to circumvent this by training on pseudo-labels, they are notoriously susceptible to training collapse. Moreover, different samples often exhibit varying annotation values. In this paper, we propose Reinforcement Learning with Active Verifiable Rewards (RLAVR), which actively acquires ground-truth labels for a small set of selected samples and integrates them with pseudo-labels, thereby stabilizing training dynamics and improving performance under limited annotation budgets. To identify valuable samples, we propose the Corrective Advantage Gap (CAG) metric and analyze the sample-level supervision value. Building on this, we introduce Correction-Aware Reliability Estimation for RLAVR (CARE), which translates the oracle CAG criterion into a practical pre-query acquisition policy to substantially improve training stability. Extensive experiments across diverse domains, model families, and model scales demonstrate the effectiveness and generality of our approach. Our code is available at this https URL. 

---
# MuCRASP: Multimodal Chain-of-thought Reasoning aware Structured Pruning 

**Authors**: Aritra Dutta, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2605.25842)  

**Abstract**: Vision-language models (VLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex multimodal tasks, but their large parameter sizes make deployment expensive. Structured pruning offers a natural solution; however, existing methods fail to preserve CoT reasoning accuracy in VLMs. We identify two key reasons: (1) CoT consistency depends on sparse transition points (pivot tokens) in the generation trajectory, while existing pruning methods are CoT-agnostic; and (2) pruning methods designed for unimodal LLMs do not account for activation-distribution differences across visual and textual modalities. Motivated by these observations, we propose MuCRASP, a structured pruning framework that targets reasoning-critical components while preserving cross-modal alignment and accounting for layer-wise sensitivity under a global parameter budget. Experiments on four VLMs across three reasoning benchmarks show that MuCRASP consistently preserves reasoning quality under increasing compression. At 30% pruning on Qwen2.5-VL-7B, MuCRASP achieves an LLM-as-a-Judge score of 8.87 versus 7.32 for the strongest baseline on physical reasoning tasks. Furthermore, MuCRASP maintains high reasoning consistency up to 50% pruning, significantly outperforming prior pruning approaches while exhibiting lower perplexity degradation. 

---
# TTPrint: Evidence-Grounded TTP Extraction via Diverge-then-Converge Verification 

**Authors**: Yutong Cheng, Changze Li, Raihan Sultan Pasha Basuki, Qian Cui, Wei Ding, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25836)  

**Abstract**: Extracting MITRE ATT&CK techniques from cyber threat intelligence (CTI) reports is an open-set, multi-label problem requiring both high recall (not missing techniques) and high precision (not hallucinating unsupported ones). Existing methods--rule-based, supervised, and LLM-based--struggle to achieve both: rule-based and supervised approaches lack generalizability across diverse attack descriptions, while LLM-based approaches that couple candidate generation and validation within a single inference step suffer from limited recall and precision simultaneously. We propose TTPrint, which addresses this challenge through a diverge-then-converge design inspired by how human analysts work: first extracting broadly, then verifying rigorously. In the divergent phase, reports are decomposed into atomic behaviors and candidate techniques are proposed broadly. A deterministic span localization stage then anchors each candidate to a specific evidence window in the source text. A convergent verification stage retains only candidates supported by both the localized evidence and the authoritative MITRE definition. We contribute two evaluation resources--a cleaned TRAM benchmark (TRAM-Clean) and a new annotated dataset (TTPrint-Bench)--to address known annotation noise in existing benchmarks and elevate the task to document-level TTP extraction. On TRAM-Clean and TTPrint-Bench, TTPrint achieves 76.48% and 87.39% macro-F1 respectively, outperforming the leading baseline by 63.5% and 29.4%. A multi-backbone analysis across six LLMs and a threshold sensitivity study further demonstrate generalizability across model choices and provide practical guidance for parameter selection. 

---
# When Search Becomes Memory: Turning Robot Design Trials into Transferable Skills 

**Authors**: Yunfei Wang, Xiaohao Xu, Yang Li, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25832)  

**Abstract**: Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and transfers learned skills to 10x10 design spaces, where reference-conditioned transfer outperforms GA on every task. These results suggest that LLM agents can convert expensive physical evaluations into reusable, auditable design principles. Our code will be released upon acceptance. 

---
# SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness 

**Authors**: Jiahao Huo, Wenjie Qu, Yibo Yan, Kening Zheng, Jiaheng Zhang, Xuming Hu, Philip S. Yu, Mingxun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25796)  

**Abstract**: Semantic-level watermarking (SWM) improves robustness against text modifications by treating sentences as the basic unit. However, robustness to paragraph-level paraphrasing remains difficult because such attacks globally disrupt watermark signals by changing sentence order. In this work, we propose SAMark, a self-anchored watermarking framework that removes the dependency on sentence order by establishing a step-independent green region in semantic space. To improve detectability, we introduce a multi-channel hyperbolic scoring mechanism that amplifies watermark signals while suppressing noise from weakly aligned candidates. We further propose a diversity-aware filtering strategy that combines hard filtering with soft regularization, extending beyond simple n-gram repetition filters to address semantic redundancy. Experimental results show that SAMark achieves up to 90.2% TP@FP1% under typical paragraph-level paraphrasing attacks, outperforming the strongest prior baseline by more than 30% on average, while maintaining generation quality competitive with unwatermarked text and breaking the robustness-quality trade-off that limits prior methods. 

---
# Efficient Benchmarking Is Just Feature Selection and Multiple Regression 

**Authors**: Sam Bowyer, Acyr Locatelli, Kris Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25773)  

**Abstract**: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous metrics. Furthermore, mRMR subsampling is much faster than competitor methods (which often involve fitting probabilistic models or running clustering algorithms), and is more likely to select the same questions under different random seeds or training data splits. Tutorial code can be found at this https URL . 

---
# CMAP: Cross-Modal Adaptive Prompting for Multi-Domain Task-Incremental Learning 

**Authors**: Sriram Mandalika  

**Link**: [PDF](https://arxiv.org/pdf/2605.25708)  

**Abstract**: Multi-domain task-incremental learning requires a model to sequentially acquire knowledge across visually diverse domains without forgetting prior tasks, and without access to task identity at inference. Parameter-efficient methods built on frozen vision-language models have made strong progress, yet all existing approaches rely exclusively on visual features for task routing, confidence estimation, and encoder adaptation, leaving CLIP's cross-modal text embedding space entirely unexploited. We address this gap through three contributions. Text-space task routing replaces visual Gaussian matching with cosine similarity to frozen CLIP text prototypes, giving order-independent routing robust to data scarcity at zero parameter cost. Multi-prototype visual-textual confidence replaces single-Gaussian class modeling with K-means visual prototypes and cross-modal alignment scores under task-calibrated thresholds. Symmetric cross-modal gating extends per-layer Gumbel gates to the text encoder conditioned on batch image features, preserving cross-modal alignment on out-of-distribution inputs. On the MTIL benchmark spanning 11 datasets and 1201 classes, our method achieves 74.2% Transfer, 80.5% Average, and 88.7% Last under Order-I, surpassing the prior state of the art by 5.0, 3.7, and 3.0 percentage points with only 2.5M trainable parameters and no external data. 

---
# Neural Router: Semantic Content Matching for Agentic AI 

**Authors**: Lauri Lovén, Abhishek Kumar, Alexander Engelhardt, Alaa Saleh, Roberto Morabito, Xiaoli Liu, Naser Hossein Motlagh, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2605.25701)  

**Abstract**: Large language models (LLMs) can serve as the semantic-matching engine of a content-based publish/subscribe broker for agentic AI across the edge-cloud computing continuum, bridging the vocabulary and modality gaps that defeat keyword and embedding filters. Framed as offline multi-label retrieval over three public datasets spanning social-media, legal, and smart-home sensor domains (six LLMs, seven baselines), our central contribution is a two-crossover cost-accuracy characterisation: an analytical context-window crossover below which a CoverAndMerge compression pipeline reduces LLM invocations, and an empirical discrimination-capacity crossover above which matching accuracy collapses independently of context budget, by a model-dependent factor of parameter count and training generation. Two findings carry practical weight: above the discrimination crossover, compression cannot recover accuracy and only frontier-scale models clear large subscription sets; and there backend choice dominates configuration choice, so model selection, not pipeline tuning, is the primary operator lever. We accompany this with three composable algorithms and a per-cluster Quality-of-Experience framework for autonomic LLM-tier selection. 

---
# RotMoLE: Enhancing Mixture of Low-Rank Experts through Rotational Gating Mechanism 

**Authors**: Mengyang Sun, Maochuan Dou, Tao Feng, Dan Zhang, Yihao Wang, Junpeng Liu, Yifan Zhu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25565)  

**Abstract**: While Large Language Models (LLMs) are commonly fine-tuned to handle domain-specific tasks before being applied to vertical applications, adapting them to complex scenarios with diverse specialized knowledge remains challenging. Meanwhile, Mixture-of-Experts (MoE) architecture has risen as a crucial paradigm for training LLMs, and some recent works have also incorporated MoE into Parameter-Efficient Fine-Tuning (PEFT) to propose the Mixture of Low-rank Experts (MoE-LoRA), to enhance the power of low-rank adapters for learning complicated knowledge. However, conventional gating mechanisms in MoE typically apply only a scalar reweighing to selected experts, thereby limiting their underlying capacity of representation and generalization. Motivated and enabled by the low-rank structures in MoE-LoRA, we propose RotMoLE, a specialized MoE framework for low-rank experts featuring an additional rotation gate. Beyond simple scaling, RotMoLE implements a rotation mechanism for each selected expert, enabling superior expert exploitation and specialization for learning diverse data, especially when expert candidates are limited. Empirical results on complex multi-task and multilingual training scenarios validate our effectiveness. 

---
# AI Content Moderation in Therapy Conversations 

**Authors**: Jiwon Kim, Claire Wang, Taeung Yoon, Sabelle Huang, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25454)  

**Abstract**: Large language models (LLMs) are increasingly being used for emotional support. They are also being developed for formal therapy purposes. However, LLMs like ChaptGPT or Llama are often developed with content moderation guardrails that prevent them from discussing sensitive subjects with users for both liability and safety purposes, and this inability to broach these subjects may affect their capacity as therapists. In this study, we perform an algorithm audit on three state-of-the-art moderation systems (OpenAI's moderation endpoint, Meta's Llama Guard, and Google's Shield Gemma) to investigate the extent to which these systems flag the content of real-life therapy sessions as undesirable. Our results raise implications for the limitations that users and organizations may encounter when designing LLMs to play the part of a therapist. 

---
# Second Guess: Detecting Uncertainty Through Abstention and Answer Stability in Small Language Models 

**Authors**: Ashwath Vaithinathan Aravindan, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.25394)  

**Abstract**: Large language models often generate confident but incorrect answers rather than abstaining when uncertain. This problem is particularly acute for small language models (SLMs), where computational constraints and autonomous operation amplify the need for reliable uncertainty detection. We propose _Second Guess_, a lightweight, parameter-free prompting technique for abstention in multiple-choice question answering (MCQA) that is well-suited for SLMs. Our key empirical insight is that models which truly know an answer will select it consistently, while uncertain models exhibit unstable behavior when an ``I don't know'' option is added. Evaluated on four open models (2B-8B parameters) and four benchmarks, Second Guess achieves the highest composite risk improvement of 10.81\%. Notably, it maintains an 8\% composite risk improvement on fine-tuned models where entropy-based methods degrade, and improves most for lower-performing models. All code and results required to reproduce this work is available in this https URL 

---
# Directional Alignment Mitigates Reward Hacking in Reinforcement Learning for Language Models 

**Authors**: Wenlong Deng, Jiaji Huang, Kaan Ozkara, Yushu Li, Christos Thrampoulidis, Xiaoxiao Li, Youngsuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.25189)  

**Abstract**: Reward hacking arises when a model improves a proxy reward by exploiting shortcuts rather than solving the intended task. We study this failure mode through the geometry of reinforcement learning updates in language models and argue that hacking emerges when optimization drifts away from a stable low-dimensional learning trajectory. We analyze this drift through dominant singular directions of parameter updates and show that reward-hacking runs exhibit substantially larger directional change than clean runs. Motivated by this observation, we introduce trusted-direction projection, which constrains gradients to remain within a clean reference subspace. Across reward-hacking experiments on mathematical reasoning, the proposed approach delays shortcut exploitation and better preserves task performance. 

---
# Trust but Verify: Prover-Verifier Deliberation for Selective LLM Prediction 

**Authors**: João Sedoc, Baotong Zhang, Dean Foster  

**Link**: [PDF](https://arxiv.org/pdf/2605.25133)  

**Abstract**: Reliably knowing when a language model is correct is almost as important as being correct. We introduce prover-verifier deliberation (PVD), an inference-time protocol grounded in interactive proof theory, as a mechanism for selective prediction: the protocol produces both an answer and a structured confidence verdict, allowing a system to report high-confidence answers while abstaining on uncertain cases. In each dialogue, a prover defends a candidate answer through checkable sub-claims while a verifier issues targeted challenges and returns \textsc{Accept}, \textsc{Challenge}, or \textsc{Reject}. Because frozen language models are imperfect provers and verifiers operating over a noisy channel, formal soundness and completeness guarantees do not transfer; instead, we characterize the protocol empirically through its coverage-precision behavior. Our main experiment uses Claude Sonnet 4.6 as prover and Claude Haiku 4.5 as verifier on GPQA Diamond. Questions accepted with no answer revision, which we call Accept + No Change (ANC), are reported as the high-confidence subset; we evaluate this subset by its precision and coverage. ANC separates reliable from unreliable answers, yielding a $\sim$30pp HC-Prec gap over the non-ANC complement. Robustness experiments with GPT and Gemini pairings show that high HC-Prec can transfer across model families, while verifier strictness and domain competence largely determine the size of the selection gap. On Humanity's Last Exam, weaker prover-verifier pairings can collapse or invert the ANC signal, illustrating a practical failure mode when the verifier operates outside its effective region. Comparisons with self-consistency, universal self-consistency, multi-agent debate, and Reflexion suggest that prover-verifier deliberation supplies a distinct argument-defensibility signal for selective prediction. 

---
# Inference-Time Alignment of Diffusion Models via Trust-Region Iterative Twisted Sequential Monte Carlo 

**Authors**: Weixin Wang, Yu Yang, Wei Deng, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25123)  

**Abstract**: We study inference-time alignment for diffusion-based generative models, aiming to steer a base model toward high-reward outputs without updating its weights. Recent Sequential Monte Carlo (SMC)-based steering methods approximate reward-tilted target distributions in a principled way, but their proposals remain largely tied to the base sampler. Since reward information is mainly used after propagation through particle reweighting and resampling, these methods can require large particle budgets and suffer from weight degeneracy and high-variance estimates. One way to reduce variance and improve particle efficiency is to iteratively learn twisting functions that provide look-ahead guidance, as in twisted SMC. However, existing learnable twisting methods are developed mainly for classical sequential inference and can be unstable when applied to diffusion-based alignment with high-dimensional state spaces and terminal, noisy, or black-box rewards. We propose Trust-Region Iterative Twisted Sequential Monte Carlo (TRI-TSMC), a trust-region framework for learning twisting functions in SMC-based inference-time alignment. Each iteration computes an exact KL-constrained update in path space, which admits a closed-form solution by tempered importance reweighting, and projects this target back to the parameterized twisted family by weighted maximum likelihood. Theoretically, we formalize the value-function interpretation of the optimal twisting function and show that it yields a zero-variance sampler. We prove that the trust-region update follows an escort path toward the target distribution, that the weighted maximum-likelihood update is a forward-KL projection, and that the path reduces residual importance-weight variance. Empirically, TRI-TSMC improves primary alignment objectives on discrete diffusion text generation and text-to-image generation under matched inference-time budgets. 

---
# AgentIR: A Workload-Adaptive Cascade Retrieval Substrate for Long-Term Conversational Memory 

**Authors**: Aojie Yuan, Haiyue Zhang, Shahin Nazarian  

**Link**: [PDF](https://arxiv.org/pdf/2605.25092)  

**Abstract**: Long-term conversational memory is a retrieval workload classical IR was not built for: the index grows during the query stream, query types shift intra-session, and the latency budget per retrieval is sub-10 ms. Lucene-class engines treat the index as static and the query as stateless, leaving the workload's structure unexploited.
AgentIR treats fusion as a per-query decision along two axes: which fusion to apply (BM25, Dense, RRF, or agent-aware RRF), and whether the ~52 ms dense channel is worth running at all. The second axis is a confidence-triggered cascade router that decides from the BM25 top-k margin alone and re-tunes across workloads without retraining. On LongMemEval (n=500), where the dense channel does add information, the cascade skips 63% of queries at parity LLM-judged accuracy (2.67x faster under two judges, paired bootstrap p>=0.88); per-qtype thresholds extend this to 5.76x under 5-fold cross-validation. On LoCoMo (n=1,982), where BM25 alone is already the strongest single system, the same trigger auto-tunes to a 100% skip rate (132x faster, +0.089 Hit@5). Capacity on a shared 8-core VM rises from ~154 to ~1,400 concurrent agents (9x).
Underneath the cascade, a time-partitioned index does O(log 1/epsilon) work independent of corpus size: 1234x corpus growth costs only 3.6x latency, ending in 1769x over sequential at sub-100 us p50 on 5M records. At parity quality with Lucene on 9 BEIR datasets up to 8.8M docs, the substrate runs 10x geo-mean over Pyserini 8T and 11x over PISA-1T BlockMax-WAND; an A100 reaches 1.8-39x over Pyserini 8T; chunked index build sustains 56.8K docs/sec on MS MARCO. Three subtle BM25/GPU correctness pitfalls that silently regress nDCG@10 by 6-8x are documented and fixed; post-fix CPU and GPU agree within 0.0002 nDCG@10 on all eight datasets that fit a single A100. 

---
# Privacy-Preserving Local Language Models for Longitudinal Data Retrieval in Chronic Dermatologic Disease: Implementation in Pemphigus Patients 

**Authors**: Abdurrahim Yilmaz, Ayşe Esra Koku Aksu, Duygu Yamen, Vefa Asli Erdemir, Mehmet Salih Gurel, Gulsum Gencoglan, Joram M. Posma, Burak Temelkuran  

**Link**: [PDF](https://arxiv.org/pdf/2605.25020)  

**Abstract**: Chronic dermatologic diseases such as pemphigus require long-term follow-up, generating extensive longitudinal clinical documentation that is difficult to review comprehensively during routine visits and increasing clinician workload as well as the risk of missing critical historical information. We evaluated whether a locally deployed, privacy-preserving small language model (SLM) could retrieve structured clinical features and generate longitudinal summaries from long-term dermatology follow-up records. In this retrospective case series, thirty pemphigus patients contributed 541 visit notes that were aggregated into full longitudinal records (89,336 words); 56 clinically relevant features were annotated by two expert dermatologists. The locally deployed SLM (Qwen3 4B Thinking 2507) was queried with each complete record to retrieve 56 features and generate one final report summaries. Across 1,680 feature retrieval tasks, mean accuracy was 82.25%. Dermatologists' ratings of AI-generated summaries were high for overall quality (8.23-8.47), clinical accuracy (7.93-8.20), and usefulness (8.47-8.50), with no significant inter-evaluator differences and an overall preference for AI summaries in 53.3% of evaluations. These findings suggest that privacy-preserving, locally deployed SLMs can outperform medical experts and reliably generate clinically meaningful longitudinal summaries. SLMs may support clinical decision-making when integrated with appropriate oversight. 

---
# Universal Boosts, Specific Suppressors: Sparse Autoencoder Steering of Medical Vision-Language Models 

**Authors**: Farhad Nooralahzadeh, Benjamin Gundersen, Nicolas Deperrois, Hidetoshi Matsuom, Mizuho Nishio, Thomas Frauenfelder, Ahmed Allam, Christian Blüthgen, Michael Moor, Michael Krauthammer  

**Link**: [PDF](https://arxiv.org/pdf/2605.24977)  

**Abstract**: Medical vision-language models (VLMs) often hallucinate findings when generating chest X-ray reports: they fabricate findings that are not present in the image, miss important ones, or locate them incorrectly. We mitigate this without weight updates by decoding-time residual steering on a per-token sparse autoencoder (SAE) basis: Top-$K$ SAEs on late layers, causal steering against clinical errors, then combined suppress/boost intervention at inference time. On the MIMIC-CXR test split, our inference-only method improves the quality of generated reports for three radiology VLMs (RadVLM, LLaVA-Rad, and CheXOne), with relative improvements of +5.4%, +7.2%, and +17.0% in the clinical composite metric, and statistically significant GREEN gains on all backbones. A cross-model feature alignment shows that the quality-promoting (boost) directions overlap strongly across architectures, whereas hallucination-linked (suppress) directions are model-specific. Therefore, transferable steering must treat suppression per-backbone, rather than sharing a universal suppress list. The same recipe transfers zero-shot to IU-Xray (Green $+7.7\%$ rel.) without retraining, confirming that the identified features are properties of the model, not of the training corpus. We release causal feature sets and an interactive feature dashboard: this https URL. 

---
# MinerU-Popo: Universal Post-Processing Model for Structured Document Parsing 

**Authors**: Bangrui Xu, Ziyang Miao, Xuanhe Zhou, Yiming Lin, Zirui Tang, Xiaomeng Zhao, Fan Wu, Cheng Tan, Fan Wu, Bin Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2605.24973)  

**Abstract**: VLM-based OCR models have become the de facto choice for document parsing, as they can accurately extract page-level elements (e.g., paragraphs within individual pages) together with their bounding boxes and textual content. However, downstream applications such as RAG require coherent document-level information, whereas these models often break cross-page continuity and fail to recover disrupted structures, such as paragraphs and tables truncated by page boundaries. Such relationships are not confined to a single page; instead, they require joint analysis of titles, paragraphs, tables, and images spanning multiple pages. A natural solution is therefore to reuse existing OCR outputs and reconstruct document-level logical structures through post-processing.
To this end, we propose MinerU-Popo, a lightweight and universal framework for POst-Processing OCR outputs, which converts page-level results from diverse parsers into coherent document-level structures. MinerU-Popo decomposes the problem into four focused subtasks: text truncation recovery, table truncation recovery, title hierarchy reconstruction, and image-text association. To address these effectively, we build a task-oriented data engine with task-specific input filtering, and use the generated data (30K) to fine-tune a lightweight post-processing model (Qwen3-VL-4B). To support long documents, we introduce dynamic chunking with overlap-based synchronization, which aligns chunk-level outputs from the fine-tuned model and preserves global consistency. Finally, we assemble the aligned outputs into a tree-structured document representation, further enriched with node chunking and summaries for downstream retrieval and analysis. Empirical results show MinerU-Popo improves title-hierarchy TEDS by at least 20% across all five tested OCR models, improves RAG accuracy and reduces per-query latency. 

---
# Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning 

**Authors**: Xuanting Xie, Zhaochen Guo, Bingheng Li, Xingtong Yu, Zhifei Liao, Zhao Kang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24867)  

**Abstract**: Chain-of-Thought (CoT) prompting has shown promise in enhancing the reasoning capabilities of large language models (LLMs) on text-attributed graphs (TAGs). This work reframes CoT-based graph learning through the principle of clustering as reasoning, offering a $k$-means interpretation of how iterative reasoning operates over graph-structured data. We observe that existing graph CoT methods rely on disjoint architectures and fixed graph representations, limiting step-by-step semantic-topological interaction and interpretability. To overcome this limitation, we propose a unified framework named KCoT that integrates CoT reasoning with graph representation learning. Our key theoretical result reveals a formal mathematical correspondence between a Transformer block and the $k$-means algorithm, allowing reasoning to be interpreted as iterative assignment and update steps. Based on this insight, we introduce a Semantic Discriminating Prompt that explicitly formulates these steps as structured CoT reasoning, together with a structure-grounded alignment strategy to fuse topological priors with evolving thought-conditioned representations. Experiments on standard benchmarks demonstrate consistent improvements over state-of-the-art methods, validating clustering as a principled mechanism for CoT-based graph learning. 

---
# Geo-Expert: Towards Expert-Level Geological Reasoning via Parameter-Efficient Fine-Tuning 

**Authors**: Chenyou Guo, Zongqi Liu, Yizhou Zhang, Zhaorui Jiang, Ze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24844)  

**Abstract**: While general-purpose Large Language Models (LLMs) applied to Geology often hallucinate when reasoning about subsurface structures and deep-time evolution, current AI in Earth sciences predominantly targets surface remote sensing and GIS. To bridge this gap, we introduce Geo-Expert, a family of parameter-efficient geological LLMs fine-tuned on a custom-curated, high-quality instruction dataset processed using our custom instruction synthesis pipeline. We investigate the impact of model scaling and architecture by fine-tuning three base models: Qwen3-8B, Qwen3-32B, and Gemma-3-27B, with Low-Rank Adaptation (LoRA) method. Our extensive evaluation on a novel domain-specific benchmark, Geo-Eval, reveals that a domain-aligned 8B model can outperform open-weight 70B generalists and proprietary GPT-4o on specialized geological reasoning, while a 32B variant approaches frontier reasoning models. The optimized 8B model further offers a competitive cost-performance ratio for deployment. This work provides a reproducible recipe for democratizing scientific LLMs and establishes a baseline for geological artificial intelligence. 

---
# Spiking the training data to correct for test set contamination 

**Authors**: Johnny Tian-Zheng Wei, Jerry Li, Ameya Godbole, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.24818)  

**Abstract**: The literature on test set contamination largely focuses on detection, but the correction of contaminated test scores is underexplored. Our core proposal is to spike the training data by intentionally contaminating some test examples at known rates. The spiked examples can then be used to calibrate predictors of model memorization which enable principled statistical correction of inflated test scores. To evaluate different correction estimators, we first present a simulation framework based on the Hubble models. Hubble models come in minimal pairs, where the perturbed model was deliberately contaminated with several test sets, while the standard model was not, serving as the counterfactual and correction target. We consider estimators that use information from a memorization predictor, correctness predictor, or both. In simulation, we establish basic statistical intuitions and show that estimators leveraging memorization and correctness information are better than naive estimation which makes no correction at all. We then instantiate several memorization and correctness predictors, and find that simple predictors such as Platt-scaled membership inference metrics provide good signal for correction. Finally, we examine the practical considerations of spiking. Simple memorization predictors need no more than 10 examples for calibration and often transfer from one dataset to another. Taken together, spiking is a promising solution for test set contamination. 

---
# RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry 

**Authors**: Bo Lv, Zhiheng Xu, KeDong Xiu, Ruyi Ding, Tianhang Zheng, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24817)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become an increasingly important paradigm for scaling Large Language Models (LLMs). As MoE models are increasingly deployed in real-world services, safety auditing becomes necessary to verify whether these models produce or facilitate harmful behaviors during operation. However, existing content-based auditing methods typically require access to user prompts, model inputs, or generated outputs, potentially exposing sensitive user information and creating a fundamental tension between LLM safety and user privacy. On the other hand, we observe that, in MoE models, sparse expert routing maps different inputs to activate different expert-execution patterns, producing measurable footprints in low-level GPU execution telemetry. Inspired by this observation, we propose RouteScan, a non-intrusive auditing framework for detecting harmful behaviors through GPU-level expert routing telemetry. Specifically, RouteScan utilizes the number of active GPU threads allocated to expert modules during the prefilling phase as a discriminative micro-architectural fingerprint, and builds a lightweight detection pipeline that isolates cross-domain invariant risk indicators for the precise identification of malicious prompts. Comprehensive evaluations on open-source MoE LLMs with distinct routing designs demonstrate that RouteScan achieves strong generalization, with an AUROC exceeding 0.93 on unseen harmful domains and 0.96 under novel jailbreak wrappers. Moreover, empirical inversion tests show that the collected expert routing telemetry provides limited information for prompt reconstruction, suggesting a practical privacy advantage over content-based auditing methods. 

---
# DUEL: Adversarial Self-Play for Multimodal Reasoning 

**Authors**: Lin Qiu, Hanqing Zeng, Yao Liu, Bingjun Sun, Guangdeng Liao, Ji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24794)  

**Abstract**: Reinforcement learning (RL) has emerged as an effective paradigm for improving the reasoning capability of vision-language models (VLMs). However, RL-based optimization typically depends on costly high-quality annotations that are difficult to scale. Existing unsupervised alternatives may drift toward biased solutions due to weak visual grounding and the lack of reliable verification signals. We propose a self-evolving post-training framework, DUEL, where supervision emerges from adversarial interactions between two policies initialized from the same pretrained VLM. A Challenger generates an image-grounded true claim together with a minimally perturbed hard-negative counterpart, while a Solver verifies both claims against the image, encouraging fine-grained visual discrimination under near-neighbor semantics. To stabilize optimization, we introduce a length-normalized log-likelihood reward that preserves informative optimization signals beyond binary outcome supervision and improves learning stability under sparse feedback. Experiments show that DUEL consistently improves visual reasoning and robust discrimination without additional human annotations, external reward models, or image editing tools. 

---
# Spectral Retrieval: Multi-Scale Sinc Convolution over Token Embeddings for Localized Retrieval in LLM Multi-Agent Systems 

**Authors**: Andrea Morandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24764)  

**Abstract**: [Abridged] - Spectral Retrieval is a plug-in re-ranking stage that interpolates between per-token MaxSim and mean-pool retrieval through a multi-scale sinc convolution over token embeddings. In standard dense retrieval each document is one mean-pooled vector; when relevance localises into a short subspan, the signal averages into noise. Spectral Retrieval reuses per-token embeddings from a late-interaction index and convolves them with a normalised sinc kernel at multiple scales. At L=1 the kernel acts as the identity, recovering per-token MaxSim; as L grows it approaches a uniform filter, recovering mean pooling. The maximum cosine over positions and scales yields a score provably no less informative than either endpoint. On a controlled synthetic benchmark with 1,000 documents and planted single-position spikes, mean-pool retrieval sits at chance (Recall@10 ~ 0.02) regardless of spike strength, while Spectral Retrieval reaches Recall@10 = 1.0 once the planted cosine exceeds the corpus-level token noise floor. On LIMIT-small with a frozen all-mpnet-base-v2 encoder, Spectral Retrieval lifts Recall@10 from 0.33 to 0.90, MRR from 0.22 to 0.79, and strict Success@10 from 0.12 to 0.84, without retraining. The method fits naturally into multi-agent LLM systems, where each agent benefits from a tighter, role-specific retrieval window over a shared corpus. 

---
# Automated Detection and Classification of Delusion-related Content in Naturalistic Audio Diaries Using Multi-Agent Language Models 

**Authors**: Feng Chen, Justin Tauscher, Changye Li, Meliha Yetisgen, Alex Cohen, Adam Kuczynski, Angelina Pei-Tzu Tsai, Benjamin Buck, Dror Ben-Zeev, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24755)  

**Abstract**: Speech monologues recorded in naturalistic settings provide opportunities to characterize mental illness phenomenology and detect symptom exacerbation. Large language models (LLMs) offer new possibilities for automating this process, as they require annotated data primarily for evaluation rather than training. In this paper, we present a novel automated, multi-agent LLM pipeline for the fine-grained, multi-label extraction of language suggestive of delusional beliefs, associated affective responses, and behavioral responses from transcripts of naturalistic audio diaries collected from people with moderate persecutory ideation. Evaluating an ensemble of three foundation models, we demonstrate that detailed diagnostic prompt instructions successfully reduce false positives for delusional theme classification, but also constrain the interpretation of affective or behavioral responses. Furthermore, comparing multi-agent adjudication frameworks shows that complex conversational debate between agents diminishes accuracy on clinically ambiguous text by inducing premature consensus. Instead, majority voting establishes robust performance (Micro F1 of 0.872 and 0.779 for delusion detection and classification respectively). This work provides a validated and scalable pipeline for the automated detection and characterization of content suggesting delusional beliefs in naturalistic speech. 

---
# Fundamental Limitation in Explaining AI 

**Authors**: Atsushi Suzuki, Jing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24727)  

**Abstract**: While large-scale models such as LLMs and diffusion models have achieved practical success, public institutions have emphasized the importance of explainability in AI. Existing methods for explaining AI, however, are not designed to provide completely faithful explanations of the behavior of large-scale AI systems. Although a completely faithful and interpretable explanation of the behavior of an AI system might be useful for AI governance, it has not been known whether providing such an explanation is theoretically possible. In this paper, we mathematically prove a fundamental quadrilemma in explaining AI, stating that AI and its explanation cannot satisfy the following four conditions simultaneously: 1) the complexity of the operation environment, 2) the goodness of the AI's performance, 3) the interpretability of the AI's explanation, and 4) the complete faithfulness of the AI's explanation. This quadrilemma suggests that, in most applications where we cannot change the environment or sacrifice good AI performance and an interpretable explanation, we should give up complete faithfulness of explanations and should instead aim to explain only the parts that are important for applications. As a consequence, the quadrilemma implies that AI governance should be designed on the premise that the faithfulness of AI explanations is always incomplete. 

---
# Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care 

**Authors**: Vassilis Lyberatos, Edmund G. Dervakos, Eleni Adamidi, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24678)  

**Abstract**: Speech and language technologies offer valuable opportunities for supporting mental health assessment through objective and interpretable cues. We present a systematic feature-based analysis framework leveraging perceptually grounded acoustic and linguistic characteristics, including prosody, vocal quality, semantic coherence, syntactic structure, and sarcasm. Using statistical analysis and interpretable machine learning (XGBoost with SHAP and LIME), we examine associations between speech features and validated symptom measures of depression, anxiety, and ADHD. Evaluated on both controlled benchmark datasets (StressID, DAIC-WOZ, Androids, EATD) and a real-world clinical dataset, the framework reveals stable and consistent relationships between symptom severity and vocal irregularities (e.g., shimmer, jitter), lexical-syntactic patterns, and affective tone. An ablation study conducted across all datasets further identifies the most informative feature groups. This work explores a transparent and clinically interpretable approach to speech-based mental health analysis. 

---
# Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24661)  

**Abstract**: LLMs have achieved remarkable success in complex reasoning tasks, yet current evaluation approaches predominantly rely on final-answer correctness, offering limited insight into the underlying reasoning processes that produce those answers. To address this gap, this study proposes a unified multi-dimensional framework for measuring reasoning quality in LLMs from a behavioral perspective, operationalizing six theoretically grounded dimensions: Correctness (CQ), Consistency (CS), Robustness (RS), Logical Coherence (LS), Efficiency (ES), and Stability (SS). Extensive experiments on seven LLMs across 975 items from four benchmarks demonstrate that the framework reveals behaviors invisible to accuracy-only metrics. Notably, logical coherence is orthogonal to correctness (r = -0.172, ns), confirming that correct answers can arise from incoherent reasoning, while Claude-Haiku-4.5 achieves the highest multi-dimensional score (Q_bal = 0.778). Furthermore, the framework exposes critical ranking inversions: DeepSeek-V3 ranks second under accuracy-priority but fifth under legal/compliance weighting, a reversal that single-metric evaluation cannot detect. Discriminant validity confirms 11/15 dimension pairs are independent (|r| < 0.50), providing psychometric support for treating each dimension as a distinct signal. The dimensional profiles produced by the framework directly support three classes of deployment decision: identifying models whose reasoning traces would fail accountability audits despite correct final answers (LS--CQ orthogonality); preventing ranking errors caused by accuracy-only benchmarking; and ensuring that no single metric silently substitutes for the six independent signals the framework captures. 

---
# GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration 

**Authors**: Junjie Zhao, Jingyi Liang, Zhenyang Cai, Jiaming Zhang, Zhenwei Wen, Shuzhi Deng, Wenjing Yi, Chunfeng Luo, Hexian Zhang, Junying Chen, Tianrui Liu, Zhuhui Bai, Zixu Zhang, Pradeep Singh, Xiang Liu, Jianquan Li, Nhan L Tran, Falk Schwendicke, Zuolin Jin, Lijian Jin, Liangyi Chen, Wei-fa Yang, Benyou Wang, Junwen Wang, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24636)  

**Abstract**: While large language models (LLMs) hold transformative potential for medicine, their reasoning robustness and safety in real-world clinical scenarios remain critically underexplored, particularly in dentistry. Here we introduce GlobalDentBench, the first multinational dental benchmark, featuring a taxonomy that encompasses 14 dental specialties across 88 countries and regions spanning six continents. The benchmark comprises 8,978 expert-validated questions across three formats (multiple-choice, short-answer, and case-based questions) and assesses three progressive reasoning levels: knowledge recall (L1), routine reasoning (L2), and individualized reasoning (L3). To ensure data quality, the automated construction framework was calibrated by six senior dentists, achieving expert agreement rates of 99.98% for multiple-choice and short-answer questions and 96.78% for the more complex case-based questions. Evaluation of 12 frontier LLMs on GlobalDentBench revealed a sharp, stepwise performance degradation with increasing reasoning complexity. Specifically, accuracy plummeted from 81.34% on multiple-choice to 64.53% on short-answer and 22.34% on case-based questions, while declining markedly from 74.01% at L1 to 55.64% at L2 and 35.71% at L3. More critically, risk analysis of real-world dental cases demonstrated an alarming overall unsafe rate of 31.01% in LLM-generated clinical recommendations, with 4.51% posing risks of irreversible patient harm and risks particularly pronounced in specialties such as orthodontics. These findings expose fundamental limitations in the medical reasoning and safety of current LLMs. Consequently, GlobalDentBench provides a scalable foundation for trustworthy clinical AI evaluation, underscoring the urgent need for rigorous validation before the safe deployment of these models in healthcare. 

---
# Learning to Reason Efficiently with A* Post-Training 

**Authors**: Andreas Opedal, Francesco Ignazio Re, Abulhair Saparov, Mrinmaya Sachan, Bernhard Schölkopf, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2605.24597)  

**Abstract**: Many applications of large language models (LLMs) require deductive reasoning, yet models frequently produce incorrect or redundant inference steps. We frame natural language inference as a search problem where the final answer is the valid proof itself, requiring a reasoning procedure in which intermediate inferences are correct. Specifically, we investigate whether LLMs can learn to generate correct and efficient proofs with guidance from A* search -- an algorithm that guarantees an optimally efficient path to a goal. We explore two training techniques: supervised fine-tuning on execution traces from A* and reinforcement learning with A*-informed process reward models. Empirically, we find that Llama-3.2 models in the 1B--3B range benefit substantially from A* post training, going from near-zero accuracy to outperforming DeepSeek-V3.2 -- a much larger model. Our analysis uncovers a trade-off: while simple correctness rewards maximize accuracy, A*-informed signals strike a balance between accuracy and efficiency. Furthermore, we find that on larger search spaces, models trained with imperfect heuristics exhibit superior accuracy. Our results demonstrate a promising direction towards reasoning guided by principles derived from classical search algorithms. 

---
# An Effective-Rank Audit of Alignment-Induced Activation Shifts: Confound Control, Constructive Calibration, and Limits 

**Authors**: Yuki Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.24583)  

**Abstract**: We audit alignment-induced shifts in residual-stream activations of three open-weight instruction-tuned LLMs (Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct) using the effective rank of the alignment modification matrix on safety-relevant inputs, rho_eps := rank_eps(M_Ds)/d, which formalizes the single-refusal-direction observation of Arditi et al. (2024) as a continuous quantity. The paper has three contributions. (1) Confound-controlled measurement: a four-variant decomposition (M_naive, M_template, M_aligned, M_DiD) separates chat-template formatting, alignment-stage shift, and the refusal-mediating direction, and recovers the Arditi refusal direction on M_DiD at |cos| in {0.77, 0.86, 0.50} (Llama/Gemma/Qwen); chat-template-controlled rho_eps is {0.0029, 0.0048, 0.0044}, and the centered SVD residual is 4-7x larger. (2) Constructive calibration on a 3-layer MLP across rho_eps in {0.008, 0.17, 0.33, 0.40} exhibits a sweet-spot vs. brittle distinction: mild rank-maximization (lambda=5) buys ablation robustness, while strong regularization at the same nominal rho_eps (lambda=50) does not. rho_eps is a diagnostic for fragility, not a target whose mechanical inflation buys robustness. (3) Limits of rank-based diagnostics: (a) not safety-specific (LRH baseline is 2-3x the safety value); (b) SVD principal ordering does not match causal ordering (Llama u_2 inert despite ranking second; cumulative ablation non-monotone at k=5); (c) the spectral-gap hypothesis required to upgrade the O(rho_eps * d) achievability bound to a matching Mirsky-route lower bound fails empirically (1/90 Llama layer-reference pairs, 0/36 MLP combinations) and structurally (kappa_lb <= 2/(eps * r)). The matching lower bound remains an open problem. 

---
# Polymorphism Is Rotation: Operational Mechanistic Interpretability from a Two-Layer Transformer to Pythia-70m 

**Authors**: Jordan F. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.24577)  

**Abstract**: Independently trained transformers compute the same function in residual-stream bases that differ by a uniform random rotation on $\mathrm{SO}(d_{\mathrm{model}})$. We call this phenomenon polymorphism: same function, mutually unintelligible interior coordinates. One matrix multiplication per model pair removes it: an orthogonal Procrustes fit on a single batch of activations transfers sparse-autoencoder feature dictionaries and steering vectors between independently trained models, with no retraining.
The phenomenon is invisible to the standard SAE universality metric. Decoder-column cosine similarity matches across seeds at 98%, the SAE-universality headline number, while an SAE trained on one seed reconstructs another seed's activations at negative explained variance, worse than predicting the constant mean. The decoder columns align; the encoder reads from a rotated frame. A single Procrustes rotation $R$ restores reconstruction to within 0.025 EV of the within-seed ceiling at every internal site.
$R$ is Haar-distributed: $\|R - I\|_F$ matches the random-orthogonal prediction $\sqrt{2 d_{\mathrm{model}}}$ to 0.1% at $d_{\mathrm{model}} = 512$, and a Kolmogorov-Smirnov test of $R$'s eigenvalue spectrum against Haar $\mathrm{SO}(d_{\mathrm{model}})$ returns $p \approx 1.000$ pooled and per-pair. Diff-of-means steering vectors transfer in three regimes by alignment with $R$'s invariant subspace: clean when pinned by shared output weights, partial when overlapping the rotated subspace, inverted otherwise. With no shared I/O (Pythia), all three collapse to universally inverted. The same rotation account holds across training checkpoints within a single run.
Validated on a 104k-parameter Dyck-3 transformer and nine independently-trained Pythia-70m seeds on The Pile, via a pre-registered four-bar operational framework. Frontier-scale (10B+) replication remains open. 

---
# The Multilingual Curse at the Retrieval Layer: Evidence from Amharic 

**Authors**: Yosef Worku Alemneh, Kidist Amde Mekonnen, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2605.24556)  

**Abstract**: Multilingual retrieval increasingly underpins cross-lingual question answering and retrieval-augmented generation. Strong zero-shot scores on multilingual benchmarks are often taken as evidence that current encoders transfer reliably across many languages. We argue that this assumption breaks down for underrepresented, morphologically rich languages, and use Amharic as a diagnostic case. Under a shared passage retrieval protocol covering dense, late-interaction, learned sparse, and cross-encoder paradigms, we compare zero-shot multilingual retrievers, Amharic-fine-tuned multilingual retrievers, and monolingual Amharic retrievers. The strongest zero-shot multilingual retriever underperforms the strongest monolingual Amharic first-stage retriever by 23% relative MRR@10. Fine-tuning two recent multilingual embedding models on the same Amharic supervision yields 32-60% relative MRR@10 gains over zero-shot, but the best Amharic-fine-tuned multilingual model remains below the strongest monolingual Amharic retriever. These findings indicate that zero-shot multilingual retrieval is not a sufficient proxy for equitable information access in the LLM era: for underrepresented languages, retrieval must be evaluated and adapted in-language rather than inferred from aggregate multilingual benchmarks. To foster future research, we publicly release the dataset, codebase, and trained models at this https URL. 

---
# Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models 

**Authors**: Seokil Ham, Jaehyuk Jang, Wonjun Lee, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.24550)  

**Abstract**: Fine-tuning-as-a-Service (FaaS) enables personalization of large language models (LLMs), but it can weaken safety-alignment under harmful fine-tuning attacks. Recent work has shown that activating harmful-behavior modules during fine-tuning can prevent models from learning undesired behaviors, but its mechanism remains unclear. In this paper, we revisit temporary jailbreaking as a defense against harmful fine-tuning and provide a gradient-level analysis showing that it saturates safety-degrading gradients while preserving benign task-relevant gradients. Based on this insight, we propose a Buffer-and-Reinforce fine-tuning framework that buffers harmful updates during user fine-tuning and reinforces safety after adaptation. Specifically, BufferLoRA induces temporary jailbreaking as a removable adapter to reduce harmful updates during user fine-tuning. After adaptation, ReinforceLoRA, trained to recover refusal behavior under the temporarily jailbroken state, is integrated with UserLoRA via QR decomposition-based merging to reinforce safety while preserving user-task performance. Extensive experiments show that our framework achieves superior safety and utility with no additional safety data during user fine-tuning and minimal computational cost. 

---
# SemanticZip: A Pilot Framework for Lossy Text Compression with LLMs as Semantic Decompressors 

**Authors**: Natalia Trukhina, Vadim Vashkelis  

**Link**: [PDF](https://arxiv.org/pdf/2605.24541)  

**Abstract**: Text compression for large language model (LLM) systems is usually framed as token deletion, retrieval, summarization, or exact reconstruction. We study a more aggressive but explicitly lossy setting: compress text into compact codes that an LLM can expand into task-relevant meaning. We call this setting SemanticZip. Unlike lossless compression, SemanticZip does not require byte-identical reconstruction; unlike ordinary summarization, it treats model-based decompression as part of the codec and evaluates whether task-relevant semantic commitments are recovered.
This paper is a pilot framework, not a benchmark claim. We formalize LLM-mediated decompression, define a protected/lossy packet architecture, and evaluate six representation regimes over five author-constructed diagnostic cases: structured prose, JSON, CCL-Core, CCL-Min, SemanticZip ASCII, and SemanticZip emoji. An independent decoder LLM reconstructs typed semantic atoms from each compressed representation, and we score Critical Atom Recall, Weighted Atom Recall, precision, and tokenizer gain. In this pilot, structured prose has the highest recoverability, with WAR = 0.956 and 19.1% o200k_base token gain. CCL-Min is the strongest balanced point, with 39.4% token gain and WAR = 0.874. SemanticZip ASCII provides the largest useful compression, with 46.5% token gain and WAR = 0.802, while emoji-heavy SemanticZip performs worse on both compression and recovery.
The main contribution is not the claim that these numbers establish a universal frontier. Rather, we introduce a reproducible experimental interface for studying lossy, LLM-decompressible text codes and a design principle: safety-critical and exact commitments should remain protected, while predictable low-risk context may be semantically zipped. 

---
# Hypothesis Generation and Inductive Inference in Children and Language Models 

**Authors**: Jeffrey Qin, Wasu Top Piriyakulki, Zhuangfei Gao, Mia Radovanovic, Jessica Sommerville, Kevin Ellis, Marta Kryven  

**Link**: [PDF](https://arxiv.org/pdf/2605.24528)  

**Abstract**: Real world decision-making requires constructing mental models under uncertainty over evidence, over the underlying causal rules, and over the state of the world itself. Which computational principles underpin human inference under such conditions, and do LLM-based agents exhibit similar behavior given matching constraints? We address these questions using an inductive inference Box Task in which participants, human children and LLM-based agents, infer a latent cause through sequential interaction with an uncertain environment. We formalize this task as program induction with Bayesian particle-based inference, admitting two complementary interpretations: (1) as a constraint satisfaction process over hypotheses, and (2) as a program synthesis problem in which hypotheses are executable programs evaluated against evidence. Using the constraint-based formulation, we show that children's behavior is best explained by a combination of subjective evidence reliability and online hypothesis generation, accounting for both their evidence-seeking patterns and their dissociation between task completion and rule generalization. Using the program synthesis formulation, we treat LLM-based agents as model organisms: controllable systems that allow systematic manipulation of task conditions. Across backends, LLM-based agents replicate children's responses to changes in evidence reliability and observability, including discounting unreliable evidence, seeking to resolve partial information, and dissociating between task completion and causal generalization. At the same time, LLM-based agents tend to over-observe and over-comply with instructions relative to children. These results suggest that while children and LLM-based agents adapt similarly to environmental structure, their information-seeking behavior exhibits distinct underlying costs and inductive biases. 

---
# What Are We Actually Decoding? Source Attribution for Non-Invasive Brain-to-Language Retrieval 

**Authors**: Xinyu Zhang, Sichao Liu, Runhao Lu, Alexandra Woolgar, Lihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24524)  

**Abstract**: In non-invasive neural language decoding, results can be inflated by sources that are not stimulus-evoked neural evidence: decoder priors, embedding-based metrics, and non-neural structural nuisances such as signal duration. The methodological challenge is therefore attribution: a reported gain is more informative when it can be traced to a specific source. We recast stimulus-locked MEG-to-audio retrieval as an auditing framework that separates apparent performance into three sources - structural shortcuts, window-level stimulus-locked evidence, and cross-window contextual aggregation - and provides a diagnostic for each. Signal-blind Gaussian noise reaches 66.3% Rank@1 (R@1) under variable-length decoding but collapses to near chance once fixed-duration windows and stimulus-identity splits are enforced, isolating structural leakage. Under these controls, fixed-window retrieval recovers measurable MEG-audio discriminability, while an oracle sentence-bucket diagnostic shows that 95.7% of Top-1 errors select the wrong sentence, localising the residual bottleneck to sentence-level competition. We audit this contextual source with Group Context Bias (GCB), an inference-time additive logit bias that pools sentence-consistent evidence across windows while leaving the base retrieval scores and candidate pool fixed. Used as a score-space intervention, GCB makes the contextual source measurable: R@1 shifts from 44% to 52% on Gwilliams and from 22% to 29% on MOUS under the same fixed setting. GCB is auditable under this design: its effect collapses under random-grouping perturbations and vanishes when local evidence is attenuated in MEG or is near chance in EEG, supporting its use as a controlled source-attribution intervention. These results suggest that brain-to-language performance should be source-attributed, not merely reported. 

---
# MindAlign: Bridging EEG, Vision, and Language for Zero-Shot Visual Decoding 

**Authors**: Zexuan Chen, Sichao Liu, Runhao Lu, Huichao Qi, Alexandra Woolgar, Xi Vincent Wang, Lihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24523)  

**Abstract**: Visual decoding from brain signals is a key challenge at the intersection of computer vision and neuroscience, requiring methods that bridge neural representations and computational models of vision. We introduce a tri-modal contrastive framework for EEG-based visual decoding that aligns EEG, visual, and textual representations within a unified latent space. Our approach follows a two-stage design. First, we pre-train an EEG encoder via masked reconstruction on unlabeled trials, learning spatio-temporal regularities that transfer robustly to downstream tasks. Second, we jointly align EEG, image, and LLM-generated textual descriptions through contrastive learning, where text supervision acts as a semantic regularizer that injects linguistic structure into the shared space without overwhelming the primary EEG-image signal. The encoder integrates subject-specific adaptation, graph-attention over channels, and temporal-spatial convolutional embeddings. On the Things-EEG2 200-way zero-shot benchmark, our framework achieves 54.1% Top-1 and 83.4% Top-5 accuracy, substantially exceeding the strongest prior baseline (32.4% / 64.0%), with paired Wilcoxon tests confirming significance (p < 0.01) over all in-subject baselines. We validate generalization on Things-MEG. Analysis reveals that compact embedding geometries (CN-CLIP) outperform much larger backbones, and that decoding aligns with established neurophysiology of visual processing. This work is a critical step towards robust, semantically-grounded visual decoding from non-invasive temporal neural signals. The source code is publicly available in this https URL. 

---
# ECHO: Terminal Agents Learn World Models for Free 

**Authors**: Vaishnavi Shrivastava, Piero Kauffmann, Ahmed Awadallah, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.24517)  

**Abstract**: CLI agents are the closest thing language models have to an embodied setting: the model emits commands, the terminal executes them, and the returned stream -- stdout, errors, files, logs, and traces -- records the consequences. We argue that this stream is a supervision signal, but standard agent RL discards it: GRPO-style training updates action tokens with sparse outcome-level rewards while ignoring environment responses already in the rollout. Failed rollouts provide little policy-gradient signal despite containing rich evidence about how the environment responds. We introduce ECHO (Environment Cross-entropy Hybrid Objective), a hybrid objective that combines the standard policy-gradient loss on action tokens with an auxiliary loss that trains the policy to predict environment observation tokens resulting from its own actions. ECHO reuses the same forward pass as GRPO, requires no additional rollouts, and turns terminal feedback into dense supervision for all rollouts. ECHO doubles GRPO pass@1 on TerminalBench-2.0: Qwen3-8B improves from 2.70% to 5.17%, and Qwen3-14B from 5.17% to 10.79%. ECHO also produces policies that better predict terminal dynamics, even on trajectories they did not generate: across held-out rollouts, it sharply reduces environment-token cross-entropy while GRPO alone barely changes it. From base Qwen3-8B, ECHO matches expert-SFT-then-GRPO performance on held-out terminal tasks without expert demonstrations, and recovers roughly half of the expert-SFT initialization benefit on TerminalBench-2.0. In some settings, the environment prediction loss alone enables verifier-free self-improvement, allowing policies to improve on unseen OOD tasks by learning only from environment interactions. Together, these results suggest that environment observations are not merely context for future actions, but a dense, on-policy supervision signal already present in every rollout. 

---
# AgentFugue: Agent Scaling for Long-Horizon Tasks through Collective Reasoning 

**Authors**: Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Tong Zhao, Xiaoxi Li, Zheng Liu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24486)  

**Abstract**: Recent progress on long-horizon agentic tasks has been driven largely by scaling up individual agents through stronger models, better tools, and more effective scaffolding. In contrast, much less is understood about scaling out: whether multiple peer agents, all targeting the same task, can become an additional source of capability without relying on explicit role specialization or workflow orchestration. We study this question and propose AgentFugue, a collective reasoning framework built around a shared reasoning hub. As peer agents explore the same task in parallel, the hub records concise notes on what each agent has established, attempted, or ruled out, and enables each agent to selectively access what other agents have discovered in a form useful for its current search. This design turns otherwise isolated trajectories into a connected ecology of reusable intermediate reasoning without requiring centralized planning. We instantiate the hub as a plug-in communication layer, trained with supervised fine-tuning and end-to-end reinforcement learning. Across the challenging long-horizon settings we study, AgentFugue improves over strong baselines. Our results suggest that collective reasoning can turn scaling out peer agent systems into a distinct source of capability gains, rather than merely a way of spending more compute. 

---
# Momentum Streams for Optimizer-Inspired Transformers 

**Authors**: Jingchu Gai, Nai-Chieh Huang, Jiayun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24425)  

**Abstract**: The residual update of a pre-norm Transformer layer admits an interpretation as one step of a first-order optimizer acting on a surrogate token energy, wherein the attention and MLP sublayers function as gradient oracles. Based on this observation, we build a family of optimizer-inspired Transformers (triple-momentum, Adam/AdamW, Muon, SOAP) and compare them under matched compute. In our main pretraining experiment, the triple-momentum TMMFormer achieves the lowest validation loss, outperforming the vanilla Transformer and prior architectural variants. A controlled ablation and supporting theory show that momentum, not preconditioning, is the main source of the gain. We further show that TMMFormer and other momentum-based designs reach flatter minima than the vanilla Transformer, which leads to less forgetting and better generalization. 

---
# SliceWorld: A Predictive and Controllable World-State Model for CT Report Generation 

**Authors**: Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.24371)  

**Abstract**: CT report generation (CTRG) requires models to summarize three-dimensional anatomical context and pathological findings from hundreds of axial slices. Existing methods typically learn a direct image-to-text mapping, providing limited mechanisms for modeling how CT evidence evolves across slices or how reports respond to controlled changes in latent lesion-related factors. We propose SliceWorld, a CT-specific world-state framework that treats an axial CT scan as an ordered sequence along the z-axis. SliceWorld encodes prefix CT evidence into factor-aware latent states containing anatomy, lesion, and uncertainty components, and projects these states into world tokens used for multi-step future-slice feature prediction, lesion-factor intervention, and LLM-based report generation. The model is first pretrained on CT slice sequences with predictive, factor-aware, and counterfactual objectives, and is then fine-tuned on paired CT-report data. Experiments on M3D-Cap and CT-RATE show that SliceWorld improves natural language generation metrics and clinically oriented automatic evaluation. Further analyses demonstrate multi-horizon future-slice prediction, measurable factor alignment, reduced-slice robustness, and selective lesion-sensitive report modulation. 

---
# Rubato: Transcribing Piano Music with Timestamps 

**Authors**: Nazif Can Tamer, Victoria Ebert, Guang Yang, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2605.24291)  

**Abstract**: We consider the conversion of musical recordings into human-readable sheet music annotated with timestamps. Such output lets a listener clearly visualize rubato (temporally expressive playing), a learner diagnose ensemble precision and timing choices against the written music, and a musicology scholar compare performance styles across recordings of the same work. We introduce (1) a prompt-conditioned encoder-decoder model, named Rubato, trained to output (2) a new textual representation for polyphonic music, named InterMo, which we designed for compatibility with sequence-to-sequence training. Our experiments demonstrate that Rubato produces timestamped piano sheet music from audio with higher notational accuracy than the best existing approaches, which are based on cascades. We find that even if the cascade is given ground-truth MIDI instead of audio, Rubato performs better, suggesting that the ceiling of existing approaches is primarily representational, not acoustic. Further, because Rubato is trained on several related tasks (with prompts), it competes with or outperforms the best single-task systems on related but simpler tasks like MIDI note grounding and beat/downbeat detection. A demo is available at this https URL . 

---
# Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought Reasoning 

**Authors**: Jinghan Jia, Joe Benton, Eric Easley  

**Link**: [PDF](https://arxiv.org/pdf/2605.24286)  

**Abstract**: Chain-of-thought (CoT) reasoning is useful for monitoring language models only when the reasoning trace faithfully reflects the computation that produces the final answer. However, models can rely on prompt-to-answer shortcuts that bypass the CoT, making the visible reasoning trace misleading even when it appears plausible. We study CoT faithfulness through a structural information-flow perspective: faithful reasoning should route answer-relevant information through the mediated path from prompt to CoT to answer, rather than through a direct prompt-to-answer shortcut. This perspective yields a task-agnostic framework based on three complementary properties, sufficiency, completeness, and necessity, which we instantiate with entropy-based, masked-KL, and gradient-based diagnostics. We show that these metrics recover externally judged faithfulness differences in hinted reasoning, and identify a low-entropy failure mode of KL-based diagnostics where gradient-based measures remain more stable. Building on this analysis, we introduce update-time interventions for verifier-based on-policy RL, including attention masking, backward-only gradient masking, CoT gradients, and adversarial perturbations of prompt representations. Across hinted arithmetic, reward-hackable code repair, and DAPO-Math models trained without hints but evaluated under wrong-hint injection, our interventions shift behavioral and structural indicators toward stronger CoT mediation. In particular, they make shortcut and reward-hacking behavior more transparent in the CoT and improve task-agnostic faithfulness metrics, while in some settings also reducing wrong-hint susceptibility. Our results suggest that controlling information flow during training is a practical route toward more faithful and monitorable CoT reasoning. Code is available at this https URL. 

---
# Agent-ToM: Learning to Monitor Autonomous LLM Agents via Theory-of-Mind Reasoning 

**Authors**: Nesreen K. Ahmed, Nima Nafisi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24216)  

**Abstract**: Monitoring autonomous large language model (LLM) agents for covert malicious behavior is challenging due to delayed, context-dependent, and long-horizon attack patterns. Agents may pursue hidden objectives while maintaining superficially benign behavior, making detection difficult even with full trajectory access. Prior monitoring approaches improve scaffolding or ensemble aggregation, but treat each trajectory independently and do not learn from prior monitoring experience. Moreover, standard reasoning methods explain observed behavior without explicitly reasoning about agent beliefs, intentions, and goal alignment required to distinguish benign task execution from covert deviation.
We propose \textbf{Agent-ToM}, a learning-to-monitor framework grounded in Theory-of-Mind (ToM) reasoning for security analysis of autonomous agents. Agent-ToM performs structured full-trajectory analysis by inferring beliefs, intent hypotheses with calibrated confidence, expected actions, and deviations from task-consistent behavioral baselines. At inference time, it employs a \textit{Reason-Verify-Refine} pipeline to construct and validate monitoring decisions. At training time, Agent-ToM distills critique signals into a persistent \textit{semantic guardrail memory}, enabling reusable belief- and intent-conditioned constraints across episodes. We evaluate Agent-ToM on adversarial agent monitoring benchmarks (SHADE-Arena and CUA-SHADE-Arena). Agent-ToM achieves strong precision-recall balance and outperforms state-of-the-art monitoring baselines, including ensemble methods, while using a coherent two-call reasoning pipeline. These results demonstrate that learning at the monitoring layer, combined with structured ToM reasoning and verification, provides an effective and deployable foundation for securing autonomous LLM agents. 

---
# TRACER: A Semantic-Aware Framework for Fine-Grained Contamination Detection in Code LLMs 

**Authors**: Yifeng Di, Xuliang Huang, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24079)  

**Abstract**: Data contamination is a known threat to the reliability of model evaluation. However, it remains underexplored in code large language models (LLMs), where contamination often goes beyond exact duplication. We present TRACER, a semantic-aware framework for fine-grained code contamination detection. TRACER models contamination using three levels of semantic overlap - Functionally Identical, Nearly Identical, and Shared Logic - and detects them through a coarse-to-fine pipeline. We also introduce the first benchmark for fine-grained code contamination detection, spanning three widely used benchmarks and three representative post-training datasets. TRACER achieves strong and consistent performance across multiple LLM backbones, with GPT-5 reaching an F1 score of 0.91 in fine-grained detection. In the binary setting, TRACER attains an F1 of 0.92, outperforming existing methods by 42%-217%. We further conduct ablation studies and error analysis to assess the contributions of individual components in TRACER. 

---
# Breaking the Chains of Probability: Neutrosophic Logic as a New Framework for Epistemic Uncertainty in Large Language Models 

**Authors**: Maikel Yelandi Leyva-Vázquez, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2605.24053)  

**Abstract**: Large Language Models (LLMs) are predominantly governed by probabilistic frameworks in which the sum of outcome probabilities is constrained to unity. This architectural limitation, often imposed by Softmax layers, leads to a collapse of uncertainty that makes it difficult to differentiate between epistemic uncertainty, paradox, and vagueness. We present an empirical investigation of the application of Neutrosophic Logic, a framework that treats Truth (T), Indeterminacy (I), and Falsity (F) as three independent dimensions, to model epistemic states in LLMs. We conducted experiments on a family of four OpenAI GPT models across five linguistic phenomena: logical paradoxes, epistemic ignorance, vagueness, ethical contradictions, and future contingencies, under three prompting strategies: neutrosophic, probabilistic, and entropy-derived. Our findings reveal that the neutrosophic approach, by allowing T+I+F > 1, a state we term hyper-truth, provides a richer representation of a model's internal state. In 35% of evaluations, hyper-truth emerged spontaneously, predominantly under ethical contradiction and logical paradox. We demonstrate that this approach preserves truth values in fuzzy contexts and offers a robust method for identifying and quantifying internal model conflict. We conclude that the integration of neutrosophic evaluation layers is a critical step toward more transparent, reliable, and ethically aware AI systems. 

---
# LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via Consistency-Regulated Reward Decomposition 

**Authors**: Yanyu Chen, Jiyue Jiang, Dianzhi Yu, Zheng Wu, Jiahong Liu, Jiaming Han, Xiao Guo, Jinhu Qi, Yu Li, Yifei Zhang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2605.24005)  

**Abstract**: The evolution of Large Language Model (LLM) reasoning is bottlenecked by the scarcity of high-quality process data. While self-alignment via endogenous rewards offers a solution, mining valid supervision faces three challenges: (1) Label Noise via Mimetic Bias, where rewards prioritize statistical likelihood over logical truth, creating a "correctness illusion" that masks compounding errors; (2) Coarse-Grained Supervision, where sparse global outcomes (e.g., in GRPO) fail to provide granular guidance, treating reasoning chains as monolithic; and (3) Distributional Collapse, where signals fail to generalize without amplifying pre-training biases. To address these, we introduce LC-ERD (Logic-Consistent Endogenous Reward Decomposition), a framework framing self-alignment as latent structure mining. We derive a Variational Logic Potential by aggregating consensus from the model's Latent Logic Expertise (LLE) to denoise the reasoning manifold, and introduce a Multi-Agent Value Decomposition protocol based on the IGM principle to quantify individual step utility. Experiments show LC-ERD delivers a robust self-evolution path, uncovering trade-offs between logic consistency and accuracy while identifying high-value reasoning patterns missed by standard rewards. Our code is available at this https URL. 

---
# Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, and system security 

**Authors**: Jinhu Qi, Muzhi Li, Jiahong Liu, Yuqin Shu, Dianzhi Yu, Shicheng Ma, Wenqian Cui, Yiyang Zhao, Yiyi Chen, Ruoxi Jiang, Irwin King, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23989)  

**Abstract**: Agentic AI systems -- Large Language Models (LLMs) augmented with planning, tool use, memory, and long-horizon interactions -- can execute complex tasks autonomously, but their multi-step trajectories introduce new failure modes that challenge trustworthiness. This survey provides a focused examination of trustworthy agentic AI through two core dimensions that are critical for high-risk deployments: Safety and Robustness, and Privacy and System Security. For each dimension, we clarify key concepts, identify where risks emerge along the agent workflow, and summarize stage-targeted mitigation strategies. Other trustworthiness aspects (value alignment, transparency, fairness, and accountability) are discussed as relevant context rather than parallel chapters. To support consistent comparison and deployment decisions, we consolidate evaluation into a unified metrics-and-benchmarks hub, emphasizing both outcome and process signals (e.g., constraint violations, trace completeness, and adversarial success rates) and offering scenario-to-metric guidance for release gating. We conclude by outlining open challenges such as self-evolving agents, runtime monitoring and verification, privacy-preserving personalization, and the trust-utility trade-off, and present a case study of real-world security failures in open-source agentic systems. Our goal is to serve as a practical reference for researchers and practitioners building trustworthy agentic systems in high-stakes environments. 

---
# Why We Need World Models for AGI: Where LLMs Fail and How World Models May Outperform 

**Authors**: Feisal Alaswad, Batoul Aljaddouh, Maher Alrahhal, Poovammal E, Talal Bonny  

**Link**: [PDF](https://arxiv.org/pdf/2605.23972)  

**Abstract**: Large language models achieve strong performance in language generation and knowledge-intensive tasks, yet remain limited in settings requiring causal reasoning, persistent state tracking, and long-horizon planning. We argue that these limitations may arise from an objective-level mismatch between sequence prediction and reasoning over latent environment dynamics. To formalize this distinction, we introduce Latent Dynamics Inference (LDI), a conceptual perspective that interprets language and multimodal observations as partial evidence of underlying transition dynamics. To empirically investigate this perspective, we introduce Flux, a sequential reasoning environment specified entirely through natural-language rules. As a proof-of-concept case study, the rules are first compiled into an explicit state-transition simulator, illustrating that structured latent transition dynamics can, in some cases, be operationally extracted from textual rule descriptions. This enables a controlled comparison between the LLMs operating purely over textual observations and reinforcement-learning agents trained directly within the extracted latent state space. Within this case study, agents operating with explicit access to the latent state space exhibit substantially more stable behavior in long-horizon gameplay, achieving an aggregate win rate of approximately 79% versus 11% for LLMs. Qualitative analysis further reveals failure modes consistent with unstable persistent state tracking, including invalid actions, state-tracking errors, and short-horizon reasoning failures. The complete implementation of the Flux environment available at this https URL Within the evaluated setting, these results suggest that strong sequence prediction alone may struggle to support robust long-horizon dynamic reasoning without mechanisms for persistent state tracking and transition modeling 

---
# Machine Psychometrics: A Mathematical Psychology of Artificial Intelligence 

**Authors**: Alex Bogdan, Adrian de Valois-Franklin  

**Link**: [PDF](https://arxiv.org/pdf/2605.23952)  

**Abstract**: Artificial agents now generate behavior rich enough to invite trust, surprise, and concern, yet our evaluation tools still privilege capability scores over psychological structure. This paper argues that the philosophical impasse between two symmetrical errors (Artificial Mind Blindness, which dismisses psychological organization in non-biological systems, and Artificial Mind Projection, which infers human-like inner life from fluent behavior alone) can be circumvented not by resolving the consciousness question, but by introducing a disciplined measurement layer beneath it. Drawing on Michael Levin's continuum view of cognition as goal-directed competency across substrates, and on the methodological repertoire of mathematical psychology (Item Response Theory, Signal Detection Theory, Bayesian cognitive modeling, calibration analysis, cognitive-bias batteries), the paper develops Machine Psychometrics as a measurement science of latent behavioral, metacognitive, communicative, and self-modeling dispositions in artificial agents. Its operational core is the Machine Mindprint: a multidimensional, domain-bounded, versioned profile spanning calibration, source integrity, suggestibility resistance, context stability, expressive alignment, tool integrity, drift monitoring, and distributional grounding. A complementary Trust Protocol turns Mindprints into deployment decisions through probe batteries, perturbation testing, reliability and validity analysis, and longitudinal monitoring across high-stakes domains. The philosophical contribution is a third stance, Artificial Mind Discipline, that neither anthropomorphizes nor dismisses, neither presupposes consciousness nor forecloses it. The aim is not to humanize artificial agents, but to understand them precisely because they are not human, through measurement before judgment. 

---
# Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning 

**Authors**: Sebastien Kawada  

**Link**: [PDF](https://arxiv.org/pdf/2605.23940)  

**Abstract**: How do multi-turn reasoning systems fail? The expected answer is logical contradiction, in which the system's maintained state becomes unsatisfiable. We show that the dominant mode is instead satisfiable drift, where the internal state stays consistent while the returned answer silently violates prior commitments. We build DRIFT-Bench (Decomposing Reasoning Into Failure Types), a solver-instrumented benchmark of 816 test problems across three constraint domains, and evaluate four methods on it across four open-weight models (8B-120B parameters). MUS-Repair, which feeds minimal unsatisfiable subsets back to the generator, is strongest in every setting (+1.8 to +15.0 pp over the best non-MUS baseline). But the central finding is what repair leaves behind. After structured feedback, models rarely contradict themselves. They forget. Residual errors are 98-100% satisfiable drift across all settings, while contradiction drops to near zero. Reliable multi-turn systems must separately validate that the returned answer respects the maintained state. Code is available at this https URL. 

---
# When Correct Beliefs Collapse: Epistemic Resilience of LLMs under Clinical Pressure 

**Authors**: Boyu Xiao, Xiuqi Tian, Xuwen Song, Haochun Wang, Guanchun Song, Sendong Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.23932)  

**Abstract**: Despite strong medical benchmark accuracy, LLMs can exhibit severe multi-turn sycophancy in clinical dialogue, abandoning initial correct diagnosis under escalating pressure. We propose \textbf{\textsc{Med-Stress}}, a targeted stress test framework that evaluates belief stability under escalating pressure. Across nine frontier large language models (LLMs), we find a clear dissociation between medical knowledge and robustness: high initial diagnostic capability does not imply high belief stability, yielding large knowledge-robustness gaps for several LLMs. To mitigate this failure mode, we propose a lightweight inference-time defense, \textbf{\texttt{RBED}} (\textbf{R}ole-\textbf{B}ased \textbf{E}pistemic \textbf{D}efense), and \textbf{\texttt{R-FT}} (\textbf{R}esilience-oriented \textbf{F}ine-\textbf{T}uning), a training-time approach that internalizes evidence-based resistance to pressure. Experiments show that \textbf{\texttt{R-FT}} nearly eliminates belief change and substantially improves robustness. 

---
# Context: Proactive Goal-Directed Intelligence via Composable Sandboxed Programs, Declarative Wiring, and Structured Interaction 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2605.23928)  

**Abstract**: We present Context, the intelligence layer of the Magarshak Architecture, which replaces reactive query-response chatbots with proactive goal-directed agents that advance shared tasks without waiting for user prompts. The architecture rests on three mutually reinforcing mechanisms. Write-time context assembly precomputes enriched typed attributes via Groker agents, assembling interaction context as a deterministic pure function of graph state; context blocks are byte-identical across turns between semantic changes, enabling near-100% KV-cache reuse. Composable sandboxed wisdom programs form a governed library of LM-generated imperative programs declaratively wired to goal types via typed stream relations, composed via phase ordering, and executed at interaction time without further LM calls. Proactive goal stream state machines drive conversations toward terminal states by inspecting graph state and emitting structured interaction content (option arrays, governance affordances, clarification prompts) without awaiting user input. We prove six formal results: the Context Stability Theorem, bounding per-turn LM cost as a function of semantic change rate; a Program Composition Correctness Theorem; a Declarative Wiring Soundness Theorem; the Proactive Dominance Theorem, proving proactive agents weakly dominate reactive agents on expected turns-to-terminal-state; Coordination Overhead Elimination and Quality Preservation, establishing Pareto improvements in multi-participant goal chats; and a Cross-Platform Vote Consistency Theorem. Implemented in the open-source Qbix / Safebox / Safebots stack. 

---
# Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning 

**Authors**: Moiz Imran, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2605.23925)  

**Abstract**: Intelligent tutoring systems increasingly provide automated feedback on student work, but robust feedback requires assessing reasoning, not only final answers. We study a failure mode we call the correct answer trap (CAT): models under-detect misconceptions when students reach a correct answer via flawed reasoning. Analysing real student responses from the Eedi mathematics platform, we show that 71% of these failures concentrate in just two question types, both sharing a common structure where flawed reasoning happens to produce the correct numerical answer. Comparing a fine-tuned T5 with a frontier large language model, we find that improved capabilities reduce but do not eliminate the problem (84% vs 57% detection accuracy). Even the best-performing model generates roughly four false alarms for every genuine detection, making stand-alone screening impractical at realistic class sizes. Our findings demonstrate that high overall accuracy can mask critical failures in reasoning assessment, and that careful analysis of student reasoning still benefits from human judgment. 

---
# Can LoRA Fusion Support Cross-Domain Tasks in Cloud-Edge Collaboration? 

**Authors**: Yatong Wang, Fali Wang, Naibin Gu, Zheng Lin, Zhengxiao Liu, Dingyu Yao, Zhiwei Zhang, Jianxin Shi, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23913)  

**Abstract**: Cloud-hosted large language models (LLMs) commonly rely on LoRA for domain adaptation, yet domain data are distributed across multiple edge devices and cannot be uploaded due to privacy constraints. This raises a fundamental question: how can knowledge from multiple private edges be integrated into a cloud LLM for cross-domain problem solving? A natural solution is to train LoRA adapters locally and fuse them in the cloud; however, existing pipelines rely on unrealistic assumptions that edge devices can host cloud-scale LLMs and are evaluated mainly on single-domain tasks. To address these limitations, we propose a prune-train-recover framework that enables local LoRA training on pruned models and privacy-preserving cloud integration. We further introduce MMLU-CD, a cross-domain benchmark that composes multiple domain samples into a single instance, enabling explicit evaluation of cross-domain problem solving. This allows us to ask a concrete question: Can existing LoRA fusion methods support cross-domain tasks in cloud-edge collaboration? Our empirical answer is negative. Existing LoRA fusion methods perform poorly on MMLU-CD, often underperforming the base LLM, revealing their inability to support cross-domain problem solving. We attribute this failure to parameter conflicts among LoRA adapters and propose a simple conflict-resolution module, LoRA-CR, which mitigates conflicting updates and improves LoRA fusion performance by up to 3.8%. These results identify conflict mitigation as a critical yet largely overlooked factor in cloud-edge LoRA fusion, warranting further investigation in future research. 

---
# In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models 

**Authors**: Sam Earle, Kay Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi  

**Link**: [PDF](https://arxiv.org/pdf/2605.23908)  

**Abstract**: We are in the midst of large-scale industrial and academic efforts to automate the processes of scientific, technological and creative production through AI-driven assistants. Historically, a fundamental property of these processes in their human form has been their open-endedness: their capacity for generating a seemingly endless supply of novel and meaningful new forms. Do artificial agents have any capacity for such fruitful unguided discovery? To answer this question, we turn to Picbreeder, the canonical exemplar of human-driven open-ended search, in which users collaboratively generated a diverse library of images through interactive evolution of small neural networks. We replicate Picbreeder, replacing human users with frontier Vision Language Models (VLMs). We observe clear qualitative differences between the output of our system and the historical human baseline, and attempt to characterize them using metrics of phylogenetic complexity and visual and semantic salience and novelty. In an effort to identify some of the causal factors contributing these differences, we study the addition of exploratory noise to the agents' selection process, of behavioral diversity between agents, and of narrative momentum in the form of memory of past actions. We make our code available at this https URL. 

---
# Adaptive Preference Optimization with Uncertainty-aware Utility Anchor 

**Authors**: Xiaobo Wang, Zixia Jia, Jiaqi Li, Qi Liu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10515)  

**Abstract**: Offline preference optimization methods are efficient for large language models (LLMs) alignment. Direct Preference optimization (DPO)-like learning, one of the most popular approaches, stands out for its efficiency in reward modeling. However, these methods typically follow the convention to use Bradley-Terry (BT) reward modeling that faces several critical assumptions, including the requirement for pairwise training data, model distribution shifting, human rationality assumption, etc. To address these limitations, we propose a general framework for offline preference optimization methods, Adaptive Preference Optimization with Utility Anchor (UAPO), which introduces an anchoring function to estimate the uncertainties brought from preference data annotation. Our method enables training even in scenarios where the data is unpaired, significantly enhancing data utilization efficiency. Moreover, the anchor design makes UAPO more robust in the training process. Experimental results demonstrate that UAPO achieves competitive outcomes without the strict dependency on data pairing, paving the way for more flexible and effective preference optimization methods. 

---
